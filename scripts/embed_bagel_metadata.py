"""Embed ``comfyui_bagel`` metadata into an existing safetensors file.

This is a bounded-memory release repair tool for converted checkpoints whose
tensor payload is already correct but whose safetensors header predates the
native BAGEL metadata schema. It rewrites only the header and streams the
existing tensor data unchanged; tensors are never loaded or cast.

Usage:
    python scripts/embed_bagel_metadata.py \
        --source bagel-7b-mot.safetensors \
        --metadata bagel-7b-mot.metadata.json \
        --output bagel-7b-mot.with-metadata.safetensors

For storage-constrained release servers, ``--in-place`` shifts the tensor
payload inside the server-side working copy and verifies its SHA-256 before
and after the shift. The remote Hub file is not touched by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

_ALIGN = 8
_COPY_BUFFER_BYTES = 8 << 20


def _read_header(source) -> tuple[dict, int, int]:
    prefix = source.read(8)
    if len(prefix) != 8:
        raise ValueError("source is too small to be a safetensors file")
    header_len = struct.unpack("<Q", prefix)[0]
    header_bytes = source.read(header_len)
    if len(header_bytes) != header_len:
        raise ValueError("truncated safetensors header")
    try:
        header = json.loads(header_bytes)
    except Exception as exc:
        raise ValueError(f"invalid safetensors JSON header: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    tensor_entries = [value for key, value in header.items() if key != "__metadata__"]
    if not tensor_entries:
        raise ValueError("safetensors file contains no tensors")
    try:
        payload_len = max(int(entry["data_offsets"][1]) for entry in tensor_entries)
    except Exception as exc:
        raise ValueError("invalid tensor data_offsets in safetensors header") from exc
    return header, 8 + header_len, payload_len


def _load_metadata(path: Path) -> str:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid metadata JSON {path}: {exc}") from exc
    # Accept either a direct ConvertedBagelMetadata object or the release
    # sidecar schema previously published for bagel-7b-mot.
    if raw.get("format") == "comfyui_bagel_sidecar":
        raw = raw.get("metadata")
    if not isinstance(raw, dict) or raw.get("format") != "comfyui_bagel":
        raise ValueError(
            "metadata must be a comfyui_bagel object or a comfyui_bagel_sidecar"
        )
    if int(raw.get("format_version", 0)) != 1:
        raise ValueError("only comfyui_bagel format_version 1 is supported")
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def _encode_header(header: dict) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (_ALIGN - len(encoded) % _ALIGN) % _ALIGN
    return encoded + b" " * padding


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(path: Path, data_start: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        file.seek(data_start)
        for chunk in iter(lambda: file.read(_COPY_BUFFER_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge_metadata(header: dict, metadata_json: str) -> bytes:
    existing_metadata = header.get("__metadata__") or {}
    if not isinstance(existing_metadata, dict):
        raise ValueError("safetensors __metadata__ must be a JSON object")
    header["__metadata__"] = {
        **existing_metadata,
        "comfyui_bagel": metadata_json,
    }
    return _encode_header(header)


def _rewrite_in_place(
    path: Path,
    old_data_start: int,
    payload_len: int,
    encoded_header: bytes,
) -> str:
    """Move payload safely within a disposable server copy, then write header."""
    new_data_start = 8 + len(encoded_header)
    delta = new_data_start - old_data_start
    payload_sha256 = _sha256_payload(path, old_data_start)

    with path.open("r+b") as file:
        if delta > 0:
            # Expanding the header requires copying backwards so unread source
            # bytes are never overwritten by their shifted destination.
            file.truncate(new_data_start + payload_len)
            remaining = payload_len
            while remaining:
                chunk_size = min(_COPY_BUFFER_BYTES, remaining)
                source_offset = old_data_start + remaining - chunk_size
                destination_offset = new_data_start + remaining - chunk_size
                file.seek(source_offset)
                chunk = file.read(chunk_size)
                if len(chunk) != chunk_size:
                    raise IOError("short read while shifting safetensors payload")
                file.seek(destination_offset)
                file.write(chunk)
                remaining -= chunk_size
        elif delta < 0:
            # Shrinking can copy forwards for the symmetric overlap reason.
            moved = 0
            while moved < payload_len:
                chunk_size = min(_COPY_BUFFER_BYTES, payload_len - moved)
                file.seek(old_data_start + moved)
                chunk = file.read(chunk_size)
                if len(chunk) != chunk_size:
                    raise IOError("short read while shifting safetensors payload")
                file.seek(new_data_start + moved)
                file.write(chunk)
                moved += chunk_size
            file.truncate(new_data_start + payload_len)

        file.seek(0)
        file.write(struct.pack("<Q", len(encoded_header)))
        file.write(encoded_header)
        file.flush()
        os.fsync(file.fileno())

    rewritten_sha256 = _sha256_payload(path, new_data_start)
    if rewritten_sha256 != payload_sha256:
        raise IOError(
            "payload verification failed after in-place rewrite; do not upload "
            f"this working copy (before={payload_sha256}, after={rewritten_sha256})"
        )
    return payload_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="rewrite the disposable server working copy without a second 29GB file",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_path = args.source.resolve()
    output_path = source_path if args.in_place else args.output.resolve()
    if not args.in_place and source_path == output_path:
        print("ERROR: use --in-place when --output equals --source", file=sys.stderr)
        return 2
    if not args.in_place and output_path.exists() and not args.force:
        print(f"ERROR: output exists (use --force): {output_path}", file=sys.stderr)
        return 2

    metadata_json = _load_metadata(args.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None if args.in_place else output_path.with_name(output_path.name + ".tmp")
    payload_sha256 = None
    try:
        with source_path.open("rb") as source:
            header, data_start, payload_len = _read_header(source)
            source_size = os.fstat(source.fileno()).st_size
            if source_size - data_start != payload_len:
                raise ValueError(
                    "tensor payload size does not match header offsets: "
                    f"header={payload_len}, file={source_size - data_start}"
                )
            encoded_header = _merge_metadata(header, metadata_json)
            if not args.in_place:
                source.seek(data_start)
                with temporary.open("wb") as destination_file:
                    destination_file.write(struct.pack("<Q", len(encoded_header)))
                    destination_file.write(encoded_header)
                    shutil.copyfileobj(
                        source, destination_file, length=_COPY_BUFFER_BYTES
                    )
        if args.in_place:
            payload_sha256 = _rewrite_in_place(
                source_path, data_start, payload_len, encoded_header
            )
        else:
            os.replace(temporary, output_path)
    except Exception as exc:
        if temporary is not None and temporary.exists():
            temporary.unlink()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "source": str(source_path),
                "output": str(output_path),
                "output_size": output_path.stat().st_size,
                # In-place mode already reads the 29GB payload twice for its
                # before/after invariant. Avoid a third full-file pass on a
                # free runtime; the Hub upload computes its own file identity.
                "output_sha256": None if args.in_place else _sha256(output_path),
                "payload_sha256": payload_sha256,
                "in_place": args.in_place,
                "metadata_key": "comfyui_bagel",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
