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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_path = args.source.resolve()
    output_path = args.output.resolve()
    if source_path == output_path:
        print("ERROR: --output must differ from --source", file=sys.stderr)
        return 2
    if output_path.exists() and not args.force:
        print(f"ERROR: output exists (use --force): {output_path}", file=sys.stderr)
        return 2

    metadata_json = _load_metadata(args.metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    try:
        with source_path.open("rb") as source:
            header, data_start, payload_len = _read_header(source)
            source_size = os.fstat(source.fileno()).st_size
            if source_size - data_start != payload_len:
                raise ValueError(
                    "tensor payload size does not match header offsets: "
                    f"header={payload_len}, file={source_size - data_start}"
                )
            existing_metadata = header.get("__metadata__") or {}
            if not isinstance(existing_metadata, dict):
                raise ValueError("safetensors __metadata__ must be a JSON object")
            header["__metadata__"] = {
                **existing_metadata,
                "comfyui_bagel": metadata_json,
            }
            encoded_header = _encode_header(header)
            source.seek(data_start)
            with temporary.open("wb") as destination:
                destination.write(struct.pack("<Q", len(encoded_header)))
                destination.write(encoded_header)
                shutil.copyfileobj(source, destination, length=_COPY_BUFFER_BYTES)
        os.replace(temporary, output_path)
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "source": str(source_path),
                "output": str(output_path),
                "output_size": output_path.stat().st_size,
                "output_sha256": _sha256(output_path),
                "metadata_key": "comfyui_bagel",
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
