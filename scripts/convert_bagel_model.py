"""Offline raw checkpoint -> converted single-file BAGEL safetensors converter.

Reads a BAGEL checkpoint (a single ``.safetensors`` file or a Hugging Face
style directory with ``model.safetensors.index.json`` shards), normalizes and
casts the coupled weights one tensor at a time, embeds
:class:`ConvertedBagelMetadata`, writes a temporary sibling file, validates it,
then atomically renames into place.

Memory is bounded: only one tensor's bytes are held at once. Raw, normalized,
and casted full state-dict dicts are never materialized, so the ~29GB BAGEL
source converts within a bounded working set whose peak is the largest
single tensor plus the transient raw and casted copies held while that one
tensor is being added.

A raw directory without a ``model.safetensors.index.json`` is rejected with an
actionable message rather than silently combined (which could mix in unrelated
files such as the VAE's ``ae.safetensors``).

Usage:
    python scripts/convert_bagel_model.py \
        --source models/bagel-7b-moT/ema.safetensors \
        --output models/bagel/bagel-7b-moT-bf16.safetensors \
        --variant BAGEL-7B-MoT --dtype bf16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict

import safetensors

# NOTE: importing this module DOES pull in the modeling package (and thus torch
# and safetensors.torch) via the converted-format / variant / tokenizer imports
# below; this converter only runs on a CUDA host where torch is installed.
# By contrast, the streaming writer (streaming_safetensors) and its standalone
# validator (validate_streaming_writer.py) are torch-free by design and can be
# run anywhere.

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modeling.bagel.converted_format import (  # noqa: E402
    CONVERTER_VERSION,
    CRITICAL_PREFIXES,
    ConvertedBagelMetadata,
    ConversionManifest,
    DTYPE_TO_TORCH,
    normalize_key,
)
from modeling.bagel.variants import detect_variant  # noqa: E402
from modeling.qwen2.tokenizer_fingerprint import (  # noqa: E402
    tokenizer_fingerprint,
)
from streaming_safetensors import (  # noqa: E402
    StreamingSafetensorsWriter,
    cleanup_temp,
)

TOKENIZER_DIR = REPO_ROOT / "modeling" / "qwen2" / "tokenizer"
TOKENIZER_MANIFEST = TOKENIZER_DIR / "MANIFEST.json"

# torch dtype string (str(tensor.dtype)) -> safetensors dtype code. The
# converter casts the whole checkpoint to a single target dtype (one of
# bfloat16/float16/float32 from DTYPE_TO_TORCH), so the casted tensor dtype is
# always exactly the chosen target. This map is therefore restricted to the
# three dtypes the converter can actually emit -- if a future source variant
# requires int/bool/float8 weights, extend this map and SAFETENSORS_DTYPES in
# lockstep with the valid safetensors codes.
_ST_DTYPE = {
    "torch.bfloat16": "BF16",
    "torch.float16": "F16",
    "torch.float32": "F32",
}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_shards(source: str) -> Dict[str, str]:
    """Map tensor key -> shard file path for a single file or HF directory.

    A directory must contain ``model.safetensors.index.json`` with a non-empty
    ``weight_map``; every referenced shard must exist. Ambiguous directories
    (no index, or an empty/invalid index) are rejected rather than silently
    combined, which would otherwise risk mixing in unrelated files such as the
    VAE's ``ae.safetensors``. Shard order is sorted by filename so conversion is
    deterministic.
    """
    src = Path(source)
    if src.is_file():
        return {src.name: str(src)}
    if src.is_dir():
        index = src / "model.safetensors.index.json"
        if not index.exists():
            raise FileNotFoundError(
                f"No model.safetensors.index.json found in directory '{source}'. "
                f"A raw BAGEL checkpoint directory must contain a HuggingFace index "
                f"(model.safetensors.index.json) so the converter knows the exact "
                f"weight shards to combine. Point --source at the single consolidated "
                f"checkpoint file (e.g. ema.safetensors) or at the directory that "
                f"contains model.safetensors.index.json."
            )
        try:
            with open(index) as f:
                weight_map = json.load(f).get("weight_map", {})
        except Exception as exc:
            raise ValueError(
                f"Failed to parse index in '{source}': {exc}"
            ) from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"model.safetensors.index.json in '{source}' has an empty or invalid "
                f"weight_map; cannot determine the model shards to combine."
            )
        missing = set()
        ordered = []
        seen = set()
        for shard in weight_map.values():
            if shard in seen:
                continue
            seen.add(shard)
            shard_path = os.path.join(str(src), shard)
            if not os.path.exists(shard_path):
                missing.add(shard)
                continue
            ordered.append(shard_path)
        if missing:
            raise FileNotFoundError(
                f"Referenced shard(s) missing from '{source}': {sorted(missing)}. "
                f"The raw checkpoint directory is incomplete; the converter will not "
                f"silently combine unrelated files."
            )
        return {Path(p).name: p for p in sorted(ordered)}
    raise FileNotFoundError(f"source not found: {source}")


def _tokenizer_info() -> tuple:
    """Return (tokenizer_fingerprint, special_token_ids) without loading tensors."""
    tok_fp = ""
    special_ids: Dict[str, int] = {}
    if TOKENIZER_DIR.exists():
        tok_fp = tokenizer_fingerprint(str(TOKENIZER_DIR))
    if TOKENIZER_MANIFEST.exists():
        tm = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
        special_ids = {
            k: v for k, v in (tm.get("special_token_ids") or {}).items() if v is not None
        }
    return tok_fp, special_ids


def _stream_convert(source: str, args: argparse.Namespace) -> Dict:
    """Stream the source checkpoint into a spooled safetensors data file.

    Returns the small accumulators the caller needs to build metadata/manifest.
    Never materializes the full state dict; only one tensor is resident at a time.
    """
    import torch  # lazy: only needed when actually converting

    target_torch = getattr(torch, DTYPE_TO_TORCH[args.dtype])
    key_mapping: Dict[str, str] = {}
    seen_norm = set()
    tensor_details: Dict[str, object] = {}
    dtype_hist: Dict[str, int] = {}
    param_count = 0
    present_roots = set()

    data_tmp = Path(args.output).with_name(Path(args.output).name + ".data.tmp")
    writer = StreamingSafetensorsWriter(str(data_tmp))
    try:
        for shard_path in _source_shards(source).values():
            with safetensors.safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    norm = normalize_key(key)
                    if norm in seen_norm:
                        raise KeyError(
                            f"two source keys normalize to the same destination "
                            f"{norm!r} (collision, including duplicate keys across "
                            f"shards); a unique converted key set cannot be produced."
                        )
                    seen_norm.add(norm)
                    key_mapping[key] = norm

                    casted = tensor.to(target_torch)
                    dtype_hist[str(casted.dtype)] = dtype_hist.get(str(casted.dtype), 0) + 1
                    param_count += int(casted.numel())
                    present_roots.add(norm.split(".", 1)[0])
                    tensor_details[norm] = [list(map(int, casted.shape)), str(casted.dtype)]
                    raw = casted.contiguous().view(torch.uint8).numpy().tobytes()
                    writer.add(
                        norm,
                        _ST_DTYPE[str(casted.dtype)],
                        list(casted.shape),
                        raw,
                    )
                    del tensor, casted
    except Exception:
        writer.close()
        cleanup_temp(str(data_tmp))
        raise

    writer.close()

    # Critical-module presence must hold before we emit a (useless) partial file.
    missing = [p for p in CRITICAL_PREFIXES if p not in present_roots]
    if missing:
        cleanup_temp(str(data_tmp))
        raise KeyError(f"converted state dict missing critical module weights: {missing}")

    return {
        "data_tmp": data_tmp,
        "writer": writer,
        "key_mapping": key_mapping,
        "tensor_details": tensor_details,
        "dtype_hist": dtype_hist,
        "param_count": param_count,
        "present_roots": present_roots,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="raw checkpoint file or directory")
    parser.add_argument("--output", required=True, help="converted safetensors output path")
    parser.add_argument("--variant", default="", help="model variant label (e.g. BAGEL-7B-MoT)")
    parser.add_argument("--dtype", default="bf16", choices=list(DTYPE_TO_TORCH))
    parser.add_argument("--source-repository", default="", help="upstream repository URL")
    parser.add_argument("--source-revision", default="", help="upstream commit/revision")
    parser.add_argument("--quantization", default="none", help="quantization label: none or df11")
    parser.add_argument("--force", action="store_true", help="overwrite existing output")
    args = parser.parse_args()

    out_path = Path(args.output)
    if out_path.exists() and not args.force:
        print(f"ERROR: output exists (use --force to overwrite): {out_path}", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    final_tmp = out_path.with_name(out_path.name + ".tmp")
    if os.path.exists(final_tmp):
        os.remove(final_tmp)

    source_hashes = {
        name: _sha256_file(path) for name, path in _source_shards(args.source).items()
    }
    tok_fp, special_ids = _tokenizer_info()

    info = _stream_convert(args.source, args)
    data_tmp = str(info["data_tmp"])
    writer = info["writer"]

    try:
        # Structural variant detection -> capabilities.
        pre_meta = ConvertedBagelMetadata(
            variant=args.variant, dtype=args.dtype, quantization=args.quantization
        )
        descriptor = detect_variant(pre_meta, list(info["key_mapping"].values()))
        capabilities = descriptor.capabilities.to_list()

        # Tensor summary: histogram (of the casted/output dtype), parameter
        # count, critical-module presence.
        tensor_summary = {
            "num_tensors": len(info["tensor_details"]),
            "dtype_histogram": info["dtype_hist"],
            "param_count": info["param_count"],
            "critical_prefixes_present": [
                p for p in CRITICAL_PREFIXES if p in info["present_roots"]
            ],
        }

        # Self-describing per-tensor shape/dtype, kept compact in the embedded
        # metadata by recording one example per critical module root.
        module_shape_examples = {}
        for p in CRITICAL_PREFIXES:
            for k, v in info["tensor_details"].items():
                if k.split(".", 1)[0] == p:
                    module_shape_examples[p] = v
                    break

        # Embed model configs so the loader never downloads them at runtime.
        # For the documented single-file usage (e.g. ema.safetensors) the configs
        # live in the file's parent directory; for a directory source they live
        # in the directory itself. Both configs are mandatory: a converted file
        # without them is useless to the loader. We fail actionably BEFORE
        # writing any output (and the outer except cleans the large data temp),
        # so the user learns about the missing configs instead of producing a
        # file that only fails later inside the loader.
        src_dir = Path(args.source)
        if not src_dir.is_dir():
            src_dir = src_dir.parent
        model_configs = {}
        missing_cfg = []
        bad_cfg = []
        for cfg_name in ("llm_config.json", "vit_config.json"):
            cfg_path = src_dir / cfg_name
            if not cfg_path.exists():
                missing_cfg.append(cfg_name)
                continue
            try:
                parsed = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception as exc:
                bad_cfg.append(f"{cfg_name} (parse error: {exc})")
                continue
            if not isinstance(parsed, dict):
                bad_cfg.append(f"{cfg_name} (not a JSON object)")
                continue
            model_configs[cfg_name] = parsed
        if missing_cfg or bad_cfg:
            raise ValueError(
                "Cannot convert: required model config file(s) are missing or "
                f"invalid in '{src_dir}'. The loader needs both configs embedded "
                "so it never downloads. "
                f"Missing: {sorted(missing_cfg)}. Invalid: {bad_cfg}. "
                "Provide a source directory, or a single-file source whose parent "
                "directory contains both llm_config.json and vit_config.json."
            )

        metadata = ConvertedBagelMetadata(
            variant=args.variant,
            source_repository=args.source_repository,
            source_revision=args.source_revision,
            source_hashes=source_hashes,
            dtype=args.dtype,
            quantization=args.quantization,
            tokenizer_fingerprint=tok_fp,
            special_token_ids=special_ids,
            capabilities=capabilities,
            tensor_summary={
                **tensor_summary,
                "module_shape_examples": module_shape_examples,
            },
            model_configs=model_configs,
            converter_version=CONVERTER_VERSION,
        )

        # Write header + streamed data into the temporary sibling file.
        writer.finalize(str(final_tmp), metadata.to_json())

        # Validate the written temp file before renaming.
        with safetensors.safe_open(str(final_tmp), framework="pt", device="cpu") as f:
            written = set(f.keys())
            header_meta = f.metadata() or {}
        if set(info["key_mapping"].values()) != written:
            print("ERROR: written keys differ from normalized keys", file=sys.stderr)
            raise RuntimeError("written keys differ from normalized keys")
        raw_metadata = header_meta.get("comfyui_bagel")
        if not raw_metadata:
            print("ERROR: metadata header missing from converted file", file=sys.stderr)
            raise RuntimeError("metadata header missing from converted file")
        try:
            written_metadata = json.loads(raw_metadata)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "converted file contains invalid comfyui_bagel metadata JSON"
            ) from exc
        required_metadata = {
            "format",
            "format_version",
            "architecture",
            "dtype",
            "capabilities",
            "model_configs",
            "converter_version",
        }
        missing_metadata = sorted(required_metadata - set(written_metadata))
        if missing_metadata or written_metadata.get("format") != "comfyui_bagel":
            raise RuntimeError(
                "converted file metadata is incomplete or has the wrong format: "
                f"missing={missing_metadata}, format={written_metadata.get('format')!r}"
            )
        if not isinstance(written_metadata.get("model_configs"), dict) or not {
            "llm_config.json",
            "vit_config.json",
        }.issubset(written_metadata["model_configs"]):
            raise RuntimeError(
                "converted file metadata must embed llm_config.json and vit_config.json"
            )

        # Atomic replace -- the existing --force destination is untouched until
        # this point, so a failed conversion never clobbers a good output.
        os.replace(str(final_tmp), str(out_path))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.path.exists(final_tmp):
            os.remove(final_tmp)
        cleanup_temp(data_tmp)
        return 1

    cleanup_temp(data_tmp)

    manifest = ConversionManifest(
        source=args.source,
        source_hashes=source_hashes,
        converted=str(out_path),
        converted_hash=_sha256_file(str(out_path)),
        variant=args.variant,
        dtype=args.dtype,
        key_mapping=info["key_mapping"],
        tensor_details=info["tensor_details"],
        converter_version=CONVERTER_VERSION,
    )
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    manifest_path.write_text(manifest.to_json())

    print(json.dumps({
        "output": str(out_path),
        "tensor_count": len(info["key_mapping"]),
        "dtype": args.dtype,
        "variant": args.variant,
        "quantization": args.quantization,
        "tokenizer_fingerprint": tok_fp,
        "special_token_ids": special_ids,
        "capabilities": capabilities,
        "tensor_summary": tensor_summary,
        "has_model_configs": bool(model_configs),
        "converted_hash": manifest.converted_hash,
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
