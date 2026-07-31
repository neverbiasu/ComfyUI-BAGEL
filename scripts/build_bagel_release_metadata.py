"""Build trustworthy embedded metadata for an existing BAGEL safetensors.

This tool reads only the checkpoint header. It does not load, cast, quantize,
or rewrite tensor payloads. A release spec supplies provenance and validated
capabilities; the model configs and packaged tokenizer manifest supply the
runtime-critical self-description.
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_MANIFEST = REPO_ROOT / "modeling/qwen2/tokenizer/MANIFEST.json"
CRITICAL_PREFIXES = (
    "language_model",
    "vit_model",
    "connector",
    "vae2llm",
    "llm2vae",
    "latent_pos_embed",
    "vit_pos_embed",
    "time_embedder",
)
DTYPE_LABELS = {
    "BF16": "torch.bfloat16",
    "F16": "torch.float16",
    "F32": "torch.float32",
    "F64": "torch.float64",
}


def _json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _safetensors_header(path: Path) -> dict:
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError(f"not a safetensors file: {path}")
        length = struct.unpack("<Q", prefix)[0]
        raw = file.read(length)
    if len(raw) != length:
        raise ValueError(f"truncated safetensors header: {path}")
    try:
        header = json.loads(raw)
    except Exception as exc:
        raise ValueError(f"invalid safetensors header in {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be an object")
    return header


def _tensor_summary(header: dict) -> dict:
    entries = {
        name: value for name, value in header.items() if name != "__metadata__"
    }
    if not entries:
        raise ValueError("checkpoint contains no tensors")
    roots = {name.split(".", 1)[0] for name in entries}
    histogram = Counter()
    param_count = 0
    examples = {}
    for name, entry in entries.items():
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        if not isinstance(dtype, str) or not isinstance(shape, list):
            raise ValueError(f"invalid tensor entry: {name}")
        histogram[DTYPE_LABELS.get(dtype, dtype)] += 1
        param_count += math.prod(int(dimension) for dimension in shape)
        root = name.split(".", 1)[0]
        if root in CRITICAL_PREFIXES and root not in examples:
            examples[root] = [shape, DTYPE_LABELS.get(dtype, dtype)]
    return {
        "num_tensors": len(entries),
        "dtype_histogram": dict(sorted(histogram.items())),
        "param_count": param_count,
        "critical_prefixes_present": [
            prefix for prefix in CRITICAL_PREFIXES if prefix in roots
        ],
        "module_shape_examples": examples,
    }


def build_metadata(
    checkpoint: Path,
    spec_path: Path,
    llm_config_path: Path,
    vit_config_path: Path,
    tokenizer_manifest_path: Path,
) -> dict:
    spec = _json_object(spec_path)
    required = {
        "variant",
        "source_repository",
        "source_revision",
        "source_hashes",
        "dtype",
        "capabilities",
    }
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"release spec is missing required fields: {missing}")
    if spec.get("quantization", "none") != "none":
        raise ValueError(
            "this metadata-only release workflow accepts non-quantized checkpoints only"
        )
    tokenizer_manifest = _json_object(tokenizer_manifest_path)
    tensor_summary = _tensor_summary(_safetensors_header(checkpoint))
    expected_dtype = {
        "bf16": "torch.bfloat16",
        "fp16": "torch.float16",
        "fp32": "torch.float32",
    }.get(spec["dtype"])
    if expected_dtype is None:
        raise ValueError(f"unsupported declared dtype: {spec['dtype']!r}")
    actual_dtypes = set(tensor_summary["dtype_histogram"])
    if actual_dtypes != {expected_dtype}:
        raise ValueError(
            f"declared dtype {spec['dtype']!r} does not match checkpoint "
            f"tensor dtypes {sorted(actual_dtypes)}"
        )
    model_options = dict(spec.get("model_options", {}))
    required_roots = {
        "language_model",
        "vae2llm",
        "llm2vae",
        "latent_pos_embed",
        "time_embedder",
    }
    if model_options.get("visual_und", True):
        required_roots.update({"vit_model", "connector", "vit_pos_embed"})
    present_roots = set(tensor_summary["critical_prefixes_present"])
    missing_roots = sorted(required_roots - present_roots)
    if missing_roots:
        raise ValueError(
            f"checkpoint is missing roots required by model_options: {missing_roots}"
        )
    metadata = {
        "format": "comfyui_bagel",
        "format_version": 1,
        "architecture": "Bagel",
        "variant": spec["variant"],
        "source_repository": spec["source_repository"],
        "source_revision": spec["source_revision"],
        "source_hashes": spec["source_hashes"],
        "dtype": spec["dtype"],
        "quantization": spec.get("quantization", "none"),
        "tokenizer_fingerprint": tokenizer_manifest[
            "vocab_fingerprint_sha256"
        ],
        "special_token_ids": tokenizer_manifest["special_token_ids"],
        "latent_format": "flux",
        "capabilities": list(spec["capabilities"]),
        "tensor_summary": tensor_summary,
        "model_configs": {
            "llm_config.json": _json_object(llm_config_path),
            "vit_config.json": _json_object(vit_config_path),
        },
        "model_options": model_options,
        "additional_special_tokens": list(
            spec.get("additional_special_tokens", [])
        ),
        "converter_version": "1.0.0",
    }
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--llm-config", required=True, type=Path)
    parser.add_argument("--vit-config", required=True, type=Path)
    parser.add_argument("--tokenizer-manifest", type=Path, default=TOKENIZER_MANIFEST)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        metadata = build_metadata(
            args.checkpoint,
            args.spec,
            args.llm_config,
            args.vit_config,
            args.tokenizer_manifest,
        )
        args.output.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "variant": metadata["variant"],
                "tensor_count": metadata["tensor_summary"]["num_tensors"],
                "capabilities": metadata["capabilities"],
                "model_options": metadata["model_options"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
