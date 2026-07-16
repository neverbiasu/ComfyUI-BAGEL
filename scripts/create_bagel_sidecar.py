#!/usr/bin/env python3
"""Create a ComfyUI-BAGEL sidecar for an existing single-file checkpoint.

Use this when a large BAGEL ``.safetensors`` file already exists but does not
embed the ``comfyui_bagel`` metadata header. The native loader accepts either:

* embedded ``comfyui_bagel`` safetensors metadata; or
* a hash-bound sidecar named ``<checkpoint>.comfyui-bagel.json``.

This script creates the sidecar without rewriting the large checkpoint.

Example:

    python scripts/create_bagel_sidecar.py \
        --checkpoint ComfyUI/models/bagel/bagel-7b-mot.safetensors \
        --config-dir /path/to/original/BAGEL-7B-MoT \
        --variant BAGEL-7B-MoT \
        --source-repository https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict

import safetensors


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = REPO_ROOT / "modeling" / "qwen2" / "tokenizer"
TOKENIZER_MANIFEST = TOKENIZER_DIR / "MANIFEST.json"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


converted_format = _load_module(
    "comfyui_bagel_converted_format",
    REPO_ROOT / "modeling" / "bagel" / "converted_format.py",
)
tokenizer_fingerprint_mod = _load_module(
    "comfyui_bagel_tokenizer_fingerprint",
    REPO_ROOT / "modeling" / "qwen2" / "tokenizer_fingerprint.py",
)


ConvertedBagelMetadata = converted_format.ConvertedBagelMetadata
CRITICAL_PREFIXES = converted_format.CRITICAL_PREFIXES
CONVERTER_VERSION = converted_format.CONVERTER_VERSION


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_model_configs(config_dir: Path) -> Dict[str, object]:
    model_configs: Dict[str, object] = {}
    missing = []
    invalid = []
    for name in ("llm_config.json", "vit_config.json"):
        path = config_dir / name
        if not path.exists():
            missing.append(name)
            continue
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append(f"{name}: {exc}")
            continue
        if not isinstance(parsed, dict):
            invalid.append(f"{name}: not a JSON object")
            continue
        model_configs[name] = parsed
    if missing or invalid:
        raise ValueError(
            f"Cannot build sidecar from config dir {config_dir}. "
            f"Missing: {missing}. Invalid: {invalid}."
        )
    return model_configs


def tokenizer_info() -> tuple[str, Dict[str, int]]:
    tok_fp = ""
    special_ids: Dict[str, int] = {}
    if TOKENIZER_DIR.exists():
        tok_fp = tokenizer_fingerprint_mod.tokenizer_fingerprint(str(TOKENIZER_DIR))
    if TOKENIZER_MANIFEST.exists():
        manifest = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))
        special_ids = {
            k: v for k, v in (manifest.get("special_token_ids") or {}).items()
            if v is not None
        }
    return tok_fp, special_ids


def inspect_checkpoint(path: Path) -> Dict[str, object]:
    dtype_hist: Dict[str, int] = {}
    module_shape_examples = {}
    present_roots = set()
    num_tensors = 0
    param_count = 0

    with safetensors.safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            num_tensors += 1
            root = key.split(".", 1)[0]
            present_roots.add(root)
            tensor = f.get_slice(key)
            shape = list(map(int, tensor.get_shape()))
            dtype = str(tensor.get_dtype())
            dtype_hist[dtype] = dtype_hist.get(dtype, 0) + 1
            n = 1
            for dim in shape:
                n *= dim
            param_count += n
            if root in CRITICAL_PREFIXES and root not in module_shape_examples:
                module_shape_examples[root] = [shape, dtype]

    missing = [p for p in CRITICAL_PREFIXES if p not in present_roots]
    if missing:
        raise ValueError(
            f"Checkpoint is missing critical BAGEL module roots: {missing}. "
            "This does not look like a complete converted BAGEL checkpoint."
        )

    return {
        "num_tensors": num_tensors,
        "dtype_histogram": dtype_hist,
        "param_count": param_count,
        "critical_prefixes_present": [p for p in CRITICAL_PREFIXES if p in present_roots],
        "module_shape_examples": module_shape_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="existing BAGEL .safetensors")
    parser.add_argument(
        "--config-dir",
        required=True,
        help="directory containing llm_config.json and vit_config.json",
    )
    parser.add_argument("--variant", default="BAGEL-7B-MoT")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--source-repository", default="")
    parser.add_argument("--source-revision", default="")
    parser.add_argument("--output", default="", help="defaults to <checkpoint>.comfyui-bagel.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"ERROR: checkpoint not found: {checkpoint}", file=sys.stderr)
        return 2
    if checkpoint.suffix != ".safetensors":
        print(f"ERROR: expected a .safetensors checkpoint: {checkpoint}", file=sys.stderr)
        return 2

    out = Path(args.output) if args.output else Path(str(checkpoint) + ".comfyui-bagel.json")
    if out.exists() and not args.force:
        print(f"ERROR: sidecar exists (use --force): {out}", file=sys.stderr)
        return 2

    try:
        tensor_summary = inspect_checkpoint(checkpoint)
        tok_fp, special_ids = tokenizer_info()
        metadata = ConvertedBagelMetadata(
            variant=args.variant,
            source_repository=args.source_repository,
            source_revision=args.source_revision,
            source_hashes={checkpoint.name: sha256_file(checkpoint)},
            dtype=args.dtype,
            quantization=args.quantization,
            tokenizer_fingerprint=tok_fp,
            special_token_ids=special_ids,
            capabilities=["text_to_image", "image_editing", "image_understanding"],
            tensor_summary=tensor_summary,
            model_configs=read_model_configs(Path(args.config_dir)),
            converter_version=CONVERTER_VERSION,
        )
        sidecar = {
            "format": "comfyui_bagel_sidecar",
            "format_version": 1,
            "checkpoint_size": checkpoint.stat().st_size,
            "checkpoint_sha256": sha256_file(checkpoint),
            "metadata": metadata.to_dict(),
        }
        out.write_text(json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote BAGEL sidecar: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
