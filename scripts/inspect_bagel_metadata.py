"""Standalone BAGEL metadata / tokenizer / variant inspector.

Reads ONLY the safetensors header (no tensor values) plus the packaged tokenizer
assets, and prints a JSON descriptor: variant, capability tier, capabilities,
metadata summary, tokenizer vocabulary fingerprint, and required special-token
IDs. Exits non-zero on a metadata/asset mismatch.

Usage:
    python scripts/inspect_bagel_metadata.py models/bagel/bagel.safetensors
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _install_stubs() -> None:
    """Stub flash_attn/dfloat11 only when the real GPU modules are absent.

    On AutoDL the real modules exist and are used; locally / on Modal the CUDA
    chain is unavailable, so we write on-disk import-compatible stubs to a temp
    directory and prepend it to ``sys.path``. We never shadow an installed real
    module.
    """
    need = []
    for name in ("flash_attn", "dfloat11"):
        try:
            importlib.import_module(name)
        except Exception:
            need.append(name)
    if not need:
        return
    stub_dir = Path(tempfile.mkdtemp(prefix="bagel_stubs_"))
    if "flash_attn" in need:
        (stub_dir / "flash_attn.py").write_text(
            "def flash_attn_varlen_func(*a, **k):\n"
            "    raise RuntimeError('flash_attn stub')\n"
            "__version__ = '0.0.0-stub'\n"
        )
    if "dfloat11" in need:
        (stub_dir / "dfloat11.py").write_text(
            "class DFloat11Model:\n"
            "    @classmethod\n"
            "    def from_pretrained(cls, *a, **k):\n"
            "        raise RuntimeError('dfloat11 stub')\n"
            "__version__ = '0.0.0-stub'\n"
        )
    sys.path.insert(0, str(stub_dir))


def _import_package_modules():
    sys.path.insert(0, str(REPO))
    from modeling.bagel import converted_format, model_types, variants  # noqa: E402
    from modeling.qwen2 import tokenizer_fingerprint as tf  # noqa: E402
    from modeling.qwen2.bagel_tokenizer import ASSETS_DIR  # noqa: E402

    return converted_format, model_types, variants, ASSETS_DIR, tf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="converted BAGEL safetensors file")
    args = parser.parse_args()

    _install_stubs()
    (
        converted_format,
        model_types,
        variants,
        ASSETS_DIR,
        tf,
    ) = _import_package_modules()

    import safetensors.torch

    if not Path(args.model).exists():
        print(json.dumps({"error": f"model not found: {args.model}"}, indent=2))
        return 1

    with safetensors.torch.safe_open(args.model, framework="pt", device="cpu") as f:
        header_meta = f.metadata() or {}
        tensor_keys = list(f.keys())

    if "comfyui_bagel" not in header_meta:
        print(json.dumps({
            "error": "not a converted comfyui_bagel file (missing metadata header)",
            "tier": model_types.CapabilityTier.UNSUPPORTED.value,
        }, indent=2))
        return 1

    metadata = converted_format.ConvertedBagelMetadata.from_json(header_meta["comfyui_bagel"])
    descriptor = variants.detect_variant(metadata, tensor_keys)

    # Pure-stdlib tokenizer validation: canonical fingerprint over the packaged
    # assets, per-file hash check against MANIFEST, and special-token cross-check.
    tokenizer_fingerprint = tf.tokenizer_fingerprint(str(ASSETS_DIR))
    special_token_ids = dict(metadata.special_token_ids or {})
    manifest_path = Path(ASSETS_DIR) / "MANIFEST.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    mismatches = []
    if metadata.tokenizer_fingerprint and metadata.tokenizer_fingerprint != tokenizer_fingerprint:
        mismatches.append(
            f"tokenizer_fingerprint mismatch (metadata={metadata.tokenizer_fingerprint!r}, "
            f"assets={tokenizer_fingerprint!r})"
        )
    if manifest.get("vocab_fingerprint_sha256") and \
            manifest["vocab_fingerprint_sha256"] != tokenizer_fingerprint:
        mismatches.append("tokenizer_fingerprint disagrees with MANIFEST")
    for name, expected in (manifest.get("files") or {}).items():
        actual = tf.file_hashes(str(ASSETS_DIR)).get(name)
        if actual is None:
            mismatches.append(f"tokenizer asset missing: {name}")
        elif actual != expected.get("sha256"):
            mismatches.append(f"tokenizer asset hash mismatch: {name}")
    recorded_special = manifest.get("special_token_ids") or {}
    for tok in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"):
        if recorded_special.get(tok) != special_token_ids.get(tok):
            mismatches.append(f"special-token id mismatch for {tok}")

    report = {
        "model": args.model,
        "descriptor": descriptor.to_dict(),
        "metadata": metadata.to_dict(),
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "manifest_fingerprint": manifest.get("vocab_fingerprint_sha256"),
        "special_token_ids": special_token_ids,
        "mismatches": mismatches,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
