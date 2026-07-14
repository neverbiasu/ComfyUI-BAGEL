"""Packaged BAGEL tokenizer construction and fingerprinting.

Loads the repository-local tokenizer bundle under ``modeling/qwen2/tokenizer``
(reusing the repo's own ``Qwen2Tokenizer``) and computes a deterministic
vocabulary fingerprint plus the required special-token IDs. Loading the
tokenizer module directly avoids the modeling package's ``flash_attn`` import
chain, so this is safe under stubbed/local environments.

The bundled assets are populated from an immutable BAGEL source revision (see
``MANIFEST.json``); when they are absent, the helpers return ``None`` rather
than raising, so the inspector can still report the rest of the descriptor.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

from .tokenizer_fingerprint import file_hashes, tokenizer_fingerprint

ASSETS_DIR = Path(__file__).resolve().parents[1] / "qwen2" / "tokenizer"

REQUIRED_SPECIAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
)


def _load_qwen2_tokenizer():
    """Import the repo ``Qwen2Tokenizer`` without pulling ``modeling`` package init."""
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from modeling.qwen2.tokenization_qwen2 import Qwen2Tokenizer

    return Qwen2Tokenizer


def load_packaged_tokenizer(assets_dir: Optional[Path] = None):
    """Load the packaged BAGEL tokenizer; raise on any genuine failure.

    A missing assets directory raises a clear ``FileNotFoundError``. A real
    load/construct failure (e.g. a corrupt asset or an incompatible revision)
    raises a descriptive ``RuntimeError`` -- it is never swallowed into an
    unhelpful ``None`` so the native loader can surface an actionable error.
    """
    assets_dir = Path(assets_dir) if assets_dir is not None else ASSETS_DIR
    if not assets_dir.exists():
        raise FileNotFoundError(
            f"BAGEL packaged tokenizer directory not found: {assets_dir}. "
            "Install the custom node with its modeling/qwen2/tokenizer assets."
        )
    if not any(assets_dir.glob("vocab.json")) and not any(assets_dir.glob("tokenizer.json")):
        raise FileNotFoundError(
            f"BAGEL packaged tokenizer assets are missing under {assets_dir} "
            "(expected vocab.json / tokenizer.json)."
        )
    try:
        Qwen2Tokenizer = _load_qwen2_tokenizer()
        return Qwen2Tokenizer.from_pretrained(str(assets_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load packaged BAGEL tokenizer from {assets_dir}: {exc}. "
            "The tokenizer assets may be corrupt or from an incompatible revision."
        ) from exc


def compute_vocab_fingerprint(assets_dir: Optional[Path] = None) -> str:
    """Canonical tokenizer fingerprint (see ``tokenizer_fingerprint``)."""
    assets_dir = Path(assets_dir) if assets_dir is not None else ASSETS_DIR
    return tokenizer_fingerprint(str(assets_dir))


def required_special_token_ids(tokenizer) -> Dict[str, Optional[int]]:
    out = {}
    for tok in REQUIRED_SPECIAL_TOKENS:
        try:
            out[tok] = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            out[tok] = None
    return out


def verify_special_token_ids(tokenizer) -> Dict[str, Optional[int]]:
    """Return the four required special-token IDs; ``None`` means absent."""
    ids = required_special_token_ids(tokenizer)
    return ids
