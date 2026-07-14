"""Deterministic BAGEL tokenizer fingerprint and asset-hash validation.

This is the SINGLE source of truth for the tokenizer fingerprint shared by the
tokenizer ``MANIFEST.json``, the converter, the inspector, and the native
loader. It is pure stdlib (no ``torch`` / ``transformers``) so every consumer --
including the standalone validators -- computes the identical value.

The fingerprint is the SHA-256 over the canonical, ordered concatenation of the
packaged tokenizer asset files. Ordering is fixed (alphabetical by filename) and
each file's bytes are prefixed with its name so renames cannot collide. This is
stable across machines and does not depend on a loaded tokenizer object.
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict

# Assets that define the BAGEL tokenizer vocabulary/behavior. Missing optional
# files are skipped (the fingerprint then covers whatever is present), but the
# converter/loader still validate the recorded per-file hashes.
ASSET_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def file_hashes(assets_dir: str) -> Dict[str, str]:
    """Return ``{filename: sha256}`` for present tokenizer asset files."""
    out: Dict[str, str] = {}
    for name in ASSET_FILES:
        p = os.path.join(assets_dir, name)
        if os.path.exists(p):
            out[name] = hashlib.sha256(_read_bytes(p)).hexdigest()
    return out


def tokenizer_fingerprint(assets_dir: str) -> str:
    """Canonical SHA-256 fingerprint of the packaged tokenizer assets.

    Deterministic and order-independent of dict iteration: files are hashed in
    the fixed ``ASSET_FILES`` order, each prefixed by its name.
    """
    h = hashlib.sha256()
    for name in ASSET_FILES:
        p = os.path.join(assets_dir, name)
        if os.path.exists(p):
            h.update(name.encode("utf-8"))
            h.update(_read_bytes(p))
    return h.hexdigest()
