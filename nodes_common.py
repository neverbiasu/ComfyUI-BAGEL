"""Shared node-layer helpers for the native BAGEL nodes.

This module is imported by every ``nodes_*.py`` node module. It owns only
node-level validation, seed, and image/tensor helpers. It does NOT define any
``NODE_CLASS_MAPPINGS`` and it does NOT import any other node module, so the
root ``__init__.py`` remains the sole mapping aggregator and node modules stay
independent of each other.

It depends only on ComfyUI, the BAGEL runtime, and the patcher (the model
layer), never on ``nodes.py`` or another node module.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

def build_handle(patcher) -> dict:
    """Extract the runtime handle from a ``BAGEL_MODEL`` patcher.

    The patcher carries the attached ``bagel_state`` (tokenizer, special-token
    IDs, vision transform, metadata, checkpoint identity). The model object the
    runtime calls lives at ``patcher.model``.
    """
    state = getattr(patcher, "bagel_state", None)
    if state is None:
        raise RuntimeError(
            "BAGEL_MODEL patcher is missing its attached runtime state. "
            "Load the model with the native BAGEL Model Loader."
        )
    return {
        "model": patcher.model,
        "tokenizer": state["tokenizer"],
        "new_token_ids": state["new_token_ids"],
        "vit_transform": state["vit_transform"],
    }


def apply_seed(seed: int) -> None:
    """Seed torch (and torch.cuda) deterministically for reproducible runs."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(int(seed))
        except Exception:
            pass


def comfy_image_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI ``IMAGE`` tensor ``[B,H,W,C]`` (0..1) to a PIL image."""
    arr = image[0].detach().cpu().float().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)
