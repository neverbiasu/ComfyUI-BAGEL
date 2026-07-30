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

import random

import numpy as np
import torch
from PIL import Image


GEN_THINK_SYSTEM_PROMPT = """You should first think about the planning process in the mind and then generate the image.
The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here"""

VLM_THINK_SYSTEM_PROMPT = """You should first think about the reasoning process in the mind and then provide the user with the answer.
The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here"""

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
        "image_transform": state["image_transform"],
        "vit_transform": state["vit_transform"],
        "variant_descriptor": state["variant_descriptor"],
    }


def require_bagel_capability(patcher, capability: str) -> None:
    """Reject a node task before GPU loading when a variant does not support it."""
    state = getattr(patcher, "bagel_state", None) or {}
    descriptor = state.get("variant_descriptor", {})
    capabilities = descriptor.get("capabilities", [])
    if capability not in capabilities:
        name = descriptor.get("name") or descriptor.get("variant") or "unknown BAGEL model"
        tier = descriptor.get("tier", "unsupported")
        raise NotImplementedError(
            f"{name} does not support BAGEL {capability} in this runtime "
            f"(capability tier: {tier})."
        )


def apply_seed(seed: int) -> None:
    """Match the official app: positive seeds are deterministic; zero is unset."""
    if int(seed) <= 0:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(int(seed))
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def comfy_image_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI ``IMAGE`` tensor ``[B,H,W,C]`` (0..1) to a PIL image."""
    arr = image[0].detach().cpu().float().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def require_single_image_batch(image: torch.Tensor, *, name: str = "image") -> None:
    """Native BAGEL currently accepts one image/latent per execution only."""
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError(f"BAGEL {name} must be a rank-4 ComfyUI tensor")
    if image.shape[0] != 1:
        raise ValueError(
            f"BAGEL {name} batch size must be 1; received {image.shape[0]}. "
            "Use one BAGEL node execution per image."
        )
