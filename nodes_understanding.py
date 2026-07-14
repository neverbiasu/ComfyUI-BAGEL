"""Native BAGEL image-understanding node. VIT-only (no VAE), returns text."""

from __future__ import annotations

import comfy.model_management as model_management
import torch

from .modeling.bagel.runtime import (
    generate_text,
    init_gen_context,
    update_context_text,
    update_vit_image,
)
from .nodes_common import build_handle, comfy_image_to_pil


class BAGELImageUnderstanding:
    """Answer a question about an image. Uses only the VIT encoder; no VAE.

    The node never loads or accepts a VAE or a LATENT; official ComfyUI VAE
    nodes remain external to understanding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "max_length": ("INT", {"default": 500, "min": 1, "max": 2000, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "do_sample": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "understand"
    OUTPUT_NODE = True
    CATEGORY = "BAGEL/Understanding"

    def understand(self, model, image, prompt, max_length, temperature, do_sample):
        model_management.load_models_gpu([model])
        handle = build_handle(model)
        m = handle["model"]
        device = next(m.parameters()).device

        pil = comfy_image_to_pil(image)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Legacy image-then-text ordering (InterleaveInferencer.__call__):
            # ViT tokens (source image) are added before the prompt text.
            gen = init_gen_context(m)
            gen = update_vit_image(handle, pil, gen)
            gen = update_context_text(handle, prompt, gen)
            text = generate_text(
                handle, gen, max_length=max_length,
                temperature=temperature, do_sample=do_sample,
            )
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "BAGELImageUnderstanding": BAGELImageUnderstanding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAGELImageUnderstanding": "BAGEL Image Understanding",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
