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
from .nodes_common import (
    VLM_THINK_SYSTEM_PROMPT,
    build_handle,
    comfy_image_to_pil,
    require_bagel_capability,
    require_single_image_batch,
)


class BAGELImageUnderstanding:
    """Answer a question about an image. Uses only the VIT encoder; no VAE.

    The node never loads or accepts a VAE or a LATENT; official ComfyUI VAE
    nodes remain external to understanding.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "Native BAGEL model from BAGEL Model Loader."}),
                "image": ("IMAGE", {"tooltip": "Image to analyse. It is resized with BAGEL's official 1024/512/16 preprocessing."}),
                "prompt": ("STRING", {"multiline": True, "default": "Can someone explain what's funny about this meme??", "tooltip": "Question or instruction about the image."}),
                "show_thinking": ("BOOLEAN", {"default": False, "tooltip": "Ask BAGEL to include its reasoning in the returned text."}),
                "do_sample": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling for text generation."}),
                "text_temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Text-generation randomness; 0 is deterministic and 1 is more creative."}),
                "max_new_tokens": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Maximum generated text length, including optional reasoning."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "understand"
    OUTPUT_NODE = True
    CATEGORY = "BAGEL/Understanding"

    def understand(self, model, image, prompt, show_thinking, do_sample,
                   text_temperature, max_new_tokens):
        require_bagel_capability(model, "image_understanding")
        require_single_image_batch(image)
        model_management.load_models_gpu([model])
        handle = build_handle(model)
        m = handle["model"]
        device = next(m.parameters()).device

        # InterleaveInferencer applies the shared 1024/512/16 resize before
        # passing the image through the NaViT-specific transform.
        pil = handle["image_transform"].resize_transform(comfy_image_to_pil(image))

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Legacy image-then-text ordering (InterleaveInferencer.__call__):
            # ViT tokens (source image) are added before the prompt text.
            gen = init_gen_context(m)
            if show_thinking:
                gen = update_context_text(handle, VLM_THINK_SYSTEM_PROMPT, gen)
            gen = update_vit_image(handle, pil, gen)
            gen = update_context_text(handle, prompt, gen)
            text = generate_text(
                handle, gen, max_length=max_new_tokens,
                temperature=text_temperature, do_sample=do_sample,
            )
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "BAGELImageUnderstanding": BAGELImageUnderstanding,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAGELImageUnderstanding": "BAGEL Image Understanding",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
