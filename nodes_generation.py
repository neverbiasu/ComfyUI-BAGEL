"""Native BAGEL text-to-image node. Returns a ComfyUI LATENT (VAE-decoupled)."""

from __future__ import annotations

import comfy.model_management as model_management
import torch

from .modeling.bagel.runtime import (
    generate_latent,
    init_gen_context,
    update_context_text,
)
from .nodes_common import apply_seed, build_handle


class BAGELTextToImage:
    """Generate an image (standard FLUX ``LATENT``) from a text prompt.

    The node never loads or accepts a VAE. The returned latent is decoded
    downstream by the official ``VAEDecode`` node using the FLUX AE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "A photo of a cat"}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 16}),
                "cfg_text_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "cfg_img_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_timesteps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "timestep_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "reasoning")
    FUNCTION = "generate"
    CATEGORY = "BAGEL/Generation"

    def generate(self, model, prompt, width, height, cfg_text_scale, cfg_img_scale,
                 num_timesteps, timestep_shift, seed):
        model_management.load_models_gpu([model])
        handle = build_handle(model)
        m = handle["model"]
        device = next(m.parameters()).device

        apply_seed(seed)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            gen = init_gen_context(m)
            gen = update_context_text(handle, prompt, gen)
            # Text-only unconditional/text branches for classifier-free guidance.
            cfg_text = init_gen_context(m)                 # no text
            cfg_img = update_context_text(handle, prompt, init_gen_context(m))  # text only
            # cfg_img drops the (absent) image; cfg_text is unconditional.

            latent = generate_latent(
                handle, gen, cfg_text, cfg_img, (height, width),
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
            )
        return (latent, "")


NODE_CLASS_MAPPINGS = {
    "BAGELTextToImage": BAGELTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAGELTextToImage": "BAGEL Text to Image",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
