"""Native BAGEL image-edit node.

Accepts BOTH:
* the original ComfyUI ``IMAGE`` for the SigLIP/NaViT (ViT) path;
* the official ``VAEEncode`` ``LATENT`` for the BAGEL VAE-token conditioning.

It returns a standard ComfyUI ``LATENT`` decoded downstream by the official
``VAEDecode`` node. The coupled BAGEL VAE is never loaded.
"""

from __future__ import annotations

import copy

import comfy.model_management as model_management
import torch

from .modeling.bagel.runtime import (
    generate_latent,
    init_gen_context,
    update_context_text,
    update_vae_latent_from_latent,
    update_vit_image,
    validate_bagel_image_shape,
)
from .nodes_common import apply_seed, build_handle, comfy_image_to_pil


class BAGELImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL",),
                "image": ("IMAGE", {"tooltip": "Source image; fed through the ViT encoder"}),
                "vae_latent": ("LATENT", {"tooltip": "Output of the official FLUX VAEEncode on the source image"}),
                "prompt": ("STRING", {"multiline": True, "default": "Make it snowy"}),
                "cfg_text_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "cfg_img_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "num_timesteps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),
                "timestep_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "reasoning")
    FUNCTION = "edit"
    CATEGORY = "BAGEL/Editing"

    def edit(self, model, image, vae_latent, prompt, cfg_text_scale, cfg_img_scale,
             num_timesteps, timestep_shift, seed):
        model_management.load_models_gpu([model])
        handle = build_handle(model)
        m = handle["model"]
        device = next(m.parameters()).device

        vae_tensor = vae_latent["samples"].to(device=device, dtype=torch.bfloat16)
        # Output pixel size follows the source VAE latent.
        _, _, h_lat, w_lat = vae_tensor.shape
        H, W = h_lat * 8, w_lat * 8
        validate_bagel_image_shape(m, (H, W))

        pil = comfy_image_to_pil(image)

        apply_seed(seed)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Legacy image-then-text ordering (InterleaveInferencer.edit):
            # 1) VAE latent tokens (source image conditioning)
            # 2) ViT tokens (source image features)
            # 3) cfg_text snapshots the image-only context BEFORE the prompt
            # 4) prompt added to gen (full image+text conditioning)
            # 5) cfg_img is text-only (prompt built from an empty context)
            gen = init_gen_context(m)
            gen = update_vae_latent_from_latent(handle, vae_tensor, gen)
            gen = update_vit_image(handle, pil, gen)
            cfg_text = copy.deepcopy(gen)  # image-only baseline (text dropped)
            gen = update_context_text(handle, prompt, gen)  # full (image+text)
            cfg_img = update_context_text(handle, prompt, init_gen_context(m))  # text-only

            latent = generate_latent(
                handle, gen, cfg_text, cfg_img, (H, W),
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
            )
        return (latent, "")


NODE_CLASS_MAPPINGS = {
    "BAGELImageEdit": BAGELImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAGELImageEdit": "BAGEL Image Edit",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
