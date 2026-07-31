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
from .nodes_common import (
    GEN_THINK_SYSTEM_PROMPT,
    apply_seed,
    build_handle,
    comfy_image_to_pil,
    require_bagel_capability,
    require_single_image_batch,
)


class BAGELImageEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "Native BAGEL model from BAGEL Model Loader."}),
                "image": ("IMAGE", {"tooltip": "Source image for BAGEL's ViT encoder. It must be the same preprocessed image sent to VAEEncode."}),
                "vae_latent": ("LATENT", {"tooltip": "Output of the official FLUX VAEEncode on the same preprocessed source image."}),
                "prompt": ("STRING", {"multiline": True, "default": "She boards a modern subway, quietly reading a folded newspaper, wearing the same clothes.", "tooltip": "Instruction describing the requested edit."}),
                "cfg_text_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.1, "tooltip": "Controls how strongly BAGEL follows the edit prompt."}),
                "cfg_img_scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.1, "tooltip": "Controls preservation of input-image details."}),
                "cfg_interval": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Start of the CFG interval. The end is fixed at 1.0."}),
                "timestep_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5, "tooltip": "Shifts denoising-step allocation: higher favours layout, lower favours detail."}),
                "num_timesteps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 5, "tooltip": "Total denoising steps."}),
                "cfg_renorm_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "CFG-Renorm minimum. 1.0 disables CFG-Renorm."}),
                "cfg_renorm_type": (["global", "local", "text_channel"], {"default": "text_channel", "tooltip": "CFG-Renorm method. text_channel is the official image-edit default."}),
                "show_thinking": ("BOOLEAN", {"default": False, "tooltip": "Generate and return the model planning text before image sampling."}),
                "max_think_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64, "tooltip": "Maximum planning tokens when Thinking is enabled."}),
                "do_sample": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling for planning-text generation when Thinking is enabled."}),
                "text_temperature": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.1, "tooltip": "Planning-text randomness when Thinking is enabled."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "0 leaves the seed unset, matching the official app; positive values are reproducible."}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "reasoning")
    FUNCTION = "edit"
    CATEGORY = "BAGEL/Editing"

    def edit(self, model, image, vae_latent, prompt, cfg_text_scale, cfg_img_scale,
             cfg_interval, timestep_shift, num_timesteps, cfg_renorm_min,
             cfg_renorm_type, show_thinking, max_think_tokens, do_sample,
             text_temperature, seed):
        require_bagel_capability(model, "image_edit")
        require_single_image_batch(image)
        if "samples" not in vae_latent:
            raise ValueError("BAGEL vae_latent must contain the ComfyUI 'samples' tensor")
        require_single_image_batch(vae_latent["samples"], name="vae_latent.samples")
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
        if tuple(image.shape[1:3]) != (H, W):
            raise ValueError(
                "BAGEL IMAGE and VAEEncode LATENT describe different sizes: "
                f"IMAGE is {tuple(image.shape[1:3])}, LATENT is {(H, W)}. "
                "Connect the same preprocessed image to BAGEL Image Edit and VAEEncode."
            )
        expected_size = handle["image_transform"].resize_transform(pil).size
        if expected_size != pil.size:
            raise ValueError(
                f"BAGEL source image {pil.size[1]}x{pil.size[0]} must be preprocessed "
                f"to {expected_size[1]}x{expected_size[0]} before both VAEEncode and "
                "BAGEL Image Edit. Use an official ImageScale node."
            )

        apply_seed(seed)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            # Legacy image-then-text ordering (InterleaveInferencer.edit):
            # 1) VAE latent tokens (source image conditioning)
            # 2) ViT tokens (source image features)
            # 3) cfg_text snapshots the image-only context BEFORE the prompt
            # 4) prompt added to gen (full image+text conditioning)
            # 5) cfg_img is text-only (prompt built from an empty context)
            gen = init_gen_context(m)
            cfg_img = init_gen_context(m)
            if show_thinking:
                gen = update_context_text(handle, GEN_THINK_SYSTEM_PROMPT, gen)
                cfg_img = update_context_text(handle, GEN_THINK_SYSTEM_PROMPT, cfg_img)
            gen = update_vae_latent_from_latent(handle, vae_tensor, gen)
            gen = update_vit_image(handle, pil, gen)
            cfg_text = copy.deepcopy(gen)  # image-only baseline (text dropped)
            gen = update_context_text(handle, prompt, gen)  # full (image+text)
            cfg_img = update_context_text(handle, prompt, cfg_img)  # text-only

            reasoning = ""
            if show_thinking:
                from .modeling.bagel.runtime import generate_text
                reasoning = generate_text(
                    handle, gen, max_length=max_think_tokens, do_sample=do_sample,
                    temperature=text_temperature,
                )
                gen = update_context_text(handle, reasoning, gen)

            latent = generate_latent(
                handle, gen, cfg_text, cfg_img, (H, W),
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=(cfg_interval, 1.0),
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
            )
        return (latent, reasoning)


NODE_CLASS_MAPPINGS = {
    "BAGELImageEdit": BAGELImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAGELImageEdit": "BAGEL Image Edit",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
