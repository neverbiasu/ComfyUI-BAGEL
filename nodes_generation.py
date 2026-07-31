"""Native BAGEL text-to-image node. Returns a ComfyUI LATENT (VAE-decoupled)."""

from __future__ import annotations

import copy

import comfy.model_management as model_management
import torch

from .modeling.bagel.runtime import (
    generate_latent,
    generate_text,
    init_gen_context,
    update_context_text,
    validate_bagel_image_shape,
)
from .nodes_common import (
    GEN_THINK_SYSTEM_PROMPT,
    apply_seed,
    build_handle,
    require_bagel_capability,
)


class BAGELTextToImage:
    """Generate an image (standard FLUX ``LATENT``) from a text prompt.

    The node never loads or accepts a VAE. The returned latent is decoded
    downstream by the official ``VAEDecode`` node using the FLUX AE.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_MODEL", {"tooltip": "Native BAGEL model from BAGEL Model Loader."}),
                "prompt": ("STRING", {"multiline": True, "default": "A female cosplayer portraying an ethereal fairy or elf, wearing a flowing dress made of delicate fabrics in soft, mystical colors like emerald green and silver. She has pointed ears, a gentle, enchanting expression, and her outfit is adorned with sparkling jewels and intricate patterns. The background is a magical forest with glowing plants, mystical creatures, and a serene atmosphere.", "tooltip": "Text prompt for image generation."}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 1024, "step": 16, "tooltip": "Output width. This is the ComfyUI equivalent of the official app's image-ratio preset; use 16-pixel aligned dimensions."}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 1024, "step": 16, "tooltip": "Output height. This is the ComfyUI equivalent of the official app's image-ratio preset; use 16-pixel aligned dimensions."}),
                "cfg_text_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 8.0, "step": 0.1, "tooltip": "Controls how strongly BAGEL follows the text prompt; the official app recommends 4.0–8.0."}),
                "cfg_interval": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "Start of the CFG interval. The end is fixed at 1.0, matching the official app."}),
                "timestep_shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.5, "tooltip": "Shifts denoising-step allocation: higher favours layout, lower favours detail."}),
                "num_timesteps": ("INT", {"default": 50, "min": 10, "max": 100, "step": 5, "tooltip": "Total denoising steps."}),
                "cfg_renorm_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "CFG-Renorm minimum. 1.0 disables CFG-Renorm."}),
                "cfg_renorm_type": (["global", "local", "text_channel"], {"default": "global", "tooltip": "CFG-Renorm method. global is the official text-to-image default."}),
                "show_thinking": ("BOOLEAN", {"default": False, "tooltip": "Generate and return the model planning text before image sampling."}),
                "max_think_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 64, "tooltip": "Maximum planning tokens when Thinking is enabled."}),
                "do_sample": ("BOOLEAN", {"default": False, "tooltip": "Enable sampling for planning-text generation when Thinking is enabled."}),
                "text_temperature": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.1, "tooltip": "Planning-text randomness when Thinking is enabled."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1, "tooltip": "0 leaves the seed unset, matching the official app; positive values are reproducible."}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "reasoning")
    FUNCTION = "generate"
    CATEGORY = "BAGEL/Generation"

    @classmethod
    def VALIDATE_INPUTS(cls, width, height, **_kwargs):
        """Reject externally supplied dimensions that cannot form BAGEL patches."""
        for name, value in (("width", width), ("height", height)):
            if not 256 <= value <= 1024:
                return f"{name} must be between 256 and 1024"
            if value % 16:
                return f"{name} must be divisible by 16"
        return True

    def generate(self, model, prompt, width, height, cfg_text_scale, cfg_interval,
                 timestep_shift, num_timesteps, cfg_renorm_min, cfg_renorm_type,
                 show_thinking, max_think_tokens, do_sample, text_temperature, seed):
        require_bagel_capability(model, "text_to_image")
        model_management.load_models_gpu([model])
        handle = build_handle(model)
        m = handle["model"]
        device = next(m.parameters()).device
        validate_bagel_image_shape(m, (height, width))

        apply_seed(seed)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            gen = init_gen_context(m)
            cfg_text = init_gen_context(m)
            cfg_img = init_gen_context(m)
            if show_thinking:
                gen = update_context_text(handle, GEN_THINK_SYSTEM_PROMPT, gen)
                cfg_img = update_context_text(handle, GEN_THINK_SYSTEM_PROMPT, cfg_img)

            # Preserve InterleaveInferencer's text ordering. cfg_text snapshots
            # the context before the user prompt; cfg_img carries text only.
            cfg_text = copy.deepcopy(gen)
            gen = update_context_text(handle, prompt, gen)
            cfg_img = update_context_text(handle, prompt, cfg_img)

            reasoning = ""
            if show_thinking:
                reasoning = generate_text(
                    handle, gen, max_length=max_think_tokens, do_sample=do_sample,
                    temperature=text_temperature,
                )
                gen = update_context_text(handle, reasoning, gen)

            latent = generate_latent(
                handle, gen, cfg_text, cfg_img, (height, width),
                cfg_text_scale=cfg_text_scale,
                # T2I has no image-conditioning widget in the official app;
                # retain inferencer.py's fixed default for this branch.
                cfg_img_scale=1.5,
                cfg_interval=(cfg_interval, 1.0),
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
            )
        return (latent, reasoning)


NODE_CLASS_MAPPINGS = {
    "BAGELTextToImage": BAGELTextToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAGELTextToImage": "BAGEL Text to Image",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
