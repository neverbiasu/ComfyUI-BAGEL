"""Native BAGEL model loader node (VAE-decoupled, no runtime download)."""

from __future__ import annotations

from .modeling.bagel.model_loader import discover_converted_bagel, load_native_bagel


class BAGELModelLoader:
    """Load a converted (single-file) BAGEL model from ``models/diffusion_models``.

    The native loader accepts ComfyUI-BAGEL metadata embedded in safetensors or
    a hash-bound repack sidecar next to a canonical single-file checkpoint. It
    never interprets raw HuggingFace shard layouts,
    never auto-downloads weights or a tokenizer, and returns a ``BAGEL_MODEL``
    patcher whose attached state carries the packaged tokenizer and an immutable
    checkpoint identity.
    """

    @classmethod
    def INPUT_TYPES(cls):
        discovered = discover_converted_bagel()
        choices = list(discovered.keys()) or ["(no converted BAGEL found)"]
        return {
            "required": {
                "model": (
                    choices,
                    {
                        "default": choices[0],
                        "tooltip": "Converted BAGEL .safetensors from convert_bagel_model.py, placed in ComfyUI/models/diffusion_models",
                    },
                ),
            }
        }

    RETURN_TYPES = ("BAGEL_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "BAGEL/Core"

    def load_model(self, model):
        discovered = discover_converted_bagel()
        if model not in discovered:
            raise ValueError(
                f"Converted BAGEL not found under models/diffusion_models: {model!r}. "
                "Place a converted .safetensors file there (see scripts/convert_bagel_model.py)."
            )
        path = discovered[model]
        return (load_native_bagel(path),)


NODE_CLASS_MAPPINGS = {
    "BAGELModelLoader": BAGELModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BAGELModelLoader": "BAGEL Model Loader",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
