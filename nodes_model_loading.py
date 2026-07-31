"""Native BAGEL model loader node (VAE-decoupled, no runtime download)."""

from __future__ import annotations

import os

from folder_paths import folder_names_and_paths, models_dir as comfy_models_dir

from .modeling.bagel.model_loader import discover_converted_bagel, load_native_bagel


def _register_bagel_model_folder() -> None:
    """Register BAGEL model paths without clobbering extra_model_paths.yaml.

    ComfyUI users often add model folders through ``extra_model_paths.yaml``.
    Custom nodes must preserve those paths and only append their default folder
    / extensions. This mirrors the community pattern used by model-heavy nodes:
    extend ``folder_names_and_paths`` rather than replacing it.
    """

    default_path = os.path.join(comfy_models_dir, "bagel")
    paths, extensions = folder_names_and_paths.get("bagel", ([], []))
    merged_paths = list(paths)
    if default_path not in merged_paths:
        merged_paths.append(default_path)

    merged_extensions = list(extensions)
    for extension in (".safetensors", ".json"):
        if extension not in merged_extensions:
            merged_extensions.append(extension)

    folder_names_and_paths["bagel"] = (merged_paths, merged_extensions)


_register_bagel_model_folder()


class BAGELModelLoader:
    """Load a converted (single-file) BAGEL model from ``models/bagel``.

    The native loader lists standard ComfyUI model files from ``models/bagel``.
    Optional ComfyUI-BAGEL metadata/sidecars are used when present; otherwise it
    falls back to the built-in BAGEL-7B-MoT config. It never auto-downloads
    weights or a tokenizer.
    """

    @classmethod
    def INPUT_TYPES(cls):
        discovered = discover_converted_bagel()
        choices = list(discovered.keys()) or ["undefined"]
        return {
            "required": {
                "model": (
                    choices,
                    {
                        "default": choices[0],
                        "tooltip": "BAGEL .safetensors placed in ComfyUI/models/bagel",
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
                f"BAGEL checkpoint not found under models/bagel: {model!r}. "
                "Place a .safetensors file there and refresh ComfyUI."
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
