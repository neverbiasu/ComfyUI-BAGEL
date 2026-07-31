"""ComfyUI node registration for BAGEL.

Native nodes are the default BAGEL nodes. Legacy all-in-one nodes remain
registered from ``nodes_deprecated.py`` with deprecated display names so older
workflows can still load.
"""

from __future__ import annotations

import importlib


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

NODE_MODULES = (
    (".nodes_model_loading", "ModelLoading"),
    (".nodes_generation", "Generation"),
    (".nodes_editing", "Editing"),
    (".nodes_understanding", "Understanding"),
    (".nodes_deprecated", "Deprecated"),
)


def register_nodes(module_path: str, name: str) -> None:
    module = importlib.import_module(module_path, package=__package__)
    classes = getattr(module, "NODE_CLASS_MAPPINGS", {})
    displays = getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    duplicates = NODE_CLASS_MAPPINGS.keys() & classes.keys()
    if duplicates:
        raise RuntimeError(f"Duplicate BAGEL node class types in {name}: {sorted(duplicates)}")
    NODE_CLASS_MAPPINGS.update(classes)
    NODE_DISPLAY_NAME_MAPPINGS.update(displays)


for _module_path, _name in NODE_MODULES:
    register_nodes(_module_path, _name)


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
