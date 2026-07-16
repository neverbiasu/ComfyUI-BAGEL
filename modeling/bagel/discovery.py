"""Pure-stdlib discovery of BAGEL safetensors files (no torch / ComfyUI import).

This module deliberately imports nothing heavy. It is the single source of
truth for the ``discover_converted_bagel`` behaviour so it can be unit-checked
with a fake ``folder_paths`` shim (see ``scripts/validate_discovery.py``).
"""

from typing import Callable, Dict, Iterable, Optional


def discover_converted_bagel(
    get_filename_list: Callable[[str], "list[str]"],
    get_full_path: Callable[[str, str], Optional[str]],
    folder_names: str | Iterable[str] = "bagel",
) -> Dict[str, str]:
    """Return ``{relative_display_name: full_path}`` for BAGEL safetensors files.

    Args:
        get_filename_list: ComfyUI ``folder_paths.get_filename_list`` (recursive,
            extension-filtered, supports nested model paths; returns relative
            names such as ``subdir/model.safetensors``).
        get_full_path: ComfyUI ``folder_paths.get_full_path`` (resolves a
            relative name to a full path across all base folders).
        folder_names: folder categories to scan. Native BAGEL models should live
            under ``"bagel"``; ``"diffusion_models"`` may be included as a
            backwards-compatible migration path.

    Nested model paths are supported via the recursive ``get_filename_list``.
    Discovery deliberately does NOT open or validate metadata. This matches
    ComfyUI's usual loader UX: list files from the model folder, then validate
    at load time.
    """
    if isinstance(folder_names, str):
        scan_folders = (folder_names,)
    else:
        scan_folders = tuple(folder_names)

    found: Dict[str, str] = {}
    for folder_name in scan_folders:
        for name in sorted(get_filename_list(folder_name)):
            if not name.endswith(".safetensors"):
                continue
            path = get_full_path(folder_name, name)
            if path is None:
                continue
            display_name = name if folder_name == "bagel" else f"{folder_name}/{name}"
            found[display_name] = path
    if not found:
        print(
            "[BAGEL] no .safetensors checkpoints found in models/bagel."
        )
    return found
