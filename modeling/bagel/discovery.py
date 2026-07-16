"""Pure-stdlib discovery of converted BAGEL files (no torch / ComfyUI import).

This module deliberately imports nothing heavy. It is the single source of
truth for the ``discover_converted_bagel`` behaviour so it can be unit-checked
with a fake ``folder_paths`` shim (see ``scripts/validate_discovery.py``).
"""

from typing import Any, Callable, Dict, Iterable, Optional


def discover_converted_bagel(
    get_filename_list: Callable[[str], "list[str]"],
    get_full_path: Callable[[str, str], Optional[str]],
    read_metadata: Callable[[str], Any],
    folder_names: str | Iterable[str] = "bagel",
) -> Dict[str, str]:
    """Return ``{relative_display_name: full_path}`` for converted BAGEL files.

    Args:
        get_filename_list: ComfyUI ``folder_paths.get_filename_list`` (recursive,
            extension-filtered, supports nested model paths; returns relative
            names such as ``subdir/model.safetensors``).
        get_full_path: ComfyUI ``folder_paths.get_full_path`` (resolves a
            relative name to a full path across all base folders).
        read_metadata: callback that parses/validates a file's converted
            metadata header and raises on non-converted / malformed files.
        folder_names: folder categories to scan. Native BAGEL models should live
            under ``"bagel"``; ``"diffusion_models"`` may be included as a
            backwards-compatible migration path.

    Nested model paths are supported via the recursive ``get_filename_list``.
    Files that are not converted BAGEL safetensors (raw checkpoints, unrelated
    files) are skipped via the ``read_metadata`` exception, never partially
    loaded.
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
            try:
                read_metadata(path)
            except Exception as exc:
                print(
                    f"[BAGEL] skipping {folder_name} entry "
                    f"{name!r}: not a loadable converted BAGEL checkpoint ({exc})"
                )
                continue
            display_name = name if folder_name == "bagel" else f"{folder_name}/{name}"
            found[display_name] = path
    if not found:
        print(
            "[BAGEL] no converted BAGEL checkpoints found in models/bagel. "
            "Expected a .safetensors file with embedded 'comfyui_bagel' metadata "
            "or a matching .comfyui-bagel.json sidecar."
        )
    return found
