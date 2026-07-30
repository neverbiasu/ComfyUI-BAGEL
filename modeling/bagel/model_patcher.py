"""ComfyUI-native model lifecycle wrapper for the complete coupled BAGEL model.

This defines a real ``comfy.model_patcher.ModelPatcher`` subclass. BAGEL
therefore integrates with ComfyUI's device/offload/cleanup lifecycle
(``model_management`` moves the patcher between devices, ``clone()`` is used for
per-prompt copies, and ``unload_all_models`` detaches it). No Accelerate
``dispatch_model`` / ``load_checkpoint_and_dispatch`` hooks are used: the
complete model is built once on a meta device and its weights are assigned from
the converted safetensors.

The attached BAGEL runtime state (tokenizer, special-token ids, vision
transform, converted metadata, checkpoint identity) is stored in ComfyUI's
native ``attachments`` dict (the supported attachment mechanism, which survives
``clone()``). The ``reload_factory`` is BAGEL-owned reload metadata -- it is NOT
a ComfyUI lifecycle callback. It is invoked by a defined BAGEL path
(:func:`reload_bagel_patcher` and by ``clone`` when it rebuilds the factory from
the immutable checkpoint identity) so a disk-backed rebuild from the converted
checkpoint is always available for cache invalidation / future multi-GPU work.
"""

from __future__ import annotations

import os
import inspect
from typing import Any, Callable, Dict, Optional


def _load_model_patcher_base():
    """Import ComfyUI's ``ModelPatcher`` (caller must run inside ComfyUI)."""
    from comfy.model_patcher import ModelPatcher

    return ModelPatcher


# Key under which BAGEL stores its runtime state in ComfyUI's attachments dict.
ATTACHMENT_KEY = "bagel"


class BagelModelPatcher:
    """Factory/builder namespace for the concrete ComfyUI patcher.

    The concrete class is subclassed from ComfyUI's ``ModelPatcher``; we cannot
    build that subclass at import time in a comfy-less environment, so the class
    is materialized lazily (once comfy is importable) and cached.
    """

    _cls = None

    @classmethod
    def cls(cls):
        if cls._cls is None:
            ModelPatcher = _load_model_patcher_base()

            class _BagelPatcher(ModelPatcher):
                """ComfyUI patcher wrapping one complete BAGEL model.

                Constructor is API-compatible with ``ModelPatcher.__init__`` so
                that ``ModelPatcher.clone()`` -- which calls
                ``self.__class__(model, load_device, offload_device, size,
                weight_inplace_update=...)`` -- works. BAGEL extras are
                keyword-only and default to ``None`` so a bare clone rebuild
                succeeds; :meth:`clone` then re-attaches them.
                """

                def __init__(
                    self,
                    model,
                    load_device,
                    offload_device,
                    size=0,
                    weight_inplace_update=False,
                    *,
                    bagel_state: Optional[Dict[str, Any]] = None,
                    checkpoint_identity: Optional[Dict[str, Any]] = None,
                ):
                    # Pass the ComfyUI constructor args (including
                    # weight_inplace_update) straight to the base.
                    super().__init__(
                        model, load_device, offload_device, size, weight_inplace_update
                    )
                    self.bagel_state = bagel_state
                    self.checkpoint_identity = dict(checkpoint_identity or {})
                    # Real ComfyUI attachment mechanism (survives clone via
                    # reference copy for non-cloneable attachment objects).
                    self.attachments[ATTACHMENT_KEY] = bagel_state
                    # BAGEL-owned reload metadata (see module docstring).
                    self.reload_factory: Callable[..., "ModelPatcher"] = _make_reload_factory(
                        self.checkpoint_identity, bagel_state, load_device, offload_device
                    )
                    # Record identity for inspection/logging only.
                    self.model_options["bagel_checkpoint"] = self.checkpoint_identity

                def clone(
                    self,
                    disable_dynamic=False,
                    model_override=None,
                    force_deepcopy=False,
                ):
                    # Newer ComfyUI releases accept clone controls for deep and
                    # multi-GPU copies, while older releases exposed clone()
                    # without arguments. Preserve both APIs.
                    clone_parameters = inspect.signature(super().clone).parameters
                    clone_kwargs = {}
                    if "disable_dynamic" in clone_parameters:
                        clone_kwargs["disable_dynamic"] = disable_dynamic
                    if "model_override" in clone_parameters:
                        clone_kwargs["model_override"] = model_override
                    if "force_deepcopy" in clone_parameters:
                        clone_kwargs["force_deepcopy"] = force_deepcopy
                    n = super().clone(**clone_kwargs)
                    # super().clone() rebuilt this instance via the base
                    # constructor (bagel_state=None); re-attach BAGEL state.
                    n.bagel_state = self.bagel_state
                    n.checkpoint_identity = self.checkpoint_identity
                    n.attachments[ATTACHMENT_KEY] = self.bagel_state
                    n.reload_factory = _make_reload_factory(
                        n.checkpoint_identity, n.bagel_state, n.load_device, n.offload_device
                    )
                    return n

            cls._cls = _BagelPatcher
        return cls._cls

    @classmethod
    def create(
        cls,
        model,
        load_device,
        offload_device,
        bagel_state: Dict[str, Any],
        checkpoint_identity: Optional[Dict[str, Any]] = None,
        size: int = 0,
    ):
        return cls.cls()(
            model,
            load_device,
            offload_device,
            size,
            False,
            bagel_state=bagel_state,
            checkpoint_identity=checkpoint_identity,
        )


def _make_reload_factory(
    checkpoint_identity: Dict[str, Any],
    bagel_state: Dict[str, Any],
    default_load_device=None,
    default_offload_device=None,
):
    """Return a callable that rebuilds the patcher from the immutable checkpoint.

    The factory closes over the checkpoint path and the previously attached
    runtime state so future multi-GPU / cache-invalidation reloads reconstruct an
    equivalent patcher without any runtime download. Invoked by
    :func:`reload_bagel_patcher` and (indirectly) by :meth:`_BagelPatcher.clone`.
    """

    identity = dict(checkpoint_identity or {})
    state_template = dict(bagel_state or {})

    def reload_factory(load_device=None, offload_device=None):
        from modeling.bagel.model_loader import load_native_bagel

        path = identity.get("path")
        if not path or not os.path.exists(path):
            raise RuntimeError(
                "Cannot reload BAGEL from disk: checkpoint identity is missing "
                f"or the file no longer exists ({path!r})."
            )
        ld = load_device if load_device is not None else default_load_device
        od = offload_device if offload_device is not None else default_offload_device
        return load_native_bagel(
            path,
            load_device=ld,
            offload_device=od,
            bagel_state_override=state_template,
        )

    return reload_factory


def reload_bagel_patcher(patcher, load_device=None, offload_device=None):
    """Defined BAGEL path that invokes the disk-backed reload factory.

    This is the supported way for BAGEL code (and ``clone``) to rebuild a patcher
    from the immutable converted checkpoint. It is BAGEL-owned, not a ComfyUI
    lifecycle callback.
    """
    factory = getattr(patcher, "reload_factory", None)
    if factory is None:
        raise RuntimeError(
            "This BAGEL_MODEL patcher has no reload_factory; load it with the "
            "native BAGEL Model Loader."
        )
    return factory(load_device, offload_device)


def make_vae_config():
    """Minimal VAE geometry config the coupled Bagel reads for latent math.

    The coupled VAE *weights* are NOT loaded; only the constants Bagel needs
    (``downsample`` -> latent_downsample, ``z_channels`` -> latent_channel) are
    provided. Official ComfyUI FLUX VAE nodes perform real encode/decode.
    """

    class _VAEConfig:
        downsample = 8
        z_channels = 16

    return _VAEConfig()
