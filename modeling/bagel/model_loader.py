"""Complete converted BAGEL model construction (no inferencer, no dispatch).

This module owns the heavy work the native loader node used to do inline:

* discovery of BAGEL ``.safetensors`` files in ComfyUI's ``models/bagel``
  folder (with a ``diffusion_models`` migration fallback);
* optional embedded or hash-bound-sidecar metadata validation with fallback to
  the built-in BAGEL-7B-MoT config;
* building the complete coupled BAGEL model on a meta device and assigning
  every converted weight (no Accelerate ``dispatch_model`` / ``load_checkpoint_and_dispatch``);
* constructing the packaged tokenizer and optionally verifying it against model
  metadata when present (no runtime download);
* wrapping everything in a ComfyUI-native :class:`BagelModelPatcher`.

The returned object is a ``BAGEL_MODEL`` handle: a ``ModelPatcher`` whose
attached ``bagel_state`` dict carries the tokenizer, special-token IDs, vision
transform, metadata, and an immutable checkpoint identity. It contains no VAE
weights and no long-lived inferencer.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional

import safetensors.torch as _sf_torch
import torch
from safetensors import safe_open as _sf_safe_open
from folder_paths import get_filename_list, get_full_path

from data.transforms import ImageTransform
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.bagel.converted_format import (
    CRITICAL_PREFIXES,
    FORMAT_NAME,
    ConvertedBagelMetadata,
)
from modeling.bagel.discovery import discover_converted_bagel as _discover_converted_bagel
from modeling.bagel.model_patcher import BagelModelPatcher, make_vae_config
from modeling.bagel.model_types import CapabilityTier
from modeling.bagel.variants import detect_variant
from modeling.qwen2.bagel_tokenizer import (
    REQUIRED_SPECIAL_TOKENS,
    load_packaged_tokenizer,
    required_special_token_ids,
)
from modeling.qwen2.tokenizer_fingerprint import tokenizer_fingerprint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOKENIZER_DIR = os.path.join(REPO_ROOT, "modeling", "qwen2", "tokenizer")

BAGEL_METADATA_KEY = FORMAT_NAME
SIDECAR_SUFFIX = ".comfyui-bagel.json"

# The packaged Qwen2 tokenizer is bundled with BAGEL's required special tokens.
# Some converted checkpoints in the wild do not carry config metadata, and the
# local Qwen2Config default does not define these PretrainedConfig attributes
# unless they are passed explicitly. BAGEL's Qwen2Model reads them during
# construction, so keep the fallback aligned with the packaged tokenizer.
DEFAULT_QWEN_PAD_TOKEN_ID = 151643  # <|endoftext|>
DEFAULT_QWEN_BOS_TOKEN_ID = 151644  # <|im_start|>
DEFAULT_QWEN_EOS_TOKEN_ID = 151645  # <|im_end|>


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_metadata(
    path: str, checkpoint_sha256: Optional[str] = None
) -> Optional[ConvertedBagelMetadata]:
    """Read + parse optional converted metadata header or sidecar."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"BAGEL checkpoint not found: {path}")
    try:
        with _sf_safe_open(path, framework="pt", device="cpu") as f:
            header = f.metadata() or {}
    except Exception as exc:  # not a safetensors file at all
        raise ValueError(
            f"{os.path.basename(path)} is not a safetensors checkpoint "
            f"(raw HuggingFace layouts are not supported by the native loader). "
            f"Convert it with scripts/convert_bagel_model.py first. Underlying error: {exc}"
        ) from exc

    metadata_json = header.get(BAGEL_METADATA_KEY)
    if metadata_json is None:
        sidecar_path = path + SIDECAR_SUFFIX
        if not os.path.isfile(sidecar_path):
            print(
                f"[BAGEL] {os.path.basename(path)} has no embedded "
                f"'{BAGEL_METADATA_KEY}' metadata or sidecar; using built-in "
                "BAGEL-7B-MoT config fallback."
            )
            return None
        try:
            with open(sidecar_path, encoding="utf-8") as f:
                sidecar = json.load(f)
        except Exception as exc:
            raise ValueError(f"Malformed BAGEL sidecar {sidecar_path}: {exc}") from exc
        if sidecar.get("format") != "comfyui_bagel_sidecar" or sidecar.get("format_version") != 1:
            raise ValueError(f"Unsupported BAGEL sidecar schema: {sidecar_path}")
        expected_size = sidecar.get("checkpoint_size")
        if expected_size != os.path.getsize(path):
            raise ValueError(
                f"BAGEL sidecar size mismatch for {os.path.basename(path)}: "
                f"expected {expected_size}, got {os.path.getsize(path)}"
            )
        expected_sha = sidecar.get("checkpoint_sha256")
        if checkpoint_sha256 is not None and expected_sha != checkpoint_sha256:
            raise ValueError(
                f"BAGEL sidecar SHA-256 mismatch for {os.path.basename(path)}: "
                f"expected {expected_sha}, got {checkpoint_sha256}"
            )
        metadata = sidecar.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"BAGEL sidecar metadata must be an object: {sidecar_path}")
        metadata_json = json.dumps(metadata)
    try:
        return ConvertedBagelMetadata.from_json(metadata_json)
    except Exception as exc:
        raise ValueError(
            f"{os.path.basename(path)} carries a malformed "
            f"'{BAGEL_METADATA_KEY}' metadata block: {exc}"
        ) from exc


def discover_converted_bagel() -> Dict[str, str]:
    """Return ``{relative_display_name: safetensors_path}`` for BAGEL files.

    Uses ComfyUI's official filename cache and path resolution
    (``get_filename_list`` + ``get_full_path``) so nested model paths and the
    standard folder resolution are honoured. Scans the dedicated ``bagel``
    folder first, then ``diffusion_models`` only as a migration fallback.
    Metadata is validated only at load time, matching ComfyUI's usual model
    loader behaviour.
    """
    return _discover_converted_bagel(
        get_filename_list,
        get_full_path,
        folder_names=("bagel", "diffusion_models"),
    )


def _build_config(metadata: Optional[ConvertedBagelMetadata]):
    """Construct the ``BagelConfig`` from metadata or built-in defaults."""
    model_configs = (metadata.model_configs if metadata else {}) or {}
    if "llm_config.json" in model_configs:
        llm_config = Qwen2Config.from_dict(model_configs["llm_config.json"])
    else:
        llm_config = Qwen2Config(
            pad_token_id=DEFAULT_QWEN_PAD_TOKEN_ID,
            bos_token_id=DEFAULT_QWEN_BOS_TOKEN_ID,
            eos_token_id=DEFAULT_QWEN_EOS_TOKEN_ID,
        )
    if getattr(llm_config, "pad_token_id", None) is None:
        llm_config.pad_token_id = DEFAULT_QWEN_PAD_TOKEN_ID
    if getattr(llm_config, "bos_token_id", None) is None:
        llm_config.bos_token_id = DEFAULT_QWEN_BOS_TOKEN_ID
    if getattr(llm_config, "eos_token_id", None) is None:
        llm_config.eos_token_id = DEFAULT_QWEN_EOS_TOKEN_ID
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    if "vit_config.json" in model_configs:
        vit_config = SiglipVisionConfig.from_dict(model_configs["vit_config.json"])
    else:
        vit_config = SiglipVisionConfig()
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    return BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=make_vae_config(),
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )


def _tensor_keys(path: str) -> list[str]:
    """Read safetensors keys without loading the checkpoint weights."""
    with _sf_safe_open(path, framework="pt", device="cpu") as f:
        return list(f.keys())


def _default_comfy_devices():
    """Ask ComfyUI for the active execution and offload devices."""
    from comfy import model_management

    return model_management.get_torch_device(), model_management.unet_offload_device()


def _build_model(state_dict: Dict[str, torch.Tensor], config: BagelConfig) -> Bagel:
    """Build the coupled BAGEL skeleton on a meta device and assign weights."""
    from accelerate import init_empty_weights

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(config.llm_config)
        vit_model = SiglipVisionModel(config.vit_config)
        model_obj = Bagel(language_model, vit_model, config)
        model_obj.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
            config.vit_config, meta=True
        )

    missing, unexpected = model_obj.load_state_dict(state_dict, assign=True, strict=False)

    missing = [k for k in missing if k not in state_dict]
    # Only *parameter* weights are critical; missing buffers are tolerated.
    critical_missing = [
        k for k in missing if k in dict(model_obj.named_parameters())
    ]
    if critical_missing:
        present_roots = {k.split(".", 1)[0] for k in state_dict}
        missing_roots = [p for p in CRITICAL_PREFIXES if p not in present_roots]
        raise KeyError(
            "Converted BAGEL weights are incomplete. Missing critical weights "
            f"(first 20): {sorted(critical_missing)[:20]}. "
            f"Missing coupled-module roots: {missing_roots}. "
            "Re-run the converter; do not point the loader at a partial file."
        )
    if unexpected:
        # Unexpected keys mean the file does not match this model.
        raise KeyError(
            "BAGEL checkpoint contains unexpected weights not present in the "
            f"model (first 20): {sorted(unexpected)[:20]}. The file "
            "may target a different variant; verify the converter and source."
        )
    return model_obj


def _load_tokenizer(metadata: Optional[ConvertedBagelMetadata]):
    """Construct the packaged tokenizer and validate it against metadata."""
    tokenizer = load_packaged_tokenizer(TOKENIZER_DIR)
    if tokenizer is None:
        raise RuntimeError(
            "BAGEL packaged tokenizer assets are missing under "
            f"{TOKENIZER_DIR}. The native loader never downloads a tokenizer; "
            "install the custom node fully (including modeling/qwen2/tokenizer)."
        )

    ids = required_special_token_ids(tokenizer)
    missing_tokens = [
        tok for tok in REQUIRED_SPECIAL_TOKENS if ids.get(tok) is None
    ]
    if missing_tokens:
        raise ValueError(
            f"Packaged tokenizer is missing required special tokens: {missing_tokens}. "
            "The tokenizer assets under modeling/qwen2/tokenizer are incompatible "
            "with this BAGEL variant."
        )

    # Cross-check the converted metadata's recorded special-token IDs.
    expected = (metadata.special_token_ids if metadata else {}) or {}
    if expected:
        mismatched = {
            tok: (ids.get(tok), expected[tok])
            for tok in REQUIRED_SPECIAL_TOKENS
            if tok in expected and ids.get(tok) != expected[tok]
        }
        if mismatched:
            raise ValueError(
                "Tokenizer special-token IDs do not match the converted metadata: "
                f"{mismatched}. The converted file and the packaged tokenizer "
                "originate from different BAGEL revisions."
            )

    # Cross-check the vocabulary fingerprint when the converter recorded one.
    # Uses the canonical pure-stdlib fingerprint over the packaged assets, so it
    # does not depend on the loaded tokenizer object or on torch.
    if metadata and metadata.tokenizer_fingerprint:
        fp = tokenizer_fingerprint(TOKENIZER_DIR)
        if fp != metadata.tokenizer_fingerprint:
            raise ValueError(
                "Tokenizer vocabulary fingerprint does not match the converted "
                f"metadata (got {fp}, expected {metadata.tokenizer_fingerprint}). "
                "The packaged tokenizer assets differ from the conversion source."
            )

    return tokenizer


def load_native_bagel(
    path: str,
    load_device: Optional[str] = None,
    offload_device: Optional[str] = None,
    bagel_state_override: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load a BAGEL safetensors file into a ``BAGEL_MODEL`` patcher.

    Returns a :class:`BagelModelPatcher` carrying the complete model plus the
    attached runtime state and an immutable checkpoint identity. No VAE weights and
    no long-lived inferencer are created; no weights are downloaded.
    """
    checkpoint_sha = _sha256_file(path)
    metadata = _read_metadata(path, checkpoint_sha256=checkpoint_sha)

    if metadata and metadata.format != FORMAT_NAME:
        raise ValueError(
            f"{os.path.basename(path)} has unknown format {metadata.format!r}; "
            f"the native loader requires {FORMAT_NAME!r}."
        )
    if metadata and metadata.format_version != 1:
        raise ValueError(
            f"{os.path.basename(path)} has unsupported format version "
            f"{metadata.format_version}; only version 1 is accepted."
        )
    if metadata and (
        metadata.model_options.get("visual_und", True) is False
        or metadata.additional_special_tokens
    ):
        raise NotImplementedError(
            f"{metadata.variant or os.path.basename(path)} is a valid converted "
            "BAGEL-family checkpoint, but its runtime contract requires a "
            "variant adapter (model_options/additional_special_tokens) that is "
            "not implemented by the current native nodes. The checkpoint is "
            "discoverable for forward compatibility; do not treat metadata "
            "conversion as runtime support."
        )

    if metadata:
        descriptor = detect_variant(metadata, _tensor_keys(path))
        if descriptor.tier != CapabilityTier.NATIVE:
            raise NotImplementedError(
                f"{descriptor.name or os.path.basename(path)} is detected as "
                f"{descriptor.tier.value}, not as the validated BAGEL-7B-MoT BF16 "
                "native runtime. Install or implement its dedicated adapter; do not "
                "run it through the base BAGEL nodes. "
                f"Detection: {descriptor.detection_source}."
            )
        descriptor_dict = descriptor.to_dict()
    else:
        # Legacy converted files without metadata retain the documented base
        # fallback. They cannot be structurally identified beyond the strict
        # base model construction check below.
        descriptor_dict = {
            "name": "BAGEL-7B-MoT",
            "architecture": "Bagel",
            "variant": "BAGEL-7B-MoT",
            "dtype": "bf16",
            "quantization": "none",
            "tier": CapabilityTier.NATIVE.value,
            "capabilities": ["text_to_image", "image_edit", "image_understanding"],
            "detection_source": "metadata-free base fallback",
        }

    default_load_device, default_offload_device = _default_comfy_devices()
    load_device = load_device or default_load_device
    offload_device = offload_device or default_offload_device

    state_dict = _sf_torch.load_file(path, device="cpu")
    config = _build_config(metadata)
    model_obj = _build_model(state_dict, config)

    dtype_label = metadata.dtype if metadata else "bf16"
    target_dtype = getattr(torch, {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }.get(dtype_label, "bfloat16"))
    model_obj = model_obj.to(target_dtype).eval()

    tokenizer = _load_tokenizer(metadata)

    new_token_ids = {
        "bos_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "start_of_image": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_image": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }
    # Keep the two original transforms distinct: image conditioning is resized
    # with the VAE transform before its VIT pass; NaViT performs its own final
    # 980/224/14 processing inside ``prepare_vit_images``.
    image_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    checkpoint_identity = {
        "path": os.path.abspath(path),
        "sha256": checkpoint_sha,
        "format_version": metadata.format_version if metadata else 0,
        "variant": metadata.variant if metadata else "BAGEL-7B-MoT",
        "dtype": metadata.dtype if metadata else "bf16",
    }

    bagel_state = {
        "tokenizer": tokenizer,
        "new_token_ids": new_token_ids,
        "image_transform": image_transform,
        "vit_transform": vit_transform,
        "variant_descriptor": descriptor_dict,
        "metadata": metadata.to_dict() if metadata else {
            "format": "fallback",
            "format_version": 0,
            "variant": "BAGEL-7B-MoT",
            "dtype": "bf16",
            "quantization": "none",
            "capabilities": ["text_to_image", "image_editing", "image_understanding"],
            "source": "built-in config fallback",
        },
        "checkpoint_identity": checkpoint_identity,
    }
    if bagel_state_override:
        # Preserve reload identity from an earlier load (clone path).
        bagel_state["checkpoint_identity"] = dict(
            bagel_state_override.get("checkpoint_identity", checkpoint_identity)
        )

    patcher = BagelModelPatcher.create(
        model_obj,
        load_device,
        offload_device,
        bagel_state,
        checkpoint_identity,
    )

    def _cached_loader(*, disable_dynamic=False):
        # ModelPatcher.clone(force_deepcopy=True) invokes this factory with
        # disable_dynamic=True. BAGEL is not dynamic, so the flag is accepted
        # for API compatibility and intentionally has no effect.
        return load_native_bagel(
            path,
            load_device=load_device,
            offload_device=offload_device,
        )

    patcher.cached_patcher_init = (_cached_loader, ())
    print(
        f"[BAGEL] loaded model from {path} "
        f"(variant={checkpoint_identity['variant']}, dtype={checkpoint_identity['dtype']})"
    )
    return patcher
