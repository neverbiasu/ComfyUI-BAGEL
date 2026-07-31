"""VAE-decoupled BAGEL runtime.

This module is the faithful, VAE-free re-implementation of the inference loop
that previously lived in ``inferencer.py`` (``InterleaveInferencer``). It calls
the model's own forward/cache-update methods directly and never constructs an
``InterleaveInferencer`` backed by the coupled VAE, so:

* Generation returns a standard ComfyUI ``LATENT`` (decoded by the official
  FLUX VAE ``VAEDecode`` node downstream).
* Editing injects a user-supplied VAE latent (from the official ``VAEEncode``)
  instead of encoding an image with the coupled VAE.
* Understanding uses only the VIT encoder (part of the model), no VAE.
* No model weights or tokenizer/configs are downloaded at runtime.

The only method body replicated here is ``forward_cache_update_vae`` (and its
``prepare_vae_images`` context builder), because the original applies
``vae_model.encode`` internally; we substitute the caller's packed VAE latent.
"""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch

from modeling.bagel.latent import (
    bagel_scaled_to_comfy_raw,
    comfy_raw_to_bagel_scaled,
    to_comfy_latent,
)
from modeling.bagel.qwen2_navit import NaiveCache


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def validate_bagel_image_shape(model, image_shape: Tuple[int, int]) -> None:
    """Reject image sizes that cannot be represented by BAGEL's position grid.

    ``latent_downsample`` includes the VAE downsample and BAGEL's latent patch
    size (normally 16 pixels).  The learned 2-D position table has
    ``max_latent_size`` entries per side, so allowing a larger packed grid
    would only fail later as an opaque CUDA index assertion.
    """
    height, width = (int(image_shape[0]), int(image_shape[1]))
    downsample = int(getattr(model, "latent_downsample", 0))
    max_side = int(getattr(model, "max_latent_size", 0))
    patch_size = int(getattr(model, "latent_patch_size", 1))
    if downsample <= 0 or max_side <= 0:
        raise ValueError("BAGEL model is missing valid latent geometry metadata")
    if height <= 0 or width <= 0:
        raise ValueError(f"BAGEL image size must be positive, got {(height, width)}")

    grid_h, grid_w = height // downsample, width // downsample
    if grid_h < 1 or grid_w < 1:
        raise ValueError(
            f"BAGEL image {(height, width)} is smaller than one latent patch "
            f"({downsample}px)"
        )
    if grid_h > max_side or grid_w > max_side:
        max_pixels = max_side * downsample
        raise ValueError(
            f"BAGEL image {(height, width)} produces a {grid_h}x{grid_w} "
            f"latent grid, but this model supports at most {max_side}x{max_side}. "
            f"Use dimensions no larger than {max_pixels}px per side."
        )
    # The packed representation drops an incomplete edge.  Reject it instead
    # of silently conditioning on a cropped latent and generating another size.
    vae_downsample = downsample // patch_size
    if vae_downsample > 0:
        h_lat, w_lat = height // vae_downsample, width // vae_downsample
        if h_lat % patch_size or w_lat % patch_size:
            raise ValueError(
                f"BAGEL image {(height, width)} yields VAE latent grid "
                f"{h_lat}x{w_lat}; both sides must be divisible by latent "
                f"patch size {patch_size}. Resize/crop to a compatible size."
            )


def _move_tensors_to_device(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_tensors_to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_move_tensors_to_device(v, device) for v in value]
    return value


# Special token ids required by the packed-sequence builders. Derived from the
# (bundled) tokenizer so no extra config lookups are needed.
def build_new_token_ids(tokenizer) -> Dict[str, int]:
    return {
        "bos_token_id": tokenizer.convert_tokens_to_ids("<|im_start|>"),
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
        "start_of_image": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "end_of_image": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    }


def init_gen_context(model) -> Dict:
    return {
        "kv_lens": [0],
        "ropes": [0],
        "past_key_values": NaiveCache(model.config.llm_config.num_hidden_layers),
    }


def update_context_text(handle, text: str, gen_context: Dict) -> Dict:
    model = handle["model"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]

    generation_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        prompts=[text],
        tokenizer=handle["tokenizer"],
        new_token_ids=handle["new_token_ids"],
    )
    generation_input = _move_tensors_to_device(generation_input, _model_device(model))
    past_key_values = model.forward_cache_update_text(
        gen_context["past_key_values"], **generation_input
    )
    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = past_key_values
    return gen_context


def prepare_vae_latent_from_size(model, curr_kvlens, curr_rope, H, W, new_token_ids):
    """Size-based replica of ``Bagel.prepare_vae_images`` (no image/transform).

    Returns the same ``generation_input`` dict the model's
    ``forward_cache_update_vae`` expects, except without ``padded_images``.
    """
    packed_vae_token_indexes = []
    packed_text_ids, packed_text_indexes = [], []
    packed_seqlens, packed_position_ids, packed_indexes = [], [], []
    packed_key_value_indexes = []
    packed_vae_position_ids = []

    _curr = curr = 0
    newlens, new_rope = [], []
    for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen

        packed_text_ids.append(new_token_ids["start_of_image"])
        packed_text_indexes.append(_curr)
        packed_indexes.append(curr)
        curr += 1
        _curr += 1

        vae_position_ids = model.get_flattened_position_ids(
            H, W, model.latent_downsample, max_num_patches_per_side=model.max_latent_size
        )
        packed_vae_position_ids.append(vae_position_ids)
        h = H // model.latent_downsample
        w = W // model.latent_downsample
        patchified_vae_latent_shapes = [(h, w)]

        num_img_tokens = w * h
        packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
        packed_indexes.extend(range(curr, curr + num_img_tokens))
        curr += num_img_tokens
        _curr += num_img_tokens

        packed_text_ids.append(new_token_ids["end_of_image"])
        packed_text_indexes.append(_curr)
        packed_indexes.append(curr)
        curr += 1
        _curr += 1

        packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
        packed_seqlens.append(num_img_tokens + 2)
        newlens.append(curr_kvlen + num_img_tokens + 2)
        new_rope.append(curr_position_id + 1)

    generation_input = {
        "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
        "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
        "packed_timesteps": torch.tensor([0]),
        "packed_vae_token_indexes": torch.tensor(
            packed_vae_token_indexes, dtype=torch.long
        ),
        "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
        "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
        "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
        "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
        "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
        "packed_key_value_indexes": torch.tensor(
            packed_key_value_indexes, dtype=torch.long
        ),
        "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
    }
    return generation_input, newlens, new_rope


def forward_cache_update_vae_from_latent(model, past_key_values, packed_latent, gen_input):
    """Replica of ``Bagel.forward_cache_update_vae`` without ``vae_model.encode``.

    ``packed_latent`` is the caller-provided packed VAE latent ``(N, p*p*C)``
    (in VAE space, pre ``vae2llm``) -- e.g. produced by
    ``latent.pack_vae_latent`` from an official ``VAEEncode`` latent.
    """
    device = _model_device(model)
    gen_input = _move_tensors_to_device(gen_input, device)
    packed_latent = packed_latent.to(device)

    packed_text_ids = gen_input["packed_text_ids"]
    packed_text_indexes = gen_input["packed_text_indexes"]
    packed_position_ids = gen_input["packed_position_ids"]
    packed_seqlens = gen_input["packed_seqlens"]
    packed_indexes = gen_input["packed_indexes"]
    key_values_lens = gen_input["key_values_lens"]
    packed_key_value_indexes = gen_input["packed_key_value_indexes"]
    packed_vae_token_indexes = gen_input["packed_vae_token_indexes"]
    packed_vae_position_ids = gen_input["packed_vae_position_ids"]
    packed_timesteps = gen_input["packed_timesteps"]

    packed_text_embedding = model.language_model.model.embed_tokens(packed_text_ids)
    packed_sequence = packed_text_embedding.new_zeros(
        (sum(packed_seqlens), model.hidden_size)
    )
    packed_sequence[packed_text_indexes] = packed_text_embedding

    packed_pos_embed = model.latent_pos_embed(packed_vae_position_ids)
    packed_timestep_embeds = model.time_embedder(packed_timesteps)
    if packed_pos_embed.dtype != packed_timestep_embeds.dtype:
        packed_pos_embed = packed_pos_embed.to(packed_timestep_embeds.dtype)

    packed_latent = (
        model.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
    )
    if packed_latent.dtype != packed_sequence.dtype:
        packed_latent = packed_latent.to(packed_sequence.dtype)
    packed_sequence[packed_vae_token_indexes] = packed_latent

    extra_inputs = {}
    if model.use_moe:
        extra_inputs = {
            "mode": "gen",
            "packed_vae_token_indexes": packed_vae_token_indexes,
            "packed_text_indexes": packed_text_indexes,
        }

    output = model.language_model.forward_inference(
        packed_query_sequence=packed_sequence,
        query_lens=packed_seqlens,
        packed_query_position_ids=packed_position_ids,
        packed_query_indexes=packed_indexes,
        past_key_values=past_key_values,
        key_values_lens=key_values_lens,
        packed_key_value_indexes=packed_key_value_indexes,
        update_past_key_values=True,
        is_causal=False,
        **extra_inputs,
    )
    return output.past_key_values


def update_vae_latent_from_latent(handle, vae_latent: torch.Tensor, gen_context: Dict) -> Dict:
    """Inject an official ComfyUI ``VAEEncode`` latent as image conditioning."""
    from modeling.bagel.latent import pack_vae_latent

    model = handle["model"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]

    # ComfyUI VAE latent is [B, C, H_lat, W_lat]; recover pixel size.
    _, _, h_lat, w_lat = vae_latent.shape
    H = h_lat * 8
    W = w_lat * 8

    generation_input, kv_lens, ropes = prepare_vae_latent_from_size(
        model, kv_lens, ropes, H, W, handle["new_token_ids"]
    )
    # ComfyUI's FLUX VAE socket exposes raw autoencoder latents. BAGEL's
    # bundled AutoEncoder applies FLUX scale/shift internally, so match that
    # convention before feeding the latent to vae2llm.
    bagel_latent = comfy_raw_to_bagel_scaled(vae_latent)
    packed_latent = pack_vae_latent(handle, bagel_latent)
    past_key_values = forward_cache_update_vae_from_latent(
        model, gen_context["past_key_values"], packed_latent, generation_input
    )
    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = past_key_values
    return gen_context


def update_vit_image(handle, image, gen_context: Dict) -> Dict:
    """Add a VIT-encoded image to the context (used by understanding)."""
    model = handle["model"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]

    generation_input, kv_lens, ropes = model.prepare_vit_images(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        images=[image],
        transforms=handle["vit_transform"],
        new_token_ids=handle["new_token_ids"],
    )
    generation_input = _move_tensors_to_device(generation_input, _model_device(model))
    past_key_values = model.forward_cache_update_vit(
        gen_context["past_key_values"], **generation_input
    )
    gen_context["kv_lens"] = kv_lens
    gen_context["ropes"] = ropes
    gen_context["past_key_values"] = past_key_values
    return gen_context


def generate_latent(
    handle,
    gen_context: Dict,
    cfg_text_context: Dict,
    cfg_img_context: Dict,
    image_shape: Tuple[int, int],
    cfg_text_scale: float = 4.0,
    cfg_img_scale: float = 1.5,
    cfg_interval: Tuple[float, float] = (0.4, 1.0),
    cfg_renorm_min: float = 0.0,
    cfg_renorm_type: str = "global",
    num_timesteps: int = 50,
    timestep_shift: float = 3.0,
    pbar: Optional[object] = None,
) -> Dict:
    """Run image generation and return a ComfyUI ``LATENT`` dict."""
    from modeling.bagel.latent import unpack_generated_latent

    model = handle["model"]
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]

    generation_input = model.prepare_vae_latent(
        curr_kvlens=kv_lens,
        curr_rope=ropes,
        image_sizes=[image_shape],
        new_token_ids=handle["new_token_ids"],
    )
    generation_input = _move_tensors_to_device(generation_input, _model_device(model))

    cfg_text_pi = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_text_context["kv_lens"],
        curr_rope=cfg_text_context["ropes"],
        image_sizes=[image_shape],
    )
    cfg_text_pi = _move_tensors_to_device(cfg_text_pi, _model_device(model))
    cfg_img_pi = model.prepare_vae_latent_cfg(
        curr_kvlens=cfg_img_context["kv_lens"],
        curr_rope=cfg_img_context["ropes"],
        image_sizes=[image_shape],
    )
    cfg_img_pi = _move_tensors_to_device(cfg_img_pi, _model_device(model))

    unpacked_latent = model.generate_image(
        past_key_values=gen_context["past_key_values"],
        cfg_text_past_key_values=cfg_text_context["past_key_values"],
        cfg_img_past_key_values=cfg_img_context["past_key_values"],
        num_timesteps=num_timesteps,
        cfg_text_scale=cfg_text_scale,
        cfg_img_scale=cfg_img_scale,
        cfg_interval=cfg_interval,
        cfg_renorm_min=cfg_renorm_min,
        cfg_renorm_type=cfg_renorm_type,
        timestep_shift=timestep_shift,
        pbar=pbar,
        **generation_input,
        cfg_text_packed_position_ids=cfg_text_pi["cfg_packed_position_ids"],
        cfg_text_packed_query_indexes=cfg_text_pi["cfg_packed_query_indexes"],
        cfg_text_key_values_lens=cfg_text_pi["cfg_key_values_lens"],
        cfg_text_packed_key_value_indexes=cfg_text_pi["cfg_packed_key_value_indexes"],
        cfg_img_packed_position_ids=cfg_img_pi["cfg_packed_position_ids"],
        cfg_img_packed_query_indexes=cfg_img_pi["cfg_packed_query_indexes"],
        cfg_img_key_values_lens=cfg_img_pi["cfg_key_values_lens"],
        cfg_img_packed_key_value_indexes=cfg_img_pi["cfg_packed_key_value_indexes"],
    )

    # Denoised latent is in BAGEL-scaled packed VAE-latent space. Convert back
    # to ComfyUI raw FLUX latent space before returning it to VAEDecode.
    packed = unpacked_latent[0]
    bagel_latent = unpack_generated_latent(handle, packed, image_shape)
    comfy_latent = bagel_scaled_to_comfy_raw(bagel_latent)
    return to_comfy_latent(comfy_latent)


def generate_text(
    handle,
    gen_context: Dict,
    max_length: int = 500,
    do_sample: bool = True,
    temperature: float = 1.0,
) -> str:
    """Run text generation (understanding) and return the decoded string."""
    max_length = int(max_length)
    if max_length < 1:
        raise ValueError(
            "BAGEL text generation requires max_length >= 1; "
            "reload or update workflows created before the native thinking controls "
            "were added."
        )
    if do_sample and temperature <= 0:
        raise ValueError("BAGEL text sampling requires text_temperature > 0")

    model = handle["model"]
    gen_context = deepcopy(gen_context)
    kv_lens = gen_context["kv_lens"]
    ropes = gen_context["ropes"]

    generation_input = model.prepare_start_tokens(kv_lens, ropes, handle["new_token_ids"])
    generation_input = _move_tensors_to_device(generation_input, _model_device(model))
    unpacked_latent = model.generate_text(
        past_key_values=gen_context["past_key_values"],
        max_length=max_length,
        do_sample=do_sample,
        temperature=temperature,
        end_token_id=handle["new_token_ids"]["eos_token_id"],
        **generation_input,
    )
    output = handle["tokenizer"].decode(unpacked_latent[:, 0])
    output = output.split("<|im_end|>")[0].split("<|im_start|>")[1]
    return output
