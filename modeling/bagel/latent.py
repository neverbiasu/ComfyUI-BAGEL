"""Native BAGEL LATENT I/O and VAE decoupling.

BAGEL's coupled VAE is the FLUX autoencoder that the official FLUX VAE nodes
already provide. These helpers expose the ComfyUI LATENT boundary so native
workflows encode/decode through the official FLUX VAE and project between the
VAE latent space and the model's internal (LLM) latent space via the coupled
``vae2llm`` / ``llm2vae`` projections.

Packing convention (mirrors ``Bagel.forward_cache_update_vae`` /
``InterleaveInferencer.decode_image``):

* A BAGEL/FLUX VAE latent has shape ``[C, H_lat, W_lat]`` with
  ``H_lat = H_px // 8`` and ``C = 16``. BAGEL's bundled autoencoder applies
  FLUX scale/shift inside ``encode``/``decode``.
* ComfyUI's standalone FLUX VAE node exposes raw autoencoder latents at the
  ``LATENT`` socket, so node boundaries must convert raw Comfy latents to/from
  BAGEL's scaled latent convention.
* It is patchified into ``(h, w, p, p, C)`` where ``h = H_lat // p`` and
  ``p = latent_patch_size`` (2), then flattened to a packed tensor
  ``(h*w, p*p*C)`` that the model diffuses in.
* ``vae2llm`` projects the packed VAE latent -> LLM hidden space;
  ``llm2vae`` projects LLM hidden -> packed VAE latent.
* Unpatchifying a packed latent (einsum ``hwpqc -> chpwq``) recovers the
  standard ``[C, H_lat, W_lat]`` VAE latent that ``vae.decode`` consumes.
"""

import argparse
import json
import sys

import torch

# FLUX VAE spatial downsampling and latent channel count.
LATENT_DOWNSAMPLE = 8
VAE_LATENT_CHANNELS = 16
# BAGEL packs 2x2 latent patches.
LATENT_PATCH_SIZE = 2
# FLUX AE latent normalization used inside BAGEL's bundled AutoEncoder.
FLUX_SCALE_FACTOR = 0.3611
FLUX_SHIFT_FACTOR = 0.1159


def to_comfy_latent(tensor: torch.Tensor) -> dict:
    """Wrap a ``[B, C, H, W]`` tensor as a ComfyUI LATENT dict."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return {"samples": tensor}


def from_comfy_latent(latent: dict) -> torch.Tensor:
    """Extract the ``[B, C, H, W]`` tensor from a ComfyUI LATENT dict."""
    return latent["samples"]


def comfy_raw_to_bagel_scaled(latent: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI FLUX VAE raw latents to BAGEL's scaled latent space."""
    return FLUX_SCALE_FACTOR * (latent - FLUX_SHIFT_FACTOR)


def bagel_scaled_to_comfy_raw(latent: torch.Tensor) -> torch.Tensor:
    """Convert BAGEL scaled FLUX latents to ComfyUI FLUX VAE raw latents."""
    return latent / FLUX_SCALE_FACTOR + FLUX_SHIFT_FACTOR


def patchify(vae_latent: torch.Tensor, patch_size: int = LATENT_PATCH_SIZE) -> torch.Tensor:
    """Pack a standard VAE latent ``[C, H_lat, W_lat]`` into ``(N, p*p*C)``.

    ``H_lat``/``W_lat`` must be divisible by ``patch_size``.
    """
    c, h_lat, w_lat = vae_latent.shape[-3:]
    h, w = h_lat // patch_size, w_lat // patch_size
    if h == 0 or w == 0:
        raise ValueError(
            f"latent spatial size {(h_lat, w_lat)} is smaller than patch size "
            f"{patch_size}"
        )
    # Match Bagel.forward_cache_update_vae: the upstream VAE transform targets
    # multiples of 16, but an arbitrary official ComfyUI VAEEncode input may
    # yield an odd latent edge. Drop only the incomplete right/bottom patch.
    vae_latent = vae_latent[:, : h * patch_size, : w * patch_size]
    packed = vae_latent.reshape(c, h, patch_size, w, patch_size)
    packed = torch.einsum("chpwq->hwpqc", packed)
    return packed.reshape(-1, patch_size * patch_size * c)


def unpatchify(packed: torch.Tensor, h: int, w: int, patch_size: int = LATENT_PATCH_SIZE,
               channel: int = VAE_LATENT_CHANNELS) -> torch.Tensor:
    """Inverse of :func:`patchify`: ``(N, p*p*C)`` -> ``[C, h*p, w*p]``."""
    n = h * w
    if packed.shape[0] != n:
        raise ValueError(
            f"packed latent has {packed.shape[0]} tokens but h*w={n} (h={h}, w={w})"
        )
    latent = packed.reshape(h, w, patch_size, patch_size, channel)
    latent = torch.einsum("hwpqc->chpwq", latent)
    return latent.reshape(channel, h * patch_size, w * patch_size)


def latent_shape_from_pixels(height: int, width: int, downsample: int = LATENT_DOWNSAMPLE) -> tuple:
    """Map pixel dimensions to VAE latent grid dimensions."""
    return (max(1, height // downsample), max(1, width // downsample))


def patch_grid_from_pixels(height: int, width: int, downsample: int = LATENT_DOWNSAMPLE,
                           patch_size: int = LATENT_PATCH_SIZE) -> tuple:
    """Return the packed grid ``(h, w)`` for a pixel-sized image."""
    h_lat, w_lat = latent_shape_from_pixels(height, width, downsample)
    return (h_lat // patch_size, w_lat // patch_size)


def vae_to_llm_latent(handle, vae_latent: torch.Tensor) -> torch.Tensor:
    """Project a standard VAE latent into the model's internal LLM latent space.

    Applies ``vae2llm`` (VAE -> LLM), as defined in ``Bagel.__init__``.
    """
    return handle["model"].vae2llm(vae_latent)


def llm_to_vae_latent(handle, llm_latent: torch.Tensor) -> torch.Tensor:
    """Project the model's internal LLM latent space back to VAE-latent space.

    Applies ``llm2vae`` (LLM -> VAE), as defined in ``Bagel.__init__``.
    """
    return handle["model"].llm2vae(llm_latent)


def pack_vae_latent(handle, vae_latent: torch.Tensor) -> torch.Tensor:
    """Patchify a standard VAE latent to the model's packed representation.

    ``vae_latent`` is a BAGEL-scaled VAE latent ``[B, C, H_lat, W_lat]`` (or
    ``[C, H_lat, W_lat]``). Returns ``(N, p*p*C)`` packed latent in *VAE space*
    (pre ``vae2llm``), as expected by ``forward_cache_update_vae_from_latent``.
    """
    if vae_latent.dim() == 4:
        vae_latent = vae_latent[0]
    return patchify(vae_latent, patch_size=handle["model"].latent_patch_size)


def unpack_generated_latent(handle, packed_latent: torch.Tensor, image_shape: tuple) -> torch.Tensor:
    """Generation output: packed latent -> standard VAE latent.

    ``generate_image`` returns the denoised latent in BAGEL-scaled *packed
    VAE-latent space* (it is mapped LLM<->VAE internally by ``_forward_flow``),
    so no extra ``llm2vae`` is needed. ``packed_latent`` has shape
    ``(N, p*p*C)`` and is unpatchified directly to ``[C, H_lat, W_lat]``.
    """
    # Bagel.latent_downsample already includes the VAE downsample and latent
    # patch size (vae_config.downsample * latent_patch_size), so this maps
    # pixels directly to the packed latent grid used by prepare_vae_latent().
    h = image_shape[0] // handle["model"].latent_downsample
    w = image_shape[1] // handle["model"].latent_downsample
    return unpatchify(
        packed_latent, h, w,
        patch_size=handle["model"].latent_patch_size,
        channel=handle["model"].latent_channel,
    )


def _self_check():
    """CPU round-trip of patchify/unpatchify + projection direction.

    Uses a tiny identity linear so no real model weights are required. Exits
    non-zero on mismatch. Must not import ``nodes`` or any ComfyUI module.
    """
    torch.manual_seed(0)
    p = LATENT_PATCH_SIZE
    c = VAE_LATENT_CHANNELS
    h, w = 4, 6  # packed grid
    h_lat, w_lat = h * p, w * p
    vae = torch.randn(c, h_lat, w_lat)

    packed = patchify(vae, p)
    assert packed.shape == (h * w, p * p * c), packed.shape
    restored = unpatchify(packed, h, w, p, c)
    assert torch.allclose(restored, vae, atol=1e-6), (restored - vae).abs().max()

    # Identity projection must preserve the packed shape and round-trip.
    class _Identity(torch.nn.Linear):
        def __init__(self, n):
            super().__init__(n, n, bias=False)
            torch.nn.init.eye_(self.weight)

    proj = _Identity(p * p * c)
    out = proj(packed)
    assert out.shape == packed.shape
    assert torch.allclose(out, packed, atol=1e-6)

    report = {
        "ok": True,
        "patch_size": p,
        "channel": c,
        "packed_grid": [h, w],
        "vae_shape": [c, h_lat, w_lat],
        "packed_shape": list(packed.shape),
    }
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true",
                        help="run CPU patchify/unpatchify + projection self-check")
    args = parser.parse_args()
    if args.self_check:
        sys.exit(_self_check())
    parser.print_help()
    sys.exit(1)
