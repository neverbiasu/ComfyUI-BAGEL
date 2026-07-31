"""FlashAttention compatibility wrapper for BAGEL packed attention.

The reference implementation uses ``flash_attn_varlen_func``.  FlashAttention
is preferred on CUDA, but it is not available on every ComfyUI platform (and
must not prevent node registration).  This module keeps the import optional
and provides a correctness-oriented SDPA fallback for the rare unsupported
environment.  The fallback is substantially slower and is not intended for
large production generations.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func
except ImportError:
    _flash_attn_varlen_func = None


_warned_fallback = False


def _as_offsets(cu_seqlens: torch.Tensor) -> list[int]:
    return [int(value) for value in cu_seqlens.detach().cpu().tolist()]


def _repeat_kv_heads(tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
    """Expand grouped-query K/V heads for SDPA versions without GQA support."""
    heads = tensor.shape[1]
    if heads == target_heads:
        return tensor
    if target_heads % heads:
        raise ValueError(
            f"Cannot expand {heads} KV heads to {target_heads} query heads in BAGEL attention"
        )
    return tensor.repeat_interleave(target_heads // heads, dim=1)


def _sdpa_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    q_offsets = _as_offsets(cu_seqlens_q)
    k_offsets = _as_offsets(cu_seqlens_k)
    if len(q_offsets) != len(k_offsets):
        raise ValueError("BAGEL packed attention has mismatched query/key sequence counts")

    outputs = []
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]
    ):
        q_seq = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_seq = _repeat_kv_heads(k[k_start:k_end].transpose(0, 1).unsqueeze(0), q_seq.shape[1])
        v_seq = _repeat_kv_heads(v[k_start:k_end].transpose(0, 1).unsqueeze(0), q_seq.shape[1])
        attn_mask = None
        if causal:
            # FlashAttention aligns a short query sequence to the *end* of its
            # key sequence when a KV cache is present. SDPA's is_causal=True
            # uses an upper-left mask instead, so build the aligned mask here.
            q_positions = torch.arange(q_seq.shape[-2], device=q_seq.device)
            q_positions += k_seq.shape[-2] - q_seq.shape[-2]
            k_positions = torch.arange(k_seq.shape[-2], device=k_seq.device)
            attn_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        output = F.scaled_dot_product_attention(
            q_seq, k_seq, v_seq, attn_mask=attn_mask, is_causal=False
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Use FlashAttention when installed, otherwise emulate its packed API."""
    if _flash_attn_varlen_func is not None:
        return _flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            **kwargs,
        )

    global _warned_fallback
    if not _warned_fallback:
        logging.warning(
            "[BAGEL] flash-attn is unavailable; using a slow PyTorch SDPA fallback. "
            "Install flash-attn for supported CUDA hardware to obtain reference performance."
        )
        _warned_fallback = True
    return _sdpa_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal=causal)
