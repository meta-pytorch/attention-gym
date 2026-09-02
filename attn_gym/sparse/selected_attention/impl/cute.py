"""CuTe DSL (SM100) backend for selected attention.

Delegates to FlashAttention-4's public ``flash_attn_func`` with
``gather_kv_indices`` for index-gather mode.  FA4 owns autograd,
compilation caching, workspace allocation, and backward orchestration.

This backend is **eager-only** — ``torch.compile`` is not supported until
FA4 exposes a compile-friendly public wrapper upstream.

Constraints
-----------
- head_dim = 512, nheads = 128, share_kv = True
- dtype = bfloat16, SM100 (compute capability 10.0)
- Attention sinks are not supported
- Requires FA4 with the -1 sentinel backward fix (commit c68c592+)
"""

from __future__ import annotations

import math

import torch
from flash_attn.cute.interface import flash_attn_func

# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def _build_unified_gather_indices(
    seq_len: int,
    sliding_window_size: int,
    kv_indices: torch.Tensor,
    index_offset: int,
    doc_ids: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build a single (batch, seq_len, padded_topk_length) int32 gather index tensor.

    Combines sliding-window positions (with optional doc-id masking) and
    offset sparse indices into one unified index array, padded to a multiple
    of 128 with -1 sentinels.

    Returns (indices, padded_length).
    """
    batch = kv_indices.shape[0]

    q_pos = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(1)
    w_off = torch.arange(sliding_window_size, device=device, dtype=torch.int32).unsqueeze(0)
    window_kv_pos = q_pos - sliding_window_size + 1 + w_off
    valid = window_kv_pos >= 0

    if doc_ids is not None:
        query_doc = doc_ids[:, :, None]
        safe_pos = window_kv_pos.clamp(0).long()
        kv_doc = doc_ids[:, safe_pos.view(-1)].view(doc_ids.shape[0], seq_len, sliding_window_size)
        same_doc = query_doc == kv_doc
        window_idxs = torch.where(
            valid.unsqueeze(0) & same_doc,
            window_kv_pos.unsqueeze(0).expand_as(same_doc).int(),
            -1,
        )
    else:
        window_idxs = torch.where(valid, window_kv_pos, -1)
        window_idxs = window_idxs.unsqueeze(0).expand(batch, -1, -1)

    offset_indices = torch.where(kv_indices >= 0, (kv_indices + index_offset).int(), -1)

    unified = torch.cat([window_idxs, offset_indices], dim=-1)
    raw_length = unified.shape[-1]
    padded_length = ((raw_length + 127) // 128) * 128
    if padded_length > raw_length:
        padding = torch.full(
            (batch, seq_len, padded_length - raw_length), -1, dtype=torch.int32, device=device
        )
        unified = torch.cat([unified, padding], dim=-1)

    return unified, padded_length


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_cute_constraints(
    query, local_kv, sparse_kv, kv_indices, attention_sink, sliding_window_size, share_kv
):
    if query.device.type != "cuda":
        raise ValueError("CuTe backend requires CUDA tensors.")
    if torch.cuda.get_device_capability(query.device) != (10, 0):
        raise ValueError("CuTe backend requires SM100.")
    if query.dtype != torch.bfloat16:
        raise TypeError("CuTe backend requires bfloat16.")
    if not share_kv:
        raise ValueError("CuTe backend requires share_kv=True.")
    if attention_sink is not None:
        raise NotImplementedError("CuTe backend does not support attention sinks.")

    _b, h, _s, d = query.shape
    if d != 512:
        raise ValueError(f"CuTe backend requires head_dim=512, got {d}.")
    if h != 128:
        raise ValueError(f"CuTe backend requires 128 query heads, got {h}.")
    if local_kv.shape[1] != 1:
        raise ValueError("CuTe backend requires KV to have 1 head.")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def selected_attention(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor | None,
    doc_ids: torch.Tensor | None,
    sliding_window_size: int,
    share_kv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTe DSL (SM100) forward+backward for selected attention.

    Eager-only — torch.compile is not supported for this backend.
    Attention sinks are not supported (assumes sink weight ≈ 0).

    Returns:
        Tuple of (output, lse) where output has shape (batch, heads, seq, head_dim)
        and lse has shape (batch, heads, seq).
    """
    _validate_cute_constraints(
        query, local_kv, sparse_kv, kv_indices, attention_sink, sliding_window_size, share_kv
    )

    _b, _h, s, d = query.shape
    device = query.device
    local_kv_len = local_kv.shape[2]

    # Build unified gather indices: sliding window + sparse, padded to 128-multiple
    gather_indices, _padded_length = _build_unified_gather_indices(
        seq_len=s,
        sliding_window_size=sliding_window_size,
        kv_indices=kv_indices,
        index_offset=local_kv_len,
        doc_ids=doc_ids,
        device=device,
    )

    # Concatenate local and sparse KV along the sequence dimension
    # local_kv: (B, 1, S, D), sparse_kv: (B, 1, X, D)
    unified_kv = torch.cat([local_kv, sparse_kv], dim=2)

    # FA4 expects BSHD layout: (batch, seqlen, nheads, hdim)
    qv_bshd = query.permute(0, 2, 1, 3)
    v_bshd = unified_kv.permute(0, 2, 1, 3)

    softmax_scale = 1.0 / math.sqrt(d)

    # Call FA4's public interface.
    # Passing k=v (same object) with hdim=512 triggers MLA mode internally:
    # FA4 moves q into qv and nulls q/k, then routes to the sparse MLA kernels.
    out, lse = flash_attn_func(
        q=qv_bshd,
        k=v_bshd,
        v=v_bshd,
        gather_kv_indices=gather_indices,
        softmax_scale=softmax_scale,
        causal=False,
        pack_gqa=True,
        return_lse=True,
    )

    # FA4's MLA path returns both tensors with sequence before heads.
    return out.permute(0, 2, 1, 3), lse.permute(0, 2, 1)
