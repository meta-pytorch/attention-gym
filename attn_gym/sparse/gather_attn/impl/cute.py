"""CuTe DSL (SM100/SM103) backend for gather attention.

Delegates to FlashAttention-4's public ``flash_attn_func`` with
``gather_kv_indices`` for index-gather mode.  FA4 owns autograd,
compilation caching, workspace allocation, and backward orchestration.

This backend is **eager-only** — ``torch.compile`` is not supported until
FA4 exposes a compile-friendly public wrapper upstream.

Constraints
-----------
- head_dim = 512, 1 <= nheads <= 128, share_kv = True (fewer than 128 heads are
  zero-padded to FA4's 64/128-head tiles in-kernel via TMA out-of-bounds)
- dtype = bfloat16, SM100 or SM103 (compute capability 10.0 or 10.3)
- Requires FA4 4.0.0b32+ for sparse MLA attention sinks and, with fewer than 128
  heads, sparse-MLA head padding
"""

from __future__ import annotations

import inspect
import warnings
from functools import cache

import torch
import torch.nn.functional as F

from attn_gym.utils import round_up

# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def _build_gather_indices(
    kv_indices: torch.Tensor,
    doc_ids: torch.Tensor | None,
    sliding_window_size: int,
    local_kv_len: int,
) -> torch.Tensor:
    """Build FA4's (batch, seq_len, padded_topk) int32 gather indices over [local_kv; sparse_kv].

    Each row lists the query's sliding-window positions (masked to its document) followed by
    its sparse selections offset past the local KV, padded to a multiple of 128 with -1.
    """
    batch, seq_len, _ = kv_indices.shape
    device = kv_indices.device
    q_pos = torch.arange(seq_len, device=device, dtype=torch.int32).unsqueeze(1)
    w_off = torch.arange(sliding_window_size, device=device, dtype=torch.int32)
    window_kv_pos = q_pos - sliding_window_size + 1 + w_off
    keep = window_kv_pos >= 0
    if doc_ids is not None:
        keep = keep & (doc_ids[:, window_kv_pos.clamp(min=0)] == doc_ids[:, :, None])
    window_idxs = torch.where(keep, window_kv_pos, -1).expand(batch, -1, -1)

    sparse_idxs = torch.where(kv_indices >= 0, (kv_indices + local_kv_len).int(), -1)
    unified = torch.cat([window_idxs, sparse_idxs], dim=-1)
    num_slots = unified.shape[-1]
    # FA4's gather prologue needs a physical tile even when the attention set is empty.
    return F.pad(unified, (0, round_up(max(num_slots, 1), 128) - num_slots), value=-1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))


def _constraint_violation(query: torch.Tensor, share_kv: bool) -> Exception | None:
    """Return the error for metadata this backend cannot run, or None when it qualifies."""
    _b, h, _s, d = query.shape
    if query.device.type != "cuda":
        return ValueError("CuTe backend requires CUDA tensors.")
    if torch.cuda.get_device_capability(query.device) not in SUPPORTED_CAPABILITIES:
        return ValueError("CuTe backend requires SM100 or SM103.")
    if query.dtype != torch.bfloat16:
        return TypeError("CuTe backend requires bfloat16.")
    if not share_kv:
        return ValueError("CuTe backend requires share_kv=True.")
    if d != 512:
        return ValueError(f"CuTe backend requires head_dim=512, got {d}.")
    if not 0 < h <= 128:
        return ValueError(f"CuTe backend requires at most 128 query heads, got {h}.")
    return None


@cache
def _fa4_available(with_sink: bool, *, padded_heads: bool = False) -> bool:
    """Probe the optional dependency once, only after tensor metadata qualifies."""
    try:
        from flash_attn.cute.interface import flash_attn_func  # noqa: F401

        if padded_heads:
            # Added with FA4's arbitrary-head sparse MLA kernels; older FA4 requires H=128.
            from flash_attn.cute.pack_gqa import sparse_mla_qhead_tile  # noqa: F401

        if with_sink:
            from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100

            return (
                "learnable_sink"
                in inspect.signature(FlashAttentionMLAForwardSm100.__call__).parameters
            )
    except ImportError:
        return False
    return True


def is_supported(query, attention_sink, share_kv) -> bool:
    """Check metadata and optional FA4 features without launching a kernel."""
    if _constraint_violation(query, share_kv) is not None:
        return False
    # FA4 requires a unit leading stride. Its contiguous repair skips singleton views,
    # which PyTorch considers contiguous even when their stride is not one.
    if (
        attention_sink is not None
        and attention_sink.numel() == 1
        and attention_sink.stride(0) != 1
    ):
        return False
    return _fa4_available(attention_sink is not None, padded_heads=query.shape[1] != 128)


def _check_backward_mode(grad: torch.Tensor) -> torch.Tensor:
    if torch.are_deterministic_algorithms_enabled():
        message = (
            "CuTe gather attention does not support deterministic backward; "
            "use kernel_options={'backend': 'triton'} for the forward call."
        )
        if torch.is_deterministic_algorithms_warn_only_enabled():
            warnings.warn(message, stacklevel=2)
        else:
            raise RuntimeError(message)
    return grad


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def gather_attn(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor | None,
    doc_ids: torch.Tensor | None,
    sliding_window_size: int,
    share_kv: bool = True,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CuTe DSL (SM100/SM103) forward+backward for gather attention.

    Eager-only — torch.compile is not supported for this backend.
    Optional per-head attention sinks are forwarded to FA4, which owns their gradients.

    Returns:
        Tuple of (output, lse) where output has shape (batch, heads, seq, head_dim)
        and lse has shape (batch, heads, seq).
    """
    if (error := _constraint_violation(query, share_kv)) is not None:
        raise error
    from flash_attn.cute.interface import flash_attn_func

    gather_indices = _build_gather_indices(
        kv_indices, doc_ids, sliding_window_size, local_kv_len=local_kv.shape[2]
    )
    # FA4 takes BSHD. Passing k=v (the same object) with hdim=512 selects MLA mode:
    # FA4 moves q into qv and routes to the sparse MLA kernels.
    kv_bshd = torch.cat([local_kv, sparse_kv], dim=2).permute(0, 2, 1, 3)
    out, lse = flash_attn_func(
        q=query.permute(0, 2, 1, 3),
        k=kv_bshd,
        v=kv_bshd,
        gather_kv_indices=gather_indices,
        softmax_scale=scale,
        learnable_sink=attention_sink,
        causal=False,
        pack_gqa=True,
        return_lse=True,
    )

    # Determinism can be enabled after forward; honor its strict/warn-only setting.
    if out.requires_grad:
        out.register_hook(_check_backward_mode)

    # Sparse MLA ignores dLSE. Match Triton's nondifferentiable auxiliary contract.
    return out.permute(0, 2, 1, 3), lse.detach().permute(0, 2, 1)
