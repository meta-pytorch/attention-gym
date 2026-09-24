"""CuTe DSL (SM100/SM103) backend for gather attention.

Delegates to FlashAttention-4's public dense/varlen entry points with
``gather_kv_indices`` for index-gather mode. FA4 owns attention autograd,
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
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
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
    from flash_attn.cute.interface import flash_attn_func, flash_attn_varlen_func

    from .indices import build_gather_indices

    gather_indices = build_gather_indices(
        kv_indices,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        sparse_kv_len=sparse_kv.shape[2],
    )
    options = {
        "softmax_scale": scale,
        "learnable_sink": attention_sink,
        "causal": False,
        "pack_gqa": True,
        "return_lse": True,
    }
    # Passing k=v (the same object) with hdim=512 selects FA4's sparse MLA path.
    if cu_seqlens is None:
        kv = torch.cat([local_kv, sparse_kv], dim=2).permute(0, 2, 1, 3)
        out, lse = flash_attn_func(
            q=query.permute(0, 2, 1, 3),
            k=kv,
            v=kv,
            gather_kv_indices=gather_indices,
            **options,
        )
    else:
        from .packed_kv import pack_kv

        kv, cu_q, cu_kv = pack_kv(local_kv, sparse_kv, cu_seqlens, cu_seqlens_k)
        out, lse = flash_attn_varlen_func(
            q=query[0].transpose(0, 1),
            k=kv,
            v=kv,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            gather_kv_indices=gather_indices[0],
            **options,
        )
        out, lse = out.unsqueeze(0), lse.unsqueeze(0)

    # Determinism can be enabled after forward; honor its strict/warn-only setting.
    if out.requires_grad:
        out.register_hook(_check_backward_mode)

    # Sparse MLA ignores dLSE. Match Triton's nondifferentiable auxiliary contract.
    return out.permute(0, 2, 1, 3), lse.detach().permute(0, 2, 1)
