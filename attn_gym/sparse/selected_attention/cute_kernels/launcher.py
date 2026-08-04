"""Compile and launch the FA4 MLA kernel for selected attention (SM100).

Uses the upstream FlashAttentionMLAForwardSm100 with is_topk_gather=True.
No sink correction is fused — caller handles it externally if needed.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack
from flash_attn.cute.flash_fwd_mla_sm100 import FlashAttentionMLAForwardSm100

try:
    import cuda.bindings.driver as cuda
except ImportError:
    from cuda import cuda


_COMPILE_CACHE_MAXSIZE = 16
_compile_cache: OrderedDict[tuple[object, ...], object] = OrderedDict()


def _tensor_sig(t: torch.Tensor | None) -> tuple[object, ...] | None:
    if t is None:
        return None
    return (t.dtype, tuple(t.shape), tuple(t.stride()))


def _cache_get(key: tuple[object, ...]) -> object | None:
    compiled = _compile_cache.get(key)
    if compiled is not None:
        _compile_cache.move_to_end(key)
    return compiled


def _cache_put(key: tuple[object, ...], compiled: object) -> None:
    _compile_cache[key] = compiled
    _compile_cache.move_to_end(key)
    if len(_compile_cache) > _COMPILE_CACHE_MAXSIZE:
        _compile_cache.popitem(last=False)


def _compile(
    query: torch.Tensor,
    value: torch.Tensor,
    gather: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
) -> object:
    """Compile the FA4 MLA kernel (cached by tensor layout signature)."""
    key = (
        "selected_fa4",
        _tensor_sig(query),
        _tensor_sig(value),
        _tensor_sig(gather),
        _tensor_sig(output),
        _tensor_sig(lse),
    )
    compiled = _cache_get(key)
    if compiled is not None:
        return compiled

    heads = query.shape[2]
    topk_length = gather.shape[-1]

    kernel = FlashAttentionMLAForwardSm100(
        is_causal=False,
        use_cpasync_load_KV=True,
        topk_length=topk_length,
        is_topk_gather=True,
        pack_gqa=True,
        qhead_per_kvhead=heads,
        nheads_kv=1,
        is_varlen_q=False,
        disable_bitmask=False,
        has_qk=False,
    )

    mQv = from_dlpack(query, assumed_align=16).mark_layout_dynamic(leading_dim=query.ndim - 1)
    mV = from_dlpack(value, assumed_align=16).mark_layout_dynamic(leading_dim=value.ndim - 1)
    mO = from_dlpack(output, assumed_align=16).mark_layout_dynamic(leading_dim=output.ndim - 1)
    mLSE = from_dlpack(lse, assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 1)
    mGather = from_dlpack(gather, assumed_align=16).mark_layout_dynamic(
        leading_dim=gather.ndim - 1
    )

    stream = cuda.CUstream(torch.cuda.current_stream(query.device).cuda_stream)

    compiled = cute.compile(
        kernel,
        None,  # mQ
        mQv,
        None,  # mK
        mV,
        mO,
        mLSE,
        1.0,  # softmax_scale (placeholder)
        None,  # mP
        None,  # mRowMax
        mIndexTopk=mGather,
        stream=stream,
    )
    _cache_put(key, compiled)
    return compiled


def selected_attention_forward(
    query: torch.Tensor,
    value: torch.Tensor,
    gather: torch.Tensor,
    sink: torch.Tensor | None = None,
    *,
    output: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the FA4 MLA kernel for selected attention.

    Args:
        query:  (B, S, H, D) bfloat16. H=128, D=512.
        value:  (B, N, 1, D) bfloat16. Unified KV buffer.
        gather: (B, S, topk_padded) int32, contiguous. Padded to mult of 128.
        sink:   Ignored (kept for API compat). Sink=0 assumed.
        output: Pre-allocated output buffer or None.
        lse:    Pre-allocated LSE buffer or None.

    Returns:
        (output, lse) — output is (B, S, H, D) bfloat16, lse is (B, S, H) float32.
    """
    if query.ndim != 4 or value.ndim != 4 or gather.ndim != 3:
        raise ValueError("query must be BSHD, value BNKD, gather BSK.")
    batch, seq_len, heads, dim = query.shape
    if heads != 128 or dim != 512:
        raise ValueError("Requires H=128, D=512.")
    if gather.shape[-1] % 128 != 0:
        raise ValueError("gather last dim must be a multiple of 128.")
    if gather.dtype != torch.int32:
        raise ValueError("gather must be int32.")

    if output is None:
        output = torch.empty(batch, seq_len, heads, dim, dtype=query.dtype, device=query.device)
    if lse is None:
        lse = torch.empty(batch, seq_len, heads, device=query.device, dtype=torch.float32)

    compiled = _compile(query, value, gather, output, lse)

    stream = cuda.CUstream(torch.cuda.current_stream(query.device).cuda_stream)

    mQv = from_dlpack(query, assumed_align=16).mark_layout_dynamic(leading_dim=query.ndim - 1)
    mV = from_dlpack(value, assumed_align=16).mark_layout_dynamic(leading_dim=value.ndim - 1)
    mO = from_dlpack(output, assumed_align=16).mark_layout_dynamic(leading_dim=output.ndim - 1)
    mLSE = from_dlpack(lse, assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 1)
    mGather = from_dlpack(gather, assumed_align=16).mark_layout_dynamic(
        leading_dim=gather.ndim - 1
    )

    compiled(
        None,  # mQ
        mQv,
        None,  # mK
        mV,
        mO,
        mLSE,
        1.0 / math.sqrt(dim),
        None,  # mP
        None,  # mRowMax
        mIndexTopk=mGather,
        stream=stream,
    )
    return output, lse
