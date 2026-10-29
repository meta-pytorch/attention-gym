"""Bounded-workspace indexer for SM100/SM103 scoring followed by exact Top-K.

A slab contains at most 1024 query rows and 32 MiB of FP32 scores, independent
of batch size and sequence length. Consecutive launches reuse that per-call
allocation on the current stream; the only retained result is the INT32 index
array. Slabs contain whole query pairs, including a masked second row for odd T.
The registered operator calls this launcher; there is no backend fallback.
"""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from attn_gym._backends.cute import (
    TMA_ALIGNMENT_BYTES,
    compile_tvm_ffi,
    get_device_properties,
    jit_cache,
    make_fake_strided_tensor,
    tensor_supports_contiguous_dim,
    tensor_supports_tma,
)
from attn_gym._backends.cute.target import (
    detect_compile_target,
    get_compile_target,
    set_compile_target,
)
from attn_gym._backends.cute.utils import initialized_cuda_device, requires_int64_abi
from attn_gym.utils import cdiv

from ..validation import validate_precision

if TYPE_CHECKING:
    from cutlass.cute.runtime import _FakeTensor

_HEAD_DIM_GRANULARITY = 16
_MAX_SEQUENCE = 1 << 20
# Limit both the row count (linear storage as T grows) and absolute scratch bytes.
_MAX_SCORE_PAIRS = 512
_SCORE_WORKSPACE_BYTES = 32 * 1024 * 1024


def score_workspace_pairs(batch: int, tokens: int, candidates: int) -> int:
    """Size a slab for positive B/T/S; return zero if a query pair cannot fit the budget."""
    return min(
        _MAX_SCORE_PAIRS, batch * cdiv(tokens, 2), _SCORE_WORKSPACE_BYTES // (8 * candidates)
    )


def _fake_score_operands(
    io_dtype: type,
    weight_dtype: type,
    heads: int,
    head_dim: int,
    use_int64_offsets: bool,
    contiguous_weight_heads: bool,
) -> tuple["_FakeTensor", "_FakeTensor", "_FakeTensor", "_FakeTensor"]:
    """Share symbolic B/T/S and stride contracts across the two scoring ABIs."""
    import cutlass
    from cutlass import cute

    sym = cute.sym_int64 if use_int64_offsets else cute.sym_int
    batch, tokens, candidates, pairs = sym(), sym(), sym(), sym()
    q = make_fake_strided_tensor(
        io_dtype,
        (batch, tokens, heads, head_dim),
        stride_divisibility=TMA_ALIGNMENT_BYTES // (io_dtype.width // 8),
        use_int64_strides=use_int64_offsets,
    )
    k = make_fake_strided_tensor(
        io_dtype,
        (batch, candidates, head_dim),
        stride_divisibility=TMA_ALIGNMENT_BYTES // (io_dtype.width // 8),
        use_int64_strides=use_int64_offsets,
    )
    weights = make_fake_strided_tensor(
        weight_dtype,
        (batch, tokens, heads),
        contiguous_dim=-1 if contiguous_weight_heads else None,
        use_int64_strides=use_int64_offsets,
    )
    scores = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (pairs, 2, candidates),
        stride_order=(2, 1, 0),
        assumed_align=TMA_ALIGNMENT_BYTES,
        use_32bit_stride=not use_int64_offsets,
    )
    return q, k, weights, scores


@jit_cache
def _compile_scores(
    dtype: torch.dtype,
    heads: int,
    head_dim: int,
    causal: bool,
    compress_ratio: int,
    use_int64_offsets: bool,
    contiguous_weight_heads: bool,
    has_candidate_bounds: bool = False,
) -> Callable[..., None]:
    """Compile symbolic B/T/S/strides, specializing only an available unit weight-head stride.

    ``has_candidate_bounds`` scores only each query's packed ``candidate_bounds`` interval.
    """
    import cutlass

    from .cute_score import IndexerScoreKernel
    from .cute_score_generic import IndexerGenericScoreKernel

    operation = (
        IndexerScoreKernel if heads in (32, 64) and head_dim == 128 else IndexerGenericScoreKernel
    )(
        heads,
        head_dim,
        causal,
        use_int64_offsets,
        compress_ratio=compress_ratio,
        contiguous_weight_heads=contiguous_weight_heads,
    )
    io_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
    q, k, weights, scores = _fake_score_operands(
        io_dtype, io_dtype, heads, head_dim, use_int64_offsets, contiguous_weight_heads
    )
    args = (q, k, weights, scores, integer(0), cutlass.Float32(1.0))
    if has_candidate_bounds:
        return compile_tvm_ffi(
            operation.with_candidate_bounds,
            *args,
            _fake_candidate_bounds(q.shape[1], use_int64_offsets),
            name=f"{operation.get_name()}_packed",
        )
    return compile_tvm_ffi(operation, *args)


def _fake_candidate_bounds(tokens, use_int64_offsets: bool):
    """Fake contiguous INT32 [T, 2] packed candidate intervals."""
    import cutlass
    from cutlass import cute

    return cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (tokens, 2),
        stride_order=(1, 0),
        assumed_align=4,
        use_32bit_stride=not use_int64_offsets,
    )


@jit_cache
def _compile_mxfp8_scores(
    weights_dtype: torch.dtype,
    heads: int,
    head_dim: int,
    causal: bool,
    compress_ratio: int,
    use_int64_offsets: bool,
    contiguous_weight_heads: bool,
    contiguous_q_scales: bool,
    contiguous_k_scales: bool,
) -> Callable[..., None]:
    """Compile MXFP8 storage with logical E8M0 scales for 32-element groups."""
    import cutlass

    from .cute_score_mxfp8 import IndexerMXFP8ScoreKernel

    weight_type = cutlass.BFloat16 if weights_dtype == torch.bfloat16 else cutlass.Float32
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
    q, k, weights, scores = _fake_score_operands(
        cutlass.Float8E4M3FN,
        weight_type,
        heads,
        head_dim,
        use_int64_offsets,
        contiguous_weight_heads,
    )
    q_scale = make_fake_strided_tensor(
        cutlass.Float8E8M0FNU,
        (*q.shape[:-1], head_dim // 32),
        contiguous_dim=-1 if contiguous_q_scales else None,
        stride_divisibility=4 if contiguous_q_scales else 1,
        use_int64_strides=use_int64_offsets,
    )
    k_scale = make_fake_strided_tensor(
        cutlass.Float8E8M0FNU,
        (*k.shape[:-1], head_dim // 32),
        contiguous_dim=-1 if contiguous_k_scales else None,
        stride_divisibility=4 if contiguous_k_scales else 1,
        use_int64_strides=use_int64_offsets,
    )
    return compile_tvm_ffi(
        IndexerMXFP8ScoreKernel(
            heads,
            head_dim,
            causal,
            use_int64_offsets,
            compress_ratio=compress_ratio,
            contiguous_weight_heads=contiguous_weight_heads,
            contiguous_q_scales=contiguous_q_scales,
            contiguous_k_scales=contiguous_k_scales,
        ),
        q,
        k,
        weights,
        q_scale,
        k_scale,
        scores,
        integer(0),
        cutlass.Float32(1.0),
    )


@jit_cache
def _compile_topk(
    topk: int,
    causal: bool,
    compress_ratio: int,
    use_int64_offsets: bool,
    deterministic: bool,
    has_candidate_bounds: bool = False,
    selector: str = "default",
) -> Callable[..., None]:
    """Compile indices-only selection with symbolic B, T, S and slab capacity."""
    import cutlass
    from cutlass import cute

    from .cute_topk import IndexerTopKKernel

    sym = cute.sym_int64 if use_int64_offsets else cute.sym_int
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
    batch, tokens, candidates, pairs = sym(), sym(), sym(), sym()
    scores = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (pairs, 2, candidates),
        stride_order=(2, 1, 0),
        assumed_align=TMA_ALIGNMENT_BYTES,
        use_32bit_stride=not use_int64_offsets,
    )
    output = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (batch, tokens, topk),
        stride_order=(2, 1, 0),
        assumed_align=TMA_ALIGNMENT_BYTES,
        use_32bit_stride=not use_int64_offsets,
    )
    assert selector in ("default", "gvr2"), selector
    if selector == "gvr2":
        from .cute_topk_gvr2 import IndexerGVR2TopKKernel

        operation = IndexerGVR2TopKKernel(
            topk, causal, use_int64_offsets, compress_ratio=compress_ratio
        )
    else:
        operation = IndexerTopKKernel(
            topk,
            causal,
            use_int64_offsets,
            compress_ratio=compress_ratio,
            deterministic=deterministic,
        )
    if has_candidate_bounds:
        return compile_tvm_ffi(
            operation.with_candidate_bounds,
            scores,
            output,
            integer(0),
            _fake_candidate_bounds(tokens, use_int64_offsets),
            name=f"{operation.get_name()}_packed",
        )
    return compile_tvm_ffi(operation, scores, output, integer(0))


def _validate(q: torch.Tensor, k: torch.Tensor) -> None:
    """Validate SM100/SM103 kernel limits and Q/K base alignment.

    Ranks, shape agreement, devices, dtype agreement, and ``topk``/``compress_ratio``
    are checked by the public ``lightning_indexer`` API.
    """
    _, queries, heads, head_dim = q.shape
    if queries > _MAX_SEQUENCE:
        raise ValueError(f"sequence length must be in [1, {_MAX_SEQUENCE}], got {queries}")
    if heads % 2:
        raise ValueError(f"number of heads must be a positive multiple of 2, got {heads}")
    if head_dim % _HEAD_DIM_GRANULARITY:
        raise ValueError(
            "head dimension must be positive and divisible by "
            f"{_HEAD_DIM_GRANULARITY}, got {head_dim}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(f"q must be float16, bfloat16 or float8_e4m3fn, got {q.dtype}")
    if q.dtype == torch.float8_e4m3fn and (heads not in (32, 64) or head_dim != 128):
        raise ValueError("MXFP8 CuTe indexing requires H in {32, 64} and D=128")
    # Singleton strides do not address another slice; TVM-FFI normalizes them for TMA.
    if any(tensor.numel() and not tensor_supports_tma(tensor.squeeze()) for tensor in (q, k)):
        raise ValueError(
            f"q and k require unit last strides and {TMA_ALIGNMENT_BYTES}-byte aligned "
            "bases and non-singleton outer strides"
        )
    properties = get_device_properties(q.device)
    if (properties.major, properties.minor) not in ((10, 0), (10, 3)):
        raise RuntimeError("this tcgen05 kernel requires an SM100 or SM103 GPU")


def resolve_selector(selector: str, topk: int, *, deterministic: bool, packed: bool) -> str:
    """Use GVR2 below its native K limit, except for deterministic or packed calls.

    GVR2 cannot provide CuTe's deterministic ordering or packed candidate bounds.
    See the ``auto`` threshold sweep in docs/indexer_performance.md.
    """
    if selector not in ("auto", "default", "gvr2"):
        raise ValueError(f"unknown indexer selector {selector!r}")
    if selector == "gvr2" and (deterministic or packed):
        raise ValueError("CuTe GVR2 does not support deterministic mode or packed cu_seqlens")
    if selector == "auto":
        if deterministic or packed:
            return "default"
        from .cute_topk_gvr2 import NATIVE_TOPK_LIMIT

        return "gvr2" if topk < NATIVE_TOPK_LIMIT else "default"
    return selector


def launch(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool = False,
    compress_ratio: int = 1,
    deterministic: bool = False,
    candidate_bounds: torch.Tensor | None = None,
    selector: str = "default",
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute weighted-ReLU Top-K with bounded per-call score storage.

    ``deterministic`` returns ascending indices with exact ties resolved to the
    lowest indices; otherwise order and tie choice depend on thread timing.
    Packed ``candidate_bounds[T, 2]`` gives each query's valid global interval; both
    kernels read it in their prologues, scoring and selecting only that interval, and
    returned indices are relative to its start. Bounds must describe complete documents
    with per-document floor compression, not arbitrary cached queries.
    """
    import cutlass

    scaled_fp8 = q.dtype == torch.float8_e4m3fn
    if scaled_fp8 and candidate_bounds is not None:
        raise ValueError("MXFP8 indexing does not support packed cu_seqlens.")
    selector = resolve_selector(
        selector, topk, deterministic=deterministic, packed=candidate_bounds is not None
    )
    validate_precision(q, k, weights, q_scale, k_scale)
    _validate(q, k)
    batch, tokens, heads, head_dim = q.shape
    candidates = k.shape[1]
    output = torch.empty((batch, tokens, topk), dtype=torch.int32, device=q.device)
    if topk == 0:
        return output
    if candidates == 0:
        return output.fill_(-1)

    pairs = score_workspace_pairs(batch, tokens, candidates)
    if pairs == 0:
        raise ValueError("CuTe candidate capacity is too large for the 32 MiB score workspace")
    scores = torch.empty((pairs, 2, candidates), dtype=torch.float32, device=q.device)
    use_int64_offsets = requires_int64_abi(
        q, k, weights, scores, output, candidate_bounds, q_scale, k_scale
    )
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
    previous = get_compile_target()
    try:
        set_compile_target(detect_compile_target(q.device.index))
        # A unit head stride avoids the generic reducer's measured strided-load overhead.
        compile_args = (
            weights.dtype,
            heads,
            head_dim,
            causal,
            compress_ratio,
            use_int64_offsets,
            weights.stride(-1) == 1,
        )
        if scaled_fp8:
            assert q_scale is not None and k_scale is not None
            score_kernel = _compile_mxfp8_scores(
                *compile_args,
                tensor_supports_contiguous_dim(q_scale, alignment_bytes=4),
                tensor_supports_contiguous_dim(k_scale, alignment_bytes=4),
            )
        elif candidate_bounds is None:
            score_kernel = _compile_scores(*compile_args)
        else:
            score_kernel = _compile_scores(*compile_args, has_candidate_bounds=True)
        topk_args = (topk, causal, compress_ratio, use_int64_offsets, deterministic)
        if candidate_bounds is None:
            topk_kernel = _compile_topk(*topk_args, selector=selector)
        else:
            topk_kernel = _compile_topk(*topk_args, has_candidate_bounds=True, selector=selector)
    finally:
        set_compile_target(previous)

    q, k, weights = q.detach(), k.detach(), weights.detach()
    scale = cutlass.Float32(1.0 / math.sqrt(heads * head_dim))
    score_args = (q, k, weights, q_scale, k_scale) if scaled_fp8 else (q, k, weights)
    total_pairs = batch * cdiv(tokens, 2)
    bounds = () if candidate_bounds is None else (candidate_bounds,)
    with initialized_cuda_device(q):
        for start in range(0, total_pairs, pairs):
            active_scores = scores[: min(pairs, total_pairs - start)]
            score_kernel(*score_args, active_scores, integer(start), scale, *bounds)
            topk_kernel(active_scores, output, integer(start), *bounds)
    return output


__all__ = ["launch", "resolve_selector"]
