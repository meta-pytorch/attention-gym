"""Bounded-workspace indexer for SM100/SM103 scoring followed by radix Top-K.

A slab contains at most 1024 query rows and 32 MiB of FP32 scores, independent
of batch size and sequence length. Consecutive launches reuse that per-call
allocation on the current stream; the only retained result is the INT32 index
array. Slabs contain whole query pairs, including a masked second row for odd T.
The registered operator calls this launcher; there is no backend fallback.
"""

import math
from collections.abc import Callable

import torch

from attn_gym._backends.cute import (
    TMA_ALIGNMENT_BYTES,
    compile_tvm_ffi,
    get_device_properties,
    jit_cache,
    make_fake_strided_tensor,
    tensor_supports_tma,
)
from attn_gym._backends.cute.target import (
    detect_compile_target,
    get_compile_target,
    set_compile_target,
)
from attn_gym._backends.cute.utils import initialized_cuda_device, requires_int64_abi
from attn_gym.utils import cdiv

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
    from cutlass import cute

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
    sym = cute.sym_int64 if use_int64_offsets else cute.sym_int
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
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
        io_dtype,
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
    args = (q, k, weights, scores, integer(0), cutlass.Float32(1.0))
    if has_candidate_bounds:
        return compile_tvm_ffi(
            operation.with_candidate_bounds,
            *args,
            _fake_candidate_bounds(tokens, use_int64_offsets),
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
def _compile_topk(
    topk: int,
    causal: bool,
    compress_ratio: int,
    use_int64_offsets: bool,
    deterministic: bool,
    has_candidate_bounds: bool = False,
) -> Callable[..., None]:
    """Compile indices-only radix selection with symbolic B, T, S and slab capacity."""
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


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    compress_ratio: int,
    candidate_bounds: torch.Tensor | None = None,
) -> None:
    """Validate SM100/SM103 indexer tensor metadata and base alignment."""
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B,T,H,D], got {tuple(q.shape)}")
    if k.ndim != 3:
        raise ValueError(f"k must have shape [B,S,D], got {tuple(k.shape)}")
    if weights.ndim != 3:
        raise ValueError(f"weights must have shape [B,T,H], got {tuple(weights.shape)}")

    batch, queries, heads, head_dim = q.shape
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    candidates = k.shape[1] if candidate_bounds is not None else queries // compress_ratio
    if tuple(k.shape) != (batch, candidates, head_dim):
        raise ValueError(
            f"k must have shape {(batch, candidates, head_dim)} for T={queries} and "
            f"compress_ratio={compress_ratio}, got {tuple(k.shape)}"
        )
    if tuple(weights.shape) != (batch, queries, heads):
        raise ValueError(
            f"weights must have shape {(batch, queries, heads)}, got {tuple(weights.shape)}"
        )
    if batch <= 0:
        raise ValueError(f"batch must be positive, got {batch}")
    if queries <= 0 or queries > _MAX_SEQUENCE:
        raise ValueError(f"sequence length must be in [1, {_MAX_SEQUENCE}], got {queries}")
    if heads <= 0 or heads % 2:
        raise ValueError(f"number of heads must be a positive multiple of 2, got {heads}")
    if head_dim <= 0 or head_dim % _HEAD_DIM_GRANULARITY:
        raise ValueError(
            "head dimension must be positive and divisible by "
            f"{_HEAD_DIM_GRANULARITY}, got {head_dim}"
        )
    if not isinstance(topk, int) or isinstance(topk, bool):
        raise TypeError(f"topk must be an int, got {type(topk).__name__}")
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    tensors = (q, k, weights)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("q, k, and weights must all be CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("q, k, and weights must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}")
    if k.dtype != q.dtype or weights.dtype != q.dtype:
        raise TypeError(
            f"q, k, and weights must have one dtype, got {q.dtype}, {k.dtype}, {weights.dtype}"
        )
    # Singleton strides do not address another slice; TVM-FFI normalizes them for TMA.
    if any(tensor.numel() and not tensor_supports_tma(tensor.squeeze()) for tensor in (q, k)):
        raise ValueError(
            f"q and k require unit last strides and {TMA_ALIGNMENT_BYTES}-byte aligned "
            "bases and non-singleton outer strides"
        )
    properties = get_device_properties(q.device)
    if (properties.major, properties.minor) not in ((10, 0), (10, 3)):
        raise RuntimeError("this tcgen05 kernel requires an SM100 or SM103 GPU")


def launch(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool = False,
    compress_ratio: int = 1,
    deterministic: bool = False,
    candidate_bounds: torch.Tensor | None = None,
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

    _validate(q, k, weights, topk, compress_ratio, candidate_bounds)
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
    use_int64_offsets = requires_int64_abi(q, k, weights, scores, output, candidate_bounds)
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
    previous = get_compile_target()
    try:
        set_compile_target(detect_compile_target(q.device.index))
        # A unit head stride avoids the generic reducer's measured strided-load overhead.
        score_args = (
            q.dtype,
            heads,
            head_dim,
            causal,
            compress_ratio,
            use_int64_offsets,
            weights.stride(-1) == 1,
        )
        topk_args = (topk, causal, compress_ratio, use_int64_offsets, deterministic)
        if candidate_bounds is None:
            score_kernel = _compile_scores(*score_args)
            topk_kernel = _compile_topk(*topk_args)
        else:
            score_kernel = _compile_scores(*score_args, has_candidate_bounds=True)
            topk_kernel = _compile_topk(*topk_args, has_candidate_bounds=True)
    finally:
        set_compile_target(previous)

    q, k, weights = q.detach(), k.detach(), weights.detach()
    scale = cutlass.Float32(1.0 / math.sqrt(heads * head_dim))
    total_pairs = batch * cdiv(tokens, 2)
    bounds = () if candidate_bounds is None else (candidate_bounds,)
    with initialized_cuda_device(q):
        for start in range(0, total_pairs, pairs):
            active_scores = scores[: min(pairs, total_pairs - start)]
            score_kernel(q, k, weights, active_scores, integer(start), scale, *bounds)
            topk_kernel(active_scores, output, integer(start), *bounds)
    return output


__all__ = ["launch"]
