"""Bounded-workspace indexer for SM100-family scoring followed by radix Top-K.

A slab contains at most 1024 query rows and 32 MiB of FP32 scores, independent
of batch size and sequence length. Consecutive launches reuse that per-call
allocation on the current stream; the only retained result is the INT32 index
array. Slabs contain whole query pairs, including a masked second row for odd T.
The registered operator calls this launcher; there is no backend fallback.
"""

import math
from collections.abc import Callable

import torch

from attn_gym._backends.arch import SM100_KERNELS_REQUIREMENT, runs_sm100_kernels
from attn_gym._backends.cute import TMA_ALIGNMENT_BYTES, get_device_properties, tensor_supports_tma
from attn_gym._backends.cute.utils import initialized_cuda_device, requires_int64_abi
from attn_gym.utils import cdiv
from attn_gym.warpdrive.cache import (
    compile_tvm_ffi,
    detect_compile_target,
    get_compile_target,
    jit_cache,
    make_fake_strided_tensor,
    set_compile_target,
)

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
    has_candidate_bounds: bool,
) -> Callable[..., None]:
    """Compile symbolic B/T/S/strides, specializing only an available unit weight-head stride.

    With ``has_candidate_bounds``, score only each query's packed ``candidate_bounds`` interval;
    otherwise the compiled kernel takes None for ``candidate_bounds``.
    """
    import cutlass
    from cutlass import cute

    from .cute_score import IndexerScoreKernel
    from .cute_score_generic import IndexerGenericScoreKernel

    io_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
    operation = (
        IndexerScoreKernel if heads in (32, 64) and head_dim == 128 else IndexerGenericScoreKernel
    )(
        heads,
        head_dim,
        causal,
        use_int64_offsets,
        compress_ratio=compress_ratio,
        contiguous_weight_heads=contiguous_weight_heads,
        dtype=io_dtype,
    )
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
    return compile_tvm_ffi(
        operation,
        q,
        k,
        weights,
        scores,
        integer(0),
        cutlass.Float32(1.0),
        _fake_candidate_bounds(has_candidate_bounds, tokens, use_int64_offsets),
        name=operation.get_name() + ("_packed" if has_candidate_bounds else ""),
    )


def _fake_candidate_bounds(enabled: bool, tokens, use_int64_offsets: bool):
    """Fake contiguous INT32 [T, 2] packed candidate intervals, or None for a dense kernel."""
    if not enabled:
        return None
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
    has_candidate_bounds: bool,
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
    return compile_tvm_ffi(
        operation,
        scores,
        output,
        integer(0),
        _fake_candidate_bounds(has_candidate_bounds, tokens, use_int64_offsets),
        name=operation.get_name() + ("_packed" if has_candidate_bounds else ""),
    )


def _validate(q: torch.Tensor, k: torch.Tensor) -> None:
    """Validate SM100-family kernel limits and Q/K base alignment.

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
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}")
    # Singleton strides do not address another slice; TVM-FFI normalizes them for TMA.
    if any(tensor.numel() and not tensor_supports_tma(tensor.squeeze()) for tensor in (q, k)):
        raise ValueError(
            f"q and k require unit last strides and {TMA_ALIGNMENT_BYTES}-byte aligned "
            "bases and non-singleton outer strides"
        )
    properties = get_device_properties(q.device)
    if not runs_sm100_kernels((properties.major, properties.minor)):
        raise RuntimeError(f"this tcgen05 kernel requires {SM100_KERNELS_REQUIREMENT}")


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
    use_int64_offsets = requires_int64_abi(q, k, weights, scores, output, candidate_bounds)
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
    previous = get_compile_target()
    try:
        set_compile_target(detect_compile_target(q.device.index))
        packed = candidate_bounds is not None
        # Packed bounds already contain both mask decisions; neither kernel reads these flags.
        if packed:
            causal, compress_ratio = False, 1
        score_kernel = _compile_scores(
            q.dtype,
            heads,
            head_dim,
            causal,
            compress_ratio,
            use_int64_offsets,
            # A unit head stride avoids the generic reducer's measured strided-load overhead.
            weights.stride(-1) == 1,
            packed,
        )
        topk_kernel = _compile_topk(
            topk, causal, compress_ratio, use_int64_offsets, deterministic, packed
        )
    finally:
        set_compile_target(previous)

    q, k, weights = q.detach(), k.detach(), weights.detach()
    scale = cutlass.Float32(1.0 / math.sqrt(heads * head_dim))
    total_pairs = batch * cdiv(tokens, 2)
    with initialized_cuda_device(q):
        for start in range(0, total_pairs, pairs):
            active_scores = scores[: min(pairs, total_pairs - start)]
            score_kernel(q, k, weights, active_scores, integer(start), scale, candidate_bounds)
            topk_kernel(active_scores, output, integer(start), candidate_bounds)
    return output


__all__ = ["launch"]
