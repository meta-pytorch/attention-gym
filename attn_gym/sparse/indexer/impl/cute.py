"""Bounded-workspace indexer for SM100/SM103 scoring and exact Top-K.

A slab contains at most 1024 query rows and 32 MiB of FP32 scores, independent
of batch size and sequence length. Consecutive launches reuse that per-call
allocation on the current stream; the only retained result is the INT32 index
array. Slabs contain whole query pairs, including a masked second row for odd T.
The registered operator calls this launcher; there is no backend fallback.
"""

import math
from collections.abc import Callable
from pathlib import Path

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
from .common import score_workspace_pairs

_HEAD_DIM_GRANULARITY = 16
_MAX_SEQUENCE = 1 << 20


@jit_cache(
    extra_sources=(
        Path(__file__).with_name("cute_score.py"),
        Path(__file__).with_name("cute_score_generic.py"),
    )
)
def _compile_scores(
    dtype: torch.dtype,
    heads: int,
    head_dim: int,
    causal: bool,
    compress_ratio: int,
    use_int64_offsets: bool,
    contiguous_weight_heads: bool,
) -> Callable[..., None]:
    """Compile symbolic B/T/S/strides, specializing only an available unit weight-head stride."""
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
    return compile_tvm_ffi(
        operation,
        q,
        k,
        weights,
        scores,
        integer(0),
        cutlass.Float32(1.0),
    )


@jit_cache(
    extra_sources=(
        Path(__file__).with_name("cute_score.py"),
        Path(__file__).with_name("cute_score_mxfp8.py"),
    )
)
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
    from cutlass import cute

    from .cute_score_mxfp8 import IndexerMXFP8ScoreKernel

    weight_type = cutlass.BFloat16 if weights_dtype == torch.bfloat16 else cutlass.Float32
    sym = cute.sym_int64 if use_int64_offsets else cute.sym_int
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
    batch, tokens, candidates, pairs = sym(), sym(), sym(), sym()
    q = make_fake_strided_tensor(
        cutlass.Float8E4M3FN,
        (batch, tokens, heads, head_dim),
        stride_divisibility=TMA_ALIGNMENT_BYTES,
        use_int64_strides=use_int64_offsets,
    )
    k = make_fake_strided_tensor(
        cutlass.Float8E4M3FN,
        (batch, candidates, head_dim),
        stride_divisibility=TMA_ALIGNMENT_BYTES,
        use_int64_strides=use_int64_offsets,
    )
    weights = make_fake_strided_tensor(
        weight_type,
        (batch, tokens, heads),
        contiguous_dim=-1 if contiguous_weight_heads else None,
        use_int64_strides=use_int64_offsets,
    )
    q_scale = make_fake_strided_tensor(
        cutlass.Float8E8M0FNU,
        (batch, tokens, heads, head_dim // 32),
        contiguous_dim=-1 if contiguous_q_scales else None,
        stride_divisibility=4 if contiguous_q_scales else 1,
        use_int64_strides=use_int64_offsets,
    )
    k_scale = make_fake_strided_tensor(
        cutlass.Float8E8M0FNU,
        (batch, candidates, head_dim // 32),
        contiguous_dim=-1 if contiguous_k_scales else None,
        stride_divisibility=4 if contiguous_k_scales else 1,
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


@jit_cache(
    extra_sources=(
        Path(__file__).with_name("cute_topk.py"),
        Path(__file__).with_name("cute_topk_gvr2.py"),
    )
)
def _compile_topk(
    topk: int,
    causal: bool,
    compress_ratio: int,
    use_int64_offsets: bool,
    selector: str = "default",
) -> Callable[..., None]:
    """Compile indices-only selection with symbolic B, T, S and slab capacity."""
    import cutlass
    from cutlass import cute

    from .cute_topk import IndexerTopKKernel

    operation = IndexerTopKKernel
    if selector == "gvr2":
        from .cute_topk_gvr2 import IndexerGVR2TopKKernel

        operation = IndexerGVR2TopKKernel
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
    return compile_tvm_ffi(
        operation(topk, causal, use_int64_offsets, compress_ratio=compress_ratio),
        scores,
        output,
        integer(0),
    )


def _validate(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    compress_ratio: int,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
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
    candidates = queries // compress_ratio
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
    if topk < 0 or topk > candidates:
        raise ValueError(f"topk must be in [0, {candidates}], got {topk}")

    tensors = (q, k, weights)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("q, k, and weights must all be CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("q, k, and weights must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError(f"q must be float16, bfloat16 or float8_e4m3fn, got {q.dtype}")
    validate_precision(q, k, weights, q_scale, k_scale)
    if q.dtype == torch.float8_e4m3fn and (heads not in (32, 64) or head_dim != 128):
        raise ValueError("MXFP8 CuTe indexing requires H in {32, 64} and D=128")
    # Singleton strides do not address another slice; TVM-FFI normalizes them for TMA.
    if any(not tensor_supports_tma(tensor.squeeze()) for tensor in (q, k)):
        raise ValueError(
            f"q and k require unit last strides and {TMA_ALIGNMENT_BYTES}-byte aligned "
            "bases and non-singleton outer strides"
        )
    properties = get_device_properties(q.device)
    if (properties.major, properties.minor) not in ((10, 0), (10, 3)):
        raise RuntimeError("this tcgen05 kernel requires an SM100 or SM103 GPU")


def resolve_selector(selector: str, topk: int) -> str:
    """Map ``"auto"`` to the cheaper exact selector: GVR2 below ``NATIVE_TOPK_LIMIT``,
    where it runs natively; radix at or above it, where GVR2 would only fall back to
    radix in-kernel. See docs/indexer_gvr2_performance.md.
    """
    from .cute_topk_gvr2 import NATIVE_TOPK_LIMIT

    if selector == "auto":
        return "gvr2" if topk < NATIVE_TOPK_LIMIT else "default"
    if selector not in ("default", "gvr2"):
        raise ValueError(f"unknown indexer selector {selector!r}")
    return selector


def launch(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool = False,
    compress_ratio: int = 1,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    selector: str = "default",
) -> torch.Tensor:
    """Compute weighted-ReLU Top-K with bounded per-call score storage."""
    import cutlass

    selector = resolve_selector(selector, topk)
    _validate(q, k, weights, topk, compress_ratio, q_scale, k_scale)
    batch, tokens, heads, head_dim = q.shape
    candidates = k.shape[1]
    output = torch.empty((batch, tokens, topk), dtype=torch.int32, device=q.device)
    if topk == 0:
        return output

    pairs = score_workspace_pairs(batch, tokens, candidates)
    scores = torch.empty((pairs, 2, candidates), dtype=torch.float32, device=q.device)
    scaled_fp8 = q.dtype == torch.float8_e4m3fn
    use_int64_offsets = requires_int64_abi(q, k, weights, scores, output, q_scale, k_scale)
    integer = cutlass.Int64 if use_int64_offsets else cutlass.Int32
    previous = get_compile_target()
    try:
        set_compile_target(detect_compile_target(q.device.index))
        # A unit head stride avoids the generic reducer's measured strided-load overhead.
        compile_args = (
            weights.dtype,  # Equals q.dtype except for MXFP8, which is validated above.
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
        else:
            score_kernel = _compile_scores(*compile_args)
        topk_kernel = _compile_topk(topk, causal, compress_ratio, use_int64_offsets, selector)
    finally:
        set_compile_target(previous)

    q, k, weights = q.detach(), k.detach(), weights.detach()
    scale = cutlass.Float32(1.0 / math.sqrt(heads * head_dim))
    score_args = (q, k, weights, q_scale, k_scale) if scaled_fp8 else (q, k, weights)
    total_pairs = batch * cdiv(tokens, 2)
    with initialized_cuda_device(q):
        for start in range(0, total_pairs, pairs):
            active_scores = scores[: min(pairs, total_pairs - start)]
            score_kernel(*score_args, active_scores, integer(start), scale)
            topk_kernel(active_scores, output, integer(start))
    return output


__all__ = ["launch", "resolve_selector"]
