# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Bounded KDA gate activation and sequence-local chunk prefix sums.
#
# Dense and ragged inputs use separate launchers because ragged execution must route
# chunks from device-resident boundaries. Both implement the one shipped gate contract:
# required bias, bounded sigmoid activation, forward scan, and FP32 output.

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, storage_cosize
from attn_gym.linear.kda.bwd.triton.gate_bwd import kda_gate_bwd_ragged
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    chunk_capacity,
    load_ragged_chunk_work,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.kda.utils import RCP_LN2, exp, input_guard


# ---------------------------------------------------------------------------
# Bounded gate prefix-sum kernels.
# ---------------------------------------------------------------------------
def _requires_int64_offsets(args):
    """Select 64-bit indexing when any dense tensor offset can exceed int32."""
    input_cosize = storage_cosize(
        (args["B"], args["T"], args["H"], args["D"]),
        (args["g_batch_stride"], *args["G_STRIDES"]),
    )
    output_cosize = storage_cosize(
        (args["B"], args["T"], args["H"], args["D"]),
        (args["o_batch_stride"], *args["O_STRIDES"]),
    )
    bias_cosize = storage_cosize((args["H"], args["D"]), args["DT_BIAS_STRIDES"])
    a_log_cosize = storage_cosize((args["H"],), args["A_LOG_STRIDES"])
    return max(input_cosize, output_cosize, bias_cosize, a_log_cosize) > 1 << 31


@triton.heuristics({"USE_INT64_OFFSETS": _requires_int64_offsets})
@triton.jit(do_not_specialize=["T"])
def _bounded_gate_chunk_cumsum_dense_kernel(
    raw_gate,
    A_log,
    dt_bias,
    output,
    lower_bound,
    scale,
    T,
    g_batch_stride,
    o_batch_stride,
    G_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Apply the bounded gate and forward prefix sum to one dense chunk."""
    i_chunk = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_d = tl.program_id(2)
    if USE_INT64_OFFSETS:
        i_chunk = i_chunk.to(tl.int64)
        i_bh = i_bh.to(tl.int64)
        i_d = i_d.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    token = i_chunk * BT + tl.arange(0, BT)
    channel = i_d * BD + tl.arange(0, BD)
    token_mask = token < T
    mask = token_mask[:, None] & (channel[None, :] < D)
    batch_offset = i_b * g_batch_stride if B > 1 else 0
    gate_input = tl.load(
        raw_gate + batch_offset + ptr_offset((token[:, None], i_h, channel[None, :]), G_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        dt_bias + ptr_offset((i_h, channel), DT_BIAS_STRIDES),
        mask=channel < D,
        other=0.0,
    ).to(tl.float32)
    gate_input += bias[None, :]
    decay_log = tl.load(A_log + ptr_offset((i_h,), A_LOG_STRIDES)).to(tl.float32)
    # Inductor may pass Python floats as fp64; keep scalar gate math in fp32.
    gate = lower_bound.to(tl.float32) * tl.sigmoid(exp(decay_log) * gate_input)
    # Preserve deployed tail handling: channel-tail lanes are excluded by load/store masks.
    gate = tl.where(token_mask[:, None], gate, 0.0)
    cumulative = tl.cumsum(gate, axis=0)
    cumulative *= scale.to(tl.float32)

    output_batch_offset = i_b * o_batch_stride if B > 1 else 0
    tl.store(
        output
        + output_batch_offset
        + ptr_offset((token[:, None], i_h, channel[None, :]), O_STRIDES),
        cumulative.to(output.dtype.element_ty),
        mask=mask,
    )


@input_guard(no_guard_contiguous=True)
def _bounded_gate_cumsum_dense(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
) -> torch.Tensor:
    """Launch the bounded gate over an ordinary dense batch."""
    batch, tokens, heads, head_dim = raw_gate.shape
    output = torch.empty_like(raw_gate, dtype=torch.float32)
    block_dim = 32
    _bounded_gate_chunk_cumsum_dense_kernel[
        (triton.cdiv(tokens, chunk_size), batch * heads, triton.cdiv(head_dim, block_dim))
    ](
        raw_gate,
        A_log,
        dt_bias,
        output,
        lower_bound,
        RCP_LN2,
        tokens,
        raw_gate.stride(0),
        output.stride(0),
        G_STRIDES=raw_gate.stride()[1:],
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=dt_bias.stride(),
        O_STRIDES=output.stride()[1:],
        B=batch,
        H=heads,
        D=head_dim,
        BT=chunk_size,
        BD=block_dim,
        num_warps=2,
        num_stages=3,
    )
    return output


@triton.jit(do_not_specialize=["num_sequences"])
def bounded_gate_chunk_cumsum_ragged_kernel(
    raw_gate,
    A_log,
    dt_bias,
    output,
    lower_bound,
    scale,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    G_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Apply the bounded gate and prefix sum to one device-routed ragged chunk."""
    global_chunk = tl.program_id(0)
    i_h = tl.program_id(1).to(tl.int64)
    i_d = tl.program_id(2).to(tl.int64)
    if global_chunk >= tl.load(chunk_offsets + num_sequences):
        return

    _sequence, _local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
        cu_seqlens,
        chunk_offsets,
        global_chunk,
        num_sequences,
        BT,
    )
    token_offset = tl.arange(0, BT)
    token = token_start.to(tl.int64) + token_offset
    channel = i_d * BD + tl.arange(0, BD)
    mask = (token_offset[:, None] < valid_tokens) & (channel[None, :] < D)
    gate_input = tl.load(
        raw_gate + ptr_offset((token[:, None], i_h, channel[None, :]), G_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        dt_bias + ptr_offset((i_h, channel), DT_BIAS_STRIDES),
        mask=channel < D,
        other=0.0,
    ).to(tl.float32)
    decay = tl.load(A_log + ptr_offset((i_h,), A_LOG_STRIDES)).to(tl.float32)
    gate = lower_bound.to(tl.float32) * tl.sigmoid(exp(decay) * (gate_input + bias[None, :]))
    gate = tl.where(mask, gate, 0.0)
    # Keep the caller-provided scalar in the gate's FP32 arithmetic domain.
    cumulative = tl.cumsum(gate, axis=0) * scale.to(tl.float32)
    tl.store(
        output + ptr_offset((token[:, None], i_h, channel[None, :]), O_STRIDES),
        cumulative,
        mask=mask,
    )


@input_guard(no_guard_contiguous=True)
def bounded_gate_cumsum_ragged(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    metadata: RaggedChunkMetadata,
    *,
    lower_bound: float,
) -> torch.Tensor:
    """Run the graph-safe gate with an already prepared packed schedule."""
    output = torch.empty_like(raw_gate, dtype=torch.float32)
    _, _, heads, head_dim = raw_gate.shape
    block_dim = 32
    bounded_gate_chunk_cumsum_ragged_kernel[
        (metadata.capacity, heads, triton.cdiv(head_dim, block_dim))
    ](
        raw_gate,
        A_log,
        dt_bias,
        output,
        lower_bound,
        RCP_LN2,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        metadata.cu_seqlens.shape[0] - 1,
        G_STRIDES=raw_gate.stride()[1:],
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=dt_bias.stride(),
        O_STRIDES=output.stride()[1:],
        D=head_dim,
        BT=metadata.chunk_size,
        BD=block_dim,
        num_warps=2,
        num_stages=3,
    )
    return output


torch.library.define(
    "attn_gym::kda_bounded_gate_fwd_dense",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, int chunk_size, float lower_bound) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_bounded_gate_fwd_ragged",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, Tensor cu_seqlens, "
    "Tensor chunk_offsets, int chunk_size, float lower_bound) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_bounded_gate_bwd_ragged",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, Tensor d_cumulative, "
    "Tensor cu_seqlens, Tensor chunk_offsets, int chunk_size, float lower_bound, "
    "bool profile_ranges) -> (Tensor, Tensor, Tensor)",
)


def _bounded_gate_cumsum_ragged_fwd_cuda(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
) -> torch.Tensor:
    """Run ragged gate activation and prefix sums with caller-prepared routing."""
    metadata = RaggedChunkMetadata(
        cu_seqlens,
        chunk_offsets,
        chunk_capacity(raw_gate.shape[1], cu_seqlens.shape[0] - 1, chunk_size),
        chunk_size,
    )
    return bounded_gate_cumsum_ragged(
        raw_gate,
        A_log,
        dt_bias,
        metadata,
        lower_bound=lower_bound,
    )


def _bounded_gate_cumsum_ragged_bwd_cuda(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
    profile_ranges: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the packed gate prefix sum with one fused launch."""
    metadata = RaggedChunkMetadata(
        cu_seqlens,
        chunk_offsets,
        chunk_capacity(raw_gate.shape[1], cu_seqlens.shape[0] - 1, chunk_size),
        chunk_size,
    )
    with (
        torch.profiler.record_function("kda/triton/gate_backward_ragged")
        if profile_ranges
        else nullcontext()
    ):
        return kda_gate_bwd_ragged(
            raw_gate,
            A_log,
            dt_bias,
            d_cumulative,
            metadata,
            lower_bound=lower_bound,
            scale=RCP_LN2,
        )


torch.library.impl("attn_gym::kda_bounded_gate_fwd_dense", "CUDA", _bounded_gate_cumsum_dense)
torch.library.impl(
    "attn_gym::kda_bounded_gate_fwd_ragged", "CUDA", _bounded_gate_cumsum_ragged_fwd_cuda
)
torch.library.impl(
    "attn_gym::kda_bounded_gate_bwd_ragged", "CUDA", _bounded_gate_cumsum_ragged_bwd_cuda
)


@torch.library.register_fake("attn_gym::kda_bounded_gate_fwd_dense")
def _bounded_gate_cumsum_dense_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
) -> torch.Tensor:
    """Describe the dense gate output."""
    del A_log, dt_bias, chunk_size, lower_bound
    return torch.empty_like(raw_gate, dtype=torch.float32)


@torch.library.register_fake("attn_gym::kda_bounded_gate_fwd_ragged")
def _bounded_gate_cumsum_ragged_fwd_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
) -> torch.Tensor:
    """Describe the ragged gate output."""
    del A_log, dt_bias, cu_seqlens, chunk_offsets, chunk_size, lower_bound
    return torch.empty_like(raw_gate, dtype=torch.float32)


@torch.library.register_fake("attn_gym::kda_bounded_gate_bwd_ragged")
def _bounded_gate_cumsum_ragged_bwd_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
    profile_ranges: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe ragged gate input and parameter gradients."""
    del d_cumulative, cu_seqlens, chunk_offsets, chunk_size, lower_bound, profile_ranges
    return (
        torch.empty_like(raw_gate),
        A_log.new_empty(A_log.shape),
        dt_bias.new_empty(dt_bias.shape),
    )


_bounded_gate_cumsum_dense_op = torch.ops.attn_gym.kda_bounded_gate_fwd_dense.default
_bounded_gate_cumsum_ragged_fwd_op = torch.ops.attn_gym.kda_bounded_gate_fwd_ragged.default
_bounded_gate_cumsum_ragged_bwd_op = torch.ops.attn_gym.kda_bounded_gate_bwd_ragged.default


class _BoundedGateCumsum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw_gate: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        chunk_size: int,
        lower_bound: float,
        fastmath: bool,
        profile_ranges: bool,
        cu_seqlens: torch.Tensor | None,
        chunk_offsets: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.chunk_size = chunk_size
        ctx.lower_bound = lower_bound
        ctx.fastmath = fastmath
        ctx.profile_ranges = profile_ranges
        ctx.ragged = cu_seqlens is not None
        if cu_seqlens is not None:
            assert chunk_offsets is not None
            output = _bounded_gate_cumsum_ragged_fwd_op(
                raw_gate,
                A_log,
                dt_bias,
                cu_seqlens,
                chunk_offsets,
                chunk_size,
                lower_bound,
            )
            ctx.save_for_backward(raw_gate, A_log, dt_bias, cu_seqlens, chunk_offsets)
            return output

        assert chunk_offsets is None
        ctx.save_for_backward(raw_gate, A_log, dt_bias)
        return _bounded_gate_cumsum_dense_op(
            raw_gate,
            A_log,
            dt_bias,
            chunk_size,
            lower_bound,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_cumulative: torch.Tensor):
        if ctx.ragged:
            raw_gate, A_log, dt_bias, cu_seqlens, chunk_offsets = ctx.saved_tensors
            dg, dA_log, d_dt_bias = _bounded_gate_cumsum_ragged_bwd_op(
                raw_gate,
                A_log,
                dt_bias,
                d_cumulative,
                cu_seqlens,
                chunk_offsets,
                ctx.chunk_size,
                ctx.lower_bound,
                ctx.profile_ranges,
            )
            return (
                dg,
                dA_log,
                d_dt_bias,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        from attn_gym.linear.kda.bwd.cute.gate_bwd_fused import fused_gate_bwd

        raw_gate, A_log, dt_bias = ctx.saved_tensors
        with (
            torch.profiler.record_function("kda/cute/gate_backward_fused")
            if ctx.profile_ranges
            else nullcontext()
        ):
            result = fused_gate_bwd(
                raw_gate,
                A_log,
                dt_bias,
                d_cumulative.float(),
                chunk_size=ctx.chunk_size,
                lower_bound=ctx.lower_bound,
                fastmath=ctx.fastmath,
            )
        return (
            result.dg,
            result.dA_partial.sum((0, 1)),
            result.d_dt_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# NOTE [Gate range ceiling]
# The bounded gate lies in [lower_bound, 0], so one token contributes at most
# |lower_bound| * log2(e) after conversion to log2 units. Across `span_steps`, the rebase
# therefore needs this exponent budget:
#
#     |lower_bound| * span_steps * log2(e) <= 128
#
# The causal reference spans BC-1 = 15 steps, giving a data-independent 5.915 limit.
# Validate it here because an overflowing rebase factor can otherwise produce non-finite
# core outputs. See NOTE [Causal gate reference] in the diagonal forward kernel.
GATE_SPAN_STEPS = 15
FP32_EXPONENT_BUDGET = 128.0
MAX_GATE_LOWER_BOUND_MAGNITUDE = FP32_EXPONENT_BUDGET / (GATE_SPAN_STEPS * math.log2(math.e))


def _bounded_gate_cumsum(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    chunk_size: int = 64,
    lower_bound: float = -5.0,
    fastmath: bool = False,
    profile_ranges: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    metadata: RaggedChunkMetadata | None = None,
) -> torch.Tensor:
    """Apply the gate after canonicalizing caller- or integration-owned routing."""
    if metadata is not None:
        assert cu_seqlens is None
        metadata.validate_chunk_size(chunk_size)
        cu_seqlens = metadata.cu_seqlens
    if raw_gate.ndim != 4:
        raise ValueError(f"raw_gate must have shape [B, T, H, D], got {tuple(raw_gate.shape)}")
    batch, tokens, heads, head_dim = raw_gate.shape
    if batch == 0 or tokens == 0 or heads == 0:
        raise ValueError(f"raw_gate must have nonempty B, T, and H, got {tuple(raw_gate.shape)}")
    if head_dim < 32 or head_dim > 1024 or head_dim % 32:
        raise ValueError(
            f"raw_gate head dimension must be a multiple of 32 in [32, 1024], got {head_dim}"
        )
    if not raw_gate.is_cuda or raw_gate.dtype != torch.bfloat16:
        raise TypeError("bounded_gate_cumsum requires a CUDA bfloat16 raw gate")
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        raise ValueError(
            f"A_log must be float32 with shape {(heads,)}, "
            f"got {tuple(A_log.shape)} and {A_log.dtype}"
        )
    if dt_bias.shape != (heads, head_dim) or dt_bias.dtype != torch.float32:
        raise ValueError(
            f"dt_bias must be float32 with shape {(heads, head_dim)}, "
            f"got {tuple(dt_bias.shape)} and {dt_bias.dtype}"
        )
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError(f"chunk_size must be a positive power of two, got {chunk_size}")
    # Plain comparisons only: `math.isfinite` on a traced float lifts it into a graph
    # operation, which breaks strict dynamic compilation. A chained comparison rejects
    # NaN and both infinities without calling into `math`.
    if not -MAX_GATE_LOWER_BOUND_MAGNITUDE <= lower_bound <= 0.0:
        raise ValueError(
            f"lower_bound must lie in "
            f"[{-MAX_GATE_LOWER_BOUND_MAGNITUDE:.3f}, 0] for the KDA intra-chunk gate "
            f"rebase, got {lower_bound}. Past the lower end the per-token decay can "
            f"exceed the FP32 exponent budget over a {GATE_SPAN_STEPS + 1}-row subchunk, "
            "and the core silently produces non-finite values. "
            "See NOTE [Gate range ceiling]."
        )
    if not all(tensor.device == raw_gate.device for tensor in (A_log, dt_bias)):
        raise ValueError("bounded_gate_cumsum inputs must be on the same device")
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed cu_seqlens require raw_gate to have batch size one")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
        if (
            cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != raw_gate.device
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError("cu_seqlens must be contiguous CUDA int32 on raw_gate.device")
        if fastmath:
            raise ValueError("packed bounded_gate_cumsum does not support fastmath=True")
        if metadata is None:
            metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, chunk_size)
    return _BoundedGateCumsum.apply(
        raw_gate,
        A_log,
        dt_bias,
        chunk_size,
        lower_bound,
        fastmath,
        profile_ranges,
        cu_seqlens,
        None if metadata is None else metadata.chunk_offsets,
    )


def bounded_gate_cumsum(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    chunk_size: int = 64,
    lower_bound: float = -5.0,
    fastmath: bool = False,
    profile_ranges: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a bounded KDA gate and sequence-local chunk prefix sums.

    Packed ``cu_seqlens`` must start at zero, end at or before the physical ``T``, and
    be monotonic; repeated offsets represent empty sequences. Output values and raw-gate
    gradients are defined only on ``[0, cu_seqlens[-1])``. The offsets remain
    device-resident and are validated by the scheduler so fixed-shape CUDA Graphs can
    replay with different boundaries and active lengths.
    """
    return _bounded_gate_cumsum(
        raw_gate,
        A_log,
        dt_bias,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
        fastmath=fastmath,
        profile_ranges=profile_ranges,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["bounded_gate_cumsum"]
