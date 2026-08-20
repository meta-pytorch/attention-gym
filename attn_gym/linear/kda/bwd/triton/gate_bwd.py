# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Fused ragged backward for KDA's bounded gate and chunk-local prefix sum.
# Dense execution uses the CuTeDSL fused gate backward.
#
# For z = raw_gate + bias, A = exp(A_log), and s = sigmoid(A * z):
#   gate = lower_bound * s
#   d_gate = scale * reverse_chunk_cumsum(d_cumulative)
#   d_raw_gate = d_gate * lower_bound * A * s * (1 - s)
#   d_A_log = sum(d_raw_gate * z)
#   d_dt_bias = sum_tokens(d_raw_gate)

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import (
    GridScheduler,
    RaggedChunkMetadata,
    ScheduleKind,
    ScheduleRequest,
    decode_ragged_task,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
    load_ragged_task_count,
)
from attn_gym.linear.kda.utils import input_guard

_HEURISTICS = {
    "USE_INT64_OFFSETS": lambda args: requires_int64_offsets(
        args["raw_gate"],
        args["A_log"],
        args["dt_bias"],
        args["d_cumulative"],
        args["dg"],
        args["dA_partial"],
        args["ddt_partial"],
        args["cu_seqlens"],
        args["chunk_offsets"],
    )
}


@triton.jit
def _kda_gate_bwd_ragged_task(
    raw_gate,
    A_log,
    dt_bias,
    d_cumulative,
    dg,
    dA_partial,
    ddt_partial,
    lower_bound,
    scale,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    global_chunk,
    i_h,
    i_d,
    G_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    DY_STRIDES: tl.constexpr,
    DG_STRIDES: tl.constexpr,
    DA_STRIDES: tl.constexpr,
    DDT_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Reverse-scan one ragged chunk and differentiate the bounded gate in place.

    The scan, pointwise derivative, and parameter-gradient partials share one launch,
    without a full FP32 intermediate.
    """
    if USE_INT64_OFFSETS:
        global_chunk = global_chunk.to(tl.int64)
        i_h = i_h.to(tl.int64)
        i_d = i_d.to(tl.int64)
    _, _, token_start, valid_tokens = load_ragged_chunk_work(
        cu_seqlens,
        chunk_offsets,
        global_chunk,
        num_sequences,
        BT,
    )
    if USE_INT64_OFFSETS:
        token_start = token_start.to(tl.int64)
    token_offset = tl.arange(0, BT)
    token = token_start + token_offset
    channel = i_d * BD + tl.arange(0, BD)
    mask = (token_offset[:, None] < valid_tokens) & (channel[None, :] < D)
    d_gate = tl.load(
        d_cumulative + ptr_offset((token[:, None], i_h, channel[None, :]), DY_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    d_gate = tl.cumsum(d_gate, axis=0, reverse=True) * scale.to(tl.float32)

    gate_input = tl.load(
        raw_gate + ptr_offset((token[:, None], i_h, channel[None, :]), G_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_input += tl.load(
        dt_bias + ptr_offset((i_h, channel), DT_BIAS_STRIDES),
        mask=channel < D,
        other=0.0,
    ).to(tl.float32)[None, :]
    decay = tl.exp(tl.load(A_log + ptr_offset((i_h,), A_LOG_STRIDES)).to(tl.float32))
    sigmoid = tl.sigmoid(decay * gate_input)
    d_raw_gate = d_gate * (lower_bound.to(tl.float32) * decay * sigmoid * (1.0 - sigmoid))
    d_raw_gate = tl.where(mask, d_raw_gate, 0.0)

    tl.store(
        dg + ptr_offset((token[:, None], i_h, channel[None, :]), DG_STRIDES),
        d_raw_gate.to(dg.dtype.element_ty),
        mask=mask,
    )
    # dyg/dA_log = dg * z; masked lanes carry d_raw_gate == 0 and drop out.
    tl.store(
        dA_partial + ptr_offset((global_chunk, i_h, i_d), DA_STRIDES),
        tl.sum(tl.sum(d_raw_gate * gate_input, 1), 0),
    )
    tl.store(
        ddt_partial + ptr_offset((global_chunk, i_h, channel), DDT_STRIDES),
        tl.sum(d_raw_gate, 0),
        mask=channel < D,
    )


@triton.heuristics(_HEURISTICS)
@triton.jit(do_not_specialize=["num_sequences"])
def kda_gate_bwd_ragged_kernel(
    raw_gate,
    A_log,
    dt_bias,
    d_cumulative,
    dg,
    dA_partial,
    ddt_partial,
    lower_bound,
    scale,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    G_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    DY_STRIDES: tl.constexpr,
    DG_STRIDES: tl.constexpr,
    DA_STRIDES: tl.constexpr,
    DDT_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Launch one CTA per capacity task; capacity-only CTAs exit immediately."""
    global_chunk = tl.program_id(0)
    if global_chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
        return
    _kda_gate_bwd_ragged_task(
        raw_gate,
        A_log,
        dt_bias,
        d_cumulative,
        dg,
        dA_partial,
        ddt_partial,
        lower_bound,
        scale,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        global_chunk,
        tl.program_id(1),
        tl.program_id(2),
        G_STRIDES,
        A_LOG_STRIDES,
        DT_BIAS_STRIDES,
        DY_STRIDES,
        DG_STRIDES,
        DA_STRIDES,
        DDT_STRIDES,
        D,
        BT,
        BD,
        USE_INT64_OFFSETS,
    )


@triton.heuristics(_HEURISTICS)
@triton.jit(do_not_specialize=["num_sequences", "num_workers"])
def kda_gate_bwd_ragged_kernel_persistent(
    raw_gate,
    A_log,
    dt_bias,
    d_cumulative,
    dg,
    dA_partial,
    ddt_partial,
    lower_bound,
    scale,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    num_workers,
    G_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    DY_STRIDES: tl.constexpr,
    DG_STRIDES: tl.constexpr,
    DA_STRIDES: tl.constexpr,
    DDT_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    H: tl.constexpr,
    DIM_BLOCKS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Stride a bounded worker grid over active (chunk, head, dim-block) tasks.

    Unvisited slots in the capacity-sized partial buffers remain zero for the
    final reductions.
    """
    worker = tl.program_id(0)
    subtasks: tl.constexpr = H * DIM_BLOCKS
    total_tasks = load_ragged_task_count(chunk_offsets, num_sequences, subtasks)
    # num_stages=1 stops the software pipeliner from double-buffering the
    # outer task loop; the extra SMEM stage costs a resident CTA per SM.
    for task in tl.range(worker, total_tasks, num_workers, num_stages=1):
        global_chunk, subtask = decode_ragged_task(task, subtasks)
        _kda_gate_bwd_ragged_task(
            raw_gate,
            A_log,
            dt_bias,
            d_cumulative,
            dg,
            dA_partial,
            ddt_partial,
            lower_bound,
            scale,
            cu_seqlens,
            chunk_offsets,
            num_sequences,
            global_chunk,
            subtask // DIM_BLOCKS,
            subtask % DIM_BLOCKS,
            G_STRIDES,
            A_LOG_STRIDES,
            DT_BIAS_STRIDES,
            DY_STRIDES,
            DG_STRIDES,
            DA_STRIDES,
            DDT_STRIDES,
            D,
            BT,
            BD,
            USE_INT64_OFFSETS,
        )


@input_guard(no_guard_contiguous=True)
def kda_gate_bwd_ragged(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    metadata: RaggedChunkMetadata,
    *,
    lower_bound: float,
    scale: float,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the packed bounded-gate prefix sum with one fused launch.

    Returns ``dg`` in ``raw_gate.dtype`` plus the reduced FP32 ``dA_log`` and
    ``d_dt_bias`` parameter gradients.

    Args:
        schedule: Internal scheduling request for tests; automatic selection is
            the default.
    """
    if raw_gate.ndim != 4 or raw_gate.shape[0] != 1:
        raise ValueError(f"raw_gate must have shape [1, T, H, D], got {tuple(raw_gate.shape)}")
    if d_cumulative.shape != raw_gate.shape:
        raise ValueError(
            f"d_cumulative must have shape {tuple(raw_gate.shape)}, "
            f"got {tuple(d_cumulative.shape)}"
        )
    _, _, heads, head_dim = raw_gate.shape
    if A_log.shape != (heads,) or dt_bias.shape != (heads, head_dim):
        raise ValueError("A_log or dt_bias shape does not match raw_gate")

    block_dim = min(128, triton.next_power_of_2(head_dim))
    dim_blocks = triton.cdiv(head_dim, block_dim)
    dg = torch.empty_like(raw_gate)
    # Zero-filled because unvisited capacity slots participate in the final reductions.
    dA_partial = A_log.new_zeros((metadata.capacity, heads, dim_blocks))
    ddt_partial = A_log.new_zeros((metadata.capacity, heads, head_dim))
    if metadata.capacity == 0:
        return dg, dA_partial.sum((0, 2)), ddt_partial.sum(0)
    args = (
        raw_gate,
        A_log,
        dt_bias,
        d_cumulative,
        dg,
        dA_partial,
        ddt_partial,
        lower_bound,
        scale,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        metadata.cu_seqlens.shape[0] - 1,
    )
    kwargs = {
        "G_STRIDES": raw_gate.stride()[1:],
        "A_LOG_STRIDES": A_log.stride(),
        "DT_BIAS_STRIDES": dt_bias.stride(),
        "DY_STRIDES": d_cumulative.stride()[1:],
        "DG_STRIDES": dg.stride()[1:],
        "DA_STRIDES": dA_partial.stride(),
        "DDT_STRIDES": ddt_partial.stride(),
        "D": head_dim,
        "BT": metadata.chunk_size,
        "BD": block_dim,
        "num_warps": 4,
        "num_stages": 2,
    }
    resolved = GridScheduler(metadata).resolve_flat(schedule, heads * dim_blocks, raw_gate.device)
    if resolved.kind is ScheduleKind.PERSISTENT:
        kda_gate_bwd_ragged_kernel_persistent[(resolved.workers,)](
            *args,
            num_workers=resolved.workers,
            H=heads,
            DIM_BLOCKS=dim_blocks,
            **kwargs,
        )
    else:
        kda_gate_bwd_ragged_kernel[(metadata.capacity, heads, dim_blocks)](*args, **kwargs)
    return dg, dA_partial.sum((0, 2)), ddt_partial.sum(0)
