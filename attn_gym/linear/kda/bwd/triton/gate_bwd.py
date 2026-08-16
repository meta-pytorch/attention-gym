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

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, load_ragged_chunk_work
from attn_gym.linear.kda.utils import input_guard


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
):
    """Reverse-scan one ragged chunk and differentiate the bounded gate in place.

    The scan, pointwise derivative, and parameter-gradient partials share one launch,
    without a full FP32 intermediate.
    """
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the packed bounded-gate prefix sum with one fused launch.

    Returns ``dg`` in ``raw_gate.dtype`` plus the reduced FP32 ``dA_log`` and
    ``d_dt_bias`` parameter gradients.
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
    # Zero-filled so chunk slots beyond the active count contribute empty partials.
    dA_partial = A_log.new_zeros((metadata.capacity, heads, dim_blocks))
    ddt_partial = A_log.new_zeros((metadata.capacity, heads, head_dim))
    kda_gate_bwd_ragged_kernel[(metadata.capacity, heads, dim_blocks)](
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
        G_STRIDES=raw_gate.stride()[1:],
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=dt_bias.stride(),
        DY_STRIDES=d_cumulative.stride()[1:],
        DG_STRIDES=dg.stride()[1:],
        DA_STRIDES=dA_partial.stride(),
        DDT_STRIDES=ddt_partial.stride(),
        D=head_dim,
        BT=metadata.chunk_size,
        BD=block_dim,
        num_warps=4,
        num_stages=2,
    )
    return dg, dA_partial.sum((0, 2)), ddt_partial.sum(0)
