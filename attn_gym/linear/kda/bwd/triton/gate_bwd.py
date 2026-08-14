# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Backward for the KDA per-channel gate (the pre-cumsum gate value).
#
# Forward, per (token t, head h, channel d):
#     z = g + dt_bias                    (dt_bias broadcast over tokens, per h,d)
#     A = exp(A_log)                     (A_log is a per-head scalar)
#   no lower bound : yg = -A * softplus(z)
#   lower bound LB : yg =  LB * sigmoid(A * z)
#
# Given the upstream gradient dyg = dL/dyg:
#   no bound :  dyg/dz    = -A * sigmoid(z)         (softplus'(z) = sigmoid(z))
#               dyg/dA_log = yg                      (since A * dsoftplus/dA_log = A)
#   bounded  :  s = sigmoid(A z)
#               dyg/dz    = LB * A * s * (1 - s)
#               dyg/dA_log = LB * A * z * s (1 - s) = dg * z
# Therefore dg = dyg * (dyg/dz), and the per-head A_log gradient is a full
# reduction of dyg * (dyg/dA_log) over every token and channel. Each program
# emits its block's partial sum into dA[i_t, i_h]; the caller reduces over the
# time-block axis. The dt_bias gradient is sum_t dg, also reduced by the caller.

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.utils import autotune_cache_kwargs, input_guard

NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]


@triton.jit
def _softplus(x):
    # log(1 + exp(x)), stable for large x where exp(x) would overflow.
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=st) for w in NUM_WARPS_AUTOTUNE for st in [2, 3]
    ],
    key=["H", "D"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_bwd_kernel(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    lower_bound,
    T,
    G_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    DYG_STRIDES: tl.constexpr,
    DG_STRIDES: tl.constexpr,
    DA_STRIDES: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m = m_t[:, None] & (o_d[None, :] < D)

    b_g = tl.load(
        g + ptr_offset((o_t[:, None], i_h, o_d[None, :]), G_STRIDES),
        mask=m,
        other=0.0,
    ).to(tl.float32)
    b_dy = tl.load(
        dyg + ptr_offset((o_t[:, None], i_h, o_d[None, :]), DYG_STRIDES),
        mask=m,
        other=0.0,
    ).to(tl.float32)
    if HAS_BIAS:
        b_g += tl.load(
            dt_bias + ptr_offset((i_h, o_d), DT_BIAS_STRIDES),
            mask=o_d < D,
            other=0.0,
        ).to(tl.float32)

    b_alog = tl.load(A_log + ptr_offset((i_h,), A_LOG_STRIDES)).to(tl.float32)
    if USE_LOWER_BOUND:
        b_a = tl.exp(b_alog)
        b_s = tl.sigmoid(b_a * b_g)
        b_dg = b_dy * (lower_bound * b_a * b_s * (1.0 - b_s))
        # dyg/dA_log = dg * z, reduced over the whole block.
        b_dalog = tl.sum(tl.sum(b_dg * b_g, 1), 0)
    else:
        b_a = -tl.exp(b_alog)
        b_yg = b_a * _softplus(b_g)
        b_dg = b_dy * (b_a * tl.sigmoid(b_g))
        # dyg/dA_log = yg; masked lanes carry b_dy == 0 and drop out of the sum.
        b_dalog = tl.sum(tl.sum(b_dy * b_yg, 1), 0)

    tl.store(
        dg + ptr_offset((o_t[:, None], i_h, o_d[None, :]), DG_STRIDES),
        b_dg.to(dg.dtype.element_ty),
        mask=m,
    )
    tl.store(dA + ptr_offset((i_t, i_h), DA_STRIDES), b_dalog)


@triton.jit(do_not_specialize=["T"])
def kda_gate_bwd_ragged_kernel(
    g,
    A_log,
    dt_bias,
    dyg,
    dg,
    dA,
    lower_bound,
    T,
    G_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    DYG_STRIDES: tl.constexpr,
    DG_STRIDES: tl.constexpr,
    DA_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Differentiate the bounded gate with a fixed graph-capture-safe schedule."""
    i_t, i_h = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)

    token = i_t * BT + tl.arange(0, BT)
    channel = tl.arange(0, BD)
    mask = (token[:, None] < T) & (channel[None, :] < D)
    gate_input = tl.load(
        g + ptr_offset((token[:, None], i_h, channel[None, :]), G_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    d_gate = tl.load(
        dyg + ptr_offset((token[:, None], i_h, channel[None, :]), DYG_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    gate_input += tl.load(
        dt_bias + ptr_offset((i_h, channel), DT_BIAS_STRIDES),
        mask=channel < D,
        other=0.0,
    ).to(tl.float32)

    decay = tl.exp(tl.load(A_log + ptr_offset((i_h,), A_LOG_STRIDES)).to(tl.float32))
    sigmoid = tl.sigmoid(decay * gate_input)
    d_raw_gate = d_gate * (lower_bound * decay * sigmoid * (1.0 - sigmoid))
    dA_log = tl.sum(tl.sum(d_raw_gate * gate_input, 1), 0)

    tl.store(
        dg + ptr_offset((token[:, None], i_h, channel[None, :]), DG_STRIDES),
        d_raw_gate,
        mask=mask,
    )
    tl.store(dA + ptr_offset((i_t, i_h), DA_STRIDES), dA_log)


@input_guard(no_guard_contiguous=True)
def kda_gate_bwd_ragged(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_gate: torch.Tensor,
    *,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the pointwise bounded gate with FP32 parameter reductions."""
    if g.ndim != 3:
        raise ValueError(f"g must have shape [T, H, D], got {tuple(g.shape)}")
    if d_gate.shape != g.shape:
        raise ValueError(f"d_gate must have shape {tuple(g.shape)}, got {tuple(d_gate.shape)}")
    tokens, heads, head_dim = g.shape
    if A_log.shape != (heads,) or dt_bias.shape != (heads, head_dim):
        raise ValueError("A_log or dt_bias shape does not match g")

    block_tokens = 32
    time_blocks = triton.cdiv(tokens, block_tokens)
    dg = torch.empty_like(g, dtype=torch.float32)
    dA_partial = A_log.new_empty((time_blocks, heads), dtype=torch.float32)
    kda_gate_bwd_ragged_kernel[(time_blocks, heads)](
        g,
        A_log,
        dt_bias,
        d_gate,
        dg,
        dA_partial,
        lower_bound,
        tokens,
        G_STRIDES=g.stride(),
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=dt_bias.stride(),
        DYG_STRIDES=d_gate.stride(),
        DG_STRIDES=dg.stride(),
        DA_STRIDES=dA_partial.stride(),
        D=head_dim,
        BT=block_tokens,
        BD=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=3,
    )
    return dg, dA_partial.sum(0), dg.sum(0)
