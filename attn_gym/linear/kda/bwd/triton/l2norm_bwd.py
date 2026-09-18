# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Backward for L2 normalization.
#
# Forward (per row, over the D feature channels):
#     rstd = 1 / sqrt(sum_d x_d^2 + eps)
#     y_d  = x_d * rstd
#
# Differentiating y w.r.t. x gives the Jacobian
#     dy_i/dx_j = rstd * delta_ij - rstd^3 * x_i * x_j ,
# and contracting with the upstream gradient dy gives the row-local rule
#     dx = rstd * (dy - y * <dy, y>),      <dy, y> = sum_d dy_d * y_d.
# y = x * rstd is recomputed in fp32 from the exact input rather than read back from the
# rounded forward output: for a radial dy the bracket cancels down to y * eps * rstd^2,
# which a bf16/fp16 y cannot resolve.

from __future__ import annotations

import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.utils import autotune_cache_kwargs

_BT_LIST = [8, 16, 32, 64, 128]


@triton.autotune(
    configs=[triton.Config({"BT": bt}, num_warps=w) for w in [1, 2, 4, 8, 16] for bt in _BT_LIST],
    # NB selects a tuned schedule, not a different binary for each row count.
    key=["D", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=["N_ROWS", "TOKENS", "NB", "NUM_SEQUENCES", "X_STRIDES", "DY_STRIDES"]
)
def l2norm_bwd_kernel(
    x,
    rstd,
    dy,
    dx,
    N_ROWS,
    cu_seqlens,
    X_STRIDES,
    RSTD_STRIDES: tl.constexpr,
    DY_STRIDES,
    DX_STRIDES: tl.constexpr,
    TOKENS,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB,
    NUM_SEQUENCES,
    IS_VARLEN: tl.constexpr,
    BT: tl.constexpr,
    X_ROW_STRIDE: tl.constexpr = 0,
    DY_ROW_STRIDE: tl.constexpr = 0,
):
    # One program per [BT] block of rows; each row is normalized over [BD].
    i_row = tl.program_id(0).to(tl.int64)
    if IS_VARLEN:
        active_rows = tl.load(cu_seqlens + NUM_SEQUENCES).to(tl.int64) * HEADS
        if i_row * BT >= active_rows:
            return
        N_ROWS = active_rows
    o_row = i_row * BT + tl.arange(0, BT)
    o_bt = o_row // HEADS
    o_d = tl.arange(0, BD).to(tl.int64)
    m_row = o_row < N_ROWS
    mask = m_row[:, None] & (o_d[None, :] < D)

    o_bthd = (
        (o_bt // TOKENS)[:, None],
        (o_bt % TOKENS)[:, None],
        (o_row % HEADS)[:, None],
        o_d[None, :],
    )
    if X_ROW_STRIDE:
        x_offsets = o_bt[:, None] * X_ROW_STRIDE + (o_row % HEADS)[:, None] * D + o_d[None, :]
    else:
        x_offsets = ptr_offset(o_bthd, X_STRIDES)
    if DY_ROW_STRIDE:
        dy_offsets = o_bt[:, None] * DY_ROW_STRIDE + (o_row % HEADS)[:, None] * D + o_d[None, :]
    else:
        dy_offsets = ptr_offset(o_bthd, DY_STRIDES)
    b_x = tl.load(x + x_offsets, mask=mask, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + dy_offsets, mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + ptr_offset((o_row,), RSTD_STRIDES), mask=m_row, other=0.0).to(
        tl.float32
    )

    b_y = b_x * b_rstd[:, None]
    b_dot = tl.sum(b_dy * b_y, 1)
    b_dx = b_rstd[:, None] * (b_dy - b_y * b_dot[:, None])
    tl.store(
        dx + ptr_offset((o_row[:, None], o_d[None, :]), DX_STRIDES),
        b_dx.to(dx.dtype.element_ty),
        mask=mask,
    )
