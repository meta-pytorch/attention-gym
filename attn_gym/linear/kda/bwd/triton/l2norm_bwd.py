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
# and contracting with the upstream gradient dy, then substituting x = y / rstd,
# collapses to the row-local rule
#     dx = rstd * (dy - y * <dy, y>),      <dy, y> = sum_d dy_d * y_d.
# Only y and rstd (both produced by the forward) are needed -- x is not.

from __future__ import annotations

import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.utils import autotune_cache_kwargs

_WARPS = [1, 2, 4, 8, 16, 32]
_BT_LIST = [8, 16, 32, 64, 128]


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in _WARPS],
    key=["D"],
    **autotune_cache_kwargs,
)
@triton.jit
def l2norm_bwd_kernel1(
    y,
    rstd,
    dy,
    dx,
    eps,
    D,
    Y_STRIDES: tl.constexpr,
    RSTD_STRIDES: tl.constexpr,
    DY_STRIDES: tl.constexpr,
    DX_STRIDES: tl.constexpr,
    BD: tl.constexpr,
):
    # One program per row; the entire feature vector fits in a single [BD] tile.
    i_t = tl.program_id(0).to(tl.int64)
    o_d = tl.arange(0, BD)
    mask = o_d < D

    b_y = tl.load(y + ptr_offset((i_t, o_d), Y_STRIDES), mask=mask, other=0.0).to(tl.float32)
    b_dy = tl.load(dy + ptr_offset((i_t, o_d), DY_STRIDES), mask=mask, other=0.0).to(tl.float32)
    b_rstd = tl.load(rstd + ptr_offset((i_t,), RSTD_STRIDES)).to(tl.float32)

    b_dot = tl.sum(b_dy * b_y)
    b_dx = b_rstd * (b_dy - b_y * b_dot)
    tl.store(
        dx + ptr_offset((i_t, o_d), DX_STRIDES),
        b_dx.to(dx.dtype.element_ty),
        mask=mask,
    )


@triton.autotune(
    configs=[triton.Config({"BT": bt}, num_warps=w) for w in [1, 2, 4, 8, 16] for bt in _BT_LIST],
    key=["D", "NB"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def l2norm_bwd_kernel(
    y,
    rstd,
    dy,
    dx,
    eps,
    T,
    Y_STRIDES: tl.constexpr,
    RSTD_STRIDES: tl.constexpr,
    DY_STRIDES: tl.constexpr,
    DX_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB: tl.constexpr,
    BT: tl.constexpr,
):
    # One program per [BT] block of rows; each row is normalized over [BD].
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_d[None, :] < D)

    b_y = tl.load(
        y + ptr_offset((o_t[:, None], o_d[None, :]), Y_STRIDES),
        mask=m_x,
        other=0.0,
    ).to(tl.float32)
    b_dy = tl.load(
        dy + ptr_offset((o_t[:, None], o_d[None, :]), DY_STRIDES),
        mask=m_x,
        other=0.0,
    ).to(tl.float32)
    b_rstd = tl.load(rstd + ptr_offset((o_t,), RSTD_STRIDES), mask=m_t, other=0.0).to(tl.float32)

    b_dot = tl.sum(b_dy * b_y, 1)
    b_dx = b_rstd[:, None] * (b_dy - b_y * b_dot[:, None])
    tl.store(
        dx + ptr_offset((o_t[:, None], o_d[None, :]), DX_STRIDES),
        b_dx.to(dx.dtype.element_ty),
        mask=m_x,
    )
