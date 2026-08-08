# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset


@triton.autotune(
    configs=[triton.Config({}, num_warps=4)],
    key=["D"],
)
@triton.jit
def l2norm_fwd_kernel1(
    x,
    y,
    rstd,
    eps,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    RSTD_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_d = tl.arange(0, BD)
    mask = o_d < D

    b_x = tl.load(x + ptr_offset((i_t, o_d), X_STRIDES), mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + ptr_offset((i_t, o_d), Y_STRIDES), b_y, mask=mask)
    tl.store(rstd + ptr_offset((i_t,), RSTD_STRIDES), b_rstd)


@triton.autotune(
    configs=[
        triton.Config({"BT": 16}, num_warps=4, num_stages=3),
    ],
    key=["D", "NB"],
)
@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    T,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    RSTD_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BD)
    m_t = o_t < T
    mask = m_t[:, None] & (o_d[None, :] < D)

    b_x = tl.load(
        x + ptr_offset((o_t[:, None], o_d[None, :]), X_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(
        y + ptr_offset((o_t[:, None], o_d[None, :]), Y_STRIDES),
        b_y.to(y.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        rstd + ptr_offset((o_t,), RSTD_STRIDES),
        b_rstd.to(rstd.dtype.element_ty),
        mask=m_t,
    )
