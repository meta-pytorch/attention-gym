# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import triton
import triton.language as tl


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
    D: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0).to(tl.int64)
    x += i_t * D
    y += i_t * D
    # Compute mean and variance
    cols = tl.arange(0, BD)
    mask = cols < D

    b_x = tl.load(x + cols, mask=mask, other=0.0).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + cols, b_y, mask=mask)
    tl.store(rstd + i_t, b_rstd)


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
    D: tl.constexpr,
    BD: tl.constexpr,
    NB,
    BT: tl.constexpr,
):
    i_t = tl.program_id(0)
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    p_rstd = tl.make_block_ptr(rstd, (T,), (1,), (i_t * BT,), (BT,), (0,))

    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))  # type: ignore
    tl.store(p_rstd, b_rstd.to(p_rstd.dtype.element_ty), boundary_check=(0,))  # type: ignore
