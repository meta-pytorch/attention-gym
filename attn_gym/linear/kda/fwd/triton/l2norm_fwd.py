# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch
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


class _L2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        rows = x.numel() // x.shape[-1]
        matrix = x.contiguous().view(rows, x.shape[-1])
        output = torch.empty_like(matrix)
        rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
        block_dim = triton.next_power_of_2(x.shape[-1])
        grid = lambda meta: (triton.cdiv(rows, meta["BT"]),)
        l2norm_fwd_kernel[grid](
            matrix,
            output,
            rstd,
            eps,
            rows,
            X_STRIDES=matrix.stride(),
            Y_STRIDES=output.stride(),
            RSTD_STRIDES=rstd.stride(),
            D=x.shape[-1],
            BD=block_dim,
            NB=triton.cdiv(rows, 16),
        )
        ctx.save_for_backward(output, rstd)
        ctx.input_shape = x.shape
        ctx.eps = eps
        return output.view_as(x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output: torch.Tensor):
        from attn_gym.linear.kda.bwd.triton.l2norm_bwd import l2norm_bwd_kernel

        output, rstd = ctx.saved_tensors
        rows, head_dim = output.shape
        d_output = d_output.contiguous().view_as(output)
        d_input = torch.empty_like(output)
        block_dim = triton.next_power_of_2(head_dim)
        grid = lambda meta: (triton.cdiv(rows, meta["BT"]),)
        l2norm_bwd_kernel[grid](
            output,
            rstd,
            d_output,
            d_input,
            ctx.eps,
            rows,
            Y_STRIDES=output.stride(),
            RSTD_STRIDES=rstd.stride(),
            DY_STRIDES=d_output.stride(),
            DX_STRIDES=d_input.stride(),
            D=head_dim,
            BD=block_dim,
            NB=triton.cdiv(rows, 16),
        )
        return d_input.view(ctx.input_shape), None


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply row-wise L2 normalization with a first-order Triton backward."""
    if not x.is_cuda:
        raise ValueError("l2norm requires a CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("l2norm requires float16, bfloat16, or float32 input")
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError(f"x must have a nonempty final dimension, got {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError(f"x must contain at least one row, got {tuple(x.shape)}")
    return _L2Norm.apply(x, eps)


__all__ = ["l2norm"]
