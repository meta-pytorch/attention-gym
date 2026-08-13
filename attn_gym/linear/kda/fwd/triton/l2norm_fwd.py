# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset

# A Triton pointer does not carry tensor shape or stride metadata. The registered-operator
# boundary keeps runtime stride tuples opaque to Dynamo while preserving ptr_offset indexing.


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
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    i_bt = i_row // H
    o_d = tl.arange(0, BD).to(tl.int64)
    mask = o_d < D
    # Inductor passes Python floats as fp64; keep the reduction in fp32.
    eps = eps.to(tl.float32)

    b_x = tl.load(
        x
        + ptr_offset(
            (i_bt // T, i_bt % T, i_row % H, o_d),
            X_STRIDES,
        ),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x) + eps)
    b_y = b_x * b_rstd
    tl.store(y + ptr_offset((i_row, o_d), Y_STRIDES), b_y, mask=mask)
    tl.store(rstd + ptr_offset((i_row,), RSTD_STRIDES), b_rstd)


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
    N_ROWS,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    RSTD_STRIDES: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NB,
    BT: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    o_row = i_row * BT + tl.arange(0, BT)
    o_bt = o_row // H
    o_d = tl.arange(0, BD).to(tl.int64)
    m_row = o_row < N_ROWS
    mask = m_row[:, None] & (o_d[None, :] < D)
    # Inductor passes Python floats as fp64; keep the reduction in fp32.
    eps = eps.to(tl.float32)

    b_x = tl.load(
        x
        + ptr_offset(
            (
                (o_bt // T)[:, None],
                (o_bt % T)[:, None],
                (o_row % H)[:, None],
                o_d[None, :],
            ),
            X_STRIDES,
        ),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    b_rstd = 1 / tl.sqrt(tl.sum(b_x * b_x, 1) + eps)
    b_y = b_x * b_rstd[:, None]

    tl.store(
        y + ptr_offset((o_row[:, None], o_d[None, :]), Y_STRIDES),
        b_y.to(y.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        rstd + ptr_offset((o_row,), RSTD_STRIDES),
        b_rstd.to(rstd.dtype.element_ty),
        mask=m_row,
    )


torch.library.define("attn_gym::kda_l2norm_fwd", "(Tensor x, float eps) -> (Tensor, Tensor)")


def _l2norm_fwd_cuda(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch L2Norm using runtime shape and stride metadata."""
    _, tokens, heads, head_dim = x.shape
    # Compact outputs enumerate rows in the same batch-token-head order reconstructed by
    # the kernel, regardless of the input's physical strides.
    rows = x.numel() // head_dim
    output = torch.empty(rows, head_dim, dtype=x.dtype, device=x.device)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    block_dim = triton.next_power_of_2(head_dim)
    grid = lambda meta: (triton.cdiv(rows, meta["BT"]),)
    l2norm_fwd_kernel[grid](
        x,
        output,
        rstd,
        eps,
        rows,
        X_STRIDES=x.stride(),
        Y_STRIDES=output.stride(),
        RSTD_STRIDES=rstd.stride(),
        T=tokens,
        H=heads,
        D=head_dim,
        BD=block_dim,
        NB=triton.cdiv(rows, 16),
    )
    return output.view_as(x), rstd


torch.library.impl("attn_gym::kda_l2norm_fwd", "CUDA", _l2norm_fwd_cuda)


@torch.library.register_fake("attn_gym::kda_l2norm_fwd")
def _l2norm_fwd_fake(x: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe compact output metadata without launching Triton."""
    del eps
    rows = x.numel() // x.shape[-1]
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty(rows, dtype=torch.float32, device=x.device),
    )


torch.library.define(
    "attn_gym::kda_l2norm_bwd",
    "(Tensor output, Tensor rstd, Tensor d_output) -> Tensor",
)


def _l2norm_bwd_cuda(
    output: torch.Tensor,
    rstd: torch.Tensor,
    d_output: torch.Tensor,
) -> torch.Tensor:
    """Launch L2Norm backward behind the same opaque stride boundary."""
    from attn_gym.linear.kda.bwd.triton.l2norm_bwd import l2norm_bwd_kernel

    rows, head_dim = output.shape
    _, tokens, heads, _ = d_output.shape
    d_input = torch.empty_like(output)
    block_dim = triton.next_power_of_2(head_dim)
    grid = lambda meta: (triton.cdiv(rows, meta["BT"]),)
    l2norm_bwd_kernel[grid](
        output,
        rstd,
        d_output,
        d_input,
        rows,
        Y_STRIDES=output.stride(),
        RSTD_STRIDES=rstd.stride(),
        DY_STRIDES=d_output.stride(),
        DX_STRIDES=d_input.stride(),
        TOKENS=tokens,
        HEADS=heads,
        D=head_dim,
        BD=block_dim,
        NB=triton.cdiv(rows, 16),
    )
    return d_input


torch.library.impl("attn_gym::kda_l2norm_bwd", "CUDA", _l2norm_bwd_cuda)


@torch.library.register_fake("attn_gym::kda_l2norm_bwd")
def _l2norm_bwd_fake(
    output: torch.Tensor,
    rstd: torch.Tensor,
    d_output: torch.Tensor,
) -> torch.Tensor:
    """Describe the compact input-gradient metadata."""
    del rstd, d_output
    return torch.empty_like(output)


_l2norm_fwd_op = torch.ops.attn_gym.kda_l2norm_fwd.default
_l2norm_bwd_op = torch.ops.attn_gym.kda_l2norm_bwd.default


class _L2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float) -> torch.Tensor:
        output, rstd = _l2norm_fwd_op(x, eps)
        ctx.save_for_backward(output.view(-1, x.shape[-1]), rstd)
        ctx.input_shape = x.shape
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output: torch.Tensor):
        output, rstd = ctx.saved_tensors
        d_input = _l2norm_bwd_op(output, rstd, d_output)
        return d_input.view(ctx.input_shape), None


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize each final-dimension row of a KDA ``[B, T, H, D]`` tensor."""
    if not x.is_cuda:
        raise ValueError("l2norm requires a CUDA tensor")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("l2norm requires float16, bfloat16, or float32 input")
    if x.ndim != 4:
        raise ValueError(f"x must have shape [B, T, H, D], got {tuple(x.shape)}")
    if x.shape[-1] < 1:
        raise ValueError(f"x must have a nonempty final dimension, got {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError(f"x must contain at least one row, got {tuple(x.shape)}")
    return _L2Norm.apply(x, eps)


__all__ = ["l2norm"]
