# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.cute.utils import get_device_properties
from attn_gym._backends.triton.utils import ptr_offset

# A Triton pointer does not carry tensor shape or stride metadata. The registered-operator
# boundary keeps runtime stride tuples opaque to Dynamo while preserving ptr_offset indexing.


@triton.jit
def l2norm_fwd_kernel(
    x,
    y,
    rstd,
    eps,
    N_ROWS,
    cu_seqlens,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    RSTD_STRIDES: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BT: tl.constexpr,
):
    i_row = tl.program_id(0).to(tl.int64)
    if IS_VARLEN:
        active_rows = tl.load(cu_seqlens + NUM_SEQUENCES).to(tl.int64) * H
        if i_row * BT >= active_rows:
            return
        N_ROWS = active_rows
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


def _l2norm_launch_config(rows: int, tuned_major: int | None) -> tuple[int, int]:
    """Select the rows per program and warp count for L2 normalization."""
    if tuned_major == 10:
        if rows <= 8:
            return 1, 1
        if rows <= 32:
            return 1, 4
        if rows <= 512:
            return 4, 4
        return 8, 4
    if tuned_major != 9 or rows > 2048:
        return 16, 4
    if rows <= 9:
        return 1, 1
    return 4, 2


torch.library.define(
    "attn_gym::kda_l2norm_fwd",
    "(Tensor x, float eps, Tensor? cu_seqlens=None) -> (Tensor, Tensor)",
)


def _l2norm_fwd_cuda(
    x: torch.Tensor,
    eps: float,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch L2Norm using runtime shape, strides, and optional packed metadata."""
    _, tokens, heads, head_dim = x.shape
    # Compact outputs enumerate rows in the same batch-token-head order reconstructed by
    # the kernel, regardless of the input's physical strides.
    rows = x.numel() // head_dim
    output = torch.empty(rows, head_dim, dtype=x.dtype, device=x.device)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    block_dim = triton.next_power_of_2(head_dim)
    major = get_device_properties(x.device).major
    has_decode_metadata_shape = cu_seqlens is not None and cu_seqlens.shape[0] == tokens + 1
    measured_decode_shape = (
        x.dtype is torch.bfloat16 and head_dim == 128 and has_decode_metadata_shape
    )
    tuned_major = None
    if measured_decode_shape and (
        major == 9 or (major == 10 and heads in (2, 4, 8, 16) and tokens <= 256)
    ):
        tuned_major = major
    block_tokens, num_warps = _l2norm_launch_config(rows, tuned_major)
    l2norm_fwd_kernel[(triton.cdiv(rows, block_tokens),)](
        x,
        output,
        rstd,
        eps,
        rows,
        cu_seqlens,
        X_STRIDES=x.stride(),
        Y_STRIDES=output.stride(),
        RSTD_STRIDES=rstd.stride(),
        T=tokens,
        H=heads,
        D=head_dim,
        BD=block_dim,
        NUM_SEQUENCES=0 if cu_seqlens is None else cu_seqlens.shape[0] - 1,
        IS_VARLEN=cu_seqlens is not None,
        BT=block_tokens,
        num_warps=num_warps,
        num_stages=3,
    )
    return output.view_as(x), rstd


torch.library.impl("attn_gym::kda_l2norm_fwd", "CUDA", _l2norm_fwd_cuda)


@torch.library.register_fake("attn_gym::kda_l2norm_fwd")
def _l2norm_fwd_fake(
    x: torch.Tensor,
    eps: float,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe compact output metadata without reading the packed endpoint."""
    del eps, cu_seqlens
    rows = x.numel() // x.shape[-1]
    return (
        torch.empty_like(x, memory_format=torch.contiguous_format),
        torch.empty(rows, dtype=torch.float32, device=x.device),
    )


torch.library.define(
    "attn_gym::kda_l2norm_bwd",
    "(Tensor output, Tensor rstd, Tensor d_output, Tensor? cu_seqlens=None) -> Tensor",
)


def _l2norm_bwd_cuda(
    output: torch.Tensor,
    rstd: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
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
        cu_seqlens,
        Y_STRIDES=output.stride(),
        RSTD_STRIDES=rstd.stride(),
        DY_STRIDES=d_output.stride(),
        DX_STRIDES=d_input.stride(),
        TOKENS=tokens,
        HEADS=heads,
        D=head_dim,
        BD=block_dim,
        NB=triton.cdiv(rows, 16),
        NUM_SEQUENCES=0 if cu_seqlens is None else cu_seqlens.shape[0] - 1,
        IS_VARLEN=cu_seqlens is not None,
    )
    return d_input


torch.library.impl("attn_gym::kda_l2norm_bwd", "CUDA", _l2norm_bwd_cuda)


@torch.library.register_fake("attn_gym::kda_l2norm_bwd")
def _l2norm_bwd_fake(
    output: torch.Tensor,
    rstd: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Describe input-gradient metadata without reading the packed endpoint."""
    del rstd, d_output, cu_seqlens
    return torch.empty_like(output)


_l2norm_fwd_op = torch.ops.attn_gym.kda_l2norm_fwd.default
_l2norm_bwd_op = torch.ops.attn_gym.kda_l2norm_bwd.default


class _L2Norm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        eps: float,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        output, rstd = _l2norm_fwd_op(x, eps, cu_seqlens)
        output_2d = output.view(-1, x.shape[-1])
        if cu_seqlens is None:
            ctx.save_for_backward(output_2d, rstd)
        else:
            ctx.save_for_backward(output_2d, rstd, cu_seqlens)
        ctx.input_shape = x.shape
        ctx.is_ragged = cu_seqlens is not None
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output: torch.Tensor):
        output, rstd, *metadata = ctx.saved_tensors
        cu_seqlens = metadata[0] if ctx.is_ragged else None
        d_input = _l2norm_bwd_op(output, rstd, d_output, cu_seqlens)
        return d_input.view(ctx.input_shape), None, None


def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    *,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize rows, optionally stopping at a packed device endpoint.

    With ``cu_seqlens``, only rows before ``cu_seqlens[-1]`` are defined in the
    output and input gradient. Packed callers must hide the inactive suffix from
    later consumers and parameter reductions.
    """
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
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError("cu_seqlens require packed x with batch size one")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
        if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous int32")
        if cu_seqlens.device != x.device:
            raise ValueError("cu_seqlens must be on the same device as x")
    return _L2Norm.apply(x, eps, cu_seqlens)


__all__ = ["l2norm"]
