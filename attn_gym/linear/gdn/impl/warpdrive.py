# SPDX-License-Identifier: BSD-3-Clause

"""Autograd integration for the optional scalar-GDN warpdrive implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from attn_gym.linear._delta_rule.span import pack_dense_batch
from attn_gym.linear._delta_rule.validation import require_inference_only
from attn_gym.linear._delta_rule.wd_support import HEAD_DIMS, IO_DTYPE_NAMES
from attn_gym.linear.gdn.impl.warpdrive_ops import (
    chunk_gdn_warpdrive_packed_bwd_op,
    chunk_gdn_warpdrive_packed_bwd_with_state_op,
    chunk_gdn_warpdrive_packed_fwd_op,
    chunk_gdn_warpdrive_packed_fwd_paged_op,
    chunk_gdn_warpdrive_packed_fwd_with_initial_state_op,
    chunk_gdn_warpdrive_packed_fwd_with_state_op,
    validate_warpdrive_available,
)


def validate_warpdrive_constraints(q: Tensor, k: Tensor, value: Tensor) -> None:
    """Validate the Q/K/V dtypes and head dimensions the scalar-GDN warpdrive kernels support.

    ``chunk_gdn`` casts gate, beta, and entry states to FP32 before dispatch
    (``cast_to_kernel_dtypes``), and the registered operators copy operands their launchers
    cannot address.
    """
    if q.shape[-1] != value.shape[-1] or q.shape[-1] not in HEAD_DIMS:
        raise ValueError(f"the warpdrive GDN backend requires K=V in {HEAD_DIMS}")
    if q.dtype not in IO_DTYPE_NAMES or k.dtype != q.dtype or value.dtype != q.dtype:
        raise TypeError("q, k, and value must share dtype float16 or bfloat16")


class ChunkGdnWarpdrivePacked(torch.autograd.Function):
    """Attach autograd to packed scalar-GDN warpdrive calls with optional recurrent state."""

    @staticmethod
    def forward(
        ctx,
        q: Tensor,
        k: Tensor,
        value: Tensor,
        gate: Tensor,
        beta: Tensor,
        initial_state: Tensor | None,
        cu_seqlens: Tensor,
        scale: float,
        output_final_state: bool,
        split_backward: bool,
        split_forward: bool,
    ) -> Tensor | tuple[Tensor, Tensor]:
        ctx.has_initial_state = initial_state is not None
        ctx.scale = scale
        ctx.split_backward = split_backward
        if initial_state is None:
            output = chunk_gdn_warpdrive_packed_fwd_op(
                q, k, value, gate, beta, cu_seqlens, split_forward, scale
            )
            ctx.save_for_backward(q, k, value, gate, beta, cu_seqlens)
        elif output_final_state:
            output, final_state = chunk_gdn_warpdrive_packed_fwd_with_state_op(
                q,
                k,
                value,
                gate,
                beta,
                initial_state,
                cu_seqlens,
                scale,
            )
            ctx.save_for_backward(q, k, value, gate, beta, initial_state, cu_seqlens)
        else:
            output = chunk_gdn_warpdrive_packed_fwd_with_initial_state_op(
                q,
                k,
                value,
                gate,
                beta,
                initial_state,
                cu_seqlens,
                scale,
            )
            ctx.save_for_backward(q, k, value, gate, beta, initial_state, cu_seqlens)
        ctx.set_materialize_grads(False)
        if output_final_state:
            assert initial_state is not None
            return output, final_state
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx,
        d_output: Tensor | None,
        d_final_state: Tensor | None = None,
    ) -> tuple[Tensor | None, ...]:
        if not ctx.has_initial_state:
            q, k, value, gate, beta, cu_seqlens = ctx.saved_tensors
            gradients = chunk_gdn_warpdrive_packed_bwd_op(
                q,
                k,
                value,
                gate,
                beta,
                torch.zeros_like(value) if d_output is None else d_output,
                cu_seqlens,
                ctx.split_backward,
                ctx.scale,
            )
            return *gradients, None, None, None, None, None, None

        q, k, value, gate, beta, initial_state, cu_seqlens = ctx.saved_tensors
        return (
            *chunk_gdn_warpdrive_packed_bwd_with_state_op(
                q,
                k,
                value,
                gate,
                beta,
                torch.zeros_like(value) if d_output is None else d_output,
                initial_state,
                d_final_state,
                cu_seqlens,
                ctx.scale,
            ),
            None,
            None,
            None,
            None,
            None,
        )


def chunk_forward(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor | None = None,
    *,
    cu_seqlens: Tensor | None = None,
    scale: float,
    output_final_state: bool = False,
    split_backward: bool = False,
    split_forward: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Run scalar GDN through the warpdrive backend behind the shared chunk contract.

    Gate, beta, and the entry state arrive in FP32 (``chunk_gdn`` casts them).
    """
    if not q.is_cuda:
        raise ValueError("the warpdrive GDN backend requires CUDA tensors")
    for name, enabled in (("split_backward", split_backward), ("split_forward", split_forward)):
        if enabled and (initial_state is not None or output_final_state):
            raise ValueError(f"{name} currently requires a no-state call")
    if not torch.compiler.is_compiling():
        validate_warpdrive_available(q)
    validate_warpdrive_constraints(q, k, value)

    batch = q.shape[0]
    output_shape = value.shape
    if cu_seqlens is None:
        cu_seqlens, q, k, value, gate, beta = pack_dense_batch(q, k, value, gate, beta)
    elif batch != 1:
        raise ValueError("packed cu_seqlens require q to have batch size one")

    if initial_state is None and output_final_state:
        initial_state = torch.zeros(
            cu_seqlens.shape[0] - 1,
            value.shape[2],
            value.shape[3],
            q.shape[3],
            dtype=torch.float32,
            device=q.device,
        )

    result = ChunkGdnWarpdrivePacked.apply(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
        output_final_state,
        split_backward,
        split_forward,
    )
    if output_final_state:
        output, final_state = result
        return output.reshape(output_shape), final_state
    return result.reshape(output_shape), None


def paged_chunk_forward(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    state_cache: Tensor,
    state_indices: Tensor,
    *,
    cu_seqlens: Tensor | None,
    has_initial_state: Tensor | None,
    scale: float,
) -> Tensor:
    """Advance selected ``state_cache`` slots in place with the warpdrive chunk kernel."""
    if not q.is_cuda:
        raise ValueError("the warpdrive GDN backend requires CUDA tensors")
    if not torch.compiler.is_compiling():
        validate_warpdrive_available(q)
    require_inference_only(
        (q, k, value, gate, beta, state_cache),
        "paged_chunk_gdn is inference-only; call under torch.no_grad() or torch.inference_mode()",
    )
    validate_warpdrive_constraints(q, k, value)

    batch = q.shape[0]
    output_shape = value.shape
    if cu_seqlens is None:
        cu_seqlens, q, k, value, gate, beta = pack_dense_batch(q, k, value, gate, beta)
    elif batch != 1:
        raise ValueError("packed cu_seqlens require q to have batch size one")

    output = chunk_gdn_warpdrive_packed_fwd_paged_op(
        q,
        k,
        value,
        gate,
        beta,
        state_cache,
        state_indices,
        has_initial_state,
        cu_seqlens,
        scale,
    )
    return output.reshape(output_shape)


__all__ = ["chunk_forward", "paged_chunk_forward", "validate_warpdrive_constraints"]
