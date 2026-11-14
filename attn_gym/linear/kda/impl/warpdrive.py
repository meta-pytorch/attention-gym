# SPDX-License-Identifier: BSD-3-Clause

"""Autograd integration for the warpdrive KDA kernels."""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.span import pack_dense_batch
from attn_gym.linear._delta_rule.validation import require_inference_only, resolve_scale
from attn_gym.linear.kda.impl.warpdrive_ops import (
    chunk_warpdrive_packed_bwd_with_exit_cotangent_op,
    chunk_warpdrive_packed_bwd_with_state_op,
    chunk_warpdrive_packed_fwd_op,
    chunk_warpdrive_packed_fwd_paged_op,
    chunk_warpdrive_packed_fwd_with_initial_state_op,
    chunk_warpdrive_packed_fwd_with_state_op,
    chunk_warpdrive_packed_local_bwd_op,
    validate_warpdrive_available,
)
from attn_gym.linear.kda.validation import cast_to_kernel_dtypes, validate_warpdrive_fastmath


def normalize_output_grad(d_output: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Materialize the contiguous output cotangent required by the warpdrive operator ABI."""
    return torch.zeros_like(value) if d_output is None else d_output.contiguous()


class ChunkKdaWarpdrive(torch.autograd.Function):
    """Attach warpdrive's BT16 backward to the warpdrive forward.

    The kernel choice depends only on which states the call carries, never on the span length,
    so a document's bits do not depend on the span it is packed into. With an entry
    state the backward carries the state cotangent in FP32 across chunks, chain pieces and work
    items (``test/linear/test_delta_rule_state_precision.py``).
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        value: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        scale: float,
        split_backward: bool,
        split_forward: bool,
        output_final_state: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ctx.split_backward = split_backward
        ctx.scale = scale
        if output_final_state:
            # A zero entry state has no cotangent: the backward then skips every state access.
            entry_state = initial_state
            if entry_state is None:
                entry_state = torch.zeros(
                    cu_seqlens.shape[0] - 1,
                    q.shape[2],
                    value.shape[-1],
                    q.shape[3],
                    dtype=torch.float32,
                    device=q.device,
                )
            output, final_state = chunk_warpdrive_packed_fwd_with_state_op(
                q, k, value, gate, beta, entry_state, cu_seqlens, scale
            )
        elif initial_state is None:
            output = chunk_warpdrive_packed_fwd_op(
                q, k, value, gate, beta, cu_seqlens, split_forward, scale
            )
        else:
            output = chunk_warpdrive_packed_fwd_with_initial_state_op(
                q, k, value, gate, beta, initial_state, cu_seqlens, scale
            )
        ctx.save_for_backward(q, k, value, gate, beta, initial_state, cu_seqlens)
        ctx.set_materialize_grads(False)
        if output_final_state:
            return output, final_state
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output: torch.Tensor | None, d_final_state: torch.Tensor | None = None):
        q, k, value, gate, beta, initial_state, cu_seqlens = ctx.saved_tensors
        d_output = normalize_output_grad(d_output, value)
        if initial_state is None and d_final_state is None:
            gradients = chunk_warpdrive_packed_local_bwd_op(
                q, k, value, gate, beta, d_output, cu_seqlens, ctx.split_backward, ctx.scale
            )
            gradients = (*gradients, None)
        elif initial_state is None:
            gradients = chunk_warpdrive_packed_bwd_with_exit_cotangent_op(
                q, k, value, gate, beta, d_output, cu_seqlens, d_final_state, ctx.scale
            )
            gradients = (*gradients, None)
        else:
            gradients = chunk_warpdrive_packed_bwd_with_state_op(
                q,
                k,
                value,
                gate,
                beta,
                d_output,
                cu_seqlens,
                initial_state,
                d_final_state,
                ctx.scale,
            )
        return *gradients, None, None, None, None, None


def validate_warpdrive_constraints(q: torch.Tensor, value: torch.Tensor) -> None:
    """Validate the device and head dimensions the warpdrive KDA kernels support.

    Dtypes and layouts are normalized instead: ``cast_to_kernel_dtypes`` applies the fused
    dtype policy and the registered operators copy operands their launchers cannot address.
    """
    if not q.is_cuda:
        raise ValueError("the warpdrive KDA backend requires CUDA tensors")
    if q.shape[-1] != 128 or value.shape[-1] != 128:
        raise ValueError("the warpdrive KDA backend requires K=V=128")


def chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    fastmath: bool = True,
    split_backward: bool = False,
    split_forward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the optional warpdrive implementation behind the shared ``chunk_kda`` contract."""
    validate_warpdrive_fastmath(fastmath)
    scale = resolve_scale(scale, q.shape[-1])
    if split_backward and (initial_state is not None or output_final_state):
        raise ValueError("split_backward currently requires a no-state call")
    if split_forward and (initial_state is not None or output_final_state):
        raise ValueError("split_forward currently requires a no-state call")
    if not torch.compiler.is_compiling():
        validate_warpdrive_available(q)
    validate_warpdrive_constraints(q, value)

    output_dtype, output_shape = q.dtype, value.shape
    q, k, value, gate, beta = cast_to_kernel_dtypes(q, k, value, gate, beta)
    if cu_seqlens is None:
        cu_seqlens, q, k, value, gate, beta = pack_dense_batch(q, k, value, gate, beta)
    if initial_state is not None:
        initial_state = initial_state.float()
        if initial_state.stride(-1) != 1:
            initial_state = initial_state.contiguous()
    result = ChunkKdaWarpdrive.apply(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
        split_backward,
        split_forward,
        output_final_state,
    )
    output, final_state = result if output_final_state else (result, None)
    return output.reshape(output_shape).to(output_dtype), final_state


def paged_chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Advance selected state-cache slots with the warpdrive KDA kernel."""
    require_inference_only(
        (q, k, value, gate, beta, state_cache),
        "paged_chunk_kda is inference-only; call under torch.no_grad() or torch.inference_mode()",
    )
    if not torch.compiler.is_compiling():
        validate_warpdrive_available(q)
    validate_warpdrive_constraints(q, value)

    output_dtype, output_shape = q.dtype, value.shape
    q, k, value, gate, beta = cast_to_kernel_dtypes(q, k, value, gate, beta)
    if cu_seqlens is None:
        cu_seqlens, q, k, value, gate, beta = pack_dense_batch(q, k, value, gate, beta)
    output = chunk_warpdrive_packed_fwd_paged_op(
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
    return output.reshape(output_shape).to(output_dtype)


__all__ = ["chunk_forward", "paged_chunk_forward"]
