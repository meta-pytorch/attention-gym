# SPDX-License-Identifier: BSD-3-Clause

"""Autograd integration for the optional scalar-GDN cuDNN implementation."""

from __future__ import annotations

import torch
from torch import Tensor

from attn_gym.linear.gdn.impl.cudnn_ops import (
    chunk_gdn_cudnn_packed_bwd_op,
    chunk_gdn_cudnn_packed_bwd_with_state_op,
    chunk_gdn_cudnn_packed_fwd_op,
    chunk_gdn_cudnn_packed_fwd_paged_op,
    chunk_gdn_cudnn_packed_fwd_with_initial_state_op,
    chunk_gdn_cudnn_packed_fwd_with_state_op,
    validate_cudnn_available,
)

_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)
_CUDNN_DIM = 128


def _pack_dense(tensor: Tensor) -> Tensor:
    """Lower dense [B, T, ...] tensors to cuDNN's packed batch-one layout."""
    return tensor.reshape(1, -1, *tensor.shape[2:])


def _validate_cudnn_constraints(
    q: Tensor,
    k: Tensor,
    value: Tensor,
    gate: Tensor,
    beta: Tensor,
    initial_state: Tensor | None,
) -> None:
    """Validate the scalar-GDN restrictions imposed by the raw cuDNN launchers."""
    if not q.is_cuda:
        raise ValueError("the cuDNN GDN backend requires CUDA tensors")
    if q.shape[0] != 1 or q.shape[-1] != _CUDNN_DIM or value.shape[-1] != _CUDNN_DIM:
        raise ValueError("the cuDNN backend requires B=1 and K=V=128")
    if q.dtype not in _SUPPORTED_IO_DTYPES or k.dtype != q.dtype or value.dtype != q.dtype:
        raise TypeError("q, k, and value must share dtype float16 or bfloat16")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("gate and beta must be float32")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError("initial_state must be float32")


class ChunkGdnCudnnPacked(torch.autograd.Function):
    """Attach autograd to packed scalar-GDN cuDNN calls with optional recurrent state."""

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
    ) -> Tensor | tuple[Tensor, Tensor]:
        ctx.has_initial_state = initial_state is not None
        ctx.scale = scale
        if initial_state is None:
            output = chunk_gdn_cudnn_packed_fwd_op(q, k, value, gate, beta, cu_seqlens, scale)
            ctx.save_for_backward(q, k, value, gate, beta, cu_seqlens)
        elif output_final_state:
            output, final_state = chunk_gdn_cudnn_packed_fwd_with_state_op(
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
            output = chunk_gdn_cudnn_packed_fwd_with_initial_state_op(
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
            gradients = chunk_gdn_cudnn_packed_bwd_op(
                q,
                k,
                value,
                gate,
                beta,
                torch.zeros_like(value) if d_output is None else d_output,
                cu_seqlens,
                ctx.scale,
            )
            return *gradients, None, None, None, None

        q, k, value, gate, beta, initial_state, cu_seqlens = ctx.saved_tensors
        return (
            *chunk_gdn_cudnn_packed_bwd_with_state_op(
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
) -> tuple[Tensor, Tensor | None]:
    """Run scalar GDN through the cuDNN backend behind the shared chunk contract."""
    if not q.is_cuda:
        raise ValueError("the cuDNN GDN backend requires CUDA tensors")
    if not torch.compiler.is_compiling():
        validate_cudnn_available(q)

    gate, beta = (tensor.to(dtype=torch.float32) for tensor in (gate, beta))
    batch = q.shape[0]
    output_shape = value.shape
    if cu_seqlens is None:
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * q.shape[1]
    elif batch != 1:
        raise ValueError("packed cu_seqlens require q to have batch size one")
    else:
        cu_seqlens = cu_seqlens.contiguous()

    if batch > 1:
        q, k, value, gate, beta = (_pack_dense(tensor) for tensor in (q, k, value, gate, beta))

    if output_final_state and initial_state is None:
        initial_state = torch.zeros(
            cu_seqlens.shape[0] - 1,
            value.shape[2],
            _CUDNN_DIM,
            _CUDNN_DIM,
            dtype=torch.float32,
            device=q.device,
        )

    _validate_cudnn_constraints(q, k, value, gate, beta, initial_state)

    result = ChunkGdnCudnnPacked.apply(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
        output_final_state,
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
    """Advance selected ``state_cache`` slots in place with the cuDNN chunk kernel."""
    if not q.is_cuda:
        raise ValueError("the cuDNN GDN backend requires CUDA tensors")
    if not torch.compiler.is_compiling():
        validate_cudnn_available(q)
    tensors = (q, k, value, gate, beta, state_cache)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(
            "paged_chunk_gdn is inference-only; call under torch.no_grad() or "
            "torch.inference_mode()"
        )

    gate, beta = (tensor.to(dtype=torch.float32) for tensor in (gate, beta))
    batch = q.shape[0]
    output_shape = value.shape
    if cu_seqlens is None:
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * q.shape[1]
    elif batch != 1:
        raise ValueError("packed cu_seqlens require q to have batch size one")
    else:
        cu_seqlens = cu_seqlens.contiguous()

    if batch > 1:
        q, k, value, gate, beta = (_pack_dense(tensor) for tensor in (q, k, value, gate, beta))

    _validate_cudnn_constraints(q, k, value, gate, beta, state_cache)

    output = chunk_gdn_cudnn_packed_fwd_paged_op(
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


__all__ = ["chunk_forward", "paged_chunk_forward"]
