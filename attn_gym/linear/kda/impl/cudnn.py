# SPDX-License-Identifier: BSD-3-Clause

"""Autograd integration for the experimental CuTeDSL 4.7 KDA kernels."""

from __future__ import annotations

import torch

from attn_gym._backends.cute import tensor_supports_contiguous_dim, tensor_supports_tma
from attn_gym.linear._delta_rule.chunk_ops import _plain_gate_scan_op
from attn_gym.linear._delta_rule.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.linear._delta_rule.validation import resolve_scale
from attn_gym.linear.kda.impl.cudnn_ops import (
    chunk_cudnn_packed_fwd_op,
    chunk_cudnn_packed_fwd_paged_op,
    chunk_cudnn_packed_fwd_with_initial_state_op,
    chunk_cudnn_packed_fwd_with_state_op,
    chunk_cudnn_packed_local_bwd_op,
    validate_cudnn_available,
)
from attn_gym.linear.kda.ops import chunk_bwd_recompute_factors_with_state_grad_op

_CHUNK_SIZE = 64
_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)


def normalize_output_grad(d_output: torch.Tensor | None, value: torch.Tensor) -> torch.Tensor:
    """Materialize the contiguous output cotangent required by the cuDNN operator ABI."""
    return torch.zeros_like(value) if d_output is None else d_output.contiguous()


class ChunkKdaCudnn(torch.autograd.Function):
    """Attach a backward to cuDNN's forward: cuDNN's own without an entry state, fused with one.

    The kernel choice depends only on whether an entry state is present, never on the span
    length, so a document's bits do not depend on the span it is packed into. With an entry
    state the fused stateful backward carries the state cotangent in FP32 between chunks
    (``test/linear/test_delta_rule_state_precision.py``); cuDNN's BT16 backward requantizes it.
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
        chunk_offsets: torch.Tensor,
        scale: float,
        split_backward: bool,
        split_forward: bool,
        output_final_state: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ctx.split_backward = split_backward
        ctx.scale = scale
        if initial_state is None:
            output = chunk_cudnn_packed_fwd_op(
                q, k, value, gate, beta, cu_seqlens, split_forward, scale
            )
        elif output_final_state:
            output, final_state = chunk_cudnn_packed_fwd_with_state_op(
                q, k, value, gate, beta, initial_state, cu_seqlens, scale
            )
        else:
            output = chunk_cudnn_packed_fwd_with_initial_state_op(
                q, k, value, gate, beta, initial_state, cu_seqlens, scale
            )
        ctx.save_for_backward(q, k, value, gate, beta, initial_state, cu_seqlens, chunk_offsets)
        ctx.set_materialize_grads(False)
        if output_final_state:
            return output, final_state
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output: torch.Tensor | None, d_final_state: torch.Tensor | None = None):
        q, k, value, gate, beta, initial_state, cu_seqlens, chunk_offsets = ctx.saved_tensors
        d_output = normalize_output_grad(d_output, value)
        if initial_state is None:
            dq, dk, dv, d_gate, d_beta = chunk_cudnn_packed_local_bwd_op(
                q, k, value, gate, beta, d_output, cu_seqlens, ctx.split_backward, ctx.scale
            )
            d_initial_state = None
        else:
            cumulative_gate = _plain_gate_scan_op(gate, cu_seqlens, chunk_offsets, False)
            dq, dk, dv, d_cumulative, d_beta, d_initial_state = (
                chunk_bwd_recompute_factors_with_state_grad_op(
                    q,
                    k,
                    value,
                    cumulative_gate,
                    beta,
                    cu_seqlens,
                    chunk_offsets,
                    d_output,
                    d_final_state,
                    initial_state,
                    ctx.scale,
                    False,
                    False,
                    "auto",
                )
            )
            d_gate = _plain_gate_scan_op(
                d_cumulative.contiguous(), cu_seqlens, chunk_offsets, True
            )
        return dq, dk, dv, d_gate, d_beta, d_initial_state, None, None, None, None, None, None


def validate_cudnn_constraints(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    """Validate only the additional layout, dtype, and shape constraints of cuDNN."""
    if not q.is_cuda:
        raise ValueError("the cuDNN KDA backend requires CUDA tensors")
    if q.shape[0] != 1 or q.shape[-1] != 128 or value.shape[-1] != 128:
        raise ValueError("the cuDNN backend requires B=1 and K=V=128")
    if q.dtype not in _SUPPORTED_IO_DTYPES or k.dtype != q.dtype or value.dtype != q.dtype:
        raise TypeError("q, k, and value must share dtype float16 or bfloat16")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("gate and beta must be float32")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError("initial_state must be float32")
    if not torch.compiler.is_compiling():
        tma_tensors = (q, k, value, gate) + (() if initial_state is None else (initial_state,))
        if any(not tensor_supports_tma(tensor) for tensor in tma_tensors):
            raise TypeError(
                "cuDNN q, k, value, gate, and state require a TMA-compatible inner mode"
            )
        if not tensor_supports_contiguous_dim(beta, alignment_bytes=4):
            raise TypeError("cuDNN beta requires a contiguous, element-aligned inner mode")
        if cu_seqlens is not None and cu_seqlens.data_ptr() % 8:
            raise TypeError("cuDNN cu_seqlens must be 8-byte aligned")


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
    fastmath: bool = False,
    autotune: bool = True,
    split_backward: bool = False,
    split_forward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the optional cuDNN implementation behind the shared ``chunk_kda`` contract."""
    scale = resolve_scale(scale, q.shape[-1])
    if fastmath:
        raise ValueError("fastmath is not supported by the cuDNN backend")
    if split_backward and (initial_state is not None or output_final_state):
        raise ValueError("split_backward currently requires a no-state call")
    if split_forward and (initial_state is not None or output_final_state):
        raise ValueError("split_forward currently requires a no-state call")
    del autotune
    if not torch.compiler.is_compiling():
        validate_cudnn_available(q)

    if cu_seqlens is None:
        cu_seqlens = torch.arange(2, dtype=torch.int32, device=q.device) * q.shape[1]
    if output_final_state and initial_state is None:
        initial_state = torch.zeros(
            cu_seqlens.shape[0] - 1,
            q.shape[2],
            value.shape[-1],
            q.shape[3],
            dtype=torch.float32,
            device=q.device,
        )
    validate_cudnn_constraints(q, k, value, gate, beta, initial_state, cu_seqlens)
    # The fused stateful backward chunks each subsequence on its own 64-token grid.
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, q.shape[1], _CHUNK_SIZE)
    result = ChunkKdaCudnn.apply(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        scale,
        split_backward,
        split_forward,
        output_final_state,
    )
    if output_final_state:
        return result
    return result, None


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
    """Advance selected state-cache slots with the cuDNN KDA kernel."""
    tensors = (q, k, value, gate, beta, state_cache)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(
            "paged_chunk_kda is inference-only; call under torch.no_grad() or "
            "torch.inference_mode()"
        )
    if not torch.compiler.is_compiling():
        validate_cudnn_available(q)

    batch = q.shape[0]
    output_shape = value.shape
    if cu_seqlens is None:
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * q.shape[1]
    elif batch != 1:
        raise ValueError("packed cu_seqlens require q to have batch size one")
    else:
        cu_seqlens = cu_seqlens.contiguous()
    if batch > 1:
        q, k, value, gate, beta = (
            tensor.reshape(1, -1, *tensor.shape[2:]) for tensor in (q, k, value, gate, beta)
        )

    validate_cudnn_constraints(q, k, value, gate, beta, state_cache, cu_seqlens)
    output = chunk_cudnn_packed_fwd_paged_op(
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
