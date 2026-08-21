"""Torch-only public wrapper for the lazily dispatched fused chunk KDA core."""

from __future__ import annotations

import torch

from attn_gym._backends.cute.utils import get_device_properties
from attn_gym.linear.kda.chunk_schedule import (
    RaggedChunkMetadata,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.kda.ops import (
    chunk_bwd_op,
    chunk_bwd_with_state_grad_op,
    chunk_fwd_op,
    chunk_fwd_ragged_op,
    chunk_fwd_ragged_paged_op,
    chunk_fwd_ragged_with_state_op,
    chunk_fwd_with_state_op,
)

_CHUNK_SIZE = 64
_HEAD_DIM = 128


def _validate_fused_constraints(q: torch.Tensor, v: torch.Tensor) -> None:
    """Validate constraints shared by every fused chunk launcher."""
    if not q.is_cuda:
        raise ValueError("the CuTe KDA core requires CUDA tensors")
    if q.shape[-1] != _HEAD_DIM or v.shape[-1] != _HEAD_DIM:
        raise ValueError("the CuTe KDA core requires K=V=128")
    if not torch.compiler.is_compiling() and get_device_properties(q.device).major < 10:
        raise ValueError("the CuTe KDA core requires CUDA capability 10.0 or newer")


class _ChunkKDA(torch.autograd.Function):
    """Attach first-order autograd to the pre-registered fused operators."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        output_final_state,
        fastmath,
        autotune,
    ):
        if cu_seqlens is None:
            assert chunk_offsets is None
            if output_final_state:
                output, state, aqk, akk = chunk_fwd_with_state_op(
                    q, k, v, cumulative_gate, beta, initial_state, autotune
                )
            else:
                output, aqk, akk = chunk_fwd_op(
                    q, k, v, cumulative_gate, beta, initial_state, autotune
                )
        elif output_final_state:
            assert chunk_offsets is not None
            output, state, aqk, akk = chunk_fwd_ragged_with_state_op(
                q,
                k,
                v,
                cumulative_gate,
                beta,
                initial_state,
                cu_seqlens,
                chunk_offsets,
                autotune,
            )
        else:
            assert chunk_offsets is not None
            output, aqk, akk = chunk_fwd_ragged_op(
                q,
                k,
                v,
                cumulative_gate,
                beta,
                initial_state,
                cu_seqlens,
                chunk_offsets,
                autotune,
            )
        ctx.save_for_backward(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            aqk,
            akk,
            initial_state,
            cu_seqlens,
            chunk_offsets,
        )
        ctx.fastmath = fastmath
        ctx.autotune = autotune
        ctx.set_materialize_grads(False)
        if output_final_state:
            return output, state
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output, d_final_state=None):
        (
            q,
            k,
            v,
            cumulative_gate,
            beta,
            aqk,
            akk,
            initial_state,
            cu_seqlens,
            chunk_offsets,
        ) = ctx.saved_tensors
        args = (
            q,
            k,
            v,
            cumulative_gate,
            beta,
            aqk,
            akk,
            cu_seqlens,
            chunk_offsets,
            d_output,
            d_final_state,
            initial_state,
            ctx.fastmath,
            ctx.autotune,
        )
        if initial_state is not None:
            dq, dk, dv, dg, db, d_initial_state = chunk_bwd_with_state_grad_op(*args)
        else:
            dq, dk, dv, dg, db = chunk_bwd_op(*args)
            d_initial_state = None
        return dq, dk, dv, dg, db, d_initial_state, None, None, None, None, None


def chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    metadata: RaggedChunkMetadata | None = None,
    output_final_state: bool = False,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    fastmath: bool = False,
    autotune: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize inputs and invoke the registered fused chunk operators."""
    if metadata is not None:
        assert cu_seqlens is None
        metadata.validate_chunk_size(_CHUNK_SIZE)
        cu_seqlens = metadata.cu_seqlens
    _validate_fused_constraints(q, v)
    output_dtype = q.dtype
    output_shape = q.shape
    q, k, v = (tensor.to(torch.bfloat16) for tensor in (q, k, v))
    cumulative_gate = cumulative_gate.float().contiguous()
    beta = beta.float().contiguous()
    batch, tokens, heads, head_dim = output_shape
    if cu_seqlens is not None and metadata is None:
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, _CHUNK_SIZE)
    if metadata is None and (batch != 1 or tokens % _CHUNK_SIZE != 0):
        packed_shape = (1, batch * tokens, heads, head_dim)
        q, k, v, cumulative_gate = (
            tensor.reshape(packed_shape) for tensor in (q, k, v, cumulative_gate)
        )
        beta = beta.reshape(packed_shape[:3])
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * tokens
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, batch * tokens, _CHUNK_SIZE)
    if state_indices is not None and metadata is None:
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * tokens
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, batch * tokens, _CHUNK_SIZE)
    if state_indices is None and initial_state is not None:
        initial_state = initial_state.float().contiguous()
    cu_seqlens = None if metadata is None else metadata.cu_seqlens
    chunk_offsets = None if metadata is None else metadata.chunk_offsets
    if state_indices is not None:
        if torch.is_grad_enabled() and any(
            tensor.requires_grad
            for tensor in (q, k, v, cumulative_gate, beta, initial_state)
            if tensor is not None
        ):
            raise RuntimeError(
                "paged chunk_kda is inference-only; call under torch.no_grad() or "
                "torch.inference_mode()"
            )
        assert initial_state is not None
        assert cu_seqlens is not None and chunk_offsets is not None
        output = chunk_fwd_ragged_paged_op(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            state_indices,
            has_initial_state,
            cu_seqlens,
            chunk_offsets,
            autotune,
        )
        state = None
    elif output_final_state:
        output, state = _ChunkKDA.apply(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            cu_seqlens,
            chunk_offsets,
            True,
            fastmath,
            autotune,
        )
    else:
        output = _ChunkKDA.apply(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            cu_seqlens,
            chunk_offsets,
            False,
            fastmath,
            autotune,
        )
        state = None
    return output.reshape(output_shape).to(output_dtype), state


__all__ = ["chunk_forward"]
