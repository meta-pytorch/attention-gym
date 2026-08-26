"""Composed fixed-length and packed Blackwell KDA core forward."""

from __future__ import annotations

import torch

from attn_gym._backends.cute import (
    get_device_properties,
    tensor_supports_contiguous_dim,
    tensor_supports_tma,
)
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, chunk_capacity
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_intra
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o import chunk_gla_fwd_o_gk
from attn_gym.linear.kda.ops import (
    chunk_bwd_op as _chunk_kda_bwd_op,
)
from attn_gym.linear.kda.ops import (
    chunk_bwd_with_state_grad_op as _chunk_kda_bwd_with_state_grad_op,
)
from attn_gym.linear.kda.ops import (
    chunk_fwd_op as _chunk_kda_fwd_op,
)
from attn_gym.linear.kda.ops import (
    chunk_fwd_ragged_op as _chunk_kda_fwd_ragged_op,
)
from attn_gym.linear.kda.ops import (
    chunk_fwd_ragged_with_state_op as _chunk_kda_fwd_ragged_with_state_op,
)
from attn_gym.linear.kda.ops import (
    chunk_fwd_with_state_op as _chunk_kda_fwd_with_state_op,
)
from attn_gym.linear.kda.utils import profiler_range

# TODO: Revisit model-approved chunk sizes: this is a major performance lever,
# but it changes the KDA decomposition and rounding order, so it can affect numerics.
_CHUNK_SIZE = 64
_HEAD_DIM = 128


def _has_supported_qkv_layout(tensor: torch.Tensor) -> bool:
    """Return whether QKV rows satisfy the vectorized kernel input contract."""
    return tensor.stride(-2) == tensor.shape[-1] and tensor_supports_tma(tensor)


def _normalize_qkv_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Copy only layouts unsupported by one or more composed KDA stages."""
    if _has_supported_qkv_layout(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _normalize_packed_cotangent(tensor: torch.Tensor) -> torch.Tensor:
    """Provide the compact 128-byte-aligned ABI required by packed delta-H."""
    if tensor.is_contiguous() and tensor_supports_contiguous_dim(tensor, alignment_bytes=128):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _validate_private_abi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    contiguous_tensors = (cumulative_gate, beta)
    if initial_state is not None:
        contiguous_tensors += (initial_state,)
    if (q.dtype, k.dtype, v.dtype) != (torch.bfloat16,) * 3:
        raise TypeError("the private chunk_kda ABI requires bfloat16 q, k, and v")
    if cumulative_gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("the private chunk_kda ABI requires float32 cumulative_gate and beta")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError("the private chunk_kda ABI requires a float32 initial_state")
    if not all(tensor.is_contiguous() for tensor in contiguous_tensors):
        raise ValueError("the private chunk_kda ABI requires contiguous gate, beta, and state")
    if not all(_has_supported_qkv_layout(tensor) for tensor in (q, k, v)):
        raise ValueError(
            "the private chunk_kda ABI requires QKV to have contiguous heads and "
            "16-byte-aligned token rows"
        )
    if get_device_properties(q.device).major < 10:
        raise ValueError("the CuTe KDA core requires CUDA capability 10.0 or newer")


def _chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    *,
    output_final_state: bool,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Run the optimized KDA core using an already selected chunk schedule."""
    scale = _HEAD_DIM**-0.5

    with profiler_range("kda/fused/chunk_kda_fwd_intra"):
        w, u, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            scale,
            metadata,
            chunk_size=_CHUNK_SIZE,
            profile_ranges=torch.autograd.profiler._is_profiler_enabled,
            autotune=autotune,
        )
    with profiler_range("kda/triton/inter_chunk_state"):
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            kg,
            w,
            u,
            cumulative_gate,
            initial_state,
            chunk_size=_CHUNK_SIZE,
            output_final_state=output_final_state,
            metadata=metadata,
            autotune=autotune,
        )
    with profiler_range("kda/triton/output_composition"):
        output = chunk_gla_fwd_o_gk(
            q,
            v_new,
            cumulative_gate,
            Aqk,
            h,
            scale,
            chunk_size=_CHUNK_SIZE,
            metadata=metadata,
            autotune=autotune,
        )
    return output, final_state, Aqk, Akk


def _chunk_kda_fwd_shared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    autotune: bool,
    output_final_state: bool,
):
    """Keep the complete composed forward behind one compiler-opaque boundary."""
    q, k, v = (_normalize_qkv_layout(tensor) for tensor in (q, k, v))
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    return _chunk_kda_fwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        None,
        output_final_state=output_final_state,
        autotune=autotune,
    )


def _chunk_kda_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, _final_state, Aqk, Akk = _chunk_kda_fwd_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        autotune,
        False,
    )
    return output, Aqk, Akk


def _chunk_kda_fwd_with_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _chunk_kda_fwd_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        autotune,
        True,
    )


def _chunk_kda_fwd_ragged_shared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    autotune: bool,
    output_final_state: bool,
):
    """Run ragged forward with caller-prepared routing and fixed-schema factors."""
    q, k, v = (_normalize_qkv_layout(tensor) for tensor in (q, k, v))
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    metadata = RaggedChunkMetadata(
        cu_seqlens,
        chunk_offsets,
        chunk_capacity(q.shape[1], cu_seqlens.shape[0] - 1, _CHUNK_SIZE),
        _CHUNK_SIZE,
    )
    return _chunk_kda_fwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        metadata,
        output_final_state=output_final_state,
        autotune=autotune,
    )


def _chunk_kda_fwd_ragged_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    autotune: bool,
):
    output, _state, Aqk, Akk = _chunk_kda_fwd_ragged_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        autotune,
        False,
    )
    return output, Aqk, Akk


def _chunk_kda_fwd_ragged_with_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    autotune: bool,
):
    return _chunk_kda_fwd_ragged_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        autotune,
        True,
    )


def _prepare_chunk_kda_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
):
    """Normalize one backward invocation and reconstruct its optional metadata."""
    if (cu_seqlens is None) != (chunk_offsets is None):
        raise ValueError(
            "cu_seqlens and chunk_offsets must either both be present or both be absent"
        )
    metadata = None
    if cu_seqlens is not None:
        assert chunk_offsets is not None
        metadata = RaggedChunkMetadata(
            cu_seqlens,
            chunk_offsets,
            chunk_capacity(q.shape[1], cu_seqlens.shape[0] - 1, _CHUNK_SIZE),
            _CHUNK_SIZE,
        )
    q, k, v = (_normalize_qkv_layout(tensor) for tensor in (q, k, v))
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    if d_output is None:
        d_output = v.new_zeros(v.shape)
    elif metadata is not None:
        d_output = _normalize_packed_cotangent(d_output)
    else:
        d_output = _normalize_qkv_layout(d_output)
    if d_final_state is not None:
        d_final_state = d_final_state.float()
        d_final_state = (
            _normalize_packed_cotangent(d_final_state)
            if metadata is not None
            else _normalize_qkv_layout(d_final_state)
        )
    return q, k, v, metadata, d_output, d_final_state


def _chunk_kda_bwd_shared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor | None,
    Akk: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    fastmath: bool,
    autotune: bool,
):
    """Run the opaque composed backward with saved or recomputed factors."""
    from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd import chunk_kda_bwd

    q, k, v, metadata, d_output, d_final_state = _prepare_chunk_kda_backward(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        cu_seqlens,
        chunk_offsets,
        d_output,
        d_final_state,
        initial_state,
    )
    if Aqk is None:
        assert Akk is None
        from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_factors

        Aqk, Akk = chunk_kda_fwd_factors(
            q,
            k,
            cumulative_gate,
            beta,
            _HEAD_DIM**-0.5,
            metadata,
        )
    else:
        assert Akk is not None
    return chunk_kda_bwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        Aqk,
        Akk,
        d_output,
        d_final_state,
        initial_state,
        metadata,
        fastmath=fastmath,
        autotune=autotune,
    )


def _chunk_kda_bwd_recompute_factors_shared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    fastmath: bool,
    autotune: bool,
):
    """Keep the recomputed-factor backward opaque to AOTAutograd."""
    return _chunk_kda_bwd_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        None,
        None,
        cu_seqlens,
        chunk_offsets,
        d_output,
        d_final_state,
        initial_state,
        fastmath,
        autotune,
    )


def _chunk_kda_bwd_cuda(*args) -> tuple[torch.Tensor, ...]:
    """Drop the unused state gradient to match the no-state operator schema."""
    dq, dk, dv, dg, db, _d_initial_state = _chunk_kda_bwd_shared(*args)
    return dq, dk, dv, dg, db


def _chunk_kda_bwd_with_state_grad_cuda(*args) -> tuple[torch.Tensor, ...]:
    """Preserve the state gradient required by the stateful operator schema."""
    return _chunk_kda_bwd_shared(*args)


def _chunk_kda_bwd_recompute_factors_cuda(*args) -> tuple[torch.Tensor, ...]:
    """Drop the recomputed state gradient for the no-state operator schema."""
    dq, dk, dv, dg, db, _d_initial_state = _chunk_kda_bwd_recompute_factors_shared(*args)
    return dq, dk, dv, dg, db


def _chunk_kda_bwd_recompute_factors_with_state_grad_cuda(
    *args,
) -> tuple[torch.Tensor, ...]:
    """Preserve the recomputed state gradient for the stateful operator schema."""
    return _chunk_kda_bwd_recompute_factors_shared(*args)


__all__ = [
    "_chunk_kda_bwd_op",
    "_chunk_kda_bwd_recompute_factors_cuda",
    "_chunk_kda_bwd_recompute_factors_with_state_grad_cuda",
    "_chunk_kda_bwd_with_state_grad_op",
    "_chunk_kda_fwd_op",
    "_chunk_kda_fwd_ragged_op",
    "_chunk_kda_fwd_ragged_with_state_op",
    "_chunk_kda_fwd_with_state_op",
]
