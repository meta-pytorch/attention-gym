"""Composed fixed-length and packed Blackwell KDA core forward."""

from __future__ import annotations

from typing import NamedTuple

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
    chunk_fwd_ragged_paged_op as _chunk_kda_fwd_ragged_paged_op,
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
    """Provide the compact 128-byte-aligned ABI required by packed token gradients."""
    if tensor.is_contiguous() and tensor_supports_contiguous_dim(tensor, alignment_bytes=128):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _normalize_state_cotangent(tensor: torch.Tensor) -> torch.Tensor:
    """Preserve supported state strides and materialize broadcast cotangents."""
    if tensor.stride(-1) == 1 and all(stride >= 0 for stride in tensor.stride()):
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
    if q.dtype not in (torch.float16, torch.bfloat16) or (k.dtype, v.dtype) != (
        q.dtype,
        q.dtype,
    ):
        raise TypeError("the private chunk_kda ABI requires matching float16 or bfloat16 q, k, v")
    if cumulative_gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("the private chunk_kda ABI requires float32 cumulative_gate and beta")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError("the private chunk_kda ABI requires a float32 initial_state")
    if initial_state is not None and (
        initial_state.stride(-1) != 1 or any(stride < 0 for stride in initial_state.stride())
    ):
        raise ValueError("the private chunk_kda ABI requires a contiguous state key mode")
    if not all(tensor.is_contiguous() for tensor in contiguous_tensors):
        raise ValueError("the private chunk_kda ABI requires contiguous gate and beta")
    if not all(_has_supported_qkv_layout(tensor) for tensor in (q, k, v)):
        raise ValueError(
            "the private chunk_kda ABI requires QKV to have contiguous heads and "
            "16-byte-aligned token rows"
        )
    if get_device_properties(q.device).major < 10:
        raise ValueError("the CuTe KDA core requires CUDA capability 10.0 or newer")


def _validate_paged_private_abi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
) -> None:
    _validate_private_abi(q, k, v, cumulative_gate, beta, None)
    heads, key_dim, value_dim = q.shape[2], q.shape[3], v.shape[-1]
    if state_cache.dtype != torch.float32:
        raise TypeError("the paged chunk state pool must use float32")
    if state_cache.ndim != 4 or state_cache.shape[1:] != (heads, value_dim, key_dim):
        raise ValueError(
            "the paged chunk state pool must have shape "
            f"[num_slots, {heads}, {value_dim}, {key_dim}]"
        )
    if state_cache.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        raise ValueError("the paged chunk state pool must have dense [H, V, K] slots")
    if state_cache.stride(0) < heads * key_dim * value_dim:
        raise ValueError("paged chunk state slots must not overlap")
    if state_cache.device != q.device or state_indices.device != q.device:
        raise ValueError("the paged chunk state pool and indices must be on the QKV device")
    if state_indices.dtype != torch.int32 or not state_indices.is_contiguous():
        raise ValueError("paged chunk state indices must be contiguous int32")
    if state_indices.shape != (cu_seqlens.shape[0] - 1,):
        raise ValueError("paged chunk state indices must have one entry per sequence")
    if has_initial_state is not None and (
        has_initial_state.shape != state_indices.shape
        or has_initial_state.dtype != torch.bool
        or not has_initial_state.is_contiguous()
        or has_initial_state.device != q.device
    ):
        raise ValueError("has_initial_state must be contiguous bool with one entry per sequence")


class ChunkKDAFactors(NamedTuple):
    """Local KDA factors shared by state summary and output composition."""

    w: torch.Tensor
    u: torch.Tensor
    kg: torch.Tensor
    aqk: torch.Tensor
    akk: torch.Tensor


def _prepare_chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
    *,
    scale: float,
    autotune: bool,
) -> ChunkKDAFactors:
    """Compute the local factors needed by both CP summaries and ordinary output."""
    with profiler_range("kda/fused/chunk_kda_fwd_intra"):
        return ChunkKDAFactors(
            *chunk_kda_fwd_intra(
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
        )


def _finish_chunk_kda_fwd(
    q: torch.Tensor,
    cumulative_gate: torch.Tensor,
    factors: ChunkKDAFactors,
    initial_state: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    *,
    scale: float,
    output_final_state: bool,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply an initial state and compose token outputs from prepared local factors."""
    with profiler_range("kda/triton/inter_chunk_state"):
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            factors.kg,
            factors.w,
            factors.u,
            cumulative_gate,
            initial_state,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
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
            factors.aqk,
            h,
            scale,
            chunk_size=_CHUNK_SIZE,
            metadata=metadata,
            autotune=autotune,
        )
    return output, final_state


def _chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    *,
    scale: float,
    output_final_state: bool,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Run the optimized KDA core using an already selected chunk schedule."""
    factors = _prepare_chunk_kda_fwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        metadata,
        scale=scale,
        autotune=autotune,
    )
    output, final_state = _finish_chunk_kda_fwd(
        q,
        cumulative_gate,
        factors,
        initial_state,
        state_indices,
        has_initial_state,
        metadata,
        scale=scale,
        output_final_state=output_final_state,
        autotune=autotune,
    )
    return output, final_state, factors.aqk, factors.akk


def _chunk_kda_fwd_shared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
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
        None,
        None,
        scale=scale,
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
    scale: float,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, _final_state, Aqk, Akk = _chunk_kda_fwd_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        scale,
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
    scale: float,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return _chunk_kda_fwd_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        scale,
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
    scale: float,
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
        None,
        None,
        metadata,
        scale=scale,
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
    scale: float,
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
        scale,
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
    scale: float,
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
        scale,
        autotune,
        True,
    )


def _chunk_kda_fwd_ragged_paged_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    autotune: bool,
) -> torch.Tensor:
    q, k, v = (_normalize_qkv_layout(tensor) for tensor in (q, k, v))
    _validate_paged_private_abi(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        state_cache,
        state_indices,
        has_initial_state,
        cu_seqlens,
    )
    metadata = RaggedChunkMetadata(
        cu_seqlens,
        chunk_offsets,
        chunk_capacity(q.shape[1], cu_seqlens.shape[0] - 1, _CHUNK_SIZE),
        _CHUNK_SIZE,
    )
    output, _state, _aqk, _akk = _chunk_kda_fwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        state_cache,
        state_indices,
        has_initial_state,
        metadata,
        scale=_HEAD_DIM**-0.5,
        output_final_state=False,
        autotune=autotune,
    )
    return output


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
        d_final_state = _normalize_state_cotangent(d_final_state.float())
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
    scale: float,
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
            scale,
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
        scale=scale,
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
    scale: float,
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
        scale,
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
    "_chunk_kda_fwd_ragged_paged_op",
    "_chunk_kda_fwd_ragged_with_state_op",
    "_chunk_kda_fwd_with_state_op",
]
