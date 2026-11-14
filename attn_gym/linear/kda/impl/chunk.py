"""Composed fixed-length and packed Triton KDA forward and backward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch

from attn_gym._backends.cute import (
    get_device_properties,
    normalize_tma_rows,
    tensor_supports_tma_rows,
)
from attn_gym._backends.profiler import profiler_range
from attn_gym.linear._delta_rule.triton.chunk_scheduler import RaggedChunkMetadata, ScheduleRequest
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk import chunk_kda_bwd_daqk
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_delta_h_triton import (
    chunk_kda_bwd_delta_h_triton,
)
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_intra import chunk_kda_bwd_intra_triton
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_wy_triton import chunk_kda_bwd_wy_triton
from attn_gym.linear.kda.fwd.factors import chunk_kda_fwd_factors, chunk_kda_fwd_intra
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o import chunk_gla_fwd_o_gk
from attn_gym.linear.kda.fwd.triton.recompute_w_u import recompute_w_u_fwd_triton

# TODO: Revisit model-approved chunk sizes: this is a major performance lever,
# but it changes the KDA decomposition and rounding order, so it can affect numerics.
_CHUNK_SIZE = 64
_HEAD_DIM = 128


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
    if not all(tensor_supports_tma_rows(tensor) for tensor in (q, k, v)):
        raise ValueError(
            "the private chunk_kda ABI requires QKV to have contiguous heads and "
            "16-byte-aligned token rows"
        )
    if get_device_properties(q.device).major < 8:
        raise ValueError("the fused KDA core requires CUDA capability 8.0 or newer")


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
    if state_cache.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("the paged chunk state pool must use float32 or bfloat16")
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
    schedule: ScheduleRequest,
    fastmath: bool = False,
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
                autotune=autotune,
                schedule=schedule,
                fastmath=fastmath,
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
    schedule: ScheduleRequest,
    fastmath: bool = False,
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
            fastmath=fastmath,
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
            schedule=schedule,
            fastmath=fastmath,
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
    schedule: ScheduleRequest,
    fastmath: bool = False,
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
        schedule=schedule,
        fastmath=fastmath,
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
        schedule=schedule,
        fastmath=fastmath,
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
    schedule: str,
    output_final_state: bool,
    fastmath: bool = False,
):
    """Keep the complete composed forward behind one compiler-opaque boundary."""
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
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
        schedule=ScheduleRequest(schedule),
        fastmath=fastmath,
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
    schedule: str,
    fastmath: bool = False,
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
        schedule,
        False,
        fastmath,
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
    schedule: str,
    fastmath: bool = False,
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
        schedule,
        True,
        fastmath,
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
    schedule: str,
    output_final_state: bool,
    fastmath: bool = False,
):
    """Run ragged forward with caller-prepared routing and fixed-schema factors."""
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    metadata = RaggedChunkMetadata.from_offsets(cu_seqlens, chunk_offsets, q.shape[1], _CHUNK_SIZE)
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
        schedule=ScheduleRequest(schedule),
        fastmath=fastmath,
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
    schedule: str,
    fastmath: bool = False,
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
        schedule,
        False,
        fastmath,
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
    schedule: str,
    fastmath: bool = False,
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
        schedule,
        True,
        fastmath,
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
    schedule: str,
) -> torch.Tensor:
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
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
    metadata = RaggedChunkMetadata.from_offsets(cu_seqlens, chunk_offsets, q.shape[1], _CHUNK_SIZE)
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
        schedule=ScheduleRequest(schedule),
        fastmath=True,
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
        metadata = RaggedChunkMetadata.from_offsets(
            cu_seqlens, chunk_offsets, q.shape[1], _CHUNK_SIZE
        )
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    d_output = v.new_zeros(v.shape) if d_output is None else d_output.contiguous()
    if d_final_state is not None:
        d_final_state = d_final_state.float().contiguous()
    return q, k, v, metadata, d_output, d_final_state


def _chunk_kda_bwd_shared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    scale: float,
    fastmath: bool,
    autotune: bool,
    schedule: str,
):
    """Run the opaque composed backward from saved factors."""
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
        schedule=ScheduleRequest(schedule),
    )


def _chunk_kda_bwd_cuda(*args) -> tuple[torch.Tensor, ...]:
    """Drop the unused state gradient to match the no-state operator schema."""
    dq, dk, dv, dg, db, _d_initial_state = _chunk_kda_bwd_shared(*args)
    return dq, dk, dv, dg, db


def _chunk_kda_bwd_with_state_grad_cuda(*args) -> tuple[torch.Tensor, ...]:
    """Preserve the state gradient required by the stateful operator schema."""
    return _chunk_kda_bwd_shared(*args)


@dataclass
class ChunkKDABwdPrepared:
    """Resolved intra factors and consumable tensors across the CP backward boundary."""

    aqk: torch.Tensor | None
    akk: torch.Tensor | None
    w: torch.Tensor | None
    qg: torch.Tensor | None
    kg: torch.Tensor | None
    h: torch.Tensor | None
    v_new: torch.Tensor | None
    d_aqk: torch.Tensor | None


def _prepare_chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor | None,
    Akk: torch.Tensor | None,
    do: torch.Tensor,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    *,
    scale: float,
    chunk_size: int,
    autotune: bool,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
    fastmath: bool = False,
) -> ChunkKDABwdPrepared:
    """Resolve intra factors and recompute local state before CP communication."""
    if Aqk is None:
        Aqk, Akk = chunk_kda_fwd_factors(
            q, k, g, beta, scale, metadata, chunk_size=chunk_size, fastmath=fastmath
        )
    assert Akk is not None
    with profiler_range("kda/triton/backward_recompute_w_u"):
        w, u, qg, kg = recompute_w_u_fwd_triton(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=g,
            metadata=metadata,
            chunk_size=chunk_size,
            autotune=autotune,
            schedule=schedule,
            fastmath=fastmath,
        )
    assert qg is not None and kg is not None
    with profiler_range("kda/triton/backward_recompute_state"):
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            kg,
            w,
            u,
            g,
            initial_state,
            chunk_size=chunk_size,
            output_final_state=False,
            metadata=metadata,
            fastmath=fastmath,
        )
    with profiler_range("kda/triton/backward_daqk"):
        d_aqk = chunk_kda_bwd_daqk(
            v_new,
            do,
            scale,
            chunk_size=chunk_size,
            metadata=metadata,
        )
    return ChunkKDABwdPrepared(Aqk, Akk, w, qg, kg, h, v_new, d_aqk)


def _finish_chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    prepared: ChunkKDABwdPrepared,
    *,
    scale: float,
    fastmath: bool,
    autotune: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Finish local gradients while releasing prepared tensors at their last use."""
    assert prepared.aqk is not None and prepared.akk is not None
    assert prepared.qg is not None and prepared.kg is not None and prepared.w is not None
    with profiler_range("kda/triton/backward_delta_h"):
        dh, d_initial_state, dv = chunk_kda_bwd_delta_h_triton(
            prepared.qg,
            prepared.kg,
            prepared.w,
            do,
            prepared.aqk,
            gk=g,
            initial_state=initial_state,
            d_final_state=d_final_state,
            scale=scale,
            metadata=metadata,
            fastmath=fastmath,
        )
    prepared.aqk = prepared.qg = prepared.kg = prepared.w = None

    assert prepared.v_new is not None and prepared.h is not None
    with profiler_range("kda/triton/backward_wy_dqkg"):
        dq, dk, dv, dg, db, dAkk = chunk_kda_bwd_wy_triton(
            q,
            k,
            v,
            prepared.v_new,
            g,
            beta,
            prepared.akk,
            prepared.h,
            do,
            dh,
            dv,
            metadata,
            scale=scale,
            fastmath=fastmath,
        )
    prepared.akk = prepared.h = prepared.v_new = None
    del dh

    assert prepared.d_aqk is not None
    with profiler_range("kda/triton/backward_intra"):
        dq, dk, dg, db = chunk_kda_bwd_intra_triton(
            q,
            k,
            g,
            beta,
            prepared.d_aqk,
            dAkk,
            dq,
            dk,
            db,
            dg,
            metadata,
            autotune=autotune,
            fastmath=fastmath,
        )
    prepared.d_aqk = None
    return dq, dk, dv, dg, db, d_initial_state


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor | None,
    Akk: torch.Tensor | None,
    do: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    *,
    scale: float,
    chunk_size: int = 64,
    fastmath: bool = False,
    autotune: bool,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Differentiate the optimized fixed-length or packed KDA core pipeline."""
    batch, tokens, _heads, head_dim = q.shape
    value_dim = v.shape[-1]
    if batch != 1 or head_dim != 128 or value_dim != 128:
        raise ValueError("the composed KDA backward requires B=1 and K=V=128")
    if chunk_size != 64:
        raise ValueError(f"the composed KDA backward requires chunk_size=64, got {chunk_size}")
    if tokens % chunk_size and metadata is None:
        raise ValueError("the composed KDA backward requires complete chunks")

    # Forward deliberately saves only minimal factors. Reconstruct large local
    # tensors before the optional CP reverse-summary communication boundary.
    prepared = _prepare_chunk_kda_bwd(
        q,
        k,
        v,
        g,
        beta,
        Aqk,
        Akk,
        do,
        initial_state,
        metadata,
        scale=scale,
        chunk_size=chunk_size,
        autotune=autotune,
        schedule=schedule,
        fastmath=fastmath,
    )
    return _finish_chunk_kda_bwd(
        q,
        k,
        v,
        g,
        beta,
        do,
        d_final_state,
        initial_state,
        metadata,
        prepared,
        scale=scale,
        fastmath=fastmath,
        autotune=autotune,
    )


__all__ = ["chunk_kda_bwd"]
