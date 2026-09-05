"""Dense training-capable fused chunk GDN implementation."""

from __future__ import annotations

from typing import NamedTuple

import torch

from attn_gym._backends.cute import (
    get_device_properties,
    normalize_compact_tensor,
    normalize_tma_rows,
)
from attn_gym._backends.triton.utils import requires_int64_offsets
from attn_gym.linear._delta_rule.span import zero_state
from attn_gym.linear._delta_rule.validation import validate_paged_state
from attn_gym.linear.gdn.bwd.triton.chunk_gdn_bwd_delta_h import chunk_gdn_bwd_delta_h
from attn_gym.linear.gdn.bwd.triton.chunk_gdn_bwd_intra import (
    chunk_gdn_bwd_intra_dense,
    chunk_gdn_bwd_intra_packed,
)
from attn_gym.linear.gdn.bwd.triton.chunk_gdn_bwd_recompute import (
    chunk_gdn_recompute_aqk_dense,
    chunk_gdn_recompute_aqk_packed,
)
from attn_gym.linear.gdn.bwd.triton.chunk_gdn_bwd_wy import chunk_gdn_bwd_wy
from attn_gym.linear.gdn.fwd.triton.chunk_gdn_fwd_intra import (
    chunk_gdn_fwd_intra_dense,
    chunk_gdn_fwd_intra_packed,
    chunk_gdn_recompute_w_u_qg_kg,
)
from attn_gym.linear.gdn.fwd.triton.chunk_gdn_fwd_output import (
    chunk_gdn_fwd_output_dense,
    chunk_gdn_fwd_output_packed,
)
from attn_gym.linear.gdn.fwd.triton.chunk_gdn_fwd_recurrence import (
    chunk_gdn_fwd_recurrence_dense,
    chunk_gdn_fwd_recurrence_packed,
)
from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
    blackwell_delta_h_bwd_dhu_dv_fused_dispatch,
)
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
    chunk_kda_bwd_wy_dqkg,
)
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk import chunk_kda_bwd_daqk
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h


def validate_supported_device(q: torch.Tensor) -> None:
    """Reject devices older than the Ampere-capable fused implementation."""
    if get_device_properties(q.device).major < 8:
        raise ValueError("fused chunk_gdn requires CUDA capability 8.0 or newer")


def use_blackwell_backward(q: torch.Tensor) -> bool:
    """Select the CuTe backward only on its validated SM100/SM103 targets."""
    properties = get_device_properties(q.device)
    return properties.major == 10 and properties.minor in (0, 3)


def reject_int64_offsets(*tensors: torch.Tensor | None) -> None:
    """Reject layouts that need the not-yet-implemented wide-address specialization."""
    if requires_int64_offsets(*tensors):
        raise ValueError("fused chunk_gdn does not yet support int64 tensor offsets")


class ChunkGDNFactors(NamedTuple):
    """Local WY factors shared by context-parallel state summaries and output composition."""

    w: torch.Tensor
    u: torch.Tensor
    kg: torch.Tensor
    inverse: torch.Tensor


def _prepare_chunk_gdn_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
) -> ChunkGDNFactors:
    """Run the intra-chunk factorization on already normalized inputs."""
    if metadata is None:
        return ChunkGDNFactors(*chunk_gdn_fwd_intra_dense(k, v, cumulative_gate, beta))
    return ChunkGDNFactors(*chunk_gdn_fwd_intra_packed(k, v, cumulative_gate, beta, metadata))


def _finish_chunk_gdn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    factors: ChunkGDNFactors,
    cumulative_gate: torch.Tensor,
    initial_state: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the state recurrence and output composition from prepared factors."""
    if metadata is None:
        h, v_new, final_state = chunk_gdn_fwd_recurrence_dense(
            factors.kg, factors.w, factors.u, cumulative_gate, initial_state
        )
        output = chunk_gdn_fwd_output_dense(q, k, v_new, h, cumulative_gate, scale)
    else:
        h, v_new, final_state = chunk_gdn_fwd_recurrence_packed(
            factors.kg, factors.w, factors.u, cumulative_gate, initial_state, metadata
        )
        output = chunk_gdn_fwd_output_packed(q, k, v_new, h, cumulative_gate, scale, metadata)
    return output, final_state


def chunk_gdn_fwd_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the dense B=1 BT64 scalar forward and return output, state, and inverse tape."""
    validate_supported_device(q)
    batch, tokens, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if batch != 1 or tokens % 64 or (key_dim, value_dim) != (128, 128):
        raise ValueError("dense fused chunk GDN requires B=1, complete BT64 chunks, and K=V=128")
    if k.shape != q.shape or v.shape[:2] != q.shape[:2] or value_heads % key_heads:
        raise ValueError("dense fused chunk GDN requires matching Q/K and H % HK == 0")
    if cumulative_gate.shape != v.shape[:3] or beta.shape != v.shape[:3]:
        raise ValueError("cumulative_gate and beta must have shape [B,T,H]")
    if initial_state is None:
        initial_state = zero_state(q, v, None)
    reject_int64_offsets(q, k, v, cumulative_gate, beta, initial_state)
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    cumulative_gate, beta, initial_state = (
        normalize_compact_tensor(tensor) for tensor in (cumulative_gate, beta, initial_state)
    )

    factors = _prepare_chunk_gdn_fwd(k, v, cumulative_gate, beta, None)
    output, final_state = _finish_chunk_gdn_fwd(
        q, k, factors, cumulative_gate, initial_state, None, scale
    )
    return output, final_state, factors.inverse


def _gdn_chunk_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Registered dense forward without a public final-state output."""
    output, _state, inverse = chunk_gdn_fwd_dense(
        q, k, v, cumulative_gate, beta, initial_state, scale
    )
    return output, inverse


def _gdn_chunk_fwd_with_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Registered dense forward with final state and inverse tape."""
    return chunk_gdn_fwd_dense(q, k, v, cumulative_gate, beta, initial_state, scale)


def chunk_gdn_fwd_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fixed-capacity packed scalar forward and return output, state, and inverse tape."""
    validate_supported_device(q)
    batch, _tokens, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if batch != 1 or (key_dim, value_dim) != (128, 128):
        raise ValueError("packed fused chunk GDN requires B=1 and K=V=128")
    if k.shape != q.shape or v.shape[:2] != q.shape[:2] or value_heads % key_heads:
        raise ValueError("packed fused chunk GDN requires matching Q/K and H % HK == 0")
    if initial_state is None:
        initial_state = zero_state(q, v, metadata)
    reject_int64_offsets(q, k, v, cumulative_gate, beta, initial_state)
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    cumulative_gate, beta, initial_state = (
        normalize_compact_tensor(tensor) for tensor in (cumulative_gate, beta, initial_state)
    )

    factors = _prepare_chunk_gdn_fwd(k, v, cumulative_gate, beta, metadata)
    output, final_state = _finish_chunk_gdn_fwd(
        q, k, factors, cumulative_gate, initial_state, metadata, scale
    )
    return output, final_state, factors.inverse


def _gdn_chunk_fwd_packed_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    capacity: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Registered packed forward without a public final-state output."""
    metadata = RaggedChunkMetadata(cu_seqlens, chunk_offsets, capacity, 64)
    output, _state, inverse = chunk_gdn_fwd_packed(
        q, k, v, cumulative_gate, beta, initial_state, metadata, scale
    )
    return output, inverse


def _gdn_chunk_fwd_packed_with_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    capacity: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Registered packed forward with final state and inverse tape."""
    metadata = RaggedChunkMetadata(cu_seqlens, chunk_offsets, capacity, 64)
    return chunk_gdn_fwd_packed(q, k, v, cumulative_gate, beta, initial_state, metadata, scale)


def _gdn_chunk_fwd_packed_paged_cuda(
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
    capacity: int,
    scale: float,
) -> torch.Tensor:
    """Run packed scalar GDN while advancing selected cache slots in place."""
    validate_supported_device(q)
    metadata = RaggedChunkMetadata(cu_seqlens, chunk_offsets, capacity, 64)
    batch, _tokens, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if batch != 1 or (key_dim, value_dim) != (128, 128):
        raise ValueError("paged fused chunk GDN requires B=1 and K=V=128")
    if k.shape != q.shape or v.shape[:2] != q.shape[:2] or value_heads % key_heads:
        raise ValueError("paged fused chunk GDN requires matching Q/K and H % HK == 0")
    if cumulative_gate.shape != v.shape[:3] or beta.shape != v.shape[:3]:
        raise ValueError("cumulative_gate and beta must have shape [B,T,H]")
    validate_paged_state(
        q,
        v,
        state_cache,
        cu_seqlens,
        state_indices,
        has_initial_state,
    )
    reject_int64_offsets(q, k, v, cumulative_gate, beta)
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    cumulative_gate, beta = (
        normalize_compact_tensor(tensor) for tensor in (cumulative_gate, beta)
    )

    w, u, restored_k, _inverse = chunk_gdn_fwd_intra_packed(k, v, cumulative_gate, beta, metadata)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        restored_k,
        w,
        u,
        cumulative_gate,
        state_cache,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        output_final_state=False,
        metadata=metadata,
        autotune=False,
    )
    assert final_state is None
    return chunk_gdn_fwd_output_packed(q, k, v_new, h, cumulative_gate, scale, metadata)


class ChunkGDNBwdPrepared(NamedTuple):
    """Recomputed local tensors shared across the context-parallel backward boundary."""

    w: torch.Tensor
    qg: torch.Tensor
    kg: torch.Tensor
    h: torch.Tensor
    v_new: torch.Tensor


def _prepare_chunk_gdn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
) -> ChunkGDNBwdPrepared:
    """Recompute local factors and forward state on normalized inputs before communication."""
    recurrence_state = zero_state(q, v, metadata) if initial_state is None else initial_state
    w, u, qg, kg = chunk_gdn_recompute_w_u_qg_kg(q, k, v, cumulative_gate, beta, inverse, metadata)
    if metadata is None:
        h, v_new, _final_state = chunk_gdn_fwd_recurrence_dense(
            kg, w, u, cumulative_gate, recurrence_state
        )
    else:
        h, v_new, _final_state = chunk_gdn_fwd_recurrence_packed(
            kg, w, u, cumulative_gate, recurrence_state, metadata
        )
    return ChunkGDNBwdPrepared(w, qg, kg, h, v_new)


def chunk_gdn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    """Differentiate dense or packed fused chunk GDN through one shared protocol."""
    validate_supported_device(q)
    reject_int64_offsets(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        inverse,
        d_output,
        d_final_state,
        initial_state,
    )
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    cumulative_gate, beta, inverse, d_output = (
        normalize_compact_tensor(tensor) for tensor in (cumulative_gate, beta, inverse, d_output)
    )
    if d_final_state is not None:
        d_final_state = normalize_compact_tensor(d_final_state)
    if initial_state is not None:
        initial_state = normalize_compact_tensor(initial_state)
    prepared = _prepare_chunk_gdn_bwd(
        q, k, v, cumulative_gate, beta, inverse, initial_state, metadata
    )
    return _finish_chunk_gdn_bwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        inverse,
        d_output,
        d_final_state,
        initial_state,
        metadata,
        prepared,
        scale,
    )


def _finish_chunk_gdn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    prepared: ChunkGDNBwdPrepared,
    scale: float,
    aqk: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Finish the local gradients from recomputed factors and the exit-state cotangent.

    ``aqk`` is the expanded-head intra-chunk Q/K factor when the caller already holds it (the
    staged reverse summary computes it first); the Blackwell path recomputes it otherwise.
    """
    w, qg, kg, h, v_new = prepared
    if not use_blackwell_backward(q):
        dh, d_initial_state, dv = chunk_gdn_bwd_delta_h(
            q,
            k,
            w,
            cumulative_gate,
            d_output,
            d_final_state,
            initial_state,
            metadata,
            scale,
        )
        dq, dk, dv, d_gate, db = chunk_gdn_bwd_wy(
            q,
            k,
            v,
            v_new,
            w,
            cumulative_gate,
            beta,
            inverse,
            h,
            d_output,
            dh,
            dv,
            metadata,
            scale,
        )
        return dq, dk, dv, d_gate, db, d_initial_state

    d_aqk = chunk_kda_bwd_daqk(
        v_new,
        d_output,
        scale,
        chunk_size=64,
        metadata=metadata,
    )

    # Reuse the proven vector-gate KDA gradient stages only after the scalar
    # recompute; forward and backward preparation remain expansion-free.
    groups = v.shape[2] // q.shape[2]
    expanded_q, expanded_k = q, k
    if groups > 1:
        expanded_q, expanded_k = (tensor.repeat_interleave(groups, dim=2) for tensor in (q, k))
    vector_gate = cumulative_gate.unsqueeze(-1).expand_as(expanded_q).contiguous()
    if aqk is None:
        aqk = (
            chunk_gdn_recompute_aqk_dense(expanded_q, expanded_k, cumulative_gate, scale)
            if metadata is None
            else chunk_gdn_recompute_aqk_packed(
                expanded_q, expanded_k, cumulative_gate, scale, metadata
            )
        )
    dh, d_initial_state, dv = blackwell_delta_h_bwd_dhu_dv_fused_dispatch(
        qg,
        kg,
        w,
        d_output,
        aqk,
        gk=vector_gate,
        h0=initial_state,
        dht=d_final_state,
        scale=scale,
        chunk_size=64,
        metadata=metadata,
    )
    dq, dk, dv, dg_raw, db, d_raw_akk = chunk_kda_bwd_wy_dqkg(
        expanded_q,
        expanded_k,
        v,
        v_new,
        vector_gate,
        beta,
        inverse,
        h,
        d_output,
        dh,
        dv,
        metadata,
        scale=scale,
        chunk_size=64,
        fastmath=False,
        autotune=False,
    )
    intra = (
        chunk_gdn_bwd_intra_dense(
            expanded_q,
            expanded_k,
            cumulative_gate,
            beta,
            d_aqk,
            d_raw_akk,
            dg_raw,
        )
        if metadata is None
        else chunk_gdn_bwd_intra_packed(
            expanded_q,
            expanded_k,
            cumulative_gate,
            beta,
            d_aqk,
            d_raw_akk,
            dg_raw,
            metadata,
        )
    )
    intra_dq, intra_dk, intra_db, d_gate = intra
    dq = dq.float() + intra_dq
    dk = dk.float() + intra_dk
    if groups > 1:
        dq = dq.view(*q.shape[:2], q.shape[2], groups, q.shape[3]).sum(3)
        dk = dk.view(*k.shape[:2], k.shape[2], groups, k.shape[3]).sum(3)
    dq = dq.to(q.dtype)
    dk = dk.to(k.dtype)
    db = db + intra_db
    return dq, dk, dv, d_gate, db, d_initial_state


def resolve_backward_metadata(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
) -> RaggedChunkMetadata | None:
    """Reconstruct shape-derived packed metadata for the merged backward schemas."""
    if cu_seqlens is None:
        assert chunk_offsets is None
        return None
    assert chunk_offsets is not None
    return RaggedChunkMetadata.from_offsets(cu_seqlens, chunk_offsets, q.shape[1], 64)


def _gdn_chunk_bwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    """Registered dense or packed backward without an initial-state gradient output."""
    dq, dk, dv, dg, db, _d_initial_state = chunk_gdn_bwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        inverse,
        d_output,
        d_final_state,
        initial_state,
        resolve_backward_metadata(q, cu_seqlens, chunk_offsets),
        scale,
    )
    return dq, dk, dv, dg, db


def _gdn_chunk_bwd_with_state_grad_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    """Registered dense or packed backward preserving the initial-state gradient."""
    return chunk_gdn_bwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        inverse,
        d_output,
        d_final_state,
        initial_state,
        resolve_backward_metadata(q, cu_seqlens, chunk_offsets),
        scale,
    )


__all__ = ["chunk_gdn_bwd", "chunk_gdn_fwd_dense", "chunk_gdn_fwd_packed"]
