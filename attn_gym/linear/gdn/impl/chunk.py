"""Dense training-capable fused chunk GDN implementation."""

from __future__ import annotations

import torch

from attn_gym._backends.cute import (
    get_device_properties,
    normalize_compact_tensor,
    normalize_tma_rows,
)
from attn_gym._backends.triton.utils import requires_int64_offsets
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
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, chunk_capacity


def validate_supported_device(q: torch.Tensor) -> None:
    """Reject devices older than the Hopper-capable fused implementation."""
    if get_device_properties(q.device).major < 9:
        raise ValueError("fused chunk_gdn requires CUDA capability 9.0 or newer")


def use_blackwell_backward(q: torch.Tensor) -> bool:
    """Select the CuTe backward only on its validated SM100/SM103 targets."""
    properties = get_device_properties(q.device)
    return properties.major == 10 and properties.minor in (0, 3)


def reject_int64_offsets(*tensors: torch.Tensor | None) -> None:
    """Reject layouts that need the not-yet-implemented wide-address specialization."""
    if requires_int64_offsets(*tensors):
        raise ValueError("fused chunk_gdn does not yet support int64 tensor offsets")


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
        initial_state = torch.zeros(
            batch,
            value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
    reject_int64_offsets(q, k, v, cumulative_gate, beta, initial_state)
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    cumulative_gate, beta, initial_state = (
        normalize_compact_tensor(tensor) for tensor in (cumulative_gate, beta, initial_state)
    )

    w, u, restored_k, inverse = chunk_gdn_fwd_intra_dense(k, v, cumulative_gate, beta)
    h, v_new, final_state = chunk_gdn_fwd_recurrence_dense(
        restored_k,
        w,
        u,
        cumulative_gate,
        initial_state,
    )
    output = chunk_gdn_fwd_output_dense(q, k, v_new, h, cumulative_gate, scale)
    return output, final_state, inverse


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
    num_sequences = metadata.cu_seqlens.shape[0] - 1
    if batch != 1 or (key_dim, value_dim) != (128, 128):
        raise ValueError("packed fused chunk GDN requires B=1 and K=V=128")
    if k.shape != q.shape or v.shape[:2] != q.shape[:2] or value_heads % key_heads:
        raise ValueError("packed fused chunk GDN requires matching Q/K and H % HK == 0")
    if initial_state is None:
        initial_state = torch.zeros(
            num_sequences,
            value_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
    reject_int64_offsets(q, k, v, cumulative_gate, beta, initial_state)
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    cumulative_gate, beta, initial_state = (
        normalize_compact_tensor(tensor) for tensor in (cumulative_gate, beta, initial_state)
    )

    w, u, restored_k, inverse = chunk_gdn_fwd_intra_packed(k, v, cumulative_gate, beta, metadata)
    h, v_new, final_state = chunk_gdn_fwd_recurrence_packed(
        restored_k,
        w,
        u,
        cumulative_gate,
        initial_state,
        metadata,
    )
    output = chunk_gdn_fwd_output_packed(q, k, v_new, h, cumulative_gate, scale, metadata)
    return output, final_state, inverse


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

    state_batch = q.shape[0] if metadata is None else metadata.cu_seqlens.shape[0] - 1
    recurrence_state = initial_state
    if recurrence_state is None:
        recurrence_state = torch.zeros(
            state_batch,
            v.shape[2],
            v.shape[-1],
            q.shape[-1],
            dtype=torch.float32,
            device=q.device,
        )
    w, u, qg, kg = chunk_gdn_recompute_w_u_qg_kg(q, k, v, cumulative_gate, beta, inverse, metadata)
    if metadata is None:
        h, v_new, _final_state = chunk_gdn_fwd_recurrence_dense(
            kg, w, u, cumulative_gate, recurrence_state
        )
    else:
        h, v_new, _final_state = chunk_gdn_fwd_recurrence_packed(
            kg, w, u, cumulative_gate, recurrence_state, metadata
        )

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
    return RaggedChunkMetadata(
        cu_seqlens,
        chunk_offsets,
        chunk_capacity(q.shape[1], cu_seqlens.shape[0] - 1, 64),
        64,
    )


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
