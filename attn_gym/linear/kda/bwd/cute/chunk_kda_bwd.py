"""Composed fixed-length and packed Blackwell KDA core backward."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd_v1_dispatch import (
    blackwell_delta_h_bwd_dhu_dispatch,
)
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import chunk_kda_bwd_intra
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
    chunk_kda_bwd_wy_dqkg,
)
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_dav import chunk_kda_bwd_dav
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.linear.kda.utils import ChunkMetadata


def chunk_kda_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    do: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    metadata: ChunkMetadata,
    *,
    chunk_size: int = 64,
    fastmath: bool = False,
    profile_ranges: bool = False,
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
    if tokens % chunk_size:
        raise ValueError("the composed KDA backward requires complete chunks")
    if initial_state is not None:
        initial_state = initial_state.contiguous()

    def record(name: str):
        return torch.profiler.record_function(name) if profile_ranges else nullcontext()

    # Forward deliberately saves only the minimal backward tape. Always
    # reconstruct the large W/U, gated Q/K, state, and corrected-value
    # intermediates here instead of retaining them for the lifetime of the graph.
    with record("kda/cute/backward_recompute_w_u"):
        w, u, qg, kg = recompute_w_u_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=g,
            metadata=metadata,
            chunk_size=chunk_size,
        )
    assert w is not None and qg is not None and kg is not None
    with record("kda/triton/backward_recompute_state"):
        h, v_new, _ = chunk_gated_delta_rule_fwd_h(
            kg,
            w,
            u,
            g,
            initial_state,
            chunk_size=chunk_size,
            output_final_state=False,
            cu_seqlens=metadata.cu_seqlens if metadata.has_multiple_sequences else None,
        )
    del u

    with record("kda/triton/backward_dav"):
        dv_intra, dAqk = chunk_kda_bwd_dav(
            v_new,
            Aqk,
            do,
            head_dim**-0.5,
            chunk_size=chunk_size,
            metadata=metadata if metadata.has_multiple_sequences else None,
        )
    with record("kda/cute/backward_delta_h"):
        dh, d_initial_state, dv = blackwell_delta_h_bwd_dhu_dispatch(
            qg,
            kg,
            w,
            do,
            dv_intra,
            gk=g,
            h0=initial_state,
            dht=d_final_state,
            scale=head_dim**-0.5,
            cu_seqlens=metadata.cu_seqlens if metadata.has_multiple_sequences else None,
            chunk_size=chunk_size,
            chunk_offsets=(
                metadata.cu_seqlens // chunk_size if metadata.has_multiple_sequences else None
            ),
            num_seqs=(
                metadata.cu_seqlens.new_full((1,), metadata.cu_seqlens.shape[0] - 1)
                if metadata.has_multiple_sequences
                else None
            ),
            num_chunks=metadata.chunk_indices.shape[0]
            if metadata.has_multiple_sequences
            else None,
        )
    del w, qg, kg, dv_intra
    with record("kda/cute/backward_wy_dqkg"):
        dq, dk, dv, dg, db, dAkk = chunk_kda_bwd_wy_dqkg(
            q,
            k,
            v,
            v_new,
            g,
            beta,
            Akk,
            h,
            do,
            dh,
            dv,
            metadata,
            chunk_size=chunk_size,
            fastmath=fastmath,
        )
    del h, v_new, dh
    with record("kda/cute/backward_intra"):
        dq, dk, dg, db = chunk_kda_bwd_intra(
            q,
            k,
            g,
            beta,
            dAqk,
            dAkk,
            dq,
            dk,
            db,
            dg,
            metadata,
        )
    # The imported kernels accumulate derivatives with respect to their exp2
    # exponent as though it were a natural exponent. Convert to d/d(log2 gate).
    dg.mul_(math.log(2.0))
    return dq, dk, dv, dg, db, d_initial_state


__all__ = ["chunk_kda_bwd"]
