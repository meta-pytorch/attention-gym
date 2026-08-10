"""Composed fixed-length Blackwell KDA core backward."""

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
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_dav import chunk_kda_bwd_kernel_dAv
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h


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
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    num_chunks: torch.Tensor,
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
    """Differentiate the optimized fixed-length KDA core pipeline."""
    batch, tokens, heads, head_dim = q.shape
    value_dim = v.shape[-1]
    if batch != 1 or head_dim != 128 or value_dim != 128:
        raise ValueError("the composed KDA backward requires B=1 and K=V=128")
    if chunk_size != 64:
        raise ValueError(f"the composed KDA backward requires chunk_size=64, got {chunk_size}")
    if tokens % chunk_size:
        raise ValueError("the composed KDA backward requires complete chunks")
    chunks = tokens // chunk_size
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
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            num_chunks=num_chunks,
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
        )
    del u

    dv_intra = torch.empty_like(v_new)
    dAqk = torch.empty_like(Aqk, dtype=torch.float32)
    with record("kda/triton/backward_dav"):
        chunk_kda_bwd_kernel_dAv[(chunks, batch * heads)](
            q=q,
            k=k,
            v=v_new,
            A=Aqk,
            do=do,
            dv=dv_intra,
            dA=dAqk,
            cu_seqlens=None,
            chunk_indices=None,
            num_chunks=None,
            scale=head_dim**-0.5,
            T=tokens,
            H=heads,
            K=head_dim,
            V=value_dim,
            BT=chunk_size,
            BK=head_dim,
            BV=64,
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
            chunk_size=chunk_size,
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
            cu_seqlens,
            chunk_indices,
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
            cu_seqlens,
            chunk_indices,
            num_chunks,
        )
    # The imported kernels accumulate derivatives with respect to their exp2
    # exponent as though it were a natural exponent. Convert to d/d(log2 gate).
    dg.mul_(math.log(2.0))
    return dq, dk, dv, dg, db, d_initial_state


__all__ = ["chunk_kda_bwd"]
