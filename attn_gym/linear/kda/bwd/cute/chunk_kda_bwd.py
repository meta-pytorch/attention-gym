"""Composed fixed-length and packed Blackwell KDA core backward."""

from __future__ import annotations

import torch

from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
    blackwell_delta_h_bwd_dhu_dv_fused_dispatch,
)
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import chunk_kda_bwd_intra
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
    chunk_kda_bwd_wy_dqkg,
)
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk import chunk_kda_bwd_daqk
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.linear.kda.utils import profiler_range


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
    metadata: RaggedChunkMetadata | None,
    *,
    chunk_size: int = 64,
    fastmath: bool = False,
    autotune: bool = True,
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
    scale = head_dim**-0.5

    # Forward deliberately saves only the minimal backward factors. Always
    # reconstruct the large W/U, gated Q/K, state, and corrected-value
    # intermediates here instead of retaining them for the lifetime of the graph.
    with profiler_range("kda/cute/backward_recompute_w_u"):
        w, u, qg, kg = recompute_w_u_fwd(
            q=q,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=g,
            metadata=metadata,
            chunk_size=chunk_size,
            autotune=autotune,
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
            autotune=autotune,
        )
    del u

    with profiler_range("kda/triton/backward_daqk"):
        dAqk = chunk_kda_bwd_daqk(
            v_new,
            do,
            scale,
            chunk_size=chunk_size,
            metadata=metadata,
        )
    with profiler_range("kda/cute/backward_delta_h"):
        dh, d_initial_state, dv = blackwell_delta_h_bwd_dhu_dv_fused_dispatch(
            qg,
            kg,
            w,
            do,
            Aqk,
            gk=g,
            h0=initial_state,
            dht=d_final_state,
            scale=scale,
            chunk_size=chunk_size,
            metadata=metadata,
        )
    del w, qg, kg
    with profiler_range("kda/cute/backward_wy_dqkg"):
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
            autotune=autotune,
        )
    del h, v_new, dh
    with profiler_range("kda/cute/backward_intra"):
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
            autotune=autotune,
        )
    return dq, dk, dv, dg, db, d_initial_state


__all__ = ["chunk_kda_bwd"]
