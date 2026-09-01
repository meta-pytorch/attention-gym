"""Composed fixed-length and packed architecture-routed KDA core backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from attn_gym._backends.cute.utils import get_device_properties
from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
    blackwell_delta_h_bwd_dhu_dv_fused_dispatch,
)
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import chunk_kda_bwd_intra
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
    chunk_kda_bwd_wy_dqkg,
)
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk import chunk_kda_bwd_daqk
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_delta_h_triton import (
    chunk_kda_bwd_delta_h_triton,
)
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_wy_triton import chunk_kda_bwd_wy_triton
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.linear.kda.utils import profiler_range


@dataclass
class ChunkKDABwdPrepared:
    """Consumable local tensors shared across the CP backward communication boundary."""

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
    Akk: torch.Tensor,
    do: torch.Tensor,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    *,
    scale: float,
    chunk_size: int,
    autotune: bool,
) -> ChunkKDABwdPrepared:
    """Recompute local forward state and output factors before CP communication."""
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
    with profiler_range("kda/triton/backward_daqk"):
        d_aqk = chunk_kda_bwd_daqk(
            v_new,
            do,
            scale,
            chunk_size=chunk_size,
            metadata=metadata,
        )
    return ChunkKDABwdPrepared(w, qg, kg, h, v_new, d_aqk)


def _finish_chunk_kda_bwd(
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
    prepared: ChunkKDABwdPrepared,
    *,
    scale: float,
    chunk_size: int,
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
    assert prepared.qg is not None and prepared.kg is not None and prepared.w is not None
    use_triton_backend = get_device_properties(q.device).major < 10
    range_backend = "triton" if use_triton_backend else "cute"
    with profiler_range(f"kda/{range_backend}/backward_delta_h"):
        if use_triton_backend:
            dh, d_initial_state, dv = chunk_kda_bwd_delta_h_triton(
                prepared.qg,
                prepared.kg,
                prepared.w,
                do,
                Aqk,
                gk=g,
                initial_state=initial_state,
                d_final_state=d_final_state,
                scale=scale,
                metadata=metadata,
            )
        else:
            dh, d_initial_state, dv = blackwell_delta_h_bwd_dhu_dv_fused_dispatch(
                prepared.qg,
                prepared.kg,
                prepared.w,
                do,
                Aqk,
                gk=g,
                h0=initial_state,
                dht=d_final_state,
                scale=scale,
                chunk_size=chunk_size,
                metadata=metadata,
            )
    prepared.qg = prepared.kg = prepared.w = None

    assert prepared.v_new is not None and prepared.h is not None
    with profiler_range(f"kda/{range_backend}/backward_wy_dqkg"):
        if use_triton_backend:
            dq, dk, dv, dg, db, dAkk = chunk_kda_bwd_wy_triton(
                q,
                k,
                v,
                prepared.v_new,
                g,
                beta,
                Akk,
                prepared.h,
                do,
                dh,
                dv,
                metadata,
                scale=scale,
                fastmath=fastmath,
            )
        else:
            dq, dk, dv, dg, db, dAkk = chunk_kda_bwd_wy_dqkg(
                q,
                k,
                v,
                prepared.v_new,
                g,
                beta,
                Akk,
                prepared.h,
                do,
                dh,
                dv,
                metadata,
                scale=scale,
                chunk_size=chunk_size,
                fastmath=fastmath,
                autotune=autotune,
            )
    prepared.h = prepared.v_new = None
    del dh

    assert prepared.d_aqk is not None
    with profiler_range("kda/cute/backward_intra"):
        dq, dk, dg, db = chunk_kda_bwd_intra(
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
        )
    prepared.d_aqk = None
    return dq, dk, dv, dg, db, d_initial_state


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
    scale: float,
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

    # Forward deliberately saves only minimal factors. Reconstruct large local
    # tensors before the optional CP reverse-summary communication boundary.
    prepared = _prepare_chunk_kda_bwd(
        q,
        k,
        v,
        g,
        beta,
        Akk,
        do,
        initial_state,
        metadata,
        scale=scale,
        chunk_size=chunk_size,
        autotune=autotune,
    )
    return _finish_chunk_kda_bwd(
        q,
        k,
        v,
        g,
        beta,
        Aqk,
        Akk,
        do,
        d_final_state,
        initial_state,
        metadata,
        prepared,
        scale=scale,
        chunk_size=chunk_size,
        fastmath=fastmath,
        autotune=autotune,
    )


__all__ = ["chunk_kda_bwd"]
