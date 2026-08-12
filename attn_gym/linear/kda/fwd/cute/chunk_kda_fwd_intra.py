# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CuTe DSL KDA forward intra-chunk path.
#
# This wrapper keeps the full forward-intra pipeline and delegates the
# inter-solve stage to the isolated K3b+K4b CuTe helper.

from __future__ import annotations

from contextlib import nullcontext

import torch
import triton
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    chunk_kda_fwd_inter_solve_cute,
)
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_kernel_intra_sub_chunk_forloop,
)
from attn_gym.linear.kda.utils import DEFAULT_CHUNK_SIZE, IS_GATHER_SUPPORTED, ChunkMetadata


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: ChunkMetadata,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    profile_ranges: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    assert chunk_size == 64, "chunk_kda_fwd_intra CuTe path requires chunk_size=64"

    max_num_grid = torch.cuda.get_device_properties(k.device).multi_processor_count

    B, T, H, K = k.shape
    _, _, _, V = v.shape
    BT = chunk_size
    BC = 16
    BK = triton.next_power_of_2(K)
    NC = triton.cdiv(BT, BC)
    assert B == 1, f"chunk_kda_fwd_intra CuTe path requires B=1, got B={B}"
    assert K == 128, f"chunk_kda_fwd_intra CuTe path requires K=128, got K={K}"
    assert V == 128, f"chunk_kda_fwd_intra CuTe path requires V=128, got V={V}"

    NT = triton.cdiv(T, BT)
    grid_NT = min(max_num_grid, NT)

    Aqk_flat = torch.empty(B * T, H * BT, device=k.device, dtype=k.dtype)
    Aqk = Aqk_flat.reshape(B, T, H, BT)
    Akkd_flat = torch.empty(B * T, H * BC, device=k.device, dtype=torch.float32)
    Akkd = Akkd_flat.reshape(B, T, H, BC)

    if not isinstance(k, FakeTensor):
        with (
            torch.profiler.record_function("kda/triton/intra_subchunk")
            if profile_ranges
            else nullcontext()
        ):
            chunk_kda_fwd_kernel_intra_sub_chunk_forloop[(grid_NT, NC, B * H)](
                q=q,
                k=k,
                g=gk,
                beta=beta,
                Aqk=Aqk,
                Akk=Akkd,
                scale=scale,
                cu_seqlens=metadata.cu_seqlens if metadata.has_multiple_sequences else None,
                chunk_indices=metadata.chunk_indices if metadata.has_multiple_sequences else None,
                num_chunks=metadata.num_chunks if metadata.has_multiple_sequences else None,
                T=T,
                q_stride_t=q.stride(1),
                q_stride_h=q.stride(2),
                k_stride_t=k.stride(1),
                k_stride_h=k.stride(2),
                H=H,
                K=K,
                BT=BT,
                BC=BC,
                BK=BK,
                USE_GATHER=IS_GATHER_SUPPORTED,
                CAUSAL_NORMREF=False,
                GRID_NT=grid_NT,
                MAX_NT=NT,
            )

    with (
        torch.profiler.record_function("kda/cute/inter_solve") if profile_ranges else nullcontext()
    ):
        Aqk, Akk = chunk_kda_fwd_inter_solve_cute(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Akkd=Akkd,
            scale=scale,
            cu_seqlens=metadata.cu_seqlens if metadata.has_multiple_sequences else None,
            chunk_size=BT,
            chunk_indices=metadata.chunk_indices if metadata.has_multiple_sequences else None,
            Aqk=Aqk,
            profile_ranges=profile_ranges,
        )

    with (
        torch.profiler.record_function("kda/cute/recompute_w_u")
        if profile_ranges
        else nullcontext()
    ):
        w, u, _qg, kg = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=gk,
            metadata=metadata,
        )
    assert w is not None and kg is not None
    return w, u, kg, Aqk, Akk


__all__ = ["chunk_kda_fwd_intra"]
