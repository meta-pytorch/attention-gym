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

import torch
import triton
from attn_gym.linear.kda.utils import (
    DEFAULT_CHUNK_SIZE,
    IS_GATHER_SUPPORTED,
    prepare_chunk_indices,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    chunk_kda_fwd_inter_solve_cute,
)
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_kernel_intra_sub_chunk_forloop,
)
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd
from torch._subclasses.fake_tensor import FakeTensor


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_indices: torch.LongTensor | None = None,
    num_seqs: int | None = None,
    num_chunks: int | torch.Tensor | None = None,
    safe_gate: bool = False,
    causal_gate_normref: bool = False,
    disable_recompute: bool = False,
    tf32x3_in_chunk_intra: bool = True,
    max_num_grid: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    del num_seqs
    del tf32x3_in_chunk_intra

    assert safe_gate, "chunk_kda_fwd_intra CuTe path requires safe_gate=True"
    assert chunk_size == 64, "chunk_kda_fwd_intra CuTe path requires chunk_size=64"

    if max_num_grid is None:
        max_num_grid = torch.cuda.get_device_properties(k.device).multi_processor_count
    if num_chunks is not None and not isinstance(num_chunks, torch.Tensor):
        num_chunks = torch.tensor(num_chunks, dtype=torch.int64, device=k.device)

    B, T, H, K = k.shape
    _, _, _, V = v.shape
    BT = chunk_size
    BC = 16
    BK = triton.next_power_of_2(K)
    NC = triton.cdiv(BT, BC)
    assert B == 1, f"chunk_kda_fwd_intra CuTe path requires B=1, got B={B}"
    assert K == 128, f"chunk_kda_fwd_intra CuTe path requires K=128, got K={K}"
    assert V == 128, f"chunk_kda_fwd_intra CuTe path requires V=128, got V={V}"

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    grid_NT = min(max_num_grid, NT)

    Aqk_flat = torch.empty(B * T, H * BT, device=k.device, dtype=k.dtype)
    Aqk = Aqk_flat.reshape(B, T, H, BT)
    Akkd_flat = torch.empty(B * T, H * BC, device=k.device, dtype=torch.float32)
    Akkd = Akkd_flat.reshape(B, T, H, BC)

    if not isinstance(k, FakeTensor):
        chunk_kda_fwd_kernel_intra_sub_chunk_forloop[(grid_NT, NC, B * H)](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            num_chunks=num_chunks,
            T=T,
            H=H,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=IS_GATHER_SUPPORTED,
            CAUSAL_NORMREF=causal_gate_normref,
            GRID_NT=grid_NT,
            MAX_NT=NT,
        )

    Akk = torch.empty(B, T, H, BT, device=k.device, dtype=k.dtype)
    Aqk, Akk = chunk_kda_fwd_inter_solve_cute(
        q=q,
        k=k,
        gk=gk,
        beta=beta,
        Akkd=Akkd,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=BT,
        chunk_indices=chunk_indices,
        num_chunks=num_chunks,
        Aqk=Aqk,
        Akk=Akk,
    )

    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        q=q if disable_recompute else None,
        gk=gk,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        num_chunks=num_chunks,
    )
    return w, u, qg, kg, Aqk, Akk


__all__ = ["chunk_kda_fwd_intra"]
