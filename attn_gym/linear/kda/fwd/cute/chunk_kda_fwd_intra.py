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
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    chunk_kda_fwd_inter_solve_cute,
    chunk_kda_fwd_inter_solve_ragged_cute,
)
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_intra_diagonal,
)
from attn_gym.linear.kda.utils import DEFAULT_CHUNK_SIZE


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    profile_ranges: bool = False,
    autotune: bool = True,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    assert chunk_size == 64, "chunk_kda_fwd_intra CuTe path requires chunk_size=64"

    B, T, H, K = k.shape
    _, _, _, V = v.shape
    BT = chunk_size
    BC = 16
    assert B == 1, f"chunk_kda_fwd_intra CuTe path requires B=1, got B={B}"
    assert K == 128, f"chunk_kda_fwd_intra CuTe path requires K=128, got K={K}"
    assert V == 128, f"chunk_kda_fwd_intra CuTe path requires V=128, got V={V}"

    if isinstance(k, FakeTensor):
        Aqk = torch.empty((B, T, H, BT), device=k.device, dtype=k.dtype)
        Akkd = torch.empty((B, T, H, BC), device=k.device, dtype=torch.float32)
    else:
        with (
            torch.profiler.record_function("kda/triton/intra_subchunk")
            if profile_ranges
            else nullcontext()
        ):
            Aqk, Akkd = chunk_kda_fwd_intra_diagonal(
                q=q,
                k=k,
                g=gk,
                beta=beta,
                scale=scale,
                metadata=metadata,
                chunk_size=BT,
            )

    with (
        torch.profiler.record_function("kda/cute/inter_solve") if profile_ranges else nullcontext()
    ):
        if metadata is not None:
            Aqk, Akk = chunk_kda_fwd_inter_solve_ragged_cute(
                q=q,
                k=k,
                gk=gk,
                beta=beta,
                Akkd=Akkd,
                Aqk=Aqk,
                scale=scale,
                metadata=metadata,
            )
        else:
            Aqk, Akk = chunk_kda_fwd_inter_solve_cute(
                q=q,
                k=k,
                gk=gk,
                beta=beta,
                Akkd=Akkd,
                scale=scale,
                chunk_size=BT,
                Aqk=Aqk,
                profile_ranges=profile_ranges,
            )

    with (
        torch.profiler.record_function("kda/cute/recompute_w_u")
        if profile_ranges
        else nullcontext()
    ):
        w, u, _qg, kg = recompute_w_u_fwd(
            autotune=autotune,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=gk,
            metadata=metadata,
        )
    assert kg is not None
    return w, u, kg, Aqk, Akk


__all__ = ["chunk_kda_fwd_intra"]
