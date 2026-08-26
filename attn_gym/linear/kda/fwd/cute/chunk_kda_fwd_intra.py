# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CuTe DSL KDA forward intra-chunk path: the persistent engine produces Aqk and
# K3-compatible factors, then K4b assembles the dense or ragged Akk inverse.

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.constants import DEFAULT_CHUNK_SIZE
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    chunk_kda_fwd_k4b_dense_cute,
    chunk_kda_fwd_k4b_ragged_cute,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine import kda_intra_engine_fwd
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd


def chunk_kda_fwd_factors(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    profile_ranges: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the BT64 Aqk/Akk factors used by forward and backward recompute."""
    assert chunk_size == 64, "chunk_kda_fwd_factors requires chunk_size=64"
    batch, tokens, heads, key_dim = k.shape
    assert batch == 1, f"chunk_kda_fwd_factors requires B=1, got B={batch}"
    assert key_dim == 128, f"chunk_kda_fwd_factors requires K=128, got K={key_dim}"

    if isinstance(k, FakeTensor):
        shape = (batch, tokens, heads, chunk_size)
        return k.new_empty(shape), k.new_empty(shape)

    with (
        torch.profiler.record_function("kda/cute/intra_engine")
        if profile_ranges
        else nullcontext()
    ):
        Aqk, AkkOD, Akkd = kda_intra_engine_fwd(q, k, gk, beta, scale, metadata)
        Akk = (
            chunk_kda_fwd_k4b_dense_cute(AkkOD, Akkd, chunk_size)
            if metadata is None
            else chunk_kda_fwd_k4b_ragged_cute(AkkOD, Akkd, metadata)
        )
    return Aqk, Akk


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
    """Produce the current BT64 forward intermediates and backward factors."""
    if v.shape[-1] != 128:
        raise ValueError(f"chunk_kda_fwd_intra requires V=128, got V={v.shape[-1]}")
    Aqk, Akk = chunk_kda_fwd_factors(
        q,
        k,
        gk,
        beta,
        scale,
        metadata,
        chunk_size=chunk_size,
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


__all__ = ["chunk_kda_fwd_factors", "chunk_kda_fwd_intra"]
