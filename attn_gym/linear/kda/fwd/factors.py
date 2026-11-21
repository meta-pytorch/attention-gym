# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# KDA forward factor composition: the FP32/TF32 BC16 diagonal stage plus Triton K3/K4 on every
# supported GPU.

from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.profiler import profiler_range
from attn_gym.linear._delta_rule.triton.chunk_scheduler import RaggedChunkMetadata, ScheduleRequest
from attn_gym.linear.kda.constants import DEFAULT_CHUNK_SIZE
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_intra_diagonal,
)
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_k3_triton import (
    chunk_kda_fwd_k3b_triton,
)
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_k4_triton import (
    chunk_kda_fwd_k4b_triton,
)
from attn_gym.linear.kda.fwd.triton.recompute_w_u import recompute_w_u_fwd_triton


def chunk_kda_fwd_factors(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    fastmath: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the BT64 Aqk/Akk factors used by forward and backward recompute."""
    assert chunk_size == 64, "chunk_kda_fwd_factors requires chunk_size=64"
    batch, tokens, heads, key_dim = k.shape
    assert batch == 1, f"chunk_kda_fwd_factors requires B=1, got B={batch}"
    assert key_dim == 128, f"chunk_kda_fwd_factors requires K=128, got K={key_dim}"

    if isinstance(k, FakeTensor):
        shape = (batch, tokens, heads, chunk_size)
        return k.new_empty(shape), k.new_empty(shape)

    Aqk, Akkd = chunk_kda_fwd_intra_diagonal(
        q, k, gk, beta, scale, metadata, chunk_size, fastmath=fastmath
    )
    AkkOD = chunk_kda_fwd_k3b_triton(q, k, gk, beta, Aqk, scale, metadata, fastmath=fastmath)
    return Aqk, chunk_kda_fwd_k4b_triton(AkkOD, Akkd, metadata, output_dtype=q.dtype)


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    autotune: bool,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
    fastmath: bool = False,
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
        fastmath=fastmath,
    )
    with profiler_range("kda/triton/recompute_w_u"):
        w, u, _qg, kg = recompute_w_u_fwd_triton(
            autotune=autotune,
            schedule=schedule,
            k=k,
            v=v,
            beta=beta,
            A=Akk,
            gk=gk,
            metadata=metadata,
            fastmath=fastmath,
        )
    assert kg is not None
    return w, u, kg, Aqk, Akk


__all__ = ["chunk_kda_fwd_factors", "chunk_kda_fwd_intra"]
