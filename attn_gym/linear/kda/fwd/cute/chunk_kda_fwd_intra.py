# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# KDA forward factor composition. Pre-Blackwell GPUs use the FP32/TF32 BC16
# diagonal stage plus Triton K3/K4; Blackwell retains its CuTe factor engines.

from __future__ import annotations

from contextlib import nullcontext

import torch
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.cute.utils import get_device_properties
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.constants import DEFAULT_CHUNK_SIZE
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    chunk_kda_fwd_inter_solve_cute,
    chunk_kda_fwd_inter_solve_ragged_cute,
    chunk_kda_fwd_k4b_dense_cute,
    chunk_kda_fwd_k4b_ragged_cute,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine import kda_intra_engine_fwd
from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_intra_diagonal,
)
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_k3_triton import (
    chunk_kda_fwd_k3b_triton,
)
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_k4_triton import (
    chunk_kda_fwd_k4b_triton,
)


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

    if get_device_properties(k.device).major < 10:
        Aqk, Akkd = chunk_kda_fwd_intra_diagonal(q, k, gk, beta, scale, metadata, chunk_size)
        AkkOD = chunk_kda_fwd_k3b_triton(q, k, gk, beta, Aqk, scale, metadata)
        return Aqk, chunk_kda_fwd_k4b_triton(AkkOD, Akkd, metadata, output_dtype=q.dtype)

    if q.dtype == torch.float16:
        # The engine stores two-sided diagonal rebase factors in the I/O dtype. Their
        # public gate-bound exponent fits BF16 but can overflow FP16, so keep the
        # diagonal products in FP32/TF32 and use FP16 only for the safe solved factors.
        Aqk, Akkd = chunk_kda_fwd_intra_diagonal(q, k, gk, beta, scale, metadata, chunk_size)
        if metadata is None:
            return chunk_kda_fwd_inter_solve_cute(
                q,
                k,
                gk,
                beta,
                Akkd,
                scale,
                chunk_size,
                Aqk=Aqk,
                profile_ranges=profile_ranges,
            )
        return chunk_kda_fwd_inter_solve_ragged_cute(
            q,
            k,
            gk,
            beta,
            Akkd,
            Aqk,
            scale,
            metadata,
        )

    with (
        torch.profiler.record_function("kda/cute/intra_engine")
        if profile_ranges
        else nullcontext()
    ):
        Aqk, AkkOD, Akkd = kda_intra_engine_fwd(q, k, gk, beta, scale, metadata)
        Akk = (
            chunk_kda_fwd_k4b_dense_cute(AkkOD, Akkd, chunk_size, output_dtype=q.dtype)
            if metadata is None
            else chunk_kda_fwd_k4b_ragged_cute(AkkOD, Akkd, metadata, output_dtype=q.dtype)
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
