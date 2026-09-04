# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Span preparation shared by the staged KDA and GDN primitives."""

from __future__ import annotations

from typing import NamedTuple

import torch

from attn_gym._backends.cute import normalize_compact_tensor, normalize_tma_rows
from attn_gym.linear._delta_rule.validation import resolve_scale
from attn_gym.linear.kda.chunk_schedule import RaggedChunkMetadata, prepare_ragged_chunk_metadata

CHUNK_SIZE = 64


class PreparedSpan(NamedTuple):
    """Normalized operands and chunk schedule of one ``B=1`` span."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    beta: torch.Tensor  # FP32, compact
    metadata: RaggedChunkMetadata | None  # None on the dense path
    cu_seqlens: torch.Tensor | None
    chunk_offsets: torch.Tensor | None
    scale: float


def prepare_span(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
    scale: float | None,
) -> PreparedSpan:
    """Normalize the operands of a ``B=1`` span and build its chunk schedule.

    A dense span with a partial tail chunk runs as one packed sequence; ``torch.arange`` keeps that
    launch capture-safe.
    """
    if q.shape[0] != 1:
        raise ValueError("staged primitives require B=1; pack sequences with cu_seqlens")
    tokens = q.shape[1]
    if cu_seqlens is None and tokens % CHUNK_SIZE:
        cu_seqlens = torch.arange(2, dtype=torch.int32, device=q.device) * tokens
    metadata = (
        None
        if cu_seqlens is None
        else prepare_ragged_chunk_metadata(cu_seqlens, tokens, CHUNK_SIZE)
    )
    q, k, v = (normalize_tma_rows(tensor) for tensor in (q, k, v))
    return PreparedSpan(
        q,
        k,
        v,
        normalize_compact_tensor(beta.float()),
        metadata,
        None if metadata is None else metadata.cu_seqlens,
        None if metadata is None else metadata.chunk_offsets,
        resolve_scale(scale, q.shape[-1]),
    )


def zero_state(
    q: torch.Tensor, v: torch.Tensor, metadata: RaggedChunkMetadata | None
) -> torch.Tensor:
    """FP32 ``[N, HV, V, K]`` zeros, one per packed sequence (per batch row when dense)."""
    sequences = q.shape[0] if metadata is None else metadata.cu_seqlens.shape[0] - 1
    return q.new_zeros(sequences, v.shape[2], v.shape[-1], q.shape[-1], dtype=torch.float32)
