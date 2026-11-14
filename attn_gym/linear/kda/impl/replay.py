# SPDX-License-Identifier: BSD-3-Clause

"""Replay-backed paged chunk KDA, shared by the fused and warpdrive backends.

``state_cache`` holds each slot's state at its last complete 64-token chunk boundary and
``replay_state`` the unfinished suffix (see ``paged_chunk_kda``). The Triton replay-cache
kernels move tokens and boundary states in and out of those caches; a chunk backend supplies
only the two forwards below. Because both backends' chunk forwards are prefix-invariant and
restart exactly from an FP32 boundary state, replay reproduces the backend's ``chunk_kda`` over
the whole stream whenever both run the same plan (docs/linear.md).
"""

from __future__ import annotations

from typing import Protocol

import torch

from attn_gym.linear.kda.ops import (
    chunk_replay_commit_op,
    chunk_replay_prefill_commit_op,
    chunk_replay_prefill_prepare_op,
    chunk_replay_prepare_op,
    chunk_replay_state_gather_op,
)
from attn_gym.linear.types import ReplayState


class ChunkForward(Protocol):
    """A chunk forward over dense ``[B, 64, H, D]`` replay windows from FP32 entry states."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        *,
        output_final_state: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class PagedChunkForward(Protocol):
    """A state-only paged chunk forward that advances ``state_cache`` in place."""

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        state_cache: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        has_initial_state: torch.Tensor | None,
    ) -> torch.Tensor: ...


def replay_paged_chunk_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    replay_state: ReplayState,
    *,
    cu_seqlens: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    chunk_forward: ChunkForward,
    paged_forward: PagedChunkForward,
) -> torch.Tensor:
    """Run replay-backed decode (one token per dense batch row) or streaming prefill."""
    replay_q, replay_k, replay_v, replay_gate, replay_beta, replay_count = replay_state
    if cu_seqlens is None and q.shape[1] == 1:
        (
            active_q,
            active_k,
            active_v,
            active_gate,
            active_beta,
            active_state,
            active_counts,
        ) = chunk_replay_prepare_op(
            q.to(torch.bfloat16),
            k.to(torch.bfloat16),
            v.to(torch.bfloat16),
            gate.float(),
            beta.float(),
            state_cache,
            replay_q,
            replay_k,
            replay_v,
            replay_gate,
            replay_beta,
            replay_count,
            state_indices,
            has_initial_state,
        )
        output, final_state = chunk_forward(
            active_q,
            active_k,
            active_v,
            active_gate,
            active_beta,
            active_state,
            output_final_state=True,
        )
        assert final_state is not None
        chunk_replay_commit_op(
            final_state,
            state_indices,
            has_initial_state,
            active_counts,
            state_cache,
            replay_count,
        )
        rows = torch.arange(active_counts.shape[0], dtype=torch.int64, device=q.device)
        return output[rows, active_counts].unsqueeze(1).to(q.dtype)

    if cu_seqlens is None:
        cu_seqlens = torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device) * q.shape[1]
    (
        prefix_q,
        prefix_k,
        prefix_v,
        prefix_gate,
        prefix_beta,
        prefix_cu_seqlens,
        output_route,
        tail_q,
        tail_k,
        tail_v,
        tail_gate,
        tail_beta,
        tail_counts,
    ) = chunk_replay_prefill_prepare_op(
        q.to(torch.bfloat16),
        k.to(torch.bfloat16),
        v.to(torch.bfloat16),
        gate.float(),
        beta.float(),
        state_cache,
        replay_q,
        replay_k,
        replay_v,
        replay_gate,
        replay_beta,
        replay_count,
        state_indices,
        has_initial_state,
        cu_seqlens,
    )
    prefix_output = paged_forward(
        prefix_q,
        prefix_k,
        prefix_v,
        prefix_gate,
        prefix_beta,
        state_cache,
        state_indices,
        cu_seqlens=prefix_cu_seqlens,
        has_initial_state=has_initial_state,
    )
    tail_output, _ = chunk_forward(
        tail_q,
        tail_k,
        tail_v,
        tail_gate,
        tail_beta,
        chunk_replay_state_gather_op(state_cache, state_indices),
        output_final_state=False,
    )
    return chunk_replay_prefill_commit_op(
        prefix_output,
        tail_output,
        output_route,
        tail_q,
        tail_k,
        tail_v,
        tail_gate,
        tail_beta,
        tail_counts,
        state_indices,
        replay_q,
        replay_k,
        replay_v,
        replay_gate,
        replay_beta,
        replay_count,
        q,
    )


__all__ = ["replay_paged_chunk_forward"]
