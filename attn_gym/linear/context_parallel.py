# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallelism for delta-rule attention: ownership plans and state routing.

Data model (NOTE [Terminology] in ``attn_gym.linear.state_summary``). The global stream is a list
of sequences. Each rank owns a list of fragments, contiguous global token ranges, and lays them out
back to back as its span. The plan cuts every fragment at sequence boundaries into
:class:`Subsequence` pieces, one local ``cu_seqlens`` segment each, so contiguous shards, zig-zag
load balancing, and document-aligned partitions are just different fragment lists. Because the
recurrence is affine, a subsequence's entry state is its predecessors' summaries folded from zero
and its exit cotangent is its successors' reverse summaries folded from zero.

Communication model. Every rank fills ``plan.slots`` summary slots in span order (identity where
nobody downstream needs one) and exchanges them once per direction, so every
``gathered[rank][slot]`` index is a host-static integer and CUDA Graph capture works as long as the
plan does not change. The short-convolution halo reuses the plan: each subsequence's history is the
tail of its predecessor subsequences. The composition helpers here are pure tensor code over the
gathered ``[world, slots, ...]`` buffers; the collective that fills them is the caller's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import NamedTuple, Protocol

import torch

from attn_gym.linear.kda.utils import profiler_range
from attn_gym.linear.state_summary import merge_state


@dataclass(frozen=True)
class Subsequence:
    """The part of one fragment inside one sequence: global tokens ``[start, stop)``."""

    sequence: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


# ``(rank, slot)`` address of a subsequence in every rank's gathered buffer.
Slot = tuple[int, int]


def _cut_at_sequences(
    cu_seqlens_global: Sequence[int], fragments: Sequence[tuple[int, int]]
) -> tuple[Subsequence, ...]:
    subsequences = []
    for fragment_start, fragment_stop in fragments:
        for sequence, (seq_start, seq_stop) in enumerate(pairwise(cu_seqlens_global)):
            start, stop = max(fragment_start, seq_start), min(fragment_stop, seq_stop)
            if start < stop:
                subsequences.append(Subsequence(sequence, start, stop))
    return tuple(subsequences)


def _validate_tiling(
    cu_seqlens_global: Sequence[int], fragments: Sequence[Sequence[tuple[int, int]]]
) -> None:
    """Require the ranks' fragments to tile ``[0, total_tokens)`` exactly once."""
    if len(cu_seqlens_global) < 2 or cu_seqlens_global[0] != 0:
        raise ValueError("cu_seqlens_global must start at zero and contain at least one sequence")
    if any(end < start for start, end in pairwise(cu_seqlens_global)):
        raise ValueError("cu_seqlens_global must be nondecreasing")
    if not fragments or any(not owned for owned in fragments):
        raise ValueError("every rank must own at least one fragment")
    for start, stop in (bounds for owned in fragments for bounds in owned):
        if start >= stop:
            raise ValueError(f"fragment [{start}, {stop}) is empty")
    ordered = sorted(bounds for owned in fragments for bounds in owned)
    covered = [0, *(stop for _, stop in ordered)]
    starts = [start for start, _ in ordered]
    if starts != covered[:-1] or covered[-1] != cu_seqlens_global[-1]:
        raise ValueError(
            f"fragments must tile [0, {cu_seqlens_global[-1]}) exactly once, got {ordered}"
        )


@dataclass(frozen=True)
class ContextParallelPlan:
    """Host-static routing for one rank derived from every rank's fragments.

    Attributes:
        table: ``table[cp_rank][slot]`` is the subsequence at span segment ``slot`` on that rank.
        cp_rank: This rank's row in ``table``: its rank within the context-parallel group.
        subsequences: This rank's subsequences in span order.
        cu_seqlens: Span offsets, one segment per subsequence.
        slots: Gather slots per rank, ``max(len(row) for row in table)``.
        predecessors: Per local subsequence, the addresses of the same sequence's earlier
            subsequences in increasing token order.
        successors: Per local subsequence, the addresses of the same sequence's later
            subsequences in decreasing token order, i.e. the order a reverse fold visits them.
        terminal: Local indices of subsequences that end their sequence, whose exit states are
            the sequence's true final states.
    """

    table: tuple[tuple[Subsequence, ...], ...]
    cp_rank: int
    subsequences: tuple[Subsequence, ...]
    cu_seqlens: tuple[int, ...]
    slots: int
    predecessors: tuple[tuple[Slot, ...], ...]
    successors: tuple[tuple[Slot, ...], ...]
    terminal: tuple[int, ...]

    @classmethod
    def from_fragments(
        cls,
        cu_seqlens_global: Sequence[int],
        fragments: Sequence[Sequence[tuple[int, int]]],
        cp_rank: int,
    ) -> ContextParallelPlan:
        """Build ``cp_rank``'s plan from every rank's fragments, half-open global token ranges.

        ``fragments[cp_rank]`` lists that rank's fragments in span order; the plan cuts each one at
        sequence boundaries. Cut points need no alignment to sequences or chunks. Together the
        fragments must tile ``[0, total_tokens)`` exactly once.
        """
        _validate_tiling(cu_seqlens_global, fragments)
        if not 0 <= cp_rank < len(fragments):
            raise ValueError(f"cp_rank {cp_rank} is outside a table of {len(fragments)} ranks")
        table = tuple(_cut_at_sequences(cu_seqlens_global, owned) for owned in fragments)
        addresses = {
            subsequence: (owner, slot)
            for owner, row in enumerate(table)
            for slot, subsequence in enumerate(row)
        }
        by_sequence: dict[int, list[Subsequence]] = {}
        for subsequence in addresses:
            by_sequence.setdefault(subsequence.sequence, []).append(subsequence)
        for siblings in by_sequence.values():
            siblings.sort(key=lambda subsequence: subsequence.start)

        local = table[cp_rank]
        predecessors, successors, terminal = [], [], []
        for index, subsequence in enumerate(local):
            siblings = by_sequence[subsequence.sequence]
            earlier = [s for s in siblings if s.start < subsequence.start]
            later = [s for s in siblings if s.start > subsequence.start]
            predecessors.append(tuple(addresses[s] for s in earlier))
            successors.append(tuple(addresses[s] for s in reversed(later)))
            if not later:
                terminal.append(index)
        return cls(
            table=table,
            cp_rank=cp_rank,
            subsequences=local,
            cu_seqlens=tuple(accumulate((s.length for s in local), initial=0)),
            slots=max(len(row) for row in table),
            predecessors=tuple(predecessors),
            successors=tuple(successors),
            terminal=tuple(terminal),
        )

    def global_token_ids(self, device: torch.device | str) -> torch.Tensor:
        """Global token id of every span token: the bridge from global tensors to the span."""
        return torch.cat([torch.arange(s.start, s.stop, device=device) for s in self.subsequences])


def _fold_chains(gathered: torch.Tensor, chains: Sequence[Sequence[Slot]]) -> list[torch.Tensor]:
    """Apply each chain of gathered summaries to the zero state, sharing common prefixes.

    Subsequences of one sequence fold through the same earlier subsequences in the same order, so a
    rank that owns many pieces of one sequence needs O(pieces) merges, not O(pieces**2).
    """
    heads, packed, key_dim = gathered.shape[-3:]
    zero = gathered.new_zeros(heads, packed - key_dim, key_dim)
    after: dict[tuple[Slot, ...], torch.Tensor] = {(): zero}
    states = []
    for chain in chains:
        prefix = tuple(chain)
        for length in range(1, len(prefix) + 1):
            if prefix[:length] not in after:
                after[prefix[:length]] = merge_state(
                    after[prefix[: length - 1]], gathered[prefix[length - 1]]
                )
        states.append(after[prefix])
    return states


def compose_entry_states(gathered: torch.Tensor, plan: ContextParallelPlan) -> torch.Tensor:
    """Fold each local subsequence's predecessor summaries from the zero state.

    ``gathered`` is ``[world, slots, HV, V + K, K]``; the result is ``[N, HV, V, K]``.
    """
    return torch.stack(_fold_chains(gathered, plan.predecessors))


def compose_exit_cotangents(
    gathered: torch.Tensor,
    d_final_state: torch.Tensor | None,
    plan: ContextParallelPlan,
) -> torch.Tensor:
    """Fold each local subsequence's successor reverse summaries onto its own exit cotangent."""
    incoming = torch.stack(_fold_chains(gathered, plan.successors))
    return incoming if d_final_state is None else incoming + d_final_state


def compose_conv_histories(
    gathered_tails: torch.Tensor, plan: ContextParallelPlan
) -> torch.Tensor:
    """Build each local subsequence's ``W - 1`` token history from predecessor tails.

    ``gathered_tails`` is ``[world, slots, W - 1, C]`` where each slot holds the last ``W - 1``
    tokens of that subsequence, front-padded with zeros when the subsequence is shorter. Concatenating
    predecessor tails in token order and keeping the last valid tokens yields exactly the tokens
    preceding the subsequence, however short the predecessors are.
    """
    history_length = gathered_tails.shape[-2]
    histories = []
    for sources in plan.predecessors:
        pieces = [
            gathered_tails[owner, slot, -min(history_length, plan.table[owner][slot].length) :]
            for owner, slot in sources
        ]
        history = gathered_tails.new_zeros(history_length, gathered_tails.shape[-1])
        valid = min(history_length, sum(piece.shape[0] for piece in pieces))
        if valid:
            history = torch.cat((history[:-valid], torch.cat(pieces)[-valid:]))
        histories.append(history)
    # Every rank must stay in the collective's backward, even one whose subsequences all start
    # sequences; the zero-weighted term keeps the reduce-scatter in its graph.
    return torch.stack(histories) + gathered_tails.flatten()[0] * 0


class PreparedForward(Protocol):
    """Forward handle of a staged delta-rule op."""

    @property
    def saved(self) -> NamedTuple: ...

    def state_summary(self, start: int, stop: int) -> torch.Tensor: ...

    def run(
        self, initial_state: torch.Tensor | None, *, output_final_state: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class PreparedBackward(Protocol):
    """Backward handle of a staged delta-rule op."""

    def state_grad_summary(self, start: int, stop: int) -> torch.Tensor: ...

    def run(self, d_final_state: torch.Tensor | None) -> tuple[torch.Tensor, ...]: ...


def summary_slots(
    prepared: PreparedForward, plan: ContextParallelPlan, neutral: torch.Tensor
) -> torch.Tensor:
    """Fill this rank's ``[slots, HV, V + K, K]`` forward slots; identity where nobody needs one."""
    slots = [neutral] * plan.slots
    with profiler_range("cp/state_summary"):
        for index, (start, stop) in enumerate(pairwise(plan.cu_seqlens)):
            if plan.successors[index]:
                slots[index] = prepared.state_summary(start, stop)
    return torch.stack(slots)


def grad_summary_slots(
    grads: PreparedBackward,
    d_final_state: torch.Tensor | None,
    plan: ContextParallelPlan,
    neutral: torch.Tensor,
) -> torch.Tensor:
    """Fill this rank's reverse slots, folding the loss's direct exit-state cotangent into each.

    Upstream ranks need the total cotangent leaving a subsequence, so the bias sent is
    ``d_final_state @ R + C`` rather than the summary's zero-exit bias ``C``.
    """
    value_dim = neutral.shape[-2] - neutral.shape[-1]
    slots = [neutral] * plan.slots
    with profiler_range("cp/state_grad_summary"):
        for index, (start, stop) in enumerate(pairwise(plan.cu_seqlens)):
            if not plan.predecessors[index]:
                continue
            summary = grads.state_grad_summary(start, stop)
            if d_final_state is not None:
                summary = torch.cat(
                    (merge_state(d_final_state[index], summary), summary[:, value_dim:]), dim=-2
                )
            slots[index] = summary
    return torch.stack(slots)


__all__ = [
    "ContextParallelPlan",
    "PreparedBackward",
    "PreparedForward",
    "Subsequence",
    "compose_conv_histories",
    "compose_entry_states",
    "compose_exit_cotangents",
    "grad_summary_slots",
    "summary_slots",
]
