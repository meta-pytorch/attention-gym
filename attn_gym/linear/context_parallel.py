# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallelism for delta-rule attention: ownership plans, state routing, one recipe.

Data model (NOTE [Terminology] in ``attn_gym.linear.state_summary``). The global stream is a list
of sequences. Each rank owns a list of fragments, contiguous global token ranges, and lays them out
back to back as its span. The plan cuts every fragment at sequence boundaries into
:class:`Subsequence` pieces, one local ``cu_seqlens`` segment each, so contiguous shards, zig-zag
load balancing, and document-aligned partitions are just different fragment lists. Because the
recurrence is affine, a subsequence's entry state is its predecessors' summaries folded from zero
and its exit cotangent is its successors' reverse summaries folded from zero.

Communication model. Every rank fills ``plan.slots`` summary slots in span order (identity where
nobody downstream needs one) and all-gathers them once per direction, so every
``gathered[rank][slot]`` index is a host-static integer and CUDA Graph capture works as long as the
plan does not change. The short-convolution halo reuses the plan: each subsequence's history is the
tail of its predecessor subsequences.

Ops. ``context_parallel_chunk`` is generic over the delta-rule variant through a :class:`StagedOp`
pair of ``prepare`` / ``prepare_backward`` callables whose handles expose ``state_summary``,
``run``, and ``saved`` (``attn_gym.linear.kda.stages``, ``attn_gym.linear.gdn.stages``). It is one
recipe, not an extension point: for a point-to-point pipeline, a recursive-doubling scan over
``compose_summaries``, DTensor, or communication overlap, copy the autograd function and swap the
collective.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import NamedTuple, Protocol

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_gather_single

from attn_gym.linear.kda.utils import profiler_range
from attn_gym.linear.state_summary import merge_state, neutral_summary


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


def _check_group(plan: ContextParallelPlan, group: dist.ProcessGroup, tokens: int) -> None:
    """Reject a plan built for a different group, rank, or local token count (host-only)."""
    if dist.get_world_size(group) != len(plan.table):
        raise ValueError(
            f"plan has {len(plan.table)} ranks but the group has {dist.get_world_size(group)}"
        )
    if dist.get_rank(group) != plan.cp_rank:
        raise ValueError(
            f"plan was built for cp_rank {plan.cp_rank}, running on {dist.get_rank(group)}"
        )
    if tokens != plan.cu_seqlens[-1]:
        raise ValueError(
            f"plan owns {plan.cu_seqlens[-1]} local tokens but the input has {tokens}"
        )


def _all_gather_slots(local_slots: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather every rank's ``[slots, ...]`` buffer into ``[world, slots, ...]``."""
    gathered = local_slots.new_empty(dist.get_world_size(group), *local_slots.shape)
    with profiler_range("cp/all_gather"):
        dist.all_gather_single(gathered, local_slots.contiguous(), group=group)
    return gathered


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


def context_parallel_conv_history(
    qkv: torch.Tensor,
    plan: ContextParallelPlan,
    group: dist.ProcessGroup,
    history_length: int,
) -> torch.Tensor | None:
    """Exchange subsequence tails and return ``causal_conv1d``'s packed ``initial_state``.

    ``qkv`` is the rank's ``[1, T, C]`` span. The collective is differentiable, so
    gradients flow back to the ranks that own the history tokens.
    """
    if history_length == 0:
        return None
    _check_group(plan, group, qkv.shape[1])
    tails = []
    for piece, (start, stop) in zip(plan.subsequences, pairwise(plan.cu_seqlens), strict=True):
        stored = min(history_length, piece.length)
        tails.append(F.pad(qkv[0, stop - stored : stop], (0, 0, history_length - stored, 0)))
    tails.extend([qkv.new_zeros(history_length, qkv.shape[-1])] * (plan.slots - len(tails)))
    with profiler_range("cp/conv_halo"):
        gathered = all_gather_single(torch.stack(tails), gather_dim=0, group=group).view(
            len(plan.table), plan.slots, history_length, qkv.shape[-1]
        )
    return compose_conv_histories(gathered, plan)


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


class StagedOp(NamedTuple):
    """A delta-rule variant's staged entry points with every op-specific option already bound.

    ``prepare(q, k, v, gate, beta, *, cu_seqlens)`` returns a :class:`PreparedForward`;
    ``prepare_backward(saved, d_output, initial_state)`` returns a :class:`PreparedBackward`,
    where ``saved`` is the forward handle's ``saved`` NamedTuple rebuilt from
    ``ctx.saved_tensors``. Options both stages must agree on, such as the resolved ``scale``,
    are bound into both callables by the op's binding (``attn_gym.linear.kda.context_parallel``).
    """

    prepare: Callable[..., PreparedForward]
    prepare_backward: Callable[..., PreparedBackward]


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


class _ContextParallelChunk(torch.autograd.Function):
    """Explicit-adjoint delta rule: summaries and one all-gather in each direction."""

    @staticmethod
    def forward(ctx, q, k, v, gate, beta, cu_seqlens, plan, group, stages):
        prepared = stages.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
        neutral = neutral_summary(v.shape[2], v.shape[-1], q.shape[-1], device=q.device)
        gathered = _all_gather_slots(summary_slots(prepared, plan, neutral), group)
        initial_state = compose_entry_states(gathered, plan)
        with profiler_range("cp/run"):
            output, final_state = prepared.run(initial_state, output_final_state=True)
        assert final_state is not None

        saved = prepared.saved
        ctx.save_for_backward(*saved, initial_state)
        ctx.saved_type = type(saved)
        ctx.plan = plan
        ctx.group = group
        ctx.stages = stages
        ctx.set_materialize_grads(False)
        return output, final_state

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output, d_final_state):
        *saved, initial_state = ctx.saved_tensors
        grads = ctx.stages.prepare_backward(ctx.saved_type._make(saved), d_output, initial_state)
        neutral = neutral_summary(*initial_state.shape[1:], device=initial_state.device)
        gathered = _all_gather_slots(
            grad_summary_slots(grads, d_final_state, ctx.plan, neutral), ctx.group
        )
        exit_cotangent = compose_exit_cotangents(gathered, d_final_state, ctx.plan)
        with profiler_range("cp/run"):
            dq, dk, dv, dgate, dbeta, _d_initial_state = grads.run(exit_cotangent)
        return dq, dk, dv, dgate, dbeta, None, None, None, None


def context_parallel_chunk(
    stages: StagedOp,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    plan: ContextParallelPlan,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a staged delta-rule op over this rank's span with state exchanged by all-gather.

    Args:
        stages: The op's entry points; ``attn_gym.linear.kda.context_parallel_kda`` and
            ``attn_gym.linear.gdn.context_parallel_gdn`` bind them with the op's options.
        q: This rank's span of queries, laid out as ``plan.subsequences``; ``k``, ``v``,
            ``gate``, and ``beta`` follow the same layout and the op's ``chunk_*`` contract.
        cu_seqlens: Device ``int32`` copy of ``plan.cu_seqlens``.
        plan: Routing for this rank from ``ContextParallelPlan.from_fragments``.
        group: Process group containing exactly the plan's ranks, in order.

    Returns:
        The span's output and one FP32 ``[N, HV, V, K]`` exit state per subsequence. Only the
        entries listed in ``plan.terminal`` are the sequences' true final states; the others are
        intermediate states that the owner of the next subsequence continued from.

    ``initial_state`` is not accepted: every sequence starts from zero and the plan supplies the
    entry states. Every rank in ``group`` must call this function the same number of times.
    """
    _check_group(plan, group, q.shape[1])
    if cu_seqlens.shape[0] != len(plan.subsequences) + 1:
        raise ValueError(
            f"cu_seqlens describes {cu_seqlens.shape[0] - 1} segments but the plan has "
            f"{len(plan.subsequences)} subsequences"
        )
    return _ContextParallelChunk.apply(q, k, v, gate, beta, cu_seqlens, plan, group, stages)


__all__ = [
    "ContextParallelPlan",
    "PreparedBackward",
    "PreparedForward",
    "StagedOp",
    "Subsequence",
    "compose_conv_histories",
    "compose_entry_states",
    "compose_exit_cotangents",
    "context_parallel_chunk",
    "context_parallel_conv_history",
    "grad_summary_slots",
    "summary_slots",
]
