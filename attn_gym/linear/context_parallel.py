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

Communication model. Only a fragment's last subsequence can continue on another fragment and only
its first can continue from one, so every rank fills one slot per fragment (``plan.slots`` is the
widest fragment list): forward, the summary of the fragment's last subsequence; backward, the
reverse map of its first; the identity where nobody downstream needs one. The slots are exchanged
once per direction. :meth:`ContextParallelPlan.routing` materializes a layout as device tensors
(token ranges for the summary launches, flat slot indices for the folds); by default sized for that
layout, or padded to caller-given caps (fragments per rank, subsequences per span) so a recipe
built on them replays under one CUDA Graph across document layouts and fragment tables. The
composition helpers here are pure tensor code over the gathered ``[world, slots, ...]`` buffers;
the collective that fills them is the caller's. The short-convolution halo reuses the same slots:
each fragment's slot holds the tail of its last subsequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import NamedTuple, Protocol

import torch

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


# One contiguous global token range a rank owns, as the subsequences it is cut into, in token
# order; its range is ``fragment[0].start`` to ``fragment[-1].stop``.
Fragment = tuple[Subsequence, ...]

# ``(cp_rank, fragment)`` address of a slot in every rank's gathered buffer.
Slot = tuple[int, int]


def _cut_at_sequences(
    cu_seqlens_global: Sequence[int], fragment_start: int, fragment_stop: int
) -> Fragment:
    subsequences = []
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
class ContextParallelRouting:
    """One layout of a plan as fixed-shape device tensors; see :meth:`ContextParallelPlan.routing`.

    Every shape depends only on the caps (``slots``, ``max_subsequences``, ``conv_history``) and
    the span length, so a CUDA Graph captured around a recipe that reads them replays with the
    tensors of another layout copied in place. Predecessors and successors are stored per local
    fragment, since only a fragment's first subsequence has predecessors and only its last has
    successors;
    ``entry_sources`` and ``exit_sources`` map every span segment to its fragment's folded state or
    to zero.

    Attributes:
        cu_seqlens: ``int32 [max_subsequences + 1]`` span offsets, padded with the span length.
        forward_bounds: ``int32 [slots, 2]`` range of each local fragment's last subsequence when
            it continues elsewhere, else empty; feeds ``state_summaries``.
        reverse_bounds: ``int32 [slots, 2]`` range of each local fragment's first subsequence
            when it continues from elsewhere, else empty; feeds ``state_grad_summaries``.
        reverse_sources: ``int64 [slots]`` local index of each fragment's first subsequence, whose
            exit-state cotangent is folded into the reverse slot's bias.
        predecessors: ``int64 [slots, world * slots - 1]`` flat ``cp_rank * slots + fragment``
            indices of the slots before each fragment in its sequence, in token order, padded
            with ``world * slots`` (the identity).
        successors: Same for each fragment's successors, in reverse token order.
        entry_sources: ``int64 [max_subsequences]`` local fragment whose folded predecessors are
            a segment's entry state, or ``slots`` for a segment that starts its sequence.
        exit_sources: Same for exit cotangents: the fragment whose folded successors apply, or
            ``slots`` for a segment that ends its sequence (or is padding).
        terminal: ``bool [max_subsequences]`` marks subsequences that end their sequence.
        tail_sources: ``int64 [slots, conv_history]`` span token behind each position of the
            fragment's tail row, or ``tokens`` (a zero row) where the tail is shorter or the slot
            is unused; feeds ``context_parallel_conv_history``.
        conv_sources: ``int64 [max_subsequences, conv_history]`` flat
            ``(rank * slots + fragment) * conv_history + position`` index into the gathered tails
            for each token preceding a segment, or ``world * slots * conv_history`` (zero row).
    """

    cp_rank: int
    world: int
    slots: int
    tokens: int
    cu_seqlens: torch.Tensor
    forward_bounds: torch.Tensor
    reverse_bounds: torch.Tensor
    reverse_sources: torch.Tensor
    predecessors: torch.Tensor
    successors: torch.Tensor
    entry_sources: torch.Tensor
    exit_sources: torch.Tensor
    terminal: torch.Tensor
    tail_sources: torch.Tensor
    conv_sources: torch.Tensor


@dataclass(frozen=True)
class ContextParallelPlan:
    """Host ownership and routing metadata for one rank, derived from every rank's fragments.

    Attributes:
        table: ``table[cp_rank][slot]`` is that rank's fragment behind gather slot ``slot``.
        cp_rank: This rank's row in ``table``: its rank within the context-parallel group.
        predecessors: Per local fragment, the slots of the same sequence's pieces before its
            first subsequence, in increasing token order.
        successors: Per local fragment, the slots of the same sequence's pieces after its last
            subsequence, in decreasing token order, i.e. the order a reverse fold visits them.
        terminal: Local indices of subsequences that end their sequence, whose exit states are
            the sequence's true final states.
    """

    table: tuple[tuple[Fragment, ...], ...]
    cp_rank: int
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
        table = tuple(
            tuple(_cut_at_sequences(cu_seqlens_global, start, stop) for start, stop in owned)
            for owned in fragments
        )
        # Every fragment holds at most one piece of a given sequence, so each sequence's pieces
        # order its ``(cp_rank, fragment)`` slots by token.
        order: dict[int, list[tuple[int, Slot]]] = {}
        for owner, row in enumerate(table):
            for index, fragment in enumerate(row):
                for piece in fragment:
                    order.setdefault(piece.sequence, []).append((piece.start, (owner, index)))
        for slots_of_sequence in order.values():
            slots_of_sequence.sort()

        # Only a fragment's first subsequence can be continued and only its last can continue.
        def neighbors(slot: int, piece: Subsequence) -> tuple[list[Slot], list[Slot]]:
            sequence_slots = [address for _, address in order[piece.sequence]]
            here = sequence_slots.index((cp_rank, slot))
            return sequence_slots[:here], sequence_slots[here + 1 :]

        local = table[cp_rank]
        subsequences = [piece for fragment in local for piece in fragment]
        return cls(
            table=table,
            cp_rank=cp_rank,
            predecessors=tuple(
                tuple(neighbors(slot, fragment[0])[0]) for slot, fragment in enumerate(local)
            ),
            successors=tuple(
                tuple(reversed(neighbors(slot, fragment[-1])[1]))
                for slot, fragment in enumerate(local)
            ),
            terminal=tuple(
                index
                for index, piece in enumerate(subsequences)
                if piece.stop == cu_seqlens_global[piece.sequence + 1]
            ),
        )

    @property
    def fragments(self) -> tuple[Fragment, ...]:
        """This rank's fragments in span order."""
        return self.table[self.cp_rank]

    @property
    def subsequences(self) -> tuple[Subsequence, ...]:
        """This rank's subsequences in span order."""
        return tuple(piece for fragment in self.fragments for piece in fragment)

    @property
    def cu_seqlens(self) -> tuple[int, ...]:
        """Span offsets, one segment per subsequence."""
        return tuple(accumulate((s.length for s in self.subsequences), initial=0))

    @property
    def slots(self) -> int:
        """Gather slots per rank: one per fragment of the widest row."""
        return max(len(row) for row in self.table)

    def global_token_ids(self, device: torch.device | str) -> torch.Tensor:
        """Global token id of every span token: the bridge from global tensors to the span."""
        return torch.cat(
            [torch.arange(f[0].start, f[-1].stop, device=device) for f in self.fragments]
        )

    def routing(
        self,
        device: torch.device | str,
        *,
        slots: int | None = None,
        max_subsequences: int | None = None,
        conv_history: int = 0,
    ) -> ContextParallelRouting:
        """Materialize this layout as device tensors; the caps make the shapes layout-independent.

        By default every tensor is sized for this layout alone. Under CUDA Graph replay pass the
        caps the loader promises never to exceed: ``slots`` (fragments per rank; default this
        table's maximum) and ``max_subsequences`` (segments per span; default this layout's
        count). Every layout within the caps and with the same span length then yields identically
        shaped tensors, whatever its fragment table. ``conv_history`` is the short convolution's
        ``W - 1`` (0: no halo tensors). Predecessor and successor rows are ``world * slots - 1`` wide.
        """
        cu_seqlens = self.cu_seqlens
        count = len(cu_seqlens) - 1
        if max_subsequences is None:
            max_subsequences = count
        if count > max_subsequences:
            raise ValueError(
                f"layout has {count} subsequences but max_subsequences is {max_subsequences}"
            )
        if slots is None:
            slots = self.slots
        if self.slots > slots:
            raise ValueError(f"a rank owns {self.slots} fragments but slots is {slots}")
        world = len(self.table)
        width = world * slots - 1  # the most slots a sequence can have before or after a fragment
        identity = world * slots
        tokens = cu_seqlens[-1]

        def flat(addresses: Sequence[Slot]) -> list[int]:
            indices = [owner * slots + slot for owner, slot in addresses]
            return indices + [identity] * (width - len(indices))

        forward_bounds = [[0, 0]] * slots
        reverse_bounds = [[0, 0]] * slots
        reverse_sources = [0] * slots
        predecessors = [flat(())] * slots
        successors = [flat(())] * slots
        entry_sources = [slots] * max_subsequences
        exit_sources = [slots] * max_subsequences
        tail_sources = [[tokens] * conv_history for _ in range(slots)]
        conv_sources = [[identity * conv_history] * conv_history for _ in range(max_subsequences)]
        first = 0
        for slot, fragment in enumerate(self.fragments):
            last = first + len(fragment) - 1
            if self.predecessors[slot]:
                reverse_bounds[slot] = [cu_seqlens[first], cu_seqlens[first + 1]]
                reverse_sources[slot] = first
                predecessors[slot] = flat(self.predecessors[slot])
                entry_sources[first] = slot
            if self.successors[slot]:
                forward_bounds[slot] = [cu_seqlens[last], cu_seqlens[last + 1]]
                successors[slot] = flat(self.successors[slot])
                exit_sources[last] = slot
            stored = min(conv_history, fragment[-1].length)
            stop = cu_seqlens[last + 1]
            tail_sources[slot][conv_history - stored :] = range(stop - stored, stop)
            # Predecessor tails in token order, each truncated to its real length; keep the last
            # ``conv_history`` tokens, front-padded with the zero row.
            picked: list[int] = []
            for owner, source in self.predecessors[slot]:
                stored = min(conv_history, self.table[owner][source][-1].length)
                base = (owner * slots + source) * conv_history
                picked.extend(range(base + conv_history - stored, base + conv_history))
            picked = picked[max(len(picked) - conv_history, 0) :]
            conv_sources[first][conv_history - len(picked) :] = picked
            first = last + 1
        terminal = [index in self.terminal for index in range(max_subsequences)]

        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device=device)

        return ContextParallelRouting(
            cp_rank=self.cp_rank,
            world=world,
            slots=slots,
            tokens=tokens,
            cu_seqlens=tensor([*cu_seqlens] + [tokens] * (max_subsequences - count), torch.int32),
            forward_bounds=tensor(forward_bounds, torch.int32),
            reverse_bounds=tensor(reverse_bounds, torch.int32),
            reverse_sources=tensor(reverse_sources, torch.int64),
            predecessors=tensor(predecessors, torch.int64),
            successors=tensor(successors, torch.int64),
            entry_sources=tensor(entry_sources, torch.int64),
            exit_sources=tensor(exit_sources, torch.int64),
            terminal=tensor(terminal, torch.bool),
            tail_sources=tensor(tail_sources, torch.int64),
            conv_sources=tensor(conv_sources, torch.int64),
        )


def fold_slots(gathered: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """Apply gathered summaries to the zero state, one row of slots at a time.

    ``gathered`` is ``[world, slots, HV, V + K, K]`` and ``sources`` is ``int64 [rows, L]`` of
    flat ``cp_rank * slots + fragment`` indices in application order, padded with
    ``world * slots``, which addresses the identity appended here. Every row costs ``L`` merges
    regardless of how many real slots it names, so the launch sequence is fixed and CUDA Graph
    capturable.
    """
    heads, packed, key_dim = gathered.shape[-3:]
    neutral = neutral_summary(heads, packed - key_dim, key_dim, device=gathered.device)
    flat = torch.cat((gathered.flatten(0, 1), neutral.unsqueeze(0)))
    state = gathered.new_zeros(sources.shape[0], heads, packed - key_dim, key_dim)
    for step in range(sources.shape[1]):
        state = merge_state(state, flat[sources[:, step]])
    return state


def _scatter_folds(folded: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """Map ``[slots, ...]`` folded states onto span segments; source ``slots`` selects zero."""
    return torch.cat((folded, folded.new_zeros(1, *folded.shape[1:])))[sources]


def compose_entry_states(gathered: torch.Tensor, routing: ContextParallelRouting) -> torch.Tensor:
    """Fold each fragment's predecessor summaries from zero and place them on its first segment.

    ``gathered`` is ``[world, slots, HV, V + K, K]``; the result is
    ``[max_subsequences, HV, V, K]``, zero for every segment that starts its sequence.
    """
    return _scatter_folds(fold_slots(gathered, routing.predecessors), routing.entry_sources)


def compose_exit_cotangents(
    gathered: torch.Tensor, d_final_state: torch.Tensor, routing: ContextParallelRouting
) -> torch.Tensor:
    """Fold each fragment's successor reverse maps onto its last segment's exit cotangent.

    ``d_final_state`` is the loss's own cotangent on every segment's exit state (zeros where the
    loss does not touch it); the fold adds what flows back from other ranks.
    """
    incoming = _scatter_folds(fold_slots(gathered, routing.successors), routing.exit_sources)
    return incoming + d_final_state


def compose_conv_histories(
    gathered_tails: torch.Tensor, routing: ContextParallelRouting
) -> torch.Tensor:
    """The ``conv_history`` tokens before each span segment, ``[max_subsequences, history, C]``.

    A short convolution at a segment's first tokens needs the tokens just before it, which end
    some other fragment, possibly on another rank. ``gathered_tails`` is every rank's slot tails,
    ``[world * slots, history, C]``; ``routing.conv_sources[segment, i]`` names the gathered token
    that is history position ``i`` (or the appended zero row when the sequence starts there), so
    the composition is one gather. Indexing keeps every rank's tails in the collective's backward,
    even ranks whose segments all start sequences.
    """
    flat = gathered_tails.flatten(0, 1)
    return torch.cat((flat, flat.new_zeros(1, flat.shape[-1])))[routing.conv_sources]


class PreparedForward(Protocol):
    """Forward handle of a staged delta-rule op."""

    @property
    def saved(self) -> NamedTuple: ...

    def state_summaries(self, bounds: torch.Tensor) -> torch.Tensor: ...

    def run(
        self, initial_state: torch.Tensor | None, *, output_final_state: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class PreparedBackward(Protocol):
    """Backward handle of a staged delta-rule op."""

    def state_grad_summaries(self, bounds: torch.Tensor) -> torch.Tensor: ...

    def run(self, d_final_state: torch.Tensor | None) -> tuple[torch.Tensor, ...]: ...


def summary_slots(prepared: PreparedForward, routing: ContextParallelRouting) -> torch.Tensor:
    """This rank's ``[slots, HV, V + K, K]`` forward slots: each fragment's last subsequence.

    Fragments whose last subsequence ends its sequence have an empty range and send the identity.
    """
    with profiler_range("cp/state_summaries"):
        return prepared.state_summaries(routing.forward_bounds)


def grad_summary_slots(
    grads: PreparedBackward, d_final_state: torch.Tensor, routing: ContextParallelRouting
) -> torch.Tensor:
    """This rank's reverse slots: each fragment's first subsequence, with the loss folded in.

    Upstream ranks need the total cotangent leaving that subsequence, so the bias sent is
    ``d_final_state @ R + C`` rather than the summary's zero-exit bias ``C``. A fragment whose
    first subsequence starts its sequence sends a slot nobody folds, so its bias is
    irrelevant.
    """
    with profiler_range("cp/state_grad_summary"):
        slots = grads.state_grad_summaries(routing.reverse_bounds)
    value_dim = d_final_state.shape[-2]
    direct = d_final_state[routing.reverse_sources]
    return torch.cat((merge_state(direct, slots), slots[..., value_dim:, :]), dim=-2)


__all__ = [
    "ContextParallelPlan",
    "ContextParallelRouting",
    "Fragment",
    "PreparedBackward",
    "PreparedForward",
    "Subsequence",
    "compose_conv_histories",
    "compose_entry_states",
    "compose_exit_cotangents",
    "fold_slots",
    "grad_summary_slots",
    "summary_slots",
]
