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

Communication model. Only a fragment's last subsequence can continue on another fragment and only
its first can continue from one, so every rank fills one slot per fragment (``plan.slots`` is the
widest fragment list): forward, the summary of the fragment's last subsequence; backward, the
reverse map of its first; the identity where nobody downstream needs one. The slots are
all-gathered once per direction. Because the slot count and the longest chain depend only on the
fragment table, not on where documents start and end, :meth:`ContextParallelPlan.routing`
materializes a layout as fixed-shape device tensors (token ranges for the summary launches, flat
slot indices for the folds) and ``context_parallel_chunk`` replays under one CUDA Graph across
document layouts. The short-convolution halo reuses the same slots: each fragment's slot holds the
tail of its last subsequence.

Ops. ``context_parallel_chunk`` is generic over the delta-rule variant through a :class:`StagedOp`
pair of ``prepare`` / ``prepare_backward`` callables whose handles expose ``state_summaries``,
``run``, and ``saved`` (``attn_gym.linear.kda.stages``, ``attn_gym.linear.gdn.stages``). It is one
recipe, not an extension point: for a point-to-point pipeline, a recursive-doubling scan over
``compose_summaries``, DTensor, or communication overlap, copy the autograd function and swap the
collective.
"""

from __future__ import annotations

import dataclasses
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


@dataclass(frozen=True)
class Fragment:
    """One contiguous global token range a rank owns, cut into its subsequences."""

    start: int
    stop: int
    subsequences: tuple[Subsequence, ...]


# ``(rank, fragment)`` address of a slot in every rank's gathered buffer.
Slot = tuple[int, int]


def _cut_at_sequences(
    cu_seqlens_global: Sequence[int], fragment: tuple[int, int]
) -> tuple[Subsequence, ...]:
    fragment_start, fragment_stop = fragment
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

    Every shape depends only on the fragment table (and the ``max_subsequences`` cap), so a CUDA
    Graph captured around ``context_parallel_chunk`` replays with the tensors of another layout
    copied in place. Chains are stored per local fragment, since only a fragment's first
    subsequence has predecessors and only its last has successors; ``entry_sources`` and
    ``exit_sources`` map every span segment to its fragment's folded state or to zero.

    Attributes:
        cu_seqlens: ``int32 [max_subsequences + 1]`` span offsets, padded with the span length.
        forward_bounds: ``int32 [slots, 2]`` range of each local fragment's last subsequence when
            it continues elsewhere, else empty; feeds ``state_summaries``.
        reverse_bounds: ``int32 [slots, 2]`` range of each local fragment's first subsequence
            when it continues from elsewhere, else empty; feeds ``state_grad_summaries``.
        reverse_sources: ``int64 [slots]`` local index of each fragment's first subsequence, whose
            exit-state cotangent is folded into the reverse slot's bias.
        predecessors: ``int64 [slots, chain]`` flat ``rank * slots + fragment`` indices of each
            fragment's predecessor slots in token order, padded with ``world * slots`` (identity).
        successors: Same for each fragment's successors, in reverse token order.
        entry_sources: ``int64 [max_subsequences]`` local fragment whose folded predecessors are
            a segment's entry state, or ``slots`` for a segment that starts its sequence.
        exit_sources: Same for exit cotangents: the fragment whose folded successors apply, or
            ``slots`` for a segment that ends its sequence (or is padding).
        terminal: ``bool [max_subsequences]`` marks subsequences that end their sequence.
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


@dataclass(frozen=True)
class ContextParallelPlan:
    """Host-static routing for one rank derived from every rank's fragments.

    Attributes:
        table: ``table[cp_rank][slot]`` is that rank's fragment behind gather slot ``slot``.
        cp_rank: This rank's row in ``table``: its rank within the context-parallel group.
        fragments: This rank's fragments in span order.
        subsequences: This rank's subsequences in span order.
        cu_seqlens: Span offsets, one segment per subsequence.
        slots: Gather slots per rank, ``max(len(row) for row in table)``: one per fragment.
        predecessors: Per local subsequence, the slots of the same sequence's earlier pieces in
            increasing token order; nonempty only for a fragment's first subsequence.
        successors: Per local subsequence, the slots of the same sequence's later pieces in
            decreasing token order, i.e. the order a reverse fold visits them; nonempty only for
            a fragment's last subsequence.
        terminal: Local indices of subsequences that end their sequence, whose exit states are
            the sequence's true final states.
    """

    table: tuple[tuple[Fragment, ...], ...]
    cp_rank: int
    fragments: tuple[Fragment, ...]
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
        table = tuple(
            tuple(
                Fragment(start, stop, _cut_at_sequences(cu_seqlens_global, (start, stop)))
                for start, stop in owned
            )
            for owned in fragments
        )
        # Every fragment holds at most one piece of a given sequence, so a chain is a list of
        # ``(rank, fragment)`` slots in token order.
        chains: dict[int, list[tuple[int, Slot]]] = {}
        for owner, row in enumerate(table):
            for index, fragment in enumerate(row):
                for piece in fragment.subsequences:
                    chains.setdefault(piece.sequence, []).append((piece.start, (owner, index)))
        for chain in chains.values():
            chain.sort()

        local = table[cp_rank]
        subsequences = tuple(piece for fragment in local for piece in fragment.subsequences)
        predecessors: list[tuple[Slot, ...]] = []
        successors: list[tuple[Slot, ...]] = []
        terminal: list[int] = []
        index = 0
        for slot, fragment in enumerate(local):
            for position, piece in enumerate(fragment.subsequences):
                chain = [address for _, address in chains[piece.sequence]]
                here = chain.index((cp_rank, slot))
                first = position == 0
                last = position == len(fragment.subsequences) - 1
                predecessors.append(tuple(chain[:here]) if first else ())
                successors.append(tuple(reversed(chain[here + 1 :])) if last else ())
                if here == len(chain) - 1:
                    terminal.append(index)
                index += 1
        return cls(
            table=table,
            cp_rank=cp_rank,
            fragments=local,
            subsequences=subsequences,
            cu_seqlens=tuple(accumulate((s.length for s in subsequences), initial=0)),
            slots=max(len(row) for row in table),
            predecessors=tuple(predecessors),
            successors=tuple(successors),
            terminal=tuple(terminal),
        )

    def global_token_ids(self, device: torch.device | str) -> torch.Tensor:
        """Global token id of every span token: the bridge from global tensors to the span."""
        return torch.cat([torch.arange(s.start, s.stop, device=device) for s in self.subsequences])

    def fragment_spans(self) -> list[tuple[int, int]]:
        """``(first, last)`` local subsequence index of each fragment, in slot order."""
        spans, index = [], 0
        for fragment in self.fragments:
            spans.append((index, index + len(fragment.subsequences) - 1))
            index += len(fragment.subsequences)
        return spans

    def routing(
        self, device: torch.device | str, *, max_subsequences: int | None = None
    ) -> ContextParallelRouting:
        """Materialize this layout as fixed-shape device tensors for ``context_parallel_chunk``.

        ``max_subsequences`` caps the span's segment count so every layout of the same fragment
        table yields identically shaped tensors (default: this layout's own count, no padding).
        The chain length is one less than the number of fragments in the table. One small
        host-to-device copy.
        """
        count = len(self.subsequences)
        if max_subsequences is None:
            max_subsequences = count
        if count > max_subsequences:
            raise ValueError(
                f"layout has {count} subsequences but max_subsequences is {max_subsequences}"
            )
        world = len(self.table)
        chain = max(sum(len(row) for row in self.table) - 1, 0)
        identity = world * self.slots
        padding = max_subsequences - count

        def flat(addresses: Sequence[Slot]) -> list[int]:
            indices = [owner * self.slots + slot for owner, slot in addresses]
            return indices + [identity] * (chain - len(indices))

        forward_bounds = [[0, 0]] * self.slots
        reverse_bounds = [[0, 0]] * self.slots
        reverse_sources = [0] * self.slots
        predecessors = [flat(())] * self.slots
        successors = [flat(())] * self.slots
        entry_sources = [self.slots] * max_subsequences
        exit_sources = [self.slots] * max_subsequences
        for slot, (first, last) in enumerate(self.fragment_spans()):
            if self.predecessors[first]:
                reverse_bounds[slot] = [self.cu_seqlens[first], self.cu_seqlens[first + 1]]
                reverse_sources[slot] = first
                predecessors[slot] = flat(self.predecessors[first])
                entry_sources[first] = slot
            if self.successors[last]:
                forward_bounds[slot] = [self.cu_seqlens[last], self.cu_seqlens[last + 1]]
                successors[slot] = flat(self.successors[last])
                exit_sources[last] = slot
        terminal = [index in self.terminal for index in range(max_subsequences)]

        def tensor(values, dtype):
            return torch.tensor(values, dtype=dtype, device=device)

        return ContextParallelRouting(
            cp_rank=self.cp_rank,
            world=world,
            slots=self.slots,
            tokens=self.cu_seqlens[-1],
            cu_seqlens=tensor([*self.cu_seqlens] + [self.cu_seqlens[-1]] * padding, torch.int32),
            forward_bounds=tensor(forward_bounds, torch.int32),
            reverse_bounds=tensor(reverse_bounds, torch.int32),
            reverse_sources=tensor(reverse_sources, torch.int64),
            predecessors=tensor(predecessors, torch.int64),
            successors=tensor(successors, torch.int64),
            entry_sources=tensor(entry_sources, torch.int64),
            exit_sources=tensor(exit_sources, torch.int64),
            terminal=tensor(terminal, torch.bool),
        )


def _check_group(
    group: dist.ProcessGroup, *, world: int, cp_rank: int, owned: int, tokens: int
) -> None:
    """Reject a plan built for a different group, rank, or local token count (host-only)."""
    if dist.get_world_size(group) != world:
        raise ValueError(f"plan has {world} ranks but the group has {dist.get_world_size(group)}")
    if dist.get_rank(group) != cp_rank:
        raise ValueError(
            f"plan was built for cp_rank {cp_rank}, running on {dist.get_rank(group)}"
        )
    if tokens != owned:
        raise ValueError(f"plan owns {owned} local tokens but the input has {tokens}")


def _check_routing(routing: ContextParallelRouting, device: torch.device) -> None:
    """Reject routing tensors whose shapes, dtypes, or device disagree (host-only, no sync)."""
    segments = routing.terminal.shape[0]
    chain = routing.predecessors.shape[-1]
    expected = {
        "cu_seqlens": ((segments + 1,), torch.int32),
        "forward_bounds": ((routing.slots, 2), torch.int32),
        "reverse_bounds": ((routing.slots, 2), torch.int32),
        "reverse_sources": ((routing.slots,), torch.int64),
        "predecessors": ((routing.slots, chain), torch.int64),
        "successors": ((routing.slots, chain), torch.int64),
        "entry_sources": ((segments,), torch.int64),
        "exit_sources": ((segments,), torch.int64),
        "terminal": ((segments,), torch.bool),
    }
    for name, (shape, dtype) in expected.items():
        tensor = getattr(routing, name)
        if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != device:
            raise ValueError(
                f"routing.{name} must be {dtype} {shape} on {device}, got {tensor.dtype} "
                f"{tuple(tensor.shape)} on {tensor.device}"
            )


def _all_gather_slots(local_slots: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Gather every rank's ``[slots, ...]`` buffer into ``[world, slots, ...]``."""
    gathered = local_slots.new_empty(dist.get_world_size(group), *local_slots.shape)
    with profiler_range("cp/all_gather"):
        dist.all_gather_single(gathered, local_slots.contiguous(), group=group)
    return gathered


def fold_slot_chains(gathered: torch.Tensor, chains: torch.Tensor) -> torch.Tensor:
    """Fold device-indexed slot chains from the zero state.

    ``gathered`` is ``[world, slots, HV, V + K, K]`` and ``chains`` is ``int64 [rows, L]`` of
    flat ``rank * slots + slot`` indices, padded with ``world * slots``, which addresses the
    identity appended here. Every row costs ``L`` merges regardless of its real chain length, so
    the launch sequence is fixed and CUDA Graph capturable.
    """
    heads, packed, key_dim = gathered.shape[-3:]
    neutral = neutral_summary(heads, packed - key_dim, key_dim, device=gathered.device)
    flat = torch.cat((gathered.flatten(0, 1), neutral.unsqueeze(0)))
    state = gathered.new_zeros(chains.shape[0], heads, packed - key_dim, key_dim)
    for step in range(chains.shape[1]):
        state = merge_state(state, flat[chains[:, step]])
    return state


def _scatter_folds(folded: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """Map ``[slots, ...]`` folded states onto span segments; source ``slots`` selects zero."""
    return torch.cat((folded, folded.new_zeros(1, *folded.shape[1:])))[sources]


def compose_entry_states(gathered: torch.Tensor, routing: ContextParallelRouting) -> torch.Tensor:
    """Fold each fragment's predecessor summaries from zero and place them on its first segment.

    ``gathered`` is ``[world, slots, HV, V + K, K]``; the result is
    ``[max_subsequences, HV, V, K]``, zero for every segment that starts its sequence.
    """
    return _scatter_folds(fold_slot_chains(gathered, routing.predecessors), routing.entry_sources)


def compose_exit_cotangents(
    gathered: torch.Tensor,
    d_final_state: torch.Tensor | None,
    routing: ContextParallelRouting,
) -> torch.Tensor:
    """Fold each fragment's successor reverse maps onto its last segment's exit cotangent."""
    incoming = _scatter_folds(fold_slot_chains(gathered, routing.successors), routing.exit_sources)
    return incoming if d_final_state is None else incoming + d_final_state


def compose_conv_histories(
    gathered_tails: torch.Tensor, plan: ContextParallelPlan
) -> torch.Tensor:
    """Build each local subsequence's ``W - 1`` token history from predecessor tails.

    ``gathered_tails`` is ``[world, slots, W - 1, C]`` where each slot holds the last ``W - 1``
    tokens of that fragment's last subsequence, front-padded with zeros when it is shorter.
    Concatenating predecessor tails in token order and keeping the last valid tokens yields
    exactly the tokens preceding the subsequence, however short the predecessors are.
    """
    history_length = gathered_tails.shape[-2]
    histories = []
    for sources in plan.predecessors:
        pieces = [
            gathered_tails[
                owner,
                slot,
                -min(history_length, plan.table[owner][slot].subsequences[-1].length) :,
            ]
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
    """Exchange fragment tails and return ``causal_conv1d``'s packed ``initial_state``.

    ``qkv`` is the rank's ``[1, T, C]`` span. The collective is differentiable, so
    gradients flow back to the ranks that own the history tokens.
    """
    if history_length == 0:
        return None
    _check_group(
        group,
        world=len(plan.table),
        cp_rank=plan.cp_rank,
        owned=plan.cu_seqlens[-1],
        tokens=qkv.shape[1],
    )
    tails = []
    for fragment, (_, last) in zip(plan.fragments, plan.fragment_spans(), strict=True):
        stop = plan.cu_seqlens[last + 1]
        stored = min(history_length, fragment.subsequences[-1].length)
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

    def state_summaries(self, bounds: torch.Tensor) -> torch.Tensor: ...

    def run(
        self, initial_state: torch.Tensor | None, *, output_final_state: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class PreparedBackward(Protocol):
    """Backward handle of a staged delta-rule op."""

    def state_grad_summaries(self, bounds: torch.Tensor) -> torch.Tensor: ...

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


def summary_slots(prepared: PreparedForward, routing: ContextParallelRouting) -> torch.Tensor:
    """This rank's ``[slots, HV, V + K, K]`` forward slots: each fragment's last subsequence.

    Fragments whose last subsequence ends its sequence have an empty range and send the identity.
    """
    with profiler_range("cp/state_summary"):
        return prepared.state_summaries(routing.forward_bounds)


def grad_summary_slots(
    grads: PreparedBackward,
    d_final_state: torch.Tensor | None,
    routing: ContextParallelRouting,
) -> torch.Tensor:
    """This rank's reverse slots: each fragment's first subsequence, with the loss folded in.

    Upstream ranks need the total cotangent leaving that subsequence, so the bias sent is
    ``d_final_state @ R + C`` rather than the summary's zero-exit bias ``C``. A fragment that
    starts its sequence sends a slot no chain addresses, so its bias is irrelevant.
    """
    with profiler_range("cp/state_grad_summary"):
        slots = grads.state_grad_summaries(routing.reverse_bounds)
    if d_final_state is None:
        return slots
    value_dim = d_final_state.shape[-2]
    direct = d_final_state[routing.reverse_sources]
    return torch.cat((merge_state(direct, slots), slots[..., value_dim:, :]), dim=-2)


class _ContextParallelChunk(torch.autograd.Function):
    """Explicit-adjoint delta rule: summaries and one all-gather in each direction.

    Every index the forward and backward use comes from the :class:`ContextParallelRouting`
    tensors, so the captured launch sequence is the same for every layout of one fragment table.
    The backward's routing tensors are saved with the activations, so mutating them between
    forward and backward is caught by autograd's version check.
    """

    @staticmethod
    def forward(ctx, q, k, v, gate, beta, routing, group, stages):
        prepared = stages.prepare(q, k, v, gate, beta, cu_seqlens=routing.cu_seqlens)
        gathered = _all_gather_slots(summary_slots(prepared, routing), group)
        initial_state = compose_entry_states(gathered, routing)
        with profiler_range("cp/run"):
            output, final_state = prepared.run(initial_state, output_final_state=True)
        assert final_state is not None

        saved = prepared.saved
        ctx.save_for_backward(
            *saved,
            initial_state,
            routing.reverse_bounds,
            routing.reverse_sources,
            routing.successors,
            routing.exit_sources,
        )
        ctx.saved_type = type(saved)
        ctx.routing = routing
        ctx.group = group
        ctx.stages = stages
        ctx.set_materialize_grads(False)
        return output, final_state

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output, d_final_state):
        *saved, initial_state, reverse_bounds, reverse_sources, successors, exit_sources = (
            ctx.saved_tensors
        )
        routing = dataclasses.replace(
            ctx.routing,
            reverse_bounds=reverse_bounds,
            reverse_sources=reverse_sources,
            successors=successors,
            exit_sources=exit_sources,
        )
        grads = ctx.stages.prepare_backward(ctx.saved_type._make(saved), d_output, initial_state)
        gathered = _all_gather_slots(grad_summary_slots(grads, d_final_state, routing), ctx.group)
        exit_cotangent = compose_exit_cotangents(gathered, d_final_state, routing)
        with profiler_range("cp/run"):
            dq, dk, dv, dgate, dbeta, _d_initial_state = grads.run(exit_cotangent)
        return dq, dk, dv, dgate, dbeta, None, None, None


def context_parallel_chunk(
    stages: StagedOp,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    routing: ContextParallelRouting,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a staged delta-rule op over this rank's span with state exchanged by all-gather.

    Args:
        stages: The op's entry points; ``attn_gym.linear.kda.context_parallel_kda`` and
            ``attn_gym.linear.gdn.context_parallel_gdn`` bind them with the op's options.
        q: This rank's span of queries, laid out as ``plan.subsequences``; ``k``, ``v``,
            ``gate``, and ``beta`` follow the same layout and the op's ``chunk_*`` contract.
        routing: This layout's tensors from ``ContextParallelPlan.routing``.
        group: Process group containing exactly the plan's ranks, in order.

    Returns:
        The span's output and one FP32 ``[max_subsequences, HV, V, K]`` exit state per span
        segment. Only the entries where ``routing.terminal`` is set are the sequences' true final
        states; the others are intermediate states that the owner of the next subsequence
        continued from, or unused rows of padded segments.

    ``initial_state`` is not accepted: every sequence starts from zero and the plan supplies the
    entry states. Every rank in ``group`` must call this function the same number of times. A
    CUDA Graph captured around a call replays for any layout whose routing has the same shapes:
    copy the new layout's tensors into the captured ones.
    """
    _check_group(
        group,
        world=routing.world,
        cp_rank=routing.cp_rank,
        owned=routing.tokens,
        tokens=q.shape[1],
    )
    _check_routing(routing, q.device)
    return _ContextParallelChunk.apply(q, k, v, gate, beta, routing, group, stages)


__all__ = [
    "ContextParallelPlan",
    "ContextParallelRouting",
    "Fragment",
    "PreparedBackward",
    "PreparedForward",
    "StagedOp",
    "Subsequence",
    "compose_conv_histories",
    "compose_entry_states",
    "compose_exit_cotangents",
    "context_parallel_chunk",
    "context_parallel_conv_history",
    "fold_slot_chains",
    "grad_summary_slots",
    "summary_slots",
]
