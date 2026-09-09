# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallelism whose numerics do not depend on the CP degree.

NOTE [Canonical Tiles]
The standard recipe (``context_parallel_chunk``) summarizes each rank's fragment with one affine
map, so changing the number of ranks changes which token ranges are summarized and how the FP32
maps compose, and the rounded result moves. This module fixes the whole arithmetic graph before
ranks are assigned:

    tile      ``[start, min(start + tile_size, doc_end))`` within one document, aligned to the
              document's first token; a document's last tile may be short. ``tile_size`` is a
              multiple of the 64-token summary chunk and part of the numerical contract.
    leaf map  each tile's ``[bias; transition]`` forward map and ``[C; R]`` reverse map, computed
              by the staged op on that tile alone (no other tile's tokens are visible).
    scan      per document, a serial fold of the leaf maps in tile order (forward) or reverse
              tile order (backward), in three-pass TF32 (fp32-accurate), fused in one launch. The
              state entering tile ``j`` depends on that document's leaves ``0..j-1`` alone.
    entry     ``merge_state(0, prefix[i - 1])`` for tile ``i``; a document's first tile enters
              with zero. Exits are the reverse analogue.

Ranks own whole tiles; ownership changes which leaves a rank computes and where the scan runs,
never the leaf arithmetic or the tree. CP=1 is therefore bitwise identical to CP=N (verified for
the fused and Mega forwards in ``test/test_context_parallel_deterministic.py``), and a document's
result does not depend on the documents packed around it. It is a distinct numerical contract
from the unsharded ``chunk_kda`` call.

A rank prepares all its tiles in one staged call, each tile a packed subsequence, with the
summary kernels' work partition pinned (``deterministic_work=True``); this is bitwise equal to
preparing every tile alone (``test/test_canonical_tile_batching.py``).

NOTE [Single-Tile Documents]
A document that fits in one tile has no state to carry: its tile enters with zero and nothing
reads its map, so it is excluded from the summaries, the exchange, and the scan
(``CanonicalTiling.summarized``). The exclusion depends on the tiling only, so the contract is
unchanged, and on packings with many short documents it removes most of the recipe's overhead.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, groupby, pairwise
from typing import NamedTuple

import torch
import torch.distributed as dist

from attn_gym._backends.profiler import profiler_range
from attn_gym.linear._delta_rule.triton.canonical_scan import canonical_scan_entries
from attn_gym.linear.context_parallel import StagedOp

SUMMARY_CHUNK = 64


class Tile(NamedTuple):
    """One canonical tile: global token range ``[start, stop)`` of ``document``."""

    document: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class CanonicalTiling:
    """Fixed tiles of a packed stream plus this rank's ownership.

    Attributes:
        tiles: Every tile of the stream in global token order.
        owned: ``owned[rank]`` lists the tile ids that rank computes, in its span order.
        cp_rank: This rank.
    """

    tiles: tuple[Tile, ...]
    owned: tuple[tuple[int, ...], ...]
    cp_rank: int

    @classmethod
    def from_fragments(
        cls,
        cu_seqlens_global: Sequence[int],
        fragments: Sequence[Sequence[tuple[int, int]]],
        cp_rank: int,
        *,
        tile_size: int,
    ) -> CanonicalTiling:
        """Tile the stream and assign whole tiles to the ranks that own their tokens.

        ``fragments[rank]`` are that rank's global token ranges. Every fragment boundary must fall
        on a tile boundary (a document boundary or ``doc_start + k * tile_size``), otherwise the
        ownership would cut a tile and the arithmetic would depend on the fragment table.
        """
        if tile_size <= 0 or tile_size % SUMMARY_CHUNK:
            raise ValueError(f"tile_size must be a positive multiple of {SUMMARY_CHUNK}")
        if not 0 <= cp_rank < len(fragments):
            raise ValueError(f"cp_rank {cp_rank} is outside a table of {len(fragments)} ranks")
        tiles = tile_stream(cu_seqlens_global, tile_size)
        # Tile boundaries -> id of the tile starting there; the stream's end closes the last one.
        boundary = {tile.start: index for index, tile in enumerate(tiles)}
        boundary[cu_seqlens_global[-1]] = len(tiles)
        owned: list[tuple[int, ...]] = []
        for rank_fragments in fragments:
            ids: list[int] = []
            for start, stop in rank_fragments:
                if start >= stop:
                    raise ValueError(f"fragment [{start}, {stop}) is empty or reversed")
                if start not in boundary or stop not in boundary:
                    raise ValueError(
                        f"fragment [{start}, {stop}) does not fall on canonical tile boundaries"
                    )
                ids.extend(range(boundary[start], boundary[stop]))
            owned.append(tuple(ids))
        if sorted(index for ids in owned for index in ids) != list(range(len(tiles))):
            raise ValueError("fragments must cover every tile exactly once")
        return cls(tuple(tiles), tuple(owned), cp_rank=cp_rank)

    @property
    def mine(self) -> tuple[int, ...]:
        return self.owned[self.cp_rank]

    @property
    def summarized(self) -> tuple[int, ...]:
        """Tile ids whose maps take part in the scan: those of multi-tile documents."""
        counts = Counter(tile.document for tile in self.tiles)
        return tuple(i for i, tile in enumerate(self.tiles) if counts[tile.document] > 1)

    @property
    def summarized_by_rank(self) -> tuple[tuple[int, ...], ...]:
        """``summarized_by_rank[rank]``: that rank's indices into ``summarized``, in span order."""
        position = {i: s for s, i in enumerate(self.summarized)}
        return tuple(tuple(position[i] for i in ids if i in position) for ids in self.owned)

    @property
    def slots(self) -> int:
        """Gather width: the most summarized tiles any rank owns."""
        return max(map(len, self.summarized_by_rank))

    def global_token_ids(self, device: torch.device | str) -> torch.Tensor:
        """Global token id of every span token of this rank (tiles concatenated in span order)."""
        pieces = [
            torch.arange(self.tiles[i].start, self.tiles[i].stop, device=device) for i in self.mine
        ]
        return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64, device=device)

    def span_offsets(self) -> tuple[int, ...]:
        """Span-local ``[start, stop)`` of each owned tile, in span order."""
        return tuple(accumulate((self.tiles[i].length for i in self.mine), initial=0))

    def routing(self, device: torch.device | str) -> CanonicalRouting:
        """Materialize the index tensors the recipe reads, once per layout and device."""
        offsets = self.span_offsets()
        summarized = self.summarized
        in_scan = set(summarized)
        positions = [p for p, i in enumerate(self.mine) if i in in_scan]
        by_rank, width = self.summarized_by_rank, self.slots
        # Row of each summarized tile in the all-gather buffer: rank-major, ``width`` per rank.
        row = {s: rank * width + r for rank, ids in enumerate(by_rank) for r, s in enumerate(ids)}
        source = [row[s] for s in range(len(summarized))]
        documents = [self.tiles[i].document for i in summarized]

        def int32(values: Sequence[int] | Sequence[tuple[int, int]]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        return CanonicalRouting(
            cu_seqlens=int32(offsets),
            summary_bounds=int32([offsets[p : p + 2] for p in positions]).reshape(-1, 2),
            summarized_positions=(
                None
                if len(positions) == len(self.mine)
                else torch.tensor(positions, dtype=torch.int64, device=device)
            ),
            document_offsets=int32(
                [0, *accumulate(len(list(run)) for _, run in groupby(documents))]
            ),
            mine=_selection(by_rank[self.cp_rank], device),
            gather_source=(
                None
                if source == list(range(len(summarized)))
                else torch.tensor(source, dtype=torch.int64, device=device)
            ),
        )


class CanonicalRouting(NamedTuple):
    """Device tensors of a :class:`CanonicalTiling` for one rank, built once per layout.

    ``cu_seqlens`` cuts the rank's span into one segment per owned tile; ``summary_bounds`` are
    the segments whose tiles take part in the scan and ``summarized_positions`` their positions
    among the owned tiles (``None`` when all do). ``document_offsets`` groups the gathered
    summarized-tile maps by document. ``mine`` selects this rank's summarized tiles in the
    gathered order, as a slice when consecutive so no gather copy is made; ``gather_source`` is
    each summarized tile's row in the all-gather buffer (``None`` when the buffer already is the
    summarized order).
    """

    cu_seqlens: torch.Tensor
    summary_bounds: torch.Tensor
    summarized_positions: torch.Tensor | None
    document_offsets: torch.Tensor
    mine: torch.Tensor | slice
    gather_source: torch.Tensor | None


def _selection(ids: Sequence[int], device: torch.device | str) -> torch.Tensor | slice:
    """A slice when ``ids`` are consecutive (no gather copy), else an index tensor."""
    if ids and list(ids) == list(range(ids[0], ids[0] + len(ids))):
        return slice(ids[0], ids[0] + len(ids))
    return torch.tensor(list(ids), dtype=torch.int64, device=device)


def tile_stream(cu_seqlens_global: Sequence[int], tile_size: int) -> list[Tile]:
    """Cut every document into ``tile_size`` tiles from its first token; the last may be short."""
    return [
        Tile(document, begin, min(begin + tile_size, stop))
        for document, (start, stop) in enumerate(pairwise(cu_seqlens_global))
        for begin in range(start, stop, tile_size)
    ]


# Module-level name so tests without a second GPU can substitute a CPU-staged collective.
_all_gather_into_tensor = dist.all_gather_into_tensor


def all_gather_tile_maps(
    local: torch.Tensor,
    tiling: CanonicalTiling,
    routing: CanonicalRouting,
    like: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Exchange every rank's summarized tile maps; ``[tiles, H, V + K, K]`` in summarized order.

    ``local`` holds this rank's summarized tile maps in its span order (zero rows for a rank
    without any). Packets are zero-padded to ``tiling.slots`` rows so the collective is a plain
    equal-size all-gather; padding rows never enter a scan. When the gathered buffer already is
    the summarized order it is returned without a copy. ``group=None`` (a single rank owning
    everything, the CP=1 reference) skips the collective.
    """
    width, world = tiling.slots, len(tiling.owned)
    if group is None:
        assert world == 1 and local.shape[0] == len(tiling.summarized)
        return local
    if local.shape[0] == width:
        packet = local
    else:
        packet = like.new_zeros((width, *like.shape[1:]))
        packet[: local.shape[0]] = local
    gathered = like.new_empty((world * width, *like.shape[1:]))
    with profiler_range("cp/all_gather_tiles"):
        _all_gather_into_tensor(gathered, packet, group=group)
    if routing.gather_source is None:
        return gathered[: len(tiling.summarized)]
    return gathered[routing.gather_source]


def canonical_entries(
    summary_maps: torch.Tensor,
    tiling: CanonicalTiling,
    routing: CanonicalRouting,
    like: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    reverse: bool,
) -> torch.Tensor:
    """Entry (or exit when ``reverse``) FP32 states of this rank's tiles, in ``tiling.mine`` order.

    ``summary_maps`` are the maps of the rank's summarized tiles (NOTE [Single-Tile Documents])
    in span order; a rank with none (possibly no tiles at all) passes zero rows and still joins
    the collective. Tiles of single-tile documents get the zero state.

    Every summarized tile's map is all-gathered and scanned per document: the serial fold in
    three-pass TF32, so the result is a function of the tiles' maps and indices only.
    """
    if summary_maps.shape[0] == 0:
        all_gather_tile_maps(summary_maps, tiling, routing, like, group)
        return like.new_zeros(
            (len(tiling.mine), like.shape[1], like.shape[2] - like.shape[3], like.shape[3])
        )
    gathered = all_gather_tile_maps(summary_maps, tiling, routing, like, group)
    entries = canonical_scan_entries(gathered, routing.document_offsets, reverse=reverse)
    entries = entries[routing.mine]
    if routing.summarized_positions is None:
        return entries
    full = entries.new_zeros((len(tiling.mine), *entries.shape[1:]))
    full[routing.summarized_positions] = entries
    return full


class _CanonicalContextParallel(torch.autograd.Function):
    """One batched staged forward/backward per rank glued by the fixed scan tree.

    Every owned tile is one packed subsequence of the rank's span (``cu_seqlens`` at tile
    boundaries), so the leaf kernels see each tile exactly as an independent call would, and the
    summary kernels run with ``deterministic_work=True`` so their work partition cannot depend on
    how many tiles the rank owns. A rank without tiles skips the kernels but still joins both
    collectives.
    """

    @staticmethod
    def forward(ctx, q, k, v, gate, beta, tiling, routing, group, stages: StagedOp):
        heads, dim, value_dim = v.shape[2], q.shape[3], v.shape[3]  # maps are per value head
        # Zero rows of the packed map shape: the collectives' shape anchor, and the summaries
        # of a rank with nothing to summarize.
        like = q.new_empty((0, heads, value_dim + dim, dim), dtype=torch.float32)
        maps = like
        if tiling.mine:
            prepared = stages.prepare(q, k, v, gate, beta, cu_seqlens=routing.cu_seqlens)
            if routing.summary_bounds.shape[0]:
                maps = prepared.state_summaries(routing.summary_bounds, deterministic_work=True)
        entry = canonical_entries(maps, tiling, routing, like, group, reverse=False)
        if tiling.mine:
            with profiler_range("cp/run"):
                output, _ = prepared.run(entry, output_final_state=False)
            ctx.save_for_backward(*prepared.saved, entry, like)
            ctx.saved_type = type(prepared.saved)
            ctx.scale = prepared.scale
        else:
            output = v[:, :0]
            ctx.save_for_backward(entry, like)
            ctx.input_meta = tuple((t.shape, t.dtype) for t in (q, k, v, gate, beta))
        ctx.tiling = tiling
        ctx.routing = routing
        ctx.group = group
        ctx.stages = stages
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output):
        tiling: CanonicalTiling = ctx.tiling
        routing: CanonicalRouting = ctx.routing
        *saved, entry, like = ctx.saved_tensors
        maps = like
        if tiling.mine:
            backward = ctx.stages.prepare_backward(
                ctx.saved_type._make(saved), d_output, entry, scale=ctx.scale
            )
            if routing.summary_bounds.shape[0]:
                maps = backward.state_grad_summaries(
                    routing.summary_bounds, deterministic_work=True
                )
        exit_cotangent = canonical_entries(maps, tiling, routing, like, ctx.group, reverse=True)
        if tiling.mine:
            with profiler_range("cp/run"):
                grads = backward.run(exit_cotangent)[:5]
        else:
            grads = tuple(like.new_empty(shape, dtype=dtype) for shape, dtype in ctx.input_meta)
        return (*grads, None, None, None, None)


def context_parallel_chunk_deterministic(
    stages: StagedOp,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    tiling: CanonicalTiling,
    group: dist.ProcessGroup,
    routing: CanonicalRouting | None = None,
) -> torch.Tensor:
    """Run a staged delta-rule op over this rank's canonical tiles (NOTE [Canonical Tiles]).

    Args:
        stages: The op's staged entry points, as for ``context_parallel_chunk``.
        q: This rank's span: its owned tiles concatenated in ``tiling.mine`` order; ``k``, ``v``,
            ``gate``, ``beta`` follow the same layout.
        tiling: The stream's canonical tiling and this rank's ownership.
        group: Process group containing exactly the tiling's ranks, in order.
        routing: ``tiling.routing(device)``; pass it to reuse the index tensors across calls
            with the same layout (built here otherwise).

    Returns:
        The span's output. Final states are not returned: every document's exit state is a
        function of the scan and can be recovered from the gathered maps if needed.

    Every rank in ``group`` must call this, and run its backward, the same number of times,
    including ranks that own no tiles: both directions all-gather maps, so an empty rank still
    contributes an all-zero packet (and gets an empty output and empty gradients).

    Memory: every rank holds the gathered maps of all summarized tiles, ``H x (V + K) x K``
    FP32 each (128 KiB per head for K = V = 128), plus one entry state per owned tile; the tile
    size sets that footprint (NOTE [Single-Tile Documents]).
    """
    world, cp_rank = dist.get_world_size(group), dist.get_rank(group)
    if world != len(tiling.owned) or cp_rank != tiling.cp_rank:
        raise ValueError("tiling was built for a different group or rank")
    expected = sum(tiling.tiles[i].length for i in tiling.mine)
    if q.shape[1] != expected:
        raise ValueError(f"tiling owns {expected} local tokens but the input has {q.shape[1]}")
    if routing is None:
        routing = tiling.routing(q.device)
    return _CanonicalContextParallel.apply(q, k, v, gate, beta, tiling, routing, group, stages)


__all__ = [
    "SUMMARY_CHUNK",
    "CanonicalRouting",
    "CanonicalTiling",
    "Tile",
    "all_gather_tile_maps",
    "canonical_entries",
    "context_parallel_chunk_deterministic",
    "tile_stream",
]
