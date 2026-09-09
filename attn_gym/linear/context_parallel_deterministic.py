# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallelism whose numerics do not depend on the CP degree.

NOTE [Canonical Tiles]
The standard recipe (``context_parallel_chunk``) summarizes each rank's fragment with one affine
map. Changing the number of ranks changes the fragments, which changes which token ranges are
summarized and how the FP32 maps are composed, so the rounded result moves. This module fixes
the whole arithmetic graph before ranks are assigned:

    tile      ``[start, min(start + tile_size, doc_end))`` within one document, aligned to the
              document's first token; the last tile of a document may be short. Tile size is a
              multiple of the 64-token summary chunk and is part of the numerical contract.
    leaf map  every tile's ``[bias; transition]`` forward map and ``[C; R]`` reverse map, computed
              by the staged op on that tile alone (no other tile's tokens are visible).
    scan      per document, a left-to-right serial fold of the leaf maps in tile order
              (forward) and reverse tile order (backward), in three-pass TF32 (fp32-accurate),
              fused in one launch. The state entering tile ``j`` is a function of that
              document's leaves ``0..j-1`` alone.
    entry     ``merge_state(0, prefix[i - 1])`` for tile ``i``; the first tile of a document enters
              with zero. Exits are the reverse analogue.

NOTE [Scan Blocks]
Exchanging every tile's map costs 128 KiB per tile per head per rank. Consecutive tiles of a
document are therefore grouped into blocks of ``scan_block`` tiles (document-relative, the last
block of a document may be short); only block maps cross ranks. A block map is the serial fold
of its tiles' maps, the per-document scan runs over block maps, and each rank folds its own
tiles from the block entry state. Every operation is a function of tile indices only, so the
tree is still fixed; it is a different tree from the flat fold (``scan_block=1``), hence a
different numerical contract, and ``scan_block`` is part of the recipe. Ranks must own whole
blocks (a block split across ranks would need its tiles exchanged); block-aligned fragments make
the exchange ``scan_block`` times smaller with identical bits. The block folds cost about as much
as a full scan, so blocks pay off only when the gather dominates (many ranks, slow interconnect,
many tiles per rank); on two NVLink GPUs ``scan_block=1`` is faster and is the default.

Ranks own whole tiles; ownership changes which leaves a rank computes and where the scan runs,
never the leaf arithmetic or the tree. The result for CP=1 with this recipe is therefore bitwise
identical to CP=N (verified for fused and Mega forwards, see the deterministic tests), and a
document's result does not depend on which other documents are packed around it. It is a
distinct numerical contract from the unsharded ``chunk_kda`` call.

A rank prepares all its tiles in one staged call, each tile a packed subsequence, with the
summary kernels' work partition pinned (``deterministic_work=True``); this is bitwise equal to
preparing every tile alone (``test/test_canonical_tile_batching.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise
from typing import NamedTuple

import torch
import torch.distributed as dist

from attn_gym._backends.profiler import profiler_range
from attn_gym.linear._delta_rule.triton.canonical_scan import canonical_scan_entries, fold_ranges
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
        scan_block: Tiles per exchanged block (NOTE [Scan Blocks]); 1 exchanges every tile.
    """

    tiles: tuple[Tile, ...]
    owned: tuple[tuple[int, ...], ...]
    cp_rank: int
    scan_block: int = 1

    @classmethod
    def from_fragments(
        cls,
        cu_seqlens_global: Sequence[int],
        fragments: Sequence[Sequence[tuple[int, int]]],
        cp_rank: int,
        *,
        tile_size: int,
        scan_block: int = 1,
    ) -> CanonicalTiling:
        """Tile the stream and assign whole tiles to the ranks that own their tokens.

        ``fragments[rank]`` are that rank's global token ranges. Every fragment boundary must fall
        on a tile boundary (a document boundary or ``doc_start + k * tile_size``), otherwise the
        ownership would cut a tile and the leaf arithmetic would depend on the fragment table.
        With ``scan_block > 1`` the boundaries must fall on block boundaries
        (``doc_start + k * scan_block * tile_size``) for the same reason (NOTE [Scan Blocks]).
        """
        if tile_size <= 0 or tile_size % SUMMARY_CHUNK:
            raise ValueError(f"tile_size must be a positive multiple of {SUMMARY_CHUNK}")
        if scan_block <= 0:
            raise ValueError("scan_block must be positive")
        if not 0 <= cp_rank < len(fragments):
            raise ValueError(f"cp_rank {cp_rank} is outside a table of {len(fragments)} ranks")
        tiles = tile_stream(cu_seqlens_global, tile_size)
        starts = {tile.start: index for index, tile in enumerate(tiles)}
        boundaries = set(starts) | {cu_seqlens_global[-1]}
        owned: list[list[int]] = []
        for rank_fragments in fragments:
            ids: list[int] = []
            for start, stop in rank_fragments:
                if start >= stop:
                    raise ValueError(f"fragment [{start}, {stop}) is empty or reversed")
                if start not in boundaries or stop not in boundaries:
                    raise ValueError(
                        f"fragment [{start}, {stop}) does not fall on canonical tile boundaries"
                    )
                index = starts.get(start)
                while index is not None and index < len(tiles) and tiles[index].stop <= stop:
                    ids.append(index)
                    index += 1
            owned.append(ids)
        seen = sorted(index for ids in owned for index in ids)
        if seen != list(range(len(tiles))):
            raise ValueError("fragments must cover every tile exactly once")
        tiling = cls(
            tiles=tuple(tiles),
            owned=tuple(tuple(ids) for ids in owned),
            cp_rank=cp_rank,
            scan_block=scan_block,
        )
        owner = {index: rank for rank, ids in enumerate(owned) for index in ids}
        for block in tiling.blocks:
            if len({owner[index] for index in block}) != 1:
                raise ValueError(
                    f"tiles {block[0]}..{block[-1]} form one scan block but are owned by "
                    "several ranks; cut fragments on block boundaries or use scan_block=1"
                )
        return tiling

    @property
    def blocks(self) -> tuple[tuple[int, ...], ...]:
        """Tile ids of every scan block, in tile order (NOTE [Scan Blocks])."""
        blocks: list[tuple[int, ...]] = []
        start = 0
        for index in range(1, len(self.tiles) + 1):
            if (
                index == len(self.tiles)
                or self.tiles[index].document != self.tiles[start].document
            ):
                for begin in range(start, index, self.scan_block):
                    blocks.append(tuple(range(begin, min(begin + self.scan_block, index))))
                start = index
        return tuple(blocks)

    @property
    def my_blocks(self) -> tuple[int, ...]:
        """Ids (into ``blocks``) of the blocks this rank owns, in its span order."""
        first_tile = {block[0]: b for b, block in enumerate(self.blocks)}
        return tuple(first_tile[index] for index in self.mine if index in first_tile)

    @property
    def mine(self) -> tuple[int, ...]:
        return self.owned[self.cp_rank]

    @property
    def slots(self) -> int:
        """Gather width: the most blocks any rank owns."""
        first_tile = {block[0] for block in self.blocks}
        return max(sum(index in first_tile for index in ids) for ids in self.owned)

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
        blocks = self.blocks
        return CanonicalRouting(
            cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=device),
            bounds=torch.tensor(list(pairwise(offsets)), dtype=torch.int32, device=device),
            mine=_selection(self.mine, device),
            tile_document_offsets=_document_offsets(
                [tile.document for tile in self.tiles], device
            ),
            block_document_offsets=_document_offsets(
                [self.tiles[block[0]].document for block in blocks], device
            ),
            my_block_offsets=torch.tensor(
                [0, *accumulate(len(blocks[b]) for b in self.my_blocks)],
                dtype=torch.int32,
                device=device,
            ),
            my_blocks=_selection(self.my_blocks, device),
        )


def _document_offsets(documents: Sequence[int], device: torch.device) -> torch.Tensor:
    """``int32 [D + 1]`` boundaries of consecutive equal ``documents`` entries."""
    offsets = [0]
    offsets += [i for i in range(1, len(documents)) if documents[i] != documents[i - 1]]
    offsets.append(len(documents))
    return torch.tensor(offsets, dtype=torch.int32, device=device)


class CanonicalRouting(NamedTuple):
    """Device tensors of a :class:`CanonicalTiling` for one rank, built once per layout.

    ``cu_seqlens``/``bounds`` describe the rank's span with one segment per owned tile;
    ``mine`` (and ``my_blocks``) select the rank's tiles (blocks) in a tile-order (block-order)
    buffer, as a slice when they are consecutive so no gather copy is made; the document
    offsets give the per-document scans their restart points.
    """

    cu_seqlens: torch.Tensor
    bounds: torch.Tensor
    mine: torch.Tensor | slice
    tile_document_offsets: torch.Tensor
    block_document_offsets: torch.Tensor
    my_block_offsets: torch.Tensor
    my_blocks: torch.Tensor | slice


def _selection(ids: Sequence[int], device: torch.device | str) -> torch.Tensor | slice:
    """A slice when ``ids`` are consecutive (no gather copy), else an index tensor."""
    if ids and list(ids) == list(range(ids[0], ids[0] + len(ids))):
        return slice(ids[0], ids[0] + len(ids))
    return torch.tensor(list(ids), dtype=torch.int64, device=device)


def tile_stream(cu_seqlens_global: Sequence[int], tile_size: int) -> list[Tile]:
    """Cut every document into ``tile_size`` tiles from its first token; the last may be short."""
    tiles = []
    for document, (start, stop) in enumerate(pairwise(cu_seqlens_global)):
        tiles.extend(
            Tile(document, begin, min(begin + tile_size, stop))
            for begin in range(start, stop, tile_size)
        )
    return tiles


def all_gather_block_maps(
    local: torch.Tensor | None,
    tiling: CanonicalTiling,
    like: torch.Tensor,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Exchange every rank's block maps and return them as ``[blocks, H, V + K, K]`` in block order.

    ``local`` holds this rank's block maps in ``tiling.my_blocks`` order (``None`` for an empty
    rank). Packets are zero-padded to ``tiling.slots`` rows so the collective is a plain
    equal-size all-gather; padding rows never enter a scan. When every rank's blocks are
    contiguous in rank order and no padding is needed, the gathered buffer already is the block
    order and is returned without a copy. ``group=None`` (a single rank owning everything, the
    CP=1 reference) skips the collective.
    """
    width = tiling.slots
    world = len(tiling.owned)
    if group is None:
        assert world == 1 and local is not None and local.shape[0] == len(tiling.blocks)
        return local
    if local is not None and local.shape[0] == width:
        packet = local
    else:
        packet = like.new_zeros((width, *like.shape[1:]))
        if local is not None:
            packet[: local.shape[0]] = local
    gathered = like.new_empty((world * width, *like.shape[1:]))
    with profiler_range("cp/all_gather_blocks"):
        dist.all_gather_into_tensor(gathered, packet, group=group)
    per_rank = [
        [b for b, block in enumerate(tiling.blocks) if block[0] in set(ids)]
        for ids in tiling.owned
    ]
    order = [b for blocks in per_rank for b in blocks]
    if order == list(range(len(order))) and len(order) == world * width:
        return gathered
    rows = [
        rank * width + row for rank, blocks in enumerate(per_rank) for row in range(len(blocks))
    ]
    placed = torch.empty_like(gathered[: len(tiling.blocks)])
    placed[torch.tensor(order, device=like.device)] = gathered[
        torch.tensor(rows, device=like.device)
    ]
    return placed


def canonical_entries(
    tile_maps: torch.Tensor | None,
    tiling: CanonicalTiling,
    routing: CanonicalRouting,
    like: torch.Tensor,
    group: dist.ProcessGroup | None,
    *,
    reverse: bool,
) -> torch.Tensor | None:
    """Entry (or exit when ``reverse``) FP32 states of this rank's tiles, in ``tiling.mine`` order.

    With ``scan_block == 1`` every tile map is gathered and scanned per document. Otherwise the
    fixed three-level tree of NOTE [Scan Blocks]: fold each owned block's tile maps into one block
    map; all-gather block maps; scan them per document for the block entry states; fold each
    owned block's tiles again from its entry state for the tile states. Every fold is the
    left-to-right (right-to-left when ``reverse``) serial fold in three-pass TF32, so the result
    is a function of the tiles' maps and indices only.
    """
    if tiling.scan_block == 1:
        gathered = all_gather_block_maps(tile_maps, tiling, like, group)
        entries = canonical_scan_entries(gathered, routing.tile_document_offsets, reverse=reverse)
        return entries[routing.mine] if tile_maps is not None else None
    block_maps = (
        fold_ranges(tile_maps, routing.my_block_offsets, reverse=reverse)
        if tile_maps is not None
        else None
    )
    gathered = all_gather_block_maps(block_maps, tiling, like, group)
    block_entries = canonical_scan_entries(
        gathered, routing.block_document_offsets, reverse=reverse
    )
    if tile_maps is None:
        return None
    return canonical_scan_entries(
        tile_maps,
        routing.my_block_offsets,
        reverse=reverse,
        initial=block_entries[routing.my_blocks],
    )


class _CanonicalContextParallel(torch.autograd.Function):
    """One batched staged forward/backward per rank glued by the fixed scan tree.

    Every owned tile is one packed subsequence of the rank's span (``cu_seqlens`` at tile
    boundaries), so the leaf kernels see each tile exactly as an independent call would; the
    summary kernels run with ``deterministic_work=True`` so their work partition cannot depend on
    how many tiles the rank owns (see ``build_state_summaries``). Batching is bitwise equal to
    per-tile calls for the fused kernels (``test/test_canonical_tile_batching.py``).
    """

    @staticmethod
    def forward(ctx, q, k, v, gate, beta, tiling, routing, group, stages: StagedOp):
        heads, dim, value_dim = q.shape[2], q.shape[3], v.shape[3]
        zero = q.new_zeros((1, heads, value_dim, dim), dtype=torch.float32)
        like = zero.new_empty((1, heads, value_dim + dim, dim))  # one packed map
        n = len(tiling.mine)
        if n:
            prepared = stages.prepare(q, k, v, gate, beta, cu_seqlens=routing.cu_seqlens)
            maps = prepared.state_summaries(routing.bounds, deterministic_work=True)
        else:
            prepared = None
            maps = None
        entry = canonical_entries(maps, tiling, routing, like, group, reverse=False)
        if n:
            with profiler_range("cp/run"):
                output, _ = prepared.run(entry, output_final_state=False)
            saved = tuple(prepared.saved)
            ctx.saved_type = type(prepared.saved)
            ctx.scale = prepared.scale
        else:
            output = v[:, :0]
            entry = zero[:0]
            saved = ()
            ctx.saved_type = None
            ctx.scale = None
        ctx.tiling = tiling
        ctx.routing = routing
        ctx.group = group
        ctx.stages = stages
        ctx.input_meta = tuple((t.shape, t.dtype) for t in (q, k, v, gate, beta))
        ctx.output_meta = (output.shape, output.dtype)
        # ``zero`` and ``like`` give an empty rank the map shapes its collective must match.
        ctx.save_for_backward(*saved, entry, zero, like)
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output):
        tiling: CanonicalTiling = ctx.tiling
        routing: CanonicalRouting = ctx.routing
        *saved, entry, zero, like = ctx.saved_tensors
        if d_output is None:
            d_output = zero.new_zeros(*ctx.output_meta)
        n = len(tiling.mine)
        if n:
            backward = ctx.stages.prepare_backward(
                ctx.saved_type._make(saved), d_output, entry, scale=ctx.scale
            )
            maps = backward.state_grad_summaries(routing.bounds, deterministic_work=True)
        else:
            backward = None
            maps = None
        exit_cotangent = canonical_entries(maps, tiling, routing, like, ctx.group, reverse=True)
        if n:
            with profiler_range("cp/run"):
                grads = backward.run(exit_cotangent)[:5]
        else:
            grads = tuple(zero.new_empty(shape, dtype=dtype) for shape, dtype in ctx.input_meta)
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
    "all_gather_block_maps",
    "canonical_entries",
    "context_parallel_chunk_deterministic",
    "tile_stream",
]
