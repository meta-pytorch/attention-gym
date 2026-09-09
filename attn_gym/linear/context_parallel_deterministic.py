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
    scan      per document, a work-efficient (Blelloch) inclusive prefix over the leaf maps in
              tile order (forward) and reverse tile order (backward). The prefix at tile ``j`` is
              a function of that document's leaves ``0..j`` and of ``j`` alone.
    entry     ``merge_state(0, prefix[i - 1])`` for tile ``i``; the first tile of a document enters
              with zero. Exits are the reverse analogue.

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
from attn_gym.linear.context_parallel import StagedOp, _ieee_fp32_matmul

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
        ownership would cut a tile and the leaf arithmetic would depend on the fragment table.
        """
        if tile_size <= 0 or tile_size % SUMMARY_CHUNK:
            raise ValueError(f"tile_size must be a positive multiple of {SUMMARY_CHUNK}")
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
        return cls(tiles=tuple(tiles), owned=tuple(tuple(ids) for ids in owned), cp_rank=cp_rank)

    @property
    def mine(self) -> tuple[int, ...]:
        return self.owned[self.cp_rank]

    @property
    def slots(self) -> int:
        """Gather width: the most tiles any rank owns."""
        return max(len(ids) for ids in self.owned)

    def global_token_ids(self, device: torch.device | str) -> torch.Tensor:
        """Global token id of every span token of this rank (tiles concatenated in span order)."""
        pieces = [
            torch.arange(self.tiles[i].start, self.tiles[i].stop, device=device) for i in self.mine
        ]
        return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64, device=device)

    def span_offsets(self) -> tuple[int, ...]:
        """Span-local ``[start, stop)`` of each owned tile, in span order."""
        return tuple(accumulate((self.tiles[i].length for i in self.mine), initial=0))


def tile_stream(cu_seqlens_global: Sequence[int], tile_size: int) -> list[Tile]:
    """Cut every document into ``tile_size`` tiles from its first token; the last may be short."""
    tiles = []
    for document, (start, stop) in enumerate(pairwise(cu_seqlens_global)):
        tiles.extend(
            Tile(document, begin, min(begin + tile_size, stop))
            for begin in range(start, stop, tile_size)
        )
    return tiles


@_ieee_fp32_matmul()
def _compose_into(right: torch.Tensor, left: torch.Tensor) -> None:
    """``right <- [B_L @ A_R + B_R; A_L @ A_R]`` in place on ``[..., V + K, K]`` packed maps."""
    value_dim = left.shape[-2] - left.shape[-1]
    composed = left @ right[..., value_dim:, :]
    composed[..., :value_dim, :] += right[..., :value_dim, :]
    right.copy_(composed)


def canonical_scan(maps: torch.Tensor) -> torch.Tensor:
    """Inclusive prefix along dim 1 of ``[D, L, H, V + K, K]`` maps, one independent row per document.

    Work-efficient (Blelloch) scan: an up-sweep folds pairs at strides 1, 2, 4, ... into the
    right element, then a down-sweep pushes each node's prefix into the elements it did not
    cover. Every stage is one batched IEEE-FP32 matmul over strided views of the buffer. The
    prefix at position ``j`` reads positions ``<= j`` only, so it is a function of the leaves
    ``0..j`` and of ``j`` alone: padding a row on the right (with any maps) does not change it,
    and neither does which rank produced a leaf or how leaves were transported. About ``2L``
    compositions per row in ``2 log2 L`` stages.
    """
    length = maps.shape[1]
    stage = maps.clone()
    stride = 1
    while 2 * stride - 1 < length:
        right = stage[:, 2 * stride - 1 :: 2 * stride]
        _compose_into(right, stage[:, stride - 1 :: 2 * stride][:, : right.shape[1]])
        stride *= 2
    stride //= 2
    while stride >= 1:
        if 3 * stride - 1 < length:
            right = stage[:, 3 * stride - 1 :: 2 * stride]
            _compose_into(right, stage[:, 2 * stride - 1 :: 2 * stride][:, : right.shape[1]])
        stride //= 2
    return stage


def boundary_states(maps: torch.Tensor, tiles: Sequence[Tile], *, reverse: bool) -> torch.Tensor:
    """Entry (or exit when ``reverse``) FP32 ``[N, H, V, K]`` state of every tile in tile order.

    Documents are scanned independently, so a document's states depend only on its own tiles,
    not on its position in the packed stream or on its neighbours. Documents are batched by the
    power-of-two bucket of their tile count and right-padded with identity maps to the bucket
    length; the prefix at a position never reads past it (see ``canonical_scan``), so the padding
    and the batching are numerically inert and only bound the wasted work to under 2x. Single-tile
    documents need no scan. The first tile of a document (last when ``reverse``) enters with zero;
    every other tile enters with the bias of the exclusive prefix (``merge_state(0, prefix)``).
    """
    count, heads, packed, key_dim = maps.shape
    value_dim = packed - key_dim
    rows: dict[int, list[int]] = {}
    for index, tile in enumerate(tiles):
        rows.setdefault(tile.document, []).append(index)
    entries = maps.new_zeros((count, heads, value_dim, key_dim))
    buckets: dict[int, list[list[int]]] = {}
    for ids in rows.values():
        if len(ids) > 1:
            ordered = list(reversed(ids)) if reverse else list(ids)
            buckets.setdefault(1 << (len(ids) - 1).bit_length(), []).append(ordered)
    identity = torch.eye(key_dim, device=maps.device)
    for length, documents in sorted(buckets.items()):
        padded = maps.new_zeros((len(documents), length, heads, packed, key_dim))
        padded[:, :, :, value_dim:, :] = identity
        row_index = torch.tensor(
            [d for d, ids in enumerate(documents) for _ in ids[1:]], device=maps.device
        )
        col_index = torch.tensor(
            [j for ids in documents for j in range(1, len(ids))], device=maps.device
        )
        tile_index = torch.tensor([i for ids in documents for i in ids[1:]], device=maps.device)
        leaf_index = torch.tensor([i for ids in documents for i in ids], device=maps.device)
        padded[
            torch.tensor([d for d, ids in enumerate(documents) for _ in ids], device=maps.device),
            torch.tensor([j for ids in documents for j in range(len(ids))], device=maps.device),
        ] = maps[leaf_index]
        prefix = canonical_scan(padded)
        entries[tile_index] = prefix[row_index, col_index - 1, :, :value_dim, :]
    return entries


def all_gather_tile_maps(
    local: torch.Tensor | None,
    tiling: CanonicalTiling,
    like: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Exchange every rank's leaf maps and return them as ``[N, H, V + K, K]`` in tile order.

    ``local`` holds this rank's maps in ``tiling.mine`` order (``None`` for an empty rank).
    Packets are zero-padded to ``tiling.slots`` rows so the collective is a plain equal-size
    all-gather; the padding rows never enter a scan. When every rank owns a contiguous run of
    tiles in rank order and no padding is needed, the gathered buffer already is the tile order
    and is returned without a copy.
    """
    width = tiling.slots
    world = len(tiling.owned)
    if local is not None and local.shape[0] == width:
        packet = local
    else:
        packet = like.new_zeros((width, *like.shape[1:]))
        if local is not None:
            packet[: local.shape[0]] = local
    gathered = like.new_empty((world * width, *like.shape[1:]))
    with profiler_range("cp/all_gather_tiles"):
        dist.all_gather_into_tensor(gathered, packet, group=group)
    order = [index for ids in tiling.owned for index in ids]
    if order == list(range(len(order))) and len(order) == world * width:
        return gathered
    rows = [rank * width + row for rank, ids in enumerate(tiling.owned) for row in range(len(ids))]
    placed = torch.empty_like(gathered[: len(tiling.tiles)])
    placed[torch.tensor(order, device=like.device)] = gathered[
        torch.tensor(rows, device=like.device)
    ]
    return placed


class _CanonicalContextParallel(torch.autograd.Function):
    """One batched staged forward/backward per rank glued by fixed per-document scans.

    Every owned tile is one packed subsequence of the rank's span (``cu_seqlens`` at tile
    boundaries), so the leaf kernels see each tile exactly as an independent call would; the
    summary kernels run with ``deterministic_work=True`` so their work partition cannot depend on
    how many tiles the rank owns (see ``build_state_summaries``). Batching is bitwise equal to
    per-tile calls for the fused kernels (``test/test_canonical_tile_batching.py``).
    """

    @staticmethod
    def forward(ctx, q, k, v, gate, beta, tiling: CanonicalTiling, group, stages: StagedOp):
        offsets = tiling.span_offsets()
        heads, dim, value_dim = q.shape[2], q.shape[3], v.shape[3]
        zero = q.new_zeros((1, heads, value_dim, dim), dtype=torch.float32)
        like = zero.new_empty((1, heads, value_dim + dim, dim))  # one packed map
        n = len(tiling.mine)
        if n:
            cu = torch.tensor(offsets, dtype=torch.int32, device=q.device)
            bounds = torch.tensor(list(pairwise(offsets)), dtype=torch.int32, device=q.device)
            prepared = stages.prepare(q, k, v, gate, beta, cu_seqlens=cu)
            maps = prepared.state_summaries(bounds, deterministic_work=True)
        else:
            prepared = None
            maps = None
        entries = boundary_states(
            all_gather_tile_maps(maps, tiling, like, group), tiling.tiles, reverse=False
        )
        if n:
            entry = entries[list(tiling.mine)]
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
        offsets = tiling.span_offsets()
        *saved, entry, zero, like = ctx.saved_tensors
        if d_output is None:
            d_output = zero.new_zeros(*ctx.output_meta)
        n = len(tiling.mine)
        if n:
            bounds = torch.tensor(list(pairwise(offsets)), dtype=torch.int32, device=zero.device)
            backward = ctx.stages.prepare_backward(
                ctx.saved_type._make(saved), d_output, entry, scale=ctx.scale
            )
            maps = backward.state_grad_summaries(bounds, deterministic_work=True)
        else:
            backward = None
            maps = None
        exits = boundary_states(
            all_gather_tile_maps(maps, tiling, like, ctx.group), tiling.tiles, reverse=True
        )
        if n:
            exit_cotangent = exits[list(tiling.mine)]
            with profiler_range("cp/run"):
                grads = backward.run(exit_cotangent)[:5]
        else:
            grads = tuple(zero.new_empty(shape, dtype=dtype) for shape, dtype in ctx.input_meta)
        return (*grads, None, None, None)


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
) -> torch.Tensor:
    """Run a staged delta-rule op over this rank's canonical tiles (NOTE [Canonical Tiles]).

    Args:
        stages: The op's staged entry points, as for ``context_parallel_chunk``.
        q: This rank's span: its owned tiles concatenated in ``tiling.mine`` order; ``k``, ``v``,
            ``gate``, ``beta`` follow the same layout.
        tiling: The stream's canonical tiling and this rank's ownership.
        group: Process group containing exactly the tiling's ranks, in order.

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
    return _CanonicalContextParallel.apply(q, k, v, gate, beta, tiling, group, stages)


__all__ = [
    "SUMMARY_CHUNK",
    "CanonicalTiling",
    "Tile",
    "all_gather_tile_maps",
    "boundary_states",
    "canonical_scan",
    "context_parallel_chunk_deterministic",
    "tile_stream",
]
