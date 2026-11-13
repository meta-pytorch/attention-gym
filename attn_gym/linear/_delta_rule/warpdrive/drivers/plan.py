# SPDX-License-Identifier: BSD-3-Clause

"""Plans and scratch buffers shared by the GDN and KDA drivers.

Everything here is a pure function of one call's shapes and device. The plan schemes are
memoized on the full shape (and SM count); scratch buffers are allocated per call. The compiled
launches are cached by the kernel builders on their static configuration.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import NamedTuple

import torch

from attn_gym._backends.cute.utils import get_device_properties

from ..piece_chain import choose_pieces, piece_table_layout
from ..ragged import new_tma_slots, tma_slot_rows
from ..work_tables import (
    WORK_ITEM_FIELDS,
    chunk_scratch_rows,
    compute_ideal_chunks,
    max_work_items,
)

# Value tiles of the unchained d_v split.
DV_SPLIT_TILES = 2

# NOTE [Chain plan cost]
# An exact chain of ``pieces`` slots per sequence runs x = tokens / (num_seqs * pieces) tokens
# per slot, but its summaries, state chain and extra launches make each token dearer and add
# fixed work. In tokens of the unchained plan it replaces (the uncut table backward, the d_v
# split forward), which costs x * pieces on the critical path, the chain costs
#
#     x * (cost.piece + CHAIN_FILL_COST * fill) + cost.overhead_tokens
#
# where fill = min(1, num_seqs * pieces * heads_out / num_sm) is the share of the SMs the chain
# grid occupies (a full grid slows every piece); the plan chains when that is cheaper. Each
# driver's ChainCost per direction and head dimension was fitted on a 152-SM GB300 (SM103) to
# CUDA-graph timings, BF16, no state (benchmarks/warpdrive_chain_plans.py, 2026-11-12): at D=128
# single sequences 1x{512..16384} at H in {2..48} and 2-8 sequence packs up to 64 tiles, at D=64
# (GDN) 1x{512..16384} at H in {2..64} and 2-8 sequence packs up to 128 tiles. At the geometry's
# piece count the chosen plan was never more than 2% slower than the faster of the two at D=128,
# and at D=64 never slower.
CHAIN_FILL_COST = 1.5


class ChainCost(NamedTuple):
    """One driver direction's chain cost; see NOTE [Chain plan cost]."""

    piece: float  # unchained tokens per chain token on an idle grid
    overhead_tokens: int


# NOTE [Summary split cost]
# A staged summary walks each subsequence's chunks serially, one CTA per (subsequence, head), so
# a few long subsequences leave most SMs idle. Cut into pieces of ``span`` chunks, the pieces
# run in parallel and their maps compose on the state chain, one step per piece. In chunk steps
# of the uncut summary, a subsequence of n chunks then costs
#
#     span * (1 + cost.fill * fill) + ceil(n / span) * cost.compose + cost.overhead
#
# instead of n, where fill = min(1, kernels * pieces * heads / num_sm) is the share of the SMs
# the ``cost.kernels`` concurrent summary kernels' pieces occupy. The host sees only the span's
# token count, so n assumes equal subsequences; the span is the cheapest one of at least
# SUMMARY_MIN_TOKENS tokens whose pieces fit one wave, and the summary splits when that beats n.
# Each SummaryCost was fitted on a 152-SM GB300 (SM103) to CUDA-graph timings of the summary
# entry points, BF16, 1 and 2 subsequences of 512-16384 tokens at 1-16 heads, spans of 2-152
# pieces (2026-11-13). Over those 240 shapes the chosen span was within 8% of the fastest one
# (or of uncut where that won) except GDN's forward on 2x4096 tokens at 16 heads, left uncut at
# 1.2x the best split. No measured winner had pieces shorter than SUMMARY_MIN_TOKENS.
SUMMARY_MIN_TOKENS = 128


class SummaryCost(NamedTuple):
    """One summary direction's split cost; see NOTE [Summary split cost]."""

    compose: float  # one composition step, in summary chunk steps
    overhead: float  # the planner, scratch clears and the chain's fixed cost, in chunk steps
    fill: float  # slowdown of a piece's chunk steps on a full grid
    kernels: int = 1  # summary kernels sharing the SMs at once


def summary_span(
    tokens: int, num_seqs: int, heads: int, device, *, b_t: int, cost: SummaryCost
) -> int:
    """Chunks per piece of a split summary, 0 to run every subsequence uncut (see NOTE
    [Summary split cost])."""
    return _summary_span(tokens, num_seqs, heads, _num_sm(device), b_t, cost)


@functools.lru_cache(maxsize=256)
def _summary_span(
    tokens: int, num_seqs: int, heads: int, num_sm: int, b_t: int, cost: SummaryCost
) -> int:
    num_seqs = max(1, num_seqs)
    chunks = -(-tokens // (b_t * num_seqs))
    wave = num_sm // (max(1, heads) * cost.kernels)
    best, best_cost = 0, chunks - cost.overhead
    for pieces in range(2, wave // num_seqs + 1):
        span = -(-chunks // pieces)
        if span * b_t < SUMMARY_MIN_TOKENS:
            break
        fill = min(1.0, cost.kernels * num_seqs * pieces * heads / num_sm)
        split = span * (1 + cost.fill * fill) + -(-chunks // span) * cost.compose
        if split < best_cost:
            best, best_cost = span, split
    return best


def int32(n: int, device) -> torch.Tensor:
    return torch.empty(n, dtype=torch.int32, device=device)


def aligned(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a tensor whose base is not 16-byte aligned; the kernels assume 16-byte bases."""
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _num_sm(device) -> int:
    return get_device_properties(device).multi_processor_count


@functools.lru_cache(maxsize=256)
def _chain_pieces(
    tokens: int,
    num_seqs: int,
    heads_out: int,
    num_sm: int,
    b_t: int,
    cost: ChainCost,
    *,
    reverse: bool = False,
) -> int:
    """Pieces per sequence of the exact chain, 0 when it would not pay for itself (see NOTE
    [Chain plan cost])."""
    pieces = choose_pieces(
        num_seqs=num_seqs,
        heads_out=heads_out,
        num_sm=num_sm,
        total_tokens=tokens,
        b_t=b_t,
        reverse=reverse,
    )
    if not pieces:
        return 0
    fill = min(1.0, num_seqs * pieces * heads_out / num_sm)
    tokens_per_piece = tokens / (num_seqs * pieces)
    saved = tokens_per_piece * (pieces - cost.piece - CHAIN_FILL_COST * fill)
    return pieces if saved >= cost.overhead_tokens else 0


@dataclass
class ForwardPlan:
    """Forward scheme for one (shape, device): the exact chain, or an uncut table whose tiles
    optionally split d_v. Plans are shape-dependent by design; bitwise ownership invariance holds
    within a plan. Each driver subclasses this with a ``build`` over its own tile size."""

    pieces: int
    tiles_per_head: int
    num_sm: int

    @property
    def chain(self) -> bool:
        return self.pieces > 0

    @classmethod
    def for_shape(
        cls,
        tokens: int,
        num_seqs: int,
        heads_out: int,
        dim_v: int,
        device,
        *,
        b_t: int,
        chain_cost: ChainCost,
    ):
        num_sm = _num_sm(device)
        pieces = _chain_pieces(tokens, num_seqs, heads_out, num_sm, b_t, chain_cost)
        return cls(pieces, 1, num_sm) if pieces else cls.uncut(num_seqs, heads_out, dim_v, num_sm)

    @classmethod
    def uncut(cls, num_seqs: int, heads_out: int, dim_v: int, num_sm: int):
        """The unchained plan, for shapes the chain does not serve (paged prefill) or loses on.

        Upstream splits d_v only at a slot budget of exactly two; any unchained plan whose two
        CTAs per tile fit in one wave gains (GDN 1x2048: 67 -> 58 us, 1x8192: 220 -> 193 us).
        """
        dv = dim_v == 128 and num_sm // (num_seqs * heads_out) >= DV_SPLIT_TILES
        return cls(0, DV_SPLIT_TILES if dv else 1, num_sm)


@dataclass
class BackwardPlan:
    """Uncut or exact-chain backward scheme for one shape; mirrors ``ForwardPlan``."""

    pieces: int
    num_sm: int

    @property
    def chain(self) -> bool:
        return self.pieces > 0

    @classmethod
    def for_shape(
        cls,
        tokens: int,
        num_seqs: int,
        heads_out: int,
        device,
        *,
        b_t: int,
        chain_cost: ChainCost,
    ):
        num_sm = _num_sm(device)
        pieces = _chain_pieces(tokens, num_seqs, heads_out, num_sm, b_t, chain_cost, reverse=True)
        return cls(pieces, num_sm)


def split_scratch(
    split: bool,
    tokens: int,
    num_seqs: int,
    heads_out: int,
    num_sm: int,
    n_tiles: int,
    b_t: int,
    device,
):
    """``(ideal_chunks, work_item_rows, item_scratch, chunk_scratch)`` of the warmup work table;
    without ``split`` one row per tile and no scratch."""
    if not split:
        return None, n_tiles, None, None
    ideal = compute_ideal_chunks(tokens, heads_out, num_sm, b_t)
    rows = max_work_items(tokens, num_seqs, heads_out, ideal, b_t, num_sm)
    item_scratch = torch.empty(rows, WORK_ITEM_FIELDS, dtype=torch.int32, device=device)
    chunk_scratch = torch.empty(
        chunk_scratch_rows(tokens, num_seqs, b_t), heads_out, dtype=torch.float32, device=device
    )
    return ideal, rows, item_scratch, chunk_scratch


def tma_slots(*, tensors: int, b_t: int, keys: int, device) -> torch.Tensor:
    """One TMA descriptor slot buffer shared by the kernels of a forward or backward call. They
    run one after another; each uses at most ``tensors`` ragged tensors and ``keys`` item keys
    (ordering contract: ``ragged.sequence_descriptors``)."""
    return new_tma_slots(tma_slot_rows(tensors, b_t, keys), device)


def chain_buffers(
    num_seqs: int,
    pieces: int,
    heads_out: int,
    dim_v: int,
    dim_k: int,
    device,
    *,
    backward: bool,
) -> dict:
    """Piece table, work-item tables, scheduler rings and FP32 piece states of an exact chain."""
    num_pieces = num_seqs * pieces
    table = piece_table_layout(num_seqs, pieces, heads_out)
    piece_table = torch.zeros(table.nbytes // 4, dtype=torch.int32, device=device)

    def words(offset: int, count: int) -> torch.Tensor:
        return piece_table[offset // 4 : offset // 4 + count]

    def state(dim: int = dim_v) -> torch.Tensor:
        return torch.empty(num_pieces, heads_out, dim, dim_k, dtype=torch.float32, device=device)

    buffers = {
        "cu_pieces": words(table.cu_pieces, num_pieces + 1),
        "main_rows": words(table.main_rows, num_seqs + 1),
        "summary_rows": words(table.summary_rows, num_seqs + 1),
        "main_count": words(table.main_count, 1),
        "summary_count": words(table.summary_count, 1),
        "work_items": torch.empty(
            num_pieces * heads_out, WORK_ITEM_FIELDS, dtype=torch.int32, device=device
        ),
        "work_items_summary": torch.empty(
            table.item_rows, WORK_ITEM_FIELDS, dtype=torch.int32, device=device
        ),
        "state_h": state(),
        "state_m": state(dim_k),
        "state_x": state(),
    }
    if not backward:
        schedulers = int32(6, device)
        return dict(
            buffers,
            scheduler_all=schedulers,
            scheduler_prefill=schedulers[0:2],
            scheduler_summary=schedulers[2:4],
        )
    schedulers = int32(6, device)
    return dict(
        buffers,
        state_g=state(),
        state_dx_end=state(),
        scheduler_all=schedulers,
        scheduler_bwd=schedulers[0:2],
        scheduler_summary=schedulers[2:4],
        scheduler_series=schedulers[4:6],
    )
