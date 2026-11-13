# SPDX-License-Identifier: BSD-3-Clause

"""Plans and scratch buffers shared by the GDN and KDA drivers.

Everything here is a pure function of one call's shapes and device. The plan schemes are
memoized on the full shape (and SM count); scratch buffers are allocated per call. The compiled
launches are cached by the kernel builders on their static configuration.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

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
    min_tokens_per_piece: int,
    *,
    reverse: bool = False,
) -> int:
    """Pieces per sequence of the exact chain, 0 when pieces would be short."""
    pieces = choose_pieces(
        num_seqs=num_seqs,
        heads_out=heads_out,
        num_sm=num_sm,
        total_tokens=tokens,
        b_t=b_t,
        reverse=reverse,
    )
    if pieces and tokens < min_tokens_per_piece * pieces * num_seqs:
        pieces = 0
    return pieces


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
        min_chain_tokens_per_piece: int,
    ):
        num_sm = _num_sm(device)
        pieces = _chain_pieces(
            tokens, num_seqs, heads_out, num_sm, b_t, min_chain_tokens_per_piece
        )
        # Upstream splits d_v only at a slot budget of exactly two; any unchained plan whose two
        # CTAs per tile fit in one wave gains (GDN 1x2048: 67 -> 58 us, 1x8192: 220 -> 193 us).
        dv = not pieces and dim_v == 128 and num_sm // (num_seqs * heads_out) >= DV_SPLIT_TILES
        return cls(pieces, DV_SPLIT_TILES if dv else 1, num_sm)


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
        min_chain_tokens_per_piece: int,
    ):
        num_sm = _num_sm(device)
        pieces = _chain_pieces(
            tokens, num_seqs, heads_out, num_sm, b_t, min_chain_tokens_per_piece, reverse=True
        )
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
