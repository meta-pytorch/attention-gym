# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Exact piece chain for the chunked linear-attention kernels: the piece rule (``choose_pieces``), the piece table
(piece-wise ``cu_pieces`` and the piece work-item tables, built by ``kernel/*_chain_prologue_f16.py``) and the fp32 state
chain that seeds every piece with its exact incoming state.  When ``B * HO`` leaves SMs idle the batch is cut into one
wave of ``num_seqs * pieces`` slots that run as independent sequences, every slot spanning the same number of chunks so a
sequence fills slots in proportion to its length; their summaries ``H_j`` (state from a zero seed) and ``M_j``
(transition, stored domain ``M_buf = M^T``) are chained in fp32::

    X_{j+1} = X_j @ M_j + H_j                       (transpose=False)
    X_j     = X_{j+1} @ M_j^T + G_j                 (transpose=True, H plays G)

``emit_summary`` also composes the running product ``summary_m`` so ``tail = seed @ summary_m + tail_zero_seed``.
Layouts, fp32 unless stated: ``H``, ``X`` ``[num_seqs * pieces, HO, V, K]``; ``M`` ``[num_seqs * pieces, HO, K, K]``;
``seed`` / ``tail`` ``[num_seqs, HO, V, K]`` (fp32 or bf16); ``summary_m`` ``[num_seqs, HO, K, K]``.  Slots are flat in
sequence order: sequence ``b`` owns ``main_rows[b] // HO .. main_rows[b + 1] // HO``, its piece ``j`` being slot
``main_rows[b] // HO + j`` (the public summary chain without a piece table keeps the fixed grid ``b * pieces + j``).
Empty pieces carry ``H = 0``, ``M = I`` and fall through exactly.  Piece table workspace (``piece_table_layout``):
``main_rows`` / ``summary_rows`` int32 ``[num_seqs + 1]`` (row bases of each sequence in the main and summary work-item
tables, entry ``num_seqs`` the row count) and ``cu_pieces`` int32 ``[num_seqs * pieces + 1]`` in real tokens, the flat
slot starts followed by the batch end, each 16-byte aligned; the rows of a sequence's last filled piece carry
``final_dst`` and those of its first ``dstate_dst``, so the main kernels write ``final_state`` / ``d_initial_state`` in place.
"""

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.experimental.primitives as nvvm
from cutlass import cute

from ..tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from ..tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16
from ..tile_dsl.tma import (
    cp_async_commit,
    cp_async_wait,
    ld_global,
    ld_global_v2,
    ld_global_v4,
    ld_shared_v2,
    ld_shared_v4,
    st_global,
    st_global_v2,
    st_global_v4,
    st_shared_v2,
    st_shared_v4,
)

USE_PDL = True

CHAIN_MIN_PIECES = 3
CHAIN_MIN_PIECES_REVERSE = 2
CHAIN_MAX_PIECES = 16
LENGTH_RULE_PIECE_TOKENS = 8192
CHAIN_MIN_UNITS_PER_PIECE = 4
DV_SPLIT_TILES = 2

PIECE_TABLE_ALIGN = 256
CHAIN_K_WARPS = 4
CHAIN_ROW_GROUPS = 2
CHAIN_WARPS = CHAIN_K_WARPS * CHAIN_ROW_GROUPS
CHAIN_LANES = 32
CHAIN_THREADS = CHAIN_WARPS * CHAIN_LANES
CHAIN_K_TILE = 8
CHAIN_M_BUFFERS = 2

DTYPE_NAMES = {
    cutlass.Float32: "float32",
    cutlass.BFloat16: "bfloat16",
    cutlass.Float16: "float16",
    cutlass.Int32: "int32",
    cutlass.Int64: "int64",
}
STATE_DTYPES = ("float32", "bfloat16")


def dtype_name(dtype) -> str:
    """Bare dtype name (``"float32"``, ``"int64"``, ...) from a torch dtype, a cutlass type or a string."""
    if dtype in DTYPE_NAMES:
        return DTYPE_NAMES[dtype]
    return str(dtype).split(".")[-1]


# ---- piece rule -----------------------------------------------------------------------------------


def piece_budget(*, num_seqs, heads_out, num_sm, total_tokens, b_t, expand_num, unit_chunks):
    """The one-wave slot budget of a plan without ``batch_invariant``: ``num_sm // tiles`` capped at ``CHAIN_MAX_PIECES``
    and at ``CHAIN_MIN_UNITS_PER_PIECE`` units of ``unit_chunks`` chunks per piece."""
    num_seqs = max(1, int(num_seqs))
    tiles = num_seqs * max(1, int(heads_out))
    total_chunks = -(-(int(total_tokens) * max(1, int(expand_num))) // int(b_t))
    return min(
        int(num_sm) // tiles,
        CHAIN_MAX_PIECES,
        total_chunks // (num_seqs * CHAIN_MIN_UNITS_PER_PIECE * int(unit_chunks)),
    )


def choose_pieces(
    *,
    num_seqs,
    heads_out,
    num_sm,
    total_tokens,
    b_t,
    cadence_tokens,
    batch_invariant,
    expand_num,
    reverse=False,
    compose_tail=False,
):
    """``(pieces, unit_chunks)`` of one plan, a pure function of shapes shared by forward and backward; ``pieces`` is the
    slot budget per sequence, ``num_seqs * pieces`` the wave the piece table hands out, 0 when the plan does not chain.
    Boundaries are multiples of ``unit_chunks = lcm(expand_num, cadence_chunks)`` chunks.  Without ``batch_invariant`` the
    budget is the one-wave geometry ``num_sm // (num_seqs * heads_out)`` capped at ``CHAIN_MAX_PIECES`` and
    ``CHAIN_MIN_UNITS_PER_PIECE``, chaining from ``CHAIN_MIN_PIECES`` (``CHAIN_MIN_PIECES_REVERSE`` for ``reverse``); the
    piece table then gives every slot the same span, ``ceil(total_chunks / (num_seqs * pieces))`` chunks, so a sequence
    fills slots in proportion to its length and an uneven batch walks the same critical path as an even one.  With
    ``batch_invariant`` the length rule applies: ``clamp(ceil(total / LENGTH_RULE_PIECE_TOKENS), 1, CHAIN_MAX_PIECES)``
    slots, each sequence filling ``ceil(len_b / LENGTH_RULE_PIECE_TOKENS)`` of them on device, so outputs are bitwise the
    same alone and in any batch; a provably one-piece batch runs uncut unless ``compose_tail`` (the summaries chain even
    then)."""
    b_t = int(b_t)
    expand_num = max(1, int(expand_num))
    cadence_chunks = max(1, int(cadence_tokens) // b_t)
    unit_chunks = math.lcm(expand_num, cadence_chunks)
    if batch_invariant:
        pieces = min(max(1, -(-int(total_tokens) // LENGTH_RULE_PIECE_TOKENS)), CHAIN_MAX_PIECES)
        if pieces == 1 and not compose_tail:
            return 0, unit_chunks
        return pieces, unit_chunks
    pieces = piece_budget(
        num_seqs=num_seqs,
        heads_out=heads_out,
        num_sm=num_sm,
        total_tokens=total_tokens,
        b_t=b_t,
        expand_num=expand_num,
        unit_chunks=unit_chunks,
    )
    if pieces >= (CHAIN_MIN_PIECES_REVERSE if reverse else CHAIN_MIN_PIECES):
        return pieces, unit_chunks
    return 0, unit_chunks


# ---- piece table ----------------------------------------------------------------------------------


class PieceTableLayout(NamedTuple):
    """Byte offsets of the piece-table workspace regions, the work-item row capacity and the total size."""

    main_rows: int
    summary_rows: int
    cu_pieces: int
    main_count: int
    summary_count: int
    item_rows: int
    nbytes: int


def piece_table_layout(num_seqs: int, pieces: int, heads_out: int) -> PieceTableLayout:
    """``main_rows`` / ``summary_rows`` int32 ``[num_seqs + 1]`` (each sequence's first row; entry ``num_seqs``
    is the table's row count, the ``main_count`` / ``summary_count`` region) and ``cu_pieces`` int32
    ``[num_seqs * pieces + 1]``, each 16-aligned; 256-aligned total.  ``item_rows`` is the work-item capacity."""
    num_seqs = int(num_seqs)
    rows_bytes = -(-4 * (num_seqs + 1) // 16) * 16
    cu_pieces = 2 * rows_bytes
    cu_bytes = -(-4 * (num_seqs * int(pieces) + 1) // 16) * 16
    nbytes = -(-(cu_pieces + cu_bytes) // PIECE_TABLE_ALIGN) * PIECE_TABLE_ALIGN
    return PieceTableLayout(
        0,
        rows_bytes,
        cu_pieces,
        4 * num_seqs,
        rows_bytes + 4 * num_seqs,
        num_seqs * int(pieces) * int(heads_out),
        nbytes,
    )


@cute.jit
def chunks_per_slot(unit_chunks: cutlass.Constexpr[int], total_chunks, slots):
    """Chunks per slot when ``total_chunks`` are shared by ``slots`` slots: ``ceil(total_chunks / slots)`` rounded up to a
    whole number of ``unit_chunks``, at least one unit."""
    span = (total_chunks + slots - cutlass.Int32(1)) // slots
    span = span if span > cutlass.Int32(1) else cutlass.Int32(1)
    units = (span + cutlass.Int32(unit_chunks - 1)) // cutlass.Int32(unit_chunks)
    return units * cutlass.Int32(unit_chunks)


@cute.jit
def piece_count(
    pieces: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    length,
    span_chunks,
):
    """Slots a sequence of ``length`` real tokens takes: ``ceil(chunks / span_chunks)`` (the length rule:
    ``clamp(ceil(length / LENGTH_RULE_PIECE_TOKENS), 1, pieces)``), at least one."""
    if cutlass.const_expr(length_rule):
        count = (length + cutlass.Int32(LENGTH_RULE_PIECE_TOKENS - 1)) // cutlass.Int32(
            LENGTH_RULE_PIECE_TOKENS
        )
        count = count if count < cutlass.Int32(pieces) else cutlass.Int32(pieces)
    else:
        chunks = (length * cutlass.Int32(expand_num) + cutlass.Int32(b_t - 1)) // cutlass.Int32(
            b_t
        )
        count = (chunks + span_chunks - cutlass.Int32(1)) // span_chunks
    return count if count > cutlass.Int32(1) else cutlass.Int32(1)


@cute.jit
def piece_span(
    pieces: cutlass.Int32,
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    length,
    span_chunks,
):
    """``(span_tokens, last)`` of a sequence of ``length`` real tokens taking ``piece_count`` slots: the piece span in
    real tokens (Int64, the sequence's chunks spread evenly over its slots and rounded up to units) and the index of its
    last filled slot (0 when empty)."""
    chunks = (length * cutlass.Int32(expand_num) + cutlass.Int32(b_t - 1)) // cutlass.Int32(b_t)
    count = piece_count(pieces, b_t, expand_num, length_rule, length, span_chunks)
    own_span = (chunks + count - cutlass.Int32(1)) // count
    own_span = own_span if own_span > cutlass.Int32(1) else cutlass.Int32(1)
    span_units = (own_span + cutlass.Int32(unit_chunks - 1)) // cutlass.Int32(unit_chunks)
    span_tokens = cutlass.Int64(span_units) * cutlass.Int64(unit_chunks * b_t // expand_num)
    span_tokens32 = span_tokens.to(cutlass.Int32)
    filled = (length + span_tokens32 - cutlass.Int32(1)) // span_tokens32
    last = filled - cutlass.Int32(1) if filled > cutlass.Int32(1) else cutlass.Int32(0)
    return span_tokens, last


@cute.jit
def piece_table_body(
    n_threads: cutlass.Constexpr[int],
    pieces: cutlass.Int32,
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    heads_out: cutlass.Int32,
    tidx,
    num_seqs: cutlass.Int32,
    mCu: cute.Tensor,
    mCuPieces: cute.Tensor,
    mMainRows: cute.Tensor,
    mSummaryRows: cute.Tensor,
):
    """Piece table of one CTA (every thread calls; CTA barriers inside).  The wave is ``num_seqs * pieces`` slots and every
    slot spans ``span = ceil(total_chunks / wave)`` chunks (units rounded up); thread ``b`` owns sequence ``b``, taking
    ``count_b = ceil(nc_b / span)`` slots (the length rule: its own count), each ``roundup(ceil(nc_b / count_b),
    unit_chunks)`` chunks.  When the per-sequence ceilings would overflow the wave the span is recomputed against
    ``wave - (num_seqs - 1)`` slots, which cannot overflow.  Slots are flat in sequence order: sequence ``b``'s first slot
    is ``main_rows[b] // heads_out`` and ``cu_pieces[slot + j] = cu[b] + min(j * span_b, len_b)``; ``main_rows`` /
    ``summary_rows`` are the per-sequence row bases of the main table (one row per filled slot and head, slot 0 of an
    empty sequence included) and of the summary table (sequences with two or more filled slots).  The entries from the
    batch end to ``cu_pieces[num_seqs * pieces]`` all hold the batch end, so the table stays monotone over the whole wave
    for the consumers that size the slots from its shape.  Deterministic, no host sync."""
    num_warps = cutlass.const_expr(n_threads // 32)
    sWarpMain = cutlass.Array(
        cutlass.Int32, num_warps, space=cutlass.AddressSpace.smem, alignment=16
    )
    sWarpSummary = cutlass.Array(
        cutlass.Int32, num_warps, space=cutlass.AddressSpace.smem, alignment=16
    )
    lane = tidx % cutlass.Int32(32)
    warp = tidx // cutlass.Int32(32)

    # ---- one round trip: this thread's first sequence and the batch total ----------------------------------------------
    first_read = tidx if tidx < num_seqs else num_seqs - cutlass.Int32(1)
    first_start = cutlass.Int32(mCu[first_read])
    first_end = cutlass.Int32(mCu[first_read + 1])
    total_tokens = cutlass.Int32(mCu[num_seqs]) - cutlass.Int32(mCu[0])

    # ---- the slot span: one wave shared by the whole batch, guarded against the per-sequence ceilings ----------------
    total_chunks = (
        total_tokens * cutlass.Int32(expand_num) + cutlass.Int32(b_t - 1)
    ) // cutlass.Int32(b_t)
    wave = num_seqs * cutlass.Int32(pieces)
    span_chunks = chunks_per_slot(unit_chunks, total_chunks, wave)
    if cutlass.const_expr(not length_rule):
        if num_seqs > cutlass.Int32(1):
            taken = cutlass.Int32(0)
            probe_start = cutlass.Int32(0)
            while probe_start < num_seqs:
                b = probe_start + tidx
                valid = b < num_seqs
                length = first_end - first_start
                if probe_start > cutlass.Int32(0):
                    b_read = b if valid else num_seqs - cutlass.Int32(1)
                    length = cutlass.Int32(mCu[b_read + 1]) - cutlass.Int32(mCu[b_read])
                count = piece_count(pieces, b_t, expand_num, length_rule, length, span_chunks)
                # ---- CTA sum of the counts: warp scan, lane 31 parks the warp total, every thread adds the words ----
                inclusive = count if valid else cutlass.Int32(0)
                for offset in [1, 2, 4, 8, 16]:
                    other = cutlass.Int32(
                        nvvm.shfl_sync(0xFFFFFFFF, inclusive, offset, 0, kind=nvvm.Shfl.UP)
                    )
                    inclusive = inclusive + (
                        other if lane >= cutlass.Int32(offset) else cutlass.Int32(0)
                    )
                if lane == cutlass.Int32(31):
                    sWarpMain[warp] = inclusive
                nvvm.barrier_cta_sync()
                total = cutlass.Int32(0)
                for i in cutlass.range_constexpr(num_warps):
                    total = total + sWarpMain[i]
                nvvm.barrier_cta_sync()
                taken = taken + total
                probe_start = probe_start + cutlass.Int32(n_threads)
            if taken > wave:
                span_chunks = chunks_per_slot(
                    unit_chunks, total_chunks, wave - num_seqs + cutlass.Int32(1)
                )

    running_main = cutlass.Int32(0)
    running_summary = cutlass.Int32(0)
    block_start = cutlass.Int32(0)
    while block_start < num_seqs:
        b = block_start + tidx
        valid = b < num_seqs
        start = first_start
        end = first_end
        if block_start > cutlass.Int32(0):
            b_read = b if valid else num_seqs - cutlass.Int32(1)
            start = cutlass.Int32(mCu[b_read])
            end = cutlass.Int32(mCu[b_read + 1])
        length = end - start
        span_tokens, last = piece_span(
            pieces, unit_chunks, b_t, expand_num, length_rule, length, span_chunks
        )
        rows = (last + cutlass.Int32(1)) * cutlass.Int32(heads_out)
        rows_main = rows if valid else cutlass.Int32(0)
        rows_summary = rows if (valid and last > cutlass.Int32(0)) else cutlass.Int32(0)

        # ---- row bases: warp scan, then the warp totals -----------------------------------
        incl_main = rows_main
        incl_summary = rows_summary
        for off in [1, 2, 4, 8, 16]:
            other_main = cutlass.Int32(
                nvvm.shfl_sync(0xFFFFFFFF, incl_main, off, 0, kind=nvvm.Shfl.UP)
            )
            other_summary = cutlass.Int32(
                nvvm.shfl_sync(0xFFFFFFFF, incl_summary, off, 0, kind=nvvm.Shfl.UP)
            )
            incl_main = incl_main + (
                other_main if lane >= cutlass.Int32(off) else cutlass.Int32(0)
            )
            incl_summary = incl_summary + (
                other_summary if lane >= cutlass.Int32(off) else cutlass.Int32(0)
            )
        if lane == cutlass.Int32(31):
            sWarpMain[warp] = incl_main
            sWarpSummary[warp] = incl_summary
        nvvm.barrier_cta_sync()
        total_main = cutlass.Int32(0)
        total_summary = cutlass.Int32(0)
        warp_base_main = cutlass.Int32(0)
        warp_base_summary = cutlass.Int32(0)
        for w in cutlass.range_constexpr(num_warps):
            count_main = sWarpMain[w]
            count_summary = sWarpSummary[w]
            total_main = total_main + count_main
            total_summary = total_summary + count_summary
            warp_base_main = warp_base_main + (
                count_main if cutlass.Int32(w) < warp else cutlass.Int32(0)
            )
            warp_base_summary = warp_base_summary + (
                count_summary if cutlass.Int32(w) < warp else cutlass.Int32(0)
            )

        # ---- per-sequence: the flat slots and the row bases ---------------------------------
        if valid:
            length64 = cutlass.Int64(length)
            row_base = running_main + warp_base_main + incl_main - rows_main
            slot_base = row_base // cutlass.Int32(heads_out)
            j = cutlass.Int32(0)
            while j <= last:
                offset = cutlass.Int64(j) * span_tokens
                offset = offset if offset < length64 else length64
                mCuPieces[slot_base + j] = start + offset.to(cutlass.Int32)
                j = j + cutlass.Int32(1)
            mMainRows[b] = row_base
            mSummaryRows[b] = running_summary + warp_base_summary + incl_summary - rows_summary
        running_main = running_main + total_main
        running_summary = running_summary + total_summary
        nvvm.barrier_cta_sync()
        block_start = block_start + cutlass.Int32(n_threads)
    # ---- the batch end, then padding to the wave: consumers that size the slots from cu_pieces' shape ----------------
    # (the T pass, the descriptor builders' callers) see the trailing slots as empty
    total_slots = running_main // cutlass.Int32(heads_out)
    batch_end = cutlass.Int32(mCu[num_seqs])
    tail = total_slots + tidx
    while tail <= num_seqs * cutlass.Int32(pieces):
        mCuPieces[tail] = batch_end
        tail = tail + cutlass.Int32(n_threads)
    if tidx == 0:
        mMainRows[num_seqs] = running_main
        mSummaryRows[num_seqs] = running_summary


# ---- state chain ----------------------------------------------------------------------------------


@cute.kernel
def frost_state_chain(
    heads_out: cutlass.Int32,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    pieces: cutlass.Int32,
    transpose: cutlass.Constexpr[bool],
    has_seed: cutlass.Constexpr[bool],
    has_tail: cutlass.Constexpr[bool],
    emit_summary: cutlass.Constexpr[bool],
    num_seqs: cutlass.Int32,
    mH: cute.Tensor | None,
    mM: cute.Tensor,
    mX: cute.Tensor | None,
    mSeed: cute.Tensor | None,
    mTail: cute.Tensor | None,
    mSummaryM: cute.Tensor | None,
    mMainRows: cute.Tensor | None,
    mSeedIndices: cute.Tensor | None,
):
    """CTA ``(seq, h, c)`` chains rows ``[c * rows, (c + 1) * rows)`` of the state and, under ``emit_summary``, the matching
    rows of the running M product.  Eight warps (two row groups times four k slices, lane ``l`` owning ``K / 32`` columns);
    each ``M_j`` is staged whole in SMEM by double-buffered ``cp.async`` and read as 8-wide k tiles, the four slice partials
    added to ``H_j`` in slice order by the row owner.  With ``mMainRows`` bound (the piece table's row bases) the walk
    covers the sequence's flat slots ``main_rows[seq] // HO .. main_rows[seq + 1] // HO``; without it the fixed grid
    ``seq * pieces + j``.  A one-slot walk copies the seed into ``X`` when there is no tail or summary and otherwise yields
    ``tail = seed @ M_0 + H_0``.  With ``emit_summary`` and neither seed nor tail only the product is walked."""
    K = cutlass.const_expr(dim_k)
    V = cutlass.const_expr(dim_v)
    HO = heads_out
    P = pieces
    product_rows = cutlass.const_expr(K * rows // V if emit_summary else 0)
    walk_state = cutlass.const_expr(has_seed or has_tail or not emit_summary)
    cols_per_lane = cutlass.const_expr(K // CHAIN_LANES)
    k_slice = cutlass.const_expr(K // CHAIN_K_WARPS)
    tiles = cutlass.const_expr(k_slice // CHAIN_K_TILE)
    m_words = cutlass.const_expr(K * K)
    walk_group_rows = cutlass.const_expr(rows // CHAIN_ROW_GROUPS)
    walk_own_rows = cutlass.const_expr(rows // CHAIN_WARPS)
    product_group_rows = cutlass.const_expr(product_rows // CHAIN_ROW_GROUPS)
    product_own_rows = cutlass.const_expr(product_rows // CHAIN_WARPS)
    part_rows = cutlass.const_expr(rows + product_rows)

    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    warp = tidx // cutlass.Int32(CHAIN_LANES)
    lane = tidx % cutlass.Int32(CHAIN_LANES)
    k_warp = warp % cutlass.Int32(CHAIN_K_WARPS)
    row_group = warp // cutlass.Int32(CHAIN_K_WARPS)
    col0 = lane * cutlass.Int32(cols_per_lane)
    bid = cute.arch.block_idx()
    seq = cutlass.Int32(bid[0])
    h = cutlass.Int32(bid[1])
    seed_seq = seq
    if cutlass.const_expr(mSeedIndices is not None):
        seed_seq = cutlass.Int32(mSeedIndices[seq])
    row0 = cutlass.Int32(bid[2]) * cutlass.Int32(rows)
    product_row0 = cutlass.Int32(bid[2]) * cutlass.Int32(product_rows)

    # ---- walk bounds --------------------------------------------------------------------------
    seq_base = seq * cutlass.Int32(P)
    count = cutlass.Int32(P)
    if cutlass.const_expr(mMainRows is not None):
        row_base = cutlass.Int32(mMainRows[seq])
        seq_base = row_base // cutlass.Int32(HO)
        filled = (cutlass.Int32(mMainRows[seq + 1]) - row_base) // cutlass.Int32(HO)
        count = filled if filled > cutlass.Int32(0) else cutlass.Int32(1)

    sWalk = cutlass.Array(cutlass.Float32, rows * K, space=cutlass.AddressSpace.smem, alignment=16)
    if cutlass.const_expr(emit_summary):
        sProduct = cutlass.Array(
            cutlass.Float32, product_rows * K, space=cutlass.AddressSpace.smem, alignment=16
        )
    sPart = cutlass.Array(
        cutlass.Float32,
        CHAIN_K_WARPS * part_rows * K,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    sM = cutlass.Array(
        cutlass.Float32, CHAIN_M_BUFFERS * m_words, space=cutlass.AddressSpace.smem, alignment=128
    )
    acc = cutlass.Array(cutlass.Float32, walk_group_rows * cols_per_lane)
    acc_product = cutlass.Array(cutlass.Float32, max(1, product_group_rows) * cols_per_lane)
    m_tile = cutlass.Array(cutlass.Float32, CHAIN_K_TILE * cols_per_lane)
    cur = cutlass.Array(cutlass.Float32, walk_own_rows * cols_per_lane)
    cur_product = cutlass.Array(cutlass.Float32, max(1, product_own_rows) * cols_per_lane)
    piece_h = cutlass.Array(cutlass.Float32, walk_own_rows * cols_per_lane)

    one_piece = seq < cutlass.Int32(0)
    if cutlass.const_expr(mMainRows is not None and not (has_tail or emit_summary)):
        one_piece = count == cutlass.Int32(1)
    if one_piece:
        if cutlass.const_expr(walk_state):
            for rr in cutlass.range_constexpr(walk_own_rows):
                seed_row = (
                    row_group * cutlass.Int32(walk_group_rows)
                    + k_warp * cutlass.Int32(walk_own_rows)
                    + cutlass.Int32(rr)
                )
                if cutlass.const_expr(has_seed):
                    seed_state_row = row0 + seed_row
                    seed_offset = (
                        cutlass.Int64(seed_seq) * cutlass.Int64(mSeed.stride[0])
                        + cutlass.Int64(h) * cutlass.Int64(mSeed.stride[1])
                        + cutlass.Int64(seed_state_row) * cutlass.Int64(mSeed.stride[2])
                        + cutlass.Int64(col0)
                    )
                    seed_iter = mSeed.iterator
                    seed_addr = seed_iter.toint() + seed_offset * cutlass.Int64(
                        mSeed.element_type.width // 8
                    )
                    if cutlass.const_expr(mSeed.element_type == cutlass.Float32):
                        if cutlass.const_expr(cols_per_lane == 2):
                            seed_vals = list(ld_global_v2(seed_addr, cutlass.Float32))
                        else:
                            seed_vals = [
                                x
                                for q in range(cols_per_lane // 4)
                                for x in ld_global_v4(
                                    seed_addr + cutlass.Int64(16 * q), cutlass.Float32
                                )
                            ]
                    else:
                        if cutlass.const_expr(cols_per_lane == 2):
                            seed_words = [ld_global(seed_addr, cutlass.Int32)]
                        elif cutlass.const_expr(cols_per_lane == 4):
                            seed_words = list(ld_global_v2(seed_addr, cutlass.Int32))
                        else:
                            seed_words = [
                                w
                                for q in range(cols_per_lane // 8)
                                for w in ld_global_v4(
                                    seed_addr + cutlass.Int64(16 * q), cutlass.Int32
                                )
                            ]
                        seed_vals = [
                            x
                            for word in seed_words
                            for x in f16x2_to_f32(word, dtype=mSeed.element_type)
                        ]
                else:
                    seed_vals = [cutlass.Float32(0.0)] * cols_per_lane
                for c in cutlass.range_constexpr(cols_per_lane):
                    cur[rr * cols_per_lane + c] = seed_vals[c]
                seed_x_row = row0 + seed_row
                seed_x_offset = (
                    cutlass.Int64(seq_base) * cutlass.Int64(mX.stride[0])
                    + cutlass.Int64(h) * cutlass.Int64(mX.stride[1])
                    + cutlass.Int64(seed_x_row) * cutlass.Int64(mX.stride[2])
                    + cutlass.Int64(col0)
                )
                seed_x_iter = mX.iterator
                seed_x_addr = seed_x_iter.toint() + seed_x_offset * cutlass.Int64(
                    mX.element_type.width // 8
                )
                seed_x_vals = [cur[rr * cols_per_lane + c] for c in range(cols_per_lane)]
                if cutlass.const_expr(cols_per_lane == 2):
                    st_global_v2(seed_x_addr, seed_x_vals, cutlass.Float32)
                else:
                    for q in cutlass.range_constexpr(cols_per_lane // 4):
                        st_global_v4(
                            seed_x_addr + cutlass.Int64(16 * q),
                            seed_x_vals[4 * q : 4 * q + 4],
                            cutlass.Float32,
                        )
    else:
        # ---- seeds --------------------------------------------------------------------------------
        for rr in cutlass.range_constexpr(walk_own_rows):
            seed_row = (
                row_group * cutlass.Int32(walk_group_rows)
                + k_warp * cutlass.Int32(walk_own_rows)
                + cutlass.Int32(rr)
            )
            if cutlass.const_expr(has_seed):
                seed_state_row = row0 + seed_row
                seed_offset = (
                    cutlass.Int64(seed_seq) * cutlass.Int64(mSeed.stride[0])
                    + cutlass.Int64(h) * cutlass.Int64(mSeed.stride[1])
                    + cutlass.Int64(seed_state_row) * cutlass.Int64(mSeed.stride[2])
                    + cutlass.Int64(col0)
                )
                seed_iter = mSeed.iterator
                seed_addr = seed_iter.toint() + seed_offset * cutlass.Int64(
                    mSeed.element_type.width // 8
                )
                if cutlass.const_expr(mSeed.element_type == cutlass.Float32):
                    if cutlass.const_expr(cols_per_lane == 2):
                        seed_vals = list(ld_global_v2(seed_addr, cutlass.Float32))
                    else:
                        seed_vals = [
                            x
                            for q in range(cols_per_lane // 4)
                            for x in ld_global_v4(
                                seed_addr + cutlass.Int64(16 * q), cutlass.Float32
                            )
                        ]
                else:
                    if cutlass.const_expr(cols_per_lane == 2):
                        seed_words = [ld_global(seed_addr, cutlass.Int32)]
                    elif cutlass.const_expr(cols_per_lane == 4):
                        seed_words = list(ld_global_v2(seed_addr, cutlass.Int32))
                    else:
                        seed_words = [
                            w
                            for q in range(cols_per_lane // 8)
                            for w in ld_global_v4(seed_addr + cutlass.Int64(16 * q), cutlass.Int32)
                        ]
                    seed_vals = [
                        x
                        for word in seed_words
                        for x in f16x2_to_f32(word, dtype=mSeed.element_type)
                    ]
            else:
                seed_vals = [cutlass.Float32(0.0)] * cols_per_lane
            for c in cutlass.range_constexpr(cols_per_lane):
                cur[rr * cols_per_lane + c] = seed_vals[c]
            seed_walk_ptr = sWalk.data_ptr(seed_row * K + col0)
            seed_walk_vals = [cur[rr * cols_per_lane + c] for c in range(cols_per_lane)]
            if cutlass.const_expr(cols_per_lane == 2):
                st_shared_v2(seed_walk_ptr, seed_walk_vals, cutlass.Float32)
            else:
                for q in cutlass.range_constexpr(cols_per_lane // 4):
                    st_shared_v4(
                        seed_walk_ptr + 4 * q, seed_walk_vals[4 * q : 4 * q + 4], cutlass.Float32
                    )
        if cutlass.const_expr(emit_summary):
            for rr in cutlass.range_constexpr(product_own_rows):
                seed_row = (
                    row_group * cutlass.Int32(product_group_rows)
                    + k_warp * cutlass.Int32(product_own_rows)
                    + cutlass.Int32(rr)
                )
                for c in cutlass.range_constexpr(cols_per_lane):
                    cur_product[rr * cols_per_lane + c] = (
                        cutlass.Float32(1.0)
                        if product_row0 + seed_row == col0 + cutlass.Int32(c)
                        else cutlass.Float32(0.0)
                    )
                seed_product_ptr = sProduct.data_ptr(seed_row * K + col0)
                seed_product_vals = [
                    cur_product[rr * cols_per_lane + c] for c in range(cols_per_lane)
                ]
                if cutlass.const_expr(cols_per_lane == 2):
                    st_shared_v2(seed_product_ptr, seed_product_vals, cutlass.Float32)
                else:
                    for q in cutlass.range_constexpr(cols_per_lane // 4):
                        st_shared_v4(
                            seed_product_ptr + 4 * q,
                            seed_product_vals[4 * q : 4 * q + 4],
                            cutlass.Float32,
                        )
        first_step = (
            cutlass.Int32(0) if cutlass.const_expr(not transpose) else count - cutlass.Int32(1)
        )
        first_buf_words = cutlass.Int32(0)
        first_slot = seq_base + first_step
        for q in cutlass.range_constexpr(K * K // 4 // CHAIN_THREADS):
            first_chunk_id = tidx + cutlass.Int32(q * CHAIN_THREADS)
            first_m_row = first_chunk_id // cutlass.Int32(K // 4)
            first_m_chunk = first_chunk_id % cutlass.Int32(K // 4)
            nvvm.cp_async_shared_global(
                sM.data_ptr(
                    first_buf_words
                    + (
                        first_m_row * K
                        + (first_m_chunk ^ ((first_m_row // cutlass.Int32(4)) % cutlass.Int32(8)))
                        * cutlass.Int32(4)
                    )
                ),
                mM.iterator
                + (
                    (cutlass.Int64(first_slot) * cutlass.Int64(HO) + cutlass.Int64(h))
                    * cutlass.Int64(K)
                    + cutlass.Int64(first_m_row)
                )
                * cutlass.Int64(K)
                + cutlass.Int64(first_m_chunk) * cutlass.Int64(4),
                16,
                nvvm.LoadCacheModifier.CG,
            )
        cp_async_commit()

        # ---- walk ---------------------------------------------------------------------------------
        i = cutlass.Int32(0)
        while i < count:
            step = i if cutlass.const_expr(not transpose) else count - cutlass.Int32(1) - i
            slot = seq_base + step
            if cutlass.const_expr(walk_state):
                for rr in cutlass.range_constexpr(walk_own_rows):
                    row = (
                        row0
                        + row_group * cutlass.Int32(walk_group_rows)
                        + k_warp * cutlass.Int32(walk_own_rows)
                        + cutlass.Int32(rr)
                    )
                    x_offset = (
                        cutlass.Int64(slot) * cutlass.Int64(mX.stride[0])
                        + cutlass.Int64(h) * cutlass.Int64(mX.stride[1])
                        + cutlass.Int64(row) * cutlass.Int64(mX.stride[2])
                        + cutlass.Int64(col0)
                    )
                    x_iter = mX.iterator
                    x_addr = x_iter.toint() + x_offset * cutlass.Int64(mX.element_type.width // 8)
                    x_vals = [cur[rr * cols_per_lane + c] for c in range(cols_per_lane)]
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_global_v2(x_addr, x_vals, cutlass.Float32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 4):
                            st_global_v4(
                                x_addr + cutlass.Int64(16 * q),
                                x_vals[4 * q : 4 * q + 4],
                                cutlass.Float32,
                            )
                    h_offset = (
                        cutlass.Int64(slot) * cutlass.Int64(mH.stride[0])
                        + cutlass.Int64(h) * cutlass.Int64(mH.stride[1])
                        + cutlass.Int64(row) * cutlass.Int64(mH.stride[2])
                        + cutlass.Int64(col0)
                    )
                    h_iter = mH.iterator
                    h_addr = h_iter.toint() + h_offset * cutlass.Int64(mH.element_type.width // 8)
                    if cutlass.const_expr(cols_per_lane == 2):
                        h_vals = list(ld_global_v2(h_addr, cutlass.Float32))
                    else:
                        h_vals = [
                            x
                            for q in range(cols_per_lane // 4)
                            for x in ld_global_v4(h_addr + cutlass.Int64(16 * q), cutlass.Float32)
                        ]
                    for c in cutlass.range_constexpr(cols_per_lane):
                        piece_h[rr * cols_per_lane + c] = h_vals[c]
            if i + cutlass.Int32(1) < count:
                next_step = (
                    step + cutlass.Int32(1)
                    if cutlass.const_expr(not transpose)
                    else step - cutlass.Int32(1)
                )
                next_buf_words = (
                    (i + cutlass.Int32(1)) % cutlass.Int32(CHAIN_M_BUFFERS)
                ) * cutlass.Int32(m_words)
                next_slot = seq_base + next_step
                for q in cutlass.range_constexpr(K * K // 4 // CHAIN_THREADS):
                    next_chunk_id = tidx + cutlass.Int32(q * CHAIN_THREADS)
                    next_m_row = next_chunk_id // cutlass.Int32(K // 4)
                    next_m_chunk = next_chunk_id % cutlass.Int32(K // 4)
                    nvvm.cp_async_shared_global(
                        sM.data_ptr(
                            next_buf_words
                            + (
                                next_m_row * K
                                + (
                                    next_m_chunk
                                    ^ ((next_m_row // cutlass.Int32(4)) % cutlass.Int32(8))
                                )
                                * cutlass.Int32(4)
                            )
                        ),
                        mM.iterator
                        + (
                            (cutlass.Int64(next_slot) * cutlass.Int64(HO) + cutlass.Int64(h))
                            * cutlass.Int64(K)
                            + cutlass.Int64(next_m_row)
                        )
                        * cutlass.Int64(K)
                        + cutlass.Int64(next_m_chunk) * cutlass.Int64(4),
                        16,
                        nvvm.LoadCacheModifier.CG,
                    )
                cp_async_commit()
                cp_async_wait(1)
            else:
                cp_async_wait(0)
            nvvm.barrier_cta_sync()
            buf_words = (i % cutlass.Int32(CHAIN_M_BUFFERS)) * cutlass.Int32(m_words)

            # ---- one sweep over the k tiles feeds both accumulations -------------------------
            if cutlass.const_expr(walk_state):
                for j in cutlass.range_constexpr(walk_group_rows * cols_per_lane):
                    acc[j] = cutlass.Float32(0.0)
            if cutlass.const_expr(emit_summary):
                for j in cutlass.range_constexpr(product_group_rows * cols_per_lane):
                    acc_product[j] = cutlass.Float32(0.0)
            for tile in cutlass.range_constexpr(tiles):
                k0 = k_warp * cutlass.Int32(k_slice) + cutlass.Int32(tile * CHAIN_K_TILE)
                if cutlass.const_expr(transpose):
                    for c in cutlass.range_constexpr(cols_per_lane):
                        m_row = col0 + cutlass.Int32(c)
                        m_vals = []
                        for v in cutlass.range_constexpr(CHAIN_K_TILE // 4):
                            m_col = k0 + cutlass.Int32(4 * v)
                            m_chunk = m_col // cutlass.Int32(4)
                            m_ptr = sM.data_ptr(
                                buf_words
                                + (
                                    m_row * K
                                    + (m_chunk ^ ((m_row // cutlass.Int32(4)) % cutlass.Int32(8)))
                                    * cutlass.Int32(4)
                                )
                                + m_col % cutlass.Int32(4)
                            )
                            m_vals += list(ld_shared_v4(m_ptr, cutlass.Float32))
                        for kk in cutlass.range_constexpr(CHAIN_K_TILE):
                            m_tile[kk * cols_per_lane + c] = m_vals[kk]
                else:
                    for kk in cutlass.range_constexpr(CHAIN_K_TILE):
                        m_row = k0 + cutlass.Int32(kk)
                        m_vals = []
                        for v in cutlass.range_constexpr(cols_per_lane // min(4, cols_per_lane)):
                            m_col = col0 + cutlass.Int32(min(4, cols_per_lane) * v)
                            m_chunk = m_col // cutlass.Int32(4)
                            m_ptr = sM.data_ptr(
                                buf_words
                                + (
                                    m_row * K
                                    + (m_chunk ^ ((m_row // cutlass.Int32(4)) % cutlass.Int32(8)))
                                    * cutlass.Int32(4)
                                )
                                + m_col % cutlass.Int32(4)
                            )
                            if cutlass.const_expr(cols_per_lane >= 4):
                                m_vals += list(ld_shared_v4(m_ptr, cutlass.Float32))
                            else:
                                m_vals += list(ld_shared_v2(m_ptr, cutlass.Float32))
                        for c in cutlass.range_constexpr(cols_per_lane):
                            m_tile[kk * cols_per_lane + c] = m_vals[c]
                if cutlass.const_expr(walk_state):
                    for r in cutlass.range_constexpr(walk_group_rows):
                        op_row = row_group * cutlass.Int32(walk_group_rows) + cutlass.Int32(r)
                        for kq in cutlass.range_constexpr(CHAIN_K_TILE // 4):
                            x4 = ld_shared_v4(
                                sWalk.data_ptr(op_row * K + k0 + cutlass.Int32(4 * kq)),
                                cutlass.Float32,
                            )
                            for e in cutlass.range_constexpr(4):
                                for c in cutlass.range_constexpr(cols_per_lane):
                                    acc[r * cols_per_lane + c] = cute.math.fma(
                                        x4[e],
                                        m_tile[(4 * kq + e) * cols_per_lane + c],
                                        acc[r * cols_per_lane + c],
                                    )
                if cutlass.const_expr(emit_summary):
                    for r in cutlass.range_constexpr(product_group_rows):
                        op_row = row_group * cutlass.Int32(product_group_rows) + cutlass.Int32(r)
                        for kq in cutlass.range_constexpr(CHAIN_K_TILE // 4):
                            x4 = ld_shared_v4(
                                sProduct.data_ptr(op_row * K + k0 + cutlass.Int32(4 * kq)),
                                cutlass.Float32,
                            )
                            for e in cutlass.range_constexpr(4):
                                for c in cutlass.range_constexpr(cols_per_lane):
                                    acc_product[r * cols_per_lane + c] = cute.math.fma(
                                        x4[e],
                                        m_tile[(4 * kq + e) * cols_per_lane + c],
                                        acc_product[r * cols_per_lane + c],
                                    )

            # ---- slice partials ------------------------------------------------------------------
            if cutlass.const_expr(walk_state):
                for r in cutlass.range_constexpr(walk_group_rows):
                    op_row = row_group * cutlass.Int32(walk_group_rows) + cutlass.Int32(r)
                    part_ptr = sPart.data_ptr(
                        (k_warp * cutlass.Int32(part_rows) + op_row) * K + col0
                    )
                    part_vals = [acc[r * cols_per_lane + c] for c in range(cols_per_lane)]
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_shared_v2(part_ptr, part_vals, cutlass.Float32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 4):
                            st_shared_v4(
                                part_ptr + 4 * q, part_vals[4 * q : 4 * q + 4], cutlass.Float32
                            )
            if cutlass.const_expr(emit_summary):
                for r in cutlass.range_constexpr(product_group_rows):
                    op_row = row_group * cutlass.Int32(product_group_rows) + cutlass.Int32(r)
                    part_ptr = sPart.data_ptr(
                        (k_warp * cutlass.Int32(part_rows) + cutlass.Int32(rows) + op_row) * K
                        + col0
                    )
                    part_vals = [acc_product[r * cols_per_lane + c] for c in range(cols_per_lane)]
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_shared_v2(part_ptr, part_vals, cutlass.Float32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 4):
                            st_shared_v4(
                                part_ptr + 4 * q, part_vals[4 * q : 4 * q + 4], cutlass.Float32
                            )
            nvvm.barrier_cta_sync()

            # ---- owned rows: H_j plus the four slice partials in slice order ---------------------
            if cutlass.const_expr(walk_state):
                for rr in cutlass.range_constexpr(walk_own_rows):
                    op_row = (
                        row_group * cutlass.Int32(walk_group_rows)
                        + k_warp * cutlass.Int32(walk_own_rows)
                        + cutlass.Int32(rr)
                    )
                    total = [piece_h[rr * cols_per_lane + c] for c in range(cols_per_lane)]
                    for w2 in cutlass.range_constexpr(CHAIN_K_WARPS):
                        part_ptr = sPart.data_ptr(
                            (cutlass.Int32(w2 * part_rows) + op_row) * K + col0
                        )
                        if cutlass.const_expr(cols_per_lane == 2):
                            part = list(ld_shared_v2(part_ptr, cutlass.Float32))
                        else:
                            part = [
                                x
                                for q in range(cols_per_lane // 4)
                                for x in ld_shared_v4(part_ptr + 4 * q, cutlass.Float32)
                            ]
                        for c in cutlass.range_constexpr(cols_per_lane):
                            total[c] = total[c] + part[c]
                    for c in cutlass.range_constexpr(cols_per_lane):
                        cur[rr * cols_per_lane + c] = total[c]
                    walk_ptr = sWalk.data_ptr(op_row * K + col0)
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_shared_v2(walk_ptr, total, cutlass.Float32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 4):
                            st_shared_v4(
                                walk_ptr + 4 * q, total[4 * q : 4 * q + 4], cutlass.Float32
                            )
            if cutlass.const_expr(emit_summary):
                for rr in cutlass.range_constexpr(product_own_rows):
                    op_row = (
                        row_group * cutlass.Int32(product_group_rows)
                        + k_warp * cutlass.Int32(product_own_rows)
                        + cutlass.Int32(rr)
                    )
                    total = [cutlass.Float32(0.0) for c in range(cols_per_lane)]
                    for w2 in cutlass.range_constexpr(CHAIN_K_WARPS):
                        part_ptr = sPart.data_ptr(
                            (cutlass.Int32(w2 * part_rows) + cutlass.Int32(rows) + op_row) * K
                            + col0
                        )
                        if cutlass.const_expr(cols_per_lane == 2):
                            part = list(ld_shared_v2(part_ptr, cutlass.Float32))
                        else:
                            part = [
                                x
                                for q in range(cols_per_lane // 4)
                                for x in ld_shared_v4(part_ptr + 4 * q, cutlass.Float32)
                            ]
                        for c in cutlass.range_constexpr(cols_per_lane):
                            total[c] = total[c] + part[c]
                    for c in cutlass.range_constexpr(cols_per_lane):
                        cur_product[rr * cols_per_lane + c] = total[c]
                    product_ptr = sProduct.data_ptr(op_row * K + col0)
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_shared_v2(product_ptr, total, cutlass.Float32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 4):
                            st_shared_v4(
                                product_ptr + 4 * q, total[4 * q : 4 * q + 4], cutlass.Float32
                            )
            nvvm.barrier_cta_sync()
            i = i + cutlass.Int32(1)

        # ---- epilogue -----------------------------------------------------------------------------
        if cutlass.const_expr(has_tail):
            for rr in cutlass.range_constexpr(walk_own_rows):
                out_row = (
                    row0
                    + row_group * cutlass.Int32(walk_group_rows)
                    + k_warp * cutlass.Int32(walk_own_rows)
                    + cutlass.Int32(rr)
                )
                tail_offset = (
                    cutlass.Int64(seq) * cutlass.Int64(mTail.stride[0])
                    + cutlass.Int64(h) * cutlass.Int64(mTail.stride[1])
                    + cutlass.Int64(out_row) * cutlass.Int64(mTail.stride[2])
                    + cutlass.Int64(col0)
                )
                tail_iter = mTail.iterator
                tail_addr = tail_iter.toint() + tail_offset * cutlass.Int64(
                    mTail.element_type.width // 8
                )
                tail_vals = [cur[rr * cols_per_lane + c] for c in range(cols_per_lane)]
                if cutlass.const_expr(mTail.element_type == cutlass.Float32):
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_global_v2(tail_addr, tail_vals, cutlass.Float32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 4):
                            st_global_v4(
                                tail_addr + cutlass.Int64(16 * q),
                                tail_vals[4 * q : 4 * q + 4],
                                cutlass.Float32,
                            )
                else:
                    tail_words = [
                        fp32_to_fp16(
                            tail_vals[2 * p], tail_vals[2 * p + 1], dtype=mTail.element_type
                        )
                        for p in range(cols_per_lane // 2)
                    ]
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_global(tail_addr, tail_words[0], cutlass.Int32)
                    elif cutlass.const_expr(cols_per_lane == 4):
                        st_global_v2(tail_addr, tail_words, cutlass.Int32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 8):
                            st_global_v4(
                                tail_addr + cutlass.Int64(16 * q),
                                tail_words[4 * q : 4 * q + 4],
                                cutlass.Int32,
                            )
        if cutlass.const_expr(emit_summary):
            for rr in cutlass.range_constexpr(product_own_rows):
                out_row = (
                    product_row0
                    + row_group * cutlass.Int32(product_group_rows)
                    + k_warp * cutlass.Int32(product_own_rows)
                    + cutlass.Int32(rr)
                )
                summary_m_addr = mSummaryM.iterator.toint() + (
                    (
                        (cutlass.Int64(seq) * cutlass.Int64(HO) + cutlass.Int64(h))
                        * cutlass.Int64(K)
                        + cutlass.Int64(out_row)
                    )
                    * cutlass.Int64(K)
                    + cutlass.Int64(col0)
                ) * cutlass.Int64(mSummaryM.element_type.width // 8)
                summary_vals = [cur_product[rr * cols_per_lane + c] for c in range(cols_per_lane)]
                if cutlass.const_expr(mSummaryM.element_type == cutlass.Float32):
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_global_v2(summary_m_addr, summary_vals, cutlass.Float32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 4):
                            st_global_v4(
                                summary_m_addr + cutlass.Int64(16 * q),
                                summary_vals[4 * q : 4 * q + 4],
                                cutlass.Float32,
                            )
                else:
                    summary_words = [
                        fp32_to_fp16(
                            summary_vals[2 * p],
                            summary_vals[2 * p + 1],
                            dtype=mSummaryM.element_type,
                        )
                        for p in range(cols_per_lane // 2)
                    ]
                    if cutlass.const_expr(cols_per_lane == 2):
                        st_global(summary_m_addr, summary_words[0], cutlass.Int32)
                    elif cutlass.const_expr(cols_per_lane == 4):
                        st_global_v2(summary_m_addr, summary_words, cutlass.Int32)
                    else:
                        for q in cutlass.range_constexpr(cols_per_lane // 8):
                            st_global_v4(
                                summary_m_addr + cutlass.Int64(16 * q),
                                summary_words[4 * q : 4 * q + 4],
                                cutlass.Int32,
                            )
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def launch_state_chain(
    heads_out: cutlass.Int32,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    pieces: cutlass.Int32,
    transpose: cutlass.Constexpr[bool],
    has_seed: cutlass.Constexpr[bool],
    has_tail: cutlass.Constexpr[bool],
    emit_summary: cutlass.Constexpr[bool],
    num_seqs: cutlass.Int32,
    mH: cute.Tensor | None,
    mM: cute.Tensor,
    mX: cute.Tensor | None,
    mSeed: cute.Tensor | None,
    mTail: cute.Tensor | None,
    mSummaryM: cute.Tensor | None,
    mMainRows: cute.Tensor | None,
    mSeedIndices: cute.Tensor | None,
    stream: cuda.CUstream,
) -> None:
    slices = cutlass.const_expr(dim_v // rows)
    frost_state_chain(
        heads_out,
        dim_v,
        dim_k,
        rows,
        pieces,
        transpose,
        has_seed,
        has_tail,
        emit_summary,
        num_seqs,
        mH,
        mM,
        mX,
        mSeed,
        mTail,
        mSummaryM,
        mMainRows,
        mSeedIndices,
    ).launch(
        grid=(num_seqs, heads_out, slices),
        block=(CHAIN_THREADS, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


def chain_rows_per_cta(dim_v, dim_k, num_seqs, heads_out, num_sm):
    """Rows of the state per chain CTA: ``dim_v // 16`` (16 CTAs per (sequence, head)) when that grid still fits
    the SMs and every warp keeps a whole row of the state and of the M product, else ``dim_v // 8``."""
    rows = dim_v // 16
    if (
        num_seqs * heads_out * 16 <= num_sm
        and rows >= CHAIN_WARPS
        and dim_k * rows // dim_v >= CHAIN_WARPS
    ):
        return rows
    return dim_v // 8


frost_state_chain.set_name_prefix("cudnn", remove_cutlass_symbol=False)
