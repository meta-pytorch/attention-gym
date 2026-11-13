# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/common/piece_chain.py
# and the table phase of kernel/{gdn,kda}_chain_prologue_f16.py), rewritten by Attention Gym in
# 2026 on CuTeDSL 4.6.2 APIs (cute.arch shuffles and barriers, SmemAllocator tensors, the raw
# NVVM cp.async op, inline PTX vector loads and stores) in place of cutlass.experimental. The
# piece tables, work tables and fp32 state chain are the upstream ones bit for bit; the per-piece
# TMA-descriptor arrays the upstream chain prologues also built are gone.
#
# CuTeDSL traces if-expressions as device selects, which min/max builtins do not reproduce on
# dynamic values, and a nested `if` guards its loads, which a merged `a and b` condition would not.
# ruff: noqa: FURB136, SIM102

"""Exact piece chain of the warpdrive GDN/KDA kernels: the piece rule, the piece table and its work
tables, and the fp32 state chain that seeds every piece with its exact incoming state.

When ``B * HO`` leaves SMs idle the batch is cut into one wave of ``num_seqs * pieces`` slots
that run as independent sequences, every slot spanning the same number of chunks, so a sequence
fills slots in proportion to its length. Their summaries ``H_j`` (state from a zero seed) and
``M_j`` (transition, stored domain ``M_buf = M^T``) chain in fp32::

    X_{j+1} = X_j @ M_j + H_j                       (forward)
    X_j     = X_{j+1} @ M_j^T + G_j                 (transpose: the backward, H playing G)

Layouts, fp32: ``H``, ``X`` ``[num_seqs * pieces, HO, V, K]``, ``M`` ``[num_seqs * pieces, HO,
K, K]`` (contiguous), ``seed`` ``[num_seqs, HO, V, K]`` (fp32 or bf16). Slots are flat in
sequence order: sequence ``b`` owns slots ``main_rows[b] // HO .. main_rows[b + 1] // HO``.

Piece table (``piece_table_layout``): ``main_rows`` / ``summary_rows`` int32 ``[num_seqs + 1]``
(each sequence's first row in the main and summary work tables; entry ``num_seqs`` is the
table's row count) and ``cu_pieces`` int32 ``[num_seqs * pieces + 1]`` in real tokens, the flat
slot starts followed by the batch end, padded with the batch end to the full wave. The main
table has one row per filled slot and head (slot 0 of an empty sequence included), the summary
table one per filled slot of the sequences with two or more; the rows of a sequence's last
filled slot carry ``final_dst`` and those of its first ``dstate_dst``.
"""

from typing import NamedTuple

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass._mlir.dialects import nvvm as mlir_nvvm

from attn_gym.warpdrive.cache import (
    compile_tvm_ffi,
    jit_cache,
    make_fake_compact_tensor,
    make_fake_strided_tensor,
)
from attn_gym.warpdrive.compat import SmemAllocator

from .ptx import f16x2_to_f32, ld_global_v, ld_shared_f32, st_global_v, st_shared_f32
from .work_tables import (
    OPT_LEVEL,
    ORDER_THREADS,
    WORK_ITEM_FIELDS,
    ceil_div,
    emit_chunk_rows,
    order_body,
    order_storage,
)

CHAIN_MIN_PIECES = 3
CHAIN_MIN_PIECES_REVERSE = 2
CHAIN_MAX_PIECES = 16
CHAIN_MIN_CHUNKS_PER_PIECE = 4

PIECE_TABLE_ALIGN = 256
CHAIN_K_WARPS = 4
CHAIN_ROW_GROUPS = 2
CHAIN_WARPS = CHAIN_K_WARPS * CHAIN_ROW_GROUPS
CHAIN_LANES = 32
CHAIN_THREADS = CHAIN_WARPS * CHAIN_LANES
CHAIN_K_TILE = 8
CHAIN_M_BUFFERS = 2

# ---- piece rule ------------------------------------------------------------------------------


def choose_pieces(*, num_seqs, heads_out, num_sm, total_tokens, b_t, reverse=False) -> int:
    """Slot budget per sequence of one plan, 0 when the plan does not chain: the one-wave
    geometry ``num_sm // (num_seqs * heads_out)`` capped at ``CHAIN_MAX_PIECES`` and at
    ``CHAIN_MIN_CHUNKS_PER_PIECE`` chunks per piece, chaining from ``CHAIN_MIN_PIECES``
    (``CHAIN_MIN_PIECES_REVERSE`` for ``reverse``). The piece table gives every slot the same
    span, ``ceil(total_chunks / (num_seqs * pieces))`` chunks."""
    num_seqs = max(1, int(num_seqs))
    tiles = num_seqs * max(1, int(heads_out))
    total_chunks = ceil_div(int(total_tokens), int(b_t))
    pieces = min(
        int(num_sm) // tiles,
        CHAIN_MAX_PIECES,
        total_chunks // (num_seqs * CHAIN_MIN_CHUNKS_PER_PIECE),
    )
    return pieces if pieces >= (CHAIN_MIN_PIECES_REVERSE if reverse else CHAIN_MIN_PIECES) else 0


class PieceTableLayout(NamedTuple):
    """Byte offsets of the piece-table workspace regions, the summary work-item capacity and
    the total size."""

    main_rows: int
    summary_rows: int
    cu_pieces: int
    main_count: int
    summary_count: int
    item_rows: int
    nbytes: int


def piece_table_layout(num_seqs: int, pieces: int, heads_out: int) -> PieceTableLayout:
    """``main_rows`` / ``summary_rows`` int32 ``[num_seqs + 1]`` (entry ``num_seqs`` is the
    ``main_count`` / ``summary_count`` cell) and ``cu_pieces`` int32 ``[num_seqs * pieces +
    1]``, each 16-byte aligned; 256-byte aligned total."""
    num_seqs = int(num_seqs)
    rows_bytes = ceil_div(4 * (num_seqs + 1), 16) * 16
    cu_pieces = 2 * rows_bytes
    cu_bytes = ceil_div(4 * (num_seqs * int(pieces) + 1), 16) * 16
    nbytes = ceil_div(cu_pieces + cu_bytes, PIECE_TABLE_ALIGN) * PIECE_TABLE_ALIGN
    return PieceTableLayout(
        0,
        rows_bytes,
        cu_pieces,
        4 * num_seqs,
        rows_bytes + 4 * num_seqs,
        num_seqs * int(pieces) * int(heads_out),
        nbytes,
    )


def chain_rows_per_cta(dim_v, dim_k, num_seqs, heads_out, num_sm):
    """State rows per chain CTA: ``dim_v // 16`` (16 CTAs per (sequence, head)) when that grid
    fits the SMs and every warp keeps a whole row of the state and of upstream's M product
    (``dim_k * rows // dim_v`` rows, kept so the geometry matches), else ``dim_v // 8``."""
    rows = dim_v // 16
    if (
        num_seqs * heads_out * 16 <= num_sm
        and rows >= CHAIN_WARPS
        and (dim_k * rows // dim_v >= CHAIN_WARPS)
    ):
        return rows
    return dim_v // 8


# ---- piece table -----------------------------------------------------------------------------


@cute.jit
def chunks_per_slot(total_chunks, slots):
    """``ceil(total_chunks / slots)``, at least one."""
    span = (total_chunks + slots - Int32(1)) // slots
    return span if span > Int32(1) else Int32(1)


@cute.jit
def piece_count(b_t: cutlass.Constexpr[int], length, span_chunks):
    """Slots a sequence of ``length`` tokens takes: ``ceil(chunks / span_chunks)``, at least
    one."""
    chunks = (length + Int32(b_t - 1)) // Int32(b_t)
    count = (chunks + span_chunks - Int32(1)) // span_chunks
    return count if count > Int32(1) else Int32(1)


@cute.jit
def piece_span(b_t: cutlass.Constexpr[int], length, span_chunks):
    """``(span_tokens, last)`` of a sequence of ``length`` tokens taking :func:`piece_count`
    slots: the piece span in tokens (Int64; its chunks spread evenly over its slots) and the
    index of its last filled slot (0 when empty)."""
    chunks = (length + Int32(b_t - 1)) // Int32(b_t)
    count = piece_count(b_t, length, span_chunks)
    own_span = (chunks + count - Int32(1)) // count
    own_span = own_span if own_span > Int32(1) else Int32(1)
    span_tokens = Int64(own_span) * Int64(b_t)
    span_tokens32 = span_tokens.to(Int32)
    filled = (length + span_tokens32 - Int32(1)) // span_tokens32
    last = filled - Int32(1) if filled > Int32(1) else Int32(0)
    return span_tokens, last


@cute.jit
def piece_table_body(
    pieces: Int32,
    b_t: cutlass.Constexpr[int],
    heads_out: Int32,
    tidx,
    num_seqs: Int32,
    mCu: cute.Tensor,
    mCuPieces: cute.Tensor,
    mMainRows: cute.Tensor,
    mSummaryRows: cute.Tensor,
):
    """Piece table of one ``ORDER_THREADS`` CTA (every thread calls; CTA barriers inside).

    The wave is ``num_seqs * pieces`` slots of ``span = ceil(total_chunks / wave)`` chunks;
    thread ``b`` owns sequence ``b``, taking ``count_b = ceil(nc_b / span)`` slots of
    ``ceil(nc_b / count_b)`` chunks. When the per-sequence ceilings would overflow the wave the
    span is recomputed against ``wave - (num_seqs - 1)`` slots, which cannot overflow.
    ``cu_pieces[slot_b + j] = cu[b] + min(j * span_b, len_b)``. Deterministic."""
    num_warps = cutlass.const_expr(ORDER_THREADS // 32)
    smem = SmemAllocator()
    sWarpMain = smem.allocate_tensor(Int32, cute.make_layout(num_warps), 16)
    sWarpSummary = smem.allocate_tensor(Int32, cute.make_layout(num_warps), 16)
    lane = tidx % Int32(32)
    warp = tidx // Int32(32)

    # ---- one round trip: this thread's first sequence and the batch total --------------------
    first_read = tidx if tidx < num_seqs else num_seqs - Int32(1)
    first_start = Int32(mCu[first_read])
    first_end = Int32(mCu[first_read + 1])
    total_tokens = Int32(mCu[num_seqs]) - Int32(mCu[0])

    # ---- the slot span: one wave shared by the batch, guarded against the ceilings -----------
    total_chunks = (total_tokens + Int32(b_t - 1)) // Int32(b_t)
    wave = num_seqs * Int32(pieces)
    span_chunks = chunks_per_slot(total_chunks, wave)
    if num_seqs > Int32(1):
        taken = Int32(0)
        probe_start = Int32(0)
        while probe_start < num_seqs:
            b = probe_start + tidx
            valid = b < num_seqs
            length = first_end - first_start
            if probe_start > Int32(0):
                b_read = b if valid else num_seqs - Int32(1)
                length = Int32(mCu[b_read + 1]) - Int32(mCu[b_read])
            count = piece_count(b_t, length, span_chunks)
            # ---- CTA sum of the counts: warp scan, lane 31 parks the warp total ----------
            inclusive = count if valid else Int32(0)
            for offset in [1, 2, 4, 8, 16]:
                other = Int32(cute.arch.shuffle_sync_up(inclusive, offset, mask_and_clamp=0))
                inclusive = inclusive + (other if lane >= Int32(offset) else Int32(0))
            if lane == Int32(31):
                sWarpMain[warp] = inclusive
            cute.arch.barrier()
            total = Int32(0)
            for i in cutlass.range_constexpr(num_warps):
                total = total + sWarpMain[i]
            cute.arch.barrier()
            taken = taken + total
            probe_start = probe_start + Int32(ORDER_THREADS)
        if taken > wave:
            span_chunks = chunks_per_slot(total_chunks, wave - num_seqs + Int32(1))

    running_main = Int32(0)
    running_summary = Int32(0)
    block_start = Int32(0)
    while block_start < num_seqs:
        b = block_start + tidx
        valid = b < num_seqs
        start = first_start
        end = first_end
        if block_start > Int32(0):
            b_read = b if valid else num_seqs - Int32(1)
            start = Int32(mCu[b_read])
            end = Int32(mCu[b_read + 1])
        length = end - start
        span_tokens, last = piece_span(b_t, length, span_chunks)
        rows = (last + Int32(1)) * Int32(heads_out)
        rows_main = rows if valid else Int32(0)
        rows_summary = rows if (valid and last > Int32(0)) else Int32(0)

        # ---- row bases: warp scan, then the warp totals --------------------------------------
        incl_main = rows_main
        incl_summary = rows_summary
        for off in [1, 2, 4, 8, 16]:
            other_main = Int32(cute.arch.shuffle_sync_up(incl_main, off, mask_and_clamp=0))
            other_summary = Int32(cute.arch.shuffle_sync_up(incl_summary, off, mask_and_clamp=0))
            incl_main = incl_main + (other_main if lane >= Int32(off) else Int32(0))
            incl_summary = incl_summary + (other_summary if lane >= Int32(off) else Int32(0))
        if lane == Int32(31):
            sWarpMain[warp] = incl_main
            sWarpSummary[warp] = incl_summary
        cute.arch.barrier()
        total_main = Int32(0)
        total_summary = Int32(0)
        warp_base_main = Int32(0)
        warp_base_summary = Int32(0)
        for w in cutlass.range_constexpr(num_warps):
            count_main = sWarpMain[w]
            count_summary = sWarpSummary[w]
            total_main = total_main + count_main
            total_summary = total_summary + count_summary
            warp_base_main = warp_base_main + (count_main if Int32(w) < warp else Int32(0))
            warp_base_summary = warp_base_summary + (
                count_summary if Int32(w) < warp else Int32(0)
            )

        # ---- per sequence: the flat slots and the row bases ----------------------------------
        if valid:
            length64 = Int64(length)
            row_base = running_main + warp_base_main + incl_main - rows_main
            slot_base = row_base // Int32(heads_out)
            j = Int32(0)
            while j <= last:
                offset = Int64(j) * span_tokens
                offset = offset if offset < length64 else length64
                mCuPieces[slot_base + j] = start + offset.to(Int32)
                j = j + Int32(1)
            mMainRows[b] = row_base
            mSummaryRows[b] = running_summary + warp_base_summary + incl_summary - rows_summary
        running_main = running_main + total_main
        running_summary = running_summary + total_summary
        cute.arch.barrier()
        block_start = block_start + Int32(ORDER_THREADS)
    # ---- the batch end, then padding to the wave: consumers sizing slots from the shape ------
    total_slots = running_main // Int32(heads_out)
    batch_end = Int32(mCu[num_seqs])
    tail = total_slots + tidx
    while tail <= num_seqs * Int32(pieces):
        mCuPieces[tail] = batch_end
        tail = tail + Int32(ORDER_THREADS)
    if tidx == 0:
        mMainRows[num_seqs] = running_main
        mSummaryRows[num_seqs] = running_summary


@cute.kernel
def chain_table_kernel(
    b_t: cutlass.Constexpr[int],
    rows_b_t: cutlass.Constexpr[int],
    pieces: Int32,
    heads_out: Int32,
    num_seqs: Int32,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor | None,
    scheduler: cute.Tensor,
    rows: cute.Tensor | None,
    row_count: cute.Tensor | None,
):
    """Every block builds the piece table; block 0 then orders the main (and summary) work
    tables and zeroes the scheduler counters, block 1 (bound ``rows``) builds the chunk row
    table of the pieces on its first warp."""
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    tidx = Int32(cute.arch.thread_idx()[0])
    piece_table_body(
        pieces, b_t, heads_out, tidx, num_seqs, cu_seqlens, cu_pieces, main_rows, summary_rows
    )
    cute.arch.barrier()
    n_tiles = heads_out * num_seqs * pieces
    if Int32(cute.arch.block_idx()[0]) == Int32(0):
        sKey, sIdx, sSpread = order_storage()
        order_body(
            "chain",
            b_t,
            tidx,
            heads_out,
            n_tiles,
            cu_pieces,
            None,
            main_count,
            work_items,
            scheduler,
            sKey,
            sIdx,
            sSpread,
            mRowBase=main_rows,
        )
        if cutlass.const_expr(work_items_summary is not None):
            cute.arch.barrier()
            order_body(
                "chain",
                b_t,
                tidx,
                heads_out,
                n_tiles,
                cu_pieces,
                None,
                summary_count,
                work_items_summary,
                None,
                sKey,
                sIdx,
                sSpread,
                mRowBase=summary_rows,
                mSlotRows=main_rows,
            )
    else:
        if cutlass.const_expr(rows is not None):
            if tidx < Int32(32):
                emit_chunk_rows(rows_b_t, cu_pieces, rows, row_count, tidx)


@cute.jit
def launch_chain_tables(
    b_t: cutlass.Constexpr[int],
    pieces: Int32,
    heads_out: Int32,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor | None,
    scheduler: cute.Tensor,
    stream: cuda.CUstream,
    *,
    rows: cute.Tensor | None = None,
    row_count: cute.Tensor | None = None,
    rows_b_t: cutlass.Constexpr[int] = 0,
) -> None:
    """The chain prologue: the piece table of ``cu_seqlens``, its work tables and, with
    ``rows``, the chunk row table of ``rows_b_t``-token chunks of the pieces. Callable from a
    launch bundle's ``@cute.jit`` host; the next launch waits on it through PDL."""
    chain_table_kernel(
        b_t,
        rows_b_t,
        pieces,
        heads_out,
        Int32(cu_seqlens.shape[0] - 1),
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler,
        rows,
        row_count,
    ).launch(
        grid=(1 if cutlass.const_expr(rows is None) else 2, 1, 1),
        block=(ORDER_THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )


# ---- state chain -----------------------------------------------------------------------------


@cute.jit
def load_row_f32(mT, addr, cols: cutlass.Constexpr[int]):
    """``cols`` consecutive fp32 values of one state row of ``mT`` (fp32 or bf16) at
    ``addr``, read as the widest vectors."""
    if cutlass.const_expr(mT.element_type is Float32):
        if cutlass.const_expr(cols == 2):
            vals = ld_global_v(addr, Float32, 2)
        else:
            vals = [
                x for q in range(cols // 4) for x in ld_global_v(addr + Int64(16 * q), Float32, 4)
            ]
    else:
        if cutlass.const_expr(cols == 2):
            words = ld_global_v(addr, Int32, 1)
        elif cutlass.const_expr(cols == 4):
            words = ld_global_v(addr, Int32, 2)
        else:
            words = [
                w for q in range(cols // 8) for w in ld_global_v(addr + Int64(16 * q), Int32, 4)
            ]
        vals = [x for word in words for x in f16x2_to_f32(word, dtype=mT.element_type)]
    return vals


@cute.jit
def store_row_f32(addr, vals, cols: cutlass.Constexpr[int]):
    """``cols`` fp32 ``vals`` to global address ``addr`` as the widest vectors."""
    if cutlass.const_expr(cols == 2):
        st_global_v(addr, vals, Float32)
    else:
        for q in cutlass.range_constexpr(cols // 4):
            st_global_v(addr + Int64(16 * q), vals[4 * q : 4 * q + 4], Float32)


@cute.jit
def store_row_shared(ptr, vals, cols: cutlass.Constexpr[int]):
    """``cols`` fp32 ``vals`` to SMEM pointer ``ptr`` as the widest vectors."""
    if cutlass.const_expr(cols == 2):
        st_shared_f32(ptr, vals)
    else:
        for q in cutlass.range_constexpr(cols // 4):
            st_shared_f32(ptr + 4 * q, vals[4 * q : 4 * q + 4])


@cute.jit
def load_row_shared(ptr, cols: cutlass.Constexpr[int]):
    """``cols`` fp32 values at SMEM pointer ``ptr`` as the widest vectors."""
    if cutlass.const_expr(cols == 2):
        return ld_shared_f32(ptr, 2)
    return [x for q in range(cols // 4) for x in ld_shared_f32(ptr + 4 * q, 4)]


@cute.jit
def element_addr(mT, i0, i1, i2, col):
    """64-bit global address of ``mT[i0, i1, i2, col]`` (unit last stride)."""
    offset = (
        Int64(i0) * Int64(mT.stride[0])
        + Int64(i1) * Int64(mT.stride[1])
        + Int64(i2) * Int64(mT.stride[2])
        + Int64(col)
    )
    return mT.iterator.toint() + offset * Int64(mT.element_type.width // 8)


@cute.jit
def prefetch_m(sM, mM, buf_words, slot, heads_out, h, tidx, dim_k: cutlass.Constexpr[int]):
    """``cp.async`` of ``M[slot, h]`` (contiguous ``dim_k x dim_k`` fp32) into the SMEM buffer
    at ``buf_words``, 16-byte chunks XOR-swizzled by row, then commit the group."""
    K = dim_k
    for q in cutlass.range_constexpr(K * K // 4 // CHAIN_THREADS):
        chunk_id = tidx + Int32(q * CHAIN_THREADS)
        m_row = chunk_id // Int32(K // 4)
        m_chunk = chunk_id % Int32(K // 4)
        dst = sM.iterator + (
            buf_words + (m_row * K + (m_chunk ^ ((m_row // Int32(4)) % Int32(8))) * Int32(4))
        )
        src = (
            mM.iterator
            + ((Int64(slot) * Int64(heads_out) + Int64(h)) * Int64(K) + Int64(m_row)) * Int64(K)
            + Int64(m_chunk) * Int64(4)
        )
        mlir_nvvm.cp_async_shared_global(
            dst.llvm_ptr, src.llvm_ptr, 16, mlir_nvvm.LoadCacheModifierKind.CG
        )
    cute.arch.cp_async_commit_group()


@cute.kernel
def state_chain_kernel(
    dim_k: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    transpose: cutlass.Constexpr[bool],
    heads_out: Int32,
    mH: cute.Tensor,
    mM: cute.Tensor,
    mX: cute.Tensor,
    mSeed: cute.Tensor | None,
    mMainRows: cute.Tensor,
    mSeedIndices: cute.Tensor | None,
):
    """CTA ``(seq, h, c)`` chains state rows ``[c * rows, (c + 1) * rows)`` over the
    sequence's flat slots ``main_rows[seq] // HO .. main_rows[seq + 1] // HO``, writing each
    slot's incoming state to ``X``. Eight warps (two row groups times four k slices, lane ``l``
    owning ``K / 32`` columns); each ``M_j`` is staged whole in SMEM by double-buffered
    ``cp.async`` and read as 8-wide k tiles, the four slice partials added to ``H_j`` in slice
    order by the row owner. A one-slot sequence copies its seed into ``X``."""
    K = cutlass.const_expr(dim_k)
    cols = cutlass.const_expr(K // CHAIN_LANES)
    k_slice = cutlass.const_expr(K // CHAIN_K_WARPS)
    tiles = cutlass.const_expr(k_slice // CHAIN_K_TILE)
    m_words = cutlass.const_expr(K * K)
    group_rows = cutlass.const_expr(rows // CHAIN_ROW_GROUPS)
    own_rows = cutlass.const_expr(rows // CHAIN_WARPS)

    cute.arch.griddepcontrol_wait()
    tidx = Int32(cute.arch.thread_idx()[0])
    warp = tidx // Int32(CHAIN_LANES)
    lane = tidx % Int32(CHAIN_LANES)
    k_warp = warp % Int32(CHAIN_K_WARPS)
    row_group = warp // Int32(CHAIN_K_WARPS)
    col0 = lane * Int32(cols)
    bid = cute.arch.block_idx()
    seq = Int32(bid[0])
    h = Int32(bid[1])
    seed_seq = seq
    if cutlass.const_expr(mSeedIndices is not None):
        seed_seq = Int32(mSeedIndices[seq])
    row0 = Int32(bid[2]) * Int32(rows)

    # ---- walk bounds -------------------------------------------------------------------------
    row_base = Int32(mMainRows[seq])
    seq_base = row_base // heads_out
    filled = (Int32(mMainRows[seq + 1]) - row_base) // heads_out
    count = filled if filled > Int32(0) else Int32(1)

    smem = SmemAllocator()
    sWalk = smem.allocate_tensor(Float32, cute.make_layout(rows * K), 16)
    sPart = smem.allocate_tensor(Float32, cute.make_layout(CHAIN_K_WARPS * rows * K), 16)
    sM = smem.allocate_tensor(Float32, cute.make_layout(CHAIN_M_BUFFERS * m_words), 128)
    acc = cute.make_rmem_tensor((group_rows * cols,), Float32)
    m_tile = cute.make_rmem_tensor((CHAIN_K_TILE * cols,), Float32)
    cur = cute.make_rmem_tensor((own_rows * cols,), Float32)
    piece_h = cute.make_rmem_tensor((own_rows * cols,), Float32)

    # ---- seeds -------------------------------------------------------------------------------
    for rr in cutlass.range_constexpr(own_rows):
        seed_row = row_group * Int32(group_rows) + k_warp * Int32(own_rows) + Int32(rr)
        if cutlass.const_expr(mSeed is not None):
            seed_vals = load_row_f32(
                mSeed, element_addr(mSeed, seed_seq, h, row0 + seed_row, col0), cols
            )
        else:
            seed_vals = [Float32(0.0)] * cols
        for c in cutlass.range_constexpr(cols):
            cur[rr * cols + c] = seed_vals[c]

    if count == Int32(1):
        for rr in cutlass.range_constexpr(own_rows):
            seed_row = row_group * Int32(group_rows) + k_warp * Int32(own_rows) + Int32(rr)
            store_row_f32(
                element_addr(mX, seq_base, h, row0 + seed_row, col0),
                [cur[rr * cols + c] for c in range(cols)],
                cols,
            )
    else:
        for rr in cutlass.range_constexpr(own_rows):
            seed_row = row_group * Int32(group_rows) + k_warp * Int32(own_rows) + Int32(rr)
            store_row_shared(
                sWalk.iterator + (seed_row * K + col0),
                [cur[rr * cols + c] for c in range(cols)],
                cols,
            )
        first_step = Int32(0) if cutlass.const_expr(not transpose) else count - Int32(1)
        prefetch_m(sM, mM, Int32(0), seq_base + first_step, heads_out, h, tidx, K)

        # ---- walk ----------------------------------------------------------------------------
        i = Int32(0)
        while i < count:
            step = i if cutlass.const_expr(not transpose) else count - Int32(1) - i
            slot = seq_base + step
            for rr in cutlass.range_constexpr(own_rows):
                row = row0 + row_group * Int32(group_rows) + k_warp * Int32(own_rows) + Int32(rr)
                store_row_f32(
                    element_addr(mX, slot, h, row, col0),
                    [cur[rr * cols + c] for c in range(cols)],
                    cols,
                )
                h_vals = load_row_f32(mH, element_addr(mH, slot, h, row, col0), cols)
                for c in cutlass.range_constexpr(cols):
                    piece_h[rr * cols + c] = h_vals[c]
            if i + Int32(1) < count:
                next_step = (
                    step + Int32(1) if cutlass.const_expr(not transpose) else step - Int32(1)
                )
                next_buf_words = ((i + Int32(1)) % Int32(CHAIN_M_BUFFERS)) * Int32(m_words)
                prefetch_m(sM, mM, next_buf_words, seq_base + next_step, heads_out, h, tidx, K)
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            buf_words = (i % Int32(CHAIN_M_BUFFERS)) * Int32(m_words)

            # ---- one sweep over the k tiles ------------------------------------------------
            for j in cutlass.range_constexpr(group_rows * cols):
                acc[j] = Float32(0.0)
            for tile in cutlass.range_constexpr(tiles):
                k0 = k_warp * Int32(k_slice) + Int32(tile * CHAIN_K_TILE)
                if cutlass.const_expr(transpose):
                    for c in cutlass.range_constexpr(cols):
                        m_row = col0 + Int32(c)
                        m_vals = []
                        for v in cutlass.range_constexpr(CHAIN_K_TILE // 4):
                            m_col = k0 + Int32(4 * v)
                            m_chunk = m_col // Int32(4)
                            m_word = (
                                buf_words
                                + (
                                    m_row * K
                                    + (m_chunk ^ ((m_row // Int32(4)) % Int32(8))) * Int32(4)
                                )
                                + m_col % Int32(4)
                            )
                            m_vals += ld_shared_f32(sM.iterator + m_word, 4)
                        for kk in cutlass.range_constexpr(CHAIN_K_TILE):
                            m_tile[kk * cols + c] = m_vals[kk]
                else:
                    for kk in cutlass.range_constexpr(CHAIN_K_TILE):
                        m_row = k0 + Int32(kk)
                        m_vals = []
                        for v in cutlass.range_constexpr(cols // min(4, cols)):
                            m_col = col0 + Int32(min(4, cols) * v)
                            m_chunk = m_col // Int32(4)
                            m_word = (
                                buf_words
                                + (
                                    m_row * K
                                    + (m_chunk ^ ((m_row // Int32(4)) % Int32(8))) * Int32(4)
                                )
                                + m_col % Int32(4)
                            )
                            m_vals += ld_shared_f32(sM.iterator + m_word, min(4, cols))
                        for c in cutlass.range_constexpr(cols):
                            m_tile[kk * cols + c] = m_vals[c]
                for r in cutlass.range_constexpr(group_rows):
                    op_row = row_group * Int32(group_rows) + Int32(r)
                    for kq in cutlass.range_constexpr(CHAIN_K_TILE // 4):
                        x4 = ld_shared_f32(sWalk.iterator + (op_row * K + k0 + Int32(4 * kq)), 4)
                        for e in cutlass.range_constexpr(4):
                            for c in cutlass.range_constexpr(cols):
                                acc[r * cols + c] = cute.math.fma(
                                    x4[e], m_tile[(4 * kq + e) * cols + c], acc[r * cols + c]
                                )

            # ---- slice partials ------------------------------------------------------------
            for r in cutlass.range_constexpr(group_rows):
                op_row = row_group * Int32(group_rows) + Int32(r)
                store_row_shared(
                    sPart.iterator + ((k_warp * Int32(rows) + op_row) * K + col0),
                    [acc[r * cols + c] for c in range(cols)],
                    cols,
                )
            cute.arch.barrier()

            # ---- owned rows: H_j plus the four slice partials in slice order ---------------
            for rr in cutlass.range_constexpr(own_rows):
                op_row = row_group * Int32(group_rows) + k_warp * Int32(own_rows) + Int32(rr)
                total = [piece_h[rr * cols + c] for c in range(cols)]
                for w2 in cutlass.range_constexpr(CHAIN_K_WARPS):
                    part = load_row_shared(
                        sPart.iterator + ((Int32(w2 * rows) + op_row) * K + col0), cols
                    )
                    for c in cutlass.range_constexpr(cols):
                        total[c] = total[c] + part[c]
                for c in cutlass.range_constexpr(cols):
                    cur[rr * cols + c] = total[c]
                store_row_shared(sWalk.iterator + (op_row * K + col0), total, cols)
            cute.arch.barrier()
            i = i + Int32(1)
    cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def launch_state_chain(
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    rows: cutlass.Constexpr[int],
    transpose: cutlass.Constexpr[bool],
    heads_out: Int32,
    num_seqs: Int32,
    mH: cute.Tensor,
    mM: cute.Tensor,
    mX: cute.Tensor,
    mSeed: cute.Tensor | None,
    mMainRows: cute.Tensor,
    mSeedIndices: cute.Tensor | None,
    stream: cuda.CUstream,
) -> None:
    """Chain the pieces' summaries into every slot's incoming state ``X`` (``transpose``: the
    backward recurrence over ``M^T``, last slot first), seeded by ``mSeed`` (zero when None,
    row ``mSeedIndices[seq]`` when bound); ``rows`` from :func:`chain_rows_per_cta`. Callable
    from a launch bundle's ``@cute.jit`` host."""
    state_chain_kernel(
        dim_k, rows, transpose, heads_out, mH, mM, mX, mSeed, mMainRows, mSeedIndices
    ).launch(
        grid=(num_seqs, heads_out, dim_v // rows),
        block=(CHAIN_THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )


# ---- standalone entry points -----------------------------------------------------------------


def _int32_vector(sym):
    return make_fake_compact_tensor(cutlass.Int32, (sym(),), assumed_align=4)


@jit_cache
def compile_chain_tables(b_t: int, summary: bool, rows_b_t: int, use_int64_offsets: bool):
    """Compile :func:`launch_chain_tables`; every extent is symbolic (int64 with
    ``use_int64_offsets``)."""
    sym = cute.sym_int64 if use_int64_offsets else cute.sym_int

    @cute.jit
    def entry(
        pieces,
        heads_out,
        cu,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        items,
        items_summary,
        sched,
        rows,
        row_count,
        stream,
    ):
        launch_chain_tables(
            b_t,
            pieces,
            heads_out,
            cu,
            cu_pieces,
            main_rows,
            summary_rows,
            main_count,
            summary_count,
            items,
            items_summary,
            sched,
            stream,
            rows=rows,
            row_count=row_count,
            rows_b_t=rows_b_t,
        )

    def table():
        return make_fake_compact_tensor(cutlass.Int32, (sym(), WORK_ITEM_FIELDS), assumed_align=4)

    return compile_tvm_ffi(
        entry,
        Int32(0),
        Int32(0),
        *(_int32_vector(sym) for _ in range(6)),
        table(),
        table() if summary else None,
        _int32_vector(sym),
        make_fake_strided_tensor(cutlass.Int32, (sym(), 4), contiguous_dim=None)
        if rows_b_t
        else None,
        _int32_vector(sym) if rows_b_t else None,
        opt_level=OPT_LEVEL,
        name=f"wd_chain_tables_{b_t}_{int(summary)}_{rows_b_t}_i64{int(use_int64_offsets)}",
    )


def build_chain_tables(
    cu_seqlens: torch.Tensor,
    piece_table: dict,
    *,
    pieces: int,
    heads_out: int,
    b_t: int,
    rows: torch.Tensor | None = None,
    row_count: torch.Tensor | None = None,
    rows_b_t: int = 0,
    use_int64_offsets: bool = False,
) -> None:
    """Run :func:`launch_chain_tables` standalone over the ``piece_table`` buffers (keys
    ``cu_pieces``, ``main_rows``, ``summary_rows``, ``main_count``, ``summary_count``,
    ``work_items``, optional ``work_items_summary``, ``scheduler``)."""
    summary = piece_table.get("work_items_summary")
    fn = compile_chain_tables(b_t, summary is not None, rows_b_t, use_int64_offsets)
    fn(
        pieces,
        heads_out,
        cu_seqlens,
        piece_table["cu_pieces"],
        piece_table["main_rows"],
        piece_table["summary_rows"],
        piece_table["main_count"],
        piece_table["summary_count"],
        piece_table["work_items"],
        summary,
        piece_table["scheduler"],
        rows,
        row_count,
    )


@jit_cache
def compile_state_chain(
    dim_v: int,
    dim_k: int,
    rows: int,
    transpose: bool,
    seed_dtype_name: str | None,
    routed: bool,
    use_int64_offsets: bool,
):
    """Compile :func:`launch_state_chain`; every extent but ``V`` / ``K`` is symbolic (int64
    with ``use_int64_offsets``)."""
    sym = cute.sym_int64 if use_int64_offsets else cute.sym_int

    @cute.jit
    def entry(heads_out, num_seqs, mH, mM, mX, mSeed, mMainRows, mSeedIndices, stream):
        launch_state_chain(
            dim_v,
            dim_k,
            rows,
            transpose,
            heads_out,
            num_seqs,
            mH,
            mM,
            mX,
            mSeed,
            mMainRows,
            mSeedIndices,
            stream,
        )

    def state(dtype, d):
        return make_fake_strided_tensor(
            dtype,
            (sym(), sym(), d, dim_k),
            stride_divisibility=4,
            assumed_align=16,
        )

    seed_dtype = None if seed_dtype_name is None else getattr(cutlass, seed_dtype_name)
    return compile_tvm_ffi(
        entry,
        Int32(0),
        Int32(0),
        state(Float32, dim_v),
        make_fake_compact_tensor(Float32, (sym(), sym(), dim_k, dim_k), assumed_align=16),
        state(Float32, dim_v),
        None if seed_dtype is None else state(seed_dtype, dim_v),
        _int32_vector(sym),
        _int32_vector(sym) if routed else None,
        opt_level=OPT_LEVEL,
        name=f"wd_state_chain_{dim_v}_{dim_k}_{rows}_{int(transpose)}_"
        f"{(seed_dtype_name or 'none').lower()}_{int(routed)}_i64{int(use_int64_offsets)}",
    )


_STATE_DTYPES = {torch.float32: "Float32", torch.bfloat16: "BFloat16"}


def state_chain(
    state_h: torch.Tensor,
    state_m: torch.Tensor,
    state_x: torch.Tensor,
    main_rows: torch.Tensor,
    *,
    seed: torch.Tensor | None = None,
    seed_indices: torch.Tensor | None = None,
    transpose: bool = False,
    rows: int | None = None,
    use_int64_offsets: bool = False,
) -> None:
    """Run :func:`launch_state_chain` standalone: ``state_x[slot]`` receives every slot's
    incoming state from ``state_h`` / ``state_m`` and the per-sequence ``seed``."""
    _, heads_out, dim_v, dim_k = state_h.shape
    num_seqs = main_rows.numel() - 1
    if rows is None:
        rows = chain_rows_per_cta(
            dim_v,
            dim_k,
            num_seqs,
            heads_out,
            torch.cuda.get_device_properties(state_h.device).multi_processor_count,
        )
    fn = compile_state_chain(
        dim_v,
        dim_k,
        rows,
        transpose,
        None if seed is None else _STATE_DTYPES[seed.dtype],
        seed_indices is not None,
        use_int64_offsets,
    )
    fn(heads_out, num_seqs, state_h, state_m, state_x, seed, main_rows, seed_indices)


for _kernel in (chain_table_kernel, state_chain_kernel):
    _kernel.set_name_prefix("wd", remove_cutlass_symbol=False)
