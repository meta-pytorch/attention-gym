# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Split-K sequence partitioning for the chunked linear-attention kernels:
GDN (scalar ``gate (T, HO)``, b_t=64) and KDA / GDN-2 (per-key-channel
``gate (T, HO, DK)``, b_t=16).

The main kernels are persistent, one CTA per (batch, head); when ``B * HO``
does not fill the SMs, long sequences serialize on a few CTAs.  This module
cuts a sequence into independent work items where the forget gates saturate
on BOTH sides of the cut, so the recurrent state (forward) and its gradient
(backward) can be rebuilt from a short warmup the item recomputes itself.
No state is exchanged between items.

Work-item row (``WORK_ITEM_FIELDS`` x int32, chunk units)::

    [batch_idx, head_idx, write_start, write_end, compute_start, compute_end, batch_start, batch_end,
     final_dst, dstate_dst]

The item writes outputs for ``[write_start, write_end)``.  Forward computes
``[compute_start, write_end)``, backward ``[write_start, compute_end)``; the
extra chunks on each side are the warmups, accurate to ``2^log2_threshold``.
``compute_start == 0`` seeds the true initial state and ``compute_end ==
num_chunks`` the true ``d_final_state``, so the uncut item ``(0, nc, 0, nc)``
reproduces the serial kernel exactly.  ``final_dst`` is the sequence whose
``final_state`` the item stores (the item writing the last chunk), ``dstate_dst``
the sequence whose ``d_initial_state`` it stores (the item writing chunk 0);
``-1`` stores nothing.  In chain mode ``batch_idx`` is the piece slot and the two
destinations name the sequence.

The pipeline:

0. plan: one thread settles the batch-wide span the piece choice may be
   overridden with and leaves it in the chunk scratch's last row.
1. scan: reduce the gate to one value per chunk into the caller's GMEM
   scratch (channel gates: max over the per-channel clamped-log2 sums, so a
   cut needs EVERY channel saturated; scalar gates: the plain sum).  Channel
   gates read only the chunks near a candidate boundary; skipping chunks only
   raises the negative sums, so the window can miss a cut but never accept a
   bad one.  Scalar gates are read whole.
2. walk: one CTA per (batch, head).  For scalar gates the tile's total decay
   first decides whether any cut can saturate and prices the expected warmup
   into the piece count.  Then one warp per candidate boundary prefix-sums the
   boundary's chunk window and binary-searches, one lane per snap offset, the
   smallest saturating warmup on each side; the cut is not pinned to
   ``j*span``: positions within ``SNAP_CHUNKS`` are scored on ``warmup_before
   + warmup_after + |offset|``, clamped to ``span // 2`` so adjacent cuts
   cannot cross.  Thread 0 emits every saturating cut.
3. order: bitonic-sort into ``work_items``, longest ``[compute_start,
   compute_end)`` first, so the ticket scheduler runs LPT, and zero the
   scheduler rings (dirty on exit).

With ``split=False`` scan and walk never launch and the prologue kernel
synthesizes the uncut item per (batch, head) from ``cu_seqlens`` alone: that
no-cuts table serves batch-invariant mode and checkpoint cadences a cut may
not cross.
"""

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.cute.runtime import from_dlpack

from ..tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from ..tile_dsl.pointwise import opaque_f32_zero, sigmoid, softplus
from ..tile_dsl.tma import ld_global_v2, ld_global_v4

from .host import get_dtype
from .._compat import current_device

USE_PDL = True

WORK_ITEM_FIELDS = 10
WORK_ITEM_FINAL_DST = 8
WORK_ITEM_DSTATE_DST = 9
WARMUP_CAP_CHANNEL = 32  # per-channel gates: a cut must saturate within one warp of chunks per side
WALK_WINDOW_CHUNKS = 1088  # scalar gates: chunk window the walk prefix-sums per boundary; the cap is what the snap search leaves of it
SNAP_CHUNKS = 32  # cut-position search half-width
MAX_BLOCKS = 2048  # piece-count ceiling; host clamps ideal_chunks so the per-tile block count fits
WARP_SIZE = 32
WARPS = 8
THREADS_PER_BLOCK = WARPS * WARP_SIZE
SCAN_WARPS = 4
SCAN_THREADS = SCAN_WARPS * WARP_SIZE
PLAN_THREADS = 128
SCAN_CTA_CAP = 16  # scan CTAs per SM before the block axis becomes a grid-stride loop
SCAN_BLOCK_LOOP_MAX = 4  # block-loop trips the cap may cost
SCAN_ROWS_PER_WARP_MAX = 4  # consecutive chunk rows per scan warp; scan_rows_per_warp() shrinks it to fill the SMs
SCAN_TOKEN_STRIDE = 1  # tokens sampled per chunk; a stride > 1 scales the effective threshold by it
OVERHEAD_TOKENS = 256  # per-item fixed cost for the piece model: state reseed + pipeline refill + typical warmup
P_WINDOW = 16  # fill-regime piece-count search width
P_BELOW = 8  # how far below the ideal-cap floor the fill-regime search may go

ORDER_THREADS = 1024
ORDER_ELEMENTS = 4
ORDER_CAPACITY = ORDER_THREADS * ORDER_ELEMENTS  # sort capacity (32 KB SMEM); past this the device-side branch copies through unsorted

DEFAULT_LOG2_THRESHOLD = -10.0 / math.log(2.0)  # e^-10, in log2 units
RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the scan's log2 domain


def compute_ideal_chunks(total_tokens: int, n_heads_out: int, num_sms: int, b_t: int) -> int:
    """Ideal per-work-item span in chunks: total work across all (b,h)
    tiles divided by the SM count, floored at one chunk and clamped so the
    per-tile block count fits ``MAX_BLOCKS``."""
    total_chunks = -(-total_tokens // b_t)
    ideal_tokens = -(-(total_tokens * n_heads_out) // num_sms)
    ideal_chunks = max(1, -(-ideal_tokens // b_t))
    return max(ideal_chunks, -(-total_chunks // MAX_BLOCKS))


def warmup_cap_chunks(gate_channels: int, expand_num: int) -> int:
    """Chunks per side a cut may recompute: one warp of chunks for per-channel
    gates (their scan is windowed around each candidate boundary), and for
    scalar gates the walk's SMEM window minus the snap search, 512 at
    ``expand_num`` 1."""
    if gate_channels > 0:
        return WARMUP_CAP_CHANNEL
    return WALK_WINDOW_CHUNKS // 2 - SNAP_CHUNKS * expand_num


def scan_rows_per_warp(need_rows: int, grid_y: int, num_sms: int) -> int:
    """Chunk rows each scan warp walks.  The largest of ``SCAN_ROWS_PER_WARP_MAX``,
    2, 1 whose grid (``ceil(need_rows / (SCAN_WARPS * rows)) * grid_y`` blocks)
    still covers the SMs."""
    for rows in (SCAN_ROWS_PER_WARP_MAX, 2, 1):
        if rows == 1 or -(-need_rows // (SCAN_WARPS * rows)) * grid_y >= num_sms:
            return rows


def chunk_scratch_rows(total_tokens: int, batch_size: int, b_t: int) -> int:
    """Rows of the ``(rows, HO)`` fp32 chunk-value scratch: per-batch chunk
    ranges are based at ``cu[b] // b_t + b``, so one extra row per sequence
    covers the ceil rounding, plus the last row, which carries the plan
    (:func:`frost_split_k_plan`) rather than a chunk value."""
    return total_tokens // b_t + batch_size + 1


def max_work_items(total_tokens: int, batch_size: int, n_heads_out: int, ideal_chunks: int, b_t: int, num_sms: int) -> int:
    """Provable upper bound on the emitted item count: spans are capped at
    ``ideal_chunks`` (at most ``ceil(nc / ideal) + 1`` pieces per (b, h)),
    zero-length sequences emit one item each, and the fill-regime search
    can widen any tile by up to ``P_WINDOW - 1`` extra pieces."""
    total_chunks = -(-total_tokens // b_t)
    window = P_WINDOW - 1 if batch_size * n_heads_out < 2 * num_sms else 0
    return n_heads_out * (-(-total_chunks // ideal_chunks) + (2 + window) * batch_size)


@cute.jit
def decode_work_item(cfg, tile_idx, mWorkItems):
    """Tile decode shared by every warp body of the main kernels: read the
    work-item row (an uncut table row IS the whole sequence).  Returns
    ``(batch_idx, head_idx, batch_start, batch_end, batch_seqlen,
    batch_num_chunks, write_start, write_end, compute_start,
    compute_end)`` with chunk-unit bounds."""
    batch_idx = mWorkItems[tile_idx, 0]
    head_idx = mWorkItems[tile_idx, 1]
    write_start = mWorkItems[tile_idx, 2]
    write_end = mWorkItems[tile_idx, 3]
    compute_start = mWorkItems[tile_idx, 4]
    compute_end = mWorkItems[tile_idx, 5]
    batch_start = mWorkItems[tile_idx, 6]
    batch_end = mWorkItems[tile_idx, 7]
    batch_seqlen = batch_end - batch_start
    batch_num_chunks = cute.ceil_div(batch_seqlen, cfg.b_t)
    return batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end


@cute.jit
def decode_head(cfg, head_idx):
    """``(head_o, v_offset)`` of tile head ``head_idx``: the gate / output head and the tile's first value column.  Under
    ``cfg.tiles_per_head`` > 1 every gate head runs as ``tiles_per_head`` tiles of ``cfg.d_v`` value columns each."""
    if cutlass.const_expr(cfg.tiles_per_head > 1):
        head_o = head_idx // cfg.tiles_per_head
        v_offset = (head_idx - head_o * cfg.tiles_per_head) * cfg.d_v
    else:
        head_o = head_idx
        v_offset = cutlass.Int32(0)
    return head_o, v_offset


@cute.jit
def emit_item(mWorkItems, mCount, batch_idx, head_idx, write_start, write_end, compute_start, compute_end, batch_start, batch_end, batch_num_chunks):
    slot = cutlass.Int32(nvvm.atomicrmw(nvvm.AtomicOp.ADD, mCount.iterator, cutlass.Int32(1)))
    mWorkItems[slot, 0] = batch_idx
    mWorkItems[slot, 1] = head_idx
    mWorkItems[slot, 2] = write_start
    mWorkItems[slot, 3] = write_end
    mWorkItems[slot, 4] = compute_start
    mWorkItems[slot, 5] = compute_end
    mWorkItems[slot, 6] = batch_start
    mWorkItems[slot, 7] = batch_end
    mWorkItems[slot, WORK_ITEM_FINAL_DST] = batch_idx if write_end == batch_num_chunks else cutlass.Int32(-1)
    mWorkItems[slot, WORK_ITEM_DSTATE_DST] = batch_idx if write_start == cutlass.Int32(0) else cutlass.Int32(-1)


@cute.jit
def clamped_log2(log_gate: cutlass.Constexpr[bool], gate_val: cutlass.Float32) -> cutlass.Float32:
    """Log2-domain decay increment, clamped to <= 0 (a gate > 1 must not
    relax the horizon)."""
    if cutlass.const_expr(log_gate):
        lg = gate_val * cutlass.Float32(RCP_LN2)
    else:
        lg = cute.math.log2(gate_val + cutlass.Float32(1e-10), fastmath=True)
    return lg if lg < cutlass.Float32(0.0) else cutlass.Float32(0.0)


@cute.jit
def expanded_cu_seqlen(expand_num: cutlass.Constexpr[int], mCuSeqlens, i):
    """One cumulative token boundary, ``mCuSeqlens[i]`` (``cu_seqlens`` or a ``cu_pieces`` slot boundary in the same units),
    scaled onto GDP's ``expand_num``-expanded sub-token timeline (1 folds to the plain load)."""
    v = cutlass.Int32(mCuSeqlens[i])
    if cutlass.const_expr(expand_num > 1):
        v = v * cutlass.Int32(expand_num)
    return v


@cute.jit
def piece_choice(
    overhead_chunks: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    batch_num_chunks,
    n_tiles,
    ideal_chunks,
    warmup_est,
    align: cutlass.Constexpr[int] = 1,
):
    """Per-tile piece choice.  Returns ``(span, num_blocks)``.

    ``warmup_est`` is the warmup (chunks) every piece is expected to recompute;
    it joins the per-piece cost but not the uncut tile's, which recomputes
    nothing.  ``align`` rounds the span up to a whole number of chunks per GDP
    block, so every cut ``jj * span`` is block-aligned and a work item owns
    whole blocks.  Rounding up only lowers the piece count, so the
    ``max_work_items`` bound still holds."""
    oh = cutlass.Int32(overhead_chunks) + warmup_est
    # ---- even-spread cap: no span past ideal_chunks ----------------------------------
    p_hi = batch_num_chunks if batch_num_chunks < cutlass.Int32(MAX_BLOCKS) else cutlass.Int32(MAX_BLOCKS)
    p_hi = p_hi if p_hi > 0 else cutlass.Int32(1)
    p = (batch_num_chunks + ideal_chunks - cutlass.Int32(1)) // ideal_chunks
    p = p if p > 0 else cutlass.Int32(1)
    p = p if p < p_hi else p_hi
    if n_tiles < cutlass.Int32(2) * num_sms:

        # ---- fill regime: wave-quantized search around the cap -----------------------
        p_start = p - cutlass.Int32(P_BELOW)
        p_start = p_start if p_start > 0 else cutlass.Int32(1)
        best = cutlass.Int32(2147483647)
        for dp in cutlass.range(P_WINDOW):
            candidate = p_start + cutlass.Int32(dp)
            candidate = candidate if candidate < p_hi else p_hi
            span_c = (batch_num_chunks + candidate - cutlass.Int32(1)) // candidate
            waves = (n_tiles * candidate + num_sms - cutlass.Int32(1)) // num_sms
            estimate = waves * (span_c + oh)
            hit = estimate < best
            best = estimate if hit else best
            p = candidate if hit else p

        # ---- margin over uncut -------------------------------------------------------
        uncut_estimate = ((n_tiles + num_sms - cutlass.Int32(1)) // num_sms) * (batch_num_chunks + cutlass.Int32(overhead_chunks))
        if cutlass.Int32(4) * best > cutlass.Int32(3) * uncut_estimate:
            p = cutlass.Int32(1)
    span = cutlass.Int32(0)
    num_blocks = cutlass.Int32(0)
    if batch_num_chunks > 0:
        span = (batch_num_chunks + p - cutlass.Int32(1)) // p
        if cutlass.const_expr(align > 1):
            span = ((span + cutlass.Int32(align - 1)) // cutlass.Int32(align)) * cutlass.Int32(align)
        num_blocks = (batch_num_chunks + span - cutlass.Int32(1)) // span
    return span, num_blocks


@cute.jit
def common_span(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    n_tiles,
    ideal_chunks,
    batch_size,
    mCuSeqlens,
):
    """Span for :func:`span_choice` to force on every tile the per-sequence
    choice left uncut, or 0 to keep that choice.

    :func:`piece_choice` scores a piece count with ``waves = n_tiles *
    candidate``, which assumes every tile splits into ``candidate`` pieces; in
    a ragged batch the sequence that sets the makespan under-splits.  A span
    shared by the batch has the exact CTA count ``sum_b ceil(nc_b / span)``,
    which is what this search scores; the best span is taken when it beats the
    uncut estimate by the 3/4 margin.  Applies only when the batch is ragged
    and the per-sequence choice's grid underfills the SMs.  Batch-global,
    evaluated once per kernel and combined per tile.  The span carries
    :func:`piece_choice`'s ``expand_num`` alignment."""
    chosen = cutlass.Int32(0)
    if n_tiles < num_sms:

        # ---- ragged batch test -------------------------------------------------------
        max_chunks = cutlass.Int32(0)
        min_chunks = cutlass.Int32(2147483647)
        b = cutlass.Int32(0)
        while b < batch_size:
            nc = cute.ceil_div(expanded_cu_seqlen(expand_num, mCuSeqlens, b + 1) - expanded_cu_seqlen(expand_num, mCuSeqlens, b), b_t)
            max_chunks = max_chunks if max_chunks > nc else nc
            min_chunks = min_chunks if min_chunks < nc else nc
            b = b + cutlass.Int32(1)

        if min_chunks < max_chunks:

            # ---- grid of the per-sequence choice -------------------------------------
            blocks = cutlass.Int32(0)
            b = cutlass.Int32(0)
            while b < batch_size:
                nc = cute.ceil_div(expanded_cu_seqlen(expand_num, mCuSeqlens, b + 1) - expanded_cu_seqlen(expand_num, mCuSeqlens, b), b_t)
                _, nb = piece_choice(overhead_chunks, num_sms, nc, n_tiles, ideal_chunks, cutlass.Int32(0), expand_num)
                blocks = blocks + nb
                b = b + cutlass.Int32(1)

            # ---- underfilled: search a span shared by the batch ----------------------
            if blocks * n_heads_out < num_sms:
                best = cutlass.Int32(2147483647)
                common = max_chunks
                for dp in cutlass.range(P_WINDOW):
                    s = (max_chunks + cutlass.Int32(dp)) // (cutlass.Int32(dp) + cutlass.Int32(1))
                    if cutlass.const_expr(expand_num > 1):
                        s = ((s + cutlass.Int32(expand_num - 1)) // cutlass.Int32(expand_num)) * cutlass.Int32(expand_num)
                    ctas = cutlass.Int32(0)
                    b = cutlass.Int32(0)
                    while b < batch_size:
                        nc = cute.ceil_div(expanded_cu_seqlen(expand_num, mCuSeqlens, b + 1) - expanded_cu_seqlen(expand_num, mCuSeqlens, b), b_t)
                        ctas = ctas + cute.ceil_div(nc, s)
                        b = b + cutlass.Int32(1)
                    ctas = ctas * n_heads_out
                    waves = (ctas + num_sms - cutlass.Int32(1)) // num_sms
                    estimate = waves * (s + cutlass.Int32(overhead_chunks))
                    hit = estimate < best
                    best = estimate if hit else best
                    common = s if hit else common
                uncut = ((n_tiles + num_sms - cutlass.Int32(1)) // num_sms) * (max_chunks + cutlass.Int32(overhead_chunks))
                chosen = common if cutlass.Int32(4) * best <= cutlass.Int32(3) * uncut else cutlass.Int32(0)
    return chosen


@cute.jit
def span_choice(
    overhead_chunks: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    batch_num_chunks,
    n_tiles,
    ideal_chunks,
    common,
    warmup_est,
    align: cutlass.Constexpr[int] = 1,
):
    """Per-tile piece choice, overridden by the batch-wide ``common`` span
    from :func:`common_span` (0 when it does not apply) whenever the
    per-sequence choice left this tile in one piece.  ``warmup_est`` and
    ``align`` are :func:`piece_choice`'s; ``common`` already carries the
    alignment.  Returns ``(span, num_blocks)``."""
    span, num_blocks = piece_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, warmup_est, align)
    if (n_tiles < num_sms) and (num_blocks == cutlass.Int32(1)) and (common > cutlass.Int32(0)):
        span = common if common < batch_num_chunks else batch_num_chunks
        num_blocks = cute.ceil_div(batch_num_chunks, span) if span > 0 else cutlass.Int32(0)
    return span, num_blocks


@cute.jit
def near_boundary(full_scan: cutlass.Constexpr[bool], window: cutlass.Constexpr[int], c, span, num_blocks):
    """True iff chunk ``c`` lies in the walk's read window of a candidate
    boundary: suffix ``[j*span - window, j*span)`` or prefix ``[j*span, j*span
    + window)`` for some ``j`` in ``[1, num_blocks)``.  ``window`` spans the
    warmup cap plus the snap search.  With ``full_scan`` every chunk of the
    tile is read: the walk then also sums the whole tile."""
    if cutlass.const_expr(full_scan):
        return c >= cutlass.Int32(0)
    j0 = c // span
    cm = c - j0 * span
    w = cutlass.Int32(window)
    pre = (cm < w) and (j0 >= cutlass.Int32(1))
    suf = (span - cm <= w) and (j0 < num_blocks - cutlass.Int32(1))
    return pre or suf


@cute.jit
def window_prefix(window_len: cutlass.Constexpr[int], sPre, base, lo_c, n_w, lane_idx, chunk_value_base, head_idx, mChunkVals):
    """Inclusive prefix of the chunk values ``[lo_c, lo_c + n_w)`` into
    ``sPre[base + 1 .. base + n_w]`` with ``sPre[base] = 0``.  The window is
    staged lane-strided, then each lane sums a contiguous run and the run
    totals are scanned across the warp."""
    for kk in cutlass.range_constexpr((window_len + WARP_SIZE - 1) // WARP_SIZE):
        i = cutlass.Int32(kk * WARP_SIZE) + lane_idx
        if i < n_w:
            sPre[base + cutlass.Int32(1) + i] = mChunkVals[chunk_value_base + lo_c + i, head_idx]
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    run = (n_w + cutlass.Int32(WARP_SIZE - 1)) // cutlass.Int32(WARP_SIZE)
    start = lane_idx * run
    end = start + run
    end = end if end < n_w else n_w
    acc = cutlass.Float32(0.0)
    i = start
    while i < end:
        acc = acc + sPre[base + cutlass.Int32(1) + i]
        sPre[base + cutlass.Int32(1) + i] = acc
        i = i + cutlass.Int32(1)
    s = acc
    for off in [1, 2, 4, 8, 16]:
        o = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, s, off, 0, kind=nvvm.Shfl.UP))
        s = s + (o if lane_idx >= cutlass.Int32(off) else cutlass.Float32(0.0))
    excl = s - acc
    i = start
    while i < end:
        sPre[base + cutlass.Int32(1) + i] = sPre[base + cutlass.Int32(1) + i] + excl
        i = i + cutlass.Int32(1)
    if lane_idx == 0:
        sPre[base] = cutlass.Float32(0.0)
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)


@cute.jit
def search_fwd(sPre, base, xi, cap: cutlass.Constexpr[int], thresh):
    """Forward state warmup at window position ``xi``: the smallest ``s`` in
    ``[1, min(cap, xi)]`` whose chunk suffix ``[xi - s, xi)`` saturates, 0 if
    none.  The prefix is non-increasing, so a binary search finds it."""
    hi = cutlass.Int32(cap) if cutlass.Int32(cap) < xi else xi
    result = cutlass.Int32(0)
    if hi > 0:
        px = sPre[base + xi]
        if px - sPre[base + xi - hi] <= thresh:
            lo = cutlass.Int32(1)
            while lo < hi:
                mid = (lo + hi) // cutlass.Int32(2)
                ok = px - sPre[base + xi - mid] <= thresh
                hi = mid if ok else hi
                lo = lo if ok else mid + cutlass.Int32(1)
            result = lo
    return result


@cute.jit
def search_bwd(sPre, base, xi, n_w, cap: cutlass.Constexpr[int], thresh):
    """Reverse dstate warmup at window position ``xi``: the smallest ``s`` in
    ``[1, min(cap, n_w - xi)]`` whose chunk prefix ``[xi, xi + s)``
    saturates, 0 if none."""
    limit = n_w - xi
    hi = cutlass.Int32(cap) if cutlass.Int32(cap) < limit else limit
    result = cutlass.Int32(0)
    if hi > 0:
        px = sPre[base + xi]
        if sPre[base + xi + hi] - px <= thresh:
            lo = cutlass.Int32(1)
            while lo < hi:
                mid = (lo + hi) // cutlass.Int32(2)
                ok = sPre[base + xi + mid] - px <= thresh
                hi = mid if ok else hi
                lo = lo if ok else mid + cutlass.Int32(1)
            result = lo
    return result


@cute.jit
def read_plan(mChunkVals):
    """The batch-wide span :func:`frost_split_k_plan` left in the scratch's
    last row, or -1 when no tile of this batch can be cut.  Spans are small
    integers, exact in fp32."""
    return mChunkVals[mChunkVals.shape[0] - cutlass.Int32(1), 0].to(cutlass.Int32)


@cute.kernel
def frost_split_k_plan(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """One CTA: settle the batch-wide span once and leave it in the chunk
    scratch's last row for the scan and the walk, -1 there when no tile of the
    batch admits a cut, and zero the item count.  Both answers are batch-global;
    the scan and the walk read them."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    sCut = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=8)
    if tidx == 0:
        sCut[0] = cutlass.Int32(0)
    nvvm.barrier_cta_sync()

    # ---- cut admissibility over the batch: one thread per entry ----------------------
    cut = cutlass.Int32(0)
    b = tidx
    while b < batch_size:
        nc = cute.ceil_div(expanded_cu_seqlen(expand_num, mCuSeqlens, b + 1) - expanded_cu_seqlen(expand_num, mCuSeqlens, b), b_t)
        _, nb = piece_choice(overhead_chunks, num_sms, nc, n_tiles, ideal_chunks, cutlass.Int32(0), expand_num)
        cut = cutlass.Int32(1) if nb > cutlass.Int32(1) else cut
        b = b + cutlass.Int32(PLAN_THREADS)
    nvvm.atomicrmw("max", sCut.data_ptr(0), cut, space=nvvm.SharedSpace.shared_cta)
    nvvm.barrier_cta_sync()

    if tidx == 0:
        mCount[0] = cutlass.Int32(0)
        any_cut = sCut[0]
        common = common_span(b_t, overhead_chunks, expand_num, n_heads_out, num_sms, n_tiles, ideal_chunks, batch_size, mCuSeqlens)
        any_cut = cutlass.Int32(1) if common > cutlass.Int32(0) else any_cut
        plan = cutlass.Float32(common) if any_cut > cutlass.Int32(0) else cutlass.Float32(-1.0)
        mChunkVals[mChunkVals.shape[0] - cutlass.Int32(1), 0] = plan


@cute.kernel
def frost_split_k_scan_channel(
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    full_scan: cutlass.Constexpr[bool],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    n_scan_blocks: cutlass.Int32,
    gate_scale_log2: cutlass.Float32,
    mGate: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """Windowed chunk scan, per-channel gate (KDA / GDN-2): CTA ``(x, h)``
    covers 16 chunk-scratch rows of head ``h``, one warp per chunk, lane
    ``l`` owning channels ``[l*cpl, (l+1)*cpl)``.  Chunks outside a cut
    window never touch the gate.  The item count is the plan kernel's."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()
    tidx = cutlass.Int32(tidx)
    head_idx = cutlass.Int32(bidx[1])
    lane_idx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    common = read_plan(mChunkVals)
    if common >= cutlass.Int32(0):
        blk = cutlass.Int32(bidx[0])
        while blk < n_scan_blocks:
            row0 = (blk * cutlass.Int32(SCAN_WARPS) + widx) * cutlass.Int32(scan_rows)

            lo = cutlass.Int32(0)
            hi = batch_size - cutlass.Int32(1)
            while lo < hi:
                mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
                take = expanded_cu_seqlen(expand_num, mCuSeqlens, mid) // cutlass.Int32(b_t) + mid <= row0
                lo = mid if take else lo
                hi = hi if take else mid - cutlass.Int32(1)
            batch_idx = lo
            batch_start = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx)
            batch_end = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx + 1)
            batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
            chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
            span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common, cutlass.Int32(0), expand_num)

            for rr in cutlass.range_constexpr(scan_rows):
                row = row0 + cutlass.Int32(rr)
                while (batch_idx + cutlass.Int32(1) < batch_size) and (
                    expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx + 1) // cutlass.Int32(b_t) + batch_idx + cutlass.Int32(1) <= row
                ):
                    batch_idx = batch_idx + cutlass.Int32(1)
                    batch_start = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx)
                    batch_end = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx + 1)
                    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
                    chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
                    span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common, cutlass.Int32(0), expand_num)
                c = row - chunk_value_base
                if (c >= 0) and (c < batch_num_chunks) and (num_blocks > 1):
                    if near_boundary(full_scan, warmup_cap + SNAP_CHUNKS * expand_num, c, span if span > 0 else cutlass.Int32(1), num_blocks):
                        # ---- per-channel clamped-log2 sums, one channel run per lane -
                        cpl = gate_channels // WARP_SIZE
                        row_elements = cutlass.Int32(mGate.stride[0])
                        lane_base = cutlass.Int64(head_idx * cutlass.Int32(mGate.stride[1]) + lane_idx * cutlass.Int32(cpl))
                        gate_elem_bytes = cutlass.const_expr(mGate.element_type.width // 8)
                        gate_addr = mGate.iterator.toint() + lane_base * cutlass.Int64(gate_elem_bytes)
                        gate_ptr = mGate.iterator + lane_base
                        a_exp = cutlass.Float32(1.0)
                        dt_vals = cutlass.Array(cutlass.Float32, cpl)
                        if cutlass.const_expr(safe_gate):
                            if cutlass.const_expr(mALog is not None):
                                a_exp = cute.math.exp2(mALog[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True)
                            else:
                                a_exp = opaque_f32_zero() + cutlass.Float32(1.0)
                            for q in cutlass.range_constexpr(cpl):
                                if cutlass.const_expr(mDtBias is not None):
                                    dt_vals[q] = mDtBias[head_idx, lane_idx * cutlass.Int32(cpl) + cutlass.Int32(q)].to(cutlass.Float32)
                                else:
                                    dt_vals[q] = opaque_f32_zero()
                        ch_acc = cutlass.Array(cutlass.Float32, cpl, alignment=16)
                        for q in cutlass.range_constexpr(cpl):
                            ch_acc[q] = cutlass.Float32(0.0)
                        oob = cutlass.Float32(0.0) if cutlass.const_expr(log_gate) else cutlass.Float32(1.0)
                        for tt in cutlass.range(0, b_t, SCAN_TOKEN_STRIDE, unroll_full=True):
                            pos = batch_start + c * cutlass.Int32(b_t) + cutlass.Int32(tt)
                            inb = pos < batch_end
                            pos_r = pos if inb else batch_start
                            grow = cutlass.Int64(pos_r) * cutlass.Int64(row_elements)
                            if cutlass.const_expr(cpl % 4 == 0):
                                for q4 in cutlass.range_constexpr(cpl // 4):
                                    addr = gate_addr + (grow + cutlass.Int64(4 * q4)) * cutlass.Int64(gate_elem_bytes)
                                    if cutlass.const_expr(mGate.element_type == cutlass.Float32):
                                        quad = ld_global_v4(addr, cutlass.Float32)
                                    else:
                                        w0, w1 = ld_global_v2(addr, cutlass.Int32)
                                        frag = cutlass.Vector.from_elements((w0, w1), cutlass.Int32).bitcast(mGate.element_type).to(cutlass.Float32)
                                        quad = (frag[0], frag[1], frag[2], frag[3])
                                    for qq, gvq in enumerate(quad):
                                        q = 4 * q4 + qq
                                        if cutlass.const_expr(safe_gate):
                                            sig = sigmoid(a_exp * (gvq + dt_vals[q]))
                                            contrib = gate_scale_log2 * sig
                                            contrib = contrib if inb else cutlass.Float32(0.0)
                                        else:
                                            gvq = gvq if inb else oob
                                            contrib = clamped_log2(log_gate, gvq)
                                        ch_acc[q] = ch_acc[q] + contrib
                            else:
                                for q in cutlass.range_constexpr(cpl):
                                    gv = (gate_ptr + grow + cutlass.Int32(q)).load().to(cutlass.Float32)
                                    if cutlass.const_expr(safe_gate):
                                        sig = sigmoid(a_exp * (gv + dt_vals[q]))
                                        contrib = gate_scale_log2 * sig
                                        contrib = contrib if inb else cutlass.Float32(0.0)
                                    else:
                                        gv = gv if inb else oob
                                        contrib = clamped_log2(log_gate, gv)
                                    ch_acc[q] = ch_acc[q] + contrib

                        # ---- chunk value = max over channels, then over lanes --------
                        m = ch_acc[0]
                        for q in cutlass.range_constexpr(1, cpl):
                            m = m if m > ch_acc[q] else ch_acc[q]
                        for off in [1, 2, 4, 8, 16]:
                            other = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, m, off, 31, kind=nvvm.Shfl.BFLY))
                            m = m if m > other else other
                        if lane_idx == 0:
                            mChunkVals[chunk_value_base + c, head_idx] = m
            blk = blk + cutlass.Int32(cute.arch.grid_dim()[0])
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.kernel
def frost_split_k_scan_scalar(
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    full_scan: cutlass.Constexpr[bool],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    n_scan_blocks: cutlass.Int32,
    mGate: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """Windowed chunk scan, scalar gate (GDN): CTA ``(x, hg)`` covers 16
    chunk-scratch rows for heads ``[hg*32, (hg+1)*32)``; lane ``l`` owns head
    ``hg*32 + l``, so reads and writes coalesce and no reduction is needed.
    With ``safe_gate`` each token contributes ``-exp(A_log[h]) * softplus(g +
    dt_bias[h]) * RCP_LN2``; an absent ``mALog`` is unit amplitude, an absent
    ``mDtBias`` zero bias.  The item count is the plan kernel's."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()
    tidx = cutlass.Int32(tidx)
    lane_idx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    h = cutlass.Int32(bidx[1]) * cutlass.Int32(WARP_SIZE) + lane_idx
    h_ok = h < n_heads_out
    h_r = h if h_ok else cutlass.Int32(n_heads_out - 1)
    a = cutlass.Float32(0.0)
    bias = cutlass.Float32(0.0)
    if cutlass.const_expr(safe_gate):
        if cutlass.const_expr(mALog is not None):
            a = -cute.math.exp2(mALog[h_r].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True) * cutlass.Float32(RCP_LN2)
        else:
            a = -(opaque_f32_zero() + cutlass.Float32(1.0)) * cutlass.Float32(RCP_LN2)
        if cutlass.const_expr(mDtBias is not None):
            bias = mDtBias[h_r].to(cutlass.Float32)
        else:
            bias = opaque_f32_zero()
    common = read_plan(mChunkVals)
    if common >= cutlass.Int32(0):
        blk = cutlass.Int32(bidx[0])
        while blk < n_scan_blocks:
            row0 = (blk * cutlass.Int32(SCAN_WARPS) + widx) * cutlass.Int32(scan_rows)

            lo = cutlass.Int32(0)
            hi = batch_size - cutlass.Int32(1)
            while lo < hi:
                mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
                take = expanded_cu_seqlen(expand_num, mCuSeqlens, mid) // cutlass.Int32(b_t) + mid <= row0
                lo = mid if take else lo
                hi = hi if take else mid - cutlass.Int32(1)
            batch_idx = lo
            batch_start = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx)
            batch_end = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx + 1)
            batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
            chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
            span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common, cutlass.Int32(0), expand_num)

            for rr in cutlass.range_constexpr(scan_rows):
                row = row0 + cutlass.Int32(rr)
                while (batch_idx + cutlass.Int32(1) < batch_size) and (
                    expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx + 1) // cutlass.Int32(b_t) + batch_idx + cutlass.Int32(1) <= row
                ):
                    batch_idx = batch_idx + cutlass.Int32(1)
                    batch_start = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx)
                    batch_end = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx + 1)
                    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
                    chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
                    span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common, cutlass.Int32(0), expand_num)
                c = row - chunk_value_base
                if (c >= 0) and (c < batch_num_chunks) and (num_blocks > 1):
                    if near_boundary(full_scan, warmup_cap + SNAP_CHUNKS * expand_num, c, span if span > 0 else cutlass.Int32(1), num_blocks):
                        # ---- clamped-log2 sum over the chunk, one head per lane ------
                        oob = cutlass.Float32(0.0) if cutlass.const_expr(log_gate) else cutlass.Float32(1.0)
                        acc = cutlass.Float32(0.0)
                        for tt in cutlass.range(0, b_t, SCAN_TOKEN_STRIDE, unroll_full=True):
                            pos = batch_start + c * cutlass.Int32(b_t) + cutlass.Int32(tt)
                            inb = pos < batch_end
                            pos_r = pos if inb else batch_start
                            if cutlass.const_expr(expand_num > 1):
                                row_r = pos_r // cutlass.Int32(expand_num)
                                inb = inb and (pos_r - row_r * cutlass.Int32(expand_num) == 0)
                                pos_r = row_r
                            gv = (mGate.iterator + cutlass.Int64(pos_r) * cutlass.Int64(mGate.stride[0]) + h_r).load().to(cutlass.Float32)
                            if cutlass.const_expr(safe_gate):
                                contrib = a * softplus(gv + bias)
                                acc = acc + (contrib if inb else cutlass.Float32(0.0))
                            else:
                                gv = gv if inb else oob
                                acc = acc + clamped_log2(log_gate, gv)
                        if h_ok:
                            mChunkVals[chunk_value_base + c, h] = acc
            blk = blk + cutlass.Int32(cute.arch.grid_dim()[0])
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.kernel
def frost_split_k_walk(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    full_scan: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mStaging: cute.Tensor,
    mCount: cute.Tensor,
):
    """One CTA per (batch, head).  With ``full_scan`` the CTA first sums the
    tile's chunk values: a tile that does not saturate even as a whole stays
    uncut, otherwise the mean decay gives the warmup every piece will
    recompute and the piece count is re-chosen with that cost priced in.
    Then every candidate boundary is probed by its own warp: the warp
    prefix-sums the boundary's chunk window into SMEM and one lane per snap
    offset binary-searches the smallest saturating warmup on each side.
    Thread 0 emits every saturating cut."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    tile = cutlass.Int32(cute.arch.block_idx()[0])
    lane_idx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    zero = cutlass.Int32(0)
    step = cutlass.const_expr(expand_num)
    window_len = cutlass.const_expr(2 * (warmup_cap + SNAP_CHUNKS * expand_num) + 1)
    sPartial = cutlass.Array(cutlass.Float32, WARPS, space=cutlass.AddressSpace.smem, alignment=16)
    sWarmup = cutlass.Array(cutlass.Int32, MAX_BLOCKS, space=cutlass.AddressSpace.smem, alignment=16)
    sPre = cutlass.Array(cutlass.Float32, WARPS * window_len, space=cutlass.AddressSpace.smem, alignment=16)

    common = read_plan(mChunkVals)

    # ---- tile decode: chunk_value_base is the row base in the chunk scratch ----------
    batch_idx = tile // n_heads_out
    head_idx = tile % n_heads_out
    batch_start = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx)
    batch_end = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx + 1)
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    chunk_value_base = batch_start // cutlass.Int32(b_t) + batch_idx
    span = cutlass.Int32(0)
    num_blocks = cutlass.Int32(1)
    if common >= cutlass.Int32(0):
        span, num_blocks = span_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common, zero, expand_num)

    if cutlass.const_expr(full_scan):
        if num_blocks > 1:
            # ---- whole-tile decay: no cut can saturate if the tile does not ---------------
            acc = cutlass.Float32(0.0)
            c = tidx
            while c < batch_num_chunks:
                acc = acc + mChunkVals[chunk_value_base + c, head_idx]
                c = c + cutlass.Int32(THREADS_PER_BLOCK)
            for off in [16, 8, 4, 2, 1]:
                acc = acc + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, acc, off, 31, kind=nvvm.Shfl.BFLY))
            if lane_idx == 0:
                sPartial[widx] = acc
            nvvm.barrier_cta_sync()
            total = cutlass.Float32(0.0)
            for w in cutlass.range_constexpr(WARPS):
                total = total + sPartial[w]
            if total > log2_thresh:
                num_blocks = cutlass.Int32(1)
            else:
                # ---- mean decay -> chunks per warmup; re-choose the pieces with it priced in
                w_f = (batch_num_chunks.to(cutlass.Float32) * log2_thresh) / total
                w_est = (w_f + cutlass.Float32(0.999)).to(cutlass.Int32)
                w_est = w_est if w_est > 0 else cutlass.Int32(1)
                _, nb0 = piece_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, zero, expand_num)
                span_w, nb_w = piece_choice(overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, w_est, expand_num)
                if (nb0 == cutlass.Int32(1)) and (common > cutlass.Int32(0)):
                    # ---- ragged underfill override under the 3/4 margin ------------------
                    s_c = common if common < batch_num_chunks else batch_num_chunks
                    oh = cutlass.Int32(overhead_chunks)
                    if cutlass.Int32(4) * (s_c + oh + w_est) <= cutlass.Int32(3) * (batch_num_chunks + oh):
                        span_w = s_c
                        nb_w = cute.ceil_div(batch_num_chunks, s_c)
                    else:
                        nb_w = cutlass.Int32(1)
                span = span_w
                num_blocks = nb_w

    if num_blocks <= 1:
        # ---- nothing to cut: emit the whole sequence ---------------------------------
        if tidx == 0:
            emit_item(
                mStaging,
                mCount,
                batch_idx,
                head_idx,
                cutlass.Int32(0),
                batch_num_chunks,
                cutlass.Int32(0),
                batch_num_chunks,
                batch_start,
                batch_end,
                batch_num_chunks,
            )

    else:
        # ---- probe every candidate boundary: one warp each -----------------------------
        big = cutlass.Int32(2147483647)
        cap = cutlass.Int32(warmup_cap)
        half = (span // cutlass.Int32(2)) // cutlass.Int32(step)
        half = half if half < cutlass.Int32(SNAP_CHUNKS) else cutlass.Int32(SNAP_CHUNKS)
        base = widx * cutlass.Int32(window_len)
        j = widx + cutlass.Int32(1)
        while j < num_blocks:
            target = j * span
            lo_c = target - half * cutlass.Int32(step) - cap
            lo_c = lo_c if lo_c > 0 else cutlass.Int32(0)
            hi_c = target + half * cutlass.Int32(step) + cap
            hi_c = hi_c if hi_c < batch_num_chunks else batch_num_chunks
            n_w = hi_c - lo_c
            window_prefix(window_len, sPre, base, lo_c, n_w, lane_idx, chunk_value_base, head_idx, mChunkVals)

            # ---- one lane per snap offset, |d| order 0, +1, -1, +2, -2, ... ------------
            best_key = big
            best_pack = cutlass.Int32(0)
            kmin = big
            n_off = cutlass.Int32(2) * half + cutlass.Int32(1)
            i = lane_idx
            searching = True
            while searching:
                ad = (i + cutlass.Int32(1)) // cutlass.Int32(2)
                neg = i % cutlass.Int32(2) == cutlass.Int32(0)
                d = -ad if neg else ad
                x = target + d * cutlass.Int32(step)
                in_range = (x > cutlass.Int32(0)) and (x < batch_num_chunks)
                xi = (x if in_range else target) - lo_c
                wb = search_fwd(sPre, base, xi, warmup_cap, log2_thresh)
                wa = search_bwd(sPre, base, xi, n_w, warmup_cap, log2_thresh)
                key = ((wb + wa + ad) * cutlass.Int32(64) + ad) * cutlass.Int32(2) + (cutlass.Int32(1) if neg else cutlass.Int32(0))
                live = in_range and (i < n_off) and (wb > 0) and (wa > 0) and (key < best_key)
                best_key = key if live else best_key
                best_pack = (wb + wa * cutlass.Int32(1024) + (d + cutlass.Int32(SNAP_CHUNKS)) * cutlass.Int32(1048576)) if live else best_pack
                i = i + cutlass.Int32(WARP_SIZE)
                # ---- prune: the next pass's offsets cannot beat a warmup pair found closer -----
                kmin = best_key
                for off in [16, 8, 4, 2, 1]:
                    other = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, kmin, off, 31, kind=nvvm.Shfl.BFLY))
                    kmin = kmin if kmin < other else other
                ad_next = (i - lane_idx + cutlass.Int32(1)) // cutlass.Int32(2)
                floor_next = (cutlass.Int32(2) + ad_next) * cutlass.Int32(128)
                searching = (i - lane_idx < n_off) and (floor_next < kmin)
            if (kmin < big) and (best_key == kmin):
                sWarmup[j] = best_pack
            if (kmin == big) and (lane_idx == 0):
                sWarmup[j] = cutlass.Int32(0)
            j = j + cutlass.Int32(WARPS)
        nvvm.barrier_cta_sync()

        # ---- emit: thread 0 walks the accepted cuts in order -------------------------
        if tidx == 0:
            prev_cut = cutlass.Int32(0)
            current_compute_start = cutlass.Int32(0)
            jj = cutlass.Int32(1)
            while jj < num_blocks:
                r = sWarmup[jj]
                if r != 0:
                    warmup_before = r % cutlass.Int32(1024)
                    warmup_after = (r // cutlass.Int32(1024)) % cutlass.Int32(1024)
                    write_end = jj * span + (r // cutlass.Int32(1048576) - cutlass.Int32(SNAP_CHUNKS)) * cutlass.Int32(step)
                    compute_end = write_end + warmup_after
                    compute_end = compute_end if compute_end < batch_num_chunks else batch_num_chunks
                    emit_item(
                        mStaging, mCount, batch_idx, head_idx, prev_cut, write_end, current_compute_start, compute_end, batch_start, batch_end, batch_num_chunks
                    )
                    current_compute_start = write_end - warmup_before
                    prev_cut = write_end
                jj = jj + cutlass.Int32(1)
            emit_item(
                mStaging,
                mCount,
                batch_idx,
                head_idx,
                prev_cut,
                batch_num_chunks,
                current_compute_start,
                batch_num_chunks,
                batch_start,
                batch_end,
                batch_num_chunks,
            )
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def gen_item_bounds(b_t: cutlass.Constexpr[int], n_heads_out, mCuSeqlens, item, expand_num: cutlass.Constexpr[int] = 1):
    """(batch, head, batch_start, batch_end, num_chunks) of the uncut
    whole-sequence item ``item``, a no-cuts table row is pure geometry."""
    batch_idx = item // n_heads_out
    head_idx = item % n_heads_out
    batch_start = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx)
    batch_end = expanded_cu_seqlen(expand_num, mCuSeqlens, batch_idx + 1)
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    return batch_idx, head_idx, batch_start, batch_end, batch_num_chunks


@cute.jit
def piece_item_bounds(
    b_t: cutlass.Constexpr[int],
    n_heads_out,
    mCuPieces,
    mRowBase,
    mSlotRows,
    item,
    expand_num: cutlass.Constexpr[int] = 1,
):
    """(slot, head, row_start, row_end, num_chunks, final_dst, dstate_dst) of piece-table
    row ``item``: the owning sequence by binary search over ``mRowBase`` (each sequence's
    first row in this table, ``[num_seqs + 1]``, the chain prologue's piece table), then the
    slot and head within its block of ``filled_slots * n_heads_out`` rows; the flat slot is
    ``item // n_heads_out`` when this IS the main table (``mSlotRows`` None) and otherwise
    the sequence's first slot ``mSlotRows[seq] // n_heads_out`` (the main table's row bases)
    plus the slot within the sequence; the kernel rows are the slot's ``cu_pieces``; the
    last filled slot stores ``final_state``, slot 0 ``d_initial_state``."""
    lo = cutlass.Int32(0)
    hi = cutlass.Int32(mRowBase.shape[0]) - cutlass.Int32(2)
    while lo < hi:
        mid = (lo + hi + cutlass.Int32(1)) >> cutlass.Int32(1)
        if mRowBase[mid] <= item:
            lo = mid
        else:
            hi = mid - cutlass.Int32(1)
    batch_idx = lo
    local = item - mRowBase[batch_idx]
    slot_in_seq = local // n_heads_out
    head_idx = local - slot_in_seq * n_heads_out
    last = (mRowBase[batch_idx + 1] - mRowBase[batch_idx]) // n_heads_out - cutlass.Int32(1)
    if cutlass.const_expr(mSlotRows is None):
        slot = item // n_heads_out
    else:
        slot = mSlotRows[batch_idx] // n_heads_out + slot_in_seq
    row_start = expanded_cu_seqlen(expand_num, mCuPieces, slot)
    row_end = expanded_cu_seqlen(expand_num, mCuPieces, slot + 1)
    num_chunks = cute.ceil_div(row_end - row_start, b_t)
    final_dst = batch_idx if slot_in_seq == last else cutlass.Int32(-1)
    dstate_dst = batch_idx if slot_in_seq == cutlass.Int32(0) else cutlass.Int32(-1)
    return slot, head_idx, row_start, row_end, num_chunks, final_dst, dstate_dst


@cute.jit
def write_item(
    gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    n_heads_out,
    mCuSeqlens,
    mStaging,
    mWorkItems,
    dst,
    src,
    expand_num: cutlass.Constexpr[int] = 1,
    chain: cutlass.Constexpr[bool] = False,
    mRowBase=None,
    mSlotRows=None,
):
    """Final-table row ``dst`` from source item ``src``: the walk's staged row, (``gen``) the
    synthesized uncut item ``(0, nc, 0, nc)``, or (``chain``) the piece item regenerated from
    ``mCuSeqlens`` (then the piece-wise ``cu_pieces``), ``mRowBase`` and ``mSlotRows``."""
    if cutlass.const_expr(chain):
        slot, head_idx, row_start, row_end, num_chunks, final_dst, dstate_dst = piece_item_bounds(
            b_t, n_heads_out, mCuSeqlens, mRowBase, mSlotRows, src, expand_num
        )
        mWorkItems[dst, 0] = slot
        mWorkItems[dst, 1] = head_idx
        mWorkItems[dst, 2] = cutlass.Int32(0)
        mWorkItems[dst, 3] = num_chunks
        mWorkItems[dst, 4] = cutlass.Int32(0)
        mWorkItems[dst, 5] = num_chunks
        mWorkItems[dst, 6] = row_start
        mWorkItems[dst, 7] = row_end
        mWorkItems[dst, WORK_ITEM_FINAL_DST] = final_dst
        mWorkItems[dst, WORK_ITEM_DSTATE_DST] = dstate_dst
    elif cutlass.const_expr(gen):
        batch_idx, head_idx, batch_start, batch_end, batch_num_chunks = gen_item_bounds(b_t, n_heads_out, mCuSeqlens, src, expand_num)
        mWorkItems[dst, 0] = batch_idx
        mWorkItems[dst, 1] = head_idx
        mWorkItems[dst, 2] = cutlass.Int32(0)
        mWorkItems[dst, 3] = batch_num_chunks
        mWorkItems[dst, 4] = cutlass.Int32(0)
        mWorkItems[dst, 5] = batch_num_chunks
        mWorkItems[dst, 6] = batch_start
        mWorkItems[dst, 7] = batch_end
        mWorkItems[dst, WORK_ITEM_FINAL_DST] = batch_idx
        mWorkItems[dst, WORK_ITEM_DSTATE_DST] = batch_idx
    else:
        for f in cutlass.range_constexpr(WORK_ITEM_FIELDS):
            mWorkItems[dst, cutlass.Int32(f)] = mStaging[src, cutlass.Int32(f)]


@cute.jit
def gen_interval_items(
    b_t: cutlass.Constexpr[int],
    n_threads: cutlass.Constexpr[int],
    tidx,
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    span_chunks: cutlass.Int32,
    mCuSeqlens: cute.Tensor,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor | None,
    expand_num: cutlass.Constexpr[int] = 1,
):
    """Checkpoint-seeded item synthesis: one work item per ``span_chunks``
    chunks of every (batch, head) tile, ``compute_start == write_start``
    (checkpoint-seeded, no warmup).  The span is a whole number of checkpoint
    intervals, so every item starts on a checkpoint row; the caller sizes it by
    the SM count.  No LPT sort; emission order is the atomic-counter order."""
    if tidx == 0:
        mCount[0] = cutlass.Int32(0)
        if cutlass.const_expr(mScheduler is not None):
            si = cutlass.Int32(0)
            while si < mScheduler.shape[0]:
                mScheduler[si] = cutlass.Int32(0)
                si = si + cutlass.Int32(1)
    nvvm.barrier_cta_sync()
    item = cutlass.Int32(tidx)
    while item < n_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_num_chunks = gen_item_bounds(b_t, n_heads_out, mCuSeqlens, item, expand_num)
        j = cutlass.Int32(0)
        while j < batch_num_chunks:
            j_end = j + span_chunks
            j_end = j_end if j_end < batch_num_chunks else batch_num_chunks
            emit_item(mWorkItems, mCount, batch_idx, head_idx, j, j_end, j, j_end, batch_start, batch_end, batch_num_chunks)
            j = j + span_chunks
        item = item + cutlass.Int32(n_threads)


@cute.jit
def compact_sequences(n_threads: cutlass.Constexpr[int], tidx, n_batch: cutlass.Int32, mCuSeqlens: cute.Tensor, sIdx, sWarp):
    """Attention Gym addition: write, in sequence order, the ids of sequences with tokens into
    ``sIdx[0, M)`` and return ``M``.  A warp ballot ranks each round of ``n_threads`` sequences, so
    the table is deterministic.  ``sWarp`` holds ``n_threads // 32`` Int32 cells."""
    lane = cute.arch.lane_idx()
    warp = tidx // cutlass.Int32(32)
    base = cutlass.Int32(0)
    start = cutlass.Int32(0)
    while start < n_batch:
        b = start + tidx
        keep = cutlass.Boolean(False)
        if b < n_batch:
            keep = mCuSeqlens[b + 1] > mCuSeqlens[b]
        ballot = cute.arch.vote_ballot_sync(keep)
        rank = cutlass.Int32(cute.arch.popc(cutlass.Uint32(ballot) & cute.arch.lanemask_lt()))
        if lane == 0:
            sWarp[warp] = cutlass.Int32(cute.arch.popc(cutlass.Uint32(ballot)))
        nvvm.barrier_cta_sync()
        offset = base
        total = cutlass.Int32(0)
        for w in cutlass.range_constexpr(n_threads // 32):
            count = sWarp[w]
            total = total + count
            if cutlass.Int32(w) < warp:
                offset = offset + count
        if keep:
            sIdx[offset + rank] = b
        base = base + total
        nvvm.barrier_cta_sync()
        start = start + cutlass.Int32(n_threads)
    return base


@cute.jit
def order_body(
    gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    n_threads: cutlass.Constexpr[int],
    order_elements: cutlass.Constexpr[int],
    tidx,
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    mCuSeqlens: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor | None,
    sKey,
    sIdx,
    sSpread,
    *,
    num_ctas: cutlass.Constexpr[int] = 0,
    expand_num: cutlass.Constexpr[int] = 1,
    chain: cutlass.Constexpr[bool] = False,
    mRowBase: cute.Tensor | None = None,
    mSlotRows: cute.Tensor | None = None,
):
    """Bitonic-sort the staged items by ``compute_end - compute_start``,
    longest first, into ``work_items``; with ``gen`` synthesize the uncut item
    per (batch, head) from ``cu_seqlens`` instead, with ``chain`` the piece
    items from the piece-wise ``cu_seqlens``, the row bases ``mRowBase`` of this
    table and the main table's row bases ``mSlotRows`` that number the flat slots
    (``mCount`` holds their number).  Thread 0 also zeroes the
    scheduler ticket ring when one is passed.  Caller owns ``sKey``/``sIdx`` (``n_threads *
    order_elements`` Int32 each) and a 2-cell ``sSpread``.  CTA-wide barriers
    inside: every thread of the calling CTA must reach it."""
    capacity = cutlass.const_expr(n_threads * order_elements)

    # ---- zero the scheduler ticket rings ---------------------------------------------
    if cutlass.const_expr(mScheduler is not None):
        if tidx == 0:
            si = cutlass.Int32(0)
            while si < mScheduler.shape[0]:
                mScheduler[si] = cutlass.Int32(0)
                si = si + cutlass.Int32(1)

    # Attention Gym modification: the uncut ``gen`` table emits no item for sequences without
    # tokens, so padded ``cu_seqlens`` cost neither sort slots nor main-kernel tiles. Compacted item
    # ``i`` maps back to the original ``sequence * n_heads_out + head`` index. Callers seed the final
    # state and the state cotangent of empty sequences. Batches of more than ``capacity``
    # sequences keep the uncompacted table.
    compact = False
    if cutlass.const_expr(gen and not chain and expand_num == 1):
        n_batch = n_tiles // n_heads_out
        compact = n_batch <= cutlass.Int32(capacity)
        n = n_tiles
        if compact:
            n = compact_sequences(n_threads, tidx, n_batch, mCuSeqlens, sIdx, sKey) * n_heads_out
        if tidx == 0:
            mCount[0] = n
    elif cutlass.const_expr(gen):
        n = n_tiles
        if tidx == 0:
            mCount[0] = n_tiles
    else:
        n = mCount[0]

    # ---- copy through unsorted: past sort capacity, or one item per CTA --------------
    if (n > cutlass.Int32(capacity)) or (n <= cutlass.Int32(num_ctas)):
        i = tidx
        while i < n:
            src = i
            if compact:
                src = sIdx[i // n_heads_out] * n_heads_out + i % n_heads_out
            write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i, src, expand_num, chain, mRowBase, mSlotRows)
            i = i + cutlass.Int32(n_threads)
    else:
        # Resolve compacted sources before sIdx is reused as the sort permutation.
        sources = []
        for e in cutlass.range_constexpr(order_elements):
            i = tidx + cutlass.Int32(e * n_threads)
            src = i
            if compact:
                if i < n:
                    src = sIdx[i // n_heads_out] * n_heads_out + i % n_heads_out
            sources.append(src)
        if tidx == 0:
            sSpread[0] = cutlass.Int32(2147483647)
            sSpread[1] = cutlass.Int32(-2147483648)

        b_pad = cutlass.Int32(1)
        while b_pad < n:
            b_pad = b_pad * cutlass.Int32(2)
        nvvm.barrier_cta_sync()
        kmin = cutlass.Int32(2147483647)
        kmax = cutlass.Int32(-2147483648)
        for e in cutlass.range_constexpr(order_elements):
            i = tidx + cutlass.Int32(e * n_threads)
            if i < n:
                if cutlass.const_expr(chain):
                    slot, head_idx, row_start, row_end, key, final_dst, dstate_dst = piece_item_bounds(
                        b_t, n_heads_out, mCuSeqlens, mRowBase, mSlotRows, i, expand_num
                    )
                elif cutlass.const_expr(gen):
                    batch_idx, head_idx, batch_start, batch_end, batch_num_chunks = gen_item_bounds(b_t, n_heads_out, mCuSeqlens, sources[e], expand_num)
                    key = batch_num_chunks
                else:
                    key = mStaging[i, 5] - mStaging[i, 4]
                sKey[i] = key
                sIdx[i] = sources[e]
                kmin = kmin if kmin < key else key
                kmax = kmax if kmax > key else key
            elif i < b_pad:
                sKey[i] = cutlass.Int32(-2147483648)
                sIdx[i] = i

        nvvm.atomicrmw("min", sSpread.data_ptr(0), kmin, space=nvvm.SharedSpace.shared_cta)
        nvvm.atomicrmw("max", sSpread.data_ptr(1), kmax, space=nvvm.SharedSpace.shared_cta)
        nvvm.barrier_cta_sync()
        if sSpread[0] == sSpread[1]:
            # ---- every key equal and nothing to sort ---------------------------------
            i2 = tidx
            while i2 < n:
                write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i2, sIdx[i2], expand_num, chain, mRowBase, mSlotRows)
                i2 = i2 + cutlass.Int32(n_threads)
        else:
            # ---- bitonic sort, descending: k = subsequence width, j = distance -------
            k = cutlass.Int32(2)
            while k <= b_pad:
                j = k // cutlass.Int32(2)
                while j > 0:
                    for e in cutlass.range_constexpr(order_elements):
                        i = tidx + cutlass.Int32(e * n_threads)
                        if i < b_pad:
                            l = i ^ j
                            if l > i:
                                ki = sKey[i]
                                kl = sKey[l]
                                up = (i // k) % cutlass.Int32(2) == 0
                                dn = (i // k) % cutlass.Int32(2) == 1
                                swap = (up and ki < kl) or (dn and ki > kl)
                                if swap:
                                    sKey[i] = kl
                                    sKey[l] = ki
                                    si = sIdx[i]
                                    sIdx[i] = sIdx[l]
                                    sIdx[l] = si
                    nvvm.barrier_cta_sync()
                    j = j // cutlass.Int32(2)
                k = k * cutlass.Int32(2)

            # ---- write the table in sorted order -------------------------------------
            for e in cutlass.range_constexpr(order_elements):
                i = tidx + cutlass.Int32(e * n_threads)
                if i < n:
                    src = sIdx[i]
                    write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i, src, expand_num, chain, mRowBase, mSlotRows)


@cute.jit
def launch(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    full_scan: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    mGate: cute.Tensor | None,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor | None,
    mStaging: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor | None,
    n_scan_ctas: cutlass.Int32,
    n_scan_blocks: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    if cutlass.const_expr(split):
        frost_split_k_plan(
            b_t,
            overhead_chunks,
            expand_num,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            mCuSeqlens,
            mChunkVals,
            mCount,
        ).launch(grid=(1, 1, 1), block=(PLAN_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)
        if cutlass.const_expr(gate_channels > 0):
            frost_split_k_scan_channel(
                b_t,
                scan_rows,
                warmup_cap,
                full_scan,
                log_gate,
                safe_gate,
                gate_channels,
                overhead_chunks,
                expand_num,
                n_heads_out,
                num_sms,
                n_tiles,
                ideal_chunks,
                batch_size,
                n_scan_blocks,
                gate_scale_log2,
                mGate,
                mALog,
                mDtBias,
                mCuSeqlens,
                mChunkVals,
                mCount,
            ).launch(
                grid=(n_scan_ctas, n_heads_out, 1),
                block=(SCAN_THREADS, 1, 1),
                stream=stream,
                use_pdl=USE_PDL,
            )
        else:
            frost_split_k_scan_scalar(
                b_t,
                scan_rows,
                warmup_cap,
                full_scan,
                log_gate,
                safe_gate,
                overhead_chunks,
                expand_num,
                n_heads_out,
                num_sms,
                n_tiles,
                ideal_chunks,
                batch_size,
                n_scan_blocks,
                mGate,
                mALog,
                mDtBias,
                mCuSeqlens,
                mChunkVals,
                mCount,
            ).launch(
                grid=(n_scan_ctas, -(-n_heads_out // WARP_SIZE), 1),
                block=(SCAN_THREADS, 1, 1),
                stream=stream,
                use_pdl=USE_PDL,
            )
        frost_split_k_walk(
            b_t,
            overhead_chunks,
            expand_num,
            warmup_cap,
            full_scan,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            log2_thresh,
            mCuSeqlens,
            mChunkVals,
            mStaging,
            mCount,
        ).launch(
            grid=(n_walk_ctas, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            stream=stream,
            use_pdl=USE_PDL,
        )


compiled_cache = {}


class SplitTableFacts(NamedTuple):
    """What :func:`launch` is compiled with (the constexprs) and fed per call (the scalars), derived once from the gate,
    cu_seqlens and the machine; :func:`build_split_table` and the unified warmup hosts share the derivation."""

    split: bool
    b_t: int
    scan_rows: int
    log_gate: bool
    safe_gate: bool
    gate_channels: int
    overhead_chunks: int
    expand_num: int
    warmup_cap: int
    full_scan: bool
    n_heads_out: int
    num_sms: int
    n_tiles: int
    ideal_chunks: int
    batch_size: int
    log2_threshold: float
    gate_scale_log2: float
    n_scan_ctas: int
    n_scan_blocks: int
    n_walk_ctas: int
    gate_elem_bytes: int


def split_table_facts(
    gate, cu_seqlens, *, split, n_tiles, ideal_chunks, num_sms, b_t, log2_threshold, log_gate, safe_gate, gate_lower_bound, expand_num
) -> SplitTableFacts:
    if log2_threshold is None:
        log2_threshold = DEFAULT_LOG2_THRESHOLD
    gate_channels = gate.shape[2] if len(gate.shape) == 3 else 0
    gate_scale_log2 = float(gate_lower_bound) * RCP_LN2 if safe_gate and gate_channels > 0 else 0.0
    n_heads_out = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    need_rows = chunk_scratch_rows(gate.shape[0] * int(expand_num), batch_size, b_t)
    grid_y = n_heads_out if gate_channels > 0 else -(-n_heads_out // WARP_SIZE)
    scan_rows = scan_rows_per_warp(need_rows, grid_y, num_sms)
    n_scan_blocks = -(-need_rows // (SCAN_WARPS * scan_rows))
    n_scan_ctas = min(
        n_scan_blocks,
        max(max(1, SCAN_CTA_CAP * num_sms // grid_y), -(-n_scan_blocks // SCAN_BLOCK_LOOP_MAX)),
    )
    return SplitTableFacts(
        bool(split),
        int(b_t),
        int(scan_rows),
        bool(log_gate),
        bool(safe_gate),
        int(gate_channels),
        max(1, OVERHEAD_TOKENS // b_t),
        int(expand_num),
        warmup_cap_chunks(gate_channels, int(expand_num)),
        gate_channels == 0,
        int(n_heads_out),
        int(num_sms),
        int(n_tiles),
        int(ideal_chunks or 0),
        int(batch_size),
        float(log2_threshold),
        float(gate_scale_log2),
        int(n_scan_ctas),
        int(n_scan_blocks),
        int(batch_size * n_heads_out),
        get_dtype(gate.dtype).width // 8,
    )


class TableRecipe(NamedTuple):
    """Build-time facts of one split-table launch: everything static is
    settled once so :func:`run_table` can replay the call as a straight
    line.  Produced by :func:`build_split_table`."""

    compiled: object
    split: bool
    safe_gate: bool
    n_heads_out: int
    n_tiles: int
    num_sms: int
    ideal_chunks: int
    batch_size: int
    log2_threshold: float
    gate_scale_log2: float
    n_scan_ctas: int
    n_scan_blocks: int
    n_walk_ctas: int
    expand_num: int


def run_table(r, gate, a_log, dt_bias, cu_seqlens, chunk_scratch, item_scratch, work_items, work_count, scheduler_counter, stream, *, expand_num=1) -> None:
    """The lowered split-table launch: no validation, no key build.  Only
    buffers move between calls; every scalar comes from the recipe
    (``expand_num`` is compiled in)."""
    r.compiled(
        r.n_heads_out,
        r.n_tiles,
        r.ideal_chunks,
        r.batch_size,
        r.log2_threshold,
        r.gate_scale_log2,
        gate if r.split else None,
        a_log if r.safe_gate else None,
        dt_bias if r.safe_gate else None,
        cu_seqlens,
        chunk_scratch if r.split else None,
        item_scratch if r.split else None,
        work_items,
        work_count,
        scheduler_counter,
        r.n_scan_ctas,
        r.n_scan_blocks,
        r.n_walk_ctas,
        cuda.CUstream(int(stream)),
    )


def build_split_table(
    gate,
    cu_seqlens,
    work_items,
    work_count,
    *,
    ideal_chunks=None,
    n_tiles,
    num_sms,
    b_t,
    chunk_scratch=None,
    item_scratch=None,
    log2_threshold=None,
    log_gate=False,
    safe_gate=False,
    a_log=None,
    dt_bias=None,
    gate_lower_bound=None,
    scheduler_counter,
    split=True,
    expand_num=1,
    opt_level,
    stream,
) -> "TableRecipe":
    """Fill ``work_items``/``work_count`` with the split-K partition of
    ``gate`` / ``cu_seqlens (B+1,) int32``.  A 2-D ``(total_tokens, HO)`` gate
    is the scalar GDN kind, 3-D ``(total_tokens, HO, DK)`` the per-key-channel
    KDA / GDN-2 kind; ``log_gate`` means natural-log decay rather than linear
    alpha.  With ``safe_gate`` the gate holds RAW logits and the scan applies
    the matching transform so cuts land on true decay: per-channel
    ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))``
    (``gate_lower_bound`` required), scalar ``-exp(a_log[h]) * softplus(g +
    dt_bias[h])``.  ``a_log`` / ``dt_bias`` may each be None under
    ``safe_gate`` (unit amplitude / zero bias).

    With ``split=False`` nothing launches; each main kernel's prologue
    synthesizes the no-cuts table itself (``order_gen``).

    ``work_items`` and ``item_scratch`` are ``(max_items, WORK_ITEM_FIELDS)``
    int32 with ``max_items >= max_work_items(...)``; ``work_count`` is
    ``(1,)`` int32; ``chunk_scratch`` is ``(>= chunk_scratch_rows(...), HO)``
    fp32, whose LAST row carries the plan rather than a chunk value.
    ``scheduler_counter`` cells are zeroed by the order phase, the count by the
    plan.  ``expand_num`` scales every device-side ``cu_seqlens`` read onto
    GDP's expanded sub-token timeline, so the gate and all chunk geometry are
    in ``cu * expand_num`` units without materializing that array (1 = off).
    Runs entirely on device, no host synchronization.  ``opt_level`` is the family's main-kernel level, so the
    standalone table is the one the family's warmup hosts nest."""
    if not safe_gate:
        a_log = None
        dt_bias = None
    f = split_table_facts(
        gate,
        cu_seqlens,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sms=num_sms,
        b_t=b_t,
        log2_threshold=log2_threshold,
        log_gate=log_gate,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound,
        expand_num=expand_num,
    )
    scan_rows, gate_channels, n_heads_out, batch_size, gate_elem_bytes = f.scan_rows, f.gate_channels, f.n_heads_out, f.batch_size, f.gate_elem_bytes
    log2_threshold, gate_scale_log2 = f.log2_threshold, f.gate_scale_log2
    n_scan_ctas, n_scan_blocks, n_walk_ctas = f.n_scan_ctas, f.n_scan_blocks, f.n_walk_ctas
    overhead_chunks, warmup_cap, full_scan = f.overhead_chunks, f.warmup_cap, f.full_scan
    cu_stream = cuda.CUstream(int(stream))

    key = (
        bool(split),
        b_t,
        scan_rows,
        num_sms,
        bool(log_gate),
        bool(safe_gate),
        gate_channels,
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        int(expand_num),
        current_device(),
        int(opt_level),
    )
    if key not in compiled_cache:

        dt_bias_c = None
        if dt_bias is not None:
            dt_bias_c = from_dlpack(dt_bias, assumed_align=4)
            dt_bias_c.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(len(dt_bias.shape))), divisibility=1)
        chunk_scratch_c = None
        item_scratch_c = None
        if split:
            chunk_scratch_c = from_dlpack(chunk_scratch, assumed_align=4)
            chunk_scratch_c.mark_layout_dynamic(leading_dim=1)
            item_scratch_c = from_dlpack(item_scratch, assumed_align=4)
            item_scratch_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_c = from_dlpack(work_items, assumed_align=4)
        work_items_c.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_c = from_dlpack(work_count, assumed_align=4)
        work_count_c.mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=1)
        compiled_cache[key] = cute.compile(
            launch,
            bool(split),
            b_t,
            int(scan_rows),
            bool(log_gate),
            bool(safe_gate),
            gate_channels,
            overhead_chunks,
            int(expand_num),
            warmup_cap,
            full_scan,
            cutlass.Int32(n_heads_out),
            int(num_sms),
            cutlass.Int32(n_tiles),
            cutlass.Int32(ideal_chunks),
            cutlass.Int32(batch_size),
            cutlass.Float32(log2_threshold),
            cutlass.Float32(gate_scale_log2),
            from_dlpack(gate, assumed_align=(8 if gate_elem_bytes == 2 else 4)).mark_layout_dynamic(leading_dim=len(gate.shape) - 1) if split else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            dt_bias_c,
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            chunk_scratch_c,
            item_scratch_c,
            work_items_c,
            work_count_c,
            from_dlpack(scheduler_counter, assumed_align=4).mark_layout_dynamic(),
            cutlass.Int32(n_scan_ctas),
            cutlass.Int32(n_scan_blocks),
            cutlass.Int32(n_walk_ctas),
            cu_stream,
            options=f"--enable-tvm-ffi --opt-level {int(opt_level)}",
        )
    compiled_cache[key](
        n_heads_out,
        n_tiles,
        ideal_chunks,
        batch_size,
        float(log2_threshold),
        float(gate_scale_log2),
        gate,
        a_log,
        dt_bias,
        cu_seqlens,
        chunk_scratch,
        item_scratch,
        work_items,
        work_count,
        scheduler_counter,
        n_scan_ctas,
        n_scan_blocks,
        n_walk_ctas,
        cu_stream,
    )
    return TableRecipe(
        compiled_cache[key],
        bool(split),
        bool(safe_gate),
        n_heads_out,
        n_tiles,
        num_sms,
        int(ideal_chunks),
        batch_size,
        float(log2_threshold),
        float(gate_scale_log2),
        n_scan_ctas,
        n_scan_blocks,
        n_walk_ctas,
        int(expand_num),
    )


frost_split_k_plan.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_split_k_scan_channel.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_split_k_scan_scalar.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_split_k_walk.set_name_prefix("cudnn", remove_cutlass_symbol=False)
