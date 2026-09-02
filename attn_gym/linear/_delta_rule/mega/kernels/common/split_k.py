# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: imports were relocated and empty sequences were changed to
# emit no work item, preventing invalid persistent-kernel TMEM lifecycle transitions.

"""Split-K sequence partitioning for ALL chunked linear-attention kernels:
GDN (scalar ``gate (T, HO)``, b_t=64) and KDA / GDN-2 (per-key-channel
``gate (T, HO, DK)``, b_t=16) share this one three-kernel pipeline.

The chunked kernels are persistent one-CTA-per-(batch, head) kernels; when
``B * HO`` does not fill the SMs, long sequences serialize on a few CTAs.
This module cuts each (batch, head) sequence into independent work items at
"forgetting horizon" boundaries: a cut is placed only where the running
log2 of the forget gates saturates below a threshold on both sides, so the
recurrent state (forward) and the state gradient (backward) entering an
item can be reconstructed from a short warmup window recomputed by the item
itself — no state is exchanged between items.

Work-item table row (``WORK_ITEM_FIELDS`` x int32, chunk units)::

    [batch_idx, head_idx, wstart, wend, cstart, cend, batch_start, batch_end]

``batch_start``/``batch_end`` are the token bounds ``cu_seqlens[b]`` /
``cu_seqlens[b+1]`` denormalized into the row: decode reads one 32-byte
vectorizable row instead of chasing a dependent ``cu_seqlens`` load pair.

The item OWNS (writes outputs for) chunks ``[wstart, wend)``.  The forward
kernel COMPUTES chunks ``[cstart, wend)`` — ``[cstart, wstart)`` is the
left warmup that rebuilds the incoming state from zero (accurate to
``2^log2_threshold`` because the gate decay over the window saturates).
The backward kernel computes ``[wstart, cend)`` — ``[wend, cend)`` is the
right warmup for the reverse dstate recurrence (the forward states come exactly
from the per-chunk state checkpoints).  ``cstart == 0`` items seed the true
initial state; ``cend == num_chunks`` items seed the true ``d_final_state``
— so the un-cut degenerate item ``(0, nc, 0, nc)`` reproduces the serial
kernel exactly.

Piece choice per (batch, head): spans never exceed ``ideal_chunks`` (total
work / SM count), so outlier-long sequences are always cut down to the
batch-wide grain.  When the grid is small (``n_tiles < 2 * num_sms``) a
wave-quantized search widens the cut further to fill the machine; in the
many-tile regime cutting only adds per-item overhead, so the cap alone
decides.

The pipeline (one fixed shape regardless of gate kind):

1. scan: a flat data-parallel grid — CTA ``(x, h)`` covers chunk-scratch
   rows ``[x * WARPS, (x + 1) * WARPS)`` of head ``h``, one warp per chunk
   — reduces the gate to per-chunk values in the caller's GMEM scratch
   (channel gates: max over the per-channel clamped-log2 sums, a cut is
   valid only when EVERY channel saturated; scalar gates: the plain sum).
   Only chunks within ``WARMUP_CAP_CHUNKS`` of a candidate boundary are
   read: skipping chunks can only RAISE the (negative) horizon sums, so the
   windowed scan never accepts an unsaturated cut, it can only skip one.
   Uncut tiles never touch the gate.
2. walk: one CTA per (batch, head) probes every candidate boundary in
   parallel (one warp per boundary, one lane per window chunk), then
   thread 0 walks the probe results and emits work items into the caller's
   ``item_scratch``.
3. order (:func:`order_body`, hosted by each kernel module's
   prologue kernel alongside its TMA-descriptor build — one launch for
   both): bitonic-sort the items into ``work_items``, longest ``[cstart,
   cend)`` first, so the ticket scheduler consumes them in LPT order —
   the makespan tail is set by whatever starts last, so the big items
   must go first.  This is what keeps ragged varlen batches balanced
   without cutting them.

The order body also zeroes the main kernels' scheduler ticket rings
(dirty on exit), and with ``split=False`` it replaces the whole pipeline:
scan and walk never launch, and the prologue kernel synthesizes the uncut
whole-sequence item per (batch, head) from ``cu_seqlens`` alone, then
LPT-sorts those.  That no-cuts table serves batch-invariant mode and
coarse checkpoint cadences (cuts may not cross a checkpoint period).
"""

import math
from typing import NamedTuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute import make_fake_strided_tensor

from ..compat import current_device, data_ptr, get_device_properties, tensor_device_index

from .elementwise import softplus
from .host import get_dtype
from .tvm_ffi import (
    WORK_ITEM_FIELDS,
    make_counter_signature,
    make_cu_seqlens_signature,
    make_work_items_signature,
    validate_cu_seqlens,
)
WARMUP_CAP_CHUNKS = 32  # hard warmup cap: a cut must saturate within one warp of chunks per side
MAX_BLOCKS = 2048  # piece-count ceiling; host clamps ideal_chunks so the per-tile block count fits
WARP_SIZE = 32
WARPS = 8
THREADS_PER_BLOCK = WARPS * WARP_SIZE
SCAN_WARPS = 4
SCAN_THREADS = SCAN_WARPS * WARP_SIZE
SCAN_ROWS_PER_WARP = 4  # consecutive chunk rows per scan warp
SCAN_TOKEN_STRIDE = 4  # sample every Nth token of a chunk: skipped tokens only RAISE the negative horizon sums
OVERHEAD_TOKENS = 256  # per-item fixed cost for the piece model: state reseed + pipeline refill + typical warmup
P_WINDOW = 16  # fill-regime piece-count search width
P_BELOW = 8  # how far below the ideal-cap floor the fill-regime search may go

ORDER_THREADS = 1024
ORDER_ELEMS = 4
ORDER_CAPACITY = ORDER_THREADS * ORDER_ELEMS  # sort capacity (32 KB SMEM); past this the device-side branch copies through unsorted

# NOTE [Forgetting Horizon]
# The state at any token is a decayed sum of everything before it, so tokens far enough back no
# longer matter. The scan turns each chunk into one number: its log2 decay, taken from the slowest
# channel. At a candidate boundary we walk that map backwards summing decays; once the sum from
# chunk A to the boundary drops below the threshold (e^-10, in log2 units), everything before A has
# been scaled by less than 2^threshold and is dropped. Candidates come from load balance (multiples of
# the ideal span); the gates only accept or reject them, and both sides must saturate within
# WARMUP_CAP_CHUNKS so the same table serves the reverse (gradient) recurrence.
#
# Example: 10 chunks, candidate boundary at 5. Walk back accumulating decay: {4} is not below the
# threshold, {4,3} is not, {4,3,2} is. So chunks 2..4 are the warmup and the state entering chunk 2
# (everything from chunks 0..1) is dropped:
#     item 0: computes [0, 5)   writes [0, 5)    seeds the real initial state
#     item 1: computes [2, 10)  writes [5, 10)   chunks 2..4 start from zero, outputs discarded
#   The state item 1 reaches at chunk 5 is not the true one, but its error is at most
#   2^threshold * |H_before|; from there the arithmetic is the serial kernel's. Items never exchange
#   state; a rejected boundary just extends the open item, so a gate that never forgets yields the
#   single uncut item, i.e. the serial kernel.
#
# This is a margin of bits past the output dtype's half-ulp. A rounding can flip only where
# |H_window| < 2^-margin * |H_before| elementwise:
#     e^-10 = 2^-14.4
#     bf16 half-ulp 2^-9  -> 5.4-bit margin
#     fp16 half-ulp 2^-12 -> 2.4-bit margin
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


def chunk_scratch_rows(total_tokens: int, batch_size: int, b_t: int) -> int:
    """Rows of the ``(rows, HO)`` fp32 chunk-value scratch: per-batch chunk
    ranges are based at ``cu[b] // b_t + b``, so one extra row per sequence
    covers the ceil rounding."""
    return total_tokens // b_t + batch_size


def max_work_items(total_tokens: int, batch_size: int, n_heads_out: int, ideal_chunks: int, b_t: int, num_sms: int) -> int:
    """Provable upper bound on the emitted item count: spans are capped at
    ``ideal_chunks`` (at most ``ceil(nc / ideal) + 1`` pieces per (b, h)),
    zero-length sequences emit no work item, and the fill-regime search
    can widen any tile by up to ``P_WINDOW - 1`` extra pieces."""
    total_chunks = -(-total_tokens // b_t)
    window = P_WINDOW - 1 if batch_size * n_heads_out < 2 * num_sms else 0
    return n_heads_out * (-(-total_chunks // ideal_chunks) + (2 + window) * batch_size)


@cute.jit
def decode_work_item(cfg, tile_idx, mWorkItems):
    """Tile decode shared by every warp body of the main kernels: read the
    work-item row (an uncut table row IS the whole sequence).  Returns
    ``(batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b,
    wstart, wend, cstart, cend)`` with chunk-unit bounds."""
    batch_idx = mWorkItems[tile_idx, 0]
    head_idx = mWorkItems[tile_idx, 1]
    wstart = mWorkItems[tile_idx, 2]
    wend = mWorkItems[tile_idx, 3]
    cstart = mWorkItems[tile_idx, 4]
    cend = mWorkItems[tile_idx, 5]
    batch_start = mWorkItems[tile_idx, 6]
    batch_end = mWorkItems[tile_idx, 7]
    seqlen_b = batch_end - batch_start
    num_chunks_b = cute.ceil_div(seqlen_b, cfg.b_t)
    return batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend


@cute.jit
def emit_item(mWorkItems, mCount, batch_idx, head_idx, wstart, wend, cstart, cend, batch_start, batch_end):
    count_addr = mCount.iterator.toint()
    wi = inline_ptx(
        "atom.global.add.s32 {$w0}, [{$r0}], 1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[count_addr],
    )
    mWorkItems[wi, 0] = batch_idx
    mWorkItems[wi, 1] = head_idx
    mWorkItems[wi, 2] = wstart
    mWorkItems[wi, 3] = wend
    mWorkItems[wi, 4] = cstart
    mWorkItems[wi, 5] = cend
    mWorkItems[wi, 6] = batch_start
    mWorkItems[wi, 7] = batch_end


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
def piece_choice(overhead_chunks: cutlass.Constexpr[int], num_chunks_b, n_tiles, num_sms, ideal_chunks):
    """Per-tile piece choice.  Returns ``(span, num_blocks)``."""
    # even spread: spans never exceed ideal_chunks (total work / SM count)
    p_hi = num_chunks_b if num_chunks_b < cutlass.Int32(MAX_BLOCKS) else cutlass.Int32(MAX_BLOCKS)
    p_hi = p_hi if p_hi > 0 else cutlass.Int32(1)
    p = (num_chunks_b + ideal_chunks - cutlass.Int32(1)) // ideal_chunks
    p = p if p > 0 else cutlass.Int32(1)
    p = p if p < p_hi else p_hi
    if n_tiles < cutlass.Int32(2) * num_sms:
        # fill regime: SMs to spare — search piece counts around the cap on
        # the wave-quantized makespan estimate
        p_start = p - cutlass.Int32(P_BELOW)
        p_start = p_start if p_start > 0 else cutlass.Int32(1)
        best = cutlass.Int32(2147483647)
        for dp in cutlass.range_constexpr(P_WINDOW):
            cand = p_start + cutlass.Int32(dp)
            cand = cand if cand < p_hi else p_hi
            span_c = (num_chunks_b + cand - cutlass.Int32(1)) // cand
            waves = (n_tiles * cand + num_sms - cutlass.Int32(1)) // num_sms
            est = waves * (span_c + cutlass.Int32(overhead_chunks))
            hit = est < best
            best = est if hit else best
            p = cand if hit else p
        # the estimate flatters marginal cuts; cut only on a clear margin over uncut
        est1 = ((n_tiles + num_sms - cutlass.Int32(1)) // num_sms) * (num_chunks_b + cutlass.Int32(overhead_chunks))
        if cutlass.Int32(4) * best > cutlass.Int32(3) * est1:
            p = cutlass.Int32(1)
    span = cutlass.Int32(0)
    num_blocks = cutlass.Int32(0)
    if num_chunks_b > 0:
        span = (num_chunks_b + p - cutlass.Int32(1)) // p
        num_blocks = (num_chunks_b + span - cutlass.Int32(1)) // span
    return span, num_blocks


@cute.jit
def tile_spans(b_t: cutlass.Constexpr[int], overhead_chunks: cutlass.Constexpr[int], n_heads_out, n_tiles, num_sms, ideal_chunks, mCuSeqlens, tile):
    """Per-tile decode + piece choice.  Returns ``(batch_idx, head_idx,
    batch_start, batch_end, num_chunks_b, cv_base, span, num_blocks)``;
    ``cv_base`` is the tile's row base in the GMEM chunk scratch."""
    batch_idx = tile // n_heads_out
    head_idx = tile % n_heads_out
    batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
    batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
    seqlen_b = batch_end - batch_start
    num_chunks_b = cute.ceil_div(seqlen_b, b_t)
    cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
    span, num_blocks = piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
    return batch_idx, head_idx, batch_start, batch_end, num_chunks_b, cv_base, span, num_blocks


@cute.jit
def near_boundary(c, span, num_blocks):
    """True iff chunk ``c`` lies in the walk's read window of a candidate
    boundary: suffix ``[j*span - W, j*span)`` or prefix ``[j*span, j*span
    + W)`` for some ``j`` in ``[1, num_blocks)``."""
    j0 = c // span
    cm = c - j0 * span
    w = cutlass.Int32(WARMUP_CAP_CHUNKS)
    pre = (cm < w) and (j0 >= cutlass.Int32(1))
    suf = (span - cm <= w) and (j0 < num_blocks - cutlass.Int32(1))
    return pre or suf


@cute.kernel
def scan_kernel(
    b_t: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    num_sms: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    gate_scale_log2: cutlass.Float32,
    mGate: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """Flat windowed chunk scan, per-channel gate (KDA / GDN-2): CTA
    ``(x, h)`` covers 16 chunk-scratch rows of head ``h``, one warp per
    chunk, lane ``l`` owning channels ``[l*cpl, (l+1)*cpl)``.  Chunks
    outside every cut window — and whole tiles the piece choice leaves
    uncut — never touch the gate.  CTA (0, 0) also zeroes the item count
    for the walk (the scheduler rings are the order kernel's job)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()
    tidx = cutlass.Int32(tidx)
    head_idx = cutlass.Int32(bidx[1])
    if cutlass.Int32(bidx[0]) == 0 and head_idx == 0 and tidx == 0:
        mCount[0] = cutlass.Int32(0)
    lidx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    row0 = (cutlass.Int32(bidx[0]) * cutlass.Int32(SCAN_WARPS) + widx) * cutlass.Int32(SCAN_ROWS_PER_WARP)

    # batch of the warp's first row: largest b with cu[b] // b_t + b <= row0
    lo = cutlass.Int32(0)
    hi = batch_size - cutlass.Int32(1)
    while lo < hi:
        mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
        take = cutlass.Int32(mCuSeqlens[mid]) // cutlass.Int32(b_t) + mid <= row0
        lo = mid if take else lo
        hi = hi if take else mid - cutlass.Int32(1)
    batch_idx = lo
    batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
    batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
    num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
    cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
    # the piece choice is computed per batch, NOT per row
    span, num_blocks = piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
    for rr in cutlass.range_constexpr(SCAN_ROWS_PER_WARP):
        row = row0 + cutlass.Int32(rr)
        while (batch_idx + cutlass.Int32(1) < batch_size) and (
            cutlass.Int32(mCuSeqlens[batch_idx + 1]) // cutlass.Int32(b_t) + batch_idx + cutlass.Int32(1) <= row
        ):
            batch_idx = batch_idx + cutlass.Int32(1)
            batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
            batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
            num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
            cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
            span, num_blocks = piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
        c = row - cv_base
        if (c >= 0) and (c < num_chunks_b) and (num_blocks > 1):
            if near_boundary(c, span if span > 0 else cutlass.Int32(1), num_blocks):
                # chunk value = max over channels of the per-channel
                # clamped-log2 sums (each lane owns a contiguous channel run)
                cpl = gate_channels // WARP_SIZE
                row_elems = cutlass.Int32(mGate.stride[0])
                lane_base = cutlass.Int64(head_idx * cutlass.Int32(mGate.stride[1]) + lidx * cutlass.Int32(cpl))
                gate_addr = mGate.iterator.toint() + lane_base * cutlass.Int64(4)
                gate_ptr = mGate.iterator + lane_base
                a_exp = cutlass.Float32(1.0)
                dt_vals = cutlass.Array(cutlass.Float32, cpl)
                if cutlass.const_expr(safe_gate):
                    # per-head rate + per-lane channel biases, fixed for the whole chunk
                    a_exp = cute.math.exp2(mALog[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True)
                    for q in cutlass.range_constexpr(cpl):
                        dt_vals[q] = mDtBias[head_idx, lidx * cutlass.Int32(cpl) + cutlass.Int32(q)].to(cutlass.Float32)
                ch_acc = cutlass.Array(cutlass.Float32, cpl, alignment=16)
                for q in cutlass.range_constexpr(cpl):
                    ch_acc[q] = cutlass.Float32(0.0)
                oob = cutlass.Float32(0.0) if cutlass.const_expr(log_gate) else cutlass.Float32(1.0)
                for tt in cutlass.range(0, b_t, SCAN_TOKEN_STRIDE, unroll_full=True):
                    pos = batch_start + c * cutlass.Int32(b_t) + cutlass.Int32(tt)
                    inb = pos < batch_end
                    pos_r = pos if inb else batch_start
                    grow = cutlass.Int64(pos_r) * cutlass.Int64(row_elems)
                    if cutlass.const_expr(cpl % 4 == 0):
                        for q4 in cutlass.range_constexpr(cpl // 4):
                            addr = gate_addr + (grow + cutlass.Int64(4 * q4)) * cutlass.Int64(4)
                            g0, g1, g2, g3 = inline_ptx(
                                "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
                                write_only_types=[cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Float32],
                                read_only_args=[addr],
                            )
                            for qq, gvq in enumerate((g0, g1, g2, g3)):
                                q = 4 * q4 + qq
                                if cutlass.const_expr(safe_gate):
                                    # the main kernel's transform, in log2 domain: scale * sigmoid(exp(a_log) * (g + dt_bias))
                                    half = cutlass.Float32(0.5)
                                    sig = cute.math.tanh(a_exp * (gvq + dt_vals[q]) * half, approx=True) * half + half
                                    contrib = gate_scale_log2 * sig
                                    contrib = contrib if inb else cutlass.Float32(0.0)
                                else:
                                    gvq = gvq if inb else oob
                                    contrib = clamped_log2(log_gate, gvq)
                                ch_acc[q] = ch_acc[q] + contrib
                    else:
                        for q in cutlass.range_constexpr(cpl):
                            gv = (gate_ptr + grow + cutlass.Int32(q)).load()
                            if cutlass.const_expr(safe_gate):
                                half = cutlass.Float32(0.5)
                                sig = cute.math.tanh(a_exp * (gv + dt_vals[q]) * half, approx=True) * half + half
                                contrib = gate_scale_log2 * sig
                                contrib = contrib if inb else cutlass.Float32(0.0)
                            else:
                                gv = gv if inb else oob
                                contrib = clamped_log2(log_gate, gv)
                            ch_acc[q] = ch_acc[q] + contrib
                m = ch_acc[0]
                for q in cutlass.range_constexpr(1, cpl):
                    m = m if m > ch_acc[q] else ch_acc[q]
                for off in [1, 2, 4, 8, 16]:
                    other = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, m, off, 31, kind=nvvm.Shfl.BFLY))
                    m = m if m > other else other
                if lidx == 0:
                    mChunkVals[cv_base + c, head_idx] = m


@cute.kernel
def scan_scalar_kernel(
    b_t: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    num_sms: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    mGate: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """Scalar-gate scan (GDN): CTA ``(x, hg)`` covers 16 chunk-scratch rows
    for heads ``[hg*32, (hg+1)*32)``; lane ``l`` owns head ``hg*32 + l``, so
    gate reads and chunk-value writes are coalesced across lanes and every
    lane accumulates its own head — no reduction.  With ``safe_gate`` the
    gate holds raw logits and each token contributes the GDN transform in
    log2 domain: ``-exp(A_log[h]) * softplus(g + dt_bias[h]) * RCP_LN2``.
    CTA (0, 0) zeroes the item count (the scheduler rings are the order
    kernel's job)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()
    tidx = cutlass.Int32(tidx)
    if cutlass.Int32(bidx[0]) == 0 and cutlass.Int32(bidx[1]) == 0 and tidx == 0:
        mCount[0] = cutlass.Int32(0)
    lidx = tidx % cutlass.Int32(WARP_SIZE)
    widx = tidx // cutlass.Int32(WARP_SIZE)
    h = cutlass.Int32(bidx[1]) * cutlass.Int32(WARP_SIZE) + lidx
    h_ok = h < n_heads_out
    h_r = h if h_ok else n_heads_out - cutlass.Int32(1)
    a_l2 = cutlass.Float32(0.0)
    bias = cutlass.Float32(0.0)
    if cutlass.const_expr(safe_gate):
        # per-head transform constants, fixed for the lane's whole sweep
        a_l2 = -cute.math.exp2(mALog[h_r].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True) * cutlass.Float32(RCP_LN2)
        bias = mDtBias[h_r].to(cutlass.Float32)
    row0 = (cutlass.Int32(bidx[0]) * cutlass.Int32(SCAN_WARPS) + widx) * cutlass.Int32(SCAN_ROWS_PER_WARP)

    # batch of the warp's first row: largest b with cu[b] // b_t + b <= row0
    lo = cutlass.Int32(0)
    hi = batch_size - cutlass.Int32(1)
    while lo < hi:
        mid = (lo + hi + cutlass.Int32(1)) // cutlass.Int32(2)
        take = cutlass.Int32(mCuSeqlens[mid]) // cutlass.Int32(b_t) + mid <= row0
        lo = mid if take else lo
        hi = hi if take else mid - cutlass.Int32(1)
    batch_idx = lo
    batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
    batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
    num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
    cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
    span, num_blocks = piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
    for rr in cutlass.range_constexpr(SCAN_ROWS_PER_WARP):
        row = row0 + cutlass.Int32(rr)
        while (batch_idx + cutlass.Int32(1) < batch_size) and (
            cutlass.Int32(mCuSeqlens[batch_idx + 1]) // cutlass.Int32(b_t) + batch_idx + cutlass.Int32(1) <= row
        ):
            batch_idx = batch_idx + cutlass.Int32(1)
            batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
            batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
            num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
            cv_base = batch_start // cutlass.Int32(b_t) + batch_idx
            span, num_blocks = piece_choice(overhead_chunks, num_chunks_b, n_tiles, num_sms, ideal_chunks)
        c = row - cv_base
        if (c >= 0) and (c < num_chunks_b) and (num_blocks > 1):
            if near_boundary(c, span if span > 0 else cutlass.Int32(1), num_blocks):
                oob = cutlass.Float32(0.0) if cutlass.const_expr(log_gate) else cutlass.Float32(1.0)
                acc = cutlass.Float32(0.0)
                for tt in cutlass.range(0, b_t, SCAN_TOKEN_STRIDE, unroll_full=True):
                    pos = batch_start + c * cutlass.Int32(b_t) + cutlass.Int32(tt)
                    inb = pos < batch_end
                    pos_r = pos if inb else batch_start
                    gate_offset = (
                        cutlass.Int64(pos_r) * cutlass.Int64(mGate.stride[0])
                        + cutlass.Int64(h_r) * cutlass.Int64(mGate.stride[1])
                    )
                    gv = (mGate.iterator + gate_offset).load()
                    if cutlass.const_expr(safe_gate):
                        contrib = a_l2 * softplus(gv + bias)
                        acc = acc + (contrib if inb else cutlass.Float32(0.0))
                    else:
                        gv = gv if inb else oob
                        acc = acc + clamped_log2(log_gate, gv)
                if h_ok:
                    mChunkVals[cv_base + c, h] = acc


@cute.kernel
def walk_kernel(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    num_sms: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mStaging: cute.Tensor,
    mCount: cute.Tensor,
):
    """One CTA per (batch, head): every candidate boundary is probed by its
    own warp (one lane per window chunk, warp-scan for the smallest
    saturating warmup), then thread 0 walks the probe results and emits the
    items."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    tidx = cutlass.Int32(tidx)

    batch_idx, head_idx, batch_start, batch_end, num_chunks_b, cv_base, span, num_blocks = tile_spans(
        b_t, overhead_chunks, n_heads_out, n_tiles, num_sms, ideal_chunks, mCuSeqlens, cutlass.Int32(bidx)
    )
    if num_blocks <= 1:
        # Single nonempty piece: no cuts, nothing scanned. Empty sequences
        # preserve state through preinitialized outputs and require no kernel work.
        if (tidx == 0) and (num_chunks_b > 0):
            emit_item(mStaging, mCount, batch_idx, head_idx, cutlass.Int32(0), num_chunks_b, cutlass.Int32(0), num_chunks_b, batch_start, batch_end)

    else:
        # packed per-boundary probe results: warm_b | warm_f << 8, 0 = no cut
        sWarm = cutlass.Array(cutlass.Int32, MAX_BLOCKS, space=cutlass.AddressSpace.smem, alignment=16)
        lidx = tidx % cutlass.Int32(WARP_SIZE)
        widx = tidx // cutlass.Int32(WARP_SIZE)
        big = cutlass.Int32(2 * WARMUP_CAP_CHUNKS)
        j = widx + cutlass.Int32(1)
        while j < num_blocks:
            wend = j * span
            # fwd warmup: smallest chunk suffix of [0, wend) that saturates
            idx = wend - cutlass.Int32(1) - lidx
            ok_b = idx >= 0
            v = mChunkVals[cv_base + (idx if ok_b else cutlass.Int32(0)), head_idx]
            acc = v if ok_b else cutlass.Float32(0.0)
            for off in [1, 2, 4, 8, 16]:
                o = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, acc, off, 0, kind=nvvm.Shfl.UP))
                acc = acc + (o if lidx >= cutlass.Int32(off) else cutlass.Float32(0.0))
            cand = lidx + cutlass.Int32(1) if (ok_b and acc <= log2_thresh) else big
            for off in [1, 2, 4, 8, 16]:
                other = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, cand, off, 31, kind=nvvm.Shfl.BFLY))
                cand = cand if cand < other else other
            warm_b = cand if cand < big else cutlass.Int32(0)
            # bwd warmup: smallest chunk prefix of [wend, nc) that saturates
            idx = wend + lidx
            ok_f = idx < num_chunks_b
            v = mChunkVals[cv_base + (idx if ok_f else cutlass.Int32(0)), head_idx]
            acc = v if ok_f else cutlass.Float32(0.0)
            for off in [1, 2, 4, 8, 16]:
                o = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, acc, off, 0, kind=nvvm.Shfl.UP))
                acc = acc + (o if lidx >= cutlass.Int32(off) else cutlass.Float32(0.0))
            cand = lidx + cutlass.Int32(1) if (ok_f and acc <= log2_thresh) else big
            for off in [1, 2, 4, 8, 16]:
                other = cutlass.Int32(nvvm.shfl_sync(0xFFFFFFFF, cand, off, 31, kind=nvvm.Shfl.BFLY))
                cand = cand if cand < other else other
            warm_f = cand if cand < big else cutlass.Int32(0)
            if lidx == 0:
                packed = warm_b + warm_f * cutlass.Int32(256)
                packed = packed if (warm_b > 0 and warm_f > 0) else cutlass.Int32(0)
                sWarm[j] = packed
            j = j + cutlass.Int32(WARPS)
        nvvm.barrier_cta_sync()

        if tidx == 0:
            prev_cut = cutlass.Int32(0)  # wstart of the open item, chunk units
            cur_cstart = cutlass.Int32(0)  # cstart of the open item, chunk units
            jj = cutlass.Int32(1)
            while jj < num_blocks:
                r = sWarm[jj]
                if r != 0:
                    wend = jj * span
                    warm_b = r % cutlass.Int32(256)
                    warm_f = r // cutlass.Int32(256)
                    cend = wend + warm_f
                    cend = cend if cend < num_chunks_b else num_chunks_b
                    emit_item(mStaging, mCount, batch_idx, head_idx, prev_cut, wend, cur_cstart, cend, batch_start, batch_end)
                    cur_cstart = wend - warm_b
                    prev_cut = wend
                jj = jj + cutlass.Int32(1)
            emit_item(mStaging, mCount, batch_idx, head_idx, prev_cut, num_chunks_b, cur_cstart, num_chunks_b, batch_start, batch_end)


@cute.jit
def gen_item_bounds(b_t: cutlass.Constexpr[int], n_heads_out, mCuSeqlens, item):
    """(batch, head, batch_start, batch_end, num_chunks) of the uncut
    whole-sequence item ``item`` — a no-cuts table row is pure geometry."""
    batch_idx = item // n_heads_out
    head_idx = item % n_heads_out
    batch_start = cutlass.Int32(mCuSeqlens[batch_idx])
    batch_end = cutlass.Int32(mCuSeqlens[batch_idx + 1])
    num_chunks_b = cute.ceil_div(batch_end - batch_start, b_t)
    return batch_idx, head_idx, batch_start, batch_end, num_chunks_b


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
):
    """Final-table row ``dst`` from source item ``src``: the walk's staged
    row, or (``gen``) the synthesized uncut item ``(0, nc, 0, nc)``."""
    if cutlass.const_expr(gen):
        batch_idx, head_idx, batch_start, batch_end, num_chunks_b = gen_item_bounds(b_t, n_heads_out, mCuSeqlens, src)
        mWorkItems[dst, 0] = batch_idx
        mWorkItems[dst, 1] = head_idx
        mWorkItems[dst, 2] = cutlass.Int32(0)
        mWorkItems[dst, 3] = num_chunks_b
        mWorkItems[dst, 4] = cutlass.Int32(0)
        mWorkItems[dst, 5] = num_chunks_b
        mWorkItems[dst, 6] = batch_start
        mWorkItems[dst, 7] = batch_end
    else:
        for f in cutlass.range_constexpr(WORK_ITEM_FIELDS):
            mWorkItems[dst, cutlass.Int32(f)] = mStaging[src, cutlass.Int32(f)]


@cute.jit
def order_body(
    gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    n_threads: cutlass.Constexpr[int],
    order_elems: cutlass.Constexpr[int],
    tidx,
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    mCuSeqlens: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mSched: cute.Tensor | None,
    sKey,
    sIdx,
    sSpread,
):
    """LPT ordering body over ``n_threads`` CTA threads and caller-owned SMEM
    staging (``sKey``/``sIdx`` of ``n_threads * order_elems`` Int32 cells +
    a 2-cell ``sSpread``): bitonic-sort the items by span ``cend - cstart``,
    longest first, into the final table.  Sorts the walk's staged items, or
    with ``gen`` synthesizes the uncut whole-sequence item per (batch, head)
    from ``cu_seqlens`` directly — the no-cuts table.  Thread 0 also zeroes
    every ``sched_ctr`` cell (the main kernels' ticket rings, dirty on exit).
    Runs on the standalone :func:`order_kernel` CTA, or fused into a main
    kernel's CTA 0 prologue.  Internally CTA-wide-barriers; every thread of
    the calling CTA must reach it."""
    capacity = cutlass.const_expr(n_threads * order_elems)
    if cutlass.const_expr(has_sched):
        if tidx == 0:
            si = cutlass.Int32(0)
            while si < mSched.shape[0]:
                mSched[si] = cutlass.Int32(0)
                si = si + cutlass.Int32(1)
    if cutlass.const_expr(gen):
        n = n_tiles
        if tidx == 0:
            mCount[0] = n_tiles
    else:
        n = mCount[0]
    if n > cutlass.Int32(capacity):
        i = tidx
        while i < n:
            write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i, i)
            i = i + cutlass.Int32(n_threads)
    else:
        if tidx == 0:
            sSpread[0] = cutlass.Int32(2147483647)
            sSpread[1] = cutlass.Int32(-2147483648)
        b_pad = cutlass.Int32(1)
        while b_pad < n:
            b_pad = b_pad * cutlass.Int32(2)
        nvvm.barrier_cta_sync()
        kmin = cutlass.Int32(2147483647)
        kmax = cutlass.Int32(-2147483648)
        for e in cutlass.range_constexpr(order_elems):
            i = tidx + cutlass.Int32(e * n_threads)
            if i < n:
                if cutlass.const_expr(gen):
                    batch_idx, head_idx, batch_start, batch_end, num_chunks_b = gen_item_bounds(b_t, n_heads_out, mCuSeqlens, i)
                    key = num_chunks_b
                else:
                    key = mStaging[i, 5] - mStaging[i, 4]
                sKey[i] = key
                sIdx[i] = i
                kmin = kmin if kmin < key else key
                kmax = kmax if kmax > key else key
            elif i < b_pad:
                # pads sort to the end of the descending order
                sKey[i] = cutlass.Int32(-2147483648)
                sIdx[i] = i
        nvvm.atomicrmw("min", sSpread.data_ptr(0), kmin, space=nvvm.SharedSpace.shared_cta)
        nvvm.atomicrmw("max", sSpread.data_ptr(1), kmax, space=nvvm.SharedSpace.shared_cta)
        nvvm.barrier_cta_sync()
        if sSpread[0] == sSpread[1]:
            # every key equal (uniform batches): copy through
            i2 = tidx
            while i2 < n:
                write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i2, i2)
                i2 = i2 + cutlass.Int32(n_threads)
        else:
            k = cutlass.Int32(2)
            while k <= b_pad:
                j = k // cutlass.Int32(2)
                while j > 0:
                    for e in cutlass.range_constexpr(order_elems):
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
            for e in cutlass.range_constexpr(order_elems):
                i = tidx + cutlass.Int32(e * n_threads)
                if i < n:
                    src = sIdx[i]
                    write_item(gen, b_t, n_heads_out, mCuSeqlens, mStaging, mWorkItems, i, src)


@cute.jit
def launch(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    has_sched: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    num_sms: cutlass.Int32,
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
    mSched: cute.Tensor | None,
    n_scan_ctas: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    if cutlass.const_expr(split):
        if cutlass.const_expr(gate_channels > 0):
            scan_kernel(
                b_t,
                log_gate,
                safe_gate,
                gate_channels,
                overhead_chunks,
                n_heads_out,
                n_tiles,
                num_sms,
                ideal_chunks,
                batch_size,
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
            )
        else:
            scan_scalar_kernel(
                b_t,
                log_gate,
                safe_gate,
                overhead_chunks,
                n_heads_out,
                n_tiles,
                num_sms,
                ideal_chunks,
                batch_size,
                mGate,
                mALog,
                mDtBias,
                mCuSeqlens,
                mChunkVals,
                mCount,
            ).launch(
                grid=(n_scan_ctas, (n_heads_out + cutlass.Int32(WARP_SIZE - 1)) // cutlass.Int32(WARP_SIZE), 1),
                block=(SCAN_THREADS, 1, 1),
                stream=stream,
            )
        walk_kernel(
            b_t,
            overhead_chunks,
            n_heads_out,
            n_tiles,
            num_sms,
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
        )


compiled_cache = {}


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
    n_walk_ctas: int
    has_sched: bool


def run_table(r, gate, a_log, dt_bias, cu_seqlens, chunk_scratch, item_scratch, work_items, work_count, sched_ctr, stream) -> None:
    """The lowered split-table launch: no validation, no key build.  Only
    buffers move between calls; every scalar comes from the recipe."""
    r.compiled(
        r.n_heads_out,
        r.n_tiles,
        r.num_sms,
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
        sched_ctr if r.has_sched else None,
        r.n_scan_ctas,
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
    sched_ctr=None,
    split=True,
    stream,
) -> "TableRecipe":
    """Fill ``work_items``/``work_count`` with the split-K partition of
    ``gate`` / ``cu_seqlens (B+1,) int32``, LPT-ordered (longest item
    first).  A 2-D ``(total_tokens, HO)`` gate is the scalar GDN kind; a
    3-D ``(total_tokens, HO, DK)`` gate is the per-key-channel KDA / GDN-2
    kind.  With ``log_gate`` the gate values are natural-log decay instead
    of raw linear alpha.  With ``safe_gate`` the gate holds RAW logits and
    the scan applies the matching transform so cuts land on true decay
    values: per-channel (KDA / GDN-2, ``gate_lower_bound`` required)
    ``gate_lower_bound * sigmoid(exp(a_log) * (g + dt_bias))`` per element;
    scalar (GDN) ``-exp(a_log[h]) * softplus(g + dt_bias[h])`` per head.

    With ``split=False`` the scan and walk never launch: the order kernel
    alone synthesizes the no-cuts table — the uncut whole-sequence item per
    (batch, head), LPT-sorted by sequence length — and ``ideal_chunks`` /
    ``chunk_scratch`` / ``item_scratch`` / the gate contents are unused.
    Batch-invariant mode and coarse checkpoint cadences (cuts may not cross
    a checkpoint period) take this path.

    ``work_items`` and ``item_scratch`` are ``(max_items,
    WORK_ITEM_FIELDS)`` int32 with ``max_items >= max_work_items(...)``
    (``>= n_tiles`` rows and no ``item_scratch`` with ``split=False``);
    ``work_count`` is ``(1,)`` int32.  Every cell of ``sched_ctr`` when
    passed — the main kernels' int32 ticket rings, ``(2,)`` per kernel
    launch that consumes this table, dirty on exit — is zeroed by the
    order kernel, which runs in both modes; the count is zeroed by the
    scan (split) or written by the order kernel (non-split).
    ``chunk_scratch`` is ``(>= chunk_scratch_rows(total_tokens,
    B, b_t), HO)`` fp32 (contents managed here).  Runs entirely on device
    — no host synchronization."""
    if log2_threshold is None:
        log2_threshold = DEFAULT_LOG2_THRESHOLD
    validate_cu_seqlens(cu_seqlens, assumed_align=4)
    if len(gate.shape) not in (2, 3):
        raise ValueError(f"gate must be (total_tokens, HO) or (total_tokens, HO, DK), got {tuple(gate.shape)}")
    gate_channels = gate.shape[2] if len(gate.shape) == 3 else 0
    if gate_channels and gate.stride(2) != 1:
        raise ValueError("per-channel gate channels must be contiguous")
    if gate_channels and gate_channels % 128 == 0 and (
        data_ptr(gate) % 16 != 0 or gate.stride(0) % 4 != 0 or gate.stride(1) % 4 != 0
    ):
        raise ValueError(
            "per-channel gate rows and head slices must be 16-byte aligned "
            "for vectorized scan loads"
        )
    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("safe_gate requires a_log and dt_bias")
    if safe_gate and gate_channels > 0 and gate_lower_bound is None:
        raise ValueError("per-channel safe_gate requires gate_lower_bound")
    if not safe_gate:
        a_log = None
        dt_bias = None
    gate_scale_log2 = float(gate_lower_bound) * RCP_LN2 if safe_gate and gate_channels > 0 else 0.0
    n_heads_out = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    expected_tiles = batch_size * n_heads_out
    if n_tiles != expected_tiles:
        raise ValueError(f"n_tiles must equal batch_size * n_heads_out ({expected_tiles}), got {n_tiles}")
    if (
        len(work_items.shape) != 2
        or work_items.shape[1] != WORK_ITEM_FIELDS
        or str(work_items.dtype) != "torch.int32"
        or not work_items.is_contiguous()
    ):
        raise ValueError(
            f"work_items must be contiguous int32 (rows, {WORK_ITEM_FIELDS}), "
            f"got {tuple(work_items.shape)} {work_items.dtype}"
        )
    if (
        tuple(work_count.shape) != (1,)
        or str(work_count.dtype) != "torch.int32"
        or not work_count.is_contiguous()
    ):
        raise ValueError("work_count must be contiguous int32 with shape (1,)")
    device_index = tensor_device_index(cu_seqlens)
    if current_device() != device_index:
        raise ValueError("the active CUDA device must match cu_seqlens.device")
    device_properties = get_device_properties(device_index)
    if split:
        if ideal_chunks is None or chunk_scratch is None or item_scratch is None:
            raise ValueError("split=True requires ideal_chunks, chunk_scratch, and item_scratch")
        if gate_channels and gate_channels % WARP_SIZE != 0:
            raise ValueError(f"per-channel gate dim must be a multiple of {WARP_SIZE}, got {gate_channels}")
        n_walk_ctas = batch_size * n_heads_out
        need_rows = chunk_scratch_rows(gate.shape[0], batch_size, b_t)
        n_scan_ctas = -(-need_rows // (SCAN_WARPS * SCAN_ROWS_PER_WARP))
        if len(chunk_scratch.shape) != 2 or chunk_scratch.shape[0] < need_rows or chunk_scratch.shape[1] != n_heads_out:
            raise ValueError(f"chunk_scratch must be (>= {need_rows}, {n_heads_out}) fp32, got {tuple(chunk_scratch.shape)}")
        required_rows = max_work_items(
            gate.shape[0], batch_size, n_heads_out, ideal_chunks, b_t, num_sms
        )
        if work_items.shape[0] < required_rows:
            raise ValueError(
                f"work_items requires at least {required_rows} rows for split=True, "
                f"got {work_items.shape[0]}"
            )
        if (
            tuple(item_scratch.shape) != tuple(work_items.shape)
            or str(item_scratch.dtype) != "torch.int32"
            or not item_scratch.is_contiguous()
        ):
            raise ValueError(
                f"item_scratch must match work_items as contiguous int32, "
                f"got {tuple(item_scratch.shape)} {item_scratch.dtype}"
            )
    else:
        if work_items.shape[0] < n_tiles or work_items.shape[1] != WORK_ITEM_FIELDS:
            raise ValueError(f"work_items must be (>= {n_tiles}, {WORK_ITEM_FIELDS}) int32, got {tuple(work_items.shape)}")
        log_gate = False
        safe_gate = False
        gate_channels = 0
        a_log = None
        dt_bias = None
        gate_scale_log2 = 0.0
        gate = None
        chunk_scratch = None
        item_scratch = None
        ideal_chunks = 0
        n_scan_ctas = 0
        n_walk_ctas = 0
    overhead_chunks = max(1, OVERHEAD_TOKENS // b_t)
    cu_stream = cuda.CUstream(int(stream))

    has_sched = sched_ctr is not None

    key = (
        device_index,
        device_properties.major,
        device_properties.minor,
        bool(split),
        b_t,
        n_heads_out,
        bool(log_gate),
        bool(safe_gate),
        gate_channels,
        has_sched,
        str(gate.dtype) if split else None,
        str(a_log.dtype) if safe_gate else None,
        str(dt_bias.dtype) if safe_gate else None,
    )
    if key not in compiled_cache:
        tokens = cute.sym_int()
        sequence_entries = cute.sym_int()
        item_rows = cute.sym_int()
        gate_c = (
            make_fake_strided_tensor(
                get_dtype(gate.dtype),
                (tokens, n_heads_out, gate_channels) if gate_channels else (tokens, n_heads_out),
                assumed_align=4,
            )
            if split
            else None
        )
        a_log_c = (
            make_fake_compact_tensor(
                get_dtype(a_log.dtype),
                (n_heads_out,),
                stride_order=(0,),
                assumed_align=4,
            )
            if safe_gate
            else None
        )
        dt_bias_c = (
            make_fake_compact_tensor(
                get_dtype(dt_bias.dtype),
                (n_heads_out, gate_channels) if gate_channels else (n_heads_out,),
                stride_order=(1, 0) if gate_channels else (0,),
                assumed_align=4,
            )
            if safe_gate
            else None
        )
        chunk_scratch_c = (
            make_fake_compact_tensor(
                cutlass.Float32,
                (cute.sym_int(), n_heads_out),
                stride_order=(1, 0),
                assumed_align=4,
            )
            if split
            else None
        )
        work_items_c = make_work_items_signature(item_rows)
        item_scratch_c = work_items_c if split else None
        work_count_c = make_counter_signature()
        sched_c = make_counter_signature(cute.sym_int()) if has_sched else None
        compiled_cache[key] = cute.compile[cute.EnableTVMFFI](
            launch,
            bool(split),
            b_t,
            bool(log_gate),
            bool(safe_gate),
            gate_channels,
            overhead_chunks,
            has_sched,
            cutlass.Int32(n_heads_out),
            cutlass.Int32(n_tiles),
            cutlass.Int32(num_sms),
            cutlass.Int32(ideal_chunks),
            cutlass.Int32(batch_size),
            cutlass.Float32(log2_threshold),
            cutlass.Float32(gate_scale_log2),
            gate_c,
            a_log_c,
            dt_bias_c,
            make_cu_seqlens_signature(sequence_entries, assumed_align=4),
            chunk_scratch_c,
            item_scratch_c,
            work_items_c,
            work_count_c,
            sched_c,
            cutlass.Int32(n_scan_ctas),
            cutlass.Int32(n_walk_ctas),
            cute.runtime.make_fake_stream(),
        )
    compiled_cache[key](
        n_heads_out,
        n_tiles,
        num_sms,
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
        sched_ctr,
        n_scan_ctas,
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
        n_walk_ctas,
        has_sched,
    )
