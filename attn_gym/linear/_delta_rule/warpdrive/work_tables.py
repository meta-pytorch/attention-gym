# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/common/split_k.py and
# the work-item phase of the kernel prologues), rewritten by Attention Gym in 2026 on CuTeDSL
# 4.6.2 APIs (cute.arch atomics, shuffles and barriers, SmemAllocator tensors, inline PTX vector
# loads) in place of cutlass.experimental. The tables are the upstream ones, byte for byte; the
# per-batch TMA-descriptor arrays the upstream prologues also built are gone (the warpdrive kernels
# address every sequence through ragged descriptors).
#
# CuTeDSL traces if-expressions as device selects, which min/max builtins do not reproduce on
# dynamic values, and a nested `if` guards its loads, which a merged `a and b` condition would not.
# ruff: noqa: FURB136, SIM102

"""Work-item tables of the warpdrive GDN/KDA kernels: the split-K partition and the LPT order.

The main kernels are persistent: each CTA claims rows of a work-item table. A row is
``WORK_ITEM_FIELDS`` int32 in chunk units::

    [batch_idx, head_idx, write_start, write_end, compute_start, compute_end, batch_start,
     batch_end, final_dst, dstate_dst]

The item writes outputs for ``[write_start, write_end)``. Forward computes ``[compute_start,
write_end)``, backward ``[write_start, compute_end)``; the extra chunks on each side are
warmups, accurate to ``2^log2_threshold``. ``compute_start == 0`` seeds the true initial state
and ``compute_end == num_chunks`` the true ``d_final_state``, so the uncut item ``(0, nc, 0,
nc)`` reproduces the serial kernel exactly. ``final_dst`` is the sequence whose final state
the item stores (the item writing the last chunk), ``dstate_dst`` the sequence whose
``d_initial_state`` it stores (the item writing chunk 0); ``-1`` stores nothing. In chain mode
``batch_idx`` is the piece slot and the destinations name the sequence.

The split table (:func:`launch_split_table`) cuts a sequence where the natural-log forget gates
saturate on both sides of the cut, so no state crosses it:

0. plan: one thread settles the batch-wide span the piece choice may be overridden with and
   leaves it in the chunk scratch's last row (``-1`` when nothing can be cut).
1. scan: reduce the gate to one clamped-log2 value per chunk into the chunk scratch (channel
   gates: max over the per-channel sums, read only near a candidate boundary; scalar gates:
   the plain sum, read whole).
2. walk: one CTA per (batch, head). Scalar gates first price the expected warmup into the
   piece count. Then one warp per candidate boundary prefix-sums the boundary's chunk window
   and binary-searches, one lane per snap offset, the smallest saturating warmup on each side,
   scoring positions within ``SNAP_CHUNKS`` on ``warmup_before + warmup_after + |offset|``.
   Thread 0 emits every saturating cut into the staging table. Emission order is the atomic
   counter's, so the staged rows are a set; the order phase fixes their order.

Every prologue kernel that reads ``cu_seqlens`` (the split plan, the work table, the chain
prologue and the GDN T-pass row table) first checks it on the device
(:func:`check_packed_offsets`). Each warpdrive call starts with one of them, so invalid offsets
abort the call before any kernel reads through them.

The order phase (:func:`order_body`, launched by :func:`launch_work_table` or the chain
prologue) bitonic-sorts the rows into ``work_items``, longest ``[compute_start, compute_end)``
first, so the claim scheduler runs LPT, and zeroes the scheduler counters. Without a split it
synthesizes the uncut item per (batch, head) from ``cu_seqlens``, compacting away sequences
without tokens (a fresh paged slot to clear stays). The bitonic network is fixed, so the table
is a function of the staged rows in staging order.
"""

import functools
import math
from typing import NamedTuple

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.ffi import ffi, mangle

from attn_gym.warpdrive.cache import (
    compile_tvm_ffi,
    jit_cache,
    make_fake_compact_tensor,
    make_fake_strided_tensor,
)
from attn_gym.warpdrive.compat import SmemAllocator

from .ptx import ld_global_v

WORK_ITEM_FIELDS = 10
WORK_ITEM_FINAL_DST = 8
WORK_ITEM_DSTATE_DST = 9

# ---- split planner ---------------------------------------------------------------------------
WARMUP_CAP_CHANNEL = 32  # per-channel gates: a cut saturates within one warp of chunks per side
WALK_WINDOW_CHUNKS = 1088  # scalar gates: chunk window the walk prefix-sums per boundary
SNAP_CHUNKS = 32  # cut-position search half-width
MAX_BLOCKS = 2048  # piece-count ceiling; ideal_chunks is clamped so a tile's blocks fit
WARP_SIZE = 32
WALK_WARPS = 8
WALK_THREADS = WALK_WARPS * WARP_SIZE
SCAN_WARPS = 4
SCAN_THREADS = SCAN_WARPS * WARP_SIZE
PLAN_THREADS = 128
SCAN_CTA_CAP = 16  # scan CTAs per SM before the block axis becomes a grid-stride loop
SCAN_BLOCK_LOOP_MAX = 4  # block-loop trips the cap may cost
SCAN_ROWS_PER_WARP_MAX = 4  # chunk rows per scan warp; scan_rows_per_warp() shrinks it
OVERHEAD_TOKENS = 256  # per-item cost in the piece model: reseed + pipeline refill + warmup
P_WINDOW = 16  # fill-regime piece-count search width
P_BELOW = 8  # how far below the ideal-cap floor the fill-regime search may go

# The threshold is a margin of bits past the output dtype's relative half-ulp: a rounding can
# flip only where |H_window| < 2^-margin * |H_before|. e^-10 = 2^-14.4, so bf16 (half-ulp 2^-8)
# keeps a 6.4-bit margin and fp16 (half-ulp 2^-11) a 3.4-bit margin.
DEFAULT_LOG2_THRESHOLD = -10.0 / math.log(2.0)
RCP_LN2 = 1.4426950408889634  # natural-log gates -> the scan's log2 domain

# ---- order phase -----------------------------------------------------------------------------
ORDER_THREADS = 1024
ORDER_ELEMENTS = 4
ORDER_CAPACITY = ORDER_THREADS * ORDER_ELEMENTS  # past this the rows copy through unsorted

INT32_MAX = 2147483647
INT32_MIN = -2147483648

# The launch bundles compile these kernels at their level; the standalone entry points match.
OPT_LEVEL = 2


# ---- host-side geometry ----------------------------------------------------------------------


def ceil_div(a: int, b: int) -> int:
    """Exact integer ``ceil(a / b)`` for nonnegative ``a`` and positive ``b``."""
    return (a + b - 1) // b


def compute_ideal_chunks(total_tokens: int, n_heads_out: int, num_sms: int, b_t: int) -> int:
    """Ideal per-item span in chunks: the total work over the SM count, at least one chunk and
    clamped so a tile's block count fits ``MAX_BLOCKS``."""
    total_chunks = ceil_div(total_tokens, b_t)
    ideal_tokens = ceil_div(total_tokens * n_heads_out, num_sms)
    ideal_chunks = max(1, ceil_div(ideal_tokens, b_t))
    return max(ideal_chunks, ceil_div(total_chunks, MAX_BLOCKS))


def warmup_cap_chunks(gate_channels: int) -> int:
    """Chunks per side a cut may recompute: one warp of chunks for per-channel gates (their scan
    is windowed around each candidate boundary), for scalar gates the walk's window minus the
    snap search."""
    if gate_channels > 0:
        return WARMUP_CAP_CHANNEL
    return WALK_WINDOW_CHUNKS // 2 - SNAP_CHUNKS


def scan_rows_per_warp(need_rows: int, grid_y: int, num_sms: int) -> int:
    """Chunk rows each scan warp walks: the largest of ``SCAN_ROWS_PER_WARP_MAX``, 2, 1 whose
    grid still covers the SMs."""
    for rows in (SCAN_ROWS_PER_WARP_MAX, 2, 1):
        if rows == 1 or ceil_div(need_rows, SCAN_WARPS * rows) * grid_y >= num_sms:
            return rows
    raise AssertionError("unreachable")


def chunk_scratch_rows(total_tokens: int, batch_size: int, b_t: int) -> int:
    """Rows of the ``(rows, HO)`` fp32 chunk scratch: sequence ``b``'s chunks start at row
    ``cu[b] // b_t + b`` (one extra row per sequence covers the ceil), plus the last row, which
    carries the plan."""
    return total_tokens // b_t + batch_size + 1


def max_work_items(
    total_tokens: int, batch_size: int, n_heads_out: int, ideal_chunks: int, b_t: int, num_sms: int
) -> int:
    """Upper bound on the split table's rows: spans are capped at ``ideal_chunks`` (at most
    ``ceil(nc / ideal) + 1`` pieces per tile) and the fill-regime search can widen a tile by up
    to ``P_WINDOW - 1`` pieces."""
    total_chunks = ceil_div(total_tokens, b_t)
    window = P_WINDOW - 1 if batch_size * n_heads_out < 2 * num_sms else 0
    return n_heads_out * (ceil_div(total_chunks, ideal_chunks) + (2 + window) * batch_size)


class SplitStatic(NamedTuple):
    """The compile-time half of a split launch (the constexprs of :func:`launch_split_table`):
    hashable, so a launch bundle passes it as one ``Constexpr`` and keys its compile on it."""

    b_t: int
    scan_rows: int
    gate_channels: int
    overhead_chunks: int
    warmup_cap: int
    num_sms: int


class SplitTableFacts(NamedTuple):
    """Everything :func:`launch_split_table` is compiled with (``static``) and fed per call (the
    runtime scalars, in launch order: ``scalars()``), derived once from the gate,
    ``cu_seqlens`` and the machine."""

    static: SplitStatic
    n_heads_out: int
    n_tiles: int
    ideal_chunks: int
    batch_size: int
    log2_threshold: float
    n_scan_ctas: int
    n_scan_blocks: int
    n_walk_ctas: int

    def scalars(self) -> tuple:
        return tuple(self[1:])


def split_table_facts(gate, cu_seqlens, *, n_tiles, ideal_chunks, num_sms, b_t) -> SplitTableFacts:
    """Facts of a split launch over ``gate`` (``(T, HO)`` scalar or ``(T, HO, DK)``
    per-channel natural-log gates) and ``cu_seqlens``."""
    gate_channels = gate.shape[2] if gate.ndim == 3 else 0
    if gate_channels and (gate_channels % WARP_SIZE or gate.stride(2) != 1):
        raise ValueError("per-channel gates need contiguous channels, a multiple of 32")
    # Four channels per lane (128-multiple channel counts) load as one vector.
    vector_bytes = 4 * gate.element_size()
    if (
        gate_channels % (4 * WARP_SIZE) == 0
        and gate_channels
        and (gate.data_ptr() % vector_bytes or gate.stride(0) % 4 or gate.stride(1) % 4)
    ):
        raise ValueError(
            f"per-channel gate rows and head slices must be {vector_bytes}-byte aligned "
            "for vectorized scan loads"
        )
    return _split_table_facts(
        tuple(gate.shape), cu_seqlens.shape[0] - 1, n_tiles, ideal_chunks, num_sms, b_t
    )


@functools.lru_cache(maxsize=256)
def _split_table_facts(
    gate_shape, batch_size, n_tiles, ideal_chunks, num_sms, b_t
) -> SplitTableFacts:
    gate_channels = gate_shape[2] if len(gate_shape) == 3 else 0
    n_heads_out = gate_shape[1]
    need_rows = chunk_scratch_rows(gate_shape[0], batch_size, b_t)
    grid_y = n_heads_out if gate_channels else ceil_div(n_heads_out, WARP_SIZE)
    scan_rows = scan_rows_per_warp(need_rows, grid_y, num_sms)
    n_scan_blocks = ceil_div(need_rows, SCAN_WARPS * scan_rows)
    n_scan_ctas = min(
        n_scan_blocks,
        max(
            max(1, SCAN_CTA_CAP * num_sms // grid_y),
            ceil_div(n_scan_blocks, SCAN_BLOCK_LOOP_MAX),
        ),
    )
    return SplitTableFacts(
        static=SplitStatic(
            b_t=int(b_t),
            scan_rows=int(scan_rows),
            gate_channels=int(gate_channels),
            overhead_chunks=max(1, OVERHEAD_TOKENS // b_t),
            warmup_cap=warmup_cap_chunks(gate_channels),
            num_sms=int(num_sms),
        ),
        n_heads_out=int(n_heads_out),
        n_tiles=int(n_tiles),
        ideal_chunks=int(ideal_chunks),
        batch_size=int(batch_size),
        log2_threshold=DEFAULT_LOG2_THRESHOLD,
        n_scan_ctas=int(n_scan_ctas),
        n_scan_blocks=int(n_scan_blocks),
        n_walk_ctas=int(batch_size * n_heads_out),
    )


# ---- device helpers --------------------------------------------------------------------------


@cute.jit
def cu_at(cu_seqlens, i):
    """One cumulative token boundary as Int32."""
    return Int32(cu_seqlens[i])


@cute.jit
def chunks_of(b_t: cutlass.Constexpr[int], cu_seqlens, b):
    """Chunks of sequence ``b``."""
    return cute.ceil_div(cu_at(cu_seqlens, b + 1) - cu_at(cu_seqlens, b), b_t)


def _string_constant(text: str) -> ir.Value:
    """Pointer to ``text`` as a NUL-terminated constant of the enclosing ``gpu.module``."""
    module = ir.InsertionPoint.current.block.owner
    while module.name != "gpu.module":
        module = module.parent
    data = text + "\0"
    symbol = "wd_str_" + mangle(text)
    if symbol not in ir.SymbolTable(module.operation):
        with ir.InsertionPoint(module.regions[0].blocks[0]):
            llvm.GlobalOp(
                sym_name=symbol,
                global_type=ir.Type.parse(f"!llvm.array<{len(data)} x i8>"),
                linkage=ir.Attribute.parse("#llvm.linkage<internal>"),
                constant=True,
                value=ir.StringAttr.get(data),
            )
    return llvm.AddressOfOp(llvm.PointerType.get(), symbol).result


def assert_fail(message: str, function: str) -> None:
    """CUDA's failed ``assert``: print ``message`` to stderr and abort the context with
    ``CUDA_ERROR_ASSERT``. CuTeDSL drops its own device asserts unless the whole module is
    compiled with assertions enabled, so this calls the driver's ``__assertfail`` directly."""
    ptr = llvm.PointerType.get()
    fail = ffi(name="__assertfail", params_types=[ptr, ptr, Int32, ptr, Int64], inline=False)
    fail(
        _string_constant(message),
        _string_constant(__name__),
        Int32(0),
        _string_constant(function),
        Int64(1),
    )


@cute.jit
def check_packed_offsets(cu_seqlens, tokens: Int32, tidx, n_threads: cutlass.Constexpr[int]):
    """Abort the launch (``invalid packed cu_seqlens``) unless ``cu_seqlens`` starts at 0, never
    decreases and ends within ``tokens``. Every thread of the CTA calls it; the closing barrier
    keeps any of them from addressing through the offsets before all of them passed. Each
    warpdrive call starts with a prologue kernel that runs it (the native KDA summaries instead
    run the same check in their Triton work selector), so no later launch of the call reads
    ``q``/``k``/``v`` through invalid offsets."""
    n_batch = Int32(cu_seqlens.shape[0]) - Int32(1)
    for b in cutlass.range(tidx, n_batch, n_threads, unroll=1):
        begin = cu_at(cu_seqlens, b)
        end = cu_at(cu_seqlens, b + Int32(1))
        invalid = begin > end
        if b == Int32(0):
            invalid = invalid or begin != Int32(0)
        if b == n_batch - Int32(1):
            invalid = invalid or end > tokens
        if invalid:
            assert_fail("invalid packed cu_seqlens", "check_packed_offsets")
    cute.arch.barrier()


@cute.jit
def emit_item(
    mWorkItems,
    mCount,
    batch_idx,
    head_idx,
    write_start,
    write_end,
    compute_start,
    compute_end,
    batch_start,
    batch_end,
    batch_num_chunks,
):
    """Append one row at the next ``mCount`` slot."""
    slot = Int32(cute.arch.atomic_add(mCount.iterator, Int32(1)))
    mWorkItems[slot, 0] = batch_idx
    mWorkItems[slot, 1] = head_idx
    mWorkItems[slot, 2] = write_start
    mWorkItems[slot, 3] = write_end
    mWorkItems[slot, 4] = compute_start
    mWorkItems[slot, 5] = compute_end
    mWorkItems[slot, 6] = batch_start
    mWorkItems[slot, 7] = batch_end
    mWorkItems[slot, WORK_ITEM_FINAL_DST] = (
        batch_idx if write_end == batch_num_chunks else Int32(-1)
    )
    mWorkItems[slot, WORK_ITEM_DSTATE_DST] = batch_idx if write_start == Int32(0) else Int32(-1)


@cute.jit
def clamped_log2(gate_val: Float32) -> Float32:
    """Log2-domain decay of a natural-log gate, clamped to <= 0 (a gate above 1 must not relax
    the horizon)."""
    lg = gate_val * Float32(RCP_LN2)
    return lg if lg < Float32(0.0) else Float32(0.0)


@cute.jit
def piece_choice(
    overhead_chunks: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    batch_num_chunks,
    n_tiles,
    ideal_chunks,
    warmup_est,
):
    """Per-tile piece choice, ``(span, num_blocks)``. ``warmup_est`` is the warmup (chunks)
    every piece is expected to recompute; it joins the per-piece cost but not the uncut tile's."""
    oh = Int32(overhead_chunks) + warmup_est
    # ---- even-spread cap: no span past ideal_chunks ------------------------------------------
    p_hi = batch_num_chunks if batch_num_chunks < Int32(MAX_BLOCKS) else Int32(MAX_BLOCKS)
    p_hi = p_hi if p_hi > 0 else Int32(1)
    p = (batch_num_chunks + ideal_chunks - Int32(1)) // ideal_chunks
    p = p if p > 0 else Int32(1)
    p = p if p < p_hi else p_hi
    if n_tiles < Int32(2) * num_sms:
        # ---- fill regime: wave-quantized search around the cap -------------------------------
        p_start = p - Int32(P_BELOW)
        p_start = p_start if p_start > 0 else Int32(1)
        best = Int32(INT32_MAX)
        for dp in cutlass.range(P_WINDOW):
            candidate = p_start + Int32(dp)
            candidate = candidate if candidate < p_hi else p_hi
            span_c = (batch_num_chunks + candidate - Int32(1)) // candidate
            waves = (n_tiles * candidate + num_sms - Int32(1)) // num_sms
            estimate = waves * (span_c + oh)
            hit = estimate < best
            best = estimate if hit else best
            p = candidate if hit else p

        # ---- margin over uncut ---------------------------------------------------------------
        uncut_estimate = ((n_tiles + num_sms - Int32(1)) // num_sms) * (
            batch_num_chunks + Int32(overhead_chunks)
        )
        if Int32(4) * best > Int32(3) * uncut_estimate:
            p = Int32(1)
    span = Int32(0)
    num_blocks = Int32(0)
    if batch_num_chunks > 0:
        span = (batch_num_chunks + p - Int32(1)) // p
        num_blocks = (batch_num_chunks + span - Int32(1)) // span
    return span, num_blocks


@cute.jit
def common_span(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    n_heads_out: Int32,
    num_sms: cutlass.Constexpr[int],
    n_tiles,
    ideal_chunks,
    batch_size,
    mCuSeqlens,
):
    """Span :func:`span_choice` forces on every tile the per-sequence choice left uncut, or 0.

    :func:`piece_choice` scores ``waves = n_tiles * candidate``, assuming every tile splits the
    same way; in a ragged batch the sequence that sets the makespan under-splits. A span shared
    by the batch has the exact CTA count ``sum_b ceil(nc_b / span)``, which this search scores,
    taking the best span when it beats the uncut estimate by the 3/4 margin. Applies only to a
    ragged batch whose per-sequence choice underfills the SMs."""
    chosen = Int32(0)
    if n_tiles < num_sms:
        # ---- ragged batch test ---------------------------------------------------------------
        max_chunks = Int32(0)
        min_chunks = Int32(INT32_MAX)
        b = Int32(0)
        while b < batch_size:
            nc = chunks_of(b_t, mCuSeqlens, b)
            max_chunks = max_chunks if max_chunks > nc else nc
            min_chunks = min_chunks if min_chunks < nc else nc
            b = b + Int32(1)

        if min_chunks < max_chunks:
            # ---- grid of the per-sequence choice ---------------------------------------------
            blocks = Int32(0)
            b = Int32(0)
            while b < batch_size:
                nc = chunks_of(b_t, mCuSeqlens, b)
                _, nb = piece_choice(overhead_chunks, num_sms, nc, n_tiles, ideal_chunks, Int32(0))
                blocks = blocks + nb
                b = b + Int32(1)

            # ---- underfilled: search a span shared by the batch ------------------------------
            if blocks * n_heads_out < num_sms:
                best = Int32(INT32_MAX)
                common = max_chunks
                for dp in cutlass.range(P_WINDOW):
                    s = (max_chunks + Int32(dp)) // (Int32(dp) + Int32(1))
                    ctas = Int32(0)
                    b = Int32(0)
                    while b < batch_size:
                        ctas = ctas + cute.ceil_div(chunks_of(b_t, mCuSeqlens, b), s)
                        b = b + Int32(1)
                    ctas = ctas * n_heads_out
                    waves = (ctas + num_sms - Int32(1)) // num_sms
                    estimate = waves * (s + Int32(overhead_chunks))
                    hit = estimate < best
                    best = estimate if hit else best
                    common = s if hit else common
                uncut = ((n_tiles + num_sms - Int32(1)) // num_sms) * (
                    max_chunks + Int32(overhead_chunks)
                )
                chosen = common if Int32(4) * best <= Int32(3) * uncut else Int32(0)
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
):
    """:func:`piece_choice`, overridden by the batch-wide ``common`` span (0: none) whenever
    the per-sequence choice left this tile in one piece."""
    span, num_blocks = piece_choice(
        overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, warmup_est
    )
    if (n_tiles < num_sms) and (num_blocks == Int32(1)) and (common > Int32(0)):
        span = common if common < batch_num_chunks else batch_num_chunks
        num_blocks = cute.ceil_div(batch_num_chunks, span) if span > 0 else Int32(0)
    return span, num_blocks


@cute.jit
def near_boundary(window: cutlass.Constexpr[int], c, span, num_blocks):
    """True iff chunk ``c`` lies in the walk's read window of a candidate boundary ``j * span``,
    ``j`` in ``[1, num_blocks)``; ``window`` spans the warmup cap plus the snap search."""
    j0 = c // span
    cm = c - j0 * span
    w = Int32(window)
    pre = (cm < w) and (j0 >= Int32(1))
    suf = (span - cm <= w) and (j0 < num_blocks - Int32(1))
    return pre or suf


@cute.jit
def window_prefix(
    window_len: cutlass.Constexpr[int],
    sPre,
    base,
    lo_c,
    n_w,
    lane_idx,
    chunk_value_base,
    head_idx,
    mChunkVals,
):
    """Inclusive prefix of the chunk values ``[lo_c, lo_c + n_w)`` into ``sPre[base + 1 ..
    base + n_w]`` with ``sPre[base] = 0``: staged lane-strided, each lane sums a contiguous run,
    and the run totals are scanned across the warp."""
    for kk in cutlass.range_constexpr((window_len + WARP_SIZE - 1) // WARP_SIZE):
        i = Int32(kk * WARP_SIZE) + lane_idx
        if i < n_w:
            sPre[base + Int32(1) + i] = mChunkVals[chunk_value_base + lo_c + i, head_idx]
    cute.arch.sync_warp()
    run = (n_w + Int32(WARP_SIZE - 1)) // Int32(WARP_SIZE)
    start = lane_idx * run
    end = start + run
    end = end if end < n_w else n_w
    acc = Float32(0.0)
    i = start
    while i < end:
        acc = acc + sPre[base + Int32(1) + i]
        sPre[base + Int32(1) + i] = acc
        i = i + Int32(1)
    s = acc
    for off in [1, 2, 4, 8, 16]:
        o = Float32(cute.arch.shuffle_sync_up(s, off, mask_and_clamp=0))
        s = s + (o if lane_idx >= Int32(off) else Float32(0.0))
    excl = s - acc
    i = start
    while i < end:
        sPre[base + Int32(1) + i] = sPre[base + Int32(1) + i] + excl
        i = i + Int32(1)
    if lane_idx == 0:
        sPre[base] = Float32(0.0)
    cute.arch.sync_warp()


@cute.jit
def search_fwd(sPre, base, xi, cap: cutlass.Constexpr[int], thresh):
    """Forward warmup at window position ``xi``: the smallest ``s`` in ``[1, min(cap, xi)]``
    whose chunk suffix ``[xi - s, xi)`` saturates, 0 if none (the prefix is non-increasing)."""
    hi = Int32(cap) if Int32(cap) < xi else xi
    result = Int32(0)
    if hi > 0:
        px = sPre[base + xi]
        if px - sPre[base + xi - hi] <= thresh:
            lo = Int32(1)
            while lo < hi:
                mid = (lo + hi) // Int32(2)
                ok = px - sPre[base + xi - mid] <= thresh
                hi = mid if ok else hi
                lo = lo if ok else mid + Int32(1)
            result = lo
    return result


@cute.jit
def search_bwd(sPre, base, xi, n_w, cap: cutlass.Constexpr[int], thresh):
    """Reverse warmup at window position ``xi``: the smallest ``s`` in ``[1, min(cap, n_w -
    xi)]`` whose chunk prefix ``[xi, xi + s)`` saturates, 0 if none."""
    limit = n_w - xi
    hi = Int32(cap) if Int32(cap) < limit else limit
    result = Int32(0)
    if hi > 0:
        px = sPre[base + xi]
        if sPre[base + xi + hi] - px <= thresh:
            lo = Int32(1)
            while lo < hi:
                mid = (lo + hi) // Int32(2)
                ok = sPre[base + xi + mid] - px <= thresh
                hi = mid if ok else hi
                lo = lo if ok else mid + Int32(1)
            result = lo
    return result


@cute.jit
def read_plan(mChunkVals):
    """The batch-wide span the plan left in the scratch's last row, -1 when no tile can be cut
    (spans are small integers, exact in fp32)."""
    return mChunkVals[mChunkVals.shape[0] - Int32(1), 0].to(Int32)


@cute.jit
def locate_row(b_t: cutlass.Constexpr[int], batch_size, mCuSeqlens, row0):
    """The sequence owning chunk-scratch row ``row0``: the last ``b`` with ``cu[b] // b_t + b <=
    row0``, by binary search."""
    lo = Int32(0)
    hi = batch_size - Int32(1)
    while lo < hi:
        mid = (lo + hi + Int32(1)) // Int32(2)
        take = cu_at(mCuSeqlens, mid) // Int32(b_t) + mid <= row0
        lo = mid if take else lo
        hi = hi if take else mid - Int32(1)
    return lo


# ---- split kernels ---------------------------------------------------------------------------


@cute.kernel
def split_plan_kernel(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_heads_out: Int32,
    n_tiles: Int32,
    ideal_chunks: Int32,
    batch_size: Int32,
    tokens: Int32,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mCount: cute.Tensor,
):
    """One CTA: check the offsets against ``tokens``, settle the batch-wide span once, leave it in
    the chunk scratch's last row (-1 when no tile of the batch admits a cut) and zero the item
    count."""
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    tidx = Int32(cute.arch.thread_idx()[0])
    check_packed_offsets(mCuSeqlens, tokens, tidx, PLAN_THREADS)
    sCut = SmemAllocator().allocate_tensor(Int32, cute.make_layout(1), 8)
    if tidx == 0:
        sCut[0] = Int32(0)
    cute.arch.barrier()

    # ---- cut admissibility over the batch: one thread per sequence ---------------------------
    cut = Int32(0)
    b = tidx
    while b < batch_size:
        nc = chunks_of(b_t, mCuSeqlens, b)
        _, nb = piece_choice(overhead_chunks, num_sms, nc, n_tiles, ideal_chunks, Int32(0))
        cut = Int32(1) if nb > Int32(1) else cut
        b = b + Int32(PLAN_THREADS)
    cute.arch.atomic_max(sCut.iterator, cut)
    cute.arch.barrier()

    if tidx == 0:
        mCount[0] = Int32(0)
        any_cut = sCut[0]
        common = common_span(
            b_t,
            overhead_chunks,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            mCuSeqlens,
        )
        any_cut = Int32(1) if common > Int32(0) else any_cut
        plan = Float32(common) if any_cut > Int32(0) else Float32(-1.0)
        mChunkVals[mChunkVals.shape[0] - Int32(1), 0] = plan


@cute.jit
def scan_row_sequence(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_tiles,
    ideal_chunks,
    common,
    mCuSeqlens,
    batch_idx,
):
    """``(batch_start, batch_end, batch_num_chunks, chunk_value_base, span, num_blocks)`` of
    sequence ``batch_idx`` for the scans."""
    batch_start = cu_at(mCuSeqlens, batch_idx)
    batch_end = cu_at(mCuSeqlens, batch_idx + 1)
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    chunk_value_base = batch_start // Int32(b_t) + batch_idx
    span, num_blocks = span_choice(
        overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common, Int32(0)
    )
    return batch_start, batch_end, batch_num_chunks, chunk_value_base, span, num_blocks


@cute.kernel
def split_scan_channel_kernel(
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_tiles: Int32,
    ideal_chunks: Int32,
    batch_size: Int32,
    n_scan_blocks: Int32,
    mGate: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
):
    """Windowed chunk scan of a per-channel gate (KDA): CTA ``(x, h)`` covers chunk-scratch rows
    of head ``h``, one warp per row, lane ``l`` owning channels ``[l * cpl, (l + 1) * cpl)``.
    Chunks outside a cut window never touch the gate."""
    cute.arch.griddepcontrol_wait()
    tidx = Int32(cute.arch.thread_idx()[0])
    bidx = cute.arch.block_idx()
    head_idx = Int32(bidx[1])
    lane_idx = tidx % Int32(WARP_SIZE)
    widx = tidx // Int32(WARP_SIZE)
    cpl = cutlass.const_expr(gate_channels // WARP_SIZE)
    gate_elem_bytes = cutlass.const_expr(mGate.element_type.width // 8)
    common = read_plan(mChunkVals)
    if common >= Int32(0):
        blk = Int32(bidx[0])
        while blk < n_scan_blocks:
            row0 = (blk * Int32(SCAN_WARPS) + widx) * Int32(scan_rows)
            batch_idx = locate_row(b_t, batch_size, mCuSeqlens, row0)
            batch_start, batch_end, batch_num_chunks, chunk_value_base, span, num_blocks = (
                scan_row_sequence(
                    b_t,
                    overhead_chunks,
                    num_sms,
                    n_tiles,
                    ideal_chunks,
                    common,
                    mCuSeqlens,
                    batch_idx,
                )
            )
            for rr in cutlass.range_constexpr(scan_rows):
                row = row0 + Int32(rr)
                while (batch_idx + Int32(1) < batch_size) and (
                    cu_at(mCuSeqlens, batch_idx + 1) // Int32(b_t) + batch_idx + Int32(1) <= row
                ):
                    batch_idx = batch_idx + Int32(1)
                    (
                        batch_start,
                        batch_end,
                        batch_num_chunks,
                        chunk_value_base,
                        span,
                        num_blocks,
                    ) = scan_row_sequence(
                        b_t,
                        overhead_chunks,
                        num_sms,
                        n_tiles,
                        ideal_chunks,
                        common,
                        mCuSeqlens,
                        batch_idx,
                    )
                c = row - chunk_value_base
                if (c >= 0) and (c < batch_num_chunks) and (num_blocks > 1):
                    if near_boundary(
                        warmup_cap + SNAP_CHUNKS,
                        c,
                        span if span > 0 else Int32(1),
                        num_blocks,
                    ):
                        # ---- per-channel clamped-log2 sums, one channel run per lane ---------
                        row_elements = Int32(mGate.stride[0])
                        lane_base = Int64(
                            head_idx * Int32(mGate.stride[1]) + lane_idx * Int32(cpl)
                        )
                        gate_addr = mGate.iterator.toint() + lane_base * Int64(gate_elem_bytes)
                        gate_ptr = mGate.iterator + lane_base
                        ch_acc = cute.make_rmem_tensor((cpl,), Float32)
                        for q in cutlass.range_constexpr(cpl):
                            ch_acc[q] = Float32(0.0)
                        for tt in cutlass.range(0, b_t, 1, unroll_full=True):
                            pos = batch_start + c * Int32(b_t) + Int32(tt)
                            inb = pos < batch_end
                            pos_r = pos if inb else batch_start
                            grow = Int64(pos_r) * Int64(row_elements)
                            if cutlass.const_expr(cpl % 4 == 0):
                                for q4 in cutlass.range_constexpr(cpl // 4):
                                    addr = gate_addr + (grow + Int64(4 * q4)) * Int64(
                                        gate_elem_bytes
                                    )
                                    if cutlass.const_expr(mGate.element_type is Float32):
                                        quad = ld_global_v(addr, Float32, 4)
                                    else:
                                        w0, w1 = ld_global_v(addr, Int32, 2)
                                        frag = (
                                            cutlass.Vector.from_elements((w0, w1), Int32)
                                            .bitcast(mGate.element_type)
                                            .to(Float32)
                                        )
                                        quad = (frag[0], frag[1], frag[2], frag[3])
                                    for qq, gvq in enumerate(quad):
                                        q = 4 * q4 + qq
                                        gvq = gvq if inb else Float32(0.0)
                                        ch_acc[q] = ch_acc[q] + clamped_log2(gvq)
                            else:
                                for q in cutlass.range_constexpr(cpl):
                                    gv = (gate_ptr + grow + Int32(q)).load().to(Float32)
                                    gv = gv if inb else Float32(0.0)
                                    ch_acc[q] = ch_acc[q] + clamped_log2(gv)

                        # ---- chunk value = max over channels, then over lanes ----------------
                        m = ch_acc[0]
                        for q in cutlass.range_constexpr(1, cpl):
                            m = m if m > ch_acc[q] else ch_acc[q]
                        for off in [1, 2, 4, 8, 16]:
                            other = Float32(cute.arch.shuffle_sync_bfly(m, off))
                            m = m if m > other else other
                        if lane_idx == 0:
                            mChunkVals[chunk_value_base + c, head_idx] = m
            blk = blk + Int32(cute.arch.grid_dim()[0])
    cute.arch.griddepcontrol_launch_dependents()


@cute.kernel
def split_scan_scalar_kernel(
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
    n_heads_out: Int32,
    n_tiles: Int32,
    ideal_chunks: Int32,
    batch_size: Int32,
    n_scan_blocks: Int32,
    mGate: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
):
    """Chunk scan of a scalar gate (GDN), read whole: CTA ``(x, hg)`` covers chunk-scratch rows
    for heads ``[hg * 32, (hg + 1) * 32)``, lane ``l`` owning head ``hg * 32 + l``, so reads and
    writes coalesce and no reduction is needed."""
    cute.arch.griddepcontrol_wait()
    tidx = Int32(cute.arch.thread_idx()[0])
    bidx = cute.arch.block_idx()
    lane_idx = tidx % Int32(WARP_SIZE)
    widx = tidx // Int32(WARP_SIZE)
    h = Int32(bidx[1]) * Int32(WARP_SIZE) + lane_idx
    h_ok = h < n_heads_out
    h_r = h if h_ok else Int32(n_heads_out - 1)
    common = read_plan(mChunkVals)
    if common >= Int32(0):
        blk = Int32(bidx[0])
        while blk < n_scan_blocks:
            row0 = (blk * Int32(SCAN_WARPS) + widx) * Int32(scan_rows)
            batch_idx = locate_row(b_t, batch_size, mCuSeqlens, row0)
            batch_start, batch_end, batch_num_chunks, chunk_value_base, _span, num_blocks = (
                scan_row_sequence(
                    b_t,
                    overhead_chunks,
                    num_sms,
                    n_tiles,
                    ideal_chunks,
                    common,
                    mCuSeqlens,
                    batch_idx,
                )
            )
            for rr in cutlass.range_constexpr(scan_rows):
                row = row0 + Int32(rr)
                while (batch_idx + Int32(1) < batch_size) and (
                    cu_at(mCuSeqlens, batch_idx + 1) // Int32(b_t) + batch_idx + Int32(1) <= row
                ):
                    batch_idx = batch_idx + Int32(1)
                    (
                        batch_start,
                        batch_end,
                        batch_num_chunks,
                        chunk_value_base,
                        _span,
                        num_blocks,
                    ) = scan_row_sequence(
                        b_t,
                        overhead_chunks,
                        num_sms,
                        n_tiles,
                        ideal_chunks,
                        common,
                        mCuSeqlens,
                        batch_idx,
                    )
                c = row - chunk_value_base
                if (c >= 0) and (c < batch_num_chunks) and (num_blocks > 1):
                    # ---- clamped-log2 sum over the chunk, one head per lane --------------
                    acc = Float32(0.0)
                    for tt in cutlass.range(0, b_t, 1, unroll_full=True):
                        pos = batch_start + c * Int32(b_t) + Int32(tt)
                        inb = pos < batch_end
                        pos_r = pos if inb else batch_start
                        gv = (
                            (
                                mGate.iterator
                                + Int64(pos_r) * Int64(mGate.stride[0])
                                + Int64(h_r) * Int64(mGate.stride[1])
                            )
                            .load()
                            .to(Float32)
                        )
                        gv = gv if inb else Float32(0.0)
                        acc = acc + clamped_log2(gv)
                    if h_ok:
                        mChunkVals[chunk_value_base + c, h] = acc
            blk = blk + Int32(cute.arch.grid_dim()[0])
    cute.arch.griddepcontrol_launch_dependents()


@cute.kernel
def split_walk_kernel(
    b_t: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    full_scan: cutlass.Constexpr[bool],
    num_sms: cutlass.Constexpr[int],
    n_heads_out: Int32,
    n_tiles: Int32,
    ideal_chunks: Int32,
    log2_thresh: Float32,
    mCuSeqlens: cute.Tensor,
    mChunkVals: cute.Tensor,
    mStaging: cute.Tensor,
    mCount: cute.Tensor,
):
    """One CTA per (batch, head). With ``full_scan`` the tile's chunk values are summed first: a
    tile that does not saturate as a whole stays uncut, otherwise the mean decay gives the warmup
    every piece will recompute and the piece count is re-chosen with it priced in. Then every
    candidate boundary is probed by its own warp; thread 0 emits every saturating cut."""
    cute.arch.griddepcontrol_wait()
    tidx = Int32(cute.arch.thread_idx()[0])
    tile = Int32(cute.arch.block_idx()[0])
    lane_idx = tidx % Int32(WARP_SIZE)
    widx = tidx // Int32(WARP_SIZE)
    zero = Int32(0)
    window_len = cutlass.const_expr(2 * (warmup_cap + SNAP_CHUNKS) + 1)
    smem = SmemAllocator()
    sPartial = smem.allocate_tensor(Float32, cute.make_layout(WALK_WARPS), 16)
    sWarmup = smem.allocate_tensor(Int32, cute.make_layout(MAX_BLOCKS), 16)
    sPre = smem.allocate_tensor(Float32, cute.make_layout(WALK_WARPS * window_len), 16)

    common = read_plan(mChunkVals)

    # ---- tile decode: chunk_value_base is the row base in the chunk scratch ------------------
    batch_idx = tile // n_heads_out
    head_idx = tile % n_heads_out
    batch_start = cu_at(mCuSeqlens, batch_idx)
    batch_end = cu_at(mCuSeqlens, batch_idx + 1)
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    chunk_value_base = batch_start // Int32(b_t) + batch_idx
    span = Int32(0)
    num_blocks = Int32(1)
    if common >= Int32(0):
        span, num_blocks = span_choice(
            overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, common, zero
        )

    if cutlass.const_expr(full_scan):
        if num_blocks > 1:
            # ---- whole-tile decay: no cut can saturate if the tile does not ------------------
            acc = Float32(0.0)
            c = tidx
            while c < batch_num_chunks:
                acc = acc + mChunkVals[chunk_value_base + c, head_idx]
                c = c + Int32(WALK_THREADS)
            for off in [16, 8, 4, 2, 1]:
                acc = acc + Float32(cute.arch.shuffle_sync_bfly(acc, off))
            if lane_idx == 0:
                sPartial[widx] = acc
            cute.arch.barrier()
            total = Float32(0.0)
            for w in cutlass.range_constexpr(WALK_WARPS):
                total = total + sPartial[w]
            if total > log2_thresh:
                num_blocks = Int32(1)
            else:
                # ---- mean decay -> chunks per warmup; re-choose the pieces with it priced in -
                w_f = (batch_num_chunks.to(Float32) * log2_thresh) / total
                w_est = (w_f + Float32(0.999)).to(Int32)
                w_est = w_est if w_est > 0 else Int32(1)
                _, nb0 = piece_choice(
                    overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, zero
                )
                span_w, nb_w = piece_choice(
                    overhead_chunks, num_sms, batch_num_chunks, n_tiles, ideal_chunks, w_est
                )
                if (nb0 == Int32(1)) and (common > Int32(0)):
                    # ---- ragged underfill override under the 3/4 margin ------------------
                    s_c = common if common < batch_num_chunks else batch_num_chunks
                    oh = Int32(overhead_chunks)
                    if Int32(4) * (s_c + oh + w_est) <= Int32(3) * (batch_num_chunks + oh):
                        span_w = s_c
                        nb_w = cute.ceil_div(batch_num_chunks, s_c)
                    else:
                        nb_w = Int32(1)
                span = span_w
                num_blocks = nb_w

    if num_blocks <= 1:
        # Empty sequences have no chunks and must not enter the persistent work queue.
        if (tidx == 0) and (batch_num_chunks > 0):
            emit_item(
                mStaging,
                mCount,
                batch_idx,
                head_idx,
                Int32(0),
                batch_num_chunks,
                Int32(0),
                batch_num_chunks,
                batch_start,
                batch_end,
                batch_num_chunks,
            )

    else:
        # ---- probe every candidate boundary: one warp each -----------------------------------
        big = Int32(INT32_MAX)
        cap = Int32(warmup_cap)
        half = span // Int32(2)
        half = half if half < Int32(SNAP_CHUNKS) else Int32(SNAP_CHUNKS)
        base = widx * Int32(window_len)
        j = widx + Int32(1)
        while j < num_blocks:
            target = j * span
            lo_c = target - half - cap
            lo_c = lo_c if lo_c > 0 else Int32(0)
            hi_c = target + half + cap
            hi_c = hi_c if hi_c < batch_num_chunks else batch_num_chunks
            n_w = hi_c - lo_c
            window_prefix(
                window_len, sPre, base, lo_c, n_w, lane_idx, chunk_value_base, head_idx, mChunkVals
            )

            # ---- one lane per snap offset, |d| order 0, +1, -1, +2, -2, ... ------------------
            best_key = big
            best_pack = Int32(0)
            kmin = big
            n_off = Int32(2) * half + Int32(1)
            i = lane_idx
            searching = True
            while searching:
                ad = (i + Int32(1)) // Int32(2)
                neg = i % Int32(2) == Int32(0)
                d = -ad if neg else ad
                x = target + d
                in_range = (x > Int32(0)) and (x < batch_num_chunks)
                xi = (x if in_range else target) - lo_c
                wb = search_fwd(sPre, base, xi, warmup_cap, log2_thresh)
                wa = search_bwd(sPre, base, xi, n_w, warmup_cap, log2_thresh)
                key = ((wb + wa + ad) * Int32(64) + ad) * Int32(2) + (
                    Int32(1) if neg else Int32(0)
                )
                live = in_range and (i < n_off) and (wb > 0) and (wa > 0) and (key < best_key)
                best_key = key if live else best_key
                best_pack = (
                    (wb + wa * Int32(1024) + (d + Int32(SNAP_CHUNKS)) * Int32(1048576))
                    if live
                    else best_pack
                )
                i = i + Int32(WARP_SIZE)
                # ---- prune: the next pass's offsets cannot beat a pair found closer --------
                kmin = best_key
                for off in [16, 8, 4, 2, 1]:
                    other = Int32(cute.arch.shuffle_sync_bfly(kmin, off))
                    kmin = kmin if kmin < other else other
                ad_next = (i - lane_idx + Int32(1)) // Int32(2)
                floor_next = (Int32(2) + ad_next) * Int32(128)
                searching = (i - lane_idx < n_off) and (floor_next < kmin)
            if (kmin < big) and (best_key == kmin):
                sWarmup[j] = best_pack
            if (kmin == big) and (lane_idx == 0):
                sWarmup[j] = Int32(0)
            j = j + Int32(WALK_WARPS)
        cute.arch.barrier()

        # ---- emit: thread 0 walks the accepted cuts in order ---------------------------------
        if tidx == 0:
            prev_cut = Int32(0)
            current_compute_start = Int32(0)
            jj = Int32(1)
            while jj < num_blocks:
                r = sWarmup[jj]
                if r != 0:
                    warmup_before = r % Int32(1024)
                    warmup_after = (r // Int32(1024)) % Int32(1024)
                    write_end = jj * span + (r // Int32(1048576) - Int32(SNAP_CHUNKS))
                    compute_end = write_end + warmup_after
                    compute_end = (
                        compute_end if compute_end < batch_num_chunks else batch_num_chunks
                    )
                    emit_item(
                        mStaging,
                        mCount,
                        batch_idx,
                        head_idx,
                        prev_cut,
                        write_end,
                        current_compute_start,
                        compute_end,
                        batch_start,
                        batch_end,
                        batch_num_chunks,
                    )
                    current_compute_start = write_end - warmup_before
                    prev_cut = write_end
                jj = jj + Int32(1)
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
    cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def launch_split_table(
    static: cutlass.Constexpr,
    n_heads_out: Int32,
    n_tiles: Int32,
    ideal_chunks: Int32,
    batch_size: Int32,
    log2_thresh: Float32,
    n_scan_ctas: Int32,
    n_scan_blocks: Int32,
    n_walk_ctas: Int32,
    gate: cute.Tensor,
    cu_seqlens: cute.Tensor,
    chunk_scratch: cute.Tensor,
    staging: cute.Tensor,
    work_count: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """Plan, scan and walk: the split rows into ``staging`` and their count into
    ``work_count``. ``static`` is a :class:`SplitStatic` and the scalars follow
    :meth:`SplitTableFacts.scalars`; callable from a launch bundle's ``@cute.jit`` host. The
    order phase consumes the staged rows."""
    b_t, scan_rows, gate_channels, overhead_chunks, warmup_cap, num_sms = static
    split_plan_kernel(
        b_t,
        overhead_chunks,
        num_sms,
        n_heads_out,
        n_tiles,
        ideal_chunks,
        batch_size,
        Int32(gate.shape[0]),
        cu_seqlens,
        chunk_scratch,
        work_count,
    ).launch(grid=(1, 1, 1), block=(PLAN_THREADS, 1, 1), stream=stream, use_pdl=True)
    if cutlass.const_expr(gate_channels > 0):
        split_scan_channel_kernel(
            b_t,
            scan_rows,
            warmup_cap,
            gate_channels,
            overhead_chunks,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            n_scan_blocks,
            gate,
            cu_seqlens,
            chunk_scratch,
        ).launch(
            grid=(n_scan_ctas, n_heads_out, 1),
            block=(SCAN_THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )
    else:
        split_scan_scalar_kernel(
            b_t,
            scan_rows,
            overhead_chunks,
            num_sms,
            n_heads_out,
            n_tiles,
            ideal_chunks,
            batch_size,
            n_scan_blocks,
            gate,
            cu_seqlens,
            chunk_scratch,
        ).launch(
            grid=(n_scan_ctas, cute.ceil_div(n_heads_out, WARP_SIZE), 1),
            block=(SCAN_THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
        )
    split_walk_kernel(
        b_t,
        overhead_chunks,
        warmup_cap,
        gate_channels == 0,
        num_sms,
        n_heads_out,
        n_tiles,
        ideal_chunks,
        log2_thresh,
        cu_seqlens,
        chunk_scratch,
        staging,
        work_count,
    ).launch(grid=(n_walk_ctas, 1, 1), block=(WALK_THREADS, 1, 1), stream=stream, use_pdl=True)


# ---- order phase -----------------------------------------------------------------------------


@cute.jit
def gen_item_bounds(b_t: cutlass.Constexpr[int], n_heads_out, mCuSeqlens, item):
    """``(batch, head, batch_start, batch_end, num_chunks)`` of uncut item ``item``."""
    batch_idx = item // n_heads_out
    head_idx = item % n_heads_out
    batch_start = cu_at(mCuSeqlens, batch_idx)
    batch_end = cu_at(mCuSeqlens, batch_idx + 1)
    batch_num_chunks = cute.ceil_div(batch_end - batch_start, b_t)
    return batch_idx, head_idx, batch_start, batch_end, batch_num_chunks


@cute.jit
def piece_item_bounds(
    b_t: cutlass.Constexpr[int], n_heads_out, mCuPieces, mRowBase, mSlotRows, item
):
    """``(slot, head, row_start, row_end, num_chunks, final_dst, dstate_dst)`` of piece-table
    row ``item``: the owning sequence by binary search over ``mRowBase`` (each sequence's first
    row in this table, ``[num_seqs + 1]``), then the slot and head within its block of
    ``filled_slots * n_heads_out`` rows. The flat slot is ``item // n_heads_out`` in the main
    table (``mSlotRows`` None), else the sequence's first main slot ``mSlotRows[seq] //
    n_heads_out`` plus the slot within the sequence; the last filled slot stores
    ``final_state``, slot 0 ``d_initial_state``."""
    lo = Int32(0)
    hi = Int32(mRowBase.shape[0]) - Int32(2)
    while lo < hi:
        mid = (lo + hi + Int32(1)) >> Int32(1)
        if mRowBase[mid] <= item:
            lo = mid
        else:
            hi = mid - Int32(1)
    batch_idx = lo
    local = item - mRowBase[batch_idx]
    slot_in_seq = local // n_heads_out
    head_idx = local - slot_in_seq * n_heads_out
    last = (mRowBase[batch_idx + 1] - mRowBase[batch_idx]) // n_heads_out - Int32(1)
    if cutlass.const_expr(mSlotRows is None):
        slot = item // n_heads_out
    else:
        slot = mSlotRows[batch_idx] // n_heads_out + slot_in_seq
    row_start = cu_at(mCuPieces, slot)
    row_end = cu_at(mCuPieces, slot + 1)
    num_chunks = cute.ceil_div(row_end - row_start, b_t)
    final_dst = batch_idx if slot_in_seq == last else Int32(-1)
    dstate_dst = batch_idx if slot_in_seq == Int32(0) else Int32(-1)
    return slot, head_idx, row_start, row_end, num_chunks, final_dst, dstate_dst


@cute.jit
def write_item(
    source: cutlass.Constexpr[str],
    b_t: cutlass.Constexpr[int],
    n_heads_out,
    mCuSeqlens,
    mStaging,
    mWorkItems,
    dst,
    src,
    mRowBase,
    mSlotRows,
):
    """Table row ``dst`` from source item ``src``: the staged row (``"staged"``), the
    synthesized uncut item ``(0, nc, 0, nc)`` (``"uncut"``) or the whole-piece item of the
    piece table (``"chain"``, ``mCuSeqlens`` being ``cu_pieces``)."""
    if cutlass.const_expr(source == "chain"):
        slot, head_idx, row_start, row_end, num_chunks, final_dst, dstate_dst = piece_item_bounds(
            b_t, n_heads_out, mCuSeqlens, mRowBase, mSlotRows, src
        )
        mWorkItems[dst, 0] = slot
        mWorkItems[dst, 1] = head_idx
        mWorkItems[dst, 2] = Int32(0)
        mWorkItems[dst, 3] = num_chunks
        mWorkItems[dst, 4] = Int32(0)
        mWorkItems[dst, 5] = num_chunks
        mWorkItems[dst, 6] = row_start
        mWorkItems[dst, 7] = row_end
        mWorkItems[dst, WORK_ITEM_FINAL_DST] = final_dst
        mWorkItems[dst, WORK_ITEM_DSTATE_DST] = dstate_dst
    elif cutlass.const_expr(source == "uncut"):
        batch_idx, head_idx, batch_start, batch_end, batch_num_chunks = gen_item_bounds(
            b_t, n_heads_out, mCuSeqlens, src
        )
        mWorkItems[dst, 0] = batch_idx
        mWorkItems[dst, 1] = head_idx
        mWorkItems[dst, 2] = Int32(0)
        mWorkItems[dst, 3] = batch_num_chunks
        mWorkItems[dst, 4] = Int32(0)
        mWorkItems[dst, 5] = batch_num_chunks
        mWorkItems[dst, 6] = batch_start
        mWorkItems[dst, 7] = batch_end
        mWorkItems[dst, WORK_ITEM_FINAL_DST] = batch_idx
        mWorkItems[dst, WORK_ITEM_DSTATE_DST] = batch_idx
    else:
        for f in cutlass.range_constexpr(WORK_ITEM_FIELDS):
            mWorkItems[dst, Int32(f)] = mStaging[src, Int32(f)]


@cute.jit
def item_key(
    source: cutlass.Constexpr[str],
    b_t: cutlass.Constexpr[int],
    n_heads_out,
    mCuSeqlens,
    mStaging,
    i,
    src,
    mRowBase,
    mSlotRows,
):
    """The sort key of row ``i`` (source ``src``): its ``compute_end - compute_start``."""
    if cutlass.const_expr(source == "chain"):
        key = piece_item_bounds(b_t, n_heads_out, mCuSeqlens, mRowBase, mSlotRows, i)[4]
    elif cutlass.const_expr(source == "uncut"):
        key = gen_item_bounds(b_t, n_heads_out, mCuSeqlens, src)[4]
    else:
        key = mStaging[i, 5] - mStaging[i, 4]
    return key


@cute.jit
def compact_sequences(
    n_threads: cutlass.Constexpr[int],
    tidx,
    n_batch: Int32,
    mCuSeqlens: cute.Tensor,
    sIdx,
    sWarp,
    mStateIndices: cute.Tensor | None,
    mHasInitialState: cute.Tensor | None,
):
    """Write, in sequence order, the ids with tokens or a fresh paged slot to clear into
    ``sIdx[0, M)`` and return ``M``; a warp ballot ranks each round of ``n_threads`` sequences.
    ``sWarp`` holds ``n_threads // 32`` Int32 cells. A fresh slot is an active route
    (``state_indices[b] > 0``) whose ``has_initial_state[b]`` is 0."""
    lane = cute.arch.lane_idx()
    warp = tidx // Int32(32)
    base = Int32(0)
    start = Int32(0)
    while start < n_batch:
        b = start + tidx
        keep = cutlass.Boolean(False)
        if b < n_batch:
            keep = mCuSeqlens[b + 1] > mCuSeqlens[b]
            if cutlass.const_expr(mStateIndices is not None and mHasInitialState is not None):
                clear_empty = mStateIndices[b] > 0 and mHasInitialState[b] == 0
                keep = keep or clear_empty
        ballot = cute.arch.vote_ballot_sync(keep)
        rank = Int32(cute.arch.popc(cutlass.Uint32(ballot) & cute.arch.lanemask_lt()))
        if lane == 0:
            sWarp[warp] = Int32(cute.arch.popc(cutlass.Uint32(ballot)))
        cute.arch.barrier()
        offset = base
        total = Int32(0)
        for w in cutlass.range_constexpr(n_threads // 32):
            count = sWarp[w]
            total = total + count
            if Int32(w) < warp:
                offset = offset + count
        if keep:
            sIdx[offset + rank] = b
        base = base + total
        cute.arch.barrier()
        start = start + Int32(n_threads)
    return base


@cute.jit
def order_body(
    source: cutlass.Constexpr[str],
    b_t: cutlass.Constexpr[int],
    tidx,
    n_heads_out: Int32,
    n_tiles: Int32,
    mCuSeqlens: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor | None,
    sKey,
    sIdx,
    sSpread,
    num_ctas: cutlass.Constexpr[int] = 0,
    mRowBase: cute.Tensor | None = None,
    mSlotRows: cute.Tensor | None = None,
    mStateIndices: cute.Tensor | None = None,
    mHasInitialState: cute.Tensor | None = None,
):
    """Bitonic-sort the table rows by ``compute_end - compute_start``, longest first, into
    ``work_items``. ``source`` is ``"staged"`` (the split walk's ``mStaging`` rows, ``mCount``
    of them), ``"uncut"`` (one synthesized item per (batch, head) of ``cu_seqlens``) or
    ``"chain"`` (whole pieces of ``cu_pieces``; ``mRowBase`` is this table's per-sequence row
    base and ``mSlotRows`` the main table's, which numbers the flat slots; ``mCount`` holds the
    row count). Thread 0 zeroes ``mScheduler`` when passed. Up to ``num_ctas`` rows (one per
    CTA) and past ``ORDER_CAPACITY`` the rows copy through unsorted. ``sKey`` / ``sIdx`` hold
    ``ORDER_CAPACITY`` Int32 and ``sSpread`` two. One CTA of ``ORDER_THREADS`` threads, all of
    which must call it (CTA barriers inside)."""
    # ---- zero the scheduler counters ---------------------------------------------------------
    if cutlass.const_expr(mScheduler is not None):
        if tidx == 0:
            si = Int32(0)
            while si < mScheduler.shape[0]:
                mScheduler[si] = Int32(0)
                si = si + Int32(1)

    # The uncut table emits no item for a sequence without tokens (unless its fresh paged slot
    # needs clearing), so padded cu_seqlens cost neither sort slots nor main-kernel tiles.
    # Compacted item ``i`` maps back to ``sequence * n_heads_out + head``; batches past the
    # sort capacity keep the uncompacted table.
    compact = False
    if cutlass.const_expr(source == "uncut"):
        n_batch = n_tiles // n_heads_out
        compact = n_batch <= Int32(ORDER_CAPACITY)
        n = n_tiles
        if compact:
            n = (
                compact_sequences(
                    ORDER_THREADS,
                    tidx,
                    n_batch,
                    mCuSeqlens,
                    sIdx,
                    sKey,
                    mStateIndices,
                    mHasInitialState,
                )
                * n_heads_out
            )
        if tidx == 0:
            mCount[0] = n
    else:
        n = mCount[0]

    # ---- copy through unsorted: past sort capacity, or one item per CTA ----------------------
    if (n > Int32(ORDER_CAPACITY)) or (n <= Int32(num_ctas)):
        i = tidx
        while i < n:
            src = i
            if compact:
                src = sIdx[i // n_heads_out] * n_heads_out + i % n_heads_out
            write_item(
                source,
                b_t,
                n_heads_out,
                mCuSeqlens,
                mStaging,
                mWorkItems,
                i,
                src,
                mRowBase,
                mSlotRows,
            )
            i = i + Int32(ORDER_THREADS)
    else:
        # Resolve compacted sources before sIdx is reused as the sort permutation.
        sources = []
        for e in cutlass.range_constexpr(ORDER_ELEMENTS):
            i = tidx + Int32(e * ORDER_THREADS)
            src = i
            if compact:
                if i < n:
                    src = sIdx[i // n_heads_out] * n_heads_out + i % n_heads_out
            sources.append(src)
        if tidx == 0:
            sSpread[0] = Int32(INT32_MAX)
            sSpread[1] = Int32(INT32_MIN)

        b_pad = Int32(1)
        while b_pad < n:
            b_pad = b_pad * Int32(2)
        cute.arch.barrier()
        kmin = Int32(INT32_MAX)
        kmax = Int32(INT32_MIN)
        for e in cutlass.range_constexpr(ORDER_ELEMENTS):
            i = tidx + Int32(e * ORDER_THREADS)
            if i < n:
                key = item_key(
                    source,
                    b_t,
                    n_heads_out,
                    mCuSeqlens,
                    mStaging,
                    i,
                    sources[e],
                    mRowBase,
                    mSlotRows,
                )
                sKey[i] = key
                sIdx[i] = sources[e]
                kmin = kmin if kmin < key else key
                kmax = kmax if kmax > key else key
            elif i < b_pad:
                sKey[i] = Int32(INT32_MIN)
                sIdx[i] = i

        cute.arch.atomic_min(sSpread.iterator, kmin)
        cute.arch.atomic_max(sSpread.iterator + 1, kmax)
        cute.arch.barrier()
        if sSpread[0] == sSpread[1]:
            # ---- every key equal: nothing to sort --------------------------------------------
            i2 = tidx
            while i2 < n:
                write_item(
                    source,
                    b_t,
                    n_heads_out,
                    mCuSeqlens,
                    mStaging,
                    mWorkItems,
                    i2,
                    sIdx[i2],
                    mRowBase,
                    mSlotRows,
                )
                i2 = i2 + Int32(ORDER_THREADS)
        else:
            # ---- bitonic sort, descending: k = subsequence width, j = distance ---------------
            k = Int32(2)
            while k <= b_pad:
                j = k // Int32(2)
                while j > 0:
                    for e in cutlass.range_constexpr(ORDER_ELEMENTS):
                        i = tidx + Int32(e * ORDER_THREADS)
                        if i < b_pad:
                            partner = i ^ j
                            if partner > i:
                                ki = sKey[i]
                                kp = sKey[partner]
                                up = (i // k) % Int32(2) == 0
                                dn = (i // k) % Int32(2) == 1
                                swap = (up and ki < kp) or (dn and ki > kp)
                                if swap:
                                    sKey[i] = kp
                                    sKey[partner] = ki
                                    si = sIdx[i]
                                    sIdx[i] = sIdx[partner]
                                    sIdx[partner] = si
                    cute.arch.barrier()
                    j = j // Int32(2)
                k = k * Int32(2)

            # ---- write the table in sorted order ---------------------------------------------
            for e in cutlass.range_constexpr(ORDER_ELEMENTS):
                i = tidx + Int32(e * ORDER_THREADS)
                if i < n:
                    write_item(
                        source,
                        b_t,
                        n_heads_out,
                        mCuSeqlens,
                        mStaging,
                        mWorkItems,
                        i,
                        sIdx[i],
                        mRowBase,
                        mSlotRows,
                    )


@cute.jit
def order_storage():
    """``(sKey, sIdx, sSpread)`` SMEM of :func:`order_body`."""
    smem = SmemAllocator()
    sKey = smem.allocate_tensor(Int32, cute.make_layout(ORDER_CAPACITY), 16)
    sIdx = smem.allocate_tensor(Int32, cute.make_layout(ORDER_CAPACITY), 16)
    sSpread = smem.allocate_tensor(Int32, cute.make_layout(2), 8)
    return sKey, sIdx, sSpread


@cute.jit
def emit_chunk_rows(b_t: cutlass.Constexpr[int], cu_seqlens, mRows, mCount, lane_idx):
    """One warp: the chunk row table, one ``(chunk, batch, batch_start, batch_end)`` entry per
    chunk in sequence order, and the row count; lanes scan the chunk counts of 32 sequences at
    a time and write their rows."""
    n_batch = Int32(cu_seqlens.shape[0]) - Int32(1)
    carry = Int32(0)
    for b0 in cutlass.range(0, n_batch, 32, unroll=1):
        b = b0 + lane_idx
        n_chunks = Int32(0)
        batch_start = Int32(0)
        batch_end = Int32(0)
        if b < n_batch:
            batch_start = cu_at(cu_seqlens, b)
            batch_end = cu_at(cu_seqlens, b + Int32(1))
            n_chunks = (batch_end - batch_start + Int32(b_t - 1)) // Int32(b_t)
        incl = n_chunks
        for offset in [1, 2, 4, 8, 16]:
            n = cute.arch.shuffle_sync_up(incl, offset, mask_and_clamp=0)
            if lane_idx >= offset:
                incl = incl + n
        base = carry + incl - n_chunks
        for j in cutlass.range(32):
            n_j = cute.arch.shuffle_sync(n_chunks, j)
            base_j = cute.arch.shuffle_sync(base, j)
            start_j = cute.arch.shuffle_sync(batch_start, j)
            end_j = cute.arch.shuffle_sync(batch_end, j)
            for c in cutlass.range(lane_idx, n_j, 32, unroll=1):
                mRows[base_j + c, 0] = c
                mRows[base_j + c, 1] = b0 + j
                mRows[base_j + c, 2] = start_j
                mRows[base_j + c, 3] = end_j
        carry = carry + cute.arch.shuffle_sync(incl, 31)
    if lane_idx == 0:
        mCount[0] = carry


@cute.kernel
def work_table_kernel(
    source: cutlass.Constexpr[str],
    b_t: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    rows_b_t: cutlass.Constexpr[int],
    n_heads_out: Int32,
    tokens: Int32,
    cu_seqlens: cute.Tensor,
    staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler: cute.Tensor | None,
    state_indices: cute.Tensor | None,
    has_initial_state: cute.Tensor | None,
    rows: cute.Tensor | None,
    row_count: cute.Tensor | None,
):
    """Every block checks the offsets against ``tokens``; block 0 then orders the table
    (:func:`order_body`), block 1 (bound ``rows``) builds the chunk row table of
    ``rows_b_t``-token chunks on its first warp."""
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    tidx = Int32(cute.arch.thread_idx()[0])
    check_packed_offsets(cu_seqlens, tokens, tidx, ORDER_THREADS)
    n_batch = Int32(cu_seqlens.shape[0]) - Int32(1)
    if Int32(cute.arch.block_idx()[0]) == Int32(0):
        sKey, sIdx, sSpread = order_storage()
        order_body(
            source,
            b_t,
            tidx,
            n_heads_out,
            n_heads_out * n_batch,
            cu_seqlens,
            staging,
            work_count,
            work_items,
            scheduler,
            sKey,
            sIdx,
            sSpread,
            num_ctas=num_ctas,
            mStateIndices=state_indices,
            mHasInitialState=has_initial_state,
        )
    else:
        if cutlass.const_expr(rows is not None):
            if tidx < Int32(32):
                emit_chunk_rows(rows_b_t, cu_seqlens, rows, row_count, tidx)


@cute.jit
def launch_work_table(
    b_t: cutlass.Constexpr[int],
    n_heads_out: Int32,
    tokens: Int32,
    cu_seqlens: cute.Tensor,
    staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler: cute.Tensor | None,
    stream: cuda.CUstream,
    *,
    num_ctas: cutlass.Constexpr[int] = 0,
    state_indices: cute.Tensor | None = None,
    has_initial_state: cute.Tensor | None = None,
    rows: cute.Tensor | None = None,
    row_count: cute.Tensor | None = None,
    rows_b_t: cutlass.Constexpr[int] = 0,
) -> None:
    """The main kernels' table prologue: check ``cu_seqlens`` against the ``tokens`` packed
    tokens (:func:`check_packed_offsets`), order ``staging`` (the split rows; ``None`` for the
    uncut table from ``cu_seqlens``) into ``work_items`` / ``work_count`` for ``n_heads_out``
    tile heads, zero ``scheduler``, and optionally build a chunk row table. Callable from a
    launch bundle's ``@cute.jit`` host; the next launch waits on it through PDL."""
    work_table_kernel(
        "uncut" if cutlass.const_expr(staging is None) else "staged",
        b_t,
        num_ctas,
        rows_b_t,
        n_heads_out,
        tokens,
        cu_seqlens,
        staging,
        work_count,
        work_items,
        scheduler,
        state_indices,
        has_initial_state,
        rows,
        row_count,
    ).launch(
        grid=(1 if cutlass.const_expr(rows is None) else 2, 1, 1),
        block=(ORDER_THREADS, 1, 1),
        stream=stream,
        use_pdl=True,
    )


# ---- standalone entry points -----------------------------------------------------------------


def _int32_table(rows, cols=None):
    """Compact int32 vector or ``(rows, cols)`` table signature."""
    shape = (rows,) if cols is None else (rows, cols)
    return make_fake_compact_tensor(cutlass.Int32, shape, assumed_align=4)


def _check_int32(name: str, tensor: torch.Tensor, rows: int, cols: int | None = None) -> None:
    """``tensor`` must be a compact int32 vector (``cols`` None) or ``(>= rows, cols)`` table."""
    shape_ok = tensor.ndim == 1 if cols is None else tensor.ndim == 2 and tensor.shape[1] == cols
    if not shape_ok or tensor.dtype != torch.int32 or not tensor.is_contiguous():
        want = "an int32 vector" if cols is None else f"an int32 (rows, {cols}) table"
        raise ValueError(f"{name} must be {want}, got {tuple(tensor.shape)} {tensor.dtype}")
    if tensor.shape[0] < rows:
        raise ValueError(f"{name} needs at least {rows} rows, got {tensor.shape[0]}")


_GATE_DTYPES = {
    torch.float32: cutlass.Float32,
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}


@jit_cache
def compile_split_table(static: SplitStatic, gate_dtype_name: str, use_int64_offsets: bool):
    """Compile :func:`launch_split_table` for one :class:`SplitStatic`; every extent is
    symbolic (int64 with ``use_int64_offsets``)."""
    gate_dtype = getattr(cutlass, gate_dtype_name)
    sym = cute.sym_int64 if use_int64_offsets else cute.sym_int
    if static.gate_channels:
        gate = make_fake_strided_tensor(
            gate_dtype, (sym(), sym(), sym()), assumed_align=8 if gate_dtype.width == 16 else 4
        )
    else:
        gate = make_fake_strided_tensor(
            gate_dtype, (sym(), sym()), contiguous_dim=None, assumed_align=4
        )
    scratch = make_fake_strided_tensor(cutlass.Float32, (sym(), sym()), assumed_align=4)
    return compile_tvm_ffi(
        launch_split_table,
        static,
        *(Int32(0) for _ in range(4)),
        Float32(0.0),
        *(Int32(0) for _ in range(3)),
        gate,
        _int32_table(sym()),
        scratch,
        _int32_table(sym(), WORK_ITEM_FIELDS),
        _int32_table(sym()),
        opt_level=OPT_LEVEL,
        name="wd_split_table_"
        + "_".join(str(v) for v in static)
        + f"_{gate_dtype_name.lower()}_i64{int(use_int64_offsets)}",
    )


def build_split_table(
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_scratch: torch.Tensor,
    staging: torch.Tensor,
    work_count: torch.Tensor,
    *,
    n_tiles: int,
    ideal_chunks: int,
    num_sms: int,
    b_t: int,
    use_int64_offsets: bool = False,
) -> SplitTableFacts:
    """Run the split planner standalone: the staged rows of natural-log ``gate`` into
    ``staging`` (``(>= max_work_items, WORK_ITEM_FIELDS)`` int32) and their count into
    ``work_count``; ``chunk_scratch`` is ``(>= chunk_scratch_rows, HO)`` fp32."""
    if gate.ndim not in (2, 3) or gate.dtype not in _GATE_DTYPES:
        raise ValueError("gate must be a (T, HO) or (T, HO, DK) fp32/bf16/fp16 tensor")
    batch = cu_seqlens.numel() - 1
    _check_int32("cu_seqlens", cu_seqlens, 2)
    if n_tiles != batch * gate.shape[1] or ideal_chunks <= 0:
        raise ValueError("n_tiles must be sequences * heads and ideal_chunks positive")
    rows = chunk_scratch_rows(gate.shape[0], batch, b_t)
    if (
        chunk_scratch.dtype != torch.float32
        or chunk_scratch.shape[0] < rows
        or chunk_scratch.shape[1] != gate.shape[1]
        or chunk_scratch.stride(1) != 1
    ):
        raise ValueError(f"chunk_scratch must be fp32 ({rows}, {gate.shape[1]}) or longer")
    items = max_work_items(gate.shape[0], batch, gate.shape[1], ideal_chunks, b_t, num_sms)
    _check_int32("staging", staging, items, WORK_ITEM_FIELDS)
    _check_int32("work_count", work_count, 1)
    facts = split_table_facts(
        gate, cu_seqlens, n_tiles=n_tiles, ideal_chunks=ideal_chunks, num_sms=num_sms, b_t=b_t
    )
    fn = compile_split_table(facts.static, _GATE_DTYPES[gate.dtype].__name__, use_int64_offsets)
    fn(*facts.scalars(), gate, cu_seqlens, chunk_scratch, staging, work_count)
    return facts


@jit_cache
def compile_work_table(
    staged: bool,
    b_t: int,
    num_ctas: int,
    scheduler: bool,
    paged: bool,
    rows_b_t: int,
    use_int64_offsets: bool,
):
    """Compile :func:`launch_work_table` for one static configuration; extents are symbolic
    (int64 with ``use_int64_offsets``)."""

    @cute.jit
    def entry(
        n_heads_out,
        tokens,
        cu,
        staging,
        count,
        items,
        sched,
        routes,
        seeds,
        rows,
        row_count,
        stream,
    ):
        launch_work_table(
            b_t,
            n_heads_out,
            tokens,
            cu,
            staging,
            count,
            items,
            sched,
            stream,
            num_ctas=num_ctas,
            state_indices=routes,
            has_initial_state=seeds,
            rows=rows,
            row_count=row_count,
            rows_b_t=rows_b_t,
        )

    sym = cute.sym_int64 if use_int64_offsets else cute.sym_int
    seqs = sym()
    return compile_tvm_ffi(
        entry,
        Int32(0),
        Int32(0),
        _int32_table(sym()),
        _int32_table(sym(), WORK_ITEM_FIELDS) if staged else None,
        _int32_table(sym()),
        _int32_table(sym(), WORK_ITEM_FIELDS),
        _int32_table(sym()) if scheduler else None,
        _int32_table(seqs) if paged else None,
        make_fake_compact_tensor(cutlass.Uint8, (seqs,), assumed_align=1) if paged else None,
        make_fake_strided_tensor(cutlass.Int32, (sym(), 4), contiguous_dim=None)
        if rows_b_t
        else None,
        _int32_table(sym()) if rows_b_t else None,
        opt_level=OPT_LEVEL,
        name=f"wd_work_table_{int(staged)}_{b_t}_{num_ctas}_{int(scheduler)}_{int(paged)}"
        f"_{rows_b_t}_i64{int(use_int64_offsets)}",
    )


def build_work_table(
    cu_seqlens: torch.Tensor,
    work_items: torch.Tensor,
    work_count: torch.Tensor,
    *,
    n_heads_out: int,
    b_t: int,
    staging: torch.Tensor | None = None,
    scheduler: torch.Tensor | None = None,
    num_ctas: int = 0,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    rows: torch.Tensor | None = None,
    row_count: torch.Tensor | None = None,
    rows_b_t: int = 0,
    tokens: int = INT32_MAX,
    use_int64_offsets: bool = False,
) -> None:
    """Run :func:`launch_work_table` standalone. With ``staging`` the split rows (``work_count``
    holds their count) are ordered; without it the uncut table is built from ``cu_seqlens``.
    ``tokens`` bounds ``cu_seqlens[-1]``; the default bounds nothing."""
    if (state_indices is None) != (has_initial_state is None):
        raise ValueError("paged routing needs both state_indices and has_initial_state")
    if (rows is None) != (rows_b_t == 0) or (rows is None) != (row_count is None):
        raise ValueError("a row table needs rows, row_count and rows_b_t together")
    _check_int32("cu_seqlens", cu_seqlens, 2)
    _check_int32("work_count", work_count, 1)
    if staging is None:
        _check_int32(
            "work_items", work_items, (cu_seqlens.numel() - 1) * n_heads_out, WORK_ITEM_FIELDS
        )
    else:
        _check_int32("staging", staging, 0, WORK_ITEM_FIELDS)
        _check_int32("work_items", work_items, staging.shape[0], WORK_ITEM_FIELDS)
    fn = compile_work_table(
        staging is not None,
        b_t,
        num_ctas,
        scheduler is not None,
        state_indices is not None,
        rows_b_t,
        use_int64_offsets,
    )
    fn(
        n_heads_out,
        tokens,
        cu_seqlens,
        staging,
        work_count,
        work_items,
        scheduler,
        state_indices,
        has_initial_state,
        rows,
        row_count,
    )


for _kernel in (
    split_plan_kernel,
    split_scan_channel_kernel,
    split_scan_scalar_kernel,
    split_walk_kernel,
    work_table_kernel,
):
    _kernel.set_name_prefix("wd", remove_cutlass_symbol=False)
