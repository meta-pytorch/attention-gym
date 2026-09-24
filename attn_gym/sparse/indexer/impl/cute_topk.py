# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indices-only radix selection over bounded FP32 indexer score slabs.

Adapted from NVIDIA cuDNN Frontend's ``compress_top_k_sm100.py`` at
1ab7d63be81f1ef2eb3b2dea5367219cf9674a5f. Each CTA selects one query row:
(1) histogram the highest 11 ordered-float bits; (2) emit strict winners and
stage the boundary bin; (3) refine its next 11 bits; (4) refine the final 10
bits and emit the final winners. The fixed shared shrink buffer avoids
full-row refinement scans unless its boundary bin overflows.

By default winners take output slots from a shared atomic counter, so index
order and the choice among exact cutoff ties depend on thread timing. The
``deterministic`` specialization instead returns ascending indices with ties
resolved to the lowest indices: winners are marked in an SMEM bitmap with
commutative ``atomic_or`` and compacted in index order. Its rows over 65536
candidates emit through the counter, then sort through bitmap windows and
overwrite their score row as staging.

Only valid causal scores are read. Packed calls restrict selection to each
query's candidate interval and return indices relative to its start. Rows with
at most K candidates require no score reads at all. The monotonic key supports
signed finite FP32 scores.
"""

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from attn_gym._backends.cute.compat import SmemAllocator

_NUM_BINS_11 = 2048
_NUM_BINS_10 = 1024
_SHRINK_MAX = 2048  # 16 KiB for ordered keys and local KV indices.
_BITMAP_BITS = _NUM_BINS_11 * 32  # One 8 KiB bitmap window of candidate indices.


@dsl_user_op
def _float_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    """Reinterpret an FP32 register as unsigned bits without numerical conversion."""
    return Uint32(llvm.bitcast(Uint32.mlir_type, value.ir_value(), loc=loc, ip=ip))


@cute.jit
def _ordered_key(value: Float32) -> Uint32:
    """Map signed finite FP32 values to monotonically increasing unsigned keys."""
    bits = _float_bits(value)
    return bits ^ Uint32(0xFFFFFFFF) if bits & Uint32(0x80000000) else bits | Uint32(0x80000000)


@cute.jit
def _warp_scan_inclusive(value: Int32, lane_id: Int32, lanes: cutlass.Constexpr) -> Int32:
    """Inclusive sum across the first power-of-two number of participating lanes."""
    mask = cutlass.const_expr(((1 << lanes) - 1) & 0xFFFFFFFF)
    for i in cutlass.range_constexpr(lanes.bit_length() - 1):
        offset = 1 << i
        other = cute.arch.shuffle_sync_up(value, offset, mask=mask, mask_and_clamp=0)
        if lane_id >= offset:
            value += other
    return value


@cute.jit
def _block_scan_inclusive(
    value: Int32,
    warp_sums: cute.Tensor,
    tidx: Int32,
    threads: cutlass.Constexpr,
) -> Int32:
    """Inclusive CTA sum using one shared partial per warp and two barriers."""
    warp_id = tidx // 32
    lane_id = tidx % 32
    value = _warp_scan_inclusive(value, lane_id, 32)
    if lane_id == 31:
        warp_sums[warp_id] = value
    cute.arch.barrier()
    if warp_id == 0 and lane_id < threads // 32:
        partial = _warp_scan_inclusive(warp_sums[lane_id], lane_id, threads // 32)
        warp_sums[lane_id] = partial
    cute.arch.barrier()
    if warp_id > 0:
        value += warp_sums[warp_id - 1]
    return value


class IndexerTopKKernel:
    """Select static K indices per row from a contiguous symbolic [P, 2, S] slab.

    ``pair_start`` locates the slab in flattened (batch, query-pair) order; odd
    sequence tails do not share pairs across batches. The caller handles K=0
    without launching and selects the wide ABI when score/output offsets or
    strides exceed int32. Sequence lengths and returned local IDs fit int32.
    The score slab is scratch: deterministic rows over 65536 candidates
    overwrite it.
    """

    block_threads = 512

    def __init__(
        self,
        topk: int,
        causal: bool,
        use_int64_offsets: bool = False,
        *,
        compress_ratio: int = 1,
        deterministic: bool = False,
    ):
        self.topk = topk
        self.deterministic = deterministic
        self.causal = causal
        self.compress_ratio = compress_ratio
        self.use_int64_offsets = use_int64_offsets

    def get_name(self) -> str:
        """Return a stable artifact name including all static specialization fields."""
        return (
            f"indexer_topk_k{self.topk}_c{int(self.causal)}_r{self.compress_ratio}_"
            f"i64{int(self.use_int64_offsets)}_d{int(self.deterministic)}_th{self.block_threads}"
        )

    @cute.jit
    def upcast_offset(self, value):
        """Widen indices before any address arithmetic in the wide specialization."""
        return Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    @cute.jit
    def visible_candidates(self, query):
        """Causal candidate count for query; the static ratio keeps r=1 division-free."""
        if cutlass.const_expr(self.compress_ratio == 1):
            return query + 1
        return (query + 1) // self.compress_ratio

    @cute.jit
    def select_lowest_ties(
        self,
        row: cute.Tensor,
        row_output: cute.Tensor,
        tie_key: Uint32,
        need: Int32,
        seg_len: Int32,
        scratch: cute.Tensor,
        warp_sums: cute.Tensor,
        use_bitmap,
        selected: cute.Tensor,
        tidx: Int32,
    ):
        """Select the ``need`` lowest-index exact ties by rescanning the row.

        Used when the boundary bin overflowed the shrink buffer or the row spans
        several bitmap windows. Each bitmap window marks
        its ties in the zeroed ``scratch`` and keeps them in index order, into
        ``selected`` or after the ``topk - need`` strict winners in ``row_output``.
        """
        seen = Int32(0)
        for window in range(0, seg_len, _BITMAP_BITS):
            span = cutlass.min(seg_len - window, _BITMAP_BITS)
            for i in range(tidx, span, self.block_threads):
                if _ordered_key(row[window + i]) == tie_key:
                    self.mark(scratch, i)
            cute.arch.barrier()
            if use_bitmap:
                self.compact_bitmap(
                    scratch,
                    selected,
                    warp_sums,
                    span,
                    window,
                    seen,
                    tidx,
                    limit=need,
                    to_bitmap=True,
                )
            else:
                first_slot = Int32(self.topk) - need
                self.compact_bitmap(
                    scratch,
                    row_output,
                    warp_sums,
                    span,
                    window,
                    first_slot + seen,
                    tidx,
                    limit=Int32(self.topk),
                )
            seen += warp_sums[self.block_threads // 32 - 1]
            cute.arch.barrier()
            for i in range(tidx, _NUM_BINS_11, self.block_threads):
                scratch[i] = Int32(0)
            cute.arch.barrier()

    @cute.jit
    def mark(self, bitmap: cute.Tensor, idx: Int32):
        """Set one index's bit; OR commutes, so marking order cannot matter."""
        cute.arch.atomic_or(
            bitmap.iterator + (idx >> 5), Int32(1) << (idx & 31), sem="relaxed", scope="cta"
        )

    @cute.jit
    def emit(self, idx: Int32, use_bitmap, selected, counters, row_output):
        """Record one selected index as a bitmap bit or in the next counter slot."""
        if cutlass.const_expr(self.deterministic):
            if use_bitmap:
                self.mark(selected, idx)
            else:
                self.append(idx, counters, row_output)
        else:
            self.append(idx, counters, row_output)

    @cute.jit
    def append(self, idx: Int32, counters: cute.Tensor, row_output: cute.Tensor):
        """Write one index to the next output slot claimed from the shared counter."""
        dst = cute.arch.atomic_add(counters.iterator, Int32(1), sem="relaxed", scope="cta")
        row_output[dst] = idx

    @cute.jit
    def compact_bitmap(
        self,
        bitmap: cute.Tensor,
        row_output: cute.Tensor,
        warp_sums: cute.Tensor,
        span,
        window,
        written: Int32,
        tidx: Int32,
        limit=None,
        to_bitmap: cutlass.Constexpr = False,
    ):
        """Write the set bits among a window's first ``span`` to ``row_output[written:]``.

        Each thread owns a contiguous power-of-two bit range sized so the CTA
        covers ``span`` bits, which keeps short rows from serializing on a few
        threads. Output is ascending. The CTA count is left in the last
        ``warp_sums`` entry; callers that reuse the bitmap must clear it.

        With ``limit``, only bits whose output rank is below it are kept. With
        ``to_bitmap``, kept bits are marked in the ``row_output`` bitmap instead.
        """
        per_thread = cute.ceil_div(span, self.block_threads)
        log_width = Int32(0)
        if per_thread > 1:
            log_width = Int32(32) - Int32(cute.arch.clz(Int32(per_thread - 1)))
        width = Int32(1) << log_width
        first_bit = tidx << log_width
        words = []
        count = Int32(0)
        for j in cutlass.range_constexpr(_BITMAP_BITS // self.block_threads // 32):
            word = Uint32(0)
            if j * 32 < width:
                word = Uint32(bitmap[(first_bit >> 5) + j])
                if width < 32:
                    mask = (Uint32(1) << Uint32(width & 31)) - Uint32(1)
                    word = (word >> Uint32(first_bit & 31)) & mask
            words.append(word)
            count += Int32(cute.arch.popc(word))
        inclusive = _block_scan_inclusive(count, warp_sums, tidx, self.block_threads)
        dst = written + inclusive - count
        cap = Int32(0x7FFFFFFF) if cutlass.const_expr(limit is None) else limit
        for j in cutlass.range_constexpr(_BITMAP_BITS // self.block_threads // 32):
            word = words[j]
            if dst >= cap:
                word = Uint32(0)
            word_base = window + first_bit + j * 32
            while word != Uint32(0):
                low = word & (~word + Uint32(1))
                idx = word_base + Int32(cute.arch.bfind(low))
                if cutlass.const_expr(to_bitmap):
                    self.mark(row_output, idx)
                else:
                    row_output[dst] = idx
                dst += 1
                word ^= low
                if dst >= cap:
                    word = Uint32(0)

    @cute.jit
    def sort_selected(
        self,
        row: cute.Tensor,
        row_output: cute.Tensor,
        bitmap: cute.Tensor,
        warp_sums: cute.Tensor,
        seg_len: Int32,
        tidx: Int32,
    ):
        """Sort a multi-window row's selected indices through an SMEM bitmap.

        The unsorted indices are staged in this CTA's dead score row. Each
        window marks its indices with commutative ``atomic_or`` and compacts
        them in ascending order. ``bitmap`` must start zeroed.
        """
        staged = cute.recast_tensor(row, Int32)
        for slot in range(tidx, self.topk, self.block_threads):
            staged[slot] = row_output[slot]
        cute.arch.barrier()
        written = Int32(0)
        for window in range(0, seg_len, _BITMAP_BITS):
            for slot in range(tidx, self.topk, self.block_threads):
                local = staged[slot] - window
                if local >= 0 and local < _BITMAP_BITS:
                    self.mark(bitmap, local)
            cute.arch.barrier()
            self.compact_bitmap(bitmap, row_output, warp_sums, _BITMAP_BITS, window, written, tidx)
            written += warp_sums[self.block_threads // 32 - 1]
            for i in range(tidx, _NUM_BINS_11, self.block_threads):
                bitmap[i] = Int32(0)
            cute.arch.barrier()  # Protect warp_sums and the bitmap before the next window.

    @cute.jit
    def find_threshold(
        self,
        histogram: cute.Tensor,
        bins: cutlass.Constexpr,
        need: Int32,
        warp_sums: cute.Tensor,
        threshold: cute.Tensor,
        tidx: Int32,
        publish_count: cutlass.Constexpr = False,
    ):
        """Find the descending cumulative-count boundary and clear the histogram.

        Exactly one bin crosses positive ``need``. Its owner publishes the bin
        ID and the number still needed from that bin for the next radix pass;
        the third result is that bin's size with ``publish_count``, else zero.
        """
        items = cutlass.const_expr(bins // self.block_threads)
        counts = [Int32(0) for _ in range(items)]
        local_sum = Int32(0)
        for i in cutlass.range_constexpr(items):
            bin_id = bins - 1 - (tidx * items + i)
            counts[i] = histogram[bin_id]
            histogram[bin_id] = Int32(0)
            local_sum += counts[i]

        inclusive = _block_scan_inclusive(local_sum, warp_sums, tidx, self.block_threads)
        running = inclusive - local_sum
        for i in cutlass.range_constexpr(items):
            count = counts[i]
            if count > 0 and running < need and running + count >= need:
                threshold[0] = Int32(bins - 1 - (tidx * items + i))
                threshold[1] = need - running
                if cutlass.const_expr(publish_count):
                    threshold[2] = count
            running += count
        cute.arch.barrier()
        count = Int32(0)
        if cutlass.const_expr(publish_count):
            count = threshold[2]
        return threshold[0], threshold[1], count

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        output: cute.Tensor,
        pair_start: Int32 | Int64,
        candidate_bounds: cute.Tensor | None,
    ):
        """Run the four-pass selection with CTA-uniform row and short-row guards."""
        tidx = cute.arch.thread_idx()[0]
        block_id = self.upcast_offset(cute.arch.block_idx()[0])
        slab_pair = block_id // 2
        qi = block_id % 2
        num_queries = self.upcast_offset(output.shape[1])
        num_candidates = self.upcast_offset(scores.shape[2])
        pairs_per_batch = cute.ceil_div(num_queries, 2)
        global_pair = self.upcast_offset(pair_start) + slab_pair
        batch = global_pair // pairs_per_batch
        q = (global_pair % pairs_per_batch) * 2 + qi

        smem = SmemAllocator()
        histogram = smem.allocate_tensor(
            Int32, cute.make_layout((_NUM_BINS_11,)), byte_alignment=128
        )
        warp_sums = smem.allocate_tensor(
            Int32, cute.make_layout((self.block_threads // 32,)), byte_alignment=128
        )
        threshold = smem.allocate_tensor(Int32, cute.make_layout((3,)), byte_alignment=128)
        # Counters: emitted winners, accepted exact ties, boundary-bin size.
        counters = smem.allocate_tensor(Int32, cute.make_layout((3,)), byte_alignment=128)
        shrink_keys = smem.allocate_tensor(
            Uint32, cute.make_layout((_SHRINK_MAX,)), byte_alignment=128
        )
        shrink_indices = smem.allocate_tensor(
            Int32, cute.make_layout((_SHRINK_MAX,)), byte_alignment=128
        )
        selected = None
        if cutlass.const_expr(self.deterministic):
            selected = smem.allocate_tensor(
                Int32, cute.make_layout((_NUM_BINS_11,)), byte_alignment=128
            )

        if batch < output.shape[0] and q < num_queries:
            row_output = output[batch, q, None]
            candidate_start = self.upcast_offset(Int32(0))
            seg_len = Int32(num_candidates)
            if cutlass.const_expr(candidate_bounds is not None):
                candidate_start = self.upcast_offset(candidate_bounds[q, 0])
                seg_len = candidate_bounds[q, 1] - candidate_bounds[q, 0]
            elif cutlass.const_expr(self.causal):
                seg_len = Int32(self.visible_candidates(q))

            if seg_len <= self.topk:
                for slot in range(tidx, self.topk, self.block_threads):
                    row_output[slot] = Int32(slot) if slot < seg_len else Int32(-1)
            else:
                row = scores[slab_pair, qi, None]
                if cutlass.const_expr(candidate_bounds is not None):
                    row = cute.domain_offset((candidate_start,), row)
                if tidx == 0:
                    counters[0] = Int32(0)
                    counters[1] = Int32(0)
                    counters[2] = Int32(0)
                use_bitmap = False
                if cutlass.const_expr(self.deterministic):
                    use_bitmap = seg_len <= _BITMAP_BITS
                for i in range(tidx, _NUM_BINS_11, self.block_threads):
                    histogram[i] = Int32(0)
                    if cutlass.const_expr(self.deterministic):
                        selected[i] = Int32(0)
                cute.arch.barrier()

                # Pass 1: highest 11-bit histogram over exactly the valid row.
                for i in range(tidx, seg_len, self.block_threads):
                    key = _ordered_key(row[i])
                    bin0 = Int32((key >> 21) & Uint32(0x7FF))
                    cute.arch.atomic_add(
                        histogram.iterator + bin0, Int32(1), sem="relaxed", scope="cta"
                    )
                cute.arch.barrier()
                bin0_threshold, need0, _ = self.find_threshold(
                    histogram, _NUM_BINS_11, Int32(self.topk), warp_sums, threshold, tidx
                )

                # Pass 2: emit strict winners and stage the first boundary bin.
                for i in range(tidx, seg_len, self.block_threads):
                    key = _ordered_key(row[i])
                    bin0 = Int32((key >> 21) & Uint32(0x7FF))
                    if bin0 > bin0_threshold:
                        self.emit(Int32(i), use_bitmap, selected, counters, row_output)
                    elif bin0 == bin0_threshold:
                        bin1 = Int32((key >> 10) & Uint32(0x7FF))
                        cute.arch.atomic_add(
                            histogram.iterator + bin1, Int32(1), sem="relaxed", scope="cta"
                        )
                        slot = cute.arch.atomic_add(
                            counters.iterator + 2, Int32(1), sem="relaxed", scope="cta"
                        )
                        if slot < _SHRINK_MAX:
                            shrink_keys[slot] = key
                            shrink_indices[slot] = Int32(i)
                cute.arch.barrier()
                bin1_threshold, need1, _ = self.find_threshold(
                    histogram, _NUM_BINS_11, need0, warp_sums, threshold, tidx
                )

                # Derive overflow from the atomic count after the barrier: no
                # concurrently written non-atomic overflow flag is necessary.
                shrink_count = counters[2]
                use_shrink = shrink_count <= _SHRINK_MAX
                scan_len = shrink_count if use_shrink else seg_len

                # Pass 3: refine the middle 11 bits, falling back to a full scan.
                for i in range(tidx, scan_len, self.block_threads):
                    key = Uint32(0)
                    idx = Int32(0)
                    if use_shrink:
                        key = shrink_keys[i]
                        idx = shrink_indices[i]
                    else:
                        key = _ordered_key(row[i])
                        idx = Int32(i)
                    bin0 = Int32((key >> 21) & Uint32(0x7FF))
                    if bin0 == bin0_threshold:
                        bin1 = Int32((key >> 10) & Uint32(0x7FF))
                        if bin1 > bin1_threshold:
                            self.emit(idx, use_bitmap, selected, counters, row_output)
                        elif bin1 == bin1_threshold:
                            bin2 = Int32(key & Uint32(0x3FF))
                            cute.arch.atomic_add(
                                histogram.iterator + bin2, Int32(1), sem="relaxed", scope="cta"
                            )
                cute.arch.barrier()
                bin2_threshold, need2, tie_count = self.find_threshold(
                    histogram,
                    _NUM_BINS_10,
                    need1,
                    warp_sums,
                    threshold,
                    tidx,
                    publish_count=self.deterministic,
                )

                # Pass 4: emit the final strict winners and exact ties. Deterministic
                # mode emits ties here only when all are needed and otherwise chooses
                # them by index afterwards.
                take_all_ties = tie_count == need2
                tie_budget = need2
                if cutlass.const_expr(self.deterministic):
                    tie_budget = need2 if take_all_ties else Int32(0)
                for i in range(tidx, scan_len, self.block_threads):
                    key = Uint32(0)
                    idx = Int32(0)
                    if use_shrink:
                        key = shrink_keys[i]
                        idx = shrink_indices[i]
                    else:
                        key = _ordered_key(row[i])
                        idx = Int32(i)
                    bin0 = Int32((key >> 21) & Uint32(0x7FF))
                    bin1 = Int32((key >> 10) & Uint32(0x7FF))
                    if bin0 == bin0_threshold and bin1 == bin1_threshold:
                        bin2 = Int32(key & Uint32(0x3FF))
                        if bin2 > bin2_threshold:
                            self.emit(idx, use_bitmap, selected, counters, row_output)
                        elif bin2 == bin2_threshold:
                            slot = cute.arch.atomic_add(
                                counters.iterator + 1, Int32(1), sem="relaxed", scope="cta"
                            )
                            if slot < tie_budget:
                                self.emit(idx, use_bitmap, selected, counters, row_output)

                if cutlass.const_expr(self.deterministic):
                    cute.arch.barrier()
                    self.order_selection(
                        row,
                        row_output,
                        histogram,
                        warp_sums,
                        selected,
                        shrink_keys,
                        shrink_indices,
                        seg_len,
                        scan_len,
                        use_bitmap,
                        use_shrink,
                        take_all_ties,
                        need2,
                        (Uint32(bin0_threshold) << 21)
                        | (Uint32(bin1_threshold) << 10)
                        | Uint32(bin2_threshold),
                        tidx,
                    )

    @cute.jit
    def order_selection(
        self,
        row: cute.Tensor,
        row_output: cute.Tensor,
        histogram: cute.Tensor,
        warp_sums: cute.Tensor,
        selected: cute.Tensor,
        shrink_keys: cute.Tensor,
        shrink_indices: cute.Tensor,
        seg_len: Int32,
        scan_len: Int32,
        use_bitmap,
        use_shrink,
        take_all_ties,
        need2: Int32,
        tie_key: Uint32,
        tidx: Int32,
    ):
        """Resolve ambiguous cutoff ties by index, then write the row ascending.

        Every strict winner is already recorded: as ``selected`` bits for rows
        within one bitmap window, otherwise in the first ``topk - need2`` slots
        of ``row_output``. ``histogram`` serves as zeroed scratch: pass 3's
        ``find_threshold`` cleared bins below 1024 and pass 2's cleared the rest,
        which pass 3 never increments.
        """
        if not take_all_ties:
            if use_bitmap and use_shrink:
                # All ties are staged: mark them in the zeroed histogram and
                # keep the need2 lowest indices without rereading the row.
                for i in range(tidx, scan_len, self.block_threads):
                    if shrink_keys[i] == tie_key:
                        self.mark(histogram, shrink_indices[i])
                cute.arch.barrier()
                self.compact_bitmap(
                    histogram,
                    selected,
                    warp_sums,
                    seg_len,
                    Int32(0),
                    Int32(0),
                    tidx,
                    limit=need2,
                    to_bitmap=True,
                )
            else:
                self.select_lowest_ties(
                    row,
                    row_output,
                    tie_key,
                    need2,
                    seg_len,
                    histogram,
                    warp_sums,
                    use_bitmap,
                    selected,
                    tidx,
                )
            cute.arch.barrier()

        if use_bitmap:
            self.compact_bitmap(selected, row_output, warp_sums, seg_len, Int32(0), Int32(0), tidx)
        else:
            self.sort_selected(row, row_output, histogram, warp_sums, seg_len, tidx)

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        output: cute.Tensor,
        pair_start: Int32 | Int64,
        stream: cuda.CUstream,
    ):
        """Launch one 512-thread CTA per physical query row in the score slab."""
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(scores, output, pair_start, None).launch(
            grid=(scores.shape[0] * 2, 1, 1),
            block=(self.block_threads, 1, 1),
            stream=stream,
        )

    @cute.jit
    def with_candidate_bounds(
        self,
        scores: cute.Tensor,
        output: cute.Tensor,
        pair_start: Int32 | Int64,
        candidate_bounds: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Packed TVM-FFI entrypoint, retaining the existing dense-call ABI."""
        self.kernel.set_name_prefix(f"{self.get_name()}_packed")
        self.kernel(scores, output, pair_start, candidate_bounds).launch(
            grid=(scores.shape[0] * 2, 1, 1),
            block=(self.block_threads, 1, 1),
            stream=stream,
        )
