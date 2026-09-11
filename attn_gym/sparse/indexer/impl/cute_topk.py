# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indices-only radix selection over bounded FP32 indexer score slabs.

Adapted from NVIDIA cuDNN Frontend's ``compress_top_k_sm100.py`` at
1ab7d63be81f1ef2eb3b2dea5367219cf9674a5f. Each CTA selects one query row:
(1) histogram the highest 11 ordered-float bits; (2) emit strict winners and
stage the boundary bin; (3) refine its next 11 bits; (4) refine the final 10
bits and emit only the required number of exact ties. The fixed shared shrink
buffer avoids full-row refinement scans unless its boundary bin overflows.

Only valid causal scores are read. Rows with at most K candidates require no
score reads at all. Output ordering and selection among ties are unspecified;
the monotonic key supports signed finite FP32 scores.
"""

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

_NUM_BINS_11 = 2048
_NUM_BINS_10 = 1024
_SHRINK_MAX = 2048  # 16 KiB for ordered keys and local KV indices.


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
    """Select static K indices per row from a contiguous symbolic [P, 2, T] slab.

    ``pair_start`` locates the slab in flattened (batch, query-pair) order; odd
    sequence tails do not share pairs across batches. The caller handles K=0
    without launching and selects the wide ABI when score/output offsets or
    strides exceed int32. Sequence lengths and returned local IDs fit int32.
    """

    block_threads = 512

    def __init__(self, topk: int, causal: bool, use_int64_offsets: bool = False):
        self.topk = topk
        self.causal = causal
        self.use_int64_offsets = use_int64_offsets

    def get_name(self) -> str:
        """Return a stable artifact name including all static specialization fields."""
        return (
            f"indexer_topk_k{self.topk}_c{int(self.causal)}_"
            f"i64{int(self.use_int64_offsets)}_th{self.block_threads}"
        )

    @cute.jit
    def upcast_offset(self, value):
        """Widen indices before any address arithmetic in the wide specialization."""
        return Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    @cute.jit
    def find_threshold(
        self,
        histogram: cute.Tensor,
        bins: cutlass.Constexpr,
        need: Int32,
        warp_sums: cute.Tensor,
        threshold: cute.Tensor,
        tidx: Int32,
    ):
        """Find the descending cumulative-count boundary and clear the histogram.

        Exactly one bin crosses positive ``need``. Its owner publishes the bin
        ID and the number still needed from that bin for the next radix pass.
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
            running += count
        cute.arch.barrier()
        return threshold[0], threshold[1]

    @cute.kernel
    def kernel(
        self,
        scores: cute.Tensor,
        output: cute.Tensor,
        pair_start: Int32 | Int64,
    ):
        """Run the four-pass selection with CTA-uniform row and short-row guards."""
        tidx = cute.arch.thread_idx()[0]
        block_id = self.upcast_offset(cute.arch.block_idx()[0])
        slab_pair = block_id // 2
        qi = block_id % 2
        seq_len = self.upcast_offset(scores.shape[2])
        pairs_per_batch = cute.ceil_div(seq_len, 2)
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
        threshold = smem.allocate_tensor(Int32, cute.make_layout((2,)), byte_alignment=128)
        # Counters: emitted winners, accepted exact ties, boundary-bin size.
        counters = smem.allocate_tensor(Int32, cute.make_layout((3,)), byte_alignment=128)
        shrink_keys = smem.allocate_tensor(
            Uint32, cute.make_layout((_SHRINK_MAX,)), byte_alignment=128
        )
        shrink_indices = smem.allocate_tensor(
            Int32, cute.make_layout((_SHRINK_MAX,)), byte_alignment=128
        )

        if batch < output.shape[0] and q < seq_len:
            row_output = output[batch, q, None]
            seg_len = Int32(seq_len)
            if cutlass.const_expr(self.causal):
                seg_len = Int32(q + 1)

            if seg_len <= self.topk:
                for slot in range(tidx, self.topk, self.block_threads):
                    row_output[slot] = Int32(slot) if slot < seg_len else Int32(-1)
            else:
                row = scores[slab_pair, qi, None]
                if tidx == 0:
                    counters[0] = Int32(0)
                    counters[1] = Int32(0)
                    counters[2] = Int32(0)
                for i in range(tidx, _NUM_BINS_11, self.block_threads):
                    histogram[i] = Int32(0)
                cute.arch.barrier()

                # Pass 1: highest 11-bit histogram over exactly the valid row.
                for i in range(tidx, seg_len, self.block_threads):
                    key = _ordered_key(row[i])
                    bin0 = Int32((key >> 21) & Uint32(0x7FF))
                    cute.arch.atomic_add(
                        histogram.iterator + bin0, Int32(1), sem="relaxed", scope="cta"
                    )
                cute.arch.barrier()
                bin0_threshold, need0 = self.find_threshold(
                    histogram, _NUM_BINS_11, Int32(self.topk), warp_sums, threshold, tidx
                )

                # Pass 2: emit strict winners and stage the first boundary bin.
                for i in range(tidx, seg_len, self.block_threads):
                    key = _ordered_key(row[i])
                    bin0 = Int32((key >> 21) & Uint32(0x7FF))
                    if bin0 > bin0_threshold:
                        dst = cute.arch.atomic_add(
                            counters.iterator, Int32(1), sem="relaxed", scope="cta"
                        )
                        row_output[dst] = Int32(i)
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
                bin1_threshold, need1 = self.find_threshold(
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
                            dst = cute.arch.atomic_add(
                                counters.iterator, Int32(1), sem="relaxed", scope="cta"
                            )
                            row_output[dst] = idx
                        elif bin1 == bin1_threshold:
                            bin2 = Int32(key & Uint32(0x3FF))
                            cute.arch.atomic_add(
                                histogram.iterator + bin2, Int32(1), sem="relaxed", scope="cta"
                            )
                cute.arch.barrier()
                bin2_threshold, need2 = self.find_threshold(
                    histogram, _NUM_BINS_10, need1, warp_sums, threshold, tidx
                )

                # Pass 4: emit the final strict winners and exactly need2 ties.
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
                            dst = cute.arch.atomic_add(
                                counters.iterator, Int32(1), sem="relaxed", scope="cta"
                            )
                            row_output[dst] = idx
                        elif bin2 == bin2_threshold:
                            slot = cute.arch.atomic_add(
                                counters.iterator + 1, Int32(1), sem="relaxed", scope="cta"
                            )
                            if slot < need2:
                                dst = cute.arch.atomic_add(
                                    counters.iterator, Int32(1), sem="relaxed", scope="cta"
                                )
                                row_output[dst] = idx

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
        self.kernel(scores, output, pair_start).launch(
            grid=(scores.shape[0] * 2, 1, 1),
            block=(self.block_threads, 1, 1),
            stream=stream,
        )
