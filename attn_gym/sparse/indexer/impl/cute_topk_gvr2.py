# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact self-sampling Guess–Verify–Refine selection over indexer score slabs.

The algorithm follows the streaming ``main`` family of FlashInfer's
``topk_varlen/kernels/gvr2_topk_decode.py`` (derived from TensorRT-LLM PR #17821,
ed94d4cfbf): a stratified float4 sample is binned into 256 data-scaled bins and
scanned once for three quantile rungs; the full row is then streamed 32 values
per thread, survivors above the guess are reserved with one warp ballot per
batch and classified into 256 affine bins while they are staged; one 256-bin
scan turns the histogram into output cursors; entire winning bins are emitted
and only the crossing bin is refined by adaptive key-space narrowing.

Exactness: the affine bin index is a weakly monotone function of the score, so
a bin strictly above the count crossing contains only top-K values and the
crossing bin is refined exactly on ordered keys with tie capping. Every
survivor of the verified threshold is histogrammed exactly once, so the count
crossing is exact even when the staging buffer overflows.
"""

import cutlass
import cutlass.cute.math as cute_math
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, Uint32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.smem_allocator import SmemAllocator

from .cute_topk import (
    _NUM_BINS_11,
    _SHRINK_MAX,
    IndexerTopKKernel,
    _float_bits,
    _ordered_key,
    _warp_scan_inclusive,
)

_BINS = 256  # data-scaled histogram bins; the warp-0 scan owns 8 per lane
_STAGE_MAX = 4096  # staged survivors (bits + index); overflow is counted but not staged
NATIVE_TOPK_LIMIT = _STAGE_MAX // 2  # larger K always takes the radix fallback
_CROSS_MAX = 1024  # crossing-bin candidates refined in shared memory
_RANK_MAX = 288  # crossing bins up to this size are ranked directly instead of narrowed
_INF = float("inf")
# Crossing-scan result slots: bin, its count, count strictly above it, total, extra rungs.
_RES_B, _RES_M, _RES_ABOVE, _RES_TOT, _RES_B2, _RES_B3 = range(6)


@dsl_user_op
def _load_float4(address: Int64, *, loc=None, ip=None):
    """Pinned ``ld.global.nc.v4.f32`` of one 16-byte aligned read-only float4.

    The asm boundary keeps four scalar registers; NVVM otherwise pairs adjacent
    128-bit copies into 64-bit register pairs (FlashInfer ``_ld_g_nc_v4_f32``).
    """
    struct = ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>")
    result = llvm.inline_asm(
        struct,
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def _bits_to_float(bits: Uint32, *, loc=None, ip=None) -> Float32:
    """Reinterpret unsigned bits as an FP32 register without numerical conversion."""
    return Float32(llvm.bitcast(Float32.mlir_type, bits.ir_value(), loc=loc, ip=ip))


@dsl_user_op
def _prefetch_l2(address: Int64, *, loc=None, ip=None):
    """``prefetch.global.L2``: register-free hint that overlaps row fetch with sampling."""
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _key_to_float(key: Uint32) -> Float32:
    """Invert ``_ordered_key``."""
    bits = key ^ Uint32(0x80000000) if key & Uint32(0x80000000) else key ^ Uint32(0xFFFFFFFF)
    return _bits_to_float(bits)


@cute.jit
def _signed_key(value: Float32) -> Int32:
    """Ordered key with its top bit flipped: signed Int32 order equals unsigned key order.

    Shared-memory atomics and Int32 tensors order operands as signed, so the
    crossing refinement keeps keys in this form and widens interval arithmetic
    to signed Int64 before intersecting endpoints.
    """
    return Int32(_ordered_key(value) ^ Uint32(0x80000000))


@cute.jit
def _bin_scale(span: Float32) -> Float32:
    """Reciprocal bin width for a window of ``span``, clamped to [1e-30, 3e38].

    The clamp keeps the scale positive and finite (about [8.5e-37, 2.56e32]), so a
    survivor's ``(value - low) * scale`` is finite or +inf but never NaN.
    """
    span = cute.arch.fmin(cute.arch.fmax(span, Float32(1e-30)), Float32(3.0e38))
    return cute.arch.rcp_approx(span * Float32(1.0 / _BINS))


@cute.jit
def _bin_index(value: Float32, low: Float32, scale: Float32) -> Int32:
    """Affine bin of ``value >= low``, weakly monotone in ``value``.

    ``value - low`` is non-negative (possibly +inf) and ``scale`` is finite and
    positive, so the product is never NaN; clamping before the truncating
    conversion keeps the float-to-integer step inside its defined range.
    """
    return Int32(cute.arch.fmin((value - low) * scale, Float32(_BINS - 1)))


@cute.jit
def _lowest_set_bit(mask: Int32) -> Int32:
    """Bit index of the lowest set bit; the caller guarantees ``mask != 0``."""
    return Int32(cute.arch.popc((mask & (Int32(0) - mask)) - Int32(1)))


@cute.jit
def _scan_cross(
    hist: cute.Tensor,
    res: cute.Tensor,
    target: Int32,
    target2: Int32,
    target3: Int32,
    tidx: Int32,
    zero: cutlass.Constexpr,
    rungs: cutlass.Constexpr,
):
    """Warp-0 descending scan of the 256 bins; no internal barrier.

    Pins the highest bin whose cumulative count from the top reaches ``target``
    (bin 0 when the total falls short) with its count and the count strictly
    above it. Leaves every bin holding its output cursor (count strictly above)
    or zero. With ``rungs`` two extra targets pin only their bin IDs.
    """
    per_lane = _BINS // 32
    if tidx < 32:
        lane = tidx
        base = lane * per_lane
        counts = cute.make_rmem_tensor((per_lane,), Int32)
        local = Int32(0)
        for j in cutlass.range_constexpr(per_lane):
            counts[j] = hist[base + j]
            local += counts[j]
        inclusive = _warp_scan_inclusive(local, lane, 32)
        total = cute.arch.shuffle_sync(inclusive, 31)
        after = total - inclusive
        if lane == 0:
            res[_RES_TOT] = total
        for j in cutlass.range_constexpr(per_lane - 1, -1, -1):
            count = counts[j]
            bin_id = base + j
            if cutlass.const_expr(zero):
                hist[bin_id] = Int32(0)
            else:
                hist[bin_id] = after
            if after < target and (after + count >= target or bin_id == 0):
                res[_RES_B] = bin_id
                res[_RES_ABOVE] = after
                res[_RES_M] = count
            if cutlass.const_expr(rungs):
                if after < target2 and (after + count >= target2 or bin_id == 0):
                    res[_RES_B2] = bin_id
                if after < target3 and (after + count >= target3 or bin_id == 0):
                    res[_RES_B3] = bin_id
            after += count


@cute.jit
def _warp_reserve(count: Int32, counter: cute.Pointer, lane: Int32) -> Int32:
    """Reserve ``count`` consecutive slots per lane with one atomic per warp."""
    inclusive = _warp_scan_inclusive(count, lane, 32)
    base = Int32(0)
    if lane == 31 and inclusive > 0:
        base = cute.arch.atomic_add(counter, inclusive, sem="relaxed", scope="cta")
    return cute.arch.shuffle_sync(base, 31) + inclusive - count


@cute.jit
def _ballot_pair_emit(
    winner,
    tie,
    idx: Int32,
    winner_base: Int32,
    winner_cap: Int32,
    tie_base: Int32,
    tie_cap: Int32,
    row_output: cute.Tensor,
    cursors: cute.Tensor,
    lane: Int32,
):
    """Warp-aggregated emission of strict winners and capped ties (full warp required)."""
    winner_mask = cute.arch.vote_ballot_sync(winner != 0)
    tie_mask = cute.arch.vote_ballot_sync(tie != 0)
    winner_start = Int32(0)
    tie_start = Int32(0)
    if lane == 0:
        if winner_mask != 0:
            winner_start = cute.arch.atomic_add(
                cursors.iterator + 1,
                Int32(cute.arch.popc(winner_mask)),
                sem="relaxed",
                scope="cta",
            )
        if tie_mask != 0:
            tie_start = cute.arch.atomic_add(
                cursors.iterator + 2, Int32(cute.arch.popc(tie_mask)), sem="relaxed", scope="cta"
            )
    winner_start = cute.arch.shuffle_sync(winner_start, 0)
    tie_start = cute.arch.shuffle_sync(tie_start, 0)
    lower = Int32(cute.arch.lanemask_lt())
    if winner != 0:
        slot = winner_start + Int32(cute.arch.popc(winner_mask & lower))
        if slot < winner_cap:
            row_output[winner_base + slot] = idx
    if tie != 0:
        slot = tie_start + Int32(cute.arch.popc(tie_mask & lower))
        if slot < tie_cap:
            row_output[tie_base + slot] = idx


class IndexerGVR2TopKKernel(IndexerTopKKernel):
    """Select indices from compact FP32 scores[P,2,S] into INT32 output[B,T,K].

    Constructor and ``(scores, output, pair_start, stream)`` TVM-FFI ABI match
    IndexerTopKKernel, including odd query pairs, compressed causal prefixes,
    and int64 address specialization. Inherited helpers implement the radix
    histogram crossing and the common slab-coordinate contract.
    """

    block_threads = 512
    unroll = 8  # float4 loads per thread per streaming batch: one 32-bit survivor mask
    # 64-register launch bounds: 2 x 512 or 4 x 256 CTAs per SM. More rows than two
    # CTAs per SM (148 SMs) switch to the narrow variant so a slab stays in one wave.
    narrow_threads = 256
    wide_row_limit = 296

    def __init__(
        self,
        topk: int,
        causal: bool,
        use_int64_offsets: bool = False,
        *,
        compress_ratio: int = 1,
        block_threads: int | None = None,
    ):
        super().__init__(topk, causal, use_int64_offsets, compress_ratio=compress_ratio)
        if block_threads is not None:
            self.block_threads = block_threads
        self.narrow = None
        if self.block_threads != self.narrow_threads:
            self.narrow = IndexerGVR2TopKKernel(
                topk,
                causal,
                use_int64_offsets,
                compress_ratio=compress_ratio,
                block_threads=self.narrow_threads,
            )

    def get_name(self) -> str:
        return (
            f"indexer_gvr2_topk_k{self.topk}_c{int(self.causal)}_r{self.compress_ratio}_"
            f"i64{int(self.use_int64_offsets)}_th{self.block_threads}_u{self.unroll}"
        )

    @cute.jit
    def sample_plan(self, seg_len: Int32, quads: Int32):
        """Per-row sample geometry: threads, float4-pair stride and sample-rank targets.

        Mirrors the upstream non-split ``route_dynamic`` formulas for one CTA per row.
        """
        topk = self.topk
        aim_static = 11 * topk // 8 if topk >= 1024 else 3 * topk // 2
        aim_static = max(min(aim_static, _STAGE_MAX // 2), topk)
        sample_factor = 64 if topk >= 1024 else 32
        aim = Int32(aim_static)
        root = Int32(cute_math.sqrt(Float32(seg_len) * Float32(6.0)) + Float32(0.5))
        aim = max(aim, root)
        if aim > _STAGE_MAX // 2:
            aim = Int32(_STAGE_MAX // 2)
        selected = Int32((Int64(sample_factor) * Int64(seg_len)) // Int64(aim))
        if selected < 256:
            selected = Int32(256)
        selected = min(selected, seg_len // 2)
        pairs = selected >> 3
        if pairs < 1:
            pairs = Int32(1)
        half = quads >> 1
        if half < 1:
            half = Int32(1)
        pairs = min(pairs, half)
        if pairs > self.block_threads:
            pairs = Int32(self.block_threads)
        stride = half // pairs
        threads = half // stride
        if threads > self.block_threads:
            threads = Int32(self.block_threads)
        if quads < 4:
            threads = Int32(0)
        target = (aim * (threads * 8)) // seg_len
        if target < 1:
            target = Int32(1)
        target_k = (Int32(topk) * (threads * 8)) // seg_len
        if target_k < 1:
            target_k = Int32(1)
        return threads, stride, target, target_k

    @cute.jit
    def sample_thresholds(
        self,
        vector_base: Int64,
        quads: Int32,
        seg_len: Int32,
        hist: cute.Tensor,
        warp_min: cute.Tensor,
        warp_max: cute.Tensor,
        res: cute.Tensor,
        tidx: Int32,
    ):
        """Bin a stratified float4-pair sample and return (guess, lower rung, window top).

        The guess estimates the value at rank ``aim`` (> K), the rung the value at
        twice that depth, and the window top bounds the classification range by
        the estimated K-th value. Any failure returns ``-inf`` guesses.
        """
        threads, stride, target, target_k = self.sample_plan(seg_len, quads)
        lane = tidx % 32
        first = cute.make_rmem_tensor((4,), Float32)
        second = cute.make_rmem_tensor((4,), Float32)
        has_sample = tidx < threads
        low = Float32(_INF)
        high = Float32(-_INF)
        if has_sample:
            quad = Int64(tidx * stride * 2)
            for t, v in enumerate(_load_float4(vector_base + quad * 16)):
                first[t] = v
            for t, v in enumerate(_load_float4(vector_base + (quad + 1) * 16)):
                second[t] = v
            for t in cutlass.range_constexpr(4):
                low = cute.arch.fmin(low, first[t])
                high = cute.arch.fmax(high, first[t])
                low = cute.arch.fmin(low, second[t])
                high = cute.arch.fmax(high, second[t])
        # Pull this thread's first streaming batch toward L2 while the sample is reduced,
        # scanned and published (clamped in-row; short rows repeat their last float4).
        if quads > 0:
            for u in cutlass.range_constexpr(self.unroll):
                quad = min(tidx + u * self.block_threads, quads - 1)
                _prefetch_l2(vector_base + Int64(quad) * 16)
        key_min = cute.arch.warp_redux_sync(_ordered_key(low), "umin")
        key_max = cute.arch.warp_redux_sync(_ordered_key(high), "umax")
        if lane == 0:
            warp_min[tidx // 32] = key_min
            warp_max[tidx // 32] = key_max
        cute.arch.barrier()
        key_min = Uint32(0xFFFFFFFF)
        key_max = Uint32(0)
        if lane < self.block_threads // 32:
            key_min = warp_min[lane]
            key_max = warp_max[lane]
        sample_min = _key_to_float(cute.arch.warp_redux_sync(key_min, "umin"))
        sample_max = _key_to_float(cute.arch.warp_redux_sync(key_max, "umax"))
        width = (sample_max - sample_min) * Float32(1.0 / _BINS)
        # A finite positive width excludes constant samples and overflowing ranges.
        usable = Int32(0)
        if threads > 0 and width > Float32(0.0) and width < Float32(_INF):
            usable = Int32(1)
        if usable != 0:
            scale = _bin_scale(sample_max - sample_min)
            if has_sample:
                for t in cutlass.range_constexpr(4):
                    bin_id = _bin_index(first[t], sample_min, scale)
                    cute.arch.atomic_add(
                        hist.iterator + bin_id, Int32(1), sem="relaxed", scope="cta"
                    )
                    bin_id = _bin_index(second[t], sample_min, scale)
                    cute.arch.atomic_add(
                        hist.iterator + bin_id, Int32(1), sem="relaxed", scope="cta"
                    )
        cute.arch.barrier()
        _scan_cross(hist, res, target, target_k, target * 2, tidx, True, True)
        cute.arch.barrier()
        total = res[_RES_TOT]
        guess = Float32(-_INF)
        rung = Float32(-_INF)
        window_top = Float32(-_INF)
        if usable != 0 and total >= target:
            guess = Float32(res[_RES_B]) * width + sample_min
            guess_k = Float32(res[_RES_B2]) * width + sample_min
            gap = cute.arch.fmax(guess_k - guess, Float32(0.0))
            window_top = cute.arch.fmax(guess + Float32(4.0) * gap, guess + Float32(8.0) * width)
            if total >= target * 2:
                lower = Float32(res[_RES_B3]) * width + sample_min
                if lower < guess:
                    rung = lower
        return guess, rung, window_top

    @cute.jit
    def stage(
        self,
        value: Float32,
        idx: Int32,
        pos: Int32,
        low: Float32,
        scale: Float32,
        hist: cute.Tensor,
        keys: cute.Tensor,
        indices: cute.Tensor,
    ):
        """Classify one survivor and stage its bits/index; overflow is counted, not staged."""
        bin_id = _bin_index(value, low, scale)
        cute.arch.atomic_add(hist.iterator + bin_id, Int32(1), sem="relaxed", scope="cta")
        if pos < _STAGE_MAX:
            keys[pos] = _float_bits(value)
            indices[pos] = idx

    @cute.jit
    def collect(
        self,
        row: cute.Tensor,
        vector_base: Int64,
        head: Int32,
        quads: Int32,
        tail_start: Int32,
        tail_count: Int32,
        low: Float32,
        scale: Float32,
        hist: cute.Tensor,
        keys: cute.Tensor,
        indices: cute.Tensor,
        cursors: cute.Tensor,
        tidx: Int32,
    ):
        """Stream the row in float4 batches; histogram and stage every value >= low.

        Each batch builds a 32-bit survivor mask, reserves slots with one warp
        scan and atomic, then walks the mask reloading survivors (cache hits)
        instead of keeping all batch values live. Unaligned head and tail
        scalars are handled after the vector body.
        """
        unroll = self.unroll
        threads = self.block_threads
        lane = tidx % 32
        step = threads * unroll
        full_batches = quads // step
        batches = cute.ceil_div(quads, step)
        frags = [cute.make_rmem_tensor((4,), Float32) for _ in range(unroll)]
        for batch in range(batches):
            first_quad = batch * step + tidx
            mask = Int32(0)
            if batch < full_batches:
                for u in cutlass.range_constexpr(unroll):
                    for q, v in enumerate(
                        _load_float4(vector_base + Int64(first_quad + u * threads) * 16)
                    ):
                        frags[u][q] = v
                for u in cutlass.range_constexpr(unroll):
                    for q in cutlass.range_constexpr(4):
                        mask |= Int32(frags[u][q] >= low) << (u * 4 + q)
            else:
                for u in cutlass.range_constexpr(unroll):
                    quad = first_quad + u * threads
                    if quad >= quads:
                        quad = quads - 1
                    for q, v in enumerate(_load_float4(vector_base + Int64(quad) * 16)):
                        frags[u][q] = v
                for u in cutlass.range_constexpr(unroll):
                    if first_quad + u * threads < quads:
                        for q in cutlass.range_constexpr(4):
                            mask |= Int32(frags[u][q] >= low) << (u * 4 + q)
            pos = _warp_reserve(Int32(cute.arch.popc(mask)), cursors.iterator, lane)
            # Survivor walk, software-pipelined one deep: the next reload is in flight
            # while the current survivor is classified and staged.
            if mask != 0:
                bit = _lowest_set_bit(mask)
                mask &= mask - 1
                idx = head + ((first_quad + (bit >> 2) * threads) << 2) + (bit & 3)
                value = row[idx]
                while mask != 0:
                    bit = _lowest_set_bit(mask)
                    mask &= mask - 1
                    next_idx = head + ((first_quad + (bit >> 2) * threads) << 2) + (bit & 3)
                    next_value = row[next_idx]
                    self.stage(value, idx, pos, low, scale, hist, keys, indices)
                    pos += 1
                    idx = next_idx
                    value = next_value
                self.stage(value, idx, pos, low, scale, hist, keys, indices)
        for i in range(tidx, head + tail_count, threads):
            idx = i if i < head else tail_start + (i - head)
            value = row[idx]
            if value >= low:
                pos = cute.arch.atomic_add(cursors.iterator, Int32(1), sem="relaxed", scope="cta")
                self.stage(value, idx, pos, low, scale, hist, keys, indices)

    @cute.jit
    def emit_bins(
        self,
        row: cute.Tensor,
        row_output: cute.Tensor,
        seg_len: Int32,
        complete,
        staged: Int32,
        crossing: Int32,
        above: Int32,
        limit: Int32,
        low: Float32,
        scale: Float32,
        hist: cute.Tensor,
        keys: cute.Tensor,
        indices: cute.Tensor,
        cross_keys: cute.Tensor,
        cross_indices: cute.Tensor,
        tidx: Int32,
    ):
        """Emit every bin above the crossing through its cursor; stage the crossing bin.

        Slots at or past ``limit`` occur only for the crossing bin when it is not
        emitted whole; they land in ``[0, crossing count)`` of the crossing scratch.
        A staging overflow re-sweeps the row with the same classification.
        """
        if complete:
            for i in range(tidx, staged, self.block_threads):
                bits = keys[i]
                idx = indices[i]
                bin_id = _bin_index(_bits_to_float(bits), low, scale)
                if bin_id >= crossing:
                    slot = cute.arch.atomic_add(
                        hist.iterator + bin_id, Int32(1), sem="relaxed", scope="cta"
                    )
                    if slot < limit:
                        row_output[slot] = idx
                    else:
                        cross_keys[slot - above] = _signed_key(_bits_to_float(bits))
                        cross_indices[slot - above] = idx
        else:
            for i in range(tidx, seg_len, self.block_threads):
                value = row[i]
                if value >= low:
                    bin_id = _bin_index(value, low, scale)
                    if bin_id >= crossing:
                        slot = cute.arch.atomic_add(
                            hist.iterator + bin_id, Int32(1), sem="relaxed", scope="cta"
                        )
                        if slot < limit:
                            row_output[slot] = Int32(i)
                        else:
                            cross_keys[slot - above] = _signed_key(value)
                            cross_indices[slot - above] = Int32(i)

    @cute.jit
    def refine_crossing(
        self,
        row_output: cute.Tensor,
        count: Int32,
        need: Int32,
        above: Int32,
        hist: cute.Tensor,
        res: cute.Tensor,
        key_range: cute.Tensor,
        cursors: cute.Tensor,
        cross_keys: cute.Tensor,
        cross_indices: cute.Tensor,
        tidx: Int32,
    ):
        """Emit the exact top-``need`` of the staged crossing bin.

        Small bins (the common case: a few candidates) are ranked directly in one
        pass; larger ones use key-space narrowing.
        """
        threads = self.block_threads
        cute.arch.barrier()  # every crossing candidate is staged; cursors are consumed
        if count <= _RANK_MAX:
            # Direct ranking under the total order (key, index): the ``need`` smallest
            # ranks are exactly the required winners, ties resolved by index.
            for i in range(tidx, count, threads):
                key = cross_keys[i]
                idx = cross_indices[i]
                rank = Int32(0)
                for j in range(count):
                    other_key = cross_keys[j]
                    if other_key > key or (other_key == key and cross_indices[j] > idx):
                        rank += 1
                if rank < need:
                    row_output[above + rank] = idx
        else:
            self.narrow_crossing(
                row_output,
                count,
                need,
                above,
                hist,
                res,
                key_range,
                cursors,
                cross_keys,
                cross_indices,
                tidx,
            )

    @cute.jit
    def narrow_crossing(
        self,
        row_output: cute.Tensor,
        count: Int32,
        need: Int32,
        above: Int32,
        hist: cute.Tensor,
        res: cute.Tensor,
        key_range: cute.Tensor,
        cursors: cute.Tensor,
        cross_keys: cute.Tensor,
        cross_indices: cute.Tensor,
        tidx: Int32,
    ):
        """Exact top-``need`` of a large crossing bin by adaptive 8-bit key narrowing.

        Each level histograms only the keys still inside the open range, using
        the range width to pick the shift, so the crossing key is pinned in at
        most four levels. Winners above the final key and a capped number of
        exact ties are emitted with warp ballots.
        """
        threads = self.block_threads
        lane = tidx % 32
        if tidx == 0:
            key_range[0] = Int32(0x7FFFFFFF)
            key_range[1] = Int32(-0x80000000)
        if tidx < _BINS:
            hist[tidx] = Int32(0)
        cute.arch.barrier()
        for i in range(tidx, count, threads):
            key = cross_keys[i]
            cute.arch.atomic_min(key_range.iterator, key, sem="relaxed", scope="cta")
            cute.arch.atomic_max(key_range.iterator + 1, key, sem="relaxed", scope="cta")
        cute.arch.barrier()
        low_key = key_range[0]
        high_key = key_range[1]
        winners = Int32(0)
        remaining_need = need
        remaining = count
        # Final rule: keys above ``low_key`` win; with ``ties`` set, exactly
        # ``remaining_need`` keys equal to it are added, otherwise it wins too.
        ties = Int32(1)
        done = Int32(0)
        while done == 0:
            if remaining_need == remaining:
                winners += remaining
                remaining_need = Int32(0)
                ties = Int32(0)
                done = Int32(1)
            elif low_key >= high_key:
                done = Int32(1)
            else:
                # Signed 64-bit interval math: the span is below 2^32 and the pinned
                # bucket is intersected with the open interval, so no endpoint can wrap.
                low64 = Int64(low_key)
                span = Int64(high_key) - low64
                shift = Int64(64) - Int64(cute.arch.clz(span | Int64(1))) - 8
                if shift < 0:
                    shift = Int64(0)
                for i in range(tidx, count, threads):
                    key = cross_keys[i]
                    if key >= low_key and key <= high_key:
                        bin_id = min(Int32((Int64(key) - low64) >> shift), Int32(_BINS - 1))
                        cute.arch.atomic_add(
                            hist.iterator + bin_id, Int32(1), sem="relaxed", scope="cta"
                        )
                cute.arch.barrier()
                _scan_cross(hist, res, remaining_need, Int32(0), Int32(0), tidx, True, False)
                cute.arch.barrier()
                level_above = res[_RES_ABOVE]
                winners += level_above
                remaining_need -= level_above
                remaining = res[_RES_M]
                new_low = low64 + (Int64(res[_RES_B]) << shift)
                new_high = new_low + (Int64(1) << shift) - Int64(1)
                if new_high < Int64(high_key):
                    high_key = Int32(new_high)
                low_key = Int32(new_low)
        if tidx == 0:
            cursors[1] = Int32(0)
            cursors[2] = Int32(0)
        cute.arch.barrier()
        for batch in range(cute.ceil_div(count, threads)):
            i = batch * threads + tidx
            winner = Int32(0)
            tie = Int32(0)
            idx = Int32(0)
            if i < count:
                key = cross_keys[i]
                idx = cross_indices[i]
                if key > low_key or (ties == 0 and key == low_key):
                    winner = Int32(1)
                if ties != 0 and key == low_key:
                    tie = Int32(1)
            _ballot_pair_emit(
                winner,
                tie,
                idx,
                above,
                winners,
                above + winners,
                remaining_need,
                row_output,
                cursors,
                lane,
            )

    @cute.kernel
    def kernel(self, scores: cute.Tensor, output: cute.Tensor, pair_start: Int32 | Int64):
        tidx = cute.arch.thread_idx()[0]
        block_id = self.upcast_offset(cute.arch.block_idx()[0])
        slab_pair = block_id // 2
        qi = block_id % 2
        num_queries = self.upcast_offset(output.shape[1])
        pairs_per_batch = cute.ceil_div(num_queries, 2)
        global_pair = self.upcast_offset(pair_start) + slab_pair
        batch = global_pair // pairs_per_batch
        query = (global_pair % pairs_per_batch) * 2 + qi
        threads = self.block_threads

        smem = SmemAllocator()
        hist = smem.allocate_tensor(Int32, cute.make_layout((_BINS,)), byte_alignment=128)
        warp_min = smem.allocate_tensor(
            Uint32, cute.make_layout((threads // 32,)), byte_alignment=128
        )
        warp_max = smem.allocate_tensor(
            Uint32, cute.make_layout((threads // 32,)), byte_alignment=128
        )
        res = smem.allocate_tensor(Int32, cute.make_layout((8,)), byte_alignment=128)
        # Cursors: staged survivors, crossing winners, crossing ties.
        cursors = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=128)
        key_range = smem.allocate_tensor(Int32, cute.make_layout((2,)), byte_alignment=128)
        # Radix fallback scalars (the fallback histogram aliases the crossing scratch).
        warp_sums = smem.allocate_tensor(
            Int32, cute.make_layout((threads // 32,)), byte_alignment=128
        )
        threshold = smem.allocate_tensor(Int32, cute.make_layout((2,)), byte_alignment=128)
        counters = smem.allocate_tensor(Int32, cute.make_layout((3,)), byte_alignment=128)
        # Staged survivor bits/indices double as the fallback's ordered keys/shrink buffer.
        keys = smem.allocate_tensor(Uint32, cute.make_layout((_STAGE_MAX,)), byte_alignment=128)
        indices = smem.allocate_tensor(Int32, cute.make_layout((_STAGE_MAX,)), byte_alignment=128)
        cross_block = smem.allocate_tensor(
            Int32, cute.make_layout((_NUM_BINS_11,)), byte_alignment=128
        )
        # Crossing scratch holds signed-flipped keys, then indices (see _signed_key).
        cross_halves = cute.logical_divide(cross_block, _CROSS_MAX)
        cross_keys = cross_halves[None, 0]
        cross_indices = cross_halves[None, 1]

        if batch < output.shape[0] and query < num_queries:
            row_output = output[batch, query, None]
            seg_len = Int32(scores.shape[2])
            if cutlass.const_expr(self.causal):
                seg_len = Int32(self.visible_candidates(query))
            if seg_len <= self.topk:
                for slot in range(tidx, self.topk, threads):
                    row_output[slot] = Int32(slot) if slot < seg_len else Int32(-1)
            else:
                if cutlass.const_expr(self.topk >= _SHRINK_MAX):
                    self.radix_select(
                        scores[slab_pair, qi, None],
                        row_output,
                        seg_len,
                        False,
                        cross_block,
                        warp_sums,
                        threshold,
                        counters,
                        keys,
                        indices,
                        tidx,
                    )
                else:
                    row = scores[slab_pair, qi, None]
                    # Vector loads need 16-byte alignment; the row base is only 4-byte aligned
                    # when S % 4 != 0, so up to three head scalars precede the float4 body.
                    row_address = Int64(row.iterator.toint())
                    head = ((Int32(16) - Int32(row_address & 15)) >> 2) & 3
                    quads = Int32(0)
                    if seg_len > head:
                        quads = (seg_len - head) >> 2
                    tail_start = head + (quads << 2)
                    tail_count = seg_len - tail_start
                    vector_base = row_address + Int64(head) * 4

                    if tidx == 0:
                        cursors[0] = Int32(0)
                    if tidx < _BINS:
                        hist[tidx] = Int32(0)
                    guess, rung, window_top = self.sample_thresholds(
                        vector_base, quads, seg_len, hist, warp_min, warp_max, res, tidx
                    )

                    # Verify ladder: the sample guess, then its lower rung. Every decision
                    # below is CTA-uniform because it derives from barrier-published counts.
                    valid = Int32(0)
                    staged = Int32(0)
                    above = Int32(0)
                    crossing_count = Int32(0)
                    crossing = Int32(0)
                    need = Int32(0)
                    low = guess
                    scale = Float32(1.0)
                    attempt = Int32(0)
                    running = Int32(0)
                    if guess > Float32(-_INF):
                        running = Int32(1)
                    while running != 0:
                        if attempt > 0:
                            if tidx < _BINS:
                                hist[tidx] = Int32(0)
                            if tidx == 0:
                                cursors[0] = Int32(0)
                            cute.arch.barrier()
                        low = guess
                        top = Float32(3.0e38)
                        if window_top > low and window_top < top:
                            top = window_top
                        scale = _bin_scale(top - low)
                        self.collect(
                            row,
                            vector_base,
                            head,
                            quads,
                            tail_start,
                            tail_count,
                            low,
                            scale,
                            hist,
                            keys,
                            indices,
                            cursors,
                            tidx,
                        )
                        cute.arch.barrier()
                        staged = cursors[0]
                        _scan_cross(
                            hist, res, Int32(self.topk), Int32(0), Int32(0), tidx, False, False
                        )
                        cute.arch.barrier()
                        if res[_RES_TOT] >= self.topk:
                            valid = Int32(1)
                            above = res[_RES_ABOVE]
                            crossing_count = res[_RES_M]
                            crossing = res[_RES_B]
                            need = Int32(self.topk) - above
                            running = Int32(0)
                        elif attempt == 0 and rung > Float32(-_INF) and rung < low:
                            guess = rung
                        else:
                            running = Int32(0)
                        attempt += 1

                    whole = need >= crossing_count
                    if valid != 0 and crossing_count <= _CROSS_MAX:
                        limit = above
                        if whole:
                            limit = above + crossing_count
                        self.emit_bins(
                            row,
                            row_output,
                            seg_len,
                            staged <= _STAGE_MAX,
                            staged,
                            crossing,
                            above,
                            limit,
                            low,
                            scale,
                            hist,
                            keys,
                            indices,
                            cross_keys,
                            cross_indices,
                            tidx,
                        )
                        if not whole:
                            self.refine_crossing(
                                row_output,
                                crossing_count,
                                need,
                                above,
                                hist,
                                res,
                                key_range,
                                cursors,
                                cross_keys,
                                cross_indices,
                                tidx,
                            )
                    elif valid != 0 and staged <= _STAGE_MAX:
                        # Every top-K value is staged; convert bits to ordered keys in place.
                        for i in range(tidx, staged, threads):
                            keys[i] = _ordered_key(_bits_to_float(keys[i]))
                        if tidx == 0:
                            counters[0] = staged
                        cute.arch.barrier()
                        self.radix_select(
                            row,
                            row_output,
                            seg_len,
                            True,
                            cross_block,
                            warp_sums,
                            threshold,
                            counters,
                            keys,
                            indices,
                            tidx,
                        )
                    else:
                        self.radix_select(
                            row,
                            row_output,
                            seg_len,
                            False,
                            cross_block,
                            warp_sums,
                            threshold,
                            counters,
                            keys,
                            indices,
                            tidx,
                        )

    @cute.jit
    def __call__(
        self,
        scores: cute.Tensor,
        output: cute.Tensor,
        pair_start: Int32 | Int64,
        stream: cuda.CUstream,
    ):
        rows = scores.shape[0] * 2
        if rows > self.wide_row_limit:
            narrow = self.narrow if cutlass.const_expr(self.narrow is not None) else self
            self.launch_variant(narrow, scores, output, pair_start, stream)
        else:
            self.launch_variant(self, scores, output, pair_start, stream)

    @staticmethod
    def launch_variant(variant, scores, output, pair_start, stream):
        variant.kernel.set_name_prefix(variant.get_name())
        variant.kernel(scores, output, pair_start).launch(
            grid=(scores.shape[0] * 2, 1, 1),
            block=(variant.block_threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1024 // variant.block_threads,
        )
