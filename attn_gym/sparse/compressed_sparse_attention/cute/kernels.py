"""Small CuTe DSL kernels used by the SM100 shared-KV CSA forward path.

The expensive selected-attention tensor-core loop is provided by FlashAttention-4. Everything
around it (compression, normalization, RoPE, index selection, sink merge, head tiling, and inverse
RoPE) stays in CuTe DSL and supports the generalized shape contract in ``assumptions.txt``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass._mlir.extras import types as T
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.reduce import row_reduce


_RMS_EPS = 1.1920928955078125e-7


@cute.jit
def _score_order_key(value, index, num_blocks):
    """Return ordered BF16 score bits; keep the index separate for tie breaks."""
    bits = cutlass.Uint16(
        llvm.bitcast(cutlass.Uint16.mlir_type, value.ir_value())
    )
    if (bits & cutlass.Uint16(0x7FFF)) == cutlass.Uint16(0):
        bits = cutlass.Uint16(0)
    ordered = cutlass.Uint32(0)
    if bits & cutlass.Uint16(0x8000):
        ordered = cutlass.Uint32(bits ^ cutlass.Uint16(0xFFFF))
    else:
        ordered = cutlass.Uint32(bits ^ cutlass.Uint16(0x8000))
    return ordered


@cute.jit
def _score_index_key(value, index, num_blocks):
    """Encode rounded score/index order as a valid numeric FP32 radix key."""

    # Ordered non-NaN BF16 scores occupy the contiguous [0x007f, 0xff80]
    # interval. FP32's positive-normal bit interval contains enough keys to
    # append an exact lower-index tie break for up to 32,638 blocks. Keeping
    # every encoded key positive avoids both the old NaN/sign-bit bug and a
    # negative-key corner in cuDNN's out-of-bounds sentinel handling.
    if const_expr(num_blocks <= 32_638):
        score_order = _score_order_key(value, index, num_blocks)
        # Canonicalize either NaN sign to the +inf rank before rebasing into the
        # contiguous non-NaN interval.
        if score_order < cutlass.Uint32(0x007F) or score_order > cutlass.Uint32(
            0xFF80
        ):
            score_order = cutlass.Uint32(0xFF80)
        score_rank = score_order - cutlass.Uint32(0x007F)
        composite_rank = (
            score_rank * cutlass.Uint32(num_blocks)
            + cutlass.Uint32(num_blocks - 1)
            - cutlass.Uint32(index)
        )
        raw = cutlass.Uint32(0x00800000) + composite_rank
        return Float32(llvm.bitcast(T.f32(), raw.ir_value()))

    # A single FP32 key cannot losslessly hold every BF16 score plus a wider
    # index. Preserve score ordering and leave exact boundary-tie choice to cuDNN.
    return value.to(Float32)


class CompressionNormRope:
    """Fuse the reference's 2R compression, RMSNorm, and tail RoPE."""

    def __init__(self, dtype, batch: int, sequence: int, dim: int, rate: int, rope: int):
        self.dtype = dtype
        self.batch = batch
        self.sequence = sequence
        self.dim = dim
        self.rate = rate
        self.rope = rope
        self.num_blocks = (sequence + rate - 1) // rate
        # Keep an integral even number of coordinates per lane for every DI multiple of 64.
        self.num_threads = math.gcd(dim // 2, 128)
        self.values_per_thread = dim // self.num_threads
        assert dim % self.num_threads == 0
        assert self.values_per_thread % 2 == 0

    @cute.jit
    def __call__(
        self,
        c_a: cute.Tensor,
        c_b: cute.Tensor,
        z_a: cute.Tensor,
        z_b: cute.Tensor,
        bias_a: cute.Tensor,
        bias_b: cute.Tensor,
        norm_weight: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(c_a, c_b, z_a, z_b, bias_a, bias_b, norm_weight, cos, sin, out).launch(
            grid=[self.batch * self.num_blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        c_a: cute.Tensor,
        c_b: cute.Tensor,
        z_a: cute.Tensor,
        z_b: cute.Tensor,
        bias_a: cute.Tensor,
        bias_b: cute.Tensor,
        norm_weight: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block_row, _, _ = cute.arch.block_idx()
        batch_idx = block_row // self.num_blocks
        block_idx = block_row - batch_idx * self.num_blocks
        d_base = tid * self.values_per_thread

        compressed = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        local_sq = Float32(0.0)
        for j in cutlass.range_constexpr(self.values_per_thread):
            d = d_base + j
            maximum = -Float32.inf
            for r in cutlass.range(self.rate, unroll=1):
                pos_a = block_idx * self.rate + r
                if pos_a < self.sequence:
                    logit_a = (
                        z_a[batch_idx, 0, pos_a, d].to(Float32)
                        + bias_a[r, d].to(Float32)
                    ).to(self.dtype).to(Float32)
                    maximum = cute.arch.fmax(maximum, logit_a)
                if block_idx > 0:
                    pos_b = (block_idx - 1) * self.rate + r
                    if pos_b < self.sequence:
                        logit_b = (
                            z_b[batch_idx, 0, pos_b, d].to(Float32)
                            + bias_b[r, d].to(Float32)
                        ).to(self.dtype).to(Float32)
                        maximum = cute.arch.fmax(maximum, logit_b)

            denominator = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                pos_a = block_idx * self.rate + r
                if pos_a < self.sequence:
                    logit_a = (
                        z_a[batch_idx, 0, pos_a, d].to(Float32)
                        + bias_a[r, d].to(Float32)
                    ).to(self.dtype).to(Float32)
                    denominator += cute.math.exp(
                        logit_a - maximum,
                        fastmath=False,
                    )
                if block_idx > 0:
                    pos_b = (block_idx - 1) * self.rate + r
                    if pos_b < self.sequence:
                        logit_b = (
                            z_b[batch_idx, 0, pos_b, d].to(Float32)
                            + bias_b[r, d].to(Float32)
                        ).to(self.dtype).to(Float32)
                        denominator += cute.math.exp(
                            logit_b - maximum,
                            fastmath=False,
                        )

            value = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                pos_a = block_idx * self.rate + r
                product_a = Float32(0.0)
                if pos_a < self.sequence:
                    logit_a = (
                        z_a[batch_idx, 0, pos_a, d].to(Float32)
                        + bias_a[r, d].to(Float32)
                    ).to(self.dtype).to(Float32)
                    probability_a = (
                        cute.math.exp(logit_a - maximum, fastmath=False) / denominator
                    ).to(self.dtype).to(Float32)
                    product_a = (
                        c_a[batch_idx, 0, pos_a, d].to(Float32) * probability_a
                    ).to(self.dtype).to(Float32)
                product_b = Float32(0.0)
                if block_idx > 0:
                    pos_b = (block_idx - 1) * self.rate + r
                    if pos_b < self.sequence:
                        logit_b = (
                            z_b[batch_idx, 0, pos_b, d].to(Float32)
                            + bias_b[r, d].to(Float32)
                        ).to(self.dtype).to(Float32)
                        probability_b = (
                            cute.math.exp(logit_b - maximum, fastmath=False) / denominator
                        ).to(self.dtype).to(Float32)
                        product_b = (
                            c_b[batch_idx, 0, pos_b, d].to(Float32) * probability_b
                        ).to(self.dtype).to(Float32)
                value += (product_a + product_b).to(self.dtype).to(Float32)

            # Compression is stored before RMSNorm in the oracle.
            value = value.to(self.dtype).to(Float32)
            compressed[j] = value
            local_sq += value * value

        smem = cutlass.utils.SmemAllocator()
        reduction = smem.allocate_tensor(
            Float32,
            cute.make_layout((1, (self.num_threads // cute.arch.WARP_SIZE, 1))),
            byte_alignment=16,
        )
        sum_sq = row_reduce(
            local_sq,
            cute.ReductionOp.ADD,
            self.num_threads,
            reduction,
            init_val=0.0,
        )
        rstd = cute.math.rsqrt(sum_sq / self.dim + _RMS_EPS, fastmath=True)
        position = block_idx * self.rate

        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j0 = pair * 2
            d0 = d_base + j0
            x0 = (
                compressed[j0] * rstd * norm_weight[d0].to(Float32)
            ).to(self.dtype).to(Float32)
            x1 = (
                compressed[j0 + 1] * rstd * norm_weight[d0 + 1].to(Float32)
            ).to(self.dtype).to(Float32)
            y0, y1 = x0, x1
            if d0 >= self.dim - self.rope:
                rope_pair = (d0 - (self.dim - self.rope)) // 2
                c = cos[position, rope_pair].to(Float32)
                s = sin[position, rope_pair].to(Float32)
                y0 = x0 * c - x1 * s
                y1 = x0 * s + x1 * c
            out[batch_idx, block_idx, 0, d0] = y0.to(self.dtype)
            out[batch_idx, block_idx, 0, d0 + 1] = y1.to(self.dtype)


class LocalNormRope:
    """Normalize/rotate the shared local KV and write a left-padded BSHD buffer."""

    def __init__(self, dtype, batch: int, sequence: int, dim: int, rope: int, pad: int):
        self.dtype = dtype
        self.batch = batch
        self.sequence = sequence
        self.dim = dim
        self.rope = rope
        self.pad = pad
        self.num_threads = 128
        self.values_per_thread = dim // self.num_threads
        assert dim % self.num_threads == 0 and self.values_per_thread % 2 == 0

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,
        norm_weight: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(kv, norm_weight, cos, sin, out).launch(
            grid=[self.batch * (self.sequence + self.pad), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        kv: cute.Tensor,
        norm_weight: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        padded_sequence = self.sequence + self.pad
        batch_idx = row // padded_sequence
        padded_pos = row - batch_idx * padded_sequence
        d_base = tid * self.values_per_thread

        if padded_pos < self.pad:
            for j in cutlass.range_constexpr(self.values_per_thread):
                out[batch_idx, padded_pos, 0, d_base + j] = self.dtype(0.0)
        else:
            position = padded_pos - self.pad
            values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
            local_sq = Float32(0.0)
            for j in cutlass.range_constexpr(self.values_per_thread):
                value = kv[batch_idx, 0, position, d_base + j].to(Float32)
                values[j] = value
                local_sq += value * value

            smem = cutlass.utils.SmemAllocator()
            reduction = smem.allocate_tensor(
                Float32,
                cute.make_layout((1, (self.num_threads // cute.arch.WARP_SIZE, 1))),
                byte_alignment=16,
            )
            sum_sq = row_reduce(
                local_sq,
                cute.ReductionOp.ADD,
                self.num_threads,
                reduction,
                init_val=0.0,
            )
            rstd = cute.math.rsqrt(sum_sq / self.dim + _RMS_EPS, fastmath=True)

            for pair in cutlass.range_constexpr(self.values_per_thread // 2):
                j0 = pair * 2
                d0 = d_base + j0
                x0 = (
                    values[j0] * rstd * norm_weight[d0].to(Float32)
                ).to(self.dtype).to(Float32)
                x1 = (
                    values[j0 + 1] * rstd * norm_weight[d0 + 1].to(Float32)
                ).to(self.dtype).to(Float32)
                y0, y1 = x0, x1
                if d0 >= self.dim - self.rope:
                    rope_pair = (d0 - (self.dim - self.rope)) // 2
                    c = cos[position, rope_pair].to(Float32)
                    s = sin[position, rope_pair].to(Float32)
                    y0 = x0 * c - x1 * s
                    y1 = x0 * s + x1 * c
                out[batch_idx, padded_pos, 0, d0] = y0.to(self.dtype)
                out[batch_idx, padded_pos, 0, d0 + 1] = y1.to(self.dtype)


class QueryRopeTranspose:
    """Fuse BHSD -> BSHD transposition with adjacent-pair tail RoPE."""

    def __init__(
        self,
        dtype,
        batch: int,
        source_heads: int,
        tile_heads: int,
        active_heads: int,
        head_offset: int,
        sequence: int,
        dim: int,
        rope: int,
        mirror_heads: int = 0,
    ):
        self.dtype = dtype
        self.batch = batch
        self.source_heads = source_heads
        self.tile_heads = tile_heads
        self.active_heads = active_heads
        self.head_offset = head_offset
        self.sequence = sequence
        self.dim = dim
        self.rope = rope
        self.mirror_heads = mirror_heads
        self.head_tile = math.gcd(tile_heads, 32)
        self.num_threads = 256
        self.pairs_per_thread = self.head_tile * dim // 2 // self.num_threads
        assert tile_heads in (64, 128)
        assert 0 < active_heads <= tile_heads
        assert mirror_heads in (0, 64) and mirror_heads <= tile_heads
        assert 0 <= head_offset and head_offset + active_heads <= source_heads

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        mirror_out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(q, cos, sin, out, mirror_out).launch(
            grid=[self.batch * self.sequence * (self.tile_heads // self.head_tile), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        mirror_out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        head_tiles = self.tile_heads // self.head_tile
        head_tile_idx = row % head_tiles
        sequence_row = row // head_tiles
        position = sequence_row % self.sequence
        batch_idx = sequence_row // self.sequence
        pair_base = tid * 2
        tile_elements = self.head_tile * self.dim

        for j in cutlass.range_constexpr(self.pairs_per_thread):
            element = pair_base + j * self.num_threads * 2
            if element < tile_elements:
                head_local = element // self.dim
                d0 = element - head_local * self.dim
                head = head_tile_idx * self.head_tile + head_local
                x0 = Float32(0.0)
                x1 = Float32(0.0)
                if head < self.active_heads:
                    source_head = self.head_offset + head
                    x0 = q[batch_idx, source_head, position, d0].to(Float32)
                    x1 = q[batch_idx, source_head, position, d0 + 1].to(Float32)
                y0, y1 = x0, x1
                if d0 >= self.dim - self.rope:
                    rope_pair = (d0 - (self.dim - self.rope)) // 2
                    c = cos[position, rope_pair].to(Float32)
                    s = sin[position, rope_pair].to(Float32)
                    y0 = x0 * c - x1 * s
                    y1 = x0 * s + x1 * c
                out[batch_idx, position, head, d0] = y0.to(self.dtype)
                out[batch_idx, position, head, d0 + 1] = y1.to(self.dtype)
                if const_expr(self.mirror_heads > 0):
                    if head < self.mirror_heads:
                        mirror_out[batch_idx, position, head, d0] = y0.to(self.dtype)
                        mirror_out[batch_idx, position, head, d0 + 1] = y1.to(self.dtype)


class IndexScores:
    """Compute a row slab of causal index scores for the scalable radix top-k path."""

    def __init__(
        self,
        dtype,
        batch: int,
        sequence: int,
        index_heads: int,
        index_dim: int,
        num_blocks: int,
        rate: int,
        rope: int,
        row_chunk: int,
    ):
        self.dtype = dtype
        self.batch = batch
        self.sequence = sequence
        self.index_heads = index_heads
        self.index_dim = index_dim
        self.num_blocks = num_blocks
        self.rate = rate
        self.rope = rope
        self.row_chunk = row_chunk
        self.num_threads = 128
        self.head_groups = (index_heads + 3) // 4
        self.pairs_per_lane = (
            index_dim // 2 + cute.arch.WARP_SIZE - 1
        ) // cute.arch.WARP_SIZE

    @cute.jit
    def __call__(
        self,
        q_i: cute.Tensor,
        compressed_indices: cute.Tensor,
        weights: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        scores: cute.Tensor,
        completed_lengths: cute.Tensor,
        row_offset: Int32,
        active_rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q_i,
            compressed_indices,
            weights,
            cos,
            sin,
            scores,
            completed_lengths,
            row_offset,
        ).launch(
            grid=[active_rows, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q_i: cute.Tensor,
        compressed_indices: cute.Tensor,
        weights: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        scores: cute.Tensor,
        completed_lengths: cute.Tensor,
        row_offset: Int32,
    ):
        """Use four cooperating warps for larger index-head/dimension shapes."""

        tid, _, _ = cute.arch.thread_idx()
        slab_row, _, _ = cute.arch.block_idx()
        row = row_offset + slab_row
        batch_idx = row // self.sequence
        position = row - batch_idx * self.sequence
        warp = tid // cute.arch.WARP_SIZE
        lane = tid % cute.arch.WARP_SIZE
        completed = (position + 1) // self.rate
        if tid == 0:
            completed_lengths[slab_row] = Int32(completed)

        smem = cutlass.utils.SmemAllocator()
        contributions = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.head_groups * 4,)),
            byte_alignment=16,
        )

        q_values = cute.make_rmem_tensor(
            (self.head_groups, self.pairs_per_lane, 2), Float32
        )
        for group in cutlass.range_constexpr(self.head_groups):
            head = group * 4 + warp
            for item in cutlass.range_constexpr(self.pairs_per_lane):
                pair = lane + item * cute.arch.WARP_SIZE
                d0 = pair * 2
                rq0 = Float32(0.0)
                rq1 = Float32(0.0)
                if head < self.index_heads and d0 < self.index_dim:
                    q0 = q_i[batch_idx, head, position, d0].to(Float32)
                    q1 = q_i[batch_idx, head, position, d0 + 1].to(Float32)
                    rq0, rq1 = q0, q1
                    if d0 >= self.index_dim - self.rope:
                        rope_pair = (d0 - (self.index_dim - self.rope)) // 2
                        c = cos[position, rope_pair].to(Float32)
                        s = sin[position, rope_pair].to(Float32)
                        rq0 = q0 * c - q1 * s
                        rq1 = q0 * s + q1 * c
                    rq0 = rq0.to(self.dtype).to(Float32)
                    rq1 = rq1.to(self.dtype).to(Float32)
                q_values[group, item, 0] = rq0
                q_values[group, item, 1] = rq1

        for n in cutlass.range(completed, unroll=1):
            k_values = cute.make_rmem_tensor(
                (self.pairs_per_lane, 2), Float32
            )
            for item in cutlass.range_constexpr(self.pairs_per_lane):
                pair = lane + item * cute.arch.WARP_SIZE
                d0 = pair * 2
                k0 = Float32(0.0)
                k1 = Float32(0.0)
                if d0 < self.index_dim:
                    k0 = compressed_indices[batch_idx, n, 0, d0].to(Float32)
                    k1 = compressed_indices[batch_idx, n, 0, d0 + 1].to(Float32)
                k_values[item, 0] = k0
                k_values[item, 1] = k1

            for group in cutlass.range_constexpr(self.head_groups):
                head = group * 4 + warp
                partial = Float32(0.0)
                if head < self.index_heads:
                    for item in cutlass.range_constexpr(self.pairs_per_lane):
                        pair = lane + item * cute.arch.WARP_SIZE
                        d0 = pair * 2
                        if d0 < self.index_dim:
                            rq0 = q_values[group, item, 0]
                            rq1 = q_values[group, item, 1]
                            k0 = k_values[item, 0]
                            k1 = k_values[item, 1]
                            partial += rq0 * k0 + rq1 * k1
                dot = cute.arch.warp_reduction_sum(partial)
                if lane == 0:
                    weighted = Float32(0.0)
                    if head < self.index_heads:
                        rounded_dot = dot.to(self.dtype).to(Float32)
                        activated = cute.arch.fmax(rounded_dot, Float32(0.0))
                        activated = (
                            activated / math.sqrt(self.index_dim * self.index_heads)
                        ).to(self.dtype).to(Float32)
                        weighted = (
                            activated * weights[batch_idx, position, head].to(Float32)
                        ).to(self.dtype).to(Float32)
                    contributions[group * 4 + warp] = weighted

            cute.arch.barrier()
            if tid == 0:
                score = Float32(0.0)
                for group in cutlass.range_constexpr(self.head_groups):
                    for group_head in cutlass.range_constexpr(4):
                        score += contributions[group * 4 + group_head]
                rounded_score = score.to(self.dtype)
                scores[slab_row, n] = _score_index_key(
                    rounded_score, n, self.num_blocks
                )
            cute.arch.barrier()


class SelectedGather:
    """Pack a radix-selected row slab and its causal local window."""

    def __init__(
        self,
        batch: int,
        sequence: int,
        num_blocks: int,
        topk: int,
        window: int,
        gather_length: int,
        row_chunk: int,
    ):
        self.batch = batch
        self.sequence = sequence
        self.num_blocks = num_blocks
        self.topk = topk
        self.window = min(window, sequence)
        self.gather_length = gather_length
        self.row_chunk = row_chunk
        self.num_threads = 128

    @cute.jit
    def __call__(
        self,
        selected_indices: cute.Tensor,
        gather: cute.Tensor,
        row_offset: Int32,
        active_rows: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            selected_indices, gather, row_offset, active_rows
        ).launch(
            grid=[cute.ceil_div(active_rows, 4), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        selected_indices: cute.Tensor,
        gather: cute.Tensor,
        row_offset: Int32,
        active_rows: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = tid // cute.arch.WARP_SIZE
        lane = tid % cute.arch.WARP_SIZE
        slab_row = block * 4 + warp

        if slab_row < active_rows:
            row = row_offset + slab_row
            batch_idx = row // self.sequence
            position = row - batch_idx * self.sequence
            local_count = min(self.window, position + 1)
            for item in cutlass.range_constexpr(
                self.gather_length // cute.arch.WARP_SIZE
            ):
                slot = lane + item * cute.arch.WARP_SIZE
                value = Int32(-1)
                if slot < self.topk:
                    value = selected_indices[slab_row, slot]
                else:
                    local_slot = slot - self.topk
                    if local_slot < local_count:
                        key = position - local_count + 1 + local_slot
                        value = Int32(self.num_blocks + key)
                gather[batch_idx, position, slot] = value


class CausalGather:
    """Build the gather list directly when every completed block is selected."""

    def __init__(
        self,
        batch: int,
        sequence: int,
        num_blocks: int,
        rate: int,
        topk: int,
        window: int,
        gather_length: int,
    ):
        self.batch = batch
        self.sequence = sequence
        self.num_blocks = num_blocks
        self.rate = rate
        self.topk = topk
        self.window = min(window, sequence)
        self.gather_length = gather_length
        self.num_threads = 128

    @cute.jit
    def __call__(self, gather: cute.Tensor, stream: cuda.CUstream):
        self.kernel(gather).launch(
            grid=[cute.ceil_div(self.batch * self.sequence, 4), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, gather: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = tid // cute.arch.WARP_SIZE
        lane = tid % cute.arch.WARP_SIZE
        row = block * 4 + warp
        if row < self.batch * self.sequence:
            batch_idx = row // self.sequence
            position = row - batch_idx * self.sequence
            completed = min(self.topk, (position + 1) // self.rate)
            local_count = min(self.window, position + 1)
            for item in cutlass.range_constexpr(
                self.gather_length // cute.arch.WARP_SIZE
            ):
                slot = lane + item * cute.arch.WARP_SIZE
                value = Int32(-1)
                if slot < completed:
                    value = Int32(slot)
                else:
                    local_slot = slot - self.topk
                    if local_slot >= 0 and local_slot < local_count:
                        key = position - local_count + 1 + local_slot
                        value = Int32(self.num_blocks + key)
                gather[batch_idx, position, slot] = value


class IndexTopK:
    """Fuse QI RoPE, causal top-k, and the complete selected-KV index list."""

    def __init__(
        self,
        dtype,
        batch: int,
        sequence: int,
        index_heads: int,
        index_dim: int,
        num_blocks: int,
        rate: int,
        topk: int,
        window: int,
        rope: int,
        gather_length: int,
    ):
        self.dtype = dtype
        self.batch = batch
        self.sequence = sequence
        self.index_heads = index_heads
        self.index_dim = index_dim
        self.num_blocks = num_blocks
        self.rate = rate
        self.topk = topk
        self.window = min(window, sequence)
        self.rope = rope
        self.gather_length = gather_length
        self.num_threads = 128
        self.head_groups = (index_heads + 3) // 4
        self.pairs_per_lane = (index_dim // 2 + cute.arch.WARP_SIZE - 1) // cute.arch.WARP_SIZE
        # The common DeepSeek indexer has four 64/128-wide heads.  Assigning one warp to
        # each query row lets the warp reuse every compressed-index load across all heads
        # and removes the CTA barriers from the quadratic block scan.  Keep the distributed
        # four-warp implementation below for larger HI/DI combinations whose stationary QI
        # values would otherwise require excessive registers in one warp.
        self.use_warp_rows = topk == 0 or (index_heads <= 4 and index_dim <= 128)
        assert index_heads > 0 and index_dim % 64 == 0 and topk >= 0
        assert gather_length >= topk + self.window and gather_length % 128 == 0

    @cute.jit
    def __call__(
        self,
        q_i: cute.Tensor,
        compressed_indices: cute.Tensor,
        weights: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        gather: cute.Tensor,
        stream: cuda.CUstream,
    ):
        if const_expr(self.use_warp_rows):
            self.kernel_warp_rows(q_i, compressed_indices, weights, cos, sin, gather).launch(
                grid=[cute.ceil_div(self.batch * self.sequence, 4), 1, 1],
                block=[self.num_threads, 1, 1],
                stream=stream,
            )
            return
        self.kernel(q_i, compressed_indices, weights, cos, sin, gather).launch(
            grid=[self.batch * self.sequence, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel_warp_rows(
        self,
        q_i: cute.Tensor,
        compressed_indices: cute.Tensor,
        weights: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        gather: cute.Tensor,
    ):
        """Scan one independent query row per warp for the small-HI indexer."""

        tid, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        warp = tid // cute.arch.WARP_SIZE
        lane = tid % cute.arch.WARP_SIZE
        row = block * 4 + warp

        if row < self.batch * self.sequence:
            batch_idx = row // self.sequence
            position = row - batch_idx * self.sequence

            # Each warp owns its gather row, so initialization and the appended local window
            # need neither shared memory nor CTA synchronization.
            for item in cutlass.range_constexpr(self.gather_length // cute.arch.WARP_SIZE):
                slot = lane + item * cute.arch.WARP_SIZE
                gather[batch_idx, position, slot] = Int32(-1)

            if const_expr(self.topk > 0):
                if const_expr(self.topk <= 64):
                    # Keep up to 64 descending elements distributed across the warp:
                    # lane L owns ranks L and L + 32. Every candidate is inserted with
                    # ballots and cooperative one-position shuffles instead of K serial
                    # comparisons and shifts in lane 0.
                    warp_best_keys = cute.make_rmem_tensor((2,), cutlass.Uint32)
                    warp_best_indices = cute.make_rmem_tensor((2,), Int32)
                    for k in cutlass.range_constexpr(2):
                        warp_best_keys[k] = cutlass.Uint32(0)
                        warp_best_indices[k] = Int32(-1)
                else:
                    serial_best_values = cute.make_rmem_tensor((self.topk,), Float32)
                    serial_best_indices = cute.make_rmem_tensor((self.topk,), Int32)
                    if lane == 0:
                        for k in cutlass.range_constexpr(self.topk):
                            serial_best_values[k] = -Float32.inf
                            serial_best_indices[k] = Int32(-1)

                # Keep the rounded, RoPE-transformed QI values stationary for the complete
                # causal scan.  This is at most 16 FP32 values/lane for HI=4, DI=128.
                q_values = cute.make_rmem_tensor(
                    (self.index_heads, self.pairs_per_lane, 2), Float32
                )
                for head in cutlass.range_constexpr(self.index_heads):
                    for item in cutlass.range_constexpr(self.pairs_per_lane):
                        pair = lane + item * cute.arch.WARP_SIZE
                        d0 = pair * 2
                        q0 = q_i[batch_idx, head, position, d0].to(Float32)
                        q1 = q_i[batch_idx, head, position, d0 + 1].to(Float32)
                        rq0, rq1 = q0, q1
                        if d0 >= self.index_dim - self.rope:
                            rope_pair = (d0 - (self.index_dim - self.rope)) // 2
                            c = cos[position, rope_pair].to(Float32)
                            s = sin[position, rope_pair].to(Float32)
                            rq0 = q0 * c - q1 * s
                            rq1 = q0 * s + q1 * c
                        q_values[head, item, 0] = rq0.to(self.dtype).to(Float32)
                        q_values[head, item, 1] = rq1.to(self.dtype).to(Float32)

                completed = (position + 1) // self.rate
                for n in cutlass.range(completed, unroll=1):
                    partials = cute.make_rmem_tensor((self.index_heads,), Float32)
                    for head in cutlass.range_constexpr(self.index_heads):
                        partials[head] = Float32(0.0)

                    # K_I is shared by every indexer head.  Load each pair once and reuse it
                    # for all heads instead of issuing one identical load from every warp.
                    for item in cutlass.range_constexpr(self.pairs_per_lane):
                        pair = lane + item * cute.arch.WARP_SIZE
                        d0 = pair * 2
                        k0 = compressed_indices[batch_idx, n, 0, d0].to(Float32)
                        k1 = compressed_indices[batch_idx, n, 0, d0 + 1].to(Float32)
                        for head in cutlass.range_constexpr(self.index_heads):
                            partials[head] += (
                                q_values[head, item, 0] * k0
                                + q_values[head, item, 1] * k1
                            )

                    score = Float32(0.0)
                    for head in cutlass.range_constexpr(self.index_heads):
                        dot = cute.arch.warp_reduction_sum(partials[head])
                        if lane == 0:
                            rounded_dot = dot.to(self.dtype).to(Float32)
                            activated = cute.arch.fmax(rounded_dot, Float32(0.0))
                            activated = (
                                activated / math.sqrt(self.index_dim * self.index_heads)
                            ).to(self.dtype).to(Float32)
                            score += (
                                activated * weights[batch_idx, position, head].to(Float32)
                            ).to(self.dtype).to(Float32)

                    if const_expr(self.topk <= 64):
                        warp_candidate_key = cutlass.Uint32(0)
                        warp_candidate_index = Int32(-1)
                        if lane == 0:
                            warp_candidate_index = Int32(n)
                            warp_candidate_key = _score_order_key(
                                score.to(self.dtype), warp_candidate_index, self.num_blocks
                            )
                        warp_candidate_key = cute.arch.shuffle_sync(
                            warp_candidate_key, offset=0
                        )
                        warp_candidate_index = cute.arch.shuffle_sync(
                            warp_candidate_index, offset=0
                        )
                        last_slot = 0 if const_expr(self.topk <= 32) else 1
                        last_lane = (self.topk - 1) % cute.arch.WARP_SIZE
                        last_key = cute.arch.shuffle_sync(
                            warp_best_keys[last_slot], offset=last_lane
                        )
                        last_index = cute.arch.shuffle_sync(
                            warp_best_indices[last_slot], offset=last_lane
                        )
                        # Once the set is full, almost all long-context candidates lose to
                        # rank 63. The predicate is warp-uniform, so skip every ballot and
                        # shift for those candidates without introducing divergence.
                        if warp_candidate_key > last_key or (
                            warp_candidate_key == last_key
                            and warp_candidate_index < last_index
                        ):
                            insertion_rank = cute.arch.popc(
                                cute.arch.vote_ballot_sync(
                                    warp_best_keys[0] > warp_candidate_key
                                    or (
                                        warp_best_keys[0] == warp_candidate_key
                                        and warp_best_indices[0] < warp_candidate_index
                                    )
                                    if lane < min(self.topk, cute.arch.WARP_SIZE)
                                    else False
                                )
                            ) + cute.arch.popc(
                                cute.arch.vote_ballot_sync(
                                    warp_best_keys[1] > warp_candidate_key
                                    or (
                                        warp_best_keys[1] == warp_candidate_key
                                        and warp_best_indices[1] < warp_candidate_index
                                    )
                                    if lane + cute.arch.WARP_SIZE < self.topk
                                    else False
                                )
                            )

                            previous_key_0 = cute.arch.shuffle_sync_up(
                                warp_best_keys[0], offset=1, mask_and_clamp=0
                            )
                            previous_index_0 = cute.arch.shuffle_sync_up(
                                warp_best_indices[0], offset=1, mask_and_clamp=0
                            )
                            previous_key_1 = cute.arch.shuffle_sync_up(
                                warp_best_keys[1], offset=1, mask_and_clamp=0
                            )
                            previous_index_1 = cute.arch.shuffle_sync_up(
                                warp_best_indices[1], offset=1, mask_and_clamp=0
                            )
                            previous_key_cross = cute.arch.shuffle_sync(
                                warp_best_keys[0], offset=31
                            )
                            previous_index_cross = cute.arch.shuffle_sync(
                                warp_best_indices[0], offset=31
                            )
                            if lane == 0:
                                previous_key_1 = previous_key_cross
                                previous_index_1 = previous_index_cross

                            rank_0 = lane
                            if rank_0 == insertion_rank:
                                warp_best_keys[0] = warp_candidate_key
                                warp_best_indices[0] = warp_candidate_index
                            elif rank_0 > insertion_rank and rank_0 < self.topk:
                                warp_best_keys[0] = previous_key_0
                                warp_best_indices[0] = previous_index_0

                            rank_1 = lane + cute.arch.WARP_SIZE
                            if rank_1 == insertion_rank:
                                warp_best_keys[1] = warp_candidate_key
                                warp_best_indices[1] = warp_candidate_index
                            elif rank_1 > insertion_rank and rank_1 < self.topk:
                                warp_best_keys[1] = previous_key_1
                                warp_best_indices[1] = previous_index_1
                    elif lane == 0:
                        serial_candidate_value = score.to(self.dtype).to(Float32)
                        serial_candidate_index = Int32(n)
                        for k in cutlass.range_constexpr(self.topk):
                            if serial_candidate_value > serial_best_values[k] or (
                                serial_candidate_value == serial_best_values[k]
                                and serial_candidate_index < serial_best_indices[k]
                            ):
                                old_value = serial_best_values[k]
                                old_index = serial_best_indices[k]
                                serial_best_values[k] = serial_candidate_value
                                serial_best_indices[k] = serial_candidate_index
                                serial_candidate_value = old_value
                                serial_candidate_index = old_index

                if const_expr(self.topk <= 64):
                    if lane < self.topk:
                        gather[batch_idx, position, lane] = warp_best_indices[0]
                    if lane + cute.arch.WARP_SIZE < self.topk:
                        gather[
                            batch_idx, position, lane + cute.arch.WARP_SIZE
                        ] = warp_best_indices[1]
                elif lane == 0:
                    for k in cutlass.range_constexpr(self.topk):
                        gather[batch_idx, position, k] = serial_best_indices[k]

            local_count = min(self.window, position + 1)
            for item in cutlass.range_constexpr(self.gather_length // cute.arch.WARP_SIZE):
                slot = lane + item * cute.arch.WARP_SIZE
                local_slot = slot - self.topk
                if local_slot >= 0 and local_slot < local_count:
                    key = position - local_count + 1 + local_slot
                    gather[batch_idx, position, slot] = Int32(self.num_blocks + key)

    @cute.kernel
    def kernel(
        self,
        q_i: cute.Tensor,
        compressed_indices: cute.Tensor,
        weights: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        gather: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        batch_idx = row // self.sequence
        position = row - batch_idx * self.sequence
        warp = tid // cute.arch.WARP_SIZE
        lane = tid % cute.arch.WARP_SIZE

        for item in cutlass.range_constexpr(self.gather_length // self.num_threads):
            gather[batch_idx, position, tid + item * self.num_threads] = Int32(-1)

        smem = cutlass.utils.SmemAllocator()
        contributions = smem.allocate_tensor(
            Float32, cute.make_layout((4,)), byte_alignment=16
        )
        score_sum = smem.allocate_tensor(Float32, cute.make_layout((1,)), byte_alignment=16)
        # CuTe rmem tensors cannot be empty, so retain one dead slot for local-only attention.
        best_values = cute.make_rmem_tensor((max(1, self.topk),), Float32)
        best_indices = cute.make_rmem_tensor((max(1, self.topk),), Int32)
        if tid == 0:
            for k in cutlass.range_constexpr(self.topk):
                best_values[k] = -Float32.inf
                best_indices[k] = Int32(-1)

        completed = (position + 1) // self.rate
        # QI is stationary across all compressed blocks. Rotate and round it once instead of
        # reloading and reapplying RoPE inside the block loop.
        q_values = cute.make_rmem_tensor(
            (self.head_groups, self.pairs_per_lane, 2), Float32
        )
        for group in cutlass.range_constexpr(self.head_groups):
            head = group * 4 + warp
            for item in cutlass.range_constexpr(self.pairs_per_lane):
                pair = lane + item * cute.arch.WARP_SIZE
                d0 = pair * 2
                rq0 = Float32(0.0)
                rq1 = Float32(0.0)
                if head < self.index_heads and d0 < self.index_dim:
                    q0 = q_i[batch_idx, head, position, d0].to(Float32)
                    q1 = q_i[batch_idx, head, position, d0 + 1].to(Float32)
                    rq0, rq1 = q0, q1
                    if d0 >= self.index_dim - self.rope:
                        rope_pair = (d0 - (self.index_dim - self.rope)) // 2
                        c = cos[position, rope_pair].to(Float32)
                        s = sin[position, rope_pair].to(Float32)
                        rq0 = q0 * c - q1 * s
                        rq1 = q0 * s + q1 * c
                    rq0 = rq0.to(self.dtype).to(Float32)
                    rq1 = rq1.to(self.dtype).to(Float32)
                q_values[group, item, 0] = rq0
                q_values[group, item, 1] = rq1

        # Future blocks are causally ineligible, so scanning them only burns work. Four warps
        # process index heads in groups; thread 0 maintains the small top-k set in registers.
        for n in cutlass.range(completed, unroll=1):
            if tid == 0:
                score_sum[0] = Float32(0.0)
            cute.arch.barrier()

            for group in cutlass.range_constexpr(self.head_groups):
                head = group * 4 + warp
                partial = Float32(0.0)
                if head < self.index_heads:
                    for item in cutlass.range_constexpr(self.pairs_per_lane):
                        pair = lane + item * cute.arch.WARP_SIZE
                        d0 = pair * 2
                        if d0 < self.index_dim:
                            rq0 = q_values[group, item, 0]
                            rq1 = q_values[group, item, 1]
                            k0 = compressed_indices[batch_idx, n, 0, d0].to(Float32)
                            k1 = compressed_indices[batch_idx, n, 0, d0 + 1].to(Float32)
                            partial += rq0 * k0 + rq1 * k1
                dot = cute.arch.warp_reduction_sum(partial)
                if lane == 0:
                    weighted = Float32(0.0)
                    if head < self.index_heads:
                        # Match the BF16/FP16 boundaries of matmul, divide, W multiply.
                        rounded_dot = dot.to(self.dtype).to(Float32)
                        activated = cute.arch.fmax(rounded_dot, Float32(0.0))
                        activated = (
                            activated / math.sqrt(self.index_dim * self.index_heads)
                        ).to(self.dtype).to(Float32)
                        weighted = (
                            activated * weights[batch_idx, position, head].to(Float32)
                        ).to(self.dtype).to(Float32)
                    contributions[warp] = weighted
                cute.arch.barrier()
                if tid == 0:
                    for group_head in cutlass.range_constexpr(4):
                        score_sum[0] += contributions[group_head]
                cute.arch.barrier()

            if tid == 0:
                score = score_sum[0].to(self.dtype).to(Float32)
                candidate_value = score
                candidate_index = Int32(n)
                for k in cutlass.range_constexpr(self.topk):
                    if candidate_value > best_values[k] or (
                        candidate_value == best_values[k]
                        and candidate_index < best_indices[k]
                    ):
                        old_value = best_values[k]
                        old_index = best_indices[k]
                        best_values[k] = candidate_value
                        best_indices[k] = candidate_index
                        candidate_value = old_value
                        candidate_index = old_index
            cute.arch.barrier()

        if tid == 0:
            for k in cutlass.range_constexpr(self.topk):
                gather[batch_idx, position, k] = best_indices[k]

        # Append causal local positions after the compressed selection. The combined attention
        # kernel consumes this one list, so local and compressed logits share a single softmax.
        local_count = min(self.window, position + 1)
        for item in cutlass.range_constexpr(self.gather_length // self.num_threads):
            slot = tid + item * self.num_threads
            local_slot = slot - self.topk
            if local_slot >= 0 and local_slot < local_count:
                key = position - local_count + 1 + local_slot
                gather[batch_idx, position, slot] = Int32(self.num_blocks + key)


class MergeSinkInverseRope:
    """Merge two normalized partial attentions with the learned sink and undo RoPE."""

    def __init__(
        self,
        dtype,
        batch: int,
        total_heads: int,
        local_tile_heads: int,
        compressed_tile_heads: int,
        active_heads: int,
        head_offset: int,
        sequence: int,
        dim: int,
        rope: int,
        has_local: bool,
        has_compressed: bool,
        store_lse: bool,
    ):
        self.dtype = dtype
        self.batch = batch
        self.total_heads = total_heads
        self.local_tile_heads = local_tile_heads
        self.compressed_tile_heads = compressed_tile_heads
        self.active_heads = active_heads
        self.head_offset = head_offset
        self.sequence = sequence
        self.dim = dim
        self.rope = rope
        self.has_local = has_local
        self.has_compressed = has_compressed
        self.store_lse = store_lse
        self.head_tile = math.gcd(active_heads, 32)
        self.num_threads = 256
        self.pairs_per_thread = self.head_tile * dim // 2 // self.num_threads
        assert local_tile_heads in (64, 128) and compressed_tile_heads == 128
        assert 0 < active_heads <= local_tile_heads
        assert 0 <= head_offset and head_offset + active_heads <= total_heads

    @cute.jit
    def __call__(
        self,
        local_out: cute.Tensor,
        local_lse: cute.Tensor,
        compressed_out: cute.Tensor,
        compressed_lse: cute.Tensor,
        sink: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        attention_lse: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        self.kernel(
            local_out,
            local_lse,
            compressed_out,
            compressed_lse,
            sink,
            cos,
            sin,
            out,
            attention_lse,
        ).launch(
            grid=[self.batch * self.sequence * (self.active_heads // self.head_tile), 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        local_out: cute.Tensor,
        local_lse: cute.Tensor,
        compressed_out: cute.Tensor,
        compressed_lse: cute.Tensor,
        sink: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        attention_lse: cute.Tensor | None,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        head_tiles = self.active_heads // self.head_tile
        head_tile_idx = row % head_tiles
        sequence_row = row // head_tiles
        position = sequence_row % self.sequence
        batch_idx = sequence_row // self.sequence
        pair_base = tid * 2
        tile_elements = self.head_tile * self.dim

        smem = cutlass.utils.SmemAllocator()
        merge_weights = smem.allocate_tensor(
            Float32,
            cute.make_layout((self.head_tile, 3)),
            byte_alignment=16,
        )
        if tid < self.head_tile:
            head = head_tile_idx * self.head_tile + tid
            lse_l = -Float32.inf
            lse_c = -Float32.inf
            if const_expr(self.has_local):
                lse_l = local_lse[batch_idx, position, head].to(Float32)
            if const_expr(self.has_compressed):
                lse_c = compressed_lse[batch_idx, position, head].to(Float32)
            sink_logit = sink[self.head_offset + head].to(Float32)
            maximum = cute.arch.fmax(cute.arch.fmax(lse_l, lse_c), sink_logit)
            weight_l = cute.math.exp(lse_l - maximum, fastmath=True)
            weight_c = cute.math.exp(lse_c - maximum, fastmath=True)
            weight_s = cute.math.exp(sink_logit - maximum, fastmath=True)
            merge_weights[tid, 0] = weight_l
            merge_weights[tid, 1] = weight_c
            merge_weights[tid, 2] = weight_l + weight_c + weight_s
            if const_expr(self.store_lse):
                assert attention_lse is not None
                attention_max = lse_l
                attention_sum = Float32(1.0)
                if const_expr(self.has_local and self.has_compressed):
                    attention_max = cute.arch.fmax(lse_l, lse_c)
                    attention_sum = cute.math.exp(
                        lse_l - attention_max, fastmath=True
                    ) + cute.math.exp(lse_c - attention_max, fastmath=True)
                elif const_expr(self.has_compressed):
                    attention_max = lse_c
                attention_lse[
                    batch_idx, position, self.head_offset + head
                ] = attention_max + cute.math.log(
                    Float32(attention_sum), fastmath=True
                )
        cute.arch.barrier()

        for j in cutlass.range_constexpr(self.pairs_per_thread):
            element = pair_base + j * self.num_threads * 2
            if element < tile_elements:
                head_local = element // self.dim
                d0 = element - head_local * self.dim
                head = head_tile_idx * self.head_tile + head_local
                global_head = self.head_offset + head
                weight_l = merge_weights[head_local, 0]
                weight_c = merge_weights[head_local, 1]
                x0 = Float32(0.0)
                x1 = Float32(0.0)
                if const_expr(self.has_local):
                    x0 += local_out[batch_idx, position, head, d0].to(Float32) * weight_l
                    x1 += local_out[batch_idx, position, head, d0 + 1].to(Float32) * weight_l
                if const_expr(self.has_compressed):
                    x0 += compressed_out[batch_idx, position, head, d0].to(Float32) * weight_c
                    x1 += (
                        compressed_out[batch_idx, position, head, d0 + 1].to(Float32)
                        * weight_c
                    )
                denominator = merge_weights[head_local, 2]
                x0 /= denominator
                x1 /= denominator
                # The reference rounds P@V before applying inverse RoPE.
                x0 = x0.to(self.dtype).to(Float32)
                x1 = x1.to(self.dtype).to(Float32)
                y0, y1 = x0, x1
                if d0 >= self.dim - self.rope:
                    rope_pair = (d0 - (self.dim - self.rope)) // 2
                    c = cos[position, rope_pair].to(Float32)
                    s = sin[position, rope_pair].to(Float32)
                    y0 = x0 * c + x1 * s
                    y1 = -x0 * s + x1 * c
                out[batch_idx, global_head, position, d0] = y0.to(self.dtype)
                out[batch_idx, global_head, position, d0 + 1] = y1.to(self.dtype)


class CompressedMergeSinkInverseRope:
    """Compute selected attention, sink normalization, and inverse RoPE directly.

    Eight heads share one 1024-thread CTA.  Each 128-thread subgroup owns one D=512
    query/output row, so QK reductions and PV updates never leave the CTA. The public wrapper
    uses this exact-order path when local attention is disabled.
    """

    def __init__(
        self,
        dtype,
        batch: int,
        total_heads: int,
        tile_heads: int,
        active_heads: int,
        head_offset: int,
        sequence: int,
        dim: int,
        rope: int,
        topk: int,
        has_local: bool,
    ):
        self.dtype = dtype
        self.batch = batch
        self.total_heads = total_heads
        self.tile_heads = tile_heads
        self.active_heads = active_heads
        self.head_offset = head_offset
        self.sequence = sequence
        self.dim = dim
        self.rope = rope
        self.head_tile = 8
        self.head_tiles = (active_heads + self.head_tile - 1) // self.head_tile
        self.threads_per_head = 128
        self.num_threads = self.head_tile * self.threads_per_head
        self.values_per_thread = dim // self.threads_per_head
        self.topk = topk
        self.has_local = has_local
        assert tile_heads == 128 and 0 < active_heads <= tile_heads
        assert 0 <= head_offset and head_offset + active_heads <= total_heads
        assert topk > 0
        assert dim == 512 and self.values_per_thread == 4

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        compressed_kv: cute.Tensor,
        gather: cute.Tensor,
        local_out: cute.Tensor,
        local_lse: cute.Tensor,
        sink: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            query, compressed_kv, gather, local_out, local_lse, sink, cos, sin, out
        ).launch(
            grid=[self.batch * self.sequence * self.head_tiles, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        query: cute.Tensor,
        compressed_kv: cute.Tensor,
        gather: cute.Tensor,
        local_out: cute.Tensor,
        local_lse: cute.Tensor,
        sink: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        head_tile_idx = row % self.head_tiles
        sequence_row = row // self.head_tiles
        position = sequence_row % self.sequence
        batch_idx = sequence_row // self.sequence
        head_local = tid // self.threads_per_head
        head_lane = tid - head_local * self.threads_per_head
        head = head_tile_idx * self.head_tile + head_local
        is_active = head < self.active_heads
        d_base = head_lane * self.values_per_thread

        query_values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        for j in cutlass.range_constexpr(self.values_per_thread):
            query_values[j] = query[batch_idx, position, head, d_base + j].to(Float32)

        smem = cutlass.utils.SmemAllocator()
        reduction = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (self.head_tile, (self.threads_per_head // cute.arch.WARP_SIZE, 1))
            ),
            byte_alignment=16,
        )
        scores = smem.allocate_tensor(
            Float32, cute.make_layout((self.head_tile, self.topk)), byte_alignment=16
        )

        for selected in cutlass.range_constexpr(self.topk):
            selected_block = gather[batch_idx, position, selected]
            partial = Float32(0.0)
            if selected_block >= 0:
                for j in cutlass.range_constexpr(self.values_per_thread):
                    partial += query_values[j] * compressed_kv[
                        batch_idx, selected_block, 0, d_base + j
                    ].to(Float32)
            dot = row_reduce(
                partial,
                cute.ReductionOp.ADD,
                self.threads_per_head,
                reduction,
                init_val=0.0,
            )
            if head_lane == 0:
                score = -Float32.inf
                if selected_block >= 0:
                    # QK and the scalar division are BF16/FP16 boundaries in the oracle.
                    score = (
                        dot.to(self.dtype).to(Float32) / math.sqrt(self.dim)
                    ).to(self.dtype).to(Float32)
                scores[head_local, selected] = score
            # Protect both the score publication and reduction scratch before its next reuse.
            cute.arch.barrier()

        lse_local = -Float32.inf
        if const_expr(self.has_local):
            if is_active:
                lse_local = local_lse[batch_idx, position, head].to(Float32)
        sink_logit = Float32(0.0)
        if is_active:
            sink_logit = sink[self.head_offset + head].to(Float32)
        maximum = cute.arch.fmax(lse_local, sink_logit)
        for selected in cutlass.range_constexpr(self.topk):
            maximum = cute.arch.fmax(maximum, scores[head_local, selected])
        local_delta = (lse_local - maximum).to(self.dtype).to(Float32)
        sink_delta = (sink_logit - maximum).to(self.dtype).to(Float32)
        local_weight = Float32(self.dtype(cute.math.exp(local_delta, fastmath=False)))
        sink_weight = Float32(self.dtype(cute.math.exp(sink_delta, fastmath=False)))
        weight_sum = local_weight
        compressed_weights = cute.make_rmem_tensor((self.topk,), Float32)
        for selected in cutlass.range_constexpr(self.topk):
            delta = (scores[head_local, selected] - maximum).to(self.dtype).to(Float32)
            weight = Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
            compressed_weights[selected] = weight
            weight_sum += weight
        denominator = Float32(self.dtype(Float32(self.dtype(weight_sum)) + sink_weight))

        values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        for j in cutlass.range_constexpr(self.values_per_thread):
            value = Float32(0.0)
            if const_expr(self.has_local):
                if is_active:
                    local_probability = Float32(self.dtype(local_weight / denominator))
                    value = (
                        local_out[batch_idx, position, head, d_base + j].to(Float32)
                        * local_probability
                    )
            for selected in cutlass.range_constexpr(self.topk):
                selected_block = gather[batch_idx, position, selected]
                if selected_block >= 0:
                    probability = Float32(
                        self.dtype(compressed_weights[selected] / denominator)
                    )
                    value += probability * compressed_kv[
                        batch_idx, selected_block, 0, d_base + j
                    ].to(Float32)
            # P@V is rounded before inverse RoPE in the reference implementation.
            values[j] = value.to(self.dtype).to(Float32)

        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j0 = pair * 2
            d0 = d_base + j0
            x0, x1 = values[j0], values[j0 + 1]
            y0, y1 = x0, x1
            if d0 >= self.dim - self.rope:
                rope_pair = (d0 - (self.dim - self.rope)) // 2
                c = cos[position, rope_pair].to(Float32)
                s = sin[position, rope_pair].to(Float32)
                y0 = x0 * c + x1 * s
                y1 = -x0 * s + x1 * c
            if is_active:
                global_head = self.head_offset + head
                out[batch_idx, global_head, position, d0] = y0.to(self.dtype)
                out[batch_idx, global_head, position, d0 + 1] = y1.to(self.dtype)


class PrefixAttention:
    """Overwrite the numerically sharp startup rows with oracle-ordered BF16 math."""

    def __init__(
        self,
        dtype,
        batch: int,
        total_heads: int,
        tile_heads: int,
        active_heads: int,
        head_offset: int,
        dim: int,
        rope: int,
        prefix: int,
        raw_query: bool = False,
        store_lse: bool = False,
    ):
        self.dtype = dtype
        self.batch = batch
        self.total_heads = total_heads
        self.tile_heads = tile_heads
        self.active_heads = active_heads
        self.head_offset = head_offset
        self.dim = dim
        self.rope = rope
        self.prefix = prefix
        self.raw_query = raw_query
        self.store_lse = store_lse
        self.num_threads = 128
        self.values_per_thread = dim // self.num_threads
        assert dim == 512 and prefix > 0
        assert tile_heads in (64, 128)
        assert 0 < active_heads <= tile_heads
        assert 0 <= head_offset and head_offset + active_heads <= total_heads

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        local_kv: cute.Tensor,
        sink: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        self.kernel(query, local_kv, sink, cos, sin, out, lse).launch(
            grid=[self.batch * self.active_heads * self.prefix, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        query: cute.Tensor,
        local_kv: cute.Tensor,
        sink: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor | None,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        position = row % self.prefix
        head_row = row // self.prefix
        head = head_row % self.active_heads
        batch_idx = head_row // self.active_heads
        global_head = self.head_offset + head
        d_base = tid * self.values_per_thread

        query_values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        if const_expr(self.raw_query):
            for pair in cutlass.range_constexpr(self.values_per_thread // 2):
                j0 = pair * 2
                d0 = d_base + j0
                x0 = query[batch_idx, global_head, position, d0].to(Float32)
                x1 = query[batch_idx, global_head, position, d0 + 1].to(Float32)
                y0, y1 = x0, x1
                if d0 >= self.dim - self.rope:
                    rope_pair = (d0 - (self.dim - self.rope)) // 2
                    c = cos[position, rope_pair].to(Float32)
                    s = sin[position, rope_pair].to(Float32)
                    y0 = x0 * c - x1 * s
                    y1 = x0 * s + x1 * c
                # Match QueryRopeTranspose's output boundary before the QK dot.
                query_values[j0] = y0.to(self.dtype).to(Float32)
                query_values[j0 + 1] = y1.to(self.dtype).to(Float32)
        else:
            for j in cutlass.range_constexpr(self.values_per_thread):
                query_values[j] = query[batch_idx, position, head, d_base + j].to(Float32)

        smem = cutlass.utils.SmemAllocator()
        reduction = smem.allocate_tensor(
            Float32, cute.make_layout((1, (self.num_threads // 32, 1))), byte_alignment=16
        )
        scores = smem.allocate_tensor(Float32, cute.make_layout((self.prefix,)), byte_alignment=16)
        for key in cutlass.range(self.prefix, unroll=1):
            partial = Float32(0.0)
            if key <= position:
                for j in cutlass.range_constexpr(self.values_per_thread):
                    partial += query_values[j] * local_kv[
                        batch_idx, key, 0, d_base + j
                    ].to(Float32)
            dot = row_reduce(
                partial,
                cute.ReductionOp.ADD,
                self.num_threads,
                reduction,
                init_val=0.0,
            )
            if tid == 0:
                score = -Float32.inf
                if key <= position:
                    score = (
                        dot.to(self.dtype).to(Float32) / math.sqrt(self.dim)
                    ).to(self.dtype).to(Float32)
                scores[key] = score
            cute.arch.barrier()

        maximum = sink[global_head].to(Float32)
        for key in cutlass.range(self.prefix, unroll=1):
            maximum = cute.arch.fmax(maximum, scores[key])
        weight_sum = Float32(0.0)
        for key in cutlass.range(self.prefix, unroll=1):
            delta = (scores[key] - maximum).to(self.dtype).to(Float32)
            weight = Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
            weight_sum += weight
        sink_delta = (sink[global_head].to(Float32) - maximum).to(self.dtype).to(Float32)
        sink_weight = Float32(self.dtype(cute.math.exp(sink_delta, fastmath=False)))
        denominator = Float32(self.dtype(Float32(self.dtype(weight_sum)) + sink_weight))
        if const_expr(self.store_lse):
            if tid == 0:
                kv_maximum = -Float32.inf
                for key in cutlass.range(self.prefix, unroll=1):
                    kv_maximum = cute.arch.fmax(kv_maximum, scores[key])
                kv_weight_sum = Float32(0.0)
                for key in cutlass.range(self.prefix, unroll=1):
                    kv_weight_sum += cute.math.exp(
                        scores[key] - kv_maximum, fastmath=False
                    )
                lse[batch_idx, position, global_head] = kv_maximum + cute.math.log(
                    kv_weight_sum, fastmath=False
                )

        values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        for j in cutlass.range_constexpr(self.values_per_thread):
            value = Float32(0.0)
            for key in cutlass.range(self.prefix, unroll=1):
                delta = (scores[key] - maximum).to(self.dtype).to(Float32)
                weight = Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
                probability = Float32(self.dtype(weight / denominator))
                if key <= position:
                    value += probability * local_kv[batch_idx, key, 0, d_base + j].to(Float32)
            values[j] = Float32(self.dtype(value))

        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j0 = pair * 2
            d0 = d_base + j0
            x0, x1 = values[j0], values[j0 + 1]
            y0, y1 = x0, x1
            if d0 >= self.dim - self.rope:
                rope_pair = (d0 - (self.dim - self.rope)) // 2
                c = cos[position, rope_pair].to(Float32)
                s = sin[position, rope_pair].to(Float32)
                y0 = x0 * c + x1 * s
                y1 = -x0 * s + x1 * c
            out[batch_idx, global_head, position, d0] = y0.to(self.dtype)
            out[batch_idx, global_head, position, d0 + 1] = y1.to(self.dtype)


def _fake(dtype, shape):
    return make_fake_tensor(dtype, shape, math.gcd(8, shape[-1]))


@lru_cache
def compile_compression(dtype, batch, sequence, dim, rate, rope):
    num_blocks = (sequence + rate - 1) // rate
    tensors = [
        _fake(dtype, (batch, 1, sequence, dim)) for _ in range(4)
    ]
    biases = [_fake(dtype, (rate, dim)) for _ in range(2)]
    weight = _fake(dtype, (dim,))
    cos = _fake(Float32, (sequence, rope // 2))
    sin = _fake(Float32, (sequence, rope // 2))
    out = _fake(dtype, (batch, num_blocks, 1, dim))
    return cute.compile(
        CompressionNormRope(dtype, batch, sequence, dim, rate, rope),
        *tensors,
        *biases,
        weight,
        cos,
        sin,
        out,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_local_norm(dtype, batch, sequence, dim, rope, pad):
    return cute.compile(
        LocalNormRope(dtype, batch, sequence, dim, rope, pad),
        _fake(dtype, (batch, 1, sequence, dim)),
        _fake(dtype, (dim,)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(dtype, (batch, sequence + pad, 1, dim)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_query_rope(
    dtype,
    batch,
    source_heads,
    tile_heads,
    active_heads,
    head_offset,
    sequence,
    dim,
    rope,
    mirror_heads=0,
):
    return cute.compile(
        QueryRopeTranspose(
            dtype,
            batch,
            source_heads,
            tile_heads,
            active_heads,
            head_offset,
            sequence,
            dim,
            rope,
            mirror_heads,
        ),
        _fake(dtype, (batch, source_heads, sequence, dim)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(dtype, (batch, sequence, tile_heads, dim)),
        _fake(dtype, (batch, sequence, mirror_heads or tile_heads, dim)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_index_scores(
    dtype,
    batch,
    sequence,
    index_heads,
    index_dim,
    num_blocks,
    rate,
    rope,
    row_chunk,
):
    return cute.compile(
        IndexScores(
            dtype,
            batch,
            sequence,
            index_heads,
            index_dim,
            num_blocks,
            rate,
            rope,
            row_chunk,
        ),
        _fake(dtype, (batch, index_heads, sequence, index_dim)),
        _fake(dtype, (batch, num_blocks, 1, index_dim)),
        _fake(dtype, (batch, sequence, index_heads)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (row_chunk, num_blocks)),
        _fake(Int32, (row_chunk,)),
        Int32(0),
        Int32(row_chunk),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_selected_gather(
    batch, sequence, num_blocks, topk, window, gather_length, row_chunk
):
    return cute.compile(
        SelectedGather(
            batch, sequence, num_blocks, topk, window, gather_length, row_chunk
        ),
        _fake(Int32, (row_chunk, topk)),
        _fake(Int32, (batch, sequence, gather_length)),
        Int32(0),
        Int32(row_chunk),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_causal_gather(
    batch, sequence, num_blocks, rate, topk, window, gather_length
):
    return cute.compile(
        CausalGather(
            batch, sequence, num_blocks, rate, topk, window, gather_length
        ),
        _fake(Int32, (batch, sequence, gather_length)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_index_topk(
    dtype,
    batch,
    sequence,
    index_heads,
    index_dim,
    num_blocks,
    rate,
    topk,
    window,
    rope,
    gather_length,
):
    return cute.compile(
        IndexTopK(
            dtype,
            batch,
            sequence,
            index_heads,
            index_dim,
            num_blocks,
            rate,
            topk,
            window,
            rope,
            gather_length,
        ),
        _fake(dtype, (batch, index_heads, sequence, index_dim)),
        _fake(dtype, (batch, num_blocks, 1, index_dim)),
        _fake(dtype, (batch, sequence, index_heads)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Int32, (batch, sequence, gather_length)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_merge(
    dtype,
    batch,
    total_heads,
    local_tile_heads,
    compressed_tile_heads,
    active_heads,
    head_offset,
    sequence,
    dim,
    rope,
    has_local,
    has_compressed,
    store_lse,
):
    local_shape = (batch, sequence, local_tile_heads, dim)
    local_lse_shape = (batch, sequence, local_tile_heads)
    compressed_shape = (batch, sequence, compressed_tile_heads, dim)
    compressed_lse_shape = (batch, sequence, compressed_tile_heads)
    return cute.compile(
        MergeSinkInverseRope(
            dtype,
            batch,
            total_heads,
            local_tile_heads,
            compressed_tile_heads,
            active_heads,
            head_offset,
            sequence,
            dim,
            rope,
            has_local,
            has_compressed,
            store_lse,
        ),
        _fake(dtype, local_shape),
        _fake(Float32, local_lse_shape),
        _fake(dtype, compressed_shape),
        _fake(Float32, compressed_lse_shape),
        _fake(dtype, (total_heads,)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(dtype, (batch, total_heads, sequence, dim)),
        _fake(Float32, (batch, sequence, total_heads)) if store_lse else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_compressed_merge(
    dtype,
    batch,
    total_heads,
    tile_heads,
    active_heads,
    head_offset,
    sequence,
    dim,
    rope,
    num_blocks,
    topk,
    gather_length,
    has_local=False,
):
    query_shape = (batch, sequence, tile_heads, dim)
    return cute.compile(
        CompressedMergeSinkInverseRope(
            dtype,
            batch,
            total_heads,
            tile_heads,
            active_heads,
            head_offset,
            sequence,
            dim,
            rope,
            topk,
            has_local,
        ),
        _fake(dtype, query_shape),
        _fake(dtype, (batch, num_blocks, 1, dim)),
        _fake(Int32, (batch, sequence, gather_length)),
        _fake(dtype, query_shape),
        _fake(Float32, (batch, sequence, tile_heads)),
        _fake(dtype, (total_heads,)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(dtype, (batch, total_heads, sequence, dim)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@lru_cache
def compile_prefix(
    dtype,
    batch,
    total_heads,
    tile_heads,
    active_heads,
    head_offset,
    sequence,
    dim,
    rope,
    prefix=16,
    raw_query=False,
    store_lse=False,
):
    return cute.compile(
        PrefixAttention(
            dtype,
            batch,
            total_heads,
            tile_heads,
            active_heads,
            head_offset,
            dim,
            rope,
            prefix,
            raw_query,
            store_lse,
        ),
        _fake(
            dtype,
            (batch, total_heads, sequence, dim)
            if raw_query
            else (batch, prefix, tile_heads, dim),
        ),
        _fake(dtype, (batch, sequence, 1, dim)),
        _fake(dtype, (total_heads,)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(dtype, (batch, total_heads, sequence, dim)),
        _fake(Float32, (batch, sequence, total_heads)) if store_lse else None,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def cute_dtype(tensor: torch.Tensor):
    return torch2cute_dtype_map[tensor.dtype]


__all__ = [
    "compile_compression",
    "compile_compressed_merge",
    "compile_index_scores",
    "compile_index_topk",
    "compile_local_norm",
    "compile_merge",
    "compile_prefix",
    "compile_query_rope",
    "compile_selected_gather",
    "cute_dtype",
]
