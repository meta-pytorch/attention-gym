"""SM100 CuTe DSL backward kernels for compressed sparse attention."""

from __future__ import annotations

from functools import lru_cache
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map
from quack.reduce import row_reduce


_RMS_EPS = 1.1920928955078125e-7


class AttentionBackward:
    """Differentiate fixed-selection shared-KV CSA attention.

    One 128-thread CTA owns one ``(batch, head, query)`` row. Shared KV gradients and
    sink gradients accumulate into FP32 buffers; query gradients have a unique writer.
    """

    def __init__(
        self,
        dtype,
        batch: int,
        heads: int,
        sequence: int,
        dim: int,
        rope: int,
        num_blocks: int,
        topk: int,
        gather_length: int,
        window: int,
    ):
        self.dtype = dtype
        self.batch = batch
        self.heads = heads
        self.sequence = sequence
        self.dim = dim
        self.rope = rope
        self.num_blocks = num_blocks
        self.topk = topk
        self.gather_length = gather_length
        self.window = min(window, sequence)
        self.has_local = window > 0
        self.has_compressed = topk > 0
        self.num_threads = 128
        self.values_per_thread = dim // self.num_threads
        assert dim == 512 and self.values_per_thread == 4

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        local_kv: cute.Tensor,
        compressed_kv: cute.Tensor,
        gather: cute.Tensor,
        sink: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        dout: cute.Tensor,
        dq: cute.Tensor,
        dlocal: cute.Tensor,
        dcompressed: cute.Tensor,
        dsink: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(
            q,
            local_kv,
            compressed_kv,
            gather,
            sink,
            cos,
            sin,
            dout,
            dq,
            dlocal,
            dcompressed,
            dsink,
        ).launch(
            grid=[self.batch * self.heads * self.sequence, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        q: cute.Tensor,
        local_kv: cute.Tensor,
        compressed_kv: cute.Tensor,
        gather: cute.Tensor,
        sink: cute.Tensor,
        cos: cute.Tensor,
        sin: cute.Tensor,
        dout: cute.Tensor,
        dq: cute.Tensor,
        dlocal: cute.Tensor,
        dcompressed: cute.Tensor,
        dsink: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        position = row % self.sequence
        head_row = row // self.sequence
        head = head_row % self.heads
        batch_idx = head_row // self.heads
        d_base = tid * self.values_per_thread

        q_values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        do_values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        dq_values = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j0 = pair * 2
            d0 = d_base + j0
            q0 = q[batch_idx, head, position, d0].to(Float32)
            q1 = q[batch_idx, head, position, d0 + 1].to(Float32)
            do0 = dout[batch_idx, head, position, d0].to(Float32)
            do1 = dout[batch_idx, head, position, d0 + 1].to(Float32)
            rq0, rq1 = q0, q1
            rdo0, rdo1 = do0, do1
            if d0 >= self.dim - self.rope:
                rope_pair = (d0 - (self.dim - self.rope)) // 2
                c = cos[position, rope_pair].to(Float32)
                s = sin[position, rope_pair].to(Float32)
                rq0 = q0 * c - q1 * s
                rq1 = q0 * s + q1 * c
                rdo0 = do0 * c - do1 * s
                rdo1 = do0 * s + do1 * c
            q_values[j0] = rq0.to(self.dtype).to(Float32)
            q_values[j0 + 1] = rq1.to(self.dtype).to(Float32)
            do_values[j0] = rdo0
            do_values[j0 + 1] = rdo1
            dq_values[j0] = Float32(0.0)
            dq_values[j0 + 1] = Float32(0.0)

        smem = cutlass.utils.SmemAllocator()
        reduction = smem.allocate_tensor(
            Float32,
            cute.make_layout((1, (self.num_threads // cute.arch.WARP_SIZE, 1))),
            byte_alignment=16,
        )
        scalars = smem.allocate_tensor(Float32, cute.make_layout((5,)), byte_alignment=16)
        if tid == 0:
            scalars[0] = sink[head].to(Float32)
        cute.arch.barrier()

        scale = 1.0 / math.sqrt(self.dim)
        # Pass 1: global maximum across local keys, selected keys, and the learned sink.
        if const_expr(self.has_local):
            for offset in cutlass.range(self.window, unroll=1):
                key = position - offset
                partial = Float32(0.0)
                if key >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        partial += q_values[j] * local_kv[
                            batch_idx, key, 0, d_base + j
                        ].to(Float32)
                dot = row_reduce(
                    partial,
                    cute.ReductionOp.ADD,
                    self.num_threads,
                    reduction,
                    init_val=0.0,
                )
                if tid == 0 and key >= 0:
                    score = (dot.to(self.dtype).to(Float32) * scale).to(self.dtype).to(Float32)
                    scalars[0] = cute.arch.fmax(scalars[0], score)
                cute.arch.barrier()
        if const_expr(self.has_compressed):
            for selected in cutlass.range_constexpr(self.topk):
                block = gather[batch_idx, position, selected]
                partial = Float32(0.0)
                if block >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        partial += q_values[j] * compressed_kv[
                            batch_idx, block, 0, d_base + j
                        ].to(Float32)
                dot = row_reduce(
                    partial,
                    cute.ReductionOp.ADD,
                    self.num_threads,
                    reduction,
                    init_val=0.0,
                )
                if tid == 0 and block >= 0:
                    score = (dot.to(self.dtype).to(Float32) * scale).to(self.dtype).to(Float32)
                    scalars[0] = cute.arch.fmax(scalars[0], score)
                cute.arch.barrier()

        # Pass 2: softmax denominator in the same logical key order as the reference.
        if tid == 0:
            sink_delta = (sink[head].to(Float32) - scalars[0]).to(self.dtype).to(Float32)
            scalars[1] = Float32(self.dtype(cute.math.exp(sink_delta, fastmath=False)))
        cute.arch.barrier()
        if const_expr(self.has_compressed):
            for selected in cutlass.range_constexpr(self.topk):
                block = gather[batch_idx, position, selected]
                partial = Float32(0.0)
                if block >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        partial += q_values[j] * compressed_kv[
                            batch_idx, block, 0, d_base + j
                        ].to(Float32)
                dot = row_reduce(
                    partial,
                    cute.ReductionOp.ADD,
                    self.num_threads,
                    reduction,
                    init_val=0.0,
                )
                if tid == 0 and block >= 0:
                    score = (dot.to(self.dtype).to(Float32) * scale).to(self.dtype).to(Float32)
                    delta = (score - scalars[0]).to(self.dtype).to(Float32)
                    scalars[1] += Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
                cute.arch.barrier()
        if const_expr(self.has_local):
            for offset in cutlass.range(self.window, unroll=1):
                key = position - offset
                partial = Float32(0.0)
                if key >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        partial += q_values[j] * local_kv[
                            batch_idx, key, 0, d_base + j
                        ].to(Float32)
                dot = row_reduce(
                    partial,
                    cute.ReductionOp.ADD,
                    self.num_threads,
                    reduction,
                    init_val=0.0,
                )
                if tid == 0 and key >= 0:
                    score = (dot.to(self.dtype).to(Float32) * scale).to(self.dtype).to(Float32)
                    delta = (score - scalars[0]).to(self.dtype).to(Float32)
                    scalars[1] += Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
                cute.arch.barrier()

        # Recompute sum_j p_j * <dO, V_j> directly.  Recovering it as <dO, O>
        # amplifies the forward output's final BF16/FP16 rounding in dSink.
        if tid == 0:
            scalars[3] = Float32(0.0)
        cute.arch.barrier()
        if const_expr(self.has_compressed):
            for selected in cutlass.range_constexpr(self.topk):
                block = gather[batch_idx, position, selected]
                qk_partial = Float32(0.0)
                dp_partial = Float32(0.0)
                if block >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        value = compressed_kv[batch_idx, block, 0, d_base + j].to(Float32)
                        qk_partial += q_values[j] * value
                        dp_partial += do_values[j] * value
                qk = row_reduce(qk_partial, cute.ReductionOp.ADD, self.num_threads, reduction, init_val=0.0)
                dp = row_reduce(dp_partial, cute.ReductionOp.ADD, self.num_threads, reduction, init_val=0.0)
                if tid == 0 and block >= 0:
                    score = (qk.to(self.dtype).to(Float32) * scale).to(self.dtype).to(Float32)
                    delta = (score - scalars[0]).to(self.dtype).to(Float32)
                    weight = Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
                    probability = Float32(self.dtype(weight / scalars[1]))
                    scalars[3] += probability * dp
                cute.arch.barrier()
        if const_expr(self.has_local):
            for offset in cutlass.range(self.window, unroll=1):
                key = position - offset
                qk_partial = Float32(0.0)
                dp_partial = Float32(0.0)
                if key >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        value = local_kv[batch_idx, key, 0, d_base + j].to(Float32)
                        qk_partial += q_values[j] * value
                        dp_partial += do_values[j] * value
                qk = row_reduce(qk_partial, cute.ReductionOp.ADD, self.num_threads, reduction, init_val=0.0)
                dp = row_reduce(dp_partial, cute.ReductionOp.ADD, self.num_threads, reduction, init_val=0.0)
                if tid == 0 and key >= 0:
                    score = (qk.to(self.dtype).to(Float32) * scale).to(self.dtype).to(Float32)
                    delta = (score - scalars[0]).to(self.dtype).to(Float32)
                    weight = Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
                    probability = Float32(self.dtype(weight / scalars[1]))
                    scalars[3] += probability * dp
                cute.arch.barrier()

        # Pass 3: dQ plus atomic shared-KV accumulation.
        if const_expr(self.has_local):
            for offset in cutlass.range(self.window, unroll=1):
                key = position - offset
                qk_partial = Float32(0.0)
                dp_partial = Float32(0.0)
                if key >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        kv_value = local_kv[batch_idx, key, 0, d_base + j].to(Float32)
                        qk_partial += q_values[j] * kv_value
                        dp_partial += do_values[j] * kv_value
                qk = row_reduce(
                    qk_partial,
                    cute.ReductionOp.ADD,
                    self.num_threads,
                    reduction,
                    init_val=0.0,
                )
                dp = row_reduce(
                    dp_partial,
                    cute.ReductionOp.ADD,
                    self.num_threads,
                    reduction,
                    init_val=0.0,
                )
                if tid == 0:
                    probability = Float32(0.0)
                    dscore = Float32(0.0)
                    if key >= 0:
                        score = (qk.to(self.dtype).to(Float32) * scale).to(self.dtype).to(Float32)
                        delta = (score - scalars[0]).to(self.dtype).to(Float32)
                        weight = Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
                        probability = Float32(self.dtype(weight / scalars[1]))
                        dscore = probability * (dp - scalars[3])
                    scalars[2] = dscore
                    scalars[4] = probability
                cute.arch.barrier()
                if key >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        d = d_base + j
                        kv_value = local_kv[batch_idx, key, 0, d].to(Float32)
                        dq_values[j] += scalars[2] * kv_value * scale
                        gradient = scalars[4] * do_values[j] + scalars[2] * q_values[j] * scale
                        linear = (batch_idx * self.sequence + key) * self.dim + d
                        cute.arch.atomic_add(dlocal.iterator + linear, gradient)
                cute.arch.barrier()
        if const_expr(self.has_compressed):
            for selected in cutlass.range_constexpr(self.topk):
                block = gather[batch_idx, position, selected]
                qk_partial = Float32(0.0)
                dp_partial = Float32(0.0)
                if block >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        kv_value = compressed_kv[batch_idx, block, 0, d_base + j].to(Float32)
                        qk_partial += q_values[j] * kv_value
                        dp_partial += do_values[j] * kv_value
                qk = row_reduce(
                    qk_partial,
                    cute.ReductionOp.ADD,
                    self.num_threads,
                    reduction,
                    init_val=0.0,
                )
                dp = row_reduce(
                    dp_partial,
                    cute.ReductionOp.ADD,
                    self.num_threads,
                    reduction,
                    init_val=0.0,
                )
                if tid == 0:
                    probability = Float32(0.0)
                    dscore = Float32(0.0)
                    if block >= 0:
                        score = (qk.to(self.dtype).to(Float32) * scale).to(self.dtype).to(Float32)
                        delta = (score - scalars[0]).to(self.dtype).to(Float32)
                        weight = Float32(self.dtype(cute.math.exp(delta, fastmath=False)))
                        probability = Float32(self.dtype(weight / scalars[1]))
                        dscore = probability * (dp - scalars[3])
                    scalars[2] = dscore
                    scalars[4] = probability
                cute.arch.barrier()
                if block >= 0:
                    for j in cutlass.range_constexpr(self.values_per_thread):
                        d = d_base + j
                        kv_value = compressed_kv[batch_idx, block, 0, d].to(Float32)
                        dq_values[j] += scalars[2] * kv_value * scale
                        gradient = scalars[4] * do_values[j] + scalars[2] * q_values[j] * scale
                        linear = (batch_idx * self.num_blocks + block) * self.dim + d
                        cute.arch.atomic_add(dcompressed.iterator + linear, gradient)
                cute.arch.barrier()

        if tid == 0:
            sink_delta = (sink[head].to(Float32) - scalars[0]).to(self.dtype).to(Float32)
            sink_weight = Float32(self.dtype(cute.math.exp(sink_delta, fastmath=False)))
            sink_probability = Float32(self.dtype(sink_weight / scalars[1]))
            dsink[batch_idx, head, position] = -sink_probability * scalars[3]

        # Gradient through Q's forward RoPE is the inverse rotation.
        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j0 = pair * 2
            d0 = d_base + j0
            x0 = dq_values[j0]
            x1 = dq_values[j0 + 1]
            y0, y1 = x0, x1
            if d0 >= self.dim - self.rope:
                rope_pair = (d0 - (self.dim - self.rope)) // 2
                c = cos[position, rope_pair].to(Float32)
                s = sin[position, rope_pair].to(Float32)
                y0 = x0 * c + x1 * s
                y1 = -x0 * s + x1 * c
            dq[batch_idx, head, position, d0] = y0.to(self.dtype)
            dq[batch_idx, head, position, d0 + 1] = y1.to(self.dtype)


def _fake(dtype, shape):
    return make_fake_tensor(dtype, shape, math.gcd(8, shape[-1]))


@lru_cache
def compile_attention_backward(
    dtype,
    batch,
    heads,
    sequence,
    dim,
    rope,
    num_blocks,
    topk,
    gather_length,
    window,
):
    return cute.compile(
        AttentionBackward(
            dtype,
            batch,
            heads,
            sequence,
            dim,
            rope,
            num_blocks,
            topk,
            gather_length,
            window,
        ),
        _fake(dtype, (batch, heads, sequence, dim)),
        _fake(dtype, (batch, sequence, 1, dim)),
        _fake(dtype, (batch, num_blocks, 1, dim)),
        _fake(Int32, (batch, sequence, max(gather_length, 1))),
        _fake(dtype, (heads,)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (sequence, rope // 2)),
        _fake(dtype, (batch, heads, sequence, dim)),
        _fake(dtype, (batch, heads, sequence, dim)),
        _fake(Float32, (batch, sequence, 1, dim)),
        _fake(Float32, (batch, num_blocks, 1, dim)),
        _fake(Float32, (batch, heads, sequence)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def cute_dtype(tensor):
    return torch2cute_dtype_map[tensor.dtype]


class LocalNormBackward:
    """Differentiate the local shared-KV RMSNorm and tail RoPE."""

    def __init__(self, dtype, batch, sequence, dim, rope):
        self.dtype = dtype
        self.batch = batch
        self.sequence = sequence
        self.dim = dim
        self.rope = rope
        self.num_threads = 128
        self.values_per_thread = dim // self.num_threads

    @cute.jit
    def __call__(self, kv, weight, cos, sin, dy, dx, dweight, stream: cuda.CUstream):
        self.kernel(kv, weight, cos, sin, dy, dx, dweight).launch(
            grid=[self.batch * self.sequence, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, kv, weight, cos, sin, dy, dx, dweight):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        b = row // self.sequence
        pos = row - b * self.sequence
        base = tid * self.values_per_thread
        xs = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        dys = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        local_sq = Float32(0.0)
        local_dot = Float32(0.0)
        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j = pair * 2
            d = base + j
            x0 = kv[b, 0, pos, d].to(Float32)
            x1 = kv[b, 0, pos, d + 1].to(Float32)
            g0 = dy[b, pos, 0, d]
            g1 = dy[b, pos, 0, d + 1]
            if d >= self.dim - self.rope:
                rp = (d - (self.dim - self.rope)) // 2
                c = cos[pos, rp]
                s = sin[pos, rp]
                u0 = g0 * c + g1 * s
                u1 = -g0 * s + g1 * c
                g0, g1 = u0, u1
            xs[j], xs[j + 1] = x0, x1
            dys[j], dys[j + 1] = g0, g1
            local_sq += x0 * x0 + x1 * x1
            local_dot += g0 * weight[d].to(Float32) * x0
            local_dot += g1 * weight[d + 1].to(Float32) * x1
        smem = cutlass.utils.SmemAllocator()
        reduction = smem.allocate_tensor(
            Float32,
            cute.make_layout((1, (self.num_threads // cute.arch.WARP_SIZE, 1))),
            byte_alignment=16,
        )
        sum_sq = row_reduce(local_sq, cute.ReductionOp.ADD, self.num_threads, reduction, init_val=0.0)
        dot = row_reduce(local_dot, cute.ReductionOp.ADD, self.num_threads, reduction, init_val=0.0)
        rstd = cute.math.rsqrt(sum_sq / self.dim + _RMS_EPS, fastmath=True)
        correction = dot * rstd * rstd / self.dim
        for j in cutlass.range_constexpr(self.values_per_thread):
            d = base + j
            x = xs[j]
            g = dys[j]
            dx[b, 0, pos, d] = (rstd * (g * weight[d].to(Float32) - x * correction)).to(self.dtype)
            cute.arch.atomic_add(dweight.iterator + d, g * x * rstd)


class CompressionBackward:
    """Differentiate compression, RMSNorm, and compressed-position RoPE."""

    def __init__(self, dtype, batch, sequence, dim, rate, rope):
        self.dtype = dtype
        self.batch = batch
        self.sequence = sequence
        self.dim = dim
        self.rate = rate
        self.rope = rope
        self.blocks = (sequence + rate - 1) // rate
        self.num_threads = 128
        self.values_per_thread = dim // self.num_threads

    @cute.jit
    def __call__(
        self, ca, cb, za, zb, ba, bb, weight, cos, sin, dy,
        dca, dcb, dza, dzb, dba, dbb, dweight, stream: cuda.CUstream,
    ):
        self.kernel(
            ca, cb, za, zb, ba, bb, weight, cos, sin, dy,
            dca, dcb, dza, dzb, dba, dbb, dweight,
        ).launch(
            grid=[self.batch * self.blocks, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, ca, cb, za, zb, ba, bb, weight, cos, sin, dy,
        dca, dcb, dza, dzb, dba, dbb, dweight,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        b = row // self.blocks
        block = row - b * self.blocks
        base = tid * self.values_per_thread
        raw = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        grad = cute.make_rmem_tensor((self.values_per_thread,), Float32)
        local_sq = Float32(0.0)

        # Recompute the exact BF16/FP16 compression values used by forward.
        for j in cutlass.range_constexpr(self.values_per_thread):
            d = base + j
            maximum = -Float32.inf
            for r in cutlass.range(self.rate, unroll=1):
                pa = block * self.rate + r
                if pa < self.sequence:
                    maximum = cute.arch.fmax(
                        maximum, (za[b, 0, pa, d].to(Float32) + ba[r, d].to(Float32)).to(self.dtype).to(Float32)
                    )
                if block > 0:
                    pb = (block - 1) * self.rate + r
                    if pb < self.sequence:
                        maximum = cute.arch.fmax(
                            maximum, (zb[b, 0, pb, d].to(Float32) + bb[r, d].to(Float32)).to(self.dtype).to(Float32)
                        )
            denom = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                pa = block * self.rate + r
                if pa < self.sequence:
                    la = (za[b, 0, pa, d].to(Float32) + ba[r, d].to(Float32)).to(self.dtype).to(Float32)
                    denom += cute.math.exp(la - maximum, fastmath=False)
                if block > 0:
                    pb = (block - 1) * self.rate + r
                    if pb < self.sequence:
                        lb = (zb[b, 0, pb, d].to(Float32) + bb[r, d].to(Float32)).to(self.dtype).to(Float32)
                        denom += cute.math.exp(lb - maximum, fastmath=False)
            value = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                pa = block * self.rate + r
                va = Float32(0.0)
                if pa < self.sequence:
                    la = (za[b, 0, pa, d].to(Float32) + ba[r, d].to(Float32)).to(self.dtype).to(Float32)
                    p = (cute.math.exp(la - maximum, fastmath=False) / denom).to(self.dtype).to(Float32)
                    va = (ca[b, 0, pa, d].to(Float32) * p).to(self.dtype).to(Float32)
                vb = Float32(0.0)
                if block > 0:
                    pb = (block - 1) * self.rate + r
                    if pb < self.sequence:
                        lb = (zb[b, 0, pb, d].to(Float32) + bb[r, d].to(Float32)).to(self.dtype).to(Float32)
                        p = (cute.math.exp(lb - maximum, fastmath=False) / denom).to(self.dtype).to(Float32)
                        vb = (cb[b, 0, pb, d].to(Float32) * p).to(self.dtype).to(Float32)
                value += (va + vb).to(self.dtype).to(Float32)
            value = value.to(self.dtype).to(Float32)
            raw[j] = value
            local_sq += value * value

        # Inverse RoPE on dY, followed by RMSNorm backward.
        pos = block * self.rate
        local_dot = Float32(0.0)
        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j = pair * 2
            d = base + j
            g0 = dy[b, block, 0, d]
            g1 = dy[b, block, 0, d + 1]
            if d >= self.dim - self.rope:
                rp = (d - (self.dim - self.rope)) // 2
                c = cos[pos, rp]
                s = sin[pos, rp]
                u0 = g0 * c + g1 * s
                u1 = -g0 * s + g1 * c
                g0, g1 = u0, u1
            grad[j], grad[j + 1] = g0, g1
            local_dot += g0 * weight[d].to(Float32) * raw[j]
            local_dot += g1 * weight[d + 1].to(Float32) * raw[j + 1]
        smem = cutlass.utils.SmemAllocator()
        reduction = smem.allocate_tensor(
            Float32,
            cute.make_layout((1, (self.num_threads // cute.arch.WARP_SIZE, 1))),
            byte_alignment=16,
        )
        sum_sq = row_reduce(local_sq, cute.ReductionOp.ADD, self.num_threads, reduction, init_val=0.0)
        dot = row_reduce(local_dot, cute.ReductionOp.ADD, self.num_threads, reduction, init_val=0.0)
        rstd = cute.math.rsqrt(sum_sq / self.dim + _RMS_EPS, fastmath=True)
        correction = dot * rstd * rstd / self.dim

        for j in cutlass.range_constexpr(self.values_per_thread):
            d = base + j
            draw = rstd * (grad[j] * weight[d].to(Float32) - raw[j] * correction)
            cute.arch.atomic_add(dweight.iterator + d, grad[j] * raw[j] * rstd)
            maximum = -Float32.inf
            for r in cutlass.range(self.rate, unroll=1):
                pa = block * self.rate + r
                if pa < self.sequence:
                    maximum = cute.arch.fmax(maximum, (za[b, 0, pa, d].to(Float32) + ba[r, d].to(Float32)).to(self.dtype).to(Float32))
                if block > 0:
                    pb = (block - 1) * self.rate + r
                    if pb < self.sequence:
                        maximum = cute.arch.fmax(maximum, (zb[b, 0, pb, d].to(Float32) + bb[r, d].to(Float32)).to(self.dtype).to(Float32))
            denom = Float32(0.0)
            for r in cutlass.range(self.rate, unroll=1):
                pa = block * self.rate + r
                if pa < self.sequence:
                    la = (za[b, 0, pa, d].to(Float32) + ba[r, d].to(Float32)).to(self.dtype).to(Float32)
                    denom += cute.math.exp(la - maximum, fastmath=False)
                if block > 0:
                    pb = (block - 1) * self.rate + r
                    if pb < self.sequence:
                        lb = (zb[b, 0, pb, d].to(Float32) + bb[r, d].to(Float32)).to(self.dtype).to(Float32)
                        denom += cute.math.exp(lb - maximum, fastmath=False)
            for r in cutlass.range(self.rate, unroll=1):
                pa = block * self.rate + r
                if pa < self.sequence:
                    la = (za[b, 0, pa, d].to(Float32) + ba[r, d].to(Float32)).to(self.dtype).to(Float32)
                    p = (cute.math.exp(la - maximum, fastmath=False) / denom).to(self.dtype).to(Float32)
                    cv = ca[b, 0, pa, d].to(Float32)
                    dca[b, 0, pa, d] = (p * draw).to(self.dtype)
                    gz = p * (cv - raw[j]) * draw
                    dza[b, 0, pa, d] = gz.to(self.dtype)
                    cute.arch.atomic_add(dba.iterator + r * self.dim + d, gz)
                if block > 0:
                    pb = (block - 1) * self.rate + r
                    if pb < self.sequence:
                        lb = (zb[b, 0, pb, d].to(Float32) + bb[r, d].to(Float32)).to(self.dtype).to(Float32)
                        p = (cute.math.exp(lb - maximum, fastmath=False) / denom).to(self.dtype).to(Float32)
                        cv = cb[b, 0, pb, d].to(Float32)
                        dcb[b, 0, pb, d] = (p * draw).to(self.dtype)
                        gz = p * (cv - raw[j]) * draw
                        dzb[b, 0, pb, d] = gz.to(self.dtype)
                        cute.arch.atomic_add(dbb.iterator + r * self.dim + d, gz)


class CastGradient:
    def __init__(self, dtype, length):
        self.dtype = dtype
        self.length = length

    @cute.jit
    def __call__(self, source, dest, stream: cuda.CUstream):
        self.kernel(source, dest).launch(grid=[cute.ceil_div(self.length, 256), 1, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, source, dest):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        i = bid * 256 + tid
        if i < self.length:
            dest[i] = source[i].to(self.dtype)


class SinkReduce:
    def __init__(self, dtype, batch, heads, sequence):
        self.dtype = dtype
        self.batch = batch
        self.heads = heads
        self.sequence = sequence

    @cute.jit
    def __call__(self, rows, result, stream: cuda.CUstream):
        self.kernel(rows, result).launch(grid=[self.heads, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, rows, result):
        tid, _, _ = cute.arch.thread_idx()
        head, _, _ = cute.arch.block_idx()
        value = Float32(0.0)
        for b in cutlass.range_constexpr(self.batch):
            for pos in cutlass.range(tid, self.sequence, 128, unroll=1):
                value += rows[b, head, pos]
        smem = cutlass.utils.SmemAllocator()
        reduction = smem.allocate_tensor(
            Float32, cute.make_layout((1, (4, 1))), byte_alignment=16,
        )
        total = row_reduce(value, cute.ReductionOp.ADD, 128, reduction, init_val=0.0)
        if tid == 0:
            result[head] = total.to(self.dtype)


class PrepareDsaBackward:
    """Rotate/transpose Q, O, dO and combine the two KV-only LSE partials."""

    def __init__(
        self, dsa_dtype, batch, total_heads, packed_heads, head_offset,
        sequence, dim, rope,
    ):
        self.dtype = dsa_dtype
        self.batch = batch
        self.total_heads = total_heads
        self.packed_heads = packed_heads
        self.head_offset = head_offset
        self.sequence = sequence
        self.dim = dim
        self.rope = rope
        self.num_threads = 128
        self.values_per_thread = dim // self.num_threads

    @cute.jit
    def __call__(
        self, q, out, dout, local_lse, compressed_lse, cos, sin,
        q_packed, out_packed, dout_packed, lse, stream: cuda.CUstream,
    ):
        self.kernel(
            q, out, dout, local_lse, compressed_lse, cos, sin,
            q_packed, out_packed, dout_packed, lse,
        ).launch(
            grid=[self.batch * self.sequence * self.packed_heads, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self, q, out, dout, local_lse, compressed_lse, cos, sin,
        q_packed, out_packed, dout_packed, lse,
    ):
        tid, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        packed_head = row % self.packed_heads
        token = row // self.packed_heads
        head = self.head_offset + packed_head
        pos = token % self.sequence
        b = token // self.sequence
        base = tid * self.values_per_thread
        for pair in cutlass.range_constexpr(self.values_per_thread // 2):
            j = pair * 2
            d = base + j
            q0 = q[b, head, pos, d].to(Float32)
            q1 = q[b, head, pos, d + 1].to(Float32)
            o0 = out[b, head, pos, d].to(Float32)
            o1 = out[b, head, pos, d + 1].to(Float32)
            g0 = dout[b, head, pos, d].to(Float32)
            g1 = dout[b, head, pos, d + 1].to(Float32)
            if d >= self.dim - self.rope:
                rp = (d - (self.dim - self.rope)) // 2
                c = cos[pos, rp]
                s = sin[pos, rp]
                q0, q1 = q0 * c - q1 * s, q0 * s + q1 * c
                o0, o1 = o0 * c - o1 * s, o0 * s + o1 * c
                g0, g1 = g0 * c - g1 * s, g0 * s + g1 * c
            q_packed[token, packed_head, d] = q0.to(self.dtype)
            q_packed[token, packed_head, d + 1] = q1.to(self.dtype)
            out_packed[token, packed_head, d] = o0.to(self.dtype)
            out_packed[token, packed_head, d + 1] = o1.to(self.dtype)
            dout_packed[token, packed_head, d] = g0.to(self.dtype)
            dout_packed[token, packed_head, d + 1] = g1.to(self.dtype)
        if tid == 0:
            a = local_lse[b, pos, head]
            c_lse = compressed_lse[b, pos, head]
            maximum = cute.arch.fmax(a, c_lse)
            combined = maximum
            if maximum != -Float32.inf:
                exp_sum = Float32(
                    cute.math.exp(a - maximum, fastmath=False)
                    + cute.math.exp(c_lse - maximum, fastmath=False)
                )
                combined = maximum + cute.math.log(
                    exp_sum,
                    fastmath=False,
                )
            lse[token, packed_head] = combined


class PackDsaKvIndices:
    def __init__(
        self, dsa_dtype, batch, sequence, dim, blocks, rate, topk, window, width
    ):
        self.dtype = dsa_dtype
        self.batch = batch
        self.sequence = sequence
        self.dim = dim
        self.blocks = blocks
        self.rate = rate
        self.topk = topk
        self.window = min(window, sequence)
        self.width = width
        self.kv_per_batch = blocks + sequence

    @cute.jit
    def __call__(
        self, compressed, local, gather, kv, indices, lengths,
        sink, sink_fp32, stream: cuda.CUstream,
    ):
        self.kv_kernel(compressed, local, kv).launch(
            grid=[cute.ceil_div(self.batch * self.kv_per_batch * self.dim, 256), 1, 1],
            block=[256, 1, 1], stream=stream,
        )
        self.index_kernel(gather, indices, lengths).launch(
            grid=[cute.ceil_div(self.batch * self.sequence * self.width, 256), 1, 1],
            block=[256, 1, 1], stream=stream,
        )
        self.sink_kernel(sink, sink_fp32).launch(
            grid=[cute.ceil_div(cute.size(sink), 256), 1, 1], block=[256, 1, 1], stream=stream,
        )

    @cute.kernel
    def kv_kernel(self, compressed, local, kv):
        tid, _, _ = cute.arch.thread_idx(); bid, _, _ = cute.arch.block_idx()
        linear = bid * 256 + tid
        total = self.batch * self.kv_per_batch * self.dim
        if linear < total:
            d = linear % self.dim
            row = linear // self.dim
            within = row % self.kv_per_batch
            b = row // self.kv_per_batch
            if within < self.blocks:
                kv[row, d] = compressed[b, within, 0, d]
            else:
                kv[row, d] = local[b, within - self.blocks, 0, d]

    @cute.kernel
    def index_kernel(self, gather, indices, lengths):
        tid, _, _ = cute.arch.thread_idx(); bid, _, _ = cute.arch.block_idx()
        linear = bid * 256 + tid
        total = self.batch * self.sequence * self.width
        if linear < total:
            slot = linear % self.width
            token = linear // self.width
            pos = token % self.sequence
            b = token // self.sequence
            compressed_count = min(self.topk, (pos + 1) // self.rate)
            local_count = min(self.window, pos + 1)
            valid_length = compressed_count + local_count
            index = Int32(-1)
            base = b * self.kv_per_batch
            if slot < compressed_count:
                block = gather[b, pos, slot]
                if block >= 0:
                    index = Int32(base + block)
            elif slot < valid_length:
                key = pos - local_count + 1 + (slot - compressed_count)
                index = Int32(base + self.blocks + key)
            indices[token, slot] = index
            if slot == 0:
                # The vendored DSA loader always executes its first tile. A single masked
                # sentinel represents an otherwise empty selected row safely.
                lengths[token] = Int32(max(1, valid_length))

    @cute.kernel
    def sink_kernel(self, sink, sink_fp32):
        tid, _, _ = cute.arch.thread_idx(); bid, _, _ = cute.arch.block_idx()
        i = bid * 256 + tid
        if i < cute.size(sink):
            sink_fp32[i] = sink[i].to(Float32)


class UnpackDsaGradients:
    def __init__(
        self, output_dtype, batch, total_heads, packed_heads, head_offset,
        sequence, dim, rope, blocks,
    ):
        self.dtype = output_dtype
        self.batch = batch
        self.total_heads = total_heads
        self.packed_heads = packed_heads
        self.head_offset = head_offset
        self.sequence = sequence
        self.dim = dim
        self.rope = rope
        self.blocks = blocks
        self.kv_per_batch = blocks + sequence

    @cute.jit
    def __call__(self, dq_packed, dkv, cos, sin, dq, dlocal, dcompressed, stream: cuda.CUstream):
        self.dq_kernel(dq_packed, cos, sin, dq).launch(
            grid=[self.batch * self.sequence * self.packed_heads, 1, 1], block=[128, 1, 1], stream=stream,
        )
        self.dkv_kernel(dkv, dlocal, dcompressed).launch(
            grid=[cute.ceil_div(self.batch * self.kv_per_batch * self.dim, 256), 1, 1],
            block=[256, 1, 1], stream=stream,
        )

    @cute.kernel
    def dq_kernel(self, source, cos, sin, dest):
        tid, _, _ = cute.arch.thread_idx(); row, _, _ = cute.arch.block_idx()
        packed_head = row % self.packed_heads; token = row // self.packed_heads
        head = self.head_offset + packed_head
        pos = token % self.sequence; b = token // self.sequence
        base = tid * (self.dim // 128)
        for pair in cutlass.range_constexpr(self.dim // 128 // 2):
            j = pair * 2; d = base + j
            x0 = source[token, packed_head, d].to(Float32); x1 = source[token, packed_head, d + 1].to(Float32)
            if d >= self.dim - self.rope:
                rp = (d - (self.dim - self.rope)) // 2
                c = cos[pos, rp]; s = sin[pos, rp]
                x0, x1 = x0 * c + x1 * s, -x0 * s + x1 * c
            dest[b, head, pos, d] = x0.to(self.dtype)
            dest[b, head, pos, d + 1] = x1.to(self.dtype)

    @cute.kernel
    def dkv_kernel(self, source, dlocal, dcompressed):
        tid, _, _ = cute.arch.thread_idx(); bid, _, _ = cute.arch.block_idx()
        linear = bid * 256 + tid
        total = self.batch * self.kv_per_batch * self.dim
        if linear < total:
            d = linear % self.dim; row = linear // self.dim
            within = row % self.kv_per_batch; b = row // self.kv_per_batch
            value = source[row, d].to(Float32)
            if within < self.blocks:
                dcompressed[b, within, 0, d] += value
            else:
                dlocal[b, within - self.blocks, 0, d] += value


@lru_cache
def compile_local_norm_backward(dtype, batch, sequence, dim, rope):
    return cute.compile(
        LocalNormBackward(dtype, batch, sequence, dim, rope),
        _fake(dtype, (batch, 1, sequence, dim)), _fake(dtype, (dim,)),
        _fake(Float32, (sequence, rope // 2)), _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (batch, sequence, 1, dim)),
        _fake(dtype, (batch, 1, sequence, dim)), _fake(Float32, (dim,)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


@lru_cache
def compile_compression_backward(dtype, batch, sequence, dim, rate, rope):
    blocks = (sequence + rate - 1) // rate
    return cute.compile(
        CompressionBackward(dtype, batch, sequence, dim, rate, rope),
        *[_fake(dtype, (batch, 1, sequence, dim)) for _ in range(4)],
        _fake(dtype, (rate, dim)), _fake(dtype, (rate, dim)), _fake(dtype, (dim,)),
        _fake(Float32, (sequence, rope // 2)), _fake(Float32, (sequence, rope // 2)),
        _fake(Float32, (batch, blocks, 1, dim)),
        *[_fake(dtype, (batch, 1, sequence, dim)) for _ in range(4)],
        _fake(Float32, (rate, dim)), _fake(Float32, (rate, dim)), _fake(Float32, (dim,)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


@lru_cache
def compile_cast_gradient(dtype, length):
    return cute.compile(
        CastGradient(dtype, length), _fake(Float32, (length,)), _fake(dtype, (length,)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


@lru_cache
def compile_sink_reduce(dtype, batch, heads, sequence):
    return cute.compile(
        SinkReduce(dtype, batch, heads, sequence),
        _fake(Float32, (batch, heads, sequence)), _fake(dtype, (heads,)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


@lru_cache
def compile_prepare_dsa_backward(
    input_dtype, dsa_dtype, batch, total_heads, packed_heads, head_offset,
    sequence, dim, rope,
):
    return cute.compile(
        PrepareDsaBackward(
            dsa_dtype, batch, total_heads, packed_heads, head_offset,
            sequence, dim, rope,
        ),
        _fake(input_dtype, (batch, total_heads, sequence, dim)),
        _fake(input_dtype, (batch, total_heads, sequence, dim)),
        _fake(input_dtype, (batch, total_heads, sequence, dim)),
        _fake(Float32, (batch, sequence, total_heads)),
        _fake(Float32, (batch, sequence, total_heads)),
        _fake(Float32, (sequence, rope // 2)), _fake(Float32, (sequence, rope // 2)),
        *[_fake(dsa_dtype, (batch * sequence, packed_heads, dim)) for _ in range(3)],
        _fake(Float32, (batch * sequence, packed_heads)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


@lru_cache
def compile_pack_dsa_kv_indices(
    input_dtype, dsa_dtype, batch, sequence, dim, blocks, rate,
    topk, window, width, heads, gather_width,
):
    return cute.compile(
        PackDsaKvIndices(
            dsa_dtype, batch, sequence, dim, blocks, rate, topk, window, width
        ),
        _fake(input_dtype, (batch, blocks, 1, dim)), _fake(input_dtype, (batch, sequence, 1, dim)),
        _fake(Int32, (batch, sequence, gather_width)),
        _fake(dsa_dtype, (batch * (blocks + sequence), dim)),
        _fake(Int32, (batch * sequence, width)),
        _fake(Int32, (batch * sequence,)),
        _fake(input_dtype, (heads,)), _fake(Float32, (heads,)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


@lru_cache
def compile_unpack_dsa_gradients(
    output_dtype, dsa_dtype, batch, total_heads, packed_heads, head_offset,
    sequence, dim, rope, blocks,
):
    return cute.compile(
        UnpackDsaGradients(
            output_dtype, batch, total_heads, packed_heads, head_offset,
            sequence, dim, rope, blocks,
        ),
        _fake(dsa_dtype, (batch * sequence, packed_heads, dim)),
        _fake(dsa_dtype, (batch * (blocks + sequence), dim)),
        _fake(Float32, (sequence, rope // 2)), _fake(Float32, (sequence, rope // 2)),
        _fake(output_dtype, (batch, total_heads, sequence, dim)),
        _fake(Float32, (batch, sequence, 1, dim)),
        _fake(Float32, (batch, blocks, 1, dim)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi",
    )


__all__ = [
    "compile_attention_backward", "compile_local_norm_backward",
    "compile_compression_backward", "compile_cast_gradient", "cute_dtype",
    "compile_sink_reduce",
    "compile_prepare_dsa_backward", "compile_pack_dsa_kv_indices",
    "compile_unpack_dsa_gradients",
]
