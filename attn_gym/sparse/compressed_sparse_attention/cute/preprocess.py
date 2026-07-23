"""CuTe DSL preprocessing kernels for the SM100 CSA specialization.

This module deliberately contains no attention dispatch.  It prepares the three shared tensors
consumed by the fused H=128, D=512 attention kernel:

* RMS-normalized and RoPE-transformed local KV;
* compressed, RMS-normalized, and RoPE-transformed attention KV; and
* compressed, RMS-normalized, and RoPE-transformed index KV.

The implementation specializes the model configuration used by the SM100 backend: one physical
shared KV head, compression rate 32, and a 64-wide YaRN RoPE tail.  All tensor arithmetic is in
CuTe DSL kernels; the Python wrapper only validates tensors, allocates outputs, and launches the
compiled kernels on PyTorch's current CUDA stream.
"""

from __future__ import annotations

from collections import OrderedDict

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack


_THREADS = 128
_WARPS = _THREADS // 32
_COMPRESSION_RATE = 32
_ROPE_DIM = 64
_ATTENTION_DIM = 512
_INDEX_DIM = 64
_RMS_EPS = torch.finfo(torch.float32).eps


def _reference_yarn_frequencies() -> tuple[float, ...]:
    """Build the 32 static FP32 constants with the oracle's operation order."""
    pair_positions = torch.arange(0, _ROPE_DIM, 2, dtype=torch.float32)
    frequencies = 1.0 / (160_000.0 ** (pair_positions / _ROPE_DIM))
    ramp = (torch.arange(_ROPE_DIM // 2, dtype=torch.float32) - 15) / 10
    smooth = 1 - ramp.clamp(0, 1)
    frequencies = frequencies / 16.0 * (1 - smooth) + frequencies * smooth
    return tuple(float(value) for value in frequencies)


_YARN_FREQUENCIES = _reference_yarn_frequencies()


@cute.jit
def _warp_sum(value: Float32) -> Float32:
    for stage in cutlass.range_constexpr(5):
        value += cute.arch.shuffle_sync_bfly(value, offset=1 << stage)
    return value


@cute.jit
def _block_rsqrt(
    local_sum: Float32,
    tidx: Int32,
    warp_sums: cute.Pointer,
    dimension: cutlass.Constexpr[int],
) -> Float32:
    """Reduce one sum per thread and broadcast ``rsqrt(mean + eps)``."""
    lane = tidx % 32
    warp = tidx // 32
    local_sum = _warp_sum(local_sum)
    if lane == 0:
        warp_sums[warp] = local_sum
    cute.arch.sync_threads()

    if warp == 0:
        block_sum = Float32(0.0)
        if lane < _WARPS:
            block_sum = warp_sums[lane]
        block_sum = _warp_sum(block_sum)
        if lane == 0:
            warp_sums[0] = cute.math.rsqrt(
                block_sum / Float32(dimension) + Float32(_RMS_EPS),
                fastmath=False,
            )
    cute.arch.sync_threads()
    return warp_sums[0]


@cute.jit
def _yarn_frequency(rotary_pair: Int32) -> Float32:
    """Return the reference implementation's 64-wide YaRN frequency."""
    # For rotary_dim=64 the reference's correction bounds are floor(15.453...)=15 and
    # ceil(24.708...)=25.  Select from static FP32 constants rather than recomputing a power
    # as exp(log(x) * y), whose last bit differs for several lanes at long positions.
    frequency = Float32(_YARN_FREQUENCIES[0])
    for pair in cutlass.range_constexpr(_ROPE_DIM // 2):
        if rotary_pair == pair:
            frequency = Float32(_YARN_FREQUENCIES[pair])
    return frequency


@cute.jit
def _rotate_tail_in_place(
    mOut: cute.Tensor,
    batch: Int32,
    row: Int32,
    position: Int32,
    tidx: Int32,
    dimension: cutlass.Constexpr[int],
    threads_per_row: cutlass.Constexpr[int],
) -> None:
    """Apply RoPE to a BF16/FP16-normalized output row in place."""
    pairs_per_thread = cute.ceil_div(_ROPE_DIM // 2, threads_per_row)
    for item in cutlass.range_constexpr(pairs_per_thread):
        rotary_pair = tidx + item * threads_per_row
        if rotary_pair < _ROPE_DIM // 2:
            out_dtype = mOut.element_type
            d0 = dimension - _ROPE_DIM + 2 * rotary_pair
            d1 = d0 + 1
            x0 = Float32(mOut[batch, row, 0, d0])
            x1 = Float32(mOut[batch, row, 0, d1])
            angle = Float32(position) * _yarn_frequency(rotary_pair)
            cosine = cute.math.cos(angle, fastmath=False)
            sine = cute.math.sin(angle, fastmath=False)
            mOut[batch, row, 0, d0] = out_dtype(x0 * cosine - x1 * sine)
            mOut[batch, row, 0, d1] = out_dtype(x1 * cosine + x0 * sine)


class _RmsNormRopeSm100:
    def __init__(self, dimension: int, position_stride: int = 1):
        self.dimension = dimension
        self.position_stride = position_stride
        if dimension == _ATTENTION_DIM:
            self.threads_per_row = 32
        elif dimension == _INDEX_DIM:
            self.threads_per_row = 8
        else:
            raise ValueError("The SM100 RMSNorm specialization supports D=512 or D=64.")
        self.rows_per_cta = _THREADS // self.threads_per_row
        self.vector_width = 8
        self.vectors_per_thread = dimension // (
            self.threads_per_row * self.vector_width
        )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mWeight: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        rows = cute.size(mX.shape[0]) * cute.size(mX.shape[2])
        self.kernel(mX, mWeight, mOut).launch(
            grid=(cute.ceil_div(rows, self.rows_per_cta), 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mWeight: cute.Tensor, mOut: cute.Tensor) -> None:
        tidx = Int32(cute.arch.thread_idx()[0])
        row_in_cta = tidx // self.threads_per_row
        lane_in_row = tidx - row_in_cta * self.threads_per_row
        flat_row = Int32(cute.arch.block_idx()[0]) * self.rows_per_cta + row_in_cta
        total_rows = Int32(cute.size(mX.shape[0]) * cute.size(mX.shape[2]))
        sequence = Int32(cute.size(mX.shape[2]))
        batch = flat_row // sequence
        row = flat_row - batch * sequence
        dimension = self.dimension
        out_dtype = mOut.element_type

        # The reference build dispatches RMSNorm to Quack.  Its D=512 specialization
        # assigns one warp to each row and two contiguous vectors of eight to each lane;
        # D=64 assigns one eight-thread subgroup and one vector of eight per thread.
        local_sum = Float32(0.0)
        if flat_row < total_rows:
            for vector in cutlass.range_constexpr(self.vectors_per_thread):
                vector_start = (
                    lane_in_row * self.vector_width
                    + vector * self.threads_per_row * self.vector_width
                )
                for item in cutlass.range_constexpr(self.vector_width):
                    value = Float32(mX[batch, 0, row, vector_start + item])
                    local_sum += value * value
        offset = self.threads_per_row // 2
        while offset > 0:
            local_sum += cute.arch.shuffle_sync_bfly(local_sum, offset=offset)
            offset //= 2
        scale = cute.math.rsqrt(
            local_sum / Float32(dimension) + Float32(_RMS_EPS), fastmath=True
        )

        # Store the RMSNorm result in the output dtype first.  The reference applies RoPE to
        # this rounded tensor, rather than fusing normalization and rotation in FP32.
        if flat_row < total_rows:
            for vector in cutlass.range_constexpr(self.vectors_per_thread):
                vector_start = (
                    lane_in_row * self.vector_width
                    + vector * self.threads_per_row * self.vector_width
                )
                for item in cutlass.range_constexpr(self.vector_width):
                    d = vector_start + item
                    mOut[batch, row, 0, d] = out_dtype(
                        (Float32(mX[batch, 0, row, d]) * scale)
                        * Float32(mWeight[d])
                    )
        cute.arch.sync_threads()
        if flat_row < total_rows:
            _rotate_tail_in_place(
                mOut,
                batch,
                row,
                row * self.position_stride,
                lane_in_row,
                dimension,
                self.threads_per_row,
            )


class _CompressOnlySm100:
    """FP16 compression matching spatial softmax and CUDA Reduce.cuh bit-for-bit."""

    def __init__(self, dimension: int, compression_rate: int):
        self.dimension = dimension
        self.compression_rate = compression_rate
        self.dimensions_per_cta = _WARPS
        assert compression_rate == _COMPRESSION_RATE
        assert dimension % self.dimensions_per_cta == 0

    @cute.jit
    def __call__(
        self,
        mCa: cute.Tensor,
        mCb: cute.Tensor,
        mZa: cute.Tensor,
        mZb: cute.Tensor,
        mBa: cute.Tensor,
        mBb: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        num_blocks = cute.size(mOut.shape[2])
        dimension_tiles = self.dimension // self.dimensions_per_cta
        rows = cute.size(mOut.shape[0]) * num_blocks * dimension_tiles
        self.kernel(mCa, mCb, mZa, mZb, mBa, mBb, mOut).launch(
            grid=(rows, 1, 1),
            block=(_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCa: cute.Tensor,
        mCb: cute.Tensor,
        mZa: cute.Tensor,
        mZb: cute.Tensor,
        mBa: cute.Tensor,
        mBb: cute.Tensor,
        mOut: cute.Tensor,
    ) -> None:
        tidx = Int32(cute.arch.thread_idx()[0])
        flat_row = Int32(cute.arch.block_idx()[0])
        warp = tidx // 32
        lane = tidx - warp * 32
        dimension_tiles = self.dimension // self.dimensions_per_cta
        dimension_tile = flat_row % dimension_tiles
        flat_block = flat_row // dimension_tiles
        num_blocks = Int32(cute.size(mOut.shape[2]))
        batch = flat_block // num_blocks
        block = flat_block - batch * num_blocks
        d = dimension_tile * self.dimensions_per_cta + warp
        out_dtype = mOut.element_type

        source_a = block * self.compression_rate + lane
        # Keep native FP16 pointwise operations.  That is exactly how TensorIterator
        # produces logits, weighted A/B values, and their pairwise sum in the oracle.
        logit_a_half = mZa[batch, 0, source_a, d] + mBa[lane, d]
        logit_a = Float32(logit_a_half)
        logit_b = -Float32.inf
        source_b = (block - 1) * self.compression_rate + lane
        if block > 0:
            logit_b_half = mZb[batch, 0, source_b, d] + mBb[lane, d]
            logit_b = Float32(logit_b_half)

        maximum = cute.arch.fmax(logit_a, logit_b)
        for stage in cutlass.range_constexpr(5):
            maximum = cute.arch.fmax(
                maximum, cute.arch.shuffle_sync_bfly(maximum, offset=1 << stage)
            )
        exp_a = cute.math.exp(logit_a - maximum, fastmath=False)
        exp_b = Float32(0.0)
        if block > 0:
            exp_b = cute.math.exp(logit_b - maximum, fastmath=False)

        if cutlass.const_expr(self.dimension > _INDEX_DIM):
            # D=512: SpatialSoftMax uses block=(1, 512), so its one reduction thread
            # visits A[0:32] and then B[0:32] in storage order.
            denominator = Float32(0.0)
            for source_lane in cutlass.range_constexpr(self.compression_rate):
                denominator += cute.arch.shuffle_sync(exp_a, offset=source_lane)
            for source_lane in cutlass.range_constexpr(self.compression_rate):
                denominator += cute.arch.shuffle_sync(exp_b, offset=source_lane)
        else:
            # D=64: SpatialSoftMax uses block=(16, 64).  Thread x visits
            # [A[x], A[x+16], B[x], B[x+16]], then a shared-memory tree reduces x.
            denominator = Float32(0.0)
            exp_a_hi = cute.arch.shuffle_sync(exp_a, offset=lane + 16)
            exp_b_hi = cute.arch.shuffle_sync(exp_b, offset=lane + 16)
            if lane < 16:
                denominator += exp_a
                denominator += exp_a_hi
                denominator += exp_b
                denominator += exp_b_hi
            for offset in (8, 4, 2, 1):
                other = cute.arch.shuffle_sync_down(denominator, offset=offset)
                if lane < offset:
                    denominator += other
            denominator = cute.arch.shuffle_sync(denominator, offset=0)

        probability_a_half = out_dtype(exp_a / denominator)
        product_a_half = mCa[batch, 0, source_a, d] * probability_a_half
        product_b_half = out_dtype(0.0)
        if block > 0:
            probability_b_half = out_dtype(exp_b / denominator)
            product_b_half = mCb[batch, 0, source_b, d] * probability_b_half
        pair_value = Float32(product_a_half + product_b_half)

        # CUDA Reduce.cuh uses output_vec_size=4 and vt0=4 here: four FP32
        # modulo-four accumulators followed by a left-to-right combine.
        acc0 = Float32(0.0)
        acc1 = Float32(0.0)
        acc2 = Float32(0.0)
        acc3 = Float32(0.0)
        for group in cutlass.range_constexpr(self.compression_rate // 4):
            acc0 += cute.arch.shuffle_sync(pair_value, offset=group * 4)
            acc1 += cute.arch.shuffle_sync(pair_value, offset=group * 4 + 1)
            acc2 += cute.arch.shuffle_sync(pair_value, offset=group * 4 + 2)
            acc3 += cute.arch.shuffle_sync(pair_value, offset=group * 4 + 3)
        compressed = acc0 + acc1
        compressed += acc2
        compressed += acc3
        if lane == 0:
            mOut[batch, 0, block, d] = out_dtype(compressed)


class _CompressRmsNormRopeSm100:
    def __init__(self, dimension: int, compression_rate: int):
        self.dimension = dimension
        self.compression_rate = compression_rate
        assert compression_rate == _COMPRESSION_RATE
        # Compression is independent across dimensions.  Give every D=512 dimension its
        # own thread instead of making 128 threads serialize four softmaxes apiece.  Keep
        # the D=64 launch unchanged: it is already small and launch-bound.
        self.threads = 512 if dimension == _ATTENTION_DIM else _THREADS
        self.warps = self.threads // 32

    @cute.jit
    def __call__(
        self,
        mCa: cute.Tensor,
        mCb: cute.Tensor,
        mZa: cute.Tensor,
        mZb: cute.Tensor,
        mBa: cute.Tensor,
        mBb: cute.Tensor,
        mWeight: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        rows = cute.size(mOut.shape[0]) * cute.size(mOut.shape[1])
        self.kernel(mCa, mCb, mZa, mZb, mBa, mBb, mWeight, mOut).launch(
            grid=(rows, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCa: cute.Tensor,
        mCb: cute.Tensor,
        mZa: cute.Tensor,
        mZb: cute.Tensor,
        mBa: cute.Tensor,
        mBb: cute.Tensor,
        mWeight: cute.Tensor,
        mOut: cute.Tensor,
    ) -> None:
        tidx = Int32(cute.arch.thread_idx()[0])
        flat_block = Int32(cute.arch.block_idx()[0])
        num_blocks = Int32(cute.size(mOut.shape[1]))
        batch = flat_block // num_blocks
        block = flat_block - batch * num_blocks
        dimension = self.dimension
        rate = self.compression_rate
        out_dtype = mOut.element_type

        allocator = cutlass.utils.SmemAllocator()
        warp_sums = allocator.allocate_array(Float32, self.warps)

        values_per_thread = cute.ceil_div(dimension, self.threads)
        for item in cutlass.range_constexpr(values_per_thread):
            d = tidx + item * self.threads
            if d < dimension:
                maximum = -Float32.inf
                for offset in cutlass.range_constexpr(rate):
                    source_a = block * rate + offset
                    logit_a = Float32(
                        out_dtype(Float32(mZa[batch, 0, source_a, d]) + Float32(mBa[offset, d]))
                    )
                    maximum = cute.arch.fmax(maximum, logit_a)
                    if block > 0:
                        source_b = (block - 1) * rate + offset
                        logit_b = Float32(
                            out_dtype(
                                Float32(mZb[batch, 0, source_b, d])
                                + Float32(mBb[offset, d])
                            )
                        )
                        maximum = cute.arch.fmax(maximum, logit_b)

                denominator = Float32(0.0)
                if cutlass.const_expr(dimension > _INDEX_DIM):
                    # With one D=512 dimension per thread, cache the exact FP32
                    # exponentials in registers.  The weighted pass can then avoid a
                    # second set of 64 SFU evaluations and redundant Z/bias loads.
                    exponentials = cute.make_rmem_tensor((2 * rate,), Float32)
                for offset in cutlass.range_constexpr(rate):
                    source_a = block * rate + offset
                    logit_a = Float32(
                        out_dtype(Float32(mZa[batch, 0, source_a, d]) + Float32(mBa[offset, d]))
                    )
                    exp_a = cute.math.exp(logit_a - maximum, fastmath=False)
                    if cutlass.const_expr(dimension > _INDEX_DIM):
                        exponentials[offset] = exp_a
                    denominator += exp_a
                    if block > 0:
                        source_b = (block - 1) * rate + offset
                        logit_b = Float32(
                            out_dtype(
                                Float32(mZb[batch, 0, source_b, d])
                                + Float32(mBb[offset, d])
                            )
                        )
                        exp_b = cute.math.exp(logit_b - maximum, fastmath=False)
                        if cutlass.const_expr(dimension > _INDEX_DIM):
                            exponentials[rate + offset] = exp_b
                        denominator += exp_b

                compressed = Float32(0.0)
                for offset in cutlass.range_constexpr(rate):
                    source_a = block * rate + offset
                    logit_a = Float32(
                        out_dtype(Float32(mZa[batch, 0, source_a, d]) + Float32(mBa[offset, d]))
                    )
                    if cutlass.const_expr(dimension > _INDEX_DIM):
                        probability_a = Float32(
                            out_dtype(exponentials[offset] / denominator)
                        )
                    else:
                        probability_a = Float32(
                            out_dtype(
                                cute.math.exp(logit_a - maximum, fastmath=False)
                                / denominator
                            )
                        )
                    product_a = Float32(
                        out_dtype(Float32(mCa[batch, 0, source_a, d]) * probability_a)
                    )

                    product_b = Float32(0.0)
                    if block > 0:
                        source_b = (block - 1) * rate + offset
                        logit_b = Float32(
                            out_dtype(
                                Float32(mZb[batch, 0, source_b, d])
                                + Float32(mBb[offset, d])
                            )
                        )
                        if cutlass.const_expr(dimension > _INDEX_DIM):
                            probability_b = Float32(
                                out_dtype(exponentials[rate + offset] / denominator)
                            )
                        else:
                            probability_b = Float32(
                                out_dtype(
                                    cute.math.exp(logit_b - maximum, fastmath=False)
                                    / denominator
                                )
                            )
                        product_b = Float32(
                            out_dtype(Float32(mCb[batch, 0, source_b, d]) * probability_b)
                        )
                    # The pointwise add is rounded in the input dtype; torch.sum then uses an
                    # FP32 reduction accumulator for BF16/FP16 CUDA inputs.
                    compressed += Float32(out_dtype(product_a + product_b))
                mOut[batch, block, 0, d] = out_dtype(compressed)

        cute.arch.sync_threads()

        # Preserve the original/reference reduction order exactly.  The first 128 threads
        # gather the same four strided dimensions they owned before the compression launch
        # was widened; the extra warps contribute zero and are ignored by _block_rsqrt.
        local_sum = Float32(0.0)
        norm_values_per_thread = cute.ceil_div(dimension, _THREADS)
        if tidx < _THREADS:
            for item in cutlass.range_constexpr(norm_values_per_thread):
                d = tidx + item * _THREADS
                if d < dimension:
                    value = Float32(mOut[batch, block, 0, d])
                    local_sum += value * value
        scale = _block_rsqrt(local_sum, tidx, warp_sums, dimension)

        for item in cutlass.range_constexpr(values_per_thread):
            d = tidx + item * self.threads
            if d < dimension:
                mOut[batch, block, 0, d] = out_dtype(
                    Float32(mOut[batch, block, 0, d]) * scale * Float32(mWeight[d])
                )
        cute.arch.sync_threads()
        _rotate_tail_in_place(
            mOut, batch, block, block * rate, tidx, dimension, _THREADS
        )


_COMPILE_CACHE_MAXSIZE = 64
_compile_cache: OrderedDict[tuple[object, ...], object] = OrderedDict()
_aux_streams: dict[int, torch.cuda.Stream] = {}


def parallel_preprocess_stream(KV: torch.Tensor) -> torch.cuda.Stream | None:
    # B=1 and B=4 are the measured throughput points where the independent local/index
    # branch can overlap the D=512 compression branch.  Keep unmeasured batch sizes on
    # one stream until their launch balance has been benchmarked as well.
    if KV.dtype != torch.bfloat16 or KV.shape[0] not in (1, 4) or KV.shape[2] < 1024:
        return None
    device = KV.device
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    stream = _aux_streams.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device_index)
        _aux_streams[device_index] = stream
    return stream


def _as_cute(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16)


def _cache_get(key: tuple[object, ...]) -> object | None:
    compiled = _compile_cache.get(key)
    if compiled is not None:
        _compile_cache.move_to_end(key)
    return compiled


def _cache_put(key: tuple[object, ...], compiled: object) -> None:
    _compile_cache[key] = compiled
    _compile_cache.move_to_end(key)
    if len(_compile_cache) > _COMPILE_CACHE_MAXSIZE:
        _compile_cache.popitem(last=False)


def _compiled_rms_rope(
    tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    stream: cuda.CUstream,
    position_stride: int = 1,
) -> object:
    key = (
        "rms_rope",
        tensor.dtype,
        tuple(tensor.shape),
        tuple(output.shape),
        tuple(output.stride()),
        tensor.shape[-1],
        position_stride,
    )
    compiled = _cache_get(key)
    if compiled is None:
        compiled = cute.compile(
            _RmsNormRopeSm100(tensor.shape[-1], position_stride),
            _as_cute(tensor),
            _as_cute(weight),
            _as_cute(output),
            stream,
        )
        _cache_put(key, compiled)
    return compiled


def _compiled_compress(
    tensors: tuple[torch.Tensor, ...],
    stream: cuda.CUstream,
) -> object:
    source = tensors[0]
    output = tensors[-1]
    key = (
        "compress",
        source.dtype,
        tuple(source.shape),
        tuple(output.shape),
        tuple(output.stride()),
        output.shape[-1],
    )
    compiled = _cache_get(key)
    if compiled is None:
        compiled = cute.compile(
            _CompressRmsNormRopeSm100(output.shape[-1], _COMPRESSION_RATE),
            *(_as_cute(tensor) for tensor in tensors),
            stream,
        )
        _cache_put(key, compiled)
    return compiled


def _compiled_compress_only(
    tensors: tuple[torch.Tensor, ...],
    stream: cuda.CUstream,
) -> object:
    source = tensors[0]
    output = tensors[-1]
    key = (
        "compress_only",
        source.dtype,
        tuple(source.shape),
        tuple(output.shape),
        output.shape[-1],
    )
    compiled = _cache_get(key)
    if compiled is None:
        compiled = cute.compile(
            _CompressOnlySm100(output.shape[-1], _COMPRESSION_RATE),
            *(_as_cute(tensor) for tensor in tensors),
            stream,
        )
        _cache_put(key, compiled)
    return compiled


def _validate_shared_tensor(name: str, tensor: torch.Tensor, dimension: int) -> None:
    if tensor.ndim != 4 or tensor.shape[1] != 1 or tensor.shape[-1] != dimension:
        raise ValueError(f"{name} must have shape [B, 1, S, {dimension}].")
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor.")
    if tensor.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"{name} must have dtype bfloat16 or float16.")


def preprocess_shared_kv(
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    *,
    compression_rate: int,
    rope_dims: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the SM100 CuTe preprocessing pipeline for one-head shared KV inputs."""
    if torch.cuda.get_device_capability(KV.device) != (10, 0):
        raise RuntimeError("CuTe CSA preprocessing targets compute capability 10.0 exclusively.")
    if compression_rate != _COMPRESSION_RATE or rope_dims != _ROPE_DIM:
        raise ValueError("CuTe CSA preprocessing requires compression_rate=32 and rope_dims=64.")

    for name, tensor in (
        ("KV", KV),
        ("C_a", C_a),
        ("C_b", C_b),
        ("Z_a", Z_a),
        ("Z_b", Z_b),
    ):
        _validate_shared_tensor(name, tensor, _ATTENTION_DIM)
    for name, tensor in (
        ("K_Ia", K_Ia),
        ("K_Ib", K_Ib),
        ("Z_Ia", Z_Ia),
        ("Z_Ib", Z_Ib),
    ):
        _validate_shared_tensor(name, tensor, _INDEX_DIM)

    sequence_length = KV.shape[2]
    if sequence_length % _COMPRESSION_RATE:
        raise ValueError("The current CuTe preprocessing specialization requires S % 32 == 0.")
    dtype = KV.dtype
    device = KV.device
    all_inputs = (
        C_a,
        C_b,
        Z_a,
        Z_b,
        K_Ia,
        K_Ib,
        Z_Ia,
        Z_Ib,
        B_a,
        B_b,
        B_Ia,
        B_Ib,
        KV_norm_weight,
        compressed_indices_norm_weight,
        compressed_kv_norm_weight,
    )
    if any(tensor.device != device or tensor.dtype != dtype for tensor in all_inputs):
        raise ValueError("All CuTe preprocessing inputs must share KV's device and dtype.")
    if any(not tensor.is_contiguous() for tensor in all_inputs):
        raise ValueError("All CuTe preprocessing inputs must be contiguous.")

    num_blocks = sequence_length // _COMPRESSION_RATE
    local_kv = torch.empty(
        KV.shape[0], sequence_length, 1, _ATTENTION_DIM, dtype=dtype, device=device
    )
    compressed_kv = torch.empty(
        KV.shape[0], num_blocks, 1, _ATTENTION_DIM, dtype=dtype, device=device
    )
    compressed_indices = torch.empty(
        KV.shape[0], num_blocks, 1, _INDEX_DIM, dtype=dtype, device=device
    )
    torch_stream = torch.cuda.current_stream(device)
    stream = cuda.CUstream(torch_stream.cuda_stream)
    index_torch_stream = parallel_preprocess_stream(KV)
    use_parallel_stream = index_torch_stream is not None
    index_stream = stream
    if use_parallel_stream:
        assert index_torch_stream is not None
        index_torch_stream.wait_stream(torch_stream)
        index_stream = cuda.CUstream(index_torch_stream.cuda_stream)

    rms_compiled = _compiled_rms_rope(KV, KV_norm_weight, local_kv, index_stream)
    rms_compiled(
        _as_cute(KV),
        _as_cute(KV_norm_weight),
        _as_cute(local_kv),
        index_stream,
    )

    if dtype == torch.float16:
        raw_attention = torch.empty(
            KV.shape[0], 1, num_blocks, _ATTENTION_DIM, dtype=dtype, device=device
        )
        raw_indices = torch.empty(
            KV.shape[0], 1, num_blocks, _INDEX_DIM, dtype=dtype, device=device
        )
        for source_args, raw, weight, output in (
            ((C_a, C_b, Z_a, Z_b, B_a, B_b), raw_attention, compressed_kv_norm_weight, compressed_kv),
            ((K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib), raw_indices, compressed_indices_norm_weight, compressed_indices),
        ):
            compression_args = (*source_args, raw)
            compression_compiled = _compiled_compress_only(compression_args, stream)
            compression_compiled(
                *(_as_cute(tensor) for tensor in compression_args), stream
            )
            norm_compiled = _compiled_rms_rope(
                raw, weight, output, stream, _COMPRESSION_RATE
            )
            norm_compiled(_as_cute(raw), _as_cute(weight), _as_cute(output), stream)
    else:
        attention_args = (
            C_a,
            C_b,
            Z_a,
            Z_b,
            B_a,
            B_b,
            compressed_kv_norm_weight,
            compressed_kv,
        )
        attention_compiled = _compiled_compress(attention_args, stream)
        attention_compiled(*(_as_cute(tensor) for tensor in attention_args), stream)

        index_args = (
            K_Ia,
            K_Ib,
            Z_Ia,
            Z_Ib,
            B_Ia,
            B_Ib,
            compressed_indices_norm_weight,
            compressed_indices,
        )
        index_compiled = _compiled_compress(index_args, index_stream)
        index_compiled(*(_as_cute(tensor) for tensor in index_args), index_stream)
    if use_parallel_stream:
        torch_stream.wait_stream(index_torch_stream)
    return local_kv, compressed_kv, compressed_indices


__all__ = ["parallel_preprocess_stream", "preprocess_shared_kv"]
