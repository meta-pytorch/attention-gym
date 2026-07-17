"""SM100 CuTe DSL backend for shared-KV compressed sparse attention.

The public entry point owns the complete CSA data path: compression, RMS normalization,
RoPE, index scoring/top-k, local and compressed attention, learned-sink merge, and inverse
RoPE.  PyTorch is used only for validation, allocation, and cached RoPE constants; input-
dependent tensor arithmetic is launched through CuTe DSL kernels.
"""

from __future__ import annotations

from functools import lru_cache
import math

import torch
from cutlass import BFloat16

from .cute_kernels import (
    compile_compression,
    compile_compressed_merge,
    compile_index_topk,
    compile_local_norm,
    compile_merge,
    compile_prefix,
    compile_query_rope,
    cute_dtype,
)
from .cute_local_attention import compressed_attention, local_attention
from .cute_preprocess import parallel_preprocess_stream, preprocess_shared_kv
from .cute_backward import (
    compile_attention_backward,
    compile_cast_gradient,
    compile_compression_backward,
    compile_local_norm_backward,
    compile_pack_dsa_kv_indices,
    compile_prepare_dsa_backward,
    compile_sink_reduce,
    compile_unpack_dsa_gradients,
)


_HEAD_DIM = 512
_INDEX_DIM = 64
_COMPRESSION_RATE = 32
_ROPE_DIMS = 64
_query_streams: dict[int, torch.cuda.Stream] = {}


def _query_stream(device_index: int) -> torch.cuda.Stream:
    stream = _query_streams.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device_index)
        _query_streams[device_index] = stream
    return stream


def _apply_cudnn_dsa_compatibility() -> None:
    """Adapt the bundled cuDNN DSA kernel to the current CuTe DSL APIs.

    cuDNN 9.13 still uses several CuTe spellings removed by newer CUTLASS DSL builds.
    Keep the adaptation at our lazy integration boundary instead of changing the
    system package.
    """
    import cutlass.cute as cute
    import cutlass.cute.nvgpu as nvgpu
    from cutlass.cute.nvgpu import tcgen05
    import cutlass.utils.blackwell_helpers as sm100_utils

    # Access through __dict__ avoids invoking tcgen05's deprecated-attribute hook.
    tcgen05.__dict__.setdefault("OperandMajorMode", nvgpu.OperandMajorMode)

    scalar_data = cute.struct._ScalarData
    scalar_value = scalar_data.__dict__["value"]
    if not getattr(scalar_value.fget, "_attention_gym_compatible", False):

        def explicit_scalar_pointer_value(self):
            return self.ptr.value

        explicit_scalar_pointer_value._attention_gym_compatible = True
        scalar_data.value = property(explicit_scalar_pointer_value)

    make_mma = sm100_utils.make_trivial_tiled_mma
    if not getattr(make_mma, "_attention_gym_compatible", False):

        def make_mma_compatible(*args, **kwargs):
            if len(args) in (6, 7) and not kwargs:
                # Legacy API supplied one dtype shared by the A and B operands.
                args = (args[0], args[0], *args[1:])
            return make_mma(*args, **kwargs)

        make_mma_compatible._attention_gym_compatible = True
        sm100_utils.make_trivial_tiled_mma = make_mma_compatible


def _require_sm100(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "The CuTe compressed sparse attention backend targets SM100 exclusively; "
            f"device {device} has compute capability {capability[0]}.{capability[1]}."
        )


@lru_cache(maxsize=None)
def _rope_tables(
    device_index: int, sequence_length: int, rope_dims: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache the oracle's FP32 YaRN cosine/sine tables for non-preprocessing kernels."""
    device = torch.device("cuda", device_index)
    with torch.cuda.device(device):
        pair_positions = torch.arange(0, rope_dims, 2, device=device, dtype=torch.float32)
        frequencies = 1.0 / (160_000.0 ** (pair_positions / rope_dims))
        correction_scale = 2 * math.log(160_000.0)
        low = math.floor(
            rope_dims * math.log(65_536 / (32.0 * 2 * math.pi)) / correction_scale
        )
        high = math.ceil(
            rope_dims * math.log(65_536 / (1.0 * 2 * math.pi)) / correction_scale
        )
        low = max(low, 0)
        high = min(high, rope_dims - 1)
        if low == high:
            high += 0.001
        ramp = (
            torch.arange(rope_dims // 2, device=device, dtype=torch.float32) - low
        ) / (high - low)
        smooth = 1 - ramp.clamp(0, 1)
        frequencies = frequencies / 16.0 * (1 - smooth) + frequencies * smooth
        positions = torch.arange(sequence_length, device=device, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        return angles.cos(), angles.sin()


def _validate_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}.")
    if tensor.device != device or tensor.dtype != dtype:
        raise ValueError(f"{name} must use device={device} and dtype={dtype}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def _validate_configuration(
    Q: torch.Tensor,
    Q_I: torch.Tensor,
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    W_I: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> tuple[int, int, int, int, int]:
    if not Q.is_cuda:
        raise ValueError("The CuTe backend requires CUDA tensors.")
    _require_sm100(Q.device)
    if Q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("The CuTe backend supports bfloat16 and float16 inputs.")
    if Q.ndim != 4:
        raise ValueError("Q must have shape [B, H, S, D].")
    batch, heads, sequence_length, head_dim = Q.shape
    if batch <= 0 or heads <= 0 or sequence_length <= 0:
        raise ValueError("B, H, and S must be positive.")
    if sequence_length >= 2**31:
        raise ValueError("The CuTe backend requires S < 2**31.")
    if head_dim != _HEAD_DIM:
        raise ValueError("The SM100 CuTe specialization requires D=512.")
    if Q_I.ndim != 4:
        raise ValueError("Q_I must have shape [B, HI, S, DI].")
    index_heads, index_dim = Q_I.shape[1], Q_I.shape[3]
    if index_heads <= 0 or index_dim <= 0 or index_dim % 64:
        raise ValueError("HI must be positive and DI must be a positive multiple of 64.")
    if not share_kv:
        raise ValueError("The SM100 CuTe specialization requires share_kv=True.")
    if compression_rate <= 0:
        raise ValueError("compression_rate must be positive.")
    if num_topk_blocks < 0:
        raise ValueError("num_topk_blocks must be nonnegative.")
    if sliding_window_size < 0:
        raise ValueError("sliding_window_size must be nonnegative.")
    if rope_dims <= 0 or rope_dims % 2 or rope_dims > min(head_dim, index_dim):
        raise ValueError("rope_dims must be positive, even, and at most min(D, DI).")

    device, dtype = Q.device, Q.dtype
    attention_shape = (batch, 1, sequence_length, head_dim)
    index_shape = (batch, 1, sequence_length, index_dim)
    expected = (
        ("Q", Q, (batch, heads, sequence_length, head_dim)),
        ("Q_I", Q_I, (batch, index_heads, sequence_length, index_dim)),
        ("KV", KV, attention_shape),
        ("C_a", C_a, attention_shape),
        ("C_b", C_b, attention_shape),
        ("Z_a", Z_a, attention_shape),
        ("Z_b", Z_b, attention_shape),
        ("B_a", B_a, (compression_rate, head_dim)),
        ("B_b", B_b, (compression_rate, head_dim)),
        ("W_I", W_I, (batch, sequence_length, index_heads)),
        ("K_Ia", K_Ia, index_shape),
        ("K_Ib", K_Ib, index_shape),
        ("Z_Ia", Z_Ia, index_shape),
        ("Z_Ib", Z_Ib, index_shape),
        ("B_Ia", B_Ia, (compression_rate, index_dim)),
        ("B_Ib", B_Ib, (compression_rate, index_dim)),
        ("KV_norm_weight", KV_norm_weight, (head_dim,)),
        (
            "compressed_indices_norm_weight",
            compressed_indices_norm_weight,
            (index_dim,),
        ),
        ("compressed_kv_norm_weight", compressed_kv_norm_weight, (head_dim,)),
        ("attention_sink", attention_sink, (heads,)),
    )
    for name, tensor, shape in expected:
        _validate_tensor(name, tensor, shape, device=device, dtype=dtype)
    return batch, heads, sequence_length, index_heads, index_dim


def _compressed_sparse_attention_forward(
    Q: torch.Tensor,
    Q_I: torch.Tensor,
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    W_I: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    _return_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run the full shared-KV CSA forward pass with SM100 CuTe DSL kernels."""
    batch, heads, sequence_length, index_heads, index_dim = _validate_configuration(
        Q,
        Q_I,
        KV,
        C_a,
        C_b,
        Z_a,
        Z_b,
        B_a,
        B_b,
        W_I,
        K_Ia,
        K_Ib,
        Z_Ia,
        Z_Ib,
        B_Ia,
        B_Ib,
        KV_norm_weight,
        compressed_indices_norm_weight,
        compressed_kv_norm_weight,
        attention_sink,
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
    )
    num_blocks = math.ceil(sequence_length / compression_rate)
    effective_topk = min(num_topk_blocks, num_blocks)
    dtype = cute_dtype(Q)
    device_index = Q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cos, sin = _rope_tables(device_index, sequence_length, rope_dims)
    gather_length = math.ceil(effective_topk / 128) * 128
    gather = None
    if effective_topk:
        gather = torch.empty(
            batch, sequence_length, gather_length, device=Q.device, dtype=torch.int32
        )

    use_specialized_preprocess = (
        index_dim == _INDEX_DIM
        and compression_rate == _COMPRESSION_RATE
        and rope_dims == _ROPE_DIMS
        and sequence_length % _COMPRESSION_RATE == 0
    )

    prefetched_query = None
    prefetched_local_query = None
    query_torch_stream = None
    use_query_overlap = batch == 1 and sequence_length >= 1024
    if use_query_overlap:
        active_heads = min(128, heads)
        local_tile_heads = 64 if active_heads <= 64 else 128
        prefetched_query = torch.empty(
            batch, sequence_length, 128, _HEAD_DIM, device=Q.device, dtype=Q.dtype
        )
        prefetched_local_query = prefetched_query
        if local_tile_heads != 128:
            prefetched_local_query = torch.empty(
                batch,
                sequence_length,
                local_tile_heads,
                _HEAD_DIM,
                device=Q.device,
                dtype=Q.dtype,
            )
        caller_stream = torch.cuda.current_stream(Q.device)
        query_torch_stream = _query_stream(device_index)
        query_torch_stream.wait_stream(caller_stream)
        with torch.cuda.stream(query_torch_stream):
            compile_query_rope(
                dtype,
                batch,
                heads,
                128,
                active_heads,
                0,
                sequence_length,
                _HEAD_DIM,
                rope_dims,
                local_tile_heads if local_tile_heads != 128 else 0,
            )(Q, cos, sin, prefetched_query, prefetched_local_query)

    if use_specialized_preprocess:
        local_kv, compressed_kv, compressed_indices = preprocess_shared_kv(
            KV,
            C_a,
            C_b,
            Z_a,
            Z_b,
            B_a,
            B_b,
            K_Ia,
            K_Ib,
            Z_Ia,
            Z_Ib,
            B_Ia,
            B_Ib,
            KV_norm_weight,
            compressed_indices_norm_weight,
            compressed_kv_norm_weight,
            compression_rate=compression_rate,
            rope_dims=rope_dims,
        )
    else:
        local_kv = torch.empty(
            batch, sequence_length, 1, _HEAD_DIM, device=Q.device, dtype=Q.dtype
        )
        compressed_kv = torch.empty(
            batch, num_blocks, 1, _HEAD_DIM, device=Q.device, dtype=Q.dtype
        )
        compressed_indices = torch.empty(
            batch, num_blocks, 1, index_dim, device=Q.device, dtype=Q.dtype
        )
        compile_local_norm(
            dtype, batch, sequence_length, _HEAD_DIM, rope_dims, 0
        )(KV, KV_norm_weight, cos, sin, local_kv)
        compile_compression(
            dtype, batch, sequence_length, _HEAD_DIM, compression_rate, rope_dims
        )(
            C_a,
            C_b,
            Z_a,
            Z_b,
            B_a,
            B_b,
            compressed_kv_norm_weight,
            cos,
            sin,
            compressed_kv,
        )
        compile_compression(
            dtype, batch, sequence_length, index_dim, compression_rate, rope_dims
        )(
            K_Ia,
            K_Ib,
            Z_Ia,
            Z_Ib,
            B_Ia,
            B_Ib,
            compressed_indices_norm_weight,
            cos,
            sin,
            compressed_indices,
        )

    output = None
    local_lse_state = None
    compressed_lse_state = None
    if _return_state:
        local_lse_state = torch.full(
            (batch, sequence_length, heads),
            -torch.inf,
            device=Q.device,
            dtype=torch.float32,
        )
        compressed_lse_state = torch.full_like(local_lse_state, -torch.inf)

    index_torch_stream = None
    if effective_topk:
        index_torch_stream = (
            parallel_preprocess_stream(KV) if use_specialized_preprocess else None
        )
        index_topk = compile_index_topk(
            dtype,
            batch,
            sequence_length,
            index_heads,
            index_dim,
            num_blocks,
            compression_rate,
            effective_topk,
            rope_dims,
            gather_length,
        )
        if index_torch_stream is not None:
            with torch.cuda.stream(index_torch_stream):
                index_topk(Q_I, compressed_indices, W_I, cos, sin, gather)
        else:
            index_topk(Q_I, compressed_indices, W_I, cos, sin, gather)

    prefix = min(16, sliding_window_size, sequence_length, compression_rate - 1)
    for head_offset in range(0, heads, 128):
        active_heads = min(128, heads - head_offset)
        tile_heads = 128
        local_tile_heads = 64 if active_heads <= 64 else 128
        if head_offset == 0 and prefetched_query is not None:
            assert prefetched_local_query is not None and query_torch_stream is not None
            torch.cuda.current_stream(Q.device).wait_stream(query_torch_stream)
            query = prefetched_query
            local_query = prefetched_local_query
        else:
            query = torch.empty(
                batch,
                sequence_length,
                tile_heads,
                _HEAD_DIM,
                device=Q.device,
                dtype=Q.dtype,
            )
            local_query = query
            if local_tile_heads != tile_heads:
                local_query = torch.empty(
                    batch,
                    sequence_length,
                    local_tile_heads,
                    _HEAD_DIM,
                    device=Q.device,
                    dtype=Q.dtype,
                )
            compile_query_rope(
                dtype,
                batch,
                heads,
                tile_heads,
                active_heads,
                head_offset,
                sequence_length,
                _HEAD_DIM,
                rope_dims,
                local_tile_heads if local_tile_heads != tile_heads else 0,
            )(Q, cos, sin, query, local_query)
        compressed_output = torch.empty_like(query)
        local_output = torch.empty_like(local_query)
        local_lse = torch.empty(
            batch,
            sequence_length,
            local_tile_heads,
            device=Q.device,
            dtype=torch.float32,
        )
        compressed_lse = torch.empty(
            batch, sequence_length, tile_heads, device=Q.device, dtype=torch.float32
        )
        if sliding_window_size:
            local_attention(
                local_query,
                local_kv,
                min(sliding_window_size, sequence_length),
                output=local_output,
                lse=local_lse,
            )
        if effective_topk and sliding_window_size:
            assert gather is not None
            if index_torch_stream is not None:
                torch.cuda.current_stream(Q.device).wait_stream(index_torch_stream)
                index_torch_stream = None
            compressed_attention(
                query,
                compressed_kv,
                gather,
                output=compressed_output,
                lse=compressed_lse,
            )
        if _return_state:
            assert local_lse_state is not None and compressed_lse_state is not None
            if sliding_window_size:
                local_lse_state[:, :, head_offset : head_offset + active_heads].copy_(
                    local_lse[:, :, :active_heads]
                )
            if effective_topk:
                compressed_lse_state[
                    :, :, head_offset : head_offset + active_heads
                ].copy_(compressed_lse[:, :, :active_heads])
        prefix_query = local_query[:, :prefix].contiguous() if prefix else None
        if output is None:
            output = torch.empty_like(Q)
        if effective_topk and not sliding_window_size:
            assert gather is not None
            if index_torch_stream is not None:
                torch.cuda.current_stream(Q.device).wait_stream(index_torch_stream)
                index_torch_stream = None
            compile_compressed_merge(
                dtype,
                batch,
                heads,
                tile_heads,
                active_heads,
                head_offset,
                sequence_length,
                _HEAD_DIM,
                rope_dims,
                num_blocks,
                effective_topk,
                gather_length,
            )(
                query,
                compressed_kv,
                gather,
                compressed_output,
                compressed_lse,
                attention_sink,
                cos,
                sin,
                output,
            )
        else:
            compile_merge(
                dtype,
                batch,
                heads,
                local_tile_heads,
                tile_heads,
                active_heads,
                head_offset,
                sequence_length,
                _HEAD_DIM,
                rope_dims,
                sliding_window_size > 0,
                effective_topk > 0,
            )(
                local_output,
                local_lse,
                compressed_output,
                compressed_lse,
                attention_sink,
                cos,
                sin,
                output,
            )
        del query, local_query

        # Before q = R - 1 no compressed block has finished. Replaying only that true prefix
        # with the oracle's BF16/FP16 boundaries removes its sharp local-only outlier.
        if prefix:
            compile_prefix(
                dtype,
                batch,
                heads,
                local_tile_heads,
                active_heads,
                head_offset,
                sequence_length,
                _HEAD_DIM,
                rope_dims,
                prefix,
            )(
                prefix_query,
                local_kv,
                attention_sink,
                cos,
                sin,
                output,
            )
    assert output is not None
    if _return_state:
        gather_state = gather
        if gather_state is None:
            gather_state = torch.empty(
                batch, sequence_length, 1, device=Q.device, dtype=torch.int32
            )
        assert local_lse_state is not None and compressed_lse_state is not None
        return output, (
            local_kv,
            compressed_kv,
            gather_state,
            cos,
            sin,
            local_lse_state,
            compressed_lse_state,
        )
    return output


class _CuteCompressedSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I,
        K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib,
        KV_norm_weight, compressed_indices_norm_weight,
        compressed_kv_norm_weight, attention_sink,
        compression_rate, num_topk_blocks, sliding_window_size, rope_dims, share_kv,
    ):
        forward_tensors = tuple(
            tensor.detach()
            for tensor in (
                Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I,
                K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib,
                KV_norm_weight, compressed_indices_norm_weight,
                compressed_kv_norm_weight, attention_sink,
            )
        )
        output = _compressed_sparse_attention_forward(
            *forward_tensors,
            compression_rate, num_topk_blocks, sliding_window_size, rope_dims, share_kv,
        )
        ctx.save_for_backward(
            Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I,
            K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib,
            KV_norm_weight, compressed_indices_norm_weight,
            compressed_kv_norm_weight, attention_sink,
        )
        ctx.compression_rate = compression_rate
        ctx.num_topk_blocks = num_topk_blocks
        ctx.sliding_window_size = sliding_window_size
        ctx.rope_dims = rope_dims
        ctx.share_kv = share_kv
        return output

    @staticmethod
    def backward(ctx, dout):
        (
            Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I,
            K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib,
            KV_norm_weight, compressed_indices_norm_weight,
            compressed_kv_norm_weight, attention_sink,
        ) = ctx.saved_tensors
        dout = dout.contiguous()
        batch, heads, sequence, dim = Q.shape
        recompute_tensors = tuple(tensor.detach() for tensor in ctx.saved_tensors)
        output, state = _compressed_sparse_attention_forward(
            *recompute_tensors,
            ctx.compression_rate,
            ctx.num_topk_blocks,
            ctx.sliding_window_size,
            ctx.rope_dims,
            ctx.share_kv,
            _return_state=True,
        )
        (
            local_kv, compressed_kv, gather, cos, sin,
            local_lse, compressed_lse,
        ) = state
        blocks = compressed_kv.shape[1]
        topk = min(ctx.num_topk_blocks, blocks)
        dtype = cute_dtype(Q)
        dsa_dtype = BFloat16

        dQ = torch.empty_like(Q)
        dlocal = torch.zeros_like(local_kv, dtype=torch.float32)
        dcompressed = torch.zeros_like(compressed_kv, dtype=torch.float32)
        has_attention = topk > 0 or ctx.sliding_window_size > 0
        if has_attention:
            _apply_cudnn_dsa_compatibility()
            from cudnn.deepseek_sparse_attention import sparse_attention_backward_wrapper

            tokens = batch * sequence
            window = min(ctx.sliding_window_size, sequence)
            index_width = math.ceil((topk + window) / 64) * 64
            kv_packed = torch.empty(
                batch * (blocks + sequence), dim,
                device=Q.device,
                dtype=torch.bfloat16,
            )
            indices = torch.empty(
                tokens, index_width, device=Q.device, dtype=torch.int32
            )
            sink_fp32 = torch.empty(heads, device=Q.device, dtype=torch.float32)
            compile_pack_dsa_kv_indices(
                dtype, dsa_dtype, batch, sequence, dim, blocks, topk, window,
                index_width, heads,
            )(
                compressed_kv, local_kv, gather, kv_packed, indices,
                attention_sink, sink_fp32,
            )
            d_attention_sink = torch.empty_like(attention_sink)
            for head_offset in range(0, heads, 64):
                packed_heads = min(64, heads - head_offset)
                q_packed = torch.empty(
                    tokens, packed_heads, dim, device=Q.device, dtype=torch.bfloat16
                )
                out_packed = torch.empty_like(q_packed)
                dout_packed = torch.empty_like(q_packed)
                lse_packed = torch.empty(
                    tokens, packed_heads, device=Q.device, dtype=torch.float32
                )
                compile_prepare_dsa_backward(
                    dtype, dsa_dtype, batch, heads, packed_heads, head_offset,
                    sequence, dim, ctx.rope_dims,
                )(
                    Q, output, dout, local_lse, compressed_lse, cos, sin,
                    q_packed, out_packed, dout_packed, lse_packed,
                )
                result = sparse_attention_backward_wrapper(
                    q_packed,
                    kv_packed,
                    out_packed,
                    dout_packed,
                    lse_packed,
                    sink_fp32[head_offset : head_offset + packed_heads],
                    indices,
                    softmax_scale=1.0 / math.sqrt(dim),
                )
                compile_unpack_dsa_gradients(
                    dtype, dsa_dtype, batch, heads, packed_heads, head_offset,
                    sequence, dim, ctx.rope_dims, blocks,
                )(
                    result["dq"], result["dkv"], cos, sin,
                    dQ, dlocal, dcompressed,
                )
                compile_cast_gradient(dtype, packed_heads)(
                    result["d_sink"],
                    d_attention_sink[head_offset : head_offset + packed_heads],
                )
                del q_packed, out_packed, dout_packed, lse_packed, result
            del (
                kv_packed,
                indices,
                sink_fp32,
                output,
                local_lse,
                compressed_lse,
                gather,
            )
        else:
            dQ.zero_()
            dlocal.zero_()
            dcompressed.zero_()
            d_attention_sink = torch.zeros_like(attention_sink)
            del output, local_lse, compressed_lse, gather

        dKV = torch.empty_like(KV)
        dKV_weight_fp32 = torch.zeros_like(KV_norm_weight, dtype=torch.float32)
        compile_local_norm_backward(dtype, batch, sequence, dim, ctx.rope_dims)(
            KV, KV_norm_weight, cos, sin, dlocal, dKV, dKV_weight_fp32
        )

        dC_a = torch.zeros_like(C_a)
        dC_b = torch.zeros_like(C_b)
        dZ_a = torch.zeros_like(Z_a)
        dZ_b = torch.zeros_like(Z_b)
        dB_a_fp32 = torch.zeros_like(B_a, dtype=torch.float32)
        dB_b_fp32 = torch.zeros_like(B_b, dtype=torch.float32)
        dcompressed_weight_fp32 = torch.zeros_like(
            compressed_kv_norm_weight, dtype=torch.float32
        )
        compile_compression_backward(
            dtype, batch, sequence, dim, ctx.compression_rate, ctx.rope_dims
        )(
            C_a, C_b, Z_a, Z_b, B_a, B_b, compressed_kv_norm_weight,
            cos, sin, dcompressed, dC_a, dC_b, dZ_a, dZ_b,
            dB_a_fp32, dB_b_fp32, dcompressed_weight_fp32,
        )

        dB_a = torch.empty_like(B_a)
        dB_b = torch.empty_like(B_b)
        dKV_weight = torch.empty_like(KV_norm_weight)
        dcompressed_weight = torch.empty_like(compressed_kv_norm_weight)
        compile_cast_gradient(dtype, B_a.numel())(
            dB_a_fp32.view(-1), dB_a.view(-1)
        )
        compile_cast_gradient(dtype, B_b.numel())(
            dB_b_fp32.view(-1), dB_b.view(-1)
        )
        compile_cast_gradient(dtype, KV_norm_weight.numel())(
            dKV_weight_fp32, dKV_weight
        )
        compile_cast_gradient(dtype, compressed_kv_norm_weight.numel())(
            dcompressed_weight_fp32, dcompressed_weight
        )

        grads = (
            dQ, None, dKV, dC_a, dC_b, dZ_a, dZ_b, dB_a, dB_b, None,
            None, None, None, None, None, None,
            dKV_weight, None, dcompressed_weight, d_attention_sink,
            None, None, None, None, None,
        )
        return tuple(g if need else None for g, need in zip(grads, ctx.needs_input_grad))


def compressed_sparse_attention(
    Q: torch.Tensor,
    Q_I: torch.Tensor,
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    W_I: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    _return_state: bool = False,
):
    args = (
        Q, Q_I, KV, C_a, C_b, Z_a, Z_b, B_a, B_b, W_I,
        K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib,
        KV_norm_weight, compressed_indices_norm_weight,
        compressed_kv_norm_weight, attention_sink,
        compression_rate, num_topk_blocks, sliding_window_size, rope_dims, share_kv,
    )
    if _return_state:
        return _compressed_sparse_attention_forward(*args, _return_state=True)
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in args[:20] if isinstance(tensor, torch.Tensor)
    ):
        return _CuteCompressedSparseAttention.apply(*args)
    return _compressed_sparse_attention_forward(*args)


__all__ = ["compressed_sparse_attention"]
