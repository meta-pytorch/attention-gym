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

from .kernels import (
    compile_compression,
    compile_index_scores,
    compile_index_topk,
    compile_local_norm,
    compile_merge,
    compile_prefix,
    compile_query_rope,
    compile_selected_gather,
    cute_dtype,
)
from .local_attention import compressed_attention
from .preprocess import parallel_preprocess_stream, preprocess_shared_kv
from .backward import (
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
_RADIX_TOPK_THRESHOLD = 64
_RADIX_TOPK_MAX = 2048
_RADIX_TOPK_MAX_BLOCKS = 8192
_query_streams: dict[int, torch.cuda.Stream] = {}
_DSA_PACKED_WORKSPACE_BYTES = 1536 * 1024 * 1024


def _query_stream(device_index: int) -> torch.cuda.Stream:
    stream = _query_streams.get(device_index)
    if stream is None:
        stream = torch.cuda.Stream(device=device_index)
        _query_streams[device_index] = stream
    return stream


def _radix_topk_indices(
    score_keys: torch.Tensor,
    completed_lengths: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select large top-k sets with cuDNN's cooperative SM90+ radix kernel."""

    try:
        from cudnn.deepseek_sparse_attention.indexer_top_k import indexer_top_k_wrapper
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "CuTe compressed sparse attention with topk >= 64 requires the cuDNN "
            "frontend radix top-k kernels."
        ) from error

    result = indexer_top_k_wrapper(
        score_keys,
        completed_lengths,
        topk,
        next_n=1,
        return_val=False,
    )
    return result["indices"]


def _dsa_head_chunk(tokens: int, dim: int, heads: int) -> int:
    """Choose a 64-aligned head chunk under a bounded packed-workspace budget."""
    bytes_per_head = tokens * (5 * dim * 2 + 2 * 4)
    budget_heads = _DSA_PACKED_WORKSPACE_BYTES // bytes_per_head
    aligned_heads = max(64, budget_heads // 64 * 64)
    return min(128, max(64, math.ceil(min(heads, aligned_heads) / 64) * 64))


def _apply_dsa_pointer_compatibility() -> None:
    """Resolve the vendored kernel's legacy struct scalar through its explicit pointer."""
    import cutlass.cute as cute

    scalar_data = cute.struct._ScalarData
    scalar_value = scalar_data.__dict__["value"]
    if not getattr(scalar_value.fget, "_attention_gym_compatible", False):

        def explicit_scalar_pointer_value(self):
            return self.ptr.value

        explicit_scalar_pointer_value._attention_gym_compatible = True
        scalar_data.value = property(explicit_scalar_pointer_value)


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
    share_kv: bool,
) -> tuple[int, int, int, int, int]:
    if not Q.is_cuda:
        raise ValueError("The CuTe backend requires CUDA tensors.")
    _require_sm100(Q.device)
    if Q.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("The CuTe backend supports bfloat16 and float16 inputs.")
    batch, heads, sequence_length, head_dim = Q.shape
    if sequence_length >= 2**31:
        raise ValueError("The CuTe backend requires S < 2**31.")
    if head_dim != _HEAD_DIM:
        raise ValueError("The SM100 CuTe specialization requires D=512.")
    index_heads, index_dim = Q_I.shape[1], Q_I.shape[3]
    if index_dim % 64:
        raise ValueError("The SM100 CuTe specialization requires DI to be a multiple of 64.")
    if not share_kv:
        raise ValueError("The SM100 CuTe specialization requires share_kv=True.")

    shared_attention_inputs = (
        ("KV", KV),
        ("C_a", C_a),
        ("C_b", C_b),
        ("Z_a", Z_a),
        ("Z_b", Z_b),
    )
    shared_index_inputs = (
        ("K_Ia", K_Ia),
        ("K_Ib", K_Ib),
        ("Z_Ia", Z_Ia),
        ("Z_Ib", Z_Ib),
    )
    for name, tensor in (*shared_attention_inputs, *shared_index_inputs):
        if tensor.shape[1] != 1:
            raise ValueError(
                f"{name} must physically have one KV head for the SM100 CuTe backend."
            )

    tensors = (
        ("Q", Q),
        ("Q_I", Q_I),
        *shared_attention_inputs,
        ("B_a", B_a),
        ("B_b", B_b),
        ("W_I", W_I),
        *shared_index_inputs,
        ("B_Ia", B_Ia),
        ("B_Ib", B_Ib),
        ("KV_norm_weight", KV_norm_weight),
        ("compressed_indices_norm_weight", compressed_indices_norm_weight),
        ("compressed_kv_norm_weight", compressed_kv_norm_weight),
        ("attention_sink", attention_sink),
    )
    for name, tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the CuTe backend.")
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
        share_kv,
    )
    num_blocks = math.ceil(sequence_length / compression_rate)
    effective_topk = min(num_topk_blocks, num_blocks)
    dtype = cute_dtype(Q)
    device_index = Q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cos, sin = _rope_tables(device_index, sequence_length, rope_dims)
    selected_window = min(sliding_window_size, sequence_length)
    selected_width = effective_topk + selected_window
    gather_length = math.ceil(selected_width / 128) * 128
    gather = None
    if selected_width:
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
    score_keys = None
    completed_lengths = None
    if selected_width:
        index_torch_stream = (
            parallel_preprocess_stream(KV)
            if use_specialized_preprocess and effective_topk
            else None
        )
        use_radix_topk = (
            effective_topk >= _RADIX_TOPK_THRESHOLD
            and effective_topk <= _RADIX_TOPK_MAX
            and num_blocks <= _RADIX_TOPK_MAX_BLOCKS
        )

        def launch_indexer() -> None:
            nonlocal score_keys, completed_lengths
            if use_radix_topk:
                score_keys = torch.empty(
                    batch * sequence_length,
                    num_blocks,
                    device=Q.device,
                    dtype=torch.float32,
                )
                completed_lengths = torch.empty(
                    batch * sequence_length,
                    device=Q.device,
                    dtype=torch.int32,
                )
                compile_index_scores(
                    dtype,
                    batch,
                    sequence_length,
                    index_heads,
                    index_dim,
                    num_blocks,
                    compression_rate,
                    rope_dims,
                )(
                    Q_I,
                    compressed_indices,
                    W_I,
                    cos,
                    sin,
                    score_keys,
                    completed_lengths,
                )
                selected_indices = _radix_topk_indices(
                    score_keys,
                    completed_lengths,
                    effective_topk,
                )
                compile_selected_gather(
                    batch,
                    sequence_length,
                    num_blocks,
                    effective_topk,
                    selected_window,
                    gather_length,
                )(selected_indices, gather)
            else:
                index_topk = compile_index_topk(
                    dtype,
                    batch,
                    sequence_length,
                    index_heads,
                    index_dim,
                    num_blocks,
                    compression_rate,
                    effective_topk,
                    selected_window,
                    rope_dims,
                    gather_length,
                )
                index_topk(Q_I, compressed_indices, W_I, cos, sin, gather)

        if index_torch_stream is not None:
            with torch.cuda.stream(index_torch_stream):
                launch_indexer()
        else:
            launch_indexer()

    # The first few very short reductions have the sharpest BF16 ordering error.
    # Replaying through row 9 covers the validated B=1 and B=4 short-/long-context
    # outliers while avoiding the former 16-row launch.
    prefix = min(10, sliding_window_size, sequence_length, compression_rate - 1)
    for head_offset in range(0, heads, 128):
        active_heads = min(128, heads - head_offset)
        tile_heads = 128
        local_tile_heads = 128
        fuse_q_rope = selected_width > 0 and active_heads == 128
        if fuse_q_rope:
            query = Q[:, head_offset : head_offset + active_heads].permute(0, 2, 1, 3)
            local_query = query
        elif head_offset == 0 and prefetched_query is not None:
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
        if output is None:
            output = torch.empty_like(Q)
        fuse_attention_epilogue = selected_width > 0 and active_heads == 128
        if fuse_attention_epilogue:
            selected_output = output[
                :, head_offset : head_offset + active_heads
            ].permute(0, 2, 1, 3)
        else:
            selected_output = torch.empty_like(query)
        store_selected_lse = _return_state or not fuse_attention_epilogue
        selected_lse = (
            torch.empty(
                batch,
                sequence_length,
                tile_heads,
                device=Q.device,
                dtype=torch.float32,
            )
            if store_selected_lse
            else None
        )
        if selected_width:
            assert gather is not None
            if index_torch_stream is not None:
                torch.cuda.current_stream(Q.device).wait_stream(index_torch_stream)
                index_torch_stream = None
            compressed_attention(
                query,
                compressed_kv,
                gather,
                local_value=local_kv,
                output=selected_output,
                lse=selected_lse,
                sink=attention_sink if fuse_attention_epilogue else None,
                cos=cos if fuse_attention_epilogue else None,
                sin=sin if fuse_attention_epilogue else None,
                head_offset=head_offset,
                fuse_q_rope=fuse_q_rope,
                rope_dims=rope_dims,
                csa_topk=effective_topk,
                csa_window=selected_window,
                store_lse=store_selected_lse,
            )
        if _return_state:
            assert local_lse_state is not None and compressed_lse_state is not None
            if selected_width:
                assert selected_lse is not None
                local_lse_state[:, :, head_offset : head_offset + active_heads].copy_(
                    selected_lse[:, :, :active_heads]
                )
        prefix_query = None
        if prefix:
            prefix_query = Q if fuse_q_rope else local_query[:, :prefix].contiguous()
        if not fuse_attention_epilogue:
            assert selected_lse is not None
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
                selected_width > 0,
                False,
            )(
                selected_output,
                selected_lse,
                selected_output,
                selected_lse,
                attention_sink,
                cos,
                sin,
                output,
            )
        del query, local_query

        # Before q = R - 1 no compressed block has finished. Replaying only that true prefix
        # with the oracle's BF16/FP16 boundaries removes its sharp local-only outlier.
        if prefix_query is not None:
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
                fuse_q_rope,
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
            _apply_dsa_pointer_compatibility()
            from .dsa_backward_sm100 import sparse_attention_backward_wrapper

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
            topk_lengths = torch.empty(tokens, device=Q.device, dtype=torch.int32)
            sink_fp32 = torch.empty(heads, device=Q.device, dtype=torch.float32)
            compile_pack_dsa_kv_indices(
                dtype, dsa_dtype, batch, sequence, dim, blocks,
                ctx.compression_rate, topk, window, index_width, heads,
                gather.shape[-1],
            )(
                compressed_kv, local_kv, gather, kv_packed, indices, topk_lengths,
                attention_sink, sink_fp32,
            )
            d_attention_sink = torch.empty_like(attention_sink)
            head_chunk = _dsa_head_chunk(tokens, dim, heads)
            for head_offset in range(0, heads, head_chunk):
                packed_heads = min(head_chunk, heads - head_offset)
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
                    topk_length=topk_lengths,
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
                topk_lengths,
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
