"""SM100 CuTe DSL backend for shared-KV compressed sparse attention.

The public entry point owns the complete CSA data path: compression, RMS normalization,
RoPE, index scoring/top-k, local and compressed attention, learned-sink merge, and inverse
RoPE.  PyTorch is used only for validation, allocation, and cached RoPE constants; input-
dependent tensor arithmetic is launched through CuTe DSL kernels.
"""

from __future__ import annotations

from collections import OrderedDict
import math

import torch
from cutlass import BFloat16, Int32

from .index_scores import exact_bf16_index_scores
from .kernels import (
    compile_causal_gather,
    compile_compression,
    compile_index_score_keys,
    compile_index_scores,
    compile_index_topk,
    compile_local_norm,
    compile_merge,
    compile_pad_index_weights,
    compile_prefix,
    compile_query_rope,
    compile_selected_gather,
    cute_dtype,
)
from .local_attention import compressed_attention, local_attention
from .preprocess import parallel_preprocess_stream, preprocess_shared_kv
from .backward import (
    compile_attention_backward,
    compile_cast_gradient,
    compile_compression_backward,
    compile_local_norm_backward,
    compile_pack_dsa_indices,
    compile_pack_dsa_kv_sink,
    compile_prepare_dsa_backward,
    compile_sink_reduce,
    compile_unpack_dsa_gradients,
)


_HEAD_DIM = 512
_INDEX_DIM = 64
_COMPRESSION_RATE = 32
_ROPE_DIMS = 64
_RADIX_TOPK_THRESHOLD = 64
_RADIX_SCORE_WORKSPACE_BYTES = 128 * 1024 * 1024
_COMBINED_GATHER_WORKSPACE_BYTES = 128 * 1024 * 1024
_TENSOR_INDEX_PACKED_HEADS = 64
_TESTED_CUDA_VERSION = "13.3"
_DIFFERENTIABLE_INPUT_INDICES = (0, 2, 3, 4, 5, 6, 7, 8, 16, 18, 19)
_query_streams: dict[int, torch.cuda.Stream] = {}
_ROPE_CACHE_MAXSIZE = 8
_rope_table_cache: OrderedDict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]
] = OrderedDict()
_DSA_PACKED_WORKSPACE_BYTES = 1536 * 1024 * 1024


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


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

    with torch.cuda.device(score_keys.device):
        result = indexer_top_k_wrapper(
            score_keys,
            completed_lengths,
            topk,
            next_n=1,
            return_val=False,
        )
    return result["indices"]


def _radix_score_row_chunk(total_rows: int, num_blocks: int, topk: int) -> int:
    """Bound known radix allocations while keeping full rows for exact selection."""
    # Besides the FP32 score matrix, cuDNN's selector materializes two INT32 planes
    # with ``num_blocks`` entries and returns one INT32 top-k list per row. Account
    # for all of those tensors (plus the completed-length scalar) when sizing a slab.
    bytes_per_row = num_blocks * (4 + 2 * 4) + topk * 4 + 4
    rows = max(1, _RADIX_SCORE_WORKSPACE_BYTES // bytes_per_row)
    if rows >= 128:
        rows = rows // 128 * 128
    return min(total_rows, rows)


def _gather_storage_bytes(batch: int, sequence: int, width: int) -> int:
    """Return the storage needed for an INT32 gather list."""
    return batch * sequence * width * 4


def _dsa_workspace_bytes(
    tokens: int, dim: int, heads: int, total_kv: int, index_width: int = 0
) -> int:
    """Estimate live packed tensors and vendored DSA workspaces for one chunk."""
    # Q, output, dOutput, and dQ are BF16. Packed LSE is FP32, while the vendored
    # LSE/OdO workspace aligns its two FP32 regions over eight tokens and 64 heads.
    rounded_tokens = math.ceil(tokens / 8) * 8
    rounded_heads = math.ceil(heads / 64) * 64
    rounded_dim = math.ceil(dim / 8) * 8
    head_dependent = (
        tokens * heads * (4 * dim * 2 + 4)
        + rounded_tokens * rounded_heads * 2 * 4
        + heads * (4 + 4 + 2 + 4)
    )
    # Packed KV, dKV, the vendored FP32 reduction workspace, and the persistent
    # FP32 shared-KV gradient accumulators coexist.
    rounded_kv = math.ceil(total_kv / 8) * 8
    kv_dependent = total_kv * dim * (2 + 2 + 4) + rounded_kv * rounded_dim * 4
    sparse_metadata = tokens * (index_width * 4 + 4)
    return head_dependent + kv_dependent + sparse_metadata


def _dsa_head_chunk(tokens: int, dim: int, heads: int, total_kv: int = 0) -> int:
    """Choose the largest head chunk under the packed-workspace budget."""
    fixed_bytes = _dsa_workspace_bytes(tokens, dim, 0, total_kv)
    if fixed_bytes >= _DSA_PACKED_WORKSPACE_BYTES:
        raise RuntimeError(
            "The CuTe backward fixed KV workspace exceeds the "
            f"{_DSA_PACKED_WORKSPACE_BYTES / 2**30:.1f} GiB budget."
        )
    max_heads = min(128, heads)
    if (
        max_heads < 1
        or _dsa_workspace_bytes(tokens, dim, 1, total_kv)
        > _DSA_PACKED_WORKSPACE_BYTES
    ):
        raise RuntimeError(
            "The CuTe backward workspace cannot fit one attention head within "
            f"the {_DSA_PACKED_WORKSPACE_BYTES / 2**30:.1f} GiB budget."
        )

    lower, upper = 1, max_heads
    while lower < upper:
        middle = (lower + upper + 1) // 2
        if (
            _dsa_workspace_bytes(tokens, dim, middle, total_kv)
            <= _DSA_PACKED_WORKSPACE_BYTES
        ):
            lower = middle
        else:
            upper = middle - 1
    return lower


def _dsa_tile_shape(
    tokens: int, dim: int, heads: int, total_kv: int, index_width: int = 0
) -> tuple[int, int]:
    """Choose head/token tiles that minimize launches without exceeding the budget."""
    fixed_bytes = _dsa_workspace_bytes(0, dim, 0, total_kv, index_width)
    if fixed_bytes >= _DSA_PACKED_WORKSPACE_BYTES:
        raise RuntimeError(
            "The CuTe backward fixed KV workspace exceeds the "
            f"{_DSA_PACKED_WORKSPACE_BYTES / 2**30:.1f} GiB budget."
        )

    candidates = {
        min(heads, candidate) for candidate in (128, 64, 32, 16, 8, 4, 2, 1)
    }
    best: tuple[int, int, int] | None = None
    for head_tile in candidates:
        if head_tile <= 0:
            continue
        low, high = 0, tokens
        while low < high:
            middle = (low + high + 1) // 2
            if (
                _dsa_workspace_bytes(
                    middle, dim, head_tile, total_kv, index_width
                )
                <= _DSA_PACKED_WORKSPACE_BYTES
            ):
                low = middle
            else:
                high = middle - 1
        token_tile = low
        if token_tile < tokens:
            alignment = 128 if token_tile >= 128 else 4
            token_tile = token_tile // alignment * alignment
        if token_tile < 1:
            continue
        launches = math.ceil(heads / head_tile) * math.ceil(tokens / token_tile)
        candidate = (launches, -head_tile, -token_tile)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        raise RuntimeError(
            "The CuTe backward workspace cannot fit one query/head tile within "
            f"the {_DSA_PACKED_WORKSPACE_BYTES / 2**30:.1f} GiB budget."
        )
    return -best[1], -best[2]


def _require_sm100(device: torch.device) -> None:
    capability = torch.cuda.get_device_capability(device)
    if capability != (10, 0):
        raise RuntimeError(
            "The CuTe compressed sparse attention backend targets SM100 exclusively; "
            f"device {device} has compute capability {capability[0]}.{capability[1]}."
        )
    if torch.version.cuda != _TESTED_CUDA_VERSION:
        raise RuntimeError(
            "The CuTe compressed sparse attention backend is validated with CUDA "
            f"{_TESTED_CUDA_VERSION}; this PyTorch build uses CUDA {torch.version.cuda}."
        )


def _rope_tables(
    device_index: int, sequence_length: int, rope_dims: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached FP32 YaRN tables with an explicit cross-stream dependency."""
    key = (device_index, sequence_length, rope_dims)
    cached = _rope_table_cache.get(key)
    device = torch.device("cuda", device_index)
    if cached is None:
        with torch.cuda.device(device):
            pair_positions = torch.arange(
                0, rope_dims, 2, device=device, dtype=torch.float32
            )
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
            positions = torch.arange(
                sequence_length, device=device, dtype=torch.float32
            )
            angles = torch.outer(positions, frequencies)
            cos, sin = angles.cos(), angles.sin()
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(device))
        cached = (cos, sin, ready)
        _rope_table_cache[key] = cached
        if len(_rope_table_cache) > _ROPE_CACHE_MAXSIZE:
            _rope_table_cache.popitem(last=False)
    else:
        _rope_table_cache.move_to_end(key)
    cos, sin, ready = cached
    consumer_stream = torch.cuda.current_stream(device)
    consumer_stream.wait_event(ready)
    # Cached tables can be evicted while kernels from another stream still use them.
    # Tell the caching allocator about each consumer before returning the tensors.
    cos.record_stream(consumer_stream)
    sin.record_stream(consumer_stream)
    return cos, sin


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
    if Q.dtype != torch.bfloat16:
        raise TypeError("The CuTe backend supports bfloat16 inputs only.")
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
    selectable_blocks = sequence_length // compression_rate
    effective_topk = min(num_topk_blocks, selectable_blocks)
    selected_window = min(sliding_window_size, sequence_length)
    has_local = selected_window > 0
    has_compressed = effective_topk > 0
    select_all_blocks = has_compressed and effective_topk == selectable_blocks
    need_index_scores = has_compressed and not select_all_blocks
    if need_index_scores and effective_topk > 2048:
        raise ValueError(
            "The current cuDNN radix selector supports at most 2048 partially "
            "selected blocks; choose K >= S // compression_rate to select all "
            "completed blocks without radix top-k."
        )
    if not has_local and not has_compressed:
        output = torch.zeros_like(Q)
        if not _return_state:
            return output
        device_index = Q.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        cos, sin = _rope_tables(device_index, sequence_length, rope_dims)
        empty_kv = torch.empty(
            batch, 0, 1, _HEAD_DIM, device=Q.device, dtype=Q.dtype
        )
        gather = torch.empty(
            batch, sequence_length, 1, device=Q.device, dtype=torch.int32
        )
        combined_lse = (
            attention_sink.view(1, 1, heads)
            .expand(batch, sequence_length, heads)
            .float()
            .contiguous()
        )
        return output, (empty_kv, empty_kv, gather, cos, sin, combined_lse)

    dtype = cute_dtype(Q)
    device_index = Q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    cos, sin = _rope_tables(device_index, sequence_length, rope_dims)
    selected_width = effective_topk + selected_window
    combined_gather_length = (
        math.ceil(selected_width / 128) * 128 if has_compressed else 0
    )
    # The dual-source FA4 path gives local and compressed logits one online softmax,
    # but explicitly listing every local key makes its metadata O(B*S*W). Keep that
    # fast path only while the list is bounded; the split path below uses FA4's
    # implicit causal-window iterator and merges the two partial softmaxes exactly.
    use_combined_attention = (
        has_local
        and has_compressed
        and effective_topk <= 64
        and _gather_storage_bytes(batch, sequence_length, combined_gather_length)
        <= _COMBINED_GATHER_WORKSPACE_BYTES
    )
    gather_window = selected_window if use_combined_attention else 0
    gather_width = effective_topk + gather_window
    gather_length = (
        math.ceil(gather_width / 128) * 128 if has_compressed else 0
    )
    gather = None
    if has_compressed:
        gather = torch.empty(
            batch, sequence_length, gather_length, device=Q.device, dtype=torch.int32
        )

    use_specialized_preprocess = (
        need_index_scores
        and index_dim == _INDEX_DIM
        and compression_rate == _COMPRESSION_RATE
        and rope_dims == _ROPE_DIMS
        and sequence_length % _COMPRESSION_RATE == 0
    )

    prefetched_query = None
    prefetched_local_query = None
    query_torch_stream = None


    local_kv = torch.empty(
        batch, 0, 1, _HEAD_DIM, device=Q.device, dtype=Q.dtype
    )
    compressed_kv = torch.empty(
        batch, 0, 1, _HEAD_DIM, device=Q.device, dtype=Q.dtype
    )
    compressed_indices = torch.empty(
        batch, 0, 1, index_dim, device=Q.device, dtype=Q.dtype
    )
    if has_local and has_compressed and use_specialized_preprocess:
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
        if has_local:
            local_kv = torch.empty(
                batch, sequence_length, 1, _HEAD_DIM, device=Q.device, dtype=Q.dtype
            )
            compile_local_norm(
                dtype, batch, sequence_length, _HEAD_DIM, rope_dims, 0
            )(KV, KV_norm_weight, cos, sin, local_kv)
        if has_compressed:
            compressed_kv = torch.empty(
                batch, num_blocks, 1, _HEAD_DIM, device=Q.device, dtype=Q.dtype
            )
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
            if need_index_scores:
                compressed_indices = torch.empty(
                    batch, num_blocks, 1, index_dim, device=Q.device, dtype=Q.dtype
                )
                compile_compression(
                    dtype,
                    batch,
                    sequence_length,
                    index_dim,
                    compression_rate,
                    rope_dims,
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
    combined_lse_state = None
    if _return_state:
        combined_lse_state = torch.full(
            (batch, sequence_length, heads),
            -torch.inf,
            device=Q.device,
            dtype=torch.float32,
        )

    index_torch_stream = None
    score_keys = None
    completed_lengths = None
    if has_compressed:
        index_torch_stream = (
            parallel_preprocess_stream(KV)
            if use_specialized_preprocess and need_index_scores
            else None
        )
        if select_all_blocks:
            assert gather is not None
            compile_causal_gather(
                batch,
                sequence_length,
                num_blocks,
                compression_rate,
                effective_topk,
                gather_window,
                gather_length,
            )(gather)
        tensor_score_row_capacity = _radix_score_row_chunk(
            batch * sequence_length,
            num_blocks,
            effective_topk,
        )
        use_tensor_index_scores = (
            need_index_scores
            and Q.dtype == torch.bfloat16
            and _is_power_of_two(index_heads)
            and index_heads <= _TENSOR_INDEX_PACKED_HEADS
            and index_dim == 64
            and num_blocks % 64 == 0
            and sequence_length % compression_rate == 0
            and tensor_score_row_capacity >= sequence_length
            and effective_topk >= _RADIX_TOPK_THRESHOLD
        )
        use_radix_topk = use_tensor_index_scores or effective_topk >= _RADIX_TOPK_THRESHOLD

        if use_tensor_index_scores:
            if output is None:
                output = torch.empty_like(Q)

        def launch_indexer() -> None:
            nonlocal score_keys, completed_lengths
            if use_radix_topk:
                total_rows = batch * sequence_length
                row_chunk = (
                    sequence_length
                    if use_tensor_index_scores
                    else _radix_score_row_chunk(
                        total_rows,
                        num_blocks,
                        effective_topk,
                    )
                )
                score_keys = torch.empty(
                    row_chunk,
                    num_blocks,
                    device=Q.device,
                    dtype=torch.float32,
                )
                completed_lengths = torch.empty(
                    row_chunk,
                    device=Q.device,
                    dtype=torch.int32,
                )
                if use_tensor_index_scores:
                    index_scores = None
                    index_score_keys = compile_index_score_keys(
                        dtype,
                        sequence_length,
                        num_blocks,
                        compression_rate,
                        row_chunk,
                    )
                    batch_chunk = row_chunk // sequence_length
                    query_elements = (
                        batch_chunk
                        * sequence_length
                        * _TENSOR_INDEX_PACKED_HEADS
                        * index_dim
                    )
                    weight_elements = (
                        0
                        if index_heads == _TENSOR_INDEX_PACKED_HEADS
                        else batch_chunk
                        * sequence_length
                        * _TENSOR_INDEX_PACKED_HEADS
                    )
                    workspace_elements = query_elements + weight_elements
                    assert output is not None
                    if output.numel() >= workspace_elements:
                        index_workspace = output.view(-1)[:workspace_elements]
                    else:
                        index_workspace = torch.empty(
                            workspace_elements,
                            device=Q.device,
                            dtype=Q.dtype,
                        )
                    tensor_index_query = index_workspace[
                        :query_elements
                    ].view(
                        batch_chunk,
                        sequence_length,
                        _TENSOR_INDEX_PACKED_HEADS,
                        index_dim,
                    )
                    if index_heads == _TENSOR_INDEX_PACKED_HEADS:
                        tensor_index_weights = None
                    else:
                        tensor_index_weights = index_workspace[
                            query_elements:workspace_elements
                        ].view(
                            batch_chunk,
                            sequence_length,
                            _TENSOR_INDEX_PACKED_HEADS,
                        )
                else:
                    index_scores = compile_index_scores(
                        dtype,
                        batch,
                        sequence_length,
                        index_heads,
                        index_dim,
                        num_blocks,
                        compression_rate,
                        rope_dims,
                        row_chunk,
                    )
                    index_score_keys = None
                    tensor_index_query = None
                    tensor_index_weights = None
                for row_offset in range(0, total_rows, row_chunk):
                    active_rows = min(row_chunk, total_rows - row_offset)
                    if use_tensor_index_scores:
                        assert tensor_index_query is not None
                        assert index_score_keys is not None
                        batch_offset = row_offset // sequence_length
                        active_batches = active_rows // sequence_length
                        query_slab = tensor_index_query[:active_batches]
                        compile_query_rope(
                            dtype,
                            active_batches,
                            index_heads,
                            _TENSOR_INDEX_PACKED_HEADS,
                            index_heads,
                            0,
                            sequence_length,
                            index_dim,
                            rope_dims,
                        )(
                            Q_I[
                                batch_offset : batch_offset + active_batches
                            ],
                            cos,
                            sin,
                            query_slab,
                            query_slab,
                        )
                        if index_heads == _TENSOR_INDEX_PACKED_HEADS:
                            weight_slab = W_I[
                                batch_offset : batch_offset + active_batches
                            ]
                        else:
                            assert tensor_index_weights is not None
                            weight_slab = tensor_index_weights[
                                :active_batches
                            ]
                            compile_pad_index_weights(
                                dtype,
                                active_batches,
                                sequence_length,
                                index_heads,
                                _TENSOR_INDEX_PACKED_HEADS,
                            )(
                                W_I[
                                    batch_offset : batch_offset
                                    + active_batches
                                ],
                                weight_slab,
                            )
                        score_slab = score_keys[:active_rows].view(
                            active_batches,
                            sequence_length,
                            num_blocks,
                        )
                        exact_bf16_index_scores(
                            query_slab,
                            compressed_indices[
                                batch_offset : batch_offset + active_batches
                            ],
                            weight_slab,
                            ratio=compression_rate,
                            qhead_per_kv_head=_TENSOR_INDEX_PACKED_HEADS,
                            out=score_slab,
                            sm_scale=1.0
                            / math.sqrt(index_dim * index_heads),
                        )
                        index_score_keys(
                            score_keys,
                            completed_lengths,
                            Int32(row_offset),
                            Int32(active_rows),
                        )
                    else:
                        assert index_scores is not None
                        index_scores(
                            Q_I,
                            compressed_indices,
                            W_I,
                            cos,
                            sin,
                            score_keys,
                            completed_lengths,
                            Int32(row_offset),
                            Int32(active_rows),
                        )
                    selected_indices = _radix_topk_indices(
                        score_keys[:active_rows],
                        completed_lengths[:active_rows],
                        effective_topk,
                    )
                    compile_selected_gather(
                        batch,
                        sequence_length,
                        num_blocks,
                        effective_topk,
                        gather_window,
                        gather_length,
                        active_rows,
                    )(
                        selected_indices,
                        gather,
                        Int32(row_offset),
                        Int32(active_rows),
                    )
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
                    gather_window,
                    rope_dims,
                    gather_length,
                )
                index_topk(Q_I, compressed_indices, W_I, cos, sin, gather)

        if need_index_scores:
            if index_torch_stream is not None:
                with torch.cuda.stream(index_torch_stream):
                    launch_indexer()
            else:
                launch_indexer()

    # Before position R - 1, no compressed block has completed. Replay that complete,
    # mathematically defined local-only prefix with the oracle's accumulation boundaries.
    prefix = (
        min(selected_window, sequence_length, compression_rate - 1)
        if has_local and has_compressed
        else 0
    )
    for head_offset in range(0, heads, 128):
        active_heads = min(128, heads - head_offset)
        tile_heads = 128
        local_tile_heads = 128
        split_attention = has_local and has_compressed and not use_combined_attention
        fuse_q_rope = has_compressed and active_heads == 128 and not split_attention
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
        fuse_attention_epilogue = (
            has_compressed and active_heads == 128 and not split_attention
        )

        if split_attention:
            assert gather is not None
            local_output = torch.empty_like(local_query)
            compressed_output = torch.empty_like(query)
            local_lse = torch.empty(
                batch,
                sequence_length,
                local_tile_heads,
                device=Q.device,
                dtype=torch.float32,
            )
            compressed_lse = torch.empty(
                batch,
                sequence_length,
                tile_heads,
                device=Q.device,
                dtype=torch.float32,
            )
            local_attention(
                local_query,
                local_kv,
                selected_window,
                output=local_output,
                lse=local_lse,
            )
            if index_torch_stream is not None:
                torch.cuda.current_stream(Q.device).wait_stream(index_torch_stream)
                index_torch_stream = None
            compressed_attention(
                query,
                compressed_kv,
                gather,
                output=compressed_output,
                lse=compressed_lse,
                head_offset=head_offset,
                rope_dims=rope_dims,
                csa_topk=effective_topk,
                csa_window=0,
                store_lse=True,
            )
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
                True,
                True,
                _return_state,
            )(
                local_output,
                local_lse,
                compressed_output,
                compressed_lse,
                attention_sink,
                cos,
                sin,
                output,
                combined_lse_state,
            )
            del local_output, local_lse, compressed_output, compressed_lse
        else:
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
            if has_compressed:
                assert gather is not None
                if index_torch_stream is not None:
                    torch.cuda.current_stream(Q.device).wait_stream(index_torch_stream)
                    index_torch_stream = None
                compressed_attention(
                    query,
                    compressed_kv,
                    gather,
                    local_value=local_kv if use_combined_attention else None,
                    output=selected_output,
                    lse=selected_lse,
                    sink=attention_sink if fuse_attention_epilogue else None,
                    cos=cos if fuse_attention_epilogue else None,
                    sin=sin if fuse_attention_epilogue else None,
                    head_offset=head_offset,
                    fuse_q_rope=fuse_q_rope,
                    rope_dims=rope_dims,
                    csa_topk=effective_topk,
                    csa_window=gather_window,
                    store_lse=store_selected_lse,
                )
            else:
                local_attention(
                    query,
                    local_kv,
                    selected_window,
                    output=selected_output,
                    lse=selected_lse,
                )
            if _return_state:
                assert combined_lse_state is not None
                assert selected_lse is not None
                combined_lse_state[
                    :, :, head_offset : head_offset + active_heads
                ].copy_(selected_lse[:, :, :active_heads])
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
                    True,
                    False,
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
                    None,
                )
            del selected_output, selected_lse
        prefix_query = None
        if prefix:
            prefix_query = Q if fuse_q_rope else local_query[:, :prefix].contiguous()
        del query, local_query

        # Before q = R - 1 no compressed block has finished. Replaying only that true prefix
        # with the oracle's BF16 boundaries removes its sharp local-only outlier.
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
                _return_state,
            )(
                prefix_query,
                local_kv,
                attention_sink,
                cos,
                sin,
                output,
                combined_lse_state,
            )
    assert output is not None
    if _return_state:
        gather_state = gather
        if gather_state is None:
            gather_state = torch.empty(
                batch, sequence_length, 1, device=Q.device, dtype=torch.int32
            )
        assert combined_lse_state is not None
        return output, (
            local_kv,
            compressed_kv,
            gather_state,
            cos,
            sin,
            combined_lse_state,
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
        dtype = cute_dtype(Q)
        dsa_dtype = BFloat16
        topk = min(ctx.num_topk_blocks, sequence // ctx.compression_rate)
        window = min(ctx.sliding_window_size, sequence)
        has_local = window > 0
        has_compressed = topk > 0
        has_attention = has_local or has_compressed

        if has_attention:
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
            local_kv, compressed_kv, gather, cos, sin, combined_lse = state
            blocks = compressed_kv.shape[1]
            local_length = local_kv.shape[1]
            dQ = torch.empty_like(Q)
            dlocal = torch.zeros_like(local_kv, dtype=torch.float32)
            dcompressed = torch.zeros_like(compressed_kv, dtype=torch.float32)
            from .dsa_backward_sm100 import sparse_attention_backward_wrapper

            tokens = batch * sequence
            index_width = math.ceil((topk + window) / 64) * 64
            total_kv = batch * (blocks + local_length)
            kv_packed = torch.empty(
                total_kv, dim,
                device=Q.device,
                dtype=torch.bfloat16,
            )
            sink_fp32 = torch.empty(heads, device=Q.device, dtype=torch.float32)
            # CuTe tensor descriptors cannot encode an empty extent. The packer is
            # specialized on the real lengths and never reads the absent source, so
            # use a one-row descriptor sentinel for local-only/compressed-only modes.
            packed_compressed_source = compressed_kv
            if blocks == 0:
                packed_compressed_source = torch.empty(
                    batch, 1, 1, dim, device=Q.device, dtype=Q.dtype
                )
            packed_local_source = local_kv
            if local_length == 0:
                packed_local_source = torch.empty(
                    batch, 1, 1, dim, device=Q.device, dtype=Q.dtype
                )
            dcompressed_target = dcompressed
            if blocks == 0:
                dcompressed_target = torch.empty(
                    batch, 1, 1, dim, device=Q.device, dtype=torch.float32
                )
            dlocal_target = dlocal
            if local_length == 0:
                dlocal_target = torch.empty(
                    batch, 1, 1, dim, device=Q.device, dtype=torch.float32
                )

            compile_pack_dsa_kv_sink(
                dtype,
                dsa_dtype,
                batch,
                sequence,
                dim,
                blocks,
                local_length,
                heads,
            )(
                packed_compressed_source,
                packed_local_source,
                attention_sink,
                kv_packed,
                sink_fp32,
            )
            head_chunk, token_chunk = _dsa_tile_shape(
                tokens, dim, heads, total_kv, index_width
            )
            indices = torch.empty(
                token_chunk, index_width, device=Q.device, dtype=torch.int32
            )
            topk_lengths = torch.empty(
                token_chunk, device=Q.device, dtype=torch.int32
            )
            d_sink_accumulator = torch.empty(
                heads, device=Q.device, dtype=torch.float32
            )
            d_sink_accumulator.zero_()
            pack_indices = compile_pack_dsa_indices(
                batch,
                sequence,
                blocks,
                ctx.compression_rate,
                topk,
                window,
                index_width,
                gather.shape[-1],
                token_chunk,
            )
            for token_offset in range(0, tokens, token_chunk):
                packed_tokens = min(token_chunk, tokens - token_offset)
                selected_indices = indices[:packed_tokens]
                selected_lengths = topk_lengths[:packed_tokens]
                pack_indices(
                    gather,
                    indices,
                    topk_lengths,
                    Int32(token_offset),
                    Int32(packed_tokens),
                )
                for head_offset in range(0, heads, head_chunk):
                    packed_heads = min(head_chunk, heads - head_offset)
                    selected_sink = sink_fp32[
                        head_offset : head_offset + packed_heads
                    ]
                    q_packed = torch.empty(
                        packed_tokens,
                        packed_heads,
                        dim,
                        device=Q.device,
                        dtype=torch.bfloat16,
                    )
                    out_packed = torch.empty_like(q_packed)
                    dout_packed = torch.empty_like(q_packed)
                    lse_packed = torch.empty(
                        packed_tokens,
                        packed_heads,
                        device=Q.device,
                        dtype=torch.float32,
                    )
                    compile_prepare_dsa_backward(
                        dtype,
                        dsa_dtype,
                        batch,
                        heads,
                        packed_heads,
                        sequence,
                        packed_tokens,
                        dim,
                        ctx.rope_dims,
                    )(
                        Q, output, dout, combined_lse, cos, sin,
                        q_packed, out_packed, dout_packed, lse_packed,
                        Int32(head_offset), Int32(token_offset),
                    )
                    result = sparse_attention_backward_wrapper(
                        q_packed,
                        kv_packed,
                        out_packed,
                        dout_packed,
                        lse_packed,
                        selected_sink,
                        selected_indices,
                        softmax_scale=1.0 / math.sqrt(dim),
                        topk_length=selected_lengths,
                    )
                    compile_unpack_dsa_gradients(
                        dtype,
                        dsa_dtype,
                        batch,
                        heads,
                        packed_heads,
                        sequence,
                        packed_tokens,
                        dim,
                        ctx.rope_dims,
                        blocks,
                        local_length,
                    )(
                        result["dq"], result["dkv"], cos, sin,
                        dQ, dlocal_target, dcompressed_target,
                        Int32(head_offset), Int32(token_offset),
                    )
                    d_sink_accumulator[
                        head_offset : head_offset + packed_heads
                    ].add_(result["d_sink"])
                    del q_packed, out_packed, dout_packed, lse_packed, result
            d_attention_sink = torch.empty_like(attention_sink)
            compile_cast_gradient(dtype, heads)(
                d_sink_accumulator, d_attention_sink
            )
            del (
                kv_packed,
                indices,
                topk_lengths,
                sink_fp32,
                d_sink_accumulator,
                output,
                combined_lse,
                gather,
            )
        else:
            dQ = torch.zeros_like(Q)
            d_attention_sink = torch.zeros_like(attention_sink)

        if has_local:
            dKV = torch.empty_like(KV)
            dKV_weight_fp32 = torch.zeros_like(KV_norm_weight, dtype=torch.float32)
            compile_local_norm_backward(dtype, batch, sequence, dim, ctx.rope_dims)(
                KV, KV_norm_weight, cos, sin, dlocal, dKV, dKV_weight_fp32
            )
            dKV_weight = torch.empty_like(KV_norm_weight)
            compile_cast_gradient(dtype, KV_norm_weight.numel())(
                dKV_weight_fp32, dKV_weight
            )
        else:
            dKV = torch.zeros_like(KV)
            dKV_weight = torch.zeros_like(KV_norm_weight)

        if has_compressed:
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
            dcompressed_weight = torch.empty_like(compressed_kv_norm_weight)
            compile_cast_gradient(dtype, B_a.numel())(
                dB_a_fp32.view(-1), dB_a.view(-1)
            )
            compile_cast_gradient(dtype, B_b.numel())(
                dB_b_fp32.view(-1), dB_b.view(-1)
            )
            compile_cast_gradient(dtype, compressed_kv_norm_weight.numel())(
                dcompressed_weight_fp32, dcompressed_weight
            )
        else:
            dC_a = torch.zeros_like(C_a)
            dC_b = torch.zeros_like(C_b)
            dZ_a = torch.zeros_like(Z_a)
            dZ_b = torch.zeros_like(Z_b)
            dB_a = torch.zeros_like(B_a)
            dB_b = torch.zeros_like(B_b)
            dcompressed_weight = torch.zeros_like(compressed_kv_norm_weight)

        grads = (
            dQ, None, dKV, dC_a, dC_b, dZ_a, dZ_b, dB_a, dB_b, None,
            None, None, None, None, None, None,
            dKV_weight, None, dcompressed_weight, d_attention_sink,
            None, None, None, None, None,
        )
        return tuple(g if need else None for g, need in zip(grads, ctx.needs_input_grad))


class _RuntimeBackwardContext:
    """Minimal context used to run the existing manual backward behind an opaque op."""

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        compression_rate: int,
        num_topk_blocks: int,
        sliding_window_size: int,
        rope_dims: int,
        share_kv: bool,
    ) -> None:
        self.saved_tensors = tensors
        self.compression_rate = compression_rate
        self.num_topk_blocks = num_topk_blocks
        self.sliding_window_size = sliding_window_size
        self.rope_dims = rope_dims
        self.share_kv = share_kv
        self.needs_input_grad = tuple(
            index in _DIFFERENTIABLE_INPUT_INDICES for index in range(25)
        )


@torch.library.custom_op(
    "attention_gym::_cute_compressed_sparse_attention_backward",
    mutates_args=(),
    device_types="cuda",
)
def _cute_compressed_sparse_attention_backward(
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
    dout: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    tensors = (
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
    )
    runtime_ctx = _RuntimeBackwardContext(
        tensors,
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
    )
    with torch.cuda.device(Q.device):
        grads = _CuteCompressedSparseAttention.backward(runtime_ctx, dout)
    differentiable_grads = tuple(grads[index] for index in _DIFFERENTIABLE_INPUT_INDICES)
    assert all(isinstance(grad, torch.Tensor) for grad in differentiable_grads)
    return differentiable_grads


@_cute_compressed_sparse_attention_backward.register_fake
def _cute_compressed_sparse_attention_backward_fake(
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
    dout,
    compression_rate,
    num_topk_blocks,
    sliding_window_size,
    rope_dims,
    share_kv,
):
    del (
        Q_I,
        W_I,
        K_Ia,
        K_Ib,
        Z_Ia,
        Z_Ib,
        B_Ia,
        B_Ib,
        compressed_indices_norm_weight,
        dout,
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
    )
    return tuple(
        torch.empty_like(tensor)
        for tensor in (
            Q,
            KV,
            C_a,
            C_b,
            Z_a,
            Z_b,
            B_a,
            B_b,
            KV_norm_weight,
            compressed_kv_norm_weight,
            attention_sink,
        )
    )


@torch.library.custom_op(
    "attention_gym::_cute_compressed_sparse_attention_forward",
    mutates_args=(),
    device_types="cuda",
)
def _cute_compressed_sparse_attention_forward_op(
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
) -> torch.Tensor:
    with torch.cuda.device(Q.device):
        return _compressed_sparse_attention_forward(
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


@_cute_compressed_sparse_attention_forward_op.register_fake
def _cute_compressed_sparse_attention_forward_fake(
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
):
    del (
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
    return torch.empty_like(Q)


def _cute_compressed_sparse_attention_setup_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs[:20])
    (
        ctx.compression_rate,
        ctx.num_topk_blocks,
        ctx.sliding_window_size,
        ctx.rope_dims,
        ctx.share_kv,
    ) = inputs[20:]


def _cute_compressed_sparse_attention_autograd_backward(ctx, dout):
    grads = _cute_compressed_sparse_attention_backward(
        *ctx.saved_tensors,
        dout,
        ctx.compression_rate,
        ctx.num_topk_blocks,
        ctx.sliding_window_size,
        ctx.rope_dims,
        ctx.share_kv,
    )
    result = [None] * 25
    for index, grad in zip(_DIFFERENTIABLE_INPUT_INDICES, grads):
        result[index] = grad
    return tuple(result)


_cute_compressed_sparse_attention_forward_op.register_autograd(
    _cute_compressed_sparse_attention_autograd_backward,
    setup_context=_cute_compressed_sparse_attention_setup_context,
)


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
        with torch.cuda.device(Q.device):
            return _compressed_sparse_attention_forward(*args, _return_state=True)
    operator_args = tuple(
        value.detach() if index < 20 and index not in _DIFFERENTIABLE_INPUT_INDICES else value
        for index, value in enumerate(args)
    )
    return _cute_compressed_sparse_attention_forward_op(*operator_args)


__all__ = ["compressed_sparse_attention"]
