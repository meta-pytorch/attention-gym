"""Blackwell shared-KV schedules for gather-attention backward.

The default sparse-dKV schedule forms query-owned top-k-by-dimension gradient tiles and
atomically scatters them into one FP32 shared-KV gradient. Deterministic mode instead uses
an inverted query map so each sparse-KV program exclusively owns its output.
"""

import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset

from .primitives import load_document_bounds, offset_strides, prune_wide_backward_configs


def prune_shared_dq_configs(configs, named_args, D, **kwargs):
    if D == 512:
        return prune_wide_backward_configs(configs, named_args, D=D, **kwargs)
    return [config for config in configs if config.kwargs["BLOCK_N"] != 16]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps, num_stages=1)
        for block_n in (16, 64, 128)
        for num_warps in (4, 8)
    ],
    key=["B", "H", "D", "TOPK", "WINDOW", "HAS_CU_SEQLENS"],
    prune_configs_by={"early_config_prune": prune_shared_dq_configs},
    cache_results=True,
)
@triton.jit
def _gather_attn_bwd_dq_shared(
    query_ptr,
    sparse_kv_ptr,
    local_kv_ptr,
    kv_indices_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    attention_sink_ptr,
    grad_query_ptr,
    grad_sink_partials_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    LOCAL_KV_STRIDES,
    KV_INDICES_STRIDES,
    num_documents,
    LSE_STRIDES,
    GRAD_SINK_PARTIALS_STRIDES,
    B: tl.constexpr,
    H: tl.constexpr,
    S,
    D: tl.constexpr,
    SPARSE_SEQ_LEN,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute query gradients across heads while reusing shared KV tiles."""
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    LOCAL_KV_STRIDES = offset_strides(LOCAL_KV_STRIDES, WIDE)
    KV_INDICES_STRIDES = offset_strides(KV_INDICES_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)
    GRAD_SINK_PARTIALS_STRIDES = offset_strides(GRAD_SINK_PARTIALS_STRIDES, WIDE)

    sequence = tl.program_id(0)
    batch = tl.program_id(1)
    head_block = tl.program_id(2)

    offsets_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_d = tl.arange(0, BLOCK_D)
    head_mask = offsets_h < H
    dimension_mask = offsets_d < D
    matrix_mask = head_mask[:, None] & dimension_mask[None, :]

    query = tl.load(
        query_ptr
        + ptr_offset((batch, offsets_h[:, None], sequence, offsets_d[None, :]), QUERY_STRIDES),
        mask=matrix_mask,
        other=0.0,
    )
    output = tl.load(
        output_ptr
        + ptr_offset((batch, offsets_h[:, None], sequence, offsets_d[None, :]), QUERY_STRIDES),
        mask=matrix_mask,
        other=0.0,
    )
    grad_output = tl.load(
        grad_output_ptr
        + ptr_offset((batch, offsets_h[:, None], sequence, offsets_d[None, :]), QUERY_STRIDES),
        mask=matrix_mask,
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + ptr_offset((batch, offsets_h, sequence), LSE_STRIDES),
        mask=head_mask,
        other=0.0,
    )
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    grad_query = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        query_start, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, sequence, num_documents, S, WIDE
        )

    if TOPK:
        offsets_k = tl.arange(0, BLOCK_K)
        for selected_start in tl.range(0, TOPK, BLOCK_K, num_stages=2):
            selected_offsets = selected_start + offsets_k
            selected_idx = tl.load(
                kv_indices_ptr
                + ptr_offset((batch, sequence, selected_offsets), KV_INDICES_STRIDES),
                mask=selected_offsets < TOPK,
                other=-1,
            )
            selected_valid = (
                (selected_offsets < TOPK)
                & (selected_idx >= 0)
                & (selected_idx < candidate_end - candidate_start)
            )
            selected_idx = tl.where(selected_valid, selected_idx, 0) + candidate_start
            sparse_values = tl.load(
                sparse_kv_ptr
                + ptr_offset(
                    (batch, 0, selected_idx[:, None], offsets_d[None, :]),
                    SPARSE_KV_STRIDES,
                ),
                mask=selected_valid[:, None] & dimension_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(query, tl.trans(sparse_values), input_precision="tf32x3") * SCALE
            probabilities = tl.where(
                head_mask[:, None] & selected_valid[None, :],
                tl.exp(scores - lse[:, None]),
                0.0,
            )
            grad_probabilities = tl.dot(
                grad_output, tl.trans(sparse_values), input_precision="tf32x3"
            )
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            grad_query += (
                tl.dot(
                    grad_scores.to(sparse_values.dtype),
                    sparse_values,
                    input_precision="tf32x3",
                )
                * SCALE
            )

    offsets_n_base = tl.arange(0, BLOCK_N)
    first_local_position = sequence - WINDOW + 1
    for local_start in tl.range(0, WINDOW, BLOCK_N, num_stages=2):
        offsets_n = first_local_position + local_start + offsets_n_base
        local_valid = (offsets_n >= 0) & (offsets_n <= sequence) & (offsets_n < S)
        local_values = tl.load(
            local_kv_ptr
            + ptr_offset(
                (batch, 0, offsets_n[:, None], offsets_d[None, :]),
                LOCAL_KV_STRIDES,
            ),
            mask=local_valid[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        if HAS_CU_SEQLENS:
            local_valid &= offsets_n >= query_start

        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        probabilities = tl.where(
            head_mask[:, None] & local_valid[None, :],
            tl.exp(scores - lse[:, None]),
            0.0,
        )
        grad_probabilities = tl.dot(grad_output, tl.trans(local_values), input_precision="tf32x3")
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_query += (
            tl.dot(
                grad_scores.to(local_values.dtype),
                local_values,
                input_precision="tf32x3",
            )
            * SCALE
        )

    tl.store(
        grad_query_ptr
        + ptr_offset((batch, offsets_h[:, None], sequence, offsets_d[None, :]), QUERY_STRIDES),
        grad_query,
        mask=matrix_mask,
    )

    sink = tl.load(attention_sink_ptr + offsets_h, mask=head_mask, other=0.0)
    sink_probability = tl.where(lse == -float("inf"), 0.0, tl.exp(sink - lse))
    sink_gradient = -sink_probability * delta
    tl.store(
        grad_sink_partials_ptr
        + ptr_offset((batch, offsets_h, sequence), GRAD_SINK_PARTIALS_STRIDES),
        sink_gradient,
        mask=head_mask,
    )


# Atomic configs accumulate into the same FP32 output, so clear it between tuning trials.
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_H": block_h, "BLOCK_K": block_k},
            num_warps=num_warps,
            num_stages=1,
        )
        for block_h, block_k, num_warps in (
            (16, 16, 4),
            (32, 16, 4),
            (16, 32, 4),
            (32, 32, 8),
            (16, 64, 8),
            (32, 64, 8),
        )
    ],
    key=["B", "H", "D", "TOPK", "HAS_CU_SEQLENS"],
    reset_to_zero=["grad_sparse_kv_ptr"],
    prune_configs_by={"early_config_prune": prune_wide_backward_configs},
    cache_results=True,
)
@triton.jit
def _gather_attn_bwd_dsparse_kv_shared_atomic(
    query_ptr,
    sparse_kv_ptr,
    kv_indices_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_sparse_kv_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    KV_INDICES_STRIDES,
    num_documents,
    LSE_STRIDES,
    GRAD_SPARSE_KV_STRIDES,
    B: tl.constexpr,
    H: tl.constexpr,
    S,
    D: tl.constexpr,
    SPARSE_SEQ_LEN,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Atomically scatter one query's sparse-KV gradient tile into shared dKV."""
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    KV_INDICES_STRIDES = offset_strides(KV_INDICES_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)
    GRAD_SPARSE_KV_STRIDES = offset_strides(GRAD_SPARSE_KV_STRIDES, WIDE)

    query_position = tl.program_id(0)
    batch = tl.program_id(1)
    tile = tl.program_id(2)
    num_selected_tiles = tl.cdiv(TOPK, BLOCK_K)
    head_block = tile // num_selected_tiles
    selected_block = tile % num_selected_tiles

    offsets_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_k = selected_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets_d = tl.arange(0, BLOCK_D)
    head_mask = offsets_h < H
    selected_mask = offsets_k < TOPK
    dimension_mask = offsets_d < D

    selected_indices = tl.load(
        kv_indices_ptr + ptr_offset((batch, query_position, offsets_k), KV_INDICES_STRIDES),
        mask=selected_mask,
        other=-1,
    )
    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        _, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, query_position, num_documents, S, WIDE
        )
    selected_valid = (
        selected_mask
        & (selected_indices >= 0)
        & (selected_indices < candidate_end - candidate_start)
    )
    selected_indices = tl.where(selected_valid, selected_indices, 0) + candidate_start

    head_dimension_mask = head_mask[:, None] & dimension_mask[None, :]
    query_offsets = ptr_offset(
        (batch, offsets_h[:, None], query_position, offsets_d[None, :]),
        QUERY_STRIDES,
    )
    query = tl.load(query_ptr + query_offsets, mask=head_dimension_mask, other=0.0)
    output = tl.load(output_ptr + query_offsets, mask=head_dimension_mask, other=0.0)
    grad_output = tl.load(
        grad_output_ptr + query_offsets,
        mask=head_dimension_mask,
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + ptr_offset((batch, offsets_h, query_position), LSE_STRIDES),
        mask=head_mask,
        other=0.0,
    )
    sparse_values = tl.load(
        sparse_kv_ptr
        + ptr_offset(
            (batch, 0, selected_indices[:, None], offsets_d[None, :]),
            SPARSE_KV_STRIDES,
        ),
        mask=selected_valid[:, None] & dimension_mask[None, :],
        other=0.0,
    )

    scores = tl.dot(query, tl.trans(sparse_values), input_precision="tf32x3") * SCALE
    probabilities = tl.where(
        head_mask[:, None] & selected_valid[None, :],
        tl.exp(scores - lse[:, None]),
        0.0,
    )
    grad_probabilities = tl.dot(
        grad_output,
        tl.trans(sparse_values),
        input_precision="tf32x3",
    )
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    grad_scores = probabilities * (grad_probabilities - delta[:, None])
    grad_values = tl.dot(
        tl.trans(probabilities.to(grad_output.dtype)),
        grad_output,
        input_precision="tf32x3",
    )
    grad_values += tl.dot(
        tl.trans((grad_scores * SCALE).to(query.dtype)),
        query,
        input_precision="tf32x3",
    )

    tl.atomic_add(
        grad_sparse_kv_ptr
        + ptr_offset(
            (batch, 0, selected_indices[:, None], offsets_d[None, :]),
            GRAD_SPARSE_KV_STRIDES,
        ),
        grad_values,
        mask=selected_valid[:, None] & dimension_mask[None, :],
    )


# Different BLOCK_H configs write different partial slots, so clear stale slots between trials.
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_H": block_h, "BLOCK_M": block_m},
            num_warps=num_warps,
            num_stages=1,
        )
        for block_h, block_m, num_warps in (
            (2, 32, 4),
            (4, 16, 4),
            (4, 32, 4),
            (4, 32, 8),
            (8, 16, 4),
            (8, 16, 8),
        )
    ],
    key=["H", "D", "TOPK"],
    reset_to_zero=["grad_sparse_kv_ptr"],
    cache_results=True,
)
@triton.jit
def _gather_attn_bwd_dsparse_kv_shared(
    query_ptr,
    sparse_kv_ptr,
    selected_queries_ptr,
    block_offsets_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_sparse_kv_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    SELECTED_QUERIES_STRIDES,
    BLOCK_OFFSETS_STRIDES,
    LSE_STRIDES,
    GRAD_SPARSE_KV_STRIDES,
    H: tl.constexpr,
    S,
    D: tl.constexpr,
    SPARSE_SEQ_LEN,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate a head tile into one partial shared sparse-KV gradient."""
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    SELECTED_QUERIES_STRIDES = offset_strides(SELECTED_QUERIES_STRIDES, WIDE)
    BLOCK_OFFSETS_STRIDES = offset_strides(BLOCK_OFFSETS_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)
    GRAD_SPARSE_KV_STRIDES = offset_strides(GRAD_SPARSE_KV_STRIDES, WIDE)

    sparse_index = tl.program_id(0)
    batch = tl.program_id(1)
    head_block = tl.program_id(2)

    offsets_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_m_base = tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    dot_rows = tl.arange(0, 16)
    head_mask = offsets_h < H
    dimension_mask = offsets_d < D
    sparse_value = tl.load(
        sparse_kv_ptr + ptr_offset((batch, 0, sparse_index, offsets_d), SPARSE_KV_STRIDES),
        mask=dimension_mask,
        other=0.0,
    )
    dot_value = tl.where(dot_rows[None, :] == 0, sparse_value[:, None], 0.0)
    grad_value = tl.zeros((16, BLOCK_D), tl.float32)
    entry_start = tl.load(
        block_offsets_ptr + ptr_offset((batch, sparse_index), BLOCK_OFFSETS_STRIDES)
    )
    entry_end = tl.load(
        block_offsets_ptr + ptr_offset((batch, sparse_index + 1), BLOCK_OFFSETS_STRIDES)
    )

    for entry_tile in tl.range(entry_start, entry_end, BLOCK_M):
        entry_offsets = entry_tile + offsets_m_base
        entry_mask = entry_offsets < entry_end
        query_positions = tl.load(
            selected_queries_ptr + ptr_offset((batch, entry_offsets), SELECTED_QUERIES_STRIDES),
            mask=entry_mask,
            other=0,
        )
        query_mask = entry_mask & (query_positions >= 0) & (query_positions < S)
        row_mask = head_mask[:, None] & query_mask[None, :]
        matrix_mask = row_mask[:, :, None] & dimension_mask[None, None, :]
        tensor_offsets = ptr_offset(
            (
                batch,
                offsets_h[:, None, None],
                query_positions[None, :, None],
                offsets_d[None, None, :],
            ),
            QUERY_STRIDES,
        )
        query = tl.reshape(
            tl.load(query_ptr + tensor_offsets, mask=matrix_mask, other=0.0),
            (BLOCK_H * BLOCK_M, BLOCK_D),
        )
        output = tl.reshape(
            tl.load(output_ptr + tensor_offsets, mask=matrix_mask, other=0.0),
            (BLOCK_H * BLOCK_M, BLOCK_D),
        )
        grad_output = tl.reshape(
            tl.load(grad_output_ptr + tensor_offsets, mask=matrix_mask, other=0.0),
            (BLOCK_H * BLOCK_M, BLOCK_D),
        )
        flat_row_mask = tl.reshape(row_mask, (BLOCK_H * BLOCK_M,))
        lse = tl.reshape(
            tl.load(
                lse_ptr
                + ptr_offset(
                    (batch, offsets_h[:, None], query_positions[None, :]),
                    LSE_STRIDES,
                ),
                mask=row_mask,
                other=0.0,
            ),
            (BLOCK_H * BLOCK_M,),
        )
        score_tile = tl.dot(query, dot_value, input_precision="tf32x3")
        scores = tl.sum(score_tile * (dot_rows[None, :] == 0), axis=1) * SCALE
        grad_probability_tile = tl.dot(grad_output, dot_value, input_precision="tf32x3")
        grad_probabilities = tl.sum(grad_probability_tile * (dot_rows[None, :] == 0), axis=1)
        delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
        probabilities = tl.where(flat_row_mask, tl.exp(scores - lse), 0.0)
        grad_scores = probabilities * (grad_probabilities - delta)
        combined_weights = tl.cat(probabilities, grad_scores * SCALE, dim=0)
        combined_values = tl.cat(grad_output, query, dim=0)
        combined_weight_tile = tl.where(
            dot_rows[:, None] == 0,
            combined_weights[None, :],
            0.0,
        )
        grad_value = tl.dot(
            combined_weight_tile.to(query.dtype),
            combined_values,
            acc=grad_value,
            input_precision="tf32x3",
        )

    tl.store(
        grad_sparse_kv_ptr
        + ptr_offset((batch, head_block, sparse_index, offsets_d), GRAD_SPARSE_KV_STRIDES),
        tl.sum(grad_value * (dot_rows[:, None] == 0), axis=0),
        mask=dimension_mask,
    )
