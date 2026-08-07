"""Blackwell shared-KV schedules for selected-attention backward.

Each dQ program owns its query and sink-partial outputs. Sparse dKV programs write one
partial per head tile into the expanded head dimension, which autograd subsequently sums.
This keeps repeated backward calls from one forward atomic-free and bitwise repeatable.
"""

import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps, num_stages=1)
        for block_n in (64, 128)
        for num_warps in (4, 8)
    ],
    key=["B", "H", "S", "D", "SPARSE_SEQ_LEN", "TOPK", "WINDOW", "HAS_DOC_IDS"],
    cache_results=True,
)
@triton.jit
def _selected_attention_bwd_dq_shared(
    query_ptr,
    sparse_kv_ptr,
    local_kv_ptr,
    kv_indices_ptr,
    doc_ids_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    attention_sink_ptr,
    grad_query_ptr,
    grad_sink_partials_ptr,
    QUERY_STRIDES: tl.constexpr,
    SPARSE_KV_STRIDES: tl.constexpr,
    LOCAL_KV_STRIDES: tl.constexpr,
    KV_INDICES_STRIDES: tl.constexpr,
    DOC_IDS_STRIDES: tl.constexpr,
    LSE_STRIDES: tl.constexpr,
    GRAD_SINK_PARTIALS_STRIDES: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    SPARSE_SEQ_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute query gradients across heads while reusing shared KV tiles."""
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
    delta = tl.sum(grad_output * output, axis=1)
    grad_query = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

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
                (selected_offsets < TOPK) & (selected_idx >= 0) & (selected_idx < SPARSE_SEQ_LEN)
            )
            selected_idx = tl.where(selected_valid, selected_idx, 0)
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

    if HAS_DOC_IDS:
        query_doc_id = tl.load(doc_ids_ptr + ptr_offset((batch, sequence), DOC_IDS_STRIDES))

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
        if HAS_DOC_IDS:
            key_doc_ids = tl.load(
                doc_ids_ptr + ptr_offset((batch, offsets_n), DOC_IDS_STRIDES),
                mask=local_valid,
                other=-1,
            )
            local_valid &= key_doc_ids == query_doc_id

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
    sink_gradient = -tl.exp(sink - lse) * delta
    tl.store(
        grad_sink_partials_ptr
        + ptr_offset((batch, offsets_h, sequence), GRAD_SINK_PARTIALS_STRIDES),
        sink_gradient,
        mask=head_mask,
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
    key=["H", "S", "D", "SPARSE_SEQ_LEN", "TOPK"],
    reset_to_zero=["grad_sparse_kv_ptr"],
    cache_results=True,
)
@triton.jit
def _selected_attention_bwd_dsparse_kv_shared(
    query_ptr,
    sparse_kv_ptr,
    selected_queries_ptr,
    block_offsets_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_sparse_kv_ptr,
    QUERY_STRIDES: tl.constexpr,
    SPARSE_KV_STRIDES: tl.constexpr,
    SELECTED_QUERIES_STRIDES: tl.constexpr,
    BLOCK_OFFSETS_STRIDES: tl.constexpr,
    LSE_STRIDES: tl.constexpr,
    GRAD_SPARSE_KV_STRIDES: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    SPARSE_SEQ_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate a head tile into one partial shared sparse-KV gradient."""
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
        delta = tl.sum(grad_output * output, axis=1)
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
