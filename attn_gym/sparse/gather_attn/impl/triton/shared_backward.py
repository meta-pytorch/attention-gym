"""Blackwell shared-KV schedules for gather-attention backward.

The default sparse-dKV schedule forms query-owned top-k-by-dimension gradient tiles and
atomically scatters them into one FP32 shared-KV gradient. Deterministic mode instead uses
an inverted query map so each sparse-KV program exclusively owns its output.
"""

import triton
import triton.language as tl

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import ptr_offset

from .primitives import (
    head_block_heads,
    in_dense_tile,
    load_document_bounds,
    offset_strides,
    prune_wide_backward_configs,
    tuning_key,
)


@triton.jit
def _resolved_dq_tile(
    query,
    grad_output,
    lse,
    delta,
    sparse_kv_ptr,
    sparse_v_ptr,
    selected_rows_ptr,
    remaining,
    batch,
    kv_head,
    offsets_k,
    offsets_d,
    dimension_mask,
    head_mask,
    grad_query,
    SPARSE_KV_STRIDES,
    SPARSE_V_STRIDES,
    V_IS_K: tl.constexpr,
    SCALE: tl.constexpr,
    MASKED: tl.constexpr,
):
    """Accumulate dQ over one tile of compacted selections; only the last tile is MASKED."""
    if MASKED:
        selected_valid = offsets_k < remaining
        selected_idx = tl.load(selected_rows_ptr + offsets_k, mask=selected_valid, other=0)
        selected_mask = selected_valid[:, None] & dimension_mask[None, :]
        probability_mask = head_mask[:, None] & selected_valid[None, :]
    else:
        selected_idx = tl.load(selected_rows_ptr + offsets_k)
        selected_mask = dimension_mask[None, :]
        probability_mask = head_mask[:, None]
    sparse_values = tl.load(
        sparse_kv_ptr
        + ptr_offset(
            (batch, kv_head, selected_idx[:, None], offsets_d[None, :]), SPARSE_KV_STRIDES
        ),
        mask=selected_mask,
        other=0.0,
    )
    if V_IS_K:
        sparse_v = sparse_values
    else:
        sparse_v = tl.load(
            sparse_v_ptr
            + ptr_offset(
                (batch, kv_head, selected_idx[:, None], offsets_d[None, :]), SPARSE_V_STRIDES
            ),
            mask=selected_mask,
            other=0.0,
        )
    scores = tl.dot(query, tl.trans(sparse_values), input_precision="tf32x3") * SCALE
    probabilities = tl.where(probability_mask, tl.exp(scores - lse[:, None]), 0.0)
    grad_probabilities = tl.dot(grad_output, tl.trans(sparse_v), input_precision="tf32x3")
    grad_scores = probabilities * (grad_probabilities - delta[:, None])
    return (
        grad_query
        + tl.dot(grad_scores.to(sparse_values.dtype), sparse_values, input_precision="tf32x3")
        * SCALE
    )


# Counts drop only the multiple-of-16 hint; == 1 folds the single-document search.
@triton.jit(do_not_specialize_on_alignment=["num_documents", "SPARSE_SEQ_LEN"])
def _gather_attn_bwd_dq_shared(
    query_ptr,
    sparse_kv_ptr,
    local_kv_ptr,
    sparse_v_ptr,
    local_v_ptr,
    query_rope_ptr,
    sparse_rope_ptr,
    local_rope_ptr,
    kv_indices_ptr,
    selected_counts_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    attention_sink_ptr,
    grad_query_ptr,
    grad_query_rope_ptr,
    grad_sink_partials_ptr,
    delta_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    LOCAL_KV_STRIDES,
    SPARSE_V_STRIDES,
    LOCAL_V_STRIDES,
    QUERY_ROPE_STRIDES,
    SPARSE_ROPE_STRIDES,
    LOCAL_ROPE_STRIDES,
    KV_INDICES_STRIDES,
    num_documents,
    LSE_STRIDES,
    GRAD_SINK_PARTIALS_STRIDES,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    V_IS_K: tl.constexpr,
    STORE_DELTA: tl.constexpr,
    S,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    SPARSE_SEQ_LEN,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SPARSE_STAGES: tl.constexpr,
    RESOLVED_INDICES: tl.constexpr,
    tile_spans_ptr=None,
    DENSE_TILE_QUERIES: tl.constexpr = 0,
):
    """Compute query gradients across the query heads that share one KV head.

    Heads map to KV heads as in the shared forward schedule; V_IS_K reuses each key tile
    as its value tile for the dP = dO V^T product. STORE_DELTA also writes rowsum(dO * O),
    laid out like the LSE, for the key-major sparse dK/dV pass. RESOLVED_INDICES reads
    ``resolve_selected_rows`` output (compacted rows and counts) as in the shared forward.
    DENSE_TILE_QUERIES skips the queries ``_gather_attn_bwd_dq_dense`` already handled.

    With ROPE_DIM (shared-KV schedule only) the key-only query receives its own gradient,
    stored with the query RoPE strides. Local tensors are None when WINDOW is 0.
    """
    tl.static_assert(RESOLVED_INDICES or (KV_HEADS == 1 and V_IS_K))
    tl.static_assert(not (RESOLVED_INDICES and ROPE_DIM))
    S = tl.cast(S, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, USE_INT64_OFFSETS)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, USE_INT64_OFFSETS)
    KV_INDICES_STRIDES = offset_strides(KV_INDICES_STRIDES, USE_INT64_OFFSETS)
    LSE_STRIDES = offset_strides(LSE_STRIDES, USE_INT64_OFFSETS)
    GRAD_SINK_PARTIALS_STRIDES = offset_strides(GRAD_SINK_PARTIALS_STRIDES, USE_INT64_OFFSETS)
    if not V_IS_K:
        SPARSE_V_STRIDES = offset_strides(SPARSE_V_STRIDES, USE_INT64_OFFSETS)
    if WINDOW:
        LOCAL_KV_STRIDES = offset_strides(LOCAL_KV_STRIDES, USE_INT64_OFFSETS)
        if not V_IS_K:
            LOCAL_V_STRIDES = offset_strides(LOCAL_V_STRIDES, USE_INT64_OFFSETS)
    if ROPE_DIM:
        QUERY_ROPE_STRIDES = offset_strides(QUERY_ROPE_STRIDES, USE_INT64_OFFSETS)
        SPARSE_ROPE_STRIDES = offset_strides(SPARSE_ROPE_STRIDES, USE_INT64_OFFSETS)
        if WINDOW:
            LOCAL_ROPE_STRIDES = offset_strides(LOCAL_ROPE_STRIDES, USE_INT64_OFFSETS)

    sequence = tl.program_id(0)
    batch = tl.program_id(1)
    head_block = tl.program_id(2)
    if in_dense_tile(tile_spans_ptr, batch, sequence, S, DENSE_TILE_QUERIES):
        return

    kv_head, offsets_h, head_mask = head_block_heads(head_block, H, KV_HEADS, BLOCK_H)
    offsets_d = tl.arange(0, BLOCK_D)
    dimension_mask = offsets_d < D
    matrix_mask = head_mask[:, None] & dimension_mask[None, :]

    # Output and grad_output share the contiguous query layout.
    tensor_offsets = ptr_offset(
        (batch, offsets_h[:, None], sequence, offsets_d[None, :]), QUERY_STRIDES
    )
    query = tl.load(query_ptr + tensor_offsets, mask=matrix_mask, other=0.0)
    output = tl.load(output_ptr + tensor_offsets, mask=matrix_mask, other=0.0)
    grad_output = tl.load(grad_output_ptr + tensor_offsets, mask=matrix_mask, other=0.0)
    if ROPE_DIM:
        offsets_rope = tl.arange(0, ROPE_DIM)
        rope_offsets = ptr_offset(
            (batch, offsets_h[:, None], sequence, offsets_rope[None, :]), QUERY_ROPE_STRIDES
        )
        query_rope = tl.load(query_rope_ptr + rope_offsets, mask=head_mask[:, None], other=0.0)
        grad_query_rope = tl.zeros((BLOCK_H, ROPE_DIM), tl.float32)
    lse = tl.load(
        lse_ptr + ptr_offset((batch, offsets_h, sequence), LSE_STRIDES),
        mask=head_mask,
        other=0.0,
    )
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    if STORE_DELTA:
        tl.store(
            delta_ptr + ptr_offset((batch, offsets_h, sequence), LSE_STRIDES),
            delta,
            mask=head_mask,
        )
    grad_query = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        query_start, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, sequence, num_documents, S, USE_INT64_OFFSETS
        )

    if TOPK and RESOLVED_INDICES:
        offsets_k = tl.arange(0, BLOCK_K)
        selected_rows_ptr = kv_indices_ptr + ptr_offset((batch, sequence, 0), KV_INDICES_STRIDES)
        selected_count = tl.load(selected_counts_ptr + batch * S + sequence)
        full_end = selected_count // BLOCK_K * BLOCK_K
        for selected_start in tl.range(0, full_end, BLOCK_K, num_stages=SPARSE_STAGES):
            grad_query = _resolved_dq_tile(
                query,
                grad_output,
                lse,
                delta,
                sparse_kv_ptr,
                sparse_v_ptr,
                selected_rows_ptr + selected_start,
                selected_count - selected_start,
                batch,
                kv_head,
                offsets_k,
                offsets_d,
                dimension_mask,
                head_mask,
                grad_query,
                SPARSE_KV_STRIDES,
                SPARSE_V_STRIDES,
                V_IS_K,
                SCALE,
                False,
            )
        if full_end < selected_count:
            grad_query = _resolved_dq_tile(
                query,
                grad_output,
                lse,
                delta,
                sparse_kv_ptr,
                sparse_v_ptr,
                selected_rows_ptr + full_end,
                selected_count - full_end,
                batch,
                kv_head,
                offsets_k,
                offsets_d,
                dimension_mask,
                head_mask,
                grad_query,
                SPARSE_KV_STRIDES,
                SPARSE_V_STRIDES,
                V_IS_K,
                SCALE,
                True,
            )
    elif TOPK:
        offsets_k = tl.arange(0, BLOCK_K)
        for selected_start in tl.range(0, TOPK, BLOCK_K, num_stages=SPARSE_STAGES):
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
            # Only the shared-KV schedule reads raw indices: one KV head that is also V.
            sparse_values = tl.load(
                sparse_kv_ptr
                + ptr_offset(
                    (batch, 0, selected_idx[:, None], offsets_d[None, :]),
                    SPARSE_KV_STRIDES,
                ),
                mask=selected_valid[:, None] & dimension_mask[None, :],
                other=0.0,
            )
            scores = tl.dot(query, tl.trans(sparse_values), input_precision="tf32x3")
            if ROPE_DIM:
                sparse_rope = tl.load(
                    sparse_rope_ptr
                    + ptr_offset(
                        (batch, 0, selected_idx[:, None], offsets_rope[None, :]),
                        SPARSE_ROPE_STRIDES,
                    ),
                    mask=selected_valid[:, None],
                    other=0.0,
                )
                scores = tl.dot(
                    query_rope, tl.trans(sparse_rope), acc=scores, input_precision="tf32x3"
                )
            scores *= SCALE
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
            if ROPE_DIM:
                grad_query_rope += (
                    tl.dot(
                        grad_scores.to(sparse_rope.dtype),
                        sparse_rope,
                        input_precision="tf32x3",
                    )
                    * SCALE
                )

    if WINDOW:
        offsets_n_base = tl.arange(0, BLOCK_N)
        first_local_position = sequence - WINDOW + 1
        for local_start in tl.range(0, WINDOW, BLOCK_N, num_stages=2):
            offsets_n = first_local_position + local_start + offsets_n_base
            local_valid = (offsets_n >= 0) & (offsets_n <= sequence) & (offsets_n < S)
            if HAS_CU_SEQLENS:
                local_valid &= offsets_n >= query_start
            local_mask = local_valid[:, None] & dimension_mask[None, :]
            local_values = tl.load(
                local_kv_ptr
                + ptr_offset(
                    (batch, kv_head, offsets_n[:, None], offsets_d[None, :]),
                    LOCAL_KV_STRIDES,
                ),
                mask=local_mask,
                other=0.0,
            )
            if V_IS_K:
                local_v = local_values
            else:
                local_v = tl.load(
                    local_v_ptr
                    + ptr_offset(
                        (batch, kv_head, offsets_n[:, None], offsets_d[None, :]),
                        LOCAL_V_STRIDES,
                    ),
                    mask=local_mask,
                    other=0.0,
                )
            scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3")
            if ROPE_DIM:
                local_rope = tl.load(
                    local_rope_ptr
                    + ptr_offset(
                        (batch, 0, offsets_n[:, None], offsets_rope[None, :]),
                        LOCAL_ROPE_STRIDES,
                    ),
                    mask=local_valid[:, None],
                    other=0.0,
                )
                scores = tl.dot(
                    query_rope, tl.trans(local_rope), acc=scores, input_precision="tf32x3"
                )
            scores *= SCALE
            probabilities = tl.where(
                head_mask[:, None] & local_valid[None, :],
                tl.exp(scores - lse[:, None]),
                0.0,
            )
            grad_probabilities = tl.dot(grad_output, tl.trans(local_v), input_precision="tf32x3")
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            grad_query += (
                tl.dot(
                    grad_scores.to(local_values.dtype),
                    local_values,
                    input_precision="tf32x3",
                )
                * SCALE
            )
            if ROPE_DIM:
                grad_query_rope += (
                    tl.dot(
                        grad_scores.to(local_rope.dtype),
                        local_rope,
                        input_precision="tf32x3",
                    )
                    * SCALE
                )

    tl.store(grad_query_ptr + tensor_offsets, grad_query, mask=matrix_mask)
    if ROPE_DIM:
        tl.store(grad_query_rope_ptr + rope_offsets, grad_query_rope, mask=head_mask[:, None])

    sink = tl.load(attention_sink_ptr + offsets_h, mask=head_mask, other=0.0)
    sink_probability = tl.where(lse == -float("inf"), 0.0, tl.exp(sink - lse))
    sink_gradient = -sink_probability * delta
    tl.store(
        grad_sink_partials_ptr
        + ptr_offset((batch, offsets_h, sequence), GRAD_SINK_PARTIALS_STRIDES),
        sink_gradient,
        mask=head_mask,
    )


@triton.jit(
    do_not_specialize=[
        "KV_INDICES_BATCH_STRIDE",
        "num_documents",
        "LSE_BATCH_STRIDE",
        "LSE_HEAD_STRIDE",
        "S",
        "SPARSE_SEQ_LEN",
    ]
)
def _gather_attn_bwd_dsparse_kv_shared_atomic(
    query_ptr,
    sparse_kv_ptr,
    sparse_v_ptr,
    query_rope_ptr,
    sparse_rope_ptr,
    kv_indices_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_sparse_kv_ptr,
    grad_sparse_v_ptr,
    grad_sparse_rope_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    SPARSE_V_STRIDES,
    QUERY_ROPE_STRIDES,
    SPARSE_ROPE_STRIDES,
    KV_INDICES_BATCH_STRIDE,
    KV_INDICES_STRIDES,
    num_documents,
    LSE_BATCH_STRIDE,
    LSE_HEAD_STRIDE,
    LSE_STRIDES,
    GRAD_SPARSE_KV_STRIDES,
    GRAD_SPARSE_V_STRIDES,
    GRAD_SPARSE_ROPE_STRIDES,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    V_IS_K: tl.constexpr,
    S,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    SPARSE_SEQ_LEN,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    tile_spans_ptr=None,
    DENSE_TILE_QUERIES: tl.constexpr = 0,
):
    """Atomically scatter one query's sparse-KV gradient tile into its KV head's dKV.

    Head blocks map to KV heads as in the shared forward schedule. With V_IS_K the key and
    value gradients add into one shared dKV; otherwise dK and dV scatter separately.
    DENSE_TILE_QUERIES skips queries whose gradients ``_gather_attn_bwd_dsparse_kv_dense``
    already wrote. With ROPE_DIM (one shared KV head that is also V) the key-only sparse RoPE
    receives only dS^T Q_rope.
    """
    tl.static_assert(not ROPE_DIM or (KV_HEADS == 1 and V_IS_K))
    S = tl.cast(S, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, USE_INT64_OFFSETS)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, USE_INT64_OFFSETS)
    KV_INDICES_STRIDES = offset_strides(
        (KV_INDICES_BATCH_STRIDE, KV_INDICES_STRIDES[0], KV_INDICES_STRIDES[1]),
        USE_INT64_OFFSETS,
    )
    LSE_STRIDES = offset_strides(
        (LSE_BATCH_STRIDE, LSE_HEAD_STRIDE, LSE_STRIDES[0]), USE_INT64_OFFSETS
    )
    GRAD_SPARSE_KV_STRIDES = offset_strides(GRAD_SPARSE_KV_STRIDES, USE_INT64_OFFSETS)
    if not V_IS_K:
        SPARSE_V_STRIDES = offset_strides(SPARSE_V_STRIDES, USE_INT64_OFFSETS)
        GRAD_SPARSE_V_STRIDES = offset_strides(GRAD_SPARSE_V_STRIDES, USE_INT64_OFFSETS)
    if ROPE_DIM:
        QUERY_ROPE_STRIDES = offset_strides(QUERY_ROPE_STRIDES, USE_INT64_OFFSETS)
        SPARSE_ROPE_STRIDES = offset_strides(SPARSE_ROPE_STRIDES, USE_INT64_OFFSETS)
        GRAD_SPARSE_ROPE_STRIDES = offset_strides(GRAD_SPARSE_ROPE_STRIDES, USE_INT64_OFFSETS)

    query_position = tl.program_id(0)
    batch = tl.program_id(1)
    tile = tl.program_id(2)
    if in_dense_tile(tile_spans_ptr, batch, query_position, S, DENSE_TILE_QUERIES):
        return
    num_selected_tiles = tl.cdiv(TOPK, BLOCK_K)
    head_block = tile // num_selected_tiles
    selected_block = tile % num_selected_tiles

    kv_head, offsets_h, head_mask = head_block_heads(head_block, H, KV_HEADS, BLOCK_H)
    offsets_k = selected_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offsets_d = tl.arange(0, BLOCK_D)
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
            cu_seqlens_ptr, cu_seqlens_k_ptr, query_position, num_documents, S, USE_INT64_OFFSETS
        )
    selected_valid = (
        selected_mask
        & (selected_indices >= 0)
        & (selected_indices < candidate_end - candidate_start)
    )
    selected_indices = tl.where(selected_valid, selected_indices, 0) + candidate_start

    head_dimension_mask = head_mask[:, None] & dimension_mask[None, :]
    # Output and grad_output share the contiguous query layout.
    tensor_offsets = ptr_offset(
        (batch, offsets_h[:, None], query_position, offsets_d[None, :]),
        QUERY_STRIDES,
    )
    query = tl.load(query_ptr + tensor_offsets, mask=head_dimension_mask, other=0.0)
    output = tl.load(output_ptr + tensor_offsets, mask=head_dimension_mask, other=0.0)
    grad_output = tl.load(
        grad_output_ptr + tensor_offsets,
        mask=head_dimension_mask,
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + ptr_offset((batch, offsets_h, query_position), LSE_STRIDES),
        mask=head_mask,
        other=0.0,
    )
    selected_dimension_mask = selected_valid[:, None] & dimension_mask[None, :]
    sparse_values = tl.load(
        sparse_kv_ptr
        + ptr_offset(
            (batch, kv_head, selected_indices[:, None], offsets_d[None, :]),
            SPARSE_KV_STRIDES,
        ),
        mask=selected_dimension_mask,
        other=0.0,
    )
    if V_IS_K:
        sparse_v = sparse_values
    else:
        sparse_v = tl.load(
            sparse_v_ptr
            + ptr_offset(
                (batch, kv_head, selected_indices[:, None], offsets_d[None, :]),
                SPARSE_V_STRIDES,
            ),
            mask=selected_dimension_mask,
            other=0.0,
        )

    scores = tl.dot(query, tl.trans(sparse_values), input_precision="tf32x3")
    if ROPE_DIM:
        offsets_rope = tl.arange(0, ROPE_DIM)
        query_rope = tl.load(
            query_rope_ptr
            + ptr_offset(
                (batch, offsets_h[:, None], query_position, offsets_rope[None, :]),
                QUERY_ROPE_STRIDES,
            ),
            mask=head_mask[:, None],
            other=0.0,
        )
        sparse_rope = tl.load(
            sparse_rope_ptr
            + ptr_offset(
                (batch, 0, selected_indices[:, None], offsets_rope[None, :]),
                SPARSE_ROPE_STRIDES,
            ),
            mask=selected_valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query_rope, tl.trans(sparse_rope), acc=scores, input_precision="tf32x3")
    scores *= SCALE
    probabilities = tl.where(
        head_mask[:, None] & selected_valid[None, :],
        tl.exp(scores - lse[:, None]),
        0.0,
    )
    grad_probabilities = tl.dot(
        grad_output,
        tl.trans(sparse_v),
        input_precision="tf32x3",
    )
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    grad_scores = probabilities * (grad_probabilities - delta[:, None])
    if V_IS_K:
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
                (batch, kv_head, selected_indices[:, None], offsets_d[None, :]),
                GRAD_SPARSE_KV_STRIDES,
            ),
            grad_values,
            mask=selected_dimension_mask,
        )
    else:
        grad_value_tile = tl.dot(
            tl.trans(probabilities.to(grad_output.dtype)),
            grad_output,
            input_precision="tf32x3",
        )
        grad_key_tile = tl.dot(
            tl.trans((grad_scores * SCALE).to(query.dtype)),
            query,
            input_precision="tf32x3",
        )
        tl.atomic_add(
            grad_sparse_kv_ptr
            + ptr_offset(
                (batch, kv_head, selected_indices[:, None], offsets_d[None, :]),
                GRAD_SPARSE_KV_STRIDES,
            ),
            grad_key_tile,
            mask=selected_dimension_mask,
        )
        tl.atomic_add(
            grad_sparse_v_ptr
            + ptr_offset(
                (batch, kv_head, selected_indices[:, None], offsets_d[None, :]),
                GRAD_SPARSE_V_STRIDES,
            ),
            grad_value_tile,
            mask=selected_dimension_mask,
        )
    if ROPE_DIM:
        grad_rope = tl.dot(
            tl.trans((grad_scores * SCALE).to(query_rope.dtype)),
            query_rope,
            input_precision="tf32x3",
        )
        tl.atomic_add(
            grad_sparse_rope_ptr
            + ptr_offset(
                (batch, 0, selected_indices[:, None], offsets_rope[None, :]),
                GRAD_SPARSE_ROPE_STRIDES,
            ),
            grad_rope,
            mask=selected_valid[:, None],
        )


@triton.jit(do_not_specialize_on_alignment=["SPARSE_SEQ_LEN"])
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
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate a head tile into one partial shared sparse-KV gradient."""
    S = tl.cast(S, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, USE_INT64_OFFSETS)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, USE_INT64_OFFSETS)
    SELECTED_QUERIES_STRIDES = offset_strides(SELECTED_QUERIES_STRIDES, USE_INT64_OFFSETS)
    BLOCK_OFFSETS_STRIDES = offset_strides(BLOCK_OFFSETS_STRIDES, USE_INT64_OFFSETS)
    LSE_STRIDES = offset_strides(LSE_STRIDES, USE_INT64_OFFSETS)
    GRAD_SPARSE_KV_STRIDES = offset_strides(GRAD_SPARSE_KV_STRIDES, USE_INT64_OFFSETS)

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


def prune_shared_dq_configs(configs, args):
    if args["D"] == 512:
        return prune_wide_backward_configs(configs, args)
    return [config for config in configs if config.kwargs["BLOCK_N"] != 16]


def _zero_grad_sparse_kv(_grid, args):
    """Clear the zero-initialized outputs that candidates accumulate into or partially write."""
    outputs = [args["grad_sparse_kv_ptr"], args.get("grad_sparse_rope_ptr")]
    outputs = [output for output in outputs if output is not None]

    def restore():
        for output in outputs:
            output.zero_()

    return restore


def _restore_grad_sparse_kv_and_v(_grid, args):
    """Restore every atomic output, zeros or the dense tiles' gradients, before each launch.

    With V_IS_K the key and value outputs are the same tensor.
    """
    outputs = (
        args["grad_sparse_kv_ptr"],
        args["grad_sparse_v_ptr"],
        args.get("grad_sparse_rope_ptr"),
    )
    unique = {id(output): output for output in outputs if output is not None}.values()
    snapshots = [(output, output.clone()) for output in unique]

    def restore():
        for output, snapshot in snapshots:
            output.copy_(snapshot)

    return restore


BWD_DQ_SHARED = TritonTuner(
    _gather_attn_bwd_dq_shared,
    [
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps, num_stages=1)
        for block_n in (16, 64, 128)
        for num_warps in (4, 8)
    ],
    key=tuning_key(
        "query_ptr",
        "H",
        "KV_HEADS",
        "V_IS_K",
        "STORE_DELTA",
        "D",
        "ROPE_DIM",
        "TOPK",
        "WINDOW",
        "HAS_CU_SEQLENS",
    ),
    prune=prune_shared_dq_configs,
)


def prune_grouped_dq_configs(configs, args):
    """Bound the selection tile by the top-k and the local tile by the head dimension."""
    max_block_k = max(32, triton.next_power_of_2(args["TOPK"]))
    block_ns = (64,) if args["D"] > 128 or args["WINDOW"] == 0 else (64, 128)
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_K"] <= max_block_k and config.kwargs["BLOCK_N"] in block_ns
    ]


# The grouped schedule also tunes the selection tile and its software-pipelining depth.
BWD_DQ_GROUPED = TritonTuner(
    _gather_attn_bwd_dq_shared,
    [
        triton.Config(
            {"BLOCK_K": block_k, "BLOCK_N": block_n, "SPARSE_STAGES": stages},
            num_warps=num_warps,
            num_stages=1,
            maxnreg=maxnreg,
        )
        for block_k, stages in ((64, 3), (64, 2), (32, 3), (32, 2))
        for block_n in (64, 128)
        # Capping registers can fit a third 4-warp CTA per SM, worth a few spills.
        for num_warps, maxnreg in ((4, None), (4, 168), (8, None))
    ],
    key=tuning_key(
        "query_ptr",
        "H",
        "KV_HEADS",
        "V_IS_K",
        "STORE_DELTA",
        "D",
        "TOPK",
        "WINDOW",
        "HAS_CU_SEQLENS",
        "DENSE_TILE_QUERIES",
    ),
    prune=prune_grouped_dq_configs,
)
# Atomic configs accumulate into the same FP32 output, so it is cleared between trials.
BWD_DSPARSE_KV_SHARED_ATOMIC = TritonTuner(
    _gather_attn_bwd_dsparse_kv_shared_atomic,
    [
        triton.Config({"BLOCK_H": block_h, "BLOCK_K": block_k}, num_warps=num_warps, num_stages=1)
        for block_h, block_k, num_warps in (
            (16, 16, 4),
            (32, 16, 4),
            (16, 32, 4),
            (32, 32, 8),
            (16, 64, 8),
            (32, 64, 8),
        )
    ],
    key=tuning_key(
        "query_ptr",
        "H",
        "KV_HEADS",
        "V_IS_K",
        "D",
        "ROPE_DIM",
        "TOPK",
        "HAS_CU_SEQLENS",
        "DENSE_TILE_QUERIES",
    ),
    prune=prune_wide_backward_configs,
    reset=_restore_grad_sparse_kv_and_v,
)
# Different BLOCK_H configs write different partial slots, so stale slots are cleared.
BWD_DSPARSE_KV_SHARED = TritonTuner(
    _gather_attn_bwd_dsparse_kv_shared,
    [
        triton.Config({"BLOCK_H": block_h, "BLOCK_M": block_m}, num_warps=num_warps, num_stages=1)
        for block_h, block_m, num_warps in (
            (2, 32, 4),
            (4, 16, 4),
            (4, 32, 4),
            (4, 32, 8),
            (8, 16, 4),
            (8, 16, 8),
        )
    ],
    key=tuning_key("query_ptr", "H", "D", "TOPK"),
    reset=_zero_grad_sparse_kv,
)
