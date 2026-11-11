"""Backward kernels and launcher for Triton gather attention."""

import warnings

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.tune import TritonTuner, tuning_key
from attn_gym._backends.triton.utils import can_use_tma, ptr_offset, requires_int64_offsets

from ..routing import uses_grouped_schedule
from .dense_backward import BWD_DQ_DENSE, BWD_DSPARSE_KV_DENSE, document_key_tiles
from .forward import ROPE_REQUIREMENTS
from .key_major_backward import _gather_attn_bwd_dsparse_kv_grouped, build_tile_query_map
from .primitives import (
    DENSE_TILE_ROWS,
    TileConfig,
    can_use_shared_kv_schedule,
    causal_window_mask,
    dense_tile_shape,
    load_bhsd,
    load_document_bounds,
    offset_strides,
    optional_strides,
    plan_dense_tiles,
    prune_wide_backward_configs,
    resolve_selected_rows,
    select_tiles,
    store_bhsd,
)
from .shared_backward import (
    BWD_DQ_GROUPED,
    BWD_DQ_SHARED,
    BWD_DSPARSE_KV_SHARED,
    BWD_DSPARSE_KV_SHARED_ATOMIC,
)


# Counts drop only the multiple-of-16 hint; == 1 folds the single-document search.
@triton.jit(do_not_specialize_on_alignment=["num_documents", "SPARSE_SEQ_LEN"])
def _gather_attn_bwd_dq(
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
    grad_sink_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    LOCAL_KV_STRIDES,
    KV_INDICES_STRIDES,
    num_documents,
    LSE_STRIDES,
    H: tl.constexpr,
    S,
    D: tl.constexpr,
    SPARSE_SEQ_LEN,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    NUM_LOCAL_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate query and sink gradients over both attention branches.

    Local KV is None when WINDOW is 0.
    """
    S = tl.cast(S, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, USE_INT64_OFFSETS)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, USE_INT64_OFFSETS)
    if WINDOW:
        LOCAL_KV_STRIDES = offset_strides(LOCAL_KV_STRIDES, USE_INT64_OFFSETS)
    KV_INDICES_STRIDES = offset_strides(KV_INDICES_STRIDES, USE_INT64_OFFSETS)
    LSE_STRIDES = offset_strides(LSE_STRIDES, USE_INT64_OFFSETS)

    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    if USE_INT64_OFFSETS:
        query_block = query_block.to(tl.int64)
        batch_head = batch_head.to(tl.int64)
    head = batch_head % H
    batch = batch_head // H

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D
    matrix_mask = query_mask[:, None] & dimension_mask[None, :]

    query = load_bhsd(query_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask)
    output = load_bhsd(output_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask)
    grad_output = load_bhsd(
        grad_output_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask
    )
    lse = tl.load(
        lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
        mask=query_mask,
        other=0.0,
    )
    # Keep both products and the reduction in FP32, not the saved output dtype.
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        query_start, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, offsets_m, num_documents, S, USE_INT64_OFFSETS
        )

    for selected_slot in tl.range(0, TOPK):
        selected_idx = tl.load(
            kv_indices_ptr
            + batch * KV_INDICES_STRIDES[0]
            + offsets_m * KV_INDICES_STRIDES[1]
            + selected_slot * KV_INDICES_STRIDES[2],
            mask=query_mask,
            other=0,
        )
        valid = query_mask & (selected_idx >= 0) & (selected_idx < candidate_end - candidate_start)
        # Mask local coordinates before adding the document base, including int64 sentinels.
        selected_idx = tl.where(valid, selected_idx, 0) + candidate_start
        sparse_value = load_bhsd(
            sparse_kv_ptr,
            SPARSE_KV_STRIDES,
            batch,
            head,
            selected_idx,
            offsets_d,
            valid[:, None] & dimension_mask[None, :],
        )
        scores = tl.sum(query.to(tl.float32) * sparse_value.to(tl.float32), axis=1) * SCALE
        probabilities = tl.where(valid, tl.exp(scores - lse), 0.0)
        grad_probs = tl.sum(grad_output.to(tl.float32) * sparse_value.to(tl.float32), axis=1)
        grad_scores = probabilities * (grad_probs - delta)
        grad_query += grad_scores[:, None] * sparse_value * SCALE

    if WINDOW:
        # FP32 runs these dots only on the 16x16 tile, where exact dots fit and are faster
        # than tf32x3 (see GENERIC_DQ_TILES); other dtypes ignore the precision.
        first_key = query_block * BLOCK_M - WINDOW + 1
        offsets_n_base = tl.arange(0, BLOCK_N)
        for key_tile in tl.range(0, NUM_LOCAL_TILES):
            local_start = first_key + key_tile * BLOCK_N
            offsets_n = local_start + offsets_n_base
            key_mask = (offsets_n >= 0) & (offsets_n < S)
            local_values = load_bhsd(
                local_kv_ptr,
                LOCAL_KV_STRIDES,
                batch,
                head,
                offsets_n,
                offsets_d,
                key_mask[:, None] & dimension_mask[None, :],
            )
            scores = tl.dot(query, tl.trans(local_values), input_precision="ieee") * SCALE
            valid = causal_window_mask(offsets_m, offsets_n, query_mask, key_mask, WINDOW)
            if HAS_CU_SEQLENS:
                valid &= offsets_n[None, :] >= query_start[:, None]

            probabilities = tl.exp(scores - lse[:, None])
            probabilities = tl.where(valid, probabilities, 0.0)
            grad_probabilities = tl.dot(
                grad_output, tl.trans(local_values), input_precision="ieee"
            )
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            grad_query += (
                tl.dot(
                    grad_scores.to(local_values.dtype),
                    local_values,
                    input_precision="ieee",
                )
                * SCALE
            )

    store_bhsd(
        grad_query_ptr,
        grad_query,
        QUERY_STRIDES,
        batch,
        head,
        offsets_m,
        offsets_d,
        matrix_mask,
    )

    # Sink gradient
    sink = tl.load(attention_sink_ptr + head)
    sink_probability = tl.where(lse == -float("inf"), 0.0, tl.exp(sink - lse))
    sink_gradient = tl.where(query_mask, -sink_probability * delta, 0.0)
    # Output-owned per-block partials, reduced in a fixed order on the host, keep the
    # sink gradient deterministic; atomics across query blocks would not be.
    tl.store(
        grad_sink_ptr + batch_head * tl.cdiv(S, BLOCK_M) + query_block,
        tl.sum(sink_gradient, axis=0),
    )


@triton.jit(do_not_specialize_on_alignment=["num_documents"])
def _gather_attn_bwd_dlocal_kv(
    query_ptr,
    local_kv_ptr,
    local_v_ptr,
    query_rope_ptr,
    local_rope_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_local_kv_ptr,
    grad_local_v_ptr,
    grad_local_rope_ptr,
    QUERY_STRIDES,
    LOCAL_KV_STRIDES,
    LOCAL_V_STRIDES,
    GRAD_LOCAL_KV_STRIDES,
    GRAD_LOCAL_V_STRIDES,
    QUERY_ROPE_STRIDES,
    LOCAL_ROPE_STRIDES,
    GRAD_LOCAL_ROPE_STRIDES,
    num_documents,
    LSE_STRIDES,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    V_IS_K: tl.constexpr,
    S,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Backward for dKV (local sliding window).

    Each program writes its query head's partial gradient; query head h reads KV head
    h // GROUP, and the launcher sums each group's partials. With V_IS_K the key and value
    gradients add into one dKV; otherwise dK and dV are written separately. With ROPE_DIM
    (GROUP == 1 and V_IS_K only), key-only query and local RoPE tensors join the scores, and
    the local RoPE receives only dS^T Q_rope.
    """
    tl.static_assert(not ROPE_DIM or (GROUP == 1 and V_IS_K))
    S = tl.cast(S, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, USE_INT64_OFFSETS)
    LOCAL_KV_STRIDES = offset_strides(LOCAL_KV_STRIDES, USE_INT64_OFFSETS)
    GRAD_LOCAL_KV_STRIDES = offset_strides(GRAD_LOCAL_KV_STRIDES, USE_INT64_OFFSETS)
    if not V_IS_K:
        LOCAL_V_STRIDES = offset_strides(LOCAL_V_STRIDES, USE_INT64_OFFSETS)
        GRAD_LOCAL_V_STRIDES = offset_strides(GRAD_LOCAL_V_STRIDES, USE_INT64_OFFSETS)
    LSE_STRIDES = offset_strides(LSE_STRIDES, USE_INT64_OFFSETS)
    if ROPE_DIM:
        QUERY_ROPE_STRIDES = offset_strides(QUERY_ROPE_STRIDES, USE_INT64_OFFSETS)
        LOCAL_ROPE_STRIDES = offset_strides(LOCAL_ROPE_STRIDES, USE_INT64_OFFSETS)
        GRAD_LOCAL_ROPE_STRIDES = offset_strides(GRAD_LOCAL_ROPE_STRIDES, USE_INT64_OFFSETS)

    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    if USE_INT64_OFFSETS:
        key_block = key_block.to(tl.int64)
        batch_head = batch_head.to(tl.int64)
    head = batch_head % H
    batch = batch_head // H
    if GROUP == 1:
        kv_head = head
    else:
        kv_head = head // GROUP

    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, BLOCK_D)
    key_mask = offsets_n < S
    dimension_mask = offsets_d < D

    local_values = load_bhsd(
        local_kv_ptr,
        LOCAL_KV_STRIDES,
        batch,
        kv_head,
        offsets_n,
        offsets_d,
        key_mask[:, None] & dimension_mask[None, :],
    )
    if V_IS_K:
        local_v = local_values
    else:
        local_v = load_bhsd(
            local_v_ptr,
            LOCAL_V_STRIDES,
            batch,
            kv_head,
            offsets_n,
            offsets_d,
            key_mask[:, None] & dimension_mask[None, :],
        )
    if ROPE_DIM:
        offsets_rope = tl.arange(0, ROPE_DIM)
        local_rope = load_bhsd(
            local_rope_ptr,
            LOCAL_ROPE_STRIDES,
            batch,
            head,
            offsets_n,
            offsets_rope,
            key_mask[:, None],
        )
        grad_rope = tl.zeros((BLOCK_N, ROPE_DIM), tl.float32)

    if HAS_CU_SEQLENS:
        _, key_end, _, _ = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, offsets_n, num_documents, S, USE_INT64_OFFSETS
        )

    grad_values = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    if not V_IS_K:
        grad_keys = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    first_query = key_block * BLOCK_N
    offsets_m_base = tl.arange(0, BLOCK_M)

    for query_tile in tl.range(0, NUM_QUERY_TILES):
        query_start = first_query + query_tile * BLOCK_M
        offsets_m = query_start + offsets_m_base
        query_mask = offsets_m < S
        matrix_mask = query_mask[:, None] & dimension_mask[None, :]
        query = load_bhsd(query_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask)
        scores = tl.dot(query, tl.trans(local_values), input_precision=DOT_PRECISION)
        if ROPE_DIM:
            query_rope = load_bhsd(
                query_rope_ptr,
                QUERY_ROPE_STRIDES,
                batch,
                head,
                offsets_m,
                offsets_rope,
                query_mask[:, None],
            )
            scores = tl.dot(
                query_rope, tl.trans(local_rope), acc=scores, input_precision=DOT_PRECISION
            )
        scores *= SCALE
        output = load_bhsd(
            output_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask
        )
        grad_output = load_bhsd(
            grad_output_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask
        )
        lse = tl.load(
            lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
            mask=query_mask,
            other=0.0,
        )
        delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, key_mask, WINDOW)
        if HAS_CU_SEQLENS:
            valid &= offsets_m[:, None] < key_end[None, :]

        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(local_v), input_precision=DOT_PRECISION)
        # A masked query may have a nonfinite delta from another document.
        grad_scores = tl.where(valid, probabilities * (grad_probabilities - delta[:, None]), 0.0)
        grad_values += tl.dot(
            tl.trans(probabilities.to(grad_output.dtype)),
            grad_output,
            input_precision=DOT_PRECISION,
        )
        grad_key_tile = (
            tl.dot(
                tl.trans(grad_scores.to(query.dtype)),
                query,
                input_precision=DOT_PRECISION,
            )
            * SCALE
        )
        if V_IS_K:
            grad_values += grad_key_tile
        else:
            grad_keys += grad_key_tile
        if ROPE_DIM:
            grad_rope += (
                tl.dot(
                    tl.trans(grad_scores.to(query_rope.dtype)),
                    query_rope,
                    input_precision=DOT_PRECISION,
                )
                * SCALE
            )

    store_bhsd(
        grad_local_kv_ptr,
        grad_values if V_IS_K else grad_keys,
        GRAD_LOCAL_KV_STRIDES,
        batch,
        head,
        offsets_n,
        offsets_d,
        key_mask[:, None] & dimension_mask[None, :],
    )
    if not V_IS_K:
        store_bhsd(
            grad_local_v_ptr,
            grad_values,
            GRAD_LOCAL_V_STRIDES,
            batch,
            head,
            offsets_n,
            offsets_d,
            key_mask[:, None] & dimension_mask[None, :],
        )
    if ROPE_DIM:
        store_bhsd(
            grad_local_rope_ptr,
            grad_rope,
            GRAD_LOCAL_ROPE_STRIDES,
            batch,
            head,
            offsets_n,
            offsets_rope,
            key_mask[:, None],
        )


@triton.jit(do_not_specialize_on_alignment=["num_documents"])
def _gather_attn_bwd_dlocal_kv_tma(
    query_desc,
    local_desc,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    output_desc,
    grad_output_desc,
    lse_ptr,
    grad_local_desc,
    num_documents,
    LSE_STRIDES,
    H: tl.constexpr,
    S,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """TMA backward for local-KV gradients."""
    S = tl.cast(S, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    LSE_STRIDES = offset_strides(LSE_STRIDES, USE_INT64_OFFSETS)

    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    key_mask = offsets_n < S
    local_values = tl.reshape(
        local_desc.load([batch, head, key_block * BLOCK_N, 0]),
        (BLOCK_N, BLOCK_D),
    )

    if HAS_CU_SEQLENS:
        _, key_end, _, _ = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, offsets_n, num_documents, S, USE_INT64_OFFSETS
        )

    grad_values = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    first_query = key_block * BLOCK_N
    offsets_m_base = tl.arange(0, BLOCK_M)

    for query_tile in tl.range(0, NUM_QUERY_TILES):
        query_start = first_query + query_tile * BLOCK_M
        offsets_m = query_start + offsets_m_base
        query_mask = offsets_m < S
        query = tl.reshape(
            query_desc.load([batch, head, query_start, 0]),
            (BLOCK_M, BLOCK_D),
        )
        output = tl.reshape(
            output_desc.load([batch, head, query_start, 0]),
            (BLOCK_M, BLOCK_D),
        )
        grad_output = tl.reshape(
            grad_output_desc.load([batch, head, query_start, 0]),
            (BLOCK_M, BLOCK_D),
        )
        lse = tl.load(
            lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
            mask=query_mask,
            other=0.0,
        )
        delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, key_mask, WINDOW)
        if HAS_CU_SEQLENS:
            valid &= offsets_m[:, None] < key_end[None, :]

        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        probabilities = tl.where(valid, tl.exp(scores - lse[:, None]), 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(local_values), input_precision="tf32x3")
        # A masked query may have a nonfinite delta from another document.
        grad_scores = tl.where(valid, probabilities * (grad_probabilities - delta[:, None]), 0.0)
        grad_values += tl.dot(
            tl.trans(probabilities.to(grad_output.dtype)),
            grad_output,
            input_precision="tf32x3",
        )
        grad_values += (
            tl.dot(
                tl.trans(grad_scores.to(query.dtype)),
                query,
                input_precision="tf32x3",
            )
            * SCALE
        )

    grad_local_desc.store(
        [batch, head, key_block * BLOCK_N, 0],
        tl.reshape(grad_values, (1, 1, BLOCK_N, BLOCK_D)),
    )


@triton.jit(do_not_specialize_on_alignment=["SPARSE_SEQ_LEN"])
def _gather_attn_bwd_dsparse_kv(
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
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute sparse-KV gradients from an inverted index of selecting queries."""
    S = tl.cast(S, tl.int64 if USE_INT64_OFFSETS else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, USE_INT64_OFFSETS)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, USE_INT64_OFFSETS)
    SELECTED_QUERIES_STRIDES = offset_strides(SELECTED_QUERIES_STRIDES, USE_INT64_OFFSETS)
    BLOCK_OFFSETS_STRIDES = offset_strides(BLOCK_OFFSETS_STRIDES, USE_INT64_OFFSETS)
    LSE_STRIDES = offset_strides(LSE_STRIDES, USE_INT64_OFFSETS)
    GRAD_SPARSE_KV_STRIDES = offset_strides(GRAD_SPARSE_KV_STRIDES, USE_INT64_OFFSETS)

    sparse_index = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m_base = tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    dot_rows = tl.arange(0, 16)
    dimension_mask = offsets_d < D

    sparse_value = tl.load(
        sparse_kv_ptr
        + batch * SPARSE_KV_STRIDES[0]
        + head * SPARSE_KV_STRIDES[1]
        + sparse_index * SPARSE_KV_STRIDES[2]
        + offsets_d * SPARSE_KV_STRIDES[3],
        mask=dimension_mask,
        other=0.0,
    )
    dot_value = tl.where(
        dot_rows[None, :] == 0,
        sparse_value[:, None],
        0.0,
    )
    grad_value = tl.zeros((16, BLOCK_D), tl.float32)
    entry_start = tl.load(
        block_offsets_ptr
        + batch * BLOCK_OFFSETS_STRIDES[0]
        + sparse_index * BLOCK_OFFSETS_STRIDES[1]
    )
    entry_end = tl.load(
        block_offsets_ptr
        + batch * BLOCK_OFFSETS_STRIDES[0]
        + (sparse_index + 1) * BLOCK_OFFSETS_STRIDES[1]
    )

    for entry_tile in tl.range(entry_start, entry_end, BLOCK_M):
        entry_offsets = entry_tile + offsets_m_base
        entry_mask = entry_offsets < entry_end
        query_positions = tl.load(
            selected_queries_ptr
            + batch * SELECTED_QUERIES_STRIDES[0]
            + entry_offsets * SELECTED_QUERIES_STRIDES[1],
            mask=entry_mask,
            other=0,
        )
        query_mask = entry_mask & (query_positions >= 0) & (query_positions < S)
        matrix_mask = query_mask[:, None] & dimension_mask[None, :]
        query = load_bhsd(
            query_ptr, QUERY_STRIDES, batch, head, query_positions, offsets_d, matrix_mask
        )
        output = load_bhsd(
            output_ptr, QUERY_STRIDES, batch, head, query_positions, offsets_d, matrix_mask
        )
        grad_output = load_bhsd(
            grad_output_ptr, QUERY_STRIDES, batch, head, query_positions, offsets_d, matrix_mask
        )
        lse = tl.load(
            lse_ptr
            + batch * LSE_STRIDES[0]
            + head * LSE_STRIDES[1]
            + query_positions * LSE_STRIDES[2],
            mask=query_mask,
            other=0.0,
        )
        score_tile = tl.dot(query, dot_value, input_precision="tf32x3")
        scores = tl.sum(score_tile * (dot_rows[None, :] == 0), axis=1) * SCALE
        grad_probability_tile = tl.dot(grad_output, dot_value, input_precision="tf32x3")
        grad_probabilities = tl.sum(grad_probability_tile * (dot_rows[None, :] == 0), axis=1)
        delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
        probabilities = tl.where(query_mask, tl.exp(scores - lse), 0.0)
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
        + batch * GRAD_SPARSE_KV_STRIDES[0]
        + head * GRAD_SPARSE_KV_STRIDES[1]
        + sparse_index * GRAD_SPARSE_KV_STRIDES[2]
        + offsets_d * GRAD_SPARSE_KV_STRIDES[3],
        tl.sum(grad_value * (dot_rows[:, None] == 0), axis=0),
        mask=dimension_mask,
    )


_BWD_DLOCAL_KV_TMA = TritonTuner(
    _gather_attn_bwd_dlocal_kv_tma,
    [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (4, 8)
        for num_stages in (1, 3)
    ],
    # The fixed tile changes the winner, so it is part of the key.
    key=tuning_key("query_desc", "H", "D", "WINDOW", "HAS_CU_SEQLENS", "BLOCK_M", "BLOCK_N"),
)
_BWD_DSPARSE_KV = TritonTuner(
    _gather_attn_bwd_dsparse_kv,
    [
        triton.Config({"BLOCK_M": block_m}, num_warps=num_warps, num_stages=num_stages)
        for block_m in (16, 32, 64)
        for num_warps in (4, 8)
        for num_stages in (1, 3)
    ],
    key=tuning_key("query_ptr", "H", "D", "TOPK"),
    prune=prune_wide_backward_configs,
)


# No stride tuple here scales with S, so unlike the attention kernels S is runtime too.
@triton.jit(do_not_specialize_on_alignment=["num_documents", "S"])
def _index_query_map_keys(
    kv_indices_ptr,
    sort_keys_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    num_documents,
    stride_b,
    stride_s,
    stride_k,
    S,
    TOPK: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Form packed global sort keys without changing attention's document-local indices."""
    strides = offset_strides((stride_b, stride_s, stride_k), USE_INT64_OFFSETS)
    query_position = tl.program_id(0)
    tile = tl.program_id(1)
    batch = tl.program_id(2)
    if USE_INT64_OFFSETS:
        query_position = query_position.to(tl.int64)
        tile = tile.to(tl.int64)
        batch = batch.to(tl.int64)
    row = batch * S + query_position
    slots = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    _, _, candidate_start, candidate_end = load_document_bounds(
        cu_seqlens_ptr, cu_seqlens_k_ptr, query_position, num_documents, S, USE_INT64_OFFSETS
    )
    indices = tl.load(
        kv_indices_ptr + ptr_offset((batch, query_position, slots), strides),
        mask=slots < TOPK,
        other=-1,
    )
    valid = (indices >= 0) & (indices < candidate_end - candidate_start)
    indices = tl.where(valid, indices, 0) + candidate_start
    tl.store(sort_keys_ptr + row * TOPK + slots, tl.where(valid, indices, -1), slots < TOPK)


def _build_index_query_map(
    kv_indices: torch.Tensor,
    sparse_seq_len: int,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert raw query-major selections into global block-major query lists (CSR format)."""
    batch, seq_len, topk = kv_indices.shape
    if seq_len == 0 or topk == 0:
        selected_queries = torch.empty(batch, 0, device=kv_indices.device, dtype=torch.int32)
        block_offsets = torch.zeros(
            batch, sparse_seq_len + 1, device=kv_indices.device, dtype=torch.int32
        )
        return selected_queries, block_offsets

    if cu_seqlens is not None:
        # This workspace belongs only to the reverse map, not the attention inputs or tape.
        sort_keys = kv_indices.new_empty((batch, seq_len * topk))
        _index_query_map_keys[(seq_len, triton.cdiv(topk, 256), batch)](
            kv_indices,
            sort_keys,
            cu_seqlens,
            cu_seqlens_k,
            cu_seqlens.numel() - 1,
            *kv_indices.stride(),
            S=seq_len,
            TOPK=topk,
            USE_INT64_OFFSETS=requires_int64_offsets(
                kv_indices, sort_keys, cu_seqlens, cu_seqlens_k
            ),
            BLOCK_K=256,
            num_warps=4,
        )
    else:
        sort_keys = kv_indices.flatten(1)
    sorted_indices, sorted_entries = torch.sort(sort_keys, dim=-1)
    selected_queries = torch.div(sorted_entries, topk, rounding_mode="floor").to(torch.int32)
    block_ids = torch.arange(
        sparse_seq_len + 1, device=kv_indices.device, dtype=sorted_indices.dtype
    )
    block_ids = block_ids.unsqueeze(0).expand(batch, -1).contiguous()
    # Searching [0, sparse_seq_len] excludes negative and out-of-pool entries in either mode.
    block_offsets = torch.searchsorted(sorted_indices, block_ids).to(torch.int32)
    return selected_queries.contiguous(), block_offsets.contiguous()


# Preferred first; later entries fit smaller shared memory (e.g. SM80, or SM86 at D=256).
# Software-pipelined query tiles make the local-KV gradient the largest consumer.
#
# The 16x16 tiles (tl.dot's minimum) run FP32 dots as "ieee": tf32x3 dots do not fit 99 KiB of
# shared memory at BLOCK_D=512. The TMA local-KV kernel keeps tf32x3, which is faster there.
#
# FP32 with a local window always takes the last dQ tile, so the other dQ entries' shared_rows
# hold for BF16 and FP16 only.
GENERIC_DQ_TILES = (
    TileConfig(64, 32, 8, 3, shared_rows=208, tmem_cols=512),
    TileConfig(64, 32, 8, 2, shared_rows=192, tmem_cols=512),
    TileConfig(32, 32, 4, 2, shared_rows=112),
    TileConfig(16, 16, 4, 1, shared_rows=64),
)
GENERIC_DLOCAL_KV_TILES = (
    TileConfig(64, 32, 8, 3, shared_rows=484, tmem_cols=512),
    TileConfig(64, 32, 8, 2, shared_rows=290, tmem_cols=512),
    TileConfig(32, 32, 4, 2, shared_rows=177),
    TileConfig(16, 16, 4, 1, shared_rows=68, dot_precision="ieee"),
)


def _launch_backward(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor | None,
    kv_indices: torch.Tensor,
    selected_queries: torch.Tensor | None,
    block_offsets: torch.Tensor | None,
    attention_sink: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    sliding_window_size: int,
    share_kv: bool,
    scale: float,
    *,
    kv_heads: int,
    sparse_v: torch.Tensor | None = None,
    local_v: torch.Tensor | None = None,
    query_rope: torch.Tensor | None = None,
    sparse_rope: torch.Tensor | None = None,
    local_rope: torch.Tensor | None = None,
    autotune: bool,
) -> tuple[torch.Tensor | None, ...]:
    """Launch backward kernels for gather attention.

    Returns ``(dq, dsparse, dlocal, dq_rope, dsparse_rope, dlocal_rope, dsink, dsparse_v,
    dlocal_v)``; a gradient is None exactly when its input is. ``kv_heads`` is the KV head
    count before the op broadcasts a single shared head; separate values or
    1 < kv_heads < heads run on the grouped schedule. Key-only RoPE runs only on the
    shared-KV schedule, whose sparse gradients are atomic, so the reverse query map may be
    None. Without ``autotune`` every kernel launches its pinned configuration.
    """
    if uses_grouped_schedule(query.shape[1], kv_heads, sparse_v is not None):
        dq, dsparse, dlocal, dsink, dsparse_v, dlocal_v = _launch_grouped_backward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            output,
            lse,
            grad_output,
            sliding_window_size,
            scale,
            kv_heads=kv_heads,
            sparse_v=sparse_v,
            local_v=local_v,
            autotune=autotune,
        )
        # Zero-width RoPE pairs add nothing to the scores; their gradients are empty.
        rope_grads = (
            None if rope is None else rope.new_empty((*rope.shape[:1], heads, *rope.shape[2:]))
            for rope, heads in (
                (query_rope, query.shape[1]),
                (sparse_rope, kv_heads),
                (local_rope, kv_heads),
            )
        )
        return (dq, dsparse, dlocal, *rope_grads, dsink, dsparse_v, dlocal_v)
    if query_rope is not None and query_rope.shape[-1] == 0:
        # Zero-width RoPE adds nothing to the scores; its gradients are empty.
        grads = _launch_backward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            selected_queries,
            block_offsets,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            output,
            lse,
            grad_output,
            sliding_window_size,
            share_kv,
            scale,
            kv_heads=kv_heads,
            autotune=autotune,
        )
        kv_heads = 1 if share_kv else query.shape[1]
        rope_grads = (
            None if rope is None else rope.new_empty((*rope.shape[:1], heads, *rope.shape[2:]))
            for rope, heads in (
                (query_rope, query_rope.shape[1]),
                (sparse_rope, kv_heads),
                (local_rope, kv_heads),
            )
        )
        return (*grads[:3], *rope_grads, *grads[6:])

    batch, heads, seq_len, head_dim = query.shape
    sparse_seq_len = sparse_kv.shape[2]
    topk = kv_indices.shape[-1]
    window = sliding_window_size
    kv_heads = 1 if share_kv else heads
    rope_dim = 0 if query_rope is None else query_rope.shape[-1]
    block_d = max(16, triton.next_power_of_2(head_dim))
    # Output and grad_output reuse the contiguous query strides.
    output = output.contiguous()
    grad_output = grad_output.contiguous()
    if query_rope is not None:
        query_rope = query_rope.contiguous()
    # Kernels never read local tensors without a window; return zeros in the KV-head shape.
    kernel_local_kv = local_kv if window else None
    kernel_local_rope = local_rope if window else None

    def zero_local_grad(tensor: torch.Tensor | None) -> torch.Tensor | None:
        return None if tensor is None else tensor.new_zeros((batch, kv_heads, *tensor.shape[2:]))

    grad_query = torch.empty_like(query)
    grad_query_rope = None if query_rope is None else torch.empty_like(query_rope)
    # Shared KV needs one partial gradient per query head before this launcher sums them.
    grad_local_kv = local_kv.new_empty(local_kv.shape) if window else zero_local_grad(local_kv)
    grad_local_rope = (
        local_rope.new_empty(local_rope.shape)
        if window and rope_dim
        else zero_local_grad(local_rope)
    )

    has_cu_seqlens = cu_seqlens is not None
    num_documents = cu_seqlens.numel() - 1 if has_cu_seqlens else 0
    use_int64_offsets = requires_int64_offsets(
        query,
        sparse_kv,
        kernel_local_kv,
        query_rope,
        sparse_rope,
        kernel_local_rope,
        kv_indices,
        selected_queries,
        block_offsets,
        output,
        grad_output,
        lse,
        grad_local_kv if window else None,
        grad_local_rope if window else None,
        cu_seqlens,
        cu_seqlens_k,
    )

    # Zero head strides share values; share_kv also requests the final head reduction.
    use_shared_schedule = share_kv and can_use_shared_kv_schedule(
        query, sparse_kv, kernel_local_kv, window, query_rope, sparse_rope, kernel_local_rope
    )
    if rope_dim:
        if not use_shared_schedule:
            raise ValueError(ROPE_REQUIREMENTS)
        if torch.are_deterministic_algorithms_enabled():
            message = "Triton gather_attn has no deterministic backward for key-only RoPE."
            if not torch.is_deterministic_algorithms_warn_only_enabled():
                raise RuntimeError(message)
            warnings.warn(message, stacklevel=2)
    if use_shared_schedule:
        block_h = 16 if head_dim == 512 else min(32, triton.next_power_of_2(heads))
        block_k = 16 if head_dim == 512 else max(16, min(64, triton.next_power_of_2(topk)))
        grad_sink_partials = torch.empty(
            batch, heads, seq_len, device=query.device, dtype=torch.float32
        )
        (BWD_DQ_SHARED if autotune else BWD_DQ_SHARED.pinned)[
            (seq_len, batch, triton.cdiv(heads, block_h))
        ](
            query,
            sparse_kv,
            kernel_local_kv,
            sparse_kv,
            kernel_local_kv,
            query_rope,
            sparse_rope,
            kernel_local_rope,
            kv_indices,
            None,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            attention_sink,
            grad_query,
            grad_query_rope,
            grad_sink_partials,
            None,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            LOCAL_KV_STRIDES=optional_strides(kernel_local_kv),
            SPARSE_V_STRIDES=sparse_kv.stride(),
            LOCAL_V_STRIDES=optional_strides(kernel_local_kv),
            QUERY_ROPE_STRIDES=optional_strides(query_rope),
            SPARSE_ROPE_STRIDES=optional_strides(sparse_rope),
            LOCAL_ROPE_STRIDES=optional_strides(kernel_local_rope),
            KV_INDICES_STRIDES=kv_indices.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            GRAD_SINK_PARTIALS_STRIDES=grad_sink_partials.stride(),
            H=heads,
            KV_HEADS=1,
            V_IS_K=True,
            STORE_DELTA=False,
            S=seq_len,
            D=head_dim,
            ROPE_DIM=rope_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            WINDOW=window,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            USE_INT64_OFFSETS=use_int64_offsets,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            SPARSE_STAGES=2,
            RESOLVED_INDICES=False,
        )
    else:
        # Full-width D=512 operands otherwise exceed per-block shared-memory limits. FP32 window
        # dots on the larger tiles stage more shared memory than shared_rows estimates.
        block_m, block_n, num_warps, num_stages, *_ = (
            GENERIC_DQ_TILES[-1]
            if head_dim == 512 or (window and query.dtype == torch.float32)
            else select_tiles(GENERIC_DQ_TILES, block_d, query.element_size(), query.device)
        )
        num_local_key_tiles = triton.cdiv(window + block_m - 1, block_n) if window else 0
        grad_sink_partials = torch.empty(
            batch,
            heads,
            triton.cdiv(seq_len, block_m),
            device=query.device,
            dtype=torch.float32,
        )
        _gather_attn_bwd_dq[(triton.cdiv(seq_len, block_m), batch * heads)](
            query,
            sparse_kv,
            kernel_local_kv,
            kv_indices,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            attention_sink,
            grad_query,
            grad_sink_partials,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            LOCAL_KV_STRIDES=optional_strides(kernel_local_kv),
            KV_INDICES_STRIDES=kv_indices.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            WINDOW=window,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            USE_INT64_OFFSETS=use_int64_offsets,
            NUM_LOCAL_TILES=num_local_key_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    grad_sink_fp32 = grad_sink_partials.sum(dim=(0, 2))

    if window:
        # Key-only RoPE adds a second operand tile; size the tiles for both.
        local_tile = (
            GENERIC_DLOCAL_KV_TILES[-1]
            if head_dim == 512
            else select_tiles(
                GENERIC_DLOCAL_KV_TILES, block_d + rope_dim, query.element_size(), query.device
            )
        )
        block_m, block_n, num_warps, num_stages, *_ = local_tile
        num_local_query_tiles = triton.cdiv(window + block_n - 1, block_m)
        local_grid = (triton.cdiv(seq_len, block_n), batch * heads)
        if not rope_dim and all(map(can_use_tma, (query, local_kv, grad_output))):
            query_desc = TensorDescriptor.from_tensor(query, [1, 1, block_m, block_d])
            local_desc = TensorDescriptor.from_tensor(local_kv, [1, 1, block_n, block_d])
            output_desc = TensorDescriptor.from_tensor(output, [1, 1, block_m, block_d])
            grad_output_desc = TensorDescriptor.from_tensor(grad_output, [1, 1, block_m, block_d])
            grad_local_desc = TensorDescriptor.from_tensor(grad_local_kv, [1, 1, block_n, block_d])
            (_BWD_DLOCAL_KV_TMA if autotune else _BWD_DLOCAL_KV_TMA.pinned)[local_grid](
                query_desc,
                local_desc,
                cu_seqlens,
                cu_seqlens_k,
                output_desc,
                grad_output_desc,
                lse,
                grad_local_desc,
                num_documents=num_documents,
                LSE_STRIDES=lse.stride(),
                H=heads,
                S=seq_len,
                D=head_dim,
                WINDOW=window,
                SCALE=scale,
                HAS_CU_SEQLENS=has_cu_seqlens,
                USE_INT64_OFFSETS=use_int64_offsets,
                NUM_QUERY_TILES=num_local_query_tiles,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
            )
        else:
            _gather_attn_bwd_dlocal_kv[local_grid](
                query,
                local_kv,
                local_kv,
                query_rope,
                kernel_local_rope,
                cu_seqlens,
                cu_seqlens_k,
                output,
                grad_output,
                lse,
                grad_local_kv,
                grad_local_kv,
                grad_local_rope if rope_dim else None,
                QUERY_STRIDES=query.stride(),
                LOCAL_KV_STRIDES=local_kv.stride(),
                LOCAL_V_STRIDES=local_kv.stride(),
                GRAD_LOCAL_KV_STRIDES=grad_local_kv.stride(),
                GRAD_LOCAL_V_STRIDES=grad_local_kv.stride(),
                QUERY_ROPE_STRIDES=optional_strides(query_rope),
                LOCAL_ROPE_STRIDES=optional_strides(kernel_local_rope),
                GRAD_LOCAL_ROPE_STRIDES=optional_strides(grad_local_rope) if rope_dim else None,
                num_documents=num_documents,
                LSE_STRIDES=lse.stride(),
                H=heads,
                GROUP=1,
                V_IS_K=True,
                S=seq_len,
                D=head_dim,
                ROPE_DIM=rope_dim,
                WINDOW=window,
                SCALE=scale,
                HAS_CU_SEQLENS=has_cu_seqlens,
                USE_INT64_OFFSETS=use_int64_offsets,
                NUM_QUERY_TILES=num_local_query_tiles,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                DOT_PRECISION=local_tile.dot_precision,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    grad_sparse_rope = None
    if topk == 0 or sparse_seq_len == 0:
        grad_sparse_kv = sparse_kv.new_zeros((batch, kv_heads, sparse_seq_len, head_dim))
        if sparse_rope is not None:
            grad_sparse_rope = sparse_rope.new_zeros((batch, kv_heads, sparse_seq_len, rope_dim))
    # Key-only RoPE has only the atomic schedule; warn-only determinism keeps it.
    elif use_shared_schedule and (rope_dim or not torch.are_deterministic_algorithms_enabled()):
        grad_sparse_kv_fp32 = torch.zeros(
            batch, 1, sparse_seq_len, head_dim, device=sparse_kv.device, dtype=torch.float32
        )
        grad_sparse_rope_fp32 = (
            torch.zeros(
                batch, 1, sparse_seq_len, rope_dim, device=sparse_kv.device, dtype=torch.float32
            )
            if rope_dim
            else None
        )
        sparse_grid = lambda meta: (
            seq_len,
            batch,
            triton.cdiv(heads, meta["BLOCK_H"]) * triton.cdiv(topk, meta["BLOCK_K"]),
        )
        (BWD_DSPARSE_KV_SHARED_ATOMIC if autotune else BWD_DSPARSE_KV_SHARED_ATOMIC.pinned)[
            sparse_grid
        ](
            query,
            sparse_kv,
            sparse_kv,
            query_rope,
            sparse_rope,
            kv_indices,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            grad_sparse_kv_fp32,
            grad_sparse_kv_fp32,
            grad_sparse_rope_fp32,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SPARSE_V_STRIDES=sparse_kv.stride(),
            QUERY_ROPE_STRIDES=optional_strides(query_rope),
            SPARSE_ROPE_STRIDES=optional_strides(sparse_rope),
            KV_INDICES_BATCH_STRIDE=kv_indices.stride(0),
            KV_INDICES_STRIDES=kv_indices.stride()[1:],
            num_documents=num_documents,
            LSE_BATCH_STRIDE=lse.stride(0),
            LSE_HEAD_STRIDE=lse.stride(1),
            LSE_STRIDES=lse.stride()[2:],
            GRAD_SPARSE_KV_STRIDES=grad_sparse_kv_fp32.stride(),
            GRAD_SPARSE_V_STRIDES=grad_sparse_kv_fp32.stride(),
            GRAD_SPARSE_ROPE_STRIDES=optional_strides(grad_sparse_rope_fp32),
            H=heads,
            KV_HEADS=1,
            V_IS_K=True,
            S=seq_len,
            D=head_dim,
            ROPE_DIM=rope_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            USE_INT64_OFFSETS=use_int64_offsets,
            BLOCK_D=block_d,
            DENSE_TILE_QUERIES=0,
        )
        grad_sparse_kv = grad_sparse_kv_fp32.to(sparse_kv.dtype)
        if rope_dim:
            grad_sparse_rope = grad_sparse_rope_fp32.to(sparse_rope.dtype)
    else:
        assert selected_queries is not None and block_offsets is not None
        # The shared CSR schedule flattens heads x query rows into a large D-wide tile.
        # Use the per-head output-owned kernel for deterministic D=512 gradients instead.
        if use_shared_schedule and head_dim != 512:
            grad_sparse_kv_partials = torch.zeros(
                sparse_kv.shape,
                device=sparse_kv.device,
                dtype=sparse_kv.dtype,
            )
            sparse_kernel = BWD_DSPARSE_KV_SHARED
            sparse_grid = lambda meta: (
                sparse_seq_len,
                batch,
                triton.cdiv(heads, meta["BLOCK_H"]),
            )
        else:
            grad_sparse_kv_partials = torch.empty(
                sparse_kv.shape,
                device=sparse_kv.device,
                dtype=sparse_kv.dtype,
            )
            sparse_kernel = _BWD_DSPARSE_KV
            sparse_grid = (sparse_seq_len, batch * heads)

        (sparse_kernel if autotune else sparse_kernel.pinned)[sparse_grid](
            query,
            sparse_kv,
            selected_queries,
            block_offsets,
            output,
            grad_output,
            lse,
            grad_sparse_kv_partials,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SELECTED_QUERIES_STRIDES=selected_queries.stride(),
            BLOCK_OFFSETS_STRIDES=block_offsets.stride(),
            LSE_STRIDES=lse.stride(),
            GRAD_SPARSE_KV_STRIDES=grad_sparse_kv_partials.stride(),
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            SCALE=scale,
            USE_INT64_OFFSETS=use_int64_offsets or requires_int64_offsets(grad_sparse_kv_partials),
            BLOCK_D=block_d,
        )
        grad_sparse_kv = (
            grad_sparse_kv_partials.sum(dim=1, keepdim=True)
            if share_kv
            else grad_sparse_kv_partials
        )

    if share_kv and window:
        grad_local_kv = grad_local_kv.sum(dim=1, keepdim=True)
        if rope_dim:
            grad_local_rope = grad_local_rope.sum(dim=1, keepdim=True)

    return (
        grad_query,
        grad_sparse_kv,
        grad_local_kv,
        grad_query_rope,
        grad_sparse_rope,
        grad_local_rope,
        grad_sink_fp32.to(attention_sink.dtype),
        None,
        None,
    )


def _prefer_atomic_dsparse_kv(
    group: int, head_dim: int, v_is_k: bool, topk: int, keys_per_document: float, tile: int
) -> bool:
    """Whether query-major atomics beat the key-major sparse dK/dV pass.

    Key-major gathers a query's ``group`` rows of Q and dO once per (query, key tile) pair,
    so it wins when each pair covers several selected keys. Atomics instead gather and add one
    K (and V) row per selection, which is cheap for small heads. The density is estimated for
    uniformly random selections, a lower bound for the clustered selections of block-sparse
    indexers. The threshold, linear in head_dim, comes from B200 sweeps over group (2-32),
    head_dim (64-256), topk (64-2048), document length (2k-32k), and shared or separate
    values: the measured crossover is about 3 rows per selection at D=64 and 6 at D=128, and
    atomics were never more than 3% faster at D=256.
    """
    selected = min(1.0, topk / max(1.0, keys_per_document))
    keys_per_pair = tile * selected / (1 - (1 - selected) ** tile)
    rows_per_selection = group / (keys_per_pair * (1 if v_is_k else 2))
    return rows_per_selection >= 3 * head_dim / 64


def _launch_grouped_backward(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor | None,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    sliding_window_size: int,
    scale: float,
    *,
    kv_heads: int,
    sparse_v: torch.Tensor | None,
    local_v: torch.Tensor | None,
    autotune: bool,
) -> tuple[torch.Tensor, ...]:
    """Backward for separate values or grouped KV heads, mirroring the grouped forward.

    dQ reuses each gathered KV row across a query-head group. Sparse dK/dV is key-major
    (see ``key_major_backward``) and deterministic, so it needs no query map from the forward;
    the default mode switches to query-major atomics for sparse selections over small heads.
    Dense tiles (see ``plan_dense_tiles``) take dQ and their dK/dV share from
    ``dense_backward``, and the gather kernels skip their queries. Local gradients write one
    partial per query head, which is summed over each KV head's group here instead of with
    atomics. Returns ``(dq, dsparse, dlocal, dsink, dsparse_v, dlocal_v)``; a gradient is
    None exactly when its input is. Local tensors are not read without a window.
    """
    batch, heads, seq_len, head_dim = query.shape
    group = heads // kv_heads
    sparse_seq_len = sparse_kv.shape[2]
    topk = kv_indices.shape[-1]
    block_d = max(16, triton.next_power_of_2(head_dim))
    v_is_k = sparse_v is None
    window = sliding_window_size
    # Kernels never read local tensors without a window.
    kernel_local_kv = local_kv if window else None
    sparse_values = sparse_kv if v_is_k else sparse_v
    local_values = kernel_local_kv if v_is_k or not window else local_v
    grad_output = grad_output.contiguous()
    has_cu_seqlens = cu_seqlens is not None
    num_documents = cu_seqlens.numel() - 1 if has_cu_seqlens else 0
    use_int64_offsets = requires_int64_offsets(
        query,
        sparse_kv,
        kernel_local_kv,
        sparse_v,
        local_values if window else None,
        kv_indices,
        output,
        grad_output,
        lse,
        cu_seqlens,
        cu_seqlens_k,
    )

    def rounded(gradient: torch.Tensor) -> torch.Tensor:
        """Round an FP32 pool gradient to the pool's dtype."""
        return gradient.to(sparse_kv.dtype)

    def sum_groups(per_head: torch.Tensor) -> torch.Tensor:
        """Reduce [B, H, N, D] query-head partials to [B, kv_heads, N, D]."""
        return per_head.view(batch, kv_heads, group, *per_head.shape[2:]).sum(dim=2)

    # tl.dot needs at least 16 rows; partial head tiles are masked.
    block_h = 16 if head_dim > 128 else max(16, min(32, triton.next_power_of_2(group)))
    grad_query = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    grad_sink_partials = torch.empty(
        (batch, heads, seq_len), dtype=torch.float32, device=query.device
    )
    sparse_backward = topk > 0 and sparse_seq_len > 0
    tile = 32
    use_atomic = (
        sparse_backward
        and not torch.are_deterministic_algorithms_enabled()
        and _prefer_atomic_dsparse_kv(
            group, head_dim, v_is_k, topk, sparse_seq_len / max(1, num_documents), tile
        )
    )
    dense_shape = dense_tile_shape(heads, kv_heads, topk, sliding_window_size)
    tile_queries, mask_words = dense_shape or (0, 0)
    # Recomputed rather than saved from the forward: about 2% of a QSA step against holding
    # the rows, bitmaps, and spans (hundreds of MB at 32k tokens) between passes.
    selected_rows, selected_counts, masks, spans = resolve_selected_rows(
        kv_indices, sparse_seq_len, cu_seqlens, cu_seqlens_k, mask_words=mask_words
    )
    # rowsum(dO * O) for the key-major and dense sparse passes, laid out like the LSE.
    key_major = sparse_backward and not use_atomic
    store_delta = key_major or dense_shape is not None
    delta = torch.empty(lse.shape, dtype=lse.dtype, device=lse.device) if store_delta else None
    tile_spans = None
    num_dense_tiles = 0
    if dense_shape is not None:
        tile_spans, dense_bounds = plan_dense_tiles(
            selected_counts, spans, cu_seqlens, tile_queries, document_bounds=True
        )
        num_dense_tiles = tile_spans.shape[1]
    if dense_shape is not None:
        dense_rows = tile_queries * group
        (BWD_DQ_DENSE if autotune else BWD_DQ_DENSE.pinned)[
            lambda meta: (
                num_dense_tiles * triton.cdiv(dense_rows, meta["BLOCK_M"]),
                batch,
                kv_heads,
            )
        ](
            query,
            sparse_kv,
            sparse_values,
            masks,
            tile_spans,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            attention_sink,
            grad_query,
            grad_sink_partials,
            delta,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SPARSE_V_STRIDES=sparse_values.stride(),
            LSE_BATCH_STRIDE=lse.stride(0),
            LSE_HEAD_STRIDE=lse.stride(1),
            LSE_STRIDES=lse.stride()[2:],
            GRAD_SINK_PARTIALS_BATCH_STRIDE=grad_sink_partials.stride(0),
            GRAD_SINK_PARTIALS_HEAD_STRIDE=grad_sink_partials.stride(1),
            GRAD_SINK_PARTIALS_STRIDES=grad_sink_partials.stride()[2:],
            num_documents=num_documents,
            num_tiles=num_dense_tiles,
            H=heads,
            KV_HEADS=kv_heads,
            V_IS_K=v_is_k,
            STORE_DELTA=True,
            S=seq_len,
            D=head_dim,
            MASK_WORDS=mask_words,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            USE_INT64_OFFSETS=use_int64_offsets,
            TILE_QUERIES=tile_queries,
            BLOCK_D=block_d,
        )
    (BWD_DQ_GROUPED if autotune else BWD_DQ_GROUPED.pinned)[
        (seq_len, batch, kv_heads * triton.cdiv(group, block_h))
    ](
        query,
        sparse_kv,
        kernel_local_kv,
        sparse_values,
        local_values,
        None,
        None,
        None,
        selected_rows,
        selected_counts,
        cu_seqlens,
        cu_seqlens_k,
        output,
        grad_output,
        lse,
        attention_sink,
        grad_query,
        None,
        grad_sink_partials,
        delta,
        QUERY_STRIDES=query.stride(),
        SPARSE_KV_STRIDES=sparse_kv.stride(),
        LOCAL_KV_STRIDES=optional_strides(kernel_local_kv),
        SPARSE_V_STRIDES=sparse_values.stride(),
        LOCAL_V_STRIDES=optional_strides(local_values),
        QUERY_ROPE_STRIDES=None,
        SPARSE_ROPE_STRIDES=None,
        LOCAL_ROPE_STRIDES=None,
        KV_INDICES_STRIDES=selected_rows.stride(),
        num_documents=num_documents,
        LSE_STRIDES=lse.stride(),
        GRAD_SINK_PARTIALS_STRIDES=grad_sink_partials.stride(),
        H=heads,
        KV_HEADS=kv_heads,
        V_IS_K=v_is_k,
        # Only the key-major pass reads gather queries' rowsum; dense rows come from BWD_DQ_DENSE.
        STORE_DELTA=key_major,
        S=seq_len,
        D=head_dim,
        ROPE_DIM=0,
        SPARSE_SEQ_LEN=sparse_seq_len,
        TOPK=topk,
        WINDOW=sliding_window_size,
        SCALE=scale,
        HAS_CU_SEQLENS=has_cu_seqlens,
        USE_INT64_OFFSETS=use_int64_offsets,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        RESOLVED_INDICES=True,
        tile_spans_ptr=tile_spans,
        DENSE_TILE_QUERIES=tile_queries,
    )
    grad_sink = grad_sink_partials.sum(dim=(0, 2)).to(attention_sink.dtype)

    local_shape = (batch, kv_heads, seq_len, head_dim)
    if not window:
        # A pool passed without a window gets a zero gradient; an absent one gets None.
        grad_local_kv, grad_local_v = (
            None
            if tensor is None
            else torch.zeros(local_shape, dtype=tensor.dtype, device=tensor.device)
            for tensor in (local_kv, local_v)
        )
    else:
        local_tile = select_tiles(
            GENERIC_DLOCAL_KV_TILES, block_d, query.element_size(), query.device
        )
        block_m, block_n, num_warps, num_stages, *_ = local_tile
        partial_local_kv = torch.empty(query.shape, dtype=local_kv.dtype, device=local_kv.device)
        partial_local_v = (
            partial_local_kv
            if v_is_k
            else torch.empty(query.shape, dtype=local_kv.dtype, device=local_kv.device)
        )
        _gather_attn_bwd_dlocal_kv[(triton.cdiv(seq_len, block_n), batch * heads)](
            query,
            local_kv,
            local_values,
            None,
            None,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            partial_local_kv,
            partial_local_v,
            None,
            QUERY_STRIDES=query.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            LOCAL_V_STRIDES=local_values.stride(),
            GRAD_LOCAL_KV_STRIDES=partial_local_kv.stride(),
            GRAD_LOCAL_V_STRIDES=partial_local_v.stride(),
            QUERY_ROPE_STRIDES=None,
            LOCAL_ROPE_STRIDES=None,
            GRAD_LOCAL_ROPE_STRIDES=None,
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            H=heads,
            GROUP=group,
            V_IS_K=v_is_k,
            S=seq_len,
            D=head_dim,
            ROPE_DIM=0,
            WINDOW=sliding_window_size,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            USE_INT64_OFFSETS=use_int64_offsets
            or requires_int64_offsets(partial_local_kv, partial_local_v),
            NUM_QUERY_TILES=triton.cdiv(sliding_window_size + block_n - 1, block_m),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            DOT_PRECISION=local_tile.dot_precision,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        grad_local_kv = sum_groups(partial_local_kv)
        grad_local_v = None if v_is_k else sum_groups(partial_local_v)
        # Free the per-head partials before the sparse dK/dV map's sort, the backward's peak.
        del partial_local_kv, partial_local_v

    sparse_shape = (batch, kv_heads, sparse_seq_len, head_dim)
    grad_dense_kv = grad_dense_v = None
    if dense_shape is not None:
        # The dense tiles' FP32 dK/dV; the gather schedules below add theirs on top.
        dense_block_n = 64
        cu_key_tiles, key_tile_grid = document_key_tiles(
            cu_seqlens_k, sparse_seq_len, dense_block_n
        )
        grad_dense_kv = torch.zeros(sparse_shape, dtype=torch.float32, device=sparse_kv.device)
        grad_dense_v = (
            grad_dense_kv
            if v_is_k
            else torch.zeros(sparse_shape, dtype=torch.float32, device=sparse_kv.device)
        )
        (BWD_DSPARSE_KV_DENSE if autotune else BWD_DSPARSE_KV_DENSE.pinned)[
            (key_tile_grid, kv_heads, batch)
        ](
            query,
            sparse_kv,
            sparse_values,
            grad_output,
            lse,
            delta,
            masks,
            tile_spans,
            dense_bounds,
            cu_seqlens_k,
            cu_key_tiles,
            grad_dense_kv,
            grad_dense_v,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SPARSE_V_STRIDES=sparse_values.stride(),
            LSE_BATCH_STRIDE=lse.stride(0),
            LSE_HEAD_STRIDE=lse.stride(1),
            LSE_STRIDES=lse.stride()[2:],
            GRAD_STRIDES=grad_dense_kv.stride(),
            num_documents=num_documents,
            num_tiles=num_dense_tiles,
            H=heads,
            KV_HEADS=kv_heads,
            V_IS_K=v_is_k,
            S=seq_len,
            SPARSE_SEQ_LEN=sparse_seq_len,
            D=head_dim,
            MASK_WORDS=mask_words,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            USE_INT64_OFFSETS=use_int64_offsets or requires_int64_offsets(grad_dense_kv),
            TILE_QUERIES=tile_queries,
            BLOCK_M=DENSE_TILE_ROWS,
            BLOCK_N=dense_block_n,
            BLOCK_D=block_d,
        )
    if not sparse_backward:
        grad_sparse_kv = torch.zeros(sparse_shape, dtype=sparse_kv.dtype, device=sparse_kv.device)
        grad_sparse_v = (
            None
            if v_is_k
            else torch.zeros(sparse_shape, dtype=sparse_kv.dtype, device=sparse_kv.device)
        )
    elif use_atomic:
        if grad_dense_kv is not None:
            grad_sparse_kv_fp32, grad_sparse_v_fp32 = grad_dense_kv, grad_dense_v
        else:
            grad_sparse_kv_fp32 = torch.zeros(
                sparse_shape, dtype=torch.float32, device=sparse_kv.device
            )
            grad_sparse_v_fp32 = (
                grad_sparse_kv_fp32
                if v_is_k
                else torch.zeros(sparse_shape, dtype=torch.float32, device=sparse_kv.device)
            )
        (BWD_DSPARSE_KV_SHARED_ATOMIC if autotune else BWD_DSPARSE_KV_SHARED_ATOMIC.pinned)[
            lambda meta: (
                seq_len,
                batch,
                kv_heads
                * triton.cdiv(group, meta["BLOCK_H"])
                * triton.cdiv(topk, meta["BLOCK_K"]),
            )
        ](
            query,
            sparse_kv,
            sparse_values,
            None,
            None,
            kv_indices,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            grad_sparse_kv_fp32,
            grad_sparse_v_fp32,
            None,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SPARSE_V_STRIDES=sparse_values.stride(),
            QUERY_ROPE_STRIDES=None,
            SPARSE_ROPE_STRIDES=None,
            KV_INDICES_BATCH_STRIDE=kv_indices.stride(0),
            KV_INDICES_STRIDES=kv_indices.stride()[1:],
            num_documents=num_documents,
            LSE_BATCH_STRIDE=lse.stride(0),
            LSE_HEAD_STRIDE=lse.stride(1),
            LSE_STRIDES=lse.stride()[2:],
            GRAD_SPARSE_KV_STRIDES=grad_sparse_kv_fp32.stride(),
            GRAD_SPARSE_V_STRIDES=grad_sparse_v_fp32.stride(),
            GRAD_SPARSE_ROPE_STRIDES=None,
            H=heads,
            KV_HEADS=kv_heads,
            V_IS_K=v_is_k,
            S=seq_len,
            D=head_dim,
            ROPE_DIM=0,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            # Expanded pools address far less storage than their contiguous gradients.
            USE_INT64_OFFSETS=use_int64_offsets
            or requires_int64_offsets(grad_sparse_kv_fp32, grad_sparse_v_fp32),
            BLOCK_D=block_d,
            tile_spans_ptr=tile_spans,
            DENSE_TILE_QUERIES=tile_queries,
        )
        grad_sparse_kv = rounded(grad_sparse_kv_fp32)
        grad_sparse_v = grad_sparse_kv if v_is_k else rounded(grad_sparse_v_fp32)
    else:
        # Each program owns its key tile's gradient rows, so this is deterministic.
        gather_counts = selected_counts
        if tile_spans is not None:
            # The dense kernels cover their tiles' queries.
            dense = (tile_spans[..., 1] >= 0).repeat_interleave(tile_queries, dim=1)
            gather_counts = torch.where(dense[:, :seq_len], 0, selected_counts)
        pair_queries, pair_masks, tile_offsets = build_tile_query_map(
            selected_rows, gather_counts, sparse_seq_len, tile
        )
        grad_sparse_kv = torch.empty(sparse_shape, dtype=sparse_kv.dtype, device=sparse_kv.device)
        grad_sparse_v = (
            grad_sparse_kv
            if v_is_k
            else torch.empty(sparse_shape, dtype=sparse_kv.dtype, device=sparse_kv.device)
        )
        # TMA gathers need 16-byte-aligned bases, which a contiguous view at an odd
        # storage offset lacks; a copy gets a fresh, aligned allocation.
        query_rows, grad_output_rows = (
            (tensor if can_use_tma(tensor) else tensor.clone()).view(-1, head_dim)
            for tensor in (query, grad_output)
        )
        # Two 4-warp CTAs per SM overlap one CTA's gathers with the other's MMAs; software
        # pipelining instead needs a third gather buffer and drops to one CTA per SM. 8 warps
        # were never more than noise faster on B200 and up to 46% slower on 32k-token
        # documents, so this one fixed config launches without tuning.
        _gather_attn_bwd_dsparse_kv_grouped[(triton.cdiv(sparse_seq_len, tile), kv_heads, batch)](
            TensorDescriptor.from_tensor(query_rows, [1, block_d]),
            sparse_kv,
            sparse_values,
            TensorDescriptor.from_tensor(grad_output_rows, [1, block_d]),
            lse,
            delta,
            pair_queries,
            pair_masks,
            tile_offsets,
            grad_sparse_kv,
            grad_sparse_v,
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SPARSE_V_STRIDES=sparse_values.stride(),
            LSE_STRIDES=lse.stride(),
            PAIR_STRIDES=pair_queries.stride(),
            TILE_OFFSETS_STRIDES=tile_offsets.stride(),
            GRAD_SPARSE_KV_STRIDES=grad_sparse_kv.stride(),
            GRAD_SPARSE_V_STRIDES=grad_sparse_v.stride(),
            H=heads,
            KV_HEADS=kv_heads,
            V_IS_K=v_is_k,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            SCALE=scale,
            # Rows enumerate (pair, group head) and pair offsets span the batch, both bounded
            # by the map's capacity (seq_len * topk padded to 16).
            USE_INT64_OFFSETS=use_int64_offsets
            or requires_int64_offsets(grad_sparse_kv, grad_sparse_v)
            or triton.cdiv(seq_len * topk, 16) * 16 * max(group, batch) >= 2**31,
            TILE=tile,
            BLOCK_M=64,
            BLOCK_D=block_d,
            grad_dense_kv_ptr=grad_dense_kv,
            grad_dense_v_ptr=grad_dense_v,
            GRAD_DENSE_STRIDES=None if grad_dense_kv is None else grad_dense_kv.stride(),
            ADD_DENSE=grad_dense_kv is not None,
            num_warps=4,
            num_stages=1,
        )

    return (
        grad_query,
        grad_sparse_kv,
        grad_local_kv,
        grad_sink,
        None if v_is_k else grad_sparse_v,
        None if v_is_k else grad_local_v,
    )
