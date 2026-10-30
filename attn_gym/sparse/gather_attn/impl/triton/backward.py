"""Backward kernels and launcher for Triton gather attention."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import can_use_tma, ptr_offset, requires_int64_offsets

from ..routing import uses_grouped_schedule
from .dense_backward import BWD_DQ_DENSE, BWD_DSPARSE_KV_DENSE, document_key_tiles
from .key_major_backward import (
    BWD_DSPARSE_KV_GROUPED,
    build_tile_query_map,
    count_gather_selections,
)
from .primitives import (
    DENSE_TILE_ROWS,
    TileConfig,
    can_use_shared_kv_schedule,
    causal_window_mask,
    dense_tile_shape,
    full_buffer,
    load_bhsd,
    load_document_bounds,
    offset_strides,
    plan_dense_tiles,
    prune_wide_backward_configs,
    resolve_selected_rows,
    select_tiles,
    store_bhsd,
    tuning_key,
    uninitialized_empty,
)
from .shared_backward import (
    BWD_DQ_GROUPED,
    BWD_DQ_SHARED,
    BWD_DSPARSE_KV_SHARED,
    BWD_DSPARSE_KV_SHARED_ATOMIC,
)


@triton.jit
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
    WIDE: tl.constexpr,
    NUM_LOCAL_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate query and sink gradients over both attention branches."""
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    LOCAL_KV_STRIDES = offset_strides(LOCAL_KV_STRIDES, WIDE)
    KV_INDICES_STRIDES = offset_strides(KV_INDICES_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)

    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    if WIDE:
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
            cu_seqlens_ptr, cu_seqlens_k_ptr, offsets_m, num_documents, S, WIDE
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
        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, key_mask, WINDOW)
        if HAS_CU_SEQLENS:
            valid &= offsets_n[None, :] >= query_start[:, None]

        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
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


@triton.jit
def _gather_attn_bwd_dlocal_kv(
    query_ptr,
    local_kv_ptr,
    local_v_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_local_kv_ptr,
    grad_local_v_ptr,
    QUERY_STRIDES,
    LOCAL_KV_STRIDES,
    LOCAL_V_STRIDES,
    GRAD_LOCAL_KV_STRIDES,
    GRAD_LOCAL_V_STRIDES,
    num_documents,
    LSE_STRIDES,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    V_IS_K: tl.constexpr,
    S,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    WIDE: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward for dKV (local sliding window).

    Each program writes its query head's partial gradient; query head h reads KV head
    h // GROUP, and the launcher sums each group's partials. With V_IS_K the key and value
    gradients add into one dKV; otherwise dK and dV are written separately.
    """
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    LOCAL_KV_STRIDES = offset_strides(LOCAL_KV_STRIDES, WIDE)
    GRAD_LOCAL_KV_STRIDES = offset_strides(GRAD_LOCAL_KV_STRIDES, WIDE)
    if not V_IS_K:
        LOCAL_V_STRIDES = offset_strides(LOCAL_V_STRIDES, WIDE)
        GRAD_LOCAL_V_STRIDES = offset_strides(GRAD_LOCAL_V_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)

    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    if WIDE:
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

    if HAS_CU_SEQLENS:
        _, key_end, _, _ = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, offsets_n, num_documents, S, WIDE
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

        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(local_v), input_precision="tf32x3")
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_values += tl.dot(
            tl.trans(probabilities.to(grad_output.dtype)),
            grad_output,
            input_precision="tf32x3",
        )
        if V_IS_K:
            grad_values += (
                tl.dot(
                    tl.trans(grad_scores.to(query.dtype)),
                    query,
                    input_precision="tf32x3",
                )
                * SCALE
            )
        else:
            grad_keys += (
                tl.dot(
                    tl.trans(grad_scores.to(query.dtype)),
                    query,
                    input_precision="tf32x3",
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


@triton.jit
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
    WIDE: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """TMA backward for local-KV gradients."""
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)

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
            cu_seqlens_ptr, cu_seqlens_k_ptr, offsets_n, num_documents, S, WIDE
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
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
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


@triton.jit
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
    WIDE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute sparse-KV gradients from an inverted index of selecting queries."""
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    SELECTED_QUERIES_STRIDES = offset_strides(SELECTED_QUERIES_STRIDES, WIDE)
    BLOCK_OFFSETS_STRIDES = offset_strides(BLOCK_OFFSETS_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)
    GRAD_SPARSE_KV_STRIDES = offset_strides(GRAD_SPARSE_KV_STRIDES, WIDE)

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


@triton.jit
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
    WIDE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Form packed global sort keys without changing attention's document-local indices."""
    strides = offset_strides((stride_b, stride_s, stride_k), WIDE)
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if WIDE:
        row = row.to(tl.int64)
        tile = tile.to(tl.int64)
    batch = row // S
    query_position = row % S
    slots = tile * BLOCK_K + tl.arange(0, BLOCK_K)
    _, _, candidate_start, candidate_end = load_document_bounds(
        cu_seqlens_ptr, cu_seqlens_k_ptr, query_position, num_documents, S, WIDE
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
        _index_query_map_keys[(batch * seq_len, triton.cdiv(topk, 256))](
            kv_indices,
            sort_keys,
            cu_seqlens,
            cu_seqlens_k,
            cu_seqlens.numel() - 1,
            *kv_indices.stride(),
            S=seq_len,
            TOPK=topk,
            WIDE=requires_int64_offsets(kv_indices, sort_keys, cu_seqlens, cu_seqlens_k),
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
    TileConfig(16, 16, 4, 1, shared_rows=68),
)


def _launch_backward(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    selected_queries: torch.Tensor,
    block_offsets: torch.Tensor,
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
    sparse_v: torch.Tensor | None,
    local_v: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Launch backward kernels for gather attention.

    Returns (dQ, dsparse_kv, dlocal_kv, dsink, dsparse_v, dlocal_v); the value gradients are
    empty when the keys are also the values. ``kv_heads`` is the KV head count before the op
    broadcasts a single shared head.
    """
    batch, heads, seq_len, head_dim = query.shape
    if uses_grouped_schedule(heads, kv_heads, sparse_v is not None):
        return _launch_grouped_backward(
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
        )
    sparse_seq_len = sparse_kv.shape[2]
    topk = kv_indices.shape[-1]
    block_d = max(16, triton.next_power_of_2(head_dim))
    # Full-width D=512 operands otherwise exceed per-block shared-memory limits.
    if head_dim == 512:
        dq_tiles = dlocal_tiles = TileConfig(16, 16, 4, 1)
    else:
        dq_tiles = select_tiles(GENERIC_DQ_TILES, block_d, query.element_size(), query.device)
        dlocal_tiles = select_tiles(
            GENERIC_DLOCAL_KV_TILES, block_d, query.element_size(), query.device
        )
    grad_output = grad_output.contiguous()

    grad_query = torch.empty_like(query)
    # Expanded shared KV has a zero head stride, but its per-head gradients
    # need distinct storage before ExpandBackward sums them.
    grad_local_kv = torch.empty(local_kv.shape, device=local_kv.device, dtype=local_kv.dtype)

    has_cu_seqlens = cu_seqlens is not None
    num_documents = cu_seqlens.numel() - 1 if has_cu_seqlens else 0
    wide = requires_int64_offsets(
        query,
        sparse_kv,
        local_kv,
        kv_indices,
        selected_queries,
        block_offsets,
        output,
        grad_output,
        lse,
        grad_local_kv,
        cu_seqlens,
        cu_seqlens_k,
    )
    use_tma = can_use_tma(query) and can_use_tma(local_kv) and can_use_tma(grad_output)

    num_local_query_tiles = (
        triton.cdiv(sliding_window_size + dlocal_tiles.block_n - 1, dlocal_tiles.block_m)
        if sliding_window_size
        else 0
    )

    # Zero head strides share values; share_kv also guarantees autograd will sum dKV heads.
    use_shared_schedule = share_kv and can_use_shared_kv_schedule(
        query, sparse_kv, local_kv, sliding_window_size
    )
    if use_shared_schedule:
        block_h = 16 if head_dim == 512 else min(32, triton.next_power_of_2(heads))
        block_k = 16 if head_dim == 512 else max(16, min(64, triton.next_power_of_2(topk)))
        grad_sink_partials = torch.empty(
            batch, heads, seq_len, device=query.device, dtype=torch.float32
        )
        BWD_DQ_SHARED[(seq_len, batch, triton.cdiv(heads, block_h))](
            query,
            sparse_kv,
            local_kv,
            sparse_kv,
            local_kv,
            kv_indices,
            None,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            attention_sink,
            grad_query,
            grad_sink_partials,
            None,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            SPARSE_V_STRIDES=sparse_kv.stride(),
            LOCAL_V_STRIDES=local_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            GRAD_SINK_PARTIALS_STRIDES=grad_sink_partials.stride(),
            B=batch,
            H=heads,
            KV_HEADS=1,
            V_IS_K=True,
            STORE_DELTA=False,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            WINDOW=sliding_window_size,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            WIDE=wide,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            SPARSE_STAGES=2,
            RESOLVED_INDICES=False,
        )
        grad_sink_fp32 = grad_sink_partials.sum(dim=(0, 2))
    else:
        block_m, block_n, num_warps, num_stages, *_ = dq_tiles
        num_local_key_tiles = (
            triton.cdiv(sliding_window_size + block_m - 1, block_n) if sliding_window_size else 0
        )
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
            local_kv,
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
            LOCAL_KV_STRIDES=local_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            WINDOW=sliding_window_size,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            WIDE=wide,
            NUM_LOCAL_TILES=num_local_key_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        grad_sink_fp32 = grad_sink_partials.sum(dim=(0, 2))

    block_m, block_n, num_warps, num_stages, *_ = dlocal_tiles
    local_grid = (triton.cdiv(seq_len, block_n), batch * heads)
    if use_tma:
        query_desc = TensorDescriptor.from_tensor(query, [1, 1, block_m, block_d])
        local_desc = TensorDescriptor.from_tensor(local_kv, [1, 1, block_n, block_d])
        output_desc = TensorDescriptor.from_tensor(output, [1, 1, block_m, block_d])
        grad_output_desc = TensorDescriptor.from_tensor(grad_output, [1, 1, block_m, block_d])
        grad_local_desc = TensorDescriptor.from_tensor(grad_local_kv, [1, 1, block_n, block_d])
        _BWD_DLOCAL_KV_TMA[local_grid](
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
            WINDOW=sliding_window_size,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            WIDE=wide,
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
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            grad_local_kv,
            grad_local_kv,
            QUERY_STRIDES=query.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            LOCAL_V_STRIDES=local_kv.stride(),
            GRAD_LOCAL_KV_STRIDES=grad_local_kv.stride(),
            GRAD_LOCAL_V_STRIDES=grad_local_kv.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            H=heads,
            GROUP=1,
            V_IS_K=True,
            S=seq_len,
            D=head_dim,
            WINDOW=sliding_window_size,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            WIDE=wide,
            NUM_QUERY_TILES=num_local_query_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if topk == 0 or sparse_seq_len == 0:
        grad_sparse_kv = torch.zeros(
            batch,
            kv_heads,
            sparse_seq_len,
            head_dim,
            device=sparse_kv.device,
            dtype=sparse_kv.dtype,
        )
    elif use_shared_schedule and not torch.are_deterministic_algorithms_enabled():
        grad_sparse_kv_fp32 = torch.zeros(
            batch,
            1,
            sparse_seq_len,
            head_dim,
            device=sparse_kv.device,
            dtype=torch.float32,
        )
        sparse_grid = lambda meta: (
            seq_len,
            batch,
            triton.cdiv(heads, meta["BLOCK_H"]) * triton.cdiv(topk, meta["BLOCK_K"]),
        )
        BWD_DSPARSE_KV_SHARED_ATOMIC[sparse_grid](
            query,
            sparse_kv,
            sparse_kv,
            kv_indices,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            grad_sparse_kv_fp32,
            grad_sparse_kv_fp32,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SPARSE_V_STRIDES=sparse_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            GRAD_SPARSE_KV_STRIDES=grad_sparse_kv_fp32.stride(),
            GRAD_SPARSE_V_STRIDES=grad_sparse_kv_fp32.stride(),
            B=batch,
            H=heads,
            KV_HEADS=1,
            V_IS_K=True,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            WIDE=wide,
            BLOCK_D=block_d,
            DENSE_TILE_QUERIES=0,
        )
        grad_sparse_kv = grad_sparse_kv_fp32.to(sparse_kv.dtype)
    else:
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

        sparse_kernel[sparse_grid](
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
            WIDE=wide or requires_int64_offsets(grad_sparse_kv_partials),
            BLOCK_D=block_d,
        )
        grad_sparse_kv = (
            grad_sparse_kv_partials.sum(dim=1, keepdim=True)
            if share_kv
            else grad_sparse_kv_partials
        )

    if share_kv:
        grad_local_kv = grad_local_kv.sum(dim=1, keepdim=True)

    # Distinct empty outputs: an operator's outputs must not alias each other.
    return (
        grad_query,
        grad_sparse_kv,
        grad_local_kv,
        grad_sink_fp32.to(attention_sink.dtype),
        query.new_empty(0),
        query.new_empty(0),
    )


def _prefer_atomic_dsparse_kv(
    group: int, head_dim: int, v_is_k: bool, topk: int, keys_per_document: float, tile: int
) -> bool:
    """Whether query-major atomics beat the key-major sparse dK/dV pass.

    Key-major gathers a query's ``group`` rows of Q and dO once per (query, key tile) pair,
    so it wins when each pair covers several selected keys. Atomics instead gather and add one
    K (and V) row per selection, which is cheap for small heads. The density is estimated for
    uniformly random selections, a lower bound for the clustered selections of block-sparse
    indexers, and the per-head-dim thresholds come from B200 sweeps over group, head_dim,
    topk, and shared or separate values.
    """
    selected = min(1.0, topk / max(1.0, keys_per_document))
    keys_per_pair = tile * selected / (1 - (1 - selected) ** tile)
    rows_per_selection = group / (keys_per_pair * (1 if v_is_k else 2))
    return rows_per_selection >= 1.8 * (head_dim / 128) ** 2


def _launch_grouped_backward(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
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
) -> tuple[torch.Tensor, ...]:
    """Backward for separate values or grouped KV heads, mirroring the grouped forward.

    dQ reuses each gathered KV row across a query-head group. Sparse dK/dV is key-major
    (see ``key_major_backward``) and deterministic, so it needs no query map from the forward;
    the default mode switches to query-major atomics for sparse selections over small heads.
    Dense tiles (see ``plan_dense_tiles``) take dQ and their dK/dV share from
    ``dense_backward``, and the gather kernels skip their queries. Local gradients write one
    partial per query head, which is summed over each KV head's group here instead of with
    atomics.
    """
    batch, heads, seq_len, head_dim = query.shape
    group = heads // kv_heads
    sparse_seq_len = sparse_kv.shape[2]
    topk = kv_indices.shape[-1]
    block_d = max(16, triton.next_power_of_2(head_dim))
    v_is_k = sparse_v is None
    sparse_values = sparse_kv if v_is_k else sparse_v
    local_values = local_kv if v_is_k else local_v
    grad_output = grad_output.contiguous()
    has_cu_seqlens = cu_seqlens is not None
    num_documents = cu_seqlens.numel() - 1 if has_cu_seqlens else 0
    wide = requires_int64_offsets(
        query,
        sparse_kv,
        local_kv,
        sparse_v,
        local_v,
        kv_indices,
        output,
        grad_output,
        lse,
        cu_seqlens,
        cu_seqlens_k,
    )

    def rounded(gradient: torch.Tensor) -> torch.Tensor:
        """Round an FP32 pool gradient to the pool's dtype, into an unfilled buffer."""
        return uninitialized_empty(
            gradient.shape, dtype=sparse_kv.dtype, device=gradient.device
        ).copy_(gradient)

    def sum_groups(per_head: torch.Tensor) -> torch.Tensor:
        """Reduce [B, H, N, D] query-head partials to [B, kv_heads, N, D]."""
        return per_head.view(batch, kv_heads, group, *per_head.shape[2:]).sum(dim=2)

    # tl.dot needs at least 16 rows; partial head tiles are masked.
    block_h = 16 if head_dim > 128 else max(16, min(32, triton.next_power_of_2(group)))
    grad_query = uninitialized_empty(query.shape, dtype=query.dtype, device=query.device)
    grad_sink_partials = uninitialized_empty(
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
    delta = (
        uninitialized_empty(lse.shape, dtype=lse.dtype, device=lse.device) if store_delta else None
    )
    tile_spans = None
    num_dense_tiles = 0
    if dense_shape is not None:
        tile_spans, dense_bounds = plan_dense_tiles(
            selected_counts, spans, cu_seqlens, tile_queries, document_bounds=True
        )
        num_dense_tiles = tile_spans.shape[1]
    # Start the gather count's host copy now; the map reads it after the dQ and dense dK/dV
    # kernels below are queued.
    gather_selections = (
        count_gather_selections(selected_counts, tile_spans, tile_queries) if key_major else None
    )
    if dense_shape is not None:
        dense_rows = tile_queries * group
        BWD_DQ_DENSE[
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
            LSE_STRIDES=lse.stride(),
            GRAD_SINK_PARTIALS_STRIDES=grad_sink_partials.stride(),
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
            WIDE=wide,
            TILE_QUERIES=tile_queries,
            BLOCK_D=block_d,
        )
    BWD_DQ_GROUPED[(seq_len, batch, kv_heads * triton.cdiv(group, block_h))](
        query,
        sparse_kv,
        local_kv,
        sparse_values,
        local_values,
        selected_rows,
        selected_counts,
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
        LOCAL_KV_STRIDES=local_kv.stride(),
        SPARSE_V_STRIDES=sparse_values.stride(),
        LOCAL_V_STRIDES=local_values.stride(),
        KV_INDICES_STRIDES=selected_rows.stride(),
        num_documents=num_documents,
        LSE_STRIDES=lse.stride(),
        GRAD_SINK_PARTIALS_STRIDES=grad_sink_partials.stride(),
        B=batch,
        H=heads,
        KV_HEADS=kv_heads,
        V_IS_K=v_is_k,
        STORE_DELTA=store_delta,
        S=seq_len,
        D=head_dim,
        SPARSE_SEQ_LEN=sparse_seq_len,
        TOPK=topk,
        WINDOW=sliding_window_size,
        SCALE=scale,
        HAS_CU_SEQLENS=has_cu_seqlens,
        WIDE=wide,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        RESOLVED_INDICES=True,
        tile_spans_ptr=tile_spans,
        DENSE_TILE_QUERIES=tile_queries,
    )
    grad_sink = grad_sink_partials.sum(dim=(0, 2)).to(attention_sink.dtype)

    local_shape = (batch, kv_heads, seq_len, head_dim)
    if sliding_window_size == 0:
        grad_local_kv = full_buffer(local_shape, 0, dtype=local_kv.dtype, device=local_kv.device)
        grad_local_v = (
            None
            if v_is_k
            else full_buffer(local_shape, 0, dtype=local_kv.dtype, device=local_kv.device)
        )
    else:
        block_m, block_n, num_warps, num_stages, *_ = select_tiles(
            GENERIC_DLOCAL_KV_TILES, block_d, query.element_size(), query.device
        )
        partial_local_kv = uninitialized_empty(
            query.shape, dtype=local_kv.dtype, device=local_kv.device
        )
        partial_local_v = (
            partial_local_kv
            if v_is_k
            else uninitialized_empty(query.shape, dtype=local_kv.dtype, device=local_kv.device)
        )
        _gather_attn_bwd_dlocal_kv[(triton.cdiv(seq_len, block_n), batch * heads)](
            query,
            local_kv,
            local_values,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            partial_local_kv,
            partial_local_v,
            QUERY_STRIDES=query.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            LOCAL_V_STRIDES=local_values.stride(),
            GRAD_LOCAL_KV_STRIDES=partial_local_kv.stride(),
            GRAD_LOCAL_V_STRIDES=partial_local_v.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            H=heads,
            GROUP=group,
            V_IS_K=v_is_k,
            S=seq_len,
            D=head_dim,
            WINDOW=sliding_window_size,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            WIDE=wide or requires_int64_offsets(partial_local_kv, partial_local_v),
            NUM_QUERY_TILES=triton.cdiv(sliding_window_size + block_n - 1, block_m),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
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
        grad_dense_kv = full_buffer(sparse_shape, 0, dtype=torch.float32, device=sparse_kv.device)
        grad_dense_v = (
            grad_dense_kv
            if v_is_k
            else full_buffer(sparse_shape, 0, dtype=torch.float32, device=sparse_kv.device)
        )
        BWD_DSPARSE_KV_DENSE[(key_tile_grid, kv_heads, batch)](
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
            LSE_STRIDES=lse.stride(),
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
            WIDE=wide or requires_int64_offsets(grad_dense_kv),
            TILE_QUERIES=tile_queries,
            BLOCK_M=DENSE_TILE_ROWS,
            BLOCK_N=dense_block_n,
            BLOCK_D=block_d,
        )
    if not sparse_backward:
        grad_sparse_kv = full_buffer(
            sparse_shape, 0, dtype=sparse_kv.dtype, device=sparse_kv.device
        )
        grad_sparse_v = (
            None
            if v_is_k
            else full_buffer(sparse_shape, 0, dtype=sparse_kv.dtype, device=sparse_kv.device)
        )
    elif use_atomic:
        if grad_dense_kv is not None:
            grad_sparse_kv_fp32, grad_sparse_v_fp32 = grad_dense_kv, grad_dense_v
        else:
            grad_sparse_kv_fp32 = full_buffer(
                sparse_shape, 0, dtype=torch.float32, device=sparse_kv.device
            )
            grad_sparse_v_fp32 = (
                grad_sparse_kv_fp32
                if v_is_k
                else full_buffer(sparse_shape, 0, dtype=torch.float32, device=sparse_kv.device)
            )
        BWD_DSPARSE_KV_SHARED_ATOMIC[
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
            kv_indices,
            cu_seqlens,
            cu_seqlens_k,
            output,
            grad_output,
            lse,
            grad_sparse_kv_fp32,
            grad_sparse_v_fp32,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SPARSE_V_STRIDES=sparse_values.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            GRAD_SPARSE_KV_STRIDES=grad_sparse_kv_fp32.stride(),
            GRAD_SPARSE_V_STRIDES=grad_sparse_v_fp32.stride(),
            B=batch,
            H=heads,
            KV_HEADS=kv_heads,
            V_IS_K=v_is_k,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            WIDE=wide,
            BLOCK_D=block_d,
            tile_spans_ptr=tile_spans,
            DENSE_TILE_QUERIES=tile_queries,
        )
        grad_sparse_kv = rounded(grad_sparse_kv_fp32)
        grad_sparse_v = grad_sparse_kv if v_is_k else rounded(grad_sparse_v_fp32)
    else:
        # Each program owns its key tile's gradient rows, so this is deterministic.
        tile_map = build_tile_query_map(selected_rows, gather_selections, sparse_seq_len, tile)
        if tile_map is None:
            # Dense tiles cover every selection, or there are none.
            if grad_dense_kv is None:
                grad_sparse_kv = full_buffer(
                    sparse_shape, 0, dtype=sparse_kv.dtype, device=sparse_kv.device
                )
                grad_sparse_v = (
                    grad_sparse_kv
                    if v_is_k
                    else full_buffer(
                        sparse_shape, 0, dtype=sparse_kv.dtype, device=sparse_kv.device
                    )
                )
            else:
                grad_sparse_kv = rounded(grad_dense_kv)
                grad_sparse_v = grad_sparse_kv if v_is_k else rounded(grad_dense_v)
        else:
            pair_queries, pair_masks, tile_offsets = tile_map
            grad_sparse_kv = uninitialized_empty(
                sparse_shape, dtype=sparse_kv.dtype, device=sparse_kv.device
            )
            grad_sparse_v = (
                grad_sparse_kv
                if v_is_k
                else uninitialized_empty(
                    sparse_shape, dtype=sparse_kv.dtype, device=sparse_kv.device
                )
            )
            # TMA gathers need 16-byte-aligned bases, which a contiguous view at an odd
            # storage offset lacks; a copy gets a fresh, aligned allocation.
            query_rows, grad_output_rows = (
                (tensor if can_use_tma(tensor) else tensor.clone()).view(-1, head_dim)
                for tensor in (query, grad_output)
            )
            BWD_DSPARSE_KV_GROUPED[(triton.cdiv(sparse_seq_len, tile), kv_heads, batch)](
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
                # Rows enumerate (pair, group head), up to seq_len * topk * group of them.
                WIDE=wide
                or requires_int64_offsets(grad_sparse_kv, grad_sparse_v, pair_queries)
                or seq_len * topk * group >= 2**31,
                TILE=tile,
                BLOCK_D=block_d,
                grad_dense_kv_ptr=grad_dense_kv,
                grad_dense_v_ptr=grad_dense_v,
                GRAD_DENSE_STRIDES=None if grad_dense_kv is None else grad_dense_kv.stride(),
                ADD_DENSE=grad_dense_kv is not None,
            )

    return (
        grad_query,
        grad_sparse_kv,
        grad_local_kv,
        grad_sink,
        query.new_empty(0) if v_is_k else grad_sparse_v,
        query.new_empty(0) if v_is_k else grad_local_v,
    )
