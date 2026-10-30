"""Forward kernels and launcher for Triton gather attention."""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import can_use_tma, ptr_offset, requires_int64_offsets

from ..routing import require_grouped_schedule, uses_grouped_schedule
from .primitives import (
    DENSE_TILE_ROWS,
    TileConfig,
    can_use_shared_kv_schedule,
    causal_window_mask,
    dense_tile_rows,
    dense_tile_shape,
    head_block_heads,
    in_dense_tile,
    load_bhsd,
    load_document_bounds,
    offset_strides,
    online_softmax_update,
    plan_dense_tiles,
    resolve_selected_rows,
    select_tiles,
    selection_bits,
    store_bhsd,
    tuning_key,
    uninitialized_empty,
)


@triton.jit
def _gather_attn_fwd(
    query_ptr,
    sparse_kv_ptr,
    local_kv_ptr,
    kv_indices_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    attention_sink_ptr,
    output_ptr,
    lse_ptr,
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
    """Apply online softmax over selected sparse entries and the local window."""
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    LOCAL_KV_STRIDES = offset_strides(LOCAL_KV_STRIDES, WIDE)
    KV_INDICES_STRIDES = offset_strides(KV_INDICES_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)

    query_block = tl.program_id(0)
    if WIDE:
        query_block = query_block.to(tl.int64)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D

    query = load_bhsd(
        query_ptr,
        QUERY_STRIDES,
        batch,
        head,
        offsets_m,
        offsets_d,
        query_mask[:, None] & dimension_mask[None, :],
    )

    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        query_start, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, offsets_m, num_documents, S, WIDE
        )

    sink = tl.load(attention_sink_ptr + head).to(tl.float32)
    running_max = tl.full((BLOCK_M,), sink, tl.float32)
    running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Each query has its own set of selected sparse entries.
    for selected_slot in tl.static_range(0, TOPK):
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
        # Promote before multiplying: tl.sum preserves floating-point input dtypes.
        logit = tl.sum(query.to(tl.float32) * sparse_value.to(tl.float32), axis=1) * SCALE
        logit = tl.where(valid, logit, -float("inf"))
        new_max = tl.maximum(running_max, logit)
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp(running_max - safe_max)
        probability = tl.exp(logit - safe_max)
        accumulator = accumulator * alpha[:, None] + probability[:, None] * sparse_value
        running_sum = running_sum * alpha + probability
        running_max = new_max

    # The local window is processed in tensor-core-friendly tiles.
    first_local_position = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for local_tile in tl.static_range(0, NUM_LOCAL_TILES):
        local_start = first_local_position + local_tile * BLOCK_N
        offsets_n = local_start + offsets_n_base
        local_mask = (offsets_n >= 0) & (offsets_n < S)
        local_values = load_bhsd(
            local_kv_ptr,
            LOCAL_KV_STRIDES,
            batch,
            head,
            offsets_n,
            offsets_d,
            local_mask[:, None] & dimension_mask[None, :],
        )
        logits = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, local_mask, WINDOW)
        if HAS_CU_SEQLENS:
            valid &= offsets_n[None, :] >= query_start[:, None]

        logits = tl.where(valid, logits, -float("inf"))
        accumulator, running_max, running_sum = online_softmax_update(
            accumulator, running_max, running_sum, logits, local_values
        )

    output = accumulator / tl.where(running_sum > 0, running_sum, 1.0)[:, None]
    store_bhsd(
        output_ptr,
        output,
        QUERY_STRIDES,
        batch,
        head,
        offsets_m,
        offsets_d,
        query_mask[:, None] & dimension_mask[None, :],
    )
    tl.store(
        lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
        running_max + tl.log(running_sum),
        mask=query_mask,
    )


@triton.jit
def _resolved_sparse_tile(
    query,
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
    accumulator,
    running_max,
    running_sum,
    SPARSE_KV_STRIDES,
    SPARSE_V_STRIDES,
    V_IS_K: tl.constexpr,
    SCALE: tl.constexpr,
    MASKED: tl.constexpr,
):
    """Merge one tile of compacted selections; only the last, partial tile is MASKED."""
    if MASKED:
        selected_valid = offsets_k < remaining
        selected_idx = tl.load(selected_rows_ptr + offsets_k, mask=selected_valid, other=0)
        selected_mask = selected_valid[:, None] & dimension_mask[None, :]
    else:
        selected_idx = tl.load(selected_rows_ptr + offsets_k)
        selected_mask = dimension_mask[None, :]
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
    logits = tl.dot(query, tl.trans(sparse_values), input_precision="tf32x3") * SCALE
    if MASKED:
        logits = tl.where(head_mask[:, None] & selected_valid[None, :], logits, -float("inf"))
    else:
        logits = tl.where(head_mask[:, None], logits, -float("inf"))
    return online_softmax_update(accumulator, running_max, running_sum, logits, sparse_v)


@triton.jit
def _gather_attn_fwd_shared(
    query_ptr,
    sparse_kv_ptr,
    local_kv_ptr,
    sparse_v_ptr,
    local_v_ptr,
    kv_indices_ptr,
    selected_counts_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    attention_sink_ptr,
    output_ptr,
    lse_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    LOCAL_KV_STRIDES,
    SPARSE_V_STRIDES,
    LOCAL_V_STRIDES,
    KV_INDICES_STRIDES,
    num_documents,
    LSE_STRIDES,
    B: tl.constexpr,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    V_IS_K: tl.constexpr,
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
    SPARSE_STAGES: tl.constexpr,
    RESOLVED_INDICES: tl.constexpr,
    tile_spans_ptr=None,
    DENSE_TILE_QUERIES: tl.constexpr = 0,
):
    """Process one sequence position across the query heads that share one KV head.

    With KV_HEADS == 1 every query head reads KV head 0. Otherwise the head-block grid axis
    enumerates (KV head, block of its H // KV_HEADS query heads), so each gathered K/V row is
    loaded once per group. V_IS_K reuses each key tile as its value tile. With
    RESOLVED_INDICES, ``kv_indices_ptr`` and ``selected_counts_ptr`` hold
    ``resolve_selected_rows`` output (compacted global pool rows and their counts) instead of
    document-local indices, so only the last selection tile needs masks. Raw indices are read
    only by the shared-KV schedule: one KV head that is also the values.
    DENSE_TILE_QUERIES skips the queries ``_gather_attn_fwd_dense`` already wrote.
    """
    tl.static_assert(RESOLVED_INDICES or (KV_HEADS == 1 and V_IS_K))
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    LOCAL_KV_STRIDES = offset_strides(LOCAL_KV_STRIDES, WIDE)
    if not V_IS_K:
        SPARSE_V_STRIDES = offset_strides(SPARSE_V_STRIDES, WIDE)
        LOCAL_V_STRIDES = offset_strides(LOCAL_V_STRIDES, WIDE)
    KV_INDICES_STRIDES = offset_strides(KV_INDICES_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)

    sequence = tl.program_id(0)
    if WIDE:
        sequence = sequence.to(tl.int64)
    batch = tl.program_id(1)
    head_block = tl.program_id(2)
    if in_dense_tile(tile_spans_ptr, batch, sequence, S, DENSE_TILE_QUERIES):
        return

    kv_head, offsets_h, head_mask = head_block_heads(head_block, H, KV_HEADS, BLOCK_H)
    offsets_d = tl.arange(0, BLOCK_D)
    dimension_mask = offsets_d < D

    query = tl.load(
        query_ptr
        + ptr_offset(
            (batch, offsets_h[:, None], sequence, offsets_d[None, :]),
            QUERY_STRIDES,
        ),
        mask=head_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )
    sink = tl.load(attention_sink_ptr + offsets_h, mask=head_mask, other=0.0).to(tl.float32)
    running_max = sink
    running_sum = tl.full((BLOCK_H,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        query_start, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, sequence, num_documents, S, WIDE
        )

    if TOPK and RESOLVED_INDICES:
        offsets_k = tl.arange(0, BLOCK_K)
        selected_rows_ptr = kv_indices_ptr + ptr_offset((batch, sequence, 0), KV_INDICES_STRIDES)
        selected_count = tl.load(selected_counts_ptr + batch * S + sequence)
        full_end = selected_count // BLOCK_K * BLOCK_K
        for selected_start in tl.range(0, full_end, BLOCK_K, num_stages=SPARSE_STAGES):
            accumulator, running_max, running_sum = _resolved_sparse_tile(
                query,
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
                accumulator,
                running_max,
                running_sum,
                SPARSE_KV_STRIDES,
                SPARSE_V_STRIDES,
                V_IS_K,
                SCALE,
                False,
            )
        if full_end < selected_count:
            accumulator, running_max, running_sum = _resolved_sparse_tile(
                query,
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
                accumulator,
                running_max,
                running_sum,
                SPARSE_KV_STRIDES,
                SPARSE_V_STRIDES,
                V_IS_K,
                SCALE,
                True,
            )
    elif TOPK:
        # Keep on-chip storage bounded when the selected set contains hundreds of entries.
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
            logits = tl.dot(query, tl.trans(sparse_values), input_precision="tf32x3") * SCALE
            logits = tl.where(head_mask[:, None] & selected_valid[None, :], logits, -float("inf"))
            accumulator, running_max, running_sum = online_softmax_update(
                accumulator, running_max, running_sum, logits, sparse_values
            )

    offsets_n_base = tl.arange(0, BLOCK_N)
    first_local_position = sequence - WINDOW + 1
    for local_start in tl.range(0, WINDOW, BLOCK_N, num_stages=2):
        offsets_n = first_local_position + local_start + offsets_n_base
        local_valid = (offsets_n >= 0) & (offsets_n <= sequence) & (offsets_n < S)
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
        if HAS_CU_SEQLENS:
            local_valid &= offsets_n >= query_start

        logits = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        logits = tl.where(head_mask[:, None] & local_valid[None, :], logits, -float("inf"))
        accumulator, running_max, running_sum = online_softmax_update(
            accumulator, running_max, running_sum, logits, local_v
        )

    tl.store(
        output_ptr
        + ptr_offset(
            (batch, offsets_h[:, None], sequence, offsets_d[None, :]),
            QUERY_STRIDES,
        ),
        accumulator / tl.where(running_sum > 0, running_sum, 1.0)[:, None],
        mask=head_mask[:, None] & dimension_mask[None, :],
    )
    tl.store(
        lse_ptr + ptr_offset((batch, offsets_h, sequence), LSE_STRIDES),
        running_max + tl.log(running_sum),
        mask=head_mask,
    )


@triton.jit
def _gather_attn_fwd_dense(
    query_ptr,
    sparse_kv_ptr,
    sparse_v_ptr,
    masks_ptr,
    tile_spans_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    attention_sink_ptr,
    output_ptr,
    lse_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    SPARSE_V_STRIDES,
    LSE_STRIDES,
    num_documents,
    num_tiles,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    V_IS_K: tl.constexpr,
    S,
    D: tl.constexpr,
    MASK_WORDS: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_CU_SEQLENS: tl.constexpr,
    WIDE: tl.constexpr,
    TILE_QUERIES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Walk every key in a dense tile's selection span, masked by each query's bitmap.

    Rows are the (query, head) pairs of TILE_QUERIES consecutive queries and the query heads
    sharing one KV head, so contiguous K/V tiles feed full tensor-core tiles instead of one
    query's gathered rows. Tiles ``plan_dense_tiles`` left to the gather kernel exit at once.
    """
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    if not V_IS_K:
        SPARSE_V_STRIDES = offset_strides(SPARSE_V_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)

    tile = tl.program_id(0)
    batch = tl.program_id(1)
    kv_head = tl.program_id(2)
    tile_row = batch.to(tl.int64) * num_tiles + tile
    span_hi = tl.load(tile_spans_ptr + tile_row * 2 + 1)
    if span_hi < 0:
        return
    span_lo = tl.load(tile_spans_ptr + tile_row * 2)

    positions, heads, row_mask = dense_tile_rows(
        tile, kv_head, 0, S, H // KV_HEADS, TILE_QUERIES, BLOCK_M, WIDE
    )
    offsets_d = tl.arange(0, BLOCK_D)
    dimension_mask = offsets_d < D
    query = tl.load(
        query_ptr
        + ptr_offset(
            (batch, heads[:, None], positions[:, None], offsets_d[None, :]), QUERY_STRIDES
        ),
        mask=row_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )
    candidate_start = 0
    if HAS_CU_SEQLENS:
        _, _, candidate_start, _ = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, tile * TILE_QUERIES, num_documents, S, WIDE
        )

    # Base-2 online softmax: the running max and logits carry a factor of log2(e).
    LOG2E: tl.constexpr = 1.4426950408889634
    sink = tl.load(attention_sink_ptr + heads, mask=row_mask, other=0.0).to(tl.float32)
    running_max = sink * LOG2E
    running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    mask_rows = masks_ptr + (batch.to(tl.int64) * S + positions) * MASK_WORDS
    offsets_n = tl.arange(0, BLOCK_N)
    for start in tl.range(span_lo // BLOCK_N * BLOCK_N, span_hi + 1, BLOCK_N):
        local = start + offsets_n
        key_mask = (local <= span_hi)[:, None] & dimension_mask[None, :]
        keys = tl.load(
            sparse_kv_ptr
            + ptr_offset(
                (batch, kv_head, (candidate_start + local)[:, None], offsets_d[None, :]),
                SPARSE_KV_STRIDES,
            ),
            mask=key_mask,
            other=0.0,
        )
        if V_IS_K:
            values = keys
        else:
            values = tl.load(
                sparse_v_ptr
                + ptr_offset(
                    (batch, kv_head, (candidate_start + local)[:, None], offsets_d[None, :]),
                    SPARSE_V_STRIDES,
                ),
                mask=key_mask,
                other=0.0,
            )
        selected = selection_bits(mask_rows, row_mask, start, MASK_WORDS, BLOCK_M, BLOCK_N)
        logits = tl.dot(query, tl.trans(keys)) * (SCALE * LOG2E)
        logits = tl.where(selected, logits, -float("inf"))
        new_max = tl.maximum(running_max, tl.max(logits, axis=1))
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.math.exp2(running_max - safe_max)
        probabilities = tl.math.exp2(logits - safe_max[:, None])
        accumulator = tl.dot(probabilities.to(values.dtype), values, accumulator * alpha[:, None])
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max

    tl.store(
        output_ptr
        + ptr_offset(
            (batch, heads[:, None], positions[:, None], offsets_d[None, :]), QUERY_STRIDES
        ),
        accumulator / tl.where(running_sum > 0, running_sum, 1.0)[:, None],
        mask=row_mask[:, None] & dimension_mask[None, :],
    )
    tl.store(
        lse_ptr + ptr_offset((batch, heads, positions), LSE_STRIDES),
        (running_max + tl.math.log2(running_sum)) / LOG2E,
        mask=row_mask,
    )


@triton.jit
def _gather_attn_fwd_tma(
    query_desc,
    sparse_kv_ptr,
    local_desc,
    kv_indices_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    attention_sink_ptr,
    output_desc,
    lse_ptr,
    SPARSE_KV_STRIDES,
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
    """TMA forward using host-created descriptors for dense tiles."""
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    SPARSE_SEQ_LEN = tl.cast(SPARSE_SEQ_LEN, tl.int64 if WIDE else tl.int32)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    KV_INDICES_STRIDES = offset_strides(KV_INDICES_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)

    query_block = tl.program_id(0)
    if WIDE:
        query_block = query_block.to(tl.int64)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D
    query = tl.reshape(
        # TMA coordinates are int32 even when descriptor byte strides need int64.
        query_desc.load([batch, head, (query_block * BLOCK_M).to(tl.int32), 0]),
        (BLOCK_M, BLOCK_D),
    )

    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        query_start, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, offsets_m, num_documents, S, WIDE
        )

    sink = tl.load(attention_sink_ptr + head).to(tl.float32)
    running_max = tl.full((BLOCK_M,), sink, tl.float32)
    running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for selected_slot in tl.static_range(0, TOPK):
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
        logit = tl.sum(query.to(tl.float32) * sparse_value.to(tl.float32), axis=1) * SCALE
        logit = tl.where(valid, logit, -float("inf"))
        new_max = tl.maximum(running_max, logit)
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp(running_max - safe_max)
        probability = tl.exp(logit - safe_max)
        accumulator = accumulator * alpha[:, None] + probability[:, None] * sparse_value
        running_sum = running_sum * alpha + probability
        running_max = new_max

    first_local_position = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for local_tile in tl.static_range(0, NUM_LOCAL_TILES):
        local_start = first_local_position + local_tile * BLOCK_N
        offsets_n = local_start + offsets_n_base
        local_mask = (offsets_n >= 0) & (offsets_n < S)
        local_values = tl.reshape(
            local_desc.load([batch, head, local_start.to(tl.int32), 0]),
            (BLOCK_N, BLOCK_D),
        )
        logits = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, local_mask, WINDOW)
        if HAS_CU_SEQLENS:
            valid &= offsets_n[None, :] >= query_start[:, None]

        logits = tl.where(valid, logits, -float("inf"))
        accumulator, running_max, running_sum = online_softmax_update(
            accumulator, running_max, running_sum, logits, local_values
        )

    output_desc.store(
        [batch, head, (query_block * BLOCK_M).to(tl.int32), 0],
        tl.reshape(
            accumulator / tl.where(running_sum > 0, running_sum, 1.0)[:, None],
            (1, 1, BLOCK_M, BLOCK_D),
        ),
    )
    tl.store(
        lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
        running_max + tl.log(running_sum),
        mask=query_mask,
    )


def prune_shared_forward_configs(configs, args):
    """Avoid local tiles that exceed shared memory for wide head dimensions."""
    if args["D"] <= 128:
        return configs
    return [config for config in configs if config.kwargs["BLOCK_N"] == 64]


_FWD_SHARED = TritonTuner(
    _gather_attn_fwd_shared,
    [
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps, num_stages=1)
        for block_n in (64, 128, 256)
        for num_warps in (4, 8)
    ],
    key=tuning_key(
        "query_ptr", "B", "H", "KV_HEADS", "V_IS_K", "D", "TOPK", "WINDOW", "HAS_CU_SEQLENS"
    ),
    prune=prune_shared_forward_configs,
)


def prune_grouped_forward_configs(configs, args):
    """Bound the selection tile by the top-k and the local tile by the head dimension."""
    max_block_k = max(32, triton.next_power_of_2(args["TOPK"]))
    max_block_n = 64 if args["D"] > 128 or args["WINDOW"] == 0 else 256
    return [
        config
        for config in configs
        if config.kwargs["BLOCK_K"] <= max_block_k and config.kwargs["BLOCK_N"] <= max_block_n
    ]


# The grouped schedule also tunes the selection tile and its software-pipelining depth.
_FWD_GROUPED = TritonTuner(
    _gather_attn_fwd_shared,
    [
        triton.Config(
            {"BLOCK_K": block_k, "BLOCK_N": block_n, "SPARSE_STAGES": stages},
            num_warps=num_warps,
            num_stages=1,
            maxnreg=maxnreg,
        )
        for block_k, stages in ((64, 3), (64, 2), (32, 3), (32, 2))
        for block_n in (64, 128, 256)
        # Capping registers can fit a third 4-warp CTA per SM, worth a few spills.
        for num_warps, maxnreg in ((4, None), (4, 168), (8, None))
    ],
    key=tuning_key(
        "query_ptr", "B", "H", "KV_HEADS", "V_IS_K", "D", "TOPK", "WINDOW", "HAS_CU_SEQLENS"
    ),
    prune=prune_grouped_forward_configs,
)
_FWD_DENSE = TritonTuner(
    _gather_attn_fwd_dense,
    [
        triton.Config({"BLOCK_N": block_n}, num_warps=8, num_stages=num_stages)
        for block_n in (64, 32)
        for num_stages in (2, 1)
    ],
    key=tuning_key("query_ptr", "H", "KV_HEADS", "V_IS_K", "D", "TILE_QUERIES", "HAS_CU_SEQLENS"),
)
_FWD_TMA = TritonTuner(
    _gather_attn_fwd_tma,
    [
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (4, 8)
        for num_stages in (1, 3)
    ],
    # The fixed tile changes the winner, so it is part of the key.
    key=tuning_key(
        "query_desc", "H", "D", "TOPK", "WINDOW", "HAS_CU_SEQLENS", "BLOCK_M", "BLOCK_N"
    ),
)

# Preferred first; later entries fit smaller shared memory (e.g. SM86 at D=256). On Blackwell,
# FP32 D=256 skips the middle tile: it fits shared memory but needs 576 tensor-memory columns.
GENERIC_FORWARD_TILES = (
    TileConfig(64, 128, 8, 3, shared_rows=256, tmem_cols=512),
    TileConfig(64, 64, 8, 3, shared_rows=192, tmem_cols=576),
    TileConfig(32, 32, 4, 1, shared_rows=96),
)


def forward_schedule(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_v: torch.Tensor | None,
    local_v: torch.Tensor | None,
    *,
    kv_heads: int,
    sliding_window_size: int,
) -> str:
    """Name the forward kernel for these inputs: grouped, shared, tma, or generic.

    Separate values and 1 < kv_heads < heads run only on the grouped schedule; inputs it
    cannot run raise rather than fall back to a schedule that would ignore them. Every other
    input keeps the shared/TMA/generic choice it had before grouped support existed.
    """
    if uses_grouped_schedule(query.shape[1], kv_heads, sparse_v is not None):
        kv_tensors = [sparse_kv, local_kv]
        if sparse_v is not None:
            kv_tensors += [sparse_v, local_v]
        require_grouped_schedule(query, kv_tensors, sliding_window_size=sliding_window_size)
        return "grouped"
    if can_use_shared_kv_schedule(query, sparse_kv, local_kv, sliding_window_size):
        return "shared"
    return "tma" if can_use_tma(query) and can_use_tma(local_kv) else "generic"


def _launch_grouped_forward(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    sliding_window_size: int,
    scale: float,
    *,
    kv_heads: int,
    sparse_v: torch.Tensor | None,
    local_v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward for separate values or grouped KV heads (see ``forward_schedule``).

    Selections are first resolved to compacted global rows. Dense tiles (see
    ``plan_dense_tiles``) walk their span in ``_gather_attn_fwd_dense``; the grouped gather
    kernel skips their queries and runs the rest.
    """
    batch, heads, seq_len, head_dim = query.shape
    topk = kv_indices.shape[-1]
    sparse_seq_len = sparse_kv.shape[2]
    block_d = max(16, triton.next_power_of_2(head_dim))
    output = uninitialized_empty(query.shape, dtype=query.dtype, device=query.device)
    lse = uninitialized_empty((batch, heads, seq_len), dtype=torch.float32, device=query.device)
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
        lse,
        cu_seqlens,
        cu_seqlens_k,
    )
    group = heads // kv_heads
    # tl.dot needs at least 16 rows; partial head tiles are masked. The D cap matches the
    # shared schedule's head tile for wide heads.
    block_h = max(16, min(32 if head_dim > 128 else 128, triton.next_power_of_2(group)))
    v_is_k = sparse_v is None
    sparse_v = sparse_kv if v_is_k else sparse_v
    local_v = local_kv if v_is_k else local_v
    dense_shape = dense_tile_shape(heads, kv_heads, topk, sliding_window_size)
    tile_queries, mask_words = dense_shape or (0, 0)
    selected_rows, selected_counts, masks, spans = resolve_selected_rows(
        kv_indices, sparse_seq_len, cu_seqlens, cu_seqlens_k, mask_words=mask_words
    )
    tile_spans = None
    if dense_shape is not None:
        tile_spans, _ = plan_dense_tiles(
            selected_counts, spans, cu_seqlens, tile_queries, document_bounds=False
        )
        num_tiles = tile_spans.shape[1]
        _FWD_DENSE[(num_tiles, batch, kv_heads)](
            query,
            sparse_kv,
            sparse_v,
            masks,
            tile_spans,
            cu_seqlens,
            cu_seqlens_k,
            attention_sink,
            output,
            lse,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SPARSE_V_STRIDES=sparse_v.stride(),
            LSE_STRIDES=lse.stride(),
            num_documents=num_documents,
            num_tiles=num_tiles,
            H=heads,
            KV_HEADS=kv_heads,
            V_IS_K=v_is_k,
            S=seq_len,
            D=head_dim,
            MASK_WORDS=mask_words,
            SCALE=scale,
            HAS_CU_SEQLENS=has_cu_seqlens,
            WIDE=wide,
            TILE_QUERIES=tile_queries,
            BLOCK_M=DENSE_TILE_ROWS,
            BLOCK_D=block_d,
        )
    _FWD_GROUPED[(seq_len, batch, kv_heads * triton.cdiv(group, block_h))](
        query,
        sparse_kv,
        local_kv,
        sparse_v,
        local_v,
        selected_rows,
        selected_counts,
        cu_seqlens,
        cu_seqlens_k,
        attention_sink,
        output,
        lse,
        QUERY_STRIDES=query.stride(),
        SPARSE_KV_STRIDES=sparse_kv.stride(),
        LOCAL_KV_STRIDES=local_kv.stride(),
        SPARSE_V_STRIDES=sparse_v.stride(),
        LOCAL_V_STRIDES=local_v.stride(),
        KV_INDICES_STRIDES=selected_rows.stride(),
        num_documents=num_documents,
        LSE_STRIDES=lse.stride(),
        B=batch,
        H=heads,
        KV_HEADS=kv_heads,
        V_IS_K=v_is_k,
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
    return output, lse


def _launch_forward(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    sliding_window_size: int,
    scale: float,
    *,
    kv_heads: int,
    sparse_v: torch.Tensor | None,
    local_v: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch forward with query, indices, and sink made contiguous by the opaque op.

    ``kv_heads`` is the KV head count before the op broadcasts a single shared head.
    """
    schedule = forward_schedule(
        query,
        sparse_kv,
        local_kv,
        sparse_v,
        local_v,
        kv_heads=kv_heads,
        sliding_window_size=sliding_window_size,
    )
    if schedule == "grouped":
        return _launch_grouped_forward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            sliding_window_size,
            scale,
            kv_heads=kv_heads,
            sparse_v=sparse_v,
            local_v=local_v,
        )
    batch, heads, seq_len, head_dim = query.shape
    topk = kv_indices.shape[-1]
    sparse_seq_len = sparse_kv.shape[2]
    block_d = max(16, triton.next_power_of_2(head_dim))
    output = torch.empty_like(query)
    lse = torch.empty(batch, heads, seq_len, device=query.device, dtype=torch.float32)
    has_cu_seqlens = cu_seqlens is not None
    num_documents = cu_seqlens.numel() - 1 if has_cu_seqlens else 0
    wide = requires_int64_offsets(
        query, sparse_kv, local_kv, kv_indices, output, lse, cu_seqlens, cu_seqlens_k
    )

    # This head-major schedule is tuned for shared KV on Blackwell.
    if schedule == "shared":
        # A smaller head tile keeps D=512 accumulators within Blackwell's resources.
        block_h = (
            triton.next_power_of_2(heads)
            if head_dim <= 128
            else min(32, triton.next_power_of_2(heads))
        )
        block_k = max(16, min(64, triton.next_power_of_2(topk)))
        _FWD_SHARED[(seq_len, batch, triton.cdiv(heads, block_h))](
            query,
            sparse_kv,
            local_kv,
            sparse_kv,
            local_kv,
            kv_indices,
            None,
            cu_seqlens,
            cu_seqlens_k,
            attention_sink,
            output,
            lse,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            SPARSE_V_STRIDES=sparse_kv.stride(),
            LOCAL_V_STRIDES=local_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            num_documents=num_documents,
            LSE_STRIDES=lse.stride(),
            B=batch,
            H=heads,
            KV_HEADS=1,
            V_IS_K=True,
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
        return output, lse

    # D=512 must also fit the generic path (non-Blackwell, FP16/FP32, or unshared KV).
    block_m, block_n, num_warps, num_stages, *_ = (
        TileConfig(16, 16, 4, 1)
        if head_dim == 512
        else select_tiles(GENERIC_FORWARD_TILES, block_d, query.element_size(), query.device)
    )
    num_local_tiles = (
        triton.cdiv(sliding_window_size + block_m - 1, block_n) if sliding_window_size else 0
    )
    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    if schedule == "tma":
        query_desc = TensorDescriptor.from_tensor(query, [1, 1, block_m, block_d])
        local_desc = TensorDescriptor.from_tensor(local_kv, [1, 1, block_n, block_d])
        output_desc = TensorDescriptor.from_tensor(output, [1, 1, block_m, block_d])
        _FWD_TMA[grid](
            query_desc,
            sparse_kv,
            local_desc,
            kv_indices,
            cu_seqlens,
            cu_seqlens_k,
            attention_sink,
            output_desc,
            lse,
            SPARSE_KV_STRIDES=sparse_kv.stride(),
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
            NUM_LOCAL_TILES=num_local_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )
    else:
        _gather_attn_fwd[grid](
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            cu_seqlens,
            cu_seqlens_k,
            attention_sink,
            output,
            lse,
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
            NUM_LOCAL_TILES=num_local_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return output, lse
