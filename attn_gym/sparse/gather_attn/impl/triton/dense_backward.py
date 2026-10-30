"""Dense-tile backward for the grouped gather-attention schedule.

``plan_dense_tiles`` marks tiles of consecutive queries whose selections are dense enough in
their span that walking every key, masked by each query's selection bitmap, beats gathering
the selections (see ``_gather_attn_fwd_dense``). These kernels compute those tiles' dQ and
their contribution to the sparse dK/dV; the gather kernels skip their queries.
"""

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import _document_ids, ptr_offset

from .primitives import (
    dense_tile_rows,
    load_document_bounds,
    offset_strides,
    selection_bits,
    tuning_key,
)


@triton.jit
def _gather_attn_bwd_dq_dense(
    query_ptr,
    sparse_kv_ptr,
    sparse_v_ptr,
    masks_ptr,
    tile_spans_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    attention_sink_ptr,
    grad_query_ptr,
    grad_sink_partials_ptr,
    delta_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    SPARSE_V_STRIDES,
    LSE_STRIDES,
    GRAD_SINK_PARTIALS_STRIDES,
    num_documents,
    num_tiles,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    V_IS_K: tl.constexpr,
    STORE_DELTA: tl.constexpr,
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
    """dQ, the sink-gradient partials, and rowsum(dO * O) for BLOCK_M of a dense tile's rows.

    Rows are (query, head) pairs as in ``_gather_attn_fwd_dense``; a tile's rows split into
    BLOCK_M-row programs so wide heads fit shared memory. STORE_DELTA writes rowsum(dO * O)
    for the sparse dK/dV kernels.
    """
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    if not V_IS_K:
        SPARSE_V_STRIDES = offset_strides(SPARSE_V_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)
    GRAD_SINK_PARTIALS_STRIDES = offset_strides(GRAD_SINK_PARTIALS_STRIDES, WIDE)

    GROUP: tl.constexpr = H // KV_HEADS
    CHUNKS: tl.constexpr = (TILE_QUERIES * GROUP + BLOCK_M - 1) // BLOCK_M
    tile = tl.program_id(0) // CHUNKS
    batch = tl.program_id(1)
    kv_head = tl.program_id(2)
    tile_row = batch.to(tl.int64) * num_tiles + tile
    span_hi = tl.load(tile_spans_ptr + tile_row * 2 + 1)
    if span_hi < 0:
        return
    span_lo = tl.load(tile_spans_ptr + tile_row * 2)

    positions, heads, row_mask = dense_tile_rows(
        tile, kv_head, tl.program_id(0) % CHUNKS * BLOCK_M, S, GROUP, TILE_QUERIES, BLOCK_M, WIDE
    )
    offsets_d = tl.arange(0, BLOCK_D)
    dimension_mask = offsets_d < D
    matrix_mask = row_mask[:, None] & dimension_mask[None, :]
    row_offsets = (batch, heads[:, None], positions[:, None], offsets_d[None, :])
    query = tl.load(
        query_ptr + ptr_offset(row_offsets, QUERY_STRIDES), mask=matrix_mask, other=0.0
    )
    output = tl.load(
        output_ptr + ptr_offset(row_offsets, QUERY_STRIDES), mask=matrix_mask, other=0.0
    )
    grad_output = tl.load(
        grad_output_ptr + ptr_offset(row_offsets, QUERY_STRIDES), mask=matrix_mask, other=0.0
    )
    statistic_offsets = ptr_offset((batch, heads, positions), LSE_STRIDES)
    lse = tl.load(lse_ptr + statistic_offsets, mask=row_mask, other=0.0)
    delta = tl.sum(grad_output.to(tl.float32) * output.to(tl.float32), axis=1)
    if STORE_DELTA:
        tl.store(delta_ptr + statistic_offsets, delta, mask=row_mask)
    candidate_start = 0
    if HAS_CU_SEQLENS:
        _, _, candidate_start, _ = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, tile * TILE_QUERIES, num_documents, S, WIDE
        )

    LOG2E: tl.constexpr = 1.4426950408889634
    lse_log2 = lse * LOG2E
    mask_rows = masks_ptr + (batch.to(tl.int64) * S + positions) * MASK_WORDS
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    offsets_n = tl.arange(0, BLOCK_N)
    for start in tl.range(span_lo // BLOCK_N * BLOCK_N, span_hi + 1, BLOCK_N):
        local = start + offsets_n
        key_mask = (local <= span_hi)[:, None] & dimension_mask[None, :]
        key_offsets = (batch, kv_head, (candidate_start + local)[:, None], offsets_d[None, :])
        keys = tl.load(
            sparse_kv_ptr + ptr_offset(key_offsets, SPARSE_KV_STRIDES), mask=key_mask, other=0.0
        )
        if V_IS_K:
            values = keys
        else:
            values = tl.load(
                sparse_v_ptr + ptr_offset(key_offsets, SPARSE_V_STRIDES),
                mask=key_mask,
                other=0.0,
            )
        selected = selection_bits(mask_rows, row_mask, start, MASK_WORDS, BLOCK_M, BLOCK_N)
        scores = tl.dot(query, tl.trans(keys)) * (SCALE * LOG2E)
        probabilities = tl.where(selected, tl.math.exp2(scores - lse_log2[:, None]), 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(values))
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_query = tl.dot(grad_scores.to(keys.dtype), keys, grad_query)

    tl.store(
        grad_query_ptr + ptr_offset(row_offsets, QUERY_STRIDES),
        grad_query * SCALE,
        mask=matrix_mask,
    )
    sink = tl.load(attention_sink_ptr + heads, mask=row_mask, other=0.0)
    sink_probability = tl.where(lse == -float("inf"), 0.0, tl.exp(sink - lse))
    tl.store(
        grad_sink_partials_ptr + ptr_offset((batch, heads, positions), GRAD_SINK_PARTIALS_STRIDES),
        -sink_probability * delta,
        mask=row_mask,
    )


def _prune_dq_dense_configs(configs, args):
    """Drop tiles whose Q, dO, and pipelined K/V operands overflow Blackwell shared memory."""
    row_bytes = args["BLOCK_D"] * args["query_ptr"].element_size()
    return [
        config
        for config in configs
        if (2 * config.kwargs["BLOCK_M"] + 2 * config.kwargs["BLOCK_N"] * config.num_stages)
        * row_bytes
        <= 224 * 1024
    ]


BWD_DQ_DENSE = TritonTuner(
    _gather_attn_bwd_dq_dense,
    [
        triton.Config({"BLOCK_M": block_m, "BLOCK_N": block_n}, num_warps=8, num_stages=stages)
        for block_m in (128, 64)
        for block_n in (64, 32)
        for stages in (2, 1)
    ],
    key=tuning_key("query_ptr", "H", "KV_HEADS", "V_IS_K", "D", "TILE_QUERIES", "HAS_CU_SEQLENS"),
    prune=_prune_dq_dense_configs,
)


@triton.jit
def _gather_attn_bwd_dsparse_kv_dense(
    query_ptr,
    sparse_kv_ptr,
    sparse_v_ptr,
    grad_output_ptr,
    lse_ptr,
    delta_ptr,
    masks_ptr,
    tile_spans_ptr,
    bounds_ptr,
    cu_seqlens_k_ptr,
    cu_key_tiles_ptr,
    grad_kv_ptr,
    grad_v_ptr,
    QUERY_STRIDES,
    SPARSE_KV_STRIDES,
    SPARSE_V_STRIDES,
    LSE_STRIDES,
    GRAD_STRIDES,
    num_documents,
    num_tiles,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    V_IS_K: tl.constexpr,
    S,
    SPARSE_SEQ_LEN,
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
    """FP32 dK and dV of one document-aligned key tile from the dense query tiles over it.

    BLOCK_M rows cover one dense tile's (query, head) pairs.

    Key tiles start at multiples of BLOCK_N within their document, so each row's selection
    bits for the tile are whole bitmap words. ``cu_key_tiles`` numbers every document's
    tiles consecutively; programs past the last tile exit. ``bounds`` (see
    ``plan_dense_tiles``) limits the walk to the document's dense tiles and skips key tiles
    past every dense span, leaving their zeroed outputs for the gather schedules to add to.
    Dense spans lie within each query's bitmap, the first four top-k widths of its document,
    so at most ``32 * MASK_WORDS / BLOCK_N`` key tiles per document walk the dense tiles, and
    a tile that misses this key tile costs two scalar loads. With V_IS_K the key and value
    gradients add into one dKV.
    """
    S = tl.cast(S, tl.int64 if WIDE else tl.int32)
    QUERY_STRIDES = offset_strides(QUERY_STRIDES, WIDE)
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, WIDE)
    if not V_IS_K:
        SPARSE_V_STRIDES = offset_strides(SPARSE_V_STRIDES, WIDE)
    LSE_STRIDES = offset_strides(LSE_STRIDES, WIDE)
    GRAD_STRIDES = offset_strides(GRAD_STRIDES, WIDE)

    key_tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = tl.program_id(2)
    if HAS_CU_SEQLENS:
        document = _document_ids(cu_key_tiles_ptr, key_tile, num_documents, False)
        if document >= num_documents:
            return
        local_start = (key_tile - tl.load(cu_key_tiles_ptr + document)) * BLOCK_N
        candidate_start = tl.load(cu_seqlens_k_ptr + document)
        candidate_end = tl.load(cu_seqlens_k_ptr + document + 1)
    else:
        document = batch
        local_start = key_tile * BLOCK_N
        candidate_start = 0
        candidate_end = SPARSE_SEQ_LEN
    first_tile = tl.load(bounds_ptr + document * 3)
    last_tile = tl.load(bounds_ptr + document * 3 + 1)
    if local_start > tl.load(bounds_ptr + document * 3 + 2):
        return

    offsets_n = tl.arange(0, BLOCK_N)
    keys = candidate_start + local_start + offsets_n
    offsets_d = tl.arange(0, BLOCK_D)
    dimension_mask = offsets_d < D
    key_mask = (keys < candidate_end)[:, None] & dimension_mask[None, :]
    key_offsets = (batch, kv_head, keys[:, None], offsets_d[None, :])
    sparse_keys = tl.load(
        sparse_kv_ptr + ptr_offset(key_offsets, SPARSE_KV_STRIDES), mask=key_mask, other=0.0
    )
    if V_IS_K:
        sparse_v = sparse_keys
    else:
        sparse_v = tl.load(
            sparse_v_ptr + ptr_offset(key_offsets, SPARSE_V_STRIDES), mask=key_mask, other=0.0
        )

    LOG2E: tl.constexpr = 1.4426950408889634
    GROUP: tl.constexpr = H // KV_HEADS
    grad_key = tl.zeros((BLOCK_D, BLOCK_N), tl.float32)
    grad_value = tl.zeros((BLOCK_D, BLOCK_N), tl.float32)
    for tile in range(first_tile, last_tile + 1):
        tile_row = batch.to(tl.int64) * num_tiles + tile
        span_lo = tl.load(tile_spans_ptr + tile_row * 2)
        span_hi = tl.load(tile_spans_ptr + tile_row * 2 + 1)
        # Gather tiles carry span_hi = -1, which no key tile reaches.
        if (span_hi >= local_start) & (span_lo < local_start + BLOCK_N):
            positions, heads, row_mask = dense_tile_rows(
                tile, kv_head, 0, S, GROUP, TILE_QUERIES, BLOCK_M, WIDE
            )
            matrix_mask = row_mask[:, None] & dimension_mask[None, :]
            row_offsets = (batch, heads[:, None], positions[:, None], offsets_d[None, :])
            query = tl.load(
                query_ptr + ptr_offset(row_offsets, QUERY_STRIDES),
                mask=matrix_mask,
                other=0.0,
            )
            grad_output = tl.load(
                grad_output_ptr + ptr_offset(row_offsets, QUERY_STRIDES),
                mask=matrix_mask,
                other=0.0,
            )
            statistic_offsets = ptr_offset((batch, heads, positions), LSE_STRIDES)
            lse = tl.load(lse_ptr + statistic_offsets, mask=row_mask, other=0.0)
            delta = tl.load(delta_ptr + statistic_offsets, mask=row_mask, other=0.0)
            mask_rows = masks_ptr + (batch.to(tl.int64) * S + positions) * MASK_WORDS
            selected = selection_bits(
                mask_rows, row_mask, local_start, MASK_WORDS, BLOCK_M, BLOCK_N
            )

            scores = tl.dot(query, tl.trans(sparse_keys)) * (SCALE * LOG2E)
            probabilities = tl.where(selected, tl.math.exp2(scores - (lse * LOG2E)[:, None]), 0.0)
            grad_probabilities = tl.dot(grad_output, tl.trans(sparse_v))
            grad_scores = probabilities * (grad_probabilities - delta[:, None])
            grad_value = tl.dot(
                tl.trans(grad_output), probabilities.to(grad_output.dtype), grad_value
            )
            grad_key = tl.dot(tl.trans(query), grad_scores.to(query.dtype), grad_key)

    store_mask = dimension_mask[:, None] & (keys < candidate_end)[None, :]
    store_offsets = ptr_offset((batch, kv_head, keys[None, :], offsets_d[:, None]), GRAD_STRIDES)
    grad_key *= SCALE
    if V_IS_K:
        grad_key += grad_value
    tl.store(grad_kv_ptr + store_offsets, grad_key, mask=store_mask)
    if not V_IS_K:
        tl.store(grad_v_ptr + store_offsets, grad_value, mask=store_mask)


BWD_DSPARSE_KV_DENSE = TritonTuner(
    _gather_attn_bwd_dsparse_kv_dense,
    [triton.Config({}, num_warps=num_warps, num_stages=1) for num_warps in (8, 4)],
    key=tuning_key("query_ptr", "H", "KV_HEADS", "V_IS_K", "D", "TILE_QUERIES", "HAS_CU_SEQLENS"),
)


def document_key_tiles(
    cu_seqlens_k: torch.Tensor | None, sparse_seq_len: int, block_n: int
) -> tuple[torch.Tensor | None, int]:
    """Number every document's ``block_n``-key tiles consecutively.

    Returns the per-document tile offsets (None without documents) and a grid size that
    bounds their total without reading the offsets back to the host.
    """
    if cu_seqlens_k is None:
        return None, triton.cdiv(sparse_seq_len, block_n)
    tiles = (cu_seqlens_k.diff() + block_n - 1) // block_n
    cu_key_tiles = torch.cat([tiles.new_zeros(1), tiles.cumsum(0)]).to(torch.int32)
    return cu_key_tiles, triton.cdiv(sparse_seq_len, block_n) + cu_seqlens_k.numel() - 1
