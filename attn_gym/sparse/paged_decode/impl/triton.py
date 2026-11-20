"""Paged sparse decode without materializing expanded entries or physical addresses.

The producer grid is ``(num_rows * kv_heads, 1, num_splits)`` for multi-token queries. Each
program finds its packed row's sequence in the query offsets and serves one KV head's
consecutive query-head group. Single-token decode uses ``(num_sequences, kv_heads, num_splits)``
and maps sequences directly to rows without a search. Padded rows read metadata but write no
results.

A row's key tiles are ordered sparse first, window second. Split ``s`` takes tiles
``s, s + num_splits, ...``. Sparse columns expand block ids into logical entries; window columns
are consecutive positions. Shared sources use one loop and exclude sparse/window overlap.
Separate sources keep the loops distinct but preserve the same split assignment. Each entry
``e`` then maps through its source's block table to physical page
``table[sequence, e // page_size]`` and slot ``e % page_size`` before loading K/V.

Each split accumulates an online softmax in base 2: a maximum, a shifted exponential sum, and
a shifted weighted-value sum. A single split adds the optional sink and writes output and any
requested LSE directly. Multiple splits write those unnormalized partials; the merge rescales
them to a common maximum, includes the sink once, and normalizes. Merge programs own disjoint
value-dimension tiles of a (row, query head); only the first value tile writes its natural-log LSE.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl
from torch import Tensor

from attn_gym._backends.triton.utils import (
    _document_ids,
    offset_strides,
    ptr_offset,
    requires_int64_offsets,
)

# log2(e) and ln(2) stay literals: Triton rechecks every module global a kernel reads on each
# launch, and reading the shared LOG2_E/LN_2 constexprs added about 5 us of host time per call
# (producer plus merge launch) on B200.


def _requires_int64_decode_offsets(args) -> bool:
    """Choose wide arithmetic for logical entry bounds or relative tensor offsets that need it.

    The block-id ceiling division and final block expansion need up to ``INDEX_BLOCK - 1``
    extra entries of headroom. Contiguous metadata has no stride products, and LSE row offsets
    are always int64. Unused separate-window tensors must not force a wide kernel variant.
    """
    separate_window = not args["SHARED_WINDOW"] and args["WINDOW"] > 0
    if (
        args["topk"] * args["INDEX_BLOCK"] > 2**31 - 1
        or args["table_width"] * args["PAGE_SIZE"] + args["INDEX_BLOCK"] - 1 > 2**31 - 1
        or (separate_window and args["wtable_width"] * args["WINDOW_PAGE_SIZE"] > 2**31 - 1)
    ):
        return True
    return requires_int64_offsets(
        args["Q"],
        args["K"],
        args["V"],
        args["BlockTable"],
        args["Indices"],
        args["Out"],
        args["PartialOut"],
        args["PartialStats"],
        *((args["WK"], args["WV"], args["WBlockTable"]) if separate_window else ()),
    )


@triton.jit
def _attend_tile(
    q,
    acc,
    m_i,
    l_i,
    K,
    V,
    k_strides,
    v_strides,
    table_row,
    entry,
    valid,
    qk_scale,
    PAGE_SIZE: tl.constexpr,
    K_ROWS: tl.constexpr,
):
    dims = tl.arange(0, q.shape[1])
    dims_v = tl.arange(0, acc.shape[1])
    # Invalid columns substitute page 0 without reading the table; the same mask suppresses
    # their K/V loads. Keep page * stride int64 even on the narrow path: narrowing it costs
    # 0.8-0.9 us on B200's small grids. Add unit-stride feature offsets after ptr_offset;
    # folding them into its sum measured 2-3% slower from batch 4 up on B200.
    page = tl.load(table_row + entry // PAGE_SIZE, mask=valid, other=0).to(tl.int64)
    offset = entry % PAGE_SIZE
    if K_ROWS:
        # Loading K in V's [entries, features] layout reuses the page/slot layout, saving
        # 0.8-2 us per call on the many-warp small-batch tiles.
        k = tl.load(
            K + ptr_offset((page[:, None], offset[:, None]), k_strides) + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k = tl.trans(k)
    else:
        k = tl.load(
            K + ptr_offset((page[None, :], offset[None, :]), k_strides) + dims[:, None],
            mask=valid[None, :],
            other=0.0,
        )
    v = tl.load(
        V + ptr_offset((page[:, None], offset[:, None]), v_strides) + dims_v[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    scores = tl.dot(q, k) * qk_scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    # An empty split has m = -inf; subtract zero to avoid exp2(-inf - -inf) producing NaN.
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.math.exp2(m_i - m_safe)
    p = tl.math.exp2(scores - m_safe[:, None])
    acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return acc, m_new, l_i


@triton.jit
def _natural_lse(m, l):
    """Convert base-2 softmax statistics to natural-log LSE; an empty sum gives ``-inf``."""
    return tl.where(l > 0, (m + tl.math.log2(l)) * 0.6931471805599453, float("-inf"))


@triton.heuristics({"USE_INT64_OFFSETS": _requires_int64_decode_offsets})
@triton.jit(
    do_not_specialize=[
        "num_rows",
        "num_sequences",
        "num_splits",
        "topk",
        "stride_table",
        "table_width",
        "stride_indices",
        "stride_wtable",
        "wtable_width",
    ]
)
def _paged_sparse_decode_kernel(
    Q,
    K,
    V,
    BlockTable,
    Indices,
    TopkLength,
    CuQ,
    SeqUsed,
    WK,
    WV,
    WBlockTable,
    Sink,
    Out,
    Lse,
    PartialOut,
    PartialStats,
    num_rows,
    num_sequences,
    num_splits,
    stride_q_row,
    stride_q_head,
    stride_k_page,
    stride_k_token,
    stride_k_head,
    stride_v_page,
    stride_v_token,
    stride_v_head,
    stride_table,
    table_width,
    stride_indices,
    stride_wk_page,
    stride_wk_token,
    stride_wk_head,
    stride_wv_page,
    stride_wv_token,
    stride_wv_head,
    stride_wtable,
    wtable_width,
    stride_out_row,
    stride_out_head,
    qk_scale,
    topk,
    INDEX_BLOCK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW_PAGE_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    SHARED_WINDOW: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_ROWS: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    SINGLE_QUERY: tl.constexpr,
    USE_PDL: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    split = tl.program_id(2)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if SINGLE_QUERY:
        kv_head = tl.program_id(1)
        sequence = tl.program_id(0)
        row = tl.load(CuQ + sequence)
    else:
        row = tl.program_id(0) // KV_HEADS
        kv_head = tl.program_id(0) % KV_HEADS
        sequence = _document_ids(CuQ, row, num_sequences, USE_INT64_OFFSETS)
        # Inactive graph capacity maps past the last sequence; keep metadata loads in bounds.
        sequence = tl.minimum(sequence, num_sequences - 1)
    query_end = tl.load(CuQ + sequence + 1)
    position = tl.load(SeqUsed + sequence) - (query_end - row)
    has_row = row < query_end
    active = has_row & (position >= 0)
    row = tl.where(has_row, row, 0)
    if USE_INT64_OFFSETS:
        table_width = table_width.to(tl.int64)
        wtable_width = wtable_width.to(tl.int64)

    window_start = tl.maximum(position + 1 - WINDOW, 0)
    if SHARED_WINDOW:
        # Alignment gives the window ownership of the query's incomplete block, even at WINDOW=0.
        window_start -= window_start % INDEX_BLOCK
    window_end = tl.minimum(position + 1, wtable_width * WINDOW_PAGE_SIZE)
    num_window = tl.where(active, tl.maximum(window_end - window_start, 0), 0)
    if HAS_TOPK_LENGTH:
        num_blocks = tl.minimum(tl.maximum(tl.load(TopkLength + row), 0), topk)
    else:
        num_blocks = topk
    if USE_INT64_OFFSETS:
        num_blocks = num_blocks.to(tl.int64)
    num_sparse = tl.where(active, num_blocks * INDEX_BLOCK, 0)
    if SHARED_WINDOW:
        # The int32 window start may fit while cdiv's intermediate addition overflows.
        sparse_limit = window_start.to(tl.int64) if USE_INT64_OFFSETS else window_start
    else:
        sparse_limit = table_width * PAGE_SIZE

    heads = tl.arange(0, BLOCK_M)
    if USE_INT64_OFFSETS:
        heads = heads.to(tl.int64)
        kv_head = kv_head.to(tl.int64)
    # Padded query slots may read metadata, but must not load Q or write outputs/partials.
    head_mask = (heads < GROUP) & has_row
    first_head = kv_head * GROUP
    dims = tl.arange(0, HEAD_DIM)
    q = tl.load(
        Q
        + row.to(tl.int64) * stride_q_row
        + (first_head + heads[:, None]) * stride_q_head
        + dims[None, :],
        mask=head_mask[:, None],
        other=0.0,
    )

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM_V), tl.float32)
    columns = tl.arange(0, BLOCK_N)
    table_row = BlockTable + sequence.to(tl.int64) * stride_table
    k_head = K + kv_head * stride_k_head
    v_head = V + kv_head * stride_v_head
    # Widen slot * stride before multiplication when needed; page products are always int64.
    k_strides = offset_strides((stride_k_page, stride_k_token), USE_INT64_OFFSETS)
    v_strides = offset_strides((stride_v_page, stride_v_token), USE_INT64_OFFSETS)

    sparse_tiles = tl.cdiv(num_sparse, BLOCK_N)
    window_tiles = tl.cdiv(num_window, BLOCK_N)
    if tl.constexpr(topk.dtype == tl.int32 and INDEX_BLOCK <= BLOCK_N // 2):
        # sparse_tiles <= ceil(topk/2) and window_tiles <= ceil(INT32_MAX/BLOCK_N), so even
        # their sum fits int32 here. Int64 loop counters and split modulo cost 6-7% on B200
        # with a large cache. Entry offsets can still exceed int32 after multiplying by BLOCK_N.
        sparse_tiles = sparse_tiles.to(tl.int32)
        window_tiles = window_tiles.to(tl.int32)
    if SHARED_WINDOW:
        # Sharing the source lets one loop handle both sets; a second loop adds latency even
        # when empty. SINGLE_TILE removes the loop and simplifies the first softmax update.
        # Runtime masks handle splits beyond this row's tiles.
        remaining_tiles = 1 if SINGLE_TILE else sparse_tiles + window_tiles - split
        for relative_tile in range(0, remaining_tiles, 1 if SINGLE_TILE else num_splits):
            tile = split + relative_tile
            tile_offset = tile.to(tl.int64) if USE_INT64_OFFSETS else tile
            column = tile_offset * BLOCK_N + columns
            is_sparse = tile < sparse_tiles
            # The is_sparse and window_column >= 0 masks are logically redundant but kept for
            # codegen. On B200, dropping is_sparse measured 2% faster at QSA batch 1 but 0.9%
            # slower at TP4 batch 64; dropping window_column >= 0 too measured up to 1.6% slower
            # at TP4 batch 64; replacing the window_column formula measured 4% slower at QSA
            # batch 4. The sparse_tile_offset cast also keeps the int32 tile count from
            # overflowing once multiplied by BLOCK_N.
            block = tl.load(
                Indices + row.to(tl.int64) * stride_indices + column // INDEX_BLOCK,
                mask=is_sparse & (column < num_sparse),
                other=-1,
            )
            # Bound block ids before multiplication: int32 wrap could otherwise make an
            # oversized id select an unrelated entry or a negative block-table offset.
            block = tl.where(block < tl.cdiv(sparse_limit, INDEX_BLOCK), block, -1)
            if USE_INT64_OFFSETS:
                block = block.to(tl.int64)
            sparse_entry = block * INDEX_BLOCK + column % INDEX_BLOCK
            sparse_tile_offset = sparse_tiles.to(tl.int64) if USE_INT64_OFFSETS else sparse_tiles
            window_column = column - sparse_tile_offset * BLOCK_N
            entry = tl.where(is_sparse, sparse_entry, window_start + window_column)
            valid = tl.where(
                is_sparse,
                (block >= 0) & (sparse_entry < sparse_limit),
                (window_column >= 0) & (window_column < num_window),
            )
            acc, m_i, l_i = _attend_tile(
                q,
                acc,
                m_i,
                l_i,
                k_head,
                v_head,
                k_strides,
                v_strides,
                table_row,
                entry,
                valid,
                qk_scale,
                PAGE_SIZE,
                K_ROWS,
            )
    else:
        for tile in range(split, sparse_tiles, num_splits):
            tile_offset = tile.to(tl.int64) if USE_INT64_OFFSETS else tile
            column = tile_offset * BLOCK_N + columns
            block = tl.load(
                Indices + row.to(tl.int64) * stride_indices + column // INDEX_BLOCK,
                mask=column < num_sparse,
                other=-1,
            )
            block = tl.where(block < tl.cdiv(sparse_limit, INDEX_BLOCK), block, -1)
            if USE_INT64_OFFSETS:
                block = block.to(tl.int64)
            entry = block * INDEX_BLOCK + column % INDEX_BLOCK
            valid = (block >= 0) & (entry < sparse_limit)
            acc, m_i, l_i = _attend_tile(
                q,
                acc,
                m_i,
                l_i,
                k_head,
                v_head,
                k_strides,
                v_strides,
                table_row,
                entry,
                valid,
                qk_scale,
                PAGE_SIZE,
                K_ROWS,
            )
        if WINDOW > 0:
            w_table_row = WBlockTable + sequence.to(tl.int64) * stride_wtable
            wk_head = WK + kv_head * stride_wk_head
            wv_head = WV + kv_head * stride_wv_head
            wk_strides = offset_strides((stride_wk_page, stride_wk_token), USE_INT64_OFFSETS)
            wv_strides = offset_strides((stride_wv_page, stride_wv_token), USE_INT64_OFFSETS)
            # For window tile w, (sparse_tiles + w) % num_splits must equal this split.
            first = (split - sparse_tiles % num_splits + num_splits) % num_splits
            for tile in range(first, window_tiles, num_splits):
                column = tile * BLOCK_N + columns
                entry = window_start + column
                valid = column < num_window
                acc, m_i, l_i = _attend_tile(
                    q,
                    acc,
                    m_i,
                    l_i,
                    wk_head,
                    wv_head,
                    wk_strides,
                    wv_strides,
                    w_table_row,
                    entry,
                    valid,
                    qk_scale,
                    WINDOW_PAGE_SIZE,
                    K_ROWS,
                )

    dims_v = tl.arange(0, HEAD_DIM_V)
    if num_splits == 1:
        if HAS_SINK:
            sink = tl.load(Sink + first_head + heads, mask=head_mask, other=0.0).to(tl.float32)
            sink = sink * 1.4426950408889634
            m_new = tl.maximum(m_i, sink)
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.math.exp2(sink - m_new)
            acc = acc * alpha[:, None]
            m_i = m_new
        out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
        tl.store(
            Out
            + row.to(tl.int64) * stride_out_row
            + (first_head + heads[:, None]) * stride_out_head
            + dims_v[None, :],
            out.to(Out.dtype.element_ty),
            mask=head_mask[:, None],
        )
        if RETURN_LSE:
            lse = _natural_lse(m_i, l_i)
            tl.store(Lse + row.to(tl.int64) * NUM_HEADS + first_head + heads, lse, mask=head_mask)
    else:
        partial_row = (split.to(tl.int64) * num_rows + row) * NUM_HEADS + first_head + heads
        # The merge reads every split for each real row/head, so empty splits must write
        # (acc=0, max=-inf, sum=0). Unnormalized sums avoid a division and log2 per split
        # followed by exp2 to reconstruct merge weights.
        tl.store(
            PartialOut + partial_row[:, None] * HEAD_DIM_V + dims_v[None, :],
            acc,
            mask=head_mask[:, None],
        )
        tl.store(PartialStats + 2 * partial_row, m_i, mask=head_mask)
        tl.store(PartialStats + 2 * partial_row + 1, l_i, mask=head_mask)


@triton.jit(do_not_specialize=["num_rows", "num_splits"])
def _merge_splits_kernel(
    PartialOut,
    PartialStats,
    Sink,
    Out,
    Lse,
    num_rows,
    num_splits,
    stride_out_row,
    stride_out_head,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    # Disjoint value tiles bound each 2-warp program's register use: merging the whole
    # [splits, head_dim_v] tile spilled and serialized small batches at high split counts.
    row = tl.program_id(0)
    head = tl.program_id(1)
    if USE_PDL:
        # Programmatic dependent launch can start this kernel before the partials are ready.
        tl.extra.cuda.gdc_wait()
    splits = tl.arange(0, BLOCK_SPLITS)
    dims = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    partial_row = (splits.to(tl.int64) * num_rows + row) * NUM_HEADS + head
    maxima = tl.load(PartialStats + 2 * partial_row, mask=splits < num_splits, other=float("-inf"))
    sums = tl.load(PartialStats + 2 * partial_row + 1, mask=splits < num_splits, other=0.0)
    m = tl.max(maxima, axis=0)
    if HAS_SINK:
        sink = tl.load(Sink + head).to(tl.float32) * 1.4426950408889634
        m = tl.maximum(m, sink)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    weights = tl.math.exp2(maxima - m_safe)
    denominator = tl.sum(weights * sums, axis=0)
    if HAS_SINK:
        denominator += tl.math.exp2(sink - m_safe)
    partial = tl.load(
        PartialOut + partial_row[:, None] * HEAD_DIM_V + dims[None, :],
        mask=(splits < num_splits)[:, None],
        other=0.0,
    )
    out = tl.sum(partial * weights[:, None], axis=0)
    out = tl.where(denominator > 0, out / denominator, 0.0)
    tl.store(
        Out + row.to(tl.int64) * stride_out_row + head * stride_out_head + dims,
        out.to(Out.dtype.element_ty),
    )
    # All value tiles compute the same denominator; only tile zero may write the shared LSE.
    if RETURN_LSE and tl.program_id(2) == 0:
        lse = _natural_lse(m_safe, denominator)
        tl.store(Lse + row.to(tl.int64) * NUM_HEADS + head, lse)


# Upper bound on splits from every launch path: single-tile launches use one split per tile up
# to this many tiles, and looped launches use fewer. The merge's value tiling relies on it.
_MAX_SPLITS = 256


def _tile_config(
    head_dim: int, head_dim_v: int, num_programs: int, sparse_entries: int, block_m: int
) -> tuple[int, int, int, bool]:
    """Choose (block_n, num_warps, num_stages, k_rows) from B200 decode measurements.

    ``num_programs`` counts (row, KV head) pairs before splitting. Up to 3 pairs, prioritize
    latency: bucket tile widths to 16/32/64 entries, aiming for 128 sparse tiles. Wider tiles
    keep larger QSA budgets on the one-tile-per-split path (2048-block batch-1 decode: 9.2 us
    versus 16.6 us with looped 16-entry tiles).

    From 4 pairs on, 64-entry tiles on 2 warps win for QSA. Combined K/V widths above 512 use
    32-entry tiles and 8 warps; large head groups also need 8 warps to avoid Q/value accumulator
    spills. ``k_rows`` selects K loads in the same [entries, features] layout as V.
    """
    if num_programs <= 3:
        if head_dim + head_dim_v <= 512 and sparse_entries > 2048:
            block_n = 32 if sparse_entries <= 4096 else 64
            return block_n, 8 if num_programs <= 2 else 4, 2, True
        return 16, 4, 2, True
    if head_dim + head_dim_v > 512:
        return 32, 8, 2, False
    if block_m * (head_dim + head_dim_v) > 16384:
        return 64, 8, 2, False
    return 64, 2, 2, False


@functools.cache
def _cta_slots(device_index: int, smem_per_cta: int) -> int:
    """Estimate concurrent CTAs from shared memory, with 1 KiB headroom and a 4-per-SM cap.

    This split-selection heuristic does not model register or other residency limits.
    """
    properties = torch.cuda.get_device_properties(device_index)
    per_sm = properties.shared_memory_per_multiprocessor // (smem_per_cta + 1024)
    return properties.multi_processor_count * max(1, min(per_sm, 4))


@functools.lru_cache(maxsize=256)
def _choose_splits(num_programs: int, num_tiles: int, slots: int) -> int:
    """Pick the split count minimizing a wave-fill cost model.

    A wave filling fraction ``f`` of the estimated CTA slots costs ``0.8 + f`` per tile.
    Multiple splits add a merge-work penalty equal to their total wave count. The constants
    were fitted on B200 across 1-1024 (row, KV head) pairs; the chosen latency stayed within
    5% of the best of 32 split counts measured at each size.

    Memoization avoids repeating the host-side sweep on every eager call. Its bounded cache
    stores only integer decisions, not kernels specialized to exact row or split counts.
    """
    best_cost, best_splits = float("inf"), 1
    for splits in range(1, min(num_tiles, 64) + 1):
        waves = num_programs * splits / slots
        full_waves, partial = divmod(waves, 1)
        tiles = -(-num_tiles // splits)
        cost = tiles * (full_waves * 1.8 + (0.8 + partial if partial else 0))
        cost += waves if splits > 1 else 0
        if cost < best_cost:
            best_cost, best_splits = cost, splits
    return best_splits


def paged_sparse_decode(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    indices: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    topk_length: Tensor | None,
    window_key_cache: Tensor | None,
    window_value_cache: Tensor | None,
    window_block_table: Tensor | None,
    attention_sink: Tensor | None,
    max_seqlen_q: int,
    index_block_size: int,
    window_size: int,
    scale: float,
    *,
    return_lse: bool,
) -> tuple[Tensor, Tensor | None]:
    num_rows, num_heads, head_dim = query.shape
    kv_heads = key_cache.shape[2]
    group = num_heads // kv_heads
    head_dim_v = value_cache.shape[-1]
    block_m = max(16, triton.next_power_of_2(group))
    for name, dim in (("head_dim", head_dim), ("head_dim_v", head_dim_v)):
        if dim < 16 or dim & (dim - 1):
            raise ValueError(
                f"The fused paged_sparse_decode needs {name} to be a power of two of at least "
                f"16, got {dim}; use impl='reference'."
            )
    if block_m * max(head_dim, head_dim_v) > 64 * 256:
        raise ValueError(
            f"The fused paged_sparse_decode keeps {group} query heads per KV head of width "
            f"{max(head_dim, head_dim_v)} in registers, which exceeds its 16K-element tile; use "
            "impl='reference'."
        )
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"The fused paged_sparse_decode needs float16 or bfloat16 inputs, got {query.dtype}."
        )
    shared_window = window_key_cache is None
    if shared_window:
        window_key_cache, window_value_cache, window_block_table = (
            key_cache,
            value_cache,
            block_table,
        )

    out = query.new_empty(num_rows, num_heads, head_dim_v)
    # torch.empty with an explicit device is cheaper per call than Tensor.new_empty here.
    lse = (
        torch.empty(num_rows, num_heads, dtype=torch.float32, device=query.device)
        if return_lse
        else None
    )
    if num_rows == 0 or seqused_k.shape[0] == 0:
        return out, lse

    num_programs = num_rows * kv_heads
    topk = indices.shape[1]
    sparse_entries = topk * index_block_size
    block_n, num_warps, num_stages, k_rows = _tile_config(
        head_dim, head_dim_v, num_programs, sparse_entries, block_m
    )
    smem_per_cta = query.element_size() * (block_m * head_dim + block_n * (head_dim + head_dim_v))
    max_window = window_size + (index_block_size - 1 if shared_window else 0)
    num_tiles = triton.cdiv(sparse_entries, block_n) + triton.cdiv(max_window, block_n)
    single_tile = shared_window and k_rows and num_tiles <= _MAX_SPLITS
    num_tiles = max(num_tiles, 1)
    if k_rows:
        # A 512-block QSA budget already fills 128 sparse tiles; the incomplete block adds
        # one window tile. Capping at 128 splits gave one CTA two tiles and delayed the merge.
        num_splits = num_tiles if single_tile else min(num_tiles, 128)
    else:
        num_splits = _choose_splits(
            num_programs, num_tiles, _cta_slots(query.device.index, smem_per_cta)
        )
    use_pdl = num_splits > 1 and torch.cuda.get_device_capability(query.device)[0] >= 9
    if num_splits > 1:
        partial_out = torch.empty(
            num_splits, num_rows, num_heads, head_dim_v, dtype=torch.float32, device=query.device
        )
        partial_stats = torch.empty(
            num_splits, num_rows, num_heads, 2, dtype=torch.float32, device=query.device
        )
    else:
        partial_out = partial_stats = out
    # Compile-time flags suppress all accesses through placeholders for absent sink/LSE tensors.
    sink_arg = attention_sink if attention_sink is not None else query
    lse_arg = lse if lse is not None else out

    num_sequences = seqused_k.shape[0]
    single_query = max_seqlen_q == 1
    grid = (
        (num_sequences, kv_heads, num_splits)
        if single_query
        else (num_rows * kv_heads, 1, num_splits)
    )
    _paged_sparse_decode_kernel[grid](
        query,
        key_cache,
        value_cache,
        block_table,
        indices,
        topk_length if topk_length is not None else indices,
        cu_seqlens_q,
        seqused_k,
        window_key_cache,
        window_value_cache,
        window_block_table,
        sink_arg,
        out,
        lse_arg,
        partial_out,
        partial_stats,
        num_rows,
        num_sequences,
        num_splits,
        *query.stride()[:2],
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        block_table.stride(0),
        block_table.shape[1],
        indices.stride(0),
        *window_key_cache.stride()[:3],
        *window_value_cache.stride()[:3],
        window_block_table.stride(0),
        window_block_table.shape[1],
        *out.stride()[:2],
        scale * 1.4426950408889634,
        topk=topk,
        INDEX_BLOCK=index_block_size,
        PAGE_SIZE=key_cache.shape[1],
        WINDOW_PAGE_SIZE=window_key_cache.shape[1],
        WINDOW=window_size,
        SHARED_WINDOW=shared_window,
        HAS_TOPK_LENGTH=topk_length is not None,
        HAS_SINK=attention_sink is not None,
        RETURN_LSE=return_lse,
        GROUP=group,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        NUM_HEADS=num_heads,
        KV_HEADS=kv_heads,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        K_ROWS=k_rows,
        SINGLE_TILE=single_tile,
        SINGLE_QUERY=single_query,
        USE_PDL=use_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if num_splits > 1:
        block_splits = triton.next_power_of_2(num_splits)
        # A 2-warp merge can hold 8K elements (32 x 256); larger tiles spilled, so cap each
        # value tile at 4K elements. _MAX_SPLITS keeps tiles at least 16 features wide, and
        # power-of-two dimensions leave no partial value tile.
        block_d = head_dim_v if block_splits * head_dim_v <= 8192 else 4096 // block_splits
        _merge_splits_kernel[(num_rows, num_heads, head_dim_v // block_d)](
            partial_out,
            partial_stats,
            sink_arg,
            out,
            lse_arg,
            num_rows,
            num_splits,
            *out.stride()[:2],
            HAS_SINK=attention_sink is not None,
            RETURN_LSE=return_lse,
            NUM_HEADS=num_heads,
            HEAD_DIM_V=head_dim_v,
            BLOCK_SPLITS=block_splits,
            BLOCK_D=block_d,
            USE_PDL=use_pdl,
            num_warps=2,
            launch_pdl=use_pdl,
        )
    return out, lse
