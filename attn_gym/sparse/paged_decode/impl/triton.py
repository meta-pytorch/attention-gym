"""Triton split-K kernel for paged sparse decode.

One program owns a (sequence, query) pair, a KV head with its query-head group, and one split of
the row's key tiles. Sparse tiles come first, then window tiles, dealt to splits round-robin; a
second kernel merges their softmax statistics. Indices stay logical: each key column reads its index
block's id, adds its offset inside the block, and translates the entry through the block table,
so no expanded token list or physical slot list is ever materialized.
"""

from __future__ import annotations

import functools
import math

import torch
import triton
import triton.language as tl
from torch import Tensor

from attn_gym._backends.triton.utils import requires_int64_offsets


def _requires_int64_decode_offsets(args) -> bool:
    """Inventory strided inputs and scratch; disabled window inputs do not affect codegen.

    Contiguous metadata uses direct indices, and LSE already uses int64 row arithmetic.
    Include an index block's overhang when bounding logical entry arithmetic.
    """
    if (
        args["topk"] * args["INDEX_BLOCK"] > 2**31 - 1
        or args["table_width"] * args["PAGE_SIZE"] + args["INDEX_BLOCK"] - 1 > 2**31 - 1
        or (
            (args["SHARED_WINDOW"] or args["WINDOW"] > 0)
            and args["wtable_width"] * args["WINDOW_PAGE_SIZE"] > 2**31 - 1
        )
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
        *(
            (args["WK"], args["WV"], args["WBlockTable"])
            if not args["SHARED_WINDOW"] and args["WINDOW"] > 0
            else ()
        ),
    )


@triton.jit
def _attend_tile(
    q,
    acc,
    m_i,
    l_i,
    K,
    V,
    page,
    offset,
    valid,
    stride_k_page,
    stride_k_token,
    stride_v_page,
    stride_v_token,
    qk_scale,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    K_ROWS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    dims = tl.arange(0, HEAD_DIM)
    dims_v = tl.arange(0, HEAD_DIM_V)
    # Callers mask page-table loads with other=0; the same mask guards these cache loads.
    # Widen before either multiply, including the within-page token offset.
    # Keep the page multiply wide even on the narrow path: narrowing it costs 0.8-0.9 us
    # on B200's small grids. Other offset products only widen for large layouts.
    page = page.to(tl.int64)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    if K_ROWS:
        # Keys in the values' [keys, dims] layout: the page and offset vectors then need no
        # second layout, which saved 0.8-2 us per call on the many-warp small-batch tiles.
        k = tl.load(
            K + page[:, None] * stride_k_page + offset[:, None] * stride_k_token + dims[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        k = tl.trans(k)
    else:
        k = tl.load(
            K + page[None, :] * stride_k_page + offset[None, :] * stride_k_token + dims[:, None],
            mask=valid[None, :],
            other=0.0,
        )
    v = tl.load(
        V + page[:, None] * stride_v_page + offset[:, None] * stride_v_token + dims_v[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    scores = tl.dot(q, k) * qk_scale
    scores = tl.where(valid[None, :], scores, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    # Rows with no valid key yet keep m = -inf; shift by zero so exp2 gives 0, not NaN.
    m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
    alpha = tl.math.exp2(m_i - m_safe)
    p = tl.math.exp2(scores - m_safe[:, None])
    acc = tl.dot(p.to(v.dtype), v, acc=acc * alpha[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return acc, m_new, l_i


@triton.heuristics({"USE_INT64_OFFSETS": _requires_int64_decode_offsets})
@triton.jit(
    do_not_specialize=[
        "num_rows",
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
    USE_PDL: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    # The grid walks (sequence, query) pairs, so every metadata load issues at once instead of
    # after a search of cu_seqlens_q; pairs past a sequence's query count do nothing.
    sequence = tl.program_id(0)
    query = tl.program_id(1) // KV_HEADS
    kv_head = tl.program_id(1) % KV_HEADS
    split = tl.program_id(2)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    query_start = tl.load(CuQ + sequence)
    query_end = tl.load(CuQ + sequence + 1)
    row = query_start + query
    position = tl.load(SeqUsed + sequence) - (query_end - row)
    has_row = row < query_end
    active = has_row & (position >= 0)
    row = tl.where(has_row, row, 0)
    if USE_INT64_OFFSETS:
        table_width = table_width.to(tl.int64)
        wtable_width = wtable_width.to(tl.int64)

    window_start = tl.maximum(position + 1 - WINDOW, 0)
    if SHARED_WINDOW:
        # The window starts on an index-block boundary so it and the blocks partition the keys.
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
        sparse_limit = window_start
    else:
        sparse_limit = table_width * PAGE_SIZE

    heads = tl.arange(0, BLOCK_M)
    if USE_INT64_OFFSETS:
        heads = heads.to(tl.int64)
        kv_head = kv_head.to(tl.int64)
    # Pairs past their sequence's queries load and store nothing.
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

    sparse_tiles = tl.cdiv(num_sparse, BLOCK_N)
    window_tiles = tl.cdiv(num_window, BLOCK_N)
    if tl.constexpr(topk.dtype == tl.int32 and INDEX_BLOCK <= BLOCK_N // 2):
        # Tile counts, unlike cache addresses, fit int32 here: sparse_tiles <= ceil(topk/2)
        # and window_tiles <= ceil(INT32_MAX/BLOCK_N). Keep the loop and split modulo narrow;
        # int64 versions cost 6-7% on B200 with a large cache. Widen tile-to-entry products below.
        sparse_tiles = sparse_tiles.to(tl.int32)
        window_tiles = window_tiles.to(tl.int32)
    if SHARED_WINDOW:
        # One loop over sparse then window tiles: a second loop costs latency even when empty.
        # Tiny grids give each split one tile. Removing the loop also lets the compiler
        # simplify the online softmax's first iteration. The masks cover unused splits.
        tile_begin = 0 if SINGLE_TILE else split
        tile_end = 1 if SINGLE_TILE else sparse_tiles + window_tiles
        tile_step = 1 if SINGLE_TILE else num_splits
        for tile_index in range(tile_begin, tile_end, tile_step):
            tile = split if SINGLE_TILE else tile_index
            tile_offset = tile.to(tl.int64) if USE_INT64_OFFSETS else tile
            column = tile_offset * BLOCK_N + columns
            is_sparse = tile < sparse_tiles
            block = tl.load(
                Indices + row.to(tl.int64) * stride_indices + column // INDEX_BLOCK,
                mask=is_sparse & (column < num_sparse),
                other=-1,
            )
            # Reject oversized ids before multiplying: an int32 wrap could turn an invalid
            # block into a valid entry (or a negative address into the page table).
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
            page = tl.load(table_row + entry // PAGE_SIZE, mask=valid, other=0)
            acc, m_i, l_i = _attend_tile(
                q,
                acc,
                m_i,
                l_i,
                k_head,
                v_head,
                page,
                entry % PAGE_SIZE,
                valid,
                stride_k_page,
                stride_k_token,
                stride_v_page,
                stride_v_token,
                qk_scale,
                HEAD_DIM,
                HEAD_DIM_V,
                K_ROWS,
                USE_INT64_OFFSETS,
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
            page = tl.load(table_row + entry // PAGE_SIZE, mask=valid, other=0)
            acc, m_i, l_i = _attend_tile(
                q,
                acc,
                m_i,
                l_i,
                k_head,
                v_head,
                page,
                entry % PAGE_SIZE,
                valid,
                stride_k_page,
                stride_k_token,
                stride_v_page,
                stride_v_token,
                qk_scale,
                HEAD_DIM,
                HEAD_DIM_V,
                K_ROWS,
                USE_INT64_OFFSETS,
            )
        if WINDOW > 0:
            w_table_row = WBlockTable + sequence.to(tl.int64) * stride_wtable
            wk_head = WK + kv_head * stride_wk_head
            wv_head = WV + kv_head * stride_wv_head
            # Window tiles continue the round-robin deal after the sparse tiles.
            first = (split - sparse_tiles % num_splits + num_splits) % num_splits
            for tile in range(first, window_tiles, num_splits):
                column = tile * BLOCK_N + columns
                entry = window_start + column
                valid = column < num_window
                page = tl.load(w_table_row + entry // WINDOW_PAGE_SIZE, mask=valid, other=0)
                acc, m_i, l_i = _attend_tile(
                    q,
                    acc,
                    m_i,
                    l_i,
                    wk_head,
                    wv_head,
                    page,
                    entry % WINDOW_PAGE_SIZE,
                    valid,
                    stride_wk_page,
                    stride_wk_token,
                    stride_wv_page,
                    stride_wv_token,
                    qk_scale,
                    HEAD_DIM,
                    HEAD_DIM_V,
                    K_ROWS,
                    USE_INT64_OFFSETS,
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
            lse = tl.where(l_i > 0, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, float("-inf"))
            tl.store(Lse + row.to(tl.int64) * NUM_HEADS + first_head + heads, lse, mask=head_mask)
    else:
        partial_row = (split.to(tl.int64) * num_rows + row) * NUM_HEADS + first_head + heads
        # Keep unnormalized sums and maxima: the merge can normalize once, avoiding a
        # division and log2 per split followed by an exp2 to reconstruct the weights.
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
    # One program per (row, head, BLOCK_D value dims): with many splits a whole
    # [splits, head_dim_v] tile in one 2-warp program spilled and serialized small batches.
    row = tl.program_id(0)
    head = tl.program_id(1)
    if USE_PDL:
        # Launched early; wait until the split kernel's partials are visible.
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
    if RETURN_LSE and tl.program_id(2) == 0:
        lse_out = tl.where(
            denominator > 0,
            (m_safe + tl.math.log2(denominator)) * 0.6931471805599453,
            float("-inf"),
        )
        tl.store(Lse + row.to(tl.int64) * NUM_HEADS + head, lse_out)


def _tile_config(
    head_dim: int, head_dim_v: int, num_programs: int, sparse_entries: int, block_m: int
) -> tuple[int, int, int, bool]:
    """Return (block_n, num_warps, num_stages, k_rows) for ``num_programs`` (row, KV head) pairs.

    Measured on B200 with QSA decode shapes. Up to 3 programs, latency rules: target 128
    sparse tiles, with widths bucketed to 16/32/64 keys. Wider tiles keep larger budgets on
    the one-tile path (2048-block batch-1 decode: 9.2 vs 16.6 us with 16-key looped tiles).
    From 4 programs on, 64-key tiles on 2 warps win for QSA. Wider rows halve the tile and
    use 8 warps; large head groups also need 8 warps to avoid spilling the Q/value accumulators.
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
    """CTAs resident at once: shared memory bounds them; cap the model at 4 per SM."""
    properties = torch.cuda.get_device_properties(device_index)
    per_sm = properties.shared_memory_per_multiprocessor // (smem_per_cta + 1024)
    return properties.multi_processor_count * max(1, min(per_sm, 4))


@functools.lru_cache(maxsize=256)
def _choose_splits(num_programs: int, num_tiles: int, slots: int) -> int:
    """Pick the split count minimizing a wave-fill cost model.

    A wave whose CTAs fill fraction ``f`` of the slots costs ``0.8 + f`` per tile a CTA owns,
    and every split beyond the first adds merge work proportional to its waves. The constants
    were fitted on B200 across 1-1024 (row, KV head) programs; the model's choice stayed within
    5% of the best of 32 split counts measured at each size.

    Memoize this host arithmetic to avoid repeating the sweep on every eager call. The bounded
    cache holds only integer decisions; compiled kernels still reuse runtime rows and splits.
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
    *,
    max_seqlen_q: int,
    index_block_size: int,
    window_size: int,
    scale: float,
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
    lse = (
        torch.empty(num_rows, num_heads, dtype=torch.float32, device=query.device)
        if return_lse
        else None
    )
    if num_rows == 0 or seqused_k.shape[0] == 0:
        return out, lse

    num_programs = num_rows * kv_heads
    topk = indices.shape[1]
    block_n, num_warps, num_stages, k_rows = _tile_config(
        head_dim, head_dim_v, num_programs, topk * index_block_size, block_m
    )
    smem_per_cta = query.element_size() * (block_m * head_dim + block_n * (head_dim + head_dim_v))
    max_window = window_size + (index_block_size - 1 if shared_window else 0)
    num_tiles = triton.cdiv(topk * index_block_size, block_n) + triton.cdiv(max_window, block_n)
    single_tile = shared_window and k_rows and num_tiles <= 256
    if k_rows:
        # Include the partial window tile: capping at 128 made one CTA process two tiles
        # for a 512-block QSA budget, delaying the entire merge on incomplete blocks.
        num_splits = max(num_tiles, 1) if single_tile else min(max(num_tiles, 1), 128)
    else:
        num_splits = _choose_splits(
            num_programs, max(num_tiles, 1), _cta_slots(query.device.index, smem_per_cta)
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

    _paged_sparse_decode_kernel[(seqused_k.shape[0], max_seqlen_q * kv_heads, num_splits)](
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
        attention_sink if attention_sink is not None else query,
        out,
        lse if lse is not None else out,
        partial_out,
        partial_stats,
        num_rows,
        num_splits,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        block_table.shape[1],
        indices.stride(0),
        window_key_cache.stride(0),
        window_key_cache.stride(1),
        window_key_cache.stride(2),
        window_value_cache.stride(0),
        window_value_cache.stride(1),
        window_value_cache.stride(2),
        window_block_table.stride(0),
        window_block_table.shape[1],
        out.stride(0),
        out.stride(1),
        scale * math.log2(math.e),
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
        USE_PDL=use_pdl,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if num_splits > 1:
        block_splits = triton.next_power_of_2(num_splits)
        # A 2-warp program merges a whole [splits, head_dim_v] tile up to 8K elements (32 x 256);
        # larger tiles spilled, so they are sliced to 4K elements per program.
        block_d = (
            head_dim_v
            if block_splits * head_dim_v <= 8192
            else min(head_dim_v, max(16, 4096 // block_splits))
        )
        _merge_splits_kernel[(num_rows, num_heads, head_dim_v // block_d)](
            partial_out,
            partial_stats,
            attention_sink if attention_sink is not None else query,
            out,
            lse if lse is not None else out,
            num_rows,
            num_splits,
            out.stride(0),
            out.stride(1),
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
