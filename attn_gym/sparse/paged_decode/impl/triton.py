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
):
    dims = tl.arange(0, HEAD_DIM)
    dims_v = tl.arange(0, HEAD_DIM_V)
    # Masked rows issue no memory traffic; page/offset are clamped to zero there.
    page = tl.where(valid, page, 0).to(tl.int64)
    offset = tl.where(valid, offset, 0)
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


@triton.jit(do_not_specialize=["num_rows", "num_splits"])
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
    TOPK: tl.constexpr,
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
    USE_PDL: tl.constexpr,
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

    window_start = tl.maximum(position + 1 - WINDOW, 0)
    if SHARED_WINDOW:
        # The window starts on an index-block boundary so it and the blocks partition the keys.
        window_start -= window_start % INDEX_BLOCK
    window_end = tl.minimum(position + 1, wtable_width * WINDOW_PAGE_SIZE)
    num_window = tl.where(active, tl.maximum(window_end - window_start, 0), 0)
    if HAS_TOPK_LENGTH:
        num_blocks = tl.minimum(tl.maximum(tl.load(TopkLength + row), 0), TOPK)
    else:
        num_blocks = TOPK
    num_sparse = tl.where(active, num_blocks * INDEX_BLOCK, 0)
    if SHARED_WINDOW:
        sparse_limit = window_start
    else:
        sparse_limit = table_width * PAGE_SIZE

    heads = tl.arange(0, BLOCK_M)
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
    if SHARED_WINDOW:
        # One loop over sparse then window tiles: a second loop costs latency even when empty.
        for tile in range(split, sparse_tiles + window_tiles, num_splits):
            column = tile * BLOCK_N + columns
            is_sparse = tile < sparse_tiles
            block = tl.load(
                Indices + row.to(tl.int64) * stride_indices + column // INDEX_BLOCK,
                mask=is_sparse & (column < num_sparse),
                other=-1,
            )
            sparse_entry = block * INDEX_BLOCK + column % INDEX_BLOCK
            window_column = column - sparse_tiles * BLOCK_N
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
            )
    else:
        for tile in range(split, sparse_tiles, num_splits):
            column = tile * BLOCK_N + columns
            block = tl.load(
                Indices + row.to(tl.int64) * stride_indices + column // INDEX_BLOCK,
                mask=column < num_sparse,
                other=-1,
            )
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
        partial_row = (split * num_rows + row).to(tl.int64) * NUM_HEADS + first_head + heads
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
    USE_PDL: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if USE_PDL:
        # Launched early; wait until the split kernel's partials are visible.
        tl.extra.cuda.gdc_wait()
    splits = tl.arange(0, BLOCK_SPLITS)
    dims = tl.arange(0, HEAD_DIM_V)
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
    if RETURN_LSE:
        lse_out = tl.where(
            denominator > 0,
            (m_safe + tl.math.log2(denominator)) * 0.6931471805599453,
            float("-inf"),
        )
        tl.store(Lse + row.to(tl.int64) * NUM_HEADS + head, lse_out)


def _tile_config(head_dim: int, head_dim_v: int) -> tuple[int, int]:
    """Return (block_n, num_warps): 64-key tiles on 2 warps won every QSA decode shape on B200.

    Wider rows halve the tile to keep a prefetched stage of K and V in shared memory.
    """
    return (64 if head_dim + head_dim_v <= 512 else 32), 2


@functools.cache
def _cta_slots(device_index: int, smem_per_cta: int) -> int:
    """CTAs resident at once: shared memory bounds them, 2-warp register use caps them at 4."""
    properties = torch.cuda.get_device_properties(device_index)
    per_sm = properties.shared_memory_per_multiprocessor // (smem_per_cta + 1024)
    return properties.multi_processor_count * max(1, min(per_sm, 4))


def _choose_splits(num_programs: int, num_tiles: int, slots: int) -> int:
    """Pick the split count minimizing a wave-fill cost model.

    A wave whose CTAs fill fraction ``f`` of the slots costs ``0.8 + f`` per tile a CTA owns,
    and every split beyond the first adds merge work proportional to its waves. The constants
    were fitted on B200 across 1-1024 (row, KV head) programs; the model's choice stayed within
    5% of the best of 32 split counts measured at each size.
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

    block_n, num_warps = _tile_config(head_dim, head_dim_v)
    smem_per_cta = query.element_size() * (block_m * head_dim + block_n * (head_dim + head_dim_v))
    topk = indices.shape[1]
    max_window = window_size + (index_block_size - 1 if shared_window else 0)
    num_tiles = triton.cdiv(topk * index_block_size, block_n) + triton.cdiv(max_window, block_n)
    num_splits = _choose_splits(
        num_rows * kv_heads, max(num_tiles, 1), _cta_slots(query.device.index, smem_per_cta)
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
        TOPK=topk,
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
        USE_PDL=use_pdl,
        num_warps=num_warps,
        num_stages=2,
    )
    if num_splits > 1:
        _merge_splits_kernel[(num_rows, num_heads)](
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
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            USE_PDL=use_pdl,
            num_warps=2,
            launch_pdl=use_pdl,
        )
    return out, lse
