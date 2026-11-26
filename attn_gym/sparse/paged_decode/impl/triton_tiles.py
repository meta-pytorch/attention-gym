"""Query tiles for sequences with many queries whose selections are top-k sets of complete blocks.

Per-row decode programs load every selected key once per query row. Consecutive prefill rows of
one sequence share most of their keys, so this schedule attends a tile of consecutive rows of a
sequence together: the tile's ``BLOCK_M`` lanes pack ``query_rows * GROUP`` (row, query head)
pairs of one KV head, and the tile walks the sequence's logical entries ``0..p_last`` in
``BLOCK_N`` chunks, loading each K/V chunk once for all its rows.

Which entries a row may see depends on its position ``p`` and its complete-block count
``(p + 1) // index_block_size``:

- **Dense rows** have at most ``topk`` complete blocks. A top-k selection then contains every
  complete block, so the row sees all entries ``0..p`` and its indices are never read.
- **Sparse rows** see their window (shared-source alignment included) plus the entries of
  selected blocks before the window start. A pre-pass writes each sparse row's selection as a
  bitmap over blocks, which the tile reads one 32-block word per row and chunk.

Chunk boundaries are fixed multiples of ``BLOCK_N`` from entry zero, and a chunk that a row cannot
see leaves its softmax state unchanged, so each row's result does not depend on which other rows
share its tile.

A sparse row's tile work grows with its position while a per-row program reads only its
``topk * index_block_size`` selected entries, so rows at or after ``tile_position_limit`` (a
measured multiple of that budget) form a per-row suffix of their sequence, and tiles cover only the
rows before it.

Programs are persistent: ``ceil(rows / query_rows) * kv_heads`` of them loop over the real
(tile, KV head) items, so the empty requests of a padded batch launch nothing. Each scans
``cu_seqlens_q`` to count tiles and find its item's sequence. Within a sequence, later
(longer-context) tiles start first. Sequences of at most ``SHORT_QUERIES`` queries are left to
per-row programs.
"""

from __future__ import annotations

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


@triton.jit
def _bitwise_or(a, b):
    return a | b


@triton.jit(
    do_not_specialize=[
        "num_sequences",
        "stride_indices",
        "stride_bitmap",
        "topk",
        "tile_limit",
    ]
)
def _selection_bitmap_kernel(
    Indices,
    TopkLength,
    CuQ,
    SeqUsed,
    Bitmap,
    num_sequences,
    stride_indices,
    stride_bitmap,
    topk,
    tile_limit,
    INDEX_BLOCK: tl.constexpr,
    WINDOW: tl.constexpr,
    HAS_TOPK_LENGTH: tl.constexpr,
    SHORT_QUERIES: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Only sparse rows of multi-query sequences need a bitmap; other rows leave theirs unwritten
    # and the tile never reads it.
    row = tl.program_id(0)
    sequence = _document_ids(CuQ, row, num_sequences, False)
    if sequence >= num_sequences:
        return
    query_start = tl.load(CuQ + sequence)
    query_end = tl.load(CuQ + sequence + 1)
    position = tl.load(SeqUsed + sequence) - (query_end - row)
    if query_end - query_start <= SHORT_QUERIES or (position + 1) // INDEX_BLOCK <= topk:
        return
    if position >= tile_limit:
        return
    window_start = tl.maximum(position + 1 - WINDOW, 0)
    limit = window_start // INDEX_BLOCK
    if HAS_TOPK_LENGTH:
        length = tl.minimum(tl.maximum(tl.load(TopkLength + row), 0), topk)
    else:
        length = topk
    indices_row = Indices + row.to(tl.int64) * stride_indices
    bitmap_row = Bitmap + row.to(tl.int64) * stride_bitmap
    for first_word in range(0, tl.cdiv(limit, 32), BLOCK_W):
        words = first_word + tl.arange(0, BLOCK_W)
        bits = tl.zeros((BLOCK_W,), tl.int32)
        for first_column in range(0, length, BLOCK_K):
            columns = first_column + tl.arange(0, BLOCK_K)
            block = tl.load(indices_row + columns, mask=columns < length, other=-1)
            # Blocks at or after the window start are covered by the window.
            hit = (block[:, None] >= 0) & (block[:, None] < limit)
            hit &= block[:, None] // 32 == words[None, :]
            bit = tl.where(hit, tl.full((1, 1), 1, tl.int32) << (block[:, None] % 32), 0)
            bits |= tl.reduce(bit, 0, _bitwise_or)
        tl.store(bitmap_row + words, bits, mask=words < tl.cdiv(limit, 32))


@triton.jit
def _find_tile(
    CuQ,
    SeqUsed,
    num_sequences,
    slot,
    tile_limit,
    QUERY_ROWS: tl.constexpr,
    SHORT_QUERIES: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Return (sequence, tiles before it, total tiles) for a tile slot.

    The sequence is the number of sequences whose inclusive tile prefix is at most the slot, and
    the largest such prefix counts the tiles before it.
    """
    found = 0
    prior = 0
    total = 0
    for first in range(0, num_sequences, BLOCK_S):
        sequences = first + tl.arange(0, BLOCK_S)
        in_range = sequences < num_sequences
        begin = tl.load(CuQ + sequences, mask=in_range, other=0)
        end = tl.load(CuQ + sequences + 1, mask=in_range, other=0)
        num_keys = tl.load(SeqUsed + sequences, mask=in_range, other=0)
        # Rows at or after tile_limit (a suffix of each sequence) use per-row programs.
        tile_rows = tl.minimum(tl.maximum(end - num_keys + tile_limit - begin, 0), end - begin)
        tiles = tl.where(end - begin > SHORT_QUERIES, tl.cdiv(tile_rows, QUERY_ROWS), 0)
        inclusive = tl.cumsum(tiles, 0) + total
        before = in_range & (inclusive <= slot)
        found += tl.sum(before.to(tl.int32), 0)
        prior = tl.maximum(prior, tl.max(tl.where(before, inclusive, 0), 0))
        total += tl.sum(tiles, 0)
    return found, prior, total


@triton.jit(
    do_not_specialize=[
        "num_sequences",
        "stride_table",
        "table_width",
        "stride_bitmap",
        "topk",
        "tile_limit",
    ]
)
def _paged_query_tile_kernel(
    Q,
    K,
    V,
    BlockTable,
    CuQ,
    SeqUsed,
    Bitmap,
    Sink,
    Out,
    Lse,
    num_sequences,
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
    stride_bitmap,
    stride_out_row,
    stride_out_head,
    qk_scale,
    topk,
    tile_limit,
    INDEX_BLOCK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    HAS_BITMAP: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_V: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    SHORT_QUERIES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_S: tl.constexpr,
    USE_PDL: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    QUERY_ROWS: tl.constexpr = BLOCK_M // GROUP
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
    # Persistent programs visit only real (tile, KV head) items: the launch is sized by the
    # padded batch, which in a serving engine's capture metadata includes many empty requests.
    # The first scan also finds this program's first item, which is usually its only one.
    item = tl.program_id(0)
    found, prior, total = _find_tile(
        CuQ,
        SeqUsed,
        num_sequences,
        item // KV_HEADS,
        tile_limit,
        QUERY_ROWS,
        SHORT_QUERIES,
        BLOCK_S,
    )
    while item < total * KV_HEADS:
        slot = item // KV_HEADS
        kv_head = item % KV_HEADS
        if item >= tl.num_programs(0):
            found, prior, _ = _find_tile(
                CuQ, SeqUsed, num_sequences, slot, tile_limit, QUERY_ROWS, SHORT_QUERIES, BLOCK_S
            )
        sequence = found
        query_start = tl.load(CuQ + sequence)
        query_end = tl.load(CuQ + sequence + 1)
        num_keys_total = tl.load(SeqUsed + sequence)
        tile_end = tl.minimum(query_end - num_keys_total + tile_limit, query_end)
        tile = tl.cdiv(tile_end - query_start, QUERY_ROWS) - 1 - (slot - prior)

        lanes = tl.arange(0, BLOCK_M)
        rows = query_start + tile * QUERY_ROWS + lanes // GROUP
        heads = kv_head * GROUP + lanes % GROUP
        has_row = (lanes < QUERY_ROWS * GROUP) & (rows < tile_end)
        rows = tl.where(has_row, rows, query_start)
        positions = tl.where(has_row, num_keys_total - (query_end - rows), -1)
        last_position = tl.max(positions, 0)
        capacity = table_width * PAGE_SIZE
        num_keys = tl.minimum(last_position + 1, capacity)
        if HAS_BITMAP:
            sparse = (positions + 1) // INDEX_BLOCK > topk
            window_start = tl.maximum(positions + 1 - WINDOW, 0)
            window_start -= window_start % INDEX_BLOCK
            # Dense rows see every entry up to their position.
            window_start = tl.where(sparse, window_start, 0)

        dims = tl.arange(0, HEAD_DIM)
        dims_v = tl.arange(0, HEAD_DIM_V)
        rows_offset = rows.to(tl.int64)
        q = tl.load(
            Q
            + rows_offset[:, None] * stride_q_row
            + heads[:, None] * stride_q_head
            + dims[None, :],
            mask=has_row[:, None],
            other=0.0,
        )
        m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM_V), tl.float32)
        table_row = BlockTable + sequence.to(tl.int64) * stride_table
        k_head = K + kv_head.to(tl.int64) * stride_k_head
        v_head = V + kv_head.to(tl.int64) * stride_v_head
        k_strides = offset_strides((stride_k_page, stride_k_token), USE_INT64_OFFSETS)
        v_strides = offset_strides((stride_v_page, stride_v_token), USE_INT64_OFFSETS)
        if HAS_BITMAP:
            bitmap_rows = Bitmap + rows_offset * stride_bitmap

        columns = tl.arange(0, BLOCK_N)
        for start in range(0, num_keys, BLOCK_N):
            entry = start + columns
            valid = entry < num_keys
            page = tl.load(table_row + entry // PAGE_SIZE, mask=valid, other=0).to(tl.int64)
            offset = entry % PAGE_SIZE
            k = tl.load(
                k_head + ptr_offset((page[:, None], offset[:, None]), k_strides) + dims[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            v = tl.load(
                v_head + ptr_offset((page[:, None], offset[:, None]), v_strides) + dims_v[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k)) * qk_scale
            visible = valid[None, :] & (entry[None, :] <= positions[:, None])
            if HAS_BITMAP:
                outside_window = entry[None, :] < window_start[:, None]
                if (32 * INDEX_BLOCK) % BLOCK_N == 0:
                    # The chunk lies within one 32-block word of each row's bitmap.
                    word = tl.load(
                        bitmap_rows + start // (32 * INDEX_BLOCK),
                        mask=start < window_start,
                        other=0,
                    )
                    bit = (entry // INDEX_BLOCK) % 32
                    selected = ((word[:, None] >> bit[None, :]) & 1) != 0
                else:
                    word = tl.load(
                        bitmap_rows[:, None] + (entry // (32 * INDEX_BLOCK))[None, :],
                        mask=outside_window,
                        other=0,
                    )
                    selected = ((word >> ((entry // INDEX_BLOCK) % 32)[None, :]) & 1) != 0
                visible &= ~outside_window | selected
            scores = tl.where(visible, scores, float("-inf"))
            m_chunk = tl.max(scores, axis=1)
            # Rescale a row only when its maximum grows by more than 2^8: skipping the accumulator
            # update otherwise leaves exponentials of at most 2^8, and rows that keep their stale
            # maximum get alpha exactly one, so a row's result never depends on its tile neighbors.
            grow = m_chunk > m_i + 8.0
            if tl.max(grow.to(tl.int32), 0) > 0:
                # Rows that keep their maximum, including rows with none yet (-inf), keep alpha = 1.
                alpha = tl.where(grow, tl.math.exp2(m_i - m_chunk), 1.0)
                m_new = tl.where(grow, m_chunk, m_i)
                acc = acc * alpha[:, None]
                l_i = l_i * alpha
                m_i = m_new
            # A row with no visible entry yet keeps m = -inf; subtract zero to avoid NaN.
            m_safe = tl.where(m_i == float("-inf"), 0.0, m_i)
            p = tl.math.exp2(scores - m_safe[:, None])
            acc = tl.dot(p.to(v.dtype), v, acc=acc)
            l_i = l_i + tl.sum(p, axis=1)

        if HAS_SINK:
            sink = tl.load(Sink + heads, mask=has_row, other=0.0).to(tl.float32)
            sink = sink * 1.4426950408889634
            m_new = tl.maximum(m_i, sink)
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.math.exp2(sink - m_new)
            acc = acc * alpha[:, None]
            m_i = m_new
        out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
        tl.store(
            Out
            + rows_offset[:, None] * stride_out_row
            + heads[:, None] * stride_out_head
            + dims_v[None, :],
            out.to(Out.dtype.element_ty),
            mask=has_row[:, None],
        )
        if RETURN_LSE:
            lse = tl.where(l_i > 0, (m_i + tl.math.log2(l_i)) * 0.6931471805599453, float("-inf"))
            tl.store(Lse + rows_offset * NUM_HEADS + heads, lse, mask=has_row)
        item += tl.num_programs(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()


def _query_tile_config(group: int) -> tuple[int, int, int, int]:
    """Choose (block_m, block_n, num_warps, num_stages) from GB300 QSA prefill measurements.

    Head dim 256 tiles of 128 lanes (ten 12-head rows) and 64 entries on 8 warps measured fastest
    among 64/128 lanes, 32/64/128 entries, 4/8 warps, and 1-3 stages. They use 197 KiB of shared
    memory, so one tile runs per SM.
    """
    return max(128, triton.next_power_of_2(group)), 64, 8, 2


# Sparse tile rows attend every entry before their position under a bitmap mask, while per-row
# programs read only their topk * index_block_size selected entries. On GB300 QSA prefill
# (512 x 4 selected entries), limits of 2048-8192 positions measured best at 8192 for 2 x 4096
# prompts and within 2% of the best (6144) for an 8192-token prompt.
_TILE_POSITION_FACTOR = 4


# Sequences with at most this many queries (decode, speculative suffixes) use per-row programs
# with split selection: a tile would leave most of its 10 rows idle.
SHORT_QUERIES = 4


def tile_position_limit(topk: int, index_block_size: int) -> int:
    """Return the first position whose multi-query rows use per-row programs instead of tiles."""
    return _TILE_POSITION_FACTOR * max(topk, 1) * index_block_size


def selection_bitmap(
    indices: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    topk_length: Tensor | None,
    capacity: int,
    index_block_size: int,
    window_size: int,
) -> Tensor | None:
    """Return block bitmaps of sparse rows before ``capacity``, or None if there are none."""
    num_rows, topk = indices.shape
    if capacity // index_block_size <= topk:
        return None
    num_words = triton.cdiv(triton.cdiv(capacity, index_block_size), 32)
    bitmap = torch.empty(num_rows, num_words, dtype=torch.int32, device=indices.device)
    _selection_bitmap_kernel[(num_rows,)](
        indices,
        topk_length if topk_length is not None else indices,
        cu_seqlens_q,
        seqused_k,
        bitmap,
        seqused_k.shape[0],
        indices.stride(0),
        bitmap.stride(0),
        topk,
        capacity,
        INDEX_BLOCK=index_block_size,
        WINDOW=window_size,
        HAS_TOPK_LENGTH=topk_length is not None,
        SHORT_QUERIES=SHORT_QUERIES,
        BLOCK_K=128,
        BLOCK_W=32,
        num_warps=4,
    )
    return bitmap


def launch(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    bitmap: Tensor | None,
    attention_sink: Tensor | None,
    sink_arg: Tensor,
    out: Tensor,
    lse_arg: Tensor,
    index_block_size: int,
    window_size: int,
    scale: float,
    topk: int,
    tile_limit: int,
    return_lse: bool,
    use_pdl: bool,
) -> None:
    """Attend every row of multi-query sequences; per-row programs serve the other rows.

    With ``use_pdl``, this launch follows the single-query producer, which triggers its dependents
    on entry. Tiles then run concurrently with it, and every tile program, including empty slots,
    waits for the producer before exiting. Completing this grid therefore implies the producer
    completed, which the split merge launched after it relies on.
    """
    num_rows, num_heads, head_dim = query.shape
    kv_heads = key_cache.shape[2]
    group = num_heads // kv_heads
    head_dim_v = value_cache.shape[-1]
    num_sequences = seqused_k.shape[0]
    page_size = key_cache.shape[1]
    capacity = block_table.shape[1] * page_size
    block_m, block_n, num_warps, num_stages = _query_tile_config(group)
    query_rows = block_m // group
    # Programs loop over the real (tile, KV head) items, so empty capacity requests launch nothing.
    # ceil(rows / query_rows) programs per KV head leave few idle and few looping: launching one
    # per SM measured 3-14% slower on large batches, and one per slot bound (which counts every
    # request) left a 16-decode + 2 x 137 step padded to 128 requests 35% slower.
    num_programs = triton.cdiv(num_rows, query_rows) * kv_heads
    _paged_query_tile_kernel[(num_programs,)](
        query,
        key_cache,
        value_cache,
        block_table,
        cu_seqlens_q,
        seqused_k,
        bitmap if bitmap is not None else cu_seqlens_q,
        sink_arg,
        out,
        lse_arg,
        num_sequences,
        *query.stride()[:2],
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        block_table.stride(0),
        block_table.shape[1],
        bitmap.stride(0) if bitmap is not None else 0,
        *out.stride()[:2],
        scale * 1.4426950408889634,
        topk,
        tile_limit,
        INDEX_BLOCK=index_block_size,
        PAGE_SIZE=page_size,
        WINDOW=window_size,
        HAS_BITMAP=bitmap is not None,
        HAS_SINK=attention_sink is not None,
        RETURN_LSE=return_lse,
        GROUP=group,
        HEAD_DIM=head_dim,
        HEAD_DIM_V=head_dim_v,
        NUM_HEADS=num_heads,
        KV_HEADS=kv_heads,
        SHORT_QUERIES=SHORT_QUERIES,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_S=256,
        USE_PDL=use_pdl,
        USE_INT64_OFFSETS=capacity > 2**31 - 1 or requires_int64_offsets(key_cache, value_cache),
        num_warps=num_warps,
        num_stages=num_stages,
        launch_pdl=use_pdl,
    )
