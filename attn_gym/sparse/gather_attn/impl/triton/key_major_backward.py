"""Key-major sparse dK/dV for the grouped gather-attention schedule.

Query-major sparse gradients scatter one top-k-by-dimension tile per query and KV head
with FP32 atomics. Here the selections are inverted instead: each program owns a tile of
``TILE`` consecutive sparse-pool keys and walks the (query, key mask) pairs that selected any
key in the tile, accumulating dK and dV like a FlashAttention dK/dV pass over a sparse
query list. Every program exclusively owns its output rows, so the result is deterministic.
"""

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset

from .primitives import offset_strides


@triton.jit
def _tile_map_keys(
    rows_ptr,
    counts_ptr,
    offsets_ptr,
    keys_ptr,
    entries,
    S,
    QUERY_BITS,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Encode one query's selections as (key tile, query, key within tile) at its offset.

    ``offsets`` packs each batch row's selections query by query; queries with a zero count
    (dense tiles) write nothing, and the caller fills the unused tail with the sentinel.
    """
    row = tl.program_id(0).to(tl.int64)
    batch = row // S
    query_position = row % S
    count = tl.load(counts_ptr + row)
    offset = tl.load(offsets_ptr + row).to(tl.int64)
    for start in range(0, count, BLOCK):
        slots = start + tl.arange(0, BLOCK)
        valid = slots < count
        key = tl.load(rows_ptr + row * TOPK + slots, mask=valid, other=0).to(tl.int64)
        # Fields (key tile | query | key within tile); the query field is QUERY_BITS wide so
        # decoding needs only shifts.
        encoded = (
            ((key // TILE) << (QUERY_BITS + TILE.bit_length() - 1))
            | (query_position << (TILE.bit_length() - 1))
            | (key % TILE)
        )
        tl.store(
            keys_ptr + batch * entries + offset + slots,
            encoded.to(keys_ptr.dtype.element_ty),
            mask=valid,
        )


@triton.jit
def _tile_map_starts(
    sorted_keys_ptr,
    starts_ptr,
    N,
    TILE: tl.constexpr,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Flag the sorted selections that start a pair.

    A pair is one query's selections within one key tile. Equal keys sort next to each
    other, so starting a pair at every repeated selection keeps each key once per pair mask.
    """
    batch = tl.program_id(1).to(tl.int64)
    entries = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    keys_ptr = sorted_keys_ptr + batch * N
    in_bounds = entries < N
    key = tl.load(keys_ptr + entries, mask=in_bounds, other=SENTINEL)
    previous = tl.load(keys_ptr + entries - 1, mask=in_bounds & (entries >= 1), other=0)
    repeated = (entries >= 1) & (key == previous)
    starts = (key != SENTINEL) & ((entries == 0) | (key // TILE != previous // TILE) | repeated)
    tl.store(starts_ptr + batch * N + entries, starts.to(tl.int32), mask=in_bounds)


@triton.jit
def _or_within_pairs(start_a, bits_a, start_b, bits_b):
    """Segmented OR: restart at every pair start."""
    return start_a | start_b, tl.where(start_b, bits_b, bits_a | bits_b)


@triton.jit
def _tile_map_pairs(
    sorted_keys_ptr,
    pair_ends_ptr,
    pair_queries_ptr,
    pair_masks_ptr,
    pair_tiles_ptr,
    N,
    QUERY_BITS,
    TILE: tl.constexpr,
    SENTINEL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write each pair's query, tile, and key mask.

    Masks are ORed within the block by a segmented scan; only pairs that cross a block
    boundary combine their parts with atomics (order-independent, so still deterministic).
    """
    batch = tl.program_id(1).to(tl.int64)
    entries = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    in_bounds = entries < N
    keys_ptr = sorted_keys_ptr + batch * N
    ends_ptr = pair_ends_ptr + batch * N
    key = tl.load(keys_ptr + entries, mask=in_bounds, other=SENTINEL)
    valid = in_bounds & (key != SENTINEL)
    # Inclusive pair counts over all batch rows: an entry belongs to pair count - 1 (rebased to
    # its row) and starts it if the count differs from its predecessor's.
    row_base = tl.load(pair_ends_ptr + batch * N - 1, mask=batch > 0, other=0)
    pair_end = tl.load(ends_ptr + entries, mask=in_bounds, other=0)
    previous_end = tl.load(ends_ptr + entries - 1, mask=in_bounds & (entries >= 1), other=0)
    next_in_bounds = entries + 1 < N
    next_end = tl.load(ends_ptr + entries + 1, mask=next_in_bounds, other=0)
    next_key = tl.load(keys_ptr + entries + 1, mask=next_in_bounds, other=SENTINEL)
    pair = batch * N + pair_end - row_base - 1
    starts = valid & (pair_end != previous_end)
    ends = valid & ((next_end != pair_end) | (next_key == SENTINEL) | ~next_in_bounds)

    query_tile = key >> (TILE.bit_length() - 1)
    tl.store(
        pair_queries_ptr + pair,
        (query_tile & ((1 << QUERY_BITS) - 1)).to(tl.int32),
        mask=starts,
    )
    tl.store(
        pair_tiles_ptr + pair,
        (query_tile >> QUERY_BITS).to(pair_tiles_ptr.dtype.element_ty),
        mask=starts,
    )
    bits = tl.where(
        valid,
        tl.full(key.shape, 1, pair_masks_ptr.dtype.element_ty)
        << (key % TILE).to(pair_masks_ptr.dtype.element_ty),
        0,
    )
    started_here, bits = tl.associative_scan((starts, bits), 0, _or_within_pairs)
    last = tl.arange(0, BLOCK) == BLOCK - 1
    tl.store(pair_masks_ptr + pair, bits, mask=ends & started_here)
    tl.atomic_or(
        pair_masks_ptr + pair, bits, mask=valid & ((ends & ~started_here) | (last & ~ends))
    )


def build_tile_query_map(
    selected_rows: torch.Tensor,
    counts: torch.Tensor,
    sparse_seq_len: int,
    tile: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Invert query-major selections into key-tile-major (query, key mask) pairs.

    ``selected_rows`` holds each query's compacted global pool rows (see
    ``resolve_selected_rows``), and ``counts`` [B, S] how many of each row to cover. Returns
    ``pair_queries`` and ``pair_masks``, both ``(batch, entries)`` with only the leading pairs
    used, and ``tile_offsets`` ``(batch, num_tiles + 1)``: tile ``t`` of batch ``b`` owns pairs
    ``tile_offsets[b, t]:tile_offsets[b, t + 1]``, ordered by query. Bit ``j`` of a pair's
    mask selects key ``t * tile + j``, and a key selected ``n`` times by one query appears in
    ``n`` pairs.

    The map has a fixed capacity, every top-k slot (``seq_len * topk``) padded to a nonzero
    multiple of 16, so building it reads nothing back to the host and an eager run and a CUDA
    graph capture take the same path. The selections are packed before the sort; unused slots
    carry the sort sentinel and sort after every tile.
    """
    # Pair masks hold one int32 bit per key of a tile, and the sort keys pack the key's
    # position in its tile into log2(tile) bits.
    assert tile <= 32 and tile & (tile - 1) == 0, tile
    batch, seq_len, topk = selected_rows.shape
    device = selected_rows.device
    offsets = counts.cumsum(dim=1) - counts
    # Triton specializes integer arguments that are 1 or divisible by 16; a nonzero multiple of
    # 16 keeps every shape on the same compiled map kernels.
    entries = max(1, triton.cdiv(seq_len * topk, 16)) * 16
    num_tiles = triton.cdiv(sparse_seq_len, tile)
    query_bits = max(1, (seq_len - 1).bit_length())
    # int32 encoded keys halve the sort when they fit.
    narrow = (num_tiles * tile) << query_bits < 2**31 - 1
    sentinel = 2**31 - 1 if narrow else 2**63 - 1
    keys = torch.full(
        (batch, entries), sentinel, device=device, dtype=torch.int32 if narrow else torch.int64
    )
    _tile_map_keys[(batch * seq_len,)](
        selected_rows,
        counts,
        offsets,
        keys,
        entries,
        seq_len,
        query_bits,
        TOPK=topk,
        TILE=tile,
        BLOCK=256,
    )
    grid = (triton.cdiv(entries, 1024), batch)
    sorted_keys = torch.sort(keys, dim=-1).values
    del keys
    starts = torch.empty((batch, entries), device=device, dtype=torch.int32)
    _tile_map_starts[grid](sorted_keys, starts, entries, TILE=tile, SENTINEL=sentinel, BLOCK=1024)
    # One flat scan: a batched last-dim cumsum takes a much slower row-wise kernel.
    pair_ends_dtype = torch.int32 if batch * entries < 2**31 else torch.int64
    pair_ends = torch.cumsum(starts.view(-1), dim=0, dtype=pair_ends_dtype)
    del starts
    pair_queries = torch.empty((batch, entries), device=device, dtype=torch.int32)
    # One int32 bit per key of a tile.
    pair_masks = torch.zeros((batch, entries), device=device, dtype=torch.int32)
    # Unused slots sort after every tile.
    pair_tiles = torch.full((batch, entries), num_tiles, device=device, dtype=torch.int32)
    _tile_map_pairs[grid](
        sorted_keys,
        pair_ends,
        pair_queries,
        pair_masks,
        pair_tiles,
        entries,
        query_bits,
        TILE=tile,
        SENTINEL=sentinel,
        BLOCK=1024,
    )
    tile_offsets = torch.searchsorted(
        pair_tiles,
        torch.arange(num_tiles + 1, device=device, dtype=torch.int32)
        .expand(batch, -1)
        .contiguous(),
    )
    return (
        pair_queries,
        pair_masks,
        tile_offsets.to(torch.int32 if entries < 2**31 else torch.int64),
    )


@triton.jit
def _gather_attn_bwd_dsparse_kv_grouped(
    query_desc,
    sparse_kv_ptr,
    sparse_v_ptr,
    grad_output_desc,
    lse_ptr,
    delta_ptr,
    pair_queries_ptr,
    pair_masks_ptr,
    tile_offsets_ptr,
    grad_sparse_kv_ptr,
    grad_sparse_v_ptr,
    SPARSE_KV_STRIDES,
    SPARSE_V_STRIDES,
    LSE_STRIDES,
    PAIR_STRIDES,
    TILE_OFFSETS_STRIDES,
    GRAD_SPARSE_KV_STRIDES,
    GRAD_SPARSE_V_STRIDES,
    H: tl.constexpr,
    KV_HEADS: tl.constexpr,
    V_IS_K: tl.constexpr,
    S,
    D: tl.constexpr,
    SPARSE_SEQ_LEN,
    SCALE: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    grad_dense_kv_ptr=None,
    grad_dense_v_ptr=None,
    GRAD_DENSE_STRIDES=None,
    ADD_DENSE: tl.constexpr = False,
):
    """Accumulate one key tile's dK and dV over the (query, head) rows that selected it.

    Rows enumerate (pair, query head of this KV head's group); their query and output-gradient
    rows are TMA-gathered from (batch * heads * seq_len, D) views. dK and dV are accumulated
    transposed, (D, TILE), so all four products run as head-dimension-major MMAs. With
    V_IS_K the key and value gradients add into one dKV. ADD_DENSE adds the dense tiles'
    FP32 contributions (``_gather_attn_bwd_dsparse_kv_dense``) before rounding.
    """
    SPARSE_KV_STRIDES = offset_strides(SPARSE_KV_STRIDES, USE_INT64_OFFSETS)
    LSE_STRIDES = offset_strides(LSE_STRIDES, USE_INT64_OFFSETS)
    PAIR_STRIDES = offset_strides(PAIR_STRIDES, USE_INT64_OFFSETS)
    TILE_OFFSETS_STRIDES = offset_strides(TILE_OFFSETS_STRIDES, USE_INT64_OFFSETS)
    GRAD_SPARSE_KV_STRIDES = offset_strides(GRAD_SPARSE_KV_STRIDES, USE_INT64_OFFSETS)
    if not V_IS_K:
        SPARSE_V_STRIDES = offset_strides(SPARSE_V_STRIDES, USE_INT64_OFFSETS)
        GRAD_SPARSE_V_STRIDES = offset_strides(GRAD_SPARSE_V_STRIDES, USE_INT64_OFFSETS)

    tile = tl.program_id(0)
    kv_head = tl.program_id(1)
    batch = tl.program_id(2)
    GROUP: tl.constexpr = H // KV_HEADS

    offsets_n = tl.arange(0, TILE)
    keys = tile * TILE + offsets_n
    offsets_d = tl.arange(0, BLOCK_D)
    dimension_mask = offsets_d < D
    key_mask = (keys < SPARSE_SEQ_LEN)[:, None] & dimension_mask[None, :]
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

    pair_start = tl.load(tile_offsets_ptr + ptr_offset((batch, tile), TILE_OFFSETS_STRIDES))
    pair_end = tl.load(tile_offsets_ptr + ptr_offset((batch, tile + 1), TILE_OFFSETS_STRIDES))
    if USE_INT64_OFFSETS:
        pair_start = pair_start.to(tl.int64)
        pair_end = pair_end.to(tl.int64)
    row_end = pair_end * GROUP
    grad_key = tl.zeros((BLOCK_D, TILE), tl.float32)
    grad_value = tl.zeros((BLOCK_D, TILE), tl.float32)
    offsets_m = tl.arange(0, BLOCK_M)
    for row_start in tl.range(pair_start * GROUP, row_end, BLOCK_M):
        rows = row_start + offsets_m
        row_mask = rows < row_end
        pair = rows // GROUP
        head = kv_head * GROUP + rows % GROUP
        pair_offsets = ptr_offset((batch, pair), PAIR_STRIDES)
        query_position = tl.load(pair_queries_ptr + pair_offsets, mask=row_mask, other=0)
        key_bits = tl.load(pair_masks_ptr + pair_offsets, mask=row_mask, other=0)
        selected = ((key_bits[:, None] >> offsets_n[None, :]) & 1) != 0

        # Masked rows gather past the last row, which TMA zero-fills.
        gather_rows = tl.where(
            row_mask, ((batch * H + head) * S + query_position).to(tl.int32), 2**31 - 1
        )
        query = query_desc.gather(gather_rows, 0)
        grad_output = grad_output_desc.gather(gather_rows, 0)
        statistic_offsets = ptr_offset((batch, head, query_position), LSE_STRIDES)
        lse = tl.load(lse_ptr + statistic_offsets, mask=row_mask, other=0.0)
        delta = tl.load(delta_ptr + statistic_offsets, mask=row_mask, other=0.0)

        scores = tl.dot(query, tl.trans(sparse_keys)) * SCALE
        probabilities = tl.where(selected, tl.exp(scores - lse[:, None]), 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(sparse_v))
        # Mask unselected keys too: a NaN value or a nonfinite delta must not reach them.
        grad_scores = tl.where(
            selected, probabilities * (grad_probabilities - delta[:, None]), 0.0
        )
        grad_value = tl.dot(
            tl.trans(grad_output), probabilities.to(grad_output.dtype), acc=grad_value
        )
        grad_key = tl.dot(tl.trans(query), grad_scores.to(query.dtype), acc=grad_key)

    store_mask = dimension_mask[:, None] & (keys < SPARSE_SEQ_LEN)[None, :]
    store_offsets = (batch, kv_head, keys[None, :], offsets_d[:, None])
    grad_key *= SCALE
    if V_IS_K:
        grad_key += grad_value
    if ADD_DENSE:
        GRAD_DENSE_STRIDES = offset_strides(GRAD_DENSE_STRIDES, USE_INT64_OFFSETS)
        dense_offsets = ptr_offset(store_offsets, GRAD_DENSE_STRIDES)
        grad_key += tl.load(grad_dense_kv_ptr + dense_offsets, mask=store_mask, other=0.0)
        if not V_IS_K:
            grad_value += tl.load(grad_dense_v_ptr + dense_offsets, mask=store_mask, other=0.0)
    tl.store(
        grad_sparse_kv_ptr + ptr_offset(store_offsets, GRAD_SPARSE_KV_STRIDES),
        grad_key,
        mask=store_mask,
    )
    if not V_IS_K:
        tl.store(
            grad_sparse_v_ptr + ptr_offset(store_offsets, GRAD_SPARSE_V_STRIDES),
            grad_value,
            mask=store_mask,
        )
