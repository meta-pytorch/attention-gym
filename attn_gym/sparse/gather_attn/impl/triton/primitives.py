"""Triton primitives shared by gather-attention kernel schedules."""

from collections.abc import Sequence
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import _document_ids, ptr_offset, requires_int64_offsets


def can_use_shared_kv_schedule(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor | None,
    sliding_window_size: int,
    query_rope: torch.Tensor | None = None,
    sparse_rope: torch.Tensor | None = None,
    local_rope: torch.Tensor | None = None,
) -> bool:
    """Check the shared Blackwell forward/backward schedule's layout and shape limits.

    The latent KV is both key and value. Optional key-only RoPE tensors add a second logit
    dot; with RoPE, ``sparse_rope`` (and ``local_rope`` under a window) must be given.
    """
    _, heads, _, head_dim = query.shape
    rope_tensors = (sparse_rope, local_rope) if sliding_window_size else (sparse_rope,)
    rope_ok = query_rope is None or (
        query_rope.shape[-1] in (16, 32, 64, 128)
        and query_rope.dtype == torch.bfloat16
        and all(
            rope is not None
            and rope.dtype == torch.bfloat16
            and rope.shape[-1] == query_rope.shape[-1]
            and rope.stride(1) == 0
            for rope in rope_tensors
        )
    )
    return (
        torch.cuda.get_device_capability(query.device)[0] >= 10
        and query.dtype == torch.bfloat16
        and sparse_kv.stride(1) == 0
        and query.stride(-1) == 1
        and sparse_kv.stride(-1) == 1
        # Without a window the kernels never read local KV, so it may be absent.
        and (sliding_window_size == 0 or (local_kv.stride(1) == 0 and local_kv.stride(-1) == 1))
        and 16 <= heads <= 128
        and head_dim % 16 == 0
        and (head_dim <= 128 or head_dim == 512)
        and rope_ok
        and sliding_window_size <= 2048
    )


# Dense tiles fill this many tensor-core rows with (query, head) pairs of one KV head.
DENSE_TILE_ROWS = 128
# Walking a dense tile's span beats gathering up to about this many keys per selection on
# B200, so the selection bitmaps cover this many top-k widths.
DENSE_SPAN_RATIO = 4


class TileConfig(NamedTuple):
    """Launch parameters of a fixed-config (not autotuned) kernel."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    # Shared memory in BLOCK_D-wide rows: the largest ``metadata.shared / (BLOCK_D *
    # element_size)`` measured for SM86 at D in {64, 128, 256} with Triton 3.8.
    shared_rows: int = 0
    # Tensor-memory columns: the largest ``metadata.tmem_size`` measured on SM103 (GB300) for
    # BF16 and FP32 at D in {64, 128, 256} with Triton 3.8. Only 64-row tiles use tcgen05 MMAs.
    tmem_cols: int = 0


# Tensor-memory columns one CTA may allocate on datacenter Blackwell; Triton rejects kernels
# above it when loading them.
TENSOR_MEMORY_COLUMNS = 512


def select_tiles(
    candidates: Sequence[TileConfig],
    block_d: int,
    element_size: int,
    device: torch.device,
) -> TileConfig:
    """Return the first candidate whose estimated shared and tensor memory fit ``device``.

    Candidates are ordered by preference; the last one is the fallback for tiny budgets.
    """
    properties = torch.cuda.get_device_properties(device)
    shared_budget = properties.shared_memory_per_block_optin
    # SM10x/SM11x accumulate tcgen05 MMAs in tensor memory; other GPUs have none.
    tmem_budget = TENSOR_MEMORY_COLUMNS if properties.major in (10, 11) else float("inf")
    return next(
        (
            c
            for c in candidates
            if c.shared_rows * block_d * element_size <= shared_budget
            and c.tmem_cols <= tmem_budget
        ),
        candidates[-1],
    )


def optional_strides(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
    """Strides of an optional kernel operand; None compiles out its constexpr-guarded uses."""
    return None if tensor is None else tensor.stride()


def tuning_key(tensor: str, *names: str):
    """Key a ``TritonTuner`` by the dtype of argument ``tensor`` and the arguments ``names``.

    Unlike ``triton.autotune``'s implicit dtype key, this also sees through TMA descriptors.
    """

    def key(args):
        value = args[tensor]
        dtype = (value.base if isinstance(value, TensorDescriptor) else value).dtype
        return (dtype, *(args[name] for name in names))

    return key


def prune_wide_backward_configs(configs, args):
    """Bound full-width D=512 gradient tiles without changing smaller-head tuning."""
    if args["D"] != 512:
        return configs
    return [
        config
        for config in configs
        if config.num_warps == 4
        and config.num_stages == 1
        and all(size <= 16 for size in config.kwargs.values())
    ]


@triton.jit
def offset_strides(strides, USE_INT64_OFFSETS: tl.constexpr):
    """Select address width before multiplying runtime strides and indices."""
    result = ()
    for axis in tl.static_range(len(strides)):
        result += (tl.cast(strides[axis], tl.int64 if USE_INT64_OFFSETS else tl.int32),)
    return result


@triton.jit
def load_bhsd(
    tensor_ptr,
    strides,
    batch,
    head,
    positions,
    offsets_d,
    mask,
):
    """Load a tile from gather attention's BHSD tensors."""
    return tl.load(
        tensor_ptr
        + ptr_offset(
            (batch, head, positions[:, None], offsets_d[None, :]),
            strides,
        ),
        mask=mask,
        other=0.0,
    )


@triton.jit
def load_document_bounds(
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    positions,
    num_documents,
    sequence_length,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Find query and candidate bounds; inactive queries have no sparse candidates."""
    document = _document_ids(cu_seqlens_ptr, positions, num_documents, USE_INT64_OFFSETS)
    offset = document.to(tl.int64) if USE_INT64_OFFSETS else document
    # The right-sided search skips empty documents and returns N for tail/padded positions.
    start = tl.load(cu_seqlens_ptr + offset)
    end = tl.load(
        cu_seqlens_ptr + offset + 1,
        mask=document < num_documents,
        other=sequence_length,
    )
    candidate_start = tl.load(cu_seqlens_k_ptr + offset)
    candidate_end = tl.load(cu_seqlens_k_ptr + tl.minimum(offset + 1, num_documents))
    return start, end, candidate_start, candidate_end


@triton.jit
def store_bhsd(
    tensor_ptr,
    value,
    strides,
    batch,
    head,
    positions,
    offsets_d,
    mask,
):
    """Store a tile to gather attention's BHSD tensors."""
    tl.store(
        tensor_ptr
        + ptr_offset(
            (batch, head, positions[:, None], offsets_d[None, :]),
            strides,
        ),
        value,
        mask=mask,
    )


@triton.jit
def causal_window_mask(
    query_positions,
    key_positions,
    query_mask,
    key_mask,
    window: tl.constexpr,
):
    """Mask valid gather-attention pairs to the causal local window."""
    return (
        query_mask[:, None]
        & key_mask[None, :]
        & (key_positions[None, :] <= query_positions[:, None])
        & (key_positions[None, :] >= query_positions[:, None] - window + 1)
    )


@triton.jit
def online_softmax_update(accumulator, running_max, running_sum, logits, values):
    """Merge one gather-attention tile into FP32 online-softmax state."""
    tile_max = tl.max(logits, axis=1)
    new_max = tl.maximum(running_max, tile_max)
    # An empty prefix with no sink has max=-inf. Keep that state for later tiles,
    # but avoid -inf - -inf while computing this tile's zero contribution.
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    alpha = tl.exp(running_max - safe_max)
    probabilities = tl.exp(logits - safe_max[:, None])
    accumulator *= alpha[:, None]
    accumulator += tl.dot(probabilities.to(values.dtype), values, input_precision="tf32x3")
    running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
    return accumulator, new_max, running_sum


# Sizes and the batch stride of ``kv_indices`` only feed bounds, offsets, and loop trip counts.
# A constexpr TOPK measured no faster on B200 (QSA top-k 2051), so top-k widths share a kernel.
@triton.jit(
    do_not_specialize=["num_documents", "KV_INDICES_BATCH_STRIDE", "S", "SPARSE_SEQ_LEN", "TOPK"]
)
def _resolve_selected_rows(
    kv_indices_ptr,
    rows_ptr,
    counts_ptr,
    masks_ptr,
    spans_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    num_documents,
    KV_INDICES_BATCH_STRIDE,
    KV_INDICES_STRIDES,
    S,
    SPARSE_SEQ_LEN,
    TOPK,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    MASK_WORDS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compact one query's valid selections, as global sparse-pool rows, to its row's front.

    With MASK_WORDS, also set the selections' bits in the query's bitmap over its first
    ``32 * MASK_WORDS`` document-local positions and store its ``[lo, hi]`` selection span.
    ``hi`` is -1 for no selections and -2 when the bitmap cannot stand in for the row: a
    repeated selection or one past the bitmap.
    """
    KV_INDICES_STRIDES = offset_strides(
        (KV_INDICES_BATCH_STRIDE, KV_INDICES_STRIDES[0], KV_INDICES_STRIDES[1]),
        USE_INT64_OFFSETS,
    )
    row = tl.program_id(0).to(tl.int64)
    batch = row // S
    query_position = row % S
    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        _, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, query_position, num_documents, S, True
        )
    count = 0
    lo = MASK_WORDS * 32
    hi = -1
    for start in range(0, TOPK, BLOCK_K):
        slots = start + tl.arange(0, BLOCK_K)
        indices = tl.load(
            kv_indices_ptr + ptr_offset((batch, query_position, slots), KV_INDICES_STRIDES),
            mask=slots < TOPK,
            other=-1,
        )
        valid = (indices >= 0) & (indices < candidate_end - candidate_start)
        # Mask local coordinates before adding the document base, including int64 sentinels.
        local = tl.where(valid, indices, 0)
        rows = local + candidate_start
        positions = count + tl.cumsum(valid.to(tl.int32), axis=0) - 1
        tl.store(rows_ptr + row * TOPK + positions, rows.to(rows_ptr.dtype.element_ty), mask=valid)
        count += tl.sum(valid.to(tl.int32), axis=0)
        if MASK_WORDS:
            inside = valid & (local < MASK_WORDS * 32)
            bit = local.to(tl.int32) % 32
            tl.atomic_or(
                masks_ptr + row * MASK_WORDS + local // 32,
                tl.full(bit.shape, 1, tl.int32) << bit,
                mask=inside,
                sem="relaxed",
            )
            lo = tl.minimum(lo, tl.min(tl.where(valid, local, MASK_WORDS * 32)).to(tl.int32))
            hi = tl.maximum(hi, tl.max(tl.where(valid, local, -1)).to(tl.int32))
    tl.store(counts_ptr + row, count)
    if MASK_WORDS:
        # Every bit is set once, so fewer set bits than selections means a repeat or a
        # selection past the bitmap.
        tl.debug_barrier()
        words = tl.arange(0, triton.next_power_of_2(MASK_WORDS))
        bits = tl.load(
            masks_ptr + row * MASK_WORDS + words,
            mask=words < MASK_WORDS,
            other=0,
            cache_modifier=".cg",
        )
        exact = tl.sum(libdevice.popc(bits), axis=0) == count
        tl.store(spans_ptr + row * 2, lo)
        tl.store(spans_ptr + row * 2 + 1, tl.where(exact, hi, -2))


def resolve_selected_rows(
    kv_indices: torch.Tensor,
    sparse_seq_len: int,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    *,
    mask_words: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Resolve document-local selections to global pool rows, compacted per query.

    Returns ``(rows, counts, masks, spans)``. ``rows`` [B, S, topk] (int32, or int64 for pools
    of 2**31 rows or more) holds each query's valid selections in their original order
    (repeats included) as global pool rows in its first ``counts[b, s]`` entries; the rest are
    unspecified and never read. The grouped kernels then skip per-gather document checks and
    loop over full tiles of rows without masks, masking only the last, partial tile.

    With ``mask_words``, ``masks`` [B, S, mask_words] int32 holds each query's selection
    bitmap over document-local positions and ``spans`` [B, S, 2] its selection span (see
    ``_resolve_selected_rows``); both feed ``plan_dense_tiles``. Otherwise they are None.
    """
    batch, seq_len, topk = kv_indices.shape
    device = kv_indices.device
    # int32 rows, unless the pool is too long for them.
    rows = torch.empty(
        (batch, seq_len, topk),
        device=device,
        dtype=torch.int32 if sparse_seq_len < 2**31 else torch.int64,
    )
    # The kernel writes every count and span; only the bitmaps accumulate.
    counts = torch.empty((batch, seq_len), device=device, dtype=torch.int32)
    if topk == 0:
        # dense_tile_shape plans no dense tiles, hence no bitmaps, without selections.
        return rows, counts.zero_(), None, None
    masks = spans = None
    if mask_words:
        masks = torch.zeros((batch, seq_len, mask_words), device=device, dtype=torch.int32)
        spans = torch.empty((batch, seq_len, 2), device=device, dtype=torch.int32)
    has_cu_seqlens = cu_seqlens is not None
    _resolve_selected_rows[(batch * seq_len,)](
        kv_indices,
        rows,
        counts,
        masks,
        spans,
        cu_seqlens,
        cu_seqlens_k,
        cu_seqlens.numel() - 1 if has_cu_seqlens else 0,
        KV_INDICES_BATCH_STRIDE=kv_indices.stride(0),
        KV_INDICES_STRIDES=kv_indices.stride()[1:],
        S=seq_len,
        SPARSE_SEQ_LEN=sparse_seq_len,
        TOPK=topk,
        HAS_CU_SEQLENS=has_cu_seqlens,
        USE_INT64_OFFSETS=requires_int64_offsets(kv_indices),
        MASK_WORDS=mask_words,
        BLOCK_K=256,
        num_warps=4,
    )
    return rows, counts, masks, spans


def dense_tile_shape(
    heads: int, kv_heads: int, topk: int, sliding_window_size: int
) -> tuple[int, int] | None:
    """Return ``(tile_queries, mask_words)`` for dense tiles, or None.

    A dense tile fills ``DENSE_TILE_ROWS`` tensor-core rows with (query, head) pairs of one KV
    head's group and walks every key in its span, so bitmaps cover ``DENSE_SPAN_RATIO``
    top-k widths. A sliding window is a second, overlapping selection set the bitmap cannot
    express, so it keeps every query on the gather kernels.
    """
    group = heads // kv_heads
    if sliding_window_size or topk == 0 or group > DENSE_TILE_ROWS:
        return None
    # Whole 128-bit (4-word) rows keep every BLOCK_N-key read inside its query's bitmap.
    return DENSE_TILE_ROWS // group, triton.cdiv(DENSE_SPAN_RATIO * topk, 128) * 4


@triton.jit
def in_dense_tile(tile_spans_ptr, batch, query_position, S, DENSE_TILE_QUERIES: tl.constexpr):
    """Whether ``plan_dense_tiles`` gave this query's tile to the dense kernel; always False
    without dense tiles (DENSE_TILE_QUERIES == 0)."""
    dense = False
    if DENSE_TILE_QUERIES:
        num_tiles = (S + DENSE_TILE_QUERIES - 1) // DENSE_TILE_QUERIES
        tile_row = batch.to(tl.int64) * num_tiles + query_position // DENSE_TILE_QUERIES
        dense = tl.load(tile_spans_ptr + tile_row * 2 + 1) >= 0
    return dense


@triton.jit
def head_block_heads(head_block, H: tl.constexpr, KV_HEADS: tl.constexpr, BLOCK_H: tl.constexpr):
    """``(kv_head, heads, head_mask)`` for one index of a head-block grid axis.

    With KV_HEADS == 1 the blocks tile all H query heads, which read KV head 0. Otherwise they
    enumerate (KV head, block of its H // KV_HEADS query heads), so each block reads one KV
    head and each gathered K/V row serves the whole block.
    """
    if KV_HEADS == 1:
        kv_head = 0
        heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        head_mask = heads < H
    else:
        group = H // KV_HEADS
        blocks_per_group = (group + BLOCK_H - 1) // BLOCK_H
        kv_head = head_block // blocks_per_group
        group_offsets = (head_block % blocks_per_group) * BLOCK_H + tl.arange(0, BLOCK_H)
        heads = kv_head * group + group_offsets
        head_mask = group_offsets < group
    return kv_head, heads, head_mask


@triton.jit
def dense_tile_rows(
    tile,
    kv_head,
    S,
    GROUP: tl.constexpr,
    TILE_QUERIES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """``(positions, heads, row_mask)`` of a dense tile's BLOCK_M rows.

    A dense tile's rows are the (query, head) pairs of its TILE_QUERIES consecutive queries
    and the GROUP query heads reading ``kv_head``, query-major.
    """
    rows = tl.arange(0, BLOCK_M)
    positions = tile * TILE_QUERIES + rows // GROUP
    if USE_INT64_OFFSETS:
        positions = positions.to(tl.int64)
    heads = kv_head * GROUP + rows % GROUP
    return positions, heads, (rows < TILE_QUERIES * GROUP) & (positions < S)


@triton.jit
def selection_bits(
    mask_rows,
    row_mask,
    local_start,
    MASK_WORDS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """[BLOCK_M, BLOCK_N] selection flags for keys ``local_start + j`` of each row's query.

    ``mask_rows`` points at each row's ``resolve_selected_rows`` bitmap of MASK_WORDS words.
    ``local_start`` is a multiple of BLOCK_N, so one bitmap word covers 32 key columns, and a
    BLOCK_N dividing the bitmap keeps every read inside the row.
    """
    tl.static_assert(BLOCK_N % 32 == 0 and MASK_WORDS * 32 % BLOCK_N == 0)
    WORDS: tl.constexpr = BLOCK_N // 32
    words = tl.load(
        mask_rows[:, None] + local_start // 32 + tl.arange(0, WORDS)[None, :],
        mask=row_mask[:, None],
        other=0,
    )
    words = tl.reshape(
        tl.broadcast_to(words[:, :, None], (BLOCK_M, WORDS, 32)), (BLOCK_M, BLOCK_N)
    )
    return ((words >> (tl.arange(0, BLOCK_N) % 32)[None, :]) & 1) != 0


@triton.jit(do_not_specialize=["num_documents", "S", "NUM_TILES"])
def _plan_dense_tiles(
    counts_ptr,
    spans_ptr,
    tile_spans_ptr,
    cu_seqlens_ptr,
    num_documents,
    S,
    NUM_TILES,
    HAS_CU_SEQLENS: tl.constexpr,
    TILE_QUERIES: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    SPAN_RATIO: tl.constexpr,
):
    """Choose whether one tile of consecutive queries walks its selection span densely.

    A dense tile has every query in one document, bitmap-exact selections, and a span of at
    most SPAN_RATIO keys per selection, short enough that walking every key in it beats
    gathering them.
    """
    tile = tl.program_id(0)
    batch = tl.program_id(1).to(tl.int64)
    offsets = tl.arange(0, BLOCK_Q)
    queries = tile * TILE_QUERIES + offsets
    active = (offsets < TILE_QUERIES) & (queries < S)
    rows = batch * S + queries
    counts = tl.load(counts_ptr + rows, mask=active, other=0)
    lo = tl.load(spans_ptr + rows * 2, mask=active, other=2**30)
    hi = tl.load(spans_ptr + rows * 2 + 1, mask=active, other=-1)
    exact = tl.min(tl.where(active, hi, 0), axis=0) >= -1
    selected = tl.sum(counts, axis=0)
    span_lo = tl.min(tl.where(counts > 0, lo, 2**30), axis=0)
    span_hi = tl.max(tl.where(counts > 0, hi, -1), axis=0)
    num_queries = tl.sum(active.to(tl.int32), axis=0)
    dense = (
        exact & (selected > 0) & ((span_hi - span_lo + 1) * num_queries <= SPAN_RATIO * selected)
    )
    if HAS_CU_SEQLENS:
        first = tile * TILE_QUERIES
        last = tl.minimum(first + TILE_QUERIES, S) - 1
        document = _document_ids(cu_seqlens_ptr, first, num_documents, False)
        dense &= document == _document_ids(cu_seqlens_ptr, last, num_documents, False)
    tile_row = batch * NUM_TILES + tile
    tl.store(tile_spans_ptr + tile_row * 2, span_lo)
    tl.store(tile_spans_ptr + tile_row * 2 + 1, tl.where(dense, span_hi, -1))


def plan_dense_tiles(
    counts: torch.Tensor,
    spans: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    tile_queries: int,
) -> torch.Tensor:
    """Return ``tile_spans`` [B, num_tiles, 2] for ``tile_queries``-query tiles.

    ``tile_spans`` holds int32 document-local ``[lo, hi]`` spans; ``hi`` is -1 for tiles left
    to the gather kernels.
    """
    batch, seq_len = counts.shape
    num_tiles = triton.cdiv(seq_len, tile_queries)
    tile_spans = torch.empty((batch, num_tiles, 2), device=counts.device, dtype=torch.int32)
    _plan_dense_tiles[(num_tiles, batch)](
        counts,
        spans,
        tile_spans,
        cu_seqlens,
        cu_seqlens.numel() - 1 if cu_seqlens is not None else 0,
        seq_len,
        num_tiles,
        HAS_CU_SEQLENS=cu_seqlens is not None,
        TILE_QUERIES=tile_queries,
        BLOCK_Q=triton.next_power_of_2(tile_queries),
        SPAN_RATIO=DENSE_SPAN_RATIO,
        num_warps=1,
    )
    return tile_spans
