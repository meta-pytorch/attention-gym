"""Paged scoring parallel across candidates, followed by exact radix selection.

Triton covers general layouts on SM80+. The SM100-family route can use a Warpdrive
scorer for compact decode inputs. Both scorers write the same bounded FP32 slab;
the radix selector returns sequence-local IDs without materializing a dense KV pool.
"""

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import (
    _document_ids,
    can_use_tma,
    offset_strides,
    ptr_offset,
    requires_int64_offsets,
)


@triton.jit
def paged_candidate_bounds(
    CuQ,
    Lengths,
    Positions,
    row,
    sequences,
    RATIO: tl.constexpr,
    EXPLICIT: tl.constexpr,
    WIDE: tl.constexpr,
    Bounds=None,
):
    """Find the query's sequence and completed prefix; record [0, visible) for CuTe."""
    sequence = _document_ids(CuQ, row, sequences, WIDE)
    if WIDE:
        sequence = sequence.to(tl.int64)
    length = tl.load(Lengths + sequence, sequence < sequences, 0)
    end = tl.load(CuQ + tl.minimum(sequence + 1, sequences))
    position = length - end + row
    if EXPLICIT:
        # Clamp before adding one: an explicit INT32_MAX position must not
        # wrap around while applying the visible cached-prefix cap.
        position = tl.minimum(tl.load(Positions + row), length - 1)
    visible = tl.minimum(tl.maximum((position + 1) // RATIO, 0), length // RATIO)
    visible = tl.where(sequence < sequences, visible, 0)
    if Bounds is not None:  # noqa: SIM102 — discard None pointer arithmetic at compile time
        if tl.program_id(1) == 0:
            tl.store(Bounds + ptr_offset((row, 0), (2, 1)), 0)
            tl.store(Bounds + ptr_offset((row, 1), (2, 1)), visible)
    return sequence, visible


@triton.jit
def weighted_relu_scores(keys, query, weights, H: tl.constexpr, D: tl.constexpr):
    """Reduce the per-head ReLU dot products to one scaled score per candidate."""
    dots = tl.dot(keys, tl.trans(query))
    return tl.sum(tl.maximum(dots, 0.0) * weights[None, :], 1) * (H * D) ** -0.5


# Triton 3.9 specializes tuple members even when the tuple is in do_not_specialize.
# Unpack caller stride tuples only at the JIT boundary; regroup them for all address math.
@triton.jit(
    do_not_specialize=[
        "sequences",
        "capacity",
        "row_start",
        "stride_q_token",
        "stride_q_head",
        "stride_q_dim",
        "stride_k_page",
        "stride_k_slot",
        "stride_k_dim",
        "stride_w_token",
        "stride_w_head",
        "stride_table_seq",
        "stride_table_page",
    ]
)
def paged_score_kernel(
    Q,
    K,
    W,
    Table,
    CuQ,
    Lengths,
    Positions,
    Scores,
    Bounds,
    sequences,
    capacity,
    row_start,
    stride_q_token,
    stride_q_head,
    stride_q_dim,
    stride_k_page,
    stride_k_slot,
    stride_k_dim,
    stride_w_token,
    stride_w_head,
    stride_table_seq,
    stride_table_page,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    EXPLICIT: tl.constexpr,
    WIDE: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    Q_UNIT_D: tl.constexpr,
    K_UNIT_D: tl.constexpr,
):
    """Gather a tile of candidate keys from arbitrary strided pages and score it."""
    local = tl.program_id(0).to(tl.int64 if WIDE else tl.int32)
    tile = tl.program_id(1)
    row = row_start + local
    sequence, visible = paged_candidate_bounds(
        CuQ, Lengths, Positions, row, sequences, RATIO, EXPLICIT, WIDE, Bounds
    )
    if visible <= TOPK or tile * BN >= visible:
        return
    candidates = tile * BN + tl.arange(0, BN)
    dims = tl.arange(0, BD)
    heads = tl.arange(0, BH)
    table_strides = offset_strides((stride_table_seq, stride_table_page), WIDE)
    pages = tl.load(
        Table + ptr_offset((sequence, candidates // PAGE), table_strides), candidates < visible, 0
    )
    k_strides = offset_strides(
        (stride_k_page, stride_k_slot, 1 if K_UNIT_D else stride_k_dim), WIDE
    )
    q_strides = offset_strides(
        (stride_q_token, stride_q_head, 1 if Q_UNIT_D else stride_q_dim), WIDE
    )
    w_strides = offset_strides((stride_w_token, stride_w_head), WIDE)
    keys = tl.load(
        K + ptr_offset((pages[:, None], (candidates % PAGE)[:, None], dims[None, :]), k_strides),
        (candidates[:, None] < visible) & (dims[None, :] < D),
        0,
    )
    query = tl.load(
        Q + ptr_offset((row, heads[:, None], dims[None, :]), q_strides),
        (heads[:, None] < H) & (dims[None, :] < D),
        0,
    )
    weights = tl.load(W + ptr_offset((row, heads), w_strides), heads < H, 0).to(tl.float32)
    scores = weighted_relu_scores(keys, query, weights, H, D)
    tl.store(Scores + ptr_offset((local, candidates), (capacity, 1)), scores, candidates < visible)


@triton.jit(
    do_not_specialize=[
        "sequences",
        "capacity",
        "row_start",
        "stride_w_token",
        "stride_w_head",
        "stride_table_seq",
        "stride_table_page",
    ]
)
def paged_score_tma_kernel(
    Q,
    K,
    W,
    Table,
    CuQ,
    Lengths,
    Positions,
    Scores,
    Bounds,
    sequences,
    capacity,
    row_start,
    stride_w_token,
    stride_w_head,
    stride_table_seq,
    stride_table_page,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    EXPLICIT: tl.constexpr,
    WIDE: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    TILES: tl.constexpr,
):
    """Score page-aligned TMA tiles, reusing the query across tiles in this CTA."""
    local = tl.program_id(0).to(tl.int64 if WIDE else tl.int32)
    row = row_start + local
    sequence, visible = paged_candidate_bounds(
        CuQ, Lengths, Positions, row, sequences, RATIO, EXPLICIT, WIDE, Bounds
    )
    start = tl.program_id(1) * BN * TILES
    if visible <= TOPK or start >= visible:
        return
    heads = tl.arange(0, BH)
    w_strides = offset_strides((stride_w_token, stride_w_head), WIDE)
    table_strides = offset_strides((stride_table_seq, stride_table_page), WIDE)
    # Descriptor coordinates are int32; Q and K retain their full byte strides in the descriptors.
    query = Q.load([row.to(tl.int32), 0, 0]).reshape(BH, BD)
    weights = tl.load(W + ptr_offset((row, heads), w_strides), heads < H, 0).to(tl.float32)
    for begin in tl.range(start, tl.minimum(start + BN * TILES, visible), BN):
        page = tl.load(Table + ptr_offset((sequence, begin // PAGE), table_strides))
        keys = K.load([page, (begin % PAGE).to(tl.int32), 0]).reshape(BN, BD)
        scores = weighted_relu_scores(keys, query, weights, H, D)
        candidates = begin + tl.arange(0, BN)
        tl.store(
            Scores + ptr_offset((local, candidates), (capacity, 1)), scores, candidates < visible
        )


@triton.jit
def float_radix_keys(scores):
    """Map FP32 scores to unsigned keys in score order, treating both zeros as equal."""
    bits = tl.where(scores == 0.0, 0.0, scores).to(tl.uint32, bitcast=True)
    return bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)


@triton.jit(do_not_specialize=["sequences", "capacity", "row_start"])
def paged_radix_topk_kernel(
    Scores,
    Out,
    CuQ,
    Lengths,
    Positions,
    sequences,
    capacity,
    row_start,
    TOPK: tl.constexpr,
    RATIO: tl.constexpr,
    EXPLICIT: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Select one row's Top-K IDs by radix threshold, keeping lower IDs at the boundary."""
    local = tl.program_id(0).to(tl.int64 if WIDE else tl.int32)
    row = row_start + local
    _, visible = paged_candidate_bounds(
        CuQ, Lengths, Positions, row, sequences, RATIO, EXPLICIT, WIDE
    )
    offsets = tl.arange(0, BLOCK)
    # Slab scores and the output are allocated contiguous, unlike caller-owned inputs.
    row_scores = Scores + ptr_offset((local, offsets), (capacity, 1))
    row_out = Out + row * TOPK
    if visible <= TOPK:
        for start in range(0, TOPK, BLOCK):
            rank = start + offsets
            tl.store(row_out + rank, tl.where(rank < visible, rank, -1), rank < TOPK)
        return
    prefix = tl.full((), 0, tl.uint32)
    need = tl.full((), TOPK, tl.int32)
    # Binary radix uses warp reductions instead of contended shared-memory
    # histograms (the high exponent bytes often occupy only one or two bins).
    for bit in range(31, -1, -1):
        trial = prefix | (tl.full((), 1, tl.uint32) << bit)
        above = tl.full((), 0, tl.int32)
        for start in range(0, visible, BLOCK):
            indices = start + offsets
            scores = tl.load(row_scores + start, indices < visible, 0)
            keys = float_radix_keys(scores)
            above += tl.sum(((indices < visible) & (keys >= trial)).to(tl.int32), 0)
        prefix = tl.where(above >= TOPK, trial, prefix)
    strict = tl.full((), 0, tl.int32)
    for start in range(0, visible, BLOCK):
        indices = start + offsets
        scores = tl.load(row_scores + start, indices < visible, 0)
        strict += tl.sum(
            ((indices < visible) & (float_radix_keys(scores) > prefix)).to(tl.int32), 0
        )
    need -= strict
    # need is the number of equal boundary scores to retain. Prefix scans break
    # those ties toward lower IDs, and write selected IDs in sequence order.
    written = tl.full((), 0, tl.int32)
    ties_seen = tl.full((), 0, tl.int32)
    for start in range(0, visible, BLOCK):
        indices = start + offsets
        scores = tl.load(row_scores + start, indices < visible, 0)
        keys = float_radix_keys(scores)
        ties = (indices < visible) & (keys == prefix)
        tie_rank = tl.cumsum(ties.to(tl.int32))
        selected = (indices < visible) & (
            (keys > prefix) | (ties & (ties_seen + tie_rank <= need))
        )
        rank = written + tl.cumsum(selected.to(tl.int32)) - 1
        tl.store(row_out + rank, indices.to(tl.int32), selected & (rank < TOPK))
        written += tl.sum(selected.to(tl.int32), 0)
        ties_seen += tl.sum(ties.to(tl.int32), 0)


def compact_warpdrive_layout(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    query_positions: torch.Tensor | None,
) -> bool:
    """Shared compact H4/D128/P64 payload ABI of the paged Warpdrive scorers."""
    return (
        query.shape[1:] == (4, 128)
        and key_cache.shape[1] == 64
        # TVM-FFI checks even the stride of a singleton dimension. is_contiguous()
        # alone would admit an unrepresentable stride for a one-query decode.
        and query.stride() == (4 * 128, 128, 1)
        and key_cache.stride() == (64 * 128, 128, 1)
        and weights.stride() == (4, 1)
        and block_table.stride() == (max(1, block_table.shape[1]), 1)
        and seqused_k.stride(0) == 1
        and (query_positions is None or query_positions.stride(0) == 1)
        # Leave int32 headroom when rounding a search interval to 32 warp probes.
        and seqused_k.numel() <= 2**31 - 32
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
    )


def can_use_warpdrive_score(
    query, key_cache, weights, block_table, seqused_k, query_positions, score_wide: bool
) -> bool:
    # Below this grid size, the FP16/BF16 Triton dot avoids TMEM/pipeline setup.
    return (
        not score_wide
        and query.shape[0] * block_table.shape[1] * key_cache.shape[1] > 32768
        and compact_warpdrive_layout(
            query, key_cache, weights, block_table, seqused_k, query_positions
        )
    )


def launch(
    query,
    key_cache,
    weights,
    block_table,
    cu_seqlens_q,
    seqused_k,
    topk,
    compress_ratio,
    query_positions,
    backend,
    q_scale=None,
    k_scale=None,
):
    """Score paged keys in slabs and return sequence-local Top-K IDs with -1 padding."""
    capability = torch.cuda.get_device_capability(query.device)
    if torch.version.hip or capability[0] < 8:
        raise ValueError("paged_lightning_indexer requires an SM80+ NVIDIA GPU.")
    mxfp8 = query.dtype == torch.float8_e4m3fn
    if mxfp8 or q_scale is not None or k_scale is not None:
        from ..validation import validate_precision

        validate_precision(query, key_cache, weights, q_scale, k_scale)
    if mxfp8:
        if backend != "cute" or capability not in ((10, 0), (10, 3)):
            raise ValueError(
                "fused paged MXFP8 scoring requires the CuTe backend on SM100 or SM103."
            )
        if not compact_warpdrive_layout(
            query, key_cache, weights, block_table, seqused_k, query_positions
        ):
            raise ValueError(
                "fused paged MXFP8 requires compact H4/D128 inputs, aligned Q/K, and 64-candidate pages."
            )
        if (
            q_scale.stride() != (16, 4, 1)
            or k_scale.stride() != (256, 4, 1)
            or (q_scale.data_ptr() | k_scale.data_ptr()) % 4
        ):
            raise ValueError("fused paged MXFP8 requires compact, four-byte-aligned E8M0 scales.")
    elif query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            "the fused paged indexer requires FP16, BF16 or explicit-scale MXFP8 inputs."
        )
    tokens, heads, dim = query.shape
    if heads > 128 or dim > 256:
        raise ValueError("the fused paged indexer requires heads <= 128 and head_dim <= 256.")
    output = query.new_empty((tokens, topk), dtype=torch.int32)
    if topk == 0 or tokens == 0:
        return output
    capacity = block_table.shape[1] * key_cache.shape[1]
    if capacity == 0 or seqused_k.numel() == 0 or key_cache.shape[0] == 0:
        return output.fill_(-1)
    if capacity > 2**22:
        raise ValueError("candidate capacity must be <= 2**22 (bounded score workspace).")
    # Raw lengths are int32: a larger ratio can never form even one completed block.
    if compress_ratio > 2**31 - 1:
        return output.fill_(-1)
    # Pair-shaped scratch shares the existing cached CuTe radix kernel's ABI.
    pairs = min(triton.cdiv(tokens, 2), (32 * 1024 * 1024) // (8 * capacity))
    slab = 2 * pairs
    scores = query.new_empty(
        (pairs, 2, capacity) if capacity > topk else (0,), dtype=torch.float32
    )
    use_cute = backend == "cute" and capacity > topk
    bounds = query.new_empty((tokens, 2), dtype=torch.int32) if use_cute else None
    # The absolute row index can exceed int32 even when query and weights are broadcast.
    shared_wide = tokens > 2**31 or requires_int64_offsets(
        scores, cu_seqlens_q, seqused_k, query_positions
    )
    score_wide = capacity > topk and (
        shared_wide or requires_int64_offsets(query, key_cache, weights, block_table, bounds)
    )
    if mxfp8 and score_wide:
        raise ValueError("fused paged MXFP8 scoring requires int32 input offsets.")
    warpdrive_score = None
    if use_cute:
        import cutlass

        from attn_gym._backends.arch import runs_sm100_kernels
        from attn_gym._backends.cute.utils import requires_int64_abi
        from attn_gym.warpdrive.cache.target import compile_target_scope, detect_compile_target

        from .cute import _compile_topk

        if not runs_sm100_kernels(capability):
            raise ValueError("the CuTe paged selector requires an SM100-family GPU.")
        cute_output = output.unsqueeze(0)
        select_wide = requires_int64_abi(scores, cute_output, bounds)
        with compile_target_scope(detect_compile_target(query.device.index)):
            topk_kernel = _compile_topk(
                topk,
                False,
                1,
                select_wide,
                torch.are_deterministic_algorithms_enabled(),
                True,
                paged=True,
            )
            if mxfp8 or can_use_warpdrive_score(
                query, key_cache, weights, block_table, seqused_k, query_positions, score_wide
            ):
                if mxfp8:
                    from .paged_score_mxfp8 import get_paged_mxfp8_score_runner as prepare_score

                    scales = (q_scale, k_scale)
                else:
                    from .paged_score_warpdrive import get_paged_score_runner as prepare_score

                    scales = ()
                warpdrive_score = prepare_score(
                    query,
                    key_cache,
                    weights,
                    block_table,
                    cu_seqlens_q,
                    seqused_k,
                    query_positions,
                    scores,
                    bounds,
                    topk,
                    compress_ratio,
                    *scales,
                )
        integer = cutlass.Int64 if select_wide else cutlass.Int32
    else:
        select_wide = shared_wide or requires_int64_offsets(output)
    tma = (
        warpdrive_score is None
        and capacity > topk
        and key_cache.shape[1] % 32 == 0
        and can_use_tma(query)
        and can_use_tma(key_cache)
    )
    if tma:
        from triton.tools.tensor_descriptor import TensorDescriptor

        # Measured on B200: split short pools to fill the GPU; amortize page
        # routing and Q loads over 4/8 tiles once there are enough work items.
        work = tokens * capacity
        bn = 32 if work <= 32768 or key_cache.shape[1] % 64 else 64
        tiles = 1 if work <= 32768 else 4 if work <= 262144 else 8
        bh, bd = max(16, triton.next_power_of_2(heads)), max(16, triton.next_power_of_2(dim))
        query_desc = TensorDescriptor.from_tensor(query, [1, bh, bd])
        key_desc = TensorDescriptor.from_tensor(key_cache, [1, bn, bd])
    with torch.cuda.device(query.device):
        for start in range(0, tokens, slab):
            rows = min(slab, tokens - start)
            if warpdrive_score is not None:
                warpdrive_score(start, rows)
            elif tma:
                paged_score_tma_kernel[(rows, triton.cdiv(capacity, bn * tiles))](
                    query_desc,
                    key_desc,
                    weights,
                    block_table,
                    cu_seqlens_q,
                    seqused_k,
                    query_positions,
                    scores,
                    bounds,
                    seqused_k.numel(),
                    capacity,
                    start,
                    *weights.stride(),
                    *block_table.stride(),
                    heads,
                    dim,
                    key_cache.shape[1],
                    topk,
                    compress_ratio,
                    query_positions is not None,
                    score_wide,
                    bn,
                    bh,
                    bd,
                    tiles,
                    num_warps=4,
                    num_stages=2,
                )
            elif capacity > topk:
                paged_score_kernel[(rows, triton.cdiv(capacity, 128))](
                    query,
                    key_cache,
                    weights,
                    block_table,
                    cu_seqlens_q,
                    seqused_k,
                    query_positions,
                    scores,
                    bounds,
                    seqused_k.numel(),
                    capacity,
                    start,
                    *query.stride(),
                    *key_cache.stride(),
                    *weights.stride(),
                    *block_table.stride(),
                    heads,
                    dim,
                    key_cache.shape[1],
                    topk,
                    compress_ratio,
                    query_positions is not None,
                    score_wide,
                    128,
                    max(16, triton.next_power_of_2(heads)),
                    max(16, triton.next_power_of_2(dim)),
                    query.stride(-1) == 1,
                    key_cache.stride(-1) == 1,
                    num_warps=4,
                )
            if use_cute:
                topk_kernel(
                    scores[: triton.cdiv(rows, 2)],
                    cute_output,
                    integer(start // 2),
                    bounds,
                )
            else:
                block = min(4096, max(256, triton.next_power_of_2(capacity)))
                paged_radix_topk_kernel[(rows,)](
                    scores,
                    output,
                    cu_seqlens_q,
                    seqused_k,
                    query_positions,
                    seqused_k.numel(),
                    capacity,
                    start,
                    topk,
                    compress_ratio,
                    query_positions is not None,
                    select_wide,
                    block,
                    num_warps=4,
                )
    return output
