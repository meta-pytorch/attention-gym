"""Paged scoring parallel across candidates, followed by exact radix selection.

Keys are gathered directly from pages; no dense repacking or host readback is
needed. Score storage is capped at 32 MiB and reused across query slabs. Radix
selection is linear in candidate count and independent of the requested Top-K.
"""

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import _document_ids, can_use_tma, requires_int64_offsets


@triton.jit
def _bounds(
    CuQ,
    Lengths,
    Positions,
    row,
    sequences,
    RATIO: tl.constexpr,
    EXPLICIT: tl.constexpr,
    WIDE: tl.constexpr,
):
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
    return sequence, visible


@triton.jit(
    do_not_specialize=[
        "sequences",
        "capacity",
        "row_start",
        "QT",
        "QH",
        "QD",
        "KH",
        "KS",
        "KD",
        "WT",
        "WH",
        "BT",
        "BS",
    ]
)
def _score(
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
    QT,
    QH,
    QD,
    KH,
    KS,
    KD,
    WT,
    WH,
    BT,
    BS,
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
    local = tl.program_id(0)
    tile = tl.program_id(1)
    if WIDE:
        local = local.to(tl.int64)
        tile = tile.to(tl.int64)
        row_start = row_start.to(tl.int64)
    row = row_start + local
    sequence, visible = _bounds(CuQ, Lengths, Positions, row, sequences, RATIO, EXPLICIT, WIDE)
    if Bounds is not None:  # noqa: SIM102 — discard None pointer arithmetic at compile time
        if tile == 0:
            tl.store(Bounds + row * 2, 0)
            tl.store(Bounds + row * 2 + 1, visible)
    if visible <= TOPK or tile * BN >= visible:
        return
    candidates = tile * BN + tl.arange(0, BN)
    dims = tl.arange(0, BD)
    heads = tl.arange(0, BH)
    if WIDE:
        dims = dims.to(tl.int64)
        heads = heads.to(tl.int64)
    if Q_UNIT_D:
        QD = 1
    if K_UNIT_D:
        KD = 1
    pages = tl.load(Table + sequence * BT + (candidates // PAGE) * BS, candidates < visible, 0)
    if WIDE:
        pages = pages.to(tl.int64)
    keys = tl.load(
        K + pages[:, None] * KH + (candidates[:, None] % PAGE) * KS + dims[None, :] * KD,
        (candidates[:, None] < visible) & (dims[None, :] < D),
        0,
    )
    query = tl.load(
        Q + row * QT + heads[:, None] * QH + dims[None, :] * QD,
        (heads[:, None] < H) & (dims[None, :] < D),
        0,
    )
    weights = tl.load(W + row * WT + heads * WH, heads < H, 0).to(tl.float32)
    dots = tl.dot(keys, tl.trans(query))
    scores = tl.sum(tl.maximum(dots, 0.0) * weights[None, :], 1) * (H * D) ** -0.5
    tl.store(Scores + local * capacity + candidates, scores, candidates < visible)


@triton.jit(do_not_specialize=["sequences", "capacity", "row_start", "WT", "WH", "BT", "BS"])
def _score_tma(
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
    WT,
    WH,
    BT,
    BS,
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
    local = tl.program_id(0)
    tile = tl.program_id(1)
    if WIDE:
        local = local.to(tl.int64)
        tile = tile.to(tl.int64)
        row_start = row_start.to(tl.int64)
    row = row_start + local
    sequence, visible = _bounds(CuQ, Lengths, Positions, row, sequences, RATIO, EXPLICIT, WIDE)
    if Bounds is not None:  # noqa: SIM102 — discard None pointer arithmetic at compile time
        if tile == 0:
            tl.store(Bounds + row * 2, 0)
            tl.store(Bounds + row * 2 + 1, visible)
    start = tile * BN * TILES
    if visible <= TOPK or start >= visible:
        return
    heads = tl.arange(0, BH)
    if WIDE:
        heads = heads.to(tl.int64)
    query = Q.load([row.to(tl.int32), 0, 0]).reshape(BH, BD)
    weights = tl.load(W + row * WT + heads * WH, heads < H, 0).to(tl.float32)
    for begin in tl.range(start, tl.minimum(start + BN * TILES, visible), BN):
        page = tl.load(Table + sequence * BT + (begin // PAGE) * BS)
        keys = K.load([page, (begin % PAGE).to(tl.int32), 0]).reshape(BN, BD)
        dots = tl.dot(keys, tl.trans(query))
        scores = tl.sum(tl.maximum(dots, 0.0) * weights[None, :], 1) * (H * D) ** -0.5
        candidates = begin + tl.arange(0, BN)
        tl.store(Scores + local * capacity + candidates, scores, candidates < visible)


@triton.jit
def _ordered(scores):
    bits = tl.where(scores == 0.0, 0.0, scores).to(tl.uint32, bitcast=True)
    return bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)


@triton.jit(do_not_specialize=["sequences", "capacity", "row_start"])
def _select(
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
    local = tl.program_id(0)
    if WIDE:
        local = local.to(tl.int64)
        row_start = row_start.to(tl.int64)
    row = row_start + local
    _, visible = _bounds(CuQ, Lengths, Positions, row, sequences, RATIO, EXPLICIT, WIDE)
    offsets = tl.arange(0, BLOCK)
    if visible <= TOPK:
        for start in range(0, TOPK, BLOCK):
            rank = start + offsets
            tl.store(Out + row * TOPK + rank, tl.where(rank < visible, rank, -1), rank < TOPK)
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
            scores = tl.load(Scores + local * capacity + indices, indices < visible, 0)
            keys = _ordered(scores)
            above += tl.sum(((indices < visible) & (keys >= trial)).to(tl.int32), 0)
        prefix = tl.where(above >= TOPK, trial, prefix)
    strict = tl.full((), 0, tl.int32)
    for start in range(0, visible, BLOCK):
        indices = start + offsets
        scores = tl.load(Scores + local * capacity + indices, indices < visible, 0)
        strict += tl.sum(((indices < visible) & (_ordered(scores) > prefix)).to(tl.int32), 0)
    need -= strict
    # need is the number of equal boundary scores to retain. Prefix scans break
    # those ties toward lower IDs, and write selected IDs in sequence order.
    written = tl.full((), 0, tl.int32)
    ties_seen = tl.full((), 0, tl.int32)
    for start in range(0, visible, BLOCK):
        indices = start + offsets
        scores = tl.load(Scores + local * capacity + indices, indices < visible, 0)
        keys = _ordered(scores)
        ties = (indices < visible) & (keys == prefix)
        tie_rank = tl.cumsum(ties.to(tl.int32))
        selected = (indices < visible) & (
            (keys > prefix) | (ties & (ties_seen + tie_rank <= need))
        )
        rank = written + tl.cumsum(selected.to(tl.int32)) - 1
        tl.store(Out + row * TOPK + rank, indices.to(tl.int32), selected & (rank < TOPK))
        written += tl.sum(selected.to(tl.int32), 0)
        ties_seen += tl.sum(ties.to(tl.int32), 0)


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
):
    if torch.version.hip or torch.cuda.get_device_capability(query.device)[0] < 8:
        raise ValueError("paged_lightning_indexer requires an SM80+ NVIDIA GPU.")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("the fused paged indexer requires FP16 or BF16 inputs.")
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
    # Pair-shaped scratch shares the existing cached CuTe radix kernel's ABI.
    pairs = min(triton.cdiv(tokens, 2), (32 * 1024 * 1024) // (8 * capacity))
    slab = 2 * pairs
    scores = query.new_empty(
        (pairs, 2, capacity) if capacity > topk else (0,), dtype=torch.float32
    )
    use_cute = backend == "cute" and capacity > topk
    bounds = query.new_empty((tokens, 2), dtype=torch.int32) if use_cute else None
    wide = requires_int64_offsets(
        query,
        key_cache,
        weights,
        block_table,
        cu_seqlens_q,
        seqused_k,
        query_positions,
        scores,
        output,
        bounds,
    )
    if use_cute:
        import cutlass

        from attn_gym._backends.arch import runs_sm100_kernels
        from attn_gym._backends.cute.utils import requires_int64_abi
        from attn_gym.warpdrive.cache.target import compile_target_scope, detect_compile_target

        from .cute import _compile_topk

        if not runs_sm100_kernels(torch.cuda.get_device_capability(query.device)):
            raise ValueError("the CuTe paged selector requires an SM100-family GPU.")
        wide = wide or requires_int64_abi(scores, output, bounds)
        with compile_target_scope(detect_compile_target(query.device.index)):
            topk_kernel = _compile_topk(
                topk, False, 1, wide, torch.are_deterministic_algorithms_enabled(), True, True
            )
        integer = cutlass.Int64 if wide else cutlass.Int32
        cute_output = output.unsqueeze(0)
    tma = (
        capacity > topk
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
            if tma:
                _score_tma[(rows, triton.cdiv(capacity, bn * tiles))](
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
                    wide,
                    bn,
                    bh,
                    bd,
                    tiles,
                    num_warps=4,
                    num_stages=2,
                )
            elif capacity > topk:
                _score[(rows, triton.cdiv(capacity, 128))](
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
                    wide,
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
                _select[(rows,)](
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
                    wide,
                    block,
                    num_warps=4,
                )
    return output
