"""Fixed per-query sparse attention shared by gather and paged decode.

One CTA owns up to 16 consecutive query heads of one KV head. It walks a compact
selection followed by the local window, including a tile straddling the two sets.
Tile size depends only on the device, dtype and feature dimensions, never batching.
Neither query packing nor cache pages change the arithmetic. Metadata
only changes addresses; FP fusion is disabled for the same reason as the recurrent
delta-rule scan. Backward remains the gather backend's existing implementation.
"""

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import _document_ids, offset_strides, ptr_offset

from .gather_attn.impl.triton.primitives import online_softmax_update, resolve_selected_rows


def _key_tile(dtype, dim, dim_v, shared_memory):
    # Large FP32 tiles require additional storage for tf32x3. Smaller Ampere GPUs
    # also cannot fit 64 BF16 keys at width 512. Keep the choice common to both APIs.
    if dtype == torch.float32 and max(dim, dim_v) >= 256:
        return 16
    if max(dim, dim_v) > 256 and shared_memory < 147456:
        return 16
    return 64


def _head_tile(dtype, dim, dim_v, shared_memory):
    # FP32 width 512 needs fewer query heads on consumer Ampere: a 16x16 tile
    # uses 133120 bytes, while a 4x16 tile uses 82432 bytes (limit 101376).
    return 4 if dtype == torch.float32 and max(dim, dim_v) > 256 and shared_memory < 133120 else 16


@triton.jit
def _load_cache(K, strides, page, slot, head, dims, valid):
    return tl.load(
        K + ptr_offset((page[:, None], slot[:, None], head, dims[None, :]), strides),
        valid,
        0.0,
    )


# Triton 3.9 still specializes tuple members. Pass arbitrary layout strides as
# scalars at the JIT boundary, then group them by operand for address arithmetic.
@triton.jit(
    do_not_specialize=[
        "T",
        "TOPK",
        "N",
        "PAGE",
        "WPAGE",
        "table_seq",
        "table_page",
        "wtable_seq",
        "wtable_page",
        "q_batch",
        "q_head",
        "q_token",
        "q_dim",
        "k_page",
        "k_slot",
        "k_head",
        "k_dim",
        "v_page",
        "v_slot",
        "v_head",
        "v_dim",
        "wk_page",
        "wk_slot",
        "wk_head",
        "wk_dim",
        "wv_page",
        "wv_slot",
        "wv_head",
        "wv_dim",
        "qr_batch",
        "qr_head",
        "qr_token",
        "qr_dim",
        "kr_page",
        "kr_slot",
        "kr_head",
        "kr_dim",
        "wr_page",
        "wr_slot",
        "wr_head",
        "wr_dim",
        "o_batch",
        "o_head",
        "o_token",
        "o_dim",
        "lse_batch",
        "lse_head",
        "lse_token",
    ]
)
def _attention(
    Q,
    K,
    V,
    WK,
    WV,
    QR,
    KR,
    WR,
    Indices,
    Counts,
    CuQ,
    Table,
    WTable,
    Meta,
    Sink,
    Out,
    Lse,
    T,
    TOPK,
    N,
    PAGE,
    WPAGE,
    table_seq,
    table_page,
    wtable_seq,
    wtable_page,
    q_batch,
    q_head,
    q_token,
    q_dim,
    k_page,
    k_slot,
    k_head,
    k_dim,
    v_page,
    v_slot,
    v_head,
    v_dim,
    wk_page,
    wk_slot,
    wk_head,
    wk_dim,
    wv_page,
    wv_slot,
    wv_head,
    wv_dim,
    qr_batch,
    qr_head,
    qr_token,
    qr_dim,
    kr_page,
    kr_slot,
    kr_head,
    kr_dim,
    wr_page,
    wr_slot,
    wr_head,
    wr_dim,
    o_batch,
    o_head,
    o_token,
    o_dim,
    lse_batch,
    lse_head,
    lse_token,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    DV: tl.constexpr,
    DR: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    BH: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    PAGED: tl.constexpr,
    HAS_CU: tl.constexpr,
    HAS_SINK: tl.constexpr,
    WIDE: tl.constexpr,
):
    query = tl.program_id(0)
    batch = tl.program_id(1)
    head_tile = tl.program_id(2)
    group = H // HK
    kv_head = head_tile // tl.cdiv(group, BH)
    heads = kv_head * group + head_tile % tl.cdiv(group, BH) * BH + tl.arange(0, BH)
    head_valid = heads < (kv_head + 1) * group
    # All address products widen before multiplication on the wide ABI.
    query = query.to(tl.int64 if WIDE else tl.int32)
    batch = batch.to(tl.int64 if WIDE else tl.int32)
    strides_q = offset_strides((q_batch, q_head, q_token, q_dim), WIDE)
    strides_k = offset_strides((k_page, k_slot, k_head, k_dim), WIDE)
    strides_v = offset_strides((v_page, v_slot, v_head, v_dim), WIDE)
    strides_wk = offset_strides((wk_page, wk_slot, wk_head, wk_dim), WIDE)
    strides_wv = offset_strides((wv_page, wv_slot, wv_head, wv_dim), WIDE)
    row = batch * T + query
    count = tl.load(Counts + row)
    if PAGED:
        table_strides = offset_strides((table_seq, table_page), WIDE)
        wtable_strides = offset_strides((wtable_seq, wtable_page), WIDE)
        sequence = tl.load(Meta + row * 3)
        window_start = tl.load(Meta + row * 3 + 1)
        window_end = tl.load(Meta + row * 3 + 2)
    else:
        sequence = batch
        first = 0
        if HAS_CU:
            doc = _document_ids(CuQ, query, N, WIDE)
            first = tl.load(CuQ + doc)
        window_start = tl.maximum(first, query + 1 - WINDOW)
        window_end = query + 1 if WINDOW else window_start
        if HAS_CU:
            window_end = tl.where(query < tl.load(CuQ + N), window_end, window_start)
    total = count + window_end - window_start
    dims = tl.arange(0, BD)
    dims_v = tl.arange(0, BV)
    q = tl.load(
        Q + ptr_offset((batch, heads[:, None], query, dims[None, :]), strides_q),
        head_valid[:, None] & (dims[None, :] < D),
        0.0,
    )
    if DR:
        dims_r = tl.arange(0, BR)
        strides_qr = offset_strides((qr_batch, qr_head, qr_token, qr_dim), WIDE)
        strides_kr = offset_strides((kr_page, kr_slot, kr_head, kr_dim), WIDE)
        strides_wr = offset_strides((wr_page, wr_slot, wr_head, wr_dim), WIDE)
        qr = tl.load(
            QR + ptr_offset((batch, heads[:, None], query, dims_r[None, :]), strides_qr),
            head_valid[:, None] & (dims_r[None, :] < DR),
            0.0,
        )
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    if HAS_SINK:
        maximum = tl.load(Sink + heads, head_valid, -float("inf")).to(tl.float32)
    denominator = tl.where(maximum == -float("inf"), 0.0, 1.0)
    accumulator = tl.zeros((BH, BV), tl.float32)
    for start in range(0, total, BN):
        column = start + tl.arange(0, BN)
        sparse = column < count
        selected = tl.load(Indices + row * TOPK + column, sparse, 0)
        entry = tl.where(sparse, selected, window_start + column - count)
        entry = entry.to(tl.int64 if WIDE else tl.int32)
        valid = column < total
        if PAGED:
            page = tl.load(
                Table + ptr_offset((sequence, entry // PAGE), table_strides),
                sparse & valid,
                0,
            )
            wpage = tl.load(
                WTable + ptr_offset((sequence, entry // WPAGE), wtable_strides),
                ~sparse & valid,
                0,
            )
            slot, wslot = entry % PAGE, entry % WPAGE
        else:
            page = tl.full((BN,), 0, tl.int64 if WIDE else tl.int32) + batch
            wpage = page
            slot = entry
            wslot = entry
        page = page.to(tl.int64 if WIDE else tl.int32)
        wpage = wpage.to(tl.int64 if WIDE else tl.int32)
        ks = _load_cache(
            K,
            strides_k,
            page,
            slot,
            kv_head,
            dims,
            (sparse & valid)[:, None] & (dims[None, :] < D),
        )
        kw = _load_cache(
            WK,
            strides_wk,
            wpage,
            wslot,
            kv_head,
            dims,
            (~sparse & valid)[:, None] & (dims[None, :] < D),
        )
        keys = tl.where(sparse[:, None], ks, kw)
        vs = _load_cache(
            V,
            strides_v,
            page,
            slot,
            kv_head,
            dims_v,
            (sparse & valid)[:, None] & (dims_v[None, :] < DV),
        )
        vw = _load_cache(
            WV,
            strides_wv,
            wpage,
            wslot,
            kv_head,
            dims_v,
            (~sparse & valid)[:, None] & (dims_v[None, :] < DV),
        )
        values = tl.where(sparse[:, None], vs, vw)
        logits = tl.dot(q, tl.trans(keys), input_precision="tf32x3")
        if DR:
            rs = _load_cache(
                KR,
                strides_kr,
                page,
                slot,
                kv_head,
                dims_r,
                (sparse & valid)[:, None] & (dims_r[None, :] < DR),
            )
            rw = _load_cache(
                WR,
                strides_wr,
                wpage,
                wslot,
                kv_head,
                dims_r,
                (~sparse & valid)[:, None] & (dims_r[None, :] < DR),
            )
            logits += tl.dot(qr, tl.trans(tl.where(sparse[:, None], rs, rw)))
        logits = tl.where(head_valid[:, None] & valid[None, :], logits * SCALE, -float("inf"))
        accumulator, maximum, denominator = online_softmax_update(
            accumulator, maximum, denominator, logits, values
        )
    output = tl.where(denominator[:, None] > 0, accumulator / denominator[:, None], 0.0)
    strides_o = offset_strides((o_batch, o_head, o_token, o_dim), WIDE)
    tl.store(
        Out + ptr_offset((batch, heads[:, None], query, dims_v[None, :]), strides_o),
        output,
        head_valid[:, None] & (dims_v[None, :] < DV),
    )
    lse = tl.where(denominator > 0, maximum + tl.log(denominator), -float("inf"))
    strides_lse = offset_strides((lse_batch, lse_head, lse_token), WIDE)
    tl.store(
        Lse + ptr_offset((batch, heads, query), strides_lse),
        lse,
        head_valid,
    )


def _launch(
    query,
    key,
    value,
    local_key,
    local_value,
    indices,
    counts,
    sink,
    out,
    lse,
    *,
    kv_heads,
    scale,
    window=0,
    cu_seqlens=None,
    query_rope=None,
    key_rope=None,
    local_rope=None,
    table=None,
    window_table=None,
    metadata=None,
):
    from attn_gym._backends.triton.utils import requires_int64_offsets

    batch, heads, tokens, dim = query.shape
    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("batch-invariant sparse attention supports FP16, BF16, and FP32 inputs.")
    if not batch or not tokens:
        return
    dim_v = value.shape[-1]
    dim_r = 0 if query_rope is None else query_rope.shape[-1]
    if max(dim, dim_v) > 512 or dim_r > 128:
        raise ValueError(
            "batch-invariant sparse attention requires head dims <= 512 and RoPE <= 128"
        )
    local_key = key if local_key is None else local_key
    local_value = value if local_value is None else local_value
    query_rope = query if not dim_r else query_rope
    key_rope = key if not dim_r else key_rope
    local_rope = key_rope if local_rope is None else local_rope
    tensors = (
        query,
        key,
        value,
        local_key,
        local_value,
        query_rope,
        key_rope,
        local_rope,
        indices,
        counts,
        sink,
        out,
        lse,
        table,
        window_table,
        metadata,
        cu_seqlens,
    )
    wide = requires_int64_offsets(*tensors)
    if table is not None:
        wide |= (
            max(table.shape[1] * key.shape[1], window_table.shape[1] * local_key.shape[1]) >= 2**31
        )
    shared_memory = torch.cuda.get_device_properties(query.device).shared_memory_per_block_optin
    head_tile = _head_tile(query.dtype, dim, dim_v, shared_memory)
    _attention[(tokens, batch, kv_heads * triton.cdiv(heads // kv_heads, head_tile))](
        query,
        key,
        value,
        local_key,
        local_value,
        query_rope,
        key_rope,
        local_rope,
        indices,
        counts,
        cu_seqlens,
        table,
        window_table,
        metadata,
        sink,
        out,
        lse,
        tokens,
        indices.shape[-1],
        0 if cu_seqlens is None else cu_seqlens.numel() - 1,
        key.shape[1],
        local_key.shape[1],
        *((0, 0) if table is None else table.stride()),
        *((0, 0) if window_table is None else window_table.stride()),
        *query.stride(),
        *key.stride(),
        *value.stride(),
        *local_key.stride(),
        *local_value.stride(),
        *query_rope.stride(),
        *key_rope.stride(),
        *local_rope.stride(),
        *out.stride(),
        *lse.stride(),
        H=heads,
        HK=kv_heads,
        D=dim,
        DV=dim_v,
        DR=dim_r,
        BD=max(16, triton.next_power_of_2(dim)),
        BV=max(16, triton.next_power_of_2(dim_v)),
        BR=max(16, triton.next_power_of_2(dim_r)),
        BN=_key_tile(
            query.dtype,
            triton.next_power_of_2(dim),
            triton.next_power_of_2(dim_v),
            shared_memory,
        ),
        BH=head_tile,
        WINDOW=window,
        SCALE=scale,
        PAGED=table is not None,
        HAS_CU=cu_seqlens is not None,
        HAS_SINK=sink is not None,
        WIDE=wide,
        num_warps=4,
        num_stages=1,
        enable_fp_fusion=False,
    )


def gather_forward(
    query,
    sparse_kv,
    local_kv,
    indices,
    sink,
    cu_seqlens,
    cu_seqlens_k,
    window,
    scale,
    *,
    kv_heads,
    sparse_v=None,
    local_v=None,
    query_rope=None,
    sparse_rope=None,
    local_rope=None,
):
    if query_rope is not None and query_rope.shape[-1]:
        from .gather_attn.impl.triton.forward import ROPE_REQUIREMENTS
        from .gather_attn.impl.triton.primitives import can_use_shared_kv_schedule

        if not can_use_shared_kv_schedule(
            query, sparse_kv, local_kv, window, query_rope, sparse_rope, local_rope
        ):
            raise ValueError(ROPE_REQUIREMENTS)
    rows, counts, _, _ = resolve_selected_rows(
        indices, sparse_kv.shape[2], cu_seqlens, cu_seqlens_k
    )
    out = query.new_empty(query.shape)
    lse = query.new_empty(query.shape[:3], dtype=torch.float32)

    def cache(tensor):
        return None if tensor is None else tensor.permute(0, 2, 1, 3)

    _launch(
        query,
        cache(sparse_kv),
        cache(sparse_kv if sparse_v is None else sparse_v),
        cache(local_kv),
        cache(local_kv if local_v is None else local_v),
        rows,
        counts,
        sink,
        out,
        lse,
        kv_heads=kv_heads,
        scale=scale,
        window=window,
        cu_seqlens=cu_seqlens,
        query_rope=query_rope,
        key_rope=cache(sparse_rope),
        local_rope=cache(local_rope),
    )
    return out, lse


@triton.jit(do_not_specialize=["T", "TOPK", "N", "CAPACITY", "WCAPACITY", "I_STRIDE"])
def _paged_rows(
    Indices,
    Length,
    CuQ,
    Used,
    Rows,
    Counts,
    Meta,
    T,
    TOPK,
    N,
    CAPACITY,
    WCAPACITY,
    I_STRIDE,
    INDEX_BLOCK: tl.constexpr,
    WINDOW: tl.constexpr,
    SHARED: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    sequence = _document_ids(CuQ, row, N, True)
    end = tl.load(CuQ + tl.minimum(sequence + 1, N))
    position = tl.load(Used + sequence, sequence < N, 0).to(tl.int64) - end + row
    active = (row < end) & (position >= 0)
    window_start = tl.maximum(position + 1 - WINDOW, 0)
    if SHARED:
        window_start -= window_start % INDEX_BLOCK
    window_end = tl.minimum(position + 1, WCAPACITY)
    window_end = tl.maximum(window_start, window_end)
    window_end = tl.where(active, window_end, window_start)
    limit = window_start if SHARED else CAPACITY
    n = TOPK if Length is None else tl.minimum(tl.maximum(tl.load(Length + row), 0), TOPK)
    n = tl.where(active, n, 0).to(tl.int64)
    count = 0
    for start in range(0, n * INDEX_BLOCK, 256):
        column = start + tl.arange(0, 256)
        block = tl.load(
            Indices + row * I_STRIDE + column // INDEX_BLOCK, column < n * INDEX_BLOCK, -1
        ).to(tl.int64)
        entry = block * INDEX_BLOCK + column % INDEX_BLOCK
        valid = (column < n * INDEX_BLOCK) & (block >= 0) & (entry < limit)
        rank = tl.cumsum(valid.to(tl.int32)) - 1
        tl.store(Rows + row * TOPK * INDEX_BLOCK + count + rank, entry, valid)
        count += tl.sum(valid.to(tl.int32), 0)
    tl.store(Counts + row, count)
    tl.store(Meta + row * 3, sequence)
    tl.store(Meta + row * 3 + 1, window_start)
    tl.store(Meta + row * 3 + 2, window_end)


def paged_forward(
    query,
    key,
    value,
    table,
    indices,
    cu_q,
    used,
    topk_length,
    window_key,
    window_value,
    window_table,
    sink,
    max_seqlen_q,
    index_block_size,
    window,
    scale,
    *,
    return_lse,
):
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"The fused paged_sparse_decode needs float16 or bfloat16 inputs, got {query.dtype}."
        )
    for name, dim in (("head_dim", query.shape[-1]), ("head_dim_v", value.shape[-1])):
        if dim < 16 or dim & (dim - 1):
            raise ValueError(
                f"The fused paged_sparse_decode needs {name} to be a power of two of at least "
                f"16, got {dim}; use impl='reference'."
            )
    tokens, heads, _ = query.shape
    out = query.new_empty(tokens, heads, value.shape[-1])
    lse = query.new_empty((tokens, heads), dtype=torch.float32)
    if tokens == 0 or used.numel() == 0:
        return out, lse if return_lse else None
    shared = window_key is None
    if shared:
        window_key, window_value, window_table = key, value, table
    capacity = table.shape[1] * key.shape[1]
    window_capacity = window_table.shape[1] * window_key.shape[1]
    rows = torch.empty(
        (1, tokens, indices.shape[1] * index_block_size),
        device=query.device,
        dtype=torch.int32 if capacity < 2**31 else torch.int64,
    )
    counts = torch.empty((tokens,), device=query.device, dtype=torch.int32)
    metadata = torch.empty((tokens, 3), device=query.device, dtype=torch.int64)
    _paged_rows[(tokens,)](
        indices,
        topk_length,
        cu_q,
        used,
        rows,
        counts,
        metadata,
        tokens,
        indices.shape[1],
        used.numel(),
        capacity,
        window_capacity,
        indices.stride(0),
        INDEX_BLOCK=index_block_size,
        WINDOW=window,
        SHARED=shared,
        num_warps=4,
    )
    _launch(
        query.permute(1, 0, 2).unsqueeze(0),
        key,
        value,
        window_key,
        window_value,
        rows,
        counts,
        sink,
        out.permute(1, 0, 2).unsqueeze(0),
        lse.T.unsqueeze(0),
        kv_heads=key.shape[2],
        scale=scale,
        table=table,
        window_table=window_table,
        metadata=metadata,
    )
    return out, lse if return_lse else None
