"""Memory-bounded TMA/tensor-core indexer for Hopper and newer NVIDIA GPUs.

Each CTA owns one query and retains only its running Top-K while streaming key
tiles. The tensor-core product is candidate-by-head so the head reduction leaves
candidate indices distributed across warps, rather than replicating selection
across head warps. FP16/BF16 store no global scores or partial selections.
Native MXFP8 uses a separate scorer and streaming selector with a reusable score
slab of at most 1024 rows and 32 MiB.
This module also prepares packed candidate bounds for the CuTe indexer backend.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.compiler import CompiledKernel
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import (
    _document_ids,
    offset_strides,
    ptr_offset,
    requires_int64_offsets,
)

from ..validation import validate_precision
from .cute import score_workspace_pairs
from .triton_slab import slab_row

# Measured shared memory is (2 * BN + BH) * BD * elem_size: Triton double-buffers the TMA key
# tile even at num_stages=1, plus one Q tile. Hopper/Blackwell allow 227 KiB per block; 192 KiB
# leaves headroom for reduction scratch and only bounds how far the key tile widens.
_TILE_SMEM_BUDGET = 192 * 1024
# Per 4-warp selector CTA, 2 * 8192 packed int64 keys alone need 256 registers/thread.
# Use bounded radix refinement before the streaming merge outgrows the register file.
_MAX_SLAB_STREAMING_K = 4096


@triton.jit
def _packed_interval(
    CuQ,
    CuK,
    query,
    num_documents,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    document = _document_ids(CuQ, query, num_documents, USE_INT64_OFFSETS)
    offset = document.to(tl.int64) if USE_INT64_OFFSETS else document
    begin = tl.load(CuK + offset)
    end = tl.load(CuK + tl.minimum(offset + 1, num_documents))
    if CAUSAL:
        local_position = query - tl.load(CuQ + offset)
        end = tl.minimum(end, begin + (local_position + 1) // RATIO)
    return begin, end


@triton.jit(do_not_specialize=["T", "S", "K", "num_documents"])
def _fill_visible_kernel(
    Out,
    CuQ,
    CuK,
    T,
    S,
    K,
    num_documents,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """All candidates fit: each CTA writes one output tile without loading Q/K or scores."""
    tile = tl.program_id(0).to(tl.int64)
    tiles_per_row = tl.cdiv(K.to(tl.int64), BLOCK)
    row = tile // tiles_per_row
    rank = (tile % tiles_per_row) * BLOCK + tl.arange(0, BLOCK)
    query = (row % T).to(tl.int32)
    if PACKED:
        begin, end = _packed_interval(CuQ, CuK, query, num_documents, CAUSAL, RATIO, True)
        count = end - begin
    else:
        count = (query + 1) // RATIO if CAUSAL else S
    tl.store(Out + row * K + rank, tl.where(rank < count, rank, -1), rank < K)


def fill_visible(
    output: Tensor,
    candidates: int,
    causal: bool,
    compress_ratio: int,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
) -> Tensor:
    """Emit ascending local indices and trailing -1s when every candidate fits."""
    batch, tokens, topk = output.shape
    block = 256
    with torch.cuda.device(output.device):
        _fill_visible_kernel[(batch * tokens * triton.cdiv(topk, block),)](
            output,
            cu_seqlens,
            cu_seqlens_k,
            tokens,
            candidates,
            topk,
            cu_seqlens.numel() - 1 if cu_seqlens is not None else 0,
            causal,
            compress_ratio,
            cu_seqlens is not None,
            block,
        )
    return output


@triton.jit(do_not_specialize=["T", "S", "WB", "num_documents"])
def _index_kernel(
    Q,
    K,
    W,
    Out,
    CuQ,
    CuK,
    num_documents,
    T,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    WB,
    WT: tl.constexpr,
    WH: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    KEEP: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    PACKED: tl.constexpr,
):
    """Stream TMA key tiles through tensor cores and a register-resident Top-K."""
    batch = tl.program_id(1)
    query = tl.program_id(0)
    if USE_INT64_OFFSETS:
        batch = batch.to(tl.int64)
        query = query.to(tl.int64)
        T = T.to(tl.int64)
    rank = tl.arange(0, KEEP)
    begin = 0
    end = (query + 1) // RATIO if CAUSAL else S
    if PACKED:
        begin, end = _packed_interval(
            CuQ, CuK, query, num_documents, CAUSAL, RATIO, USE_INT64_OFFSETS
        )
    if end - begin <= TOPK:
        # Every visible candidate is selected, so scores cannot change membership.
        indices = tl.where(rank < end - begin, rank, -1)
        tl.store(Out + ptr_offset((batch, query, rank), (T * TOPK, TOPK, 1)), indices, rank < TOPK)
        return

    heads = tl.arange(0, BH)
    if USE_INT64_OFFSETS:
        heads = heads.to(tl.int64)
    weights = tl.load(W + ptr_offset((batch, query, heads), (WB, WT, WH)), heads < H, 0).to(
        tl.float32
    )
    # TMA coordinates are int32; the descriptor handles wide byte strides.
    q = Q.load([batch.to(tl.int32), query.to(tl.int32), 0, 0]).reshape(BH, BD)
    best = tl.full((KEEP,), -9223372036854775808, tl.int64)
    for start in tl.range(begin, end, BN):
        k = K.load([batch.to(tl.int32), start.to(tl.int32), 0]).reshape(BN, BD)
        dots = tl.dot(k, tl.trans(q))
        scores = tl.sum(tl.maximum(dots, 0.0) * weights[None, :], axis=1)
        scores = scores * (H * D) ** -0.5
        scores = tl.gather(scores, rank % BN, axis=0)
        bits = tl.where(scores == 0.0, 0.0, scores).to(tl.int32, bitcast=True)
        ordinal = bits ^ ((bits >> 31) & 0x7FFFFFFF)
        candidate = start + rank
        packed = (ordinal.to(tl.int64) << 32) | (0xFFFFFFFF - candidate.to(tl.int64))
        packed = tl.where((rank < BN) & (candidate < end), packed, -9223372036854775808)
        best = tl.topk(tl.join(best, packed).reshape(2 * KEEP), KEEP)
    indices = (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32)
    indices = tl.where(best == -9223372036854775808, -1, indices - begin)
    tl.store(Out + ptr_offset((batch, query, rank), (T * TOPK, TOPK, 1)), indices, rank < TOPK)


@triton.jit(
    do_not_specialize=[
        "FirstBatch",
        "FirstPair",
        "T",
        "S",
        "QB",
        "QT",
        "QH",
        "QD",
        "KB",
        "KT",
        "KD",
        "WB",
        "WT",
        "WH",
        "QSB",
        "QST",
        "QSH",
        "QSG",
        "KSB",
        "KSS",
        "KSG",
    ]
)
def _mxfp8_score_kernel(
    Q,
    K,
    W,
    QS,
    KS,
    Scores,
    FirstBatch,
    FirstPair,
    T,
    S,
    H: tl.constexpr,
    D: tl.constexpr,
    QB,
    QT,
    QH,
    QD,
    KB,
    KT,
    KD,
    WB,
    WT,
    WH,
    QSB,
    QST,
    QSH,
    QSG,
    KSB,
    KSS,
    KSG,
    DENSE_ROWS: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    BN: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.program_id(1) * BN
    if USE_INT64_OFFSETS:
        row = row.to(tl.int64)
        FirstBatch = FirstBatch.to(tl.int64)
        T = T.to(tl.int64)
        S = S.to(tl.int64)
    batch, query = slab_row(row, FirstBatch, FirstPair, T, USE_INT64_OFFSETS)
    end = (query + 1) // RATIO if CAUSAL else S
    if (query < T) & (start < end):
        h = tl.arange(0, H)
        d = tl.arange(0, D)
        g = tl.arange(0, D // 32)
        n = start + tl.arange(0, BN)
        if USE_INT64_OFFSETS:
            h = h.to(tl.int64)
            d = d.to(tl.int64)
            g = g.to(tl.int64)
            n = n.to(tl.int64)
        # Derive dense batch strides from row length and model dimensions so vectorized
        # loads retain their alignment. Transposed/padded strides may all depend on T/S.
        q_strides = offset_strides(
            (T * H * D, H * D, D, 1) if DENSE_ROWS else (QB, QT, QH, QD), USE_INT64_OFFSETS
        )
        k_strides = offset_strides(
            (S * D, D, 1) if DENSE_ROWS else (KB, KT, KD), USE_INT64_OFFSETS
        )
        w_strides = offset_strides(
            (T * H, H, 1) if DENSE_ROWS else (WB, WT, WH), USE_INT64_OFFSETS
        )
        qs_strides = offset_strides(
            (T * H * (D // 32), H * (D // 32), D // 32, 1) if DENSE_ROWS else (QSB, QST, QSH, QSG),
            USE_INT64_OFFSETS,
        )
        ks_strides = offset_strides(
            (S * (D // 32), D // 32, 1) if DENSE_ROWS else (KSB, KSS, KSG), USE_INT64_OFFSETS
        )
        q = tl.load(Q + ptr_offset((batch, query, h[:, None], d[None, :]), q_strides))
        qs = tl.load(QS + ptr_offset((batch, query, h[:, None], g[None, :]), qs_strides))
        w = tl.load(W + ptr_offset((batch, query, h), w_strides)).to(tl.float32)
        k = tl.load(
            K + ptr_offset((batch, n[:, None], d[None, :]), k_strides), n[:, None] < end, 0.0
        )
        ks = tl.load(
            KS + ptr_offset((batch, n[:, None], g[None, :]), ks_strides), n[:, None] < end, 127
        )
        # Scales keep [matrix row, D/32] layout, including the transposed Q operand.
        dots = tl.dot_scaled(k, ks, "e4m3", tl.trans(q), qs, "e4m3")
        scores = tl.sum(tl.maximum(dots, 0.0) * w[None, :], axis=1) * (H * D) ** -0.5
        tl.store(Scores + row * S + n, scores, n < end)


def launch_mxfp8_scores(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    scores: Tensor,
    pair_start: int,
    causal: bool,
    compress_ratio: int,
    wide: bool,
) -> CompiledKernel:
    """Write visible scores in a compact [pairs,2,S] slab; other slots are untouched."""
    # Same-size byte views preserve logical E8M0 strides without packing or conversion.
    return _mxfp8_score_kernel[(scores.shape[0] * 2, triton.cdiv(k.shape[1], 128))](
        q,
        k,
        weights,
        q_scale.view(torch.uint8),
        k_scale.view(torch.uint8),
        scores,
        *divmod(pair_start, triton.cdiv(q.shape[1], 2)),
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[3],
        *q.stride(),
        *k.stride(),
        *weights.stride(),
        *q_scale.stride(),
        *k_scale.stride(),
        all(tensor.is_contiguous() for tensor in (q, k, weights, q_scale, k_scale)),
        causal,
        compress_ratio,
        128,
        wide,
        num_warps=4,
        num_stages=1,
    )


@triton.jit(do_not_specialize=["PairStart", "T", "S"])
def _streaming_topk_kernel(
    Scores,
    Out,
    PairStart,
    T,
    S,
    TOPK: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    KEEP: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    row = tl.program_id(0)
    if USE_INT64_OFFSETS:
        row = row.to(tl.int64)
        PairStart = PairStart.to(tl.int64)
    pair = PairStart + row // 2
    batch = pair // tl.cdiv(T, 2)
    query = 2 * (pair % tl.cdiv(T, 2)) + row % 2
    if query < T:
        rank = tl.arange(0, KEEP)
        best = tl.full((KEEP,), -9223372036854775808, tl.int64)
        end = (query + 1) // RATIO if CAUSAL else S
        for start in tl.range(0, end, KEEP):
            candidate = start + rank
            scores = tl.load(Scores + row * S + candidate, candidate < end, 0.0)
            bits = tl.where(scores == 0.0, 0.0, scores).to(tl.int32, bitcast=True)
            ordinal = bits ^ ((bits >> 31) & 0x7FFFFFFF)
            packed = (ordinal.to(tl.int64) << 32) | (0xFFFFFFFF - candidate.to(tl.int64))
            packed = tl.where(candidate < end, packed, -9223372036854775808)
            best = tl.topk(tl.join(best, packed).reshape(2 * KEEP), KEEP)
        indices = (0xFFFFFFFF - (best & 0xFFFFFFFF)).to(tl.int32)
        indices = tl.where(best == -9223372036854775808, -1, indices)
        tl.store(Out + (batch * T + query) * TOPK + rank, indices, rank < TOPK)


def _launch_streaming_topk(
    scores: Tensor, output: Tensor, start: int, causal: bool, compress_ratio: int, wide: bool
) -> None:
    topk = output.shape[-1]
    _streaming_topk_kernel[(scores.shape[0] * 2,)](
        scores,
        output,
        start,
        output.shape[1],
        scores.shape[-1],
        topk,
        causal,
        compress_ratio,
        max(128, triton.next_power_of_2(topk)),
        wide,
        num_warps=4,
    )


def launch(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
    *,
    selector: str = "default",
) -> Tensor:
    """Return nondifferentiable INT32 indices without a quadratic workspace.

    Q/K require contiguous last dimensions and 16-byte-aligned bases/outer
    strides. FP16/BF16, H/D <= 256, T <= 2**20, and non-negative Top-K are supported; the
    register-resident selection makes each key tile cost grow with Top-K.
    MXFP8 (SM100/SM103, H in {32, 64}, D=128) uses a separate kernel that accepts
    arbitrary nonnegative strides.
    Gradient-requiring inputs are allowed: selection returns indices, not trainable scores.
    The public registered operator keeps host descriptors outside graph tracing.
    """
    if q.dtype == torch.float8_e4m3fn and (cu_seqlens is not None or cu_seqlens_k is not None):
        raise ValueError("MXFP8 indexing does not support packed cu_seqlens.")
    capability = torch.cuda.get_device_capability(q.device) if q.is_cuda else (0, 0)
    if torch.version.hip or capability[0] < 9:
        raise ValueError("The Triton TMA indexer requires Hopper or newer NVIDIA GPUs.")
    validate_precision(q, k, weights, q_scale, k_scale)
    scaled_fp8 = q.dtype == torch.float8_e4m3fn
    if scaled_fp8:
        if capability not in ((10, 0), (10, 3)):
            raise ValueError("Native MXFP8 Triton indexing requires SM100 or SM103.")
        if q.shape[2] not in (32, 64) or q.shape[3] != 128:
            raise ValueError("MXFP8 Triton indexing requires H in {32, 64} and D=128.")
    elif q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("The Triton indexer requires FP16 or BF16 inputs, or scaled MXFP8.")
    batch, tokens, heads, dim = q.shape
    candidates = k.shape[1]
    if heads > 256 or dim > 256 or dim % 8 or tokens > 2**20:
        raise ValueError(
            "The Triton indexer requires H <= 256, D <= 256 divisible by 8, and T <= 2**20."
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}.")
    if cu_seqlens is None and candidates != tokens // compress_ratio:
        raise ValueError(
            f"k must hold T // compress_ratio candidates, got T={tokens}, S={candidates}, "
            f"compress_ratio={compress_ratio}."
        )
    if candidates == 0:
        return torch.full((batch, tokens, topk), -1, dtype=torch.int32, device=q.device)
    if not scaled_fp8:
        for tensor in (q, k):
            if tensor.stride(-1) != 1 or any(s % 8 for s in tensor.stride()[:-1]):
                raise ValueError(
                    "TMA requires a contiguous last dimension and 16-byte outer strides."
                )
            if tensor.data_ptr() % 16:
                raise ValueError("TMA requires 16-byte-aligned Q/K base pointers.")
    # S may include inactive capacity in packed mode. No document can contain more
    # complete candidate groups than the entire query tensor.
    all_fit = topk >= min(candidates, tokens // compress_ratio)
    use_slab_radix = scaled_fp8 and (
        selector == "gvr2" or (selector == "auto" and topk > _MAX_SLAB_STREAMING_K)
    )
    # The streaming merge sorts 2*next_power_of_2(topk) entries. Reject unrepresentable
    # partial selections before allocating a potentially enormous output.
    if not all_fit and not use_slab_radix and topk > tl.TRITON_MAX_TENSOR_NUMEL // 2:
        raise ValueError(
            "Triton streaming topk exceeds its tile limit; use selector='gvr2' for dense inputs"
        )
    output = torch.empty((batch, tokens, topk), dtype=torch.int32, device=q.device)
    if topk == 0:
        return output
    if all_fit:
        return fill_visible(output, candidates, causal, compress_ratio, cu_seqlens, cu_seqlens_k)
    if scaled_fp8:
        assert q_scale is not None and k_scale is not None
        pairs = score_workspace_pairs(batch, tokens, candidates)
        scores = torch.empty((pairs, 2, candidates), dtype=torch.float32, device=q.device)
        wide = requires_int64_offsets(q, k, weights, q_scale, k_scale, scores, output)
        total_pairs = batch * triton.cdiv(tokens, 2)
        if use_slab_radix:
            from .triton_gvr2 import launch_topk
        else:
            launch_topk = _launch_streaming_topk
        with torch.cuda.device(q.device):
            for start in range(0, total_pairs, pairs):
                slab = scores[: min(pairs, total_pairs - start)]
                launch_mxfp8_scores(
                    q, k, weights, q_scale, k_scale, slab, start, causal, compress_ratio, wide
                )
                launch_topk(slab, output, start, causal, compress_ratio, wide)
        return output
    bh = max(16, triton.next_power_of_2(heads))
    bd = max(16, triton.next_power_of_2(dim))
    keep = max(128, triton.next_power_of_2(topk))
    # Wider key tiles amortize the O(KEEP) merge (2.4-3.8x for K >= 512 on GB200).
    bn = 128
    while bn < keep and (2 * bn + bh) * bd * q.element_size() <= _TILE_SMEM_BUDGET:
        bn *= 2
    with torch.cuda.device(q.device):
        q_desc = TensorDescriptor.from_tensor(q, [1, 1, bh, bd])
        k_desc = TensorDescriptor.from_tensor(k, [1, bn, bd])
        _index_kernel[(tokens, batch)](
            q_desc,
            k_desc,
            weights,
            output,
            cu_seqlens,
            cu_seqlens_k,
            0 if cu_seqlens is None else cu_seqlens.numel() - 1,
            tokens,
            candidates,
            heads,
            dim,
            topk,
            *weights.stride(),
            causal,
            compress_ratio,
            bh,
            bd,
            bn,
            keep,
            requires_int64_offsets(q, k, weights, output, cu_seqlens, cu_seqlens_k),
            cu_seqlens is not None,
            num_warps=8,
            num_stages=1,
        )
    return output


# A single document folds the document search away (1.4 -> 1.2 us on GB200), so num_documents
# keeps its == 1 specialization but not the multiple-of-16 one.
@triton.jit(do_not_specialize=["tokens"], do_not_specialize_on_alignment=["num_documents"])
def _candidate_bounds_kernel(
    CuQ,
    CuK,
    Bounds,
    num_documents,
    tokens,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block = tl.program_id(0)
    if USE_INT64_OFFSETS:
        block = block.to(tl.int64)
    query = block * BLOCK + tl.arange(0, BLOCK)
    start, end = _packed_interval(CuQ, CuK, query, num_documents, CAUSAL, RATIO, USE_INT64_OFFSETS)
    tl.store(Bounds + query * 2, start, query < tokens)
    tl.store(Bounds + query * 2 + 1, end, query < tokens)


def prepare_candidate_bounds(
    cu_seqlens: Tensor,
    cu_seqlens_k: Tensor,
    tokens: int,
    causal: bool,
    compress_ratio: int,
) -> Tensor:
    bounds = cu_seqlens.new_empty((tokens, 2))
    if tokens == 0:
        return bounds
    with torch.cuda.device(cu_seqlens.device):
        _candidate_bounds_kernel[(triton.cdiv(tokens, 256),)](
            cu_seqlens,
            cu_seqlens_k,
            bounds,
            cu_seqlens.numel() - 1,
            tokens,
            causal,
            compress_ratio,
            requires_int64_offsets(cu_seqlens, cu_seqlens_k, bounds),
            256,
            num_warps=4,
        )
    return bounds
