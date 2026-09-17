"""Memory-bounded TMA/tensor-core indexer for Hopper and newer NVIDIA GPUs.

Each CTA owns one query and retains only its running Top-K while streaming key
tiles. The tensor-core product is candidate-by-head so the head reduction leaves
candidate indices distributed across warps, rather than replicating selection
across head warps. FP16/BF16 store no global scores or partial selections.
Native MXFP8 uses a separate scorer and streaming selector with a reusable score
slab of at most 1024 rows and 32 MiB.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.compiler import CompiledKernel
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets

from ..validation import validate_precision
from .common import score_strides, score_workspace_pairs

# Measured shared memory is (2 * BN + BH) * BD * elem_size: Triton double-buffers the TMA key
# tile even at num_stages=1, plus one Q tile. Hopper/Blackwell allow 227 KiB per block; 192 KiB
# leaves headroom for reduction scratch and only bounds how far the key tile widens.
_TILE_SMEM_BUDGET = 192 * 1024


@triton.jit
def _index_kernel(
    Q,
    K,
    W,
    Out,
    T: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    WB: tl.constexpr,
    WT: tl.constexpr,
    WH: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    KEEP: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Stream TMA key tiles through tensor cores and a register-resident Top-K."""
    batch = tl.program_id(1)
    query = tl.program_id(0)
    if WIDE:
        batch = batch.to(tl.int64)
        query = query.to(tl.int64)
    heads = tl.arange(0, BH)
    if WIDE:
        heads = heads.to(tl.int64)
    weights = tl.load(W + ptr_offset((batch, query, heads), (WB, WT, WH)), heads < H, 0).to(
        tl.float32
    )
    # TMA coordinates are int32; the descriptor handles wide byte strides.
    q = Q.load([batch.to(tl.int32), query.to(tl.int32), 0, 0]).reshape(BH, BD)
    best = tl.full((KEEP,), -9223372036854775808, tl.int64)
    rank = tl.arange(0, KEEP)
    end = (query + 1) // RATIO if CAUSAL else S
    for start in tl.range(0, end, BN):
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
    indices = tl.where(best == -9223372036854775808, -1, indices)
    tl.store(Out + ptr_offset((batch, query, rank), (T * TOPK, TOPK, 1)), indices, rank < TOPK)


@triton.jit
def _mxfp8_score_kernel(
    Q,
    K,
    W,
    QS,
    KS,
    Scores,
    PairStart,
    T: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    W_STRIDES: tl.constexpr,
    QS_STRIDES: tl.constexpr,
    KS_STRIDES: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    BN: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.program_id(1) * BN
    if WIDE:
        row = row.to(tl.int64)
        PairStart = PairStart.to(tl.int64)
    pair = PairStart + row // 2
    batch = pair // tl.cdiv(T, 2)
    query = 2 * (pair % tl.cdiv(T, 2)) + row % 2
    end = (query + 1) // RATIO if CAUSAL else S
    if (query < T) & (start < end):
        h = tl.arange(0, H)
        d = tl.arange(0, D)
        g = tl.arange(0, D // 32)
        n = start + tl.arange(0, BN)
        if WIDE:
            h = h.to(tl.int64)
            d = d.to(tl.int64)
            g = g.to(tl.int64)
            n = n.to(tl.int64)
        q = tl.load(Q + ptr_offset((batch, query, h[:, None], d[None, :]), Q_STRIDES))
        qs = tl.load(QS + ptr_offset((batch, query, h[:, None], g[None, :]), QS_STRIDES))
        w = tl.load(W + ptr_offset((batch, query, h), W_STRIDES)).to(tl.float32)
        k = tl.load(
            K + ptr_offset((batch, n[:, None], d[None, :]), K_STRIDES), n[:, None] < end, 0.0
        )
        ks = tl.load(
            KS + ptr_offset((batch, n[:, None], g[None, :]), KS_STRIDES), n[:, None] < end, 127
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
        pair_start,
        q.shape[1],
        k.shape[1],
        q.shape[2],
        q.shape[3],
        score_strides(q),
        score_strides(k),
        score_strides(weights),
        score_strides(q_scale),
        score_strides(k_scale),
        causal,
        compress_ratio,
        128,
        wide,
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _streaming_topk_kernel(
    Scores,
    Out,
    PairStart,
    T: tl.constexpr,
    S: tl.constexpr,
    TOPK: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    KEEP: tl.constexpr,
    WIDE: tl.constexpr,
):
    row = tl.program_id(0)
    if WIDE:
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


def _validate_mxfp8(q: Tensor) -> None:
    """Check the native domain after precision metadata has been validated."""
    if torch.cuda.get_device_capability(q.device) not in ((10, 0), (10, 3)):
        raise ValueError("Native MXFP8 Triton indexing requires SM100 or SM103.")
    if q.shape[2] not in (32, 64) or q.shape[3] != 128:
        raise ValueError("MXFP8 Triton indexing requires H in {32, 64} and D=128.")


def launch(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
) -> Tensor:
    """Return nondifferentiable INT32 indices without a quadratic workspace.

    Q/K require contiguous last dimensions and 16-byte-aligned bases/outer
    strides. FP16/BF16, H/D <= 256, T <= 2**20, and any Top-K <= S are supported; the
    register-resident selection makes each key tile cost grow with Top-K.
    MXFP8 (SM100/SM103, H in {32, 64}, D=128) uses a separate kernel that accepts
    arbitrary nonnegative strides.
    Gradient-requiring inputs are allowed: selection returns indices, not trainable scores.
    The public registered operator keeps host descriptors outside graph tracing.
    """
    if torch.version.hip or not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] < 9:
        raise ValueError("The Triton TMA indexer requires Hopper or newer NVIDIA GPUs.")
    validate_precision(q, k, weights, q_scale, k_scale)
    scaled_fp8 = q.dtype == torch.float8_e4m3fn
    if scaled_fp8:
        _validate_mxfp8(q)
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
    if candidates != tokens // compress_ratio:
        raise ValueError(
            f"k must hold T // compress_ratio candidates, got T={tokens}, S={candidates}, "
            f"compress_ratio={compress_ratio}."
        )
    if scaled_fp8:
        output = torch.empty((batch, tokens, topk), dtype=torch.int32, device=q.device)
        if topk == 0:
            return output
        assert q_scale is not None and k_scale is not None
        pairs = score_workspace_pairs(batch, tokens, candidates)
        scores = torch.empty((pairs, 2, candidates), dtype=torch.float32, device=q.device)
        wide = requires_int64_offsets(q, k, weights, q_scale, k_scale, scores, output)
        total_pairs = batch * triton.cdiv(tokens, 2)
        with torch.cuda.device(q.device):
            for start in range(0, total_pairs, pairs):
                slab = scores[: min(pairs, total_pairs - start)]
                launch_mxfp8_scores(
                    q, k, weights, q_scale, k_scale, slab, start, causal, compress_ratio, wide
                )
                _streaming_topk_kernel[(slab.shape[0] * 2,)](
                    slab,
                    output,
                    start,
                    tokens,
                    candidates,
                    topk,
                    causal,
                    compress_ratio,
                    max(128, triton.next_power_of_2(topk)),
                    wide,
                    num_warps=4,
                )
        return output
    for tensor in (q, k):
        if tensor.stride(-1) != 1 or any(s % 8 for s in tensor.stride()[:-1]):
            raise ValueError("TMA requires a contiguous last dimension and 16-byte outer strides.")
        if tensor.data_ptr() % 16:
            raise ValueError("TMA requires 16-byte-aligned Q/K base pointers.")
    output = torch.empty((batch, tokens, topk), dtype=torch.int32, device=q.device)
    if topk == 0:
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
            requires_int64_offsets(q, k, weights, output),
            num_warps=8,
            num_stages=1,
        )
    return output
