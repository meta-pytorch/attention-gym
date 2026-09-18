"""Bounded-slab scoring and exact GVR2-style selection in Triton.

A 256-element self-sample guesses the crossing score. Full-row counts verify it;
if the guess misses, four byte-radix passes refine the appropriate side exactly.
The sample never discards an unverified candidate. Only the sample is sorted;
scores live in the shared bounded slab (``common.score_workspace_pairs``).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets

from .common import score_strides, score_workspace_pairs


@triton.jit
def _score_kernel(
    Q,
    K,
    W,
    Scores,
    PairStart,
    T: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    Q_STRIDES: tl.constexpr,
    K_STRIDES: tl.constexpr,
    W_STRIDES: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    BH: tl.constexpr,
    BD: tl.constexpr,
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
        h = tl.arange(0, BH)
        d = tl.arange(0, BD)
        n = start + tl.arange(0, BN)
        if WIDE:
            h = h.to(tl.int64)
            d = d.to(tl.int64)
            n = n.to(tl.int64)
        q = tl.load(
            Q + ptr_offset((batch, query, h[:, None], d[None, :]), Q_STRIDES),
            (h[:, None] < H) & (d[None, :] < D),
            0.0,
        )
        k = tl.load(
            K + ptr_offset((batch, n[:, None], d[None, :]), K_STRIDES),
            (n[:, None] < end) & (d[None, :] < D),
            0.0,
        )
        w = tl.load(W + ptr_offset((batch, query, h), W_STRIDES), h < H, 0.0).to(tl.float32)
        dots = tl.dot(k, tl.trans(q))
        scores = tl.sum(tl.maximum(dots, 0.0) * w[None, :], axis=1) * (H * D) ** -0.5
        tl.store(Scores + row * S + n, scores, n < end)


@triton.jit
def _ordinal(values):
    # Unsigned monotonic order, with both signed zeros sharing one tie bucket.
    bits = tl.where(values == 0.0, 0.0, values).to(tl.uint32, bitcast=True)
    return bits ^ tl.where((bits >> 31) != 0, 0xFFFFFFFF, 0x80000000).to(tl.uint32)


@triton.jit
def _gvr2_topk_kernel(
    Scores,
    Out,
    PairStart,
    T: tl.constexpr,
    S: tl.constexpr,
    TOPK: tl.constexpr,
    CAUSAL: tl.constexpr,
    RATIO: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr = 1024,
    SAMPLE: tl.constexpr = 256,
):
    row = tl.program_id(0)
    if WIDE:
        row = row.to(tl.int64)
        PairStart = PairStart.to(tl.int64)
    pair = PairStart + row // 2
    batch = pair // tl.cdiv(T, 2)
    query = 2 * (pair % tl.cdiv(T, 2)) + row % 2
    if query < T:
        end = tl.cast((query + 1) // RATIO if CAUSAL else S, tl.int32)
        out = Out + (batch * T + query) * TOPK
        lane = tl.arange(0, BLOCK)
        if end <= TOPK:
            for start in range(0, TOPK, BLOCK):
                n = start + lane
                tl.store(out + n, tl.where(n < end, n, -1), n < TOPK)
        else:
            source = Scores + row * S
            sample_lane = tl.arange(0, SAMPLE)
            sample_count = tl.minimum(end, SAMPLE)
            sample_index = sample_lane * end // sample_count
            sample = tl.load(source + sample_index, sample_lane < sample_count, float("-inf"))
            sampled_order = tl.sort(_ordinal(sample), descending=True)
            guess_rank = tl.minimum(sample_count - 1, TOPK * sample_count // end)
            guess = tl.sum(tl.where(sample_lane == guess_rank, sampled_order, 0), 0)
            greater = 0
            equal = 0
            for start in range(0, end, BLOCK):
                n = start + lane
                values = _ordinal(tl.load(source + n, n < end, 0.0))
                greater += tl.sum(((values > guess) & (n < end)).to(tl.int32), 0)
                equal += tl.sum(((values == guess) & (n < end)).to(tl.int32), 0)
            boundary = guess
            remaining = TOPK - greater
            if (greater >= TOPK) | (greater + equal < TOPK):
                # A bad guess changes only the exact radix search interval, never correctness.
                search_above = greater >= TOPK
                remaining = tl.where(search_above, TOPK, TOPK - greater - equal)
                prefix = tl.full((), 0, tl.uint32)
                prefix_mask = tl.full((), 0, tl.uint32)
                bins = tl.arange(0, 256)
                for shift in tl.static_range(24, -1, -8):
                    histogram = tl.full((256,), 0, tl.int32)
                    for start in range(0, end, BLOCK):
                        n = start + lane
                        values = _ordinal(tl.load(source + n, n < end, 0.0))
                        side = tl.where(search_above, values > guess, values < guess)
                        active = (n < end) & side & ((values & prefix_mask) == prefix)
                        digit = ((values >> shift) & 255).to(tl.int32)
                        histogram += tl.histogram(tl.where(active, digit, 256), 256)
                    above = tl.sum(histogram, 0) - tl.cumsum(histogram, 0)
                    crossing = (above < remaining) & (above + histogram >= remaining)
                    bucket = tl.max(tl.where(crossing, bins, 0), 0)
                    remaining -= tl.sum(tl.where(bins == bucket, above, 0), 0)
                    prefix |= bucket.to(tl.uint32) << shift
                    prefix_mask |= tl.full((), 255 << shift, tl.uint32)
                boundary = prefix
            # remaining is the exact number of boundary ties still needed. Index-order scans
            # give deterministic tie handling and disjoint output addresses without atomics.
            written = 0
            ties = 0
            for start in range(0, end, BLOCK):
                n = start + lane
                values = _ordinal(tl.load(source + n, n < end, 0.0))
                strict = (n < end) & (values > boundary)
                tied = (n < end) & (values == boundary)
                strict_rank = tl.cumsum(strict.to(tl.int32), 0) - 1 + written
                tie_rank = tl.cumsum(tied.to(tl.int32), 0) - 1 + ties
                tl.store(out + strict_rank, n, strict)
                tl.store(out + TOPK - remaining + tie_rank, n, tied & (tie_rank < remaining))
                written += tl.sum(strict.to(tl.int32), 0)
                ties += tl.sum(tied.to(tl.int32), 0)


def launch(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
) -> Tensor:
    """Return exact finite-score Top-K sets; score order is unspecified.

    FP16/BF16: Hopper+, H/D<=256, D divisible by 8, T<=2**20.
    All operands accept arbitrary nonnegative strides.
    """
    if torch.version.hip or not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] < 9:
        raise ValueError("The Triton GVR2 indexer requires Hopper or newer NVIDIA GPUs.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("Triton GVR2 requires FP16/BF16 inputs.")
    if k.dtype != q.dtype or weights.dtype != q.dtype:
        raise TypeError("q, k, and weights must have one dtype")
    batch, tokens, heads, dim = q.shape
    candidates = k.shape[1]
    if heads > 256 or dim > 256 or dim % 8 or tokens > 2**20:
        raise ValueError("Triton GVR2 requires H/D <= 256, D divisible by 8 and T <= 2**20.")
    if compress_ratio < 1 or candidates != tokens // compress_ratio:
        raise ValueError("k must hold T // compress_ratio candidates with positive ratio.")
    output = torch.empty((batch, tokens, topk), device=q.device, dtype=torch.int32)
    if topk == 0:
        return output
    pairs = score_workspace_pairs(batch, tokens, candidates)
    scores = torch.empty((pairs, 2, candidates), device=q.device, dtype=torch.float32)
    wide = requires_int64_offsets(q, k, weights, scores, output)
    total_pairs = batch * triton.cdiv(tokens, 2)
    with torch.cuda.device(q.device):
        for start in range(0, total_pairs, pairs):
            slab = scores[: min(pairs, total_pairs - start)]
            _score_kernel[(slab.shape[0] * 2, triton.cdiv(candidates, 128))](
                q,
                k,
                weights,
                slab,
                start,
                tokens,
                candidates,
                heads,
                dim,
                score_strides(q),
                score_strides(k),
                score_strides(weights),
                causal,
                compress_ratio,
                max(16, triton.next_power_of_2(heads)),
                max(16, triton.next_power_of_2(dim)),
                128,
                wide,
                num_warps=4,
                num_stages=1,
            )
            _gvr2_topk_kernel[(slab.shape[0] * 2,)](
                slab,
                output,
                start,
                tokens,
                candidates,
                topk,
                causal,
                compress_ratio,
                wide,
                num_warps=4,
                enable_fp_fusion=False,
            )
    return output
