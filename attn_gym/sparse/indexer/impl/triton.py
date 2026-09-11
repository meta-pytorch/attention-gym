"""Memory-bounded TMA/tensor-core indexer for Hopper and newer NVIDIA GPUs.

Each CTA owns one query and retains only its running Top-K while streaming key
tiles. The tensor-core product is candidate-by-head so the head reduction leaves
candidate indices distributed across warps, rather than replicating selection
across head warps. No scores or partial selections are stored in global memory.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets


@triton.jit
def _index_kernel(
    Q,
    K,
    W,
    Out,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOPK: tl.constexpr,
    WB: tl.constexpr,
    WT: tl.constexpr,
    WH: tl.constexpr,
    CAUSAL: tl.constexpr,
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
    end = query + 1 if CAUSAL else T
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


def launch(q: Tensor, k: Tensor, weights: Tensor, topk: int, causal: bool) -> Tensor:
    """Return nondifferentiable INT32 indices without a quadratic workspace.

    Q/K require contiguous last dimensions and 16-byte-aligned bases/outer
    strides. FP16/BF16, H/D <= 256, T <= 2**20, and Top-K <= 512 are supported.
    Inputs and FP32 intermediate scores must be finite. Gradient-requiring
    inputs are allowed: selection returns indices, not trainable scores.
    The public registered operator keeps host descriptors outside graph tracing.
    """
    if torch.version.hip or not q.is_cuda or torch.cuda.get_device_capability(q.device)[0] < 9:
        raise ValueError("The Triton TMA indexer requires Hopper or newer NVIDIA GPUs.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("The Triton indexer requires FP16 or BF16 inputs.")
    batch, tokens, heads, dim = q.shape
    if heads > 256 or dim > 256 or dim % 8 or topk > 512 or tokens > 2**20:
        raise ValueError(
            "The Triton indexer requires H <= 256, D <= 256 divisible by 8, "
            "K <= 512, and T <= 2**20."
        )
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
    bn = 128
    keep = max(bn, triton.next_power_of_2(topk))
    with torch.cuda.device(q.device):
        q_desc = TensorDescriptor.from_tensor(q, [1, 1, bh, bd])
        k_desc = TensorDescriptor.from_tensor(k, [1, bn, bd])
        _index_kernel[(tokens, batch)](
            q_desc,
            k_desc,
            weights,
            output,
            tokens,
            heads,
            dim,
            topk,
            *weights.stride(),
            causal,
            bh,
            bd,
            bn,
            keep,
            requires_int64_offsets(q, k, weights, output),
            num_warps=8,
            num_stages=1,
        )
    return output
