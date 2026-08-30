"""Backward-only recomputation for scalar chunk GDN."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "num_sequences"])
def chunk_gdn_recompute_aqk_kernel(
    q,
    k,
    cumulative_gate,
    aqk,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    num_sequences,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Recompute causal scalar-decayed QK without reciprocal gate factors."""
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    row = tl.arange(0, BT)
    column = tl.arange(0, BT)
    if IS_VARLEN:
        if chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        _sequence, _local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens, chunk_offsets, chunk, num_sequences, BT
        )
        token = token_start + row
        token_mask = row < valid_tokens
    else:
        token = chunk * BT + row
        token_mask = row < BT
        valid_tokens = BT

    qk = tl.zeros((BT, BT), dtype=tl.float32)
    for key_block in range(0, K, BK):
        feature = key_block + tl.arange(0, BK)
        q_tile = tl.load(
            q + token[:, None] * (H * K) + head * K + feature[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        k_tile = tl.load(
            k + token[:, None] * (H * K) + head * K + feature[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        qk += tl.dot(q_tile, tl.trans(k_tile))

    gate = tl.load(cumulative_gate + token * H + head, mask=token_mask, other=0.0).to(tl.float32)
    gate_delta = gate[:, None] - gate[None, :]
    causal = (
        (row[:, None] >= column[None, :])
        & (row[:, None] < valid_tokens)
        & (column[None, :] < valid_tokens)
    )
    decay = tl.where(causal, tl.exp2(tl.where(causal, gate_delta, 0.0)), 0.0)
    output_offset = token[:, None] * (H * BT) + head * BT + column[None, :]
    tl.store(
        aqk + output_offset,
        tl.where(causal, qk * decay * scale, 0.0),
        mask=token_mask[:, None],
    )


def chunk_gdn_recompute_aqk_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run dense B=1 BT64 Aqk recomputation for the reused delta-H backward."""
    batch, tokens, heads, key_dim = q.shape
    if batch != 1 or tokens % 64 or key_dim != 128 or k.shape != q.shape:
        raise ValueError("dense fused chunk GDN Aqk recompute requires B=1, BT64, and K=128")
    aqk = torch.empty(batch, tokens, heads, 64, dtype=q.dtype, device=q.device)
    chunk_gdn_recompute_aqk_kernel[(tokens // 64, heads)](
        q,
        k,
        cumulative_gate,
        aqk,
        None,
        None,
        scale,
        tokens,
        0,
        H=heads,
        K=key_dim,
        BT=64,
        BK=32,
        num_warps=8,
        num_stages=2,
    )
    return aqk


def chunk_gdn_recompute_aqk_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata,
) -> torch.Tensor:
    """Run fixed-capacity packed Aqk recomputation for reused delta-H backward."""
    metadata.validate_chunk_size(64)
    batch, tokens, heads, key_dim = q.shape
    if batch != 1 or key_dim != 128 or k.shape != q.shape:
        raise ValueError("packed fused chunk GDN Aqk recompute requires B=1 and K=128")
    aqk = torch.zeros(batch, tokens, heads, 64, dtype=q.dtype, device=q.device)
    chunk_gdn_recompute_aqk_kernel[(metadata.capacity, heads)](
        q,
        k,
        cumulative_gate,
        aqk,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        scale,
        tokens,
        metadata.cu_seqlens.shape[0] - 1,
        H=heads,
        K=key_dim,
        BT=64,
        BK=32,
        num_warps=8,
        num_stages=2,
    )
    return aqk


__all__ = ["chunk_gdn_recompute_aqk_dense", "chunk_gdn_recompute_aqk_packed"]
