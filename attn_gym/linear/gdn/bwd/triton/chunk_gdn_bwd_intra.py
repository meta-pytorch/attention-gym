"""Numerically safe scalar-GDN intra-factor VJP."""

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
def chunk_gdn_bwd_intra_kernel(
    q,
    k,
    cumulative_gate,
    beta,
    d_aqk,
    d_akk,
    d_gate_raw,
    d_q,
    d_k,
    d_beta,
    d_gate,
    cu_seqlens,
    chunk_offsets,
    T,
    num_sequences,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Differentiate scalar-decayed QK/KK factors for one chunk and head."""
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

    gate = tl.load(cumulative_gate + token * H + head, mask=token_mask, other=0.0).to(tl.float32)
    gate_delta = gate[:, None] - gate[None, :]
    causal = (
        (row[:, None] >= column[None, :])
        & (row[:, None] < valid_tokens)
        & (column[None, :] < valid_tokens)
    )
    strict = causal & (row[:, None] > column[None, :])
    decay = tl.where(causal, tl.exp2(tl.where(causal, gate_delta, 0.0)), 0.0)

    matrix_offset = token[:, None] * (H * BT) + head * BT + column[None, :]
    d_aqk_tile = tl.load(d_aqk + matrix_offset, mask=token_mask[:, None], other=0.0).to(tl.float32)
    d_aqk_tile = tl.where(causal, d_aqk_tile, 0.0)
    d_akk_tile = tl.load(d_akk + matrix_offset, mask=token_mask[:, None], other=0.0).to(tl.float32)
    d_akk_tile = tl.where(strict, d_akk_tile, 0.0)
    beta_row = tl.load(beta + token * H + head, mask=token_mask, other=0.0).to(tl.float32)
    aq_weight = d_aqk_tile * decay
    kk_weight = d_akk_tile * decay * beta_row[:, None]

    raw_qk = tl.zeros((BT, BT), dtype=tl.float32)
    raw_kk = tl.zeros((BT, BT), dtype=tl.float32)
    raw_gate_grad = tl.zeros((BT,), dtype=tl.float32)
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
        raw_qk += tl.dot(q_tile, tl.trans(k_tile))
        raw_kk += tl.dot(k_tile, tl.trans(k_tile))
        gate_tile = tl.load(
            d_gate_raw + token[:, None] * (H * K) + head * K + feature[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        raw_gate_grad += tl.sum(gate_tile, axis=1)

        d_q_tile = tl.dot(aq_weight.to(q_tile.dtype), k_tile)
        d_k_tile = tl.dot(tl.trans(aq_weight).to(q_tile.dtype), q_tile)
        d_k_tile += tl.dot(kk_weight.to(q_tile.dtype), k_tile)
        d_k_tile += tl.dot(tl.trans(kk_weight).to(q_tile.dtype), k_tile)
        output_offset = token[:, None] * (H * K) + head * K + feature[None, :]
        tl.store(d_q + output_offset, d_q_tile.to(d_q.dtype.element_ty), mask=token_mask[:, None])
        tl.store(d_k + output_offset, d_k_tile.to(d_k.dtype.element_ty), mask=token_mask[:, None])

    d_beta_row = tl.sum(d_akk_tile * decay * raw_kk, axis=1)
    gate_term = aq_weight * raw_qk + kk_weight * raw_kk
    # Both sources are ln2-free natural-exponent contributions, so this reverse
    # cumsum directly returns the gradient of the per-token natural-log gate.
    d_gate_row = tl.cumsum(
        raw_gate_grad + tl.sum(gate_term, axis=1) - tl.sum(gate_term, axis=0),
        axis=0,
        reverse=True,
    )
    tl.store(d_beta + token * H + head, d_beta_row, mask=token_mask)
    tl.store(d_gate + token * H + head, d_gate_row, mask=token_mask)


def chunk_gdn_bwd_intra_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    d_aqk: torch.Tensor,
    d_akk: torch.Tensor,
    d_gate_raw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run dense B=1 BT64 scalar-GDN intra-factor gradients."""
    batch, tokens, heads, key_dim = q.shape
    if batch != 1 or tokens % 64 or key_dim != 128:
        raise ValueError(
            "dense fused chunk GDN backward requires B=1, complete BT64 chunks, and K=128"
        )
    if k.shape != q.shape or cumulative_gate.shape != q.shape[:3]:
        raise ValueError("k must match q and cumulative_gate must have shape [B,T,H]")
    expected_factor_shape = (batch, tokens, heads, 64)
    if d_aqk.shape != expected_factor_shape or d_akk.shape != expected_factor_shape:
        raise ValueError(f"factor gradients must have shape {expected_factor_shape}")
    if d_gate_raw.shape != q.shape:
        raise ValueError("d_gate_raw must match expanded q")

    d_q = torch.empty_like(q, dtype=torch.float32)
    d_k = torch.empty_like(k, dtype=torch.float32)
    d_beta = torch.empty_like(beta, dtype=torch.float32)
    d_gate = torch.empty_like(cumulative_gate, dtype=torch.float32)
    chunk_gdn_bwd_intra_kernel[(tokens // 64, heads)](
        q,
        k,
        cumulative_gate,
        beta,
        d_aqk,
        d_akk,
        d_gate_raw,
        d_q,
        d_k,
        d_beta,
        d_gate,
        None,
        None,
        tokens,
        0,
        H=heads,
        K=key_dim,
        BT=64,
        BK=64,
        num_warps=4,
        num_stages=2,
    )
    return d_q, d_k, d_beta, d_gate


def chunk_gdn_bwd_intra_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    d_aqk: torch.Tensor,
    d_akk: torch.Tensor,
    d_gate_raw: torch.Tensor,
    metadata: RaggedChunkMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fixed-capacity packed scalar-GDN intra-factor gradients."""
    metadata.validate_chunk_size(64)
    batch, tokens, heads, key_dim = q.shape
    if batch != 1 or key_dim != 128 or k.shape != q.shape:
        raise ValueError("packed fused chunk GDN backward requires B=1 and K=128")
    expected_factor_shape = (batch, tokens, heads, 64)
    if d_aqk.shape != expected_factor_shape or d_akk.shape != expected_factor_shape:
        raise ValueError(f"factor gradients must have shape {expected_factor_shape}")
    if d_gate_raw.shape != q.shape:
        raise ValueError("d_gate_raw must match expanded q")

    d_q = torch.zeros_like(q, dtype=torch.float32)
    d_k = torch.zeros_like(k, dtype=torch.float32)
    d_beta = torch.zeros_like(beta, dtype=torch.float32)
    d_gate = torch.zeros_like(cumulative_gate, dtype=torch.float32)
    chunk_gdn_bwd_intra_kernel[(metadata.capacity, heads)](
        q,
        k,
        cumulative_gate,
        beta,
        d_aqk,
        d_akk,
        d_gate_raw,
        d_q,
        d_k,
        d_beta,
        d_gate,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        tokens,
        metadata.cu_seqlens.shape[0] - 1,
        H=heads,
        K=key_dim,
        BT=64,
        BK=64,
        num_warps=4,
        num_stages=2,
    )
    return d_q, d_k, d_beta, d_gate


__all__ = ["chunk_gdn_bwd_intra_dense", "chunk_gdn_bwd_intra_packed"]
