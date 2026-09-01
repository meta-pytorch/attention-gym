"""Dense scalar-gate specialization of the KDA inter-chunk state recurrence."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, load_ragged_sequence_work


@triton.jit(do_not_specialize=["T"])
def chunk_gdn_fwd_recurrence_kernel(
    k_desc,
    w_desc,
    u_desc,
    v_new_desc,
    h_desc,
    cumulative_gate,
    initial_state_desc,
    final_state_desc,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """Keep one KxBV state tile resident while traversing BT64 chunks."""
    batch_head = tl.program_id(0)
    value_tile = tl.program_id(1)
    batch = batch_head // H
    head = batch_head % H
    state_vk = initial_state_desc.load([batch, head, value_tile * BV, 0])
    state = tl.trans(tl.reshape(state_vk, [BV, K])).to(tl.float32)
    for chunk in tl.range(0, T // BT, warp_specialize=True, num_stages=3):
        token = chunk * BT
        h_desc.store(
            [batch, chunk, head, 0, value_tile * BV],
            tl.reshape(state.to(k_desc.dtype), [1, 1, 1, K, BV]),
        )
        w = tl.reshape(w_desc.load([batch, token, head, 0]), [BT, K])
        u = tl.reshape(u_desc.load([batch, token, head, value_tile * BV]), [BT, BV])
        v_new = u.to(tl.float32) - tl.dot(w, state.to(w.dtype))
        v_new_desc.store(
            [batch, token, head, value_tile * BV],
            tl.reshape(v_new.to(u.dtype), [1, BT, 1, BV]),
        )

        final_gate = tl.load(
            cumulative_gate + ptr_offset((batch * T + token + BT - 1, head), (H, 1))
        ).to(tl.float32)
        state *= tl.exp2(final_gate)
        restored_key = tl.reshape(k_desc.load([batch, token, head, 0]), [BT, K])
        state = tl.dot(tl.trans(restored_key), v_new.to(restored_key.dtype), acc=state)

    final_state_desc.store(
        [batch, head, value_tile * BV, 0],
        tl.reshape(tl.trans(state), [1, 1, BV, K]),
    )


def chunk_gdn_fwd_recurrence_dense(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    cumulative_gate: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run dense BT64 recurrence with precomputed restored keys."""
    batch, tokens, heads, key_dim = k.shape
    value_dim = u.shape[-1]
    if tokens % 64 or (key_dim, value_dim) != (128, 128):
        raise ValueError("dense fused chunk GDN requires complete BT64 chunks and K=V=128")
    if w.shape != k.shape or cumulative_gate.shape != k.shape[:3]:
        raise ValueError("w must match k and cumulative_gate must have shape [B,T,H]")
    if initial_state.shape != (batch, heads, value_dim, key_dim):
        raise ValueError("initial_state must have shape [B,H,V,K]")

    chunks = tokens // 64
    block_value = 64
    h = torch.empty(batch, chunks, heads, key_dim, value_dim, dtype=k.dtype, device=k.device)
    v_new = torch.empty_like(u)
    final_state = torch.empty_like(initial_state)
    chunk_gdn_fwd_recurrence_kernel[(batch * heads, value_dim // block_value)](
        TensorDescriptor.from_tensor(k, [1, 64, 1, key_dim]),
        TensorDescriptor.from_tensor(w, [1, 64, 1, key_dim]),
        TensorDescriptor.from_tensor(u, [1, 64, 1, block_value]),
        TensorDescriptor.from_tensor(v_new, [1, 64, 1, block_value]),
        TensorDescriptor.from_tensor(h, [1, 1, 1, key_dim, block_value]),
        cumulative_gate,
        TensorDescriptor.from_tensor(initial_state, [1, 1, block_value, key_dim]),
        TensorDescriptor.from_tensor(final_state, [1, 1, block_value, key_dim]),
        tokens,
        H=heads,
        K=key_dim,
        BT=64,
        BV=block_value,
        num_warps=4,
        num_stages=3,
    )
    return h, v_new, final_state


@triton.jit
def chunk_gdn_fwd_recurrence_packed_kernel(
    restored_k,
    w,
    u,
    cumulative_gate,
    initial_state,
    h,
    v_new,
    final_state,
    cu_seqlens,
    chunk_offsets,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """Traverse one packed sequence/head/value tile with a resident FP32 state."""
    sequence_head = tl.program_id(0)
    value_tile = tl.program_id(1)
    sequence = sequence_head // H
    head = sequence_head % H

    bos, eos, chunk_begin = load_ragged_sequence_work(cu_seqlens, chunk_offsets, sequence)
    bos, eos, chunk_begin = bos.to(tl.int64), eos.to(tl.int64), chunk_begin.to(tl.int64)
    chunks = (eos - bos + BT - 1) // BT
    key = tl.arange(0, K)
    value = value_tile * BV + tl.arange(0, BV)
    state_offset = ptr_offset(
        (sequence, head, value[None, :], key[:, None]), (H * V * K, V * K, K, 1)
    )
    state = tl.load(initial_state + state_offset).to(tl.float32)
    row = tl.arange(0, BT)

    for local_chunk in tl.range(0, chunks):
        global_chunk = chunk_begin + local_chunk
        token_start = bos + local_chunk * BT
        token = token_start + row
        token_mask = token < eos
        h_offset = ptr_offset(
            (global_chunk, head, key[:, None], value[None, :]), (H * K * V, K * V, V, 1)
        )
        tl.store(h + h_offset, state.to(h.dtype.element_ty))

        w_offset = ptr_offset((token[:, None], head, key[None, :]), (H * K, K, 1))
        u_offset = ptr_offset((token[:, None], head, value[None, :]), (H * V, V, 1))
        w_tile = tl.load(w + w_offset, mask=token_mask[:, None], other=0.0)
        u_tile = tl.load(u + u_offset, mask=token_mask[:, None], other=0.0)
        corrected = u_tile.to(tl.float32) - tl.dot(w_tile, state.to(w_tile.dtype))
        tl.store(
            v_new + u_offset,
            corrected.to(v_new.dtype.element_ty),
            mask=token_mask[:, None],
        )

        final_token = tl.minimum(token_start + BT, eos) - 1
        final_gate = tl.load(cumulative_gate + ptr_offset((final_token, head), (H, 1))).to(
            tl.float32
        )
        state *= tl.exp2(final_gate)
        restored = tl.load(restored_k + w_offset, mask=token_mask[:, None], other=0.0)
        state = tl.dot(tl.trans(restored), corrected.to(restored.dtype), acc=state)

    tl.store(final_state + state_offset, state)


def chunk_gdn_fwd_recurrence_packed(
    restored_k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    cumulative_gate: torch.Tensor,
    initial_state: torch.Tensor,
    metadata: RaggedChunkMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fixed-capacity packed recurrence with empty-sequence state identity."""
    metadata.validate_chunk_size(64)
    batch, tokens, heads, key_dim = restored_k.shape
    value_dim = u.shape[-1]
    num_sequences = metadata.cu_seqlens.shape[0] - 1
    if batch != 1 or tokens == 0 or (key_dim, value_dim) != (128, 128):
        raise ValueError("packed fused chunk GDN recurrence requires B=1, T>0, and K=V=128")
    expected_state = (num_sequences, heads, value_dim, key_dim)
    if initial_state.shape != expected_state:
        raise ValueError(f"initial_state must have shape {expected_state}")

    h = restored_k.new_empty(1, metadata.capacity, heads, key_dim, value_dim)
    v_new = torch.empty_like(u)
    final_state = torch.empty_like(initial_state)
    block_value = 64
    chunk_gdn_fwd_recurrence_packed_kernel[(num_sequences * heads, value_dim // block_value)](
        restored_k,
        w,
        u,
        cumulative_gate,
        initial_state,
        h,
        v_new,
        final_state,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=64,
        BV=block_value,
        num_warps=4,
        num_stages=2,
    )
    return h, v_new, final_state


__all__ = ["chunk_gdn_fwd_recurrence_dense", "chunk_gdn_fwd_recurrence_packed"]
