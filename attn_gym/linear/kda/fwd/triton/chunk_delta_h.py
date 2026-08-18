# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inter-chunk KDA state recurrence.

A single kernel (B=1, K=V=128, 64-token chunks, Blackwell) keeps the full
[K, BV] state in one accumulator and overlaps next-chunk descriptor loads with
the serially dependent state MMAs; 16-bit inputs run the tuned warp-specialized
schedule and FP32 a smaller ordinarily pipelined one. Host-side tensor
descriptors do not survive dynamo/inductor tracing, so the launch sits behind a
compiler-opaque ``torch.library`` op pair.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import can_use_tma, ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, load_ragged_chunk_work
from attn_gym.linear.kda.ops import delta_h_op as _delta_h_op
from attn_gym.linear.kda.ops import delta_h_with_state_op as _delta_h_with_state_op
from attn_gym.linear.kda.utils import exp2


@triton.jit(do_not_specialize=["T"])
def chunk_delta_h_kernel_k128_wsp(
    k_desc,
    w_desc,
    u_desc,
    vnew_desc,
    h_desc,
    decay_desc,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    k,
    w,
    u,
    v_new,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Warp-specialized K=128 inter-chunk recurrence.

    Holds the full [K, BV] state in one accumulator and walks full chunks with
    a warp-specialized descriptor loop. A partial tail chunk uses masked
    pointers because a descriptor load would cross the sequence boundary in
    the token-major packed layout. ``decay_desc`` holds precomputed ``exp2``
    of each chunk's last-row cumulative gate. Descriptor coordinates are
    element indices and stay int32; the raw-pointer tail and state paths
    promote to int64 when tensor sizes require it. Compute follows the
    blockdim64 kernel it replaced: dots run in the input dtype with FP32
    accumulation, and the state stays FP32.
    """
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if USE_INT64_OFFSETS:
        i_state = i_nh.to(tl.int64)
    else:
        i_state = i_nh
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
        T = eos - bos
    else:
        bos = i_n * T
        boh = i_n * tl.cdiv(T, BT)
    NT = tl.cdiv(T, BT)
    NT_full = T // BT

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    b_h = tl.zeros([K, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + i_state * K * V + ptr_offset((o_k[:, None], o_v[None, :]), (V, 1))).to(
            tl.float32
        )

    for i_t in tl.range(0, NT_full, warp_specialize=WARP_SPECIALIZE, num_stages=NUM_STAGES):
        chunk = boh + i_t
        tok = bos + i_t * BT
        h_desc.store(
            [0, chunk, i_h, 0, i_v * BV],
            tl.reshape(b_h.to(k.dtype.element_ty), [1, 1, 1, K, BV]),
        )
        b_w = tl.reshape(w_desc.load([0, tok, i_h, 0]), [BT, K])
        b_u = tl.reshape(u_desc.load([0, tok, i_h, i_v * BV]), [BT, BV])
        b_vnew = b_u.to(tl.float32) - tl.dot(b_w, b_h.to(k.dtype.element_ty))
        vnew_desc.store(
            [0, tok, i_h, i_v * BV],
            tl.reshape(b_vnew.to(k.dtype.element_ty), [1, BT, 1, BV]),
        )
        b_decay = tl.reshape(decay_desc.load([chunk, i_h, 0]), [K])
        b_h = b_h * b_decay[:, None]
        b_k = tl.reshape(k_desc.load([0, tok, i_h, 0]), [BT, K])
        b_h = tl.dot(tl.permute(b_k, [1, 0]), b_vnew.to(k.dtype.element_ty), acc=b_h)

    if NT_full < NT:
        # Peeled partial tail chunk: masked pointers keep the loads and the
        # v_new store inside this sequence's tokens.
        chunk = boh + NT_full
        h_desc.store(
            [0, chunk, i_h, 0, i_v * BV],
            tl.reshape(b_h.to(k.dtype.element_ty), [1, 1, 1, K, BV]),
        )
        o_t = bos + NT_full * BT + tl.arange(0, BT)
        m_t = o_t < bos + T
        if USE_INT64_OFFSETS:
            o_t = o_t.to(tl.int64)
        b_w = tl.load(
            w + ptr_offset((o_t[:, None], i_h, o_k[None, :]), (H * K, K, 1)),
            mask=m_t[:, None],
            other=0.0,
        )
        b_u = tl.load(
            u + ptr_offset((o_t[:, None], i_h, o_v[None, :]), (H * V, V, 1)),
            mask=m_t[:, None],
            other=0.0,
        )
        b_vnew = b_u.to(tl.float32) - tl.dot(b_w, b_h.to(k.dtype.element_ty))
        tl.store(
            v_new + ptr_offset((o_t[:, None], i_h, o_v[None, :]), (H * V, V, 1)),
            b_vnew.to(k.dtype.element_ty),
            mask=m_t[:, None],
        )
        b_decay = tl.reshape(decay_desc.load([chunk, i_h, 0]), [K])
        b_h = b_h * b_decay[:, None]
        b_k = tl.load(
            k + ptr_offset((o_t[:, None], i_h, o_k[None, :]), (H * K, K, 1)),
            mask=m_t[:, None],
            other=0.0,
        )
        b_h = tl.dot(tl.permute(b_k, [1, 0]), b_vnew.to(k.dtype.element_ty), acc=b_h)

    if STORE_FINAL_STATE:
        tl.store(
            ht + i_state * K * V + ptr_offset((o_k[:, None], o_v[None, :]), (V, 1)),
            b_h,
        )


@triton.jit(do_not_specialize=["num_sequences"])
def _chunk_decay_last_kernel(
    gk,
    decay,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Materialize exp2 of each chunk's last-row cumulative gate in one launch."""
    global_chunk, i_h = tl.program_id(0), tl.program_id(1)
    if IS_VARLEN:
        if global_chunk >= tl.load(chunk_offsets + num_sequences):
            return
        _, _, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            global_chunk,
            num_sequences,
            BT,
        )
        last_idx = token_start + valid_tokens - 1
    else:
        last_idx = global_chunk * BT + BT - 1
    o_k = tl.arange(0, K)
    b_g = tl.load(gk + last_idx.to(tl.int64) * H * K + i_h * K + o_k)
    tl.store(decay + global_chunk.to(tl.int64) * H * K + i_h * K + o_k, exp2(b_g))


def _chunk_decay_last(
    gk: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    chunks: int,
    chunk_size: int,
) -> torch.Tensor:
    """Per-chunk exp2 last-row decay factors for the recurrence kernel.

    Kept as a Triton kernel so exp2 stays bitwise-identical to the previous
    in-loop decay; torch.exp2 makes no such guarantee.
    """
    _, _, heads, key_dim = gk.shape
    decay = torch.empty(chunks, heads, key_dim, dtype=torch.float32, device=gk.device)
    _chunk_decay_last_kernel[(chunks, heads)](
        gk,
        decay,
        cu_seqlens,
        chunk_offsets,
        num_sequences=0 if cu_seqlens is None else cu_seqlens.shape[0] - 1,
        H=heads,
        K=key_dim,
        BT=chunk_size,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=1,
    )
    return decay


# The op pair's schemas, fakes, and dispatch registrations live in
# attn_gym.linear.kda.ops; this module provides the CUDA entry points.
_CHUNK_SIZE = 64
_BLOCK_VALUE_DIM = 64


def _delta_h_launch(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    capacity: int,
    final_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate outputs and launch the recurrence; runs eagerly inside the op."""
    batch, tokens, heads, key_dim = k.shape
    value_dim = u.shape[-1]
    h = k.new_empty(batch, capacity, heads, key_dim, value_dim)
    v_new = torch.empty_like(u)
    decay = _chunk_decay_last(gk, cu_seqlens, chunk_offsets, capacity, _CHUNK_SIZE)
    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    # FP32 tiles double the descriptor staging past the shared-memory limit
    # at the 16-bit tile shape, and warp specialization pins its own stage
    # count; for 4-byte inputs halve the value tile and use an ordinarily
    # pipelined loop instead. 16-bit inputs keep the tuned contract schedule.
    use_16bit_config = k.element_size() == 2
    block_value_dim = _BLOCK_VALUE_DIM if use_16bit_config else _BLOCK_VALUE_DIM // 2
    chunk_delta_h_kernel_k128_wsp[(state_batch * heads, value_dim // block_value_dim)](
        TensorDescriptor.from_tensor(k, [1, _CHUNK_SIZE, 1, key_dim]),
        TensorDescriptor.from_tensor(w, [1, _CHUNK_SIZE, 1, key_dim]),
        TensorDescriptor.from_tensor(u, [1, _CHUNK_SIZE, 1, block_value_dim]),
        TensorDescriptor.from_tensor(v_new, [1, _CHUNK_SIZE, 1, block_value_dim]),
        TensorDescriptor.from_tensor(h, [1, 1, 1, key_dim, block_value_dim]),
        TensorDescriptor.from_tensor(decay, [1, 1, key_dim]),
        initial_state,
        final_state,
        cu_seqlens,
        chunk_offsets,
        k,
        w,
        u,
        v_new,
        tokens,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=_CHUNK_SIZE,
        BV=block_value_dim,
        WARP_SPECIALIZE=use_16bit_config,
        NUM_STAGES=3 if use_16bit_config else 2,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        IS_VARLEN=cu_seqlens is not None,
        USE_INT64_OFFSETS=requires_int64_offsets(k, w, u, v_new, h, initial_state, final_state),
        num_warps=4,
    )
    return h, v_new


def _delta_h_cuda(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _delta_h_launch(k, w, u, gk, initial_state, cu_seqlens, chunk_offsets, capacity, None)


def _delta_h_with_state_cuda(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_batch = k.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    final_state = torch.empty(
        state_batch, k.shape[2], k.shape[3], u.shape[-1], dtype=torch.float32, device=k.device
    )
    h, v_new = _delta_h_launch(
        k, w, u, gk, initial_state, cu_seqlens, chunk_offsets, capacity, final_state
    )
    return h, v_new, final_state


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    chunk_size: int = 64,
    output_final_state: bool = True,
    metadata: RaggedChunkMetadata | None = None,
    autotune: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the fixed-length or packed inter-chunk KDA state recurrence.

    ``autotune`` is accepted for launcher-ABI parity with the other stages;
    the warp-specialized kernel has a single fixed configuration, so pinned
    and autotuned launches are identical.
    """
    del autotune
    batch, tokens, heads, key_dim = k.shape
    value_dim = u.shape[-1]
    if metadata is None:
        cu_seqlens = None
        chunk_offsets = None
        chunks = tokens // chunk_size
    else:
        metadata.validate_chunk_size(chunk_size)
        cu_seqlens = metadata.cu_seqlens
        chunk_offsets = metadata.chunk_offsets
        chunks = metadata.capacity
    if tokens % chunk_size and metadata is None:
        raise ValueError(
            f"the inter-chunk state recurrence requires complete chunks, got T={tokens}"
        )
    if (key_dim, value_dim, chunk_size) != (128, 128, _CHUNK_SIZE):
        raise ValueError(
            "the inter-chunk state recurrence requires K=V=128 with 64-token chunks, "
            f"got K={key_dim}, V={value_dim}, chunk_size={chunk_size}"
        )
    if batch != 1:
        raise ValueError("the inter-chunk state recurrence requires batch size one")
    if not k.is_cuda:
        raise ValueError("the inter-chunk state recurrence requires CUDA tensors")
    if k.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(
            "the inter-chunk state recurrence requires bfloat16, float16, or float32 inputs"
        )
    if not (k.dtype == w.dtype == u.dtype):
        raise TypeError("the inter-chunk state recurrence requires matching k, w, and u dtypes")
    if w.shape != k.shape or gk.shape != k.shape:
        raise ValueError("k, w, and gk must have the same shape")
    if u.shape != (batch, tokens, heads, value_dim):
        raise ValueError("u must have shape [B, T, H, V]")
    if torch.cuda.get_device_capability(k.device)[0] < 10:
        raise ValueError("the inter-chunk state recurrence requires CUDA capability 10.0 or newer")
    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    expected_state_shape = (state_batch, heads, key_dim, value_dim)
    if initial_state is not None:
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"initial_state must have shape {expected_state_shape}, "
                f"got {tuple(initial_state.shape)}"
            )
        initial_state = initial_state.contiguous()

    if tokens == 0:
        # Zero-size tensors cannot back descriptors; the recurrence is empty
        # and the final state is the (possibly zero) initial state.
        h = k.new_empty(batch, chunks, heads, key_dim, value_dim)
        final_state = None
        if output_final_state:
            final_state = (
                initial_state.float().clone()
                if initial_state is not None
                else torch.zeros(expected_state_shape, dtype=torch.float32, device=k.device)
            )
        return h, torch.empty_like(u), final_state

    if not all(can_use_tma(t) for t in (k, w, u)):
        raise ValueError(
            "the inter-chunk state recurrence requires 16-byte-aligned, "
            "last-dimension-contiguous k, w, and u"
        )

    if output_final_state:
        h, v_new, final_state = _delta_h_with_state_op(
            k, w, u, gk, initial_state, cu_seqlens, chunk_offsets, chunks
        )
        return h, v_new, final_state
    h, v_new = _delta_h_op(k, w, u, gk, initial_state, cu_seqlens, chunk_offsets, chunks)
    return h, v_new, None


__all__ = ["chunk_gated_delta_rule_fwd_h"]
