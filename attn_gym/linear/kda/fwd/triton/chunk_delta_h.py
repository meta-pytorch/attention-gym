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
from attn_gym.linear.kda.chunk_scheduler import (
    GridScheduler,
    RaggedChunkMetadata,
    ScheduleKind,
    ScheduleRequest,
    load_ragged_sequence_extent,
)
from attn_gym.linear.kda.ops import delta_h_op as _delta_h_op
from attn_gym.linear.kda.ops import delta_h_paged_op as _delta_h_paged_op
from attn_gym.linear.kda.ops import delta_h_with_state_op as _delta_h_with_state_op
from attn_gym.linear.kda.utils import exp2


@triton.jit
def _run_chunk_delta_h_sequence(
    k_desc,
    w_desc,
    u_desc,
    vnew_desc,
    h_desc,
    gk_desc,
    h0,
    ht,
    state_indices,
    has_initial_state,
    cu_seqlens,
    chunk_offsets,
    k,
    w,
    u,
    v_new,
    T,
    i_nh,
    i_v,
    H0_STRIDES,
    HT_STRIDES,
    DYNAMIC_STATE_LAYOUT: tl.constexpr,
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
    USE_STATE_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Process one sequence, head, and value tile through the state recurrence.

    The full ``[K, BV]`` state stays in one FP32 accumulator while complete
    chunks use tensor descriptors and a partial tail uses masked pointers.
    Descriptor coordinates remain int32; raw-pointer state and tail paths widen
    when the reachable layout requires int64 offsets.
    """
    i_n, i_h = i_nh // H, i_nh % H
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

    if USE_STATE_INDICES:
        i_state = tl.load(state_indices + i_n).to(tl.int64)
        if i_state <= 0:
            for i_t in tl.range(0, NT):
                chunk = boh + i_t
                h_desc.store(
                    [0, chunk, i_h, 0, i_v * BV],
                    tl.reshape(
                        tl.zeros([K, BV], dtype=k.dtype.element_ty),
                        [1, 1, 1, K, BV],
                    ),
                )
                o_t = bos + i_t * BT + tl.arange(0, BT)
                m_t = o_t < bos + T
                if USE_INT64_OFFSETS:
                    o_t = o_t.to(tl.int64)
                tl.store(
                    v_new + ptr_offset((o_t[:, None], i_h, o_v[None, :]), (H * V, V, 1)),
                    0.0,
                    mask=m_t[:, None],
                )
            return
    elif USE_INT64_OFFSETS:
        i_state = i_n.to(tl.int64)
    else:
        i_state = i_n
    if DYNAMIC_STATE_LAYOUT:
        state_indices_4d = (i_state, i_h, o_v[None, :], o_k[:, None])
        p_h0 = ptr_offset(state_indices_4d, H0_STRIDES)
        p_ht = ptr_offset(state_indices_4d, HT_STRIDES)
    else:
        p_h0 = i_state * H * V * K + i_h * V * K
        p_h0 += ptr_offset((o_v[None, :], o_k[:, None]), (K, 1))
        p_ht = p_h0

    b_h = tl.zeros([K, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        m_state = o_v[None, :] < V
        if USE_HAS_INITIAL_STATE:
            m_state &= tl.load(has_initial_state + i_n)
        b_h += tl.load(
            h0 + p_h0,
            mask=m_state,
            other=0.0,
        ).to(
            tl.float32,
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
        b_decay = tl.reshape(gk_desc.load([0, tok + BT - 1, i_h, 0]), [K])
        b_h = b_h * exp2(b_decay)[:, None]
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
        b_decay = tl.reshape(gk_desc.load([0, bos + T - 1, i_h, 0]), [K])
        b_h = b_h * exp2(b_decay)[:, None]
        b_k = tl.load(
            k + ptr_offset((o_t[:, None], i_h, o_k[None, :]), (H * K, K, 1)),
            mask=m_t[:, None],
            other=0.0,
        )
        b_h = tl.dot(tl.permute(b_k, [1, 0]), b_vnew.to(k.dtype.element_ty), acc=b_h)

    if STORE_FINAL_STATE:
        tl.store(
            ht + p_ht,
            b_h,
        )


@triton.jit(do_not_specialize=["T"])
def chunk_delta_h_kernel_k128_wsp(
    k_desc,
    w_desc,
    u_desc,
    vnew_desc,
    h_desc,
    gk_desc,
    h0,
    ht,
    state_indices,
    has_initial_state,
    cu_seqlens,
    chunk_offsets,
    k,
    w,
    u,
    v_new,
    T,
    h0_stride_0,
    h0_stride_1,
    h0_stride_2,
    h0_stride_3,
    ht_stride_0,
    ht_stride_1,
    ht_stride_2,
    ht_stride_3,
    DYNAMIC_STATE_LAYOUT: tl.constexpr,
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
    USE_STATE_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Warp-specialized K=128 inter-chunk recurrence."""
    _run_chunk_delta_h_sequence(
        k_desc,
        w_desc,
        u_desc,
        vnew_desc,
        h_desc,
        gk_desc,
        h0,
        ht,
        state_indices,
        has_initial_state,
        cu_seqlens,
        chunk_offsets,
        k,
        w,
        u,
        v_new,
        T,
        tl.program_id(0),
        tl.program_id(1),
        (h0_stride_0, h0_stride_1, h0_stride_2, h0_stride_3),
        (ht_stride_0, ht_stride_1, ht_stride_2, ht_stride_3),
        DYNAMIC_STATE_LAYOUT,
        H,
        K,
        V,
        BT,
        BV,
        WARP_SPECIALIZE,
        NUM_STAGES,
        USE_INITIAL_STATE,
        STORE_FINAL_STATE,
        IS_VARLEN,
        USE_STATE_INDICES,
        USE_HAS_INITIAL_STATE,
        USE_INT64_OFFSETS,
    )


@triton.jit(do_not_specialize=["T"])
def chunk_delta_h_kernel_k128_persistent(
    k_desc,
    w_desc,
    u_desc,
    vnew_desc,
    h_desc,
    gk_desc,
    h0,
    ht,
    state_indices,
    has_initial_state,
    cu_seqlens,
    chunk_offsets,
    k,
    w,
    u,
    v_new,
    T,
    h0_stride_0,
    h0_stride_1,
    h0_stride_2,
    h0_stride_3,
    ht_stride_0,
    ht_stride_1,
    ht_stride_2,
    ht_stride_3,
    DYNAMIC_STATE_LAYOUT: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    NUM_SEQUENCES: tl.constexpr,
    NUM_WORKERS: tl.constexpr,
):
    """Stride persistent workers over sequence recurrences after the first wave."""
    worker = tl.program_id(0)
    i_v = tl.program_id(1)
    sequence_extent = load_ragged_sequence_extent(cu_seqlens, NUM_SEQUENCES)
    active_tasks = sequence_extent * H
    task_end = NUM_SEQUENCES * H if STORE_FINAL_STATE else active_tasks
    for i_nh in tl.range(NUM_WORKERS + worker, task_end, NUM_WORKERS):
        _run_chunk_delta_h_sequence(
            k_desc,
            w_desc,
            u_desc,
            vnew_desc,
            h_desc,
            gk_desc,
            h0,
            ht,
            state_indices,
            has_initial_state,
            cu_seqlens,
            chunk_offsets,
            k,
            w,
            u,
            v_new,
            T,
            i_nh,
            i_v,
            (h0_stride_0, h0_stride_1, h0_stride_2, h0_stride_3),
            (ht_stride_0, ht_stride_1, ht_stride_2, ht_stride_3),
            DYNAMIC_STATE_LAYOUT,
            H,
            K,
            V,
            BT,
            BV,
            WARP_SPECIALIZE,
            NUM_STAGES,
            USE_INITIAL_STATE,
            STORE_FINAL_STATE,
            True,
            USE_STATE_INDICES,
            USE_HAS_INITIAL_STATE,
            USE_INT64_OFFSETS,
        )


# The op pair's schemas, fakes, and dispatch registrations live in
# attn_gym.linear.kda.ops; this module provides the CUDA entry points.
_CHUNK_SIZE = 64
_BLOCK_VALUE_DIM = 64
_MIN_PERSISTENT_SEQUENCES = 32
_MIN_PERSISTENT_HEADS = 8
_AUTO_PERSISTENT_MAX_AVERAGE_TOKENS = _CHUNK_SIZE


def _persistent_sequence_workers(
    metadata: RaggedChunkMetadata,
    tokens: int,
    heads: int,
    value_tiles: int,
    device: torch.device,
    schedule: ScheduleRequest,
) -> int:
    """Return the sequence-head workers for the selected recurrence schedule."""
    num_sequences = metadata.cu_seqlens.shape[0] - 1
    resolved = GridScheduler(metadata, ctas_per_sm=1).resolve_sequences(
        schedule, heads * value_tiles, device
    )
    if resolved.kind is ScheduleKind.STATIC:
        return 0
    # Longer capacity-average recurrences favor the fully warp-specialized grid;
    # explicit PERSISTENT remains available for targeted tuning experiments.
    if schedule is ScheduleRequest.AUTO and (
        num_sequences < _MIN_PERSISTENT_SEQUENCES
        or heads < _MIN_PERSISTENT_HEADS
        or tokens > num_sequences * _AUTO_PERSISTENT_MAX_AVERAGE_TOKENS
    ):
        return 0
    return max(1, resolved.workers // value_tiles)


def _delta_h_launch(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    capacity: int,
    final_state: torch.Tensor | None,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate outputs and launch the recurrence; runs eagerly inside the op."""
    batch, tokens, heads, key_dim = k.shape
    value_dim = u.shape[-1]
    h = k.new_empty(batch, capacity, heads, key_dim, value_dim)
    v_new = torch.empty_like(u)
    if not can_use_tma(gk):
        gk = gk.clone(memory_format=torch.contiguous_format)
    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    compact_state_strides = (heads * value_dim * key_dim, value_dim * key_dim, key_dim, 1)
    h0_strides = initial_state.stride() if initial_state is not None else compact_state_strides
    ht_strides = final_state.stride() if final_state is not None else compact_state_strides
    # FP32 tiles double the descriptor staging past the shared-memory limit
    # at the 16-bit tile shape, and warp specialization pins its own stage
    # count; for 4-byte inputs halve the value tile and use an ordinarily
    # pipelined loop instead. 16-bit inputs keep the tuned contract schedule.
    use_16bit_config = k.element_size() == 2
    block_value_dim = _BLOCK_VALUE_DIM if use_16bit_config else _BLOCK_VALUE_DIM // 2
    descriptors = (
        TensorDescriptor.from_tensor(k, [1, _CHUNK_SIZE, 1, key_dim]),
        TensorDescriptor.from_tensor(w, [1, _CHUNK_SIZE, 1, key_dim]),
        TensorDescriptor.from_tensor(u, [1, _CHUNK_SIZE, 1, block_value_dim]),
        TensorDescriptor.from_tensor(v_new, [1, _CHUNK_SIZE, 1, block_value_dim]),
        TensorDescriptor.from_tensor(h, [1, 1, 1, key_dim, block_value_dim]),
        TensorDescriptor.from_tensor(gk, [1, 1, 1, key_dim]),
    )
    kernel_args = (
        *descriptors,
        initial_state,
        final_state,
        state_indices,
        has_initial_state,
        cu_seqlens,
        chunk_offsets,
        k,
        w,
        u,
        v_new,
        tokens,
        *h0_strides,
        *ht_strides,
    )
    kernel_options = {
        "DYNAMIC_STATE_LAYOUT": (initial_state is not None and not initial_state.is_contiguous())
        or (final_state is not None and not final_state.is_contiguous()),
        "H": heads,
        "K": key_dim,
        "V": value_dim,
        "BT": _CHUNK_SIZE,
        "BV": block_value_dim,
        "WARP_SPECIALIZE": use_16bit_config,
        "NUM_STAGES": 3 if use_16bit_config else 2,
        "USE_INITIAL_STATE": initial_state is not None,
        "STORE_FINAL_STATE": final_state is not None,
        "USE_STATE_INDICES": state_indices is not None,
        "USE_HAS_INITIAL_STATE": has_initial_state is not None,
        "USE_INT64_OFFSETS": requires_int64_offsets(
            k, w, u, gk, v_new, h, initial_state, final_state
        ),
        "num_warps": 4,
    }
    value_tiles = value_dim // block_value_dim
    sequence_workers = 0
    if cu_seqlens is not None:
        assert chunk_offsets is not None
        sequence_workers = _persistent_sequence_workers(
            RaggedChunkMetadata(cu_seqlens, chunk_offsets, capacity, _CHUNK_SIZE),
            tokens,
            heads,
            value_tiles,
            k.device,
            schedule,
        )
    if sequence_workers:
        # Keep one machine-sized wave warp-specialized, then stride persistent workers
        # over the remaining work; Triton cannot warp-specialize the nested loop. This
        # reduced N=512, M=32 graph replay from 1.22 ms to 0.68 ms on B200.
        chunk_delta_h_kernel_k128_wsp[(sequence_workers, value_tiles)](
            *kernel_args,
            **kernel_options,
            IS_VARLEN=True,
        )
        persistent_options = kernel_options | {
            "WARP_SPECIALIZE": False,
            "NUM_STAGES": 2,
        }
        chunk_delta_h_kernel_k128_persistent[(sequence_workers, value_tiles)](
            *kernel_args,
            **persistent_options,
            NUM_SEQUENCES=state_batch,
            NUM_WORKERS=sequence_workers,
        )
    else:
        chunk_delta_h_kernel_k128_wsp[(state_batch * heads, value_tiles)](
            *kernel_args,
            **kernel_options,
            IS_VARLEN=cu_seqlens is not None,
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
    return _delta_h_launch(
        k, w, u, gk, initial_state, None, None, cu_seqlens, chunk_offsets, capacity, None
    )


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
        state_batch, k.shape[2], u.shape[-1], k.shape[3], dtype=torch.float32, device=k.device
    )
    h, v_new = _delta_h_launch(
        k,
        w,
        u,
        gk,
        initial_state,
        None,
        None,
        cu_seqlens,
        chunk_offsets,
        capacity,
        final_state,
    )
    return h, v_new, final_state


def _delta_h_paged_cuda(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _delta_h_launch(
        k,
        w,
        u,
        gk,
        state_cache,
        state_indices,
        has_initial_state,
        cu_seqlens,
        chunk_offsets,
        capacity,
        state_cache,
    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
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
    expected_state_shape = (state_batch, heads, value_dim, key_dim)
    if state_indices is not None:
        if initial_state is None:
            raise ValueError("state_indices requires initial_state as the paged state pool")
        if output_final_state:
            raise ValueError("paged state is advanced in place; drop output_final_state")
        paged_state_shape = (heads, value_dim, key_dim)
        if initial_state.ndim != 4 or initial_state.shape[1:] != paged_state_shape:
            raise ValueError(
                "the paged state pool must have shape "
                f"[num_slots, {heads}, {value_dim}, {key_dim}], got {tuple(initial_state.shape)}"
            )
        if initial_state.dtype != torch.float32:
            raise TypeError("the paged state pool must use float32")
        if initial_state.stride()[1:] != (value_dim * key_dim, key_dim, 1):
            raise TypeError("the paged state pool must be contiguous within each [H, V, K] slot")
        if initial_state.stride(0) < heads * key_dim * value_dim:
            raise ValueError("paged state pool slots must not overlap")
        if (
            state_indices.shape != (state_batch,)
            or state_indices.dtype != torch.int32
            or not state_indices.is_contiguous()
            or state_indices.device != k.device
        ):
            raise ValueError(f"state_indices must be contiguous int32 with shape ({state_batch},)")
        if has_initial_state is not None and (
            has_initial_state.shape != (state_batch,)
            or has_initial_state.dtype != torch.bool
            or not has_initial_state.is_contiguous()
            or has_initial_state.device != k.device
        ):
            raise ValueError(
                f"has_initial_state must be contiguous bool with shape ({state_batch},)"
            )
    elif has_initial_state is not None:
        raise ValueError("has_initial_state requires state_indices")
    elif initial_state is not None:
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"initial_state must have shape {expected_state_shape}, "
                f"got {tuple(initial_state.shape)}"
            )
        if initial_state.stride(-1) != 1 or any(stride < 0 for stride in initial_state.stride()):
            raise TypeError("initial_state requires a contiguous key mode")

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

    if state_indices is not None:
        assert initial_state is not None
        h, v_new = _delta_h_paged_op(
            k,
            w,
            u,
            gk,
            initial_state,
            state_indices,
            has_initial_state,
            cu_seqlens,
            chunk_offsets,
            chunks,
        )
        return h, v_new, None
    if output_final_state:
        h, v_new, final_state = _delta_h_with_state_op(
            k, w, u, gk, initial_state, cu_seqlens, chunk_offsets, chunks
        )
        return h, v_new, final_state
    h, v_new = _delta_h_op(k, w, u, gk, initial_state, cu_seqlens, chunk_offsets, chunks)
    return h, v_new, None


__all__ = ["chunk_gated_delta_rule_fwd_h"]
