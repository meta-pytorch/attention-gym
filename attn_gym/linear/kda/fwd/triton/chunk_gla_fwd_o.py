# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Forward-only GLA output projection, based on
# genai/llama4x/llama4x/ops/fla/ops/gla/chunk.py.

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.cute.utils import get_device_properties
from attn_gym._backends.triton.utils import (
    PinnedConfigKernel,
    can_use_tma,
    ptr_offset,
    requires_int64_offsets,
)
from attn_gym.linear.kda.chunk_scheduler import (
    GridScheduler,
    RaggedChunkMetadata,
    ScheduleKind,
    ScheduleRequest,
    decode_ragged_task,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
    load_ragged_task_count,
)
from attn_gym.linear.kda.utils import autotune_cache_kwargs, exp, exp2


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INT64_OFFSETS": lambda args: requires_int64_offsets(
            args["q"],
            args["v"],
            args["g"],
            args["h"],
            args["o"],
            args["A"],
            args["cu_seqlens"],
            args["chunk_offsets"],
        ),
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=4),
        triton.Config({"BK": 32, "BV": 64}, num_warps=2, num_stages=4),
        triton.Config({"BK": 32, "BV": 64}, num_warps=4, num_stages=4),
        triton.Config({"BK": 64, "BV": 32}, num_warps=4, num_stages=4),
        triton.Config({"BK": 64, "BV": 64}, num_warps=2, num_stages=4),
        triton.Config({"BK": 64, "BV": 64}, num_warps=8, num_stages=4),
    ],
    key=["H", "K", "V", "T", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=[
        "T",
        "num_sequences",
        "q_stride_t",
        "q_stride_h",
    ]
)
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    q_stride_t,
    q_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    num_sequences,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_INT64_OFFSETS:
        i_v = i_v.to(tl.int64)
        i_t = i_t.to(tl.int64)
        i_bh = i_bh.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        if i_t >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        i_tg = i_t
        i_n, i_t, token_start, _ = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            i_t,
            num_sequences,
            BT,
        )
        if USE_INT64_OFFSETS:
            i_n = i_n.to(tl.int64)
            i_t = i_t.to(tl.int64)
            token_start = token_start.to(tl.int64)
        eos = tl.load(cu_seqlens + ptr_offset((i_n, 1), (1, 1))).to(tl.int32)
        if USE_INT64_OFFSETS:
            eos = eos.to(tl.int64)
        # token_start == bos + i_t * BT; only eos still needs a load for masking.
        bos = token_start - i_t * BT
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos = i_b * T

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    m_s = o_i[:, None] >= o_i[None, :]

    q += bos * q_stride_t + i_h * q_stride_h
    g += ptr_offset((bos, i_h), (H * K, K))
    h += ptr_offset((i_tg, i_h), (H * K * V, K * V))
    v += ptr_offset((bos, i_h), (H * V, V))
    o += ptr_offset((bos, i_h), (H * V, V))
    A += ptr_offset((bos, i_h), (H * BT, BT))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qg = m_t[:, None] & m_k[None, :]
        p_q = q + o_t[:, None] * q_stride_t + o_k[None, :]
        p_g = g + ptr_offset((o_t[:, None], o_k[None, :]), (H * K, 1))
        p_h = h + ptr_offset((o_k[:, None], o_v[None, :]), (V, 1))

        # [BT, BK]
        b_q = tl.load(p_q, mask=m_qg, other=0.0)
        # [BT, BK]
        b_g = tl.load(p_g, mask=m_qg, other=0.0).to(tl.float32)
        # [BT, BK]
        if USE_EXP2:
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        else:
            b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v = v + ptr_offset((o_t[:, None], o_v[None, :]), (H * V, 1))
    p_o = o + ptr_offset((o_t[:, None], o_v[None, :]), (H * V, 1))
    p_A = A + ptr_offset((o_t[:, None], o_i[None, :]), (H * BT, 1))
    # [BT, BV]
    b_v = tl.load(p_v, mask=m_tv, other=0.0)
    # [BT, BT]
    b_A = tl.load(p_A, mask=m_t[:, None], other=0.0)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit
def _compose_output_tma(
    q_desc,
    v_desc,
    g_desc,
    h_desc,
    o_desc,
    A_desc,
    scale,
    batch,
    token_start,
    head,
    chunk,
    value_tile,
    K: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Compose one complete output tile with TMA-backed tensor descriptors."""
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for key_tile in range(tl.cdiv(K, BK)):
        key_start = key_tile * BK
        b_q = q_desc.load([batch, token_start, head, key_start])
        b_q = tl.reshape(b_q, [BT, BK])
        b_g = g_desc.load([batch, token_start, head, key_start])
        b_g = tl.reshape(b_g, [BT, BK]).to(tl.float32)
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)

        b_h = h_desc.load([batch, chunk, head, key_start, value_tile * BV])
        b_h = tl.reshape(b_h, [BK, BV]).to(b_qg.dtype)
        b_o += tl.dot(b_qg, b_h)

    b_o *= scale
    b_A = A_desc.load([batch, token_start, head, 0])
    b_A = tl.reshape(b_A, [BT, BT])
    offset = tl.arange(0, BT)
    b_A = tl.where(offset[:, None] >= offset[None, :], b_A, 0.0)
    b_v = v_desc.load([batch, token_start, head, value_tile * BV])
    b_v = tl.reshape(b_v, [BT, BV])
    b_o += tl.dot(b_A.to(b_v.dtype), b_v)
    o_desc.store(
        [batch, token_start, head, value_tile * BV],
        tl.reshape(b_o.to(b_v.dtype), [1, BT, 1, BV]),
    )


@triton.jit
def chunk_gla_fwd_kernel_o_tma(
    q_desc,
    v_desc,
    g_desc,
    h_desc,
    o_desc,
    A_desc,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Compose fixed KDA output tiles with TMA-backed tensor descriptors."""
    value_tile, chunk, batch_head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    batch, head = batch_head // H, batch_head % H
    _compose_output_tma(
        q_desc,
        v_desc,
        g_desc,
        h_desc,
        o_desc,
        A_desc,
        scale,
        batch,
        chunk * BT,
        head,
        chunk,
        value_tile,
        K,
        BT,
        BK,
        BV,
    )


@triton.jit
def _compose_ragged_output_task(
    q_desc,
    v_desc,
    g_desc,
    h_desc,
    o_desc,
    A_desc,
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_offsets,
    scale,
    q_stride_t,
    q_stride_h,
    i_v,
    global_chunk,
    i_h,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    num_sequences,
):
    """Compose one active (value-tile, chunk, head) ragged output tile.

    Full chunks go through TMA; partial tails use masked pointers.
    """
    _, _, token_start, valid_tokens = load_ragged_chunk_work(
        cu_seqlens,
        chunk_offsets,
        global_chunk,
        num_sequences,
        BT,
    )
    if valid_tokens == BT:
        _compose_output_tma(
            q_desc,
            v_desc,
            g_desc,
            h_desc,
            o_desc,
            A_desc,
            scale,
            0,
            token_start,
            i_h,
            global_chunk,
            i_v,
            K,
            BT,
            BK,
            BV,
        )
    else:
        o_i = tl.arange(0, BT)
        o_t = token_start + o_i
        o_v = i_v * BV + tl.arange(0, BV)
        m_t = o_i < valid_tokens
        m_v = o_v < V
        m_tv = m_t[:, None] & m_v[None, :]
        b_o = tl.zeros([BT, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_qg = m_t[:, None] & m_k[None, :]
            p_q = q + o_t[:, None] * q_stride_t + i_h * q_stride_h + o_k[None, :]
            p_g = g + ptr_offset((o_t[:, None], i_h, o_k[None, :]), (H * K, K, 1))
            p_h = h + ptr_offset(
                (global_chunk, i_h, o_k[:, None], o_v[None, :]),
                (H * K * V, K * V, V, 1),
            )
            b_q = tl.load(p_q, mask=m_qg, other=0.0)
            b_g = tl.load(p_g, mask=m_qg, other=0.0).to(tl.float32)
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
            b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))

        b_o *= scale
        p_v = v + ptr_offset((o_t[:, None], i_h, o_v[None, :]), (H * V, V, 1))
        p_A = A + ptr_offset((o_t[:, None], i_h, o_i[None, :]), (H * BT, BT, 1))
        b_v = tl.load(p_v, mask=m_tv, other=0.0)
        b_A = tl.load(p_A, mask=m_t[:, None], other=0.0)
        b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A, 0.0).to(b_v.dtype)
        b_o += tl.dot(b_A, b_v)
        p_o = o + ptr_offset((o_t[:, None], i_h, o_v[None, :]), (H * V, V, 1))
        tl.store(p_o, b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit(do_not_specialize=["num_sequences"])
def chunk_gla_fwd_kernel_o_ragged_tma(
    q_desc,
    v_desc,
    g_desc,
    h_desc,
    o_desc,
    A_desc,
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_offsets,
    scale,
    q_stride_t,
    q_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    num_sequences,
):
    """Launch one CTA per capacity task; capacity-only CTAs exit immediately."""
    i_v, global_chunk, i_h = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if global_chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
        return
    _compose_ragged_output_task(
        q_desc,
        v_desc,
        g_desc,
        h_desc,
        o_desc,
        A_desc,
        q,
        v,
        g,
        h,
        o,
        A,
        cu_seqlens,
        chunk_offsets,
        scale,
        q_stride_t,
        q_stride_h,
        i_v,
        global_chunk,
        i_h,
        H,
        K,
        V,
        BT,
        BK,
        BV,
        num_sequences,
    )


@triton.jit(do_not_specialize=["num_sequences", "num_workers"])
def chunk_gla_fwd_kernel_o_ragged_tma_persistent(
    q_desc,
    v_desc,
    g_desc,
    h_desc,
    o_desc,
    A_desc,
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_offsets,
    scale,
    q_stride_t,
    q_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    num_sequences,
    num_workers,
):
    """Stride a bounded worker grid over active (chunk, head, value-tile) tasks."""
    worker = tl.program_id(0)
    num_value_tiles: tl.constexpr = tl.cdiv(V, BV)
    subtasks: tl.constexpr = H * num_value_tiles
    total_tasks = load_ragged_task_count(chunk_offsets, num_sequences, subtasks)
    # num_stages=1 stops the software pipeliner from double-buffering the
    # outer task loop; the extra SMEM stage costs a resident CTA per SM.
    for task in tl.range(worker, total_tasks, num_workers, num_stages=1):
        # TMA descriptor offsets are int32; chunk capacity and subtask indices
        # each remain in range even when their flattened product needs int64.
        global_chunk, remainder = decode_ragged_task(task, subtasks)
        global_chunk = global_chunk.to(tl.int32)
        remainder = remainder.to(tl.int32)
        _compose_ragged_output_task(
            q_desc,
            v_desc,
            g_desc,
            h_desc,
            o_desc,
            A_desc,
            q,
            v,
            g,
            h,
            o,
            A,
            cu_seqlens,
            chunk_offsets,
            scale,
            q_stride_t,
            q_stride_h,
            remainder % num_value_tiles,
            global_chunk,
            remainder // num_value_tiles,
            H,
            K,
            V,
            BT,
            BK,
            BV,
            num_sequences,
        )


def _can_use_tensor_descriptors(*tensors: torch.Tensor) -> bool:
    """Return whether all fixed KDA tensors satisfy host TMA requirements."""
    return all(can_use_tma(tensor) for tensor in tensors)


_PINNED_FWD_O = PinnedConfigKernel(chunk_gla_fwd_kernel_o)


def _select_ragged_output_schedule_kind(
    request: ScheduleRequest,
    resolved_kind: ScheduleKind,
    device_major: int,
) -> ScheduleKind:
    """Keep Hopper on its faster static ragged-TMA launch under automatic selection."""
    if request is ScheduleRequest.AUTO and device_major == 9:
        return ScheduleKind.STATIC
    return resolved_kind


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
    metadata: RaggedChunkMetadata | None = None,
    autotune: bool = True,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
) -> torch.Tensor:
    """Compose fixed-length or packed KDA intra- and inter-chunk output terms.

    Args:
        schedule: Internal scheduling request for tests. Automatic selection is
            the default and dense inputs keep their exact launch grid.

    Raises:
        ValueError: If persistent scheduling is forced for a packed input outside
            the TMA path, where no persistent kernel exists to honor it.
    """
    if metadata is not None:
        metadata.validate_chunk_size(chunk_size)
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if metadata is None and tokens % chunk_size:
        raise ValueError(f"the dense KDA output kernel requires complete chunks, got T={tokens}")
    chunks = triton.cdiv(tokens, chunk_size) if metadata is None else metadata.capacity
    if g.shape != q.shape:
        raise ValueError("g must have the same shape as q")
    if v.shape != (batch, tokens, heads, value_dim):
        raise ValueError("v must have shape [B, T, H, V]")
    if A.shape != (batch, tokens, heads, chunk_size):
        raise ValueError(
            f"A must have shape {(batch, tokens, heads, chunk_size)}, got {tuple(A.shape)}"
        )
    expected_h_shape = (batch, chunks, heads, key_dim, value_dim)
    if h.shape != expected_h_shape:
        raise ValueError(f"h must have shape {expected_h_shape}, got {tuple(h.shape)}")

    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    if chunks == 0:
        return output
    ragged_tma = (
        metadata is not None
        and batch == 1
        and (key_dim, value_dim, chunk_size) == (128, 128, 64)
        and _can_use_tensor_descriptors(q, v, g, h, output, A)
    )
    if metadata is not None:
        subtasks = heads * triton.cdiv(value_dim, 64)  # 64 = the persistent kernel's BV tile
        resolved = GridScheduler(metadata).resolve_flat(
            schedule,
            subtasks,
            q.device,
            eligible=ragged_tma,
            requirement="the TMA path on packed inputs: batch=1, K=V=128, chunk_size=64, "
            "and TMA-capable tensors",
        )
    if (
        metadata is None
        and (key_dim, value_dim, chunk_size) == (128, 128, 64)
        and _can_use_tensor_descriptors(q, v, g, h, output, A)
    ):
        block_key_dim = 32
        block_value_dim = 64
        chunk_gla_fwd_kernel_o_tma[
            (
                triton.cdiv(value_dim, block_value_dim),
                chunks,
                batch * heads,
            )
        ](
            TensorDescriptor.from_tensor(q, [1, chunk_size, 1, block_key_dim]),
            TensorDescriptor.from_tensor(v, [1, chunk_size, 1, block_value_dim]),
            TensorDescriptor.from_tensor(g, [1, chunk_size, 1, block_key_dim]),
            TensorDescriptor.from_tensor(h, [1, 1, 1, block_key_dim, block_value_dim]),
            TensorDescriptor.from_tensor(output, [1, chunk_size, 1, block_value_dim]),
            TensorDescriptor.from_tensor(A, [1, chunk_size, 1, chunk_size]),
            scale,
            H=heads,
            K=key_dim,
            V=value_dim,
            BT=chunk_size,
            BK=block_key_dim,
            BV=block_value_dim,
            num_warps=2,
            num_stages=3,
        )
    else:
        if ragged_tma:
            block_key_dim = 32
            block_value_dim = 64
            value_tiles = triton.cdiv(value_dim, block_value_dim)
            args = (
                TensorDescriptor.from_tensor(q, [1, chunk_size, 1, block_key_dim]),
                TensorDescriptor.from_tensor(v, [1, chunk_size, 1, block_value_dim]),
                TensorDescriptor.from_tensor(g, [1, chunk_size, 1, block_key_dim]),
                TensorDescriptor.from_tensor(h, [1, 1, 1, block_key_dim, block_value_dim]),
                TensorDescriptor.from_tensor(output, [1, chunk_size, 1, block_value_dim]),
                TensorDescriptor.from_tensor(A, [1, chunk_size, 1, chunk_size]),
                q,
                v,
                g,
                h,
                output,
                A,
                metadata.cu_seqlens,
                metadata.chunk_offsets,
                scale,
                q.stride(1),
                q.stride(2),
            )
            kwargs = {
                "H": heads,
                "K": key_dim,
                "V": value_dim,
                "BT": chunk_size,
                "BK": block_key_dim,
                "BV": block_value_dim,
                "num_sequences": metadata.cu_seqlens.shape[0] - 1,
                "num_warps": 2,
                "num_stages": 3,
                # The partial-tail pointer branch otherwise pushes this kernel
                # past the dense register budget and loses a resident CTA.
                "maxnreg": 136,
            }
            schedule_kind = _select_ragged_output_schedule_kind(
                schedule,
                resolved.kind,
                get_device_properties(q.device).major,
            )
            if schedule_kind is ScheduleKind.PERSISTENT:
                chunk_gla_fwd_kernel_o_ragged_tma_persistent[(resolved.workers,)](
                    *args, num_workers=resolved.workers, **kwargs
                )
            else:
                chunk_gla_fwd_kernel_o_ragged_tma[(value_tiles, metadata.capacity, heads)](
                    *args, **kwargs
                )
            return output

        def grid(meta):
            return (triton.cdiv(value_dim, meta["BV"]), chunks, batch * heads)

        kernel = chunk_gla_fwd_kernel_o if autotune else _PINNED_FWD_O
        kernel[grid](
            q=q,
            v=v,
            g=g,
            h=h,
            o=output,
            A=A,
            cu_seqlens=None if metadata is None else metadata.cu_seqlens,
            chunk_offsets=None if metadata is None else metadata.chunk_offsets,
            scale=scale,
            T=tokens,
            q_stride_t=q.stride(1),
            q_stride_h=q.stride(2),
            H=heads,
            K=key_dim,
            V=value_dim,
            BT=chunk_size,
            num_sequences=(0 if metadata is None else metadata.cu_seqlens.shape[0] - 1),
            USE_EXP2=True,
        )
    return output


__all__ = ["chunk_gla_fwd_o_gk"]
