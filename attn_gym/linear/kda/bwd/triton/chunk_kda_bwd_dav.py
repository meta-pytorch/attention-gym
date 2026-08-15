# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Triton dAv backward for KDA (the intra-chunk ``Aqk @ v_new`` path). This is
# the one KDA backward compute stage with no CuTe implementation, so it ships as
# Triton. Ported from the original source with local utility-based pointer access.
#
# Original source: genai/llama4x/llama4x/ops/fla/ops/kda/chunk_bwd.py

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import can_use_tma, ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    chunk_capacity,
    load_ragged_chunk_work,
)
from attn_gym.linear.kda.utils import (
    ChunkMetadata,
    autotune_cache_kwargs,
)


def _uses_tensor_descriptors(args) -> bool:
    return isinstance(args["v"], TensorDescriptor)


def _requires_int64_offsets(args) -> bool:
    if _uses_tensor_descriptors(args):
        return False
    return requires_int64_offsets(
        args["v"],
        args["A"],
        args["do"],
        args["dv"],
        args["dA"],
        args["cu_seqlens"],
        args["chunk_indices"],
        args["chunk_offsets"],
        args["num_chunks"],
    )


@triton.jit
def _dav_tile_tma(
    v_desc,
    A_desc,
    do_desc,
    dv_desc,
    dA_desc,
    scale,
    i_b,
    row,
    i_h,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    """Differentiate one complete chunk starting at ``row`` through TMA descriptors."""
    o_i = tl.arange(0, BT)
    b_A = A_desc.load([i_b, row, i_h, 0])
    b_A = tl.trans(tl.reshape(b_A, [BT, BT]))
    b_A = tl.where(o_i[:, None] <= o_i[None, :], b_A, 0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    for i_v in range(tl.cdiv(V, BV)):
        b_v = v_desc.load([i_b, row, i_h, i_v * BV])
        b_v = tl.trans(tl.reshape(b_v, [BT, BV]))
        b_do = do_desc.load([i_b, row, i_h, i_v * BV])
        b_do = tl.reshape(b_do, [BT, BV])
        b_dA += tl.dot(b_do, b_v)
        b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
        dv_desc.store(
            [i_b, row, i_h, i_v * BV],
            tl.reshape(b_dv.to(b_do.dtype), [1, BT, 1, BV]),
        )
    b_dA = tl.where(o_i[:, None] >= o_i[None, :], b_dA * scale, 0.0)
    dA_desc.store(
        [i_b, row, i_h, 0],
        tl.reshape(b_dA, [1, BT, 1, BT]),
    )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
        "IS_RAGGED": lambda args: args["chunk_offsets"] is not None,
        "USE_TMA": _uses_tensor_descriptors,
        "USE_INT64_OFFSETS": _requires_int64_offsets,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
    ],
    key=["H", "V", "BT", "BV", "USE_TMA"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_chunks", "num_sequences"])
def chunk_kda_bwd_kernel_dAv(
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    num_chunks,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NUM_CHUNKS: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    num_sequences,
    USE_TMA: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Differentiate KDA value tiles with pointer or TMA specialization."""
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    if USE_TMA:
        i_b, i_h = i_bh // H, i_bh % H
        _dav_tile_tma(v, A, do, dv, dA, scale, i_b, i_t * BT, i_h, V, BT, BV)
    else:
        if USE_INT64_OFFSETS:
            i_t = i_t.to(tl.int64)
            i_bh = i_bh.to(tl.int64)
        i_b, i_h = i_bh // H, i_bh % H

        if IS_VARLEN:
            if IS_RAGGED:
                if i_t >= tl.load(chunk_offsets + num_sequences):
                    return
                i_n, i_t, _, _ = load_ragged_chunk_work(
                    cu_seqlens,
                    chunk_offsets,
                    i_t,
                    num_sequences,
                    BT,
                )
            else:
                if HAS_NUM_CHUNKS and i_t >= tl.load(num_chunks):
                    return
                i_n, i_t = (
                    tl.load(chunk_indices + ptr_offset((i_t, 0), (2, 1))).to(tl.int32),
                    tl.load(chunk_indices + ptr_offset((i_t, 1), (2, 1))).to(tl.int32),
                )
            if USE_INT64_OFFSETS:
                i_n = i_n.to(tl.int64)
                i_t = i_t.to(tl.int64)
            bos, eos = (
                tl.load(cu_seqlens + ptr_offset((i_n,), (1,))).to(tl.int32),
                tl.load(cu_seqlens + ptr_offset((i_n + 1,), (1,))).to(tl.int32),
            )
            if USE_INT64_OFFSETS:
                bos = bos.to(tl.int64)
                eos = eos.to(tl.int64)
            T = eos - bos
        else:
            bos = i_b * T

        v += ptr_offset((bos, i_h), (H * V, V))
        A += ptr_offset((bos, i_h), (H * BT, BT))
        do += ptr_offset((bos, i_h), (H * V, V))
        dv += ptr_offset((bos, i_h), (H * V, V))
        dA += ptr_offset((bos, i_h), (H * BT, BT))

        o_i = tl.arange(0, BT)
        o_t = i_t * BT + o_i
        m_t = o_t < T
        m_tt = m_t[:, None] & m_t[None, :]
        # The leading BT feature axis is always in bounds; only token columns can be ragged.
        b_A = tl.load(
            A + ptr_offset((o_i[:, None], o_t[None, :]), (1, H * BT)),
            mask=m_t[None, :],
            other=0.0,
        )
        m_A = (o_t[:, None] <= o_t[None, :]) & m_tt
        b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)

        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = (o_v[:, None] < V) & m_t[None, :]
            m_do = m_t[:, None] & (o_v[None, :] < V)
            # [BV, BT]
            b_v = tl.load(
                v + ptr_offset((o_v[:, None], o_t[None, :]), (1, H * V)),
                mask=m_v,
                other=0.0,
            )
            # [BT, BV]
            b_do = tl.load(
                do + ptr_offset((o_t[:, None], o_v[None, :]), (H * V, 1)),
                mask=m_do,
                other=0.0,
            )

            # [BT, BT]
            b_dA += tl.dot(b_do, b_v)
            # [BT, BV]
            b_dv = tl.dot(b_A.to(b_do.dtype), b_do)
            tl.store(
                dv + ptr_offset((o_t[:, None], o_v[None, :]), (H * V, 1)),
                b_dv.to(dv.dtype.element_ty),
                mask=m_do,
            )

        b_dA = tl.where(o_i[:, None] >= o_i[None, :], b_dA * scale, 0.0)
        tl.store(
            dA + ptr_offset((o_t[:, None], o_i[None, :]), (H * BT, 1)),
            b_dA.to(dA.dtype.element_ty),
            mask=m_t[:, None],
        )


def _can_use_tensor_descriptors(*tensors: torch.Tensor) -> bool:
    """Return whether all fixed KDA tensors satisfy host TMA requirements."""
    return all(can_use_tma(tensor) for tensor in tensors)


@triton.jit(do_not_specialize=["num_sequences"])
def chunk_kda_bwd_kernel_dAv_ragged_tma(
    v_desc,
    A_desc,
    do_desc,
    dv_desc,
    dA_desc,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_offsets,
    scale,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    num_sequences,
):
    """Differentiate full ragged value chunks with TMA and partial tails with masked pointers."""
    global_chunk, i_h = tl.program_id(0), tl.program_id(1)
    if global_chunk >= tl.load(chunk_offsets + num_sequences):
        return
    _, _, token_start, valid_tokens = load_ragged_chunk_work(
        cu_seqlens,
        chunk_offsets,
        global_chunk,
        num_sequences,
        BT,
    )
    o_i = tl.arange(0, BT)
    if valid_tokens == BT:
        _dav_tile_tma(
            v_desc, A_desc, do_desc, dv_desc, dA_desc, scale, 0, token_start, i_h, V, BT, BV
        )
    else:
        o_t = token_start + o_i
        m_t = o_i < valid_tokens
        # The leading BT feature axis is always in bounds; only token columns can be ragged.
        b_A = tl.load(
            A + ptr_offset((o_t[None, :], i_h, o_i[:, None]), (H * BT, BT, 1)),
            mask=m_t[None, :],
            other=0.0,
        )
        m_A = (o_i[:, None] <= o_i[None, :]) & m_t[:, None] & m_t[None, :]
        b_A = tl.where(m_A, b_A, 0).to(do.dtype.element_ty)
        b_dA = tl.zeros([BT, BT], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            o_v = i_v * BV + tl.arange(0, BV)
            m_v = o_v < V
            # [BV, BT]
            b_v = tl.load(
                v + ptr_offset((o_t[None, :], i_h, o_v[:, None]), (H * V, V, 1)),
                mask=m_v[:, None] & m_t[None, :],
                other=0.0,
            )
            # [BT, BV]
            b_do = tl.load(
                do + ptr_offset((o_t[:, None], i_h, o_v[None, :]), (H * V, V, 1)),
                mask=m_t[:, None] & m_v[None, :],
                other=0.0,
            )
            b_dA += tl.dot(b_do, b_v)
            b_dv = tl.dot(b_A, b_do)
            tl.store(
                dv + ptr_offset((o_t[:, None], i_h, o_v[None, :]), (H * V, V, 1)),
                b_dv.to(dv.dtype.element_ty),
                mask=m_t[:, None] & m_v[None, :],
            )
        b_dA = tl.where(o_i[:, None] >= o_i[None, :], b_dA * scale, 0.0)
        tl.store(
            dA + ptr_offset((o_t[:, None], i_h, o_i[None, :]), (H * BT, BT, 1)),
            b_dA.to(dA.dtype.element_ty),
            mask=m_t[:, None],
        )


def _launch_dav_ragged(
    v: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    chunk_size: int,
    metadata: RaggedChunkMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the packed dAv kernel with TMA descriptors when the layout allows."""
    _, tokens, heads, value_dim = v.shape
    dv = torch.empty_like(v)
    dA = torch.empty_like(A, dtype=torch.float32)
    if metadata.capacity == 0:
        return dv, dA

    block_value_dim = 64
    if (value_dim, chunk_size) == (128, 64) and _can_use_tensor_descriptors(v, A, do, dv, dA):
        chunk_kda_bwd_kernel_dAv_ragged_tma[(metadata.capacity, heads)](
            TensorDescriptor.from_tensor(v, [1, chunk_size, 1, block_value_dim]),
            TensorDescriptor.from_tensor(A, [1, chunk_size, 1, chunk_size]),
            TensorDescriptor.from_tensor(do, [1, chunk_size, 1, block_value_dim]),
            TensorDescriptor.from_tensor(dv, [1, chunk_size, 1, block_value_dim]),
            TensorDescriptor.from_tensor(dA, [1, chunk_size, 1, chunk_size]),
            v,
            A,
            do,
            dv,
            dA,
            metadata.cu_seqlens,
            metadata.chunk_offsets,
            scale,
            H=heads,
            V=value_dim,
            BT=chunk_size,
            BV=block_value_dim,
            num_sequences=metadata.cu_seqlens.shape[0] - 1,
            num_warps=2,
            num_stages=3,
        )
    else:
        chunk_kda_bwd_kernel_dAv[(metadata.capacity, heads)](
            v=v,
            A=A,
            do=do,
            dv=dv,
            dA=dA,
            cu_seqlens=metadata.cu_seqlens,
            chunk_indices=None,
            chunk_offsets=metadata.chunk_offsets,
            num_chunks=None,
            scale=scale,
            T=tokens,
            H=heads,
            V=value_dim,
            BT=chunk_size,
            BV=block_value_dim,
            num_sequences=metadata.cu_seqlens.shape[0] - 1,
        )
    return dv, dA


# The ragged launcher mixes TMA descriptors with aliased raw pointers, which the
# compile stack cannot functionalize correctly, so it lives behind an opaque op.
torch.library.define(
    "attn_gym::kda_chunk_bwd_dav_ragged",
    "(Tensor v, Tensor A, Tensor do, Tensor cu_seqlens, Tensor chunk_offsets, "
    "float scale, int chunk_size) -> (Tensor, Tensor)",
)


def _chunk_kda_bwd_dav_ragged_cuda(
    v: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = RaggedChunkMetadata(
        cu_seqlens,
        chunk_offsets,
        chunk_capacity(v.shape[1], cu_seqlens.shape[0] - 1, chunk_size),
        chunk_size,
    )
    return _launch_dav_ragged(v, A, do, scale, chunk_size, metadata)


torch.library.impl("attn_gym::kda_chunk_bwd_dav_ragged", "CUDA", _chunk_kda_bwd_dav_ragged_cuda)


@torch.library.register_fake("attn_gym::kda_chunk_bwd_dav_ragged")
def _chunk_kda_bwd_dav_ragged_fake(
    v: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    scale: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del do, cu_seqlens, chunk_offsets, scale, chunk_size
    return torch.empty_like(v), torch.empty_like(A, dtype=torch.float32)


_dav_ragged_op = torch.ops.attn_gym.kda_chunk_bwd_dav_ragged.default


def chunk_kda_bwd_dav(
    v: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
    metadata: ChunkMetadata | RaggedChunkMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate the fixed-length or packed KDA intra-chunk value term."""
    batch, tokens, heads, value_dim = v.shape
    if A.shape != (batch, tokens, heads, chunk_size):
        raise ValueError("A must have shape [B, T, H, chunk_size]")
    if do.shape != v.shape:
        raise ValueError("do must have the same shape as v")
    if isinstance(metadata, RaggedChunkMetadata):
        metadata.validate_chunk_size(chunk_size)
        if batch != 1:
            raise ValueError("ragged KDA dAv metadata requires batch size 1")
        return _dav_ragged_op(
            v, A, do, metadata.cu_seqlens, metadata.chunk_offsets, scale, chunk_size
        )
    if tokens % chunk_size:
        raise ValueError(f"the KDA dAv kernel requires complete chunks, got T={tokens}")

    dv = torch.empty_like(v)
    dA = torch.empty_like(A, dtype=torch.float32)
    chunks = tokens // chunk_size
    if chunks == 0:
        return dv, dA

    block_value_dim = 64
    if (
        metadata is None
        and (value_dim, chunk_size) == (128, 64)
        and _can_use_tensor_descriptors(v, A, do, dv, dA)
    ):
        chunk_kda_bwd_kernel_dAv[(chunks, batch * heads)](
            v=TensorDescriptor.from_tensor(v, [1, chunk_size, 1, block_value_dim]),
            A=TensorDescriptor.from_tensor(A, [1, chunk_size, 1, chunk_size]),
            do=TensorDescriptor.from_tensor(do, [1, chunk_size, 1, block_value_dim]),
            dv=TensorDescriptor.from_tensor(dv, [1, chunk_size, 1, block_value_dim]),
            dA=TensorDescriptor.from_tensor(dA, [1, chunk_size, 1, chunk_size]),
            cu_seqlens=None,
            chunk_indices=None,
            chunk_offsets=None,
            num_chunks=None,
            scale=scale,
            T=tokens,
            H=heads,
            V=value_dim,
            BT=chunk_size,
            BV=block_value_dim,
            num_sequences=0,
        )
    else:
        chunk_kda_bwd_kernel_dAv[(chunks, batch * heads)](
            v=v,
            A=A,
            do=do,
            dv=dv,
            dA=dA,
            cu_seqlens=None if metadata is None else metadata.cu_seqlens,
            chunk_indices=(
                metadata.chunk_indices if isinstance(metadata, ChunkMetadata) else None
            ),
            chunk_offsets=None,
            num_chunks=metadata.num_chunks if isinstance(metadata, ChunkMetadata) else None,
            scale=scale,
            T=tokens,
            H=heads,
            V=value_dim,
            BT=chunk_size,
            BV=block_value_dim,
            num_sequences=0,
        )
    return dv, dA


__all__ = ["chunk_kda_bwd_dav"]
