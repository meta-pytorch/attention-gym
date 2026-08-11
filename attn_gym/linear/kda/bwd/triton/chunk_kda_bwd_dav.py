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

import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.utils import (
    autotune_cache_kwargs,
)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
        "USE_INT64_OFFSETS": lambda args: requires_int64_offsets(
            args["q"],
            args["k"],
            args["v"],
            args["A"],
            args["do"],
            args["dv"],
            args["dA"],
            args["cu_seqlens"],
            args["chunk_indices"],
            args["num_chunks"],
        ),
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
    ],
    key=["H", "K", "V", "BT", "BK", "BV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_chunks"])
def chunk_kda_bwd_kernel_dAv(
    q,
    k,
    v,
    A,
    do,
    dv,
    dA,
    cu_seqlens,
    chunk_indices,
    num_chunks,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NUM_CHUNKS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    if USE_INT64_OFFSETS:
        i_t = i_t.to(tl.int64)
        i_bh = i_bh.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
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

    q += ptr_offset((bos, i_h), (H * K, K))
    k += ptr_offset((bos, i_h), (H * K, K))
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

    b_dA = tl.where(o_t[:, None] >= o_t[None, :], b_dA * scale, 0.0)
    tl.store(
        dA + ptr_offset((o_t[:, None], o_i[None, :]), (H * BT, 1)),
        b_dA.to(dA.dtype.element_ty),
        mask=m_t[:, None],
    )
