# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.utils import (
    ChunkMetadata,
    autotune_cache_kwargs,
    exp,
    exp2,
)


def _requires_int64_offsets(args):
    return requires_int64_offsets(
        args["k"],
        args["v"],
        args["w"],
        args["v_new"],
        args["g"],
        args["gk"],
        args["h"],
        args["h0"],
        args["ht"],
        args["cu_seqlens"],
        args["chunk_offsets"],
        args["num_seqs"],
    )


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NUM_SEQS": lambda args: args["num_seqs"] is not None,
        "USE_INT64_OFFSETS": _requires_int64_offsets,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": 32}, num_warps=4, num_stages=3),
        triton.Config({"BV": 16}, num_warps=4, num_stages=3),
        triton.Config({"BV": 64}, num_warps=4, num_stages=3),
    ],
    key=["H", "K", "V", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_seqs"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    num_seqs,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NUM_SEQS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    if USE_INT64_OFFSETS:
        i_nh = i_nh.to(tl.int64)
        i_v = i_v.to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        if HAS_NUM_SEQS and i_n >= tl.load(num_seqs):
            return
        bos, eos = (
            tl.load(cu_seqlens + ptr_offset((i_n,), (1,))).to(tl.int32),
            tl.load(cu_seqlens + ptr_offset((i_n,), (1,)) + 1).to(tl.int32),
        )
        boh = tl.load(chunk_offsets + ptr_offset((i_n,), (1,))).to(tl.int32)
        if USE_INT64_OFFSETS:
            bos = bos.to(tl.int64)
            eos = eos.to(tl.int64)
            boh = boh.to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        bos = i_n * T
        eos = bos + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += ptr_offset((boh, i_h), (H * K * V, K * V))
    v += ptr_offset((bos, i_h), (H * V, V))
    k += ptr_offset((bos, i_h), (H * K, K))
    w += ptr_offset((bos, i_h), (H * K, K))
    if SAVE_NEW_VALUE:
        v_new += ptr_offset((bos, i_h), (H * V, V))

    if USE_INITIAL_STATE:
        h0 = h0 + ptr_offset((i_nh,), (K * V,))
    if STORE_FINAL_STATE:
        ht = ht + ptr_offset((i_nh,), (K * V,))

    o_k1 = tl.arange(0, 64)
    o_v = ptr_offset((i_v,), (BV,)) + tl.arange(0, BV)
    m_v = o_v < V
    if K > 64:
        o_k2 = 64 + o_k1
    if K > 128:
        o_k3 = 128 + o_k1
    if K > 192:
        o_k4 = 192 + o_k1

    # load initial state
    if USE_INITIAL_STATE:
        m_h0 = (o_k1[:, None] < K) & m_v[None, :]
        b_h1 += tl.load(
            h0 + ptr_offset((o_k1[:, None], o_v[None, :]), (V, 1)),
            mask=m_h0,
            other=0.0,
        ).to(tl.float32)
        if K > 64:
            m_h0 = (o_k2[:, None] < K) & m_v[None, :]
            b_h2 += tl.load(  # pyrefly: ignore[unbound-name]
                h0 + ptr_offset((o_k2[:, None], o_v[None, :]), (V, 1)),
                mask=m_h0,
                other=0.0,
            ).to(tl.float32)
        if K > 128:
            m_h0 = (o_k3[:, None] < K) & m_v[None, :]
            b_h3 += tl.load(  # pyrefly: ignore[unbound-name]
                h0 + ptr_offset((o_k3[:, None], o_v[None, :]), (V, 1)),
                mask=m_h0,
                other=0.0,
            ).to(tl.float32)
        if K > 192:
            m_h0 = (o_k4[:, None] < K) & m_v[None, :]
            b_h4 += tl.load(  # pyrefly: ignore[unbound-name]
                h0 + ptr_offset((o_k4[:, None], o_v[None, :]), (V, 1)),
                mask=m_h0,
                other=0.0,
            ).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_offset = i_t
        if USE_INT64_OFFSETS:
            i_t_offset = i_t_offset.to(tl.int64)
        h_t = h + ptr_offset((i_t_offset,), (H * K * V,))
        m_h = (o_k1[:, None] < K) & m_v[None, :]
        tl.store(
            h_t + ptr_offset((o_k1[:, None], o_v[None, :]), (V, 1)),
            b_h1.to(h.dtype.element_ty),
            mask=m_h,
        )
        if K > 64:
            m_h = (o_k2[:, None] < K) & m_v[None, :]
            tl.store(
                h_t + ptr_offset((o_k2[:, None], o_v[None, :]), (V, 1)),
                b_h2.to(h.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                mask=m_h,
            )
        if K > 128:
            m_h = (o_k3[:, None] < K) & m_v[None, :]
            tl.store(
                h_t + ptr_offset((o_k3[:, None], o_v[None, :]), (V, 1)),
                b_h3.to(h.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                mask=m_h,
            )
        if K > 192:
            m_h = (o_k4[:, None] < K) & m_v[None, :]
            tl.store(
                h_t + ptr_offset((o_k4[:, None], o_v[None, :]), (V, 1)),
                b_h4.to(h.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                mask=m_h,
            )

        o_t = ptr_offset((i_t_offset,), (BT,)) + tl.arange(0, BT)
        m_t = o_t < T
        m_w = m_t[:, None] & (o_k1[None, :] < K)
        b_w = tl.load(
            w + ptr_offset((o_t[:, None], o_k1[None, :]), (H * K, 1)),
            mask=m_w,
            other=0.0,
        )
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
        if K > 64:
            m_w = m_t[:, None] & (o_k2[None, :] < K)
            b_w = tl.load(
                w + ptr_offset((o_t[:, None], o_k2[None, :]), (H * K, 1)),
                mask=m_w,
                other=0.0,
            )
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
        if K > 128:
            m_w = m_t[:, None] & (o_k3[None, :] < K)
            b_w = tl.load(
                w + ptr_offset((o_t[:, None], o_k3[None, :]), (H * K, 1)),
                mask=m_w,
                other=0.0,
            )
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
        if K > 192:
            m_w = m_t[:, None] & (o_k4[None, :] < K)
            b_w = tl.load(
                w + ptr_offset((o_t[:, None], o_k4[None, :]), (H * K, 1)),
                mask=m_w,
                other=0.0,
            )
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))  # pyrefly: ignore[unbound-name]

        m_v_block = m_t[:, None] & m_v[None, :]
        v_offset = ptr_offset((o_t[:, None], o_v[None, :]), (H * V, 1))
        b_v = tl.load(v + v_offset, mask=m_v_block, other=0.0) - b_v

        if SAVE_NEW_VALUE:
            tl.store(
                v_new + v_offset,
                b_v.to(v_new.dtype.element_ty),
                mask=m_v_block,
            )

        last_idx = min(ptr_offset((i_t_offset,), (BT,)) + BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + ptr_offset((bos, last_idx, i_h), (H, H, 1)))
            b_g = tl.load(
                g + ptr_offset((bos, o_t, i_h), (H, H, 1)),
                mask=m_t,
                other=0.0,
            )
            if USE_EXP2:  # pyrefly: ignore[unbound-name]
                b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                b_g_last = exp2(b_g_last)
            else:
                b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last  # pyrefly: ignore[unbound-name]
            if K > 128:
                b_h3 *= b_g_last  # pyrefly: ignore[unbound-name,unsupported-operation]
            if K > 192:
                b_h4 *= b_g_last  # pyrefly: ignore[unbound-name,unsupported-operation]

        if USE_GK:
            b_gk_last1 = tl.load(
                gk + ptr_offset((bos, last_idx, i_h, o_k1), (H * K, H * K, K, 1)),
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:  # pyrefly: ignore[unbound-name,unsupported-operation]
                b_h1 *= exp2(b_gk_last1)[
                    :, None  # pyrefly: ignore[unbound-name,unsupported-operation]
                ]  # pyrefly: ignore[unsupported-operation]
            else:
                b_h1 *= exp(b_gk_last1)[:, None]  # pyrefly: ignore[unsupported-operation]
            if K > 64:
                b_gk_last2 = tl.load(
                    gk
                    + ptr_offset(
                        (bos, last_idx, i_h, o_k2),
                        (H * K, H * K, K, 1),
                    ),  # pyrefly: ignore[unbound-name,unsupported-operation]
                    mask=(o_k2 < K),
                    other=0.0,  # pyrefly: ignore[unbound-name,unsupported-operation]
                )
                if USE_EXP2:
                    b_h2 *= exp2(b_gk_last2)[  # pyrefly: ignore[unbound-name]
                        :, None
                    ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                else:
                    b_h2 *= exp(b_gk_last2)[  # pyrefly: ignore[unbound-name]
                        :, None
                    ]  # pyrefly: ignore[unbound-name,unsupported-operation]
            if K > 128:
                b_gk_last3 = tl.load(
                    gk
                    + ptr_offset(
                        (bos, last_idx, i_h, o_k3),
                        (H * K, H * K, K, 1),
                    ),
                    mask=(o_k3 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h3 *= exp2(b_gk_last3)[  # pyrefly: ignore[unbound-name]
                        :, None
                    ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                else:
                    b_h3 *= exp(b_gk_last3)[  # pyrefly: ignore[unbound-name]
                        :, None
                    ]  # pyrefly: ignore[unbound-name,unsupported-operation]
            if K > 192:
                b_gk_last4 = tl.load(
                    gk
                    + ptr_offset(
                        (bos, last_idx, i_h, o_k4),
                        (H * K, H * K, K, 1),
                    ),  # pyrefly: ignore[unbound-name]
                    mask=(o_k4 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h4 *= exp2(b_gk_last4)[  # pyrefly: ignore[unbound-name]
                        :, None  # pyrefly: ignore[unbound-name]
                    ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                else:
                    b_h4 *= exp(b_gk_last4)[  # pyrefly: ignore[unbound-name]
                        :, None
                    ]  # pyrefly: ignore[unbound-name,unsupported-operation]
        b_v = b_v.to(k.dtype.element_ty)

        m_k = (o_k1[:, None] < K) & m_t[None, :]
        b_k = tl.load(
            k + ptr_offset((o_k1[:, None], o_t[None, :]), (1, H * K)),
            mask=m_k,
            other=0.0,
        )
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            m_k = (o_k2[:, None] < K) & m_t[None, :]
            b_k = tl.load(
                k + ptr_offset((o_k2[:, None], o_t[None, :]), (1, H * K)),
                mask=m_k,
                other=0.0,
            )
            b_h2 += tl.dot(b_k, b_v)  # pyrefly: ignore[unbound-name]
        if K > 128:
            m_k = (o_k3[:, None] < K) & m_t[None, :]
            b_k = tl.load(
                k + ptr_offset((o_k3[:, None], o_t[None, :]), (1, H * K)),
                mask=m_k,
                other=0.0,
            )
            b_h3 += tl.dot(b_k, b_v)  # pyrefly: ignore[unbound-name]
        if K > 192:
            m_k = (o_k4[:, None] < K) & m_t[None, :]
            b_k = tl.load(
                k + ptr_offset((o_k4[:, None], o_t[None, :]), (1, H * K)),
                mask=m_k,
                other=0.0,
            )
            b_h4 += tl.dot(b_k, b_v)  # pyrefly: ignore[unbound-name]

    if STORE_FINAL_STATE:
        m_ht = (o_k1[:, None] < K) & m_v[None, :]
        tl.store(
            ht + ptr_offset((o_k1[:, None], o_v[None, :]), (V, 1)),
            b_h1.to(ht.dtype.element_ty),
            mask=m_ht,
        )
        if K > 64:
            m_ht = (o_k2[:, None] < K) & m_v[None, :]
            tl.store(
                ht + ptr_offset((o_k2[:, None], o_v[None, :]), (V, 1)),
                b_h2.to(ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                mask=m_ht,
            )
        if K > 128:
            m_ht = (o_k3[:, None] < K) & m_v[None, :]
            tl.store(
                ht + ptr_offset((o_k3[:, None], o_v[None, :]), (V, 1)),
                b_h3.to(ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                mask=m_ht,
            )
        if K > 192:
            m_ht = (o_k4[:, None] < K) & m_v[None, :]
            tl.store(
                ht + ptr_offset((o_k4[:, None], o_v[None, :]), (V, 1)),
                b_h4.to(ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                mask=m_ht,
            )


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NUM_SEQS": lambda args: args["num_seqs"] is not None,
        "USE_INT64_OFFSETS": _requires_int64_offsets,
    }
)
@triton.jit(do_not_specialize=["T", "num_seqs"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_forloop(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    num_seqs,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NUM_SEQS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    GRID_N: tl.constexpr,
    MAX_N: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    if USE_INT64_OFFSETS:
        i_nh = i_nh.to(tl.int64)
        i_v = i_v.to(tl.int64)
    i_h = i_nh % H
    i_n_start = i_nh // H

    if HAS_NUM_SEQS:
        upper_limit = tl.load(num_seqs)

    o_k1 = tl.arange(0, 64)
    o_v = ptr_offset((i_v,), (BV,)) + tl.arange(0, BV)
    m_v = o_v < V
    if K > 64:
        o_k2 = 64 + o_k1
    if K > 128:
        o_k3 = 128 + o_k1
    if K > 192:
        o_k4 = 192 + o_k1

    for _iter in range((MAX_N + GRID_N - 1) // GRID_N):
        i_n = i_n_start + _iter * GRID_N
        _run = i_n < MAX_N
        if IS_VARLEN and HAS_NUM_SEQS and _run:
            _run = i_n < upper_limit  # pyrefly: ignore [unbound-name]
        if _run:
            if IS_VARLEN:
                bos, eos = (
                    tl.load(cu_seqlens + ptr_offset((i_n,), (1,))).to(tl.int32),
                    tl.load(cu_seqlens + ptr_offset((i_n,), (1,)) + 1).to(tl.int32),
                )
                boh = tl.load(chunk_offsets + ptr_offset((i_n,), (1,))).to(tl.int32)
                if USE_INT64_OFFSETS:
                    bos = bos.to(tl.int64)
                    eos = eos.to(tl.int64)
                    boh = boh.to(tl.int64)
                T_local = eos - bos
                NT = tl.cdiv(T_local, BT)
            else:
                bos = i_n * T
                eos = bos + T
                T_local = T
                NT = tl.cdiv(T_local, BT)
                boh = i_n * NT

            # [BK, BV]
            b_h1 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 64:
                b_h2 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 128:
                b_h3 = tl.zeros([64, BV], dtype=tl.float32)
            if K > 192:
                b_h4 = tl.zeros([64, BV], dtype=tl.float32)

            # calculate offset from base pointers
            h_off = h + ptr_offset((boh, i_h), (H * K * V, K * V))
            v_off = v + ptr_offset((bos, i_h), (H * V, V))
            k_off = k + ptr_offset((bos, i_h), (H * K, K))
            w_off = w + ptr_offset((bos, i_h), (H * K, K))
            if SAVE_NEW_VALUE:
                v_new_off = v_new + ptr_offset((bos, i_h), (H * V, V))

            # load initial state
            if USE_INITIAL_STATE:
                h0_off = h0 + ptr_offset((i_n, i_h), (H * K * V, K * V))
                m_h0 = (o_k1[:, None] < K) & m_v[None, :]
                b_h1 += tl.load(
                    h0_off + ptr_offset((o_k1[:, None], o_v[None, :]), (V, 1)),
                    mask=m_h0,
                    other=0.0,
                ).to(tl.float32)
                if K > 64:
                    m_h0 = (o_k2[:, None] < K) & m_v[None, :]
                    b_h2 += tl.load(  # pyrefly: ignore[unbound-name]
                        h0_off + ptr_offset((o_k2[:, None], o_v[None, :]), (V, 1)),
                        mask=m_h0,
                        other=0.0,
                    ).to(tl.float32)
                if K > 128:
                    m_h0 = (o_k3[:, None] < K) & m_v[None, :]
                    b_h3 += tl.load(  # pyrefly: ignore[unbound-name]
                        h0_off + ptr_offset((o_k3[:, None], o_v[None, :]), (V, 1)),
                        mask=m_h0,
                        other=0.0,
                    ).to(tl.float32)
                if K > 192:
                    m_h0 = (o_k4[:, None] < K) & m_v[None, :]
                    b_h4 += tl.load(  # pyrefly: ignore[unbound-name]
                        h0_off + ptr_offset((o_k4[:, None], o_v[None, :]), (V, 1)),
                        mask=m_h0,
                        other=0.0,
                    ).to(tl.float32)

            # main recurrence
            for i_t in range(NT):
                i_t_offset = i_t
                if USE_INT64_OFFSETS:
                    i_t_offset = i_t_offset.to(tl.int64)
                h_t = h_off + ptr_offset((i_t_offset,), (H * K * V,))
                m_h = (o_k1[:, None] < K) & m_v[None, :]
                tl.store(
                    h_t + ptr_offset((o_k1[:, None], o_v[None, :]), (V, 1)),
                    b_h1.to(h.dtype.element_ty),
                    mask=m_h,
                )
                if K > 64:
                    m_h = (o_k2[:, None] < K) & m_v[None, :]
                    tl.store(
                        h_t + ptr_offset((o_k2[:, None], o_v[None, :]), (V, 1)),
                        b_h2.to(h.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        mask=m_h,
                    )
                if K > 128:
                    m_h = (o_k3[:, None] < K) & m_v[None, :]
                    tl.store(
                        h_t + ptr_offset((o_k3[:, None], o_v[None, :]), (V, 1)),
                        b_h3.to(h.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        mask=m_h,
                    )
                if K > 192:
                    m_h = (o_k4[:, None] < K) & m_v[None, :]
                    tl.store(
                        h_t + ptr_offset((o_k4[:, None], o_v[None, :]), (V, 1)),
                        b_h4.to(h.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        mask=m_h,
                    )

                o_t = ptr_offset((i_t_offset,), (BT,)) + tl.arange(0, BT)
                m_t = o_t < T_local
                m_w = m_t[:, None] & (o_k1[None, :] < K)
                b_w = tl.load(
                    w_off + ptr_offset((o_t[:, None], o_k1[None, :]), (H * K, 1)),
                    mask=m_w,
                    other=0.0,
                )
                b_v = tl.dot(b_w, b_h1.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
                if K > 64:
                    m_w = m_t[:, None] & (o_k2[None, :] < K)
                    b_w = tl.load(
                        w_off + ptr_offset((o_t[:, None], o_k2[None, :]), (H * K, 1)),
                        mask=m_w,
                        other=0.0,
                    )
                    b_v += tl.dot(
                        b_w,
                        b_h2.to(b_w.dtype),  # pyrefly: ignore [unbound-name]
                    )  # pyrefly: ignore[unbound-name]
                if K > 128:
                    m_w = m_t[:, None] & (o_k3[None, :] < K)
                    b_w = tl.load(
                        w_off + ptr_offset((o_t[:, None], o_k3[None, :]), (H * K, 1)),
                        mask=m_w,
                        other=0.0,
                    )
                    b_v += tl.dot(
                        b_w,
                        b_h3.to(b_w.dtype),  # pyrefly: ignore [unbound-name]
                    )  # pyrefly: ignore[unbound-name]
                if K > 192:
                    m_w = m_t[:, None] & (o_k4[None, :] < K)
                    b_w = tl.load(
                        w_off + ptr_offset((o_t[:, None], o_k4[None, :]), (H * K, 1)),
                        mask=m_w,
                        other=0.0,
                    )
                    b_v += tl.dot(
                        b_w,
                        b_h4.to(b_w.dtype),  # pyrefly: ignore [unbound-name]
                    )  # pyrefly: ignore[unbound-name]

                m_v_block = m_t[:, None] & m_v[None, :]
                v_offset = ptr_offset((o_t[:, None], o_v[None, :]), (H * V, 1))
                b_v = tl.load(v_off + v_offset, mask=m_v_block, other=0.0) - b_v

                if SAVE_NEW_VALUE:
                    tl.store(
                        v_new_off + v_offset,  # pyrefly: ignore[unbound-name]
                        b_v.to(v_new.dtype.element_ty),
                        mask=m_v_block,
                    )

                last_idx = min(ptr_offset((i_t_offset,), (BT,)) + BT, T_local) - 1
                if USE_G:
                    b_g_last = tl.load(g + ptr_offset((bos, last_idx, i_h), (H, H, 1)))
                    b_g = tl.load(
                        g + ptr_offset((bos, o_t, i_h), (H, H, 1)),
                        mask=m_t,
                        other=0.0,
                    )
                    if USE_EXP2:  # pyrefly: ignore[unbound-name]
                        b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                        b_g_last = exp2(b_g_last)
                    else:
                        b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                        b_g_last = exp(b_g_last)
                    b_h1 *= b_g_last
                    if K > 64:
                        b_h2 *= b_g_last  # pyrefly: ignore[unbound-name]
                    if K > 128:
                        b_h3 *= b_g_last  # pyrefly: ignore[unbound-name,unsupported-operation]
                    if K > 192:
                        b_h4 *= b_g_last  # pyrefly: ignore[unbound-name,unsupported-operation]

                if USE_GK:
                    b_gk_last1 = tl.load(
                        gk
                        + ptr_offset(
                            (bos, last_idx, i_h, o_k1),
                            (H * K, H * K, K, 1),
                        ),
                        mask=(o_k1 < K),
                        other=0.0,
                    )
                    if USE_EXP2:  # pyrefly: ignore[unbound-name,unsupported-operation]
                        b_h1 *= exp2(b_gk_last1)[
                            :,
                            None,  # pyrefly: ignore[unbound-name,unsupported-operation]
                        ]  # pyrefly: ignore[unsupported-operation]
                    else:
                        b_h1 *= exp(b_gk_last1)[:, None]  # pyrefly: ignore[unsupported-operation]
                    if K > 64:
                        b_gk_last2 = tl.load(
                            gk
                            + ptr_offset(
                                (bos, last_idx, i_h, o_k2),
                                (H * K, H * K, K, 1),
                            ),  # pyrefly: ignore[unbound-name,unsupported-operation]
                            mask=(o_k2 < K),
                            other=0.0,  # pyrefly: ignore[unbound-name,unsupported-operation]
                        )
                        if USE_EXP2:
                            b_h2 *= exp2(b_gk_last2)[  # pyrefly: ignore[unbound-name]
                                :, None
                            ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                        else:
                            b_h2 *= exp(b_gk_last2)[  # pyrefly: ignore[unbound-name]
                                :, None
                            ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                    if K > 128:
                        b_gk_last3 = tl.load(
                            gk
                            + ptr_offset(
                                (bos, last_idx, i_h, o_k3),
                                (H * K, H * K, K, 1),
                            ),
                            mask=(o_k3 < K),
                            other=0.0,
                        )
                        if USE_EXP2:
                            b_h3 *= exp2(b_gk_last3)[  # pyrefly: ignore[unbound-name]
                                :, None
                            ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                        else:
                            b_h3 *= exp(b_gk_last3)[  # pyrefly: ignore[unbound-name]
                                :, None
                            ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                    if K > 192:
                        b_gk_last4 = tl.load(
                            gk
                            + ptr_offset(
                                (bos, last_idx, i_h, o_k4),
                                (H * K, H * K, K, 1),
                            ),  # pyrefly: ignore[unbound-name]
                            mask=(o_k4 < K),
                            other=0.0,
                        )
                        if USE_EXP2:
                            b_h4 *= exp2(b_gk_last4)[  # pyrefly: ignore[unbound-name]
                                :, None  # pyrefly: ignore[unbound-name]
                            ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                        else:
                            b_h4 *= exp(b_gk_last4)[  # pyrefly: ignore[unbound-name]
                                :, None
                            ]  # pyrefly: ignore[unbound-name,unsupported-operation]
                b_v = b_v.to(k.dtype.element_ty)

                m_k = (o_k1[:, None] < K) & m_t[None, :]
                b_k = tl.load(
                    k_off + ptr_offset((o_k1[:, None], o_t[None, :]), (1, H * K)),
                    mask=m_k,
                    other=0.0,
                )
                b_h1 += tl.dot(b_k, b_v)
                if K > 64:
                    m_k = (o_k2[:, None] < K) & m_t[None, :]
                    b_k = tl.load(
                        k_off + ptr_offset((o_k2[:, None], o_t[None, :]), (1, H * K)),
                        mask=m_k,
                        other=0.0,
                    )
                    b_h2 += tl.dot(b_k, b_v)  # pyrefly: ignore [unbound-name]
                if K > 128:
                    m_k = (o_k3[:, None] < K) & m_t[None, :]
                    b_k = tl.load(
                        k_off + ptr_offset((o_k3[:, None], o_t[None, :]), (1, H * K)),
                        mask=m_k,
                        other=0.0,
                    )
                    b_h3 += tl.dot(b_k, b_v)  # pyrefly: ignore [unbound-name]
                if K > 192:
                    m_k = (o_k4[:, None] < K) & m_t[None, :]
                    b_k = tl.load(
                        k_off + ptr_offset((o_k4[:, None], o_t[None, :]), (1, H * K)),
                        mask=m_k,
                        other=0.0,
                    )
                    b_h4 += tl.dot(b_k, b_v)  # pyrefly: ignore[unbound-name]

            if STORE_FINAL_STATE:
                ht_off = ht + ptr_offset((i_n, i_h), (H * K * V, K * V))
                m_ht = (o_k1[:, None] < K) & m_v[None, :]
                tl.store(
                    ht_off + ptr_offset((o_k1[:, None], o_v[None, :]), (V, 1)),
                    b_h1.to(ht.dtype.element_ty),
                    mask=m_ht,
                )
                if K > 64:
                    m_ht = (o_k2[:, None] < K) & m_v[None, :]
                    tl.store(
                        ht_off + ptr_offset((o_k2[:, None], o_v[None, :]), (V, 1)),
                        b_h2.to(ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        mask=m_ht,
                    )
                if K > 128:
                    m_ht = (o_k3[:, None] < K) & m_v[None, :]
                    tl.store(
                        ht_off + ptr_offset((o_k3[:, None], o_v[None, :]), (V, 1)),
                        b_h3.to(ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        mask=m_ht,
                    )
                if K > 192:
                    m_ht = (o_k4[:, None] < K) & m_v[None, :]
                    tl.store(
                        ht_off + ptr_offset((o_k4[:, None], o_v[None, :]), (V, 1)),
                        b_h4.to(ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        mask=m_ht,
                    )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    chunk_size: int = 64,
    output_final_state: bool = True,
    metadata: ChunkMetadata | RaggedChunkMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run the fixed-length or packed inter-chunk KDA state recurrence."""
    batch, tokens, heads, key_dim = k.shape
    value_dim = u.shape[-1]
    cu_seqlens = None if metadata is None else metadata.cu_seqlens
    if isinstance(metadata, RaggedChunkMetadata):
        metadata.validate_chunk_size(chunk_size)
        chunks = metadata.capacity
        chunk_offsets = metadata.chunk_offsets
    else:
        chunks = tokens // chunk_size
        chunk_offsets = None if cu_seqlens is None else cu_seqlens // chunk_size
    if tokens % chunk_size and not isinstance(metadata, RaggedChunkMetadata):
        raise ValueError(
            f"the inter-chunk state recurrence requires complete chunks, got T={tokens}"
        )
    if key_dim > 256:
        raise ValueError(f"the inter-chunk state recurrence requires K <= 256, got {key_dim}")
    if w.shape != k.shape or gk.shape != k.shape:
        raise ValueError("k, w, and gk must have the same shape")
    if u.shape != (batch, tokens, heads, value_dim):
        raise ValueError("u must have shape [B, T, H, V]")
    if cu_seqlens is not None and batch != 1:
        raise ValueError("packed cu_seqlens require batch size one")
    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    expected_state_shape = (state_batch, heads, key_dim, value_dim)
    if initial_state is not None:
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"initial_state must have shape {expected_state_shape}, "
                f"got {tuple(initial_state.shape)}"
            )
        initial_state = initial_state.contiguous()

    h = k.new_empty(batch, chunks, heads, key_dim, value_dim)
    v_new = torch.empty_like(u)
    final_state = (
        torch.empty(expected_state_shape, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    def grid(meta):
        return (state_batch * heads, triton.cdiv(value_dim, meta["BV"]))

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=None,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        num_seqs=None,
        T=tokens,
        H=heads,
        K=key_dim,
        V=value_dim,
        BT=chunk_size,
        USE_EXP2=True,
    )
    return h, v_new, final_state


__all__ = ["chunk_gated_delta_rule_fwd_h"]
