# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import triton
import triton.language as tl
from attn_gym.linear.kda.utils import (
    exp,
    exp2,
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
    }
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
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        if HAS_NUM_SEQS:
            if i_n >= tl.load(num_seqs):
                return
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
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
    h += (boh * H + i_h).to(tl.int64) * K * V
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * H + i_h).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = tl.make_block_ptr(h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            p_h0_2 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            b_h2 += tl.load(  # pyrefly: ignore[unbound-name]
                p_h0_2, boundary_check=(0, 1)
            ).to(  # pyrefly: ignore[unbound-name]
                tl.float32
            )  # pyrefly: ignore[unbound-name]
        if K > 128:
            p_h0_3 = tl.make_block_ptr(
                h0,
                (K, V),
                (V, 1),
                (128, i_v * BV),
                (64, BV),
                (1, 0),  # pyrefly: ignore[unbound-name]
            )
            b_h3 += tl.load(  # pyrefly: ignore[unbound-name]
                p_h0_3, boundary_check=(0, 1)
            ).to(  # pyrefly: ignore[unbound-name]
                tl.float32
            )  # pyrefly: ignore[unbound-name]
        if K > 192:  # pyrefly: ignore[unbound-name]
            p_h0_4 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            b_h4 += tl.load(  # pyrefly: ignore[unbound-name]
                p_h0_4, boundary_check=(0, 1)
            ).to(  # pyrefly: ignore[unbound-name]
                tl.float32
            )  # pyrefly: ignore[unbound-name]

    # main recurrence
    for i_t in range(NT):
        p_h1 = tl.make_block_ptr(
            h + i_t * H * K * V, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
        )  # pyrefly: ignore[unbound-name]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_h2 = tl.make_block_ptr(
                h + i_t * H * K * V, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )  # pyrefly: ignore[unbound-name]
            tl.store(
                p_h2,
                b_h2.to(p_h2.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
            )  # pyrefly: ignore[unbound-name]
        if K > 128:
            p_h3 = tl.make_block_ptr(  # pyrefly: ignore[unbound-name]
                h + i_t * H * K * V, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(
                p_h3,
                b_h3.to(p_h3.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
            )  # pyrefly: ignore[unbound-name]
        if K > 192:
            p_h4 = tl.make_block_ptr(
                h + i_t * H * K * V, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(  # pyrefly: ignore[unbound-name]
                p_h4,
                b_h4.to(p_h4.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
            )  # pyrefly: ignore[unbound-name]

        p_w = tl.make_block_ptr(w, (T, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (H * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
        p_v = tl.make_block_ptr(
            v, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = tl.make_block_ptr(  # pyrefly: ignore[unbound-name]
                g + bos * H + i_h, (T,), (H,), (i_t * BT,), (BT,), (0,)
            )  # pyrefly: ignore[unbound-name]
            b_g = tl.load(p_g, boundary_check=(0,))
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
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            if USE_EXP2:  # pyrefly: ignore[unbound-name,unsupported-operation]
                b_h1 *= exp2(b_gk_last1)[
                    :, None  # pyrefly: ignore[unbound-name,unsupported-operation]
                ]  # pyrefly: ignore[unsupported-operation]
            else:
                b_h1 *= exp(b_gk_last1)[
                    :, None
                ]  # pyrefly: ignore[unsupported-operation]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk
                    + (bos + last_idx) * H * K
                    + i_h * K
                    + o_k2,  # pyrefly: ignore[unbound-name,unsupported-operation]
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
                o_k3 = 128 + o_k1  # pyrefly: ignore[unbound-name,unsupported-operation]
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
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
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk
                    + (bos + last_idx) * H * K
                    + i_h * K
                    + o_k4,  # pyrefly: ignore[unbound-name]
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

        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))  # pyrefly: ignore[unbound-name]
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1)
            )  # pyrefly: ignore[unbound-name]
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h2 += tl.dot(b_k, b_v)  # pyrefly: ignore[unbound-name]
        if K > 128:
            p_k = tl.make_block_ptr(
                k,
                (K, T),
                (1, H * K),
                (128, i_t * BT),
                (64, BT),
                (0, 1),  # pyrefly: ignore[unbound-name]
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h3 += tl.dot(b_k, b_v)  # pyrefly: ignore[unbound-name]
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            b_h4 += tl.dot(b_k, b_v)  # pyrefly: ignore[unbound-name]

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0))
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(
                p_ht,
                b_h2.to(p_ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
            )  # pyrefly: ignore[unbound-name]
        if K > 128:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(
                p_ht,
                b_h3.to(p_ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
            )  # pyrefly: ignore[unbound-name]
        if K > 192:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
            )
            tl.store(
                p_ht,
                b_h4.to(p_ht.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
            )  # pyrefly: ignore[unbound-name]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NUM_SEQS": lambda args: args["num_seqs"] is not None,
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
    GRID_N: tl.constexpr,
    MAX_N: tl.constexpr,
):
    i_nh, i_v = tl.program_id(0), tl.program_id(1)
    i_h = i_nh % H
    i_n_start = i_nh // H

    if HAS_NUM_SEQS:
        upper_limit = tl.load(num_seqs)

    for _iter in range((MAX_N + GRID_N - 1) // GRID_N):
        i_n = i_n_start + _iter * GRID_N
        _run = i_n < MAX_N
        if IS_VARLEN:
            if HAS_NUM_SEQS:
                if _run:
                    _run = i_n < upper_limit  # pyrefly: ignore [unbound-name]
        if _run:
            if IS_VARLEN:
                bos, eos = (
                    tl.load(cu_seqlens + i_n).to(tl.int32),
                    tl.load(cu_seqlens + i_n + 1).to(tl.int32),
                )
                T_local = eos - bos
                NT = tl.cdiv(T_local, BT)
                boh = tl.load(chunk_offsets + i_n).to(tl.int32)
            else:
                bos, eos = i_n * T, i_n * T + T
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
            h_off = h + (boh * H + i_h).to(tl.int64) * K * V
            v_off = v + (bos * H + i_h).to(tl.int64) * V
            k_off = k + (bos * H + i_h).to(tl.int64) * K
            w_off = w + (bos * H + i_h).to(tl.int64) * K
            if SAVE_NEW_VALUE:
                v_new_off = v_new + (bos * H + i_h).to(tl.int64) * V

            i_nh_abs = i_n * H + i_h

            # load initial state
            if USE_INITIAL_STATE:
                h0_off = h0 + i_nh_abs * K * V
                p_h0_1 = tl.make_block_ptr(
                    h0_off, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
                )
                b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
                if K > 64:
                    p_h0_2 = tl.make_block_ptr(
                        h0_off, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                    )
                    # pyrefly: ignore [unbound-name]
                    b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
                if K > 128:
                    p_h0_3 = tl.make_block_ptr(
                        h0_off,
                        (K, V),
                        (V, 1),
                        (128, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    # pyrefly: ignore [unbound-name]
                    b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
                if K > 192:
                    p_h0_4 = tl.make_block_ptr(
                        h0_off, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                    )
                    # pyrefly: ignore [unbound-name]
                    b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

            # main recurrence
            for i_t in range(NT):
                p_h1 = tl.make_block_ptr(
                    h_off + i_t * H * K * V,
                    (K, V),
                    (V, 1),
                    (0, i_v * BV),
                    (64, BV),
                    (1, 0),
                )  # pyrefly: ignore[unbound-name]
                tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    p_h2 = tl.make_block_ptr(
                        h_off + i_t * H * K * V,
                        (K, V),
                        (V, 1),
                        (64, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )  # pyrefly: ignore[unbound-name]
                    tl.store(
                        p_h2,
                        b_h2.to(p_h2.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
                    )  # pyrefly: ignore[unbound-name]
                if K > 128:
                    p_h3 = tl.make_block_ptr(  # pyrefly: ignore[unbound-name]
                        h_off + i_t * H * K * V,
                        (K, V),
                        (V, 1),
                        (128, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    tl.store(
                        p_h3,
                        b_h3.to(p_h3.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
                    )  # pyrefly: ignore[unbound-name]
                if K > 192:
                    p_h4 = tl.make_block_ptr(
                        h_off + i_t * H * K * V,
                        (K, V),
                        (V, 1),
                        (192, i_v * BV),
                        (64, BV),
                        (1, 0),
                    )
                    tl.store(  # pyrefly: ignore[unbound-name]
                        p_h4,
                        b_h4.to(p_h4.dtype.element_ty),  # pyrefly: ignore[unbound-name]
                        boundary_check=(0, 1),  # pyrefly: ignore[unbound-name]
                    )  # pyrefly: ignore[unbound-name]

                p_w = tl.make_block_ptr(
                    w_off, (T_local, K), (H * K, 1), (i_t * BT, 0), (BT, 64), (1, 0)
                )
                b_w = tl.load(p_w, boundary_check=(0, 1))
                b_v = tl.dot(b_w, b_h1.to(b_w.dtype))  # pyrefly: ignore[unbound-name]
                if K > 64:
                    p_w = tl.make_block_ptr(
                        w_off,
                        (T_local, K),
                        (H * K, 1),
                        (i_t * BT, 64),
                        (BT, 64),
                        (1, 0),
                    )
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    b_v += tl.dot(
                        b_w,
                        b_h2.to(b_w.dtype),  # pyrefly: ignore [unbound-name]
                    )  # pyrefly: ignore[unbound-name]
                if K > 128:
                    p_w = tl.make_block_ptr(
                        w_off,
                        (T_local, K),
                        (H * K, 1),
                        (i_t * BT, 128),
                        (BT, 64),
                        (1, 0),
                    )
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    b_v += tl.dot(
                        b_w,
                        b_h3.to(b_w.dtype),  # pyrefly: ignore [unbound-name]
                    )  # pyrefly: ignore[unbound-name]
                if K > 192:
                    p_w = tl.make_block_ptr(
                        w_off,
                        (T_local, K),
                        (H * K, 1),
                        (i_t * BT, 192),
                        (BT, 64),
                        (1, 0),
                    )
                    b_w = tl.load(p_w, boundary_check=(0, 1))
                    b_v += tl.dot(
                        b_w,
                        b_h4.to(b_w.dtype),  # pyrefly: ignore [unbound-name]
                    )  # pyrefly: ignore[unbound-name]
                p_v = tl.make_block_ptr(
                    v_off,
                    (T_local, V),
                    (H * V, 1),
                    (i_t * BT, i_v * BV),
                    (BT, BV),
                    (1, 0),
                )
                b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

                if SAVE_NEW_VALUE:
                    p_v = tl.make_block_ptr(
                        # pyrefly: ignore [unbound-name]
                        v_new_off,
                        (T_local, V),
                        (H * V, 1),
                        (i_t * BT, i_v * BV),
                        (BT, BV),
                        (1, 0),
                    )
                    tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

                last_idx = min((i_t + 1) * BT, T_local) - 1
                if USE_G:
                    m_t = (i_t * BT + tl.arange(0, BT)) < T_local
                    b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
                    p_g = tl.make_block_ptr(  # pyrefly: ignore[unbound-name]
                        g + bos * H + i_h, (T_local,), (H,), (i_t * BT,), (BT,), (0,)
                    )  # pyrefly: ignore[unbound-name]
                    b_g = tl.load(p_g, boundary_check=(0,))
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
                    o_k1 = tl.arange(0, 64)
                    b_gk_last1 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                        mask=(o_k1 < K),
                        other=0.0,
                    )
                    if USE_EXP2:  # pyrefly: ignore[unbound-name,unsupported-operation]
                        b_h1 *= exp2(b_gk_last1)[
                            :,
                            None,  # pyrefly: ignore[unbound-name,unsupported-operation]
                        ]  # pyrefly: ignore[unsupported-operation]
                    else:
                        b_h1 *= exp(b_gk_last1)[
                            :, None
                        ]  # pyrefly: ignore[unsupported-operation]
                    if K > 64:
                        o_k2 = 64 + o_k1
                        b_gk_last2 = tl.load(
                            gk
                            + (bos + last_idx) * H * K
                            + i_h * K
                            + o_k2,  # pyrefly: ignore[unbound-name,unsupported-operation]
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
                        o_k3 = (
                            128 + o_k1
                        )  # pyrefly: ignore[unbound-name,unsupported-operation]
                        b_gk_last3 = tl.load(
                            gk + (bos + last_idx) * H * K + i_h * K + o_k3,
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
                        o_k4 = 192 + o_k1
                        b_gk_last4 = tl.load(
                            gk
                            + (bos + last_idx) * H * K
                            + i_h * K
                            + o_k4,  # pyrefly: ignore[unbound-name]
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

                p_k = tl.make_block_ptr(
                    k_off, (K, T_local), (1, H * K), (0, i_t * BT), (64, BT), (0, 1)
                )
                b_k = tl.load(
                    p_k, boundary_check=(0, 1)
                )  # pyrefly: ignore[unbound-name]
                b_h1 += tl.dot(b_k, b_v)
                if K > 64:
                    p_k = tl.make_block_ptr(
                        k_off,
                        (K, T_local),
                        (1, H * K),
                        (64, i_t * BT),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h2 += tl.dot(b_k, b_v)  # pyrefly: ignore [unbound-name]
                if K > 128:
                    p_k = tl.make_block_ptr(
                        k_off,
                        (K, T_local),
                        (1, H * K),
                        (128, i_t * BT),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h3 += tl.dot(b_k, b_v)  # pyrefly: ignore [unbound-name]
                if K > 192:
                    p_k = tl.make_block_ptr(
                        k_off,
                        (K, T_local),
                        (1, H * K),
                        (192, i_t * BT),
                        (64, BT),
                        (0, 1),
                    )
                    b_k = tl.load(p_k, boundary_check=(0, 1))
                    b_h4 += tl.dot(b_k, b_v)  # pyrefly: ignore[unbound-name]

            if STORE_FINAL_STATE:
                ht_off = ht + i_nh_abs * K * V
                p_ht = tl.make_block_ptr(
                    ht_off, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
                )
                tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
                if K > 64:
                    p_ht = tl.make_block_ptr(
                        ht_off, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                    )
                    tl.store(
                        # pyrefly: ignore [unbound-name]
                        p_ht,
                        # pyrefly: ignore [unbound-name]
                        b_h2.to(p_ht.dtype.element_ty),
                        boundary_check=(0, 1),
                    )
                if K > 128:
                    p_ht = tl.make_block_ptr(
                        ht_off, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                    )
                    tl.store(
                        # pyrefly: ignore [unbound-name]
                        p_ht,
                        # pyrefly: ignore [unbound-name]
                        b_h3.to(p_ht.dtype.element_ty),
                        boundary_check=(0, 1),
                    )
                if K > 192:
                    p_ht = tl.make_block_ptr(
                        ht_off, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                    )
                    tl.store(
                        # pyrefly: ignore [unbound-name]
                        p_ht,
                        # pyrefly: ignore [unbound-name]
                        b_h4.to(p_ht.dtype.element_ty),
                        boundary_check=(0, 1),
                    )

