# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# FORWARD-ONLY KDA gate chunk-cumsum.
# Ported faithfully (functions copied verbatim) from:
#   genai/llama4x/llama4x/ops/fla/ops/kda/gate.py
#
# Only imports were rewritten to ``attn_gym.linear.kda.utils``. The `softplus`
# helper (originally `fla.ops.utils.softplus`) is inlined verbatim below.

from __future__ import annotations

import triton
import triton.language as tl
from attn_gym.linear.kda.utils import (
    autotune_cache_kwargs,
    exp,
    IS_NVIDIA,
)


# ---------------------------------------------------------------------------
# Inlined verbatim from fla.ops.utils.softplus.
# REVISED FROM
# https://github.com/shawntan/stickbreaking-attention/blob/main/stickbreaking_attention/sb_varlen/softplus.py
# ---------------------------------------------------------------------------
def _generate_softplus(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 20.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p mul.f32            ${out_reg}, ${in_reg}, 1.4426950408889634;
        @!p ex2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
        @!p mul.f32            ${out_reg}, ${out_reg}, 0.6931471805599453;
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_softplus2(num_pack):
    template = """
        .reg .pred p;
        setp.gt.f32  p, ${in_reg}, 15.;
        @p  mov.f32  ${out_reg}, ${in_reg};
        @!p ex2.approx.ftz.f32 ${out_reg}, ${in_reg};
        @!p add.f32            ${out_reg}, ${out_reg}, 1.0;
        @!p lg2.approx.ftz.f32 ${out_reg}, ${out_reg};
    """
    out_str = ""

    for i in range(num_pack):
        inner_str = template.format(out_reg=i, in_reg=i + num_pack)
        out_str += "{" + inner_str + "}\n"
    # flatten out because torch.compile doesn't like newlines
    out_str = " ".join(out_str.split("\n"))
    return out_str


def _generate_constraints(num_pack):
    return (
        ",".join("=r" for i in range(num_pack))
        + ","
        + ",".join("r" for i in range(num_pack))
    )


_NUM_REG = 1
s_softplus: tl.constexpr = tl.constexpr(_generate_softplus(_NUM_REG))
s_softplus2: tl.constexpr = tl.constexpr(_generate_softplus2(_NUM_REG))
s_constraints: tl.constexpr = tl.constexpr(_generate_constraints(_NUM_REG))
NUM_REG: tl.constexpr = tl.constexpr(_NUM_REG)


@triton.jit
def softplus_nv(x):
    # equivalent to:
    # return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)
    return tl.inline_asm_elementwise(
        asm=s_softplus,
        constraints=s_constraints,
        pack=NUM_REG,
        args=[
            x,
        ],
        dtype=tl.float32,
        is_pure=True,
    )


@triton.jit
def softplus_triton(x):
    return tl.where(x < 20.0, tl.math.log(1 + tl.math.exp(x)), x)


@triton.jit
def softplus2_nv(x):
    # equivalent to:
    # return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)
    return tl.inline_asm_elementwise(
        asm=s_softplus2,
        constraints=s_constraints,
        pack=NUM_REG,
        args=[
            x,
        ],
        dtype=tl.float32,
        is_pure=True,
    )


@triton.jit
def softplus2_triton(x):
    return tl.where(x < 15.0, tl.math.log2(1 + tl.math.exp2(x)), x)


if IS_NVIDIA:
    softplus = softplus_nv
    softplus2 = softplus2_nv
else:
    softplus = softplus_triton
    softplus2 = softplus2_triton


# ---------------------------------------------------------------------------
# Chunk-cumsum forward path (copied verbatim from the source gate.py).
# ---------------------------------------------------------------------------
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "USE_REPEAT": lambda args: args["S_in"] != args["S"],
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
    }
)
@triton.autotune(
    configs=[
        # Best config from autotuning: BS: 32, num_warps: 2, num_ctas: 1, num_stages: 3
        triton.Config({"BS": 32}, num_warps=2, num_stages=3),
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_chunks"])
def kda_gate_chunk_cumsum_vector_kernel(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    num_chunks,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    S_in: tl.constexpr,
    F_REPEAT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    USE_REPEAT: tl.constexpr,
    HAS_NUM_CHUNKS: tl.constexpr,
):
    i_t, i_bh, i_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        if HAS_NUM_CHUNKS:
            if i_t >= tl.load(num_chunks):
                return
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if USE_REPEAT:
        # Input g has reduced dimension S_in = S / F_REPEAT
        # We read from S_in and produce output of dimension S
        # i_s indexes output blocks of size BS in the full dimension S
        # Map output column indices to input column indices via integer division
        b_out_cols = i_s * BS + tl.arange(0, BS)  # [BS] output column indices
        b_in_cols = b_out_cols // F_REPEAT  # [BS] input column indices

        # Load g from reduced dimension using gather
        b_t_offs = i_t * BT + tl.arange(0, BT)  # [BT]
        b_s_ptrs = s + ((bos + b_t_offs[:, None]) * H + i_h) * S_in + b_in_cols[None, :]
        b_mask = (b_t_offs[:, None] < T) & (b_out_cols[None, :] < S)
        b_s = tl.load(b_s_ptrs, mask=b_mask, other=0.0).to(tl.float32)
    else:
        p_s = tl.make_block_ptr(
            s + (bos * H + i_h) * S,
            (T, S),
            (H * S, 1),
            (i_t * BT, i_s * BS),
            (BT, BS),
            (1, 0),
        )
        # [BT, BS]
        b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

    p_o = tl.make_block_ptr(
        o + (bos * H + i_h) * S,
        (T, S),
        (H * S, 1),
        (i_t * BT, i_s * BS),
        (BT, BS),
        (1, 0),
    )

    # Apply dt_bias if exists (dt_bias is always in full dimension S)
    if HAS_BIAS:
        p_b = tl.make_block_ptr(dt_bias + i_h * S, (S,), (1,), (i_s * BS,), (BS,), (0,))
        b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + i_h).to(tl.float32)
    if not USE_LOWER_BOUND:  # pyrefly: ignore[unsupported-operation]
        # Apply gate: -exp(A_log) * softplus(g + bias)
        b_gate = -exp(b_A) * softplus(b_s)  # pyrefly: ignore[unsupported-operation]
    else:
        b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

    # Apply chunk local cumsum
    if REVERSE:
        b_o = tl.cumsum(b_gate, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_gate, axis=0)

    if HAS_SCALE:
        b_o *= scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "USE_REPEAT": lambda args: args["S_in"] != args["S"],
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BS": 32}, num_warps=2, num_stages=3),
    ],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_chunks"])
def kda_gate_chunk_cumsum_vector_kernel_forloop(
    s,
    A_log,
    dt_bias,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    lower_bound,
    T,
    num_chunks,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    S_in: tl.constexpr,
    F_REPEAT: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    USE_REPEAT: tl.constexpr,
    HAS_NUM_CHUNKS: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    GRID_NT: tl.constexpr = 0,
    # pyrefly: ignore [bad-function-definition]
    MAX_NT: tl.constexpr = 0,
):
    i_t_start, i_bh, i_s = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    for _iter in range((MAX_NT + GRID_NT - 1) // GRID_NT):
        i_t_orig = i_t_start + _iter * GRID_NT
        _run = i_t_orig < MAX_NT
        if IS_VARLEN:
            if HAS_NUM_CHUNKS:
                if _run:
                    _run = i_t_orig < tl.load(num_chunks)
        if _run:
            if IS_VARLEN:
                i_n, i_t = (
                    tl.load(chunk_indices + i_t_orig * 2).to(tl.int32),
                    tl.load(chunk_indices + i_t_orig * 2 + 1).to(tl.int32),
                )
                bos, eos = (
                    tl.load(cu_seqlens + i_n).to(tl.int32),
                    tl.load(cu_seqlens + i_n + 1).to(tl.int32),
                )
                T_local = eos - bos
            else:
                i_t = i_t_orig
                bos, eos = i_b * T, i_b * T + T
                T_local = T

            if USE_REPEAT:
                b_out_cols = i_s * BS + tl.arange(0, BS)
                b_in_cols = b_out_cols // F_REPEAT

                b_t_offs = i_t * BT + tl.arange(0, BT)
                b_s_ptrs = (
                    s
                    + ((bos + b_t_offs[:, None]) * H + i_h) * S_in
                    + b_in_cols[None, :]
                )
                b_mask = (b_t_offs[:, None] < T_local) & (b_out_cols[None, :] < S)
                b_s = tl.load(b_s_ptrs, mask=b_mask, other=0.0).to(tl.float32)
            else:
                p_s = tl.make_block_ptr(
                    s + (bos * H + i_h) * S,
                    (T_local, S),
                    (H * S, 1),
                    (i_t * BT, i_s * BS),
                    (BT, BS),
                    (1, 0),
                )
                b_s = tl.load(p_s, boundary_check=(0, 1)).to(tl.float32)

            p_o = tl.make_block_ptr(
                o + (bos * H + i_h) * S,
                (T_local, S),
                (H * S, 1),
                (i_t * BT, i_s * BS),
                (BT, BS),
                (1, 0),
            )

            if HAS_BIAS:
                p_b = tl.make_block_ptr(
                    dt_bias + i_h * S, (S,), (1,), (i_s * BS,), (BS,), (0,)
                )
                b_bias = tl.load(p_b, boundary_check=(0,)).to(tl.float32)
                b_s = b_s + b_bias[None, :]

            b_A = tl.load(A_log + i_h).to(tl.float32)
            if not USE_LOWER_BOUND:  # pyrefly: ignore[unsupported-operation]
                # pyrefly: ignore [unsupported-operation]
                b_gate = -exp(b_A) * softplus(
                    b_s
                )  # pyrefly: ignore[unsupported-operation]
            else:
                b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_s)

            if REVERSE:
                b_o = tl.cumsum(b_gate, axis=0, reverse=True)
            else:
                b_o = tl.cumsum(b_gate, axis=0)

            if HAS_SCALE:
                b_o *= scale
            tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))

