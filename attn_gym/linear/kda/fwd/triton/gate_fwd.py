# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# FORWARD-ONLY KDA gate chunk-cumsum, derived from:
#   genai/llama4x/llama4x/ops/fla/ops/kda/gate.py
#
# The gate math and launch configuration follow the source implementation. The
# indexing uses logical stride tuples so inputs and outputs need not be contiguous.
# The `softplus` helper (originally `fla.ops.utils.softplus`) remains inlined below.

from __future__ import annotations

from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, storage_cosize
from attn_gym.linear.kda.bwd.triton.gate_bwd import kda_gate_bwd_ragged
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    chunk_capacity,
    load_ragged_chunk_work,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.kda.utils import (
    IS_NVIDIA,
    RCP_LN2,
    autotune_cache_kwargs,
    exp,
    input_guard,
    prepare_chunk_indices,
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
    return ",".join("=r" for i in range(num_pack)) + "," + ",".join("r" for i in range(num_pack))


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
# Chunk-cumsum forward kernels.
# s_batch_stride + S_STRIDES describe s as logical (B, T, H, S_in), while
# o_batch_stride + O_STRIDES describe o as logical (B, T, H, S). In varlen mode,
# T is the total number of packed tokens.
# ---------------------------------------------------------------------------
def _requires_int64_offsets(args):
    """Select 64-bit indexing when any tensor offset can exceed int32."""
    input_cosize = storage_cosize(
        (args["B"], args["T"], args["H"], args["S_in"]),
        (args["s_batch_stride"], *args["S_STRIDES"]),
    )
    output_cosize = storage_cosize(
        (args["B"], args["T"], args["H"], args["S"]),
        (args["o_batch_stride"], *args["O_STRIDES"]),
    )
    bias_cosize = storage_cosize((args["H"], args["S"]), args["DT_BIAS_STRIDES"])
    a_log_cosize = storage_cosize((args["H"],), args["A_LOG_STRIDES"])
    return max(input_cosize, output_cosize, bias_cosize, a_log_cosize) > 1 << 31


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "USE_REPEAT": lambda args: args["S_in"] != args["S"],
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
        "USE_INT64_OFFSETS": _requires_int64_offsets,
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
    s_batch_stride,
    o_batch_stride,
    S_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
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
    USE_INT64_OFFSETS: tl.constexpr,
):
    tl.static_assert(not IS_VARLEN or B == 1, "packed varlen requires B == 1")
    i_t = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_s = tl.program_id(2)
    if USE_INT64_OFFSETS:
        i_t = i_t.to(tl.int64)
        i_bh = i_bh.to(tl.int64)
        i_s = i_s.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        if HAS_NUM_CHUNKS and i_t >= tl.load(num_chunks):
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
        if USE_INT64_OFFSETS:
            i_t = i_t.to(tl.int64)
            bos = bos.to(tl.int64)
    else:
        bos = 0

    b_out_cols = i_s * BS + tl.arange(0, BS)
    b_in_cols = b_out_cols // F_REPEAT if USE_REPEAT else b_out_cols
    b_t_offs = i_t * BT + tl.arange(0, BT)
    token = bos + b_t_offs if IS_VARLEN else b_t_offs
    m_t = b_t_offs < T
    b_mask = m_t[:, None] & (b_out_cols[None, :] < S)
    s_batch_offset = i_b * s_batch_stride if B > 1 else 0
    b_s = tl.load(
        s
        + s_batch_offset
        + ptr_offset(
            (token[:, None], i_h, b_in_cols[None, :]),
            S_STRIDES,
        ),
        mask=b_mask,
        other=0.0,
    ).to(tl.float32)

    # Apply dt_bias if exists (dt_bias is always in full dimension S)
    if HAS_BIAS:
        b_bias = tl.load(
            dt_bias + ptr_offset((i_h, b_out_cols), DT_BIAS_STRIDES),
            mask=b_out_cols < S,
            other=0.0,
        ).to(tl.float32)
        b_s = b_s + b_bias[None, :]

    b_A = tl.load(A_log + ptr_offset((i_h,), A_LOG_STRIDES)).to(tl.float32)
    if not USE_LOWER_BOUND:  # pyrefly: ignore[unsupported-operation]
        # Apply gate: -exp(A_log) * softplus(g + bias)
        b_gate = -exp(b_A) * softplus(b_s)  # pyrefly: ignore[unsupported-operation]
    else:
        # Inductor passes Python floats as fp64; keep the gate math in fp32.
        b_gate = lower_bound.to(tl.float32) * tl.sigmoid(exp(b_A) * b_s)

    b_gate = tl.where(m_t[:, None], b_gate, 0.0)

    # Apply chunk local cumsum
    b_o = tl.cumsum(b_gate, axis=0, reverse=REVERSE)

    if HAS_SCALE:
        b_o *= scale.to(tl.float32)
    o_batch_offset = i_b * o_batch_stride if B > 1 else 0
    tl.store(
        o
        + o_batch_offset
        + ptr_offset(
            (token[:, None], i_h, b_out_cols[None, :]),
            O_STRIDES,
        ),
        b_o.to(o.dtype.element_ty),
        mask=b_mask,
    )


@triton.jit(do_not_specialize=["num_sequences"])
def bounded_gate_chunk_cumsum_ragged_kernel(
    raw_gate,
    A_log,
    dt_bias,
    output,
    lower_bound,
    scale,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    G_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    """Apply the bounded gate and prefix sum to one device-routed ragged chunk."""
    global_chunk = tl.program_id(0)
    i_h = tl.program_id(1).to(tl.int64)
    i_d = tl.program_id(2).to(tl.int64)
    if global_chunk >= tl.load(chunk_offsets + num_sequences):
        return

    _sequence, _local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
        cu_seqlens,
        chunk_offsets,
        global_chunk,
        num_sequences,
        BT,
    )
    token_offset = tl.arange(0, BT)
    token = token_start.to(tl.int64) + token_offset
    channel = i_d * BD + tl.arange(0, BD)
    mask = (token_offset[:, None] < valid_tokens) & (channel[None, :] < D)
    gate_input = tl.load(
        raw_gate + ptr_offset((token[:, None], i_h, channel[None, :]), G_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(
        dt_bias + ptr_offset((i_h, channel), DT_BIAS_STRIDES),
        mask=channel < D,
        other=0.0,
    ).to(tl.float32)
    decay = tl.load(A_log + ptr_offset((i_h,), A_LOG_STRIDES)).to(tl.float32)
    gate = lower_bound.to(tl.float32) * tl.sigmoid(exp(decay) * (gate_input + bias[None, :]))
    gate = tl.where(mask, gate, 0.0)
    cumulative = tl.cumsum(gate, axis=0) * scale
    tl.store(
        output + ptr_offset((token[:, None], i_h, channel[None, :]), O_STRIDES),
        cumulative,
        mask=mask,
    )


@input_guard(no_guard_contiguous=True)
def bounded_gate_cumsum_ragged(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    chunk_size: int,
    lower_bound: float,
) -> tuple[torch.Tensor, RaggedChunkMetadata]:
    """Prepare device routing and run the graph-safe packed gate forward."""
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, raw_gate.shape[1], chunk_size)
    output = torch.empty_like(raw_gate, dtype=torch.float32)
    _, _, heads, head_dim = raw_gate.shape
    block_dim = 32
    bounded_gate_chunk_cumsum_ragged_kernel[
        (metadata.capacity, heads, triton.cdiv(head_dim, block_dim))
    ](
        raw_gate,
        A_log,
        dt_bias,
        output,
        lower_bound,
        RCP_LN2,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        metadata.cu_seqlens.shape[0] - 1,
        G_STRIDES=raw_gate.stride()[1:],
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=dt_bias.stride(),
        O_STRIDES=output.stride()[1:],
        D=head_dim,
        BT=metadata.chunk_size,
        BD=block_dim,
        num_warps=2,
        num_stages=3,
    )
    return output, metadata


torch.library.define(
    "attn_gym::kda_bounded_gate_fwd_ragged",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, Tensor cu_seqlens, "
    "int chunk_size, float lower_bound) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_bounded_gate_bwd_ragged",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, Tensor d_cumulative, "
    "Tensor cu_seqlens, Tensor chunk_offsets, int chunk_size, float lower_bound, "
    "bool profile_ranges) -> (Tensor, Tensor, Tensor)",
)


def _bounded_gate_cumsum_ragged_fwd_cuda(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ragged gate activation, scheduling, and prefix sums as one functional op."""
    output, metadata = bounded_gate_cumsum_ragged(
        raw_gate,
        A_log,
        dt_bias,
        cu_seqlens,
        chunk_size=chunk_size,
        lower_bound=lower_bound,
    )
    return output, metadata.chunk_offsets


def _bounded_gate_cumsum_ragged_bwd_cuda(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
    profile_ranges: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the packed gate prefix sum with one fused launch."""
    metadata = RaggedChunkMetadata(
        cu_seqlens,
        chunk_offsets,
        chunk_capacity(raw_gate.shape[1], cu_seqlens.shape[0] - 1, chunk_size),
        chunk_size,
    )
    with (
        torch.profiler.record_function("kda/triton/gate_backward_ragged")
        if profile_ranges
        else nullcontext()
    ):
        return kda_gate_bwd_ragged(
            raw_gate,
            A_log,
            dt_bias,
            d_cumulative,
            metadata,
            lower_bound=lower_bound,
            scale=RCP_LN2,
        )


torch.library.impl(
    "attn_gym::kda_bounded_gate_fwd_ragged", "CUDA", _bounded_gate_cumsum_ragged_fwd_cuda
)
torch.library.impl(
    "attn_gym::kda_bounded_gate_bwd_ragged", "CUDA", _bounded_gate_cumsum_ragged_bwd_cuda
)


@torch.library.register_fake("attn_gym::kda_bounded_gate_fwd_ragged")
def _bounded_gate_cumsum_ragged_fwd_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe ragged gate output and scheduler tape metadata."""
    del A_log, dt_bias, chunk_size, lower_bound
    return torch.empty_like(raw_gate, dtype=torch.float32), torch.empty_like(cu_seqlens)


@torch.library.register_fake("attn_gym::kda_bounded_gate_bwd_ragged")
def _bounded_gate_cumsum_ragged_bwd_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
    profile_ranges: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe ragged gate input and parameter gradients."""
    del d_cumulative, cu_seqlens, chunk_offsets, chunk_size, lower_bound, profile_ranges
    return (
        torch.empty_like(raw_gate),
        A_log.new_empty(A_log.shape),
        dt_bias.new_empty(dt_bias.shape),
    )


_bounded_gate_cumsum_ragged_fwd_op = torch.ops.attn_gym.kda_bounded_gate_fwd_ragged.default
_bounded_gate_cumsum_ragged_bwd_op = torch.ops.attn_gym.kda_bounded_gate_bwd_ragged.default


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_LOWER_BOUND": lambda args: args["lower_bound"] is not None,
        "USE_REPEAT": lambda args: args["S_in"] != args["S"],
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
        "USE_INT64_OFFSETS": _requires_int64_offsets,
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
    s_batch_stride,
    o_batch_stride,
    S_STRIDES: tl.constexpr,
    A_LOG_STRIDES: tl.constexpr,
    DT_BIAS_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
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
    USE_INT64_OFFSETS: tl.constexpr,
    # pyrefly: ignore [bad-function-definition]
    GRID_NT: tl.constexpr = 0,
    # pyrefly: ignore [bad-function-definition]
    MAX_NT: tl.constexpr = 0,
):
    tl.static_assert(not IS_VARLEN or B == 1, "packed varlen requires B == 1")
    i_t_start = tl.program_id(0)
    i_bh = tl.program_id(1)
    i_s = tl.program_id(2)
    if USE_INT64_OFFSETS:
        i_t_start = i_t_start.to(tl.int64)
        i_bh = i_bh.to(tl.int64)
        i_s = i_s.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    b_out_cols = i_s * BS + tl.arange(0, BS)
    b_in_cols = b_out_cols // F_REPEAT if USE_REPEAT else b_out_cols
    s_batch_offset = i_b * s_batch_stride if B > 1 else 0
    o_batch_offset = i_b * o_batch_stride if B > 1 else 0
    p_s = s + s_batch_offset + ptr_offset((0, i_h, b_in_cols), S_STRIDES)
    p_o = o + o_batch_offset + ptr_offset((0, i_h, b_out_cols), O_STRIDES)
    if HAS_BIAS:
        b_bias = tl.load(
            dt_bias + ptr_offset((i_h, b_out_cols), DT_BIAS_STRIDES),
            mask=b_out_cols < S,
            other=0.0,
        ).to(tl.float32)
    b_A = tl.load(A_log + ptr_offset((i_h,), A_LOG_STRIDES)).to(tl.float32)

    for _iter in range((MAX_NT + GRID_NT - 1) // GRID_NT):
        i_t_orig = i_t_start + _iter * GRID_NT
        _run = i_t_orig < MAX_NT
        if IS_VARLEN and HAS_NUM_CHUNKS and _run:
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
                if USE_INT64_OFFSETS:
                    i_t = i_t.to(tl.int64)
                    bos = bos.to(tl.int64)
            else:
                i_t = i_t_orig
                bos = 0
                T_local = T

            b_t_offs = i_t * BT + tl.arange(0, BT)
            token = bos + b_t_offs if IS_VARLEN else b_t_offs
            m_t = b_t_offs < T_local
            b_mask = m_t[:, None] & (b_out_cols[None, :] < S)
            b_s = tl.load(
                p_s[None, :] + token[:, None] * S_STRIDES[0],
                mask=b_mask,
                other=0.0,
            ).to(tl.float32)

            if HAS_BIAS:
                b_s = b_s + b_bias[None, :]
            if not USE_LOWER_BOUND:  # pyrefly: ignore[unsupported-operation]
                # pyrefly: ignore [unsupported-operation]
                b_gate = -exp(b_A) * softplus(b_s)  # pyrefly: ignore[unsupported-operation]
            else:
                # Inductor passes Python floats as fp64; keep the gate math in fp32.
                b_gate = lower_bound.to(tl.float32) * tl.sigmoid(exp(b_A) * b_s)

            b_gate = tl.where(m_t[:, None], b_gate, 0.0)
            b_o = tl.cumsum(b_gate, axis=0, reverse=REVERSE)

            if HAS_SCALE:
                b_o *= scale.to(tl.float32)
            tl.store(
                p_o[None, :] + token[:, None] * O_STRIDES[0],
                b_o.to(o.dtype.element_ty),
                mask=b_mask,
            )


@input_guard(no_guard_contiguous=True)
def kda_gate_chunk_cumsum(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    *,
    chunk_size: int = 64,
    lower_bound: float | None = None,
    scale: float | None = None,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    num_chunks: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Apply the KDA gate map and chunk-local cumulative sum in one launch."""
    if g.ndim != 4:
        raise ValueError(f"g must have shape [B, T, H, D], got {tuple(g.shape)}")
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError(f"chunk_size must be a positive power of two, got {chunk_size}")
    batch, tokens, heads, head_dim = g.shape
    if batch == 0 or tokens == 0 or heads == 0 or head_dim == 0:
        raise ValueError(f"g must have no empty dimensions, got {tuple(g.shape)}")
    if A_log.shape != (heads,):
        raise ValueError(f"A_log must have shape {(heads,)}, got {tuple(A_log.shape)}")
    if dt_bias is not None and dt_bias.shape != (heads, head_dim):
        raise ValueError(
            f"dt_bias must have shape {(heads, head_dim)}, got {tuple(dt_bias.shape)}"
        )
    if cu_seqlens is not None and batch != 1:
        raise ValueError("packed variable-length inputs must have batch size one")
    for name, metadata in (
        ("cu_seqlens", cu_seqlens),
        ("chunk_indices", chunk_indices),
    ):
        if metadata is not None and not metadata.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunks = triton.cdiv(tokens, chunk_size) if cu_seqlens is None else len(chunk_indices)
    output = torch.empty_like(g, dtype=output_dtype)

    def grid(meta):
        return (chunks, batch * heads, triton.cdiv(head_dim, meta["BS"]))

    kda_gate_chunk_cumsum_vector_kernel[grid](
        g,
        A_log,
        dt_bias,
        output,
        scale,
        cu_seqlens,
        chunk_indices,
        lower_bound,
        tokens,
        num_chunks,
        g.stride(0),
        output.stride(0),
        S_STRIDES=g.stride()[1:],
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=(0, 0) if dt_bias is None else dt_bias.stride(),
        O_STRIDES=output.stride()[1:],
        B=batch,
        H=heads,
        S=head_dim,
        S_in=head_dim,
        F_REPEAT=1,
        BT=chunk_size,
        REVERSE=reverse,
    )
    return output


class _BoundedGateCumsum(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        raw_gate: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        chunk_size: int,
        lower_bound: float,
        fastmath: bool,
        profile_ranges: bool,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        ctx.chunk_size = chunk_size
        ctx.lower_bound = lower_bound
        ctx.fastmath = fastmath
        ctx.profile_ranges = profile_ranges
        ctx.ragged = cu_seqlens is not None
        if cu_seqlens is not None:
            output, chunk_offsets = _bounded_gate_cumsum_ragged_fwd_op(
                raw_gate,
                A_log,
                dt_bias,
                cu_seqlens,
                chunk_size,
                lower_bound,
            )
            ctx.save_for_backward(raw_gate, A_log, dt_bias, cu_seqlens, chunk_offsets)
            return output

        ctx.save_for_backward(raw_gate, A_log, dt_bias)
        return kda_gate_chunk_cumsum(
            raw_gate,
            A_log,
            dt_bias,
            chunk_size=chunk_size,
            lower_bound=lower_bound,
            scale=RCP_LN2,
        )

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_cumulative: torch.Tensor):
        if ctx.ragged:
            raw_gate, A_log, dt_bias, cu_seqlens, chunk_offsets = ctx.saved_tensors
            dg, dA_log, d_dt_bias = _bounded_gate_cumsum_ragged_bwd_op(
                raw_gate,
                A_log,
                dt_bias,
                d_cumulative,
                cu_seqlens,
                chunk_offsets,
                ctx.chunk_size,
                ctx.lower_bound,
                ctx.profile_ranges,
            )
            return (
                dg,
                dA_log,
                d_dt_bias,
                None,
                None,
                None,
                None,
                None,
            )

        from attn_gym.linear.kda.bwd.cute.gate_bwd_fused import fused_gate_bwd

        raw_gate, A_log, dt_bias = ctx.saved_tensors
        with (
            torch.profiler.record_function("kda/cute/gate_backward_fused")
            if ctx.profile_ranges
            else nullcontext()
        ):
            result = fused_gate_bwd(
                raw_gate,
                A_log,
                dt_bias,
                d_cumulative.float(),
                chunk_size=ctx.chunk_size,
                lower_bound=ctx.lower_bound,
                fastmath=ctx.fastmath,
            )
        return (
            result.dg,
            result.dA_partial.sum((0, 1)),
            result.d_dt_bias,
            None,
            None,
            None,
            None,
            None,
        )


def bounded_gate_cumsum(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    chunk_size: int = 64,
    lower_bound: float = -5.0,
    fastmath: bool = False,
    profile_ranges: bool = False,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply a bounded KDA gate and sequence-local chunk prefix sums.

    Packed ``cu_seqlens`` must start at zero, end at ``T``, and be monotonic; repeated
    offsets represent empty sequences. Values remain device-resident and are validated
    by the scheduler so fixed-shape CUDA Graphs can replay with different boundaries.
    """
    if raw_gate.ndim != 4:
        raise ValueError(f"raw_gate must have shape [B, T, H, D], got {tuple(raw_gate.shape)}")
    batch, tokens, heads, head_dim = raw_gate.shape
    if batch == 0 or tokens == 0 or heads == 0:
        raise ValueError(f"raw_gate must have nonempty B, T, and H, got {tuple(raw_gate.shape)}")
    if head_dim < 32 or head_dim > 1024 or head_dim % 32:
        raise ValueError(
            f"raw_gate head dimension must be a multiple of 32 in [32, 1024], got {head_dim}"
        )
    if not raw_gate.is_cuda or raw_gate.dtype != torch.bfloat16:
        raise TypeError("bounded_gate_cumsum requires a CUDA bfloat16 raw gate")
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        raise ValueError(
            f"A_log must be float32 with shape {(heads,)}, "
            f"got {tuple(A_log.shape)} and {A_log.dtype}"
        )
    if dt_bias.shape != (heads, head_dim) or dt_bias.dtype != torch.float32:
        raise ValueError(
            f"dt_bias must be float32 with shape {(heads, head_dim)}, "
            f"got {tuple(dt_bias.shape)} and {dt_bias.dtype}"
        )
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError(f"chunk_size must be a positive power of two, got {chunk_size}")
    if not all(tensor.device == raw_gate.device for tensor in (A_log, dt_bias)):
        raise ValueError("bounded_gate_cumsum inputs must be on the same device")
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed cu_seqlens require raw_gate to have batch size one")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
        if (
            cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != raw_gate.device
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError("cu_seqlens must be contiguous CUDA int32 on raw_gate.device")
        if fastmath:
            raise ValueError("packed bounded_gate_cumsum does not support fastmath=True")
    return _BoundedGateCumsum.apply(
        raw_gate,
        A_log,
        dt_bias,
        chunk_size,
        lower_bound,
        fastmath,
        profile_ranges,
        cu_seqlens,
    )


__all__ = ["bounded_gate_cumsum", "kda_gate_chunk_cumsum"]
