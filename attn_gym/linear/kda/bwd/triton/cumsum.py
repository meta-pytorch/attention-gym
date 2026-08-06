# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Chunk-local cumulative sum along the time axis. In the KDA backward pass the
# per-channel decay gate is accumulated *in reverse* within each chunk (the
# adjoint of a forward prefix-sum is a reverse prefix-sum), so both directions
# are supported.
#
# Two layouts are handled:
#   * scalar gates  (B, T, H)      -> chunk_local_cumsum_scalar_kernel
#   * vector gates  (B, T, H, S)   -> chunk_local_cumsum_vector_kernel
# with optional variable-length packing via (cu_seqlens, chunk_indices) and an
# optional output scale.

import triton
import triton.language as tl

from attn_gym.linear.kda.utils import autotune_cache_kwargs, check_shared_mem

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    # Resolve the row span of this chunk (constant time for the fixed-length
    # case, or looked up from the packing metadata for varlen batches).
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    if HEAD_FIRST:
        off = bos * H + i_h * T + o_t
    else:
        off = bos * H + i_h + o_t * H

    b_s = tl.load(s + off, mask=m_t, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
    if HAS_SCALE:
        b_o = b_o * scale
    tl.store(o + off, b_o.to(o.dtype.element_ty), mask=m_t)


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BS": bs}, num_warps=w) for bs in BS_LIST for w in [2, 4, 8]],
    key=["B", "H", "S", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m = (o_t[:, None] < T) & (o_s[None, :] < S)
    # (B, T, H, S) row-major: token stride H*S, head offset i_h*S (or i_h*T*S
    # when the head axis precedes time in a head-first packing).
    if HEAD_FIRST:
        off = (bos * H + i_h * T) * S + o_t[:, None] * S + o_s[None, :]
    else:
        off = (bos * H + i_h) * S + o_t[:, None] * (H * S) + o_s[None, :]

    b_s = tl.load(s + off, mask=m, other=0.0).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
    if HAS_SCALE:
        b_o = b_o * scale
    tl.store(o + off, b_o.to(o.dtype.element_ty), mask=m)
