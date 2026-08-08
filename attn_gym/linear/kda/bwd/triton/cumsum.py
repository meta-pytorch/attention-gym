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
# The kernels use logical (B, T, H[, S]) strides, including for physical
# head-first layouts. The leading batch stride is passed separately so Triton can
# specialize on its alignment without treating its T-dependent value as constexpr;
# inner strides remain constexpr for codegen. Variable-length packing and an
# optional output scale are also supported.

import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
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
    s_batch_stride,
    o_batch_stride,
    S_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    tl.static_assert(not IS_VARLEN or B == 1, "packed varlen requires B == 1")
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
        bos = 0

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    token = bos + o_t if IS_VARLEN else o_t

    s_batch_offset = i_b * s_batch_stride if B > 1 else 0
    b_s = tl.load(
        s + s_batch_offset + ptr_offset((token, i_h), S_STRIDES),
        mask=m_t,
        other=0.0,
    ).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
    if HAS_SCALE:
        b_o = b_o * scale
    o_batch_offset = i_b * o_batch_stride if B > 1 else 0
    tl.store(
        o + o_batch_offset + ptr_offset((token, i_h), O_STRIDES),
        b_o.to(o.dtype.element_ty),
        mask=m_t,
    )


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
    s_batch_stride,
    o_batch_stride,
    S_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    tl.static_assert(not IS_VARLEN or B == 1, "packed varlen requires B == 1")
    i_s = tl.program_id(0).to(tl.int64)
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
        bos = 0

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m = (o_t[:, None] < T) & (o_s[None, :] < S)
    token = bos + o_t if IS_VARLEN else o_t

    s_batch_offset = i_b * s_batch_stride if B > 1 else 0
    b_s = tl.load(
        s
        + s_batch_offset
        + ptr_offset(
            (token[:, None], i_h, o_s[None, :]),
            S_STRIDES,
        ),
        mask=m,
        other=0.0,
    ).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
    if HAS_SCALE:
        b_o = b_o * scale
    o_batch_offset = i_b * o_batch_stride if B > 1 else 0
    tl.store(
        o
        + o_batch_offset
        + ptr_offset(
            (token[:, None], i_h, o_s[None, :]),
            O_STRIDES,
        ),
        b_o.to(o.dtype.element_ty),
        mask=m,
    )
