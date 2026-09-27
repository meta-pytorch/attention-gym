# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Grouped-head gradient reduction (GVA/GQA) for linear-attention backward.

With grouped heads a backward kernel that emits per-output-head gradients at
``HO`` heads leaves the true per-native-head gradient as the sum over each
head's group of ``r = HO // H`` consecutive output heads:

    out[t, i, ...] = sum_{j < r} in[t, i * r + j, ...]

Flat 1-D grid over output words: each thread owns one 4-byte word (a packed
f16x2/bf16x2 pair, or one fp32 element), gathers it from all ``r`` group
heads (coalesced, strided by ``inner_words``), accumulates in fp32, and
stores one word back.  Head count and group size are runtime values; the
group loop runs over all ``r`` heads with a select on the first so the
compiler's runtime unroll (8/4/2/1 blocks) issues a group's loads together.
Serves the f16/bf16 ``[total, HO, D]`` tensor grads (dQ/dK for GVA, dK/dV for
GQA) and the fp32 ``[total, HO]`` Gate/Beta grads.
"""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

from .._compat import current_device
from ..tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from ..tile_dsl.pointwise import f16x2_to_f32, fp32_to_fp16
from .host import get_dtype

USE_PDL = True

BLOCK = 256


@cute.kernel
def frost_head_reduce(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    total_words: cutlass.Int64,
    out_row_words: cutlass.Int64,
    out_head_words: cutlass.Int64,
    h_count: cute.FastDivmodDivisorV2,
    r: cutlass.Int32,
    inner_words: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    gw = cutlass.Int64(cutlass.Int32(bidx)) * cutlass.Int64(BLOCK) + cutlass.Int64(
        cutlass.Int32(tidx)
    )
    if gw < total_words:
        seg = gw // cutlass.Int64(inner_words)
        w_off = gw - seg * cutlass.Int64(inner_words)
        base = seg * (cutlass.Int64(r) * cutlass.Int64(inner_words)) + w_off
        t_idx, h_idx = divmod(seg.to(cutlass.Int32), h_count)
        out_off = (
            cutlass.Int64(t_idx) * out_row_words + cutlass.Int64(h_idx) * out_head_words + w_off
        )
        if cutlass.const_expr(io_dtype == cutlass.Float32):
            in_p = cute.recast_ptr(mIn.iterator, dtype=cutlass.Float32)
            out_p = cute.recast_ptr(mOut.iterator, dtype=cutlass.Float32)
            acc = cutlass.Float32(0.0)
            for i in cutlass.range(r):
                v = (in_p + (base + cutlass.Int64(i) * cutlass.Int64(inner_words))).load()
                acc = v if i == 0 else acc + v
            (out_p + out_off).store(acc)
        else:
            in_p = cute.recast_ptr(mIn.iterator, dtype=cutlass.Int32)
            out_p = cute.recast_ptr(mOut.iterator, dtype=cutlass.Int32)
            acc_lo = cutlass.Float32(0.0)
            acc_hi = cutlass.Float32(0.0)
            for i in cutlass.range(r):
                lo, hi = f16x2_to_f32(
                    (in_p + (base + cutlass.Int64(i) * cutlass.Int64(inner_words))).load(),
                    dtype=io_dtype,
                )
                acc_lo = lo if i == 0 else acc_lo + lo
                acc_hi = hi if i == 0 else acc_hi + hi
            (out_p + out_off).store(fp32_to_fp16(acc_lo, acc_hi, dtype=io_dtype))
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def launch(
    mIn: cute.Tensor,
    mOut: cute.Tensor,
    total_words: cutlass.Int64,
    out_row_words: cutlass.Int64,
    out_head_words: cutlass.Int64,
    grid_x: cutlass.Int32,
    h_count: cutlass.Int32,
    r: cutlass.Int32,
    inner_words: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    frost_head_reduce(
        mIn,
        mOut,
        total_words,
        out_row_words,
        out_head_words,
        cute.FastDivmodDivisorV2(h_count),
        r,
        inner_words,
        io_dtype,
    ).launch(
        grid=(grid_x, 1, 1),
        block=(BLOCK, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
    )


compiled_cache = {}


def head_group_reduce(src, dst, *, stream) -> None:
    """Reduce ``src (total, HO, D)`` into ``dst (total, H, D)``, or the
    rank-2 ``(total, HO)`` into ``(total, H)``, by summing each group of
    ``r = HO // H`` consecutive heads (fp32 accumulation).

    ``src`` is contiguous (kernel-internal wide buffer); ``dst`` needs a
    stride-1 innermost dim with free outer strides (f16/bf16 outer strides
    must be even, word-pair stores). Same-dtype (f16/bf16, or fp32),
    DLPack-compatible CUDA tensors; the f16/bf16 inner extent ``D`` must be
    even.  Compile-cache-and-replay per ``(dtype, rank, D)``; head counts are runtime."""
    if len(src.shape) == 2:
        total, HO = src.shape
        D = 1
        H = dst.shape[1]
    else:
        total, HO, D = src.shape
        H = dst.shape[1]
    io_dtype = get_dtype(src.dtype)
    is_fp32 = io_dtype == cutlass.Float32
    r = HO // H
    inner_words = D if is_fp32 else D // 2
    total_words = total * H * inner_words
    grid_x = -(-total_words // BLOCK)
    cu_stream = cuda.CUstream(int(stream))

    dst_strides = tuple(dst.stride())
    out_row_words = dst_strides[0] if is_fp32 else dst_strides[0] // 2
    out_head_words = (
        (dst_strides[1] if is_fp32 else dst_strides[1] // 2) if len(dst.shape) == 3 else 1
    )

    key = (str(src.dtype).split(".")[-1], len(src.shape), D, current_device())
    if key not in compiled_cache:
        src_c = from_dlpack(src, assumed_align=4).mark_layout_dynamic(
            leading_dim=len(src.shape) - 1
        )
        compiled_cache[key] = cute.compile(
            launch,
            src_c,
            from_dlpack(dst, assumed_align=4).mark_layout_dynamic(leading_dim=len(dst.shape) - 1),
            cutlass.Int64(total_words),
            cutlass.Int64(out_row_words),
            cutlass.Int64(out_head_words),
            cutlass.Int32(grid_x),
            cutlass.Int32(H),
            cutlass.Int32(r),
            inner_words,
            io_dtype,
            cu_stream,
            options="--enable-tvm-ffi",
        )
    compiled_cache[key](
        src, dst, total_words, out_row_words, out_head_words, grid_x, H, r, cu_stream
    )


frost_head_reduce.set_name_prefix("cudnn", remove_cutlass_symbol=False)
