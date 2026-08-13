# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
# NOTE: do NOT use `from __future__ import annotations` — cute.struct
# requires eager-evaluated annotations.


from collections.abc import Iterable
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import Boolean, Float32, Int32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cutlass_dsl import Constexpr, T, dsl_user_op

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache, run_tunable
from attn_gym._backends.cute.target import detect_compile_target, get_compile_target
from attn_gym.linear.kda.utils import ChunkMetadata

BT = 64  # chunk_size
SUBCHUNKS = 4
BC = BT // SUBCHUNKS
KEY_DIM = 128  # full head_dim
KEY_DIM_PER_CTA = 32  # head_dim per K phase
K_PHASES = KEY_DIM // KEY_DIM_PER_CTA  # four 32-wide head-dim phases
KC_TOTAL = K_PHASES * SUBCHUNKS  # work items per (chunk, head): 16
# The gate algebra accumulates d/d(natural exponent); this kernel owns the last dg
# write, so it converts the complete gradient to d/d(log2 gate).
LN2 = 0.6931471805599453
# q and k may arrive as unbound QKV views: only their innermost dimension is
# contiguous, and the 16-byte cp.async stages need every outer stride and the
# base pointer to be 16-byte (8 bf16 element) aligned.
_MIN_ALIGN_BYTES = 16
_MIN_ALIGN_ELEMENTS_BF16 = _MIN_ALIGN_BYTES // 2

# Kernel phase map:
# 1. Host wrapper validates tensors and prepares caller-owned varlen metadata.
# 2. Launcher compiles the HMMA-grid CuTe kernel for the tensor signature.
# 3. Kernel maps the grid to (k_phase, subchunk, chunk_block, head_idx).
# 4. Kernel stages Q/K/G, runs the HMMA loops, then writes dq/dk/db/dg.


@dsl_user_op
def _mma_tf32_m16n8k8(
    a0,
    a1,
    a2,
    a3,
    b0,
    b1,
    c0,
    c1,
    c2,
    c3,
    *,
    loc=None,
    ip=None,
):
    """Warp-level TF32 MMA matching Triton's dot lowering."""
    a0b = llvm.bitcast(T.i32(), a0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a1b = llvm.bitcast(T.i32(), a1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a2b = llvm.bitcast(T.i32(), a2.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    a3b = llvm.bitcast(T.i32(), a3.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b0b = llvm.bitcast(T.i32(), b0.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b1b = llvm.bitcast(T.i32(), b1.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32, f32, f32)>"),
        [
            a0b,
            a1b,
            a2b,
            a3b,
            b0b,
            b1b,
            c0.ir_value(loc=loc, ip=ip),
            c1.ir_value(loc=loc, ip=ip),
            c2.ir_value(loc=loc, ip=ip),
            c3.ir_value(loc=loc, ip=ip),
        ],
        """{
            mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
                {$0, $1, $2, $3},
                {$4, $5, $6, $7},
                {$8, $9},
                {$10, $11, $12, $13};
        }""",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    d0 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    d1 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    d2 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    d3 = cutlass.Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))
    return d0, d1, d2, d3


@dsl_user_op
def _cp_async_cg_g2s_16b(
    gmem_ptr: cute.Pointer,
    smem_ptr: cute.Pointer,
    src_bytes: Int32,
    *,
    loc=None,
    ip=None,
):
    """Issue one 16-byte cp.async from global to shared memory."""
    gmem_addr = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_addr,
            gmem_addr,
            src_bytes.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 0x10, $2;",
        "r,l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldmatrix_x4_b16(smem_ptr: cute.Pointer, *, loc=None, ip=None):
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [smem_addr],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ldmatrix_x4_trans_b16(smem_ptr: cute.Pointer, *, loc=None, ip=None):
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [smem_addr],
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _stmatrix_x4_b16_f32(
    smem_ptr: cute.Pointer,
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_addr,
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
        ],
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_shared_u32x4(smem_ptr: cute.Pointer, *, loc=None, ip=None):
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [smem_addr],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_shared_u32(smem_ptr: cute.Pointer, *, loc=None, ip=None):
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    result = llvm.inline_asm(
        T.i32(),
        [smem_addr],
        "ld.shared.b32 $0, [$1];",
        "=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def _bitcast_i32_to_f32(x: Int32, *, loc=None, ip=None):
    return cutlass.Float32(llvm.bitcast(T.f32(), x.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def _bf16x2_to_f32_pair(x: Int32, *, loc=None, ip=None):
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [x.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .u32 lo, hi, lo_f32, hi_f32;\n\t"
        "and.b32 lo, $2, 0x0000ffff;\n\t"
        "shr.u32 hi, $2, 16;\n\t"
        "shl.b32 lo_f32, lo, 16;\n\t"
        "shl.b32 hi_f32, hi, 16;\n\t"
        "mov.b32 $0, lo_f32;\n\t"
        "mov.b32 $1, hi_f32;\n\t"
        "}\n",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_global_f32_b32_pred(
    gmem_ptr: cute.Pointer,
    pred: Boolean,
    *,
    loc=None,
    ip=None,
):
    ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    bits = llvm.inline_asm(
        T.i32(),
        [
            ptr_i64,
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "mov.b32 $0, 0;\n\t"
        "setp.ne.b32 p, $2, 0;\n\t"
        "@p ld.global.b32 $0, [$1];\n\t"
        "}\n",
        "=r,l,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(llvm.bitcast(T.f32(), bits, loc=loc, ip=ip))


@dsl_user_op
def _ld_global_f32x2_pred(
    gmem_ptr: cute.Pointer,
    pred: Boolean,
    *,
    loc=None,
    ip=None,
):
    ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [
            ptr_i64,
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "mov.b32 $0, 0;\n\t"
        "mov.b32 $1, 0;\n\t"
        "setp.ne.b32 p, $3, 0;\n\t"
        "@p ld.global.v2.b32 {$0, $1}, [$2];\n\t"
        "}\n",
        "=r,=r,l,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return (
        cutlass.Float32(llvm.bitcast(T.f32(), v0, loc=loc, ip=ip)),
        cutlass.Float32(llvm.bitcast(T.f32(), v1, loc=loc, ip=ip)),
    )


@dsl_user_op
def _ld_global_f32x4_lo2_pred(
    gmem_ptr: cute.Pointer,
    pred: Boolean,
    *,
    loc=None,
    ip=None,
):
    ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [
            ptr_i64,
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 unused0;\n\t"
        ".reg .b32 unused1;\n\t"
        "mov.b32 $0, 0;\n\t"
        "mov.b32 $1, 0;\n\t"
        "setp.ne.b32 p, $3, 0;\n\t"
        "@p ld.global.v4.b32 {$0, $1, unused0, unused1}, [$2];\n\t"
        "}\n",
        "=r,=r,l,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return (
        cutlass.Float32(llvm.bitcast(T.f32(), v0, loc=loc, ip=ip)),
        cutlass.Float32(llvm.bitcast(T.f32(), v1, loc=loc, ip=ip)),
    )


@dsl_user_op
def _add_f32x2(
    a0: Float32,
    a1: Float32,
    b0: Float32,
    b1: Float32,
    *,
    loc=None,
    ip=None,
):
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [
            Float32(a0).ir_value(loc=loc, ip=ip),
            Float32(a1).ir_value(loc=loc, ip=ip),
            Float32(b0).ir_value(loc=loc, ip=ip),
            Float32(b1).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b64 lhs, rhs, out;\n\t"
        "mov.b64 lhs, {$2, $3};\n\t"
        "mov.b64 rhs, {$4, $5};\n\t"
        "add.f32x2 out, lhs, rhs;\n\t"
        "mov.b64 {$0, $1}, out;\n\t"
        "}\n",
        "=f,=f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        cutlass.Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _cvt_bf16x2_f32(
    a: Float32,
    b: Float32,
    *,
    loc=None,
    ip=None,
) -> Int32:
    packed = llvm.inline_asm(
        T.i32(),
        [
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        ],
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        "=r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(packed)


@dsl_user_op
def _st_shared_f32x4_v2_b64(
    smem_ptr: cute.Pointer,
    a: Float32,
    b: Float32,
    c: Float32,
    d: Float32,
    *,
    loc=None,
    ip=None,
) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    a_bits = llvm.bitcast(T.i32(), Float32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b_bits = llvm.bitcast(T.i32(), Float32(b).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    c_bits = llvm.bitcast(T.i32(), Float32(c).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    d_bits = llvm.bitcast(T.i32(), Float32(d).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            smem_addr,
            a_bits,
            b_bits,
            c_bits,
            d_bits,
        ],
        "{\n\t"
        ".reg .b64 p0;\n\t"
        ".reg .b64 p1;\n\t"
        "mov.b64 p0, {$1, $2};\n\t"
        "mov.b64 p1, {$3, $4};\n\t"
        "st.shared.v2.b64 [$0], {p0, p1};\n\t"
        "}\n",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_u32x4(
    smem_ptr: cute.Pointer,
    a: Int32,
    b: Int32,
    c: Int32,
    d: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    smem_addr = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            smem_addr,
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u32x4_pred(
    gmem_ptr: cute.Pointer,
    a: Int32,
    b: Int32,
    c: Int32,
    d: Int32,
    pred: Boolean,
    *,
    loc=None,
    ip=None,
) -> None:
    ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    llvm.inline_asm(
        None,
        [
            ptr_i64,
            a.ir_value(loc=loc, ip=ip),
            b.ir_value(loc=loc, ip=ip),
            c.ir_value(loc=loc, ip=ip),
            d.ir_value(loc=loc, ip=ip),
            Int32(pred).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, $5, 0;\n\t"
        "@p st.global.v4.b32 [$0], {$1, $2, $3, $4};\n\t"
        "}\n",
        "l,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bar_warp_sync(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "bar.warp.sync -1;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _st_global_bf16_ldmatrix_epilogue_16x32(
    sEpi_tile,
    gmem_ptr,
    top0: Int32,
    top1: Int32,
    top2: Int32,
    top3: Int32,
    bottom0: Int32,
    bottom1: Int32,
    bottom2: Int32,
    bottom3: Int32,
    tidx: Int32,
    *,
    row_base,
    valid,
    row_stride,
):
    lane = tidx & 31
    r11 = lane & 3
    r14 = lane & 28
    r15 = lane & 24
    r20 = (lane & 7) << 4
    store_bytes = ((r11 << 5) | (r15 << 4)) ^ (r14 << 2)
    load_bytes = (((lane << 6) & 384) | r20) ^ (r15 << 2)
    out_row0 = (lane >> 2) & 7
    out_col = (lane & 3) * 8
    row_valid0 = (row_base + out_row0) < valid
    row_valid1 = (row_base + out_row0 + 8) < valid

    _bar_warp_sync()
    _st_shared_u32x4(sEpi_tile.iterator + store_bytes // 2, top0, top1, top2, top3)
    _bar_warp_sync()
    v0, v1, v2, v3 = _ldmatrix_x4_b16(sEpi_tile.iterator + load_bytes // 2)
    _bar_warp_sync()
    _st_shared_u32x4(
        sEpi_tile.iterator + store_bytes // 2,
        bottom0,
        bottom1,
        bottom2,
        bottom3,
    )
    _bar_warp_sync()
    v4, v5, v6, v7 = _ldmatrix_x4_b16(sEpi_tile.iterator + load_bytes // 2)
    _st_global_u32x4_pred(
        gmem_ptr + out_row0 * row_stride + out_col,
        v0,
        v1,
        v2,
        v3,
        Boolean(row_valid0),
    )
    _st_global_u32x4_pred(
        gmem_ptr + (out_row0 + 8) * row_stride + out_col,
        v4,
        v5,
        v6,
        v7,
        Boolean(row_valid1),
    )
    _bar_warp_sync()


@cute.jit
def _st_shared_dg_triton_swizzle(
    sEpi_tile,
    group: Constexpr,
    top0: Float32,
    top1: Float32,
    bottom0: Float32,
    bottom1: Float32,
    tidx: Int32,
):
    lane = tidx & 31
    lane_odd = lane & 1
    lane_bit1 = lane & 2
    lane_bit2 = (lane >> 2) & 1
    lane_high = lane & 24
    row_half = Int32(0)
    if lane_odd != 0:
        row_half = Int32(1088)
    col_pair = lane_bit1 << 3
    quad_half = Int32(0)
    if lane_bit2 != 0:
        quad_half = Int32(544)
    high_half = lane_high << 4
    store_bytes = (row_half | col_pair | quad_half | high_half) ^ Int32(group * 32)
    _st_shared_f32x4_v2_b64(
        sEpi_tile.iterator + store_bytes // 2,
        top0,
        top1,
        bottom0,
        bottom1,
    )


@cute.jit
def _st_global_f32_triton_shared_epilogue_16x32(
    sEpi_tile,
    gmem_ptr,
    tidx: Int32,
    *,
    row_base,
    valid,
    row_stride,
):
    _bar_warp_sync()
    lane = tidx & 31
    low3 = lane & 7
    lane_bit3 = lane & 8
    lane_bit4 = lane & 16
    row_xor = Int32(0)
    if lane_bit3 != 0:
        row_xor = Int32(544)
    load_base = (row_xor ^ (low3 << 4)) | (lane_bit4 << 3)
    v0, v1, v2, v3 = _ld_shared_u32x4(sEpi_tile.iterator + load_base // 2)
    v4, v5, v6, v7 = _ld_shared_u32x4(sEpi_tile.iterator + (load_base + 256) // 2)
    v8, v9, v10, v11 = _ld_shared_u32x4(sEpi_tile.iterator + ((load_base ^ Int32(64)) + 1024) // 2)
    v12, v13, v14, v15 = _ld_shared_u32x4(
        sEpi_tile.iterator + ((load_base ^ Int32(64)) + 1280) // 2
    )
    out_row0 = (lane >> 3) & 3
    out_col = low3 * 4
    row_valid0 = (row_base + out_row0) < valid
    row_valid1 = (row_base + out_row0 + 4) < valid
    row_valid2 = (row_base + out_row0 + 8) < valid
    row_valid3 = (row_base + out_row0 + 12) < valid
    _st_global_u32x4_pred(
        gmem_ptr + out_row0 * row_stride + out_col,
        v0,
        v1,
        v8,
        v9,
        Boolean(row_valid0),
    )
    _st_global_u32x4_pred(
        gmem_ptr + (out_row0 + 4) * row_stride + out_col,
        v4,
        v5,
        v12,
        v13,
        Boolean(row_valid1),
    )
    _st_global_u32x4_pred(
        gmem_ptr + (out_row0 + 8) * row_stride + out_col,
        v2,
        v3,
        v10,
        v11,
        Boolean(row_valid2),
    )
    _st_global_u32x4_pred(
        gmem_ptr + (out_row0 + 12) * row_stride + out_col,
        v6,
        v7,
        v14,
        v15,
        Boolean(row_valid3),
    )
    _bar_warp_sync()


@dsl_user_op
def _st_global_f32_b32_pred(
    gmem_ptr: cute.Pointer,
    a: Float32,
    pred: Boolean,
    *,
    loc=None,
    ip=None,
) -> None:
    ptr_i64 = gmem_ptr.toint(loc=loc, ip=ip).ir_value()
    a_bits = llvm.bitcast(T.i32(), Float32(a).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [ptr_i64, a_bits, Int32(pred).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $2, 0;\n\t@p st.global.b32 [$0], $1;\n\t}\n",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _hmma_load_b_groups_smem_ldmatrix(sB, row_base, k_outer, tidx):
    lane8 = tidx & 7
    group8 = tidx // 8
    smem_ptr = sB.iterator + (row_base + k_outer * 8 + lane8) * KEY_DIM_PER_CTA + group8 * 8
    r0, r1, r2, r3 = _ldmatrix_x4_trans_b16(smem_ptr)
    b00, b01 = _bf16x2_to_f32_pair(r0)
    b10, b11 = _bf16x2_to_f32_pair(r1)
    b20, b21 = _bf16x2_to_f32_pair(r2)
    b30, b31 = _bf16x2_to_f32_pair(r3)
    return b00, b01, b10, b11, b20, b21, b30, b31


@cute.jit
def _hmma_stmatrix_source_from_lane_values(b0, b1, tidx, source_reg: Constexpr):
    target_l8 = ((tidx & 6) // 2) + (Int32(4) if (tidx & 16) != 0 else Int32(0))
    src_lane = source_reg * 8 + target_l8
    src_b0 = cute.arch.shuffle_sync(b0, src_lane)
    src_b1 = cute.arch.shuffle_sync(b1, src_lane)
    return src_b1 if (tidx & 8) != 0 else src_b0


@cute.jit
def _hmma_stage_b_group_stmatrix(sB, b0, b1, tidx):
    store_bytes = ((tidx << 4) & 368) ^ (Int32(160) if (tidx & 8) != 0 else Int32(0))
    _stmatrix_x4_b16_f32(
        sB.iterator + (store_bytes // 2),
        _hmma_stmatrix_source_from_lane_values(b0, b1, tidx, 0),
        _hmma_stmatrix_source_from_lane_values(b0, b1, tidx, 1),
        _hmma_stmatrix_source_from_lane_values(b0, b1, tidx, 2),
        _hmma_stmatrix_source_from_lane_values(b0, b1, tidx, 3),
    )


@cute.jit
def _hmma_load_b_group_stmatrix(sB, tidx):
    load_bytes = ((tidx << 4) & 320) | ((tidx & 3) * 8)
    load_bytes = load_bytes | (Int32(160) if (tidx & 8) != 0 else Int32(0))
    packed0 = _ld_shared_u32(sB.iterator + (load_bytes // 2))
    packed1 = _ld_shared_u32(sB.iterator + ((load_bytes ^ 32) // 2))
    return _bitcast_i32_to_f32(packed0), _bitcast_i32_to_f32(packed1)


@cute.jit
def _hmma_load_da(mdA, row_start, head_idx, row, col, valid):
    return _ld_global_f32_b32_pred(
        mdA.iterator
        + (row_start + row) * mdA.layout.stride[1]
        + head_idx * mdA.layout.stride[2]
        + col,
        Boolean((row < valid) & (col < valid)),
    )


@cute.jit
def _hmma_load_da_pred(mdA, row_start, head_idx, row, col, valid, pred):
    return _ld_global_f32_b32_pred(
        mdA.iterator
        + (row_start + row) * mdA.layout.stride[1]
        + head_idx * mdA.layout.stride[2]
        + col,
        Boolean(pred & (row < valid) & (col < valid)),
    )


@cute.jit
def _hmma_load_da_pair(mdA, row_start, head_idx, row, col, valid, row_stride, head_stride):
    v0, v1 = _ld_global_f32x4_lo2_pred(
        mdA.iterator + (row_start + row) * row_stride + head_idx * head_stride + col,
        Boolean((row < valid) & (col < valid)),
    )
    v1 = v1 if (col + 1) < valid else Float32(0.0)
    return v0, v1


@cute.jit
def _hmma_load_da_pair_pred(
    mdA,
    row_start,
    head_idx,
    row,
    col,
    valid,
    row_stride,
    head_stride,
    pred0,
    pred1,
):
    v0, v1 = _ld_global_f32x4_lo2_pred(
        mdA.iterator + (row_start + row) * row_stride + head_idx * head_stride + col,
        Boolean(pred0 & (row < valid) & (col < valid)),
    )
    v1 = v1 if pred1 & ((col + 1) < valid) else Float32(0.0)
    return v0, v1


@cute.jit
def _copy_qkg_smem_cpasync(
    sQ,
    sK,
    sG,
    mQ,
    mK,
    mG,
    row_start,
    head_idx,
    k_start,
    valid,
    tidx,
):
    q_row_stride = mQ.layout.stride[1]
    q_head_stride = mQ.layout.stride[2]
    k_row_stride = mK.layout.stride[1]
    k_head_stride = mK.layout.stride[2]
    g_row_stride = mG.layout.stride[1]
    g_head_stride = mG.layout.stride[2]

    for i in cutlass.range_constexpr(8):
        elem = tidx * 8 + i * 32 * 8
        row = elem // KEY_DIM_PER_CTA
        col = elem - row * KEY_DIM_PER_CTA
        src_bytes = Int32(16) if row < valid else Int32(0)
        _cp_async_cg_g2s_16b(
            mQ.iterator
            + (row_start + row) * q_row_stride
            + head_idx * q_head_stride
            + k_start
            + col,
            sQ.iterator + elem,
            src_bytes,
        )
        _cp_async_cg_g2s_16b(
            mK.iterator
            + (row_start + row) * k_row_stride
            + head_idx * k_head_stride
            + k_start
            + col,
            sK.iterator + elem,
            src_bytes,
        )

    for i in cutlass.range_constexpr(16):
        elem = tidx * 4 + i * 32 * 4
        row = elem // KEY_DIM_PER_CTA
        col = elem - row * KEY_DIM_PER_CTA
        src_bytes = Int32(16) if row < valid else Int32(0)
        _cp_async_cg_g2s_16b(
            mG.iterator
            + (row_start + row) * g_row_stride
            + head_idx * g_head_stride
            + k_start
            + col,
            sG.iterator + elem,
            src_bytes,
        )

    cute.arch.cp_async_commit_group()
    cute.arch.cp_async_wait_group(0)
    cute.arch.barrier()


@cute.kernel
def _chunk_kda_bwd_intra_hmma_grid_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mG: cute.Tensor,
    mBeta: cute.Tensor,
    mdAqk: cute.Tensor,
    mdAkk: cute.Tensor,
    mDq: cute.Tensor,
    mDk: cute.Tensor,
    mDb2: cute.Tensor,
    mDg: cute.Tensor,
    mDq2: cute.Tensor,
    mDk2: cute.Tensor,
    mDg2: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    mNumChunks: cute.Tensor,
    use_i32_metadata: Constexpr,
    grid_chunks: Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    # Grid is (KC_TOTAL, grid_chunks, H), mirroring Triton's for-loop variant.
    # kc and head are real grid axes; only the chunk axis is capped at
    # grid_chunks and walked by each CTA. These per-(kc, head) coordinates are
    # loop-invariant, so they are computed once outside the chunk loop.
    kc_idx, chunk_start, head_idx = cute.arch.block_idx()
    gid = tidx // 4
    tid = tidx % 4
    k_phase = kc_idx // SUBCHUNKS
    subchunk_idx = kc_idx - k_phase * SUBCHUNKS
    k_start = k_phase * KEY_DIM_PER_CTA
    row_base = subchunk_idx * BC
    row0 = row_base + gid
    row1 = row0 + 8

    # num_chunks is a runtime device scalar, so the chunk axis is a fixed cap
    # (grid_chunks) and each CTA strides chunk_start, chunk_start + grid_chunks,
    # ... while < num_chunks. This replaces the old grid that was sized by the
    # padded chunk_indices.shape[0] (often ~16x the live chunk count). The
    # numerator stays >= 0 because chunk_start < grid_chunks.
    num_chunks_runtime = mNumChunks[0].to(Int32)
    iters = (num_chunks_runtime - chunk_start + grid_chunks - 1) // grid_chunks

    # SMEM is allocated once and reused across the chunk-loop iterations.
    smem = cutlass.utils.SmemAllocator()
    sQ_tile = smem.allocate_tensor(
        element_type=cutlass.BFloat16,
        layout=cute.make_layout((BT * KEY_DIM_PER_CTA,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )
    sK_tile = smem.allocate_tensor(
        element_type=cutlass.BFloat16,
        layout=cute.make_layout((BT * KEY_DIM_PER_CTA,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )
    sG_tile = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=cute.make_layout((BT * KEY_DIM_PER_CTA,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )
    sBeta_tile = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=cute.make_layout((BT,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )
    sEpi_tile = smem.allocate_tensor(
        element_type=cutlass.BFloat16,
        layout=cute.make_layout((2048,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )

    for chunk_iter in cutlass.range(iters, unroll=1):
        chunk_block = chunk_start + chunk_iter * grid_chunks
        seq_idx = Int32(0)
        chunk_idx = Int32(0)
        bos = Int32(0)
        eos = Int32(0)
        if use_i32_metadata:
            seq_idx = mChunkIndices[chunk_block, 0].to(Int32)
            chunk_idx = mChunkIndices[chunk_block, 1].to(Int32)
            bos = mCuSeqlens[seq_idx].to(Int32)
            eos = mCuSeqlens[seq_idx + 1].to(Int32)
        else:
            seq_idx = mChunkIndices[chunk_block, 0].to(Int32)
            chunk_idx = mChunkIndices[chunk_block, 1].to(Int32)
            bos = mCuSeqlens[seq_idx].to(Int32)
            eos = mCuSeqlens[seq_idx + 1].to(Int32)
        row_start = bos + chunk_idx * BT
        valid = cutlass.min(eos - row_start, Int32(BT))
        # Match Triton's default safe-gate normalization reference
        # (causal_gate_normref=False): use the middle row of this BC subchunk,
        # clamped for partial chunks. This keeps local diagonal dA scaling
        # finite for dense causal dA without relying on a near-diagonal dA
        # property.
        ref_prev = cutlass.min(row_base, valid - 1)
        ref_qk = cutlass.min(row_base + BC // 2, valid - 1)
        ref_future = cutlass.min(row_base + BC, valid) - 1
        daqk_row_stride = mdAqk.layout.stride[1]
        daqk_head_stride = mdAqk.layout.stride[2]
        dakk_row_stride = mdAkk.layout.stride[1]
        dakk_head_stride = mdAkk.layout.stride[2]
        dq_row_stride = mDq.layout.stride[1]
        dq_head_stride = mDq.layout.stride[2]
        dq2_row_stride = mDq2.layout.stride[1]
        dq2_head_stride = mDq2.layout.stride[2]
        dk_row_stride = mDk.layout.stride[1]
        dk_head_stride = mDk.layout.stride[2]
        dk2_row_stride = mDk2.layout.stride[1]
        dk2_head_stride = mDk2.layout.stride[2]
        dg_row_stride = mDg.layout.stride[1]
        dg_head_stride = mDg.layout.stride[2]
        dg2_row_stride = mDg2.layout.stride[1]
        dg2_head_stride = mDg2.layout.stride[2]
        db2_k_stride = mDb2.layout.stride[0]
        db2_row_stride = mDb2.layout.stride[1]
        db2_head_stride = mDb2.layout.stride[2]

        z = Float32(0.0)
        beta_row0 = tidx
        beta_row1 = tidx + 32
        beta_val0 = z
        beta_val1 = z
        if (beta_row0 >= row_base) & (beta_row0 < valid):
            beta_val0 = mBeta[0, row_start + beta_row0, head_idx]
        if (beta_row1 >= row_base) & (beta_row1 < valid):
            beta_val1 = mBeta[0, row_start + beta_row1, head_idx]
        sBeta_tile[beta_row0] = beta_val0
        sBeta_tile[beta_row1] = beta_val1
        _copy_qkg_smem_cpasync(
            sQ_tile,
            sK_tile,
            sG_tile,
            mQ,
            mK,
            mG,
            row_start,
            head_idx,
            k_start,
            valid,
            tidx,
        )
        gref_qk0 = sG_tile[ref_qk * KEY_DIM_PER_CTA + gid]
        gref_qk1 = sG_tile[ref_qk * KEY_DIM_PER_CTA + 8 + gid]
        gref_qk2 = sG_tile[ref_qk * KEY_DIM_PER_CTA + 16 + gid]
        gref_qk3 = sG_tile[ref_qk * KEY_DIM_PER_CTA + 24 + gid]
        gref_prev0 = sG_tile[ref_prev * KEY_DIM_PER_CTA + gid]
        gref_prev1 = sG_tile[ref_prev * KEY_DIM_PER_CTA + 8 + gid]
        gref_prev2 = sG_tile[ref_prev * KEY_DIM_PER_CTA + 16 + gid]
        gref_prev3 = sG_tile[ref_prev * KEY_DIM_PER_CTA + 24 + gid]
        gref_future0 = sG_tile[ref_future * KEY_DIM_PER_CTA + gid]
        gref_future1 = sG_tile[ref_future * KEY_DIM_PER_CTA + 8 + gid]
        gref_future2 = sG_tile[ref_future * KEY_DIM_PER_CTA + 16 + gid]
        gref_future3 = sG_tile[ref_future * KEY_DIM_PER_CTA + 24 + gid]
        q00 = z
        q01 = z
        q02 = z
        q03 = z
        q10 = z
        q11 = z
        q12 = z
        q13 = z
        q20 = z
        q21 = z
        q22 = z
        q23 = z
        q30 = z
        q31 = z
        q32 = z
        q33 = z
        k00 = z
        k01 = z
        k02 = z
        k03 = z
        k10 = z
        k11 = z
        k12 = z
        k13 = z
        k20 = z
        k21 = z
        k22 = z
        k23 = z
        k30 = z
        k31 = z
        k32 = z
        k33 = z
        qd00 = z
        qd01 = z
        qd02 = z
        qd03 = z
        qd10 = z
        qd11 = z
        qd12 = z
        qd13 = z
        qd20 = z
        qd21 = z
        qd22 = z
        qd23 = z
        qd30 = z
        qd31 = z
        qd32 = z
        qd33 = z
        kd00 = z
        kd01 = z
        kd02 = z
        kd03 = z
        kd10 = z
        kd11 = z
        kd12 = z
        kd13 = z
        kd20 = z
        kd21 = z
        kd22 = z
        kd23 = z
        kd30 = z
        kd31 = z
        kd32 = z
        kd33 = z
        t00 = z
        t01 = z
        t02 = z
        t03 = z
        t10 = z
        t11 = z
        t12 = z
        t13 = z
        t20 = z
        t21 = z
        t22 = z
        t23 = z
        t30 = z
        t31 = z
        t32 = z
        t33 = z
        d00 = z
        d01 = z
        d02 = z
        d03 = z
        d10 = z
        d11 = z
        d12 = z
        d13 = z
        d20 = z
        d21 = z
        d22 = z
        d23 = z
        d30 = z
        d31 = z
        d32 = z
        d33 = z

        for j_block in cutlass.range(subchunk_idx):
            key_base = j_block * BC
            for k_outer in cutlass.range_constexpr(2):
                key0 = key_base + k_outer * 8 + 2 * tid
                key1 = key0 + 1
                aq0, aq2 = _hmma_load_da_pair(
                    mdAqk,
                    row_start,
                    head_idx,
                    row0,
                    key0,
                    valid,
                    daqk_row_stride,
                    daqk_head_stride,
                )
                ak0, ak2 = _hmma_load_da_pair(
                    mdAkk,
                    row_start,
                    head_idx,
                    row0,
                    key0,
                    valid,
                    dakk_row_stride,
                    dakk_head_stride,
                )
                aq1, aq3 = _hmma_load_da_pair(
                    mdAqk,
                    row_start,
                    head_idx,
                    row1,
                    key0,
                    valid,
                    daqk_row_stride,
                    daqk_head_stride,
                )
                ak1, ak3 = _hmma_load_da_pair(
                    mdAkk,
                    row_start,
                    head_idx,
                    row1,
                    key0,
                    valid,
                    dakk_row_stride,
                    dakk_head_stride,
                )
                kb00, kb01, kb10, kb11, kb20, kb21, kb30, kb31 = _hmma_load_b_groups_smem_ldmatrix(
                    sK_tile, key_base, k_outer, tidx
                )

                for group in cutlass.range_constexpr(4):
                    col_a = group * 8
                    col_load = col_a + gid
                    gref = gref_prev0
                    if group == 1:
                        gref = gref_prev1
                    if group == 2:
                        gref = gref_prev2
                    if group == 3:
                        gref = gref_prev3
                    gk0 = sG_tile[key0 * KEY_DIM_PER_CTA + col_load]
                    gk1 = sG_tile[key1 * KEY_DIM_PER_CTA + col_load]
                    kb0 = kb00
                    kb1 = kb01
                    if group == 1:
                        kb0 = kb10
                        kb1 = kb11
                    if group == 2:
                        kb0 = kb20
                        kb1 = kb21
                    if group == 3:
                        kb0 = kb30
                        kb1 = kb31
                    b0 = kb0 * cute.math.exp2(gref - gk0, fastmath=True)
                    b1 = kb1 * cute.math.exp2(gref - gk1, fastmath=True)
                    b0 = b0 if key0 < valid else z
                    b1 = b1 if key1 < valid else z
                    _hmma_stage_b_group_stmatrix(sEpi_tile, b0, b1, tidx)
                    _bar_warp_sync()
                    b0, b1 = _hmma_load_b_group_stmatrix(sEpi_tile, tidx)
                    if group == 0:
                        q00, q01, q02, q03 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, q00, q01, q02, q03
                        )
                        k00, k01, k02, k03 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, b0, b1, k00, k01, k02, k03
                        )
                    if group == 1:
                        q10, q11, q12, q13 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, q10, q11, q12, q13
                        )
                        k10, k11, k12, k13 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, b0, b1, k10, k11, k12, k13
                        )
                    if group == 2:
                        q20, q21, q22, q23 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, q20, q21, q22, q23
                        )
                        k20, k21, k22, k23 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, b0, b1, k20, k21, k22, k23
                        )
                    if group == 3:
                        q30, q31, q32, q33 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, q30, q31, q32, q33
                        )
                        k30, k31, k32, k33 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, b0, b1, k30, k31, k32, k33
                        )

        key_base = row_base
        for k_outer in cutlass.range_constexpr(2):
            key0 = key_base + k_outer * 8 + 2 * tid
            key1 = key0 + 1
            keep00 = (k_outer * 8 + 2 * tid) <= gid
            keep01 = (k_outer * 8 + 2 * tid + 1) <= gid
            keep10 = (k_outer * 8 + 2 * tid) <= (gid + 8)
            keep11 = (k_outer * 8 + 2 * tid + 1) <= (gid + 8)
            aq0, aq2 = _hmma_load_da_pair_pred(
                mdAqk,
                row_start,
                head_idx,
                row0,
                key0,
                valid,
                daqk_row_stride,
                daqk_head_stride,
                keep00,
                keep01,
            )
            ak0, ak2 = _hmma_load_da_pair_pred(
                mdAkk,
                row_start,
                head_idx,
                row0,
                key0,
                valid,
                dakk_row_stride,
                dakk_head_stride,
                keep00,
                keep01,
            )
            aq1, aq3 = _hmma_load_da_pair_pred(
                mdAqk,
                row_start,
                head_idx,
                row1,
                key0,
                valid,
                daqk_row_stride,
                daqk_head_stride,
                keep10,
                keep11,
            )
            ak1, ak3 = _hmma_load_da_pair_pred(
                mdAkk,
                row_start,
                head_idx,
                row1,
                key0,
                valid,
                dakk_row_stride,
                dakk_head_stride,
                keep10,
                keep11,
            )
            kb00, kb01, kb10, kb11, kb20, kb21, kb30, kb31 = _hmma_load_b_groups_smem_ldmatrix(
                sK_tile, key_base, k_outer, tidx
            )

            for group in cutlass.range_constexpr(4):
                col_a = group * 8
                col_load = col_a + gid
                gref = gref_qk0
                if group == 1:
                    gref = gref_qk1
                if group == 2:
                    gref = gref_qk2
                if group == 3:
                    gref = gref_qk3
                gk0 = sG_tile[key0 * KEY_DIM_PER_CTA + col_load]
                gk1 = sG_tile[key1 * KEY_DIM_PER_CTA + col_load]
                kb0 = kb00
                kb1 = kb01
                if group == 1:
                    kb0 = kb10
                    kb1 = kb11
                if group == 2:
                    kb0 = kb20
                    kb1 = kb21
                if group == 3:
                    kb0 = kb30
                    kb1 = kb31
                b0 = kb0 * cute.math.exp2(gref - gk0, fastmath=True)
                b1 = kb1 * cute.math.exp2(gref - gk1, fastmath=True)
                b0 = b0 if key0 < valid else z
                b1 = b1 if key1 < valid else z
                _hmma_stage_b_group_stmatrix(sEpi_tile, b0, b1, tidx)
                _bar_warp_sync()
                b0, b1 = _hmma_load_b_group_stmatrix(sEpi_tile, tidx)
                if group == 0:
                    qd00, qd01, qd02, qd03 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, qd00, qd01, qd02, qd03
                    )
                    kd00, kd01, kd02, kd03 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, b0, b1, kd00, kd01, kd02, kd03
                    )
                if group == 1:
                    qd10, qd11, qd12, qd13 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, qd10, qd11, qd12, qd13
                    )
                    kd10, kd11, kd12, kd13 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, b0, b1, kd10, kd11, kd12, kd13
                    )
                if group == 2:
                    qd20, qd21, qd22, qd23 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, qd20, qd21, qd22, qd23
                    )
                    kd20, kd21, kd22, kd23 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, b0, b1, kd20, kd21, kd22, kd23
                    )
                if group == 3:
                    qd30, qd31, qd32, qd33 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, qd30, qd31, qd32, qd33
                    )
                    kd30, kd31, kd32, kd33 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, b0, b1, kd30, kd31, kd32, kd33
                    )

        for j_block in cutlass.range(subchunk_idx + 1, SUBCHUNKS):
            q_base = j_block * BC
            for k_outer in cutlass.range_constexpr(2):
                query0 = q_base + k_outer * 8 + 2 * tid
                query1 = query0 + 1
                aq0 = _hmma_load_da(mdAqk, row_start, head_idx, query0, row0, valid)
                aq1 = _hmma_load_da(mdAqk, row_start, head_idx, query0, row1, valid)
                aq2 = _hmma_load_da(mdAqk, row_start, head_idx, query1, row0, valid)
                aq3 = _hmma_load_da(mdAqk, row_start, head_idx, query1, row1, valid)
                ak0 = _hmma_load_da(mdAkk, row_start, head_idx, query0, row0, valid)
                ak1 = _hmma_load_da(mdAkk, row_start, head_idx, query0, row1, valid)
                ak2 = _hmma_load_da(mdAkk, row_start, head_idx, query1, row0, valid)
                ak3 = _hmma_load_da(mdAkk, row_start, head_idx, query1, row1, valid)
                bq0 = sBeta_tile[query0]
                bq1 = sBeta_tile[query1]
                qq00, qq01, qq10, qq11, qq20, qq21, qq30, qq31 = _hmma_load_b_groups_smem_ldmatrix(
                    sQ_tile, q_base, k_outer, tidx
                )
                kk00, kk01, kk10, kk11, kk20, kk21, kk30, kk31 = _hmma_load_b_groups_smem_ldmatrix(
                    sK_tile, q_base, k_outer, tidx
                )

                for group in cutlass.range_constexpr(4):
                    col_a = group * 8
                    col_load = col_a + gid
                    gref = gref_future0
                    if group == 1:
                        gref = gref_future1
                    if group == 2:
                        gref = gref_future2
                    if group == 3:
                        gref = gref_future3
                    gq0 = sG_tile[query0 * KEY_DIM_PER_CTA + col_load]
                    gq1 = sG_tile[query1 * KEY_DIM_PER_CTA + col_load]
                    qq0 = qq00
                    qq1 = qq01
                    kk0 = kk00
                    kk1 = kk01
                    if group == 1:
                        qq0 = qq10
                        qq1 = qq11
                        kk0 = kk10
                        kk1 = kk11
                    if group == 2:
                        qq0 = qq20
                        qq1 = qq21
                        kk0 = kk20
                        kk1 = kk21
                    if group == 3:
                        qq0 = qq30
                        qq1 = qq31
                        kk0 = kk30
                        kk1 = kk31
                    e0 = cute.math.exp2(gq0 - gref, fastmath=True)
                    e1 = cute.math.exp2(gq1 - gref, fastmath=True)
                    b0 = qq0 * e0
                    b1 = qq1 * e1
                    c0 = kk0 * bq0 * e0
                    c1 = kk1 * bq1 * e1
                    b0 = b0 if query0 < valid else z
                    b1 = b1 if query1 < valid else z
                    c0 = c0 if query0 < valid else z
                    c1 = c1 if query1 < valid else z
                    if group == 0:
                        t00, t01, t02, t03 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, t00, t01, t02, t03
                        )
                        t00, t01, t02, t03 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, c0, c1, t00, t01, t02, t03
                        )
                    if group == 1:
                        t10, t11, t12, t13 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, t10, t11, t12, t13
                        )
                        t10, t11, t12, t13 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, c0, c1, t10, t11, t12, t13
                        )
                    if group == 2:
                        t20, t21, t22, t23 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, t20, t21, t22, t23
                        )
                        t20, t21, t22, t23 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, c0, c1, t20, t21, t22, t23
                        )
                    if group == 3:
                        t30, t31, t32, t33 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, t30, t31, t32, t33
                        )
                        t30, t31, t32, t33 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, c0, c1, t30, t31, t32, t33
                        )

        for k_outer in cutlass.range_constexpr(2):
            query0 = row_base + k_outer * 8 + 2 * tid
            query1 = query0 + 1
            keep00 = gid <= (k_outer * 8 + 2 * tid)
            keep01 = gid <= (k_outer * 8 + 2 * tid + 1)
            keep10 = (gid + 8) <= (k_outer * 8 + 2 * tid)
            keep11 = (gid + 8) <= (k_outer * 8 + 2 * tid + 1)
            aq0 = _hmma_load_da_pred(mdAqk, row_start, head_idx, query0, row0, valid, keep00)
            aq1 = _hmma_load_da_pred(mdAqk, row_start, head_idx, query0, row1, valid, keep10)
            aq2 = _hmma_load_da_pred(mdAqk, row_start, head_idx, query1, row0, valid, keep01)
            aq3 = _hmma_load_da_pred(mdAqk, row_start, head_idx, query1, row1, valid, keep11)
            ak0 = _hmma_load_da_pred(mdAkk, row_start, head_idx, query0, row0, valid, keep00)
            ak1 = _hmma_load_da_pred(mdAkk, row_start, head_idx, query0, row1, valid, keep10)
            ak2 = _hmma_load_da_pred(mdAkk, row_start, head_idx, query1, row0, valid, keep01)
            ak3 = _hmma_load_da_pred(mdAkk, row_start, head_idx, query1, row1, valid, keep11)
            bq0 = sBeta_tile[query0]
            bq1 = sBeta_tile[query1]
            qq00, qq01, qq10, qq11, qq20, qq21, qq30, qq31 = _hmma_load_b_groups_smem_ldmatrix(
                sQ_tile, row_base, k_outer, tidx
            )
            kk00, kk01, kk10, kk11, kk20, kk21, kk30, kk31 = _hmma_load_b_groups_smem_ldmatrix(
                sK_tile, row_base, k_outer, tidx
            )

            for group in cutlass.range_constexpr(4):
                col_a = group * 8
                col_load = col_a + gid
                gref = gref_qk0
                if group == 1:
                    gref = gref_qk1
                if group == 2:
                    gref = gref_qk2
                if group == 3:
                    gref = gref_qk3
                gq0 = sG_tile[query0 * KEY_DIM_PER_CTA + col_load]
                gq1 = sG_tile[query1 * KEY_DIM_PER_CTA + col_load]
                qq0 = qq00
                qq1 = qq01
                kk0 = kk00
                kk1 = kk01
                if group == 1:
                    qq0 = qq10
                    qq1 = qq11
                    kk0 = kk10
                    kk1 = kk11
                if group == 2:
                    qq0 = qq20
                    qq1 = qq21
                    kk0 = kk20
                    kk1 = kk21
                if group == 3:
                    qq0 = qq30
                    qq1 = qq31
                    kk0 = kk30
                    kk1 = kk31
                e0 = cute.math.exp2(gq0 - gref, fastmath=True)
                e1 = cute.math.exp2(gq1 - gref, fastmath=True)
                b0 = qq0 * e0
                b1 = qq1 * e1
                c0 = kk0 * bq0 * e0
                c1 = kk1 * bq1 * e1
                b0 = b0 if query0 < valid else z
                b1 = b1 if query1 < valid else z
                c0 = c0 if query0 < valid else z
                c1 = c1 if query1 < valid else z
                if group == 0:
                    d00, d01, d02, d03 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, d00, d01, d02, d03
                    )
                    d00, d01, d02, d03 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, c0, c1, d00, d01, d02, d03
                    )
                if group == 1:
                    d10, d11, d12, d13 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, d10, d11, d12, d13
                    )
                    d10, d11, d12, d13 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, c0, c1, d10, d11, d12, d13
                    )
                if group == 2:
                    d20, d21, d22, d23 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, d20, d21, d22, d23
                    )
                    d20, d21, d22, d23 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, c0, c1, d20, d21, d22, d23
                    )
                if group == 3:
                    d30, d31, d32, d33 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, d30, d31, d32, d33
                    )
                    d30, d31, d32, d33 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, c0, c1, d30, d31, d32, d33
                    )

        db_acc0 = z
        db_acc1 = z
        dq_top0 = Int32(0)
        dq_top1 = Int32(0)
        dq_top2 = Int32(0)
        dq_top3 = Int32(0)
        dq_bottom0 = Int32(0)
        dq_bottom1 = Int32(0)
        dq_bottom2 = Int32(0)
        dq_bottom3 = Int32(0)
        dk_top0 = Int32(0)
        dk_top1 = Int32(0)
        dk_top2 = Int32(0)
        dk_top3 = Int32(0)
        dk_bottom0 = Int32(0)
        dk_bottom1 = Int32(0)
        dk_bottom2 = Int32(0)
        dk_bottom3 = Int32(0)
        dg_top00 = z
        dg_top01 = z
        dg_top10 = z
        dg_top11 = z
        dg_top20 = z
        dg_top21 = z
        dg_top30 = z
        dg_top31 = z
        dg_bottom00 = z
        dg_bottom01 = z
        dg_bottom10 = z
        dg_bottom11 = z
        dg_bottom20 = z
        dg_bottom21 = z
        dg_bottom30 = z
        dg_bottom31 = z
        for group in cutlass.range_constexpr(4):
            col_a = group * 8
            col0 = k_start + col_a + 2 * tid
            local_col0 = col_a + 2 * tid
            dq_word0 = Int32(0)
            dq_word1 = Int32(0)
            dk_word0 = Int32(0)
            dk_word1 = Int32(0)
            dg_top0 = z
            dg_top1 = z
            dg_bottom0 = z
            dg_bottom1 = z
            gref_prev_offset = ref_prev * KEY_DIM_PER_CTA + local_col0
            gref_qk_offset = ref_qk * KEY_DIM_PER_CTA + local_col0
            gref_t_offset = ref_future * KEY_DIM_PER_CTA + local_col0
            gref_prev0 = sG_tile[gref_prev_offset]
            gref_prev1 = sG_tile[gref_prev_offset + 1]
            gref_qk0 = sG_tile[gref_qk_offset]
            gref_qk1 = sG_tile[gref_qk_offset + 1]
            gref_t0 = sG_tile[gref_t_offset]
            gref_t1 = sG_tile[gref_t_offset + 1]
            qk_from_prev0 = cute.math.exp2(gref_prev0 - gref_qk0, fastmath=True)
            qk_from_prev1 = cute.math.exp2(gref_prev1 - gref_qk1, fastmath=True)
            tref0 = cute.math.exp2(gref_t0 - gref_qk0, fastmath=True)
            tref1 = cute.math.exp2(gref_t1 - gref_qk1, fastmath=True)

            for lane_row in cutlass.range_constexpr(2):
                row = row0 if lane_row == 0 else row1
                q_acc0 = q00 if lane_row == 0 else q02
                q_acc1 = q01 if lane_row == 0 else q03
                k_acc0 = k00 if lane_row == 0 else k02
                k_acc1 = k01 if lane_row == 0 else k03
                q_diag0 = qd00 if lane_row == 0 else qd02
                q_diag1 = qd01 if lane_row == 0 else qd03
                k_diag0 = kd00 if lane_row == 0 else kd02
                k_diag1 = kd01 if lane_row == 0 else kd03
                t_acc0 = t00 if lane_row == 0 else t02
                t_acc1 = t01 if lane_row == 0 else t03
                d_acc0 = d00 if lane_row == 0 else d02
                d_acc1 = d01 if lane_row == 0 else d03
                if group == 1:
                    q_acc0 = q10 if lane_row == 0 else q12
                    q_acc1 = q11 if lane_row == 0 else q13
                    k_acc0 = k10 if lane_row == 0 else k12
                    k_acc1 = k11 if lane_row == 0 else k13
                    q_diag0 = qd10 if lane_row == 0 else qd12
                    q_diag1 = qd11 if lane_row == 0 else qd13
                    k_diag0 = kd10 if lane_row == 0 else kd12
                    k_diag1 = kd11 if lane_row == 0 else kd13
                    t_acc0 = t10 if lane_row == 0 else t12
                    t_acc1 = t11 if lane_row == 0 else t13
                    d_acc0 = d10 if lane_row == 0 else d12
                    d_acc1 = d11 if lane_row == 0 else d13
                if group == 2:
                    q_acc0 = q20 if lane_row == 0 else q22
                    q_acc1 = q21 if lane_row == 0 else q23
                    k_acc0 = k20 if lane_row == 0 else k22
                    k_acc1 = k21 if lane_row == 0 else k23
                    q_diag0 = qd20 if lane_row == 0 else qd22
                    q_diag1 = qd21 if lane_row == 0 else qd23
                    k_diag0 = kd20 if lane_row == 0 else kd22
                    k_diag1 = kd21 if lane_row == 0 else kd23
                    t_acc0 = t20 if lane_row == 0 else t22
                    t_acc1 = t21 if lane_row == 0 else t23
                    d_acc0 = d20 if lane_row == 0 else d22
                    d_acc1 = d21 if lane_row == 0 else d23
                if group == 3:
                    q_acc0 = q30 if lane_row == 0 else q32
                    q_acc1 = q31 if lane_row == 0 else q33
                    k_acc0 = k30 if lane_row == 0 else k32
                    k_acc1 = k31 if lane_row == 0 else k33
                    q_diag0 = qd30 if lane_row == 0 else qd32
                    q_diag1 = qd31 if lane_row == 0 else qd33
                    k_diag0 = kd30 if lane_row == 0 else kd32
                    k_diag1 = kd31 if lane_row == 0 else kd33
                    t_acc0 = t30 if lane_row == 0 else t32
                    t_acc1 = t31 if lane_row == 0 else t33
                    d_acc0 = d30 if lane_row == 0 else d32
                    d_acc1 = d31 if lane_row == 0 else d33

                row_valid = Boolean(row < valid)
                out_row = row_start + row
                row_offset = row * KEY_DIM_PER_CTA + local_col0
                qval0 = sQ_tile[row_offset].to(Float32)
                qval1 = sQ_tile[row_offset + 1].to(Float32)
                kval0 = sK_tile[row_offset].to(Float32)
                kval1 = sK_tile[row_offset + 1].to(Float32)
                grow0 = sG_tile[row_offset]
                grow1 = sG_tile[row_offset + 1]
                beta_row = sBeta_tile[row]
                qscale_prev0 = cute.math.exp2(grow0 - gref_prev0, fastmath=True)
                qscale_prev1 = cute.math.exp2(grow1 - gref_prev1, fastmath=True)
                qscale_diag0 = qscale_prev0 * qk_from_prev0
                qscale_diag1 = qscale_prev1 * qk_from_prev1
                dscale0 = cute.math.exp2(gref_qk0 - grow0, fastmath=True)
                dscale1 = cute.math.exp2(gref_qk1 - grow1, fastmath=True)
                tscale0 = dscale0 * tref0
                tscale1 = dscale1 * tref1
                dq_in0, dq_in1 = _ld_global_f32x4_lo2_pred(
                    mDq.iterator + out_row * dq_row_stride + head_idx * dq_head_stride + col0,
                    row_valid,
                )
                dk_in0, dk_in1 = _ld_global_f32x4_lo2_pred(
                    mDk.iterator + out_row * dk_row_stride + head_idx * dk_head_stride + col0,
                    row_valid,
                )
                dg_in0, dg_in1 = _ld_global_f32x2_pred(
                    mDg.iterator + out_row * dg_row_stride + head_idx * dg_head_stride + col0,
                    row_valid,
                )
                dq_add0 = q_acc0 * qscale_prev0 + q_diag0 * qscale_diag0
                dq_add1 = q_acc1 * qscale_prev1 + q_diag1 * qscale_diag1
                dq_out0, dq_out1 = _add_f32x2(dq_in0, dq_in1, dq_add0, dq_add1)
                if lane_row == 0:
                    dq_word0 = _cvt_bf16x2_f32(dq_out0, dq_out1)
                else:
                    dq_word1 = _cvt_bf16x2_f32(dq_out0, dq_out1)
                dk_qk0 = k_acc0 * qscale_prev0 + k_diag0 * qscale_diag0
                dk_qk1 = k_acc1 * qscale_prev1 + k_diag1 * qscale_diag1
                dkt0 = d_acc0 * dscale0 + t_acc0 * tscale0
                dkt1 = d_acc1 * dscale1 + t_acc1 * tscale1
                dk_beta0 = dk_qk0 * beta_row
                dk_beta1 = dk_qk1 * beta_row
                dk_add0 = dk_beta0 + dkt0
                dk_add1 = dk_beta1 + dkt1
                dk_out0, dk_out1 = _add_f32x2(dk_in0, dk_in1, dk_add0, dk_add1)
                if lane_row == 0:
                    dk_word0 = _cvt_bf16x2_f32(dk_out0, dk_out1)
                else:
                    dk_word1 = _cvt_bf16x2_f32(dk_out0, dk_out1)
                dg_out0 = (dg_in0 + dq_add0 * qval0 + (dk_beta0 - dkt0) * kval0) * Float32(LN2)
                dg_out1 = (dg_in1 + dq_add1 * qval1 + (dk_beta1 - dkt1) * kval1) * Float32(LN2)
                if lane_row == 0:
                    dg_top0 = dg_out0
                    dg_top1 = dg_out1
                else:
                    dg_bottom0 = dg_out0
                    dg_bottom1 = dg_out1
                db_pair = dk_qk0 * kval0 + dk_qk1 * kval1
                if lane_row == 0:
                    db_acc0 = db_acc0 + db_pair
                else:
                    db_acc1 = db_acc1 + db_pair

            if group == 0:
                dq_top0 = dq_word0
                dq_bottom0 = dq_word1
                dk_top0 = dk_word0
                dk_bottom0 = dk_word1
                dg_top00 = dg_top0
                dg_top01 = dg_top1
                dg_bottom00 = dg_bottom0
                dg_bottom01 = dg_bottom1
            if group == 1:
                dq_top1 = dq_word0
                dq_bottom1 = dq_word1
                dk_top1 = dk_word0
                dk_bottom1 = dk_word1
                dg_top10 = dg_top0
                dg_top11 = dg_top1
                dg_bottom10 = dg_bottom0
                dg_bottom11 = dg_bottom1
            if group == 2:
                dq_top2 = dq_word0
                dq_bottom2 = dq_word1
                dk_top2 = dk_word0
                dk_bottom2 = dk_word1
                dg_top20 = dg_top0
                dg_top21 = dg_top1
                dg_bottom20 = dg_bottom0
                dg_bottom21 = dg_bottom1
            if group == 3:
                dq_top3 = dq_word0
                dq_bottom3 = dq_word1
                dk_top3 = dk_word0
                dk_bottom3 = dk_word1
                dg_top30 = dg_top0
                dg_top31 = dg_top1
                dg_bottom30 = dg_bottom0
                dg_bottom31 = dg_bottom1

        _st_shared_dg_triton_swizzle(
            sEpi_tile,
            0,
            dg_top00,
            dg_top01,
            dg_bottom00,
            dg_bottom01,
            tidx,
        )
        _st_shared_dg_triton_swizzle(
            sEpi_tile,
            1,
            dg_top10,
            dg_top11,
            dg_bottom10,
            dg_bottom11,
            tidx,
        )
        _st_shared_dg_triton_swizzle(
            sEpi_tile,
            2,
            dg_top20,
            dg_top21,
            dg_bottom20,
            dg_bottom21,
            tidx,
        )
        _st_shared_dg_triton_swizzle(
            sEpi_tile,
            3,
            dg_top30,
            dg_top31,
            dg_bottom30,
            dg_bottom31,
            tidx,
        )

        _st_global_f32_triton_shared_epilogue_16x32(
            sEpi_tile,
            mDg2.iterator
            + (row_start + row_base) * dg2_row_stride
            + head_idx * dg2_head_stride
            + k_start,
            tidx,
            row_base=row_base,
            valid=valid,
            row_stride=dg2_row_stride,
        )
        _st_global_bf16_ldmatrix_epilogue_16x32(
            sEpi_tile,
            mDq2.iterator
            + (row_start + row_base) * dq2_row_stride
            + head_idx * dq2_head_stride
            + k_start,
            dq_top0,
            dq_top1,
            dq_top2,
            dq_top3,
            dq_bottom0,
            dq_bottom1,
            dq_bottom2,
            dq_bottom3,
            tidx,
            row_base=row_base,
            valid=valid,
            row_stride=dq2_row_stride,
        )
        _st_global_bf16_ldmatrix_epilogue_16x32(
            sEpi_tile,
            mDk2.iterator
            + (row_start + row_base) * dk2_row_stride
            + head_idx * dk2_head_stride
            + k_start,
            dk_top0,
            dk_top1,
            dk_top2,
            dk_top3,
            dk_bottom0,
            dk_bottom1,
            dk_bottom2,
            dk_bottom3,
            tidx,
            row_base=row_base,
            valid=valid,
            row_stride=dk2_row_stride,
        )

        db_acc0 = db_acc0 + cute.arch.shuffle_sync_bfly(db_acc0, 1)
        db_acc0 = db_acc0 + cute.arch.shuffle_sync_bfly(db_acc0, 2)
        db_acc1 = db_acc1 + cute.arch.shuffle_sync_bfly(db_acc1, 1)
        db_acc1 = db_acc1 + cute.arch.shuffle_sync_bfly(db_acc1, 2)
        _st_global_f32_b32_pred(
            mDb2.iterator
            + k_phase * db2_k_stride
            + (row_start + row0) * db2_row_stride
            + head_idx * db2_head_stride,
            db_acc0,
            Boolean((tid == 0) & (row0 < valid)),
        )
        _st_global_f32_b32_pred(
            mDb2.iterator
            + k_phase * db2_k_stride
            + (row_start + row1) * db2_row_stride
            + head_idx * db2_head_stride,
            db_acc1,
            Boolean((tid == 0) & (row1 < valid)),
        )

        # Reused SMEM (sQ/sK/sG/sBeta/sEpi) is overwritten by the next
        # persistent iteration's staging; fence reads of this iteration first.
        cute.arch.barrier()


class ChunkKdaBwdIntraHmmaGrid:
    def __init__(self, use_i32_metadata: bool, grid_chunks: int):
        self.use_i32_metadata = use_i32_metadata
        self.grid_chunks = grid_chunks

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mdAqk: cute.Tensor,
        mdAkk: cute.Tensor,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDb2: cute.Tensor,
        mDg: cute.Tensor,
        mDq2: cute.Tensor,
        mDk2: cute.Tensor,
        mDg2: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mChunkIndices: cute.Tensor,
        mNumChunks: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        _chunk_kda_bwd_intra_hmma_grid_kernel(
            mQ,
            mK,
            mG,
            mBeta,
            mdAqk,
            mdAkk,
            mDq,
            mDk,
            mDb2,
            mDg,
            mDq2,
            mDk2,
            mDg2,
            mCuSeqlens,
            mChunkIndices,
            mNumChunks,
            self.use_i32_metadata,
            self.grid_chunks,
            _name_prefix="cutlass_dsl_chunk_kda_bwd_intra",
        ).launch(
            grid=(KC_TOTAL, self.grid_chunks, cute.size(mQ.shape[2])),
            block=(32, 1, 1),
            stream=stream,
        )


class ChunkKdaBwdIntraConfig(NamedTuple):
    """Compile-time persistent-grid choice for intra-chunk backward."""

    grid_chunks: int


@jit_cache
def _compile_chunk_kda_bwd_intra(heads: int, chunks: int, grid_chunks: int):
    """Compile one persistent fixed-length intra-chunk backward specialization."""
    if not 1 <= grid_chunks <= chunks:
        raise ValueError(f"grid_chunks must be in [1, {chunks}], got {grid_chunks}")
    op = ChunkKdaBwdIntraHmmaGrid(
        use_i32_metadata=True,
        grid_chunks=grid_chunks,
    )
    tokens, sequences = cute.sym_int(), cute.sym_int()

    def normal(dtype, shape):
        return make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=128,
        )

    def column_token_head(dtype, columns: int):
        return make_fake_compact_tensor(
            dtype,
            (columns, tokens, heads),
            stride_order=(0, 2, 1),
            assumed_align=128,
        )

    def strided_column_token_head(dtype, columns: int):
        """Transposed view of a [1, T, H, columns] input with runtime token/head strides."""
        return make_fake_tensor(
            dtype,
            (columns, tokens, heads),
            stride=(
                1,
                cute.sym_int(divisibility=_MIN_ALIGN_ELEMENTS_BF16),
                cute.sym_int(divisibility=_MIN_ALIGN_ELEMENTS_BF16),
            ),
            assumed_align=_MIN_ALIGN_BYTES,
        )

    q = strided_column_token_head(cutlass.BFloat16, KEY_DIM)
    k = strided_column_token_head(cutlass.BFloat16, KEY_DIM)
    g = column_token_head(cutlass.Float32, KEY_DIM)
    beta = normal(cutlass.Float32, (1, tokens, heads))
    dAqk = column_token_head(cutlass.Float32, BT)
    dAkk = column_token_head(cutlass.Float32, BT)
    dq = column_token_head(cutlass.Float32, KEY_DIM)
    dk = column_token_head(cutlass.Float32, KEY_DIM)
    db_partial = normal(cutlass.Float32, (K_PHASES, tokens, heads))
    dg = column_token_head(cutlass.Float32, KEY_DIM)
    dq2 = column_token_head(cutlass.BFloat16, KEY_DIM)
    dk2 = column_token_head(cutlass.BFloat16, KEY_DIM)
    dg2 = column_token_head(cutlass.Float32, KEY_DIM)
    cu_seqlens = normal(cutlass.Int32, (sequences,))
    chunk_indices = normal(cutlass.Int32, (chunks, 2))
    num_chunks = normal(cutlass.Int32, (1,))
    return compile_tvm_ffi(
        op,
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db_partial,
        dg,
        dq2,
        dk2,
        dg2,
        cu_seqlens,
        chunk_indices,
        num_chunks,
        name=f"kda_bwd_intra_h{heads}_c{chunks}_gc{grid_chunks}",
    )


def _column_token_head(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[0].permute(2, 0, 1)


class ChunkKdaBwdIntraTunable:
    class Args(NamedTuple):
        q: torch.Tensor
        k: torch.Tensor
        g: torch.Tensor
        beta: torch.Tensor
        dAqk: torch.Tensor
        dAkk: torch.Tensor
        dq: torch.Tensor
        dk: torch.Tensor
        db: torch.Tensor
        dg: torch.Tensor
        dq2: torch.Tensor
        dk2: torch.Tensor
        dg2: torch.Tensor
        db_partial: torch.Tensor
        cu_seqlens: torch.Tensor
        chunk_indices: torch.Tensor
        num_chunks: torch.Tensor

    # The public wrapper replaces this universal fallback with a target-aware default.
    default_config = ChunkKdaBwdIntraConfig(1)

    @staticmethod
    def default_for(chunks: int, sm_count: int) -> ChunkKdaBwdIntraConfig:
        """Prefer a smaller power-of-two grid when it evenly partitions the chunks."""
        grid_limit = min(chunks, 1 << (sm_count.bit_length() - 1))
        amortized_grid = max(1, grid_limit // 2)
        grid_chunks = amortized_grid if chunks % amortized_grid == 0 else min(chunks, sm_count)
        return ChunkKdaBwdIntraConfig(grid_chunks)

    @classmethod
    def configs(cls, args: Args) -> tuple[ChunkKdaBwdIntraConfig, ...]:
        """Generate persistent-grid candidates from sequence length and target size."""
        chunks = args.q.shape[1] // BT
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA tuning requires a CUDA target with an SM count")
        grid_limit = min(chunks, 1 << (sm_count.bit_length() - 1))
        values = (
            cls.default_for(chunks, sm_count).grid_chunks,
            max(1, grid_limit // 4),
            grid_limit,
            min(chunks, sm_count),
            min(chunks, grid_limit * 2),
        )
        return tuple(ChunkKdaBwdIntraConfig(value) for value in dict.fromkeys(values))

    @staticmethod
    def compile_call(config: ChunkKdaBwdIntraConfig, args: Args) -> tuple[int, int, int]:
        return args.q.shape[2], args.q.shape[1] // BT, config.grid_chunks

    compile = staticmethod(_compile_chunk_kda_bwd_intra)

    @staticmethod
    def launch(
        compiled, config: ChunkKdaBwdIntraConfig, args: Args
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del config
        compiled(
            _column_token_head(args.q),
            _column_token_head(args.k),
            _column_token_head(args.g),
            args.beta,
            _column_token_head(args.dAqk),
            _column_token_head(args.dAkk),
            _column_token_head(args.dq),
            _column_token_head(args.dk),
            args.db_partial,
            _column_token_head(args.dg),
            _column_token_head(args.dq2),
            _column_token_head(args.dk2),
            _column_token_head(args.dg2),
            args.cu_seqlens,
            args.chunk_indices,
            args.num_chunks.reshape(1),
        )
        return (
            args.dq2,
            args.dk2,
            args.dg2,
            args.db + args.db_partial.sum(0).unsqueeze(0),
        )


def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    metadata: ChunkMetadata,
    *,
    config: ChunkKdaBwdIntraConfig | None = None,
    tune: bool = False,
    configs: Iterable[ChunkKdaBwdIntraConfig] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the final intra-chunk backward stage.

    The returned gate gradient includes the ``ln(2)`` factor for differentiating ``2**g``;
    the incoming ``dg`` and this stage's contribution are both scaled at the final writer.
    """
    batch, tokens, heads, head_dim = q.shape
    if batch != 1 or head_dim != KEY_DIM:
        raise ValueError("the intra-chunk backward requires B=1 and K=128")
    if tokens % BT:
        raise ValueError("the intra-chunk backward requires complete 64-token chunks")

    # The kernel combines the incoming gradients with its intra-chunk contribution into
    # new output buffers. Beta is different: four K-phase CTAs contribute to every token,
    # so they write FP32 partials to global workspace for a race-free post-launch reduction.
    dq2 = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk2 = torch.empty_like(k, memory_format=torch.contiguous_format)
    dg2 = torch.empty_like(g, memory_format=torch.contiguous_format)
    db_partial = torch.empty(
        K_PHASES,
        tokens,
        heads,
        dtype=torch.float32,
        device=q.device,
    )
    args = ChunkKdaBwdIntraTunable.Args(
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dk=dk,
        db=db,
        dg=dg,
        dq2=dq2,
        dk2=dk2,
        dg2=dg2,
        db_partial=db_partial,
        cu_seqlens=metadata.cu_seqlens,
        chunk_indices=metadata.chunk_indices,
        num_chunks=metadata.num_chunks,
    )
    target = detect_compile_target(q.device.index)
    if not tune and config is None:
        if target.sm_count is None:
            raise RuntimeError("KDA launch requires a CUDA target with an SM count")
        config = ChunkKdaBwdIntraTunable.default_for(tokens // BT, target.sm_count)
    result, _ = run_tunable(
        ChunkKdaBwdIntraTunable,
        args,
        config=config,
        autotune=tune,
        configs=configs,
        parallel_compile=_compile_chunk_kda_bwd_intra.disk_cache_enabled(),
        target=target,
    )
    return result


__all__ = ["ChunkKdaBwdIntraConfig", "chunk_kda_bwd_intra"]
