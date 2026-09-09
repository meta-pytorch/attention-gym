# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: TMEM reduction loads, vector helpers, and packed-half math
# unused by the vendored kernels were removed.


import inspect

import cutlass
from cutlass import cute
from cutlass._mlir.dialects import nvvm as nvvm_ops
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

# nvidia-cutlass-dsl 4.7.0a0 nightlies from 2026-07-27 (6986e65) on regenerated
# the nvvm bindings to take an explicit result type -- *_packed_f32x2(res,
# src_a, src_b) -- but the cutlass.experimental.primitives wrappers still call
# the two-positional form, so going through them raises "missing 1 required
# positional argument: 'src_b'".  Call the dialect ops directly and adapt to
# whichever binding generation is installed; older ones infer the result type.
_PACKED_RES_FIRST = "res" in inspect.signature(nvvm_ops.mul_packed_f32x2).parameters


def _packed_f32x2(op, vec_a, vec_b):
    """``op(vec_a, vec_b)`` for nvvm.{mul,add}_packed_f32x2 on f32x2 IR values."""
    if _PACKED_RES_FIRST:
        return op(vec_a.type, vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)
    return op(vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)


@cute.jit
def fp32_to_fp16(lo, hi, *, dtype=cutlass.Float16):
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"fp32_to_fp16: dtype must be Float16 or BFloat16, got {dtype}")
    tag = "f16" if cutlass.const_expr(dtype == cutlass.Float16) else "bf16"
    return inline_ptx(
        f"cvt.rn.{tag}x2.f32 $0, $2, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


@cute.jit
def f16x2_to_f32(word, *, dtype=cutlass.Float16):
    """Unpack one Int32 (= 2 packed halves) into ``(lo_f32, hi_f32)`` Float32.

    The inverse of :func:`fp32_to_fp16`.  bf16 IS the top 16 bits of an fp32,
    so f32 = bf16 << 16 (bit move, no PRMT storm); masks stay 32-bit (a Python
    ``0xFFFF0000`` promotes to i64 -> mov.b32 mismatch).  fp16 needs the real
    ``cvt.f32.f16`` converts.
    """
    if cutlass.const_expr(dtype != cutlass.Float16 and dtype != cutlass.BFloat16):
        raise TypeError(f"f16x2_to_f32: dtype must be Float16 or BFloat16, got {dtype}")
    if cutlass.const_expr(dtype == cutlass.BFloat16):
        lo, hi = inline_ptx(
            "{ .reg .b16 l, h; mov.b32 {l, h}, $2; mov.b32 $0, {0, l}; mov.b32 $1, {0, h}; }",
            write_only_types=[cutlass.Float32, cutlass.Float32],
            read_only_args=[word],
        )
    else:
        lo, hi = inline_ptx(
            "{ .reg .b16 h0, h1; mov.b32 {h0, h1}, $2; cvt.f32.f16 $0, h0; cvt.f32.f16 $1, h1; }",
            write_only_types=[cutlass.Float32, cutlass.Float32],
            read_only_args=[word],
        )
    return lo, hi


@cute.jit
def opaque_f32_zero():
    """A 0.0f the optimizer cannot prove constant.

    Use for values that feed packed-asm operands (:func:`fmul2` /
    :func:`ffma2`) and could otherwise fold to a literal: the
    ``nvvm.inline_ptx`` lowering gives constant float operands the ``n``
    immediate constraint, which ICEs libNVVM."""
    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Float32])


@cute.jit
def fmul2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 multiply (SM100 FMUL2): ``(a_lo * b_lo, a_hi * b_hi)``.

    ``mul.f32x2`` operates on register pairs; the ``mov.b64`` packs map to
    register-pair allocation and usually fold away in SASS."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; "
        "mul.f32x2 pc, pa, pb; mov.b64 {$0, $1}, pc; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi],
    )


@cute.jit
def fadd2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 add (SM100 FADD2): ``(a_lo + b_lo, a_hi + b_hi)``.

    Native NVVM op, not inline asm: libNVVM rejects ``add.f32x2`` in asm
    blocks (mul/fma made it in, add did not)."""
    vec_a = cutlass.Vector.from_elements((a_lo, a_hi), cutlass.Float32)
    vec_b = cutlass.Vector.from_elements((b_lo, b_hi), cutlass.Float32)
    res = cutlass.Vector(
        _packed_f32x2(nvvm_ops.add_packed_f32x2, vec_a.ir_value(), vec_b.ir_value()),
        dtype=cutlass.Float32,
    )
    return cutlass.Float32(res[0]), cutlass.Float32(res[1])


@cute.jit
def ffma2(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi):
    """Packed fp32 fma (SM100 FFMA2): ``(a_lo*b_lo + c_lo, a_hi*b_hi + c_hi)``."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc, pd; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; "
        "mov.b64 pc, {$6, $7}; fma.rn.f32x2 pd, pa, pb, pc; mov.b64 {$0, $1}, pd; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi],
    )


@cute.jit
def movmatrix_16b(value: cutlass.Int32) -> cutlass.Int32:
    """Transpose one packed m8n8 b16 register fragment."""
    return inline_ptx(
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[value],
    )


@cute.jit
def sub_f16x2(lhs: cutlass.Int32, rhs: cutlass.Int32, input_dtype: cutlass.Constexpr) -> cutlass.Int32:
    """Subtract two packed pairs using the compile-time input dtype."""
    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return inline_ptx("sub.bf16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs])
    return inline_ptx("sub.f16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs])


@cute.jit
def beta_residual_f16x2(v_pair: cutlass.Int32, beta_lo, beta_hi, state_k_lo=None, state_k_hi=None, *, dtype=cutlass.Float16):
    """Stage the delta-rule update ``beta * (v - state_k)`` for one packed pair.

    ``v`` arrives as a packed b16 pair and ``state_k`` as the fp32 MMA
    accumulator; ``state_k`` may be omitted when no state is carried in.  The
    subtraction and beta scaling run in fp32 so the only rounding is the
    final pack into the b16 MMA operand.  Also returns the fp32 residual for
    consumers such as the dBeta v-term.
    """
    v_lo, v_hi = f16x2_to_f32(v_pair, dtype=dtype)
    if cutlass.const_expr(state_k_lo is not None):
        v_lo, v_hi = fadd2(v_lo, v_hi, -state_k_lo, -state_k_hi)
    y_lo, y_hi = fmul2(beta_lo, beta_hi, v_lo, v_hi)
    return fp32_to_fp16(y_lo, y_hi, dtype=dtype), v_lo, v_hi
