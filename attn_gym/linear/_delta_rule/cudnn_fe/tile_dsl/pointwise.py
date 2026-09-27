# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe. TMEM reduction loads, register-tile and vector helpers,
# FP8/FP4/MX conversions, the exp2 emulation, and the packed mul/sigmoid, lane_group_sum,
# l2norm_inv, and sigmoid2 helpers unused by the vendored kernels were removed; fadd2 uses the
# cute.arch packed wrapper; beta_residual_f16x2 was added to stage the KDA delta residual in
# FP32 (optionally without beta).


import cutlass
from cutlass import cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx


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
def pack_u16x2(lo: cutlass.Uint16, hi: cutlass.Uint16) -> cutlass.Int32:
    """Two 16-bit patterns into one 32-bit word (lo = low half)."""
    return cute.arch.inline_ptx(
        "mov.b32 $0, {$1, $2};",
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
def opaque_i32_zero():
    """A packed-zero b32 word the optimizer cannot prove constant.

    Same libNVVM immediate-constraint hazard as :func:`opaque_f32_zero`, for
    the packed 16x2 operands of :func:`sub_f16x2`."""
    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Int32])


@cute.jit
def opaque_i32(value: cutlass.Int32) -> cutlass.Int32:
    """Identity mov.b32 that pins a per-lane loop invariant in its register."""
    return inline_ptx("mov.b32 $0, $1;", write_only_types=[cutlass.Int32], read_only_args=[value])


@cute.jit
def fmul2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 multiply (SM100 FMUL2): ``(a_lo * b_lo, a_hi * b_hi)``.

    Kept as inline PTX: ``cute.arch.mul_packed_f32x2`` lets LLVM reschedule around the op and
    changes the SASS of the v1.30 kernels (stack/spill deltas in gdn_recompute and kda_summary).
    The ``mov.b64`` packs map to register-pair allocation and fold away in SASS."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; "
        "mul.f32x2 pc, pa, pb; mov.b64 {$0, $1}, pc; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi],
    )


@cute.jit
def fadd2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 add (SM100 FADD2): ``(a_lo + b_lo, a_hi + b_hi)``."""
    return cute.arch.add_packed_f32x2((a_lo, a_hi), (b_lo, b_hi))


@cute.jit
def ffma2(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi):
    """Packed fp32 fma (SM100 FFMA2): ``(a_lo*b_lo + c_lo, a_hi*b_hi + c_hi)``.

    Kept as inline PTX for the same SASS reason as :func:`fmul2`."""
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
def sub_f16x2(
    lhs: cutlass.Int32, rhs: cutlass.Int32, input_dtype: cutlass.Constexpr
) -> cutlass.Int32:
    """Subtract two packed pairs using the compile-time input dtype."""
    if cutlass.const_expr(input_dtype is cutlass.BFloat16):
        return inline_ptx(
            "sub.bf16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs]
        )
    return inline_ptx(
        "sub.f16x2 $0, $1, $2;", write_only_types=[cutlass.Int32], read_only_args=[lhs, rhs]
    )


@cute.jit
def beta_residual_f16x2(
    v_pair: cutlass.Int32,
    beta_lo,
    beta_hi,
    state_k_lo=None,
    state_k_hi=None,
    *,
    dtype=cutlass.Float16,
):
    """Stage the delta-rule update ``beta * (v - state_k)`` for one packed pair.

    ``v`` arrives as a packed b16 pair and ``state_k`` as the fp32 MMA
    accumulator; ``state_k`` may be omitted when no state is carried in.  The
    subtraction and beta scaling run in fp32 so the only rounding is the
    final pack into the b16 MMA operand.  Also returns the fp32 residual for
    consumers such as the dBeta v-term.  ``beta_lo=None`` skips the scaling
    (callers that apply beta later, e.g. through the KDA prep factors).
    """
    v_lo, v_hi = f16x2_to_f32(v_pair, dtype=dtype)
    if cutlass.const_expr(state_k_lo is not None):
        v_lo, v_hi = fadd2(v_lo, v_hi, -state_k_lo, -state_k_hi)
    if cutlass.const_expr(beta_lo is None):
        return fp32_to_fp16(v_lo, v_hi, dtype=dtype), v_lo, v_hi
    y_lo, y_hi = fmul2(beta_lo, beta_hi, v_lo, v_hi)
    return fp32_to_fp16(y_lo, y_hi, dtype=dtype), v_lo, v_hi


@cute.jit
def sigmoid(x: cutlass.Float32) -> cutlass.Float32:
    """sigmoid(x) via the tanh identity (single MUFU on Blackwell)."""
    half = cutlass.Float32(0.5)
    return cute.math.tanh(x * half, approx=True) * half + half


@cute.jit
def softplus(x: cutlass.Float32) -> cutlass.Float32:
    """log(1 + exp(x)) with the linear tail (x > 20 returns x: exp saturates
    fp32 there and log1p(exp(x)) == x to fp32 precision)."""
    result = x
    if x < cutlass.Float32(20.0):
        result = cute.math.log(
            cutlass.Float32(1.0) + cute.math.exp(x, fastmath=True), fastmath=True
        )
    return result


@cute.jit
def softplus2(x_lo, x_hi):
    """``(softplus(x_lo), softplus(x_hi))`` with the ``1 + exp`` step packed
    into one FADD2 and the linear tail applied as a select."""
    one = cutlass.Float32(1.0)
    tail = cutlass.Float32(20.0)
    exp_lo = cute.math.exp(x_lo, fastmath=True)
    exp_hi = cute.math.exp(x_hi, fastmath=True)
    sum_lo, sum_hi = fadd2(exp_lo, exp_hi, one, one)
    log_lo = cute.math.log(sum_lo, fastmath=True)
    log_hi = cute.math.log(sum_hi, fastmath=True)
    return (log_lo if x_lo < tail else x_lo), (log_hi if x_hi < tail else x_hi)
