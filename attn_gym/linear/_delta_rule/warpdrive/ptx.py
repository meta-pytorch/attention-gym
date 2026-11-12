# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/tile_dsl/), rewritten
# by Attention Gym in 2026 on CuTeDSL 4.6.2 APIs: cute.arch wrappers and inline PTX replace the
# cutlass.experimental.primitives bindings, so these helpers need no CuTeDSL 4.7 module.

"""Warp-level primitives shared by the warpdrive GDN/KDA kernels.

The warpdrive math runs on mma.sync fragments, ``ldmatrix``/``stmatrix`` over 128-byte swizzled
SMEM tiles and 16x256b TMEM loads. Each helper emits the same instruction the upstream
kernels issue, so the register fragments (and the numerics) match them exactly. Shared
memory addresses are 32-bit ``.shared`` offsets (``ptr.toint()`` of an SMEM pointer).
"""

import cutlass
from cutlass import Int32, cute
from cutlass._mlir.dialects import nvvm as mlir_nvvm
from cutlass.cute.arch import inline_ptx

FULL_MASK = 0xFFFFFFFF
SHFL_UP = mlir_nvvm.ShflKind.up
SHFL_IDX = mlir_nvvm.ShflKind.idx
SHFL_DOWN = mlir_nvvm.ShflKind.down
SHFL_BFLY = mlir_nvvm.ShflKind.bfly


def io_tag(dtype) -> str:
    if dtype is cutlass.Float16:
        return "f16"
    if dtype is cutlass.BFloat16:
        return "bf16"
    raise TypeError(f"warpdrive kernels take Float16 or BFloat16 operands, got {dtype}")


@cute.jit
def swizzle_xor_128b(row, col_elem):
    """Element column of 16-bit ``(row, col_elem)`` in a 128-byte swizzled row-major tile."""
    return col_elem ^ ((row & 7) * 8)


@cute.jit
def shfl_up(value, offset: cutlass.Constexpr[int]):
    return cute.arch.shuffle_sync(value, offset, FULL_MASK, 0, kind=SHFL_UP)


@cute.jit
def shfl_idx(value, lane, clamp: cutlass.Constexpr[int] = 31):
    return cute.arch.shuffle_sync(value, lane, FULL_MASK, clamp, kind=SHFL_IDX)


@cute.jit
def fp32_to_fp16(lo, hi, *, dtype: cutlass.Constexpr):
    """Round two fp32 values into one packed 16-bit pair (``lo`` in the low half)."""
    return inline_ptx(
        f"cvt.rn.{io_tag(dtype)}x2.f32 $0, $2, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo, hi],
    )


@cute.jit
def f16x2_to_f32(word, *, dtype: cutlass.Constexpr):
    """Unpack one packed 16-bit pair into ``(lo, hi)`` fp32 (inverse of ``fp32_to_fp16``)."""
    if cutlass.const_expr(dtype is cutlass.BFloat16):
        return inline_ptx(
            "{ .reg .b16 l, h; mov.b32 {l, h}, $2; mov.b32 $0, {0, l}; mov.b32 $1, {0, h}; }",
            write_only_types=[cutlass.Float32, cutlass.Float32],
            read_only_args=[word],
        )
    return inline_ptx(
        "{ .reg .b16 h0, h1; mov.b32 {h0, h1}, $2; cvt.f32.f16 $0, h0; cvt.f32.f16 $1, h1; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[word],
    )


@cute.jit
def opaque_f32_zero():
    """A 0.0f the optimizer cannot fold into an immediate packed-asm operand."""
    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Float32])


@cute.jit
def opaque_i32_zero():
    """A packed-zero b32 word the optimizer cannot prove constant."""
    return inline_ptx("mov.b32 $0, 0;", write_only_types=[cutlass.Int32])


@cute.jit
def fmul2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 multiply ``(a_lo * b_lo, a_hi * b_hi)`` (SM100 FMUL2)."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; "
        "mul.f32x2 pc, pa, pb; mov.b64 {$0, $1}, pc; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi],
    )


@cute.jit
def movmatrix_16b(value):
    """Transpose one packed m8n8 b16 register fragment."""
    return inline_ptx(
        "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[value],
    )


@cute.jit
def ldmatrix(addr, num: cutlass.Constexpr[int], trans: cutlass.Constexpr[bool] = False):
    """``ldmatrix.m8n8.x{num}`` from 32-bit shared address ``addr``; returns ``num`` b32 words."""
    regs = ", ".join(f"${i}" for i in range(num))
    t = ".trans" if trans else ""
    out = inline_ptx(
        f"ldmatrix.sync.aligned.m8n8.x{num}{t}.shared.b16 {{{regs}}}, [${num}];",
        write_only_types=[cutlass.Int32] * num,
        read_only_args=[addr],
    )
    return [out] if cutlass.const_expr(num == 1) else list(out)


@cute.jit
def stmatrix(addr, words, trans: cutlass.Constexpr[bool] = False):
    """``stmatrix.m8n8.x{len(words)}`` of packed b32 ``words`` to 32-bit shared ``addr``."""
    num = len(words)
    regs = ", ".join(f"${i + 1}" for i in range(num))
    t = ".trans" if trans else ""
    inline_ptx(
        f"stmatrix.sync.aligned.m8n8.x{num}{t}.shared.b16 [$0], {{{regs}}};",
        read_only_args=[addr, *words],
    )


# 32-bit registers per thread of one x1 tcgen05.ld/st of each shape.
_TMEM_REGS = {"32x32b": 1, "16x64b": 1, "16x128b": 2, "16x256b": 4}


@cute.jit
def tmem_ld(
    shape: cutlass.Constexpr[str], num: cutlass.Constexpr[int], taddr, dtype=cutlass.Float32
):
    """``tcgen05.ld.sync.aligned.{shape}.x{num}.b32`` at TMEM address ``taddr`` (lane << 16 |
    column); returns the thread's registers as ``dtype`` values. Complete with
    ``cute.arch.fence_view_async_tmem_load()`` before relying on them.

    For ``16x256b``, value ``k`` of each x1 block is row ``lane // 4 + 8 * ((k // 2) % 2)``,
    column ``2 * (lane % 4) + k % 2 + 8 * block`` (the mma.sync accumulator fragment)."""
    n = _TMEM_REGS[shape] * num
    regs = ", ".join(f"${i}" for i in range(n))
    out = inline_ptx(
        f"tcgen05.ld.sync.aligned.{shape}.x{num}.b32 {{{regs}}}, [${n}];",
        write_only_types=[dtype] * n,
        read_only_args=[Int32(taddr)],
    )
    return [out] if cutlass.const_expr(n == 1) else list(out)


@cute.jit
def tmem_st(shape: cutlass.Constexpr[str], taddr, values):
    """``tcgen05.st.sync.aligned.{shape}.x{num}.b32`` of 32-bit ``values`` (num from their count).
    Complete with ``cute.arch.fence_view_async_tmem_store()`` before a reader is signalled."""
    n = len(values)
    num = n // _TMEM_REGS[shape]
    regs = ", ".join(f"${i + 1}" for i in range(n))
    inline_ptx(
        f"tcgen05.st.sync.aligned.{shape}.x{num}.b32 [$0], {{{regs}}};",
        read_only_args=[Int32(taddr), *values],
    )


@cute.jit
def tmem_load_16x256b_x8(taddr):
    """32 fp32 values of 16 TMEM lanes x 64 columns at ``taddr`` (see ``tmem_ld``)."""
    return tmem_ld("16x256b", 8, taddr)


@cute.jit
def elect_sync():
    """``elect.sync`` over the full warp: true in one (the same) lane each call.

    Emitted as the NVVM op, not inline PTX: ptxas must see that the predicate selects one lane,
    or it wraps every uniform instruction under it (``tcgen05.mma``, ``tcgen05.commit``,
    ``mbarrier.arrive``) in a per-lane ``ELECT`` loop."""
    return cutlass.Boolean(mlir_nvvm.elect_sync())


@cute.jit
def _mma_sync(
    acc,
    a_frag,
    b_frag,
    k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
    shape: cutlass.Constexpr[str],
    a_words: cutlass.Constexpr[int],
    b_words: cutlass.Constexpr[int],
):
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    a_stride = len(a_frag) // M_BLOCKS
    tag = io_tag(dtype)
    a_regs = ",".join(f"${4 + i}" for i in range(a_words))
    b_regs = ",".join(f"${4 + a_words + i}" for i in range(b_words))
    c0 = 4 + a_words + b_words
    ptx = (
        f"mma.sync.aligned.{shape}.row.col.f32.{tag}.{tag}.f32"
        f" {{$0,$1,$2,$3}}, {{{a_regs}}}, {{{b_regs}}}, {{${c0},${c0 + 1},${c0 + 2},${c0 + 3}}};"
    )
    for m_block in cutlass.range_constexpr(M_BLOCKS):
        a_off = m_block * a_stride + k_step * a_words
        acc_base = m_block * N_FRAGS * 4
        for n_frag in cutlass.range_constexpr(N_FRAGS):
            s = acc_base + n_frag * 4
            out = inline_ptx(
                ptx,
                write_only_types=[cutlass.Float32] * 4,
                read_only_args=[
                    *[a_frag[a_off + i] for i in range(a_words)],
                    *[b_frag[n_frag * b_words + i] for i in range(b_words)],
                    acc[s],
                    acc[s + 1],
                    acc[s + 2],
                    acc[s + 3],
                ],
            )
            for i in cutlass.range_constexpr(4):
                acc[s + i] = out[i]


@cute.jit
def mma_step(
    acc,
    a_frag,
    b_frag,
    *,
    k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
):
    """``acc[M, N] += A[M, 16] @ B[16, N]`` with m16n8k16 mma.sync on packed fragments."""
    _mma_sync(acc, a_frag, b_frag, k_step, M, N, dtype, "m16n8k16", 4, 2)


@cute.jit
def mma_step_k8(
    acc,
    a_frag,
    b_frag,
    *,
    k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
):
    """``acc[M, N] += A[M, 8] @ B[8, N]`` with m16n8k8 mma.sync on packed fragments."""
    _mma_sync(acc, a_frag, b_frag, k_step, M, N, dtype, "m16n8k8", 2, 1)


@cute.jit
def sub_f16x2(lhs, rhs, *, dtype: cutlass.Constexpr):
    """Packed 16-bit subtract ``lhs - rhs`` of two b32 words (``sub.{f16,bf16}x2``)."""
    return inline_ptx(
        f"sub.{io_tag(dtype)}x2 $0, $1, $2;",
        write_only_types=[cutlass.Int32],
        read_only_args=[lhs, rhs],
    )


@cute.jit
def st_shared_v4(addr, words):
    """16-byte ``st.shared.v4.b32`` of four b32 ``words`` at 32-bit shared address ``addr``."""
    inline_ptx(
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        read_only_args=[Int32(addr), *words],
    )


@cute.jit
def fadd2(a_lo, a_hi, b_lo, b_hi):
    """Packed fp32 add ``(a_lo + b_lo, a_hi + b_hi)`` (SM100 FADD2)."""
    return cute.arch.add_packed_f32x2((a_lo, a_hi), (b_lo, b_hi))


@cute.jit
def st_global_v4_f32(addr, a, b, c, d):
    """``st.global.v4.f32`` of four fp32 values at 16-byte aligned 64-bit address ``addr``."""
    inline_ptx(
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        read_only_args=[cutlass.Int64(addr), a, b, c, d],
    )


@cute.jit
def cp_async_4b_zfill(smem_addr, gmem_addr, src_bytes):
    """``cp.async.ca`` of 4 bytes from global address ``gmem_addr`` (Int64) to 32-bit shared
    address ``smem_addr``; ``src_bytes`` (0 or 4) below 4 zero-fills without reading."""
    inline_ptx(
        "{ .reg .u64 g; cvta.to.global.u64 g, $1; cp.async.ca.shared.global [$0], [g], 4, $2; }",
        read_only_args=[Int32(smem_addr), cutlass.Int64(gmem_addr), Int32(src_bytes)],
    )


@cute.jit
def ffma2(a_lo, a_hi, b_lo, b_hi, c_lo, c_hi):
    """Packed fp32 fma ``(a_lo * b_lo + c_lo, a_hi * b_hi + c_hi)`` (SM100 FFMA2)."""
    return inline_ptx(
        "{ .reg .b64 pa, pb, pc, pd; mov.b64 pa, {$2, $3}; mov.b64 pb, {$4, $5}; "
        "mov.b64 pc, {$6, $7}; fma.rn.f32x2 pd, pa, pb, pc; mov.b64 {$0, $1}, pd; }",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[a_lo, a_hi, b_lo, b_hi, c_lo, c_hi],
    )


@cute.jit
def shfl_down(value, offset: cutlass.Constexpr[int], clamp: cutlass.Constexpr[int] = 31):
    return cute.arch.shuffle_sync(value, offset, FULL_MASK, clamp, kind=SHFL_DOWN)


@cute.jit
def shfl_bfly(value, offset: cutlass.Constexpr[int], clamp: cutlass.Constexpr[int] = 31):
    return cute.arch.shuffle_sync(value, offset, FULL_MASK, clamp, kind=SHFL_BFLY)


@cute.jit
def cp_async_ca_4(dst, src, src_bytes):
    """4-byte ``cp.async.ca.shared.global`` from 64-bit global ``src`` to 32-bit shared ``dst``;
    ``src_bytes`` (0 or 4) below 4 zero-fills the rest."""
    inline_ptx(
        "cp.async.ca.shared.global [$0], [$1], 4, $2;",
        read_only_args=[Int32(dst), cutlass.Int64(src), Int32(src_bytes)],
    )


def _vector_operands(first: int, n: int) -> str:
    """PTX operands ``$first .. $first + n - 1``, braced as a vector when ``n > 1``."""
    regs = ", ".join(f"${first + i}" for i in range(n))
    return regs if n == 1 else "{" + regs + "}"


@cute.jit
def ld_global_v(addr, dtype: cutlass.Constexpr, n: cutlass.Constexpr[int]):
    """``ld.global[.v2|.v4]`` of ``n`` 32-bit ``dtype`` values (``f32`` for Float32, else
    ``b32``) at 64-bit global address ``addr``; returns a list."""
    tag = "f32" if cutlass.const_expr(dtype is cutlass.Float32) else "b32"
    vec = "" if cutlass.const_expr(n == 1) else f".v{n}"
    out = inline_ptx(
        f"ld.global{vec}.{tag} {_vector_operands(0, n)}, [${n}];",
        write_only_types=[dtype] * n,
        read_only_args=[addr],
    )
    return [out] if cutlass.const_expr(n == 1) else list(out)


@cute.jit
def st_global_v(addr, values, dtype: cutlass.Constexpr):
    """``st.global[.v2|.v4]`` of 32-bit ``dtype`` ``values`` at 64-bit global address
    ``addr``."""
    n = len(values)
    tag = "f32" if cutlass.const_expr(dtype is cutlass.Float32) else "b32"
    vec = "" if cutlass.const_expr(n == 1) else f".v{n}"
    inline_ptx(
        f"st.global{vec}.{tag} [$0], {_vector_operands(1, n)};",
        read_only_args=[addr, *values],
    )


@cute.jit
def ld_shared_f32(ptr, n: cutlass.Constexpr[int]):
    """``ld.shared.v{n}.f32`` at SMEM pointer ``ptr`` (its LLVM pointer is the operand, as the
    upstream helpers pass it); returns a list."""
    return list(
        inline_ptx(
            f"ld.shared.v{n}.f32 {_vector_operands(0, n)}, [${n}];",
            write_only_types=[cutlass.Float32] * n,
            read_only_args=[ptr.llvm_ptr],
        )
    )


@cute.jit
def st_shared_f32(ptr, values):
    """``st.shared.v{len(values)}.f32`` of fp32 ``values`` at SMEM pointer ``ptr``."""
    inline_ptx(
        f"st.shared.v{len(values)}.f32 [$0], {_vector_operands(1, len(values))};",
        read_only_args=[ptr.llvm_ptr, *values],
    )
