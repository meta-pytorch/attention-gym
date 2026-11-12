# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/tile_dsl/handles.py and
# mma.py), rewritten by Attention Gym in 2026 on CuTeDSL 4.6.2 APIs: the instruction and shared
# memory descriptors are packed here from the PTX ISA bitfields and tcgen05.mma is inline PTX.

"""Hand-built tcgen05 MMA issue for the warpdrive kernels.

The warpdrive kernels address their SMEM operands as 128-byte (or 32-byte) swizzled boxes and their
TMEM operands as raw column offsets, and step the descriptors per K block themselves. This module
keeps that geometry: ``MmaDesc`` holds one GEMM's static shape and per-K-step descriptor advance,
``smem_desc`` packs a shared-memory matrix descriptor, ``instr_desc`` the 32-bit instruction
descriptor, and ``mma_ss``/``mma_ts_step`` issue the K loop from an elected lane. Completion is
the caller's ``tcgen05.commit`` (a warpdrive ``Umma`` channel signal).
"""

from dataclasses import dataclass

import cutlass
from cutlass import Int32, Int64, cute
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.arch import inline_ptx

from .ptx import elect_sync

# tcgen05 shared-memory descriptor swizzle modes (bits [61, 64)).
SWIZZLE_NONE = 0
SWIZZLE_128B = 2
SWIZZLE_64B = 4
SWIZZLE_32B = 6

_INPUT_FORMAT = {cutlass.Float16: 0, cutlass.BFloat16: 1}
_OUTPUT_FORMAT = {cutlass.Float16: 0, cutlass.Float32: 1}


def instr_desc(
    c_dtype, a_dtype, b_dtype, *, m: int, n: int, a_mn_major: bool = False, b_mn_major=False
) -> int:
    """``.kind::f16`` instruction descriptor: dense, no negation, no saturation.

    Bits: [4, 6) D format, [7, 10) A format, [10, 13) B format, 15/16 A/B MN-major,
    [17, 23) N >> 3, [24, 29) M >> 4 (PTX ISA, "Instruction descriptor")."""
    if m % 16 or n % 8 or not (64 <= m <= 256) or not (8 <= n <= 256):
        raise ValueError(f"unsupported tcgen05 tile M={m}, N={n}")
    return (
        _OUTPUT_FORMAT[c_dtype] << 4
        | _INPUT_FORMAT[a_dtype] << 7
        | _INPUT_FORMAT[b_dtype] << 10
        | int(a_mn_major) << 15
        | int(b_mn_major) << 16
        | (n >> 3) << 17
        | (m >> 4) << 24
    )


@cute.jit
def smem_desc(addr, leading_byte_offset: cutlass.Constexpr[int], stride_byte_offset, swizzle):
    """Shared-memory matrix descriptor (version 1) for 32-bit shared address ``addr``.

    Bits: [0, 14) address >> 4, [16, 30) leading byte offset >> 4, [32, 46) stride byte offset
    >> 4, [46, 49) version 1, [61, 64) swizzle (PTX ISA, "Shared memory descriptor")."""
    static = (
        (leading_byte_offset >> 4) << 16
        | (stride_byte_offset >> 4) << 32
        | 1 << 46
        | swizzle << 61
    )
    return Int64(static) | Int64((Int32(addr) >> 4) & 0x3FFF)


@cute.jit
def desc_opaque(value):
    """Identity ``mov.b64`` so LLVM cannot reassociate a descriptor base into per-K-step
    64-bit literals (the upstream kernels' schedule depends on it)."""
    return inline_ptx("mov.b64 $0, $1;", write_only_types=[Int64], read_only_args=[value])


@dataclass(frozen=True)
class MmaDesc:
    """Static geometry of one tcgen05 MMA and its per-K-step operand advance.

    ``atranspose``/``btranspose`` select MN-major operands; ``k_subtile`` is the K extent of one
    swizzle box (the whole K for MN-major operands). Byte strides are in bytes."""

    M: int
    N: int
    K: int
    bpe_a: int
    bpe_b: int
    idesc: int
    tile_k_hw: int = 16
    btranspose: bool = False
    atranspose: bool = False
    k_subtile: int = -1

    def __post_init__(self):
        if self.k_subtile < 0:
            if self.btranspose:
                ks = self.K
            elif (self.K * self.bpe_a) % 128 == 0:
                ks = 128 // self.bpe_a
            else:
                ks = 64 // self.bpe_a
            object.__setattr__(self, "k_subtile", ks)

    @property
    def num_k_steps(self) -> int:
        return self.K // self.tile_k_hw

    @property
    def steps_per_subtile(self) -> int:
        return self.k_subtile // self.tile_k_hw

    @property
    def num_subtiles(self) -> int:
        return self.num_k_steps // self.steps_per_subtile

    @staticmethod
    def _swizzle_bytes(inner_bytes: int) -> int:
        if inner_bytes % 128 == 0:
            return 128
        if inner_bytes % 64 == 0:
            return 64
        return 32

    @property
    def swz_a_bytes(self) -> int:
        return self._swizzle_bytes((self.M if self.atranspose else self.K) * self.bpe_a)

    @property
    def swz_b_bytes(self) -> int:
        return self._swizzle_bytes((self.N if self.btranspose else self.K) * self.bpe_b)

    @property
    def smem_advance_A_intra(self) -> int:
        if self.atranspose:
            return self.tile_k_hw * self.swz_a_bytes
        return self.tile_k_hw * self.bpe_a

    @property
    def smem_advance_B_intra(self) -> int:
        if self.btranspose:
            return self.tile_k_hw * self.swz_b_bytes
        return self.tile_k_hw * self.bpe_b

    @property
    def smem_subtile_A(self) -> int:
        return self.K * self.swz_a_bytes if self.atranspose else self.swz_a_bytes * self.M

    @property
    def smem_subtile_B(self) -> int:
        return self.K * self.swz_b_bytes if self.btranspose else self.swz_b_bytes * self.N

    @property
    def sps_A(self) -> int:
        if self.atranspose:
            return self.K // self.tile_k_hw
        return (self.swz_a_bytes // self.bpe_a) // self.tile_k_hw

    @property
    def sps_B(self) -> int:
        if self.btranspose:
            return self.K // self.tile_k_hw
        return (self.swz_b_bytes // self.bpe_b) // self.tile_k_hw

    @property
    def tmem_advance_A(self) -> int:
        """TMEM columns per K step of a 16-bit TMEM A operand (two values per column)."""
        return self.tile_k_hw * self.bpe_a // 4


@cute.jit
def _mma_ss_ptx(tmem_d, desc_a, desc_b, idesc: cutlass.Constexpr[int], accumulate):
    inline_ptx(
        "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, $3, p; }",
        read_only_args=[Int32(tmem_d), desc_a, desc_b, Int32(idesc), Int32(accumulate)],
    )


@cute.jit
def _mma_ts_ptx(tmem_d, tmem_a, desc_b, idesc: cutlass.Constexpr[int], accumulate):
    inline_ptx(
        "{ .reg .pred p; setp.ne.b32 p, $4, 0; "
        "tcgen05.mma.cta_group::1.kind::f16 [$0], [$1], $2, $3, p; }",
        read_only_args=[Int32(tmem_d), Int32(tmem_a), desc_b, Int32(idesc), Int32(accumulate)],
    )


@cute.jit
def mma_ss(
    desc: cutlass.Constexpr,
    desc_a_base,
    desc_b_base,
    tmem_d,
    accumulate,
    k_start: cutlass.Constexpr[int] = 0,
    k_count: cutlass.Constexpr = None,
):
    """Issue ``k_count`` K steps of an SMEM x SMEM MMA into TMEM column address ``tmem_d``.

    ``accumulate`` applies to the first step; later steps always accumulate."""
    k_count = desc.num_k_steps if cutlass.const_expr(k_count is None) else k_count
    for kk in cutlass.range_constexpr(k_count):
        k = k_start + kk
        inc_a = (
            desc.smem_advance_A_intra * (k % desc.sps_A) + desc.smem_subtile_A * (k // desc.sps_A)
        ) >> 4
        inc_b = (
            desc.smem_advance_B_intra * (k % desc.sps_B) + desc.smem_subtile_B * (k // desc.sps_B)
        ) >> 4
        acc = accumulate if cutlass.const_expr(kk == 0) else cutlass.Boolean(True)
        if elect_sync():
            _mma_ss_ptx(tmem_d, desc_a_base + inc_a, desc_b_base + inc_b, desc.idesc, acc)


@cute.jit
def mma_ts_step(
    desc: cutlass.Constexpr, tmem_a, desc_b_base, tmem_d, k: cutlass.Constexpr[int], accumulate
):
    """Issue K step ``k`` of a TMEM x SMEM MMA (16-bit A at TMEM column address ``tmem_a``)."""
    inc_b = (desc.smem_advance_B_intra * k) >> 4
    if elect_sync():
        _mma_ts_ptx(
            tmem_d, tmem_a + k * desc.tmem_advance_A, desc_b_base + inc_b, desc.idesc, accumulate
        )


@cute.jit
def mma_ts(desc: cutlass.Constexpr, tmem_a, desc_b_base, tmem_d, accumulate):
    """Issue every K step of a TMEM x SMEM MMA (16-bit A at TMEM column address ``tmem_a``),
    stepping the B descriptor across its swizzle boxes. ``accumulate`` applies to the first
    step; later steps always accumulate."""
    for k in cutlass.range_constexpr(desc.num_k_steps):
        inc_b = (
            desc.smem_advance_B_intra * (k % desc.sps_B) + desc.smem_subtile_B * (k // desc.sps_B)
        ) >> 4
        acc = accumulate if cutlass.const_expr(k == 0) else cutlass.Boolean(True)
        if elect_sync():
            _mma_ts_ptx(
                tmem_d, tmem_a + k * desc.tmem_advance_A, desc_b_base + inc_b, desc.idesc, acc
            )


@cute.jit
def _tmem_ptr(addr):
    return llvm.inttoptr(llvm.PointerType.get(6), Int32(addr).ir_value())


@cute.jit
def mma_ts_op(
    desc: cutlass.Constexpr,
    tmem_a,
    desc_b_base,
    tmem_d,
    k: cutlass.Constexpr[int],
    accumulate,
    issue,
):
    """``mma_ts_step`` issued through the NVVM ``tcgen05.mma`` op (the upstream wrapper's
    lowering) from the lanes where ``issue`` holds (an ``elect_sync()`` taken once at role
    entry): a constant ``accumulate`` folds into the instruction's predicate instead of a
    per-MMA ``setp``."""
    inc_b = (desc.smem_advance_B_intra * k) >> 4
    if issue:
        nvvm.tcgen05_mma(
            nvvm.Tcgen05MMAKind.F16,
            nvvm.CTAGroupKind.CTA_1,
            _tmem_ptr(tmem_d),
            _tmem_ptr(tmem_a + k * desc.tmem_advance_A),
            Int64(desc_b_base + inc_b).ir_value(),
            Int32(desc.idesc).ir_value(),
            cutlass.Boolean(accumulate).ir_value(),
        )


@cute.jit
def mma_ss_op(desc: cutlass.Constexpr, desc_a_base, desc_b_base, tmem_d, accumulate, issue):
    """``mma_ss`` (every K step) issued through the NVVM ``tcgen05.mma`` op, the upstream
    wrapper's lowering, from the lanes where ``issue`` holds (an ``elect_sync()`` taken once at
    role entry): a constant ``accumulate`` and the always-accumulating later steps fold into the
    instruction's predicate instead of a per-MMA ``setp``."""
    for k in cutlass.range_constexpr(desc.num_k_steps):
        inc_a = (
            desc.smem_advance_A_intra * (k % desc.sps_A) + desc.smem_subtile_A * (k // desc.sps_A)
        ) >> 4
        inc_b = (
            desc.smem_advance_B_intra * (k % desc.sps_B) + desc.smem_subtile_B * (k // desc.sps_B)
        ) >> 4
        acc = accumulate if cutlass.const_expr(k == 0) else cutlass.Boolean(True)
        if issue:
            nvvm.tcgen05_mma(
                nvvm.Tcgen05MMAKind.F16,
                nvvm.CTAGroupKind.CTA_1,
                _tmem_ptr(tmem_d),
                Int64(desc_a_base + inc_a).ir_value(),
                Int64(desc_b_base + inc_b).ir_value(),
                Int32(desc.idesc).ir_value(),
                cutlass.Boolean(acc).ir_value(),
            )


@cute.jit
def mma_ts_all_op(desc: cutlass.Constexpr, tmem_a, desc_b_base, tmem_d, accumulate, issue):
    """Every K step of a TMEM x SMEM MMA through ``mma_ts_op``, stepping the B descriptor
    across its swizzle boxes (the upstream ``num_subtiles_B x sps_B`` loop). ``accumulate``
    applies to the first step; later steps always accumulate."""
    for kk in cutlass.range_constexpr(desc.num_k_steps):
        box = kk // desc.sps_B
        acc = accumulate if cutlass.const_expr(kk == 0) else cutlass.Boolean(True)
        mma_ts_op(
            desc,
            tmem_a + box * desc.sps_B * desc.tmem_advance_A,
            desc_b_base + box * (desc.smem_subtile_B >> 4),
            tmem_d,
            kk - box * desc.sps_B,
            acc,
            issue,
        )
