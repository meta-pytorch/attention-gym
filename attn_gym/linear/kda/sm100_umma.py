# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared SM100 (Blackwell) tcgen05 UMMA helpers for KDA CuTe kernels.

Extracted verbatim from ``chunk_kda_bwd_wy_dqkg_fused.py`` so the fused WY/dQKG
backward and the intra-chunk engine share one copy of the descriptor,
``tcgen05.mma``/``.ws`` wrappers, TMEM load/store intrinsics, swizzled SMEM
accessors, IDESC constants, and chunk-route helpers.
"""

import os

import cutlass
import torch
from cutlass import cute, pipeline
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.arch import (
    elect_one,
)
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import BFloat16, Float32, Int32
from cutlass.cutlass_dsl import dsl_user_op

from attn_gym._backends.cute.target import get_compile_target
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import load_ragged_chunk_work

# ============================================================================
# Inlined SM100 tcgen05 helper wrappers
# ============================================================================

# ---- ptx_umma_ext.py ----
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Adapted from cuLA's SM100 tcgen05 MMA extension wrappers.
#
# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuteDSL UMMA extension wrappers for SM100 (Blackwell) ``tcgen05.mma``.

CuteDSL's high-level ``cute.gemm()`` / ``make_tiled_mma()`` API does not
expose all ``tcgen05.mma`` instruction variants.  This module provides
low-level wrappers for the two categories currently needed:

1. **Masked MMA** – SS and TS forms with the 128-bit ``disable-output-lane``
   mask operand (``{m0, m1, m2, m3}``).  Implemented via the native
   ``nvvm.tcgen05_mma`` MLIR op with its ``write_disable_mask`` parameter
   (``vector<4xi32>``).

2. **Weight-stationary (WS) MMA** – ``tcgen05.mma.ws`` SS / TS forms for
   both ``kind::tf32`` and ``kind::f16``.  Implemented via
   ``llvm.inline_asm``.

----------------------------------------------------------------------
PTX instruction forms
----------------------------------------------------------------------
SS (SMEM A, SMEM B):
    tcgen05.mma.cta_group::1.kind::tf32  [tmem_c], desc_a, desc_b,
                                          desc_val, {m0,m1,m2,m3}, p;

TS (TMEM A, SMEM B):
    tcgen05.mma.cta_group::1.kind::tf32  [tmem_c], [tmem_a], desc_b,
                                          desc_val, {m0,m1,m2,m3}, p;

WS_SS (weight-stationary, SMEM A, SMEM B):
    tcgen05.mma.ws.cta_group::1.kind::tf32  [tmem_c], desc_a, desc_b,
                                             desc_val, p;
    tcgen05.mma.ws.cta_group::1.kind::f16   [tmem_c], desc_a, desc_b,
                                             desc_val, p;

WS_TS (weight-stationary, TMEM A, SMEM B):
    tcgen05.mma.ws.cta_group::1.kind::tf32  [tmem_c], [tmem_a], desc_b,
                                             desc_val, p;
    tcgen05.mma.ws.cta_group::1.kind::f16   [tmem_c], [tmem_a], desc_b,
                                             desc_val, p;

----------------------------------------------------------------------
Disable-output-lane mask layout (4 × uint32 = 128 bits)
----------------------------------------------------------------------
Each uint32 covers 32 M-dimension rows (8 rows × 4 elements per group).
  0x00000000  → group is ACTIVE    (output written)
  0xFFFFFFFF  → group is DISABLED  (output suppressed)

Predefined SS mask constants (SMEM A variants):
  SS_NO_MASK  = (0, 0, 0, 0)                       all rows active
  SS_MASK0    = (0, 0xFF…, 0, 0xFF…)               odd groups disabled
  SS_MASK1    = (0xFF…, 0, 0xFF…, 0)               even groups disabled
  SS_MASK2    = (0xFF…, 0xFF…, 0, 0xFF…)           group 2 only active
  SS_MASK3    = (0xFF…, 0xFF…, 0xFF…, 0)           group 3 only active

Predefined TS mask constants (TMEM A variants):
  TS_NO_MASK  = (0, 0, 0, 0)                       all rows active
  TS_MASK0    = (0, 0xFF…, 0xFF…, 0xFF…)           group 0 only active
  TS_MASK1    = (0xFF…, 0, 0xFF…, 0xFF…)           group 1 only active
  TS_MASK2    = (0xFF…, 0xFF…, 0, 0xFF…)           group 2 only active
  TS_MASK3    = (0xFF…, 0xFF…, 0xFF…, 0)           group 3 only active
  TS_MASK02   = (0, 0xFF…, 0, 0xFF…)               groups 0,2 only active
  TS_MASK13   = (0xFF…, 0, 0xFF…, 0)               groups 1,3 only active

Public API (all decorated with @cute.jit)
----------------------------------------------------------------------
Descriptor helpers (call inside @cute.jit):
    Tcgen05SmemDescriptor          — 64-bit SMEM descriptor object
    initialize_tcgen05_descriptor  — fill descriptor bitfields

Low-level primitives (pass mask words explicitly):
    tcgen05mma_ss(desc_a, desc_b, tmem_c, desc_val, scale_out,
                  mask0, mask1, mask2, mask3)
    tcgen05mma_ts(tmem_a, desc_b, tmem_c, desc_val, scale_out,
                  mask0, mask1, mask2, mask3)
    tcgen05mma_ws_ss_tf32(desc_a, desc_b, tmem_c, desc_val, scale_out)
    tcgen05mma_ws_ts_tf32(tmem_a, desc_b, tmem_c, desc_val, scale_out)
    tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, desc_val, scale_out)
    tcgen05mma_ws_ts_f16(tmem_a, desc_b, tmem_c, desc_val, scale_out)

Named convenience wrappers (pre-set masks, pass only MMA operands):
    tcgen05mma_ss_no_mask / tcgen05mma_ss_mask0 / …mask1 / …mask2 / …mask3
    tcgen05mma_ts_no_mask / tcgen05mma_ts_mask0 / …mask1 / …mask2 / …mask3
    tcgen05mma_ts_mask02  / tcgen05mma_ts_mask13
"""

__all__ = [
    # collector enums (re-exported for convenience)
    "CollectorBBuffer",
    "CollectorOp",
    # descriptor helpers
    "Tcgen05SmemDescriptor",
    "initialize_tcgen05_descriptor",
    # low-level primitives
    "tcgen05mma_ss",
    "tcgen05mma_ss_mask0",
    "tcgen05mma_ss_mask1",
    "tcgen05mma_ss_mask2",
    "tcgen05mma_ss_mask3",
    # SS named wrappers
    "tcgen05mma_ss_no_mask",
    "tcgen05mma_ts",
    "tcgen05mma_ts_mask0",
    "tcgen05mma_ts_mask02",
    "tcgen05mma_ts_mask1",
    "tcgen05mma_ts_mask2",
    "tcgen05mma_ts_mask3",
    "tcgen05mma_ts_mask13",
    # TS named wrappers
    "tcgen05mma_ts_no_mask",
    "tcgen05mma_ws_ss_f16",
    "tcgen05mma_ws_ss_tf32",
    "tcgen05mma_ws_ts_f16",
    "tcgen05mma_ws_ts_tf32",
]

from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import nvvm as _nvvm

# Re-export collector enums for caller convenience.
CollectorBBuffer = _nvvm.Tcgen05MMACollectorBBuffer
CollectorOp = _nvvm.Tcgen05MMACollectorOp

# ---------------------------------------------------------------------------
# Mask constants (4 × uint32).  0 = ACTIVE, 0xFFFFFFFF = DISABLED.
# ---------------------------------------------------------------------------
_ALL_ACTIVE = 0x00000000
_ALL_OFF = 0xFFFFFFFF

# SS masks (SMEM A, SMEM B)
SS_NO_MASK = (_ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE)
SS_MASK0 = (_ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)  # {0,F,0,F}
SS_MASK1 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE)  # {F,0,F,0}
SS_MASK2 = (_ALL_OFF, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)  # {F,F,0,F}
SS_MASK3 = (_ALL_OFF, _ALL_OFF, _ALL_OFF, _ALL_ACTIVE)  # {F,F,F,0}

# TS masks (TMEM A, SMEM B)
TS_NO_MASK = (_ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE, _ALL_ACTIVE)
TS_MASK0 = (_ALL_ACTIVE, _ALL_OFF, _ALL_OFF, _ALL_OFF)  # {0,F,F,F}
TS_MASK1 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_OFF)  # {F,0,F,F}
TS_MASK2 = (_ALL_OFF, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)  # {F,F,0,F}
TS_MASK3 = (_ALL_OFF, _ALL_OFF, _ALL_OFF, _ALL_ACTIVE)  # {F,F,F,0}
TS_MASK02 = (_ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE, _ALL_OFF)  # {0,F,0,F}
TS_MASK13 = (_ALL_OFF, _ALL_ACTIVE, _ALL_OFF, _ALL_ACTIVE)  # {F,0,F,0}


# ---------------------------------------------------------------------------
# Tcgen05SmemDescriptor — 64-bit SMEM descriptor stored as 2×Int32
# ---------------------------------------------------------------------------


class Tcgen05SmemDescriptor:
    """64-bit shared-memory descriptor for tcgen05 MMA (Blackwell / SM100).

    The descriptor encodes SMEM base address, leading/stride byte offsets,
    swizzle mode, and other fields required by the ``tcgen05.mma`` PTX
    instruction to locate a matrix tile in shared memory.

    64-bit layout (PTX ISA Table 40)::

      Bit 63                                                      Bit 0
      ┌──────────┬────────┬─────┬──────────┬────┬──────────┬──────┬──────────────┐
      │ 63    61 │ 60  53 │  52 │ 51    49 │ 48 │ 45    32 │31 30 │ 29   16│15 14│ 13     0│
      │layout_typ│ reservd│l_abs│base_offst│ 46 │   SBO    │ rsvd │  LBO   │rsvd │start_adr│
      │  (3 bit) │ (8 bit)│(1b) │  (3 bit) │=0b001│(14 bit)│(2 b) │(14 bit)│(2b) │(14 bit) │
      └──────────┴────────┴─────┴──────────┴────┴──────────┴──────┴────────┴─────┴─────────┘

    Field descriptions:

    - **start_address** [bits 0-13]: SMEM base pointer, encoded as
      ``smem_ptr >> 4`` (16-byte aligned). The hardware reconstructs the
      full address as ``encoded_value << 4``.

    - **LBO** (Leading Byte Offset) [bits 16-29]: distance in bytes between
      consecutive elements along the leading dimension, encoded as
      ``lbo_bytes >> 4``.  When ``lbo_mode=1`` this is an absolute byte
      address rather than a relative offset.

    - **SBO** (Stride Byte Offset) [bits 32-45]: distance in bytes between
      consecutive elements along the stride dimension, encoded as
      ``sbo_bytes >> 4``.

    - **version** [bits 46-48]: fixed constant ``0b001`` (= 1).

    - **base_offset** [bits 49-51]: 3-bit alignment correction when the
      SMEM tile does not start at a natural swizzle-pattern boundary
      (1024B for 128B swizzle, 512B for 64B, 256B for 32B).
      Computed as ``(start_addr >> 7) & 0x7``.  Usually 0.

    - **lbo_mode** (leading_abs) [bit 52]: 0 → LBO is a relative byte
      offset; 1 → LBO is an absolute byte address.

    - **layout_type** (swizzle_mode) [bits 61-63]:
        - 0 = SWIZZLE_NONE
        - 1 = SWIZZLE_128B_BASE32B  (128-byte pattern, 32-byte atom)
        - 2 = SWIZZLE_128B          (128-byte pattern)
        - 4 = SWIZZLE_64B           (64-byte pattern)
        - 6 = SWIZZLE_32B           (32-byte pattern)

    Storage: two Int32 registers (desc[0] = low 32 bits, desc[1] = high 32
    bits), recast to a single Int64 for the PTX ``l``-constraint operand.

    Usage inside a @cute.jit kernel::

        desc = Tcgen05SmemDescriptor()
        initialize_tcgen05_descriptor(desc, smem_ptr, lbo, sbo, 0, True, swizzle)
    """

    def __init__(self, desc_64: cute.Int64 = None):
        # desc[0]: low  32 bits → start_address[0:14] | LBO[16:30]
        # desc[1]: high 32 bits → SBO[0:14] | version[14:16] | base_offset[17:20]
        #                         | lbo_mode[20] | layout_type[29:32]
        self.desc = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        # Alias the 2×i32 as 1×i64 for PTX "l" constraint (64-bit operand)
        self.desc_i64 = cute.make_tensor(
            cute.recast_ptr(self.desc.iterator, dtype=cute.Int64), (1,)
        )
        if desc_64 is not None:
            self.desc_i64[0] = desc_64

    def __add__(self, byte_offset):
        """Return a new descriptor offset by ``byte_offset`` bytes.

        Only the start_address field (bits 0-13 of desc[0]) is modified.
        Since it is stored in 16-byte units, we add ``byte_offset >> 4``.
        All other fields (LBO, SBO, swizzle, etc.) are copied unchanged.
        """
        res = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        res_i64 = cute.make_tensor(cute.recast_ptr(res.iterator, dtype=cute.Int64), (1,))
        res[0] = self.desc[0] + (byte_offset >> 4)  # adjust start_address
        res[1] = self.desc[1]  # high word unchanged
        return Tcgen05SmemDescriptor(res_i64[0])


# ---------------------------------------------------------------------------
# initialize_tcgen05_descriptor
# ---------------------------------------------------------------------------


def initialize_tcgen05_descriptor(
    desc,
    start_address,
    leading_byte_offset,
    stride_byte_offset,
    base_offset,
    leading_abs,
    swizzle_mode,
):
    """Pack SMEM descriptor bitfields into *desc* (a Tcgen05SmemDescriptor).

    Constructs the 64-bit descriptor in two 32-bit halves (desc[0] and desc[1]).
    All address/offset fields must be pre-divided by 16 (``>> 4``) before
    passing, because the hardware stores them in 16-byte granularity.

    Low 32 bits — desc[0]::

      ┌────────────────┬──────┬──────────────────┐
      │ bits 29…16     │15…14 │ bits 13…0        │
      │ LBO (14 bits)  │ rsvd │ start_addr >> 4   │
      └────────────────┴──────┴──────────────────┘

      - [0:14)   start_address >> 4  — SMEM tile base pointer in 16B units.
      - [14:16)  reserved (0).
      - [16:30)  leading_byte_offset — LBO in 16B units (caller passes >> 4).

    High 32 bits — desc[1]::

      ┌────────┬────────┬─────┬──────────┬────────┬──────────────────┐
      │ 31…29  │ 28…21  │  20 │ 19…17    │ 16…14  │ bits 13…0        │
      │ layout │  rsvd  │l_abs│base_off  │version │ SBO (14 bits)    │
      │ (3 bit)│ (8 bit)│(1b) │  (3 bit) │=0b001  │                  │
      └────────┴────────┴─────┴──────────┴────────┴──────────────────┘

      - [0:14)   stride_byte_offset — SBO in 16B units (caller passes >> 4).
      - [14:16)  version = 1 (fixed constant 0b001, only bit 14 set).
      - [17:20)  base_offset & 0x7 — swizzle alignment correction.
                 Typically 0.  Non-zero when the tile doesn't start at
                 the natural swizzle boundary (1024B/512B/256B).
      - [20:21)  lbo_mode — 0 = LBO is relative offset, 1 = absolute address.
      - [29:32)  layout_type (swizzle_mode & 0x7):
                   0 = SWIZZLE_NONE
                   1 = SWIZZLE_128B_BASE32B  (Swizzle<2,5,2>)
                   2 = SWIZZLE_128B          (Swizzle<3,4,3>)
                   4 = SWIZZLE_64B           (Swizzle<2,4,3>)
                   6 = SWIZZLE_32B           (Swizzle<1,4,3>)

    Args:
        desc:                 Tcgen05SmemDescriptor to fill.
        start_address:        CuTeDSL Pointer to the SMEM tile start.
        leading_byte_offset:  Leading-dimension byte offset, already >> 4.
        stride_byte_offset:   Stride  byte offset, already >> 4.
        base_offset:          Swizzle alignment correction (raw int, bits 17-19).
        leading_abs:          Bool — True → LBO is absolute address.
        swizzle_mode:         Swizzle layout_type integer (bits 29-31).
    """
    # Encode start_address: take SMEM pointer, shift right by 4 to get 16B units
    ptr_val = start_address.toint() >> 4

    # --- Low 32 bits (desc[0]) ---
    # bits [0:14)  = start_address >> 4
    # bits [16:30) = leading_byte_offset (already in 16B units)
    desc.desc[0] = cutlass.Int32(ptr_val) | cutlass.Int32(cutlass.Int32(leading_byte_offset) << 16)

    # --- High 32 bits (desc[1]) ---
    # bits [0:14)  = stride_byte_offset (already in 16B units)
    # bit  [14]    = version = 1  (fixed)
    # bits [17:20) = base_offset & 0x7  (swizzle alignment correction)
    # bit  [20]    = lbo_mode  (0=relative, 1=absolute)
    # bits [29:32) = layout_type  (swizzle mode)
    desc.desc[1] = (
        cutlass.Int32(stride_byte_offset)
        | cutlass.Int32(1 << 14)  # version = 1
        | cutlass.Int32(cutlass.Int32(base_offset & 0x7) << 17)
        | cutlass.Int32(cutlass.Int32(int(leading_abs)) << 20)
        | cutlass.Int32(cutlass.Int32(swizzle_mode & 0x7) << 29)
    )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _ir(val, loc=None, ip=None):
    """Extract raw MLIR IR value from a CuTeDSL wrapper."""
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val


# ===========================================================================
# Low-level primitives
# ===========================================================================

# ---------------------------------------------------------------------------
# tcgen05mma_ss  —  SMEM A, SMEM B (non-warp-specialised)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ss(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    mask0: int,
    mask1: int,
    mask2: int,
    mask3: int,
):
    """Issue ``tcgen05.mma.cta_group::1.kind::tf32`` with SMEM operands.

    ``mask{0-3}`` are the four uint32 words of the 128-bit
    ``disable-output-lane`` mask (0=active, 0xFFFFFFFF=disabled).

    Caller must ensure single-thread execution (e.g. via ``elect_one``);
    no internal ``elect.sync`` is performed.

    Args:
        desc_a:    64-bit SMEM descriptor for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate into C, 0 → overwrite C (clear accumulators).
        mask0-3:   Four uint32 words of the disable-output-lane mask.
    """

    @dsl_user_op
    def _do(
        c_val,
        da_val,
        db_val,
        dv_val,
        sc_val,
        m0_val,
        m1_val,
        m2_val,
        m3_val,
        *,
        loc=None,
        ip=None,
    ):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i32_ty = ir.IntegerType.get_signless(32)
        i1_ty = ir.IntegerType.get_signless(1)
        vec4i32_ty = ir.VectorType.get([4], i32_ty)

        c_ir = _ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        da_ir = _ir(da_val, loc, ip)  # i64 SMEM descriptor
        db_ir = _ir(db_val, loc, ip)  # i64 SMEM descriptor
        dv_ir = _ir(dv_val, loc, ip)
        sc_ir = _ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        m0_ir = _ir(m0_val, loc, ip)
        m1_ir = _ir(m1_val, loc, ip)
        m2_ir = _ir(m2_val, loc, ip)
        m3_ir = _ir(m3_val, loc, ip)

        undef = llvm.mlir_undef(vec4i32_ty, loc=loc, ip=ip)
        idx0 = _arith.constant(i32_ty, 0, loc=loc, ip=ip)
        idx1 = _arith.constant(i32_ty, 1, loc=loc, ip=ip)
        idx2 = _arith.constant(i32_ty, 2, loc=loc, ip=ip)
        idx3 = _arith.constant(i32_ty, 3, loc=loc, ip=ip)
        v = llvm.InsertElementOp(undef, m0_ir, idx0, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m1_ir, idx1, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m2_ir, idx2, loc=loc, ip=ip)
        mask = llvm.InsertElementOp(v, m3_ir, idx3, loc=loc, ip=ip)

        _nvvm.tcgen05_mma(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            d=d_ptr,
            a=da_ir,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            write_disable_mask=mask,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
        cutlass.Int32(mask0),
        cutlass.Int32(mask1),
        cutlass.Int32(mask2),
        cutlass.Int32(mask3),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ts  —  TMEM A, SMEM B (non-warp-specialised)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ts(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    mask0: int,
    mask1: int,
    mask2: int,
    mask3: int,
):
    """Issue ``tcgen05.mma.cta_group::1.kind::tf32`` with TMEM A operand.

    Matrix A is read from TMEM via indirect addressing ``[tmem_a]``.
    Matrix B is read from SMEM via descriptor.
    Caller must ensure single-thread execution (e.g. via ``elect_one``).

    Args:
        tmem_a:    TMEM base address (uint32) for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate into C, 0 → overwrite C.
        mask0-3:   Four uint32 words of the disable-output-lane mask.
    """

    @dsl_user_op
    def _do(
        c_val,
        a_val,
        db_val,
        dv_val,
        sc_val,
        m0_val,
        m1_val,
        m2_val,
        m3_val,
        *,
        loc=None,
        ip=None,
    ):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i32_ty = ir.IntegerType.get_signless(32)
        i1_ty = ir.IntegerType.get_signless(1)
        vec4i32_ty = ir.VectorType.get([4], i32_ty)

        c_ir = _ir(c_val, loc, ip)
        a_ir = _ir(a_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        a_ptr = llvm.inttoptr(ptr6_ty, a_ir, loc=loc, ip=ip)
        b_ir = _ir(db_val, loc, ip)
        dv_ir = _ir(dv_val, loc, ip)
        sc_ir = _ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        m0_ir = _ir(m0_val, loc, ip)
        m1_ir = _ir(m1_val, loc, ip)
        m2_ir = _ir(m2_val, loc, ip)
        m3_ir = _ir(m3_val, loc, ip)

        undef = llvm.mlir_undef(vec4i32_ty, loc=loc, ip=ip)
        idx0 = _arith.constant(i32_ty, 0, loc=loc, ip=ip)
        idx1 = _arith.constant(i32_ty, 1, loc=loc, ip=ip)
        idx2 = _arith.constant(i32_ty, 2, loc=loc, ip=ip)
        idx3 = _arith.constant(i32_ty, 3, loc=loc, ip=ip)
        v = llvm.InsertElementOp(undef, m0_ir, idx0, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m1_ir, idx1, loc=loc, ip=ip)
        v = llvm.InsertElementOp(v, m2_ir, idx2, loc=loc, ip=ip)
        mask = llvm.InsertElementOp(v, m3_ir, idx3, loc=loc, ip=ip)

        _nvvm.tcgen05_mma(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            d=d_ptr,
            a=a_ptr,
            b=b_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            write_disable_mask=mask,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
        cutlass.Int32(mask0),
        cutlass.Int32(mask1),
        cutlass.Int32(mask2),
        cutlass.Int32(mask3),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ws_ss_tf32  —  weight-stationary, SMEM A, SMEM B, kind::tf32
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ss_tf32(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    collector_b_buffer=None,
    collector_op=None,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::tf32`` (weight-stationary form).

    This variant does NOT take a ``disable-output-lane`` mask; the
    optional ``zero-column-mask-desc`` operand is omitted.

    Args:
        desc_a:    64-bit SMEM descriptor for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate, 0 → overwrite.
        collector_b_buffer: Optional ``CollectorBBuffer`` enum (B0–B3).
                            Defaults to None (hardware default: ``b0::discard``).
        collector_op: Optional ``CollectorOp`` enum (FILL/USE/LASTUSE/DISCARD).
                      Defaults to None (hardware default: discard).
    """

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i1_ty = ir.IntegerType.get_signless(1)

        c_ir = _ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        da_ir = _ir(da_val, loc, ip)
        db_ir = _ir(db_val, loc, ip)
        dv_ir = _ir(dv_val, loc, ip)
        sc_ir = _ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            d=d_ptr,
            a=da_ir,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ws_ss_f16  —  weight-stationary, SMEM A, SMEM B, kind::f16
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ss_f16(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    collector_b_buffer=None,
    collector_op=None,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::f16`` (weight-stationary form).

    Same as the tf32 variant but uses ``.kind::f16`` for half-precision
    input types (f16 / bf16).  K dimension is 16 instead of 8.

    This variant does NOT take a ``disable-output-lane`` mask; the
    optional ``zero-column-mask-desc`` operand is omitted.

    Args:
        desc_a:    64-bit SMEM descriptor for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate, 0 → overwrite.
        collector_b_buffer: Optional ``CollectorBBuffer`` enum (B0–B3).
                            Defaults to None (hardware default: ``b0::discard``).
        collector_op: Optional ``CollectorOp`` enum (FILL/USE/LASTUSE/DISCARD).
                      Defaults to None (hardware default: discard).
    """

    @dsl_user_op
    def _do(c_val, da_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i1_ty = ir.IntegerType.get_signless(1)

        c_ir = _ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        da_ir = _ir(da_val, loc, ip)
        db_ir = _ir(db_val, loc, ip)
        dv_ir = _ir(dv_val, loc, ip)
        sc_ir = _ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.F16,
            d=d_ptr,
            a=da_ir,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        desc_a.desc_i64[0],
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ws_ts_tf32  —  weight-stationary, TMEM A, SMEM B, kind::tf32
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ts_tf32(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    collector_b_buffer=None,
    collector_op=None,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::tf32`` with TMEM A (weight-stationary).

    Matrix A is read from TMEM via indirect addressing ``[tmem_a]``.
    Matrix B is read from SMEM via descriptor.
    This variant does NOT take a ``disable-output-lane`` mask; the
    optional ``zero-column-mask-desc`` operand is omitted.

    Args:
        tmem_a:    TMEM base address (uint32) for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate, 0 → overwrite.
        collector_b_buffer: Optional ``CollectorBBuffer`` enum (B0–B3).
                            Defaults to None (hardware default: ``b0::discard``).
        collector_op: Optional ``CollectorOp`` enum (FILL/USE/LASTUSE/DISCARD).
                      Defaults to None (hardware default: discard).
    """

    @dsl_user_op
    def _do(c_val, a_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i1_ty = ir.IntegerType.get_signless(1)

        c_ir = _ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        a_ir = _ir(a_val, loc, ip)
        a_ptr = llvm.inttoptr(ptr6_ty, a_ir, loc=loc, ip=ip)
        db_ir = _ir(db_val, loc, ip)
        dv_ir = _ir(dv_val, loc, ip)
        sc_ir = _ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.TF32,
            d=d_ptr,
            a=a_ptr,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ---------------------------------------------------------------------------
# tcgen05mma_ws_ts_f16  —  weight-stationary, TMEM A, SMEM B, kind::f16
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ws_ts_f16(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
    collector_b_buffer=None,
    collector_op=None,
):
    """Issue ``tcgen05.mma.ws.cta_group::1.kind::f16`` with TMEM A (weight-stationary).

    Same as the tf32 variant but uses ``.kind::f16`` for half-precision
    input types (f16 / bf16).  K dimension is 16 instead of 8.

    Matrix A is read from TMEM via indirect addressing ``[tmem_a]``.
    Matrix B is read from SMEM via descriptor.
    This variant does NOT take a ``disable-output-lane`` mask; the
    optional ``zero-column-mask-desc`` operand is omitted.

    Args:
        tmem_a:    TMEM base address (uint32) for matrix A.
        desc_b:    64-bit SMEM descriptor for matrix B.
        tmem_c:    TMEM base address (uint32) for accumulators C/D.
        desc_val:  High 32 bits of the UMMA instruction descriptor (idescE>>32).
        scale_out: 1 → accumulate, 0 → overwrite.
        collector_b_buffer: Optional ``CollectorBBuffer`` enum (B0–B3).
                            Defaults to None (hardware default: ``b0::discard``).
        collector_op: Optional ``CollectorOp`` enum (FILL/USE/LASTUSE/DISCARD).
                      Defaults to None (hardware default: discard).
    """

    @dsl_user_op
    def _do(c_val, a_val, db_val, dv_val, sc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        i1_ty = ir.IntegerType.get_signless(1)

        c_ir = _ir(c_val, loc, ip)
        d_ptr = llvm.inttoptr(ptr6_ty, c_ir, loc=loc, ip=ip)
        a_ir = _ir(a_val, loc, ip)
        a_ptr = llvm.inttoptr(ptr6_ty, a_ir, loc=loc, ip=ip)
        db_ir = _ir(db_val, loc, ip)
        dv_ir = _ir(dv_val, loc, ip)
        sc_ir = _ir(sc_val, loc, ip)
        enable_d = _arith.trunci(i1_ty, sc_ir, loc=loc, ip=ip)

        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.F16,
            d=d_ptr,
            a=a_ptr,
            b=db_ir,
            idesc=dv_ir,
            enable_input_d=enable_d,
            collector_b_buffer=collector_b_buffer,
            collector_op=collector_op,
            loc=loc,
            ip=ip,
        )

    _do(
        cutlass.Int32(tmem_c),
        cutlass.Int32(tmem_a),
        desc_b.desc_i64[0],
        cutlass.Int32(desc_val),
        cutlass.Int32(scale_out),
    )


# ===========================================================================
# Named convenience wrappers
# ===========================================================================
# These call the low-level primitives with pre-set mask constants so callers
# do not need to repeat the literal values.  Signature: same as the base
# function but without the mask0-3 args.

# ---------------------------------------------------------------------------
# SS named wrappers  (SMEM A)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ss_no_mask(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA with no output-lane disable (all rows active)."""
    tcgen05mma_ss(
        desc_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        SS_NO_MASK[0],
        SS_NO_MASK[1],
        SS_NO_MASK[2],
        SS_NO_MASK[3],
    )


@cute.jit
def tcgen05mma_ss_mask0(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0, 0xF…, 0, 0xF…} — groups 0,2 active (1,3 disabled)."""
    tcgen05mma_ss(
        desc_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        SS_MASK0[0],
        SS_MASK0[1],
        SS_MASK0[2],
        SS_MASK0[3],
    )


@cute.jit
def tcgen05mma_ss_mask1(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0, 0xF…, 0} — groups 1,3 active (0,2 disabled)."""
    tcgen05mma_ss(
        desc_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        SS_MASK1[0],
        SS_MASK1[1],
        SS_MASK1[2],
        SS_MASK1[3],
    )


@cute.jit
def tcgen05mma_ss_mask2(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0xF…, 0, 0xF…} — group 2 only active."""
    tcgen05mma_ss(
        desc_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        SS_MASK2[0],
        SS_MASK2[1],
        SS_MASK2[2],
        SS_MASK2[3],
    )


@cute.jit
def tcgen05mma_ss_mask3(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """SS MMA: mask={0xF…, 0xF…, 0xF…, 0} — group 3 only active."""
    tcgen05mma_ss(
        desc_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        SS_MASK3[0],
        SS_MASK3[1],
        SS_MASK3[2],
        SS_MASK3[3],
    )


# ---------------------------------------------------------------------------
# TS named wrappers  (TMEM A)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05mma_ts_no_mask(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA with no output-lane disable (all rows active)."""
    tcgen05mma_ts(
        tmem_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        TS_NO_MASK[0],
        TS_NO_MASK[1],
        TS_NO_MASK[2],
        TS_NO_MASK[3],
    )


@cute.jit
def tcgen05mma_ts_mask0(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0, 0xF…, 0xF…, 0xF…} — group 0 only active."""
    tcgen05mma_ts(
        tmem_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        TS_MASK0[0],
        TS_MASK0[1],
        TS_MASK0[2],
        TS_MASK0[3],
    )


@cute.jit
def tcgen05mma_ts_mask1(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0, 0xF…, 0xF…} — group 1 only active."""
    tcgen05mma_ts(
        tmem_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        TS_MASK1[0],
        TS_MASK1[1],
        TS_MASK1[2],
        TS_MASK1[3],
    )


@cute.jit
def tcgen05mma_ts_mask2(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0xF…, 0, 0xF…} — group 2 only active."""
    tcgen05mma_ts(
        tmem_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        TS_MASK2[0],
        TS_MASK2[1],
        TS_MASK2[2],
        TS_MASK2[3],
    )


@cute.jit
def tcgen05mma_ts_mask3(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0xF…, 0xF…, 0} — group 3 only active."""
    tcgen05mma_ts(
        tmem_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        TS_MASK3[0],
        TS_MASK3[1],
        TS_MASK3[2],
        TS_MASK3[3],
    )


@cute.jit
def tcgen05mma_ts_mask02(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0, 0xF…, 0, 0xF…} — groups 0,2 active (1,3 disabled).

    Used in the KDA intra-chunk backward kernel for the QK/KG phase where
    only even row-groups of the M tile contribute to the triangular region.
    """
    tcgen05mma_ts(
        tmem_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        TS_MASK02[0],
        TS_MASK02[1],
        TS_MASK02[2],
        TS_MASK02[3],
    )


@cute.jit
def tcgen05mma_ts_mask13(
    tmem_a: int,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """TS MMA: mask={0xF…, 0, 0xF…, 0} — groups 1,3 active (0,2 disabled).

    Used in the KDA intra-chunk backward kernel for the QK/KG phase where
    only odd row-groups of the M tile contribute to the triangular region.
    """
    tcgen05mma_ts(
        tmem_a,
        desc_b,
        tmem_c,
        desc_val,
        scale_out,
        TS_MASK13[0],
        TS_MASK13[1],
        TS_MASK13[2],
        TS_MASK13[3],
    )


# ---- intrinsics_sm100.py ----
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#
# Adapted from cuLA's SM100 Tensor Memory intrinsic wrappers.
#
# Copyright 2025-2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVVM wrappers for SM100 (Blackwell) Tensor Memory intrinsics.

Provides low-level, CuteDSL-compatible helpers that move data between
Tensor Memory (TMEM) and registers / shared memory via the native
``nvvm.tcgen05.*`` MLIR ops.

**T2R / R2T** – ``tcgen05.ld`` / ``tcgen05.st`` with ``.32x32b`` shape.
**S2T**       – ``tcgen05.cp`` with ``.128x256b`` shape (SMEM → TMEM)
PTX reference
-------------
    tcgen05.ld.sync.aligned.32x32b.xN.b32  {r0, ..., rN-1}, [taddr];
    tcgen05.st.sync.aligned.32x32b.xN.b32  [taddr], {r0, ..., rN-1};

where ``N ∈ {2, 4, 8, 16, 32, 64, 128}`` and each ``r`` is a 32-bit
register.  ``taddr`` encodes both the TMEM column index (bits [15:0])
and the lane index (bits [31:16]).

See https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld

Usage inside a ``@cute.kernel`` or ``@cute.jit`` function::

    from this file's inlined SM100 helper wrappers import (
        tcgen05_ld_32x32b, tcgen05_st_32x32b,
        reinterpret_cast, subvec, store_256b,
    )
    from cutlass.cute.typing import Float32, Int32

    # Load 32 × 32-bit values from TMEM → opaque vector<32 x i32>
    vec_i32 = tcgen05_ld_32x32b(32, taddr)

    # Zero-cost reinterpret as f32 (single vector.bitcast, no instructions)
    vec_f32 = reinterpret_cast(vec_i32, Int32, 32, Float32)

    # Store to global via store_256b (4 × 256-bit stores)
    # store_256b takes vector<8 x i32>, so reinterpret back and slice
    vec_i32_back = reinterpret_cast(vec_f32, Float32, 32, Int32)
    for chunk in range(4):  # 32 / 8 = 4 chunks
        store_256b(gmem_addr + chunk * 32, subvec(vec_i32_back, chunk * 8, 8))

    # Store back to TMEM
    tcgen05_st_32x32b(32, taddr, vec_i32_back)
"""

__all__ = [
    "reinterpret_cast",
    "store_256b",
    "subvec",
    "tcgen05_cp_128x256b",
    "tcgen05_ld_32x32b",
    "tcgen05_st_32x32b",
    "umma_arrive",
    "umma_arrive_noelect",
]

from cutlass import cute
from cutlass._mlir import ir as _ir_mod
from cutlass._mlir.dialects import (
    nvvm as _nvvm,
)
from cutlass._mlir.dialects import (
    vector as _vector,
)


def _to_ir(val, loc=None, ip=None):
    """Extract raw MLIR IR value from a CuteDSL wrapper."""
    return val.ir_value(loc=loc, ip=ip) if hasattr(val, "ir_value") else val


# ---------------------------------------------------------------------------
# tcgen05.ld.sync.aligned.32x32b.xN.b32  (via nvvm.tcgen05.ld)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05_ld_32x32b(num: int, taddr: int):
    """Load *num* × 32-bit values from TMEM → an opaque ``vector<N x i32>``.

    ``num`` must be a **compile-time constant** in {2, 4, 8, 16, 32, 64, 128}.
    Returns a single opaque MLIR vector value (``vector<num x i32>``).

    Use :func:`reinterpret_cast` to reinterpret the element type (zero-cost),
    and :func:`subvec` to slice a contiguous sub-vector.

    Parameters
    ----------
    num : int
        Number of 32-bit registers to load.  Must be a compile-time constant.
    taddr : int
        TMEM address (bits [31:16] = lane, bits [15:0] = column).
    """

    @dsl_user_op
    def _do(addr_val, *, loc=None, ip=None):
        i32_ty = _ir_mod.IntegerType.get_signless(32)
        ptr6_ty = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(ptr6_ty, _to_ir(addr_val, loc, ip), loc=loc, ip=ip)
        vec_i32_ty = _ir_mod.VectorType.get([num], i32_ty)
        return _nvvm.tcgen05_ld(
            res=vec_i32_ty,
            shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
            tmem_addr=tmem_ptr,
            loc=loc,
            ip=ip,
        )

    return _do(Int32(taddr))


# ---------------------------------------------------------------------------
# tcgen05.st.sync.aligned.32x32b.xN.b32  (via nvvm.tcgen05.st)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05_st_32x32b(num: int, taddr: int, vec):
    """Store *num* × 32-bit values from an opaque vector → TMEM.

    ``num`` must be a **compile-time constant** in {2, 4, 8, 16, 32, 64, 128}.

    Parameters
    ----------
    num : int
        Number of 32-bit registers to store.  Must be a compile-time constant.
    taddr : int
        TMEM address (bits [31:16] = lane, bits [15:0] = column).
    vec : opaque vector
        An opaque ``vector<num x i32>`` value (from :func:`tcgen05_ld_32x32b`
        or :func:`reinterpret_cast`).
    """

    @dsl_user_op
    def _do(addr_val, vec_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(ptr6_ty, _to_ir(addr_val, loc, ip), loc=loc, ip=ip)
        _nvvm.tcgen05_st(
            shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
            tmem_addr=tmem_ptr,
            val=_to_ir(vec_val, loc, ip),
            loc=loc,
            ip=ip,
        )

    _do(Int32(taddr), vec)


# ---------------------------------------------------------------------------
# reinterpret_cast  (zero-cost vector.bitcast)
# ---------------------------------------------------------------------------


@cute.jit
def reinterpret_cast(vec, src_type, src_num, tgt_type):
    """Zero-cost reinterpret of a vector's element type (single ``vector.bitcast``).

    Analogous to C++ ``reinterpret_cast``: no instructions emitted, just
    re-labels the bits.  The total bit-width is preserved:
    ``src_num * src_type.width == tgt_num * tgt_type.width``.

    Parameters
    ----------
    vec : opaque vector
        Source vector (e.g. ``vector<N x i32>`` from :func:`tcgen05_ld_32x32b`).
    src_type : CuTeDSL type
        Element type of *vec* (e.g. ``Int32``).
    src_num : int
        Number of elements in *vec* (compile-time constant).
    tgt_type : CuTeDSL type
        Desired element type (e.g. ``Float32``, ``BFloat16``, ``Float16``).

    Returns
    -------
    opaque vector
        ``vector<M x tgt_type>`` where ``M = src_num * src_type.width // tgt_type.width``.

    Examples
    --------
    ::

        vec_i32  = tcgen05_ld_32x32b(8, taddr)                     # vector<8 x i32>
        vec_f32  = reinterpret_cast(vec_i32, Int32, 8, Float32)    # vector<8 x f32>
        vec_bf16 = reinterpret_cast(vec_i32, Int32, 8, BFloat16)   # vector<16 x bf16>
        vec_back = reinterpret_cast(vec_bf16, BFloat16, 16, Int32) # vector<8 x i32>
    """
    tgt_num = src_num * src_type.width // tgt_type.width

    @dsl_user_op
    def _do(v, *, loc=None, ip=None):
        tgt_vec_ty = _ir_mod.VectorType.get([tgt_num], tgt_type.mlir_type)
        return _vector.bitcast(tgt_vec_ty, _to_ir(v, loc, ip), loc=loc, ip=ip)

    return _do(vec)


# ---------------------------------------------------------------------------
# subvec  (extract a contiguous sub-vector)
# ---------------------------------------------------------------------------


@cute.jit
def subvec(vec, offset, size):
    """Extract a contiguous sub-vector (``vector.extract_strided_slice``).

    Parameters
    ----------
    vec : opaque vector
        Source vector.
    offset : int
        Starting element index (compile-time constant).
    size : int
        Number of elements to extract (compile-time constant).

    Returns
    -------
    opaque vector
        ``vector<size x elem_type>``.
    """

    @dsl_user_op
    def _do(v, *, loc=None, ip=None):
        ir_v = _to_ir(v, loc, ip)
        elem_ty = _ir_mod.VectorType(ir_v.type).element_type
        res_ty = _ir_mod.VectorType.get([size], elem_ty)
        return _vector.extract_strided_slice(
            res_ty,
            ir_v,
            offsets=[offset],
            sizes=[size],
            strides=[1],
            loc=loc,
            ip=ip,
        )

    return _do(vec)


# ---------------------------------------------------------------------------
# st.global.L1::no_allocate.v8.f32  (256-bit direct R2G store)
# ---------------------------------------------------------------------------

_STORE_256B_ASM = "st.global.L1::no_allocate.v8.f32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};"
_STORE_256B_CONSTRAINTS = "l,r,r,r,r,r,r,r,r"


@cute.jit
def store_256b(gmem_ptr, vec):
    """Store 256 bits (8 × 32-bit) to global memory, bypassing L1 allocation.

    Issues ``st.global.L1::no_allocate.v8.f32`` with ``"r"`` (integer register)
    constraints — type-agnostic, just like C++ ``reinterpret_cast<uint32_t*>``.

    Parameters
    ----------
    gmem_ptr : pointer
        Global-memory destination address (must be 32-byte aligned).
    vec : opaque vector
        A ``vector<8 x i32>`` (use :func:`subvec` to slice from a larger vector).
    """

    @dsl_user_op
    def _do(addr, v, *, loc=None, ip=None):
        i32_ty = _ir_mod.IntegerType.get_signless(32)
        ir_v = _to_ir(v, loc, ip)
        elems = [
            llvm.extractelement(
                ir_v,
                position=_arith.constant(i32_ty, i, loc=loc, ip=ip),
                loc=loc,
                ip=ip,
            )
            for i in range(8)
        ]
        operands = [_to_ir(addr, loc, ip)] + elems
        llvm.inline_asm(
            _ir_mod.Type.parse("!llvm.void"),
            operands,
            _STORE_256B_ASM,
            _STORE_256B_CONSTRAINTS,
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )

    _do(gmem_ptr, vec)


# ---------------------------------------------------------------------------
# tcgen05.cp.cta_group::1.128x256b  (via nvvm.tcgen05.cp)
# ---------------------------------------------------------------------------


@cute.jit
def tcgen05_cp_128x256b(taddr: int, smem_desc: Tcgen05SmemDescriptor):
    """Async copy SMEM → TMEM with shape ``128x256b`` (``cta_group::1``).

    Issues ``tcgen05.cp.cta_group::1.128x256b  [taddr], s-desc;``
    via the native ``nvvm.tcgen05.cp`` MLIR op.

    The instruction copies a 128-row × 256-bit tile from shared memory
    (described by *smem_desc*) into Tensor Memory at *taddr*.  The copy
    is **asynchronous** — use ``tcgen05.commit`` + ``mbarrier.wait`` to
    synchronize.

    PTX reference
    -------------
        tcgen05.cp.cta_group::1.128x256b  [taddr], s-desc;

    Parameters
    ----------
    taddr : int
        TMEM destination address (uint32, passed as ``!llvm.ptr<6>``).
    smem_desc : Tcgen05SmemDescriptor
        64-bit SMEM matrix descriptor (same format as ``tcgen05.mma``
        descriptors — see ``Tcgen05SmemDescriptor``).
    """

    @dsl_user_op
    def _do(addr_val, desc_val, *, loc=None, ip=None):
        ptr6_ty = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(ptr6_ty, _to_ir(addr_val, loc, ip), loc=loc, ip=ip)
        _nvvm.tcgen05_cp(
            shape=_nvvm.Tcgen05CpShape.SHAPE_128x256b,
            taddr=tmem_ptr,
            smem_desc=_to_ir(desc_val, loc, ip),
            cta_group=_nvvm.Tcgen05GroupKind.CTA_1,
            loc=loc,
            ip=ip,
        )

    _do(Int32(taddr), smem_desc.desc_i64[0])


@cute.jit
def tcgen05_fence_before():
    """tcgen05.fence::before_thread_sync — non-blocking ordering fence."""
    _nvvm.tcgen05_fence(kind=_nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_fence_after():
    """tcgen05.fence::after_thread_sync — non-blocking ordering fence."""
    _nvvm.tcgen05_fence(kind=_nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)


@cute.jit
def umma_arrive(mbar_ptr: cute.Pointer):
    """tcgen05.commit.cta_group::1.mbarrier::arrive::one — signal MMA done."""
    with elect_one():
        tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)


@cute.jit
def umma_arrive_noelect(mbar_ptr: cute.Pointer):
    """tcgen05.commit.cta_group::1.mbarrier::arrive::one — signal MMA done."""
    tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)


PRINT_DEBUG = False

LN2 = 0.6931471805599453
RCP_LN2 = 1.4426950408889634

COMPILE_OPTIONS = "--enable-tvm-ffi"
USE_FAST_MATH: bool = os.getenv("MSLK_USE_FAST_MATH", "1") == "1"

# Mapping from torch dtype to cutlass dtype (for beta_dtype conversion)
_torch_to_cutlass_dtype = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def require_blackwell_target() -> None:
    """Reject compilation targets that cannot execute this SM100/SM103 kernel."""
    target = get_compile_target()
    if target.device_type != "cuda" or target.capability not in ((10, 0), (10, 3)):
        raise RuntimeError(
            f"chunk_kda_bwd_wy_dqkg_fused requires Blackwell (SM100/SM103), got target={target}"
        )


def make_thread_cooperative_group(size: int):
    return pipeline.CooperativeGroup(pipeline.Agent.Thread, size)


@cute.jit
def _resolve_chunk_route(
    cu_seqlens: cute.Tensor | None,
    chunk_offsets: cute.Tensor | None,
    global_chunk: Int32,
):
    """Resolve one direct dense or device-scheduled ragged chunk."""
    if cutlass.const_expr(chunk_offsets is not None):
        _, _, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            global_chunk,
            Int32(64),
        )
        return token_start, Int32(0), valid_tokens

    return Int32(0), global_chunk, Int32(64)


# Instruction descriptor for M=64, N=64, BF16, dense, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=8 at [17:22], TransposeB at [16],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N64_K_MN = (4 << 24) | (8 << 17) | (1 << 16) | (1 << 10) | (1 << 7) | (1 << 4)

# Instruction descriptor for M=64, N=128, BF16, dense, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=16 at [17:22], TransposeB at [16],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N128_K_MN = (4 << 24) | (16 << 17) | (1 << 16) | (1 << 10) | (1 << 7) | (1 << 4)

# Instruction descriptor for M=64, N=128, BF16, dense
# Bits: M>>4=4 at [24:28], N>>3=16 at [17:22],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N128_K_K = (4 << 24) | (16 << 17) | (1 << 10) | (1 << 7) | (1 << 4)

# Instruction descriptor for M=64, N=128, BF16, dense, TransposeA=1, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=16 at [17:22],
#       TransposeB at [16], TransposeA at [15],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N128_MN_MN = (
    (4 << 24) | (16 << 17) | (1 << 16) | (1 << 15) | (1 << 10) | (1 << 7) | (1 << 4)
)

# Instruction descriptor for M=64, N=64, BF16, dense, TransposeA=1, TransposeB=1
# Bits: M>>4=4 at [24:28], N>>3=8 at [17:22],
#       TransposeB at [16], TransposeA at [15],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N64_MN_MN = (
    (4 << 24) | (8 << 17) | (1 << 16) | (1 << 15) | (1 << 10) | (1 << 7) | (1 << 4)
)

# Instruction descriptor for M=64, N=64, BF16, dense
# Bits: M>>4=4 at [24:28], N>>3=8 at [17:22],
#       TransposeB at [16], TransposeA at [15],
#       btype=bf16(1) at [10:12], atype=bf16(1) at [7:9], dtype=f32(1) at [4:5]
IDESC_F16_M64_N64_K_K = (4 << 24) | (8 << 17) | (1 << 10) | (1 << 7) | (1 << 4)

ELEM_BYTES_BF16 = BFloat16.width // 8


@cute.jit
def smem_load_bf16x8_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """
    Load 8 consecutive bfloat16 from SMEM with Swizzle<3,4,3> layout.
    raw_ptr: BFloat16 SMEM base pointer (NOT recast_ptr — raw buffer start)
    row: row index in [0, T_TILE=64)
    col_base: 8-aligned column index in [0, K_TILE=128)
    Logical layout: [BT=64, BV=128] K-major, with the BV=128 dim split into
    two halves of 64 elements (high half offset by 4096 elements).
    Swizzle<3,4,3> on bf16: phys_elem = elem ^ ((row & 7) << 3) within a half.
    Returns an 8-element rmem fragment (bf16).
    """
    half = col_base >> Int32(6)
    k_inner = col_base & Int32(63)
    swizzled = k_inner ^ ((row & Int32(7)) << Int32(3))
    elem_off = half * Int32(4096) + row * Int32(64) + swizzled
    aligned_ptr = cute.make_ptr(
        BFloat16,
        (raw_ptr + elem_off).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(aligned_ptr, cute.make_layout((8,), stride=(1,)))
    rmem_t = cute.make_fragment_like(smem_t)
    cute.autovec_copy(smem_t, rmem_t)
    return rmem_t


@cute.jit
def smem_store_bf16x8_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32, data: cute.Tensor):
    """
    Store 8 consecutive bfloat16 to SMEM with Swizzle<3,4,3> layout.
    raw_ptr: BFloat16 SMEM base pointer (NOT recast_ptr — raw buffer start)
    row: row index in [0, T_TILE=64)
    col_base: 8-aligned column index in [0, K_TILE=128)
    data: 8-element rmem fragment (bf16) to store.

    NOTE: For the K-major→MN-major dv re-swizzle, source layout
    `(BT,BV) K-major Swizzle<3,4,3>` and destination layout
    `(BV,BT) MN-major Swizzle<3,4,3>` produce **identical** physical
    addresses for the same (row=t, col=v). So this helper uses the same
    address formula as the load helper, and the caller passes (row=t, col=v)
    for both load (src K-maj) and store (dst MN-maj), implicitly transposing.
    """
    half = col_base >> Int32(6)
    k_inner = col_base & Int32(63)
    swizzled = k_inner ^ ((row & Int32(7)) << Int32(3))
    elem_off = half * Int32(4096) + row * Int32(64) + swizzled
    smem_ptr = cute.make_ptr(
        BFloat16,
        (raw_ptr + elem_off).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(smem_ptr, cute.make_layout((8,), stride=(1,)))
    cute.autovec_copy(data, smem_t)


@cute.jit
def smem_load_f32x4_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32):
    """
    Load 4 consecutive float32 from SMEM with K_SW128 layout.
    Logical layout: [BT=64, BK=128] ROW_MAJOR, tiled over a Float32 K_SW128 atom.
    The atom provides a 32-element row stride. The 128-element column is broken
    into 4 blocks of 32 elements.
    PyCutlass tiles this such that outer blocks stride by 2048 elements:
      elem_idx = row * 32 + (col_base % 32) + (col_base / 32) * 2048

    The TMA hardware performs a 128B Swizzle on physical byte addresses:
      byte_idx = elem_idx * 4
      swizzled_byte = byte_idx ^ (((byte_idx >> 7) & 7) << 4)
    Dividing by 4 yields the element XOR offset:
      elem_xor = ((elem_idx >> 5) & 7) << 2
    Because (elem_idx >> 5) simplifies to 'row + (col_outer * 64)',
    the XOR offset simplifies exactly to ((row & 7) << 2).
    This only affects the inner 32-element column block.
    """
    c_inner = col_base & Int32(31)
    c_outer = col_base >> Int32(5)
    swizzled_inner = c_inner ^ ((row & Int32(7)) << Int32(2))

    elem_offset = row * Int32(32) + swizzled_inner + c_outer * Int32(2048)

    aligned_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    t = cute.make_tensor(aligned_ptr, cute.make_layout((4,), stride=(1,)))
    vals = t.load()
    return (vals[0], vals[1], vals[2], vals[3])


@cute.jit
def smem_store_f32x4_sw128(raw_ptr: cute.Pointer, row: Int32, col_base: Int32, data: cute.Tensor):
    """
    Store 4 consecutive float32 to SMEM with K_SW128 layout.
    Inverse of smem_load_f32x4_sw128 — same address formula, write path.
    raw_ptr: Float32 SMEM base pointer (raw buffer start)
    row: row index in [0, BT)
    col_base: 4-aligned column index (multiples of 4)
    data: 4-element rmem fragment (f32) to store.
    """
    c_inner = col_base & Int32(31)
    c_outer = col_base >> Int32(5)
    swizzled_inner = c_inner ^ ((row & Int32(7)) << Int32(2))
    elem_offset = row * Int32(32) + swizzled_inner + c_outer * Int32(2048)
    smem_ptr = cute.make_ptr(
        Float32,
        (raw_ptr + elem_offset).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(smem_ptr, cute.make_layout((4,), stride=(1,)))
    cute.autovec_copy(data, smem_t)


@cute.jit
def mma_ws_ss_m64n128_k_k_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N128_K_K, scale)
            scale = 1


@cute.jit
def mma_ws_ss_m64n128_mn_mn_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N128_MN_MN, scale)
            scale = 1


@cute.jit
def mma_ws_ss_m64n64_k_k_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N64_K_K, scale)
            scale = 1


@cute.jit
def mma_ws_ss_m64n64_mn_mn_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    is_accum: bool = False,
):
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 0 if not is_accum else 1
        for ks in cutlass.range_constexpr(K // 16):
            a_off = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_BF16
            b_off = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_BF16
            desc_a = desc_a_base + a_off
            desc_b = desc_b_base + b_off
            tcgen05mma_ws_ss_f16(desc_a, desc_b, tmem_c, IDESC_F16_M64_N64_MN_MN, scale)
            scale = 1
