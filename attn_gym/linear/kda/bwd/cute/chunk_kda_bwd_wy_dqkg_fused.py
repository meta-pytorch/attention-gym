# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CuTe DSL fused WY / (I-Akk)^-1 chunk-level backward for KDA on SM100 / GB200.
# Adapted from the cuLA TMA/UMMA implementation and wrapped in the MSLK KDA API.
#
# pyre-ignore-all-errors

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute

# ============================================================================
# SM100 primitives used by this kernel
# ============================================================================
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass._mlir.dialects import vector as _vector
from cutlass.cute.arch import elect_one, mbarrier_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import make_umma_smem_desc, smem_descriptor_to_int
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cute.tensor import TensorSSA
from cutlass.cute.typing import BFloat16, Float16, Float32, Int32
from cutlass.cutlass_dsl import dsl_user_op

from attn_gym._backends.cute import run_tunable
from attn_gym._backends.cute.utils import cute_dtype, get_device_properties, requires_int64_abi
from attn_gym.linear._delta_rule.triton.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.constants import is_sm100_kda_capability
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import (
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.utils import cdiv
from attn_gym.warpdrive import (
    Barriers,
    Consumer,
    Local,
    NamedBarrier,
    Producer,
    Ring,
    SmemPlan,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.cache import (
    CompileTarget,
    compile_tvm_ffi,
    detect_compile_target,
    get_compile_target,
    jit_cache,
)
from attn_gym.warpdrive.compat import LayoutEnum

# The retained tcgen05 descriptor and TMEM vector bridges are adapted from cuLA.
# Copyright (c) 2025 ANTGROUP. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class Tcgen05SmemDescriptor:
    """64-bit shared-memory descriptor for tcgen05 MMA."""

    def __init__(self, desc_64: cute.Int64):
        self.desc = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        self.desc_i64 = cute.make_tensor(
            cute.recast_ptr(self.desc.iterator, dtype=cute.Int64),
            (1,),
        )
        self.desc_i64[0] = desc_64

    def __add__(self, byte_offset):
        """Return a descriptor with its start address advanced by a byte offset."""
        result = cute.make_rmem_tensor((2,), dtype=cutlass.Int32)
        result_i64 = cute.make_tensor(
            cute.recast_ptr(result.iterator, dtype=cute.Int64),
            (1,),
        )
        result[0] = self.desc[0] + (byte_offset >> 4)
        result[1] = self.desc[1]
        return Tcgen05SmemDescriptor(result_i64[0])


def _ir(value, loc=None, ip=None):
    """Extract an MLIR value from a CuTeDSL wrapper."""
    return value.ir_value(loc=loc, ip=ip) if hasattr(value, "ir_value") else value


@cute.jit
def tcgen05mma_ws_ss_f16(
    desc_a: Tcgen05SmemDescriptor,
    desc_b: Tcgen05SmemDescriptor,
    tmem_c: int,
    desc_val: int,
    scale_out: int,
):
    """Issue weight-stationary kind::f16 MMA with shared-memory operands."""

    @dsl_user_op
    def _do(c_value, a_value, b_value, idesc_value, scale_value, *, loc=None, ip=None):
        pointer_type = llvm.PointerType.get(address_space=6)
        predicate_type = ir.IntegerType.get_signless(1)
        destination = llvm.inttoptr(pointer_type, _ir(c_value, loc, ip), loc=loc, ip=ip)
        enable_d = _arith.trunci(
            predicate_type,
            _ir(scale_value, loc, ip),
            loc=loc,
            ip=ip,
        )
        # CUTLASS 4.8 renamed these four parameters but preserved their positional order.
        _nvvm.tcgen05_mma_ws(
            _nvvm.Tcgen05MMAKind.F16,
            destination,
            _ir(a_value, loc, ip),
            _ir(b_value, loc, ip),
            idesc=_ir(idesc_value, loc, ip),
            enable_input_d=enable_d,
            collector_b_buffer=None,
            collector_op=None,
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


@cute.jit
def tcgen05_ld_32x32b(num: int, taddr: int):
    """Load an opaque vector of 32-bit values from TMEM."""

    @dsl_user_op
    def _do(address, *, loc=None, ip=None):
        pointer_type = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(pointer_type, _ir(address, loc, ip), loc=loc, ip=ip)
        vector_type = ir.VectorType.get([num], ir.IntegerType.get_signless(32))
        return _nvvm.tcgen05_ld(
            res=vector_type,
            shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
            tmem_addr=tmem_ptr,
            loc=loc,
            ip=ip,
        )

    return _do(Int32(taddr))


@cute.jit
def tcgen05_st_32x32b(taddr: int, values):
    """Store an opaque vector of 32-bit values to TMEM."""

    @dsl_user_op
    def _do(address, vector, *, loc=None, ip=None):
        pointer_type = llvm.PointerType.get(address_space=6)
        tmem_ptr = llvm.inttoptr(pointer_type, _ir(address, loc, ip), loc=loc, ip=ip)
        _nvvm.tcgen05_st(
            shape=_nvvm.Tcgen05LdStShape.SHAPE_32X32B,
            tmem_addr=tmem_ptr,
            val=_ir(vector, loc, ip),
            loc=loc,
            ip=ip,
        )

    _do(Int32(taddr), values)


@cute.jit
def reinterpret_cast(values, source_type, source_count, target_type):
    """Bitcast an opaque vector while preserving its total width."""
    target_count = source_count * source_type.width // target_type.width

    @dsl_user_op
    def _do(vector, *, loc=None, ip=None):
        target_vector_type = ir.VectorType.get([target_count], target_type.mlir_type)
        return _vector.bitcast(
            target_vector_type,
            _ir(vector, loc, ip),
            loc=loc,
            ip=ip,
        )

    return _do(values)


@cute.jit
def subvec(values, offset, size):
    """Extract a contiguous compile-time slice from an opaque vector."""

    @dsl_user_op
    def _do(vector, *, loc=None, ip=None):
        ir_vector = _ir(vector, loc, ip)
        result_type = ir.VectorType.get([size], ir.VectorType(ir_vector.type).element_type)
        return _vector.extract_strided_slice(
            result_type,
            ir_vector,
            offsets=[offset],
            sizes=[size],
            strides=[1],
            loc=loc,
            ip=ip,
        )

    return _do(values)


@cute.jit
def store_256b(gmem_addr, values):
    """Store ``vector<8 x i32>`` to global memory without allocating in L1."""
    gmem_ptr = cute.make_ptr(
        Int32,
        gmem_addr,
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    cute.arch.store(gmem_ptr, values, level1_eviction_priority="evict_no_allocate")


@cute.jit
def tcgen05_fence_before():
    """Order prior TMEM accesses before thread synchronization."""
    _nvvm.tcgen05_fence(kind=_nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC)


@cute.jit
def tcgen05_fence_after():
    """Order subsequent TMEM accesses after thread synchronization."""
    _nvvm.tcgen05_fence(kind=_nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC)


@cute.jit
def umma_arrive(mbar_ptr: cute.Pointer):
    """Signal completion of one warp-group MMA sequence."""
    with elect_one():
        tcgen05.commit(mbar_ptr, cta_group=tcgen05.CtaGroup.ONE)


def require_blackwell_target() -> None:
    """Reject compilation targets that cannot execute this SM100/SM103 kernel."""
    target = get_compile_target()
    if target.device_type != "cuda" or not is_sm100_kda_capability(target.effective_capability):
        raise RuntimeError(
            f"chunk_kda_bwd_wy_dqkg_fused requires Blackwell (SM100/SM103), got target={target}"
        )


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


# ── TMEM column offset constants (cta_group::1, M=64, .ws Layout E) ──
TMEM_DA_ACC_OFF = 0  # [0,32)   32 cols dA fp32 acc; Phase 3 overwrites [0,16)
TMEM_DQ_ACC_OFF = 32  # [32,96)  64 cols dq fp32 acc; Phase 3 result uses [32,64)
TMEM_DK_ACC_OFF = 96  # [96,160) 64 cols dk fp32 acc
TMEM_DW_ACC_OFF = 160  # [160,224] 64 cols dw fp32 acc
TMEM_FLEX_OFF = 224  # [224,256) 32 cols dvb time-shared
TMEM_DKGB_ACC_OFF = 272  # [272,336) 64 cols dkgb fp32 acc
TMEM_DA2_ACC_OFF = 336  # [336,368) 32 cols dA fp32 acc for dA@A and A@dA
TMEM_DQ_SCALED_OFF = 368  # [368,432) 64 cols dq_scaled (stored for dg)
TMEM_TOTAL = 512

# tcgen05 kind::f16 uses type value 0 for FP16 and 1 for BF16 in both operand fields.
IDESC_F16_INPUT_TYPE_MASK = (1 << 10) | (1 << 7)
IDESC_F16_M64_N128_K_K = (4 << 24) | (16 << 17) | (1 << 4)
IDESC_F16_M64_N128_MN_MN = (4 << 24) | (16 << 17) | (1 << 16) | (1 << 15) | (1 << 4)
IDESC_F16_M64_N64_MN_MN = (4 << 24) | (8 << 17) | (1 << 16) | (1 << 15) | (1 << 4)
IDESC_F16_M64_N64_K_K = (4 << 24) | (8 << 17) | (1 << 4)

ELEM_BYTES_F16 = Float16.width // 8


def instruction_descriptor_for_io_dtype(base: int, io_dtype: type[cutlass.Numeric]) -> int:
    """Set the tcgen05 operand type fields while preserving the descriptor shape bits."""
    if io_dtype is BFloat16:
        return base | IDESC_F16_INPUT_TYPE_MASK
    if io_dtype is Float16:
        return base
    raise ValueError(f"unsupported tcgen05 kind::f16 I/O dtype: {io_dtype}")


@cute.jit
def smem_load_f16x8_sw128(
    raw_ptr: cute.Pointer, row: Int32, col_base: Int32, io_dtype: type[cutlass.Numeric]
):
    """
    Load 8 consecutive 16-bit values from SMEM with Swizzle<3,4,3> layout.
    raw_ptr: selected I/O dtype SMEM base pointer (NOT recast_ptr — raw buffer start)
    row: row index in [0, T_TILE=64)
    col_base: 8-aligned column index in [0, K_TILE=128)
    Logical layout: [BT=64, BV=128] K-major, with the BV=128 dim split into
    two halves of 64 elements (high half offset by 4096 elements).
    Swizzle<3,4,3> on 16-bit elements: phys_elem = elem ^ ((row & 7) << 3) within a half.
    Returns an 8-element rmem fragment in the selected I/O dtype.
    """
    half = col_base >> Int32(6)
    k_inner = col_base & Int32(63)
    swizzled = k_inner ^ ((row & Int32(7)) << Int32(3))
    elem_off = half * Int32(4096) + row * Int32(64) + swizzled
    aligned_ptr = cute.make_ptr(
        io_dtype,
        (raw_ptr + elem_off).toint(),
        cute.AddressSpace.smem,
        assumed_align=16,
    )
    smem_t = cute.make_tensor(aligned_ptr, cute.make_layout((8,), stride=(1,)))
    rmem_t = cute.make_fragment_like(smem_t)
    cute.autovec_copy(smem_t, rmem_t)
    return rmem_t


@cute.jit
def smem_store_f16x8_sw128(
    raw_ptr: cute.Pointer,
    row: Int32,
    col_base: Int32,
    data: cute.Tensor,
    io_dtype: type[cutlass.Numeric],
):
    """
    Store 8 consecutive 16-bit values to SMEM with Swizzle<3,4,3> layout.
    raw_ptr: selected I/O dtype SMEM base pointer (NOT recast_ptr — raw buffer start)
    row: row index in [0, T_TILE=64)
    col_base: 8-aligned column index in [0, K_TILE=128)
    data: 8-element rmem fragment in the selected I/O dtype to store.

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
        io_dtype,
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
def mma_ws_ss_call(
    a_smem_layout: cute.Layout,
    desc_a_base: Tcgen05SmemDescriptor,
    b_smem_layout: cute.Layout,
    desc_b_base: Tcgen05SmemDescriptor,
    tmem_c: Int32,
    K: Int32,
    instruction_descriptor: int,
    is_accum: bool = False,
):
    """Issue one weight-stationary MMA sequence over the descriptor K tiles."""
    with elect_one():
        a_outer = a_smem_layout.outer
        b_outer = b_smem_layout.outer
        scale = 1 if is_accum else 0
        for ks in cutlass.range_constexpr(K // 16):
            a_offset = cute.crd2idx(((0, 0), 0, ks, 0), a_outer) * ELEM_BYTES_F16
            b_offset = cute.crd2idx(((0, 0), 0, ks, 0), b_outer) * ELEM_BYTES_F16
            tcgen05mma_ws_ss_f16(
                desc_a_base + a_offset,
                desc_b_base + b_offset,
                tmem_c,
                instruction_descriptor,
                scale,
            )
            scale = 1


@dataclass(frozen=True)
class WyParams:
    """Launch values, layouts and GMEM tensors shared by the WY/dQKG roles."""

    # MMA objects
    vloop_tiled_mma: cute.TiledMma
    dA_vloop_tiled_mma: cute.TiledMma
    dvb_tiled_mma: cute.TiledMma
    # TMA atoms and their GMEM coordinate tensors (G and dG are None with a scalar gate)
    tma_atom_dv: cute.CopyAtom
    tma_tensor_dv: cute.Tensor
    tma_atom_A: cute.CopyAtom
    tma_tensor_A: cute.Tensor
    tma_atom_h: cute.CopyAtom
    tma_tensor_h: cute.Tensor
    tma_atom_dh: cute.CopyAtom
    tma_tensor_dh: cute.Tensor
    tma_atom_do: cute.CopyAtom
    tma_tensor_do: cute.Tensor
    tma_atom_g: cute.CopyAtom | None
    tma_tensor_g: cute.Tensor | None
    tma_atom_v: cute.CopyAtom
    tma_tensor_v: cute.Tensor
    tma_atom_k: cute.CopyAtom
    tma_tensor_k: cute.Tensor
    tma_atom_vnew: cute.CopyAtom
    tma_tensor_vnew: cute.Tensor
    tma_atom_q: cute.CopyAtom
    tma_tensor_q: cute.Tensor
    tma_atom_dg: cute.CopyAtom | None
    tma_tensor_dg: cute.Tensor | None
    # MMA operand SMEM layouts
    vloop_opA_smem: cute.ComposedLayout
    vloop_opB_smem: cute.ComposedLayout
    v_opB_smem: cute.ComposedLayout
    A_mn_opA_smem: cute.ComposedLayout
    dv_mn_opB_smem: cute.ComposedLayout
    dw_k_opA_smem: cute.ComposedLayout
    dw_mn_opB_smem: cute.ComposedLayout
    kg_k_opB_smem: cute.ComposedLayout
    A_k_opB_smem: cute.ComposedLayout
    dA_k_opA_smem: cute.ComposedLayout
    dA_mn_opB_smem: cute.ComposedLayout
    g_epi_smem_layout: cute.ComposedLayout
    k_epi_smem_layout: cute.ComposedLayout
    q_epi_smem_layout: cute.ComposedLayout
    # GMEM tensors
    g_gmem: cute.Tensor
    beta_gmem: cute.Tensor
    dq_gmem: cute.Tensor
    dk_gmem: cute.Tensor
    dv2_gmem: cute.Tensor
    dg_gmem: cute.Tensor
    db_gmem: cute.Tensor
    dA_gmem: cute.Tensor
    # Work: work unit block_idx_x + i * grid_dim_x for i < num_iters
    cu_seqlens: cute.Tensor | None
    chunk_offsets: cute.Tensor | None
    H: Int32
    HV: Int32
    K: Int32
    V: Int32
    num_iters: Int32
    block_idx_x: Int32
    grid_dim_x: Int32


@dataclass(frozen=True)
class WySmem:
    """SMEM views of the plan buffers: staged TMA/MMA views, raw stage bases and scalars."""

    sA: cute.Tensor
    sDv: cute.Tensor
    sH: cute.Tensor
    sDh: cute.Tensor
    sDo: cute.Tensor
    sVnew: cute.Tensor
    sV: cute.Tensor
    sDv_ptr_base: Int32
    sDo_ptr_base: Int32
    sVnew_ptr_base: Int32
    sV_ptr_base: Int32
    sH_ptr_base: Int32
    sDh_ptr_base: Int32
    sA_ptr_base: Int32
    vloop_opA_bytes_per_stage: int
    v_opB_bytes_per_stage: int
    vloop_opB_bytes_per_stage: int
    A_bytes_per_stage: int
    sDv_mn: cute.Tensor
    sDw_mn: cute.Tensor
    sDw_k: cute.Tensor
    sDv_k: cute.Tensor
    sV_k: cute.Tensor
    sA_mn: cute.Tensor
    sDo_k: cute.Tensor
    sVnew_k: cute.Tensor
    sH_k: cute.Tensor
    sDh_k: cute.Tensor
    sKG_k: cute.Tensor
    sA_k: cute.Tensor
    sDA_mn: cute.Tensor
    sDA_k: cute.Tensor
    sG_raw: cute.Tensor
    sG_raw_ptr: cute.Pointer
    sK_raw: cute.Tensor
    sK_raw_ptr: cute.Pointer
    sDw_raw_ptr: cute.Pointer
    sQ_raw: cute.Tensor
    sQ_raw_ptr: cute.Pointer
    sBeta: cute.Tensor
    sDb: cute.Tensor
    sDgk: cute.Tensor
    sGn: cute.Tensor
    sGs: cute.Tensor


class ChunkKdaBwdWyDqkgFused(WarpSpecialized):
    """
    CuTe DSL kernel for chunk_kda_bwd_kernel_wy_dqkg_fused.

    Computes backward gradients dq, dk, dv2, dg, db, dA for the KDA
    chunkwise delta-rule backward pass.

    Roles: ``cuda`` (warps 0-7, two CUDA-core warpgroups), ``mma`` (8), ``load`` (9, TMA) and
    ``aux`` (10-11: beta/scalar-gate loads and the dg store). One work unit is one (chunk,
    value head); every channel takes one epoch per unit, the V-loop rings one per V tile.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim: int = 128,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        scale: float = 1.0,
        use_fast_math: bool = True,
        use_int64_offsets: bool = False,
        scalar_gate: bool = False,
    ):
        assert chunk_size == 64, "chunk_size must be 64"
        assert head_dim == 128, f"head_dim must be 128, got {head_dim}"
        require_blackwell_target()

        self.use_fast_math = use_fast_math
        self.use_int64_offsets = use_int64_offsets
        # Scalar gate: g is one decay per (token, value head) and dg is summed over K.
        self.scalar_gate = scalar_gate
        self.acc_dtype = cutlass.Float32
        self.io_dtype = io_dtype
        self.g_dtype = cutlass.Float32
        self.scale = scale
        self.idesc_m64n128_k_k = instruction_descriptor_for_io_dtype(
            IDESC_F16_M64_N128_K_K, io_dtype
        )
        self.idesc_m64n128_mn_mn = instruction_descriptor_for_io_dtype(
            IDESC_F16_M64_N128_MN_MN, io_dtype
        )
        self.idesc_m64n64_k_k = instruction_descriptor_for_io_dtype(
            IDESC_F16_M64_N64_K_K, io_dtype
        )
        self.idesc_m64n64_mn_mn = instruction_descriptor_for_io_dtype(
            IDESC_F16_M64_N64_MN_MN, io_dtype
        )

        # Tile sizes
        self.BT = chunk_size  # 64
        self.BK = 128  # K tiling for V-loop GEMM (single K tile)
        self.BV = 64  # V tiling for V-loop GEMM (single V tile)

        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        # Number of V tiles
        self.num_v_tiles = (head_dim + self.BV - 1) // self.BV  # 128/64 = 2

        # ── Pipeline stages ──
        # V-loop TMA: 2-stage double buffer
        self.vloop_stage = 2
        self.kloop_stage = 1
        # The G/K/Q loads and the dG store index SMEM stage 0 directly.
        assert self.kloop_stage == 1, "K-loop TMA copies hardcode SMEM stage 0"
        self.a_stage = 2
        self.mma_stage = 1

        # ── MMA tiler shapes ──
        # V-loop GEMMs: [BT, BV] × [BV, BK] → [BT, BK]
        # dq = do @ h :       (BT, BK, BV)  — M=BT, N=BK, K=BV
        # dk = v_new @ dh :   (BT, BK, BV)
        # dw = dv @ h :       (BT, BK, BV)
        self.vloop_gemm_tiler = (self.BT, self.BK, self.BV)

        # V-loop i_k==0 GEMMs: [BT, BV] × [BV, BT] → [BT, BT]
        # dA = dv @ v^T :     (BT, BT, BV)
        self.dA_vloop_tiler = (self.BT, self.BT, self.BV)

        # V-loop i_k==0: A @ dv : [BT, BT] × [BT, BV] → [BT, BV]
        self.dvb_tiler = (self.BT, self.BV, self.BT)

        # K-loop GEMMs:
        # dA += dw @ kg^T :  [BT, BK] × [BK, BT] → [BT, BT]  →  (BT, BT, BK)
        self.kloop_dA_tiler = (self.BT, self.BT, self.BK)
        # dkgb = A @ dw :    [BT, BT] × [BT, BK] → [BT, BK]  →  (BT, BK, BT)
        self.kloop_dkgb_tiler = (self.BT, self.BK, self.BT)

        # dA-post GEMMs:
        # dA @ A :  [BT, BT] × [BT, BT] → [BT, BT]  →  (BT, BT, BT)
        # A @ dA :  same
        self.dApost_tiler = (self.BT, self.BT, self.BT)

        # ── Storage, in declaration (allocation) order ──
        io, f32 = io_dtype, cutlass.Float32
        io_bytes = io.width // 8
        plan = self.smem_plan = SmemPlan()
        a_buf = plan.buffer("A", io, align=1024)  # [BT, BT] x a_stage
        k_buf = plan.buffer("K", io, align=1024)  # [BT, BK]; kg overwrites it
        g_buf = plan.buffer("G", f32, align=1024)  # [BT, BK]; kdk and dg scratch
        q_buf = plan.buffer("Q", io, align=1024)  # [BT, BK]; dA and dA2 overwrite it
        h_buf = plan.buffer("H", io, align=1024)  # [BK, BV] x vloop_stage
        dh_buf = plan.buffer("Dh", io, align=1024)
        do_buf = plan.buffer("Do", io, align=1024)  # [BT, BV] x vloop_stage
        dv_buf = plan.buffer("Dv", io, align=1024)
        vnew_buf = plan.buffer("Vnew", io, align=1024)
        v_buf = plan.buffer("V", io, align=1024)  # [BV, BT] x vloop_stage
        dw_buf = plan.buffer("Dw", io, align=1024)  # [BT, BK]
        for name in ("Beta", "Db", "Gn", "Gs", "Dgk"):
            plan.buffer(name, f32, align=128)

        # ── Channels ──
        vloop_a_bytes = self.BT * self.BV * io_bytes
        self.a = Ring(self.a_stage, Tma(self.BT * self.BT * io_bytes), Umma, buffer=a_buf)
        self.do = Ring(self.vloop_stage, Tma(vloop_a_bytes), Umma, buffer=do_buf)
        self.dv = Ring(self.vloop_stage, Tma(vloop_a_bytes), Umma, buffer=dv_buf)
        self.vnew = Ring(self.vloop_stage, Tma(vloop_a_bytes), Umma, buffer=vnew_buf)
        # The MMA warp and the CUDA-core threads both read h, dh and v: one commit plus every
        # CUDA thread (count 257), the threads fencing their generic loads before the refill.
        joint = {"mma": Umma, "cuda": Threads}
        h_bytes = self.BK * self.BV * io_bytes
        self.h = Ring(self.vloop_stage, Tma(2 * h_bytes), joint, buffer={"H": h_buf, "Dh": dh_buf})
        self.v = Ring(self.vloop_stage, Tma(self.BV * self.BT * io_bytes), joint, buffer=v_buf)
        # G is read and overwritten (kdk, dg) by the CUDA warps and stored by the aux warps. A
        # scalar gate arrives with beta, so its stage only orders that scratch's reuse.
        k_bytes = self.BT * self.BK * io_bytes
        g_fill = Warps if scalar_gate else Tma(self.BT * self.BK * (f32.width // 8))
        self.g = Ring(self.kloop_stage, g_fill, Warps, buffer=g_buf)
        self.k = Ring(self.kloop_stage, Tma(k_bytes), Warps, buffer=k_buf)
        self.q = Ring(self.kloop_stage, Tma(k_bytes), Warps, buffer=q_buf)
        # TMEM accumulators: MMA warp -> CUDA threads, released after their tcgen05.ld.
        acc = {"release_fence": "tmem_load"}
        self.dvb = Ring(self.mma_stage, Umma, Threads, **acc)
        self.dkgb = Ring(self.mma_stage, Umma, Threads, **acc)
        # dq, dw, dk and dA accumulate over both V tiles: the MMA warp holds each accumulator
        # in a span, waiting for it in the first tile and committing it in the last (dA after
        # the K loop). Taking them before the V loop serializes the next unit's MMAs behind the
        # CUDA warps' dk/dA reads, and unrolling that loop spills the MMA warp's hoisted
        # descriptors (78 STL at 56 registers).
        self.dq_acc = Ring(self.mma_stage, Umma, Threads, **acc)
        self.dw_acc = Ring(self.mma_stage, Umma, Threads, **acc)
        self.dk_acc = Ring(self.mma_stage, Umma, Threads, **acc)
        self.dA_acc = Ring(self.mma_stage, Umma, Threads, **acc)
        self.dA2 = Ring(self.mma_stage, Umma, Threads, **acc)
        self.dA3 = Ring(self.mma_stage, Umma, Threads, **acc)
        # CUDA threads write MMA operands: dw into Dw, kg over K, dA and dA2 over Q.
        self.dw_op = Ring(self.kloop_stage, Threads, Umma, buffer=dw_buf)
        self.kg_op = Ring(self.kloop_stage, Threads, Umma, commit_fence="smem_async")
        self.dA_op = Ring(self.mma_stage, Threads, Umma, commit_fence="smem_async")
        self.dA2_op = Ring(self.mma_stage, Threads, Umma, commit_fence="smem_async")
        # beta (and a scalar gate) from the aux warps; dg staged in G for the aux store, which
        # is a TMA store for full chunks.
        self.beta = Ring(1, Threads, Threads)
        self.dg = Ring(self.kloop_stage, Threads, Threads, commit_fence="smem_async")
        # The MMA warp commits and waits here to finish each V tile's MMAs.
        self.vloop_done = Barriers(self.mma_stage, counts=1)
        self.cuda_sync = NamedBarrier(threads=32 * 8)
        # One region over all 512 columns, so the allocation starts at column 0 and the roles
        # address TMEM by the fixed TMEM_*_OFF columns.
        self.tmem_plan = TmemPlan().region("acc", TMEM_TOTAL)

    def _smem(self, params: WyParams) -> WySmem:
        """The SMEM views a role uses, built inside the role so that pointers the other roles
        need are not live across the register-budget change."""
        vloop_opA_smem_no_stage = cute.select(params.vloop_opA_smem, mode=[0, 1, 2])
        vloop_opB_smem_no_stage = cute.select(params.vloop_opB_smem, mode=[0, 1, 2])
        A_mn_opA_smem_no_stage = cute.select(params.A_mn_opA_smem, mode=[0, 1, 2])
        v_opB_smem_no_stage = cute.select(params.v_opB_smem, mode=[0, 1, 2])

        def base(name: str):
            """Address of SMEM buffer ``name``, for the raw swizzled views below."""
            return self.buffer(name).iterator.toint()

        def view(name: str, dtype, layout):
            """``layout``'s swizzle on the iterator over buffer ``name``, as
            ``make_umma_smem_desc`` requires (a swizzle composed on the layout breaks it)."""
            return cute.make_tensor(
                cute.recast_ptr(
                    cute.make_ptr(dtype, base(name), cute.AddressSpace.smem, assumed_align=128),
                    swizzle_=layout.inner,
                    dtype=dtype,
                ),
                layout.outer,
            )

        def raw_ptr(name: str, dtype):
            return cute.make_ptr(dtype, base(name), cute.AddressSpace.smem)

        io, g_dtype = self.io_dtype, self.g_dtype
        return WySmem(
            sA=self.buffer("A"),
            sDv=self.buffer("Dv"),
            sH=self.buffer("H"),
            sDh=self.buffer("Dh"),
            sDo=self.buffer("Do"),
            sVnew=self.buffer("Vnew"),
            sV=self.buffer("V"),
            sDv_ptr_base=base("Dv"),
            sDo_ptr_base=base("Do"),
            sVnew_ptr_base=base("Vnew"),
            sV_ptr_base=base("V"),
            sH_ptr_base=base("H"),
            sDh_ptr_base=base("Dh"),
            sA_ptr_base=base("A"),
            vloop_opA_bytes_per_stage=cute.size_in_bytes(io, vloop_opA_smem_no_stage),
            v_opB_bytes_per_stage=cute.size_in_bytes(io, v_opB_smem_no_stage),
            vloop_opB_bytes_per_stage=cute.size_in_bytes(io, vloop_opB_smem_no_stage),
            A_bytes_per_stage=cute.size_in_bytes(io, A_mn_opA_smem_no_stage),
            sDv_mn=view("Dv", io, params.dv_mn_opB_smem),
            sDw_mn=view("Dw", io, params.dw_mn_opB_smem),
            sDw_k=view("Dw", io, params.dw_k_opA_smem),
            sDv_k=view("Dv", io, params.vloop_opA_smem),
            sV_k=view("V", io, params.v_opB_smem),
            sA_mn=view("A", io, params.A_mn_opA_smem),
            sDo_k=view("Do", io, params.vloop_opA_smem),
            sVnew_k=view("Vnew", io, params.vloop_opA_smem),
            sH_k=view("H", io, params.vloop_opB_smem),
            sDh_k=view("Dh", io, params.vloop_opB_smem),
            sKG_k=view("K", io, params.kg_k_opB_smem),
            sA_k=view("A", io, params.A_k_opB_smem),
            sDA_mn=view("Q", io, params.dA_mn_opB_smem),
            sDA_k=view("Q", io, params.dA_k_opA_smem),
            sG_raw=view("G", g_dtype, params.g_epi_smem_layout),
            sG_raw_ptr=raw_ptr("G", g_dtype),
            sK_raw=view("K", io, params.k_epi_smem_layout),
            sK_raw_ptr=raw_ptr("K", io),
            sDw_raw_ptr=raw_ptr("Dw", io),
            sQ_raw=view("Q", io, params.q_epi_smem_layout),
            sQ_raw_ptr=raw_ptr("Q", io),
            sBeta=self.buffer("Beta"),
            sDb=self.buffer("Db"),
            sDgk=self.buffer("Dgk"),
            sGn=self.buffer("Gn"),
            sGs=self.buffer("Gs"),
        )

    def get_name(self) -> str:
        return "kda_bwd_wy_dqkg"

    @cute.jit
    def _work(self, params: WyParams, wu_iter):
        """(i_t, i_hv, i_h, tok_offset, tile_idx, sub_seq_len) of the role's ``wu_iter``-th
        work unit."""
        work_idx = params.block_idx_x + wu_iter * params.grid_dim_x
        G = params.HV // params.H
        i_t = work_idx // params.HV  # chunk index (global)
        i_hv = work_idx % params.HV  # value-head index
        i_h = i_hv // G  # q/k head index
        tok_offset, tile_idx, sub_seq_len = _resolve_chunk_route(
            params.cu_seqlens,
            params.chunk_offsets,
            Int32(i_t),
        )
        return i_t, i_hv, i_h, tok_offset, tile_idx, sub_seq_len

    def _compute_grid(self, HV, total_nt, max_ctas):
        """Compute grid dimensions for persistent kernel launch.

        Grid: (min(max_ctas, total_tiles), 1, 1), ``max_ctas`` being SMs times the schedule's
        waves. Each CTA handles multiple tiles via stride-by-gridDim.x loop.
        """
        total_tiles = total_nt * HV
        grid_x = cutlass.min(max_ctas, total_tiles)
        return (grid_x, Int32(1), Int32(1))

    @cute.jit
    def upcast(self, value):
        """Promote an address operand before its first overflowing multiply."""
        return cutlass.Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    @cute.jit
    def __call__(
        self,
        # ── Inputs ──
        q_in: cute.Tensor,  # [B, T, H, K] io_dtype
        k_in: cute.Tensor,  # [B, T, H, K] io_dtype
        v_in: cute.Tensor,  # [B, T, HV, V] io_dtype
        v_new_in: cute.Tensor,  # [B, T, HV, V] io_dtype
        g_in: cute.Tensor,  # [B, T, HV, K] fp32, or [B, T, HV] with scalar_gate
        beta_in: cute.Tensor,  # [B, T, HV]   fp32
        A_in: cute.Tensor,  # [B, T, HV, BT] io_dtype
        h_in: cute.Tensor,  # [B, NT, HV, K, V] io_dtype
        do_in: cute.Tensor,  # [B, T, HV, V] io_dtype
        dh_in: cute.Tensor,  # [B, NT, HV, K, V] io_dtype
        dv_in: cute.Tensor,  # [B, T, HV, V] io_dtype
        # ── Outputs ──
        dq_in: cute.Tensor,  # [B, T, HV, K] fp32
        dk_in: cute.Tensor,  # [B, T, HV, K] fp32
        dv2_in: cute.Tensor,  # [B, T, HV, V] io_dtype
        dg_in: cute.Tensor,  # [B, T, HV, K] fp32, or [B, T, HV] with scalar_gate
        db_in: cute.Tensor,  # [B, T, HV]    fp32
        dA_in: cute.Tensor,  # [B, T, HV, BT] fp32
        # ── Metadata ──
        cu_seqlens_in: cute.Tensor | None,  # [N+1] int32
        chunk_offsets_in: cute.Tensor | None,  # [N+1] int32
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32, Int32],  # (B, T, H, HV, K, V)
        total_nt: Int32,
        max_ctas: Int32,
        stream,
    ):
        # ── Extract pointers ──
        q_ptr = q_in.iterator
        k_ptr = k_in.iterator
        v_ptr = v_in.iterator
        v_new_ptr = v_new_in.iterator
        g_ptr = g_in.iterator
        beta_ptr = beta_in.iterator
        A_ptr = A_in.iterator
        h_ptr = h_in.iterator
        do_ptr = do_in.iterator
        dh_ptr = dh_in.iterator
        dv_ptr = dv_in.iterator
        dq_ptr = dq_in.iterator
        dk_ptr = dk_in.iterator
        dv2_ptr = dv2_in.iterator
        dg_ptr = dg_in.iterator
        db_ptr = db_in.iterator
        dA_ptr = dA_in.iterator

        _, T, H, HV, K, V = problem_size
        BT = self.BT

        data_B = Int32(1)
        NT = total_nt

        # ===================== GMEM layouts =====================
        # Token-indexed tensors: (T, dim, (H, data_B))
        def strided_token_layout(tensor: cute.Tensor, dim: Int32, head_count: Int32):
            """Rebuild a [B, T, H, D] input as (T, D, (H, data_B)) from its own strides."""
            return cute.make_layout(
                (T, dim, (head_count, data_B)),
                stride=(
                    tensor.layout.stride[1],
                    tensor.layout.stride[3],
                    (tensor.layout.stride[2], tensor.layout.stride[0]),
                ),
            )

        # q, k, and v may be unbound QKV views, so each keeps its own runtime
        # strides; every other token-indexed tensor is compact.
        q = cute.make_tensor(q_ptr, strided_token_layout(q_in, K, H))
        k = cute.make_tensor(k_ptr, strided_token_layout(k_in, K, H))
        v = cute.make_tensor(v_ptr, strided_token_layout(v_in, V, HV))

        tv_layout = cute.make_layout(
            (T, V, (HV, data_B)),
            stride=(self.upcast(HV * V), 1, (V, self.upcast(T) * HV * V)),
        )
        v_new = cute.make_tensor(v_new_ptr, tv_layout)
        do = cute.make_tensor(do_ptr, tv_layout)
        dv = cute.make_tensor(dv_ptr, tv_layout)
        dv2 = cute.make_tensor(dv2_ptr, tv_layout)

        # beta: (T, (HV, data_B)) fp32
        beta_layout = cute.make_layout(
            (T, (HV, data_B)),
            stride=(self.upcast(HV), (1, self.upcast(T) * HV)),
        )
        beta = cute.make_tensor(beta_ptr, beta_layout)

        # g: (T, K, (HV, data_B)) fp32, or (T, (HV, data_B)) for a scalar gate
        if cutlass.const_expr(self.scalar_gate):
            g = cute.make_tensor(g_ptr, beta_layout)
        else:
            g_layout = cute.make_layout(
                (T, K, (HV, data_B)),
                stride=(self.upcast(HV * K), 1, (K, self.upcast(T) * HV * K)),
            )
            g = cute.make_tensor(g_ptr, g_layout)

        # A: (T, BT, (HV, data_B)) io_dtype
        # NOTE: for A as operand A, A is loaded as transposed view to do MMA
        a_t_layout = cute.make_layout(
            (BT, T, (HV, data_B)),
            stride=(1, self.upcast(HV * BT), (BT, self.upcast(T) * HV * BT)),
        )
        A_T = cute.make_tensor(A_ptr, a_t_layout)

        # dq, dk: (T, K, (HV, data_B)) fp32
        dqk_layout = cute.make_layout(
            (T, K, (HV, data_B)),
            stride=(self.upcast(HV * K), 1, (K, self.upcast(T) * HV * K)),
        )
        dq = cute.make_tensor(dq_ptr, dqk_layout)
        dk = cute.make_tensor(dk_ptr, dqk_layout)

        # dg: (T, K, (HV, data_B)) fp32, or (T, (HV, data_B)) for a scalar gate
        dg = cute.make_tensor(
            dg_ptr, beta_layout if cutlass.const_expr(self.scalar_gate) else dqk_layout
        )

        # db: (T, (HV, data_B)) fp32
        db = cute.make_tensor(db_ptr, beta_layout)

        # dA: (T, BT, (HV, data_B)) fp32
        dA_layout = cute.make_layout(
            (T, BT, (HV, data_B)),
            stride=(self.upcast(HV * BT), 1, (BT, self.upcast(T) * HV * BT)),
        )
        dA_out = cute.make_tensor(dA_ptr, dA_layout)

        h_nt_total = NT

        # h row-major: (K, V, (h_nt_total, HV)) as operand B
        h_layout = cute.make_layout(
            (K, V, (h_nt_total, HV)),
            stride=(V, 1, (self.upcast(HV * K * V), K * V)),
        )
        h = cute.make_tensor(h_ptr, h_layout)
        dh = cute.make_tensor(dh_ptr, h_layout)

        # ===================== MMA setup (4 objects) =====================
        # All use tcgen05.mma.ws (Layout E, M=64, cta_group::1).
        # 1. vloop_tiled_mma: SS K,K (64,128) — dq, dk, dw
        #    dq += do @ h, dk += vnew @ dh, dw += dv @ h
        vloop_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            cute.nvgpu.OperandMajorMode.K,  # A: K-major
            cute.nvgpu.OperandMajorMode.K,  # B: K-major
            self.acc_dtype,
            self.cta_group,
            self.vloop_gemm_tiler[:2],  # (64, 128)
            # default a_source=OperandSource.SMEM → SS mode
        )

        # 2. dA_vloop_tiled_mma: SS K,K (64,64) — dA vloop + kpost_dA
        #    dA += dv @ v^T, dA += dw @ kg^T
        dA_vloop_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.dA_vloop_tiler[:2],  # (64, 64)
            # default a_source=OperandSource.SMEM → SS mode
        )

        # 3. dvb_tiled_mma: SS MN,MN (64,64) — dvb + dkgb
        #    dvb = A @ dv, dkgb = A @ dw
        dvb_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dvb_tiler[:2],  # (64, 64)
        )

        # dkgb_tiled_mma: SS MN,MN (64,128) - dkgb
        dkgb_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.kloop_dkgb_tiler[:2],  # (64, 128)
        )

        # dA_kloop_tiled_mma: SS K,K (64, 64)
        # dA += dw @ kg^T
        dA_kloop_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.kloop_dA_tiler[:2],  # (64, 64)
        )

        # dA2post_tiled_mma: SS K,K (64,64)
        # dA = dA @ A
        dA2post_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.dApost_tiler[:2],  # (64, 64)
        )

        # dA3post_tiled_mma: SS MN,MN (64,64)
        # dA = A @ dA
        dA3post_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dApost_tiler[:2],  # (64, 64)
        )

        # ===================== SMEM layouts =====================
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # SS opA layout: do/vnew/dv [BT,BV]=[64,64] K-major
        vloop_opA_smem = sm100_utils.make_smem_layout_a(
            vloop_tiled_mma,
            self.vloop_gemm_tiler,
            self.io_dtype,
            self.vloop_stage,
        )

        # SS opB layout: h/dh [BK,BV]=[128,64] K-major
        vloop_opB_smem = sm100_utils.make_smem_layout_b(
            vloop_tiled_mma,
            self.vloop_gemm_tiler,
            self.io_dtype,
            self.vloop_stage,
        )

        # SS opB layout: v [BV,BT]=[128,64] K-major (dA vloop)
        v_opB_smem = sm100_utils.make_smem_layout_b(
            dA_vloop_tiled_mma,
            self.dA_vloop_tiler,
            self.io_dtype,
            self.vloop_stage,
        )

        # SS opA layout: A MN-major [BT,BT]=[64,64]
        A_mn_opA_smem = sm100_utils.make_smem_layout_a(
            dvb_tiled_mma,
            self.dvb_tiler,
            self.io_dtype,
            self.a_stage,
        )

        # opB: dv MN-major [BV,BT]=[64,64]
        dv_mn_opB_smem = sm100_utils.make_smem_layout_b(
            dvb_tiled_mma,
            self.dvb_tiler,
            self.io_dtype,
            self.vloop_stage,
        )

        # opA: dw K-major [BT,BK]=[64,128]
        dw_k_opA_smem = sm100_utils.make_smem_layout_a(
            dA_vloop_tiled_mma,
            self.kloop_dA_tiler,
            self.io_dtype,
            self.kloop_stage,
        )

        # opB: dw MN-major [BK,BT]
        dw_mn_opB_smem = sm100_utils.make_smem_layout_b(
            dkgb_tiled_mma,
            self.kloop_dkgb_tiler,
            self.io_dtype,
            self.kloop_stage,
        )

        # opB: kg^T K-major [BT, BK]
        kg_k_opB_smem = sm100_utils.make_smem_layout_b(
            dA_kloop_tiled_mma,
            self.kloop_dA_tiler,
            self.io_dtype,
            self.kloop_stage,
        )

        # opA: dA K-major [BT,BT]
        dA_k_opA_smem = sm100_utils.make_smem_layout_a(
            dA2post_tiled_mma,
            self.dApost_tiler,
            self.io_dtype,
            self.mma_stage,
        )

        # opB: A K-major [BT,BT]
        A_k_opB_smem = sm100_utils.make_smem_layout_b(
            dA2post_tiled_mma,
            self.dApost_tiler,
            self.io_dtype,
            self.a_stage,
        )

        # opB: dA MN-major [BT,BT]
        dA_mn_opB_smem = sm100_utils.make_smem_layout_b(
            dA3post_tiled_mma,
            self.dApost_tiler,
            self.io_dtype,
            self.mma_stage,
        )

        # --- Epilogue (non-MMA) layouts ---
        g_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.g_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.kloop_stage,
        )

        k_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.kloop_stage,
        )

        q_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            1,
        )

        dg_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.g_dtype,
            LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.kloop_stage,
        )

        # ===================== Cluster layout =====================
        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (vloop_tiled_mma.thr_id.shape,),
        )

        # ===================== TMA descriptors =====================
        # Strip stage dimension for TMA atom creation (expects 3 modes, not 4)
        vloop_opA_smem_no_stage = cute.select(vloop_opA_smem, mode=[0, 1, 2])
        vloop_opB_smem_no_stage = cute.select(vloop_opB_smem, mode=[0, 1, 2])
        v_opB_smem_no_stage = cute.select(v_opB_smem, mode=[0, 1, 2])
        A_mn_opA_smem_no_stage = cute.select(A_mn_opA_smem, mode=[0, 1, 2])

        tma_atom_dv, tma_tensor_dv = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            dv,
            vloop_opA_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_A, tma_tensor_A = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            A_T,
            A_mn_opA_smem_no_stage,
            self.dvb_tiler,
            dvb_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_h, tma_tensor_h = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            h,
            vloop_opB_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_dh, tma_tensor_dh = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            dh,
            vloop_opB_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            do,
            vloop_opA_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_vnew, tma_tensor_vnew = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            v_new,
            vloop_opA_smem_no_stage,
            self.vloop_gemm_tiler,
            vloop_tiled_mma,
            cluster_layout.shape,
        )

        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_opB_smem_no_stage,
            self.dA_vloop_tiler,
            dA_vloop_tiled_mma,
            cluster_layout.shape,
        )

        # A scalar gate is loaded by the aux warps and dg is reduced before its store, so
        # neither needs a TMA descriptor; the G tile stays allocated as epilogue scratch.
        g_epi_smem_no_stage = cute.select(g_epi_smem_layout, mode=[0, 1])
        tma_atom_g = tma_tensor_g = tma_atom_dg = tma_tensor_dg = None
        if cutlass.const_expr(not self.scalar_gate):
            tma_atom_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
                tma_load_op,
                g,
                g_epi_smem_no_stage,
                (self.BT, self.BK),
            )

        k_epi_smem_no_stage = cute.select(k_epi_smem_layout, mode=[0, 1])
        tma_atom_k, tma_tensor_k = cpasync.make_tiled_tma_atom(
            tma_load_op,
            k,
            k_epi_smem_no_stage,
            (self.BT, self.BK),
        )

        q_epi_smem_no_stage = cute.select(q_epi_smem_layout, mode=[0, 1])
        tma_atom_q, tma_tensor_q = cpasync.make_tiled_tma_atom(
            tma_load_op,
            q,
            q_epi_smem_no_stage,
            (self.BT, self.BK),
        )

        dg_epi_smem_no_stage = cute.select(dg_epi_smem_layout, mode=[0, 1])
        if cutlass.const_expr(not self.scalar_gate):
            tma_atom_dg, tma_tensor_dg = cpasync.make_tiled_tma_atom(
                tma_store_op,
                dg,
                dg_epi_smem_no_stage,
                (self.BT, self.BK),
            )

        # ===================== TMA byte counts =====================
        # The channels declare each stage's bytes; they must match the TMA boxes.
        tma_boxes = [
            (self.a, A_mn_opA_smem_no_stage, self.io_dtype),
            (self.dv, vloop_opA_smem_no_stage, self.io_dtype),
            (self.do, vloop_opA_smem_no_stage, self.io_dtype),
            (self.vnew, vloop_opA_smem_no_stage, self.io_dtype),
            (self.v, v_opB_smem_no_stage, self.io_dtype),
            (self.k, k_epi_smem_no_stage, self.io_dtype),
            (self.q, q_epi_smem_no_stage, self.io_dtype),
        ]
        if cutlass.const_expr(not self.scalar_gate):
            tma_boxes.append((self.g, g_epi_smem_no_stage, self.g_dtype))
        # h and dh share each stage: one barrier completes on both copies.
        assert (
            2 * cute.size_in_bytes(self.io_dtype, vloop_opB_smem_no_stage)
            == self.h.producer.tx_bytes
        )
        for chan, box, dtype in tma_boxes:
            assert cute.size_in_bytes(dtype, box) == chan.producer.tx_bytes, chan.name

        # ===================== Grid =====================
        grid = self._compute_grid(HV, total_nt, max_ctas)

        # ===================== Launch kernel =====================
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            # MMA objects
            vloop_tiled_mma,
            dA_vloop_tiled_mma,
            dvb_tiled_mma,
            # TMA atoms
            tma_atom_dv,
            tma_tensor_dv,
            tma_atom_A,
            tma_tensor_A,
            tma_atom_h,
            tma_tensor_h,
            tma_atom_dh,
            tma_tensor_dh,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_g,
            tma_tensor_g,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_vnew,
            tma_tensor_vnew,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_dg,
            tma_tensor_dg,
            # SMEM layouts
            vloop_opA_smem,
            vloop_opB_smem,
            v_opB_smem,
            A_mn_opA_smem,
            dv_mn_opB_smem,
            dw_k_opA_smem,
            dw_mn_opB_smem,
            kg_k_opB_smem,
            A_k_opB_smem,
            dA_k_opA_smem,
            dA_mn_opB_smem,
            g_epi_smem_layout,
            k_epi_smem_layout,
            q_epi_smem_layout,
            # GMEM tensors
            g,
            beta,
            dq,
            dk,
            dv2,
            dg,
            db,
            dA_out,
            # Metadata
            cu_seqlens_in,
            chunk_offsets_in,
            problem_size,
            total_nt,
        ).launch(
            grid=grid,
            block=[self.num_threads, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        # MMA objects
        vloop_tiled_mma: cute.TiledMma,
        dA_vloop_tiled_mma: cute.TiledMma,
        dvb_tiled_mma: cute.TiledMma,
        # TMA atoms + tensors
        tma_atom_dv: cute.CopyAtom,
        tma_tensor_dv: cute.Tensor,
        tma_atom_A: cute.CopyAtom,
        tma_tensor_A: cute.Tensor,
        tma_atom_h: cute.CopyAtom,
        tma_tensor_h: cute.Tensor,
        tma_atom_dh: cute.CopyAtom,
        tma_tensor_dh: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        tma_tensor_do: cute.Tensor,
        tma_atom_g: cute.CopyAtom | None,
        tma_tensor_g: cute.Tensor | None,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_vnew: cute.CopyAtom,
        tma_tensor_vnew: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_dg: cute.CopyAtom | None,
        tma_tensor_dg: cute.Tensor | None,
        # SMEM layouts
        vloop_opA_smem: cute.ComposedLayout,
        vloop_opB_smem: cute.ComposedLayout,
        v_opB_smem: cute.ComposedLayout,
        A_mn_opA_smem: cute.ComposedLayout,
        dv_mn_opB_smem: cute.ComposedLayout,
        dw_k_opA_smem: cute.ComposedLayout,
        dw_mn_opB_smem: cute.ComposedLayout,
        kg_k_opB_smem: cute.ComposedLayout,
        A_k_opB_smem: cute.ComposedLayout,
        dA_k_opA_smem: cute.ComposedLayout,
        dA_mn_opB_smem: cute.ComposedLayout,
        g_epi_smem_layout: cute.ComposedLayout,
        k_epi_smem_layout: cute.ComposedLayout,
        q_epi_smem_layout: cute.ComposedLayout,
        # GMEM tensors
        g_gmem: cute.Tensor,
        beta_gmem: cute.Tensor,
        dq_gmem: cute.Tensor,
        dk_gmem: cute.Tensor,
        dv2_gmem: cute.Tensor,
        dg_gmem: cute.Tensor,
        db_gmem: cute.Tensor,
        dA_gmem: cute.Tensor,
        # Metadata
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32, Int32],  # (B, T, H, HV, K, V)
        capacity: Int32,
    ):
        BT, BK = self.BT, self.BK
        self.begin_roles(
            layouts={
                "A": A_mn_opA_smem,
                "K": k_epi_smem_layout,
                "G": g_epi_smem_layout,
                "Q": q_epi_smem_layout,
                "H": vloop_opB_smem,
                "Dh": vloop_opB_smem,
                "Do": vloop_opA_smem,
                "Dv": vloop_opA_smem,
                "Vnew": vloop_opA_smem,
                "V": v_opB_smem,
                "Dw": dw_k_opA_smem,
                "Beta": cute.make_layout((BT,), stride=(1,)),
                # (BT, 2): one column per warpgroup, each contiguous for the final reduce.
                "Db": cute.make_layout((BT, 2), stride=(1, BT)),
                "Gn": cute.make_layout((BK,), stride=(1,)),
                # Per-row scalar gate, published with beta when scalar_gate is set.
                "Gs": cute.make_layout((BT,), stride=(1,)),
                "Dgk": cute.make_layout((BK,), stride=(1,)),
            }
        )
        _B, _T, H, HV, K, V = problem_size

        # ===================== Persistent work decode =====================
        # Grid: (min(max_ctas, total_tiles), 1, 1) — persistent
        block_idx_x = cute.arch.block_idx()[0]
        grid_dim_x = cute.arch.grid_dim()[0]
        if cutlass.const_expr(chunk_offsets is not None):
            active_chunks = load_ragged_chunk_count(chunk_offsets)
        else:
            active_chunks = capacity
        total_work_units = active_chunks * HV
        num_iters = (total_work_units - block_idx_x + grid_dim_x - 1) // grid_dim_x

        params = WyParams(
            vloop_tiled_mma=vloop_tiled_mma,
            dA_vloop_tiled_mma=dA_vloop_tiled_mma,
            dvb_tiled_mma=dvb_tiled_mma,
            tma_atom_dv=tma_atom_dv,
            tma_tensor_dv=tma_tensor_dv,
            tma_atom_A=tma_atom_A,
            tma_tensor_A=tma_tensor_A,
            tma_atom_h=tma_atom_h,
            tma_tensor_h=tma_tensor_h,
            tma_atom_dh=tma_atom_dh,
            tma_tensor_dh=tma_tensor_dh,
            tma_atom_do=tma_atom_do,
            tma_tensor_do=tma_tensor_do,
            tma_atom_g=tma_atom_g,
            tma_tensor_g=tma_tensor_g,
            tma_atom_v=tma_atom_v,
            tma_tensor_v=tma_tensor_v,
            tma_atom_k=tma_atom_k,
            tma_tensor_k=tma_tensor_k,
            tma_atom_vnew=tma_atom_vnew,
            tma_tensor_vnew=tma_tensor_vnew,
            tma_atom_q=tma_atom_q,
            tma_tensor_q=tma_tensor_q,
            tma_atom_dg=tma_atom_dg,
            tma_tensor_dg=tma_tensor_dg,
            vloop_opA_smem=vloop_opA_smem,
            vloop_opB_smem=vloop_opB_smem,
            v_opB_smem=v_opB_smem,
            A_mn_opA_smem=A_mn_opA_smem,
            dv_mn_opB_smem=dv_mn_opB_smem,
            dw_k_opA_smem=dw_k_opA_smem,
            dw_mn_opB_smem=dw_mn_opB_smem,
            kg_k_opB_smem=kg_k_opB_smem,
            A_k_opB_smem=A_k_opB_smem,
            dA_k_opA_smem=dA_k_opA_smem,
            dA_mn_opB_smem=dA_mn_opB_smem,
            g_epi_smem_layout=g_epi_smem_layout,
            k_epi_smem_layout=k_epi_smem_layout,
            q_epi_smem_layout=q_epi_smem_layout,
            g_gmem=g_gmem,
            beta_gmem=beta_gmem,
            dq_gmem=dq_gmem,
            dk_gmem=dk_gmem,
            dv2_gmem=dv2_gmem,
            dg_gmem=dg_gmem,
            db_gmem=db_gmem,
            dA_gmem=dA_gmem,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            H=H,
            HV=HV,
            K=K,
            V=V,
            num_iters=num_iters,
            block_idx_x=block_idx_x,
            grid_dim_x=grid_dim_x,
        )
        self.run_roles(params)

    @role(warps=range(8), regs=224)
    @cute.jit
    def cuda(
        self,
        params: WyParams,
        *,
        beta: Consumer,
        h: Consumer,
        v: Consumer,
        dvb: Consumer,
        g: Consumer,
        dq_acc: Consumer,
        dw_acc: Consumer,
        dw_op: Producer,
        k: Consumer,
        kg_op: Producer,
        dkgb: Consumer,
        dk_acc: Consumer,
        q: Consumer,
        dg: Producer,
        dA_acc: Consumer,
        dA_op: Producer,
        dA2: Consumer,
        dA2_op: Producer,
        dA3: Consumer,
        tmem: Tmem,
        me: Local,
    ):
        """Two CUDA-core warpgroups: TMEM epilogues, the dg/db reductions and the operands
        the MMA warp reads back (dw, kg, dA, dA2). TMEM is addressed by fixed columns."""
        smem = self._smem(params)
        HV, K, V = params.HV, params.K, params.V
        BT = self.BT
        sBeta, sDb, sDgk, sGn, sGs = smem.sBeta, smem.sDb, smem.sDgk, smem.sGn, smem.sGs
        sG_raw, sG_raw_ptr = smem.sG_raw, smem.sG_raw_ptr
        sK_raw_ptr, sDw_raw_ptr, sQ_raw_ptr = (
            smem.sK_raw_ptr,
            smem.sDw_raw_ptr,
            smem.sQ_raw_ptr,
        )
        dv2_gmem, dq_gmem, dk_gmem = params.dv2_gmem, params.dq_gmem, params.dk_gmem
        db_gmem, dA_gmem = params.db_gmem, params.dA_gmem
        tidx = me.tid
        lane_idx = tidx % 32

        wg_idx = tidx // 128
        local_tidx = tidx % 128
        warp_id = local_tidx // 32
        warp_row_tile = warp_id % 2
        warp_col_tile = warp_id // 2
        row = warp_row_tile * 32 + lane_idx  # BT1
        bk_num_cols = self.BK // 2
        bv_num_cols = self.BV // 2
        bk_num_cols_per_wg = bk_num_cols // 2
        bv_num_cols_per_wg = bv_num_cols // 2
        bt_num_cols_per_wg = self.BT // 4
        # ref: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e
        bv_col_base = warp_col_tile * (self.BV // 2) + wg_idx * bv_num_cols_per_wg
        bk_col_base = warp_col_tile * (self.BK // 2) + wg_idx * bk_num_cols_per_wg
        bt_col_base = warp_col_tile * (self.BT // 2) + wg_idx * bt_num_cols_per_wg
        # 8 fp32 store each time for store_256b
        num_stores_f32 = bk_num_cols_per_wg // 8

        for wu_iter in cutlass.range(0, params.num_iters, unroll=0):
            _, i_hv, _, tok_offset, tile_idx, sub_seq_len = self._work(params, wu_iter)

            # NOTE: must sync before next wu_iter's `sDgk[local_tidx] = 0`
            # init, otherwise WG0 of next iter may overwrite sDgk while
            # WG1 of this iter (row == sub_seq_len - 1 lane) is still
            # reading sDgk[col] above. This was the source of the
            # non-deterministic dg accuracy bug.
            self.cuda_sync.sync(aligned=True)
            # fill db, dgk to 0. Each wg zeroes its own sDb column.
            if local_tidx < self.BT:
                sDb[local_tidx, 0] = Float32(0.0)
                sDb[local_tidx, 1] = Float32(0.0)
            if local_tidx < self.BK:
                sDgk[local_tidx] = Float32(0.0)
            self.cuda_sync.sync(aligned=True)

            beta_h = beta.wait()
            cute.arch.fence_proxy("async.shared", space="cta")

            beta_val = sBeta[(row,)]
            db_val = Float32(0.0)
            for v_iter in cutlass.range(self.num_v_tiles):
                # dgk += sum(h * dh, axis=0)
                h_h = h.wait()

                sH_raw_ptr = cute.make_ptr(
                    self.io_dtype,
                    smem.sH_ptr_base + h_h.index * smem.vloop_opB_bytes_per_stage,
                    cute.AddressSpace.smem,
                )
                sDh_raw_ptr = cute.make_ptr(
                    self.io_dtype,
                    smem.sDh_ptr_base + h_h.index * smem.vloop_opB_bytes_per_stage,
                    cute.AddressSpace.smem,
                )
                # each thread in one WG processes one row
                self.cuda_sync.sync(aligned=True)
                if wg_idx == 0:
                    for i in cutlass.range_constexpr(self.BV // 8):
                        col = i * 8
                        h_vals = smem_load_f16x8_sw128(sH_raw_ptr, local_tidx, col, self.io_dtype)
                        dh_vals = smem_load_f16x8_sw128(
                            sDh_raw_ptr, local_tidx, col, self.io_dtype
                        )
                        h_dh_vals = cute.make_rmem_tensor((8,), Float32)
                        h_dh_vals.store(h_vals.load().to(Float32) * dh_vals.load().to(Float32))
                        for j in cutlass.range_constexpr(8):
                            sDgk[(local_tidx,)] += h_dh_vals[j]

                h_h.release()

                dvb_h = dvb.wait()
                tcgen05_fence_after()
                dvb_i32 = tcgen05_ld_32x32b(
                    bv_num_cols_per_wg, TMEM_FLEX_OFF + wg_idx * bv_num_cols_per_wg
                )
                tcgen05_fence_before()
                dvb_h.release()

                dvb_f32 = reinterpret_cast(dvb_i32, Int32, bv_num_cols_per_wg, Float32)
                dvb_f32_val = TensorSSA(dvb_f32, (bv_num_cols_per_wg,), Float32)

                # db += sum(dvb * v, axis=1)
                v_h = v.wait()
                rV_io = cute.make_rmem_tensor((bv_num_cols_per_wg,), self.io_dtype)
                sV_raw_ptr_cur = cute.make_ptr(
                    self.io_dtype,
                    smem.sV_ptr_base + v_h.index * smem.v_opB_bytes_per_stage,
                    cute.AddressSpace.smem,
                )
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(bv_num_cols_per_wg // 8):
                        col_base = bv_col_base + i * 8
                        vals = smem_load_f16x8_sw128(sV_raw_ptr_cur, row, col_base, self.io_dtype)
                        rV_io[i * 8 + 0] = vals[0]
                        rV_io[i * 8 + 1] = vals[1]
                        rV_io[i * 8 + 2] = vals[2]
                        rV_io[i * 8 + 3] = vals[3]
                        rV_io[i * 8 + 4] = vals[4]
                        rV_io[i * 8 + 5] = vals[5]
                        rV_io[i * 8 + 6] = vals[6]
                        rV_io[i * 8 + 7] = vals[7]
                else:
                    rV_io.fill(self.io_dtype(0.0))
                rV_fp32 = cute.make_rmem_tensor((bv_num_cols_per_wg,), Float32)
                rV_fp32.store(rV_io.load().to(Float32))
                rV_fp32.store(rV_fp32.load() * dvb_f32_val)
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(bv_num_cols_per_wg):
                        db_val += rV_fp32[i]

                v_h.release()

                # ── dv2 epilogue: dv2 = dvb * beta, cast to the I/O dtype, store to gmem ──
                dvb_f32_rmem = cute.make_rmem_tensor((bv_num_cols_per_wg,), Float32)
                dvb_f32_rmem.store(dvb_f32_val * beta_val)

                dvb_io_rmem = cute.make_rmem_tensor((bv_num_cols_per_wg,), self.io_dtype)
                dvb_io_rmem.store(dvb_f32_rmem.load().to(self.io_dtype))

                # 16-bit I/O vector → i32 vector for store_256b (8 i32 = 16 I/O values = 32 bytes per store).
                dvb_io_val = dvb_io_rmem.load()
                dvb_i32_vec = reinterpret_cast(
                    dvb_io_val, self.io_dtype, bv_num_cols_per_wg, Int32
                )
                # bv_num_cols I/O values = bv_num_cols // 16 stores of 256b each.
                num_stores_per_row = bv_num_cols_per_wg // 16  # = 4 for BV=128

                base_addr = (
                    dv2_gmem.iterator
                    + self.upcast(tok_offset + tile_idx * self.BT + row) * HV * V
                    + i_hv * V
                    + v_iter * self.BV
                    + bv_col_base
                ).toint()
                if row < sub_seq_len:
                    for s in cutlass.range_constexpr(num_stores_per_row):
                        chunk = subvec(dvb_i32_vec, s * 8, 8)
                        store_256b(base_addr + s * 32, chunk)

            # gk_exp = exp2(g)
            g_h = g.wait()
            # write to gn
            if cutlass.const_expr(self.scalar_gate):
                sGn[local_tidx] = sGs[(sub_seq_len - 1,)]
            else:
                sGn[local_tidx] = sG_raw[(sub_seq_len - 1, local_tidx, 0)]

            # row-major load, match TMEM layout
            rG = cute.make_rmem_tensor((self.BK // 4,), self.g_dtype)
            if cutlass.const_expr(self.scalar_gate):
                # sGs is zero past sub_seq_len, matching the vector-gate fill below.
                rG.fill(sGs[(row,)])
            elif row < sub_seq_len:
                for i in cutlass.range_constexpr(self.BK // 4 // 4):
                    col_base = bk_col_base + i * 4
                    vals = smem_load_f32x4_sw128(sG_raw_ptr, row, col_base)
                    rG[i * 4 + 0] = vals[0]
                    rG[i * 4 + 1] = vals[1]
                    rG[i * 4 + 2] = vals[2]
                    rG[i * 4 + 3] = vals[3]
            else:
                rG.fill(Float32(0.0))
            rG_val = rG.load()
            rG_exp_val = cute.exp2(rG_val, fastmath=self.use_fast_math)

            # wait for dq, dq=dq*gk_exp*scale, GMEM store
            dq_h = dq_acc.wait()
            tcgen05_fence_after()
            dq_i32 = tcgen05_ld_32x32b(
                bk_num_cols_per_wg, TMEM_DQ_ACC_OFF + wg_idx * bk_num_cols_per_wg
            )
            tcgen05_fence_before()
            dq_h.release()

            dq_f32 = reinterpret_cast(dq_i32, Int32, bk_num_cols_per_wg, Float32)
            dq_f32_val = TensorSSA(dq_f32, (bk_num_cols_per_wg,), Float32)

            rDq = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
            rDq.store(dq_f32_val * rG_exp_val * Float32(self.scale))

            dq_f32_val_store = rDq.load()
            dq_i32_vec = reinterpret_cast(dq_f32_val_store, Float32, bk_num_cols_per_wg, Int32)
            # store to TMEM first to reduce register usage
            tcgen05_st_32x32b(TMEM_DQ_SCALED_OFF + wg_idx * bk_num_cols_per_wg, dq_i32_vec)
            cute.arch.fence_view_async_tmem_store()
            dq_base_addr = (
                dq_gmem.iterator
                + self.upcast(tok_offset + tile_idx * self.BT + row) * HV * K
                + i_hv * K
                + bk_col_base
            ).toint()
            if row < sub_seq_len:
                for s in cutlass.range_constexpr(num_stores_f32):
                    chunk = subvec(dq_i32_vec, s * 8, 8)
                    store_256b(dq_base_addr + s * 32, chunk)

            # wait for dw
            dw_h = dw_acc.wait()
            tcgen05_fence_after()
            dw_i32 = tcgen05_ld_32x32b(
                bk_num_cols_per_wg, TMEM_DW_ACC_OFF + wg_idx * bk_num_cols_per_wg
            )
            tcgen05_fence_before()
            dw_h.release()

            # dw = -dw, convert to the I/O dtype, write to smem
            dw_f32 = reinterpret_cast(dw_i32, Int32, bk_num_cols_per_wg, Float32)
            dw_f32_val = TensorSSA(dw_f32, (bk_num_cols_per_wg,), Float32)

            dw_io_rmem = cute.make_rmem_tensor((bk_num_cols_per_wg,), self.io_dtype)
            if row < sub_seq_len:
                dw_io_rmem.store((-dw_f32_val).to(self.io_dtype))
            else:
                dw_io_rmem.fill(self.io_dtype(0.0))

            with dw_op.produce():
                # store 8 I/O values each time
                dw_smem_num_stores = bk_num_cols_per_wg // 8
                for i in cutlass.range_constexpr(dw_smem_num_stores):
                    col_base = bk_col_base + i * 8
                    chunk = cute.local_tile(dw_io_rmem, (8,), (i,))
                    smem_store_f16x8_sw128(sDw_raw_ptr, row, col_base, chunk, self.io_dtype)

            k_h = k.wait()
            # compute kg = k * gk_exp
            rK = cute.make_rmem_tensor((self.BK // 4,), self.io_dtype)
            if row < sub_seq_len:
                for i in cutlass.range_constexpr(self.BK // 4 // 8):
                    col_base = bk_col_base + i * 8
                    vals = smem_load_f16x8_sw128(sK_raw_ptr, row, col_base, self.io_dtype)
                    rK[i * 8 + 0] = vals[0]
                    rK[i * 8 + 1] = vals[1]
                    rK[i * 8 + 2] = vals[2]
                    rK[i * 8 + 3] = vals[3]
                    rK[i * 8 + 4] = vals[4]
                    rK[i * 8 + 5] = vals[5]
                    rK[i * 8 + 6] = vals[6]
                    rK[i * 8 + 7] = vals[7]
            else:
                rK.fill(self.io_dtype(0.0))
            rK_fp32 = cute.make_rmem_tensor((self.BK // 4,), Float32)
            rK_fp32.store(rK.load().to(Float32))
            rK_fp32_val = rK_fp32.load()
            rKG_val = rK_fp32_val * rG_exp_val

            # write kg to K smem,
            # notify dA += dw @ kg^T
            rKG_io = cute.make_rmem_tensor((self.BK // 4,), self.io_dtype)
            rKG_io.store(rKG_val.to(self.io_dtype))

            with kg_op.produce():
                for i in cutlass.range_constexpr(self.BK // 4 // 8):
                    col_base = bk_col_base + i * 8
                    chunk_kg = cute.local_tile(rKG_io, (8,), (i,))
                    smem_store_f16x8_sw128(sK_raw_ptr, row, col_base, chunk_kg, self.io_dtype)

            # wait for dkgb
            dkgb_h = dkgb.wait()
            tcgen05_fence_after()
            dkgb_i32 = tcgen05_ld_32x32b(
                bk_num_cols_per_wg, TMEM_DKGB_ACC_OFF + wg_idx * bk_num_cols_per_wg
            )
            tcgen05_fence_before()
            dkgb_h.release()

            # db += sum(dkgb * kg, axis=1)
            dkgb_f32 = reinterpret_cast(dkgb_i32, Int32, bk_num_cols_per_wg, Float32)
            dkgb_f32_val = TensorSSA(dkgb_f32, (bk_num_cols_per_wg,), Float32)
            rKgb_kg = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
            rKgb_kg.store(dkgb_f32_val * rKG_val)

            if row < sub_seq_len:
                for i in cutlass.range_constexpr(bk_num_cols_per_wg):
                    db_val += rKgb_kg[i]

            # Deterministic db reduction without atomicAdd.
            # 4 partitions per row come from 4 warps (warp_row_tile in {0,1},
            # warp_col_tile in {0,1}) x 2 wgs. Reduce in a fixed order so
            # the result is bitwise reproducible across launches:
            #   Phase 1: warp_col_tile==0 writes its db_val into
            #            sDb[row, wg_idx]   (single writer per slot)
            #   Phase 2: warp_col_tile==1 RMW-adds its db_val into the
            #            same slot          (still single writer per slot)
            #   Phase 3: WG0 sums the 2 wg-slots in fixed order and stores
            #            to GMEM.
            # No race, no atomic, no fp ordering nondeterminism.
            if warp_col_tile == 0 and row < sub_seq_len:
                sDb[row, wg_idx] = db_val
            self.cuda_sync.sync(aligned=True)
            if warp_col_tile == 1 and row < sub_seq_len:
                sDb[row, wg_idx] = sDb[row, wg_idx] + db_val
            self.cuda_sync.sync(aligned=True)
            # store db to GMEM (WG0 only). Sum order is fixed (slot 0 + slot 1).
            if wg_idx == 0 and local_tidx < sub_seq_len:
                db_sum = sDb[(local_tidx, 0)] + sDb[(local_tidx, 1)]
                db_gmem[(tok_offset + tile_idx * self.BT + local_tidx, (i_hv, Int32(0)))] = db_sum

            # dk = dk * exp2(gn[None, :] - g)
            dk_h = dk_acc.wait()
            tcgen05_fence_after()
            dk_i32 = tcgen05_ld_32x32b(
                bk_num_cols_per_wg, TMEM_DK_ACC_OFF + wg_idx * bk_num_cols_per_wg
            )
            tcgen05_fence_before()
            dk_h.release()

            dk_f32 = reinterpret_cast(dk_i32, Int32, bk_num_cols_per_wg, Float32)
            dk_f32_val = TensorSSA(dk_f32, (bk_num_cols_per_wg,), Float32)

            rDk = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
            for i in cutlass.range_constexpr(bk_num_cols_per_wg):
                exp_g_gn = cute.exp2(
                    sGn[(bk_col_base + i,)] - rG_val[i],
                    fastmath=self.use_fast_math,
                )
                rDk[i] = dk_f32_val[i] * exp_g_gn
            if row >= sub_seq_len:
                rDk.fill(Float32(0.0))

            # kdk = k * dk
            rKdk = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
            rKdk.store(rK_fp32.load() * rDk.load())

            # gb = gk_exp * beta[:, None]
            rGb = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
            rGb.store(rG_exp_val * beta_val)

            # dk = dk + dkgb * gb
            rDk.store(rDk.load() + dkgb_f32_val * rGb.load())
            rDk_val = rDk.load()
            dk_i32_vec = reinterpret_cast(rDk_val, Float32, bk_num_cols_per_wg, Int32)
            # GMEM store dk
            # 8 fp32 store each time for store_256b
            dk_base_addr = (
                dk_gmem.iterator
                + self.upcast(tok_offset + tile_idx * self.BT + row) * HV * K
                + i_hv * K
                + bk_col_base
            ).toint()
            if row < sub_seq_len:
                for s in cutlass.range_constexpr(num_stores_f32):
                    chunk_dk = subvec(dk_i32_vec, s * 8, 8)
                    store_256b(dk_base_addr + s * 32, chunk_dk)

            # dgk += sum(kdk, axis=0)
            # write kdk to G SMEM then do BT-dim reduce
            for i in cutlass.range_constexpr(self.BK // 4 // 4):
                col_base = bk_col_base + i * 4
                chunk_kdk = cute.local_tile(rKdk, (4,), (i,))
                smem_store_f32x4_sw128(sG_raw_ptr, row, col_base, chunk_kdk)
            self.cuda_sync.sync(aligned=True)

            # dgk *= exp2(gn)
            if wg_idx == 0:
                sDgk[(local_tidx,)] *= cute.exp2(sGn[(local_tidx,)], fastmath=self.use_fast_math)

            self.cuda_sync.sync(aligned=True)
            if wg_idx == 0:
                sum = Float32(0.0)
                for r in cutlass.range(self.BT, unroll_full=True):
                    sum += sG_raw[(r, local_tidx, 0)]
                sDgk[(local_tidx,)] += sum

            # dg1 = kg * dkgb * beta[:, None], can reuse kg RMEM
            rDg = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
            rDg.store(rKG_val * dkgb_f32_val * beta_val)

            q_h = q.wait()
            # dg2 = q * dq - kdk + dg1
            rQ = cute.make_rmem_tensor((bk_num_cols_per_wg,), self.io_dtype)
            if row < sub_seq_len:
                for i in cutlass.range_constexpr(self.BK // 4 // 8):
                    col_base = bk_col_base + i * 8
                    vals = smem_load_f16x8_sw128(sQ_raw_ptr, row, col_base, self.io_dtype)
                    rQ[i * 8 + 0] = vals[0]
                    rQ[i * 8 + 1] = vals[1]
                    rQ[i * 8 + 2] = vals[2]
                    rQ[i * 8 + 3] = vals[3]
                    rQ[i * 8 + 4] = vals[4]
                    rQ[i * 8 + 5] = vals[5]
                    rQ[i * 8 + 6] = vals[6]
                    rQ[i * 8 + 7] = vals[7]
            else:
                rQ.fill(self.io_dtype(0.0))
            dq_scaled_i32 = tcgen05_ld_32x32b(
                bk_num_cols_per_wg, TMEM_DQ_SCALED_OFF + wg_idx * bk_num_cols_per_wg
            )
            cute.arch.fence_view_async_tmem_load()
            dq_scaled_f32 = reinterpret_cast(dq_scaled_i32, Int32, bk_num_cols_per_wg, Float32)
            dq_scaled_f32_val = TensorSSA(dq_scaled_f32, (bk_num_cols_per_wg,), Float32)
            rDg.store(rQ.load().to(Float32) * dq_scaled_f32_val + rDg.load() - rKdk.load())

            self.cuda_sync.sync(aligned=True)
            # dg = dg2 + m_last * dgk, GMEM store dg
            if row == sub_seq_len - 1:
                for i in cutlass.range_constexpr(bk_num_cols_per_wg):
                    col = bk_col_base + i
                    rDg[i] += sDgk[(col,)]

            # Stage dg to SMEM first. A dedicated store warp later does
            # SMEM -> RMEM -> GMEM with store_256b, keeping GMEM store
            # address/vector live ranges out of the high-register CC path.
            with dg.produce():
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(bk_num_cols_per_wg // 4):
                        col_base = bk_col_base + i * 4
                        chunk_dg = cute.local_tile(rDg, (4,), (i,))
                        smem_store_f32x4_sw128(sG_raw_ptr, row, col_base, chunk_dg)

            g_h.release()

            dA_h = dA_acc.wait()
            tcgen05_fence_after()
            dA_i32 = tcgen05_ld_32x32b(
                bt_num_cols_per_wg, TMEM_DA_ACC_OFF + wg_idx * bt_num_cols_per_wg
            )
            tcgen05_fence_before()
            dA_h.release()
            # NOTE: only release k smem after dA finished, because kg reuses k smem in dA += dw @ kg^T
            k_h.release()

            # dA = dA * beta[None, :], apply strict lower-triangular mask.
            # Triton reference multiplies by the column beta (`b_beta[None, :]`)
            # and keeps only `row > col`.
            dA_f32 = reinterpret_cast(dA_i32, Int32, bt_num_cols_per_wg, Float32)
            dA_f32_val = TensorSSA(dA_f32, (bt_num_cols_per_wg,), Float32)
            rDA = cute.make_rmem_tensor((bt_num_cols_per_wg,), self.io_dtype)
            for i in cutlass.range_constexpr(bt_num_cols_per_wg):
                col = bt_col_base + i
                beta_col = sBeta[(col,)]
                dA_scaled = (dA_f32_val[i] * beta_col).to(self.io_dtype)
                if col < row:
                    rDA[i] = dA_scaled
                else:
                    rDA[i] = self.io_dtype(0.0)
            if row >= sub_seq_len:
                rDA.fill(self.io_dtype(0.0))

            # notify dA2 = dA @ A
            with dA_op.produce():
                for i in cutlass.range_constexpr(bt_num_cols_per_wg // 8):
                    col_base = bt_col_base + i * 8
                    chunk_dA = cute.local_tile(rDA, (8,), (i,))
                    smem_store_f16x8_sw128(sQ_raw_ptr, row, col_base, chunk_dA, self.io_dtype)

            beta_h.release()

            # wait for dA2
            dA2_h = dA2.wait()
            tcgen05_fence_after()
            dA2_i32 = tcgen05_ld_32x32b(
                bt_num_cols_per_wg, TMEM_DA2_ACC_OFF + wg_idx * bt_num_cols_per_wg
            )
            tcgen05_fence_before()
            cute.arch.fence_view_async_tmem_load()

            # write dA2 to smem notify dA2 = A @ dA2
            with dA2_op.produce():
                dA2_f32 = reinterpret_cast(dA2_i32, Int32, bt_num_cols_per_wg, Float32)
                dA2_f32_val = TensorSSA(dA2_f32, (bt_num_cols_per_wg,), Float32)
                rDA2 = cute.make_rmem_tensor((bt_num_cols_per_wg,), self.io_dtype)
                if row < sub_seq_len:
                    rDA2.store(dA2_f32_val.to(self.io_dtype))
                else:
                    rDA2.fill(self.io_dtype(0.0))
                for i in cutlass.range_constexpr(bt_num_cols_per_wg // 8):
                    col_base = bt_col_base + i * 8
                    chunk_dA2 = cute.local_tile(rDA2, (8,), (i,))
                    smem_store_f16x8_sw128(sQ_raw_ptr, row, col_base, chunk_dA2, self.io_dtype)

            # wait for dA2
            dA3_h = dA3.wait()
            tcgen05_fence_after()
            dA3_i32 = tcgen05_ld_32x32b(
                bt_num_cols_per_wg, TMEM_DA2_ACC_OFF + wg_idx * bt_num_cols_per_wg
            )
            tcgen05_fence_before()

            # release mma dA2 after dA3 is finished, protect DA2 TMEM
            dA2_h.release()
            dA3_h.release()
            # NOTE: release smem Q because we reuse to store input-dtype dA
            q_h.release()

            # dA = -dA, apply strict lower-triangular mask
            dA3_f32 = reinterpret_cast(dA3_i32, Int32, bt_num_cols_per_wg, Float32)
            dA3_f32_val = TensorSSA(dA3_f32, (bt_num_cols_per_wg,), Float32)
            rDA3 = cute.make_rmem_tensor((bt_num_cols_per_wg,), Float32)
            rDA3.store(-dA3_f32_val)
            for i in cutlass.range_constexpr(bt_num_cols_per_wg):
                col = bt_col_base + i
                if col >= row:
                    rDA3[i] = Float32(0.0)
            rDA3_val = rDA3.load()
            dA3_i32_vec = reinterpret_cast(rDA3_val, Float32, bt_num_cols_per_wg, Int32)
            # GMEM store dA
            num_stores_dA = bt_num_cols_per_wg // 8
            dA_base_addr = (
                dA_gmem.iterator
                + self.upcast(tok_offset + tile_idx * self.BT + row) * HV * BT
                + i_hv * BT
                + bt_col_base
            ).toint()
            if row < sub_seq_len:
                for s in cutlass.range_constexpr(num_stores_dA):
                    chunk_dA_store = subvec(dA3_i32_vec, s * 8, 8)
                    store_256b(dA_base_addr + s * 32, chunk_dA_store)
        return (
            beta,
            h,
            v,
            dvb,
            g,
            dq_acc,
            dw_acc,
            dw_op,
            k,
            kg_op,
            dkgb,
            dk_acc,
            q,
            dg,
            dA_acc,
            dA_op,
            dA2,
            dA2_op,
            dA3,
        )

    @role(warps=9, regs=56)
    @cute.jit
    def load(
        self,
        params: WyParams,
        *,
        a: Producer,
        h: Producer,
        do: Producer,
        dv: Producer,
        v: Producer,
        vnew: Producer,
        g: Producer,
        k: Producer,
        q: Producer,
    ):
        """TMA loads: A per unit; h, dh, dO, dv, v and v_new per V tile; then G, K and Q."""
        smem = self._smem(params)
        cpasync.prefetch_descriptor(params.tma_atom_A)
        cpasync.prefetch_descriptor(params.tma_atom_dv)
        cpasync.prefetch_descriptor(params.tma_atom_h)
        cpasync.prefetch_descriptor(params.tma_atom_dh)
        cpasync.prefetch_descriptor(params.tma_atom_do)
        cpasync.prefetch_descriptor(params.tma_atom_v)
        cpasync.prefetch_descriptor(params.tma_atom_vnew)
        cpasync.prefetch_descriptor(params.tma_atom_k)
        cpasync.prefetch_descriptor(params.tma_atom_q)
        if cutlass.const_expr(not self.scalar_gate):
            cpasync.prefetch_descriptor(params.tma_atom_g)
            cpasync.prefetch_descriptor(params.tma_atom_dg)

        for wu_iter in cutlass.range(0, params.num_iters, unroll=0):
            i_t, i_hv, i_h, tok_offset, tile_idx, _ = self._work(params, wu_iter)

            # Load A
            tma_A_v = cute.domain_offset((0, tok_offset, (0, 0)), params.tma_tensor_A)
            tAsA, tAgA = self._tma_partition_A(
                params.tma_atom_A,
                tma_A_v,
                smem.sA,
                self.dvb_tiler,  # [BT, BV, BT]
                params.dvb_tiled_mma,
                Int32(0),
                i_hv,
            )
            with a.produce() as s:
                cute.copy(
                    params.tma_atom_A,
                    tAgA[(None, 0, tile_idx)],
                    tAsA[(None, s.index)],
                    tma_bar_ptr=s.barrier,
                )

            # V-loop
            for v_iter in cutlass.range(self.num_v_tiles):
                tma_h_v = cute.domain_offset((0, v_iter * self.BV, (0, 0)), params.tma_tensor_h)
                tHsH, tHgH = self._tma_partition_B(
                    params.tma_atom_h,
                    tma_h_v,
                    smem.sH,
                    self.vloop_gemm_tiler,  # [BT, BK, BV]
                    params.vloop_tiled_mma,
                    i_hv,
                    i_t,
                )
                tma_dh_v = cute.domain_offset((0, v_iter * self.BV, (0, 0)), params.tma_tensor_dh)
                tDHsDH, tDHgDH = self._tma_partition_B(
                    params.tma_atom_dh,
                    tma_dh_v,
                    smem.sDh,
                    self.vloop_gemm_tiler,  # [BT, BK, BV]
                    params.vloop_tiled_mma,
                    i_hv,
                    i_t,
                )
                with h.produce() as s:
                    cute.copy(
                        params.tma_atom_h,
                        tHgH[(None, 0, 0)],
                        tHsH[(None, s.index)],
                        tma_bar_ptr=s.barrier,
                    )
                    cute.copy(
                        params.tma_atom_dh,
                        tDHgDH[(None, 0, 0)],
                        tDHsDH[(None, s.index)],
                        tma_bar_ptr=s.barrier,
                    )

                tma_do_v = cute.domain_offset(
                    (tok_offset, v_iter * self.BV, (0, 0)), params.tma_tensor_do
                )
                tDOsDo, tDOgDo = self._tma_partition_A(
                    params.tma_atom_do,
                    tma_do_v,
                    smem.sDo,
                    self.vloop_gemm_tiler,  # [BT, BK, BV]
                    params.vloop_tiled_mma,
                    Int32(0),
                    i_hv,
                )
                with do.produce() as s:
                    cute.copy(
                        params.tma_atom_do,
                        tDOgDo[(None, tile_idx, 0)],
                        tDOsDo[(None, s.index)],
                        tma_bar_ptr=s.barrier,
                    )

                tma_dv_v = cute.domain_offset(
                    (tok_offset, v_iter * self.BV, (0, 0)), params.tma_tensor_dv
                )
                tDVsDv, tDVgDV = self._tma_partition_A(
                    params.tma_atom_dv,
                    tma_dv_v,
                    smem.sDv,
                    self.vloop_gemm_tiler,  # [BT, BK, BV]
                    params.vloop_tiled_mma,
                    Int32(0),
                    i_hv,
                )
                with dv.produce() as s:
                    cute.copy(
                        params.tma_atom_dv,
                        tDVgDV[(None, tile_idx, 0)],
                        tDVsDv[(None, s.index)],
                        tma_bar_ptr=s.barrier,
                    )

                tma_v_v = cute.domain_offset(
                    (tok_offset, v_iter * self.BV, (0, 0)), params.tma_tensor_v
                )
                tVsV, tVgV = self._tma_partition_B(
                    params.tma_atom_v,
                    tma_v_v,
                    smem.sV,
                    self.dA_vloop_tiler,  # [BT, BT, BV]
                    params.dA_vloop_tiled_mma,
                    Int32(0),
                    i_hv,
                )
                with v.produce() as s:
                    cute.copy(
                        params.tma_atom_v,
                        tVgV[(None, tile_idx, 0)],
                        tVsV[(None, s.index)],
                        tma_bar_ptr=s.barrier,
                    )

                # load v_new
                tma_vnew_v = cute.domain_offset(
                    (tok_offset, v_iter * self.BV, (0, 0)), params.tma_tensor_vnew
                )
                tVnewsVnew, tVnewgVnew = self._tma_partition_A(
                    params.tma_atom_vnew,
                    tma_vnew_v,
                    smem.sVnew,
                    self.vloop_gemm_tiler,  # [BT, BK, BV]
                    params.vloop_tiled_mma,
                    Int32(0),
                    i_hv,
                )
                with vnew.produce() as s:
                    cute.copy(
                        params.tma_atom_vnew,
                        tVnewgVnew[(None, tile_idx, 0)],
                        tVnewsVnew[(None, s.index)],
                        tma_bar_ptr=s.barrier,
                    )

            # Load g. A scalar gate arrives with beta, so this stage only hands the
            # G tile to the consumers once both have released the previous chunk.
            with g.produce() as s:
                if cutlass.const_expr(not self.scalar_gate):
                    tma_g_v = cute.domain_offset((tok_offset, 0, (0, 0)), params.tma_tensor_g)
                    tGsG, tGgG = self._epilog_partition_varlen(
                        params.tma_atom_g,
                        tma_g_v[None, None, (i_hv, Int32(0))],
                        (self.BT, self.BK),
                        smem.sG_raw,
                    )
                    cute.copy(
                        params.tma_atom_g,
                        tGgG[(None, tile_idx, 0)],
                        tGsG[(None, 0)],  # kloop_stage is 1
                        tma_bar_ptr=s.barrier,
                    )

            # Load k
            tma_k_v = cute.domain_offset((tok_offset, 0, (0, 0)), params.tma_tensor_k)
            tKsK, tKgK = self._epilog_partition_varlen(
                params.tma_atom_k,
                tma_k_v[None, None, (i_h, Int32(0))],
                (self.BT, self.BK),
                smem.sK_raw,
            )
            with k.produce() as s:
                cute.copy(
                    params.tma_atom_k,
                    tKgK[(None, tile_idx, 0)],
                    tKsK[(None, 0)],  # kloop_stage is 1
                    tma_bar_ptr=s.barrier,
                )

            tma_q_v = cute.domain_offset((tok_offset, 0, (0, 0)), params.tma_tensor_q)
            tQsQ, tQgQ = self._epilog_partition_varlen(
                params.tma_atom_q,
                tma_q_v[None, None, (i_h, Int32(0))],
                (self.BT, self.BK),
                smem.sQ_raw,
            )
            with q.produce() as s:
                cute.copy(
                    params.tma_atom_q,
                    tQgQ[(None, tile_idx, 0)],
                    tQsQ[(None, 0)],  # kloop_stage is 1
                    tma_bar_ptr=s.barrier,
                )
        return a, h, do, dv, v, vnew, g, k, q

    @role(warps=8, regs=56)
    @cute.jit
    def mma(
        self,
        params: WyParams,
        *,
        a: Consumer,
        h: Consumer,
        do: Consumer,
        dv: Consumer,
        v: Consumer,
        vnew: Consumer,
        dvb: Producer,
        dq_acc: Producer,
        dw_acc: Producer,
        dA_acc: Producer,
        dk_acc: Producer,
        dw_op: Consumer,
        kg_op: Consumer,
        dkgb: Producer,
        dA_op: Consumer,
        dA2: Producer,
        dA2_op: Consumer,
        dA3: Producer,
        vloop_done: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        """Weight-stationary tcgen05 MMAs into fixed TMEM columns; the dq/dw/dk/dA accumulators
        are spans waited for in the first V tile and committed in the last (dA after the K
        loop)."""
        smem = self._smem(params)
        lane_idx = me.lane
        vloop_phase = Int32(0)  # vloop_done completes once per V tile
        for wu_iter in cutlass.range(0, params.num_iters, unroll=0):
            _, _, _, _, _, sub_seq_len = self._work(params, wu_iter)

            zeros8 = cute.make_rmem_tensor((8,), dtype=self.io_dtype)
            zeros8.fill(self.io_dtype(0.0))

            a_h = a.wait()
            sA_raw_ptr = cute.make_ptr(
                self.io_dtype,
                smem.sA_ptr_base + a_h.index * smem.A_bytes_per_stage,
                cute.AddressSpace.smem,
            )
            if sub_seq_len < self.BT:
                for i in cutlass.range_constexpr(self.BT // 32):
                    row = i * 32 + lane_idx
                    for col in cutlass.range_constexpr(self.BT // 8):
                        col_base = col * 8
                        # A participates in full 64x64 MMAs, so neutralize both
                        # invalid rows and invalid columns of a partial chunk.
                        if row >= sub_seq_len or col_base >= sub_seq_len:
                            smem_store_f16x8_sw128(
                                sA_raw_ptr, row, col_base, zeros8, self.io_dtype
                            )
                        elif col_base + 8 > sub_seq_len:
                            values = smem_load_f16x8_sw128(
                                sA_raw_ptr, row, col_base, self.io_dtype
                            )
                            for element in cutlass.range_constexpr(8):
                                if col_base + element >= sub_seq_len:
                                    values[element] = self.io_dtype(0.0)
                            smem_store_f16x8_sw128(
                                sA_raw_ptr, row, col_base, values, self.io_dtype
                            )
                # Make generic-proxy SMEM stores visible to UMMA async-proxy readers.
                cute.arch.fence_proxy("async.shared", space="cta")

            dq_span, dw_span = dq_acc.span(), dw_acc.span()
            dk_span, dA_span = dk_acc.span(), dA_acc.span()
            for v_iter in cutlass.range(self.num_v_tiles):
                is_accum = v_iter != 0
                h_h = h.wait()
                do_h = do.wait()
                sDo_raw_ptr = cute.make_ptr(
                    self.io_dtype,
                    smem.sDo_ptr_base + do_h.index * smem.vloop_opA_bytes_per_stage,
                    cute.AddressSpace.smem,
                )
                if sub_seq_len < self.BT:
                    for i in cutlass.range_constexpr(self.BT // 32):
                        row = i * 32 + lane_idx
                        if row >= sub_seq_len:
                            for col in cutlass.range_constexpr(self.BV // 8):
                                # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                smem_store_f16x8_sw128(
                                    sDo_raw_ptr, row, col * 8, zeros8, self.io_dtype
                                )
                    cute.arch.fence_proxy("async.shared", space="cta")

                dq_span.begin(v_iter)
                # dq+=do@h
                sDo_k_cur = smem.sDo_k[(None, None, None, do_h.index)]
                sH_k_cur = smem.sH_k[(None, None, None, h_h.index)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDo_k_cur.iterator, sDo_k_cur.layout, "k")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sH_k_cur.iterator, sH_k_cur.layout, "k")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    params.vloop_opA_smem,
                    desc_a_base,
                    params.vloop_opB_smem,
                    desc_b_base,
                    TMEM_DQ_ACC_OFF,
                    self.BV,
                    self.idesc_m64n128_k_k,
                    is_accum,
                )

                do_h.release()
                dq_span.commit(v_iter)

                dv_h = dv.wait()
                sDv_raw = cute.make_ptr(
                    self.io_dtype,
                    smem.sDv_ptr_base + dv_h.index * smem.vloop_opA_bytes_per_stage,
                    cute.AddressSpace.smem,
                )
                if sub_seq_len < self.BT:
                    for i in cutlass.range_constexpr(self.BT // 32):
                        row = i * 32 + lane_idx
                        if row >= sub_seq_len:
                            for col in cutlass.range_constexpr(self.BV // 8):
                                # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                smem_store_f16x8_sw128(
                                    sDv_raw, row, col * 8, zeros8, self.io_dtype
                                )
                    cute.arch.fence_proxy("async.shared", space="cta")

                with dvb.produce():
                    sDv_mn_cur = smem.sDv_mn[(None, None, None, dv_h.index)]
                    sA_mn_cur = smem.sA_mn[(None, None, None, a_h.index)]
                    desc_a_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn")
                    )
                    desc_b_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sDv_mn_cur.iterator, sDv_mn_cur.layout, "mn")
                    )
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_call(
                        params.A_mn_opA_smem,
                        desc_a_base,
                        params.dv_mn_opB_smem,
                        desc_b_base,
                        TMEM_FLEX_OFF,
                        self.BT,
                        self.idesc_m64n64_mn_mn,
                    )

                # dw += dv @ h
                dw_span.begin(v_iter)

                sDv_k_cur = smem.sDv_k[(None, None, None, dv_h.index)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDv_k_cur.iterator, sDv_k_cur.layout, "k")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sH_k_cur.iterator, sH_k_cur.layout, "k")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    params.vloop_opA_smem,
                    desc_a_base,
                    params.vloop_opB_smem,
                    desc_b_base,
                    TMEM_DW_ACC_OFF,
                    self.BV,
                    self.idesc_m64n128_k_k,
                    is_accum,
                )

                # dA += dv @ v^T
                v_h = v.wait()
                sV_raw = cute.make_ptr(
                    self.io_dtype,
                    smem.sV_ptr_base + v_h.index * smem.v_opB_bytes_per_stage,
                    cute.AddressSpace.smem,
                )
                if sub_seq_len < self.BT:
                    for i in cutlass.range_constexpr(self.BT // 32):
                        row = i * 32 + lane_idx
                        if row >= sub_seq_len:
                            for col in cutlass.range_constexpr(self.BV // 8):
                                # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                smem_store_f16x8_sw128(sV_raw, row, col * 8, zeros8, self.io_dtype)
                    cute.arch.fence_proxy("async.shared", space="cta")

                dA_span.begin(v_iter)
                sV_k_cur = smem.sV_k[(None, None, None, v_h.index)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDv_k_cur.iterator, sDv_k_cur.layout, "k")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sV_k_cur.iterator, sV_k_cur.layout, "k")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    params.vloop_opA_smem,
                    desc_a_base,
                    params.v_opB_smem,
                    desc_b_base,
                    TMEM_DA_ACC_OFF,
                    self.BV,
                    self.idesc_m64n64_k_k,
                    is_accum,
                )

                # the dv release is a tcgen05.commit covering dv@h and dv@v^T
                dv_h.release()
                dw_span.commit(v_iter)

                v_h.release()

                # dk += v_new @ dh
                vnew_h = vnew.wait()
                sDvnew_raw_ptr = cute.make_ptr(
                    self.io_dtype,
                    smem.sVnew_ptr_base + vnew_h.index * smem.vloop_opA_bytes_per_stage,
                    cute.AddressSpace.smem,
                )
                if sub_seq_len < self.BT:
                    for i in cutlass.range_constexpr(self.BT // 32):
                        row = i * 32 + lane_idx
                        if row >= sub_seq_len:
                            for col in cutlass.range_constexpr(self.BV // 8):
                                # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                smem_store_f16x8_sw128(
                                    sDvnew_raw_ptr, row, col * 8, zeros8, self.io_dtype
                                )
                    cute.arch.fence_proxy("async.shared", space="cta")

                dk_span.begin(v_iter)
                sVnew_k_cur = smem.sVnew_k[(None, None, None, vnew_h.index)]
                sDh_k_cur = smem.sDh_k[(None, None, None, h_h.index)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sVnew_k_cur.iterator, sVnew_k_cur.layout, "k")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDh_k_cur.iterator, sDh_k_cur.layout, "k")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    params.vloop_opA_smem,
                    desc_a_base,
                    params.vloop_opB_smem,
                    desc_b_base,
                    TMEM_DK_ACC_OFF,
                    self.BV,
                    self.idesc_m64n128_k_k,
                    is_accum,
                )

                # the vnew release is a tcgen05.commit
                vnew_h.release()
                dk_span.commit(v_iter)

                h_h.release()

                # add tcgen05.commit and mbar.wait to make sure dq/dk/dw MMA finished. The
                # unit's two V tiles complete the barrier twice, so tile j waits on parity j.
                umma_arrive(vloop_done)
                mbarrier_wait(vloop_done, vloop_phase)
                vloop_phase ^= 1

            dw_s = dw_op.wait()
            cute.arch.fence_proxy("async.shared", space="cta")
            # dkgb = A @ dw
            with dkgb.produce():
                sA_mn_cur = smem.sA_mn[(None, None, None, a_h.index)]
                sDw_mn_cur = smem.sDw_mn[(None, None, None, 0)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDw_mn_cur.iterator, sDw_mn_cur.layout, "mn")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    params.A_mn_opA_smem,
                    desc_a_base,
                    params.dw_mn_opB_smem,
                    desc_b_base,
                    TMEM_DKGB_ACC_OFF,
                    self.BT,
                    self.idesc_m64n128_mn_mn,
                )

            kg_s = kg_op.wait()
            cute.arch.fence_proxy("async.shared", space="cta")
            # dA += dw @ kg^T
            sDw_k_cur = smem.sDw_k[(None, None, None, 0)]
            sKG_k_cur = smem.sKG_k[(None, None, None, 0)]
            desc_a_i64 = smem_descriptor_to_int(
                make_umma_smem_desc(sDw_k_cur.iterator, sDw_k_cur.layout, "k")
            )
            desc_b_i64 = smem_descriptor_to_int(
                make_umma_smem_desc(sKG_k_cur.iterator, sKG_k_cur.layout, "k")
            )
            desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
            desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
            mma_ws_ss_call(
                params.dw_k_opA_smem,
                desc_a_base,
                params.kg_k_opB_smem,
                desc_b_base,
                TMEM_DA_ACC_OFF,
                self.BK,
                self.idesc_m64n64_k_k,
                True,
            )

            dA_span.commit()
            kg_s.release()
            dw_s.release()

            # dA2 = dA @ A
            dA2_h = dA2.acquire()
            dA_s = dA_op.wait()
            cute.arch.fence_proxy("async.shared", space="cta")

            sDA_k_cur = smem.sDA_k[(None, None, None, 0)]
            sA_k_cur = smem.sA_k[(None, None, None, a_h.index)]
            desc_a_i64 = smem_descriptor_to_int(
                make_umma_smem_desc(sDA_k_cur.iterator, sDA_k_cur.layout, "k")
            )
            desc_b_i64 = smem_descriptor_to_int(
                make_umma_smem_desc(sA_k_cur.iterator, sA_k_cur.layout, "k")
            )
            desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
            desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
            mma_ws_ss_call(
                params.dA_k_opA_smem,
                desc_a_base,
                params.A_k_opB_smem,
                desc_b_base,
                TMEM_DA2_ACC_OFF,
                self.BT,
                self.idesc_m64n64_k_k,
            )

            dA2_h.commit()
            dA_s.release()

            # dA3 = A @ dA2
            dA3_h = dA3.acquire()
            dA2_s = dA2_op.wait()
            cute.arch.fence_proxy("async.shared", space="cta")

            sA_mn_cur = smem.sA_mn[(None, None, None, a_h.index)]
            sDA_mn_cur = smem.sDA_mn[(None, None, None, 0)]
            desc_a_i64 = smem_descriptor_to_int(
                make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn")
            )
            desc_b_i64 = smem_descriptor_to_int(
                make_umma_smem_desc(sDA_mn_cur.iterator, sDA_mn_cur.layout, "mn")
            )
            desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
            desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
            mma_ws_ss_call(
                params.A_mn_opA_smem,
                desc_a_base,
                params.dA_mn_opB_smem,
                desc_b_base,
                TMEM_DA2_ACC_OFF,
                self.BT,
                self.idesc_m64n64_mn_mn,
            )

            dA3_h.commit()
            dA2_s.release()

            a_h.release()
        return (
            a,
            h,
            do,
            dv,
            v,
            vnew,
            dvb,
            dq_acc,
            dw_acc,
            dA_acc,
            dk_acc,
            dw_op,
            kg_op,
            dkgb,
            dA_op,
            dA2,
            dA2_op,
            dA3,
        )

    @role(warps=range(10, 12), regs=56)
    @cute.jit
    def aux(self, params: WyParams, *, beta: Producer, g: Consumer, dg: Consumer, me: Local):
        """Load beta (and a scalar gate) per row, then store the dg the CUDA warps staged in G:
        a TMA store for a full chunk, row-predicated vector stores for a tail."""
        smem = self._smem(params)
        HV, K = params.HV, params.K
        sBeta, sGs = smem.sBeta, smem.sGs
        sG_raw, sG_raw_ptr = smem.sG_raw, smem.sG_raw_ptr
        beta_gmem, g_gmem, dg_gmem = params.beta_gmem, params.g_gmem, params.dg_gmem
        tidx = me.tid

        for wu_iter in cutlass.range(0, params.num_iters, unroll=0):
            _, i_hv, _, tok_offset, tile_idx, sub_seq_len = self._work(params, wu_iter)

            with beta.produce():
                beta_f32 = Float32(0.0)
                if tidx < sub_seq_len:
                    beta_f32 = Float32(
                        beta_gmem[(tok_offset + tile_idx * self.BT + tidx, (i_hv, Int32(0)))]
                    )
                sBeta[(tidx,)] = beta_f32
                if cutlass.const_expr(self.scalar_gate):
                    g_f32 = Float32(0.0)
                    if tidx < sub_seq_len:
                        g_f32 = Float32(
                            g_gmem[(tok_offset + tile_idx * self.BT + tidx, (i_hv, Int32(0)))]
                        )
                    sGs[(tidx,)] = g_f32

                cute.arch.fence_proxy("async.shared", space="cta")

            g_h = g.wait()
            dg_h = dg.wait()

            if cutlass.const_expr(self.scalar_gate):
                # Scalar dg is the K-sum of the staged vector dg, one row per thread.
                if tidx < sub_seq_len:
                    dg_row = Float32(0.0)
                    for i in cutlass.range_constexpr(self.BK // 4):
                        vals = smem_load_f32x4_sw128(sG_raw_ptr, tidx, Int32(i * 4))
                        dg_row += vals[0] + vals[1] + vals[2] + vals[3]
                    dg_gmem[(tok_offset + tile_idx * self.BT + tidx, (i_hv, Int32(0)))] = dg_row
            else:
                tma_dg_v = cute.domain_offset((tok_offset, 0, (0, 0)), params.tma_tensor_dg)
                tDGsDG, tDGgDG = self._epilog_partition_varlen(
                    params.tma_atom_dg,
                    tma_dg_v[None, None, (i_hv, Int32(0))],
                    (self.BT, self.BK),
                    sG_raw,
                )
                if sub_seq_len < self.BT:
                    # Tail chunk, direct store
                    store_lane_row = tidx >> Int32(4)  # 0..3
                    store_col_base = (tidx & Int32(15)) * Int32(8)  # 0,8,...,120
                    for row_quad in cutlass.range_constexpr(self.BT // 4):
                        store_row = row_quad * 4 + store_lane_row
                        if store_row < sub_seq_len:
                            vals0 = smem_load_f32x4_sw128(sG_raw_ptr, store_row, store_col_base)
                            vals1 = smem_load_f32x4_sw128(
                                sG_raw_ptr, store_row, store_col_base + Int32(4)
                            )
                            dg_store_rmem = cute.make_rmem_tensor((8,), Float32)
                            for j in cutlass.range_constexpr(4):
                                dg_store_rmem[j] = vals0[j]
                                dg_store_rmem[j + 4] = vals1[j]
                            dg_store_i32_vec = reinterpret_cast(
                                dg_store_rmem.load(), Float32, 8, Int32
                            )
                            dg_base_addr = (
                                dg_gmem.iterator
                                + self.upcast(tok_offset + tile_idx * self.BT + store_row) * HV * K
                                + i_hv * K
                                + store_col_base
                            ).toint()
                            store_256b(dg_base_addr, dg_store_i32_vec)
                else:
                    # Non-tail chunk, TMA store
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.copy(
                        params.tma_atom_dg,
                        tDGsDG[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                        tDGgDG[(None, tile_idx, 0)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            dg_h.release()
            g_h.release()
        return beta, g, dg

    @cute.jit
    def _tma_partition_A(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        """Partition a TMA tensor as MMA A-operand (M,K dims).

        ``tma_tensor`` should already have domain_offset applied for varlen.

        For tile_shape = (BT, BK, BV) = (M, N, K):
          coord = (None, 0, None) — slices out the N-tile axis (mode 1) at 0,
          leaving mode 0 (M=BT) and mode 2 (K=BV) free for TMA to iterate.

        Returns (tXsX, tXgX) — SMEM partition and GMEM coordinate partition.
        """
        coord = (None, 0, None)
        gX = cute.local_tile(
            tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx))
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_A(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _tma_partition_B(self, tma_atom, tma_tensor, smem, tile_shape, tiled_mma, batch_idx, hidx):
        """Partition a TMA tensor as MMA B-operand (N,K dims).

        Mirrors the identical helper in recompute_wu.py / fwd_o.py.
        ``tma_tensor`` should already have domain_offset applied for varlen.

        For tile_shape = (BT, BK, BV) = (M, N, K):
          coord = (0, None, None) — slices out the M-tile axis (mode 0) at 0,
          leaving mode 1 (N=BK) and mode 2 (K=BV) free for TMA to iterate.

        Returns (tXsX, tXgX) — SMEM partition and GMEM coordinate partition.
        """
        coord = (0, None, None)
        gX = cute.local_tile(
            tma_tensor, cute.slice_(tile_shape, coord), (None, None, (hidx, batch_idx))
        )
        thr_mma = tiled_mma.get_slice(0)
        tCgX = thr_mma.partition_B(gX)
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(tCgX, 0, 3),
        )
        return tXsX, tXgX

    @cute.jit
    def _epilog_partition_varlen(self, atom, gC_2d, epi_tile, sC):
        """Partition for varlen epilog TMA load (2D tensor with domain_offset).

        Uses local_tile instead of flat_divide to correctly preserve TMA basis
        stride coordinates through domain_offset.  Matches Flash Attention's
        pattern: slice mode2 → domain_offset(2D) → local_tile → tma_partition.

        Uses (None, None) to keep all tile-count modes, producing the same
        rank as _epilog_partition (flat_divide) so copy indexing is unchanged.
        """
        gC_tiled = cute.local_tile(gC_2d, epi_tile, (None, None))
        sC_g = cute.group_modes(sC, 0, 2)
        gC_g = cute.group_modes(gC_tiled, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            sC_g,
            gC_g,
        )
        return bSG_sC, bSG_gC


# q, k, and v may arrive as unbound QKV views: only their innermost dimension is
# contiguous, and TMA needs every outer stride and the base pointer to be 16-byte
# (8 16-bit elements) aligned.
_MIN_ALIGN_BYTES = 16
_MIN_ALIGN_ELEMENTS_IO = _MIN_ALIGN_BYTES // 2


class ChunkKdaBwdWyDqkgConfig(NamedTuple):
    """Persistent-grid schedule for fused WY/dQKG backward."""

    grid_waves: int


@jit_cache
def _compile_chunk_kda_bwd_wy_dqkg(
    heads: int,
    head_dim: int,
    chunk_size: int,
    io_dtype: torch.dtype,
    scale: float,
    fastmath: bool,
    ragged: bool,
    use_int64_offsets: bool = False,
    key_heads: int | None = None,
    scalar_gate: bool = False,
):
    """Compile one persistent dense or ragged WY/dQKG specialization.

    ``key_heads`` defaults to ``heads``; fewer q/k heads share each q/k head across
    ``heads // key_heads`` consecutive value heads, and dQ/dK stay per value head.
    ``scalar_gate`` takes g and returns dg as ``[B, T, heads]`` instead of
    ``[B, T, heads, head_dim]``.
    """
    key_heads = heads if key_heads is None else key_heads
    cutlass_io_dtype = cute_dtype(io_dtype).cute_type
    op = ChunkKdaBwdWyDqkgFused(
        chunk_size=chunk_size,
        head_dim=head_dim,
        io_dtype=cutlass_io_dtype,
        scale=scale,
        use_fast_math=fastmath,
        use_int64_offsets=use_int64_offsets,
        scalar_gate=scalar_gate,
    )
    tokens, chunks, sequences = (cute.sym_int() for _ in range(3))
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int

    def tensor(dtype, shape):
        return make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=128,
        )

    def strided_io_tensor(shape):
        return make_fake_tensor(
            cutlass_io_dtype,
            shape,
            stride=tuple(sym_int(divisibility=_MIN_ALIGN_ELEMENTS_IO) for _ in shape[:-1]) + (1,),
            assumed_align=_MIN_ALIGN_BYTES,
        )

    gate_shape = (1, tokens, heads) if scalar_gate else (1, tokens, heads, head_dim)
    q = strided_io_tensor((1, tokens, key_heads, head_dim))
    k = strided_io_tensor((1, tokens, key_heads, head_dim))
    v = strided_io_tensor((1, tokens, heads, head_dim))
    v_new = tensor(cutlass_io_dtype, (1, tokens, heads, head_dim))
    g = tensor(cutlass.Float32, gate_shape)
    beta = tensor(cutlass.Float32, (1, tokens, heads))
    A = tensor(cutlass_io_dtype, (1, tokens, heads, chunk_size))
    h = tensor(cutlass_io_dtype, (1, chunks, heads, head_dim, head_dim))
    do = tensor(cutlass_io_dtype, (1, tokens, heads, head_dim))
    dh = tensor(cutlass_io_dtype, (1, chunks, heads, head_dim, head_dim))
    dv = tensor(cutlass_io_dtype, (1, tokens, heads, head_dim))
    dq = tensor(cutlass.Float32, (1, tokens, heads, head_dim))
    dk = tensor(cutlass.Float32, (1, tokens, heads, head_dim))
    dv2 = tensor(cutlass_io_dtype, (1, tokens, heads, head_dim))
    dg = tensor(cutlass.Float32, gate_shape)
    db = tensor(cutlass.Float32, (1, tokens, heads))
    dA = tensor(cutlass.Float32, (1, tokens, heads, chunk_size))
    cu_seqlens = tensor(cutlass.Int32, (sequences,)) if ragged else None
    chunk_offsets = tensor(cutlass.Int32, (sequences,)) if ragged else None
    return compile_tvm_ffi(
        op,
        q,
        k,
        v,
        v_new,
        g,
        beta,
        A,
        h,
        do,
        dh,
        dv,
        dq,
        dk,
        dv2,
        dg,
        db,
        dA,
        cu_seqlens,
        chunk_offsets,
        (Int32(1), Int32(1), Int32(key_heads), Int32(heads), Int32(head_dim), Int32(head_dim)),
        Int32(1),
        Int32(1),
        name=(
            f"kda_bwd_wy_dqkg_hk{key_heads}_h{heads}_d{head_dim}_c{chunk_size}_"
            f"{str(io_dtype).removeprefix('torch.')}_fm{int(fastmath)}_"
            f"rg{int(ragged)}_i64{int(use_int64_offsets)}_sg{int(scalar_gate)}"
        ),
    )


class ChunkKdaBwdWyDqkgTunable:
    class Args(NamedTuple):
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        v_new: torch.Tensor
        g: torch.Tensor
        beta: torch.Tensor
        A: torch.Tensor
        h: torch.Tensor
        do: torch.Tensor
        dh: torch.Tensor
        dv: torch.Tensor
        dq: torch.Tensor
        dk: torch.Tensor
        dv2: torch.Tensor
        dg: torch.Tensor
        db: torch.Tensor
        dA: torch.Tensor
        cu_seqlens: torch.Tensor | None
        chunk_offsets: torch.Tensor | None
        chunk_size: int
        scale: float
        fastmath: bool

    @staticmethod
    def default_config(
        _args: Args,
        *,
        target: CompileTarget,
    ) -> ChunkKdaBwdWyDqkgConfig:
        """Choose the deterministic single-wave persistent schedule."""
        if target.sm_count is None:
            raise RuntimeError("KDA launch requires a CUDA target with an SM count")
        return ChunkKdaBwdWyDqkgConfig(grid_waves=1)

    @staticmethod
    def tuning_key(args: Args, *, target: CompileTarget) -> tuple[int]:
        """Key winners by the power-of-two bucket of persistent-grid work waves.

        The tile count (chunk capacity times heads) changes with the document count at a
        fixed token count; the wave count it implies is what separates the 1- and 2-wave
        schedules, so nearby capacities share a winner instead of re-benchmarking.
        """
        if target.sm_count is None:
            raise RuntimeError("KDA tuning requires a CUDA target with an SM count")
        waves = cdiv(args.h.shape[1] * args.v.shape[2], target.sm_count)
        return (1 << (waves - 1).bit_length(),)

    @staticmethod
    def configs(
        args: Args,
    ) -> tuple[ChunkKdaBwdWyDqkgConfig, ...]:
        del args
        return tuple(ChunkKdaBwdWyDqkgConfig(waves) for waves in (1, 2))

    @staticmethod
    def compile_call(
        config: ChunkKdaBwdWyDqkgConfig,
        args: Args,
    ) -> tuple[int, int, int, torch.dtype, float, bool, bool, bool, int, bool]:
        return (
            args.v.shape[2],
            args.q.shape[3],
            args.chunk_size,
            args.q.dtype,
            args.scale,
            args.fastmath,
            args.chunk_offsets is not None,
            requires_int64_abi(
                args.q,
                args.k,
                args.v,
                args.v_new,
                args.g,
                args.beta,
                args.A,
                args.h,
                args.do,
                args.dh,
                args.dv,
                args.dq,
                args.dk,
                args.dv2,
                args.dg,
                args.db,
                args.dA,
            ),
            args.q.shape[2],
            args.g.ndim == 3,
        )

    compile = staticmethod(_compile_chunk_kda_bwd_wy_dqkg)

    @staticmethod
    def launch(
        compiled,
        config: ChunkKdaBwdWyDqkgConfig,
        args: Args,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if config.grid_waves < 1:
            raise ValueError(f"grid_waves must be positive, got {config.grid_waves}")
        _batch, tokens, key_heads, head_dim = args.q.shape
        heads = args.v.shape[2]
        sequences = 1 if args.cu_seqlens is None else args.cu_seqlens.shape[0] - 1
        compiled(
            args.q,
            args.k,
            args.v,
            args.v_new,
            args.g,
            args.beta,
            args.A,
            args.h,
            args.do,
            args.dh,
            args.dv,
            args.dq,
            args.dk,
            args.dv2,
            args.dg,
            args.db,
            args.dA,
            args.cu_seqlens,
            args.chunk_offsets,
            (
                Int32(sequences),
                Int32(tokens),
                Int32(key_heads),
                Int32(heads),
                Int32(head_dim),
                Int32(head_dim),
            ),
            Int32(args.h.shape[1]),
            # The schedule only sizes the persistent grid, so it is a launch argument.
            Int32(get_device_properties(args.q.device).multi_processor_count * config.grid_waves),
        )
        return args.dq, args.dk, args.dv2, args.dg, args.db, args.dA


def chunk_kda_bwd_wy_dqkg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    do: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    metadata: RaggedChunkMetadata | None = None,
    *,
    scale: float,
    chunk_size: int = 64,
    fastmath: bool = False,
    config: ChunkKdaBwdWyDqkgConfig | None = None,
    autotune: bool = False,
    configs: Iterable[ChunkKdaBwdWyDqkgConfig] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run or tune the direct dense or ragged fused WY/dQKG backward stage.

    ``q`` and ``k`` may have fewer heads than ``v``; each q/k head is shared by
    ``H // HK`` consecutive value heads and the returned dQ/dK are per value head.
    ``g`` is either a per-key-dimension gate ``[B, T, H, K]`` or a scalar gate ``[B, T, H]``;
    ``dg`` has the same shape as ``g``, a scalar dg being the sum over K of the vector one.
    """
    batch, tokens, key_heads, head_dim = q.shape
    heads = v.shape[2]
    if batch != 1 or head_dim != 128 or v.shape[-1] != 128:
        raise ValueError("the fused WY backward requires B=1 and K=V=128")
    if k.shape != q.shape or heads % key_heads:
        raise ValueError("k must match q and value heads must be a multiple of q/k heads")
    if g.shape not in ((batch, tokens, heads), (batch, tokens, heads, head_dim)):
        raise ValueError(
            f"g must have shape {(batch, tokens, heads)} or {(*v.shape[:3], head_dim)}"
        )
    if chunk_size != 64:
        raise ValueError(f"the fused WY backward requires chunk_size=64, got {chunk_size}")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"the fused WY backward requires FP16 or BF16 inputs, got {q.dtype}")
    io_tensors = (k, v, v_new, A, h, do, dh, dv)
    if any(tensor.dtype != q.dtype for tensor in io_tensors):
        raise ValueError("q, k, v, v_new, A, h, do, dh, and dv must have the same dtype")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("g and beta must have dtype torch.float32")
    if metadata is None:
        if tokens % chunk_size:
            raise ValueError("the fused WY backward requires complete chunks")
        chunks = tokens // chunk_size
        cu_seqlens = None
        chunk_offsets = None
    else:
        metadata.validate_chunk_size(chunk_size)
        chunks = metadata.capacity
        cu_seqlens = metadata.cu_seqlens
        chunk_offsets = metadata.chunk_offsets
    expected_h = (batch, chunks, heads, head_dim, head_dim)
    if h.shape != expected_h or dh.shape != expected_h:
        raise ValueError(f"h and dh must have shape {expected_h}")
    dqk_shape = (batch, tokens, heads, head_dim)
    if tokens == 0:
        return (
            q.new_empty(dqk_shape, dtype=torch.float32),
            q.new_empty(dqk_shape, dtype=torch.float32),
            torch.empty_like(v, memory_format=torch.contiguous_format),
            torch.empty_like(g, memory_format=torch.contiguous_format),
            torch.empty_like(beta, memory_format=torch.contiguous_format),
            torch.zeros_like(A, dtype=torch.float32, memory_format=torch.contiguous_format),
        )

    args = ChunkKdaBwdWyDqkgTunable.Args(
        q=q,
        k=k,
        v=v,
        v_new=v_new,
        g=g,
        beta=beta,
        A=A,
        h=h,
        do=do,
        dh=dh,
        dv=dv,
        # Every generated output is compact even when v is a strided QKV view.
        dq=q.new_empty(dqk_shape, dtype=torch.float32),
        dk=q.new_empty(dqk_shape, dtype=torch.float32),
        dv2=torch.empty_like(v, memory_format=torch.contiguous_format),
        # Every active dg row is written (tail chunks store row-predicated); the inactive capacity
        # is never loaded (intra stages predicate rows) and the reverse gate scan zeroes it.
        dg=torch.empty_like(g, memory_format=torch.contiguous_format),
        db=torch.empty_like(beta, memory_format=torch.contiguous_format),
        dA=torch.empty_like(A, dtype=torch.float32, memory_format=torch.contiguous_format),
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        chunk_size=chunk_size,
        scale=scale,
        fastmath=fastmath,
    )
    # An explicit config pins the schedule regardless of the plumbed flag.
    autotune = autotune and config is None
    result, _ = run_tunable(
        ChunkKdaBwdWyDqkgTunable,
        args,
        config=config,
        autotune=autotune,
        configs=configs,
        parallel_compile=_compile_chunk_kda_bwd_wy_dqkg.disk_cache_enabled(),
        target=detect_compile_target(q.device.index),
    )
    return result


__all__ = ["ChunkKdaBwdWyDqkgConfig", "chunk_kda_bwd_wy_dqkg"]
