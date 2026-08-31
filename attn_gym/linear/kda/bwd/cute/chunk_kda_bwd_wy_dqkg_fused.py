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
from typing import NamedTuple

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute, pipeline, utils

# ============================================================================
# SM100 primitives used by this kernel
# ============================================================================
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith as _arith
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass._mlir.dialects import vector as _vector
from cutlass.cute.arch import (
    elect_one,
    mbarrier_arrive,
    mbarrier_arrive_and_expect_tx,
    mbarrier_init,
    mbarrier_init_fence,
    mbarrier_wait,
)
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import make_umma_smem_desc, smem_descriptor_to_int
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cute.tensor import TensorSSA
from cutlass.cute.typing import BFloat16, Float16, Float32, Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache, run_tunable
from attn_gym._backends.cute.target import CompileTarget, detect_compile_target, get_compile_target
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import (
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)

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
        _nvvm.tcgen05_mma_ws(
            mma_kind=_nvvm.Tcgen05MMAKind.F16,
            d=destination,
            a=_ir(a_value, loc, ip),
            b=_ir(b_value, loc, ip),
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
def tcgen05_st_32x32b(num: int, taddr: int, values):
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


# Mapping from torch dtypes to CuTeDSL scalar types.
_torch_to_cutlass_dtype = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
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


class ChunkKdaBwdWyDqkgFused:
    """
    CuTe DSL kernel for chunk_kda_bwd_kernel_wy_dqkg_fused.

    Computes backward gradients dq, dk, dv2, dg, db, dA for the KDA
    chunkwise delta-rule backward pass.

    Architecture: 1 CudaCore WG + 1 MMA warp + TMA/Aux warps.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim_k: int = 128,
        head_dim_v: int = 128,
        acc_dtype: type[cutlass.Numeric] = cutlass.Float32,
        io_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        g_dtype: type[cutlass.Numeric] = cutlass.Float32,
        scale: float = 1.0,
        grid_waves: int = 1,
        use_fast_math: bool = True,
        use_int64_offsets: bool = False,
    ):
        assert chunk_size == 64, "chunk_size must be 64"
        assert head_dim_k == 128 and head_dim_v == 128, (
            f"head_dim_k and head_dim_v must both be 128, got head_dim_k={head_dim_k}, head_dim_v={head_dim_v}"
        )
        require_blackwell_target()

        self.use_fast_math = use_fast_math
        self.use_int64_offsets = use_int64_offsets
        self.chunk_size = chunk_size
        self.head_dim_k = head_dim_k
        self.head_dim_v = head_dim_v
        self.acc_dtype = acc_dtype
        self.io_dtype = io_dtype
        self.g_dtype = g_dtype
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

        # Warp layout: WG0/WG1 (8 CudaCore warps) + WG2 (MMA/Load/Aux/Store)
        self.threads_per_warp = 32
        self.cuda_warp_ids = (0, 1, 2, 3)  # WG0: CudaCore + Store
        self.cuda2_warp_ids = (4, 5, 6, 7)  # WG1: CudaCore + Store
        self.mma_warp_id = 8  # WG2: MMA dispatch
        self.load_warp_id = 9  # WG2: TMA Load
        self.aux_warp_ids = (10, 11)  # WG2: Aux/Load/Store Aux
        self.threads_per_cta = self.threads_per_warp * 12  # 384 threads (3 WGs)

        # Keep the CTA register pool fixed while favoring the long CUDA-core epilogue.
        self.num_regs_cuda = 224
        self.num_regs_others = 56
        if grid_waves < 1:
            raise ValueError(f"grid_waves must be positive, got {grid_waves}")
        self.grid_waves = grid_waves

        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        # Number of K/V tiles
        self.num_k_tiles = (head_dim_k + self.BK - 1) // self.BK  # 128/128 = 1
        self.num_v_tiles = (head_dim_v + self.BV - 1) // self.BV  # 128/64 = 2

        # ── Pipeline stages ──
        # V-loop TMA: 2-stage double buffer
        self.vloop_stage = 2
        self.kloop_stage = 1
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

        # Named barriers
        self.tmem_dealloc_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
        )
        self.cuda_wg_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * 8,
        )
        self.buffer_align_bytes = 1024

        # Persistent scheduling
        self.persistent = True
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA compilation requires a CUDA target with an SM count")
        self.num_sm = sm_count

    def _compute_grid(self, HV, total_nt):
        """Compute grid dimensions for persistent kernel launch.

        Grid: (min(num_sm * grid_waves, total_tiles), 1, 1)
        Each CTA handles multiple tiles via stride-by-gridDim.x loop.
        """
        total_tiles = total_nt * HV
        grid_x = cutlass.min(Int32(self.num_sm * self.grid_waves), total_tiles)
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
        g_in: cute.Tensor,  # [B, T, HV, K] fp32
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
        dg_in: cute.Tensor,  # [B, T, HV, K] fp32
        db_in: cute.Tensor,  # [B, T, HV]    fp32
        dA_in: cute.Tensor,  # [B, T, HV, BT] fp32
        # ── Metadata ──
        cu_seqlens_in: cute.Tensor | None,  # [N+1] int32
        chunk_offsets_in: cute.Tensor | None,  # [N+1] int32
        problem_size: tuple[Int32, Int32, Int32, Int32, Int32, Int32],  # (B, T, H, HV, K, V)
        total_nt: Int32,
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

        # g: (T, K, (HV, data_B)) fp32
        g_layout = cute.make_layout(
            (T, K, (HV, data_B)),
            stride=(self.upcast(HV * K), 1, (K, self.upcast(T) * HV * K)),
        )
        g = cute.make_tensor(g_ptr, g_layout)

        # beta: (T, (HV, data_B)) fp32
        beta_layout = cute.make_layout(
            (T, (HV, data_B)),
            stride=(self.upcast(HV), (1, self.upcast(T) * HV)),
        )
        beta = cute.make_tensor(beta_ptr, beta_layout)

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

        # dg: (T, K, (HV, data_B)) fp32
        dg = cute.make_tensor(dg_ptr, dqk_layout)

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
            tcgen05.OperandMajorMode.K,  # A: K-major
            tcgen05.OperandMajorMode.K,  # B: K-major
            self.acc_dtype,
            self.cta_group,
            self.vloop_gemm_tiler[:2],  # (64, 128)
            # default a_source=OperandSource.SMEM → SS mode
        )

        # 2. dA_vloop_tiled_mma: SS K,K (64,64) — dA vloop + kpost_dA
        #    dA += dv @ v^T, dA += dw @ kg^T
        dA_vloop_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.dA_vloop_tiler[:2],  # (64, 64)
            # default a_source=OperandSource.SMEM → SS mode
        )

        # 3. dvb_tiled_mma: SS MN,MN (64,64) — dvb + dkgb
        #    dvb = A @ dv, dkgb = A @ dw
        dvb_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dvb_tiler[:2],  # (64, 64)
        )

        # dkgb_tiled_mma: SS MN,MN (64,128) - dkgb
        dkgb_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.kloop_dkgb_tiler[:2],  # (64, 128)
        )

        # dA_kloop_tiled_mma: SS K,K (64, 64)
        # dA += dw @ kg^T
        dA_kloop_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.kloop_dA_tiler[:2],  # (64, 64)
        )

        # dA2post_tiled_mma: SS K,K (64,64)
        # dA = dA @ A
        dA2post_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.dApost_tiler[:2],  # (64, 64)
            # tcgen05.OperandSource.SMEM,  # SS mode
        )

        # dA3post_tiled_mma: SS MN,MN (64,64)
        # dA = A @ dA
        dA3post_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.io_dtype,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_dtype,
            self.cta_group,
            self.dApost_tiler[:2],  # (64, 64)
            # tcgen05.OperandSource.SMEM,  # SS mode
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
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.kloop_stage,
        )

        k_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            self.kloop_stage,
        )

        q_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BK),
            1,
        )

        dg_epi_smem_layout = sm100_utils.make_smem_layout_epi(
            self.g_dtype,
            utils.LayoutEnum.ROW_MAJOR,
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

        g_epi_smem_no_stage = cute.select(g_epi_smem_layout, mode=[0, 1])
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
        tma_atom_dg, tma_tensor_dg = cpasync.make_tiled_tma_atom(
            tma_store_op,
            dg,
            dg_epi_smem_no_stage,
            (self.BT, self.BK),
        )

        # ===================== TMA byte counts =====================
        self.tma_bytes_A = cute.size_in_bytes(self.io_dtype, A_mn_opA_smem_no_stage)
        self.tma_bytes_dv = cute.size_in_bytes(self.io_dtype, vloop_opA_smem_no_stage)
        self.tma_bytes_h = cute.size_in_bytes(self.io_dtype, vloop_opB_smem_no_stage)
        self.tma_bytes_dh = cute.size_in_bytes(self.io_dtype, vloop_opB_smem_no_stage)
        self.tma_bytes_do = cute.size_in_bytes(self.io_dtype, vloop_opA_smem_no_stage)
        self.tma_bytes_vnew = cute.size_in_bytes(self.io_dtype, vloop_opA_smem_no_stage)
        self.tma_bytes_g = cute.size_in_bytes(self.g_dtype, g_epi_smem_no_stage)
        self.tma_bytes_v = cute.size_in_bytes(self.io_dtype, v_opB_smem_no_stage)
        self.tma_bytes_k = cute.size_in_bytes(self.io_dtype, k_epi_smem_no_stage)
        self.tma_bytes_q = cute.size_in_bytes(self.io_dtype, q_epi_smem_no_stage)

        # ===================== SharedStorage =====================
        @cute.struct
        class SharedStorage:
            # ======= mbarrier =======
            bar_load_A: cute.struct.MemRange[Int64, self.a_stage * 2]
            bar_load_dv: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_mma_dvb: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_load_beta: cute.struct.MemRange[Int64, 1 * 2]
            bar_tma_h: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_mma_cuda_h: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_tma_dh: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_mma_cuda_dh: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_tma_v: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_mma_cuda_v: cute.struct.MemRange[Int64, self.vloop_stage]
            bar_load_do: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_load_g: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_load_vnew: cute.struct.MemRange[Int64, self.vloop_stage * 2]
            bar_load_q: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_load_k: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_mma_dq: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dw: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dk: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dkgb: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dA: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dA2: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_dA3: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_mma_done_vloop: cute.struct.MemRange[Int64, self.mma_stage]
            bar_prologue_dw: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_prologue_kg: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            bar_prologue_dA2: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_prologue_dA3: cute.struct.MemRange[Int64, self.mma_stage * 2]
            bar_store_dg: cute.struct.MemRange[Int64, self.kloop_stage * 2]
            # TMEM holding buffer
            tmem_holding_buf: Int32
            # A, stage=2, [BT,BT], 16KB
            buf_A: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(A_mn_opA_smem)],
                self.buffer_align_bytes,
            ]
            # k, stage=1, [BT,BK], 16KB
            buf_k: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_epi_smem_layout)],
                self.buffer_align_bytes,
            ]
            # g, stage=1, [BT,BK], 32KB
            buf_g: cute.struct.Align[
                cute.struct.MemRange[self.g_dtype, cute.cosize(g_epi_smem_layout)],
                self.buffer_align_bytes,
            ]
            # q, stage=1, [BT,BK], 16KB
            buf_q: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_epi_smem_layout)],
                self.buffer_align_bytes,
            ]
            # V-loop buffers, stage=2
            # h, dh, [BK,BV] 32KB*2
            buf_h: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opB_smem)],
                self.buffer_align_bytes,
            ]
            buf_dh: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opB_smem)],
                self.buffer_align_bytes,
            ]
            # do, dv, v_new, v, [BT,BV] 16KB*4
            buf_do: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opA_smem)],
                self.buffer_align_bytes,
            ]
            buf_dv: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opA_smem)],
                self.buffer_align_bytes,
            ]
            buf_vnew: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(vloop_opA_smem)],
                self.buffer_align_bytes,
            ]
            buf_v: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_opB_smem)],
                self.buffer_align_bytes,
            ]

            # dw, stage=1, [BT,BK] 16KB
            buf_dw: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(dw_k_opA_smem)],
                self.buffer_align_bytes,
            ]
            # Scalars
            s_beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BT],
                128,
            ]
            # 2 slots per row, one per warpgroup, for deterministic db reduction
            # (avoids cross-wg fp32 atomicAdd on shared memory).
            s_db: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BT * 2],
                128,
            ]
            s_gn: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK],
                128,
            ]
            s_dgk: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK],
                128,
            ]

        self.shared_storage = SharedStorage

        # ===================== Grid =====================
        grid = self._compute_grid(HV, total_nt)

        # ===================== Launch kernel =====================
        self.kernel(
            # MMA objects (4)
            vloop_tiled_mma,
            dA_vloop_tiled_mma,
            dvb_tiled_mma,
            dA_kloop_tiled_mma,
            dA2post_tiled_mma,
            dA3post_tiled_mma,
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
            q,
            k,
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
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        # MMA objects (4)
        vloop_tiled_mma: cute.TiledMma,
        dA_vloop_tiled_mma: cute.TiledMma,
        dvb_tiled_mma: cute.TiledMma,
        dA_kloop_tiled_mma: cute.TiledMma,
        dA2post_tiled_mma: cute.TiledMma,
        dA3post_tiled_mma: cute.TiledMma,
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
        tma_atom_g: cute.CopyAtom,
        tma_tensor_g: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        tma_tensor_v: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        tma_tensor_k: cute.Tensor,
        tma_atom_vnew: cute.CopyAtom,
        tma_tensor_vnew: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        tma_tensor_q: cute.Tensor,
        tma_atom_dg: cute.CopyAtom,
        tma_tensor_dg: cute.Tensor,
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
        q_gmem: cute.Tensor,
        k_gmem: cute.Tensor,
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
        _B, _T, H, HV, K, V = problem_size
        BT = self.BT

        # ===================== Persistent work decode =====================
        # Grid: (min(num_sm * grid_waves, total_tiles), 1, 1) — persistent
        block_idx_x = cute.arch.block_idx()[0]
        grid_dim_x = cute.arch.grid_dim()[0]
        thread_idx = cute.arch.thread_idx()[0]
        lane_idx = thread_idx % 32

        if cutlass.const_expr(chunk_offsets is not None):
            active_chunks = load_ragged_chunk_count(chunk_offsets)
        else:
            active_chunks = capacity
        total_work_units = active_chunks * HV
        num_iters = (total_work_units - block_idx_x + grid_dim_x - 1) // grid_dim_x

        num_cuda_warps_total = len(self.cuda_warp_ids) + len(self.cuda2_warp_ids)

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_A)
            cpasync.prefetch_descriptor(tma_atom_dv)
            cpasync.prefetch_descriptor(tma_atom_h)
            cpasync.prefetch_descriptor(tma_atom_dh)
            cpasync.prefetch_descriptor(tma_atom_do)
            cpasync.prefetch_descriptor(tma_atom_g)
            cpasync.prefetch_descriptor(tma_atom_v)
            cpasync.prefetch_descriptor(tma_atom_vnew)
            cpasync.prefetch_descriptor(tma_atom_k)
            cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_dg)

        # ===================== SMEM allocation =====================
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Barrier Initialization
        bar_mma_done_vloop_ptr = storage.bar_mma_done_vloop.data_ptr()
        # NOTE: for h, dh and v, consumer contains both MMA and CUDA Core, so we use original mbarrier declaration instead of pipeline utils
        bar_tma_h_ptr = storage.bar_tma_h.data_ptr()
        bar_mma_cuda_h_ptr = storage.bar_mma_cuda_h.data_ptr()
        bar_tma_dh_ptr = storage.bar_tma_dh.data_ptr()
        bar_mma_cuda_dh_ptr = storage.bar_mma_cuda_dh.data_ptr()
        bar_tma_v_ptr = storage.bar_tma_v.data_ptr()
        bar_mma_cuda_v_ptr = storage.bar_mma_cuda_v.data_ptr()
        if warp_idx == 0:
            with elect_one():
                for i in cutlass.range(self.mma_stage, unroll_full=True):
                    mbarrier_init(bar_mma_done_vloop_ptr + i, 1)
                for i in cutlass.range(self.vloop_stage, unroll_full=True):
                    mbarrier_init(bar_tma_h_ptr + i, 1)
                    mbarrier_init(bar_mma_cuda_h_ptr + i, num_cuda_warps_total * 32 + 1)
                    mbarrier_init(bar_tma_dh_ptr + i, 1)
                    mbarrier_init(bar_mma_cuda_dh_ptr + i, num_cuda_warps_total * 32 + 1)
                    mbarrier_init(bar_tma_v_ptr + i, 1)
                    mbarrier_init(bar_mma_cuda_v_ptr + i, num_cuda_warps_total * 32 + 1)
                mbarrier_init_fence()

        # ====== Pipeline Definition ======
        pipeline_load_A = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.bar_load_A.data_ptr(),
            num_stages=self.a_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_bytes_A,
        )
        pipeline_load_dv = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.bar_load_dv.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_bytes_dv,
        )
        pipeline_load_do = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.bar_load_do.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_bytes_do,
        )
        pipeline_load_vnew = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.bar_load_vnew.data_ptr(),
            num_stages=self.vloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_bytes_vnew,
        )
        pipeline_load_g = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_g.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(
                num_cuda_warps_total + len(self.aux_warp_ids)
            ),
            tx_count=self.tma_bytes_g,
        )
        pipeline_load_k = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_k.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total),
            tx_count=self.tma_bytes_k,
        )
        pipeline_load_q = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.bar_load_q.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total),
            tx_count=self.tma_bytes_q,
        )
        pipeline_mma_dvb = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dvb.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dq = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dq.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dk = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dk.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dw = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dw.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dA = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dA.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dA2 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dA2.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_mma_dA3 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dA3.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_prologue_dw = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.bar_prologue_dw.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        )
        pipeline_prologue_kg = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.bar_prologue_kg.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        )
        pipeline_prologue_dA2 = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.bar_prologue_dA2.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        )
        pipeline_prologue_dA3 = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.bar_prologue_dA3.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
        )
        pipeline_mma_dkgb = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.bar_mma_dkgb.data_ptr(),
            num_stages=self.mma_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_load_beta = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_load_beta.data_ptr(),
            num_stages=1,
            producer_group=make_thread_cooperative_group(len(self.aux_warp_ids) * 32),
            consumer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
        )
        pipeline_store_dg = pipeline.PipelineAsync.create(
            barrier_storage=storage.bar_store_dg.data_ptr(),
            num_stages=self.kloop_stage,
            producer_group=make_thread_cooperative_group(num_cuda_warps_total * 32),
            consumer_group=make_thread_cooperative_group(len(self.aux_warp_ids) * 32),
        )

        # ===================== TMEM allocation =====================
        tmem_alloc_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_bar,
            allocator_warp_id=self.load_warp_id,
        )
        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        vloop_opA_smem_no_stage = cute.select(vloop_opA_smem, mode=[0, 1, 2])
        vloop_opB_smem_no_stage = cute.select(vloop_opB_smem, mode=[0, 1, 2])
        A_mn_opA_smem_no_stage = cute.select(A_mn_opA_smem, mode=[0, 1, 2])
        v_opB_smem_no_stage = cute.select(v_opB_smem, mode=[0, 1, 2])

        sA = storage.buf_A.get_tensor(A_mn_opA_smem.outer, swizzle=A_mn_opA_smem.inner)
        sDv = storage.buf_dv.get_tensor(vloop_opA_smem.outer, swizzle=vloop_opA_smem.inner)
        sH = storage.buf_h.get_tensor(vloop_opB_smem.outer, swizzle=vloop_opB_smem.inner)
        sDh = storage.buf_dh.get_tensor(vloop_opB_smem.outer, swizzle=vloop_opB_smem.inner)
        sDo = storage.buf_do.get_tensor(vloop_opA_smem.outer, swizzle=vloop_opA_smem.inner)
        sVnew = storage.buf_vnew.get_tensor(vloop_opA_smem.outer, swizzle=vloop_opA_smem.inner)
        sV = storage.buf_v.get_tensor(v_opB_smem.outer, swizzle=v_opB_smem.inner)

        sDv_ptr_base = storage.buf_dv.data_ptr().toint()
        vloop_opA_bytes_per_stage = cute.size_in_bytes(self.io_dtype, vloop_opA_smem_no_stage)
        sDo_ptr_base = storage.buf_do.data_ptr().toint()
        sVnew_ptr_base = storage.buf_vnew.data_ptr().toint()
        sV_ptr_base = storage.buf_v.data_ptr().toint()
        v_opB_bytes_per_stage = cute.size_in_bytes(self.io_dtype, v_opB_smem_no_stage)
        sH_ptr_base = storage.buf_h.data_ptr().toint()
        sDh_ptr_base = storage.buf_dh.data_ptr().toint()
        vloop_opB_bytes_per_stage = cute.size_in_bytes(self.io_dtype, vloop_opB_smem_no_stage)
        sA_ptr_base = storage.buf_A.data_ptr().toint()
        A_bytes_per_stage = cute.size_in_bytes(self.io_dtype, A_mn_opA_smem_no_stage)

        # NOTE: make_umma_smem_desc requires the iterator to carry the swizzle
        # (and ≥16B alignment). When constructing a tensor over a ComposedLayout
        # via make_ptr+make_tensor, the swizzle ends up composed on the layout
        # rather than the iterator, which breaks make_umma_smem_desc. Use
        # recast_ptr to move the swizzle onto the iterator and pair it with the
        # underlying (non-swizzle) outer layout.
        sDv_mn = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dv.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dv_mn_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            dv_mn_opB_smem.outer,
        )
        sDw_mn = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dw.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dw_mn_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            dw_mn_opB_smem.outer,
        )
        sDw_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dw.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dw_k_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            dw_k_opA_smem.outer,
        )
        sDv_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dv.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=vloop_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opA_smem.outer,
        )
        sV_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_v.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=v_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            v_opB_smem.outer,
        )
        sA_mn = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_A.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=A_mn_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            A_mn_opA_smem.outer,
        )
        sDo_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_do.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=vloop_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opA_smem.outer,
        )
        sVnew_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_vnew.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=vloop_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opA_smem.outer,
        )
        sH_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_h.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=vloop_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opB_smem.outer,
        )
        sDh_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_dh.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=vloop_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            vloop_opB_smem.outer,
        )
        sKG_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_k.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=kg_k_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            kg_k_opB_smem.outer,
        )
        sA_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_A.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=A_k_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            A_k_opB_smem.outer,
        )
        sDA_mn = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_q.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dA_mn_opB_smem.inner,
                dtype=self.io_dtype,
            ),
            dA_mn_opB_smem.outer,
        )
        sDA_k = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_q.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=dA_k_opA_smem.inner,
                dtype=self.io_dtype,
            ),
            dA_k_opA_smem.outer,
        )
        sG_raw = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.g_dtype,
                    storage.buf_g.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=g_epi_smem_layout.inner,
                dtype=self.g_dtype,
            ),
            g_epi_smem_layout.outer,
        )
        sG_raw_ptr = cute.make_ptr(
            self.g_dtype, storage.buf_g.data_ptr().toint(), cute.AddressSpace.smem
        )
        sK_raw = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_k.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=k_epi_smem_layout.inner,
                dtype=self.io_dtype,
            ),
            k_epi_smem_layout.outer,
        )
        sK_raw_ptr = cute.make_ptr(
            self.io_dtype, storage.buf_k.data_ptr().toint(), cute.AddressSpace.smem
        )
        sDw_raw_ptr = cute.make_ptr(
            self.io_dtype, storage.buf_dw.data_ptr().toint(), cute.AddressSpace.smem
        )
        sQ_raw = cute.make_tensor(
            cute.recast_ptr(
                cute.make_ptr(
                    self.io_dtype,
                    storage.buf_q.data_ptr().toint(),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                swizzle_=q_epi_smem_layout.inner,
                dtype=self.io_dtype,
            ),
            q_epi_smem_layout.outer,
        )
        sQ_raw_ptr = cute.make_ptr(
            self.io_dtype, storage.buf_q.data_ptr().toint(), cute.AddressSpace.smem
        )

        # Scalar SMEM buffers (plain layouts, no swizzle)
        sBeta = cute.make_tensor(
            cute.make_ptr(Float32, storage.s_beta.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BT,), stride=(1,)),
        )
        # sDb layout: (BT, 2). Inner dim = wg_idx slot. Stride (1, BT) so each
        # wg's column is contiguous (better for the reduce in Phase 3).
        sDb = cute.make_tensor(
            cute.make_ptr(Float32, storage.s_db.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BT, 2), stride=(1, self.BT)),
        )
        sDgk = cute.make_tensor(
            cute.make_ptr(Float32, storage.s_dgk.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BK,), stride=(1,)),
        )
        sGn = cute.make_tensor(
            cute.make_ptr(Float32, storage.s_gn.data_ptr().toint(), cute.AddressSpace.smem),
            cute.make_layout((self.BK,), stride=(1,)),
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline.pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        tmem.allocate(TMEM_TOTAL)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

        # ===================== Warp dispatch =====================
        # CUDA Core loop body
        if warp_idx in self.cuda_warp_ids or warp_idx in self.cuda2_warp_ids:
            cute.arch.setmaxregister_increase(self.num_regs_cuda)

            load_beta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            load_g_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kloop_stage
            )
            mma_dvb_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            mma_dq_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            mma_dw_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            mma_dk_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            load_k_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kloop_stage
            )
            prologue_dw_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kloop_stage
            )
            prologue_kg_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kloop_stage
            )
            mma_dgkb_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            load_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kloop_stage
            )
            mma_dA_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            mma_dA2_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            mma_dA3_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            prologue_dA2_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            prologue_dA3_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            store_dg_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kloop_stage
            )

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

            vloop_stage_idx = 0
            vloop_phase = 0
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                G = HV // H
                i_t = work_idx // HV  # chunk index (global)
                i_hv = work_idx % HV  # value-head index
                i_h = i_hv // G  # q/k head index
                tok_offset, tile_idx, sub_seq_len = _resolve_chunk_route(
                    cu_seqlens,
                    chunk_offsets,
                    Int32(i_t),
                )

                # NOTE: must sync before next wu_iter's `sDgk[local_tidx] = 0`
                # init, otherwise WG0 of next iter may overwrite sDgk while
                # WG1 of this iter (row == sub_seq_len - 1 lane) is still
                # reading sDgk[col] above. This was the source of the
                # non-deterministic dg accuracy bug.
                self.cuda_wg_sync_barrier.arrive_and_wait()
                # fill db, dgk to 0. Each wg zeroes its own sDb column.
                if local_tidx < self.BT:
                    sDb[local_tidx, 0] = Float32(0.0)
                    sDb[local_tidx, 1] = Float32(0.0)
                if local_tidx < self.BK:
                    sDgk[local_tidx] = Float32(0.0)
                self.cuda_wg_sync_barrier.arrive_and_wait()

                pipeline_load_beta.consumer_wait(load_beta_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")

                beta_val = sBeta[(row,)]
                db_val = Float32(0.0)
                for v_iter in cutlass.range(self.num_v_tiles):
                    # dgk += sum(h * dh, axis=0)
                    mbarrier_wait(bar_tma_h_ptr + vloop_stage_idx, vloop_phase)
                    mbarrier_wait(bar_tma_dh_ptr + vloop_stage_idx, vloop_phase)

                    sH_raw_ptr = cute.make_ptr(
                        self.io_dtype,
                        sH_ptr_base + vloop_stage_idx * vloop_opB_bytes_per_stage,
                        cute.AddressSpace.smem,
                    )
                    sDh_raw_ptr = cute.make_ptr(
                        self.io_dtype,
                        sDh_ptr_base + vloop_stage_idx * vloop_opB_bytes_per_stage,
                        cute.AddressSpace.smem,
                    )
                    # each thread in one WG processes one row
                    self.cuda_wg_sync_barrier.arrive_and_wait()
                    if wg_idx == 0:
                        for i in cutlass.range_constexpr(self.BV // 8):
                            col = i * 8
                            h_vals = smem_load_f16x8_sw128(
                                sH_raw_ptr, local_tidx, col, self.io_dtype
                            )
                            dh_vals = smem_load_f16x8_sw128(
                                sDh_raw_ptr, local_tidx, col, self.io_dtype
                            )
                            h_dh_vals = cute.make_rmem_tensor((8,), Float32)
                            h_dh_vals.store(h_vals.load().to(Float32) * dh_vals.load().to(Float32))
                            for j in cutlass.range_constexpr(8):
                                sDgk[(local_tidx,)] += h_dh_vals[j]

                    mbarrier_arrive(bar_mma_cuda_h_ptr + vloop_stage_idx)
                    mbarrier_arrive(bar_mma_cuda_dh_ptr + vloop_stage_idx)

                    pipeline_mma_dvb.consumer_wait(mma_dvb_consumer_state)
                    tcgen05_fence_after()
                    dvb_i32 = tcgen05_ld_32x32b(
                        bv_num_cols_per_wg, TMEM_FLEX_OFF + wg_idx * bv_num_cols_per_wg
                    )
                    tcgen05_fence_before()
                    cute.arch.fence_view_async_tmem_load()

                    pipeline_mma_dvb.consumer_release(mma_dvb_consumer_state)
                    mma_dvb_consumer_state.advance()

                    dvb_f32 = reinterpret_cast(dvb_i32, Int32, bv_num_cols_per_wg, Float32)
                    dvb_f32_val = TensorSSA(dvb_f32, (bv_num_cols_per_wg,), Float32)

                    # db += sum(dvb * v, axis=1)
                    mbarrier_wait(bar_tma_v_ptr + vloop_stage_idx, vloop_phase)
                    rV_io = cute.make_rmem_tensor((bv_num_cols_per_wg,), self.io_dtype)
                    sV_raw_ptr_cur = cute.make_ptr(
                        self.io_dtype,
                        sV_ptr_base + vloop_stage_idx * v_opB_bytes_per_stage,
                        cute.AddressSpace.smem,
                    )
                    if row < sub_seq_len:
                        for i in cutlass.range_constexpr(bv_num_cols_per_wg // 8):
                            col_base = bv_col_base + i * 8
                            vals = smem_load_f16x8_sw128(
                                sV_raw_ptr_cur, row, col_base, self.io_dtype
                            )
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

                    mbarrier_arrive(bar_mma_cuda_v_ptr + vloop_stage_idx)

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

                    vloop_stage_idx = (vloop_stage_idx + 1) % self.vloop_stage
                vloop_phase ^= 1

                # gk_exp = exp2(g)
                pipeline_load_g.consumer_wait(load_g_consumer_state)
                # write to gn
                sGn[local_tidx] = sG_raw[(sub_seq_len - 1, local_tidx, 0)]

                # row-major load, match TMEM layout
                rG = cute.make_rmem_tensor((self.BK // 4,), self.g_dtype)
                if row < sub_seq_len:
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
                pipeline_mma_dq.consumer_wait(mma_dq_consumer_state)
                tcgen05_fence_after()
                dq_i32 = tcgen05_ld_32x32b(
                    bk_num_cols_per_wg, TMEM_DQ_ACC_OFF + wg_idx * bk_num_cols_per_wg
                )
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dq.consumer_release(mma_dq_consumer_state)
                mma_dq_consumer_state.advance()

                dq_f32 = reinterpret_cast(dq_i32, Int32, bk_num_cols_per_wg, Float32)
                dq_f32_val = TensorSSA(dq_f32, (bk_num_cols_per_wg,), Float32)

                rDq = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                rDq.store(dq_f32_val * rG_exp_val * Float32(self.scale))

                dq_f32_val_store = rDq.load()
                dq_i32_vec = reinterpret_cast(dq_f32_val_store, Float32, bk_num_cols_per_wg, Int32)
                # store to TMEM first to reduce register usage
                tcgen05_st_32x32b(
                    bk_num_cols_per_wg,
                    TMEM_DQ_SCALED_OFF + wg_idx * bk_num_cols_per_wg,
                    dq_i32_vec,
                )
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
                pipeline_mma_dw.consumer_wait(mma_dw_consumer_state)
                tcgen05_fence_after()
                dw_i32 = tcgen05_ld_32x32b(
                    bk_num_cols_per_wg, TMEM_DW_ACC_OFF + wg_idx * bk_num_cols_per_wg
                )
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dw.consumer_release(mma_dw_consumer_state)
                mma_dw_consumer_state.advance()

                # dw = -dw, convert to the I/O dtype, write to smem
                dw_f32 = reinterpret_cast(dw_i32, Int32, bk_num_cols_per_wg, Float32)
                dw_f32_val = TensorSSA(dw_f32, (bk_num_cols_per_wg,), Float32)

                dw_io_rmem = cute.make_rmem_tensor((bk_num_cols_per_wg,), self.io_dtype)
                if row < sub_seq_len:
                    dw_io_rmem.store((-dw_f32_val).to(self.io_dtype))
                else:
                    dw_io_rmem.fill(self.io_dtype(0.0))

                pipeline_prologue_dw.producer_acquire(prologue_dw_producer_state)
                # store 8 I/O values each time
                dw_smem_num_stores = bk_num_cols_per_wg // 8
                for i in cutlass.range_constexpr(dw_smem_num_stores):
                    col_base = bk_col_base + i * 8
                    chunk = cute.local_tile(dw_io_rmem, (8,), (i,))
                    smem_store_f16x8_sw128(sDw_raw_ptr, row, col_base, chunk, self.io_dtype)

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_prologue_dw.producer_commit(prologue_dw_producer_state)
                prologue_dw_producer_state.advance()

                pipeline_load_k.consumer_wait(load_k_consumer_state)
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

                pipeline_prologue_kg.producer_acquire(prologue_kg_producer_state)
                for i in cutlass.range_constexpr(self.BK // 4 // 8):
                    col_base = bk_col_base + i * 8
                    chunk_kg = cute.local_tile(rKG_io, (8,), (i,))
                    smem_store_f16x8_sw128(sK_raw_ptr, row, col_base, chunk_kg, self.io_dtype)

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_prologue_kg.producer_commit(prologue_kg_producer_state)
                prologue_kg_producer_state.advance()

                # wait for dkgb
                pipeline_mma_dkgb.consumer_wait(mma_dgkb_consumer_state)
                tcgen05_fence_after()
                dkgb_i32 = tcgen05_ld_32x32b(
                    bk_num_cols_per_wg, TMEM_DKGB_ACC_OFF + wg_idx * bk_num_cols_per_wg
                )
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dkgb.consumer_release(mma_dgkb_consumer_state)
                mma_dgkb_consumer_state.advance()

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
                self.cuda_wg_sync_barrier.arrive_and_wait()
                if warp_col_tile == 1 and row < sub_seq_len:
                    sDb[row, wg_idx] = sDb[row, wg_idx] + db_val
                self.cuda_wg_sync_barrier.arrive_and_wait()
                # store db to GMEM (WG0 only). Sum order is fixed (slot 0 + slot 1).
                if wg_idx == 0 and local_tidx < sub_seq_len:
                    db_sum = sDb[(local_tidx, 0)] + sDb[(local_tidx, 1)]
                    db_gmem[(tok_offset + tile_idx * self.BT + local_tidx, (i_hv, Int32(0)))] = (
                        db_sum
                    )

                # dk = dk * exp2(gn[None, :] - g)
                pipeline_mma_dk.consumer_wait(mma_dk_consumer_state)
                tcgen05_fence_after()
                dk_i32 = tcgen05_ld_32x32b(
                    bk_num_cols_per_wg, TMEM_DK_ACC_OFF + wg_idx * bk_num_cols_per_wg
                )
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dk.consumer_release(mma_dk_consumer_state)
                mma_dk_consumer_state.advance()

                dk_f32 = reinterpret_cast(dk_i32, Int32, bk_num_cols_per_wg, Float32)
                dk_f32_val = TensorSSA(dk_f32, (bk_num_cols_per_wg,), Float32)

                rDk = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(bk_num_cols_per_wg):
                        exp_g_gn = cute.exp2(
                            sGn[(bk_col_base + i,)] - rG_val[i],
                            fastmath=self.use_fast_math,
                        )
                        rDk[i] = dk_f32_val[i] * exp_g_gn
                else:
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
                self.cuda_wg_sync_barrier.arrive_and_wait()

                # dgk *= exp2(gn)
                if wg_idx == 0:
                    sDgk[(local_tidx,)] *= cute.exp2(
                        sGn[(local_tidx,)], fastmath=self.use_fast_math
                    )

                self.cuda_wg_sync_barrier.arrive_and_wait()
                if wg_idx == 0:
                    sum = Float32(0.0)
                    for r in cutlass.range(self.BT, unroll_full=True):
                        sum += sG_raw[(r, local_tidx, 0)]
                    sDgk[(local_tidx,)] += sum

                # dg1 = kg * dkgb * beta[:, None], can reuse kg RMEM
                rDg = cute.make_rmem_tensor((bk_num_cols_per_wg,), Float32)
                rDg.store(rKG_val * dkgb_f32_val * beta_val)

                pipeline_load_q.consumer_wait(load_q_consumer_state)
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

                self.cuda_wg_sync_barrier.arrive_and_wait()
                # dg = dg2 + m_last * dgk, GMEM store dg
                if row == sub_seq_len - 1:
                    for i in cutlass.range_constexpr(bk_num_cols_per_wg):
                        col = bk_col_base + i
                        rDg[i] += sDgk[(col,)]

                # Stage dg to SMEM first. A dedicated store warp later does
                # SMEM -> RMEM -> GMEM with store_256b, keeping GMEM store
                # address/vector live ranges out of the high-register CC path.
                pipeline_store_dg.producer_acquire(store_dg_producer_state)
                if row < sub_seq_len:
                    for i in cutlass.range_constexpr(bk_num_cols_per_wg // 4):
                        col_base = bk_col_base + i * 4
                        chunk_dg = cute.local_tile(rDg, (4,), (i,))
                        smem_store_f32x4_sw128(sG_raw_ptr, row, col_base, chunk_dg)

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_store_dg.producer_commit(store_dg_producer_state)
                store_dg_producer_state.advance()

                pipeline_load_g.consumer_release(load_g_consumer_state)
                load_g_consumer_state.advance()

                pipeline_mma_dA.consumer_wait(mma_dA_consumer_state)
                tcgen05_fence_after()
                dA_i32 = tcgen05_ld_32x32b(
                    bt_num_cols_per_wg, TMEM_DA_ACC_OFF + wg_idx * bt_num_cols_per_wg
                )
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_mma_dA.consumer_release(mma_dA_consumer_state)
                mma_dA_consumer_state.advance()
                # NOTE: only release k smem after dA finished, because kg reuses k smem in dA += dw @ kg^T
                pipeline_load_k.consumer_release(load_k_consumer_state)
                load_k_consumer_state.advance()

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

                pipeline_prologue_dA2.producer_acquire(prologue_dA2_producer_state)

                for i in cutlass.range_constexpr(bt_num_cols_per_wg // 8):
                    col_base = bt_col_base + i * 8
                    chunk_dA = cute.local_tile(rDA, (8,), (i,))
                    smem_store_f16x8_sw128(sQ_raw_ptr, row, col_base, chunk_dA, self.io_dtype)
                # notify dA2 = dA @ A
                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_prologue_dA2.producer_commit(prologue_dA2_producer_state)
                prologue_dA2_producer_state.advance()

                pipeline_load_beta.consumer_release(load_beta_consumer_state)
                load_beta_consumer_state.advance()

                # wait for dA2
                pipeline_mma_dA2.consumer_wait(mma_dA2_consumer_state)
                tcgen05_fence_after()
                dA2_i32 = tcgen05_ld_32x32b(
                    bt_num_cols_per_wg, TMEM_DA2_ACC_OFF + wg_idx * bt_num_cols_per_wg
                )
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                pipeline_prologue_dA3.producer_acquire(prologue_dA3_producer_state)
                # write dA2 to smem notify dA2 = A @ dA2
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

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_prologue_dA3.producer_commit(prologue_dA3_producer_state)
                prologue_dA3_producer_state.advance()

                # wait for dA2
                pipeline_mma_dA3.consumer_wait(mma_dA3_consumer_state)
                tcgen05_fence_after()
                dA3_i32 = tcgen05_ld_32x32b(
                    bt_num_cols_per_wg, TMEM_DA2_ACC_OFF + wg_idx * bt_num_cols_per_wg
                )
                tcgen05_fence_before()
                cute.arch.fence_view_async_tmem_load()

                # release mma dA2 after dA3 is finished, protect DA2 TMEM
                pipeline_mma_dA2.consumer_release(mma_dA2_consumer_state)
                mma_dA2_consumer_state.advance()
                pipeline_mma_dA3.consumer_release(mma_dA3_consumer_state)
                mma_dA3_consumer_state.advance()
                # NOTE: release smem Q because we reuse to store input-dtype dA
                pipeline_load_q.consumer_release(load_q_consumer_state)
                load_q_consumer_state.advance()

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

        # Load loop body
        elif warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            load_A_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.a_stage
            )
            load_dv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.vloop_stage
            )
            load_do_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.vloop_stage
            )
            load_vnew_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.vloop_stage
            )
            load_g_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kloop_stage
            )
            load_k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kloop_stage
            )
            load_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.kloop_stage
            )

            vloop_stage_idx = 0
            vloop_phase = 1  # init as 1 for producer
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                G = HV // H
                i_t = work_idx // HV  # chunk index (global)
                i_hv = work_idx % HV  # value-head index
                i_h = i_hv // G  # q/k head index

                tok_offset, tile_idx, sub_seq_len = _resolve_chunk_route(
                    cu_seqlens,
                    chunk_offsets,
                    Int32(i_t),
                )

                # Load A
                tma_A_v = cute.domain_offset((0, tok_offset, (0, 0)), tma_tensor_A)
                tAsA, tAgA = self._tma_partition_A(
                    tma_atom_A,
                    tma_A_v,
                    sA,
                    self.dvb_tiler,  # [BT, BV, BT]
                    dvb_tiled_mma,
                    Int32(0),
                    i_hv,
                )
                pipeline_load_A.producer_acquire(load_A_producer_state)
                cute.copy(
                    tma_atom_A,
                    tAgA[(None, 0, tile_idx)],
                    tAsA[(None, load_A_producer_state.index)],
                    tma_bar_ptr=pipeline_load_A.producer_get_barrier(load_A_producer_state),
                )
                load_A_producer_state.advance()

                # V-loop
                for v_iter in cutlass.range(self.num_v_tiles):
                    tma_h_v = cute.domain_offset((0, v_iter * self.BV, (0, 0)), tma_tensor_h)
                    tHsH, tHgH = self._tma_partition_B(
                        tma_atom_h,
                        tma_h_v,
                        sH,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        i_hv,
                        i_t,
                    )
                    mbarrier_wait(bar_mma_cuda_h_ptr + vloop_stage_idx, vloop_phase)
                    with elect_one():
                        mbarrier_arrive_and_expect_tx(
                            bar_tma_h_ptr + vloop_stage_idx, self.tma_bytes_h
                        )
                    cute.copy(
                        tma_atom_h,
                        tHgH[(None, 0, 0)],
                        tHsH[(None, vloop_stage_idx)],
                        tma_bar_ptr=bar_tma_h_ptr + vloop_stage_idx,
                    )

                    tma_dh_v = cute.domain_offset((0, v_iter * self.BV, (0, 0)), tma_tensor_dh)
                    tDHsDH, tDHgDH = self._tma_partition_B(
                        tma_atom_dh,
                        tma_dh_v,
                        sDh,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        i_hv,
                        i_t,
                    )
                    mbarrier_wait(bar_mma_cuda_dh_ptr + vloop_stage_idx, vloop_phase)
                    with elect_one():
                        mbarrier_arrive_and_expect_tx(
                            bar_tma_dh_ptr + vloop_stage_idx, self.tma_bytes_dh
                        )
                    cute.copy(
                        tma_atom_dh,
                        tDHgDH[(None, 0, 0)],
                        tDHsDH[(None, vloop_stage_idx)],
                        tma_bar_ptr=bar_tma_dh_ptr + vloop_stage_idx,
                    )

                    tma_do_v = cute.domain_offset(
                        (tok_offset, v_iter * self.BV, (0, 0)), tma_tensor_do
                    )
                    tDOsDo, tDOgDo = self._tma_partition_A(
                        tma_atom_do,
                        tma_do_v,
                        sDo,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        Int32(0),
                        i_hv,
                    )
                    pipeline_load_do.producer_acquire(load_do_producer_state)
                    cute.copy(
                        tma_atom_do,
                        tDOgDo[(None, tile_idx, 0)],
                        tDOsDo[(None, vloop_stage_idx)],
                        tma_bar_ptr=pipeline_load_do.producer_get_barrier(load_do_producer_state),
                    )
                    load_do_producer_state.advance()

                    tma_dv_v = cute.domain_offset(
                        (tok_offset, v_iter * self.BV, (0, 0)), tma_tensor_dv
                    )
                    tDVsDv, tDVgDV = self._tma_partition_A(
                        tma_atom_dv,
                        tma_dv_v,
                        sDv,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        Int32(0),
                        i_hv,
                    )
                    pipeline_load_dv.producer_acquire(load_dv_producer_state)
                    cute.copy(
                        tma_atom_dv,
                        tDVgDV[(None, tile_idx, 0)],
                        tDVsDv[(None, vloop_stage_idx)],
                        tma_bar_ptr=pipeline_load_dv.producer_get_barrier(load_dv_producer_state),
                    )
                    load_dv_producer_state.advance()

                    tma_v_v = cute.domain_offset(
                        (tok_offset, v_iter * self.BV, (0, 0)), tma_tensor_v
                    )
                    tVsV, tVgV = self._tma_partition_B(
                        tma_atom_v,
                        tma_v_v,
                        sV,
                        self.dA_vloop_tiler,  # [BT, BT, BV]
                        dA_vloop_tiled_mma,
                        Int32(0),
                        i_hv,
                    )
                    mbarrier_wait(bar_mma_cuda_v_ptr + vloop_stage_idx, vloop_phase)
                    with elect_one():
                        mbarrier_arrive_and_expect_tx(
                            bar_tma_v_ptr + vloop_stage_idx, self.tma_bytes_v
                        )
                    cute.copy(
                        tma_atom_v,
                        tVgV[(None, tile_idx, 0)],
                        tVsV[(None, vloop_stage_idx)],
                        tma_bar_ptr=bar_tma_v_ptr + vloop_stage_idx,
                    )

                    # load v_new
                    tma_vnew_v = cute.domain_offset(
                        (tok_offset, v_iter * self.BV, (0, 0)), tma_tensor_vnew
                    )
                    tVnewsVnew, tVnewgVnew = self._tma_partition_A(
                        tma_atom_vnew,
                        tma_vnew_v,
                        sVnew,
                        self.vloop_gemm_tiler,  # [BT, BK, BV]
                        vloop_tiled_mma,
                        Int32(0),
                        i_hv,
                    )
                    pipeline_load_vnew.producer_acquire(load_vnew_producer_state)
                    cute.copy(
                        tma_atom_vnew,
                        tVnewgVnew[(None, tile_idx, 0)],
                        tVnewsVnew[(None, vloop_stage_idx)],
                        tma_bar_ptr=pipeline_load_vnew.producer_get_barrier(
                            load_vnew_producer_state
                        ),
                    )
                    load_vnew_producer_state.advance()

                    vloop_stage_idx = (vloop_stage_idx + 1) % self.vloop_stage
                vloop_phase ^= 1

                # Load g
                tma_g_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_g)
                tGsG, tGgG = self._epilog_partition_varlen(
                    tma_atom_g,
                    tma_g_v[None, None, (i_hv, Int32(0))],
                    (self.BT, self.BK),
                    sG_raw,
                )
                pipeline_load_g.producer_acquire(load_g_producer_state)
                cute.copy(
                    tma_atom_g,
                    tGgG[(None, tile_idx, 0)],
                    tGsG[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                    tma_bar_ptr=pipeline_load_g.producer_get_barrier(load_g_producer_state),
                )
                load_g_producer_state.advance()

                # Load k
                tma_k_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_k)
                tKsK, tKgK = self._epilog_partition_varlen(
                    tma_atom_k,
                    tma_k_v[None, None, (i_h, Int32(0))],
                    (self.BT, self.BK),
                    sK_raw,
                )
                pipeline_load_k.producer_acquire(load_k_producer_state)
                cute.copy(
                    tma_atom_k,
                    tKgK[(None, tile_idx, 0)],
                    tKsK[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                    tma_bar_ptr=pipeline_load_k.producer_get_barrier(load_k_producer_state),
                )
                load_k_producer_state.advance()

                tma_q_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_q)
                tQsQ, tQgQ = self._epilog_partition_varlen(
                    tma_atom_q,
                    tma_q_v[None, None, (i_h, Int32(0))],
                    (self.BT, self.BK),
                    sQ_raw,
                )
                pipeline_load_q.producer_acquire(load_q_producer_state)
                cute.copy(
                    tma_atom_q,
                    tQgQ[(None, tile_idx, 0)],
                    tQsQ[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                    tma_bar_ptr=pipeline_load_q.producer_get_barrier(load_q_producer_state),
                )
                load_q_producer_state.advance()

        # MMA loop body
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_others)

            load_A_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.a_stage
            )
            load_dv_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.vloop_stage
            )
            mma_dvb_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            load_do_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.vloop_stage
            )
            load_vnew_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.vloop_stage
            )
            mma_dq_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            mma_dk_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            mma_dw_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            prologue_dw_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kloop_stage
            )
            prologue_kg_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kloop_stage
            )
            mma_dgkb_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            mma_dA_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            mma_dA2_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            mma_dA3_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_stage
            )
            prologue_dA2_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )
            prologue_dA3_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_stage
            )

            vloop_stage_idx = 0
            a_stage_idx = 0
            mma_vloop_phase = 0
            vloop_phase = 0
            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                G = HV // H
                i_t = work_idx // HV  # chunk index (global)
                i_hv = work_idx % HV  # value-head index (unused in MMA warp)
                i_h = i_hv // G  # q/k head index (unused in MMA warp)

                tok_offset, tile_idx, sub_seq_len = _resolve_chunk_route(
                    cu_seqlens,
                    chunk_offsets,
                    Int32(i_t),
                )

                zeros8 = cute.make_rmem_tensor((8,), dtype=self.io_dtype)
                zeros8.fill(self.io_dtype(0.0))

                pipeline_load_A.consumer_wait(load_A_consumer_state)
                sA_raw_ptr = cute.make_ptr(
                    self.io_dtype,
                    sA_ptr_base + a_stage_idx * A_bytes_per_stage,
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

                for v_iter in cutlass.range(self.num_v_tiles):
                    is_accum = v_iter != 0
                    mbarrier_wait(bar_tma_h_ptr + vloop_stage_idx, vloop_phase)
                    pipeline_load_do.consumer_wait(load_do_consumer_state)
                    sDo_raw_ptr = cute.make_ptr(
                        self.io_dtype,
                        sDo_ptr_base + vloop_stage_idx * vloop_opA_bytes_per_stage,
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

                    if v_iter == 0:
                        pipeline_mma_dq.producer_acquire(mma_dq_producer_state)

                    # dq+=do@h
                    sDo_k_cur = sDo_k[(None, None, None, vloop_stage_idx)]
                    sH_k_cur = sH_k[(None, None, None, vloop_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sDo_k_cur.iterator, sDo_k_cur.layout, "k")
                    )
                    desc_b_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sH_k_cur.iterator, sH_k_cur.layout, "k")
                    )
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_call(
                        vloop_opA_smem,
                        desc_a_base,
                        vloop_opB_smem,
                        desc_b_base,
                        TMEM_DQ_ACC_OFF,
                        self.BV,
                        self.idesc_m64n128_k_k,
                        is_accum,
                    )

                    pipeline_load_do.consumer_release(load_do_consumer_state)
                    load_do_consumer_state.advance()

                    if v_iter == self.num_v_tiles - 1:
                        pipeline_mma_dq.producer_commit(mma_dq_producer_state)
                        mma_dq_producer_state.advance()

                    pipeline_load_dv.consumer_wait(load_dv_consumer_state)
                    sDv_raw = cute.make_ptr(
                        self.io_dtype,
                        sDv_ptr_base + vloop_stage_idx * vloop_opA_bytes_per_stage,
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

                    # if lane_idx == 0:
                    #     cute.printf("V_iter", v_iter)
                    #     cute.print_tensor(sDv[None, None, None, vloop_stage_idx])
                    pipeline_mma_dvb.producer_acquire(mma_dvb_producer_state)
                    sDv_mn_cur = sDv_mn[(None, None, None, vloop_stage_idx)]
                    sA_mn_cur = sA_mn[(None, None, None, a_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn")
                    )
                    desc_b_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sDv_mn_cur.iterator, sDv_mn_cur.layout, "mn")
                    )
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_call(
                        A_mn_opA_smem,
                        desc_a_base,
                        dv_mn_opB_smem,
                        desc_b_base,
                        TMEM_FLEX_OFF,
                        self.BT,
                        self.idesc_m64n64_mn_mn,
                    )

                    pipeline_mma_dvb.producer_commit(mma_dvb_producer_state)
                    mma_dvb_producer_state.advance()

                    # dw += dv @ h
                    if v_iter == 0:
                        pipeline_mma_dw.producer_acquire(mma_dw_producer_state)

                    sDv_k_cur = sDv_k[(None, None, None, vloop_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sDv_k_cur.iterator, sDv_k_cur.layout, "k")
                    )
                    desc_b_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sH_k_cur.iterator, sH_k_cur.layout, "k")
                    )
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_call(
                        vloop_opA_smem,
                        desc_a_base,
                        vloop_opB_smem,
                        desc_b_base,
                        TMEM_DW_ACC_OFF,
                        self.BV,
                        self.idesc_m64n128_k_k,
                        is_accum,
                    )

                    # dA += dv @ v^T
                    mbarrier_wait(bar_tma_v_ptr + vloop_stage_idx, vloop_phase)
                    sV_raw = cute.make_ptr(
                        self.io_dtype,
                        sV_ptr_base + vloop_stage_idx * v_opB_bytes_per_stage,
                        cute.AddressSpace.smem,
                    )
                    if sub_seq_len < self.BT:
                        for i in cutlass.range_constexpr(self.BT // 32):
                            row = i * 32 + lane_idx
                            if row >= sub_seq_len:
                                for col in cutlass.range_constexpr(self.BV // 8):
                                    # dv tile uses the same Swizzle<3,4,3> physical mapping.
                                    smem_store_f16x8_sw128(
                                        sV_raw, row, col * 8, zeros8, self.io_dtype
                                    )
                        cute.arch.fence_proxy("async.shared", space="cta")

                    if v_iter == 0:
                        pipeline_mma_dA.producer_acquire(mma_dA_producer_state)

                    sV_k_cur = sV_k[(None, None, None, vloop_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sDv_k_cur.iterator, sDv_k_cur.layout, "k")
                    )
                    desc_b_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sV_k_cur.iterator, sV_k_cur.layout, "k")
                    )
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_call(
                        vloop_opA_smem,
                        desc_a_base,
                        v_opB_smem,
                        desc_b_base,
                        TMEM_DA_ACC_OFF,
                        self.BV,
                        self.idesc_m64n64_k_k,
                        is_accum,
                    )

                    # dv pipeline calls tcgen05.commit for dv@h and dv@v^T
                    pipeline_load_dv.consumer_release(load_dv_consumer_state)
                    load_dv_consumer_state.advance()

                    if v_iter == self.num_v_tiles - 1:
                        pipeline_mma_dw.producer_commit(mma_dw_producer_state)
                        mma_dw_producer_state.advance()

                    umma_arrive(bar_mma_cuda_h_ptr + vloop_stage_idx)
                    umma_arrive(bar_mma_cuda_v_ptr + vloop_stage_idx)

                    # dk += v_new @ dh
                    pipeline_load_vnew.consumer_wait(load_vnew_consumer_state)
                    sDvnew_raw_ptr = cute.make_ptr(
                        self.io_dtype,
                        sVnew_ptr_base + vloop_stage_idx * vloop_opA_bytes_per_stage,
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

                    mbarrier_wait(bar_tma_dh_ptr + vloop_stage_idx, vloop_phase)
                    if v_iter == 0:
                        pipeline_mma_dk.producer_acquire(mma_dk_producer_state)

                    sVnew_k_cur = sVnew_k[(None, None, None, vloop_stage_idx)]
                    sDh_k_cur = sDh_k[(None, None, None, vloop_stage_idx)]
                    desc_a_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sVnew_k_cur.iterator, sVnew_k_cur.layout, "k")
                    )
                    desc_b_i64 = smem_descriptor_to_int(
                        make_umma_smem_desc(sDh_k_cur.iterator, sDh_k_cur.layout, "k")
                    )
                    desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                    desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                    mma_ws_ss_call(
                        vloop_opA_smem,
                        desc_a_base,
                        vloop_opB_smem,
                        desc_b_base,
                        TMEM_DK_ACC_OFF,
                        self.BV,
                        self.idesc_m64n128_k_k,
                        is_accum,
                    )

                    # vnew pipeline calls tcgen05.commit
                    pipeline_load_vnew.consumer_release(load_vnew_consumer_state)
                    load_vnew_consumer_state.advance()

                    if v_iter == self.num_v_tiles - 1:
                        pipeline_mma_dk.producer_commit(mma_dk_producer_state)
                        mma_dk_producer_state.advance()

                    umma_arrive(bar_mma_cuda_dh_ptr + vloop_stage_idx)

                    # add tcgen05.commit and mbar.wait to make sure dq/dk/dw MMA finished
                    umma_arrive(bar_mma_done_vloop_ptr + 0)
                    mbarrier_wait(bar_mma_done_vloop_ptr + 0, mma_vloop_phase)
                    mma_vloop_phase ^= 1

                    vloop_stage_idx = (vloop_stage_idx + 1) % self.vloop_stage
                vloop_phase ^= 1

                pipeline_prologue_dw.consumer_wait(prologue_dw_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")
                # dkgb = A @ dw
                pipeline_mma_dkgb.producer_acquire(mma_dgkb_producer_state)
                sA_mn_cur = sA_mn[(None, None, None, a_stage_idx)]
                sDw_mn_cur = sDw_mn[(None, None, None, 0)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDw_mn_cur.iterator, sDw_mn_cur.layout, "mn")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    A_mn_opA_smem,
                    desc_a_base,
                    dw_mn_opB_smem,
                    desc_b_base,
                    TMEM_DKGB_ACC_OFF,
                    self.BT,
                    self.idesc_m64n128_mn_mn,
                )

                pipeline_mma_dkgb.producer_commit(mma_dgkb_producer_state)
                mma_dgkb_producer_state.advance()

                pipeline_prologue_kg.consumer_wait(prologue_kg_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")
                # dA += dw @ kg^T
                sDw_k_cur = sDw_k[(None, None, None, 0)]
                sKG_k_cur = sKG_k[(None, None, None, 0)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDw_k_cur.iterator, sDw_k_cur.layout, "k")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sKG_k_cur.iterator, sKG_k_cur.layout, "k")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    dw_k_opA_smem,
                    desc_a_base,
                    kg_k_opB_smem,
                    desc_b_base,
                    TMEM_DA_ACC_OFF,
                    self.BK,
                    self.idesc_m64n64_k_k,
                    True,
                )

                pipeline_mma_dA.producer_commit(mma_dA_producer_state)
                mma_dA_producer_state.advance()
                pipeline_prologue_kg.consumer_release(prologue_kg_consumer_state)
                prologue_kg_consumer_state.advance()

                pipeline_prologue_dw.consumer_release(prologue_dw_consumer_state)
                prologue_dw_consumer_state.advance()

                # dA2 = dA @ A
                pipeline_mma_dA2.producer_acquire(mma_dA2_producer_state)
                pipeline_prologue_dA2.consumer_wait(prologue_dA2_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")

                sDA_k_cur = sDA_k[(None, None, None, 0)]
                sA_k_cur = sA_k[(None, None, None, a_stage_idx)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDA_k_cur.iterator, sDA_k_cur.layout, "k")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sA_k_cur.iterator, sA_k_cur.layout, "k")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    dA_k_opA_smem,
                    desc_a_base,
                    A_k_opB_smem,
                    desc_b_base,
                    TMEM_DA2_ACC_OFF,
                    self.BT,
                    self.idesc_m64n64_k_k,
                )

                pipeline_mma_dA2.producer_commit(mma_dA2_producer_state)
                mma_dA2_producer_state.advance()
                pipeline_prologue_dA2.consumer_release(prologue_dA2_consumer_state)
                prologue_dA2_consumer_state.advance()

                # dA3 = A @ dA2
                pipeline_mma_dA3.producer_acquire(mma_dA3_producer_state)
                pipeline_prologue_dA3.consumer_wait(prologue_dA3_consumer_state)
                cute.arch.fence_proxy("async.shared", space="cta")

                sA_mn_cur = sA_mn[(None, None, None, a_stage_idx)]
                sDA_mn_cur = sDA_mn[(None, None, None, 0)]
                desc_a_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sA_mn_cur.iterator, sA_mn_cur.layout, "mn")
                )
                desc_b_i64 = smem_descriptor_to_int(
                    make_umma_smem_desc(sDA_mn_cur.iterator, sDA_mn_cur.layout, "mn")
                )
                desc_a_base = Tcgen05SmemDescriptor(desc_a_i64)
                desc_b_base = Tcgen05SmemDescriptor(desc_b_i64)
                mma_ws_ss_call(
                    A_mn_opA_smem,
                    desc_a_base,
                    dA_mn_opB_smem,
                    desc_b_base,
                    TMEM_DA2_ACC_OFF,
                    self.BT,
                    self.idesc_m64n64_mn_mn,
                )

                pipeline_mma_dA3.producer_commit(mma_dA3_producer_state)
                mma_dA3_producer_state.advance()
                pipeline_prologue_dA3.consumer_release(prologue_dA3_consumer_state)
                prologue_dA3_consumer_state.advance()

                pipeline_load_A.consumer_release(load_A_consumer_state)
                load_A_consumer_state.advance()

                a_stage_idx = (a_stage_idx + 1) % self.a_stage

        # Load aux loop body
        elif warp_idx in self.aux_warp_ids:
            cute.arch.setmaxregister_decrease(self.num_regs_others)
            tidx = thread_idx - (self.threads_per_cta - 64)

            load_beta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            load_g_store_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kloop_stage
            )
            store_dg_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.kloop_stage
            )

            for wu_iter in cutlass.range(0, num_iters, unroll=0):
                work_idx = block_idx_x + wu_iter * grid_dim_x
                G = HV // H
                i_t = work_idx // HV  # chunk index (global)
                i_hv = work_idx % HV  # value-head index
                i_h = i_hv // G  # q/k head index (unused in aux warp)

                tok_offset, tile_idx, sub_seq_len = _resolve_chunk_route(
                    cu_seqlens,
                    chunk_offsets,
                    Int32(i_t),
                )

                pipeline_load_beta.producer_acquire(load_beta_producer_state)
                beta_f32 = Float32(0.0)
                if tidx < sub_seq_len:
                    beta_f32 = Float32(
                        beta_gmem[(tok_offset + tile_idx * self.BT + tidx, (i_hv, Int32(0)))]
                    )
                sBeta[(tidx,)] = beta_f32

                cute.arch.fence_proxy("async.shared", space="cta")
                pipeline_load_beta.producer_commit(load_beta_producer_state)
                load_beta_producer_state.advance()

                pipeline_load_g.consumer_wait(load_g_store_consumer_state)
                pipeline_store_dg.consumer_wait(store_dg_consumer_state)

                tma_dg_v = cute.domain_offset((tok_offset, 0, (0, 0)), tma_tensor_dg)
                tDGsDG, tDGgDG = self._epilog_partition_varlen(
                    tma_atom_dg,
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
                            dg_store_rmem[0] = vals0[0]
                            dg_store_rmem[1] = vals0[1]
                            dg_store_rmem[2] = vals0[2]
                            dg_store_rmem[3] = vals0[3]
                            dg_store_rmem[4] = vals1[0]
                            dg_store_rmem[5] = vals1[1]
                            dg_store_rmem[6] = vals1[2]
                            dg_store_rmem[7] = vals1[3]
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
                        tma_atom_dg,
                        tDGsDG[(None, 0)],  # hardcode stage to 0 because kloop_stage is 1
                        tDGgDG[(None, tile_idx, 0)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)

                pipeline_store_dg.consumer_release(store_dg_consumer_state)
                store_dg_consumer_state.advance()
                pipeline_load_g.consumer_release(load_g_store_consumer_state)
                load_g_store_consumer_state.advance()

        # ===================== TMEM cleanup =====================
        tmem.relinquish_alloc_permit()
        self.tmem_dealloc_sync_barrier.arrive_and_wait()
        tmem.free(tmem_ptr, TMEM_TOTAL)

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
    grid_waves: int,
    ragged: bool,
    use_int64_offsets: bool = False,
):
    """Compile one persistent dense or ragged WY/dQKG specialization."""
    cutlass_io_dtype = _torch_to_cutlass_dtype[io_dtype]
    op = ChunkKdaBwdWyDqkgFused(
        chunk_size=chunk_size,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        io_dtype=cutlass_io_dtype,
        scale=scale,
        grid_waves=grid_waves,
        use_fast_math=fastmath,
        use_int64_offsets=use_int64_offsets,
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

    q = strided_io_tensor((1, tokens, heads, head_dim))
    k = strided_io_tensor((1, tokens, heads, head_dim))
    v = strided_io_tensor((1, tokens, heads, head_dim))
    v_new = tensor(cutlass_io_dtype, (1, tokens, heads, head_dim))
    g = tensor(cutlass.Float32, (1, tokens, heads, head_dim))
    beta = tensor(cutlass.Float32, (1, tokens, heads))
    A = tensor(cutlass_io_dtype, (1, tokens, heads, chunk_size))
    h = tensor(cutlass_io_dtype, (1, chunks, heads, head_dim, head_dim))
    do = tensor(cutlass_io_dtype, (1, tokens, heads, head_dim))
    dh = tensor(cutlass_io_dtype, (1, chunks, heads, head_dim, head_dim))
    dv = tensor(cutlass_io_dtype, (1, tokens, heads, head_dim))
    dq = tensor(cutlass.Float32, (1, tokens, heads, head_dim))
    dk = tensor(cutlass.Float32, (1, tokens, heads, head_dim))
    dv2 = tensor(cutlass_io_dtype, (1, tokens, heads, head_dim))
    dg = tensor(cutlass.Float32, (1, tokens, heads, head_dim))
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
        (Int32(1), Int32(1), Int32(heads), Int32(heads), Int32(head_dim), Int32(head_dim)),
        Int32(1),
        name=(
            f"kda_bwd_wy_dqkg_h{heads}_d{head_dim}_c{chunk_size}_"
            f"{str(io_dtype).removeprefix('torch.')}_fm{int(fastmath)}_"
            f"gw{grid_waves}_rg{int(ragged)}_i64{int(use_int64_offsets)}"
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
        """Key winners by the static persistent-grid work envelope."""
        if target.sm_count is None:
            raise RuntimeError("KDA tuning requires a CUDA target with an SM count")
        return (args.h.shape[1] * args.q.shape[2],)

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
    ) -> tuple[int, int, int, torch.dtype, float, bool, int, bool, bool]:
        return (
            args.q.shape[2],
            args.q.shape[3],
            args.chunk_size,
            args.q.dtype,
            args.scale,
            args.fastmath,
            config.grid_waves,
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
        del config
        _batch, tokens, heads, head_dim = args.q.shape
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
                Int32(heads),
                Int32(heads),
                Int32(head_dim),
                Int32(head_dim),
            ),
            Int32(args.h.shape[1]),
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
    """Run or tune the direct dense or ragged fused WY/dQKG backward stage."""
    batch, tokens, heads, head_dim = q.shape
    if batch != 1 or head_dim != 128 or v.shape[-1] != 128:
        raise ValueError("the fused WY backward requires B=1 and K=V=128")
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
    if tokens == 0:
        return (
            torch.empty_like(g, memory_format=torch.contiguous_format),
            torch.empty_like(g, memory_format=torch.contiguous_format),
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
        dq=torch.empty_like(g, memory_format=torch.contiguous_format),
        dk=torch.empty_like(g, memory_format=torch.contiguous_format),
        dv2=torch.empty_like(v, memory_format=torch.contiguous_format),
        # Dense BT64 writes every dg element. Ragged execution preserves the
        # zero initialization required by predicated tails and sanitizer runs.
        dg=(
            torch.empty_like(g, memory_format=torch.contiguous_format)
            if metadata is None
            else torch.zeros_like(g, memory_format=torch.contiguous_format)
        ),
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
