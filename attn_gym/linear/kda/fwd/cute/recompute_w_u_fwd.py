# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CuTe DSL implementation of recompute_w_u_fwd (SM100 / Blackwell).
#
# This is the narrow production CuTe path for varlen B=1 full chunks:
# BT=64 and head_dim K=V=128. It recomputes:
#   w  = A @ (k * beta * exp2(gk))  or A @ (k * beta) when gk is absent
#   u  = A @ (v * beta)
#   qg = q * exp2(gk)               when both q and gk are present
#   kg = k * exp2(gk_last - gk)     when gk is present
#
# The grid is persistent over runtime (chunk, head) work items. num_chunks is a
# caller-owned CUDA scalar loaded in-kernel, so CUDA graph replay can change the
# active chunk count without recompiling or reallocating metadata.
#
# ----------------------------------------------------------------------------
# MMA input precision (dot_precision knob)
# ----------------------------------------------------------------------------
# The accumulator is always fp32 in TMEM; only the *operand* precision of the
# W = A @ KB and U = A @ VB tensor-core dots changes:
#
#   "bf16"  (default) -- operands cast to bf16; tcgen05.mma.kind::f16, K-inst 16,
#                        8 mantissa bits. Fastest: fewest MMA issues and 2-byte
#                        SMEM operands. Matches what Triton actually runs (below).
#   "tf32"            -- operands cast to tf32; tcgen05.mma.kind::tf32, K-inst 8,
#                        10 mantissa bits. More accurate than bf16 at a higher
#                        MMA cost (twice the K-issues).
#   "tf32x3"          -- tf32 base plus the two first-order residual products
#                        (A_hi@V_hi + A_hi@V_lo + A_lo@V_hi) on the U dot,
#                        emulating ~fp32 accuracy. ONLY valid when A is fp32 --
#                        once A is bf16-rounded the residual is meaningless, so
#                        the wrapper rejects tf32x3 for non-fp32 A. W stays
#                        single-pass tf32 even in this mode.
#
# What Triton does, by contrast: the shipped Triton kernel casts both dot
# operands to A.dtype before tl.dot(..., input_precision="tf32x3"). When A is
# bf16 (the common case) that cast forces a bf16 tensor-core op and the
# input_precision request is silently ignored -- so Triton effectively runs
# bf16 regardless of its config. tf32/tf32x3 only take effect for Triton when A
# is fp32. CuTe can run any of the three precisions independently of the A
# storage dtype, and feeds the MMA without the intermediate bf16 round-trip
# Triton applies to k*beta / v*beta (so CuTe tf32 is strictly more accurate
# than Triton's "tf32x3"-on-bf16-A, which is really bf16).
#
# W and U occupy disjoint TMEM column regions: W at cols [0:128], U at
# cols [128:256].

# pyre-ignore-all-errors
# NOTE: do NOT use `from __future__ import annotations` — cute.struct
# requires eager-evaluated annotations.

import enum

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Boolean, Float32, Int32, cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cutlass_dsl import Constexpr
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.target import get_compile_target

# ============================================================================
# Constants
# ============================================================================

DATA_ALIGN_BYTES = 16
INDEX_ALIGN_BYTES = 4

# I/O dtype: gmem inputs K/V/Q and outputs W/U/qg/kg are bf16.
IO_DTYPE = cutlass.BFloat16
# MMA operand dtype is chosen at launch from dot_precision (see _mma_config):
# tf32 (4 bytes, K-inst 8) for "tf32"/"tf32x3", or bf16 (2 bytes, K-inst 16) for
# "bf16". Both accumulate in fp32 with CtaGroup.ONE. The values below are the
# tf32 defaults used to size layouts/comments; the bf16 variant uses K-inst 16.
MMA_DTYPE = cutlass.TFloat32
ACC_DTYPE = cutlass.Float32

# MMA instruction shape -- K=8 for tf32 (kind::tf32), K=16 for bf16 (kind::f16).
# N=128 covers a single operand (W OR U, not both). Two independent MMAs are
# issued (W = A@KB and U = A@VB) targeting non-overlapping TMEM regions:
#   W accumulator: TMEM rows [0:64], cols [0:128]
#   U accumulator: TMEM rows [0:64], cols [128:256]
# Row bias was tried for U, but UMMA did not honor it; column offsetting keeps
# the accumulator regions disjoint.
MMA_INST_SHAPE_MNK = (64, 128, 8)
MMA_INST_SHAPE_MNK_BF16 = (64, 128, 16)
# Block tile for the GEMM -- (M=BT=64, N=128, K_red=BT=64). K_red=64 is reduced
# with (64 / K-inst) MMA K-issues per operand: 8 issues for tf32, 4 for bf16.
MMA_TILER_MNK = (64, 128, 64)
MMA_K_INST = MMA_INST_SHAPE_MNK[2]
MMA_K_INST_BF16 = MMA_INST_SHAPE_MNK_BF16[2]


class MmaPrecision(enum.Enum):
    """MMA operand precision knob (see the module header for the trade-offs)."""

    BF16 = "bf16"  # bf16 operands, kind::f16 MMA, K-inst 16. Fastest.
    TF32 = "tf32"  # tf32 operands, kind::tf32 MMA, K-inst 8. More accurate.
    TF32X3 = "tf32x3"  # tf32 + residual products on U; fp32 A only.

    @property
    def is_bf16(self) -> bool:
        return self is MmaPrecision.BF16

    @property
    def is_tf32x3(self) -> bool:
        return self is MmaPrecision.TF32X3


def _normalize_precision(precision: "str | MmaPrecision") -> MmaPrecision:
    if isinstance(precision, MmaPrecision):
        return precision
    try:
        return MmaPrecision(precision)
    except ValueError:
        valid = ", ".join(repr(p.value) for p in MmaPrecision)
        raise AssertionError(
            f"recompute_w_u_fwd dot_precision must be one of {valid}; got {precision!r}."
        )


def _mma_config(precision: MmaPrecision):
    """Return (op, mma_dtype) for the requested operand precision.

    The kernel derives its compile-time K-inst (MMA_K_INST{,_BF16}) directly
    from the precision, so this host helper only builds the op + operand dtype.
    """
    if precision.is_bf16:
        op = tcgen05.MmaF16BF16Op(
            cutlass.BFloat16,
            ACC_DTYPE,
            MMA_INST_SHAPE_MNK_BF16,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
        )
        return op, cutlass.BFloat16
    op = tcgen05.MmaTF32Op(
        MMA_INST_SHAPE_MNK,
        tcgen05.CtaGroup.ONE,
        tcgen05.OperandSource.SMEM,
        tcgen05.OperandMajorMode.K,
        tcgen05.OperandMajorMode.MN,
    )
    return op, MMA_DTYPE


THREADS_PER_CTA = 256
BT = 64  # chunk size
KEY_DIM = 128  # hardcoded head_dim for staging SMEM
VAL_DIM = 128

AB_STAGES = 1  # no K-tile pipelining (single K-tile per block)
# ACC_STAGE=2: split W and U as separate pipeline stages at disjoint TMEM
# regions (W at col 0, U at col 128). In the single-pass U path this lets
# EPI_W run while MMA_U is in flight; in the tf32x3 U path the same two stages
# are reused for U_hi, U_v_res, and U_a_res.
ACC_STAGE = 2


# ============================================================================
# Helpers
# ============================================================================


@cute.jit
def _predicate_valid(tCoord: cute.Tensor, valid: Int32, check_cols: Constexpr) -> cute.Tensor:
    # tCoord is the source-partitioned identity tile, shape ((v, 1), rest_m,
    # rest_n). rest_m is the row-repeat mode: a single thread owns several rows
    # (e.g. rows m, m+16, m+32, m+48). The predicate MUST be evaluated per
    # rest_m — broadcasting one row's predicate across all repeats would load
    # rows >= valid and over-read past the packed-token boundary. check_cols also
    # masks the column coord (for the square A tile, which is valid x valid).
    pred = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tCoord, mode=[0, 1]),
                cute.size(tCoord, mode=[1]),
                cute.size(tCoord, mode=[2]),
            ),
        ),
        Boolean,
    )
    for rest_v in cutlass.range_constexpr(pred.shape[0]):
        for rest_m in cutlass.range_constexpr(pred.shape[1]):
            for rest_k in cutlass.range_constexpr(pred.shape[2]):
                coord = tCoord[(0, rest_v), rest_m, rest_k]
                keep = coord[0] < valid
                if cutlass.const_expr(check_cols):
                    keep = keep & (coord[1] < valid)
                pred[rest_v, rest_m, rest_k] = keep
    return pred


@cute.jit
def _make_input_tiles(
    mK: cute.Tensor,
    mV: cute.Tensor,
    mQ: cute.Tensor,
    mG: cute.Tensor,
    mA: cute.Tensor,
    time_base: Int32,
    key_head_idx: Int32,
    value_head_idx: Int32,
):
    # Address the exact packed-token row. Document starts are not guaranteed to
    # be BT-aligned, so using time_base // BT would floor into a previous chunk.
    k_shifted = cute.domain_offset((0, time_base, key_head_idx, 0), mK)
    v_shifted = cute.domain_offset((0, time_base, value_head_idx, 0), mV)
    q_shifted = cute.domain_offset((0, time_base, value_head_idx, 0), mQ)
    g_shifted = cute.domain_offset((0, time_base, key_head_idx, 0), mG)
    a_shifted = cute.domain_offset((0, time_base, value_head_idx, 0), mA)
    gK_tile = cute.make_tensor(
        k_shifted.iterator,
        cute.make_layout((BT, KEY_DIM), stride=(mK.layout.stride[1], mK.layout.stride[3])),
    )
    gV_tile = cute.make_tensor(
        v_shifted.iterator,
        cute.make_layout((BT, VAL_DIM), stride=(mV.layout.stride[1], mV.layout.stride[3])),
    )
    gQ_tile = cute.make_tensor(
        q_shifted.iterator,
        cute.make_layout((BT, KEY_DIM), stride=(mQ.layout.stride[1], mQ.layout.stride[3])),
    )
    gG_tile = cute.make_tensor(
        g_shifted.iterator,
        cute.make_layout((BT, KEY_DIM), stride=(mG.layout.stride[1], mG.layout.stride[3])),
    )
    gA_tile = cute.make_tensor(
        a_shifted.iterator,
        cute.make_layout((BT, BT), stride=(mA.layout.stride[1], mA.layout.stride[3])),
    )
    return gK_tile, gV_tile, gQ_tile, gG_tile, gA_tile


@cute.jit
def _copy_input_tiles(
    tiled_copy_bf16: cute.TiledCopy,
    tiled_copy_fp32_kv: cute.TiledCopy,
    tiled_copy_a: cute.TiledCopy,
    thr_copy_bf16: cute.TiledCopy,
    thr_copy_fp32_kv: cute.TiledCopy,
    thr_copy_a: cute.TiledCopy,
    gK_tile: cute.Tensor,
    gV_tile: cute.Tensor,
    gQ_tile: cute.Tensor,
    gG_tile: cute.Tensor,
    gA_tile: cute.Tensor,
    sK_stage: cute.Tensor,
    sV_stage: cute.Tensor,
    sQ_stage: cute.Tensor,
    sG_stage: cute.Tensor,
    sA_stage: cute.Tensor,
    valid: Int32,
    has_q: Constexpr,
    has_gk: Constexpr,
):
    cKV = cute.make_identity_tensor((BT, KEY_DIM))
    cA = cute.make_identity_tensor((BT, BT))

    if valid >= BT:
        cute.copy(
            tiled_copy_bf16,
            thr_copy_bf16.partition_S(gK_tile),
            thr_copy_bf16.partition_D(sK_stage),
        )
        cute.copy(
            tiled_copy_bf16,
            thr_copy_bf16.partition_S(gV_tile),
            thr_copy_bf16.partition_D(sV_stage),
        )
        if cutlass.const_expr(has_q):
            cute.copy(
                tiled_copy_bf16,
                thr_copy_bf16.partition_S(gQ_tile),
                thr_copy_bf16.partition_D(sQ_stage),
            )
        if cutlass.const_expr(has_gk):
            cute.copy(
                tiled_copy_fp32_kv,
                thr_copy_fp32_kv.partition_S(gG_tile),
                thr_copy_fp32_kv.partition_D(sG_stage),
            )
        cute.arch.cp_async_commit_group()
        cute.copy(
            tiled_copy_a,
            thr_copy_a.partition_S(gA_tile),
            thr_copy_a.partition_D(sA_stage),
        )
    else:
        pred_bf16 = _predicate_valid(thr_copy_bf16.partition_S(cKV), valid, check_cols=False)
        cute.copy(
            tiled_copy_bf16,
            thr_copy_bf16.partition_S(gK_tile),
            thr_copy_bf16.partition_D(sK_stage),
            pred=pred_bf16,
        )
        cute.copy(
            tiled_copy_bf16,
            thr_copy_bf16.partition_S(gV_tile),
            thr_copy_bf16.partition_D(sV_stage),
            pred=pred_bf16,
        )
        if cutlass.const_expr(has_q):
            cute.copy(
                tiled_copy_bf16,
                thr_copy_bf16.partition_S(gQ_tile),
                thr_copy_bf16.partition_D(sQ_stage),
                pred=pred_bf16,
            )
        if cutlass.const_expr(has_gk):
            pred_g = _predicate_valid(thr_copy_fp32_kv.partition_S(cKV), valid, check_cols=False)
            cute.copy(
                tiled_copy_fp32_kv,
                thr_copy_fp32_kv.partition_S(gG_tile),
                thr_copy_fp32_kv.partition_D(sG_stage),
                pred=pred_g,
            )
        cute.arch.cp_async_commit_group()
        pred_a = _predicate_valid(thr_copy_a.partition_S(cA), valid, check_cols=True)
        cute.copy(
            tiled_copy_a,
            thr_copy_a.partition_S(gA_tile),
            thr_copy_a.partition_D(sA_stage),
            pred=pred_a,
        )
    cute.arch.cp_async_commit_group()


@cute.jit
def _store_epilogue_tile(
    src: cute.Tensor,
    dst: cute.Tensor,
    coord: cute.Tensor,
    valid: Int32,
):
    if valid >= BT:
        cute.autovec_copy(src, dst)
    else:
        for i in cutlass.range_constexpr(cute.size(src)):
            if coord[i][0] < valid:
                dst[i] = src[i]


# ============================================================================
# Shared storage
# ============================================================================


@cute.struct
class SharedStorage:
    """
    SMEM layout for the kernel.

    Fields:
    - acc_mbar_ptr: PipelineUmmaAsync mbarriers for W/U accumulator stages.
    - ab_mbar_ptr: kept for layout stability; CVT->MMA sync uses a named
        barrier because AB_STAGES=1 has no K-tile overlap.
    - tmem_holding_buf: holds the TMEM allocation pointer (32-bit).
    - sK/sV/sQ/sG/sA/sBeta staging: cp.async source tiles and per-row beta.
        MMA-ready swizzled A/KB/VB buffers are allocated dynamically below with
        allocate_tensor so their layouts match the tiled MMA descriptors.
    """

    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGE * 2]
    tmem_holding_buf: cutlass.Int32
    # cp.async staging: bf16 inputs K, V, Q and fp32 g/A.
    sK_staging: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, BT * KEY_DIM], 128]
    sV_staging: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, BT * VAL_DIM], 128]
    sQ_staging: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, BT * KEY_DIM], 128]
    sG_staging: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, BT * KEY_DIM], 128]
    sA_staging: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, BT * BT], 128]
    # Per-chunk beta broadcast buffer: one fp32 per row, read by all 32 col-threads.
    sBeta_staging: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, BT], 128]


# ============================================================================
# Kernel
# ============================================================================


@cute.kernel
def _kernel_varlen_b1_full_chunk(
    tiled_mma: cute.TiledMma,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mBeta: cute.Tensor,
    mA: cute.Tensor,
    mG: cute.Tensor,
    mW: cute.Tensor,
    mU: cute.Tensor,
    mQG: cute.Tensor,
    mKG: cute.Tensor,
    mCuSeqlens: cute.Tensor,
    mChunkIndices: cute.Tensor,
    mNumChunks: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
    num_persistent_ctas: Constexpr,
    a_is_fp32: Constexpr,
    has_q: Constexpr,
    has_gk: Constexpr,
    u_dot_precision_tf32x3: Constexpr,
    prefetch_next_tile: Constexpr,
    mma_is_bf16: Constexpr,
):
    """
    Kernel body for the UMMA path.

    Grid: (num_persistent_ctas, 1, 1). CTAs stride over runtime
    (chunk, head) work items, bounded by mNumChunks[0] loaded in-kernel.
    Block: (256, 1, 1).

    Life of a block:
      1. Resolve varlen chunk (seq_idx, chunk_idx, time_base) from metadata.
      2. Load source tiles (K, V, Q, beta, g, A) from gmem.
      3. CVT: build TF32 MMA operands A, kb, and vb in swizzled SMEM; store
         optional qg and kg directly to gmem.
      4. UMMA: compute W = A @ kb and U = A @ vb into separate TMEM columns.
         If requested for fp32 A, U is expanded to Triton-style tf32x3.
      5. Epilogue: copy TMEM -> rmem, cast to bf16, and store W/U to gmem.
    """
    # Operand precision is compile-time: bf16 (kind::f16, K-inst 16) or tf32
    # (kind::tf32, K-inst 8). The accumulator is fp32 either way.
    mma_dtype: Constexpr = cutlass.BFloat16 if mma_is_bf16 else MMA_DTYPE
    mma_k_inst: Constexpr = MMA_K_INST_BF16 if mma_is_bf16 else MMA_K_INST

    # CVT scatter geometry: each thread reads/writes CVT_VEC contiguous elements
    # per row. CVT_VEC stays 4 for BOTH precisions on purpose. The (8 x 32)
    # stride-4 layout is tuned to spread SMEM accesses across 8 banks; widening
    # bf16 to 8 (a (16 x 16) layout) cuts the store *instruction* count but lands
    # on the higher-bank-conflict pattern the stride-4 layout was chosen to avoid
    # and measured ~1.5x SLOWER at T=16384 (the kernel is SMEM-bank-conflict
    # bound, not store-width bound). So bf16 keeps the 4-wide (STS.64) scatter.
    CVT_VEC: Constexpr = 4
    CVT_NCOL: Constexpr = KEY_DIM // CVT_VEC
    CVT_NROW: Constexpr = THREADS_PER_CTA // CVT_NCOL
    CVT_ROWSTEPS: Constexpr = BT // CVT_NROW
    # A (BT x BT) scatter over the K=BT dim — same reasoning, kept 4-wide.
    A_VEC: Constexpr = 4
    A_NCOL: Constexpr = BT // A_VEC
    A_NROW: Constexpr = THREADS_PER_CTA // A_NCOL
    A_ROWSTEPS: Constexpr = BT // A_NROW

    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    warp_idx = cute.arch.make_warp_uniform(warp_idx)
    cta_idx, _, _ = cute.arch.block_idx()
    num_key_heads = cute.size(mK.shape[2])
    num_value_heads = cute.size(mV.shape[2])
    head_group_size = num_value_heads // num_key_heads
    num_chunks_runtime = mNumChunks[0].to(Int32)
    total_work_runtime = num_chunks_runtime * num_value_heads
    iters_per_cta_outer = (
        total_work_runtime + num_persistent_ctas - 1 - cta_idx
    ) // num_persistent_ctas

    if cta_idx < num_persistent_ctas:
        key_dim = cute.size(mK.shape[3])
        value_dim = cute.size(mV.shape[3])

        # ---- 2. SMEM allocation ----
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # MMA-ready swizzled SMEM for A and B operands.
        # A is shared by both MMAs; B is split into KB (for W) and VB (for U).
        sA_mma = smem.allocate_tensor(
            element_type=mma_dtype,
            layout=a_smem_layout.outer,
            byte_alignment=128,
            swizzle=a_smem_layout.inner,
        )
        sKb_mma = smem.allocate_tensor(
            element_type=mma_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )
        sVb_mma = smem.allocate_tensor(
            element_type=mma_dtype,
            layout=b_smem_layout.outer,
            byte_alignment=128,
            swizzle=b_smem_layout.inner,
        )

        # ---- TMEM allocation ----
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=THREADS_PER_CTA,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
        )
        # W accumulator uses TMEM cols [0:128], U uses cols [128:256].
        # (Row-bias was attempted at rows [64:128] but is not honored by UMMA.)
        num_tmem_cols = 256
        tmem.allocate(num_tmem_cols)

        # ---- Pipelines ----
        # Only MMA completion uses a pipeline (PipelineUmmaAsync). CVT->MMA sync
        # is done via a simple named barrier. AB_STAGES=1 means no overlap, so a
        # full pipeline is unnecessary.
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=ACC_STAGE,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=THREADS_PER_CTA),
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
        ).make_participants()

        # CVT -> MMA named barrier (all 256 threads arrive).
        cvt_mma_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=THREADS_PER_CTA,
        )

        # ---- 3a. cp.async staging for bf16 inputs ----
        sK_stage = cute.make_tensor(
            storage.sK_staging.data_ptr(),
            cute.make_layout((BT, KEY_DIM), stride=(KEY_DIM, 1)),
        )
        sV_stage = cute.make_tensor(
            storage.sV_staging.data_ptr(),
            cute.make_layout((BT, VAL_DIM), stride=(VAL_DIM, 1)),
        )
        sQ_stage = cute.make_tensor(
            storage.sQ_staging.data_ptr(),
            cute.make_layout((BT, KEY_DIM), stride=(KEY_DIM, 1)),
        )
        sG_stage = cute.make_tensor(
            storage.sG_staging.data_ptr(),
            cute.make_layout((BT, KEY_DIM), stride=(KEY_DIM, 1)),
        )
        # sA_staging is allocated as fp32 (max size). For bf16 A, reinterpret
        # the pointer as bf16 — uses the same memory, half the elements.
        if cutlass.const_expr(a_is_fp32):
            sA_stage = cute.make_tensor(
                storage.sA_staging.data_ptr(),
                cute.make_layout((BT, BT), stride=(BT, 1)),
            )
        else:
            sA_stage = cute.make_tensor(
                cute.recast_ptr(storage.sA_staging.data_ptr(), dtype=cutlass.BFloat16),
                cute.make_layout((BT, BT), stride=(BT, 1)),
            )
        sBeta_stage = cute.make_tensor(storage.sBeta_staging.data_ptr(), cute.make_layout(BT))

        atom_bf16 = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )
        atom_fp32 = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.Float32,
            num_bits_per_copy=128,
        )
        tiled_copy_bf16 = cute.make_tiled_copy_tv(
            atom_bf16,
            cute.make_layout((16, 16), stride=(16, 1)),
            cute.make_layout((1, 8)),
        )
        tiled_copy_fp32_kv = cute.make_tiled_copy_tv(
            atom_fp32,
            cute.make_layout((8, 32), stride=(32, 1)),
            cute.make_layout((1, 4)),
        )
        # A copy: fp32 uses (16,16) threads × (1,4) vals = 4 fp32/thread/rep, 4 reps.
        # bf16 uses (32,8) threads × (1,8) vals = 8 bf16/thread/rep, 2 reps.
        if cutlass.const_expr(a_is_fp32):
            tiled_copy_a = cute.make_tiled_copy_tv(
                atom_fp32,
                cute.make_layout((16, 16), stride=(16, 1)),
                cute.make_layout((1, 4)),
            )
        else:
            tiled_copy_a = cute.make_tiled_copy_tv(
                atom_bf16,
                cute.make_layout((32, 8), stride=(8, 1)),
                cute.make_layout((1, 8)),
            )
        thr_copy_bf16 = tiled_copy_bf16.get_slice(tidx)
        thr_copy_fp32_kv = tiled_copy_fp32_kv.get_slice(tidx)
        thr_copy_a = tiled_copy_a.get_slice(tidx)

        # ---- TMEM pointer setup (one-time, hoisted out of chunk loop) ----
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
        # Col-offset pointer: TMEM addressing is (row << 16) | col; adding 128
        # to the low bits shifts col by 128 fp32 lanes.
        tmem_ptr_u = tmem_ptr + 128

        # Partition for MMA. Each MMA handles a single (M=64, N=128) tile; K=64.
        # These are loop-invariant — same for every chunk.
        thr_mma = tiled_mma.get_slice(0)
        acc_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
        tCtAcc_tmpl = tiled_mma.make_fragment_C(acc_shape)
        acc_w = cute.make_tensor(tmem_ptr, tCtAcc_tmpl.layout)
        acc_u = cute.make_tensor(tmem_ptr_u, tCtAcc_tmpl.layout)
        tCrA = tiled_mma.make_fragment_A(sA_mma)
        tCrB_w = tiled_mma.make_fragment_B(sKb_mma)
        tCrB_u = tiled_mma.make_fragment_B(sVb_mma)

        # Release TMEM allocation lock so other warps can proceed. Only needed
        # once since alloc happens once.
        tmem.relinquish_alloc_permit()

        # ---- Persistent-loop prologue: prefetch the first work item ----
        # The raw staging buffers are only needed through CVT/A packing. Once
        # those are complete, later iterations can reuse the same raw buffers
        # for the next work item while the current tile drains from TMEM.
        if cutlass.const_expr(prefetch_next_tile) and iters_per_cta_outer > 0:
            prologue_work_idx = cta_idx
            prologue_chunk_block = prologue_work_idx % num_chunks_runtime
            prologue_value_head_idx = prologue_work_idx // num_chunks_runtime
            prologue_key_head_idx = prologue_value_head_idx // head_group_size
            prologue_seq_idx = mChunkIndices[prologue_chunk_block, 0].to(Int32)
            prologue_chunk_idx = mChunkIndices[prologue_chunk_block, 1].to(Int32)
            prologue_bos = mCuSeqlens[prologue_seq_idx].to(Int32)
            prologue_eos = mCuSeqlens[prologue_seq_idx + 1].to(Int32)
            prologue_time_base = prologue_bos + prologue_chunk_idx * BT
            prologue_valid = cutlass.min(prologue_eos - prologue_time_base, Int32(BT))

            (
                prologue_gK_tile,
                prologue_gV_tile,
                prologue_gQ_tile,
                prologue_gG_tile,
                prologue_gA_tile,
            ) = _make_input_tiles(
                mK,
                mV,
                mQ,
                mG,
                mA,
                prologue_time_base,
                prologue_key_head_idx,
                prologue_value_head_idx,
            )
            _copy_input_tiles(
                tiled_copy_bf16,
                tiled_copy_fp32_kv,
                tiled_copy_a,
                thr_copy_bf16,
                thr_copy_fp32_kv,
                thr_copy_a,
                prologue_gK_tile,
                prologue_gV_tile,
                prologue_gQ_tile,
                prologue_gG_tile,
                prologue_gA_tile,
                sK_stage,
                sV_stage,
                sQ_stage,
                sG_stage,
                sA_stage,
                prologue_valid,
                has_q,
                has_gk,
            )

        # ---- Persistent loop over runtime (chunk, head) work items ----
        for persistent_iter in cutlass.range(iters_per_cta_outer, unroll=1):
            work_idx = persistent_iter * num_persistent_ctas + cta_idx
            chunk_block = work_idx % num_chunks_runtime
            value_head_idx = work_idx // num_chunks_runtime
            key_head_idx = value_head_idx // head_group_size

            # ---- Varlen chunk coords (for CVT / EPI addressing) ----
            seq_idx = mChunkIndices[chunk_block, 0].to(Int32)
            chunk_idx = mChunkIndices[chunk_block, 1].to(Int32)
            bos = mCuSeqlens[seq_idx].to(Int32)
            eos = mCuSeqlens[seq_idx + 1].to(Int32)
            time_base = bos + chunk_idx * BT
            valid = cutlass.min(eos - time_base, Int32(BT))

            if cutlass.const_expr(not prefetch_next_tile):
                gK_tile, gV_tile, gQ_tile, gG_tile, gA_tile = _make_input_tiles(
                    mK, mV, mQ, mG, mA, time_base, key_head_idx, value_head_idx
                )
                _copy_input_tiles(
                    tiled_copy_bf16,
                    tiled_copy_fp32_kv,
                    tiled_copy_a,
                    thr_copy_bf16,
                    thr_copy_fp32_kv,
                    thr_copy_a,
                    gK_tile,
                    gV_tile,
                    gQ_tile,
                    gG_tile,
                    gA_tile,
                    sK_stage,
                    sV_stage,
                    sQ_stage,
                    sG_stage,
                    sA_stage,
                    valid,
                    has_q,
                    has_gk,
                )

            # Stage beta[time_base:time_base+BT, value_head_idx] into SMEM once per chunk.
            # beta staging only depends on gmem mBeta (independent of the K/V/Q/G
            # cp.async), so it is issued BEFORE the cp.async wait and shares the
            # single post-wait barrier — saving one full-block barrier per tile.
            # Layout is transposed so each CVT thread's CVT_ROWSTEPS beta values
            # sit contiguously → 1 vectorized LDS per thread. Mapping:
            # beta[orig_row] → sBeta[(orig_row % CVT_NROW)*CVT_ROWSTEPS +
            # orig_row // CVT_NROW] (tr inner, row_step outer).
            if tidx < BT:
                safe_tidx = cutlass.min(tidx, valid - 1)
                sBeta_stage[(tidx % CVT_NROW) * CVT_ROWSTEPS + (tidx // CVT_NROW)] = mBeta[
                    0, time_base + safe_tidx, value_head_idx
                ]

            # Wait until only 1 group is in flight — i.e., KVQG done, A may still be loading.
            cute.arch.cp_async_wait_group(1)
            cute.arch.barrier()

            # ---- 3b. CVT: 256 threads, (CVT_NROW x CVT_NCOL) layout ----
            # Thread (tr, tc): tr = tidx//CVT_NCOL, tc = tidx%CVT_NCOL.
            # Owns rows [tr, tr+CVT_NROW, ...] and cols [tc*CVT_VEC, +CVT_VEC).
            # CVT_VEC contig cols per thread → one 128-bit LDS/STS per access.
            tr = tidx // CVT_NCOL
            tc = tidx % CVT_NCOL
            col_start = tc * CVT_VEC

            # Hoist g_last[col_start:col_start+CVT_VEC] once via vectorized SMEM
            # read. Only meaningful when gating is active.
            g_last_vals = cute.make_rmem_tensor(CVT_VEC, Float32)
            if cutlass.const_expr(has_gk):
                # gk_last is the gate at the chunk's LAST VALID row. For a partial
                # (varlen) chunk the valid length is < BT, so the fixed row BT-1
                # falls past the sequence's valid tokens (into padding or the next
                # sequence) and corrupts kg = k * exp2(gk_last - gk). Clamp to the
                # real last row: valid - 1 = min(BT, eos - time_base) - 1.
                g_last_row = valid - 1
                sG_last_slice = cute.make_tensor(
                    sG_stage.iterator + g_last_row * KEY_DIM + col_start,
                    cute.make_layout(CVT_VEC),
                )
                cute.autovec_copy(sG_last_slice, g_last_vals)

            # Hoist beta as fp32. Production GDN beta is fp32, while older
            # benchmarks may pass bf16; staging in fp32 preserves both paths.
            # Per-thread beta positions are tr*CVT_ROWSTEPS + row_step.
            sBeta_tr_slice = cute.make_tensor(
                sBeta_stage.iterator + tr * CVT_ROWSTEPS,
                cute.make_layout(CVT_ROWSTEPS),
            )
            beta_frag = cute.make_rmem_tensor(CVT_ROWSTEPS, Float32)
            cute.autovec_copy(sBeta_tr_slice, beta_frag)

            # Per-iter register fragments — CVT_VEC elements (one scatter phase).
            k_frag = cute.make_rmem_tensor(CVT_VEC, IO_DTYPE)
            v_frag = cute.make_rmem_tensor(CVT_VEC, IO_DTYPE)
            q_frag = cute.make_rmem_tensor(CVT_VEC, IO_DTYPE)
            g_frag = cute.make_rmem_tensor(CVT_VEC, Float32)
            qg_frag = cute.make_rmem_tensor(CVT_VEC, IO_DTYPE)
            kg_frag = cute.make_rmem_tensor(CVT_VEC, IO_DTYPE)
            # kb/vb feed the MMA B operand → mma_dtype.
            kb_frag = cute.make_rmem_tensor(CVT_VEC, mma_dtype)
            vb_frag = cute.make_rmem_tensor(CVT_VEC, mma_dtype)

            for row_step in cutlass.range(CVT_ROWSTEPS, unroll_full=True):
                row = tr + row_step * CVT_NROW
                safe_row = cutlass.min(row, valid - 1)
                row_in_range = row < valid
                time_idx = time_base + row
                beta_val = beta_frag[row_step]
                k_inst = row % mma_k_inst
                k_outer = row // mma_k_inst

                # Vectorized 128-bit SMEM reads of CVT_VEC contig cols.
                sK_slice = cute.make_tensor(
                    sK_stage.iterator + safe_row * KEY_DIM + col_start,
                    cute.make_layout(CVT_VEC),
                )
                sV_slice = cute.make_tensor(
                    sV_stage.iterator + safe_row * VAL_DIM + col_start,
                    cute.make_layout(CVT_VEC),
                )
                cute.autovec_copy(sK_slice, k_frag)
                cute.autovec_copy(sV_slice, v_frag)
                if cutlass.const_expr(has_q):
                    sQ_slice = cute.make_tensor(
                        sQ_stage.iterator + safe_row * KEY_DIM + col_start,
                        cute.make_layout(CVT_VEC),
                    )
                    cute.autovec_copy(sQ_slice, q_frag)
                if cutlass.const_expr(has_gk):
                    sG_slice = cute.make_tensor(
                        sG_stage.iterator + safe_row * KEY_DIM + col_start,
                        cute.make_layout(CVT_VEC),
                    )
                    cute.autovec_copy(sG_slice, g_frag)

                # Compute.
                for c_off in cutlass.range_constexpr(CVT_VEC):
                    k_val = k_frag[c_off].to(Float32)
                    v_val = v_frag[c_off].to(Float32)

                    if cutlass.const_expr(has_gk):
                        g_val = g_frag[c_off]
                        g_last = g_last_vals[c_off]
                        exp2_g = cute.math.exp2(g_val, fastmath=True)
                        exp2_last_minus_g = cute.math.exp2(g_last - g_val, fastmath=True)
                        kb_frag[c_off] = (k_val * beta_val * exp2_g).to(mma_dtype)
                        kg_frag[c_off] = (k_val * exp2_last_minus_g).to(IO_DTYPE)
                    else:
                        kb_frag[c_off] = (k_val * beta_val).to(mma_dtype)

                    vb_frag[c_off] = (v_val * beta_val).to(mma_dtype)

                    if cutlass.const_expr(has_q and has_gk):
                        q_val = q_frag[c_off].to(Float32)
                        qg_frag[c_off] = (q_val * exp2_g).to(IO_DTYPE)

                # Scatter kb/vb into swizzled SMEM B operands (separate for W and U).
                # Build a rank-1 view over the N-mode at fixed (k_inst, k_outer) and
                # tile-divide into CVT_VEC-wide chunks. Each thread owns tile index =
                # tc (since col_start = tc*CVT_VEC). autovec_copy over the CVT_VEC
                # sub-view folds the scalar stores into one 128-bit STS (addresses
                # are contiguous within one 16B swizzle chunk for the aligned
                # col_start: 4 tf32 = 8 bf16 = 16 B).
                sKb_col = sKb_mma[(None, k_inst), 0, k_outer, 0]
                sVb_col = sVb_mma[(None, k_inst), 0, k_outer, 0]
                sKb_vec = cute.tiled_divide(sKb_col, (CVT_VEC,))
                sVb_vec = cute.tiled_divide(sVb_col, (CVT_VEC,))
                cute.autovec_copy(kb_frag, sKb_vec[None, tc])
                cute.autovec_copy(vb_frag, sVb_vec[None, tc])

                # Inline QG/KG gmem stores (vectorized 128-bit). QG only when q and gk
                # are both present; KG only when gk is present (matches triton semantics).
                if cutlass.const_expr(has_q and has_gk) and row_in_range:
                    gQG_slice = cute.make_tensor(
                        mQG.iterator
                        + (time_idx * mQG.layout.stride[1])
                        + (value_head_idx * mQG.layout.stride[2])
                        + col_start * mQG.layout.stride[3],
                        cute.make_layout(CVT_VEC),
                    )
                    cute.autovec_copy(qg_frag, gQG_slice)
                if cutlass.const_expr(has_gk) and row_in_range:
                    gKG_slice = cute.make_tensor(
                        mKG.iterator
                        + (time_idx * mKG.layout.stride[1])
                        + (value_head_idx * mKG.layout.stride[2])
                        + col_start * mKG.layout.stride[3],
                        cute.make_layout(CVT_VEC),
                    )
                    cute.autovec_copy(kg_frag, gKG_slice)

            # Wait for A cp.async (group 0) to complete — was allowed to overlap with KVQG CVT.
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # Process A (BT × BT) with vectorized scatter over the K=BT dim.
            # Thread remap: (tr_a, tc_a) = (tidx // A_NCOL, tidx % A_NCOL).
            # Each thread owns A_ROWSTEPS rows × A_VEC cols. col_start_a =
            # tc_a*A_VEC; the thread's A_VEC K-cols all fall in one a_k_outer group
            # (= col_start_a // mma_k_inst) and a single A_VEC-wide tile within the
            # mma_k_inst-wide a_k_inst dimension, so the scalar stores fold into one
            # 128-bit STS (4 tf32 = 8 bf16 = 16 B).
            tr_a = tidx // A_NCOL
            tc_a = tidx % A_NCOL
            col_start_a = tc_a * A_VEC
            a_k_outer_t = col_start_a // mma_k_inst
            a_tile_in_outer = (col_start_a % mma_k_inst) // A_VEC
            a_mma_frag = cute.make_rmem_tensor(A_VEC, mma_dtype)
            for a_row_step in cutlass.range(A_ROWSTEPS, unroll_full=True):
                a_row = tr_a + a_row_step * A_NROW
                a_row_in_range = a_row < valid
                sA_slice = cute.make_tensor(
                    sA_stage.iterator + a_row * BT + col_start_a,
                    cute.make_layout(A_VEC),
                )
                if cutlass.const_expr(a_is_fp32):
                    a_frag = cute.make_rmem_tensor(A_VEC, Float32)
                    cute.autovec_copy(sA_slice, a_frag)
                    for a_c_off in cutlass.range_constexpr(A_VEC):
                        a_col = col_start_a + a_c_off
                        keep_a = a_row_in_range & (a_col <= a_row) & (a_col < valid)
                        a_val = a_frag[a_c_off] if keep_a else Float32(0.0)
                        a_mma_frag[a_c_off] = a_val.to(mma_dtype)
                else:
                    a_bf16_frag = cute.make_rmem_tensor(A_VEC, IO_DTYPE)
                    cute.autovec_copy(sA_slice, a_bf16_frag)
                    for a_c_off in cutlass.range_constexpr(A_VEC):
                        a_col = col_start_a + a_c_off
                        keep_a = a_row_in_range & (a_col <= a_row) & (a_col < valid)
                        a_val = a_bf16_frag[a_c_off].to(Float32) if keep_a else Float32(0.0)
                        a_mma_frag[a_c_off] = a_val.to(mma_dtype)
                # Rank-1 view over a_k_inst at fixed (a_row, a_k_outer); tile into
                # A_VEC-wide chunks and write via autovec_copy → 128-bit STS.
                sA_row = sA_mma[(a_row, None), 0, a_k_outer_t, 0]
                sA_vec = cute.tiled_divide(sA_row, (A_VEC,))
                cute.autovec_copy(a_mma_frag, sA_vec[None, a_tile_in_outer])

            # Signal AB ready (fills both A and B SMEM tiles).
            cute.arch.fence_proxy("async.shared", space="cta")
            cvt_mma_barrier.arrive_and_wait()

            # ---- 4. MMA ----
            # Two independent MMAs into disjoint TMEM regions:
            #   W = A @ KB  -> acc_w at TMEM rows [0:64], cols [0:128]
            #   U = A @ VB  -> acc_u at TMEM rows [0:64], cols [128:256]
            # TMEM ptrs, acc_w/acc_u, tCrA/tCrB_* are all hoisted outside the loop.

            # Issue MMAs — warp 0 drives. W and U use separate pipeline stages
            # so EPI_W can start as soon as MMA_W completes (while MMA_U is
            # still in flight) and MMA_{n+1,W} can start while EPI_{n,U} runs.
            if warp_idx == 0:
                num_k_blocks = cute.size(tCrA, mode=[2])

                # MMA_W — stage 0, TMEM cols [0:128]
                acc_empty_w = acc_producer.acquire_and_advance()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_coord = (None, None, k_block_idx, 0)
                    cute.gemm(
                        tiled_mma,
                        acc_w,
                        tCrA[k_block_coord],
                        tCrB_w[k_block_coord],
                        acc_w,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                acc_empty_w.commit()

                if cutlass.const_expr(not u_dot_precision_tf32x3):
                    # MMA_U — stage 1, TMEM cols [128:256]
                    acc_empty_u = acc_producer.acquire_and_advance()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, 0)
                        cute.gemm(
                            tiled_mma,
                            acc_u,
                            tCrA[k_block_coord],
                            tCrB_u[k_block_coord],
                            acc_u,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    acc_empty_u.commit()

            if cutlass.const_expr(prefetch_next_tile and not u_dot_precision_tf32x3):
                next_persistent_iter = persistent_iter + 1
                if next_persistent_iter < iters_per_cta_outer:
                    next_work_idx = next_persistent_iter * num_persistent_ctas + cta_idx
                    next_chunk_block = next_work_idx % num_chunks_runtime
                    next_value_head_idx = next_work_idx // num_chunks_runtime
                    next_key_head_idx = next_value_head_idx // head_group_size
                    next_seq_idx = mChunkIndices[next_chunk_block, 0].to(Int32)
                    next_chunk_idx = mChunkIndices[next_chunk_block, 1].to(Int32)
                    next_bos = mCuSeqlens[next_seq_idx].to(Int32)
                    next_eos = mCuSeqlens[next_seq_idx + 1].to(Int32)
                    next_time_base = next_bos + next_chunk_idx * BT
                    next_valid = cutlass.min(next_eos - next_time_base, Int32(BT))

                    (
                        next_gK_tile,
                        next_gV_tile,
                        next_gQ_tile,
                        next_gG_tile,
                        next_gA_tile,
                    ) = _make_input_tiles(
                        mK,
                        mV,
                        mQ,
                        mG,
                        mA,
                        next_time_base,
                        next_key_head_idx,
                        next_value_head_idx,
                    )
                    _copy_input_tiles(
                        tiled_copy_bf16,
                        tiled_copy_fp32_kv,
                        tiled_copy_a,
                        thr_copy_bf16,
                        thr_copy_fp32_kv,
                        thr_copy_a,
                        next_gK_tile,
                        next_gV_tile,
                        next_gQ_tile,
                        next_gG_tile,
                        next_gA_tile,
                        sK_stage,
                        sV_stage,
                        sQ_stage,
                        sG_stage,
                        sA_stage,
                        next_valid,
                        has_q,
                        has_gk,
                    )

            # ---- 5. Epilogue: TMEM -> rmem -> bf16 -> gmem ----
            # W and U are waited independently so EPI_W can overlap with MMA_U
            # (TMEM alloc permit was released once before the chunk loop.)

            # Sub-tile epilogue for ILP.
            # For M=64 tiles the correct TMEM load atom is Ld16x256b (16 rows × 256b
            # lanes). With Repetition.x8 each warp reads 16 rows × 64 fp32 cols.
            # Each accumulator is 64×128 (fp32), so 2 subtiles of 64 cols each.
            subtile_cnt = 4  # 128 cols / 4 = 32 cols per subtile
            epi_tiler = (
                (
                    cute.size(acc_w, mode=[0, 0]),
                    cute.size(acc_w, mode=[0, 1]) // subtile_cnt,
                ),
            )
            acc_w_epi = cute.zipped_divide(acc_w, epi_tiler)
            acc_u_epi = cute.zipped_divide(acc_u, epi_tiler)

            tmem_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition.x4),
                ACC_DTYPE,
            )
            tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_atom, acc_w_epi[None, 0])
            # Only first 128 threads participate in TMEM epilogue (TMEM atom is 128-thread).
            epi_tidx = tidx % 128
            tmem_thr_copy = tmem_tiled_copy.get_slice(epi_tidx)

            tDtW = tmem_thr_copy.partition_S(acc_w_epi)
            tDtU = tmem_thr_copy.partition_S(acc_u_epi)

            # Build MMA-partitioned views of gW and gU. Each is shape (BT, head_dim).
            w_stride_t = mW.layout.stride[1]
            w_stride_k = mW.layout.stride[3]
            u_stride_t = mU.layout.stride[1]
            u_stride_v = mU.layout.stride[3]
            mW_shifted = cute.domain_offset((0, time_base, value_head_idx, 0), mW)
            mU_shifted = cute.domain_offset((0, time_base, value_head_idx, 0), mU)
            gW = cute.make_tensor(
                mW_shifted.iterator,
                cute.make_layout((BT, key_dim), stride=(w_stride_t, w_stride_k)),
            )
            gU = cute.make_tensor(
                mU_shifted.iterator,
                cute.make_layout((BT, value_dim), stride=(u_stride_t, u_stride_v)),
            )

            # MMA-partition then epi-subtile so mode=1 (subtile index) matches tDt.
            tCgW = thr_mma.partition_C(gW)
            tCgU = thr_mma.partition_C(gU)
            gW_epi = cute.zipped_divide(tCgW, epi_tiler)
            gU_epi = cute.zipped_divide(tCgU, epi_tiler)
            cW = cute.make_identity_tensor((BT, KEY_DIM))
            cU = cute.make_identity_tensor((BT, VAL_DIM))
            tCcW = thr_mma.partition_C(cW)
            tCcU = thr_mma.partition_C(cU)
            cW_epi = cute.zipped_divide(tCcW, epi_tiler)
            cU_epi = cute.zipped_divide(tCcU, epi_tiler)

            tDgW = tmem_thr_copy.partition_D(gW_epi)
            tDgU = tmem_thr_copy.partition_D(gU_epi)
            tDcW = tmem_thr_copy.partition_D(cW_epi)
            tDcU = tmem_thr_copy.partition_D(cU_epi)

            tCrAcc = cute.make_rmem_tensor(tDgW[None, None, 0].shape, ACC_DTYPE)
            tCrC = cute.make_rmem_tensor(tDgW[None, None, 0].shape, IO_DTYPE)

            # Drain W stage first (stage 0): this runs in parallel with MMA_U.
            acc_full_w = acc_consumer.wait_and_advance()
            if tidx < 128:
                for w_i in cutlass.range_constexpr(subtile_cnt):
                    cute.copy(tmem_tiled_copy, tDtW[None, None, w_i], tCrAcc)
                    tCrC.store(tCrAcc.load().to(IO_DTYPE))
                    _store_epilogue_tile(
                        tCrC,
                        tDgW[None, None, w_i],
                        tDcW[None, None, w_i],
                        valid,
                    )
            acc_full_w.release()

            if cutlass.const_expr(u_dot_precision_tf32x3):
                # Triton tf32x3 expands the U dot into three TF32 products:
                # A_hi @ V_hi + A_hi @ V_lo + A_lo @ V_hi. Run them
                # sequentially so the existing A/V SMEM operands can be reused.
                if warp_idx == 0:
                    num_k_blocks = cute.size(tCrA, mode=[2])
                    acc_empty_u_hi = acc_producer.acquire_and_advance()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, 0)
                        cute.gemm(
                            tiled_mma,
                            acc_u,
                            tCrA[k_block_coord],
                            tCrB_u[k_block_coord],
                            acc_u,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    acc_empty_u_hi.commit()
                acc_full_u_hi = acc_consumer.wait_and_advance()
                acc_full_u_hi.release()
                cute.arch.barrier()

                vb_res_frag = cute.make_rmem_tensor(4, mma_dtype)
                for row_step_res in cutlass.range(8, unroll_full=True):
                    row_res = tr + row_step_res * 8
                    safe_row_res = cutlass.min(row_res, valid - 1)
                    k_inst_res = row_res % mma_k_inst
                    k_outer_res = row_res // mma_k_inst
                    beta_val_res = beta_frag[row_step_res]
                    sV_slice_res = cute.make_tensor(
                        sV_stage.iterator + safe_row_res * VAL_DIM + col_start,
                        cute.make_layout(4),
                    )
                    cute.autovec_copy(sV_slice_res, v_frag)
                    for c_off_res in cutlass.range_constexpr(4):
                        vb_full = v_frag[c_off_res].to(Float32) * beta_val_res
                        vb_hi = vb_full.to(mma_dtype)
                        vb_res_frag[c_off_res] = (vb_full - vb_hi.to(Float32)).to(mma_dtype)
                    sVb_res_col = sVb_mma[(None, k_inst_res), 0, k_outer_res, 0]
                    sVb_res_4 = cute.tiled_divide(sVb_res_col, (4,))
                    cute.autovec_copy(vb_res_frag, sVb_res_4[None, tc])

                cute.arch.fence_proxy("async.shared", space="cta")
                cvt_mma_barrier.arrive_and_wait()

                if warp_idx == 0:
                    num_k_blocks = cute.size(tCrA, mode=[2])
                    acc_empty_u_v_res = acc_producer.acquire_and_advance()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, 0)
                        cute.gemm(
                            tiled_mma,
                            acc_u,
                            tCrA[k_block_coord],
                            tCrB_u[k_block_coord],
                            acc_u,
                        )
                    acc_empty_u_v_res.commit()
                acc_full_u_v_res = acc_consumer.wait_and_advance()
                acc_full_u_v_res.release()
                cute.arch.barrier()

                a_res_pack_frag = cute.make_rmem_tensor(4, mma_dtype)
                for a_res_row_step in cutlass.range(4, unroll_full=True):
                    a_res_row = tr_a + a_res_row_step * 16
                    a_res_row_in_range = a_res_row < valid
                    sA_res_slice = cute.make_tensor(
                        sA_stage.iterator + a_res_row * BT + col_start_a,
                        cute.make_layout(4),
                    )
                    a_res_load_frag = cute.make_rmem_tensor(4, Float32)
                    cute.autovec_copy(sA_res_slice, a_res_load_frag)
                    for a_res_c_off in cutlass.range_constexpr(4):
                        a_res_col = col_start_a + a_res_c_off
                        keep_a_res = (
                            a_res_row_in_range & (a_res_col <= a_res_row) & (a_res_col < valid)
                        )
                        a_res_val = a_res_load_frag[a_res_c_off] if keep_a_res else Float32(0.0)
                        a_hi = a_res_val.to(mma_dtype)
                        a_res_pack_frag[a_res_c_off] = (a_res_val - a_hi.to(Float32)).to(mma_dtype)
                    sA_res_row = sA_mma[(a_res_row, None), 0, a_k_outer_t, 0]
                    sA_res_4 = cute.tiled_divide(sA_res_row, (4,))
                    cute.autovec_copy(a_res_pack_frag, sA_res_4[None, a_tile_in_outer])

                vb_hi_frag = cute.make_rmem_tensor(4, mma_dtype)
                for row_step_hi in cutlass.range(8, unroll_full=True):
                    row_hi = tr + row_step_hi * 8
                    safe_row_hi = cutlass.min(row_hi, valid - 1)
                    k_inst_hi = row_hi % mma_k_inst
                    k_outer_hi = row_hi // mma_k_inst
                    beta_val_hi = beta_frag[row_step_hi]
                    sV_slice_hi = cute.make_tensor(
                        sV_stage.iterator + safe_row_hi * VAL_DIM + col_start,
                        cute.make_layout(4),
                    )
                    cute.autovec_copy(sV_slice_hi, v_frag)
                    for c_off_hi in cutlass.range_constexpr(4):
                        vb_hi_frag[c_off_hi] = (v_frag[c_off_hi].to(Float32) * beta_val_hi).to(
                            mma_dtype
                        )
                    sVb_hi_col = sVb_mma[(None, k_inst_hi), 0, k_outer_hi, 0]
                    sVb_hi_4 = cute.tiled_divide(sVb_hi_col, (4,))
                    cute.autovec_copy(vb_hi_frag, sVb_hi_4[None, tc])

                cute.arch.fence_proxy("async.shared", space="cta")
                cvt_mma_barrier.arrive_and_wait()

                if warp_idx == 0:
                    num_k_blocks = cute.size(tCrA, mode=[2])
                    acc_empty_u_a_res = acc_producer.acquire_and_advance()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (None, None, k_block_idx, 0)
                        cute.gemm(
                            tiled_mma,
                            acc_u,
                            tCrA[k_block_coord],
                            tCrB_u[k_block_coord],
                            acc_u,
                        )
                    acc_empty_u_a_res.commit()

                if cutlass.const_expr(prefetch_next_tile):
                    next_persistent_iter = persistent_iter + 1
                    if next_persistent_iter < iters_per_cta_outer:
                        next_work_idx = next_persistent_iter * num_persistent_ctas + cta_idx
                        next_chunk_block = next_work_idx % num_chunks_runtime
                        next_value_head_idx = next_work_idx // num_chunks_runtime
                        next_key_head_idx = next_value_head_idx // head_group_size
                        next_seq_idx = mChunkIndices[next_chunk_block, 0].to(Int32)
                        next_chunk_idx = mChunkIndices[next_chunk_block, 1].to(Int32)
                        next_bos = mCuSeqlens[next_seq_idx].to(Int32)
                        next_eos = mCuSeqlens[next_seq_idx + 1].to(Int32)
                        next_time_base = next_bos + next_chunk_idx * BT
                        next_valid = cutlass.min(next_eos - next_time_base, Int32(BT))

                        (
                            next_gK_tile,
                            next_gV_tile,
                            next_gQ_tile,
                            next_gG_tile,
                            next_gA_tile,
                        ) = _make_input_tiles(
                            mK,
                            mV,
                            mQ,
                            mG,
                            mA,
                            next_time_base,
                            next_key_head_idx,
                            next_value_head_idx,
                        )
                        _copy_input_tiles(
                            tiled_copy_bf16,
                            tiled_copy_fp32_kv,
                            tiled_copy_a,
                            thr_copy_bf16,
                            thr_copy_fp32_kv,
                            thr_copy_a,
                            next_gK_tile,
                            next_gV_tile,
                            next_gQ_tile,
                            next_gG_tile,
                            next_gA_tile,
                            sK_stage,
                            sV_stage,
                            sQ_stage,
                            sG_stage,
                            sA_stage,
                            next_valid,
                            has_q,
                            has_gk,
                        )

            # Drain U stage (stage 1).
            acc_full_u = acc_consumer.wait_and_advance()
            if tidx < 128:
                for u_i in cutlass.range_constexpr(subtile_cnt):
                    cute.copy(tmem_tiled_copy, tDtU[None, None, u_i], tCrAcc)
                    tCrC.store(tCrAcc.load().to(IO_DTYPE))
                    _store_epilogue_tile(
                        tCrC,
                        tDgU[None, None, u_i],
                        tDcU[None, None, u_i],
                        valid,
                    )
            acc_full_u.release()

        # Deallocate TMEM
        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)


# ============================================================================
# Host function: build tiled_mma and TMA atoms, then launch
# ============================================================================


class RecomputeWUForwardVarlenB1FullChunk:
    """
    Launch wrapper for the varlen B=1 full-chunk UMMA kernel.

    The specialization knobs are compile-time constants: optional q/gk outputs,
    A dtype, and the MMA operand precision (MmaPrecision). num_chunks remains a
    runtime device scalar passed through to the kernel.
    """

    def __init__(
        self,
        num_persistent_ctas: int,
        chunk_size: int = BT,
        a_is_fp32: bool = True,
        has_q: bool = True,
        has_gk: bool = True,
        precision: MmaPrecision = MmaPrecision.BF16,
        prefetch_next_tile: bool = False,
    ):
        assert chunk_size == BT, (
            f"RecomputeWUForwardVarlenB1FullChunk requires chunk_size={BT}, "
            f"got {chunk_size}. chunk_size=32 is not supported yet "
            f"(tf32 UMMA requires M>=64)."
        )
        self.num_persistent_ctas = num_persistent_ctas
        self.chunk_size = chunk_size
        self.a_is_fp32 = a_is_fp32
        self.has_q = has_q
        self.has_gk = has_gk
        self.precision = precision
        self.prefetch_next_tile = prefetch_next_tile
        self.threads_per_cta = THREADS_PER_CTA

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mBeta: cute.Tensor,
        mA: cute.Tensor,
        mG: cute.Tensor,
        mW: cute.Tensor,
        mU: cute.Tensor,
        mQG: cute.Tensor,
        mKG: cute.Tensor,
        mCuSeqlens: cute.Tensor,
        mChunkIndices: cute.Tensor,
        mNumChunks: cute.Tensor,
        stream: cuda.CUstream = None,
    ):
        a_is_fp32: Constexpr = self.a_is_fp32
        has_q: Constexpr = self.has_q
        has_gk: Constexpr = self.has_gk
        prefetch_next_tile: Constexpr = self.prefetch_next_tile
        # The single precision enum drives the kernel's compile-time flags; the
        # two booleans below are mutually exclusive by construction.
        mma_is_bf16: Constexpr = self.precision.is_bf16
        u_dot_precision_tf32x3: Constexpr = self.precision.is_tf32x3

        # Build the tiled MMA from the requested operand precision: bf16
        # (kind::f16, K-inst 16) or tf32 (kind::tf32, K-inst 8); fp32 accumulator.
        # B is MN-major so the CVT scatter of consecutive N cols at fixed K lands
        # on a contiguous SMEM stripe — eliminates the bank conflicts that K-major
        # produced.
        op, mma_dtype = _mma_config(self.precision)
        tiled_mma = cute.make_tiled_mma(op)

        # SMEM layouts (swizzled) for A and B operands.
        a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma,
            MMA_TILER_MNK,
            mma_dtype,
            AB_STAGES,
        )
        b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma,
            MMA_TILER_MNK,
            mma_dtype,
            AB_STAGES,
        )

        _kernel_varlen_b1_full_chunk.set_name_prefix("cutlass_dsl_recompute_w_u_fwd")
        _kernel_varlen_b1_full_chunk(
            tiled_mma,
            mQ,
            mK,
            mV,
            mBeta,
            mA,
            mG,
            mW,
            mU,
            mQG,
            mKG,
            mCuSeqlens,
            mChunkIndices,
            mNumChunks,
            a_smem_layout,
            b_smem_layout,
            self.num_persistent_ctas,
            a_is_fp32,
            has_q,
            has_gk,
            u_dot_precision_tf32x3,
            prefetch_next_tile,
            mma_is_bf16,
        ).launch(
            grid=(self.num_persistent_ctas, 1, 1),
            block=(THREADS_PER_CTA, 1, 1),
            stream=stream,
        )


# ============================================================================
# Entry point
# ============================================================================


@jit_cache
def _compile_recompute_w_u(
    chunk_size: int,
    key_heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    a_is_fp32: bool,
    beta_is_fp32: bool,
    has_q: bool,
    has_gk: bool,
    precision: MmaPrecision,
    prefetch_next_tile: bool,
    cu_seqlens_i32: bool,
    chunk_indices_i32: bool,
    num_chunks_i32: bool,
):
    """Compile one persistent recompute specialization from fake tensors."""
    target = get_compile_target()
    if (
        target.device_type != "cuda"
        or target.capability is None
        or target.capability < (10, 0)
        or target.sm_count is None
    ):
        raise ValueError(
            "KDA recompute requires a CUDA capability >= 10.0 target with a known SM count; "
            f"got target={target}"
        )
    assert chunk_size == BT, f"KDA recompute requires chunk_size={BT}, got {chunk_size}"
    assert key_dim == KEY_DIM, f"KDA recompute requires key_dim={KEY_DIM}, got {key_dim}"
    assert value_dim == VAL_DIM, f"KDA recompute requires value_dim={VAL_DIM}, got {value_dim}"

    tokens = cute.sym_int(divisibility=chunk_size)
    sequence_entries, chunks = cute.sym_int(), cute.sym_int()
    tensor_shape = (1, tokens)
    q = make_fake_compact_tensor(
        IO_DTYPE,
        (*tensor_shape, value_heads, key_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    k = make_fake_compact_tensor(
        IO_DTYPE,
        (*tensor_shape, key_heads, key_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    v = make_fake_compact_tensor(
        IO_DTYPE,
        (*tensor_shape, value_heads, value_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    beta = make_fake_compact_tensor(
        cutlass.Float32 if beta_is_fp32 else IO_DTYPE,
        (*tensor_shape, value_heads),
        stride_order=(2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    A = make_fake_compact_tensor(
        cutlass.Float32 if a_is_fp32 else IO_DTYPE,
        (*tensor_shape, value_heads, chunk_size),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    gk = make_fake_compact_tensor(
        cutlass.Float32 if has_gk else IO_DTYPE,
        (*tensor_shape, key_heads, key_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    w = make_fake_compact_tensor(
        IO_DTYPE,
        (*tensor_shape, value_heads, key_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    u = make_fake_compact_tensor(
        IO_DTYPE,
        (*tensor_shape, value_heads, value_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    qg = make_fake_compact_tensor(
        IO_DTYPE,
        (*tensor_shape, value_heads, key_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    kg = make_fake_compact_tensor(
        IO_DTYPE,
        (*tensor_shape, value_heads, key_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    cu_seqlens = make_fake_compact_tensor(
        cutlass.Int32 if cu_seqlens_i32 else cutlass.Int64,
        (sequence_entries,),
        stride_order=(0,),
        assumed_align=INDEX_ALIGN_BYTES if cu_seqlens_i32 else 8,
    )
    chunk_indices = make_fake_compact_tensor(
        cutlass.Int32 if chunk_indices_i32 else cutlass.Int64,
        (chunks, 2),
        stride_order=(1, 0),
        assumed_align=INDEX_ALIGN_BYTES if chunk_indices_i32 else 8,
    )
    num_chunks = make_fake_compact_tensor(
        cutlass.Int32 if num_chunks_i32 else cutlass.Int64,
        (1,),
        stride_order=(0,),
        assumed_align=INDEX_ALIGN_BYTES if num_chunks_i32 else 8,
    )
    op = RecomputeWUForwardVarlenB1FullChunk(
        num_persistent_ctas=target.sm_count * 2,
        chunk_size=chunk_size,
        a_is_fp32=a_is_fp32,
        has_q=has_q,
        has_gk=has_gk,
        precision=precision,
        prefetch_next_tile=prefetch_next_tile,
    )
    return compile_tvm_ffi(
        op,
        q,
        k,
        v,
        beta,
        A,
        gk,
        w,
        u,
        qg,
        kg,
        cu_seqlens,
        chunk_indices,
        num_chunks,
        name=(
            f"kda_recompute_hk{key_heads}_hv{value_heads}_k{key_dim}_v{value_dim}_"
            f"a{int(a_is_fp32)}_b{int(beta_is_fp32)}_q{int(has_q)}_g{int(has_gk)}_"
            f"{precision.value}_p{int(prefetch_next_tile)}_i{int(cu_seqlens_i32)}"
            f"{int(chunk_indices_i32)}{int(num_chunks_i32)}"
        ),
    )


def _assert_tensor_contract(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype | tuple[torch.dtype, ...],
    device: torch.device,
    assumed_align: int = DATA_ALIGN_BYTES,
) -> None:
    assert tuple(tensor.shape) == shape, (
        f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
    )
    allowed_dtypes = dtype if isinstance(dtype, tuple) else (dtype,)
    assert tensor.dtype in allowed_dtypes, (
        f"{name} must have dtype "
        f"{allowed_dtypes[0] if len(allowed_dtypes) == 1 else allowed_dtypes}, "
        f"got {tensor.dtype}"
    )
    assert tensor.device == device, f"{name} must be on {device}, got {tensor.device}"
    assert tensor.is_contiguous(), f"{name} must be contiguous, got strides {tensor.stride()}"
    assert tensor.data_ptr() % assumed_align == 0, (
        f"{name} data pointer must be {assumed_align}-byte aligned"
    )


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    num_chunks: torch.Tensor | None = None,
    chunk_size: int = BT,
    dot_precision: str | MmaPrecision = "bf16",
    experimental_prefetch: bool = True,
    # Back-compat alias: older callers may pass `g=` instead of `gk=`.
    g: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    Recompute KDA W/U intermediates on the native CuTe path.

    Supported scope: varlen B=1, full chunks, chunk_size=64, K=V=128. The
    caller must provide cu_seqlens, chunk_indices, and num_chunks. num_chunks is
    a CUDA scalar tensor read by the kernel at runtime, not specialized by
    value, which keeps padded CUDA graph replay from recompiling when the active
    chunk count changes.

    Computes per chunk:
      - w = A @ (k * beta * exp2(gk)), or A @ (k * beta) without gk
      - u = A @ (v * beta)
      - qg = q * exp2(gk), returned only when both q and gk are provided
      - kg = k * exp2(gk_last - gk), returned only when gk is provided

    dot_precision selects the MMA operand precision (see the module header):
      - "bf16"   (default): bf16 operands, kind::f16 MMA. Fastest.
      - "tf32"            : tf32 operands, kind::tf32 MMA. More accurate.
      - "tf32x3"          : tf32 base + first-order residual products on U for
                            ~fp32 accuracy. Only valid when A is fp32.
    The accumulator is always fp32; W is single-pass for every mode.
    """
    if gk is None and g is not None:
        gk = g
    has_gk = gk is not None
    has_q = q is not None and has_gk

    precision = _normalize_precision(dot_precision)
    assert chunk_size == BT, (
        f"recompute_w_u_fwd only supports chunk_size={BT} on the native CuTe "
        f"path; got {chunk_size}. chunk_size=32 is not supported yet "
        f"(tf32 UMMA requires M>=64)."
    )

    assert k.ndim == 4, f"k must be 4D [B, T, H, K], got shape {tuple(k.shape)}"
    assert v.ndim == 4, f"v must be 4D [B, T, H, V], got shape {tuple(v.shape)}"
    assert beta.ndim == 3, f"beta must be 3D [B, T, H], got shape {tuple(beta.shape)}"
    assert A.ndim == 4, f"A must be 4D [B, T, H, BT], got shape {tuple(A.shape)}"
    if q is not None:
        assert q.ndim == 4, f"q must be 4D [B, T, H, K], got shape {tuple(q.shape)}"
    if gk is not None:
        assert gk.ndim == 4, f"gk must be 4D [B, T, H, K], got shape {tuple(gk.shape)}"

    B, T, H_K, K = k.shape
    H_V = v.shape[2]
    V = v.shape[3]
    device = k.device
    assert B == 1, f"recompute_w_u_fwd requires B=1, got B={B}"
    assert cu_seqlens is not None, "recompute_w_u_fwd requires cu_seqlens (varlen path only)"
    assert T % BT == 0, f"recompute_w_u_fwd requires T % BT == 0 (T={T}, BT={BT})"
    assert K == KEY_DIM, f"recompute_w_u_fwd requires head_dim K={KEY_DIM}, got {K}"
    assert V == VAL_DIM, f"recompute_w_u_fwd requires head_dim V={VAL_DIM}, got {V}"
    assert H_V % H_K == 0, (
        f"recompute_w_u_fwd requires value heads ({H_V}) to be divisible by key heads ({H_K})"
    )
    assert beta.shape[2] == H_V, f"beta heads ({beta.shape[2]}) must match v heads ({H_V})"
    assert A.shape[2] == H_V, f"A heads ({A.shape[2]}) must match v heads ({H_V})"
    if q is not None:
        assert q.shape[2] == H_V, f"q heads ({q.shape[2]}) must match v heads ({H_V})"
    if gk is not None:
        assert gk.shape[2] == H_K, f"gk heads ({gk.shape[2]}) must match k heads ({H_K})"
    assert A.shape[-1] == BT, f"A.shape[-1]={A.shape[-1]} does not match chunk_size={BT}"
    assert A.dtype in (
        torch.float32,
        torch.bfloat16,
    ), f"A.dtype must be float32 or bfloat16, got {A.dtype}"
    assert beta.dtype in (
        torch.float32,
        torch.bfloat16,
    ), f"beta.dtype must be float32 or bfloat16, got {beta.dtype}"

    if not isinstance(k, FakeTensor):
        if not k.is_cuda:
            raise RuntimeError("recompute_w_u_fwd kernel requires CUDA tensors")
        _assert_tensor_contract(
            "k",
            k,
            shape=(B, T, H_K, KEY_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        _assert_tensor_contract(
            "v",
            v,
            shape=(B, T, H_V, VAL_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        _assert_tensor_contract(
            "beta",
            beta,
            shape=(B, T, H_V),
            dtype=(torch.float32, torch.bfloat16),
            device=device,
        )
        _assert_tensor_contract(
            "A",
            A,
            shape=(B, T, H_V, BT),
            dtype=(torch.float32, torch.bfloat16),
            device=device,
        )
        if q is not None:
            _assert_tensor_contract(
                "q",
                q,
                shape=(B, T, H_V, KEY_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
        if gk is not None:
            _assert_tensor_contract(
                "gk",
                gk,
                shape=(B, T, H_K, KEY_DIM),
                dtype=torch.float32,
                device=device,
            )

    # Output allocation — only allocate the outputs we will actually produce.
    w = k.new_empty((B, T, H_V, KEY_DIM))
    u = torch.empty_like(v)
    qg_out: torch.Tensor | None = torch.empty_like(q) if has_q else None
    kg_out: torch.Tensor | None = k.new_empty((B, T, H_V, KEY_DIM)) if has_gk else None

    if isinstance(k, FakeTensor):
        return w, u, qg_out, kg_out

    # Require caller-owned varlen metadata. num_chunks follows Triton: it is a
    # runtime device scalar loaded in-kernel, not specialized by value. The
    # wrapper must not allocate hidden metadata tensors on the hot path.
    assert chunk_indices is not None, (
        "recompute_w_u_fwd CuTe path requires caller-provided chunk_indices"
    )
    assert num_chunks is not None, (
        "recompute_w_u_fwd CuTe path requires caller-provided num_chunks"
    )
    assert isinstance(num_chunks, torch.Tensor), (
        "recompute_w_u_fwd CuTe path requires num_chunks as a caller-provided CUDA tensor"
    )
    assert cu_seqlens.dtype in (
        torch.int32,
        torch.int64,
    ), f"cu_seqlens must be int32 or int64, got {cu_seqlens.dtype}"
    assert cu_seqlens.ndim == 1 and cu_seqlens.numel() >= 2, (
        f"cu_seqlens must be 1D with >=2 entries, got shape {tuple(cu_seqlens.shape)}"
    )
    assert cu_seqlens.device == k.device, (
        f"cu_seqlens must be on {k.device}, got {cu_seqlens.device}"
    )
    assert cu_seqlens.is_contiguous(), "cu_seqlens must be contiguous"
    assert chunk_indices.dtype in (
        torch.int32,
        torch.int64,
    ), f"chunk_indices must be int32 or int64, got {chunk_indices.dtype}"
    assert chunk_indices.ndim == 2 and chunk_indices.shape[-1] == 2, (
        f"chunk_indices must have shape (num_chunks, 2), got {tuple(chunk_indices.shape)}"
    )
    assert chunk_indices.device == k.device, (
        f"chunk_indices must be on {k.device}, got {chunk_indices.device}"
    )
    assert chunk_indices.is_contiguous(), "chunk_indices must be contiguous"
    assert num_chunks.dtype in (
        torch.int32,
        torch.int64,
    ), f"num_chunks must be int32 or int64, got {num_chunks.dtype}"
    assert num_chunks.numel() == 1, (
        f"num_chunks must contain one scalar value, got shape {tuple(num_chunks.shape)}"
    )
    assert num_chunks.device == k.device, (
        f"num_chunks must be on {k.device}, got {num_chunks.device}"
    )
    assert num_chunks.is_contiguous(), "num_chunks must be contiguous"

    num_chunks_i = num_chunks.reshape(1)

    # Stubs for optional inputs/outputs — kernel's constexpr branches fully
    # remove the corresponding loads/stores, but every mX param still needs a
    # valid tensor with compatible shape (kernel derives tile shapes from them).
    q_in = q if has_q else v
    # gG_full uses mG[*, *, *, K=128]. k has this shape, but its fp32-vs-bf16
    # dtype mismatch is fine because the cp.async and SMEM reads are dead code
    # when has_gk=False. Reusing k avoids a hidden CUDA allocation on the hot
    # optional no-gate path and keeps graph capture allocation-free.
    gk_in = gk if has_gk else k
    qg_in = qg_out if qg_out is not None else v
    kg_in = kg_out if kg_out is not None else v

    a_is_fp32 = A.dtype == torch.float32
    # tf32x3 emulates fp32 by adding first-order residual products; once A is
    # bf16-rounded the residual is meaningless, so it is only valid for fp32 A.
    assert not (precision.is_tf32x3 and not a_is_fp32), (
        "recompute_w_u_fwd dot_precision='tf32x3' requires fp32 A "
        f"(got A.dtype={A.dtype}); use 'bf16' or 'tf32' for non-fp32 A."
    )
    compiled = _compile_recompute_w_u(
        BT,
        H_K,
        H_V,
        K,
        V,
        a_is_fp32,
        beta.dtype == torch.float32,
        has_q,
        has_gk,
        precision,
        experimental_prefetch,
        cu_seqlens.dtype == torch.int32,
        chunk_indices.dtype == torch.int32,
        num_chunks_i.dtype == torch.int32,
    )
    compiled(
        q_in.detach(),
        k.detach(),
        v.detach(),
        beta.detach(),
        A.detach(),
        gk_in.detach(),
        w,
        u,
        qg_in,
        kg_in,
        cu_seqlens,
        chunk_indices,
        num_chunks_i,
    )

    return w, u, qg_out, kg_out
