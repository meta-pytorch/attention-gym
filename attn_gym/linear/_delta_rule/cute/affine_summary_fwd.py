# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL forward affine-summary kernel for delta-rule context parallelism.

One local shard of a delta-rule sequence acts on the recurrent state as an affine
map ``H_out = H_in @ A + B``. This kernel computes that summary directly from the
shared per-chunk WY factors (``kg``, ``w``, ``u``, ``cumulative_gate`` as produced
by ``chunk_kda_fwd_intra``) instead of running the full recurrence twice with
zero/identity probe states (see ``examples/kda_context_parallel.py``).

Math (K-first augmented state ``X`` of shape ``[K, V + K]``, fp32):

    X = [0 | I]
    per 64-token chunk c:
        Tmp        = W[c] @ X                 # [BT, V + K]
        Tmp[:, :V] = U[c] - Tmp[:, :V]
        Tmp[:, V:] =      - Tmp[:, V:]
        X          = diag(exp2(gk_last[c])) @ X + Kg[c]^T @ Tmp

The output is the V-first packed transpose ``X^T`` of shape ``[H, V + K, K]``:
rows ``[0:V]`` hold the state bias and rows ``[V:V+K]`` the state transition.
These are distinct from KDA's token-token query/key matrix ``A``.

Architecture (SM100 warp specialization, mirrors ``kda/bwd/cute/chunk_delta_h_bwd.py``):
  - One CTA per (BN-column tile of the augmented state, head); 6 warps.
  - CUDA warps (0-3) keep the fp32 ``[K, BN]`` state tile in registers across all
    chunks and produce the two SMEM MMA B operands per chunk.
  - Load warp (4) TMA-stages ``w``, ``kg^T``, ``u``, and the chunk-final gate row.
  - MMA warp (5) issues two SS-mode UMMA groups per chunk into TMEM:
      MMA1: W  @ X          -> wx  (BT=64, BN, K=128)
      MMA2: Kg^T @ [U; -wx] -> kt  (K=128, BN, K=64)
    ``U`` rides its own TMA ring straight into an MMA2 B-operand pass, so the
    ``Kg^T @ U`` product runs off the serial recurrence chain; the CUDA warps
    only split ``-wx`` for the remaining accumulate passes.

The fp32-valued MMA B operands (state ``X`` and ``wx``) are split into hi/lo
halves of the I/O dtype and accumulated in two UMMA passes. The A operands
(``w``, ``kg``) and ``u`` retain their original bf16/fp16 storage.

Limitations:
  - B=1, dense complete chunks only (``T % 64 == 0``), fixed K=V=128, BT=64.
  - Contiguous inputs with an int32-addressable ABI; SM100 (tcgen05) only.
"""

# NOTE: no `from __future__ import annotations` — cute.struct requires
# eager-evaluated annotations.

from enum import IntEnum
from typing import NamedTuple

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
import torch.nn.functional as F
from cuda.bindings import driver as cuda
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Float32, Int32, Int64
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.cute import compile_tvm_ffi, get_device_properties, jit_cache
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import requires_int64_abi

BT = 64  # chunk size
KEY_DIM = 128
VAL_DIM = 128
SUMMARY_DIM = VAL_DIM + KEY_DIM  # V-first packed [bias; transition] rows
DATA_ALIGN_BYTES = 16

_IO_TYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_CUTE_IO_TYPES = {"bf16": cutlass.BFloat16, "fp16": cutlass.Float16}


@cute.jit
def _sequence_feature_head_batch_view(tensor):
    """Re-rank ``[B,T,H,D]`` as ``[T,D,(H,B)]`` for a TMA operand."""
    layout = cute.group_modes(cute.select(tensor.layout, mode=[1, 3, 2, 0]), 2, 4)
    return cute.make_tensor(tensor.iterator, layout)


@cute.jit
def _feature_sequence_head_batch_view(tensor):
    """Re-rank ``[B,T,H,D]`` as ``[D,T,(H,B)]`` for a TMA operand."""
    layout = cute.group_modes(cute.select(tensor.layout, mode=[3, 1, 2, 0]), 2, 4)
    return cute.make_tensor(tensor.iterator, layout)


class WarpRole(IntEnum):
    """Warp-role boundaries in the affine-summary kernel."""

    CUDA = 0
    LOAD = 4
    MMA = 5
    END = 6


class TmaOp(NamedTuple):
    """One TMA copy atom and its tensor-map descriptor."""

    atom: cute.CopyAtom
    desc: cute.Tensor


class TmaOps(NamedTuple):
    """TMA operations owned by the affine-summary kernel."""

    w: TmaOp
    kg: TmaOp
    u: TmaOp
    gk: TmaOp


class Mmas(NamedTuple):
    """Tensor-core operations owned by the MMA warp."""

    wx: cute.TiledMma
    kt: cute.TiledMma
    ktu: cute.TiledMma


class SmemLayouts(NamedTuple):
    """Shared-memory layouts in their device-kernel construction order."""

    w: cute.ComposedLayout
    kg: cute.ComposedLayout
    u: cute.ComposedLayout
    xb: cute.ComposedLayout
    xb_store: cute.ComposedLayout
    tmpb: cute.ComposedLayout
    tmpb_store: cute.ComposedLayout


class _AffineSummaryFwdOp:
    """Warp-specialized SM100 chunk recurrence over one augmented-state column tile.

    The fp32 ``[K, BN]`` state slice persists in CUDA-warp registers; each chunk
    snapshots it (and later ``Tmp``) into SMEM as hi/lo I/O-dtype halves feeding
    two-pass SS-mode UMMAs.
    """

    CUDA_WARP_IDS = tuple(range(WarpRole.CUDA, WarpRole.LOAD))
    WARP_SZ = cute.arch.WARP_SIZE
    CTA_THREADS = WarpRole.END * WARP_SZ

    def __init__(self, dtype_name: str, heads: int, state_bn: int):
        assert state_bn in (32, 64), f"state_bn must be 32 or 64, got {state_bn}"
        assert SUMMARY_DIM % state_bn == 0
        self.dtype_name = dtype_name
        self.io_type = _CUTE_IO_TYPES[dtype_name]
        self.heads = heads
        self.BN = state_bn

        # MMA tile shapes (M, N, K).
        # MMA1: w @ X -> wx — w is (BT, K) K-major, X is (K, BN) snapshot in SMEM.
        self.wx_tile = (BT, self.BN, KEY_DIM)
        # MMA2: kg^T @ Tmp -> kt — kg^T is (K, BT) MN-major, Tmp is (BT, BN) in SMEM.
        self.kt_tile = (KEY_DIM, self.BN, BT)

        # TMA pipeline depths. The recurrence is a serial latency chain, so
        # shallow rings are enough for the loads to run ahead of it.
        self.w_depth = 2
        self.kg_depth = 2
        self.u_depth = 2
        self.gk_depth = 2

        self.tmem_free_bar = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.CTA_THREADS,
        )
        self.align = 1024

    def get_name(self) -> str:
        """Return a stable profiler and artifact name for this specialization."""
        return f"delta_affine_summary_fwd_h{self.heads}_bn{self.BN}_{self.dtype_name}"

    # ------------------------------------------------------------------
    # Host-side setup (__call__): GMEM views → MMA → TMA → SMEM → launch
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mKg: cute.Tensor,
        mW: cute.Tensor,
        mU: cute.Tensor,
        mG: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launch one CTA per (augmented-state column tile, head)."""
        # Re-rank the dense [1, T, H, D] inputs into the logical TMA views.
        g_w = _sequence_feature_head_batch_view(mW)
        g_kgt = _feature_sequence_head_batch_view(mKg)
        g_u_vt = _feature_sequence_head_batch_view(mU)
        g_gk_k = _feature_sequence_head_batch_view(mG)

        # --- MMA configurations (SS-mode: both operands from SMEM) ---
        # MMA1: w (A, K-major) × X snapshot (B, K-major) → wx.
        mma_wx = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.wx_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # MMA2: kg^T (A, MN-major) × Tmp (B, K-major) → kt.
        mma_kt = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.kt_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # MMA2a: kg^T (A, MN-major) × U (B, MN-major, V-contiguous TMA) → kt.
        # Same accumulator tile as MMA2; only the B major mode differs.
        mma_ktu = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.kt_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        self.tm_wx, self.tm_kt, self.tm_tot = self._plan_tmem(mma_wx, mma_kt)

        # --- SMEM staged layouts ---
        # A operands ride their TMA rings; B operands hold hi/lo halves of the
        # fp32-valued state and Tmp tiles (2 "stages" = the two MMA passes).
        s_w_staged = sm100_utils.make_smem_layout_a(
            mma_wx,
            self.wx_tile,
            self.io_type,
            self.w_depth,
        )
        s_xb_staged = sm100_utils.make_smem_layout_b(mma_wx, self.wx_tile, self.io_type, 2)
        s_xb_store_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
            (KEY_DIM, self.BN),
            2,
        )
        s_kg_staged = sm100_utils.make_smem_layout_a(
            mma_kt,
            self.kt_tile,
            self.io_type,
            self.kg_depth,
        )
        # u is TMA-loaded directly into its own MMA2a B-operand ring.
        s_u_staged = sm100_utils.make_smem_layout_b(
            mma_ktu,
            self.kt_tile,
            self.io_type,
            self.u_depth,
        )
        s_tmpb_staged = sm100_utils.make_smem_layout_b(mma_kt, self.kt_tile, self.io_type, 2)
        s_tmpb_store_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
            (BT, self.BN),
            2,
        )

        clust_lay = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (mma_wx.thr_id.shape,),
        )

        # --- TMA descriptors ---
        tma_ld = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        s_w_one = cute.select(s_w_staged, mode=[0, 1, 2])
        atom_w, desc_w = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld,
            g_w,
            s_w_one,
            self.wx_tile,
            mma_wx,
            clust_lay.shape,
        )
        s_kg_one = cute.select(s_kg_staged, mode=[0, 1, 2])
        atom_kg, desc_kg = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld,
            g_kgt,
            s_kg_one,
            self.kt_tile,
            mma_kt,
            clust_lay.shape,
        )
        s_u_one = cute.select(s_u_staged, mode=[0, 1, 2])
        atom_u, desc_u = cute.nvgpu.make_tiled_tma_atom_B(
            tma_ld,
            g_u_vt,
            s_u_one,
            self.kt_tile,
            mma_ktu,
            clust_lay.shape,
        )
        # gk: K×1 fp32 tile (last timestep per chunk).
        s_gk_2d = cute.make_layout((KEY_DIM, 1))
        atom_gk, desc_gk = cpasync.make_tiled_tma_atom(
            tma_ld,
            g_gk_k,
            s_gk_2d,
            (KEY_DIM, 1),
        )

        self.w_bytes = cute.size_in_bytes(self.io_type, s_w_one)
        self.kg_bytes = cute.size_in_bytes(self.io_type, s_kg_one)
        self.u_bytes = cute.size_in_bytes(self.io_type, s_u_one)
        self.gk_bytes = cute.size_in_bytes(Float32, s_gk_2d)

        io_type = self.io_type
        align = self.align

        @cute.struct
        class Shared:
            bar_w: cute.struct.MemRange[Int64, self.w_depth * 2]
            bar_kg: cute.struct.MemRange[Int64, self.kg_depth * 2]
            bar_u: cute.struct.MemRange[Int64, self.u_depth * 2]
            bar_gk: cute.struct.MemRange[Int64, self.gk_depth * 2]
            bar_xb: cute.struct.MemRange[Int64, 1 * 2]
            bar_tmpb: cute.struct.MemRange[Int64, 1 * 2]
            bar_wx: cute.struct.MemRange[Int64, 1 * 2]
            bar_kt: cute.struct.MemRange[Int64, 1 * 2]
            tmem_buf: Int32
            sW: cute.struct.Align[cute.struct.MemRange[io_type, cute.cosize(s_w_staged)], align]
            sKg: cute.struct.Align[cute.struct.MemRange[io_type, cute.cosize(s_kg_staged)], align]
            sXb: cute.struct.Align[cute.struct.MemRange[io_type, cute.cosize(s_xb_staged)], align]
            sTmpb: cute.struct.Align[
                cute.struct.MemRange[io_type, cute.cosize(s_tmpb_staged)], align
            ]
            sU: cute.struct.Align[cute.struct.MemRange[io_type, cute.cosize(s_u_staged)], align]
            sGK: cute.struct.Align[cute.struct.MemRange[Float32, KEY_DIM * self.gk_depth], 128]

        self.shared_type = Shared

        num_chunks = Int32(cute.size(mW.shape[1])) // BT
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            Mmas(mma_wx, mma_kt, mma_ktu),
            TmaOps(
                TmaOp(atom_w, desc_w),
                TmaOp(atom_kg, desc_kg),
                TmaOp(atom_u, desc_u),
                TmaOp(atom_gk, desc_gk),
            ),
            mOut,
            SmemLayouts(
                s_w_staged,
                s_kg_staged,
                s_u_staged,
                s_xb_staged,
                s_xb_store_staged,
                s_tmpb_staged,
                s_tmpb_store_staged,
            ),
            num_chunks,
        ).launch(
            grid=(SUMMARY_DIM // self.BN, self.heads, 1),
            block=(self.CTA_THREADS, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    def _plan_tmem(self, mma_wx: cute.TiledMma, mma_kt: cute.TiledMma) -> tuple[int, int, int]:
        """Assign disjoint TMEM column regions to the two accumulators."""
        wx_shape = mma_wx.partition_shape_C(self.wx_tile[:2])
        n_wx = tcgen05.find_tmem_tensor_col_offset(
            mma_wx.make_fragment_C(cute.append(wx_shape, 1))
        )
        kt_shape = mma_kt.partition_shape_C(self.kt_tile[:2])
        n_kt = tcgen05.find_tmem_tensor_col_offset(
            mma_kt.make_fragment_C(cute.append(kt_shape, 1))
        )
        raw = n_wx + n_kt
        total = 32
        while total < raw:
            total *= 2
        assert total <= 512, f"TMEM overflow: {total}>512"
        return 0, n_wx, total

    # ------------------------------------------------------------------
    # Warp roles
    # ------------------------------------------------------------------

    @cute.jit
    def run_load(
        self,
        head,
        col_tile,
        is_value,
        num_chunks,
        tma,
        mma_wx,
        mma_kt,
        mma_ktu,
        sW,
        sKg,
        sU,
        gk_3d,
        pw_P,
        pkg_P,
        pu_P,
        pgk_P,
    ):
        """Own the per-chunk TMA G2S loads for w, kg^T, u, and the gate row."""
        batch = Int32(0)
        tWs, tWg = self._part_a(tma.w.atom, tma.w.desc, sW, self.wx_tile, mma_wx, batch, head)
        tKgs, tKgg = self._part_a(tma.kg.atom, tma.kg.desc, sKg, self.kt_tile, mma_kt, batch, head)
        sUp, gUp = self._part_b(tma.u.atom, tma.u.desc, sU, self.kt_tile, mma_ktu, batch, head)
        gGK_l = tma.gk.desc[None, None, (head, batch)]
        sGKp, gGKp = self._part_epi(tma.gk.atom, gGK_l, (KEY_DIM, 1), gk_3d)

        for ct in cutlass.range(0, num_chunks, unroll=0):
            wh = pw_P.acquire_and_advance()
            cute.copy(
                atom=tma.w.atom,
                src=tWg[None, ct, 0],
                dst=tWs[None, wh.index],
                tma_bar_ptr=wh.barrier,
            )
            kgh = pkg_P.acquire_and_advance()
            cute.copy(
                atom=tma.kg.atom,
                src=tKgg[None, 0, ct],
                dst=tKgs[None, kgh.index],
                tma_bar_ptr=kgh.barrier,
            )
            if is_value:
                uh = pu_P.acquire_and_advance()
                cute.copy(
                    atom=tma.u.atom,
                    src=gUp[None, col_tile, ct],
                    dst=sUp[None, uh.index],
                    tma_bar_ptr=uh.barrier,
                )
            gkh = pgk_P.acquire_and_advance()
            cute.copy(
                atom=tma.gk.atom,
                src=gGKp[(None, 0, ct * BT + BT - 1)],
                dst=sGKp[None, gkh.index],
                tma_bar_ptr=gkh.barrier,
            )

    @cute.jit
    def run_mma(
        self,
        is_value,
        num_chunks,
        mma_wx,
        mma_kt,
        mma_ktu,
        t_wx_acc,
        t_w_a,
        t_xb_b,
        t_kt_acc,
        t_kg_a,
        t_kg_au,
        t_u_b,
        t_tmpb_b,
        pw_C,
        pkg_C,
        pu_C,
        pxb_C,
        ptmpb_C,
        pwx_P,
        pkt_P,
    ):
        """Consume operand stages and produce the two TMEM results per chunk.

        The kt accumulator gathers ``Kg^T @ U`` (value tiles, straight from the
        TMA ring, off the recurrence critical path) plus two hi/lo accumulate
        passes over ``Kg^T @ (-wx)``; MMA1 likewise runs two passes over the
        hi/lo state snapshot, recovering ~fp32 operand precision.
        """
        for _ct in cutlass.range(0, num_chunks, unroll=0):
            # --- MMA2a: kg^T(SMEM A) × U(SMEM B) → kt(TMEM), value tiles only ---
            kgh = pkg_C.wait_and_advance()
            ktd = pkt_P.acquire_and_advance()
            if is_value:
                uh = pu_C.wait_and_advance()
                for kp in cutlass.range_constexpr(cute.size(t_kg_au, mode=[2])):
                    mma_ktu.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma_ktu,
                        t_kt_acc[None, None, None, ktd.index],
                        t_kg_au[None, None, kp, kgh.index],
                        t_u_b[None, None, kp, uh.index],
                        t_kt_acc[None, None, None, ktd.index],
                    )
                uh.release()

            # --- MMA1: w(SMEM A) × X snapshot(SMEM B) → wx(TMEM) ---
            xbh = pxb_C.wait_and_advance()
            wh = pw_C.wait_and_advance()
            wxd = pwx_P.acquire_and_advance()
            for split in cutlass.range_constexpr(2):
                for kp in cutlass.range_constexpr(cute.size(t_w_a, mode=[2])):
                    mma_wx.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(split != 0 or kp != 0))
                    cute.gemm(
                        mma_wx,
                        t_wx_acc[None, None, None, wxd.index],
                        t_w_a[None, None, kp, wh.index],
                        t_xb_b[None, None, kp, split],
                        t_wx_acc[None, None, None, wxd.index],
                    )
            wxd.commit()
            wh.release()
            xbh.release()

            # --- MMA2b: kg^T(SMEM A) × -wx hi/lo(SMEM B) → kt(TMEM) ---
            tmpbh = ptmpb_C.wait_and_advance()
            for split in cutlass.range_constexpr(2):
                for kp in cutlass.range_constexpr(cute.size(t_kg_a, mode=[2])):
                    # The first pass overwrites unless MMA2a already seeded kt.
                    first = split == 0 and kp == 0
                    mma_kt.set(tcgen05.Field.ACCUMULATE, is_value if first else True)
                    cute.gemm(
                        mma_kt,
                        t_kt_acc[None, None, None, ktd.index],
                        t_kg_a[None, None, kp, kgh.index],
                        t_tmpb_b[None, None, kp, split],
                        t_kt_acc[None, None, None, ktd.index],
                    )
            ktd.commit()
            kgh.release()
            tmpbh.release()

    @cute.jit
    def run_state(
        self,
        head,
        col_base,
        num_chunks,
        mOut,
        t_wx_acc,
        t_kt_acc,
        sXb_store,
        sTmpb_store,
        gk_buf,
        pxb_P,
        ptmpb_P,
        pgk_C,
        pwx_C,
        pkt_C,
    ):
        """Own the fp32 state registers; produce B operands and fold in results."""
        tid, _, _ = cute.arch.thread_idx()
        local_tid = tid % (self.WARP_SZ * len(self.CUDA_WARP_IDS))

        # --- T2R setup: read MMA1 (wx) accumulator (BT, BN fp32) from TMEM ---
        t2r_wx_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BN // 8), tcgen05.Pack.NONE),
            Float32,
        )
        wx_flat = t_wx_acc[((None, None), 0, 0, None)]
        tc_t2r_wx = tcgen05.make_tmem_copy(t2r_wx_atom, wx_flat[(None, None, 0)])
        sl_wx = tc_t2r_wx.get_slice(local_tid)
        p_t_wx = sl_wx.partition_S(wx_flat)

        # --- T2R setup: read MMA2 (kt) accumulator (K, BN fp32) from TMEM ---
        t2r_kt_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BN // 8), tcgen05.Pack.NONE),
            Float32,
        )
        kt_flat = t_kt_acc[((None, None), 0, 0, None)]
        tc_t2r_kt = tcgen05.make_tmem_copy(t2r_kt_atom, kt_flat[(None, None, 0)])
        sl_kt = tc_t2r_kt.get_slice(local_tid)
        p_t_kt = sl_kt.partition_S(kt_flat)

        # Identity tensors provide both coordinate maps and fragment shapes.
        coords_tv = sl_wx.partition_D(cute.make_identity_tensor((BT, self.BN)))
        coords_kv = sl_kt.partition_D(cute.make_identity_tensor((KEY_DIM, self.BN)))

        # R2S X snapshot → sXb (COL_MAJOR, partition matches T2R kt).
        r2s_atom_xb = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.COL_MAJOR,
            self.io_type,
            Float32,
            tc_t2r_kt,
        )
        tc_r2s_xb = cute.make_tiled_copy_D(r2s_atom_xb, tc_t2r_kt)
        thr_r2s_xb = tc_r2s_xb.get_slice(local_tid)

        # R2S Tmp → sTmpb (COL_MAJOR, partition matches T2R wx).
        r2s_atom_tmpb = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.COL_MAJOR,
            self.io_type,
            Float32,
            tc_t2r_wx,
        )
        tc_r2s_tmpb = cute.make_tiled_copy_D(r2s_atom_tmpb, tc_t2r_wx)
        thr_r2s_tmpb = tc_r2s_tmpb.get_slice(local_tid)

        # X starts as this tile's slice of [0 | I]: value-tile columns are zero
        # and identity-tile column (col_base + nc) carries a one at key row kc.
        st = cute.make_rmem_tensor(coords_kv.shape, Float32)
        st.fill(Float32(0.0))
        for ei in cutlass.range(cute.size(st), unroll_full=True):
            kc, nc = coords_kv[ei]
            if col_base + nc == VAL_DIM + kc:
                st[ei] = Float32(1.0)

        for _ct in cutlass.range(0, num_chunks, unroll=0):
            # ---- Phase 1: R2S X hi/lo → sXb (B operand for MMA1) ----
            xbh = pxb_P.acquire_and_advance()
            st_hi = cute.make_rmem_tensor(st.shape, self.io_type)
            st_hi.store(st.load().to(self.io_type))
            st_lo = cute.make_rmem_tensor(st.shape, self.io_type)
            st_lo.store((st.load() - st_hi.load().to(Float32)).to(self.io_type))
            cute.copy(
                tc_r2s_xb,
                tc_r2s_xb.retile(st_hi),
                thr_r2s_xb.partition_D(sXb_store[(None, None, 0)]),
            )
            cute.copy(
                tc_r2s_xb,
                tc_r2s_xb.retile(st_lo),
                thr_r2s_xb.partition_D(sXb_store[(None, None, 1)]),
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            xbh.commit()

            # ---- Phase 2: T2R wx = W @ X ----
            wxh = pwx_C.wait_and_advance()
            wx_reg = cute.make_rmem_tensor(coords_tv.shape, Float32)
            cute.copy(tc_t2r_wx, p_t_wx[(None, None, None, wxh.index)], wx_reg)
            cute.arch.fence_view_async_tmem_load()
            wxh.release()

            # ---- Phase 3: R2S -wx hi/lo (the U term accumulates via MMA2a) ----
            tmpbh = ptmpb_P.acquire_and_advance()
            tmp = cute.make_rmem_tensor(coords_tv.shape, Float32)
            for ei in cutlass.range_constexpr(cute.size(tmp)):
                tmp[ei] = Float32(0.0) - wx_reg[ei]
            tmp_hi = cute.make_rmem_tensor(tmp.shape, self.io_type)
            tmp_hi.store(tmp.load().to(self.io_type))
            tmp_lo = cute.make_rmem_tensor(tmp.shape, self.io_type)
            tmp_lo.store((tmp.load() - tmp_hi.load().to(Float32)).to(self.io_type))
            cute.copy(
                tc_r2s_tmpb,
                tc_r2s_tmpb.retile(tmp_hi),
                thr_r2s_tmpb.partition_D(sTmpb_store[(None, None, 0)]),
            )
            cute.copy(
                tc_r2s_tmpb,
                tc_r2s_tmpb.retile(tmp_lo),
                thr_r2s_tmpb.partition_D(sTmpb_store[(None, None, 1)]),
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            tmpbh.commit()

            # ---- Phase 4: decay X, then add kt = Kg^T @ Tmp ----
            gkh = pgk_C.wait_and_advance()
            for ei in cutlass.range(cute.size(st), unroll_full=True):
                kc, nc = coords_kv[ei]
                st[ei] = st[ei] * cute.math.exp2(gk_buf[kc, gkh.index], fastmath=True)
            gkh.release()

            kth = pkt_C.wait_and_advance()
            kt_reg = cute.make_rmem_tensor(coords_kv.shape, Float32)
            cute.copy(tc_t2r_kt, p_t_kt[(None, None, None, kth.index)], kt_reg)
            cute.arch.fence_view_async_tmem_load()
            kth.release()
            for ei in cutlass.range_constexpr(cute.size(st)):
                st[ei] = st[ei] + kt_reg[ei]

        # Epilogue: store the V-first packed transpose out[h, col, k] = X[k, col].
        for ei in cutlass.range(cute.size(st), unroll_full=True):
            kc, nc = coords_kv[ei]
            mOut[head, col_base + nc, kc] = st[ei]

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mmas: Mmas,
        tma: TmaOps,
        mOut: cute.Tensor,
        smem_layouts: SmemLayouts,
        num_chunks: Int32,
    ):
        mma_wx, mma_kt, mma_ktu = mmas
        (
            s_w_staged,
            s_kg_staged,
            s_u_staged,
            s_xb_staged,
            s_xb_store_staged,
            s_tmpb_staged,
            s_tmpb_store_staged,
        ) = smem_layouts
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_id == WarpRole.LOAD:
            cpasync.prefetch_descriptor(tma.w.atom)
            cpasync.prefetch_descriptor(tma.kg.atom)
            cpasync.prefetch_descriptor(tma.u.atom)
            cpasync.prefetch_descriptor(tma.gk.atom)

        sa = utils.SmemAllocator()
        sm = sa.allocate(self.shared_type)

        gk_3d = sm.sGK.get_tensor(
            cute.make_layout(
                (KEY_DIM, 1, self.gk_depth),
                stride=(1, KEY_DIM, KEY_DIM),
            )
        )
        gk_buf = gk_3d[(None, 0, None)]

        # #
        # Pipeline creation
        #
        n_cuda = self.WARP_SZ * len(self.CUDA_WARP_IDS)

        # w → MMA1 (TmaUmma)
        pw_P, pw_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.w_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.w_bytes,
            barrier_storage=sm.bar_w.data_ptr(),
        ).make_participants()

        # kg^T → MMA2 (TmaUmma)
        pkg_P, pkg_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.kg_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.kg_bytes,
            barrier_storage=sm.bar_kg.data_ptr(),
        ).make_participants()

        # u → MMA2 (TmaUmma)
        pu_P, pu_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.u_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.u_bytes,
            barrier_storage=sm.bar_u.data_ptr(),
        ).make_participants()

        # gk → CUDA warps (TmaAsync). Each consuming warp contributes one
        # elected-lane arrival, so the group count is the warp count, not 128 threads.
        pgk_P, pgk_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.gk_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.CUDA_WARP_IDS)
            ),
            tx_count=self.gk_bytes,
            barrier_storage=sm.bar_gk.data_ptr(),
        ).make_participants()

        # X snapshot → MMA1 (AsyncUmma, 1 stage covering both hi/lo halves)
        pxb_P, pxb_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, n_cuda),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=sm.bar_xb.data_ptr(),
        ).make_participants()

        # Tmp → MMA2 (AsyncUmma, 1 stage covering both hi/lo halves)
        ptmpb_P, ptmpb_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, n_cuda),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=sm.bar_tmpb.data_ptr(),
        ).make_participants()

        # MMA1 done → CUDA (UmmaAsync)
        pwx_P, pwx_C = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, n_cuda),
            barrier_storage=sm.bar_wx.data_ptr(),
        ).make_participants()

        # MMA2 done → CUDA (UmmaAsync)
        pkt_P, pkt_C = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, n_cuda),
            barrier_storage=sm.bar_kt.data_ptr(),
        ).make_participants()

        # #
        # TMEM allocation
        #
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.CTA_THREADS)
        tmem = utils.TmemAllocator(
            sm.tmem_buf,
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=WarpRole.LOAD,
        )
        tmem.allocate(self.tm_tot)
        tmem.wait_for_alloc()
        tp = tmem.retrieve_ptr(Float32)

        # #
        # SMEM view tensors and MMA fragments
        #
        sW = sm.sW.get_tensor(s_w_staged.outer, swizzle=s_w_staged.inner)
        sKg = sm.sKg.get_tensor(s_kg_staged.outer, swizzle=s_kg_staged.inner)
        sU = sm.sU.get_tensor(s_u_staged.outer, swizzle=s_u_staged.inner)
        sXb = sm.sXb.get_tensor(s_xb_staged.outer, swizzle=s_xb_staged.inner)
        sXb_store = sm.sXb.get_tensor(s_xb_store_staged.outer, swizzle=s_xb_store_staged.inner)
        sTmpb = sm.sTmpb.get_tensor(s_tmpb_staged.outer, swizzle=s_tmpb_staged.inner)
        sTmpb_store = sm.sTmpb.get_tensor(
            s_tmpb_store_staged.outer, swizzle=s_tmpb_store_staged.inner
        )

        t_w_a = mma_wx.make_fragment_A(sW)
        t_xb_b = mma_wx.make_fragment_B(sXb)
        wx_shape = mma_wx.partition_shape_C(self.wx_tile[:2])
        wx_fk = mma_wx.make_fragment_C(cute.append(wx_shape, 1))
        t_wx_acc = cute.make_tensor(tp + self.tm_wx, wx_fk.layout)

        t_kg_a = mma_kt.make_fragment_A(sKg)
        t_kg_au = mma_ktu.make_fragment_A(sKg)
        t_u_b = mma_ktu.make_fragment_B(sU)
        t_tmpb_b = mma_kt.make_fragment_B(sTmpb)
        kt_shape = mma_kt.partition_shape_C(self.kt_tile[:2])
        kt_fk = mma_kt.make_fragment_C(cute.append(kt_shape, 1))
        t_kt_acc = cute.make_tensor(tp + self.tm_kt, kt_fk.layout)

        # #
        # Role dispatch: one CTA owns one (column tile, head) state slice.
        #
        col_tile, head, _ = cute.arch.block_idx()
        col_base = col_tile * self.BN
        is_value = col_base < VAL_DIM

        if warp_id in self.CUDA_WARP_IDS:
            self.run_state(
                head=head,
                col_base=col_base,
                num_chunks=num_chunks,
                mOut=mOut,
                t_wx_acc=t_wx_acc,
                t_kt_acc=t_kt_acc,
                sXb_store=sXb_store,
                sTmpb_store=sTmpb_store,
                gk_buf=gk_buf,
                pxb_P=pxb_P,
                ptmpb_P=ptmpb_P,
                pgk_C=pgk_C,
                pwx_C=pwx_C,
                pkt_C=pkt_C,
            )
        elif warp_id == WarpRole.LOAD:
            self.run_load(
                head=head,
                col_tile=col_tile,
                is_value=is_value,
                num_chunks=num_chunks,
                tma=tma,
                mma_wx=mma_wx,
                mma_kt=mma_kt,
                mma_ktu=mma_ktu,
                sW=sW,
                sKg=sKg,
                sU=sU,
                gk_3d=gk_3d,
                pw_P=pw_P,
                pkg_P=pkg_P,
                pu_P=pu_P,
                pgk_P=pgk_P,
            )
        elif warp_id == WarpRole.MMA:
            self.run_mma(
                is_value=is_value,
                num_chunks=num_chunks,
                mma_wx=mma_wx,
                mma_kt=mma_kt,
                mma_ktu=mma_ktu,
                t_wx_acc=t_wx_acc,
                t_w_a=t_w_a,
                t_xb_b=t_xb_b,
                t_kt_acc=t_kt_acc,
                t_kg_a=t_kg_a,
                t_kg_au=t_kg_au,
                t_u_b=t_u_b,
                t_tmpb_b=t_tmpb_b,
                pw_C=pw_C,
                pkg_C=pkg_C,
                pu_C=pu_C,
                pxb_C=pxb_C,
                ptmpb_C=ptmpb_C,
                pwx_P=pwx_P,
                pkt_P=pkt_P,
            )

        # TMEM teardown (all warps)
        tmem.relinquish_alloc_permit()
        self.tmem_free_bar.arrive_and_wait()
        tmem.free(tp)

    # ------------------------------------------------------------------
    # TMA partition helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _part_a(self, atom, desc, smem, tile, mma, batch, head):
        """Partition an A operand for TMA copy (SS-mode)."""
        g = cute.local_tile(desc, cute.slice_(tile, (None, 0, None)), (None, None, (head, batch)))
        part = mma.get_slice(0).partition_A(g)
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(part, 0, 3),
        )

    @cute.jit
    def _part_b(self, atom, desc, smem, tile, mma, batch, head):
        """Partition a TMA tensor as an MMA B operand."""
        g = cute.local_tile(desc, cute.slice_(tile, (0, None, None)), (None, None, (head, batch)))
        part = mma.get_slice(0).partition_B(g)
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(part, 0, 3),
        )

    @cute.jit
    def _part_epi(self, atom, g_mnl, tile, s_buf):
        """Partition for epilogue-style TMA."""
        g_div = cute.flat_divide(g_mnl, tile)
        sg = cute.group_modes(s_buf, 0, 2)
        gg = cute.group_modes(g_div, 0, 2)
        return cpasync.tma_partition(atom, 0, cute.make_layout(1), sg, gg)


@jit_cache
def _compile_affine_summary(dtype_name: str, heads: int, state_bn: int):
    """Compile one dtype/head-count specialization from fake tensors."""
    target = get_compile_target()
    if target.device_type != "cuda" or target.capability is None or target.capability < (10, 0):
        raise ValueError(
            f"affine_summary_fwd requires an SM100 (CUDA capability >= 10.0) target; "
            f"got target={target}"
        )
    tokens = cute.sym_int(divisibility=BT)

    def factor(dtype, last_dim: int):
        return make_fake_compact_tensor(
            dtype,
            (1, tokens, heads, last_dim),
            stride_order=(3, 2, 1, 0),
            assumed_align=DATA_ALIGN_BYTES,
        )

    io_dtype = _CUTE_IO_TYPES[dtype_name]
    out = make_fake_compact_tensor(
        Float32,
        (heads, SUMMARY_DIM, KEY_DIM),
        stride_order=(2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    return compile_tvm_ffi(
        _AffineSummaryFwdOp(dtype_name, heads, state_bn),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, VAL_DIM),
        factor(Float32, KEY_DIM),
        out,
    )


def affine_summary_fwd(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    cumulative_gate: torch.Tensor,
) -> torch.Tensor:
    """Compute one shard's packed affine state summary from its WY chunk factors.

    Args:
        kg: Gated keys ``k * exp2(gk_last - gk)``, shape ``[1, T, H, 128]``,
            bf16 or fp16.
        w: WY factor ``w``, same shape and dtype as ``kg``.
        u: WY pseudo-values ``u``, shape ``[1, T, H, 128]``, same dtype as ``kg``.
        cumulative_gate: Cumulative log2 gates, shape ``[1, T, H, 128]``, fp32.

    Returns:
        FP32 tensor of shape ``[H, 256, 128]``, V-first packed as state bias then
        state transition for ``H_out = H_in @ transition + bias``.

    Supported scope: B=1, fixed K=V=128 and BT=64, contiguous int32-addressable
    inputs, SM100. A partial final chunk is neutral-padded by the wrapper.
    """
    assert kg.ndim == 4, f"kg must be 4D [1, T, H, K], got shape {tuple(kg.shape)}"
    batch, tokens, heads, key_dim = kg.shape
    assert batch == 1, f"affine_summary_fwd requires B=1, got B={batch}"
    assert tokens > 0, "affine_summary_fwd requires at least one token"
    assert key_dim == KEY_DIM, f"affine_summary_fwd requires K={KEY_DIM}, got {key_dim}"
    assert w.shape == kg.shape, f"w must match kg shape {tuple(kg.shape)}, got {tuple(w.shape)}"
    assert u.shape == (1, tokens, heads, VAL_DIM), (
        f"u must have shape {(1, tokens, heads, VAL_DIM)}, got {tuple(u.shape)}"
    )
    assert cumulative_gate.shape == kg.shape, (
        f"cumulative_gate must match kg shape {tuple(kg.shape)}, "
        f"got {tuple(cumulative_gate.shape)}"
    )
    assert kg.dtype in _IO_TYPE_NAMES, f"kg dtype must be bf16 or fp16, got {kg.dtype}"
    assert w.dtype == kg.dtype and u.dtype == kg.dtype, (
        f"kg/w/u dtypes must match, got {kg.dtype}/{w.dtype}/{u.dtype}"
    )
    assert cumulative_gate.dtype == torch.float32, (
        f"cumulative_gate must be fp32, got {cumulative_gate.dtype}"
    )

    out = torch.empty(
        (heads, SUMMARY_DIM, KEY_DIM),
        dtype=torch.float32,
        device=kg.device,
    )
    if isinstance(kg, FakeTensor):
        return out

    assert kg.is_cuda, "affine_summary_fwd requires CUDA tensors"
    for name, tensor in (("kg", kg), ("w", w), ("u", u), ("cumulative_gate", cumulative_gate)):
        assert tensor.device == kg.device, f"{name} must be on {kg.device}, got {tensor.device}"
        assert tensor.is_contiguous(), f"{name} must be contiguous, got strides {tensor.stride()}"
        assert tensor.data_ptr() % DATA_ALIGN_BYTES == 0, (
            f"{name} data pointer must be {DATA_ALIGN_BYTES}-byte aligned"
        )
    pad = (-tokens) % BT
    if pad:
        padding = (0, 0, 0, 0, 0, pad)
        kg, w, u = (F.pad(tensor, padding) for tensor in (kg, w, u))
        cumulative_gate = torch.cat(
            (
                cumulative_gate,
                cumulative_gate[:, -1:].expand(-1, pad, -1, -1),
            ),
            dim=1,
        )
    assert not requires_int64_abi(kg, w, u, cumulative_gate, out), (
        "affine_summary_fwd currently requires int32-addressable tensors"
    )

    # BN=32 is faster per CTA (measured ~1.5x vs BN=64 on GB200); fall back to
    # BN=64 only when the extra column tiles would spill past one CTA wave and
    # serialize whole recurrence chains.
    sm_count = get_device_properties(kg.device).multi_processor_count
    state_bn = 32 if heads * (SUMMARY_DIM // 32) <= sm_count else 64
    compiled = _compile_affine_summary(_IO_TYPE_NAMES[kg.dtype], heads, state_bn)
    compiled(kg.detach(), w.detach(), u.detach(), cumulative_gate.detach(), out)
    return out
