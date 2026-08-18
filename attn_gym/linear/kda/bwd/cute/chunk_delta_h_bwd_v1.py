# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""
BlackwellDeltaHBwdV1 — SM100 warp-specialized kernel for KDA backward dhu recurrence.

Implements the backward hidden-state gradient update across chunks (reverse order):

    dh_out[c]  = snapshot(dh)
    dv[c]      = GEMM_DV(K[c], dh) * gate + dv_intra[c]
    dh         = diag(exp2(gk_last[c] + g_last[c])) @ dh
                 + GEMM_QDO(Q[c]^T, do[c]) * scale
                 - GEMM_WDV(W[c], dv[c])

Architecture follows FMHA/MSLK Blackwell patterns:
  - 8-warp CTA (256 threads), occupancy 1 for maximum SMEM/regs
  - Separate warp roles: CUDA(0-3), Load(4), GK(5), MMA(6), Store(7)
  - SS-mode MMA: both operands from SMEM, accumulators in TMEM
  - 3 MMAs per chunk: MMA1(k@dh→dv), MMA2(q^T@do→qdo), MMA3(w@dv2→wdv)
  - MMA2 uses MN-major B-operand for direct TMA loading of do
  - Reverse chunk iteration: NT-1 down to 0
"""

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Float32, Int32, Int64
from torch._guards import active_fake_mode

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import compile_tvm_ffi, requires_int64_abi
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata

# ============================================================================
# BlackwellDeltaHBwdV1 — warp-specialized backward inter-chunk recurrence
# ============================================================================


class BlackwellDeltaHBwdV1:
    """
    Warp-specialized SM100 kernel for gated delta rule backward dhu recurrence.

    dh state persists in CUDA warp registers (FP32) across all chunks — zero
    GMEM round-trips.  SS-mode (both operands from SMEM) with BV=16.

    3 MMA operations per chunk:
      MMA1: k @ dh → dv          (BT=64,  BV=16, K=128)
      MMA2: q^T @ do → qdo       (BK=128, BV=16, K=64)
      MMA3: w @ dv2 → wdv        (BK=128, BV=16, K=64)
    """

    # Warp role assignment (8 warps total)
    CUDA_WARP_IDS = (0, 1, 2, 3)
    LOAD_WARP_ID = 4  # TMA G2S producer
    GK_WARP_ID = 5  # dedicated gk+g exp2 precompute
    MMA_WARP_ID = 6  # tcgen05 3×GEMM
    STORE_WARP_ID = 7  # TMA S2G consumer
    WARP_SZ = 32
    N_WARPS = 8
    CTA_THREADS = N_WARPS * WARP_SZ  # 256

    def __init__(
        self,
        chunk_size: int = 64,
        head_k: int = 128,
        head_v: int = 128,
        head_bv: int = 16,
        acc_type=cutlass.Float32,
        io_type=cutlass.BFloat16,
        varlen: bool = False,
        num_heads: int | None = None,
        use_int64_offsets: bool = False,
    ):
        assert head_k == 128 and head_v == 128, (
            f"Only head_k=head_v=128 supported, got {head_k},{head_v}"
        )
        assert head_bv in (16, 32), f"BV must be 16 or 32, got {head_bv}"
        self.chunk_size = chunk_size
        self.head_k = head_k
        self.head_v = head_v
        self.acc_type = acc_type
        self.io_type = io_type
        self.varlen = varlen
        self.num_heads = num_heads
        self.use_int64_offsets = use_int64_offsets

        # Tile dimensions
        self.BT = chunk_size  # 64
        self.BK = head_k  # 128
        self.BV = head_bv  # N-dim for SS-mode MMA (16 or 32)

        # Register budget per thread — BV=32 needs more for wider dh state
        self.cuda_regs = 128 if head_bv <= 16 else 160
        self.aux_regs = 40  # MMA/Load/Store/GK: minimal

        self.min_occ = 1

        # MMA tile shapes (M, N, K)
        # MMA1: k @ dh → dv — k is (BT,BK) K-major, dh is (BK,BV) K-major
        self.dv_tile = (self.BT, self.BV, self.BK)
        # MMA2: q^T @ do → qdo — q^T is (BK,BT) MN-major, do is (BT,BV) MN-major
        self.qdo_tile = (self.BK, self.BV, self.BT)
        # MMA3: w @ dv2 → wdv — w is (BK,BT) MN-major, dv2 is (BT,BV) K-major
        self.wdv_tile = (self.BK, self.BV, self.BT)

        # Pipeline depths — BV=32 uses shallower k pipeline to fit SMEM
        self.k_depth = 6 if head_bv <= 16 else 4
        self.q_depth = 2
        self.w_depth = 2
        self.do_depth = 2  # do B-operand (MN-major TMA)
        self.dv_in_depth = 2  # dv_intra load
        self.gk_depth = 2
        self.dh_epi_depth = 2  # dh snapshot store
        self.dv2_epi_depth = 2  # dv2 store
        self.dv_acc_depth = 1  # MMA1 accumulator
        self.qdo_acc_depth = 1
        self.wdv_acc_depth = 1

        self.cluster = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE

        # Named barriers
        self.tmem_free_bar = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.CTA_THREADS,
        )
        self.align = 1024

    @cute.jit
    def upcast(self, value):
        """Promote an address operand before its first overflowing multiply."""
        return cutlass.Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    def get_name(self) -> str:
        """Return a stable artifact and profiler name for this specialization."""
        head_tag = f"_h{self.num_heads}" if self.num_heads is not None else ""
        return (
            f"kda_bwd_dhu_v1_vl{int(self.varlen)}{head_tag}"
            f"_k{self.head_k}_v{self.head_v}_bt{self.BT}_bv{self.BV}"
            f"_i64{int(self.use_int64_offsets)}"
        )

    # ------------------------------------------------------------------
    # Host-side setup (__call__): GMEM layouts → TMA → SMEM → launch
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        w_in: cute.Tensor,
        do_in: cute.Tensor,
        dv_in: cute.Tensor,
        gk_in: cute.Tensor,
        dht_in: cute.Tensor,
        dh0_in: cute.Tensor,
        dh_out_in: cute.Tensor,
        dv2_out_in: cute.Tensor,
        cu_seqlens_in: cute.Tensor,
        chunk_offsets_in: cute.Tensor,
        problem_shape: tuple[Int32, Int32, Int32, Int32, Int32],
        scale: Float32,
        use_gk: Int32,
        use_dht: Int32,
        use_dh0: Int32,
        stream,
    ):
        # --- Extract raw pointers ---
        qp = q_in.iterator
        kp = k_in.iterator
        wp = w_in.iterator
        dop = do_in.iterator
        dvp = dv_in.iterator
        gkp = gk_in.iterator
        dhtp = dht_in.iterator
        dh0p = dh0_in.iterator
        dhop = dh_out_in.iterator
        dv2p = dv2_out_in.iterator
        cup = cu_seqlens_in.iterator
        cop = chunk_offsets_in.iterator

        B, T, H, K, V = problem_shape
        if cutlass.const_expr(self.varlen):
            dB = Int32(1)
            # The ragged chunk capacity arrives via the dh tensor's chunk axis.
            NT = Int32(cute.size(dh_out_in.shape[0]))
        else:
            dB = B
            NT = (T + self.BT - 1) // self.BT

        # --- GMEM tensor construction ---
        # K: (T, K, (H, dB)) — K-contiguous for MMA1 A-operand (k is K-major)
        g_k = cute.make_tensor(
            kp,
            cute.make_layout(
                (T, K, (H, dB)),
                stride=(self.upcast(H * K), 1, (K, self.upcast(T) * H * K)),
            ),
        )
        # Q^T: (K, T, (H, dB)) — K-contiguous, then transposed for MMA2 A-operand
        g_qt = cute.make_tensor(
            qp,
            cute.make_layout(
                (K, T, (H, dB)),
                stride=(1, self.upcast(H * K), (K, self.upcast(T) * H * K)),
            ),
        )
        # W: (K, T, (H, dB)) — K-contiguous for MMA3 A-operand (w^T)
        g_wt = cute.make_tensor(
            wp,
            cute.make_layout(
                (K, T, (H, dB)),
                stride=(1, self.upcast(H * K), (K, self.upcast(T) * H * K)),
            ),
        )
        # do transposed: (V, T, (H, dB)) — V-contiguous for MMA2 MN-major B-operand TMA
        g_do_vt = cute.make_tensor(
            dop,
            cute.make_layout(
                (V, T, (H, dB)),
                stride=(1, self.upcast(H * V), (V, self.upcast(T) * H * V)),
            ),
        )
        # dv_in: (T, V, (H, dB)) — V-contiguous
        dv_lay = cute.make_layout(
            (T, V, (H, dB)),
            stride=(self.upcast(H * V), 1, (V, self.upcast(T) * H * V)),
        )
        # dv_in transposed: (V, T, (H, dB))
        g_dv_in_t = cute.make_tensor(
            dvp,
            cute.make_layout(
                (V, T, (H, dB)),
                stride=(1, self.upcast(H * V), (V, self.upcast(T) * H * V)),
            ),
        )
        # dv2_out: (T, V, (H, dB))
        g_dv2 = cute.make_tensor(dv2p, dv_lay)
        # dv2_out transposed: (V, T, (H, dB))
        g_dv2_t = cute.make_tensor(
            dv2p,
            cute.make_layout(
                (V, T, (H, dB)),
                stride=(1, self.upcast(H * V), (V, self.upcast(T) * H * V)),
            ),
        )

        # dh_out transposed for TMA store: (V, K, (NT, H, dB))
        g_dh_out_t = cute.make_tensor(
            dhop,
            cute.make_layout(
                (V, K, (NT, H, dB)),
                stride=(
                    1,
                    V,
                    (self.upcast(H * K * V), K * V, self.upcast(NT) * H * K * V),
                ),
            ),
        )

        # dht: (K, V, (H, B)) — final state gradient input
        g_dht = cute.make_tensor(
            dhtp,
            cute.make_layout((K, V, (H, B)), stride=(V, 1, (K * V, self.upcast(H * K * V)))),
        )
        # dh0 transposed for store: (V, K, (H, B))
        g_dh0_t = cute.make_tensor(
            dh0p,
            cute.make_layout((V, K, (H, B)), stride=(1, V, (K * V, self.upcast(H * K * V)))),
        )

        # gk K-contiguous: (K, T, (H, dB))
        g_gk_k = cute.make_tensor(
            gkp,
            cute.make_layout(
                (K, T, (H, dB)),
                stride=(1, self.upcast(H * K), (K, self.upcast(T) * H * K)),
            ),
        )

        # --- MMA configurations (SS-mode: both operands from SMEM) ---
        # MMA1: k @ dh → dv  — k is (BT,BK) as A K-major, dh is (BK,BV) as B K-major
        mma_dv = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.K,  # A (k): K-major
            tcgen05.OperandMajorMode.K,  # B (dh): K-major
            self.acc_type,
            self.cta_group,
            self.dv_tile[:2],  # (BT=64, BV=16)
            tcgen05.OperandSource.SMEM,
        )
        # MMA2: q^T @ do → qdo  — q^T is (BK,BT) A MN-major, do is (BT,BV) B MN-major
        mma_qdo = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.MN,  # A (q^T): MN-major
            tcgen05.OperandMajorMode.MN,  # B (do): MN-major (V-contiguous)
            self.acc_type,
            self.cta_group,
            self.qdo_tile[:2],  # (BK=128, BV=16)
            tcgen05.OperandSource.SMEM,
        )
        # MMA3: w @ dv2 → wdv  — w is (BK,BT) A MN-major, dv2 is (BT,BV) B K-major
        mma_wdv = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.MN,  # A (w^T): MN-major
            tcgen05.OperandMajorMode.K,  # B (dv2): K-major
            self.acc_type,
            self.cta_group,
            self.wdv_tile[:2],  # (BK=128, BV=16)
            tcgen05.OperandSource.SMEM,
        )

        # Plan TMEM column allocation (3 separate accumulator regions)
        (self.tm_dv, self.tm_qdo, self.tm_wdv, self.tm_tot) = self._plan_tmem(
            mma_dv,
            self.dv_tile,
            self.dv_acc_depth,
            mma_qdo,
            self.qdo_tile,
            self.qdo_acc_depth,
            mma_wdv,
            self.wdv_tile,
            self.wdv_acc_depth,
        )

        # --- SMEM staged layouts ---
        tma_ld = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_st = cpasync.CopyBulkTensorTileS2GOp()

        # MMA1 A-operand: k — (BT, BK) K-major staged
        s_k_staged = sm100_utils.make_smem_layout_a(
            mma_dv,
            self.dv_tile,
            self.io_type,
            self.k_depth,
        )
        # MMA1 B-operand: dh — (BK, BV) K-major, 1-stage (CUDA R2S)
        s_dhb_staged = sm100_utils.make_smem_layout_b(
            mma_dv,
            self.dv_tile,
            self.io_type,
            1,
        )
        # R2S store view for dh B-operand
        s_dhb_store_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
            (self.BK, self.BV),
            1,
        )

        # MMA2 A-operand: q^T — (BK, BT) MN-major staged
        s_q_staged = sm100_utils.make_smem_layout_a(
            mma_qdo,
            self.qdo_tile,
            self.io_type,
            self.q_depth,
        )
        # MMA2 B-operand: do — (BT, BV) MN-major, 1-stage (CUDA R2S, masked).
        # Mirrors the dv2 B-operand path but MN-major: TMA loads do into sDoEpi,
        # the COMPUTE warp masks the partial-chunk over-read in registers and
        # R2S's into sDo, so the MMA warp does no masking on its critical path.
        s_do_staged = sm100_utils.make_smem_layout_b(
            mma_qdo,
            self.qdo_tile,
            self.io_type,
            1,
        )
        # R2S store view for do B-operand. do is MN-major (N=BV contiguous), so
        # the alias is ROW_MAJOR (BV inner) — NOT COL_MAJOR like the K-major dh/
        # dv2 B-operands. This matches the make_smem_layout_b swizzle (SW32 for
        # bv16 / SW64 for bv32).
        s_do_store_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BV),
            1,
        )
        # do load: (V, T) epilogue layout for TMA load into CUDA warps
        s_doepi_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),
            self.do_depth,
        )

        # MMA3 A-operand: w^T — (BK, BT) MN-major staged
        s_w_staged = sm100_utils.make_smem_layout_a(
            mma_wdv,
            self.wdv_tile,
            self.io_type,
            self.w_depth,
        )
        # MMA3 B-operand: dv2 — (BT, BV) K-major, 1-stage (CUDA R2S)
        s_dv2b_staged = sm100_utils.make_smem_layout_b(
            mma_wdv,
            self.wdv_tile,
            self.io_type,
            1,
        )
        # R2S store view for dv2 B-operand
        s_dv2b_store_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
            (self.BT, self.BV),
            1,
        )

        # dv_in load: (V, T) epilogue layout for TMA load into CUDA warps
        s_dvin_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),
            self.dv_in_depth,
        )

        # dh snapshot epilogue: (V, K) for Store warp TMA S2G
        s_dh_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BK),
            self.dh_epi_depth,
        )
        # R2S dh → sH_epi (ROW_MAJOR transposed for stmatrix)
        s_dh_r2s_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BK, self.BV),
            self.dh_epi_depth,
        )

        # dv2 epilogue: (V, T) for Store warp TMA S2G
        s_dv2_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),
            self.dv2_epi_depth,
        )
        # R2S dv → sVst (ROW_MAJOR transposed for stmatrix)
        s_dv2_r2s_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.ROW_MAJOR,
            (self.BT, self.BV),
            self.dv2_epi_depth,
        )

        clust_lay = cute.tiled_divide(
            cute.make_layout(self.cluster),
            (mma_dv.thr_id.shape,),
        )

        # --- TMA descriptors ---
        # k: A-operand of MMA1 (K-major)
        s_k_one = cute.select(s_k_staged, mode=[0, 1, 2])
        atom_k, desc_k = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld,
            g_k,
            s_k_one,
            self.dv_tile,
            mma_dv,
            clust_lay.shape,
        )
        # q^T: A-operand of MMA2 (MN-major)
        s_q_one = cute.select(s_q_staged, mode=[0, 1, 2])
        atom_q, desc_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld,
            g_qt,
            s_q_one,
            self.qdo_tile,
            mma_qdo,
            clust_lay.shape,
        )
        # do: epilogue-style TMA load into sDoEpi (masked + R2S → sDo for MMA2)
        s_doepi_one = cute.select(s_doepi_staged, mode=[0, 1])
        atom_doepi, desc_doepi = cpasync.make_tiled_tma_atom(
            tma_ld,
            g_do_vt,
            s_doepi_one,
            (self.BV, self.BT),
        )
        # w^T: A-operand of MMA3 (MN-major)
        s_w_one = cute.select(s_w_staged, mode=[0, 1, 2])
        atom_w, desc_w = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld,
            g_wt,
            s_w_one,
            self.wdv_tile,
            mma_wdv,
            clust_lay.shape,
        )

        # dv_in: epilogue-style TMA load
        s_dvin_one = cute.select(s_dvin_staged, mode=[0, 1])
        atom_dvin, desc_dvin = cpasync.make_tiled_tma_atom(
            tma_ld,
            g_dv_in_t,
            s_dvin_one,
            (self.BV, self.BT),
        )

        # dh_out store TMA
        s_dh_epi_one = cute.select(s_dh_epi_staged, mode=[0, 1])
        atom_dhst, desc_dhst = cpasync.make_tiled_tma_atom(
            tma_st,
            g_dh_out_t,
            s_dh_epi_one,
            (self.BV, self.BK),
        )

        # dv2 store TMA
        s_dv2_epi_one = cute.select(s_dv2_epi_staged, mode=[0, 1])
        atom_dv2st, desc_dv2st = cpasync.make_tiled_tma_atom(
            tma_st,
            g_dv2_t,
            s_dv2_epi_one,
            (self.BV, self.BT),
        )

        # gk TMA: BK×1 fp32 tile (last timestep per chunk)
        s_gk_2d = cute.make_layout((self.BK, 1))
        atom_gk, desc_gk = cpasync.make_tiled_tma_atom(
            tma_ld,
            g_gk_k,
            s_gk_2d,
            (self.BK, 1),
        )

        # CopyUniversal for varlen partial-chunk dv2 store
        cp_bits = 128
        cp_elems = cp_bits // self.io_type.width
        copy_atom_uni = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_type,
            num_bits_per_copy=cp_bits,
        )
        vn_thr_d0 = self.BV // cp_elems
        vn_thr_d1 = self.WARP_SZ // vn_thr_d0
        vn_thr_lay = cute.make_ordered_layout((vn_thr_d0, vn_thr_d1), order=(0, 1))
        vn_val_lay = cute.make_layout((cp_elems, 1))
        copy_dv2_tiled = cute.make_tiled_copy_tv(copy_atom_uni, vn_thr_lay, vn_val_lay)

        cu_lens = cute.make_tensor(cup, cute.make_layout((B + 1,)))
        ch_offs = cute.make_tensor(cop, cute.make_layout((B + 1,)))

        # TMA byte counts
        self.k_bytes = cute.size_in_bytes(self.io_type, s_k_one)
        self.q_bytes = cute.size_in_bytes(self.io_type, s_q_one)
        self.do_bytes = cute.size_in_bytes(self.io_type, s_doepi_one)
        self.w_bytes = cute.size_in_bytes(self.io_type, s_w_one)
        self.dvin_bytes = cute.size_in_bytes(self.io_type, s_dvin_one)
        self.gk_bytes = self.BK * 4  # 512 bytes

        # --- SharedStorage struct ---
        @cute.struct
        class Shared:
            # Pipeline barriers (each needs depth * 2 Int64s)
            bar_k: cute.struct.MemRange[Int64, self.k_depth * 2]
            bar_q: cute.struct.MemRange[Int64, self.q_depth * 2]
            bar_doepi: cute.struct.MemRange[Int64, self.do_depth * 2]
            bar_w: cute.struct.MemRange[Int64, self.w_depth * 2]
            bar_dvin: cute.struct.MemRange[Int64, self.dv_in_depth * 2]
            bar_gk: cute.struct.MemRange[Int64, self.gk_depth * 2]
            bar_gk_rdy: cute.struct.MemRange[Int64, self.gk_depth * 2]
            bar_dhb: cute.struct.MemRange[Int64, 1 * 2]  # dh B-operand CUDA→MMA
            bar_dv: cute.struct.MemRange[Int64, self.dv_acc_depth * 2]  # MMA1 done
            bar_qdo: cute.struct.MemRange[Int64, self.qdo_acc_depth * 2]  # MMA2 done
            bar_dob: cute.struct.MemRange[Int64, 1 * 2]  # do B-operand CUDA→MMA
            bar_dv2b: cute.struct.MemRange[Int64, 1 * 2]  # dv2 B-operand CUDA→MMA
            bar_wdv: cute.struct.MemRange[Int64, self.wdv_acc_depth * 2]  # MMA3 done
            bar_dh_epi: cute.struct.MemRange[Int64, self.dh_epi_depth * 2]
            bar_dv2_epi: cute.struct.MemRange[Int64, self.dv2_epi_depth * 2]
            tmem_buf: Int32
            # SMEM data buffers
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_k_staged)], self.align
            ]
            sDhb: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_dhb_staged)],
                self.align,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_q_staged)], self.align
            ]
            sDo: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_do_staged)], self.align
            ]
            sDoEpi: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_doepi_staged)],
                self.align,
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_w_staged)], self.align
            ]
            sDv2b: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_dv2b_staged)],
                self.align,
            ]
            sDvIn: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_dvin_staged)],
                self.align,
            ]
            sDhEpi: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_dh_epi_staged)],
                self.align,
            ]
            sDv2Epi: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_dv2_epi_staged)],
                self.align,
            ]
            sGK: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK * self.gk_depth], 128
            ]
            sGK_exp: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, self.BK * self.gk_depth], 128
            ]

        self.shared_type = Shared
        self.grid = self._launch_grid(B, H, V)

        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            mma_dv,
            mma_qdo,
            mma_wdv,
            atom_k,
            desc_k,
            atom_q,
            desc_q,
            atom_doepi,
            desc_doepi,
            atom_w,
            desc_w,
            atom_dvin,
            desc_dvin,
            atom_dhst,
            desc_dhst,
            atom_dv2st,
            desc_dv2st,
            atom_gk,
            desc_gk,
            copy_dv2_tiled,
            g_dht,
            g_dh0_t,
            g_dv2,
            s_k_staged,
            s_dhb_staged,
            s_dhb_store_staged,
            s_q_staged,
            s_do_staged,
            s_do_store_staged,
            s_doepi_staged,
            s_w_staged,
            s_dv2b_staged,
            s_dv2b_store_staged,
            s_dvin_staged,
            s_dh_epi_staged,
            s_dh_r2s_staged,
            s_dv2_epi_staged,
            s_dv2_r2s_staged,
            cu_lens,
            ch_offs,
            problem_shape,
            scale,
            use_gk,
            use_dht,
            use_dh0,
        ).launch(
            grid=self.grid,
            block=[self.CTA_THREADS, 1, 1],
            cluster=self.cluster,
            stream=stream,
            min_blocks_per_mp=self.min_occ,
        )

    # ------------------------------------------------------------------
    # Device kernel: warp-specialized backward recurrence
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mma_dv: cute.TiledMma,
        mma_qdo: cute.TiledMma,
        mma_wdv: cute.TiledMma,
        atom_k: cute.CopyAtom,
        desc_k: cute.Tensor,
        atom_q: cute.CopyAtom,
        desc_q: cute.Tensor,
        atom_doepi: cute.CopyAtom,
        desc_doepi: cute.Tensor,
        atom_w: cute.CopyAtom,
        desc_w: cute.Tensor,
        atom_dvin: cute.CopyAtom,
        desc_dvin: cute.Tensor,
        atom_dhst: cute.CopyAtom,
        desc_dhst: cute.Tensor,
        atom_dv2st: cute.CopyAtom,
        desc_dv2st: cute.Tensor,
        atom_gk: cute.CopyAtom,
        desc_gk: cute.Tensor,
        copy_dv2_tiled: cute.TiledCopy,
        g_dht: cute.Tensor,
        g_dh0_t: cute.Tensor,
        g_dv2: cute.Tensor,
        s_k_staged: cute.ComposedLayout,
        s_dhb_staged: cute.ComposedLayout,
        s_dhb_store_staged: cute.ComposedLayout,
        s_q_staged: cute.ComposedLayout,
        s_do_staged: cute.ComposedLayout,
        s_do_store_staged: cute.ComposedLayout,
        s_doepi_staged: cute.ComposedLayout,
        s_w_staged: cute.ComposedLayout,
        s_dv2b_staged: cute.ComposedLayout,
        s_dv2b_store_staged: cute.ComposedLayout,
        s_dvin_staged: cute.ComposedLayout,
        s_dh_epi_staged: cute.ComposedLayout,
        s_dh_r2s_staged: cute.ComposedLayout,
        s_dv2_epi_staged: cute.ComposedLayout,
        s_dv2_r2s_staged: cute.ComposedLayout,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        problem_shape: tuple[Int32, Int32, Int32, Int32, Int32],
        scale: Float32,
        use_gk: Int32,
        use_dht: Int32,
        use_dh0: Int32,
    ):
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()

        # Prefetch TMA descriptors (load warp)
        if warp_id == self.LOAD_WARP_ID:
            cpasync.prefetch_descriptor(atom_k)
            cpasync.prefetch_descriptor(atom_q)
            cpasync.prefetch_descriptor(atom_doepi)
            cpasync.prefetch_descriptor(atom_w)
            cpasync.prefetch_descriptor(atom_dvin)
            cpasync.prefetch_descriptor(atom_gk)

        # SMEM allocation
        sa = utils.SmemAllocator()
        sm = sa.allocate(self.shared_type)

        gk_buf = sm.sGK.get_tensor(cute.make_layout((self.BK, self.gk_depth)))
        gk_exp_buf = sm.sGK_exp.get_tensor(cute.make_layout((self.BK, self.gk_depth)))
        gk_3d = sm.sGK.get_tensor(
            cute.make_layout(
                (self.BK, 1, self.gk_depth),
                stride=(1, self.BK, self.BK),
            )
        )

        # #
        # Pipeline creation
        #

        # k → MMA1 (TmaUmma, deep pipeline)
        pk_P, pk_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.k_bytes,
            barrier_storage=sm.bar_k.data_ptr(),
        ).make_participants()

        # dh B-operand → MMA1 (AsyncUmma, 1-stage: CUDA R2S → MMA reads)
        pdhb_P, pdhb_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=sm.bar_dhb.data_ptr(),
        ).make_participants()

        # MMA1 done → CUDA (UmmaAsync)
        pdv_P, pdv_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.dv_acc_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            barrier_storage=sm.bar_dv.data_ptr(),
        ).make_participants()

        # q → MMA2 (TmaUmma)
        pq_P, pq_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.q_bytes,
            barrier_storage=sm.bar_q.data_ptr(),
        ).make_participants()

        # do → CUDA (TmaAsync): TMA loads do into sDoEpi for the COMPUTE warp
        pdoepi_P, pdoepi_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.do_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.CUDA_WARP_IDS)
            ),
            tx_count=self.do_bytes,
            barrier_storage=sm.bar_doepi.data_ptr(),
        ).make_participants()

        # do B-operand → MMA2 (AsyncUmma, 1-stage: CUDA R2S → MMA reads)
        pdob_P, pdob_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=sm.bar_dob.data_ptr(),
        ).make_participants()

        # MMA2 done → CUDA (UmmaAsync)
        pqdo_P, pqdo_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qdo_acc_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            barrier_storage=sm.bar_qdo.data_ptr(),
        ).make_participants()

        # w → MMA3 (TmaUmma)
        pw_P, pw_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.w_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.w_bytes,
            barrier_storage=sm.bar_w.data_ptr(),
        ).make_participants()

        # dv2 B-operand → MMA3 (AsyncUmma, 1-stage: CUDA R2S → MMA reads)
        pdv2b_P, pdv2b_C = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=sm.bar_dv2b.data_ptr(),
        ).make_participants()

        # MMA3 done → CUDA (UmmaAsync)
        pwdv_P, pwdv_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.wdv_acc_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            barrier_storage=sm.bar_wdv.data_ptr(),
        ).make_participants()

        # dv_in load → CUDA (TmaAsync)
        # NOTE: PipelineTmaAsync uses warp-elected barrier arrives in release(),
        # so consumer_group = num_warps (4), NOT num_threads (128).
        pdvin_P, pdvin_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.dv_in_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.CUDA_WARP_IDS)
            ),
            tx_count=self.dvin_bytes,
            barrier_storage=sm.bar_dvin.data_ptr(),
        ).make_participants()

        # dh epi → Store warp (Async, 2-stage)
        pdh_epi_P, pdh_epi_C = pipeline.PipelineAsync.create(
            num_stages=self.dh_epi_depth,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.WARP_SZ),
            barrier_storage=sm.bar_dh_epi.data_ptr(),
        ).make_participants()

        # dv2 epi → Store warp (Async, 2-stage)
        pdv2_epi_P, pdv2_epi_C = pipeline.PipelineAsync.create(
            num_stages=self.dv2_epi_depth,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.WARP_SZ),
            barrier_storage=sm.bar_dv2_epi.data_ptr(),
        ).make_participants()

        # gk TMA → GK warp (TmaAsync, 2-stage)
        pgk_P, pgk_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.gk_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.gk_bytes,
            barrier_storage=sm.bar_gk.data_ptr(),
        ).make_participants()

        # gk ready → CUDA (Async, 2-stage)
        pgk_rdy_P, pgk_rdy_C = pipeline.PipelineAsync.create(
            num_stages=self.gk_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.WARP_SZ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.WARP_SZ * len(self.CUDA_WARP_IDS)
            ),
            barrier_storage=sm.bar_gk_rdy.data_ptr(),
        ).make_participants()

        # #
        # TMEM allocation
        #
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.CTA_THREADS)
        tmem = utils.TmemAllocator(
            sm.tmem_buf,
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=self.LOAD_WARP_ID,
        )
        tmem.allocate(self.tm_tot)
        tmem.wait_for_alloc()
        tp = tmem.retrieve_ptr(self.acc_type)

        # #
        # SMEM view tensors
        #
        sK = sm.sK.get_tensor(s_k_staged.outer, swizzle=s_k_staged.inner)
        sDhb = sm.sDhb.get_tensor(s_dhb_staged.outer, swizzle=s_dhb_staged.inner)
        sDhb_store = sm.sDhb.get_tensor(s_dhb_store_staged.outer, swizzle=s_dhb_store_staged.inner)
        sQ = sm.sQ.get_tensor(s_q_staged.outer, swizzle=s_q_staged.inner)
        sDo = sm.sDo.get_tensor(s_do_staged.outer, swizzle=s_do_staged.inner)
        sDo_store = sm.sDo.get_tensor(s_do_store_staged.outer, swizzle=s_do_store_staged.inner)
        sDoEpi = sm.sDoEpi.get_tensor(s_doepi_staged.outer, swizzle=s_doepi_staged.inner)
        sW = sm.sW.get_tensor(s_w_staged.outer, swizzle=s_w_staged.inner)
        sDv2b = sm.sDv2b.get_tensor(s_dv2b_staged.outer, swizzle=s_dv2b_staged.inner)
        sDv2b_store = sm.sDv2b.get_tensor(
            s_dv2b_store_staged.outer, swizzle=s_dv2b_store_staged.inner
        )
        sDvIn = sm.sDvIn.get_tensor(s_dvin_staged.outer, swizzle=s_dvin_staged.inner)
        sDhEpi = sm.sDhEpi.get_tensor(s_dh_epi_staged.outer, swizzle=s_dh_epi_staged.inner)
        sDhEpi_store = sm.sDhEpi.get_tensor(s_dh_r2s_staged.outer, swizzle=s_dh_r2s_staged.inner)
        sDv2Epi = sm.sDv2Epi.get_tensor(s_dv2_epi_staged.outer, swizzle=s_dv2_epi_staged.inner)
        sDv2Epi_store = sm.sDv2Epi.get_tensor(
            s_dv2_r2s_staged.outer, swizzle=s_dv2_r2s_staged.inner
        )

        # #
        # MMA fragments — SS-mode: both A and B from SMEM, accumulators in TMEM
        #

        # MMA1: k (A) × dh (B) → dv_acc
        t_k_a = mma_dv.make_fragment_A(sK)
        t_dhb_b = mma_dv.make_fragment_B(sDhb)
        dv_sh = mma_dv.partition_shape_C(self.dv_tile[:2])
        dv_fk = mma_dv.make_fragment_C(cute.append(dv_sh, self.dv_acc_depth))
        t_dv_acc = cute.make_tensor(tp + self.tm_dv, dv_fk.layout)

        # MMA2: q^T (A) × do (B) → qdo_acc
        t_q_a = mma_qdo.make_fragment_A(sQ)
        t_do_b = mma_qdo.make_fragment_B(sDo)
        qdo_sh = mma_qdo.partition_shape_C(self.qdo_tile[:2])
        qdo_fk = mma_qdo.make_fragment_C(cute.append(qdo_sh, self.qdo_acc_depth))
        t_qdo_acc = cute.make_tensor(tp + self.tm_qdo, qdo_fk.layout)

        # MMA3: w (A) × dv2 (B) → wdv_acc
        t_w_a = mma_wdv.make_fragment_A(sW)
        t_dv2b_b = mma_wdv.make_fragment_B(sDv2b)
        wdv_sh = mma_wdv.partition_shape_C(self.wdv_tile[:2])
        wdv_fk = mma_wdv.make_fragment_C(cute.append(wdv_sh, self.wdv_acc_depth))
        t_wdv_acc = cute.make_tensor(tp + self.tm_wdv, wdv_fk.layout)

        # #
        # Block indices
        #
        B, T, H, _K, V = problem_shape
        BT = self.BT

        # Persistent 1D grid for both varlen and non-varlen (CUDA graph compatible)
        bx = cute.arch.block_idx()[0]
        gdx = cute.arch.grid_dim()[0]
        n_vtiles = (V + self.BV - 1) // self.BV

        w_tiles = n_vtiles * H * B

        n_iters = (w_tiles - bx + gdx - 1) // gdx
        w_idx = Int32(0)
        v_tile = Int32(0)
        h_idx = Int32(0)
        seq_idx = Int32(0)
        bos = Int32(0)
        seq_len = Int32(0)
        NT = Int32(0)
        db = Int32(0)
        co = Int32(0)

        # ///////////////////////////////////////////////////////////////////////////////
        #  CUDA CORE (warps 0-3) — dh state owner
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_id in self.CUDA_WARP_IDS:
            cute.arch.setmaxregister_increase(self.cuda_regs)
            local_tid = tid % (self.WARP_SZ * len(self.CUDA_WARP_IDS))

            # --- T2R setup: read MMA1 (dv) accumulator (BT,BV fp32) from TMEM ---
            t2r_dv_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BV // 8), tcgen05.Pack.NONE),
                self.acc_type,
            )
            dv_flat = t_dv_acc[((None, None), 0, 0, None)]
            fake_s_dv = cute.make_tensor(
                cute.make_ptr(self.io_type, 0, cute.AddressSpace.smem),
                cute.dice(self.dv_tile, (1, 1, None)),
            )
            tc_t2r_dv = tcgen05.make_tmem_copy(t2r_dv_atom, dv_flat[(None, None, 0)])
            sl_dv = tc_t2r_dv.get_slice(local_tid)
            p_t_dv = sl_dv.partition_S(dv_flat)
            p_s_dv = sl_dv.partition_D(fake_s_dv)

            # --- T2R setup: read MMA2 (qdo) accumulator (BK,BV fp32) from TMEM ---
            t2r_qdo_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BV // 8), tcgen05.Pack.NONE),
                self.acc_type,
            )
            qdo_flat = t_qdo_acc[((None, None), 0, 0, None)]
            fake_s_qdo = cute.make_tensor(
                cute.make_ptr(self.io_type, 0, cute.AddressSpace.smem),
                cute.dice(self.qdo_tile, (1, 1, None)),
            )
            tc_t2r_qdo = tcgen05.make_tmem_copy(t2r_qdo_atom, qdo_flat[(None, None, 0)])
            sl_qdo = tc_t2r_qdo.get_slice(local_tid)
            p_t_qdo = sl_qdo.partition_S(qdo_flat)
            p_s_qdo = sl_qdo.partition_D(fake_s_qdo)

            # --- T2R setup: read MMA3 (wdv) accumulator (BK,BV fp32) from TMEM ---
            t2r_wdv_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BV // 8), tcgen05.Pack.NONE),
                self.acc_type,
            )
            wdv_flat = t_wdv_acc[((None, None), 0, 0, None)]
            fake_s_wdv = cute.make_tensor(
                cute.make_ptr(self.io_type, 0, cute.AddressSpace.smem),
                cute.dice(self.wdv_tile, (1, 1, None)),
            )
            tc_t2r_wdv = tcgen05.make_tmem_copy(t2r_wdv_atom, wdv_flat[(None, None, 0)])
            sl_wdv = tc_t2r_wdv.get_slice(local_tid)
            p_t_wdv = sl_wdv.partition_S(wdv_flat)
            p_s_wdv = sl_wdv.partition_D(fake_s_wdv)

            # dh state lives in registers partitioned like qdo/wdv (BK, BV)
            st = cute.make_rmem_tensor(p_s_qdo.shape, self.acc_type)

            # Identity tensors for coordinate mapping
            # dv tile: (BT, BV) coords
            id_tv = cute.make_identity_tensor(cute.dice(self.dv_tile, (1, 1, None)))
            coords_tv = sl_dv.partition_D(id_tv)
            # qdo/wdv tile: (BK, BV) coords — dh state uses this
            id_kv = cute.make_identity_tensor(cute.dice(self.qdo_tile, (1, 1, None)))
            coords_kv = sl_qdo.partition_D(id_kv)

            # R2S dh → sDhb (COL_MAJOR, partition matches T2R qdo)
            r2s_atom_dhb = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR,
                self.io_type,
                self.acc_type,
                tc_t2r_qdo,
            )
            tc_r2s_dhb = cute.make_tiled_copy_D(r2s_atom_dhb, tc_t2r_qdo)
            thr_r2s_dhb = tc_r2s_dhb.get_slice(local_tid)

            # R2S dh → sDhEpi (ROW_MAJOR, transposed for TMA store)
            r2s_atom_dh_epi = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_type,
                self.acc_type,
                tc_t2r_qdo,
            )
            tc_r2s_dh_epi = cute.make_tiled_copy_D(r2s_atom_dh_epi, tc_t2r_qdo)
            thr_r2s_dh_epi = tc_r2s_dh_epi.get_slice(local_tid)

            # R2S dv2 → sDv2b (COL_MAJOR, partition matches T2R dv)
            r2s_atom_dv2b = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.COL_MAJOR,
                self.io_type,
                self.acc_type,
                tc_t2r_dv,
            )
            tc_r2s_dv2b = cute.make_tiled_copy_D(r2s_atom_dv2b, tc_t2r_dv)
            thr_r2s_dv2b = tc_r2s_dv2b.get_slice(local_tid)

            # R2S dv2 → sDv2Epi (ROW_MAJOR, transposed for TMA store)
            r2s_atom_dv2_epi = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR,
                self.io_type,
                self.acc_type,
                tc_t2r_dv,
            )
            tc_r2s_dv2_epi = cute.make_tiled_copy_D(r2s_atom_dv2_epi, tc_t2r_dv)
            thr_r2s_dv2_epi = tc_r2s_dv2_epi.get_slice(local_tid)

            # ========== Tile loop ==========
            tile_idx = Int32(0)
            has_work = tile_idx < n_iters

            while has_work:
                # Decode work-unit → tile coordinates (persistent scheduling)
                w_idx = bx + tile_idx * gdx
                v_tile = w_idx % n_vtiles
                tmp = w_idx // n_vtiles
                h_idx = tmp % H
                seq_idx = tmp // H
                if cutlass.const_expr(self.varlen):
                    bos = cu_seqlens[seq_idx]
                    seq_len = cu_seqlens[seq_idx + 1] - bos
                else:
                    bos = Int32(0)
                    seq_len = T
                NT = (seq_len + BT - 1) // BT
                db = Int32(0) if cutlass.const_expr(self.varlen) else seq_idx

                # Initialize dh from dht or zeros
                if use_dht:
                    gDht_slice = g_dht[None, None, (h_idx, seq_idx)]
                    for ei in cutlass.range(cute.size(st), unroll_full=True):
                        kc, vc = coords_kv[ei]
                        st[ei] = gDht_slice[(kc, vc % self.BV + v_tile * self.BV)]
                else:
                    for ei in cutlass.range(cute.size(st), unroll_full=True):
                        st[ei] = Float32(0.0)

                # ========== Chunk loop: reverse order (NT-1 down to 0) ==========
                for ct in cutlass.range(0, NT, unroll=0):
                    rev_ct = NT - 1 - ct  # reverse index

                    # ---- Phase 1: R2S dh→sDhb + sDhEpi (snapshot dh state) ----
                    dhb_h = pdhb_P.acquire_and_advance()
                    dh_epi_h = pdh_epi_P.acquire_and_advance()
                    st_bf16 = cute.make_rmem_tensor(st.shape, self.io_type)
                    st_bf16.store(st.load().to(self.io_type))

                    # R2S dh → sDhb (B-operand for MMA1)
                    r2s_src_dhb = tc_r2s_dhb.retile(st_bf16)
                    dst_dhb = thr_r2s_dhb.partition_D(sDhb_store[(None, None, 0)])
                    cute.copy(tc_r2s_dhb, r2s_src_dhb, dst_dhb)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    dhb_h.commit()

                    # R2S dh → sDhEpi (for Store warp TMA S2G)
                    r2s_src_dh = tc_r2s_dh_epi.retile(st_bf16)
                    dst_dh = thr_r2s_dh_epi.partition_D(sDhEpi_store[(None, None, dh_epi_h.index)])
                    cute.copy(tc_r2s_dh_epi, r2s_src_dh, dst_dh)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    dh_epi_h.commit()

                    # ---- Produce masked do B-operand for MMA2 ----
                    # Mirrors the dv2 mask: read the TMA-loaded do tile, zero the
                    # partial chunk's over-read token rows in registers (reusing
                    # valid/coords_tv from the dv_reg mask), then R2S into sDo.
                    # do is MN-major, so we use the ROW_MAJOR R2S atom
                    # tc_r2s_dv2_epi (the same one used for the dv2 epi store), not
                    # the COL_MAJOR tc_r2s_dv2b used for the K-major dv2 B-operand.
                    dob_h = pdob_P.acquire_and_advance()
                    doepi_h = pdoepi_C.wait_and_advance()
                    do_reg = cute.make_rmem_tensor(p_s_dv.shape, self.acc_type)
                    for ei in cutlass.range_constexpr(cute.size(coords_tv)):
                        tc, vc = coords_tv[ei]
                        do_reg[ei] = sDoEpi[(vc, tc, doepi_h.index)].to(self.acc_type)
                    doepi_h.release()
                    if cutlass.const_expr(self.varlen):
                        valid = seq_len - rev_ct * self.BT
                        if valid < self.BT:
                            for ei in cutlass.range_constexpr(cute.size(coords_tv)):
                                tc, vc = coords_tv[ei]
                                if tc >= valid:
                                    do_reg[ei] = Float32(0.0)
                    do_bf16 = cute.make_rmem_tensor(do_reg.shape, self.io_type)
                    do_bf16.store(do_reg.load().to(self.io_type))
                    r2s_src_dob = tc_r2s_dv2_epi.retile(do_bf16)
                    dst_dob = thr_r2s_dv2_epi.partition_D(sDo_store[(None, None, 0)])
                    cute.copy(tc_r2s_dv2_epi, r2s_src_dob, dst_dob)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    dob_h.commit()

                    # ---- Phase 2: T2R dv result from MMA1, gate, add dv_in ----
                    dvh = pdv_C.wait_and_advance()
                    dv_reg = cute.make_rmem_tensor(p_s_dv.shape, self.acc_type)
                    cute.copy(tc_t2r_dv, p_t_dv[(None, None, None, dvh.index)], dv_reg)
                    cute.arch.fence_view_async_tmem_load()
                    dvh.release()

                    # Load dv_in from SMEM (TMA loaded by load warp)
                    dvin_h = pdvin_C.wait_and_advance()
                    dvin_reg = cute.make_rmem_tensor(p_s_dv.shape, self.acc_type)
                    for ei in cutlass.range_constexpr(cute.size(coords_tv)):
                        tc, vc = coords_tv[ei]
                        dvin_reg[ei] = sDvIn[(vc, tc, dvin_h.index)].to(self.acc_type)
                    dvin_h.release()

                    # Combine: dv = dv_mma + dv_in
                    # Note: per-timestep g gating (exp2(g_last-g[t])) is NOT applied here;
                    # only gk-based gating is supported in v1 (matching forward cuteDSL).
                    for ei in cutlass.range_constexpr(cute.size(dv_reg)):
                        dv_reg[ei] = dv_reg[ei] + dvin_reg[ei]

                    # Mask partial chunk for varlen
                    if cutlass.const_expr(self.varlen):
                        valid = seq_len - rev_ct * self.BT
                        if valid < self.BT:
                            for ei in cutlass.range_constexpr(cute.size(coords_tv)):
                                tc, vc = coords_tv[ei]
                                if tc >= valid:
                                    dv_reg[ei] = Float32(0.0)

                    # ---- Phase 3: R2S dv2→sDv2b + sDv2Epi ----
                    dv2b_h = pdv2b_P.acquire_and_advance()
                    dv2_epi_h = pdv2_epi_P.acquire_and_advance()
                    dv_bf16 = cute.make_rmem_tensor(dv_reg.shape, self.io_type)
                    dv_bf16.store(dv_reg.load().to(self.io_type))

                    # R2S dv2 → sDv2b (B-operand for MMA3)
                    r2s_src_dv2b = tc_r2s_dv2b.retile(dv_bf16)
                    dst_dv2b = thr_r2s_dv2b.partition_D(sDv2b_store[(None, None, 0)])
                    cute.copy(tc_r2s_dv2b, r2s_src_dv2b, dst_dv2b)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    dv2b_h.commit()

                    # R2S dv2 → sDv2Epi (for Store warp TMA S2G)
                    r2s_src_dv2 = tc_r2s_dv2_epi.retile(dv_bf16)
                    dst_dv2 = thr_r2s_dv2_epi.partition_D(
                        sDv2Epi_store[(None, None, dv2_epi_h.index)]
                    )
                    cute.copy(tc_r2s_dv2_epi, r2s_src_dv2, dst_dv2)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    dv2_epi_h.commit()

                    # ---- Phase 4: Decay dh and apply qdo, wdv updates ----
                    # Decay: dh *= exp2(gk_last[k])
                    if use_gk:
                        gk_rdy2 = pgk_rdy_C.wait_and_advance()
                        for ei in cutlass.range(cute.size(st), unroll_full=True):
                            kc, vc = coords_kv[ei]
                            st[ei] = st[ei] * gk_exp_buf[(kc, gk_rdy2.index)]
                        gk_rdy2.release()

                    # T2R qdo result from MMA2
                    qdoh = pqdo_C.wait_and_advance()
                    qdo_reg = cute.make_rmem_tensor(p_s_qdo.shape, self.acc_type)
                    cute.copy(tc_t2r_qdo, p_t_qdo[(None, None, None, qdoh.index)], qdo_reg)
                    cute.arch.fence_view_async_tmem_load()
                    qdoh.release()

                    # T2R wdv result from MMA3
                    wdvh = pwdv_C.wait_and_advance()
                    wdv_reg = cute.make_rmem_tensor(p_s_wdv.shape, self.acc_type)
                    cute.copy(tc_t2r_wdv, p_t_wdv[(None, None, None, wdvh.index)], wdv_reg)
                    cute.arch.fence_view_async_tmem_load()
                    wdvh.release()

                    # dh += qdo * scale - wdv
                    for ei in cutlass.range_constexpr(cute.size(st)):
                        st[ei] = st[ei] + qdo_reg[ei] * scale - wdv_reg[ei]

                # ========== Store dh0 (initial state gradient) ==========
                if use_dh0:
                    gDh0 = g_dh0_t[None, None, (h_idx, seq_idx)]
                    for ei in cutlass.range(cute.size(st), unroll_full=True):
                        kc, vc = coords_kv[ei]
                        gDh0[vc % self.BV + v_tile * self.BV, kc] = st[ei]

                tile_idx = tile_idx + 1
                has_work = tile_idx < n_iters

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA (warp 6) — 3 sequential GEMMs per chunk
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_id == self.MMA_WARP_ID:
            cute.arch.setmaxregister_decrease(self.aux_regs)
            mma_tid = tid - self.MMA_WARP_ID * self.WARP_SZ

            tile_idx = Int32(0)
            has_work = tile_idx < n_iters

            while has_work:
                w_idx = bx + tile_idx * gdx
                bi_m = (w_idx // ((V + self.BV - 1) // self.BV)) // H
                tail_valid_rows = Int32(0)
                if cutlass.const_expr(self.varlen):
                    seq_len = cu_seqlens[bi_m + 1] - cu_seqlens[bi_m]
                    NT = (seq_len + BT - 1) // BT
                    # Chunks iterate in reverse, so ct == 0 is the trailing
                    # (possibly partial) chunk of this sequence.
                    tail_valid_rows = seq_len % self.BT
                else:
                    NT = (T + BT - 1) // BT

                for ct in cutlass.range(0, NT, unroll=0):
                    # --- MMA1: k(SMEM A) × dh(SMEM B) → dv_acc(TMEM) ---
                    dhbh = pdhb_C.wait_and_advance()
                    kh = pk_C.wait_and_advance()
                    dvd = pdv_P.acquire_and_advance()
                    for kp in cutlass.range(cute.size(t_k_a, mode=[2]), unroll_full=True):
                        mma_dv.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            mma_dv,
                            t_dv_acc[None, None, None, dvd.index],
                            t_k_a[None, None, kp, kh.index],
                            t_dhb_b[None, None, kp, dhbh.index],
                            t_dv_acc[None, None, None, dvd.index],
                        )
                    dvd.commit()
                    kh.release()
                    dhbh.release()

                    # --- MMA2: q^T(SMEM A) × do(SMEM B) → qdo_acc(TMEM) ---
                    # do is the COMPUTE-warp-produced, partial-chunk-masked
                    # B-operand (1-stage R2S), mirroring dv2/MMA3, so only the
                    # A-operand tail neutralization below runs on the MMA warp.
                    qh = pq_C.wait_and_advance()
                    if cutlass.const_expr(self.varlen):  # noqa: SIM102
                        if ct == 0 and tail_valid_rows != 0:
                            self._neutralize_reduction_rows(sQ, qh.index, tail_valid_rows, mma_tid)
                    doh = pdob_C.wait_and_advance()
                    qdod = pqdo_P.acquire_and_advance()
                    for kp in cutlass.range(cute.size(t_q_a, mode=[2]), unroll_full=True):
                        mma_qdo.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            mma_qdo,
                            t_qdo_acc[None, None, None, qdod.index],
                            t_q_a[None, None, kp, qh.index],
                            t_do_b[None, None, kp, doh.index],
                            t_qdo_acc[None, None, None, qdod.index],
                        )
                    qdod.commit()
                    qh.release()
                    doh.release()

                    # --- MMA3: w(SMEM A) × dv2(SMEM B) → wdv_acc(TMEM) ---
                    dv2bh = pdv2b_C.wait_and_advance()
                    wh = pw_C.wait_and_advance()
                    if cutlass.const_expr(self.varlen):  # noqa: SIM102
                        if ct == 0 and tail_valid_rows != 0:
                            self._neutralize_reduction_rows(sW, wh.index, tail_valid_rows, mma_tid)
                    wdvd = pwdv_P.acquire_and_advance()
                    for kp in cutlass.range(cute.size(t_w_a, mode=[2]), unroll_full=True):
                        mma_wdv.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            mma_wdv,
                            t_wdv_acc[None, None, None, wdvd.index],
                            t_w_a[None, None, kp, wh.index],
                            t_dv2b_b[None, None, kp, dv2bh.index],
                            t_wdv_acc[None, None, None, wdvd.index],
                        )
                    wdvd.commit()
                    wh.release()
                    dv2bh.release()

                tile_idx = tile_idx + 1
                has_work = tile_idx < n_iters

        # ///////////////////////////////////////////////////////////////////////////////
        #  STORE (warp 7) — TMA S2G for dh snapshot and dv2
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_id == self.STORE_WARP_ID:
            cute.arch.setmaxregister_decrease(self.aux_regs)
            cpasync.prefetch_descriptor(atom_dhst)
            cpasync.prefetch_descriptor(atom_dv2st)
            st_ltid = tid - self.STORE_WARP_ID * self.WARP_SZ

            tile_idx = Int32(0)
            has_work = tile_idx < n_iters

            while has_work:
                w_idx = bx + tile_idx * gdx
                v_tile = w_idx % ((V + self.BV - 1) // self.BV)
                tmp = w_idx // ((V + self.BV - 1) // self.BV)
                h_idx = tmp % H
                seq_idx = tmp // H
                if cutlass.const_expr(self.varlen):
                    bos = cu_seqlens[seq_idx]
                    seq_len = cu_seqlens[seq_idx + 1] - bos
                    NT = (seq_len + BT - 1) // BT
                    db = Int32(0)
                    co = chunk_offsets[seq_idx]
                else:
                    bos = Int32(0)
                    seq_len = T
                    NT = (T + BT - 1) // BT
                    db = seq_idx
                    co = Int32(0)

                # Domain-offset store TMA descriptors
                if cutlass.const_expr(self.varlen):
                    ddh = cute.domain_offset((0, 0, (co, 0, 0)), desc_dhst)
                    ddv2 = cute.domain_offset((0, bos, (0, 0)), desc_dv2st)
                else:
                    ddh = desc_dhst
                    ddv2 = desc_dv2st

                gDH_s = ddh[None, None, (None, h_idx, db)]
                a_dhst, sDHst, gDHst = self._part_epi(atom_dhst, gDH_s, (self.BV, self.BK), sDhEpi)

                gDV2_s = ddv2[None, None, (h_idx, db)]
                a_dv2st, sDV2st, gDV2st = self._part_epi(
                    atom_dv2st, gDV2_s, (self.BV, self.BT), sDv2Epi
                )

                for ct in cutlass.range(0, NT, unroll=0):
                    rev_ct = NT - 1 - ct  # reverse index matching CUDA warp

                    # Store dh snapshot via TMA S2G
                    dhh = pdh_epi_C.wait_and_advance()
                    cute.copy(a_dhst, sDHst[None, dhh.index], gDHst[(None, v_tile, 0, rev_ct)])
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    dhh.release()

                    # Store dv2 via TMA S2G (with varlen partial-chunk fallback)
                    dv2h = pdv2_epi_C.wait_and_advance()
                    if cutlass.const_expr(self.varlen):
                        rem = seq_len - rev_ct * self.BT
                        if rem >= self.BT:
                            cute.copy(
                                a_dv2st,
                                sDV2st[None, dv2h.index],
                                gDV2st[(None, v_tile, rev_ct)],
                            )
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        else:
                            # Partial chunk: element-wise store with bounds check
                            sVn_slice = sDv2Epi[None, None, dv2h.index]
                            thr_cp = copy_dv2_tiled.get_slice(st_ltid)
                            tOs = thr_cp.partition_S(sVn_slice)
                            cVn = cute.make_identity_tensor((self.BV, self.BT))
                            tOc = thr_cp.partition_S(cVn)
                            tOr = cute.make_fragment_like(tOs, self.io_type)
                            cute.autovec_copy(tOs, tOr)

                            vn_token = self.upcast(bos + rev_ct * BT)
                            vn_raw = (
                                g_dv2.iterator + vn_token * H * V + h_idx * V + v_tile * self.BV
                            )
                            vn_ptr = cute.make_ptr(
                                self.io_type,
                                vn_raw.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            )
                            vn_stride = cute.assume(
                                H * V,
                                divby=128 // self.io_type.width,
                            )
                            gVn_chunk = cute.make_tensor(
                                vn_ptr,
                                cute.make_layout((self.BV, self.BT), stride=(1, vn_stride)),
                            )
                            tOg = thr_cp.partition_D(gVn_chunk)
                            for rb in cutlass.range_constexpr(cute.size(tOr.shape[2])):
                                bt_c = tOc[0, 0, rb][1]
                                if bt_c < rem:
                                    cute.copy(
                                        copy_dv2_tiled,
                                        tOr[None, None, rb],
                                        tOg[None, None, rb],
                                    )
                    else:
                        cute.copy(
                            a_dv2st,
                            sDV2st[None, dv2h.index],
                            gDV2st[(None, v_tile, rev_ct)],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                    dv2h.release()

                tile_idx = tile_idx + 1
                has_work = tile_idx < n_iters

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD (warp 4) — reverse TMA prefetch
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_id == self.LOAD_WARP_ID:
            cute.arch.setmaxregister_decrease(self.aux_regs)

            tile_idx = Int32(0)
            has_work = tile_idx < n_iters

            while has_work:
                w_idx = bx + tile_idx * gdx
                v_tile = w_idx % ((V + self.BV - 1) // self.BV)
                tmp = w_idx // ((V + self.BV - 1) // self.BV)
                h_idx = tmp % H
                seq_idx = tmp // H
                if cutlass.const_expr(self.varlen):
                    bos = cu_seqlens[seq_idx]
                    seq_len = cu_seqlens[seq_idx + 1] - bos
                    NT = (seq_len + BT - 1) // BT
                    db = Int32(0)
                else:
                    bos = Int32(0)
                    seq_len = T
                    NT = (T + BT - 1) // BT
                    db = seq_idx

                # Domain-offset TMA descriptors for varlen
                if cutlass.const_expr(self.varlen):
                    dk = cute.domain_offset((bos, 0, (0, 0)), desc_k)
                    dq = cute.domain_offset((0, bos, (0, 0)), desc_q)
                    dw_desc = cute.domain_offset((0, bos, (0, 0)), desc_w)
                    ddo = cute.domain_offset((0, bos, (0, 0)), desc_doepi)
                    ddvin = cute.domain_offset((0, bos, (0, 0)), desc_dvin)
                    dgk = cute.domain_offset((0, bos, (0, 0)), desc_gk)
                else:
                    dk = desc_k
                    dq = desc_q
                    dw_desc = desc_w
                    ddo = desc_doepi
                    ddvin = desc_dvin
                    dgk = desc_gk

                # Partition TMA operands
                tKs, tKg = self._part_a(atom_k, dk, sK, self.dv_tile, mma_dv, db, h_idx)
                tQs, tQg = self._part_a(atom_q, dq, sQ, self.qdo_tile, mma_qdo, db, h_idx)
                gDo_l = ddo[None, None, (h_idx, db)]
                _, sSDoEpi, gSDoEpi = self._part_epi(atom_doepi, gDo_l, (self.BV, self.BT), sDoEpi)
                tWs, tWg = self._part_a(atom_w, dw_desc, sW, self.wdv_tile, mma_wdv, db, h_idx)

                gDvIn_l = ddvin[None, None, (h_idx, db)]
                _, sSDvIn, gSDvIn = self._part_epi(atom_dvin, gDvIn_l, (self.BV, self.BT), sDvIn)

                gGK_l = dgk[None, None, (h_idx, db)]
                _, sSGK, gSGK = self._part_epi(atom_gk, gGK_l, (self.BK, 1), gk_3d)

                # ---------- Chunk loop (reverse TMA loads) ----------
                for ct in cutlass.range(0, NT, unroll=0):
                    rev_ct = NT - 1 - ct

                    # gk load FIRST (last timestep of chunk, clamped for partial)
                    if use_gk:
                        gk_t = rev_ct * self.BT + self.BT - 1
                        rem = seq_len - rev_ct * self.BT
                        if rem < self.BT:
                            gk_t = seq_len - 1
                        gkh = pgk_P.acquire_and_advance()
                        cute.copy(
                            atom=atom_gk,
                            src=gSGK[(None, 0, gk_t)],
                            dst=sSGK[None, gkh.index],
                            tma_bar_ptr=gkh.barrier,
                        )

                    # k load (A-operand for MMA1)
                    kh = pk_P.acquire_and_advance()
                    cute.copy(
                        atom=atom_k,
                        src=tKg[None, rev_ct, 0],
                        dst=tKs[None, kh.index],
                        tma_bar_ptr=kh.barrier,
                    )
                    # q^T load (A-operand for MMA2)
                    qh = pq_P.acquire_and_advance()
                    cute.copy(
                        atom=atom_q,
                        src=tQg[None, 0, rev_ct],
                        dst=tQs[None, qh.index],
                        tma_bar_ptr=qh.barrier,
                    )
                    # do load (epi → sDoEpi; COMPUTE masks + R2S → sDo for MMA2)
                    doepih = pdoepi_P.acquire_and_advance()
                    cute.copy(
                        atom=atom_doepi,
                        src=gSDoEpi[(None, v_tile, rev_ct)],
                        dst=sSDoEpi[None, doepih.index],
                        tma_bar_ptr=doepih.barrier,
                    )
                    # w^T load (A-operand for MMA3)
                    wh = pw_P.acquire_and_advance()
                    cute.copy(
                        atom=atom_w,
                        src=tWg[None, 0, rev_ct],
                        dst=tWs[None, wh.index],
                        tma_bar_ptr=wh.barrier,
                    )
                    # dv_in load
                    dvinh = pdvin_P.acquire_and_advance()
                    cute.copy(
                        atom=atom_dvin,
                        src=gSDvIn[(None, v_tile, rev_ct)],
                        dst=sSDvIn[None, dvinh.index],
                        tma_bar_ptr=dvinh.barrier,
                    )

                tile_idx = tile_idx + 1
                has_work = tile_idx < n_iters

        # ///////////////////////////////////////////////////////////////////////////////
        #  GK PRECOMPUTE (warp 5) — dedicated exp2 precompute
        # ///////////////////////////////////////////////////////////////////////////////
        elif warp_id == self.GK_WARP_ID:
            cute.arch.setmaxregister_decrease(self.aux_regs)
            gk_tid = tid - self.GK_WARP_ID * self.WARP_SZ

            tile_idx = Int32(0)
            has_work = tile_idx < n_iters

            while has_work:
                w_idx = bx + tile_idx * gdx
                bi_h = (w_idx // ((V + self.BV - 1) // self.BV)) // H
                if cutlass.const_expr(self.varlen):
                    tok_h = cu_seqlens[bi_h]
                    NT = (cu_seqlens[bi_h + 1] - tok_h + BT - 1) // BT
                else:
                    NT = (T + BT - 1) // BT

                for _ct in cutlass.range(0, NT, unroll=0):
                    if use_gk:
                        gkh = pgk_C.wait_and_advance()
                        gk_rdy = pgk_rdy_P.acquire_and_advance()
                        for i in cutlass.range(4, unroll_full=True):
                            idx = gk_tid * 4 + i
                            gk_exp_buf[(idx, gk_rdy.index)] = cute.exp2(gk_buf[(idx, gkh.index)])
                        gkh.release()
                        cute.arch.fence_proxy("async.shared", space="cta")
                        gk_rdy.commit()

                tile_idx = tile_idx + 1
                has_work = tile_idx < n_iters

        # TMEM teardown (all warps)
        tmem.relinquish_alloc_permit()
        self.tmem_free_bar.arrive_and_wait()
        tmem.free(tp)

    # ------------------------------------------------------------------
    # SMEM and TMA helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _neutralize_reduction_rows(self, smem, stage, valid_rows, tid):
        """Zero token columns ``[valid_rows, BT)`` of a Q/W stage and publish to UMMA.

        MMA2/MMA3 reduce over tokens, so a tail chunk's over-read tokens must be
        finite on both operands: the masked B-operand contributes zero, but
        ``0 * NaN`` is NaN and the physical suffix past ``cu_seqlens[-1]`` is
        undefined. This runs for every ragged tail, not just the terminal one,
        because any tail's over-read window may cross the terminal offset
        (e.g. trailing empty sequences).
        """
        key_atom = cute.size(smem, mode=[0, 0, 0])
        token_atom = cute.size(smem, mode=[0, 1])
        # Trace-time guards for the hard-coded coordinate nesting below; a
        # make_smem_layout_a change that reshapes the atom would otherwise
        # silently redirect these writes inside the same buffer.
        assert cute.size(smem, mode=[1]) == 1, "expected a single rest-M mode"
        assert key_atom * cute.size(smem, mode=[0, 0, 1]) == self.BK, "key modes must tile BK"
        assert token_atom * cute.size(smem, mode=[2]) == self.BT, "token modes must tile BT"
        for token in cutlass.range(valid_rows, self.BT, unroll=1):
            for key_block in cutlass.range_constexpr(self.BK // self.WARP_SZ):
                key = tid + key_block * self.WARP_SZ
                coord = (
                    ((key % key_atom, key // key_atom), token % token_atom),
                    0,
                    token // token_atom,
                    stage,
                )
                smem[coord] = self.io_type(0.0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _part_a(self, atom, desc, smem, tile, mma, batch, head):
        """Partition A-operand for TMA copy (SS-mode)."""
        g = cute.local_tile(desc, cute.slice_(tile, (None, 0, None)), (None, None, (head, batch)))
        thr = mma.get_slice(0)
        part = thr.partition_A(g)
        s, d = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(part, 0, 3),
        )
        return s, d

    @cute.jit
    def _part_b(self, atom, desc, smem, tile, mma, batch, head):
        """Partition B-operand for TMA copy."""
        g = cute.local_tile(desc, cute.slice_(tile, (0, None, None)), (None, None, (head, batch)))
        thr = mma.get_slice(0)
        part = thr.partition_B(g)
        s, d = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(part, 0, 3),
        )
        return s, d

    @cute.jit
    def _part_epi(self, atom, g_mnl, tile, s_buf):
        """Partition for epilogue-style TMA."""
        g_div = cute.flat_divide(g_mnl, tile)
        sg = cute.group_modes(s_buf, 0, 2)
        gg = cute.group_modes(g_div, 0, 2)
        ss, gs = cpasync.tma_partition(atom, 0, cute.make_layout(1), sg, gg)
        return atom, ss, gs

    @staticmethod
    def _plan_tmem(
        mma_dv,
        tile_dv,
        dv_depth,
        mma_qdo,
        tile_qdo,
        qdo_depth,
        mma_wdv,
        tile_wdv,
        wdv_depth,
    ):
        """SS-mode: TMEM only for 3 accumulator regions."""
        CAP = 512
        dv_c = mma_dv.partition_shape_C(tile_dv[:2])
        n_dv = tcgen05.find_tmem_tensor_col_offset(
            mma_dv.make_fragment_C(cute.append(dv_c, dv_depth))
        )
        qdo_c = mma_qdo.partition_shape_C(tile_qdo[:2])
        n_qdo = tcgen05.find_tmem_tensor_col_offset(
            mma_qdo.make_fragment_C(cute.append(qdo_c, qdo_depth))
        )
        wdv_c = mma_wdv.partition_shape_C(tile_wdv[:2])
        n_wdv = tcgen05.find_tmem_tensor_col_offset(
            mma_wdv.make_fragment_C(cute.append(wdv_c, wdv_depth))
        )
        o_dv = 0
        o_qdo = o_dv + n_dv
        o_wdv = o_qdo + n_qdo
        raw = o_wdv + n_wdv
        total = 1
        while total < raw:
            total *= 2
        assert total <= CAP, f"TMEM overflow: {total}>{CAP}"
        return o_dv, o_qdo, o_wdv, total

    def _launch_grid(self, B, H, V):
        # Target metadata is inherited by forked compile workers; do not query CUDA here.
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA compilation requires a CUDA target with an SM count")
        return (sm_count, 1, 1)


# ============================================================================
# Compile cache + TVM-FFI
# ============================================================================


@jit_cache
def _compile_bwd_dhu(varlen, H, K, V, chunk_size, bv=16, use_int64_offsets=False):
    """Compile one BlackwellDeltaHBwdV1 variant."""
    kern = BlackwellDeltaHBwdV1(
        chunk_size=chunk_size,
        head_k=K,
        head_v=V,
        head_bv=bv,
        varlen=varlen,
        num_heads=H,
        use_int64_offsets=use_int64_offsets,
    )
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    sa, sb, snt, sn, sns = (sym_int() for _ in range(5))

    if varlen:
        qf = make_fake_compact_tensor(
            cutlass.BFloat16, (sa, H, K), stride_order=(2, 1, 0), assumed_align=128
        )
        kf = make_fake_compact_tensor(
            cutlass.BFloat16, (sa, H, K), stride_order=(2, 1, 0), assumed_align=128
        )
        wf = make_fake_compact_tensor(
            cutlass.BFloat16, (sa, H, K), stride_order=(2, 1, 0), assumed_align=128
        )
        dof = make_fake_compact_tensor(
            cutlass.BFloat16, (sa, H, V), stride_order=(2, 1, 0), assumed_align=128
        )
        dvf = make_fake_compact_tensor(
            cutlass.BFloat16, (sa, H, V), stride_order=(2, 1, 0), assumed_align=128
        )
        gkf = make_fake_compact_tensor(
            cutlass.Float32, (sa, H, K), stride_order=(2, 1, 0), assumed_align=128
        )
        dhtf = make_fake_compact_tensor(
            cutlass.Float32,
            (sns, H, K, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dh0f = make_fake_compact_tensor(
            cutlass.Float32,
            (sns, H, K, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dhof = make_fake_compact_tensor(
            cutlass.BFloat16,
            (snt, H, K, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dv2f = make_fake_compact_tensor(
            cutlass.BFloat16, (sa, H, V), stride_order=(2, 1, 0), assumed_align=128
        )
    else:
        qf = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sa, sb, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        kf = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sa, sb, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        wf = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sa, sb, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dof = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sa, sb, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dvf = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sa, sb, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        gkf = make_fake_compact_tensor(
            cutlass.Float32,
            (sa, sb, H, K),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dhtf = make_fake_compact_tensor(
            cutlass.Float32,
            (sns, H, K, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dh0f = make_fake_compact_tensor(
            cutlass.Float32,
            (sns, H, K, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )
        dhof = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sa, snt, H, K, V),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=128,
        )
        dv2f = make_fake_compact_tensor(
            cutlass.BFloat16,
            (sa, sb, H, V),
            stride_order=(3, 2, 1, 0),
            assumed_align=128,
        )

    cuf = make_fake_compact_tensor(cutlass.Int32, (sn,), assumed_align=128)
    cof = make_fake_compact_tensor(cutlass.Int32, (sn,), assumed_align=128)

    return compile_tvm_ffi(
        kern,
        qf,
        kf,
        wf,
        dof,
        dvf,
        gkf,
        dhtf,
        dh0f,
        dhof,
        dv2f,
        cuf,
        cof,
        (Int32(1), Int32(1), Int32(H), Int32(K), Int32(V)),
        Float32(1.0),
        Int32(0),
        Int32(0),
        Int32(0),
    )


def _is_fake_mode() -> bool:
    """True when torch.compile is tracing (FakeTensorMode active)."""
    return active_fake_mode() is not None


# ============================================================================
# Public API wrapper
# ============================================================================


def _get_dummy(shape, dtype, device) -> torch.Tensor:
    """Create an invocation-local placeholder for an unused kernel argument."""
    return torch.empty(shape, dtype=dtype, device=device)


def blackwell_delta_h_bwd_dhu_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float = 1.0,
    chunk_size: int = 64,
    dv2_out: torch.Tensor | None = None,
    bv: int = 16,
    metadata: RaggedChunkMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the CuTeDSL SM100 delta-H backward leaf.

    ``metadata=None`` selects dense execution. Ragged callers pass one canonical metadata
    object; empty boundary ranges encode inactive logical sequences. Ragged ``dh`` has
    ``metadata.capacity`` chunk slots, and values beyond ``chunk_offsets[-1]`` are undefined.
    Returns ``(dh, dh0, dv2)``, where ``dh0`` is absent when no initial state was supplied.
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    BT = chunk_size
    dev = q.device

    assert K == 128, "BlackwellDeltaHBwd requires head_k=128"

    if metadata is None:
        NT = (T + BT - 1) // BT
    else:
        metadata.validate_chunk_size(chunk_size)
        NT = metadata.capacity

    fgk = 1 if gk is not None else 0
    fdht = 1 if dht is not None else 0
    fdh0 = 1 if h0 is not None else 0

    dh_out = q.new_empty(B, NT, H, K, V)
    dh0_out = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = dv2_out if dv2_out is not None else torch.empty_like(dv)

    if _is_fake_mode():
        return dh_out, dh0_out, dv2

    if metadata is not None:
        assert B == 1, "varlen requires B=1"
        N = metadata.cu_seqlens.shape[0] - 1
        q_k, k_k, w_k = q[0], k[0], w[0]
        do_k, dv_k = do[0], dv[0]
        gk_k = gk[0] if gk is not None else _get_dummy((T, H, K), torch.float32, dev)
        dht_k = dht if dht is not None else _get_dummy((N, H, K, V), torch.float32, dev)
        dh0_k = dh0_out if dh0_out is not None else _get_dummy((N, H, K, V), torch.float32, dev)
        dho = dh_out[0]  # (NT, H, K, V)
        dv2_k = dv2[0]

        ps = (Int32(N), Int32(T), Int32(H), Int32(K), Int32(V))
        use_int64_offsets = requires_int64_abi(
            q_k,
            k_k,
            w_k,
            do_k,
            dv_k,
            gk_k,
            dht_k,
            dh0_k,
            dho,
            dv2_k,
        )
        fn = _compile_bwd_dhu(True, H, K, V, chunk_size, bv, use_int64_offsets=use_int64_offsets)
        fn(
            q_k,
            k_k,
            w_k,
            do_k,
            dv_k,
            gk_k,
            dht_k,
            dh0_k,
            dho,
            dv2_k,
            metadata.cu_seqlens,
            metadata.chunk_offsets,
            ps,
            Float32(scale),
            Int32(fgk),
            Int32(fdht),
            Int32(fdh0),
        )
    else:
        gk_k = gk if gk is not None else _get_dummy((B, T, H, K), torch.float32, dev)
        dht_k = dht if dht is not None else _get_dummy((B, H, K, V), torch.float32, dev)
        dh0_k = dh0_out if dh0_out is not None else _get_dummy((B, H, K, V), torch.float32, dev)
        cu_d = _get_dummy((2,), torch.int32, dev)
        co_d = _get_dummy((2,), torch.int32, dev)

        ps = (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V))
        use_int64_offsets = requires_int64_abi(
            q,
            k,
            w,
            do,
            dv,
            gk_k,
            dht_k,
            dh0_k,
            dh_out,
            dv2,
        )
        fn = _compile_bwd_dhu(False, H, K, V, chunk_size, bv, use_int64_offsets=use_int64_offsets)
        fn(
            q,
            k,
            w,
            do,
            dv,
            gk_k,
            dht_k,
            dh0_k,
            dh_out,
            dv2,
            cu_d,
            co_d,
            ps,
            Float32(scale),
            Int32(fgk),
            Int32(fdht),
            Int32(fdh0),
        )

    return dh_out, dh0_out, dv2
