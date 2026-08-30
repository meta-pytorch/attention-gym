# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""
BlackwellDeltaHBwd — SM100 warp-specialized kernel for KDA backward dhu recurrence.

Implements the backward hidden-state gradient update across chunks (reverse order):

    dh_out[c]  = snapshot(dh)
    dv[c]      = GEMM_DV(K[c], dh) + dv_intra[c]
    dh         = diag(exp2(gk_last[c])) @ dh
                 + GEMM_QDO(Q[c]^T, do[c]) * scale
                 - GEMM_WDV(W[c], dv[c])

Architecture follows FMHA/MSLK Blackwell patterns:
  - 8-warp CTA (256 threads), occupancy 1 for maximum SMEM/regs
  - Separate warp roles: CUDA(0-3), Load(4), GK(5), MMA(6), Store(7)
  - SS-mode MMA: both operands from SMEM, accumulators in TMEM
  - dO is TMA-loaded directly into the shared MMA B-operand ring
  - Reverse chunk iteration: NT-1 down to 0

MMA operations per chunk:
  MMA1: k @ dh       → dv    (BT=64,  BV, K=128)
  MMA2: q^T @ dO     → qdo   (BK=128, BV, K=64)
  MMA3: w @ dv       → wdv   (BK=128, BV, K=64)
  MMA4: Aqk^T @ dO   → dv    (BT=64,  BV, K=64)
"""

from enum import IntEnum
from typing import NamedTuple

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Float32, Int32, Int64
from torch._guards import active_fake_mode

from attn_gym._backends.cute import (
    get_device_properties,
    make_fake_strided_tensor,
    tensor_supports_contiguous_dim,
)
from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import compile_tvm_ffi, requires_int64_abi
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import load_ragged_sequence_extent
from attn_gym.utils import ceildiv

_MIN_SEQUENCE_EXTENT_SEQUENCES = 32
_MIN_SEQUENCE_EXTENT_HEADS = 8
_IO_TYPES = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}
_IO_TYPE_NAMES = {
    cutlass.Float16: "fp16",
    cutlass.BFloat16: "bf16",
}


def select_delta_h_bv(
    value_dim: int,
    heads: int,
    logical_batch: int,
    device: torch.device,
) -> int:
    """Select the delta-H value tile width for the logical workload."""
    value_tiles = ceildiv(value_dim, 16) * heads * logical_batch
    return 32 if value_tiles > get_device_properties(device).multi_processor_count else 16


def should_bound_sequence_extent(
    tokens: int,
    sequences: int,
    heads: int,
    chunk_size: int,
    has_initial_state: bool,
) -> bool:
    """Select the measured graph-overcapture region for sequence-bounded dHU."""
    return (
        not has_initial_state
        and sequences >= _MIN_SEQUENCE_EXTENT_SEQUENCES
        and heads >= _MIN_SEQUENCE_EXTENT_HEADS
        and tokens <= sequences * chunk_size
    )


# ============================================================================
# BlackwellDeltaHBwd — warp-specialized backward inter-chunk recurrence
# ============================================================================


class WarpRole(IntEnum):
    """Warp-role boundaries in the persistent delta-H kernel."""

    CUDA = 0
    LOAD = 4
    GATE = 5
    MMA = 6
    STORE = 7
    END = 8


class TmaOp(NamedTuple):
    """One TMA copy atom and its tensor-map descriptor."""

    atom: cute.CopyAtom
    desc: cute.Tensor


class TmaOps(NamedTuple):
    """TMA operations owned by the delta-H kernel."""

    k: TmaOp
    q: TmaOp
    do: TmaOp
    w: TmaOp
    aqk: TmaOp
    dh_store: TmaOp
    dv2_store: TmaOp
    gk: TmaOp


class Mmas(NamedTuple):
    """Tensor-core operations owned by the MMA warp."""

    dv: cute.TiledMma
    qdo: cute.TiledMma
    aqdo: cute.TiledMma
    wdv: cute.TiledMma


class SmemLayouts(NamedTuple):
    """Shared-memory layouts in their device-kernel construction order."""

    k: cute.ComposedLayout
    dhb: cute.ComposedLayout
    dhb_store: cute.ComposedLayout
    q: cute.ComposedLayout
    do: cute.ComposedLayout
    w: cute.ComposedLayout
    dv2b: cute.ComposedLayout
    dv2b_store: cute.ComposedLayout
    aqk: cute.ComposedLayout
    dh_epi: cute.ComposedLayout
    dh_r2s: cute.ComposedLayout
    dv2_epi: cute.ComposedLayout
    dv2_r2s: cute.ComposedLayout


class GmemViews(NamedTuple):
    """Logical TMA and state views over the runtime tensor storage."""

    k: cute.Tensor
    qt: cute.Tensor
    wt: cute.Tensor
    gk_k: cute.Tensor
    do_vt: cute.Tensor
    dv2_t: cute.Tensor
    dv2: cute.Tensor
    aqk_t: cute.Tensor
    dh_out_t: cute.Tensor
    dht: cute.Tensor
    dh0_t: cute.Tensor


class BlackwellDeltaHBwd:
    """
    Warp-specialized SM100 kernel for gated delta rule backward dhu recurrence.

    dh state persists in CUDA warp registers (FP32) across all chunks — zero
    GMEM round-trips. SS-mode keeps both operands in SMEM and supports BV=16 or 32.

    MMA operations per chunk:
      MMA1: k @ dh       → dv    (BT=64,  BV, K=128)
      MMA2: q^T @ dO     → qdo   (BK=128, BV, K=64)
      MMA3: w @ dv       → wdv   (BK=128, BV, K=64)
      MMA4: Aqk^T @ dO   → dv    (BT=64,  BV, K=64)
    """

    # Warp role assignment (8 warps total)
    CUDA_WARP_IDS = tuple(range(WarpRole.CUDA, WarpRole.LOAD))
    WARP_SZ = cute.arch.WARP_SIZE
    N_WARPS = WarpRole.END
    CTA_THREADS = N_WARPS * WARP_SZ
    assert len(CUDA_WARP_IDS) * WARP_SZ == 128

    def __init__(
        self,
        num_heads: int,
        io_type: type[cutlass.Numeric] = cutlass.BFloat16,
        head_bv: int = 16,
        varlen: bool = False,
        use_int64_offsets: bool = False,
        bound_sequence_extent: bool = False,
        dynamic_state_layout: bool = False,
    ):
        assert head_bv in (16, 32), f"BV must be 16 or 32, got {head_bv}"
        self.head_k = 128
        self.head_v = 128
        self.acc_type = cutlass.Float32
        self.io_type = io_type
        self.num_heads = num_heads
        self.varlen = varlen
        self.use_int64_offsets = use_int64_offsets
        self.bound_sequence_extent = bound_sequence_extent
        self.dynamic_state_layout = dynamic_state_layout
        assert not bound_sequence_extent or varlen, "sequence extent applies only to varlen inputs"

        # Tile dimensions
        self.BT = 64
        self.BK = self.head_k
        self.BV = head_bv  # N-dim for SS-mode MMA (16 or 32)

        # Register budget per thread — BV=32 needs more for wider dh state
        self.cuda_regs = 128 if head_bv <= 16 else 160
        self.aux_regs = 40  # MMA/Load/Store/GK: minimal

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
        self.aqk_depth = 2  # Aqk A-operand load
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
        return (
            f"kda_bwd_dhu_dv_fused_vl{int(self.varlen)}_h{self.num_heads}"
            f"_k{self.head_k}_v{self.head_v}_bt{self.BT}_bv{self.BV}_{_IO_TYPE_NAMES[self.io_type]}"
            f"_i64{int(self.use_int64_offsets)}_se{int(self.bound_sequence_extent)}"
            f"_ds{int(self.dynamic_state_layout)}"
        )

    # ------------------------------------------------------------------
    # Host-side setup (__call__): GMEM layouts → TMA → SMEM → launch
    # ------------------------------------------------------------------

    @cute.jit
    def make_varlen_gmem_views(
        self,
        q_in,
        k_in,
        w_in,
        do_in,
        aqk,
        gk_in,
        dht_in,
        dh0_in,
        dh_out_in,
        dv2_out_in,
        problem_shape,
    ):
        """Re-rank packed storage into the logical TMA/state views."""
        B, T, H, K, V = problem_shape
        dB = Int32(1)
        NT = Int32(cute.size(dh_out_in.shape[0]))
        g_k = cute.make_tensor(
            k_in.iterator,
            cute.make_layout(
                (T, K, (H, dB)),
                stride=(self.upcast(H * K), 1, (K, self.upcast(T) * H * K)),
            ),
        )
        g_qt = cute.make_tensor(
            q_in.iterator,
            cute.make_layout(
                (K, T, (H, dB)),
                stride=(1, self.upcast(H * K), (K, self.upcast(T) * H * K)),
            ),
        )
        g_wt = cute.make_tensor(w_in.iterator, g_qt.layout)
        g_gk_k = cute.make_tensor(gk_in.iterator, g_qt.layout)
        g_do_vt = cute.make_tensor(
            do_in.iterator,
            cute.make_layout(
                (V, T, (H, dB)),
                stride=(1, self.upcast(H * V), (V, self.upcast(T) * H * V)),
            ),
        )
        g_dv2_t = cute.make_tensor(dv2_out_in.iterator, g_do_vt.layout)
        g_dv2 = cute.make_tensor(
            dv2_out_in.iterator,
            cute.make_layout(
                (T, V, (H, dB)),
                stride=(self.upcast(H * V), 1, (V, self.upcast(T) * H * V)),
            ),
        )
        g_aqk_t = cute.make_tensor(
            aqk.iterator,
            cute.make_layout(
                (self.BT, T, (H, dB)),
                stride=(
                    1,
                    self.upcast(H * self.BT),
                    (self.BT, self.upcast(T) * H * self.BT),
                ),
            ),
        )
        g_dh_out_t = cute.make_tensor(
            dh_out_in.iterator,
            cute.make_layout(
                (V, K, (NT, H, dB)),
                stride=(
                    1,
                    V,
                    (self.upcast(H * K * V), K * V, self.upcast(NT) * H * K * V),
                ),
            ),
        )
        if cutlass.const_expr(self.dynamic_state_layout):
            g_dht = cute.make_tensor(
                dht_in.iterator,
                cute.group_modes(cute.select(dht_in.layout, mode=[3, 2, 1, 0]), 2, 4),
            )
            g_dh0_t = cute.make_tensor(
                dh0_in.iterator,
                cute.group_modes(cute.select(dh0_in.layout, mode=[2, 3, 1, 0]), 2, 4),
            )
        else:
            g_dht = cute.make_tensor(
                dht_in.iterator,
                cute.make_layout(
                    (K, V, (H, B)),
                    stride=(1, K, (K * V, self.upcast(H * K * V))),
                ),
            )
            g_dh0_t = cute.make_tensor(
                dh0_in.iterator,
                cute.make_layout(
                    (V, K, (H, B)),
                    stride=(K, 1, (K * V, self.upcast(H * K * V))),
                ),
            )
        return GmemViews(
            g_k,
            g_qt,
            g_wt,
            g_gk_k,
            g_do_vt,
            g_dv2_t,
            g_dv2,
            g_aqk_t,
            g_dh_out_t,
            g_dht,
            g_dh0_t,
        )

    @cute.jit
    def make_dense_gmem_views(
        self,
        q_in,
        k_in,
        w_in,
        do_in,
        aqk,
        gk_in,
        dht_in,
        dh0_in,
        dh_out_in,
        dv2_out_in,
    ):
        """Re-rank dense runtime tensors with source-derived layouts."""
        return GmemViews(
            cute.make_tensor(
                k_in.iterator,
                cute.group_modes(cute.select(k_in.layout, mode=[1, 3, 2, 0]), 2, 4),
            ),
            cute.make_tensor(
                q_in.iterator,
                cute.group_modes(cute.select(q_in.layout, mode=[3, 1, 2, 0]), 2, 4),
            ),
            cute.make_tensor(
                w_in.iterator,
                cute.group_modes(cute.select(w_in.layout, mode=[3, 1, 2, 0]), 2, 4),
            ),
            cute.make_tensor(
                gk_in.iterator,
                cute.group_modes(cute.select(gk_in.layout, mode=[3, 1, 2, 0]), 2, 4),
            ),
            cute.make_tensor(
                do_in.iterator,
                cute.group_modes(cute.select(do_in.layout, mode=[3, 1, 2, 0]), 2, 4),
            ),
            cute.make_tensor(
                dv2_out_in.iterator,
                cute.group_modes(cute.select(dv2_out_in.layout, mode=[3, 1, 2, 0]), 2, 4),
            ),
            cute.make_tensor(
                dv2_out_in.iterator,
                cute.group_modes(cute.select(dv2_out_in.layout, mode=[1, 3, 2, 0]), 2, 4),
            ),
            cute.make_tensor(
                aqk.iterator,
                cute.group_modes(cute.select(aqk.layout, mode=[3, 1, 2, 0]), 2, 4),
            ),
            cute.make_tensor(
                dh_out_in.iterator,
                cute.group_modes(cute.select(dh_out_in.layout, mode=[4, 3, 1, 2, 0]), 2, 5),
            ),
            cute.make_tensor(
                dht_in.iterator,
                cute.group_modes(cute.select(dht_in.layout, mode=[3, 2, 1, 0]), 2, 4),
            ),
            cute.make_tensor(
                dh0_in.iterator,
                cute.group_modes(cute.select(dh0_in.layout, mode=[2, 3, 1, 0]), 2, 4),
            ),
        )

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        w_in: cute.Tensor,
        do_in: cute.Tensor,
        aqk: cute.Tensor,
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
        if cutlass.const_expr(self.varlen):
            views = self.make_varlen_gmem_views(
                q_in,
                k_in,
                w_in,
                do_in,
                aqk,
                gk_in,
                dht_in,
                dh0_in,
                dh_out_in,
                dv2_out_in,
                problem_shape,
            )
        else:
            views = self.make_dense_gmem_views(
                q_in,
                k_in,
                w_in,
                do_in,
                aqk,
                gk_in,
                dht_in,
                dh0_in,
                dh_out_in,
                dv2_out_in,
            )
        (
            g_k,
            g_qt,
            g_wt,
            g_gk_k,
            g_do_vt,
            g_dv2_t,
            g_dv2,
            g_aqk_t,
            g_dh_out_t,
            g_dht,
            g_dh0_t,
        ) = views
        cu_lens = cu_seqlens_in
        chunk_offsets = chunk_offsets_in

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
        # MMA4: Aqk^T @ do → dv_intra, accumulated into MMA1's dv TMEM tile.
        aqdo_tile = (self.BT, self.BV, self.BT)
        mma_aqdo = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_type,
            self.cta_group,
            aqdo_tile[:2],
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
        # TMA loads dO directly into its depth-2 MMA B-operand ring.
        s_do_staged = sm100_utils.make_smem_layout_b(
            mma_qdo,
            self.qdo_tile,
            self.io_type,
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

        # Aqk^T A-operand for the fused K=64 dv-intra MMA.
        s_aqk_staged = sm100_utils.make_smem_layout_a(
            mma_aqdo,
            aqdo_tile,
            self.io_type,
            self.aqk_depth,
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
        s_do_one = cute.select(s_do_staged, mode=[0, 1, 2])
        atom_do, desc_do = cute.nvgpu.make_tiled_tma_atom_B(
            tma_ld,
            g_do_vt,
            s_do_one,
            self.qdo_tile,
            mma_qdo,
            clust_lay.shape,
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

        # Aqk^T: TMA load directly into the fused MMA A-operand ring.
        s_aqk_one = cute.select(s_aqk_staged, mode=[0, 1, 2])
        atom_aqk, desc_aqk = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld,
            g_aqk_t,
            s_aqk_one,
            aqdo_tile,
            mma_aqdo,
            clust_lay.shape,
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

        cp_bits = 128
        cp_elems = cp_bits // self.io_type.width
        copy_atom_uni = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_type,
            num_bits_per_copy=cp_bits,
        )
        vn_thr_d0 = self.BV // cp_elems
        vn_thr_lay = cute.make_ordered_layout((vn_thr_d0, self.WARP_SZ // vn_thr_d0), order=(0, 1))
        copy_dv2_tiled = cute.make_tiled_copy_tv(
            copy_atom_uni, vn_thr_lay, cute.make_layout((cp_elems, 1))
        )

        # TMA byte counts
        self.k_bytes = cute.size_in_bytes(self.io_type, s_k_one)
        self.q_bytes = cute.size_in_bytes(self.io_type, s_q_one)
        self.do_bytes = cute.size_in_bytes(self.io_type, s_do_one)
        self.w_bytes = cute.size_in_bytes(self.io_type, s_w_one)
        self.aqk_bytes = cute.size_in_bytes(self.io_type, s_aqk_one)
        self.gk_bytes = cute.size_in_bytes(cutlass.Float32, s_gk_2d)

        # --- SharedStorage struct ---
        @cute.struct
        class Shared:
            # Pipeline barriers (each needs depth * 2 Int64s)
            bar_k: cute.struct.MemRange[Int64, self.k_depth * 2]
            bar_q: cute.struct.MemRange[Int64, self.q_depth * 2]
            bar_do: cute.struct.MemRange[Int64, self.do_depth * 2]
            bar_w: cute.struct.MemRange[Int64, self.w_depth * 2]
            bar_aqk: cute.struct.MemRange[Int64, self.aqk_depth * 2]
            bar_gk: cute.struct.MemRange[Int64, self.gk_depth * 2]
            bar_gk_rdy: cute.struct.MemRange[Int64, self.gk_depth * 2]
            bar_dhb: cute.struct.MemRange[Int64, 1 * 2]  # dh B-operand CUDA→MMA
            bar_dv: cute.struct.MemRange[Int64, self.dv_acc_depth * 2]  # MMA1 done
            bar_qdo: cute.struct.MemRange[Int64, self.qdo_acc_depth * 2]  # MMA2 done
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
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_w_staged)], self.align
            ]
            sDv2b: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_dv2b_staged)],
                self.align,
            ]
            sAqk: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_aqk_staged)],
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

        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            Mmas(mma_dv, mma_qdo, mma_aqdo, mma_wdv),
            TmaOps(
                TmaOp(atom_k, desc_k),
                TmaOp(atom_q, desc_q),
                TmaOp(atom_do, desc_do),
                TmaOp(atom_w, desc_w),
                TmaOp(atom_aqk, desc_aqk),
                TmaOp(atom_dhst, desc_dhst),
                TmaOp(atom_dv2st, desc_dv2st),
                TmaOp(atom_gk, desc_gk),
            ),
            g_dht,
            g_dh0_t,
            g_dv2,
            copy_dv2_tiled,
            cu_lens,
            chunk_offsets,
            SmemLayouts(
                s_k_staged,
                s_dhb_staged,
                s_dhb_store_staged,
                s_q_staged,
                s_do_staged,
                s_w_staged,
                s_dv2b_staged,
                s_dv2b_store_staged,
                s_aqk_staged,
                s_dh_epi_staged,
                s_dh_r2s_staged,
                s_dv2_epi_staged,
                s_dv2_r2s_staged,
            ),
            problem_shape,
            scale,
            use_gk,
            use_dht,
            use_dh0,
        ).launch(
            grid=self._launch_grid(),
            block=[self.CTA_THREADS, 1, 1],
            cluster=self.cluster,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _decode_work(self, work_index):
        """Return the value tile, head, and sequence for one persistent work item."""
        value_tiles = self.head_v // self.BV
        value_tile = work_index % value_tiles
        tile = work_index // value_tiles
        return value_tile, tile % self.num_heads, tile // self.num_heads

    @cute.jit
    def run_store(
        self,
        bx,
        gdx,
        n_iters,
        tokens,
        dense_chunks,
        atom_dhst,
        desc_dhst,
        atom_dv2st,
        desc_dv2st,
        copy_dv2_tiled,
        g_dv2,
        cu_seqlens,
        chunk_offsets,
        sDhEpi,
        sDv2Epi,
        pdh_epi_C,
        pdv2_epi_C,
    ):
        """Store packed chunk states and dV without writing tail padding."""
        cute.arch.setmaxregister_decrease(self.aux_regs)
        cpasync.prefetch_descriptor(atom_dhst)
        cpasync.prefetch_descriptor(atom_dv2st)
        tid, _, _ = cute.arch.thread_idx()
        store_tid = tid - WarpRole.STORE * self.WARP_SZ

        tile_idx = Int32(0)
        has_work = tile_idx < n_iters
        while has_work:
            work_index = bx + tile_idx * gdx
            value_tile, head_idx, sequence_idx = self._decode_work(work_index)
            if cutlass.const_expr(self.varlen):
                bos = cu_seqlens[sequence_idx]
                sequence_length = cu_seqlens[sequence_idx + 1] - bos
                num_chunks = (sequence_length + self.BT - 1) // self.BT
                batch_idx = Int32(0)
                chunk_offset = chunk_offsets[sequence_idx]
                dh_desc = cute.domain_offset((0, 0, (chunk_offset, 0, 0)), desc_dhst)
                dv_desc = cute.domain_offset((0, bos, (0, 0)), desc_dv2st)
            else:
                bos = Int32(0)
                sequence_length = tokens
                num_chunks = dense_chunks
                batch_idx = sequence_idx
                dh_desc = desc_dhst
                dv_desc = desc_dv2st

            gDH_s = dh_desc[None, None, (None, head_idx, batch_idx)]
            sDHst, gDHst = self._part_epi(atom_dhst, gDH_s, (self.BV, self.BK), sDhEpi)
            gDV2_s = dv_desc[None, None, (head_idx, batch_idx)]
            sDV2st, gDV2st = self._part_epi(atom_dv2st, gDV2_s, (self.BV, self.BT), sDv2Epi)

            for chunk in cutlass.range(0, num_chunks, unroll=0):
                reverse_chunk = num_chunks - 1 - chunk
                state_handle = pdh_epi_C.wait_and_advance()
                cute.copy(
                    atom_dhst,
                    sDHst[None, state_handle.index],
                    gDHst[(None, value_tile, 0, reverse_chunk)],
                )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                state_handle.release()

                value_handle = pdv2_epi_C.wait_and_advance()
                remaining = sequence_length - reverse_chunk * self.BT
                if cutlass.const_expr(self.varlen):
                    if remaining < self.BT:
                        source = sDv2Epi[None, None, value_handle.index]
                        thread_copy = copy_dv2_tiled.get_slice(store_tid)
                        source_partition = thread_copy.partition_S(source)
                        coordinates = thread_copy.partition_S(
                            cute.make_identity_tensor((self.BV, self.BT))
                        )
                        registers = cute.make_fragment_like(source_partition, self.io_type)
                        cute.autovec_copy(source_partition, registers)
                        token = self.upcast(bos + reverse_chunk * self.BT)
                        raw = (
                            g_dv2.iterator
                            + token * self.num_heads * self.head_v
                            + head_idx * self.head_v
                            + value_tile * self.BV
                        )
                        pointer = cute.make_ptr(
                            self.io_type,
                            raw.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        stride = cute.assume(
                            self.num_heads * self.head_v,
                            divby=128 // self.io_type.width,
                        )
                        destination = cute.make_tensor(
                            pointer,
                            cute.make_layout((self.BV, self.BT), stride=(1, stride)),
                        )
                        destination_partition = thread_copy.partition_D(destination)
                        for row_block in cutlass.range_constexpr(cute.size(registers.shape[2])):
                            token_coord = coordinates[0, 0, row_block][1]
                            if token_coord < remaining:
                                cute.copy(
                                    copy_dv2_tiled,
                                    registers[None, None, row_block],
                                    destination_partition[None, None, row_block],
                                )
                    else:
                        cute.copy(
                            atom_dv2st,
                            sDV2st[None, value_handle.index],
                            gDV2st[(None, value_tile, reverse_chunk)],
                        )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                else:
                    cute.copy(
                        atom_dv2st,
                        sDV2st[None, value_handle.index],
                        gDV2st[(None, value_tile, reverse_chunk)],
                    )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                value_handle.release()

            tile_idx = tile_idx + 1
            has_work = tile_idx < n_iters

    @cute.jit
    def run_state(
        self,
        bx,
        gdx,
        n_iters,
        tokens,
        dense_chunks,
        cu_seqlens,
        g_dht,
        g_dh0_t,
        t_dv_acc,
        t_qdo_acc,
        t_wdv_acc,
        sDhb_store,
        sDhEpi_store,
        sDv2b_store,
        sDv2Epi_store,
        pdhb_P,
        pdh_epi_P,
        pdv_C,
        pdv2b_P,
        pdv2_epi_P,
        pgk_rdy_C,
        pqdo_C,
        pwdv_C,
        gk_exp_buf,
        use_gk,
        use_dht,
        use_dh0,
        scale,
    ):
        """Own dh registers; consume result/gate stages and produce dh/dV operand stages."""
        cute.arch.setmaxregister_increase(self.cuda_regs)
        tid, _, _ = cute.arch.thread_idx()
        local_tid = tid % (self.WARP_SZ * len(self.CUDA_WARP_IDS))

        # --- T2R setup: read MMA1 (dv) accumulator (BT,BV fp32) from TMEM ---
        t2r_dv_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BV // 8), tcgen05.Pack.NONE),
            self.acc_type,
        )
        dv_flat = t_dv_acc[((None, None), 0, 0, None)]
        tc_t2r_dv = tcgen05.make_tmem_copy(t2r_dv_atom, dv_flat[(None, None, 0)])
        sl_dv = tc_t2r_dv.get_slice(local_tid)
        p_t_dv = sl_dv.partition_S(dv_flat)

        # --- T2R setup: read MMA2 (qdo) accumulator (BK,BV fp32) from TMEM ---
        t2r_qdo_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BV // 8), tcgen05.Pack.NONE),
            self.acc_type,
        )
        qdo_flat = t_qdo_acc[((None, None), 0, 0, None)]
        tc_t2r_qdo = tcgen05.make_tmem_copy(t2r_qdo_atom, qdo_flat[(None, None, 0)])
        sl_qdo = tc_t2r_qdo.get_slice(local_tid)
        p_t_qdo = sl_qdo.partition_S(qdo_flat)

        # --- T2R setup: read MMA3 (wdv) accumulator (BK,BV fp32) from TMEM ---
        t2r_wdv_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BV // 8), tcgen05.Pack.NONE),
            self.acc_type,
        )
        wdv_flat = t_wdv_acc[((None, None), 0, 0, None)]
        tc_t2r_wdv = tcgen05.make_tmem_copy(t2r_wdv_atom, wdv_flat[(None, None, 0)])
        sl_wdv = tc_t2r_wdv.get_slice(local_tid)
        p_t_wdv = sl_wdv.partition_S(wdv_flat)

        # Identity tensors provide both coordinate maps and register-fragment shapes.
        # dv tile: (BT, BV) coords
        id_tv = cute.make_identity_tensor(cute.dice(self.dv_tile, (1, 1, None)))
        coords_tv = sl_dv.partition_D(id_tv)
        # qdo/wdv tile: (BK, BV) coords — dh state uses this
        id_kv = cute.make_identity_tensor(cute.dice(self.qdo_tile, (1, 1, None)))
        coords_kv = sl_qdo.partition_D(id_kv)
        coords_wdv = sl_wdv.partition_D(id_kv)
        st = cute.make_rmem_tensor(coords_kv.shape, self.acc_type)

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
            v_tile, h_idx, seq_idx = self._decode_work(w_idx)
            if cutlass.const_expr(self.varlen):
                bos = cu_seqlens[seq_idx]
                sequence_length = cu_seqlens[seq_idx + 1] - bos
                num_chunks = (sequence_length + self.BT - 1) // self.BT
            else:
                sequence_length = tokens
                num_chunks = dense_chunks

            # Initialize dh from dht or zeros
            if use_dht:
                gDht_slice = g_dht[None, None, (h_idx, seq_idx)]
                for ei in cutlass.range(cute.size(st), unroll_full=True):
                    kc, vc = coords_kv[ei]
                    st[ei] = gDht_slice[(kc, vc + v_tile * self.BV)]
            else:
                for ei in cutlass.range(cute.size(st), unroll_full=True):
                    st[ei] = Float32(0.0)

            # ========== Chunk loop: reverse order ==========
            for ct in cutlass.range(0, num_chunks, unroll=0):
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

                # ---- Phase 2: T2R fused dv result ----
                dvh = pdv_C.wait_and_advance()
                dv_reg = cute.make_rmem_tensor(coords_tv.shape, self.acc_type)
                cute.copy(tc_t2r_dv, p_t_dv[(None, None, None, dvh.index)], dv_reg)
                cute.arch.fence_view_async_tmem_load()
                dvh.release()

                if cutlass.const_expr(self.varlen):
                    valid_rows = sequence_length - (num_chunks - 1 - ct) * self.BT
                    if valid_rows < self.BT:
                        for ei in cutlass.range_constexpr(cute.size(coords_tv)):
                            token_coord, _value_coord = coords_tv[ei]
                            if token_coord >= valid_rows:
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
                dst_dv2 = thr_r2s_dv2_epi.partition_D(sDv2Epi_store[(None, None, dv2_epi_h.index)])
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
                qdo_reg = cute.make_rmem_tensor(coords_kv.shape, self.acc_type)
                cute.copy(tc_t2r_qdo, p_t_qdo[(None, None, None, qdoh.index)], qdo_reg)
                cute.arch.fence_view_async_tmem_load()
                qdoh.release()

                # T2R wdv result from MMA3
                wdvh = pwdv_C.wait_and_advance()
                wdv_reg = cute.make_rmem_tensor(coords_wdv.shape, self.acc_type)
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
                    gDh0[vc + v_tile * self.BV, kc] = st[ei]

            tile_idx = tile_idx + 1
            has_work = tile_idx < n_iters

    @cute.jit
    def run_load(
        self,
        bx,
        gdx,
        n_iters,
        tokens,
        dense_chunks,
        cu_seqlens,
        mma_dv,
        mma_qdo,
        mma_aqdo,
        mma_wdv,
        atom_k,
        desc_k,
        atom_q,
        desc_q,
        atom_do,
        desc_do,
        atom_w,
        desc_w,
        atom_aqk,
        desc_aqk,
        atom_gk,
        desc_gk,
        sK,
        sQ,
        sDo,
        sW,
        sAqk,
        gk_3d,
        use_gk,
        pgk_P,
        pk_P,
        pq_P,
        pdo_P,
        pw_P,
        paqk_P,
    ):
        """Own TMA G2S loads and produce the recurrence operand pipelines."""
        cute.arch.setmaxregister_decrease(self.aux_regs)

        tile_idx = Int32(0)
        has_work = tile_idx < n_iters

        while has_work:
            w_idx = bx + tile_idx * gdx
            v_tile, h_idx, seq_idx = self._decode_work(w_idx)
            if cutlass.const_expr(self.varlen):
                bos = cu_seqlens[seq_idx]
                sequence_length = cu_seqlens[seq_idx + 1] - bos
                num_chunks = (sequence_length + self.BT - 1) // self.BT
                db = Int32(0)
                k_desc = cute.domain_offset((bos, 0, (0, 0)), desc_k)
                q_desc = cute.domain_offset((0, bos, (0, 0)), desc_q)
                do_desc = cute.domain_offset((0, bos, (0, 0)), desc_do)
                w_desc = cute.domain_offset((0, bos, (0, 0)), desc_w)
                aqk_desc = cute.domain_offset((0, bos, (0, 0)), desc_aqk)
                gk_desc = cute.domain_offset((0, bos, (0, 0)), desc_gk)
            else:
                sequence_length = tokens
                num_chunks = dense_chunks
                db = seq_idx
                k_desc = desc_k
                q_desc = desc_q
                do_desc = desc_do
                w_desc = desc_w
                aqk_desc = desc_aqk
                gk_desc = desc_gk

            # Partition TMA operands
            tKs, tKg = self._part_a(atom_k, k_desc, sK, self.dv_tile, mma_dv, db, h_idx)
            tQs, tQg = self._part_a(atom_q, q_desc, sQ, self.qdo_tile, mma_qdo, db, h_idx)
            tDos, tDog = self._part_b(
                atom_do,
                do_desc,
                sDo,
                self.qdo_tile,
                mma_qdo,
                db,
                h_idx,
            )
            tWs, tWg = self._part_a(atom_w, w_desc, sW, self.wdv_tile, mma_wdv, db, h_idx)

            tAqks, tAqkg = self._part_a(
                atom_aqk,
                aqk_desc,
                sAqk,
                (self.BT, self.BV, self.BT),
                mma_aqdo,
                db,
                h_idx,
            )

            gGK_l = gk_desc[None, None, (h_idx, db)]
            sSGK, gSGK = self._part_epi(atom_gk, gGK_l, (self.BK, 1), gk_3d)

            # ---------- Chunk loop (reverse TMA loads) ----------
            for ct in cutlass.range(0, num_chunks, unroll=0):
                rev_ct = num_chunks - 1 - ct

                # gk load FIRST, clamping the tail to its final valid token.
                if use_gk:
                    gk_t = rev_ct * self.BT + self.BT - 1
                    remaining = sequence_length - rev_ct * self.BT
                    if remaining < self.BT:
                        gk_t = sequence_length - 1
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
                do_handle = pdo_P.acquire_and_advance()
                cute.copy(
                    atom=atom_do,
                    src=tDog[None, v_tile, rev_ct],
                    dst=tDos[None, do_handle.index],
                    tma_bar_ptr=do_handle.barrier,
                )
                # w^T load (A-operand for MMA3)
                wh = pw_P.acquire_and_advance()
                cute.copy(
                    atom=atom_w,
                    src=tWg[None, 0, rev_ct],
                    dst=tWs[None, wh.index],
                    tma_bar_ptr=wh.barrier,
                )
                aqk_handle = paqk_P.acquire_and_advance()
                cute.copy(
                    atom=atom_aqk,
                    src=tAqkg[None, 0, rev_ct],
                    dst=tAqks[None, aqk_handle.index],
                    tma_bar_ptr=aqk_handle.barrier,
                )

            tile_idx = tile_idx + 1
            has_work = tile_idx < n_iters

    @cute.jit
    def run_gate(
        self,
        bx,
        gdx,
        n_iters,
        dense_chunks,
        cu_seqlens,
        use_gk,
        gk_buf,
        gk_exp_buf,
        pgk_C,
        pgk_rdy_P,
    ):
        """Consume gk TMA stages and produce exp2 gate stages for the state warps."""
        cute.arch.setmaxregister_decrease(self.aux_regs)
        tid, _, _ = cute.arch.thread_idx()
        gk_tid = tid - WarpRole.GATE * self.WARP_SZ

        tile_idx = Int32(0)
        has_work = tile_idx < n_iters

        while has_work:
            work_index = bx + tile_idx * gdx
            _value_tile, _head_idx, sequence_idx = self._decode_work(work_index)
            if cutlass.const_expr(self.varlen):
                num_chunks = (
                    cu_seqlens[sequence_idx + 1] - cu_seqlens[sequence_idx] + self.BT - 1
                ) // self.BT
            else:
                num_chunks = dense_chunks
            for _ct in cutlass.range(0, num_chunks, unroll=0):
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

    @cute.jit
    def run_mma(
        self,
        bx,
        gdx,
        n_iters,
        dense_chunks,
        cu_seqlens,
        sQ,
        sDo,
        sW,
        sAqk,
        mma_dv,
        mma_qdo,
        mma_aqdo,
        mma_wdv,
        t_dv_acc,
        t_k_a,
        t_dhb_b,
        t_qdo_acc,
        t_aqk_a,
        t_do_aq_b,
        t_q_a,
        t_do_b,
        t_wdv_acc,
        t_w_a,
        t_dv2b_b,
        pdhb_C,
        pk_C,
        pdv_P,
        pq_C,
        pdo_C,
        pqdo_P,
        paqk_C,
        pdv2b_C,
        pw_C,
        pwdv_P,
    ):
        """Consume operands, neutralize packed tails, and produce TMEM results."""
        cute.arch.setmaxregister_decrease(self.aux_regs)
        tid, _, _ = cute.arch.thread_idx()
        mma_tid = tid - WarpRole.MMA * self.WARP_SZ

        tile_idx = Int32(0)
        has_work = tile_idx < n_iters

        while has_work:
            work_index = bx + tile_idx * gdx
            _value_tile, _head_idx, sequence_idx = self._decode_work(work_index)
            if cutlass.const_expr(self.varlen):
                sequence_length = cu_seqlens[sequence_idx + 1] - cu_seqlens[sequence_idx]
                num_chunks = (sequence_length + self.BT - 1) // self.BT
                tail_valid_rows = sequence_length % self.BT
            else:
                num_chunks = dense_chunks
                tail_valid_rows = Int32(0)
            for ct in cutlass.range(0, num_chunks, unroll=0):
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
                kh.release()
                dhbh.release()

                # --- MMA2: q^T(SMEM A) × do(SMEM B) → qdo_acc(TMEM) ---
                qh = pq_C.wait_and_advance()
                doh = pdo_C.wait_and_advance()
                qdod = pqdo_P.acquire_and_advance()
                if cutlass.const_expr(self.varlen):  # noqa: SIM102
                    if ct == 0 and tail_valid_rows != 0:
                        self._neutralize_do_reduction_rows(
                            sDo,
                            doh.index,
                            tail_valid_rows,
                            mma_tid,
                        )

                # Aqk^T × do accumulates into MMA1 before releasing dv.
                aqkh = paqk_C.wait_and_advance()
                if cutlass.const_expr(self.varlen):  # noqa: SIM102
                    if ct == 0 and tail_valid_rows != 0:
                        self._neutralize_aqk_reduction_rows(
                            sAqk, aqkh.index, tail_valid_rows, mma_tid
                        )
                for kp in cutlass.range(cute.size(t_aqk_a, mode=[2]), unroll_full=True):
                    mma_aqdo.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
                    cute.gemm(
                        mma_aqdo,
                        t_dv_acc[None, None, None, dvd.index],
                        t_aqk_a[None, None, kp, aqkh.index],
                        t_do_aq_b[None, None, kp, doh.index],
                        t_dv_acc[None, None, None, dvd.index],
                    )
                dvd.commit()
                aqkh.release()

                # --- MMA2: q^T × do → qdo ---
                if cutlass.const_expr(self.varlen):  # noqa: SIM102
                    if ct == 0 and tail_valid_rows != 0:
                        self._neutralize_reduction_rows(sQ, qh.index, tail_valid_rows, mma_tid)
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

    # ------------------------------------------------------------------
    # Device kernel: warp-specialized backward recurrence
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mmas: Mmas,
        tma: TmaOps,
        g_dht: cute.Tensor,
        g_dh0_t: cute.Tensor,
        g_dv2: cute.Tensor,
        copy_dv2_tiled: cute.TiledCopy,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        smem_layouts: SmemLayouts,
        problem_shape: tuple[Int32, Int32, Int32, Int32, Int32],
        scale: Float32,
        use_gk: Int32,
        use_dht: Int32,
        use_dh0: Int32,
    ):
        mma_dv, mma_qdo, mma_aqdo, mma_wdv = mmas
        atom_k, desc_k = tma.k
        atom_q, desc_q = tma.q
        atom_do, desc_do = tma.do
        atom_w, desc_w = tma.w
        atom_aqk, desc_aqk = tma.aqk
        atom_dhst, desc_dhst = tma.dh_store
        atom_dv2st, desc_dv2st = tma.dv2_store
        atom_gk, desc_gk = tma.gk
        (
            s_k_staged,
            s_dhb_staged,
            s_dhb_store_staged,
            s_q_staged,
            s_do_staged,
            s_w_staged,
            s_dv2b_staged,
            s_dv2b_store_staged,
            s_aqk_staged,
            s_dh_epi_staged,
            s_dh_r2s_staged,
            s_dv2_epi_staged,
            s_dv2_r2s_staged,
        ) = smem_layouts
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch TMA descriptors (load warp)
        if warp_id == WarpRole.LOAD:
            cpasync.prefetch_descriptor(atom_k)
            cpasync.prefetch_descriptor(atom_q)
            cpasync.prefetch_descriptor(atom_do)
            cpasync.prefetch_descriptor(atom_w)
            cpasync.prefetch_descriptor(atom_aqk)
            cpasync.prefetch_descriptor(atom_gk)

        # SMEM allocation
        sa = utils.SmemAllocator()
        sm = sa.allocate(self.shared_type)

        gk_exp_buf = sm.sGK_exp.get_tensor(cute.make_layout((self.BK, self.gk_depth)))
        gk_3d = sm.sGK.get_tensor(
            cute.make_layout(
                (self.BK, 1, self.gk_depth),
                stride=(1, self.BK, self.BK),
            )
        )
        gk_buf = gk_3d[(None, 0, None)]

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

        pdo_P, pdo_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.do_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.do_bytes,
            barrier_storage=sm.bar_do.data_ptr(),
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

        paqk_P, paqk_C = pipeline.PipelineTmaUmma.create(
            num_stages=self.aqk_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.aqk_bytes,
            barrier_storage=sm.bar_aqk.data_ptr(),
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
            allocator_warp_id=WarpRole.LOAD,
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
        sW = sm.sW.get_tensor(s_w_staged.outer, swizzle=s_w_staged.inner)
        sDv2b = sm.sDv2b.get_tensor(s_dv2b_staged.outer, swizzle=s_dv2b_staged.inner)
        sDv2b_store = sm.sDv2b.get_tensor(
            s_dv2b_store_staged.outer, swizzle=s_dv2b_store_staged.inner
        )
        sAqk = sm.sAqk.get_tensor(s_aqk_staged.outer, swizzle=s_aqk_staged.inner)
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

        # Aqk^T (A) × do (B) accumulates into MMA1's dv tile.
        t_aqk_a = mma_aqdo.make_fragment_A(sAqk)
        t_do_aq_b = mma_aqdo.make_fragment_B(sDo)

        # MMA3: w (A) × dv2 (B) → wdv_acc
        t_w_a = mma_wdv.make_fragment_A(sW)
        t_dv2b_b = mma_wdv.make_fragment_B(sDv2b)
        wdv_sh = mma_wdv.partition_shape_C(self.wdv_tile[:2])
        wdv_fk = mma_wdv.make_fragment_C(cute.append(wdv_sh, self.wdv_acc_depth))
        t_wdv_acc = cute.make_tensor(tp + self.tm_wdv, wdv_fk.layout)

        # #
        # Block indices
        #
        B, T, _H, _K, _V = problem_shape
        dense_chunks = (T + self.BT - 1) // self.BT

        # Persistent work is routed by (sequence, head, value tile).
        bx = cute.arch.block_idx()[0]
        gdx = cute.arch.grid_dim()[0]
        value_tiles = self.head_v // self.BV
        sequence_extent = B
        if cutlass.const_expr(self.bound_sequence_extent):
            tid, _, _ = cute.arch.thread_idx()
            if tid % self.WARP_SZ == 0:
                sequence_extent = load_ragged_sequence_extent(cu_seqlens)
            sequence_extent = Int32(cute.arch.shuffle_sync(sequence_extent, 0))
        work_tiles = value_tiles * self.num_heads * sequence_extent
        n_iters = (work_tiles - bx + gdx - 1) // gdx

        # ///////////////////////////////////////////////////////////////////////////////
        #  CUDA CORE (warps 0-3) — dh state owner
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_id in self.CUDA_WARP_IDS:
            self.run_state(
                bx=bx,
                gdx=gdx,
                n_iters=n_iters,
                tokens=T,
                dense_chunks=dense_chunks,
                cu_seqlens=cu_seqlens,
                g_dht=g_dht,
                g_dh0_t=g_dh0_t,
                t_dv_acc=t_dv_acc,
                t_qdo_acc=t_qdo_acc,
                t_wdv_acc=t_wdv_acc,
                sDhb_store=sDhb_store,
                sDhEpi_store=sDhEpi_store,
                sDv2b_store=sDv2b_store,
                sDv2Epi_store=sDv2Epi_store,
                pdhb_P=pdhb_P,
                pdh_epi_P=pdh_epi_P,
                pdv_C=pdv_C,
                pdv2b_P=pdv2b_P,
                pdv2_epi_P=pdv2_epi_P,
                pgk_rdy_C=pgk_rdy_C,
                pqdo_C=pqdo_C,
                pwdv_C=pwdv_C,
                gk_exp_buf=gk_exp_buf,
                use_gk=use_gk,
                use_dht=use_dht,
                use_dh0=use_dh0,
                scale=scale,
            )
        elif warp_id == WarpRole.LOAD:
            self.run_load(
                bx=bx,
                gdx=gdx,
                n_iters=n_iters,
                tokens=T,
                dense_chunks=dense_chunks,
                cu_seqlens=cu_seqlens,
                mma_dv=mma_dv,
                mma_qdo=mma_qdo,
                mma_aqdo=mma_aqdo,
                mma_wdv=mma_wdv,
                atom_k=atom_k,
                desc_k=desc_k,
                atom_q=atom_q,
                desc_q=desc_q,
                atom_do=atom_do,
                desc_do=desc_do,
                atom_w=atom_w,
                desc_w=desc_w,
                atom_aqk=atom_aqk,
                desc_aqk=desc_aqk,
                atom_gk=atom_gk,
                desc_gk=desc_gk,
                sK=sK,
                sQ=sQ,
                sDo=sDo,
                sW=sW,
                sAqk=sAqk,
                gk_3d=gk_3d,
                use_gk=use_gk,
                pgk_P=pgk_P,
                pk_P=pk_P,
                pq_P=pq_P,
                pdo_P=pdo_P,
                pw_P=pw_P,
                paqk_P=paqk_P,
            )
        elif warp_id == WarpRole.GATE:
            self.run_gate(
                bx=bx,
                gdx=gdx,
                n_iters=n_iters,
                dense_chunks=dense_chunks,
                cu_seqlens=cu_seqlens,
                use_gk=use_gk,
                gk_buf=gk_buf,
                gk_exp_buf=gk_exp_buf,
                pgk_C=pgk_C,
                pgk_rdy_P=pgk_rdy_P,
            )
        elif warp_id == WarpRole.MMA:
            self.run_mma(
                bx=bx,
                gdx=gdx,
                n_iters=n_iters,
                dense_chunks=dense_chunks,
                cu_seqlens=cu_seqlens,
                sQ=sQ,
                sDo=sDo,
                sW=sW,
                sAqk=sAqk,
                mma_dv=mma_dv,
                mma_qdo=mma_qdo,
                mma_aqdo=mma_aqdo,
                mma_wdv=mma_wdv,
                t_dv_acc=t_dv_acc,
                t_k_a=t_k_a,
                t_dhb_b=t_dhb_b,
                t_qdo_acc=t_qdo_acc,
                t_aqk_a=t_aqk_a,
                t_do_aq_b=t_do_aq_b,
                t_q_a=t_q_a,
                t_do_b=t_do_b,
                t_wdv_acc=t_wdv_acc,
                t_w_a=t_w_a,
                t_dv2b_b=t_dv2b_b,
                pdhb_C=pdhb_C,
                pk_C=pk_C,
                pdv_P=pdv_P,
                pq_C=pq_C,
                pdo_C=pdo_C,
                pqdo_P=pqdo_P,
                paqk_C=paqk_C,
                pdv2b_C=pdv2b_C,
                pw_C=pw_C,
                pwdv_P=pwdv_P,
            )
        elif warp_id == WarpRole.STORE:
            self.run_store(
                bx=bx,
                gdx=gdx,
                n_iters=n_iters,
                tokens=T,
                dense_chunks=dense_chunks,
                atom_dhst=atom_dhst,
                desc_dhst=desc_dhst,
                atom_dv2st=atom_dv2st,
                desc_dv2st=desc_dv2st,
                copy_dv2_tiled=copy_dv2_tiled,
                g_dv2=g_dv2,
                cu_seqlens=cu_seqlens,
                chunk_offsets=chunk_offsets,
                sDhEpi=sDhEpi,
                sDv2Epi=sDv2Epi,
                pdh_epi_C=pdh_epi_C,
                pdv2_epi_C=pdv2_epi_C,
            )

        # TMEM teardown (all warps)
        tmem.relinquish_alloc_permit()
        self.tmem_free_bar.arrive_and_wait()
        tmem.free(tp)

    # ------------------------------------------------------------------
    # SMEM and TMA helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _neutralize_do_reduction_rows(self, smem, stage, valid_rows, tid):
        """Zero invalid token rows in the direct dO MMA B-operand stage."""
        value_atom = cute.size(smem, mode=[0, 0])
        token_atom = cute.size(smem, mode=[0, 1])
        assert value_atom == self.BV, "value atom must span BV"
        assert cute.size(smem, mode=[1]) == 1, "expected a single rest-N mode"
        assert token_atom * cute.size(smem, mode=[2]) == self.BT, "token modes must tile BT"
        for token in cutlass.range(valid_rows, self.BT, unroll=1):
            if tid < self.BV:
                smem[((tid, token % token_atom), 0, token // token_atom, stage)] = self.io_type(
                    0.0
                )
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _neutralize_aqk_reduction_rows(self, smem, stage, valid_rows, tid):
        """Zero invalid query columns in the 64x64 Aqk operand layout."""
        row_atom = cute.size(smem, mode=[0, 0])
        token_atom = cute.size(smem, mode=[0, 1])
        assert cute.size(smem, mode=[1]) == 1, "expected a single rest-M mode"
        assert row_atom == self.BT
        assert token_atom * cute.size(smem, mode=[2]) == self.BT
        for token in cutlass.range(valid_rows, self.BT, unroll=1):
            for row_block in cutlass.range_constexpr(self.BT // self.WARP_SZ):
                row = tid + row_block * self.WARP_SZ
                smem[((row, token % token_atom), 0, token // token_atom, stage)] = self.io_type(
                    0.0
                )
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _neutralize_reduction_rows(self, smem, stage, valid_rows, tid):
        """Zero a ragged tail's invalid reduction columns before UMMA reads it."""
        row_atom = cute.size(smem, mode=[0, 0, 0])
        token_atom = cute.size(smem, mode=[0, 1])
        assert cute.size(smem, mode=[1]) == 1, "expected a single rest-M mode"
        assert row_atom * cute.size(smem, mode=[0, 0, 1]) == self.BK
        assert token_atom * cute.size(smem, mode=[2]) == self.BT
        for token in cutlass.range(valid_rows, self.BT, unroll=1):
            for row_block in cutlass.range_constexpr(self.BK // self.WARP_SZ):
                row = tid + row_block * self.WARP_SZ
                coordinate = (
                    ((row % row_atom, row // row_atom), token % token_atom),
                    0,
                    token // token_atom,
                    stage,
                )
                smem[coordinate] = self.io_type(0.0)
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

    def _launch_grid(self):
        # Target metadata is inherited by forked compile workers; do not query CUDA here.
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA compilation requires a CUDA target with an SM count")
        return (sm_count, 1, 1)


# ============================================================================
# Compile cache + TVM-FFI
# ============================================================================


def make_fake_tensor(dtype, shape):
    """Create one aligned compact tensor for a TVM-FFI compile signature."""
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=128,
    )


def make_state_signature_tensor(dtype, shape, use_int64_offsets, dynamic_strides):
    """Create the compact fast-path or dynamic-stride state signature."""
    if not dynamic_strides:
        return make_fake_tensor(dtype, shape)
    return make_fake_strided_tensor(
        dtype,
        shape,
        assumed_align=dtype.width // 8,
        use_int64_strides=use_int64_offsets,
    )


def supports_state_layout(state: torch.Tensor) -> bool:
    """Return whether raw state loads can address this nonnegative-strided layout."""
    return all(stride >= 0 for stride in state.stride()) and tensor_supports_contiguous_dim(
        state,
        alignment_bytes=state.element_size(),
    )


def requires_dynamic_state_layout(*states: torch.Tensor) -> bool:
    """Keep the 128-byte-aligned compact signature only for its original fast path."""
    return any(
        not state.is_contiguous() or not tensor_supports_contiguous_dim(state, alignment_bytes=128)
        for state in states
    )


@jit_cache
def _compile_delta_h_bwd(H, bv, io_type, use_int64_offsets, dynamic_state_layout):
    """Compile one dense BlackwellDeltaHBwd variant."""
    K = V = 128
    chunk_size = 64
    kern = BlackwellDeltaHBwd(
        head_bv=bv,
        num_heads=H,
        io_type=io_type,
        use_int64_offsets=use_int64_offsets,
        dynamic_state_layout=dynamic_state_layout,
    )
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    sa, sb, snt, sns, sn = (sym_int() for _ in range(5))

    qf, kf, wf = (make_fake_tensor(io_type, (sa, sb, H, K)) for _ in range(3))
    dof = make_fake_tensor(io_type, (sa, sb, H, V))
    aqkf = make_fake_tensor(io_type, (sa, sb, H, chunk_size))
    gkf = make_fake_tensor(cutlass.Float32, (sa, sb, H, K))
    dhtf = make_state_signature_tensor(
        cutlass.Float32,
        (sns, H, V, K),
        use_int64_offsets,
        dynamic_state_layout,
    )
    dh0f = make_state_signature_tensor(
        cutlass.Float32,
        (sns, H, V, K),
        use_int64_offsets,
        dynamic_state_layout,
    )
    dhof = make_fake_tensor(io_type, (sa, snt, H, K, V))
    dv2f = make_fake_tensor(io_type, (sa, sb, H, V))
    cuf = make_fake_tensor(cutlass.Int32, (sn,))
    cof = make_fake_tensor(cutlass.Int32, (sn,))

    return compile_tvm_ffi(
        kern,
        qf,
        kf,
        wf,
        dof,
        aqkf,
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


@jit_cache
def _compile_delta_h_bwd_packed(
    heads: int,
    bv: int,
    io_type: type[cutlass.Numeric],
    use_int64_offsets: bool,
    bound_sequence_extent: bool,
    dynamic_state_layout: bool,
):
    """Compile one packed fused specialization with a width-matched TVM ABI."""
    key_dim = value_dim = 128
    chunk_size = 64
    kernel = BlackwellDeltaHBwd(
        num_heads=heads,
        head_bv=bv,
        io_type=io_type,
        varlen=True,
        use_int64_offsets=use_int64_offsets,
        bound_sequence_extent=bound_sequence_extent,
        dynamic_state_layout=dynamic_state_layout,
    )
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    tokens, chunks, sequences, metadata_entries = (sym_int() for _ in range(4))

    def token_tensor(dtype, width):
        return make_fake_tensor(dtype, (tokens, heads, width))

    state_shape = (sequences, heads, value_dim, key_dim)
    dht_state = make_state_signature_tensor(
        cutlass.Float32,
        state_shape,
        use_int64_offsets,
        dynamic_state_layout,
    )
    dh0_state = make_state_signature_tensor(
        cutlass.Float32,
        state_shape,
        use_int64_offsets,
        dynamic_state_layout,
    )
    chunk_state = make_fake_tensor(io_type, (chunks, heads, key_dim, value_dim))
    metadata = make_fake_tensor(cutlass.Int32, (metadata_entries,))
    return compile_tvm_ffi(
        kernel,
        token_tensor(io_type, key_dim),
        token_tensor(io_type, key_dim),
        token_tensor(io_type, key_dim),
        token_tensor(io_type, value_dim),
        token_tensor(io_type, chunk_size),
        token_tensor(cutlass.Float32, key_dim),
        dht_state,
        dh0_state,
        chunk_state,
        token_tensor(io_type, value_dim),
        metadata,
        metadata,
        (Int32(1), Int32(1), Int32(heads), Int32(key_dim), Int32(value_dim)),
        Float32(1.0),
        Int32(0),
        Int32(0),
        Int32(0),
    )


# ============================================================================
# Public API wrapper
# ============================================================================


def _blackwell_delta_h_bwd_dhu_dv_fused_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    aqk: torch.Tensor,
    metadata: RaggedChunkMetadata,
    bv: int,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float = 1.0,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run packed delta-H with fused intra-chunk dV.

    Chunk-state slots beyond ``metadata.chunk_offsets[-1]`` and inactive token rows are undefined.
    """
    batch, tokens, heads, key_dim = q.shape
    value_dim = do.shape[-1]
    metadata.validate_chunk_size(chunk_size)
    if batch != 1 or key_dim != 128 or value_dim != 128 or chunk_size != 64:
        raise ValueError("packed delta-H+dV fusion requires B=1, K=V=128, and chunk_size=64")
    expected_key = (1, tokens, heads, key_dim)
    expected_value = (1, tokens, heads, value_dim)
    if k.shape != expected_key or w.shape != expected_key:
        raise ValueError(f"k and w must have shape {expected_key}")
    if do.shape != expected_value:
        raise ValueError(f"do must have shape {expected_value}")
    expected_aqk = (1, tokens, heads, chunk_size)
    if aqk.shape != expected_aqk:
        raise ValueError(f"Aqk must have shape {expected_aqk}")
    if gk is not None and gk.shape != q.shape:
        raise ValueError("gk must match q")
    tensor_inputs = (q, k, w, do, aqk, *((gk,) if gk is not None else ()))
    if any(tensor.device != q.device for tensor in tensor_inputs):
        raise ValueError("packed delta-H+dV inputs must share q.device")
    if any(not tensor.is_contiguous() for tensor in tensor_inputs):
        raise ValueError("packed delta-H+dV inputs must be contiguous")
    if q.dtype not in _IO_TYPES or any(tensor.dtype != q.dtype for tensor in (k, w, do, aqk)):
        raise TypeError("q, k, w, do, and Aqk must share dtype float16 or bfloat16")
    io_type = _IO_TYPES[q.dtype]
    if gk is not None and gk.dtype != torch.float32:
        raise TypeError("gk must be float32")
    sequences = metadata.cu_seqlens.shape[0] - 1
    expected_state = (sequences, heads, value_dim, key_dim)
    for name, state in (("h0", h0), ("dht", dht)):
        if state is not None and state.shape != expected_state:
            raise ValueError(f"{name} must have shape {expected_state}")
        if state is not None and (
            state.dtype != torch.float32
            or state.device != q.device
            or not supports_state_layout(state)
        ):
            raise TypeError(f"{name} must be float32 with a contiguous key mode on q.device")
    if bv not in (16, 32):
        raise ValueError(f"bv must be 16 or 32, got {bv}")

    chunk_state = q.new_empty(1, metadata.capacity, heads, key_dim, value_dim)
    d_initial_state = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    d_value = torch.empty_like(do)
    if active_fake_mode() is not None:
        return chunk_state, d_initial_state, d_value

    device = q.device
    gate_kernel = (
        gk[0]
        if gk is not None
        else torch.empty((tokens, heads, key_dim), dtype=torch.float32, device=device)
    )
    final_state_kernel = (
        dht if dht is not None else torch.empty(expected_state, dtype=torch.float32, device=device)
    )
    initial_state_gradient_kernel = (
        d_initial_state
        if d_initial_state is not None
        else torch.empty(expected_state, dtype=torch.float32, device=device)
    )
    kernel_tensors = (
        q[0],
        k[0],
        w[0],
        do[0],
        aqk[0],
        gate_kernel,
        final_state_kernel,
        initial_state_gradient_kernel,
        chunk_state[0],
        d_value[0],
    )
    use_int64_offsets = requires_int64_abi(
        *kernel_tensors,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
    )
    compiled = _compile_delta_h_bwd_packed(
        heads,
        bv,
        io_type,
        use_int64_offsets,
        should_bound_sequence_extent(
            tokens,
            sequences,
            heads,
            chunk_size,
            h0 is not None,
        ),
        requires_dynamic_state_layout(final_state_kernel, initial_state_gradient_kernel),
    )
    compiled(
        *kernel_tensors,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        (Int32(sequences), Int32(tokens), Int32(heads), Int32(key_dim), Int32(value_dim)),
        Float32(scale),
        Int32(gk is not None),
        Int32(dht is not None),
        Int32(h0 is not None),
    )
    return chunk_state, d_initial_state, d_value


def blackwell_delta_h_bwd_dhu_dv_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    aqk: torch.Tensor,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float = 1.0,
    chunk_size: int = 64,
    bv: int = 16,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the dense B=1 CuTeDSL SM100 delta-H backward leaf."""
    B, T, H, K = q.shape
    V = do.shape[-1]
    BT = chunk_size
    dev = q.device

    if B != 1:
        raise ValueError(f"the dense dv-fusion kernel requires B=1, got {B}")
    if chunk_size != 64:
        raise ValueError(f"the dense dv-fusion kernel requires chunk_size=64, got {chunk_size}")
    if T % chunk_size:
        raise ValueError(
            f"the dense dv-fusion kernel requires complete chunks, got T={T}, chunk_size={chunk_size}"
        )
    if (K, V) != (128, 128):
        raise ValueError(f"the dense dv-fusion kernel requires K=V=128, got K={K}, V={V}")
    expected_aqk = (B, T, H, BT)
    if aqk.shape != expected_aqk:
        raise ValueError(f"the dense delta-H kernel requires Aqk with shape {expected_aqk}")
    expected_state = (B, H, V, K)
    for name, state in (("h0", h0), ("dht", dht)):
        if state is not None and state.shape != expected_state:
            raise ValueError(f"{name} must have shape {expected_state}")
        if state is not None and (
            state.dtype != torch.float32
            or state.device != q.device
            or not supports_state_layout(state)
        ):
            raise TypeError(f"{name} must be float32 with a contiguous key mode on q.device")

    dh_out = q.new_empty(B, T // BT, H, K, V)
    dh0_out = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(do)

    if active_fake_mode() is not None:
        return dh_out, dh0_out, dv2

    state_shape = (B, H, V, K)
    gk_k = gk if gk is not None else torch.empty((B, T, H, K), dtype=torch.float32, device=dev)
    dht_k = dht if dht is not None else torch.empty(state_shape, dtype=torch.float32, device=dev)
    dh0_k = (
        dh0_out
        if dh0_out is not None
        else torch.empty(state_shape, dtype=torch.float32, device=dev)
    )

    if q.dtype not in _IO_TYPES or any(tensor.dtype != q.dtype for tensor in (k, w, do, aqk)):
        raise TypeError("q, k, w, do, and Aqk must share dtype float16 or bfloat16")
    io_type = _IO_TYPES[q.dtype]
    use_int64_offsets = requires_int64_abi(
        q,
        k,
        w,
        do,
        aqk,
        gk_k,
        dht_k,
        dh0_k,
        dh_out,
        dv2,
    )
    fn = _compile_delta_h_bwd(
        H,
        bv,
        io_type,
        use_int64_offsets,
        requires_dynamic_state_layout(dht_k, dh0_k),
    )
    dummy_metadata = torch.empty(2, dtype=torch.int32, device=dev)
    fn(
        q,
        k,
        w,
        do,
        aqk,
        gk_k,
        dht_k,
        dh0_k,
        dh_out,
        dv2,
        dummy_metadata,
        dummy_metadata,
        (Int32(B), Int32(T), Int32(H), Int32(K), Int32(V)),
        Float32(scale),
        Int32(gk is not None),
        Int32(dht is not None),
        Int32(h0 is not None),
    )

    return dh_out, dh0_out, dv2


def blackwell_delta_h_bwd_dhu_dv_fused_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    aqk: torch.Tensor,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float = 1.0,
    chunk_size: int = 64,
    metadata: RaggedChunkMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the dense or packed dv-fused kernel with automatic BV selection."""
    batch, _tokens, heads, _head_dim = q.shape
    value_dim = do.shape[-1]
    logical_batch = batch if metadata is None else metadata.cu_seqlens.shape[0] - 1
    bv = select_delta_h_bv(value_dim, heads, logical_batch, q.device)
    if metadata is not None:
        return _blackwell_delta_h_bwd_dhu_dv_fused_packed(
            q,
            k,
            w,
            do,
            aqk,
            metadata,
            gk=gk,
            h0=h0,
            dht=dht,
            scale=scale,
            chunk_size=chunk_size,
            bv=bv,
        )
    return blackwell_delta_h_bwd_dhu_dv_fused(
        q,
        k,
        w,
        do,
        aqk,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        chunk_size=chunk_size,
        bv=bv,
    )


__all__ = [
    "blackwell_delta_h_bwd_dhu_dv_fused",
    "blackwell_delta_h_bwd_dhu_dv_fused_dispatch",
]
