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

MMA operations per chunk:
  MMA1: k @ dh       → dv    (BT=64,  BV, K=128)
  MMA2: q^T @ dO     → qdo   (BK=128, BV, K=64)
  MMA3: w @ dv       → wdv   (BK=128, BV, K=64)
  MMA4: Aqk^T @ dO   → dv    (BT=64,  BV, K=64)

A warpdrive kernel of 8 warps, grid capped at one CTA per SM. The state warps keep dh in FP32
registers across all chunks of an item; SS-mode MMAs read both operands from SMEM and accumulate in
TMEM. One work item is one (value tile, head, sequence), run in reverse chunk order::

    load (4) --k, q, dO, w, Aqk: TMA rings---------------> mma (6)
    load (4) --gk: TMA ring--> gate (5) --gk_rdy--> state (0-3)
    mma (6)  --dv, qdo, wdv: TMEM--------------------------> state (0-3)
    state    --dhb, dv2b: MMA B operands in SMEM---------> mma (6)
    state    --dh_epi, dv2_epi: staged dh and dV---------> store (7) --TMA--> dh, dV

Every channel carries one epoch per chunk of an item, except gk (per-key gate only) and
gk_rdy, which carry none when the gate is disabled.
"""

from dataclasses import dataclass
from typing import NamedTuple

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Float32, Int32, Int64
from torch._guards import active_fake_mode

from attn_gym._backends.cute import get_device_properties, tensor_supports_contiguous_dim
from attn_gym._backends.cute.utils import cute_dtype, requires_int64_abi
from attn_gym.linear._delta_rule.triton.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.constants import is_sm100_kda_capability
from attn_gym.utils import cdiv
from attn_gym.warpdrive import (
    Consumer,
    Local,
    Producer,
    Ring,
    RoleConfig,
    SmemPlan,
    StageView,
    StaticPersistent,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WarpSpecialized,
    WorkCursor,
    WorkMap,
    WorkStream,
    bulk_commit_group,
    bulk_wait_group,
    role,
)
from attn_gym.warpdrive.cache import (
    compile_tvm_ffi,
    get_compile_target,
    jit_cache,
    make_fake_strided_tensor,
)
from attn_gym.warpdrive.compat import LayoutEnum

_MIN_SEQUENCE_EXTENT_SEQUENCES = 32
_MIN_SEQUENCE_EXTENT_HEADS = 8
_IO_TYPES = (torch.float16, torch.bfloat16)


def select_delta_h_bv(
    value_dim: int,
    heads: int,
    logical_batch: int,
    device: torch.device,
) -> int:
    """Select the delta-H value tile width for the logical workload."""
    value_tiles = cdiv(value_dim, 16) * heads * logical_batch
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


@dataclass(frozen=True)
class ValueHeadSequence(WorkMap):
    """Work items ``(value tile, head, sequence)``, value tile fastest.

    The space is ``(sequences, *rest)``; ``rest`` passes through for the schedule. Value tiles
    and heads are compile-time, so decoding divides by constants. Neighbouring CTAs take the
    value tiles of one head and sequence, which share their K, Q, W and Aqk tiles in L2.
    """

    value_tiles: int
    heads: int
    words = 3
    decodes_any_index = True

    def bind(self, space):
        sequences, *rest = space
        return (Int32(cutlass.max(sequences, 0)), *rest)

    def count64(self, space):
        return Int64(self.value_tiles * self.heads) * Int64(space[0])

    def decode(self, index, space, state):
        tile = index // self.value_tiles
        return True, (index % self.value_tiles, tile % self.heads, tile // self.heads), state

    def get_key(self) -> str:
        return f"vhs{self.value_tiles}x{self.heads}"

    def describe(self) -> str:
        return f"ValueHeadSequence({self.value_tiles} value tiles x {self.heads} heads)"


@dataclass(frozen=True)
class SequenceBoundedPersistent(StaticPersistent):
    """``StaticPersistent`` over a packed ``ValueHeadSequence`` that can stop at the last sequence
    holding tokens.

    The bound space is ``(sequences, cu_seqlens, bound)``. With ``bound != 0`` (a runtime flag,
    so both modes share one compile) an index whose sequence starts at or past the active token
    count ``cu_seqlens[-1]`` is terminal: that sequence and every later one is empty, and a CUDA
    graph captured for many short documents skips the slots a replay leaves unused instead of
    visiting each of them. Indices are sequence-major and increase within a unit, so the first
    such index ends the unit's stream. Only launch data is read, so every role computes the
    same sequence.
    """

    items_per_sequence: int

    @cute.jit
    def bounded(self, index, problem):
        sequences, cu_seqlens, bound = problem.space
        result = index
        # A uniform runtime branch: the host enables it only for short-document packs, where
        # trimming graph-overcaptured sequences outweighs the two metadata reads per item.
        if bound != 0:  # noqa: SIM102
            # Nested so the metadata read never sees a terminal index.
            if index < problem.total:  # noqa: SIM102
                if cu_seqlens[index // self.items_per_sequence] >= cu_seqlens[sequences]:
                    result = problem.total
        return result

    def first_index(self, problem):
        return self.bounded(problem.first_unit(), problem)

    def next_index(self, index, problem):
        return self.bounded(index + problem.units(), problem)

    def describe(self) -> str:
        return f"SequenceBoundedPersistent({self.items_per_sequence} items per sequence)"


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
    gk: TmaOp | None  # None for a scalar gate, which the gate warp loads directly


class Mmas(NamedTuple):
    """Tensor-core operations owned by the MMA warp."""

    dv: cute.TiledMma
    qdo: cute.TiledMma
    aqdo: cute.TiledMma
    wdv: cute.TiledMma


class SmemLayouts(NamedTuple):
    """Shared-memory layouts of the ``SmemPlan`` buffers, by buffer name."""

    K: cute.ComposedLayout
    Dhb: cute.ComposedLayout
    DhbR2S: cute.ComposedLayout
    Q: cute.ComposedLayout
    Do: cute.ComposedLayout
    W: cute.ComposedLayout
    Dv2b: cute.ComposedLayout
    Dv2bR2S: cute.ComposedLayout
    Aqk: cute.ComposedLayout
    DhEpi: cute.ComposedLayout
    DhR2S: cute.ComposedLayout
    Dv2Epi: cute.ComposedLayout
    Dv2R2S: cute.ComposedLayout
    GK: cute.Layout
    GKExp: cute.Layout


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


@dataclass(frozen=True)
class DeltaHParams:
    """Kernel-prologue values the roles read."""

    mmas: Mmas
    tma: TmaOps
    g_gate: cute.Tensor | None  # scalar gate (T, (H, B)); None for a per-key gate
    g_dht: cute.Tensor
    g_dh0_t: cute.Tensor
    g_dv2: cute.Tensor
    copy_dv2_tiled: cute.TiledCopy
    cu_seqlens: cute.Tensor
    chunk_offsets: cute.Tensor
    sK: cute.Tensor
    sQ: cute.Tensor
    sDo: cute.Tensor
    sW: cute.Tensor
    sAqk: cute.Tensor
    sGK: cute.Tensor  # (BK, 1, stage), the gk TMA destination
    sDhEpi: cute.Tensor
    sDv2Epi: cute.Tensor
    # MMA fragments over the SMEM stages: (MMA, MMA_M, MMA_K, stage)
    t_k_a: cute.Tensor
    t_dhb_b: cute.Tensor
    t_q_a: cute.Tensor
    t_do_b: cute.Tensor
    t_aqk_a: cute.Tensor
    t_do_aq_b: cute.Tensor
    t_w_a: cute.Tensor
    t_dv2b_b: cute.Tensor
    # TMEM accumulator layouts, one stage each
    dv_layout: cute.Layout
    qdo_layout: cute.Layout
    wdv_layout: cute.Layout
    tokens: Int32
    dense_chunks: Int32
    scale: Float32
    use_gk: Int32
    use_dht: Int32
    use_dh0: Int32


AUX_ROLES = ("load", "gate", "mma", "store")  # warps 4-7, one warp each


class BlackwellDeltaHBwd(WarpSpecialized):
    """
    Warp-specialized SM100 kernel for gated delta rule backward dhu recurrence.

    dh state persists in CUDA warp registers (FP32) across all chunks — zero
    GMEM round-trips. SS-mode keeps both operands in SMEM and supports BV=16 or 32.

    MMA operations per chunk:
      MMA1: k @ dh       → dv    (BT=64,  BV, K=128)
      MMA2: q^T @ dO     → qdo   (BK=128, BV, K=64)
      MMA3: w @ dv       → wdv   (BK=128, BV, K=64)
      MMA4: Aqk^T @ dO   → dv    (BT=64,  BV, K=64)

    One work item is one ``(value tile, head, sequence)``; static persistence gives each CTA
    items ``b, b + grid, ...``. Per item every channel carries one epoch per chunk, except
    ``gk`` (per-key gate only) and ``gk_rdy``, which carry none when ``use_gk`` is off.
    """

    def __init__(
        self,
        num_heads: int,
        io_type: type[cutlass.Numeric] = cutlass.BFloat16,
        head_bv: int = 16,
        varlen: bool = False,
        use_int64_offsets: bool = False,
        dynamic_state_layout: bool = False,
        fastmath: bool = False,
        scalar_gate: bool = False,
    ):
        if head_bv not in (16, 32):
            raise ValueError(f"BV must be 16 or 32, got {head_bv}")
        self.head_k = 128
        self.head_v = 128
        self.acc_type = cutlass.Float32
        self.io_type = io_type
        self.num_heads = num_heads
        self.varlen = varlen
        self.use_int64_offsets = use_int64_offsets
        self.dynamic_state_layout = dynamic_state_layout
        self.fastmath = fastmath
        # Scalar gate: gk is one decay per (token, head) instead of one per key dimension.
        self.scalar_gate = scalar_gate

        # Tile dimensions
        self.BT = 64
        self.BK = self.head_k
        self.BV = head_bv  # N-dim for SS-mode MMA (16 or 32)

        # MMA tile shapes (M, N, K)
        # MMA1: k @ dh → dv — k is (BT,BK) K-major, dh is (BK,BV) K-major
        self.dv_tile = (self.BT, self.BV, self.BK)
        # MMA2: q^T @ do → qdo — q^T is (BK,BT) MN-major, do is (BT,BV) MN-major
        self.qdo_tile = (self.BK, self.BV, self.BT)
        # MMA3: w @ dv2 → wdv — w is (BK,BT) MN-major, dv2 is (BT,BV) K-major
        self.wdv_tile = (self.BK, self.BV, self.BT)

        # The BV=32 dh state needs more registers in the state warps. The single-warp roles'
        # budgets are measured, not derived from live state: 40 is slower than these, as are
        # 64 at BV=32 (long dense sequences) and 56 at BV=16 (packs).
        if head_bv > 16:
            aux = {name: RoleConfig(warp, regs=56) for warp, name in enumerate(AUX_ROLES, 4)}
            self.roles = {"state": RoleConfig(range(4), regs=160), **aux}

        # Shallower k ring at BV=32 to fit SMEM.
        k_depth = 6 if head_bv <= 16 else 4
        operand_bytes = io_type.width // 8
        plan = self.smem_plan = SmemPlan()
        k = plan.buffer("K", io_type, align=1024)
        # The state warps write the MMA operands and the TMA-store stages through second,
        # store-friendly layouts of the same bytes.
        dhb = plan.buffer("Dhb", io_type, align=1024)
        dhb_r2s = plan.buffer("DhbR2S", io_type, align=1024, overlaps=dhb)
        q = plan.buffer("Q", io_type, align=1024)
        do = plan.buffer("Do", io_type, align=1024)
        w = plan.buffer("W", io_type, align=1024)
        dv2b = plan.buffer("Dv2b", io_type, align=1024)
        dv2b_r2s = plan.buffer("Dv2bR2S", io_type, align=1024, overlaps=dv2b)
        aqk = plan.buffer("Aqk", io_type, align=1024)
        dh_epi = plan.buffer("DhEpi", io_type, align=1024)
        dh_r2s = plan.buffer("DhR2S", io_type, align=1024, overlaps=dh_epi)
        dv2_epi = plan.buffer("Dv2Epi", io_type, align=1024)
        dv2_r2s = plan.buffer("Dv2R2S", io_type, align=1024, overlaps=dv2_epi)
        gk = plan.buffer("GK", cutlass.Float32, align=128)
        gk_exp = plan.buffer("GKExp", cutlass.Float32, align=128)

        # TMA operand rings, load -> MMA. dO feeds both the dV-intra and the qdo MMA.
        self.k = Ring(k_depth, Tma(self.BT * self.BK * operand_bytes), Umma, buffer=k)
        self.q = Ring(2, Tma(self.BK * self.BT * operand_bytes), Umma, buffer=q)
        self.do = Ring(2, Tma(self.BT * self.BV * operand_bytes), Umma, buffer=do)
        self.w = Ring(2, Tma(self.BK * self.BT * operand_bytes), Umma, buffer=w)
        self.aqk = Ring(2, Tma(self.BT * self.BT * operand_bytes), Umma, buffer=aqk)
        # The chunk's last-token gate, load -> gate warp. The gate warp reads it with ordinary
        # loads, so the derived release fence orders those reads before the next TMA refill.
        self.gk = Ring(2, Tma(self.BK * 4), Warps, buffer=gk)
        # exp2 of the gate, gate warp -> state warps; both sides use ordinary SMEM accesses.
        self.gk_rdy = Ring(2, Threads, Threads, buffer=gk_exp)
        # dh and dv2 as MMA B operands, state -> MMA (derived commit fence: smem_async).
        self.dhb = Ring(1, Threads, Umma, buffer={"mma": dhb, "r2s": dhb_r2s})
        self.dv2b = Ring(1, Threads, Umma, buffer={"mma": dv2b, "r2s": dv2b_r2s})
        # TMEM results, MMA -> state. tcgen05.wait::ld before each release, so the next MMA
        # cannot overwrite columns still being read.
        self.dv = Ring(1, Umma, Threads, release_fence="tmem_load")
        self.qdo = Ring(1, Umma, Threads, release_fence="tmem_load")
        self.wdv = Ring(1, Umma, Threads, release_fence="tmem_load")
        # dh snapshots and dv2, state -> store. The store warp reads them with TMA stores
        # (async proxy), so the state warps fence their ordinary stores before arriving.
        epi = {"commit_fence": "smem_async"}
        self.dh_epi = Ring(2, Threads, Threads, buffer={"tma": dh_epi, "r2s": dh_r2s}, **epi)
        self.dv2_epi = Ring(2, Threads, Threads, buffer={"tma": dv2_epi, "r2s": dv2_r2s}, **epi)
        # Each accumulator is BV FP32 columns, whether M is 64 (dv) or 128.
        self.tmem_plan = (
            TmemPlan().region("dv", self.BV).region("qdo", self.BV).region("wdv", self.BV)
        )
        value_tiles = self.head_v // self.BV
        self.work = WorkStream(
            mapping=ValueHeadSequence(value_tiles, num_heads),
            schedule=SequenceBoundedPersistent(value_tiles * num_heads)
            if varlen
            else StaticPersistent(),
        )

    @cute.jit
    def upcast(self, value):
        """Promote an address operand before its first overflowing multiply."""
        return cutlass.Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    def get_name(self) -> str:
        """Return a stable artifact and profiler name for this specialization."""
        return (
            f"kda_bwd_dhu_dv_fused_vl{int(self.varlen)}_h{self.num_heads}"
            f"_k{self.head_k}_v{self.head_v}_bt{self.BT}_bv{self.BV}_{cute_dtype(self.io_type).name}"
            f"_i64{int(self.use_int64_offsets)}"
            f"_ds{int(self.dynamic_state_layout)}_fm{int(self.fastmath)}"
        )

    @cute.jit
    def sequence(self, params: DeltaHParams, sequence_idx):
        """Return ``(bos, length, num_chunks, batch)`` of one sequence.

        Packed sequences start at ``cu_seqlens`` in batch 0; dense sequences are batches of
        ``tokens`` each. Every role derives its chunk count here, so each channel's producer
        and consumer agree on its epochs.
        """
        if cutlass.const_expr(self.varlen):
            bos = params.cu_seqlens[sequence_idx]
            length = params.cu_seqlens[sequence_idx + 1] - bos
            num_chunks = (length + self.BT - 1) // self.BT
            batch = Int32(0)
        else:
            bos = Int32(0)
            length = params.tokens
            num_chunks = params.dense_chunks
            batch = sequence_idx
        return bos, length, num_chunks, batch

    # ------------------------------------------------------------------
    # Roles
    # ------------------------------------------------------------------

    @role(warps=range(4), regs=128)
    @cute.jit
    def state(
        self,
        params: DeltaHParams,
        *,
        dhb: Producer,
        dh_epi: Producer,
        dv: Consumer,
        dv2b: Producer,
        dv2_epi: Producer,
        gk_rdy: Consumer,
        qdo: Consumer,
        wdv: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        """Own dh registers; consume result/gate stages and produce dh/dV operand stages."""
        local_tid = me.tmem_row
        t_dv_acc = cute.make_tensor(tmem.ptr("dv"), params.dv_layout)
        t_qdo_acc = cute.make_tensor(tmem.ptr("qdo"), params.qdo_layout)
        t_wdv_acc = cute.make_tensor(tmem.ptr("wdv"), params.wdv_layout)

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
            LayoutEnum.COL_MAJOR,
            self.io_type,
            self.acc_type,
            tc_t2r_qdo,
        )
        tc_r2s_dhb = cute.make_tiled_copy_D(r2s_atom_dhb, tc_t2r_qdo)
        thr_r2s_dhb = tc_r2s_dhb.get_slice(local_tid)

        # R2S dh → sDhEpi (ROW_MAJOR, transposed for TMA store)
        r2s_atom_dh_epi = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR,
            self.io_type,
            self.acc_type,
            tc_t2r_qdo,
        )
        tc_r2s_dh_epi = cute.make_tiled_copy_D(r2s_atom_dh_epi, tc_t2r_qdo)
        thr_r2s_dh_epi = tc_r2s_dh_epi.get_slice(local_tid)

        # R2S dv2 → sDv2b (COL_MAJOR, partition matches T2R dv)
        r2s_atom_dv2b = sm100_utils.get_smem_store_op(
            LayoutEnum.COL_MAJOR,
            self.io_type,
            self.acc_type,
            tc_t2r_dv,
        )
        tc_r2s_dv2b = cute.make_tiled_copy_D(r2s_atom_dv2b, tc_t2r_dv)
        thr_r2s_dv2b = tc_r2s_dv2b.get_slice(local_tid)

        # R2S dv2 → sDv2Epi (ROW_MAJOR, transposed for TMA store)
        r2s_atom_dv2_epi = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR,
            self.io_type,
            self.acc_type,
            tc_t2r_dv,
        )
        tc_r2s_dv2_epi = cute.make_tiled_copy_D(r2s_atom_dv2_epi, tc_t2r_dv)
        thr_r2s_dv2_epi = tc_r2s_dv2_epi.get_slice(local_tid)

        while work.valid:
            v_tile, h_idx, seq_idx = work.item
            _, sequence_length, num_chunks, _ = self.sequence(params, seq_idx)
            # Initialize dh from dht or zeros
            if params.use_dht:
                gDht_slice = params.g_dht[None, None, (h_idx, seq_idx)]
                for ei in cutlass.range(cute.size(st), unroll_full=True):
                    kc, vc = coords_kv[ei]
                    st[ei] = gDht_slice[(kc, vc + v_tile * self.BV)]
            else:
                for ei in cutlass.range(cute.size(st), unroll_full=True):
                    st[ei] = Float32(0.0)

            # ========== Chunk loop: reverse order ==========
            for ct in cutlass.range(0, num_chunks, unroll=0):
                # ---- Phase 1: R2S dh→sDhb + sDhEpi (snapshot dh state) ----
                dhb_h = dhb.acquire()
                dh_epi_h = dh_epi.acquire()
                st_bf16 = cute.make_rmem_tensor(st.shape, self.io_type)
                st_bf16.store(st.load().to(self.io_type))

                # R2S dh → sDhb (B-operand for MMA1)
                r2s_src_dhb = tc_r2s_dhb.retile(st_bf16)
                dst_dhb = thr_r2s_dhb.partition_D(dhb_h.view("r2s", write=True))
                cute.copy(tc_r2s_dhb, r2s_src_dhb, dst_dhb)
                dhb_h.commit()

                # R2S dh → sDhEpi (for Store warp TMA S2G)
                r2s_src_dh = tc_r2s_dh_epi.retile(st_bf16)
                dst_dh = thr_r2s_dh_epi.partition_D(dh_epi_h.view("r2s", write=True))
                cute.copy(tc_r2s_dh_epi, r2s_src_dh, dst_dh)
                dh_epi_h.commit()

                # ---- Phase 2: T2R fused dv result ----
                dvh = dv.wait()
                dv_reg = cute.make_rmem_tensor(coords_tv.shape, self.acc_type)
                cute.copy(tc_t2r_dv, p_t_dv[(None, None, None, dvh.index)], dv_reg)
                dvh.release()

                if cutlass.const_expr(self.varlen):
                    valid_rows = sequence_length - (num_chunks - 1 - ct) * self.BT
                    if valid_rows < self.BT:
                        for ei in cutlass.range_constexpr(cute.size(coords_tv)):
                            token_coord, _value_coord = coords_tv[ei]
                            if token_coord >= valid_rows:
                                dv_reg[ei] = Float32(0.0)

                # ---- Phase 3: R2S dv2→sDv2b + sDv2Epi ----
                dv2b_h = dv2b.acquire()
                dv2_epi_h = dv2_epi.acquire()
                dv_bf16 = cute.make_rmem_tensor(dv_reg.shape, self.io_type)
                dv_bf16.store(dv_reg.load().to(self.io_type))

                # R2S dv2 → sDv2b (B-operand for MMA3)
                r2s_src_dv2b = tc_r2s_dv2b.retile(dv_bf16)
                dst_dv2b = thr_r2s_dv2b.partition_D(dv2b_h.view("r2s", write=True))
                cute.copy(tc_r2s_dv2b, r2s_src_dv2b, dst_dv2b)
                dv2b_h.commit()

                # R2S dv2 → sDv2Epi (for Store warp TMA S2G)
                r2s_src_dv2 = tc_r2s_dv2_epi.retile(dv_bf16)
                dst_dv2 = thr_r2s_dv2_epi.partition_D(dv2_epi_h.view("r2s", write=True))
                cute.copy(tc_r2s_dv2_epi, r2s_src_dv2, dst_dv2)
                dv2_epi_h.commit()

                # ---- Phase 4: Decay dh and apply qdo, wdv updates ----
                # Decay: dh *= exp2(gk_last[k])
                if params.use_gk:
                    gate_h = gk_rdy.wait()
                    gate = gate_h.view(read=True)
                    for ei in cutlass.range(cute.size(st), unroll_full=True):
                        kc, vc = coords_kv[ei]
                        st[ei] = st[ei] * gate[kc]
                    gate_h.release()

                # T2R qdo result from MMA2
                qdoh = qdo.wait()
                qdo_reg = cute.make_rmem_tensor(coords_kv.shape, self.acc_type)
                cute.copy(tc_t2r_qdo, p_t_qdo[(None, None, None, qdoh.index)], qdo_reg)
                qdoh.release()

                # T2R wdv result from MMA3
                wdvh = wdv.wait()
                wdv_reg = cute.make_rmem_tensor(coords_wdv.shape, self.acc_type)
                cute.copy(tc_t2r_wdv, p_t_wdv[(None, None, None, wdvh.index)], wdv_reg)
                wdvh.release()

                # dh += qdo * scale - wdv
                for ei in cutlass.range_constexpr(cute.size(st)):
                    st[ei] = st[ei] + qdo_reg[ei] * params.scale - wdv_reg[ei]

            # ========== Store dh0 (initial state gradient) ==========
            if params.use_dh0:
                gDh0 = params.g_dh0_t[None, None, (h_idx, seq_idx)]
                for ei in cutlass.range(cute.size(st), unroll_full=True):
                    kc, vc = coords_kv[ei]
                    gDh0[vc + v_tile * self.BV, kc] = st[ei]
            work.advance()
        return dhb, dh_epi, dv, dv2b, dv2_epi, gk_rdy, qdo, wdv, work

    @role(warps=4, regs=64)
    @cute.jit
    def load(
        self,
        params: DeltaHParams,
        *,
        gk: Producer,
        k: Producer,
        q: Producer,
        do: Producer,
        w: Producer,
        aqk: Producer,
        work: WorkCursor,
    ):
        """Own TMA G2S loads and produce the recurrence operand rings."""
        mma_dv, mma_qdo, mma_aqdo, mma_wdv = params.mmas
        atom_k, desc_k = params.tma.k
        atom_q, desc_q = params.tma.q
        atom_do, desc_do = params.tma.do
        atom_w, desc_w = params.tma.w
        atom_aqk, desc_aqk = params.tma.aqk
        cpasync.prefetch_descriptor(atom_k)
        cpasync.prefetch_descriptor(atom_q)
        cpasync.prefetch_descriptor(atom_do)
        cpasync.prefetch_descriptor(atom_w)
        cpasync.prefetch_descriptor(atom_aqk)
        atom_gk = desc_gk = None
        if cutlass.const_expr(not self.scalar_gate):
            atom_gk, desc_gk = params.tma.gk
            cpasync.prefetch_descriptor(atom_gk)

        while work.valid:
            v_tile, h_idx, seq_idx = work.item
            bos, sequence_length, num_chunks, db = self.sequence(params, seq_idx)
            if cutlass.const_expr(self.varlen):
                k_desc = cute.domain_offset((bos, 0, (0, 0)), desc_k)
                q_desc = cute.domain_offset((0, bos, (0, 0)), desc_q)
                do_desc = cute.domain_offset((0, bos, (0, 0)), desc_do)
                w_desc = cute.domain_offset((0, bos, (0, 0)), desc_w)
                aqk_desc = cute.domain_offset((0, bos, (0, 0)), desc_aqk)
                if cutlass.const_expr(not self.scalar_gate):
                    gk_desc = cute.domain_offset((0, bos, (0, 0)), desc_gk)
            else:
                k_desc = desc_k
                q_desc = desc_q
                do_desc = desc_do
                w_desc = desc_w
                aqk_desc = desc_aqk
                gk_desc = desc_gk

            # Partition TMA operands
            tKs, tKg = self._part_a(atom_k, k_desc, params.sK, self.dv_tile, mma_dv, db, h_idx)
            tQs, tQg = self._part_a(atom_q, q_desc, params.sQ, self.qdo_tile, mma_qdo, db, h_idx)
            tDos, tDog = self._part_b(
                atom_do, do_desc, params.sDo, self.qdo_tile, mma_qdo, db, h_idx
            )
            tWs, tWg = self._part_a(atom_w, w_desc, params.sW, self.wdv_tile, mma_wdv, db, h_idx)
            tAqks, tAqkg = self._part_a(
                atom_aqk,
                aqk_desc,
                params.sAqk,
                (self.BT, self.BV, self.BT),
                mma_aqdo,
                db,
                h_idx,
            )
            k = k.views(K=StageView(tKs, axis=1))
            q = q.views(Q=StageView(tQs, axis=1))
            do = do.views(Do=StageView(tDos, axis=1))
            w = w.views(W=StageView(tWs, axis=1))
            aqk = aqk.views(Aqk=StageView(tAqks, axis=1))

            if cutlass.const_expr(not self.scalar_gate):
                gGK_l = gk_desc[None, None, (h_idx, db)]
                sSGK, gSGK = self._part_epi(atom_gk, gGK_l, (self.BK, 1), params.sGK)
                gk = gk.views(GK=StageView(sSGK, axis=1))

            # ---------- Chunk loop (reverse TMA loads) ----------
            for ct in cutlass.range(0, num_chunks, unroll=0):
                rev_ct = num_chunks - 1 - ct

                # gk load FIRST, clamping the tail to its final valid token.
                if cutlass.const_expr(not self.scalar_gate):  # noqa: SIM102
                    if params.use_gk:
                        gk_t = rev_ct * self.BT + self.BT - 1
                        remaining = sequence_length - rev_ct * self.BT
                        if remaining < self.BT:
                            gk_t = sequence_length - 1
                        with gk.produce() as h:
                            cute.copy(
                                atom=atom_gk,
                                src=gSGK[(None, 0, gk_t)],
                                dst=h.view("GK", write=True, by="tma"),
                                tma_bar_ptr=h.barrier,
                            )

                # k load (A-operand for MMA1)
                with k.produce() as h:
                    cute.copy(
                        atom=atom_k,
                        src=tKg[None, rev_ct, 0],
                        dst=h.view("K", write=True, by="tma"),
                        tma_bar_ptr=h.barrier,
                    )
                # q^T load (A-operand for MMA2)
                with q.produce() as h:
                    cute.copy(
                        atom=atom_q,
                        src=tQg[None, 0, rev_ct],
                        dst=h.view("Q", write=True, by="tma"),
                        tma_bar_ptr=h.barrier,
                    )
                with do.produce() as h:
                    cute.copy(
                        atom=atom_do,
                        src=tDog[None, v_tile, rev_ct],
                        dst=h.view("Do", write=True, by="tma"),
                        tma_bar_ptr=h.barrier,
                    )
                # w^T load (A-operand for MMA3)
                with w.produce() as h:
                    cute.copy(
                        atom=atom_w,
                        src=tWg[None, 0, rev_ct],
                        dst=h.view("W", write=True, by="tma"),
                        tma_bar_ptr=h.barrier,
                    )
                with aqk.produce() as h:
                    cute.copy(
                        atom=atom_aqk,
                        src=tAqkg[None, 0, rev_ct],
                        dst=h.view("Aqk", write=True, by="tma"),
                        tma_bar_ptr=h.barrier,
                    )
            work.advance()
        return gk, k, q, do, w, aqk, work

    @role(warps=5, regs=64)
    @cute.jit
    def gate(
        self,
        params: DeltaHParams,
        *,
        gk: Consumer,
        gk_rdy: Producer,
        work: WorkCursor,
        me: Local,
    ):
        """Consume gk TMA stages and produce exp2 gate stages for the state warps.

        A scalar gate has no TMA stage: each lane loads the chunk's last-token decay and
        broadcasts its exp2 across the key dimensions it owns.
        """
        gk_tid = me.tid
        while work.valid:
            _value_tile, head_idx, sequence_idx = work.item
            bos, sequence_length, num_chunks, db = self.sequence(params, sequence_idx)
            for ct in cutlass.range(0, num_chunks, unroll=0):
                if params.use_gk:
                    if cutlass.const_expr(self.scalar_gate):
                        rev_ct = num_chunks - 1 - ct
                        gk_t = rev_ct * self.BT + self.BT - 1
                        if sequence_length - rev_ct * self.BT < self.BT:
                            gk_t = sequence_length - 1
                        gk_exp = cute.exp2(
                            Float32(params.g_gate[(bos + gk_t, (head_idx, db))]),
                            fastmath=self.fastmath,
                        )
                        with gk_rdy.produce() as ready:
                            gk_exp_buf = ready.view(write=True)
                            for i in cutlass.range(4, unroll_full=True):
                                gk_exp_buf[gk_tid * 4 + i] = gk_exp
                    else:
                        gkh = gk.wait()
                        ready = gk_rdy.acquire()
                        gk_buf = gkh.data
                        gk_exp_buf = ready.view(write=True)
                        for i in cutlass.range(4, unroll_full=True):
                            idx = gk_tid * 4 + i
                            gk_exp_buf[idx] = cute.exp2(gk_buf[(idx, 0)], fastmath=self.fastmath)
                        gkh.release()
                        ready.commit()
            work.advance()
        return gk, gk_rdy, work

    @role(warps=6, regs=64)
    @cute.jit
    def mma(
        self,
        params: DeltaHParams,
        *,
        dhb: Consumer,
        k: Consumer,
        dv: Producer,
        q: Consumer,
        do: Consumer,
        qdo: Producer,
        aqk: Consumer,
        dv2b: Consumer,
        w: Consumer,
        wdv: Producer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        """Consume operands, neutralize packed tails, and produce TMEM results."""
        mma_dv, mma_qdo, mma_aqdo, mma_wdv = params.mmas
        mma_tid = me.tid
        t_dv_acc = cute.make_tensor(tmem.ptr("dv"), params.dv_layout)
        t_qdo_acc = cute.make_tensor(tmem.ptr("qdo"), params.qdo_layout)
        t_wdv_acc = cute.make_tensor(tmem.ptr("wdv"), params.wdv_layout)
        t_k_a = params.t_k_a
        t_q_a = params.t_q_a
        t_aqk_a = params.t_aqk_a
        t_w_a = params.t_w_a
        k = k.views(K=StageView(t_k_a, axis=3))
        dhb = dhb.views(mma=StageView(params.t_dhb_b, axis=3))
        q = q.views(Q=StageView(t_q_a, axis=3))
        do = do.views(Do=StageView(params.t_do_b, axis=3))
        aqk = aqk.views(Aqk=StageView(t_aqk_a, axis=3))
        w = w.views(W=StageView(t_w_a, axis=3))
        dv2b = dv2b.views(mma=StageView(params.t_dv2b_b, axis=3))

        while work.valid:
            _value_tile, _head_idx, sequence_idx = work.item
            _, sequence_length, num_chunks, _ = self.sequence(params, sequence_idx)
            tail_valid_rows = Int32(0)
            if cutlass.const_expr(self.varlen):
                tail_valid_rows = sequence_length % self.BT
            for ct in cutlass.range(0, num_chunks, unroll=0):
                # --- MMA1: k(SMEM A) × dh(SMEM B) → dv_acc(TMEM) ---
                dhbh = dhb.wait()
                kh = k.wait()
                dvd = dv.acquire()
                for kp in cutlass.range(cute.size(t_k_a, mode=[2]), unroll_full=True):
                    mma_dv.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma_dv,
                        t_dv_acc[None, None, None, dvd.index],
                        kh.view("K", None, None, kp, read=True, by="umma"),
                        dhbh.view("mma", None, None, kp, read=True, by="umma"),
                        t_dv_acc[None, None, None, dvd.index],
                    )
                kh.release()
                dhbh.release()

                # --- MMA2: q^T(SMEM A) × do(SMEM B) → qdo_acc(TMEM) ---
                qh = q.wait()
                doh = do.wait()
                qdod = qdo.acquire()
                if cutlass.const_expr(self.varlen):  # noqa: SIM102
                    if ct == 0 and tail_valid_rows != 0:
                        self._neutralize_do_reduction_rows(doh.data, tail_valid_rows, mma_tid)

                # Aqk^T × do accumulates into MMA1 before releasing dv.
                aqkh = aqk.wait()
                if cutlass.const_expr(self.varlen):  # noqa: SIM102
                    if ct == 0 and tail_valid_rows != 0:
                        self._neutralize_aqk_reduction_rows(aqkh.data, tail_valid_rows, mma_tid)
                for kp in cutlass.range(cute.size(t_aqk_a, mode=[2]), unroll_full=True):
                    mma_aqdo.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
                    cute.gemm(
                        mma_aqdo,
                        t_dv_acc[None, None, None, dvd.index],
                        aqkh.view("Aqk", None, None, kp, read=True, by="umma"),
                        # The dV-intra MMA's own partition of the dO stage.
                        params.t_do_aq_b[None, None, kp, doh.index],
                        t_dv_acc[None, None, None, dvd.index],
                    )
                dvd.commit()
                aqkh.release()

                # --- MMA2: q^T × do → qdo ---
                if cutlass.const_expr(self.varlen):  # noqa: SIM102
                    if ct == 0 and tail_valid_rows != 0:
                        self._neutralize_reduction_rows(qh.data, tail_valid_rows, mma_tid)
                for kp in cutlass.range(cute.size(t_q_a, mode=[2]), unroll_full=True):
                    mma_qdo.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma_qdo,
                        t_qdo_acc[None, None, None, qdod.index],
                        qh.view("Q", None, None, kp, read=True, by="umma"),
                        doh.view("Do", None, None, kp, read=True, by="umma"),
                        t_qdo_acc[None, None, None, qdod.index],
                    )
                qdod.commit()
                qh.release()
                doh.release()

                # --- MMA3: w(SMEM A) × dv2(SMEM B) → wdv_acc(TMEM) ---
                dv2bh = dv2b.wait()
                wh = w.wait()
                if cutlass.const_expr(self.varlen):  # noqa: SIM102
                    if ct == 0 and tail_valid_rows != 0:
                        self._neutralize_reduction_rows(wh.data, tail_valid_rows, mma_tid)
                wdvd = wdv.acquire()
                for kp in cutlass.range(cute.size(t_w_a, mode=[2]), unroll_full=True):
                    mma_wdv.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma_wdv,
                        t_wdv_acc[None, None, None, wdvd.index],
                        wh.view("W", None, None, kp, read=True, by="umma"),
                        dv2bh.view("mma", None, None, kp, read=True, by="umma"),
                        t_wdv_acc[None, None, None, wdvd.index],
                    )
                wdvd.commit()
                wh.release()
                dv2bh.release()
            work.advance()
        return dhb, k, dv, q, do, qdo, aqk, dv2b, w, wdv, work

    @role(warps=7, regs=64)
    @cute.jit
    def store(
        self,
        params: DeltaHParams,
        *,
        dh_epi: Consumer,
        dv2_epi: Consumer,
        work: WorkCursor,
        me: Local,
    ):
        """Store packed chunk states and dV without writing tail padding."""
        atom_dhst, desc_dhst = params.tma.dh_store
        atom_dv2st, desc_dv2st = params.tma.dv2_store
        copy_dv2_tiled = params.copy_dv2_tiled
        g_dv2 = params.g_dv2
        cpasync.prefetch_descriptor(atom_dhst)
        cpasync.prefetch_descriptor(atom_dv2st)
        store_tid = me.tid

        while work.valid:
            value_tile, head_idx, sequence_idx = work.item
            bos, sequence_length, num_chunks, batch_idx = self.sequence(params, sequence_idx)
            if cutlass.const_expr(self.varlen):
                chunk_offset = params.chunk_offsets[sequence_idx]
                dh_desc = cute.domain_offset((0, 0, (chunk_offset, 0, 0)), desc_dhst)
                dv_desc = cute.domain_offset((0, bos, (0, 0)), desc_dv2st)
            else:
                dh_desc = desc_dhst
                dv_desc = desc_dv2st

            gDH_s = dh_desc[None, None, (None, head_idx, batch_idx)]
            sDHst, gDHst = self._part_epi(atom_dhst, gDH_s, (self.BV, self.BK), params.sDhEpi)
            gDV2_s = dv_desc[None, None, (head_idx, batch_idx)]
            sDV2st, gDV2st = self._part_epi(atom_dv2st, gDV2_s, (self.BV, self.BT), params.sDv2Epi)
            dh_epi = dh_epi.views(tma=StageView(sDHst, axis=1))
            dv2_epi = dv2_epi.views(tma=StageView(sDV2st, axis=1))

            for chunk in cutlass.range(0, num_chunks, unroll=0):
                reverse_chunk = num_chunks - 1 - chunk
                with dh_epi.consume() as state_handle:
                    cute.copy(
                        atom_dhst,
                        state_handle.view("tma", read=True, by="tma"),
                        gDHst[(None, value_tile, 0, reverse_chunk)],
                    )
                    bulk_commit_group()
                    bulk_wait_group(0, read=True)

                value_handle = dv2_epi.wait()
                remaining = sequence_length - reverse_chunk * self.BT
                if cutlass.const_expr(self.varlen):
                    if remaining < self.BT:
                        # The TMA view is registered under "tma"; this tail reads the
                        # staged tile itself with ordinary loads.
                        value_handle.access("tma", read=True)
                        source = params.sDv2Epi[None, None, value_handle.index]
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
                            value_handle.view("tma", read=True, by="tma"),
                            gDV2st[(None, value_tile, reverse_chunk)],
                        )
                        bulk_commit_group()
                        bulk_wait_group(0, read=True)
                else:
                    cute.copy(
                        atom_dv2st,
                        value_handle.view("tma", read=True, by="tma"),
                        gDV2st[(None, value_tile, reverse_chunk)],
                    )
                    bulk_commit_group()
                    bulk_wait_group(0, read=True)
                value_handle.release()
            work.advance()
        return dh_epi, dv2_epi, work

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
        if cutlass.const_expr(self.scalar_gate):
            g_gk_k = cute.make_tensor(
                gk_in.iterator,
                cute.make_layout((T, (H, dB)), stride=(self.upcast(H), (1, self.upcast(T) * H))),
            )
        else:
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
                cute.group_modes(cute.select(gk_in.layout, mode=[1, 2, 0]), 1, 3),
            )
            if cutlass.const_expr(self.scalar_gate)
            else cute.make_tensor(
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
        bound_sequence_extent: Int32,
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

        # --- MMA configurations (SS-mode: both operands from SMEM) ---
        # MMA1: k @ dh → dv  — k is (BT,BK) as A K-major, dh is (BK,BV) as B K-major
        mma_dv = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.K,  # A (k): K-major
            cute.nvgpu.OperandMajorMode.K,  # B (dh): K-major
            self.acc_type,
            tcgen05.CtaGroup.ONE,
            self.dv_tile[:2],  # (BT=64, BV=16)
            tcgen05.OperandSource.SMEM,
        )
        # MMA2: q^T @ do → qdo  — q^T is (BK,BT) A MN-major, do is (BT,BV) B MN-major
        mma_qdo = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.MN,  # A (q^T): MN-major
            cute.nvgpu.OperandMajorMode.MN,  # B (do): MN-major (V-contiguous)
            self.acc_type,
            tcgen05.CtaGroup.ONE,
            self.qdo_tile[:2],  # (BK=128, BV=16)
            tcgen05.OperandSource.SMEM,
        )
        # MMA4: Aqk^T @ do → dv_intra, accumulated into MMA1's dv TMEM tile.
        aqdo_tile = (self.BT, self.BV, self.BT)
        mma_aqdo = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_type,
            tcgen05.CtaGroup.ONE,
            aqdo_tile[:2],
            tcgen05.OperandSource.SMEM,
        )

        # MMA3: w @ dv2 → wdv  — w is (BK,BT) A MN-major, dv2 is (BT,BV) B K-major
        mma_wdv = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.MN,  # A (w^T): MN-major
            cute.nvgpu.OperandMajorMode.K,  # B (dv2): K-major
            self.acc_type,
            tcgen05.CtaGroup.ONE,
            self.wdv_tile[:2],  # (BK=128, BV=16)
            tcgen05.OperandSource.SMEM,
        )

        # --- SMEM staged layouts ---
        tma_ld = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_st = cpasync.CopyBulkTensorTileS2GOp()

        # MMA1 A-operand: k — (BT, BK) K-major staged
        s_k_staged = sm100_utils.make_smem_layout_a(
            mma_dv,
            self.dv_tile,
            self.io_type,
            self.k.n,
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
            LayoutEnum.COL_MAJOR,
            (self.BK, self.BV),
            1,
        )

        # MMA2 A-operand: q^T — (BK, BT) MN-major staged
        s_q_staged = sm100_utils.make_smem_layout_a(
            mma_qdo,
            self.qdo_tile,
            self.io_type,
            self.q.n,
        )
        # TMA loads dO directly into its depth-2 MMA B-operand ring.
        s_do_staged = sm100_utils.make_smem_layout_b(
            mma_qdo,
            self.qdo_tile,
            self.io_type,
            self.do.n,
        )

        # MMA3 A-operand: w^T — (BK, BT) MN-major staged
        s_w_staged = sm100_utils.make_smem_layout_a(
            mma_wdv,
            self.wdv_tile,
            self.io_type,
            self.w.n,
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
            LayoutEnum.COL_MAJOR,
            (self.BT, self.BV),
            1,
        )

        # Aqk^T A-operand for the fused K=64 dv-intra MMA.
        s_aqk_staged = sm100_utils.make_smem_layout_a(
            mma_aqdo,
            aqdo_tile,
            self.io_type,
            self.aqk.n,
        )

        # dh snapshot epilogue: (V, K) for Store warp TMA S2G
        s_dh_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.COL_MAJOR,
            (self.BV, self.BK),
            self.dh_epi.n,
        )
        # R2S dh → sH_epi (ROW_MAJOR transposed for stmatrix)
        s_dh_r2s_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.ROW_MAJOR,
            (self.BK, self.BV),
            self.dh_epi.n,
        )

        # dv2 epilogue: (V, T) for Store warp TMA S2G
        s_dv2_epi_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.COL_MAJOR,
            (self.BV, self.BT),
            self.dv2_epi.n,
        )
        # R2S dv → sVst (ROW_MAJOR transposed for stmatrix)
        s_dv2_r2s_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.ROW_MAJOR,
            (self.BT, self.BV),
            self.dv2_epi.n,
        )

        clust_lay = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),  # one CTA
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

        # gk TMA: BK×1 fp32 tile (last timestep per chunk). A scalar gate is below the TMA
        # box minimum, so the gate warp reads it from global memory instead.
        s_gk_2d = cute.make_layout((self.BK, 1))
        tma_gk = None
        if cutlass.const_expr(not self.scalar_gate):
            tma_gk = TmaOp(
                *cpasync.make_tiled_tma_atom(
                    tma_ld,
                    g_gk_k,
                    s_gk_2d,
                    (self.BK, 1),
                )
            )

        cp_bits = 128
        cp_elems = cp_bits // self.io_type.width
        copy_atom_uni = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.io_type,
            num_bits_per_copy=cp_bits,
        )
        vn_thr_d0 = self.BV // cp_elems
        vn_thr_lay = cute.make_ordered_layout(
            (vn_thr_d0, cute.arch.WARP_SIZE // vn_thr_d0), order=(0, 1)
        )
        copy_dv2_tiled = cute.make_tiled_copy_tv(
            copy_atom_uni, vn_thr_lay, cute.make_layout((cp_elems, 1))
        )

        B, T, _H, _K, _V = problem_shape
        space = (
            (B, cu_seqlens_in, bound_sequence_extent) if cutlass.const_expr(self.varlen) else (B,)
        )
        bound = self.work.bind(space, None, self._max_ctas())
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
                tma_gk,
            ),
            g_gk_k if cutlass.const_expr(self.scalar_gate) else None,
            g_dht,
            g_dh0_t,
            g_dv2,
            copy_dv2_tiled,
            cu_seqlens_in,
            chunk_offsets_in,
            SmemLayouts(
                K=s_k_staged,
                Dhb=s_dhb_staged,
                DhbR2S=s_dhb_store_staged,
                Q=s_q_staged,
                Do=s_do_staged,
                W=s_w_staged,
                Dv2b=s_dv2b_staged,
                Dv2bR2S=s_dv2b_store_staged,
                Aqk=s_aqk_staged,
                DhEpi=s_dh_epi_staged,
                DhR2S=s_dh_r2s_staged,
                Dv2Epi=s_dv2_epi_staged,
                Dv2R2S=s_dv2_r2s_staged,
                GK=cute.make_layout((self.BK, 1, self.gk.n), stride=(1, self.BK, self.BK)),
                GKExp=cute.make_layout((self.BK, self.gk_rdy.n)),
            ),
            T,
            scale,
            use_gk,
            use_dht,
            use_dh0,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)

    # ------------------------------------------------------------------
    # Device kernel: bind storage, build fragments, run the roles
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mmas: Mmas,
        tma: TmaOps,
        g_gate: cute.Tensor | None,
        g_dht: cute.Tensor,
        g_dh0_t: cute.Tensor,
        g_dv2: cute.Tensor,
        copy_dv2_tiled: cute.TiledCopy,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        smem_layouts: SmemLayouts,
        tokens: Int32,
        scale: Float32,
        use_gk: Int32,
        use_dht: Int32,
        use_dh0: Int32,
        bound,
    ):
        self.begin_roles(work=bound, layouts=smem_layouts._asdict())
        mma_dv, mma_qdo, mma_aqdo, mma_wdv = mmas
        sK, sQ, sDo, sW, sAqk = (self.buffer(name) for name in ("K", "Q", "Do", "W", "Aqk"))
        sDhb, sDv2b = self.buffer("Dhb"), self.buffer("Dv2b")

        # #
        # MMA fragments — SS-mode: both A and B from SMEM, accumulators in TMEM
        #
        accumulators = []
        for mma, tile, region in (
            (mma_dv, self.dv_tile, "dv"),
            (mma_qdo, self.qdo_tile, "qdo"),
            (mma_wdv, self.wdv_tile, "wdv"),
        ):
            fragment = mma.make_fragment_C(cute.append(mma.partition_shape_C(tile[:2]), 1))
            # The plan reserves BV columns per accumulator; check the MMA agrees.
            assert tcgen05.find_tmem_tensor_col_offset(fragment) == self.BV, region
            accumulators.append(fragment.layout)
        dv_layout, qdo_layout, wdv_layout = accumulators

        params = DeltaHParams(
            mmas=mmas,
            tma=tma,
            g_gate=g_gate,
            g_dht=g_dht,
            g_dh0_t=g_dh0_t,
            g_dv2=g_dv2,
            copy_dv2_tiled=copy_dv2_tiled,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            sK=sK,
            sQ=sQ,
            sDo=sDo,
            sW=sW,
            sAqk=sAqk,
            sGK=self.buffer("GK"),
            sDhEpi=self.buffer("DhEpi"),
            sDv2Epi=self.buffer("Dv2Epi"),
            # MMA1: k (A) × dh (B) → dv_acc
            t_k_a=mma_dv.make_fragment_A(sK),
            t_dhb_b=mma_dv.make_fragment_B(sDhb),
            # MMA2: q^T (A) × do (B) → qdo_acc
            t_q_a=mma_qdo.make_fragment_A(sQ),
            t_do_b=mma_qdo.make_fragment_B(sDo),
            # Aqk^T (A) × do (B) accumulates into MMA1's dv tile.
            t_aqk_a=mma_aqdo.make_fragment_A(sAqk),
            t_do_aq_b=mma_aqdo.make_fragment_B(sDo),
            # MMA3: w (A) × dv2 (B) → wdv_acc
            t_w_a=mma_wdv.make_fragment_A(sW),
            t_dv2b_b=mma_wdv.make_fragment_B(sDv2b),
            dv_layout=dv_layout,
            qdo_layout=qdo_layout,
            wdv_layout=wdv_layout,
            tokens=tokens,
            dense_chunks=(tokens + self.BT - 1) // self.BT,
            scale=scale,
            use_gk=use_gk,
            use_dht=use_dht,
            use_dh0=use_dh0,
        )
        self.run_roles(params)

    # ------------------------------------------------------------------
    # SMEM and TMA helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _neutralize_do_reduction_rows(self, smem, valid_rows, tid):
        """Zero invalid token rows in one direct dO MMA B-operand stage."""
        value_atom = cute.size(smem, mode=[0, 0])
        token_atom = cute.size(smem, mode=[0, 1])
        assert value_atom == self.BV, "value atom must span BV"
        assert cute.size(smem, mode=[1]) == 1, "expected a single rest-N mode"
        assert token_atom * cute.size(smem, mode=[2]) == self.BT, "token modes must tile BT"
        for token in cutlass.range(valid_rows, self.BT, unroll=1):
            if tid < self.BV:
                smem[((tid, token % token_atom), 0, token // token_atom)] = self.io_type(0.0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _neutralize_aqk_reduction_rows(self, smem, valid_rows, tid):
        """Zero invalid query columns in one 64x64 Aqk operand stage."""
        row_atom = cute.size(smem, mode=[0, 0])
        token_atom = cute.size(smem, mode=[0, 1])
        assert cute.size(smem, mode=[1]) == 1, "expected a single rest-M mode"
        assert row_atom == self.BT
        assert token_atom * cute.size(smem, mode=[2]) == self.BT
        for token in cutlass.range(valid_rows, self.BT, unroll=1):
            for row_block in cutlass.range_constexpr(self.BT // cute.arch.WARP_SIZE):
                row = tid + row_block * cute.arch.WARP_SIZE
                smem[((row, token % token_atom), 0, token // token_atom)] = self.io_type(0.0)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()

    @cute.jit
    def _neutralize_reduction_rows(self, smem, valid_rows, tid):
        """Zero a ragged tail's invalid reduction columns in one stage before UMMA reads it."""
        row_atom = cute.size(smem, mode=[0, 0, 0])
        token_atom = cute.size(smem, mode=[0, 1])
        assert cute.size(smem, mode=[1]) == 1, "expected a single rest-M mode"
        assert row_atom * cute.size(smem, mode=[0, 0, 1]) == self.BK
        assert token_atom * cute.size(smem, mode=[2]) == self.BT
        for token in cutlass.range(valid_rows, self.BT, unroll=1):
            for row_block in cutlass.range_constexpr(self.BK // cute.arch.WARP_SIZE):
                row = tid + row_block * cute.arch.WARP_SIZE
                coordinate = (
                    ((row % row_atom, row // row_atom), token % token_atom),
                    0,
                    token // token_atom,
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
    def _max_ctas() -> int:
        # Target metadata is inherited by forked compile workers; do not query CUDA here.
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA compilation requires a CUDA target with an SM count")
        return sm_count


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
def _compile_delta_h_bwd(
    H,
    bv,
    io_type,
    use_int64_offsets,
    dynamic_state_layout,
    fastmath: bool = False,
    scalar_gate: bool = False,
):
    """Compile one dense BlackwellDeltaHBwd variant."""
    target = get_compile_target()
    if target.device_type != "cuda" or not is_sm100_kda_capability(target.effective_capability):
        raise ValueError(f"KDA delta-H backward requires an SM100 or SM103 target; got {target}")
    K = V = 128
    chunk_size = 64
    kern = BlackwellDeltaHBwd(
        head_bv=bv,
        num_heads=H,
        io_type=io_type,
        use_int64_offsets=use_int64_offsets,
        dynamic_state_layout=dynamic_state_layout,
        fastmath=fastmath,
        scalar_gate=scalar_gate,
    )
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    sa, sb, snt, sns, sn = (sym_int() for _ in range(5))

    qf, kf, wf = (make_fake_tensor(io_type, (sa, sb, H, K)) for _ in range(3))
    dof = make_fake_tensor(io_type, (sa, sb, H, V))
    aqkf = make_fake_tensor(io_type, (sa, sb, H, chunk_size))
    gkf = make_fake_tensor(cutlass.Float32, (sa, sb, H) if scalar_gate else (sa, sb, H, K))
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
        Int32(0),
    )


@jit_cache
def _compile_delta_h_bwd_packed(
    heads: int,
    bv: int,
    io_type: type[cutlass.Numeric],
    use_int64_offsets: bool,
    dynamic_state_layout: bool,
    fastmath: bool = False,
    scalar_gate: bool = False,
):
    """Compile one packed fused specialization with a width-matched TVM ABI."""
    target = get_compile_target()
    if target.device_type != "cuda" or not is_sm100_kda_capability(target.effective_capability):
        raise ValueError(f"KDA delta-H backward requires an SM100 or SM103 target; got {target}")
    key_dim = value_dim = 128
    chunk_size = 64
    kernel = BlackwellDeltaHBwd(
        num_heads=heads,
        head_bv=bv,
        io_type=io_type,
        varlen=True,
        use_int64_offsets=use_int64_offsets,
        dynamic_state_layout=dynamic_state_layout,
        fastmath=fastmath,
        scalar_gate=scalar_gate,
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
        (
            make_fake_tensor(cutlass.Float32, (tokens, heads))
            if scalar_gate
            else token_tensor(cutlass.Float32, key_dim)
        ),
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
    *,
    fastmath: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run packed delta-H with fused intra-chunk dV.

    ``gk`` is a per-key-dimension gate shaped like ``q`` or a scalar gate shaped ``q.shape[:3]``.
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
    scalar_gate = gk is not None and gk.shape == q.shape[:3]
    if gk is not None and not scalar_gate and gk.shape != q.shape:
        raise ValueError("gk must match q or q.shape[:3]")
    tensor_inputs = (q, k, w, do, aqk, *((gk,) if gk is not None else ()))
    if any(tensor.device != q.device for tensor in tensor_inputs):
        raise ValueError("packed delta-H+dV inputs must share q.device")
    if any(not tensor.is_contiguous() for tensor in tensor_inputs):
        raise ValueError("packed delta-H+dV inputs must be contiguous")
    if q.dtype not in _IO_TYPES or any(tensor.dtype != q.dtype for tensor in (k, w, do, aqk)):
        raise TypeError("q, k, w, do, and Aqk must share dtype float16 or bfloat16")
    io_type = cute_dtype(q.dtype).cute_type
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
    # Contiguous like the Hopper backward, so one fake describes both.
    d_initial_state = None if h0 is None else h0.new_empty(h0.shape)
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
        requires_dynamic_state_layout(final_state_kernel, initial_state_gradient_kernel),
        fastmath,
        scalar_gate,
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
        Int32(should_bound_sequence_extent(tokens, sequences, heads, chunk_size, h0 is not None)),
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
    *,
    fastmath: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the dense B=1 CuTeDSL SM100 delta-H backward leaf.

    ``gk`` is a per-key-dimension gate shaped like ``q`` or a scalar gate shaped ``q.shape[:3]``.
    """
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
    scalar_gate = gk is not None and gk.shape == q.shape[:3]
    if gk is not None and not scalar_gate and gk.shape != q.shape:
        raise ValueError("gk must match q or q.shape[:3]")
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
    dh0_out = None if h0 is None else h0.new_empty(h0.shape)
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
    io_type = cute_dtype(q.dtype).cute_type
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
        fastmath,
        scalar_gate,
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
        Int32(0),
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
    *,
    fastmath: bool = False,
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
            fastmath=fastmath,
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
        fastmath=fastmath,
    )


__all__ = [
    "blackwell_delta_h_bwd_dhu_dv_fused",
    "blackwell_delta_h_bwd_dhu_dv_fused_dispatch",
]
