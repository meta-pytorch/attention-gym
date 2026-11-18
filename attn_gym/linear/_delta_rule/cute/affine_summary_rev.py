# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Blackwell TMA/UMMA reverse affine summaries for delta-rule context parallelism.

The local backward recurrence is affine in its incoming final-state cotangent:
``dH_in = dH_out @ R + C``. This kernel evaluates the recurrence on persistent
``[K, BN]`` tiles of the augmented FP32 state ``X = [0 | I]`` and returns its
V-first packed transpose ``[local_bias; reverse_transition]``.

Terms (chunk, range, column tile, and why the routing sends one range per fragment) are defined
in ``affine_summary_fwd``; *work item* and *partial* in
``attn_gym.linear._delta_rule.triton.work_items``. Each CTA is persistent: it walks a
grid-strided sequence of flat tile indices, decodes each into (column tile, head, work item), and
reads the item's token range from the device ``work`` table, so nothing about the layout is baked
into the launch. Four CUDA warps hold the FP32 state in registers, one warp issues TMA loads, one
computes gate exponentials, and one issues the four UMMA operations. Warpdrive pads the four
roles' seven warps to eight for the 192/56-register warpgroup split. Bias tiles load the local
output-gradient sources; transition tiles skip those streams entirely. Only the final FP32
partial is written to global memory.

The chunk recurrence is a serial latency chain, so when ``heads * tiles`` leaves SMs idle the
wrapper cuts the ranges' chunks into work items that each scan from ``[0 | I]`` and
``compose_work_items`` folds the partials per range on the device.

The fp32-valued MMA B operands (state ``X`` and the corrected write gradient
``dv``) are split into hi/lo halves of the I/O dtype and accumulated in two UMMA
passes, exactly as the forward summary treats ``X`` and ``wx``, so the reverse
transition is the transpose of the map the forward summary actually evaluates.
"""

from dataclasses import dataclass
from typing import NamedTuple

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.arch import runs_sm100_kernels
from attn_gym._backends.cute import get_device_properties
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear._delta_rule.cute.affine_summary_fwd import (
    _CUTE_IO_TYPES,
    _IO_TYPE_NAMES,
    BT,
    DATA_ALIGN_BYTES,
    KEY_DIM,
    MAX_RANGES,
    SUMMARY_DIM,
    VAL_DIM,
    TmaOp,
    _aligned,
    _feature_sequence_head_batch_view,
    _sequence_feature_head_batch_view,
    decode_work_row,
    plan_work_budget,
    range_tail,
    smem_tile_writer,
    store_rounded,
    tma_partition_a,
    tma_partition_b,
    tma_partition_gate,
    tmem_tile_reader,
    zero_tail_tokens,
)
from attn_gym.linear._delta_rule.routing import uses_legacy_cute
from attn_gym.linear._delta_rule.triton.work_items import compose_work_items, work_table
from attn_gym.utils import cdiv
from attn_gym.warpdrive import (
    Consumer,
    Local,
    Producer,
    Ring,
    SmemPlan,
    StaticPersistent,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, get_compile_target, jit_cache
from attn_gym.warpdrive.compat import LayoutEnum

# The flat compile cache builds the op internally. Keep host launch resources separate;
# StaticPersistent reads only max_ctas, but carries the same workspace ABI as other schedules.
_LAUNCH_WORK = WorkStream(schedule=StaticPersistent())


class TmaOps(NamedTuple):
    """TMA operations owned by the reverse-summary kernel."""

    k: TmaOp
    q: TmaOp
    do: TmaOp
    w: TmaOp
    aqk: TmaOp
    gate: TmaOp


class Mmas(NamedTuple):
    """Tensor-core operations owned by the MMA warp."""

    dv: cute.TiledMma
    qdo: cute.TiledMma
    aqdo: cute.TiledMma
    wdv: cute.TiledMma


@dataclass(frozen=True)
class ReverseSummaryParams:
    """Shared layouts and runtime inputs; channel state stays local to each role."""

    work_table: cute.Tensor
    out: cute.Tensor
    mmas: Mmas
    tma: TmaOps
    scale: Float32
    dv_layout: cute.Layout
    qdo_layout: cute.Layout
    wdv_layout: cute.Layout
    s_k: cute.Tensor
    s_state: cute.Tensor
    s_state_store: cute.Tensor
    s_q: cute.Tensor
    s_do: cute.Tensor
    s_w: cute.Tensor
    s_dv: cute.Tensor
    s_dv_store: cute.Tensor
    s_aqk: cute.Tensor
    gate_3d: cute.Tensor
    gate: cute.Tensor
    gate_exp: cute.Tensor


class BlackwellDeltaAffineSummaryRev(WarpSpecialized):
    """Compute reverse affine summaries with persistent TMA/UMMA state tiles."""

    # Augmented-state columns per CTA. A BN=16 tile covers half the columns in only slightly
    # less time per chunk; work items fill idle SMs instead.
    BN = 32

    def __init__(
        self,
        num_heads: int,
        io_type: type[cutlass.Numeric],
        use_int64_offsets: bool,
        whole_ranges: bool,
        fastmath: bool = False,
    ):
        self.num_heads = num_heads
        self.io_type = io_type
        self.acc_type = cutlass.Float32
        self.BT = BT
        self.BK = KEY_DIM
        self.use_int64_offsets = use_int64_offsets
        self.fastmath = fastmath
        # With ``whole_ranges`` the work table is ``bounds`` itself, ``[R, 2]`` token ranges,
        # and every row is one item: the ``budget == 1`` case needs no planner and no fold.
        self.whole_ranges = whole_ranges

        self.dv_tile = (self.BT, self.BN, self.BK)
        self.qdo_tile = (self.BK, self.BN, self.BT)
        self.aqdo_tile = (self.BT, self.BN, self.BT)
        self.wdv_tile = (self.BK, self.BN, self.BT)

        self.k_depth = 4
        self.q_depth = 2
        self.do_depth = 2
        self.w_depth = 2
        self.aqk_depth = 2
        self.gate_depth = 2
        self.dv_acc_depth = 1
        self.qdo_acc_depth = 1
        self.wdv_acc_depth = 1

        self.cta_group = tcgen05.CtaGroup.ONE
        self.smem_plan = SmemPlan()
        buffers = {
            name: self.smem_plan.buffer(name, self.io_type, align=1024)
            for name in ("K", "State", "Q", "Do", "W", "DvOperand", "Aqk")
        }
        state_store = self.smem_plan.buffer(
            "StateStore", self.io_type, align=1024, overlaps=buffers["State"]
        )
        dv_store = self.smem_plan.buffer(
            "DvStore", self.io_type, align=1024, overlaps=buffers["DvOperand"]
        )
        gate = self.smem_plan.buffer("Gate", Float32, align=128)
        gate_exp = self.smem_plan.buffer("GateExp", Float32, align=128)
        io_bytes = self.io_type.width // 8
        self.k = Ring(self.k_depth, Tma(self.BT * self.BK * io_bytes), Umma, buffer=buffers["K"])
        self.q = Ring(self.q_depth, Tma(self.BK * self.BT * io_bytes), Umma, buffer=buffers["Q"])
        self.do = Ring(
            self.do_depth, Tma(self.BN * self.BT * io_bytes), Umma, buffer=buffers["Do"]
        )
        self.w = Ring(self.w_depth, Tma(self.BK * self.BT * io_bytes), Umma, buffer=buffers["W"])
        self.aqk = Ring(
            self.aqk_depth, Tma(self.BT * self.BT * io_bytes), Umma, buffer=buffers["Aqk"]
        )
        # One elected gate-warp release follows its generic reads and async-proxy fence.
        self.gate = Ring(self.gate_depth, Tma(self.BK * 4), Warps, buffer=gate)
        self.gate_ready = Ring(self.gate_depth, Threads, Threads, buffer=gate_exp)
        self.state_pipe = Ring(
            2, Threads, Umma, buffer={"State": buffers["State"], "Store": state_store}
        )
        self.dv_operand = Ring(
            2, Threads, Umma, buffer={"Dv": buffers["DvOperand"], "Store": dv_store}
        )
        self.dv_pipe = Ring(self.dv_acc_depth, Umma, Threads, release_fence="tmem_load")
        self.qdo_pipe = Ring(self.qdo_acc_depth, Umma, Threads, release_fence="tmem_load")
        self.wdv_pipe = Ring(self.wdv_acc_depth, Umma, Threads, release_fence="tmem_load")
        self.tmem_plan = (
            TmemPlan()
            .region("dv", self.BN, self.dv_acc_depth)
            .region("qdo", self.BN, self.qdo_acc_depth)
            .region("wdv", self.BN, self.wdv_acc_depth)
        )
        self.work = WorkStream(schedule=StaticPersistent())

    def get_name(self) -> str:
        """Return a stable artifact and profiler name for this specialization."""
        dtype_name = "bf16" if self.io_type is cutlass.BFloat16 else "fp16"
        return (
            f"delta_affine_summary_rev_h{self.num_heads}_bn{self.BN}_{dtype_name}"
            f"_i64{int(self.use_int64_offsets)}_whole{int(self.whole_ranges)}"
            f"_fm{int(self.fastmath)}"
        )

    @cute.jit
    def __call__(
        self,
        qg: cute.Tensor,
        kg: cute.Tensor,
        w: cute.Tensor,
        dout: cute.Tensor,
        aqk: cute.Tensor,
        cumulative_gate: cute.Tensor,
        work_table: cute.Tensor,
        out: cute.Tensor,
        scale: Float32,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        """Construct TMA/UMMA descriptors and launch the persistent kernel.

        ``work_table[w] = (start, chunk_begin, chunk_end, length)`` names a run of chunks of one
        range (see ``decode_work_row``); ``out[w]`` receives its reverse map.
        """
        g_k = _sequence_feature_head_batch_view(kg)
        g_qt = _feature_sequence_head_batch_view(qg)
        g_wt = _feature_sequence_head_batch_view(w)
        g_gate = _feature_sequence_head_batch_view(cumulative_gate)
        g_do = _feature_sequence_head_batch_view(dout)
        g_aqk = _feature_sequence_head_batch_view(aqk)

        # For one augmented-state tile X [K, BN], each reverse chunk evaluates:
        #   dv    = kg @ X + Aqk^T @ dO       [BT, BN]
        #   X     = decay * X + scale * qg^T @ dO - w^T @ dv
        # dO is source-only: output gradients occupy the bias columns, while the
        # transition columns are zero because they have no local output source.
        # The four UMMAs below materialize those four matrix products. Separate
        # operations are needed because their reduction dimensions and physical
        # operand layouts differ, even when they share an output accumulator.

        # MMA1: kg [BT,K] reads X [K,BN] to begin the local write gradient dv.
        mma_dv = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_type,
            self.cta_group,
            self.dv_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # MMA2: qg^T [K,BT] reads dO [BT,BN] for the direct query contribution.
        mma_qdo = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_type,
            self.cta_group,
            self.qdo_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # MMA3: Aqk^T [BT,BT] reads dO and accumulates the local term into dv.
        mma_aqdo = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            self.acc_type,
            self.cta_group,
            self.aqdo_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # MMA4: w^T [K,BT] reads the completed dv for the subtractive WY term.
        mma_wdv = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_type,
            self.cta_group,
            self.wdv_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        mmas = Mmas(mma_dv, mma_qdo, mma_aqdo, mma_wdv)

        s_k = sm100_utils.make_smem_layout_a(
            mma_dv,
            self.dv_tile,
            self.io_type,
            self.k_depth,
        )
        # B operands hold hi/lo halves of the FP32 state and dv tiles. Each half is
        # one pipeline stage consumed by one accumulate pass of the same MMA.
        s_state = sm100_utils.make_smem_layout_b(
            mma_dv,
            self.dv_tile,
            self.io_type,
            2,
        )
        s_state_store = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.COL_MAJOR,
            (self.BK, self.BN),
            2,
        )
        s_q = sm100_utils.make_smem_layout_a(
            mma_qdo,
            self.qdo_tile,
            self.io_type,
            self.q_depth,
        )
        s_do = sm100_utils.make_smem_layout_b(
            mma_qdo,
            self.qdo_tile,
            self.io_type,
            self.do_depth,
        )
        s_w = sm100_utils.make_smem_layout_a(
            mma_wdv,
            self.wdv_tile,
            self.io_type,
            self.w_depth,
        )
        s_dv = sm100_utils.make_smem_layout_b(
            mma_wdv,
            self.wdv_tile,
            self.io_type,
            2,
        )
        s_dv_store = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.COL_MAJOR,
            (self.BT, self.BN),
            2,
        )
        s_aqk = sm100_utils.make_smem_layout_a(
            mma_aqdo,
            self.aqdo_tile,
            self.io_type,
            self.aqk_depth,
        )
        smem_layouts = {
            "K": s_k,
            "State": s_state,
            "StateStore": s_state_store,
            "Q": s_q,
            "Do": s_do,
            "W": s_w,
            "DvOperand": s_dv,
            "DvStore": s_dv_store,
            "Aqk": s_aqk,
            "Gate": cute.make_layout((self.BK, 1, self.gate_depth), stride=(1, self.BK, self.BK)),
            "GateExp": cute.make_layout((self.BK, self.gate_depth)),
        }

        tma_load = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        cluster_layout = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (mma_dv.thr_id.shape,),
        )

        s_k_one = cute.select(s_k, mode=[0, 1, 2])
        atom_k, desc_k = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load,
            g_k,
            s_k_one,
            self.dv_tile,
            mma_dv,
            cluster_layout.shape,
        )
        s_q_one = cute.select(s_q, mode=[0, 1, 2])
        atom_q, desc_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load,
            g_qt,
            s_q_one,
            self.qdo_tile,
            mma_qdo,
            cluster_layout.shape,
        )
        s_do_one = cute.select(s_do, mode=[0, 1, 2])
        atom_do, desc_do = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load,
            g_do,
            s_do_one,
            self.qdo_tile,
            mma_qdo,
            cluster_layout.shape,
        )
        s_w_one = cute.select(s_w, mode=[0, 1, 2])
        atom_w, desc_w = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load,
            g_wt,
            s_w_one,
            self.wdv_tile,
            mma_wdv,
            cluster_layout.shape,
        )
        s_aqk_one = cute.select(s_aqk, mode=[0, 1, 2])
        atom_aqk, desc_aqk = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load,
            g_aqk,
            s_aqk_one,
            self.aqdo_tile,
            mma_aqdo,
            cluster_layout.shape,
        )
        s_gate = cute.make_layout((self.BK, 1))
        atom_gate, desc_gate = cpasync.make_tiled_tma_atom(
            tma_load,
            g_gate,
            s_gate,
            (self.BK, 1),
        )
        tma_ops = TmaOps(
            TmaOp(atom_k, desc_k),
            TmaOp(atom_q, desc_q),
            TmaOp(atom_do, desc_do),
            TmaOp(atom_w, desc_w),
            TmaOp(atom_aqk, desc_aqk),
            TmaOp(atom_gate, desc_gate),
        )

        bound = self.work.bind(
            (SUMMARY_DIM // self.BN, self.num_heads, Int32(out.shape[0])), workspace, max_ctas
        )
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(mmas, tma_ops, work_table, out, smem_layouts, scale, bound).launch(
            **self.launch_config(bound),
            stream=stream,
        )

    @role(warps=range(4), regs=192)
    @cute.jit
    def run_state(
        self,
        params: ReverseSummaryParams,
        *,
        state_pipe: Producer,
        dv_pipe: Consumer,
        dv_operand: Producer,
        gate_ready: Consumer,
        qdo_pipe: Consumer,
        wdv_pipe: Consumer,
        work: WorkCursor,
        me: Local,
        tmem: Tmem,
    ):
        """Keep one FP32 state tile in registers across the reverse chunk scan."""
        out = params.out
        s_state_store = params.s_state_store
        s_dv_store = params.s_dv_store
        gate_exp = params.gate_exp
        scale = params.scale
        t_dv_acc = cute.make_tensor(tmem.ptr("dv"), params.dv_layout)
        t_qdo_acc = cute.make_tensor(tmem.ptr("qdo"), params.qdo_layout)
        t_wdv_acc = cute.make_tensor(tmem.ptr("wdv"), params.wdv_layout)
        local_tid = me.tmem_row

        t2r_dv, dv_slice, partitioned_dv = tmem_tile_reader(t_dv_acc, local_tid, self.BN)
        t2r_qdo, qdo_slice, partitioned_qdo = tmem_tile_reader(t_qdo_acc, local_tid, self.BN)
        t2r_wdv, wdv_slice, partitioned_wdv = tmem_tile_reader(t_wdv_acc, local_tid, self.BN)

        dv_coordinates = dv_slice.partition_D(
            cute.make_identity_tensor(cute.dice(self.dv_tile, (1, 1, None)))
        )
        state_coordinates = qdo_slice.partition_D(
            cute.make_identity_tensor(cute.dice(self.qdo_tile, (1, 1, None)))
        )
        wdv_coordinates = wdv_slice.partition_D(
            cute.make_identity_tensor(cute.dice(self.qdo_tile, (1, 1, None)))
        )
        state = cute.make_rmem_tensor(state_coordinates.shape, self.acc_type)

        state_writer = smem_tile_writer(t2r_qdo, local_tid, self.io_type)
        dv_writer = smem_tile_writer(t2r_dv, local_tid, self.io_type)

        while work.valid:
            column_tile, head, row = work.item
            _start, _length, chunk_begin, chunk_end = decode_work_row(
                params.work_table, row, self.whole_ranges
            )
            has_source = column_tile < VAL_DIM // self.BN

            for element in cutlass.range(cute.size(state), unroll_full=True):
                key, column = state_coordinates[element]
                global_column = column_tile * self.BN + column
                if global_column == VAL_DIM + key:
                    state[element] = Float32(1.0)
                else:
                    state[element] = Float32(0.0)

            for _chunk in cutlass.range(chunk_begin, chunk_end, unroll=0):
                # Publish the hi half as soon as it exists so MMA1 starts its first
                # pass while the lo residual is still being formed and stored.
                hi_handle = state_pipe.acquire()
                hi_handle.access("Store", write=True, by="threads")
                state_hi = store_rounded(
                    state.load(),
                    state_writer,
                    s_state_store[(None, None, hi_handle.index)],
                    self.io_type,
                )
                hi_handle.commit()
                lo_handle = state_pipe.acquire()
                lo_handle.access("Store", write=True, by="threads")
                store_rounded(
                    state.load() - state_hi.load().to(Float32),
                    state_writer,
                    s_state_store[(None, None, lo_handle.index)],
                    self.io_type,
                )
                lo_handle.commit()

                dv_handle = dv_pipe.wait()
                dv = cute.make_rmem_tensor(dv_coordinates.shape, self.acc_type)
                cute.copy(
                    t2r_dv,
                    partitioned_dv[(None, None, None, dv_handle.index)],
                    dv,
                )
                dv_handle.release()

                hi_handle = dv_operand.acquire()
                hi_handle.access("Store", write=True, by="threads")
                dv_hi = store_rounded(
                    dv.load(), dv_writer, s_dv_store[(None, None, hi_handle.index)], self.io_type
                )
                hi_handle.commit()
                lo_handle = dv_operand.acquire()
                lo_handle.access("Store", write=True, by="threads")
                store_rounded(
                    dv.load() - dv_hi.load().to(Float32),
                    dv_writer,
                    s_dv_store[(None, None, lo_handle.index)],
                    self.io_type,
                )
                lo_handle.commit()

                gate_handle = gate_ready.wait()
                for element in cutlass.range(cute.size(state), unroll_full=True):
                    key, _column = state_coordinates[element]
                    state[element] = state[element] * gate_exp[(key, gate_handle.index)]
                gate_handle.release()

                qdo = cute.make_rmem_tensor(state_coordinates.shape, self.acc_type)
                if has_source:
                    qdo_handle = qdo_pipe.wait()
                    cute.copy(
                        t2r_qdo,
                        partitioned_qdo[(None, None, None, qdo_handle.index)],
                        qdo,
                    )
                    qdo_handle.release()

                wdv_handle = wdv_pipe.wait()
                wdv = cute.make_rmem_tensor(wdv_coordinates.shape, self.acc_type)
                cute.copy(
                    t2r_wdv,
                    partitioned_wdv[(None, None, None, wdv_handle.index)],
                    wdv,
                )
                wdv_handle.release()

                if has_source:
                    for element in cutlass.range_constexpr(cute.size(state)):
                        state[element] = state[element] + qdo[element] * scale - wdv[element]
                else:
                    for element in cutlass.range_constexpr(cute.size(state)):
                        state[element] = state[element] - wdv[element]

            for element in cutlass.range(cute.size(state), unroll_full=True):
                key, column = state_coordinates[element]
                out[row, head, column_tile * self.BN + column, key] = state[element]

            work.advance()

        return state_pipe, dv_pipe, dv_operand, gate_ready, qdo_pipe, wdv_pipe, work

    @role(warps=4, regs=56)
    @cute.jit
    def run_load(
        self,
        params: ReverseSummaryParams,
        *,
        k: Producer,
        q: Producer,
        do: Producer,
        w: Producer,
        aqk: Producer,
        gate: Producer,
        work: WorkCursor,
    ):
        """Issue reverse-order TMA loads for all active operand streams."""
        s_k = params.s_k
        s_q = params.s_q
        s_do = params.s_do
        s_w = params.s_w
        s_aqk = params.s_aqk
        mma_dv, mma_qdo, mma_aqdo, mma_wdv = params.mmas
        tma = params.tma
        s_gate = params.gate_3d
        cpasync.prefetch_descriptor(tma.k.atom)
        cpasync.prefetch_descriptor(tma.q.atom)
        cpasync.prefetch_descriptor(tma.do.atom)
        cpasync.prefetch_descriptor(tma.w.atom)
        cpasync.prefetch_descriptor(tma.aqk.atom)
        cpasync.prefetch_descriptor(tma.gate.atom)
        atom_k, desc_k = tma.k
        atom_q, desc_q = tma.q
        atom_do, desc_do = tma.do
        atom_w, desc_w = tma.w
        atom_aqk, desc_aqk = tma.aqk
        atom_gate, desc_gate = tma.gate

        while work.valid:
            column_tile, head, row = work.item
            start, length, chunk_begin, chunk_end = decode_work_row(
                params.work_table, row, self.whole_ranges
            )
            has_source = column_tile < VAL_DIM // self.BN

            # A partial last chunk clamps the gate row here; the MMA warp zeroes every
            # token-indexed operand stage past the range boundary.
            k_desc = cute.domain_offset((start, 0, (0, 0)), desc_k)
            w_desc = cute.domain_offset((0, start, (0, 0)), desc_w)
            gate_desc = cute.domain_offset((0, start, (0, 0)), desc_gate)
            q_desc = cute.domain_offset((0, start, (0, 0)), desc_q)
            do_desc = cute.domain_offset((0, start, (0, 0)), desc_do)
            aqk_desc = cute.domain_offset((0, start, (0, 0)), desc_aqk)
            k_smem, k_gmem = tma_partition_a(atom_k, k_desc, s_k, self.dv_tile, mma_dv, head)
            w_smem, w_gmem = tma_partition_a(atom_w, w_desc, s_w, self.wdv_tile, mma_wdv, head)
            gate_smem, gate_gmem = tma_partition_gate(atom_gate, gate_desc, s_gate, head)
            q_smem, q_gmem = tma_partition_a(atom_q, q_desc, s_q, self.qdo_tile, mma_qdo, head)
            do_smem, do_gmem = tma_partition_b(
                atom_do, do_desc, s_do, self.qdo_tile, mma_qdo, head
            )
            aqk_smem, aqk_gmem = tma_partition_a(
                atom_aqk, aqk_desc, s_aqk, self.aqdo_tile, mma_aqdo, head
            )

            for chunk in cutlass.range(chunk_begin, chunk_end, unroll=0):
                reverse_chunk = chunk_end - 1 - (chunk - chunk_begin)

                gate_handle = gate.acquire()
                cute.copy(
                    atom=atom_gate,
                    src=gate_gmem[
                        (None, 0, cutlass.min(reverse_chunk * self.BT + self.BT - 1, length - 1))
                    ],
                    dst=gate_smem[None, gate_handle.index],
                    tma_bar_ptr=gate_handle.barrier,
                )
                gate_handle.commit()

                k_handle = k.acquire()
                cute.copy(
                    atom=atom_k,
                    src=k_gmem[None, reverse_chunk, 0],
                    dst=k_smem[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )
                k_handle.commit()
                w_handle = w.acquire()
                cute.copy(
                    atom=atom_w,
                    src=w_gmem[None, 0, reverse_chunk],
                    dst=w_smem[None, w_handle.index],
                    tma_bar_ptr=w_handle.barrier,
                )
                w_handle.commit()

                if has_source:
                    q_handle = q.acquire()
                    cute.copy(
                        atom=atom_q,
                        src=q_gmem[None, 0, reverse_chunk],
                        dst=q_smem[None, q_handle.index],
                        tma_bar_ptr=q_handle.barrier,
                    )
                    q_handle.commit()
                    do_handle = do.acquire()
                    cute.copy(
                        atom=atom_do,
                        src=do_gmem[None, column_tile, reverse_chunk],
                        dst=do_smem[None, do_handle.index],
                        tma_bar_ptr=do_handle.barrier,
                    )
                    do_handle.commit()
                    aqk_handle = aqk.acquire()
                    cute.copy(
                        atom=atom_aqk,
                        src=aqk_gmem[None, 0, reverse_chunk],
                        dst=aqk_smem[None, aqk_handle.index],
                        tma_bar_ptr=aqk_handle.barrier,
                    )
                    aqk_handle.commit()

            work.advance()

        return k, q, do, w, aqk, gate, work

    @role(warps=5, regs=56)
    @cute.jit
    def run_gate(
        self,
        params: ReverseSummaryParams,
        *,
        gate: Consumer,
        gate_ready: Producer,
        work: WorkCursor,
        me: Local,
    ):
        """Transform cumulative log2 gate stages into multiplicative decays."""
        gate_exp = params.gate_exp
        gate_tid = me.tid

        gate_data = params.gate
        while work.valid:
            _column_tile, _head, row = work.item
            _start, _length, chunk_begin, chunk_end = decode_work_row(
                params.work_table, row, self.whole_ranges
            )
            for _chunk in cutlass.range(chunk_begin, chunk_end, unroll=0):
                gate_handle = gate.wait()
                ready_handle = gate_ready.acquire()
                for index in cutlass.range(4, unroll_full=True):
                    key = gate_tid * 4 + index
                    gate_exp[(key, ready_handle.index)] = cute.math.exp2(
                        gate_data[(key, gate_handle.index)], fastmath=self.fastmath
                    )
                gate_handle.release()
                ready_handle.commit()

            work.advance()

        return gate, gate_ready, work

    @role(warps=6, regs=56)
    @cute.jit
    def run_mma(
        self,
        params: ReverseSummaryParams,
        *,
        state_pipe: Consumer,
        k: Consumer,
        dv_pipe: Producer,
        q: Consumer,
        do: Consumer,
        qdo_pipe: Producer,
        aqk: Consumer,
        dv_operand: Consumer,
        w: Consumer,
        wdv_pipe: Producer,
        work: WorkCursor,
        me: Local,
        tmem: Tmem,
    ):
        """Issue the four UMMA operations for each reverse chunk.

        The MMA warp alone patches each TMA stage after its full wait. ``zero_tail_tokens``
        fences every writer and synchronizes the warp before the elected UMMA issuer reads it;
        the handle's UMMA release completes those reads before the next TMA fill.

        MMA1 (``kg @ X``) and MMA4 (``w^T @ dv``) each run two accumulate passes over the
        hi/lo halves of their fp32-valued B operand. On a partial last chunk, ``kg``, ``qg^T``,
        ``dO``, ``Aqk^T``, and ``w^T`` are zeroed past the range boundary before UMMA reads them.
        Multiplying by a zero mask would retain NaN/Inf from uninitialized capacity slack.
        """
        mmas = params.mmas
        s_k = params.s_k
        s_q = params.s_q
        s_do = params.s_do
        s_aqk = params.s_aqk
        s_w = params.s_w
        t_dv_acc = cute.make_tensor(tmem.ptr("dv"), params.dv_layout)
        t_qdo_acc = cute.make_tensor(tmem.ptr("qdo"), params.qdo_layout)
        t_wdv_acc = cute.make_tensor(tmem.ptr("wdv"), params.wdv_layout)
        mma_dv, mma_qdo, mma_aqdo, mma_wdv = mmas
        t_k = mma_dv.make_fragment_A(params.s_k)
        t_state = mma_dv.make_fragment_B(params.s_state)
        t_q = mma_qdo.make_fragment_A(params.s_q)
        t_do = mma_qdo.make_fragment_B(params.s_do)
        t_aqk = mma_aqdo.make_fragment_A(params.s_aqk)
        t_do_aq = mma_aqdo.make_fragment_B(params.s_do)
        t_w = mma_wdv.make_fragment_A(params.s_w)
        t_dv = mma_wdv.make_fragment_B(params.s_dv)
        mma_tid = me.tid

        while work.valid:
            column_tile, _head, row = work.item
            _start, length, chunk_begin, chunk_end = decode_work_row(
                params.work_table, row, self.whole_ranges
            )
            has_source = column_tile < VAL_DIM // self.BN
            # The range's last chunk is the first one this range visits when it owns it.
            last_chunk, tail_valid_rows = range_tail(length)

            for chunk in cutlass.range(chunk_begin, chunk_end, unroll=0):
                reverse_chunk = chunk_end - 1 - (chunk - chunk_begin)
                is_tail = reverse_chunk == last_chunk and tail_valid_rows != 0
                # The source products do not depend on the state, so they run while
                # the state warps are still producing the X snapshot: Aqk^T dO seeds
                # the dv accumulator and qg^T dO completes off the recurrence chain.
                dv_handle = dv_pipe.acquire()
                if has_source:
                    q_handle = q.wait()
                    do_handle = do.wait()
                    aqk_handle = aqk.wait()
                    if is_tail:
                        zero_tail_tokens(
                            s_do, do_handle.index, tail_valid_rows, mma_tid, self.io_type, 1
                        )
                        zero_tail_tokens(
                            s_q, q_handle.index, tail_valid_rows, mma_tid, self.io_type, 1
                        )
                        zero_tail_tokens(
                            s_aqk, aqk_handle.index, tail_valid_rows, mma_tid, self.io_type, 1
                        )
                    for k_block in cutlass.range(cute.size(t_aqk, mode=[2]), unroll_full=True):
                        mma_aqdo.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(k_block != 0))
                        cute.gemm(
                            mma_aqdo,
                            t_dv_acc[None, None, None, dv_handle.index],
                            t_aqk[None, None, k_block, aqk_handle.index],
                            t_do_aq[None, None, k_block, do_handle.index],
                            t_dv_acc[None, None, None, dv_handle.index],
                        )
                    aqk_handle.release()

                    qdo_handle = qdo_pipe.acquire()
                    for k_block in cutlass.range(cute.size(t_q, mode=[2]), unroll_full=True):
                        mma_qdo.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(k_block != 0))
                        cute.gemm(
                            mma_qdo,
                            t_qdo_acc[None, None, None, qdo_handle.index],
                            t_q[None, None, k_block, q_handle.index],
                            t_do[None, None, k_block, do_handle.index],
                            t_qdo_acc[None, None, None, qdo_handle.index],
                        )
                    qdo_handle.commit()
                    q_handle.release()
                    do_handle.release()

                k_handle = k.wait()
                if is_tail:
                    zero_tail_tokens(
                        s_k, k_handle.index, tail_valid_rows, mma_tid, self.io_type, 0
                    )
                for split in cutlass.range_constexpr(2):
                    # Each hi/lo half is one pipeline stage, published separately.
                    state_handle = state_pipe.wait()
                    state_handle.access("State", read=True, by="umma")
                    for k_block in cutlass.range(cute.size(t_k, mode=[2]), unroll_full=True):
                        # The first pass overwrites unless Aqk^T dO already seeded dv.
                        mma_dv.set(
                            tcgen05.Field.ACCUMULATE,
                            cutlass.Boolean(split != 0) | (k_block != 0) | has_source,
                        )
                        cute.gemm(
                            mma_dv,
                            t_dv_acc[None, None, None, dv_handle.index],
                            t_k[None, None, k_block, k_handle.index],
                            t_state[None, None, k_block, state_handle.index],
                            t_dv_acc[None, None, None, dv_handle.index],
                        )
                    state_handle.release()
                k_handle.release()
                dv_handle.commit()

                w_handle = w.wait()
                if is_tail:
                    zero_tail_tokens(
                        s_w, w_handle.index, tail_valid_rows, mma_tid, self.io_type, 1
                    )
                wdv_handle = wdv_pipe.acquire()
                for split in cutlass.range_constexpr(2):
                    dv_operand_handle = dv_operand.wait()
                    dv_operand_handle.access("Dv", read=True, by="umma")
                    for k_block in cutlass.range(cute.size(t_w, mode=[2]), unroll_full=True):
                        mma_wdv.set(
                            tcgen05.Field.ACCUMULATE,
                            cutlass.Boolean(split != 0 or k_block != 0),
                        )
                        cute.gemm(
                            mma_wdv,
                            t_wdv_acc[None, None, None, wdv_handle.index],
                            t_w[None, None, k_block, w_handle.index],
                            t_dv[None, None, k_block, dv_operand_handle.index],
                            t_wdv_acc[None, None, None, wdv_handle.index],
                        )
                    dv_operand_handle.release()
                wdv_handle.commit()
                w_handle.release()

            work.advance()

        return state_pipe, k, dv_pipe, q, do, qdo_pipe, aqk, dv_operand, w, wdv_pipe, work

    @cute.kernel
    def kernel(
        self,
        mmas: Mmas,
        tma: TmaOps,
        work_table: cute.Tensor,
        out: cute.Tensor,
        smem_layouts: dict,
        scale: Float32,
        bound,
    ):
        """``smem_layouts`` maps each ``smem_plan`` buffer to its staged layout."""
        self.begin_roles(work=bound, layouts=smem_layouts)
        mma_dv, mma_qdo, _mma_aqdo, mma_wdv = mmas
        dv_fragment = mma_dv.make_fragment_C(
            cute.append(mma_dv.partition_shape_C(self.dv_tile[:2]), self.dv_acc_depth)
        )
        assert tcgen05.find_tmem_tensor_col_offset(dv_fragment) <= self.BN * self.dv_acc_depth
        qdo_fragment = mma_qdo.make_fragment_C(
            cute.append(mma_qdo.partition_shape_C(self.qdo_tile[:2]), self.qdo_acc_depth)
        )
        assert tcgen05.find_tmem_tensor_col_offset(qdo_fragment) <= self.BN * self.qdo_acc_depth
        wdv_fragment = mma_wdv.make_fragment_C(
            cute.append(mma_wdv.partition_shape_C(self.wdv_tile[:2]), self.wdv_acc_depth)
        )
        assert tcgen05.find_tmem_tensor_col_offset(wdv_fragment) <= self.BN * self.wdv_acc_depth
        params = ReverseSummaryParams(
            work_table=work_table,
            out=out,
            mmas=mmas,
            tma=tma,
            scale=scale,
            dv_layout=dv_fragment.layout,
            qdo_layout=qdo_fragment.layout,
            wdv_layout=wdv_fragment.layout,
            s_k=self.buffer("K"),
            s_state=self.buffer("State"),
            s_state_store=self.buffer("StateStore"),
            s_q=self.buffer("Q"),
            s_do=self.buffer("Do"),
            s_w=self.buffer("W"),
            s_dv=self.buffer("DvOperand"),
            s_dv_store=self.buffer("DvStore"),
            s_aqk=self.buffer("Aqk"),
            gate_3d=self.buffer("Gate"),
            gate=self.buffer("Gate")[(None, 0, None)],
            gate_exp=self.buffer("GateExp"),
        )
        self.run_roles(params)


@jit_cache
def _compile_affine_summary_rev(
    dtype_name: str,
    heads: int,
    use_int64_offsets: bool,
    whole_ranges: bool,
    fastmath: bool,
):
    """Compile one reverse-summary dtype/head specialization."""
    target = get_compile_target()
    if target.device_type != "cuda" or not runs_sm100_kernels(target.effective_capability):
        raise ValueError(f"affine_summary_rev requires an SM100-family target; got {target}")
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    tokens = sym_int()

    def factor(dtype, last_dim: int):
        return make_fake_compact_tensor(
            dtype,
            (1, tokens, heads, last_dim),
            stride_order=(3, 2, 1, 0),
            assumed_align=DATA_ALIGN_BYTES,
        )

    io_dtype = _CUTE_IO_TYPES[dtype_name]
    work = make_fake_compact_tensor(
        Int32, (sym_int(), 2 if whole_ranges else 4), stride_order=(1, 0)
    )
    out = make_fake_compact_tensor(
        Float32,
        (sym_int(), heads, SUMMARY_DIM, KEY_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    return compile_tvm_ffi(
        BlackwellDeltaAffineSummaryRev(heads, io_dtype, use_int64_offsets, whole_ranges, fastmath),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, VAL_DIM),
        factor(io_dtype, BT),
        factor(Float32, KEY_DIM),
        work,
        out,
        Float32(1.0),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
    )


@torch.compiler.disable
def build_state_grad_summaries(
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    dout: torch.Tensor,
    Aqk: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
    bounds: torch.Tensor,
    *,
    fastmath: bool | None = None,
) -> torch.Tensor:
    """Compute one packed reverse affine summary per token range of a stream.

    Args:
        qg: Gated queries, shape ``[1, T, H, 128]``, bf16 or fp16.
        kg: Gated keys, with the same shape and dtype as ``qg``.
        w: WY factor, with the same shape and dtype as ``qg``.
        dout: Output gradient, shape ``[1, T, H, 128]``, same dtype as ``qg``.
        Aqk: Intra-chunk query/key factor, shape ``[1, T, H, 64]``, same dtype as ``qg``.
        cumulative_gate: Cumulative log2 gates, shape ``[1, T, H, 128]``, fp32.
        scale: Query scaling factor used by the local backward recurrence.
        bounds: ``int32 [R, 2]`` device tensor of half-open token ranges ``[start, stop)``; the
            same contract as ``build_state_summaries``.
        fastmath: Select fast or non-fast per-chunk decay, matching the local backward.
            ``None`` selects fast Triton or non-fast SM100 exponentials.

    Returns:
        FP32 tensor of shape ``[R, H, 256, 128]``, V-first packed as local bias then reverse
        transition for ``dH_in = dH_out @ transition + bias``, one per row of ``bounds``.

    The ranges are read on the device, so the launch is CUDA Graph capturable and replayable
    across different ``bounds`` values of the same shape. The tokens right after a partial last
    chunk are read but dropped by selection, so they may hold anything. Supported scope: B=1,
    fixed K=V=128 and BT=64, contiguous CUDA inputs. SM100 uses the native UMMA kernel; other
    capability 8.0+ targets use Triton. Both backends split long ranges into work items when
    the static budget permits, then compose on the device.
    """
    assert qg.ndim == 4, f"qg must be 4D [1, T, H, K], got shape {tuple(qg.shape)}"
    batch, tokens, heads, key_dim = qg.shape
    assert batch == 1, f"build_state_grad_summaries requires B=1, got B={batch}"
    assert tokens > 0, "build_state_grad_summaries requires at least one token"
    assert heads > 0, "build_state_grad_summaries requires at least one head"
    assert key_dim == KEY_DIM, f"build_state_grad_summaries requires K={KEY_DIM}, got {key_dim}"
    for name, tensor in (("kg", kg), ("w", w), ("dout", dout)):
        assert tensor.shape == qg.shape, (
            f"{name} must match qg shape {tuple(qg.shape)}, got {tuple(tensor.shape)}"
        )
    assert Aqk.shape == (1, tokens, heads, BT), (
        f"Aqk must have shape {(1, tokens, heads, BT)}, got {tuple(Aqk.shape)}"
    )
    assert cumulative_gate.shape == qg.shape, (
        f"cumulative_gate must match qg shape {tuple(qg.shape)}, "
        f"got {tuple(cumulative_gate.shape)}"
    )
    assert qg.dtype in _IO_TYPE_NAMES, f"qg dtype must be bf16 or fp16, got {qg.dtype}"
    for name, tensor in (("kg", kg), ("w", w), ("dout", dout), ("Aqk", Aqk)):
        assert tensor.dtype == qg.dtype, (
            f"{name} dtype must match qg dtype {qg.dtype}, got {tensor.dtype}"
        )
    assert cumulative_gate.dtype == torch.float32, (
        f"cumulative_gate must be fp32, got {cumulative_gate.dtype}"
    )
    assert bounds.ndim == 2 and bounds.shape[1] == 2 and bounds.shape[0] > 0, (
        f"bounds must have shape [R, 2] with R > 0, got {tuple(bounds.shape)}"
    )
    assert bounds.dtype == torch.int32, f"bounds must be int32, got {bounds.dtype}"

    if isinstance(qg, FakeTensor):
        raise TypeError(
            "build_state_grad_summaries does not support torch.export; run the context-parallel "
            "summary eagerly or under CUDA Graph capture"
        )

    assert qg.is_cuda, "build_state_grad_summaries requires CUDA tensors"
    inputs = (
        ("qg", qg),
        ("kg", kg),
        ("w", w),
        ("dout", dout),
        ("Aqk", Aqk),
        ("cumulative_gate", cumulative_gate),
        ("bounds", bounds),
    )
    for name, tensor in inputs:
        assert tensor.device == qg.device, f"{name} must be on {qg.device}, got {tensor.device}"
        assert tensor.is_contiguous(), f"{name} must be contiguous, got strides {tensor.stride()}"
    ranges = bounds.shape[0]
    if ranges > MAX_RANGES:
        raise ValueError(f"at most {MAX_RANGES} ranges per launch, got {ranges}")
    properties = get_device_properties(qg.device)
    capability = (properties.major, properties.minor)
    if capability < (8, 0):
        raise ValueError(f"affine_summary_rev requires CUDA capability 8.0+, got {capability}")
    portable = not uses_legacy_cute(qg.device)
    if fastmath is None:
        fastmath = portable
    if portable:
        if capability[0] in (10, 12):
            from attn_gym._backends.triton.utils import configure_triton_allocator

            with torch.cuda.device(qg.device):
                configure_triton_allocator()
        from attn_gym.linear._delta_rule.triton.affine_summary_rev import (
            _select_block_columns,
            launch_affine_summary_rev,
        )

        state_bn = _select_block_columns(heads, capability)
    else:
        qg, kg, w, dout, Aqk, cumulative_gate = (
            _aligned(tensor) for tensor in (qg, kg, w, dout, Aqk, cumulative_gate)
        )
        state_bn = BlackwellDeltaAffineSummaryRev.BN
    # As in the forward: a static budget from the span's chunk count, ranges cut on the device.
    budget = plan_work_budget(
        cdiv(tokens, BT), heads * (SUMMARY_DIM // state_bn), properties.multi_processor_count
    )
    work, range_ids = work_table(bounds, budget)
    partials = torch.empty(
        (work.shape[0], heads, SUMMARY_DIM, KEY_DIM), dtype=torch.float32, device=qg.device
    )
    if portable:
        launch_affine_summary_rev(
            qg,
            kg,
            w,
            dout,
            Aqk,
            cumulative_gate,
            scale,
            work,
            partials,
            capability,
            whole_ranges=range_ids is None,
            fastmath=fastmath,
        )
    else:
        use_int64_offsets = requires_int64_abi(qg, kg, w, dout, Aqk, cumulative_gate, partials)
        compiled = _compile_affine_summary_rev(
            _IO_TYPE_NAMES[qg.dtype],
            heads,
            use_int64_offsets,
            whole_ranges=range_ids is None,
            fastmath=fastmath,
        )
        workspace, max_ctas = _LAUNCH_WORK.launch_args(qg.device)
        compiled(
            qg.detach(),
            kg.detach(),
            w.detach(),
            dout.detach(),
            Aqk.detach(),
            cumulative_gate.detach(),
            work,
            partials,
            Float32(scale),
            workspace,
            max_ctas,
        )
    # The cotangent flows through the later item first.
    return compose_work_items(partials, range_ids, ranges, reverse=True)
