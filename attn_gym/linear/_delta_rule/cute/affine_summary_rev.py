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

Each 224-thread CTA persistently owns augmented-column tiles. Four CUDA warps
hold the FP32 state in registers, one warp issues TMA loads, one computes gate
exponentials, and one issues the four UMMA operations. Bias tiles load the local
output-gradient sources; transition tiles skip those streams entirely. Only the
final FP32 summary is written to global memory.

The fp32-valued MMA B operands (state ``X`` and the corrected write gradient
``dv``) are split into hi/lo halves of the I/O dtype and accumulated in two UMMA
passes, exactly as the forward summary treats ``X`` and ``wx``, so the reverse
transition is the transpose of the map the forward summary actually evaluates.
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
from cutlass import Float32, Int32, cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Int64
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.cute import compile_tvm_ffi, get_device_properties, jit_cache
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear.kda.constants import is_sm100_kda_capability

BT = 64
KEY_DIM = 128
VAL_DIM = 128
SUMMARY_DIM = VAL_DIM + KEY_DIM
DATA_ALIGN_BYTES = 16

_IO_TYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_CUTE_IO_TYPES = {"bf16": cutlass.BFloat16, "fp16": cutlass.Float16}


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize the uncommon contiguous view that misses the TMA alignment."""
    return tensor if tensor.data_ptr() % DATA_ALIGN_BYTES == 0 else tensor.clone()


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
    """Warp-role boundaries in the persistent summary kernel."""

    STATE = 0
    LOAD = 4
    GATE = 5
    MMA = 6
    END = 7


class TmaOp(NamedTuple):
    """One TMA copy atom and its tensor-map descriptor."""

    atom: cute.CopyAtom
    desc: cute.Tensor


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


class SmemLayouts(NamedTuple):
    """Shared-memory layouts in device-kernel construction order."""

    k: cute.ComposedLayout
    state: cute.ComposedLayout
    state_store: cute.ComposedLayout
    q: cute.ComposedLayout
    do: cute.ComposedLayout
    w: cute.ComposedLayout
    dv: cute.ComposedLayout
    dv_store: cute.ComposedLayout
    aqk: cute.ComposedLayout


class BlackwellDeltaAffineSummaryRev:
    """Compute reverse affine summaries with persistent TMA/UMMA state tiles."""

    STATE_WARP_IDS = tuple(range(WarpRole.STATE, WarpRole.LOAD))
    WARP_SIZE = cute.arch.WARP_SIZE
    CTA_THREADS = WarpRole.END * WARP_SIZE

    def __init__(
        self,
        num_heads: int,
        io_type: type[cutlass.Numeric],
        state_tile_width: int,
        use_int64_offsets: bool,
    ):
        # This limits columns handled by one CTA, not the number of attention heads.
        assert state_tile_width in (16, 32), (
            f"state tile width must be 16 or 32, got {state_tile_width}"
        )
        self.num_heads = num_heads
        self.io_type = io_type
        self.acc_type = cutlass.Float32
        self.BT = BT
        self.BK = KEY_DIM
        self.BN = state_tile_width
        self.use_int64_offsets = use_int64_offsets

        self.state_regs = 128 if state_tile_width == 16 else 160
        self.aux_regs = 40

        self.dv_tile = (self.BT, self.BN, self.BK)
        self.qdo_tile = (self.BK, self.BN, self.BT)
        self.aqdo_tile = (self.BT, self.BN, self.BT)
        self.wdv_tile = (self.BK, self.BN, self.BT)

        self.k_depth = 6 if state_tile_width == 16 else 4
        self.q_depth = 2
        self.do_depth = 2
        self.w_depth = 2
        self.aqk_depth = 2
        self.gate_depth = 2
        self.dv_acc_depth = 1
        self.qdo_acc_depth = 1
        self.wdv_acc_depth = 1

        self.cluster = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tmem_free_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.CTA_THREADS,
        )
        self.smem_alignment = 1024

    def get_name(self) -> str:
        """Return a stable artifact and profiler name for this specialization."""
        dtype_name = "bf16" if self.io_type is cutlass.BFloat16 else "fp16"
        return (
            f"delta_affine_summary_rev_h{self.num_heads}_bn{self.BN}_{dtype_name}"
            f"_i64{int(self.use_int64_offsets)}"
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
        out: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        """Construct TMA/UMMA descriptors and launch the persistent kernel."""
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

        self.tm_dv, self.tm_qdo, self.tm_wdv, self.tm_total = self._plan_tmem(mmas)

        s_k = sm100_utils.make_smem_layout_a(
            mma_dv,
            self.dv_tile,
            self.io_type,
            self.k_depth,
        )
        # B operands hold hi/lo halves of the FP32 state and dv tiles; the two
        # "stages" are the two accumulate passes of one MMA, not a pipeline ring.
        s_state = sm100_utils.make_smem_layout_b(
            mma_dv,
            self.dv_tile,
            self.io_type,
            2,
        )
        s_state_store = sm100_utils.make_smem_layout_epi(
            self.io_type,
            utils.LayoutEnum.COL_MAJOR,
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
            utils.LayoutEnum.COL_MAJOR,
            (self.BT, self.BN),
            2,
        )
        s_aqk = sm100_utils.make_smem_layout_a(
            mma_aqdo,
            self.aqdo_tile,
            self.io_type,
            self.aqk_depth,
        )
        smem_layouts = SmemLayouts(
            s_k,
            s_state,
            s_state_store,
            s_q,
            s_do,
            s_w,
            s_dv,
            s_dv_store,
            s_aqk,
        )

        tma_load = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        cluster_layout = cute.tiled_divide(
            cute.make_layout(self.cluster),
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

        self.k_bytes = cute.size_in_bytes(self.io_type, s_k_one)
        self.q_bytes = cute.size_in_bytes(self.io_type, s_q_one)
        self.do_bytes = cute.size_in_bytes(self.io_type, s_do_one)
        self.w_bytes = cute.size_in_bytes(self.io_type, s_w_one)
        self.aqk_bytes = cute.size_in_bytes(self.io_type, s_aqk_one)
        self.gate_bytes = cute.size_in_bytes(Float32, s_gate)

        @cute.struct
        class SharedStorage:
            bar_k: cute.struct.MemRange[Int64, self.k_depth * 2]
            bar_q: cute.struct.MemRange[Int64, self.q_depth * 2]
            bar_do: cute.struct.MemRange[Int64, self.do_depth * 2]
            bar_w: cute.struct.MemRange[Int64, self.w_depth * 2]
            bar_aqk: cute.struct.MemRange[Int64, self.aqk_depth * 2]
            bar_gate: cute.struct.MemRange[Int64, self.gate_depth * 2]
            bar_gate_ready: cute.struct.MemRange[Int64, self.gate_depth * 2]
            bar_state: cute.struct.MemRange[Int64, 2]
            bar_dv: cute.struct.MemRange[Int64, self.dv_acc_depth * 2]
            bar_qdo: cute.struct.MemRange[Int64, self.qdo_acc_depth * 2]
            bar_dv_operand: cute.struct.MemRange[Int64, 2]
            bar_wdv: cute.struct.MemRange[Int64, self.wdv_acc_depth * 2]
            tmem_buf: Int32
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_k)], self.smem_alignment
            ]
            sState: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_state)], self.smem_alignment
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_q)], self.smem_alignment
            ]
            sDo: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_do)], self.smem_alignment
            ]
            sW: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_w)], self.smem_alignment
            ]
            sDv: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_dv)], self.smem_alignment
            ]
            sAqk: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_aqk)], self.smem_alignment
            ]
            sGate: cute.struct.Align[cute.struct.MemRange[Float32, self.BK * self.gate_depth], 128]
            sGateExp: cute.struct.Align[
                cute.struct.MemRange[Float32, self.BK * self.gate_depth], 128
            ]

        self.shared_type = SharedStorage
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            mmas,
            tma_ops,
            out,
            smem_layouts,
            Int32(cute.size(w.shape[1])),
            scale,
        ).launch(
            grid=self._launch_grid(),
            block=(self.CTA_THREADS, 1, 1),
            cluster=self.cluster,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _decode_work(self, work_index):
        """Return the augmented-column tile and head for a work item."""
        column_tiles = SUMMARY_DIM // self.BN
        return work_index % column_tiles, work_index // column_tiles

    @cute.jit
    def run_state(
        self,
        block,
        grid,
        iterations,
        chunks,
        out,
        t_dv_acc,
        t_qdo_acc,
        t_wdv_acc,
        s_state_store,
        s_dv_store,
        state_producer,
        dv_consumer,
        dv_operand_producer,
        gate_consumer,
        qdo_consumer,
        wdv_consumer,
        gate_exp,
        scale,
    ):
        """Keep one FP32 state tile in registers across the reverse chunk scan."""
        cute.arch.setmaxregister_increase(self.state_regs)
        tid, _, _ = cute.arch.thread_idx()
        local_tid = tid % (self.WARP_SIZE * len(self.STATE_WARP_IDS))

        t2r_dv_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BN // 8), tcgen05.Pack.NONE),
            self.acc_type,
        )
        dv_flat = t_dv_acc[((None, None), 0, 0, None)]
        t2r_dv = tcgen05.make_tmem_copy(t2r_dv_atom, dv_flat[(None, None, 0)])
        dv_slice = t2r_dv.get_slice(local_tid)
        partitioned_dv = dv_slice.partition_S(dv_flat)

        t2r_qdo_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BN // 8), tcgen05.Pack.NONE),
            self.acc_type,
        )
        qdo_flat = t_qdo_acc[((None, None), 0, 0, None)]
        t2r_qdo = tcgen05.make_tmem_copy(t2r_qdo_atom, qdo_flat[(None, None, 0)])
        qdo_slice = t2r_qdo.get_slice(local_tid)
        partitioned_qdo = qdo_slice.partition_S(qdo_flat)

        t2r_wdv_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(self.BN // 8), tcgen05.Pack.NONE),
            self.acc_type,
        )
        wdv_flat = t_wdv_acc[((None, None), 0, 0, None)]
        t2r_wdv = tcgen05.make_tmem_copy(t2r_wdv_atom, wdv_flat[(None, None, 0)])
        wdv_slice = t2r_wdv.get_slice(local_tid)
        partitioned_wdv = wdv_slice.partition_S(wdv_flat)

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

        r2s_state_atom = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.COL_MAJOR,
            self.io_type,
            self.acc_type,
            t2r_qdo,
        )
        r2s_state = cute.make_tiled_copy_D(r2s_state_atom, t2r_qdo)
        r2s_state_slice = r2s_state.get_slice(local_tid)

        r2s_dv_atom = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.COL_MAJOR,
            self.io_type,
            self.acc_type,
            t2r_dv,
        )
        r2s_dv = cute.make_tiled_copy_D(r2s_dv_atom, t2r_dv)
        r2s_dv_slice = r2s_dv.get_slice(local_tid)

        tile_index = Int32(0)
        has_work = tile_index < iterations
        while has_work:
            column_tile, head = self._decode_work(block + tile_index * grid)
            has_source = column_tile < VAL_DIM // self.BN

            for element in cutlass.range(cute.size(state), unroll_full=True):
                key, column = state_coordinates[element]
                global_column = column_tile * self.BN + column
                if global_column == VAL_DIM + key:
                    state[element] = Float32(1.0)
                else:
                    state[element] = Float32(0.0)

            for _chunk in cutlass.range(chunks, unroll=0):
                state_handle = state_producer.acquire_and_advance()
                state_hi = cute.make_rmem_tensor(state.shape, self.io_type)
                state_hi.store(state.load().to(self.io_type))
                state_lo = cute.make_rmem_tensor(state.shape, self.io_type)
                state_lo.store((state.load() - state_hi.load().to(Float32)).to(self.io_type))
                cute.copy(
                    r2s_state,
                    r2s_state.retile(state_hi),
                    r2s_state_slice.partition_D(s_state_store[(None, None, 0)]),
                )
                cute.copy(
                    r2s_state,
                    r2s_state.retile(state_lo),
                    r2s_state_slice.partition_D(s_state_store[(None, None, 1)]),
                )
                cute.arch.fence_view_async_shared()
                state_handle.commit()

                dv_handle = dv_consumer.wait_and_advance()
                dv = cute.make_rmem_tensor(dv_coordinates.shape, self.acc_type)
                cute.copy(
                    t2r_dv,
                    partitioned_dv[(None, None, None, dv_handle.index)],
                    dv,
                )
                cute.arch.fence_view_async_tmem_load()
                dv_handle.release()

                dv_operand_handle = dv_operand_producer.acquire_and_advance()
                dv_hi = cute.make_rmem_tensor(dv.shape, self.io_type)
                dv_hi.store(dv.load().to(self.io_type))
                dv_lo = cute.make_rmem_tensor(dv.shape, self.io_type)
                dv_lo.store((dv.load() - dv_hi.load().to(Float32)).to(self.io_type))
                cute.copy(
                    r2s_dv,
                    r2s_dv.retile(dv_hi),
                    r2s_dv_slice.partition_D(s_dv_store[(None, None, 0)]),
                )
                cute.copy(
                    r2s_dv,
                    r2s_dv.retile(dv_lo),
                    r2s_dv_slice.partition_D(s_dv_store[(None, None, 1)]),
                )
                cute.arch.fence_view_async_shared()
                dv_operand_handle.commit()

                gate_handle = gate_consumer.wait_and_advance()
                for element in cutlass.range(cute.size(state), unroll_full=True):
                    key, _column = state_coordinates[element]
                    state[element] = state[element] * gate_exp[(key, gate_handle.index)]
                gate_handle.release()

                qdo = cute.make_rmem_tensor(state_coordinates.shape, self.acc_type)
                if has_source:
                    qdo_handle = qdo_consumer.wait_and_advance()
                    cute.copy(
                        t2r_qdo,
                        partitioned_qdo[(None, None, None, qdo_handle.index)],
                        qdo,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    qdo_handle.release()

                wdv_handle = wdv_consumer.wait_and_advance()
                wdv = cute.make_rmem_tensor(wdv_coordinates.shape, self.acc_type)
                cute.copy(
                    t2r_wdv,
                    partitioned_wdv[(None, None, None, wdv_handle.index)],
                    wdv,
                )
                cute.arch.fence_view_async_tmem_load()
                wdv_handle.release()

                if has_source:
                    for element in cutlass.range_constexpr(cute.size(state)):
                        state[element] = state[element] + qdo[element] * scale - wdv[element]
                else:
                    for element in cutlass.range_constexpr(cute.size(state)):
                        state[element] = state[element] - wdv[element]

            for element in cutlass.range(cute.size(state), unroll_full=True):
                key, column = state_coordinates[element]
                out[head, column_tile * self.BN + column, key] = state[element]

            tile_index = tile_index + 1
            has_work = tile_index < iterations

    @cute.jit
    def run_load(
        self,
        block,
        grid,
        iterations,
        chunks,
        mma_dv,
        mma_qdo,
        mma_aqdo,
        mma_wdv,
        tma,
        s_k,
        s_q,
        s_do,
        s_w,
        s_aqk,
        s_gate,
        k_producer,
        q_producer,
        do_producer,
        w_producer,
        aqk_producer,
        gate_producer,
    ):
        """Issue reverse-order TMA loads for all active operand streams."""
        cute.arch.setmaxregister_decrease(self.aux_regs)
        atom_k, desc_k = tma.k
        atom_q, desc_q = tma.q
        atom_do, desc_do = tma.do
        atom_w, desc_w = tma.w
        atom_aqk, desc_aqk = tma.aqk
        atom_gate, desc_gate = tma.gate

        tile_index = Int32(0)
        has_work = tile_index < iterations
        while has_work:
            column_tile, head = self._decode_work(block + tile_index * grid)
            has_source = column_tile < VAL_DIM // self.BN

            k_smem, k_gmem = self._partition_a(atom_k, desc_k, s_k, self.dv_tile, mma_dv, head)
            w_smem, w_gmem = self._partition_a(
                atom_w,
                desc_w,
                s_w,
                self.wdv_tile,
                mma_wdv,
                head,
            )
            gate_smem, gate_gmem = self._partition_epilogue(
                atom_gate,
                desc_gate[None, None, (head, 0)],
                (self.BK, 1),
                s_gate,
            )
            q_smem, q_gmem = self._partition_a(
                atom_q,
                desc_q,
                s_q,
                self.qdo_tile,
                mma_qdo,
                head,
            )
            do_smem, do_gmem = self._partition_b(
                atom_do,
                desc_do,
                s_do,
                self.qdo_tile,
                mma_qdo,
                head,
            )
            aqk_smem, aqk_gmem = self._partition_a(
                atom_aqk,
                desc_aqk,
                s_aqk,
                self.aqdo_tile,
                mma_aqdo,
                head,
            )

            for chunk in cutlass.range(chunks, unroll=0):
                reverse_chunk = chunks - 1 - chunk

                gate_handle = gate_producer.acquire_and_advance()
                cute.copy(
                    atom=atom_gate,
                    src=gate_gmem[(None, 0, reverse_chunk * self.BT + self.BT - 1)],
                    dst=gate_smem[None, gate_handle.index],
                    tma_bar_ptr=gate_handle.barrier,
                )

                k_handle = k_producer.acquire_and_advance()
                cute.copy(
                    atom=atom_k,
                    src=k_gmem[None, reverse_chunk, 0],
                    dst=k_smem[None, k_handle.index],
                    tma_bar_ptr=k_handle.barrier,
                )
                w_handle = w_producer.acquire_and_advance()
                cute.copy(
                    atom=atom_w,
                    src=w_gmem[None, 0, reverse_chunk],
                    dst=w_smem[None, w_handle.index],
                    tma_bar_ptr=w_handle.barrier,
                )

                if has_source:
                    q_handle = q_producer.acquire_and_advance()
                    cute.copy(
                        atom=atom_q,
                        src=q_gmem[None, 0, reverse_chunk],
                        dst=q_smem[None, q_handle.index],
                        tma_bar_ptr=q_handle.barrier,
                    )
                    do_handle = do_producer.acquire_and_advance()
                    cute.copy(
                        atom=atom_do,
                        src=do_gmem[None, column_tile, reverse_chunk],
                        dst=do_smem[None, do_handle.index],
                        tma_bar_ptr=do_handle.barrier,
                    )
                    aqk_handle = aqk_producer.acquire_and_advance()
                    cute.copy(
                        atom=atom_aqk,
                        src=aqk_gmem[None, 0, reverse_chunk],
                        dst=aqk_smem[None, aqk_handle.index],
                        tma_bar_ptr=aqk_handle.barrier,
                    )

            tile_index = tile_index + 1
            has_work = tile_index < iterations

    @cute.jit
    def run_gate(
        self,
        block,
        grid,
        iterations,
        chunks,
        gate,
        gate_exp,
        gate_consumer,
        gate_ready_producer,
    ):
        """Transform cumulative log2 gate stages into multiplicative decays."""
        cute.arch.setmaxregister_decrease(self.aux_regs)
        tid, _, _ = cute.arch.thread_idx()
        gate_tid = tid - WarpRole.GATE * self.WARP_SIZE

        tile_index = Int32(0)
        has_work = tile_index < iterations
        while has_work:
            for _chunk in cutlass.range(chunks, unroll=0):
                gate_handle = gate_consumer.wait_and_advance()
                ready_handle = gate_ready_producer.acquire_and_advance()
                for index in cutlass.range(4, unroll_full=True):
                    key = gate_tid * 4 + index
                    gate_exp[(key, ready_handle.index)] = cute.exp2(gate[(key, gate_handle.index)])
                gate_handle.release()
                cute.arch.fence_view_async_shared()
                ready_handle.commit()

            tile_index = tile_index + 1
            has_work = tile_index < iterations

    @cute.jit
    def run_mma(
        self,
        block,
        grid,
        iterations,
        chunks,
        mmas,
        t_dv_acc,
        t_k,
        t_state,
        t_qdo_acc,
        t_q,
        t_do,
        t_aqk,
        t_do_aq,
        t_wdv_acc,
        t_w,
        t_dv,
        state_consumer,
        k_consumer,
        dv_producer,
        q_consumer,
        do_consumer,
        qdo_producer,
        aqk_consumer,
        dv_operand_consumer,
        w_consumer,
        wdv_producer,
    ):
        """Issue the four UMMA operations for each reverse chunk.

        MMA1 (``kg @ X``) and MMA4 (``w^T @ dv``) each run two accumulate passes
        over the hi/lo halves of their fp32-valued B operand.
        """
        cute.arch.setmaxregister_decrease(self.aux_regs)
        mma_dv, mma_qdo, mma_aqdo, mma_wdv = mmas

        tile_index = Int32(0)
        has_work = tile_index < iterations
        while has_work:
            column_tile, _head = self._decode_work(block + tile_index * grid)
            has_source = column_tile < VAL_DIM // self.BN

            for _chunk in cutlass.range(chunks, unroll=0):
                state_handle = state_consumer.wait_and_advance()
                k_handle = k_consumer.wait_and_advance()
                dv_handle = dv_producer.acquire_and_advance()
                for split in cutlass.range_constexpr(2):
                    for k_block in cutlass.range(cute.size(t_k, mode=[2]), unroll_full=True):
                        mma_dv.set(
                            tcgen05.Field.ACCUMULATE,
                            cutlass.Boolean(split != 0 or k_block != 0),
                        )
                        cute.gemm(
                            mma_dv,
                            t_dv_acc[None, None, None, dv_handle.index],
                            t_k[None, None, k_block, k_handle.index],
                            t_state[None, None, k_block, split],
                            t_dv_acc[None, None, None, dv_handle.index],
                        )
                k_handle.release()
                state_handle.release()

                if has_source:
                    q_handle = q_consumer.wait_and_advance()
                    do_handle = do_consumer.wait_and_advance()
                    aqk_handle = aqk_consumer.wait_and_advance()
                    for k_block in cutlass.range(cute.size(t_aqk, mode=[2]), unroll_full=True):
                        mma_aqdo.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
                        cute.gemm(
                            mma_aqdo,
                            t_dv_acc[None, None, None, dv_handle.index],
                            t_aqk[None, None, k_block, aqk_handle.index],
                            t_do_aq[None, None, k_block, do_handle.index],
                            t_dv_acc[None, None, None, dv_handle.index],
                        )
                    aqk_handle.release()
                    dv_handle.commit()

                    qdo_handle = qdo_producer.acquire_and_advance()
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
                else:
                    dv_handle.commit()

                dv_operand_handle = dv_operand_consumer.wait_and_advance()
                w_handle = w_consumer.wait_and_advance()
                wdv_handle = wdv_producer.acquire_and_advance()
                for split in cutlass.range_constexpr(2):
                    for k_block in cutlass.range(cute.size(t_w, mode=[2]), unroll_full=True):
                        mma_wdv.set(
                            tcgen05.Field.ACCUMULATE,
                            cutlass.Boolean(split != 0 or k_block != 0),
                        )
                        cute.gemm(
                            mma_wdv,
                            t_wdv_acc[None, None, None, wdv_handle.index],
                            t_w[None, None, k_block, w_handle.index],
                            t_dv[None, None, k_block, split],
                            t_wdv_acc[None, None, None, wdv_handle.index],
                        )
                wdv_handle.commit()
                w_handle.release()
                dv_operand_handle.release()

            tile_index = tile_index + 1
            has_work = tile_index < iterations

    @cute.kernel
    def kernel(
        self,
        mmas: Mmas,
        tma: TmaOps,
        out: cute.Tensor,
        smem_layouts: SmemLayouts,
        tokens: Int32,
        scale: Float32,
    ):
        """Dispatch the persistent state, TMA, gate, and MMA warp roles."""
        mma_dv, mma_qdo, mma_aqdo, mma_wdv = mmas
        (
            s_k_layout,
            s_state_layout,
            s_state_store_layout,
            s_q_layout,
            s_do_layout,
            s_w_layout,
            s_dv_layout,
            s_dv_store_layout,
            s_aqk_layout,
        ) = smem_layouts
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_id == WarpRole.LOAD:
            cpasync.prefetch_descriptor(tma.k.atom)
            cpasync.prefetch_descriptor(tma.q.atom)
            cpasync.prefetch_descriptor(tma.do.atom)
            cpasync.prefetch_descriptor(tma.w.atom)
            cpasync.prefetch_descriptor(tma.aqk.atom)
            cpasync.prefetch_descriptor(tma.gate.atom)

        allocator = utils.SmemAllocator()
        storage = allocator.allocate(self.shared_type)

        gate_3d = storage.sGate.get_tensor(
            cute.make_layout(
                (self.BK, 1, self.gate_depth),
                stride=(1, self.BK, self.BK),
            )
        )
        gate = gate_3d[(None, 0, None)]
        gate_exp = storage.sGateExp.get_tensor(cute.make_layout((self.BK, self.gate_depth)))

        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.k_bytes,
            barrier_storage=storage.bar_k.data_ptr(),
        ).make_participants()
        # X snapshot → MMA1 (AsyncUmma, 1 stage covering both hi/lo halves)
        state_producer, state_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARP_SIZE * len(self.STATE_WARP_IDS),
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=storage.bar_state.data_ptr(),
        ).make_participants()
        dv_producer, dv_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.dv_acc_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARP_SIZE * len(self.STATE_WARP_IDS),
            ),
            barrier_storage=storage.bar_dv.data_ptr(),
        ).make_participants()
        q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.q_bytes,
            barrier_storage=storage.bar_q.data_ptr(),
        ).make_participants()
        do_producer, do_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.do_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.do_bytes,
            barrier_storage=storage.bar_do.data_ptr(),
        ).make_participants()
        qdo_producer, qdo_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qdo_acc_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARP_SIZE * len(self.STATE_WARP_IDS),
            ),
            barrier_storage=storage.bar_qdo.data_ptr(),
        ).make_participants()
        w_producer, w_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.w_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.w_bytes,
            barrier_storage=storage.bar_w.data_ptr(),
        ).make_participants()
        # dv snapshot → MMA4 (AsyncUmma, 1 stage covering both hi/lo halves)
        dv_operand_producer, dv_operand_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARP_SIZE * len(self.STATE_WARP_IDS),
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=storage.bar_dv_operand.data_ptr(),
        ).make_participants()
        wdv_producer, wdv_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.wdv_acc_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARP_SIZE * len(self.STATE_WARP_IDS),
            ),
            barrier_storage=storage.bar_wdv.data_ptr(),
        ).make_participants()
        aqk_producer, aqk_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.aqk_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.aqk_bytes,
            barrier_storage=storage.bar_aqk.data_ptr(),
        ).make_participants()
        gate_producer, gate_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.gate_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.gate_bytes,
            barrier_storage=storage.bar_gate.data_ptr(),
        ).make_participants()
        gate_ready_producer, gate_ready_consumer = pipeline.PipelineAsync.create(
            num_stages=self.gate_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.WARP_SIZE),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.WARP_SIZE * len(self.STATE_WARP_IDS),
            ),
            barrier_storage=storage.bar_gate_ready.data_ptr(),
        ).make_participants()

        tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.CTA_THREADS)
        tmem = utils.TmemAllocator(
            storage.tmem_buf,
            barrier_for_retrieve=tmem_barrier,
            allocator_warp_id=WarpRole.LOAD,
        )
        tmem.allocate(self.tm_total)
        tmem.wait_for_alloc()
        tmem_pointer = tmem.retrieve_ptr(self.acc_type)

        s_k = storage.sK.get_tensor(s_k_layout.outer, swizzle=s_k_layout.inner)
        s_state = storage.sState.get_tensor(
            s_state_layout.outer,
            swizzle=s_state_layout.inner,
        )
        s_state_store = storage.sState.get_tensor(
            s_state_store_layout.outer,
            swizzle=s_state_store_layout.inner,
        )
        s_q = storage.sQ.get_tensor(s_q_layout.outer, swizzle=s_q_layout.inner)
        s_do = storage.sDo.get_tensor(s_do_layout.outer, swizzle=s_do_layout.inner)
        s_w = storage.sW.get_tensor(s_w_layout.outer, swizzle=s_w_layout.inner)
        s_dv = storage.sDv.get_tensor(s_dv_layout.outer, swizzle=s_dv_layout.inner)
        s_dv_store = storage.sDv.get_tensor(
            s_dv_store_layout.outer,
            swizzle=s_dv_store_layout.inner,
        )
        s_aqk = storage.sAqk.get_tensor(s_aqk_layout.outer, swizzle=s_aqk_layout.inner)

        t_k = mma_dv.make_fragment_A(s_k)
        t_state = mma_dv.make_fragment_B(s_state)
        dv_shape = mma_dv.partition_shape_C(self.dv_tile[:2])
        dv_fragment = mma_dv.make_fragment_C(cute.append(dv_shape, self.dv_acc_depth))
        t_dv_acc = cute.make_tensor(tmem_pointer + self.tm_dv, dv_fragment.layout)

        t_q = mma_qdo.make_fragment_A(s_q)
        t_do = mma_qdo.make_fragment_B(s_do)
        qdo_shape = mma_qdo.partition_shape_C(self.qdo_tile[:2])
        qdo_fragment = mma_qdo.make_fragment_C(cute.append(qdo_shape, self.qdo_acc_depth))
        t_qdo_acc = cute.make_tensor(tmem_pointer + self.tm_qdo, qdo_fragment.layout)

        t_aqk = mma_aqdo.make_fragment_A(s_aqk)
        t_do_aq = mma_aqdo.make_fragment_B(s_do)

        t_w = mma_wdv.make_fragment_A(s_w)
        t_dv = mma_wdv.make_fragment_B(s_dv)
        wdv_shape = mma_wdv.partition_shape_C(self.wdv_tile[:2])
        wdv_fragment = mma_wdv.make_fragment_C(cute.append(wdv_shape, self.wdv_acc_depth))
        t_wdv_acc = cute.make_tensor(tmem_pointer + self.tm_wdv, wdv_fragment.layout)

        block = cute.arch.block_idx()[0]
        grid = cute.arch.grid_dim()[0]
        work_tiles = self.num_heads * (SUMMARY_DIM // self.BN)
        iterations = (work_tiles - block + grid - 1) // grid
        chunks = tokens // self.BT

        if warp_id in self.STATE_WARP_IDS:
            self.run_state(
                block,
                grid,
                iterations,
                chunks,
                out,
                t_dv_acc,
                t_qdo_acc,
                t_wdv_acc,
                s_state_store,
                s_dv_store,
                state_producer,
                dv_consumer,
                dv_operand_producer,
                gate_ready_consumer,
                qdo_consumer,
                wdv_consumer,
                gate_exp,
                scale,
            )
        elif warp_id == WarpRole.LOAD:
            self.run_load(
                block,
                grid,
                iterations,
                chunks,
                mma_dv,
                mma_qdo,
                mma_aqdo,
                mma_wdv,
                tma,
                s_k,
                s_q,
                s_do,
                s_w,
                s_aqk,
                gate_3d,
                k_producer,
                q_producer,
                do_producer,
                w_producer,
                aqk_producer,
                gate_producer,
            )
        elif warp_id == WarpRole.GATE:
            self.run_gate(
                block,
                grid,
                iterations,
                chunks,
                gate,
                gate_exp,
                gate_consumer,
                gate_ready_producer,
            )
        elif warp_id == WarpRole.MMA:
            self.run_mma(
                block,
                grid,
                iterations,
                chunks,
                mmas,
                t_dv_acc,
                t_k,
                t_state,
                t_qdo_acc,
                t_q,
                t_do,
                t_aqk,
                t_do_aq,
                t_wdv_acc,
                t_w,
                t_dv,
                state_consumer,
                k_consumer,
                dv_producer,
                q_consumer,
                do_consumer,
                qdo_producer,
                aqk_consumer,
                dv_operand_consumer,
                w_consumer,
                wdv_producer,
            )

        tmem.relinquish_alloc_permit()
        self.tmem_free_barrier.arrive_and_wait()
        tmem.free(tmem_pointer)

    @cute.jit
    def _partition_a(self, atom, desc, smem, tile, mma, head):
        """Partition a dense TMA tensor as an SS-mode MMA A operand."""
        gmem = cute.local_tile(desc, cute.slice_(tile, (None, 0, None)), (None, None, (head, 0)))
        partitioned = mma.get_slice(0).partition_A(gmem)
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(partitioned, 0, 3),
        )

    @cute.jit
    def _partition_b(self, atom, desc, smem, tile, mma, head):
        """Partition a dense TMA tensor as an SS-mode MMA B operand."""
        gmem = cute.local_tile(desc, cute.slice_(tile, (0, None, None)), (None, None, (head, 0)))
        partitioned = mma.get_slice(0).partition_B(gmem)
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 3),
            cute.group_modes(partitioned, 0, 3),
        )

    @cute.jit
    def _partition_epilogue(self, atom, gmem, tile, smem):
        """Partition a simple two-dimensional TMA tile."""
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(smem, 0, 2),
            cute.group_modes(cute.flat_divide(gmem, tile), 0, 2),
        )

    def _plan_tmem(self, mmas: Mmas):
        """Assign disjoint TMEM column regions to the three accumulators."""
        mma_dv, mma_qdo, _mma_aqdo, mma_wdv = mmas
        dv_shape = mma_dv.partition_shape_C(self.dv_tile[:2])
        dv_columns = tcgen05.find_tmem_tensor_col_offset(
            mma_dv.make_fragment_C(cute.append(dv_shape, self.dv_acc_depth))
        )
        qdo_shape = mma_qdo.partition_shape_C(self.qdo_tile[:2])
        qdo_columns = tcgen05.find_tmem_tensor_col_offset(
            mma_qdo.make_fragment_C(cute.append(qdo_shape, self.qdo_acc_depth))
        )
        wdv_shape = mma_wdv.partition_shape_C(self.wdv_tile[:2])
        wdv_columns = tcgen05.find_tmem_tensor_col_offset(
            mma_wdv.make_fragment_C(cute.append(wdv_shape, self.wdv_acc_depth))
        )
        dv_offset = 0
        qdo_offset = dv_columns
        wdv_offset = qdo_offset + qdo_columns
        required = wdv_offset + wdv_columns
        total = 1
        while total < required:
            total *= 2
        assert total <= 512, f"TMEM overflow: {total}>512"
        return dv_offset, qdo_offset, wdv_offset, total

    def _launch_grid(self):
        """Bound the persistent launch to the logical tile count and SM count."""
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("affine_summary_rev compilation requires an SM count")
        return (min(sm_count, self.num_heads * (SUMMARY_DIM // self.BN)), 1, 1)


def select_summary_tile_width(heads: int, device: torch.device) -> int:
    """Choose the per-CTA state-column width from the available SM count."""
    work_tiles_at_width_16 = (SUMMARY_DIM // 16) * heads
    sm_count = get_device_properties(device).multi_processor_count
    return 32 if work_tiles_at_width_16 > sm_count else 16


@jit_cache
def _compile_affine_summary_rev(
    dtype_name: str,
    heads: int,
    state_tile_width: int,
    use_int64_offsets: bool,
):
    """Compile one reverse-summary dtype/head/tile specialization."""
    target = get_compile_target()
    if target.device_type != "cuda" or not is_sm100_kda_capability(target.effective_capability):
        raise ValueError(f"affine_summary_rev requires an SM100/SM103 target; got {target}")
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    tokens = sym_int(divisibility=BT)

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
        BlackwellDeltaAffineSummaryRev(
            heads,
            io_dtype,
            state_tile_width,
            use_int64_offsets,
        ),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, VAL_DIM),
        factor(io_dtype, BT),
        factor(Float32, KEY_DIM),
        out,
        Float32(1.0),
    )


@torch.compiler.disable
def build_state_grad_summary(
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    dout: torch.Tensor,
    Aqk: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compute one shard's packed reverse affine summary.

    Args:
        qg: Gated queries, shape ``[1, T, H, 128]``, bf16 or fp16.
        kg: Gated keys, with the same shape and dtype as ``qg``.
        w: WY factor, with the same shape and dtype as ``qg``.
        dout: Output gradient, shape ``[1, T, H, 128]``, same dtype as ``qg``.
        Aqk: Intra-chunk query/key factor, shape ``[1, T, H, 64]``, same dtype as ``qg``.
        cumulative_gate: Cumulative log2 gates, shape ``[1, T, H, 128]``, fp32.
        scale: Query scaling factor used by the local backward recurrence.

    Returns:
        FP32 tensor of shape ``[H, 256, 128]``, V-first packed as local bias then
        reverse transition for ``dH_in = dH_out @ transition + bias``.

    Supported scope: B=1, fixed K=V=128 and BT=64, contiguous CUDA inputs.
    SM100 uses the native UMMA kernel; other capability 8.0+ targets use Triton.
    A partial final chunk is neutral-padded by the wrapper.
    """
    assert qg.ndim == 4, f"qg must be 4D [1, T, H, K], got shape {tuple(qg.shape)}"
    batch, tokens, heads, key_dim = qg.shape
    assert batch == 1, f"build_state_grad_summary requires B=1, got B={batch}"
    assert tokens > 0, "build_state_grad_summary requires at least one token"
    assert heads > 0, "build_state_grad_summary requires at least one head"
    assert key_dim == KEY_DIM, f"build_state_grad_summary requires K={KEY_DIM}, got {key_dim}"
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

    out = torch.empty(
        (heads, SUMMARY_DIM, KEY_DIM),
        dtype=torch.float32,
        device=qg.device,
    )
    if isinstance(qg, FakeTensor):
        raise TypeError(
            "build_state_grad_summary does not support torch.export; run the context-parallel "
            "summary eagerly or under CUDA Graph capture"
        )

    assert qg.is_cuda, "build_state_grad_summary requires CUDA tensors"
    inputs = (
        ("qg", qg),
        ("kg", kg),
        ("w", w),
        ("dout", dout),
        ("Aqk", Aqk),
        ("cumulative_gate", cumulative_gate),
    )
    for name, tensor in inputs:
        assert tensor.device == qg.device, f"{name} must be on {qg.device}, got {tensor.device}"
        assert tensor.is_contiguous(), f"{name} must be contiguous, got strides {tensor.stride()}"
    pad = (-tokens) % BT
    if pad:
        padding = (0, 0, 0, 0, 0, pad)
        qg, kg, w, dout, Aqk = (F.pad(tensor, padding) for tensor in (qg, kg, w, dout, Aqk))
        cumulative_gate = torch.cat(
            (
                cumulative_gate,
                cumulative_gate[:, -1:].expand(-1, pad, -1, -1),
            ),
            dim=1,
        )
    properties = get_device_properties(qg.device)
    capability = (properties.major, properties.minor)
    if capability < (8, 0):
        raise ValueError(f"affine_summary_rev requires CUDA capability 8.0+, got {capability}")
    if not is_sm100_kda_capability(capability):
        if capability[0] in (10, 12):
            from attn_gym._backends.triton.utils import configure_triton_allocator

            with torch.cuda.device(qg.device):
                configure_triton_allocator()
        from attn_gym.linear._delta_rule.triton.affine_summary_rev import (
            launch_affine_summary_rev,
        )

        launch_affine_summary_rev(
            qg,
            kg,
            w,
            dout,
            Aqk,
            cumulative_gate,
            scale,
            out,
            capability,
        )
        return out

    qg, kg, w, dout, Aqk, cumulative_gate = (
        _aligned(tensor) for tensor in (qg, kg, w, dout, Aqk, cumulative_gate)
    )
    use_int64_offsets = requires_int64_abi(qg, kg, w, dout, Aqk, cumulative_gate, out)

    state_tile_width = select_summary_tile_width(heads, qg.device)
    compiled = _compile_affine_summary_rev(
        _IO_TYPE_NAMES[qg.dtype], heads, state_tile_width, use_int64_offsets
    )
    compiled(
        qg.detach(),
        kg.detach(),
        w.detach(),
        dout.detach(),
        Aqk.detach(),
        cumulative_gate.detach(),
        out,
        Float32(scale),
    )
    return out
