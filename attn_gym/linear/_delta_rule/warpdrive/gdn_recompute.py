# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# gdn_recompute_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and
# channels with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence
# descriptor arrays, and only the checkpoint-series form the backward drivers compile remains.

"""GDN backward recompute for SM100: the checkpoint series, the recurrent state at the start of
every 64-token chunk, which the GDN bprop reads.

Per chunk ``c`` of a work item (tokens ``[c * 64, (c + 1) * 64)`` of its sequence, one value
head ``h``; ``S`` is the ``d_v x d_k`` fp32 state, ``g`` the gate's cumulative log2 sum over the
chunk, ``Tinv`` the ``gdn_tinv`` tile)::

    checkpoint[c] = fp16(S)                                    (state entering the chunk)
    Y   = V - fp16((K S^T) * exp2(g))                          (packed 16-bit)
    U   = Tinv @ Y
    S   = exp2(g[63]) * S + (fp16(U * exp2(g[63] - g)))^T @ K

Work items are rows of the bprop's work table ``(batch, head, write_start, write_end,
compute_start, ...)``: the chunks ``[compute_start, write_end)`` run, the checkpoints of
``[write_start, write_end)`` are stored. The state starts from ``state_in[batch, head]`` when
``use_initial_state`` and ``compute_start == 0``, else from zero (a split item's warmup); with
no initial state, chunk 0's checkpoint is zero.

Layouts: ``checkpoints[row, head, v, k]`` holds chunk ``c`` of sequence ``b`` at ``row =
cu[b] // 64 + b + c``, the row of its ``Tinv`` tile in ``tinv[row, head]``: closed-form, so
neither the recompute nor the bprop scans ``cu_seqlens``. The rows between one sequence's last
chunk and the next base are padding; the series has ``checkpoint_rows(tokens, sequences)``
rows.

K and V rows past a chunk's sequence end load as zeros (``ragged.RaggedTma``), as from
upstream's per-sequence descriptors. Gates there are neutral (zero log) and the ``Tinv`` tile
has zero rows and columns there, so those tokens' ``Y`` and ``U`` are zero and the state update
adds nothing for them.

Work order is upstream's: CTA ``u`` of ``G`` (one OneShot item each) starts at table row ``u``
and claims later rows from the driver's zeroed scheduler counter, ``G + atomicAdd``, which the
load warp publishes to every role through ``sched``.

Roles and channels (8 warps)::

    load (5)     --k: 3 x K (TMA)-->            mma (6)  (KS and KV GEMMs; released after KV)
    load         --tinv: 3 x Tinv (TMA)-->      mma      (U GEMM)
    load         --v: 2 x V (TMA)-->            compute (0-3)
    load         --sched: 2 x next row-->       gate, mma, compute, store
    gate (4)     --gate: 3 x (cumsum, cumprod)--> compute
    compute      --state_in / y / du (TMEM)-->  mma      (the KS, U and KV A operands)
    mma          --ks / u (TMEM)-->             compute  (the KS and U accumulators)
    mma          --state (TMEM)-->              compute  (KV GEMM complete)

    compute      --ckpt: 1 x fp16 state-->      store (7): TMA store to checkpoints

Each chunk's GEMMs and compute steps form one serial chain (KV -> restage -> KS -> Y -> U ->
dU -> KV), and the kernel's speed is that chain's latency. The six TMEM handoffs use one-way
``sig`` barriers: each region is reused only after a later handoff in that same chain (see
``SIGNALS`` for the ordering arguments). A second full/empty handshake adds latency without
adding ordering. The decay factors are computed after ``y`` is published so they fill the U
GEMM's latency, as upstream's schedule does.

Upstream's table invariant holds: an item with ``write_start > 0`` has ``compute_start <
write_start`` (the split tables' warmups are at least one chunk); otherwise its first checkpoint
would never be produced and the store warp would wait for it forever, in either kernel.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync

from attn_gym.linear._delta_rule.wd_support import HEAD_DIMS, IO_DTYPE_NAMES
from attn_gym.warpdrive import (
    Barriers,
    Consumer,
    Local,
    OneShot,
    Producer,
    Ring,
    RoleConfig,
    SmemPlan,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    bulk_commit_group,
    bulk_wait_group,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_strided_tensor
from attn_gym.warpdrive.compat import LayoutEnum
from attn_gym.warpdrive.protocol import current_trace
from attn_gym.warpdrive.thread_sync import tcgen05_after_thread_sync, tcgen05_before_thread_sync
from attn_gym.warpdrive.tracing import wait_range

from .contract import tma_divisibility
from .ptx import (
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    opaque_f32_zero,
    shfl_idx,
    shfl_up,
    st_shared_v4,
    sub_f16x2,
    swizzle_xor_128b,
    tmem_ld,
    tmem_st,
)
from .ragged import (
    RaggedTma,
    fake_tma_slots,
    make_ragged_tma,
    new_tma_slots,
    ragged_load,
    sequence_descriptors,
    slot_region,
    tma_slot_rows,
)
from .umma import SWIZZLE_128B, MmaDesc, desc_opaque, instr_desc, mma_ts_step, smem_desc

B_T = 64
RCP_LN2 = 1.4426950408889634  # 1 / ln 2: natural-log gates -> the kernel's log2 domain
BOX = 64  # 16-bit elements per 128-byte swizzle row
BOX_BYTES = B_T * BOX * 2  # one 64 x 64 16-bit box
K_STAGES = 3
V_STAGES = 2
TINV_STAGES = 3
GATE_STAGES = 3
SCHED_STAGES = 2

# Producer -> consumer, with the later handoff that licenses reuse:
#   state_in: compute -> MMA. Restaged after the previous KV completes, hence after KS read it.
#   y: compute -> MMA. Rewritten after the previous KV, which follows U reading Y.
#   du: compute -> MMA. Rewritten after the previous KV reading dU completes.
#   ks: MMA -> compute. Rewritten after the next state_in, published after compute read KS
#       and completed Y -> U -> dU -> KV -> state for the previous chunk.
#   u: MMA -> compute. Rewritten after the next y, whose publication follows that same chain.
#   state: MMA -> compute. Rescaled/restaged after KV completion; the next KV waits for du,
#       published after the rescale's TMEM-store fence.
# KS/U loads precede the same warp's Y/dU TMEM stores; the tcgen05.ld -> st pipeline and the
# wait::st before publication order those reads before the accumulators are reused (a register
# dependency alone would not order memory). No separate tcgen05.wait::ld belongs on this chain;
# a path that publishes without those stores and their wait cannot reuse this argument.
# Each item's final state wait drains KV before the next item can seed or reuse any columns.
# Parities belong to the physical CTA and carry across items, including conditional KS steps.
SIGNALS = (("state_in", 128), ("y", 128), ("du", 128), ("ks", 1), ("u", 1), ("state", 1))
SIG = {name: i for i, (name, _) in enumerate(SIGNALS)}


@cute.jit
def _wait(bars, phases, name: cutlass.Constexpr[str]):
    """Wait on one recurrence signal and flip only its parity, with a measured wait range."""
    i = SIG[name]
    with wait_range(current_trace(), f"{name}.wait"):
        cute.arch.mbarrier_wait(bars + i, Int32((phases >> i) & 1))
    tcgen05_after_thread_sync()  # every signal hands TMEM between threads
    return phases ^ Int32(1 << i)


@cute.jit
def before_sync_and_arrive(bars, name: cutlass.Constexpr[str]):
    tcgen05_before_thread_sync()
    cute.arch.mbarrier_arrive(bars + SIG[name])


@cute.jit
def _commit(bars, name: cutlass.Constexpr[str]):
    with cute.arch.elect_one():
        cute.nvgpu.tcgen05.commit(bars + SIG[name])


_CUTE_DTYPES = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16}
_STATE_DTYPES = {torch.float32: cutlass.Float32, torch.bfloat16: cutlass.BFloat16}


def checkpoint_rows(total_tokens: int, num_seqs: int) -> int:
    """Rows of the checkpoint series: one per chunk plus one padding row per sequence (the T
    pass's ``tinv_rows``)."""
    return total_tokens // B_T + num_seqs


@dataclass(frozen=True)
class RecomputeParams:
    k_load: RaggedTma  # (token, d_k, key head) loads, zero past each sequence end
    v_load: RaggedTma  # (token, d_v, value head)
    tma_tinv: cute.CopyAtom
    mTinv: cute.Tensor  # TMA view (i, j, head, row)
    tma_ckpt: cute.CopyAtom
    mCkpt: cute.Tensor  # TMA view (v, k, head, row)
    gate: cute.Tensor  # (token, head) natural-log gate
    state_in: cute.Tensor | None  # (batch, head, v, k)
    items: cute.Tensor  # (row, fields) work table
    count: cute.Tensor  # (1,) work rows
    scheduler: cute.Tensor  # (>= 1,) zeroed claim counter
    sK: cute.Tensor
    sV: cute.Tensor
    sTinv: cute.Tensor
    sCkpt: cute.Tensor
    box_layout: cute.ComposedLayout  # one 64 x 64 swizzled box (K, V, Tinv)
    ckpt_box_layout: cute.ComposedLayout  # one d_v x 64 swizzled checkpoint box
    key_group: Int32  # value heads per key head
    value_group: Int32  # gate heads per value head
    units: Int32  # launched CTAs


class Item:
    """One work-table row's fields (chunk units except the token bounds)."""

    def __init__(self, params: RecomputeParams, row):
        t = params.items
        self.batch, self.head = t[row, 0], t[row, 1]
        self.write_start, self.write_end = t[row, 2], t[row, 3]
        self.compute_start = t[row, 4]
        self.batch_start, self.batch_end = t[row, 6], t[row, 7]


class GdnRecompute(WarpSpecialized):
    """Warpdrive GDN checkpoint-series recompute for fp16/bf16 ``d_k, d_v`` in {64, 128}."""

    def __init__(self, dtype_name: str, d_k: int, d_v: int, use_initial_state: bool):
        if d_k not in HEAD_DIMS or d_v not in HEAD_DIMS:
            raise ValueError(f"the recompute serves d_k, d_v in {HEAD_DIMS}, got {d_k}, {d_v}")
        self.dtype_name, self.d_k, self.d_v = dtype_name, d_k, d_v
        self.use_initial_state = use_initial_state
        self.io_dtype = _CUTE_DTYPES[dtype_name]
        # Upstream's register split: the compute group takes 256 when it also seeds the state.
        compute_regs, other_regs = (256, 24) if use_initial_state else (232, 48)
        self.roles = {
            "compute": RoleConfig(range(4), regs=compute_regs),
            "gate_load": RoleConfig(4, regs=other_regs),
            "load": RoleConfig(5, regs=other_regs),
            "mma": RoleConfig(6, regs=other_regs),
            "store": RoleConfig(7, regs=other_regs),
        }
        self.k_stage_bytes = d_k // BOX * BOX_BYTES
        self.v_stage_bytes = d_v // BOX * BOX_BYTES
        self.smem_plan = SmemPlan()
        k_buf = self.smem_plan.buffer("K", self.io_dtype, align=1024)
        v_buf = self.smem_plan.buffer("V", self.io_dtype, align=1024)
        tinv_buf = self.smem_plan.buffer("Tinv", self.io_dtype, align=1024)
        ckpt_buf = self.smem_plan.buffer("Ckpt", self.io_dtype, align=1024)
        self.k = Ring(K_STAGES, Tma(self.k_stage_bytes), Umma, buffer=k_buf)
        self.tinv = Ring(TINV_STAGES, Tma(BOX_BYTES), Umma, buffer=tinv_buf)
        self.v = Ring(V_STAGES, Tma(self.v_stage_bytes), Threads, buffer=v_buf)
        self.sched = Ring(SCHED_STAGES, Warps, Warps, payload=(Int32, (1,)))
        self.gate = Ring(GATE_STAGES, Threads, Threads, payload=(Float32, (B_T, 2)))
        # One-way TMEM signals: reuse is ordered by the later signal in the same serial
        # recurrence, not by a second full/empty handshake (see SIGNALS).
        self.sig = Barriers(len(SIGNALS), counts=[count for _, count in SIGNALS])
        self.ckpt = Ring(1, Warps, Threads, commit_fence="smem_async", buffer=ckpt_buf)
        self.tmem_plan = (
            TmemPlan()
            .region("state", d_k)
            .region("state_in", d_k // 2)
            .region("ks", B_T)
            .region("u", B_T)
            .region("y", B_T // 2)
            .region("du", B_T // 2)
        )
        self.work = WorkStream(schedule=OneShot(grid="1d"))

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: K and V, keyed by head."""
        return tma_slot_rows(2, B_T, heads)

    def get_name(self) -> str:
        init = "_init" if self.use_initial_state else ""
        return f"wd_gdn_recompute_{self.dtype_name}_dk{self.d_k}_dv{self.d_v}{init}"

    # ---- shared helpers ----------------------------------------------------------------

    @cute.jit
    def _smem_ptr(self, addr):
        """16-bit SMEM pointer at 1024-byte aligned 32-bit shared address ``addr``."""
        return cute.make_ptr(self.io_dtype, addr, cute.AddressSpace.smem, assumed_align=1024)

    @cute.jit
    def _next_row(self, sched):
        """Consumer side of ``sched``: the next work row this CTA runs."""
        with sched.consume() as h:
            row = h.data[0]
        return row, sched

    # ---- load (warp 5): K, V, Tinv and the scheduler claims ------------------------------

    @role(warps=5, regs=48)
    @cute.jit
    def load(
        self,
        params: RecomputeParams,
        *,
        k: Producer,
        v: Producer,
        tinv: Producer,
        sched: Producer,
        work: WorkCursor,
        me: Local,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.count[0]
            row = unit
            while row < total:
                it = Item(params, row)
                k, v, tinv = self.load_item(params, it, k, v, tinv)
                # Claim and publish the next row (upstream's scheduler_publish_next).
                claimed = Int32(0)
                if me.lane == 0:
                    claimed = cute.arch.atomic_add(
                        params.scheduler.iterator, Int32(1), sem="relaxed", scope="gpu"
                    )
                row = shfl_idx(claimed, 0) + params.units
                with sched.produce() as h:
                    h.data[0] = row
            work.advance()
        cute.arch.griddepcontrol_launch_dependents()
        return k, v, tinv, sched, work

    @cute.jit
    def load_item(
        self,
        params: RecomputeParams,
        it: Item,
        k,
        v,
        tinv,
    ):
        """K of every chunk, one ahead of its V and Tinv (upstream's issue order)."""
        head_k = it.head // params.key_group
        head_v = it.head // params.value_group
        if it.write_end > it.compute_start:
            loads = (params.k_load, params.v_load)
            k_desc, v_desc = sequence_descriptors(loads, it.batch_end, it.head, B_T)
            k = self.load_k(params, k, it, it.compute_start, head_k, k_desc)
            for chunk in cutlass.range(it.compute_start + 1, it.write_end, unroll=1):
                k = self.load_k(params, k, it, chunk, head_k, k_desc)
                v = self.load_v(params, v, it, chunk - 1, head_v, v_desc)
                tinv = self.load_tinv(params, tinv, it, chunk - 1)
            v = self.load_v(params, v, it, it.write_end - 1, head_v, v_desc)
            tinv = self.load_tinv(params, tinv, it, it.write_end - 1)
        return k, v, tinv

    @cute.jit
    def _load_boxes(
        self,
        params,
        load,
        it: Item,
        chunk,
        head,
        n_boxes: cutlass.Constexpr[int],
        base,
        h,
        desc,
    ):
        """TMA ``n_boxes`` 64 x 64 boxes of chunk ``chunk``'s rows into ``base``; rows past the
        sequence end load as zeros."""
        token = it.batch_start + chunk * B_T
        valid = cutlass.min(it.batch_end - token, B_T)
        for kb in cutlass.range_constexpr(n_boxes):
            dst = cute.make_tensor(self._smem_ptr(base + kb * BOX_BYTES), params.box_layout)
            ragged_load(load, dst, token, valid, head, kb, h.barrier, B_T, BOX, desc)

    @cute.jit
    def load_k(self, params: RecomputeParams, k, it: Item, chunk, head_k, desc):
        with k.produce() as h:
            h.access(write=True, by="tma")
            base = params.sK.iterator.toint() + h.index * self.k_stage_bytes
            self._load_boxes(
                params,
                params.k_load,
                it,
                chunk,
                head_k,
                self.d_k // BOX,
                base,
                h,
                desc,
            )
        return k

    @cute.jit
    def load_v(self, params: RecomputeParams, v, it: Item, chunk, head_v, desc):
        with v.produce() as h:
            h.access(write=True, by="tma")
            base = params.sV.iterator.toint() + h.index * self.v_stage_bytes
            self._load_boxes(
                params,
                params.v_load,
                it,
                chunk,
                head_v,
                self.d_v // BOX,
                base,
                h,
                desc,
            )
        return v

    @cute.jit
    def load_tinv(self, params: RecomputeParams, tinv, it: Item, chunk):
        with tinv.produce() as h:
            h.access(write=True, by="tma")
            base = params.sTinv.iterator.toint() + h.index * BOX_BYTES
            dst = cute.make_tensor(self._smem_ptr(base), params.box_layout)
            row = it.batch_start // B_T + it.batch + chunk
            src = cute.local_tile(params.mTinv[None, None, it.head, row], (B_T, B_T), (0, 0))
            tSs, tSg = cpasync.tma_partition(
                params.tma_tinv,
                0,
                cute.make_layout(1),
                cute.group_modes(dst, 0, 2),
                cute.group_modes(src, 0, 2),
            )
            cute.copy(params.tma_tinv, tSg, tSs, tma_bar_ptr=h.barrier)
        return tinv

    # ---- gate (warp 4): cumulative log2 gate and its exp2 per chunk ----------------------

    @role(warps=4, regs=48)
    @cute.jit
    def gate_load(
        self,
        params: RecomputeParams,
        *,
        gate: Producer,
        sched: Consumer,
        work: WorkCursor,
        me: Local,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.count[0]
            row = unit
            while row < total:
                it = Item(params, row)
                for chunk in cutlass.range(it.compute_start, it.write_end, unroll=1):
                    gate = self.gate_chunk(params, gate, it, chunk, me.lane)
                row, sched = self._next_row(sched)
            work.advance()
        return gate, sched, work

    @cute.jit
    def gate_chunk(self, params: RecomputeParams, gate, it: Item, chunk, lane):
        """One chunk's 64-token log2 cumsum (neutral past the sequence end) and its exp2."""
        start = it.batch_start + chunk * B_T
        end = it.batch_end
        toks = (start + lane, start + lane + 32)
        valid = [t < end for t in toks]
        raw = [params.gate[cutlass.min(t, end - 1), it.head] for t in toks]
        zero = Float32(0.0)
        g = [raw[i].to(Float32) if valid[i] else zero for i in range(2)]
        rcp_ln2 = opaque_f32_zero() + Float32(RCP_LN2)
        g[0], g[1] = fmul2(g[0], g[1], rcp_ln2, rcp_ln2)
        for offset in [1, 2, 4, 8, 16]:
            for col in cutlass.range_constexpr(2):
                n = shfl_up(g[col], offset)
                if lane >= offset:
                    g[col] = g[col] + n
        g[1] += shfl_idx(g[0], 31)
        with gate.produce() as gh:
            for col in cutlass.range_constexpr(2):
                gh.data[lane + 32 * col, 0] = g[col]
                gh.data[lane + 32 * col, 1] = cute.math.exp2(g[col], fastmath=True)
        return gate

    # ---- mma (warp 6): KS = state @ K^T, U = Y @ Tinv^T, state += dU @ K ----------------

    @role(warps=6, regs=48)
    @cute.jit
    def mma(
        self,
        params: RecomputeParams,
        *,
        k: Consumer,
        tinv: Consumer,
        sig: Barriers,
        sched: Consumer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        ph = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.count[0]
            row = unit
            while row < total:
                k, tinv, ph = self.mma_item(params, Item(params, row), k, tinv, sig, ph, tmem)
                row, sched = self._next_row(sched)
            work.advance()
        return k, tinv, sched, work

    @cute.jit
    def mma_item(self, params: RecomputeParams, it: Item, k, tinv, sig, ph, tmem):
        io, d_k, d_v = self.io_dtype, self.d_k, self.d_v
        ks_desc = MmaDesc(
            M=d_v,
            N=B_T,
            K=d_k,
            bpe_a=2,
            bpe_b=2,
            idesc=instr_desc(Float32, io, io, m=d_v, n=B_T),
        )
        u_desc = MmaDesc(
            M=d_v,
            N=B_T,
            K=B_T,
            bpe_a=2,
            bpe_b=2,
            idesc=instr_desc(Float32, io, io, m=d_v, n=B_T),
        )
        kv_desc = MmaDesc(
            M=d_v,
            N=d_k,
            K=B_T,
            bpe_a=2,
            bpe_b=2,
            btranspose=True,
            idesc=instr_desc(Float32, io, io, m=d_v, n=d_k, b_mn_major=True),
        )
        ks_a_step = ks_desc.steps_per_subtile * ks_desc.tmem_advance_A  # TMEM cols per box
        state_col = tmem.ptr("state").toint()
        state_in_col = tmem.ptr("state_in").toint()
        ks_col = tmem.ptr("ks").toint()
        u_col = tmem.ptr("u").toint()
        y_col = tmem.ptr("y").toint()
        du_col = tmem.ptr("du").toint()
        seed = cutlass.Boolean(False)
        if cutlass.const_expr(self.use_initial_state):
            seed = it.compute_start == 0
        for local in cutlass.range(it.write_end - it.compute_start, unroll=1):
            have_state = local > 0 or seed
            kh = k.wait()
            kh.access(read=True, by="umma")
            k_addr = params.sK.iterator.toint() + kh.index * self.k_stage_bytes
            desc_k = desc_opaque(smem_desc(k_addr, 16, 1024, SWIZZLE_128B))
            desc_kt = desc_opaque(smem_desc(k_addr, BOX_BYTES, 1024, SWIZZLE_128B))
            # ---- KS = state(fp16) @ K^T ----------------------------------------------
            if have_state:
                ph = _wait(sig, ph, "state_in")
                tmem.access("state_in", read=True, by="umma")
                tmem.access("ks", write=True, by="umma")
                for box in cutlass.range_constexpr(d_k // BOX):
                    for kk in cutlass.range_constexpr(ks_desc.steps_per_subtile):
                        mma_ts_step(
                            ks_desc,
                            state_in_col + box * ks_a_step,
                            desc_k + (box * BOX_BYTES >> 4),
                            ks_col,
                            kk,
                            cutlass.Boolean(box > 0 or kk > 0),
                        )
                _commit(sig, "ks")
            # ---- U = Y @ Tinv^T -----------------------------------------------------
            th = tinv.wait()
            th.access(read=True, by="umma")
            ph = _wait(sig, ph, "y")
            tmem.access("y", read=True, by="umma")
            tmem.access("u", write=True, by="umma")
            t_addr = params.sTinv.iterator.toint() + th.index * BOX_BYTES
            desc_t = desc_opaque(smem_desc(t_addr, 16, 1024, SWIZZLE_128B))
            for kk in cutlass.range_constexpr(B_T // 16):
                mma_ts_step(u_desc, y_col, desc_t, u_col, kk, cutlass.Boolean(kk > 0))
            _commit(sig, "u")
            th.release()
            # ---- state += dU @ K (accumulate onto the rescaled state) ----------------
            ph = _wait(sig, ph, "du")
            tmem.access("du", read=True, by="umma")
            tmem.access("state", read=True, write=True, by="umma")
            for kk in cutlass.range_constexpr(B_T // 16):
                mma_ts_step(
                    kv_desc,
                    du_col,
                    desc_kt,
                    state_col,
                    kk,
                    cutlass.Boolean(True) if cutlass.const_expr(kk > 0) else have_state,
                )
            _commit(sig, "state")
            kh.release()
        return k, tinv, ph

    # ---- compute (warps 0-3): state restage, checkpoints, Y and the decayed U ------------

    @role(warps=range(4), regs=232)
    @cute.jit
    def compute(
        self,
        params: RecomputeParams,
        *,
        v: Consumer,
        gate: Consumer,
        sig: Barriers,
        ckpt: Producer,
        sched: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        ph = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.count[0]
            row = unit
            while row < total:
                v, gate, ckpt, ph = self.compute_item(
                    params,
                    Item(params, row),
                    v,
                    gate,
                    ckpt,
                    sig,
                    ph,
                    tmem,
                    me,
                )
                row, sched = self._next_row(sched)
            work.advance()
        return v, gate, ckpt, sched, work

    @cute.jit
    def _state_row(self, tid):
        """``(v row, row valid)`` of compute thread ``tid``'s TMEM lane: M = 64 accumulators use
        the first 16 lanes of each warp."""
        if cutlass.const_expr(self.d_v == 128):
            return tid, cutlass.Boolean(True)
        return (tid // 32) * 16 + (tid % 32) % 16, (tid % 32) < 16

    @cute.jit
    def compute_item(
        self,
        params: RecomputeParams,
        it: Item,
        v,
        gate,
        ckpt,
        sig,
        ph,
        tmem,
        me,
    ):
        io, d_v = self.io_dtype, self.d_v
        tid, lane = me.tid, me.lane
        lane_base = 32 * (me.tmem_row // 32)
        ks_col = tmem.ptr("ks").toint()
        u_col = tmem.ptr("u").toint()
        y_col = tmem.ptr("y").toint()
        du_col = tmem.ptr("du").toint()
        dv_halves = d_v // 64
        n_vals = 32
        # ldmatrix.trans coordinates of this thread's V fragment (token row, value column).
        v_row = tid % 8 + (tid // 16 % 2) * 8
        if cutlass.const_expr(d_v == 128):
            v_col = (tid // 8 % 2) * 8 + (tid // 32 % 2) * 32
            v_box = (tid // 64) * BOX_BYTES
        else:
            v_col = (tid // 32) * 16 + (tid // 8 % 2) * 8
            v_box = 0
        n_local = it.write_end - it.compute_start
        seed = cutlass.Boolean(False)
        if cutlass.const_expr(self.use_initial_state):
            seed = it.compute_start == 0
            if n_local > 0 and seed:
                self.seed_state(params, it, tmem, me)
        for local in cutlass.range(n_local, unroll=1):
            chunk = it.compute_start + local
            if cutlass.const_expr(not self.use_initial_state):  # noqa: SIM102 (static)
                if chunk == 0 and it.write_start == 0:
                    ckpt = self.zero_checkpoint(params, ckpt, tid)
            valid_state = local > 0 or seed
            gh = gate.wait()
            cs = gh.data
            cumprod_total = cs[B_T - 1, 1]
            if local > 0:
                # KV of the previous chunk complete: the state columns are this group's
                # until the next du publication (see the ``state`` signal).
                ph = _wait(sig, ph, "state")
            # ---- state: restage as the KS operand, checkpoint, rescale --------------
            if valid_state:
                ckpt = self.restage(params, it, chunk, cumprod_total, ckpt, sig, tmem, me)
            cumprod_vals = [
                cs[(lane % 4) * 2 + ((kv // 4) * 8 + kv % 2), 1] for kv in range(n_vals)
            ]
            # ---- Y = V - fp16(KS * cumprod) ----------------------------------------------
            vh = v.wait()
            vh.access(read=True, by="threads")
            v_stage = params.sV.iterator.toint() + vh.index * self.v_stage_bytes + v_box
            words = []
            for half in cutlass.range_constexpr(dv_halves):
                for block in cutlass.range_constexpr(4):
                    r = v_row + block * 16
                    words += ldmatrix(
                        v_stage + (r * BOX + swizzle_xor_128b(r, v_col + half * 16)) * 2,
                        4,
                        trans=True,
                    )
            if valid_state:
                ph = _wait(sig, ph, "ks")
                tmem.access("ks", read=True)
                acc = []
                for half in cutlass.range_constexpr(dv_halves):
                    acc += tmem_ld("16x256b", 8, ((lane_base + half * 16) << 16) + ks_col)
                new = []
                for half in cutlass.range_constexpr(dv_halves):
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(
                            acc[half * 32 + 2 * j],
                            acc[half * 32 + 2 * j + 1],
                            cumprod_vals[2 * j],
                            cumprod_vals[2 * j + 1],
                        )
                        new.append(
                            sub_f16x2(
                                words[half * 16 + j], fp32_to_fp16(s0, s1, dtype=io), dtype=io
                            )
                        )
                words = new
            # Tokens past the sequence end take upstream's zero-filled Y (see module doc).
            n_valid = it.batch_end - it.batch_start - chunk * B_T
            if n_valid < B_T:
                masked = []
                for half in cutlass.range_constexpr(dv_halves):
                    for j in cutlass.range_constexpr(16):
                        tok = (lane % 4) * 2 + (j // 2) * 8
                        w = words[half * 16 + j]
                        lo = w & Int32(0xFFFF) if tok < n_valid else Int32(0)
                        masked.append(w if tok + 1 < n_valid else lo)
                words = masked
            tmem.access("y", write=True)
            for half in cutlass.range_constexpr(dv_halves):
                tmem_st(
                    "16x128b",
                    ((lane_base + half * 16) << 16) + y_col,
                    words[half * 16 : half * 16 + 16],
                )
            cute.arch.fence_view_async_tmem_store()
            before_sync_and_arrive(sig, "y")
            # ---- decayed U -> the KV GEMM operand ------------------------------------
            # The decay factors fill the U GEMM's latency (upstream's schedule).
            last = cs[B_T - 1, 0]
            cumsum_vals = [
                cs[(lane % 4) * 2 + ((kv // 4) * 8 + kv % 2), 0] for kv in range(n_vals)
            ]
            decay = []
            for kv in cutlass.range_constexpr(0, n_vals, 2):
                d0, d1 = cute.arch.add_packed_f32x2(
                    (last, last), (-cumsum_vals[kv], -cumsum_vals[kv + 1])
                )
                decay.append(cute.math.exp2(d0, fastmath=True))
                decay.append(cute.math.exp2(d1, fastmath=True))
            gh.release()
            ph = _wait(sig, ph, "u")
            tmem.access("u", read=True)
            acc = []
            for half in cutlass.range_constexpr(dv_halves):
                acc += tmem_ld("16x256b", 8, ((lane_base + half * 16) << 16) + u_col)
            tmem.access("du", write=True)
            for half in cutlass.range_constexpr(dv_halves):
                pack = []
                for j in cutlass.range_constexpr(16):
                    s0, s1 = fmul2(
                        acc[half * 32 + 2 * j],
                        acc[half * 32 + 2 * j + 1],
                        decay[2 * j],
                        decay[2 * j + 1],
                    )
                    pack.append(fp32_to_fp16(s0, s1, dtype=io))
                tmem_st("16x128b", ((lane_base + half * 16) << 16) + du_col, pack)
            cute.arch.fence_view_async_tmem_store()
            before_sync_and_arrive(sig, "du")
            # The V release retains its proxy fence before the TMA ring refills this stage.
            vh.release()
        if n_local > 0:
            ph = _wait(sig, ph, "state")  # the item's last KV: the columns are free again
        return v, gate, ckpt, ph

    @cute.jit
    def restage(self, params: RecomputeParams, it: Item, chunk, scale, ckpt, sig, tmem, me):
        """fp16 state into ``state_in``, the chunk's checkpoint, then ``state *= scale``."""
        io = self.io_dtype
        lane0 = (32 * (me.tmem_row // 32)) << 16
        state_col = tmem.ptr("state").toint()
        state_in_col = tmem.ptr("state_in").toint()
        n_ld = self.d_k // 32
        vecs = [tmem_ld("32x32b", 32, lane0 + state_col + i * 32) for i in range(n_ld)]
        tmem.access("state_in", write=True)
        for i in cutlass.range_constexpr(n_ld):
            pack = [fp32_to_fp16(vecs[i][2 * j], vecs[i][2 * j + 1], dtype=io) for j in range(16)]
            tmem_st("32x32b", lane0 + state_in_col + i * 16, pack)
        cute.arch.fence_view_async_tmem_store()
        before_sync_and_arrive(sig, "state_in")
        if chunk >= it.write_start:
            row, row_valid = self._state_row(me.tid)
            with ckpt.produce() as h:
                h.access(write=True, by="threads")
                base = params.sCkpt.iterator.toint()
                if row_valid:
                    for i in cutlass.range_constexpr(n_ld):
                        for g in cutlass.range_constexpr(4):
                            packs = [
                                fp32_to_fp16(
                                    vecs[i][g * 8 + 2 * t], vecs[i][g * 8 + 2 * t + 1], dtype=io
                                )
                                for t in range(4)
                            ]
                            col = i * 32 + g * 8
                            box, in_box = col // BOX, col % BOX
                            off = box * self.d_v * BOX + row * BOX + swizzle_xor_128b(row, in_box)
                            st_shared_v4(base + off * 2, packs)
        for i in cutlass.range_constexpr(n_ld):
            scaled = []
            for j in cutlass.range_constexpr(16):
                s0, s1 = fmul2(vecs[i][2 * j], vecs[i][2 * j + 1], scale, scale)
                scaled += [s0, s1]
            tmem_st("32x32b", lane0 + state_col + i * 32, scaled)
        cute.arch.fence_view_async_tmem_store()
        return ckpt

    @cute.jit
    def zero_checkpoint(self, params: RecomputeParams, ckpt, tid):
        """Chunk 0's checkpoint with no initial state: zeros."""
        with ckpt.produce() as h:
            h.access(write=True, by="threads")
            base = params.sCkpt.iterator.toint()
            zero = Int32(0)
            for z in cutlass.range_constexpr(self.d_k * self.d_v * 2 // 16 // 128):
                st_shared_v4(base + (tid + z * 128) * 16, [zero, zero, zero, zero])
        return ckpt

    @cute.jit
    def seed_state(self, params: RecomputeParams, it: Item, tmem, me):
        """``state_in[batch, head]`` (fp32 or bf16) into the state columns."""
        lane0 = (32 * (me.tmem_row // 32)) << 16
        state_col = tmem.ptr("state").toint()
        row, _ = self._state_row(me.tid)
        src = params.state_in[it.batch, it.head, row, None]
        dtype = src.element_type
        vw = 16 // (dtype.width // 8)
        for i in cutlass.range_constexpr(self.d_k // 32):
            words = []
            for g in cutlass.range_constexpr(32 // vw):
                addr = (src.iterator + (i * 32 + g * vw)).toint()
                vec = cute.make_tensor(
                    cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=16),
                    cute.make_layout(vw),
                ).load()
                words += [vec[t].to(Float32) for t in range(vw)]
            tmem_st("32x32b", lane0 + state_col + i * 32, words)
        cute.arch.fence_view_async_tmem_store()

    # ---- store (warp 7): checkpoint TMA stores ---------------------------------------------

    @role(warps=7, regs=48)
    @cute.jit
    def store(
        self,
        params: RecomputeParams,
        *,
        ckpt: Consumer,
        sched: Consumer,
        work: WorkCursor,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.count[0]
            row = unit
            while row < total:
                it = Item(params, row)
                first = it.batch_start // B_T + it.batch
                begin = cutlass.max(it.write_start, it.compute_start)
                for chunk in cutlass.range(begin, it.write_end, unroll=1):
                    ckpt = self.store_checkpoint(params, ckpt, it.head, first + chunk)
                row, sched = self._next_row(sched)
            work.advance()
        bulk_wait_group(0)
        return ckpt, sched, work

    @cute.jit
    def store_checkpoint(self, params: RecomputeParams, ckpt, head, row):
        with ckpt.consume() as h:
            h.access(read=True, by="tma")
            base = params.sCkpt.iterator.toint()
            for kb in cutlass.range_constexpr(self.d_k // BOX):
                src = cute.make_tensor(
                    self._smem_ptr(base + kb * self.d_v * BOX * 2), params.ckpt_box_layout
                )
                dst = cute.local_tile(
                    params.mCkpt[None, None, head, row], (self.d_v, BOX), (0, kb)
                )
                tSs, tSg = cpasync.tma_partition(
                    params.tma_ckpt,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(src, 0, 2),
                    cute.group_modes(dst, 0, 2),
                )
                cute.copy(params.tma_ckpt, tSs, tSg)
            bulk_commit_group()
            bulk_wait_group(0, read=True)
        return ckpt

    # ---- kernel and entry point --------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        k_load,
        v_load,
        tma_tinv,
        mTinv,
        tma_ckpt,
        mCkpt,
        gate,
        state_in,
        items,
        count,
        scheduler,
        layouts,
        box_layout,
        ckpt_box_layout,
        key_group,
        value_group,
        units,
        trace,
        bound,
    ):
        # The work table and Tinv tiles come from the preceding launches.
        cute.arch.griddepcontrol_wait()
        k_layout, v_layout, tinv_layout, ckpt_layout = layouts
        self.begin_roles(
            work=bound,
            layouts={
                "K": k_layout,
                "V": v_layout,
                "Tinv": tinv_layout,
                "Ckpt": ckpt_layout,
            },
        )
        params = RecomputeParams(
            k_load=k_load,
            v_load=v_load,
            tma_tinv=tma_tinv,
            mTinv=mTinv,
            tma_ckpt=tma_ckpt,
            mCkpt=mCkpt,
            gate=gate,
            state_in=state_in,
            items=items,
            count=count,
            scheduler=scheduler,
            sK=self.buffer("K"),
            sV=self.buffer("V"),
            sTinv=self.buffer("Tinv"),
            sCkpt=self.buffer("Ckpt"),
            box_layout=box_layout,
            ckpt_box_layout=ckpt_box_layout,
            key_group=key_group,
            value_group=value_group,
            units=units,
        )
        self.run_roles(params, trace_buffer=trace)

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,  # (tokens, key heads, d_k)
        v: cute.Tensor,  # (tokens, value heads, d_v)
        gate: cute.Tensor,  # (tokens, heads) natural-log gate
        tinv: cute.Tensor,  # (tinv rows, heads, 64, 64)
        items: cute.Tensor,  # (rows, fields) int32 work table
        count: cute.Tensor,  # (1,) int32
        scheduler: cute.Tensor,  # (>= 1,) int32, zeroed
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        state_in: cute.Tensor | None,  # (num_seqs, heads, d_v, d_k)
        checkpoints: cute.Tensor,  # (rows, heads, d_v, d_k)
        workspace: cute.Tensor,
        max_ctas: Int32,
        trace: cute.Tensor | None,  # int64 globaltimer buffer when tracing is enabled
        stream: cuda.CUstream,
    ):
        self.launch(
            k,
            v,
            gate,
            tinv,
            items,
            count,
            scheduler,
            tma_slots,
            state_in,
            checkpoints,
            max_ctas,
            stream,
            trace,
        )

    @cute.jit
    def launch(
        self,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        tinv: cute.Tensor,
        items: cute.Tensor,
        count: cute.Tensor,
        scheduler: cute.Tensor,
        tma_slots: cute.Tensor,
        state_in: cute.Tensor | None,
        checkpoints: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
        trace: cute.Tensor | None = None,
    ):
        """Launch the recompute; callable from another ``@cute.jit`` host, such as a fused
        launch bundle. It is a programmatic dependent launch: every thread waits for the
        preceding kernel (the work-table prologue) before its roles start."""
        io = self.io_dtype
        heads = Int32(gate.shape[1])
        key_group = heads // Int32(k.shape[1])
        value_group = heads // Int32(v.shape[1])
        box = sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, BOX), 1)
        box_layout = cute.select(box, mode=[0, 1])
        ckpt_box = sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (self.d_v, BOX), 1)
        ckpt_box_layout = cute.select(ckpt_box, mode=[0, 1])
        elems = BOX_BYTES // 2
        layouts = (
            cute.make_layout((self.k_stage_bytes // 2, K_STAGES)),
            cute.make_layout((self.v_stage_bytes // 2, V_STAGES)),
            cute.make_layout((elems, TINV_STAGES)),
            cute.make_layout((self.d_v * self.d_k, 1)),
        )
        # (token, dim, head), (i, j, head, row) and (v, k, head, row) TMA views.
        mK = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[0, 2, 1]))
        mV = cute.make_tensor(v.iterator, cute.select(v.layout, mode=[0, 2, 1]))
        mTinv = cute.make_tensor(tinv.iterator, cute.select(tinv.layout, mode=[2, 3, 1, 0]))
        mCkpt = cute.make_tensor(
            checkpoints.iterator, cute.select(checkpoints.layout, mode=[2, 3, 1, 0])
        )
        g2s = cpasync.CopyBulkTensorTileG2SOp(cute.nvgpu.tcgen05.CtaGroup.ONE)
        k_load, v_load = (
            make_ragged_tma(g2s, t, box_layout, B_T, BOX, slot_region(tma_slots, i, B_T, heads))
            for i, t in enumerate((mK, mV))
        )
        tma_tinv, mTinv_t = cpasync.make_tiled_tma_atom(g2s, mTinv, box_layout, (B_T, B_T))
        tma_ckpt, mCkpt_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mCkpt, ckpt_box_layout, (self.d_v, BOX)
        )
        # Upstream's grid: one CTA per SM, each starting at its own table row.
        units = cutlass.max(max_ctas, 1)
        bound = self.work.bind((units, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            k_load,
            v_load,
            tma_tinv,
            mTinv_t,
            tma_ckpt,
            mCkpt_t,
            gate,
            state_in,
            items,
            count,
            scheduler,
            layouts,
            box_layout,
            ckpt_box_layout,
            key_group,
            value_group,
            units,
            trace,
            bound,
        ).launch(**self.launch_config(bound), stream=stream, use_pdl=True)


def bundle_hook(io_dtype, d_k: int, d_v: int, use_initial_state: bool, num_sm: int):
    """``(op, num_sm)`` constexpr for launch bundles that run the recompute inside their own
    ``@cute.jit`` host: ``op.launch(..., Int32(num_sm), stream)``."""
    name = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}[io_dtype]
    return GdnRecompute(name, d_k, d_v, use_initial_state), num_sm


@jit_cache
def compile_gdn_recompute(op: GdnRecompute, state_dtype_name: str | None):
    """Compile for dynamic tokens, heads, sequences, work rows and checkpoint rows."""
    io = op.io_dtype
    i32 = cutlass.Int32
    tokens, key_heads, value_heads, heads, seqs = (cute.sym_int() for _ in range(5))
    tinv_rows, ckpt_rows, items = (cute.sym_int() for _ in range(3))
    state = None
    if op.use_initial_state:
        state_dtype = getattr(cutlass, state_dtype_name)
        state = make_fake_strided_tensor(
            state_dtype,
            (seqs, heads, op.d_v, op.d_k),
            stride_divisibility=tma_divisibility(state_dtype),
        )
    return compile_tvm_ffi(
        op,
        make_fake_strided_tensor(io, (tokens, key_heads, op.d_k), stride_divisibility=8),
        make_fake_strided_tensor(io, (tokens, value_heads, op.d_v), stride_divisibility=8),
        make_fake_strided_tensor(Float32, (tokens, heads), contiguous_dim=None),
        make_fake_strided_tensor(io, (tinv_rows, heads, B_T, B_T), stride_divisibility=8),
        make_fake_strided_tensor(i32, (items, 10), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        fake_tma_slots(),
        state,
        make_fake_strided_tensor(io, (ckpt_rows, heads, op.d_v, op.d_k), stride_divisibility=8),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        Int32(1),
        make_fake_strided_tensor(cutlass.Int64, (cute.sym_int(),))
        if op.trace_backend == "globaltimer"
        else None,
        name=op.get_name(),
    )


def gdn_recompute(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    tinv: torch.Tensor,
    items: torch.Tensor,
    count: torch.Tensor,
    checkpoints: torch.Tensor,
    *,
    state_in: torch.Tensor | None = None,
    scheduler: torch.Tensor | None = None,
    op: GdnRecompute | None = None,
    trace: torch.Tensor | None = None,
    max_ctas: int | None = None,
) -> None:
    """Fill ``checkpoints`` for packed ``k [T, HK, DK]``, ``v [T, HV, DV]``, fp32 natural-log
    ``gate [T, H]``, ``gdn_tinv`` tiles and an int32 work table ``items`` (see the module
    docstring). ``scheduler`` (int32, zeroed) is allocated when omitted. ``op`` (a
    ``GdnRecompute`` with ``enable_trace("globaltimer")``) and its ``trace`` buffer capture a role
    timeline. ``max_ctas`` caps the grid (default: one CTA per SM); with fewer CTAs than rows,
    each CTA claims several items in turn."""
    if k.dtype not in IO_DTYPE_NAMES or v.dtype != k.dtype or checkpoints.dtype != k.dtype:
        raise TypeError("k, v and checkpoints must share a 16-bit dtype")
    if state_in is not None and state_in.dtype not in _STATE_DTYPES:
        raise TypeError(f"state_in must be fp32 or bf16, got {state_in.dtype}")
    if scheduler is None:
        scheduler = torch.zeros(2, dtype=torch.int32, device=k.device)
    if op is None:
        op = GdnRecompute(IO_DTYPE_NAMES[k.dtype], k.shape[-1], v.shape[-1], state_in is not None)
    state_name = None if state_in is None else _STATE_DTYPES[state_in.dtype].__name__
    fn = compile_gdn_recompute(op, state_name)
    workspace, sms = op.work.launch_args(k.device)
    if max_ctas is not None:
        sms = max_ctas
    fn(
        k,
        v,
        gate,
        tinv,
        items,
        count,
        scheduler,
        new_tma_slots(op.tma_slot_rows(gate.shape[1]), k.device),
        state_in,
        checkpoints,
        workspace,
        sms,
        trace,
    )
