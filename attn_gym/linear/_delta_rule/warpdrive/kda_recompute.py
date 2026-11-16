# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# kda_recompute_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and
# channels with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence
# descriptor arrays, and only the checkpoint-series form the backward drivers compile remains.

"""KDA backward recompute (BT = 16, per-key-channel decay) for SM100: the checkpoint series, the
recurrent state entering every 16-token chunk, which the KDA bprop reads.

Per chunk of a work item (``g`` the in-chunk cumulative log2 gate per key channel, ``S`` the
``[d_v, d_k]`` fp32 state carried across chunks)::

    checkpoint[c] = io(S)                                   (state entering the chunk)
    K_decay = K exp2(g)   K_inv = K exp2(-g)   K_restore = K exp2(g_last - g)
    L   = beta * strict_lower(K_decay K_inv^T)        T = (I + L)^-1  (16x16 blockwise, mma.sync)
    Y   = beta (V - K_decay S)                        U = T Y
    S  <- exp2(g_last) S + K_restore^T U

Work items are rows of the bprop's work table (``WORK_ITEM_FIELDS``): batch, head, write_start,
write_end, compute_start, compute_end (chunks), batch_start, batch_end (tokens). Chunks
``[compute_start, write_end)`` run and the checkpoints of ``[write_start, write_end)`` are
stored, chunk ``c`` of sequence (or chain piece) ``b`` at row ``batch_start // 16 + b + c``; as
upstream, an item with ``write_start > 0`` stores no row for its own first chunk (the split
tables give such items a warmup, ``compute_start < write_start``). The state starts from ``state_in[batch, head]`` when ``use_initial_state`` and ``compute_start == 0``,
else from zero (a split item's warmup); with no initial state chunk 0's checkpoint is zero. The
work table and the zeroed scheduler counter come from the caller (``work_tables`` in the
launch bundles, ``kda_prefill.uncut_work_items`` standalone). CTA ``b`` starts at item ``b``;
the load warp claims the next item after issuing an item's loads and broadcasts it to every role.

K/V/gate rows past a sequence end load as zeros (``ragged.RaggedTma``), as from
upstream's per-sequence descriptors; beta is zero there.

Roles (16 warps) and their handoffs. Two-way SMEM/TMA handoffs are warpdrive Rings; ``rr``
marks a round-robin side of the two ``cg0`` instances, which take the CTA's cumulative chunks
alternately (``c % 2``)::

    load (14)     --raw: 6 x K,V,Gate (TMA)-->        cg0 (both instances), cg1
                  --beta: 6 x f32[16]-->              regmma, cg1
                  --sched: 8 x next item-->           every other role
    cg0 (0-7) rr  --kk: K_decay, K_inv-->             mma, regmma
                  --restore: K_restore-->             mma
                  --exchange: exp2(g) rows-->         cg1 (signal-only; see __init__)
    regmma (12)   --tinv: T-->                        mma (13)
    cg1 (8-11)    --ckpt: 2 x io checkpoint row-->    store (15): TMA store
    cg0           ..ckpt (left key half, raw arrivals; see __init__)

The TMEM state recurrence between the MMA warp and the compute groups (io state operand, KS / U
accumulators, Y / U operands, state-update completion) uses the upstream signal-only barriers
(``tstate``, ``TSTATE_COUNTS``): every stage they guard is reused only after a later handoff of
the same recurrence, so a channel's reuse wait would only lengthen the per-chunk critical path.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.linear._delta_rule.wd_support import HEAD_DIMS
from attn_gym.warpdrive import (
    AFTER_FENCE_COVERED,
    BEFORE_FENCE_COVERED,
    BOTH_FENCES_COVERED,
    Barriers,
    Consumer,
    Instance,
    Local,
    NamedBarriers,
    OneShot,
    PdlWait,
    Producer,
    Ring,
    RoleConfig,
    Sharing,
    SmemPlan,
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
from attn_gym.warpdrive.thread_sync import tcgen05_after_thread_sync

from .blockwise_inverse import invert_unit_lower_16x16_fragments
from .contract import tma_divisibility
from .kda_common import (
    B_T,
    CUTE_DTYPES,
    TORCH_NAMES,
    WORK_ITEM_FIELDS,
    beta_residual,
    box128,
    lds_16_f32,
    lds_f32,
    lds_v4,
    lds_v4_f32,
    load_rows,
    load_state_16,
    mask_pair,
    opaque_i32,
    raw_beta_release,
    spin_wait,
    sts_f32,
    swz32,
    swz128,
    umma_commit,
    warp_arrive,
)
from .ptx import (
    elect_sync,
    f16x2_to_f32,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    mma_step,
    st_shared_v4,
    stmatrix,
    tmem_ld,
    tmem_st,
)
from .ragged import (
    RaggedTma,
    fake_tma_slots,
    make_ragged_tma,
    new_tma_slots,
    sequence_descriptors,
    slot_region,
    tma_slot_rows,
)
from .umma import SWIZZLE_32B, SWIZZLE_128B, MmaDesc, desc_opaque, instr_desc, mma_ts_op, smem_desc

CG0_WARPS = [range(4), range(4, 8)]  # two round-robin compute-group-0 instances
CG0_REGS = 160
LOG2_E = 1.4426950408889634
RAW_STAGES = 6  # upstream's checkpoint build trims the raw ring to 6 stages for the staging rows
SCHED_STAGES = 8
DECAY_STAGES = 2
INTER_STAGES = 2
CKPT_STAGES = 2

# TMEM state handoffs (``tstate`` barriers, upstream counts), each completing once per epoch:
#   SIN_L  left key half of the io state operand: CG0 instance (chunks after an item's first),
#          CG1 (the seed); MMA waits before KS k-steps [0, n/2). Reuse: rewritten after SACC_L.
#   SIN_R  right half, CG1; MMA waits before the rest of KS.
#   KS, U  KS / U accumulators, MMA commit; CG1 reads. KS(c+1) / U(c+1) follow YIN / UIN waits
#          that CG1 signals after reading them.
#   YIN    Y operand, CG1; MMA waits before U. Rewritten after U completes.
#   UIN    U operand (CG1) and the left-half state decay (CG0 instance): 4 + 4 arrivals; MMA
#          waits before the state update.
#   SACC_L / SACC_R (2 stages)  left / right state update committed; CG0 / CG1 read the state.
SIN_L, SIN_R, KS, U, YIN, UIN, SACC_L, SACC_R = 0, 1, 2, 3, 4, 5, 6, 8
TSTATE_COUNTS = (4, 4, 1, 1, 4, 8, 1, 1, 1, 1)


@cute.jit
def _raw_beta_wait(raw, beta, n, fence: cutlass.Constexpr[bool] = True):
    """CG1's waits on chunk ``n``'s ``raw`` and ``beta`` stages, driven by one shared counter
    (the upstream raw_index): both are ``RAW_STAGES``-stage rings CG1 consumes once per chunk, in
    order; then ``tcgen05.fence::after_thread_sync`` as after a managed wait in a TMEM role
    (``fence=False`` omits it, ``AFTER_FENCE_COVERED``). Returns the stage."""
    stage = n % RAW_STAGES
    phase = (n // RAW_STAGES) & 1
    cute.arch.mbarrier_wait(raw.full_barrier(stage), phase)
    cute.arch.mbarrier_wait(beta.full_barrier(stage), phase)
    if cutlass.const_expr(fence):
        tcgen05_after_thread_sync()
    return stage


@dataclass(frozen=True)
class KdaRecomputeParams:
    k_load: RaggedTma  # (token, dim, head) loads, zero past each sequence end
    v_load: RaggedTma
    g_load: RaggedTma
    tma_ckpt: cute.CopyAtom
    mCkpt: cute.Tensor  # TMA store view (v, k, head, row)
    box_io: cute.ComposedLayout  # one 16 x 64 16-bit swizzled box
    box_g: cute.ComposedLayout  # one 16 x 128-byte gate box
    box_ckpt: cute.ComposedLayout  # one d_v x 64 16-bit swizzled checkpoint box
    beta: cute.Tensor  # (token, head)
    state_in: cute.Tensor | None  # (sequence, head, d_v, d_k)
    work_items: cute.Tensor  # (rows, WORK_ITEM_FIELDS)
    work_count: cute.Tensor  # (1,)
    scheduler: cute.Tensor  # (>= 1,) zeroed ticket counter
    k_ratio: Int32  # heads per k / v head
    v_ratio: Int32
    s_k: Int32  # SMEM byte addresses of the buffers (stage 0)
    s_v: Int32
    s_g: Int32
    s_ex: Int32
    s_kdecay: Int32
    s_krestore: Int32
    s_kinv: Int32
    s_tinv: Int32
    s_ckpt: Int32
    s_beta: Int32


class KdaRecompute(WarpSpecialized):
    """Warpdrive KDA checkpoint-series recompute for ``io`` fp16/bf16 operands, an fp32 or ``io``
    gate and ``d_k, d_v`` in {64, 128}; the initial state dtype (fp32 or bf16) follows the traced
    tensor."""

    def __init__(self, io: str, gate: str, d_k: int, d_v: int, use_initial_state: bool, num_ctas):
        if d_k not in HEAD_DIMS or d_v not in HEAD_DIMS:
            raise ValueError(f"the KDA recompute serves d_k, d_v in {HEAD_DIMS}, got {d_k}, {d_v}")
        if gate not in ("fp32", io):
            raise ValueError(f"the gate must be fp32 or the io dtype, got {gate}")
        self.io, self.gate, self.d_k, self.d_v = io, gate, d_k, d_v
        self.use_initial_state, self.num_ctas = use_initial_state, num_ctas
        # One cg0 body for both instances is 1% faster at d_k = 128 (8x4096 H32: 559 vs 565 us)
        # but 2.5% slower at d_k = 64 (2x8192 H64: 369 vs 360 us).
        self.roles = {"cg0": RoleConfig(CG0_WARPS, regs=CG0_REGS, shared_body=d_k == 128)}
        self.io_dtype = CUTE_DTYPES[io]
        self.gate_dtype = CUTE_DTYPES[gate]
        gate_bytes = self.gate_dtype.width // 8
        # An fp32 gate is exchanged in place (exp2 prefixes over the raw gate stage); a 16-bit
        # gate exchanges through its own 4-stage fp32 buffer.
        self.ex_stages = RAW_STAGES if gate == "fp32" else 4
        io_t, f32 = self.io_dtype, Float32
        plan = self.smem_plan = SmemPlan()
        # Upstream's SMEM order.
        buf = {
            "Kdecay": plan.buffer("Kdecay", io_t, align=1024),
            "Krestore": plan.buffer("Krestore", io_t, align=1024),
            "Tinv": plan.buffer("Tinv", io_t, align=1024),
            "K": plan.buffer("K", io_t, align=1024),
            "V": plan.buffer("V", io_t, align=1024),
            "Gate": plan.buffer("Gate", self.gate_dtype, align=1024),
        }
        if gate != "fp32":
            buf["Exchange"] = plan.buffer("Exchange", f32, align=1024)
        buf["Kinv"] = plan.buffer("Kinv", io_t, align=1024)
        buf["Beta"] = plan.buffer("Beta", f32, align=1024)
        buf["Ckpt"] = plan.buffer("Ckpt", io_t, align=1024)
        self.buf = buf
        rr = Sharing.ROUND_ROBIN
        io_b = io_t.width // 8
        raw_bytes = B_T * (d_k * io_b + d_v * io_b + d_k * gate_bytes)
        # The TMA check reports raw unverified: it cannot follow the per-item domain_offset views.
        self.raw = Ring(
            RAW_STAGES,
            Tma(raw_bytes),
            {"cg0": Warps, "cg1": Warps},
            buffer={k: buf[k] for k in ("K", "V", "Gate")},
        )
        # Signal-only (the upstream gate_exchange_ready): CG0 instances arrive raw, CG1 waits. An
        # fp32 gate exchanges inside the raw gate stage, which the raw ring recycles after CG1's
        # release; a 16-bit gate's own stage is rewritten 4 chunks later by the instance that
        # wrote it, after its chunk c - 2 waited for the left state update of chunk c - 3,
        # which follows CG1's U operand of chunk c - 3 and so its exchange reads of chunk c - 4.
        self.exchange = Ring(
            self.ex_stages,
            consumer=Warps,
            external="producer",
            override={"full_counts": 4},
            buffer=None if gate == "fp32" else buf["Exchange"],
        )
        self.beta = Ring(RAW_STAGES, Warps, Warps, buffer=buf["Beta"])
        self.sched = Ring(SCHED_STAGES, Warps, Warps, payload=(Int32, (1,)))
        smem_commit = "smem_async"  # generic SMEM stores, then tcgen05.mma / TMA reads
        self.kk = Ring(
            DECAY_STAGES,
            Warps,
            {"mma": Umma, "regmma": Warps},
            producer_sharing=rr,
            commit_fence=smem_commit,
            buffer={"Kdecay": buf["Kdecay"], "Kinv": buf["Kinv"]},
        )
        self.restore = Ring(
            DECAY_STAGES,
            Warps,
            Umma,
            producer_sharing=rr,
            commit_fence=smem_commit,
            buffer=buf["Krestore"],
        )
        self.tinv = Ring(INTER_STAGES, Warps, Umma, commit_fence=smem_commit, buffer=buf["Tinv"])
        # A checkpoint row holds both key halves: CG1 writes the right half of every row and the
        # whole seed row (chunk 0), the CG0 instance of chunk c > 0 the left half. CG1 drives the
        # managed producer; CG0 waits on the same empty barrier and adds its 4 warp arrivals raw
        # (as CG1 does on a seed row), so every row completes after 8 (upstream's count).
        self.ckpt = Ring(
            CKPT_STAGES,
            Warps,
            Warps,
            commit_fence=smem_commit,
            override={"full_counts": 8},
            buffer=buf["Ckpt"],
        )
        # The TMEM state recurrence (upstream protocol): signal-only barriers whose stage reuse
        # is implied by the recurrence itself (see ``TSTATE_COUNTS``), so no channel reuse waits
        # sit on the per-chunk critical path.
        self.tstate = Barriers(len(TSTATE_COUNTS), counts=list(TSTATE_COUNTS))
        # Each CG0 instance syncs its four warps between its exchange stores and their reads.
        self.exchange_sync = NamedBarriers(2, threads=128)
        self.tmem_plan = (
            TmemPlan()
            .region("state", d_k)
            .region("sin", d_k // 2)
            .region("ks", B_T)
            .region("u", B_T)
            .region("yin", B_T // 2)
            .region("uin", B_T // 2)
        )
        self.work = WorkStream(schedule=OneShot(grid="1d"), ready=PdlWait())
        # MMA geometry: KS (TMEM state operand x K_decay), U (TMEM Y x the 16 x 16 T tile), the
        # state update (TMEM U x MN-major K_restore), split in two key halves when d_k = 128.
        self.n_update = d_k // 2 if d_k // 2 >= 64 else d_k
        acc_idesc = instr_desc(f32, io_t, io_t, m=d_v, n=B_T)
        self.ks_desc = MmaDesc(M=d_v, N=B_T, K=d_k, bpe_a=2, bpe_b=2, idesc=acc_idesc)
        self.t_desc = MmaDesc(M=d_v, N=B_T, K=B_T, bpe_a=2, bpe_b=2, idesc=acc_idesc)
        self.kr_desc = MmaDesc(
            M=d_v,
            N=self.n_update,
            K=B_T,
            bpe_a=2,
            bpe_b=2,
            btranspose=True,
            idesc=instr_desc(f32, io_t, io_t, m=d_v, n=self.n_update, b_mn_major=True),
        )

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: K, V and gate, keyed by head."""
        return tma_slot_rows(3, B_T, heads)

    def get_name(self) -> str:
        init = "_init" if self.use_initial_state else ""
        return f"wd_kda_recompute_{self.io}_g{self.gate}_k{self.d_k}_v{self.d_v}{init}"

    # ---- work decode -------------------------------------------------------------------

    @cute.jit
    def item(self, params: KdaRecomputeParams, tile):
        """``(batch, head, write_start, write_end, compute_start, batch_start, seqlen)`` of work
        row ``tile``."""
        row = params.work_items
        start = row[tile, 6]
        return (
            row[tile, 0],
            row[tile, 1],
            row[tile, 2],
            row[tile, 3],
            row[tile, 4],
            start,
            row[tile, 7] - start,
        )

    @cute.jit
    def next_tile(self, sched, skip_fence=None):
        """The next tile of the item list; ``skip_fence`` as for the consume (its reason is
        ``AFTER_FENCE_COVERED``/``BEFORE_FENCE_COVERED``)."""
        reason = BEFORE_FENCE_COVERED if skip_fence == "before" else AFTER_FENCE_COVERED
        with sched.consume(skip_fence=skip_fence, reason=reason) as h:
            tile = h.data[0]
        return sched, tile

    @cute.jit
    def first_stored(self, write_start, compute_start):
        """First chunk whose checkpoint row an item stores: chunk 0 (the seed or zeros, staged
        with the item's first chunk) when it writes from 0, else the first chunk at or past
        ``write_start`` after its own first (the right half is staged with a later chunk)."""
        first = cutlass.max(write_start, compute_start + 1)
        if write_start == 0:
            first = Int32(0)
        return first

    @cute.jit
    def _state_row(self, lane, warp):
        """``(v row, row valid)`` of this thread's TMEM lane: M = 64 accumulators use the first
        16 lanes of each warp."""
        if cutlass.const_expr(self.d_v == 128):
            return warp * 32 + lane, cutlass.Boolean(True)
        return warp * 16 + lane % 16, lane < 16

    @cute.jit
    def _ckpt_offset(self, stage, row, dk):
        """Byte offset of checkpoint ``(row, dk..dk + 7)`` in staging ``stage``: d_v x 64 boxes
        of 128-byte swizzled rows, one per 64 key columns (the TMA store's boxes)."""
        idx = stage * (self.d_k * self.d_v) + (dk // 64) * (self.d_v * 64) + row * 64
        return 2 * (idx + swz128(row, dk % 64))

    # ---- load (warp 14): beta one chunk ahead, K/Gate/V TMA, next item -------------------

    @role(warps=14, regs=56)
    @cute.jit
    def load(
        self,
        params: KdaRecomputeParams,
        *,
        raw: Producer,
        beta: Producer,
        sched: Producer,
        work: WorkCursor,
        me: Local,
    ):
        lane = me.lane
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                fields = self.item(params, tile)
                raw, beta = self.load_item(params, fields, raw, beta, lane)
                with sched.produce() as sh:
                    if elect_sync():
                        sh.data[0] = Int32(self.num_ctas) + cute.arch.atomic_add(
                            params.scheduler.iterator.llvm_ptr,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                    cute.arch.sync_warp()
                    tile = sh.data[0]
            work.advance()
        cute.arch.griddepcontrol_launch_dependents()
        return raw, beta, sched, work

    @cute.jit
    def load_item(self, params, fields, raw, beta, lane):
        """Beta (one chunk ahead) and the K, Gate and V boxes of every chunk of one item."""
        _b, head, _ws, write_end, compute_start, start, seqlen = fields
        head_k = head // params.k_ratio
        head_v = head // params.v_ratio
        loads = (params.k_load, params.g_load, params.v_load)
        descs = sequence_descriptors(loads, start + seqlen, head, B_T, write_end > compute_start)
        carry = Float32(0.0)
        token = compute_start * B_T + lane
        if lane < B_T and token < seqlen:
            carry = params.beta[start + token, head].to(Float32)
        for chunk in cutlass.range(compute_start, write_end, 1, unroll=1):
            with beta.produce() as bh:
                bh.access(write=True, by="threads")
                if lane < B_T:
                    sts_f32(params.s_beta + bh.index * (B_T * 4) + 4 * lane, carry)
            carry = Float32(0.0)
            token = (chunk + 1) * B_T + lane
            if lane < B_T and token < seqlen:
                carry = params.beta[start + token, head].to(Float32)
            rows = (start + chunk * B_T, cutlass.min(seqlen - chunk * B_T, B_T))
            raw = self.load_chunk(params, raw, rows, head_k, head_v, head, descs)
        return raw, beta

    @cute.jit
    def load_chunk(self, params, raw, rows, head_k, head_v, head, descs):
        """TMA one chunk's K, Gate and V boxes (tokens ``rows = (token, valid)``, zero past
        them) into one ``raw`` stage through the item's ``descs``."""
        k_desc, g_desc, v_desc = descs
        io_b = self.io_dtype.width // 8
        g_cols = 128 // (self.gate_dtype.width // 8)
        box = B_T * 128  # bytes of one 16-row 128-byte box
        with raw.produce() as h:
            h.access("K", write=True, by="tma")
            h.access("V", write=True, by="tma")
            h.access("Gate", write=True, by="tma")
            k_stage = h.index * (self.d_k * B_T * io_b)
            for kb in cutlass.range_constexpr(self.d_k // 64):
                load_rows(
                    params.k_load,
                    self.io_dtype,
                    params.s_k + k_stage + kb * box,
                    params.box_io,
                    rows,
                    head_k,
                    kb,
                    64,
                    h.barrier,
                    k_desc,
                )
            g_stage = h.index * (self.d_k * B_T * (self.gate_dtype.width // 8))
            for kb in cutlass.range_constexpr(self.d_k // g_cols):
                load_rows(
                    params.g_load,
                    self.gate_dtype,
                    params.s_g + g_stage + kb * box,
                    params.box_g,
                    rows,
                    head,
                    kb,
                    g_cols,
                    h.barrier,
                    g_desc,
                )
            v_stage = h.index * (self.d_v * B_T * io_b)
            for kb in cutlass.range_constexpr(self.d_v // 64):
                load_rows(
                    params.v_load,
                    self.io_dtype,
                    params.s_v + v_stage + kb * box,
                    params.box_io,
                    rows,
                    head_v,
                    kb,
                    64,
                    h.barrier,
                    v_desc,
                )
        return raw

    # ---- compute group 0 (warps 0-7): gate scan, decay operands, left state half --------

    @role(warps=CG0_WARPS, regs=CG0_REGS)
    @cute.jit
    def cg0(
        self,
        params: KdaRecomputeParams,
        *,
        stage: Instance,
        raw: Consumer,
        kk: Producer,
        restore: Producer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        d_k = self.d_k
        channel_rows = d_k // 4
        if cutlass.const_expr(channel_rows < 32):
            channel = warp * channel_rows + lane % channel_rows
        else:
            channel = warp * 32 + lane
        segment = channel // 32
        seg_base = segment * (B_T * 32)
        col = channel - segment * 32
        prefix_offsets = [opaque_i32(seg_base + swz128(j ^ segment, col, 4)) for j in range(8)]
        if cutlass.const_expr(self.gate == "fp32"):
            gate_offsets = [opaque_i32(seg_base + swz128(j, col, 4)) for j in range(8)]
        else:
            raw_seg = channel // 64
            raw_col = channel - raw_seg * 64
            gate_offsets = [
                opaque_i32(raw_seg * (B_T * 64) + swz128(j, raw_col, 2)) for j in range(8)
            ]
        decay_row = warp * 4 + lane // 8
        lane_in_row = lane - (lane // 8) * 8
        lane_base = tmem.base.toint() + (
            (32 * (warp % 4)) << 16
        )  # warp-uniform: this subpartition
        state_col = lane_base + tmem.cols("state")
        sin_col = lane_base + tmem.cols("sin")
        ckpt_row, ckpt_valid = self._state_row(lane, warp % 4)
        barrier = self.exchange_sync[stage]
        raw_next = Int32(0)  # next raw epoch this instance takes (both instances join raw)
        rows_before = Int32(0)  # checkpoint rows of the CTA's earlier items
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            cum_base = Int32(0)
            while tile < total:
                _b, _h, write_start, write_end, compute_start, _s, seqlen = self.item(params, tile)
                n_local = write_end - compute_start
                first = self.first_stored(write_start, compute_start)
                for local in cutlass.range((stage + cum_base) & 1, n_local, 2, unroll=1):
                    c = cum_base + local
                    if raw_next < c:  # the other instance's chunk: release its stage
                        raw.consume_unread(skip_fence="both", reason=BOTH_FENCES_COVERED)
                    raw_next = c + 1
                    chunk = compute_start + local
                    raw, kk, restore = self.cg0_chunk(
                        params,
                        raw,
                        kk,
                        restore,
                        tstate,
                        c,
                        barrier,
                        chunk * B_T,
                        seqlen,
                        local,
                        chunk >= write_start,
                        rows_before + chunk - first,
                        lane,
                        decay_row,
                        lane_in_row,
                        prefix_offsets,
                        gate_offsets,
                        seg_base,
                        segment,
                        col,
                        state_col,
                        sin_col,
                        ckpt_row,
                        ckpt_valid,
                    )
                if write_end > first:
                    rows_before += write_end - first
                cum_base += n_local
                sched, tile = self.next_tile(sched, skip_fence="before")
            work.advance()
        return raw, kk, restore, sched, work

    @cute.jit
    def cg0_chunk(
        self,
        params,
        raw,
        kk,
        restore,
        tstate,
        c,
        barrier,
        chunk_start,
        seqlen,
        local,
        store_row,
        row_index,
        lane,
        decay_row,
        lane_in_row,
        prefix_offsets,
        gate_offsets,
        seg_base,
        segment,
        col,
        state_col,
        sin_col,
        ckpt_row,
        ckpt_valid,
    ):
        io = self.io_dtype
        d_k = self.d_k
        dk_halves = d_k // 64
        channel_rows = d_k // 4
        store_rows = B_T * channel_rows // 32
        store_row_base = (lane // channel_rows) * store_rows
        rh = raw.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
        rh.access("K", read=True, by="threads")
        rh.access("Gate", read=True, by="threads")
        g_stage_bytes = d_k * B_T * (self.gate_dtype.width // 8)
        s_gate = params.s_g + rh.index * g_stage_bytes
        # ---- gate prefix scan (log2 domain), exp2 -------------------------------------------
        g = cute.make_rmem_tensor((B_T,), Float32)
        for row in cutlass.range_constexpr(B_T):
            if cutlass.const_expr(self.gate == "fp32"):
                raw_g = lds_f32(s_gate + 4 * (gate_offsets[row % 8] + row * 32))
            else:
                raw_g = lds_16_f32(
                    s_gate + 2 * (gate_offsets[row % 8] + row * 64), self.gate_dtype
                )
            g[row] = raw_g * Float32(LOG2_E)
        if cutlass.const_expr(self.gate == "fp32"):
            # The exp2(g) stores below go in place over this raw gate stage, into slots that
            # only this warp read: the warp sync orders every lane's reads before them.
            cute.arch.sync_warp()
        ragged = chunk_start + B_T > seqlen
        if ragged:  # padded rows carry no decay
            for row in cutlass.range_constexpr(B_T):
                g[row] = Float32(0.0) if chunk_start + row >= seqlen else g[row]
        acc = Float32(0.0)
        for pair in cutlass.range_constexpr(B_T // 2):
            g0 = g[2 * pair]
            g1 = g[2 * pair + 1]
            p0, pair_sum = cute.arch.add_packed_f32x2((acc, g0), (g0, g1))
            p1 = acc + pair_sum
            g[2 * pair] = p0
            g[2 * pair + 1] = p1
            acc = p1
        for row in cutlass.range_constexpr(B_T):
            g[row] = cute.math.exp2(g[row], fastmath=True)
        ex_stage = c % self.ex_stages  # == rh.index for an fp32 gate
        if cutlass.const_expr(self.gate == "fp32"):
            s_ex = s_gate  # in place over this raw gate stage
        else:
            s_ex = params.s_ex + ex_stage * (d_k * B_T * 4)
        for row_off in cutlass.range_constexpr(store_rows):
            if cutlass.const_expr(channel_rows < 32):
                row = store_row_base + row_off
                value = g[row_off + B_T // 2] if lane >= channel_rows else g[row_off]
                idx = seg_base + swz128(row ^ segment, row * 32 + col, 4)
            else:
                value = g[row_off]
                idx = prefix_offsets[row_off % 8] + row_off * 32
            sts_f32(s_ex + 4 * idx, value)
        barrier.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
        warp_arrive(self.barrier_ptr("exchange") + ex_stage, fence=False)

        # ---- raw K rows (zeroed past the sequence end), exp2(g) of the row --------------------
        k_stage = rh.index * (d_k * B_T * 2)
        k_words = []
        row_valid = chunk_start + decay_row < seqlen
        for half in cutlass.range_constexpr(dk_halves):
            idx = box128(decay_row, half * 64 + lane_in_row * 8)
            kw = lds_v4(params.s_k + k_stage + 2 * idx)
            if ragged:
                for i in cutlass.range_constexpr(4):
                    kw[i] = kw[i] if row_valid else Int32(0)
            k_words += kw
        raw_k = []
        for i in cutlass.range_constexpr(dk_halves * 4):
            lo, hi = f16x2_to_f32(k_words[i], dtype=io)
            raw_k += [lo, hi]
        exp_g = []
        for half in cutlass.range_constexpr(dk_halves):
            dim_base = half * 64 + lane_in_row * 8
            for grp in cutlass.range_constexpr(2):
                f_dim = dim_base + grp * 4
                f_seg = f_dim // 32
                idx = (
                    f_seg * (B_T * 32)
                    + decay_row * 32
                    + swz128(decay_row ^ f_seg, f_dim - f_seg * 32, 4)
                )
                exp_g += lds_v4_f32(s_ex + 4 * idx)

        # ---- K decay / K inv / K restore ---------------------------------------------------
        kd_pack = []
        ki_pack = []
        kr_pack = []
        for half in cutlass.range_constexpr(dk_halves):
            dim_base = half * 64 + lane_in_row * 8
            exp_last = []
            for grp in cutlass.range_constexpr(2):
                f_dim = dim_base + grp * 4
                f_seg = f_dim // 32
                idx = (
                    f_seg * (B_T * 32)
                    + (B_T - 1) * 32
                    + swz128((B_T - 1) ^ f_seg, f_dim - f_seg * 32, 4)
                )
                exp_last += lds_v4_f32(s_ex + 4 * idx)
            for p in cutlass.range_constexpr(4):
                r0 = half * 8 + 2 * p
                kd0, kd1 = fmul2(raw_k[r0], raw_k[r0 + 1], exp_g[r0], exp_g[r0 + 1])
                kd_pack.append(fp32_to_fp16(kd0, kd1, dtype=io))
                inv0 = cute.math.rcp(exp_g[r0], approx=True, ftz=True)
                inv1 = cute.math.rcp(exp_g[r0 + 1], approx=True, ftz=True)
                ki0, ki1 = fmul2(raw_k[r0], raw_k[r0 + 1], inv0, inv1)
                ki_pack.append(fp32_to_fp16(ki0, ki1, dtype=io))
                kr0, kr1 = fmul2(ki0, ki1, exp_last[2 * p], exp_last[2 * p + 1])
                kr_pack.append(fp32_to_fp16(kr0, kr1, dtype=io))
        kh = kk.acquire(skip_fence="both", reason=BOTH_FENCES_COVERED)
        kh.access("Kdecay", write=True, by="threads")
        kh.access("Kinv", write=True, by="threads")
        decay_stage = kh.index * (d_k * B_T * 2)
        for half in cutlass.range_constexpr(dk_halves):
            off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
            st_shared_v4(params.s_kinv + off, ki_pack[4 * half : 4 * half + 4])
            st_shared_v4(params.s_kdecay + off, kd_pack[4 * half : 4 * half + 4])
        kh.commit()
        with restore.produce(skip_fence="before", reason=BEFORE_FENCE_COVERED) as qh:
            qh.access(write=True, by="threads")
            decay_stage = qh.index * (d_k * B_T * 2)
            for half in cutlass.range_constexpr(dk_halves):
                off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
                st_shared_v4(params.s_krestore + off, kr_pack[4 * half : 4 * half + 4])

        # ---- left key half of the state: io operand, decay by exp2(g_last), checkpoint ----
        if c > 0:  # the previous chunk's left update has landed
            with self.trace_region("sacc_l.wait"):
                spin_wait(tstate + SACC_L + ((c - 1) & 1), ((c - 1) >> 1) & 1)
        if local > 0:
            blocks = [tmem_ld("32x32b", 32, state_col + b * 32) for b in range(dk_halves)]
            packed = []
            for b in cutlass.range_constexpr(dk_halves):
                words = [
                    fp32_to_fp16(blocks[b][2 * i], blocks[b][2 * i + 1], dtype=io)
                    for i in range(16)
                ]
                tmem_st("32x32b", sin_col + b * 16, words)
                packed.append(words)
            cute.arch.fence_view_async_tmem_store()
            warp_arrive(tstate + SIN_L)
            for b in cutlass.range_constexpr(dk_halves):
                scaled = []
                for grp in cutlass.range_constexpr(8):
                    dim = b * 32 + grp * 4
                    seg = dim // 32
                    idx = seg * (B_T * 32) + (B_T - 1) * 32 + swz128((B_T - 1) ^ seg, dim % 32, 4)
                    f = lds_v4_f32(s_ex + 4 * idx)
                    for t in cutlass.range_constexpr(2):
                        s0, s1 = fmul2(
                            blocks[b][grp * 4 + 2 * t],
                            blocks[b][grp * 4 + 2 * t + 1],
                            f[2 * t],
                            f[2 * t + 1],
                        )
                        scaled += [s0, s1]
                tmem_st("32x32b", state_col + b * 32, scaled)
            cute.arch.fence_view_async_tmem_store()
            if store_row:
                self.ckpt_left(params, row_index, packed, ckpt_row, ckpt_valid)
        warp_arrive(tstate + UIN)  # the left half is decayed (with CG1's U operand)
        rh.release()
        return raw, kk, restore

    @cute.jit
    def ckpt_left(self, params, row_index, packed, ckpt_row, ckpt_valid):
        """The left key half of checkpoint row ``row_index`` (the CTA's row count), from the
        packed state operand words: raw side of the ``ckpt`` ring (see ``__init__``)."""
        stage = row_index % CKPT_STAGES
        full = self.barrier_ptr("ckpt")
        spin_wait(full + CKPT_STAGES + stage, ((row_index // CKPT_STAGES) + 1) & 1, fence=False)
        if ckpt_valid:
            for b in cutlass.range_constexpr(self.d_k // 64):
                for wg in cutlass.range_constexpr(4):
                    st_shared_v4(
                        params.s_ckpt + self._ckpt_offset(stage, ckpt_row, b * 32 + wg * 8),
                        packed[b][wg * 4 : wg * 4 + 4],
                    )
        cute.arch.fence_view_async_shared()
        warp_arrive(full + stage)

    # ---- compute group 1 (warps 8-11): seed, right state half, Y / U staging -------------

    @role(warps=range(8, 12), regs=136)
    @cute.jit
    def cg1(
        self,
        params: KdaRecomputeParams,
        *,
        raw: Consumer,
        beta: Consumer,
        exchange: Consumer,
        ckpt: Producer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        ov_row = (lane // 16) * 8 + (lane & 7)
        ov_col = ((lane // 8) & 1) * 8
        value_dim, row_valid = self._state_row(lane, warp)
        value_base = warp * (32 if self.d_v == 128 else 16)
        lane_base = tmem.base.toint() + (
            (32 * (warp % 4)) << 16
        )  # warp-uniform: this subpartition
        cols = {name: tmem.cols(name) for name in ("state", "sin", "ks", "u", "yin", "uin")}
        ov_lo = box128(ov_row, value_base + ov_col)
        ov_hi = box128(ov_row, value_base + 16 + ov_col)
        geo = (lane, lane_base, cols, ov_lo, ov_hi, value_dim, row_valid)
        # Completed waits on the KS and U accumulators and on the right-half state updates.
        n_ks = Int32(0)
        n_u = Int32(0)
        n_upd = Int32(0)
        n_raw = Int32(0)  # chunks consumed from raw and beta (see _raw_beta_wait)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                batch, head, write_start, write_end, compute_start, _s, seqlen = self.item(
                    params, tile
                )
                n_local = write_end - compute_start
                if n_local > 0:
                    exchange, ckpt, n_ks, n_u = self.cg1_first(
                        params,
                        raw,
                        beta,
                        exchange,
                        ckpt,
                        tstate,
                        n_raw,
                        n_ks,
                        n_u,
                        geo,
                        batch,
                        head,
                        write_start,
                        compute_start,
                        seqlen,
                    )
                for local in cutlass.range(1, n_local, 1, unroll=1):
                    chunk = compute_start + local
                    exchange, ckpt, n_ks, n_u, n_upd = self.cg1_chunk(
                        params,
                        raw,
                        beta,
                        exchange,
                        ckpt,
                        tstate,
                        n_raw + local,
                        n_ks,
                        n_u,
                        n_upd,
                        geo,
                        chunk >= write_start,
                        chunk * B_T,
                        seqlen,
                    )
                n_raw += n_local
                if n_local > 0:
                    with self.trace_region("sacc_r.wait"):
                        # The last update; the next item's fenced waits precede its TMEM accesses.
                        spin_wait(tstate + SACC_R + (n_upd & 1), (n_upd >> 1) & 1, fence=False)
                    n_upd += 1
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, beta, exchange, ckpt, sched, work

    @cute.jit
    def cg1_first(
        self,
        params,
        raw,
        beta,
        exchange,
        ckpt,
        tstate,
        n_raw,
        n_ks,
        n_u,
        geo,
        batch,
        head,
        write_start,
        compute_start,
        seqlen,
    ):
        """An item's first chunk: the seed of both state halves (or zeros) and the chunk-0
        checkpoint row, Y (carrying the seeded state), U."""
        lane, lane_base, cols, _lo, _hi, _vd, _rv = geo
        io = self.io_dtype
        # Without an initial state no seed loads TMEM before the next fenced wait.
        skip = None if self.use_initial_state else "after"
        eh = exchange.wait(skip_fence=skip, reason=AFTER_FENCE_COVERED)
        s_ex = params.s_g + eh.index * (self.d_k * B_T * 4)
        if cutlass.const_expr(self.gate != "fp32"):
            s_ex = params.s_ex + eh.index * (self.d_k * B_T * 4)
        seed = cutlass.Boolean(False)
        if cutlass.const_expr(self.use_initial_state):
            seed = compute_start == 0
            if seed:
                if write_start == 0:
                    ckpt = self.seed_state(params, ckpt, geo, tstate, batch, head, s_ex, True)
                else:
                    ckpt = self.seed_state(params, ckpt, geo, tstate, batch, head, s_ex, False)
        else:
            if write_start == 0:
                ckpt = self.zero_row(params, ckpt, geo)
        chunk_start = compute_start * B_T
        stage = _raw_beta_wait(raw, beta, n_raw)
        v_lo, v_hi = self.load_v(params, stage, geo, chunk_start, seqlen)
        s_beta = params.s_beta + stage * (B_T * 4)
        betas = []
        for r in cutlass.range_constexpr(4):
            token0 = ((r // 2) * 4 + (lane & 3)) * 2
            betas.append((lds_f32(s_beta + 4 * token0), lds_f32(s_beta + 4 * token0 + 4)))
        y_lo = [Int32(0)] * 4
        y_hi = [Int32(0)] * 4
        if seed:
            with self.trace_region("ks.wait"):
                spin_wait(tstate + KS, n_ks & 1)
            n_ks += 1
            ks_lo = tmem_ld("16x256b", 2, lane_base + cols["ks"])
            for r in cutlass.range_constexpr(4):
                y_lo[r] = beta_residual(
                    v_lo[r], *betas[r], ks_lo[2 * r], ks_lo[2 * r + 1], io, True
                )
            if cutlass.const_expr(self.d_v == 128):
                ks_hi = tmem_ld("16x256b", 2, lane_base + (16 << 16) + cols["ks"])
                for r in cutlass.range_constexpr(4):
                    y_hi[r] = beta_residual(
                        v_hi[r], *betas[r], ks_hi[2 * r], ks_hi[2 * r + 1], io, True
                    )
        else:
            for r in cutlass.range_constexpr(4):
                y_lo[r] = beta_residual(v_lo[r], *betas[r], None, None, io, False)
                if cutlass.const_expr(self.d_v == 128):
                    y_hi[r] = beta_residual(v_hi[r], *betas[r], None, None, io, False)
        self.stage_y(tstate, geo, y_lo, y_hi)
        raw_beta_release(raw, beta, stage)
        n_u = self.stage_u(tstate, n_u, geo)
        eh.release(signal=False)  # signal-only channel (see __init__)
        return exchange, ckpt, n_ks, n_u

    @cute.jit
    def seed_state(
        self, params, ckpt, geo, tstate, batch, head, s_ex, with_row: cutlass.Constexpr[bool]
    ):
        """Initial state row -> io state operand (left half, then right, each published as soon
        as it is stored), fp32 state decayed by the first chunk's exp2(g_last), and with
        ``with_row`` (the item writes chunk 0) the chunk-0 checkpoint row from the operand
        words."""
        _l, lane_base, cols, _lo, _hi, value_dim, row_valid = geo
        io = self.io_dtype
        st = params.state_in
        addr = (st.iterator + st.layout((batch, head, value_dim, 0))).toint()
        sbytes = st.element_type.width // 8
        half_blocks = self.d_k // 32
        for half in cutlass.range_constexpr(2):
            blocks = range(half * half_blocks, (half + 1) * half_blocks)
            vecs = []
            for i in cutlass.range_constexpr(blocks.start, blocks.stop):
                vecs.append(load_state_16(addr + i * 16 * sbytes, st.element_type))
            packed = []
            for j, i in enumerate(blocks):
                words = [
                    fp32_to_fp16(vecs[j][2 * p], vecs[j][2 * p + 1], dtype=io) for p in range(8)
                ]
                tmem_st("32x32b", lane_base + cols["sin"] + i * 8, words)
                packed.append(words)
            cute.arch.fence_view_async_tmem_store()
            warp_arrive(tstate + (SIN_L if half == 0 else SIN_R))
            for j, i in enumerate(blocks):
                scaled = []
                for grp in cutlass.range_constexpr(4):
                    dim = i * 16 + grp * 4
                    seg = dim // 32
                    idx = seg * (B_T * 32) + (B_T - 1) * 32 + swz128((B_T - 1) ^ seg, dim % 32, 4)
                    f = lds_v4_f32(s_ex + 4 * idx)
                    for t in cutlass.range_constexpr(2):
                        s0, s1 = fmul2(
                            vecs[j][grp * 4 + 2 * t],
                            vecs[j][grp * 4 + 2 * t + 1],
                            f[2 * t],
                            f[2 * t + 1],
                        )
                        scaled += [s0, s1]
                tmem_st("32x32b", lane_base + cols["state"] + i * 16, scaled)
            if cutlass.const_expr(with_row):
                if cutlass.const_expr(half == 0):
                    ch = ckpt.acquire()
                    ch.access(write=True, by="threads")
                if row_valid:
                    for j, i in enumerate(blocks):
                        for wg in cutlass.range_constexpr(2):
                            st_shared_v4(
                                params.s_ckpt
                                + self._ckpt_offset(ch.index, value_dim, i * 16 + wg * 8),
                                packed[j][wg * 4 : wg * 4 + 4],
                            )
        cute.arch.fence_view_async_tmem_store()
        if cutlass.const_expr(with_row):
            ckpt = self.commit_whole_row(ch, ckpt)
        return ckpt

    @cute.jit
    def zero_row(self, params, ckpt, geo):
        """Chunk 0's checkpoint row with no initial state: zeros."""
        _l, _lb, _c, _lo, _hi, value_dim, row_valid = geo
        ch = ckpt.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        ch.access(write=True, by="threads")
        zero = Int32(0)
        if row_valid:
            for i in cutlass.range_constexpr(self.d_k // 8):
                st_shared_v4(
                    params.s_ckpt + self._ckpt_offset(ch.index, value_dim, i * 8),
                    [zero, zero, zero, zero],
                )
        return self.commit_whole_row(ch, ckpt)

    @cute.jit
    def commit_whole_row(self, ch, ckpt):
        """Publish a row CG1 wrote alone: CG0's 4 arrivals (raw, after this group's fence) and
        the managed commit's 4."""
        cute.arch.fence_view_async_shared()
        warp_arrive(ch.full_barrier)
        ch.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        return ckpt

    @cute.jit
    def load_v(self, params, stage, geo, chunk_start, seqlen):
        """This thread's transposed V pairs (``ldmatrix.trans``), zeroed past the sequence end."""
        lane, _lb, _c, ov_lo, ov_hi, _vd, _rv = geo
        s_v = params.s_v + stage * (self.d_v * B_T * 2)
        v_lo = ldmatrix(s_v + 2 * ov_lo, 4, trans=True)
        v_hi = [Int32(0)] * 4
        if cutlass.const_expr(self.d_v == 128):
            v_hi = ldmatrix(s_v + 2 * ov_hi, 4, trans=True)
        if chunk_start + B_T > seqlen:
            valid = seqlen - chunk_start
            for r in cutlass.range_constexpr(4):
                token0 = ((r // 2) * 4 + (lane & 3)) * 2
                v_lo[r] = mask_pair(v_lo[r], token0, valid)
                v_hi[r] = mask_pair(v_hi[r], token0, valid)
        return v_lo, v_hi

    @cute.jit
    def stage_y(self, tstate, geo, y_lo, y_hi):
        """Publish the packed Y operand to TMEM."""
        _l, lane_base, cols, _lo, _hi, _vd, _rv = geo
        tmem_st("16x128b", lane_base + cols["yin"], y_lo)
        if cutlass.const_expr(self.d_v == 128):
            tmem_st("16x128b", lane_base + (16 << 16) + cols["yin"], y_hi)
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + YIN)

    @cute.jit
    def stage_u(self, tstate, n_u, geo):
        """Wait for U = T Y and publish its io copy as the state update's operand."""
        _l, lane_base, cols, _lo, _hi, _vd, _rv = geo
        with self.trace_region("u.wait"):
            spin_wait(tstate + U, n_u & 1)
        vals = tmem_ld("32x32b", B_T, lane_base + cols["u"])
        words = [fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=self.io_dtype) for i in range(8)]
        tmem_st("32x32b", lane_base + cols["uin"], words)
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + UIN)
        return n_u + 1

    @cute.jit
    def cg1_chunk(
        self,
        params,
        raw,
        beta,
        exchange,
        ckpt,
        tstate,
        n_raw,
        n_ks,
        n_u,
        n_upd,
        geo,
        store_row,
        chunk_start,
        seqlen,
    ):
        """A chunk after the first: right state half (operand, decay, checkpoint), Y, U."""
        lane, lane_base, cols, _lo, _hi, value_dim, row_valid = geo
        io = self.io_dtype
        d_k = self.d_k
        half_blocks = d_k // 32
        stage = _raw_beta_wait(raw, beta, n_raw, fence=False)
        with self.trace_region("sacc_r.wait"):
            spin_wait(tstate + SACC_R + (n_upd & 1), (n_upd >> 1) & 1)
        n_upd += 1
        state = [
            tmem_ld("32x32b", 16, lane_base + cols["state"] + (half_blocks + j) * 16)
            for j in range(half_blocks)
        ]
        packed = []
        for j in cutlass.range_constexpr(half_blocks):
            words = [
                fp32_to_fp16(state[j][2 * p], state[j][2 * p + 1], dtype=io) for p in range(8)
            ]
            tmem_st("32x32b", lane_base + cols["sin"] + (half_blocks + j) * 8, words)
            packed.append(words)
        eh = exchange.wait()
        s_ex = params.s_g + eh.index * (d_k * B_T * 4)
        if cutlass.const_expr(self.gate != "fp32"):
            s_ex = params.s_ex + eh.index * (d_k * B_T * 4)
        scaled = []
        for j in cutlass.range_constexpr(half_blocks):
            vals = []
            for grp in cutlass.range_constexpr(4):
                dim = (half_blocks + j) * 16 + grp * 4
                seg = dim // 32
                idx = seg * (B_T * 32) + (B_T - 1) * 32 + swz128((B_T - 1) ^ seg, dim % 32, 4)
                f = lds_v4_f32(s_ex + 4 * idx)
                for t in cutlass.range_constexpr(2):
                    s0, s1 = fmul2(
                        state[j][grp * 4 + 2 * t],
                        state[j][grp * 4 + 2 * t + 1],
                        f[2 * t],
                        f[2 * t + 1],
                    )
                    vals += [s0, s1]
            scaled.append(vals)
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + SIN_R)
        for j in cutlass.range_constexpr(half_blocks):
            tmem_st("32x32b", lane_base + cols["state"] + (half_blocks + j) * 16, scaled[j])
        # ---- the right key half of the chunk's checkpoint row -------------------------------
        if store_row:
            with ckpt.produce(skip_fence="after", reason=AFTER_FENCE_COVERED) as ch:
                ch.access(write=True, by="threads")
                if row_valid:
                    for j in cutlass.range_constexpr(half_blocks):
                        for wg in cutlass.range_constexpr(2):
                            dk = (half_blocks + j) * 16 + wg * 8
                            st_shared_v4(
                                params.s_ckpt + self._ckpt_offset(ch.index, value_dim, dk),
                                packed[j][wg * 4 : wg * 4 + 4],
                            )
        # ---- Y = beta (V - K_decay S), U -------------------------------------------------------
        v_lo, v_hi = self.load_v(params, stage, geo, chunk_start, seqlen)
        with self.trace_region("ks.wait"):
            spin_wait(tstate + KS, n_ks & 1)
        n_ks += 1
        ks_lo = tmem_ld("16x256b", 2, lane_base + cols["ks"])
        ks_hi = [Float32(0.0)] * 8
        if cutlass.const_expr(self.d_v == 128):
            ks_hi = tmem_ld("16x256b", 2, lane_base + (16 << 16) + cols["ks"])
        s_beta = params.s_beta + stage * (B_T * 4)
        y_lo = []
        y_hi = [Int32(0)] * 4
        for r in cutlass.range_constexpr(4):
            token0 = ((r // 2) * 4 + (lane & 3)) * 2
            b0 = lds_f32(s_beta + 4 * token0)
            b1 = lds_f32(s_beta + 4 * token0 + 4)
            y_lo.append(beta_residual(v_lo[r], b0, b1, ks_lo[2 * r], ks_lo[2 * r + 1], io, True))
            if cutlass.const_expr(self.d_v == 128):
                y_hi[r] = beta_residual(v_hi[r], b0, b1, ks_hi[2 * r], ks_hi[2 * r + 1], io, True)
        self.stage_y(tstate, geo, y_lo, y_hi)
        raw_beta_release(raw, beta, stage)
        n_u = self.stage_u(tstate, n_u, geo)
        eh.release(signal=False)  # signal-only channel (see __init__)
        return exchange, ckpt, n_ks, n_u, n_upd

    # ---- register MMA (warp 12): KK, L, T_inv ---------------------------------------------

    @role(warps=12, regs=56)
    @cute.jit
    def regmma(
        self,
        params: KdaRecomputeParams,
        *,
        kk: Consumer,
        beta: Consumer,
        tinv: Producer,
        sched: Consumer,
        work: WorkCursor,
        me: Local,
    ):
        lane = me.lane
        io = self.io_dtype
        ki_row = lane % 8 + (8 if lane // 16 else 0)
        ki_col = 8 if (lane // 8) % 2 else 0
        kd_row = lane % 8 + (8 if (lane // 8) % 2 else 0)
        kd_col = 8 if (lane // 8) // 2 else 0
        t_row = (lane & 7) + (8 if (lane // 8) & 1 else 0)
        t_col = 8 if lane // 8 >= 2 else 0
        t_idx = t_row * B_T + swz32(t_row, t_col)
        ki_offs = [opaque_i32(ki_row * 64 + swz128(ki_row, i * 16 + ki_col)) for i in range(4)]
        kd_offs = [opaque_i32(swz128(kd_row, kd_row * 64 + i * 16 + kd_col)) for i in range(4)]
        row_lo = lane // 4
        row_hi = row_lo + 8
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, _h, _ws, write_end, compute_start, _s, _len = self.item(params, tile)
                for _ in cutlass.range(write_end - compute_start, unroll=1):
                    kh = kk.wait()
                    kh.access("Kdecay", read=True, by="threads")
                    kh.access("Kinv", read=True, by="threads")
                    stage = kh.index * (self.d_k * B_T * 2)
                    acc = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        acc[i] = Float32(0.0)
                    for i in cutlass.range_constexpr(self.d_k // 16):
                        box = (i // 4) * (B_T * 64)
                        kif = ldmatrix(params.s_kinv + stage + 2 * (ki_offs[i % 4] + box), 4)
                        kdf = ldmatrix(params.s_kdecay + stage + 2 * (kd_offs[i % 4] + box), 4)
                        mma_step(acc, kdf, kif, k_step=0, M=16, N=16, dtype=io)
                    bh = beta.wait()
                    bh.access(read=True, by="threads")
                    s_beta = params.s_beta + bh.index * (B_T * 4)
                    beta_lo = lds_f32(s_beta + 4 * row_lo)
                    beta_hi = lds_f32(s_beta + 4 * row_hi)
                    l_regs = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        r = row_hi if cutlass.const_expr(i % 4 >= 2) else row_lo
                        c = (i // 4) * 8 + 2 * (lane % 4) + (i % 2)
                        l_regs[i] = acc[i] if r > c else Float32(0.0)
                    for p in cutlass.range_constexpr(4):
                        bs = beta_hi if cutlass.const_expr(p % 2 == 1) else beta_lo
                        l_regs[2 * p], l_regs[2 * p + 1] = fmul2(
                            l_regs[2 * p], l_regs[2 * p + 1], bs, bs
                        )
                    bh.release()
                    t = invert_unit_lower_16x16_fragments(l_regs, lane, io)
                    with tinv.produce() as th:
                        th.access(write=True, by="threads")
                        stmatrix(
                            params.s_tinv + th.index * (B_T * B_T * 2) + 2 * t_idx,
                            [fp32_to_fp16(t[2 * i], t[2 * i + 1], dtype=io) for i in range(4)],
                        )
                    kh.release()
                sched, tile = self.next_tile(sched)
            work.advance()
        return kk, beta, tinv, sched, work

    # ---- store (warp 15): checkpoint TMA stores -------------------------------------------

    @role(warps=15, regs=56)
    @cute.jit
    def store(
        self,
        params: KdaRecomputeParams,
        *,
        ckpt: Consumer,
        sched: Consumer,
        work: WorkCursor,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                batch, head, write_start, write_end, compute_start, start, _len = self.item(
                    params, tile
                )
                first_row = start // B_T + batch
                first = self.first_stored(write_start, compute_start)
                for chunk in cutlass.range(first, write_end, 1, unroll=1):
                    ckpt = self.store_row(params, ckpt, head, first_row + chunk)
                sched, tile = self.next_tile(sched)
            work.advance()
        bulk_wait_group(0)
        return ckpt, sched, work

    @cute.jit
    def store_row(self, params, ckpt, head, row):
        with ckpt.consume() as h:
            h.access(read=True, by="tma")
            base = params.s_ckpt + h.index * (self.d_k * self.d_v * 2)
            for kb in cutlass.range_constexpr(self.d_k // 64):
                src = cute.make_tensor(
                    cute.make_ptr(
                        self.io_dtype,
                        base + kb * (self.d_v * 64 * 2),
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    params.box_ckpt,
                )
                dst = cute.local_tile(params.mCkpt[None, None, head, row], (self.d_v, 64), (0, kb))
                s, d = cpasync.tma_partition(
                    params.tma_ckpt,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(src, 0, 2),
                    cute.group_modes(dst, 0, 2),
                )
                cute.copy(params.tma_ckpt, s, d)
            bulk_commit_group()
            bulk_wait_group(0, read=True)
        return ckpt

    # ---- tcgen05 MMA (warp 13) ------------------------------------------------------------

    @role(warps=13, regs=56)
    @cute.jit
    def mma(
        self,
        params: KdaRecomputeParams,
        *,
        kk: Consumer,
        restore: Consumer,
        tinv: Consumer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
    ):
        elect = elect_sync()
        ksd, td, krd = self.ks_desc, self.t_desc, self.kr_desc
        t0 = tmem.base.toint()
        sin_col = t0 + tmem.cols("sin")
        ks_col = t0 + tmem.cols("ks")
        u_col = t0 + tmem.cols("u")
        yin_col = t0 + tmem.cols("yin")
        uin_col = t0 + tmem.cols("uin")
        state_col = t0 + tmem.cols("state")
        a_seg = ksd.sps_B * ksd.tmem_advance_A  # TMEM columns of one 64-wide K box
        b_seg = ksd.smem_subtile_B >> 4
        half_steps = ksd.num_k_steps // 2
        split = self.n_update < self.d_k
        stage_bytes = self.d_k * B_T * 2
        # Completed waits on the two state-operand halves, Y and the U operand.
        n_sin = Int32(0)
        n_y = Int32(0)
        n_uin = Int32(0)
        cum = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, _h, _ws, write_end, compute_start, _s, _len = self.item(params, tile)
                seeded = cutlass.Boolean(False)
                if cutlass.const_expr(self.use_initial_state):
                    seeded = compute_start == 0
                for local in cutlass.range(write_end - compute_start, unroll=1):
                    have_state = local > 0 or seeded
                    kh = kk.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    kh.access("Kdecay", read=True, by="umma")
                    desc_kd = desc_opaque(
                        smem_desc(params.s_kdecay + kh.index * stage_bytes, 16, 1024, SWIZZLE_128B)
                    )
                    if have_state:
                        # ---- KS = S K_decay^T: the left half of S lands first ----------------
                        with self.trace_region("sin_l.wait"):
                            spin_wait(tstate + SIN_L, n_sin & 1)
                        for f in cutlass.range_constexpr(ksd.num_k_steps):
                            if cutlass.const_expr(f == half_steps):
                                with self.trace_region("sin_r.wait"):
                                    spin_wait(tstate + SIN_R, n_sin & 1)
                            sb = f // ksd.sps_B
                            mma_ts_op(
                                ksd,
                                sin_col + sb * a_seg,
                                desc_kd + sb * b_seg,
                                ks_col,
                                f - sb * ksd.sps_B,
                                cutlass.Boolean(f > 0),
                                elect,
                            )
                        n_sin += 1
                        umma_commit(tstate + KS, elect)
                    kh.release(signal=False)  # the role's elected lane commits (below)
                    umma_commit(kh.empty_barrier, elect)
                    qh = restore.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    qh.access(read=True, by="umma")
                    # ---- U = T Y ----------------------------------------------------------------
                    th = tinv.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    th.access(read=True, by="umma")
                    desc_t = desc_opaque(
                        smem_desc(params.s_tinv + th.index * (B_T * B_T * 2), 16, 256, SWIZZLE_32B)
                    )
                    with self.trace_region("yin.wait"):
                        spin_wait(tstate + YIN, n_y & 1)
                    n_y += 1
                    mma_ts_op(td, yin_col, desc_t, u_col, 0, cutlass.Boolean(False), elect)
                    th.release(signal=False)
                    umma_commit(th.empty_barrier, elect)
                    umma_commit(tstate + U, elect)
                    # ---- S += K_restore^T U, left then right key half ---------------------------
                    desc_kr = desc_opaque(
                        smem_desc(
                            params.s_krestore + qh.index * stage_bytes, 2048, 1024, SWIZZLE_128B
                        )
                    )
                    with self.trace_region("uin.wait"):
                        spin_wait(tstate + UIN, n_uin & 1)
                    n_uin += 1
                    mma_ts_op(krd, uin_col, desc_kr, state_col, 0, have_state, elect)
                    umma_commit(tstate + SACC_L + (cum & 1), elect)
                    if cutlass.const_expr(split):
                        mma_ts_op(
                            krd,
                            uin_col,
                            desc_kr + ((B_T * 128) >> 4),
                            state_col + self.n_update,
                            0,
                            have_state,
                            elect,
                        )
                    qh.release(signal=False)
                    umma_commit(qh.empty_barrier, elect)
                    umma_commit(tstate + SACC_R + (cum & 1), elect)
                    cum += 1
                sched, tile = self.next_tile(sched)
            work.advance()
        return kk, restore, tinv, sched, work

    # ---- kernel and entry point ---------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        k_load,
        v_load,
        g_load,
        tma_ckpt,
        mCkpt,
        box_io,
        box_g,
        box_ckpt,
        beta,
        state_in,
        work_items,
        work_count,
        scheduler,
        k_ratio,
        v_ratio,
        layouts,
        trace_buffer,
        bound,
    ):
        self.begin_roles(work=bound, layouts=layouts)
        addr = {name: self.buffer(name).iterator.toint() for name in self.buf}
        params = KdaRecomputeParams(
            k_load=k_load,
            v_load=v_load,
            g_load=g_load,
            tma_ckpt=tma_ckpt,
            mCkpt=mCkpt,
            box_io=box_io,
            box_g=box_g,
            box_ckpt=box_ckpt,
            beta=beta,
            state_in=state_in,
            work_items=work_items,
            work_count=work_count,
            scheduler=scheduler,
            k_ratio=k_ratio,
            v_ratio=v_ratio,
            s_k=addr["K"],
            s_v=addr["V"],
            s_g=addr["Gate"],
            s_ex=addr["Exchange"] if "Exchange" in addr else addr["Gate"],
            s_kdecay=addr["Kdecay"],
            s_krestore=addr["Krestore"],
            s_kinv=addr["Kinv"],
            s_tinv=addr["Tinv"],
            s_ckpt=addr["Ckpt"],
            s_beta=addr["Beta"],
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def launch(
        self,
        k: cute.Tensor,  # (tokens, k heads, d_k)
        v: cute.Tensor,  # (tokens, v heads, d_v)
        gate: cute.Tensor,  # (tokens, heads, d_k) natural-log decay
        beta: cute.Tensor,  # (tokens, heads)
        state_in: cute.Tensor | None,  # (sequences, heads, d_v, d_k)
        checkpoints: cute.Tensor,  # (rows, heads, d_v, d_k)
        work_items: cute.Tensor,  # (rows, WORK_ITEM_FIELDS) int32
        work_count: cute.Tensor,  # (1,) int32 rows to run
        scheduler: cute.Tensor,  # (>= 1,) int32, zero
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        stream: cuda.CUstream,
        trace_buffer: cute.Tensor | None = None,
    ):
        """Launch on ``num_ctas`` CTAs (PDL); callable from another ``@cute.jit`` host such as
        a launch bundle, after a kernel that wrote ``work_items``/``work_count`` and zeroed
        ``scheduler``."""
        io, gdt = self.io_dtype, self.gate_dtype
        heads = Int32(gate.shape[1])
        k_ratio = heads // Int32(k.shape[1])
        v_ratio = heads // Int32(v.shape[1])
        box_io = cute.select(
            sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1), mode=[0, 1]
        )
        g_cols = 128 // (gdt.width // 8)
        box_g = cute.select(
            sm100_utils.make_smem_layout_epi(gdt, LayoutEnum.ROW_MAJOR, (B_T, g_cols), 1),
            mode=[0, 1],
        )
        box_ckpt = cute.select(
            sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (self.d_v, 64), 1),
            mode=[0, 1],
        )

        def tdh(t):  # (token, head, dim) -> (token, dim, head)
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        def ragged(op, t, box, cols, index):  # clipped at each sequence end, keyed by head
            slots = slot_region(tma_slots, index, B_T, heads)
            return make_ragged_tma(op, tdh(t), box, B_T, cols, slots)

        k_load = ragged(g2s, k, box_io, 64, 0)
        v_load = ragged(g2s, v, box_io, 64, 1)
        g_load = ragged(g2s, gate, box_g, g_cols, 2)
        ckpt_view = cute.make_tensor(
            checkpoints.iterator, cute.select(checkpoints.layout, mode=[2, 3, 1, 0])
        )
        tma_ckpt, mCkpt = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), ckpt_view, box_ckpt, (self.d_v, 64)
        )
        d_k, d_v = self.d_k, self.d_v
        layouts = {
            "Kdecay": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Krestore": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Kinv": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Tinv": cute.make_layout((B_T * B_T, INTER_STAGES)),
            "K": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "V": cute.make_layout((d_v * B_T, RAW_STAGES)),
            "Gate": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "Beta": cute.make_layout((B_T, RAW_STAGES)),
            "Ckpt": cute.make_layout((d_k * d_v, CKPT_STAGES)),
        }
        if cutlass.const_expr(self.gate != "fp32"):
            layouts["Exchange"] = cute.make_layout((d_k * B_T, self.ex_stages))
        bound = self.work.bind((self.num_ctas, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            k_load,
            v_load,
            g_load,
            tma_ckpt,
            mCkpt,
            box_io,
            box_g,
            box_ckpt,
            beta,
            state_in,
            work_items,
            work_count,
            scheduler,
            k_ratio,
            v_ratio,
            layouts,
            trace_buffer,
            bound,
        ).launch(**self.launch_config(bound), stream=stream, use_pdl=True)

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        state_in: cute.Tensor | None,
        checkpoints: cute.Tensor,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        scheduler: cute.Tensor,
        tma_slots: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.launch(
            k, v, gate, beta, state_in, checkpoints, work_items, work_count, scheduler, tma_slots,
            stream,
        )  # fmt: skip


# ---- standalone entry point (tests) ---------------------------------------------------------


def checkpoint_rows(tokens: int, num_sequences: int) -> int:
    """Rows the closed-form checkpoint map ``cu[b] // 16 + b + c`` can address."""
    return max(tokens // B_T + num_sequences, 1)


@jit_cache
def compile_kda_recompute(op: KdaRecompute, beta_name: str, state_name: str | None):
    """Compile for dynamic tokens, heads, sequences, work rows and checkpoint rows."""
    io, i32 = op.io_dtype, cutlass.Int32
    tokens, hk, hv, heads, rows, ckpt_rows = (cute.sym_int() for _ in range(6))

    def packed(dtype, h, d):
        return make_fake_strided_tensor(
            dtype, (tokens, h, d), stride_divisibility=tma_divisibility(dtype)
        )

    state = None
    if state_name is not None:
        state = make_fake_strided_tensor(
            CUTE_DTYPES[state_name],
            (cute.sym_int(), heads, op.d_v, op.d_k),
            stride_divisibility=4,
        )
    return compile_tvm_ffi(
        op,
        packed(io, hk, op.d_k),
        packed(io, hv, op.d_v),
        packed(op.gate_dtype, heads, op.d_k),
        make_fake_strided_tensor(CUTE_DTYPES[beta_name], (tokens, heads), contiguous_dim=None),
        state,
        make_fake_strided_tensor(io, (ckpt_rows, heads, op.d_v, op.d_k), stride_divisibility=8),
        make_fake_strided_tensor(i32, (rows, WORK_ITEM_FIELDS), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        fake_tma_slots(),
        name=op.get_name(),
    )


def kda_recompute(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    work_items: torch.Tensor,
    work_count: torch.Tensor,
    checkpoints: torch.Tensor,
    *,
    initial_state: torch.Tensor | None = None,
    scheduler: torch.Tensor | None = None,
) -> None:
    """Fill ``checkpoints [rows, H, d_v, d_k]`` for packed ``k [T, HK, d_k]``, ``v [T, HV, d_v]``,
    natural-log channel gate ``[T, H, d_k]``, beta ``[T, H]`` and an int32 work table (see the
    module docstring; ``kda_prefill.uncut_work_items`` builds the uncut one). ``scheduler``
    (int32, zeroed) is allocated when omitted."""
    if k.dtype not in (torch.float16, torch.bfloat16) or v.dtype != k.dtype:
        raise TypeError("k and v must share a 16-bit dtype")
    if checkpoints.dtype != k.dtype:
        raise TypeError("checkpoints must have the io dtype")
    if scheduler is None:
        scheduler = torch.zeros(1, dtype=torch.int32, device=k.device)
    op = KdaRecompute(
        TORCH_NAMES[k.dtype],
        TORCH_NAMES[gate.dtype],
        k.shape[-1],
        v.shape[-1],
        initial_state is not None,
        torch.cuda.get_device_properties(k.device).multi_processor_count,
    )
    state_name = None if initial_state is None else TORCH_NAMES[initial_state.dtype]
    fn = compile_kda_recompute(op, TORCH_NAMES[beta.dtype], state_name)
    slots = new_tma_slots(op.tma_slot_rows(gate.shape[1]), k.device)
    fn(k, v, gate, beta, initial_state, checkpoints, work_items, work_count, scheduler, slots)
