# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# kda_prefill_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and
# channels with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence
# descriptor arrays, the natural-log gate is the only gate form, and the state checkpoints the
# drivers never request are gone.

"""KDA chunked prefill forward (BT = 16, per-key-channel decay) for SM100.

Per chunk of 16 tokens (``g`` the in-chunk cumulative log2 gate per key channel, ``S`` the
``[d_k, d_v]`` fp32 state carried across chunks)::

    K_decay = K exp2(g)   K_inv = K exp2(-g)   K_restore = K exp2(g_last - g)   Q_decay = Q exp2(g)
    L   = beta * strict_lower(K_decay K_inv^T)        T = (I + L)^-1  (16x16 blockwise, mma.sync)
    A   = lower(Q_decay K_inv^T)                      Y = beta (V - K_decay S)
    U   = T Y                                         O = scale (Q_decay S + A U)
    S  <- exp2(g_last) S + K_restore^T U

The work table (one row per (sequence, head x value tile) item, see ``WORK_ITEM_FIELDS``) and
the zeroed scheduler counter come from the caller (``work_tables`` in the launch bundles,
``uncut_work_items`` standalone). CTA ``b`` starts at item ``b``; the load warp claims the next
item from the counter after issuing an item's loads and broadcasts it to every role.

Q/K/V/Gate rows past a sequence end load as zeros (``ragged.RaggedTma``), as from
upstream's per-sequence descriptors; beta is zero there and O rows past the end are clipped by
the ragged TMA store view.

Roles (16 warps) and their handoffs. Two-way SMEM/TMA handoffs are warpdrive Rings; ``rr``
marks a round-robin side of the two ``cg0`` instances, which take the CTA's cumulative chunks
alternately (``c % 2``)::

    load (14)     --raw: 8 x Q,K,V,Gate (TMA)-->      cg0 (both instances), cg1
                  --beta: 8 x f32[16]-->              regmma, cg1
                  --sched: 8 x next item-->           every other role
    cg0 (0-7) rr  --kk: K_decay, K_inv-->             mma, regmma, epilogue
                  --qk: Q_decay, K_restore-->         mma, epilogue
                  --exchange: exp2(g) rows-->         cg1 (signal-only; see __init__)
    cg1 (8-11)    --o_smem: 2 x O-->                  epilogue (15): TMA store
    regmma (12)   --tinv-->  mma (13)  <--a--  epilogue
    mma           --o_acc: 2 x O accumulator-->       cg1

The TMEM state recurrence between the MMA warp and the compute groups (fp16 state operand,
KS / U accumulators, Y / U operands, state-update completion) uses the upstream signal-only
barriers (``tstate``, ``TSTATE``): every stage they guard is reused only after a later handoff
of the same recurrence, so a channel's reuse wait would only lengthen the per-chunk critical
path. CG1 drives its ``raw`` and ``beta`` consumer ends with one shared chunk counter through
the end accessors (``raw_beta_wait``), as upstream shares one stage index between them.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    Barriers,
    Consumer,
    Instance,
    Local,
    NamedBarriers,
    OneShot,
    PdlWait,
    Producer,
    Ring,
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
from attn_gym.warpdrive.cache import (
    compile_tvm_ffi,
    jit_cache,
    make_fake_compact_tensor,
    make_fake_strided_tensor,
)
from attn_gym.warpdrive.compat import LayoutEnum

from .blockwise_inverse import invert_unit_lower_16x16_fragments
from .contract import tma_divisibility
from .kda_common import (
    B_T,
    CUTE_DTYPES,
    FINAL_DST,
    TORCH_NAMES,
    WORK_ITEM_FIELDS,
    as_i32,
    beta_residual,
    box128,
    lds_16_f32,
    lds_f32,
    lds_v2_f32,
    lds_v4,
    lds_v4_f32,
    load_rows,
    load_state_16,
    mask_pair,
    opaque_i32,
    raw_beta_release,
    raw_beta_wait,
    resolve_paged_state,
    spin_wait,
    stg_v4,
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
    ragged_store,
    sequence_descriptors,
    slot_region,
    tma_slot_rows,
)
from .umma import (
    SWIZZLE_32B,
    SWIZZLE_128B,
    MmaDesc,
    desc_opaque,
    instr_desc,
    mma_ts_op,
    smem_desc,
)

DIMS = (64, 128)
LOG2_E = 1.4426950408889634
RAW_STAGES = 8
SCHED_STAGES = 8
DECAY_STAGES = 2
INTER_STAGES = 2
O_STAGES = 2
# CG1 releases chunk c-1's O accumulator only after staging Y(c), while the MMA warp claims
# chunk c's slot before KS/QS/U(c): with one slot that would cycle, so keep two or more.
QS_STAGES = 2
assert QS_STAGES >= 2, "CG1's late o_acc release needs two O accumulator slots"

# TMEM state handoffs (``tstate`` barriers, upstream counts), each completing once per epoch:
#   SIN_L  left key half of the fp16 state operand: CG0 instance (chunks after an item's first),
#          CG1 (the seed); MMA waits before KS k-steps [0, n/2). Reuse: rewritten after SACC_L.
#   SIN_R  right half, CG1; MMA waits before the rest of KS.
#   KS, U  KS / U accumulators, MMA commit; CG1 reads. KS(c+1) / U(c+1) follow YIN / UIN waits
#          that CG1 signals after reading them.
#   YIN    Y operand, CG1; MMA waits before U. Rewritten after U completes.
#   UIN    U operand (CG1) and the left-half state decay (CG0 instance): 4 + 4 arrivals; MMA
#          waits before the state update.
#   SACC_L / SACC_R (2 stages)  left / right state update committed; CG0 / CG1 read the state.
# The TMEM loads need no tcgen05.wait::ld where the same warp issues a tcgen05.st after them
# and completes it (wait::st) before the signal that lets the MMA overwrite their columns: a
# warp's ld -> st pair is pipelined (PTX ISA 9.7.18.6.2), while a register dependency orders no
# memory (9.7.18.6.4.5). Each load names its store; drain_o and final_state have none and wait
# for their loads (final_state's can be the role's last TMEM access before teardown frees TMEM).
SIN_L, SIN_R, KS, U, YIN, UIN, SACC_L, SACC_R = 0, 1, 2, 3, 4, 5, 6, 8
TSTATE_COUNTS = (4, 4, 1, 1, 4, 8, 1, 1, 1, 1)


@dataclass(frozen=True)
class KdaPrefillParams:
    q_load: RaggedTma  # (token, dim, head) loads, zero past each sequence end
    k_load: RaggedTma
    v_load: RaggedTma
    g_load: RaggedTma
    o_store: RaggedTma  # (token, d_v, head) stores, dropping rows past each sequence end
    box_io: cute.ComposedLayout  # one 16 x 64 16-bit swizzled box
    box_g: cute.ComposedLayout  # one 16 x 128-byte gate box
    beta: cute.Tensor  # (token, head)
    state_in: cute.Tensor | None  # (row, head, d_v, d_k)
    state_out: cute.Tensor | None
    state_indices: cute.Tensor | None
    has_initial_state: cute.Tensor | None
    work_items: cute.Tensor  # (rows, WORK_ITEM_FIELDS)
    work_count: cute.Tensor  # (1,)
    scheduler: cute.Tensor  # (>= 1,) zeroed ticket counter
    q_ratio: Int32  # tile heads per q / k / v head
    k_ratio: Int32
    v_ratio: Int32
    scale: Float32
    s_q: Int32  # SMEM byte addresses of the buffers (stage 0)
    s_k: Int32
    s_v: Int32
    s_g: Int32
    s_ex: Int32
    s_kdecay: Int32
    s_qdecay: Int32
    s_krestore: Int32
    s_kinv: Int32
    s_a: Int32
    s_tinv: Int32
    s_o: Int32
    s_beta: Int32


class KdaPrefill(WarpSpecialized):
    """Warpdrive KDA prefill for ``io`` fp16/bf16 operands and an fp32 or ``io`` gate.

    ``d_v`` is the tile width (64 under the d_v split, ``tiles_per_head = 2``). The static state
    flags and ``paged_state`` (0 none, 1 routes, 2 routes + fresh-slot mask) select code; the
    state dtype follows the traced tensors."""

    def __init__(
        self,
        io: str,
        gate: str,
        d_k: int,
        d_v: int,
        tiles_per_head: int,
        use_initial_state: bool,
        store_final_state: bool,
        paged_state: int,
        num_ctas: int,
    ):
        if d_k not in DIMS or d_v not in DIMS:
            raise ValueError(f"the KDA prefill serves d_k, d_v in {DIMS}, got {d_k}, {d_v}")
        if gate not in ("fp32", io):
            raise ValueError(f"the gate must be fp32 or the io dtype, got {gate}")
        if paged_state and not (use_initial_state and store_final_state):
            raise ValueError("paged state routes one pool in and out")
        self.io, self.gate, self.d_k, self.d_v = io, gate, d_k, d_v
        self.tiles_per_head, self.num_ctas = tiles_per_head, num_ctas
        self.use_initial_state, self.store_final_state = use_initial_state, store_final_state
        self.paged_state = paged_state
        self.io_dtype = CUTE_DTYPES[io]
        self.gate_dtype = CUTE_DTYPES[gate]
        gate_bytes = self.gate_dtype.width // 8
        # An fp32 gate is exchanged in place (exp2 prefixes over the raw gate stage); a 16-bit
        # gate exchanges through its own 4-stage fp32 buffer.
        self.ex_stages = RAW_STAGES if gate == "fp32" else 4
        io_t, f32 = self.io_dtype, Float32
        plan = self.smem_plan = SmemPlan()
        buf = {
            "Kdecay": plan.buffer("Kdecay", io_t, align=1024),
            "Qdecay": plan.buffer("Qdecay", io_t, align=1024),
            "Krestore": plan.buffer("Krestore", io_t, align=1024),
            "A": plan.buffer("A", io_t, align=1024),
            "Tinv": plan.buffer("Tinv", io_t, align=1024),
            "Q": plan.buffer("Q", io_t, align=1024),
            "K": plan.buffer("K", io_t, align=1024),
            "V": plan.buffer("V", io_t, align=1024),
            "Gate": plan.buffer("Gate", self.gate_dtype, align=1024),
        }
        if gate != "fp32":
            buf["Exchange"] = plan.buffer("Exchange", f32, align=1024)
        buf["Kinv"] = plan.buffer("Kinv", io_t, align=1024)
        buf["O"] = plan.buffer("O", io_t, align=1024)
        buf["Beta"] = plan.buffer("Beta", f32, align=16)
        self.buf = buf
        rr = Sharing.ROUND_ROBIN
        raw_bytes = B_T * (
            2 * d_k * (io_t.width // 8) + d_v * (io_t.width // 8) + d_k * gate_bytes
        )
        # The TMA check reports raw unverified: it cannot follow the per-item domain_offset views.
        self.raw = Ring(
            RAW_STAGES,
            Tma(raw_bytes),
            {"cg0": Warps, "cg1": Warps},
            buffer={k: buf[k] for k in ("Q", "K", "V", "Gate")},
        )
        # Signal-only (the upstream gate_exchange_ready): CG0 instances arrive raw, CG1 waits. An
        # fp32 gate exchanges inside the raw gate stage, which the raw ring recycles; a 16-bit
        # gate's own stage is rewritten 4 chunks later, after CG1 read it (CG0 has by then
        # waited for the state update of 3 chunks later, whose U operand CG1 published after).
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
            {"mma": Umma, "regmma": Warps, "epilogue": Warps},
            producer_sharing=rr,
            commit_fence=smem_commit,
            buffer={"Kdecay": buf["Kdecay"], "Kinv": buf["Kinv"]},
        )
        self.qk = Ring(
            DECAY_STAGES,
            Warps,
            {"mma": Umma, "epilogue": Warps},
            producer_sharing=rr,
            commit_fence=smem_commit,
            buffer={"Qdecay": buf["Qdecay"], "Krestore": buf["Krestore"]},
        )
        # The MMA warp spins on the single-warp T_inv / A handoffs, as upstream does: a suspended
        # T_inv wait delayed U on the per-chunk recurrence (0.9% at d128). Spinning o_acc too
        # (CG1's four warps) cost more issue slots than it saved.
        self.tinv = Ring(
            INTER_STAGES, Warps, Umma, commit_fence=smem_commit, buffer=buf["Tinv"], wait="spin"
        )
        self.a = Ring(
            INTER_STAGES, Warps, Umma, commit_fence=smem_commit, buffer=buf["A"], wait="spin"
        )
        self.o_smem = Ring(O_STAGES, Warps, Warps, commit_fence=smem_commit, buffer=buf["O"])
        self.o_acc = Ring(QS_STAGES, Umma, Warps)
        # The TMEM state recurrence (upstream protocol): signal-only barriers whose stage reuse
        # is implied by the recurrence itself (see ``TSTATE``), so no channel reuse waits sit
        # on the per-chunk critical path.
        self.tstate = Barriers(len(TSTATE_COUNTS), counts=list(TSTATE_COUNTS))
        # Each CG0 instance syncs its four warps between its exchange stores and their reads.
        self.exchange_sync = NamedBarriers(2, threads=128)
        self.tmem_plan = (
            TmemPlan()
            .region("state", d_k)
            .region("sin", d_k // 2)
            .region("qs", B_T, QS_STAGES)
            .region("ks", B_T)
            .region("u", B_T)
            .region("yin", B_T // 2)
            .region("uin", B_T // 2)
        )
        self.work = WorkStream(schedule=OneShot(grid="1d"), ready=PdlWait())
        # MMA geometry: KS/QS (TMEM state operand x K/Q decay), U and O (TMEM Y/U x the 16 x 16
        # T_inv/A tiles), the state update (TMEM U x MN-major K_restore), split in two key halves
        # when d_k = 128.
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
        """Rows of the TMA slot buffer for ``heads`` gate heads: Q, K, V, gate and O, keyed by
        tile head."""
        return tma_slot_rows(5, B_T, heads * self.tiles_per_head)

    def get_name(self) -> str:
        flags = f"{int(self.use_initial_state)}{int(self.store_final_state)}{self.paged_state}"
        return (
            f"wd_kda_prefill_{self.io}_g{self.gate}_k{self.d_k}_v{self.d_v}"
            f"_t{self.tiles_per_head}_f{flags}"
        )

    # ---- work decode -------------------------------------------------------------------

    @cute.jit
    def item(self, params: KdaPrefillParams, tile):
        """``(batch, head_o, v_offset, write_start, write_end, compute_start, batch_start,
        seqlen, tile head)`` of work row ``tile``."""
        row = params.work_items
        head = row[tile, 1]
        head_o = head
        v_offset = Int32(0)
        if cutlass.const_expr(self.tiles_per_head > 1):
            head_o = head // self.tiles_per_head
            v_offset = (head - head_o * self.tiles_per_head) * self.d_v
        start = row[tile, 6]
        return (
            row[tile, 0],
            head_o,
            v_offset,
            row[tile, 2],
            row[tile, 3],
            row[tile, 4],
            start,
            row[tile, 7] - start,
            head,
        )

    @cute.jit
    def next_tile(self, sched):
        with sched.consume() as h:
            tile = h.data[0]
        return sched, tile

    # ---- load (warp 14): beta one chunk ahead, Q/K/Gate/V TMA, next item -----------------

    @role(warps=14, regs=56)
    @cute.jit
    def load(
        self,
        params: KdaPrefillParams,
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
        """Beta (one chunk ahead) and the Q, K, Gate and V boxes of every chunk of one item."""
        _b, head_o, v_offset, _ws, write_end, compute_start, start, seqlen, head = fields
        head_q = head // params.q_ratio
        head_k = head // params.k_ratio
        head_v = head // params.v_ratio
        loads = (params.q_load, params.k_load, params.g_load, params.v_load)
        moves_rows = write_end > compute_start
        descs = sequence_descriptors(loads, start + seqlen, head, B_T, moves_rows)
        carry = Float32(0.0)
        token = compute_start * B_T + lane
        if lane < B_T and token < seqlen:
            carry = params.beta[start + token, head_o].to(Float32)
        for chunk in cutlass.range(compute_start, write_end, 1, unroll=1):
            with beta.produce() as bh:
                bh.access(write=True, by="threads")
                if lane < B_T:
                    sts_f32(params.s_beta + bh.index * (B_T * 4) + 4 * lane, carry)
            carry = Float32(0.0)
            token = (chunk + 1) * B_T + lane
            if lane < B_T and token < seqlen:
                carry = params.beta[start + token, head_o].to(Float32)
            rows = (start + chunk * B_T, cutlass.min(seqlen - chunk * B_T, B_T))
            raw = self.load_chunk(
                params, raw, rows, head_q, head_k, head_v, head_o, v_offset, descs
            )
        return raw, beta

    @cute.jit
    def load_chunk(self, params, raw, rows, head_q, head_k, head_v, head_o, v_offset, descs):
        """TMA one chunk's Q, K, Gate and V boxes (tokens ``rows = (token, valid)``, zero past
        them) into one ``raw`` stage through the item's ``descs``."""
        q_desc, k_desc, g_desc, v_desc = descs
        io, io_b = self.io_dtype, self.io_dtype.width // 8
        g_cols = 128 // (self.gate_dtype.width // 8)
        box = B_T * 128  # bytes of one 16-row 128-byte box
        with raw.produce() as h:
            h.access("Q", write=True, by="tma")
            h.access("K", write=True, by="tma")
            h.access("V", write=True, by="tma")
            h.access("Gate", write=True, by="tma")
            qk_stage = h.index * (self.d_k * B_T * io_b)
            bar = h.barrier
            for kb in cutlass.range_constexpr(self.d_k // 64):
                off = qk_stage + kb * box
                load_rows(
                    params.q_load,
                    io,
                    params.s_q + off,
                    params.box_io,
                    rows,
                    head_q,
                    kb,
                    64,
                    bar,
                    q_desc,
                )
                load_rows(
                    params.k_load,
                    io,
                    params.s_k + off,
                    params.box_io,
                    rows,
                    head_k,
                    kb,
                    64,
                    bar,
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
                    head_o,
                    kb,
                    g_cols,
                    bar,
                    g_desc,
                )
            v_stage = h.index * (self.d_v * B_T * io_b)
            for kb in cutlass.range_constexpr(self.d_v // 64):
                load_rows(
                    params.v_load,
                    io,
                    params.s_v + v_stage + kb * box,
                    params.box_io,
                    rows,
                    head_v,
                    v_offset // 64 + kb,
                    64,
                    bar,
                    v_desc,
                )
        return raw

    # ---- compute group 0 (warps 0-7): gate scan, decay operands, left state half --------

    @role(warps=[range(4), range(4, 8)], regs=160, shared_body=True)
    @cute.jit
    def cg0(
        self,
        params: KdaPrefillParams,
        *,
        stage: Instance,
        raw: Consumer,
        kk: Producer,
        qk: Producer,
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
        pack_col = opaque_i32(2 * (lane % 4))
        pack_col_x = opaque_i32(2 * (lane % 4) ^ 4)
        decay_row = warp * 4 + lane // 8
        lane_in_row = lane - (lane // 8) * 8
        lane_base = tmem.base.toint() + (
            (32 * (warp % 4)) << 16
        )  # warp-uniform: this subpartition
        state_col = lane_base + tmem.cols("state")
        sin_col = lane_base + tmem.cols("sin")
        barrier = self.exchange_sync[stage]
        raw_next = Int32(0)  # next raw epoch this instance takes (both instances join raw)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            cum_base = Int32(0)
            while tile < total:
                _b, _h, _v, _ws, write_end, compute_start, _s, seqlen, _t = self.item(params, tile)
                n_local = write_end - compute_start
                for local in cutlass.range((stage + cum_base) & 1, n_local, 2, unroll=1):
                    c = cum_base + local
                    if raw_next < c:  # the other instance's chunk: release its stage
                        raw.consume_unread()
                    raw_next = c + 1
                    raw, kk, qk = self.cg0_chunk(
                        params,
                        raw,
                        kk,
                        qk,
                        tstate,
                        c,
                        barrier,
                        (compute_start + local) * B_T,
                        seqlen,
                        local,
                        lane,
                        decay_row,
                        lane_in_row,
                        prefix_offsets,
                        gate_offsets,
                        seg_base,
                        segment,
                        col,
                        pack_col,
                        pack_col_x,
                        state_col,
                        sin_col,
                    )
                cum_base += n_local
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, kk, qk, sched, work

    @cute.jit
    def cg0_chunk(
        self,
        params,
        raw,
        kk,
        qk,
        tstate,
        c,
        barrier,
        chunk_start,
        seqlen,
        local,
        lane,
        decay_row,
        lane_in_row,
        prefix_offsets,
        gate_offsets,
        seg_base,
        segment,
        col,
        pack_col,
        pack_col_x,
        state_col,
        sin_col,
    ):
        io = self.io_dtype
        d_k = self.d_k
        dk_halves = d_k // 64
        channel_rows = d_k // 4
        store_rows = B_T * channel_rows // 32
        store_row_base = (lane // channel_rows) * store_rows
        rh = raw.wait()
        rh.access("Q", read=True, by="threads")
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
        if cutlass.const_expr(self.gate == "fp32"):
            s_ex = s_gate  # in place over this raw gate stage
        else:
            s_ex = params.s_ex + (c & (self.ex_stages - 1)) * (d_k * B_T * 4)
        for row_off in cutlass.range_constexpr(store_rows):
            if cutlass.const_expr(channel_rows < 32):
                row = store_row_base + row_off
                value = g[row_off + B_T // 2] if lane >= channel_rows else g[row_off]
                idx = seg_base + swz128(row ^ segment, row * 32 + col, 4)
            else:
                value = g[row_off]
                idx = prefix_offsets[row_off % 8] + row_off * 32
            sts_f32(s_ex + 4 * idx, value)
        barrier.sync(aligned=True)
        warp_arrive(self.barrier_ptr("exchange") + (c & (self.ex_stages - 1)))

        # ---- raw Q / K rows (zeroed past the sequence end), exp2(g) of the row ---------------
        qk_stage = rh.index * (d_k * B_T * 2)
        q_words = []
        k_words = []
        row_valid = chunk_start + decay_row < seqlen
        for half in cutlass.range_constexpr(dk_halves):
            idx = box128(decay_row, half * 64 + lane_in_row * 8)
            qw = lds_v4(params.s_q + qk_stage + 2 * idx)
            kw = lds_v4(params.s_k + qk_stage + 2 * idx)
            if ragged:
                for i in cutlass.range_constexpr(4):
                    qw[i] = qw[i] if row_valid else Int32(0)
                    kw[i] = kw[i] if row_valid else Int32(0)
            q_words += qw
            k_words += kw
        raw_q = []
        raw_k = []
        for i in cutlass.range_constexpr(dk_halves * 4):
            lo, hi = f16x2_to_f32(q_words[i], dtype=io)
            raw_q += [lo, hi]
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
        kh = kk.acquire()
        kh.access("Kdecay", write=True, by="threads")
        kh.access("Kinv", write=True, by="threads")
        decay_stage = kh.index * (d_k * B_T * 2)
        for half in cutlass.range_constexpr(dk_halves):
            off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
            st_shared_v4(params.s_kinv + off, ki_pack[4 * half : 4 * half + 4])
            st_shared_v4(params.s_kdecay + off, kd_pack[4 * half : 4 * half + 4])
        kh.commit()

        # ---- Q decay, K restore --------------------------------------------------------------
        qd_pack = []
        for i in cutlass.range_constexpr(dk_halves * 4):
            qd0, qd1 = fmul2(raw_q[2 * i], raw_q[2 * i + 1], exp_g[2 * i], exp_g[2 * i + 1])
            qd_pack.append(fp32_to_fp16(qd0, qd1, dtype=io))
        qh = qk.acquire()
        qh.access("Qdecay", write=True, by="threads")
        qh.access("Krestore", write=True, by="threads")
        decay_stage = qh.index * (d_k * B_T * 2)
        for half in cutlass.range_constexpr(dk_halves):
            off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
            st_shared_v4(params.s_qdecay + off, qd_pack[4 * half : 4 * half + 4])
        for half in cutlass.range_constexpr(dk_halves):
            off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
            st_shared_v4(params.s_krestore + off, kr_pack[4 * half : 4 * half + 4])
        qh.commit()

        # ---- left key half of the state: fp16 operand, then decay by exp2(g_last) -----------
        if c > 0:  # the previous chunk's left update has landed
            with self.trace_region("sacc_l.wait"):
                spin_wait(tstate + SACC_L + ((c - 1) & 1), ((c - 1) >> 1) & 1)
        if local > 0:
            state = cute.make_rmem_tensor((d_k // 2,), Float32)
            # Ordered before the state update (UIN) by the SIN store below and its wait::st.
            if cutlass.const_expr(self.d_v == 128):
                blocks = [tmem_ld("32x32b", 32, state_col + b * 32) for b in range(dk_halves)]
                for b in cutlass.range_constexpr(dk_halves):
                    for i in cutlass.range_constexpr(32):
                        state[b * 32 + i] = blocks[b][i]
                for b in cutlass.range_constexpr(dk_halves):
                    words = [
                        fp32_to_fp16(blocks[b][2 * i], blocks[b][2 * i + 1], dtype=io)
                        for i in range(16)
                    ]
                    tmem_st("32x32b", sin_col + b * 16, words)
            else:
                vals = tmem_ld("16x256b", d_k // 16, state_col)
                for i in cutlass.range_constexpr(d_k // 4):
                    state[i] = vals[i]
                words = []
                for m in cutlass.range_constexpr(d_k // 16):
                    words.append(fp32_to_fp16(vals[4 * m], vals[4 * m + 1], dtype=io))
                    words.append(fp32_to_fp16(vals[4 * m + 2], vals[4 * m + 3], dtype=io))
                tmem_st("16x128b", sin_col, words)
            cute.arch.fence_view_async_tmem_store()
            warp_arrive(tstate + SIN_L)
            if cutlass.const_expr(self.d_v == 128):
                for b in cutlass.range_constexpr(dk_halves):
                    scaled = []
                    for grp in cutlass.range_constexpr(8):
                        dim = b * 32 + grp * 4
                        seg = dim // 32
                        idx = (
                            seg * (B_T * 32)
                            + (B_T - 1) * 32
                            + swz128((B_T - 1) ^ seg, dim - seg * 32, 4)
                        )
                        f = lds_v4_f32(s_ex + 4 * idx)
                        for t in cutlass.range_constexpr(2):
                            s0, s1 = fmul2(
                                state[b * 32 + grp * 4 + 2 * t],
                                state[b * 32 + grp * 4 + 2 * t + 1],
                                f[2 * t],
                                f[2 * t + 1],
                            )
                            scaled += [s0, s1]
                    tmem_st("32x32b", state_col + b * 32, scaled)
            else:
                scaled = []
                for m in cutlass.range_constexpr(d_k // 16):
                    dim = 8 * m
                    seg = dim // 32
                    sxor = (7 ^ (seg & 7)) * 4
                    base = seg * (B_T * 32) + (B_T - 1) * 32 + ((dim % 32) ^ (sxor & 0x18))
                    f = lds_v2_f32(
                        s_ex
                        + 4 * (base + (pack_col_x if cutlass.const_expr(sxor & 4) else pack_col))
                    )
                    s0, s1 = fmul2(state[4 * m], state[4 * m + 1], f[0], f[1])
                    s2, s3 = fmul2(state[4 * m + 2], state[4 * m + 3], f[0], f[1])
                    scaled += [s0, s1, s2, s3]
                tmem_st("16x256b", state_col, scaled)
            cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + UIN)  # the left half is decayed (with CG1's U operand)
        rh.release()
        return raw, kk, qk

    # ---- compute group 1 (warps 8-11): seed, right state half, O drain, Y / U staging ----

    @role(warps=range(8, 12), regs=136)
    @cute.jit
    def cg1(
        self,
        params: KdaPrefillParams,
        *,
        raw: Consumer,
        beta: Consumer,
        exchange: Consumer,
        o_acc: Consumer,
        o_smem: Producer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        ov_row = (lane // 16) * 8 + (lane & 7)
        ov_col = ((lane // 8) & 1) * 8
        if cutlass.const_expr(self.d_v == 128):
            value_dim = warp * 32 + lane
            value_base = warp * 32
            row_valid = cutlass.Boolean(True)
        else:
            value_dim = warp * 16 + lane % 16
            value_base = warp * 16
            row_valid = lane < 16
        lane_base = tmem.base.toint() + (
            (32 * (warp % 4)) << 16
        )  # warp-uniform: this subpartition
        cols = {name: tmem.cols(name) for name in ("state", "sin", "qs", "ks", "u", "yin", "uin")}
        ov_lo = box128(ov_row, value_base + ov_col)
        ov_hi = box128(ov_row, value_base + 16 + ov_col)
        pack_col = opaque_i32(2 * (lane % 4))
        pack_col_x = opaque_i32(2 * (lane % 4) ^ 4)
        geo = (lane, lane_base, cols, ov_lo, ov_hi, pack_col, pack_col_x, value_dim, row_valid)
        # Completed waits on the KS and U accumulators and on the right-half state updates.
        n_ks = Int32(0)
        n_u = Int32(0)
        n_upd = Int32(0)
        n_raw = Int32(0)  # chunks consumed from raw and beta (see raw_beta_wait)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                batch, head_o, v_offset, _ws, write_end, compute_start, _s, seqlen, _t = self.item(
                    params, tile
                )
                n_local = write_end - compute_start
                slot, active, load_initial, clear_empty = resolve_paged_state(
                    batch, params.state_indices, params.has_initial_state
                )
                seed = cutlass.Boolean(False)
                if cutlass.const_expr(self.use_initial_state):
                    seed = compute_start == 0
                seed_row = batch
                if cutlass.const_expr(self.paged_state > 0):
                    seed_row = slot
                if n_local > 0:
                    exchange, n_ks, n_u = self.cg1_first(
                        params,
                        raw,
                        beta,
                        exchange,
                        tstate,
                        n_raw,
                        n_ks,
                        n_u,
                        geo,
                        seed,
                        seed_row,
                        head_o,
                        v_offset,
                        load_initial,
                        compute_start * B_T,
                        seqlen,
                    )
                for local in cutlass.range(1, n_local, 1, unroll=1):
                    exchange, o_acc, o_smem, n_ks, n_u, n_upd = self.cg1_chunk(
                        params,
                        raw,
                        beta,
                        exchange,
                        o_acc,
                        o_smem,
                        tstate,
                        n_raw + local,
                        n_ks,
                        n_u,
                        n_upd,
                        geo,
                        active,
                        (compute_start + local) * B_T,
                        seqlen,
                    )
                n_raw += n_local
                if n_local > 0:
                    with self.trace_region("sacc_r.wait"):
                        spin_wait(
                            tstate + SACC_R + (n_upd & 1), (n_upd >> 1) & 1
                        )  # the last update
                    n_upd += 1
                    o_acc, o_smem = self.drain_o(params, o_acc, o_smem, geo, active)
                final_dst = params.work_items[tile, FINAL_DST]
                if cutlass.const_expr(self.store_final_state):
                    self.final_state(
                        params,
                        geo,
                        batch,
                        head_o,
                        v_offset,
                        seqlen,
                        final_dst,
                        slot,
                        active,
                        clear_empty,
                    )
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, beta, exchange, o_acc, o_smem, sched, work

    @cute.jit
    def cg1_first(
        self,
        params,
        raw,
        beta,
        exchange,
        tstate,
        n_raw,
        n_ks,
        n_u,
        geo,
        seed,
        seed_row,
        head_o,
        v_offset,
        load_initial,
        chunk_start,
        seqlen,
    ):
        """An item's first chunk: optional seed of both state halves, Y (carrying the seeded
        state), U."""
        lane, lane_base, cols, _lo, _hi, _pc, _pcx, _vd, _rv = geo
        io = self.io_dtype
        eh = exchange.wait()
        s_ex = params.s_g + eh.index * (self.d_k * B_T * 4)
        if cutlass.const_expr(self.gate != "fp32"):
            s_ex = params.s_ex + eh.index * (self.d_k * B_T * 4)
        if cutlass.const_expr(self.use_initial_state):  # noqa: SIM102 (static, then dynamic)
            if seed:
                self.seed_state(
                    params, geo, tstate, seed_row, head_o, v_offset, load_initial, s_ex
                )
        stage = raw_beta_wait(raw, beta, n_raw, RAW_STAGES)
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
            # Ordered before KS(c+1) by stage_y's Y store and its wait::st.
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
        n_u = self.stage_u(tstate, n_u, geo)
        raw_beta_release(raw, beta, stage)
        eh.release(signal=False)  # signal-only channel (see __init__)
        return exchange, n_ks, n_u

    @cute.jit
    def seed_state(self, params, geo, tstate, seed_row, head_o, v_offset, load_initial, s_ex):
        """Initial state row -> fp16 state operand (left half, then right, each published as
        soon as it is stored) and fp32 state decayed by the first chunk's exp2(g_last)."""
        _l, lane_base, cols, _lo, _hi, _pc, _pcx, value_dim, _rv = geo
        io = self.io_dtype
        st = params.state_in
        addr = (st.iterator + st.layout((seed_row, head_o, value_dim + v_offset, 0))).toint()
        sbytes = st.element_type.width // 8
        for half in cutlass.range_constexpr(2):
            blocks = range(half * (self.d_k // 32), (half + 1) * (self.d_k // 32))
            vecs = []
            for i in cutlass.range_constexpr(blocks.start, blocks.stop):
                vals = load_state_16(addr + i * 16 * sbytes, st.element_type)
                if cutlass.const_expr(self.paged_state > 0):
                    vals = [Float32(cutlass.select_(load_initial, v, Float32(0.0))) for v in vals]
                vecs.append(vals)
            for j, i in enumerate(blocks):
                words = [
                    fp32_to_fp16(vecs[j][2 * p], vecs[j][2 * p + 1], dtype=io) for p in range(8)
                ]
                tmem_st("32x32b", lane_base + cols["sin"] + i * 8, words)
            cute.arch.fence_view_async_tmem_store()
            warp_arrive(tstate + (SIN_L if half == 0 else SIN_R))
            for j, i in enumerate(blocks):
                scaled = []
                for grp in cutlass.range_constexpr(4):
                    dim = i * 16 + grp * 4
                    seg = dim // 32
                    idx = (
                        seg * (B_T * 32)
                        + (B_T - 1) * 32
                        + swz128((B_T - 1) ^ seg, dim - seg * 32, 4)
                    )
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
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_v(self, params, stage, geo, chunk_start, seqlen):
        """This thread's transposed V pairs (``ldmatrix.trans``), zeroed past the sequence end."""
        lane, _lb, _c, ov_lo, ov_hi, _pc, _pcx, _vd, _rv = geo
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
        _l, lane_base, cols, _lo, _hi, _pc, _pcx, _vd, _rv = geo
        tmem_st("16x128b", lane_base + cols["yin"], y_lo)
        if cutlass.const_expr(self.d_v == 128):
            tmem_st("16x128b", lane_base + (16 << 16) + cols["yin"], y_hi)
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + YIN)

    @cute.jit
    def stage_u(self, tstate, n_u, geo):
        """Wait for U = T Y and publish its fp16 copy as the next MMA operand."""
        _l, lane_base, cols, _lo, _hi, _pc, _pcx, _vd, _rv = geo
        with self.trace_region("u.wait"):
            spin_wait(tstate + U, n_u & 1)
        # Ordered before U(c+1) by the UIN store below and its wait::st.
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
        o_acc,
        o_smem,
        tstate,
        n_raw,
        n_ks,
        n_u,
        n_upd,
        geo,
        active,
        chunk_start,
        seqlen,
    ):
        """A chunk after the first: right state half, the previous chunk's O, Y, U."""
        lane, lane_base, cols, ov_lo, ov_hi, pack_col, pack_col_x, _vd, _rv = geo
        io = self.io_dtype
        d_k = self.d_k
        half_blocks = d_k // 32
        stage = raw_beta_wait(raw, beta, n_raw, RAW_STAGES)
        with self.trace_region("sacc_r.wait"):
            spin_wait(tstate + SACC_R + (n_upd & 1), (n_upd >> 1) & 1)
        n_upd += 1
        state = cute.make_rmem_tensor((d_k // 2,), Float32)
        # Ordered before the state update (UIN) by the SIN store below and its wait::st.
        if cutlass.const_expr(self.d_v == 128):
            for j in cutlass.range_constexpr(half_blocks):
                vals = tmem_ld("32x32b", 16, lane_base + cols["state"] + (half_blocks + j) * 16)
                for i in cutlass.range_constexpr(16):
                    state[j * 16 + i] = vals[i]
        else:
            vals = tmem_ld("16x256b", d_k // 16, lane_base + cols["state"] + d_k // 2)
            for i in cutlass.range_constexpr(d_k // 4):
                state[i] = vals[i]
        oh = o_smem.acquire()
        if cutlass.const_expr(self.d_v == 128):
            for j in cutlass.range_constexpr(half_blocks):
                words = [
                    fp32_to_fp16(state[j * 16 + 2 * p], state[j * 16 + 2 * p + 1], dtype=io)
                    for p in range(8)
                ]
                tmem_st("32x32b", lane_base + cols["sin"] + (half_blocks + j) * 8, words)
        else:
            words = []
            for m in cutlass.range_constexpr(d_k // 16):
                words.append(fp32_to_fp16(state[4 * m], state[4 * m + 1], dtype=io))
                words.append(fp32_to_fp16(state[4 * m + 2], state[4 * m + 3], dtype=io))
            tmem_st("16x128b", lane_base + cols["sin"] + d_k // 4, words)
        eh = exchange.wait()
        s_ex = params.s_g + eh.index * (d_k * B_T * 4)
        if cutlass.const_expr(self.gate != "fp32"):
            s_ex = params.s_ex + eh.index * (d_k * B_T * 4)
        scaled = cute.make_rmem_tensor((d_k // 2,), Float32)
        if cutlass.const_expr(self.d_v == 128):
            for j in cutlass.range_constexpr(half_blocks):
                for grp in cutlass.range_constexpr(4):
                    dim = (half_blocks + j) * 16 + grp * 4
                    seg = dim // 32
                    idx = (
                        seg * (B_T * 32)
                        + (B_T - 1) * 32
                        + swz128((B_T - 1) ^ seg, dim - seg * 32, 4)
                    )
                    f = lds_v4_f32(s_ex + 4 * idx)
                    for t in cutlass.range_constexpr(2):
                        k = j * 16 + grp * 4 + 2 * t
                        scaled[k], scaled[k + 1] = fmul2(
                            state[k], state[k + 1], f[2 * t], f[2 * t + 1]
                        )
        else:
            for m in cutlass.range_constexpr(d_k // 16):
                dim = d_k // 2 + 8 * m
                seg = dim // 32
                sxor = (7 ^ (seg & 7)) * 4
                base = seg * (B_T * 32) + (B_T - 1) * 32 + ((dim % 32) ^ (sxor & 0x18))
                f = lds_v2_f32(
                    s_ex + 4 * (base + (pack_col_x if cutlass.const_expr(sxor & 4) else pack_col))
                )
                scaled[4 * m], scaled[4 * m + 1] = fmul2(
                    state[4 * m], state[4 * m + 1], f[0], f[1]
                )
                scaled[4 * m + 2], scaled[4 * m + 3] = fmul2(
                    state[4 * m + 2], state[4 * m + 3], f[0], f[1]
                )
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + SIN_R)
        oa = o_acc.wait()
        # Ordered before oa.release() by the state store below and stage_y's wait::st.
        o_lo = tmem_ld("16x256b", 2, lane_base + cols["qs"] + oa.index * B_T)
        o_hi = [Float32(0.0)] * 8
        if cutlass.const_expr(self.d_v == 128):
            o_hi = tmem_ld("16x256b", 2, lane_base + (16 << 16) + cols["qs"] + oa.index * B_T)
        if cutlass.const_expr(self.d_v == 128):
            for j in cutlass.range_constexpr(half_blocks):
                tmem_st(
                    "32x32b",
                    lane_base + cols["state"] + (half_blocks + j) * 16,
                    [scaled[j * 16 + i] for i in range(16)],
                )
        else:
            tmem_st(
                "16x256b",
                lane_base + cols["state"] + d_k // 2,
                [scaled[i] for i in range(d_k // 4)],
            )
        self.stage_o(params, oh, o_lo, o_hi, ov_lo, ov_hi, active)
        # ---- Y = beta (V - K_decay S), U -------------------------------------------------------
        v_lo, v_hi = self.load_v(params, stage, geo, chunk_start, seqlen)
        s_beta = params.s_beta + stage * (B_T * 4)
        with self.trace_region("ks.wait"):
            spin_wait(tstate + KS, n_ks & 1)
        n_ks += 1
        # Ordered before KS(c+1) by stage_y's Y store and its wait::st.
        ks_lo = tmem_ld("16x256b", 2, lane_base + cols["ks"])
        ks_hi = [Float32(0.0)] * 8
        if cutlass.const_expr(self.d_v == 128):
            ks_hi = tmem_ld("16x256b", 2, lane_base + (16 << 16) + cols["ks"])
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
        # Publish the staged O only now: its smem_async commit fence (MEMBAR.ALL.CTA) stalls the
        # warp, which between SIN_R and Y lay on the per-chunk recurrence; here it overlaps the
        # U wait (0.9% at d128, 3% at d64). Needs QS_STAGES >= 2 (see its declaration).
        oh.commit()
        oa.release()
        n_u = self.stage_u(tstate, n_u, geo)
        raw_beta_release(raw, beta, stage)
        eh.release(signal=False)  # signal-only channel (see __init__)
        return exchange, o_acc, o_smem, n_ks, n_u, n_upd

    @cute.jit
    def stage_o(self, params, oh, o_lo, o_hi, ov_lo, ov_hi, active):
        """Scale one O accumulator tile and stage it (``stmatrix.trans``) for the TMA store."""
        oh.access(write=True, by="threads")
        s_o = params.s_o + oh.index * (B_T * self.d_v * 2)
        scale = params.scale
        lo = []
        hi = []
        for r in cutlass.range_constexpr(4):
            a0, a1 = fmul2(o_lo[2 * r], o_lo[2 * r + 1], scale, scale)
            if cutlass.const_expr(self.paged_state > 0):
                # Null routes emit zero output (predicated, so NaN inputs cannot leak).
                a0 = Float32(cutlass.select_(active, a0, Float32(0.0)))
                a1 = Float32(cutlass.select_(active, a1, Float32(0.0)))
            lo.append(fp32_to_fp16(a0, a1, dtype=self.io_dtype))
            if cutlass.const_expr(self.d_v == 128):
                b0, b1 = fmul2(o_hi[2 * r], o_hi[2 * r + 1], scale, scale)
                if cutlass.const_expr(self.paged_state > 0):
                    b0 = Float32(cutlass.select_(active, b0, Float32(0.0)))
                    b1 = Float32(cutlass.select_(active, b1, Float32(0.0)))
                hi.append(fp32_to_fp16(b0, b1, dtype=self.io_dtype))
        stmatrix(s_o + 2 * ov_lo, lo, trans=True)
        if cutlass.const_expr(self.d_v == 128):
            stmatrix(s_o + 2 * ov_hi, hi, trans=True)

    @cute.jit
    def drain_o(self, params, o_acc, o_smem, geo, active):
        """The item's last chunk's O."""
        _l, lane_base, cols, ov_lo, ov_hi, _pc, _pcx, _vd, _rv = geo
        oa = o_acc.wait()
        o_lo = tmem_ld("16x256b", 2, lane_base + cols["qs"] + oa.index * B_T)
        o_hi = [Float32(0.0)] * 8
        if cutlass.const_expr(self.d_v == 128):
            o_hi = tmem_ld("16x256b", 2, lane_base + (16 << 16) + cols["qs"] + oa.index * B_T)
        with o_smem.produce() as oh:
            self.stage_o(params, oh, o_lo, o_hi, ov_lo, ov_hi, active)
        # Unlike cg1_chunk, no TMEM store follows these loads before the release: wait for them.
        cute.arch.fence_view_async_tmem_load()
        oa.release()
        return o_acc, o_smem

    @cute.jit
    def final_state(
        self, params, geo, batch, head_o, v_offset, seqlen, final_dst, slot, active, clear_empty
    ):
        """TMEM state -> ``state_out``; empty items copy the initial state or write zeros (paged:
        fresh empty routes clear their slot)."""
        _l, lane_base, cols, _lo, _hi, _pc, _pcx, value_dim, row_valid = geo
        out = params.state_out
        owns = final_dst >= 0
        if cutlass.const_expr(self.paged_state > 0):
            owns = owns and active
        if seqlen > 0:
            if owns:
                row = final_dst
                if cutlass.const_expr(self.paged_state > 0):
                    row = slot
                addr = (out.iterator + out.layout((row, head_o, value_dim + v_offset, 0))).toint()
                obytes = out.element_type.width // 8
                for kb in cutlass.range_constexpr(0, self.d_k, 32):
                    vals = tmem_ld("32x32b", 32, lane_base + cols["state"] + kb)
                    if row_valid:
                        if cutlass.const_expr(out.element_type is Float32):
                            for g in cutlass.range_constexpr(8):
                                stg_v4(
                                    addr + (kb + g * 4) * obytes,
                                    [as_i32(vals[g * 4 + t]) for t in range(4)],
                                )
                        else:
                            for g in cutlass.range_constexpr(4):
                                words = [
                                    fp32_to_fp16(
                                        vals[g * 8 + 2 * t],
                                        vals[g * 8 + 2 * t + 1],
                                        dtype=out.element_type,
                                    )
                                    for t in range(4)
                                ]
                                stg_v4(addr + (kb + g * 8) * obytes, words)
                # The CTA's last item has no later TMEM store to order these loads before
                # teardown frees TMEM.
                cute.arch.fence_view_async_tmem_load()
        else:
            if cutlass.const_expr(self.paged_state > 0):
                if row_valid and owns and clear_empty:
                    for k in cutlass.range(self.d_k, unroll=1):  # cold: empty items
                        out[slot, head_o, value_dim + v_offset, k] = Float32(0.0).to(
                            out.element_type
                        )
            else:
                if row_valid and final_dst >= 0:
                    for k in cutlass.range(self.d_k, unroll=1):  # cold: empty items
                        if cutlass.const_expr(params.state_in is not None):
                            out[final_dst, head_o, value_dim + v_offset, k] = params.state_in[
                                batch, head_o, value_dim + v_offset, k
                            ].to(out.element_type)
                        else:
                            out[final_dst, head_o, value_dim + v_offset, k] = Float32(0.0).to(
                                out.element_type
                            )

    # ---- register MMA (warp 12): KK, L, T_inv ---------------------------------------------

    @role(warps=12, regs=56)
    @cute.jit
    def regmma(
        self,
        params: KdaPrefillParams,
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
        # Kept in registers: otherwise ptxas rematerializes the lane id (S2R) in the chunk loop
        # in front of the beta loads, and the T_inv chain, on the d128 critical path, waits.
        beta_lo_off = opaque_i32(4 * row_lo)
        beta_hi_off = opaque_i32(4 * row_hi)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, _h, _v, _ws, write_end, compute_start, _s, _len, _t = self.item(params, tile)
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
                    beta_lo = lds_f32(s_beta + beta_lo_off)
                    beta_hi = lds_f32(s_beta + beta_hi_off)
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

    # ---- epilogue (warp 15): A = lower(Q_decay K_inv^T), O stores ---------------------------

    @role(warps=15, regs=56)
    @cute.jit
    def epilogue(
        self,
        params: KdaPrefillParams,
        *,
        kk: Consumer,
        qk: Consumer,
        a: Producer,
        o_smem: Consumer,
        sched: Consumer,
        work: WorkCursor,
        me: Local,
    ):
        lane = me.lane
        io = self.io_dtype
        ki_row = lane % 8 + (8 if lane // 16 else 0)
        ki_col = 8 if (lane // 8) % 2 else 0
        qd_row = lane % 8 + (8 if (lane // 8) % 2 else 0)
        qd_col = 8 if (lane // 8) // 2 else 0
        a_row = (lane & 7) + (8 if (lane // 8) & 1 else 0)
        a_col = 8 if lane // 8 >= 2 else 0
        a_idx = a_row * B_T + swz32(a_row, a_col)
        ki_offs = [opaque_i32(ki_row * 64 + swz128(ki_row, i * 16 + ki_col)) for i in range(4)]
        qd_offs = [opaque_i32(swz128(qd_row, qd_row * 64 + i * 16 + qd_col)) for i in range(4)]
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, head_o, v_offset, write_start, write_end, compute_start, start, seqlen, key = (
                    self.item(params, tile)
                )
                n_local = write_end - compute_start
                o_desc = sequence_descriptors(
                    (params.o_store,), start + seqlen, key, B_T, n_local > 0
                )
                for local in cutlass.range(n_local, unroll=1):
                    chunk = compute_start + local
                    qh = qk.wait()
                    kh = kk.wait()
                    qh.access("Qdecay", read=True, by="threads")
                    kh.access("Kinv", read=True, by="threads")
                    stage = qh.index * (self.d_k * B_T * 2)
                    acc = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        acc[i] = Float32(0.0)
                    for i in cutlass.range_constexpr(self.d_k // 16):
                        box = (i // 4) * (B_T * 64)
                        kif = ldmatrix(params.s_kinv + stage + 2 * (ki_offs[i % 4] + box), 4)
                        qdf = ldmatrix(params.s_qdecay + stage + 2 * (qd_offs[i % 4] + box), 4)
                        mma_step(acc, qdf, kif, k_step=0, M=16, N=16, dtype=io)
                    for i in cutlass.range_constexpr(8):
                        r = lane // 4 + (8 if i % 4 >= 2 else 0)
                        c = (i // 4) * 8 + 2 * (lane % 4) + (i % 2)
                        acc[i] = acc[i] if r >= c else Float32(0.0)
                    with a.produce() as ah:
                        ah.access(write=True, by="threads")
                        stmatrix(
                            params.s_a + ah.index * (B_T * B_T * 2) + 2 * a_idx,
                            [fp32_to_fp16(acc[2 * i], acc[2 * i + 1], dtype=io) for i in range(4)],
                        )
                    kh.release()
                    qh.release()
                    if local > 0:
                        o_smem = self.store_o(
                            params,
                            o_smem,
                            chunk - 1,
                            write_start,
                            start,
                            seqlen,
                            head_o,
                            v_offset,
                            o_desc,
                            True,
                        )
                if n_local > 0:  # the last chunk's store
                    last = (write_end - 1, write_start, start, seqlen, head_o, v_offset, o_desc)
                    o_smem = self.store_o(params, o_smem, *last, False)
                sched, tile = self.next_tile(sched)
            work.advance()
        bulk_wait_group(0)
        return kk, qk, a, o_smem, sched, work

    @cute.jit
    def store_o(
        self,
        params,
        o_smem,
        chunk,
        write_start,
        start,
        seqlen,
        head_o,
        v_offset,
        descs,
        warmup_check: cutlass.Constexpr[bool],
    ):
        """TMA-store one staged O chunk through the item's ``descs`` (rows past the sequence end
        are clipped); warm-up chunks before ``write_start`` release their stage unstored."""
        with o_smem.consume() as oh:
            oh.access(read=True, by="tma")
            do_store = cutlass.Boolean(True)
            if cutlass.const_expr(warmup_check):
                do_store = chunk >= write_start
            if do_store:
                valid = cutlass.min(seqlen - chunk * B_T, B_T)
                token = start + chunk * B_T
                (desc,) = descs
                s_o = params.s_o + oh.index * (B_T * self.d_v * 2)
                for vb in cutlass.range_constexpr(self.d_v // 64):
                    src = cute.make_tensor(
                        cute.make_ptr(
                            self.io_dtype,
                            s_o + vb * (B_T * 128),
                            cute.AddressSpace.smem,
                            assumed_align=1024,
                        ),
                        params.box_io,
                    )
                    col = v_offset // 64 + vb
                    ragged_store(params.o_store, src, token, valid, head_o, col, B_T, 64, desc)
                bulk_commit_group()
            bulk_wait_group(0, read=True)
        return o_smem

    # ---- tcgen05 MMA (warp 13) ------------------------------------------------------------

    @role(warps=13, regs=56)
    @cute.jit
    def mma(
        self,
        params: KdaPrefillParams,
        *,
        kk: Consumer,
        qk: Consumer,
        tinv: Consumer,
        a: Consumer,
        o_acc: Producer,
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
        qs_col = t0 + tmem.cols("qs")
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
                _b, _h, _v, _ws, write_end, compute_start, _s, _len, _t = self.item(params, tile)
                seeded = cutlass.Boolean(False)
                if cutlass.const_expr(self.use_initial_state):
                    seeded = compute_start == 0
                for local in cutlass.range(write_end - compute_start, unroll=1):
                    have_state = local > 0 or seeded
                    kh = kk.wait()
                    kh.access("Kdecay", read=True, by="umma")
                    desc_kd = desc_opaque(
                        smem_desc(params.s_kdecay + kh.index * stage_bytes, 16, 1024, SWIZZLE_128B)
                    )
                    qh = qk.wait()
                    qh.access("Qdecay", read=True, by="umma")
                    qh.access("Krestore", read=True, by="umma")
                    oh = o_acc.acquire()
                    qs = qs_col + oh.index * B_T
                    # KS and QS under one branch: a second ``have_state`` branch after the managed
                    # acquire made ptxas wrap the QS MMAs in per-lane ELECT loops.
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
                        # ---- QS = S Q_decay^T into the O accumulator -------------------------
                        desc_qd = desc_opaque(
                            smem_desc(
                                params.s_qdecay + qh.index * stage_bytes, 16, 1024, SWIZZLE_128B
                            )
                        )
                        for sb in cutlass.range_constexpr(ksd.num_k_steps // ksd.sps_B):
                            for k in cutlass.range_constexpr(ksd.sps_B):
                                mma_ts_op(
                                    ksd,
                                    sin_col + sb * a_seg,
                                    desc_qd + sb * b_seg,
                                    qs,
                                    k,
                                    cutlass.Boolean(sb + k > 0),
                                    elect,
                                )
                    kh.release(signal=False)  # the role's elected lane commits (below)
                    umma_commit(kh.empty_barrier, elect)
                    # ---- U = T_inv Y ------------------------------------------------------------
                    th = tinv.wait()
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
                    # ---- O += U A -------------------------------------------------------------
                    ah = a.wait()
                    ah.access(read=True, by="umma")
                    desc_a = desc_opaque(
                        smem_desc(params.s_a + ah.index * (B_T * B_T * 2), 16, 256, SWIZZLE_32B)
                    )
                    mma_ts_op(td, uin_col, desc_a, qs, 0, have_state, elect)
                    oh.commit(signal=False)
                    umma_commit(oh.full_barrier, elect)
                    ah.release(signal=False)
                    umma_commit(ah.empty_barrier, elect)
                    cum += 1
                sched, tile = self.next_tile(sched)
            work.advance()
        return kk, qk, tinv, a, o_acc, sched, work

    # ---- kernel and entry point ---------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        q_load,
        k_load,
        v_load,
        g_load,
        o_store,
        box_io,
        box_g,
        beta,
        state_in,
        state_out,
        state_indices,
        has_initial_state,
        work_items,
        work_count,
        scheduler,
        q_ratio,
        k_ratio,
        v_ratio,
        scale,
        layouts,
        trace_buffer,
        bound,
    ):
        self.begin_roles(work=bound, layouts=layouts)
        addr = {name: self.buffer(name).iterator.toint() for name in self.buf}
        params = KdaPrefillParams(
            q_load=q_load,
            k_load=k_load,
            v_load=v_load,
            g_load=g_load,
            o_store=o_store,
            box_io=box_io,
            box_g=box_g,
            beta=beta,
            state_in=state_in,
            state_out=state_out,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            work_items=work_items,
            work_count=work_count,
            scheduler=scheduler,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
            v_ratio=v_ratio,
            scale=scale,
            s_q=addr["Q"],
            s_k=addr["K"],
            s_v=addr["V"],
            s_g=addr["Gate"],
            s_ex=addr["Exchange"] if "Exchange" in addr else addr["Gate"],
            s_kdecay=addr["Kdecay"],
            s_qdecay=addr["Qdecay"],
            s_krestore=addr["Krestore"],
            s_kinv=addr["Kinv"],
            s_a=addr["A"],
            s_tinv=addr["Tinv"],
            s_o=addr["O"],
            s_beta=addr["Beta"],
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def launch(
        self,
        q: cute.Tensor,  # (tokens, q heads, d_k)
        k: cute.Tensor,  # (tokens, k heads, d_k)
        v: cute.Tensor,  # (tokens, v heads, d_v x tiles_per_head)
        gate: cute.Tensor,  # (tokens, heads, d_k) natural-log decay
        beta: cute.Tensor,  # (tokens, heads)
        state_in: cute.Tensor | None,  # (rows, heads, d_v x tiles_per_head, d_k)
        o: cute.Tensor,  # (tokens, heads, d_v x tiles_per_head)
        state_out: cute.Tensor | None,
        state_indices: cute.Tensor | None,  # (sequences,) int32 paged routes
        has_initial_state: cute.Tensor | None,  # (sequences,) uint8
        work_items: cute.Tensor,  # (rows, WORK_ITEM_FIELDS) int32
        work_count: cute.Tensor,  # (1,) int32 rows to run
        scheduler: cute.Tensor,  # (>= 1,) int32, zero
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        scale: Float32,
        stream: cuda.CUstream,
        trace_buffer: cute.Tensor | None = None,
    ):
        """Launch on ``num_ctas`` CTAs (PDL); callable from another ``@cute.jit`` host such as
        a launch bundle, after a kernel that wrote ``work_items``/``work_count`` and zeroed
        ``scheduler``."""
        io, gdt = self.io_dtype, self.gate_dtype
        heads_out = Int32(gate.shape[1]) * self.tiles_per_head
        q_ratio = heads_out // Int32(q.shape[1])
        k_ratio = heads_out // Int32(k.shape[1])
        v_ratio = heads_out // Int32(v.shape[1])
        box_io = cute.select(
            sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1), mode=[0, 1]
        )
        g_cols = 128 // (gdt.width // 8)
        box_g = cute.select(
            sm100_utils.make_smem_layout_epi(gdt, LayoutEnum.ROW_MAJOR, (B_T, g_cols), 1),
            mode=[0, 1],
        )

        def tdh(t):  # (token, head, dim) -> (token, dim, head)
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        def ragged(op, t, box, cols, index):  # clipped at each sequence end, keyed by tile head
            slots = slot_region(tma_slots, index, B_T, heads_out)
            return make_ragged_tma(op, tdh(t), box, B_T, cols, slots)

        q_load = ragged(g2s, q, box_io, 64, 0)
        k_load = ragged(g2s, k, box_io, 64, 1)
        v_load = ragged(g2s, v, box_io, 64, 2)
        g_load = ragged(g2s, gate, box_g, g_cols, 3)
        o_store = ragged(cpasync.CopyBulkTensorTileS2GOp(), o, box_io, 64, 4)
        d_k, d_v = self.d_k, self.d_v
        layouts = {
            "Kdecay": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Qdecay": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Krestore": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Kinv": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "A": cute.make_layout((B_T * B_T, INTER_STAGES)),
            "Tinv": cute.make_layout((B_T * B_T, INTER_STAGES)),
            "Q": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "K": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "V": cute.make_layout((d_v * B_T, RAW_STAGES)),
            "Gate": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "O": cute.make_layout((d_v * B_T, O_STAGES)),
            "Beta": cute.make_layout((B_T, RAW_STAGES)),
        }
        if cutlass.const_expr(self.gate != "fp32"):
            layouts["Exchange"] = cute.make_layout((d_k * B_T, self.ex_stages))
        bound = self.work.bind((self.num_ctas, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            q_load,
            k_load,
            v_load,
            g_load,
            o_store,
            box_io,
            box_g,
            beta,
            state_in,
            state_out,
            state_indices,
            has_initial_state,
            work_items,
            work_count,
            scheduler,
            q_ratio,
            k_ratio,
            v_ratio,
            scale,
            layouts,
            trace_buffer,
            bound,
        ).launch(**self.launch_config(bound), stream=stream, use_pdl=True)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        state_in: cute.Tensor | None,
        o: cute.Tensor,
        state_out: cute.Tensor | None,
        state_indices: cute.Tensor | None,
        has_initial_state: cute.Tensor | None,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        scheduler: cute.Tensor,
        tma_slots: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.launch(
            q,
            k,
            v,
            gate,
            beta,
            state_in,
            o,
            state_out,
            state_indices,
            has_initial_state,
            work_items,
            work_count,
            scheduler,
            tma_slots,
            scale,
            stream,
        )


# ---- standalone entry point (tests) ---------------------------------------------------------


def uncut_work_items(cu_seqlens: torch.Tensor, heads: int, tiles_per_head: int = 1):
    """``(work_items, work_count, scheduler)`` of the uncut forward: one item per (sequence, head
    x value tile) with any tokens, longest first (the LPT order of ``work_tables``)."""
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    starts = cu_seqlens[:-1].tolist()
    rows = []
    for b, (start, length) in enumerate(zip(starts, lens, strict=True)):
        chunks = -(-length // B_T)
        if chunks:
            for h in range(heads * tiles_per_head):
                rows.append([b, h, 0, chunks, 0, chunks, start, start + length, b, b])
    rows.sort(key=lambda r: -(r[5] - r[4]))
    device = cu_seqlens.device
    table = torch.tensor(rows or [[0] * WORK_ITEM_FIELDS], dtype=torch.int32, device=device)
    count = torch.tensor([len(rows)], dtype=torch.int32, device=device)
    return table, count, torch.zeros(1, dtype=torch.int32, device=device)


@jit_cache
def compile_kda_prefill(op: KdaPrefill, beta_name: str, state_name: str | None):
    """Compile for dynamic tokens, heads, sequences and work rows."""
    io, i32 = op.io_dtype, cutlass.Int32
    tokens, hq, hk, hv, heads, rows = (cute.sym_int() for _ in range(6))
    dv_full = op.d_v * op.tiles_per_head

    def packed(dtype, h, d):
        return make_fake_strided_tensor(
            dtype, (tokens, h, d), stride_divisibility=tma_divisibility(dtype)
        )

    state = None
    if state_name is not None:
        state = make_fake_strided_tensor(
            CUTE_DTYPES[state_name],
            (cute.sym_int(), heads, dv_full, op.d_k),
            stride_divisibility=4,
        )
    return compile_tvm_ffi(
        op,
        packed(io, hq, op.d_k),
        packed(io, hk, op.d_k),
        packed(io, hv, dv_full),
        packed(op.gate_dtype, heads, op.d_k),
        make_fake_strided_tensor(CUTE_DTYPES[beta_name], (tokens, heads), contiguous_dim=None),
        state if op.use_initial_state else None,
        packed(io, heads, dv_full),
        state if op.store_final_state else None,
        None,
        None,
        # Compact like the bundle prologues' tables (a dynamic row stride costs a few %).
        make_fake_compact_tensor(
            i32, (rows, WORK_ITEM_FIELDS), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        fake_tma_slots(),
        Float32(1.0),
        name=op.get_name(),
        opt_level=2,  # the launch bundles' level
    )


def kda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    tiles_per_head: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Packed ``[T, H, D]`` KDA forward: natural-log channel gate ``[T, H, d_k]``, beta ``[T, H]``,
    state ``[N, H, d_v, d_k]``; returns ``(o, final_state)``."""
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k and v must share a 16-bit dtype")
    heads, d_k, d_v = gate.shape[1], q.shape[-1], v.shape[-1]
    num_seqs = cu_seqlens.shape[0] - 1
    o = torch.empty(q.shape[0], heads, d_v, dtype=q.dtype, device=q.device)
    final = None
    if output_final_state:
        final = (
            initial_state.clone()
            if initial_state is not None
            else torch.zeros(num_seqs, heads, d_v, d_k, dtype=torch.float32, device=q.device)
        )
    state_name = None
    for t in (initial_state, final):
        if t is not None:
            state_name = TORCH_NAMES[t.dtype]
    op = KdaPrefill(
        TORCH_NAMES[q.dtype],
        TORCH_NAMES[gate.dtype],
        d_k,
        d_v // tiles_per_head,
        tiles_per_head,
        initial_state is not None,
        final is not None,
        0,
        torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    table, count, scheduler = uncut_work_items(cu_seqlens, heads, tiles_per_head)
    fn = compile_kda_prefill(op, TORCH_NAMES[beta.dtype], state_name)
    slots = new_tma_slots(op.tma_slot_rows(gate.shape[1]), q.device)
    fn(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        o,
        final,
        None,
        None,
        table,
        count,
        scheduler,
        slots,
        scale,
    )
    return o, final
