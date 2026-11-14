# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# kda_summary_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and
# channels with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-piece
# descriptor arrays, and the zero-seeded H with the natural-log gate is the only form.

"""KDA piece summaries (BT = 16, per-key-channel decay) for SM100.

Per (piece, head) work item, the state a piece produces from a zero seed (``H``, ``[d_v, d_k]``)
and its transition (``M``, ``[d_k, d_k]``), the two recurrences the piece chain composes::

    X_end = X_start @ M + H

Per chunk of 16 tokens and chain ``X`` in {H, M} (``g`` the in-chunk cumulative log2 gate per
key channel, ``S_X`` the fp32 state carried across chunks)::

    K_decay = K exp2(g)   K_inv = K exp2(-g)   K_restore = K exp2(g_last - g)
    L    = beta * strict_lower(K_decay K_inv^T)        T = (I + L)^-1  (16x16 blockwise, mma.sync)
    Y_H  = beta (V - K_decay S_H)                       Y_M = beta (0 - K_decay S_M)
    U_X  = T Y_X                                        S_X <- exp2(g_last) S_X + K_restore^T U_X

``H`` starts from zero (its first chunk has no ``K_decay S_H``); ``M`` starts from the identity
when the item starts at the piece's first chunk (``compute_start == 0``) and from zero
otherwise. Items are the first ``work_count[0]`` rows of the work table (``WORK_ITEM_FIELDS``;
``piece_chain`` builds it and zeroes the scheduler counter): chunks
``[compute_start, write_end)`` of the piece at tokens ``[batch_start, batch_end)``. The states
land at ``state_h[piece, head]`` and ``state_m[piece, head]`` when ``write_end`` is the piece's
last chunk; an empty piece writes ``H = 0`` and ``M = I``.

K/V/gate rows past a piece end load as zeros (``ragged.RaggedTma``), as from upstream's
per-piece descriptors; beta is zero there.

Roles (16 warps) and their handoffs. ``rr`` marks a round-robin side of a two-instance role:
``cg0`` and ``regmma`` instances take the CTA's cumulative chunks alternately (``c % 2``)::

    load (14)        --raw: 8 x K,V,Gate (TMA)-->      cg0 (both instances), cg1
                     --beta: 8 x f32[16]-->            regmma (both instances), cg1
                     --sched: 8 x next item-->         every other role
    cg0 (0-7) rr     --kk: K_decay, K_inv-->           mma, regmma (both instances)
                     --kr: K_restore-->                mma
                     --exchange: exp2(g) rows-->       cg1 (signal-only)
    regmma (12, 15) rr --tinv-->                       mma (13)

The TMEM recurrences of both chains between the MMA warp and the compute groups (fp16 state
operand halves, KS / U accumulators, Y / U operands, state-update completion) use the upstream
signal-only barriers (``tstate``, ``TSTATE``): every stage they guard is reused only after a
later handoff of the same recurrence, so a channel's reuse wait would only lengthen the
per-chunk critical path.
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
    Barriers,
    Consumer,
    Instance,
    Local,
    NamedBarrier,
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
    TORCH_NAMES,
    WORK_ITEM_FIELDS,
    as_i32,
    beta_residual,
    box128,
    ldg_l2_f32,
    lds_16_f32,
    lds_f32,
    lds_v4,
    lds_v4_f32,
    load_rows,
    mask_pair,
    mma_tf32_16x16,
    opaque_i32,
    raw_beta_release,
    raw_beta_wait,
    spin_wait,
    stg_v4,
    sts_f32,
    sts_tf32x8,
    swz32,
    swz128,
    tf32_fragment_offsets,
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
    opaque_f32_zero,
    opaque_i32_zero,
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

LOG2_E = 1.4426950408889634
RAW_STAGES = 8
SCHED_STAGES = 8
DECAY_STAGES = 2
INTER_STAGES = 2

# TMEM state handoffs of one chain (``tstate`` barriers, upstream counts), each completing once
# per epoch; chain H uses barriers [0, CHAIN), chain M [CHAIN, 2 * CHAIN):
#   SIN_L  left key half of the fp16 state operand: the CG0 instance (chunks after an item's
#          first), CG1 (M's identity seed); MMA waits before KS k-steps [0, n/2). Reuse:
#          rewritten after SACC_L.
#   SIN_R  right half, CG1; MMA waits before the rest of KS.
#   KS, U  KS / U accumulators, MMA commit; CG1 reads. KS(c+1) / U(c+1) follow YIN / UIN waits
#          that CG1 signals after reading them.
#   YIN    Y operand, CG1; MMA waits before U. Rewritten after U completes.
#   UIN    U operand (CG1) and the left-half state decay of both chains (CG0 instance, at the
#          chunk's end): 4 + 4 arrivals; MMA waits before the state update.
#   SACC_L / SACC_R (2 stages)  left / right state update committed; CG0 / CG1 read the state.
SIN_L, SIN_R, KS, U, YIN, UIN, SACC_L, SACC_R = 0, 1, 2, 3, 4, 5, 6, 8
CHAIN = 10
H, M = 0, CHAIN
TSTATE_COUNTS = (4, 4, 1, 1, 4, 8, 1, 1, 1, 1) * 2


@dataclass(frozen=True)
class KdaSummaryParams:
    k_load: RaggedTma  # (token, dim, head) loads, zero past each sequence end
    v_load: RaggedTma
    g_load: RaggedTma
    box_io: cute.ComposedLayout  # one 16 x 64 16-bit swizzled box
    box_g: cute.ComposedLayout  # one 16 x 128-byte gate box
    beta: cute.Tensor  # (token, head)
    state_h: cute.Tensor  # (piece, head, d_v, d_k) fp32
    state_m: cute.Tensor  # (piece, head, d_k, d_k) fp32
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
    s_kinv: Int32  # fp32 (tf32) under fp16 io
    s_kdecay32: Int32  # fp16 io only: the tf32 copy of K_decay
    s_tinv: Int32
    s_beta: Int32


class KdaSummary(WarpSpecialized):
    """Warpdrive KDA piece summaries for ``io`` fp16/bf16 operands and an fp32 or ``io`` gate;
    fp32 ``H`` and ``M``.
    ``fastmath=False`` exponentiates the gates with the non-flushing ``exp2`` and reciprocal."""

    def __init__(
        self, io: str, gate: str, d_k: int, d_v: int, num_ctas: int, fastmath: bool = True
    ):
        if d_k not in HEAD_DIMS or d_v not in HEAD_DIMS:
            raise ValueError(f"the KDA summary serves d_k, d_v in {HEAD_DIMS}, got {d_k}, {d_v}")
        if gate not in ("fp32", io):
            raise ValueError(f"the gate must be fp32 or the io dtype, got {gate}")
        self.io, self.gate, self.d_k, self.d_v, self.num_ctas = io, gate, d_k, d_v, num_ctas
        self.fastmath = fastmath
        self.io_dtype = CUTE_DTYPES[io]
        self.gate_dtype = CUTE_DTYPES[gate]
        # fp16 computes KK from tf32 decay tiles (``kda_common``).
        self.tf32_decay = io == "fp16"
        # An fp32 gate is exchanged in place (exp2 prefixes over the raw gate stage); a 16-bit
        # gate exchanges through its own 4-stage fp32 buffer.
        self.ex_stages = RAW_STAGES if gate == "fp32" else 4
        io_t, f32 = self.io_dtype, Float32
        plan = self.smem_plan = SmemPlan()
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
        if self.tf32_decay:
            for name in ("Kinv", "Kdecay32"):
                buf[name] = plan.buffer(name, f32, align=1024)
        else:
            buf["Kinv"] = plan.buffer("Kinv", io_t, align=1024)
        buf["Beta"] = plan.buffer("Beta", f32, align=16)
        self.buf = buf
        rr = Sharing.ROUND_ROBIN
        io_b = io_t.width // 8
        raw_bytes = B_T * (d_k * io_b + d_v * io_b + d_k * (self.gate_dtype.width // 8))
        # The TMA check reports raw unverified: it cannot follow the per-item domain_offset views.
        self.raw = Ring(
            RAW_STAGES,
            Tma(raw_bytes),
            {"cg0": Warps, "cg1": Warps},
            buffer={k: buf[k] for k in ("K", "V", "Gate")},
        )
        # Signal-only (the upstream gate_exchange_ready): CG0 instances arrive raw, CG1 waits. An
        # fp32 gate exchanges inside the raw gate stage, which the raw ring recycles; a 16-bit
        # gate's own stage is rewritten 4 chunks later, after CG1 read it (CG0 has by then
        # waited for the state update of 3 chunks later, whose U operands CG1 published after).
        self.exchange = Ring(
            self.ex_stages,
            consumer=Warps,
            external="producer",
            override={"full_counts": 4},
            buffer=None if gate == "fp32" else buf["Exchange"],
        )
        self.beta = Ring(RAW_STAGES, Warps, Warps, buffer=buf["Beta"])
        self.sched = Ring(SCHED_STAGES, Warps, Warps, payload=(Int32, (1,)))
        smem_commit = "smem_async"  # generic SMEM stores, then tcgen05.mma reads
        # kk and tinv waits spin (the upstream waits): same median as suspend, without its
        # occasional +7% outlier launches.
        self.kk = Ring(
            DECAY_STAGES,
            Warps,
            {"mma": Umma, "regmma": Warps},
            producer_sharing=rr,
            wait="spin",
            commit_fence=smem_commit,
            buffer={k: buf[k] for k in ("Kdecay", "Kinv", "Kdecay32") if k in buf},
        )
        self.kr = Ring(
            DECAY_STAGES,
            Warps,
            Umma,
            producer_sharing=rr,
            commit_fence=smem_commit,
            buffer=buf["Krestore"],
        )
        self.tinv = Ring(
            INTER_STAGES,
            Warps,
            Umma,
            producer_sharing=rr,
            wait="spin",
            commit_fence=smem_commit,
            buffer=buf["Tinv"],
        )
        # The TMEM recurrences of both chains (upstream protocol): signal-only barriers whose
        # stage reuse is implied by the recurrence itself (see ``TSTATE``).
        self.tstate = Barriers(len(TSTATE_COUNTS), counts=list(TSTATE_COUNTS))
        # Each CG0 instance syncs its four warps between its exchange stores and their reads.
        self.exchange0 = NamedBarrier(threads=128)
        self.exchange1 = NamedBarrier(threads=128)
        tmem_plan = TmemPlan()
        for chain in "hm":
            (
                tmem_plan.region(f"state_{chain}", d_k)
                .region(f"sin_{chain}", d_k // 2)
                .region(f"ks_{chain}", B_T)
                .region(f"u_{chain}", B_T)
                .region(f"yin_{chain}", B_T // 2)
                .region(f"uin_{chain}", B_T // 2)
            )
        self.tmem_plan = tmem_plan
        self.work = WorkStream(schedule=OneShot(grid="1d"), ready=PdlWait())
        # MMA geometry per chain (M = d_v rows for H, d_k for M): KS (TMEM state operand x
        # K_decay), U (TMEM Y x the 16 x 16 T_inv tile), the state update (TMEM U x MN-major
        # K_restore), split in two key halves when d_k = 128.
        self.n_update = d_k // 2 if d_k // 2 >= 64 else d_k
        self.ks_desc, self.t_desc, self.kr_desc = {}, {}, {}
        for chain, rows in ((H, d_v), (M, d_k)):
            acc_idesc = instr_desc(f32, io_t, io_t, m=rows, n=B_T)
            self.ks_desc[chain] = MmaDesc(M=rows, N=B_T, K=d_k, bpe_a=2, bpe_b=2, idesc=acc_idesc)
            self.t_desc[chain] = MmaDesc(M=rows, N=B_T, K=B_T, bpe_a=2, bpe_b=2, idesc=acc_idesc)
            self.kr_desc[chain] = MmaDesc(
                M=rows,
                N=self.n_update,
                K=B_T,
                bpe_a=2,
                bpe_b=2,
                btranspose=True,
                idesc=instr_desc(f32, io_t, io_t, m=rows, n=self.n_update, b_mn_major=True),
            )

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: K, V and gate, keyed by head."""
        return tma_slot_rows(3, B_T, heads)

    def get_name(self) -> str:
        name = f"wd_kda_summary_{self.io}_g{self.gate}_k{self.d_k}_v{self.d_v}"
        return name if self.fastmath else f"{name}_nofm"

    # ---- work decode -------------------------------------------------------------------

    @cute.jit
    def item(self, params: KdaSummaryParams, tile):
        """``(piece, head, write_end, compute_start, batch_start, seqlen)`` of work row
        ``tile``."""
        row = params.work_items
        start = row[tile, 6]
        return row[tile, 0], row[tile, 1], row[tile, 3], row[tile, 4], start, row[tile, 7] - start

    @cute.jit
    def next_tile(self, sched):
        with sched.consume() as h:
            tile = h.data[0]
        return sched, tile

    # ---- load (warp 14): beta one chunk ahead, K/Gate/V TMA, next item -------------------

    @role(warps=14, regs=56)
    @cute.jit
    def load(
        self,
        params: KdaSummaryParams,
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
    def beta_row(self, params, start, seqlen, head, chunk, lane):
        """Lane ``lane``'s beta of ``chunk`` as fp32 (zero past the piece end and on lanes past
        the chunk), loaded through L2 only (``ldg_l2_f32``)."""
        value = Float32(0.0)
        token = chunk * B_T + lane
        if lane < B_T and token < seqlen:
            beta = params.beta
            addr = (beta.iterator + beta.layout((start + token, head))).toint()
            value = ldg_l2_f32(addr, beta.element_type)
        return value

    @cute.jit
    def load_item(self, params, fields, raw, beta, lane):
        """Beta (one chunk ahead) and the K, Gate and V boxes of every chunk of one item."""
        _p, head, write_end, compute_start, start, seqlen = fields
        head_k = head // params.k_ratio
        head_v = head // params.v_ratio
        loads = (params.k_load, params.g_load, params.v_load)
        descs = sequence_descriptors(loads, start + seqlen, head, B_T, write_end > compute_start)
        carry = self.beta_row(params, start, seqlen, head, compute_start, lane)
        for chunk in cutlass.range(compute_start, write_end, 1, unroll=1):
            with beta.produce() as bh:
                bh.access(write=True, by="threads")
                if lane < B_T:
                    sts_f32(params.s_beta + bh.index * (B_T * 4) + 4 * lane, carry)
            carry = self.beta_row(params, start, seqlen, head, chunk + 1, lane)
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

    # ---- compute group 0 (warps 0-7): gate scan, decay operands, left state halves ------

    @role(warps=[range(4), range(4, 8)], regs=160)
    @cute.jit
    def cg0(
        self,
        params: KdaSummaryParams,
        *,
        stage: Instance,
        raw: Consumer,
        kk: Producer,
        kr: Producer,
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
        cols = {
            chain: (lane_base + tmem.cols(f"state_{name}"), lane_base + tmem.cols(f"sin_{name}"))
            for chain, name in ((H, "h"), (M, "m"))
        }
        barrier = self.exchange0 if stage == 0 else self.exchange1
        raw_next = Int32(0)  # next raw epoch this instance takes (both instances join raw)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            cum_base = Int32(0)
            while tile < total:
                _p, _h, write_end, compute_start, _s, seqlen = self.item(params, tile)
                n_local = write_end - compute_start
                for local in cutlass.range((stage + cum_base) & 1, n_local, 2, unroll=1):
                    c = cum_base + local
                    if raw_next < c:  # the other instance's chunk: release its stage
                        with raw.consume():
                            pass
                    raw_next = c + 1
                    raw, kk, kr = self.cg0_chunk(
                        params,
                        raw,
                        kk,
                        kr,
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
                        cols,
                    )
                cum_base += n_local
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, kk, kr, sched, work

    @cute.jit
    def cg0_chunk(
        self,
        params,
        raw,
        kk,
        kr,
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
        cols,
    ):
        io = self.io_dtype
        d_k = self.d_k
        dk_halves = d_k // 64
        channel_rows = d_k // 4
        store_rows = B_T * channel_rows // 32
        store_row_base = (lane // channel_rows) * store_rows
        rh = raw.wait()
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
            g[row] = cute.math.exp2(g[row], fastmath=self.fastmath)
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

        # ---- raw K rows (zeroed past the piece end), exp2(g) of the row ----------------------
        k_stage = rh.index * (d_k * B_T * 2)
        k_words = []
        row_valid = chunk_start + decay_row < seqlen
        for half in cutlass.range_constexpr(dk_halves):
            kw = lds_v4(params.s_k + k_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8))
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
        kd32 = []  # fp32 K_decay / K_inv for the tf32 tiles (fp16 io)
        ki32 = []
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
                inv0 = cute.math.rcp(exp_g[r0], approx=True, ftz=self.fastmath)
                inv1 = cute.math.rcp(exp_g[r0 + 1], approx=True, ftz=self.fastmath)
                ki0, ki1 = fmul2(raw_k[r0], raw_k[r0 + 1], inv0, inv1)
                if cutlass.const_expr(self.tf32_decay):
                    kd32 += [kd0, kd1]
                    ki32 += [ki0, ki1]
                else:
                    ki_pack.append(fp32_to_fp16(ki0, ki1, dtype=io))
                kr0, kr1 = fmul2(ki0, ki1, exp_last[2 * p], exp_last[2 * p + 1])
                kr_pack.append(fp32_to_fp16(kr0, kr1, dtype=io))
        kh = kk.acquire()
        kh.access("Kdecay", write=True, by="threads")
        kh.access("Kinv", write=True, by="threads")
        decay_stage = kh.index * (d_k * B_T * 2)
        for half in cutlass.range_constexpr(dk_halves):
            off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
            if cutlass.const_expr(self.tf32_decay):
                for base, vals in ((params.s_kinv, ki32), (params.s_kdecay32, kd32)):
                    sts_tf32x8(
                        base + 2 * decay_stage,
                        decay_row,
                        half * 64 + lane_in_row * 8,
                        vals[8 * half : 8 * half + 8],
                        d_k,
                    )
            else:
                st_shared_v4(params.s_kinv + off, ki_pack[4 * half : 4 * half + 4])
            st_shared_v4(params.s_kdecay + off, kd_pack[4 * half : 4 * half + 4])
        if cutlass.const_expr(self.tf32_decay):
            kh.access("Kdecay32", write=True, by="threads")
        kh.commit()
        rsh = kr.acquire()
        rsh.access(write=True, by="threads")
        restore_stage = rsh.index * (d_k * B_T * 2)
        for half in cutlass.range_constexpr(dk_halves):
            off = restore_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
            st_shared_v4(params.s_krestore + off, kr_pack[4 * half : 4 * half + 4])
        rsh.commit()

        # ---- left key half of both states: fp16 operand, then decay by exp2(g_last) ---------
        for chain in cutlass.range_constexpr(H, M + 1, CHAIN):
            state_col, sin_col = cols[chain]
            if c > 0:  # the previous chunk's left update has landed
                with self.trace_region("sacc_l.wait"):
                    spin_wait(tstate + chain + SACC_L + ((c - 1) & 1), ((c - 1) >> 1) & 1)
            if local > 0:
                blocks = [tmem_ld("32x32b", 32, state_col + b * 32) for b in range(dk_halves)]
                for b in cutlass.range_constexpr(dk_halves):
                    words = [
                        fp32_to_fp16(blocks[b][2 * i], blocks[b][2 * i + 1], dtype=io)
                        for i in range(16)
                    ]
                    tmem_st("32x32b", sin_col + b * 16, words)
                cute.arch.fence_view_async_tmem_store()
                warp_arrive(tstate + chain + SIN_L)
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
                                blocks[b][grp * 4 + 2 * t],
                                blocks[b][grp * 4 + 2 * t + 1],
                                f[2 * t],
                                f[2 * t + 1],
                            )
                            scaled += [s0, s1]
                    tmem_st("32x32b", state_col + b * 32, scaled)
                cute.arch.fence_view_async_tmem_store()
        # Both left halves are decayed (with CG1's U operands).
        warp_arrive(tstate + H + UIN)
        warp_arrive(tstate + M + UIN)
        rh.release()
        return raw, kk, kr

    # ---- compute group 1 (warps 8-11): M seed, right state halves, Y / U, stores --------

    @role(warps=range(8, 12), regs=136)
    @cute.jit
    def cg1(
        self,
        params: KdaSummaryParams,
        *,
        raw: Consumer,
        beta: Consumer,
        exchange: Consumer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        ov_row = (lane // 16) * 8 + (lane & 7)
        ov_col = ((lane // 8) & 1) * 8
        # TMEM row of this thread: H has d_v rows, M d_k (a 64-row tile uses 16 lanes of each
        # subpartition).
        if cutlass.const_expr(self.d_v == 128):
            value_dim = warp * 32 + lane
            value_base = warp * 32
            row_valid_h = cutlass.Boolean(True)
        else:
            value_dim = warp * 16 + lane % 16
            value_base = warp * 16
            row_valid_h = lane < 16
        if cutlass.const_expr(self.d_k == 128):
            key_dim = warp * 32 + lane
            row_valid_m = cutlass.Boolean(True)
        else:
            key_dim = warp * 16 + lane % 16
            row_valid_m = lane < 16
        lane_base = tmem.base.toint() + (
            (32 * (warp % 4)) << 16
        )  # warp-uniform: this subpartition
        cols = {}
        for chain, name in ((H, "h"), (M, "m")):
            for region in ("state", "sin", "ks", "u", "yin", "uin"):
                cols[chain, region] = lane_base + tmem.cols(f"{region}_{name}")
        ov_lo = box128(ov_row, value_base + ov_col)
        ov_hi = box128(ov_row, value_base + 16 + ov_col)
        geo = (lane, cols, ov_lo, ov_hi, value_dim, row_valid_h, key_dim, row_valid_m)
        # Completed waits on the KS accumulators, the U accumulators (both chains per epoch)
        # and the right-half state updates (both chains per epoch).
        n_ks_h = Int32(0)
        n_ks_m = Int32(0)
        n_u = Int32(0)
        n_upd = Int32(0)
        n_raw = Int32(0)  # chunks consumed from raw and beta (see raw_beta_wait)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                piece, head, write_end, compute_start, _s, seqlen = self.item(params, tile)
                n_local = write_end - compute_start
                seeded = compute_start == 0
                if n_local > 0:
                    exchange, n_ks_m, n_u = self.cg1_first(
                        params,
                        raw,
                        beta,
                        exchange,
                        tstate,
                        n_raw,
                        n_ks_m,
                        n_u,
                        geo,
                        seeded,
                        compute_start * B_T,
                        seqlen,
                    )
                for local in cutlass.range(1, n_local, 1, unroll=1):
                    exchange, n_ks_h, n_ks_m, n_u, n_upd = self.cg1_chunk(
                        params,
                        raw,
                        beta,
                        exchange,
                        tstate,
                        n_raw + local,
                        n_ks_h,
                        n_ks_m,
                        n_u,
                        n_upd,
                        geo,
                        (compute_start + local) * B_T,
                        seqlen,
                    )
                n_raw += n_local
                if n_local > 0:  # the last updates (M's commit follows every MMA of the chunk)
                    with self.trace_region("sacc_r.wait"):
                        spin_wait(tstate + H + SACC_R + (n_upd & 1), (n_upd >> 1) & 1)
                        spin_wait(tstate + M + SACC_R + (n_upd & 1), (n_upd >> 1) & 1)
                    n_upd += 1
                self.store_states(params, geo, piece, head, write_end, seqlen)
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, beta, exchange, sched, work

    @cute.jit
    def cg1_first(
        self,
        params,
        raw,
        beta,
        exchange,
        tstate,
        n_raw,
        n_ks_m,
        n_u,
        geo,
        seeded,
        chunk_start,
        seqlen,
    ):
        """An item's first chunk: M's identity seed, Y (only M can carry a state), U."""
        lane = geo[0]
        io = self.io_dtype
        eh = exchange.wait()
        s_ex = params.s_g + eh.index * (self.d_k * B_T * 4)
        if cutlass.const_expr(self.gate != "fp32"):
            s_ex = params.s_ex + eh.index * (self.d_k * B_T * 4)
        if seeded:
            self.seed_identity(geo, tstate, s_ex)
        stage = raw_beta_wait(raw, beta, n_raw, RAW_STAGES)
        v_lo, v_hi = self.load_v(params, stage, geo, chunk_start, seqlen)
        betas = self.load_betas(params, stage, lane)
        y_lo = [Int32(0)] * 4
        y_hi = [Int32(0)] * 4
        for r in cutlass.range_constexpr(4):
            y_lo[r] = beta_residual(v_lo[r], *betas[r], None, None, io, False)
            y_hi[r] = beta_residual(v_hi[r], *betas[r], None, None, io, False)
        self.stage_y(tstate, geo, H, y_lo, y_hi)
        zero = opaque_i32_zero()
        if seeded:
            with self.trace_region("ks.wait"):
                spin_wait(tstate + M + KS, n_ks_m & 1)
            n_ks_m += 1
            ys_lo, ys_hi = self.residual_with_state(geo, M, [zero] * 4, [zero] * 4, betas)
            for r in cutlass.range_constexpr(4):
                y_lo[r] = ys_lo[r]
                y_hi[r] = ys_hi[r]
        else:
            for r in cutlass.range_constexpr(4):
                y_lo[r] = beta_residual(zero, *betas[r], None, None, io, False)
                y_hi[r] = beta_residual(zero, *betas[r], None, None, io, False)
        self.stage_y(tstate, geo, M, y_lo, y_hi)
        raw_beta_release(raw, beta, stage)
        n_u = self.stage_u(tstate, n_u, geo)
        eh.release(signal=False)  # signal-only channel (see __init__)
        return exchange, n_ks_m, n_u

    @cute.jit
    def seed_identity(self, geo, tstate, s_ex):
        """M's identity seed -> fp16 state operand (left half, then right, each published as
        soon as it is stored) and fp32 state decayed by the first chunk's exp2(g_last)."""
        _l, cols, _lo, _hi, _vd, _rvh, key_dim, _rvm = geo
        io = self.io_dtype
        zero = opaque_f32_zero()
        one = zero + Float32(1.0)
        for half in cutlass.range_constexpr(2):
            blocks = range(half * (self.d_k // 32), (half + 1) * (self.d_k // 32))
            vecs = []
            for i in cutlass.range_constexpr(blocks.start, blocks.stop):
                vecs.append([one if key_dim == i * 16 + j else zero for j in range(16)])
            for j, i in enumerate(blocks):
                words = [
                    fp32_to_fp16(vecs[j][2 * p], vecs[j][2 * p + 1], dtype=io) for p in range(8)
                ]
                tmem_st("32x32b", cols[M, "sin"] + i * 8, words)
            cute.arch.fence_view_async_tmem_store()
            warp_arrive(tstate + M + (SIN_L if half == 0 else SIN_R))
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
                tmem_st("32x32b", cols[M, "state"] + i * 16, scaled)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_v(self, params, stage, geo, chunk_start, seqlen):
        """This thread's transposed V pairs (``ldmatrix.trans``), zeroed past the piece end."""
        lane, _c, ov_lo, ov_hi, _vd, _rvh, _kd, _rvm = geo
        s_v = params.s_v + stage * (self.d_v * B_T * 2)
        v_lo = ldmatrix(s_v + 2 * ov_lo, 4, trans=True)
        v_hi = v_lo
        if cutlass.const_expr(self.d_v == 128):
            v_hi = ldmatrix(s_v + 2 * ov_hi, 4, trans=True)
        if chunk_start + B_T > seqlen:
            valid = seqlen - chunk_start
            for r in cutlass.range_constexpr(4):
                token0 = ((r // 2) * 4 + (lane & 3)) * 2
                v_lo[r] = mask_pair(v_lo[r], token0, valid)
                if cutlass.const_expr(self.d_v == 128):
                    v_hi[r] = mask_pair(v_hi[r], token0, valid)
        return v_lo, v_hi

    @cute.jit
    def load_betas(self, params, stage, lane):
        s_beta = params.s_beta + stage * (B_T * 4)
        betas = []
        for r in cutlass.range_constexpr(4):
            token0 = ((r // 2) * 4 + (lane & 3)) * 2
            betas.append((lds_f32(s_beta + 4 * token0), lds_f32(s_beta + 4 * token0 + 4)))
        return betas

    @cute.jit
    def residual_with_state(self, geo, chain: cutlass.Constexpr, v_lo, v_hi, betas):
        """``beta (V - KS)`` of ``chain`` from its KS accumulator (after its ``KS`` wait)."""
        _l, cols, _lo, _hi, _vd, _rvh, _kd, _rvm = geo
        io = self.io_dtype
        two = self.d_v == 128 if chain == H else self.d_k == 128
        y_lo = [Int32(0)] * 4
        y_hi = [Int32(0)] * 4
        ks_lo = tmem_ld("16x256b", 2, cols[chain, "ks"])
        for r in cutlass.range_constexpr(4):
            y_lo[r] = beta_residual(v_lo[r], *betas[r], ks_lo[2 * r], ks_lo[2 * r + 1], io, True)
        if cutlass.const_expr(two):
            ks_hi = tmem_ld("16x256b", 2, cols[chain, "ks"] + (16 << 16))
            for r in cutlass.range_constexpr(4):
                y_hi[r] = beta_residual(
                    v_hi[r], *betas[r], ks_hi[2 * r], ks_hi[2 * r + 1], io, True
                )
        return y_lo, y_hi

    @cute.jit
    def stage_y(self, tstate, geo, chain: cutlass.Constexpr, y_lo, y_hi):
        """Publish ``chain``'s packed Y operand to TMEM."""
        _l, cols, _lo, _hi, _vd, _rvh, _kd, _rvm = geo
        two = self.d_v == 128 if chain == H else self.d_k == 128
        tmem_st("16x128b", cols[chain, "yin"], y_lo)
        if cutlass.const_expr(two):
            tmem_st("16x128b", cols[chain, "yin"] + (16 << 16), y_hi)
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + chain + YIN)

    @cute.jit
    def stage_u(self, tstate, n_u, geo):
        """Wait for U_X = T Y_X and publish its fp16 copy as the next MMA operand, H then M."""
        _l, cols, _lo, _hi, _vd, _rvh, _kd, _rvm = geo
        for chain in cutlass.range_constexpr(H, M + 1, CHAIN):
            with self.trace_region("u.wait"):
                spin_wait(tstate + chain + U, n_u & 1)
            vals = tmem_ld("32x32b", B_T, cols[chain, "u"])
            words = [
                fp32_to_fp16(vals[2 * i], vals[2 * i + 1], dtype=self.io_dtype) for i in range(8)
            ]
            tmem_st("32x32b", cols[chain, "uin"], words)
            cute.arch.fence_view_async_tmem_store()
            warp_arrive(tstate + chain + UIN)
        return n_u + 1

    @cute.jit
    def right_half(self, params, tstate, n_upd, geo, chain: cutlass.Constexpr, exchange, eh):
        """``chain``'s right key half after the previous update: fp16 operand, then the fp32
        decay by exp2(g_last). Takes the exchange stage on the first chain."""
        _l, cols, _lo, _hi, _vd, _rvh, _kd, _rvm = geo
        io = self.io_dtype
        half_blocks = self.d_k // 32
        with self.trace_region("sacc_r.wait"):
            spin_wait(tstate + chain + SACC_R + (n_upd & 1), (n_upd >> 1) & 1)
        state = []
        for j in cutlass.range_constexpr(half_blocks):
            state.append(tmem_ld("32x32b", 16, cols[chain, "state"] + (half_blocks + j) * 16))
        for j in cutlass.range_constexpr(half_blocks):
            words = [
                fp32_to_fp16(state[j][2 * p], state[j][2 * p + 1], dtype=io) for p in range(8)
            ]
            tmem_st("32x32b", cols[chain, "sin"] + (half_blocks + j) * 8, words)
        if cutlass.const_expr(chain == H):
            eh = exchange.wait()
        s_ex = params.s_g + eh.index * (self.d_k * B_T * 4)
        if cutlass.const_expr(self.gate != "fp32"):
            s_ex = params.s_ex + eh.index * (self.d_k * B_T * 4)
        scaled = []
        for j in cutlass.range_constexpr(half_blocks):
            block = []
            for grp in cutlass.range_constexpr(4):
                dim = (half_blocks + j) * 16 + grp * 4
                seg = dim // 32
                idx = (
                    seg * (B_T * 32) + (B_T - 1) * 32 + swz128((B_T - 1) ^ seg, dim - seg * 32, 4)
                )
                f = lds_v4_f32(s_ex + 4 * idx)
                for t in cutlass.range_constexpr(2):
                    s0, s1 = fmul2(
                        state[j][grp * 4 + 2 * t],
                        state[j][grp * 4 + 2 * t + 1],
                        f[2 * t],
                        f[2 * t + 1],
                    )
                    block += [s0, s1]
            scaled.append(block)
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + chain + SIN_R)
        for j in cutlass.range_constexpr(half_blocks):
            tmem_st("32x32b", cols[chain, "state"] + (half_blocks + j) * 16, scaled[j])
        return exchange, eh

    @cute.jit
    def cg1_chunk(
        self,
        params,
        raw,
        beta,
        exchange,
        tstate,
        n_raw,
        n_ks_h,
        n_ks_m,
        n_u,
        n_upd,
        geo,
        chunk_start,
        seqlen,
    ):
        """A chunk after the first: right state halves of H and M, Y, U."""
        lane = geo[0]
        stage = raw_beta_wait(raw, beta, n_raw, RAW_STAGES)
        exchange, eh = self.right_half(params, tstate, n_upd, geo, H, exchange, None)
        exchange, eh = self.right_half(params, tstate, n_upd, geo, M, exchange, eh)
        n_upd += 1
        # ---- Y_H = beta (V - KS_H), Y_M = beta (0 - KS_M), U ---------------------------------
        v_lo, v_hi = self.load_v(params, stage, geo, chunk_start, seqlen)
        betas = self.load_betas(params, stage, lane)
        with self.trace_region("ks.wait"):
            spin_wait(tstate + H + KS, n_ks_h & 1)
        n_ks_h += 1
        y_lo, y_hi = self.residual_with_state(geo, H, v_lo, v_hi, betas)
        self.stage_y(tstate, geo, H, y_lo, y_hi)
        zero = opaque_i32_zero()
        with self.trace_region("ks.wait"):
            spin_wait(tstate + M + KS, n_ks_m & 1)
        n_ks_m += 1
        y_lo, y_hi = self.residual_with_state(geo, M, [zero] * 4, [zero] * 4, betas)
        self.stage_y(tstate, geo, M, y_lo, y_hi)
        raw_beta_release(raw, beta, stage)
        n_u = self.stage_u(tstate, n_u, geo)
        eh.release(signal=False)  # signal-only channel (see __init__)
        return exchange, n_ks_h, n_ks_m, n_u, n_upd

    @cute.jit
    def store_states(self, params, geo, piece, head, write_end, seqlen):
        """TMEM H and M -> ``state_h`` / ``state_m`` when the item ends its piece; an empty
        piece writes H = 0 and M = I."""
        _l, cols, _lo, _hi, value_dim, row_valid_h, key_dim, row_valid_m = geo
        d_k = self.d_k
        outs = (
            (H, params.state_h, value_dim, row_valid_h),
            (M, params.state_m, key_dim, row_valid_m),
        )
        if seqlen > 0:
            if write_end == (seqlen + B_T - 1) // B_T:
                for chain, out, row, valid in outs:
                    addr = (out.iterator + out.layout((piece, head, row, 0))).toint()
                    for kb in cutlass.range_constexpr(0, d_k, 32):
                        vals = tmem_ld("32x32b", 32, cols[chain, "state"] + kb)
                        if valid:
                            for g in cutlass.range_constexpr(8):
                                stg_v4(
                                    addr + (kb + g * 4) * 4,
                                    [as_i32(vals[g * 4 + t]) for t in range(4)],
                                )
        else:
            for _chain, out, row, valid in outs:
                if valid:  # cold: empty pieces
                    for k in cutlass.range(d_k, unroll=1):
                        out[piece, head, row, k] = Float32(0.0)
            if row_valid_m:
                params.state_m[piece, head, key_dim, key_dim] = Float32(1.0)

    # ---- register MMA (warps 12, 15): KK, L, T_inv ----------------------------------------

    @role(warps=[range(12, 13), range(15, 16)], regs=56)
    @cute.jit
    def regmma(
        self,
        params: KdaSummaryParams,
        *,
        stage: Instance,
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
        if cutlass.const_expr(self.tf32_decay):
            a_offs32 = tf32_fragment_offsets(lane, self.d_k, b_operand=False)
            b_offs32 = tf32_fragment_offsets(lane, self.d_k, b_operand=True)
        row_lo = lane // 4
        row_hi = row_lo + 8
        seen = Int32(0)  # next kk / beta epoch this instance takes (both instances join them)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            cum_base = Int32(0)
            while tile < total:
                _p, _h, write_end, compute_start, _s, _len = self.item(params, tile)
                n_local = write_end - compute_start
                for local in cutlass.range((stage + cum_base) & 1, n_local, 2, unroll=1):
                    c = cum_base + local
                    if seen < c:  # the other instance's chunk: release its stages
                        with kk.consume():
                            pass
                        with beta.consume():
                            pass
                    seen = c + 1
                    kh = kk.wait()
                    kh.access("Kdecay", read=True, by="threads")
                    kh.access("Kinv", read=True, by="threads")
                    stage_off = kh.index * (self.d_k * B_T * 2)
                    acc = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        acc[i] = Float32(0.0)
                    if cutlass.const_expr(self.tf32_decay):
                        kh.access("Kdecay32", read=True, by="threads")
                        mma_tf32_16x16(
                            acc,
                            params.s_kdecay32 + 2 * stage_off,
                            params.s_kinv + 2 * stage_off,
                            a_offs32,
                            b_offs32,
                            self.d_k,
                        )
                    else:
                        for i in cutlass.range_constexpr(self.d_k // 16):
                            box = (i // 4) * (B_T * 64)
                            kif = ldmatrix(
                                params.s_kinv + stage_off + 2 * (ki_offs[i % 4] + box), 4
                            )
                            kdf = ldmatrix(
                                params.s_kdecay + stage_off + 2 * (kd_offs[i % 4] + box), 4
                            )
                            mma_step(acc, kdf, kif, k_step=0, M=16, N=16, dtype=io)
                    bh = beta.wait()
                    bh.access(read=True, by="threads")
                    s_beta = params.s_beta + bh.index * (B_T * 4)
                    beta_lo = lds_f32(s_beta + 4 * row_lo)
                    beta_hi = lds_f32(s_beta + 4 * row_hi)
                    l_regs = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        r = row_hi if cutlass.const_expr(i % 4 >= 2) else row_lo
                        cc = (i // 4) * 8 + 2 * (lane % 4) + (i % 2)
                        l_regs[i] = acc[i] if r > cc else Float32(0.0)
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
                cum_base += n_local
                sched, tile = self.next_tile(sched)
            work.advance()
        return kk, beta, tinv, sched, work

    # ---- tcgen05 MMA (warp 13) ------------------------------------------------------------

    @role(warps=13, regs=56)
    @cute.jit
    def mma(
        self,
        params: KdaSummaryParams,
        *,
        kk: Consumer,
        kr: Consumer,
        tinv: Consumer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
    ):
        elect = elect_sync()
        t0 = tmem.base.toint()
        cols = {}
        for chain, name in ((H, "h"), (M, "m")):
            for region in ("state", "sin", "ks", "u", "yin", "uin"):
                cols[chain, region] = t0 + tmem.cols(f"{region}_{name}")
        split = self.n_update < self.d_k
        stage_bytes = self.d_k * B_T * 2
        # Completed waits on each chain's state-operand halves, on Y and on the U operand
        # (both chains per epoch).
        n_sin_h = Int32(0)
        n_sin_m = Int32(0)
        n_y = Int32(0)
        n_uin = Int32(0)
        cum = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _p, _h, write_end, compute_start, _s, _len = self.item(params, tile)
                seeded = compute_start == 0
                for local in cutlass.range(write_end - compute_start, unroll=1):
                    have_h = local > 0
                    have_m = local > 0 or seeded
                    kh = kk.wait()
                    kh.access("Kdecay", read=True, by="umma")
                    qh = kr.wait()
                    qh.access(read=True, by="umma")
                    # ---- KS_X = S_X K_decay^T, H then M (H has a state only if M has) ----------
                    if have_m:
                        if have_h:
                            n_sin_h = self.ks_gemm(params, tstate, cols, H, n_sin_h, kh, elect)
                        n_sin_m = self.ks_gemm(params, tstate, cols, M, n_sin_m, kh, elect)
                    kh.release(signal=False)  # the role's elected lane commits (below)
                    umma_commit(kh.empty_barrier, elect)
                    # ---- U_X = T_inv Y_X --------------------------------------------------------
                    th = tinv.wait()
                    th.access(read=True, by="umma")
                    # Each descriptor is built right after its wait: one kept live across a wait
                    # moved the channel state out of uniform registers (R2UR per MMA).
                    for chain in cutlass.range_constexpr(H, M + 1, CHAIN):
                        with self.trace_region("yin.wait"):
                            spin_wait(tstate + chain + YIN, n_y & 1)
                        desc_t = desc_opaque(
                            smem_desc(
                                params.s_tinv + th.index * (B_T * B_T * 2), 16, 256, SWIZZLE_32B
                            )
                        )
                        mma_ts_op(
                            self.t_desc[chain],
                            cols[chain, "yin"],
                            desc_t,
                            cols[chain, "u"],
                            0,
                            cutlass.Boolean(False),
                            elect,
                        )
                        umma_commit(tstate + chain + U, elect)
                    n_y += 1
                    th.release(signal=False)
                    umma_commit(th.empty_barrier, elect)
                    # ---- S_X += K_restore^T U_X, left then right key half, H then M -------------
                    for chain in cutlass.range_constexpr(H, M + 1, CHAIN):
                        krd = self.kr_desc[chain]
                        with self.trace_region("uin.wait"):
                            spin_wait(tstate + chain + UIN, n_uin & 1)
                        desc_kr = desc_opaque(
                            smem_desc(
                                params.s_krestore + qh.index * stage_bytes,
                                2048,
                                1024,
                                SWIZZLE_128B,
                            )
                        )
                        mma_ts_op(
                            krd,
                            cols[chain, "uin"],
                            desc_kr,
                            cols[chain, "state"],
                            0,
                            have_h if chain == H else have_m,
                            elect,
                        )
                        umma_commit(tstate + chain + SACC_L + (cum & 1), elect)
                        if cutlass.const_expr(split):
                            mma_ts_op(
                                krd,
                                cols[chain, "uin"],
                                desc_kr + ((B_T * 128) >> 4),
                                cols[chain, "state"] + self.n_update,
                                0,
                                have_h if chain == H else have_m,
                                elect,
                            )
                        if cutlass.const_expr(chain == M):
                            qh.release(signal=False)
                            umma_commit(qh.empty_barrier, elect)
                        umma_commit(tstate + chain + SACC_R + (cum & 1), elect)
                    n_uin += 1
                    cum += 1
                sched, tile = self.next_tile(sched)
            work.advance()
        return kk, kr, tinv, sched, work

    @cute.jit
    def ks_gemm(self, params, tstate, cols, chain: cutlass.Constexpr, n_sin, kh, elect):
        """KS_X = S_X K_decay^T after the left half of the state operand lands, the right half
        before the second half of the K steps; returns the next wait count."""
        ksd = self.ks_desc[chain]
        a_seg = ksd.sps_B * ksd.tmem_advance_A  # TMEM columns of one 64-wide K box
        b_seg = ksd.smem_subtile_B >> 4
        stage_bytes = self.d_k * B_T * 2
        desc_kd = desc_opaque(
            smem_desc(params.s_kdecay + kh.index * stage_bytes, 16, 1024, SWIZZLE_128B)
        )
        with self.trace_region("sin_l.wait"):
            spin_wait(tstate + chain + SIN_L, n_sin & 1)
        for f in cutlass.range_constexpr(ksd.num_k_steps):
            if cutlass.const_expr(f == ksd.num_k_steps // 2):
                with self.trace_region("sin_r.wait"):
                    spin_wait(tstate + chain + SIN_R, n_sin & 1)
            sb = f // ksd.sps_B
            mma_ts_op(
                ksd,
                cols[chain, "sin"] + sb * a_seg,
                desc_kd + sb * b_seg,
                cols[chain, "ks"],
                f - sb * ksd.sps_B,
                cutlass.Boolean(f > 0),
                elect,
            )
        umma_commit(tstate + chain + KS, elect)
        return n_sin + 1

    # ---- kernel and entry point ---------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        k_load,
        v_load,
        g_load,
        box_io,
        box_g,
        beta,
        state_h,
        state_m,
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
        params = KdaSummaryParams(
            k_load=k_load,
            v_load=v_load,
            g_load=g_load,
            box_io=box_io,
            box_g=box_g,
            beta=beta,
            state_h=state_h,
            state_m=state_m,
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
            s_kdecay32=addr.get("Kdecay32", addr["Kinv"]),
            s_tinv=addr["Tinv"],
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
        state_h: cute.Tensor,  # (pieces, heads, d_v, d_k) fp32
        state_m: cute.Tensor,  # (pieces, heads, d_k, d_k) fp32
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

        def tdh(t):  # (token, head, dim) -> (token, dim, head)
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        def ragged(op, t, box, cols, index):  # clipped at each sequence end, keyed by head
            slots = slot_region(tma_slots, index, B_T, heads)
            return make_ragged_tma(op, tdh(t), box, B_T, cols, slots)

        k_load = ragged(g2s, k, box_io, 64, 0)
        v_load = ragged(g2s, v, box_io, 64, 1)
        g_load = ragged(g2s, gate, box_g, g_cols, 2)
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
        }
        if cutlass.const_expr(self.tf32_decay):
            layouts["Kdecay32"] = cute.make_layout((d_k * B_T, DECAY_STAGES))
        if cutlass.const_expr(self.gate != "fp32"):
            layouts["Exchange"] = cute.make_layout((d_k * B_T, self.ex_stages))
        bound = self.work.bind((self.num_ctas, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            k_load,
            v_load,
            g_load,
            box_io,
            box_g,
            beta,
            state_h,
            state_m,
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
        state_h: cute.Tensor,
        state_m: cute.Tensor,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        scheduler: cute.Tensor,
        tma_slots: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.launch(
            k,
            v,
            gate,
            beta,
            state_h,
            state_m,
            work_items,
            work_count,
            scheduler,
            tma_slots,
            stream,
        )


# ---- standalone entry point (tests) ---------------------------------------------------------


@jit_cache
def compile_kda_summary(op: KdaSummary, beta_name: str):
    """Compile for dynamic tokens, heads, pieces and work rows."""
    io, i32 = op.io_dtype, cutlass.Int32
    tokens, hk, hv, heads, rows, pieces = (cute.sym_int() for _ in range(6))

    def packed(dtype, h, d):
        return make_fake_strided_tensor(
            dtype, (tokens, h, d), stride_divisibility=tma_divisibility(dtype)
        )

    def state(rows_, cols_):
        return make_fake_strided_tensor(
            Float32, (pieces, heads, rows_, cols_), stride_divisibility=4
        )

    return compile_tvm_ffi(
        op,
        packed(io, hk, op.d_k),
        packed(io, hv, op.d_v),
        packed(op.gate_dtype, heads, op.d_k),
        make_fake_strided_tensor(CUTE_DTYPES[beta_name], (tokens, heads)),
        state(op.d_v, op.d_k),
        state(op.d_k, op.d_k),
        # Compact like the chain prologue's table: a dynamic row stride measured 3-5% slower.
        make_fake_compact_tensor(
            i32, (rows, WORK_ITEM_FIELDS), stride_order=(1, 0), assumed_align=16
        ),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        fake_tma_slots(),
        name=op.get_name(),
        opt_level=2,  # the chain forward's level (every KDA module compiles at 2)
    )


def kda_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_h: torch.Tensor,
    state_m: torch.Tensor,
    work_items: torch.Tensor,
    work_count: torch.Tensor,
    scheduler: torch.Tensor | None = None,
    *,
    fastmath: bool = True,
) -> None:
    """Packed ``[T, H, D]`` piece summaries into ``state_h [pieces, H, d_v, d_k]`` and
    ``state_m [pieces, H, d_k, d_k]`` (fp32): natural-log channel gate ``[T, H, d_k]``, beta
    ``[T, H]``, the first ``work_count[0]`` rows of ``work_items`` (see the module docstring)."""
    if k.dtype not in (torch.float16, torch.bfloat16) or v.dtype != k.dtype:
        raise TypeError("k and v must share a 16-bit dtype")
    if state_h.dtype != torch.float32 or state_m.dtype != torch.float32:
        raise TypeError("H and M are fp32")
    if not work_items.is_contiguous() or beta.stride(-1) != 1:
        raise ValueError("work_items must be compact and beta contiguous over heads")
    if scheduler is None:
        scheduler = torch.zeros(1, dtype=torch.int32, device=k.device)
    op = KdaSummary(
        TORCH_NAMES[k.dtype],
        TORCH_NAMES[gate.dtype],
        k.shape[-1],
        v.shape[-1],
        torch.cuda.get_device_properties(k.device).multi_processor_count,
        fastmath,
    )
    fn = compile_kda_summary(op, TORCH_NAMES[beta.dtype])
    slots = new_tma_slots(op.tma_slot_rows(gate.shape[1]), k.device)
    fn(k, v, gate, beta, state_h, state_m, work_items, work_count, scheduler, slots)
