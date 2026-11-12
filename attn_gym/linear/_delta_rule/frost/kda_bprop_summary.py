# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# kda_bprop_summary_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and
# channels with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence
# descriptor arrays, one CTA runs one work item, and the natural-log fp32 gate and fp32 state
# gradients the drivers compile are the only forms.

"""KDA backward state summaries for SM100 (BT = 16, per-key-channel decay): per (sequence or
piece, head) work item, the gradient ``C`` its outputs send to the state entering it, from the
exit cotangent ``dS_end`` (zero, or ``d_final_state`` when seeded). The chain backward composes
them per piece (``dX_start = M^T dX_end + C``); ``summary.build_state_grad_summaries`` returns
them per sequence.

Per chunk ``c``, last to first (tokens ``[c * 16, (c + 1) * 16)``), with ``g`` the in-chunk
cumulative log2 gate per key channel and ``D`` the ``[d_v, d_k]`` fp32 state gradient (TMEM)::

    K_decay = K exp2(g)   K_inv = K exp2(-g)   Q_decay = scale Q exp2(g)
    L   = beta * strict_lower(K_decay K_inv^T)        T = (I + L)^-1  (16x16 blockwise, mma.sync)
    A   = lower(Q_decay K_inv^T)
    dU  = D16 @ K_inv^T + dO^T @ A                     (no D16 term on a zero-seeded last chunk)
    D  += dO^T @ Q_decay  (overwrite on a zero-seeded last chunk)
    dY  = dU16 @ T
    D  += (-beta dY)16 @ K_decay
    D  *= exp2(g_last of chunk c - 1);  D16 = 16-bit(D)          (the next chunk's operand)

A seeded item starts from ``D = exp2(g_last) dS_end`` of its last chunk. Items are the first
``work_count[0]`` rows of a split-K work table, ``(sequence, head, write_start, -, -,
compute_end, start, end, ...)``: chunks ``compute_end - 1`` down to ``write_start`` of the
sequence at tokens ``[start, end)``. ``C`` lands at ``d_initial_state[sequence, head]`` when
``write_start == 0``; an item without chunks writes ``dS_end`` (zero unseeded).

Q/K/gate/dO rows past a sequence end load as zeros (``frost.ragged.RaggedTma``), as from
upstream's per-sequence descriptors; beta is zero there.

Roles (12 warps; upstream's idle warps 8-11 are not launched) and their handoffs::

    load (10)     --raw:   2 x Q, K, Gate [16, d_k] (TMA)-->        cg0 (0-3)
                  --dout:  2 x dO [16, d_v] (TMA)-->                mma (9)
                  --beta:  4 x beta[16]-->                          regmma (8), cg1 (4-7)
    cg0           --kk:    2 x K_decay, K_inv-->                    mma, regmma, epilogue (11)
                  --qk:    2 x Q_decay-->                           mma, epilogue
                  --dscale: 2 x exp2(g_last)[d_k]-->                cg1
    regmma        --tinv:  2 x T [16, 16]-->                        mma
    epilogue      --a:     2 x A [16, 16]-->                        mma
    mma          <-trip:   1 (UMMA / threads)->                     cg1

``trip`` carries the TMEM recurrence between the MMA warp and compute group 1, three round trips
per chunk, each MMA acquire standing for one upstream wait. For an item's ``r``-th chunk in
processing order, epoch ``3r`` issues dU once CG1 published the previous chunk's D16 and
rescaled D (which the Q term then accumulates into), and CG1 packs dU16; ``3r + 1`` issues dY
and CG1 stages ``(-beta dY)16``; ``3r + 2`` issues the K term and CG1 rescales D and stages D16
for the next chunk, or stores ``C`` after the item's last one. A seeded item opens with one
more epoch, empty on the MMA side, for CG1's seed. Every TMEM region is rewritten only after a
later epoch of this one recurrence, as upstream's one-way signals imply. ``dout`` is released
by the Q term's completion (upstream: CG1 after the K term), the other SMEM rings by their
last reader. One CTA runs one work-table row (``OneShot`` over the capacity, limited to the
device-written count) instead of upstream's atomic ticket scheduler.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    Consumer,
    LastEntry,
    Local,
    NamedBarrier,
    OneShot,
    Producer,
    Ring,
    SmemPlan,
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
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_strided_tensor
from attn_gym.warpdrive.compat import LayoutEnum

from .blockwise_inverse import invert_unit_lower_16x16_fragments
from .kda_common import (
    B_T,
    WORK_ITEM_FIELDS,
    box128,
    lds_f32,
    lds_v4,
    lds_v4_f32,
    load_rows,
    load_state_16,
    opaque_i32,
    sts_f32,
    swz32,
    swz128,
)
from .ptx import (
    f16x2_to_f32,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    mma_step,
    st_global_v4_f32,
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
from .umma import (
    SWIZZLE_32B,
    SWIZZLE_128B,
    MmaDesc,
    desc_opaque,
    instr_desc,
    mma_ss,
    mma_ts,
    smem_desc,
)

DIMS = (64, 128)
LOG2_E = 1.4426950408889634
RAW_STAGES = 2
BETA_STAGES = 4
DECAY_STAGES = 2
INTER_STAGES = 2

_CUTE_DTYPES = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16}
_IO_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16"}


@dataclass(frozen=True)
class SummaryItems(WorkMap):
    """Rows of a split-K work table: ``(sequence, head, write_start, compute_end, start, end)``,
    chunks counted from the sequence's first token ``start``."""

    words = 6

    def count64(self, space):
        table, _ = space
        return Int64(table.shape[0])

    def decode(self, index, space, state):
        table, _ = space
        row = (table[index, 0], table[index, 1], table[index, 2], table[index, 5])
        return True, (*row, table[index, 6], table[index, 7]), state

    def get_key(self) -> str:
        return "kda_bprop_summary_items"


@dataclass(frozen=True)
class KdaBpropSummaryParams:
    q_load: RaggedTma  # (token, dim, head) loads, zero past each sequence end
    k_load: RaggedTma
    g_load: RaggedTma
    do_load: RaggedTma
    box_io: cute.ComposedLayout  # one 16 x 64 16-bit swizzled box
    box_g: cute.ComposedLayout  # one 16 x 32 fp32 swizzled box
    beta: cute.Tensor  # (token, head) fp32
    d_initial_state: cute.Tensor  # (sequence, head, d_v, d_k) fp32
    d_final_state: cute.Tensor | None
    q_ratio: Int32  # heads per q / k head
    k_ratio: Int32
    scale: Float32
    s_q: Int32  # SMEM byte addresses of the buffers (stage 0)
    s_k: Int32
    s_g: Int32
    s_do: Int32
    s_kdecay: Int32
    s_kinv: Int32
    s_qdecay: Int32
    s_a: Int32
    s_tinv: Int32


class KdaBpropSummary(WarpSpecialized):
    """Warpdrive KDA backward summaries for ``io`` fp16/bf16 operands, ``d_k``/``d_v`` 64 or 128,
    an fp32 natural-log gate and beta, and fp32 state gradients; ``seeded`` starts every item
    from ``d_final_state`` instead of zero."""

    def __init__(self, io: str, d_k: int, d_v: int, seeded: bool):
        if d_k not in DIMS or d_v not in DIMS:
            raise ValueError(f"the KDA bprop summary serves d_k, d_v in {DIMS}, got {d_k}, {d_v}")
        self.io, self.d_k, self.d_v, self.seeded = io, d_k, d_v, seeded
        self.io_dtype = io_t = _CUTE_DTYPES[io]
        plan = self.smem_plan = SmemPlan()
        names = ("Kdecay", "Kinv", "Qdecay", "A", "Tinv", "dO", "Q", "K")
        buf = {name: plan.buffer(name, io_t, align=1024) for name in names}
        buf["Gate"] = plan.buffer("Gate", Float32, align=1024)
        self.buf = buf
        self.qk_stage_bytes = d_k * B_T * 2
        self.do_stage_bytes = d_v * B_T * 2
        # The TMA check reports raw and dout unverified: it cannot follow copies through the
        # per-item domain_offset views.
        self.raw = Ring(
            RAW_STAGES,
            Tma(2 * self.qk_stage_bytes + d_k * B_T * 4),
            Warps,
            buffer={k: buf[k] for k in ("Q", "K", "Gate")},
        )
        self.dout = Ring(RAW_STAGES, Tma(self.do_stage_bytes), Umma, buffer=buf["dO"])
        self.beta = Ring(
            BETA_STAGES, Warps, {"regmma": Warps, "cg1": Warps}, payload=(Float32, (B_T,))
        )
        self.dscale = Ring(DECAY_STAGES, Warps, Warps, payload=(Float32, (d_k,)))
        self.kk = Ring(
            DECAY_STAGES,
            Warps,
            {"mma": Umma, "regmma": Warps, "epilogue": Warps},
            buffer={"Kdecay": buf["Kdecay"], "Kinv": buf["Kinv"]},
        )
        self.qk = Ring(DECAY_STAGES, Warps, {"mma": Umma, "epilogue": Warps}, buffer=buf["Qdecay"])
        self.tinv = Ring(INTER_STAGES, Warps, Umma, buffer=buf["Tinv"])
        self.a = Ring(INTER_STAGES, Warps, Umma, buffer=buf["A"])
        # Each release hands the MMA warp TMEM that CG1 loaded from and stored to.
        self.trip = Ring(1, Umma, Threads, release_fence=("tmem_load", "tmem_store"))
        # CG0 syncs its four warps between its in-place exp2(g) exchange stores and their reads.
        self.exchange = NamedBarrier(threads=128)
        self.tmem_plan = (
            TmemPlan()
            .region("dstate", d_k)  # D, fp32 [d_v, d_k]
            .region("dstate_in", d_k // 2)  # D16: the dU GEMM's A operand
            .region("du", B_T)
            .region("dy", B_T)
            .region("nbdy", B_T // 2)  # (-beta dY)16: the K term's A operand
            .region("du_in", B_T // 2)  # dU16: the dY GEMM's A operand
        )
        self.work = WorkStream(mapping=SummaryItems(), schedule=OneShot())

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: Q, K, gate and dO, keyed by head."""
        return tma_slot_rows(4, B_T, heads)

    def get_name(self) -> str:
        return f"frost_kda_bprop_summary_{self.io}_k{self.d_k}_v{self.d_v}_s{int(self.seeded)}"

    def _descs(self):
        """Static geometry of the five GEMMs (TMEM accumulators are ``d_v`` rows)."""
        io, d_k, d_v = self.io_dtype, self.d_k, self.d_v

        def desc(n, kk, atranspose=False, btranspose=False):
            idesc = instr_desc(
                Float32, io, io, m=d_v, n=n, a_mn_major=atranspose, b_mn_major=btranspose
            )
            return MmaDesc(
                M=d_v,
                N=n,
                K=kk,
                bpe_a=2,
                bpe_b=2,
                idesc=idesc,
                atranspose=atranspose,
                btranspose=btranspose,
            )

        return (
            desc(B_T, d_k),  # dU = D16 @ K_inv^T
            desc(B_T, B_T, atranspose=True, btranspose=True),  # dU += dO^T @ A
            desc(d_k, B_T, atranspose=True, btranspose=True),  # D += dO^T @ Q_decay
            desc(B_T, B_T, btranspose=True),  # dY = dU16 @ T
            desc(d_k, B_T, btranspose=True),  # D += (-beta dY)16 @ K_decay
        )

    # ---- load (warp 10): Q, K, Gate, dO and beta, last chunk first ----------------------

    @role(warps=10, regs=56)
    @cute.jit
    def load(
        self,
        params: KdaBpropSummaryParams,
        *,
        raw: Producer,
        dout: Producer,
        beta: Producer,
        work: WorkCursor,
        me: Local,
    ):
        lane = me.lane
        while work.valid:
            raw, dout, beta = self._load_item(params, work.item, raw, dout, beta, lane)
            work.advance()
        cute.arch.griddepcontrol_launch_dependents()
        return raw, dout, beta, work

    @cute.jit
    def _load_item(self, params, item, raw, dout, beta, lane):
        """Q, K, gate, dO and beta of every chunk of one item, last chunk first."""
        _seq, head, write_start, compute_end, start, end = item
        seqlen = end - start
        head_q = head // params.q_ratio
        head_k = head // params.k_ratio
        # Beta one chunk ahead (zero past the sequence end), so its load latency hides
        # behind the next chunk's TMA issue.
        carry = self._beta(params, head, start, seqlen, compute_end - 1, lane)
        loads = (params.q_load, params.k_load, params.g_load, params.do_load)
        descs = sequence_descriptors(loads, end, head, B_T, compute_end > write_start)
        q_desc, k_desc, g_desc, do_desc = descs
        for r in cutlass.range(compute_end - write_start, unroll=1):
            chunk = compute_end - 1 - r
            rows = (start + chunk * B_T, cutlass.min(seqlen - chunk * B_T, B_T))
            with raw.produce() as h:
                h.access("Q", write=True, by="tma")
                h.access("K", write=True, by="tma")
                h.access("Gate", write=True, by="tma")
                qk_stage = h.index * self.qk_stage_bytes
                for kb in cutlass.range_constexpr(self.d_k // 64):
                    load_rows(
                        params.q_load,
                        self.io_dtype,
                        params.s_q + qk_stage + kb * (B_T * 128),
                        params.box_io,
                        rows,
                        head_q,
                        kb,
                        64,
                        h.barrier,
                        q_desc,
                    )
                    load_rows(
                        params.k_load,
                        self.io_dtype,
                        params.s_k + qk_stage + kb * (B_T * 128),
                        params.box_io,
                        rows,
                        head_k,
                        kb,
                        64,
                        h.barrier,
                        k_desc,
                    )
                g_stage = h.index * (self.d_k * B_T * 4)
                for kb in cutlass.range_constexpr(self.d_k // 32):
                    load_rows(
                        params.g_load,
                        Float32,
                        params.s_g + g_stage + kb * (B_T * 128),
                        params.box_g,
                        rows,
                        head,
                        kb,
                        32,
                        h.barrier,
                        g_desc,
                    )
            with dout.produce() as h:
                h.access(write=True, by="tma")
                do_stage = h.index * self.do_stage_bytes
                for kb in cutlass.range_constexpr(self.d_v // 64):
                    load_rows(
                        params.do_load,
                        self.io_dtype,
                        params.s_do + do_stage + kb * (B_T * 128),
                        params.box_io,
                        rows,
                        head,
                        kb,
                        64,
                        h.barrier,
                        do_desc,
                    )
            with beta.produce() as bh:
                if lane < B_T:
                    bh.data[lane] = carry
            carry = self._beta(params, head, start, seqlen, chunk - 1, lane)
        return raw, dout, beta

    @cute.jit
    def _beta(self, params: KdaBpropSummaryParams, head, start, seqlen, chunk, lane):
        """This lane's beta of ``chunk`` (zero past the sequence end or before its start)."""
        token = chunk * B_T + lane
        value = Float32(0.0)
        if lane < B_T and token >= 0 and token < seqlen:
            value = params.beta[start + token, head].to(Float32)
        return value

    # ---- compute group 0 (warps 0-3): gate scan and decay operands ------------------------

    @role(warps=range(4), regs=144)
    @cute.jit
    def cg0(
        self,
        params: KdaBpropSummaryParams,
        *,
        raw: Consumer,
        kk: Producer,
        qk: Producer,
        dscale: Producer,
        work: WorkCursor,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        channel_rows = self.d_k // 4
        if cutlass.const_expr(channel_rows < 32):
            channel = warp * channel_rows + lane % channel_rows
        else:
            channel = warp * 32 + lane
        segment = channel // 32
        seg_base = segment * (B_T * 32)
        col = channel - segment * 32
        # fp32 gate rows as TMA lays them out, and the in-place exp2(g) exchange (row ^ segment
        # swizzle, upstream's layout).
        gate_offsets = [opaque_i32(seg_base + swz128(j, col, 4)) for j in range(8)]
        prefix_offsets = [opaque_i32(seg_base + swz128(j ^ segment, col, 4)) for j in range(8)]
        geo = (lane, channel, segment, seg_base, col, gate_offsets, prefix_offsets)
        while work.valid:
            _seq, _head, write_start, compute_end, start, end = work.item
            seqlen = end - start
            for r in cutlass.range(compute_end - write_start, unroll=1):
                raw, kk, qk, dscale = self.cg0_chunk(
                    params, raw, kk, qk, dscale, (compute_end - 1 - r) * B_T, seqlen, warp, geo
                )
            work.advance()
        return raw, kk, qk, dscale, work

    @cute.jit
    def cg0_chunk(self, params, raw, kk, qk, dscale, chunk_start, seqlen, warp, geo):
        lane, channel, segment, seg_base, col, gate_offsets, prefix_offsets = geo
        io = self.io_dtype
        d_k = self.d_k
        dk_halves = d_k // 64
        channel_rows = d_k // 4
        store_rows = B_T * channel_rows // 32
        store_row_base = (lane // channel_rows) * store_rows
        decay_row = warp * 4 + lane // 8
        lane_in_row = lane - (lane // 8) * 8
        rh = raw.wait()
        rh.access("Q", read=True, by="threads")
        rh.access("K", read=True, by="threads")
        rh.access("Gate", read=True, write=True, by="threads")
        s_gate = params.s_g + rh.index * (d_k * B_T * 4)
        # ---- gate prefix scan (log2 domain), exp2 -------------------------------------------
        g = cute.make_rmem_tensor((B_T,), Float32)
        for row in cutlass.range_constexpr(B_T):
            g[row] = lds_f32(s_gate + 4 * (gate_offsets[row % 8] + row * 32)) * Float32(LOG2_E)
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
        # exp2(g) in place over this raw gate stage: a channel's stores land in slots that only
        # its own warp read, and the warp sync orders every lane's reads before them.
        cute.arch.sync_warp()
        for row_off in cutlass.range_constexpr(store_rows):
            if cutlass.const_expr(channel_rows < 32):
                row = store_row_base + row_off
                value = g[row_off + B_T // 2] if lane >= channel_rows else g[row_off]
                idx = seg_base + swz128(row ^ segment, row * 32 + col, 4)
            else:
                value = g[row_off]
                idx = prefix_offsets[row_off % 8] + row_off * 32
            sts_f32(s_gate + 4 * idx, value)
        # ---- decay scale exp2(g_last) per key channel -> CG1 ---------------------------------
        with dscale.produce() as dh:
            if lane < channel_rows:
                sts_f32(dh.data.iterator.toint() + 4 * channel, g[B_T - 1])
        self.exchange.sync(aligned=True)

        # ---- raw Q / K rows (zeroed past the sequence end), exp2(g) of the row ---------------
        qk_stage = rh.index * self.qk_stage_bytes
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
                exp_g += lds_v4_f32(s_gate + 4 * idx)
        rh.release()

        # ---- K decay / K inv ----------------------------------------------------------------
        kd_pack = []
        ki_pack = []
        for i in cutlass.range_constexpr(dk_halves * 4):
            kd0, kd1 = fmul2(raw_k[2 * i], raw_k[2 * i + 1], exp_g[2 * i], exp_g[2 * i + 1])
            kd_pack.append(fp32_to_fp16(kd0, kd1, dtype=io))
            inv0 = cute.math.rcp(exp_g[2 * i], approx=True, ftz=True)
            inv1 = cute.math.rcp(exp_g[2 * i + 1], approx=True, ftz=True)
            ki0, ki1 = fmul2(raw_k[2 * i], raw_k[2 * i + 1], inv0, inv1)
            ki_pack.append(fp32_to_fp16(ki0, ki1, dtype=io))
        with kk.produce() as kh:
            kh.access("Kdecay", write=True, by="threads")
            kh.access("Kinv", write=True, by="threads")
            decay_stage = kh.index * self.qk_stage_bytes
            for half in cutlass.range_constexpr(dk_halves):
                off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
                st_shared_v4(params.s_kinv + off, ki_pack[4 * half : 4 * half + 4])
                st_shared_v4(params.s_kdecay + off, kd_pack[4 * half : 4 * half + 4])

        # ---- Q decay = (scale Q) exp2(g) ------------------------------------------------------
        scale = params.scale
        qd_pack = []
        for i in cutlass.range_constexpr(dk_halves * 4):
            q0, q1 = fmul2(raw_q[2 * i], raw_q[2 * i + 1], scale, scale)
            qd0, qd1 = fmul2(q0, q1, exp_g[2 * i], exp_g[2 * i + 1])
            qd_pack.append(fp32_to_fp16(qd0, qd1, dtype=io))
        with qk.produce() as qh:
            qh.access(write=True, by="threads")
            decay_stage = qh.index * self.qk_stage_bytes
            for half in cutlass.range_constexpr(dk_halves):
                off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
                st_shared_v4(params.s_qdecay + off, qd_pack[4 * half : 4 * half + 4])
        return raw, kk, qk, dscale

    # ---- register MMA (warp 8): KK, L, T ----------------------------------------------------

    @role(warps=8, regs=56)
    @cute.jit
    def regmma(
        self,
        params: KdaBpropSummaryParams,
        *,
        kk: Consumer,
        beta: Consumer,
        tinv: Producer,
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
            _seq, _head, write_start, compute_end, _start, _end = work.item
            for _ in cutlass.range(compute_end - write_start, unroll=1):
                kh = kk.wait()
                kh.access("Kdecay", read=True, by="threads")
                kh.access("Kinv", read=True, by="threads")
                stage = kh.index * self.qk_stage_bytes
                acc = cute.make_rmem_tensor((8,), Float32)
                for i in cutlass.range_constexpr(8):
                    acc[i] = Float32(0.0)
                for i in cutlass.range_constexpr(self.d_k // 16):
                    box = (i // 4) * (B_T * 64)
                    kif = ldmatrix(params.s_kinv + stage + 2 * (ki_offs[i % 4] + box), 4)
                    kdf = ldmatrix(params.s_kdecay + stage + 2 * (kd_offs[i % 4] + box), 4)
                    mma_step(acc, kdf, kif, k_step=0, M=16, N=16, dtype=io)
                bh = beta.wait()
                s_beta = bh.data.iterator.toint()
                beta_lo = lds_f32(s_beta + 4 * row_lo)
                beta_hi = lds_f32(s_beta + 4 * row_hi)
                bh.release()
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
                t = invert_unit_lower_16x16_fragments(l_regs, lane, io)
                with tinv.produce() as th:
                    th.access(write=True, by="threads")
                    stmatrix(
                        params.s_tinv + th.index * (B_T * B_T * 2) + 2 * t_idx,
                        [fp32_to_fp16(t[2 * i], t[2 * i + 1], dtype=io) for i in range(4)],
                    )
                kh.release()
            work.advance()
        return kk, beta, tinv, work

    # ---- epilogue (warp 11): A = lower(Q_decay K_inv^T) -------------------------------------

    @role(warps=11, regs=56)
    @cute.jit
    def epilogue(
        self,
        params: KdaBpropSummaryParams,
        *,
        kk: Consumer,
        qk: Consumer,
        a: Producer,
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
            _seq, _head, write_start, compute_end, _start, _end = work.item
            for _ in cutlass.range(compute_end - write_start, unroll=1):
                qh = qk.wait()
                kh = kk.wait()
                qh.access(read=True, by="threads")
                kh.access("Kinv", read=True, by="threads")
                stage = qh.index * self.qk_stage_bytes
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
            work.advance()
        return kk, qk, a, work

    # ---- tcgen05 MMA (warp 9): every GEMM of the reverse recurrence --------------------------

    @role(warps=9, regs=56)
    @cute.jit
    def mma(
        self,
        params: KdaBpropSummaryParams,
        *,
        kk: Consumer,
        qk: Consumer,
        dout: Consumer,
        tinv: Consumer,
        a: Consumer,
        trip: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        dui_desc, dua_desc, qterm_desc, dy_desc, kterm_desc = self._descs()
        dstate_col = tmem.ptr("dstate").toint()
        dstate_in_col = tmem.ptr("dstate_in").toint()
        du_col = tmem.ptr("du").toint()
        dy_col = tmem.ptr("dy").toint()
        nbdy_col = tmem.ptr("nbdy").toint()
        du_in_col = tmem.ptr("du_in").toint()
        while work.valid:
            _seq, _head, write_start, compute_end, _start, _end = work.item
            n_chunks = compute_end - write_start
            if cutlass.const_expr(self.seeded):  # noqa: SIM102 (static, then dynamic)
                if n_chunks > 0:
                    th = trip.acquire()  # CG1 seeds D and D16 from d_final_state
                    th.commit()
            for r in cutlass.range(n_chunks, unroll=1):
                have = r > 0
                if cutlass.const_expr(self.seeded):
                    have = cutlass.Boolean(True)
                kh = kk.wait()
                kh.access("Kinv", read=True, by="umma")
                kh.access("Kdecay", read=True, by="umma")
                doh = dout.wait()
                doh.access(read=True, by="umma")
                kstage = kh.index * self.qk_stage_bytes
                # ---- dU = D16 @ K_inv^T: CG1 staged D16 and rescaled D -----------------------
                th = trip.acquire()
                tmem.access("du", write=True, by="umma")
                tmem.access("dstate", read=True, write=True, by="umma")
                if have:
                    tmem.access("dstate_in", read=True, by="umma")
                    desc_ki = desc_opaque(
                        smem_desc(params.s_kinv + kstage, 16, 8 * 128, SWIZZLE_128B)
                    )
                    mma_ts(dui_desc, dstate_in_col, desc_ki, du_col, False)
                # ---- dU (+)= dO^T @ A ----------------------------------------------------------
                ah = a.wait()
                ah.access(read=True, by="umma")
                do_addr = params.s_do + doh.index * self.do_stage_bytes
                desc_do = desc_opaque(smem_desc(do_addr, B_T * 128, 8 * 128, SWIZZLE_128B))
                a_addr = params.s_a + ah.index * (B_T * B_T * 2)
                desc_a = desc_opaque(smem_desc(a_addr, 16, 8 * B_T * 2, SWIZZLE_32B))
                mma_ss(dua_desc, desc_do, desc_a, du_col, have)
                th.commit()
                ah.release()
                # ---- D (+)= dO^T @ Q_decay -----------------------------------------------------
                qh = qk.wait()
                qh.access(read=True, by="umma")
                qd_addr = params.s_qdecay + qh.index * self.qk_stage_bytes
                desc_qd = desc_opaque(smem_desc(qd_addr, B_T * 128, 8 * 128, SWIZZLE_128B))
                mma_ss(qterm_desc, desc_do, desc_qd, dstate_col, have)
                qh.release()
                doh.release()
                # ---- dY = dU16 @ T: CG1 staged dU16 ----------------------------------------------
                tih = tinv.wait()
                tih.access(read=True, by="umma")
                th = trip.acquire()
                tmem.access("du_in", read=True, by="umma")
                tmem.access("dy", write=True, by="umma")
                t_addr = params.s_tinv + tih.index * (B_T * B_T * 2)
                desc_t = desc_opaque(smem_desc(t_addr, 16, 8 * B_T * 2, SWIZZLE_32B))
                mma_ts(dy_desc, du_in_col, desc_t, dy_col, False)
                th.commit()
                tih.release()
                # ---- D += (-beta dY)16 @ K_decay: CG1 staged (-beta dY)16 ----------------------
                th = trip.acquire()
                tmem.access("nbdy", read=True, by="umma")
                tmem.access("dstate", read=True, write=True, by="umma")
                desc_kd = desc_opaque(
                    smem_desc(params.s_kdecay + kstage, B_T * 128, 8 * 128, SWIZZLE_128B)
                )
                mma_ts(kterm_desc, nbdy_col, desc_kd, dstate_col, True)
                th.commit()
                kh.release()
            work.advance()
        return kk, qk, dout, tinv, a, trip, work

    # ---- compute group 1 (warps 4-7): seed, dU16, (-beta dY)16, D rescale, store ---------

    @role(warps=range(4, 8), regs=168)
    @cute.jit
    def cg1(
        self,
        params: KdaBpropSummaryParams,
        *,
        beta: Consumer,
        dscale: Consumer,
        trip: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        io = self.io_dtype
        lane, warp = me.lane, me.warp
        if cutlass.const_expr(self.d_v == 128):
            value_dim = warp * 32 + lane
            row_valid = cutlass.Boolean(True)
        else:
            # M = 64 GEMMs fill TMEM lanes 0-15 of each warp's 32.
            value_dim = warp * 16 + lane % 16
            row_valid = lane < 16
        lane0 = (32 * (me.tmem_row // 32)) << 16
        hi = 16 << 16  # the upper 16 lanes of this warp's 32
        cols = {
            name: lane0 + tmem.ptr(name).toint()
            for name in ("dstate", "dstate_in", "du", "dy", "nbdy", "du_in")
        }
        beta_col = (lane % 4) * 2
        out = params.d_initial_state
        while work.valid:
            seq, head, write_start, compute_end, start, end = work.item
            n_chunks = compute_end - write_start
            if n_chunks > 0:
                if cutlass.const_expr(self.seeded):
                    th = trip.wait()
                    tmem.access("dstate", write=True)
                    tmem.access("dstate_in", write=True)
                    with dscale.consume() as dh:
                        # Only an item ending at the sequence end carries the exit cotangent.
                        seed = compute_end == (end - start + B_T - 1) // B_T
                        self.seed(params, dh, cols, seq, head, value_dim, seed)
                    th.release()
                else:
                    with dscale.consume():
                        pass  # the last chunk's exp2(g_last) rescales nothing
            for r in cutlass.range(n_chunks, unroll=1):
                # ---- dU -> dU16, the dY GEMM's A operand ----------------------------------------
                th = trip.wait()
                tmem.access("du", read=True)
                tmem.access("du_in", write=True)
                du_lo = tmem_ld("16x256b", 2, cols["du"])
                if cutlass.const_expr(self.d_v == 128):
                    du_hi = tmem_ld("16x256b", 2, cols["du"] + hi)
                cute.arch.fence_view_async_tmem_load()
                tmem_st(
                    "16x128b",
                    cols["du_in"],
                    [fp32_to_fp16(du_lo[2 * i], du_lo[2 * i + 1], dtype=io) for i in range(4)],
                )
                if cutlass.const_expr(self.d_v == 128):
                    tmem_st(
                        "16x128b",
                        cols["du_in"] + hi,
                        [fp32_to_fp16(du_hi[2 * i], du_hi[2 * i + 1], dtype=io) for i in range(4)],
                    )
                th.release()
                # ---- dY -> (-beta dY)16, the K term's A operand -----------------------------------
                th = trip.wait()
                tmem.access("dy", read=True)
                tmem.access("nbdy", write=True)
                dy_lo = tmem_ld("16x256b", 2, cols["dy"])
                if cutlass.const_expr(self.d_v == 128):
                    dy_hi = tmem_ld("16x256b", 2, cols["dy"] + hi)
                cute.arch.fence_view_async_tmem_load()
                bh = beta.wait()
                s_beta = bh.data.iterator.toint()
                betas = [lds_f32(s_beta + 4 * (beta_col + c)) for c in (0, 1, 8, 9)]
                bh.release()
                self.stage_neg_beta_dy(cols["nbdy"], dy_lo, betas)
                if cutlass.const_expr(self.d_v == 128):
                    self.stage_neg_beta_dy(cols["nbdy"] + hi, dy_hi, betas)
                th.release()
                # ---- D: rescale and D16 for the next chunk, or store C ------------------------------
                th = trip.wait()
                tmem.access("dstate", read=True, write=True)
                if r + 1 < n_chunks:
                    tmem.access("dstate_in", write=True)
                    with dscale.consume() as dh:
                        self.rescale(dh, cols)
                elif write_start == 0:
                    self.store(out, cols, seq, head, value_dim, row_valid)
                th.release()
            if n_chunks == 0 and row_valid:
                # An item without chunks passes the exit cotangent through (cold path).
                for kd in cutlass.range(self.d_k, unroll=1):
                    value = Float32(0.0)
                    if cutlass.const_expr(self.seeded):
                        value = params.d_final_state[seq, head, value_dim, kd].to(Float32)
                    out[seq, head, value_dim, kd] = value
            work.advance()
        return beta, dscale, trip, work

    @cute.jit
    def seed(self, params, dh, cols, seq, head, value_dim, seed):
        """``d_final_state`` row -> D (decayed by the item's last chunk's exp2(g_last)) and its
        16-bit copy D16."""
        st = params.d_final_state
        src = (st.iterator + st.layout((seq, head, value_dim, 0))).toint()
        s_scale = dh.data.iterator.toint()
        for i in cutlass.range(self.d_k // 16, unroll=1):
            vals = load_state_16(src + i * 64, Float32)
            vals = [cutlass.select_(seed, v, Float32(0.0)) for v in vals]
            for grp in cutlass.range_constexpr(4):
                f = lds_v4_f32(s_scale + 4 * (i * 16 + grp * 4))
                for t in cutlass.range_constexpr(2):
                    k = grp * 4 + 2 * t
                    vals[k], vals[k + 1] = fmul2(vals[k], vals[k + 1], f[2 * t], f[2 * t + 1])
            tmem_st("32x32b", cols["dstate"] + i * 16, vals)
            words = [
                fp32_to_fp16(vals[2 * p], vals[2 * p + 1], dtype=self.io_dtype) for p in range(8)
            ]
            tmem_st("32x32b", cols["dstate_in"] + i * 8, words)

    @cute.jit
    def stage_neg_beta_dy(self, col, dy, betas):
        """``-(beta dY)`` of 16 TMEM lanes x 16 tokens -> its packed 16-bit operand at ``col``."""
        b0, b1, b8, b9 = betas
        words = []
        for e2 in cutlass.range_constexpr(4):
            e = 2 * e2
            lo, hi = (b8, b9) if cutlass.const_expr(e >= 4) else (b0, b1)
            s0, s1 = fmul2(dy[e], dy[e + 1], lo, hi)
            words.append(fp32_to_fp16(-s0, -s1, dtype=self.io_dtype))
        tmem_st("16x128b", col, words)

    @cute.jit
    def rescale(self, dh, cols):
        """``D *= exp2(g_last)`` of the next chunk, and D16 = 16-bit(D) for its dU GEMM."""
        s_scale = dh.data.iterator.toint()
        for i in cutlass.range(self.d_k // 32, unroll=1):
            vec = tmem_ld("32x32b", 32, cols["dstate"] + i * 32)
            cute.arch.fence_view_async_tmem_load()
            scaled = []
            for grp in cutlass.range_constexpr(8):
                f = lds_v4_f32(s_scale + 4 * (i * 32 + grp * 4))
                for t in cutlass.range_constexpr(2):
                    k = grp * 4 + 2 * t
                    s0, s1 = fmul2(vec[k], vec[k + 1], f[2 * t], f[2 * t + 1])
                    scaled += [s0, s1]
            words = [
                fp32_to_fp16(scaled[2 * p], scaled[2 * p + 1], dtype=self.io_dtype)
                for p in range(16)
            ]
            tmem_st("32x32b", cols["dstate_in"] + i * 16, words)
            tmem_st("32x32b", cols["dstate"] + i * 32, scaled)

    @cute.jit
    def store(self, out, cols, seq, head, value_dim, row_valid):
        """C: TMEM -> ``d_initial_state[seq, head, value_dim]``."""
        row_addr = (out.iterator + out.layout((seq, head, value_dim, 0))).toint()
        for i in cutlass.range_constexpr(self.d_k // 32):
            vals = tmem_ld("32x32b", 32, cols["dstate"] + i * 32)
            cute.arch.fence_view_async_tmem_load()
            if row_valid:
                for g in cutlass.range_constexpr(8):
                    st_global_v4_f32(row_addr + (i * 32 + g * 4) * 4, *vals[g * 4 : g * 4 + 4])

    # ---- kernel and entry point ---------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        q_load,
        k_load,
        g_load,
        do_load,
        box_io,
        box_g,
        beta,
        d_initial_state,
        d_final_state,
        q_ratio,
        k_ratio,
        scale,
        layouts,
        bound,
    ):
        # The work table and the preceding launches' outputs.
        cute.arch.griddepcontrol_wait()
        self.begin_roles(work=bound, layouts=layouts)
        addr = {name: self.buffer(name).iterator.toint() for name in self.buf}
        params = KdaBpropSummaryParams(
            q_load=q_load,
            k_load=k_load,
            g_load=g_load,
            do_load=do_load,
            box_io=box_io,
            box_g=box_g,
            beta=beta,
            d_initial_state=d_initial_state,
            d_final_state=d_final_state,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
            scale=scale,
            s_q=addr["Q"],
            s_k=addr["K"],
            s_g=addr["Gate"],
            s_do=addr["dO"],
            s_kdecay=addr["Kdecay"],
            s_kinv=addr["Kinv"],
            s_qdecay=addr["Qdecay"],
            s_a=addr["A"],
            s_tinv=addr["Tinv"],
        )
        self.run_roles(params)

    @cute.jit
    def launch(
        self,
        q: cute.Tensor,  # (tokens, q heads, d_k)
        k: cute.Tensor,  # (tokens, k heads, d_k)
        gate: cute.Tensor,  # (tokens, heads, d_k) fp32 natural-log decay
        do: cute.Tensor,  # (tokens, heads, d_v)
        beta: cute.Tensor,  # (tokens, heads) fp32
        d_initial_state: cute.Tensor,  # (sequences, heads, d_v, d_k) fp32
        d_final_state: cute.Tensor | None,  # (sequences, heads, d_v, d_k) fp32 when seeded
        work_items: cute.Tensor,  # (capacity, WORK_ITEM_FIELDS) int32
        work_count: cute.Tensor,  # (1,) int32 rows to run
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        scale: Float32,
        stream: cuda.CUstream,
    ):
        """Launch one CTA per work-table row (the first ``work_count[0]`` run); callable from
        another ``@cute.jit`` host, such as a fused launch bundle. It is a programmatic dependent
        launch: every thread waits for the preceding kernel before its roles start."""
        if cutlass.const_expr(d_initial_state.element_type is not Float32):
            raise TypeError("the KDA bprop summary writes fp32 state gradients")
        if cutlass.const_expr(gate.element_type is not Float32):
            raise TypeError("the KDA bprop summary takes an fp32 gate")
        if cutlass.const_expr((d_final_state is not None) != self.seeded):
            raise ValueError("d_final_state must be given exactly when the op is seeded")
        io = self.io_dtype
        heads = Int32(gate.shape[1])
        q_ratio = heads // Int32(q.shape[1])
        k_ratio = heads // Int32(k.shape[1])
        box_io = cute.select(
            sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1), mode=[0, 1]
        )
        box_g = cute.select(
            sm100_utils.make_smem_layout_epi(Float32, LayoutEnum.ROW_MAJOR, (B_T, 32), 1),
            mode=[0, 1],
        )

        def tdh(t):  # (token, head, dim) -> (token, dim, head)
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        def ragged(op, t, box, cols, index):  # clipped at each sequence end, keyed by head
            slots = slot_region(tma_slots, index, B_T, heads)
            return make_ragged_tma(op, tdh(t), box, B_T, cols, slots)

        q_load = ragged(g2s, q, box_io, 64, 0)
        k_load = ragged(g2s, k, box_io, 64, 1)
        g_load = ragged(g2s, gate, box_g, 32, 2)
        do_load = ragged(g2s, do, box_io, 64, 3)
        d_k, d_v = self.d_k, self.d_v
        layouts = {
            "Kdecay": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Kinv": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Qdecay": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "A": cute.make_layout((B_T * B_T, INTER_STAGES)),
            "Tinv": cute.make_layout((B_T * B_T, INTER_STAGES)),
            "dO": cute.make_layout((d_v * B_T, RAW_STAGES)),
            "Q": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "K": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "Gate": cute.make_layout((d_k * B_T, RAW_STAGES)),
        }
        count = cute.make_tensor(work_count.iterator, cute.make_layout(1))
        bound = self.work.bind((work_items, count), limit=LastEntry(count))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            q_load,
            k_load,
            g_load,
            do_load,
            box_io,
            box_g,
            beta,
            d_initial_state,
            d_final_state,
            q_ratio,
            k_ratio,
            scale,
            layouts,
            bound,
        ).launch(**self.launch_config(bound), stream=stream, use_pdl=True)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        gate: cute.Tensor,
        do: cute.Tensor,
        beta: cute.Tensor,
        d_initial_state: cute.Tensor,
        d_final_state: cute.Tensor | None,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        tma_slots: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        self.launch(
            q,
            k,
            gate,
            do,
            beta,
            d_initial_state,
            d_final_state,
            work_items,
            work_count,
            tma_slots,
            scale,
            stream,
        )


def bundle_hook(io_dtype, gate_dtype, d_k: int, d_v: int, log_gate: bool) -> KdaBpropSummary:
    """The zero-seeded op for launch bundles that run it inside their own ``@cute.jit`` host:
    ``op.launch(q, k, gate, do, beta, d_initial_state, None, work_items, work_count, tma_slots,
    scale, stream)``."""
    if not log_gate or gate_dtype is not Float32:
        raise ValueError("the KDA bprop summary takes fp32 natural-log gates")
    names = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}
    return KdaBpropSummary(names[io_dtype], d_k, d_v, seeded=False)


@jit_cache
def compile_kda_bprop_summary(op: KdaBpropSummary):
    """Compile for dynamic tokens, heads, sequences and work-table capacity."""
    io, i32 = op.io_dtype, cutlass.Int32
    tokens, hq, hk, heads, seqs = (cute.sym_int() for _ in range(5))

    def packed(dtype, h, d):
        return make_fake_strided_tensor(dtype, (tokens, h, d), stride_divisibility=8)

    def state():
        return make_fake_strided_tensor(
            Float32, (seqs, heads, op.d_v, op.d_k), stride_divisibility=4
        )

    return compile_tvm_ffi(
        op,
        packed(io, hq, op.d_k),
        packed(io, hk, op.d_k),
        packed(Float32, heads, op.d_k),
        packed(io, heads, op.d_v),
        make_fake_strided_tensor(Float32, (tokens, heads), contiguous_dim=None),
        state(),
        state() if op.seeded else None,
        make_fake_strided_tensor(i32, (cute.sym_int(), WORK_ITEM_FIELDS), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        fake_tma_slots(),
        Float32(1.0),
        name=op.get_name(),
    )


def kda_bprop_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    d_initial_state: torch.Tensor,
    work_items: torch.Tensor,
    work_count: torch.Tensor,
    *,
    scale: float,
    d_final_state: torch.Tensor | None = None,
) -> None:
    """Fill ``d_initial_state`` (``[N, H, d_v, d_k]`` fp32, row-contiguous) for the work table
    (see the module docstring); packed ``[T, H, D]`` operands, fp32 natural-log gate
    ``[T, H, d_k]`` and beta ``[T, H]``; ``d_final_state`` seeds every item's exit cotangent."""
    if q.dtype not in _IO_NAMES or {k.dtype, do.dtype} != {q.dtype}:
        raise TypeError(f"q, k and do must share a 16-bit dtype, got {q.dtype}/{k.dtype}")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("gate and beta must be fp32")
    states = [t for t in (d_initial_state, d_final_state) if t is not None]
    if any(t.dtype != torch.float32 for t in states):
        raise TypeError("the state gradients are fp32")
    if any(t.stride(-1) != 1 or t.stride(-2) % 4 or t.data_ptr() % 16 for t in states):
        raise ValueError("the state gradients need contiguous, 16-byte aligned rows")
    heads = gate.shape[1]
    if heads % q.shape[1] or heads % k.shape[1] or do.shape[1] != heads:
        raise ValueError("heads must match dO and be a multiple of the query and key head counts")
    op = KdaBpropSummary(_IO_NAMES[q.dtype], q.shape[-1], do.shape[-1], d_final_state is not None)
    slots = new_tma_slots(op.tma_slot_rows(heads), q.device)
    compile_kda_bprop_summary(op)(
        q, k, gate, do, beta, d_initial_state, d_final_state, work_items, work_count, slots, scale
    )
