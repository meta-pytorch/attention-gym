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

The kernel visits chunks from last to first. Here ``g`` is the in-chunk cumulative log2 gate
per key channel, ``D`` the FP32 state-gradient accumulator in tensor memory (value × key) and
``io`` the Q/K/V dtype::

    K_decay = K exp2(g)   K_inv = K exp2(-g)   K_restore = K exp2(g_last - g)   Q_decay = Q exp2(g)
    L   = beta * strict_lower(K_decay K_inv^T)        T = (I + L)^-1  (16x16 blockwise, mma.sync)
    A   = lower(Q_decay K_inv^T)
    dU  = D16 @ K_restore^T + scale (dO^T @ A)    (no D16 term on a zero-seeded last chunk)
    D  += io(scale dO)^T @ Q_decay                 (overwrite on a zero-seeded last chunk)
    dY  = solve_dtype(dU) @ T
    D  += io(-beta dY) @ K_decay
    D16 = io(D);  D *= exp2(g_last of chunk c - 1)                (the next chunk's operands)

All precisions use the per-tile forward's unscaled Q_decay and A. The state contribution to
dU uses the forward's rounded K_restore and D16 cast before decay, so a small decay cannot
round the state gradient to zero before a large inverse decay cancels it. With TF32 decay,
KK and A use TF32 operands. With a TF32 solve, stage dU in TF32 for the multiply by the inverse.
FP16 splits the Q term's scale as in ``kda_bprop`` (``q_state_scales``).

A seeded item starts with ``D16 = io(dS_end)`` and ``D = exp2(g_last) dS_end``. Items are the first
``work_count[0]`` rows of a split-K work table, ``(sequence, head, write_start, -, -,
compute_end, start, end, ...)``: chunks ``compute_end - 1`` down to ``write_start`` of the
sequence at tokens ``[start, end)``. ``C`` lands at ``d_initial_state[sequence, head]`` when
``write_start == 0``; an item without chunks writes ``dS_end`` (zero unseeded).

Q/K/gate/dO rows past a sequence end load as zeros (``ragged.RaggedTma``), as from
upstream's per-sequence descriptors; beta is zero there.

Roles (12 warps; upstream's idle warps 8-11 are not launched) and their handoffs::

    load (10)     --raw:   2 x Q, K, Gate [16, d_k] (TMA)-->        cg0 (0-3)
                  --dout:  2 x dO [16, d_v] (TMA)-->                mma (9), cg1 (see below)
                  --beta:  4 x beta[16]-->                          regmma (8), cg1 (4-7)
    cg0           --kk:    2 x K_decay, K_inv, K_restore-->         mma, regmma, epilogue (11)
                  --qk:    2 x Q_decay-->                           mma, epilogue
                  --dscale: 2 x exp2(g_last)[d_k]-->                cg1
    regmma        --tinv:  2 x T [16, 16]-->                        mma
    epilogue      --a:     2 x A [16, 16]-->                        mma
    mma          <-trip:   1 (UMMA / threads)->                     cg1

The ``trip`` channel orders MMA reads and writes against compute group 1 (CG1). Each chunk
exchanges dU, dY and the updated state gradient D. A TF32 solve adds a handoff to stage dO
in tensor memory before the dU product. CG1 scales dO in place after that product and before
MMA reads it again for the Q term. The load warp cannot refill ``dout`` until that second read
completes. A seeded item also opens with a handoff for its exit cotangent. The last reader
releases each of the other SMEM rings.
One CTA runs one work-table row (``OneShot`` over the capacity, limited to the
device-written count) instead of upstream's atomic ticket scheduler.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.linear._delta_rule.wd_support import HEAD_DIMS
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

from .kda_bprop import q_state_scales, scale_du, scale_packed, stage_scaled_do
from .kda_common import (
    B_T,
    TF32_TILE_BYTES,
    WORK_ITEM_FIELDS,
    OperandDtypes,
    box128,
    decay_fragment_offsets,
    decay_stage_addresses,
    gate_offset,
    invert_solve,
    lds_f32,
    lds_v4,
    lds_v4_f32,
    load_rows,
    load_state,
    mma_decay_16x16,
    opaque_i32,
    operand_dtypes,
    precision_suffix,
    solve_mma,
    solve_mma_desc,
    solve_tile_desc,
    store_solve_operand,
    store_tf32_k_major,
    sts_f32,
    sts_tf32x8,
    swz32,
    swz128,
)
from .ptx import (
    f16x2_to_f32,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
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
    s_kinv: Int32  # K_inv occupies FP32 words with TF32 decay.
    s_qdecay: Int32
    s_a: Int32
    s_tinv: Int32


class KdaBpropSummary(WarpSpecialized):
    """Warpdrive KDA backward summaries for ``io`` fp16/bf16 operands, ``d_k``/``d_v`` 64 or 128,
    an fp32 natural-log gate and beta, and fp32 state gradients; ``seeded`` starts every item
    from ``d_final_state`` instead of zero.
    ``fastmath=False`` exponentiates the gates with the non-flushing ``exp2`` and reciprocal.
    The summary uses the forward's decay and solve dtypes (see module docstring)."""

    def __init__(
        self,
        io: str,
        d_k: int,
        d_v: int,
        seeded: bool,
        fastmath: bool = True,
        *,
        precision: OperandDtypes | None = None,
    ):
        if d_k not in HEAD_DIMS or d_v not in HEAD_DIMS:
            raise ValueError(
                f"the KDA bprop summary serves d_k, d_v in {HEAD_DIMS}, got {d_k}, {d_v}"
            )
        self.io, self.d_k, self.d_v, self.seeded = io, d_k, d_v, seeded
        self.fastmath = fastmath
        self.io_dtype = io_t = _CUTE_DTYPES[io]
        self.precision = precision if precision is not None else operand_dtypes(io, "bf16")
        self.tf32_decay = self.precision.decay_dtype == "tf32"
        self.tf32_solve = self.precision.solve_dtype == "tf32"
        self.split_scale = io == "fp16"  # as in ``kda_bprop``
        self.solve_t = Float32 if self.tf32_solve else io_t
        self.solve_stage_bytes = TF32_TILE_BYTES if self.tf32_solve else B_T * B_T * 2
        self.g_solve32 = solve_mma_desc(d_v, Float32)
        self.decay_t = Float32 if self.tf32_decay else io_t
        plan = self.smem_plan = SmemPlan()
        names = ("Kdecay", "Kinv", "Qdecay", "A", "Tinv", "dO", "Q", "K")
        dtype = {"Kinv": self.decay_t, "A": self.solve_t, "Tinv": self.solve_t}
        buf = {name: plan.buffer(name, dtype.get(name, io_t), align=1024) for name in names}
        buf["Gate"] = plan.buffer("Gate", Float32, align=1024)
        if self.tf32_decay:
            for name in ("Kdecay32", "Qdecay32"):
                buf[name] = plan.buffer(name, Float32, align=1024)
        buf["Krestore"] = plan.buffer("Krestore", io_t, align=1024)
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
            buffer={k: buf[k] for k in ("Kdecay", "Kinv", "Kdecay32", "Krestore") if k in buf},
        )
        self.qk = Ring(
            DECAY_STAGES,
            Warps,
            {"mma": Umma, "epilogue": Warps},
            buffer=(
                {k: buf[k] for k in ("Qdecay", "Qdecay32")} if self.tf32_decay else buf["Qdecay"]
            ),
        )
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
            .region("du_in", B_T if self.tf32_solve else B_T // 2)  # dO, then dU
        )
        self.work = WorkStream(mapping=SummaryItems(), schedule=OneShot())

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: Q, K, gate and dO, keyed by head."""
        return tma_slot_rows(4, B_T, heads)

    def get_name(self) -> str:
        name = f"wd_kda_bprop_summary_{self.io}_k{self.d_k}_v{self.d_v}_s{int(self.seeded)}"
        name += precision_suffix(self.io, self.precision)
        return name if self.fastmath else f"{name}_nofm"

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
            desc(B_T, d_k),  # state contribution to dU: D16 @ K_restore^T
            desc(B_T, B_T, atranspose=True, btranspose=True),  # output contribution before scaling
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
            g[row] = cute.math.exp2(g[row], fastmath=self.fastmath)
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
        # The last-token decay builds K_restore.
        dims = [
            half * 64 + lane_in_row * 8 + grp * 4 for half in range(dk_halves) for grp in (0, 1)
        ]
        exp_g = []
        exp_last = []
        for f_dim in dims:
            exp_g += lds_v4_f32(s_gate + gate_offset(decay_row, f_dim, d_k))
            exp_last += lds_v4_f32(s_gate + gate_offset(B_T - 1, f_dim, d_k))
        rh.release()

        # ---- K decay / K inv ----------------------------------------------------------------
        kd_pack = []
        ki_pack = []
        kr_pack = []
        kd32 = []  # FP32 decay operands before rounding to TF32.
        ki32 = []
        for i in cutlass.range_constexpr(dk_halves * 4):
            kd0, kd1 = fmul2(raw_k[2 * i], raw_k[2 * i + 1], exp_g[2 * i], exp_g[2 * i + 1])
            kd_pack.append(fp32_to_fp16(kd0, kd1, dtype=io))
            inv0 = cute.math.rcp(exp_g[2 * i], approx=True, ftz=self.fastmath)
            inv1 = cute.math.rcp(exp_g[2 * i + 1], approx=True, ftz=self.fastmath)
            ki0, ki1 = fmul2(raw_k[2 * i], raw_k[2 * i + 1], inv0, inv1)
            if cutlass.const_expr(self.tf32_decay):
                kd32 += [kd0, kd1]
                ki32 += [ki0, ki1]
            else:
                ki_pack.append(fp32_to_fp16(ki0, ki1, dtype=io))
            kr0, kr1 = fmul2(ki0, ki1, exp_last[2 * i], exp_last[2 * i + 1])
            kr_pack.append(fp32_to_fp16(kr0, kr1, dtype=io))
        with kk.produce() as kh:
            kh.access("Kdecay", write=True, by="threads")
            if cutlass.const_expr(self.tf32_decay):
                tiles32 = [
                    (kh.view(name, write=True).iterator.toint(), vals)
                    for name, vals in (("Kinv", ki32), ("Kdecay32", kd32))
                ]
            else:
                kh.access("Kinv", write=True, by="threads")
            s_kr = kh.view("Krestore", write=True).iterator.toint()
            decay_stage = kh.index * self.qk_stage_bytes
            for half in cutlass.range_constexpr(dk_halves):
                box = 2 * box128(decay_row, half * 64 + lane_in_row * 8)
                off = decay_stage + box
                if cutlass.const_expr(self.tf32_decay):
                    for base, vals in tiles32:
                        sts_tf32x8(
                            base,
                            decay_row,
                            half * 64 + lane_in_row * 8,
                            vals[8 * half : 8 * half + 8],
                            d_k,
                        )
                else:
                    st_shared_v4(params.s_kinv + off, ki_pack[4 * half : 4 * half + 4])
                st_shared_v4(s_kr + box, kr_pack[4 * half : 4 * half + 4])
                st_shared_v4(params.s_kdecay + off, kd_pack[4 * half : 4 * half + 4])

        # ---- Q decay = Q exp2(g) (scale applies to the output) ---------------------------------
        q_factor, _ = q_state_scales(params.scale, self.split_scale)
        qd_pack = []
        qd32 = []
        for i in cutlass.range_constexpr(dk_halves * 4):
            qd0, qd1 = fmul2(raw_q[2 * i], raw_q[2 * i + 1], exp_g[2 * i], exp_g[2 * i + 1])
            qd_pack.append(scale_packed(fp32_to_fp16(qd0, qd1, dtype=io), q_factor))
            qd32 += [qd0, qd1]
        with qk.produce() as qh:
            qh.access("Qdecay", write=True, by="threads")
            if cutlass.const_expr(self.tf32_decay):
                s_qd32 = qh.view("Qdecay32", write=True).iterator.toint()
            decay_stage = qh.index * self.qk_stage_bytes
            for half in cutlass.range_constexpr(dk_halves):
                off = decay_stage + 2 * box128(decay_row, half * 64 + lane_in_row * 8)
                st_shared_v4(params.s_qdecay + off, qd_pack[4 * half : 4 * half + 4])
                if cutlass.const_expr(self.tf32_decay):
                    sts_tf32x8(
                        s_qd32,
                        decay_row,
                        half * 64 + lane_in_row * 8,
                        qd32[8 * half : 8 * half + 8],
                        d_k,
                    )
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
        t_row = (lane & 7) + (8 if (lane // 8) & 1 else 0)
        t_col = 8 if lane // 8 >= 2 else 0
        t_idx = t_row * B_T + swz32(t_row, t_col)
        kd_offs, ki_offs = decay_fragment_offsets(lane, self.d_k, self.tf32_decay)
        row_lo = lane // 4
        row_hi = row_lo + 8
        while work.valid:
            _seq, _head, write_start, compute_end, _start, _end = work.item
            for _ in cutlass.range(compute_end - write_start, unroll=1):
                kh = kk.wait()
                acc = cute.make_rmem_tensor((8,), Float32)
                for i in cutlass.range_constexpr(8):
                    acc[i] = Float32(0.0)
                operands = decay_stage_addresses(
                    kh, "Kdecay", kh, params.s_kdecay, params.s_kinv, self.d_k, self.tf32_decay
                )
                mma_decay_16x16(acc, *operands, kd_offs, ki_offs, self.d_k, self.decay_t)
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
                t = invert_solve(l_regs, lane, self.solve_t)
                with tinv.produce() as th:
                    th.access(write=True, by="threads")
                    addr = params.s_tinv + th.index * self.solve_stage_bytes
                    if cutlass.const_expr(self.tf32_solve):
                        store_tf32_k_major(addr, t, lane, transpose=True)
                    else:
                        stmatrix(
                            addr + 2 * t_idx,
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
        a_row = (lane & 7) + (8 if (lane // 8) & 1 else 0)
        a_col = 8 if lane // 8 >= 2 else 0
        a_idx = a_row * B_T + swz32(a_row, a_col)
        qd_offs, ki_offs = decay_fragment_offsets(lane, self.d_k, self.tf32_decay)
        while work.valid:
            _seq, _head, write_start, compute_end, _start, _end = work.item
            for _ in cutlass.range(compute_end - write_start, unroll=1):
                qh = qk.wait()
                kh = kk.wait()
                acc = cute.make_rmem_tensor((8,), Float32)
                for i in cutlass.range_constexpr(8):
                    acc[i] = Float32(0.0)
                operands = decay_stage_addresses(
                    qh, "Qdecay", kh, params.s_qdecay, params.s_kinv, self.d_k, self.tf32_decay
                )
                mma_decay_16x16(acc, *operands, qd_offs, ki_offs, self.d_k, self.decay_t)
                for i in cutlass.range_constexpr(8):
                    r = lane // 4 + (8 if i % 4 >= 2 else 0)
                    c = (i // 4) * 8 + 2 * (lane % 4) + (i % 2)
                    acc[i] = acc[i] if r >= c else Float32(0.0)
                with a.produce() as ah:
                    ah.access(write=True, by="threads")
                    addr = params.s_a + ah.index * self.solve_stage_bytes
                    if cutlass.const_expr(self.tf32_solve):
                        store_tf32_k_major(addr, acc, lane, transpose=True)
                    else:
                        stmatrix(
                            addr + 2 * a_idx,
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
        me: Local,
    ):
        issue = me.lane == 0
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
                kh.access("Kdecay", read=True, by="umma")
                kstage = kh.index * self.qk_stage_bytes
                s_dui = kh.view("Krestore", read=True, by="umma").iterator.toint()
                doh = dout.wait()
                doh.access(read=True, by="umma")
                # ---- dU state term: CG1 staged D16 before rescaling D -------------------------
                th = trip.acquire()
                tmem.access("du", write=True, by="umma")
                tmem.access("dstate", read=True, write=True, by="umma")
                if have:
                    tmem.access("dstate_in", read=True, by="umma")
                    desc_ki = desc_opaque(smem_desc(s_dui, 16, 8 * 128, SWIZZLE_128B))
                    mma_ts(dui_desc, dstate_in_col, desc_ki, dy_col, False)
                # ---- dU output term: CG1 scales dO^T @ A after this product ---------------------
                ah = a.wait()
                ah.access(read=True, by="umma")
                do_addr = params.s_do + doh.index * self.do_stage_bytes
                desc_do = desc_opaque(smem_desc(do_addr, B_T * 128, 8 * 128, SWIZZLE_128B))
                a_addr = params.s_a + ah.index * self.solve_stage_bytes
                if cutlass.const_expr(self.tf32_solve):
                    th.commit()
                    th = trip.acquire()  # dO TF32 now in TMEM.
                    solve_mma(
                        self.g_solve32,
                        du_in_col,
                        solve_tile_desc(a_addr, Float32),
                        du_col,
                        False,
                        issue,
                    )
                else:
                    desc_a = desc_opaque(smem_desc(a_addr, 16, 8 * B_T * 2, SWIZZLE_32B))
                    mma_ss(dua_desc, desc_do, desc_a, du_col, False)
                th.commit()
                ah.release()
                th = trip.acquire()  # CG1 has scaled dO for the Q state term.
                # ---- D (+)= dO^T @ Q_decay -----------------------------------------------------
                qh = qk.wait()
                qh.access("Qdecay", read=True, by="umma")
                qd_addr = params.s_qdecay + qh.index * self.qk_stage_bytes
                desc_qd = desc_opaque(smem_desc(qd_addr, B_T * 128, 8 * 128, SWIZZLE_128B))
                mma_ss(qterm_desc, desc_do, desc_qd, dstate_col, have)
                qh.release()
                doh.release()
                # dY uses dU rounded to the solve dtype.
                tih = tinv.wait()
                tih.access(read=True, by="umma")
                tmem.access("du_in", read=True, by="umma")
                tmem.access("dy", write=True, by="umma")
                t_addr = params.s_tinv + tih.index * self.solve_stage_bytes
                if cutlass.const_expr(self.tf32_solve):
                    solve_mma(
                        self.g_solve32,
                        du_in_col,
                        solve_tile_desc(t_addr, Float32),
                        dy_col,
                        False,
                        issue,
                    )
                else:
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

    # ---- compute group 1 (warps 4-7): seed, stage dU and -beta dY, rescale and store D ----

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
        value_base = warp * (self.d_v // 4)
        ov_row = ((lane >> 4) << 3) + (lane & 7)
        ov_col = ((lane >> 3) & 1) << 3
        ov_lo = box128(ov_row, value_base + ov_col)
        ov_hi = box128(ov_row, value_base + 16 + ov_col)
        out = params.d_initial_state
        base = Int32(0)
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
                if cutlass.const_expr(self.tf32_solve):
                    with trip.consume() as th:
                        tmem.access("du_in", write=True)
                        addr = params.s_do + ((base + r) & 1) * self.do_stage_bytes
                        self.stage_dout(cols["du_in"], addr + 2 * ov_lo)
                        if cutlass.const_expr(self.d_v == 128):
                            self.stage_dout(cols["du_in"] + hi, addr + 2 * ov_hi)
                # Stage dU for the dY product.
                th = trip.wait()
                tmem.access("du", read=True)
                tmem.access("du_in", write=True)
                du_lo = tmem_ld("16x256b", 2, cols["du"])
                if cutlass.const_expr(self.d_v == 128):
                    du_hi = tmem_ld("16x256b", 2, cols["du"] + hi)
                have = cutlass.Boolean(True) if cutlass.const_expr(self.seeded) else r > 0
                du_lo = scale_du(du_lo, params.scale, cols["dy"], have)
                if cutlass.const_expr(self.d_v == 128):
                    du_hi = scale_du(du_hi, params.scale, cols["dy"] + hi, have)
                # Scale dO between the unscaled dU read and the Q-term read; trip orders both.
                addr = params.s_do + ((base + r) & 1) * self.do_stage_bytes
                _, do_scale = q_state_scales(params.scale, self.split_scale)
                stage_scaled_do(addr, addr, ov_lo, do_scale, io)
                if cutlass.const_expr(self.d_v == 128):
                    stage_scaled_do(addr, addr, ov_hi, do_scale, io)
                cute.arch.fence_view_async_shared()
                cute.arch.fence_view_async_tmem_load()
                store_solve_operand(((cols["du_in"], du_lo),), self.solve_t)
                if cutlass.const_expr(self.d_v == 128):
                    store_solve_operand(((cols["du_in"] + hi, du_hi),), self.solve_t)
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
            base += n_chunks
            work.advance()
        return beta, dscale, trip, work

    @cute.jit
    def stage_dout(self, col, addr):
        frag = []
        for word in ldmatrix(addr, 4, trans=True):
            frag += list(f16x2_to_f32(word, dtype=self.io_dtype))
        store_solve_operand(((col, frag),), Float32)

    @cute.jit
    def seed(self, params, dh, cols, seq, head, value_dim, seed):
        """Seed D16 from the unscaled exit cotangent and D from its decayed FP32 value."""
        src = params.d_final_state[seq, head, value_dim, None]
        s_scale = dh.data.iterator.toint()
        for i in cutlass.range(self.d_k // 16, unroll=1):
            vals = load_state(cute.local_tile(src, (16,), (i,)))
            vals = [cutlass.select_(seed, v, Float32(0.0)) for v in vals]
            words = [  # D16 = 16-bit(dS_end)
                fp32_to_fp16(vals[2 * p], vals[2 * p + 1], dtype=self.io_dtype) for p in range(8)
            ]
            for grp in cutlass.range_constexpr(4):
                f = lds_v4_f32(s_scale + 4 * (i * 16 + grp * 4))
                for t in cutlass.range_constexpr(2):
                    k = grp * 4 + 2 * t
                    vals[k], vals[k + 1] = fmul2(vals[k], vals[k + 1], f[2 * t], f[2 * t + 1])
            tmem_st("32x32b", cols["dstate"] + i * 16, vals)
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
        """Stage the dU operand before rescaling D for the next chunk."""
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
                fp32_to_fp16(vec[2 * p], vec[2 * p + 1], dtype=self.io_dtype) for p in range(16)
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
        layouts["Krestore"] = cute.make_layout((d_k * B_T, DECAY_STAGES))
        if cutlass.const_expr(self.tf32_decay):
            for name in ("Kdecay32", "Qdecay32"):
                layouts[name] = cute.make_layout((d_k * B_T, DECAY_STAGES))
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


def bundle_hook(
    io_dtype,
    gate_dtype,
    d_k: int,
    d_v: int,
    log_gate: bool,
    fastmath: bool = True,
    *,
    precision: OperandDtypes | None = None,
) -> KdaBpropSummary:
    """The zero-seeded op for launch bundles that run it inside their own ``@cute.jit`` host:
    ``op.launch(q, k, gate, do, beta, d_initial_state, None, work_items, work_count, tma_slots,
    scale, stream)``."""
    if not log_gate or gate_dtype is not Float32:
        raise ValueError("the KDA bprop summary takes fp32 natural-log gates")
    names = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}
    return KdaBpropSummary(
        names[io_dtype], d_k, d_v, seeded=False, fastmath=fastmath, precision=precision
    )


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
    fastmath: bool = True,
    intra_chunk_dtypes: str = "bf16",
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
    op = KdaBpropSummary(
        _IO_NAMES[q.dtype],
        q.shape[-1],
        do.shape[-1],
        d_final_state is not None,
        fastmath,
        precision=operand_dtypes(_IO_NAMES[q.dtype], intra_chunk_dtypes),
    )
    slots = new_tma_slots(op.tma_slot_rows(heads), q.device)
    compile_kda_bprop_summary(op)(
        q, k, gate, do, beta, d_initial_state, d_final_state, work_items, work_count, slots, scale
    )
