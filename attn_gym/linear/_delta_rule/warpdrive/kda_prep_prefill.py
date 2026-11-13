# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# kda_prep_prefill_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and
# channels with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence
# descriptor arrays, and the state checkpoints and full-width value tiles the driver never
# requests are gone.

"""Prep-fed KDA chunked prefill forward (BT = 16) for SM100: the state recurrence over the
per-chunk records of ``kda_prep``.

Per chunk of 16 tokens, beta and the chunk inverse already folded into the records (``S`` the
``[d_k, d_v]`` fp32 state carried across chunks)::

    k_decay, q_decay, T  [16, d_k]      A  [16, 16]      diag = exp2(g_last)  [d_k]
    Y = V - k_decay S                   O = scale (q_decay S + A Y)
    S <- diag .* S + T^T Y

The driver runs this kernel only under the d_v split, so value tiles are 64 columns wide
(``tiles_per_head`` tiles per gate head). The work table and the zeroed scheduler counter come
from the caller (``work_tables`` in the launch bundle, ``uncut_work_items`` standalone);
the records are indexed by chunk row ``sequence_start // 16 + sequence + chunk`` and head. CTA
``b`` starts at item ``b``; the load warp claims the next item after issuing an item's loads
and broadcasts it to every role.

V rows past a sequence end load as zeros (``ragged.RaggedTma``), as from upstream's
per-sequence descriptors (the records are zero there), and O rows past the end are clipped by
the ragged TMA store view.

Roles (15 of 16 warps; warp 12, upstream's scheduler-only warp, is the last warpgroup's
padding) and their handoffs::

    load (14)   --raw: 8 x V, diag (TMA)-->          cg0 (0-3), cg2 (8-11)
                --rec: 4 x k_decay, q_decay, T, A (TMA)-->  mma (13)
                --sched: 8 x next item-->            every other role
    mma         --o_acc: 2 x O accumulator-->       cg1 (4-7)
    cg1         --o_smem: 2 x O-->                  epilogue (15): TMA store

The TMEM state recurrence between the MMA warp and compute groups 0 and 2 (b16 state operand,
KS accumulator, Y operand, state-update completion) uses the upstream signal-only barriers
(``tstate``, ``TSTATE``): every stage they guard is reused only after a later handoff of the
same recurrence, so a channel's reuse wait would only lengthen the per-chunk critical path.
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
    Local,
    OneShot,
    PdlWait,
    Producer,
    Ring,
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

from .kda_common import (
    A_WORDS,
    B_T,
    CUTE_DTYPES,
    FINAL_DST,
    TORCH_NAMES,
    WORK_ITEM_FIELDS,
    as_i32,
    box128,
    lds_v2_f32,
    lds_v4_f32,
    load_state_16,
    mask_pair,
    opaque_i32,
    resolve_paged_state,
    spin_wait,
    stg_v4,
    umma_commit,
    warp_arrive,
)
from .kda_prefill import (
    uncut_work_items,
)
from .ptx import (
    elect_sync,
    f16x2_to_f32,
    fadd2,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    stmatrix,
    tmem_ld,
    tmem_st,
)
from .ragged import (
    RaggedTma,
    fake_tma_slots,
    make_ragged_tma,
    new_tma_slots,
    ragged_load,
    ragged_store,
    sequence_descriptors,
    slot_region,
    tma_slot_rows,
)
from .umma import SWIZZLE_32B, SWIZZLE_128B, MmaDesc, desc_opaque, instr_desc, mma_ts_op, smem_desc

D_V = 64  # value tile width: the driver only runs the prep scheme under the d_v split
DIMS = (64, 128)
RAW_STAGES = 8
REC_STAGES = 4
SCHED_STAGES = 8
O_STAGES = 2
QS_STAGES = 2

# TMEM state handoffs (``tstate`` barriers, upstream counts), each completing once per epoch:
#   SIN_L  left key half of the b16 state operand: CG0 (chunks after an item's first), CG2 (the
#          seed); MMA waits before KS k-steps [0, n/2). Reuse: rewritten after SACC_L.
#   SIN_R  right half, CG2; MMA waits before the rest of KS.
#   KS     KS accumulator, MMA commit; CG2 reads. KS(c+1) follows the UIN(c) wait that CG2
#          signals after reading KS(c).
#   UIN    Y operand (CG2) and the left-half state decay (CG0): 4 + 4 arrivals; MMA waits
#          before the state update. Rewritten after the O GEMM reads it: KS(c+1) completion
#          (CG2 waits it) or, at an item's first unseeded chunk, ODONE.
#   SACC_L / SACC_R (4 stages)  left / right state update committed; CG0 / CG2 read the state.
#   ODONE  the chunk's O GEMM completed (MMA commits every chunk; CG2 waits only for an item's
#          last, before the next item may rewrite the Y operand without a KS wait. Absent
#          upstream, where that rewrite can overtake the in-flight O GEMM).
SIN_L, SIN_R, KS, UIN, SACC_L, SACC_R, ODONE = 0, 1, 2, 3, 4, 8, 12
TSTATE_COUNTS = (4, 4, 1, 8, *(1,) * 8, 1)


@cute.jit
def _residual(v_word, ks0, ks1, dtype: cutlass.Constexpr):
    """``v - ks`` for one packed V pair in fp32; only the final pack rounds."""
    v0, v1 = f16x2_to_f32(v_word, dtype=dtype)
    y0, y1 = fadd2(v0, v1, -ks0, -ks1)
    return fp32_to_fp16(y0, y1, dtype=dtype)


@cute.jit
def _tma_box(atom, src, addr, layout, bar, dtype: cutlass.Constexpr):
    """TMA one box ``src`` (a tile of a TMA view) into SMEM byte address ``addr``."""
    dst = cute.make_tensor(
        cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=128), layout
    )
    s, d = cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(dst, 0, cute.rank(dst)),
        cute.group_modes(src, 0, cute.rank(src)),
    )
    cute.copy(atom, d, s, tma_bar_ptr=bar)


@dataclass(frozen=True)
class KdaPrepPrefillParams:
    v_load: RaggedTma  # (token, dim, head) loads, zero past each sequence end
    tma_kd: cute.CopyAtom
    tma_qd: cute.CopyAtom
    tma_t: cute.CopyAtom
    tma_a: cute.CopyAtom
    tma_diag: cute.CopyAtom
    mKd: cute.Tensor  # records (t, dim, head, row)
    mQd: cute.Tensor
    mT: cute.Tensor
    mA: cute.Tensor  # (word, head, row)
    mDiag: cute.Tensor  # (dim, head, row)
    o_store: RaggedTma  # (token, d_v, head) stores, dropping rows past each sequence end
    box_io: cute.ComposedLayout  # one 16 x 64 16-bit swizzled box
    state_in: cute.Tensor | None  # (row, head, d_v, d_k)
    state_out: cute.Tensor | None
    state_indices: cute.Tensor | None
    has_initial_state: cute.Tensor | None
    work_items: cute.Tensor  # (rows, WORK_ITEM_FIELDS)
    work_count: cute.Tensor  # (1,)
    scheduler: cute.Tensor  # (>= 1,) zeroed ticket counter
    v_ratio: Int32  # tile heads per v head
    scale: Float32
    s_v: Int32  # SMEM byte addresses of the buffers (stage 0)
    s_diag: Int32
    s_kd: Int32
    s_qd: Int32
    s_t: Int32
    s_a: Int32
    s_o: Int32


class KdaPrepPrefill(WarpSpecialized):
    """Warpdrive prep-fed KDA prefill for ``io`` fp16/bf16 records, V and O.

    The static state flags and ``paged_state`` (0 none, 1 routes, 2 routes + fresh-slot mask)
    select code; the state dtype follows the traced tensors."""

    def __init__(
        self,
        io: str,
        d_k: int,
        tiles_per_head: int,
        use_initial_state: bool,
        store_final_state: bool,
        paged_state: int,
        num_ctas: int,
    ):
        if d_k not in DIMS:
            raise ValueError(f"the KDA prep prefill serves d_k in {DIMS}, got {d_k}")
        if paged_state and not (use_initial_state and store_final_state):
            raise ValueError("paged state routes one pool in and out")
        self.io, self.d_k, self.tiles_per_head, self.num_ctas = io, d_k, tiles_per_head, num_ctas
        self.use_initial_state, self.store_final_state = use_initial_state, store_final_state
        self.paged_state = paged_state
        self.io_dtype = io_t = CUTE_DTYPES[io]
        io_b = io_t.width // 8
        plan = self.smem_plan = SmemPlan()
        buf = {
            "Kdecay": plan.buffer("Kdecay", io_t, align=1024),
            "Qdecay": plan.buffer("Qdecay", io_t, align=1024),
            "T": plan.buffer("T", io_t, align=1024),
            "A": plan.buffer("A", io_t, align=1024),
            "V": plan.buffer("V", io_t, align=1024),
            "Diag": plan.buffer("Diag", Float32, align=1024),
            "O": plan.buffer("O", io_t, align=1024),
        }
        self.buf = buf
        # The TMA check reports both TMA rings unverified: it cannot follow the per-item
        # domain_offset / row-sliced views.
        self.raw = Ring(
            RAW_STAGES,
            Tma(B_T * D_V * io_b + d_k * 4),
            {"cg0": Warps, "cg2": Warps},
            buffer={k: buf[k] for k in ("V", "Diag")},
        )
        # Upstream frees a record stage on three commits (after QS, the state update and the O
        # GEMM) and the loader waits for all three; the last of them is this ring's release.
        self.rec = Ring(
            REC_STAGES,
            Tma(3 * B_T * d_k * io_b + B_T * B_T * io_b),
            Umma,
            buffer={k: buf[k] for k in ("Kdecay", "Qdecay", "T", "A")},
        )
        self.sched = Ring(SCHED_STAGES, Warps, Warps, payload=(Int32, (1,)))
        self.o_smem = Ring(O_STAGES, Warps, Warps, commit_fence="smem_async", buffer=buf["O"])
        # The O drain issues no TMEM store: its release waits for the loads (tcgen05.wait::ld).
        self.o_acc = Ring(QS_STAGES, Umma, Warps, release_fence="tmem_load")
        self.tstate = Barriers(len(TSTATE_COUNTS), counts=list(TSTATE_COUNTS))
        self.tmem_plan = (
            TmemPlan()
            .region("state", d_k)
            .region("sin", d_k // 2)
            .region("qs", B_T, QS_STAGES)
            .region("ks", B_T)
            .region("uin", B_T // 2)
        )
        self.work = WorkStream(schedule=OneShot(grid="1d"), ready=PdlWait())
        # MMA geometry: KS/QS (TMEM state operand x k_decay/q_decay), O (TMEM Y x the 16 x 16
        # A tile), the state update (TMEM Y x MN-major T), split in two key halves at d_k = 128.
        self.n_update = d_k // 2 if d_k // 2 >= 64 else d_k
        f32 = Float32
        acc_idesc = instr_desc(f32, io_t, io_t, m=D_V, n=B_T)
        self.ks_desc = MmaDesc(M=D_V, N=B_T, K=d_k, bpe_a=2, bpe_b=2, idesc=acc_idesc)
        self.a_desc = MmaDesc(M=D_V, N=B_T, K=B_T, bpe_a=2, bpe_b=2, idesc=acc_idesc)
        self.t_desc = MmaDesc(
            M=D_V,
            N=self.n_update,
            K=B_T,
            bpe_a=2,
            bpe_b=2,
            btranspose=True,
            idesc=instr_desc(f32, io_t, io_t, m=D_V, n=self.n_update, b_mn_major=True),
        )

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer for ``heads`` heads: V and O, keyed by tile head."""
        return tma_slot_rows(2, B_T, heads * self.tiles_per_head)

    def get_name(self) -> str:
        flags = f"{int(self.use_initial_state)}{int(self.store_final_state)}{self.paged_state}"
        return f"wd_kda_prep_prefill_{self.io}_k{self.d_k}_t{self.tiles_per_head}_f{flags}"

    # ---- work decode -------------------------------------------------------------------

    @cute.jit
    def item(self, params: KdaPrepPrefillParams, tile):
        """``(batch, head_o, v_offset, write_start, write_end, compute_start, batch_start,
        seqlen, tile head)`` of work row ``tile``."""
        row = params.work_items
        head = row[tile, 1]
        head_o = head // self.tiles_per_head
        v_offset = (head - head_o * self.tiles_per_head) * D_V
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

    # ---- load (warp 14): V + diag, the four records, next item -----------------------------

    @role(warps=14, regs=56)
    @cute.jit
    def load(
        self,
        params: KdaPrepPrefillParams,
        *,
        raw: Producer,
        rec: Producer,
        sched: Producer,
        work: WorkCursor,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                raw, rec = self.load_item(params, self.item(params, tile), raw, rec)
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
        return raw, rec, sched, work

    @cute.jit
    def load_item(self, params, fields, raw, rec):
        """The V box, diag row and records of every chunk of one item."""
        batch, head_o, v_offset, _ws, write_end, compute_start, start, seqlen, head = fields
        head_v = head // params.v_ratio
        moves_rows = write_end > compute_start
        (v_desc,) = sequence_descriptors((params.v_load,), start + seqlen, head, B_T, moves_rows)
        rec_row = start // B_T + batch  # the sequence's first record row
        for chunk in cutlass.range(compute_start, write_end, 1, unroll=1):
            raw, rec = self.load_chunk(
                params,
                raw,
                rec,
                (start + chunk * B_T, cutlass.min(seqlen - chunk * B_T, B_T)),
                rec_row + chunk,
                head_v,
                head_o,
                v_offset,
                v_desc,
            )
        return raw, rec

    @cute.jit
    def load_chunk(self, params, raw, rec, tokens, row, head_v, head_o, v_offset, v_desc):
        """TMA one chunk's V box (``tokens = (token, valid)``, zero past them; through ``v_desc``)
        and diag row into a ``raw`` stage, then its k_decay, q_decay, T and A records into a
        ``rec`` stage (upstream's order)."""
        io, d_k = self.io_dtype, self.d_k
        box = B_T * 128  # bytes of one 16-row 128-byte box
        with raw.produce() as h:
            h.access("V", write=True, by="tma")
            h.access("Diag", write=True, by="tma")
            dst = cute.make_tensor(
                cute.make_ptr(
                    io,
                    params.s_v + h.index * (B_T * D_V * 2),
                    cute.AddressSpace.smem,
                    assumed_align=128,
                ),
                params.box_io,
            )
            token, valid = tokens
            ragged_load(
                params.v_load,
                dst,
                token,
                valid,
                head_v,
                v_offset // 64,
                h.barrier,
                B_T,
                64,
                v_desc,
            )
            _tma_box(
                params.tma_diag,
                cute.local_tile(params.mDiag[None, head_o, row], (d_k,), (0,)),
                params.s_diag + h.index * (d_k * 4),
                cute.make_layout((d_k,)),
                h.barrier,
                Float32,
            )
        with rec.produce() as h:
            for name in ("Kdecay", "Qdecay", "T", "A"):
                h.access(name, write=True, by="tma")
            stage = h.index * (d_k * B_T * 2)
            for atom, view, base in (
                (params.tma_kd, params.mKd, params.s_kd),
                (params.tma_qd, params.mQd, params.s_qd),
                (params.tma_t, params.mT, params.s_t),
            ):
                for kb in cutlass.range_constexpr(d_k // 64):
                    _tma_box(
                        atom,
                        cute.local_tile(view[None, None, head_o, row], (B_T, 64), (0, kb)),
                        base + stage + kb * box,
                        params.box_io,
                        h.barrier,
                        io,
                    )
            _tma_box(
                params.tma_a,
                cute.local_tile(params.mA[None, head_o, row], (A_WORDS,), (0,)),
                params.s_a + h.index * (B_T * B_T * 2),
                cute.make_layout((A_WORDS,)),
                h.barrier,
                Int32,
            )
        return raw, rec

    # ---- compute group 0 (warps 0-3): left state half ----------------------------------------

    @role(warps=range(4), regs=160)
    @cute.jit
    def cg0(
        self,
        params: KdaPrepPrefillParams,
        *,
        raw: Consumer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        io, d_k = self.io_dtype, self.d_k
        lane, warp = me.lane, me.warp
        pack_col = opaque_i32(2 * (lane % 4))
        lane_base = tmem.base.toint() + ((32 * warp) << 16)  # warp-uniform: this subpartition
        state_col = lane_base + tmem.cols("state")
        sin_col = lane_base + tmem.cols("sin")
        cum = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, _h, _v, _ws, write_end, compute_start, _s, _len, _t = self.item(params, tile)
                for local in cutlass.range(write_end - compute_start, unroll=1):
                    rh = raw.wait()
                    rh.access("Diag", read=True, by="threads")
                    s_diag = params.s_diag + rh.index * (d_k * 4)
                    if cum > 0:  # the previous chunk's left update has landed
                        with self.trace_region("sacc_l.wait"):
                            spin_wait(tstate + SACC_L + ((cum - 1) & 3), ((cum - 1) >> 2) & 1)
                    if local > 0:
                        vals = tmem_ld("16x256b", d_k // 16, state_col)
                        words = []
                        for m in cutlass.range_constexpr(d_k // 16):
                            words.append(fp32_to_fp16(vals[4 * m], vals[4 * m + 1], dtype=io))
                            words.append(fp32_to_fp16(vals[4 * m + 2], vals[4 * m + 3], dtype=io))
                        tmem_st("16x128b", sin_col, words)
                        cute.arch.fence_view_async_tmem_store()
                        warp_arrive(tstate + SIN_L)
                        scaled = []
                        for m in cutlass.range_constexpr(d_k // 16):
                            f = lds_v2_f32(s_diag + 4 * (8 * m + pack_col))
                            s0, s1 = fmul2(vals[4 * m], vals[4 * m + 1], f[0], f[1])
                            s2, s3 = fmul2(vals[4 * m + 2], vals[4 * m + 3], f[0], f[1])
                            scaled += [s0, s1, s2, s3]
                        tmem_st("16x256b", state_col, scaled)
                        cute.arch.fence_view_async_tmem_store()
                    warp_arrive(tstate + UIN)  # the left half is decayed (with CG2's Y operand)
                    rh.release()
                    cum += 1
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, sched, work

    # ---- compute group 1 (warps 4-7): O drain ------------------------------------------------

    @role(warps=range(4, 8), regs=160)
    @cute.jit
    def cg1(
        self,
        params: KdaPrepPrefillParams,
        *,
        o_acc: Consumer,
        o_smem: Producer,
        sched: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        ov_row = (lane // 16) * 8 + (lane & 7)
        ov_col = ((lane // 8) & 1) * 8
        ov = box128(ov_row, warp * 16 + ov_col)
        qs_col = tmem.base.toint() + ((32 * warp) << 16) + tmem.cols("qs")
        scale = params.scale
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                batch, _h, _v, _ws, write_end, compute_start, _s, _len, _t = self.item(
                    params, tile
                )
                _slot, active, _li, _ce = resolve_paged_state(
                    batch, params.state_indices, params.has_initial_state
                )
                for _ in cutlass.range(write_end - compute_start, unroll=1):
                    oh = o_smem.acquire()
                    oh.access(write=True, by="threads")
                    oa = o_acc.wait()
                    acc = tmem_ld("16x256b", 2, qs_col + oa.index * B_T)
                    words = []
                    for r in cutlass.range_constexpr(4):
                        a0, a1 = fmul2(acc[2 * r], acc[2 * r + 1], scale, scale)
                        if cutlass.const_expr(self.paged_state > 0):
                            # Null routes emit zero output (predicated, so NaN inputs cannot leak).
                            a0 = Float32(cutlass.select_(active, a0, Float32(0.0)))
                            a1 = Float32(cutlass.select_(active, a1, Float32(0.0)))
                        words.append(fp32_to_fp16(a0, a1, dtype=self.io_dtype))
                    stmatrix(params.s_o + oh.index * (B_T * D_V * 2) + 2 * ov, words, trans=True)
                    oh.commit()
                    oa.release()
                sched, tile = self.next_tile(sched)
            work.advance()
        return o_acc, o_smem, sched, work

    # ---- compute group 2 (warps 8-11): seed, right state half, Y, final state -----------------

    @role(warps=range(8, 12), regs=136)
    @cute.jit
    def cg2(
        self,
        params: KdaPrepPrefillParams,
        *,
        raw: Consumer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        ov_row = (lane // 16) * 8 + (lane & 7)
        ov_col = ((lane // 8) & 1) * 8
        lane_base = tmem.base.toint() + ((32 * warp) << 16)  # warp-uniform: this subpartition
        cols = {name: lane_base + tmem.cols(name) for name in ("state", "sin", "ks", "uin")}
        geo = (
            lane,
            cols,
            box128(ov_row, warp * 16 + ov_col),
            opaque_i32(2 * (lane % 4)),
            warp * 16 + lane % 16,  # value row of the state
        )
        # Completed waits on the KS accumulator and the right-half state updates.
        n_ks = Int32(0)
        n_upd = Int32(0)
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
                    raw, n_ks = self.cg2_first(
                        params,
                        raw,
                        tstate,
                        n_ks,
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
                    raw, n_ks, n_upd = self.cg2_chunk(
                        params,
                        raw,
                        tstate,
                        n_ks,
                        n_upd,
                        geo,
                        (compute_start + local) * B_T,
                        seqlen,
                    )
                if n_local > 0:
                    with self.trace_region("sacc_r.wait"):
                        spin_wait(
                            tstate + SACC_R + (n_upd & 3), (n_upd >> 2) & 1
                        )  # the last update
                    n_upd += 1
                    with self.trace_region("odone.wait"):  # the last O GEMM read the Y operand
                        spin_wait(tstate + ODONE, (n_upd - 1) & 1)
                if cutlass.const_expr(self.store_final_state):
                    self.final_state(
                        params,
                        geo,
                        batch,
                        head_o,
                        v_offset,
                        seqlen,
                        params.work_items[tile, FINAL_DST],
                        slot,
                        active,
                        clear_empty,
                    )
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, sched, work

    @cute.jit
    def cg2_first(
        self,
        params,
        raw,
        tstate,
        n_ks,
        geo,
        seed,
        seed_row,
        head_o,
        v_offset,
        load_initial,
        chunk_start,
        seqlen,
    ):
        """An item's first chunk: optional seed of both state halves, then Y (V, or V - KS
        of the seeded state)."""
        _l, cols, _ov, _pc, _vd = geo
        io = self.io_dtype
        rh = raw.wait()
        rh.access("V", read=True, by="threads")
        rh.access("Diag", read=True, by="threads")
        if cutlass.const_expr(self.use_initial_state):  # noqa: SIM102 (static, then dynamic)
            if seed:
                self.seed_state(
                    params,
                    geo,
                    tstate,
                    seed_row,
                    head_o,
                    v_offset,
                    load_initial,
                    params.s_diag + rh.index * (self.d_k * 4),
                )
        v = self.load_v(params, rh.index, geo, chunk_start, seqlen)
        y = v
        if seed:
            with self.trace_region("ks.wait"):
                spin_wait(tstate + KS, n_ks & 1)
            n_ks += 1
            ks = tmem_ld("16x256b", 2, cols["ks"])
            y = [_residual(v[r], ks[2 * r], ks[2 * r + 1], io) for r in range(4)]
        tmem_st("16x128b", cols["uin"], y)
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + UIN)
        rh.release()
        return raw, n_ks

    @cute.jit
    def seed_state(self, params, geo, tstate, seed_row, head_o, v_offset, load_initial, s_diag):
        """Initial state row -> b16 state operand (left half, then right, each published as
        soon as it is stored) and fp32 state decayed by the first chunk's diag."""
        _l, cols, _ov, _pc, value_dim = geo
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
                tmem_st("32x32b", cols["sin"] + i * 8, words)
            cute.arch.fence_view_async_tmem_store()
            warp_arrive(tstate + (SIN_L if half == 0 else SIN_R))
            for j, i in enumerate(blocks):
                scaled = []
                for grp in cutlass.range_constexpr(4):
                    f = lds_v4_f32(s_diag + 4 * (i * 16 + grp * 4))
                    for t in cutlass.range_constexpr(2):
                        s0, s1 = fmul2(
                            vecs[j][grp * 4 + 2 * t],
                            vecs[j][grp * 4 + 2 * t + 1],
                            f[2 * t],
                            f[2 * t + 1],
                        )
                        scaled += [s0, s1]
                tmem_st("32x32b", cols["state"] + i * 16, scaled)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def load_v(self, params, stage, geo, chunk_start, seqlen):
        """This thread's transposed V pairs (``ldmatrix.trans``), zeroed past the sequence end."""
        lane, _c, ov, _pc, _vd = geo
        v = ldmatrix(params.s_v + stage * (B_T * D_V * 2) + 2 * ov, 4, trans=True)
        if chunk_start + B_T > seqlen:
            valid = seqlen - chunk_start
            for r in cutlass.range_constexpr(4):
                v[r] = mask_pair(v[r], ((r // 2) * 4 + (lane & 3)) * 2, valid)
        return v

    @cute.jit
    def cg2_chunk(self, params, raw, tstate, n_ks, n_upd, geo, chunk_start, seqlen):
        """A chunk after the first: right state half (b16 operand, diag decay), then Y."""
        _l, cols, _ov, pack_col, _vd = geo
        io = self.io_dtype
        d_k = self.d_k
        rh = raw.wait()
        rh.access("V", read=True, by="threads")
        rh.access("Diag", read=True, by="threads")
        s_diag = params.s_diag + rh.index * (d_k * 4)
        with self.trace_region("sacc_r.wait"):
            spin_wait(tstate + SACC_R + (n_upd & 3), (n_upd >> 2) & 1)
        n_upd += 1
        vals = tmem_ld("16x256b", d_k // 16, cols["state"] + d_k // 2)
        words = []
        for m in cutlass.range_constexpr(d_k // 16):
            words.append(fp32_to_fp16(vals[4 * m], vals[4 * m + 1], dtype=io))
            words.append(fp32_to_fp16(vals[4 * m + 2], vals[4 * m + 3], dtype=io))
        tmem_st("16x128b", cols["sin"] + d_k // 4, words)
        scaled = []
        for m in cutlass.range_constexpr(d_k // 16):
            f = lds_v2_f32(s_diag + 4 * (d_k // 2 + 8 * m + pack_col))
            s0, s1 = fmul2(vals[4 * m], vals[4 * m + 1], f[0], f[1])
            s2, s3 = fmul2(vals[4 * m + 2], vals[4 * m + 3], f[0], f[1])
            scaled += [s0, s1, s2, s3]
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + SIN_R)
        tmem_st("16x256b", cols["state"] + d_k // 2, scaled)
        # ---- Y = V - k_decay S ----------------------------------------------------------------
        v = self.load_v(params, rh.index, geo, chunk_start, seqlen)
        with self.trace_region("ks.wait"):
            spin_wait(tstate + KS, n_ks & 1)
        n_ks += 1
        ks = tmem_ld("16x256b", 2, cols["ks"])
        y = [_residual(v[r], ks[2 * r], ks[2 * r + 1], io) for r in range(4)]
        tmem_st("16x128b", cols["uin"], y)
        cute.arch.fence_view_async_tmem_store()
        warp_arrive(tstate + UIN)
        rh.release()
        return raw, n_ks, n_upd

    @cute.jit
    def final_state(
        self, params, geo, batch, head_o, v_offset, seqlen, final_dst, slot, active, clear_empty
    ):
        """TMEM state -> ``state_out``; empty items copy the initial state or write zeros (paged:
        fresh empty routes clear their slot)."""
        lane, cols, _ov, _pc, value_dim = geo
        row_valid = lane < 16  # 64 value rows: lanes 16-31 of each subpartition hold no state
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
                    vals = tmem_ld("32x32b", 32, cols["state"] + kb)
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

    # ---- epilogue (warp 15): O stores ---------------------------------------------------------

    @role(warps=15, regs=56)
    @cute.jit
    def epilogue(
        self,
        params: KdaPrepPrefillParams,
        *,
        o_smem: Consumer,
        sched: Consumer,
        work: WorkCursor,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                o_smem = self.store_item(params, o_smem, self.item(params, tile))
                sched, tile = self.next_tile(sched)
            work.advance()
        bulk_wait_group(0)
        return o_smem, sched, work

    @cute.jit
    def store_item(self, params, o_smem, fields):
        """TMA-store the staged O chunks of one item."""
        _b, head_o, v_offset, write_start, write_end, compute_start, start, seqlen, head = fields
        moves_rows = write_end > compute_start
        (o_desc,) = sequence_descriptors((params.o_store,), start + seqlen, head, B_T, moves_rows)
        for chunk in cutlass.range(compute_start, write_end, 1, unroll=1):
            o_smem = self.store_o(
                params, o_smem, chunk, write_start, start, seqlen, head_o, v_offset, o_desc
            )
        return o_smem

    @cute.jit
    def store_o(self, params, o_smem, chunk, write_start, start, seqlen, head_o, v_offset, o_desc):
        """TMA-store one staged O chunk through ``o_desc`` (rows past the sequence end are
        clipped); warm-up chunks before ``write_start`` release their stage unstored."""
        with o_smem.consume() as oh:
            oh.access(read=True, by="tma")
            if chunk >= write_start:
                valid = cutlass.min(seqlen - chunk * B_T, B_T)
                src = cute.make_tensor(
                    cute.make_ptr(
                        self.io_dtype,
                        params.s_o + oh.index * (B_T * D_V * 2),
                        cute.AddressSpace.smem,
                        assumed_align=1024,
                    ),
                    params.box_io,
                )
                token, col = start + chunk * B_T, v_offset // 64
                ragged_store(params.o_store, src, token, valid, head_o, col, B_T, 64, o_desc)
                bulk_commit_group()
            bulk_wait_group(0, read=True)
        return o_smem

    # ---- tcgen05 MMA (warp 13) ------------------------------------------------------------

    @role(warps=13, regs=56)
    @cute.jit
    def mma(
        self,
        params: KdaPrepPrefillParams,
        *,
        rec: Consumer,
        o_acc: Producer,
        sched: Consumer,
        work: WorkCursor,
        tstate: Barriers,
        tmem: Tmem,
    ):
        elect = elect_sync()
        ksd, ad, td = self.ks_desc, self.a_desc, self.t_desc
        t0 = tmem.base.toint()
        sin_col = t0 + tmem.cols("sin")
        ks_col = t0 + tmem.cols("ks")
        uin_col = t0 + tmem.cols("uin")
        qs_col = t0 + tmem.cols("qs")
        state_col = t0 + tmem.cols("state")
        a_seg = ksd.sps_B * ksd.tmem_advance_A  # TMEM columns of one 64-wide K box
        b_seg = ksd.smem_subtile_B >> 4
        half_steps = ksd.num_k_steps // 2
        split = self.n_update < self.d_k
        stage_bytes = self.d_k * B_T * 2
        n_sin = Int32(0)  # completed waits on the two state-operand halves
        cum = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, _h, _v, _ws, write_end, compute_start, _s, _len, _t = self.item(params, tile)
                n_local = write_end - compute_start
                seeded = cutlass.Boolean(False)
                if cutlass.const_expr(self.use_initial_state):
                    seeded = compute_start == 0
                for local in cutlass.range(n_local, unroll=1):
                    have_state = local > 0 or seeded
                    kh = rec.wait()
                    for name in ("Kdecay", "Qdecay", "T", "A"):
                        kh.access(name, read=True, by="umma")
                    stage = kh.index * stage_bytes
                    desc_kd = desc_opaque(smem_desc(params.s_kd + stage, 16, 1024, SWIZZLE_128B))
                    desc_qd = desc_opaque(smem_desc(params.s_qd + stage, 16, 1024, SWIZZLE_128B))
                    # Upstream waits for the O accumulator after KS; acquiring first keeps KS
                    # and QS under one have_state branch (two made ptxas wrap the QS MMAs in
                    # per-lane ELECT loops). Each descriptor is built next to its GEMM: built
                    # together, they pushed the channel state out of uniform registers.
                    oh = o_acc.acquire()
                    qs = qs_col + oh.index * B_T
                    if have_state:
                        # ---- KS = S k_decay^T: the left half of S lands first --------------
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
                        # ---- QS = S q_decay^T into the O accumulator ------------------------
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
                    # ---- S += T^T Y, left then right key half -------------------------------
                    desc_t = desc_opaque(
                        smem_desc(params.s_t + stage, B_T * 128, 1024, SWIZZLE_128B)
                    )
                    with self.trace_region("uin.wait"):
                        spin_wait(tstate + UIN, cum & 1)
                    mma_ts_op(td, uin_col, desc_t, state_col, 0, have_state, elect)
                    umma_commit(tstate + SACC_L + (cum & 3), elect)
                    if cutlass.const_expr(split):
                        mma_ts_op(
                            td,
                            uin_col,
                            desc_t + ((B_T * 128) >> 4),
                            state_col + self.n_update,
                            0,
                            have_state,
                            elect,
                        )
                    umma_commit(tstate + SACC_R + (cum & 3), elect)
                    # ---- O += Y A -----------------------------------------------------------
                    desc_a = desc_opaque(
                        smem_desc(params.s_a + kh.index * (B_T * B_T * 2), 16, 256, SWIZZLE_32B)
                    )
                    mma_ts_op(ad, uin_col, desc_a, qs, 0, have_state, elect)
                    oh.commit(signal=False)
                    umma_commit(oh.full_barrier, elect)
                    kh.release(signal=False)  # the role's elected lane commits
                    umma_commit(kh.empty_barrier, elect)
                    umma_commit(tstate + ODONE, elect)
                    cum += 1
                sched, tile = self.next_tile(sched)
            work.advance()
        return rec, o_acc, sched, work

    # ---- kernel and entry point ---------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        v_load,
        tma_kd,
        tma_qd,
        tma_t,
        tma_a,
        tma_diag,
        o_store,
        mKd,
        mQd,
        mT,
        mA,
        mDiag,
        box_io,
        state_in,
        state_out,
        state_indices,
        has_initial_state,
        work_items,
        work_count,
        scheduler,
        v_ratio,
        scale,
        layouts,
        trace_buffer,
        bound,
    ):
        self.begin_roles(work=bound, layouts=layouts)
        addr = {name: self.buffer(name).iterator.toint() for name in self.buf}
        params = KdaPrepPrefillParams(
            v_load=v_load,
            tma_kd=tma_kd,
            tma_qd=tma_qd,
            tma_t=tma_t,
            tma_a=tma_a,
            tma_diag=tma_diag,
            o_store=o_store,
            mKd=mKd,
            mQd=mQd,
            mT=mT,
            mA=mA,
            mDiag=mDiag,
            box_io=box_io,
            state_in=state_in,
            state_out=state_out,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            work_items=work_items,
            work_count=work_count,
            scheduler=scheduler,
            v_ratio=v_ratio,
            scale=scale,
            s_v=addr["V"],
            s_diag=addr["Diag"],
            s_kd=addr["Kdecay"],
            s_qd=addr["Qdecay"],
            s_t=addr["T"],
            s_a=addr["A"],
            s_o=addr["O"],
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def launch(
        self,
        v: cute.Tensor,  # (tokens, v heads, d_v x tiles_per_head)
        k_decay: cute.Tensor,  # (record rows, heads, 16, d_k)
        q_decay: cute.Tensor,
        t: cute.Tensor,
        a: cute.Tensor,  # (record rows, heads, 128) int32: 32-byte swizzled 16 x 16 b16 tiles
        diag: cute.Tensor,  # (record rows, heads, d_k) fp32
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
        a launch bundle, after kernels that wrote the records, ``work_items``/``work_count`` and
        zeroed ``scheduler``."""
        io = self.io_dtype
        heads_out = Int32(diag.shape[1]) * self.tiles_per_head
        v_ratio = heads_out // Int32(v.shape[1])
        box_io = cute.select(
            sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1), mode=[0, 1]
        )

        def tdh(x):  # (token, head, dim) -> (token, dim, head)
            return cute.make_tensor(x.iterator, cute.select(x.layout, mode=[0, 2, 1]))

        def record(x):  # (row, head, t, dim) -> (t, dim, head, row)
            return cute.make_tensor(x.iterator, cute.select(x.layout, mode=[2, 3, 1, 0]))

        def vector(x):  # (row, head, n) -> (n, head, row)
            return cute.make_tensor(x.iterator, cute.select(x.layout, mode=[2, 1, 0]))

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        def ragged(op, x, index):  # clipped at each sequence end, keyed by tile head
            slots = slot_region(tma_slots, index, B_T, heads_out)
            return make_ragged_tma(op, tdh(x), box_io, B_T, 64, slots)

        v_load = ragged(g2s, v, 0)
        tma_kd, mKd = cpasync.make_tiled_tma_atom(g2s, record(k_decay), box_io, (B_T, 64))
        tma_qd, mQd = cpasync.make_tiled_tma_atom(g2s, record(q_decay), box_io, (B_T, 64))
        tma_t, mT = cpasync.make_tiled_tma_atom(g2s, record(t), box_io, (B_T, 64))
        tma_a, mA = cpasync.make_tiled_tma_atom(
            g2s, vector(a), cute.make_layout((A_WORDS,)), (A_WORDS,)
        )
        tma_diag, mDiag = cpasync.make_tiled_tma_atom(
            g2s, vector(diag), cute.make_layout((self.d_k,)), (self.d_k,)
        )
        o_store = ragged(cpasync.CopyBulkTensorTileS2GOp(), o, 1)
        d_k = self.d_k
        layouts = {
            "Kdecay": cute.make_layout((d_k * B_T, REC_STAGES)),
            "Qdecay": cute.make_layout((d_k * B_T, REC_STAGES)),
            "T": cute.make_layout((d_k * B_T, REC_STAGES)),
            "A": cute.make_layout((B_T * B_T, REC_STAGES)),
            "V": cute.make_layout((D_V * B_T, RAW_STAGES)),
            "Diag": cute.make_layout((d_k, RAW_STAGES)),
            "O": cute.make_layout((D_V * B_T, O_STAGES)),
        }
        bound = self.work.bind((self.num_ctas, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            v_load,
            tma_kd,
            tma_qd,
            tma_t,
            tma_a,
            tma_diag,
            o_store,
            mKd,
            mQd,
            mT,
            mA,
            mDiag,
            box_io,
            state_in,
            state_out,
            state_indices,
            has_initial_state,
            work_items,
            work_count,
            scheduler,
            v_ratio,
            scale,
            layouts,
            trace_buffer,
            bound,
        ).launch(**self.launch_config(bound), stream=stream, use_pdl=True)

    @cute.jit
    def __call__(
        self,
        v: cute.Tensor,
        k_decay: cute.Tensor,
        q_decay: cute.Tensor,
        t: cute.Tensor,
        a: cute.Tensor,
        diag: cute.Tensor,
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
            v,
            k_decay,
            q_decay,
            t,
            a,
            diag,
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


def record_rows(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """First record row of each sequence: ``start // 16 + sequence`` (one padding row per
    sequence keeps it closed-form, as ``kda_prep`` lays its records out)."""
    seq = torch.arange(cu_seqlens.shape[0] - 1, device=cu_seqlens.device)
    return torch.div(cu_seqlens[:-1], B_T, rounding_mode="floor") + seq


@jit_cache
def compile_kda_prep_prefill(op: KdaPrepPrefill, state_name: str | None):
    """Compile for dynamic tokens, heads, sequences, record rows and work rows."""
    io, i32 = op.io_dtype, cutlass.Int32
    tokens, hv, heads, rows, items = (cute.sym_int() for _ in range(5))
    dv_full = D_V * op.tiles_per_head

    def packed(h):
        return make_fake_strided_tensor(io, (tokens, h, dv_full), stride_divisibility=8)

    def rec(dtype, *tail):
        return make_fake_strided_tensor(dtype, (rows, heads, *tail), stride_divisibility=8)

    state = None
    if state_name is not None:
        state = make_fake_strided_tensor(
            CUTE_DTYPES[state_name],
            (cute.sym_int(), heads, dv_full, op.d_k),
            stride_divisibility=4,
        )
    return compile_tvm_ffi(
        op,
        packed(hv),
        rec(io, B_T, op.d_k),
        rec(io, B_T, op.d_k),
        rec(io, B_T, op.d_k),
        rec(i32, A_WORDS),
        rec(Float32, op.d_k),
        state if op.use_initial_state else None,
        packed(heads),
        state if op.store_final_state else None,
        None,
        None,
        make_fake_strided_tensor(i32, (items, WORK_ITEM_FIELDS), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        fake_tma_slots(),
        Float32(1.0),
        name=op.get_name(),
    )


def kda_prep_prefill(
    v: torch.Tensor,
    k_decay: torch.Tensor,
    q_decay: torch.Tensor,
    t: torch.Tensor,
    a: torch.Tensor,
    diag: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Packed ``[T, H, d_v]`` KDA forward over prep records ``[rows, H, 16, d_k]`` (k_decay,
    q_decay, T), ``[rows, H, 128]`` int32 (A) and ``[rows, H, d_k]`` fp32 (diag), sequence ``b``
    chunk ``j`` at row ``record_rows(cu_seqlens)[b] + j``; state ``[N, H, d_v, d_k]``. Returns
    ``(o, final_state)``."""
    if v.dtype not in (torch.float16, torch.bfloat16) or k_decay.dtype != v.dtype:
        raise TypeError("v and the records must share a 16-bit dtype")
    heads, d_k, d_v = diag.shape[1], diag.shape[2], v.shape[-1]
    if d_v % D_V:
        raise ValueError(f"d_v must be a multiple of {D_V}")
    num_seqs = cu_seqlens.shape[0] - 1
    o = torch.empty(v.shape[0], heads, d_v, dtype=v.dtype, device=v.device)
    final = None
    if output_final_state:
        final = (
            initial_state.clone()
            if initial_state is not None
            else torch.zeros(num_seqs, heads, d_v, d_k, dtype=torch.float32, device=v.device)
        )
    state_name = None
    for x in (initial_state, final):
        if x is not None:
            state_name = TORCH_NAMES[x.dtype]
    tiles = d_v // D_V
    op = KdaPrepPrefill(
        TORCH_NAMES[v.dtype],
        d_k,
        tiles,
        initial_state is not None,
        final is not None,
        0,
        torch.cuda.get_device_properties(v.device).multi_processor_count,
    )
    table, count, scheduler = uncut_work_items(cu_seqlens, heads, tiles)
    fn = compile_kda_prep_prefill(op, state_name)
    fn(
        v,
        k_decay,
        q_decay,
        t,
        a,
        diag,
        initial_state,
        o,
        final,
        None,
        None,
        table,
        count,
        scheduler,
        new_tma_slots(op.tma_slot_rows(heads), v.device),
        scale,
    )
    return o, final
