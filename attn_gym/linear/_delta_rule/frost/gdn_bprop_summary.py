# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# gdn_bprop_summary_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and
# channels with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence
# descriptor arrays, one CTA runs one work item, and the zero-seeded natural-log-gate form the
# chain backward compiles is the only one.

"""GDN backward piece summaries for SM100: per (piece, head) work item, the gradient ``G`` a
piece's outputs send to the state entering it, from a zero state gradient at its end. The chain
backward composes them into each piece's state gradient (``dX_start = M^T dX_end + G``).

Per chunk ``c``, last to first (tokens ``[c * 64, (c + 1) * 64)`` of the piece), with ``g`` the
log2 gate cumsum over the chunk, ``T`` the beta-folded chunk inverse of ``frost.gdn_tinv``, the
state gradient ``D`` (``[d_v, d_k]`` fp32, held in TMEM) and its 16-bit copy ``D16`` taken
before the chunk's rescale::

    A    = (Q K^T) * exp2(g_i - g_j) [i >= j] * scale            (16-bit, SMEM)
    dV   = (D16 @ K^T) * exp2(g[63] - g_j) + dO^T @ A            (dU, the chunk's value gradient)
    dY'  = -exp2(g_j) * (dU @ T)
    D    = exp2(g[63]) * D + (dO * exp2(g_i) * scale)^T @ Q + dY'^T @ K

``D`` starts from zero (the last chunk has no ``D16`` term), and the rescale of chunk ``c``
uses chunk ``c - 1``'s ``exp2(g[63])``: the gradient leaving chunk ``c`` enters chunk ``c - 1``
through that chunk's total decay. Items are the first ``work_count[0]`` rows of a split-K work
table, ``work_items[i] = (piece, head, write_start, -, -, compute_end, start, end, ...)``: chunks
``compute_end - 1`` down to ``write_start`` of the piece at tokens ``[start, end)``. ``G`` lands at
``state_g[piece, head]`` (``[d_v, d_k]`` fp32) when ``write_start == 0``, zero for an item
without chunks. The chain prologue's tables cover whole pieces, ``(0, num_chunks)``.

Q, K and dO rows past the piece's end load as zeros (``frost.ragged.RaggedTma``), as from
upstream's per-sequence descriptors. Gates there are neutral and ``T``'s rows and columns are
zero.

Roles and channels (16 warps; warp 15 pads the register warpgroup)::

    load (13)  --q:     2 x Q [64, d_k] (TMA)-->          mma (12)  (QK, then the Q term)
    load       --k:     4 x K [64, d_k] (TMA)-->          mma       (QK, dV, then the K term)
    load       --dout:  2 x dO [64, d_v] (TMA)-->         mma (dU) + pairwise (dO')
    load       --tinv:  1 x T [64, 64] (TMA)-->           mma       (dY)
    gate (14)  --gate:  2 x log2 cumsum, cumprod-->       pairwise + du + state
    mma        --a_acc: 1 x QK (TMEM)-->                  pairwise (0-3)
    pairwise   --do_prime: 1 x dO' (TMEM)-->              mma       (Q term)
    pairwise   --a_tile:   1 x A [64, 64] (SMEM)-->       mma       (dU)
    mma       <-trip:   1 (UMMA / threads)->              du (4-7)
    mma        --dstate:   1 x D (TMEM)-->                state (8-11)
    state      --dstate_in: 1 x D16 (TMEM)-->             mma       (dV)

Each chunk's GEMMs and the du/state steps form one serial chain (K term -> D16 -> dV -> rescale
-> dU -> pack -> dY -> dY' -> K term), as upstream's one-way signals do. ``trip`` is the dU
group's round trip: the MMA warp acquires it once the group released the previous epoch (the
operand its next GEMM reads is written), issues one GEMM and commits; the group waits for that
result, reads it, writes the next operand and releases. Per chunk it carries ``dV`` (chunks
after the first), ``dU``, ``dY`` and the K term, whose epoch only orders the next chunk's ``dV``
behind the ``dY'`` the K term read. ``dstate`` and ``dstate_in`` are upstream's two-way
handoffs. The QK of chunk ``c - 1`` is issued during chunk ``c`` (upstream's order): it reads Q
and K through ``peek()`` previews that chunk ``c - 1``'s waits own.
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
    OneShot,
    Producer,
    Ring,
    SmemPlan,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkMap,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_strided_tensor
from attn_gym.warpdrive.compat import LayoutEnum

from .ptx import (
    f16x2_to_f32,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    opaque_f32_zero,
    shfl_idx,
    shfl_up,
    st_global_v4_f32,
    stmatrix,
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
from .umma import SWIZZLE_128B, MmaDesc, desc_opaque, instr_desc, mma_ss, mma_ts, smem_desc

B_T = 64
STATE_DIMS = (64, 128)
RCP_LN2 = 1.4426950408889634  # 1 / ln 2: natural-log gates -> the kernel's log2 domain
BOX = 64  # 16-bit elements per 128-byte swizzle row
BOX_BYTES = B_T * BOX * 2  # one 64 x 64 16-bit box
Q_STAGES = 2
K_STAGES = 4
DO_STAGES = 2
GATE_STAGES = 2
WORK_ITEM_FIELDS = 10
# Every channel spins on try_wait, as upstream's barriers do: the kernel is one latency-bound
# chain, and leaving any one channel on the default suspend-hinted wait (NANOSLEEP.SYNCS
# fallback) measured 0.5-3% slower, the loader's and the chain's channels most.
SPIN = "spin"

_CUTE_DTYPES = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16}
_IO_DTYPES = {torch.float16: "fp16", torch.bfloat16: "bf16"}


@dataclass(frozen=True)
class BpropSummaryItems(WorkMap):
    """Rows of the chain prologue's work table: ``(piece, head, write_start, compute_end, start,
    end)``, chunks counted from the piece's first token ``start``."""

    words = 6

    def count64(self, space):
        table, _ = space
        return Int64(table.shape[0])

    def decode(self, index, space, state):
        table, _ = space
        row = (table[index, 0], table[index, 1], table[index, 2], table[index, 5])
        return True, (*row, table[index, 6], table[index, 7]), state

    def get_key(self) -> str:
        return "gdn_bprop_summary_items"


@dataclass(frozen=True)
class BpropSummaryParams:
    q_load: RaggedTma  # (token, d_k, key head) loads, zero past each piece end
    k_load: RaggedTma
    do_load: RaggedTma  # (token, d_v, head)
    tma_tinv: cute.CopyAtom
    mTinv: cute.Tensor  # TMA view (i, j, head, row)
    gate: cute.Tensor  # (token, head) natural-log gate
    state_g: cute.Tensor  # (piece, head, d_v, d_k) fp32
    sQ: cute.Tensor  # whole rings (iterator = stage 0)
    sK: cute.Tensor
    sdO: cute.Tensor
    sTinv: cute.Tensor
    sA: cute.Tensor
    box_layout: cute.ComposedLayout  # one 64 x 64 swizzled box: every TMA destination
    q_group: Int32  # heads per query head
    k_group: Int32  # heads per key head
    scale: Float32


class GdnBpropSummary(WarpSpecialized):
    """Warpdrive GDN backward piece summaries for ``dtype_name`` fp16/bf16 operands, ``d_k``/``d_v``
    64 or 128 and fp32 summaries."""

    def __init__(self, dtype_name: str, d_k: int, d_v: int):
        if d_k not in STATE_DIMS or d_v not in STATE_DIMS:
            raise ValueError(
                f"the GDN bprop summary serves d_k, d_v in {STATE_DIMS}, got {d_k}, {d_v}"
            )
        self.dtype_name, self.d_k, self.d_v = dtype_name, d_k, d_v
        self.io_dtype = _CUTE_DTYPES[dtype_name]
        self.qk_stage_bytes = d_k // BOX * BOX_BYTES
        self.do_stage_bytes = d_v // BOX * BOX_BYTES
        self.smem_plan = SmemPlan()
        q_buf = self.smem_plan.buffer("Q", self.io_dtype, align=1024)
        k_buf = self.smem_plan.buffer("K", self.io_dtype, align=1024)
        do_buf = self.smem_plan.buffer("dO", self.io_dtype, align=1024)
        tinv_buf = self.smem_plan.buffer("Tinv", self.io_dtype, align=1024)
        a_buf = self.smem_plan.buffer("A", self.io_dtype, align=1024)
        # The TMA check reports these rings unverified: it cannot follow copies through the
        # sequence-offset (domain_offset) views. Each stage takes d / 64 boxes of 64 x 64.
        self.q = Ring(Q_STAGES, Tma(self.qk_stage_bytes), Umma, buffer=q_buf, wait=SPIN)
        self.k = Ring(K_STAGES, Tma(self.qk_stage_bytes), Umma, buffer=k_buf, wait=SPIN)
        self.dout = Ring(
            DO_STAGES,
            Tma(self.do_stage_bytes),
            {"mma": Umma, "pairwise": Threads},
            buffer=do_buf,
            wait=SPIN,
        )
        self.tinv = Ring(1, Tma(BOX_BYTES), Umma, buffer=tinv_buf, wait=SPIN)
        self.gate = Ring(GATE_STAGES, Threads, Threads, payload=(Float32, (B_T, 2)), wait=SPIN)
        # QK accumulator: the pairwise group releases it once its loads completed (a fenced
        # wait before the A math), so the next QK never waits: the MMA warp issues it after the
        # dU GEMM, which waited for the A tile those loads produced.
        self.a_acc = Ring(1, Umma, Threads, release_fence="tmem_load", wait=SPIN)
        self.do_prime = Ring(1, Threads, Umma, commit_fence="tmem_store", wait=SPIN)
        self.a_tile = Ring(1, Threads, Umma, buffer=a_buf, wait=SPIN)
        # Each release hands the MMA warp TMEM the group loaded from and stored to.
        trip_fence = ("tmem_load", "tmem_store")
        self.trip = Ring(1, Umma, Threads, release_fence=trip_fence, wait=SPIN)
        self.dstate = Ring(1, Umma, Threads, release_fence=trip_fence, wait=SPIN)
        self.dstate_in = Ring(1, Threads, Umma, commit_fence="tmem_store", wait=SPIN)
        self.tmem_plan = (
            TmemPlan()
            .region("dstate", d_k)  # D, fp32 [d_v, d_k]
            .region("dvdk", B_T)  # dV, then dU
            .region("dstate_in", d_k // 2)  # D16: the dV GEMM's A operand
            .region("dy", B_T)
            .region("a", B_T)  # QK
            .region("do_prime", B_T // 2)  # dO': the Q term's A operand
            .region("du", B_T // 2)  # dU, then dY': the dY and K-term A operands
        )
        self.work = WorkStream(mapping=BpropSummaryItems(), schedule=OneShot())

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: Q, K and dO, keyed by head."""
        return tma_slot_rows(3, B_T, heads)

    def get_name(self) -> str:
        return f"frost_gdn_bprop_summary_{self.dtype_name}_dk{self.d_k}_dv{self.d_v}"

    # ---- shared helpers ----------------------------------------------------------------

    @cute.jit
    def _smem_ptr(self, addr):
        """16-bit SMEM pointer at 1024-byte aligned 32-bit shared address ``addr``."""
        return cute.make_ptr(self.io_dtype, addr, cute.AddressSpace.smem, assumed_align=1024)

    @staticmethod
    def _frag_token(lane, kv):
        """Token (column) of value ``kv`` of a thread's 16x256b x8 TMEM load."""
        return (lane % 4) * 2 + (kv // 4) * 8 + kv % 2

    @cute.jit
    def _mask_tokens(self, words, lane, n_valid):
        """Zero the 16-bit halves of packed 16x256b words whose token is ``>= n_valid``."""
        out = []
        for j in cutlass.range_constexpr(len(words)):
            tok = self._frag_token(lane, 2 * j)
            keep = cutlass.select_(tok < n_valid, Int32(0xFFFF), Int32(0))
            keep = keep | cutlass.select_(tok + 1 < n_valid, Int32(-65536), Int32(0))
            out.append(words[j] & keep)
        return out

    # ---- load (warp 13): Q, K, dO and T, last chunk first ------------------------------

    @role(warps=13, regs=32)
    @cute.jit
    def load(
        self,
        params: BpropSummaryParams,
        *,
        q: Producer,
        k: Producer,
        dout: Producer,
        tinv: Producer,
        work: WorkCursor,
    ):
        while work.valid:
            q, k, dout, tinv = self._load_item(params, work.item, q, k, dout, tinv)
            work.advance()
        cute.arch.griddepcontrol_launch_dependents()
        return q, k, dout, tinv, work

    @cute.jit
    def _load_item(self, params: BpropSummaryParams, item, q, k, dout, tinv):
        """Q, K, dO and T of every chunk of one work item, last chunk first."""
        box = params.box_layout
        piece, head, write_start, compute_end, start, end = item
        head_q = head // params.q_group
        head_k = head // params.k_group
        loads = (params.q_load, params.k_load, params.do_load)
        descs = sequence_descriptors(loads, end, head, B_T, compute_end > write_start)
        q_desc, k_desc, do_desc = descs
        for r in cutlass.range(compute_end - write_start, unroll=1):
            chunk = compute_end - 1 - r
            token = start + chunk * B_T
            rows = (token, cutlass.min(end - token, B_T))
            with k.produce() as h:
                h.access(write=True, by="tma")
                self._load_ragged(params.k_load, rows, head_k, self.d_k, params.sK, h, box, k_desc)
            with q.produce() as h:
                h.access(write=True, by="tma")
                self._load_ragged(params.q_load, rows, head_q, self.d_k, params.sQ, h, box, q_desc)
            with dout.produce() as h:
                h.access(write=True, by="tma")
                self._load_ragged(
                    params.do_load, rows, head, self.d_v, params.sdO, h, box, do_desc
                )
            with tinv.produce() as h:
                h.access(write=True, by="tma")
                src = params.mTinv[None, None, head, start // B_T + piece + chunk]
                self._load_boxes(params.tma_tinv, src, B_T, params.sTinv, h, box)
        return q, k, dout, tinv

    @cute.jit
    def _load_ragged(self, load, rows, head, cols: cutlass.Constexpr[int], ring, h, box, desc):
        """``_load_boxes`` of the tokens ``rows = (token, valid)``; later rows load as zeros."""
        token, valid = rows
        stage = ring.iterator.toint() + h.index * (cols // BOX * BOX_BYTES)
        for kb in cutlass.range_constexpr(cols // BOX):
            dst = cute.make_tensor(self._smem_ptr(stage + kb * BOX_BYTES), box)
            ragged_load(load, dst, token, valid, head, kb, h.barrier, B_T, BOX, desc)

    @cute.jit
    def _load_boxes(self, atom, src, cols: cutlass.Constexpr[int], ring, h, box):
        """TMA the 64 rows x ``cols`` columns of ``src`` into stage ``h.index`` of ``ring`` as
        ``cols / 64`` swizzled boxes of 64 x 64."""
        stage = ring.iterator.toint() + h.index * (cols // BOX * BOX_BYTES)
        for kb in cutlass.range_constexpr(cols // BOX):
            dst = cute.make_tensor(self._smem_ptr(stage + kb * BOX_BYTES), box)
            tile = cute.local_tile(src, (B_T, BOX), (0, kb))
            tSs, tSg = cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(dst, 0, 2),
                cute.group_modes(tile, 0, 2),
            )
            cute.copy(atom, tSg, tSs, tma_bar_ptr=h.barrier)

    # ---- gate (warp 14): cumulative log2 gate and its exp2 per chunk ---------------------

    @role(warps=14, regs=32)
    @cute.jit
    def gate_load(
        self, params: BpropSummaryParams, *, gate: Producer, work: WorkCursor, me: Local
    ):
        lane = me.lane
        zero = Float32(0.0)
        while work.valid:
            _piece, head, write_start, compute_end, start, end = work.item
            for r in cutlass.range(compute_end - write_start, unroll=1):
                chunk = compute_end - 1 - r
                # 64-token log2 cumsum (warp scan per 32-token column, then the carry); tokens
                # past the piece end add nothing.
                toks = [start + chunk * B_T + lane + 32 * col for col in range(2)]
                valid = [t < end for t in toks]
                raw = [params.gate[cutlass.min(t, end - 1), head] for t in toks]
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
            work.advance()
        return gate, work

    # ---- mma (warp 12): every GEMM of the reverse recurrence -----------------------------

    @role(warps=12, regs=32)
    @cute.jit
    def mma(
        self,
        params: BpropSummaryParams,
        *,
        q: Consumer,
        k: Consumer,
        dout: Consumer,
        tinv: Consumer,
        a_acc: Producer,
        do_prime: Consumer,
        a_tile: Consumer,
        trip: Producer,
        dstate: Producer,
        dstate_in: Consumer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        while work.valid:
            _piece, _head, write_start, compute_end, _start, _end = work.item
            n_chunks = compute_end - write_start
            if n_chunks > 0:
                # QK of the item's first chunk; that chunk's q/k waits own the previews.
                with a_acc.produce():
                    self._issue_qk(params, q.peek(), k.peek(), tmem)
                # Chunk r issues the QK of chunk r + 1 (peeled last chunk: no next QK).
                for r in cutlass.range(n_chunks - 1, unroll=1):
                    q, k, dout, tinv, a_acc, do_prime, a_tile, trip, dstate, dstate_in = (
                        self._mma_chunk(
                            params,
                            r > 0,
                            True,
                            q,
                            k,
                            dout,
                            tinv,
                            a_acc,
                            do_prime,
                            a_tile,
                            trip,
                            dstate,
                            dstate_in,
                            tmem,
                        )
                    )
                q, k, dout, tinv, a_acc, do_prime, a_tile, trip, dstate, dstate_in = (
                    self._mma_chunk(
                        params,
                        n_chunks > 1,
                        False,
                        q,
                        k,
                        dout,
                        tinv,
                        a_acc,
                        do_prime,
                        a_tile,
                        trip,
                        dstate,
                        dstate_in,
                        tmem,
                    )
                )
            work.advance()
        return q, k, dout, tinv, a_acc, do_prime, a_tile, trip, dstate, dstate_in, work

    def _descs(self):
        """Static geometry of the six GEMMs (TMEM accumulators are ``d_v`` rows, except QK)."""
        io, d_k, d_v = self.io_dtype, self.d_k, self.d_v

        def desc(m, n, kk, **kw):
            mn = {"a_mn_major": kw.get("atranspose", False), "b_mn_major": kw.get("btranspose")}
            idesc = instr_desc(Float32, io, io, m=m, n=n, **{k: bool(v) for k, v in mn.items()})
            return MmaDesc(M=m, N=n, K=kk, bpe_a=2, bpe_b=2, idesc=idesc, **kw)

        return (
            desc(B_T, B_T, d_k, btranspose=False),  # QK = Q @ K^T
            desc(d_v, B_T, d_k, btranspose=False),  # dV = D16 @ K^T
            desc(d_v, d_k, B_T, btranspose=True),  # D += dO'^T @ Q, D += dY'^T @ K
            desc(d_v, B_T, B_T, atranspose=True, btranspose=True),  # dU += dO^T @ A
            desc(d_v, B_T, B_T, btranspose=True),  # dY = dU @ T
        )

    @cute.jit
    def _issue_qk(self, params: BpropSummaryParams, qh, kh, tmem):
        """``QK = Q K^T`` into the ``a`` columns; ``qh``/``kh`` own or preview the stages."""
        qk, _, _, _, _ = self._descs()
        qh.access("Q", read=True, by="umma")
        kh.access("K", read=True, by="umma")
        tmem.access("a", write=True, by="umma")
        q_addr = params.sQ.iterator.toint() + qh.index * self.qk_stage_bytes
        k_addr = params.sK.iterator.toint() + kh.index * self.qk_stage_bytes
        desc_q = desc_opaque(smem_desc(q_addr, 16, 8 * 128, SWIZZLE_128B))
        desc_k = desc_opaque(smem_desc(k_addr, 16, 8 * 128, SWIZZLE_128B))
        mma_ss(qk, desc_q, desc_k, tmem.ptr("a").toint(), False)

    @cute.jit
    def _mma_chunk(
        self,
        params: BpropSummaryParams,
        have,
        has_next: cutlass.Constexpr[bool],
        q,
        k,
        dout,
        tinv,
        a_acc,
        do_prime,
        a_tile,
        trip,
        dstate,
        dstate_in,
        tmem,
    ):
        """One chunk's GEMMs in upstream's order: dV (``have``: D has a state), the Q term, dU,
        the next chunk's QK (``has_next``), dY and the K term."""
        _, dv_desc, d_desc, du_desc, dy_desc = self._descs()
        dstate_col = tmem.ptr("dstate").toint()
        dvdk_col = tmem.ptr("dvdk").toint()
        kh = k.wait()  # QK previewed this stage
        kh.access("K", read=True, by="umma")
        k_addr = params.sK.iterator.toint() + kh.index * self.qk_stage_bytes
        desc_k = desc_opaque(smem_desc(k_addr, 16, 8 * 128, SWIZZLE_128B))
        # K^T as the MN-major B operand: 64-column boxes B_T * 128 bytes apart.
        desc_kt = desc_opaque(smem_desc(k_addr, B_T * 128, 8 * 128, SWIZZLE_128B))
        # ---- dV = D16 @ K^T ----------------------------------------------------------------
        if have:
            dih = dstate_in.wait()
            th = trip.acquire()  # the dU group read the previous dU
            tmem.access("dstate_in", read=True, by="umma")
            tmem.access("dvdk", write=True, by="umma")
            mma_ts(dv_desc, tmem.ptr("dstate_in").toint(), desc_k, dvdk_col, False)
            th.commit()
            dih.release()
        # ---- D += dO'^T @ Q ----------------------------------------------------------------
        dph = do_prime.wait()
        dsh = dstate.acquire()  # the state group rescaled D
        qh = q.wait()  # QK previewed this stage
        qh.access("Q", read=True, by="umma")
        tmem.access("do_prime", read=True, by="umma")
        tmem.access("dstate", read=True, write=True, by="umma")
        q_addr = params.sQ.iterator.toint() + qh.index * self.qk_stage_bytes
        desc_qt = desc_opaque(smem_desc(q_addr, B_T * 128, 8 * 128, SWIZZLE_128B))
        mma_ts(d_desc, tmem.ptr("do_prime").toint(), desc_qt, dstate_col, have)
        qh.release()
        dph.release()
        # ---- dU (+)= dO^T @ A ----------------------------------------------------------------
        doh = dout.wait()
        ath = a_tile.wait()
        th = trip.acquire()  # the dU group rescaled dV (or read the previous chunk's dU)
        doh.access("dO", read=True, by="umma")
        ath.access("A", read=True, by="umma")
        tmem.access("dvdk", read=True, write=True, by="umma")
        do_addr = params.sdO.iterator.toint() + doh.index * self.do_stage_bytes
        desc_do = desc_opaque(smem_desc(do_addr, B_T * 128, 8 * 128, SWIZZLE_128B))
        a_addr = params.sA.iterator.toint()
        desc_a = desc_opaque(smem_desc(a_addr, (B_T // 2) * 128, 8 * 128, SWIZZLE_128B))
        mma_ss(du_desc, desc_do, desc_a, dvdk_col, have)
        th.commit()
        ath.release()
        doh.release()
        # ---- the next chunk's QK, behind the dU GEMM that read this chunk's A -----------------
        if cutlass.const_expr(has_next):
            with a_acc.produce():
                self._issue_qk(params, q.peek(), k.peek(), tmem)
        # ---- dY = dU @ T -----------------------------------------------------------------
        th = trip.acquire()  # dU is in the du columns
        tih = tinv.wait()
        tih.access("Tinv", read=True, by="umma")
        tmem.access("du", read=True, by="umma")
        tmem.access("dy", write=True, by="umma")
        t_addr = params.sTinv.iterator.toint() + tih.index * BOX_BYTES
        desc_t = desc_opaque(smem_desc(t_addr, (B_T // 2) * 128, 8 * 128, SWIZZLE_128B))
        mma_ts(dy_desc, tmem.ptr("du").toint(), desc_t, tmem.ptr("dy").toint(), False)
        th.commit()
        tih.release()
        # ---- D += dY'^T @ K ------------------------------------------------------------------
        th = trip.acquire()  # dY' is in the du columns
        tmem.access("du", read=True, by="umma")
        tmem.access("dstate", read=True, write=True, by="umma")
        mma_ts(d_desc, tmem.ptr("du").toint(), desc_kt, dstate_col, True)
        dsh.commit()
        th.commit()
        kh.release()
        return q, k, dout, tinv, a_acc, do_prime, a_tile, trip, dstate, dstate_in

    # ---- pairwise (warps 0-3): dO' and the A tile ----------------------------------------

    @role(warps=range(4), regs=208)
    @cute.jit
    def pairwise(
        self,
        params: BpropSummaryParams,
        *,
        gate: Consumer,
        dout: Consumer,
        a_acc: Consumer,
        do_prime: Producer,
        a_tile: Producer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        io, d_v = self.io_dtype, self.d_v
        t, warp, lane = me.tid, me.warp, me.lane
        lane_base = 32 * (me.tmem_row // 32)
        dv_halves = d_v // 64
        n_vals = 32
        scale = params.scale
        mask_zero = opaque_f32_zero()
        # ldmatrix.trans coordinates of this thread's dO fragment (token row, value column).
        frag_row = t % 8 + (t // 16 % 2) * 8
        if cutlass.const_expr(d_v == 128):
            frag_col = (t // 8 % 2) * 8 + (t // 32 % 2) * 32
            frag_box = (t // 64) * BOX_BYTES
        else:
            frag_col = (t // 32) * 16 + (t // 8 % 2) * 8
            frag_box = 0
        # stmatrix coordinates of this thread's A rows.
        store_row = warp * 16 + lane % 16
        store_col = (lane // 16) * 8
        do_prime_col = tmem.ptr("do_prime").toint()
        a_col = tmem.ptr("a").toint()
        while work.valid:
            _piece, _head, write_start, compute_end, start, end = work.item
            for r in cutlass.range(compute_end - write_start, unroll=1):
                chunk = compute_end - 1 - r
                n_valid = end - (start + chunk * B_T)
                # ---- pairwise decay exp2(g_i - g_j), causal -----------------------------
                gh = gate.wait()
                cs = gh.data
                col_cumsums = [
                    cs[(lane % 4) * 2 + g * 8 + b, 0] for g in range(8) for b in range(2)
                ]
                row_cumsums = [cs[warp * 16 + lane // 4 + rr * 8, 0] for rr in range(2)]
                decay_t = []
                for kv in cutlass.range_constexpr(n_vals):
                    row = warp * 16 + lane // 4 + ((kv // 2) % 2) * 8
                    col = self._frag_token(lane, kv)
                    decay_t.append(
                        cute.math.exp2(
                            row_cumsums[(kv // 2) % 2] - col_cumsums[(kv // 4) * 2 + kv % 2],
                            fastmath=True,
                        )
                        if row >= col
                        else mask_zero
                    )
                cumprod_vals = [cs[self._frag_token(lane, kv), 1] for kv in range(n_vals)]
                ah = a_acc.wait()
                # ---- dO fragments -------------------------------------------------------
                doh = dout.wait()
                doh.access("dO", read=True, by="threads")
                do_stage = params.sdO.iterator.toint() + doh.index * self.do_stage_bytes
                frag = []
                for half in cutlass.range_constexpr(dv_halves):
                    words = []
                    for block in cutlass.range_constexpr(4):
                        rr = frag_row + block * 16
                        off = rr * BOX + swizzle_xor_128b(rr, frag_col + half * 16)
                        words += ldmatrix(do_stage + frag_box + off * 2, 4, trans=True)
                    frag.append(words)
                doh.release()
                # ---- dO' = dO * cumprod * scale -> the Q term's A operand -----------------
                dph = do_prime.acquire()
                tmem.access("do_prime", write=True)
                for half in cutlass.range_constexpr(dv_halves):
                    pack = []
                    for j in cutlass.range_constexpr(16):
                        lo, hi = f16x2_to_f32(frag[half][j], dtype=io)
                        p0, p1 = fmul2(lo, hi, cumprod_vals[2 * j], cumprod_vals[2 * j + 1])
                        s0, s1 = fmul2(p0, p1, scale, scale)
                        pack.append(fp32_to_fp16(s0, s1, dtype=io))
                    if n_valid < B_T:
                        # Tokens past the piece end: upstream's dO there is zero-filled.
                        pack = self._mask_tokens(pack, lane, n_valid)
                    tmem_st("16x128b", ((lane_base + half * 16) << 16) + do_prime_col, pack)
                dph.commit()
                # ---- A = QK * decay * scale -> SMEM ---------------------------------------
                tmem.access("a", read=True)
                a_vec = tmem_ld("16x256b", 8, (lane_base << 16) + a_col)
                cute.arch.fence_view_async_tmem_load()
                a_pack = []
                for kk in cutlass.range_constexpr(n_vals // 2):
                    p0, p1 = fmul2(
                        a_vec[2 * kk], a_vec[2 * kk + 1], decay_t[2 * kk], decay_t[2 * kk + 1]
                    )
                    s0, s1 = fmul2(p0, p1, scale, scale)
                    a_pack.append(fp32_to_fp16(s0, s1, dtype=io))
                ah.release()
                if n_valid < B_T:
                    # Rows past the piece end: upstream's Q there is zero-filled. Their columns
                    # are causal-masked once these rows are.
                    for kk in cutlass.range_constexpr(n_vals // 2):
                        row = warp * 16 + lane // 4 + (kk % 2) * 8
                        a_pack[kk] = cutlass.select_(row < n_valid, a_pack[kk], Int32(0))
                ath = a_tile.acquire()
                ath.access("A", write=True, by="threads")
                a_base = params.sA.iterator.toint()
                for c in cutlass.range_constexpr(B_T // 16):
                    off = store_row * B_T + swizzle_xor_128b(store_row, store_col + c * 16)
                    stmatrix(a_base + off * 2, a_pack[c * 4 : c * 4 + 4])
                ath.commit()
                gh.release()
            work.advance()
        return gate, dout, a_acc, do_prime, a_tile, work

    # ---- du (warps 4-7): dV rescale, dU and dY' ------------------------------------------

    @role(warps=range(4, 8), regs=144)
    @cute.jit
    def du(
        self,
        params: BpropSummaryParams,
        *,
        gate: Consumer,
        trip: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        io, d_v = self.io_dtype, self.d_v
        lane = me.lane
        lane_base = 32 * (me.tmem_row // 32)
        dv_halves = d_v // 64
        n_vals = 32
        dvdk_col = tmem.ptr("dvdk").toint()
        du_col = tmem.ptr("du").toint()
        dy_col = tmem.ptr("dy").toint()
        while work.valid:
            _piece, _head, write_start, compute_end, start, end = work.item
            for r in cutlass.range(compute_end - write_start, unroll=1):
                chunk = compute_end - 1 - r
                n_valid = end - (start + chunk * B_T)
                gh = gate.wait()
                cs = gh.data
                cumprod_vals = [cs[self._frag_token(lane, kv), 1] for kv in range(n_vals)]
                # ---- dV *= exp2(g[63] - g_j), in place --------------------------------------
                if r > 0:
                    last_cumsumlog = cs[B_T - 1, 0]
                    col_cumsumlog = [
                        cs[(lane % 4) * 2 + g * 8 + b, 0] for g in range(8) for b in range(2)
                    ]
                    decay_scale = [
                        cute.math.exp2(last_cumsumlog - col_cumsumlog[i], fastmath=True)
                        for i in range(16)
                    ]
                    decay_vals = [decay_scale[(kv // 4) * 2 + kv % 2] for kv in range(n_vals)]
                    th = trip.wait()
                    tmem.access("dvdk", read=True, write=True)
                    dv = [
                        tmem_ld("16x256b", 8, ((lane_base + half * 16) << 16) + dvdk_col)
                        for half in range(dv_halves)
                    ]
                    cute.arch.fence_view_async_tmem_load()
                    for half in cutlass.range_constexpr(dv_halves):
                        scaled = []
                        for j in cutlass.range_constexpr(16):
                            s0, s1 = fmul2(
                                dv[half][2 * j],
                                dv[half][2 * j + 1],
                                decay_vals[2 * j],
                                decay_vals[2 * j + 1],
                            )
                            scaled += [s0, s1]
                        tmem_st("16x256b", ((lane_base + half * 16) << 16) + dvdk_col, scaled)
                    th.release()
                # ---- dU -> 16-bit, the dY GEMM's A operand ----------------------------------
                th = trip.wait()
                tmem.access("dvdk", read=True)
                tmem.access("du", write=True)
                for half in cutlass.range_constexpr(dv_halves):
                    acc = tmem_ld("16x256b", 8, ((lane_base + half * 16) << 16) + dvdk_col)
                    cute.arch.fence_view_async_tmem_load()
                    pack = [fp32_to_fp16(acc[2 * j], acc[2 * j + 1], dtype=io) for j in range(16)]
                    if n_valid < B_T:
                        # Tokens past the piece end: upstream's dU there is zero (zero K rows).
                        pack = self._mask_tokens(pack, lane, n_valid)
                    tmem_st("16x128b", ((lane_base + half * 16) << 16) + du_col, pack)
                th.release()
                # ---- dY' = -cumprod * dY -> 16-bit, the K term's A operand -------------------
                th = trip.wait()
                tmem.access("dy", read=True)
                tmem.access("du", write=True)
                dy = [
                    tmem_ld("16x256b", 8, ((lane_base + half * 16) << 16) + dy_col)
                    for half in range(dv_halves)
                ]
                cute.arch.fence_view_async_tmem_load()
                neg_one = Float32(-1.0)
                cumprod_neg = [cumprod_vals[kv] * neg_one for kv in range(n_vals)]
                for half in cutlass.range_constexpr(dv_halves):
                    dyp = []
                    for j in cutlass.range_constexpr(16):
                        n0, n1 = fmul2(
                            dy[half][2 * j],
                            dy[half][2 * j + 1],
                            cumprod_neg[2 * j],
                            cumprod_neg[2 * j + 1],
                        )
                        dyp += [n0, n1]
                    pack = [fp32_to_fp16(dyp[2 * j], dyp[2 * j + 1], dtype=io) for j in range(16)]
                    tmem_st("16x128b", ((lane_base + half * 16) << 16) + du_col, pack)
                th.release()
                gh.release()
                # ---- K term issued: the next dV may overwrite dvdk --------------------------
                th = trip.wait()
                th.release()
            work.advance()
        return gate, trip, work

    # ---- state (warps 8-11): D16 and the decay rescale, then the summary store ------------

    @role(warps=range(8, 12), regs=128)
    @cute.jit
    def state(
        self,
        params: BpropSummaryParams,
        *,
        gate: Consumer,
        dstate: Consumer,
        dstate_in: Producer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        io, d_k = self.io_dtype, self.d_k
        t = me.tmem_row
        lane0 = (32 * (t // 32)) << 16
        n_ld = d_k // 32
        if cutlass.const_expr(self.d_v == 128):
            state_row = t
            row_valid = cutlass.Boolean(True)
        else:
            # M = 64 GEMMs fill TMEM lanes 0-15 of each warp's 32.
            state_row = (t // 32) * 16 + t % 16
            row_valid = (t % 32) < 16
        dstate_col = lane0 + tmem.ptr("dstate").toint()
        dstate_in_col = lane0 + tmem.ptr("dstate_in").toint()
        out = params.state_g
        while work.valid:
            piece, head, write_start, compute_end, _start, _end = work.item
            n_chunks = compute_end - write_start
            if n_chunks > 0:
                # The last chunk's gate seeds nothing (D starts from zero).
                gh = gate.wait()
                gh.release()
                for _ in cutlass.range(n_chunks - 1, unroll=1):
                    # ---- the next chunk's D16, then D *= that chunk's total decay ----------
                    dh = dstate.wait()
                    ih = dstate_in.acquire()
                    gh = gate.wait()
                    cumprod_top = gh.data[B_T - 1, 1]
                    tmem.access("dstate", read=True, write=True)
                    tmem.access("dstate_in", write=True)
                    for i in cutlass.range_constexpr(n_ld):
                        vec = tmem_ld("32x32b", 32, dstate_col + i * 32)
                        cute.arch.fence_view_async_tmem_load()
                        pack = [
                            fp32_to_fp16(vec[2 * j], vec[2 * j + 1], dtype=io) for j in range(16)
                        ]
                        tmem_st("32x32b", dstate_in_col + i * 16, pack)
                        scaled = []
                        for j in cutlass.range_constexpr(16):
                            s0, s1 = fmul2(vec[2 * j], vec[2 * j + 1], cumprod_top, cumprod_top)
                            scaled += [s0, s1]
                        tmem_st("32x32b", dstate_col + i * 32, scaled)
                    ih.commit()
                    dh.release()
                    gh.release()
                # ---- G: TMEM -> GMEM --------------------------------------------------------
                dh = dstate.wait()
                tmem.access("dstate", read=True)
                if write_start == 0:
                    row_addr = (out.iterator + out.layout((piece, head, state_row, 0))).toint()
                    for i in cutlass.range_constexpr(n_ld):
                        vals = tmem_ld("32x32b", 32, dstate_col + i * 32)
                        cute.arch.fence_view_async_tmem_load()
                        if row_valid:
                            for g in cutlass.range_constexpr(8):
                                st_global_v4_f32(
                                    row_addr + (i * 32 + g * 4) * 4, *vals[g * 4 : g * 4 + 4]
                                )
                dh.release()
            elif write_start == 0:
                # An item without chunks: G = 0.
                for c in cutlass.range(d_k):
                    if row_valid:
                        out[piece, head, state_row, c] = Float32(0.0)
            work.advance()
        return gate, dstate, dstate_in, work

    # ---- kernel and entry point --------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        q_load,
        k_load,
        do_load,
        tma_tinv,
        mTinv,
        gate,
        state_g,
        layouts,
        box_layout,
        q_group,
        k_group,
        scale,
        bound,
    ):
        # The work table and T tiles come from the preceding launches.
        cute.arch.griddepcontrol_wait()
        q_layout, k_layout, do_layout, square_layout = layouts
        self.begin_roles(
            work=bound,
            layouts={
                "Q": q_layout,
                "K": k_layout,
                "dO": do_layout,
                "Tinv": square_layout,
                "A": square_layout,
            },
        )
        params = BpropSummaryParams(
            q_load=q_load,
            k_load=k_load,
            do_load=do_load,
            tma_tinv=tma_tinv,
            mTinv=mTinv,
            gate=gate,
            state_g=state_g,
            sQ=self.buffer("Q"),
            sK=self.buffer("K"),
            sdO=self.buffer("dO"),
            sTinv=self.buffer("Tinv"),
            sA=self.buffer("A"),
            box_layout=box_layout,
            q_group=q_group,
            k_group=k_group,
            scale=scale,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (tokens, query heads, d_k)
        k: cute.Tensor,  # (tokens, key heads, d_k)
        gate: cute.Tensor,  # (tokens, heads) natural-log gate
        do: cute.Tensor,  # (tokens, heads, d_v)
        tinv: cute.Tensor,  # (tinv rows, heads, 64, 64)
        state_g: cute.Tensor,  # (pieces, heads, d_v, d_k) fp32
        work_items: cute.Tensor,  # (capacity, 10) int32
        work_count: cute.Tensor,  # (1,) int32
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        scale: Float32,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.launch(
            q, k, gate, do, tinv, state_g, work_items, work_count, tma_slots, scale, stream
        )

    @cute.jit
    def launch(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        gate: cute.Tensor,
        do: cute.Tensor,
        tinv: cute.Tensor,
        state_g: cute.Tensor,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        tma_slots: cute.Tensor,
        scale: Float32,
        stream: cuda.CUstream,
    ):
        """Launch one CTA per work-table row (the first ``work_count[0]`` run); callable from
        another ``@cute.jit`` host, such as a fused launch bundle. It is a programmatic dependent
        launch: every thread waits for the preceding kernel before its roles start."""
        if cutlass.const_expr(state_g.element_type is not Float32):
            raise TypeError(f"the GDN bprop summary writes fp32, got {state_g.element_type}")
        io = self.io_dtype
        heads = Int32(gate.shape[1])
        q_group = heads // Int32(q.shape[1])
        k_group = heads // Int32(k.shape[1])
        box = sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, BOX), 1)
        box_layout = cute.select(box, mode=[0, 1])
        layouts = (
            cute.make_layout((self.qk_stage_bytes // 2, Q_STAGES)),
            cute.make_layout((self.qk_stage_bytes // 2, K_STAGES)),
            cute.make_layout((self.do_stage_bytes // 2, DO_STAGES)),
            cute.make_layout((BOX_BYTES // 2, 1)),
        )
        # (token, d, head) and (i, j, head, row) TMA views.
        mQ = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[0, 2, 1]))
        mK = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[0, 2, 1]))
        mdO = cute.make_tensor(do.iterator, cute.select(do.layout, mode=[0, 2, 1]))
        mTinv = cute.make_tensor(tinv.iterator, cute.select(tinv.layout, mode=[2, 3, 1, 0]))
        load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        q_load, k_load, do_load = (
            make_ragged_tma(
                load_op, t, box_layout, B_T, BOX, slot_region(tma_slots, i, B_T, heads)
            )
            for i, t in enumerate((mQ, mK, mdO))
        )
        tma_tinv, mTinv_t = cpasync.make_tiled_tma_atom(load_op, mTinv, box_layout, (B_T, B_T))
        count = cute.make_tensor(work_count.iterator, cute.make_layout(1))
        bound = self.work.bind((work_items, count), limit=LastEntry(count))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            q_load,
            k_load,
            do_load,
            tma_tinv,
            mTinv_t,
            gate,
            state_g,
            layouts,
            box_layout,
            q_group,
            k_group,
            scale,
            bound,
        ).launch(**self.launch_config(bound), stream=stream, use_pdl=True)


def bundle_hook(io_dtype, d_k: int, d_v: int, log_gate: bool) -> GdnBpropSummary:
    """The bprop summary op for launch bundles that run it inside their own ``@cute.jit`` host:
    ``op.launch(q, k, gate, do, tinv, state_g, work_items, work_count, tma_slots, scale,
    stream)``."""
    if not log_gate:
        raise ValueError("the GDN bprop summary takes natural-log gates")
    name = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}[io_dtype]
    return GdnBpropSummary(name, d_k, d_v)


@jit_cache
def compile_gdn_bprop_summary(op: GdnBpropSummary):
    """Compile for dynamic tokens, heads, pieces, tile rows and work-table capacity."""
    io = op.io_dtype
    tokens, key_heads, query_heads, heads, pieces = (cute.sym_int() for _ in range(5))
    i32 = cutlass.Int32
    return compile_tvm_ffi(
        op,
        make_fake_strided_tensor(io, (tokens, query_heads, op.d_k), stride_divisibility=8),
        make_fake_strided_tensor(io, (tokens, key_heads, op.d_k), stride_divisibility=8),
        make_fake_strided_tensor(Float32, (tokens, heads), contiguous_dim=None),
        make_fake_strided_tensor(io, (tokens, heads, op.d_v), stride_divisibility=8),
        make_fake_strided_tensor(io, (cute.sym_int(), heads, B_T, B_T), stride_divisibility=8),
        make_fake_strided_tensor(
            Float32, (pieces, heads, op.d_v, op.d_k), stride_divisibility=op.d_k
        ),
        make_fake_strided_tensor(i32, (cute.sym_int(), WORK_ITEM_FIELDS)),
        make_fake_strided_tensor(i32, (1,)),
        fake_tma_slots(),
        Float32(1.0),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


def gdn_bprop_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    do: torch.Tensor,
    tinv: torch.Tensor,
    state_g: torch.Tensor,
    work_items: torch.Tensor,
    work_count: torch.Tensor,
    *,
    scale: float,
) -> None:
    """Fill ``state_g`` for the work table (see the module docstring for the layouts); ``tinv``
    holds ``frost.gdn_tinv``'s tiles over the same pieces."""
    if q.dtype not in _IO_DTYPES or {k.dtype, do.dtype, tinv.dtype} != {q.dtype}:
        raise TypeError(f"q, k, do and tinv must share a 16-bit dtype, got {q.dtype}/{k.dtype}")
    if state_g.dtype != torch.float32:
        raise TypeError("the bprop summaries are fp32")
    if gate.dtype != torch.float32:
        raise TypeError(f"gate must be fp32, got {gate.dtype}")
    heads = gate.shape[1]
    if heads % q.shape[1] or heads % k.shape[1] or do.shape[1] != heads:
        raise ValueError("heads must match dO and be a multiple of the query and key head counts")
    if not state_g.is_contiguous() or state_g.data_ptr() % 16:
        raise ValueError("the bprop summaries must be contiguous and 16-byte aligned")
    op = GdnBpropSummary(_IO_DTYPES[q.dtype], q.shape[-1], do.shape[-1])
    fn = compile_gdn_bprop_summary(op)
    workspace, sms = op.work.launch_args(q.device)
    slots = new_tma_slots(op.tma_slot_rows(heads), q.device)
    fn(q, k, gate, do, tinv, state_g, work_items, work_count, slots, scale, workspace, sms)
