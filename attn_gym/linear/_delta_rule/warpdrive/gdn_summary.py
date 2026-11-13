# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# gdn_summary_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and channels
# with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence descriptor
# arrays, one CTA runs one work item, and the natural-log gate with zero-seeded H is the only form.

"""GDN piece summaries for SM100: per (piece, head) work item, the state a piece produces from a
zero seed (``H``) and its transition (``M``), the two recurrences the exact piece chain composes::

    X_end = X_start @ M + H        (states [V, K]; M stored as M^T in the K x K buffer)

Per chunk ``c`` (tokens ``[c * 64, (c + 1) * 64)`` of the piece) and chain ``X`` in {H, M}, with
``g`` the log2 gate cumsum, ``T`` the beta-folded chunk inverse of ``gdn_tinv`` and the
state ``S_X`` held transposed in TMEM::

    KS   = K @ S_X                          Y_H = V - exp2(g) * KS,  Y_M = -exp2(g) * KS
    U    = T @ Y                            S_X = exp2(g[63]) * S_X + K^T @ (exp2(g[63] - g) * U)

``H`` starts from zero (its first chunk has no ``KS``) and ``M`` from the identity. Items are the
first ``work_count[0]`` rows of a split-K work table, ``work_items[i] = (piece, head, -, chunk_end,
chunk_begin, -, start, end, ...)``: chunks ``[chunk_begin, chunk_end)`` of the piece at tokens
``[start, end)``. ``M`` is seeded only at ``chunk_begin == 0`` (otherwise both chains start from
zero), and the states land at ``state_h[piece, head]`` (``[V, K]``) and ``state_m[piece, head]``
(``[K, K]``) when ``chunk_end`` is the piece's last chunk; an item without chunks writes ``H = 0``
and ``M = I``. The chain prologue's tables cover whole pieces, ``(0, num_chunks)``.

K and V rows past the piece's end load as zeros (``ragged.RaggedTma``), as from
upstream's per-sequence descriptors; ``T``'s rows and columns past the end are zero as well.

Roles and channels (12 warps; warp 11 pads the register warpgroup)::

    load (9)   --k:    3 x K [64, d_k] (TMA)-->      mma (10)
    load       --tinv: 3 x T [64, 64] (TMA)-->       mma
    load       --v:    3 x V [64, d_v] (TMA)-->      chain_h (4-7)
    gate (8)   --gate: 3 x log2 cumsum, cumprod-->   chain_m (0-3) + chain_h
    mma        <-trip_m: 1 (UMMA / threads)->        chain_m
    mma        <-trip_h: 1 (UMMA / threads)->        chain_h

``trip_X`` is a round trip per epoch: the MMA warp acquires it once the chain group released the
previous epoch (its TMEM operand for the next GEMM is written), issues one GEMM and commits; the
chain group waits for that result, reads it, writes the next operand and releases. Per item it
carries one seed epoch (no GEMM) and three per chunk: ``KS`` (no GEMM before ``X`` has a state),
``U`` and the state update, whose wait at the next chunk (or the item end) yields the state.
"""

import functools
from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.arch import inline_ptx
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.linear._delta_rule.wd_support import HEAD_DIMS, IO_DTYPE_NAMES
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
    fadd2,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    opaque_f32_zero,
    opaque_i32_zero,
    shfl_idx,
    shfl_up,
    st_global_v4_f32,
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
from .umma import SWIZZLE_128B, MmaDesc, desc_opaque, instr_desc, mma_ts, smem_desc

B_T = 64
RCP_LN2 = 1.4426950408889634  # 1 / ln 2: natural-log gates -> the kernel's log2 domain
STAGES = 3  # K, V, T and gate rings
WORK_ITEM_FIELDS = 10

_CUTE_DTYPES = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16}


@dataclass(frozen=True)
class SummaryItems(WorkMap):
    """Rows of the chain prologue's work table: ``(piece, head, chunk_begin, chunk_end, start,
    end)``, chunks counted from the piece's first token ``start``."""

    words = 6

    def count64(self, space):
        table, _ = space
        return Int64(table.shape[0])

    def decode(self, index, space, state):
        table, _ = space
        row = (table[index, 0], table[index, 1], table[index, 4], table[index, 3])
        return True, (*row, table[index, 6], table[index, 7]), state

    def get_key(self) -> str:
        return "gdn_summary_items"


@dataclass(frozen=True)
class SummaryParams:
    k_load: RaggedTma  # (token, d_k, key head) loads, zero past each piece end
    v_load: RaggedTma  # (token, d_v, value head)
    tma_tinv: cute.CopyAtom
    mTinv: cute.Tensor  # TMA view (i, j, head, row)
    gate: cute.Tensor  # (token, head) natural-log gate
    state_h: cute.Tensor  # (piece, head, d_v, d_k) fp32
    state_m: cute.Tensor  # (piece, head, d_k, d_k) fp32
    sK: cute.Tensor  # whole rings (iterator = stage 0)
    sV: cute.Tensor
    sTinv: cute.Tensor
    box_layout: cute.ComposedLayout  # one 64 x 64 swizzled box: every TMA destination
    key_group: Int32  # heads per key head
    value_group: Int32  # heads per value head


class GdnSummary(WarpSpecialized):
    """Warpdrive GDN piece summaries for ``dtype_name`` fp16/bf16 operands, ``d_k``/``d_v`` 64 or
    128 and fp32 states."""

    def __init__(self, dtype_name: str, d_k: int, d_v: int):
        if d_k not in HEAD_DIMS or d_v not in HEAD_DIMS:
            raise ValueError(f"the GDN summary serves d_k, d_v in {HEAD_DIMS}, got {d_k}, {d_v}")
        self.dtype_name, self.d_k, self.d_v = dtype_name, d_k, d_v
        self.io_dtype = _CUTE_DTYPES[dtype_name]
        self.bpe = bpe = self.io_dtype.width // 8
        self.smem_plan = SmemPlan()
        k_buf = self.smem_plan.buffer("K", self.io_dtype, align=1024)
        v_buf = self.smem_plan.buffer("V", self.io_dtype, align=1024)
        tinv_buf = self.smem_plan.buffer("Tinv", self.io_dtype, align=1024)
        # The TMA check reports these rings unverified: it cannot follow copies through the
        # sequence-offset (domain_offset) views. Each stage takes d / 64 boxes of 64 x 64.
        self.k = Ring(STAGES, Tma(B_T * d_k * bpe), Umma, buffer=k_buf)
        self.tinv = Ring(STAGES, Tma(B_T * B_T * bpe), Umma, buffer=tinv_buf)
        self.v = Ring(STAGES, Tma(B_T * d_v * bpe), Threads, buffer=v_buf)
        self.gate = Ring(STAGES, Threads, Threads, payload=(Float32, (B_T, 2)))
        # Each release hands the MMA warp TMEM the chain group loaded from and stored to.
        trip_fence = ("tmem_load", "tmem_store")
        self.trip_m = Ring(1, Umma, Threads, release_fence=trip_fence)
        self.trip_h = Ring(1, Umma, Threads, release_fence=trip_fence)
        self.tmem_plan = (
            TmemPlan()
            .region("state_h", d_k, 1)
            .region("state_m", d_k, 1)
            .region("in_h", d_k // 2, 1)  # 16-bit GEMM A operands: state, Y, decayed U
            .region("in_m", d_k // 2, 1)
            .region("acc_h", B_T, 1)  # KS, then U
            .region("acc_m", B_T, 1)
        )
        self.work = WorkStream(mapping=SummaryItems(), schedule=OneShot())

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: K and V, keyed by head."""
        return tma_slot_rows(2, B_T, heads)

    def get_name(self) -> str:
        return f"wd_gdn_summary_{self.dtype_name}_dk{self.d_k}_dv{self.d_v}"

    # ---- roles -------------------------------------------------------------------------

    @role(warps=9, regs=24)
    @cute.jit
    def load(
        self,
        params: SummaryParams,
        *,
        k: Producer,
        tinv: Producer,
        v: Producer,
        work: WorkCursor,
    ):
        while work.valid:
            k, tinv, v = self._load_item(params, work.item, k, tinv, v)
            work.advance()
        return k, tinv, v, work

    @cute.jit
    def _load_item(self, params: SummaryParams, item, k, tinv, v):
        """K, T and V of every chunk of one work item."""
        box = params.box_layout
        piece, head, chunk_begin, chunk_end, start, end = item
        head_k = head // params.key_group
        head_v = head // params.value_group
        loads = (params.k_load, params.v_load)
        descs = sequence_descriptors(loads, end, head, B_T, chunk_end > chunk_begin)
        k_desc, v_desc = descs
        for chunk in cutlass.range(chunk_begin, chunk_end):
            token = start + chunk * B_T
            rows = (token, cutlass.min(end - token, B_T))
            with k.produce() as h:
                h.access(write=True, by="tma")
                self._load_ragged(params.k_load, rows, head_k, self.d_k, params.sK, h, box, k_desc)
            with tinv.produce() as h:
                h.access(write=True, by="tma")
                src = params.mTinv[None, None, head, start // B_T + piece + chunk]
                self._load_boxes(params.tma_tinv, src, B_T, params.sTinv, h, box)
            with v.produce() as h:
                h.access(write=True, by="tma")
                self._load_ragged(params.v_load, rows, head_v, self.d_v, params.sV, h, box, v_desc)
        return k, tinv, v

    @cute.jit
    def _load_ragged(self, load, rows, head, cols: cutlass.Constexpr[int], ring, h, box, desc):
        """``_load_boxes`` of the tokens ``rows = (token, valid)``; later rows load as zeros."""
        token, valid = rows
        box_bytes = B_T * 64 * self.bpe
        stage = ring.iterator.toint() + h.index * (cols // 64 * box_bytes)
        for kb in cutlass.range_constexpr(cols // 64):
            dst = cute.make_tensor(self._smem_ptr(stage + kb * box_bytes), box)
            ragged_load(load, dst, token, valid, head, kb, h.barrier, B_T, 64, desc)

    @cute.jit
    def _load_boxes(self, atom, src, cols: cutlass.Constexpr[int], ring, h, box):
        """TMA the 64 rows x ``cols`` columns of ``src`` into stage ``h.index`` of ``ring`` as
        ``cols / 64`` swizzled boxes of 64 x 64."""
        box_bytes = B_T * 64 * self.bpe
        stage = ring.iterator.toint() + h.index * (cols // 64 * box_bytes)
        for kb in cutlass.range_constexpr(cols // 64):
            dst = cute.make_tensor(self._smem_ptr(stage + kb * box_bytes), box)
            tile = cute.local_tile(src, (B_T, 64), (0, kb))
            tSs, tSg = cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(dst, 0, 2),
                cute.group_modes(tile, 0, 2),
            )
            cute.copy(atom, tSg, tSs, tma_bar_ptr=h.barrier)

    @role(warps=8, regs=24)
    @cute.jit
    def gate_load(self, params: SummaryParams, *, gate: Producer, work: WorkCursor, me: Local):
        lane = me.lane
        zero = Float32(0.0)
        while work.valid:
            _piece, head, chunk_begin, chunk_end, start, end = work.item
            for chunk in cutlass.range(chunk_begin, chunk_end):
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

    @role(warps=10, regs=24)
    @cute.jit
    def mma(
        self,
        params: SummaryParams,
        *,
        k: Consumer,
        tinv: Consumer,
        trip_h: Producer,
        trip_m: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        io, d_k, d_v, bpe = self.io_dtype, self.d_k, self.d_v, self.bpe
        # Rows are d_v for chain H and d_k for chain M (states are held transposed).
        id_h = instr_desc(Float32, io, io, m=d_v, n=B_T)
        id_m = instr_desc(Float32, io, io, m=d_k, n=B_T)
        ks_h = MmaDesc(M=d_v, N=B_T, K=d_k, bpe_a=bpe, bpe_b=bpe, idesc=id_h)
        ks_m = MmaDesc(M=d_k, N=B_T, K=d_k, bpe_a=bpe, bpe_b=bpe, idesc=id_m)
        u_h = MmaDesc(M=d_v, N=B_T, K=B_T, bpe_a=bpe, bpe_b=bpe, idesc=id_h)
        u_m = MmaDesc(M=d_k, N=B_T, K=B_T, bpe_a=bpe, bpe_b=bpe, idesc=id_m)
        id_kv_h = instr_desc(Float32, io, io, m=d_v, n=d_k, b_mn_major=True)
        id_kv_m = instr_desc(Float32, io, io, m=d_k, n=d_k, b_mn_major=True)
        kv_h = MmaDesc(M=d_v, N=d_k, K=B_T, bpe_a=bpe, bpe_b=bpe, idesc=id_kv_h, btranspose=True)
        kv_m = MmaDesc(M=d_k, N=d_k, K=B_T, bpe_a=bpe, bpe_b=bpe, idesc=id_kv_m, btranspose=True)
        state_h, state_m = tmem.ptr("state_h").toint(), tmem.ptr("state_m").toint()
        in_h, in_m = tmem.ptr("in_h").toint(), tmem.ptr("in_m").toint()
        acc_h, acc_m = tmem.ptr("acc_h").toint(), tmem.ptr("acc_m").toint()
        k_base = params.sK.iterator.toint()
        tinv_base = params.sTinv.iterator.toint()
        while work.valid:
            _piece, _head, chunk_begin, chunk_end, _start, _end = work.item
            n_local = chunk_end - chunk_begin
            seed_m = chunk_begin == 0
            if n_local > 0:
                # Seed epochs: no GEMM; each chain group's release hands over its first operand.
                with trip_h.produce():
                    pass
                with trip_m.produce():
                    pass
            for local in cutlass.range(n_local):
                have_h = local > 0
                have_m = local > 0 or seed_m
                with k.consume() as kh:
                    kh.access(read=True, by="umma")
                    k_addr = k_base + kh.index * (B_T * d_k * bpe)
                    desc_k = desc_opaque(smem_desc(k_addr, 16, 8 * 128, SWIZZLE_128B))
                    # K^T as the MN-major B operand: 64-column boxes B_T * 128 bytes apart.
                    desc_kt = desc_opaque(smem_desc(k_addr, B_T * 128, 8 * 128, SWIZZLE_128B))
                    # KS = state(T) @ K^T, for chains that have a state.
                    with trip_h.produce():
                        if have_h:
                            self._umma_access(tmem, "h", "acc")
                            mma_ts(ks_h, in_h, desc_k, acc_h, False)
                    with trip_m.produce():
                        if have_m:
                            self._umma_access(tmem, "m", "acc")
                            mma_ts(ks_m, in_m, desc_k, acc_m, False)
                    # U = Y(T) @ T^-1.
                    with tinv.consume() as th:
                        th.access(read=True, by="umma")
                        t_addr = tinv_base + th.index * (B_T * B_T * bpe)
                        desc_t = desc_opaque(smem_desc(t_addr, 16, 8 * 128, SWIZZLE_128B))
                        with trip_h.produce():
                            self._umma_access(tmem, "h", "acc")
                            mma_ts(u_h, in_h, desc_t, acc_h, False)
                        with trip_m.produce():
                            self._umma_access(tmem, "m", "acc")
                            mma_ts(u_m, in_m, desc_t, acc_m, False)
                    # state += decayed U(T) @ K.
                    with trip_h.produce():
                        self._umma_access(tmem, "h", "state")
                        mma_ts(kv_h, in_h, desc_kt, state_h, have_h)
                    with trip_m.produce():
                        self._umma_access(tmem, "m", "state")
                        mma_ts(kv_m, in_m, desc_kt, state_m, have_m)
            work.advance()
        return k, tinv, trip_h, trip_m, work

    @staticmethod
    def _umma_access(tmem, chain: str, out: str):
        """Annotate one GEMM of chain ``chain``: A from ``in_<chain>``, D in ``<out>_<chain>``."""
        tmem.access("in_" + chain, read=True, by="umma")
        tmem.access(out + "_" + chain, read=out == "state", write=True, by="umma")

    @role(warps=range(4), regs=240)
    @cute.jit
    def chain_m(
        self,
        params: SummaryParams,
        *,
        gate: Consumer,
        trip_m: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        gate, trip_m, _, work = self._chain(False, params, gate, trip_m, None, work, tmem, me)
        return gate, trip_m, work

    @role(warps=range(4, 8), regs=240)
    @cute.jit
    def chain_h(
        self,
        params: SummaryParams,
        *,
        gate: Consumer,
        trip_h: Consumer,
        v: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        gate, trip_h, v, work = self._chain(True, params, gate, trip_h, v, work, tmem, me)
        return gate, trip_h, v, work

    @cute.jit
    def _chain(self, is_h: cutlass.Constexpr[bool], params, gate, trip, v, work, tmem, me):
        """One chain group: owns chain ``X``'s state from its seed to the final store. Chain H
        (``is_h``) seeds zero and builds Y from V; chain M seeds the identity with V = 0."""
        io, d_k, d_v, bpe = self.io_dtype, self.d_k, self.d_v, self.bpe
        rows = d_v if is_h else d_k
        row_halves = rows // 64
        n_ld = d_k // 32
        t = me.tmem_row  # this thread's TMEM lane
        lane = me.lane
        v_row = t % 8 + (t // 16 % 2) * 8
        if cutlass.const_expr(rows == 128):
            v_col = (t // 8 % 2) * 8 + (t // 32 % 2) * 32
            v_segment = (t // 64) * (B_T * 64)
            state_row = t
            row_valid = cutlass.Boolean(True)
        else:
            # M = 64 GEMMs fill TMEM lanes 0-15 of each warp's 32.
            v_col = (t // 32) * 16 + (t // 8 % 2) * 8
            v_segment = 0
            state_row = (t // 32) * 16 + t % 16
            row_valid = (t % 32) < 16
        lane_base = (32 * (t // 32)) << 16
        name = "h" if is_h else "m"
        state_col = lane_base + tmem.ptr("state_" + name).toint()
        in_col = lane_base + tmem.ptr("in_" + name).toint()
        acc_col = lane_base + tmem.ptr("acc_" + name).toint()
        state_out = params.state_h if is_h else params.state_m
        v_base = params.sV.iterator.toint()
        n_vals = 32
        # Gate-table row of value k of a 16x256b TMEM load (its token within the chunk).
        frag_tok = [(lane % 4) * 2 + ((kv // 4) * 8 + kv % 2) for kv in range(n_vals)]
        while work.valid:
            piece, head, chunk_begin, chunk_end, start, end = work.item
            num_chunks = (end - start + (B_T - 1)) // B_T
            n_local = chunk_end - chunk_begin
            seed = chunk_begin == 0
            if n_local > 0:
                if cutlass.const_expr(not is_h):  # noqa: SIM102 (static, then dynamic)
                    if seed:
                        # Identity seed; the opaque one is bound to the item so it is not hoisted.
                        one = inline_ptx(
                            "mov.b32 $0, $1;",
                            write_only_types=[Float32],
                            read_only_args=[opaque_f32_zero() + Float32(1.0), piece],
                        )
                        tmem.access("state_" + name, write=True)
                        for i in cutlass.range_constexpr(n_ld):
                            words = [
                                one if state_row == i * 32 + c else Float32(0.0) for c in range(32)
                            ]
                            tmem_st("32x32b", state_col + i * 32, words)
                        cute.arch.fence_view_async_tmem_store()
                for local in cutlass.range(n_local):
                    if cutlass.const_expr(is_h):
                        valid_state = local > 0
                    else:
                        # Runtime even in the chain plans (every item seeds): a constant lets
                        # ptxas hoist the rescale math above the state release, which delays the
                        # KS GEMM (10% slower).
                        valid_state = local > 0 or seed
                    gh = gate.wait()
                    cumprod_total = gh.data[B_T - 1, 1]

                    # ---- state restage (16-bit GEMM operand), then rescale ---------------
                    # The seed epoch or the previous state update. A chain without a state loads
                    # and drops the stale columns.
                    with trip.consume():
                        tmem.access("state_" + name, read=True)
                        state_vecs = [
                            tmem_ld("32x32b", 32, state_col + i * 32) for i in range(n_ld)
                        ]
                        cute.arch.fence_view_async_tmem_load()
                        if valid_state:
                            tmem.access("in_" + name, write=True)
                            for i in cutlass.range_constexpr(n_ld):
                                pack = [
                                    fp32_to_fp16(
                                        state_vecs[i][2 * j], state_vecs[i][2 * j + 1], dtype=io
                                    )
                                    for j in range(16)
                                ]
                                tmem_st("32x32b", in_col + i * 16, pack)
                    if valid_state:
                        # The next state update reads this rescale after a later release.
                        tmem.access("state_" + name, write=True)
                        for i in cutlass.range_constexpr(n_ld):
                            scaled = []
                            for j in cutlass.range_constexpr(16):
                                s0, s1 = fmul2(
                                    state_vecs[i][2 * j],
                                    state_vecs[i][2 * j + 1],
                                    cumprod_total,
                                    cumprod_total,
                                )
                                scaled += [s0, s1]
                            tmem_st("32x32b", state_col + i * 32, scaled)
                        cute.arch.fence_view_async_tmem_store()

                    # ---- per-row gate factors ---------------------------------------------
                    cumprod_vals = [gh.data[frag_tok[kv], 1] for kv in range(n_vals)]
                    last_cumsumlog = gh.data[B_T - 1, 0]
                    cumsumlog_vals = [gh.data[frag_tok[kv], 0] for kv in range(n_vals)]
                    decay_vals = []
                    for kv in cutlass.range_constexpr(0, n_vals, 2):
                        d0, d1 = fadd2(
                            last_cumsumlog,
                            last_cumsumlog,
                            -cumsumlog_vals[kv],
                            -cumsumlog_vals[kv + 1],
                        )
                        decay_vals.append(cute.math.exp2(d0, fastmath=True))
                        decay_vals.append(cute.math.exp2(d1, fastmath=True))
                    gh.release()

                    # ---- Y = V - cumprod * KS (packed 16-bit) ---------------------------
                    if cutlass.const_expr(is_h):
                        vh = v.wait()
                        vh.access(read=True, by="threads")
                        v_stage = v_base + vh.index * (B_T * d_v * bpe)
                        v_frag = []
                        for half in cutlass.range_constexpr(row_halves):
                            words = []
                            for block in cutlass.range_constexpr(4):
                                r = v_row + block * 16
                                off = v_segment + r * 64 + swizzle_xor_128b(r, v_col + half * 16)
                                words += ldmatrix(v_stage + off * bpe, 4, trans=True)
                            v_frag.append(words)
                    else:
                        zero_word = opaque_i32_zero()
                        v_frag = [[zero_word for _ in range(16)] for _ in range(row_halves)]
                    n_valid = end - (start + (chunk_begin + local) * B_T)
                    with trip.consume():
                        tmem.access("in_" + name, write=True)
                        if valid_state:
                            tmem.access("acc_" + name, read=True)
                            for half in cutlass.range_constexpr(row_halves):
                                ks = tmem_ld("16x256b", 8, acc_col + ((half * 16) << 16))
                                cute.arch.fence_view_async_tmem_load()
                                for j in cutlass.range_constexpr(16):
                                    s0, s1 = fmul2(
                                        ks[2 * j],
                                        ks[2 * j + 1],
                                        cumprod_vals[2 * j],
                                        cumprod_vals[2 * j + 1],
                                    )
                                    v_frag[half][j] = sub_f16x2(
                                        v_frag[half][j], fp32_to_fp16(s0, s1, dtype=io), dtype=io
                                    )
                        if n_valid < B_T:
                            # Zero Y past the piece end, where upstream's K and V are zero-filled.
                            for half in cutlass.range_constexpr(row_halves):
                                for j in cutlass.range_constexpr(16):
                                    tok = frag_tok[2 * j]
                                    keep = cutlass.select_(tok < n_valid, Int32(0xFFFF), Int32(0))
                                    keep = keep | cutlass.select_(
                                        tok + 1 < n_valid, Int32(-65536), Int32(0)
                                    )
                                    v_frag[half][j] = v_frag[half][j] & keep
                        for half in cutlass.range_constexpr(row_halves):
                            tmem_st("16x128b", in_col + ((half * 16) << 16), v_frag[half])
                    if cutlass.const_expr(is_h):
                        vh.release()

                    # ---- U epilogue: decayed U for the state update ----------------------
                    with trip.consume():
                        tmem.access("acc_" + name, read=True)
                        tmem.access("in_" + name, write=True)
                        u_vecs = [
                            tmem_ld("16x256b", 8, acc_col + ((half * 16) << 16))
                            for half in range(row_halves)
                        ]
                        cute.arch.fence_view_async_tmem_load()
                        for half in cutlass.range_constexpr(row_halves):
                            pack = []
                            for j in cutlass.range_constexpr(16):
                                u0, u1 = fmul2(
                                    u_vecs[half][2 * j],
                                    u_vecs[half][2 * j + 1],
                                    decay_vals[2 * j],
                                    decay_vals[2 * j + 1],
                                )
                                pack.append(fp32_to_fp16(u0, u1, dtype=io))
                            tmem_st("16x128b", in_col + ((half * 16) << 16), pack)

                # ---- final state: TMEM -> GMEM -----------------------------------------
                with trip.consume():
                    tmem.access("state_" + name, read=True)
                    if chunk_end == num_chunks:
                        row_addr = (
                            state_out.iterator + state_out.layout((piece, head, state_row, 0))
                        ).toint()
                        for i in cutlass.range_constexpr(n_ld):
                            vals = tmem_ld("32x32b", 32, state_col + i * 32)
                            cute.arch.fence_view_async_tmem_load()
                            if row_valid:
                                for g in cutlass.range_constexpr(8):
                                    st_global_v4_f32(
                                        row_addr + (i * 32 + g * 4) * 4,
                                        *vals[g * 4 : g * 4 + 4],
                                    )
            else:
                if chunk_end == num_chunks:
                    # An item without chunks: H = 0, M = I.
                    for c in cutlass.range(d_k):
                        if row_valid:
                            state_out[piece, head, state_row, c] = Float32(0.0)
                    if cutlass.const_expr(not is_h):  # noqa: SIM102 (static, then dynamic)
                        if row_valid:
                            state_out[piece, head, state_row, state_row] = Float32(1.0)
            work.advance()
        return gate, trip, v, work

    @cute.jit
    def _smem_ptr(self, addr):
        """16-bit SMEM pointer at 1024-byte aligned 32-bit shared address ``addr``."""
        return cute.make_ptr(self.io_dtype, addr, cute.AddressSpace.smem, assumed_align=1024)

    # ---- kernel and entry point --------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        k_load,
        v_load,
        tma_tinv,
        mTinv,
        gate,
        state_h,
        state_m,
        k_layout,
        v_layout,
        tinv_layout,
        box_layout,
        key_group,
        value_group,
        bound,
    ):
        self.begin_roles(work=bound, layouts={"K": k_layout, "V": v_layout, "Tinv": tinv_layout})
        params = SummaryParams(
            k_load=k_load,
            v_load=v_load,
            tma_tinv=tma_tinv,
            mTinv=mTinv,
            gate=gate,
            state_h=state_h,
            state_m=state_m,
            sK=self.buffer("K"),
            sV=self.buffer("V"),
            sTinv=self.buffer("Tinv"),
            box_layout=box_layout,
            key_group=key_group,
            value_group=value_group,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,  # (tokens, key heads, d_k)
        v: cute.Tensor,  # (tokens, value heads, d_v)
        gate: cute.Tensor,  # (tokens, heads) natural-log gate
        tinv: cute.Tensor,  # (tinv rows, heads, 64, 64)
        state_h: cute.Tensor,  # (pieces, heads, d_v, d_k) fp32
        state_m: cute.Tensor,  # (pieces, heads, d_k, d_k) fp32
        work_items: cute.Tensor,  # (capacity, 10) int32
        work_count: cute.Tensor,  # (1,) int32
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.launch(k, v, gate, tinv, state_h, state_m, work_items, work_count, tma_slots, stream)

    @cute.jit
    def launch(
        self,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        tinv: cute.Tensor,
        state_h: cute.Tensor,
        state_m: cute.Tensor,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        tma_slots: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launch one CTA per work-table row (the first ``work_count[0]`` run); callable from
        another ``@cute.jit`` host, such as a fused launch bundle."""
        io = self.io_dtype
        heads = Int32(gate.shape[1])
        key_group = heads // Int32(k.shape[1])
        value_group = heads // Int32(v.shape[1])
        box = sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1)
        box_layout = cute.select(box, mode=[0, 1])
        k_layout = cute.make_layout((B_T * self.d_k, STAGES))
        v_layout = cute.make_layout((B_T * self.d_v, STAGES))
        tinv_layout = cute.make_layout((B_T * B_T, STAGES))
        # (token, d, head) and (i, j, head, row) TMA views.
        mK = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[0, 2, 1]))
        mV = cute.make_tensor(v.iterator, cute.select(v.layout, mode=[0, 2, 1]))
        mTinv = cute.make_tensor(tinv.iterator, cute.select(tinv.layout, mode=[2, 3, 1, 0]))
        load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        k_load, v_load = (
            make_ragged_tma(load_op, t, box_layout, B_T, 64, slot_region(tma_slots, i, B_T, heads))
            for i, t in enumerate((mK, mV))
        )
        tma_tinv, mTinv_t = cpasync.make_tiled_tma_atom(load_op, mTinv, box_layout, (B_T, B_T))
        count = cute.make_tensor(work_count.iterator, cute.make_layout(1))
        bound = self.work.bind((work_items, count), limit=LastEntry(count))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            k_load,
            v_load,
            tma_tinv,
            mTinv_t,
            gate,
            state_h,
            state_m,
            k_layout,
            v_layout,
            tinv_layout,
            box_layout,
            key_group,
            value_group,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)


def bundle_hook(io_dtype, state_dtype, d_k: int, d_v: int, log_gate: bool) -> GdnSummary:
    """The summary op for launch bundles that run it inside their own ``@cute.jit`` host:
    ``op.launch(k, v, gate, tinv, state_h, state_m, work_items, work_count, tma_slots,
    stream)``."""
    if not log_gate:
        raise ValueError("the GDN summary takes natural-log gates")
    if state_dtype is not cutlass.Float32:
        raise TypeError(f"the GDN summary writes fp32 states, got {state_dtype}")
    name = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}[io_dtype]
    return GdnSummary(name, d_k, d_v)


@jit_cache
def compile_gdn_summary(op: GdnSummary):
    """Compile for dynamic tokens, heads, pieces, tile rows and work-table capacity."""
    io = op.io_dtype
    tokens, key_heads, value_heads, heads, pieces = (cute.sym_int() for _ in range(5))
    i32 = cutlass.Int32
    return compile_tvm_ffi(
        op,
        make_fake_strided_tensor(io, (tokens, key_heads, op.d_k), stride_divisibility=8),
        make_fake_strided_tensor(io, (tokens, value_heads, op.d_v), stride_divisibility=8),
        make_fake_strided_tensor(Float32, (tokens, heads), contiguous_dim=None),
        make_fake_strided_tensor(io, (cute.sym_int(), heads, B_T, B_T), stride_divisibility=8),
        make_fake_strided_tensor(
            Float32, (pieces, heads, op.d_v, op.d_k), stride_divisibility=op.d_k
        ),
        make_fake_strided_tensor(
            Float32, (pieces, heads, op.d_k, op.d_k), stride_divisibility=op.d_k
        ),
        make_fake_strided_tensor(i32, (cute.sym_int(), WORK_ITEM_FIELDS)),
        make_fake_strided_tensor(i32, (1,)),
        fake_tma_slots(),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


@functools.cache
def _op(dtype_name: str, d_k: int, d_v: int) -> GdnSummary:
    """One op per static configuration. Launches look its build up with ``by_static_key``
    (the hot-path lookup), which still honours ``cache_clear()`` and the compile target."""
    return GdnSummary(dtype_name, d_k, d_v)


def check_summary_outputs(what: str, d_k: int, *outputs: torch.Tensor) -> None:
    """Raise unless every FP32 summary output fits the compile: unit-stride rows, a base and
    outer strides divisible by ``d_k`` elements, and no two indices on one element (overlapping
    rows would make items overwrite each other)."""
    for output in outputs:
        if output.stride(-1) != 1 or any(s % d_k for s in output.stride()[:-1]):
            raise ValueError(f"{what} need unit-stride rows and d_k-divisible strides")
        if output.data_ptr() % (4 * d_k):
            raise ValueError(f"{what} must be aligned to d_k fp32 elements")
        reach = 0  # the largest offset the dims sorted so far address
        for stride, size in sorted(zip(output.stride(), output.shape)):
            if size > 1 and stride <= reach:
                raise ValueError(f"{what} must not overlap themselves")
            reach += stride * (size - 1)


def gdn_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    tinv: torch.Tensor,
    state_h: torch.Tensor,
    state_m: torch.Tensor,
    work_items: torch.Tensor,
    work_count: torch.Tensor,
) -> None:
    """Fill ``state_h`` / ``state_m`` for the work table (see the module docstring for the
    layouts); ``tinv`` holds ``gdn_tinv``'s tiles over the same pieces. The states may be views
    into a larger buffer, such as the two halves of a packed ``[H; M]`` map."""
    if k.dtype not in IO_DTYPE_NAMES or v.dtype != k.dtype or tinv.dtype != k.dtype:
        raise TypeError(f"k, v and tinv must share a 16-bit dtype, got {k.dtype}/{v.dtype}")
    if state_h.dtype != torch.float32 or state_m.dtype != torch.float32:
        raise TypeError("the summary states are fp32")
    if gate.dtype != torch.float32:
        raise TypeError(f"gate must be fp32, got {gate.dtype}")
    heads = gate.shape[1]
    if heads % k.shape[1] or heads % v.shape[1]:
        raise ValueError("heads must be a multiple of the key and value head counts")
    check_summary_outputs("the summary states", k.shape[-1], state_h, state_m)
    config = (IO_DTYPE_NAMES[k.dtype], k.shape[-1], v.shape[-1])
    op = _op(*config)
    fn = compile_gdn_summary.by_static_key(config, lambda: compile_gdn_summary(op))
    workspace, sms = op.work.launch_args(k.device)
    slots = new_tma_slots(op.tma_slot_rows(heads), k.device)
    fn(k, v, gate, tinv, state_h, state_m, work_items, work_count, slots, workspace, sms)
