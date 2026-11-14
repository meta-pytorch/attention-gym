# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# kda_prep_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and channels
# with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence descriptor
# arrays, and the natural-log gate is the only gate form.

"""KDA prep (BT = 16, per-key-channel decay) for SM100: the per-(chunk, head) records the
prep-fed prefill (``kda_prep_prefill``) TMA-loads instead of computing them.

Per chunk of 16 tokens of head ``h`` (``g`` the in-chunk cumulative log2 gate per key channel,
rows past the sequence end carry no gate increment and zero Q / K / beta)::

    k_decay[t, d] = K[t, d] exp2(g[t, d])          q_decay[t, d] = Q[t, d] exp2(g[t, d])
    K_inv[t, d]   = K[t, d] exp2(-g[t, d])         K_restore[t, d] = K_inv[t, d] exp2(g[15, d])
    diag[d]       = exp2(g[15, d])
    L = beta[i] strict_lower(k_decay K_inv^T)       inverse_beta = (I + L)^-1 diag(beta)
    t = inverse_beta^T K_restore                    a = lower(q_decay K_inv^T) inverse_beta

``k_decay``, ``q_decay`` and ``t`` are ``[row, head, 16, d_k]`` 16-bit images of the prefill's
128-byte swizzled SMEM boxes (TMA stores), ``a`` is the ``[row, head, 128]`` int32 image of its
32-byte swizzled 16 x 16 tile and ``diag`` ``[row, head, d_k]`` fp32, at record row
``batch_start // 16 + batch + chunk``.

Work: the row table ``rows[r] = (chunk, batch, batch_start, batch_end)`` and ``row_count[0]``
come from the caller (``work_tables`` in the launch bundle, ``prep_rows``
standalone); items ``r * heads + head`` split into contiguous blocks, one per CTA, with the
upstream block formula. Q / K / gate rows past a sequence end load as zeros
(``ragged.RaggedTma``), as from upstream's per-sequence descriptors.

Roles: one four-warp ``compute`` role (the upstream kernel_prep layout, four CTAs per SM), every
warp sharing phase 1 (the gate scan and the decay operands of the whole chunk), then phase 2:
warp 0 KK / L / inverse / inverse_beta and the raw loads, warp 2 the a record, warps 1 and 3 the
two key halves of t (warp 1 also stores the records). The four warps step through each item on
the ``item_sync`` named barrier (four syncs per item, as upstream); the two-stage ``raw`` ring of
Q / K / gate TMA loads is managed on its consumer side, and warp 0 drives its producer side raw
(a TMA side binds a single-warp role, and a role cannot hold both ends of a channel)::

    compute warp 0 (raw producer) --raw: 2 x Q,K,Gate (TMA)--> compute (all four warps)
"""

import itertools
from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.linear._delta_rule.wd_support import HEAD_DIMS
from attn_gym.warpdrive import (
    Consumer,
    Local,
    NamedBarrier,
    OneShot,
    PdlWait,
    Ring,
    SmemPlan,
    Warps,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    bulk_commit_group,
    bulk_wait_group,
    fence_async_shared,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_strided_tensor
from attn_gym.warpdrive.compat import LayoutEnum

from .blockwise_inverse import invert_unit_lower_16x16_fragments
from .contract import tma_divisibility
from .kda_common import (
    A_WORDS,
    B_T,
    CUTE_DTYPES,
    TORCH_NAMES,
    box128,
    lds_16_f32,
    lds_f32,
    lds_v4,
    lds_v4_f32,
    load_rows,
    mma_tf32_16x16,
    opaque_i32,
    spin_wait,
    sts_f32,
    sts_tf32x8,
    swz32,
    swz128,
    tf32_fragment_offsets,
)
from .ptx import (
    f16x2_to_f32,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    mma_step,
    movmatrix_16b,
    st_shared_v4,
    stmatrix,
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

LOG2_E = 1.4426950408889634
RAW_STAGES = 2
CTAS_PER_SM = 4
# fp16 adds 20 KB of tf32 decay tiles (``kda_common``), which leaves room for three CTAs.
TF32_CTAS_PER_SM = 3


@dataclass(frozen=True)
class KdaPrepParams:
    q_load: RaggedTma  # (token, dim, head) loads, zero past each sequence end
    k_load: RaggedTma
    g_load: RaggedTma
    tma_kd: cute.CopyAtom  # record stores
    tma_qd: cute.CopyAtom
    tma_t: cute.CopyAtom
    mKdecay: cute.Tensor  # TMA store views (16, d_k, head, row)
    mQdecay: cute.Tensor
    mT: cute.Tensor
    box_io: cute.ComposedLayout  # one 16 x 64 16-bit swizzled box
    box_g: cute.ComposedLayout  # one 16 x 128-byte gate box
    beta: cute.Tensor  # (token, head)
    a: cute.Tensor  # (row, head, A_WORDS) int32
    diag: cute.Tensor  # (row, head, d_k) fp32
    rows: cute.Tensor  # (row, 4) row table
    row_count: cute.Tensor  # (1,)
    heads: cute.FastDivmodDivisorV2
    q_ratio: cute.FastDivmodDivisorV2  # heads per q / k head
    k_ratio: cute.FastDivmodDivisorV2
    units: cute.FastDivmodDivisorV2  # launched CTAs
    s_q: Int32  # SMEM byte addresses of the buffers (stage 0)
    s_k: Int32
    s_g: Int32
    s_ex: Int32
    s_kinv: Int32  # fp32 (tf32) under fp16 io
    s_krestore: Int32
    s_inter: Int32
    s_beta: Int32
    s_kdecay: Int32
    s_qdecay: Int32
    s_t: Int32
    s_kdecay32: Int32  # fp16 io only: tf32 copies of K_decay / Q_decay
    s_qdecay32: Int32


class KdaPrep(WarpSpecialized):
    """Warpdrive KDA prep for ``io`` fp16/bf16 operands and an fp32 or ``io`` gate; beta may be
    any float dtype (its loads follow the traced tensor). ``num_sm`` sizes the grid.
    ``fastmath=False`` exponentiates the gates with the non-flushing ``exp2`` and reciprocal."""

    def __init__(self, io: str, gate: str, d_k: int, num_sm: int, fastmath: bool = True):
        if d_k not in HEAD_DIMS:
            raise ValueError(f"the KDA prep serves d_k in {HEAD_DIMS}, got {d_k}")
        if gate not in ("fp32", io):
            raise ValueError(f"the gate must be fp32 or the io dtype, got {gate}")
        self.io, self.gate, self.d_k, self.num_sm = io, gate, d_k, num_sm
        self.fastmath = fastmath
        self.io_dtype = CUTE_DTYPES[io]
        self.gate_dtype = CUTE_DTYPES[gate]
        # fp16 computes A and KK from tf32 decay tiles (``kda_common``).
        self.tf32_decay = io == "fp16"
        self.ctas_per_sm = TF32_CTAS_PER_SM if self.tf32_decay else CTAS_PER_SM
        io_t, f32 = self.io_dtype, Float32
        # Declaration order (and 1024-byte alignment) follows the upstream SharedStorage.
        plan = self.smem_plan = SmemPlan()
        buf = {
            "Q": plan.buffer("Q", io_t, align=1024),
            "K": plan.buffer("K", io_t, align=1024),
            "Gate": plan.buffer("Gate", self.gate_dtype, align=1024),
        }
        # An fp32 gate is exchanged in place (exp2 prefixes over the raw gate stage); a 16-bit
        # gate exchanges through one fp32 tile.
        if gate != "fp32":
            buf["Exchange"] = plan.buffer("Exchange", f32, align=1024)
        buf["Kinv"] = plan.buffer("Kinv", f32 if self.tf32_decay else io_t, align=1024)
        buf["Krestore"] = plan.buffer("Krestore", io_t, align=1024)
        buf["Inter"] = plan.buffer("Inter", io_t, align=1024)
        buf["Beta"] = plan.buffer("Beta", f32, align=128)
        buf["Kdecay"] = plan.buffer("Kdecay", io_t, align=1024)
        buf["Qdecay"] = plan.buffer("Qdecay", io_t, align=1024)
        buf["T"] = plan.buffer("T", io_t, align=1024)
        if self.tf32_decay:
            for name in ("Kdecay32", "Qdecay32"):
                buf[name] = plan.buffer(name, f32, align=1024)
        self.buf = buf
        self.raw_bytes = B_T * d_k * (2 * (io_t.width // 8) + self.gate_dtype.width // 8)
        # Warp 0 of the consumer role drives the producer side raw (see ``load_item``). The TMA
        # check cannot attribute those copies to a producer handle. With an external producer the
        # layer cannot derive the TMA ring's release fence, so it is declared: the consumers read
        # (and, for an fp32 gate, rewrite) the stage with generic accesses before its TMA refill.
        self.raw = Ring(
            RAW_STAGES,
            consumer=Warps,
            external="producer",
            override={"full_counts": 1},
            release_fence="smem_async",
            buffer={k: buf[k] for k in ("Q", "K", "Gate")},
        )
        # Orders every other buffer: the four warps sync on it four times per item (upstream
        # barrier 1); the comments at each sync name the lifetimes it closes.
        self.item_sync = NamedBarrier(threads=128)
        self.work = WorkStream(schedule=OneShot(grid="1d"), ready=PdlWait())

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: Q, K and gate, keyed by head."""
        return tma_slot_rows(3, B_T, heads)

    def get_name(self) -> str:
        name = f"wd_kda_prep_{self.io}_g{self.gate}_k{self.d_k}_sm{self.num_sm}"
        return name if self.fastmath else f"{name}_nofm"

    # ---- work decode -------------------------------------------------------------------

    @cute.jit
    def item_block(self, params: KdaPrepParams, unit):
        """``(first item, items)`` of CTA ``unit`` (the upstream split of ``row_count * heads``
        items: a contiguous block per CTA, the spare items spread evenly)."""
        units = params.units
        n_all = Int32(params.row_count[0]) * params.heads.divisor
        per_cta = n_all // units
        spare = n_all - per_cta * units.divisor
        spare_q = (unit * spare) // units
        spare_r = unit * spare - spare_q * units.divisor
        begin = unit * per_cta + spare_q
        n_items = per_cta + (Int32(1) if spare_r + spare >= units.divisor else Int32(0))
        return begin, n_items

    @cute.jit
    def item(self, params: KdaPrepParams, item):
        """``(row, head, chunk_start, batch_start, seqlen)`` of ``item`` (chunk_start in
        tokens within the sequence)."""
        r = item // params.heads
        head = item - r * params.heads.divisor
        start = params.rows[r, 2]
        return r, head, params.rows[r, 0] * B_T, start, params.rows[r, 3] - start

    # ---- the raw producer side (warp 0) --------------------------------------------------

    @cute.jit
    def load_item(self, params: KdaPrepParams, raw, epoch, item):
        """TMA item ``item``'s Q / K / gate boxes into ``raw`` stage ``epoch % 2`` as the ring's
        producer: wait for the four warps' releases of epoch ``epoch - 2`` (already ordered by
        ``item_sync``, so the wait passes at once), arm the stage's bytes, issue the copies."""
        _r, head, chunk_start, start, seqlen = self.item(params, item)
        stage = epoch & (RAW_STAGES - 1)
        spin_wait(raw.empty_barrier(stage), ((epoch >> 1) & 1) ^ 1)
        full = raw.full_barrier(stage)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full, self.raw_bytes)
        rows = (start + chunk_start, cutlass.min(seqlen - chunk_start, B_T))
        loads = (params.q_load, params.k_load, params.g_load)
        descs = sequence_descriptors(loads, start + seqlen, head, B_T)
        self._load_raw(params, stage, full, rows, head, descs)

    @cute.jit
    def _load_raw(self, params: KdaPrepParams, stage, full, rows, head, descs):
        """Issue the Q / K / gate copies of one item's chunk ``rows`` into ``raw`` stage
        ``stage`` through ``descs``, completing on ``full``."""
        q_desc, k_desc, g_desc = descs
        io, io_b = self.io_dtype, self.io_dtype.width // 8
        g_b = self.gate_dtype.width // 8
        qk_stage = stage * (self.d_k * B_T * io_b)
        for kb in cutlass.range_constexpr(self.d_k // 64):
            off = qk_stage + kb * (B_T * 128)
            head_q, head_k = head // params.q_ratio, head // params.k_ratio
            load_rows(
                params.q_load,
                io,
                params.s_q + off,
                params.box_io,
                rows,
                head_q,
                kb,
                64,
                full,
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
                full,
                k_desc,
            )
        g_cols = 128 // g_b
        g_stage = stage * (self.d_k * B_T * g_b)
        for kb in cutlass.range_constexpr(self.d_k // g_cols):
            load_rows(
                params.g_load,
                self.gate_dtype,
                params.s_g + g_stage + kb * (B_T * 128),
                params.box_g,
                rows,
                head,
                kb,
                g_cols,
                full,
                g_desc,
            )

    @cute.jit
    def store_records(self, params: KdaPrepParams, head, tile_row):
        """TMA-store the item's k_decay, q_decay and t tiles (warp 1) as one bulk group."""
        self._store_record(params.tma_kd, params.mKdecay, params, params.s_kdecay, head, tile_row)
        self._store_record(params.tma_qd, params.mQdecay, params, params.s_qdecay, head, tile_row)
        self._store_record(params.tma_t, params.mT, params, params.s_t, head, tile_row)
        bulk_commit_group()

    @cute.jit
    def _store_record(self, atom, view, params: KdaPrepParams, s_rec, head, tile_row):
        for kb in cutlass.range_constexpr(self.d_k // 64):
            src = cute.make_tensor(
                cute.make_ptr(
                    self.io_dtype,
                    s_rec + kb * (B_T * 128),
                    cute.AddressSpace.smem,
                    assumed_align=1024,
                ),
                params.box_io,
            )
            dst = cute.local_tile(view[None, None, head, tile_row], (B_T, 64), (0, kb))
            s, d = cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(src, 0, 2),
                cute.group_modes(dst, 0, 2),
            )
            cute.copy(atom, s, d)

    # ---- compute (warps 0-3) ---------------------------------------------------------------

    @role(warps=range(4))
    @cute.jit
    def compute(self, params: KdaPrepParams, *, raw: Consumer, work: WorkCursor, me: Local):
        lane, warp = me.lane, me.warp
        d_k = self.d_k
        # ---- phase 1 lane geometry (the prefill's CG0) ---------------------------------------
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
        # ---- phase 2 ldmatrix / stmatrix lane decode (the prefill's warps 12 and 15) ------
        ki_row = lane % 8 + (8 if lane // 16 else 0)
        ki_col = 8 if (lane // 8) % 2 else 0
        a_row = lane % 8 + (8 if (lane // 8) % 2 else 0)
        a_col = 8 if (lane // 8) // 2 else 0
        geo = (
            lane,
            warp,
            channel,
            seg_base,
            segment,
            col,
            prefix_offsets,
            gate_offsets,
            decay_row,
            lane_in_row,
            a_row * B_T + swz32(a_row, a_col),
            [opaque_i32(ki_row * 64 + swz128(ki_row, i * 16 + ki_col)) for i in range(4)],
            [opaque_i32(swz128(a_row, a_row * 64 + i * 16 + a_col)) for i in range(4)],
            [
                opaque_i32(
                    (n // 4) * (B_T * 64) + a_row * 64 + swz128(a_row, (n % 4) * 16 + a_col)
                )
                for n in range(d_k // 16)
            ],
        )
        n_done = Int32(0)  # items (raw epochs) this CTA has consumed
        while work.valid:
            unit, _, _ = work.item
            begin, n_items = self.item_block(params, unit)
            if warp == 0:
                for s0 in cutlass.range(cutlass.min(Int32(RAW_STAGES), n_items), unroll=1):
                    self.load_item(params, raw, n_done + s0, begin + s0)
            for j in cutlass.range(n_items, unroll=1):
                raw = self.item_body(params, raw, begin, j, n_items, n_done + j, geo)
            n_done += n_items
            work.advance()
        if warp == 1:
            bulk_wait_group(0, read=True)
        cute.arch.griddepcontrol_launch_dependents()
        return raw, work

    @cute.jit
    def item_body(self, params: KdaPrepParams, raw, begin, j, n_items, epoch, geo):
        """One item's records; returns the ``raw`` end."""
        (
            lane,
            warp,
            channel,
            seg_base,
            segment,
            col,
            prefix_offsets,
            gate_offsets,
            decay_row,
            lane_in_row,
            inverse_idx,
            ki_offs,
            kd_offs,
            t_offs,
        ) = geo
        io = self.io_dtype
        d_k = self.d_k
        dk_halves = d_k // 64
        channel_rows = d_k // 4
        store_rows = B_T * channel_rows // 32
        store_row_base = (lane // channel_rows) * store_rows
        t_groups = d_k // 16
        row_lo = lane // 4
        row_hi = row_lo + 8
        col_pair = 2 * (lane % 4)
        r, head, chunk_start, start, seqlen = self.item(params, begin + j)
        tile_row = start // B_T + params.rows[r, 1] + chunk_start // B_T

        # ---- the chunk's beta row: gathered by warp 0, stored after the operands ----------
        beta_value = Float32(0.0)
        if warp == 0 and lane < B_T and chunk_start + lane < seqlen:
            beta_value = params.beta[start + chunk_start + lane, head].to(Float32)

        rh = raw.wait()
        rh.access("Q", read=True, by="threads")
        rh.access("K", read=True, by="threads")
        rh.access("Gate", read=True, by="threads")
        rh.access("Gate", write=True, by="threads")
        g_stage_bytes = d_k * B_T * (self.gate_dtype.width // 8)
        s_gate = params.s_g + rh.index * g_stage_bytes

        # ---- gate prefix scan (log2 domain), exp2 ------------------------------------------
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
        s_ex = s_gate if cutlass.const_expr(self.gate == "fp32") else params.s_ex
        for row_off in cutlass.range_constexpr(store_rows):
            if cutlass.const_expr(channel_rows < 32):
                row = store_row_base + row_off
                value = g[row_off + B_T // 2] if lane >= channel_rows else g[row_off]
                idx = seg_base + swz128(row ^ segment, row * 32 + col, 4)
            else:
                value = g[row_off]
                idx = prefix_offsets[row_off % 8] + row_off * 32
            sts_f32(s_ex + 4 * idx, value)
        if warp == 1:  # the previous item's record stores have read the record tiles
            bulk_wait_group(0, read=True)
        # Publishes the exchange rows; orders the previous item's record TMA reads before this
        # item's record stores.
        self.item_sync.sync()

        # ---- diag record: exp2(g[15, d]) per key channel ------------------------------------
        params.diag[tile_row, head, channel] = g[B_T - 1]

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
            kd_pack = []
            ki_pack = []
            kd32 = []  # fp32 K_decay / K_inv for the tf32 tiles (fp16 io)
            ki32 = []
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
            off = 2 * box128(decay_row, dim_base)
            if cutlass.const_expr(self.tf32_decay):
                sts_tf32x8(params.s_kinv, decay_row, dim_base, ki32, d_k)
                sts_tf32x8(params.s_kdecay32, decay_row, dim_base, kd32, d_k)
            else:
                st_shared_v4(params.s_kinv + off, ki_pack)
            st_shared_v4(params.s_kdecay + off, kd_pack)

        # ---- Q decay, K restore --------------------------------------------------------------
        for half in cutlass.range_constexpr(dk_halves):
            qd_pack = []
            qd32 = []
            for p in cutlass.range_constexpr(4):
                r0 = half * 8 + 2 * p
                qd0, qd1 = fmul2(raw_q[r0], raw_q[r0 + 1], exp_g[r0], exp_g[r0 + 1])
                qd_pack.append(fp32_to_fp16(qd0, qd1, dtype=io))
                qd32 += [qd0, qd1]
            st_shared_v4(
                params.s_qdecay + 2 * box128(decay_row, half * 64 + lane_in_row * 8), qd_pack
            )
            if cutlass.const_expr(self.tf32_decay):
                sts_tf32x8(params.s_qdecay32, decay_row, half * 64 + lane_in_row * 8, qd32, d_k)
        for half in cutlass.range_constexpr(dk_halves):
            off = 2 * box128(decay_row, half * 64 + lane_in_row * 8)
            st_shared_v4(params.s_krestore + off, kr_pack[4 * half : 4 * half + 4])

        # ---- the chunk's beta; the raw stage is consumed ----------------------------------
        if warp == 0 and lane < B_T:
            sts_f32(params.s_beta + 4 * lane, beta_value)
        rh.release()
        # Publishes K_inv, K_restore, the record operands and beta; every warp has released the
        # raw stage, so warp 0 may refill it.
        self.item_sync.sync()
        if warp == 0 and j + RAW_STAGES < n_items:
            self.load_item(params, raw, epoch + RAW_STAGES, begin + j + RAW_STAGES)

        # ---- phase 2, warp 0: KK = K decay @ K inv^T, L, (I + L)^-1, inverse_beta ----------
        if warp == 0:
            kk = cute.make_rmem_tensor((8,), Float32)
            for i in cutlass.range_constexpr(8):
                kk[i] = Float32(0.0)
            if cutlass.const_expr(self.tf32_decay):
                mma_tf32_16x16(
                    kk,
                    params.s_kdecay32,
                    params.s_kinv,
                    tf32_fragment_offsets(lane, d_k, b_operand=False),
                    tf32_fragment_offsets(lane, d_k, b_operand=True),
                    d_k,
                )
            else:
                for i in cutlass.range_constexpr(d_k // 16):
                    box = (i // 4) * (B_T * 64)
                    kif = ldmatrix(params.s_kinv + 2 * (ki_offs[i % 4] + box), 4)
                    kdf = ldmatrix(params.s_kdecay + 2 * (kd_offs[i % 4] + box), 4)
                    mma_step(kk, kdf, kif, k_step=0, M=16, N=16, dtype=io)
            beta_lo = lds_f32(params.s_beta + 4 * row_lo)
            beta_hi = lds_f32(params.s_beta + 4 * row_hi)
            l_regs = cute.make_rmem_tensor((8,), Float32)
            for i in cutlass.range_constexpr(8):
                rr = row_hi if cutlass.const_expr(i % 4 >= 2) else row_lo
                cc = (i // 4) * 8 + col_pair + (i % 2)
                l_regs[i] = kk[i] if rr > cc else Float32(0.0)
            for p in cutlass.range_constexpr(4):
                bs = beta_hi if cutlass.const_expr(p % 2 == 1) else beta_lo
                l_regs[2 * p], l_regs[2 * p + 1] = fmul2(l_regs[2 * p], l_regs[2 * p + 1], bs, bs)
            inv = invert_unit_lower_16x16_fragments(l_regs, lane, io)
            for p in cutlass.range_constexpr(4):
                cc = (p // 2) * 8 + col_pair
                beta_c0 = lds_f32(params.s_beta + 4 * cc)
                beta_c1 = lds_f32(params.s_beta + 4 * (cc + 1))
                inv[2 * p], inv[2 * p + 1] = fmul2(inv[2 * p], inv[2 * p + 1], beta_c0, beta_c1)
            stmatrix(
                params.s_inter + 2 * inverse_idx,
                [fp32_to_fp16(inv[2 * i], inv[2 * i + 1], dtype=io) for i in range(4)],
            )

        # ---- phase 2, warp 2: A = lower(Q decay @ K inv^T) ----------------------------------
        a_acc = cute.make_rmem_tensor((8,), Float32)
        for i in cutlass.range_constexpr(8):
            a_acc[i] = Float32(0.0)
        if warp == 2:
            if cutlass.const_expr(self.tf32_decay):
                mma_tf32_16x16(
                    a_acc,
                    params.s_qdecay32,
                    params.s_kinv,
                    tf32_fragment_offsets(lane, d_k, b_operand=False),
                    tf32_fragment_offsets(lane, d_k, b_operand=True),
                    d_k,
                )
            else:
                for i in cutlass.range_constexpr(d_k // 16):
                    box = (i // 4) * (B_T * 64)
                    kif = ldmatrix(params.s_kinv + 2 * (ki_offs[i % 4] + box), 4)
                    qdf = ldmatrix(params.s_qdecay + 2 * (kd_offs[i % 4] + box), 4)
                    mma_step(a_acc, qdf, kif, k_step=0, M=16, N=16, dtype=io)
            for i in cutlass.range_constexpr(8):
                rr = row_hi if cutlass.const_expr(i % 4 >= 2) else row_lo
                cc = (i // 4) * 8 + col_pair + (i % 2)
                a_acc[i] = a_acc[i] if rr >= cc else Float32(0.0)

        # ---- phase 2, warps 1 and 3: K restore fragments, transposed -----------------------
        kr_b = [Int32(0) for _ in range(4 * (t_groups // 2))]
        for half in cutlass.range_constexpr(2):
            if warp == 1 + 2 * half:
                for n in cutlass.range_constexpr(t_groups // 2):
                    kr = ldmatrix(params.s_krestore + 2 * t_offs[half * (t_groups // 2) + n], 4)
                    for i in cutlass.range_constexpr(4):
                        kr_b[4 * n + i] = movmatrix_16b(kr[i])
        # Publishes inverse_beta; K_inv / K_decay / Q_decay / K_restore / beta reads are done.
        self.item_sync.sync()

        ainv = ldmatrix(params.s_inter + 2 * inverse_idx, 4)

        # ---- warp 2: a = A @ inverse_beta (SW32 order, packed words) ------------------------
        if warp == 2:
            a_rec = cute.make_rmem_tensor((8,), Float32)
            for i in cutlass.range_constexpr(8):
                a_rec[i] = Float32(0.0)
            mma_step(
                a_rec,
                [fp32_to_fp16(a_acc[2 * i], a_acc[2 * i + 1], dtype=io) for i in range(4)],
                [movmatrix_16b(ainv[i]) for i in range(4)],
                k_step=0,
                M=16,
                N=16,
                dtype=io,
            )
            for p in cutlass.range_constexpr(4):
                rr = row_hi if cutlass.const_expr(p % 2 == 1) else row_lo
                cc = (p // 2) * 8 + col_pair
                word = (rr * B_T + swz32(rr, cc)) // 2
                params.a[tile_row, head, word] = fp32_to_fp16(
                    a_rec[2 * p], a_rec[2 * p + 1], dtype=io
                )

        # ---- warps 1 and 3: t = inverse_beta^T @ K restore, key halves ----------------------
        for half in cutlass.range_constexpr(2):
            if warp == 1 + 2 * half:
                tf = [movmatrix_16b(ainv[i]) for i in (0, 2, 1, 3)]
                for n in cutlass.range_constexpr(t_groups // 2):
                    t_acc = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        t_acc[i] = Float32(0.0)
                    mma_step(t_acc, tf, kr_b[4 * n : 4 * n + 4], k_step=0, M=16, N=16, dtype=io)
                    stmatrix(
                        params.s_t + 2 * t_offs[half * (t_groups // 2) + n],
                        [fp32_to_fp16(t_acc[2 * i], t_acc[2 * i + 1], dtype=io) for i in range(4)],
                    )

        # ---- the record tiles SMEM -> GMEM ---------------------------------------------------
        fence_async_shared()  # generic record writes, then the TMA store reads
        # Publishes the record tiles; inverse_beta reads are done.
        self.item_sync.sync()
        if warp == 1:
            self.store_records(params, head, tile_row)
        return raw

    # ---- kernel and entry point ---------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        q_load,
        k_load,
        g_load,
        tma_kd,
        tma_qd,
        tma_t,
        mKdecay,
        mQdecay,
        mT,
        box_io,
        box_g,
        beta,
        a,
        diag,
        rows,
        row_count,
        heads,
        q_ratio,
        k_ratio,
        units,
        layouts,
        bound,
    ):
        self.begin_roles(work=bound, layouts=layouts)
        addr = {name: self.buffer(name).iterator.toint() for name in self.buf}
        params = KdaPrepParams(
            q_load=q_load,
            k_load=k_load,
            g_load=g_load,
            tma_kd=tma_kd,
            tma_qd=tma_qd,
            tma_t=tma_t,
            mKdecay=mKdecay,
            mQdecay=mQdecay,
            mT=mT,
            box_io=box_io,
            box_g=box_g,
            beta=beta,
            a=a,
            diag=diag,
            rows=rows,
            row_count=row_count,
            heads=heads,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
            units=units,
            s_q=addr["Q"],
            s_k=addr["K"],
            s_g=addr["Gate"],
            s_ex=addr["Exchange"] if "Exchange" in addr else addr["Gate"],
            s_kinv=addr["Kinv"],
            s_krestore=addr["Krestore"],
            s_inter=addr["Inter"],
            s_beta=addr["Beta"],
            s_kdecay=addr["Kdecay"],
            s_qdecay=addr["Qdecay"],
            s_t=addr["T"],
            s_kdecay32=addr.get("Kdecay32", addr["Kinv"]),
            s_qdecay32=addr.get("Qdecay32", addr["Kinv"]),
        )
        self.run_roles(params)

    @cute.jit
    def launch(
        self,
        q: cute.Tensor,  # (tokens, q heads, d_k)
        k: cute.Tensor,  # (tokens, k heads, d_k)
        gate: cute.Tensor,  # (tokens, heads, d_k) natural-log decay
        beta: cute.Tensor,  # (tokens, heads)
        k_decay: cute.Tensor,  # (rows, heads, 16, d_k)
        q_decay: cute.Tensor,
        t: cute.Tensor,
        a: cute.Tensor,  # (rows, heads, 128) int32
        diag: cute.Tensor,  # (rows, heads, d_k) fp32
        rows: cute.Tensor,  # (rows, 4) int32 row table
        row_count: cute.Tensor,  # (1,) int32 valid rows
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        stream: cuda.CUstream,
    ):
        """Launch (PDL) on up to ``num_sm * 4`` CTAs; callable from another ``@cute.jit`` host
        such as a launch bundle, after a kernel that wrote ``rows``/``row_count``."""
        io, gdt = self.io_dtype, self.gate_dtype
        heads = Int32(gate.shape[1])
        box_io = cute.select(
            sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1), mode=[0, 1]
        )
        g_cols = 128 // (gdt.width // 8)
        box_g = cute.select(
            sm100_utils.make_smem_layout_epi(gdt, LayoutEnum.ROW_MAJOR, (B_T, g_cols), 1),
            mode=[0, 1],
        )

        def tdh(x):  # (token, head, dim) -> (token, dim, head)
            return cute.make_tensor(x.iterator, cute.select(x.layout, mode=[0, 2, 1]))

        def rec(x):  # (row, head, 16, d_k) -> (16, d_k, head, row)
            return cute.make_tensor(x.iterator, cute.select(x.layout, mode=[2, 3, 1, 0]))

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        s2g = cpasync.CopyBulkTensorTileS2GOp()

        def ragged(op, t, box, cols, index):  # clipped at each sequence end, keyed by head
            slots = slot_region(tma_slots, index, B_T, heads)
            return make_ragged_tma(op, tdh(t), box, B_T, cols, slots)

        q_load = ragged(g2s, q, box_io, 64, 0)
        k_load = ragged(g2s, k, box_io, 64, 1)
        g_load = ragged(g2s, gate, box_g, g_cols, 2)
        tma_kd, mKdecay = cpasync.make_tiled_tma_atom(s2g, rec(k_decay), box_io, (B_T, 64))
        tma_qd, mQdecay = cpasync.make_tiled_tma_atom(s2g, rec(q_decay), box_io, (B_T, 64))
        tma_t, mT = cpasync.make_tiled_tma_atom(s2g, rec(t), box_io, (B_T, 64))
        tile = self.d_k * B_T
        layouts = {
            "Q": cute.make_layout((tile, RAW_STAGES)),
            "K": cute.make_layout((tile, RAW_STAGES)),
            "Gate": cute.make_layout((tile, RAW_STAGES)),
            "Kinv": cute.make_layout((tile, 1)),
            "Krestore": cute.make_layout((tile, 1)),
            "Inter": cute.make_layout((B_T * B_T, 1)),
            "Beta": cute.make_layout((B_T, 1)),
            "Kdecay": cute.make_layout((tile, 1)),
            "Qdecay": cute.make_layout((tile, 1)),
            "T": cute.make_layout((tile, 1)),
        }
        if cutlass.const_expr(self.tf32_decay):
            for name in ("Kdecay32", "Qdecay32"):
                layouts[name] = cute.make_layout((tile, 1))
        if cutlass.const_expr(self.gate != "fp32"):
            layouts["Exchange"] = cute.make_layout((tile, 1))
        units = cutlass.max(
            cutlass.min(Int32(self.num_sm * self.ctas_per_sm), Int32(k_decay.shape[0]) * heads), 1
        )
        bound = self.work.bind((units, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            q_load,
            k_load,
            g_load,
            tma_kd,
            tma_qd,
            tma_t,
            mKdecay,
            mQdecay,
            mT,
            box_io,
            box_g,
            beta,
            a,
            diag,
            rows,
            row_count,
            cute.FastDivmodDivisorV2(heads),
            cute.FastDivmodDivisorV2(heads // Int32(q.shape[1])),
            cute.FastDivmodDivisorV2(heads // Int32(k.shape[1])),
            cute.FastDivmodDivisorV2(units),
            layouts,
            bound,
        ).launch(
            **self.launch_config(bound, min_blocks_per_mp=self.ctas_per_sm),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        k_decay: cute.Tensor,
        q_decay: cute.Tensor,
        t: cute.Tensor,
        a: cute.Tensor,
        diag: cute.Tensor,
        rows: cute.Tensor,
        row_count: cute.Tensor,
        tma_slots: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.launch(
            q, k, gate, beta, k_decay, q_decay, t, a, diag, rows, row_count, tma_slots, stream
        )


# ---- standalone entry point (tests) ---------------------------------------------------------


def record_rows(total_tokens: int, num_seqs: int) -> int:
    """Record rows: one per chunk plus one padding row per sequence."""
    return total_tokens // B_T + num_seqs


def prep_rows(cu_seqlens: torch.Tensor, capacity: int):
    """``(rows, row_count)``: one ``(chunk, batch, batch_start, batch_end)`` row per chunk in
    sequence order, padded to ``capacity`` rows (as ``work_tables.emit_chunk_rows``
    writes them)."""
    bounds = cu_seqlens.tolist()
    table = [
        [c, b, s, e]
        for b, (s, e) in enumerate(itertools.pairwise(bounds))
        for c in range(-(-(e - s) // B_T))
    ]
    rows = torch.zeros(capacity, 4, dtype=torch.int32)
    if table:
        rows[: len(table)] = torch.tensor(table, dtype=torch.int32)
    device = cu_seqlens.device
    return rows.to(device), torch.tensor([len(table)], dtype=torch.int32, device=device)


@jit_cache
def compile_kda_prep(op: KdaPrep, beta_name: str):
    """Compile for dynamic tokens, heads and record rows."""
    io, i32 = op.io_dtype, cutlass.Int32
    tokens, hq, hk, heads, rows = (cute.sym_int() for _ in range(5))

    def packed(dtype, h):
        return make_fake_strided_tensor(
            dtype, (tokens, h, op.d_k), stride_divisibility=tma_divisibility(dtype)
        )

    def record():
        return make_fake_strided_tensor(io, (rows, heads, B_T, op.d_k), stride_divisibility=8)

    return compile_tvm_ffi(
        op,
        packed(io, hq),
        packed(io, hk),
        packed(op.gate_dtype, heads),
        make_fake_strided_tensor(CUTE_DTYPES[beta_name], (tokens, heads), contiguous_dim=None),
        record(),
        record(),
        record(),
        make_fake_strided_tensor(i32, (rows, heads, A_WORDS), stride_divisibility=4),
        make_fake_strided_tensor(Float32, (rows, heads, op.d_k), stride_divisibility=4),
        make_fake_strided_tensor(i32, (rows, 4), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        fake_tma_slots(),
        name=op.get_name(),
    )


def kda_prep(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Packed ``[T, H, d_k]`` q / k (grouped heads allowed), natural-log channel gate
    ``[T, H, d_k]`` and beta ``[T, H]``; returns the records (see the module docstring) with
    the ``rows`` / ``row_count`` table that addresses them."""
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        raise TypeError("q and k must share a 16-bit dtype")
    tokens, heads, d_k = gate.shape
    n_rows = record_rows(tokens, cu_seqlens.shape[0] - 1)
    rows, row_count = prep_rows(cu_seqlens, n_rows)
    dev = q.device
    out = {
        name: torch.zeros(n_rows, heads, B_T, d_k, dtype=q.dtype, device=dev)
        for name in ("k_decay", "q_decay", "t")
    }
    out["a"] = torch.zeros(n_rows, heads, A_WORDS, dtype=torch.int32, device=dev)
    out["diag"] = torch.zeros(n_rows, heads, d_k, dtype=torch.float32, device=dev)
    op = KdaPrep(
        TORCH_NAMES[q.dtype],
        TORCH_NAMES[gate.dtype],
        d_k,
        torch.cuda.get_device_properties(dev).multi_processor_count,
    )
    fn = compile_kda_prep(op, TORCH_NAMES[beta.dtype])
    fn(
        q,
        k,
        gate,
        beta,
        out["k_decay"],
        out["q_decay"],
        out["t"],
        out["a"],
        out["diag"],
        rows,
        row_count,
        new_tma_slots(op.tma_slot_rows(heads), dev),
    )
    return {**out, "rows": rows, "row_count": row_count}
