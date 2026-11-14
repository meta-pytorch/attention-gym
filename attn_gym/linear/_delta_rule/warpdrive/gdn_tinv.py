# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/gdn_tinv_f16.py),
# rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and channels with CuTeDSL 4.6.2
# APIs: one TMA descriptor per tensor replaces the per-sequence descriptor arrays, and the
# natural-log gate is the only gate form.

"""GDN chunk-factor pass (the T pass) for SM100: one beta-folded chunk-inverse tile per (chunk,
head), the ``U`` GEMM operand of the GDN prefill, recompute, summary and backward kernels.

Per chunk ``c`` of sequence ``b`` (tokens ``[c * 64, (c + 1) * 64)``, heads independent)::

    g[t]      = cumsum(gate)[t] / ln 2                  (natural-log gate, in the log2 domain)
    M[i, j]   = (K K^T)[i, j] * exp2(g[i] - g[j]) * beta[i]          for i >= j
    T         = (I + strict_lower(M))^-1                (blockwise 8 -> 16 -> 32 -> 64)
    tile[i,j] = T[i, j] * beta[j]

``tinv[row, head]`` holds the tile of chunk ``c`` of sequence ``b`` at ``row = cu[b] // 64 + b +
c``; the rows between one sequence's last chunk and the next base are padding (``tinv_rows``).
K rows past a chunk's sequence end load as zeros (``ragged.RaggedTma``), as from
upstream's per-sequence descriptors, and have zero beta, so the tile stores them as zero rows
and columns.

The row table ``rows[r] = (chunk, batch, batch_start, batch_end)`` lists every valid chunk in
order and ``row_count[0]`` counts them; the recompute and backward summary read both. A one-warp
kernel checks ``cu_seqlens`` (``work_tables.check_packed_offsets``) and builds them before the
main launch.

Work: tiles ``t = r * heads + head`` run in pairs ``(2p, 2p + 1)``: one ``M = N = 128`` KK GEMM
``[K0; K1] @ [K0; K1]^T`` per pair, whose diagonal blocks are the two members' ``K K^T``.
CTA ``u`` of ``G`` owns the contiguous tiles ``[u * n // G, (u + 1) * n // G)`` of the
``n = row_count[0] * heads`` tiles (one OneShot item per CTA) and walks them as CTA-local
pairs, so consecutive pairs share row-table and gate/beta cache lines.

Roles and channels (12 warps)::

    load (8)  --k:   4 x [K0; K1] SMEM (TMA)-->        mma (9)
    mma       --acc: 4 x 128 TMEM columns (UMMA)-->    compute[0] (0-3) / compute[1] (4-7)
    gate (11) --gate: 8 x cumsum/beta payload-->       compute[0] / compute[1]
    compute   --tile: 4 x two 64 x 64 SMEM tiles-->    store (10): TMA store to tinv

``acc``, ``gate`` and ``tile`` alternate between the two compute instances by pair (round
robin). Each compute instance syncs its four warps on its own named barrier between the
inverse steps.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.linear._delta_rule.wd_support import HEAD_DIMS, IO_DTYPE_NAMES
from attn_gym.warpdrive import (
    Consumer,
    Instance,
    Local,
    NamedBarriers,
    OneShot,
    Producer,
    Ring,
    Sharing,
    SmemPlan,
    StageView,
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

from .blockwise_inverse import (
    blockwise_8x8_to_16x16,
    blockwise_16x16_to_32x32,
    blockwise_32x32_to_64x64,
    invert_diagonal_8x8,
)
from .ptx import (
    f16x2_to_f32,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    opaque_f32_zero,
    shfl_idx,
    shfl_up,
    stmatrix,
    swizzle_xor_128b,
    tmem_load_16x256b_x8,
)
from .ragged import (
    RaggedTma,
    descriptor_at,
    fake_tma_slots,
    make_ragged_tma,
    new_tma_slots,
    ragged_load,
    sequence_descriptors,
    slot_region,
    tma_slot_rows,
)
from .work_tables import check_packed_offsets, emit_chunk_rows

B_T = 64
RCP_LN2 = 1.4426950408889634  # 1 / ln 2: natural-log gates -> the kernel's log2 domain
K_STAGES = 4
ACC_STAGES = 4
GATE_STAGES = 8  # 4 per compute instance
TILE_STAGES = 4  # 2 per compute instance; one stage holds both members' tiles
TILE_ELEMS = B_T * B_T

_CUTE_DTYPES = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16, "fp32": cutlass.Float32}
_BETA_DTYPES = {**IO_DTYPE_NAMES, torch.float32: "fp32"}


def tinv_rows(total_tokens: int, num_seqs: int) -> int:
    """Tile rows of the ``tinv`` buffer: one row per chunk plus one padding row per sequence."""
    return total_tokens // B_T + num_seqs


@dataclass(frozen=True)
class TinvParams:
    tiled_mma: cute.TiledMma
    k_load: RaggedTma  # (token, d_k, key head) loads, zero past each sequence end
    tma_tinv: cute.CopyAtom
    mTinv: cute.Tensor  # TMA view (i, j, head, row)
    gate: cute.Tensor  # (token, head) fp32 natural-log gate
    beta: cute.Tensor  # (token, head)
    rows: cute.Tensor  # (row, 4) row table
    row_count: cute.Tensor  # (1,) valid chunk rows
    sK: cute.Tensor  # whole K ring (iterator = stage 0)
    sTile: cute.Tensor  # whole tile ring
    tCrA: StageView
    tCrB: StageView
    acc_layout: cute.Layout
    box_layout: cute.ComposedLayout  # one 64 x 64 swizzled box: K TMA destination, tile
    heads: Int32
    key_group: Int32  # value heads per key head
    units: Int32  # launched CTAs, one tile block each


@cute.jit
def tile_block(params: TinvParams, unit):
    """``(first tile, end tile, pairs)`` of CTA ``unit``: a contiguous block of the
    ``row_count[0] * heads`` tiles, so consecutive pairs share row-table and gate/beta lines."""
    n_items = cutlass.Int64(params.row_count[0] * params.heads)
    begin = Int32(cutlass.Int64(unit) * n_items // params.units)
    end = Int32(cutlass.Int64(unit + 1) * n_items // params.units)
    return begin, end, (end - begin + 1) // 2


@cute.jit
def pair_members(params: TinvParams, t0, end):
    """Row-table rows and heads of the pair at tile ``t0`` of a block ending at ``end``, and
    whether member 1 exists. A missing member 1 aliases member 0, so its loads stay in bounds."""
    r0 = t0 // params.heads
    h0 = t0 - r0 * params.heads
    have_m1 = t0 + 1 < end
    r1 = (t0 + 1) // params.heads
    h1 = t0 + 1 - r1 * params.heads
    if not have_m1:
        r1, h1 = r0, h0
    return r0, h0, r1, h1, have_m1


@cute.jit
def tile_row(params: TinvParams, r):
    """``tinv`` row of row-table entry ``r``: ``batch_start // 64 + batch + chunk``."""
    return params.rows[r, 2] // B_T + params.rows[r, 1] + params.rows[r, 0]


class GdnTinv(WarpSpecialized):
    """Warpdrive GDN T pass for ``dtype_name`` fp16/bf16 K and tiles, ``d_k`` 64 or 128.

    Beta may be fp32 or 16-bit; its loads follow the traced tensor's element type."""

    def __init__(self, dtype_name: str, d_k: int):
        if d_k not in HEAD_DIMS:
            raise ValueError(f"the chunk-factor pass serves d_k in {HEAD_DIMS}, got {d_k}")
        self.dtype_name, self.d_k = dtype_name, d_k
        self.io_dtype = _CUTE_DTYPES[dtype_name]
        self.smem_plan = SmemPlan()
        k_buf = self.smem_plan.buffer("K", self.io_dtype, align=1024)
        kb_buf = self.smem_plan.buffer("KB", self.io_dtype, align=1024, overlaps=k_buf)
        tile_buf = self.smem_plan.buffer("Tile", self.io_dtype, align=1024)
        k_bytes = 2 * B_T * d_k * (self.io_dtype.width // 8)
        rr = Sharing.ROUND_ROBIN
        # load_pair issues 2 members x d_k / 64 boxes of 64 x 64 per stage: exactly k_bytes. The
        # TMA check reports this ring unverified: it cannot follow copies through the
        # sequence-offset (domain_offset) views.
        self.k = Ring(K_STAGES, Tma(k_bytes), Umma, buffer={"K": k_buf, "KB": kb_buf})
        self.acc = Ring(ACC_STAGES, Umma, Warps, consumer_sharing=rr, release_fence="tmem_load")
        self.gate = Ring(
            GATE_STAGES, Threads, Threads, consumer_sharing=rr, payload=(Float32, (B_T, 2, 2))
        )
        self.tile = Ring(
            TILE_STAGES,
            Warps,
            Warps,
            producer_sharing=rr,
            commit_fence="smem_async",  # stmatrix writes, then the TMA store reads
            buffer=tile_buf,
        )
        self.inverse_sync = NamedBarriers(2, threads=128)
        self.tmem_plan = TmemPlan().region("acc", 2 * B_T, ACC_STAGES)
        self.work = WorkStream(schedule=OneShot(grid="1d"))

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: K, keyed by head."""
        return tma_slot_rows(1, B_T, heads)

    def get_name(self) -> str:
        return f"wd_gdn_tinv_{self.dtype_name}_dk{self.d_k}"

    # ---- roles -------------------------------------------------------------------------
    # Every role walks its CTA's tile block (one OneShot item) pair by pair.

    @role(warps=8, regs=96)
    @cute.jit
    def load(self, params: TinvParams, *, k: Producer, work: WorkCursor):
        while work.valid:
            unit, _, _ = work.item
            begin, end, n_pairs = tile_block(params, unit)
            # Rows of sequences ending inside the tensor's first chunk lead the row table (and so
            # every block): load the block's pairs starting there through their own descriptors,
            # then the rest through the shared one.
            n_head = Int32(0)
            scanning = n_pairs > 0
            while scanning:
                scanning = False
                if params.rows[(begin + 2 * n_head) // params.heads, 3] < B_T:
                    n_head += 1
                    scanning = n_head < n_pairs
            # This CTA's tiles of a row share a descriptor: cache its (end, address).
            cache = (Int32(-1), Int64(0))
            for j in cutlass.range(n_head, unroll=1):
                k, cache = self.load_pair(params, k, begin + 2 * j, end, cache)
            for j in cutlass.range(n_head, n_pairs, unroll=1):
                k = self.load_pair(params, k, begin + 2 * j, end)
            work.advance()
        return k, work

    @cute.jit
    def load_pair(self, params: TinvParams, k, t0, end, cache=None):
        """TMA both members' K tiles into one ``k`` stage. With ``cache`` (returned updated) a
        member may be a chunk of a sequence ending inside the tensor's first chunk: this CTA's
        first tile of such a row writes the row's descriptor, keyed by that tile's head (only
        this CTA owns the tile, so only this warp writes the slot), and its later tiles of the
        row reuse it. Without, members issue through the shared descriptor."""
        r0, h0, r1, h1, _have_m1 = pair_members(params, t0, end)
        with k.produce() as h:
            h.access("K", write=True, by="tma")
            stage = params.sK.iterator.toint() + h.index * (2 * B_T * self.d_k * 2)
            for member in cutlass.range_constexpr(2):
                r = r0 if member == 0 else r1
                head = h0 if member == 0 else h1
                token = params.rows[r, 2] + params.rows[r, 0] * B_T
                valid = cutlass.min(params.rows[r, 3] - token, B_T)
                rows = (stage + member * TILE_ELEMS * 2, token, valid, head // params.key_group)
                desc = None
                if cutlass.const_expr(cache is not None):
                    cached_end, cached_addr = cache
                    if token + valid != cached_end:
                        (row_desc,) = sequence_descriptors(
                            (params.k_load,), token + valid, head, B_T
                        )
                        cached_end, cached_addr = token + valid, row_desc.toint()
                    cache = (cached_end, cached_addr)
                    desc = descriptor_at(cached_addr)
                self._load_member(params, rows, h.barrier, desc)
        if cutlass.const_expr(cache is None):
            return k
        return k, cache

    @cute.jit
    def _load_member(self, params: TinvParams, rows, bar, desc=None):
        """TMA one member's K tile, ``rows = (member smem address, token, valid, key head)``."""
        addr, token, valid, head_k = rows
        for kb in cutlass.range_constexpr(self.d_k // 64):
            dst = cute.make_tensor(
                self._smem_ptr(addr + kb * 2 * TILE_ELEMS * 2), params.box_layout
            )
            ragged_load(params.k_load, dst, token, valid, head_k, kb, bar, B_T, 64, desc)

    @role(warps=9, regs=96)
    @cute.jit
    def mma(
        self,
        params: TinvParams,
        *,
        k: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        tiled_mma = params.tiled_mma
        tAcc = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        k = k.views(K=params.tCrA, KB=params.tCrB)
        while work.valid:
            unit, _, _ = work.item
            _begin, _end, n_pairs = tile_block(params, unit)
            for _ in cutlass.range(n_pairs, unroll=1):
                with k.consume() as kh, acc.produce() as ah:
                    tmem.access("acc", 0, write=True, by="umma")
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(self.d_k // 16):
                        cute.gemm(
                            tiled_mma,
                            tAcc[None, None, None, ah.index],
                            kh.view("K", None, None, kb, read=True, by="umma"),
                            kh.view("KB", None, None, kb, read=True, by="umma"),
                            tAcc[None, None, None, ah.index],
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            work.advance()
        return k, acc, work

    @role(warps=11, regs=96)
    @cute.jit
    def gate_load(self, params: TinvParams, *, gate: Producer, work: WorkCursor, me: Local):
        while work.valid:
            unit, _, _ = work.item
            begin, end, n_pairs = tile_block(params, unit)
            for j in cutlass.range(n_pairs, unroll=1):
                gate = self.gate_pair(params, gate, begin + 2 * j, end, me.lane)
            work.advance()
        return gate, work

    @cute.jit
    def gate_pair(self, params: TinvParams, gate, t0, end, lane):
        """Both members' log2 gate cumsum and beta into one ``gate`` stage."""
        r0, h0, r1, h1, have_m1 = pair_members(params, t0, end)
        zero = Float32(0.0)
        # Issue both members' loads before any use: the row-table and gate/beta reads are this
        # role's critical path. A missing member 1 aliases member 0 and is not stored.
        vals = []
        for member in cutlass.range_constexpr(2):
            r = r0 if member == 0 else r1
            head = h0 if member == 0 else h1
            start = params.rows[r, 2] + params.rows[r, 0] * B_T
            seq_end = params.rows[r, 3]
            toks = (start + lane, start + lane + 32)
            valid = [t < seq_end for t in toks]
            raw_g = [params.gate[cutlass.min(t, seq_end - 1), head] for t in toks]
            raw_b = [params.beta[cutlass.min(t, seq_end - 1), head] for t in toks]
            vals.append((valid, raw_g, raw_b))
        with gate.produce() as gh:
            for member in cutlass.range_constexpr(2):
                valid, raw_g, raw_b = vals[member]
                g = [raw_g[i].to(Float32) if valid[i] else zero for i in range(2)]
                rcp_ln2 = opaque_f32_zero() + Float32(RCP_LN2)
                g[0], g[1] = fmul2(g[0], g[1], rcp_ln2, rcp_ln2)
                # 64-token cumsum: warp scan per 32-token column, then the carry.
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(2):
                        n = shfl_up(g[col], offset)
                        if lane >= offset:
                            g[col] = g[col] + n
                g[1] += shfl_idx(g[0], 31)
                live = cutlass.Boolean(True) if member == 0 else have_m1
                if live:
                    for i in cutlass.range_constexpr(2):
                        gh.data[lane + 32 * i, member, 0] = g[i]
                        gh.data[lane + 32 * i, member, 1] = (
                            raw_b[i].to(Float32) if valid[i] else zero
                        )
        return gate

    @role(warps=[range(4), range(4, 8)], regs=200, shared_body=True)
    @cute.jit
    def compute(
        self,
        params: TinvParams,
        *,
        stage: Instance,
        acc: Consumer,
        gate: Consumer,
        tile: Producer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        while work.valid:
            unit, _, _ = work.item
            begin, end, n_pairs = tile_block(params, unit)
            # Instance ``stage`` takes the block's pairs stage, stage + 2, ... (round robin).
            for j in cutlass.range(stage, n_pairs, 2, unroll=1):
                acc, gate, tile = self.compute_pair(
                    params, begin + 2 * j, end, stage, acc, gate, tile, tmem, me
                )
            work.advance()
        return acc, gate, tile, work

    @cute.jit
    def compute_pair(self, params: TinvParams, t0, end, stage, acc, gate, tile, tmem, me):
        """One pair's KK epilogues, inverse and beta scaling; returns the updated endpoints."""
        dtype = self.io_dtype
        barrier = self.inverse_sync[stage]
        warp, lane, tid = me.warp, me.lane, me.tid
        mask_zero = opaque_f32_zero()
        inverse_local_warp = warp % 2
        pair_half = warp // 2
        half_row_base = inverse_local_warp * 32
        store_row = warp * 16 + lane % 16
        store_col = (lane // 16) * 8
        store_row_frag = lane % 16
        n_vals = 32
        acc_frags = B_T // 16
        rows_u = (
            half_row_base + lane // 4,
            half_row_base + lane // 4 + 8,
            half_row_base + lane // 4 + 16,
            half_row_base + lane // 4 + 24,
        )
        # TMEM: this warp's 32 lanes, two 16-lane halves; KK member block = 64 columns.
        tmem_col0 = tmem.ptr("acc").toint()
        lane_lo = (32 * (me.tmem_row // 32)) << 16
        lane_hi = (32 * (me.tmem_row // 32) + 16) << 16
        tile_base = params.sTile.iterator.toint()
        _r0, _h0, _r1, _h1, have_m1 = pair_members(params, t0, end)
        gh = gate.wait()
        cs = gh.data
        # ---- KK epilogue factors for this warp's member ------------------------
        kk_idx = Int32(1) if (have_m1 and pair_half == 1) else Int32(0)
        do_kk = have_m1 or pair_half == 0
        row_cs = [cs[r, kk_idx, 0] for r in rows_u]
        col_cs = []
        for gi in cutlass.range_constexpr(8):
            for bi in cutlass.range_constexpr(2):
                col_cs.append(cs[(lane % 4) * 2 + gi * 8 + bi, kk_idx, 0])
        decay = []
        for u in cutlass.range_constexpr(2):
            for kv in cutlass.range_constexpr(n_vals):
                hi_row = ((kv // 2) % 2) == 1
                chunk_row = rows_u[2 * u + (1 if hi_row else 0)]
                chunk_col = (lane % 4) * 2 + ((kv // 4) * 8 + kv % 2)
                rc = row_cs[2 * u + (1 if hi_row else 0)]
                cc = col_cs[(kv // 4) * 2 + kv % 2]
                decay.append(
                    cute.math.exp2(rc - cc, fastmath=True) if chunk_row >= chunk_col else mask_zero
                )
        kk_beta = [cs[r, kk_idx, 1] for r in rows_u]

        th = tile.acquire()
        th.access(write=True, by="threads")
        th.access(read=True, by="threads")
        tile0 = tile_base + th.index * (2 * TILE_ELEMS * 2)
        tile1 = tile0 + TILE_ELEMS * 2
        kk_tile = tile1 if pair_half == 1 else tile0

        # Warps without a member (2-3 of a lone last tile) load and drop member 0's columns.
        ah = acc.wait()
        tmem.access("acc", 0, read=True, by="threads")
        col = tmem_col0 + ah.index * (2 * B_T) + kk_idx * B_T
        kk_lo = tmem_load_16x256b_x8(col + lane_lo)
        kk_hi = tmem_load_16x256b_x8(col + lane_hi)
        cute.arch.fence_view_async_tmem_load()
        ah.release()
        if do_kk:
            for u in cutlass.range_constexpr(2):
                kk = kk_hi if u == 1 else kk_lo
                pack = []
                for kv in cutlass.range_constexpr(n_vals // 2):
                    row_beta = kk_beta[2 * u + 1] if kv % 2 == 1 else kk_beta[2 * u]
                    p0, p1 = fmul2(
                        kk[2 * kv],
                        kk[2 * kv + 1],
                        decay[u * n_vals + 2 * kv],
                        decay[u * n_vals + 2 * kv + 1],
                    )
                    v0, v1 = fmul2(p0, p1, row_beta, row_beta)
                    pack.append(fp32_to_fp16(v0, v1, dtype=dtype))
                st_row = half_row_base + u * 16 + store_row_frag
                for c in cutlass.range_constexpr(acc_frags):
                    off = st_row * B_T + swizzle_xor_128b(st_row, store_col + c * 16)
                    stmatrix(kk_tile + off * 2, pack[c * 4 : c * 4 + 4])

        # ---- pair inverse: warps 0-1 own member 0, warps 2-3 member 1 ----------
        inv_tile = tile1 if (have_m1 and warp >= 2) else tile0
        do_inv = have_m1 or warp < 2
        t0 = self._flat(tile0)
        t1 = self._flat(tile1)
        ti = self._flat(inv_tile)
        barrier.sync(aligned=True)
        if do_inv:
            invert_diagonal_8x8(ti, (inverse_local_warp * 32 + lane) // 8, tid, B_T, dtype)
        barrier.sync(aligned=True)
        blockwise_8x8_to_16x16(t0, warp * 16, lane, B_T, dtype)
        if have_m1:
            blockwise_8x8_to_16x16(t1, warp * 16, lane, B_T, dtype)
        barrier.sync(aligned=True)
        if do_inv:
            blockwise_16x16_to_32x32(ti, inverse_local_warp * 32, lane, B_T, dtype)
        barrier.sync(aligned=True)
        blockwise_32x32_to_64x64(ti, inverse_local_warp, lane, barrier, B_T, dtype)
        barrier.sync(aligned=True)

        # ---- beta column scaling: T^-1[i, j] *= beta[j] ------------------------
        for member in cutlass.range_constexpr(2):
            live = cutlass.Boolean(True) if member == 0 else have_m1
            if live:
                base = tile0 if member == 0 else tile1
                beta_col = [
                    cs[(lane % 4) * 2 + ((kv // 4) * 8 + kv % 2), member, 1]
                    for kv in range(n_vals)
                ]
                frags = []
                for c in cutlass.range_constexpr(acc_frags):
                    off = store_row * B_T + swizzle_xor_128b(store_row, store_col + c * 16)
                    frags += ldmatrix(base + off * 2, 4)
                pack = []
                for jj in cutlass.range_constexpr(n_vals // 2):
                    lo, hi = f16x2_to_f32(frags[jj], dtype=dtype)
                    s0, s1 = fmul2(lo, hi, beta_col[2 * jj], beta_col[2 * jj + 1])
                    pack.append(fp32_to_fp16(s0, s1, dtype=dtype))
                for c in cutlass.range_constexpr(acc_frags):
                    off = store_row * B_T + swizzle_xor_128b(store_row, store_col + c * 16)
                    stmatrix(base + off * 2, pack[c * 4 : c * 4 + 4])
        gh.release()
        th.commit()
        return acc, gate, tile

    @cute.jit
    def _smem_ptr(self, addr):
        """16-bit SMEM pointer at 1024-byte aligned 32-bit shared address ``addr``."""
        return cute.make_ptr(self.io_dtype, addr, cute.AddressSpace.smem, assumed_align=1024)

    @cute.jit
    def _flat(self, addr):
        """Flat 16-bit SMEM tensor over one tile at 32-bit shared address ``addr``."""
        return cute.make_tensor(self._smem_ptr(addr), cute.make_layout(TILE_ELEMS))

    @role(warps=10, regs=96)
    @cute.jit
    def store(self, params: TinvParams, *, tile: Consumer, work: WorkCursor):
        while work.valid:
            unit, _, _ = work.item
            begin, end, n_pairs = tile_block(params, unit)
            for j in cutlass.range(n_pairs, unroll=1):
                tile = self.store_pair(params, tile, begin + 2 * j, end)
            work.advance()
        bulk_wait_group(0)
        return tile, work

    @cute.jit
    def store_pair(self, params: TinvParams, tile, t0, end):
        """TMA-store one stage's tiles to their ``tinv`` rows; release once read."""
        r0, h0, r1, h1, have_m1 = pair_members(params, t0, end)
        with tile.consume() as th:
            th.access(read=True, by="tma")
            stage = params.sTile.iterator.toint() + th.index * (2 * TILE_ELEMS * 2)
            for member in cutlass.range_constexpr(2):
                live = cutlass.Boolean(True) if member == 0 else have_m1
                if live:
                    r = r0 if member == 0 else r1
                    head = h0 if member == 0 else h1
                    src = cute.make_tensor(
                        self._smem_ptr(stage + member * TILE_ELEMS * 2), params.box_layout
                    )
                    dst = cute.local_tile(
                        params.mTinv[None, None, head, tile_row(params, r)], (B_T, B_T), (0, 0)
                    )
                    tSs, tSg = cpasync.tma_partition(
                        params.tma_tinv,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(src, 0, 2),
                        cute.group_modes(dst, 0, 2),
                    )
                    cute.copy(params.tma_tinv, tSs, tSg)
            bulk_commit_group()
            bulk_wait_group(0, read=True)
        return tile

    # ---- kernel and entry point --------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        tiled_mma,
        k_load,
        tma_tinv,
        mTinv,
        gate,
        beta,
        rows,
        row_count,
        k_layout,
        kb_layout,
        tile_layout,
        box_layout,
        heads,
        key_group,
        units,
        bound,
    ):
        self.begin_roles(work=bound, layouts={"K": k_layout, "KB": kb_layout, "Tile": tile_layout})
        sK, sKB = self.buffer("K"), self.buffer("KB")
        acc_shape = tiled_mma.partition_shape_C((2 * B_T, 2 * B_T))
        acc_staged = tiled_mma.make_fragment_C(cute.append(acc_shape, ACC_STAGES))
        params = TinvParams(
            tiled_mma=tiled_mma,
            k_load=k_load,
            tma_tinv=tma_tinv,
            mTinv=mTinv,
            gate=gate,
            beta=beta,
            rows=rows,
            row_count=row_count,
            sK=sK,
            sTile=self.buffer("Tile"),
            tCrA=StageView(tiled_mma.make_fragment_A(sK), axis=3),
            tCrB=StageView(tiled_mma.make_fragment_B(sKB), axis=3),
            acc_layout=acc_staged.layout,
            box_layout=box_layout,
            heads=heads,
            key_group=key_group,
            units=units,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,  # (tokens, key heads, d_k)
        gate: cute.Tensor,  # (tokens, heads) fp32
        beta: cute.Tensor,  # (tokens, heads)
        cu_seqlens: cute.Tensor,  # (num_seqs + 1,) int32
        tinv: cute.Tensor,  # (tinv rows, heads, 64, 64)
        rows: cute.Tensor,  # (tinv rows, 4) int32
        row_count: cute.Tensor,  # (1,) int32
        workspace: cute.Tensor,
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.launch(k, gate, beta, cu_seqlens, tinv, rows, row_count, tma_slots, max_ctas, stream)

    @cute.jit
    def launch(
        self,
        k: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        tinv: cute.Tensor,
        rows: cute.Tensor,
        row_count: cute.Tensor,
        tma_slots: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
        emit_row_table: cutlass.Constexpr[bool] = True,
    ):
        """Launch the row-table kernel (unless the caller already built ``rows`` and
        ``row_count`` on this stream) and the T pass; callable from another ``@cute.jit``
        host, such as a fused launch bundle."""
        io = self.io_dtype
        heads = Int32(gate.shape[1])
        key_group = heads // Int32(k.shape[1])
        tile = (2 * B_T, 2 * B_T, self.d_k)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            io,
            io,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            tile[:2],
        )
        k_layout = sm100_utils.make_smem_layout_a(tiled_mma, tile, io, K_STAGES)
        kb_layout = sm100_utils.make_smem_layout_b(tiled_mma, tile, io, K_STAGES)
        box = sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1)
        box_layout = cute.select(box, mode=[0, 1])
        tile_layout = cute.make_layout((2 * TILE_ELEMS, TILE_STAGES))
        # (token, d_k, key head) and (i, j, head, row) TMA views.
        mK = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[0, 2, 1]))
        mTinv = cute.make_tensor(tinv.iterator, cute.select(tinv.layout, mode=[2, 3, 1, 0]))
        k_load = make_ragged_tma(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mK,
            box_layout,
            B_T,
            64,
            slot_region(tma_slots, 0, B_T, heads),
        )
        tma_tinv, mTinv_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mTinv, box_layout, (B_T, B_T)
        )
        if cutlass.const_expr(emit_row_table):
            emit_rows(cu_seqlens, Int32(k.shape[0]), rows, row_count).launch(
                grid=(1, 1, 1), block=(32, 1, 1), stream=stream
            )
        # One CTA per SM, each owning a contiguous block of the (device-counted) tiles.
        units = cutlass.max(cutlass.min(max_ctas, Int32(tinv.shape[0]) * heads), 1)
        bound = self.work.bind((units, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tiled_mma,
            k_load,
            tma_tinv,
            mTinv_t,
            gate,
            beta,
            rows,
            row_count,
            k_layout,
            kb_layout,
            tile_layout,
            box_layout,
            heads,
            key_group,
            units,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)


@cute.kernel
def emit_rows(cu_seqlens: cute.Tensor, tokens: Int32, rows: cute.Tensor, row_count: cute.Tensor):
    """One warp: check ``cu_seqlens`` against ``tokens``, then the row table of 64-token chunks
    (``work_tables.emit_chunk_rows``)."""
    lane = Int32(cute.arch.thread_idx()[0])
    check_packed_offsets(cu_seqlens, tokens, lane, 32)
    emit_chunk_rows(B_T, cu_seqlens, rows, row_count, lane)


def bundle_hook(io_dtype, d_k: int, num_sm: int, log_gate: bool) -> tuple[GdnTinv, int]:
    """``(op, num_sm)`` constexpr for launch bundles that run the T pass inside their own
    ``@cute.jit`` host: ``op.launch(..., Int32(num_sm), stream, emit_row_table=...)``."""
    if not log_gate:
        raise ValueError("the GDN T pass takes natural-log gates")
    name = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}[io_dtype]
    return GdnTinv(name, d_k), num_sm


@jit_cache
def compile_gdn_tinv(op: GdnTinv, beta_dtype_name: str):
    """Compile for dynamic tokens, heads, sequences and tile rows."""
    io = op.io_dtype
    tokens, key_heads, heads, rows = (cute.sym_int() for _ in range(4))
    # K/Q are often head slices of a fused projection: dynamic token/head strides.
    k = make_fake_strided_tensor(io, (tokens, key_heads, op.d_k), stride_divisibility=8)
    gate = make_fake_strided_tensor(Float32, (tokens, heads), contiguous_dim=None)
    beta = make_fake_strided_tensor(
        _CUTE_DTYPES[beta_dtype_name], (tokens, heads), contiguous_dim=None
    )
    i32 = cutlass.Int32
    return compile_tvm_ffi(
        op,
        k,
        gate,
        beta,
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        make_fake_strided_tensor(io, (rows, heads, B_T, B_T), stride_divisibility=8),
        make_fake_strided_tensor(i32, (rows, 4), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        fake_tma_slots(),
        Int32(1),
        name=op.get_name(),
    )


def gdn_tinv(
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    tinv: torch.Tensor,
    rows: torch.Tensor,
    row_count: torch.Tensor,
) -> None:
    """Fill ``tinv``, ``rows`` and ``row_count`` for packed ``k [T, HK, D]`` and ``[T, H]``
    natural-log ``gate`` / ``beta`` (see the module docstring for the layouts)."""
    if k.dtype not in IO_DTYPE_NAMES or tinv.dtype != k.dtype:
        raise TypeError(f"k and tinv must share a 16-bit dtype, got {k.dtype}/{tinv.dtype}")
    if gate.dtype != torch.float32 or beta.dtype not in _BETA_DTYPES:
        raise TypeError(f"gate must be fp32 and beta fp32/16-bit, got {gate.dtype}/{beta.dtype}")
    heads = gate.shape[1]
    if heads % k.shape[1] or tinv.shape[1:] != (heads, B_T, B_T):
        raise ValueError("tinv must be [rows, heads, 64, 64] with heads a multiple of k heads")
    op = GdnTinv(IO_DTYPE_NAMES[k.dtype], k.shape[-1])
    fn = compile_gdn_tinv(op, _BETA_DTYPES[beta.dtype])
    workspace, sms = op.work.launch_args(k.device)
    slots = new_tma_slots(op.tma_slot_rows(heads), k.device)
    fn(k, gate, beta, cu_seqlens, tinv, rows, row_count, workspace, slots, sms)
