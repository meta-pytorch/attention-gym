# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/gdn_prefill_f16.py),
# rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and channels with CuTeDSL 4.6.2
# APIs: one TMA descriptor per tensor replaces the per-sequence descriptor arrays (a ragged view
# clips the O stores), and the checkpoint, linear-gate, 16-bit-beta and final-index paths are
# gone.

"""Chunked scalar-gate GDN prefill (forward) for the SM100 family.

Per chunk ``c`` (64 tokens) of a work item, with ``g`` the in-chunk cumulative log2 gate::

    T[i, j]  = exp2(g[i] - g[j]) (i >= j)            decay between tokens
    KS, QS   = K S, Q S                               S: the d_k x d_v state entering the chunk
    Y        = V - fp16(KS * exp2(g))                 delta-rule residual
    U        = Tinv Y                                 Tinv: the beta-folded chunk inverse
    A        = (Q K^T * T) * scale                    (i >= j)
    O        = QS * exp2(g) * scale + A U
    S       <- S * exp2(g[63]) + K^T (U * exp2(g[63] - g))

``tinv_source="compute"`` (``GdnPrefillCompute``) builds ``Tinv`` in compute group 0 from one
``M = 128`` ``[K; Q] K^T`` GEMM per chunk and a blockwise inverse over chunk pairs (the T pass of
``gdn_tinv`` fused in); ``"gmem"`` (``GdnPrefillGmem``, the chain path) TMA-loads the tiles
``gdn_tinv`` wrote and issues ``Q K^T`` alone.

Work: ``work_tables`` writes the item table (``[batch, head, write_start, write_end,
compute_start, compute_end, batch_start, batch_end, final_dst, dstate_dst]``, chunk units) in
LPT order and zeroes the claim counter. One OneShot item per CTA (``grid = num_sms``); inside it
every role walks the CTA's items in the ticket order upstream uses: the first item is
``blockIdx.x``, and after issuing an item's loads the load warp claims the next
(``num_ctas + atomicAdd``) and publishes it on the ``ticket`` ring, which the eleven other warps
read when they finish the item. Warpdrive's broadcast schedules would need a thirteenth warp,
which the 224/256/24 register split cannot fund.

Q/K/V loads zero-fill rows past a sequence end (``ragged.RaggedTma``), as upstream's
per-sequence descriptors do, so no sequence reads another's tokens; ragged TMA store views clip
O at each sequence end.

Roles (12 warps, registers 224/256/24): compute0 (warps 0-3), compute1 (4-7), gate_load (8),
load (9), mma (10), store (11). Channels (stages; producer -> consumers)::

    ticket    2  Warps load -> compute0, compute1, gate_load, mma, store (11 warp arrivals)
    kq        4  Tma load -> raw mma               [K;Q] / [Q;K] stages by chunk parity
    v         2  Tma load -> compute1
    tinv      3  Tma load (gmem) | compute0 (compute) -> mma
    gate      3  gate_load -> compute0, compute1   cumsum log2 gate and its exp2
    beta      3  gate_load cp.async -> compute0    (compute only)
    cg0       2  Umma mma -> compute0              [KK; QK] or QK accumulators (TMEM)
    a         3  compute0 -> mma                   A tiles
    o         1  compute1 -> store                 staged O (TMA store)

The compute1 <-> MMA ping-pong over TMEM is the critical loop. Its handoffs are one-way, as
upstream: each stage is released by a later handoff of the same chunk, so these rings have a
raw producer (an elected ``tcgen05.commit`` or 128 thread arrivals after ``tcgen05.wait::st``)
that never waits for a release, and a managed consumer that closes with
``release(signal=False)``::

    state      mma -> compute1   KV GEMM done (S)
    cg1        mma -> compute1   KS, then U done (two epochs per chunk)
    qs         mma -> compute1   QS, then O done (two epochs per chunk)
    qs_scaled  compute1 -> mma   QS rescaled in place: the O GEMM may accumulate
    state_in   compute1 -> mma   fp16 S staged
    slot0      compute1 -> mma   Y, then U staged (two epochs per chunk)
    slot1      compute1 -> mma   decayed U staged

Explicit releases on these cost 2-10% (every added acquire and arrive sits on the critical
loop).
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    AFTER_FENCE_COVERED,
    BEFORE_FENCE_COVERED,
    BOTH_FENCES_COVERED,
    Consumer,
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
    WorkStream,
    bulk_commit_group,
    bulk_wait_group,
    fence_async_shared,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_strided_tensor
from attn_gym.warpdrive.compat import LayoutEnum, SmemAllocator
from attn_gym.warpdrive.thread_sync import tcgen05_after_thread_sync, tcgen05_before_thread_sync

from . import umma
from .blockwise_inverse import (
    blockwise_8x8_to_16x16,
    blockwise_16x16_to_32x32,
    blockwise_32x32_to_64x64,
    invert_diagonal_8x8,
)
from .ptx import (
    cp_async_4b_zfill,
    f16x2_to_f32,
    fadd2,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    opaque_f32_zero,
    shfl_idx,
    shfl_up,
    stmatrix,
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
    ragged_store,
    sequence_descriptors,
    slot_region,
    tma_slot_rows,
)

B_T = 64
BOX_ELEMS = B_T * 64  # one 64-row x 128-byte swizzled TMA box of 16-bit values
TILE_BYTES = B_T * B_T * 2
RCP_LN2 = 1.4426950408889634  # 1 / ln 2: natural-log gates -> the kernel's log2 domain
KQ_STAGES = 4
V_STAGES = 2
GATE_STAGES = 3
BETA_STAGES = 3
TINV_STAGES = 3
A_STAGES = 3
CG0_STAGES = 2
TICKET_STAGES = 2
KEY_DIMS = (64, 128)
VALUE_DIMS = (64, 128)
TINV_SOURCES = ("compute", "gmem")
_CUTE_DTYPES = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16, "fp32": cutlass.Float32}
_NAMES = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16", cutlass.Float32: "fp32"}

# Work-item table columns (work_tables).
_BATCH, _HEAD, _WRITE_START, _WRITE_END, _COMPUTE_START = 0, 1, 2, 3, 4
_BATCH_START, _BATCH_END, _FINAL_DST = 6, 7, 8


@dataclass(frozen=True)
class PrefillParams:
    q_load: RaggedTma  # (token, d_k, q head) loads, zero past each sequence end
    k_load: RaggedTma
    v_load: RaggedTma  # (token, d_v, v head)
    o_store: RaggedTma  # (token, d_v, head) stores, dropping rows past each sequence end
    tma_tinv: object  # gmem: TMA atom of the tinv tiles
    mTinv: object  # gmem: TMA view (i, j, head, row)
    box_layout: cute.ComposedLayout
    gate: cute.Tensor  # (token, gate head) fp32 natural-log gate
    beta: object  # compute: (token, gate head) fp32
    state_in: object  # (d_v, d_k, gate head, sequence or pool slot)
    state_out: object
    seed_indices: object  # paged: per-sequence pool route
    has_initial_state: object  # paged: uint8 per sequence
    work_items: cute.Tensor  # (rows, 10) int32
    work_count: cute.Tensor  # (1,) int32
    scheduler: cute.Tensor  # (2,) int32 claim counter, zeroed by the prologue
    sGate: cute.Tensor  # (64, 2, GATE_STAGES): cumsum log2 gate, exp2 of it
    sBeta: object  # (64, BETA_STAGES)
    sO: cute.Tensor
    sKQ: cute.Tensor
    sTinv: cute.Tensor
    sA: cute.Tensor
    sV: cute.Tensor
    scale: Float32
    q_ratio: Int32
    k_ratio: Int32
    v_ratio: Int32
    units: Int32


@cute.jit
def resolve_paged_state(batch_idx, state_indices, has_initial_state):
    """``(slot, active, load_initial, clear_empty)`` of sequence ``batch_idx``: non-positive
    routes are null (zero output, pool untouched); a zero ``has_initial_state`` seeds from zero
    and, for an empty sequence, clears the slot."""
    route = state_indices[batch_idx]
    active = route > 0
    slot = route if active else Int32(0)
    load_initial = active
    clear_empty = cutlass.Boolean(False)
    if cutlass.const_expr(has_initial_state is not None):
        load_initial = active and has_initial_state[batch_idx] != 0
        clear_empty = active and has_initial_state[batch_idx] == 0
    return slot, active, load_initial, clear_empty


def _one_way(producer_arrivals: int) -> Ring:
    """A one-stage handoff whose raw producer never waits for a release (see GdnPrefill)."""
    return Ring(1, external="producer", override={"full_counts": producer_arrivals})


class GdnPrefill(WarpSpecialized):
    """Shared roles and launch of the two GDN prefill variants (see the module docstring).

    ``d_v`` is one tile's value width: ``tiles_per_head`` 2 runs every head as two 64-column
    tiles (the d_v split)."""

    tinv_source = ""

    def __init__(
        self,
        dtype_name: str,
        state_dtype_name: str,
        d_k: int,
        d_v: int,
        tiles_per_head: int,
        use_initial_state: bool,
        store_final_state: bool,
        paged_state: bool,
    ):
        if d_k not in KEY_DIMS or d_v not in VALUE_DIMS:
            raise ValueError(f"GDN prefill serves d_k in {KEY_DIMS}, d_v in {VALUE_DIMS}")
        if tiles_per_head not in (1, 2) or (tiles_per_head == 2 and d_v != 64):
            raise ValueError("the d_v split runs two 64-column tiles per head")
        if paged_state and not (use_initial_state and store_final_state):
            raise ValueError("paged state reads and writes the pool")
        self.dtype_name, self.state_dtype_name = dtype_name, state_dtype_name
        self.d_k, self.d_v, self.tiles_per_head = d_k, d_v, tiles_per_head
        self.use_initial_state, self.store_final_state = use_initial_state, store_final_state
        self.paged_state = paged_state
        self.io_dtype = _CUTE_DTYPES[dtype_name]
        compute = self.tinv_source == "compute"
        plan = self.smem_plan = SmemPlan()
        # SMEM keeps upstream's order: O and KQ first (allocated ahead of begin_roles, see
        # kernel), then the barriers, the gate/beta arrays and the 1024-byte aligned Tinv, A and
        # V tiles, which end at the 227 KB cap with d_v 128. The arrays must sit after KQ:
        # placing them ahead of it (warpdrive allocates plan buffers after its barriers)
        # measured 5% slower with the d_v split (more contended shared loads).
        gate_buf = plan.buffer("Gate", Float32, align=128)
        beta_buf = plan.buffer("Beta", Float32, align=128) if compute else None
        tinv_buf = plan.buffer("Tinv", self.io_dtype, align=1024)
        a_buf = plan.buffer("A", self.io_dtype, align=1024)
        v_buf = plan.buffer("V", self.io_dtype, align=1024)
        self.ticket = Ring(TICKET_STAGES, Warps, Warps, payload=(Int32, (1,)))
        # Each KQ stage takes 2 x d_k / 64 boxes, each V stage d_v / 64 and each tinv stage one.
        # The TMA check reports these rings unverified: it cannot follow copies through the
        # per-item domain_offset views. The MMA warp drives kq's consumer side raw (waits two
        # chunks ahead of its releases; see GdnPrefill.lookahead).
        self.kq = Ring(
            KQ_STAGES,
            Tma(2 * B_T * d_k * 2),
            external="consumer",
            override={"empty_counts": 1},
        )
        # compute1 consumes V into registers (ldmatrix) and publishes Y (tcgen05.wait::st)
        # before it releases the stage, so no generic read is pending: no proxy fence (as
        # CUTLASS's PipelineTmaAsync consumer release).
        self.v = Ring(V_STAGES, Tma(B_T * d_v * 2), Threads, release_fence=(), buffer=v_buf)
        self.gate = Ring(GATE_STAGES, Threads, Threads, buffer=gate_buf)
        if compute:
            # cp.async fills: one noinc arrival per gate lane, as the Threads count expects.
            self.beta = Ring(BETA_STAGES, Threads, Threads, buffer=beta_buf)
            self.tinv = Ring(TINV_STAGES, Threads, Umma, buffer=tinv_buf)
            self.inverse = NamedBarrier(threads=128)
        else:
            self.tinv = Ring(TINV_STAGES, Tma(TILE_BYTES), Umma, buffer=tinv_buf)
        # compute0 fences once before committing both stages of a pair.
        self.a = Ring(A_STAGES, Threads, Umma, commit_fence=(), buffer=a_buf)
        self.cg0 = Ring(CG0_STAGES, Umma, Threads, release_fence="tmem_load")
        # compute1 stmatrix-es O, then the store warp's TMA store reads it.
        self.o = Ring(1, Threads, Threads, commit_fence="smem_async")
        # The compute1 <-> MMA ping-pong over TMEM: one-way signals whose storage is released
        # by a later handoff of the same chunk (see the module docstring), as upstream. The
        # producer side is raw (an elected tcgen05.commit, or every compute1 thread's arrive
        # after tcgen05.wait::st); consumers wait managed and close with release(signal=False).
        self.state = _one_way(1)  # KV GEMM done: S (TMEM)
        self.cg1 = _one_way(1)  # KS, then U GEMM done (cg1 TMEM, two epochs per chunk)
        self.qs = _one_way(1)  # QS, then O GEMM done (qs TMEM, two epochs per chunk)
        self.qs_scaled = _one_way(128)  # QS rescaled in place: the O GEMM may accumulate
        self.state_in = _one_way(128)  # fp16 S staged (KS/QS A operand)
        self.slot0 = _one_way(128)  # Y, then U staged (two epochs per chunk)
        self.slot1 = _one_way(128)  # decayed U staged (KV A operand)
        # Upstream's TMEM map: state, q-state/O, fp16 state, cg0 x 2, cg1, Y/U and decayed U.
        self.tmem_plan = (
            TmemPlan()
            .region("state", d_k)
            .region("qs", B_T)
            .region("state_in", d_k // 2)
            .region("cg0", B_T, CG0_STAGES)
            .region("cg1", B_T)
            .region("slot0", B_T // 2)
            .region("slot1", B_T // 2)
        )
        self.work = WorkStream(schedule=OneShot(grid="1d"))

    def get_name(self) -> str:
        flags = "".join(
            c
            for c, on in (
                ("i", self.use_initial_state),
                ("f", self.store_final_state),
                ("p", self.paged_state),
            )
            if on
        )
        return (
            f"wd_gdn_prefill_{self.tinv_source}_{self.dtype_name}_dk{self.d_k}_dv{self.d_v}"
            f"x{self.tiles_per_head}_{flags or 'n'}"
        )

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer for ``heads`` gate heads: Q, K, V and O, keyed by tile
        head."""
        return tma_slot_rows(4, B_T, heads * self.tiles_per_head)

    # ---- shared helpers ----------------------------------------------------------------

    @cute.jit
    def item(self, params: PrefillParams, tile):
        """``(batch, head, write_start, write_end, compute_start, batch_start, batch_end)``."""
        wi = params.work_items
        return (
            wi[tile, _BATCH],
            wi[tile, _HEAD],
            wi[tile, _WRITE_START],
            wi[tile, _WRITE_END],
            wi[tile, _COMPUTE_START],
            wi[tile, _BATCH_START],
            wi[tile, _BATCH_END],
        )

    @cute.jit
    def head(self, head_idx):
        """``(gate/output head, first value column)`` of tile head ``head_idx``."""
        head_o = head_idx
        v_offset = Int32(0)
        if cutlass.const_expr(self.tiles_per_head > 1):
            head_o = head_idx // self.tiles_per_head
            v_offset = (head_idx - head_o * self.tiles_per_head) * self.d_v
        return head_o, v_offset

    @cute.jit
    def next_tile(self, ticket, skip_fence=None):
        """Read the next item the load warp published; returns ``(tile, ticket)``.
        ``skip_fence`` as for the consume (its reason is ``AFTER_FENCE_COVERED`` or
        ``BEFORE_FENCE_COVERED``)."""
        reason = BEFORE_FENCE_COVERED if skip_fence == "before" else AFTER_FENCE_COVERED
        with ticket.consume(skip_fence=skip_fence, reason=reason) as th:
            tile = th.data[0]
        return tile, ticket

    @cute.jit
    def _smem_ptr(self, addr):
        """16-bit SMEM pointer at 1024-byte aligned 32-bit shared address ``addr``."""
        return cute.make_ptr(self.io_dtype, addr, cute.AddressSpace.smem, assumed_align=1024)

    @cute.jit
    def _flat(self, addr):
        """Flat 16-bit SMEM tensor over one 64 x 64 tile at 32-bit shared address ``addr``."""
        return cute.make_tensor(self._smem_ptr(addr), cute.make_layout(B_T * B_T))

    @cute.jit
    def _box_copy(self, params: PrefillParams, atom, gtile, smem_addr, bar, store: bool):
        """TMA one 64 x 64 box between ``gtile`` (a TMA coordinate tile) and SMEM."""
        stile = cute.make_tensor(self._smem_ptr(smem_addr), params.box_layout)
        tSs, tSg = cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(stile, 0, 2),
            cute.group_modes(gtile, 0, 2),
        )
        if cutlass.const_expr(store):
            cute.copy(atom, tSs, tSg)
        else:
            cute.copy(atom, tSg, tSs, tma_bar_ptr=bar)

    @cute.jit
    def _load_rows(
        self, params: PrefillParams, load, token, valid, head, col, smem_addr, bar, desc
    ):
        """TMA the 64 x 64 box of rows ``[token, token + 64)``, column block ``col``; rows at or
        past token ``token + valid`` load as zeros, as from upstream's per-sequence descriptors."""
        stile = cute.make_tensor(self._smem_ptr(smem_addr), params.box_layout)
        ragged_load(load, stile, token, valid, head, col, bar, B_T, 64, desc)

    # ---- load warp ---------------------------------------------------------------------

    @cute.jit
    def load_unit(self, params: PrefillParams, unit, kq, v, tinv, ticket):
        """Every item of this CTA: Q/K, V (and tinv) loads, then the next ticket."""
        total = params.work_count[0]
        tile = unit
        while tile < total:
            batch_idx, head_idx, _ws, write_end, compute_start, batch_start, batch_end = self.item(
                params, tile
            )
            head_o, v_offset = self.head(head_idx)
            head_q = head_idx // params.q_ratio
            head_k = head_idx // params.k_ratio
            head_v = head_idx // params.v_ratio
            if write_end > compute_start:
                loads = (params.k_load, params.q_load, params.v_load)
                k_desc, q_desc, v_desc = sequence_descriptors(loads, batch_end, head_idx, B_T)
                kq = self.load_kq(
                    params,
                    kq,
                    batch_start + compute_start * B_T,
                    batch_end,
                    head_q,
                    head_k,
                    0,
                    (k_desc, q_desc),
                )
                for chunk in cutlass.range(compute_start + 1, write_end, unroll=1):
                    member = (chunk - compute_start) & 1
                    kq = self.load_kq(
                        params,
                        kq,
                        batch_start + chunk * B_T,
                        batch_end,
                        head_q,
                        head_k,
                        member,
                        (k_desc, q_desc),
                    )
                    v, tinv = self.load_v_tinv(
                        params,
                        v,
                        tinv,
                        batch_idx,
                        batch_start,
                        batch_end,
                        chunk - 1,
                        head_o,
                        head_v,
                        v_offset,
                        v_desc,
                    )
                v, tinv = self.load_v_tinv(
                    params,
                    v,
                    tinv,
                    batch_idx,
                    batch_start,
                    batch_end,
                    write_end - 1,
                    head_o,
                    head_v,
                    v_offset,
                    v_desc,
                )
            tile, ticket = self.publish(params, ticket)
        return kq, v, tinv, ticket

    @cute.jit
    def load_kq(
        self,
        params: PrefillParams,
        kq,
        token,
        batch_end,
        head_q,
        head_k,
        member,
        descs,
    ):
        """K and Q of the chunk at ``token`` into one stage: ``[K; Q]`` per 64-column subtile for
        ``member`` 0, ``[Q; K]`` for 1 (the cg0 GEMM reads ``K`` rows at the member's block).
        Rows past ``batch_end`` load as zeros; ``descs``: the item's K and Q descriptors."""
        valid = cutlass.min(B_T, batch_end - token)
        k_desc, q_desc = descs
        with kq.produce() as h:
            stage = params.sKQ.iterator.toint() + h.index * (2 * B_T * self.d_k * 2)
            for kb in cutlass.range_constexpr(self.d_k // 64):
                k_addr = stage + (kb * 2 + member) * (BOX_ELEMS * 2)
                q_addr = stage + (kb * 2 + 1 - member) * (BOX_ELEMS * 2)
                self._load_rows(
                    params, params.k_load, token, valid, head_k, kb, k_addr, h.barrier, k_desc
                )
                self._load_rows(
                    params, params.q_load, token, valid, head_q, kb, q_addr, h.barrier, q_desc
                )
        return kq

    @cute.jit
    def load_v_tinv(
        self,
        params: PrefillParams,
        v,
        tinv,
        batch_idx,
        batch_start,
        batch_end,
        chunk,
        head_o,
        head_v,
        v_offset,
        desc,
    ):
        """V (and, from GMEM, the chunk-inverse tile) of ``chunk``: one chunk behind K/Q. Rows
        past ``batch_end`` load as zeros."""
        token = batch_start + chunk * B_T
        valid = cutlass.min(B_T, batch_end - token)
        with v.produce() as h:
            h.access(write=True, by="tma")
            stage = params.sV.iterator.toint() + h.index * (B_T * self.d_v * 2)
            for box in cutlass.range_constexpr(self.d_v // 64):
                self._load_rows(
                    params,
                    params.v_load,
                    token,
                    valid,
                    head_v,
                    v_offset // 64 + box,
                    stage + box * (BOX_ELEMS * 2),
                    h.barrier,
                    desc,
                )
        if cutlass.const_expr(self.tinv_source == "gmem"):
            # gdn_tinv's row of (piece) batch_idx's chunk: batch_start // 64 + batch_idx + chunk.
            row = batch_start // B_T + batch_idx + chunk
            with tinv.produce() as h:
                h.access(write=True, by="tma")
                self._box_copy(
                    params,
                    params.tma_tinv,
                    cute.local_tile(params.mTinv[None, None, head_o, row], (B_T, B_T), (0, 0)),
                    params.sTinv.iterator.toint() + h.index * TILE_BYTES,
                    h.barrier,
                    False,
                )
        return v, tinv

    @cute.jit
    def publish(self, params: PrefillParams, ticket):
        """Claim the next item and publish it to the other warps; returns ``(tile, ticket)``."""
        th = ticket.acquire()
        with cute.arch.elect_one():
            fetched = cute.arch.atomic_add(
                params.scheduler.iterator, Int32(1), sem="relaxed", scope="gpu"
            )
            th.data[0] = params.units + fetched
        cute.arch.sync_warp()
        tile = th.data[0]
        th.commit()
        return tile, ticket

    # ---- gate warp ---------------------------------------------------------------------

    @cute.jit
    def gate_unit(self, params: PrefillParams, unit, gate, beta, ticket, lane):
        total = params.work_count[0]
        tile = unit
        while tile < total:
            _b, head_idx, _ws, write_end, compute_start, batch_start, batch_end = self.item(
                params, tile
            )
            head_o, _v_offset = self.head(head_idx)
            for chunk in cutlass.range(compute_start, write_end, unroll=1):
                gate, beta = self.gate_chunk(
                    params, gate, beta, lane, head_o, batch_start + chunk * B_T, batch_end
                )
            tile, ticket = self.next_tile(ticket)
        return gate, beta, ticket

    @cute.jit
    def gate_chunk(self, params: PrefillParams, gate, beta, lane, head_o, chunk_offset, batch_end):
        """One chunk's cumulative log2 gate and its exp2 (and, in compute mode, beta)."""
        toks = [chunk_offset + lane + 32 * col for col in range(2)]
        valid = [t < batch_end for t in toks]
        g = [
            params.gate[cutlass.min(t, batch_end - 1), head_o].to(Float32) if ok else Float32(0.0)
            for t, ok in zip(toks, valid)
        ]
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
                pos = lane + 32 * col
                params.sGate[pos, 0, gh.index] = g[col]
                params.sGate[pos, 1, gh.index] = cute.math.exp2(g[col], fastmath=True)
        if cutlass.const_expr(self.tinv_source == "compute"):
            bh = beta.acquire()
            for col in cutlass.range_constexpr(2):
                pos = lane + 32 * col
                src = params.beta.iterator + params.beta.layout((toks[col], head_o))
                dst = params.sBeta.iterator + params.sBeta.layout((pos, bh.index))
                cp_async_4b_zfill(dst.toint(), src.toint(), Int32(4) * Int32(valid[col]))
            # Each lane's arrival lands when its copies do: the 32 arrivals the count expects.
            cute.arch.cp_async_mbarrier_arrive_noinc(bh.full_barrier)
            bh.commit(signal=False)
        return gate, beta

    # ---- MMA warp ----------------------------------------------------------------------

    def _mma_descs(self):
        io, bpe = self.io_dtype, 2
        f32 = Float32

        def desc(m, n, k, **kw):
            mn_major = kw.pop("b_mn_major", False)
            idesc = umma.instr_desc(f32, io, io, m=m, n=n, b_mn_major=mn_major)
            return umma.MmaDesc(M=m, N=n, K=k, bpe_a=bpe, bpe_b=bpe, idesc=idesc, **kw)

        return {
            "kq_k": desc(2 * B_T, B_T, self.d_k),  # [K; Q] K^T (compute)
            "q_k": desc(B_T, B_T, self.d_k),  # Q K^T (gmem)
            "state_k": desc(self.d_v, B_T, self.d_k),  # KS^T, QS^T (A: fp16 S in TMEM)
            "u_a": desc(self.d_v, B_T, B_T),  # U^T = Y^T Tinv^T, O^T += U^T A^T
            "decay_u_k": desc(self.d_v, self.d_k, B_T, btranspose=True, b_mn_major=True),
        }

    @cute.jit
    def cg0_gemm(self, params: PrefillParams, kq_stage, acc_col, member):
        """cg0 GEMM of one KQ stage: ``[first; second] K^T`` at M = 128 (compute) or ``Q K^T`` at
        M = 64 (gmem); ``member`` (0/1, dynamic) selects which block holds K."""
        descs = self._mma_descs()
        box = (B_T * 64 * 2) >> 4
        seg = (2 * B_T * 64 * 2) >> 4
        steps = descs["kq_k"].steps_per_subtile
        desc_kq = umma.desc_opaque(
            umma.smem_desc(
                params.sKQ.iterator.toint() + kq_stage * (2 * B_T * self.d_k * 2),
                16,
                8 * 128,
                umma.SWIZZLE_128B,
            )
        )
        k_off = member * box
        for subtile in cutlass.range_constexpr(self.d_k // 64):
            if cutlass.const_expr(self.tinv_source == "gmem"):
                umma.mma_ss(
                    descs["q_k"],
                    desc_kq + (box - k_off + subtile * seg),
                    desc_kq + (k_off + subtile * seg),
                    acc_col,
                    subtile > 0,
                    k_count=steps,
                )
            else:
                umma.mma_ss(
                    descs["kq_k"],
                    desc_kq + subtile * seg,
                    desc_kq + (k_off + subtile * seg),
                    acc_col,
                    subtile > 0,
                    k_count=steps,
                )

    @cute.jit
    def lookahead(self, params: PrefillParams, cg0, issued, cg0_col, member, tmem):
        """The cg0 GEMM of the ``issued``-th chunk, two chunks ahead of the main GEMMs that read
        the same KQ stage. kq's consumer side is raw: this waits for the TMA fill, and the KV
        GEMM of that chunk releases the stage (protocol §7 items 1-2: one release covers every
        read of the stage, as MMAs complete in order). Returns ``(cg0, issued + 1)``."""
        # The raw kq wait below fences the GEMM.
        ch = cg0.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        tmem.access("cg0", write=True, by="umma")
        stage = issued % KQ_STAGES
        cute.arch.mbarrier_wait(self.barrier_ptr("kq") + stage, (issued // KQ_STAGES) % 2)
        tcgen05_after_thread_sync()
        self.cg0_gemm(params, stage, cg0_col + ch.index * B_T, member)
        ch.commit()
        return cg0, issued + 1

    @cute.jit
    def commit_to(self, name: str):
        """Raw producer signal of a one-way handoff: an elected ``tcgen05.commit`` of this
        warp's MMAs to the channel's full barrier."""
        with cute.arch.elect_one():
            cute.nvgpu.tcgen05.commit(self.barrier_ptr(name))

    @cute.jit
    def mma_unit(
        self,
        params: PrefillParams,
        unit,
        cg0,
        tinv,
        a,
        state_in,
        slot0,
        slot1,
        qs_scaled,
        ticket,
        issued,
        done,
        cols,
        tmem,
    ):
        descs = self._mma_descs()
        state_col, qs_col, state_in_col, cg0_col, cg1_col, inp0_col, inp1_col = cols
        box = (B_T * 64 * 2) >> 4
        seg = (2 * B_T * 64 * 2) >> 4
        steps = descs["kq_k"].steps_per_subtile
        a_seg = steps * descs["state_k"].tmem_advance_A
        kq_base = params.sKQ.iterator.toint()
        kq_stage_bytes = 2 * B_T * self.d_k * 2
        kq_empty = self.barrier_ptr("kq") + KQ_STAGES
        total = params.work_count[0]
        tile = unit
        while tile < total:
            _b, _h, _ws, write_end, compute_start, _bs, _be = self.item(params, tile)
            n_local = write_end - compute_start
            if n_local > 0:
                cg0, issued = self.lookahead(params, cg0, issued, cg0_col, 0, tmem)
            if n_local > 1:
                cg0, issued = self.lookahead(params, cg0, issued, cg0_col, 1, tmem)
            for peel in cutlass.range_constexpr(1 if self.use_initial_state else 2):
                have_state = self.use_initial_state or peel == 1
                stop = n_local
                if cutlass.const_expr(not (peel == 1 or self.use_initial_state)):
                    stop = cutlass.min(n_local, 1)
                for local in cutlass.range(peel, stop, 1, unroll=1):
                    member = local & 1
                    kq_idx = done % KQ_STAGES  # filled: its lookahead GEMM waited for it
                    kq_stage = kq_base + kq_idx * kq_stage_bytes
                    k_off = member * box
                    base_desc = umma.smem_desc(kq_stage, 16, 8 * 128, umma.SWIZZLE_128B)
                    desc_k = umma.desc_opaque(base_desc + k_off)
                    desc_q = umma.desc_opaque(base_desc + (box - k_off))
                    desc_k_trans = umma.desc_opaque(
                        umma.smem_desc(kq_stage, 2 * B_T * 128, 8 * 128, umma.SWIZZLE_128B) + k_off
                    )
                    if member == 1 and local + 2 < n_local:
                        cg0, issued = self.lookahead(params, cg0, issued, cg0_col, 1, tmem)
                    if cutlass.const_expr(have_state):
                        sh = state_in.wait()
                        tmem.access("state_in", read=True, by="umma")
                        # KS^T into cg1. Its previous epoch, U(c - 1), was read by compute1
                        # before it staged this chunk's fp16 S (§7 item 4).
                        tmem.access("cg1", write=True, by="umma")
                        for subtile in cutlass.range_constexpr(self.d_k // 64):
                            for k in cutlass.range_constexpr(steps):
                                umma.mma_ts_step(
                                    descs["state_k"],
                                    state_in_col + subtile * a_seg,
                                    desc_k + subtile * seg,
                                    cg1_col,
                                    k,
                                    cutlass.Boolean(subtile > 0 or k > 0),
                                )
                        self.commit_to("cg1")
                        # QS^T into qs. O(c - 1) there was read before this chunk's fp16 S
                        # was staged (§7 item 3).
                        tmem.access("qs", write=True, by="umma")
                        for subtile in cutlass.range_constexpr(self.d_k // 64):
                            for k in cutlass.range_constexpr(steps):
                                umma.mma_ts_step(
                                    descs["state_k"],
                                    state_in_col + subtile * a_seg,
                                    desc_q + subtile * seg,
                                    qs_col,
                                    k,
                                    cutlass.Boolean(subtile > 0 or k > 0),
                                )
                        self.commit_to("qs")
                        # compute1 restages S only after this chunk's KV GEMM (§7 item 7).
                        sh.release(signal=False)
                    # U^T = Y^T Tinv^T into cg1: compute1 published Y after reading KS (§7 4).
                    th = tinv.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    th.access(read=True, by="umma")
                    desc_tinv = umma.desc_opaque(
                        umma.smem_desc(
                            params.sTinv.iterator.toint() + th.index * TILE_BYTES,
                            16,
                            8 * 128,
                            umma.SWIZZLE_128B,
                        )
                    )
                    yh = slot0.wait()
                    tmem.access("slot0", read=True, by="umma")
                    tmem.access("cg1", write=True, by="umma")
                    for k in cutlass.range_constexpr(B_T // 16):
                        umma.mma_ts_step(
                            descs["u_a"], inp0_col, desc_tinv, cg1_col, k, cutlass.Boolean(k > 0)
                        )
                    self.commit_to("cg1")
                    th.release()
                    # compute1 overwrites Y with U after this GEMM's cg1 signal (§7 item 5).
                    yh.release(signal=False)
                    if member == 0 and local + 2 < n_local:
                        cg0, issued = self.lookahead(params, cg0, issued, cg0_col, 0, tmem)
                    # O^T (+)= U^T A^T on the rescaled QS^T.
                    ah = a.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    ah.access(read=True, by="umma")
                    desc_a = umma.desc_opaque(
                        umma.smem_desc(
                            params.sA.iterator.toint() + ah.index * TILE_BYTES,
                            16,
                            8 * 128,
                            umma.SWIZZLE_128B,
                        )
                    )
                    if cutlass.const_expr(have_state):
                        qh = qs_scaled.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    ui = slot0.wait()
                    tmem.access("slot0", read=True, by="umma")
                    tmem.access("qs", read=True, write=True, by="umma")
                    for k in cutlass.range_constexpr(B_T // 16):
                        umma.mma_ts_step(
                            descs["u_a"],
                            inp0_col,
                            desc_a,
                            qs_col,
                            k,
                            cutlass.Boolean(True) if k > 0 else cutlass.Boolean(have_state),
                        )
                    ah.release()
                    self.commit_to("qs")
                    if cutlass.const_expr(have_state):
                        qh.release(signal=False)  # the next QS follows the next fp16 S (§7 3)
                    # The next Y follows this chunk's KV GEMM, hence this O GEMM (§7 item 5).
                    ui.release(signal=False)
                    # S^T (+)= (decayed U)^T K. compute1 rescaled S before staging the decayed
                    # U (§7 item 8).
                    dh = slot1.wait()
                    tmem.access("slot1", read=True, by="umma")
                    tmem.access("state", read=True, write=True, by="umma")
                    for k in cutlass.range_constexpr(B_T // 16):
                        umma.mma_ts_step(
                            descs["decay_u_k"],
                            inp1_col,
                            desc_k_trans,
                            state_col,
                            k,
                            cutlass.Boolean(True) if k > 0 else cutlass.Boolean(have_state),
                        )
                    self.commit_to("state")
                    with cute.arch.elect_one():  # releases the KQ stage (§7 item 2)
                        cute.nvgpu.tcgen05.commit(kq_empty + kq_idx)
                    done = done + 1
                    # The next decayed U follows this chunk's KV GEMM (§7 item 6).
                    dh.release(signal=False)
            tile, ticket = self.next_tile(ticket)
        return cg0, tinv, a, state_in, slot0, slot1, qs_scaled, ticket, issued, done

    @role(warps=10, regs=24)
    @cute.jit
    def mma(
        self,
        params: PrefillParams,
        *,
        cg0: Producer,
        tinv: Consumer,
        a: Consumer,
        state_in: Consumer,
        slot0: Consumer,
        slot1: Consumer,
        qs_scaled: Consumer,
        ticket: Consumer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        cols = tuple(
            tmem.ptr(name).toint()
            for name in ("state", "qs", "state_in", "cg0", "cg1", "slot0", "slot1")
        )
        issued = Int32(0)  # cg0 GEMMs issued: the epoch of kq's raw lookahead waits
        done = Int32(0)  # KV GEMMs issued: the epoch of kq's raw releases
        while work.valid:
            unit, _, _ = work.item
            (cg0, tinv, a, state_in, slot0, slot1, qs_scaled, ticket, issued, done) = (
                self.mma_unit(
                    params,
                    unit,
                    cg0,
                    tinv,
                    a,
                    state_in,
                    slot0,
                    slot1,
                    qs_scaled,
                    ticket,
                    issued,
                    done,
                    cols,
                    tmem,
                )
            )
            work.advance()
        return cg0, tinv, a, state_in, slot0, slot1, qs_scaled, ticket, work

    # ---- compute group 1 ---------------------------------------------------------------

    @role(warps=range(4, 8), regs=256)
    @cute.jit
    def compute1(
        self,
        params: PrefillParams,
        *,
        gate: Consumer,
        v: Consumer,
        state: Consumer,
        qs: Consumer,
        cg1: Consumer,
        o: Producer,
        ticket: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        while work.valid:
            unit, _, _ = work.item
            gate, v, state, qs, cg1, o, ticket = self.cg1_unit(
                params, unit, gate, v, state, qs, cg1, o, ticket, tmem, me
            )
            work.advance()
        return gate, v, state, qs, cg1, o, ticket, work

    @cute.jit
    def cg1_unit(self, params: PrefillParams, unit, gate, v, state, qs, cg1, o, ticket, tmem, me):
        tid = me.tid  # TMEM lane = d_v row of S^T, QS^T, U^T and O^T
        lane0 = (32 * (me.tmem_row // 32)) << 16
        state_col = tmem.ptr("state").toint()
        if cutlass.const_expr(self.d_v == 128):
            state_gmem_row = tid
            state_row_valid = cutlass.Boolean(True)
        else:
            state_gmem_row = (tid // 32) * 16 + (tid % 32) % 16
            state_row_valid = (tid % 32) < 16
        total = params.work_count[0]
        tile = unit
        while tile < total:
            batch_idx, head_idx, _ws, write_end, compute_start, _bs, _be = self.item(params, tile)
            head_o, v_offset = self.head(head_idx)
            seed_row = batch_idx
            state_active = cutlass.Boolean(True)
            load_initial = cutlass.Boolean(True)
            clear_empty = cutlass.Boolean(False)
            if cutlass.const_expr(self.paged_state):
                seed_row, state_active, load_initial, clear_empty = resolve_paged_state(
                    batch_idx, params.seed_indices, params.has_initial_state
                )
            n_local = write_end - compute_start
            if n_local > 0:
                if cutlass.const_expr(self.use_initial_state):
                    seed = compute_start == 0
                    if cutlass.const_expr(self.paged_state):
                        seed = seed and load_initial
                    self.seed_state(
                        params,
                        seed,
                        state_gmem_row + v_offset,
                        head_o,
                        seed_row,
                        lane0 + state_col,
                    )
                # Chunk 0 (upstream's peel): no state epoch to wait for; with an initial state
                # it restages the seed, without one it has no state at all.
                gate, v, state, qs, cg1, o = self.cg1_chunk(
                    params,
                    gate,
                    v,
                    state,
                    qs,
                    cg1,
                    o,
                    tmem,
                    me,
                    state_active,
                    self.use_initial_state,
                    False,
                )
                for _local in cutlass.range(1, n_local, 1, unroll=1):
                    gate, v, state, qs, cg1, o = self.cg1_chunk(
                        params, gate, v, state, qs, cg1, o, tmem, me, state_active, True, True
                    )
                # The last KV GEMM's state: the final state. The next item's KV GEMMs follow
                # this item's last loads in compute1's order (§7 items 5-7).
                # Without a final state store no TMEM load follows before the next fenced wait.
                skip = None if self.store_final_state else "after"
                sh = state.wait(skip_fence=skip, reason=AFTER_FENCE_COVERED)
                tmem.access("state", read=True)
                if cutlass.const_expr(self.store_final_state):
                    store_final = state_active
                    final_row = seed_row
                    if cutlass.const_expr(not self.paged_state):
                        final_row = params.work_items[tile, _FINAL_DST]
                        store_final = final_row >= 0
                    if store_final:
                        self.store_state(
                            params,
                            state_row_valid,
                            state_gmem_row + v_offset,
                            head_o,
                            final_row,
                            lane0 + state_col,
                        )
                sh.release(signal=False)
            else:
                if cutlass.const_expr(self.store_final_state):
                    self.empty_state(
                        params,
                        tile,
                        state_active,
                        clear_empty,
                        state_row_valid,
                        state_gmem_row + v_offset,
                        head_o,
                        seed_row,
                    )
            tile, ticket = self.next_tile(ticket)
        return gate, v, state, qs, cg1, o, ticket

    @cute.jit
    def publish_tmem(self, name: str):
        """Raw producer signal of a compute1 -> MMA handoff: every thread completes its TMEM
        accesses and orders them before its arrival on the channel's full barrier (128
        arrivals)."""
        cute.arch.fence_view_async_tmem_store()
        tcgen05_before_thread_sync()
        cute.arch.mbarrier_arrive(self.barrier_ptr(name))

    @cute.jit
    def seed_state(self, params: PrefillParams, seed, row, head_o, seq, taddr):
        """Seed S^T in TMEM: row ``row`` of ``state_in`` (32 columns per x32 store), or zeros."""
        src_t = params.state_in
        vw = 16 // (src_t.element_type.width // 8)
        if seed:
            elem = src_t.element_type
            base = src_t[row, None, head_o, seq].iterator.toint()
            for i in cutlass.range_constexpr(self.d_k // 32):
                words = []
                for gi in cutlass.range_constexpr(32 // vw):
                    chunk = cute.make_rmem_tensor((vw,), elem)
                    src = cute.make_tensor(
                        cute.make_ptr(
                            elem,
                            base + (i * 32 + gi * vw) * (elem.width // 8),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(vw),
                    )
                    cute.autovec_copy(src, chunk)
                    for t in cutlass.range_constexpr(vw):
                        words.append(chunk[t].to(Float32))
                tmem_st("32x32b", taddr + i * 32, words)
            cute.arch.fence_view_async_tmem_store()
        else:
            zero = opaque_f32_zero()  # a register: tcgen05.st takes no immediates
            for i in cutlass.range_constexpr(self.d_k // 32):
                tmem_st("32x32b", taddr + i * 32, [zero for _ in range(32)])
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_state(self, params: PrefillParams, row_valid, row, head_o, seq, taddr):
        """Final S^T row ``row`` from TMEM to ``state_out`` (16-byte vectors)."""
        dst_t = params.state_out
        out = dst_t.element_type
        vw = 16 // (out.width // 8)
        base = dst_t[row, None, head_o, seq].iterator.toint()
        for i in cutlass.range_constexpr(self.d_k // 32):
            vals = tmem_ld("32x32b", 32, taddr + i * 32)
            if row_valid:
                for gi in cutlass.range_constexpr(32 // vw):
                    frag = cute.make_rmem_tensor((vw,), out)
                    for t in cutlass.range_constexpr(vw):
                        frag[t] = vals[gi * vw + t].to(out)
                    dst = cute.make_tensor(
                        cute.make_ptr(
                            out,
                            base + (i * 32 + gi * vw) * (out.width // 8),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(vw),
                    )
                    cute.autovec_copy(frag, dst)
        # The CTA's last item has no later TMEM store to order these loads before teardown
        # frees TMEM.
        cute.arch.fence_view_async_tmem_load()

    @cute.jit
    def empty_state(
        self,
        params: PrefillParams,
        tile,
        state_active,
        clear_empty,
        row_valid,
        row,
        head_o,
        seq,
    ):
        """An item without chunks (a paged fresh route, or an empty table row): the exit state
        is the entry state (or zero)."""
        write = state_active and clear_empty
        final_row = seq
        if cutlass.const_expr(not self.paged_state):
            final_row = params.work_items[tile, _FINAL_DST]
            write = final_row >= 0
        if write:
            g_out = params.state_out[None, None, head_o, final_row]
            out = params.state_out.element_type
            if cutlass.const_expr(self.use_initial_state and not self.paged_state):
                g_in = params.state_in[None, None, head_o, seq]
                for c in cutlass.range(self.d_k, unroll_full=True):
                    if row_valid:
                        g_out[row, c] = g_in[row, c].to(out)
            else:
                for c in cutlass.range(self.d_k, unroll_full=True):
                    if row_valid:
                        g_out[row, c] = Float32(0.0).to(out)

    @cute.jit
    def cg1_chunk(
        self,
        params: PrefillParams,
        gate,
        v,
        state,
        qs,
        cg1,
        o,
        tmem,
        me,
        state_active,
        valid_state: cutlass.Constexpr[bool],
        wait_state: cutlass.Constexpr[bool],
    ):
        """One chunk of compute group 1: restage and rescale S, Y = V - KS, rescale QS, publish
        U and decayed U, and stage O. Returns the endpoints."""
        dtype = self.io_dtype
        tid, lane = me.tid, me.lane
        warp_lane = 32 * (me.tmem_row // 32)
        lane0 = warp_lane << 16
        state_col = tmem.ptr("state").toint()
        state_in_col = tmem.ptr("state_in").toint()
        qs_col = tmem.ptr("qs").toint()
        cg1_col = tmem.ptr("cg1").toint()
        inp0_col = tmem.ptr("slot0").toint()
        inp1_col = tmem.ptr("slot1").toint()
        dv_halves = self.d_v // 64
        n_ld = self.d_k // 32
        v_o_row = tid % 8 + (tid // 16 % 2) * 8
        if cutlass.const_expr(self.d_v == 128):
            o_col_base = (tid // 32 % 2) * 32
            v_o_segment = (tid // 64) * (B_T * 64)
        else:
            o_col_base = (tid // 32) * 16
            v_o_segment = 0
        v_o_col = o_col_base + (tid // 8 % 2) * 8
        sG = params.sGate

        # Only a state restage without its own fenced state wait reads TMEM next.
        skip = None if valid_state and not wait_state else "after"
        gh = gate.wait(skip_fence=skip, reason=AFTER_FENCE_COVERED)
        g_idx = gh.index
        cumprod_total = sG[B_T - 1, 1, g_idx]
        # ---- state restage (fp16 S for KS/QS) and rescale S *= exp2(g[63]) --------------
        if cutlass.const_expr(valid_state):
            if cutlass.const_expr(wait_state):
                sh = state.wait()
            tmem.access("state", read=True, write=True)
            tmem.access("state_in", write=True)  # KS/QS(c - 1) read it before KV(c - 1) (§7 7)
            state_halves = []
            state_regs = [[Float32(0.0) for _ in range(n_ld)] for _ in range(32)]
            if cutlass.const_expr(self.d_v == 64):
                for hh in cutlass.range_constexpr(self.d_k // 64):
                    state_halves.append(tmem_ld("16x256b", 8, lane0 + state_col + hh * 64))
                pack = [
                    fp32_to_fp16(state_halves[hh][2 * j], state_halves[hh][2 * j + 1], dtype=dtype)
                    for hh in range(self.d_k // 64)
                    for j in range(16)
                ]
                tmem_st("16x128b", lane0 + state_in_col, pack)
            else:
                for i in cutlass.range_constexpr(n_ld):
                    vec = tmem_ld("32x32b", 32, lane0 + state_col + i * 32)
                    for k in cutlass.range_constexpr(32):
                        state_regs[k][i] = vec[k]
                for i in cutlass.range_constexpr(n_ld):
                    pack = [
                        fp32_to_fp16(state_regs[2 * j][i], state_regs[2 * j + 1][i], dtype=dtype)
                        for j in range(16)
                    ]
                    tmem_st("32x32b", lane0 + state_in_col + i * 16, pack)
            self.publish_tmem("state_in")
            if cutlass.const_expr(self.d_v == 64):
                for hh in cutlass.range_constexpr(self.d_k // 64):
                    scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(
                            state_halves[hh][2 * j],
                            state_halves[hh][2 * j + 1],
                            cumprod_total,
                            cumprod_total,
                        )
                        scaled += [s0, s1]
                    tmem_st("16x256b", lane0 + state_col + hh * 64, scaled)
            else:
                for i in cutlass.range_constexpr(n_ld):
                    scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(
                            state_regs[2 * j][i],
                            state_regs[2 * j + 1][i],
                            cumprod_total,
                            cumprod_total,
                        )
                        scaled += [s0, s1]
                    tmem_st("32x32b", lane0 + state_col + i * 32, scaled)
            # The KV GEMM accumulates S only after the decayed U, published after this (§7 8).
            cute.arch.fence_view_async_tmem_store()
            if cutlass.const_expr(wait_state):
                sh.release(signal=False)

        # ---- per-row gate vectors ----------------------------------------------------------
        cumprod_vals = [sG[(lane % 4) * 2 + ((k // 4) * 8 + k % 2), 1, g_idx] for k in range(32)]
        last_cs = sG[B_T - 1, 0, g_idx]
        cs_vals = [sG[(lane % 4) * 2 + ((k // 4) * 8 + k % 2), 0, g_idx] for k in range(32)]
        decay_scale = []
        for k in cutlass.range_constexpr(0, 32, 2):
            d0, d1 = fadd2(last_cs, last_cs, -cs_vals[k], -cs_vals[k + 1])
            decay_scale.append(cute.math.exp2(d0, fastmath=True))
            decay_scale.append(cute.math.exp2(d1, fastmath=True))
        gh.release()

        # ---- Y = V - fp16(KS * exp2(g)) ------------------------------------------------------
        # With a state, the restage's wait fenced the TMEM accesses that follow.
        skip = "both" if valid_state else "before"
        reason = BOTH_FENCES_COVERED if valid_state else BEFORE_FENCE_COVERED
        vh = v.wait(skip_fence=skip, reason=reason)
        v_stage = params.sV.iterator.toint() + vh.index * (B_T * self.d_v * 2)
        v_frag = []
        for half in cutlass.range_constexpr(dv_halves):
            words = []
            for block in cutlass.range_constexpr(4):
                row = v_o_row + block * 16
                off = v_o_segment + row * 64 + swizzle_xor_128b(row, v_o_col + half * 16)
                words += ldmatrix(v_stage + off * 2, 4, trans=True)
            v_frag.append(words)
        if cutlass.const_expr(valid_state):
            kh = cg1.wait()  # KS
            tmem.access("cg1", read=True)
            for half in cutlass.range_constexpr(dv_halves):
                ks = tmem_ld("16x256b", 8, ((warp_lane + half * 16) << 16) + cg1_col)
                for j in cutlass.range_constexpr(16):
                    s0, s1 = fmul2(
                        ks[2 * j], ks[2 * j + 1], cumprod_vals[2 * j], cumprod_vals[2 * j + 1]
                    )
                    v_frag[half][j] = sub_f16x2(
                        v_frag[half][j], fp32_to_fp16(s0, s1, dtype=dtype), dtype=dtype
                    )
            kh.release(signal=False)  # U(c) overwrites KS after Y is published (§7 item 4)
        # The U input of c - 1 was read by O(c - 1), done before KV(c - 1) (§7 item 5).
        tmem.access("slot0", write=True)
        for half in cutlass.range_constexpr(dv_halves):
            tmem_st("16x128b", ((warp_lane + half * 16) << 16) + inp0_col, v_frag[half])
        self.publish_tmem("slot0")

        # ---- QS *= exp2(g) * scale, in place ---------------------------------------------
        if cutlass.const_expr(valid_state):
            qh = qs.wait()
            tmem.access("qs", read=True, write=True)
            q_addrs = [((warp_lane + half * 16) << 16) + qs_col for half in range(dv_halves)]
            q_vecs = [tmem_ld("16x256b", 8, q_addrs[half]) for half in range(dv_halves)]
            for half in cutlass.range_constexpr(dv_halves):
                scaled = []
                for j in cutlass.range_constexpr(16):
                    p0, p1 = fmul2(
                        q_vecs[half][2 * j],
                        q_vecs[half][2 * j + 1],
                        cumprod_vals[2 * j],
                        cumprod_vals[2 * j + 1],
                    )
                    s0, s1 = fmul2(p0, p1, params.scale, params.scale)
                    scaled += [s0, s1]
                tmem_st("16x256b", q_addrs[half], scaled)
            self.publish_tmem("qs_scaled")
            qh.release(signal=False)  # the O GEMM accumulates after qs_scaled

        # ---- U and decayed U ---------------------------------------------------------------
        uh = cg1.wait()  # U
        tmem.access("cg1", read=True)
        vh.release()
        u_vecs = [
            tmem_ld("16x256b", 8, ((warp_lane + half * 16) << 16) + cg1_col)
            for half in range(dv_halves)
        ]
        uh.release(signal=False)  # KS(c + 1) follows the next fp16 S (§7 item 4)
        u_regs = [[u_vecs[h][k] for h in range(dv_halves)] for k in range(32)]
        tmem.access("slot0", write=True)  # Y was read: the U GEMM signalled cg1 (§7 item 5)
        for half in cutlass.range_constexpr(dv_halves):
            pack = [
                fp32_to_fp16(u_regs[2 * j][half], u_regs[2 * j + 1][half], dtype=dtype)
                for j in range(16)
            ]
            tmem_st("16x128b", ((warp_lane + half * 16) << 16) + inp0_col, pack)
        self.publish_tmem("slot0")
        tmem.access("slot1", write=True)  # KV(c - 1) read it before signalling state (§7 6)
        for half in cutlass.range_constexpr(dv_halves):
            for j in cutlass.range_constexpr(16):
                u_regs[2 * j][half], u_regs[2 * j + 1][half] = fmul2(
                    u_regs[2 * j][half],
                    u_regs[2 * j + 1][half],
                    decay_scale[2 * j],
                    decay_scale[2 * j + 1],
                )
            pack = [
                fp32_to_fp16(u_regs[2 * j][half], u_regs[2 * j + 1][half], dtype=dtype)
                for j in range(16)
            ]
            tmem_st("16x128b", ((warp_lane + half * 16) << 16) + inp1_col, pack)
        self.publish_tmem("slot1")

        # ---- O: TMEM -> swizzled SMEM for the TMA store ------------------------------------
        oh = qs.wait()  # O
        tmem.access("qs", read=True)
        o_regs = [
            tmem_ld("16x256b", 8, ((warp_lane + half * 16) << 16) + qs_col)
            for half in range(dv_halves)
        ]
        cute.arch.fence_view_async_tmem_load()
        oh.release(signal=False)  # QS(c + 1) follows the next fp16 S (§7 item 3)
        if cutlass.const_expr(self.paged_state):
            # Null-route padding may hold NaNs: select, never multiply, to emit zeros.
            o_regs = [[x if state_active else Float32(0.0) for x in regs] for regs in o_regs]
        ob = o.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        o_stage = params.sO.iterator.toint()
        for half in cutlass.range_constexpr(dv_halves):
            for block in cutlass.range_constexpr(4):
                pack = [
                    fp32_to_fp16(
                        o_regs[half][8 * block + 2 * j],
                        o_regs[half][8 * block + 2 * j + 1],
                        dtype=dtype,
                    )
                    for j in range(4)
                ]
                row = v_o_row + block * 16
                off = v_o_segment + row * 64 + swizzle_xor_128b(row, v_o_col + half * 16)
                stmatrix(o_stage + off * 2, pack, trans=True)
        ob.commit()
        return gate, v, state, qs, cg1, o

    # ---- store warp --------------------------------------------------------------------

    @role(warps=11, regs=24)
    @cute.jit
    def store(self, params: PrefillParams, *, o: Consumer, ticket: Consumer, work: WorkCursor):
        while work.valid:
            unit, _, _ = work.item
            o, ticket = self.store_unit(params, unit, o, ticket)
            work.advance()
        bulk_wait_group(0)
        return o, ticket, work

    @cute.jit
    def store_unit(self, params: PrefillParams, unit, o, ticket):
        total = params.work_count[0]
        tile = unit
        while tile < total:
            _b, head_idx, write_start, write_end, compute_start, batch_start, batch_end = (
                self.item(params, tile)
            )
            item = (head_idx, write_start, write_end, compute_start, batch_start, batch_end)
            o = self.store_item(params, o, item)
            tile, ticket = self.next_tile(ticket)
        return o, ticket

    @cute.jit
    def store_item(self, params: PrefillParams, o, item):
        """TMA-store the staged O chunks of one item from ``write_start`` on."""
        head_idx, write_start, write_end, compute_start, batch_start, batch_end = item
        head_o, v_offset = self.head(head_idx)
        moves_rows = write_end > compute_start
        (desc,) = sequence_descriptors((params.o_store,), batch_end, head_idx, B_T, moves_rows)
        for chunk in cutlass.range(compute_start, write_end, unroll=1):
            with o.consume():
                if chunk >= write_start:  # warm-up chunks stage O but store nothing
                    tok0 = batch_start + chunk * B_T
                    valid = cutlass.min(B_T, batch_end - tok0)
                    for box in cutlass.range_constexpr(self.d_v // 64):
                        stile = cute.make_tensor(
                            self._smem_ptr(params.sO.iterator.toint() + box * (BOX_ELEMS * 2)),
                            params.box_layout,
                        )
                        col = v_offset // 64 + box
                        ragged_store(
                            params.o_store, stile, tok0, valid, head_o, col, B_T, 64, desc
                        )
                    bulk_commit_group()
                    bulk_wait_group(0, read=True)
        return o

    # ---- kernel and entry point --------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        q_load,
        k_load,
        v_load,
        o_store,
        tma_tinv,
        mTinv,
        box_layout,
        gate,
        beta,
        state_in,
        state_out,
        seed_indices,
        has_initial_state,
        work_items,
        work_count,
        scheduler,
        scale,
        q_ratio,
        k_ratio,
        v_ratio,
        units,
        gate_layout,
        beta_layout,
        o_layout,
        kq_layout,
        tinv_layout,
        a_layout,
        v_layout,
        bound,
    ):
        # The prologue that writes the item table and zeroes the claim counter precedes this
        # launch (programmatic dependent launch).
        cute.arch.griddepcontrol_wait()
        compute = self.tinv_source == "compute"
        # O and KQ ahead of the layer's barriers (see __init__): every SmemAllocator draws
        # from the kernel's one running allocation, in program order.
        lead = SmemAllocator()
        sO = lead.allocate_tensor(self.io_dtype, o_layout, byte_alignment=1024)
        sKQ = lead.allocate_tensor(self.io_dtype, kq_layout, byte_alignment=1024)
        layouts = {
            "Gate": gate_layout,
            "Tinv": tinv_layout,
            "A": a_layout,
            "V": v_layout,
        }
        if cutlass.const_expr(compute):
            layouts["Beta"] = beta_layout
        self.begin_roles(work=bound, layouts=layouts)
        params = PrefillParams(
            q_load=q_load,
            k_load=k_load,
            v_load=v_load,
            o_store=o_store,
            tma_tinv=tma_tinv,
            mTinv=mTinv,
            box_layout=box_layout,
            gate=gate,
            beta=beta,
            state_in=state_in,
            state_out=state_out,
            seed_indices=seed_indices,
            has_initial_state=has_initial_state,
            work_items=work_items,
            work_count=work_count,
            scheduler=scheduler,
            sGate=self.buffer("Gate"),
            sBeta=self.buffer("Beta") if compute else None,
            sO=sO,
            sKQ=sKQ,
            sTinv=self.buffer("Tinv"),
            sA=self.buffer("A"),
            sV=self.buffer("V"),
            scale=scale,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
            v_ratio=v_ratio,
            units=units,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor | None,
        o: cute.Tensor,
        state_in: cute.Tensor | None,
        state_out: cute.Tensor | None,
        tinv: cute.Tensor | None,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        scheduler: cute.Tensor,
        tma_slots: cute.Tensor,
        scale: Float32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.launch(
            q,
            k,
            v,
            gate,
            beta,
            o,
            state_in,
            state_out,
            None,
            None,
            tinv,
            work_items,
            work_count,
            scheduler,
            tma_slots,
            scale,
            max_ctas,
            stream,
        )

    @cute.jit
    def launch(
        self,
        q: cute.Tensor,  # (tokens, q heads, d_k)
        k: cute.Tensor,
        v: cute.Tensor,  # (tokens, v heads, d_v * tiles_per_head)
        gate: cute.Tensor,  # (tokens, heads) fp32
        beta: cute.Tensor | None,  # compute: (tokens, heads) fp32
        o: cute.Tensor,  # (tokens, heads, d_v * tiles_per_head)
        state_in: cute.Tensor | None,  # (sequences, heads, d_v * tiles_per_head, d_k)
        state_out: cute.Tensor | None,
        seed_indices: cute.Tensor | None,
        has_initial_state: cute.Tensor | None,
        tinv: cute.Tensor | None,  # gmem: (tinv rows, heads, 64, 64)
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        scheduler: cute.Tensor,
        tma_slots: cute.Tensor,  # (>= op.tma_slot_rows(heads_out), 16) int64
        scale: Float32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on ``max_ctas`` CTAs after the item-table prologue; callable from another
        ``@cute.jit`` host, such as a fused launch bundle."""
        io = self.io_dtype
        heads_out = Int32(gate.shape[1]) * self.tiles_per_head
        box = sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1)
        box_layout = cute.select(box, mode=[0, 1])
        load = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        def token_major(t):  # (tokens, heads, dim) -> (tokens, dim, heads)
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))

        def ragged(op, t, index):  # boxes clipped at each sequence end, keyed by tile head
            slots = slot_region(tma_slots, index, B_T, heads_out)
            return make_ragged_tma(op, token_major(t), box_layout, B_T, 64, slots)

        q_load, k_load, v_load = (ragged(load, t, i) for i, t in enumerate((q, k, v)))
        o_store = ragged(cpasync.CopyBulkTensorTileS2GOp(), o, 3)
        tma_tinv, mTinv = None, None
        if cutlass.const_expr(self.tinv_source == "gmem"):
            tinv_view = cute.make_tensor(
                tinv.iterator, cute.select(tinv.layout, mode=[2, 3, 1, 0])
            )
            tma_tinv, mTinv = cpasync.make_tiled_tma_atom(load, tinv_view, box_layout, (B_T, B_T))

        def state_view(t):  # (seq, heads, d_v, d_k) -> (d_v, d_k, heads, seq)
            if cutlass.const_expr(t is None):
                return None
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[2, 3, 1, 0]))

        units = cutlass.max(max_ctas, 1)
        bound = self.work.bind((units, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            q_load,
            k_load,
            v_load,
            o_store,
            tma_tinv,
            mTinv,
            box_layout,
            gate,
            beta,
            state_view(state_in),
            state_view(state_out),
            seed_indices,
            has_initial_state,
            work_items,
            work_count,
            scheduler,
            scale,
            heads_out // Int32(q.shape[1]),
            heads_out // Int32(k.shape[1]),
            heads_out // Int32(v.shape[1]),
            units,
            cute.make_layout((B_T, 2, GATE_STAGES)),
            cute.make_layout((B_T, BETA_STAGES)),
            cute.make_layout((B_T * self.d_v, 1)),
            cute.make_layout((2 * B_T * self.d_k, KQ_STAGES)),
            cute.make_layout((B_T * B_T, TINV_STAGES)),
            cute.make_layout((B_T * B_T, A_STAGES)),
            cute.make_layout((B_T * self.d_v, V_STAGES)),
            bound,
        ).launch(**self.launch_config(bound), stream=stream, use_pdl=True)


@cute.jit
def decay_tile(sG, g_idx, rows, lane, mask_zero):
    """``exp2(g[row] - g[col])`` (``row >= col``, else zero) at the 16x256b accumulator
    fragment of 16-row bands ``rows`` (pairs of lo/hi rows) and this lane's columns."""
    cols = []
    for gi in cutlass.range_constexpr(8):
        for bi in cutlass.range_constexpr(2):
            cols.append(sG[(lane % 4) * 2 + gi * 8 + bi, 0, g_idx])
    row_cs = [sG[r, 0, g_idx] for r in rows]
    decay = []
    for u in cutlass.range_constexpr(len(rows) // 2):
        for k in cutlass.range_constexpr(32):
            hi = ((k // 2) % 2) == 1
            row = rows[2 * u + (1 if hi else 0)]
            col = (lane % 4) * 2 + ((k // 4) * 8 + k % 2)
            decay.append(
                cute.math.exp2(
                    row_cs[2 * u + (1 if hi else 0)] - cols[(k // 4) * 2 + k % 2], fastmath=True
                )
                if row >= col
                else mask_zero
            )
    return decay


@cute.jit
def store_scaled(tile_addr, st_row, col0, vals, decay, factors, dtype: cutlass.Constexpr):
    """``vals * decay * factor`` (factor per value pair) as 16-bit, stmatrix into row ``st_row``
    of a swizzled 64 x 64 tile (the 16 x 64 band of one 16x256b fragment)."""
    pack = []
    for k in cutlass.range_constexpr(16):
        p0, p1 = fmul2(vals[2 * k], vals[2 * k + 1], decay[2 * k], decay[2 * k + 1])
        f = factors[k]
        v0, v1 = fmul2(p0, p1, f, f)
        pack.append(fp32_to_fp16(v0, v1, dtype=dtype))
    for c in cutlass.range_constexpr(B_T // 16):
        off = st_row * B_T + swizzle_xor_128b(st_row, col0 + c * 16)
        stmatrix(tile_addr + off * 2, pack[c * 4 : c * 4 + 4])


class GdnPrefillCompute(GdnPrefill):
    """Warmup/uncut/d_v-split prefill: the chunk inverse is built in compute group 0."""

    tinv_source = "compute"

    @role(warps=9, regs=24)
    @cute.jit
    def load(
        self,
        params: PrefillParams,
        *,
        kq: Producer,
        v: Producer,
        ticket: Producer,
        work: WorkCursor,
    ):
        while work.valid:
            unit, _, _ = work.item
            kq, v, _tinv, ticket = self.load_unit(params, unit, kq, v, None, ticket)
            work.advance()
        kq.tail()
        v.tail()
        cute.arch.griddepcontrol_launch_dependents()
        return kq, v, ticket, work

    @role(warps=8, regs=24)
    @cute.jit
    def gate_load(
        self,
        params: PrefillParams,
        *,
        gate: Producer,
        beta: Producer,
        ticket: Consumer,
        work: WorkCursor,
        me: Local,
    ):
        while work.valid:
            unit, _, _ = work.item
            gate, beta, ticket = self.gate_unit(params, unit, gate, beta, ticket, me.lane)
            work.advance()
        return gate, beta, ticket, work

    @role(warps=range(4), regs=224)
    @cute.jit
    def compute0(
        self,
        params: PrefillParams,
        *,
        gate: Consumer,
        beta: Consumer,
        cg0: Consumer,
        tinv: Producer,
        a: Producer,
        ticket: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = unit
            while tile < total:
                _b, _h, _ws, write_end, compute_start, _bs, _be = self.item(params, tile)
                n_local = write_end - compute_start
                n_pairs = n_local // 2
                for _p in cutlass.range(n_pairs, unroll=1):
                    gate, beta, cg0, tinv, a = self.cg0_pair(
                        params, True, gate, beta, cg0, tinv, a, tmem, me
                    )
                if n_local - n_pairs * 2 == 1:
                    gate, beta, cg0, tinv, a = self.cg0_pair(
                        params, False, gate, beta, cg0, tinv, a, tmem, me
                    )
                tile, ticket = self.next_tile(ticket, skip_fence="before")
            work.advance()
        return gate, beta, cg0, tinv, a, ticket, work

    @cute.jit
    def cg0_pair(
        self,
        params: PrefillParams,
        have_m1: cutlass.Constexpr[bool],
        gate,
        beta,
        cg0,
        tinv,
        a,
        tmem,
        me,
    ):
        """One chunk pair: both KK epilogues, the pair inverse, the beta scaling and both A
        epilogues. Warp half ``h`` handles the KK rows of member ``h`` and the A rows of member
        ``1 - h`` (the KQ stages put them in the same TMEM lanes). ``have_m1`` False is an odd
        item's last chunk: half 0 does its KK, half 1 its A, warps 0-1 its inverse."""
        dtype = self.io_dtype
        warp, lane, tid = me.warp, me.lane, me.tid
        inverse_local_warp = warp % 2
        pair_half = warp // 2
        half_row_base = inverse_local_warp * 32
        store_row = warp * 16 + lane % 16
        store_col = (lane // 16) * 8
        store_row_frag = lane % 16
        mask_zero = opaque_f32_zero()
        lane_lo = (32 * (me.tmem_row // 32)) << 16
        lane_hi = (32 * (me.tmem_row // 32) + 16) << 16
        cg0_col = tmem.ptr("cg0").toint()
        sG, sB = params.sGate, params.sBeta
        do_a = have_m1 or pair_half == 1
        rows = (
            half_row_base + lane // 4,
            half_row_base + lane // 4 + 8,
            half_row_base + lane // 4 + 16,
            half_row_base + lane // 4 + 24,
        )

        g0 = gate.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
        g_idx0 = g0.index
        g_idx1 = g_idx0
        if cutlass.const_expr(have_m1):
            g1 = gate.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
            g_idx1 = g1.index
        kk_g = g_idx1 if pair_half == 1 else g_idx0
        a_g = g_idx0 if pair_half == 1 else g_idx1
        decay_kk = decay_tile(sG, kk_g, rows, lane, mask_zero)
        b0 = beta.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
        b_idx0 = b0.index
        b_idx1 = b_idx0
        if cutlass.const_expr(have_m1):
            b1 = beta.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
            b_idx1 = b1.index
        kk_b = b_idx1 if pair_half == 1 else b_idx0
        kk_beta = [sB[r, kk_b] for r in rows]

        # ---- KK epilogue: M[i, j] = (K K^T)[i, j] * T[i, j] * beta[i] -----------------
        # Half 0 starts on member 0's accumulator while member 1's GEMM still runs; half 1
        # waits for member 1's (the MMA warp issues them in that order).
        tinv_base = params.sTinv.iterator.toint()
        c0 = cg0.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        tmem.access("cg0", read=True)
        t0 = tinv.acquire(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        t0.access(write=True, by="threads")
        t0.access(read=True, by="threads")
        t_addr0 = tinv_base + t0.index * TILE_BYTES
        if pair_half == 0:
            self.kk_epilogue(c0.index, t_addr0, kk_beta, decay_kk, me, tmem)
        c_idx1 = c0.index
        t_addr1 = t_addr0
        if cutlass.const_expr(have_m1):
            c1 = cg0.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            t1 = tinv.acquire(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            t1.access(write=True, by="threads")
            t1.access(read=True, by="threads")
            c_idx1 = c1.index
            t_addr1 = tinv_base + t1.index * TILE_BYTES
            if pair_half == 1:
                self.kk_epilogue(c1.index, t_addr1, kk_beta, decay_kk, me, tmem)
        a_acc = c0.index if pair_half == 1 else c_idx1

        # ---- pair inverse: warps 0-1 own member 0, warps 2-3 member 1 ------------------
        inv_tile = t_addr1 if (have_m1 and warp >= 2) else t_addr0
        do_inv = have_m1 or warp < 2
        ti = self._flat(inv_tile)
        barrier = self.inverse
        barrier.sync(aligned=True, skip_fence="after", reason=AFTER_FENCE_COVERED)
        if do_inv:
            invert_diagonal_8x8(ti, (inverse_local_warp * 32 + lane) // 8, tid, B_T, dtype)
        barrier.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
        blockwise_8x8_to_16x16(self._flat(t_addr0), warp * 16, lane, B_T, dtype)
        if cutlass.const_expr(have_m1):
            blockwise_8x8_to_16x16(self._flat(t_addr1), warp * 16, lane, B_T, dtype)
        barrier.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
        if do_inv:
            blockwise_16x16_to_32x32(ti, inverse_local_warp * 32, lane, B_T, dtype)
        barrier.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
        blockwise_32x32_to_64x64(
            ti,
            inverse_local_warp,
            lane,
            barrier,
            B_T,
            dtype,
            skip_fence="both",
            reason=BOTH_FENCES_COVERED,
        )
        barrier.sync(aligned=True, skip_fence="before", reason=BEFORE_FENCE_COVERED)

        # ---- beta column scaling: Tinv[i, j] *= beta[j], member by member -------------
        self.scale_columns(sB, b_idx0, t_addr0, store_row, store_col, lane)
        t0.commit()  # fence.proxy.async, then the U GEMM may read it
        b0.release()
        if cutlass.const_expr(have_m1):
            self.scale_columns(sB, b_idx1, t_addr1, store_row, store_col, lane)
            t1.commit()
            b1.release()

        # ---- A epilogue (the other member): A = (Q K^T) * T * scale -------------------
        decay_a = decay_tile(sG, a_g, rows, lane, mask_zero)
        g0.release()
        if cutlass.const_expr(have_m1):
            g1.release()
        a_lo = tmem_ld("16x256b", 8, cg0_col + a_acc * B_T + lane_lo)
        a_hi = tmem_ld("16x256b", 8, cg0_col + a_acc * B_T + lane_hi)
        c0.release()  # tcgen05.wait::ld first
        if cutlass.const_expr(have_m1):
            c1.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        a0 = a.acquire(skip_fence="both", reason=BOTH_FENCES_COVERED)
        a0.access(write=True, by="threads")
        a_base = params.sA.iterator.toint()
        a_addr0 = a_base + a0.index * TILE_BYTES
        a_addr1 = a_addr0
        if cutlass.const_expr(have_m1):
            a1 = a.acquire(skip_fence="both", reason=BOTH_FENCES_COVERED)
            a1.access(write=True, by="threads")
            a_addr1 = a_base + a1.index * TILE_BYTES
        my_a = a_addr0 if pair_half == 1 else a_addr1
        if do_a:
            factors = [params.scale for _ in range(16)]
            for u in cutlass.range_constexpr(2):
                store_scaled(
                    my_a,
                    half_row_base + u * 16 + store_row_frag,
                    store_col,
                    a_hi if u == 1 else a_lo,
                    decay_a[u * 32 : u * 32 + 32],
                    factors,
                    dtype,
                )
        fence_async_shared()  # one fence covers both stages' stmatrix (commit_fence=())
        a0.commit()
        if cutlass.const_expr(have_m1):
            a1.commit()
        return gate, beta, cg0, tinv, a

    @cute.jit
    def kk_epilogue(self, acc_idx, tile_addr, kk_beta, decay_kk, me, tmem):
        """This warp's 32 KK rows of one member: TMEM -> scaled 16-bit -> swizzled tile."""
        lane = me.lane
        half_row_base = (me.warp % 2) * 32
        lane_lo = (32 * (me.tmem_row // 32)) << 16
        lane_hi = (32 * (me.tmem_row // 32) + 16) << 16
        col = tmem.ptr("cg0").toint() + acc_idx * B_T
        kk_lo = tmem_ld("16x256b", 8, col + lane_lo)
        kk_hi = tmem_ld("16x256b", 8, col + lane_hi)
        cute.arch.fence_view_async_tmem_load()
        for u in cutlass.range_constexpr(2):
            factors = [kk_beta[2 * u + (k % 2)] for k in range(16)]
            store_scaled(
                tile_addr,
                half_row_base + u * 16 + lane % 16,
                (lane // 16) * 8,
                kk_hi if u == 1 else kk_lo,
                decay_kk[u * 32 : u * 32 + 32],
                factors,
                self.io_dtype,
            )

    @cute.jit
    def scale_columns(self, sB, b_idx, tile_addr, store_row, store_col, lane):
        """Scale column ``j`` of one inverted tile by ``beta[j]`` (ldmatrix -> stmatrix)."""
        dtype = self.io_dtype
        beta_col = [sB[(lane % 4) * 2 + ((k // 4) * 8 + k % 2), b_idx] for k in range(32)]
        frags = []
        for c in cutlass.range_constexpr(B_T // 16):
            off = store_row * B_T + swizzle_xor_128b(store_row, store_col + c * 16)
            frags += ldmatrix(tile_addr + off * 2, 4)
        pack = []
        for j in cutlass.range_constexpr(16):
            lo, hi = f16x2_to_f32(frags[j], dtype=dtype)
            s0, s1 = fmul2(lo, hi, beta_col[2 * j], beta_col[2 * j + 1])
            pack.append(fp32_to_fp16(s0, s1, dtype=dtype))
        for c in cutlass.range_constexpr(B_T // 16):
            off = store_row * B_T + swizzle_xor_128b(store_row, store_col + c * 16)
            stmatrix(tile_addr + off * 2, pack[c * 4 : c * 4 + 4])


class GdnPrefillGmem(GdnPrefill):
    """Chain prefill: the chunk-inverse tiles come from ``gdn_tinv`` through TMA."""

    tinv_source = "gmem"

    @role(warps=9, regs=24)
    @cute.jit
    def load(
        self,
        params: PrefillParams,
        *,
        kq: Producer,
        v: Producer,
        tinv: Producer,
        ticket: Producer,
        work: WorkCursor,
    ):
        while work.valid:
            unit, _, _ = work.item
            kq, v, tinv, ticket = self.load_unit(params, unit, kq, v, tinv, ticket)
            work.advance()
        kq.tail()
        v.tail()
        tinv.tail()
        cute.arch.griddepcontrol_launch_dependents()
        return kq, v, tinv, ticket, work

    @role(warps=8, regs=24)
    @cute.jit
    def gate_load(
        self,
        params: PrefillParams,
        *,
        gate: Producer,
        ticket: Consumer,
        work: WorkCursor,
        me: Local,
    ):
        while work.valid:
            unit, _, _ = work.item
            gate, _beta, ticket = self.gate_unit(params, unit, gate, None, ticket, me.lane)
            work.advance()
        return gate, ticket, work

    @role(warps=range(4), regs=224)
    @cute.jit
    def compute0(
        self,
        params: PrefillParams,
        *,
        gate: Consumer,
        cg0: Consumer,
        a: Producer,
        ticket: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = unit
            while tile < total:
                _b, _h, _ws, write_end, compute_start, _bs, _be = self.item(params, tile)
                n_local = write_end - compute_start
                n_pairs = n_local // 2
                for _p in cutlass.range(n_pairs, unroll=1):
                    gate, cg0, a = self.a_pair(params, True, gate, cg0, a, tmem, me)
                if n_local - n_pairs * 2 == 1:
                    gate, cg0, a = self.a_pair(params, False, gate, cg0, a, tmem, me)
                tile, ticket = self.next_tile(ticket, skip_fence="before")
            work.advance()
        return gate, cg0, a, ticket, work

    @cute.jit
    def a_pair(
        self, params: PrefillParams, have_m1: cutlass.Constexpr[bool], gate, cg0, a, tmem, me
    ):
        """A epilogues of a chunk pair (or of a lone last chunk): every warp stages 16 rows of
        each member's Q K^T (the M = 64 accumulator fills lanes 0-15 of each subpartition)."""
        dtype = self.io_dtype
        warp, lane = me.warp, me.lane
        store_row = warp * 16 + lane % 16
        store_col = (lane // 16) * 8
        row_lo = warp * 16 + lane // 4
        mask_zero = opaque_f32_zero()
        lane_lo = (32 * (me.tmem_row // 32)) << 16
        cg0_col = tmem.ptr("cg0").toint()
        g0 = gate.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
        g_idx = [g0.index]
        if cutlass.const_expr(have_m1):
            g1 = gate.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
            g_idx.append(g1.index)
        factors = [params.scale for _ in range(16)]
        for member in cutlass.range_constexpr(2 if have_m1 else 1):
            decay = decay_tile(params.sGate, g_idx[member], (row_lo, row_lo + 8), lane, mask_zero)
            ch = cg0.wait()
            vals = tmem_ld("16x256b", 8, cg0_col + ch.index * B_T + lane_lo)
            ch.release()  # tcgen05.wait::ld first
            ah = a.acquire(skip_fence="both", reason=BOTH_FENCES_COVERED)
            ah.access(write=True, by="threads")
            store_scaled(
                params.sA.iterator.toint() + ah.index * TILE_BYTES,
                store_row,
                store_col,
                vals,
                decay,
                factors,
                dtype,
            )
            fence_async_shared()
            ah.commit()
        g0.release()
        if cutlass.const_expr(have_m1):
            g1.release()
        return gate, cg0, a


def bundle_hook(
    io_dtype,
    state_dtype,
    beta_dtype,
    *,
    num_sm: int,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    log_gate: bool,
    tinv_source: str,
    d_k: int,
    d_v: int,
    tiles_per_head: int = 1,
    paged_state: bool = False,
    has_final_indices: bool = False,
) -> tuple[GdnPrefill, int]:
    """``(op, num_sm)`` constexpr for the launch bundles: ``op.launch(..., Int32(num_sm),
    stream)`` inside their ``@cute.jit`` host. Rejects what this kernel does not implement:
    checkpoints, linear gates, final-state index remapping and, with ``tinv_source="compute"``,
    16-bit beta."""
    if enable_checkpoints or not log_gate or has_final_indices:
        raise ValueError(
            "the GDN prefill takes natural-log gates, no checkpoints or final_indices"
        )
    if tinv_source not in TINV_SOURCES:
        raise ValueError(f"tinv_source must be one of {TINV_SOURCES}, got {tinv_source!r}")
    if tinv_source == "compute" and beta_dtype is not cutlass.Float32:
        raise TypeError(f"the GDN prefill takes fp32 beta, got {beta_dtype}")
    cls = GdnPrefillCompute if tinv_source == "compute" else GdnPrefillGmem
    op = cls(
        _NAMES[io_dtype],
        _NAMES[state_dtype],
        d_k,
        d_v,
        tiles_per_head,
        use_initial_state,
        store_final_state,
        paged_state,
    )
    return op, num_sm


WORK_ITEM_FIELDS = 10


@jit_cache
def compile_gdn_prefill(op: GdnPrefill):
    """Compile ``op`` for dynamic tokens, heads, sequences, tinv rows and table rows."""
    io = op.io_dtype
    state = _CUTE_DTYPES[op.state_dtype_name]
    tokens, q_heads, k_heads, v_heads, heads, seqs, rows = (cute.sym_int() for _ in range(7))
    d_v = op.d_v * op.tiles_per_head
    compute = op.tinv_source == "compute"

    def qkv(h, d):  # head slices of fused projections: dynamic token and head strides
        return make_fake_strided_tensor(io, (tokens, h, d), stride_divisibility=8)

    def state_t():
        return make_fake_strided_tensor(state, (seqs, heads, d_v, op.d_k), stride_divisibility=4)

    i32 = cutlass.Int32
    return compile_tvm_ffi(
        op,
        qkv(q_heads, op.d_k),
        qkv(k_heads, op.d_k),
        qkv(v_heads, d_v),
        make_fake_strided_tensor(Float32, (tokens, heads), contiguous_dim=None),
        make_fake_strided_tensor(Float32, (tokens, heads), contiguous_dim=None)
        if compute
        else None,
        qkv(heads, d_v),
        state_t() if op.use_initial_state else None,
        state_t() if op.store_final_state else None,
        None
        if compute
        else make_fake_strided_tensor(
            io, (cute.sym_int(), heads, B_T, B_T), stride_divisibility=8
        ),
        make_fake_strided_tensor(i32, (rows, WORK_ITEM_FIELDS), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (2,)),
        fake_tma_slots(),
        Float32(1.0),
        Int32(1),
        name=op.get_name(),
    )


def uncut_work_items(cu_seqlens: torch.Tensor, heads: int, tiles_per_head: int = 1):
    """``(work_items, work_count)`` of the uncut table: one item per non-empty sequence and
    tile head, longest first (the order ``work_tables`` writes)."""
    cu = cu_seqlens.tolist()
    items = []
    for b in range(len(cu) - 1):
        chunks = (cu[b + 1] - cu[b] + B_T - 1) // B_T
        if chunks:
            for h in range(heads * tiles_per_head):
                items.append([b, h, 0, chunks, 0, chunks, cu[b], cu[b + 1], b, -1])
    items.sort(key=lambda r: -r[3])
    device = cu_seqlens.device
    table = torch.tensor(items or [[0] * WORK_ITEM_FIELDS], dtype=torch.int32, device=device)
    return table, torch.tensor([len(items)], dtype=torch.int32, device=device)


def gdn_prefill(
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
    tinv_source: str = "compute",
    tiles_per_head: int = 1,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Uncut GDN prefill of packed ``q/k [T, HK, D]``, ``v [T, H, DV]``, natural-log ``gate``
    and ``beta [T, H]`` (fp32). ``tinv_source="gmem"`` runs ``gdn_tinv`` first and needs an
    initial state (the chain's contract). Builds the work table on the host: for tests and
    standalone use; the launch bundles (``bundles``) launch it after the work-table
    prologue in ``work_tables``."""
    from .gdn_tinv import gdn_tinv, tinv_rows

    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k and v must share a 16-bit dtype")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("gate and beta must be fp32")
    if tinv_source == "gmem" and (initial_state is None or tiles_per_head != 1):
        raise ValueError("the gmem prefill takes an initial state and no d_v split")
    tokens, _, d_k = q.shape
    heads, d_v = gate.shape[1], v.shape[2]
    num_seqs = cu_seqlens.shape[0] - 1
    o = torch.empty(tokens, heads, d_v, dtype=q.dtype, device=q.device)
    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    final = None
    if output_final_state:
        final = (
            initial_state.clone()
            if initial_state is not None
            else torch.zeros(num_seqs, heads, d_v, d_k, dtype=torch.float32, device=q.device)
        )
    tinv = None
    if tinv_source == "gmem":
        rows = tinv_rows(tokens, num_seqs)
        tinv = torch.empty(rows, heads, B_T, B_T, dtype=q.dtype, device=q.device)
        row_table = torch.empty(rows, 4, dtype=torch.int32, device=q.device)
        gdn_tinv(k, gate, beta, cu_seqlens, tinv, row_table, torch.empty_like(cu_seqlens[:1]))
    work_items, work_count = uncut_work_items(cu_seqlens, heads, tiles_per_head)
    scheduler = torch.zeros(2, dtype=torch.int32, device=q.device)
    names = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
    cls = GdnPrefillCompute if tinv_source == "compute" else GdnPrefillGmem
    op = cls(
        names[q.dtype],
        names[state_dtype],
        d_k,
        d_v // tiles_per_head,
        tiles_per_head,
        initial_state is not None,
        output_final_state,
        False,
    )
    fn = compile_gdn_prefill(op)
    _, sms = op.work.launch_args(q.device)
    fn(
        q,
        k,
        v,
        gate,
        beta if tinv_source == "compute" else None,
        o,
        initial_state,
        final,
        tinv,
        work_items,
        work_count,
        scheduler,
        new_tma_slots(op.tma_slot_rows(heads), q.device),
        scale,
        sms,
    )
    return o, final
