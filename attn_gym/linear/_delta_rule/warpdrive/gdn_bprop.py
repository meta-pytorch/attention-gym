# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/gdn_bprop_f16.py),
# rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and channels with CuTeDSL 4.6.2
# APIs: one TMA descriptor per tensor replaces the per-sequence descriptor arrays, and only the
# natural-log gate, fp32 beta form the backward drivers compile remains.

"""GDN backward (bprop) for SM100: per work item, the gradients of one value head over the chunks
``[write_start, compute_end)`` of one sequence (or chain piece), walked backward.

Per chunk (64 tokens, ``g`` the log2 gate cumsum, ``S`` the checkpoint entering the chunk, ``dS``
the state gradient carried backward in TMEM; see the upstream kernel for the derivation)::

    T    = (I + strict(K K^T 2^(g_i - g_j)) Beta)^-1      (beta-free, blockwise inverse in SMEM)
    Y    = V - 2^g (K S^T)          U = Y (T Beta)^T       A = (Q K^T) 2^(g_i - g_j) scale
    dU   = dO^T A + 2^(g_63 - g) (K dS)                     Z = T^T dU
    dV   = Beta Z                   dA = dO U^T            dM = Z U^T
    dQ   = (S^T dO 2^g scale) + K^T dA
    dK   = dK_inter + Q^T dA + K^T dM^T + K^T dM - 2^g (dV S^T)
    dS   = 2^(g_63) dS + (2^g scale dO)^T Q - (Beta 2^g Z)^T K

plus the dGate/dBeta reductions, all with upstream's instruction order (the outputs match the
upstream kernel bit for bit). With ``use_dstate_in`` the item's state gradient starts from
``dstate_in[batch, head]`` when ``compute_end`` is the sequence's last chunk (else zero, a split
item's warmup); with ``use_dstate0`` the final ``dS`` lands at ``dstate0[dstate_dst, head]``.

Work items are rows of the split-K work table (``work_tables``): ``(batch, head,
write_start, write_end, -, compute_end, batch_start, batch_end, -, dstate_dst)``; gradients of the
chunks ``[write_start, write_end)`` are stored. The checkpoint entering chunk ``c`` is
``checkpoints[cu[b] // 64 + b + c, head]`` (``gdn_recompute``'s closed-form rows).

Ragged TMA views (``ragged.RaggedTma``) replace upstream's per-sequence descriptor arrays:
loads zero-fill rows past a sequence end and stores drop them, as per-sequence descriptors do.

Work order is upstream's: CTA ``u`` of ``G`` (one OneShot item each) starts at row ``u``; the
load warp claims later rows from the zeroed scheduler counter (``G + atomicAdd``) and pins the
last partial wave (row ``tail_base + u``), publishing each next row through ``sched``.

Roles (16 warps): ``cg0`` (0-3) builds the A, dA and dM tiles, the inverse and dGate/dBeta
M-terms; ``cg1`` (4-7) stages the TMEM operands (Y, dO', dU, dY') and the dQ, dV, Z tiles;
``cg2`` (8-11) reads the dV/dK accumulator, prepares the next chunk's state gradient and stores
dK; ``mma`` (12) issues the 18 GEMMs of a chunk; ``load`` (13) the TMA loads; ``gate_load``
(14) the gate/beta staging and the dGate/dBeta stores; ``epi`` (15) the dQ/dK/dV TMA stores.

Channels. SMEM stages are managed rings: ``k`` (2) and ``q``, ``v``, ``do``, ``state`` (1) from
TMA to the MMA warp (``Umma`` release) and the compute group that also reads the tile
(``Threads``); ``gate``/``beta`` (2) from the gate warp to the compute groups, which write
their dGate/dBeta partials back into the stage before releasing it; ``a`` (two epochs per chunk:
A, then dA in the same buffer), ``tinv``, ``dm``, ``u`` (U written over V), ``z`` (Z staged in
the dQ tile) from the compute groups to the MMA warp; ``dq``, ``dk``, ``dv`` from the compute
groups to the store warp (``dv`` also to the MMA warp, which reads dV for dK's state path).

The TMEM accumulator chain is upstream's raw one-way signals (``sig``, one mbarrier each with
upstream's arrival counts): six column regions carry 2-5 productions per chunk between the MMA
warp and the three compute groups, and each signal's reuse is ordered by a later signal of the
chain (noted at the declaration). A managed round trip there would add a wait to the chunk's
critical path (see ``gdn_recompute``).
"""

import functools
from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync

from attn_gym.warpdrive import (
    AFTER_FENCE_COVERED,
    BEFORE_FENCE_COVERED,
    BOTH_FENCES_COVERED,
    Barriers,
    Consumer,
    Local,
    NamedBarrier,
    OneShot,
    Producer,
    Ring,
    RoleConfig,
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
from attn_gym.warpdrive.compat import LayoutEnum
from attn_gym.warpdrive.protocol import current_trace
from attn_gym.warpdrive.thread_sync import tcgen05_after_thread_sync, tcgen05_before_thread_sync
from attn_gym.warpdrive.tracing import wait_range

from .blockwise_inverse import (
    blockwise_8x8_to_16x16,
    blockwise_16x16_to_32x32,
    blockwise_32x32_to_64x64,
    invert_diagonal_8x8,
)
from .contract import tma_divisibility
from .ptx import (
    cp_async_ca_4,
    elect_sync,
    f16x2_to_f32,
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    opaque_f32_zero,
    shfl_bfly,
    shfl_down,
    shfl_idx,
    shfl_up,
    st_shared_v4,
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
from .umma import SWIZZLE_128B, MmaDesc, instr_desc, mma_ss, mma_ts, smem_desc

B_T = 64
DIMS = (64, 128)
RCP_LN2 = 1.4426950408889634  # 1 / ln 2: natural-log gates -> the kernel's log2 domain
BOX = 64  # 16-bit elements per 128-byte swizzle row
BOX_BYTES = B_T * BOX * 2  # one 64 x 64 16-bit box
SQ = B_T * B_T  # elements of a 64 x 64 tile
K_STAGES = 2
SCALAR_STAGES = 2  # gate and beta rings
SCHED_STAGES = 2
# FP16 Z staging floor (a power of two, so Beta = 0 stays exact); see cg1's Z publish.
Z_FP16_BETA_FLOOR = 2.0**-10
TAIL_SENTINEL = 1 << 28  # a row past any table: ends the work loop
WORK_ITEM_FIELDS = 10
WORK_ITEM_DSTATE_DST = 9

_CUTE_DTYPES = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16}
_IO_DTYPES = {torch.float16: "fp16", torch.bfloat16: "bf16"}
_SCALAR_DTYPES = {
    torch.float32: "Float32",
    torch.bfloat16: "BFloat16",
    torch.float16: "Float16",
}
_STATE_DTYPES = {torch.float32: "Float32", torch.bfloat16: "BFloat16"}

# The raw TMEM-chain signals: (name, arriving role). Arrival counts follow the role: one
# tcgen05.commit from the MMA warp, or every thread of a compute group. The comment names what
# the signal hands over and the later signal that orders its reuse.
SIGNALS = (
    ("kk_full", "mma"),  # KK in acc_a -> cg0; the next acc_a write waits kk_free (KS) or t-inv
    ("kk_free", "cg0"),  # cg0 read KK -> KS may overwrite acc_a (ST chunks only)
    ("a_acc_full", "mma"),  # QK in acc_b -> cg0; U overwrites acc_b after tinv (cg0 read it)
    ("ks_full", "mma"),  # KS in acc_a -> cg1; Z overwrites after du_in_full (cg1 read it)
    ("u_acc_full", "mma"),  # U in acc_b -> cg1; dA overwrites after u (cg1 read it)
    ("z_acc_full", "mma"),  # Z in acc_a -> cg1; dM overwrites after z (cg1 read it)
    ("da_acc_full", "mma"),  # dA in acc_b -> cg0; the next QK waits a (cg0 read it)
    ("dm_acc_full", "mma"),  # dM in acc_a -> cg0; the next KK waits dm (cg0 read it)
    ("spath_full", "mma"),  # dK state path in the inputs region -> cg2 (next dO' waits cg2_chunk)
    ("du_scale_full", "mma"),  # dV inter in dvdk -> cg1 (rescale in place)
    ("du_scale_free", "cg1"),  # rescaled -> dU intra accumulates
    ("du_total_full", "mma"),  # dU in dvdk -> cg1; dK inter overwrites after u (cg1 read it)
    ("dk_scale_full", "mma"),  # dK inter in dvdk -> cg2 (rescale in place)
    ("dk_scale_free", "cg2"),  # rescaled -> dK attn accumulates
    ("dk_attn_full", "mma"),  # dK attn in dvdk -> cg2
    ("dk_attn_free", "cg2"),  # cg2 read it -> dM-terms accumulate
    ("dk_total_full", "mma"),  # dK in dvdk -> cg2
    ("dk_total_free", "cg2"),  # cg2 read it -> the next dV inter (or dU intra) overwrites
    ("dq_scale_full", "mma"),  # dQ inter (dstate_in columns) -> cg1 (rescale in place)
    ("dq_scale_free", "cg1"),  # rescaled -> dQ attn accumulates
    ("dq_total_full", "mma"),  # dQ -> cg1
    ("dq_total_free", "cg1"),  # cg1 read dQ -> next dQ inter (mma); dstate_in rewrite (cg2)
    ("dstate_full", "mma"),  # dS after the K term -> cg2
    ("dstate_scaled", "cg1"),  # dS rescaled by cg1 (or released by cg2 at an item end)
    ("dstate_in_full", "cg2"),  # fp16 dS (dstate_in columns) -> dV inter
    ("dstate_in_free", "mma"),  # dV inter read it -> cg2 may rewrite it
    ("do_prime_full", "cg1"),  # dO' (inputs region) -> Q term
    ("du_in_full", "cg1"),  # fp16 dU (inputs + 32) -> Z GEMM
    ("dyp_full", "cg1"),  # dY' over dU -> K term
    ("y_full", "cg1"),  # Y and 2^g KS (y, gks regions) -> U GEMM
    ("dstate_smem_full", "cg2"),  # sDstate written and dS read -> cg0 dot, cg1 rescale, dK inter
    ("state_dot_done", "cg0"),  # cg0 read sDstate -> cg2 may rewrite it
    ("dbeta_cg1", "cg1"),  # cg1's dBeta V-term in the beta stage -> cg0
    ("dgate_cg1", "cg1"),  # cg1's dGate part folded -> cg2
    ("dgate_cg0", "cg0"),  # cg0's dGate part folded -> cg1
    ("cg2_chunk", "cg2"),  # cg2 latched the chunk's decay (and read the last state path)
)
SIG = {name: i for i, (name, _) in enumerate(SIGNALS)}


@dataclass(frozen=True)
class BpropParams:
    q_load: RaggedTma  # (token, d, q head) loads, zero past each sequence end
    k_load: RaggedTma
    v_load: RaggedTma
    do_load: RaggedTma
    tma_ckpt: cute.CopyAtom
    mCkpt: cute.Tensor  # TMA view (v, k, head, row)
    dq_store: RaggedTma  # (token, d, head) stores, dropping rows past each sequence end
    dk_store: RaggedTma
    dv_store: RaggedTma
    gate: cute.Tensor  # (token, head) natural-log gate
    beta: cute.Tensor  # (token, head) fp32
    dgate: cute.Tensor
    dbeta: cute.Tensor
    dstate_in: cute.Tensor | None  # (batch, head, v, k)
    dstate0: cute.Tensor | None  # (sequence, head, v, k)
    items: cute.Tensor  # (row, fields) work table
    count: cute.Tensor  # (1,) work rows
    scheduler: cute.Tensor  # (>= 1,) zeroed claim counter
    box_layout: cute.ComposedLayout  # one 64 x 64 swizzled box
    ckpt_box_layout: cute.ComposedLayout  # one d_v x 64 swizzled checkpoint box
    q_group: Int32  # output heads per q / k / v head
    k_group: Int32
    v_group: Int32
    scale: Float32
    units: Int32  # launched CTAs


class Item:
    """One work-table row's fields (chunk units except the token bounds)."""

    def __init__(self, params: BpropParams, row):
        t = params.items
        self.batch, self.head = t[row, 0], t[row, 1]
        self.write_start, self.write_end = t[row, 2], t[row, 3]
        self.compute_end = t[row, 5]
        self.batch_start, self.batch_end = t[row, 6], t[row, 7]

    @property
    def n_chunks(self):
        return self.compute_end - self.write_start


def _wait(bars, phases, name: str, fence: bool = True):
    """Wait for raw signal ``name`` at this role's parity; returns the flipped parity mask. A
    ``<name>.wait`` range when the kernel traces waits. ``fence=False`` omits the after fence
    where a later one precedes the role's next tcgen05 op (``AFTER_FENCE_COVERED``)."""
    i = SIG[name]
    with wait_range(current_trace(), f"{name}.wait"):
        cute.arch.mbarrier_wait(bars + i, Int32((phases >> i) & 1))
    if fence:
        tcgen05_after_thread_sync()  # every signal hands TMEM between threads
    return phases ^ Int64(1 << i)


@cute.jit
def before_sync_and_arrive(bars, name: cutlass.Constexpr[str]):
    """``tcgen05.fence::before_thread_sync``, then arrive on raw signal ``name``."""
    tcgen05_before_thread_sync()
    arrive(bars, name)


@cute.jit
def arrive(bars, name: cutlass.Constexpr[str]):
    """Arrive on raw signal ``name`` without a fence: only where an earlier before fence
    already follows the role's last tcgen05 op (``BEFORE_FENCE_COVERED``)."""
    cute.arch.mbarrier_arrive(bars + SIG[name])


@cute.jit
def _commit(bars, name: cutlass.Constexpr[str]):
    """``tcgen05.commit`` to raw signal ``name``; the caller elects the lane."""
    cute.nvgpu.tcgen05.commit(bars + SIG[name])


def _phases(*names) -> int:
    """Parity mask whose ``names`` bits start at 1: waits that pass on a fresh barrier."""
    return sum(1 << SIG[name] for name in set(names))


@cute.jit
def warp_reduce_scatter_sum_16(vals, lane):
    """Reduce-scatter 16 fragment token-partials (tcol = (lane%4)*2 + (j//2)*8 + (j%2)); lane L
    returns the sums of tokens (L//4)*8 + (L%4)*2 and +1."""
    level = vals
    for k in cutlass.range_constexpr(3):
        off = cutlass.const_expr(4 << k)
        hi_lane = (lane // off) % 2 == 1
        nxt = []
        for i in cutlass.range_constexpr(len(level)):
            if cutlass.const_expr((i & 2) == 0):
                lo = level[i]
                hi = level[i + 2]
                send = lo if hi_lane else hi
                recv = shfl_bfly(send, off)
                keep = hi if hi_lane else lo
                nxt.append(keep + recv)
        level = nxt
    return level[0], level[1]


@cute.jit
def _token_masks(lane, n_valid):
    """AND masks of the packed 16-bit fragment words of a 64-token row: word ``j`` holds tokens
    ``2 * (lane % 4) + 8 * (j // 2)`` and the next; ``masks[j // 2]`` keeps those below
    ``n_valid`` and zeroes (+0) the rest."""
    masks = []
    for p in cutlass.range_constexpr(8):
        tok = 2 * (lane % 4) + 8 * p
        lo = Int32(0xFFFF) if tok < n_valid else Int32(0)
        hi = Int32(-65536) if tok + 1 < n_valid else Int32(0)  # 0xFFFF0000
        masks.append(lo | hi)
    return masks


@cute.jit
def _mask_words(words, masks):
    """Zero the 16-bit halves of fragment ``words`` (16 per row) that ``masks`` clears."""
    return [words[j] & masks[j // 2] for j in range(len(words))]


class GdnBprop(WarpSpecialized):
    """Warpdrive GDN bprop for fp16/bf16 operands with ``d_k == d_v`` in {64, 128}."""

    def __init__(
        self,
        dtype_name: str,
        d: int,
        use_initial_state: bool,
        use_dstate_in: bool,
        use_dstate0: bool,
        *,
        column_fold_delay: int = 0,
    ):
        """``column_fold_delay`` is a test hook: the ``clock64`` cycles cg0's warp 1 idles
        before it reads the dGate column partials, so a test can order those reads after warp
        0's next sKK write (see cg0's column fold). Production builds keep 0."""
        if d not in DIMS:
            raise ValueError(f"the bprop serves d_k = d_v in {DIMS}, got {d}")
        self.dtype_name, self.d = dtype_name, d
        self.use_initial_state = use_initial_state
        self.use_dstate_in = use_dstate_in
        self.use_dstate0 = use_dstate0
        self.column_fold_delay = column_fold_delay
        self.io_dtype = _CUTE_DTYPES[dtype_name]
        self.roles = {
            "cg0": RoleConfig(range(4), regs=208),
            "cg1": RoleConfig(range(4, 8), regs=144),
            "cg2": RoleConfig(range(8, 12), regs=128),
            "mma": RoleConfig(12, regs=32),
            "load": RoleConfig(13, regs=32),
            "gate_load": RoleConfig(14, regs=32),
            "epi": RoleConfig(15, regs=32),
        }
        io = self.io_dtype
        qk, st = B_T * d, d * d
        self.tile_bytes = qk * 2  # one [64, d] tile (Q, K, V, dO, dQ, dK, dV)
        self.smem_plan = plan = SmemPlan()
        # Scalars first: they share the region below the 1 KiB-aligned tiles with the barriers
        # (upstream's SMEM is at the 227 KiB cap).
        cumsum = plan.buffer("Cumsum", Float32, align=128)
        cumprod = plan.buffer("Cumprod", Float32, align=128)
        beta = plan.buffer("Beta", Float32, align=128)
        q_buf = plan.buffer("Q", io, align=1024)
        k_buf = plan.buffer("K", io, align=1024)
        do_buf = plan.buffer("dO", io, align=1024)
        state_buf = plan.buffer("State", io, align=1024)
        tinv_buf = plan.buffer("Tinv", io, align=1024)
        plan.buffer("KK", io, align=1024)  # L (beta-free, strict) then cg0's dGate scratch
        a_buf = plan.buffer("A", io, align=1024)  # A, then dA
        plan.buffer("Dm", io, align=1024)  # M_kk -> T (inverse in place), then dM
        v_buf = plan.buffer("V", io, align=1024)  # V, then U
        plan.buffer("Dstate", io, align=1024)  # fp16 dS^T (dK inter's A operand)
        dq_buf = plan.buffer("dQ", io, align=1024)  # Z (dM core operand), then dQ
        dk_buf = plan.buffer("dK", io, align=1024)  # dGate/dBeta scratch, then dK
        dv_buf = plan.buffer("dV", io, align=1024)
        self._buffer_shapes = {
            "Cumsum": (B_T, SCALAR_STAGES),
            "Cumprod": (B_T, SCALAR_STAGES),
            "Beta": (B_T, SCALAR_STAGES),
            "Q": (qk, 1),
            "K": (qk, K_STAGES),
            "dO": (qk, 1),
            "State": (st, 1),
            "Tinv": (SQ, 1),
            "KK": (SQ, 1),
            "A": (SQ, 1),
            "Dm": (SQ, 1),
            "V": (qk, 1),
            "Dstate": (st, 1),
            "dQ": (qk, 1),
            "dK": (qk, 1),
            "dV": (qk, 1),
        }
        tile = Tma(self.tile_bytes)
        self.sched = Ring(SCHED_STAGES, Warps, Warps, payload=(Int32, (1,)))
        self.k = Ring(K_STAGES, tile, {"mma": Umma, "cg2": Threads}, buffer=k_buf)
        # cg1 releases q, v and do right after a commit whose smem_async fence already ordered
        # its reads (see cg1_item).
        cg1_reads = {"mma": Umma, "cg1": Threads}
        self.q = Ring(1, tile, cg1_reads, buffer=q_buf, release_fence=())
        self.v = Ring(1, tile, cg1_reads, buffer=v_buf, release_fence=())
        self.do = Ring(1, tile, cg1_reads, buffer=do_buf, release_fence=())
        self.state = Ring(1, Tma(st * 2), {"mma": Umma, "cg0": Threads}, buffer=state_buf)
        scalars = {"cumsum": cumsum, "cumprod": cumprod}
        self.gate = Ring(SCALAR_STAGES, Threads, Threads, buffer=scalars)
        self.beta = Ring(SCALAR_STAGES, Threads, Threads, buffer=beta)
        self.a = Ring(1, Threads, Umma, buffer=a_buf)
        self.tinv = Ring(1, Threads, Umma, buffer=tinv_buf)
        self.dm = Ring(1, Threads, Umma, commit_fence="smem_async")
        self.u = Ring(1, Threads, Umma, commit_fence="smem_async")
        self.z = Ring(1, Threads, Umma, commit_fence="smem_async")
        self.dq = Ring(1, Threads, Threads, buffer=dq_buf, commit_fence="smem_async")
        self.dk = Ring(1, Threads, Threads, buffer=dk_buf, commit_fence="smem_async")
        self.dv = Ring(
            1,
            Threads,
            {"epi": Threads, "mma": Umma},
            buffer=dv_buf,
            commit_fence="smem_async",
        )
        arrivals = {"mma": 1, "cg0": 128, "cg1": 128, "cg2": 128}
        self.sig = Barriers(len(SIGNALS), counts=[arrivals[who] for _, who in SIGNALS])
        self.cg0_sync = NamedBarrier(threads=128)
        self.inv_sync = NamedBarrier(threads=64)  # the inverse's 32 -> 64 step (2 warps)
        self.cg1_sync = NamedBarrier(threads=128)
        self.cg2_sync = NamedBarrier(threads=128)
        self.tmem_plan = (
            TmemPlan()
            .region("dstate", d)  # dS^T accumulator [v, k]
            .region("dvdk", B_T)  # dV inter -> dU -> dK inter -> dK attn -> dK
            .region("dstate_in", max(d // 2, B_T))  # fp16 dS^T, then the dQ accumulator
            .region("acc_a", B_T)  # KK -> KS -> Z -> dM core
            .region("acc_b", B_T)  # QK -> U -> dA
            .region("inputs", B_T // 2 * 2)  # dO' @0, dU then dY' @32; then dK's state path
            .region("y", B_T // 2)  # Y, then Q^T upstream (never read: dropped)
            .region("gks", B_T // 2)  # fp16 2^g KS for the dGate state term
        )
        self.work = WorkStream(schedule=OneShot(grid="1d"))

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: Q, K, V, dO, dQ, dK and dV, keyed by head."""
        return tma_slot_rows(7, B_T, heads)

    def get_name(self) -> str:
        flags = "".join(
            str(int(f)) for f in (self.use_initial_state, self.use_dstate_in, self.use_dstate0)
        )
        return f"wd_gdn_bprop_{self.dtype_name}_d{self.d}_f{flags}"

    # ---- shared helpers ----------------------------------------------------------------

    @property
    def first_state_chunk(self) -> int:
        """The first chunk with an entering state: 0 with an initial state, else 1."""
        return 0 if self.use_initial_state else 1

    @cute.jit
    def _ptr(self, addr, dtype=None, align: cutlass.Constexpr[int] = 1024):
        """SMEM pointer at ``align``-byte aligned 32-bit shared address ``addr``."""
        dtype = self.io_dtype if cutlass.const_expr(dtype is None) else dtype
        return cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=align)

    @staticmethod
    @cute.jit
    def _addr(buf):
        return buf.iterator.toint()

    @cute.jit
    def _next_row(self, sched, skip_fence=None):
        """The next work-table row; ``skip_fence="before"`` drops the release's before fence
        (``BEFORE_FENCE_COVERED``)."""
        with sched.consume(skip_fence=skip_fence, reason=BEFORE_FENCE_COVERED) as h:
            row = h.data[0]
        return row, sched

    @cute.jit
    def _tmem(self, tmem, name: cutlass.Constexpr[str]):
        return tmem.ptr(name).toint()

    @staticmethod
    @cute.jit
    def _tmem_lanes():
        """First TMEM lane of this warp's 32 (``32 * (hardware warp % 4)``, the lanes
        ``tcgen05.ld``/``st`` reach), warp-uniform: TMEM addresses built from it stay in uniform
        registers, where ``Local.tmem_row`` (a thread index) needs an ``R2UR`` per access."""
        return 32 * (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4)

    # ---- load (warp 13): Q, K, V, dO, checkpoints and the scheduler claims ---------------

    @role(warps=13, regs=32)
    @cute.jit
    def load(
        self,
        params: BpropParams,
        *,
        k: Producer,
        q: Producer,
        v: Producer,
        do: Producer,
        state: Producer,
        sched: Producer,
        work: WorkCursor,
        me: Local,
    ):
        while work.valid:
            _, _, unit = work.item
            total = params.count[0]
            units = params.units
            # Upstream's tail pinning: the last partial wave, when at least half full, runs
            # row tail_base + u on CTA u instead of the ticket order.
            tail_count = ((total - 1) % units) + 1
            tail_base = total
            if tail_count * 2 >= units:
                tail_base = total - tail_count
            tail_row = tail_base + unit
            if tail_row >= total:
                tail_row = Int32(TAIL_SENTINEL)
            row = unit
            while row < total:
                it = Item(params, row)
                k, q, v, do, state = self.load_item(params, it, k, q, v, do, state)
                with sched.produce() as h:
                    claimed = Int32(0)
                    if me.lane == 0:
                        claimed = cute.arch.atomic_add(
                            params.scheduler.iterator, Int32(1), sem="relaxed", scope="gpu"
                        )
                    granted = shfl_idx(claimed, 0) + units
                    pinned = tail_row if row < tail_base else Int32(TAIL_SENTINEL)
                    row = granted if granted < tail_base else pinned
                    h.data[0] = row
            work.advance()
        cute.arch.griddepcontrol_launch_dependents()
        return k, q, v, do, state, sched, work

    @cute.jit
    def _load_tile(self, load: RaggedTma, box, it: Item, chunk, head, ring, h, desc):
        """TMA the ``[64, d]`` rows of chunk ``chunk`` into stage ``h.index`` of ``ring`` as
        ``d / 64`` swizzled boxes; rows past the sequence end zero-fill."""
        start = it.batch_start + chunk * B_T
        valid = cutlass.min(it.batch_end - start, B_T)
        base = self._addr(ring) + h.index * self.tile_bytes
        for kb in cutlass.range_constexpr(self.d // BOX):
            dst = cute.make_tensor(self._ptr(base + kb * BOX_BYTES), box)
            ragged_load(load, dst, start, valid, head, kb, h.barrier, B_T, BOX, desc)

    @cute.jit
    def load_item(
        self,
        params: BpropParams,
        it: Item,
        k,
        q,
        v,
        do,
        state,
    ):
        head_q = it.head // params.q_group
        head_k = it.head // params.k_group
        head_v = it.head // params.v_group
        box = params.box_layout
        loads = (params.k_load, params.q_load, params.v_load, params.do_load)
        descs = sequence_descriptors(loads, it.batch_end, it.head, B_T, it.n_chunks > 0)
        k_desc, q_desc, v_desc, do_desc = descs
        for rev in cutlass.range(it.n_chunks, unroll=1):
            chunk = it.compute_end - 1 - rev
            with k.produce() as h:
                self._load_tile(params.k_load, box, it, chunk, head_k, self.buffer("K"), h, k_desc)
            with q.produce() as h:
                self._load_tile(params.q_load, box, it, chunk, head_q, self.buffer("Q"), h, q_desc)
            with v.produce() as h:
                self._load_tile(params.v_load, box, it, chunk, head_v, self.buffer("V"), h, v_desc)
            with do.produce() as h:
                self._load_tile(
                    params.do_load, box, it, chunk, it.head, self.buffer("dO"), h, do_desc
                )
            if chunk >= self.first_state_chunk:
                with state.produce() as h:
                    row = it.batch_start // B_T + it.batch + chunk
                    base = self._addr(self.buffer("State"))
                    for kb in cutlass.range_constexpr(self.d // BOX):
                        dst = cute.make_tensor(
                            self._ptr(base + kb * self.d * BOX * 2), params.ckpt_box_layout
                        )
                        src = cute.local_tile(
                            params.mCkpt[None, None, it.head, row], (self.d, BOX), (0, kb)
                        )
                        tS, tG = cpasync.tma_partition(
                            params.tma_ckpt,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(dst, 0, 2),
                            cute.group_modes(src, 0, 2),
                        )
                        cute.copy(params.tma_ckpt, tG, tS, tma_bar_ptr=h.barrier)
        return k, q, v, do, state

    # ---- gate_load (warp 14): gate/beta staging and the dGate/dBeta stores ---------------

    @role(warps=14, regs=32)
    @cute.jit
    def gate_load(
        self,
        params: BpropParams,
        *,
        gate: Producer,
        beta: Producer,
        sched: Consumer,
        work: WorkCursor,
        me: Local,
    ):
        lane = me.lane
        stored = Int32(0)  # chunks whose dGate/dBeta this warp has stored (the store epoch)
        while work.valid:
            _, _, unit = work.item
            total = params.count[0]
            row = unit
            while row < total:
                gate, beta, stored = self.gate_item(
                    params, Item(params, row), gate, beta, stored, lane
                )
                row, sched = self._next_row(sched)
            work.advance()
        return gate, beta, sched, work

    @cute.jit
    def gate_item(self, params: BpropParams, it: Item, gate, beta, stored, lane):
        n = it.n_chunks
        write_end_token = cutlass.min(it.batch_start + it.write_end * B_T, it.batch_end)
        s_cumsum, s_cumprod = self.buffer("Cumsum"), self.buffer("Cumprod")
        s_beta = self.buffer("Beta")
        for rev in cutlass.range(n + 1, unroll=1):
            # ---- prefetch chunk rev: the log2 cumsum, its exp2 and beta --------------------
            if rev < n:
                start = it.batch_start + (it.compute_end - 1 - rev) * B_T
                valid = [start + lane + 32 * col < it.batch_end for col in range(2)]
                g = [Float32(0.0), Float32(0.0)]  # neutral past the end (log gate 0)
                for col in cutlass.range_constexpr(2):
                    if valid[col]:
                        last = it.batch_end - start - 1
                        g[col] = params.gate[
                            start + cutlass.min(lane + 32 * col, last), it.head
                        ].to(Float32)
                rcp_ln2 = opaque_f32_zero() + Float32(RCP_LN2)
                g[0], g[1] = fmul2(g[0], g[1], rcp_ln2, rcp_ln2)
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(2):
                        up = shfl_up(g[col], offset)
                        if lane >= offset:
                            g[col] = g[col] + up
                g[1] += shfl_idx(g[0], 31)
                with gate.produce() as h:
                    for col in cutlass.range_constexpr(2):
                        s_cumsum[lane + 32 * col, h.index] = g[col]
                        s_cumprod[lane + 32 * col, h.index] = cute.math.exp2(g[col], fastmath=True)
                # Beta through cp.async (zero past the end); each lane's copy completion is
                # its arrival on the stage (upstream's cp.async.mbarrier.arrive.noinc).
                bh = beta.acquire()
                stage = self._addr(s_beta) + bh.index * (B_T * 4)
                for col in cutlass.range_constexpr(2):
                    pos = lane + 32 * col
                    src = params.beta.iterator + params.beta.layout((start + pos, it.head))
                    cp_async_ca_4(stage + pos * 4, src.toint(), Int32(4) * Int32(valid[col]))
                cute.arch.cp_async_mbarrier_arrive_noinc(bh.full_barrier)
                bh.commit(signal=False)
            # ---- store chunk rev - 1's dGate and dBeta, once the compute groups released it --
            if rev > 0:
                start = it.batch_start + (it.compute_end - rev) * B_T
                stage = stored % SCALAR_STAGES
                parity = (stored // SCALAR_STAGES) % 2
                # Raw wait: the release of the stored epoch, at the parity the acquire that
                # refills its stage would wait for (that acquire then passes at once).
                cute.arch.mbarrier_wait(gate.empty_barrier(stage), parity)
                dg = [s_cumsum[lane + 32 * col, stage] for col in range(2)]
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(2):
                        down = shfl_down(dg[col], offset)
                        if lane < 32 - offset:
                            dg[col] = dg[col] + down
                dg[0] = dg[0] + shfl_idx(dg[1], 0, 0)
                for col in cutlass.range_constexpr(2):
                    pos = lane + 32 * col
                    if start + pos < write_end_token:
                        params.dgate[start + pos, it.head] = dg[col].to(params.dgate.element_type)
                cute.arch.mbarrier_wait(beta.empty_barrier(stage), parity)
                for col in cutlass.range_constexpr(2):
                    pos = lane + 32 * col
                    if start + pos < write_end_token:
                        params.dbeta[start + pos, it.head] = s_beta[pos, stage]
                stored = stored + 1
        return gate, beta, stored

    # ---- mma (warp 12): every tcgen05 GEMM of a chunk ------------------------------------

    def _descs(self):
        """Static geometry of each GEMM (upstream's MmaDesc table); ``d_k == d_v == d``."""
        io, d = self.io_dtype, self.d

        def desc(m, n, k, *, at=False, bt=False):
            idesc = instr_desc(Float32, io, io, m=m, n=n, a_mn_major=at, b_mn_major=bt)
            return MmaDesc(
                M=m, N=n, K=k, bpe_a=2, bpe_b=2, idesc=idesc, atranspose=at, btranspose=bt
            )

        return {
            "kk": desc(B_T, B_T, d),  # KK, QK
            "ks": desc(d, B_T, d),  # KS (state x K^T), dV inter (dS_in x K^T)
            "state_t": desc(d, B_T, d, at=True),  # dQ inter, dK inter, dK state path
            "du": desc(d, B_T, B_T, at=True, bt=True),  # dU intra
            "dstate": desc(d, d, B_T, bt=True),  # dS Q term, K term
            "u": desc(d, B_T, B_T),  # U
            "z": desc(d, B_T, B_T, bt=True),  # Z
            "da": desc(B_T, B_T, d),  # dA, dM core
            "k_t_t": desc(d, B_T, B_T, at=True, bt=True),  # dK attn, dK dM-term (dM^T)
            "k_t": desc(d, B_T, B_T, at=True),  # dQ attn, dK dM-term (dM)
        }

    @role(warps=12, regs=32)
    @cute.jit
    def mma(
        self,
        params: BpropParams,
        *,
        k: Consumer,
        q: Consumer,
        v: Consumer,
        do: Consumer,
        state: Consumer,
        a: Consumer,
        tinv: Consumer,
        u: Consumer,
        z: Consumer,
        dm: Consumer,
        dv: Consumer,
        sched: Consumer,
        sig: Barriers,
        work: WorkCursor,
        tmem: Tmem,
    ):
        starts = ["dk_total_free", "dq_total_free"]
        if cutlass.const_expr(not self.use_dstate_in):
            starts.append("dstate_scaled")  # the first Q term needs no rescale
        ph = Int64(_phases(*starts))
        while work.valid:
            _, _, unit = work.item
            total = params.count[0]
            row = unit
            while row < total:
                k, q, v, do, state, a, tinv, u, z, dm, dv, ph = self.mma_item(
                    Item(params, row), k, q, v, do, state, a, tinv, u, z, dm, dv, sig, ph, tmem
                )
                row, sched = self._next_row(sched)
            work.advance()
        return k, q, v, do, state, a, tinv, u, z, dm, dv, sched, work

    @cute.jit
    def _sdesc(self, buf: cutlass.Constexpr[str], lbo: cutlass.Constexpr[int], offset=0):
        """Shared-memory descriptor of buffer ``buf`` (128-byte swizzle, 8-row groups 1 KiB
        apart) at byte ``offset``, with leading byte offset ``lbo``."""
        # Not desc_opaque: upstream's descriptors stay foldable, so ptxas rematerializes them
        # from the SMEM base instead of holding ~20 of them in the MMA warp's 32 registers.
        addr = self._addr(self.buffer(buf)) + offset
        return smem_desc(addr, lbo, 8 * 128, SWIZZLE_128B)

    @cute.jit
    def mma_item(self, it: Item, k, q, v, do, state, a, tinv, u, z, dm, dv, sig, ph, tmem):
        d = self._descs()
        lbo_t = B_T * 128  # MN-major [*, 64] operands: 64-column boxes 64 rows apart
        lbo_s = self.d * 128  # MN-major state operands: boxes d rows apart
        lbo_sq = B_T // 2 * 128  # MN-major 64 x 64 tiles
        sdesc = self._sdesc
        dstate_col = self._tmem(tmem, "dstate")
        dvdk_col = self._tmem(tmem, "dvdk")
        dsin_col = self._tmem(tmem, "dstate_in")  # fp16 dS^T, then the dQ accumulator
        acc_a = self._tmem(tmem, "acc_a")
        acc_b = self._tmem(tmem, "acc_b")
        inputs = self._tmem(tmem, "inputs")
        do_prime_col = inputs
        du_col = inputs + B_T // 2
        y_col = self._tmem(tmem, "y")
        fsc = self.first_state_chunk
        for rev in cutlass.range(it.n_chunks, unroll=1):
            chunk = it.compute_end - 1 - rev
            have_dstate = cutlass.Boolean(True)
            if cutlass.const_expr(not self.use_dstate_in):
                have_dstate = rev > 0
            has_state = chunk >= fsc
            # ---- KK = K @ K^T -> acc_a --------------------------------------------------------
            kh = k.wait()
            k_off = kh.index * self.tile_bytes
            desc_k = sdesc("K", 16, k_off)
            desc_kt = sdesc("K", lbo_t, k_off)
            mma_ss(d["kk"], desc_k, desc_k, acc_a, False)
            if elect_sync():
                _commit(sig, "kk_full")
            # ---- QK = Q @ K^T -> acc_b --------------------------------------------------------
            qh = q.wait()
            mma_ss(d["kk"], sdesc("Q", 16), desc_k, acc_b, False)
            if elect_sync():
                _commit(sig, "a_acc_full")
            # ---- KS = state @ K^T -> acc_a (needs cg0's KK read) ------------------------------
            if has_state:
                # The state stage this chunk reads; the wait below owns it.
                state.peek(skip_fence="after", reason=AFTER_FENCE_COVERED)
                ph = _wait(sig, ph, "kk_free")
                mma_ss(d["ks"], sdesc("State", 16), desc_k, acc_a, False)
                if elect_sync():
                    _commit(sig, "ks_full")
            # ---- dV inter = dS_in(T) @ K^T -> dvdk ---------------------------------------------
            if have_dstate:
                ph = _wait(sig, ph, "dstate_in_full", fence=False)
            ph = _wait(sig, ph, "dk_total_free")
            if have_dstate:
                mma_ts(d["ks"], dsin_col, desc_k, dvdk_col, False)
                if elect_sync():
                    _commit(sig, "du_scale_full")
                    _commit(sig, "dstate_in_free")
            # ---- dQ inter = state^T @ dO -> dq accumulator --------------------------------------
            doh = do.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            if has_state:
                ph = _wait(sig, ph, "dq_total_free")
                mma_ss(d["state_t"], sdesc("State", lbo_s), sdesc("dO", 16), dsin_col, False)
                if elect_sync():
                    _commit(sig, "dq_scale_full")
            # ---- dU intra += dO^T @ A -> dvdk ---------------------------------------------------
            ah = a.wait()
            if have_dstate:
                ph = _wait(sig, ph, "du_scale_free")
            mma_ss(d["du"], sdesc("dO", lbo_t), sdesc("A", lbo_sq), dvdk_col, have_dstate)
            if elect_sync():
                _commit(sig, "du_total_full")
            ah.release()  # A read; cg0 writes dA over it after the dA GEMM (da_acc_full)
            # ---- dS Q term += dO'(T) @ Q -------------------------------------------------------
            ph = _wait(sig, ph, "do_prime_full", fence=False)
            ph = _wait(sig, ph, "dstate_scaled")
            mma_ts(d["dstate"], do_prime_col, sdesc("Q", lbo_t), dstate_col, have_dstate)
            # ---- U^T = Y(T) @ (T Beta)^T -> acc_b ------------------------------------------------
            th = tinv.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            ph = _wait(sig, ph, "y_full", fence=False)
            vh = v.wait()  # cg1 observed V before publishing Y: this passes at once
            mma_ts(d["u"], y_col, sdesc("Tinv", 16), acc_b, False)
            if elect_sync():
                _commit(sig, "u_acc_full")
            th.release()
            # ---- Z = dU(T) @ T (beta-free T in sDm, read transposed) -> acc_a --------------------
            ph = _wait(sig, ph, "du_in_full")
            mma_ts(d["z"], du_col, sdesc("Dm", lbo_sq), acc_a, False)
            if elect_sync():
                _commit(sig, "z_acc_full")
            # ---- dK inter = dS(sDstate)^T @ U^T -> dvdk -----------------------------------------
            uh = u.wait()
            if have_dstate:
                ph = _wait(sig, ph, "dstate_smem_full")
                mma_ss(d["state_t"], sdesc("Dstate", lbo_s), sdesc("V", 16), dvdk_col, False)
                if elect_sync():
                    _commit(sig, "dk_scale_full")
            # ---- dA = dO @ U^T -> acc_b ---------------------------------------------------------
            mma_ss(d["da"], sdesc("dO", 16), sdesc("V", 16), acc_b, False)
            if elect_sync():
                _commit(sig, "da_acc_full")
            doh.release()
            # ---- dM core = Z(sdQ) @ U^T -> acc_a ------------------------------------------------
            zh = z.wait()
            mma_ss(d["da"], sdesc("dQ", 16), sdesc("V", 16), acc_a, False)
            if elect_sync():
                _commit(sig, "dm_acc_full")
            vh.release()
            uh.release()
            zh.release()
            # ---- dS K term += dY'(T) @ K ---------------------------------------------------------
            ph = _wait(sig, ph, "dyp_full")
            mma_ts(d["dstate"], du_col, desc_kt, dstate_col, True)
            if elect_sync():
                _commit(sig, "dstate_full")
            # ---- dK attn += Q^T @ dA -> dvdk ------------------------------------------------------
            if have_dstate:
                ph = _wait(sig, ph, "dk_scale_free", fence=False)
            dah = a.wait()  # dA (cg0's da_ready)
            mma_ss(d["k_t_t"], sdesc("Q", lbo_t), sdesc("A", lbo_sq), dvdk_col, have_dstate)
            if elect_sync():
                _commit(sig, "dk_attn_full")
            qh.release()
            # ---- dQ attn += K^T @ dA -> dq accumulator -------------------------------------------
            if has_state:
                ph = _wait(sig, ph, "dq_scale_free")
            else:
                ph = _wait(sig, ph, "dq_total_free")
            mma_ss(d["k_t"], desc_kt, sdesc("A", 16), dsin_col, has_state)
            if elect_sync():
                _commit(sig, "dq_total_full")
            dah.release()
            # ---- dK state path = state^T @ dV^T -> the inputs region ------------------------------
            dvh = dv.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            if has_state:
                sh = state.wait()  # owns the stage peeked for KS
                mma_ss(d["state_t"], sdesc("State", lbo_s), sdesc("dV", 16), inputs, False)
                if elect_sync():
                    _commit(sig, "spath_full")
                sh.release()
            dvh.release()  # upstream's sdv_done
            # ---- dK dM-terms += K^T @ dM^T + K^T @ dM -> dvdk ---------------------------------------
            mh = dm.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            ph = _wait(sig, ph, "dk_attn_free")
            mma_ss(d["k_t_t"], desc_kt, sdesc("Dm", lbo_sq), dvdk_col, True)
            mma_ss(d["k_t"], desc_kt, sdesc("Dm", 16), dvdk_col, True)
            if elect_sync():
                _commit(sig, "dk_total_full")
            mh.release()
            kh.release()
        return k, q, v, do, state, a, tinv, u, z, dm, dv, ph

    # ---- cg0 (warps 0-3): A/dA/dM tiles, the inverse T, dGate/dBeta M-terms ----------------

    @role(warps=range(4), regs=208)
    @cute.jit
    def cg0(
        self,
        params: BpropParams,
        *,
        gate: Consumer,
        beta: Consumer,
        state: Consumer,
        a: Producer,
        tinv: Producer,
        dm: Producer,
        sched: Consumer,
        sig: Barriers,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        ph = Int64(0)
        while work.valid:
            _, _, unit = work.item
            total = params.count[0]
            row = unit
            while row < total:
                gate, beta, state, a, tinv, dm, ph = self.cg0_item(
                    params, Item(params, row), gate, beta, state, a, tinv, dm, sig, ph, tmem, me
                )
                row, sched = self._next_row(sched, skip_fence="before")
            work.advance()
        return gate, beta, state, a, tinv, dm, sched, work

    @cute.jit
    def _mask_rows(self, packs, warp, lane, n_valid):
        """Zero the packed tile-fragment words (word ``j``: row ``warp * 16 + lane // 4 + 8 *
        (j % 2)``) of rows ``>= n_valid``."""
        keep = []
        for r in cutlass.range_constexpr(2):
            keep.append(Int32(-1) if warp * 16 + lane // 4 + r * 8 < n_valid else Int32(0))
        return [packs[j] & keep[j % 2] for j in range(len(packs))]

    @cute.jit
    def _store_frags(self, base, store_row, store_col, packs):
        """stmatrix 16 packed words (an m16n8 fp32 fragment pair per 16 columns) into a
        swizzled 64 x 64 tile at 32-bit shared address ``base``."""
        for c in cutlass.range_constexpr(4):
            off = store_row * B_T + swizzle_xor_128b(store_row, store_col + c * 16)
            stmatrix(base + off * 2, packs[c * 4 : c * 4 + 4])

    @cute.jit
    def _load_frags(self, base, store_row, store_col):
        frags = []
        for c in cutlass.range_constexpr(4):
            off = store_row * B_T + swizzle_xor_128b(store_row, store_col + c * 16)
            frags += ldmatrix(base + off * 2, 4)
        return frags

    @cute.jit
    def cg0_item(
        self, params: BpropParams, it: Item, gate, beta, state, a, tinv, dm, sig, ph, tmem, me
    ):
        io = self.io_dtype
        tid, lane, warp = me.tid, me.lane, me.warp
        lane0 = self._tmem_lanes() << 16
        store_row = warp * 16 + lane % 16
        store_col = (lane // 16) * 8
        n_vals = 32
        acc_zero = Float32(0.0)
        mask_zero = opaque_f32_zero()
        scale = params.scale
        dstate_in0 = 1 if cutlass.const_expr(self.use_dstate_in) else 0
        acc_a = lane0 + self._tmem(tmem, "acc_a")  # KK, then dM core
        acc_b = lane0 + self._tmem(tmem, "acc_b")  # QK, then dA
        s_cumsum, s_cumprod = self.buffer("Cumsum"), self.buffer("Cumprod")
        s_beta = self.buffer("Beta")
        kk_base = self._addr(self.buffer("KK"))
        dm_base = self._addr(self.buffer("Dm"))
        a_base = self._addr(self.buffer("A"))
        tinv_base = self._addr(self.buffer("Tinv"))
        dm_tile = cute.make_tensor(self._ptr(dm_base), cute.make_layout(SQ))
        skk = cute.make_tensor(self._ptr(kk_base, Float32), cute.make_layout(4 * B_T))
        for i in cutlass.range(it.n_chunks, unroll=1):
            has_state = i < it.compute_end - self.first_state_chunk
            # Q and dO rows past the sequence end load as zeros (see the module docstring); A
            # and dA also zero those rows explicitly.
            n_valid = it.batch_end - it.batch_start - (it.compute_end - 1 - i) * B_T
            partial = n_valid < B_T
            # ---- T-pairwise decay ----------------------------------------------------------
            gh = gate.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            gs = gh.index
            row_cumsums = [s_cumsum[warp * 16 + lane // 4 + r * 8, gs] for r in range(2)]
            col_cumsums = [
                s_cumsum[(lane % 4) * 2 + g * 8 + b, gs] for g in range(8) for b in range(2)
            ]
            decay_t = []
            decay_t_strict = []
            for kv in cutlass.range_constexpr(n_vals):
                chunk_row = warp * 16 + lane // 4 + ((kv // 2) % 2) * 8
                chunk_col = (lane % 4) * 2 + ((kv // 4) * 8 + kv % 2)
                decay_t.append(
                    cute.math.exp2(
                        row_cumsums[(kv // 2) % 2] - col_cumsums[(kv // 4) * 2 + (kv % 2)],
                        fastmath=True,
                    )
                    if chunk_row >= chunk_col
                    else mask_zero
                )
                decay_t_strict.append(mask_zero if chunk_row == chunk_col else decay_t[kv])
            cumprod_total = s_cumprod[B_T - 1, gs]
            bh = beta.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            bs = bh.index
            g_beta = [
                s_beta[warp * 16 + lane // 4 + ((kv // 2) % 2) * 8, bs] for kv in range(n_vals)
            ]
            # ---- KK epilogue: L = KK * strict decay -> sKK; M_kk = Beta L -> sDm -----------
            # sDm is free: its last reader, the previous chunk's dM-terms GEMM, completed
            # before this chunk's kk_full commit.
            ph = _wait(sig, ph, "kk_full")
            kk_vec = tmem_ld("16x256b", 8, acc_a)
            l_pack = []
            kk_pack = []
            for j in cutlass.range_constexpr(n_vals // 2):
                p0, p1 = fmul2(
                    kk_vec[2 * j],
                    kk_vec[2 * j + 1],
                    decay_t_strict[2 * j],
                    decay_t_strict[2 * j + 1],
                )
                l_pack.append(fp32_to_fp16(p0, p1, dtype=io))
                v0, v1 = fmul2(p0, p1, g_beta[2 * j], g_beta[2 * j + 1])
                kk_pack.append(fp32_to_fp16(v0, v1, dtype=io))
            for c in cutlass.range_constexpr(4):
                off = store_row * B_T + swizzle_xor_128b(store_row, store_col + c * 16)
                stmatrix(kk_base + off * 2, l_pack[c * 4 : c * 4 + 4])
                stmatrix(dm_base + off * 2, kk_pack[c * 4 : c * 4 + 4])
            # cg0 never stores to TMEM, so each accumulator read completes (tcgen05.wait::ld)
            # before the signal that lets the MMA rewrite it, and before teardown frees TMEM.
            cute.arch.fence_view_async_tmem_load()
            if has_state:
                before_sync_and_arrive(sig, "kk_free")
            # ---- A epilogue: A = QK * decay * scale -> sA ----------------------------------
            ph = _wait(sig, ph, "a_acc_full")
            a_vec = tmem_ld("16x256b", 8, acc_b)
            a_pack = []
            for j in cutlass.range_constexpr(n_vals // 2):
                p0, p1 = fmul2(a_vec[2 * j], a_vec[2 * j + 1], decay_t[2 * j], decay_t[2 * j + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                a_pack.append(fp32_to_fp16(v0, v1, dtype=io))
            if partial:
                a_pack = self._mask_rows(a_pack, warp, lane, n_valid)
            ah = a.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            self._store_frags(a_base, store_row, store_col, a_pack)
            cute.arch.fence_view_async_tmem_load()
            ah.commit()
            # ---- blockwise inverse in place in sDm: T = (I + M_kk)^-1 ----------------------
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            if warp < 2:
                invert_diagonal_8x8(dm_tile, tid // 8, tid, B_T, io)
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            blockwise_8x8_to_16x16(dm_tile, warp * 16, lane, B_T, io)
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            if warp < 2:
                blockwise_16x16_to_32x32(dm_tile, warp * 32, lane, B_T, io)
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            if warp < 2:
                blockwise_32x32_to_64x64(
                    dm_tile,
                    warp,
                    lane,
                    self.inv_sync,
                    B_T,
                    io,
                    skip_fence="both",
                    reason=BOTH_FENCES_COVERED,
                )
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            # ---- U GEMM operand: T[i, j] * Beta[j] -> sTinv ----------------------------------
            beta_col = [
                s_beta[(lane % 4) * 2 + ((kv // 4) * 8 + kv % 2), bs] for kv in range(n_vals)
            ]
            tinv_frags = self._load_frags(dm_base, store_row, store_col)
            tinv_pack = []
            for j in cutlass.range_constexpr(n_vals // 2):
                lo, hi = f16x2_to_f32(tinv_frags[j], dtype=io)
                s0, s1 = fmul2(lo, hi, beta_col[2 * j], beta_col[2 * j + 1])
                tinv_pack.append(fp32_to_fp16(s0, s1, dtype=io))
            th = tinv.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            self._store_frags(tinv_base, store_row, store_col, tinv_pack)
            th.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            # ---- dGate last-state term: cumprod_total * <state, dS> --------------------------
            have_dstate = i + dstate_in0 >= 1
            if have_dstate:
                ph = _wait(sig, ph, "dstate_smem_full", fence=False)
            hdh_last_w = acc_zero
            if has_state:
                sh = state.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
                if have_dstate:
                    hdh_last_w = cumprod_total * self._state_dot(tid)
                sh.release()
            arrive(sig, "state_dot_done")
            # ---- dA epilogue -> sA (A's reader, dU intra, completed before da_acc_full) ------
            ph = _wait(sig, ph, "da_acc_full")
            da_vec = tmem_ld("16x256b", 8, acc_b)
            da_pack = []
            for j in cutlass.range_constexpr(n_vals // 2):
                p0, p1 = fmul2(
                    da_vec[2 * j], da_vec[2 * j + 1], decay_t[2 * j], decay_t[2 * j + 1]
                )
                v0, v1 = fmul2(p0, p1, scale, scale)
                da_pack.append(fp32_to_fp16(v0, v1, dtype=io))
            if partial:
                da_pack = self._mask_rows(da_pack, warp, lane, n_valid)
            dah = a.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            self._store_frags(a_base, store_row, store_col, da_pack)
            cute.arch.fence_view_async_tmem_load()
            dah.commit()
            # ---- dM epilogue: sDm = -Beta[i] strict * (Z U^T) (T's reader, the Z GEMM, ------
            # ---- completed before dm_acc_full). FP16 stages s Z (s = max(Beta, 2^-10)), so ---
            # ---- row i is undone here in FP32. ----------------------------------------------
            ph = _wait(sig, ph, "dm_acc_full")
            dm_vec = tmem_ld("16x256b", 8, acc_a)
            dm_beta = g_beta
            dm_row_beta = [g_beta[0], g_beta[2]]
            dm_inv_s = [acc_zero, acc_zero]
            if cutlass.const_expr(io is cutlass.Float16):
                for rp in cutlass.range_constexpr(2):
                    # Beta / s is exactly min(Beta * 2^10, 1): the floor is a power of two.
                    dm_row_beta[rp] = cute.arch.fmin(
                        g_beta[2 * rp] * Float32(1.0 / Z_FP16_BETA_FLOOR), Float32(1.0)
                    )
                    dm_inv_s[rp] = cute.arch.rcp_approx(
                        cute.arch.fmax(g_beta[2 * rp], Float32(Z_FP16_BETA_FLOOR))
                    )
                dm_beta = [dm_row_beta[(kv // 2) % 2] for kv in range(n_vals)]
            dm_pack = []
            for j in cutlass.range_constexpr(n_vals // 2):
                p0, p1 = fmul2(
                    dm_vec[2 * j],
                    dm_vec[2 * j + 1],
                    decay_t_strict[2 * j],
                    decay_t_strict[2 * j + 1],
                )
                p0, p1 = fmul2(p0, p1, dm_beta[2 * j], dm_beta[2 * j + 1])
                v0 = Float32(0.0) - p0
                v1 = Float32(0.0) - p1
                dm_pack.append(fp32_to_fp16(v0, v1, dtype=io))
            mh = dm.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            self._store_frags(dm_base, store_row, store_col, dm_pack)
            cute.arch.fence_view_async_tmem_load()
            mh.commit()
            # ---- dBeta/dGate M-terms: E = dM core * L ------------------------------------------
            kk_frag = self._load_frags(kk_base, store_row, store_col)
            row_acc = [None] * 8
            col_part = [None] * 16
            for j in cutlass.range_constexpr(n_vals // 2):
                k_lo, k_hi = f16x2_to_f32(kk_frag[j], dtype=io)
                lo, hi = fmul2(dm_vec[2 * j], dm_vec[2 * j + 1], k_lo, k_hi)
                r = cutlass.const_expr((j % 2) * 4 + (j // 2) % 4)
                if cutlass.const_expr(j < 8):
                    row_acc[r] = lo + hi
                else:
                    row_acc[r] += lo + hi
                lo, hi = fmul2(lo, hi, dm_beta[2 * j], dm_beta[2 * j + 1])
                c0 = cutlass.const_expr((j // 2) * 2)
                if cutlass.const_expr(j % 2 == 0):
                    col_part[c0], col_part[c0 + 1] = lo, hi
                else:
                    col_part[c0], col_part[c0 + 1] = fadd2(col_part[c0], col_part[c0 + 1], lo, hi)
            row_part = [
                (row_acc[0] + row_acc[1]) + (row_acc[2] + row_acc[3]),
                (row_acc[4] + row_acc[5]) + (row_acc[6] + row_acc[7]),
            ]
            for off in [1, 2]:
                for rp in cutlass.range_constexpr(2):
                    row_part[rp] += shfl_bfly(row_part[rp], off)
            # ---- dGate column fold (sKK is cg0's scratch once every thread read L) ----------
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            col_lo, col_hi = warp_reduce_scatter_sum_16(col_part, lane)
            tok0 = (lane // 4) * 8 + (lane % 4) * 2
            skk[warp * 64 + tok0] = col_lo
            skk[warp * 64 + tok0 + 1] = col_hi + hdh_last_w if lane == 31 else col_hi
            if lane % 4 == 0:
                for rp in cutlass.range_constexpr(2):
                    crow = warp * 16 + lane // 4 + rp * 8
                    s_cumsum[crow, gs] = acc_zero - row_part[rp] * dm_row_beta[rp]
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            if cutlass.const_expr(self.column_fold_delay > 0):  # noqa: SIM102 (static)
                if warp == 1:
                    t0 = cute.arch.clock64()
                    while cute.arch.clock64() - t0 < Int64(self.column_fold_delay):
                        pass
            if tid < 64:
                col_sum = skk[tid] + skk[64 + tid] + skk[128 + tid] + skk[192 + tid]
                s_cumsum[tid, gs] = s_cumsum[tid, gs] + col_sum
            arrive(sig, "dgate_cg0")
            gh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            # Warp 0's next L store covers sKK's scratch: it waits for warp 1's partial reads.
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            # ---- dBeta M-term ----------------------------------------------------------------
            ph = _wait(sig, ph, "dbeta_cg1", fence=False)
            if lane % 4 == 0:
                for rp in cutlass.range_constexpr(2):
                    crow = warp * 16 + lane // 4 + rp * 8
                    if cutlass.const_expr(io is cutlass.Float16):
                        db = s_beta[crow, bs] - row_part[rp] * dm_inv_s[rp]
                    else:
                        db = s_beta[crow, bs] - row_part[rp]
                    s_beta[crow, bs] = db
            bh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        return gate, beta, state, a, tinv, dm, ph

    @cute.jit
    def _state_dot(self, tid):
        """``<S, dS>`` over this warp's rows of the fp16 state and state gradient tiles (both
        ``[v, k]`` in 64-column swizzled boxes), summed over the warp."""
        io, d = self.io_dtype, self.d
        dstate_base = self._addr(self.buffer("Dstate"))
        state_base = self._addr(self.buffer("State"))
        lo = [opaque_f32_zero() for _ in range(4)]
        hi = [opaque_f32_zero() for _ in range(4)]
        slabs = d // 64
        reps = (slabs * d + 127) // 128
        step = 128 // d
        row = tid % d
        for rep in cutlass.range_constexpr(reps):
            slab = tid // d + Int32(rep * step)
            if slab < Int32(slabs):
                for oct_ in cutlass.range_constexpr(8):
                    off = (slab * (d * 64) + row * 64 + swizzle_xor_128b(row, oct_ * 8)) * 2
                    ds_words = cute.make_tensor(
                        self._ptr(dstate_base + off, Int32, 16), cute.make_layout(4)
                    ).load()
                    s_words = cute.make_tensor(
                        self._ptr(state_base + off, Int32, 16), cute.make_layout(4)
                    ).load()
                    for w in cutlass.range_constexpr(4):
                        ds_lo, ds_hi = f16x2_to_f32(ds_words[w], dtype=io)
                        s_lo, s_hi = f16x2_to_f32(s_words[w], dtype=io)
                        lo[w], hi[w] = ffma2(ds_lo, ds_hi, s_lo, s_hi, lo[w], hi[w])
        dot = ((lo[0] + lo[1]) + (lo[2] + lo[3])) + ((hi[0] + hi[1]) + (hi[2] + hi[3]))
        for off in [1, 2, 4, 8, 16]:
            dot += shfl_bfly(dot, off)
        return dot

    # ---- cg1 (warps 4-7): TMEM operand staging, U/Z/dV/dQ tiles, dBeta V-term -----------

    @role(warps=range(4, 8), regs=144)
    @cute.jit
    def cg1(
        self,
        params: BpropParams,
        *,
        gate: Consumer,
        beta: Consumer,
        q: Consumer,
        v: Consumer,
        do: Consumer,
        u: Producer,
        z: Producer,
        dq: Producer,
        dk: Producer,
        dv: Producer,
        sched: Consumer,
        sig: Barriers,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        ph = Int64(0)
        while work.valid:
            _, _, unit = work.item
            total = params.count[0]
            row = unit
            while row < total:
                gate, beta, q, v, do, u, z, dq, dk, dv, ph = self.cg1_item(
                    params,
                    Item(params, row),
                    gate,
                    beta,
                    q,
                    v,
                    do,
                    u,
                    z,
                    dq,
                    dk,
                    dv,
                    sig,
                    ph,
                    tmem,
                    me,
                )
                row, sched = self._next_row(sched)
            work.advance()
        return gate, beta, q, v, do, u, z, dq, dk, dv, sched, work

    @cute.jit
    def _frag_offsets(self, tid):
        """Element offsets of thread ``tid``'s ldmatrix.trans / stmatrix.trans fragments of a
        swizzled ``[64, d]`` tile: ``offs[half][block]``."""
        frag_row = tid % 8 + (tid // 16 % 2) * 8
        if cutlass.const_expr(self.d == 128):
            frag_col = (tid // 8 % 2) * 8 + (tid // 32 % 2) * 32
            segment = (tid // 64) * (B_T * 64)
        else:
            frag_col = (tid // 32) * 16 + (tid // 8 % 2) * 8
            segment = Int32(0)
        offs = []
        for half in cutlass.range_constexpr(self.d // 64):
            row_offs = []
            for block in cutlass.range_constexpr(4):
                r = frag_row + block * 16
                row_offs.append(segment + r * 64 + swizzle_xor_128b(r, frag_col + half * 16))
            offs.append(row_offs)
        return offs

    @cute.jit
    def _load_tile_frags(self, base, offs):
        """ldmatrix.trans of a ``[64, d]`` tile: ``frags[half]`` (16 packed words each)."""
        frags = []
        for half in cutlass.range_constexpr(self.d // 64):
            words = []
            for block in cutlass.range_constexpr(4):
                words += ldmatrix(base + offs[half][block] * 2, 4, trans=True)
            frags.append(words)
        return frags

    @cute.jit
    def _store_tile(self, base, offs, regs, scale=None):
        """stmatrix.trans of fp32 accumulator rows ``regs[half]`` (times ``scale`` pairs when
        given) into a ``[64, d]`` tile."""
        io = self.io_dtype
        for half in cutlass.range_constexpr(self.d // 64):
            for block in cutlass.range_constexpr(4):
                pack = []
                for j in cutlass.range_constexpr(4):
                    e = cutlass.const_expr(8 * block + 2 * j)
                    x0, x1 = regs[half][e], regs[half][e + 1]
                    if cutlass.const_expr(scale is not None):
                        x0, x1 = fmul2(x0, x1, scale[e], scale[e + 1])
                    pack.append(fp32_to_fp16(x0, x1, dtype=io))
                stmatrix(base + offs[half][block] * 2, pack, trans=True)

    @cute.jit
    def _rows(self, lane_base, half, col):
        """TMEM address of column ``col`` for this warp's ``half``-th 16 lanes."""
        return ((lane_base + half * 16) << 16) + col

    @cute.jit
    def cg1_item(
        self,
        params: BpropParams,
        it: Item,
        gate,
        beta,
        q,
        v,
        do,
        u,
        z,
        dq,
        dk,
        dv,
        sig,
        ph,
        tmem,
        me,
    ):
        io, d = self.io_dtype, self.d
        tid, lane, warp = me.tid, me.lane, me.warp
        lane_base = self._tmem_lanes()
        lane0 = lane_base << 16
        halves = d // 64
        n_vals = 32
        scale = params.scale
        offs = self._frag_offsets(tid)
        dstate_col = self._tmem(tmem, "dstate")
        dvdk_col = self._tmem(tmem, "dvdk")
        dsin_col = self._tmem(tmem, "dstate_in")
        ks_col = self._tmem(tmem, "acc_a")  # KS, then Z
        u_col = self._tmem(tmem, "acc_b")
        do_prime_col = self._tmem(tmem, "inputs")
        du_col = do_prime_col + B_T // 2  # dU, then dY'
        y_col = self._tmem(tmem, "y")
        gks_col = self._tmem(tmem, "gks")
        s_cumsum, s_cumprod = self.buffer("Cumsum"), self.buffer("Cumprod")
        s_beta = self.buffer("Beta")
        v_base = self._addr(self.buffer("V"))
        do_base = self._addr(self.buffer("dO"))
        q_base = self._addr(self.buffer("Q"))
        dq_base = self._addr(self.buffer("dQ"))
        dv_base = self._addr(self.buffer("dV"))
        # dBeta/dGate scratch in sdK (idle from its TMA store until cg2's dK fold), after
        # cg2's part_k and cg1's part_q slots.
        sdk = cute.make_tensor(
            self._ptr(self._addr(self.buffer("dK")), Float32), cute.make_layout(1024)
        )
        part_q_off = 4 * 64
        v_term_off = 8 * 64
        tok0 = (lane // 4) * 8 + (lane % 4) * 2
        for rev in cutlass.range(it.n_chunks, unroll=1):
            chunk = it.compute_end - 1 - rev
            have_dstate = cutlass.Boolean(True)
            if cutlass.const_expr(not self.use_dstate_in):
                have_dstate = rev > 0
            has_state = chunk >= self.first_state_chunk
            n_valid = it.batch_end - it.batch_start - chunk * B_T
            gh = gate.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            gs = gh.index
            bh = beta.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            bs = bh.index
            cumprod_fp32 = [
                s_cumprod[(lane % 4) * 2 + g * 8 + b, gs] for g in range(8) for b in range(2)
            ]
            cumprod_vals = [cumprod_fp32[(kv // 4) * 2 + (kv % 2)] for kv in range(n_vals)]
            # ---- dS *= this chunk's cumprod (after cg2 read dS into sDstate) -----------------
            if have_dstate:
                ph = _wait(sig, ph, "dstate_smem_full")
                cumprod_top = s_cumprod[B_T - 1, gs]
                if cutlass.const_expr(d == 64):
                    vec = tmem_ld("16x256b", 8, lane0 + dstate_col)
                    scaled = []
                    for j in cutlass.range_constexpr(16):
                        h0, h1 = fmul2(vec[2 * j], vec[2 * j + 1], cumprod_top, cumprod_top)
                        scaled += [h0, h1]
                    tmem_st("16x256b", lane0 + dstate_col, scaled)
                else:
                    for i in cutlass.range_constexpr(d // 32):
                        vec = tmem_ld("32x32b", 32, lane0 + dstate_col + i * 32)
                        scaled = []
                        for j in cutlass.range_constexpr(16):
                            h0, h1 = fmul2(vec[2 * j], vec[2 * j + 1], cumprod_top, cumprod_top)
                            scaled += [h0, h1]
                        tmem_st("32x32b", lane0 + dstate_col + i * 32, scaled)
                cute.arch.fence_view_async_tmem_store()
                before_sync_and_arrive(sig, "dstate_scaled")
            # ---- V and dO fragments ------------------------------------------------------
            vh = v.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            v_frag = self._load_tile_frags(v_base, offs)
            doh = do.wait()
            do_frag = self._load_tile_frags(do_base, offs)
            # ---- Y = V - fp16(cumprod * KS) and 2^g KS -> TMEM ------------------------------
            if has_state:
                ph = _wait(sig, ph, "ks_full")
                for half in cutlass.range_constexpr(halves):
                    ks = tmem_ld("16x256b", 8, self._rows(lane_base, half, ks_col))
                    gks_pack = []
                    y_pack = []
                    for j in cutlass.range_constexpr(16):
                        g0, g1 = fmul2(
                            ks[2 * j], ks[2 * j + 1], cumprod_vals[2 * j], cumprod_vals[2 * j + 1]
                        )
                        word = fp32_to_fp16(g0, g1, dtype=io)
                        gks_pack.append(word)
                        y_pack.append(sub_f16x2(v_frag[half][j], word, dtype=io))
                    tmem_st("16x128b", self._rows(lane_base, half, y_col), y_pack)
                    tmem_st("16x128b", self._rows(lane_base, half, gks_col), gks_pack)
            else:
                for half in cutlass.range_constexpr(halves):
                    tmem_st("16x128b", self._rows(lane_base, half, y_col), v_frag[half])
            cute.arch.fence_view_async_tmem_store()
            before_sync_and_arrive(sig, "y_full")
            # ---- dV inter *= 2^(g_63 - g) in place ------------------------------------------
            if have_dstate:
                last_cumsum = s_cumsum[B_T - 1, gs]
                col_cumsum = [
                    s_cumsum[(lane % 4) * 2 + g * 8 + b, gs] for g in range(8) for b in range(2)
                ]
                decay_scale_fp32 = [
                    cute.math.exp2(last_cumsum - col_cumsum[i], fastmath=True) for i in range(16)
                ]
                decay_scale = [decay_scale_fp32[(kv // 4) * 2 + (kv % 2)] for kv in range(n_vals)]
                ph = _wait(sig, ph, "du_scale_full")
                dv_vecs = []
                for half in cutlass.range_constexpr(halves):
                    dv_vecs.append(tmem_ld("16x256b", 8, self._rows(lane_base, half, dvdk_col)))
                for half in cutlass.range_constexpr(halves):
                    scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(
                            dv_vecs[half][2 * j],
                            dv_vecs[half][2 * j + 1],
                            decay_scale[2 * j],
                            decay_scale[2 * j + 1],
                        )
                        scaled += [s0, s1]
                    tmem_st("16x256b", self._rows(lane_base, half, dvdk_col), scaled)
                cute.arch.fence_view_async_tmem_store()
                before_sync_and_arrive(sig, "du_scale_free")
            # ---- dO' = dO * cumprod * scale -> TMEM (cg2 read the last state path) -----------
            ph = _wait(sig, ph, "cg2_chunk")
            for half in cutlass.range_constexpr(halves):
                do_pack = []
                for j in cutlass.range_constexpr(16):
                    lo, hi = f16x2_to_f32(do_frag[half][j], dtype=io)
                    p0, p1 = fmul2(lo, hi, cumprod_vals[2 * j], cumprod_vals[2 * j + 1])
                    q0, q1 = fmul2(p0, p1, scale, scale)
                    do_pack.append(fp32_to_fp16(q0, q1, dtype=io))
                if n_valid < B_T:  # upstream's zero dO rows past the sequence end
                    do_pack = _mask_words(do_pack, _token_masks(lane, n_valid))
                tmem_st("16x128b", self._rows(lane_base, half, do_prime_col), do_pack)
            cute.arch.fence_view_async_tmem_store()
            before_sync_and_arrive(sig, "do_prime_full")
            # ---- dU -> fp16 TMEM --------------------------------------------------------------
            ph = _wait(sig, ph, "du_total_full")
            for half in cutlass.range_constexpr(halves):
                du = tmem_ld("16x256b", 8, self._rows(lane_base, half, dvdk_col))
                du_pack = [fp32_to_fp16(du[2 * j], du[2 * j + 1], dtype=io) for j in range(16)]
                tmem_st("16x128b", self._rows(lane_base, half, du_col), du_pack)
            cute.arch.fence_view_async_tmem_store()
            before_sync_and_arrive(sig, "du_in_full")
            # ---- dQ inter *= cumprod * scale in place ---------------------------------------
            if has_state:
                ph = _wait(sig, ph, "dq_scale_full")
                dqi = []
                for half in cutlass.range_constexpr(halves):
                    dqi.append(tmem_ld("16x256b", 8, self._rows(lane_base, half, dsin_col)))
                for half in cutlass.range_constexpr(halves):
                    scaled = []
                    for j in cutlass.range_constexpr(16):
                        p0, p1 = fmul2(
                            dqi[half][2 * j],
                            dqi[half][2 * j + 1],
                            cumprod_vals[2 * j],
                            cumprod_vals[2 * j + 1],
                        )
                        s0, s1 = fmul2(p0, p1, scale, scale)
                        scaled += [s0, s1]
                    tmem_st("16x256b", self._rows(lane_base, half, dsin_col), scaled)
                cute.arch.fence_view_async_tmem_store()
                before_sync_and_arrive(sig, "dq_scale_free")
            # ---- U -> sV (over V) --------------------------------------------------------------
            ph = _wait(sig, ph, "u_acc_full")
            u_regs = []
            for half in cutlass.range_constexpr(halves):
                u_regs.append(tmem_ld("16x256b", 8, self._rows(lane_base, half, u_col)))
            uh = u.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            self._store_tile(v_base, offs, u_regs)
            uh.commit()
            # The u commit's fence (every thread) ordered this group's V/dO reads before the
            # async proxy; the releases need no fence of their own (release_fence=()).
            vh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            doh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            # ---- Z -> sdQ (the dM core operand) -------------------------------------------------
            ph = _wait(sig, ph, "z_acc_full")
            z_regs = []
            for half in cutlass.range_constexpr(halves):
                z_regs.append(tmem_ld("16x256b", 8, self._rows(lane_base, half, ks_col)))
            # Z = T^T dU with T beta-free: the gradient w.r.t. Beta*Y. The Beta loads precede
            # the cg1 barrier below, after which the beta stage holds dBeta's V-term.
            beta_fp32 = [
                s_beta[(lane % 4) * 2 + g * 8 + b, bs] for g in range(8) for b in range(2)
            ]
            beta_vals = [beta_fp32[(kv // 4) * 2 + (kv % 2)] for kv in range(n_vals)]
            dqh = dq.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            zh = z.acquire()
            if cutlass.const_expr(io is cutlass.Float16):
                # FP16 range floor: beta-free Z can exceed 65504 where Beta*Z cannot, so FP16
                # stages s Z with s = max(Beta, 2^-10) and cg0 divides dM core row i by s[i].
                z_s = [
                    cute.arch.fmax(beta_vals[kv], Float32(Z_FP16_BETA_FLOOR))
                    for kv in range(n_vals)
                ]
                self._store_tile(dq_base, offs, z_regs, z_s)
            else:
                self._store_tile(dq_base, offs, z_regs)
            zh.commit()
            # ---- dY' = -Beta cumprod Z -> TMEM over dU ---------------------------------------
            neg_beta_cumprod = [
                Float32(0.0) - cumprod_vals[kv] * beta_vals[kv] for kv in range(n_vals)
            ]
            for half in cutlass.range_constexpr(halves):
                dyp_pack = []
                for j in cutlass.range_constexpr(16):
                    n0, n1 = fmul2(
                        z_regs[half][2 * j],
                        z_regs[half][2 * j + 1],
                        neg_beta_cumprod[2 * j],
                        neg_beta_cumprod[2 * j + 1],
                    )
                    dyp_pack.append(fp32_to_fp16(n0, n1, dtype=io))
                tmem_st("16x128b", self._rows(lane_base, half, du_col), dyp_pack)
            cute.arch.fence_view_async_tmem_store()
            before_sync_and_arrive(sig, "dyp_full")
            # ---- dV = Beta Z -> sdV (the dV output and dK's state-path operand) ----------------
            dvh = dv.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            self._store_tile(dv_base, offs, z_regs, beta_vals)
            dvh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            dkh = dk.acquire()  # sdK is this group's scratch until the dQ dot
            # ---- dBeta V-term rowsum(Z * Y); dGate state term Beta * rowsum(Z * 2^g KS) ------
            gsum = Float32(0.0)
            part_y = self._row_parts(z_regs, lane_base, y_col)
            py_lo, py_hi = warp_reduce_scatter_sum_16(part_y, lane)
            if has_state:
                part_g = self._row_parts(z_regs, lane_base, gks_col)
                pg_lo, pg_hi = warp_reduce_scatter_sum_16(part_g, lane)
                sdk[v_term_off + warp * 64 + tok0] = py_lo
                sdk[v_term_off + warp * 64 + tok0 + 1] = py_hi
                sdk[v_term_off + 256 + warp * 64 + tok0] = pg_lo
                sdk[v_term_off + 256 + warp * 64 + tok0 + 1] = pg_hi
                self.cg1_sync.sync(aligned=True, skip_fence="after", reason=AFTER_FENCE_COVERED)
                if tid < 64:
                    ysum = (
                        sdk[v_term_off + tid]
                        + sdk[v_term_off + 64 + tid]
                        + sdk[v_term_off + 128 + tid]
                        + sdk[v_term_off + 192 + tid]
                    )
                    gsum = s_beta[tid, bs] * (
                        sdk[v_term_off + 256 + tid]
                        + sdk[v_term_off + 320 + tid]
                        + sdk[v_term_off + 384 + tid]
                        + sdk[v_term_off + 448 + tid]
                    )
                    s_beta[tid, bs] = ysum
            else:
                sdk[v_term_off + warp * 64 + tok0] = py_lo
                sdk[v_term_off + warp * 64 + tok0 + 1] = py_hi
                self.cg1_sync.sync(aligned=True, skip_fence="after", reason=AFTER_FENCE_COVERED)
                if tid < 64:
                    ysum = (
                        sdk[v_term_off + tid]
                        + sdk[v_term_off + 64 + tid]
                        + sdk[v_term_off + 128 + tid]
                        + sdk[v_term_off + 192 + tid]
                    )
                    s_beta[tid, bs] = ysum
            bh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            arrive(sig, "dbeta_cg1")
            self.cg1_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            # ---- Q fragments (upstream also stored them to the Y TMEM slot, never read) -------
            qh = q.wait(skip_fence="both", reason=BOTH_FENCES_COVERED)
            q_frag = self._load_tile_frags(q_base, offs)
            # Release Q right away (with its own proxy fence), as upstream does: the next
            # chunk's Q load, and so its QK GEMM, waits for this release.
            fence_async_shared()
            qh.release()
            # ---- dQ -> sdQ (over Z: the dM core GEMM, its reader, precedes dq_total_full) ------
            ph = _wait(sig, ph, "dq_total_full")
            dq_regs = []
            for half in cutlass.range_constexpr(halves):
                dq_regs.append(tmem_ld("16x256b", 8, self._rows(lane_base, half, dsin_col)))
            cute.arch.fence_view_async_tmem_load()
            before_sync_and_arrive(sig, "dq_total_free")
            self._store_tile(dq_base, offs, dq_regs)
            dqh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            # ---- dGate Q part: rowsum(dQ * Q) ----------------------------------------------------
            part_q = [None] * 16
            for half in cutlass.range_constexpr(halves):
                for block in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(4):
                        q_lo, q_hi = f16x2_to_f32(q_frag[half][4 * block + i], dtype=io)
                        e0 = cutlass.const_expr(8 * block + 2 * i)
                        p = cutlass.const_expr((e0 // 4) * 2 + (e0 % 2))
                        if cutlass.const_expr(half == 0 and i % 2 == 0):
                            part_q[p], part_q[p + 1] = fmul2(
                                dq_regs[half][e0], dq_regs[half][e0 + 1], q_lo, q_hi
                            )
                        else:
                            part_q[p], part_q[p + 1] = ffma2(
                                dq_regs[half][e0],
                                dq_regs[half][e0 + 1],
                                q_lo,
                                q_hi,
                                part_q[p],
                                part_q[p + 1],
                            )
            pq_lo, pq_hi = warp_reduce_scatter_sum_16(part_q, lane)
            # Tokens past the sequence end load zero Q and dO; their dQ . Q partials are also
            # zeroed explicitly.
            pq_lo = pq_lo if tok0 < n_valid else Float32(0.0)
            pq_hi = pq_hi if tok0 + 1 < n_valid else Float32(0.0)
            sdk[part_q_off + warp * 64 + tok0] = pq_lo
            sdk[part_q_off + warp * 64 + tok0 + 1] = pq_hi
            self.cg1_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            ph = _wait(sig, ph, "dgate_cg0", fence=False)
            if tid < 64:
                pq_sum = (
                    sdk[part_q_off + tid]
                    + sdk[part_q_off + 64 + tid]
                    + sdk[part_q_off + 128 + tid]
                    + sdk[part_q_off + 192 + tid]
                )
                s_cumsum[tid, gs] = s_cumsum[tid, gs] + pq_sum - gsum
            # cg1 only used sdK as scratch: cg2 overwrites it with the dK tile (and fences that
            # for the TMA store) after dgate_cg1, so cg1's arrival needs no proxy fence.
            cute.arch.mbarrier_arrive(dkh.full_barrier)
            dkh.commit(signal=False)
            gh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            arrive(sig, "dgate_cg1")
        return gate, beta, q, v, do, u, z, dq, dk, dv, ph

    @cute.jit
    def _row_parts(self, z_regs, lane_base, col):
        """Per-token partials of ``rowsum(Z * X)`` for the fp16 TMEM slot ``X`` at ``col``."""
        io = self.io_dtype
        part = [None] * 16
        for half in cutlass.range_constexpr(self.d // 64):
            x = tmem_ld("16x128b", 8, self._rows(lane_base, half, col), Int32)
            for j in cutlass.range_constexpr(16):
                lo, hi = f16x2_to_f32(x[j], dtype=io)
                e0 = cutlass.const_expr(2 * j)
                p = cutlass.const_expr((e0 // 4) * 2 + (e0 % 2))
                if cutlass.const_expr(half == 0 and j % 2 == 0):
                    part[p], part[p + 1] = fmul2(z_regs[half][e0], z_regs[half][e0 + 1], lo, hi)
                else:
                    part[p], part[p + 1] = ffma2(
                        z_regs[half][e0], z_regs[half][e0 + 1], lo, hi, part[p], part[p + 1]
                    )
        return part

    # ---- cg2 (warps 8-11): the dV/dK accumulator's read side, next dS, dK -----------------

    @role(warps=range(8, 12), regs=128)
    @cute.jit
    def cg2(
        self,
        params: BpropParams,
        *,
        gate: Consumer,
        k: Consumer,
        dk: Producer,
        sched: Consumer,
        sig: Barriers,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        ph = Int64(_phases("dstate_in_free"))
        while work.valid:
            _, _, unit = work.item
            total = params.count[0]
            row = unit
            while row < total:
                gate, k, dk, ph = self.cg2_item(
                    params, row, Item(params, row), gate, k, dk, sig, ph, tmem, me
                )
                row, sched = self._next_row(sched)
            work.advance()
        return gate, k, dk, sched, work

    @cute.jit
    def _state_row(self, tid):
        """``(v row, row valid)`` of thread ``tid``'s TMEM lane: M = 64 accumulators use the
        first 16 lanes of each warp."""
        if cutlass.const_expr(self.d == 128):
            return tid, cutlass.Boolean(True)
        return (tid // 32) * 16 + (tid % 32) % 16, (tid % 32) < 16

    @cute.jit
    def _dstate_to_smem(self, lane0, dstate_col, row, row_valid):
        """fp16 ``dS^T`` rows from TMEM into sDstate (``[v, k]`` in 64-column swizzled boxes)."""
        io, d = self.io_dtype, self.d
        base = self._addr(self.buffer("Dstate"))
        for i in cutlass.range_constexpr(d // 32):
            vec = tmem_ld("32x32b", 32, lane0 + dstate_col + i * 32)
            for g in cutlass.range_constexpr(4):
                pack = [
                    fp32_to_fp16(vec[g * 8 + 2 * t], vec[g * 8 + 2 * t + 1], dtype=io)
                    for t in range(4)
                ]
                col = i * 32 + g * 8
                box, in_box = col // BOX, col % BOX
                off = box * d * BOX + row * BOX + swizzle_xor_128b(row, in_box)
                if row_valid:
                    st_shared_v4(base + off * 2, pack)
        cute.arch.fence_proxy("async.shared", space="cta")

    @cute.jit
    def _state_vector(self, tensor, i0, head, row, col0, count: cutlass.Constexpr[int]):
        """``tensor[i0, head, row, col0 : col0 + count]`` as fp32 (16-byte loads)."""
        src = tensor[i0, head, row, None]
        dtype = src.element_type
        vw = 16 // (dtype.width // 8)
        vals = []
        for g in cutlass.range_constexpr(count // vw):
            addr = (src.iterator + (col0 + g * vw)).toint()
            vec = cute.make_tensor(
                cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout(vw),
            ).load()
            vals += [vec[t].to(Float32) for t in range(vw)]
        return vals

    @cute.jit
    def _scale_dk_inter(self, lane_base, dvdk_col, k_stage, offs, decay, masks=None):
        """dK inter *= ``decay`` in place; returns this thread's part of ``<K, dK inter>``, with
        the K fragment words ANDed with ``masks`` (``_token_masks``) when given."""
        io = self.io_dtype
        lo = [opaque_f32_zero() for _ in range(4)]
        hi = [opaque_f32_zero() for _ in range(4)]
        for half in cutlass.range_constexpr(self.d // 64):
            addr = self._rows(lane_base, half, dvdk_col)
            dki = tmem_ld("16x256b", 8, addr)
            scaled = []
            for j in cutlass.range_constexpr(16):
                s0, s1 = fmul2(dki[2 * j], dki[2 * j + 1], decay[2 * j], decay[2 * j + 1])
                scaled += [s0, s1]
            tmem_st("16x256b", addr, scaled)
            for block in cutlass.range_constexpr(4):
                k_frag = ldmatrix(k_stage + offs[half][block] * 2, 4, trans=True)
                for i in cutlass.range_constexpr(4):
                    k_word = k_frag[i]
                    if cutlass.const_expr(masks is not None):
                        k_word = k_word & masks[(4 * block + i) // 2]
                    k_lo, k_hi = f16x2_to_f32(k_word, dtype=io)
                    lo[i], hi[i] = ffma2(
                        scaled[8 * block + 2 * i],
                        scaled[8 * block + 2 * i + 1],
                        k_lo,
                        k_hi,
                        lo[i],
                        hi[i],
                    )
        return ((lo[0] + lo[1]) + (lo[2] + lo[3])) + ((hi[0] + hi[1]) + (hi[2] + hi[3]))

    @cute.jit
    def cg2_item(self, params: BpropParams, item_row, it: Item, gate, k, dk, sig, ph, tmem, me):
        io, d = self.io_dtype, self.d
        tid, lane, warp = me.tid, me.lane, me.warp
        lane_base = self._tmem_lanes()
        lane0 = lane_base << 16
        n_vals = 32
        acc_zero = Float32(0.0)
        offs = self._frag_offsets(tid)
        state_row, row_valid = self._state_row(tid)
        dvdk_col = self._tmem(tmem, "dvdk")
        spath_col = self._tmem(tmem, "inputs")
        dstate_col = self._tmem(tmem, "dstate")
        dsin_col = self._tmem(tmem, "dstate_in")
        s_cumsum, s_cumprod = self.buffer("Cumsum"), self.buffer("Cumprod")
        k_base = self._addr(self.buffer("K"))
        dk_base = self._addr(self.buffer("dK"))
        sdk = cute.make_tensor(self._ptr(dk_base, Float32), cute.make_layout(4 * 64))
        dstate_in0 = 1 if cutlass.const_expr(self.use_dstate_in) else 0
        tok0 = (lane // 4) * 8 + (lane % 4) * 2
        n = it.n_chunks
        # ---- dS seed: dstate_in when the item ends the sequence (a split warmup seeds zero) --
        if cutlass.const_expr(self.use_dstate_in):  # noqa: SIM102 (static)
            if n > 0:
                n_seq_chunks = (it.batch_end - it.batch_start + (B_T - 1)) // B_T
                seed = it.compute_end == n_seq_chunks
                ph = _wait(sig, ph, "dstate_in_free")
                for i in cutlass.range_constexpr(d // 16):
                    block = cute.make_rmem_tensor((16,), Float32)
                    for t in cutlass.range_constexpr(16):
                        block[t] = Float32(0.0)
                    if seed:
                        vals = self._state_vector(
                            params.dstate_in, it.batch, it.head, state_row, i * 16, 16
                        )
                        for t in cutlass.range_constexpr(16):
                            block[t] = vals[t]
                    tmem_st("32x32b", lane0 + dstate_col + i * 16, [block[t] for t in range(16)])
                    pack = [
                        fp32_to_fp16(block[2 * p], block[2 * p + 1], dtype=io) for p in range(8)
                    ]
                    tmem_st("32x32b", lane0 + dsin_col + i * 8, pack)
                cute.arch.fence_view_async_tmem_store()
                before_sync_and_arrive(sig, "dstate_in_full")
                self._dstate_to_smem(lane0, dstate_col, state_row, row_valid)
                before_sync_and_arrive(sig, "dstate_smem_full")
        for rev in cutlass.range(n, unroll=1):
            chunk = it.compute_end - 1 - rev
            # ---- latch the chunk's decay 2^(g_63 - g) ------------------------------------------
            gh = gate.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            gs = gh.index
            last_cumsum = s_cumsum[B_T - 1, gs]
            decay_fp32 = [
                cute.math.exp2(
                    last_cumsum - s_cumsum[(lane % 4) * 2 + (i // 2) * 8 + i % 2, gs],
                    fastmath=True,
                )
                for i in range(16)
            ]
            decay = [decay_fp32[(kv // 4) * 2 + (kv % 2)] for kv in range(n_vals)]
            before_sync_and_arrive(sig, "cg2_chunk")
            # The MMA warp waited for this K stage before the dK GEMMs this group waits for.
            kh = k.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
            k_stage = k_base + kh.index * self.tile_bytes
            # ---- dK inter *= decay in place; <K, dK inter> for dGate -------------------------
            k_dot = Float32(0.0)
            if rev + dstate_in0 >= 1:
                ph = _wait(sig, ph, "dk_scale_full")
                k_part = Float32(0.0)
                if cutlass.const_expr(self.use_dstate_in):
                    # Only the item's last chunk (rev 0, the seeded one) can end inside the
                    # sequence. Its K rows past the end load as zeros; items whose sequence
                    # ends before token 64 also mask them, in a copy only those one-chunk items
                    # run. part_k needs no mask (cg2 drops those tokens' partials).
                    if it.batch_end < B_T:
                        n_valid = it.batch_end - it.batch_start - chunk * B_T
                        k_part = self._scale_dk_inter(
                            lane_base, dvdk_col, k_stage, offs, decay, _token_masks(lane, n_valid)
                        )
                    else:
                        k_part = self._scale_dk_inter(lane_base, dvdk_col, k_stage, offs, decay)
                else:
                    k_part = self._scale_dk_inter(lane_base, dvdk_col, k_stage, offs, decay)
                cute.arch.fence_view_async_tmem_store()
                before_sync_and_arrive(sig, "dk_scale_free")
                k_dot = k_part
                for off in [1, 2, 4, 8, 16]:
                    k_dot += shfl_bfly(k_dot, off)
            # ---- dK attn -> registers ---------------------------------------------------------
            ph = _wait(sig, ph, "dk_attn_full")
            dks = []
            for half in cutlass.range_constexpr(d // 64):
                dks.append(tmem_ld("16x256b", 8, self._rows(lane_base, half, dvdk_col)))
            cute.arch.fence_view_async_tmem_load()
            before_sync_and_arrive(sig, "dk_attn_free")
            dkh = dk.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            # ---- dGate K part: rowsum(dK attn * K) ---------------------------------------------
            part_k = [acc_zero] * 16
            for half in cutlass.range_constexpr(d // 64):
                for block in cutlass.range_constexpr(4):
                    k_frag = ldmatrix(k_stage + offs[half][block] * 2, 4, trans=True)
                    for i in cutlass.range_constexpr(4):
                        k_lo, k_hi = f16x2_to_f32(k_frag[i], dtype=io)
                        e0 = cutlass.const_expr(8 * block + 2 * i)
                        p = cutlass.const_expr((e0 // 4) * 2 + (e0 % 2))
                        if cutlass.const_expr(half == 0 and i % 2 == 0):
                            part_k[p], part_k[p + 1] = fmul2(
                                dks[half][e0], dks[half][e0 + 1], k_lo, k_hi
                            )
                        else:
                            part_k[p], part_k[p + 1] = ffma2(
                                dks[half][e0],
                                dks[half][e0 + 1],
                                k_lo,
                                k_hi,
                                part_k[p],
                                part_k[p + 1],
                            )
            kh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            pk_lo, pk_hi = warp_reduce_scatter_sum_16(part_k, lane)
            # Tokens past the sequence end load zero K rows; their partials are also zeroed
            # explicitly.
            n_valid = it.batch_end - it.batch_start - chunk * B_T
            pk_lo = pk_lo if tok0 < n_valid else acc_zero
            pk_hi = pk_hi if tok0 + 1 < n_valid else acc_zero
            sdk[warp * 64 + tok0] = pk_lo
            sdk[warp * 64 + tok0 + 1] = pk_hi - k_dot if lane == 31 else pk_hi
            ph = _wait(sig, ph, "dq_total_free", fence=False)  # dQ left the dstate_in columns
            # ---- the next chunk's dS: fp16 operand, then sDstate --------------------------------
            if chunk >= it.write_start + 1:
                ph = _wait(sig, ph, "dstate_full", fence=False)
                ph = _wait(sig, ph, "dstate_in_free")
                for i in cutlass.range_constexpr(d // 32):
                    vec = tmem_ld("32x32b", 32, lane0 + dstate_col + i * 32)
                    pack = [fp32_to_fp16(vec[2 * j], vec[2 * j + 1], dtype=io) for j in range(16)]
                    tmem_st("32x32b", lane0 + dsin_col + i * 16, pack)
                cute.arch.fence_view_async_tmem_store()
                before_sync_and_arrive(sig, "dstate_in_full")
            if chunk >= it.write_start + 1:
                ph = _wait(sig, ph, "state_dot_done")
                self._dstate_to_smem(lane0, dstate_col, state_row, row_valid)
                before_sync_and_arrive(sig, "dstate_smem_full")
            else:
                # cg0's last-chunk arrival goes unwaited; flipping the parity counts it. Its
                # sDstate read still precedes this group's next write (the next item's seed or
                # chunk): cg0 reads, then arrives dgate_cg0, which cg1 waits for before the
                # dgate_cg1 this group waits for below. cg0 cannot lap a waited phase either:
                # the arrival after one is for a later chunk of the same item, and cg0 waits
                # first for the dstate_smem_full this group arrives after that wait.
                ph = ph ^ Int64(1 << SIG["state_dot_done"])
            # ---- dGate fold ----------------------------------------------------------------------
            ph = _wait(sig, ph, "dgate_cg1", fence=False)
            self.cg2_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            if tid < 64:
                dgate_sum = sdk[tid] + sdk[64 + tid] + sdk[128 + tid] + sdk[192 + tid]
                s_cumsum[tid, gs] = s_cumsum[tid, gs] - dgate_sum
            self.cg2_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            # ---- dK = dM-terms total - cumprod * (dV S^T) -> sdK --------------------------------
            if chunk >= self.first_state_chunk:
                cumprod_neg = [
                    s_cumprod[(lane % 4) * 2 + (i // 2) * 8 + i % 2, gs] * Float32(-1.0)
                    for i in range(16)
                ]
                cumprod_neg_vals = [cumprod_neg[(kv // 4) * 2 + (kv % 2)] for kv in range(n_vals)]
                ph = _wait(sig, ph, "spath_full", fence=False)
                ph = _wait(sig, ph, "dk_total_full")
                for half in cutlass.range_constexpr(d // 64):
                    sp = tmem_ld("16x256b", 8, self._rows(lane_base, half, spath_col))
                    dmr = tmem_ld("16x256b", 8, self._rows(lane_base, half, dvdk_col))
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(half == d // 64 - 1):
                        before_sync_and_arrive(sig, "dk_total_free")
                    for block in cutlass.range_constexpr(4):
                        dk_row = []
                        for j in cutlass.range_constexpr(4):
                            e = cutlass.const_expr(8 * block + 2 * j)
                            n0, n1 = fmul2(
                                sp[e], sp[e + 1], cumprod_neg_vals[e], cumprod_neg_vals[e + 1]
                            )
                            dk_row += [dmr[e] + n0, dmr[e + 1] + n1]
                        dk_pack = [
                            fp32_to_fp16(dk_row[2 * j], dk_row[2 * j + 1], dtype=io)
                            for j in range(4)
                        ]
                        stmatrix(dk_base + offs[half][block] * 2, dk_pack, trans=True)
            else:
                ph = _wait(sig, ph, "dk_total_full")
                for half in cutlass.range_constexpr(d // 64):
                    dmr = tmem_ld("16x256b", 8, self._rows(lane_base, half, dvdk_col))
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(half == d // 64 - 1):
                        before_sync_and_arrive(sig, "dk_total_free")
                    for block in cutlass.range_constexpr(4):
                        dk_pack = [
                            fp32_to_fp16(
                                dmr[8 * block + 2 * j], dmr[8 * block + 2 * j + 1], dtype=io
                            )
                            for j in range(4)
                        ]
                        stmatrix(dk_base + offs[half][block] * 2, dk_pack, trans=True)
            dkh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
            gh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        # ---- the item's final dS: d_initial_state, and hand the accumulator back -------------
        if n > 0:
            # Only the d_initial_state store loads TMEM before the next fenced wait.
            ph = _wait(sig, ph, "dstate_full", fence=self.use_dstate0)
            if cutlass.const_expr(self.use_dstate0):
                dst = params.items[item_row, WORK_ITEM_DSTATE_DST]
                if dst >= 0:
                    for i in cutlass.range_constexpr(d // 32):
                        vec = tmem_ld("32x32b", 32, lane0 + dstate_col + i * 32)
                        if row_valid:
                            self._store_state_row(
                                params.dstate0, dst, it.head, state_row, i * 32, vec
                            )
                    # Before dstate_scaled lets the MMA rewrite dS, or teardown frees TMEM.
                    cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(not self.use_dstate_in):
                before_sync_and_arrive(sig, "dstate_scaled")
        else:
            if cutlass.const_expr(self.use_dstate0):
                dst = params.items[item_row, WORK_ITEM_DSTATE_DST]
                if dst >= 0:
                    out = params.dstate0[dst, it.head, None, None]
                    for i in cutlass.range_constexpr(d // 32):
                        for kk in cutlass.range_constexpr(32):
                            if row_valid:
                                if cutlass.const_expr(self.use_dstate_in):
                                    val = params.dstate_in[
                                        it.batch, it.head, state_row, i * 32 + kk
                                    ]
                                    out[state_row, i * 32 + kk] = val.to(out.element_type)
                                else:
                                    out[state_row, i * 32 + kk] = Float32(0.0).to(out.element_type)
        return gate, k, dk, ph

    @cute.jit
    def _store_state_row(self, tensor, i0, head, row, col0, vals):
        """``tensor[i0, head, row, col0 : col0 + 32] = vals`` with 16-byte stores."""
        dst = tensor[i0, head, row, None]
        dtype = dst.element_type
        vw = 16 // (dtype.width // 8)
        for g in cutlass.range_constexpr(32 // vw):
            addr = (dst.iterator + (col0 + g * vw)).toint()
            frag = cute.make_rmem_tensor((vw,), dtype)
            for t in cutlass.range_constexpr(vw):
                frag[t] = vals[g * vw + t].to(dtype)
            cute.make_tensor(
                cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=16),
                cute.make_layout(vw),
            ).store(frag.load())

    # ---- epi (warp 15): dV, dQ, dK TMA stores ---------------------------------------------

    @role(warps=15, regs=32)
    @cute.jit
    def epi(
        self,
        params: BpropParams,
        *,
        dq: Consumer,
        dk: Consumer,
        dv: Consumer,
        sched: Consumer,
        work: WorkCursor,
    ):
        while work.valid:
            _, _, unit = work.item
            total = params.count[0]
            row = unit
            while row < total:
                it = Item(params, row)
                dq, dk, dv = self.store_item(params, it, dq, dk, dv)
                row, sched = self._next_row(sched)
            work.advance()
        bulk_wait_group(0, read=False)
        return dq, dk, dv, sched, work

    @cute.jit
    def store_item(self, params: BpropParams, it: Item, dq, dk, dv):
        """Store the dV, dQ and dK tiles of the item's written chunks."""
        stores = (params.dv_store, params.dq_store, params.dk_store)
        descs = sequence_descriptors(stores, it.batch_end, it.head, B_T, it.n_chunks > 0)
        dv_desc, dq_desc, dk_desc = descs
        for rev in cutlass.range(it.n_chunks, unroll=1):
            chunk = it.compute_end - 1 - rev
            store = chunk < it.write_end
            dvh = dv.wait()
            if store:
                self._store_tile_tma(params.dv_store, params.box_layout, it, chunk, "dV", dv_desc)
            dqh = dq.wait()
            if store:
                self._store_tile_tma(params.dq_store, params.box_layout, it, chunk, "dQ", dq_desc)
            dkh = dk.wait()
            if store:
                self._store_tile_tma(params.dk_store, params.box_layout, it, chunk, "dK", dk_desc)
            bulk_wait_group(2, read=True)
            dvh.release()
            bulk_wait_group(1, read=True)
            dqh.release()
            bulk_wait_group(0, read=True)
            dkh.release()
        return dq, dk, dv

    @cute.jit
    def _store_tile_tma(
        self, store: RaggedTma, box, it: Item, chunk, buf: cutlass.Constexpr[str], desc
    ):
        """TMA-store a ``[64, d]`` tile to chunk ``chunk``'s rows, dropping rows past the end."""
        start = it.batch_start + chunk * B_T
        valid = cutlass.min(it.batch_end - start, B_T)
        base = self._addr(self.buffer(buf))
        for kb in cutlass.range_constexpr(self.d // BOX):
            src = cute.make_tensor(self._ptr(base + kb * BOX_BYTES), box)
            ragged_store(store, src, start, valid, it.head, kb, B_T, BOX, desc)
        bulk_commit_group()

    # ---- kernel and entry point --------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        q_load, k_load, v_load, do_load, tma_ckpt, mCkpt,
        dq_store, dk_store, dv_store,
        gate, beta, dgate, dbeta, dstate_in, dstate0, items, count, scheduler,
        box_layout, ckpt_box_layout, q_group, k_group, v_group, scale, units, trace, bound,
    ):  # fmt: skip
        # The work table and checkpoints come from the preceding launches.
        cute.arch.griddepcontrol_wait()
        layouts = {name: cute.make_layout(shape) for name, shape in self._buffer_shapes.items()}
        self.begin_roles(work=bound, layouts=layouts)
        params = BpropParams(
            q_load=q_load,
            k_load=k_load,
            v_load=v_load,
            do_load=do_load,
            tma_ckpt=tma_ckpt,
            mCkpt=mCkpt,
            dq_store=dq_store,
            dk_store=dk_store,
            dv_store=dv_store,
            gate=gate,
            beta=beta,
            dgate=dgate,
            dbeta=dbeta,
            dstate_in=dstate_in,
            dstate0=dstate0,
            items=items,
            count=count,
            scheduler=scheduler,
            box_layout=box_layout,
            ckpt_box_layout=ckpt_box_layout,
            q_group=q_group,
            k_group=k_group,
            v_group=v_group,
            scale=scale,
            units=units,
        )
        self.run_roles(params, trace_buffer=trace)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (tokens, q heads, d)
        k: cute.Tensor,  # (tokens, k heads, d)
        v: cute.Tensor,  # (tokens, v heads, d)
        gate: cute.Tensor,  # (tokens, heads) natural-log gate
        beta: cute.Tensor,  # (tokens, heads) fp32
        do: cute.Tensor,  # (tokens, heads, d)
        checkpoints: cute.Tensor,  # (rows, heads, d, d)
        dq: cute.Tensor,  # (tokens, heads, d)
        dk: cute.Tensor,
        dv: cute.Tensor,
        dgate: cute.Tensor,  # (tokens, heads) fp32
        dbeta: cute.Tensor,  # (tokens, heads) fp32
        dstate_in: cute.Tensor | None,  # (sequences, heads, d, d)
        dstate0: cute.Tensor | None,  # (sequences, heads, d, d)
        items: cute.Tensor,  # (rows, 10) int32 work table
        count: cute.Tensor,  # (1,) int32
        scheduler: cute.Tensor,  # (>= 1,) int32, zeroed
        scale: Float32,
        workspace: cute.Tensor,
        max_ctas: Int32,
        trace: cute.Tensor | None,  # int64 globaltimer buffer when tracing is enabled
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        stream: cuda.CUstream,
    ):
        self.launch(
            q, k, v, gate, beta, do, checkpoints, dq, dk, dv, dgate, dbeta, dstate_in, dstate0,
            items, count, scheduler, tma_slots, scale, max_ctas, stream, trace,
        )  # fmt: skip

    @cute.jit
    def launch(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        do: cute.Tensor,
        checkpoints: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dgate: cute.Tensor,
        dbeta: cute.Tensor,
        dstate_in: cute.Tensor | None,
        dstate0: cute.Tensor | None,
        items: cute.Tensor,
        count: cute.Tensor,
        scheduler: cute.Tensor,
        tma_slots: cute.Tensor,
        scale: Float32,
        max_ctas: Int32,
        stream: cuda.CUstream,
        trace: cute.Tensor | None = None,
    ):
        """Launch the bprop; callable from another ``@cute.jit`` host, such as a fused launch
        bundle. It is not a programmatic dependent launch: the kernels before it (the
        recompute, then upstream's bprop prologue) release their dependents early, so under PDL
        the bprop's CTAs are placed while theirs still hold SMs, and a grid with idle CTAs
        (fewer table rows than SMs) then runs its busy CTAs on slow TPC positions (2x8192 H64
        d64: 660 vs 591 us; whole backward 905 vs 840 us). The kernel keeps its
        ``griddepcontrol.wait`` for callers that launch it with PDL."""
        io = self.io_dtype
        heads = Int32(gate.shape[1])
        box = sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, BOX), 1)
        box_layout = cute.select(box, mode=[0, 1])
        ckpt_box = sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (self.d, BOX), 1)
        ckpt_box_layout = cute.select(ckpt_box, mode=[0, 1])
        g2s = cpasync.CopyBulkTensorTileG2SOp(cute.nvgpu.tcgen05.CtaGroup.ONE)
        s2g = cpasync.CopyBulkTensorTileS2GOp()

        def token_view(t):  # (tokens, heads, d) -> (token, d, head)
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))

        def ragged(op, t, index):  # (token, d, head) boxes clipped at each sequence end
            slots = slot_region(tma_slots, index, B_T, heads)
            return make_ragged_tma(op, token_view(t), box_layout, B_T, BOX, slots)

        q_load, k_load, v_load, do_load = (ragged(g2s, t, i) for i, t in enumerate((q, k, v, do)))
        dq_store, dk_store, dv_store = (ragged(s2g, t, 4 + i) for i, t in enumerate((dq, dk, dv)))
        # (v, k, head, row) checkpoint view.
        ckpt = cute.make_tensor(
            checkpoints.iterator, cute.select(checkpoints.layout, mode=[2, 3, 1, 0])
        )
        tma_ckpt, mCkpt = cpasync.make_tiled_tma_atom(g2s, ckpt, ckpt_box_layout, (self.d, BOX))
        # Upstream's grid: one CTA per SM, each starting at its own table row.
        units = cutlass.max(max_ctas, 1)
        # One OneShot item per CTA, bound as (1, 1, units): with static tile extents the per-role
        # decode of the item index folds to the index itself, where (units, 1, 1) divides by the
        # runtime extent in every role (~130 SASS instructions).
        bound = self.work.bind((1, 1, units))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            q_load, k_load, v_load, do_load, tma_ckpt, mCkpt,
            dq_store, dk_store, dv_store,
            gate, beta, dgate, dbeta, dstate_in, dstate0, items, count, scheduler,
            box_layout, ckpt_box_layout,
            heads // Int32(q.shape[1]), heads // Int32(k.shape[1]), heads // Int32(v.shape[1]),
            scale, units, trace, bound,
        ).launch(**self.launch_config(bound), stream=stream)  # fmt: skip


def bundle_hook(
    io_dtype,
    d_k: int,
    d_v: int,
    use_initial_state: bool,
    use_dstate_in: bool,
    use_dstate0: bool,
    log_gate: bool,
    num_sm: int,
) -> tuple[GdnBprop, int]:
    """``(op, num_sm)`` constexpr for launch bundles that run the bprop inside their own
    ``@cute.jit`` host: ``op.launch(..., Int32(num_sm), stream)``."""
    if not log_gate:
        raise ValueError("the GDN bprop takes natural-log gates")
    if d_k != d_v:
        raise ValueError(f"the GDN bprop needs d_k == d_v, got {d_k}, {d_v}")
    name = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16"}[io_dtype]
    return GdnBprop(name, d_k, use_initial_state, use_dstate_in, use_dstate0), num_sm


@jit_cache
def compile_gdn_bprop(
    op: GdnBprop, gate_dtype: str, dstate_in_dtype: str | None, dstate0_dtype: str | None
):
    """Compile for dynamic tokens, heads, sequences, work rows and checkpoint rows."""
    io, d, i32 = op.io_dtype, op.d, cutlass.Int32
    tokens, q_heads, k_heads, v_heads, heads = (cute.sym_int() for _ in range(5))
    ckpt_rows, items, seqs, dst_seqs = (cute.sym_int() for _ in range(4))

    def tile(h):
        return make_fake_strided_tensor(io, (tokens, h, d), stride_divisibility=8)

    def scalar(dtype):
        return make_fake_strided_tensor(dtype, (tokens, heads), contiguous_dim=None)

    def state(dtype_name, rows):
        if dtype_name is None:
            return None
        dtype = getattr(cutlass, dtype_name)
        return make_fake_strided_tensor(
            dtype, (rows, heads, d, d), stride_divisibility=tma_divisibility(dtype)
        )

    return compile_tvm_ffi(
        op,
        tile(q_heads),
        tile(k_heads),
        tile(v_heads),
        scalar(getattr(cutlass, gate_dtype)),
        scalar(Float32),
        tile(heads),
        make_fake_strided_tensor(io, (ckpt_rows, heads, d, d), stride_divisibility=8),
        tile(heads),
        tile(heads),
        tile(heads),
        scalar(Float32),
        scalar(Float32),
        state(dstate_in_dtype, seqs),
        state(dstate0_dtype, dst_seqs),
        make_fake_strided_tensor(i32, (items, WORK_ITEM_FIELDS), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        Float32(1.0),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        Int32(1),
        make_fake_strided_tensor(cutlass.Int64, (cute.sym_int(),))
        if op.trace_backend == "globaltimer"
        else None,
        fake_tma_slots(),
        name=op.get_name(),
        opt_level=2,  # upstream's level (the chain bundle compiles the bprop at 2 as well)
    )


@functools.lru_cache
def _compiled(dtype_name, d, use_initial_state, use_dstate_in, use_dstate0, *dtypes):
    """``(op, compiled)`` per static configuration: the per-call ``jit_cache`` key of a
    warpdrive op costs ~150 us of host time, which the uncut driver would pay per backward."""
    op = GdnBprop(dtype_name, d, use_initial_state, use_dstate_in, use_dstate0)
    return op, compile_gdn_bprop(op, *dtypes)


def _dtype_name(tensor, names):
    return None if tensor is None else names[tensor.dtype]


def gdn_bprop(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    checkpoints: torch.Tensor,
    items: torch.Tensor,
    count: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dgate: torch.Tensor,
    dbeta: torch.Tensor,
    *,
    scale: float,
    use_initial_state: bool,
    dstate_in: torch.Tensor | None = None,
    dstate0: torch.Tensor | None = None,
    scheduler: torch.Tensor | None = None,
    num_sm: int | None = None,
    op: GdnBprop | None = None,
    trace: torch.Tensor | None = None,
    tma_slots: torch.Tensor | None = None,
) -> None:
    """Fill ``dq, dk, dv`` (at the gate's head count), fp32 ``dgate``/``dbeta`` and optionally
    ``dstate0`` for packed ``q``/``k``/``v``/``do`` ``[T, H, D]``, a natural-log ``gate [T, H]``,
    fp32 ``beta``, ``gdn_recompute`` checkpoints and an int32 work table (see the module
    docstring). ``scheduler`` (int32, zeroed) is allocated when omitted; ``num_sm`` defaults to
    the device's SM count. ``op`` (a ``GdnBprop`` with ``enable_trace("globaltimer")``) and its
    ``trace`` buffer capture a role timeline. ``tma_slots`` (``ragged.new_tma_slots``, at
    least ``op.tma_slot_rows(heads)`` rows) is allocated when omitted."""
    if q.dtype not in _IO_DTYPES or any(t.dtype != q.dtype for t in (k, v, do, checkpoints)):
        raise TypeError("q, k, v, do and checkpoints must share a 16-bit dtype")
    if beta.dtype != torch.float32 or dgate.dtype != torch.float32 or dbeta.dtype != torch.float32:
        raise TypeError("beta, dgate and dbeta must be fp32")
    if q.shape[-1] != v.shape[-1]:
        raise ValueError("the bprop needs d_k == d_v")
    if scheduler is None:
        scheduler = torch.zeros(2, dtype=torch.int32, device=q.device)
    dtypes = (
        _SCALAR_DTYPES[gate.dtype],
        _dtype_name(dstate_in, _STATE_DTYPES),
        _dtype_name(dstate0, _STATE_DTYPES),
    )
    if op is None:
        flags = (use_initial_state, dstate_in is not None, dstate0 is not None)
        op, fn = _compiled(_IO_DTYPES[q.dtype], q.shape[-1], *flags, *dtypes)
    else:
        fn = compile_gdn_bprop(op, *dtypes)
    workspace, sms = op.work.launch_args(q.device)
    if tma_slots is None:
        tma_slots = new_tma_slots(op.tma_slot_rows(gate.shape[1]), q.device)
    fn(
        q, k, v, gate, beta, do, checkpoints, dq, dk, dv, dgate, dbeta, dstate_in, dstate0,
        items, count, scheduler, float(scale), workspace, sms if num_sm is None else num_sm, trace,
        tma_slots,
    )  # fmt: skip
