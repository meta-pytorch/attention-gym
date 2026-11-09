"""Single-token swap-AB decode on SM100, following FA4's ``flash_fwd_decode_sm100.py``.

``S^T = K Q^T`` and ``O^T = V^T P^T`` put 128 KV tokens on MMA M and the query heads of
one KV head on N. Each ``(split, kv_head, batch)`` item reduces a contiguous range of KV
blocks to normalized fp32 O and natural-log LSE partials. ``fa4_decode`` combines them
with FA4's combine kernel.

Supports bf16 Q/K/V and dense contiguous KV, optional ``seqused_k`` lengths and per-sequence
split counts. ``tile_hq = next_pow2(heads / kv_heads)`` is at most 32. Defaults are 128 head
dimensions and 256-token KV blocks. Head dimensions, block size, K step, S/P ring depths
and role register budgets are constructor options; see docs/warpdrive/configuration.md.

KV rows past a sequence's length must be finite: P is zero there, but V is not zero-filled
at that boundary. No paging, varlen Q, masks or score mods.

Roles (16 warps, FA4's placement):

    warp 0        mma_qk     S^T(j) = K(j) Q^T, two 128-token subtiles per block
    warp 1        mma_pv     O^T(slot j % 2) += V^T(j) P^T(j)
    warp 2        load       K/V TMA loads, then this split's LSE and 1/L
    warp 3        q_store    Q TMA load, then the O partial's TMA store
    warps 4-11    softmax[0], softmax[1]: even and odd KV blocks
    warps 12-15   correction rescales O, merges the two O slots, stages O

mbarrier rings carry async handoffs. Named-barrier channels (``sync="named"``) carry
thread mailboxes; three ``NamedBarrier`` flags coordinate split statistics:

    q        Ring(1)  q_store -TMA-> mma_qk
    k, v     Ring     load -TMA-> mma_qk / mma_pv; K stays two blocks ahead of V.
                      SharedKVDecode uses one exclusive-consumer kv ring and the
                      QKIssued/PVIssued named barriers (shared_kv=True in wrappers).
    s        Ring(4)  mma_qk -UMMA-> softmax[0|1], round-robin: softmax[i] takes blocks i, i+2, ...
    p        Ring(4)  softmax[0|1] -> mma_pv, round-robin; P in SMEM, the PV MMA's B operand
                      (``s_stages`` and ``p_stages``)
    o        Ring(2)  mma_pv -UMMA-> correction; slot j % 2 accumulates blocks j, j+2, ...
    row_max  named    softmax[0|1] -> correction, round-robin; the running max M(j) in sRowMax
    row_sum  named    softmax[0|1] -> correction, round-robin; per-thread sums in TMEM L
    max_done, sum_done, stats_ready  one-shot NamedBarrier flags: correction <-> load's stats
    epi      Ring(1)  correction -> q_store (sO)
    mufu     Turnstile  softmax[0] <-> softmax[1] around the row_max exchange (FA4's MaxTurn)

softmax[j % 2] atomically merges its block max into M(j - 1), then waits for all exchange
threads with ``commit(wait=True)`` before reading M(j). The turnstile serializes warpgroups;
arrival counts alone would let both enter at once.

The first PV overwrites its O slot, so no TMEM zero-fill must precede MMA. For a one-block
split, the correction tail treats unwritten O/L slot 1 as zero. Empty splits run the same
tail with zero sums and O. Dropped splits still initialize barriers and TMEM before roles
see no work.

Mailboxes wait on pre-seeded empty barriers before signalling. Done flags arrive without
waiting: a OneShot CTA completes each flag at most once. ``SharedKVDecode`` implements
FA4's shared-ring protocol; the default keeps K/V rings separate. Benchmark comparisons
are in docs/warpdrive/examples.md.
"""

import math
from dataclasses import dataclass
from functools import partial

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.utils import get_smem_capacity_in_bytes
from flash_attn.cute.pack_gqa import pack_gqa_layout, select_modes
from quack import copy_utils

from attn_gym.warpdrive import (
    Consumer,
    Instance,
    Local,
    NamedBarrier,
    OneShot,
    PdlWait,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    Sharing,
    SmemPlan,
    StageView,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Turn,
    Turnstile,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkMap,
    WorkStream,
    bulk_commit_group,
    bulk_wait_group,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.compat import FastDivmodDivisor
from attn_gym.warpdrive.examples.tma_tiles import tma_partition_tiles

TILE_M = 128  # MMA M: KV tokens of one S subtile, head dims of one O tile (one lane per row)
MMA_N_MIN = 8
TILE_KS = (16, 32, 64)  # MMA K step: 32 B to 128 B K-major rows, up to the SW128 atom
O_STAGES = 2  # even / odd blocks: one O slot per softmax warpgroup, merged by the correction
WARPS_PER_WG = 4
WG_SIZE = 128
LN2 = math.log(2.0)

exp2_fast = partial(cute.math.exp2, fastmath=True)
warp_reduce_max = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_atomic_max = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")


@dataclass(frozen=True)
class SplitKV(WorkMap):
    """Map ``(num_splits, kv_heads, batch, table)`` to ``(split, kv_head, batch, splits)`` items.

    Without ``per_sequence``, each sequence has ``num_splits`` and split is the fastest axis.
    With it, indices are split-major and int32 ``table[b]`` caps sequence b's split count;
    later splits are dropped. A preceding kernel may write the table: use ``ready=PdlWait()``.
    The fourth item word is the count used to divide that sequence's KV range.
    """

    per_sequence: bool = False
    words = 4
    skips = True

    def bind(self, space):
        splits, heads, batch, _table = space
        # These divisors depend only on launch shapes, not the PDL-produced split table.
        major = heads * batch if self.per_sequence else splits
        return (*space, FastDivmodDivisor(major), FastDivmodDivisor(heads))

    def count64(self, space):
        splits, heads, batch, *_ = space
        return Int64(splits) * heads * batch

    def decode(self, index, space, state):
        splits, _, _, table, major_divisor, head_divisor = space
        if self.per_sequence:
            split, rest = divmod(index, major_divisor)
            b, head = divmod(rest, head_divisor)
            count = table[b]
            return split < count, (split, head, b, count), state
        bh, split = divmod(index, major_divisor)
        b, head = divmod(bh, head_divisor)
        return True, (split, head, b, splits), state

    def get_key(self) -> str:
        return "splitkv_seq" if self.per_sequence else "splitkv"


@dataclass(frozen=True)
class DecodeParams:
    """Layouts, buffers and runtime sizes shared by the decode roles."""

    mma_qk: cute.TiledMma
    mma_pv: cute.TiledMma
    tma_q: cute.CopyAtom
    tma_k: cute.CopyAtom
    tma_v: cute.CopyAtom
    tma_o: cute.CopyAtom
    mQ: cute.Tensor  # TMA coordinates: (qpk, d, (h_kv, b))
    mK: cute.Tensor  # (s, d, (h_kv, b))
    mV: cute.Tensor  # V^T: (d, s, (h_kv, b))
    mO: cute.Tensor  # (d, qpk, (h_kv, b), split)
    mLSE: cute.Tensor  # (split, b, h)
    mSeqUsedK: cute.Tensor  # (b,)
    sQ: cute.Tensor
    sK: cute.Tensor
    sV: cute.Tensor
    sP: cute.Tensor
    sRowMax: cute.Tensor  # running max per head; 1/L at the end
    sRowSums: cute.Tensor  # (warp, head): the correction warps' row sums for the stats warp
    scale_log2: Float32
    num_heads_kv: Int32
    qhead_per_kvhead: Int32


@dataclass(frozen=True)
class SplitRange:
    """One work item's KV blocks ``[n_block_min, n_block_min + n_blocks)`` and its rows."""

    n_block_min: Int32
    n_blocks: Int32
    seqlen_k: Int32
    split: Int32
    kv_head: Int32
    batch: Int32
    bh: Int32  # batch * kv_heads + kv_head: the (h_kv, b) coordinate


def gemm_view(mX, order, head_idx=None, qhead_per_kvhead=None, num_heads_kv=None):
    """Permute ``mX`` to ``order`` and group modes 2/3 (KV head, batch).

    With ``head_idx``, first split that mode into (qhead_per_kvhead, num_heads_kv).
    """
    if head_idx is not None:
        mX = cute.flatten(pack_gqa_layout(mX, qhead_per_kvhead, num_heads_kv, head_idx))
    return cute.group_modes(select_modes(mX, order), 2, 4)


class SwapABDecode(WarpSpecialized):
    # SharedKVDecode needs different channel parameters on the K/V roles.
    shared_kv = False

    def __init__(
        self,
        tile_hq: int,
        kv_stages: tuple[int, int] | None = None,
        per_sequence=False,
        *,
        tile_n: int = 256,
        tile_k: int = 64,
        s_stages: int = 4,
        p_stages: int = 4,
        head_dim: int = 128,
        head_dim_v: int = 128,
        roles: dict[str, RoleConfig] | None = None,
    ):
        """Configure split-KV tiles, channels and roles.

        ``tile_hq`` is a power-of-two query-head tile up to 32. ``tile_n`` is a power-of-two
        multiple of 128 KV tokens; ``tile_k`` is the MMA K step, one of ``TILE_KS``.
        S (TMEM) and P (SMEM) depths must be even for the two softmax warpgroups.
        Q/K ``head_dim`` must be a multiple of ``tile_k``; V/O ``head_dim_v`` of 128.
        ``roles`` overrides placement and budgets by name. Defaults use FA4's 64/120/208
        register split for 32-head tiles, with no register adjustment otherwise.
        TMEM/SMEM overflow raises at construction or first collection.
        """
        if tile_hq not in (1, 2, 4, 8, 16, 32):
            raise ValueError(f"tile_hq must be a power of two up to 32, got {tile_hq}")
        s_subtiles = tile_n // TILE_M
        if tile_n % TILE_M or s_subtiles & (s_subtiles - 1):
            raise ValueError(f"tile_n must be a power-of-two multiple of {TILE_M}, got {tile_n}")
        if tile_hq * s_subtiles > 128:
            raise ValueError(
                f"tile_hq x tile_n / {TILE_M} = {tile_hq * s_subtiles} exceeds the 128-column "
                "tcgen05.ld repetition that reads one S block"
            )
        if tile_k not in TILE_KS:
            raise ValueError(f"tile_k must be one of {TILE_KS}, got {tile_k}")
        if head_dim <= 0 or head_dim % tile_k:
            raise ValueError(
                f"head_dim must be a positive multiple of tile_k={tile_k}, got {head_dim}"
            )
        if head_dim_v <= 0 or head_dim_v % TILE_M:
            raise ValueError(
                f"head_dim_v must be a positive multiple of {TILE_M}, got {head_dim_v}"
            )
        for name, stages in (("s_stages", s_stages), ("p_stages", p_stages)):
            if stages < 2 or stages % 2:
                raise ValueError(
                    f"{name} must be an even number >= 2 (two softmax warpgroups), got {stages}"
                )
        shared_kv = self.shared_kv
        self.tile_hq = tile_hq
        self.tile_hq_mma = max(MMA_N_MIN, tile_hq)
        self.per_sequence = per_sequence
        self.tile_n, self.tile_k = tile_n, tile_k
        self.s_stages, self.p_stages = s_stages, p_stages
        self.head_dim, self.head_dim_v = head_dim, head_dim_v
        self.s_subtiles = s_subtiles
        self.k_tiles = head_dim // tile_k  # QK K steps
        self.pv_k_tiles = tile_n // tile_k  # PV K steps
        self.d_tiles = head_dim_v // TILE_M  # O tiles
        self.kv_stage_bytes = TILE_M * tile_k * 2  # a K tile and a V^T tile have the same bytes
        self.kv_stages = kv_stages or self.default_kv_stages()
        if min(self.kv_stages) < 1:
            raise ValueError(f"kv_stages must be positive, got {self.kv_stages}")
        # FA4 adjusts registers only for 32-head tiles: 64 + 2 x 120 + 208 = 512 per thread
        # position, 64K / 128 threads.
        defaults = {}
        if tile_hq > 16:
            defaults = {
                "mma_qk": RoleConfig(0, regs=64),
                "mma_pv": RoleConfig(1, regs=64),
                "load": RoleConfig(2, regs=64),
                "q_store": RoleConfig(3, regs=64),
                "softmax": RoleConfig((range(4, 8), range(8, 12)), regs=120),
                "correction": RoleConfig(range(12, 16), regs=208),
            }
        self.roles = {**defaults, **(roles or {})}
        # The stats flags below count one load warp and one correction warpgroup.
        for name, warps in (("load", 1), ("correction", WARPS_PER_WG)):
            config = self.roles.get(name)
            if config is not None and [len(r) for r in config.warps] != [warps]:
                raise ValueError(
                    f"roles[{name!r}] must be one instance of {warps} warp(s), got {config.warps}"
                )
        n, bf16 = self.tile_hq_mma, cutlass.BFloat16
        n_k, n_v = self.kv_stages

        self.smem_plan = SmemPlan()
        p_buf = self.smem_plan.buffer("P", bf16, align=1024)
        self.smem_plan.buffer("Q", bf16, align=1024)
        k_buf = self.smem_plan.buffer("K", bf16, align=1024)
        # Shared: each stage holds a K tile or a V^T tile, read with its own MMA layout.
        v_buf = self.smem_plan.buffer("V", bf16, align=1024, overlaps=k_buf if shared_kv else None)
        # O reuses K after the final O commit: S full -> P -> PV -> O full completes all
        # QK reads and, with shared K/V, all PV reads.
        o_buf = self.smem_plan.buffer("O", Float32, align=1024, overlaps=k_buf)

        self.q = Ring(1, Tma(n * head_dim * 2), Umma)
        if shared_kv:
            self.kv = Ring(
                n_k + n_v,
                Tma(self.kv_stage_bytes),
                Umma,
                consumer_roles=("mma_qk", "mma_pv"),
                consumer_sharing=Sharing.EXCLUSIVE,
                buffer={"K": k_buf, "V": v_buf},
            )
            # FA4's QKIssued/PVIssued: each MMA warp arrives on one after issuing a block and
            # syncs on the other before waiting for its next, so both parity waits are exact.
            self.qk_issued = NamedBarrier(threads=64)
            self.pv_issued = NamedBarrier(threads=64)
        else:
            self.k = Ring(n_k, Tma(self.kv_stage_bytes), Umma, buffer=k_buf)
            self.v = Ring(n_v, Tma(self.kv_stage_bytes), Umma, buffer=v_buf)
        rr = Sharing.ROUND_ROBIN  # softmax[0|1] take alternate KV blocks
        self.s = Ring(s_stages, Umma, Threads, consumer_sharing=rr)
        # The derived smem_async fence orders P's SMEM stores before the PV MMA reads them.
        self.p = Ring(p_stages, Threads, Umma, producer_sharing=rr, buffer=p_buf)
        self.o = Ring(O_STAGES, Umma, Threads, release_fence="tmem_store")
        self.row_max = Ring(
            1, Threads, Threads, sync="named", producer_sharing=rr, turnstile="mufu"
        )
        self.row_sum = Ring(
            O_STAGES,
            Threads,
            Threads,
            sync="named",
            producer_sharing=rr,
            commit_fence="tmem_store",
        )
        # OneShot completes each stats flag at most once. The signaller can arrive without
        # waiting because it cannot reach another generation.
        stats_threads = WG_SIZE + 32
        self.max_done = NamedBarrier(threads=stats_threads)  # M is final in sRowMax
        self.sum_done = NamedBarrier(threads=stats_threads)  # row sums are in sRowSums
        self.stats_ready = NamedBarrier(threads=stats_threads)  # 1/L is in sRowMax
        self.epi = Ring(1, Threads, Threads, commit_fence="smem_async", buffer=o_buf)
        self.mufu = Turnstile()
        self.tmem_plan = (
            TmemPlan()
            .region("S", s_subtiles * n, s_stages)
            .region("L", n, O_STAGES)
            .region("O", self.d_tiles * n, O_STAGES)
        )
        _ = self.tmem_plan.alloc_cols  # raises when the plan exceeds 512 columns
        mapping = SplitKV(per_sequence)
        self.work = WorkStream(
            mapping=mapping, schedule=OneShot(), ready=PdlWait() if per_sequence else None
        )

    def default_kv_stages(self) -> tuple[int, int]:
        """Reserve one SMEM stage below capacity and split the rest about 7:5 between K/V.

        K runs two blocks ahead; too few V stages can starve PV. The split is a heuristic,
        not a workload-specific optimum; see docs/warpdrive/examples.md for benchmarks.
        """
        n = self.tile_hq_mma
        # P, Q, barriers and stats; the O staging tile shares the K ring's bytes.
        fixed = self.p_stages * n * self.tile_n * 2 + n * self.head_dim * 2 + 1024
        total = (get_smem_capacity_in_bytes("sm_100") - fixed) // self.kv_stage_bytes - 1
        if total < 2:
            raise ValueError(
                f"tile_hq={self.tile_hq}, tile_n={self.tile_n}, p_stages={self.p_stages} leave "
                f"room for {total} K/V stages; shrink the P ring or the KV block"
            )
        n_v = max(total * 5 // 12, min(3, total - 1))
        return total - n_v, n_v

    def get_name(self) -> str:
        n_k, n_v = self.kv_stages
        shape = ""
        if (self.tile_n, self.tile_k, self.head_dim, self.head_dim_v) != (256, 64, 128, 128):
            shape = f"_n{self.tile_n}k{self.tile_k}d{self.head_dim}dv{self.head_dim_v}"
        if (self.s_stages, self.p_stages) != (4, 4):
            shape += f"_s{self.s_stages}p{self.p_stages}"
        if self.shared_kv:
            shape += "_shared"
        return (
            f"warpdrive_fa4_decode_swapab_hq{self.tile_hq}_k{n_k}v{n_v}{shape}_"
            f"{self.work.get_key()}{self.trace_key()}"
        )

    def split_range(self, params: DecodeParams, item) -> SplitRange:
        """Return a split's contiguous KV block range; empty sequences have zero blocks."""
        split, kv_head, batch, splits = item
        seqlen_k = params.mSeqUsedK[batch]
        n_total = (seqlen_k + self.tile_n - 1) // self.tile_n
        per_split = (n_total + splits - 1) // splits
        n_min = split * per_split
        n_max = cutlass.min(n_min + per_split, n_total)
        return SplitRange(
            n_min,
            cutlass.max(n_max - n_min, 0),
            seqlen_k,
            split,
            kv_head,
            batch,
            batch * params.num_heads_kv + kv_head,
        )

    def tmem_tensors(self, params: DecodeParams, tmem):
        """(S, L, O) TMEM tensors: ((128, N), 1, 1, subtile, stage), ((128, N), 1, 1, slot) and
        ((128, N), 1, 1, d tile, slot)."""
        thr_qk = params.mma_qk.get_slice(0)
        acc_shape = thr_qk.partition_shape_C((TILE_M, self.tile_hq_mma))
        s_frag = thr_qk.make_fragment_C(
            cute.append(cute.append(acc_shape, self.s_subtiles), self.s_stages)
        )
        l_frag = thr_qk.make_fragment_C(cute.append(acc_shape, 2))
        o_frag = params.mma_pv.get_slice(0).make_fragment_C(
            cute.append(cute.append(acc_shape, self.d_tiles), O_STAGES)
        )
        return (
            cute.make_tensor(tmem.ptr("S", 0), s_frag.layout),
            cute.make_tensor(tmem.ptr("L", 0), l_frag.layout),
            cute.make_tensor(tmem.ptr("O", 0), o_frag.layout),
        )

    def tile_row(self, me: Local, role_name: str):
        """Return this thread's TMEM row; a 128-row subtile requires four warps."""
        if me.warps != WARPS_PER_WG:
            raise ProtocolError(
                f"{role_name}: a {TILE_M}-row TMEM subtile is one row per thread of 4 warps, got "
                f"{me.warps} warps"
            )
        return me.tmem_row

    def vec_copies(self, tLtL, tidx):
        """TMEM <-> register copies of one N-column accumulator row per thread."""
        tLtL_0 = tLtL[(None, None), 0, 0, 0]
        rep = tcgen05.Repetition(self.tile_hq)
        t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(rep, tcgen05.Pack.NONE), Float32), tLtL_0
        )
        r2t = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.St32x32bOp(rep, tcgen05.Unpack.NONE), Float32), tLtL_0
        )
        thr_t2r, thr_r2t = t2r.get_slice(tidx), r2t.get_slice(tidx)
        vec_layout = thr_t2r.partition_D(tLtL_0).layout
        return t2r, r2t, thr_t2r, thr_r2t, vec_layout

    @role(warps=3)
    @cute.jit
    def q_store(self, params: DecodeParams, *, q: Producer, epi: Consumer, work: WorkCursor):
        thr_qk = params.mma_qk.get_slice(0)
        n = self.tile_hq_mma
        while work.valid:
            r = self.split_range(params, work.item)
            # Each sQ stage holds tile_k dimensions for all query heads of this KV head.
            if r.n_blocks > 0:
                gQ = cute.local_tile(params.mQ, (n, self.tile_k), (0, None, r.bh))
                load_q, _, _ = copy_utils.tma_get_copy_fn(
                    params.tma_q, 0, cute.make_layout(1), thr_qk.partition_B(gQ), params.sQ
                )
                with q.produce() as h:
                    for dk in cutlass.range_constexpr(self.k_tiles):
                        load_q(src_idx=dk, dst_idx=dk, tma_bar_ptr=h.barrier)
            with epi.consume() as h:
                sO = h.data
                for dm in cutlass.range_constexpr(self.d_tiles):
                    gO = cute.local_tile(params.mO, (TILE_M, n), (dm, 0, r.bh, r.split))
                    store_o, _, _ = copy_utils.tma_get_copy_fn(
                        params.tma_o,
                        0,
                        cute.make_layout(1),
                        sO[None, None, dm],
                        gO,
                        single_stage=True,
                    )
                    store_o()
                bulk_commit_group()
                bulk_wait_group(0, read=True)  # sO overlaps the K ring: read before release
            work.advance()
        return q, epi, work

    @role(warps=2)
    @cute.jit
    def load(
        self,
        params: DecodeParams,
        *,
        k: Producer,
        v: Producer,
        work: WorkCursor,
    ):
        thr_qk, thr_pv, tKsK, tVsV = self.load_setup(params)
        k = k.views(K=StageView(tKsK, axis=1))
        v = v.views(V=StageView(tVsV, axis=1))
        lookahead = 2  # FA4's issue order: K(s) with V(s - 2)
        while work.valid:
            r = self.split_range(params, work.item)
            for s in cutlass.range(lookahead + r.n_blocks, unroll=1):
                if s < r.n_blocks:
                    k = self.fill(k, "K", params.tma_k, self.k_sources(params, thr_qk, r, s))
                if s >= lookahead:
                    j = s - lookahead
                    v = self.fill(v, "V", params.tma_v, self.v_sources(params, thr_pv, r, j))
            self.split_stats(params, r)
            work.advance()
        return k, v, work

    def load_setup(self, params: DecodeParams):
        """Prefetch descriptors and return MMA slices plus the K/V SMEM partitions.

        These partitions do not depend on the global tile.
        """
        for atom in (params.tma_q, params.tma_k, params.tma_v, params.tma_o):
            cpasync.prefetch_descriptor(atom)
        thr_qk, thr_pv = params.mma_qk.get_slice(0), params.mma_pv.get_slice(0)
        tiler = (TILE_M, self.tile_k)
        gK0 = cute.zipped_divide(
            cute.local_tile(params.mK, (TILE_M, self.head_dim), (0, 0, 0)), tiler
        )[(None, None), (0, None)]
        tKsK, _ = tma_partition_tiles(params.tma_k, params.sK, thr_qk.partition_A(gK0))
        gV0 = cute.zipped_divide(
            cute.local_tile(params.mV, (self.head_dim_v, TILE_M), (0, 0, 0)), tiler
        )[(None, None), (None, 0)]
        tVsV, _ = tma_partition_tiles(params.tma_v, params.sV, thr_pv.partition_A(gV0))
        return thr_qk, thr_pv, tKsK, tVsV

    def k_sources(self, params: DecodeParams, thr_qk, r: SplitRange, s):
        """Block s's K tiles, one per stage: subtile-major, then K step."""
        tiler = (TILE_M, self.tile_k)
        for sm in range(self.s_subtiles):
            unit = (s + r.n_block_min) * self.s_subtiles + sm
            gK = cute.local_tile(params.mK, (TILE_M, self.head_dim), (unit, 0, r.bh))
            gK_sub = cute.zipped_divide(gK, tiler)[(None, None), (0, None)]
            _, tKgK = tma_partition_tiles(params.tma_k, params.sK, thr_qk.partition_A(gK_sub))
            for dk in range(self.k_tiles):
                yield tKgK[None, dk]

    def v_sources(self, params: DecodeParams, thr_pv, r: SplitRange, j):
        """Block j's V^T tiles, one per stage: PV K step-major, then head-dim tile."""
        tiler = (TILE_M, self.tile_k)
        block = j + r.n_block_min
        for sk in range(self.pv_k_tiles):
            unit = block * self.s_subtiles + sk // (TILE_M // self.tile_k)
            gVt = cute.zipped_divide(
                cute.local_tile(params.mV, (self.head_dim_v, TILE_M), (0, unit, r.bh)),
                tiler,
            )
            gVt_sub = gVt[(None, None), (None, sk % (TILE_M // self.tile_k))]
            _, tVgV = tma_partition_tiles(params.tma_v, params.sV, thr_pv.partition_A(gVt_sub))
            for dm in range(self.d_tiles):
                yield tVgV[None, dm]

    def fill(self, end, name: str, atom, sources, probe: bool = False):
        """Copy each source into successive stages of ``end``'s buffer ``name``.

        With ``probe``, acquire tokens are probed after the previous commit, except for
        the first acquire. Tokens cannot cross dynamic regions between groups.
        Shared K/V copies carry alias annotations because their stages overlap.
        """
        access = {"write": True, "by": "tma"} if self.shared_kv else {}
        token, sources = None, iter(sources)
        src = next(sources, None)
        while src is not None:
            with end.produce(token=token) as h:
                cute.copy(atom, src, h.view(name, **access), tma_bar_ptr=h.barrier)
            src = next(sources, None)
            token = end.try_acquire() if probe and src is not None else None
        return end

    @cute.jit
    def split_stats(self, params: DecodeParams, r: SplitRange):
        """Compute split LSE and 1/L from the final max and per-warp row sums.

        Return 1/L to correction through sRowMax.
        """
        lane = cute.arch.lane_idx()
        owns_head = lane < self.tile_hq
        row_max = -Float32.inf
        self.max_done.sync()
        if owns_head:
            row_max = params.sRowMax[lane]
        self.sum_done.sync()
        if owns_head:
            sums = params.sRowSums
            row_sum = (sums[0, lane] + sums[1, lane]) + (sums[2, lane] + sums[3, lane])
            inv_sum = Float32(0.0)
            if row_sum > Float32(0.0):
                inv_sum = cute.arch.rcp_approx(row_sum)
            params.sRowMax[lane] = inv_sum
            if lane < params.qhead_per_kvhead:
                lse = -Float32.inf
                if row_sum > Float32(0.0):
                    lse = (row_max + cute.math.log2(row_sum, fastmath=True)) * LN2
                head = r.kv_head * params.qhead_per_kvhead + lane
                params.mLSE[r.split, r.batch, head] = lse
        self.stats_ready.arrive()

    @role(warps=0)
    @cute.jit
    def mma_qk(
        self,
        params: DecodeParams,
        *,
        q: Consumer,
        k: Consumer,
        s: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        mma = params.mma_qk
        atom = cute.make_mma_atom(mma.op)
        tCrQ = mma.make_fragment_B(params.sQ)
        k = k.views(K=StageView(mma.make_fragment_A(params.sK), axis=3))
        tStS, _, _ = self.tmem_tensors(params, tmem)
        qk_k_blocks = self.tile_k // cute.size(mma.shape_mnk, mode=[2])
        while work.valid:
            r = self.split_range(params, work.item)
            if r.n_blocks > 0:
                qh = q.wait()
                for _ in cutlass.range(r.n_blocks, unroll=1):
                    sh = s.acquire()
                    for sm in cutlass.range_constexpr(self.s_subtiles):
                        atom.set(tcgen05.Field.ACCUMULATE, False)
                        acc = tStS[None, None, None, sm, sh.index]
                        for dk in cutlass.range_constexpr(self.k_tiles):
                            with k.consume() as kh:
                                for i in cutlass.range_constexpr(qk_k_blocks):
                                    cute.gemm(
                                        atom,
                                        acc,
                                        kh.view("K", None, None, i),
                                        tCrQ[None, None, i, dk],
                                        acc,
                                    )
                                    atom.set(tcgen05.Field.ACCUMULATE, True)
                    sh.commit()
                qh.release()
            work.advance()
        return q, k, s, work

    @role(warps=1)
    @cute.jit
    def mma_pv(
        self,
        params: DecodeParams,
        *,
        v: Consumer,
        p: Consumer,
        o: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        mma = params.mma_pv
        atom = cute.make_mma_atom(mma.op)
        v = v.views(V=StageView(mma.make_fragment_A(params.sV), axis=3))
        p = p.views(P=StageView(mma.make_fragment_B(params.sP), axis=3))
        _, _, tOtO = self.tmem_tensors(params, tmem)
        pv_k_blocks = self.tile_k // cute.size(mma.shape_mnk, mode=[2])
        while work.valid:
            r = self.split_range(params, work.item)
            for j in cutlass.range(r.n_blocks, unroll=1):
                ph = p.wait()
                # Free once the correction has rescaled what PV(j - 2) left there.
                oh = o.acquire()
                for dk in cutlass.range_constexpr(self.pv_k_tiles):
                    for dm in cutlass.range_constexpr(self.d_tiles):
                        acc = tOtO[None, None, None, dm, oh.index]
                        with v.consume() as vh:
                            for i in cutlass.range_constexpr(pv_k_blocks):
                                # Blocks 0 and 1 start their slot: no zero-fill to order first.
                                if cutlass.const_expr(dk == 0 and i == 0):
                                    atom.set(tcgen05.Field.ACCUMULATE, j >= O_STAGES)
                                cute.gemm(
                                    atom,
                                    acc,
                                    vh.view("V", None, None, i),
                                    ph.view("P", None, None, dk * pv_k_blocks + i),
                                    acc,
                                )
                                atom.set(tcgen05.Field.ACCUMULATE, True)
                ph.release()
                oh.commit()
            # The correction tail always merges two slots; a one-block split commits an empty
            # second one, which the tail reads as zero.
            if r.n_blocks == 1:
                with o.produce():
                    pass
            work.advance()
        return v, p, o, work

    @role(warps=(range(4, 8), range(8, 12)))
    @cute.jit
    def softmax(
        self,
        params: DecodeParams,
        *,
        stage: Instance,
        s: Consumer,
        p: Producer,
        row_max: Producer,
        row_sum: Producer,
        mufu: Turn,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        tile_hq, n = self.tile_hq, self.tile_hq_mma
        tidx = self.tile_row(me, "softmax")
        lane = me.lane
        owns_head = lane < tile_hq
        tStS, tLtL, _ = self.tmem_tensors(params, tmem)
        _, r2t, _, thr_r2t, vec_layout = self.vec_copies(tLtL, tidx)
        vec_size = cute.cosize(vec_layout)
        # Each thread owns one token row of each 128-token subtile and tile_hq head columns.
        tStS_0 = tStS[(None, None), 0, 0, None, 0]
        rep_s = tcgen05.Repetition(tile_hq * self.s_subtiles if n == tile_hq else tile_hq)
        t2r_s = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(rep_s, tcgen05.Pack.NONE), Float32), tStS_0
        )
        r2s_p = cute.make_tiled_copy_D(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16), t2r_s
        )
        thr_t2r_s, thr_r2s_p = t2r_s.get_slice(tidx), r2s_p.get_slice(tidx)
        tSrS_layout = thr_t2r_s.partition_D(tStS_0).layout
        tSrP_layout = cute.make_layout(
            (*vec_layout.shape, self.s_subtiles), stride=(*vec_layout.stride, vec_size)
        )
        scores_layout = cute.make_layout((tile_hq, self.s_subtiles), stride=(1, vec_size))
        # (token row, head, subtile) -> one P stage's (head, token) index, head + N * token.
        sP_view_layout = cute.make_layout((TILE_M, n, self.s_subtiles), stride=(n, 1, TILE_M * n))
        tLtL_mine = tLtL[(None, None), 0, 0, stage]
        while work.valid:
            r = self.split_range(params, work.item)
            for j in cutlass.range(stage, r.n_blocks, 2, unroll=1):
                tSrS = cute.make_rmem_tensor(tSrS_layout, Float32)
                tSrP = cute.make_rmem_tensor(tSrP_layout, cutlass.BFloat16)
                tLrL = cute.make_rmem_tensor(vec_layout, Float32)
                row_max_r = cute.make_rmem_tensor(vec_layout, Float32)
                scores_r = cute.make_tensor(tSrS.iterator, scores_layout)
                probs_r = cute.make_tensor(tSrP.iterator, scores_layout)
                row_sum_valid = tLrL[(None, 0), 0, 0]
                row_max_valid = row_max_r[(None, 0), 0, 0]

                sh = s.wait()
                row_max_r.fill(-Float32.inf)
                cute.copy(
                    t2r_s, thr_t2r_s.partition_S(tStS[(None, None), 0, 0, None, sh.index]), tSrS
                )
                cute.arch.fence_view_async_tmem_load()
                sh.release()

                # Token rows past seqlen_k in the last block.
                valid_rows = r.seqlen_k - (j + r.n_block_min) * self.tile_n
                for sm in cutlass.range_constexpr(self.s_subtiles):
                    if tidx + sm * TILE_M >= valid_rows:
                        scores_r[None, sm].fill(-Float32.inf)
                scores = scores_r.load()
                row_max_valid.store(
                    scores.reduce(cute.ReductionOp.MAX, -Float32.inf, (None, 0)).reshape(
                        (tile_hq,)
                    )
                )
                # Block max per head over this warp's rows; lane h holds head h's.
                lane_max = -Float32.inf
                for h in cutlass.range_constexpr(tile_hq):
                    row_max_r[h] = warp_reduce_max(row_max_r[h])
                    if h == lane:
                        lane_max = row_max_r[h]
                lane_max *= params.scale_log2

                # Correction must read M(j-1) before replacement. The turn serializes
                # warpgroups; commit(wait=True) completes every atomic max before reading M(j).
                with mufu.turn():
                    mh = row_max.acquire()
                    if owns_head:
                        smem_atomic_max(
                            params.sRowMax.iterator + params.sRowMax.layout(lane), lane_max
                        )
                    mh.commit(wait=True)
                    cute.autovec_copy(params.sRowMax, row_max_r)

                # P = exp2(S * scale * log2(e) - M(j)) in bf16, into SMEM for the PV MMA.
                ph = p.acquire()
                row_max_tile = row_max_valid.load().reshape((tile_hq, 1))
                probs = exp2_fast(params.scale_log2 * scores - row_max_tile)
                if cutlass.const_expr(tile_hq < n):
                    tSrP.fill(cutlass.BFloat16(0))
                probs_r.store(probs.to(cutlass.BFloat16))
                sP_view = cute.composition(ph.data, sP_view_layout)
                cute.autovec_copy(tSrP, thr_r2s_p.partition_D(sP_view))
                ph.commit()  # fence.proxy.async: the PV MMA reads P through the async proxy

                # Sum this thread's subtiles per head. Correction reduces the TMEM mailbox
                # across threads once, in its tail.
                row_sum_valid.store(
                    probs.reduce(cute.ReductionOp.ADD, Float32(0.0), (None, 0)).reshape((tile_hq,))
                )
                lh = row_sum.acquire()
                cute.copy(r2t, tLrL, thr_r2t.partition_D(tLtL_mine))
                lh.commit()  # tcgen05.wait::st before the arrival
            # The correction merges two row-sum epochs; a one-block split's second is empty.
            if cutlass.const_expr(stage == 1):
                if r.n_blocks == 1:
                    with row_sum.produce():
                        pass
            work.advance()
        return s, p, row_max, row_sum, mufu, work

    @role(warps=range(12, 16))
    @cute.jit
    def correction(
        self,
        params: DecodeParams,
        *,
        o: Consumer,
        row_max: Consumer,
        row_sum: Consumer,
        epi: Producer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        tile_hq = self.tile_hq
        lane = me.lane
        # The warp's TMEM lane quadrant (tidx // 32), which indexes its sRowSums row.
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % WARPS_PER_WG
        owns_head = lane < tile_hq
        _, tLtL, tOtO = self.tmem_tensors(params, tmem)
        tidx = self.tile_row(me, "correction")
        t2r, r2t, thr_t2r, thr_r2t, vec_layout = self.vec_copies(tLtL, tidx)
        vec_size = cute.cosize(vec_layout)
        r2s_o = cute.make_tiled_copy_D(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32), t2r
        )
        thr_r2s_o = r2s_o.get_slice(tidx)
        row_sum_slot_layout = cute.make_layout(
            (*vec_layout.shape, O_STAGES), stride=(*vec_layout.stride, vec_size)
        )
        o_tail_layout = cute.make_layout(
            (*vec_layout.shape, self.d_tiles, O_STAGES),
            stride=(*vec_layout.stride, vec_size, self.d_tiles * vec_size),
        )
        while work.valid:
            r = self.split_range(params, work.item)
            n_blocks = r.n_blocks
            tLrL = cute.make_rmem_tensor(vec_layout, Float32)
            row_sum_final = cute.make_rmem_tensor(vec_layout, Float32)
            acc_scale = cute.make_rmem_tensor(vec_layout, Float32)
            row_sum_acc = cute.make_rmem_tensor(row_sum_slot_layout, Float32)
            tOrO_tail = cute.make_rmem_tensor(o_tail_layout, Float32)
            row_sum_acc.fill(Float32(0.0))
            row_sum_final.fill(Float32(0.0))
            tOrO_tail.fill(Float32(0.0))
            acc_scale.fill(Float32(0.0))
            if n_blocks > 0:
                # Read M(0) and M(1) up front: rescaling slot j needs M(j) and M(j + 2).
                row_max_prev2, row_max_prev = -Float32.inf, -Float32.inf
                for preload in cutlass.range_constexpr(O_STAGES):
                    row_max_prev2 = row_max_prev
                    if preload == 0 or n_blocks > 1:
                        with row_max.consume():
                            row_max_prev = Float32(0.0)
                            if owns_head:
                                row_max_prev = params.sRowMax[lane]

                # Block s uses O slot s % 2: collect its row sums, read M(s + 2) (telling the
                # stats warp when it is final), then scale the slot by exp2(M(s) - M(s + 2)) once
                # PV(s) has landed, so PV(s + 2) accumulates on top.
                phase = Int32(0)
                for s in cutlass.range(n_blocks - O_STAGES, unroll=O_STAGES):
                    with row_sum.consume():
                        cute.copy(t2r, thr_t2r.partition_S(tLtL[(None, None), 0, 0, phase]), tLrL)
                        cute.arch.fence_view_async_tmem_load()
                    mh = row_max.wait()
                    if s == n_blocks - O_STAGES - 1:
                        self.max_done.arrive()
                    row_max_cur = Float32(0.0)
                    if owns_head:
                        row_max_cur = params.sRowMax[lane]
                    mh.release()

                    oh = o.wait()
                    acc_scale_lane = exp2_fast(row_max_prev2 - row_max_cur)
                    for h in cutlass.range_constexpr(tile_hq):
                        acc_scale[h] = cute.arch.shuffle_sync(acc_scale_lane, h)
                    for dm in cutlass.range_constexpr(self.d_tiles):
                        tOtO_slot = tOtO[(None, None), 0, 0, dm, phase]
                        tOrO = cute.make_rmem_tensor(vec_layout, Float32)
                        cute.copy(t2r, thr_t2r.partition_S(tOtO_slot), tOrO)
                        cute.arch.fence_view_async_tmem_load()
                        tOrO.store(tOrO.load() * acc_scale.load())
                        cute.copy(r2t, tOrO, thr_r2t.partition_D(tOtO_slot))
                    oh.release()  # tcgen05.wait::st first (release_fence)

                    row_sum_acc_cur = row_sum_acc[None, None, None, phase]
                    tLrL.store(tLrL.load() * acc_scale.load())
                    row_sum_acc_cur.store(row_sum_acc_cur.load() * acc_scale.load() + tLrL.load())
                    row_max_prev2, row_max_prev = row_max_prev, row_max_cur
                    phase ^= 1
                if n_blocks <= O_STAGES:
                    self.max_done.arrive()

                # Tail: blocks n-2 and n-1 are relative to M(n-2) and M(n-1). Bring slot
                # (n-2) % 2 up to the final max and add the slots. A one-block split has no
                # block n-2: its slot 1 (row sums and O) was never written and reads as zero.
                acc_scale_lane = exp2_fast(row_max_prev2 - row_max_prev)
                for h in cutlass.range_constexpr(tile_hq):
                    acc_scale[h] = cute.arch.shuffle_sync(acc_scale_lane, h)
                tail_phase = n_blocks % O_STAGES
                for ph in cutlass.range_constexpr(O_STAGES):
                    if tail_phase == ph:
                        fin = ph ^ 1
                        lh0 = row_sum.wait()
                        lh1 = row_sum.wait()
                        tLrL.fill(Float32(0.0))
                        if n_blocks > 1:
                            cute.copy(t2r, thr_t2r.partition_S(tLtL[(None, None), 0, 0, ph]), tLrL)
                        cute.copy(
                            t2r, thr_t2r.partition_S(tLtL[(None, None), 0, 0, fin]), row_sum_final
                        )
                        cute.arch.fence_view_async_tmem_load()
                        lh0.release()
                        lh1.release()
                        tLrL.store(tLrL.load() + row_sum_acc[None, None, None, ph].load())
                        row_sum_final.store(
                            row_sum_final.load()
                            + row_sum_acc[None, None, None, fin].load()
                            + acc_scale.load() * tLrL.load()
                        )
            else:
                self.max_done.arrive()

            # LSE and 1/L need no O values, so the stats round trip can overlap the final PV.
            for h in cutlass.range_constexpr(tile_hq):
                row_sum_final[h] = cute.arch.warp_reduction_sum(row_sum_final[h])
                if lane == 0:
                    params.sRowSums[warp, h] = row_sum_final[h]
            self.sum_done.arrive()
            # 1/L comes back through sRowMax.
            self.stats_ready.sync()
            cute.autovec_copy(params.sRowMax, tLrL)

            # Then the last two O slots. A one-block split skips slot (n-2) % 2, whose register
            # tile stays zero.
            if n_blocks > 0:
                for ph in cutlass.range_constexpr(O_STAGES):
                    if n_blocks % O_STAGES == ph:
                        fin = ph ^ 1
                        oh0 = o.wait()
                        oh1 = o.wait()
                        if n_blocks > 1:
                            cute.copy(
                                t2r,
                                thr_t2r.partition_S(tOtO[(None, None), 0, 0, None, ph]),
                                tOrO_tail[None, None, None, None, 0],
                            )
                        cute.copy(
                            t2r,
                            thr_t2r.partition_S(tOtO[(None, None), 0, 0, None, fin]),
                            tOrO_tail[None, None, None, None, 1],
                        )
                        cute.arch.fence_view_async_tmem_load()
                        oh0.release()
                        oh1.release()

            tail0 = tOrO_tail[None, None, None, None, 0].load().reshape((vec_size, self.d_tiles))
            tail1 = tOrO_tail[None, None, None, None, 1].load().reshape((vec_size, self.d_tiles))
            o_final = tail1 + acc_scale.load().reshape((vec_size, 1)) * tail0
            o_final *= tLrL.load().reshape((vec_size, 1))
            with epi.produce() as eh:
                tOsO = thr_r2s_o.partition_D(eh.data)
                tOsO.store(o_final.to(Float32).reshape(tOsO.shape))
            work.advance()
        return o, row_max, row_sum, epi, work

    @cute.kernel
    def kernel(
        self,
        mma_qk,
        mma_pv,
        tma_q,
        mQ,
        tma_k,
        mK,
        tma_v,
        mV,
        tma_o,
        mO,
        mLSE,
        mSeqUsedK,
        num_heads_kv,
        scale_log2,
        sQ_layout,
        sK_layout,
        sV_layout,
        sP_layout,
        sO_layout,
        bound,
    ):
        if cutlass.const_expr(self.work.schedule.persistent):
            raise ValueError(
                "SwapABDecode runs at most one work item per CTA: its round-robin rings and "
                f"one-shot stats flags assume it, got {self.work.schedule.describe()}"
            )
        layouts = {"P": sP_layout, "Q": sQ_layout, "K": sK_layout, "V": sV_layout, "O": sO_layout}
        smem = self.begin_roles(work=bound, layouts=layouts)
        thr_qk = mma_qk.get_slice(0)
        acc = thr_qk.make_fragment_C(thr_qk.partition_shape_C((TILE_M, self.tile_hq_mma)))
        rep = tcgen05.Repetition(self.tile_hq)
        t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(rep, tcgen05.Pack.NONE), Float32),
            acc[(None, None), 0, 0],
        )
        vec_layout = t2r.get_slice(0).partition_D(acc[(None, None), 0, 0]).layout
        sRowMax = smem.allocate_tensor(Float32, vec_layout, 128)
        sRowSums = smem.allocate_tensor(
            Float32, cute.make_layout((WARPS_PER_WG, self.tile_hq)), 16
        )
        # run_roles' CTA sync publishes this initialization to every role.
        tidx = cute.arch.thread_idx()[0]
        if tidx < self.tile_hq:
            sRowMax[tidx] = -Float32.inf
        params = DecodeParams(
            mma_qk=mma_qk,
            mma_pv=mma_pv,
            tma_q=tma_q,
            tma_k=tma_k,
            tma_v=tma_v,
            tma_o=tma_o,
            mQ=mQ,
            mK=mK,
            mV=mV,
            mO=mO,
            mLSE=mLSE,
            mSeqUsedK=mSeqUsedK,
            sQ=self.buffer("Q"),
            sK=self.buffer("K"),
            sV=self.buffer("V"),
            sP=self.buffer("P"),
            sRowMax=sRowMax,
            sRowSums=sRowSums,
            scale_log2=scale_log2,
            num_heads_kv=num_heads_kv,
            qhead_per_kvhead=mLSE.shape[2] // num_heads_kv,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,  # (b, h, d)
        k: cute.Tensor,  # (b, s, h_kv, d)
        v: cute.Tensor,
        o_partial: cute.Tensor,  # (splits, b, h, d) fp32
        lse_partial: cute.Tensor,  # (splits, b, h) fp32
        seqused_k: cute.Tensor,  # (b,) int32
        splits_table: cute.Tensor,  # (b,) int32; read only with per_sequence
        scale_log2: Float32,
        stream: cuda.CUstream,
    ):
        bf16 = cutlass.BFloat16
        n = self.tile_hq_mma
        num_heads, num_heads_kv = q.shape[1], k.shape[2]
        qpk = num_heads // num_heads_kv
        gqa = (qpk, num_heads_kv)
        mK = gemm_view(k, [1, 3, 2, 0])  # (s, d, (h_kv, b))
        mVt = gemm_view(v, [3, 1, 2, 0])  # (d, s, (h_kv, b))
        mQ = gemm_view(q, [0, 3, 2, 1], 1, *gqa)  # (qpk, d, (h_kv, b))
        mO = gemm_view(o_partial, [4, 0, 3, 2, 1], 2, *gqa)  # (d, qpk, (h_kv, b), split)
        tiler = (TILE_M, n, self.tile_k)
        major = OperandMajorMode
        one = tcgen05.CtaGroup.ONE
        # QK: K and Q are both K-major (head_dim contiguous). PV reduces over tokens, contiguous
        # in neither V nor P: both MN-major.
        mma_qk = sm100_utils.make_trivial_tiled_mma(
            bf16, major.K, major.K, Float32, one, (TILE_M, n)
        )
        mma_pv = sm100_utils.make_trivial_tiled_mma(
            bf16, major.MN, major.MN, Float32, one, (TILE_M, n)
        )
        n_k, n_v = self.kv_stages
        if cutlass.const_expr(self.shared_kv):  # both views of the shared ring: every stage
            n_k = n_v = n_k + n_v
        sQ_layout = sm100_utils.make_smem_layout_b(mma_qk, tiler, bf16, self.k_tiles)
        sP_layout = sm100_utils.make_smem_layout_b(
            mma_pv, (None, n, self.tile_n), bf16, self.p_stages
        )
        sK_layout = sm100_utils.make_smem_layout_a(mma_qk, tiler, bf16, n_k)
        sV_layout = sm100_utils.make_smem_layout_a(mma_pv, tiler, bf16, n_v)
        o_atom = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.MN_SW128, Float32)
        sO_layout = cute.tile_to_shape(o_atom, (self.d_tiles * TILE_M, n), order=(1, 0))
        sO_layout = cute.flat_divide(sO_layout, (TILE_M, n))
        load = cpasync.CopyBulkTensorTileG2SOp(one)
        vmnk = (1, 1, 1, 1)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(
            load, mQ, cute.select(sQ_layout, mode=[0, 1, 2]), tiler, mma_qk, vmnk
        )
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_A(
            load, mK, cute.select(sK_layout, mode=[0, 1, 2]), tiler, mma_qk, vmnk
        )
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_A(
            load, mVt, cute.select(sV_layout, mode=[0, 1, 2]), tiler, mma_pv, vmnk
        )
        tma_o, mO_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, cute.select(sO_layout, mode=[0, 1]), (TILE_M, n)
        )
        space = (o_partial.shape[0], num_heads_kv, q.shape[0], splits_table)
        bound = self.work.bind(space)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            mma_qk,
            mma_pv,
            tma_q,
            mQ_t,
            tma_k,
            mK_t,
            tma_v,
            mV_t,
            tma_o,
            mO_t,
            lse_partial,
            seqused_k,
            num_heads_kv,
            scale_log2,
            sQ_layout,
            sK_layout,
            sV_layout,
            sP_layout,
            sO_layout,
            bound,
        ).launch(
            **self.launch_config(bound),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.per_sequence,  # overlap with the kernel that writes the split counts
        )


class SharedKVDecode(SwapABDecode):
    """Use one exclusive-consumer K/V ring with FA4's QKIssued/PVIssued protocol.

    Select with ``shared_kv=True`` in wrappers. The ring has ``sum(kv_stages)`` stages.
    Load order is K(0), K(1), then K(j+2), V(j), ending with the last two V blocks.
    Each MMA warp waits/releases its own stages and ``skip()``s the other's.

    A parity wait for epoch e is exact once e-n was filled. A consumer's previous wait
    proves this only within one ring lap. PV's first V wait (epoch 3 * K-block stages),
    and later waits on short rings, need the named-barrier handshakes:

    * QK syncs on pv_issued before each block's first K-stage wait, then arrives on
      qk_issued after issuing the block.
    * PV arrives on pv_issued before its first block, after K(0) and K(1), and after
      each V block. It syncs on qk_issued before each V block.

    Before V(j)'s wait, QK has issued the preceding K(j+2); before K(j)'s wait for j>=3,
    PV has issued the preceding V(j-3). Earlier epochs are filled, making both waits exact.
    Each side syncs between arrivals on the other barrier, so neither arrives twice in one
    generation. PV's final pv_issued arrival is unmatched; OneShot forbids another item
    from reusing that generation.
    """

    shared_kv = True

    @role(warps=2)
    @cute.jit
    def load(self, params: DecodeParams, *, kv: Producer, work: WorkCursor):
        thr_qk, thr_pv, tKsK, tVsV = self.load_setup(params)
        kv = kv.views(K=StageView(tKsK, axis=1), V=StageView(tVsV, axis=1))
        lookahead = 2
        while work.valid:
            r = self.split_range(params, work.item)
            for s in cutlass.range(lookahead + r.n_blocks, unroll=1):
                if s < r.n_blocks:
                    sources = self.k_sources(params, thr_qk, r, s)
                    kv = self.fill(kv, "K", params.tma_k, sources, probe=True)
                if s >= lookahead:
                    sources = self.v_sources(params, thr_pv, r, s - lookahead)
                    kv = self.fill(kv, "V", params.tma_v, sources, probe=True)
            self.split_stats(params, r)
            work.advance()
        return kv, work

    @role(warps=0)
    @cute.jit
    def mma_qk(
        self,
        params: DecodeParams,
        *,
        q: Consumer,
        kv: Consumer,
        s: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        mma = params.mma_qk
        atom = cute.make_mma_atom(mma.op)
        tCrQ = mma.make_fragment_B(params.sQ)
        kv = kv.views(K=StageView(mma.make_fragment_A(params.sK), axis=3))
        tStS, _, _ = self.tmem_tensors(params, tmem)
        qk_k_blocks = self.tile_k // cute.size(mma.shape_mnk, mode=[2])
        k_stages = self.s_subtiles * self.k_tiles
        while work.valid:
            r = self.split_range(params, work.item)
            if r.n_blocks > 0:
                qh = q.wait()
                for block in cutlass.range(r.n_blocks, unroll=1):
                    if block >= 3:  # V(block - 3) was filled between K(block - 1) and K(block)
                        kv.skip(self.d_tiles * self.pv_k_tiles)
                    self.pv_issued.sync()
                    token = kv.try_wait()
                    sh = s.acquire()
                    for sm in cutlass.range_constexpr(self.s_subtiles):
                        atom.set(tcgen05.Field.ACCUMULATE, False)
                        acc = tStS[None, None, None, sm, sh.index]
                        for dk in cutlass.range_constexpr(self.k_tiles):
                            with kv.consume(token=token) as kh:
                                for i in cutlass.range_constexpr(qk_k_blocks):
                                    cute.gemm(
                                        atom,
                                        acc,
                                        kh.view("K", None, None, i, read=True, by="umma"),
                                        tCrQ[None, None, i, dk],
                                        acc,
                                    )
                                    atom.set(tcgen05.Field.ACCUMULATE, True)
                            if cutlass.const_expr(sm * self.k_tiles + dk == k_stages - 1):
                                self.qk_issued.arrive()
                            else:
                                token = kv.try_wait()
                    sh.commit()
                # PV's waits for V(n - 2) and V(n - 1) follow two more rendezvous.
                for _ in cutlass.range_constexpr(2):
                    self.pv_issued.sync()
                    self.qk_issued.arrive()
                qh.release()
            work.advance()
        return q, kv, s, work

    @role(warps=1)
    @cute.jit
    def mma_pv(
        self,
        params: DecodeParams,
        *,
        kv: Consumer,
        p: Consumer,
        o: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        mma = params.mma_pv
        atom = cute.make_mma_atom(mma.op)
        kv = kv.views(V=StageView(mma.make_fragment_A(params.sV), axis=3))
        p = p.views(P=StageView(mma.make_fragment_B(params.sP), axis=3))
        _, _, tOtO = self.tmem_tensors(params, tmem)
        pv_k_blocks = self.tile_k // cute.size(mma.shape_mnk, mode=[2])
        k_stages = self.s_subtiles * self.k_tiles
        v_stages = self.pv_k_tiles * self.d_tiles
        while work.valid:
            r = self.split_range(params, work.item)
            if r.n_blocks > 0:
                self.pv_issued.arrive()
                for ahead in cutlass.range_constexpr(2):  # K(0) and K(1) come before V(0)
                    if ahead < r.n_blocks:
                        kv.skip(k_stages)
                    self.qk_issued.sync()
                    self.pv_issued.arrive()
                for j in cutlass.range(r.n_blocks, unroll=1):
                    if j + 2 < r.n_blocks:  # K(j + 2) was filled just before V(j)
                        kv.skip(k_stages)
                    self.qk_issued.sync()
                    token = kv.try_wait()
                    ph = p.wait()
                    # Free once the correction has rescaled what PV(j - 2) left there.
                    oh = o.acquire()
                    for dk in cutlass.range_constexpr(self.pv_k_tiles):
                        for dm in cutlass.range_constexpr(self.d_tiles):
                            last = dk * self.d_tiles + dm == v_stages - 1
                            acc = tOtO[None, None, None, dm, oh.index]
                            with kv.consume(token=token) as vh:
                                for i in cutlass.range_constexpr(pv_k_blocks):
                                    # Blocks 0 and 1 start their slot: no zero-fill first.
                                    if cutlass.const_expr(dk == 0 and i == 0):
                                        atom.set(tcgen05.Field.ACCUMULATE, j >= O_STAGES)
                                    cute.gemm(
                                        atom,
                                        acc,
                                        vh.view("V", None, None, i, read=True, by="umma"),
                                        ph.view("P", None, None, dk * pv_k_blocks + i),
                                        acc,
                                    )
                                    atom.set(tcgen05.Field.ACCUMULATE, True)
                                if cutlass.const_expr(last):
                                    self.pv_issued.arrive()
                            if cutlass.const_expr(not last):
                                token = kv.try_wait()
                    ph.release()
                    oh.commit()
            # The correction tail always merges two slots; a one-block split commits an empty
            # second one, which the tail reads as zero.
            if r.n_blocks == 1:
                with o.produce():
                    pass
            work.advance()
        return kv, p, o, work


@jit_cache(uncached=lambda op: op.trace_protocol)
def compile_decode(op: SwapABDecode):
    """Compile for dynamic batch, heads, KV length and split count.

    Memory and disk caches use ``op.__cute_cache_key__()``.
    """
    b, h, hkv, s, ns = (cute.sym_int() for _ in range(5))
    bf16, i32 = cutlass.BFloat16, cutlass.Int32
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (b, h, op.head_dim)),
        make_fake_compact_tensor(bf16, (b, s, hkv, op.head_dim)),
        make_fake_compact_tensor(bf16, (b, s, hkv, op.head_dim_v)),
        make_fake_compact_tensor(Float32, (ns, b, h, op.head_dim_v)),
        make_fake_compact_tensor(Float32, (ns, b, h), assumed_align=4),
        make_fake_compact_tensor(i32, (b,), assumed_align=4),
        make_fake_compact_tensor(i32, (b,), assumed_align=4),
        Float32(1.0),
        name=op.get_name(),
    )


def _frozen_roles(roles):
    return None if roles is None else tuple(sorted(roles.items()))


def tile_hq_for(num_heads: int, num_heads_kv: int) -> int:
    """Return the next power of two covering the GQA group."""
    return 1 << (num_heads // num_heads_kv - 1).bit_length()


def decode_partials(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_splits: int,
    *,
    seqused_k: torch.Tensor | None = None,
    splits_per_seq: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    kv_stages: tuple[int, int] | None = None,
    **config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fp32 O/LSE split partials in FA4's combine layout.

    O is ``(splits, b, 1, h, d_v)``; LSE is ``(splits, b, h, 1)``. Inputs are contiguous
    bf16: Q ``(b, 1, h, d)``, K ``(b, s, h_kv, d)`` and V ``(b, s, h_kv, d_v)``.
    ``splits_per_seq`` is int32 ``(b,)`` with counts <= ``num_splits``. Partials past each
    sequence's count stay unwritten; splits without KV blocks write O=0 and LSE=-inf.

    ``config`` accepts ``SwapABDecode`` options ``tile_n``, ``tile_k``, ``s_stages``,
    ``p_stages`` and ``roles``, plus ``shared_kv=True`` to select ``SharedKVDecode``.
    ``head_dim`` and ``head_dim_v`` follow the inputs.
    """
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError("decode_partials expects bfloat16 q, k, v")
    if (
        q.ndim != 4
        or q.shape[1] != 1
        or k.ndim != 4
        or k.shape[:3] != v.shape[:3]
        or k.shape[-1] != q.shape[-1]
    ):
        raise ValueError("expected q (b, 1, h, d), k (b, s, h_kv, d) and v (b, s, h_kv, d_v)")
    batch, _, heads, head_dim = q.shape
    heads_kv, head_dim_v = k.shape[2], v.shape[-1]
    if k.shape[0] != batch or heads % heads_kv or heads // heads_kv > 32:
        raise ValueError("heads must be a multiple of kv heads, at most 32 per kv head")
    if any(not t.is_contiguous() for t in (q, k, v)):
        raise ValueError("q, k, v must be contiguous")
    if num_splits < 1:
        raise ValueError("num_splits must be positive")
    dev = q.device
    if seqused_k is None:
        seqused_k = torch.full((batch,), k.shape[1], dtype=torch.int32, device=dev)
    per_sequence = splits_per_seq is not None
    table = splits_per_seq if per_sequence else seqused_k  # unread without per_sequence
    o_partial = torch.empty(
        num_splits, batch, 1, heads, head_dim_v, device=dev, dtype=torch.float32
    )
    lse_partial = torch.empty(num_splits, batch, heads, 1, device=dev, dtype=torch.float32)
    options = dict(config, head_dim=head_dim, head_dim_v=head_dim_v)
    shared_kv = options.pop("shared_kv", False)
    if type(shared_kv) is not bool:
        raise ValueError(f"shared_kv must be a bool, got {shared_kv!r}")
    kernel_class = SharedKVDecode if shared_kv else SwapABDecode
    frozen = {k: _frozen_roles(v) if k == "roles" else v for k, v in options.items()}
    key = (
        tile_hq_for(heads, heads_kv),
        kv_stages,
        per_sequence,
        shared_kv,
        *sorted(frozen.items()),
    )
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = kernel_class(key[0], kv_stages, per_sequence, **options)
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    compile_decode.by_static_key(key, lambda: compile_decode(op))(
        q[:, 0],
        k,
        v,
        o_partial[:, :, 0],
        lse_partial[..., 0],
        seqused_k,
        table,
        scale * math.log2(math.e),
    )
    return o_partial, lse_partial


def fa4_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_splits: int,
    *,
    seqused_k: torch.Tensor | None = None,
    splits_per_seq: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    kv_stages: tuple[int, int] | None = None,
    **config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return single-token attention output and LSE using FA4's split-combine kernel.

    Output is bf16 ``(b, 1, h, d_v)``; LSE is fp32 ``(b, h, 1)``.
    See ``decode_partials`` for input and config contracts.
    """
    from flash_attn.cute.interface import _flash_attn_fwd_combine

    o_partial, lse_partial = decode_partials(
        q,
        k,
        v,
        num_splits,
        seqused_k=seqused_k,
        splits_per_seq=splits_per_seq,
        softmax_scale=softmax_scale,
        kv_stages=kv_stages,
        **config,
    )
    out = torch.empty(*q.shape[:3], v.shape[-1], device=q.device, dtype=q.dtype)
    lse = torch.empty(q.shape[0], q.shape[2], 1, device=q.device, dtype=torch.float32)
    _flash_attn_fwd_combine(
        o_partial,
        lse_partial.transpose(-1, -2),
        out,
        lse.transpose(-1, -2),
        num_splits_dynamic_ptr=splits_per_seq,
    )
    return out, lse


_OPS: dict = {}
