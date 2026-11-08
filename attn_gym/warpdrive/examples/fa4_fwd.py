"""Cut-down FlashAttention-4 forward (SM100) written with ``warpdrive``.

Scope: bf16 in/out, fp32 accumulation, head_dim 64 or 128, non-causal, no masking, work items
of (256 query rows, head, batch) scheduled by a ``WorkStream`` (``OneShot`` by default, or a
persistent schedule), 1-CTA MMAs. ``seqlen_k`` must be a multiple of the K/V tile (``tile_n``,
128 or 64; no key masking); ``seqlen_q`` may be any length because TMA zero-fills the loads of
the last Q tile and clips its O store. Returns O and the log-sum-exp (natural log).

The schedule mirrors FA4's non-ping-pong path (``flash_attn/cute/flash_fwd_sm100.py``):
two 128-row Q tiles per CTA (``q_stage=2``), ``tile_n``-column K/V tiles, warps 0-3 softmax for
tile 0, 4-7 softmax for tile 1, 8-11 correction, 12 MMA, 13 epilogue, 14 load, 15 idle. The
tiles, head dimension, stage count, rescale threshold and the roles' placement and register
budgets are constructor arguments (docs/warpdrive/configuration.md).

Dependency graph (channels, all mbarriers; slot j = Q tile j):

    load --q[j] (TMA)--> mma            Q held for the whole KV loop
    load --kv ring (TMA)--> mma         K_i and V_i alternate in one 3-stage ring
    mma --s[j] (UMMA)--> softmax[j] + correction   fan-out on S_j(i) full; the empty side is
                                        the join "P_j(i) written AND O_j rescaled" (256 arrivals)
    softmax[j] --stats[j] (SMEM payload)--> correction   per-row rescale factor, then row sum/max
    mma --s[j] final commit--> correction + softmax[j]   final PV done (O_j ready to store);
                                        its release lets the next work item reuse S_j and O_j
    correction --epi[j] (sO)--> epilogue  TMA store of O
    softmax[0] <-> softmax[1] turnstile around exp2 (FA4's s0_s1_sequence; off by default,
                                        as in FA4)

TMEM (512 columns at the defaults): S0 [0,128), S1 [128,256), O0 [256,384), O1 [384,512);
P_j is bf16 packed two per column into the upper half of S_j ([64,128) of each S slot). S slots
are ``tile_n`` columns and O slots ``head_dim``.

SMEM is an ``SmemPlan``: sO, sQ and sK, with V reusing K's stages (``overlaps=``), bound to the
epi, q and kv channels so each handle selects its own stage. ``overlap_o_q=True`` also places sO
over sQ (FA4's ``overlap_sO_sQ``, 64 KB less SMEM); it needs one work item per CTA, because this
example has no epilogue -> load edge to keep a persistent CTA's next Q off the O being stored.
Accesses to the aliases are annotated (``tmem.access``, ``h.view(..., read=True)``) for the
alias diagnostic, which reports no potential hazard for this kernel.

2-CTA (``cluster=CtaPair()``, FA4's default schedule): a work item is 512 query rows. Q tile j of
the item is one 256-row MMA tile split across the pair (each CTA holds 128 rows, as FA4's
``work_q_block``), K and V tiles are split along the MMA's N (each CTA loads 64 keys of K and 64
head dims of V, so 6 KV stages fit in the same SMEM), and the leader's MMA warp issues
``CtaGroup.TWO`` MMAs. The roles are unchanged; the layer derives the pair-scoped q/kv/s channels
(leader-armed TMA bytes of both CTAs, multicast S commits, S releases from both CTAs joining on
the leader's barrier) and keeps stats/epi/turnstile CTA-local.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, cute
from cutlass.base_dsl import BaseDSL
from cutlass.cute.nvgpu import cpasync, tcgen05
from flash_attn.cute.softmax import SoftmaxSm100
from quack import copy_utils, layout_utils

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.compat import Arch, LayoutEnum
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Consumer,
    Instance,
    Local,
    OneShot,
    Producer,
    ProtocolError,
    Raster,
    Ring,
    RoleConfig,
    SingleCta,
    Slots,
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
from attn_gym.warpdrive.examples.tma_tiles import tma_partition_tiles

M_BLOCK = 128  # Q rows per CTA MMA tile: one softmax thread per row
Q_STAGE = 2  # Q tiles per work item, one softmax warpgroup each
TILE_NS = (64, 128)  # keys per K/V tile; TMEM holds Q_STAGE x (tile_n + head_dim) columns
HEAD_DIMS = (64, 128)
LOG2_E = math.log2(math.e)
LN2 = math.log(2.0)


@dataclass(frozen=True)
class Fa4Params:
    """Kernel-prologue values the roles read; built once in ``FlashAttentionForward.kernel``."""

    mma_qk: cute.TiledMma  # S = Q K^T
    mma_pv: cute.TiledMma  # O += P V, P from TMEM
    tma_q: cute.CopyAtom
    tma_k: cute.CopyAtom
    tma_v: cute.CopyAtom
    tma_o: cute.CopyAtom
    mQ: cute.Tensor  # (s, d, h, b) TMA views; V is (d, s, h, b)
    mK: cute.Tensor
    mV: cute.Tensor
    mO: cute.Tensor
    mLSE: cute.Tensor  # (b, h, s_q)
    sQ: cute.Tensor
    sK: cute.Tensor
    sV: cute.Tensor  # aliases sK's stages
    sO: cute.Tensor
    p_layout: cute.ComposedLayout
    scale_log2: Float32
    s_layout: cute.Layout  # TMEM S and O accumulators
    o_layout: cute.Layout
    n_blocks: cutlass.Int32  # K/V blocks per head
    seqlen_q: cutlass.Int32


def umma(tiled_mma, acc, a, b, accumulate, ks=None):
    """Issue one tile MMA as K-block tcgen05.mma instructions; ``accumulate`` may be dynamic.
    ``ks``: the K-block range to issue (default: all)."""
    atom = cute.make_mma_atom(tiled_mma.op)
    ks = range(cute.size(a.shape[2])) if ks is None else ks
    for k in ks:
        atom.set(tcgen05.Field.ACCUMULATE, accumulate if k == ks[0] else True)
        cute.gemm(atom, acc, a[None, None, k], b[None, None, k], acc)


# FA4 split_P_arrive: S/P/O is released after 3/4 of the P stores (3 of 4 at tile_n 128: 96 of 128
# keys; 1 of 2 at tile_n 64). The MMA issues the K blocks those stores cover before waiting for
# the rest, so both sides derive their split from the same store count (``split_p_stores``).
SPLIT_P = 3


class FlashAttentionForward(WarpSpecialized):
    def __init__(
        self,
        kv_stages: int | None = None,
        turnstile: bool = False,
        trace: bool = False,
        schedule=None,
        cluster=None,
        split_p: bool = False,
        roles: dict[str, RoleConfig] | None = None,
        mapping: WorkMap | None = None,
        overlap_o_q: bool = False,
        use_ldred_rowmax: bool = False,
        tile_n: int = 128,
        head_dim: int = 128,
        rescale_threshold: float = 8.0,
    ):
        if use_ldred_rowmax and not BaseDSL._get_dsl().get_arch_enum().is_family_of(Arch.sm_103f):
            raise ValueError("use_ldred_rowmax requires an SM103-family target")
        if tile_n not in TILE_NS:
            raise ValueError(f"tile_n must be one of {TILE_NS}, got {tile_n}")
        if head_dim not in HEAD_DIMS:
            raise ValueError(f"head_dim must be one of {HEAD_DIMS}, got {head_dim}")
        if not rescale_threshold >= 0.0:
            raise ValueError(
                f"rescale_threshold must be >= 0 (log2 units), got {rescale_threshold}"
            )
        self.tile_n, self.head_dim = tile_n, head_dim
        self.p_stores = tile_n // 32  # 16-column tcgen05.st of bf16 pairs per P tile
        self.split_p_stores = self.p_stores * SPLIT_P // 4
        # FA4 bf16: skip rescaling O unless the row max grew by more than 2**rescale_threshold.
        self.rescale_threshold = float(rescale_threshold)
        # Placement and register budgets by role name, over the @role defaults (FA4's 192/80/48
        # split). The math still assumes four warps for softmax and correction, one for the rest.
        self.roles = roles
        self.cluster = cluster or SingleCta()
        if self.cluster.size != self.cluster.pair_size:
            raise ValueError(f"the FA4 example runs on one CTA or a CTA pair, got {self.cluster}")
        self.split_p = split_p
        self.overlap_o_q = overlap_o_q
        self.use_ldred_rowmax = use_ldred_rowmax
        schedule = schedule or OneShot()
        if overlap_o_q and schedule.persistent:
            raise ValueError(
                "overlap_o_q needs one work item per CTA: a persistent CTA would load the next "
                "item's Q while the epilogue still stores O from the same bytes"
            )
        cs = self.cluster.size
        # Each CTA holds 1/cs of every K/V tile (2-CTA MMAs split B along N): cs x the stages.
        self.kv_stages = kv_stages or 3 * cs
        bf16 = cutlass.BFloat16
        self.smem_plan = SmemPlan()
        if not overlap_o_q:
            self.smem_plan.buffer("O", bf16, align=1024)
        q_buf = self.smem_plan.buffer("Q", bf16, align=1024)
        k_buf = self.smem_plan.buffer("K", bf16, align=1024)
        v_buf = self.smem_plan.buffer("V", bf16, align=1024, overlaps=k_buf)  # K/V share stages
        if overlap_o_q:  # FA4's overlap_sO_sQ: O reuses Q's bytes once the item's QKs are done
            self.smem_plan.buffer("O", bf16, align=1024, overlaps=q_buf)
        q_bytes, kv_bytes = M_BLOCK * head_dim * 2, tile_n * head_dim * 2  # bf16 tiles
        self.q = Slots(Q_STAGE, Tma(q_bytes), Umma, producer_role="load", buffer=q_buf)
        self.kv = Ring(self.kv_stages, Tma(kv_bytes // cs), Umma, buffer={"K": k_buf, "V": v_buf})
        self.s = Slots(
            Q_STAGE,
            Umma,
            Threads,
            producer_role="mma",
            consumer_roles=("softmax", "correction"),
            release_fence="tmem_store",  # P (softmax) and rescaled O (correction) via tcgen05.st
        )
        # Split-P arrive (FA4): softmax releases s after the first 3/4 of P, so the MMA starts
        # PV_j on those keys while the last quarter is stored; p_last carries the last quarter.
        # Declared either way so the role signatures are fixed; unused when split_p is off.
        self.p_last = Slots(Q_STAGE, Threads, Umma, commit_fence="tmem_store")
        self.stats = Slots(Q_STAGE, Threads, Threads, payload=(Float32, (M_BLOCK, 2)))
        self.epi = Slots(
            Q_STAGE, Threads, Threads, commit_fence="smem_async", buffer=self.smem_plan["O"]
        )
        self.mufu = Turnstile(enabled=turnstile)
        self.tmem_plan = (
            TmemPlan()
            .region("S", tile_n, Q_STAGE)
            .region("O", head_dim, Q_STAGE)
            .alias("P", of="S", offset=tile_n // 2, cols=tile_n // 2)
        )
        # Work items: (query block, head, batch). The default Raster(along="m") keeps a head's
        # query blocks adjacent for K/V L2 reuse. Any WorkMap over this space can replace it.
        self.work = WorkStream(mapping=mapping or Raster(along="m"), schedule=schedule)
        self.trace_protocol = trace

    def get_name(self) -> str:
        ts = ("_ts" if self.mufu.enabled else "") + ("_splitp" if self.split_p else "")
        ts += "_oq" if self.overlap_o_q else ""
        ts += "_ldred" if self.use_ldred_rowmax else ""
        if (self.tile_n, self.head_dim) != (128, 128):
            ts += f"_n{self.tile_n}_d{self.head_dim}"
        return (
            f"warpdrive_fa4_fwd_{self.cluster.get_key()}_kv{self.kv_stages}{ts}_"
            f"{self.work.get_key()}{self.trace_key()}"
        )

    # ------------------------------------------------------------------ roles

    @role(warps=14, regs=48)
    @cute.jit
    def load(self, params: Fa4Params, *, q: Producer, kv: Producer, work: WorkCursor):
        rank = self.cluster.rank()  # this CTA's rows of Q, keys of K, head dims of V
        thr = params.mma_qk.get_slice(rank), params.mma_pv.get_slice(rank)
        while work.valid:
            q, kv = self.load_item(params, q, kv, thr, work.item)
            work.advance()
        return q, kv, work

    @cute.jit
    def load_item(self, params: Fa4Params, q, kv, thr, item):
        """Issue one work item's Q, K and V loads. ``thr``: this CTA's QK and PV MMA slices.
        Returns the ends: the caller reassigns them."""
        thr_qk, thr_pv = thr
        mma_m = M_BLOCK * self.cluster.size
        m_blk, head, batch = item
        gQ = cute.local_tile(
            params.mQ[None, None, head, batch], (Q_STAGE * mma_m, self.head_dim), (m_blk, 0)
        )
        gQ = layout_utils.select(cute.flat_divide(gQ, (mma_m,)), mode=[0, 2, 1])
        load_q, _, _ = copy_utils.tma_get_copy_fn(
            params.tma_q, 0, cute.make_layout(1), thr_qk.partition_A(gQ), params.sQ
        )
        gK = cute.local_tile(
            params.mK[None, None, head, batch], (self.tile_n, self.head_dim), (None, 0)
        )
        gV = cute.local_tile(
            params.mV[None, None, head, batch], (self.head_dim, self.tile_n), (0, None)
        )
        tKsK, tKgK = tma_partition_tiles(params.tma_k, params.sK, thr_qk.partition_B(gK))
        tVsV, tVgV = tma_partition_tiles(params.tma_v, params.sV, thr_pv.partition_B(gV))
        kv = kv.views(K=StageView(tKsK, axis=1), V=StageView(tVsV, axis=1))
        # FA4 order: K0, Q0, Q1, V0, then (K_i, V_i).
        with kv.produce() as h:
            cute.copy(
                params.tma_k,
                tKgK[None, 0],
                h.view("K", write=True, by="tma"),
                tma_bar_ptr=h.barrier,
            )
        for j in cutlass.range_constexpr(Q_STAGE):
            with q.produce(j) as h:
                h.access(write=True, by="tma")  # load_q selects stage j itself
                load_q(src_idx=j, dst_idx=j, tma_bar_ptr=h.barrier)
        with kv.produce() as h:
            cute.copy(
                params.tma_v,
                tVgV[None, 0],
                h.view("V", write=True, by="tma"),
                tma_bar_ptr=h.barrier,
            )
        for i in cutlass.range(1, params.n_blocks, unroll=1):
            with kv.produce() as h:
                cute.copy(
                    params.tma_k,
                    tKgK[None, i],
                    h.view("K", write=True, by="tma"),
                    tma_bar_ptr=h.barrier,
                )
            with kv.produce() as h:
                cute.copy(
                    params.tma_v,
                    tVgV[None, i],
                    h.view("V", write=True, by="tma"),
                    tma_bar_ptr=h.barrier,
                )
        return q, kv

    @role(warps=12, regs=48, cta="leader")
    @cute.jit
    def mma(
        self,
        params: Fa4Params,
        *,
        q: Consumer,
        kv: Consumer,
        s: Producer,
        p_last: Consumer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        thr_qk, thr_pv = params.mma_qk.get_slice(0), params.mma_pv.get_slice(0)
        # MMA fragments over the Q, K and V stages; handles select their own stage.
        q = q.views(Q=StageView(thr_qk.make_fragment_A(params.sQ), axis=3))
        kv = kv.views(
            K=StageView(thr_qk.make_fragment_B(params.sK), axis=3),
            V=StageView(thr_pv.make_fragment_B(params.sV), axis=3),
        )
        tS = [cute.make_tensor(tmem.ptr("S", j), params.s_layout) for j in range(Q_STAGE)]
        tO = [cute.make_tensor(tmem.ptr("O", j), params.o_layout) for j in range(Q_STAGE)]
        # P as the TMEM A operand of PV: bf16 view of the S columns (2 elements per column).
        tOrP0 = thr_pv.make_fragment_A(cute.make_tensor(tmem.base, params.p_layout.outer))
        tOrP0 = tOrP0[None, None, None, 0]
        tOrP = [
            cute.make_tensor(tOrP0.iterator + tmem.cols("P", j) * 2, tOrP0.layout)
            for j in range(Q_STAGE)
        ]
        while work.valid:
            # S_j(0) = Q_j K_0^T. Q handles stay open across the KV loop.
            k0 = kv.wait()
            qh = []
            for j in cutlass.range_constexpr(Q_STAGE):
                qh.append(q.wait(j))
                sp = s.acquire(j)  # previous item's final S_j released: O_j stored, P_j read
                tmem.access("S", j, write=True, by="umma")
                umma(
                    params.mma_qk,
                    tS[j],
                    qh[j].view(read=True, by="umma"),
                    k0.view("K", read=True, by="umma"),
                    False,
                )
                sp.commit()
            k0.release()

            # Block i: O_j += P_j(i-1) V_(i-1); S_j(i) = Q_j K_i^T. V(i-1) is released after
            # the second PV but before K_i is: overlapping, non-nested handle lifetimes.
            for i in cutlass.range(params.n_blocks - 1, unroll=1):
                v = kv.wait()
                for j in cutlass.range_constexpr(Q_STAGE):
                    sl = s.acquire(j)  # P_j written and O_j rescaled (softmax[j], correction)
                    p_last = self.pv(
                        p_last,
                        j,
                        tO[j],
                        tOrP[j],
                        v.view("V", read=True, by="umma"),
                        i > 0,
                        params,
                        tmem,
                    )
                    if cutlass.const_expr(j == 0):
                        k = kv.wait()
                    if cutlass.const_expr(j == Q_STAGE - 1):
                        v.release()
                    # Overwrites S_j (and the P_j alias) after PV_j read P_j: an MMA's TMEM A
                    # read and a later MMA's D write from one thread are pipelined (PTX ISA
                    # 9.7.18.6.2), so no extra barrier is needed. The reverse order is not.
                    tmem.access("S", j, write=True, by="umma")
                    umma(
                        params.mma_qk,
                        tS[j],
                        qh[j].view(read=True, by="umma"),
                        k.view("K", read=True, by="umma"),
                        False,
                    )
                    # No O_full commit here: correction rescales O_j only after waiting S_j(i)
                    # full, and this commit completes after every earlier MMA, including PV_j.
                    sl.commit()
                k.release()
            for h in qh:
                h.release()

            # Final PV. Its S_j commit means "O_j final": correction stores O_j, and both S
            # consumers release it, which is what lets the next item's QK_j overwrite S_j/P_j
            # and its first PV_j overwrite O_j. One commit per acquire keeps S_j balanced
            # across items (n + 1 commits, n + 1 waits per consumer).
            v_last = kv.wait()
            for j in cutlass.range_constexpr(Q_STAGE):
                with s.produce(j):
                    p_last = self.pv(
                        p_last,
                        j,
                        tO[j],
                        tOrP[j],
                        v_last.view("V", read=True, by="umma"),
                        params.n_blocks > 1,
                        params,
                        tmem,
                    )
            v_last.release()
            work.advance()
        return q, kv, s, p_last, work

    def pv(self, p_last, j, tO, tOrP, tOrV, accumulate, params: Fa4Params, tmem):
        """O_j (+)= P_j V. With split-P the last quarter of the K blocks waits for p_last[j]
        (plain Python: inlined; returns the end so loops carry its phase)."""
        if cutlass.const_expr(not self.split_p):
            tmem.access("P", j, read=True, by="umma")
            umma(params.mma_pv, tO, tOrP, tOrV, accumulate)
            return p_last
        n = cute.size(tOrP.shape[2])
        first = n * self.split_p_stores // self.p_stores  # K blocks the early P stores cover
        p_cols = self.tile_n // 2  # P packs two bf16 per 32-bit column; K block k reads n-th
        tmem.access("P", j, read=True, by="umma", cols=range(p_cols * first // n))
        umma(params.mma_pv, tO, tOrP, tOrV, accumulate, range(first))
        with p_last.consume(j):  # release = commit: this PV has read the last quarter
            tmem.access("P", j, read=True, by="umma", cols=range(p_cols * first // n, p_cols))
            umma(params.mma_pv, tO, tOrP, tOrV, True, range(first, n))
        return p_last

    @role(warps=(range(4), range(4, 8)), regs=192)
    @cute.jit
    def softmax(
        self,
        params: Fa4Params,
        *,
        stage: Instance,
        s: Consumer[Instance],
        p_last: Producer[Instance],
        stats: Producer[Instance],
        mufu: Turn,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        tidx = self.tile_row(me, "softmax")
        thr_qk = params.mma_qk.get_slice(self.cluster.rank())
        tSAcc = cute.make_tensor(tmem.ptr("S", stage), params.s_layout)[(None, None), 0, 0]
        tStP = cute.make_tensor(
            tmem.ptr("P", stage),
            cute.composition(tSAcc.layout, cute.make_layout((M_BLOCK, self.tile_n // 2))),
        )
        mma_m = M_BLOCK * self.cluster.size
        tScS = thr_qk.partition_C(cute.make_identity_tensor((mma_m, self.tile_n)))[
            (None, None), 0, 0
        ]
        ld_op = (
            tcgen05.copy.LdRed32x32bOp(tcgen05.copy.Repetition(32))
            if self.use_ldred_rowmax
            else tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32))
        )
        ld_atom = cute.make_copy_atom(ld_op, Float32)
        thr_ld = tcgen05.make_tmem_copy(ld_atom, tSAcc).get_slice(tidx)
        st_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_st = tcgen05.make_tmem_copy(st_atom, tStP).get_slice(tidx)
        t = SimpleNamespace(
            tidx=tidx,
            tmem=tmem,
            thr_ld=thr_ld,
            thr_st=thr_st,
            tStS=thr_ld.partition_S(tSAcc),
            tStP=thr_st.partition_D(tStP),
            rS_shape=thr_ld.partition_D(tScS).shape,
            rP_shape=thr_st.partition_S(
                cute.make_identity_tensor((M_BLOCK, self.tile_n // 2))
            ).shape,
        )
        while work.valid:
            sm = SoftmaxSm100.create(params.scale_log2, rescale_threshold=self.rescale_threshold)
            sm.reset()
            s, p_last, stats, mufu = self.softmax_step(stage, s, p_last, stats, mufu, sm, t, True)
            for _ in cutlass.range(params.n_blocks - 1, unroll=1):
                s, p_last, stats, mufu = self.softmax_step(
                    stage, s, p_last, stats, mufu, sm, t, False
                )
            with stats.produce(stage) as st:
                st.data[tidx, 0] = sm.row_sum[0]
                st.data[tidx, 1] = sm.row_max[0]
            with s.consume(stage):  # the final "O_j done" commit; nothing to read
                pass
            work.advance()
        return s, p_last, stats, mufu, work

    def softmax_step(self, stage, s, p_last, stats, mufu, sm, t, is_first):
        """One KV block for one Q tile (plain Python: inlined into the role at trace time).

        Returns the ends so the caller's loop carries their phases.
        """
        sh = s.wait(stage)
        rS = cute.make_rmem_tensor(t.rS_shape, Float32)
        t.tmem.access("S", stage, read=True)
        if cutlass.const_expr(self.use_ldred_rowmax):
            # The unmasked x32 loads each return a row max alongside their scores.
            rMax = cute.make_rmem_tensor(((1, 1), *rS.shape[1:]), Float32)
            cute.copy(t.thr_ld, t.tStS, (rS, rMax))
            # Reuse the software updater on four maxima to preserve its fast exp2/FTZ
            # rescale arithmetic; FA4's precomputed-max updater uses non-fast exp2.
            row_max, acc_scale = sm.update_row_max(rMax.load(), is_first)
        else:
            cute.copy(t.thr_ld, t.tStS, rS)
            row_max, acc_scale = sm.update_row_max(rS.load(), is_first)
        with stats.produce(stage) as st:
            st.data[t.tidx, 0] = Float32(1.0) if is_first else acc_scale
        sm.scale_subtract_rowmax(rS, row_max)
        rP32 = cute.make_rmem_tensor(t.rP_shape, Float32)
        rP = cute.make_tensor(cute.recast_ptr(rP32.iterator, dtype=cutlass.BFloat16), rS.layout)
        with mufu.turn():
            sm.apply_exp2_convert(rS, rP)
        n_st = cute.size(t.tStP.shape[2])
        split = self.split_p_stores if self.split_p else n_st
        p_cols = self.tile_n // 2  # P's 32-bit columns (two bf16 each); store i writes n_st-th
        t.tmem.access("P", stage, write=True, cols=range(p_cols * split // n_st))
        for i in range(split):  # plain Python: no preprocessor here
            cute.copy(t.thr_st, rP32[None, None, i], t.tStP[None, None, i])
        sh.release()  # layer fences the tcgen05.st of P before arriving
        if cutlass.const_expr(self.split_p):
            with p_last.produce(stage):  # layer fences the last quarter's tcgen05.st
                t.tmem.access("P", stage, write=True, cols=range(p_cols * split // n_st, p_cols))
                for i in range(split, n_st):
                    cute.copy(t.thr_st, rP32[None, None, i], t.tStP[None, None, i])
        sm.update_row_sum(rS.load(), acc_scale, is_first)
        return s, p_last, stats, mufu

    @role(warps=range(8, 12), regs=80)
    @cute.jit
    def correction(
        self,
        params: Fa4Params,
        *,
        s: Consumer,
        stats: Consumer,
        epi: Producer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        tidx = self.tile_row(me, "correction")
        rank = self.cluster.rank()
        thr_pv = params.mma_pv.get_slice(rank)
        tO = [cute.make_tensor(tmem.ptr("O", j), params.o_layout) for j in range(Q_STAGE)]
        while work.valid:
            m_blk, head, batch = work.item
            for _ in cutlass.range(params.n_blocks, unroll=1):
                for j in cutlass.range_constexpr(Q_STAGE):
                    with stats.consume(j) as st:
                        scale = st.data[tidx, 0]
                    # S_j(i) full: QK_j(i) was issued after PV_j(i-1), so O_j holds PV_j(i-1).
                    sh = s.wait(j)
                    if cute.arch.vote_ballot_sync(scale < 1.0) != 0:
                        self.rescale_o(thr_pv, tO[j], tidx, scale)
                    sh.release()  # joins softmax[j]'s P_j release on the same barrier
            for j in cutlass.range_constexpr(Q_STAGE):
                with stats.consume(j) as st:
                    row_sum = st.data[tidx, 0]
                    row_max = st.data[tidx, 1]
                empty_row = row_sum == 0.0 or row_sum != row_sum  # noqa: PLR0124 (NaN)
                inv_sum = cute.arch.rcp_approx(row_sum if not empty_row else 1.0)
                # Final S_j commit: O_j holds the last PV. The epi handle is sO's stage j.
                with s.consume(j), epi.produce(j) as eo:
                    self.store_o_smem(
                        params.mma_pv, thr_pv, tO[j], tidx, inv_sum, eo.view(write=True)
                    )
                # FA4 work_q_block: Q tile j of the item, this CTA's 128 rows of it.
                row = ((m_blk * Q_STAGE + j) * self.cluster.size + rank) * M_BLOCK + tidx
                if row < params.seqlen_q:
                    lse = (
                        row_max * params.scale_log2 + cute.math.log2(row_sum, fastmath=True)
                    ) * LN2
                    params.mLSE[batch, head, row] = lse if not empty_row else -Float32.inf
            work.advance()
        return s, stats, epi, work

    def tile_row(self, me: Local, role_name: str):
        """This thread's row of the 128-row tile, its TMEM row (plain Python, inlined)."""
        if me.warps != 4:
            raise ProtocolError(
                f"{role_name}: a {M_BLOCK}-row tile is one row per thread of 4 warps, got "
                f"{me.warps} warps"
            )
        return me.tmem_row

    @cute.jit
    def rescale_o(self, thr_pv, tO, tidx, scale):
        """O_j *= scale for this thread's row, 16 columns at a time (FA4 correction_rescale)."""
        cols = 16
        mma_m = M_BLOCK * self.cluster.size
        tOcO = thr_pv.partition_C(cute.make_identity_tensor((mma_m, self.head_dim)))
        ld = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(cols)), Float32)
        st = cute.make_copy_atom(tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(cols)), Float32)
        tO_i = cute.composition(tO, cute.make_layout((M_BLOCK, cols)))
        tOcO_i = cute.composition(tOcO, cute.make_layout((M_BLOCK, cols)))
        thr_ld = tcgen05.make_tmem_copy(ld, tO_i).get_slice(tidx)
        thr_st = tcgen05.make_tmem_copy(st, tO_i).get_slice(tidx)
        tOtO_t2r = thr_ld.partition_S(tO_i)
        tOtO_r2t = thr_st.partition_D(tO_i)
        r_shape = thr_ld.partition_D(tOcO_i).shape
        for i in cutlass.range_constexpr(self.head_dim // cols):
            r = cute.make_rmem_tensor(r_shape, Float32)
            cute.copy(thr_ld, cute.make_tensor(tOtO_t2r.iterator + i * cols, tOtO_t2r.layout), r)
            for e in cutlass.range(0, cute.size(r), 2, unroll_full=True):
                r[e], r[e + 1] = cute.arch.mul_packed_f32x2((r[e], r[e + 1]), (scale, scale))
            cute.copy(thr_st, r, cute.make_tensor(tOtO_r2t.iterator + i * cols, tOtO_r2t.layout))

    @cute.jit
    def store_o_smem(self, mma_pv, thr_pv, tO, tidx, scale, sO):
        """sO = bf16(O_j * scale) (FA4 correction_epilogue, TMA-O path)."""
        cols = 16
        cs = self.cluster.size
        # sO is per-CTA sized: partition it with slice 0 of the MMA (FA4 does the same).
        tOsO = mma_pv.get_slice(0).partition_C(sO)
        tOcO = thr_pv.partition_C(cute.make_identity_tensor((M_BLOCK * cs, self.head_dim)))
        tO_i = cute.logical_divide(tO, cute.make_layout((M_BLOCK, cols)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((M_BLOCK, cols)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((M_BLOCK, cols)))
        ld_atom = sm100_utils.get_tmem_load_op(
            (M_BLOCK * cs, self.head_dim, self.tile_n),
            LayoutEnum.ROW_MAJOR,
            cutlass.BFloat16,
            Float32,
            (M_BLOCK, cols),
            use_2cta_instrs=cs > 1,
        )
        tiled_ld = tcgen05.make_tmem_copy(ld_atom, tO_i[(None, None), 0])
        thr_ld = tiled_ld.get_slice(tidx)
        st_atom = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, cutlass.BFloat16, Float32, tiled_ld
        )
        tiled_st = cute.make_tiled_copy_D(st_atom, tiled_ld)
        tOtO_t2r = thr_ld.partition_S(tO_i[(None, None), None])
        tOsO_r2s = copy_utils.partition_D_position_independent(thr_ld, tOsO_i[(None, None), None])
        tOcO_t2r = thr_ld.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range_constexpr(self.head_dim // cols):
            r = cute.make_rmem_tensor(tOcO_t2r[None, 0, 0, i].shape, Float32)
            cute.copy(tiled_ld, tOtO_t2r[None, 0, 0, i], r)
            for e in cutlass.range(0, cute.size(r), 2, unroll_full=True):
                r[e], r[e + 1] = cute.arch.mul_packed_f32x2((r[e], r[e + 1]), (scale, scale))
            copy_utils.cvt_copy(tiled_st, r, tOsO_r2s[None, 0, 0, i])

    @role(warps=13, regs=48)
    @cute.jit
    def epilogue(self, params: Fa4Params, *, epi: Consumer, work: WorkCursor):
        rank = self.cluster.rank()
        while work.valid:
            epi = self.epilogue_item(params, epi, rank, work.item)
            work.advance()
        return epi, work

    def epilogue_item(self, params: Fa4Params, epi, rank, item):
        """TMA-store one work item's O tiles from sO (plain Python: inlined into the role, so
        the loops below unroll at trace time). Returns the end for the caller's loop."""
        mma_m = M_BLOCK * self.cluster.size
        m_blk, head, batch = item
        gO = cute.local_tile(
            params.mO[None, None, head, batch], (Q_STAGE * mma_m, self.head_dim), (m_blk, 0)
        )
        gO = layout_utils.select(cute.flat_divide(gO, (mma_m,)), mode=[0, 2, 1])
        if cutlass.const_expr(self.cluster.size > 1):  # this CTA's 128 rows of each tile
            gO = cute.flat_divide(gO, (M_BLOCK,))[None, rank, None, None]
        store_o, _, _ = copy_utils.tma_get_copy_fn(
            params.tma_o, 0, cute.make_layout(1), params.sO, gO
        )
        eh = []
        for j in range(Q_STAGE):
            eh.append(epi.wait(j))
            eh[j].access(read=True, by="tma")  # store_o selects stage j itself
            store_o(src_idx=j, dst_idx=j)
            bulk_commit_group()
        for j in range(Q_STAGE):
            bulk_wait_group(Q_STAGE - 1 - j, read=True)  # sO stage j read: release it
            eh[j].release()
        return epi

    # ----------------------------------------------------------------- kernel

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
        scale_log2,
        sQ_layout,
        sK_layout,
        sV_layout,
        p_layout,
        sO_layout,
        bound,
    ):
        # K and V stages share one buffer: a tile_n x head_dim K tile and a head_dim x tile_n
        # V tile have the same bytes.
        layouts = {"O": sO_layout, "Q": sQ_layout, "K": sK_layout, "V": sV_layout}
        self.begin_roles(work=bound, layouts=layouts)
        sO, sQ, sK, sV = self.buffer("O"), self.buffer("Q"), self.buffer("K"), self.buffer("V")
        rank = self.cluster.rank()
        thr_qk, thr_pv = mma_qk.get_slice(rank), mma_pv.get_slice(rank)
        mma_m = M_BLOCK * self.cluster.size
        params = Fa4Params(
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
            sQ=sQ,
            sK=sK,
            sV=sV,
            sO=sO,
            p_layout=p_layout,
            scale_log2=scale_log2,
            s_layout=thr_qk.make_fragment_C(thr_qk.partition_shape_C((mma_m, self.tile_n))).layout,
            o_layout=thr_pv.make_fragment_C(
                thr_pv.partition_shape_C((mma_m, self.head_dim))
            ).layout,
            n_blocks=mK.shape[0] // self.tile_n,
            seqlen_q=mQ.shape[0],
        )
        self.run_roles(params)

    # ------------------------------------------------------------------- host

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        scale_log2: Float32,
        workspace: cute.Tensor,
        max_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        bf16 = cutlass.BFloat16
        # (b, s, h, d) -> (s, d, h, b); V -> (d, s, h, b) so it is the MN-major B operand of PV.
        mQ, mK, mV, mO = (
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
            for t in (q, k, v, o)
        )
        mV = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=[1, 0, 2, 3]))
        major = cute.nvgpu.OperandMajorMode
        group, mma_m = self.cluster.cta_group, M_BLOCK * self.cluster.size
        mma_qk = sm100_utils.make_trivial_tiled_mma(
            bf16, bf16, major.K, major.K, Float32, group, (mma_m, self.tile_n)
        )
        mma_pv = sm100_utils.make_trivial_tiled_mma(
            bf16,
            bf16,
            major.K,
            major.MN,
            Float32,
            group,
            (mma_m, self.head_dim),
            tcgen05.OperandSource.TMEM,
        )
        tiler_qk, tiler_pv = (
            (mma_m, self.tile_n, self.head_dim),
            (mma_m, self.head_dim, self.tile_n),
        )
        sQ_layout = sm100_utils.make_smem_layout_a(mma_qk, tiler_qk, bf16, Q_STAGE)
        sK_layout = sm100_utils.make_smem_layout_b(mma_qk, tiler_qk, bf16, self.kv_stages)
        p_layout = sm100_utils.make_smem_layout_a(mma_pv, tiler_pv, bf16, Q_STAGE)
        sV_layout = sm100_utils.make_smem_layout_b(mma_pv, tiler_pv, bf16, self.kv_stages)
        sO_layout = sm100_utils.make_smem_layout_epi(
            bf16, LayoutEnum.ROW_MAJOR, (M_BLOCK, self.head_dim), Q_STAGE
        )
        load_op = cpasync.CopyBulkTensorTileG2SOp(group)
        vmnk = self.cluster.layout_vmnk().shape
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_A(
            load_op, mQ, cute.select(sQ_layout, mode=[0, 1, 2]), tiler_qk, mma_qk, vmnk
        )
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_B(
            load_op, mK, cute.select(sK_layout, mode=[0, 1, 2]), tiler_qk, mma_qk, vmnk
        )
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_B(
            load_op, mV, cute.select(sV_layout, mode=[0, 1, 2]), tiler_pv, mma_pv, vmnk
        )
        tma_o, mO_t = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mO,
            cute.select(sO_layout, mode=[0, 1]),
            (M_BLOCK, self.head_dim),
        )
        m_blocks = cute.ceil_div(mQ.shape[0], Q_STAGE * mma_m)
        bound = self.work.bind((m_blocks, mQ.shape[2], mQ.shape[3]), workspace, max_ctas)
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
            lse,
            scale_log2,
            sQ_layout,
            sK_layout,
            sV_layout,
            p_layout,
            sO_layout,
            bound,
        ).launch(**self.launch_config(bound), stream=stream, min_blocks_per_mp=1)


@jit_cache(uncached=lambda op: op.trace_protocol)
def compile_fa4_fwd(op: FlashAttentionForward):
    """Compile once with dynamic batch, heads, and sequence lengths. Cached in memory and on
    disk under ``op.__cute_cache_key__()``."""
    b, h = cute.sym_int(), cute.sym_int()
    sq, sk = cute.sym_int(), cute.sym_int(divisibility=op.tile_n)
    bf16 = cutlass.BFloat16
    qo = (b, sq, h, op.head_dim)
    kv = (b, sk, h, op.head_dim)
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, qo),
        make_fake_compact_tensor(bf16, kv),
        make_fake_compact_tensor(bf16, kv),
        make_fake_compact_tensor(bf16, qo),
        make_fake_compact_tensor(Float32, (b, h, sq), assumed_align=4),
        Float32(1.0),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        cutlass.Int32(1),
        name=op.get_name(),
    )


def fa4_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    kv_stages: int | None = None,
    turnstile: bool = False,
    schedule=None,
    max_ctas: int | None = None,
    cluster=None,
    split_p: bool = False,
    roles: dict[str, RoleConfig] | None = None,
    mapping: WorkMap | None = None,
    use_ldred_rowmax: bool = False,
    tile_n: int = 128,
    rescale_threshold: float = 8.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention forward for ``(batch, seqlen, heads, head_dim)`` bf16 tensors, head_dim 64 or
    128; returns (O, LSE).

    ``LSE`` is ``(batch, heads, seqlen_q)`` fp32 (natural log), as in FA4. ``schedule``: work
    scheduling (default ``OneShot()``; ``StaticPersistent()``, ``AtomicPersistent()``,
    ``ClusterLaunchControl()`` run several (Q block, head, batch) items per CTA). ``max_ctas``
    caps the static/atomic persistent grid in CTAs (default: the SM count). ``cluster=CtaPair()``
    runs FA4's default 2-CTA schedule (512-row work items, 6 KV stages by default).
    ``mapping``: the order of the ``(query block, head, batch)`` items (default
    ``Raster(along="m")``; e.g. ``Raster(reverse="m")`` or ``Fa4Mapping(...)``).
    ``roles``: placement and register budgets by role name (``{"softmax": RoleConfig(...)}``)
    over the kernel's defaults. ``tile_n``: keys per K/V tile (64 or 128; ``seqlen_k`` must be
    a multiple). ``rescale_threshold``: FA4's lazy-rescale threshold in log2 units.
    ``use_ldred_rowmax`` (SM103-family targets) reduces unmasked score tiles' maxima during the
    TMEM load instead of in registers. Both paths keep the same rescale threshold and softmax
    arithmetic.
    """
    if any(t.dtype != torch.bfloat16 for t in (q, k, v)):
        raise TypeError("fa4_fwd expects bfloat16 q, k, v")
    if q.ndim != 4 or q.shape[-1] not in HEAD_DIMS or k.shape != v.shape:
        raise ValueError(
            f"expected (b, s, h, d) tensors with d in {HEAD_DIMS} and k.shape == v.shape"
        )
    if k.shape[0] != q.shape[0] or k.shape[2] != q.shape[2] or k.shape[3] != q.shape[3]:
        raise ValueError("q and k/v must share batch, head counts and head_dim (no GQA)")
    if k.shape[1] == 0 or k.shape[1] % tile_n:
        raise ValueError(f"seqlen_k must be a positive multiple of {tile_n} (no key masking)")
    if any(not t.is_contiguous() or t.data_ptr() % 16 for t in (q, k, v)):
        raise ValueError("q, k, v must be contiguous and 16-byte aligned")
    head_dim = q.shape[-1]
    scale = softmax_scale if softmax_scale is not None else head_dim**-0.5
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[0], q.shape[2], q.shape[1], device=q.device, dtype=torch.float32)
    roles_key = None if roles is None else tuple(sorted(roles.items()))
    key = (
        kv_stages,
        turnstile,
        schedule,
        cluster,
        split_p,
        roles_key,
        mapping,
        use_ldred_rowmax,
        tile_n,
        head_dim,
        rescale_threshold,
    )
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = FlashAttentionForward(
            kv_stages,
            turnstile,
            schedule=schedule,
            cluster=cluster,
            split_p=split_p,
            roles=roles,
            mapping=mapping,
            use_ldred_rowmax=use_ldred_rowmax,
            tile_n=tile_n,
            head_dim=head_dim,
            rescale_threshold=rescale_threshold,
        )
    fn = compile_fa4_fwd.by_static_key(key, lambda: compile_fa4_fwd(op))
    workspace, sms = op.work.launch_args(q.device)
    fn(q, k, v, out, lse, scale * LOG2_E, workspace, max_ctas or sms)
    return out, lse


_OPS: dict = {}
