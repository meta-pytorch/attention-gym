"""FlashAttention-4 backward (SM100) main kernels written with ``warpdrive``: the 1-CTA one
(``FlashAttentionBackward``, described here) and FA4's default 2-CTA one
(``FlashAttentionBackward2Cta``, ``fa4_bwd(..., cluster=CtaPair())``, described in its class).

Scope: bf16 Q/K/V/dO, fp32 accumulation, head_dim 128, 128 x 128 tiles, MHA (no GQA), no
varlen/paging/block sparsity/score mods, non-causal or causal (bottom-right aligned, as FA4).
Ragged sequence lengths are allowed: TMA zero-fills the loads of the last Q/K/V/dO tiles, keys
past ``seqlen_k`` are masked and the dK/dV stores are clipped. The pre- and post-processing
kernels (``dPsum = rowsum(dO * O)``, ``LSE * log2(e)``, zeroing and converting the fp32 dQ
accumulator) are FA4's own (``flash_attn.cute.interface._bwd_preprocess`` and
``_bwd_postprocess_convert``), reused as is; this file ports the main kernel,
``flash_attn/cute/flash_bwd_sm100.py`` at its 1-CTA, hdim-128 configuration (``split_P_dS``
off, ``Q_stage=2``, ``dO_stage=1``, registers 152/136/88, or 136/144/88 when causal). The
2-CTA kernel is a subclass with its own load and MMA schedules and a relay warp; the protocol
declaration (``__init__``, with ``pair`` branches where the pair's differs), the MMA issue
helpers, the compute warps, the epilogue and the dQ reduce are this class's.

A work item is one 128-key K/V tile ``(n_block, head, batch)`` (FA4's ``SingleTileScheduler``
order, which ``Raster`` over ``(n_blocks, heads, batch)`` reproduces). The CTA keeps K, V, dK and
dV for the item and loops over the Q tiles that see its keys. Per Q tile t:

    S^T = K Q_t^T          P^T = exp2(S^T * scale_log2 - LSE_t)     (in TMEM over S)
    dP^T = V dO_t^T        dS^T = P^T * (dP^T - dPsum_t)             (in TMEM over dP, and SMEM)
    dV += P^T dO_t         dK += dS^T Q_t          dQ_t = dS K        (TMA reduce-add into fp32)

Roles (FA4's warp layout; 14 warps, padded to 16 for the register budgets):

    warps 0-3   reduce   dQ_t: TMEM -> SMEM -> cp.reduce.async.bulk.add.f32 into dQaccum
    warps 4-11  compute  P and dS for each tile; the dK/dV epilogue (TMEM -> SMEM -> TMA store)
    warp 12     mma      the five GEMMs, software-pipelined as FA4's 1-CTA schedule
    warp 13     load     K, V once; Q, LSE, dO, dPsum per Q tile

Channels (all mbarriers; epochs per item with n Q tiles). Thread sides are ``Warps``: every
thread signals, one elected lane per warp arrives, as FA4's pipelines count:

    load --k, v (TMA, 1 stage)--> mma                  1 each; held for the whole item
    load --q (TMA, 2 stages)--> mma                    n; Q_t released after dK += dS_t^T Q_t
    load --do (TMA, 1 stage)--> mma                    n; released after dV += P_t^T dO_t
    load --lse (bulk copy, 2) / dpsum (1)--> compute   n each
    mma --s (UMMA)--> compute                          n; empty = "P_t stored over S_t"
    mma --dp (UMMA)--> compute                         n; empty = "dS_t stored over dP_t"
    compute --ds--> mma                                n; dS_t in sdS; empty = dQ_t MMA done
    mma --dq (UMMA)--> reduce                          n; empty = dQ_t read out of TMEM
    mma --dkv[0] dV, dkv[1] dK (UMMA)--> compute       1 each, after the last dV / dK MMA

Every ring starts empty, so the MMA's first acquires of s, dp, dq and dkv pass, and the load
runs ahead by the q/do stage counts. Per tile t the MMA warp issues dP_t, dV_t, S_{t+1}, dK_t,
dQ_t, FA4's order. Q_t is read by S_t (issued during tile t-1) and by dK_t, so S_{t+1} reads
the q ring's next stage through ``q.peek()``, a read-only preview that the next tile's
``q.wait()`` handle owns and releases (the last tile is peeled, so the loop's final preview
has a wait after it). With one Q stage that peek would wait for the release of Q_t, which
comes after S_{t+1}: a deadlock, so ``Q_STAGE`` is fixed at 2. The last tile's s acquire is
retired (there is no S_{t+1}), which is why the kernel runs one item per CTA, as FA4's 1-CTA
kernel does.

TMEM (512 columns): S [0,128), dV [128,256), dP [256,384), dK [384,512). P (bf16 pairs) aliases
S's first 64 columns, dS aliases dP's first 64, and dQ aliases all of dP (FA4's 1-CTA map).
The MMA warp holds the dq handle from before dP_t (which overwrites dQ_{t-1}'s columns) until
after dQ_t, and the s handle for tile t+1 from before dV_t (which reads P_t) until after S_{t+1}:
overlapping handle lifetimes, not nested ``with`` blocks. An MMA's TMEM read and a later MMA's
write from the same thread are pipelined (PTX ISA 9.7.18.6.2), so P_t is read by dV_t before
S_{t+1} overwrites it, and dS_t by dK_t before dQ_t overwrites it. The compute warpgroups
share every TMEM lane (each takes alternate 32-column groups), so a named barrier over both
separates the S loads from the P stores and each dP load from the dS stores.

SMEM is an ``SmemPlan``: Q (2 stages) with its MN-major view Qt, K with Kt, V, dO with dOt, dS,
the dQ staging buffer (``dq_stages``), LSE/dPsum rows, and the dK/dV epilogue tiles over Q
and dO (FA4 reuses those bytes once the last MMA reading them has committed, which again needs
one work item per CTA). 231 KB of the 232 KB limit.
"""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from flash_attn.cute import blackwell_helpers as fa_mma
from flash_attn.cute import copy_utils as fa_copy_utils
from flash_attn.cute import mask as fa_mask
from flash_attn.cute import utils as fa_utils

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.compat import LayoutEnum
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Barriers,
    Consumer,
    CtaPair,
    Local,
    NamedBarrier,
    OneShot,
    Producer,
    ProtocolError,
    Raster,
    Ring,
    RoleConfig,
    Slots,
    SmemPlan,
    StageView,
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
from attn_gym.warpdrive.examples.tma_tiles import tma_partition_tiles

TILE_M = 128  # Q rows per tile (the MMA N of S^T and dP^T)
TILE_N = 128  # keys per work item (the MMA M; one compute thread pair per key row)
HEAD_DIM = 128
# S_{t+1} reads Q_{t+1} before dK_t releases Q_t, so one Q stage deadlocks this schedule.
Q_STAGE = 2
DQ_COLS = 32  # dQ columns per TMA reduce-add (FA4's dQ_reduce_ncol at 1-CTA)
DQ_T2R_COLS = 32  # dQ columns per TMEM load in the reduce warps
# FA4's 2-CTA dQ reduce: 8 columns per reduce-add (dQ_reduce_ncol), 4 staging buffers.
PAIR_DQ_COLS, PAIR_DQ_STAGES = 8, 4
XCHG_ELEMS = TILE_N * (TILE_M // 2)  # the exchanged dS half: 128 keys x 64 queries
XCHG_BYTES = XCHG_ELEMS * 2  # bf16
EPI_COLS = 64  # dK/dV columns per compute warpgroup (one 128-byte-row TMA store box)
LOG2_E = math.log2(math.e)


@dataclass(frozen=True)
class BwdParams:
    """Kernel-prologue values the roles read; built once in ``FlashAttentionBackward.kernel``."""

    mma_s: cute.TiledMma  # S^T = K Q^T
    mma_dp: cute.TiledMma  # dP^T = V dO^T
    mma_dv: cute.TiledMma  # dV += P^T dO, P from TMEM
    mma_dk: cute.TiledMma  # dK += dS^T Q, dS from TMEM
    mma_dq: cute.TiledMma  # dQ = dS K, dS from SMEM
    tma_q: cute.CopyAtom
    tma_k: cute.CopyAtom
    tma_v: cute.CopyAtom
    tma_do: cute.CopyAtom
    tma_dk: cute.CopyAtom
    tma_dv: cute.CopyAtom
    mQ: cute.Tensor  # (s, d, h, b) TMA views; dO is (d, s, h, b)
    mK: cute.Tensor
    mV: cute.Tensor
    mdO: cute.Tensor
    mdK: cute.Tensor
    mdV: cute.Tensor
    mLSE: cute.Tensor  # (s_rounded, h, b): LSE * log2(e), +inf past seqlen_q
    mdPsum: cute.Tensor  # (s_rounded, h, b)
    mdQacc: cute.Tensor  # (s_rounded * d, h, b), FA4's TMEM_FRAG_1CTA tile order
    sQ: cute.Tensor
    sQt: cute.Tensor
    sK: cute.Tensor
    sKt: cute.Tensor
    sV: cute.Tensor
    sdO: cute.Tensor
    sdOt: cute.Tensor
    sLSE: cute.Tensor
    sdPsum: cute.Tensor
    sdQacc: cute.Tensor
    sdK: cute.Tensor  # epilogue tiles (over Q / dO; over K / V with a CTA pair)
    sdV: cute.Tensor
    p_layout: cute.ComposedLayout  # P / dS as TMEM A operands
    ds_layout: cute.ComposedLayout
    softmax_scale: Float32
    scale_log2: Float32
    seqlen_q: Int32
    seqlen_k: Int32
    m_blocks: Int32
    # CTA pair only: the separately loaded Qt / dOt / Kt operands.
    tma_qt: cute.CopyAtom | None = None
    tma_dot: cute.CopyAtom | None = None
    tma_kt: cute.CopyAtom | None = None
    mQt: cute.Tensor | None = None  # (d, s, h, b); dOt is (s, d, h, b), Kt (d, s_k, h, b)
    mdOt: cute.Tensor | None = None
    mKt: cute.Tensor | None = None


def cols(warp: int, st: int, width: int) -> range:
    """Columns of stage ``st`` of a compute warp's ``width``-column S/dP loads or P/dS stores,
    for the alias diagnostic: the warpgroups interleave, warps 8-11 taking the first ``width``
    columns of each pair of stages."""
    first = 2 * width * st + width * (warp < 8)
    return range(first, first + width)


def half_cols(t, wg):
    """Warpgroup ``wg``'s half of mode 1 of a TMEM copy partition (FA4's split_wg)."""
    shape = cute.product_each(t.shape)
    t = cute.logical_divide(t, (shape[0], shape[1] // 2))
    return t[(None, (None, wg)) + (None,) * (cute.rank(t) - 2)]


def umma(tiled_mma, acc, a, b, accumulate):
    """Issue one tile MMA as K-block ``tcgen05.mma`` instructions; ``accumulate`` may be
    dynamic (plain Python: inlined)."""
    atom = cute.make_mma_atom(tiled_mma.op)
    for k in range(cute.size(a.shape[2])):
        atom.set(tcgen05.Field.ACCUMULATE, accumulate if k == 0 else True)
        cute.gemm(atom, acc, a[None, None, k], b[None, None, k], acc)


class FlashAttentionBackward(WarpSpecialized):
    def __init__(
        self,
        causal: bool = False,
        roles: dict[str, RoleConfig] | None = None,
        dq_stages: int | None = None,
        trace: bool = False,
    ):
        pair = self.cluster.size > 1  # the class's cluster (CtaPair on the subclass)
        self.causal = causal
        self.dq_stages = check_dq_stages(dq_stages, pair)
        # Q stages and dQ columns per reduce-add (the pair: FA4's 2-CTA Q_stage, dQ_reduce_ncol).
        self.q_stages, self.dq_cols = (1, PAIR_DQ_COLS) if pair else (Q_STAGE, DQ_COLS)
        if pair:
            # FA4's 2-CTA register split, causal or not: reduce and compute 136 (compute's @role
            # default), the rest 104 (declared by the pair's load, mma and relay roles).
            defaults = {"reduce": RoleConfig(range(4), regs=136)}
        elif causal:
            # FA4's causal split moves 8 registers from the dQ reduce to the compute warps.
            defaults = {
                "reduce": RoleConfig(range(4), regs=136),
                "compute": RoleConfig(range(4, 12), regs=144),
            }
        else:
            defaults = {}
        self.roles = {**defaults, **(roles or {})}  # by role name, over the @role defaults
        bf16 = cutlass.BFloat16
        self.smem_plan = plan = SmemPlan()
        lse_buf = plan.buffer("LSE", Float32, align=128)
        dpsum_buf = plan.buffer("dPsum", Float32, align=128)
        # One CTA reads Q, K and dO MN-major in place (Qt, Kt, dOt); the pair splits those B
        # operands along other dimensions than Q, K and dO, so it loads them separately.
        q_buf = plan.buffer("Q", bf16, align=1024)
        qt_buf = plan.buffer("Qt", bf16, align=1024, overlaps=None if pair else q_buf)
        k_buf = plan.buffer("K", bf16, align=1024)
        kt_buf = plan.buffer("Kt", bf16, align=1024, overlaps=None if pair else k_buf)
        v_buf = plan.buffer("V", bf16, align=1024)
        do_buf = plan.buffer("dO", bf16, align=1024)
        dot_buf = plan.buffer("dOt", bf16, align=1024, overlaps=None if pair else do_buf)
        ds_buf = plan.buffer("dS", bf16, align=1024)
        if pair:
            dsx_buf = plan.buffer("dSx", bf16, align=1024)  # the peer's half of dS, staged
        plan.buffer("dQacc", Float32, align=1024)
        # FA4's epilogue tiles reuse operands that are dead once dkv publishes dK / dV, and one
        # item per CTA means no later load refills them: dK over Q and dV over dO (read last by
        # the dK / dV MMAs), or in the pair over K and V (read last by the S / dP MMAs).
        plan.buffer("dKepi", bf16, align=1024, overlaps=k_buf if pair else q_buf)
        plan.buffer("dVepi", bf16, align=1024, overlaps=v_buf if pair else do_buf)
        tile_bytes = TILE_M * HEAD_DIM * 2
        if pair:
            # Per CTA: its 128 keys of K and V, half the 256-key Kt, and 64 queries of each of
            # Q, Qt, dO and dOt (dO and dOt share a stage).
            self.k = Ring(1, Tma(tile_bytes), Umma, buffer=k_buf)
            self.kt = Ring(1, Tma(tile_bytes), Umma, buffer=kt_buf)
            self.v = Ring(1, Tma(tile_bytes), Umma, buffer=v_buf)
            self.q = Ring(1, Tma(tile_bytes // 2), Umma, buffer=q_buf)
            self.qt = Ring(1, Tma(tile_bytes // 2), Umma, buffer=qt_buf)
            self.do = Ring(1, Tma(tile_bytes), Umma, buffer={"dO": do_buf, "dOt": dot_buf})
        else:
            self.k = Ring(1, Tma(tile_bytes), Umma, buffer={"K": k_buf, "Kt": kt_buf})
            self.v = Ring(1, Tma(tile_bytes), Umma, buffer=v_buf)
            self.q = Ring(Q_STAGE, Tma(tile_bytes), Umma, buffer={"Q": q_buf, "Qt": qt_buf})
            self.do = Ring(1, Tma(tile_bytes), Umma, buffer={"dO": do_buf, "dOt": dot_buf})
        # LSE/dPsum rows are read with ordinary loads and refilled by the async proxy, so their
        # releases take the derived smem_async fence. The pair's dpsum opts out: make_ds fences
        # (fence.proxy.async) and syncs every compute thread after its last dPsum read, before
        # the release.
        self.lse = Ring(self.q_stages, Tma(TILE_M * 4), Warps, buffer=lse_buf)
        dpsum_fence = () if pair else None
        self.dpsum = Ring(1, Tma(TILE_M * 4), Warps, buffer=dpsum_buf, release_fence=dpsum_fence)
        self.s = Ring(1, Umma, Warps, release_fence="tmem_store")  # P stored over S
        self.dp = Ring(1, Umma, Warps, release_fence="tmem_store")  # dS stored over dP
        if pair:
            # ds hands over sdS and sdSx (dS in TMEM reaches the dK MMA through dp). Its commits
            # need no fence: make_ds fences the sdS stores before the exchange copy, and compute
            # waits for its S loads (tcgen05.wait::ld) itself.
            self.ds = Ring(1, Warps, Umma, buffer={"dS": ds_buf, "dSx": dsx_buf})
        else:
            # dS is handed over both in TMEM (tcgen05.st over dP) and in SMEM (sdS): both fences.
            self.ds = Ring(
                1, Warps, Umma, buffer=ds_buf, commit_fence=("tmem_store", "smem_async")
            )
        self.dq = Ring(1, Umma, Warps)
        self.dkv = Slots(2, Umma, Warps)  # slot 0: dV, slot 1: dK
        if pair:
            # The dS exchange (raw: warpdrive has no channel one CTA fills for its peer). The
            # peer's compute thread 0 arms it with the copy's bytes; the relay warp waits on it.
            self.xchg = Barriers(1, counts=1)
        # Syncs inside a role: the 8 compute warps (P over S, dS over dP), the 4 reduce warps
        # (the dQ staging buffer), and each compute warpgroup's dK/dV epilogue tile.
        self.compute_sync = NamedBarrier(threads=256)
        self.reduce_sync = NamedBarrier(threads=128)
        self.epi_sync_wg0 = NamedBarrier(threads=128)
        self.epi_sync_wg1 = NamedBarrier(threads=128)
        tmem = (
            TmemPlan()
            .region("S", TILE_M)
            .region("dV", HEAD_DIM)
            .region("dP", TILE_M)
            .region("dK", HEAD_DIM)
            .alias("P", of="S", offset=0, cols=TILE_M // 2)
            .alias("dS", of="dP", offset=0, cols=TILE_M // 2)
        )
        if pair:  # this CTA's 64 dQ columns, over S's tail (FA4's 2-CTA map)
            self.tmem_plan = tmem.alias("dQ", of="S", offset=TILE_M // 2, cols=HEAD_DIM // 2)
        else:
            self.tmem_plan = tmem.alias("dQ", of="dP", offset=0, cols=HEAD_DIM)
        self.work = WorkStream(mapping=Raster(), schedule=OneShot())
        self.trace_protocol = trace

    def get_name(self) -> str:
        pair = self.cluster.size > 1
        suffix = ("_pair" if pair else "") + ("_causal" if self.causal else "")
        if not pair and self.dq_stages != 2:
            suffix += f"_dq{self.dq_stages}"
        return f"warpdrive_fa4_bwd{suffix}_{self.work.get_key()}{self.trace_key()}"

    @cute.jit
    def m_block_range(self, params: BwdParams, n_block):
        """The Q tiles that see key tile ``n_block`` (``TILE_N`` keys per CTA of the cluster;
        causal: bottom-right aligned)."""
        m_lo = Int32(0)
        if cutlass.const_expr(self.causal):
            first_q = n_block * TILE_N * self.cluster.size + params.seqlen_q - params.seqlen_k
            m_lo = cutlass.max(first_q, 0) // TILE_M
        return m_lo, params.m_blocks

    # ------------------------------------------------------------------ roles

    @role(warps=13, regs=88)
    @cute.jit
    def load(
        self,
        params: BwdParams,
        *,
        k: Producer,
        v: Producer,
        q: Producer,
        do: Producer,
        lse: Producer,
        dpsum: Producer,
        work: WorkCursor,
    ):
        thr_s, thr_dp = params.mma_s.get_slice(0), params.mma_dp.get_slice(0)
        thr_dv = params.mma_dv.get_slice(0)
        copy_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
        while work.valid:
            n_block, head, batch = work.item
            m_lo, m_hi = self.m_block_range(params, n_block)
            gK = cute.local_tile(
                params.mK[None, None, head, batch], (TILE_N, HEAD_DIM), (n_block, 0)
            )
            gV = cute.local_tile(
                params.mV[None, None, head, batch], (TILE_N, HEAD_DIM), (n_block, 0)
            )
            gQ = cute.local_tile(params.mQ[None, None, head, batch], (TILE_M, HEAD_DIM), (None, 0))
            gdO = cute.local_tile(
                params.mdO[None, None, head, batch], (HEAD_DIM, TILE_M), (0, None)
            )
            gLSE = cute.local_tile(params.mLSE[None, head, batch], (TILE_M,), (None,))
            gdPsum = cute.local_tile(params.mdPsum[None, head, batch], (TILE_M,), (None,))
            tKsK, tKgK = tma_partition_tiles(params.tma_k, params.sK, thr_s.partition_A(gK))
            tVsV, tVgV = tma_partition_tiles(params.tma_v, params.sV, thr_dp.partition_A(gV))
            tQsQ, tQgQ = tma_partition_tiles(params.tma_q, params.sQ, thr_s.partition_B(gQ))
            tOsO, tOgO = tma_partition_tiles(params.tma_do, params.sdO, thr_dv.partition_B(gdO))
            k, v = k.views(K=StageView(tKsK, axis=1)), v.views(V=StageView(tVsV, axis=1))
            q, do = q.views(Q=StageView(tQsQ, axis=1)), do.views(dO=StageView(tOsO, axis=1))
            # FA4's order: K, Q_0, LSE_0, V, dO_0, dPsum_0, then (Q, LSE, dO, dPsum) per tile.
            with k.produce() as h:
                self.tma_load(params.tma_k, tKgK, h, "K")
            for m in cutlass.range(m_lo, m_hi, unroll=1):
                with q.produce() as h:
                    self.tma_load(params.tma_q, tQgQ[None, m], h, "Q")
                with lse.produce() as h, cute.arch.elect_one():
                    self.bulk_load(copy_stats, gLSE[None, m], h)
                if m == m_lo:
                    with v.produce() as h:
                        self.tma_load(params.tma_v, tVgV, h, "V")
                with do.produce() as h:
                    self.tma_load(params.tma_do, tOgO[None, m], h, "dO")
                with dpsum.produce() as h, cute.arch.elect_one():
                    self.bulk_load(copy_stats, gdPsum[None, m], h)
            work.advance()
        return k, v, q, do, lse, dpsum, work

    def tma_load(self, atom, src, h, name):
        """One TMA tile load into the stage of the open producer handle ``h`` (plain Python)."""
        cute.copy(atom, src, h.view(name, write=True, by="tma"), tma_bar_ptr=h.barrier)

    def bulk_load(self, atom, src, h):
        """One LSE / dPsum row by bulk copy into the stage of ``h`` (plain Python)."""
        cute.copy(atom, src, h.view(write=True, by="tma"), mbar_ptr=h.barrier)

    @role(warps=12, regs=88)
    @cute.jit
    def mma(
        self,
        params: BwdParams,
        *,
        k: Consumer,
        v: Consumer,
        q: Consumer,
        do: Consumer,
        s: Producer,
        dp: Producer,
        ds: Consumer,
        dq: Producer,
        dkv: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        t = self.mma_views(params, tmem)
        while work.valid:
            n_block, _, _ = work.item
            m_lo, m_hi = self.m_block_range(params, n_block)
            kh = k.wait()
            dkv_h = (dkv.acquire(0), dkv.acquire(1))  # dV, dK of this item
            with s.produce():  # S_0 = K Q_0^T; Q_0 is owned and released by tile 0's q.wait()
                self.issue_s(params, q.peek(), kh, t, tmem)
            vh = v.wait()  # V lands after Q_0: S_0 does not wait for it
            # FA4's 1-CTA order, rotated so each handle closes in the iteration that opened it:
            # tile t issues dP_t, dV_t, S_{t+1}, dK_t, dQ_t.
            for i in cutlass.range(m_hi - m_lo - 1, unroll=1):
                q, do, s, dp, ds, dq = self.mma_tile(
                    params, q, do, s, dp, ds, dq, kh, vh, t, tmem, i == 0
                )
            q, do, s, dp, ds, dq = self.mma_tile(
                params, q, do, s, dp, ds, dq, kh, vh, t, tmem, m_hi - m_lo == 1, dkv_h
            )
            kh.release()
            vh.release()
            work.advance()
        return k, v, q, do, s, dp, ds, dq, dkv, work

    def mma_views(self, params: BwdParams, tmem):
        """The MMA warp's slices, TMEM accumulators and TMEM P / dS A operands (plain Python).
        The pair leader (rank 0) issues every MMA, so the slices are rank 0's."""
        thr = SimpleNamespace(
            s=params.mma_s.get_slice(0),
            dp=params.mma_dp.get_slice(0),
            dv=params.mma_dv.get_slice(0),
            dk=params.mma_dk.get_slice(0),
            dq=params.mma_dq.get_slice(0),
        )
        # P and dS as TMEM A operands: bf16 views of the S / dP columns (2 elements per column).
        rP = thr.dv.make_fragment_A(cute.make_tensor(tmem.base, params.p_layout.outer))
        rP = rP[None, None, None, 0]
        rdS = thr.dk.make_fragment_A(cute.make_tensor(tmem.base, params.ds_layout.outer))
        rdS = rdS[None, None, None, 0]
        keys = TILE_N * self.cluster.size  # the MMA M of S^T, dP^T, dV and dK
        return SimpleNamespace(
            thr=thr,
            tS=self.acc(thr.s, tmem, "S", (keys, TILE_M)),
            tdP=self.acc(thr.dp, tmem, "dP", (keys, TILE_M)),
            tdV=self.acc(thr.dv, tmem, "dV", (keys, HEAD_DIM)),
            tdK=self.acc(thr.dk, tmem, "dK", (keys, HEAD_DIM)),
            tdQ=self.acc(thr.dq, tmem, "dQ", (TILE_M, HEAD_DIM)),
            rP=cute.make_tensor(rP.iterator + tmem.cols("P") * 2, rP.layout),
            rdS=cute.make_tensor(rdS.iterator + tmem.cols("dS") * 2, rdS.layout),
        )

    def smem_mma(self, tiled_mma, thr, acc, a, b, zero_init, ta=None):
        """``acc (+)= a b`` with SMEM operands given as stage views (``a``: a TMEM fragment
        instead when ``ta``, its TMEM address, is given). FA4's ``gemm_ptx_w_idx``, which
        issues the K blocks from precomputed descriptors (plain Python: inlined)."""
        rA = a if ta is not None else thr.make_fragment_A(a)
        sA = None if ta is not None else a
        rB = thr.make_fragment_B(b)
        fa_mma.gemm_ptx_w_idx(
            tiled_mma,
            acc,
            rA,
            rB,
            sA=sA,
            sB=b,
            zero_init=zero_init,
            tA_addr=ta,
            cta_group=self.cluster.size,
        )

    # The five MMAs (plain Python: inlined). ``qh``, ``doh``: the handle (or ``peek`` preview)
    # of the stage read; ``first``: the item's first tile zero-initializes dV and dK.

    def issue_s(self, params: BwdParams, qh, kh, t, tmem):
        """S^T = K Q^T."""
        tmem.access("S", write=True, by="umma")
        sK, sQ = kh.view("K", read=True, by="umma"), qh.view("Q", read=True, by="umma")
        self.smem_mma(params.mma_s, t.thr.s, t.tS, sK, sQ, True)

    def issue_dp(self, params: BwdParams, doh, vh, t, tmem):
        """dP^T = V dO^T."""
        tmem.access("dP", write=True, by="umma")
        sV, sdOt = vh.view("V", read=True, by="umma"), doh.view("dOt", read=True, by="umma")
        self.smem_mma(params.mma_dp, t.thr.dp, t.tdP, sV, sdOt, True)

    def issue_dv(self, params: BwdParams, doh, t, tmem, first):
        """dV += P^T dO, P from TMEM."""
        tmem.access("P", read=True, by="umma")
        tmem.access("dV", write=True, by="umma")
        sdO = doh.view("dO", read=True, by="umma")
        self.smem_mma(params.mma_dv, t.thr.dv, t.tdV, t.rP, sdO, first, tmem.ptr("P").toint())

    def issue_dk(self, params: BwdParams, qth, t, tmem, first):
        """dK += dS^T Q, dS from TMEM; ``qth`` holds Qt."""
        tmem.access("dS", read=True, by="umma")
        tmem.access("dK", write=True, by="umma")
        sQt = qth.view("Qt", read=True, by="umma")
        self.smem_mma(params.mma_dk, t.thr.dk, t.tdK, t.rdS, sQt, first, tmem.ptr("dS").toint())

    def issue_dq(self, params: BwdParams, dsh, kth, t, tmem):
        """dQ = dS K from sdS and Kt."""
        tmem.access("dQ", write=True, by="umma")
        sdS, sKt = dsh.view("dS", read=True, by="umma"), kth.view("Kt", read=True, by="umma")
        rdS, rKt = t.thr.dq.make_fragment_A(sdS), t.thr.dq.make_fragment_B(sKt)
        umma(params.mma_dq, t.tdQ, rdS, rKt, False)

    def mma_tile(
        self, params: BwdParams, q, do, s, dp, ds, dq, kh, vh, t, tmem, first, dkv_h=None
    ):
        """One Q tile t: dP_t, dV_t, S_{t+1}, dK_t, dQ_t. The last tile (``dkv_h``, the item's
        dV and dK handles) has no S_{t+1} and publishes dV and dK as soon as their last MMA is
        issued. Returns the ends (plain Python: inlined into the role)."""
        last = dkv_h is not None
        qh, doh = q.wait(), do.wait()
        dqh = dq.acquire()  # dP_t overwrites dQ_{t-1}'s columns: the reduce warps have read them
        with dp.produce():  # dS_{t-1} was stored over dP_{t-1}
            self.issue_dp(params, doh, vh, t, tmem)
        sh = s.acquire()  # P_t stored over S_t
        self.issue_dv(params, doh, t, tmem, first)
        doh.release()
        if last:
            dkv_h[0].commit()  # dV final: the commit covers every MMA issued so far
            sh.retire()  # no S_{t+1}: one item per CTA, so nothing consumes this stage
        else:
            # S_{t+1} previews Q_{t+1} (tile t+1's q.wait() owns and releases it) and overwrites
            # P_t behind the dV MMA that read it (same-thread MMA order).
            self.issue_s(params, q.peek(), kh, t, tmem)
            sh.commit()
        with ds.consume() as dsh:  # dS_t in TMEM (over dP_t) and in sdS
            self.issue_dk(params, qh, t, tmem, first)
            qh.release()
            if last:
                dkv_h[1].commit()  # dK final
            self.issue_dq(params, dsh, kh, t, tmem)
            dqh.commit()
        return q, do, s, dp, ds, dq

    def acc(self, thr_mma, tmem, name, shape):
        """The fp32 TMEM accumulator ``name`` for an MMA tile of ``shape`` (plain Python)."""
        layout = thr_mma.make_fragment_C(thr_mma.partition_shape_C(shape)).layout
        return cute.make_tensor(tmem.ptr(name), layout)

    @role(warps=range(4, 12), regs=136)
    @cute.jit
    def compute(
        self,
        params: BwdParams,
        *,
        lse: Consumer,
        dpsum: Consumer,
        s: Consumer,
        dp: Consumer,
        ds: Producer,
        dkv: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        pair = self.cluster.size > 1
        c = self.compute_views(params, tmem, me)
        lse, dpsum = lse.views(LSE=c.lse_view), dpsum.views(dPsum=c.dpsum_view)
        while work.valid:
            n_block, head, batch = work.item
            m_lo, m_hi = self.m_block_range(params, n_block)
            key_oob = c.key >= params.seqlen_k - n_block * TILE_N * self.cluster.size
            for m in cutlass.range(m_lo, m_hi, unroll=1):
                rS = cute.make_rmem_tensor(c.tScS_t2r.shape, Float32)
                lse, s, lh, sh = self.load_s(c, lse, s, tmem, rS)
                if cutlass.const_expr(pair):
                    if m > m_lo:
                        # dQ_{m-1} overwrites the tail of S_m: dS_{m-1} is handed over once S_m
                        # is read.
                        cute.arch.fence_view_async_tmem_load()
                        self.signal_ds(ds)
                self.store_p(c, params, rS, lh, sh, tmem, n_block, m, key_oob)
                dpsum, dp, ds, dsh = self.make_ds(c, rS, dpsum, dp, ds, tmem)
                if cutlass.const_expr(pair):
                    # The pair's commit is deferred to the next tile's signal_ds. A handle must
                    # close in the iteration that opened it, and rotating the loop instead keeps
                    # P_m in registers across the back edge (causal main kernel 1.5-2.4% slower
                    # on GB300, November 2026; docs/warpdrive/examples.md#attention-backward).
                    dsh.commit(signal=False)
                else:
                    dsh.commit()  # layer fences the dS tcgen05.st and sdS stores for the dQ MMA
            if cutlass.const_expr(pair):
                self.signal_ds(ds)  # the last tile's dS: no S_{m+1} to read first
            dkv = self.epilogue(params, dkv, tmem, c.tidx, n_block, head, batch)
            work.advance()
        return lse, dpsum, s, dp, ds, dkv, work

    def compute_views(self, params: BwdParams, tmem, me: Local):
        """The compute warps' partitions of S, P, dP and dS in TMEM, of the LSE / dPsum rows and
        of sdS (plain Python: inlined into the role, from converged code)."""
        if me.warps != 8:
            raise ProtocolError(
                f"compute: the TMEM copies span two warpgroups, got {me.warps} warps"
            )
        # Both warpgroups work on every tile: each thread holds 64 of a key row's 128 columns
        # (FA4's make_tmem_copy with two warpgroups). By default tidx 0-127 are warps 8-11
        # (FA4's wg 0).
        tidx = me.tmem_row
        rank = self.cluster.rank()  # this CTA's TILE_N rows of the pair's S^T / dP^T
        tile = (TILE_N * self.cluster.size, TILE_M)
        thr_s, thr_dp = params.mma_s.get_slice(rank), params.mma_dp.get_slice(rank)
        tStS = self.acc(thr_s, tmem, "S", tile)
        tdPtdP = self.acc(thr_dp, tmem, "dP", tile)
        p_cols = TILE_M // 2  # P / dS: bf16 pairs in half the S / dP columns
        half = (cute.make_layout((TILE_N, p_cols)), 1, 1)
        tStP = cute.make_tensor(tmem.ptr("P"), cute.composition(tStS, half).layout)
        tdPtdS = cute.make_tensor(tmem.ptr("dS"), cute.composition(tdPtdP, half).layout)
        tScS = thr_s.partition_C(cute.make_identity_tensor(tile))
        tScP = cute.composition(tScS, half)
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        st_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_ld = fa_copy_utils.make_tmem_copy(ld_atom, 2).get_slice(tidx)
        thr_st = fa_copy_utils.make_tmem_copy(st_atom, 2).get_slice(tidx)
        tScS_t2r = thr_ld.partition_D(tScS)

        # LSE / dPsum broadcast along the key rows: (keys, tile_m, stage) views of the rows.
        def stats(buf, stages):
            t = cute.make_tensor(
                buf.iterator, cute.make_layout((tile[0], TILE_M, stages), stride=(0, 1, TILE_M))
            )
            return thr_ld.partition_D(thr_s.partition_C(t))  # ((32, 1), 2, 1, 1, stage)

        # dS also goes to SMEM for the dQ MMA, as the row-major (key, query) epilogue tile:
        # the dQ MMA's M-major A layout with the same 128-byte swizzle. Its query halves are
        # 64-column blocks, one per tScS_t2r stage.
        r2s_atom = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, cutlass.BFloat16, Float32, thr_ld
        )
        epi = sm100_utils.make_smem_layout_epi(
            cutlass.BFloat16, LayoutEnum.ROW_MAJOR, (TILE_N, TILE_M), 1
        )
        sdS_2d = cute.slice_(epi.outer, (None, None, 0))
        return SimpleNamespace(
            tidx=tidx,
            rank=rank,
            thr_ld=thr_ld,
            thr_st=thr_st,
            thr_r2s=cute.make_tiled_copy_D(r2s_atom, thr_ld).get_slice(tidx),
            tStS_t2r=thr_ld.partition_S(tStS),  # (((32, 32), 1), 2, 1, 1): two 32-column stages
            tdPtdP_t2r=thr_ld.partition_S(tdPtdP),
            tScS_t2r=tScS_t2r,
            tStP_r2t=thr_st.partition_D(tStP),
            tdPtdS_r2t=thr_st.partition_D(tdPtdS),
            rP_shape=thr_st.partition_S(tScP).shape,
            lse_view=StageView(stats(params.sLSE, self.q_stages), axis=4),
            dpsum_view=StageView(stats(params.sdPsum, 1), axis=4),
            sdS_epi=cute.make_layout((sdS_2d.shape,), stride=(sdS_2d.stride,)),
            key=tScS_t2r[0][0],  # this thread's key row within the (cluster's) key tile
            n_stages=cute.size(tScS_t2r, mode=[1]),
        )

    def load_s(self, c, lse, s, tmem, rS):
        """Wait for LSE_t and S_t and load this thread's S^T columns into ``rS``. Returns the
        ends and the open lse and s handles (plain Python: inlined)."""
        lh, sh = lse.wait(), s.wait()
        for st in range(c.n_stages):
            tmem.access("S", read=True, cols=lambda w, st=st: cols(w, st, 32))
        cute.copy(c.thr_ld, c.tStS_t2r, rS)
        return lse, s, lh, sh

    @cute.jit
    def store_p(self, c, params: BwdParams, rS, lh, sh, tmem, n_block, m, key_oob):
        """P^T = exp2(S^T * scale_log2 - LSE_t) in place in ``rS`` (kept in fp32 for dS), stored
        as bf16 over S's first columns; releases the ``lh`` (LSE_t) and ``sh`` (S_t) handles."""
        if cutlass.const_expr(self.causal):
            rS = self.mask_causal(params, rS, c.tScS_t2r, n_block, m)
        else:
            if key_oob:  # keys past seqlen_k (the causal mask covers them too)
                for i in cutlass.range_constexpr(cute.size(rS)):
                    rS[i] = -Float32.inf
        rP32 = cute.make_rmem_tensor(c.rP_shape, Float32)
        rP = cute.recast_tensor(rP32, cutlass.BFloat16)
        for st in cutlass.range_constexpr(c.n_stages):
            rS_st = rS[None, st, 0, 0]
            rLSE = cute.make_rmem_tensor(c.tScS_t2r[None, 0, 0, 0].shape, Float32)
            cute.autovec_copy(lh.view(None, None, st, 0, 0, read=True), rLSE)
            for e in cutlass.range_constexpr(cute.size(rS_st) // 2):
                rS_st[2 * e], rS_st[2 * e + 1] = cute.arch.fma_packed_f32x2(
                    (rS_st[2 * e], rS_st[2 * e + 1]),
                    (params.scale_log2, params.scale_log2),
                    (-rLSE[2 * e], -rLSE[2 * e + 1]),
                )
                rS_st[2 * e] = cute.math.exp2(rS_st[2 * e], fastmath=True)
                rS_st[2 * e + 1] = cute.math.exp2(rS_st[2 * e + 1], fastmath=True)
            fa_utils.cvt_f16(rS_st, rP[None, st, 0, 0])
            if cutlass.const_expr(st == 0):
                # P overwrites S columns that the other warpgroup may still be loading.
                cute.arch.fence_view_async_tmem_load()
                self.compute_sync.sync(aligned=True)
            tmem.access("P", write=True, cols=lambda w, st=st: cols(w, st, 16))
            cute.copy(c.thr_st, rP32[None, st, None, None], c.tStP_r2t[None, st, None, None])
        sh.release()  # layer fences the tcgen05.st of P
        lh.release()

    @cute.jit
    def make_ds(self, c, rS, dpsum, dp, ds, tmem):
        """dS^T = P^T * (dP^T - dPsum_t) (``rS`` holds P^T in fp32), stored over dP's first
        columns for the dK MMA and into sdS for the dQ MMA; releases dP_t and dPsum_t. Returns
        the ends and the open ds handle. With a CTA pair, sdS holds this CTA's query half only: the
        other half goes to the peer's sdS through ``send_ds``."""
        pair = self.cluster.size > 1
        dh, dph = dpsum.wait(), dp.wait()
        rdS = [None] * c.n_stages  # bf16 dS^T per stage, stored to SMEM after the loop
        for st in cutlass.range_constexpr(c.n_stages):
            # dPsum is read before the dP load and the barrier, which hide its latency.
            rdPsum = cute.make_rmem_tensor(c.tScS_t2r[None, 0, 0, 0].shape, Float32)
            cute.autovec_copy(dh.view(None, None, st, 0, 0, read=True), rdPsum)
            rdP = cute.make_rmem_tensor(c.tScS_t2r[None, 0, None, None].shape, Float32)
            tmem.access("dP", read=True, cols=lambda w, st=st: cols(w, st, 32))
            cute.copy(c.thr_ld, c.tdPtdP_t2r[None, st, None, None], rdP)
            cute.arch.fence_view_async_tmem_load()
            # dS overwrites dP columns the other warpgroup may still be loading.
            self.compute_sync.sync(aligned=True)
            rdP_st, rS_st = rdP[None, 0, 0], rS[None, st, 0, 0]
            for e in cutlass.range_constexpr(cute.size(rdP_st) // 2):
                rdP_st[2 * e], rdP_st[2 * e + 1] = cute.arch.sub_packed_f32x2(
                    (rdP_st[2 * e], rdP_st[2 * e + 1]), (rdPsum[2 * e], rdPsum[2 * e + 1])
                )
                rdP_st[2 * e], rdP_st[2 * e + 1] = cute.arch.mul_packed_f32x2(
                    (rS_st[2 * e], rS_st[2 * e + 1]), (rdP_st[2 * e], rdP_st[2 * e + 1])
                )
            rdS[st] = cute.make_fragment_like(rdP_st, cutlass.BFloat16)
            fa_utils.cvt_f16(rdP_st, rdS[st])
            tmem.access("dS", write=True, cols=lambda w, st=st: cols(w, st, 16))
            cute.copy(c.thr_st, cute.recast_tensor(rdS[st], Float32), c.tdPtdS_r2t[None, st, 0, 0])
        # The acquire waits for the previous dQ MMA, which a CTA pair issues after dP_t. FA4
        # acquires before its first SMEM store, after stage 0, and the pair waits there;
        # acquiring once both stages are in TMEM overlaps that wait with stage 1.
        dsh = ds.acquire()  # the previous dQ MMA has read sdS (and sdSx)
        tRS_sdS = c.thr_r2s.partition_D(
            cute.make_tensor(dsh.view("dS", write=True).iterator, c.sdS_epi)
        )
        if cutlass.const_expr(pair):
            # The (key, query) tile's layout over the half-tile sdSx: only its first 64-query
            # block, the peer block's layout in sdS, is written.
            tRS_sdSx = c.thr_r2s.partition_D(
                cute.make_tensor(dsh.view("dSx", write=True).iterator, c.sdS_epi)
            )
        for st in cutlass.range_constexpr(c.n_stages):
            if cutlass.const_expr(not pair):
                cute.autovec_copy(rdS[st], tRS_sdS[None, st])
            else:
                if st == c.rank:  # stage st is query half st: this CTA's rows of dQ
                    cute.autovec_copy(rdS[st], tRS_sdS[None, st])
                else:
                    # Straight to sdSx, which the acquire also guards (the previous exchange
                    # copy completed before its dQ MMA). FA4 holds this half in registers until
                    # after the loop, which its serial schedule's sdSx-over-dQaccum alias needs.
                    cute.autovec_copy(rdS[st], tRS_sdSx[None, 0])
        dph.release()  # layer fences the tcgen05.st of dS
        if cutlass.const_expr(pair):
            # Every thread's sdS / sdSx stores reach the async proxy before the exchange copy
            # (and before the dQ MMA, which the ds commit hands them to). This also orders the
            # dPsum reads before the release below: the pair's dpsum has no release fence.
            fence_async_shared()
            self.compute_sync.sync(aligned=True)
        dh.release()
        if cutlass.const_expr(pair):
            self.send_ds(c, dsh)
        return dpsum, dp, ds, dsh

    # The pair's dS handoff (see FlashAttentionBackward2Cta).

    def signal_ds(self, ds):
        """The raw signal of a ds commit closed with ``signal=False``: the channel's ``Warps``
        count (every lane is done, then one elected arrival per warp) on the leader's full
        barrier, its only target (rank 0: the kernel runs on one CTA pair). Needs no fence:
        make_ds fenced the sdS stores, and the caller waits for its S loads (plain Python:
        inlined)."""
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(ds.full_barrier(0), peer_cta_rank_in_cluster=Int32(0))

    @cute.jit
    def send_ds(self, c, dsh):
        """Thread 0 copies the staged peer half of dS_t (sdSx) into the peer's sdS at this CTA's
        key block, completing on the peer's xchg barrier, which it arms first (FA4's DSMEM
        exchange). make_ds has fenced and synchronized every thread's sdSx stores."""
        if c.tidx == 0:
            peer = c.rank ^ 1
            xchg = self.barrier_ptr("xchg")
            src = dsh.view("dSx", read=True, by="tma").iterator
            dst = dsh.view("dS").iterator + c.rank * XCHG_ELEMS  # in the peer's copy of sdS
            cute.arch.mbarrier_arrive_and_expect_tx(
                xchg, XCHG_BYTES, peer_cta_rank_in_cluster=peer
            )
            fa_copy_utils.cpasync_bulk_s2cluster(
                src, dst, xchg, XCHG_BYTES, peer_cta_rank_in_cluster=peer
            )

    @cute.jit
    def mask_causal(self, params: BwdParams, rS, tScS_t2r, n_block, m_block):
        """S^T = -inf where key > query + seqlen_k - seqlen_q (FA4's bottom-right causal mask,
        with its R2P bitmasks over the two warpgroups' interleaved query columns)."""
        key_limit = params.seqlen_k - n_block * TILE_N * self.cluster.size - tScS_t2r[0][0]
        query_limit = params.seqlen_q - m_block * TILE_M - tScS_t2r[0][1]
        first = query_limit - key_limit  # this thread's first unmasked query column
        if key_limit <= 0:
            first = TILE_M
        first = fa_mask.row_to_r2p_idx(first, cute.size(tScS_t2r, mode=[0]), 2)
        fa_mask.mask_r2p_lambda(rS, lambda c: fa_mask.r2p_bitmask_above(first, c), rank1=True)
        return rS

    @cute.jit
    def epilogue(self, params: BwdParams, dkv, tmem, tidx, n_block, head, batch):
        """dV, then dK * softmax_scale: TMEM -> bf16 SMEM tile (64 columns per warpgroup) -> TMA
        store (FA4 epilogue_dK_or_dV_tma) of this CTA's TILE_N key rows. Returns the end for the
        caller."""
        wg = tidx // 128
        rank, keys = self.cluster.rank(), TILE_N * self.cluster.size
        r2s = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            ),
            cute.make_ordered_layout((128, 1), order=(1, 0)),
            cute.make_ordered_layout((1, 8), order=(1, 0)),
        ).get_slice(tidx % 128)
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        # One warp per warpgroup: hardware warp % 4 is the TMEM lane quadrant, tidx % 128 // 32.
        leader = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4 == 0
        for j in cutlass.range_constexpr(2):  # slot 0: dV (scale 1), slot 1: dK (softmax_scale)
            name, mma, tma, mD, sD = (
                ("dV", params.mma_dv, params.tma_dv, params.mdV, params.sdV)
                if j == 0
                else ("dK", params.mma_dk, params.tma_dk, params.mdK, params.sdK)
            )
            thr_mma = mma.get_slice(rank)
            tAcc = self.acc(thr_mma, tmem, name, (keys, HEAD_DIM))
            thr_ld = tcgen05.make_tmem_copy(ld_atom, tAcc).get_slice(tidx % 128)
            tAcc_t2r = half_cols(thr_ld.partition_S(tAcc), wg)[None, None, 0, 0]
            cAcc = thr_mma.partition_C(cute.make_identity_tensor((keys, HEAD_DIM)))
            rAcc = cute.make_rmem_tensor(
                half_cols(thr_ld.partition_D(cAcc), wg)[None, None, 0, 0].shape, Float32
            )
            with dkv.consume(j):
                tmem.access(name, read=True)
                cute.copy(thr_ld, tAcc_t2r, rAcc)
                cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(j == 1):
                for e in cutlass.range_constexpr(cute.size(rAcc) // 2):
                    rAcc[2 * e], rAcc[2 * e + 1] = cute.arch.mul_packed_f32x2(
                        (rAcc[2 * e], rAcc[2 * e + 1]),
                        (params.softmax_scale, params.softmax_scale),
                    )
            rD = cute.make_rmem_tensor(rAcc.shape, cutlass.BFloat16)
            rD.store(rAcc.load().to(cutlass.BFloat16))
            sD_wg = sD[None, None, wg]
            tDsD_r2s = r2s.partition_D(sD_wg)
            cute.copy(r2s, cute.make_tensor(rD.iterator, tDsD_r2s.shape), tDsD_r2s)
            fence_async_shared()
            if wg == 0:
                self.epi_sync_wg0.sync(aligned=True)
            else:
                self.epi_sync_wg1.sync(aligned=True)
            block = (n_block * self.cluster.size + rank, wg)  # this CTA's key rows
            gD = cute.local_tile(mD[None, None, head, batch], (TILE_N, EPI_COLS), block)
            tDsD, tDgD = tma_partition_tiles(tma, sD_wg, gD, tile_rank=2)
            if leader:
                cute.copy(tma, tDsD, tDgD)
                bulk_commit_group()
        if leader:
            bulk_wait_group(0, read=True)  # the tiles' SMEM is read before the CTA exits
        return dkv

    @role(warps=range(4), regs=152)
    @cute.jit
    def reduce(self, params: BwdParams, *, dq: Consumer, work: WorkCursor, tmem: Tmem, me: Local):
        if cutlass.const_expr(me.warps != 4):
            raise ProtocolError(
                f"reduce: the dQ TMEM load covers 128 rows with 4 warps, got {me.warps}"
            )
        tidx = me.tmem_row
        rank = self.cluster.rank()  # with a CTA pair, this CTA holds half of each dQ tile
        thr_dq = params.mma_dq.get_slice(rank)
        tdQ = self.acc(thr_dq, tmem, "dQ", (TILE_M, HEAD_DIM))
        ld_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(DQ_T2R_COLS)), Float32
        )
        thr_ld = tcgen05.make_tmem_copy(ld_atom, tdQ).get_slice(tidx)
        tdQ_t2r = thr_ld.partition_S(tdQ)
        rdQ_shape = thr_ld.partition_D(
            thr_dq.partition_C(cute.make_identity_tensor((TILE_M, HEAD_DIM)))
        ).shape
        thr_r2s = fa_copy_utils.tiled_copy_1d(Float32, 128, num_copy_elems=4).get_slice(tidx)
        tdQsdQ = thr_r2s.partition_D(params.sdQacc)
        tma_warp = me.warp == 0
        # This CTA's dQ columns, reduce-added dq_cols at a time: chunks rank * chunks + c of the
        # tile's flat fp32 accumulator (FA4's TMEM_FRAG_1CTA / TMEM_FRAG_2CTA orders).
        chunks = HEAD_DIM // self.cluster.size // self.dq_cols
        first_chunk = rank * chunks
        while work.valid:
            n_block, head, batch = work.item
            m_lo, m_hi = self.m_block_range(params, n_block)
            gdQ = cute.local_tile(params.mdQacc[None, head, batch], (TILE_M * HEAD_DIM,), (None,))
            gdQ = cute.flat_divide(gdQ, (TILE_M * self.dq_cols,))  # (elements, chunk, m_block)
            for m in cutlass.range(m_lo, m_hi, unroll=1):
                rdQ = cute.make_rmem_tensor(rdQ_shape, Float32)
                with dq.consume():
                    tmem.access("dQ", read=True)
                    cute.copy(thr_ld, tdQ_t2r, rdQ)
                    cute.arch.fence_view_async_tmem_load()
                rdQ_chunks = cute.make_tensor(rdQ.iterator, (self.dq_cols, chunks))
                for c in cutlass.range_constexpr(chunks):
                    buf = c % self.dq_stages
                    dst = tdQsdQ[None, None, buf]
                    cute.copy(
                        thr_r2s, cute.make_tensor(rdQ_chunks[None, c].iterator, dst.shape), dst
                    )
                    fence_async_shared()
                    self.reduce_sync.sync(aligned=True)
                    if tma_warp:
                        with cute.arch.elect_one():
                            fa_copy_utils.cpasync_reduce_bulk_add_f32(
                                params.sdQacc[None, buf].iterator,
                                gdQ[None, first_chunk + c, m].iterator,
                                TILE_M * self.dq_cols * 4,
                            )
                        bulk_commit_group()
                        bulk_wait_group(self.dq_stages - 1, read=True)  # the other buffer is free
                    self.reduce_sync.sync(aligned=True)
            work.advance()
        if tma_warp:
            bulk_wait_group(0, read=True)
        return dq, work

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        mmas,
        tmas,
        mQ,
        mK,
        mV,
        mdO,
        mdK,
        mdV,
        mLSE,
        mdPsum,
        mdQacc,
        layouts,
        p_layout,
        ds_layout,
        softmax_scale,
        scale_log2,
        bound,
        pair_args,
    ):
        self.begin_roles(work=bound, layouts=layouts)
        b = self.buffer
        mma_s, mma_dp, mma_dv, mma_dk, mma_dq = mmas
        tma_q, tma_k, tma_v, tma_do, tma_dk, tma_dv = tmas
        pair = {}
        if cutlass.const_expr(self.cluster.size > 1):
            names = ("tma_qt", "tma_dot", "tma_kt", "mQt", "mdOt", "mKt")
            pair = dict(zip(names, pair_args))
        params = BwdParams(
            mma_s=mma_s,
            mma_dp=mma_dp,
            mma_dv=mma_dv,
            mma_dk=mma_dk,
            mma_dq=mma_dq,
            tma_q=tma_q,
            tma_k=tma_k,
            tma_v=tma_v,
            tma_do=tma_do,
            tma_dk=tma_dk,
            tma_dv=tma_dv,
            mQ=mQ,
            mK=mK,
            mV=mV,
            mdO=mdO,
            mdK=mdK,
            mdV=mdV,
            mLSE=mLSE,
            mdPsum=mdPsum,
            mdQacc=mdQacc,
            sQ=b("Q"),
            sQt=b("Qt"),
            sK=b("K"),
            sKt=b("Kt"),
            sV=b("V"),
            sdO=b("dO"),
            sdOt=b("dOt"),
            sLSE=b("LSE"),
            sdPsum=b("dPsum"),
            sdQacc=b("dQacc"),
            sdK=b("dKepi"),
            sdV=b("dVepi"),
            p_layout=p_layout,
            ds_layout=ds_layout,
            softmax_scale=softmax_scale,
            scale_log2=scale_log2,
            seqlen_q=mQ.shape[0],
            seqlen_k=mK.shape[0],
            m_blocks=cute.ceil_div(mQ.shape[0], TILE_M),
            **pair,
        )
        self.run_roles(params)

    # ------------------------------------------------------------------- host

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        do: cute.Tensor,
        lse_log2: cute.Tensor,
        dpsum: cute.Tensor,
        dq_acc: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        softmax_scale: Float32,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        bf16 = cutlass.BFloat16
        # (b, s, h, d) -> (s, d, h, b); dO -> (d, s, h, b), the MN-major B operand of dV.
        mQ, mK, mV, mdO, mdK, mdV = (
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 3, 2, 0]))
            for t in (q, k, v, do, dk, dv)
        )
        mdO = cute.make_tensor(mdO.iterator, cute.select(mdO.layout, mode=[1, 0, 2, 3]))
        # (b, h, s) -> (s, h, b)
        mLSE, mdPsum, mdQacc = (
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=[2, 1, 0]))
            for t in (lse_log2, dpsum, dq_acc)
        )
        major, group = cute.nvgpu.OperandMajorMode, self.cluster.cta_group
        tmem_a = tcgen05.OperandSource.TMEM

        def mma(a_major, b_major, tiler, a_source=tcgen05.OperandSource.SMEM):
            return sm100_utils.make_trivial_tiled_mma(
                bf16, bf16, a_major, b_major, Float32, group, tiler[:2], a_source
            )

        keys = TILE_N * self.cluster.size  # a CTA pair's MMAs span both CTAs' keys
        tiler_s = (keys, TILE_M, HEAD_DIM)  # S^T = K Q^T
        tiler_dp = (keys, TILE_M, HEAD_DIM)  # dP^T = V dO^T
        tiler_dv = (keys, HEAD_DIM, TILE_M)  # dV = P^T dO
        tiler_dk = (keys, HEAD_DIM, TILE_M)  # dK = dS^T Q
        tiler_dq = (TILE_M, HEAD_DIM, keys)  # dQ = dS K
        mma_s = mma(major.K, major.K, tiler_s)
        mma_dp = mma(major.K, major.K, tiler_dp)
        mma_dv = mma(major.K, major.MN, tiler_dv, tmem_a)
        mma_dk = mma(major.K, major.MN, tiler_dk, tmem_a)
        mma_dq = mma(major.MN, major.MN, tiler_dq)
        sK = sm100_utils.make_smem_layout_a(mma_s, tiler_s, bf16, 1)
        sQ = sm100_utils.make_smem_layout_b(mma_s, tiler_s, bf16, self.q_stages)
        sV = sm100_utils.make_smem_layout_a(mma_dp, tiler_dp, bf16, 1)
        sdOt = sm100_utils.make_smem_layout_b(mma_dp, tiler_dp, bf16, 1)
        p_layout = sm100_utils.make_smem_layout_a(mma_dv, tiler_dv, bf16, 1)
        sdO = sm100_utils.make_smem_layout_b(mma_dv, tiler_dv, bf16, 1)
        ds_layout = sm100_utils.make_smem_layout_a(mma_dk, tiler_dk, bf16, 1)
        sQt = sm100_utils.make_smem_layout_b(mma_dk, tiler_dk, bf16, self.q_stages)
        sdS = sm100_utils.make_smem_layout_a(mma_dq, tiler_dq, bf16, 1)
        sKt = sm100_utils.make_smem_layout_b(mma_dq, tiler_dq, bf16, 1)
        s_epi = sm100_utils.make_smem_layout_epi(bf16, LayoutEnum.ROW_MAJOR, (TILE_N, EPI_COLS), 2)
        layouts = {
            "LSE": cute.make_layout((TILE_M, self.q_stages), stride=(1, TILE_M)),
            "dPsum": cute.make_layout((TILE_M, 1)),
            "Q": sQ,
            "Qt": sQt,
            "K": sK,
            "Kt": sKt,
            "V": sV,
            "dO": sdO,
            "dOt": sdOt,
            "dS": sdS,
            "dQacc": cute.make_layout((TILE_M * self.dq_cols, self.dq_stages)),
            "dKepi": s_epi,
            "dVepi": s_epi,
        }
        load_op = cpasync.CopyBulkTensorTileG2SOp(group)
        vmnk = self.cluster.layout_vmnk().shape
        sel = lambda layout: cute.select(layout, mode=[0, 1, 2])
        tma_k, mK_t = cute.nvgpu.make_tiled_tma_atom_A(load_op, mK, sel(sK), tiler_s, mma_s, vmnk)
        tma_q, mQ_t = cute.nvgpu.make_tiled_tma_atom_B(load_op, mQ, sel(sQ), tiler_s, mma_s, vmnk)
        tma_v, mV_t = cute.nvgpu.make_tiled_tma_atom_A(
            load_op, mV, sel(sV), tiler_dp, mma_dp, vmnk
        )
        tma_do, mdO_t = cute.nvgpu.make_tiled_tma_atom_B(
            load_op, mdO, sel(sdO), tiler_dv, mma_dv, vmnk
        )
        store_op = cpasync.CopyBulkTensorTileS2GOp()
        epi_tile = cute.select(s_epi, mode=[0, 1])
        tma_dk, mdK_t = cpasync.make_tiled_tma_atom(store_op, mdK, epi_tile, (TILE_N, EPI_COLS))
        tma_dv, mdV_t = cpasync.make_tiled_tma_atom(store_op, mdV, epi_tile, (TILE_N, EPI_COLS))
        pair_args = ()
        if cutlass.const_expr(self.cluster.size > 1):
            # The exchanged half of dS: one 64-query column block of the (key, query) tile.
            layouts["dSx"] = sm100_utils.make_smem_layout_epi(
                bf16, LayoutEnum.ROW_MAJOR, (TILE_N, TILE_M // 2), 1
            )
            # Qt, dOt and Kt are loaded separately (the pair splits the B operands of dK, dP
            # and dQ along other dimensions than Q, dO and K): Q (s, d) -> Qt (d, s),
            # dO (d, s) -> dOt (s, d), K (s, d) -> Kt (d, s).
            swap = lambda t: cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 0, 2, 3]))
            tma_qt, mQt_t = cute.nvgpu.make_tiled_tma_atom_B(
                load_op, swap(mQ), sel(sQt), tiler_dk, mma_dk, vmnk
            )
            tma_dot, mdOt_t = cute.nvgpu.make_tiled_tma_atom_B(
                load_op, swap(mdO), sel(sdOt), tiler_dp, mma_dp, vmnk
            )
            tma_kt, mKt_t = cute.nvgpu.make_tiled_tma_atom_B(
                load_op, swap(mK), sel(sKt), tiler_dq, mma_dq, vmnk
            )
            pair_args = (tma_qt, tma_dot, tma_kt, mQt_t, mdOt_t, mKt_t)
        n_blocks = cute.ceil_div(mK.shape[0], keys)
        bound = self.work.bind((n_blocks, mQ.shape[2], mQ.shape[3]), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            (mma_s, mma_dp, mma_dv, mma_dk, mma_dq),
            (tma_q, tma_k, tma_v, tma_do, tma_dk, tma_dv),
            mQ_t,
            mK_t,
            mV_t,
            mdO_t,
            mdK_t,
            mdV_t,
            mLSE,
            mdPsum,
            mdQacc,
            layouts,
            p_layout,
            ds_layout,
            softmax_scale,
            softmax_scale * LOG2_E,
            bound,
            pair_args,
        ).launch(**self.launch_config(bound), stream=stream, min_blocks_per_mp=1)


class FlashAttentionBackward2Cta(FlashAttentionBackward):
    """FA4's default SM100 backward (``flash_bwd_sm100.py``, ``use_2cta_instrs``: its pipelined
    2-CTA schedule, ``Q_stage = dO_stage = 1``, registers 136/136/104): a CTA pair owns one work
    item of 256 keys ``(n_block, head, batch)``, 128 per CTA, and the leader issues every MMA as
    a ``CtaGroup.TWO`` MMA over both CTAs' operands. S^T, dP^T, dV and dK are 256-row MMAs whose
    accumulator rows stay in their CTA, so the compute warps and the dK/dV epilogue are the 1-CTA
    kernel's (its ``compute`` and ``make_ds`` branch on the pair for the dS handoff below). The
    other MMA operands split differently across the pair:

    * Q and dO are the B operands of S^T and dV (each CTA holds 64 query rows), while dK and dP^T
      read Q^T and dO^T split along the head dimension (resp. queries), and dQ = dS K reads K^T
      split along the head dimension for all 256 keys: Qt, dOt and Kt are separate loads.
    * dQ = dS K is a 128-row MMA over 256 keys: each CTA holds dS for its 64 query rows and all
      256 keys (sdS, the dQ A operand), but computes dS^T for its own 128 keys and all queries.
      Each CTA therefore keeps its own query half and copies the other half (sdSx, 16 KB) into
      the peer's sdS with ``cp.async.bulk.shared::cluster``, completing on the peer's ``xchg``
      barrier; the peer's relay warp forwards that arrival to the leader on the ds channel.

    Roles (16 warps; FA4's 2-CTA layout):

        warps 0-3   reduce   dQ: this CTA's 64 columns, 8 per reduce-add, 4 staging buffers
        warps 4-11  compute  P and dS; the exchange copy; the dK/dV epilogue
        warp 12     mma      the five pair MMAs (leader only)
        warp 13     load     K, V, Kt once; Q, Qt, dO + dOt, LSE, dPsum per Q tile
        warp 14     relay    xchg barrier (the peer's dS half landed) -> ds

    Channels (n Q tiles per item; every one with an MMA side is pair-scoped: TMA loads of both
    CTAs complete on the leader's barrier, which arms both CTAs' bytes, UMMA commits multicast
    to both CTAs, and both CTAs' releases join on the leader's barrier):

        load --k, v, kt (TMA, 1 stage)--> mma           1 each; held for the whole item
        load --q (TMA, 1 stage)--> mma                  n; released right after S_t
        load --qt (TMA, 1 stage)--> mma                 n; released after dK_t
        load --do (TMA, 1 stage: dO and dOt)--> mma     n; dP_t previews it, dV_t owns it
        load --lse, dpsum (bulk copy, 1 stage)--> compute   n each, CTA-local
        mma --s (UMMA)--> compute                       n; empty = "P_t stored over S_t"
        mma --dp (UMMA)--> compute                      n; empty = "dS_t stored over dP_t"
        compute + relay --ds--> mma                     n; full = "dS_t in both CTAs' sdS
                                                        and S_{t+1} read"; empty = dQ_t done
        mma --dq (UMMA)--> reduce                       n; empty = dQ_t read out of TMEM
        mma --dkv[0] dV, dkv[1] dK (UMMA)--> compute    1 each

    The MMA warp issues FA4's order S_0, dP_0, then per tile t: dV_t, S_{t+1}, dK_t, dP_{t+1},
    dQ_t (FA4's loop body, rotated so each handle closes in the iteration that opened it; the
    last tile has no S_{t+1} or dP_{t+1}, and its s and dp acquires are retired). TMEM: S
    [0,128), dV [128,256), dP [256,384), dK [384,512), P over S [0,64), dS over dP [256,320)
    and dQ (this CTA's half: 64 columns) over S's tail [64,128). dQ_t is issued after S_{t+1},
    so the compute warps signal ds for dS_t only once they have read S_{t+1}: in the next
    iteration, through a raw arrival (``signal_ds``), since the handle closes in its own.

    SMEM: Q, Qt, dO, dOt (16 KiB each), K, Kt, V, dS (32 KiB each), sdSx (16 KiB), four 4 KiB
    dQ staging buffers, LSE/dPsum rows; the dK/dV epilogue tiles go over K and V, which the
    item's last S and dP MMAs read before dK and dV are published. About 225 of the 227 KiB.
    """

    cluster = CtaPair()

    # ------------------------------------------------------------------ roles

    @role(warps=13, regs=104)
    @cute.jit
    def load(
        self,
        params: BwdParams,
        *,
        k: Producer,
        kt: Producer,
        v: Producer,
        q: Producer,
        qt: Producer,
        do: Producer,
        lse: Producer,
        dpsum: Producer,
        work: WorkCursor,
    ):
        rank = self.cluster.rank()  # this CTA's half of every MMA operand
        thr_s, thr_dp = params.mma_s.get_slice(rank), params.mma_dp.get_slice(rank)
        thr_dv, thr_dk = params.mma_dv.get_slice(rank), params.mma_dk.get_slice(rank)
        thr_dq = params.mma_dq.get_slice(rank)
        copy_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
        keys = TILE_N * self.cluster.size
        while work.valid:
            n_block, head, batch = work.item
            m_lo, m_hi = self.m_block_range(params, n_block)
            gK = cute.local_tile(
                params.mK[None, None, head, batch], (keys, HEAD_DIM), (n_block, 0)
            )
            gV = cute.local_tile(
                params.mV[None, None, head, batch], (keys, HEAD_DIM), (n_block, 0)
            )
            gKt = cute.local_tile(
                params.mKt[None, None, head, batch], (HEAD_DIM, keys), (0, n_block)
            )
            gQ = cute.local_tile(params.mQ[None, None, head, batch], (TILE_M, HEAD_DIM), (None, 0))
            gQt = cute.local_tile(
                params.mQt[None, None, head, batch], (HEAD_DIM, TILE_M), (0, None)
            )
            gdO = cute.local_tile(
                params.mdO[None, None, head, batch], (HEAD_DIM, TILE_M), (0, None)
            )
            gdOt = cute.local_tile(
                params.mdOt[None, None, head, batch], (TILE_M, HEAD_DIM), (None, 0)
            )
            gLSE = cute.local_tile(params.mLSE[None, head, batch], (TILE_M,), (None,))
            gdPsum = cute.local_tile(params.mdPsum[None, head, batch], (TILE_M,), (None,))
            tKsK, tKgK = tma_partition_tiles(params.tma_k, params.sK, thr_s.partition_A(gK))
            tVsV, tVgV = tma_partition_tiles(params.tma_v, params.sV, thr_dp.partition_A(gV))
            tKtsKt, tKtgKt = tma_partition_tiles(
                params.tma_kt, params.sKt, thr_dq.partition_B(gKt)
            )
            tQsQ, tQgQ = tma_partition_tiles(params.tma_q, params.sQ, thr_s.partition_B(gQ))
            tQtsQt, tQtgQt = tma_partition_tiles(
                params.tma_qt, params.sQt, thr_dk.partition_B(gQt)
            )
            tOsO, tOgO = tma_partition_tiles(params.tma_do, params.sdO, thr_dv.partition_B(gdO))
            tOtsOt, tOtgOt = tma_partition_tiles(
                params.tma_dot, params.sdOt, thr_dp.partition_B(gdOt)
            )
            k, v = k.views(K=StageView(tKsK, axis=1)), v.views(V=StageView(tVsV, axis=1))
            kt, q = kt.views(Kt=StageView(tKtsKt, axis=1)), q.views(Q=StageView(tQsQ, axis=1))
            qt = qt.views(Qt=StageView(tQtsQt, axis=1))
            do = do.views(dO=StageView(tOsO, axis=1), dOt=StageView(tOtsOt, axis=1))
            # FA4's 2-CTA order: K, Q_0, LSE_0, V, dO_0 + dOt_0, dPsum_0, Kt, then per tile
            # m > 0: Qt_{m-1}, Q_m, LSE_m, dO_m + dOt_m, dPsum_m; Qt of the last tile at the end.
            with k.produce() as h:
                self.tma_load(params.tma_k, tKgK, h, "K")
            q, lse = self.load_q_lse(params, q, lse, tQgQ, gLSE, copy_stats, m_lo)
            with v.produce() as h:
                self.tma_load(params.tma_v, tVgV, h, "V")
            do, dpsum = self.load_do_dpsum(
                params, do, dpsum, tOgO, tOtgOt, gdPsum, copy_stats, m_lo
            )
            with kt.produce() as h:
                self.tma_load(params.tma_kt, tKtgKt, h, "Kt")
            for m in cutlass.range(m_lo + 1, m_hi, unroll=1):
                with qt.produce() as h:
                    self.tma_load(params.tma_qt, tQtgQt[None, m - 1], h, "Qt")
                q, lse = self.load_q_lse(params, q, lse, tQgQ, gLSE, copy_stats, m)
                do, dpsum = self.load_do_dpsum(
                    params, do, dpsum, tOgO, tOtgOt, gdPsum, copy_stats, m
                )
            with qt.produce() as h:
                self.tma_load(params.tma_qt, tQtgQt[None, m_hi - 1], h, "Qt")
            work.advance()
        return k, kt, v, q, qt, do, lse, dpsum, work

    # The per-tile loads as helpers: the first tile is peeled (plain Python: inlined).

    def load_q_lse(self, params: BwdParams, q, lse, tQgQ, gLSE, copy_stats, m):
        """Q_m and LSE_m; returns the ends."""
        with q.produce() as h:
            self.tma_load(params.tma_q, tQgQ[None, m], h, "Q")
        with lse.produce() as h, cute.arch.elect_one():
            self.bulk_load(copy_stats, gLSE[None, m], h)
        return q, lse

    def load_do_dpsum(self, params: BwdParams, do, dpsum, tOgO, tOtgOt, gdPsum, copy_stats, m):
        """dO_m and dOt_m (one do stage), then dPsum_m; returns the ends."""
        with do.produce() as h:
            self.tma_load(params.tma_do, tOgO[None, m], h, "dO")
            self.tma_load(params.tma_dot, tOtgOt[None, m], h, "dOt")
        with dpsum.produce() as h, cute.arch.elect_one():
            self.bulk_load(copy_stats, gdPsum[None, m], h)
        return do, dpsum

    @role(warps=12, regs=104, cta="leader")
    @cute.jit
    def mma(
        self,
        params: BwdParams,
        *,
        k: Consumer,
        kt: Consumer,
        v: Consumer,
        q: Consumer,
        qt: Consumer,
        do: Consumer,
        s: Producer,
        dp: Producer,
        ds: Consumer,
        dq: Producer,
        dkv: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        t = self.mma_views(params, tmem)
        while work.valid:
            n_block, _, _ = work.item
            m_lo, m_hi = self.m_block_range(params, n_block)
            kh = k.wait()
            dkv_h = (dkv.acquire(0), dkv.acquire(1))  # dV, dK of this item
            qh = q.wait()
            with s.produce():  # S_0 = K Q_0^T
                self.issue_s(params, qh, kh, t, tmem)
            qh.release()  # only S reads Q (dK reads Qt)
            vh = v.wait()
            with dp.produce():  # dP_0 = V dO_0^T; dO_0 is owned and released by tile 0's do.wait()
                self.issue_dp(params, do.peek(), vh, t, tmem)
            kth = kt.wait()
            kv = (kh, vh, kth)
            for i in cutlass.range(m_hi - m_lo - 1, unroll=1):
                q, qt, do, s, dp, ds, dq = self.mma_tile(
                    params, q, qt, do, s, dp, ds, dq, kv, t, tmem, i == 0
                )
            q, qt, do, s, dp, ds, dq = self.mma_tile(
                params, q, qt, do, s, dp, ds, dq, kv, t, tmem, m_hi - m_lo == 1, dkv_h
            )
            kh.release()
            vh.release()
            kth.release()
            work.advance()
        return k, kt, v, q, qt, do, s, dp, ds, dq, dkv, work

    def issue_dq(self, params: BwdParams, dsh, kth, t, tmem):
        """dQ = dS K over the pair's 256 keys: FA4's ``gemm_w_idx`` in two unroll groups."""
        tmem.access("dQ", write=True, by="umma")
        sdS, sKt = dsh.view("dS", read=True, by="umma"), kth.view("Kt", read=True, by="umma")
        rdS, rKt = t.thr.dq.make_fragment_A(sdS), t.thr.dq.make_fragment_B(sKt)
        fa_mma.gemm_w_idx(params.mma_dq, t.tdQ, rdS, rKt, zero_init=True, num_unroll_groups=2)

    def mma_tile(
        self, params: BwdParams, q, qt, do, s, dp, ds, dq, kv, t, tmem, first, dkv_h=None
    ):
        """One Q tile t: dV_t, S_{t+1}, dK_t, dP_{t+1}, dQ_t. The last tile (``dkv_h``, the
        item's dV and dK handles) has no S_{t+1} or dP_{t+1}: it retires those acquires and
        publishes dV and dK after their last MMAs. Returns the ends (plain Python: inlined)."""
        kh, vh, kth = kv
        last = dkv_h is not None
        doh = do.wait()  # dO_t, which dP_t previewed
        sh = s.acquire()  # P_t stored over S_t
        self.issue_dv(params, doh, t, tmem, first)
        doh.release()
        if last:
            dkv_h[0].commit()  # dV final: the commit covers every MMA issued so far
            sh.retire()  # no S_{t+1}: one item per CTA pair, so nothing consumes this stage
        else:
            qh = q.wait()
            # S_{t+1}'s last 64 columns hold dQ_{t-1}: the reduce warps have read it.
            dqh = dq.acquire()
            # S_{t+1} overwrites P_t behind the dV MMA that read it (same-thread MMA order).
            self.issue_s(params, qh, kh, t, tmem)
            sh.commit()
            qh.release()
        qth = qt.wait()
        dph = dp.acquire()  # dS_t stored over dP_t
        self.issue_dk(params, qth, t, tmem, first)
        qth.release()
        if last:
            dkv_h[1].commit()  # dK final
            dph.retire()  # no dP_{t+1}
            dsh = ds.wait()
            dqh = dq.acquire()
        else:
            # dP_{t+1} previews dO_{t+1} (tile t+1's do.wait() owns and releases it) and
            # overwrites dS_t behind the dK MMA that read it.
            self.issue_dp(params, do.peek(), vh, t, tmem)
            dph.commit()
            dsh = ds.wait()  # dS_t in both CTAs' sdS, and S_{t+1} (under dQ_t) read
        self.issue_dq(params, dsh, kth, t, tmem)
        dqh.commit()
        dsh.release()
        return q, qt, do, s, dp, ds, dq

    @role(warps=14, regs=104)
    @cute.jit
    def relay(self, params: BwdParams, *, ds: Producer, work: WorkCursor):
        """Turn the peer's exchange copy into an arrival on the leader's ds barrier: the copy
        completes on this CTA's xchg barrier, which the leader cannot wait on."""
        xchg = self.barrier_ptr("xchg")
        phase = Int32(0)
        while work.valid:
            n_block, _, _ = work.item
            m_lo, m_hi = self.m_block_range(params, n_block)
            for _ in cutlass.range(m_hi - m_lo, unroll=1):
                with ds.produce():
                    cute.arch.mbarrier_wait(xchg, phase)  # the peer's half of dS_t landed
                phase = phase ^ 1
            work.advance()
        return ds, work


# The roles trace FA4's MMA issue, mask, conversion and reduce-add helpers: key on their source.
@jit_cache(
    uncached=lambda op: op.trace_protocol,
    extra_sources=tuple(m.__file__ for m in (fa_mma, fa_copy_utils, fa_mask, fa_utils)),
)
def compile_fa4_bwd(op: FlashAttentionBackward):
    """Compile once with dynamic batch, heads and sequence lengths."""
    if op.work.schedule.persistent:
        # The retired last acquires and the epilogue tiles over Q/dO (K/V in the pair) hold for
        # one item.
        raise ValueError(f"{type(op).__name__} runs one work item per CTA (OneShot)")
    b, h = cute.sym_int(), cute.sym_int()
    sq, sk, sq_pad = cute.sym_int(), cute.sym_int(), cute.sym_int(divisibility=TILE_M)
    bf16 = cutlass.BFloat16
    qo = (b, sq, h, HEAD_DIM)
    kv = (b, sk, h, HEAD_DIM)
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, qo),
        make_fake_compact_tensor(bf16, kv),
        make_fake_compact_tensor(bf16, kv),
        make_fake_compact_tensor(bf16, qo),
        make_fake_compact_tensor(Float32, (b, h, sq_pad)),
        make_fake_compact_tensor(Float32, (b, h, sq_pad)),
        make_fake_compact_tensor(Float32, (b, h, cute.sym_int(divisibility=TILE_M * HEAD_DIM))),
        make_fake_compact_tensor(bf16, kv),
        make_fake_compact_tensor(bf16, kv),
        Float32(1.0),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        cutlass.Int32(1),
        name=op.get_name(),
    )


_OPS: dict = {}


def _op(causal, roles, dq_stages, cluster) -> tuple:
    pair = is_pair(cluster)
    dq_stages = check_dq_stages(dq_stages, pair)  # normalized: one compile key per kernel
    key = (causal, None if roles is None else tuple(sorted(roles.items())), dq_stages, pair)
    op = _OPS.get(key)
    if op is None:
        if pair:
            op = FlashAttentionBackward2Cta(causal, roles)
        else:
            op = FlashAttentionBackward(causal, roles, dq_stages)
        _OPS[key] = op
    return op, key


def check_dq_stages(dq_stages, pair: bool) -> int:
    """The dQ staging buffers: the 1-CTA kernel's option (1 or 2, default 2); the pair has
    FA4's fixed 4 x 8 columns."""
    if pair:
        if dq_stages is not None:
            raise ValueError("dq_stages is a 1-CTA option; the CTA pair uses FA4's 4 x 8 columns")
        return PAIR_DQ_STAGES
    if dq_stages not in (None, 1, 2):
        raise ValueError(f"dq_stages must be 1 or 2 (SMEM), got {dq_stages}")
    return 2 if dq_stages is None else dq_stages


def is_pair(cluster) -> bool:
    """Whether ``cluster`` selects the 2-CTA kernel: ``None`` or ``SingleCta()`` runs the 1-CTA
    one, ``CtaPair()`` the pair."""
    if cluster is None or cluster.size == 1:
        return False
    if (cluster.m, cluster.n, cluster.pair) != (2, 1, True):
        raise ValueError(f"the FA4 backward runs on one CTA or a CTA pair, got {cluster}")
    return True


def fa4_bwd_main(
    q,
    k,
    v,
    dout,
    lse_log2,
    dpsum,
    dq_accum,
    dk,
    dv,
    softmax_scale,
    *,
    causal=False,
    roles=None,
    dq_stages=None,
    cluster=None,
):
    """The main kernel alone, on FA4's preprocessed buffers (``_bwd_preprocess``): accumulates
    ``dq_accum`` and writes ``dk``/``dv``. The keywords are ``fa4_bwd``'s."""
    op, key = _op(causal, roles, dq_stages, cluster)
    fn = compile_fa4_bwd.by_static_key(key, lambda: compile_fa4_bwd(op))
    workspace, sms = op.work.launch_args(q.device)
    fn(q, k, v, dout, lse_log2, dpsum, dq_accum, dk, dv, softmax_scale, workspace, sms)


def fa4_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: float | None = None,
    *,
    causal: bool = False,
    roles: dict[str, RoleConfig] | None = None,
    dq_stages: int | None = None,
    cluster=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention backward for ``(batch, seqlen, heads, 128)`` bf16 tensors; returns
    ``(dq, dk, dv)``. ``lse`` is the forward's ``(batch, heads, seqlen_q)`` fp32 natural-log LSE.
    FA4's preprocess and dQ postprocess kernels run around the warpdrive main kernel.
    ``causal`` is bottom-right aligned as in FA4. ``cluster=CtaPair()`` runs
    ``FlashAttentionBackward2Cta`` (FA4's default kernel), else ``FlashAttentionBackward``
    (FA4's 1-CTA one); ``roles`` goes to either, ``dq_stages`` (dQ staging buffers, 1 or 2,
    default 2) to the 1-CTA kernel only."""
    from flash_attn.cute.bwd_accum_tile import AccumTileOrder
    from flash_attn.cute.cute_dsl_utils import torch2cute_dtype_map
    from flash_attn.cute.interface import (
        _bwd_postprocess_convert,
        _bwd_preprocess,
        _get_device_arch,
    )

    check_fa4_bwd_args(q, k, v, out, dout, lse)
    pair = is_pair(cluster)
    check_dq_stages(dq_stages, pair)
    b, sq, h, d = q.shape
    scale = softmax_scale if softmax_scale is not None else d**-0.5
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    if q.numel() == 0 or k.numel() == 0:
        return dq.zero_(), dk.zero_(), dv.zero_()
    sq_pad = -(-sq // TILE_M) * TILE_M
    dq_accum = torch.empty(b, h, sq_pad * d, device=q.device, dtype=torch.float32)
    dpsum = torch.empty(b, h, sq_pad, device=q.device, dtype=torch.float32)
    lse_log2 = torch.empty_like(dpsum)
    dtype = torch2cute_dtype_map[q.dtype]
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum, None, None, None, dtype, d, d, TILE_M,
        fake_mode=False,
    )  # fmt: skip
    fa4_bwd_main(
        q,
        k,
        v,
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk,
        dv,
        scale,
        causal=causal,
        roles=roles,
        dq_stages=dq_stages,
        cluster=cluster,
    )
    order = AccumTileOrder.TMEM_FRAG_2CTA if pair else AccumTileOrder.TMEM_FRAG_1CTA
    _bwd_postprocess_convert(
        dq_accum, dq, scale, None, None, _get_device_arch(), dtype, d, TILE_M, 128, 1, False,
        use_2cta_instrs=pair, accum_order=order, fake_mode=False,
    )  # fmt: skip
    return dq, dk, dv


def fa4_config(q, k, causal: bool, disable_2cta: bool = True):
    """FA4's ``BwdConfig`` for these inputs; ``disable_2cta=True`` gives the 1-CTA one this
    port matches (for comparisons through ``_flash_attn_bwd(..., config=...)``)."""
    from flash_attn.cute.config import BwdHeuristicInputs, select_bwd_config
    from flash_attn.cute.interface import _get_bwd_inputs, _get_device_arch

    b, sq, h, d = q.shape
    fields = {
        "causal": causal,
        "local": False,
        "deterministic": False,
        "dtype": str(q.dtype),
        "head_dim": d,
        "head_dim_v": d,
        "num_heads": h,
        "num_heads_kv": k.shape[2],
        "batch_size": b,
        "total_q": b * sq,
        "total_k": b * k.shape[1],
        "max_seqlen_q": sq,
        "max_seqlen_k": k.shape[1],
        "has_cu_seqlens": False,
        "has_score_mod": False,
        "has_score_mod_bwd": False,
        "has_softcap": False,
        "zero_softmax_scale": False,
        "has_mask_mod": False,
        "has_learnable_sink": False,
    }
    # FA4 revisions differ in which of these facts their backward heuristics read.
    inputs = _get_bwd_inputs(
        arch=_get_device_arch(),
        device_index=q.device.index,
        block_sparse_tensors=None,
        V_in_regs=False,
        **{name: value for name, value in fields.items() if name in BwdHeuristicInputs._fields},
    )
    return select_bwd_config(inputs._replace(disable_2cta=disable_2cta))


def check_fa4_bwd_args(q, k, v, out, dout, lse):
    tensors = (q, k, v, out, dout)
    if any(t.dtype != torch.bfloat16 for t in tensors) or lse.dtype != torch.float32:
        raise TypeError("fa4_bwd expects bf16 q, k, v, out, dout and fp32 lse")
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM or k.shape != v.shape:
        raise ValueError(f"expected (b, s, h, {HEAD_DIM}) tensors and k.shape == v.shape")
    if k.shape[0] != q.shape[0] or k.shape[2:] != q.shape[2:]:
        raise ValueError("q and k/v must share batch, heads and head_dim (no GQA)")
    if out.shape != q.shape or dout.shape != q.shape:
        raise ValueError("out and dout must have q's shape")
    if lse.shape != (q.shape[0], q.shape[2], q.shape[1]):
        raise ValueError("lse must be (batch, heads, seqlen_q)")
    if len({t.device for t in (*tensors, lse)}) != 1 or not q.is_cuda:
        raise ValueError("all tensors must be on one CUDA device")
    if any(not t.is_contiguous() or t.data_ptr() % 16 for t in (*tensors, lse)):
        raise ValueError("inputs must be contiguous and 16-byte aligned")
