"""TMA producers that fill other CTAs' stages (``Tma(fills=...)``) and multicast channels with a
rank0 consumer: derived scopes, placement errors, the TMA check, and toy protocols on 2-, 4- and
8-CTA clusters that move distinct tiles and compare every element."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Cluster,
    ClusterLaunchControl,
    Consumer,
    CtaPair,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    StageView,
    StaticPersistent,
    Threads,
    Tma,
    Umma,
    WarpSpecialized,
    WorkCursor,
    role,
)
from attn_gym.warpdrive.cluster import rank_value
from attn_gym.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    PersistentGemmParams,
    compile_persistent_gemm,
)

ROWS = 32  # one 32x64 bf16 tile lands in each CTA per stage
TILE_BYTES = ROWS * 64 * 2
CLUSTERS = 2  # clusters launched per toy run
PAIRS_2X2 = Cluster(2, 2, pair=True)


# --------------------------------------------------------------------------- #
# Toy: ``load`` puts tile ``(i % tiles) * size + key(d)`` into CTA d's stage i, where ``key(d)``
# is d, or the first CTA of d's multicast line (the CTAs of a line share their tile). With
# ``fills`` the filling CTA issues the copy of every CTA it fills; without, each CTA issues its
# own part of its line's tile. ``use`` (and ``obs`` on rank 0) copy each stage to global memory.
# --------------------------------------------------------------------------- #


class FillToy(WarpSpecialized):
    observer = False

    def __init__(self, cluster, fills=None, multicast="", consumer="all", slow=False):
        self.cluster = cluster
        self.fills, self.mode, self.slow = fills, multicast or None, slow
        load_cta = {"pair": "leader", "cluster": "rank0", None: "all"}[fills]
        self.roles = {"load": RoleConfig(0, cta=load_cta), "use": RoleConfig(1, cta=consumer)}
        self.consumer_cta = consumer
        self.ld = Ring(2, Tma(TILE_BYTES, multicast=multicast, fills=fills), Threads)
        # Leader-only consumers in a pair read 2-CTA copies (bytes of both CTAs complete on the
        # leader's barrier); everything else completes where it lands.
        self.group = "TWO" if cluster.pair and consumer == "leader" else "ONE"
        self.key = (
            f"{cluster.get_key()}_{fills or 'x'}_{multicast or 'x'}_{consumer}"
            f"{'_obs' if self.observer else ''}{'_slow' if slow else ''}"
        )

    def tile_key(self, d):
        """The tile index offset of CTA ``d`` (a Python int or device rank)."""
        if self.mode is None:
            return d
        keys = {r: self.cluster.line(r, self.mode)[0] for r in range(self.cluster.size)}
        return rank_value(keys, d)

    def copies(self, rank):
        """The CTAs whose copies this CTA issues (device ranks)."""
        if self.fills == "pair":
            return [rank + j for j in range(self.cluster.pair_size)]
        if self.fills == "cluster":
            return list(range(self.cluster.size))
        return [rank]

    @role(warps=0)
    @cute.jit
    def load(self, ld: Producer, ctx):
        rank = self.cluster.rank()
        copies = self.copies(rank)
        parts = []
        for d in copies:
            if cutlass.const_expr(self.fills is not None):
                share = self.cluster.fill(d, self.mode)
            elif cutlass.const_expr(self.mode is not None):
                share = self.cluster.multicast(self.mode)
            else:  # each CTA loads its own stage
                share = SimpleNamespace(coord=0, layout=cute.make_layout(1), mask=None)
            ts, tg = cpasync.tma_partition(ctx.tma, share.coord, share.layout, ctx.s, ctx.g)
            parts.append((ts, tg, share.mask, ctx.base + self.tile_key(d)))
        for i in cutlass.range(ctx.n, unroll=1):
            with ld.produce() as h:
                for ts, tg, mask, key in parts:
                    tile = (i % ctx.tiles) * self.cluster.size + key
                    src, dst = tg[None, tile], ts[None, h.index]
                    if cutlass.const_expr(mask is None):
                        cute.copy(ctx.tma, src, dst, tma_bar_ptr=h.barrier)
                    else:
                        cute.copy(ctx.tma, src, dst, tma_bar_ptr=h.barrier, mcast_mask=mask)
        return ld

    @cute.jit
    def drain(self, ld, ctx, out):
        rank = self.cluster.rank()
        unit = cute.arch.block_idx()[0] // self.cluster.size
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ld.consume() as h:
                if cutlass.const_expr(self.slow):
                    self.lag(rank)
                row = ((unit * self.cluster.size + rank) * ctx.n + i) * ROWS + lane
                for c in cutlass.range(64, unroll_full=True):
                    out[row, c] = ctx.stage[lane, c, h.index]
        return ld

    @cute.jit
    def lag(self, rank):
        """The filled CTAs lag far behind their filler, which finishes first."""
        if rank != 0:
            start = cute.arch.clock64()
            while cute.arch.clock64() - start < 20000:
                pass

    @role(warps=1)
    @cute.jit
    def use(self, ld: Consumer, ctx):
        return self.drain(ld, ctx, ctx.out)

    @cute.kernel
    def kernel(self, tma, tiles, out, obs, n: cutlass.Int32, ntiles: cutlass.Int32, s_layout):
        smem = self.begin_roles()
        s = smem.allocate_tensor(cutlass.BFloat16, s_layout, 128)
        unit = cute.arch.block_idx()[0] // self.cluster.size
        ctx = SimpleNamespace(
            tma=tma,
            s=cute.group_modes(s, 0, 2),
            g=cute.group_modes(tiles, 0, 2),
            stage=s,
            out=out,
            obs=obs,
            n=n,
            tiles=ntiles // (CLUSTERS * self.cluster.size),
            base=unit * (ntiles // CLUSTERS),
        )
        self.run_roles(ctx)

    @cute.jit
    def __call__(
        self, src: cute.Tensor, out: cute.Tensor, obs: cute.Tensor, n: cutlass.Int32, stream
    ):
        group = tcgen05.CtaGroup.TWO if self.group == "TWO" else tcgen05.CtaGroup.ONE
        s_layout = cute.make_ordered_layout((ROWS, 64, 2), order=(1, 0, 2))
        ctas = 1 if self.mode is None else len(self.cluster.line(0, self.mode))
        local = self.fills is None and self.mode is None
        op = cpasync.CopyBulkTensorTileG2SOp if local else cpasync.CopyBulkTensorTileG2SMulticastOp
        tma, tg = cpasync.make_tiled_tma_atom(
            op(group),
            src,
            cute.select(s_layout, mode=[0, 1]),
            (ROWS, 64),
            num_multicast=ctas,
        )
        tiles = cute.local_tile(tg, (ROWS, 64), (None, 0))
        ntiles = src.shape[0] // ROWS
        self.kernel(tma, tiles, out, obs, n, ntiles, s_layout).launch(
            **self.launch_config(grid=(CLUSTERS * self.cluster.size, 1, 1)), stream=stream
        )


class ObservedFillToy(FillToy):
    """``FillToy`` with a second consumer on rank 0 that copies rank 0's stages to ``obs``."""

    observer = True

    @role(warps=2, cta="rank0")
    @cute.jit
    def obs(self, ld: Consumer, ctx):
        return self.drain(ld, ctx, ctx.obs)


def compile_toy(op):
    rows = cute.sym_int(divisibility=ROWS)
    src = make_fake_compact_tensor(cutlass.BFloat16, (rows, 64))
    out = make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(), 64))
    obs = make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(), 64))
    return compile_tvm_ffi(op, src, out, obs, cutlass.Int32(1), name=f"warpdrive_fill_{op.key}")


def run_toy(op, n, tiles=3, reps=2):
    """Run ``op`` and check every element each consumer CTA read; repeat for bitwise equality."""
    cs = op.cluster.size
    total = CLUSTERS * tiles * cs
    src = torch.randn(total * ROWS, 64, device="cuda").to(torch.bfloat16)
    fn = compile_toy(op)
    want = torch.empty(CLUSTERS, cs, n, ROWS, 64, dtype=torch.bfloat16, device="cuda")
    for unit in range(CLUSTERS):
        for d in range(cs):
            key = op.tile_key(d)
            for i in range(n):
                tile = unit * tiles * cs + (i % tiles) * cs + key
                want[unit, d, i] = src[tile * ROWS : (tile + 1) * ROWS]
    readers = op.cluster.ctas(op.consumer_cta)
    first = None
    for _ in range(reps):
        out = torch.full((CLUSTERS * cs * n * ROWS, 64), float("nan"), device="cuda")
        out = out.to(torch.bfloat16)
        obs = out.clone()
        fn(src, out, obs, n)
        torch.cuda.synchronize()
        got = out.view(CLUSTERS, cs, n, ROWS, 64)[:, list(readers)]
        assert torch.equal(got, want[:, list(readers)]), op.key
        if op.observer:
            assert torch.equal(obs.view(CLUSTERS, cs, n, ROWS, 64)[:, 0], want[:, 0]), op.key
        if first is None:
            first = got
        assert torch.equal(got, first)
    return op


# --------------------------------------------------------------------------- #
# Derived scopes
# --------------------------------------------------------------------------- #


def test_pair_leader_fills_both_ctas():
    """The leader issues both CTAs' copies and arms both barriers; the peer's consumer releases
    the leader. Each barrier expects one arrival (the arm), the bytes landing in its CTA."""
    op = FillToy(CtaPair(), fills="pair")
    text = op.describe()
    sc = op.ld.scope
    assert sc.pair and sc.tx == TILE_BYTES and sc.tx_ctas == 1
    assert sc.full_targets == {0: (0, 1)} and sc.empty_targets == {0: (0,), 1: (0,)}
    assert sc.full_counts == {0: [1, 1], 1: [1, 1]} and sc.empty_counts[0] == [64, 64]
    assert not sc.tail  # thread releases: the exit cluster barrier orders them
    assert "tx 4096 B, rank 0 arms 0+1\n" in text
    assert "scope     pair-scoped, fills pair, remote_scope cta\n" in text


def test_fills_compose_with_multicast():
    """Pair leaders of a 2x2 cluster fill their pair and multicast along N: rank 1's stage holds
    a half from each leader, so it releases both, but only its own leader arms it."""
    op = FillToy(PAIRS_2X2, fills="pair", multicast="n")
    text = op.describe()
    sc = op.ld.scope
    assert sc.full_targets == {0: (0, 1), 2: (2, 3)}
    assert sc.empty_targets == {r: (0, 2) for r in range(4)}
    assert sc.full_counts == {r: [1, 1] for r in range(4)}
    assert sc.empty_counts == {0: [128, 128], 1: [1, 1], 2: [128, 128], 3: [1, 1]}
    assert "tx 4096 B, rank 0 arms 0+1, rank 2 arms 2+3\n" in text
    assert "fills pair, multicast n" in text


def test_rank0_fills_the_cluster():
    op = FillToy(Cluster(1, 4), fills="cluster")
    op.describe()
    sc = op.ld.scope
    assert sc.full_targets == {0: (0, 1, 2, 3)}
    assert sc.empty_targets == {r: (0,) for r in range(4)}
    assert sc.empty_counts[0] == [128, 128]


def test_pair_fill_for_a_leader_mma_completes_on_the_leader():
    """A leader-only consumer reads 2-CTA operands: both CTAs' bytes complete on the leader's
    barrier, which the leader arms for both."""
    op = FillToy(CtaPair(), fills="pair", consumer="leader")
    text = op.describe()
    sc = op.ld.scope
    assert sc.tx == 2 * TILE_BYTES and sc.tx_ctas == 2
    assert sc.full_targets == {0: (0,)} and sc.empty_targets == {0: (0,)}
    assert "tx 8192 B (2 CTAs' copies), rank 0 arms 0\n" in text


def test_multicast_with_a_rank0_consumer():
    """A rank0 observer joins the consumers of a multicast channel: it reads rank 0's stage,
    which both CTAs along N write, so it releases both."""
    op = ObservedFillToy(Cluster(1, 2), multicast="n")
    op.describe()
    sc = op.ld.scope
    assert sc.full_targets == {0: (0,), 1: (1,)}
    assert sc.empty_targets == {0: (0, 1), 1: (0, 1)}
    assert sc.empty_counts == {0: [96, 96], 1: [96, 96]}  # use in both CTAs, obs on rank 0


class _Placed(WarpSpecialized):
    """One TMA producer and one Threads consumer at the given placements."""

    def __init__(self, cluster, tma, load, use):
        self.cluster = cluster
        self.ld = Ring(2, tma, Threads)
        self.roles = {"load": RoleConfig(0, cta=load), "use": RoleConfig(1, cta=use)}

    @role(warps=0)
    @cute.jit
    def load(self, ld: Producer, ctx):
        return ld

    @role(warps=1)
    @cute.jit
    def use(self, ld: Consumer, ctx):
        return ld


@pytest.mark.parametrize(
    "cluster, tma, load, use, message",
    [
        # Without the declaration nobody fills the peer's stage.
        (CtaPair(), Tma(64), "leader", "all", r"CTAs \[1\] wait on .*fills='pair'"),
        (Cluster(1, 2), Tma(64), "rank0", "all", r"fills='cluster'"),
        (CtaPair(), Tma(64, fills="pair"), "all", "all", r"runs in CTAs \[0, 1\]"),
        (PAIRS_2X2, Tma(64, fills="cluster"), "leader", "all", "on rank0"),
        (Cluster(1, 2), Tma(64, fills="pair"), "all", "all", "has no pairs"),
        # Multicast lands in rank 1, where no consumer waits (or arms): junk completions.
        (Cluster(1, 2), Tma(64, multicast="n"), "all", "rank0", r"\[1\], which run no"),
        # A rank0 consumer cannot stand in for the other pair's consumer.
        (PAIRS_2X2, Tma(64, multicast="n"), "all", "rank0", r"leaders \[2\]"),
        # A rank0 producer multicasting without fills: nobody arms rank 1's barrier.
        (Cluster(1, 2), Tma(64, multicast="n"), "rank0", "all", "no producer runs there"),
    ],
)
def test_fill_placement_errors(cluster, tma, load, use, message):
    with pytest.raises(ProtocolError, match=message):
        _Placed(cluster, tma, load, use)._collect()


def test_fills_declaration():
    assert Tma(64, fills="pair").fills == "pair" and Tma(64).fills is None
    with pytest.raises(ValueError, match="Tma.fills must be one of"):
        Tma(64, fills="peer")


# --------------------------------------------------------------------------- #
# Toy runs
# --------------------------------------------------------------------------- #

RUNS = {
    "pair_fill": lambda: FillToy(CtaPair(), fills="pair"),
    "pair_fill_slow_peer": lambda: FillToy(CtaPair(), fills="pair", slow=True),
    "pair_fill_leader_2cta": lambda: FillToy(CtaPair(), fills="pair", consumer="leader"),
    "2x2_pair_fill_mcast_n": lambda: FillToy(PAIRS_2X2, fills="pair", multicast="n"),
    "4x1_pair_fill": lambda: FillToy(Cluster(4, 1, pair=True), fills="pair"),
    "1x4_rank0_fill": lambda: FillToy(Cluster(1, 4), fills="cluster", slow=True),
    "2x2_rank0_fill_mcast_m": lambda: FillToy(Cluster(2, 2), fills="cluster", multicast="m"),
    "2x2p_rank0_fill_leader": lambda: FillToy(PAIRS_2X2, fills="cluster", consumer="leader"),
    "1x2_mcast_rank0_obs": lambda: ObservedFillToy(Cluster(1, 2), multicast="n"),
    "pair_local_rank0_obs": lambda: ObservedFillToy(CtaPair()),
    "2x2p_local_rank0_obs": lambda: ObservedFillToy(PAIRS_2X2),
    "2x2_mcast_m_rank0_obs": lambda: ObservedFillToy(Cluster(2, 2), multicast="m"),
    "2x2p_fill_rank0_obs": lambda: ObservedFillToy(PAIRS_2X2, fills="pair"),
    "4x2p_pair_fill_mcast_n": lambda: FillToy(
        Cluster(4, 2, pair=True), fills="pair", multicast="n"
    ),
}


@pytest.mark.parametrize("n", [1, 5, 64])
@pytest.mark.parametrize("name", list(RUNS))
def test_fill_toy_runs(requires_blackwell, name, n):
    op = run_toy(RUNS[name](), n)
    assert "verified CtaGroup." in op.describe(), op.describe()


def test_fill_copy_bytes_mismatch_rejects(requires_blackwell):
    """The TMA check counts a filling CTA's copies for every CTA it fills: declaring the bytes of
    both CTAs (a per-pair count) leaves each barrier waiting for bytes that never land."""
    op = FillToy(CtaPair(), fills="pair")
    op.ld = Ring(2, Tma(2 * TILE_BYTES, fills="pair"), Threads)
    with pytest.raises(Exception, match="for the 2 CTAs it fills.*declare Tma\\(4096\\)"):
        compile_toy(op)


def test_duplicate_fill_copies_are_unverified(requires_blackwell):
    """Two copies into the leader's own stage total the bytes of both CTAs, but the peer's stage
    never fills: the check cannot count bytes per destination, so it reports the channel
    unverified instead of verified."""

    class Duplicate(FillToy):
        def __init__(self):
            super().__init__(CtaPair(), fills="pair")

        def copies(self, rank):
            return [rank, rank]

    op = Duplicate()
    with pytest.warns(UserWarning, match="could not verify"):
        compile_toy(op)
    assert "fewer distinct masks than the 2 CTAs it fills" in op.describe()


def test_fill_mask_needs_a_fills_channel(requires_blackwell):
    class Undeclared(FillToy):
        def __init__(self):
            super().__init__(CtaPair(), fills="pair")
            self.ld = Ring(2, Tma(TILE_BYTES), Threads)
            # Every CTA fills "its pair" without the declaration.
            self.roles = {**self.roles, "load": RoleConfig(0)}

        def copies(self, rank):
            return [rank]

        def tile_key(self, d):
            return d

    with pytest.raises(Exception, match="takes a Cluster.fill mask, but its Tma declares no"):
        compile_toy(Undeclared())


# --------------------------------------------------------------------------- #
# The persistent GEMM with leader-issued loads: each pair leader copies both CTAs' A/B rows
# (Tma(fills="pair")), so the peer's 2-CTA copies complete on the leader's barrier and the
# leader's 2-CTA MMA reads them from the peer's SMEM. Slower than loading from both CTAs (see
# docs/warpdrive/clusters.md), so it lives here as an end-to-end test of filled 2-CTA operands.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LeaderLoadParams(PersistentGemmParams):
    peer_tAgA: cute.Tensor = None  # the peer's A/B tiles and masks, copied by its leader
    peer_tBgB: cute.Tensor = None
    peer_mcast_a: object = None
    peer_mcast_b: object = None


class LeaderLoadGemm(PersistentGemm):
    params_type = LeaderLoadParams

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        tma = self.ab.producer
        self.ab = Ring(
            self.ab.n,
            Tma(tma.tx_bytes, multicast=tma.multicast, fills="pair"),
            Umma,
            buffer=self.ab.buffer,
        )

    def get_name(self) -> str:
        return super().get_name() + "_leaderload"

    def load_op(self, sharing_ctas: int):
        return cpasync.CopyBulkTensorTileG2SMulticastOp(self.cluster.cta_group)  # fill masks

    def load_partitions(self, tiled_mma, mma_rank, tma_a, tma_b, sA, sB, gA, gB) -> dict:
        """The leader's own loads and its peer's (only the leader issues them). Both CTAs of a
        pair sit at the same position along N and M, so the SMEM partitions agree."""
        leader = self.cluster.rank() - mma_rank
        out = {}
        for j, prefix in ((0, ""), (1, "peer_")):
            fill_a, fill_b = self.cluster.fill(leader + j, "n"), self.cluster.fill(leader + j, "m")
            thr = tiled_mma.get_slice(j)
            tAsA, tAgA = cpasync.tma_partition(
                tma_a,
                fill_a.coord,
                fill_a.layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr.partition_A(gA), 0, 3),
            )
            tBsB, tBgB = cpasync.tma_partition(
                tma_b,
                fill_b.coord,
                fill_b.layout,
                cute.group_modes(sB, 0, 3),
                cute.group_modes(thr.partition_B(gB), 0, 3),
            )
            out |= {f"{prefix}tAgA": tAgA, f"{prefix}tBgB": tBgB}
            out |= {f"{prefix}mcast_a": fill_a.mask, f"{prefix}mcast_b": fill_b.mask}
            if j == 0:
                out |= {"tAsA": StageView(tAsA, axis=1), "tBsB": StageView(tBsB, axis=1)}
        return out

    @role(warps=4, regs=40, cta="leader")
    @cute.jit
    def load(self, params: LeaderLoadParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)
        copies = [
            (params.tAgA, params.tBgB, params.mcast_a, params.mcast_b),
            (params.peer_tAgA, params.peer_tBgB, params.peer_mcast_a, params.peer_mcast_b),
        ]
        while work.valid:
            m, n = self.mma_tile_mn(params, work.item)
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as s:
                    a, b = s.view("A"), s.view("B")
                    for tAgA, tBgB, mask_a, mask_b in copies:
                        cute.copy(
                            params.tma_a,
                            tAgA[None, m, k],
                            a,
                            tma_bar_ptr=s.barrier,
                            mcast_mask=mask_a,
                        )
                        cute.copy(
                            params.tma_b,
                            tBgB[None, n, k],
                            b,
                            tma_bar_ptr=s.barrier,
                            mcast_mask=mask_b,
                        )
            work.advance()
        return ab, work


def run_leader_gemm(op, m, n, k, max_ctas=None, reps=2):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    fn = compile_persistent_gemm(op)
    first = None
    for _ in range(reps):
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        workspace, sms = op.work.launch_args(a.device)
        fn(a, b, out, workspace, max_ctas or sms)
        err = (out.float() - ref).abs().max().item()
        assert err <= 2 * torch_err + 1e-3, (op.get_name(), (m, n, k), err, torch_err)
        first = out if first is None else first
        assert torch.equal(out, first), "leader-load GEMM is not bitwise deterministic"


def test_leader_load_gemm_scope():
    op = LeaderLoadGemm(cluster=PAIRS_2X2)
    text = op.describe()
    ab = op.ab.scope
    assert ab.full_targets == {0: (0,), 2: (2,)} and ab.tx_ctas == 2
    # Each leader's copies reach both pairs along N, so each leader's MMA releases both.
    assert ab.empty_targets == {0: (0, 2), 2: (0, 2)} and ab.tail
    assert "  4      load         40  leader\n" in text
    assert "tx 65536 B (2 CTAs' copies), rank 0 arms 0, rank 2 arms 2\n" in text


SCHEDULES = {"static": StaticPersistent(), "clc": ClusterLaunchControl(stages=2)}


@pytest.mark.parametrize(
    "cluster", [CtaPair(), PAIRS_2X2, Cluster(4, 1, pair=True)], ids=["pair", "2x2p", "4x1p"]
)
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_leader_load_gemm(requires_blackwell, cluster, schedule):
    op = LeaderLoadGemm(schedule=SCHEDULES[schedule], cluster=cluster, tile_n=256)
    for m, n, k in [(300, 520, 136), (3000, 1000, 520)]:
        run_leader_gemm(op, m, n, k)
    run_leader_gemm(op, 1100, 2064, 200, max_ctas=7, reps=3)  # a capped grid wraps every ring
    assert "tma       verified CtaGroup.TWO" in op.describe()
