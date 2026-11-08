"""Clusters larger than a CTA pair: the Cluster declaration, multicast TMA channels, per-rank
scopes, a toy protocol across two MMA pairs, and the persistent GEMM on 2x2/4x1 clusters."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    AtomicPersistent,
    Cluster,
    ClusterLaunchControl,
    Consumer,
    CtaPair,
    Instance,
    OneShot,
    Producer,
    ProtocolError,
    Ring,
    SingleCta,
    Slots,
    StaticPersistent,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.cluster import rank_in, rank_value
from attn_gym.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    compile_persistent_gemm,
    persistent_gemm,
)

SCHEDULES = {
    "oneshot": OneShot(),
    "static": StaticPersistent(),
    "atomic": AtomicPersistent(stages=2),
    "clc": ClusterLaunchControl(stages=2),
}
PAIRS_2X2 = Cluster(2, 2, pair=True)


# --------------------------------------------------------------------------- #
# Declarations and geometry
# --------------------------------------------------------------------------- #


def test_cluster_geometry():
    c = PAIRS_2X2
    assert (c.size, c.pair_size, c.launch_cluster()) == (4, 2, (4, 1, 1))
    # Ranks are (v, mi, ni) with the MMA pair fastest, then M, then N.
    assert [c.coords(r) for r in range(4)] == [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)]
    assert c.ctas("all") == (0, 1, 2, 3) and c.ctas("leader") == (0, 2) and c.ctas("rank0") == (0,)
    assert [c.line(r, "n") for r in range(4)] == [(0, 2), (1, 3), (0, 2), (1, 3)]
    assert [c.line(r, "m") for r in range(4)] == [(0,), (1,), (2,), (3,)]
    flat = Cluster(2, 2)
    assert flat.ctas("leader") == (0, 1, 2, 3) and flat.reach(3, ("m", "n")) == (1, 2, 3)
    # The pair-only shapes keep their keys (kernel names and caches depend on them).
    assert (SingleCta().get_key(), CtaPair().get_key(), c.get_key()) == (
        "cta1",
        "pair",
        "m2n2_pairs",
    )
    assert CtaPair() == CtaPair() and CtaPair() != Cluster(2, 1, pair=True)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"m": 0}, "Cluster.m must be a positive int"),
        ({"n": 2.0}, "Cluster.n must be a positive int"),
        ({"m": True}, "Cluster.m must be a positive int"),
        ({"pair": 1, "m": 2}, "Cluster.pair must be a bool"),
        ({"m": 3, "pair": True}, "needs an even m"),
        ({"m": 4, "n": 4}, "at most 8 CTAs"),
    ],
)
def test_cluster_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Cluster(**kwargs)


def test_tma_multicast_declaration():
    assert Tma(64, multicast="nm").multicast == ("m", "n")
    assert Tma(64, multicast=["n", "n"]).multicast == ("n",)
    assert Tma(64).multicast == ()
    with pytest.raises(ValueError, match="Tma.multicast must name cluster modes"):
        Tma(64, multicast="k")
    with pytest.raises(ValueError, match="Tma.multicast must name cluster modes"):
        Tma(64, multicast=1)


def test_2x2_gemm_declared_graph():
    """Two MMA pairs side by side along N multicast A. Each CTA's A/B stage is read by its own
    pair's MMA and, through its multicast A half, by the other pair's: every leader's commit
    releases all four CTAs, and every CTA's empty barrier waits for both leaders."""
    op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=PAIRS_2X2, tile_n=224)
    text = op.describe()
    ab = op.ab.scope
    assert op.ab.producer.multicast == ("n",)
    assert ab.full_ctas == (0, 2) and ab.full_targets == {0: (0,), 2: (2,)}
    assert ab.tx == 2 * op.ab.producer.tx_bytes and ab.tx_ctas == 2
    assert ab.empty_targets == {0: (0, 1, 2, 3), 2: (0, 1, 2, 3)} and ab.empty_mask == 0b1111
    assert ab.empty_counts == {r: [2] * op.ab.n for r in range(4)} and ab.tail
    # The accumulator stays per pair: each leader commits into its own pair's CTAs.
    acc = op.acc.scope
    assert acc.full_mask == {0: 0b0011, 2: 0b1100}
    assert acc.empty_targets == {0: (0,), 1: (0,), 2: (2,), 3: (2,)}
    assert acc.empty_counts[0] == acc.empty_counts[2] == [256, 256]
    # One scheduler per cluster (rank 0) publishes to all four CTAs; every stream user of every
    # CTA releases rank 0's slot: 4 x (128 + 32) for epilogue and load, 2 x 32 for the leaders'
    # MMA warps, 32 for the scheduler.
    sched = next(r for r in op._ws_roles if r.name == "work.scheduler")
    assert sched.cta == "rank0"
    work = op._ws_chans["work_ring"].scope
    assert work.full_ctas == (0, 1, 2, 3) and work.empty_counts[0] == [736, 736]
    assert "  cluster  cluster 2 x 2 (M x N), tcgen05 CtaGroup.TWO, leaders = even ranks" in text
    assert "full      rank 0 1 x6, rank 2 1 x6, tx 61440 B (leader-armed)\n" in text
    assert "scope     cross-CTA, multicast n, release_mask 0b1111, tail\n" in text
    assert "commit_mask rank 0 0b0011, rank 2 0b1100" in text
    assert "  6      work.scheduler    40  rank0\n" in text


def test_flat_2x2_multicasts_both_operands():
    """Without pairs, CTAs along N share A and CTAs along M share B: each stage is read by three
    CTAs' MMAs (its own, and one along each mode), so each commit names a different trio."""
    op = PersistentGemm(cluster=Cluster(2, 2))
    op.describe()
    ab = op.ab.scope
    assert op.ab.producer.multicast == ("m", "n") and ab.tx_ctas == 1
    assert ab.full_targets == {r: (r,) for r in range(4)}
    assert ab.empty_mask == {0: 0b0111, 1: 0b1011, 2: 0b1101, 3: 0b1110}
    assert ab.empty_counts == {r: [3] * op.ab.n for r in range(4)}
    assert not op.acc.scope.pair  # MMA and epilogue run in every CTA


class _MulticastWithRank0(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIRS_2X2
        self.ch = Ring(2, Tma(1024, multicast="n"), Threads)

    @role(warps=0)
    @cute.jit
    def load(self, ch: Producer, ctx):
        return ch

    @role(warps=1, cta="rank0")
    @cute.jit
    def use(self, ch: Consumer, ctx):
        return ch


class _AllToRank0Tma(_MulticastWithRank0):
    def __init__(self):
        self.cluster = PAIRS_2X2
        self.ch = Ring(2, Tma(1024), Threads)


class _LeaderMulticast(WarpSpecialized):
    """Leader-only producer and consumer multicasting along N between the two pair leaders."""

    def __init__(self):
        self.cluster = PAIRS_2X2
        self.ch = Ring(2, Tma(1024, multicast="n"), Umma)

    @role(warps=0, cta="leader")
    @cute.jit
    def load(self, ch: Producer, ctx):
        return ch

    @role(warps=1, cta="leader")
    @cute.jit
    def mma(self, ch: Consumer, ctx):
        return ch


class _Rank0Mma(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIRS_2X2
        self.acc = Ring(2, Umma, Threads)

    @role(warps=0, cta="rank0")
    @cute.jit
    def mma(self, acc: Producer, ctx):
        return acc

    @role(warps=1)
    @cute.jit
    def epilogue(self, acc: Consumer, ctx):
        return acc


class _FlatLeaderTmem(WarpSpecialized):
    def __init__(self):
        self.cluster = Cluster(2, 2)
        self.tmem_plan = TmemPlan().region("acc", 32)

    @role(warps=0, cta="leader")
    @cute.jit
    def mma(self, tmem: Tmem, ctx):
        pass


def test_same_placement_multicast_arms_locally():
    """Each leader arms its own barrier for the bytes landing in it and releases both leaders,
    whose copies reach it."""
    op = _LeaderMulticast()
    op.describe()
    sc = op.ch.scope
    assert sc.pair and sc.tx == 1024 and sc.tx_ctas == 1
    assert sc.full_targets == {0: (0,), 2: (2,)} and sc.empty_targets == {0: (0, 2), 2: (0, 2)}
    assert sc.empty_mask == 0b101 and sc.empty_counts[0] == [2, 2] and sc.tail


def test_device_rank_helpers_on_host_ranks():
    """With a Python rank the helpers evaluate directly; the device forms are exercised by the
    toys below."""
    assert rank_in(3, (0, 1, 2, 3), 4) is True
    assert [rank_in(r, (0, 2), 4) for r in range(4)] == [True, False, True, False]
    assert [rank_in(r, (1, 3), 4) for r in range(4)] == [False, True, False, True]
    assert [bool(rank_in(r, (0, 3), 4)) for r in range(4)] == [True, False, False, True]
    assert rank_value({0: 7, 1: 7}, 1) == 7
    counts = {0: 160, 1: 128}  # TMEM users of a pair: the leader also runs the MMA warp
    assert [rank_value(counts, r) for r in (0, 1)] == [160, 128]
    assert rank_value({0: 9}, lambda: pytest.fail("a constant must not read the rank")) == 9


def test_cluster_placement_errors():
    # A rank0 consumer alone leaves the other pair's multicast copies with no waiter.
    with pytest.raises(ProtocolError, match=r"complete on leaders \[2\], where no consumer"):
        _MulticastWithRank0()._collect()
    # 2-CTA copies of the second pair would complete on rank 2's barrier, which nobody waits on.
    with pytest.raises(ProtocolError, match="can only feed leader-only consumers"):
        _AllToRank0Tma()._collect()
    # Ranks 2 and 3 would read an accumulator no MMA of their pair wrote.
    with pytest.raises(ProtocolError, match="needs the MMAs of CTAs \\[0, 2\\].*cta='leader'"):
        _Rank0Mma()._collect()
    _FlatLeaderTmem()._collect()  # without pairs every CTA is a leader, so it allocates

    class FlatRank0Mma(_Rank0Mma):  # rank 1's epilogue would read TMEM no MMA of its own wrote
        def __init__(self):
            super().__init__()
            self.cluster = Cluster(1, 2)

    with pytest.raises(ProtocolError, match="needs the MMAs of CTAs \\[0, 1\\]"):
        FlatRank0Mma()._collect()


@pytest.mark.parametrize(
    "cluster", [Cluster(1, 3), Cluster(3, 1), Cluster(2, 3, pair=True)], ids=str
)
def test_gemm_rejects_non_power_of_two_multicast(cluster):
    """Regression: 3-way multicast compiled, then faulted with an illegal instruction."""
    with pytest.raises(ValueError, match="power-of-two count of MMA tiles"):
        PersistentGemm(cluster=cluster)


def test_fa4_rejects_clusters_beyond_a_pair():
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import FlashAttentionForward

    with pytest.raises(ValueError, match="one CTA or a CTA pair"):
        FlashAttentionForward(cluster=PAIRS_2X2)


# --------------------------------------------------------------------------- #
# Toy protocol across two MMA pairs: producers in every CTA -> each pair's leader -> sinks in
# every CTA. The producers' Threads arrivals pick their pair leader by rank, the leaders'
# commits carry per-rank masks, and every producer tails.
# --------------------------------------------------------------------------- #


class ClusterToy(WarpSpecialized):
    def __init__(self, cluster):
        self.cluster = cluster
        self.ch = Slots(2, Threads, Umma)
        self.back = Ring(2, Umma, Threads)

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def prod(self, stage: Instance, ch: Producer[Instance], ctx):
        for _ in cutlass.range(ctx.n + stage, unroll=1):
            ch.acquire(stage).commit()
        return ch

    @role(warps=2, cta="leader")
    @cute.jit
    def mma(self, ch: Consumer, back: Producer, ctx):
        for _ in cutlass.range(ctx.n, unroll=1):
            for j in cutlass.range_constexpr(2):
                with ch.consume(j), back.produce():
                    pass
        with ch.consume(1), back.produce():
            pass
        return ch, back

    @role(warps=3)
    @cute.jit
    def sink(self, back: Consumer, ctx):
        count = cutlass.Int32(0)
        for _ in cutlass.range(2 * ctx.n + 1, unroll=1):
            with back.consume():
                count = count + 1
        slot = cute.arch.block_idx()[0] // self.cluster.size * 8 + self.cluster.rank()
        if cute.arch.lane_idx() == 0:
            ctx.out[slot] = count
        return back

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(f"warpdrive_cluster_toy_{self.cluster.get_key()}")
        grid = (3 * self.cluster.size, 1, 1)  # three clusters
        self.kernel(out, n).launch(**self.launch_config(grid=grid), stream=stream)


def test_cluster_toy_scopes():
    op = ClusterToy(Cluster(4, 2, pair=True))
    op.describe()
    ch = op.ch.scope
    # Each pair's producers join on their own leader; leaders release their own pair.
    assert ch.full_targets == {r: (r - r % 2,) for r in range(8)}
    assert ch.full_counts[2] == [64, 64] and ch.full_counts[3] == [1, 1]
    assert ch.empty_mask == {r: 0b11 << r for r in range(0, 8, 2)}
    assert op.back.scope.empty_targets == {r: (r - r % 2,) for r in range(8)}


class ThreadsToy(WarpSpecialized):
    """Threads on both sides across pairs: ``up`` joins both CTAs of a pair on its leader, which
    releases both (two targets picked by rank); ``down`` fans the leader back out to both."""

    def __init__(self, cluster):
        self.cluster = cluster
        self.up = Ring(2, Threads, Threads)
        self.down = Ring(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def src(self, up: Producer, ctx):
        for _ in cutlass.range(ctx.n, unroll=1):
            up.acquire().commit()
        return up

    @role(warps=1, cta="leader")
    @cute.jit
    def lead(self, up: Consumer, down: Producer, ctx):
        for _ in cutlass.range(ctx.n, unroll=1):
            with up.consume(), down.produce():
                pass
        return up, down

    @role(warps=2)
    @cute.jit
    def sink(self, down: Consumer, ctx):
        count = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with down.consume():
                count = count + 1
        slot = cute.arch.block_idx()[0] // self.cluster.size * 8 + self.cluster.rank()
        if cute.arch.lane_idx() == 0:
            ctx.out[slot] = count
        return down

    kernel = ClusterToy.kernel

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(f"warpdrive_threads_toy_{self.cluster.get_key()}")
        grid = (3 * self.cluster.size, 1, 1)
        self.kernel(out, n).launch(**self.launch_config(grid=grid), stream=stream)


def test_threads_toy_scopes():
    op = ThreadsToy(Cluster(4, 2, pair=True))
    op.describe()
    assert op.up.scope.empty_targets == {r: (r, r + 1) for r in range(0, 8, 2)}
    assert op.down.scope.full_targets == {r: (r, r + 1) for r in range(0, 8, 2)}
    assert op.up.scope.full_counts[4] == [64, 64] and op.down.scope.empty_counts[4] == [64, 64]


@jit_cache
def compile_cluster_toy(op, name: str):
    fake = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, fake, cutlass.Int32(1), name=name)


@pytest.mark.parametrize("toy", [ClusterToy, ThreadsToy])
@pytest.mark.parametrize("cluster", [PAIRS_2X2, Cluster(4, 2, pair=True), Cluster(2, 2)])
@pytest.mark.parametrize("n", [1, 5, 64])
def test_cluster_toy_runs(requires_blackwell, toy, cluster, n):
    out = torch.zeros(3 * 8, device="cuda", dtype=torch.int32)
    compile_cluster_toy(toy(cluster), f"{toy.__name__.lower()}_{cluster.get_key()}")(out, n)
    torch.cuda.synchronize()
    got = out.view(3, 8)[:, : cluster.size]
    want = 2 * n + 1 if toy is ClusterToy else n
    assert (got == want).all(), got


# --------------------------------------------------------------------------- #
# Persistent GEMM on larger clusters
# --------------------------------------------------------------------------- #

GEMM_SHAPES = [
    (1, 8, 8),  # one cluster tile, three of its four CTAs entirely out of bounds
    (300, 520, 136),  # ragged; 520 / 256 needs a second pair along N in a partial cluster tile
    (256, 448, 64),  # exactly one 2x2 cluster tile at tile_n=224
    (1000, 3000, 520),  # an odd count of pair tiles along N: the last cluster tile is half empty
    (4097, 2056, 520),  # more than one wave
]


def _check_gemm(cluster, schedule, m, n, k, *, tile_n=256, max_ctas=None, reps=2):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    first = None
    for _ in range(reps):  # later launches reuse the self-resetting atomic counter
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        persistent_gemm(
            a, b, out, schedule=schedule, max_ctas=max_ctas, cluster=cluster, tile_n=tile_n
        )
        err = (out.float() - ref).abs().max().item()
        assert err <= 2 * torch_err + 1e-3, (cluster, schedule, (m, n, k), err, torch_err)
        if first is None:
            first = out
        assert torch.equal(out, first), "cluster GEMM is not bitwise deterministic"


@pytest.mark.parametrize("m,n,k", GEMM_SHAPES)
@pytest.mark.parametrize("schedule", list(SCHEDULES))
@pytest.mark.parametrize("tile_n", [256, 224])
def test_2x2_gemm(requires_blackwell, tile_n, schedule, m, n, k):
    _check_gemm(PAIRS_2X2, SCHEDULES[schedule], m, n, k, tile_n=tile_n)


@pytest.mark.parametrize(
    "cluster",
    [
        Cluster(4, 1, pair=True),
        Cluster(4, 2, pair=True),
        Cluster(2, 2),
        Cluster(1, 2),
        Cluster(2, 1),
    ],
    ids=["4x1_pairs", "4x2_pairs", "2x2", "1x2", "2x1"],
)
@pytest.mark.parametrize("schedule", ["static", "clc"])
def test_other_cluster_gemms(requires_blackwell, cluster, schedule):
    """B multicast along M (4x1 pairs, 2x1), A along N (1x2), and both (2x2 without pairs; 4x2
    pairs, where each stage is released by three leaders)."""
    for m, n, k in [(300, 520, 136), (3000, 1000, 520)]:
        _check_gemm(cluster, SCHEDULES[schedule], m, n, k)


@pytest.mark.parametrize("schedule", ["static", "atomic", "clc"])
@pytest.mark.parametrize("max_ctas", [4, 7, 16])
def test_2x2_gemm_capped_grid(requires_blackwell, schedule, max_ctas):
    """4 and 7 CTAs launch one cluster, so every ring and the work ring wrap many times."""
    _check_gemm(PAIRS_2X2, SCHEDULES[schedule], 1100, 2064, 200, max_ctas=max_ctas, reps=3)


def test_multicast_copy_needs_a_multicast_channel(requires_blackwell):
    """Positive evidence of a multicast copy into a channel declared without multicast rejects:
    the other pair's MMA would read this CTA's stage without releasing it."""

    class Undeclared(PersistentGemm):
        def __init__(self):
            super().__init__(cluster=PAIRS_2X2)
            self.ab = Ring(self.ab.n, Tma(self.ab.producer.tx_bytes), Umma, buffer=self.ab.buffer)

    with pytest.raises(Exception, match="is multicast along n into channel ab") as info:
        compile_persistent_gemm(Undeclared())
    assert isinstance(info.value, ProtocolError) or "ProtocolError" in repr(info.value)


def test_2x2_gemm_tma_check_verifies_multicast(requires_blackwell):
    op = PersistentGemm(cluster=PAIRS_2X2, tile_n=224)
    compile_persistent_gemm(op)
    assert "tma       verified CtaGroup.TWO, 30720 B/stage per CTA" in op.describe()


@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
def test_odd_subtile_epilogue_rotates_buffers(requires_blackwell, cluster):
    """tile_n=224 stores 7 subtiles per tile through 2 staging buffers. Regression: every tile
    restarted at buffer 0, rewriting the last subtile's buffer while its TMA store still read it
    (wrong output under persistent schedules, on every cluster shape)."""
    _check_gemm(cluster, StaticPersistent(), 4097, 2056, 520, tile_n=224, max_ctas=16, reps=3)
