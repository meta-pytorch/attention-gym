"""Test cluster geometry, derived scopes, placement checks, and cross-CTA handoffs.

GEMM and FA4 run on CTA pairs and larger clusters under every schedule.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.nvgpu import tcgen05
from test_warpdrive_scheduling import SCHEDULES, check_fa4_twice, check_gemm

from attn_gym.warpdrive import (
    Cluster,
    ClusterLaunchControl,
    Consumer,
    CtaPair,
    Instance,
    Producer,
    ProtocolError,
    Raster,
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
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.cluster import check_atoms, rank_in, rank_value
from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm, compile_persistent_gemm

PAIR = CtaPair()
PAIRS_2X2 = Cluster(2, 2, pair=True)
PAIRS_4X2 = Cluster(4, 2, pair=True)


# Declarations and geometry


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
    assert Tma(64, multicast="nm").multicast == ("m", "n")
    assert Tma(64, multicast=["n", "n"]).multicast == ("n",)
    assert Tma(64).multicast == ()


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: Cluster(m=0), "Cluster.m must be a positive int"),
        (lambda: Cluster(n=2.0), "Cluster.n must be a positive int"),
        (lambda: Cluster(m=True), "Cluster.m must be a positive int"),
        (lambda: Cluster(pair=1, m=2), "Cluster.pair must be a bool"),
        (lambda: Cluster(m=3, pair=True), "needs an even m"),
        (lambda: Cluster(m=4, n=4), "at most 8 CTAs"),
        (lambda: Tma(64, multicast="k"), "Tma.multicast must name cluster modes"),
        (lambda: Tma(64, multicast=1), "Tma.multicast must name cluster modes"),
        (lambda: role(warps=0, cta="peer"), "role cta must be one of"),
        # rank0 is the broadcast scheduler's placement; roles declare all or leader.
        (lambda: role(warps=0, cta="rank0"), r"role cta must be one of \('all', 'leader'\)"),
        # Three-way partitions do not divide these tile extents; they compile but fault at runtime.
        (lambda: PersistentGemm(cluster=Cluster(1, 3)), "power-of-two count of MMA tiles"),
        (lambda: PersistentGemm(cluster=Cluster(3, 1)), "power-of-two count of MMA tiles"),
        (
            lambda: PersistentGemm(cluster=Cluster(2, 3, pair=True)),
            "power-of-two count of MMA tiles",
        ),
    ],
)
def test_declaration_validation(make, message):
    with pytest.raises(ValueError, match=message):
        make()


def test_fa4_rejects_clusters_beyond_a_pair():
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import FlashAttentionForward

    with pytest.raises(ValueError, match="one CTA or a CTA pair"):
        FlashAttentionForward(cluster=PAIRS_2X2)


def test_default_tile_n():
    """Default tile widths: 224 for a pair, 256 for one CTA or a larger cluster."""
    assert PersistentGemm().tile_n == 256
    assert PersistentGemm(cluster=PAIR).tile_n == 224
    assert PersistentGemm(cluster=Cluster(2, 1, pair=True)).tile_n == 224
    assert PersistentGemm(cluster=Cluster(1, 2)).tile_n == 256
    assert PersistentGemm(cluster=PAIRS_2X2).tile_n == 256
    assert PersistentGemm(cluster=SingleCta(), tile_n=224).tile_n == 224


def test_device_rank_helpers_on_host_ranks():
    """With a Python rank the helpers evaluate directly; the device forms run in the toys."""
    assert rank_in(3, (0, 1, 2, 3), 4) is True
    assert [rank_in(r, (0, 2), 4) for r in range(4)] == [True, False, True, False]
    assert [rank_in(r, (1, 3), 4) for r in range(4)] == [False, True, False, True]
    assert [bool(rank_in(r, (0, 3), 4)) for r in range(4)] == [True, False, False, True]
    assert rank_value({0: 7, 1: 7}, 1) == 7
    counts = {0: 160, 1: 128}  # TMEM users of a pair: the leader also runs the MMA warp
    assert [rank_value(counts, r) for r in (0, 1)] == [160, 128]
    assert rank_value({0: 9}, lambda: pytest.fail("a constant must not read the rank")) == 9


# Derived scopes of the examples


def test_leader_placement_is_a_noop_for_one_cta():
    """The same role source runs on one CTA: `cta="leader"` places it in the only CTA."""
    op = PersistentGemm()
    text = op.describe()
    assert "leader" not in text and "pair-scoped" not in text and "scope" not in text
    assert "full      1 x4, tx 49152 B\n" in text


def test_pair_gemm_declared_graph():
    op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=PAIR, tile_n=256)
    text = op.describe()
    # Both CTAs' TMA bytes (2 x 32 KB) complete on the leader, armed once by its producer.
    assert "full      leader 1 x6, tx 65536 B (leader-armed)\n" in text
    assert "scope     pair-scoped, release_mask 0b11, tail\n" in text
    acc = op.acc.scope
    assert acc.full_counts == {0: [1, 1], 1: [1, 1]} and acc.full_mask == 0b11
    assert acc.empty_ctas == (0,) and acc.empty_counts[0] == [256, 256]
    # The leader's scheduler publishes into both CTAs (st.async bytes), and every consumer
    # releases the leader's slot: epilogue and load in both CTAs, the leader-only MMA and
    # scheduler once: 2 x (128 + 32) + 32 + 32. The st.async payload completes by bytes; the
    # releases only hand the slot back, so they count at CTA scope.
    assert "  6      work.scheduler    40  leader\n" in text
    assert "  work_ring  Ring(2)  work.scheduler -> epilogue, load, mma, work.scheduler\n" in text
    assert (
        "scope     pair-scoped, remote_scope cta\n             storage   payload Int32(4,)\n"
        in text
    )
    work = op._ws_chans["work_ring"].scope
    assert work.full_counts == {0: [1, 1], 1: [1, 1]} and work.tx == 16
    assert work.empty_ctas == (0,) and work.empty_counts[0] == [384, 384]
    assert not op._ws_chans["work_clc"].scope.pair


def test_2x2_gemm_declared_graph():
    """Two MMA pairs side by side along N multicast A. Each CTA's A/B stage is read by its own
    pair's MMA and, through its multicast A half, by the other pair's: every leader's commit
    releases all four CTAs, and every CTA's empty barrier waits for both leaders."""
    op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=PAIRS_2X2, tile_n=224)
    text = op.describe()
    ab = op.ab.scope
    assert ab.full_targets == {0: (0,), 2: (2,)} and ab.tx_ctas == 2
    assert ab.empty_targets == {0: (0, 1, 2, 3), 2: (0, 1, 2, 3)}
    assert ab.empty_counts == {r: [2] * op.ab.n for r in range(4)}
    # The accumulator stays per pair: each leader commits into its own pair's CTAs.
    assert op.acc.scope.empty_targets == {0: (0,), 1: (0,), 2: (2,), 3: (2,)}
    # One scheduler per cluster (rank 0) publishes to all four CTAs; every stream user of every
    # CTA releases rank 0's slot: 4 x (128 + 32) for epilogue and load, 2 x 32 for the leaders'
    # MMA warps, 32 for the scheduler.
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


def test_fa4_pair_declared_graph():
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import FlashAttentionForward

    op = FlashAttentionForward(cluster=PAIR)
    op.describe()
    assert op.kv.scope.pair and op.kv.scope.full_ctas == (0,) and op.kv.scope.tx == 32768
    assert op.q.scope.tx == 65536  # Q: two 32 KB halves
    # S joins softmax[j] and correction of both CTAs on the leader: 2 x (128 + 128).
    assert op.s.scope.empty_ctas == (0,) and op.s.scope.empty_counts[0] == [512, 512]
    assert not op.stats.scope.pair and op.stats.scope.full_counts == {0: [128] * 2, 1: [128] * 2}


# Placement and scope checks


def _channel_op(cluster, ch, producer, consumers):
    """Place a warp-0 producer and consumers on channel ``ch``.

    ``producer`` selects its CTA placement. Each ``consumers`` entry is ``(warps, cta)``.
    """

    class Op(WarpSpecialized):
        def __init__(self):
            self.cluster = cluster
            self.ch = ch

        @role(warps=0, cta=producer)
        @cute.jit
        def prod(self, ch: Producer, ctx):
            return ch

    for i, (warps, cta) in enumerate(consumers):

        @role(warps=warps, cta=cta)
        @cute.jit
        def use(self, ch: Consumer, ctx):
            return ch

        use.__name__ = f"use{i}"
        setattr(Op, use.__name__, use)
    return Op()


class _LeaderOnlyTmem(WarpSpecialized):
    def __init__(self, cluster=PAIR):
        self.cluster = cluster
        self.tmem_plan = TmemPlan().region("acc", 32)

    @role(warps=0, cta="leader")
    @cute.jit
    def mma(self, tmem: Tmem, ctx):
        pass


@pytest.mark.parametrize(
    "make, message",
    [
        # Issues tcgen05 commits but runs in both CTAs.
        (lambda: _channel_op(PAIR, Slots(1, Threads, Umma), "all", [(1, "all")]),
         r"pin it with @role\(\.\.\., cta='leader'\)"),
        (lambda: _channel_op(PAIR, Slots(1, Tma(1024), Threads), "leader", [(1, "all")]),
         r"CTAs \[1\] wait on stages no TMA copy fills"),
        (lambda: _channel_op(PAIR, Slots(1, Threads, Threads, payload=(cutlass.Int32, (4,))),
                             "leader", [(1, "all")]),
         "layer-owned payload is per-CTA SMEM"),
        (lambda: _LeaderOnlyTmem(), "every Tmem role .* is leader-only"),
    ],
    ids=["unpinned_mma", "leader_tma_to_both", "pair_payload", "leader_only_tmem"],
)  # fmt: skip
def test_placement_errors(make, message):
    with pytest.raises(ProtocolError, match=message):
        make()._collect()


def test_valid_placements_derive_local_scopes():
    """Each CTA's copies land in its own stage (CtaGroup.ONE), where consumers wait. Mixed
    placement makes the channel pair-scoped, so consumers release both CTAs' producers, as every
    pair-scoped channel's signals reach the pair. A leader-only multicast along N arms each
    leader's barrier for the bytes landing in it and releases both leaders, whose copies reach
    it. Without pairs every CTA is a leader, so a leader Tmem role allocates."""
    mixed = _channel_op(
        PAIR,
        Slots(1, Tma(1024), Threads),
        "all",
        [([range(1, 2), range(2, 3)], "all"), (3, "leader")],
    )
    mixed._collect()
    sc = mixed.ch.scope
    assert sc.tx == 1024 and sc.tx_ctas == 1
    assert sc.full_targets == {0: (0,), 1: (1,)} and sc.empty_targets == {0: (0, 1), 1: (0, 1)}
    assert sc.empty_counts == {0: [160], 1: [160]}
    leaders = _channel_op(PAIRS_2X2, Ring(2, Tma(1024, multicast="n"), Umma), "leader",
                          [(1, "leader")])  # fmt: skip
    leaders._collect()
    sc = leaders.ch.scope
    assert sc.pair and sc.tx == 1024 and sc.tx_ctas == 1
    assert sc.full_targets == {0: (0,), 2: (2,)} and sc.empty_targets == {0: (0, 2), 2: (0, 2)}
    assert sc.empty_mask == 0b101 and sc.empty_counts[0] == [2, 2] and sc.tail
    _LeaderOnlyTmem(Cluster(2, 2))._collect()


@cute.jit
def _mma_atom_check(cluster: cutlass.Constexpr, group: cutlass.Constexpr):
    bf16, major = cutlass.BFloat16, cute.nvgpu.OperandMajorMode
    m = 256 if group == tcgen05.CtaGroup.TWO else 128
    tiled = sm100_utils.make_trivial_tiled_mma(
        bf16, bf16, major.K, major.K, cutlass.Float32, group, (m, 256)
    )
    check_atoms({"tiled_mma": tiled}, cluster)


def test_mismatched_mma_atom_is_rejected(requires_blackwell):
    cute.compile(_mma_atom_check, PAIR, tcgen05.CtaGroup.TWO)
    with pytest.raises(Exception, match="spans 1 CTA.*declares CtaPair"):
        cute.compile(_mma_atom_check, PAIR, tcgen05.CtaGroup.ONE)
    with pytest.raises(Exception, match="spans 2 CTA.*declares SingleCta"):
        cute.compile(_mma_atom_check, SingleCta(), tcgen05.CtaGroup.TWO)


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


# Toy protocols, launched as three clusters.
# ``ClusterToy``: producers in every CTA -> each pair's leader-only "MMA" role -> sinks in every
# CTA. ``ch``: Threads producers (one per slot) join on their pair leader's full barrier (picked
# by rank); the leader releases with a multicast tcgen05.commit (no MMAs needed, per-rank
# masks), so producers tail. ``back``: the leader commits and the sinks release on the leader.
# ``ThreadsToy``: Threads on both sides, the leader releasing and fanning out to its pair.
# Sinks write their count to ``out[cluster * 8 + rank]``.


class ClusterToy(WarpSpecialized):
    def __init__(self, cluster, mode: str = "ok"):
        self.cluster = cluster
        self.mode = mode
        self.ch = Slots(2, Threads, Umma)
        self.back = Ring(2, Umma, Threads)

    def get_name(self) -> str:
        return f"warpdrive_{type(self).__name__.lower()}_{self.cluster.get_key()}_{self.mode}"

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def prod(self, stage: Instance, ch: Producer[Instance], ctx):
        snapshot = ch  # "snapshot": a stale copy used later must not become the tail's state
        # Uneven commit counts per slot: slot 1 gets one more, so the slots' parities differ.
        for _ in cutlass.range(ctx.n + stage, unroll=1):
            h = ch.acquire(stage)
            if cutlass.const_expr(self.mode == "retire"):
                h.retire()
            else:
                h.commit()
        if cutlass.const_expr(self.mode == "snapshot"):
            if ctx.n > 0:  # a dynamic region that carries (rebuilds) the stale copy
                ctx.out[24 + stage] = snapshot.full_barrier(stage).toint()
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

    @cute.jit
    def _write(self, count, ctx):
        # warp-uniform: not inside the lane-0 branch
        slot = cute.arch.block_idx()[0] // self.cluster.size * 8 + self.cluster.rank()
        if cute.arch.lane_idx() == 0:
            ctx.out[slot] = count

    @role(warps=3)
    @cute.jit
    def sink(self, back: Consumer, ctx):
        count = cutlass.Int32(0)
        for _ in cutlass.range(2 * ctx.n + 1, unroll=1):
            with back.consume():
                count = count + 1
        self._write(count, ctx)
        return back

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(self.get_name())
        grid = (3 * self.cluster.size, 1, 1)
        self.kernel(out, n).launch(**self.launch_config(grid=grid), stream=stream)


class ThreadsToy(WarpSpecialized):
    def __init__(self, cluster, mode: str = "ok"):
        self.cluster = cluster
        self.mode = mode
        self.up = Ring(2, Threads, Threads)
        self.down = Ring(2, Threads, Threads)

    get_name, _write, kernel, __call__ = (
        ClusterToy.get_name,
        ClusterToy._write,
        ClusterToy.kernel,
        ClusterToy.__call__,
    )

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
        self._write(count, ctx)
        return down


@jit_cache
def compile_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name=op.get_name())


def test_toy_scopes():
    text = ClusterToy(PAIR).describe()
    # Threads from both CTAs join on the leader (32 + 32 per slot); the multicast commit lands
    # once on each CTA's empty barrier; producers drain because the release is asynchronous.
    assert "  2      mma         -  leader\n" in text
    assert (
        "  ch    Slots(2) prod[0]@0, prod[1]@1 -> mma\n"
        "        signal    Threads -> Umma\n"
        "        full      leader 64 x2\n"
        "        empty     leader 1 x2, peer 1 x2\n"
        "        scope     pair-scoped, release_mask 0b11, remote_scope cta, tail\n"
        "  back  Ring(2)  mma -> sink\n"
        "        signal    Umma -> Threads\n"
        "        full      leader 1 x2, peer 1 x2\n"
        "        empty     leader 64 x2\n"
    ) in text
    # Across pairs, each pair's producers join on their own leader, which releases its pair.
    op = ClusterToy(PAIRS_4X2)
    op.describe()
    assert op.ch.scope.full_targets == {r: (r - r % 2,) for r in range(8)}
    assert op.ch.scope.empty_mask == {r: 0b11 << r for r in range(0, 8, 2)}
    assert op.back.scope.empty_targets == {r: (r - r % 2,) for r in range(8)}
    # Threads on both sides: ``up`` joins both CTAs of a pair on its leader, which releases
    # both (two targets picked by rank); ``down`` fans the leader back out to both.
    op = ThreadsToy(PAIRS_4X2)
    op.describe()
    assert op.up.scope.empty_targets == {r: (r, r + 1) for r in range(0, 8, 2)}
    assert op.down.scope.full_targets == {r: (r, r + 1) for r in range(0, 8, 2)}
    assert op.up.scope.full_counts[4] == [64, 64] and op.down.scope.empty_counts[4] == [64, 64]


TOY_RUNS = [
    (ClusterToy, PAIR, "ok"),
    (ClusterToy, PAIR, "snapshot"),
    (ClusterToy, PAIRS_2X2, "ok"),
    (ClusterToy, PAIRS_4X2, "ok"),
    (ClusterToy, Cluster(2, 2), "ok"),
    (ThreadsToy, PAIRS_2X2, "ok"),
    (ThreadsToy, PAIRS_4X2, "ok"),
    (ThreadsToy, Cluster(2, 2), "ok"),
]


@pytest.mark.parametrize(
    "toy, cluster, mode", TOY_RUNS, ids=[f"{t.__name__}-{c.get_key()}-{m}" for t, c, m in TOY_RUNS]
)
def test_toy_runs(requires_blackwell, toy, cluster, mode):
    """Test remote arrivals, per-rank multicast commits, slot tails, and cluster exit.

    Each producer must tail only its bound slot. Slot 1 completes n + 1 phases; waiting
    for its parity from prod[0] would hang for even n.
    """
    fn = compile_toy(toy(cluster, mode))
    for n in (1, 2, 5, 64):
        out = torch.zeros(3 * 8 + 2, device="cuda", dtype=torch.int32)
        fn(out, n)
        torch.cuda.synchronize()
        got = out[:24].view(3, 8)[:, : cluster.size]
        want = 2 * n + 1 if toy is ClusterToy else n
        assert (got == want).all(), (n, got)


def test_retire_on_tailed_pair_channel_is_rejected(requires_blackwell):
    with pytest.raises(Exception, match="drain every stage before exit") as info:
        compile_toy(ClusterToy(PAIR, "retire"))
    assert isinstance(info.value, ProtocolError) or "ProtocolError" in repr(info.value)


# WorkStream in a pair: work units are clusters


class PairStream(WarpSpecialized):
    """Leader-only stream users; ``lead`` counts the items it sees."""

    def __init__(self, work):
        self.cluster = PAIR
        self.work = work

    @role(warps=0, cta="leader")
    @cute.jit
    def lead(self, work: WorkCursor, ctx):
        while work.valid:
            m, n, _ = work.item
            if cute.arch.lane_idx() == 0:
                cute.arch.atomic_add(ctx.out.iterator + m * ctx.n_tiles + n, cutlass.Int32(1))
            work.advance()
        return work

    @role(warps=1, cta="leader")
    @cute.jit
    def other(self, work: WorkCursor, ctx):
        while work.valid:
            work.advance()
        return work

    @cute.kernel
    def kernel(self, out: cute.Tensor, n_tiles: cutlass.Int32, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(out=out, n_tiles=n_tiles))

    @cute.jit
    def __call__(self, out: cute.Tensor, m_tiles: cutlass.Int32, n_tiles: cutlass.Int32,
                 workspace: cute.Tensor, max_ctas: cutlass.Int32, stream: cuda.CUstream):  # fmt: skip
        bound = self.work.bind((m_tiles, n_tiles, 1), workspace, max_ctas)
        self.kernel(out, n_tiles, bound).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_pair_stream(op: PairStream, name: str):
    def fake():
        return make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)

    i32 = cutlass.Int32(1)
    return compile_tvm_ffi(op, fake(), i32, i32, fake(), i32, name=name)


@pytest.mark.parametrize("schedule", ["oneshot", "static", "atomic", "clc"])
def test_pair_stream_with_leader_only_users(requires_blackwell, schedule):
    """All stream users leader-only: the broadcast ring is CTA-local to the leader, and its
    TMA-like publication still completes (st.async to the leader itself)."""
    op = PairStream(WorkStream(mapping=Raster(), schedule=SCHEDULES[schedule]))
    fn = compile_pair_stream(op, f"warpdrive_pair_stream_{schedule}")
    for m_tiles, n_tiles, cap in [(1, 1, 148), (7, 9, 4), (40, 30, 148)]:
        out = torch.zeros(m_tiles * n_tiles, device="cuda", dtype=torch.int32)
        ws, _ = op.work.launch_args("cuda")
        fn(out, m_tiles, n_tiles, ws, cap)
        torch.cuda.synchronize()
        assert out.tolist() == [1] * (m_tiles * n_tiles), (m_tiles, n_tiles, cap)


# Persistent GEMM on clusters

PAIR_GEMM_SHAPES = [
    (1, 8, 8),  # the peer CTA's half and most of the tile are out of bounds
    (300, 520, 136),  # ragged M, N, K
    (256, 224, 64),  # exactly one 2-CTA tile at the default tile_n
    (256, 256, 64),  # a second, 32-wide ragged tile along N
    (1000, 1000, 1000),  # 4 x 4 tiles: far fewer clusters than SMs
    (4097, 2056, 520),  # 17 x 9 tiles, ragged, more than one wave
]
GEMM_2X2_SHAPES = [
    (1, 8, 8),  # one cluster tile, three of its four CTAs entirely out of bounds
    (300, 520, 136),  # ragged; 520 / 256 needs a second pair along N in a partial cluster tile
    (256, 448, 64),  # exactly one 2x2 cluster tile at tile_n=224
    (1000, 3000, 520),  # an odd count of pair tiles along N: the last cluster tile is half empty
    (4097, 2056, 520),  # more than one wave
]
BUILTIN = ["oneshot", "static", "atomic", "clc"]


@pytest.mark.parametrize("m,n,k", PAIR_GEMM_SHAPES)
@pytest.mark.parametrize("schedule", BUILTIN)
def test_pair_gemm(requires_blackwell, schedule, m, n, k):
    """The pair's default tile_n (224)."""
    check_gemm(SCHEDULES[schedule], m, n, k, cluster=PAIR)


@pytest.mark.parametrize("tile_n", [256, 192, 128])
@pytest.mark.parametrize("schedule", ["static", "clc"])
def test_pair_gemm_tile_n(requires_blackwell, schedule, tile_n):
    """Other widths than the default 224: each CTA loads tile_n / 2 rows of B; ragged M and N
    edges."""
    check_gemm(SCHEDULES[schedule], 3000, 1000, 520, cluster=PAIR, tile_n=tile_n)


@pytest.mark.parametrize("m,n,k", GEMM_2X2_SHAPES)
@pytest.mark.parametrize("schedule", BUILTIN)
def test_2x2_gemm(requires_blackwell, schedule, m, n, k):
    """tile_n=224 (test_cluster_gemm_capped_grid runs 2x2 pairs at 256)."""
    check_gemm(SCHEDULES[schedule], m, n, k, cluster=PAIRS_2X2, tile_n=224)


@pytest.mark.parametrize(
    "cluster",
    [Cluster(4, 1, pair=True), PAIRS_4X2, Cluster(2, 2), Cluster(1, 2), Cluster(2, 1)],
    ids=["4x1_pairs", "4x2_pairs", "2x2", "1x2", "2x1"],
)
@pytest.mark.parametrize("schedule", ["static", "clc"])
def test_other_cluster_gemms(requires_blackwell, cluster, schedule):
    """B multicast along M (4x1 pairs, 2x1), A along N (1x2), and both (2x2 without pairs; 4x2
    pairs, where each stage is released by three leaders)."""
    for m, n, k in [(300, 520, 136), (3000, 1000, 520)]:
        check_gemm(SCHEDULES[schedule], m, n, k, cluster=cluster, tile_n=256)


@pytest.mark.parametrize(
    "cluster, schedule, max_ctas",
    [
        # 3 CTAs launch one pair and 7 one 2x2 cluster (the cap rounds down to whole clusters),
        # so its rings and the work ring wrap many times.
        *[(PAIR, s, c) for s in ["static", "atomic", "atomic1", "clc1"] for c in (3, 10)],
        *[(PAIRS_2X2, s, c) for s in ["static", "atomic", "clc"] for c in (7, 16)],
    ],
    ids=lambda v: v.get_key() if hasattr(v, "get_key") else str(v),
)
def test_cluster_gemm_capped_grid(requires_blackwell, cluster, schedule, max_ctas):
    tile_n = 256 if cluster.size > 2 else None
    check_gemm(SCHEDULES[schedule], 1100, 2064, 200, max_ctas=max_ctas, cluster=cluster,
               tile_n=tile_n, reps=3)  # fmt: skip


@pytest.mark.parametrize("cluster", [SingleCta(), PAIR], ids=["single", "pair"])
def test_odd_subtile_epilogue_rotates_buffers(requires_blackwell, cluster):
    """Keep the staging-buffer phase across tiles with an odd subtile count.

    tile_n=224 stores 7 subtiles through 2 buffers. Restarting at buffer 0 can overwrite
    the last subtile before its TMA store finishes, on any persistent cluster shape.
    """
    check_gemm(StaticPersistent(), 4097, 2056, 520, tile_n=224, max_ctas=16, cluster=cluster,
               reps=3)  # fmt: skip


# 2-CTA FA4-style forward


@pytest.mark.parametrize(
    "b,h,sq,sk,split_p",
    [
        (1, 1, 256, 128, False),  # the second Q tile of the only item is out of bounds
        (1, 1, 384, 256, True),  # ... partly out of bounds
        (2, 3, 640, 512, False),
        (8, 16, 1024, 384, True),  # 128 items: persistent clusters run several; odd key blocks
    ],
)
@pytest.mark.parametrize("schedule", BUILTIN)
def test_pair_fa4(requires_blackwell, schedule, b, h, sq, sk, split_p):
    check_fa4_twice(b, h, sq, sk, schedule=SCHEDULES[schedule], cluster=PAIR, split_p=split_p)


@pytest.mark.parametrize("split_p", [False, True], ids=["nosplit", "splitp"])
@pytest.mark.parametrize("schedule", ["static", "atomic", "clc"])
def test_pair_fa4_three_clusters_turnstile(requires_blackwell, schedule, split_p):
    """6 CTAs = 3 clusters, each running ~43 items: every slot and ring wraps."""
    check_fa4_twice(8, 16, 1024, 384, schedule=SCHEDULES[schedule], max_ctas=6, turnstile=True,
                    cluster=PAIR, split_p=split_p)  # fmt: skip
