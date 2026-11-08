"""2-CTA (CtaPair) support in the warp-role layer: derived scopes, trace-time checks, a toy pair
protocol, and the 2-CTA persistent GEMM and FA4 examples under every scheduling schedule."""

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.nvgpu import tcgen05

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    AtomicPersistent,
    ClusterLaunchControl,
    Consumer,
    CtaPair,
    Instance,
    OneShot,
    Producer,
    ProtocolError,
    Raster,
    Ring,
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
from attn_gym.warpdrive.cluster import check_atoms

SCHEDULES = {
    "oneshot": OneShot(),
    "static": StaticPersistent(),
    "atomic": AtomicPersistent(stages=2),
    "atomic1": AtomicPersistent(stages=1),
    "clc": ClusterLaunchControl(stages=2),
    "clc1": ClusterLaunchControl(stages=1),
}
PAIR = CtaPair()


# --------------------------------------------------------------------------- #
# Toy pair protocol: producers in both CTAs -> leader-only "MMA" role -> sinks in both CTAs.
# ``ch``: Threads producers (one per slot, both CTAs) joined on the leader's full barrier; the
# leader releases with a multicast tcgen05.commit (no MMAs needed), so producers tail.
# ``back``: the leader commits (multicast) and the sinks of both CTAs release on the leader.
# --------------------------------------------------------------------------- #


class PairToy(WarpSpecialized):
    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.cluster = PAIR
        self.ch = Slots(2, Threads, Umma)
        self.back = Ring(2, Umma, Threads)

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
                ctx.out[3 + stage] = snapshot.full_barrier(stage).toint()
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
        if cute.arch.lane_idx() == 0:
            ctx.out[2] = 2 * ctx.n + 1
        return ch, back

    @role(warps=3)
    @cute.jit
    def sink(self, back: Consumer, ctx):
        count = cutlass.Int32(0)
        for _ in cutlass.range(2 * ctx.n + 1, unroll=1):
            with back.consume():
                count = count + 1
        rank = self.cluster.rank()  # warp-uniform: not inside the lane-0 branch
        if cute.arch.lane_idx() == 0:
            ctx.out[rank] = count
        return back

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(f"warpdrive_pair_toy_{self.mode}")
        self.kernel(out, n).launch(**self.launch_config(grid=(2, 1, 1)), stream=stream)


@jit_cache
def compile_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name=f"warpdrive_pair_toy_{op.mode}")


def test_pair_toy_derived_scopes():
    op = PairToy()
    text = op.describe()
    assert "  cluster  CTA pair (2, 1, 1), tcgen05 CtaGroup.TWO, leader = rank 0" in text
    assert [(r.name, r.cta) for r in op._ws_roles if r.name == "mma"] == [("mma", "leader")]
    # Threads from both CTAs join on the leader (32 + 32 per slot); the multicast commit lands
    # once on each CTA's empty barrier; producers drain because the release is asynchronous.
    ch = op.ch.scope
    assert ch.pair and ch.full_ctas == (0,) and ch.full_counts[0] == [64, 64]
    assert ch.empty_ctas == (0, 1) and ch.empty_counts == {0: [1, 1], 1: [1, 1]}
    assert ch.empty_mask == 0b11 and ch.tail and op.ch.remote_scope == "cta"
    back = op.back.scope
    assert back.full_ctas == (0, 1) and back.full_counts == {0: [1, 1], 1: [1, 1]}
    assert back.empty_ctas == (0,) and back.empty_counts[0] == [64, 64]
    # Key lines of the channel block: per-CTA counts with uniform slots collapsed.
    assert "  ch    Slots(2) prod[0]@0, prod[1]@1 -> mma" in text
    assert "full      leader 64 x2\n" in text
    assert "scope     pair-scoped, release_mask 0b11, remote_scope cta, tail" in text


@pytest.mark.parametrize("mode", ["ok", "snapshot"])
@pytest.mark.parametrize("n", [1, 2, 5, 64])
def test_pair_toy_runs(requires_blackwell, n, mode):
    """Remote arrivals, multicast commits, slot-bound producer tails, and the exit protocol.
    Regression: tails once waited every slot, so prod[0] waited slot 1's parity and hung for even
    n (slot 1 completes n + 1 phases)."""
    out = torch.full((5,), -1, device="cuda", dtype=torch.int32)
    compile_toy(PairToy(mode))(out, n)
    torch.cuda.synchronize()
    assert out[:3].tolist() == [2 * n + 1] * 3


def test_retire_on_tailed_pair_channel_is_rejected(requires_blackwell):
    with pytest.raises(Exception, match="drain every stage before exit") as info:
        compile_toy(PairToy("retire"))
    assert isinstance(info.value, ProtocolError) or "ProtocolError" in repr(info.value)


# --------------------------------------------------------------------------- #
# Host-side placement and scope checks
# --------------------------------------------------------------------------- #


class _Unpinned(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.ch = Slots(1, Threads, Umma)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        return ch

    @role(warps=1)  # issues tcgen05 commits but runs in both CTAs
    @cute.jit
    def mma(self, ch: Consumer, ctx):
        return ch


class _LeaderTmaToBoth(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.ch = Slots(1, Tma(1024), Threads)

    @role(warps=0, cta="leader")
    @cute.jit
    def load(self, ch: Producer, ctx):
        return ch

    @role(warps=1)
    @cute.jit
    def use(self, ch: Consumer, ctx):
        return ch


class _BothTmaToMixed(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.ch = Slots(1, Tma(1024), Threads)

    @role(warps=0)
    @cute.jit
    def load(self, ch: Producer, ctx):
        return ch

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def use(self, ch: Consumer, ctx):
        return ch

    @role(warps=3, cta="leader")
    @cute.jit
    def lead(self, ch: Consumer, ctx):
        return ch


class _LeaderOnlyTmem(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.tmem_plan = TmemPlan().region("acc", 32)

    @role(warps=0, cta="leader")
    @cute.jit
    def mma(self, tmem: Tmem, ctx):
        pass


class _PairPayload(WarpSpecialized):
    def __init__(self):
        self.cluster = PAIR
        self.ch = Slots(1, Threads, Threads, payload=(cutlass.Int32, (4,)))

    @role(warps=0, cta="leader")
    @cute.jit
    def prod(self, ch: Producer, ctx):
        return ch

    @role(warps=1)
    @cute.jit
    def use(self, ch: Consumer, ctx):
        return ch


def test_pair_placement_errors():
    with pytest.raises(ProtocolError, match=r"pin it with @role\(\.\.\., cta='leader'\)"):
        _Unpinned()._collect()
    with pytest.raises(ProtocolError, match=r"CTAs \[1\] wait on stages no TMA copy fills"):
        _LeaderTmaToBoth()._collect()
    with pytest.raises(ProtocolError, match="layer-owned payload is per-CTA SMEM"):
        _PairPayload()._collect()
    with pytest.raises(ProtocolError, match="every Tmem role .* is leader-only"):
        _LeaderOnlyTmem()._collect()
    with pytest.raises(ValueError, match="role cta must be one of"):
        role(warps=0, cta="peer")


def test_extra_leader_consumer_of_local_tma():
    """Each CTA's copies land in its own stage (CtaGroup.ONE), where consumers wait. Mixed
    placement makes the channel pair-scoped, so consumers release both CTAs' producers, as every
    pair-scoped channel's signals reach the pair."""
    op = _BothTmaToMixed()
    op._collect()
    sc = op.ch.scope
    assert sc.tx == 1024 and sc.tx_ctas == 1
    assert sc.full_targets == {0: (0,), 1: (1,)} and sc.empty_targets == {0: (0, 1), 1: (0, 1)}
    assert sc.empty_counts == {0: [160], 1: [160]}


def test_leader_placement_is_a_noop_for_one_cta():
    """The same role source runs on one CTA: `cta="leader"` places it in the only CTA."""
    from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm

    op = PersistentGemm()
    text = op.describe()
    assert "leader" not in text and "pair-scoped" not in text and "scope" not in text
    assert op.ab.scope.full_counts == {0: [1] * 4} and not op.ab.scope.pair
    assert "full      1 x4, tx 49152 B\n" in text


def test_pair_gemm_declared_graph():
    from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm

    op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=PAIR, tile_n=256)
    text = op.describe()
    chans = op._ws_chans
    # Both CTAs' TMA bytes (2 x 32 KB) complete on the leader, armed once by its producer.
    ab = chans["ab"].scope
    assert ab.pair and ab.full_ctas == (0,) and ab.full_counts[0] == [1] * 6
    assert ab.tx == 65536 and ab.empty_mask == 0b11 and ab.tail
    acc = chans["acc"].scope
    assert acc.full_counts == {0: [1, 1], 1: [1, 1]} and acc.full_mask == 0b11
    assert acc.empty_ctas == (0,) and acc.empty_counts[0] == [256, 256]
    # The leader's scheduler publishes into both CTAs (st.async bytes), and every consumer
    # releases the leader's slot: epilogue and load in both CTAs, the leader-only MMA and
    # scheduler once: 2 x (128 + 32) + 32 + 32.
    sched = next(r for r in op._ws_roles if r.name == "work.scheduler")
    assert (sched.first_warp, sched.regs, sched.cta) == (6, 40, "leader")
    work = chans["work_ring"].scope
    assert work.full_counts == {0: [1, 1], 1: [1, 1]} and work.tx == 16
    assert work.empty_ctas == (0,) and work.empty_counts[0] == [384, 384]
    # The st.async payload completes by bytes; the releases only hand the slot back, so they
    # count at CTA scope (a cluster-scope release per thread cost ~1 us per item: examples.md).
    assert chans["work_ring"].remote_scope == "cta"
    assert not chans["work_ring"].remote_scope_overridden
    clc = chans["work_clc"].scope
    assert not clc.pair and clc.full_counts[0] == [1]
    # Golden lines.
    assert "  6      work.scheduler    40  leader\n" in text
    assert "full      leader 1 x6, tx 65536 B (leader-armed)\n" in text
    assert "scope     pair-scoped, release_mask 0b11, tail\n" in text
    assert "  work_ring  Ring(2)  work.scheduler -> epilogue, load, mma, work.scheduler\n" in text
    assert (
        "scope     pair-scoped, remote_scope cta\n             storage   payload Int32(4,)\n"
        in text
    )


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


@cute.jit
def _mma_atom_check(cluster: cutlass.Constexpr, group: cutlass.Constexpr):
    bf16, major = cutlass.BFloat16, cute.nvgpu.OperandMajorMode
    m = 256 if group == tcgen05.CtaGroup.TWO else 128
    tiled = sm100_utils.make_trivial_tiled_mma(
        bf16, bf16, major.K, major.K, cutlass.Float32, group, (m, 256)
    )
    check_atoms({"tiled_mma": tiled}, cluster)


def test_mismatched_mma_atom_is_rejected(requires_blackwell):
    from attn_gym.warpdrive import SingleCta

    cute.compile(_mma_atom_check, PAIR, tcgen05.CtaGroup.TWO)
    with pytest.raises(Exception, match="spans 1 CTA.*declares CtaPair"):
        cute.compile(_mma_atom_check, PAIR, tcgen05.CtaGroup.ONE)
    with pytest.raises(Exception, match="spans 2 CTA.*declares SingleCta"):
        cute.compile(_mma_atom_check, SingleCta(), tcgen05.CtaGroup.TWO)


# --------------------------------------------------------------------------- #
# WorkStream in a pair: work units are clusters
# --------------------------------------------------------------------------- #


class PairStream(WarpSpecialized):
    """Every stream user counts the items it sees; ``cta`` places them."""

    def __init__(self, schedule, cta: str):
        self.cluster = PAIR
        self.cta = cta
        self.work = WorkStream(mapping=Raster(), schedule=schedule)

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
    def __call__(
        self,
        out: cute.Tensor,
        m_tiles: cutlass.Int32,
        n_tiles: cutlass.Int32,
        workspace: cute.Tensor,
        max_ctas: cutlass.Int32,
        stream: cuda.CUstream,
    ):
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
    op = PairStream(SCHEDULES[schedule], "leader")

    fn = compile_pair_stream(op, f"warpdrive_pair_stream_{schedule}")
    for m_tiles, n_tiles, cap in [(1, 1, 148), (7, 9, 4), (40, 30, 148)]:
        out = torch.zeros(m_tiles * n_tiles, device="cuda", dtype=torch.int32)
        ws, _ = op.work.launch_args("cuda")
        fn(out, m_tiles, n_tiles, ws, cap)
        torch.cuda.synchronize()
        assert out.tolist() == [1] * (m_tiles * n_tiles), (m_tiles, n_tiles, cap)


def test_pair_record_workspace_covers_whole_clusters(requires_blackwell):
    op = PairStream(OneShot(), "leader")
    op.work = WorkStream(mapping=Raster(), schedule=StaticPersistent(), record=3)
    op.describe()  # binds the roles the log is sized by
    ws, cap = op.work.launch_args("cuda", max_ctas=1)
    roles = len(op.work.roles)
    assert cap == 1 and ws.numel() >= 4 + roles * 2 * 3 * 2  # one cluster = two CTAs


# --------------------------------------------------------------------------- #
# 2-CTA persistent GEMM
# --------------------------------------------------------------------------- #

GEMM_SHAPES = [
    (1, 8, 8),  # the peer CTA's half and most of the tile are out of bounds
    (300, 520, 136),  # ragged M, N, K
    (256, 224, 64),  # exactly one 2-CTA tile at the default tile_n
    (256, 256, 64),  # a second, 32-wide ragged tile along N
    (1000, 1000, 1000),  # 4 x 4 tiles: far fewer clusters than SMs
    (4097, 2056, 520),  # 17 x 9 tiles, ragged, more than one wave
]


def _check_gemm(schedule, m, n, k, max_ctas=None, reps=2, tile_n=None):
    """``tile_n=None`` runs the pair's default width (224)."""
    from attn_gym.warpdrive.examples.persistent_gemm import persistent_gemm

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    first = None
    for _ in range(reps):  # later launches reuse the (self-reset) atomic counter
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        persistent_gemm(
            a, b, out, schedule=schedule, max_ctas=max_ctas, cluster=PAIR, tile_n=tile_n
        )
        err = (out.float() - ref).abs().max().item()
        assert err <= 2 * torch_err + 1e-3, (schedule, (m, n, k), err, torch_err)
        if first is None:
            first = out
        assert torch.equal(out, first), "2-CTA GEMM is not bitwise deterministic"


@pytest.mark.parametrize("m,n,k", GEMM_SHAPES)
@pytest.mark.parametrize("schedule", ["oneshot", "static", "atomic", "clc"])
def test_pair_gemm(requires_blackwell, schedule, m, n, k):
    _check_gemm(SCHEDULES[schedule], m, n, k)


def test_default_tile_n():
    """224 for a pair alone (the measured default), 256 for one CTA and larger clusters."""
    from attn_gym.warpdrive import Cluster, SingleCta
    from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm

    assert PersistentGemm().tile_n == 256
    assert PersistentGemm(cluster=PAIR).tile_n == 224
    assert PersistentGemm(cluster=Cluster(2, 1, pair=True)).tile_n == 224
    assert PersistentGemm(cluster=Cluster(1, 2)).tile_n == 256
    assert PersistentGemm(cluster=Cluster(2, 2, pair=True)).tile_n == 256
    assert PersistentGemm(cluster=SingleCta(), tile_n=224).tile_n == 224


@pytest.mark.parametrize("tile_n", [256, 192, 128])
@pytest.mark.parametrize("schedule", ["static", "clc"])
def test_pair_gemm_tile_n(requires_blackwell, schedule, tile_n):
    """Other widths than the default 224: each CTA loads tile_n / 2 rows of B; ragged M and N
    edges."""
    _check_gemm(SCHEDULES[schedule], 3000, 1000, 520, tile_n=tile_n)


def test_pair_gemm_odd_subtiles_rotate_c_buffers(requires_blackwell):
    """tile_n=224 stores 7 subtiles per tile through 2 C staging buffers. Regression: each tile
    restarted at buffer 0 and rewrote the buffer the previous tile's last TMA store could still
    be reading."""
    _check_gemm(SCHEDULES["static"], 4097, 2056, 520, max_ctas=16, reps=3, tile_n=224)


@pytest.mark.parametrize("schedule", ["static", "atomic", "atomic1", "clc1"])
@pytest.mark.parametrize("max_ctas", [2, 3, 10])
def test_pair_gemm_capped_grid(requires_blackwell, schedule, max_ctas):
    """2 and 3 CTAs launch one cluster, so its rings and the work ring wrap many times."""
    _check_gemm(SCHEDULES[schedule], 1100, 1032, 200, max_ctas=max_ctas, reps=3)


def test_pair_gemm_counter_resets(requires_blackwell):
    from attn_gym.warpdrive.examples.persistent_gemm import _OPS, default_tile_n

    _check_gemm(SCHEDULES["atomic"], 1500, 1800, 64, max_ctas=8, reps=3)
    op = _OPS[(SCHEDULES["atomic"], None, PAIR, default_tile_n(PAIR))]
    ws, _ = op.work.launch_args("cuda")
    torch.cuda.synchronize()
    assert ws[:2].tolist() == [0, 0]


def test_pair_gemm_compiles_once_across_shapes(requires_blackwell):
    from attn_gym.warpdrive.examples.persistent_gemm import (
        PersistentGemm,
        compile_persistent_gemm,
    )

    compile_persistent_gemm.cache_clear()
    op = PersistentGemm(schedule=AtomicPersistent(stages=3), cluster=PAIR, ab_stages=5)
    fns = set()
    for m, n, k, cap in [
        (1, 8, 8, 148),
        (16, 256, 64, 3),
        (257, 264, 136, 148),
        (4096, 520, 24, 64),
    ]:
        torch.manual_seed(m)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        fn = compile_persistent_gemm(op)
        fns.add(id(fn))
        ws, _ = op.work.launch_args("cuda")
        fn(a, b, out, ws, cap)
        ref = torch.nn.functional.linear(a.float(), b.float())
        assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item() + 1e-2
    info = compile_persistent_gemm.cache_info()
    assert info.currsize == 1 and info.hits + info.misses == 4 and len(fns) == 1


# --------------------------------------------------------------------------- #
# 2-CTA FA4-style forward
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("split_p", [False, True], ids=["nosplit", "splitp"])
@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize(
    "b,h,sq,sk", [(1, 1, 256, 128), (1, 1, 384, 256), (2, 3, 640, 512), (8, 16, 1024, 384)]
)
@pytest.mark.parametrize("schedule", ["oneshot", "static", "atomic", "clc"])
def test_pair_fa4(requires_blackwell, schedule, b, h, sq, sk, nonuniform, split_p):
    """sq=256/384: the second Q tile of the only item is (partly) out of bounds; (8, 16, 1024)
    has 128 items, so persistent clusters run several; sk=384 is an odd key-block count."""
    _check_fa4(SCHEDULES[schedule], b, h, sq, sk, nonuniform, split_p)


@pytest.mark.parametrize("schedule", ["static", "atomic", "clc"])
def test_pair_fa4_three_clusters_turnstile(requires_blackwell, schedule):
    """6 CTAs = 3 clusters, each running ~43 items: every slot and ring wraps."""
    _check_fa4(SCHEDULES[schedule], 8, 16, 1024, 384, True, True, max_ctas=6, turnstile=True)


def _check_fa4(schedule, b, h, sq, sk, nonuniform, split_p, max_ctas=None, turnstile=False):
    pytest.importorskip("flash_attn.cute")
    from test_warpdrive import fa4_inputs

    from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

    q, k, v = fa4_inputs(b, h, sq, sk, nonuniform)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)
    ref_lse = torch.logsumexp((qf @ kf.transpose(-1, -2)) / math.sqrt(128), dim=-1)
    tol = 2**-7 * ref.abs().max().item() + 4e-3
    first = None
    for _ in range(2):
        o, lse = fa4_fwd(
            q,
            k,
            v,
            schedule=schedule,
            max_ctas=max_ctas,
            turnstile=turnstile,
            cluster=PAIR,
            split_p=split_p,
        )
        assert (o.float() - ref).abs().max().item() <= tol
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)
        if first is None:
            first = o
        assert torch.equal(o, first)
