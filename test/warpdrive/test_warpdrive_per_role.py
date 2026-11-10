"""Per-role consumer mechanisms: ``Ring(n, Tma(b), {"mma": Umma, "cuda": Threads})``.

One joining consumer side where an MMA warp and CUDA threads both read every stage. Each role
contributes its own arrivals and signals and fences with its own mechanism.
"""

import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.testing.warpdrive import alias_section, status
from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    Local,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    Sharing,
    SmemPlan,
    StageView,
    Threads,
    Tma,
    Umma,
    Warps,
    WarpSpecialized,
    WorkCursor,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.gemm import TILE_K, TILE_M, Gemm, GemmParams
from attn_gym.warpdrive.examples.tma_tiles import tma_partition_tiles

JOINT = {"mma": Umma, "cuda": Threads}


class Model(WarpSpecialized):
    """Single-CTA model: roles loop ``n`` times; ``layouts()`` gives the SMEM plan's layouts."""

    def layouts(self):
        return {}

    @cute.kernel
    def kernel(self, n: cutlass.Int32):
        self.begin_roles(layouts=self.layouts())
        self.run_roles(SimpleNamespace(n=n))

    @cute.jit
    def __call__(self, unused: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("warpdrive_per_role_model")
        self.kernel(n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


@jit_cache
def compile_model(op, name: str):
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(op, unused, cutlass.Int32(1), name=name)


class JointRing(Model):
    """A TMA-filled ring read by one MMA warp and a four-warp CUDA role.

    ``h`` holds X; Y overlaps X, so the producer's TMA refill of Y must follow both readers'
    releases: the MMA's by its commit, the CUDA threads' by their proxy fence.
    """

    def __init__(self, consumer=None, roles=None, **channel):
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        y = self.smem_plan.buffer("Y", cutlass.Int32, overlaps=x)
        self.h = Ring(2, Tma(256), consumer or JOINT, buffer={"X": x, "Y": y}, **channel)
        self.roles = roles

    def layouts(self):
        return {"X": cute.make_layout((64, 2)), "Y": cute.make_layout((64, 2))}

    @role(warps=0)
    @cute.jit
    def load(self, ctx, *, h: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with h.produce() as s:
                s.access("Y", write=True, by="tma")
        return h

    @role(warps=5)
    @cute.jit
    def mma(self, ctx, *, h: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with h.consume() as s:
                s.access("X", read=True, by="umma")
        return h

    @role(warps=range(1, 5))
    @cute.jit
    def cuda(self, ctx, *, h: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with h.consume() as s:
                s.access("X", read=True)  # ld.shared of the stage
        return h


# Declaration


@pytest.mark.parametrize(
    "consumer, empty",
    [(JOINT, 1 + 128), ({"mma": Umma, "cuda": Warps}, 1 + 4)],
    ids=["threads", "warps"],
)
def test_counts_sum_each_roles_contribution(consumer, empty):
    """The empty count is the per-role sum; nothing but an override can change it."""
    op = JointRing(consumer=consumer)
    op.describe()
    assert op.h.consumer_roles == ("mma", "cuda")
    assert op.h.scope.empty_counts[0] == [empty, empty]
    assert op.h.scope.full_counts[0] == [1, 1]
    assert op.h.release_fence == ("smem_async",)
    assert op.h.fences("consumer", "cuda") == ("smem_async",)
    assert op.h.fences("consumer", "mma") == ()


def test_describe_shows_per_role_signals_counts_and_fences():
    text = JointRing().describe()
    assert "h  Ring(2)  load -> cuda, mma" in text, text
    assert "signal    Tma(256 B) -> {mma: Umma, cuda: Threads}" in text, text
    assert "empty     129 x2" in text, text
    assert "release_fence smem_async (cuda)" in text, text


def test_describe_shows_an_overridden_thread_release_fence():
    text = JointRing(release_fence=()).describe()
    assert "release_fence smem_async -> none  (differs from derived)" in text, text


@pytest.mark.parametrize("fmt", ["mermaid", "dot", "html"])
def test_graph_formats_render_per_role_signals(fmt):
    text = JointRing().describe(format=fmt)
    assert "{mma: Umma, cuda: Threads}" in text
    if fmt == "html":  # the junction's consumer bar splits between the two mechanisms
        assert 'class="bar umma"' in text and 'class="bar threads"' in text


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"producer": {"a": Threads}}, "consumer-only"),
        ({"consumer": {}}, "non-empty"),
        ({"consumer": {"mma": Tma(16)}}, "producer mechanism"),
        ({"consumer": {"mma": object()}}, "mechanism must be"),
        ({"consumer_roles": ("mma", "cuda")}, "drop consumer_roles"),
        ({"consumer_sharing": Sharing.ROUND_ROBIN}, "Sharing.JOIN"),
        ({"consumer_sharing": Sharing.EXCLUSIVE}, "Sharing.JOIN"),
        ({"producer": Threads, "sync": "named"}, "mbarrier channel"),
        ({"producer": Tma(16, multicast="n")}, "CTA-local only"),
        ({"producer": Tma(16, fills="pair")}, "CTA-local only"),
        ({"external": "consumer", "override": {"empty_counts": 1}}, "external='consumer'"),
        ({"consumer": {"mma": Umma}, "release_fence": "smem_async"}, "has none"),
    ],
    ids=[
        "producer_dict",
        "empty",
        "tma",
        "unknown",
        "roles",
        "round_robin",
        "exclusive",
        "named",
        "multicast",
        "fills",
        "external",
        "umma_fence",
    ],
)
def test_declaration_rejects(kwargs, match):
    args = {"producer": Tma(16), "consumer": JOINT, **kwargs}
    with pytest.raises(ValueError, match=match):
        Ring(2, args.pop("producer"), args.pop("consumer"), **args)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"consumer": {"mma": Umma, "cuda": Threads, "epi": Threads}}, r"\['epi'\] take no"),
        ({"consumer": {"mma": Umma}}, "consumers are declared as"),
        ({"roles": {"mma": RoleConfig(range(5, 7))}}, "spans 2 warps"),
        ({"roles": {"mma": RoleConfig([range(5, 6), range(6, 7)])}}, "signals once"),
        ({"consumer": {"mma": Threads, "cuda": Umma}}, "spans 4 warps"),
    ],
    ids=["unbound_key", "unkeyed_role", "wide_umma", "two_umma_instances", "umma_warpgroup"],
)
def test_binding_rejects(kwargs, match):
    with pytest.raises(ProtocolError, match=match):
        JointRing(**kwargs).describe()


class PairPlaced(JointRing):
    def __init__(self):
        super().__init__()
        self.cluster = CtaPair()

    @role(warps=5, cta="leader")
    @cute.jit
    def mma(self, ctx, *, h: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with h.consume():
                pass
        return h


def test_cross_cta_placement_is_rejected():
    with pytest.raises(ProtocolError, match="CTA-local only"):
        PairPlaced().describe()


# Tracing: signals, fences and the alias diagnostic


def record_arch_calls(monkeypatch, names) -> list[str]:
    emitted = []
    for name in names:
        original = getattr(cute.arch, name)

        def record(*args, _name=name, _original=original, **kwargs):
            emitted.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cute.arch, name, record)
    return emitted


@pytest.mark.parametrize("fence", [None, ()], ids=["derived", "opt_out"])
def test_each_role_signals_and_fences_with_its_own_mechanism(monkeypatch, fence):
    """The CUDA threads fence and arrive; the MMA warp commits without a fence. Where the
    derived fence is declared away, the alias diagnostic flags the CUDA read, not the MMA's."""
    names = ("fence_view_async_shared", "mbarrier_arrive")
    emitted = record_arch_calls(monkeypatch, names)
    commits = []
    original_commit = cute.nvgpu.tcgen05.commit
    monkeypatch.setattr(
        cute.nvgpu.tcgen05,
        "commit",
        lambda *a, **k: (commits.append(a), original_commit(*a, **k))[1],
    )
    channel = {} if fence is None else {"release_fence": fence}
    op = JointRing(**channel)
    tag = "derived" if fence is None else "opt_out"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_model.__wrapped__(op, f"per_role_fences_{tag}")  # traced, not cached
    assert len(commits) == 1  # the MMA warp's release
    assert emitted.count("mbarrier_arrive") == 1  # the CUDA threads' release
    if fence is None:
        assert emitted == ["fence_view_async_shared", "mbarrier_arrive"]
        assert status(op, "Y ~ X") == "no hazard found", alias_section(op)
    else:
        assert emitted == ["mbarrier_arrive"]
        text = alias_section(op)
        assert status(op, "Y ~ X") == "unverified", text
        assert "cuda" in text and "no fence.proxy.async" in text, text
        assert any("alias diagnostic" in str(w.message) for w in caught)


class WritesForJointReaders(Model):
    """Thread-written SMEM read by an MMA warp and a ``Warps`` role: the writers fence for
    the MMA; the MMA warp commits; each ``Warps`` reader warp syncs and elects one arrival."""

    def __init__(self):
        self.ring = Ring(2, Threads, {"mma": Umma, "cuda": Warps}, payload=(cutlass.Int32, (32,)))

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ring: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.produce() as h:
                h.data[cute.arch.lane_idx()] = 1
        return ring

    @role(warps=1)
    @cute.jit
    def mma(self, ctx, *, ring: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.consume():
                pass
        return ring

    @role(warps=range(2, 4))
    @cute.jit
    def cuda(self, ctx, *, ring: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.consume() as h:
                _ = h.data[cute.arch.lane_idx()]
        return ring


def test_thread_writers_fence_for_an_umma_role_and_warps_readers_elect(monkeypatch):
    names = ("fence_view_async_shared", "sync_warp", "mbarrier_arrive")
    emitted = record_arch_calls(monkeypatch, names)
    op = WritesForJointReaders()
    assert op.ring.commit_fence == ("smem_async",)
    assert op.ring.release_fence == ()
    compile_model.__wrapped__(op, "per_role_commit_fence")  # traced, not cached
    assert op.ring.scope.empty_counts[0] == [1 + 2, 1 + 2]
    # Producer: fence, then arrive. Warps reader: warp sync, then one elected arrive.
    assert emitted == [
        "fence_view_async_shared",
        "mbarrier_arrive",
        "sync_warp",
        "mbarrier_arrive",
    ]


# Runtime: a GEMM whose A/B ring is also read by CUDA threads


@dataclass(frozen=True)
class JointParams(GemmParams):
    sA: cute.Tensor  # the A stages as flat elements: (TILE_M * TILE_K, stage)
    sums: cute.Tensor  # per-thread A sums: (128, k_tiles, m_tiles, n_tiles)


CUDA_DELAY = 20_000  # cycles the last CUDA warp waits before reading each stage


@cute.jit
def spin(cycles):
    start = cute.arch.clock64()
    while cute.arch.clock64() - start < cycles:
        pass


class JointGemm(Gemm):
    """``Gemm`` whose A/B ring is also consumed by four CUDA warps summing each A stage.

    The CUDA role's last warp reads late: a stage released before it reads (a short count)
    is refilled under it and its sums change.
    """

    def __init__(self):
        super().__init__(tile_n=64)
        a, b = self.smem_plan.buffers["A"], self.smem_plan.buffers["B"]
        stage_bytes = (TILE_M + self.tile[1]) * TILE_K * 2
        self.ab = Ring(self.ab.n, Tma(stage_bytes), JOINT, buffer={"A": a, "B": b})

    def get_name(self) -> str:
        return "warpdrive_per_role_gemm"

    @role(warps=range(8, 12), regs=160)
    @cute.jit
    def cuda(self, params: JointParams, *, ab: Consumer, work: WorkCursor, me: Local):
        tid = me.tid
        while work.valid:
            m, n, _ = work.item
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.consume() as s:
                    if me.warp == me.warps - 1:
                        spin(CUDA_DELAY)
                    s.access("A", read=True)
                    acc = cutlass.Float32(0)
                    for e in cutlass.range(TILE_M * TILE_K // me.threads, unroll_full=True):
                        acc += params.sA[tid + me.threads * e, s.index].to(cutlass.Float32)
                    params.sums[tid, k, m, n] = acc
            work.advance()
        return ab, work

    @cute.kernel
    def kernel(self, tiled_mma, tma_a, mA, tma_b, mB, mC, sums, a_layout, b_layout, bound):
        self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout})
        sA, sB = self.buffer("A"), self.buffer("B")
        thr_mma = tiled_mma.get_slice(0)
        gA = cute.local_tile(mA, self.tile, (None, None, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self.tile, (None, None, None), proj=(None, 1, 1))
        tAsA, tAgA = tma_partition_tiles(tma_a, sA, thr_mma.partition_A(gA))
        tBsB, tBgB = tma_partition_tiles(tma_b, sB, thr_mma.partition_B(gB))
        acc_shape = tiled_mma.partition_shape_C(self.tile[:2])
        acc_fake = tiled_mma.make_fragment_C(acc_shape)
        acc_staged = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        cC = cute.make_identity_tensor(mC.shape)
        tile_c = cute.local_tile(mC, self.tile, (None, None, None), proj=(1, 1, None))
        tile_cc = cute.local_tile(cC, self.tile, (None, None, None), proj=(1, 1, None))
        epi_cols = cute.size(acc_fake, mode=[0, 1]) // self.epi_subtiles
        params = JointParams(
            tiled_mma=tiled_mma,
            tma_a=tma_a,
            tma_b=tma_b,
            tAsA=StageView(tAsA, axis=1),
            tAgA=tAgA,
            tBsB=StageView(tBsB, axis=1),
            tBgB=tBgB,
            tCrA=StageView(tiled_mma.make_fragment_A(sA), axis=3),
            tCrB=StageView(tiled_mma.make_fragment_B(sB), axis=3),
            num_k_tiles=cute.size(gA, mode=[3]),
            acc_fake=acc_fake,
            acc_layout=acc_staged.layout,
            epi_tiler=((cute.size(acc_fake, mode=[0, 0]), epi_cols),),
            tCgC=thr_mma.partition_C(tile_c),
            tCcC=thr_mma.partition_C(tile_cc),
            mC=mC,
            sA=cute.make_tensor(sA.iterator, cute.make_layout((TILE_M * TILE_K, self.ab.n))),
            sums=sums,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(self, a, b, c, sums, stream):
        op = tcgen05.MmaF16BF16Op(
            self.dtype,
            cutlass.Float32,
            (TILE_M, self.tile[1], 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.tile, self.dtype, self.ab.n)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.tile, self.dtype, self.ab.n)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mA = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, a, cute.select(a_layout, mode=[0, 1, 2]), self.tile, tiled_mma
        )
        tma_b, mB = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, b, cute.select(b_layout, mode=[0, 1, 2]), self.tile, tiled_mma
        )
        m_tiles, n_tiles = cute.ceil_div(c.layout.shape, self.tile[:2])
        bound = self.work.bind((m_tiles, n_tiles, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(tiled_mma, tma_a, mA, tma_b, mB, c, sums, a_layout, b_layout, bound).launch(
            **self.launch_config(bound), stream=stream
        )


@jit_cache
def compile_joint_gemm(op: JointGemm):
    m = cute.sym_int()
    n, k = cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)
    bf16 = cutlass.BFloat16
    tiles = (cute.sym_int(), cute.sym_int(), cute.sym_int())
    sums = make_fake_compact_tensor(cutlass.Float32, (128, *tiles))
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (m, k)),
        make_fake_compact_tensor(bf16, (n, k)),
        make_fake_compact_tensor(bf16, (m, n)),
        sums,
        name=op.get_name(),
    )


def test_joint_gemm_describe():
    text = JointGemm().describe()
    assert "ab   Ring(3)  load -> cuda, mma" in text, text
    assert "{mma: Umma, cuda: Threads}" in text, text
    assert "empty     129 x3" in text, text


@pytest.mark.parametrize("m, n, k", [(128, 64, 64), (300, 136, 776)])
def test_joint_gemm_mma_and_cuda_threads_read_every_stage(requires_blackwell, m, n, k):
    """Both readers see every stage across ring wraparounds; neither sees a refill early."""
    torch.manual_seed(0)
    a = torch.randint(-3, 4, (m, k), device="cuda").to(torch.bfloat16)
    b = torch.randint(-3, 4, (n, k), device="cuda").to(torch.bfloat16)
    m_tiles, k_tiles = -(-m // TILE_M), -(-k // TILE_K)
    n_tiles = -(-n // 64)
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    sums = torch.full((128, k_tiles, m_tiles, n_tiles), float("nan"), device="cuda")
    op = JointGemm()
    compile_joint_gemm(op)(a, b, out, sums)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), (a.float() @ b.float().T).bfloat16().float())
    padded = torch.zeros(m_tiles * TILE_M, k_tiles * TILE_K, device="cuda")
    padded[:m, :k] = a.float()
    tiles = padded.view(m_tiles, TILE_M, k_tiles, TILE_K).sum(dim=(1, 3)).T  # (k, m)
    # Every n tile's CTA reads the same A tiles.
    torch.testing.assert_close(sums.sum(0), tiles[..., None].expand_as(sums[0]), rtol=0, atol=0)
