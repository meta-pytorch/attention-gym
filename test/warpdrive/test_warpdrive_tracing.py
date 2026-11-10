"""Test layer trace ranges and their exported PipelinePlan.

Compile-only tests check stable IKET names, absence of disabled IKET ops, and synthetic
profiler joins without run-iket. GPU tests decode globaltimer ranges from a small
persistent kernel.
"""

import json
from collections import Counter
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute

from attn_gym.testing.perfetto_trace import read_slices
from attn_gym.warpdrive import (
    ClusterLaunchControl,
    Consumer,
    CtaPair,
    Producer,
    ProtocolError,
    Ring,
    StaticPersistent,
    Threads,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
    tracing,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    compile_persistent_gemm,
)
from attn_gym.warpdrive.pipeline_plan import plan_from_warpdrive
from attn_gym.warpdrive.profiler import parse_iket
from attn_gym.warpdrive.profiler.pipeline import (
    analyze_pipeline,
    join,
    write_pipeline_perfetto,
)
from attn_gym.warpdrive.trace_session import trace_session


class _Recorder:
    """Record IKET range names while emitting the real instrumentation ops."""

    def __init__(self, real):
        self.real, self.pushed, self.payloads, self.pops = real, [], [], 0

    def range_push(self, name, *payload):
        self.pushed.append(name)
        self.payloads.append(bool(payload))
        return self.real.range_push(name, *payload) if self.real else None

    def range_pop(self):
        self.pops += 1
        return self.real.range_pop() if self.real else None


@pytest.fixture
def iket_recorder(monkeypatch, traced_compiles):
    rec = _Recorder(tracing.iket)
    monkeypatch.setattr(tracing, "iket", rec)
    compile_persistent_gemm.cache_clear()  # trace even if another test compiled the same key
    return rec


def _compile_gemm(op):
    compile_persistent_gemm(op)
    return op


def test_disabled_emits_no_iket_ops(monkeypatch, traced_compiles):
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError("IKET op emitted with IKET disabled")

    monkeypatch.setattr(tracing, "iket", Forbidden())
    compile_persistent_gemm.cache_clear()
    op = _compile_gemm(PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair()))
    assert op.trace_key() == "" and "iket" not in op.get_name()


def test_gemm_range_names_match_plan(iket_recorder):
    op = _compile_gemm(PersistentGemm(schedule=StaticPersistent()).enable_iket())
    plan = plan_from_warpdrive(op)
    names = set(iket_recorder.pushed)
    assert set(plan.roles) == set(plan.regions) == {"epilogue", "load", "mma"}
    assert set(plan.regions) <= names  # every plan region is an emitted range
    # The epilogue's NamedBarrier sync is a wait range too, though plans model channels only.
    waits = {"ab.acquire", "ab.wait", "acc.acquire", "acc.wait", "epi_sync.sync"}
    assert names == {"epilogue", "load", "mma", *waits}
    assert set(plan.resources) == {"ab", "acc"}
    # Every traced push has a pop: wait ranges, each item's range (closed by advance()), and the
    # last item's range a loop exit other than `valid` would leave open (closed at role exit).
    assert len(iket_recorder.pushed) == iket_recorder.pops
    deps = {(d.source, d.target, d.resource, d.kind, d.distance) for d in plan.dependencies}
    # ab is used inside the K loop (reuse distance 1); acc once per tile (2 stages -> 2).
    assert deps == {
        ("load", "mma", "ab", "data", 0),
        ("mma", "load", "ab", "reuse", 1),
        ("mma", "epilogue", "acc", "data", 0),
        ("epilogue", "mma", "acc", "reuse", 2),
    }
    assert op.get_name().endswith("_iket")


def _synthetic_capture(path, plan, iterations, warps_per_role):
    """An IKET JSON with one range per role, work item, and warp, as the layer names them."""
    strings, ranges, locations = [], [], []
    t = 1_000
    for it in range(iterations):
        for plan_role in plan.roles.values():
            for warp in range(plan_role.warp_start, plan_role.warp_end)[:warps_per_role]:
                locations.append({"ctaId": [0, 0, 0], "warpId": warp})
                if plan_role.name not in strings:
                    strings.append(plan_role.name)
                ranges.append(
                    {
                        "rangeNameIdx": strings.index(plan_role.name),
                        "warpLocIdxs": [len(locations) - 1],
                        "startTs": t,
                        "endTs": t + 900,
                    }
                )
            t += 100
        t += 1_000
    payload = {
        "launches": [{"kernelName": "synthetic", "ranges": ranges}],
        "stringTable": strings,
        "locationTable": locations,
    }
    path.write_text(json.dumps(payload))


def test_pair_plan_per_cta_and_roles_only(tmp_path, iket_recorder):
    """Per-CTA plans of a roles-only pair capture, and the leader's joins a synthetic capture
    end to end."""
    op = _compile_gemm(
        PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair()).enable_iket(waits=False)
    )
    names = set(iket_recorder.pushed)
    assert names == {"epilogue", "load", "mma", "work.scheduler"}  # waits=False: no waits
    leader, peer = plan_from_warpdrive(op, cta=0), plan_from_warpdrive(op, cta=1)
    assert set(leader.regions) == names
    assert set(peer.regions) == {"epilogue", "load"}  # leader-only roles are not in the peer
    assert (
        leader.roles["mma"].label == "mma (leader)" and peer.roles["load"].label == "load (peer)"
    )
    assert op.get_name().endswith("_iketroles")
    timeline = leader.schedule(iterations=3)
    trace = tmp_path / "capture.trace.json"
    _synthetic_capture(trace, leader, 3, warps_per_role=2)
    measured = join(parse_iket(trace), timeline)
    analysis = analyze_pipeline(timeline, measured)
    write_pipeline_perfetto(tmp_path / "pipeline.pftrace", timeline, measured, analysis=analysis)
    assert {r.role for r in measured.regions} == set(leader.roles)
    assert (tmp_path / "pipeline.pftrace").stat().st_size > 0


def test_fa4_pair_names_and_feedback_edges():
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import FlashAttentionForward

    plan = plan_from_warpdrive(
        FlashAttentionForward(schedule=ClusterLaunchControl(), cluster=CtaPair())
    )
    assert list(plan.roles) == [
        "softmax[0]",
        "softmax[1]",
        "correction",
        "mma",
        "epilogue",
        "load",
        "work.scheduler",
    ]
    feedback = {(d.source, d.target) for d in plan.dependencies if d.kind == "state"}
    assert ("softmax[0]", "mma") in feedback and ("softmax[1]", "mma") in feedback  # p_last
    reuse = {
        (d.source, d.target, d.resource): d.distance
        for d in plan.dependencies
        if d.kind == "reuse"
    }
    # stats: softmax[j] is bound to slot j, so it reuses one slot (distance 1, not 2).
    assert reuse[("correction", "softmax[0]", "stats")] == 1
    # The work ring's join completes with its last consumer in data order (epilogue); slots
    # are released on receipt, one item before the holding region ends (2 stages + 1).
    assert [k for k in reuse if k[2] == "work_ring"] == [
        ("epilogue", "work.scheduler", "work_ring")
    ]
    assert reuse[("epilogue", "work.scheduler", "work_ring")] == 3
    plan.schedule(iterations=4)  # no same-iteration cycle


# globaltimer backend

ITEMS, CTAS = 7, 3  # StaticPersistent: CTA c runs items c, c + 3, ...


class RingRoles(WarpSpecialized):
    """A producer and a two-warp consumer over a persistent work stream; the consumer marks a
    region with a payload."""

    def __init__(self):
        self.ring = Ring(2, Threads, Threads, payload=(Int32, (32,)))
        self.work = WorkStream(schedule=StaticPersistent())

    @role(warps=2)
    @cute.jit
    def prod(self, ctx, *, ring: Producer, work: WorkCursor):
        while work.valid:
            i, _, _ = work.item
            with ring.produce() as h:
                h.data[cute.arch.lane_idx()] = i
            work.advance()
        return ring, work

    @role(warps=range(2))
    @cute.jit
    def cons(self, ctx, *, ring: Consumer, work: WorkCursor):
        while work.valid:
            i, _, _ = work.item
            with ring.consume() as h, self.trace_region("store", payload=i * 10):
                if cute.arch.thread_idx()[0] == 0:
                    ctx.out[i] = h.data[0] + 1
            work.advance()
        return ring, work

    @cute.kernel
    def kernel(self, out: cute.Tensor, trace: cute.Tensor, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(out=out), trace_buffer=trace)

    @cute.jit
    def __call__(
        self, out: cute.Tensor, trace: cute.Tensor, workspace: cute.Tensor, stream: cuda.CUstream
    ):
        bound = self.work.bind((out.shape[0], 1, 1), workspace, CTAS)
        self.kernel.set_name_prefix("warpdrive_traced_ring")
        self.kernel(out, trace, bound).launch(**self.launch_config(bound), stream=stream)


class TracedRing(RingRoles):
    """``RingRoles`` plus a raw role without a cursor (one range over its body)."""

    @role(warps=3, raw=True)  # raw: a persistent stream does not require it to loop
    @cute.jit
    def idle(self, ctx):
        pass


@jit_cache
def compile_traced_ring(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(Int32, (cute.sym_int(),), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        name="warpdrive_traced_ring",
    )


def _run_traced(op, num_ctas, **kwargs):
    fn = compile_traced_ring(op)
    workspace, _ = op.work.launch_args("cuda")
    out = torch.zeros(ITEMS, device="cuda", dtype=torch.int32)
    with trace_session(op, num_ctas, **kwargs) as session:
        fn(out, session.prof.tensor, workspace)
    torch.testing.assert_close(out.cpu(), torch.arange(1, ITEMS + 1, dtype=torch.int32))
    return session


def test_globaltimer_ranges_decode_and_join_the_plan(requires_blackwell, tmp_path):
    """Decode ranges by CTA, role, and item, then join them to plan_from_warpdrive.

    Each role has one recording warp. Ring acquire/wait ranges supply measured waits.
    A disk-loaded kernel must decode using range names restored from its cached trace.
    """
    compile_traced_ring.cache_clear()
    compile_traced_ring(RingRoles().enable_trace("globaltimer"))
    hits = compile_traced_ring.cache_info().hits
    op = RingRoles().enable_trace("globaltimer")
    session = _run_traced(op, CTAS, trace_path=tmp_path / "ring.pftrace")
    assert compile_traced_ring.cache_info().hits == hits + 1
    ranges = session.ranges
    warps = {"cons": 0, "prod": 2}  # each role's first warp records, as its warp
    for cta in range(CTAS):
        items = list(range(cta, ITEMS, CTAS))
        by_name = {}
        for e in ranges:
            if e.cta == (cta,):
                by_name.setdefault(e.name, []).append(e)
        # Work-item ranges carry the item index; the region carries its own payload.
        assert sorted(e.payload for e in by_name["prod"]) == items
        assert sorted(e.payload for e in by_name["cons"]) == items
        assert sorted(e.payload for e in by_name["store"]) == [10 * i for i in items]
        waits = {"ring.acquire": "prod", "ring.wait": "cons"}
        for name, owner in waits.items():
            assert len(by_name[name]) == len(items)
            assert {e.warp for e in by_name[name]} == {warps[owner]}
        for name, owner in warps.items():
            assert {e.warp for e in by_name[name]} == {owner}
        # Nested ranges sit inside their item's range.
        for item in by_name["cons"]:
            inner = [e for e in by_name["store"] if e.payload == 10 * item.payload]
            assert len(inner) == 1
            assert item.start_ns <= inner[0].start_ns
            assert inner[0].end_ns <= item.end_ns
    tracks = {(item.process, item.track) for item in read_slices(tmp_path / "ring.pftrace")}
    assert tracks == {(f"CTA {cta}", name) for cta in range(CTAS) for name in warps}

    timeline = plan_from_warpdrive(op).schedule(iterations=len(range(0, ITEMS, CTAS)))
    measured = join(ranges, timeline, cta=(0,))
    assert {r.name for r in measured.regions} == {"prod", "cons"}
    assert all(r.wait_ns is not None and r.sample_count == 1 for r in measured.regions)
    analysis = analyze_pipeline(timeline, measured)
    assert {b.wait_source for b in analysis.blockers} == {"measured"}
    path = write_pipeline_perfetto(
        tmp_path / "pipeline.pftrace", timeline, measured, analysis=analysis
    )
    assert {s.track.split(" ·")[0] for s in read_slices(path)} == {"prod", "cons"}


def test_globaltimer_short_buffer_and_capacity(requires_blackwell):
    """Record only CTAs and ranges that fit the buffer; warn when a role exhausts its slots.

    Raw roles without cursors record one range over their whole body.
    """
    session = _run_traced(TracedRing().enable_trace("globaltimer", waits=False), CTAS - 1)
    assert {e.cta for e in session.ranges} == {(cta,) for cta in range(CTAS - 1)}
    assert not {e.name for e in session.ranges} & {"ring.acquire", "ring.wait"}
    idle = [(e.cta, e.warp, e.payload) for e in session.ranges if e.name == "idle"]
    assert sorted(idle) == [((cta,), 3, None) for cta in range(CTAS - 1)]
    op = TracedRing().enable_trace("globaltimer", events_per_role=2)
    with pytest.warns(UserWarning, match="filled their 2 records"):
        session = _run_traced(op, CTAS)
    assert max(Counter((e.cta, e.warp) for e in session.ranges).values()) == 2


class AdvanceInRegion(TracedRing):
    @role(warps=range(2))
    @cute.jit
    def cons(self, ctx, *, ring: Consumer, work: WorkCursor):
        while work.valid:
            with ring.consume(), self.trace_region("store"):
                work.advance()
        return ring, work


class RegionOutsideRoles(TracedRing):
    @cute.kernel
    def kernel(self, out: cute.Tensor, trace: cute.Tensor, bound):
        self.begin_roles(work=bound)
        with self.trace_region("setup"):
            pass
        self.run_roles(SimpleNamespace(out=out), trace_buffer=trace)


@pytest.mark.parametrize("backend", ["iket", "globaltimer"])
def test_ranges_must_nest(traced_compiles, backend):
    """Closing a work item's range while a region inside it is open is rejected at trace time
    (both backends pair a pop with the latest push)."""
    with pytest.raises(ProtocolError, match="ranges must nest"):
        compile_traced_ring.__wrapped__(AdvanceInRegion().enable_trace(backend))


def test_globaltimer_needs_a_buffer_and_a_role():
    class NoBuffer(TracedRing):
        @cute.kernel
        def kernel(self, out: cute.Tensor, trace: cute.Tensor, bound):
            self.begin_roles(work=bound)
            self.run_roles(SimpleNamespace(out=out))

    with pytest.raises(ProtocolError, match="trace_buffer"):
        compile_traced_ring.__wrapped__(NoBuffer().enable_trace("globaltimer"))
    with pytest.raises(ProtocolError, match="outside a role"):
        compile_traced_ring.__wrapped__(RegionOutsideRoles().enable_trace("globaltimer"))
    compile_traced_ring.__wrapped__(RegionOutsideRoles().enable_iket())  # IKET may: no error
    op = TracedRing().enable_trace("globaltimer")
    with pytest.raises(ProtocolError, match="compiled kernel"):
        trace_session(op, CTAS)
    with pytest.raises(ValueError, match="trace backend"):
        op.enable_trace("nvtx")
    assert op.trace_key() == "_gtimer"
    assert op.enable_trace("globaltimer", waits=False).trace_key() == "_gtimerroles"


def test_iket_region_payload(iket_recorder):
    """With IKET, a trace_region payload becomes the range's payload; layer ranges carry none."""
    compile_traced_ring.cache_clear()
    compile_traced_ring.__wrapped__(TracedRing().enable_iket())
    pushed = dict(zip(iket_recorder.pushed, iket_recorder.payloads))
    assert pushed == {
        "prod": False,
        "cons": False,
        "idle": False,
        "ring.acquire": False,
        "ring.wait": False,
        "store": True,
    }
