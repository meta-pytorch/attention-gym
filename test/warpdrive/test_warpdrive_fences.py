"""Fence rules (``fences.py``): the derived defaults for every channel configuration, the table
itself, and the trace-time check of explicit fences against annotated accesses."""

import itertools
import warnings
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.warpdrive import (
    Consumer,
    Instance,
    Producer,
    ProtocolError,
    Ring,
    Slots,
    SmemPlan,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WarpSpecialized,
    fence_async_shared,
    role,
    unchecked,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.fences import RULES, Access, required

AGENT_ACCESSES = [Access(a, w) for a in ("threads", "tma", "umma") for w in (False, True)]
THREAD_MECHS = (Threads, Warps)


# Host-only: the table and every channel's derived fences


# Hand-reviewed: (storage, the signalling side's access, the other side's next access) -> the
# fence the signal needs. Every other combination needs none.
NEEDS = {
    ("smem", Access("threads", True), Access("umma", False)): "smem_async",
    ("smem", Access("threads", True), Access("tma", False)): "smem_async",
    ("smem", Access("threads", False), Access("tma", True)): "smem_async",
    ("smem", Access("threads", True), Access("tma", True)): "smem_async",
    ("tmem", Access("threads", True), Access("threads", False)): "tmem_store",
    ("tmem", Access("threads", True), Access("threads", True)): "tmem_store",
    ("tmem", Access("threads", True), Access("umma", False)): "tmem_store",
    ("tmem", Access("threads", True), Access("umma", True)): "tmem_store",
    ("tmem", Access("threads", False), Access("threads", True)): "tmem_load",
    ("tmem", Access("threads", False), Access("umma", True)): "tmem_load",
}


def test_rule_table_matches_the_reviewed_pairs():
    for storage, mine, other in itertools.product(
        ("smem", "tmem"), AGENT_ACCESSES, AGENT_ACCESSES
    ):
        expected = NEEDS.get((storage, mine, other))
        got = required(storage, [mine], [other])
        assert got == ((expected,) if expected else ()), (storage, mine, other)


def test_every_rule_cites_a_reason():
    assert all(rule.reason and rule.citation for rule in RULES)


def _storage(kind):
    if kind == "payload":
        return {"payload": (cutlass.Int32, (32,))}
    if kind == "buffer":
        return {"buffer": SmemPlan().buffer("X", cutlass.Int32, align=128)}
    return {}


def _external(side):
    if side is None:
        return {}
    return {
        "external": side,
        "override": {"full_counts" if side == "producer" else "empty_counts": 1},
    }


class CustomTma(Tma):
    pass


PRODUCERS = [Tma(128), Tma(128, multicast="n"), CustomTma(128), Umma, Threads, Warps]
CONSUMERS = [
    Umma,
    Threads,
    Warps,
    {"mma": Umma, "cuda": Threads},
    {"mma": Umma, "cuda": Warps},
    {"a": Threads, "b": Warps},
]


def _expected(producer, consumer, storage, external):
    """The configuration view of the table, stated by hand."""
    consumers = consumer.values() if isinstance(consumer, dict) else [consumer]
    tma = isinstance(producer, Tma)
    release = tma and any(m in THREAD_MECHS for m in consumers) and external != "consumer"
    commit = (
        producer in THREAD_MECHS
        and Umma in consumers
        and storage is not None
        and external != "producer"
    )
    return ("smem_async",) if commit else (), ("smem_async",) if release else ()


def test_derived_fences_for_every_channel_configuration():
    """Accepted mechanisms, storage and external sides preserve defaults and replacements."""
    built = 0
    for producer, consumer, storage, external in itertools.product(
        PRODUCERS, CONSUMERS, (None, "payload", "buffer"), (None, "producer", "consumer")
    ):
        try:
            ring = Ring(2, producer, consumer, **_storage(storage), **_external(external))
        except ValueError:
            continue
        built += 1
        want = _expected(producer, consumer, storage, external)
        assert (ring.derived_commit_fence, ring.derived_release_fence) == want, ring
        assert (ring.commit_fence, ring.release_fence) == want
        if not isinstance(producer, Tma) and external is None:
            replaced = Ring(2, producer, consumer, **_storage(storage), commit_fence=())
            assert replaced.derived_commit_fence == want[0] and replaced.commit_fence == ()
    assert built > 150
    named = Ring(2, Threads, Threads, sync="named", payload=(cutlass.Int32, (32,)))
    assert (named.commit_fence, named.release_fence) == ((), ())


# Trace-time check of explicit fences


@jit_cache
def compile_toy(op, name: str):
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(op, unused, cutlass.Int32(1), name=name)


def trace(op, name: str) -> list[str]:
    """Compile ``op`` uncached, returning the fence-check warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_toy.__wrapped__(op, name)
    return [str(w.message) for w in caught if "fence check" in str(w.message)]


def verdict(op, chan: str, key: str) -> str | None:
    return op.checks().fences.get((chan, key))


class Model(WarpSpecialized):
    def layouts(self):
        return {"X": cute.make_layout((32, 2))}

    @cute.kernel
    def kernel(self, n: cutlass.Int32):
        self.begin_roles(layouts=self.layouts())
        self.run_roles(SimpleNamespace(n=n))

    @cute.jit
    def __call__(self, unused: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("warpdrive_fence_model")
        self.kernel(n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


class RefilledRing(Model):
    """A TMA ring whose consumers read the stage with generic loads: the next TMA fill of that
    stage is an async-proxy write after those reads. The alias diagnostic sees no overlap of
    two names here, so only the fence check can catch a wrong ``release_fence=()``."""

    def __init__(
        self, release_fence, mode: str = "annotated", alias_check: str = "warn", renamed=False
    ):
        self.mode = mode
        self.alias_check = alias_check
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        self.ring = Ring(
            2, Tma(128), Warps, buffer={"tile": x} if renamed else x, release_fence=release_fence
        )

    @role(warps=0)
    @cute.jit
    def load(self, ctx, *, ring: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.produce():
                pass
        return ring

    @role(warps=1)
    @cute.jit
    def compute(self, ctx, *, ring: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            if cutlass.const_expr(self.mode == "span"):
                h = ring.span()
                for i in cutlass.range(2):
                    h.begin(i)
                    h.access(read=True)
                    h.release(i)
            elif cutlass.const_expr(self.mode == "raw_release"):
                h = ring.wait()
                h.access(read=True)
                h.release(signal=False)
                cute.arch.mbarrier_arrive(h.empty_barrier)
            elif cutlass.const_expr(self.mode == "unclosed"):
                with unchecked("unclosed", reason="test: released by the arrival below"):
                    h = ring.wait()
                h.access(read=True)
                cute.arch.mbarrier_arrive(h.empty_barrier)
            else:
                with ring.consume() as h:
                    if cutlass.const_expr(self.mode == "unannotated"):
                        h.data[cute.arch.lane_idx()] += 1
                    elif cutlass.const_expr(self.mode == "opted_out"):
                        with unchecked("fence", reason="test: fenced by a raw protocol"):
                            h.access(read=True)
                    elif cutlass.const_expr(self.mode == "fenced_in_branch"):
                        if ctx.n > 1:
                            h.access(read=True)
                            fence_async_shared()
                    else:
                        h.access(read=True)
                    if cutlass.const_expr(self.mode == "hand_fenced"):
                        fence_async_shared()
                    elif cutlass.const_expr(self.mode == "conditional_fence"):
                        if ctx.n > 1:
                            fence_async_shared()
                    elif cutlass.const_expr(self.mode == "loop_fence"):
                        for _ in cutlass.range(ctx.n, unroll=1):
                            fence_async_shared()
        return ring


@pytest.mark.parametrize("renamed", [False, True])
def test_release_fence_opt_out_before_a_tma_refill_is_a_missing_fence(renamed):
    op = RefilledRing((), renamed=renamed)
    warned = trace(op, "fence_refill_missing")
    assert verdict(op, "ring", "release_fence") == "missing fence"
    assert len(warned) == 1 and "ring release_fence" in warned[0], warned
    text = op.describe()
    assert "ring release_fence none  missing fence" in text, text
    assert "missing smem_async: compute read X by threads" in text, text
    assert "tma write" in text, text


def test_strict_mode_raises_on_a_missing_fence():
    with pytest.raises(ProtocolError, match="fence check"):
        trace(RefilledRing((), alias_check="strict"), "fence_refill_strict")


@pytest.mark.parametrize(
    "fence, mode, expected",
    [
        (None, "annotated", None),  # derived: nothing to check
        ("smem_async", "annotated", "covered"),
        ((), "hand_fenced", "covered"),
        ((), "unannotated", "unverified"),
        ((), "opted_out", "unverified"),
        ((), "raw_release", "unverified"),
        ((), "unclosed", "unverified"),
        ((), "conditional_fence", "unverified"),
        ((), "loop_fence", "unverified"),
        ("smem_async", "span", "covered"),
        ((), "fenced_in_branch", "covered"),
    ],
    ids=[
        "derived",
        "explicit",
        "hand_fenced",
        "unannotated",
        "opted_out",
        "raw_release",
        "unclosed",
        "conditional_fence",
        "loop_fence",
        "span_own_fence",
        "fenced_in_branch",
    ],
)
def test_refilled_ring_verdicts(fence, mode, expected):
    op = RefilledRing(fence, mode)
    assert trace(op, f"fence_refill_{mode}_{len(fence or ())}_{int(fence is None)}") == []
    assert verdict(op, "ring", "release_fence") == expected
    if mode == "opted_out":
        assert "unchecked fence @" in op.describe()


def test_alias_check_off_skips_the_fence_check():
    op = RefilledRing((), alias_check="off")
    assert trace(op, "fence_refill_off") == []
    assert op.checks().fences == {}


class StoresForAsyncReader(Model):
    """Threads write the stage for an async reader: an MMA consumer (implied by its
    mechanism) or a thread role that TMA-stores the stage (declared with ``by="tma"``)."""

    def __init__(self, consumer, commit_fence, renamed=False, annotation="read"):
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        self.ring = Ring(
            2, Threads, consumer, buffer={"tile": x} if renamed else x, commit_fence=commit_fence
        )
        self.tma_store = consumer is Threads
        self.annotation = annotation

    @role(warps=0)
    @cute.jit
    def write(self, ctx, *, ring: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.produce() as h:
                h.access(write=True)
        return ring

    @role(warps=1)
    @cute.jit
    def read(self, ctx, *, ring: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.consume() as h:
                if cutlass.const_expr(self.tma_store):
                    if cutlass.const_expr(self.annotation == "read"):
                        h.access(read=True, by="tma")
                    elif cutlass.const_expr(self.annotation == "raw"):
                        h.view()
        return ring


@pytest.mark.parametrize(
    "consumer, fence, expected",
    [
        (Umma, (), "missing fence"),
        (Threads, (), "missing fence"),
        (Threads, "smem_async", "covered"),
    ],
    ids=["umma_opt_out", "tma_store_reader", "tma_store_reader_fenced"],
)
@pytest.mark.parametrize("renamed", [False, True])
def test_commit_fence_against_an_async_reader(consumer, fence, expected, renamed):
    op = StoresForAsyncReader(consumer, fence, renamed=renamed)
    trace(op, f"fence_commit_{consumer.__name__.lower()}_{len(fence)}")
    assert verdict(op, "ring", "commit_fence") == expected, op.describe()


class StoresOverAccumulator(Model):
    """Threads load S and store P over it, then release to the MMA that reads P (FA4's S
    ring): the release needs ``tmem_store``; the load's own wait is not visible."""

    def __init__(self, release_fence):
        self.ring = Ring(1, Umma, Threads, release_fence=release_fence)
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    def layouts(self):
        return {}

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, ring: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.produce():
                tmem.access("P", read=True, by="umma")
                tmem.access("S", write=True, by="umma")
        return ring

    @role(warps=range(1, 5))
    @cute.jit
    def softmax(self, ctx, *, ring: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.consume():
                tmem.access("S", read=True)
                tmem.access("P", write=True)
        return ring


@pytest.mark.parametrize(
    "fence, expected, text",
    [
        ("tmem_load", "unverified", "unverified: tmem_store: softmax write P[0] by threads"),
        ("tmem_store", "unverified", "unverified: tmem_load: softmax read S[0] by threads"),
        (("tmem_store", "tmem_load"), "covered", None),
    ],
    ids=["no_store_wait", "no_load_wait", "both"],
)
def test_tmem_release_fence_against_annotated_accesses(fence, expected, text):
    op = StoresOverAccumulator(fence)
    trace(op, f"fence_tmem_{'_'.join((fence,) if isinstance(fence, str) else fence)}")
    assert verdict(op, "ring", "release_fence") == expected, op.describe()
    if text:
        assert text in op.describe()


class SlotBoundReaders(Model):
    """Threads fill two slots. The one MMA consumer is bound to slot 0; thread consumers read
    both. Only slot 0 has an async reader, so only its commit needs ``smem_async``."""

    def __init__(self, fence_slot_0: bool):
        self.fence_slot_0 = fence_slot_0
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        self.ds = Slots(2, Threads, {"mma": Umma, "cuda": Threads}, buffer=x, commit_fence=())

    @role(warps=0)
    @cute.jit
    def write(self, ctx, *, ds: Producer):
        with ds.produce(0) as h:
            h.access(write=True)
            if cutlass.const_expr(self.fence_slot_0):
                fence_async_shared()
        with ds.produce(1) as h:
            h.access(write=True)
        return ds

    @role(warps=1)
    @cute.jit
    def mma(self, ctx, *, slot: Instance, ds: Consumer[Instance]):
        with ds.consume(slot) as h:
            h.access(read=True, by="umma")
        return ds

    @role(warps=[range(2, 3), range(3, 4)])
    @cute.jit
    def cuda(self, ctx, *, slot: Instance, ds: Consumer[Instance]):
        with ds.consume(slot) as h:
            h.access(read=True)
        return ds


@pytest.mark.parametrize("fence_slot_0, expected", [(True, "covered"), (False, "missing fence")])
def test_implied_async_reader_applies_only_to_its_bound_slot(fence_slot_0, expected):
    op = SlotBoundReaders(fence_slot_0)
    trace(op, f"fence_slot_bound_{int(fence_slot_0)}")
    assert verdict(op, "ds", "commit_fence") == expected, op.describe()


@pytest.mark.parametrize("annotation", ["raw", "missing"])
def test_missing_opposite_smem_annotation_is_unverified(annotation):
    op = StoresForAsyncReader(Threads, (), annotation=annotation)
    assert trace(op, f"fence_opposite_{annotation}") == []
    assert verdict(op, "ring", "commit_fence") == "unverified", op.describe()


class TmemFootprints(Model):
    def __init__(self, separation):
        self.separation = separation
        self.ring = Ring(1, Umma, Threads, release_fence=())
        self.tmem_plan = TmemPlan().region("X", 64, 2)
        self.alias_check = "strict" if separation != "overlap" else "warn"

    def layouts(self):
        return {}

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, ring: Producer, tmem: Tmem):
        with ring.produce():
            if cutlass.const_expr(self.separation == "raw"):
                tmem.ptr("X")
            elif cutlass.const_expr(self.separation != "missing"):
                tmem.access("X", read=True, by="umma", cols=range(32), lanes=range(32))
        return ring

    @role(warps=range(4, 6))
    @cute.jit
    def store(self, ctx, *, ring: Consumer, tmem: Tmem):
        with ring.consume():
            tmem.access(
                "X",
                1 if self.separation == "slot" else 0,
                write=True,
                cols=range(32, 64) if self.separation == "cols" else range(32),
                lanes=range(32, 64) if self.separation == "lanes" else None,
                warps=(5,) if self.separation == "warp" else (4,),
            )
        return ring


@pytest.mark.parametrize(
    "separation", ["cols", "lanes", "slot", "warp", "overlap", "raw", "missing"]
)
def test_tmem_fences_respect_footprints(separation):
    op = TmemFootprints(separation)
    warned = trace(op, f"fence_footprints_{separation}")
    expected = "unverified" if separation in ("overlap", "raw", "missing") else "covered"
    assert verdict(op, "ring", "release_fence") == expected, op.describe()
    # Only overlapping footprints need the store wait; a manual one is invisible to the trace.
    assert ("unverified: tmem_store" in op.describe()) == (separation == "overlap")
    assert warned == []


@pytest.mark.parametrize(
    "producer, consumer", [(Threads, Umma), (Umma, Threads), (Tma(128), Warps)]
)
def test_explicit_overrides_always_replace_defaults(producer, consumer):
    overrides = {"release_fence": ("tmem_load", "smem_async")}
    if not isinstance(producer, Tma):
        overrides["commit_fence"] = ("tmem_store", "smem_async")
    ring = Ring(2, producer, consumer, **_storage("buffer"), **overrides)
    for key, value in overrides.items():
        assert getattr(ring, key) == value
    empty = Ring(2, producer, consumer, **_storage("buffer"), **dict.fromkeys(overrides, ()))
    for key in overrides:
        assert getattr(empty, key) == ()


class ExternalConsumer(Model):
    def __init__(self, consumer, fence):
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        self.ring = Ring(
            2,
            Threads,
            consumer,
            buffer=x,
            external="consumer",
            override={"empty_counts": 1},
            commit_fence=fence,
        )

    @role(warps=0)
    @cute.jit
    def write(self, ctx, *, ring: Producer):
        with ring.produce() as h:
            h.access(write=True)
        return ring


@pytest.mark.parametrize("consumer", [Umma, Threads])
@pytest.mark.parametrize("fence", [(), "smem_async"])
def test_external_counterpart_is_unverified(consumer, fence):
    op = ExternalConsumer(consumer, fence)
    assert trace(op, f"fence_external_{consumer.__name__.lower()}_{len(fence)}") == []
    assert verdict(op, "ring", "commit_fence") == "unverified"


class OtherChannelFence(Model):
    def __init__(self, placement):
        self.placement = placement
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        self.ring = Ring(2, Tma(128), Warps, buffer=x, release_fence=())
        self.other = Ring(2, Tma(128), Warps)

    @role(warps=0)
    @cute.jit
    def load(self, ctx, *, ring: Producer, other: Producer):
        with ring.produce():
            pass
        with other.produce():
            pass
        return ring, other

    @role(warps=1)
    @cute.jit
    def compute(self, ctx, *, ring: Consumer, other: Consumer):
        with ring.consume() as h:
            if cutlass.const_expr(self.placement == "before"):
                with other.consume():
                    pass
            h.access(read=True)
            if cutlass.const_expr(self.placement == "after"):
                with other.consume():
                    pass
            elif cutlass.const_expr(self.placement == "conditional"):
                if ctx.n > 1:
                    with other.consume():
                        pass
        return ring, other


@pytest.mark.parametrize(
    "placement, expected",
    [("before", "missing fence"), ("after", "covered"), ("conditional", "unverified")],
)
def test_other_channel_fences_obey_access_order_and_coverage(placement, expected):
    op = OtherChannelFence(placement)
    warned = trace(op, f"fence_other_{placement}")
    assert verdict(op, "ring", "release_fence") == expected, op.describe()
    assert bool(warned) == (expected == "missing fence")


class SharedTmemHandoff(Model):
    def __init__(self, first, fence):
        self.first = first
        self.a = Ring(1, Umma, Threads, release_fence=fence)
        self.b = Ring(1, Umma, Threads, release_fence=())
        self.tmem_plan = TmemPlan().region("P", 64)

    def layouts(self):
        return {}

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, a: Producer, b: Producer, tmem: Tmem):
        ha = a.acquire()
        hb = b.acquire()
        tmem.access("P", read=True, by="umma")
        hb.commit()
        ha.commit()
        return a, b

    @role(warps=range(1, 5))
    @cute.jit
    def store(self, ctx, *, a: Consumer, b: Consumer, tmem: Tmem):
        ha = a.wait()
        hb = b.wait()
        tmem.access("P", write=True)
        if cutlass.const_expr(self.first == "a"):
            ha.release()
            hb.release()
        else:
            hb.release()
            ha.release()
        return a, b


@pytest.mark.parametrize(
    "first, fence, expected",
    [
        ("a", "tmem_store", "covered"),
        ("b", "tmem_store", "unverified"),
        ("b", (), "unverified"),
    ],
)
def test_tmem_access_shared_by_two_handoffs(first, fence, expected):
    op = SharedTmemHandoff(first, fence)
    trace(op, f"fence_shared_handoff_{first}_{len(fence)}")
    assert verdict(op, "b", "release_fence") == expected, op.describe()
    if not fence:  # no fence at all: the requirement itself is reported, not only the ambiguity
        assert "unverified: tmem_store" in op.describe()
        assert "another open channel" not in op.describe()


class PayloadRing(RefilledRing):
    def __init__(self):
        self.mode = "annotated"
        self.ring = Ring(2, Tma(128), Warps, payload=(cutlass.Int32, (32,)), release_fence=())

    def layouts(self):
        return {}


def test_payload_annotation_reports_the_recording_gap():
    op = PayloadRing()
    assert trace(op, "fence_payload") == []
    assert verdict(op, "ring", "release_fence") == "unverified"
    assert "payload accesses are not recorded" in op.describe()


def test_fence_warning_and_verdict_restore_from_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    cold, warm = RefilledRing(()), RefilledRing(())
    recorded = []
    for op in (cold, warm):
        compile_toy.cache_clear()
        with pytest.warns(UserWarning, match="fence check") as caught:
            compile_toy(op, "fence_cache_restore")
        recorded.append([str(w.message) for w in caught if "fence check" in str(w.message)])
    info = compile_toy.cache_info()
    assert (info.hits, info.misses, info.currsize) == (1, 0, 1)
    assert recorded[0] == recorded[1]
    assert cold.checks() == warm.checks()
    assert verdict(warm, "ring", "release_fence") == "missing fence"


class RetiredProducer(StoresForAsyncReader):
    @role(warps=0)
    @cute.jit
    def write(self, ctx, *, ring: Producer):
        h = ring.acquire()
        h.access(write=True)
        h.retire()
        return ring


def test_retired_handle_publishes_no_access():
    op = RetiredProducer(Umma, ())
    assert trace(op, "fence_retire") == []
    assert verdict(op, "ring", "commit_fence") == "covered"


class SharedStage(Model):
    def __init__(self, release_fence, mode="annotated", threaded=False):
        self.mode = mode
        self.threaded = threaded
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        y = self.smem_plan.buffer("Y", cutlass.Int32, align=128)
        self.ring = Ring(
            2,
            Threads if threaded else Tma(256),
            {"reader": Warps, "peeker": Warps},
            buffer={"x": x, "y": y},
            release_fence=release_fence,
            commit_fence=() if threaded else None,
        )

    def layouts(self):
        return {"X": cute.make_layout((32, 2)), "Y": cute.make_layout((32, 2))}

    @role(warps=0)
    @cute.jit
    def load(self, ctx, *, ring: Producer):
        with ring.produce() as h:
            if cutlass.const_expr(self.threaded):
                h.access("x", write=True)
                h.access("y", write=True)
        return ring

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def reader(self, ctx, *, ring: Consumer):
        with ring.consume() as h:
            h.access("x", read=True)
            if cutlass.const_expr(self.mode == "raw_buffer"):
                h.view("y")
            else:
                h.access("y", read=True)
        return ring

    @role(warps=3)
    @cute.jit
    def peeker(self, ctx, *, ring: Consumer):
        with ring.consume() as h:
            if cutlass.const_expr(self.mode == "raw_role"):
                h.view("x")
            else:
                h.access("x", read=True)
                h.access("y", read=True)
        return ring


@pytest.mark.parametrize(
    "fence, mode, threaded, key, expected",
    [
        ((), "annotated", False, "release_fence", "missing fence"),
        ("smem_async", "annotated", False, "release_fence", "covered"),
        ("smem_async", "raw_buffer", False, "release_fence", "unverified"),
        ("smem_async", "raw_role", False, "release_fence", "unverified"),
        (None, "annotated", True, "commit_fence", "covered"),
    ],
)
def test_per_role_fences_check_each_instance_and_storage(fence, mode, threaded, key, expected):
    op = SharedStage(fence, mode, threaded)
    trace(op, f"fence_shared_stage_{mode}_{len(fence or ())}_{int(threaded)}")
    assert verdict(op, "ring", key) == expected, op.describe()
