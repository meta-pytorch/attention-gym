"""Compile small single-CTA kernels to test alias-ordering verdicts; never launch them.

FA4 mutants are in test_warpdrive_alias.py.
"""

import re
import warnings
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.testing.warpdrive import alias_section, status
from attn_gym.warpdrive import (
    Consumer,
    Producer,
    Ring,
    Sharing,
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
    named_barrier,
    role,
)
from attn_gym.warpdrive.alias import RoleEvents, completion
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.protocol import Event


@jit_cache
def compile_toy(op, name: str):
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(op, unused, cutlass.Int32(1), name=name)


def compile_quietly(op, name: str):
    """Compile ``op`` (entry point ``(unused, n)``), returning the alias warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn = compile_toy(op, name)
    return fn, [str(w.message) for w in caught if "alias diagnostic" in str(w.message)]


def record_arch_calls(monkeypatch, labels: dict[str, str]) -> list[str]:
    """Record ``labels[name]`` each time tracing calls ``cute.arch.<name>``, outside the TMEM
    teardown (its drain waits are not channel fences)."""
    emitted = []
    for name, label in labels.items():
        original = getattr(cute.arch, name)

        def record(*args, _label=label, _original=original, **kwargs):
            emitted.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cute.arch, name, record)
    teardown = WarpSpecialized._tmem_teardown

    def quiet_teardown(self, unit, *args):
        start = len(emitted)
        teardown(self, unit, *args)
        del emitted[start:]

    monkeypatch.setattr(WarpSpecialized, "_tmem_teardown", quiet_teardown)
    return emitted


class Model(WarpSpecialized):
    """Roles loop ``n`` times; ``layouts()`` gives the SMEM plan's layouts, if any. The models
    are access orderings, not complete protocols, so their epochs need not balance."""

    liveness_check = "off"

    def layouts(self):
        return {}

    @cute.kernel
    def kernel(self, n: cutlass.Int32):
        self.begin_roles(layouts=self.layouts())
        self.run_roles(SimpleNamespace(n=n))

    @cute.jit
    def __call__(self, unused: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("warpdrive_alias_model")
        self.kernel(n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


# FA4-backward-style cross-warp case: eight warps share an S region's lanes in two warpgroups;
# each warpgroup overwrites its half of the P columns (an alias inside S) while the other
# warpgroup's warp on the same lanes may still be loading S.


class CrossWarpPOverS(Model):
    def __init__(self, barrier: bool):
        self.barrier = barrier
        self.s = Slots(1, Umma, Threads, release_fence="tmem_store")
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=8)
    @cute.jit
    def mma(self, ctx, *, s: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with s.produce(0):
                tmem.view("S", write=True, by="umma")  # the pointer an S = Q K^T MMA writes
        return s

    @role(warps=range(8))
    @cute.jit
    def softmax(self, ctx, *, s: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with s.consume(0):
                tmem.access("S", read=True)  # every warp loads S on its 32 lanes
                if cutlass.const_expr(self.barrier):
                    named_barrier(3, 256)
                # Warpgroup g writes P columns [32 g, 32 g + 32): the other warpgroup's warp on
                # the same lanes reads those S columns.
                tmem.access(
                    "P", write=True, cols=lambda w: range(32 * (w // 4), 32 * (w // 4) + 32)
                )
        return s


def test_cross_warp_p_over_s_needs_a_barrier():
    op = CrossWarpPOverS(barrier=False)
    _, alias = compile_quietly(op, "cross_warp_0")
    assert len(alias) == 1 and "P~S" in alias[0]
    text = alias_section(op)
    assert status(op, "P ~ S") == "unverified"
    found = re.search(
        r"softmax read S\[0\] by threads @ .*and softmax write P\[0\] by threads @ .*: warps "
        r"(\d) and (\d) touch the same TMEM lanes and columns in one handle with no recorded "
        r"named_barrier",
        text,
    )
    assert found and abs(int(found[1]) - int(found[2])) == 4, text  # same lanes, other group

    fixed = CrossWarpPOverS(barrier=True)
    _, alias = compile_quietly(fixed, "cross_warp_1")
    assert alias == []
    assert status(fixed, "P ~ S") == "no hazard found"


def test_diagnostic_emits_no_device_code():
    """The same kernel lowers to identical IR with the diagnostic on and off."""
    ir = []
    for mode in ("warn", "off"):
        op = CrossWarpPOverS(barrier=True)
        op.alias_check = mode
        fn = compile_toy.__wrapped__(op, "cross_warp_1")  # the compiled IR, not a cached artifact
        ir.append(str(fn.ir_module))
    assert ir[0] == ir[1]


# A Ring hands storage over stage by stage: with two stages the MMA may rewrite one S region
# while the epilogue still reads the previous epoch's P from it.


class RingTmemHandoff(Model):
    """The MMA writes S on ``acc``; the epilogue reads P (in S) and releases, waiting for its
    loads first (``tmem_load``) unless ``waited=False``."""

    def __init__(self, stages: int, waited: bool = True):
        self.acc = Ring(stages, Umma, Threads, release_fence="tmem_load" if waited else ())
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=4)
    @cute.jit
    def mma(self, ctx, *, acc: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with acc.produce():
                tmem.access("S", write=True, by="umma")
        return acc

    @role(warps=range(4))
    @cute.jit
    def epilogue(self, ctx, *, acc: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with acc.consume():
                tmem.access("P", read=True)
        return acc


class RingTmemHandoffBesideControl(RingTmemHandoff):
    """``RingTmemHandoff(1, waited=False)`` with a second one-stage ring held around both
    accesses. Its thread arrivals cannot complete the MMA write, so that handoff fails with a
    hazard; ``acc`` alone still orders the pair up to the unrecorded load wait."""

    def __init__(self):
        super().__init__(1, waited=False)
        self.ctl = Ring(1, Threads, Threads)

    @role(warps=4)
    @cute.jit
    def mma(self, ctx, *, acc: Producer, ctl: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ctl.produce(), acc.produce():
                tmem.access("S", write=True, by="umma")
        return acc, ctl

    @role(warps=range(4))
    @cute.jit
    def epilogue(self, ctx, *, acc: Consumer, ctl: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ctl.consume(), acc.consume():
                tmem.access("P", read=True)
        return acc, ctl


class UmmaReadReleasedByThreads(Model):
    """``prod`` writes TMEM S without waiting for its stores and hands it over on ``s``; the
    consumer's MMA reads P (in S) and releases ``s`` with a thread arrival, which does not
    complete the MMA read before ``prod`` rewrites S."""

    def __init__(self):
        self.s = Slots(1, Threads, Threads)
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, s: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with s.produce(0):
                tmem.access("S", write=True)
        return s

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, s: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with s.consume(0):
                tmem.access("P", read=True, by="umma")
        return s


# One role, no synchronization between iterations: an MMA's D write to S in iteration i and the
# A read of P in iteration i + 1 are not a pipelined pair (the trace order alone, A read then
# D write, is).


class LoopCarriedMma(Model):
    def __init__(self):
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            tmem.access("P", read=True, by="umma")  # PV
            tmem.access("S", write=True, by="umma")  # next QK


# Unknown epoch correspondence: the MMA peels its last tile out of the loop (as FA4 backward
# does), so its traced per-item channel counts (two unrolled iterations plus one) differ from
# the softmax's (two). The round trip through ``s`` orders the loop-carried S write -> P read
# pair in reality, but the trace cannot match the epochs: unverified, not a potential hazard.
# A pair no epoch correspondence could order (S written before P is read in one handle) stays
# a potential hazard.


class PeeledRoundTrip(Model):
    def __init__(self, peel: bool, swap: bool = False):
        self.peel, self.swap = peel, swap
        self.s = Slots(1, Umma, Threads, release_fence="tmem_store")
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @cute.jit
    def tile(self, s, tmem):
        with s.produce(0):
            if cutlass.const_expr(self.swap):
                tmem.access("S", write=True, by="umma")  # QK overwrites P before PV reads it
                tmem.access("P", read=True, by="umma")
            else:
                tmem.access("P", read=True, by="umma")  # PV of the previous tile
                tmem.access("S", write=True, by="umma")  # QK of this tile
        return s

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, s: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n - 1 if self.peel else ctx.n, unroll=1):
            s = self.tile(s, tmem)
        if cutlass.const_expr(self.peel):
            s = self.tile(s, tmem)
        return s

    @role(warps=1)
    @cute.jit
    def softmax(self, ctx, *, s: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with s.consume(0):
                tmem.access("S", read=True)
                tmem.access("P", write=True)
        return s


@pytest.mark.parametrize(
    "peel, swap, expected",
    [
        (False, False, "no hazard found"),
        (True, False, "unverified"),
        (True, True, "potential hazard"),
    ],
    ids=["balanced", "peeled", "peeled_real_hazard"],
)
def test_unknown_epoch_correspondence_is_unverified_not_a_hazard(peel, swap, expected):
    op = PeeledRoundTrip(peel, swap)
    _, alias = compile_quietly(op, f"peeled_{int(peel)}{int(swap)}")
    text = alias_section(op)
    assert status(op, "P ~ S") == expected, text
    if expected == "unverified":
        assert alias == []  # a modeling gap, not a finding
        assert re.search(
            r"unverified: mma write S\[0\] by umma @ .* before mma read P\[0\] by umma @ .*: "
            r"ordered only through channel epochs the trace does not match \(s\[0\]: mma "
            r"commits 3 vs softmax waits 2 per traced work item.*unknown epoch correspondence",
            text,
        ), text
    elif expected == "potential hazard":
        assert len(alias) == 1 and "not a pipelined pair" in text


# A TMA load is published only by its own handle's transaction count: a reader that waits on
# another TMA channel the loader fills is not ordered after it.


class TmaThroughOtherChannel(Model):
    def __init__(self):
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        y = self.smem_plan.buffer("Y", cutlass.Int32, overlaps=x)
        self.a = Slots(1, Tma(128), Threads, buffer=x)
        self.b = Slots(1, Tma(128), Threads, buffer=y)  # another TMA transaction count

    def layouts(self):
        return {"X": cute.make_layout((32, 1)), "Y": cute.make_layout((32, 1))}

    @role(warps=0)
    @cute.jit
    def load(self, ctx, *, a: Producer, b: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.produce(0) as h:
                h.access(write=True, by="tma")
            with b.produce(0):
                pass
        return a, b

    @role(warps=1)
    @cute.jit
    def sink(self, ctx, *, a: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.consume(0):
                pass
        return a

    @role(warps=2)
    @cute.jit
    def user(self, ctx, *, b: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with b.consume(0) as h:
                h.access(read=True)
        return b


class ThreadsWriteForUmma(Model):
    """Threads write X; an MMA reads its overlap Y through the async proxy.

    ``fence="derived"`` uses the channel's commit fence. ``"hand"`` opts out and calls
    ``fence_async_shared()`` in the writer; None opts out without fencing.
    """

    def __init__(self, fence: str | None):
        self.fence = fence
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        y = self.smem_plan.buffer("Y", cutlass.Int32, overlaps=x)
        commit_fence = None if fence == "derived" else ()
        self.a = Ring(2, Threads, Umma, buffer={"X": x, "Y": y}, commit_fence=commit_fence)

    def layouts(self):
        return {"X": cute.make_layout((32, 2)), "Y": cute.make_layout((32, 2))}

    @role(warps=0)
    @cute.jit
    def writer(self, ctx, *, a: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.produce() as h:
                h.access("X", write=True)
                if cutlass.const_expr(self.fence == "hand"):
                    fence_async_shared()
        return a

    @role(warps=1)
    @cute.jit
    def mma(self, ctx, *, a: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.consume() as h:
                h.access("Y", read=True, by="umma")
        return a


class StoresToTmemAndSmemForUmma(Model):
    """One commit publishes TMEM dS over dP and SMEM X under Y to an MMA.

    Like FA4 backward's dS commit, it needs both fence kinds in order.
    """

    def __init__(self, fences):
        self.fences = fences
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        y = self.smem_plan.buffer("Y", cutlass.Int32, overlaps=x)
        self.tmem_plan = TmemPlan().region("dP", 64).alias("dS", of="dP", offset=0, cols=64)
        # Two slots so the buffers have static stage spans; only slot 0 is used.
        self.ds = Slots(2, Threads, Umma, buffer={"X": x, "Y": y}, commit_fence=fences)

    def layouts(self):
        return {"X": cute.make_layout((32, 2)), "Y": cute.make_layout((32, 2))}

    @role(warps=range(4))
    @cute.jit
    def compute(self, ctx, *, ds: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ds.produce(0) as h:
                tmem.access("dS", write=True)  # tcgen05.st
                h.access("X", write=True)  # st.shared
        return ds

    @role(warps=4)
    @cute.jit
    def mma(self, ctx, *, ds: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ds.consume(0) as h:
                tmem.access("dP", read=True, by="umma")  # the dS columns as an A operand
                h.access("Y", read=True, by="umma")
        return ds


FENCE_CALLS = {
    "fence_view_async_tmem_store": "tmem_store",
    "fence_view_async_tmem_load": "tmem_load",
    "fence_view_async_shared": "smem_async",
}


@pytest.mark.parametrize(
    "fences, tmem_ok, smem_ok",
    [
        (("tmem_store", "smem_async"), True, True),
        (("smem_async", "tmem_store"), True, True),
        ("smem_async", False, True),
        ("tmem_store", True, False),
        (None, False, True),
        ((), False, False),
    ],
    ids=["both", "both_reversed", "smem_only", "tmem_only", "derived", "none"],
)
def test_commit_fence_tuple_covers_both_stores(monkeypatch, fences, tmem_ok, smem_ok):
    """Each fence covers its store kind and runs once per commit, in declaration order.

    None derives smem_async. This model has no fences outside the commit.
    """
    emitted = record_arch_calls(monkeypatch, FENCE_CALLS)
    op = StoresToTmemAndSmemForUmma(fences)
    if fences is None:
        declared = ("smem_async",)
    else:
        declared = (fences,) if isinstance(fences, str) else fences
    tag = "derived" if fences is None else "_".join(declared) or "none"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compile_toy.__wrapped__(op, f"fence_tuple_{tag}")  # traced, not cached
    assert emitted == list(declared)
    text = alias_section(op)
    assert status(op, "dS ~ dP") == ("no hazard found" if tmem_ok else "unverified"), text
    assert status(op, "Y ~ X") == ("no hazard found" if smem_ok else "unverified"), text
    assert ("[tmem_store]" in text) != tmem_ok and ("[proxy]" in text) != smem_ok


def test_raw_commit_emits_no_commit_fences(monkeypatch):
    """``commit(signal=False)`` leaves the declared commit fences to the raw code."""
    emitted = record_arch_calls(monkeypatch, FENCE_CALLS)

    class Op(StoresToTmemAndSmemForUmma):
        @role(warps=range(4))
        @cute.jit
        def compute(self, ctx, *, ds: Producer, tmem: Tmem):
            for _ in cutlass.range(ctx.n, unroll=1):
                h = ds.acquire(0)
                tmem.access("dS", write=True)
                h.access("X", write=True)
                h.commit(signal=False)
            return ds

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the raw variant is unverified
        compile_toy.__wrapped__(Op(("tmem_store", "smem_async")), "fence_order_raw")
    assert emitted == []


class ReadsTmaFilledStage(Model):
    """Consumers read each stage of a ring with generic loads, then the producer refills it:
    without a proxy fence a TMA refill may land in a stage still being read."""

    def __init__(self, producer, consumer, **fence):
        self.ring = Ring(2, producer, consumer, payload=(cutlass.Int32, (32,)), **fence)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ring: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.produce():
                pass
        return ring

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ring: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.consume() as h:
                h.data[cute.arch.lane_idx()] += 1  # generic ld/st.shared of the stage
        return ring


@pytest.mark.parametrize(
    "producer, consumer, fence, fenced",
    [
        (Tma(128), Threads, {}, True),
        (Tma(128), Warps, {}, True),
        (Tma(128), Warps, {"release_fence": ()}, False),
        (Tma(128), Umma, {}, False),
        (Threads, Threads, {}, False),
    ],
    ids=["tma_threads", "tma_warps", "tma_warps_opt_out", "tma_umma", "threads_threads"],
)
def test_tma_refilled_stage_is_proxy_fenced_before_release(
    monkeypatch, producer, consumer, fence, fenced
):
    """A generic stage read needs fence.proxy.async before release to order the next TMA fill.

    Every reading thread must fence. Tma-ring consumers emit it unless the channel declares
    ``release_fence=()``.
    """
    names = ("fence_view_async_shared", "mbarrier_arrive")
    emitted = record_arch_calls(monkeypatch, {name: name for name in names})
    op = ReadsTmaFilledStage(producer, consumer, **fence)
    tag = f"{getattr(producer, '__name__', 'tma')}_{consumer.__name__}_{len(fence)}".lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the Tma producer issues no copies
        compile_toy.__wrapped__(op, f"tma_refill_fence_{tag}")  # traced, not cached
    assert emitted.count("fence_view_async_shared") == fenced
    if fenced:  # the fence precedes the consumer's arrival on the empty barrier
        assert emitted[emitted.index("fence_view_async_shared") + 1] == "mbarrier_arrive"


@pytest.mark.parametrize(
    "fence, shown",
    [
        (None, None),
        ("smem_async", "release_fence smem_async -> smem_async\n"),
        ((), "release_fence smem_async -> none  (differs from derived)"),
        ("tmem_store", "release_fence smem_async -> tmem_store  (differs from derived)"),
    ],
    ids=["derived", "explicit_default", "opt_out", "replaced"],
)
def test_describe_shows_a_replaced_release_fence(fence, shown):
    """An explicit release_fence replaces the derived one (it does not add to it), and
    describe() shows what it replaced."""
    op = ReadsTmaFilledStage(Tma(128), Warps, release_fence=fence)
    text = op.describe()
    assert ("release_fence smem_async ->" in text) == (shown is not None), text
    if shown is not None:
        assert shown in text, text


def test_external_consumer_of_a_tma_ring_derives_no_release_fence():
    """External consumers supply their own fences; the layer must derive none for them."""
    ring = Ring(2, Tma(128), Threads, external="consumer", override={"empty_counts": 1})
    assert ring.release_fence == ()


class WritesStageForUmma(Model):
    """Generic-store producer with configurable consumer and storage.

    MMA reads through the async proxy; each writer needs a proxy fence to avoid stale reads.
    ``storage`` is ``"payload"``, ``"buffer"``, or None for a TMEM handoff.
    """

    def __init__(self, producer, consumer, storage, **fence):
        self.storage = storage
        place = {}
        if storage == "payload":
            place = {"payload": (cutlass.Int32, (32,))}
        elif storage == "buffer":
            self.smem_plan = SmemPlan()
            place = {"buffer": self.smem_plan.buffer("X", cutlass.Int32, align=128)}
        self.ring = Ring(2, producer, consumer, **place, **fence)

    def layouts(self):
        return {"X": cute.make_layout((32, 2))} if self.storage == "buffer" else {}

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ring: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.produce() as h:
                if cutlass.const_expr(self.storage is not None):
                    h.data[cute.arch.lane_idx()] = 1
        return ring

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ring: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.consume():
                pass
        return ring


@pytest.mark.parametrize(
    "producer, consumer, storage, fence, fenced",
    [
        (Threads, Umma, "buffer", {}, True),
        (Warps, Umma, "payload", {}, True),
        (Threads, Umma, "buffer", {"commit_fence": ()}, False),
        (Threads, Umma, None, {}, False),
        (Threads, Threads, "buffer", {}, False),
    ],
    ids=["threads_buffer", "warps_payload", "opt_out", "tmem_handoff", "threads_consumer"],
)
def test_thread_written_smem_for_umma_is_proxy_fenced_before_commit(
    monkeypatch, producer, consumer, storage, fence, fenced
):
    """Every SMEM writer fences before committing to an MMA consumer.

    Threads/Warps -> Umma derives fence.proxy.async unless ``commit_fence=()`` opts out.
    Without SMEM storage, no fence is derived.
    """
    names = ("fence_view_async_shared", "mbarrier_arrive")
    emitted = record_arch_calls(monkeypatch, {name: name for name in names})
    op = WritesStageForUmma(producer, consumer, storage, **fence)
    tag = f"{producer.__name__}_{consumer.__name__}_{storage}_{len(fence)}".lower()
    compile_toy.__wrapped__(op, f"umma_commit_fence_{tag}")  # traced, not cached
    assert emitted.count("fence_view_async_shared") == fenced
    if fenced:  # the full-barrier arrival must follow the fence
        assert emitted[emitted.index("fence_view_async_shared") + 1] == "mbarrier_arrive"


@pytest.mark.parametrize(
    "fence, shown",
    [
        (None, None),
        ("smem_async", "commit_fence smem_async -> smem_async\n"),
        ((), "commit_fence smem_async -> none  (differs from derived)"),
        (
            ("tmem_store", "smem_async"),
            "commit_fence smem_async -> tmem_store+smem_async  (differs from derived)",
        ),
    ],
    ids=["derived", "explicit_default", "opt_out", "replaced"],
)
def test_describe_shows_a_replaced_commit_fence(fence, shown):
    op = WritesStageForUmma(Threads, Umma, "buffer", commit_fence=fence)
    text = op.describe()
    assert ("commit_fence smem_async ->" in text) == (shown is not None), text
    if shown is not None:
        assert shown in text, text


def test_external_producer_of_an_umma_channel_derives_no_commit_fence():
    """External producers supply their own fences."""
    ring = Ring(
        2,
        Threads,
        Umma,
        payload=(cutlass.Int32, (32,)),
        external="producer",
        override={"full_counts": 1},
    )
    assert ring.commit_fence == ()


class LoadsUmmaStage(Model):
    """MMA-to-thread handoff model for TMEM stage reuse.

    Consumers must finish TMEM loads with tcgen05.wait::ld before release lets the next MMA
    overwrite those columns.
    """

    def __init__(self, consumer, **fence):
        self.ring = Ring(2, Umma, consumer, **fence)

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, ring: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.produce():
                pass
        return ring

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ring: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ring.consume():
                pass
        return ring


@pytest.mark.parametrize("consumer", [Threads, Warps])
@pytest.mark.parametrize("fence", ["tmem_load", ("tmem_load", "tmem_store"), None])
def test_tmem_load_release_fence_waits_for_loads_before_the_release(monkeypatch, consumer, fence):
    """Every consumer runs tcgen05.wait::ld before its empty-barrier arrival.

    Fences run in declaration order; no release fence is derived for this channel.
    """
    emitted = record_arch_calls(monkeypatch, {**FENCE_CALLS, "mbarrier_arrive": "arrive"})
    op = LoadsUmmaStage(consumer, release_fence=fence)
    kinds = (fence,) if isinstance(fence, str) else fence or ()
    tag = "_".join(kinds) or "none"
    compile_toy.__wrapped__(op, f"tmem_load_fence_{consumer.__name__.lower()}_{tag}")
    assert emitted == [*kinds, "arrive"]


class LoopInsideHandle(LoopCarriedMma):
    """The same loop inside one handle: the handle must not hide the loop-carried pair."""

    def __init__(self):
        super().__init__()
        self.s = Slots(1, Umma, Threads)

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, s: Producer, tmem: Tmem):
        with s.produce(0):
            for _ in cutlass.range(ctx.n, unroll=1):
                tmem.access("P", read=True, by="umma")
                tmem.access("S", write=True, by="umma")
        return s

    @role(warps=1)
    @cute.jit
    def sink(self, ctx, *, s: Consumer):
        with s.consume(0):
            pass
        return s


# A consumer that releases with signal=False and arrives itself.
# The reader -> next writer order exists only through that raw signal, which the diagnostic
# cannot see: unverified (its documented answer for raw signals), not a potential hazard.


class RawReleaseOverlap(Model):
    """The producer writes Q, the consumer reads P (Q overlaps P) in both slots of one
    channel; ``raw``: the consumer releases with ``signal=False`` plus one elected arrival."""

    def __init__(self, raw: bool):
        self.raw = raw
        self.smem_plan = SmemPlan()
        p = self.smem_plan.buffer("P", cutlass.Int32)
        q = self.smem_plan.buffer("Q", cutlass.Int32, overlaps=p)
        override = {"empty_counts": 1} if raw else None
        self.c = Slots(2, Threads, Threads, buffer={"P": p, "Q": q}, override=override)

    def layouts(self):
        return {"P": cute.make_layout((32, 2)), "Q": cute.make_layout((32, 2))}

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, c: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for s in cutlass.range_constexpr(2):
                with c.produce(s) as h:
                    h.access("Q", write=True)
        return c

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for s in cutlass.range_constexpr(2):
                h = c.wait(s)
                h.access("P", read=True)
                if cutlass.const_expr(self.raw):
                    cute.arch.sync_warp()
                    if cute.arch.lane_idx() == 0:
                        cute.arch.mbarrier_arrive(h.empty_barrier)
                    h.release(signal=False)
                else:
                    h.release()
        return c


class RawCommitBesideTma(Model):
    """The producer writes Q and publishes it with ``commit(signal=False)`` plus an arrival,
    after committing a TMA channel the consumer also waits on. The TMA commit does not publish
    the write (complete_tx), but the raw signal might: unverified, not a potential hazard."""

    def __init__(self):
        self.smem_plan = SmemPlan()
        p = self.smem_plan.buffer("P", cutlass.Int32)
        q = self.smem_plan.buffer("Q", cutlass.Int32, overlaps=p)
        self.c = Slots(1, Threads, Threads, buffer={"P": p, "Q": q}, override={"full_counts": 1})
        self.t = Slots(1, Tma(128), Threads)

    def layouts(self):
        return {"P": cute.make_layout((32, 1)), "Q": cute.make_layout((32, 1))}

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, c: Producer, t: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            ht = t.acquire(0)
            h = c.acquire(0)
            h.access("Q", write=True)
            ht.commit()
            cute.arch.sync_warp()
            if cute.arch.lane_idx() == 0:
                cute.arch.mbarrier_arrive(h.full_barrier)
            h.commit(signal=False)
        return c, t

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer, t: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with t.consume(0):
                pass
            with c.consume(0) as h:
                h.access("P", read=True)
        return c, t


class RawReleaseBesideUmma(RawReleaseOverlap):
    """A raw-signaled release leaves read publication unverified.

    The consumer closes with ``release(signal=False)`` and arrives on the empty barrier itself.
    Around the read it also holds ``d``, a single-warp Umma-consumer channel. Releasing ``d``
    cannot publish the thread read, but that failed edge must not override the raw signal
    that might.
    """

    def __init__(self):
        super().__init__(raw=False)
        self.d = Slots(1, Threads, Umma)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, c: Producer, d: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with d.produce(0):
                pass
            for s in cutlass.range_constexpr(2):
                with c.produce(s) as h:
                    h.access("Q", write=True)
        return c, d

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer, d: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            g = d.wait(0)
            for s in cutlass.range_constexpr(2):
                h = c.wait(s)
                h.access("P", read=True)
                h.release(signal=False)
                cute.arch.mbarrier_arrive(h.empty_barrier)
            g.release()
        return c, d


class SameRoleRawRoundTrip(Model):
    """One role TMA-writes Q in one handle of ``t`` and reads P (Q overlaps P) in the next;
    ``sink`` releases both stages, with ``signal=False`` when ``raw``. The write -> read order
    is a round trip through that release; the read -> next write order has no edge at all."""

    def __init__(self, raw: bool):
        self.raw = raw
        self.smem_plan = SmemPlan()
        p = self.smem_plan.buffer("P", cutlass.Int32, align=128)
        q = self.smem_plan.buffer("Q", cutlass.Int32, overlaps=p)
        override = {"empty_counts": 1} if raw else None
        self.t = Slots(1, Tma(128), Threads, buffer={"P": p, "Q": q}, override=override)

    def layouts(self):
        return {"P": cute.make_layout((32, 1)), "Q": cute.make_layout((32, 1))}

    @role(warps=0)
    @cute.jit
    def load(self, ctx, *, t: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with t.produce(0) as h:
                h.access("Q", write=True, by="tma")
            with t.produce(0) as h:
                h.access("P", read=True)
        return t

    @role(warps=1)
    @cute.jit
    def sink(self, ctx, *, t: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for _k in cutlass.range_constexpr(2):
                h = t.wait(0)
                if cutlass.const_expr(self.raw):
                    if cute.arch.lane_idx() == 0:
                        cute.arch.mbarrier_arrive(h.empty_barrier)
                    h.release(signal=False)
                else:
                    h.release()
        return t


# A round-robin ring: its epochs alternate between the instances, which the model's epoch
# matching does not follow, so an ordering through it is unverified, as through an exclusive
# ring, never a potential hazard.


class RoundRobinRoundTrip(Model):
    """``prod`` writes S and publishes it on round-robin ``s``; instance j of ``soft`` reads its
    warp's lanes of P (an alias in S) in epochs j, j + 2, ..., then hands the epoch back on
    round-robin ``back``, which prod waits for before its next write: a valid ordering.
    ``early``: soft reads P before waiting for it, which no path orders after the write."""

    def __init__(self, early: bool):
        self.early = early
        rr = Sharing.ROUND_ROBIN
        self.s = Ring(2, Threads, Threads, consumer_sharing=rr, commit_fence="tmem_store")
        self.back = Ring(2, Threads, Threads, producer_sharing=rr)
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, s: Producer, back: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            tmem.access("S", write=True, lanes=range(32, 96))
            with s.produce():
                pass
            with back.consume():
                pass
        return s, back

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def soft(self, ctx, *, s: Consumer, back: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            if cutlass.const_expr(self.early):
                tmem.access("P", read=True)
            with s.consume():
                if cutlass.const_expr(not self.early):
                    tmem.access("P", read=True)
            with back.produce():
                pass
        return s, back


HAZARD, UNVERIFIED, SAFE = "potential hazard", "unverified", "no hazard found"
CHANNEL = r"no channel path orders them \[channel\]"
VERDICTS = {  # id: (model, pair, status, evidence regex in the alias section)
    "ring1": (lambda: RingTmemHandoff(1), "P ~ S", SAFE, ""),
    # tcgen05.ld completes only at tcgen05.wait::ld: a release without it may let the next MMA
    # overwrite columns a load still reads.
    "ring1_unwaited_load": (
        lambda: RingTmemHandoff(1, waited=False),
        "P ~ S",
        UNVERIFIED,
        r"tcgen05\.ld completes only at .*\[tmem_load\]",
    ),
    # One shared handoff stopped only by an unverified completion orders the pair on its own:
    # another shared channel that cannot complete the accesses must not turn it into a hazard.
    "ring1_unwaited_load_beside_control": (
        RingTmemHandoffBesideControl,
        "P ~ S",
        UNVERIFIED,
        r"tcgen05\.ld completes only at .*\[tmem_load\]",
    ),
    "ring2": (lambda: RingTmemHandoff(2), "P ~ S", HAZARD, CHANNEL),
    # One handoff fails twice: an unverified store and an MMA read a thread arrival cannot
    # complete. The hazard must not hide behind the unverified finding.
    "umma_read_thread_release": (UmmaReadReleasedByThreads, "P ~ S", HAZARD, r"\[commit\]"),
    "loop_carried": (
        LoopCarriedMma,
        "P ~ S",
        HAZARD,
        r"mma write S\[0\] by umma @ .* then mma read P\[0\] by umma @ .*not a pipelined pair",
    ),
    "loop_in_handle": (LoopInsideHandle, "P ~ S", HAZARD, ""),  # the handle must not hide it
    "tma_other_channel": (TmaThroughOtherChannel, "Y ~ X", HAZARD, r"\[complete_tx\]"),
    "no_proxy_fence": (lambda: ThreadsWriteForUmma(None), "Y ~ X", UNVERIFIED, r"\[proxy\]"),
    "proxy_fence": (lambda: ThreadsWriteForUmma("hand"), "Y ~ X", SAFE, ""),
    "derived_proxy_fence": (lambda: ThreadsWriteForUmma("derived"), "Y ~ X", SAFE, ""),
    # Keep signal=False closes in the graph: raw signaling leaves ordering unverified,
    # not a potential hazard, even under strict mode.
    "release": (lambda: RawReleaseOverlap(False), "Q ~ P", SAFE, ""),
    "raw_release": (
        lambda: RawReleaseOverlap(True),
        "Q ~ P",
        UNVERIFIED,
        r"only a raw signal \(signal=False\) would order them",
    ),
    "raw_commit_beside_tma": (RawCommitBesideTma, "Q ~ P", UNVERIFIED, ""),
    "raw_release_beside_umma": (RawReleaseBesideUmma, "Q ~ P", UNVERIFIED, "only a raw signal"),
    # The write -> read pair, ordered only through the raw release, is unverified. Reporting it
    # must not end the scan before the read -> next-write pair, a potential hazard.
    "unverified_then_hazard": (
        lambda: SameRoleRawRoundTrip(raw=True),
        "Q ~ P",
        HAZARD,
        r"\[pipelined\]",
    ),
    # Keep round-robin edges in the graph: unknown epoch matching is unverified,
    # not a potential hazard, even under strict mode.
    "round_robin": (
        lambda: RoundRobinRoundTrip(early=False),
        "P ~ S",
        UNVERIFIED,
        "s: round-robin ring, epochs alternate between instances",
    ),
    "round_robin_early_read": (lambda: RoundRobinRoundTrip(early=True), "P ~ S", HAZARD, CHANNEL),
}


@pytest.mark.parametrize("case", VERDICTS)
def test_alias_verdicts(case):
    """Each model's pair gets its verdict; only a potential hazard raises under ``strict``."""
    make, pair, expected, evidence = VERDICTS[case]
    op = make()
    # no_proxy_fence also has a missing fence, which raises in strict mode.
    if expected != HAZARD and case != "no_proxy_fence":
        op.alias_check = "strict"
    compile_quietly(op, case)
    text = alias_section(op)
    assert status(op, pair) == expected, text
    assert re.search(evidence, text), text


def _event(kind: str, where: tuple = (), /, **data) -> Event:
    return Event(kind, data, "toy.py:1", where, ())


IF, LOOP = ((7, "if"),), ((7, "loop"),)


@pytest.mark.parametrize(
    "load, store, wait, signal, completed",
    [
        ((), None, (), (), False),  # nothing completes the load
        ((), (), (), (), True),  # a later same-warp store, waited: ld -> st is pipelined
        ((), IF, (), (), False),  # the store may be skipped
        ((), LOOP, (), (), False),  # in a loop that may run zero times
        ((), (), IF, (), False),  # the wait::st may be skipped
        ((), IF, IF, IF, True),  # the signal runs only where the store and its wait ran
        (IF, IF, IF, (), True),  # and the load only there
    ],
    ids=[
        "no_store",
        "store",
        "store_in_if",
        "store_in_loop",
        "wait_in_if",
        "signal_in_if",
        "load_in_if",
    ],
)
def test_load_completes_only_through_a_store_on_every_path(load, store, wait, signal, completed):
    """A tcgen05.ld handed to a later writer completes through a later tcgen05.st of the same
    warp and its wait::st only if both run whenever the load and the signal do. The wait::st is
    another channel's fenced release, as managed signals record it: on the close event."""
    loaded = SimpleNamespace(
        agent="threads", read=True, write=False, storage=("tmem", "x"), parts=None
    )
    writer = SimpleNamespace(agent="umma", read=False, write=True)
    events = [_event("access", load)]
    if store is not None:
        events.append(
            _event("access", store, write=True, agent="threads", storage=("tmem", "y"), warps=None)
        )
    other = Slots(1, Umma, Threads, release_fence="tmem_store")
    release = SimpleNamespace(chan=other, producer=False, role="r")
    events += [
        _event("close", wait, handle=release, signal=True, fenced=True),
        _event("close", signal),
    ]
    role = RoleEvents("r", (0,), False, events, None)
    seq = list(range(len(events)))
    found = completion(
        loaded, writer, role, seq, 0, len(events) - 1, mech="Threads", fence=(), own=False
    )
    assert (found is None) == completed, found
    if not completed:
        assert (found.severity, found.rule) == (UNVERIFIED, "tmem_load")
