"""Alias-diagnostic regression models: small single-CTA kernels (compiled, never launched) whose
alias orderings the diagnostic must flag or accept. FA4 mutants: test_warpdrive_alias.py."""

import re
import warnings
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
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
    unchecked,
)


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


class Model(WarpSpecialized):
    """Roles loop ``n`` times; ``layouts()`` gives the SMEM plan's layouts, if any."""

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


# --------------------------------------------------------------------------- #
# FA4-backward-style cross-warp case: eight warps share an S region's lanes in two warpgroups;
# each warpgroup overwrites its half of the P columns (an alias inside S) while the other
# warpgroup's warp on the same lanes may still be loading S.
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# A Ring hands storage over stage by stage: with two stages the MMA may rewrite one S region
# while the epilogue still reads the previous epoch's P from it.
# --------------------------------------------------------------------------- #


class RingTmemHandoff(Model):
    def __init__(self, stages: int):
        self.acc = Ring(stages, Umma, Threads)
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


@pytest.mark.parametrize("stages, expected", [(1, "no hazard found"), (2, "potential hazard")])
def test_ring_hands_over_unstaged_tmem_only_with_one_stage(stages, expected):
    op = RingTmemHandoff(stages)
    compile_quietly(op, f"ring_tmem_{stages}")
    assert status(op, "P ~ S") == expected, alias_section(op)
    if stages == 2:
        assert "no channel path orders them [channel]" in alias_section(op)


# --------------------------------------------------------------------------- #
# One role, no synchronization between iterations: an MMA's D write to S in iteration i and the
# A read of P in iteration i + 1 are not a pipelined pair (the trace order alone, A read then
# D write, is).
# --------------------------------------------------------------------------- #


class LoopCarriedMma(Model):
    def __init__(self):
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            tmem.access("P", read=True, by="umma")  # PV
            tmem.access("S", write=True, by="umma")  # next QK


def test_loop_carried_same_role_pair_is_checked():
    op = LoopCarriedMma()
    compile_quietly(op, "loop_carried")
    assert status(op, "P ~ S") == "potential hazard"
    assert re.search(
        r"mma write S\[0\] by umma @ .* then mma read P\[0\] by umma @ .*not a pipelined pair",
        alias_section(op),
    ), alias_section(op)


# --------------------------------------------------------------------------- #
# Unknown epoch correspondence: the MMA peels its last tile out of the loop (as FA4 backward
# does), so its traced per-item channel counts (two unrolled iterations plus one) differ from
# the softmax's (two). The round trip through ``s`` orders the loop-carried S write -> P read
# pair in reality, but the trace cannot match the epochs: unverified, not a potential hazard.
# A pair no epoch correspondence could order (S written before P is read in one handle) stays
# a potential hazard.
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# A TMA load is published only by its own handle's transaction count: a reader that waits on
# another TMA channel the loader fills is not ordered after it.
# --------------------------------------------------------------------------- #


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


def test_tma_load_is_published_only_by_its_own_handle():
    op = TmaThroughOtherChannel()
    compile_quietly(op, "tma_other_channel")
    assert status(op, "Y ~ X") == "potential hazard", alias_section(op)
    assert "[complete_tx]" in alias_section(op)


class ThreadsWriteForUmma(Model):
    """Threads write X, an MMA reads its overlap Y: the generic-to-async proxy fence comes from
    ``fence_async_shared()`` in the writer, not from the channel."""

    def __init__(self, fence: bool):
        self.fence = fence
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        y = self.smem_plan.buffer("Y", cutlass.Int32, overlaps=x)
        self.a = Ring(2, Threads, Umma, buffer={"X": x, "Y": y})

    def layouts(self):
        return {"X": cute.make_layout((32, 2)), "Y": cute.make_layout((32, 2))}

    @role(warps=0)
    @cute.jit
    def writer(self, ctx, *, a: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.produce() as h:
                h.access("X", write=True)
                if cutlass.const_expr(self.fence):
                    fence_async_shared()
        return a

    @role(warps=1)
    @cute.jit
    def mma(self, ctx, *, a: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.consume() as h:
                h.access("Y", read=True, by="umma")
        return a


@pytest.mark.parametrize("fence", [False, True])
def test_fence_async_shared_supplies_the_proxy_fence(fence):
    op = ThreadsWriteForUmma(fence)
    compile_quietly(op, f"threads_for_umma_{int(fence)}")
    expected = "no hazard found" if fence else "unverified"
    assert status(op, "Y ~ X") == expected, alias_section(op)
    assert ("[proxy]" in alias_section(op)) != fence


class StoresToTmemAndSmemForUmma(Model):
    """One commit hands over both a TMEM store (dS over dP) and an SMEM store (X under Y) to an
    MMA, as FA4 backward's dS commit does: the side needs both fence kinds, in order."""

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


@pytest.mark.parametrize(
    "fences, tmem_ok, smem_ok",
    [
        (("tmem_store", "smem_async"), True, True),
        (("smem_async", "tmem_store"), True, True),
        ("smem_async", False, True),
        ("tmem_store", True, False),
        (None, False, False),
    ],
    ids=["both", "both_reversed", "smem_only", "tmem_only", "none"],
)
def test_commit_fence_tuple_covers_both_stores(fences, tmem_ok, smem_ok):
    op = StoresToTmemAndSmemForUmma(fences)
    tag = "_".join(fences) if isinstance(fences, tuple) else str(fences).lower()
    compile_quietly(op, f"fence_tuple_{tag}")
    text = alias_section(op)
    assert status(op, "dS ~ dP") == ("no hazard found" if tmem_ok else "unverified"), text
    assert status(op, "Y ~ X") == ("no hazard found" if smem_ok else "unverified"), text
    assert ("[tmem_store]" in text) != tmem_ok and ("[proxy]" in text) != smem_ok


@pytest.mark.parametrize(
    "fences, signal",
    [
        (("tmem_store", "smem_async"), True),
        (("smem_async", "tmem_store"), True),
        ("smem_async", True),
        (("tmem_store", "smem_async"), False),
    ],
    ids=["both", "both_reversed", "single", "raw_signal"],
)
def test_commit_fences_are_emitted_in_declaration_order(monkeypatch, fences, signal):
    """The layer emits exactly the declared fences, in order, once per managed commit, and none
    for ``commit(signal=False)``; the only fences in this kernel are the commit's."""
    emitted = []
    for kind, name in (
        ("tmem_store", "fence_view_async_tmem_store"),
        ("smem_async", "fence_view_async_shared"),
    ):
        original = getattr(cute.arch, name)

        def record(*args, _kind=kind, _original=original, **kwargs):
            emitted.append(_kind)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cute.arch, name, record)

    class Op(StoresToTmemAndSmemForUmma):
        @role(warps=range(4))
        @cute.jit
        def compute(self, ctx, *, ds: Producer, tmem: Tmem):
            for _ in cutlass.range(ctx.n, unroll=1):
                h = ds.acquire(0)
                tmem.access("dS", write=True)
                h.access("X", write=True)
                h.commit(signal=signal)
            return ds

    tag = ("_".join(fences) if isinstance(fences, tuple) else fences) + f"_{int(signal)}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the single-fence and raw variants are unverified
        compile_toy.__wrapped__(Op(fences), f"fence_order_{tag}")  # traced, not cached
    expected = list(fences if isinstance(fences, tuple) else (fences,)) if signal else []
    assert emitted == expected


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
    """A generic read of a TMA-filled stage is ordered before the next fill only by
    fence.proxy.async in the reading thread before its release, so every consumer release of a
    Tma ring emits one unless the channel declares ``release_fence=()``."""
    emitted = []
    for name in ("fence_view_async_shared", "mbarrier_arrive"):
        original = getattr(cute.arch, name)

        def record(*args, _name=name, _original=original, **kwargs):
            emitted.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(cute.arch, name, record)

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
    """Raw code fences an external consumer itself: the layer derives nothing that the external
    side's no-fences check would reject."""
    ring = Ring(2, Tma(128), Threads, external="consumer", override={"empty_counts": 1})
    assert ring.release_fence == ()


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


def test_a_handle_around_the_loop_does_not_hide_the_loop_carried_pair():
    op = LoopInsideHandle()
    compile_quietly(op, "loop_in_handle")
    assert status(op, "P ~ S") == "potential hazard", alias_section(op)


# --------------------------------------------------------------------------- #
# Runtime fuzz finding F-C1: a consumer that releases with signal=False and arrives itself.
# The reader -> next writer order exists only through that raw signal, which the diagnostic
# cannot see: unverified (its documented answer for raw signals), not a potential hazard.
# --------------------------------------------------------------------------- #


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


class UnclosedRelease(RawReleaseOverlap):
    """The consumer opens under ``unchecked("unclosed")`` and arrives on the empty barrier
    itself instead of releasing: the close the order needs is not traced at all."""

    def __init__(self):
        super().__init__(raw=False)

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for s in cutlass.range_constexpr(2):
                with unchecked("unclosed", reason="released by the arrival below"):
                    h = c.wait(s)
                h.access("P", read=True)
                cute.arch.mbarrier_arrive(h.empty_barrier)
        return c


class UnclosedBesideUmma(UnclosedRelease):
    """As ``UnclosedRelease``, but the consumer also holds a stage of ``d``, a single-warp Umma
    consumer channel, around the read: ``d``'s release cannot publish the read (a failed edge),
    which must not outrank the never-closed handle that might."""

    def __init__(self):
        super().__init__()
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
                with unchecked("unclosed", reason="released by the arrival below"):
                    h = c.wait(s)
                h.access("P", read=True)
                cute.arch.mbarrier_arrive(h.empty_barrier)
            g.release()
        return c, d


@pytest.mark.parametrize("model", [UnclosedRelease, UnclosedBesideUmma])
def test_unclosed_handle_is_unverified_not_a_hazard(model):
    op = model()
    op.alias_check = "strict"
    compile_quietly(op, f"unclosed_{model.__name__.lower()}")
    assert status(op, "Q ~ P") == "unverified", alias_section(op)
    assert "never closes" in alias_section(op)


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


def test_an_unverified_pair_does_not_hide_a_later_hazard():
    """The write -> read pair, ordered only through the raw release, is unverified. Reporting it
    must not end the scan before the read -> next-write pair, a potential hazard."""
    op = SameRoleRawRoundTrip(raw=True)
    compile_quietly(op, "same_role_raw")
    assert status(op, "Q ~ P") == "potential hazard", alias_section(op)
    assert "[pipelined]" in alias_section(op)


def test_raw_commit_beside_a_tma_commit_is_unverified():
    op = RawCommitBesideTma()
    op.alias_check = "strict"
    compile_quietly(op, "raw_commit_beside_tma")
    assert status(op, "Q ~ P") == "unverified", alias_section(op)


@pytest.mark.parametrize("raw, expected", [(False, "no hazard found"), (True, "unverified")])
def test_raw_release_is_unverified_not_a_hazard(raw, expected):
    """Once reported ``potential hazard: ... no channel path orders them`` (and raised under
    ``alias_check="strict"``) because closes with ``signal=False`` were dropped from the graph."""
    op = RawReleaseOverlap(raw)
    op.alias_check = "strict"
    compile_quietly(op, f"raw_release_overlap_{int(raw)}")
    assert status(op, "Q ~ P") == expected, alias_section(op)
    if raw:
        assert "only a raw signal (signal=False) would order them" in alias_section(op)


# --------------------------------------------------------------------------- #
# A round-robin ring: its epochs alternate between the instances, which the model's epoch
# matching does not follow, so an ordering through it is unverified, as through an exclusive
# ring, never a potential hazard.
# --------------------------------------------------------------------------- #


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


def test_round_robin_ordering_is_unverified_not_a_hazard():
    """Once reported ``no channel path orders them`` (and raised under ``strict``): the
    round-robin rings' edges were dropped from the graph."""
    op = RoundRobinRoundTrip(early=False)
    op.alias_check = "strict"
    compile_quietly(op, "round_robin_round_trip")
    assert status(op, "P ~ S") == "unverified", alias_section(op)
    assert "s: round-robin ring, epochs alternate between instances" in alias_section(op)


def test_read_before_a_round_robin_wait_is_a_potential_hazard():
    op = RoundRobinRoundTrip(early=True)
    compile_quietly(op, "round_robin_early_read")
    assert status(op, "P ~ S") == "potential hazard", alias_section(op)
    assert "no channel path orders them [channel]" in alias_section(op)
