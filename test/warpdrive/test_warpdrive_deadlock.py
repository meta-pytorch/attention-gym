"""Trace-time deadlock checks: a role waiting for an epoch that needs a stage it still holds."""

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.testing.warpdrive import raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    Instance,
    Producer,
    Ring,
    Sharing,
    Slots,
    Threads,
    Tma,
    WarpSpecialized,
    role,
    unchecked,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor


class Toy(WarpSpecialized):
    """One CTA; roles get ``ctx = (out, n)``."""

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("warpdrive_deadlock")
        self.kernel(out, n).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_toy(op, key: str):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name=f"warpdrive_deadlock_{key}")


class HoldAndWait(Toy):
    """``mode`` selects how the consumer of ``ch`` (``stages`` deep) holds one epoch while it
    takes a later one; the producer fills ``epochs`` in order."""

    def __init__(self, mode: str, stages: int = 2, epochs: int = 3):
        self.mode, self.epochs = mode, epochs
        self.ch = Ring(stages, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        for _ in cutlass.range_constexpr(self.epochs):
            with ch.produce():
                pass
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        mode = self.mode
        held = ch.wait()
        if cutlass.const_expr(mode == "wait"):
            a, b = ch.wait(), ch.wait()  # b is two epochs after held
            a.release()
            b.release()
        elif cutlass.const_expr(mode == "peek"):
            ch.peek()  # one epoch after held
        elif cutlass.const_expr(mode == "branch"):
            a = ch.wait()
            if ctx.n > 1:
                with ch.consume():
                    pass
            a.release()
        elif cutlass.const_expr(mode == "nearby"):  # one epoch after held, two stages
            with ch.consume():
                pass
        elif cutlass.const_expr(mode == "loop"):  # one lap later only on a second trip
            for _ in cutlass.range(ctx.n, unroll=1):
                with ch.consume():
                    pass
        elif cutlass.const_expr(mode == "raw_signal"):
            a, b = ch.wait(), ch.wait()
            a.release()
            b.release()
        elif cutlass.const_expr(mode == "lap_in_loop"):  # each trip takes two epochs
            for _ in cutlass.range(ctx.n, unroll=1):
                a, b = ch.wait(), ch.wait()
                a.release()
                b.release()
        elif cutlass.const_expr(mode == "dynamic_skip"):
            pass  # see SkipHold
        if cutlass.const_expr(mode == "raw_signal"):
            cute.arch.mbarrier_arrive(held.empty_barrier)  # the author signals it earlier
            held.release(signal=False)
        else:
            held.release()
        if cutlass.const_expr(mode == "peek"):
            with ch.consume():
                pass
        return ch


class ProducerHold(Toy):
    """The producer acquires a ring lap past a stage it has not committed yet."""

    def __init__(self, tma: bool):
        self.ch = Ring(1, Tma(64) if tma else Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        held = ch.acquire()
        with ch.produce():
            pass
        held.commit()
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        for _ in cutlass.range_constexpr(2):
            with ch.consume():
                pass
        return ch


class SlotHold(Toy):
    """The consumer waits on one slot twice before releasing the first wait."""

    def __init__(self):
        self.ch = Slots(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        for _ in cutlass.range_constexpr(2):
            with ch.produce(0):
                pass
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        held = ch.wait(0)
        with ch.consume(0):
            pass
        held.release()
        return ch


class SpanHold(Toy):
    """The consumer begins a span a ring lap after a stage it still holds."""

    def __init__(self):
        self.ch = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        for _ in cutlass.range_constexpr(2):
            with ch.produce():
                pass
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        held = ch.wait()
        s = ch.span()
        for i in cutlass.range(2):
            s.begin(i)
            s.release(i)
        held.release()
        return ch


class SkipHold(Toy):
    """Exclusive consumers ``a`` (epochs 0 and 2) and ``b`` (epoch 1); ``a`` holds epoch 0
    across its skip."""

    def __init__(self, stages: int):
        self.ch = Ring(
            stages, Threads, Threads, consumer_sharing=Sharing.EXCLUSIVE, consumer_roles=("a", "b")
        )

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        for _ in cutlass.range_constexpr(3):
            with ch.produce():
                pass
        return ch

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, ch: Consumer):
        held = ch.wait()
        ch.skip(1)
        with ch.consume():
            pass
        held.release()
        return ch

    @role(warps=2)
    @cute.jit
    def b(self, ctx, *, ch: Consumer):
        ch.skip(1)
        with ch.consume():
            pass
        ch.skip(1)
        return ch


HOLD = r"which this role holds until {op} at .*\. On a {lap} that epoch is ready only after"


@pytest.mark.parametrize(
    "make, key, error",
    [
        (
            lambda: HoldAndWait("wait"),
            "wait",
            r"ch\.wait\(\) at .* takes an epoch at least 2 after ch",
        ),
        (
            lambda: HoldAndWait("peek", 1, 2),
            "peek",
            r"ch\.peek\(\) at .* takes an epoch at least 1",
        ),
        (lambda: HoldAndWait("branch"), "branch", r"ch\.wait\(\) at .* takes an epoch at least 2"),
        (
            lambda: ProducerHold(False),
            "producer",
            r"ch\.acquire\(\) at .* takes an epoch at least 1",
        ),
        (SlotHold, "slot", r"ch\.wait\(\) at .* takes an epoch at least 1 after ch\[0\] wait"),
        (SpanHold, "span", r"ch\.span begin\(\) at .* at least 1 after ch\[i\] wait"),
        (
            lambda: SkipHold(2),
            "skip",
            r"ch\.wait\(\) at .* takes an epoch at least 2 after ch\[i\]",
        ),
    ],
)
def test_waiting_a_ring_lap_past_a_held_stage_raises(make, key, error):
    raises_protocol(error, lambda: compile_toy(make(), f"hold_{key}"))


def test_the_error_names_the_held_stage_and_its_release():
    raises_protocol(
        HOLD.format(op="release", lap="2-stage ring"),
        lambda: compile_toy(HoldAndWait("wait"), "hold_message"),
    )


@pytest.mark.parametrize(
    "make, key",
    [
        # One epoch after the held one on two stages: the producer still has a free stage.
        (lambda: HoldAndWait("nearby", 2, 2), "nearby"),
        # Two stages, so the first trip is a free stage; later trips are the loop's to prove.
        (lambda: HoldAndWait("loop", 2, 2), "loop"),
        # The author signals the held stage before the second wait.
        (lambda: HoldAndWait("raw_signal"), "raw_signal"),
        # A TMA fill completes the stage without the producer's commit.
        (lambda: ProducerHold(True), "tma"),
        # Three stages leave one free past the held epoch and the skipped one.
        (lambda: SkipHold(3), "skip"),
    ],
)
@pytest.mark.filterwarnings("ignore:warpdrive. TMA check could not verify")  # no copies issued
def test_waits_that_do_not_need_a_held_stage_compile(make, key):
    compile_toy(make(), f"free_{key}")


def test_fa4_backward_with_one_q_stage_deadlocks(monkeypatch):
    """S_{t+1} previews Q_{t+1} before dK_t releases Q_t: one Q stage never fills it."""
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples import fa4_bwd

    monkeypatch.setattr(fa4_bwd, "Q_STAGE", 1)
    op = fa4_bwd.FlashAttentionBackward(causal=False)
    # Q_STAGE is not in the compile key: compile this one call without the caches.
    raises_protocol(
        r"mma: q\.peek\(\) at fa4_bwd\.py:\d+ takes an epoch at least 1 after q\[i\] wait opened at "
        r"fa4_bwd\.py:\d+, which this role holds until release",
        lambda: fa4_bwd.compile_fa4_bwd.__wrapped__(op),
    )


class RoundRobinHold(Toy):
    """Two round-robin consumers each hold their first epoch across their next wait, two
    epochs later."""

    def __init__(self, stages: int):
        self.ch = Ring(stages, Threads, Threads, consumer_sharing=Sharing.ROUND_ROBIN)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        for _ in cutlass.range_constexpr(4):
            with ch.produce():
                pass
        return ch

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def cons(self, ctx, *, stage: Instance, ch: Consumer):
        held = ch.wait()
        with ch.consume():
            pass
        held.release()
        return ch


class Arms(Toy):
    """``mode`` places a held handle and a later wait where they never run together, or holds
    one slot while waiting on another."""

    def __init__(self, mode: str):
        self.mode = mode
        fixed = mode in ("other_slot", "dynamic_slot")
        self.ch = Slots(2, Threads, Threads) if fixed else Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        if cutlass.const_expr(self.mode in ("other_slot", "dynamic_slot")):
            with ch.produce(0):
                pass
            for _ in cutlass.range_constexpr(2):
                with ch.produce(1):
                    pass
        else:
            for _ in cutlass.range_constexpr(2):
                with ch.produce():
                    pass
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        mode = self.mode
        if cutlass.const_expr(mode == "arms"):
            if ctx.n > 1:
                ch.wait()  # held past its arm
            else:
                for _ in cutlass.range_constexpr(2):
                    with ch.consume():
                        pass
        elif cutlass.const_expr(mode == "other_slot"):
            held = ch.wait(0)
            for _ in cutlass.range_constexpr(2):
                with ch.consume(1):
                    pass
            held.release()
        elif cutlass.const_expr(mode == "dynamic_slot"):
            ch.wait(0)
            ch.wait(ctx.n)
        return ch


class ExternalHold(Toy):
    """Raw code fills ``ch``; the consumer holds one stage across a second wait."""

    def __init__(self):
        self.ch = Ring(1, Threads, Threads, external="producer", override={"full_counts": 1})

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        a, b = ch.wait(), ch.wait()
        a.release()
        b.release()
        return ch


class DynamicSkipHold(SkipHold):
    """``a`` holds epoch 0 across ``skip(n)``, which may be 0."""

    def __init__(self):
        super().__init__(2)

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, ch: Consumer):
        held = ch.wait()
        ch.skip(ctx.n - 1)
        with ch.consume():
            pass
        held.release()
        return ch


@cute.jit
def _acquired(end):
    end.acquire().commit()
    return end


@cute.jit
def _waited(end):
    end.wait().release()
    return end


class SpanLifetime(Toy):
    """``mode`` places a wait on ``ch`` around a span of the consumer (``producer=True``: the
    producer, acquires) over a ``trips``-trip loop."""

    def __init__(self, mode: str, producer: bool = False, trips: int = 2):
        self.mode, self.producer, self.trips = mode, producer, trips
        self.epochs = 1 + (trips if mode == "last_trip" else 1)  # the span's and the waits'
        self.ch = Ring(1, Threads, Threads)

    @cute.jit
    def spanned(self, ctx, ch, take):
        mode = self.mode
        s = ch.span()
        if cutlass.const_expr(mode == "before_begin"):
            ch = take(ch)
        for i in cutlass.range(self.trips):
            s.begin(i)
            if cutlass.const_expr(mode == "after_loop"):
                pass
            elif cutlass.const_expr(self.producer):
                s.commit(i)
            else:
                s.release(i)
            if cutlass.const_expr(mode == "last_trip"):
                ch = take(ch)
            elif cutlass.const_expr(mode == "branch"):  # runs after the last-trip signal
                if i == self.trips - 1:
                    ch = take(ch)
        if cutlass.const_expr(mode == "after_loop"):
            if cutlass.const_expr(self.producer):
                s.commit()
            else:
                s.release()
            ch = take(ch)
        return ch

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        if cutlass.const_expr(self.producer):
            return self.spanned(ctx, ch, _acquired)
        for _ in cutlass.range_constexpr(self.epochs):
            with ch.produce():
                pass
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        if cutlass.const_expr(not self.producer):
            return self.spanned(ctx, ch, _waited)
        for _ in cutlass.range_constexpr(self.epochs):
            with ch.consume():
                pass
        return ch


@pytest.mark.parametrize(
    "make, key, error",
    [
        (
            lambda: RoundRobinHold(2),
            "rr",
            r"ch\.wait\(\) at .* takes an epoch at least 2 after.* at least 4 stages",
        ),
        (lambda: SpanLifetime("before_begin"), "span_early", r"after ch\[i\] span opened at"),
        (lambda: SpanLifetime("last_trip"), "span_last_trip", r"until release in the last trip"),
        (lambda: SpanLifetime("last_trip", True), "span_last_p", r"until commit in the last trip"),
    ],
)
def test_holds_of_other_shapes_raise(make, key, error):
    raises_protocol(error, lambda: compile_toy(make(), f"hold2_{key}"))


@pytest.mark.parametrize(
    "make, key",
    [
        (lambda: RoundRobinHold(4), "rr"),  # each instance's next epoch is two stages on
        (lambda: HoldAndWait("lap_in_loop", 3, 3), "lap_in_loop"),  # a later trip: not checked
        (lambda: Arms("other_slot"), "other_slot"),
        (ExternalHold, "external"),  # raw code may never wait for the hand-back
        (DynamicSkipHold, "dynamic_skip"),  # skip(0) keeps the next wait one stage on
        # A span handed back after its loop, in its loop's only trip, or before a wait that
        # runs only after the last trip's signal.
        *[
            (lambda m=mode, p=producer, t=trips: SpanLifetime(m, p, t), f"span_{mode}{producer:d}")
            for mode, trips in (("after_loop", 2), ("last_trip", 1), ("branch", 2))
            for producer in (False, True)
        ],
    ],
)
def test_holds_that_may_be_free_compile(make, key):
    compile_toy(make(), f"free2_{key}")


def test_a_handle_held_past_its_arm_is_a_lifetime_error_not_a_deadlock():
    """The other arm's waits never run with the hold; the leak is what is wrong."""
    raises_protocol(
        r"handles never closed: ch\[i\] wait", lambda: compile_toy(Arms("arms"), "arms")
    )


def test_a_dynamic_slot_still_names_the_slot_rule():
    raises_protocol(r"needs a Python int slot", lambda: compile_toy(Arms("dynamic_slot"), "dyn"))


class Graph(Toy):
    """``mode`` selects a protocol over ``a`` (p -> q) and ``b`` (q -> p)."""

    def __init__(self, mode: str, liveness_check: str = "strict"):
        self.mode, self.liveness_check = mode, liveness_check
        self.a = Ring(1, Threads, Threads)
        self.b = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer, b: Consumer):
        mode = self.mode
        if cutlass.const_expr(mode == "cross"):  # each waits for what the other sends after
            with b.consume(), a.produce():
                pass
        elif cutlass.const_expr(mode in ("static", "unchecked", "state", "unbalanced")):
            with a.produce():
                pass
            with b.consume():
                pass
            if cutlass.const_expr(mode == "unbalanced"):
                with a.produce():
                    pass
        else:
            for _ in cutlass.range(ctx.n, unroll=1):
                with a.produce():
                    pass
                with b.consume():
                    pass
            if cutlass.const_expr(mode == "unbalanced_n"):  # one fill more than the n taken
                with a.produce():
                    pass
        return a, b

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer, b: Producer):
        mode = self.mode
        if cutlass.const_expr(mode in ("cross", "static", "unchecked", "state", "unbalanced")):
            if cutlass.const_expr(mode == "unchecked"):
                with unchecked("liveness", reason="raw code orders this handoff"), a.consume():
                    pass
            else:
                with a.consume():
                    pass
            if cutlass.const_expr(mode == "state"):
                b.state = b.state
            with b.produce():
                pass
        else:
            for _ in cutlass.range(ctx.n, unroll=1):
                if cutlass.const_expr(mode == "lane"):
                    if cute.arch.lane_idx() == 0:
                        with a.consume():
                            pass
                else:
                    with a.consume():
                        pass
                with b.produce():
                    pass
        return a, b


def test_straight_line_cross_wait_reports_sites_and_override():
    raises_protocol(
        r"deadlock:.*test_warpdrive_deadlock.*Override with unchecked",
        lambda: compile_toy(Graph("cross"), "small_cross"),
    )


@pytest.mark.parametrize(
    "mode, prefix",
    [
        ("static", "liveness: verified"),
        ("loop", "liveness: unverified (runtime loop/branch or unknown region: p at test_"),
        ("unbalanced_n", "liveness: unverified (runtime loop/branch"),
        ("lane", "liveness: unverified (runtime loop/branch"),
        ("unchecked", 'liveness: unverified (unchecked("liveness")'),
        ("state", "liveness: unverified (b.state"),
    ],
)
def test_extracted_coverage_is_explicit(mode, prefix):
    op = Graph(mode)
    compile_toy(op, f"small_{mode}")
    assert prefix in op.describe()
    assert op.checks().liveness == prefix.split(": ")[1].split(" (")[0]
    restored = Graph(mode)
    compile_toy(restored, f"small_{mode}")
    assert restored.describe() == op.describe()


def test_static_imbalance_is_an_error():
    raises_protocol("unbalanced epochs", lambda: compile_toy(Graph("unbalanced"), "imbalance"))


def test_warn_and_off_modes():
    op = Graph("cross", "warn")
    with pytest.warns(UserWarning, match="deadlock"):
        compile_toy(op, "small_warn")
    assert op.checks().liveness == "error"
    assert "liveness: error (see the reported finding)" in op.describe()
    op = Graph("cross", "off")
    compile_toy(op, "small_off")
    assert op.checks().liveness is None


def _publish(end):  # advances ``end`` in place and returns nothing
    end.acquire().commit()


class NamedJoin(Toy):
    def __init__(self):
        self.a = Ring(1, Threads, Threads, sync="named")
        self.b = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p1(self, ctx, *, a: Producer, b: Producer):
        a.acquire().commit()
        b.acquire().commit()
        return a, b

    @role(warps=1)
    @cute.jit
    def p2(self, ctx, *, a: Producer, b: Consumer):
        b.wait().release()
        a.acquire().commit()
        return a, b

    @role(warps=2)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        a.wait().release()
        return a


class NamedCommit(Toy):
    def __init__(self, absent=False):
        self.absent = absent
        self.a = Ring(1, Threads, Threads, sync="named")
        self.b = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer, b: Producer):
        a.acquire().commit(wait=True)
        b.acquire().commit()
        return a, b

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer, b: Consumer):
        b.wait().release()
        if cutlass.const_expr(not self.absent):
            a.wait().release()
        return a, b


@pytest.mark.parametrize(
    "make, finding",
    [
        (NamedJoin, "deadlock"),
        (NamedCommit, "deadlock"),
        (lambda: NamedCommit(True), "unbalanced epochs"),
    ],
)
def test_named_channel_extraction(make, finding):
    raises_protocol(finding, lambda: compile_toy(make(), "named_extraction"))


class RawBarrier(Toy):
    def __init__(self, mode):
        self.mode = mode
        self.a = Ring(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        a.acquire().commit()
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        h = a.wait()
        if cutlass.const_expr(self.mode == "handle"):
            ptr = h.empty_barrier
        elif cutlass.const_expr(self.mode == "kernel"):
            ptr = self.barrier_ptr("a")
        else:
            ptr = a.full_barrier(1)
        h.release()
        cute.arch.mbarrier_wait(ptr, 0)
        return a


@pytest.mark.parametrize("mode", ["end", "handle", "kernel"])
def test_raw_barrier_pointer_extraction_is_unverified(mode):
    op = RawBarrier(mode)
    compile_toy(op, f"raw_barrier_{mode}")
    assert op.checks().liveness == "unverified"
    assert "barrier" in op._ws_liveness.detail


class InPlaceHelper(Toy):
    """Without reassignment, the loop reuses the end's initial stage and phase every trip."""

    def __init__(self):
        self.ch = Ring(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            _publish(ch)
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume():
                pass
        return ch


def test_an_end_advanced_where_no_region_carries_it_raises():
    raises_protocol(
        r"prod: the ch end advances at .* inside a loop or branch that does not carry it",
        lambda: compile_toy(InPlaceHelper(), "inplace_helper"),
    )
