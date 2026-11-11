"""Trace-time deadlock checks: a role waiting for an epoch that needs a stage it still holds."""

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith

from attn_gym.testing.warpdrive import raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    Instance,
    NamedBarrier,
    Producer,
    Ring,
    Sharing,
    Slots,
    StaticPersistent,
    Threads,
    Tma,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    events,
    role,
    unchecked,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.expr import Reader


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


# The event graph: deadlock and balance across roles


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


def test_a_dependency_cycle_names_its_sites_and_the_override():
    raises_protocol(
        r"Graph: deadlock: p: b\.wait\(\) of epoch 0 at test_warpdrive_deadlock\.py:\d+ waits "
        r"for q: b\.commit\(\) of epoch 0 at .* waits for q: a\.wait\(\) of epoch 0 at .* waits "
        r"for p: a\.commit\(\) .*\. Override with unchecked\(\"liveness\", reason=\.\.\.\) around "
        r"the ops, or liveness_check=\"warn\" or \"off\" on the kernel\.",
        lambda: compile_toy(Graph("cross"), "graph_cross"),
    )


@pytest.mark.parametrize(
    "mode, error",
    [
        ("unbalanced", r"unbalanced epochs: q takes \d+ epochs of a, \d+ filled"),
        ("lane", r"q: a branch that depends on the lane encloses channel or barrier ops"),
    ],
)
def test_epochs_and_participants_must_balance(mode, error):
    raises_protocol(error, lambda: compile_toy(Graph(mode), f"graph_{mode}"))


@pytest.mark.parametrize(
    "mode, line",
    [
        ("static", "liveness: verified"),
        ("loop", "liveness: no deadlock found (assumes loop bounds and conditions of 1 runtime"),
        ("unchecked", 'liveness: unverified (unchecked("liveness") at test_warpdrive_deadlock'),
        ("state", "liveness: unverified (q reads or assigns b.state at test_warpdrive_deadlock"),
    ],
)
def test_the_verdict_says_what_it_proves(mode, line):
    op = Graph(mode)
    compile_toy(op, f"graph_ok_{mode}")
    assert line in op.describe()
    assert op.checks().liveness == line.split(": ")[1].split(" (")[0]


def test_warn_reports_and_off_skips_a_finding():
    warned = Graph("cross", liveness_check="warn")
    with pytest.warns(UserWarning, match="warpdrive: Graph: deadlock"):
        compile_toy(warned, "graph_warn")
    off = Graph("cross", liveness_check="off")
    compile_toy(off, "graph_off")
    assert off.checks().liveness is None


def test_an_analysis_failure_is_unverified_not_an_error(monkeypatch):
    from attn_gym.warpdrive import liveness

    def broken(*args, **kwargs):
        raise RuntimeError("unexpected IR")

    monkeypatch.setattr(liveness, "DEBUG", False)
    monkeypatch.setattr(liveness, "unroll", broken)
    op = Graph("static")
    compile_toy.__wrapped__(op, "graph_broken")
    assert "liveness: unverified (analysis failed: RuntimeError: unexpected IR)" in op.describe()


def test_the_graph_finds_fa4_backward_with_one_q_stage(monkeypatch):
    """Without the hold check, the cycle through load's refill of Q still rejects it."""
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive import channels
    from attn_gym.warpdrive.examples import fa4_bwd

    monkeypatch.setattr(fa4_bwd, "Q_STAGE", 1)
    monkeypatch.setattr(channels._End, "_check_hold", lambda self, *args, **kwargs: None)
    op = fa4_bwd.FlashAttentionBackward(causal=False)
    raises_protocol(
        r"deadlock \(with runtime values \[2\]\): .*mma: q\.peek\(\) of epoch 1 at fa4_bwd\.py:\d+ "
        r"waits for load: q\.acquire\(\) of epoch 1 at fa4_bwd\.py:\d+ waits for mma: "
        r"q\.release\(\) of epoch 0",
        lambda: fa4_bwd.compile_fa4_bwd.__wrapped__(op),
    )


# Shapes that once read as findings but cannot hang (and one that can).


class Quotient(Toy):
    def __init__(self):
        self.a = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        for i in cutlass.range(ctx.n, unroll=1):
            with a.produce():
                pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        for i in cutlass.range(ctx.n // 2, unroll=1):
            with a.consume():
                pass
            with a.consume():
                pass
        for i in cutlass.range(ctx.n % 2, unroll=1):
            with a.consume():
                pass
        return a


class Rotating(Toy):
    def __init__(self):
        self.a = Ring(2, Threads, Threads, consumer_sharing=Sharing.ROUND_ROBIN)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        for i in cutlass.range(4, unroll=1):
            with a.produce():
                pass
        return a

    @role(warps=[range(1, 3), range(3, 5)])
    @cute.jit
    def q(self, ctx, *, a: Consumer, stage: Instance):
        for i in cutlass.range(stage, 4, 2, unroll=1):
            if i >= 0:
                with a.consume():
                    pass
        return a


class NamedCommit(Toy):
    def __init__(self):
        self.a = Ring(1, Threads, Threads, sync="named")
        self.b = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer, b: Producer):
        a.acquire().commit(wait=True)
        with b.produce():
            pass
        return a, b

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer, b: Consumer):
        with b.consume():
            pass
        with a.consume():
            pass
        return a, b


class RawSlots(Toy):
    def __init__(self):
        self.a = Slots(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        with a.produce(0):
            pass
        with a.produce(0):
            pass
        with a.produce(1):
            pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        h = a.wait(0)
        cute.arch.mbarrier_arrive(h.empty_barrier)
        j = a.wait(1)
        h.release(signal=False)
        j.release()
        with a.consume(0):
            pass
        return a


class LaneProbe(Toy):
    def __init__(self):
        self.a = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        with a.produce():
            pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        if cute.arch.lane_idx() == 0:
            a.try_wait()
        with a.consume():
            pass
        return a


class Unsigned(Toy):
    def __init__(self):
        self.a = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        for i in cutlass.range(2, unroll=1):
            if cutlass.Uint32(i - 1) < cutlass.Uint32(1):
                with a.produce():
                    pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        with a.consume():
            pass
        return a


@pytest.mark.parametrize(
    "make",
    [Quotient, Rotating, RawSlots, LaneProbe, Unsigned],
    ids=["related_counts", "warp_split_round_robin", "raw_slot_release", "lane_probe", "unsigned"],
)
def test_runnable_shapes_are_not_findings(make):
    """n // 2 and n % 2 trips, a round-robin role split by warp, a raw release of one slot, a
    probe by one lane and an unsigned loop test all complete when run."""
    op = make()
    compile_toy(op, f"runnable_{make.__name__.lower()}")
    assert "liveness: error" not in op.describe()


def test_a_waiting_commit_is_a_rendezvous():
    """commit(wait=True) waits for the consumer, which first waits for b, published after it."""
    raises_protocol(
        r"deadlock: .*a\.commit\(wait=True\)",
        lambda: compile_toy(NamedCommit(), "named_commit"),
    )


class Related(Toy):
    """The producer fills ``n`` epochs and the consumer takes 4: balanced for a caller that
    passes ``n == 4``, a value the samples miss."""

    def __init__(self):
        self.a = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with a.produce():
                pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        for _ in cutlass.range_constexpr(4):
            with a.consume():
                pass
        return a


class Casts(Toy):
    """Integer semantics the device has: the zero extension of -1 is not below 1, and -1 // 2
    floors to -1, so each loop produces in one trip."""

    def __init__(self):
        self.a = Ring(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        for i in cutlass.range(2, unroll=1):
            if cutlass.Int64(cutlass.Uint32(i - 1)) < 1:
                with a.produce():
                    pass
        for i in cutlass.range(2, unroll=1):
            if cutlass.Int32(i - 1) // 2 == -1:
                with a.produce():
                    pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        for _ in cutlass.range_constexpr(2):
            with a.consume():
                pass
        return a


class ElectedLane(Toy):
    """Lane 0 fills an Umma stage: its commit is one elected arrive, which the branch keeps."""

    def __init__(self):
        self.a = Ring(1, Umma, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        if cute.arch.lane_idx() == 0:
            with a.produce():
                pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        with a.consume():
            pass
        return a


class LaneSkip(Toy):
    """r takes epoch 0; lane 0 of q skips it: per-lane channel state, not an arrival."""

    def __init__(self):
        self.a = Ring(
            2, Threads, Threads, consumer_sharing=Sharing.EXCLUSIVE, consumer_roles=("q", "r")
        )

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        for _ in cutlass.range_constexpr(2):
            with a.produce():
                pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        if cute.arch.lane_idx() == 0:
            a.skip(1)
        with a.consume():
            pass
        return a

    @role(warps=2)
    @cute.jit
    def r(self, ctx, *, a: Consumer):
        with a.consume():
            pass
        return a


class LaneBarrierId(Toy):
    """Lane 0 reads a named barrier's id, which may feed raw code the check does not see."""

    def __init__(self):
        self.x = NamedBarrier(threads=64)

    @role(warps=0)
    @cute.jit
    def a(self, ctx):
        if cute.arch.lane_idx() == 0:
            ctx.out[0] = self.x.id
        self.x.sync()

    @role(warps=1)
    @cute.jit
    def b(self, ctx):
        self.x.sync()


class PerItem(WarpSpecialized):
    """One epoch filled per work item, four taken: balanced for a CTA that runs four items."""

    def __init__(self):
        self.a = Ring(1, Threads, Threads)
        self.work = WorkStream(schedule=StaticPersistent())

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer, work: WorkCursor):
        while work.valid:
            with a.produce():
                pass
            work.advance()
        return a, work

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer, work: WorkCursor):
        for _ in cutlass.range_constexpr(4):
            with a.consume():
                pass
        while work.valid:
            work.advance()
        return a, work

    @cute.kernel
    def kernel(self, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace())

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream):
        bound = self.work.bind((n, 1, 1), max_ctas=cutlass.Int32(1))
        self.kernel.set_name_prefix("warpdrive_deadlock_per_item")
        self.kernel(bound).launch(**self.launch_config(bound), stream=stream)


class Overridden(Toy):
    """The full barrier expects 64 arrivals from a 32-thread producer."""

    def __init__(self):
        self.a = Ring(1, Threads, Threads, override={"full_counts": 64})

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        with a.produce():
            pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        with a.consume():
            pass
        return a


class Long(Toy):
    def __init__(self):
        self.a = Ring(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        for _ in cutlass.range(20_000, unroll=1):
            with a.produce():
                pass
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        for _ in cutlass.range(20_000, unroll=1):
            with a.consume():
                pass
        return a


@pytest.mark.parametrize(
    ("make", "why"),
    [
        (Related, "no checked value balances"),
        (ElectedLane, "encloses an elected signal"),
        (LaneSkip, "per-lane channel state"),
        (LaneBarrierId, "per-lane channel state"),
        (PerItem, "no checked value balances"),
        (Overridden, "a overrides its full_counts"),
        (Long, "p runs over"),
    ],
    ids=[
        "sampled_counts",
        "elected_lane",
        "lane_skip",
        "lane_barrier_id",
        "per_item",
        "overridden_counts",
        "long",
    ],
)
def test_what_the_model_cannot_decide_is_unverified(make, why):
    op = make()
    compile_toy(op, f"unverified_{make.__name__.lower()}")
    line = next(x for x in op.describe().splitlines() if x.startswith("liveness:"))
    assert line.startswith("liveness: unverified (") and why in line


def test_integer_ops_keep_their_device_semantics():
    op = Casts()
    compile_toy(op, "casts")
    assert "liveness: verified" in op.describe()


@pytest.mark.parametrize(
    ("expr", "value"),
    [
        (("div", ("c", -1), ("c", 2)), 0),  # arith.divsi truncates
        (("floordiv", ("c", -1), ("c", 2)), -1),
        (("rem", ("c", -1), ("c", 2)), -1),
    ],
)
def test_integer_expressions_evaluate_as_on_the_device(expr, value):
    assert events.evaluate(expr, {}) == value


def test_casts_read_from_mlir_keep_their_width():
    """Narrowing index casts wrap, and an i1 true sign-extends to -1."""
    with ir.Context(), ir.Location.unknown(), ir.InsertionPoint(ir.Module.create().body):
        i32, i1, index = (
            ir.IntegerType.get_signless(32),
            ir.IntegerType.get_signless(1),
            ir.IndexType.get(),
        )
        big = arith.constant(index, 2**32)
        values = {
            arith.index_castui(i32, big): 0,
            arith.index_cast(i32, big): 0,
            arith.extsi(i32, arith.trunci(i1, arith.constant(i32, 3))): -1,
            arith.extui(ir.IntegerType.get_signless(64), arith.constant(i32, -1)): 2**32 - 1,
        }
        reader = Reader(lambda block: None, [])
        for value, device in values.items():
            assert events.evaluate(reader.read(value), {}) == device


class Absent(Toy):
    """commit(wait=True) syncs with a consumer that never takes the epoch."""

    def __init__(self):
        self.a = Ring(1, Threads, Threads, sync="named")

    @role(warps=0)
    @cute.jit
    def p(self, ctx, *, a: Producer):
        a.acquire().commit(wait=True)
        return a

    @role(warps=1)
    @cute.jit
    def q(self, ctx, *, a: Consumer):
        return a


def test_a_rendezvous_needs_every_consumer():
    raises_protocol(
        r"deadlock: p: a\.commit\(wait=True\) .* waits forever: 0 of 1 consumers reach epoch 0",
        lambda: compile_toy(Absent(), "absent_consumer"),
    )


def test_a_failed_region_read_is_unverified(monkeypatch):
    def fail(*args):
        raise RuntimeError("boom")

    monkeypatch.setattr(events, "DEBUG", False)
    monkeypatch.setattr(events, "_regions", fail)
    op = Quotient()
    compile_toy.__wrapped__(op, "failed_read")
    assert "liveness: unverified (reading the loops of p failed: RuntimeError('boom'))" in (
        op.describe()
    )


def _publish(end):  # advances ``end`` in place and returns nothing
    end.acquire().commit()


class InPlaceHelper(Toy):
    """The producer's loop calls a helper that acquires without reassigning the end, so the
    loop does not carry the end and every trip would wait on the first trip's parity."""

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
