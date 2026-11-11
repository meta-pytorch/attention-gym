"""Test spans: a stage held across the iterations of a counted loop.

Cover trace errors, accepted forms, the alias diagnostic's epoch matching through a span, and
the runtime wait in the first iteration and signal in the last (or after the loop).
"""

import re
import warnings
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.testing.warpdrive import alias_section, status
from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    Producer,
    Ring,
    Sharing,
    Slots,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor

N = 32  # payload words per stage: one per lane
TRIPS = 3
DELAY = 200_000  # cycles


@cute.jit
def spin(cycles):
    start = cute.arch.clock64()
    while cute.arch.clock64() - start < cycles:
        pass


class Toy(WarpSpecialized):
    """Roles get ``ctx = (out, n, delay)``; one CTA."""

    label = "toy"

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32, delay: cutlass.Int64):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n, delay=delay))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, delay: cutlass.Int64, stream):
        self.kernel.set_name_prefix(f"warpdrive_spans_{self.label}")
        self.kernel(out, n, delay).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_toy(op, key: str):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    name = f"warpdrive_spans_{key}"
    return compile_tvm_ffi(op, out, cutlass.Int32(1), cutlass.Int64(0), name=name)


# Trace-time misuse


class Misuse(Toy):
    """``mode`` selects one misuse of a span of ``ch`` in the producer."""

    def __init__(self, mode: str):
        self.mode = mode
        self.ch = Ring(2, Threads, Threads)
        self.named = Ring(2, Threads, Threads, sync="named")

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer, named: Producer):
        mode = self.mode
        if cutlass.const_expr(mode == "named"):
            named.span()
        if cutlass.const_expr(mode == "outer_region"):
            outer = ch.span()  # one epoch taken per launch, waited for once per item
        for it in cutlass.range(ctx.n, unroll=1):
            with named.produce():
                pass
            if cutlass.const_expr(mode == "outer_region"):
                for v in cutlass.range(TRIPS):
                    outer.begin(v)
                    outer.commit(v)
            elif cutlass.const_expr(mode == "never_begun"):
                ch.span()
            elif cutlass.const_expr(mode == "constexpr"):
                s = ch.span()
                for v in cutlass.range_constexpr(TRIPS):
                    s.begin(v)
                    s.commit(v)
            elif cutlass.const_expr(mode == "dynamic_bound"):
                s = ch.span()
                for v in cutlass.range(ctx.n):
                    s.begin(v)
                    s.commit(v)
            elif cutlass.const_expr(mode == "start"):
                s = ch.span()
                for v in cutlass.range(1, TRIPS):
                    s.begin(v)
                    s.commit(v)
            elif cutlass.const_expr(mode == "step"):
                s = ch.span()
                for v in cutlass.range(0, TRIPS, 2):
                    s.begin(v)
                    s.commit(v)
            elif cutlass.const_expr(mode == "branch"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    if ctx.n > 1:
                        s.begin(v)
                    s.commit(v)
            elif cutlass.const_expr(mode == "nested"):
                s = ch.span()
                for _ in cutlass.range(TRIPS):
                    for v in cutlass.range(TRIPS):
                        s.begin(v)
                        s.commit(v)
            elif cutlass.const_expr(mode == "other_index"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(it)
                    s.commit(v)
            elif cutlass.const_expr(mode == "derived_index"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v + 0)
                    s.commit(v)
            elif cutlass.const_expr(mode == "begin_twice"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                    s.begin(v)
                    s.commit(v)
            elif cutlass.const_expr(mode == "commit_first"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.commit(v)
                    s.begin(v)
            elif cutlass.const_expr(mode == "commit_without_begin"):
                s = ch.span()
                s.commit()
            elif cutlass.const_expr(mode == "unclosed"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
            elif cutlass.const_expr(mode == "closed_twice"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                    s.commit(v)
                s.commit()
            elif cutlass.const_expr(mode == "other_loop"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                for v in cutlass.range(TRIPS):
                    s.commit(v)
            elif cutlass.const_expr(mode == "unguarded_in_loop"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                    s.commit()
            elif cutlass.const_expr(mode == "after_in_branch"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                if ctx.n > 1:
                    s.commit()
            elif cutlass.const_expr(mode == "retire"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                s.retire()
            elif cutlass.const_expr(mode == "release"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                    s.release(v)
            elif cutlass.const_expr(mode == "positional_signal"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                s.commit(False)
        return ch, named

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer, named: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(), named.consume():
                pass
        return ch, named


class TmaSpan(Toy):
    def __init__(self):
        self.tma = Ring(2, Tma(64), Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, tma: Producer):
        tma.span()
        return tma

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, tma: Consumer):
        with tma.consume():
            pass
        return tma


COUNTED = r"must run directly in the body of a `for i in cutlass\.range\(n\)` loop"
CROSS = r"is in a different dynamic region .*inside the loop of begin\(\) pass the loop index"


@pytest.mark.parametrize(
    "mode, error",
    [
        ("named", r"named\.span\(\) at .*: a named-barrier channel cannot hold a span"),
        ("outer_region", rf"ch\[i\] span opened at .*: begin\(i\) at .* {COUNTED}"),
        ("never_begun", r"ch\[i\] span opened at .* took an epoch but never waited for it"),
        ("constexpr", rf"begin\(i\) at .* {COUNTED}"),
        ("dynamic_bound", rf"begin\(i\) at .* {COUNTED}"),
        ("start", rf"begin\(i\) at .* {COUNTED}"),
        ("step", rf"begin\(i\) at .* {COUNTED}"),
        ("branch", rf"begin\(i\) at .* {COUNTED}"),
        ("nested", rf"begin\(i\) at .* {COUNTED}"),
        ("other_index", r"begin\(i\) at .*: `i` must be that loop's index"),
        ("derived_index", r"begin\(i\) at .*: `i` must be that loop's index"),
        ("begin_twice", r"begin\(\) at .* after begin\(\) at .*; a span waits once"),
        ("commit_first", r"commit\(\) at .* before begin\(i\) waited for the stage"),
        ("commit_without_begin", r"commit\(\) at .* before begin\(i\) waited for the stage"),
        ("unclosed", r"handles never closed: ch\[i\] span opened at"),
        ("closed_twice", r"ch\[i\] span closed twice: commit at .*, then commit at"),
        ("other_loop", r"commit\(i\) at .* is in another loop than begin\(\) at"),
        ("unguarded_in_loop", rf"span opened at .* but commit at .* {CROSS}"),
        ("after_in_branch", rf"span opened at .* but commit at .* {CROSS}"),
        ("retire", r"retire\(\) at .*: a span cannot be retired"),
        ("release", r"release\(\) on a ch\[i\] span handle; producer handles commit"),
        ("positional_signal", r"commit\(False\) at .*: a span's commit\(\) takes the loop index"),
    ],
)
def test_misuse_raises_while_tracing(mode, error):
    _raises_protocol(error, lambda: compile_toy(Misuse(mode), f"misuse_{mode}"))


def test_tma_producer_cannot_hold_a_span():
    error = r"tma\.span\(\) at .*: a Tma producer cannot hold a span"
    _raises_protocol(error, lambda: compile_toy(TmaSpan(), "misuse_tma"))


class Accepted(Toy):
    """Valid spans beyond the runtime toy's (compiled only)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.ch = Ring(2, Threads, Threads)
        self.slots = Slots(2, Threads, Threads)
        self.rr = Ring(2, Threads, Threads, consumer_sharing=Sharing.ROUND_ROBIN)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer, slots: Producer, rr: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            if cutlass.const_expr(self.mode == "with"):
                with ch.span() as s:  # the block's exit commits after the loop
                    for v in cutlass.range(TRIPS):
                        s.begin(v)
            else:
                s = ch.span()
                for v in cutlass.range(TRIPS, unroll=1):
                    s.begin(v)
                    s.commit(v, signal=False)
                    if v == TRIPS - 1:
                        cute.arch.mbarrier_arrive(s.full_barrier)
            first, second = slots.span(0), slots.span(1)  # overlapping spans of one loop
            for v in cutlass.range(TRIPS):
                first.begin(v)
                second.begin(v)
                first.commit(v)
            second.commit()
        for _ in cutlass.range_constexpr(2):
            with rr.produce():
                pass
        return ch, slots, rr

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def cons(self, ctx, *, ch: Consumer, slots: Consumer, rr: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            ch.peek()  # the span owns the previewed epoch
            s = ch.span()
            for v in cutlass.range(TRIPS):
                s.begin(v)
                s.release(v)
            for k in cutlass.range_constexpr(2):
                with slots.consume(k):
                    pass
        r = rr.span()  # this instance's own epoch of the round-robin ring
        for v in cutlass.range(TRIPS):
            r.begin(v)
        r.release()
        return ch, slots, rr


@pytest.mark.parametrize("mode", ["with", "raw_signal"])
def test_valid_uses_compile(mode):
    op = Accepted(mode)
    compile_toy(op, f"accepted_{mode}")
    text = op.describe()
    # A span takes one epoch where it opens, so the round-robin ownership check stays exact.
    assert re.search(r"rr .*\n(.*\n)*? *epochs +verified", text), text
    assert "unchecked" not in text, text


# Alias diagnostic: a span's wait and signal run once per pass of its loop


class SpanRelay(Toy):
    """The MMA warp accumulates into S under a span of ``c``; the softmax warp, woken by a relay
    that holds ``c``'s stage meanwhile, reads S and writes P (inside S). The next accumulation
    is ordered after that P write only through ``c``'s release -> the span's wait, and each
    S write precedes the softmax only through the span's commit."""

    def __init__(self, close: str):
        self.close = close
        self.label = f"relay_{close}"
        self.c = Slots(1, Umma, Threads)
        self.wake = Ring(1, Threads, Threads)
        self.done = Ring(1, Threads, Threads, commit_fence="tmem_store")
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, c: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            s = c.span(0)
            for v in cutlass.range(TRIPS):
                s.begin(v)
                tmem.access("S", write=True, by="umma")
                if cutlass.const_expr(self.close == "loop"):
                    s.commit(v)
            if cutlass.const_expr(self.close == "after"):
                s.commit()
        return c

    @role(warps=1)
    @cute.jit
    def relay(self, ctx, *, c: Consumer, wake: Producer, done: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            h = c.wait(0)
            with wake.produce():
                pass
            with done.consume():
                pass
            h.release()
        return c, wake, done

    @role(warps=2)
    @cute.jit
    def softmax(self, ctx, *, wake: Consumer, done: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with wake.consume():
                pass
            tmem.access("S", read=True)
            tmem.access("P", write=True)
            with done.produce():
                pass
        return wake, done


@pytest.mark.parametrize("close", ["loop", "after"])
def test_alias_diagnostic_matches_a_span_to_one_epoch(close):
    """Counting the span's wait or commit once per traced iteration would leave the epochs of
    ``c`` unmatched and the pair unverified."""
    op = SpanRelay(close)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile_toy(op, op.label)
    assert status(op, "P ~ S") == "no hazard found", alias_section(op)


# Runtime


class SpanToy(Toy):
    """The producer accumulates ``TRIPS`` addends into a stage, one per iteration of a counted
    loop, under a span committed in the last iteration (``close="loop"``) or after the loop;
    the consumer checks each item's total under a handle or, read ``TRIPS`` times, under a
    span. ``late``: the role that spins in every iteration, so the other one's wait decides.

    ``out``: wrong words per consumer lane."""

    def __init__(self, stages: int, close: str, consumer: str, late: str):
        self.close, self.consumer, self.late = close, consumer, late
        self.label = f"toy_{stages}_{close}_{consumer}_{late}"
        self.ch = Ring(stages, Threads, Threads, payload=(cutlass.Int32, (N,)))

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        lane = cute.arch.lane_idx()
        for it in cutlass.range(ctx.n, unroll=1):
            s = ch.span()
            for v in cutlass.range(TRIPS):
                s.begin(v)
                if cutlass.const_expr(self.late == "producer"):
                    spin(ctx.delay)
                word = cutlass.Int32(0)
                if v != 0:
                    word = s.data[lane]
                s.data[lane] = word + it * N + lane + v
                if cutlass.const_expr(self.close == "loop"):
                    s.commit(v)
            if cutlass.const_expr(self.close == "after"):
                s.commit()
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        lane = cute.arch.lane_idx()
        bad = cutlass.Int32(0)
        for it in cutlass.range(ctx.n, unroll=1):
            total = TRIPS * (it * N + lane) + TRIPS * (TRIPS - 1) // 2
            if cutlass.const_expr(self.consumer == "span"):
                s = ch.span()
                for v in cutlass.range(TRIPS):
                    s.begin(v)
                    if cutlass.const_expr(self.late == "consumer"):
                        spin(ctx.delay)
                    bad = bad + (s.data[lane] != total)
                    s.release(v)
            else:
                with ch.consume() as h:
                    if cutlass.const_expr(self.late == "consumer"):
                        spin(ctx.delay)
                    bad = bad + (h.data[lane] != total)
        ctx.out[lane] = bad
        return ch


@pytest.mark.parametrize(
    "stages, close, consumer",
    [(1, "loop", "handle"), (2, "loop", "span"), (2, "after", "handle"), (1, "after", "span")],
)
@pytest.mark.parametrize("late", ["producer", "consumer"])
def test_span_waits_in_the_first_iteration_and_signals_in_the_last(
    requires_blackwell, stages, close, consumer, late
):
    n = 7  # an odd count: every stage is reused on both parities
    op = SpanToy(stages, close, consumer, late)
    fn = compile_toy(op, op.label)
    for _ in range(3):
        out = torch.full((N,), -1, device="cuda", dtype=torch.int32)
        fn(out, n, DELAY)
        torch.cuda.synchronize()
        assert out.tolist() == [0] * N
