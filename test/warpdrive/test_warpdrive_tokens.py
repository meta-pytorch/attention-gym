"""Test try_wait/try_acquire tokens and peek previews.

Cover trace errors, preview-read alias attribution, both runtime probe outcomes,
and previews across ring wraps.
"""

import re
import warnings
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.testing.warpdrive import alias_section
from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    Producer,
    Ring,
    Sharing,
    Slots,
    SmemPlan,
    Threads,
    Token,
    Umma,
    WarpSpecialized,
    role,
    unchecked,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor

N = 32  # payload words per stage: one per consumer lane
DELAY = 200_000  # cycles: far longer than one try_wait probe may suspend


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
        self.kernel.set_name_prefix(f"warpdrive_tokens_{self.label}")
        self.kernel(out, n, delay).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_toy(op, key: str):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    name = f"warpdrive_tokens_{key}"
    return compile_tvm_ffi(op, out, cutlass.Int32(1), cutlass.Int64(0), name=name)


# Trace-time misuse


def look(preview):
    return preview.view()


class Misuse(Toy):
    """``mode`` selects one misuse in the consumer of ``ch``; everything else is valid."""

    def __init__(self, mode: str):
        self.mode = mode
        self.ch = Ring(2, Threads, Threads, payload=(cutlass.Int32, (N,)))
        self.other = Ring(2, Threads, Threads)
        self.back = Ring(2, Threads, Threads)
        self.slots = Slots(2, Threads, Threads)
        self.named = Ring(2, Threads, Threads, sync="named")

    @role(warps=0)
    @cute.jit
    def prod(
        self,
        ctx,
        *,
        ch: Producer,
        other: Producer,
        back: Consumer,
        slots: Producer,
        named: Producer,
    ):
        if cutlass.const_expr(self.mode == "named_try_acquire"):
            named.try_acquire()
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.produce(), other.produce(), back.consume(), named.produce():
                pass
            for s in cutlass.range_constexpr(2):
                with slots.produce(s):
                    pass
        return ch, other, back, slots, named

    @role(warps=1)
    @cute.jit
    def cons(
        self,
        ctx,
        *,
        ch: Consumer,
        other: Consumer,
        back: Producer,
        slots: Consumer,
        named: Consumer,
    ):
        mode = self.mode
        if cutlass.const_expr(mode == "region"):
            token = ch.try_wait()
            for _ in cutlass.range(ctx.n, unroll=1):
                ch.wait(token=token).release()
        elif cutlass.const_expr(mode.startswith("carry_")):
            token = ch.try_wait()
            if cutlass.const_expr(mode == "carry_entry"):
                ch.wait().release()  # the token entering the loop names a consumed epoch
            for _ in cutlass.range(ctx.n, unroll=1):
                if cutlass.const_expr(mode == "carry_not_first"):
                    ch.wait().release()
                ch.wait(token=token).release()
                token = ch.try_wait()
                if cutlass.const_expr(mode == "carry_back_edge"):
                    ch.wait().release()  # the token passed on names a consumed epoch
                if cutlass.const_expr(mode == "carry_branch"):
                    if ctx.n > 1:  # the token passed on depends on the branch taken
                        ch.wait(token=token).release()
                        token = ch.try_wait()
        elif cutlass.const_expr(mode == "branch_merged"):
            token = ch.try_wait()
            if ctx.n > 1:
                token = ch.try_wait()
            ch.wait(token=token).release()
        elif cutlass.const_expr(mode == "preview_region"):
            preview = ch.peek()
            for _ in cutlass.range(ctx.n, unroll=1):
                # Through a helper: a method call on the local name would make CuTeDSL carry
                # the preview through the loop, which it rejects itself.
                look(preview)
                ch.wait().release()
        else:
            if cutlass.const_expr(mode == "boolean"):
                ch.wait(token=cutlass.Boolean(True)).release()
            elif cutlass.const_expr(mode == "forged"):
                ch.wait(token=Token(ch, None, cutlass.Boolean(True))).release()
            elif cutlass.const_expr(mode == "other_end"):
                ch.wait(token=other.try_wait()).release()
            elif cutlass.const_expr(mode == "other_side"):
                ch.wait(token=back.try_acquire()).release()
            elif cutlass.const_expr(mode == "stale"):
                token = ch.try_wait()
                ch.wait().release()
                ch.wait(token=token).release()
            elif cutlass.const_expr(mode == "wrap"):
                token = ch.try_wait()
                for _ in cutlass.range_constexpr(4):  # the same stage and parity again
                    ch.wait().release()
                ch.wait(token=token).release()
            elif cutlass.const_expr(mode == "reused"):
                token = ch.try_wait()
                ch.wait(token=token).release()
                ch.wait(token=token).release()
            elif cutlass.const_expr(mode == "state"):
                token = ch.try_wait()
                ch.state = ch.state
                ch.wait(token=token).release()
            elif cutlass.const_expr(mode == "state_read"):
                token = ch.try_wait()
                ch.state.advance()  # raw code moving the live state in place
                ch.wait(token=token).release()
            elif cutlass.const_expr(mode == "preview_state"):
                ch.peek()
                ch.state = ch.state
                ch.wait().release()
            elif cutlass.const_expr(mode == "slot"):
                slots.wait(1, token=slots.try_wait(0)).release()
            elif cutlass.const_expr(mode == "named_try_wait"):
                named.try_wait()
            elif cutlass.const_expr(mode == "named_peek"):
                named.peek()
            elif cutlass.const_expr(mode == "preview_write"):
                ch.peek().view(write=True)
            elif cutlass.const_expr(mode == "preview_release"):
                ch.peek().release()
            elif cutlass.const_expr(mode == "preview_stale"):
                preview = ch.peek()
                ch.wait().release()
                preview.view()
            for _ in cutlass.range(ctx.n, unroll=1):
                with ch.consume(), other.consume(), back.produce(), named.consume():
                    pass
                for s in cutlass.range_constexpr(2):
                    with slots.consume(s):
                        pass
            if cutlass.const_expr(mode == "preview_unconsumed"):
                ch.peek()  # no wait follows
        return ch, other, back, slots, named


@pytest.mark.parametrize(
    "mode, error",
    [
        (
            "boolean",
            r"ch\.wait\(token=\.\.\.\) at .* takes the Token returned by ch\.try_wait\(\), got Boolean \(a raw readiness Boolean needs",
        ),
        ("forged", r"this token was not returned by ch\.try_wait\(\)"),
        ("other_end", r"got a token of the other consumer end of cons"),
        ("other_side", r"got a token of the back producer end of cons"),
        ("stale", r"stale token probed at test_warpdrive_tokens\.py:\d+"),
        ("wrap", r"stale token"),
        ("reused", r"stale token"),
        ("state", r"stale token"),
        ("state_read", r"stale token"),
        ("region", r"used in a different dynamic region"),
        (
            "carry_entry",
            r"carried through a loop or branch, and the value entering at .* is stale",
        ),
        ("carry_not_first", r"stale token probed at"),
        (
            "carry_back_edge",
            (
                r"token carried by the loop: the value one iteration passes to the next \(probed at "
                r"test_warpdrive_tokens\.py:\d+\) is stale"
            ),
        ),
        ("branch_merged", r"carried through a loop or branch, and a branch merged it"),
        ("carry_branch", r"passes to the next \(probed at .*\) is stale: a branch merged it"),
        ("slot", r"the token probed slot 0 .*, not 1"),
        ("named_try_wait", r"named\.try_wait\(\) at .*cannot be probed or previewed"),
        ("named_try_acquire", r"named\.try_acquire\(\) at .*cannot be probed or previewed"),
        ("named_peek", r"named\.peek\(\) at .*cannot be probed or previewed"),
        ("preview_write", r"view\(\.\.\., write=True\) on ch\[i\] peek .*read-only borrow"),
        ("preview_release", r"release\(\) on ch\[i\] peek .*read-only borrow"),
        ("preview_stale", r"after its epoch was handed back \(release at .*\): a stale preview"),
        ("preview_region", r"ch\[i\] peek taken at .* in a different dynamic region"),
        ("preview_unconsumed", r"previews no later wait consumes: ch\[i\] peek at"),
        ("preview_state", r"ch\[i\] peek taken at .*: end\.state was read or assigned before"),
    ],
)
def test_misuse_raises_while_tracing(mode, error):
    _raises_protocol(error, lambda: compile_toy(Misuse(mode), f"misuse_{mode}"))


class Accepted(Toy):
    """Valid token and preview uses the checks must not reject (compiled only)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.ch = Ring(2, Threads, Threads, payload=(cutlass.Int32, (N,)))
        self.slots = Slots(2, Threads, Threads)
        self.rr = Ring(2, Threads, Threads, consumer_sharing=Sharing.ROUND_ROBIN)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer, slots: Producer, rr: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            for k in cutlass.range_constexpr(2):  # FA4's K steps: re-probe after each commit
                token = ch.try_acquire()
                with ch.produce(token=token):
                    pass
            for s in cutlass.range_constexpr(2):
                with slots.produce(s):
                    pass
            for _ in cutlass.range_constexpr(2):
                with rr.produce():
                    pass
        return ch, slots, rr

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def cons(self, ctx, *, ch: Consumer, slots: Consumer, rr: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            if cutlass.const_expr(self.mode == "slots"):
                token = slots.try_wait(1)
                slots.wait(0).release()  # another slot keeps slot 1's epoch
                slots.wait(1, token=token).release()
            else:
                for s in cutlass.range_constexpr(2):
                    with slots.consume(s):
                        pass
            if cutlass.const_expr(self.mode == "helper"):
                token = ch.try_wait()
                ch = pass_through(ch)  # a helper returning the end keeps its record
                ch.wait(token=token).release()
                ch.peek().view()
                ch = pass_through(ch)
                ch.wait().release()
            elif cutlass.const_expr(self.mode == "token_then_peek"):
                token = ch.try_wait()
                ch.peek().view()
                ch.wait(token=token).release()
                ch.wait().release()
            else:
                token = ch.try_wait()
                for k in cutlass.range_constexpr(2):
                    with ch.consume(token=token):
                        pass
                    if cutlass.const_expr(k == 0):
                        token = ch.try_wait()
            if cutlass.const_expr(self.mode == "carried"):
                token = ch.try_wait()
                for _ in cutlass.range(2, unroll=1):
                    ch.wait(token=token).release()
                    token = ch.try_wait()
                ch.wait(token=token).release()  # the loop's last probe, after the loop
                token = ch.try_wait()
                k = cutlass.Int32(0)
                while k < ctx.n:  # an scf.while carries it the same way
                    ch.wait(token=token).release()
                    token = ch.try_wait()
                    k += 1
                for _ in cutlass.range(2, unroll=1):  # nested: each loop carries it
                    for _ in cutlass.range(2, unroll=1):
                        ch.wait(token=token).release()
                        token = ch.try_wait()
                ch.wait(token=token).release()
                slot_token = slots.try_wait(1)  # a slot's token, carried past slot 0's waits
                for _ in cutlass.range(2, unroll=1):
                    slots.wait(0).release()
                    slots.wait(1, token=slot_token).release()
                    slot_token = slots.try_wait(1)
            token = rr.try_wait()  # this instance's own epoch of the round-robin ring
            rr.peek()
            rr.wait(token=token).release()
        return ch, slots, rr


def pass_through(end):
    return end


@pytest.mark.parametrize("mode", ["unrolled", "slots", "helper", "token_then_peek", "carried"])
def test_valid_uses_compile(mode):
    compile_toy(Accepted(mode), f"accepted_{mode}")


# Alias diagnostic: a preview's reads belong to the handle that owns its epoch


class PreviewOverlap(Toy):
    """The producer writes Q, the consumer previews P (Q overlaps P) and then owns the epoch
    with a wait. ``consumer``: the release mechanism that must publish the preview's read."""

    def __init__(self, consumer):
        self.label = f"overlap_{consumer.__name__.lower()}"
        self.smem_plan = SmemPlan()
        p = self.smem_plan.buffer("P", cutlass.Int32)
        q = self.smem_plan.buffer("Q", cutlass.Int32, overlaps=p)
        self.c = Ring(2, Threads, consumer, buffer={"P": p, "Q": q})

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32, delay: cutlass.Int64):
        layouts = {"P": cute.make_layout((32, 2)), "Q": cute.make_layout((32, 2))}
        self.begin_roles(layouts=layouts)
        self.run_roles(SimpleNamespace(out=out, n=n, delay=delay))

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, c: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with c.produce() as h:
                h.access("Q", write=True)
        return c

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, c: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            c.peek().access("P", read=True)
            c.wait().release()
        return c


@pytest.mark.parametrize(
    "consumer, expected", [(Threads, "no hazard found"), (Umma, "potential hazard")]
)
def test_preview_reads_are_checked_against_the_owning_release(consumer, expected):
    op = PreviewOverlap(consumer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compile_toy(op, op.label)
    text = alias_section(op)
    assert re.search(rf"(P ~ Q|Q ~ P) \(smem\)  {expected}", text), text
    if consumer is Umma:  # the Umma release does not publish the preview's thread read
        assert "cons read P by threads" in text and "[commit]" in text, text


# Runtime: both consumer probe outcomes on a payload ring


class CarriedToy(Toy):
    """Probe before the loop and after each release; use readiness in the next wait.

    ``carry="raw"`` carries token.ready as a Boolean under unchecked("token"); "token"
    carries a checked Token. A delayed producer makes every probe find the stage empty;
    a delayed consumer makes every probe find it full. ``out`` holds consumer probe
    results followed by wrong words. Producer probes are in test_warpdrive_hardening.py.
    """

    def __init__(self, carry: str, late: str):
        self.carry, self.late = carry, late
        self.label = f"carried_{carry}_{late}"
        self.ch = Ring(2, Threads, Threads, payload=(cutlass.Int32, (N,)))

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            if cutlass.const_expr(self.late == "producer"):
                spin(ctx.delay)
            with ch.produce() as h:
                h.data[lane] = i * N + lane
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        lane = cute.arch.lane_idx()
        bad = cutlass.Int32(0)
        if cutlass.const_expr(self.late == "consumer"):
            spin(ctx.delay)
        if cutlass.const_expr(self.carry == "raw"):
            ready = ch.try_wait().ready
            for i in cutlass.range(ctx.n, unroll=1):
                if lane == 0:
                    ctx.out[i] = cutlass.Int32(ready)
                with unchecked("token", reason="probed at the end of the previous iteration"):
                    h = ch.wait(token=ready)
                bad = bad + (h.data[lane] != i * N + lane)
                h.release()
                if cutlass.const_expr(self.late == "consumer"):
                    spin(ctx.delay)
                ready = ch.try_wait().ready  # the next iteration's epoch
        else:
            token = ch.try_wait()
            for i in cutlass.range(ctx.n, unroll=1):
                if lane == 0:
                    ctx.out[i] = cutlass.Int32(token.ready)
                h = ch.wait(token=token)
                bad = bad + (h.data[lane] != i * N + lane)
                h.release()
                if cutlass.const_expr(self.late == "consumer"):
                    spin(ctx.delay)
                token = ch.try_wait()  # the next iteration's epoch
        ctx.out[ctx.n + lane] = bad
        return ch


@pytest.mark.parametrize("carry", ["raw", "token"])
@pytest.mark.parametrize("late", ["producer", "consumer"])
def test_readiness_carried_across_iterations(requires_blackwell, carry, late):
    n = 6
    op = CarriedToy(carry, late)
    fn = compile_toy(op, op.label)
    if carry == "raw":
        assert re.search(
            r"unchecked token @ .*'probed at the end of the previous iteration' -> "
            r"ch\.wait\(token=Boolean\)",
            op.describe(),
        ), op.describe()
    for _ in range(3):
        out = torch.full((n + N,), -1, device="cuda", dtype=torch.int32)
        fn(out, n, DELAY)
        torch.cuda.synchronize()
        assert out[n:].tolist() == [0] * N
        assert out[:n].tolist() == [int(late == "consumer")] * n


# Runtime: previews across wraps


class PeekToy(Toy):
    """The producer (late by ``delay``) fills stage i with ``i * N + lane``; the consumer reads
    each epoch through a preview and then through its handle. ``pattern``:

    * ``"before"``: peek, then wait, every iteration (Ring);
    * ``"next"``: FA4 backward's shape: peek epoch 0 before the loop; each iteration waits for
      epoch i, previews i + 1 while holding i, then releases i; the last iteration is peeled;
    * ``"slots"``: peek then wait on each of two Slots.

    ``out``: per epoch the preview word minus the handle word, then the wrong words."""

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.label = f"peek_{pattern}"
        payload = (cutlass.Int32, (N,))
        if pattern == "slots":
            self.ch = Slots(2, Threads, Threads, payload=payload)
        else:
            self.ch = Ring(2, Threads, Threads, payload=payload)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(self.pattern == "slots"):
            for i in cutlass.range(ctx.n // 2, unroll=1):
                for s in cutlass.range_constexpr(2):
                    spin(ctx.delay)
                    with ch.produce(s) as h:
                        h.data[lane] = (2 * i + s) * N + lane
        else:
            for i in cutlass.range(ctx.n, unroll=1):
                spin(ctx.delay)
                with ch.produce() as h:
                    h.data[lane] = i * N + lane
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        lane = cute.arch.lane_idx()
        bad = cutlass.Int32(0)
        if cutlass.const_expr(self.pattern == "before"):
            for i in cutlass.range(ctx.n, unroll=1):
                seen = ch.peek().view(read=True)[lane]
                with ch.consume() as h:
                    word = h.data[lane]
                    bad = bad + (word != i * N + lane)
                    if lane == 0:
                        ctx.out[i] = seen - word
        elif cutlass.const_expr(self.pattern == "slots"):
            for i in cutlass.range(ctx.n // 2, unroll=1):
                for s in cutlass.range_constexpr(2):
                    seen = ch.peek(s).view()[lane]
                    with ch.consume(s) as h:
                        word = h.data[lane]
                        bad = bad + (word != (2 * i + s) * N + lane)
                        if lane == 0:
                            ctx.out[2 * i + s] = seen - word
        else:
            seen = ch.peek().view()[lane]
            for i in cutlass.range(ctx.n - 1, unroll=1):
                h = ch.wait()
                word = h.data[lane]
                bad = bad + (word != i * N + lane)
                if lane == 0:
                    ctx.out[i] = seen - word
                seen = ch.peek().view()[lane]  # epoch i + 1 while holding i
                h.release()
            last = ch.wait()  # the peeled last epoch owns the loop's final preview
            word = last.data[lane]
            bad = bad + (word != (ctx.n - 1) * N + lane)
            if lane == 0:
                ctx.out[ctx.n - 1] = seen - word
            last.release()
        ctx.out[ctx.n + lane] = bad
        return ch


@pytest.mark.parametrize("pattern", ["before", "next", "slots"])
def test_peek_reads_the_epoch_its_wait_takes(requires_blackwell, pattern):
    n = 8  # four wraps of each stage
    fn = compile_toy(PeekToy(pattern), f"peek_{pattern}")
    for delay in (0, 20_000, 0):  # the producer on time, late, then a fresh launch again
        out = torch.full((n + N,), -1, device="cuda", dtype=torch.int32)
        fn(out, n, delay)
        torch.cuda.synchronize()
        assert out.tolist() == [0] * (n + N), delay
