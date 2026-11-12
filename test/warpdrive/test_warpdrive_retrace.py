"""Test what a kernel reports after the same instance is traced again.

Every trace starts from no reports, so a retrace shows its own results only, and one that
fails part-way shows none of the earlier trace's.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute
from test_warpdrive_tokens import Toy, compile_toy

from attn_gym.testing.warpdrive import raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    Instance,
    NamedBarrier,
    Producer,
    ProtocolError,
    Ring,
    Sharing,
    SmemPlan,
    Threads,
    role,
    unchecked,
)


def trace_fails():
    raise ProtocolError("this trace fails in cons")


class Retraced(Toy):
    """prod (warp 0) fills two epochs of ``c`` and of ``d`` and syncs on ``done``; two
    round-robin consumers (warps 1 and 2) take one epoch of ``c`` each and ``d_waits[j]`` of
    ``d``. The reports cover Q overlapping P, epoch owners, a user of ``done``, and the
    producer's fence-check opt-out.

    ``_broken`` makes ``cons`` raise while it is traced, after ``prod`` was.
    """

    def __init__(self, d_waits: tuple[int, int] = (1, 1)):
        self.d_waits = d_waits
        self.smem_plan = SmemPlan()
        p = self.smem_plan.buffer("P", cutlass.Int32)
        q = self.smem_plan.buffer("Q", cutlass.Int32, overlaps=p)
        self.c = Ring(
            2, Threads, Threads, buffer={"P": p, "Q": q}, consumer_sharing=Sharing.ROUND_ROBIN
        )
        self.d = Ring(2, Threads, Threads, consumer_sharing=Sharing.ROUND_ROBIN)
        self.done = NamedBarrier(threads=32)
        self.trace_protocol = True
        self._broken = False

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32, delay: cutlass.Int64):
        layouts = {"P": cute.make_layout((32, 2)), "Q": cute.make_layout((32, 2))}
        self.begin_roles(layouts=layouts)
        self.run_roles(SimpleNamespace(out=out, n=n, delay=delay))

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, c: Producer, d: Producer):
        for _ in cutlass.range_constexpr(2):
            with c.produce() as h, unchecked("fence", reason="retrace toy"):
                h.access("Q", write=True)
            with d.produce():
                pass
        self.done.sync()
        return c, d

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def cons(self, ctx, *, stage: Instance, c: Consumer, d: Consumer):
        with c.consume() as h:
            h.access("P", read=True)
        for _ in cutlass.range_constexpr(self.d_waits[stage]):
            d.wait().release()
        if cutlass.const_expr(self._broken):
            trace_fails()
        return c, d


def trace(op: Retraced):
    """Trace and compile ``op`` itself: ``__wrapped__`` takes no build from either cache."""
    compile_toy.__wrapped__(op, "retraced")


# describe() lines present on success but absent when cons fails during tracing.
TRACED_ONLY = (
    "liveness: verified",
    "epochs    verified",
    "sync by prod",
    "Unchecked opt-outs",
    "Alias diagnostic",
    "role cons[1]:",  # the protocol log
)


def test_retrace_reports_what_the_first_trace_did():
    op = Retraced()
    trace(op)
    first = op.describe(), op.checks()
    assert all(line in first[0] for line in TRACED_ONLY), first[0]
    trace(op)
    assert (op.describe(), op.checks()) == first


def test_failed_retrace_leaves_no_reports_of_the_earlier_trace():
    op = Retraced()
    untraced = op.checks()
    trace(op)
    traced = op.checks()
    assert traced.liveness == "verified" and traced.channels["c"].epochs == "verified"
    assert traced.aliases == {("Q", "P"): "no hazard found"}

    op._broken = True
    raises_protocol("this trace fails in cons", lambda: trace(op))
    assert op.checks() == untraced
    text = op.describe()
    assert not any(line in text for line in TRACED_ONLY), text
    assert "users known after a compile" in text
    assert "role prod:" in text  # the log holds what this trace reached
    assert op.__cute_trace_record__() is None


def test_failed_ownership_check_leaves_no_verdicts():
    """``c`` passes the ownership check before ``d`` fails it."""
    op = Retraced(d_waits=(2, 0))
    raises_protocol("channel d: cons.* waits on epoch 2", lambda: trace(op))
    assert op.checks() == Retraced(d_waits=(2, 0)).checks()
