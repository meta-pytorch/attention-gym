"""Exclusive consumers (``Ring(..., consumer_sharing=Sharing.EXCLUSIVE)``): each epoch is owned
by one of several consumer roles, which waits and releases it, while the others ``skip()`` it.

Declaration, collection and trace-time errors; the static ownership check and ``describe()``;
the alias diagnostic's treatment of exclusive epochs; runtime toys with interleaved and grouped
ownership (ring wraps, owner changes across generations behind a named-barrier rendezvous,
runtime-dependent skips, a delayed owner, zero/one/odd work counts, repeated launches)."""

import re
import warnings
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    NamedBarrier,
    OneShot,
    Producer,
    Ring,
    Sharing,
    Slots,
    StaticPersistent,
    Threads,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    advance_state,
    role,
)

N = 32  # payload words per stage: one per lane
DELAY = 20_000  # cycles a delayed owner spins before each of its waits


@cute.jit
def spin(cycles):
    start = cute.arch.clock64()
    while cute.arch.clock64() - start < cycles:
        pass


def exclusive(n=4, consumer=Threads, **kw):
    return Ring(
        n,
        Threads,
        consumer,
        consumer_roles=("a", "b"),
        consumer_sharing=Sharing.EXCLUSIVE,
        payload=(cutlass.Int32, (N,)),
        **kw,
    )


class Toy(WarpSpecialized):
    """``prod`` (warp 0) fills ``kv``; consumers ``a`` (warp 1) and ``b`` (warp 2). Roles get
    ``ctx = (out, n, delay_a, delay_b)``; one CTA."""

    label = "toy"

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32, da: cutlass.Int64, db: cutlass.Int64):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n, delay_a=da, delay_b=db))

    @cute.jit
    def __call__(self, out, n: cutlass.Int32, da: cutlass.Int64, db: cutlass.Int64, stream):
        self.kernel.set_name_prefix(f"warpdrive_exclusive_{self.label}")
        self.kernel(out, n, da, db).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_toy(op, key: str):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(
        op,
        out,
        cutlass.Int32(1),
        cutlass.Int64(0),
        cutlass.Int64(0),
        name=f"warpdrive_exclusive_{key}",
    )


@cute.jit
def fill(ctx, kv, epochs):
    """Epoch e's payload is ``e * N + lane``."""
    lane = cute.arch.lane_idx()
    for e in cutlass.range(epochs, unroll=1):
        with kv.produce() as h:
            h.data[lane] = e * N + lane
    return kv


@cute.jit
def take(ctx, kv, e, who: int, delay):
    """Own epoch ``e``: record ``who`` as its owner and count a wrong payload word."""
    lane = cute.arch.lane_idx()
    spin(delay)
    with kv.consume() as h:
        if h.data[lane] != e * N + lane:
            cute.arch.atomic_add(ctx.out.iterator + (ctx.out.shape[0] - 1), cutlass.Int32(1))
        if lane == 0:
            ctx.out[e] = who
    return kv


# --------------------------------------------------------------------------- #
# Declaration and collection errors
# --------------------------------------------------------------------------- #


def test_declaration_errors():
    excl = Sharing.EXCLUSIVE
    with pytest.raises(ValueError, match="consumer_sharing must be a Sharing"):
        Ring(2, Threads, Threads, consumer_sharing="exclusive")
    with pytest.raises(ValueError, match="needs a Ring"):
        Slots(2, Threads, Threads, consumer_roles=("a", "b"), consumer_sharing=excl)
    with pytest.raises(ValueError, match="needs an mbarrier ring: exclusive consumers skip"):
        exclusive(sync="named")
    with pytest.raises(ValueError, match="producer_sharing must be Sharing.JOIN"):
        exclusive(producer_sharing=Sharing.ROUND_ROBIN)
    with pytest.raises(ValueError, match="producer_sharing=Sharing.EXCLUSIVE is not supported"):
        Ring(2, Threads, Threads, producer_sharing=excl)
    with pytest.raises(ValueError, match="needs consumer_roles"):
        Ring(2, Threads, Threads, consumer_sharing=excl)


def _kernel(channels, roles, cluster=None):
    """A kernel class with the given channel attributes and role functions."""
    cls = type("Probe", (WarpSpecialized,), dict(roles))

    def init(self):
        for name, make in channels.items():
            setattr(self, name, make())
        if cluster is not None:
            self.cluster = cluster

    cls.__init__ = init
    return cls()


def _role(warps, side, cta="all"):
    @role(warps=warps, cta=cta)
    @cute.jit
    def fn(self, ctx, *, kv: side):
        return kv

    return fn


@pytest.mark.parametrize(
    "roles, message",
    [
        (
            {"p": _role(0, Producer), "a": _role(1, Consumer)},
            r"needs at least two consumer instances, got \['a'\]",
        ),
        (
            {"p": _role(0, Producer), "a": _role(1, Consumer), "b": _role(range(2, 4), Consumer)},
            r"must contribute equal arrivals .*\{'a': 32, 'b': 64\}",
        ),
        (
            {
                "p": _role(0, Producer),
                "q": _role(3, Producer),
                "a": _role(1, Consumer),
                "b": _role(2, Consumer),
            },
            r"need one producer instance .*\['p', 'q'\]",
        ),
    ],
    ids=["one_consumer", "unequal", "two_producers"],
)
def test_collection_errors(roles, message):
    op = _kernel({"kv": lambda: exclusive(2)}, roles)
    _raises_protocol(message, op.describe)


def test_overrides_apply_to_exclusive_rings():
    """Full-side overrides leave ownership alone; overridden empty counts are the author's, so
    unequal consumers are accepted and the counts are marked unverified."""
    roles = {"p": _role(0, Producer), "a": _role(1, Consumer), "b": _role(range(2, 4), Consumer)}
    op = _kernel({"kv": lambda: exclusive(2, override={"empty_counts": 1})}, roles)
    text = op.describe()
    assert "empty     1 x2" in text
    assert "empty counts overridden: each owner's release must supply them (unverified)" in text
    roles = {"p": _role(0, Producer), "a": _role(1, Consumer), "b": _role(2, Consumer)}
    op = _kernel({"kv": lambda: exclusive(2, override={"full_counts": 1})}, roles)
    text = op.describe()
    assert "override  full_counts 32 x2 -> 1 x2  (differs from derived)" in text
    assert "counts are one consumer's" in text


def test_declared_role_must_consume():
    roles = {"p": _role(0, Producer), "a": _role(1, Consumer), "a2": _role(2, Consumer)}
    op = _kernel(
        {"kv": lambda: Ring(2, Threads, Threads, consumer_roles=("a", "b", "a2"), **_EXCL)},
        roles,
    )
    _raises_protocol(r"exclusive consumer role\(s\) \['b'\] take no Consumer end", op.describe)


_EXCL = {"consumer_sharing": Sharing.EXCLUSIVE}


def test_umma_consumers_and_cross_cta_placement():
    # Two single-warp Umma consumers are accepted (each epoch has one commit); a cross-CTA
    # ring is not.
    roles = {"p": _role(0, Producer), "a": _role(1, Consumer), "b": _role(2, Consumer)}
    op = _kernel({"kv": lambda: exclusive(2, Umma)}, roles)
    assert "empty     1 x2" in op.describe()
    roles = {
        "p": _role(0, Producer, cta="leader"),
        "a": _role(1, Consumer),
        "b": _role(2, Consumer),
    }
    unstaged = {"kv": lambda: Ring(2, Threads, Threads, consumer_roles=("a", "b"), **_EXCL)}
    op = _kernel(unstaged, roles, cluster=CtaPair())
    _raises_protocol("exclusive consumers are CTA-local", op.describe)


# --------------------------------------------------------------------------- #
# Trace-time misuse and the static ownership check
# --------------------------------------------------------------------------- #


class Static(Toy):
    """Six epochs, static ownership ``a a b a a b`` on three stages unless ``mode`` breaks it.
    A runnable program in mode "ok": three stages match the three-epoch period, so every
    consumer saw each stage's previous epoch itself and needs no rendezvous."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.label = f"static_{mode}"
        self.kv = exclusive(3)
        self.join = Ring(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, kv: Producer, join: Producer):
        lane = cute.arch.lane_idx()
        for e in cutlass.range_constexpr(6):
            with kv.produce() as h:
                h.data[lane] = e * N + lane
        if cutlass.const_expr(self.mode == "join_skip"):
            with join.produce():
                pass
        return kv, join

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, kv: Consumer):
        mode = self.mode
        for g in cutlass.range_constexpr(2):
            if cutlass.const_expr(mode == "token" and g == 1):
                token = kv.try_wait()
                kv.skip(1)
                kv.wait(token=token).release()
                kv.wait().release()
                continue
            if cutlass.const_expr(mode == "preview" and g == 1):
                kv.peek()
                kv.skip(1)
            if cutlass.const_expr(mode == "bad_count"):
                kv.skip(0)
            if cutlass.const_expr(mode == "bool_count"):
                kv.skip(True)
            if cutlass.const_expr(mode == "dynamic_count"):
                kv.skip(ctx.n)
            if cutlass.const_expr(g == 1):
                kv.skip(1)  # b's epoch 2
            take(ctx, kv, 3 * g, 1, ctx.delay_a)
            take(ctx, kv, 3 * g + 1, 1, ctx.delay_a)
        return kv

    @role(warps=2)
    @cute.jit
    def b(self, ctx, *, kv: Consumer, join: Consumer):
        for g in cutlass.range_constexpr(2):
            if cutlass.const_expr(self.mode == "double" and g == 0):
                take(ctx, kv, 0, 2, ctx.delay_b)  # a owns epoch 0 too
                kv.skip(1)
            elif cutlass.const_expr(self.mode == "raw_state" and g == 0):
                state = kv.state  # a raw skip of a's two epochs
                state = advance_state(state, 2)
                kv.state = state
            else:
                kv.skip(2)
            if cutlass.const_expr(not (self.mode == "gap" and g == 1)):
                take(ctx, kv, 3 * g + 2, 2, ctx.delay_b)
        if cutlass.const_expr(self.mode == "past"):
            kv.skip(1)
            take(ctx, kv, 6, 2, ctx.delay_b)
        if cutlass.const_expr(self.mode == "join_skip"):
            join.skip(1)
            join.wait().release()
        return kv, join


class Wide(Static):
    """Static ownership ``a a a a a b`` on three stages: b's only wait (epoch 5) comes a ring
    lap after launch, so the check names it as needing a rendezvous (compiled only)."""

    def __init__(self):
        super().__init__("wide")

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, kv: Consumer):
        for e in cutlass.range_constexpr(5):
            take(ctx, kv, e, 1, ctx.delay_a)
        return kv

    @role(warps=2)
    @cute.jit
    def b(self, ctx, *, kv: Consumer, join: Consumer):
        kv.skip(5)
        take(ctx, kv, 5, 2, ctx.delay_b)
        return kv, join


@pytest.mark.parametrize(
    "mode, error",
    [
        ("join_skip", r"join\.skip\(\) at .*only consumers of an exclusive ring"),
        ("bad_count", r"kv\.skip\(0\) at .*must be a positive Python int"),
        ("bool_count", r"kv\.skip\(True\) at .*must be a positive Python int"),
        ("token", r"stale token probed at test_warpdrive_exclusive\.py:\d+"),
        ("preview", r"kv\.skip\(\) at .* with an unconsumed preview \(kv\[i\] peek"),
        ("double", r"epoch 0 of 6 per launch is waited by a and b"),
        ("gap", r"epoch 5 of 6 per launch is waited by no consumer"),
        ("past", r"b waits on epoch 7, but prod fills only 6 per launch"),
    ],
)
def test_misuse_raises_while_tracing(mode, error):
    _raises_protocol(error, lambda: compile_toy(Static(mode), f"static_{mode}"))


class Retired(Static):
    """The producer acquires three epochs and retires one unpublished: the last (``kind``
    "last"), or the middle one while it holds all three ("middle"); a owns epoch 0, b the other
    published one. "wait_retired": b waits for the retired middle epoch instead."""

    B_SKIPS: ClassVar[dict[str, int]] = {"last": 1, "middle": 2, "wait_retired": 1}

    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(f"retired_{kind}")

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, kv: Producer, join: Producer):
        if cutlass.const_expr(self.kind == "last"):
            for _ in cutlass.range_constexpr(2):
                with kv.produce():
                    pass
            kv.acquire().retire()
        else:
            h0, h1, h2 = kv.acquire(), kv.acquire(), kv.acquire()
            h1.retire()
            h0.commit()
            h2.commit()
        return kv, join

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, kv: Consumer):
        kv.wait().release()
        return kv

    @role(warps=2)
    @cute.jit
    def b(self, ctx, *, kv: Consumer, join: Consumer):
        kv.skip(self.B_SKIPS[self.kind])
        kv.wait().release()
        return kv, join


@pytest.mark.parametrize("kind", ["last", "middle"])
def test_static_ownership_leaves_a_retired_epoch_unowned(kind):
    op = Retired(kind)
    compile_toy(op, f"static_retired_{kind}")
    assert "epochs    verified: 2 epochs per launch (a 1, b 1)\n" in op.describe()


def test_static_ownership_rejects_a_wait_for_a_retired_epoch():
    _raises_protocol(
        "b waits on epoch 1, but prod retires it unpublished",
        lambda: compile_toy(Retired("wait_retired"), "static_retired_wait_retired"),
    )


class PerItem(WarpSpecialized):
    """``a`` takes even and ``b`` odd epochs of each work item's two. ``mode``: "ok"; "short"
    (a does not skip b's trailing epoch, so under a persistent schedule its next item starts on
    the wrong epoch); "prologue" (the producer fills both epochs once, before its work loop);
    "launch" (every role uses the ring once, before its work loop; a skips nothing).
    Compiled only."""

    def __init__(self, schedule, mode: str):
        self.mode = mode
        self.kv = exclusive(2)
        self.work = WorkStream(schedule=schedule)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, kv: Producer, work: WorkCursor):
        once = self.mode in ("prologue", "launch")
        if cutlass.const_expr(once):
            for _ in cutlass.range_constexpr(2):
                with kv.produce():
                    pass
        while work.valid:
            if cutlass.const_expr(not once):
                for _ in cutlass.range_constexpr(2):
                    with kv.produce():
                        pass
            work.advance()
        return kv, work

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, kv: Consumer, work: WorkCursor):
        if cutlass.const_expr(self.mode == "launch"):
            kv.wait().release()
        while work.valid:
            if cutlass.const_expr(self.mode != "launch"):
                kv.wait().release()
                if cutlass.const_expr(self.mode != "short"):
                    kv.skip(1)
            work.advance()
        return kv, work

    @role(warps=2)
    @cute.jit
    def b(self, ctx, *, kv: Consumer, work: WorkCursor):
        if cutlass.const_expr(self.mode == "launch"):
            kv.skip(1)
            kv.wait().release()
        while work.valid:
            if cutlass.const_expr(self.mode != "launch"):
                kv.skip(1)
                kv.wait().release()
            work.advance()
        return kv, work

    @cute.kernel
    def kernel(self, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace())

    @cute.jit
    def __call__(self, unused: cute.Tensor, n: cutlass.Int32, stream):
        bound = self.work.bind((n, 1, 1), max_ctas=cutlass.Int32(1))
        self.kernel.set_name_prefix(f"warpdrive_exclusive_items_{self.mode}")
        self.kernel(bound).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_items(op):
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    name = f"warpdrive_exclusive_items_{op.mode}_{op.work.get_key()}"
    return compile_tvm_ffi(op, unused, cutlass.Int32(1), name=name)


def test_static_ownership_per_work_item():
    op = PerItem(StaticPersistent(), "ok")
    compile_items(op)
    assert "epochs    verified: 2 epochs per work item (a 1, b 1)\n" in op.describe()
    # A launch-once prologue does not pair with per-item waits.
    op = PerItem(OneShot(), "prologue")
    compile_items(op)
    assert (
        "epochs    unverified (some roles use it once per work item, others once per launch)"
    ) in op.describe()
    # One item per CTA: b's trailing epoch needs no skip from a. A persistent CTA's next item
    # would start a one epoch early.
    op = PerItem(OneShot(), "short")
    compile_items(op)
    assert "epochs    verified: 2 epochs per work item (a 1, b 1)\n" in op.describe()
    _raises_protocol(
        "a advances 1 epochs per work item .*, but prod acquires 2",
        lambda: compile_items(PerItem(StaticPersistent(), "short")),
    )
    # Used once per launch, the ring has no next item to line up, even under a persistent
    # schedule.
    op = PerItem(StaticPersistent(), "launch")
    compile_items(op)
    assert "epochs    verified: 2 epochs per launch (a 1, b 1)\n" in op.describe()


@pytest.mark.parametrize(
    "mode, reason",
    [
        ("dynamic_count", "a's skip count is dynamic"),
        ("raw_state", "b reads or assigns end.state"),
    ],
)
def test_dynamic_or_raw_skips_leave_ownership_unverified(mode, reason):
    op = Static(mode)
    compile_toy(op, f"static_{mode}")
    assert f"epochs    unverified ({reason})" in op.describe()


def test_static_check_names_a_wait_that_needs_a_rendezvous():
    op = Wide()
    compile_toy(op, "static_wide")
    assert (
        "epochs    verified: 6 epochs per launch (a 5, b 1); b's wait of epoch 5 is exact only "
        "after epoch 2 was filled, which it did not observe: order it with a rendezvous"
    ) in op.describe()


def test_static_ownership_is_verified_and_described(requires_blackwell):
    op = Static()
    fn = compile_toy(op, "static_ok")
    text = op.describe()
    assert "kv    Ring(3)  prod -> a, b" in text, text
    assert "empty     32 x3" in text  # one consumer's arrivals, not the join's 64
    assert "owners    exclusive: one of a, b owns each epoch, the others skip it" in text
    assert "epochs    verified: 6 epochs per launch (a 4, b 2)\n" in text
    for _ in range(3):
        out = torch.full((7,), -1, device="cuda", dtype=torch.int32)
        out[-1] = 0
        fn(out, 0, 0, 0)
        torch.cuda.synchronize()
        assert out.tolist() == [1, 1, 2, 1, 1, 2, 0]


# --------------------------------------------------------------------------- #
# Runtime: interleaved and grouped ownership
# --------------------------------------------------------------------------- #


class Interleaved(Toy):
    """``a`` owns even epochs, ``b`` odd ones, of ``n`` epochs on a four-stage ring. Each stage's
    epochs alternate between the same consumer (period 2 divides 4 stages), so no rendezvous
    is needed: each waiter saw the stage's previous epoch itself. ``out``: the owner of every
    epoch (1 = a, 2 = b), then the count of wrong payload words."""

    label = "interleaved"

    def __init__(self):
        self.kv = exclusive(4)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, kv: Producer):
        return fill(ctx, kv, ctx.n)

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, kv: Consumer):
        for i in cutlass.range((ctx.n + 1) // 2, unroll=1):
            kv = take(ctx, kv, 2 * i, 1, ctx.delay_a)
            kv.skip(1)
        return kv

    @role(warps=2)
    @cute.jit
    def b(self, ctx, *, kv: Consumer):
        for i in cutlass.range(ctx.n // 2, unroll=1):
            kv.skip(1)
            kv = take(ctx, kv, 2 * i + 1, 2, ctx.delay_b)
        return kv


class Grouped(Toy):
    """Rounds of ``group + 1`` epochs on a ``stages``-stage ring: ``a`` owns a ``group``-epoch
    group, then ``b`` one epoch. When the stage count does not divide the round, a stage's
    owner changes between generations. With ``group >= stages`` b's waits come more than a
    ring lap after its previous one (its first wait included): its parity test is exact only
    once the stage's previous epoch was filled, which b never observed (without a rendezvous
    the toy hangs). FA4's recipe keeps the two within one window: ``a_done`` (a arrives after
    its group, b syncs before its epoch) and ``b_done`` (the reverse). ``a``'s skip of b's
    previous epoch depends on the runtime round index, as the decode's skips do."""

    def __init__(self, stages: int, group: int):
        self.group = group
        self.label = f"grouped_{stages}_{group}"
        self.kv = exclusive(stages)
        self.a_done = NamedBarrier(threads=64)
        self.b_done = NamedBarrier(threads=64)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, kv: Producer):
        return fill(ctx, kv, (self.group + 1) * ctx.n)

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, kv: Consumer):
        period = self.group + 1
        for r in cutlass.range(ctx.n, unroll=1):
            if r > 0:
                self.b_done.sync()  # b took the round's last epoch: every earlier one is filled
                kv.skip(1)
            for i in cutlass.range_constexpr(self.group):
                kv = take(ctx, kv, period * r + i, 1, ctx.delay_a)
            self.a_done.arrive()
        if ctx.n > 0:
            self.b_done.sync()  # b's last arrival: every generation completes
        return kv

    @role(warps=2)
    @cute.jit
    def b(self, ctx, *, kv: Consumer):
        period = self.group + 1
        for r in cutlass.range(ctx.n, unroll=1):
            kv.skip(self.group)
            self.a_done.sync()  # a took this round's group: every earlier epoch is filled
            kv = take(ctx, kv, period * r + self.group, 2, ctx.delay_b)
            self.b_done.arrive()
        return kv


@cute.jit
def own(ctx, kv, e, who: int, delay):
    """``take()`` whose release leaves the arrival to the caller; returns (kv, stage)."""
    lane = cute.arch.lane_idx()
    spin(delay)
    h = kv.wait()
    if h.data[lane] != e * N + lane:
        cute.arch.atomic_add(ctx.out.iterator + (ctx.out.shape[0] - 1), cutlass.Int32(1))
    if lane == 0:
        ctx.out[e] = who
    stage = h.index
    h.release(signal=False)
    return kv, stage


class Halves(Toy):
    """``a`` (one warp) owns epochs [0, n // 2), ``b`` (two warps) the rest, after a dynamic
    ``skip(n // 2)``. The consumers contribute unequal arrivals, so ``empty_counts=1``
    overrides the count: each releases with ``signal=False``, then one thread arrives once the
    role's threads have read the stage. b waits only after ``a_done``, so every earlier epoch
    is filled and its first wait is exact."""

    label = "halves"

    def __init__(self):
        self.kv = exclusive(4, override={"empty_counts": 1})
        self.a_done = NamedBarrier(threads=96)
        self.b_read = NamedBarrier(threads=64)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, kv: Producer):
        return fill(ctx, kv, ctx.n)

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, kv: Consumer):
        for e in cutlass.range(ctx.n // 2, unroll=1):
            kv, stage = own(ctx, kv, e, 1, ctx.delay_a)
            cute.arch.sync_warp()
            if cute.arch.lane_idx() == 0:
                cute.arch.mbarrier_arrive(kv.empty_barrier(stage))
        self.a_done.arrive()
        return kv

    @role(warps=range(2, 4))
    @cute.jit
    def b(self, ctx, *, kv: Consumer):
        half = ctx.n // 2
        kv.skip(half)
        self.a_done.sync()
        for e in cutlass.range(half, ctx.n, unroll=1):
            kv, stage = own(ctx, kv, e, 2, ctx.delay_b)
            self.b_read.sync()
            if cute.arch.thread_idx()[0] == 64:
                cute.arch.mbarrier_arrive(kv.empty_barrier(stage))
        return kv


def _run(fn, epochs, n, delay_a=0, delay_b=0):
    out = torch.full((epochs + 1,), -1, device="cuda", dtype=torch.int32)
    out[-1] = 0
    fn(out, n, delay_a, delay_b)
    torch.cuda.synchronize()
    return out[:-1].tolist(), int(out[-1])


@pytest.mark.parametrize("n", [0, 1, 7, 24])
@pytest.mark.parametrize("late", ["none", "a", "b"])
def test_interleaved_owners(requires_blackwell, n, late):
    op = Interleaved()
    fn = compile_toy(op, "interleaved")
    assert (
        "epochs    unverified (prod's acquire runs inside a dynamic loop or branch)"
        in op.describe()
    )
    for _ in range(2):  # repeated launches start from fresh barriers
        owners, bad = _run(fn, n, n, DELAY * (late == "a"), DELAY * (late == "b"))
        assert bad == 0 and owners == [1 + e % 2 for e in range(n)]


@pytest.mark.parametrize("n", [0, 1, 9, 24])
@pytest.mark.parametrize("late", ["none", "a", "b"])
def test_unequal_owners_with_overridden_counts_and_dynamic_skip(requires_blackwell, n, late):
    op = Halves()
    fn = compile_toy(op, "halves")
    for _ in range(2):
        owners, bad = _run(fn, n, n, DELAY * (late == "a"), DELAY * (late == "b"))
        assert bad == 0 and owners == [1 if e < n // 2 else 2 for e in range(n)]


@pytest.mark.parametrize("stages, group", [(3, 2), (4, 2), (3, 4)])
@pytest.mark.parametrize("n", [0, 1, 5, 11])
@pytest.mark.parametrize("late", ["none", "a", "b"])
def test_grouped_owners_with_rendezvous(requires_blackwell, stages, group, n, late):
    op = Grouped(stages, group)
    fn = compile_toy(op, op.label)
    period = group + 1
    for _ in range(2):
        epochs = period * n
        owners, bad = _run(fn, epochs, n, DELAY * (late == "a"), DELAY * (late == "b"))
        assert bad == 0 and owners == [2 if e % period == group else 1 for e in range(epochs)]


# --------------------------------------------------------------------------- #
# Alias diagnostic
# --------------------------------------------------------------------------- #


class AliasModel(WarpSpecialized):
    """``prod`` writes TMEM S (threads) and publishes it on ``kv``; ``a`` reads P (an alias in
    S) inside its kv handle, then signals ``back`` (a -> prod), which orders the read before
    prod's next write. Only ``kv`` orders the write before the read: through an exclusive ring
    the diagnostic cannot match epochs, so the pair is unverified; joined, it is checked."""

    def __init__(self, mode: str):
        self.mode = mode
        roles = ("a", "b")
        self.kv = Ring(
            2,
            Threads,
            Threads,
            consumer_roles=roles,
            consumer_sharing=Sharing(mode),
            commit_fence="tmem_store",
        )
        self.back = Ring(1, Threads, Threads)
        self.tmem_plan = TmemPlan().region("S", 128).alias("P", of="S", offset=64, cols=64)

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, kv: Producer, back: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            tmem.access("S", write=True, lanes=range(32, 64))  # the lanes a reads
            with kv.produce():
                pass
            with back.consume():
                pass
        return kv, back

    @role(warps=1)
    @cute.jit
    def a(self, ctx, *, kv: Consumer, back: Producer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            with kv.consume():
                tmem.access("P", read=True, lanes=range(32, 64))
            with back.produce():
                pass
        return kv, back

    @role(warps=2)
    @cute.jit
    def b(self, ctx, *, kv: Consumer, tmem: Tmem):
        for _ in cutlass.range(ctx.n, unroll=1):
            if cutlass.const_expr(self.mode == "join"):
                with kv.consume():
                    pass
            else:
                kv.skip(1)  # a owns every epoch
        return kv

    @cute.kernel
    def kernel(self, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(n=n))

    @cute.jit
    def __call__(self, unused: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix(f"warpdrive_exclusive_alias_{self.mode}")
        self.kernel(n).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


@jit_cache
def compile_model(op):
    unused = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(
        op, unused, cutlass.Int32(1), name=f"warpdrive_exclusive_alias_{op.mode}"
    )


@pytest.mark.parametrize(
    "mode, expected", [("join", "no hazard found"), ("exclusive", "unverified")]
)
def test_alias_through_an_exclusive_ring_is_unverified(mode, expected):
    op = AliasModel(mode)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        compile_model(op)
    text = op.describe()
    text = text[text.index("Alias diagnostic") :]
    match = re.search(r"\n  (?:P ~ S|S ~ P) \(tmem\)  ([a-z ]+)", text)
    assert match and match.group(1).strip().split(" among")[0] == expected, text
    if mode == "exclusive":
        assert "kv: exclusive ring, epoch owners follow runtime skips" in text, text
