"""Test exclusive rings: one consumer owns each epoch; the others skip it.

Cover declaration and trace errors, static ownership, describe(), and alias diagnostics.
Runtime toys exercise ring wraps, owner changes with a named-barrier rendezvous, dynamic
skips, delayed owners, zero/one/odd work counts, and repeated launches.
"""

import warnings
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.testing.warpdrive import alias_section, status
from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    NamedBarrier,
    OneShot,
    Producer,
    ProtocolError,
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
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor

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
def compile_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(
        op,
        out,
        cutlass.Int32(1),
        cutlass.Int64(0),
        cutlass.Int64(0),
        name=f"warpdrive_exclusive_{op.label}",
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


# Declaration and collection errors


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


_EXCL = {"consumer_sharing": Sharing.EXCLUSIVE}
_P, _A, _B = _role(0, Producer), _role(1, Consumer), _role(2, Consumer)


def _probe(roles, cluster=None, consumer_roles=("a", "b")):
    """A kernel with the given roles on ``kv``, an exclusive Ring of ``consumer_roles``."""
    channels = {"kv": lambda: Ring(2, Threads, Threads, consumer_roles=consumer_roles, **_EXCL)}
    return _kernel(channels, roles, cluster)


VALUE_ERRORS = {
    "sharing_type": (
        lambda: Ring(2, Threads, Threads, consumer_sharing="exclusive"),
        "consumer_sharing must be a Sharing",
    ),
    "slots": (lambda: Slots(2, Threads, Threads, consumer_roles=("a",), **_EXCL), "needs a Ring"),
    "named": (lambda: exclusive(sync="named"), "needs an mbarrier ring: exclusive consumers skip"),
    "producer_round_robin": (
        lambda: exclusive(producer_sharing=Sharing.ROUND_ROBIN),
        "producer_sharing must be Sharing.JOIN",
    ),
    "producer_exclusive": (
        lambda: Ring(2, Threads, Threads, producer_sharing=Sharing.EXCLUSIVE),
        "producer_sharing=Sharing.EXCLUSIVE is not supported",
    ),
    "no_consumer_roles": (lambda: Ring(2, Threads, Threads, **_EXCL), "needs consumer_roles"),
}
PROTOCOL_ERRORS = {
    "one_consumer": (lambda: _probe({"p": _P, "a": _A}), r"at least two consumer instances, got"),
    "unequal": (
        lambda: _probe({"p": _P, "a": _A, "b": _role(range(2, 4), Consumer)}),
        r"must contribute equal arrivals .*\{'a': 32, 'b': 64\}",
    ),
    "two_producers": (
        lambda: _probe({"p": _P, "q": _role(3, Producer), "a": _A, "b": _B}),
        r"need one producer instance .*\['p', 'q'\]",
    ),
    "role_without_end": (
        lambda: _probe({"p": _P, "a": _A, "a2": _B}, consumer_roles=("a", "b", "a2")),
        r"exclusive consumer role\(s\) \['b'\] take no Consumer end",
    ),
    "cross_cta": (
        lambda: _probe({"p": _role(0, Producer, cta="leader"), "a": _A, "b": _B}, CtaPair()),
        "exclusive consumers are CTA-local",
    ),
}


@pytest.mark.parametrize(
    "error, make, message",
    [(ValueError, *case) for case in VALUE_ERRORS.values()]
    + [(ProtocolError, *case) for case in PROTOCOL_ERRORS.values()],
    ids=[*VALUE_ERRORS, *PROTOCOL_ERRORS],
)
def test_declaration_errors(error, make, message):
    with pytest.raises(error, match=message):
        make().describe()


def test_overrides_apply_to_exclusive_rings():
    """Full-side overrides do not change ownership; empty-count overrides allow unequal consumers.

    Overridden counts are unverified and remain the author's responsibility. Two single-warp
    Umma consumers are valid because each epoch has one commit.
    """
    roles = {"p": _P, "a": _A, "b": _role(range(2, 4), Consumer)}
    op = _kernel({"kv": lambda: exclusive(2, override={"empty_counts": 1})}, roles)
    text = op.describe()
    assert "empty     1 x2" in text
    assert "empty counts overridden: each owner's release must supply them (unverified)" in text
    op = _kernel(
        {"kv": lambda: exclusive(2, override={"full_counts": 1})}, {"p": _P, "a": _A, "b": _B}
    )
    text = op.describe()
    assert "override  full_counts 32 x2 -> 1 x2  (differs from derived)" in text
    assert "counts are one consumer's" in text
    op = _kernel({"kv": lambda: exclusive(2, Umma)}, {"p": _P, "a": _A, "b": _B})
    assert "empty     1 x2" in op.describe()


# Trace-time misuse and the static ownership check


class Static(Toy):
    """Six epochs use ownership ``a a b a a b`` on three stages unless ``mode`` breaks it.

    Mode "ok" is runnable: the stage count matches the ownership period, so each consumer
    saw its stage's previous epoch and needs no rendezvous.
    """

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


def _compile(op):
    return compile_items(op) if isinstance(op, PerItem) else compile_toy(op)


@pytest.mark.parametrize(
    "make, error",
    [
        (lambda: Static("join_skip"), r"join\.skip\(\) at .*only consumers of an exclusive ring"),
        (lambda: Static("bad_count"), r"kv\.skip\(0\) at .*must be a positive Python int"),
        (lambda: Static("bool_count"), r"kv\.skip\(True\) at .*must be a positive Python int"),
        (lambda: Static("token"), r"stale token probed at test_warpdrive_exclusive\.py:\d+"),
        (lambda: Static("preview"), r"kv\.skip\(\) at .* with an unconsumed preview \(kv\[i\] pe"),
        (lambda: Static("double"), r"epoch 0 of 6 per launch is waited by a and b"),
        (lambda: Static("gap"), r"epoch 5 of 6 per launch is waited by no consumer"),
        (lambda: Static("past"), r"b waits on epoch 7, but prod fills only 6 per launch"),
        (lambda: Retired("wait_retired"), "b waits on epoch 1, but prod retires it unpublished"),
        # A persistent CTA's next item would start one epoch early.
        (
            lambda: PerItem(StaticPersistent(), "short"),
            "a advances 1 epochs per work item .*, but prod acquires 2",
        ),
    ],
    ids=[
        "join_skip",
        "bad_count",
        "bool_count",
        "token",
        "preview",
        "double",
        "gap",
        "past",
        "wait_retired",
        "per_item_short",
    ],
)
def test_misuse_raises_while_tracing(make, error):
    _raises_protocol(error, lambda: _compile(make()))


@pytest.mark.parametrize(
    "make, epochs",
    [
        (lambda: Retired("last"), "verified: 2 epochs per launch (a 1, b 1)\n"),
        (lambda: Retired("middle"), "verified: 2 epochs per launch (a 1, b 1)\n"),
        (
            lambda: PerItem(StaticPersistent(), "ok"),
            "verified: 2 epochs per work item (a 1, b 1)\n",
        ),
        # A launch-once prologue does not pair with per-item waits.
        (
            lambda: PerItem(OneShot(), "prologue"),
            "unverified (some roles use it once per work item, others once per launch)",
        ),
        # One item per CTA: b's trailing epoch needs no skip from a.
        (lambda: PerItem(OneShot(), "short"), "verified: 2 epochs per work item (a 1, b 1)\n"),
        # Used once per launch, the ring has no next item to line up, even under a persistent
        # schedule.
        (
            lambda: PerItem(StaticPersistent(), "launch"),
            "verified: 2 epochs per launch (a 1, b 1)\n",
        ),
        (lambda: Static("dynamic_count"), "unverified (a's skip count is dynamic)"),
        (lambda: Static("raw_state"), "unverified (b reads or assigns end.state)"),
        (
            Wide,
            (
                "verified: 6 epochs per launch (a 5, b 1); b's wait of epoch 5 is exact only "
                "after epoch 2 was filled, which it did not observe: order it with a rendezvous"
            ),
        ),
    ],
    ids=[
        "retired_last",
        "retired_middle",
        "per_item",
        "per_item_prologue",
        "one_shot_short",
        "launch_once",
        "dynamic_count",
        "raw_state",
        "needs_rendezvous",
    ],
)
def test_static_ownership_described(make, epochs):
    """A retired epoch is unowned; dynamic or raw skips leave ownership unverified."""
    op = make()
    _compile(op)
    assert f"epochs    {epochs}" in op.describe()


def test_static_ownership_is_verified_and_described(requires_blackwell):
    op = Static()
    fn = compile_toy(op)
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


# Runtime: split and grouped ownership


class Grouped(Toy):
    """Alternate ``group`` epochs owned by ``a`` with one owned by ``b``.

    If ``stages`` does not divide ``group + 1``, a stage changes owners between generations.
    With ``group >= stages``, b waits more than a ring lap after its previous wait, including
    its first wait. Its parity test is exact only after the stage's previous epoch was filled,
    which b did not observe. Without a rendezvous, the toy hangs.

    ``a_done`` keeps the roles within one window: a arrives after its group, b syncs before
    its epoch. ``b_done`` orders the reverse handoff. Like decode, a skips b's previous epoch
    based on the runtime round index.
    """

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


def check_owners(fn, ns, owner):
    """Check ownership with neither consumer delayed, then with each delayed, twice.

    Each launch starts from fresh barriers. ``ns`` maps launch arguments to epoch counts;
    epoch ``e`` must belong to ``owner(e, n)``.
    """
    for n, epochs in ns:
        for late in ("none", "a", "b"):
            for _ in range(2):
                out = torch.full((epochs + 1,), -1, device="cuda", dtype=torch.int32)
                out[-1] = 0
                fn(out, n, DELAY * (late == "a"), DELAY * (late == "b"))
                torch.cuda.synchronize()
                expected = [owner(e, n) for e in range(epochs)] + [0]  # then no wrong words
                assert out.tolist() == expected, (n, late)


def test_unequal_owners_with_overridden_counts_and_dynamic_skip(requires_blackwell):
    fn = compile_toy(Halves())
    check_owners(fn, [(n, n) for n in (0, 1, 9, 24)], lambda e, n: 1 if e < n // 2 else 2)


@pytest.mark.parametrize("stages, group", [(3, 2), (4, 2), (3, 4)])
def test_grouped_owners_with_rendezvous(requires_blackwell, stages, group):
    op = Grouped(stages, group)
    fn = compile_toy(op)
    assert "epochs    unverified (prod's acquire runs inside a dynamic loop or branch)" in (
        op.describe()
    )
    period = group + 1
    ns = [(n, period * n) for n in (0, 1, 5, 11)]
    check_owners(fn, ns, lambda e, n: 2 if e % period == group else 1)


# Alias diagnostic


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
    assert status(op, "P ~ S") == expected
    if mode == "exclusive":
        assert "kv: exclusive ring, epoch owners follow runtime skips" in alias_section(op)
