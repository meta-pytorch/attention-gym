import math
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from attn_gym.testing.warpdrive import raises_protocol
from attn_gym.warpdrive import (
    Barriers,
    Consumer,
    CtaPair,
    Instance,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    Slots,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Turn,
    Turnstile,
    Umma,
    WarpSpecialized,
    WorkStream,
    role,
)
from attn_gym.warpdrive import kernel as kernel_module
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.gemm import Gemm, compile_gemm, gemm
from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm, persistent_gemm

# A two-role toy kernel: producer warp 0 -> consumer warp 1 over two fixed slots with a
# layer-owned SMEM payload. ``mode`` injects protocol violations at trace time.


class Toy(WarpSpecialized):
    """Launches ``grid_ctas(n)`` CTAs whose roles get ``context(out, n)``."""

    def context(self, out, n):
        return SimpleNamespace(out=out, n=n)

    def grid_ctas(self, n):
        return 1

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(self.context(out, n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(f"warpdrive_{type(self).__name__.lower()}")
        grid = (self.grid_ctas(n), 1, 1)
        self.kernel(out, n).launch(grid=grid, block=(self.num_threads, 1, 1), stream=stream)


class SlotsToy(Toy):
    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.ch = Slots(2, Threads, Threads, producer_role="prod", payload=(cutlass.Int32, (32,)))

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        lane = cute.arch.lane_idx()
        if cutlass.const_expr(self.mode == "leak"):
            ch.acquire(0)
        if cutlass.const_expr(self.mode == "cross_region"):
            early = ch.acquire(0)
        for i in cutlass.range(ctx.n, unroll=1):
            for j in cutlass.range_constexpr(2):
                if cutlass.const_expr(self.mode == "cross_region" and j == 0):
                    early.commit()
                else:
                    h = ch.acquire(j)
                    h.data[lane] = (2 * i + j) * 32 + lane
                    h.commit()
                    if cutlass.const_expr(self.mode == "double"):
                        h.commit()
                    if cutlass.const_expr(self.mode == "wrong_side"):
                        h.release()
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        lane = cute.arch.lane_idx()
        acc = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            for j in cutlass.range_constexpr(2):
                with ch.consume(j) as h:
                    acc = acc + h.data[lane]
        ctx.out[lane] = acc
        return ch


class TurnToy(Toy):
    """Two role instances log their seats in turn.

    The role renames its seat; teardown must use the returned one. ``missing`` returns nothing.
    """

    def __init__(self, missing: bool = False):
        self.ts = Turnstile()
        self.missing = missing

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def worker(self, seat: Instance, ts: Turn, ctx):
        current = ts
        ts = None
        for _ in cutlass.range(ctx.n, unroll=1):
            with current.turn():
                if cute.arch.lane_idx() == 0:
                    pos = ctx.out[0]
                    ctx.out[pos + 1] = seat
                    ctx.out[0] = pos + 1
                    cute.arch.fence_acq_rel_cta()
        if cutlass.const_expr(self.missing):
            return None
        return current


class BoundToy(Toy):
    """Two producer instances fill separate slots; one consumer reads both.

    A ``roles=`` override expands the single-warp default. Dependencies are keyword-only,
    mixed with positional state.
    """

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.roles = {"prod": RoleConfig([range(1), range(1, 2)])}
        self.ch = Slots(2, Threads, Threads, payload=(cutlass.Int32, (32,)))

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, stage: Instance, ch: Producer[Instance]):
        slot = 1 - stage if cutlass.const_expr(self.mode == "wrong_slot") else stage
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce(slot) as h:
                h.data[cute.arch.lane_idx()] = (stage + 1) * (i + 1)
        return ch

    @role(warps=2)
    @cute.jit
    def cons(self, ch: Consumer, *, ctx):
        lane = cute.arch.lane_idx()
        for _ in cutlass.range(ctx.n, unroll=1):
            for j in cutlass.range_constexpr(2):
                with ch.consume(j) as h:
                    ctx.out[lane + 32 * j] = ctx.out[lane + 32 * j] + h.data[lane]
        return ch


class HoleToy(Toy):
    """Producer warp 0 feeds consumer warps 4-7; a raw idle role occupies warps 8-11.

    Unassigned warps 1-3 share warp 0's register budget: setmaxnreg is .aligned per
    warpgroup. At 12 warps the default is 168 registers, so producer and idle roles
    decrease their budgets and the consumer increases its budget.
    """

    def __init__(self):
        self.ch = Slots(1, Threads, Threads, payload=(cutlass.Int32, (32,)))

    @role(warps=0, regs=56)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce(0) as h:
                h.data[lane] = i * 32 + lane
        return ch

    @role(warps=range(4, 8), regs=232)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        lane = cute.arch.lane_idx()
        acc = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(0) as h:
                acc = acc + h.data[lane]
        if cute.arch.warp_idx() == 4:
            ctx.out[lane] = acc
        return ch

    @role(warps=range(8, 12), regs=56, raw=True)
    @cute.jit
    def idle(self, ctx):
        pass


class OverBudgetHoleToy(HoleToy):
    """Warp 0's 232 registers also bind holes 1-3: 4 warpgroups' worth over 65536."""

    @role(warps=0, regs=232)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        return ch


class FullPoolToy(HoleToy):
    """232 + 232 + ``idle`` registers per thread over three warpgroups: with 40 the budgets use
    exactly the 168 x 384 registers 12 warps start with (CUTLASS's split)."""

    @role(warps=0, regs=232)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce(0) as h:
                h.data[lane] = i * 32 + lane
        return ch

    @role(warps=range(8, 12), regs=40)
    @cute.jit
    def idle(self, ctx):
        pass


class PoolOverflowToy(FullPoolToy):
    """A 48-register idle group exceeds the launch's register pool.

    The budgets total 65536, but 12 warps start with 168 x 384 = 64512 registers.
    setmaxnreg.inc can take only registers other warps released; the increase would hang.
    """

    @role(warps=range(8, 12), regs=48)
    @cute.jit
    def idle(self, ctx):
        pass


@jit_cache
def compile_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name="warpdrive_toy")


def test_slots_payload_round_trip(requires_blackwell):
    fn = compile_toy(SlotsToy())
    lane = torch.arange(32, device="cuda", dtype=torch.int32)
    for n in (1, 7):  # one pass, then wraps of both slots, on one compiled kernel
        out = torch.full((32,), -1, device="cuda", dtype=torch.int32)
        fn(out, n)
        expected = sum((2 * i + j) * 32 for i in range(n) for j in range(2)) + 2 * n * lane
        torch.testing.assert_close(out, expected.to(torch.int32))


def test_turnstile_alternates_seats(requires_blackwell):
    fn = compile_toy(TurnToy())
    for n in (1, 5):
        log = torch.zeros(2 * n + 1, device="cuda", dtype=torch.int32)
        fn(log, n)
        assert log.tolist() == [2 * n] + [0, 1] * n


def test_bound_producers_fan_in(requires_blackwell):
    fn = compile_toy(BoundToy())
    for n in (1, 9):
        out = torch.zeros(64, device="cuda", dtype=torch.int32)
        fn(out, n)
        tri = n * (n + 1) // 2
        assert out.tolist() == [tri] * 32 + [2 * tri] * 32


# TMEM allocator placement: one CTA allocates from a warp other than warp 0 (which
# initializes the barriers) when a TMEM role has one. Every toy takes all 512 columns, so a
# CTA that did not free them would block the next CTA on its SM in tcgen05.alloc.


class TmemPlacementToy(Toy):
    """Each warp of a TMEM role marks ``out[cta, warp]``."""

    def __init__(self):
        self.tmem_plan = TmemPlan().region("acc", 512)

    def grid_ctas(self, n):
        return n


@cute.jit
def mark_warp(ctx):
    """``out[cta, warp] = 1`` for the calling warp."""
    if cute.arch.lane_idx() == 0:
        ctx.out[cute.arch.block_idx()[0] * 4 + cute.arch.warp_idx()] = cutlass.Int32(1)


class TmemOnWarpZero(TmemPlacementToy):
    """Only warp 0 uses TMEM: it allocates (the fallback)."""

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, tmem: Tmem):
        mark_warp(ctx)

    @role(warps=1)
    @cute.jit
    def other(self, ctx):
        pass


class TmemOnWarpgroup(TmemPlacementToy):
    """One TMEM role on warps 0-3: its warp 1 allocates."""

    @role(warps=range(4))
    @cute.jit
    def epilogue(self, ctx, *, tmem: Tmem):
        mark_warp(ctx)


class TmemOnTwoRoles(TmemPlacementToy):
    """TMEM roles on warps 0 and 2: the second role's warp 2 allocates and frees."""

    @role(warps=0)
    @cute.jit
    def mma(self, ctx, *, tmem: Tmem):
        mark_warp(ctx)

    @role(warps=1)
    @cute.jit
    def other(self, ctx):
        pass

    @role(warps=2)
    @cute.jit
    def epilogue(self, ctx, *, tmem: Tmem):
        mark_warp(ctx)


@pytest.mark.parametrize(
    "toy, allocator, warps",
    [
        (TmemOnWarpZero, ("mma", 0), [1, 0, 0, 0]),
        (TmemOnWarpgroup, ("epilogue", 1), [1, 1, 1, 1]),
        (TmemOnTwoRoles, ("epilogue", 2), [1, 0, 1, 0]),
    ],
)
def test_tmem_allocator_avoids_the_barrier_warp(
    requires_blackwell, traced_compiles, monkeypatch, toy, allocator, warps
):
    """The traced kernel allocates from the expected warp, inside the role that frees, and
    every CTA frees its columns (the CTAs sharing an SM would otherwise wait forever)."""
    allocator_warps = []
    real = kernel_module.TmemAllocator

    def spy(*args, **kwargs):
        allocator_warps.append(kwargs["allocator_warp_id"])
        return real(*args, **kwargs)

    monkeypatch.setattr(kernel_module, "TmemAllocator", spy)
    op = toy()
    compile_toy.cache_clear()
    fn = compile_toy(op)
    monkeypatch.undo()
    assert (op._ws_tmem_alloc_role.name, allocator_warps) == (allocator[0], [allocator[1]])
    ctas = 4 * torch.cuda.get_device_properties(0).multi_processor_count  # several per SM
    out = torch.zeros(ctas * 4, device="cuda", dtype=torch.int32)
    fn(out, ctas)
    torch.cuda.synchronize()
    assert out.view(ctas, 4).tolist() == [warps] * ctas


@pytest.mark.parametrize("toy", [HoleToy, FullPoolToy])
def test_register_budgets_run(requires_blackwell, toy):
    """Holes take their warpgroup's budget; FullPoolToy's budgets use exactly the pool."""
    if toy is HoleToy:
        assert "  1-3    (padding)    56\n" in HoleToy().describe()
    n = 9
    out = torch.full((32,), -1, device="cuda", dtype=torch.int32)
    compile_toy(toy())(out, n)
    lane = torch.arange(32, dtype=torch.int32)
    assert torch.equal(out.cpu(), 32 * n * (n - 1) // 2 + n * lane)


def test_register_budget_branches_are_warpgroup_uniform(monkeypatch, traced_compiles):
    """Register adjustments must be warpgroup-uniform and precede every role's IKET range.

    setmaxnreg is .sync.aligned per warpgroup. Emit one adjustment per run of adjacent
    warpgroups with the same budget, including raw roles. HoleToy's 56 / 232 / 56 split
    needs three branches: ptxas allocates a merged branch jointly, so merging non-adjacent
    warpgroups can spill registers. Compile only.
    """
    from attn_gym.warpdrive import tracing

    events = []
    for fn in ("setmaxregister_increase", "setmaxregister_decrease"):
        real = getattr(cute.arch, fn)
        monkeypatch.setattr(
            cute.arch, fn, lambda regs, *a, _real=real, **kw: (events.append(regs), _real(regs))
        )
    iket = tracing.iket

    def push(name):
        events.append(name)
        return iket.range_push(name)

    monkeypatch.setattr(
        tracing, "iket", SimpleNamespace(range_push=push, range_pop=iket.range_pop)
    )
    compile_toy.cache_clear()  # trace even if another test compiled the same key
    compile_toy(HoleToy().enable_iket(waits=False))
    assert [e for e in events if isinstance(e, int)] == [56, 232, 56]
    regs_of = {"prod": 56, "cons": 232, "idle": 56}
    current, seen = None, set()
    for e in events:
        if isinstance(e, int):
            current = e
        elif e in regs_of:
            assert current == regs_of[e], (e, events)
            seen.add(e)
    assert seen == set(regs_of)


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: SlotsToy("leak"), r"handles never closed: ch\[0\] acquire opened at test_"),
        (lambda: SlotsToy("double"), r"ch\[[01]\] acquire closed twice: commit at .*, then"),
        (lambda: SlotsToy("cross_region"), r"different dynamic region"),
        (lambda: SlotsToy("wrong_side"), r"release\(\) on a ch\[[01]\] acquire handle"),
        (lambda: BoundToy("wrong_slot"), r"bound to slot 0, but acquire\(1\)"),
        (lambda: TurnToy(missing=True), r"return all warpdrive state arguments; missing ts"),
        (
            lambda: TypedCtxToy("namespace"),
            r"run_roles got a SimpleNamespace context, but \w+\.ctx is annotated ToyCtx",
        ),
        (
            lambda: NamespaceCtxToy("mutable"),
            r"MutableToyCtx is a mutable dataclass; declare it @dataclass\(frozen=True\)",
        ),
        (
            lambda: NamespaceCtxToy("tuple"),
            r"context TupleCtx has no attributes to share; pass a frozen dataclass",
        ),
    ],
    ids=[
        "leak",
        "double",
        "cross_region",
        "wrong_side",
        "wrong_slot",
        "missing_return",
        "ctx_namespace",
        "ctx_mutable",
        "ctx_tuple",
    ],
)
def test_trace_time_errors(make, message):
    """Protocol and context errors raised while tracing a role body or run_roles."""
    raises_protocol(message, lambda: compile_toy(make()))


# Role context: a typed frozen dataclass, or a namespace


@dataclass(frozen=True)
class ToyCtx:
    out: cute.Tensor
    n: cutlass.Int32


@dataclass
class MutableToyCtx:
    out: cute.Tensor
    n: cutlass.Int32


class TupleCtx(NamedTuple):
    out: cute.Tensor
    n: cutlass.Int32


@dataclass(frozen=True)
class OtherCtx:
    out: cute.Tensor


class TypedCtxToy(Toy):
    """Producer warp 0 feeds consumer warp 1 over one slot with ``ctx: ToyCtx``.

    The consumer writes through the context inside a dynamic branch. ``ctx`` selects the
    value passed to run_roles; ``seen`` records each role's context type at trace time.
    """

    def __init__(self, ctx: str = "typed"):
        self.ctx = ctx
        self.seen = []
        self.ch = Slots(1, Threads, Threads, payload=(cutlass.Int32, (32,)))

    def context(self, out, n):
        cls = {"typed": ToyCtx, "mutable": MutableToyCtx, "tuple": TupleCtx}
        return cls.get(self.ctx, SimpleNamespace)(out=out, n=n)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx: ToyCtx):
        self.seen.append(type(ctx))
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce(0) as h:
                h.data[lane] = i * 32 + lane
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx: ToyCtx):
        self.seen.append(type(ctx))
        lane = cute.arch.lane_idx()
        acc = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(0) as h:
                acc = acc + h.data[lane]
        if ctx.n > 0:
            ctx.out[lane] = acc
        return ch


class NamespaceCtxToy(TypedCtxToy):
    """TypedCtxToy with unannotated contexts, which accept any context type (traced only)."""

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        return ch


def test_typed_context_reaches_roles(requires_blackwell, traced_compiles):
    """A frozen dataclass reaches the roles as itself (the other toys pass a namespace)."""
    op = TypedCtxToy()
    compile_toy.cache_clear()  # the roles record their context types while tracing
    out = torch.full((32,), -1, device="cuda", dtype=torch.int32)
    compile_toy(op)(out, 5)
    lane = torch.arange(32, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(out, 32 * 10 + 5 * lane)
    assert op.seen == [ToyCtx, ToyCtx]


class KeywordRawToy(Toy):
    """A raw role taking its context positionally and its barriers keyword-only."""

    def __init__(self):
        self.scratch = Barriers(1, counts=1)

    def context(self, out, n):
        return ToyCtx(out, n)

    @role(warps=0, raw=True)
    @cute.jit
    def worker(self, ctx: ToyCtx, *, scratch: Barriers):
        lane = cute.arch.lane_idx()
        ctx.out[lane] = ctx.n + lane


def test_raw_role_keyword_dependency(requires_blackwell):
    out = torch.empty(32, device="cuda", dtype=torch.int32)
    compile_toy(KeywordRawToy())(out, 3)
    torch.testing.assert_close(out, torch.arange(32, device="cuda", dtype=torch.int32) + 3)


class TypoAnnotationToy(WarpSpecialized):
    def __init__(self):
        self.ab = Ring(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def load(self, ab: Producer, ctx: ToyCtx):
        return ab

    @role(warps=1)
    @cute.jit
    def mma(self, ab: "Consumr", ctx: ToyCtx):  # noqa: F821 (the typo under test)
        pass


class LookalikeToy(WarpSpecialized):
    def __init__(self):
        self.work = WorkStream()

    @role(warps=0)
    @cute.jit
    def worker(self, work: WorkStream, ctx):
        pass


class TwoContextTypesToy(TypedCtxToy):
    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx: OtherCtx):
        pass


class MutableAnnotationToy(TypedCtxToy):
    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx: MutableToyCtx):
        pass


class TwoUnannotatedToy(TypedCtxToy):
    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, extra, ctx):
        pass


class VarArgsToy(WarpSpecialized):
    @role(warps=0)
    @cute.jit
    def worker(self, *args):
        pass


# Host-side declaration checks (no compile)


class WrongRoleToy(SlotsToy):
    def __init__(self):
        self.mode = "ok"
        self.ch = Slots(2, Threads, Threads, producer_role="cons")


class BoundSlotsToy(WarpSpecialized):
    """Two instances, each bound to its own slot of ``ch``; one consumer of both."""

    def __init__(self, n=2, producer=Threads):
        self.ch = Slots(n, producer, Threads)

    @role(warps=[range(4), range(4, 8)])
    @cute.jit
    def prod(self, stage: Instance, ch: Producer[Instance], ctx):
        return ch

    @role(warps=range(8, 12))
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        return ch


class WideRolesToy(WarpSpecialized):
    """One producer role on warps 0-3, one consumer role on warps 4-7."""

    def __init__(self, producer=Threads, consumer=Threads):
        self.ch = Ring(2, producer, consumer)

    @role(warps=range(4))
    @cute.jit
    def prod(self, ch: Producer, ctx):
        return ch

    @role(warps=range(4, 8))
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        return ch


def test_multi_warp_arrival_counts():
    """Threads sides count every thread of a multi-warp (or bound multi-instance) role."""
    for toy in (BoundSlotsToy(), WideRolesToy()):
        toy.describe()
        assert toy.ch.slot_counts("producer") == [128, 128]
        assert toy.ch.slot_counts("consumer") == [128, 128]


def test_tmem_plan_columns():
    plan = (
        TmemPlan().region("S", 128, 2).region("O", 128, 2).alias("P", of="S", offset=64, cols=64)
    )
    assert [plan.col("P", j) for j in range(2)] == [64, 192]
    assert [plan.col("O", j) for j in range(2)] == [256, 384]
    assert plan.alloc_cols == 512


PROTOCOL_ERRORS = {
    # Only the context-typed or single unannotated parameter receives the context; anything
    # else is an error naming the parameter, not a silent context injection.
    "typo": (TypoAnnotationToy, r"role mma: parameter ab is annotated 'Consumr', which does not"),
    "lookalike": (LookalikeToy, r"work is annotated WorkStream.*did you mean `work: WorkCursor`"),
    "two_types": (TwoContextTypesToy, r"unrelated types \(cons\.ctx: OtherCtx, prod\.ctx: ToyC"),
    "two_unannotated": (TwoUnannotatedToy, r"role cons: parameters \['extra', 'ctx'\] would all"),
    "mutable_ctx": (MutableAnnotationToy, r"cons\.ctx is annotated MutableToyCtx, which is not a"),
    "varargs": (VarArgsToy, r"variadic parameter .* cannot be injected"),
    "wrong_role": (WrongRoleToy, "producer is declared as role 'cons'"),
    "slot_past_n": (lambda: BoundSlotsToy(n=1), r"slot 1 >= ch\.n=1"),
    "bound_umma": (lambda: BoundSlotsToy(producer=Umma), "signals once"),
    # Tma/Umma sides expect one arrival; each warp of a wider role would signal.
    "wide_tma_producer": (lambda: WideRolesToy(Tma(1024), Umma), "role prod spans 4 warps"),
    "wide_umma_producer": (lambda: WideRolesToy(Umma), "role prod spans 4 warps"),
    "wide_umma_consumer": (lambda: WideRolesToy(Threads, Umma), "role cons spans 4 warps"),
}

VALUE_ERRORS = {
    # 232 x 32 x (4 + 4) + 56 x 32 x 4 = 66560: counting warp 0 alone would accept it.
    "hole_budget": (OverBudgetHoleToy, "register budgets need 66560 registers"),
    # The budgets share the default budget of every launched warp, not 65536.
    "pool_overflow": (PoolOverflowToy, r"need 65536 registers, but 12 warps start with 168"),
    "commit_fence": (lambda: Slots(2, Tma(1024), Umma, commit_fence="smem_async"), "no commit"),
    "tmem_alias": (
        lambda: TmemPlan().region("S", 128, 2).alias("P", of="S", offset=96, cols=64),
        "exceeds S",
    ),
    "tmem_cols": (lambda: TmemPlan().region("S", 128, 5).alloc_cols, "needs 640 > 512"),
    "gemm_tile_n": (lambda: Gemm(tile_n=48), "multiple of 32"),
    "pgemm_epi": (lambda: PersistentGemm(epi_tile_n=128), "epi_tile_n must be one of"),
    "pgemm_tile_n": (
        lambda: PersistentGemm(tile_n=224, epi_tile_n=64),
        "multiple of epi_tile_n=64",
    ),
    "pgemm_c_stages_0": (lambda: PersistentGemm(c_stages=0), "c_stages must be in"),
    "pgemm_c_stages_9": (lambda: PersistentGemm(c_stages=9), "c_stages must be in"),
    "pgemm_mixed_budgets": (  # load shares the warpgroup of mma
        lambda: PersistentGemm(roles={"load": RoleConfig(4, regs=48)}),
        "mixed register budgets",
    ),
    "pgemm_early_release": (lambda: PersistentGemm(early_release=1), "must be a bool"),
}


@pytest.mark.parametrize(
    "error, make, message",
    [(ProtocolError, *case) for case in PROTOCOL_ERRORS.values()]
    + [(ValueError, *case) for case in VALUE_ERRORS.values()],
    ids=[*PROTOCOL_ERRORS, *VALUE_ERRORS],
)
def test_declaration_errors(error, make, message):
    """Host-side declaration errors, raised by the constructor or by describe()."""
    with pytest.raises(error, match=message):
        make().describe()


# GEMM example


def check_gemm(fn, m, n, k, **config):
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    err = (fn(a, b, **config).float() - ref).abs().max().item()
    assert err <= 2 * torch_err + 1e-3, (config, err, torch_err)


@pytest.mark.parametrize(
    "m, n, k", [(128, 256, 64), (300, 520, 136), (1, 8, 8), (1000, 1000, 1000)]
)
def test_gemm(requires_blackwell, m, n, k):
    check_gemm(gemm, m, n, k)


@pytest.mark.parametrize("tile_n", [64, 192])
def test_gemm_tile_n(requires_blackwell, tile_n):
    """Narrower tiles: the TMEM region, A/B stage bytes and epilogue subtile count follow."""
    fn = compile_gemm(Gemm(tile_n=tile_n))

    def run(a, b):
        out = a.new_empty(a.shape[0], b.shape[0])
        fn(a, b, out)
        return out

    check_gemm(run, 300, 520, 136)


PGEMM_CONFIGS = [
    {"c_stages": 1},
    {"epi_tile_n": 64},
    {"epi_tile_n": 16, "c_stages": 4},
    {"tile_n": 224, "epi_tile_n": 16},
    {"acc_stages": 1},
    {"early_release": False},
    {"ab_stages": 2},
]


@pytest.mark.parametrize(
    "config", PGEMM_CONFIGS, ids=["_".join(f"{k}{v}" for k, v in c.items()) for c in PGEMM_CONFIGS]
)
def test_persistent_gemm_configs(requires_blackwell, config):
    """Vary epilogue and stage settings separately on a ragged multi-wave shape.

    Role placements are covered in test_warpdrive_role_config.py.
    """
    check_gemm(persistent_gemm, 1000, 1000, 264, **config)


def test_persistent_gemm_wrapper_keys_roles_by_content(requires_blackwell):
    """Two role dicts with the same entries in another order reuse one kernel."""
    from attn_gym.warpdrive.examples import persistent_gemm as pg

    load, mma = RoleConfig(4, regs=48), RoleConfig(5, regs=48)
    before = len(pg._OPS)
    check_gemm(persistent_gemm, 256, 256, 64, roles={"load": load, "mma": mma})
    assert len(pg._OPS) == before + 1
    check_gemm(persistent_gemm, 256, 256, 64, roles={"mma": mma, "load": load})
    assert len(pg._OPS) == before + 1


def test_persistent_gemm_config_reaches_names_and_keys():
    tuned = PersistentGemm(roles={"epilogue": RoleConfig(range(4), regs=152)})
    assert "0-3    epilogue    152" in tuned.describe()
    assert tuned.__cute_cache_key__() != PersistentGemm().__cute_cache_key__()
    assert PersistentGemm(c_stages=3).get_name() != PersistentGemm().get_name()
    assert "_late_" in PersistentGemm(early_release=False).get_name()


GEMMS = pytest.mark.parametrize("fn", [gemm, persistent_gemm], ids=lambda f: f.__name__)


@GEMMS
@pytest.mark.parametrize("m, n, k", [(64, 24, 0), (0, 8, 64)])
def test_gemm_zero_work(requires_blackwell, fn, m, n, k):
    """Empty K gives zeros (no MMA writes the accumulator); empty outputs launch nothing."""
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    assert fn(a, b, out) is out
    assert torch.equal(out, torch.zeros_like(out))


@GEMMS
def test_gemm_out_must_not_overlap_operands(requires_blackwell, fn):
    """512^3 has 8 output tiles: a tile's store would overwrite rows another tile still loads.
    An ``out`` adjacent to both operands is accepted."""
    size = 512
    buf = torch.randn(3 * size * size, device="cuda", dtype=torch.bfloat16)
    a, out, b = (buf[i * size * size : (i + 1) * size * size].view(size, size) for i in range(3))
    for alias in (b, buf[size * size // 2 :][: size * size].view(size, size)):
        with pytest.raises(ValueError, match="out must not overlap a or b"):
            fn(a, b, alias)
    ref = torch.nn.functional.linear(a.float(), b.float())
    fn(a, b, out)
    assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item()


@GEMMS
@pytest.mark.parametrize(
    "m, n, k, out, error",
    [
        (2, 8, 0, torch.ones(1), TypeError),  # float32 out for an empty-K problem
        (2, 8, 0, torch.ones(1, dtype=torch.bfloat16), ValueError),  # wrong shape, empty K
        (0, 8, 16, torch.empty(3, 8, dtype=torch.bfloat16), ValueError),  # wrong shape, M = 0
        (4, 8, 0, torch.empty(8, 4, dtype=torch.bfloat16).t(), ValueError),  # transposed out
    ],
    ids=["dtype_empty_k", "shape_empty_k", "shape_empty_m", "layout"],
)
def test_gemm_checks_out_before_zero_work_returns(fn, m, n, k, out, error):
    """Zero-work problems return before dispatch; ``out`` still has to fit the contract."""
    a = torch.zeros(m, k, dtype=torch.bfloat16)
    b = torch.zeros(n, k, dtype=torch.bfloat16)
    with pytest.raises(error, match="out must"):
        fn(a, b, out)


# FA4-style attention forward example


def fa4_inputs(b, h, sq, sk, head_dim=128):
    """Distinct magnitude per 128-row Q tile (slot/stage mixups), growing K blocks (forces
    O rescales), per-key-block V offsets, per-head scale."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, s, h, head_dim, device="cuda") for s in (sq, sk, sk))
    q = q * (0.5 + 0.75 * (torch.arange(sq, device="cuda") // 128 % 4)).view(1, -1, 1, 1)
    kblk = torch.arange(sk, device="cuda") // 128
    k = k * (0.5 + 0.25 * kblk).view(1, -1, 1, 1)
    v = v + 0.5 * kblk.view(1, -1, 1, 1)
    q = q * (1 + torch.arange(h, device="cuda")).view(1, 1, -1, 1) ** 0.5
    return tuple(x.to(torch.bfloat16) for x in (q, k, v))


def check_fa4(q, k, v, **options):
    from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

    o, lse = fa4_fwd(q, k, v, **options)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)
    ref_lse = torch.logsumexp((qf @ kf.transpose(-1, -2)) / math.sqrt(q.shape[-1]), dim=-1)
    # bf16 output rounding bounds the error: half an ulp of the largest |O|, plus slack for
    # bf16 P in the PV MMA.
    tol = 2**-7 * ref.abs().max().item() + 4e-3
    assert (o.float() - ref).abs().max().item() <= tol
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)
    return o


@pytest.mark.parametrize("turnstile", [False, True])
@pytest.mark.parametrize("b, h, sq, sk", [(1, 1, 256, 128), (1, 1, 256, 1024), (2, 3, 640, 512)])
def test_fa4_fwd(requires_blackwell, b, h, sq, sk, turnstile):
    pytest.importorskip("flash_attn.cute")
    check_fa4(*fa4_inputs(b, h, sq, sk), turnstile=turnstile)


FA4_CONFIGS = [
    {"tile_n": 64},
    {"tile_n": 64, "split_p": True},
    {"head_dim": 64},
    {"head_dim": 64, "tile_n": 64, "cluster": CtaPair()},
    {"rescale_threshold": 0.0},
    {"kv_stages": 2},
]


@pytest.mark.parametrize(
    "options", FA4_CONFIGS, ids=["_".join(f"{k}{v}" for k, v in c.items()) for c in FA4_CONFIGS]
)
def test_fa4_fwd_configs(requires_blackwell, options):
    """Vary tile, head-dim, stage, and threshold settings separately on ragged Q lengths.

    Inputs force rescales. Role placements are covered in test_warpdrive_role_config.py.
    """
    pytest.importorskip("flash_attn.cute")
    options = dict(options)
    head_dim = options.pop("head_dim", 128)
    sk = 4 * options.get("tile_n", 128)
    check_fa4(*fa4_inputs(2, 3, 640, sk, head_dim=head_dim), **options)


def test_fa4_rejects_unfit_configurations(requires_blackwell):
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import FlashAttentionForward, fa4_fwd

    with pytest.raises(ValueError, match="tile_n must be one of"):
        FlashAttentionForward(tile_n=256)  # 2 x (256 + 128) TMEM columns would not fit anyway
    with pytest.raises(ValueError, match="head_dim must be one of"):
        FlashAttentionForward(head_dim=96)
    with pytest.raises(ValueError, match="rescale_threshold"):
        FlashAttentionForward(rescale_threshold=-1.0)
    small = FlashAttentionForward(tile_n=64, head_dim=64)
    assert small.tmem_plan.alloc_cols == 256 and small.kv.producer.tx_bytes == 64 * 64 * 2
    assert small.get_name() != FlashAttentionForward().get_name()
    q = torch.randn(1, 128, 1, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 200, 1, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 128"):
        fa4_fwd(q, k, k)


def test_fa4_declared_graph():
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import FlashAttentionForward

    op = FlashAttentionForward()
    summary = op.describe()
    # Fan-out/join: S_j is consumed by softmax[j] and correction, so its empty barrier waits
    # for 128 + 128 thread arrivals; the producer side is one tcgen05.commit.
    assert op.s.slot_counts("consumer") == [256, 256]
    assert op.s.slot_counts("producer") == [1, 1]
    assert op.stats.slot_counts("producer") == [128, 128]
    assert op.num_threads == 512
    assert re.search(r"\n  4-7 +softmax\[1\] +192  all\n", summary), summary


def test_describe_lists_the_traced_protocol(requires_blackwell):
    """With trace_protocol enabled, describe() lists operations, source lines, and TMEM regions.

    Channel operations are indented by dynamic region.
    """
    op = Gemm(trace=True)
    assert "Traced protocol" not in op.describe()  # nothing traced before the compile
    compile_gemm(op)
    text = op.describe()
    protocol = text[text.index("Traced protocol\n") :]
    assert re.search(r"\n  role load:\n +ab\[i\] +acquire +@ gemm\.py:\d+\n", protocol), protocol
    assert re.search(r"\n    epilogue: acc\[0\]", protocol), protocol
    assert all(len(line) <= 100 for line in text.splitlines())
