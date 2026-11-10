"""Probe declarations with a seeded oracle independent of scope derivation.

Each case changes one property. E labels mark required rejections; G labels mark
domain-hardening targets, not runtime-valid kernels.
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from types import FunctionType

from cutlass import cute

from attn_gym.warpdrive import (
    AtomicPersistent,
    Barriers,
    ClusterLaunchControl,
    Consumer,
    CtaPair,
    Instance,
    Producer,
    ProtocolError,
    Raster,
    Ring,
    SingleCta,
    Slots,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Turn,
    Turnstile,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
    unchecked,
)


@dataclass(frozen=True)
class Finding:
    seed: int
    case: str
    contract: str
    classification: str
    detail: str


def is_expected_error(exc: Exception, error: type[Exception], message: str) -> bool:
    """``exc`` or an exception in its cause/context chain is ``error`` with ``message``."""
    seen = set()
    pending = [exc]
    while pending:
        cause = pending.pop()
        if id(cause) in seen:
            continue
        seen.add(id(cause))
        if isinstance(cause, error) and message in str(cause):
            return True
        pending.extend(e for e in (cause.__cause__, cause.__context__) if e is not None)
    return False


@dataclass
class FuzzReport:
    seed: int
    cases: int = 0
    accepted: int = 0
    rejected: int = 0
    obligations: int = 0
    findings: list[Finding] = field(default_factory=list)

    def record(self, case, exc: Exception | None) -> None:
        """Classify a case against its expected exception, case.error.

        ``case`` is a DeclarationCase or Mutation; ``exc`` is the raised exception, or None.
        """
        self.cases += 1
        if exc is not None and case.error is not None:
            if is_expected_error(exc, case.error, case.message):
                self.rejected += 1
                return
        if exc is not None:
            kind, detail = "unexpected exception", f"{type(exc).__name__}: {exc}"
        elif case.error is not None:
            kind, detail = "missing rejection", f"expected {case.error.__name__}: {case.message}"
        else:
            if case.obligation:
                self.obligations += 1
            else:
                self.accepted += 1
            return
        self.findings.append(Finding(self.seed, case.name, case.contract, kind, detail))


@dataclass(frozen=True)
class DeclarationCase:
    name: str
    contract: str
    run: Callable[[], object]
    error: type[Exception] | None = None
    message: str = ""
    obligation: bool = False


def _body(self):
    pass


def declaration_role(warps=0, regs=None, cta="all", params=(), raw=False):
    """Build a fresh JIT-decorated role for host collection only (never compiled)."""
    fn = FunctionType(_body.__code__, globals(), "worker")
    fn.__signature__ = inspect.Signature(
        [
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            *(
                inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=a)
                for n, a in params
            ),
        ]
    )
    return role(warps=warps, regs=regs, cta=cta, raw=raw)(cute.jit(fn))


def collect_pair(
    channel,
    *,
    cluster=None,
    producer_warps=0,
    consumer_warps=1,
    consumer_cta="all",
    producer_ann=Producer,
):
    return collect_roles(
        {
            "prod": declaration_role(producer_warps, params=(("ch", producer_ann),)),
            "cons": declaration_role(consumer_warps, cta=consumer_cta, params=(("ch", Consumer),)),
        },
        {"ch": channel, "cluster": cluster or SingleCta()},
    )


def collect_roles(roles, declarations=None):
    cls = type("DeclarationProbe", (WarpSpecialized,), roles)
    op = cls()
    for name, value in (declarations or {}).items():
        setattr(op, name, value)
    return op.describe()


def collect_one(*params, raw=False, **declarations):
    """Describe a kernel with one role ``a`` taking ``params`` (name, annotation) pairs."""
    return collect_roles({"a": declaration_role(params=params, raw=raw)}, declarations)


def declaration_cases(seed: int) -> list[DeclarationCase]:
    """Boundary-biased mutations plus valid random depths, widths, budgets, and TMEM maps."""
    rng = random.Random(seed)
    cases = []

    def add(name, contract, fn, message="", error=ValueError):
        cases.append(DeclarationCase(name, contract, fn, error, message))

    def reject(name, contract, fn, message):
        add(name, contract, fn, message, ProtocolError)

    for value in ([], -1, 32, range(-1, 1), range(31, 33), True, 1.5, None):
        add(f"warps={value!r}", "G2", lambda v=value: declaration_role(v), "role warps")
    for value in (0, 23, 25, 257, 24.0, True, "40"):
        add(f"regs={value!r}", "G2", lambda v=value: declaration_role(0, v), "role regs")
    add("no roles", "G2", lambda: collect_roles({}), "at least one role")
    add(
        "overlap",
        "E03",
        lambda: collect_roles({"a": declaration_role(0), "b": declaration_role(0)}),
        "claimed by",
    )
    add(
        "mixed regs",
        "E04",
        lambda: collect_roles({"a": declaration_role(0, 40), "b": declaration_role(1, 48)}),
        "mixed register",
    )
    for value in (0, -1, 1.5, True):
        add(f"depth={value!r}", "E29/G3", lambda v=value: Ring(v), "stage")
        add(f"barriers={value!r}", "E35/G3", lambda v=value: Barriers(v), "Barriers")
        add(f"tx={value!r}", "G3", lambda v=value: Tma(v), "Tma.tx_bytes")
    for side in ("producer", "consumer"):
        for value in (object(), Threads(), Umma(), Tma):
            add(
                f"{side}={type(value).__name__}",
                "G3",
                lambda s=side, v=value: Ring(1, **{s: v}),
                "mechanism",
            )
    add("TMA consumer local", "E43/G3", lambda: Ring(1, Threads, Tma(16)), "producer mechanism")
    for value in ([1.5], [True], [0], [1, 2], "x", 1.5):
        add(f"counts={value!r}", "E36/G3", lambda v=value: Barriers(1, counts=v), "positive ints")
    for kind in ("commit_fence", "release_fence"):
        for value in ("wrong", ("smem_async", "wrong"), ("smem_async",) * 2, ["smem_async"]):
            add(f"bad {kind}={value!r}", "E30", lambda k=kind, v=value: Ring(1, **{k: v}), kind)
    for value in ("smem_async", ("tmem_store", "smem_async")):
        add(
            f"TMA commit fence {value!r}",
            "E31",
            lambda v=value: Ring(1, Tma(16), commit_fence=v),
            "no commit to fence",
        )
    add("bad remote scope", "E32", lambda: Ring(1, remote_scope="gpu"), "remote_scope")
    add("unknown override", "E33", lambda: Ring(1, override={"typo": 1}), "unknown")
    add("too many slots", "E34", lambda: Slots(32), "at most 31")
    for value in (0, -1, 1.5, True):
        m = "positive int"
        add(f"region cols={value!r}", "E48/G4", lambda v=value: TmemPlan().region("a", v), m)
        add(f"region slots={value!r}", "E48/G4", lambda v=value: TmemPlan().region("a", 32, v), m)
        add(
            f"alias cols={value!r}",
            "G4",
            lambda v=value: TmemPlan().region("a", 32).alias("b", of="a", offset=0, cols=v),
            m,
        )
    add(
        "duplicate region",
        "G4",
        lambda: TmemPlan().region("a", 32).region("a", 16),
        "already declared",
    )
    add(
        "self alias",
        "G4",
        lambda: TmemPlan().region("a", 32).alias("a", of="a", offset=0, cols=16),
        "already declared",
    )
    reject(
        "unknown parent",
        "G4",
        lambda: TmemPlan().alias("b", of="missing", offset=0, cols=16),
        "no TMEM region",
    )
    add(
        "alias overflow",
        "E49",
        lambda: TmemPlan().region("a", 32).alias("b", of="a", offset=31, cols=2),
        "exceeds",
    )
    add("TMEM capacity", "E52", lambda: TmemPlan().region("a", 513).alloc_cols, "513 > 512")
    reject("missing TMEM", "E28", lambda: collect_one(("tmem", Tmem)), "no tmem_plan")
    work = ("work", WorkCursor)
    reject("missing WorkCursor", "E68/G7", lambda: collect_one(work), "no WorkStream named")
    for schedule in (AtomicPersistent(), ClusterLaunchControl()):
        reject(
            f"{type(schedule).__name__} channel collision",
            "G7",
            lambda p=schedule: collect_one(work, work=WorkStream(schedule=p), work_ring=Ring(1)),
            "reserved",
        )
    shared = Ring(1)
    reject(
        "same channel twice",
        "G7",
        lambda: collect_one(("a", Producer), ("b", Consumer), a=shared, b=shared),
        "same declaration",
    )
    for value in (-1, 32, True, 1.5):
        add(
            f"scheduler_warp={value!r}",
            "G2",
            lambda v=value: WorkStream(scheduler_warp=v),
            "scheduler_warp",
        )
    add("bad along", "E82", lambda: Raster(along="k"), "along")
    for value in (0, 1.5, True):
        add(f"swizzle={value!r}", "E83", lambda v=value: Raster(swizzle=v), "swizzle")
    add("bad reverse", "E84", lambda: Raster(reverse=("k",)), "reverse")
    reject(
        "slot-bound Ring",
        "E19",
        lambda: collect_pair(Ring(1), producer_ann=Producer[Instance]),
        "slot-bound",
    )
    reject(
        "missing consumer",
        "E24",
        lambda: collect_one(("ch", Producer), ch=Ring(1)),
        "no consumer role",
    )
    for mechanism in (Tma(16), Umma):
        reject(
            f"multiwarp {mechanism!r}",
            "E26",
            lambda m=mechanism: collect_pair(
                Ring(1, m), producer_warps=range(2), consumer_warps=2
            ),
            "signals once",
        )
    reject(
        "wrong producer role",
        "E21",
        lambda: collect_pair(Ring(1, producer_role="typo")),
        "producer is declared",
    )
    reject(
        "wrong consumer role",
        "E22",
        lambda: collect_pair(Ring(1, consumer_roles=("typo",))),
        "consumers are declared",
    )
    for key, value in (
        ("full_counts", [0]),
        ("full_counts", [1.5]),
        ("full_counts", [True]),
        ("full_counts", 1.5),
        ("full_counts", {1: [32]}),
        ("full_counts", {False: [1]}),
        ("empty_mask", -1),
        ("full_mask", 1.5),
        ("full_mask", True),
        ("tx", 0),
    ):
        reject(
            f"override {key}={value!r}",
            "E44-E47",
            lambda k=key, v=value: collect_pair(Ring(1, override={k: v})),
            "override",
        )
    reject(
        "pair UMMA all",
        "E37",
        lambda: collect_pair(Ring(1, Umma), cluster=CtaPair()),
        "tcgen05.commit",
    )
    payload = Ring(1, payload=(int, (1,)))
    reject(
        "pair payload",
        "E40",
        lambda: collect_pair(payload, cluster=CtaPair(), consumer_cta="leader"),
        "pair-scoped",
    )
    reject(
        "one turn seat",
        "E27",
        lambda: collect_one(("turn", Turn), turn=Turnstile()),
        "2 instances",
    )
    reject("typo annotation", "E07", lambda: collect_one(("ch", "Consumr")), "does not resolve")
    reject(
        "implementation annotation",
        "E08",
        lambda: collect_one(("work", WorkStream)),
        "layer class",
    )
    empty = inspect.Parameter.empty
    reject(
        "two contexts",
        "E12",
        lambda: collect_one(("a", empty), ("b", empty)),
        "receive the context",
    )
    reject(
        "raw producer",
        "E10",
        lambda: collect_one(("ch", Producer), raw=True, ch=Ring(1)),
        "raw role",
    )
    reject("missing Barriers", "E11", lambda: collect_one(("bars", Barriers)), "no Barriers named")

    def enter_unchecked(reason):
        with unchecked("unclosed", reason=reason):
            pass

    for value in ("", "   ", None, 1):
        add(
            f"unchecked reason={value!r}",
            "E77",
            lambda v=value: enter_unchecked(v),
            "non-empty reason",
        )

    for _ in range(24):
        depth, width = rng.randint(1, 8), rng.randint(1, 4)
        add(
            f"valid Ring({depth}) consumer width {width}",
            "R4",
            lambda d=depth, w=width: collect_pair(Ring(d), consumer_warps=range(1, 1 + w)),
            error=None,
        )
        cols, slots = rng.randint(1, 64), rng.randint(1, 4)
        offset = rng.randrange(cols)
        alias_cols = rng.randint(1, cols - offset)
        add(
            f"valid TMEM {cols}x{slots} alias {offset}+{alias_cols}",
            "R8",
            lambda c=cols, s=slots, o=offset, a=alias_cols: (
                TmemPlan().region("a", c, s).alias("b", of="a", offset=o, cols=a).describe()
            ),
            error=None,
        )
        warp, regs = rng.randrange(16), rng.choice((None, 24, 40, 64, 128))
        add(
            f"valid role {warp}/{regs}",
            "R9",
            lambda w=warp, r=regs: collect_roles({"a": declaration_role(w, r)}),
            error=None,
        )
    rng.shuffle(cases)
    return cases


def exercise_case(report: FuzzReport, case: DeclarationCase) -> None:
    try:
        case.run()
    except Exception as exc:  # noqa: BLE001 — unexpected exceptions are fuzz outcomes.
        report.record(case, exc)
    else:
        report.record(case, None)


def run_static_fuzz(seed: int, budget: int) -> FuzzReport:
    if budget < 1:
        raise ValueError("budget must be positive")
    report = FuzzReport(seed)
    cases = declaration_cases(seed)
    for i in range(budget):
        if i and i % len(cases) == 0:
            cases = declaration_cases(seed + i)
        exercise_case(report, cases[i % len(cases)])
    return report
