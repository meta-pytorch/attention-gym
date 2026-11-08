"""Compile-only source mutations of locally valid channel and work-loop templates.

No compiled callable is ever invoked. Source files are kept in ``MUTATIONS_DIR`` so a finding
is independently reproducible; an isolated process should bound long campaigns.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import random
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path

import cutlass
from cutlass import cute

from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import ProtocolError
from warpdrive_fuzz.mutate_tma import TmaMutation, compile_tma_mutation
from warpdrive_fuzz.static import Finding, FuzzReport

# Generated mutation sources (CuTeDSL needs source files); override with
# WARPDRIVE_FUZZ_MUTATIONS_DIR to keep them, e.g. under agent_space/.
MUTATIONS_DIR = Path(tempfile.gettempdir()) / "warpdrive_fuzz_mutations"


@dataclass(frozen=True)
class Mutation:
    name: str
    contract: str
    body: str
    kind: str = "channel"
    error: type[Exception] | None = ProtocolError
    message: str = ""
    obligation: bool = False
    tma: TmaMutation | None = None


CHANNEL_TEMPLATE = """\
import cutlass
from cutlass import cute
from types import SimpleNamespace
from attn_gym.warpdrive import (
    WarpSpecialized, Ring, Slots, Threads, Producer, Consumer, Instance, role, unchecked,
)

class Probe(WarpSpecialized):
    def __init__(self):
        self.ch = DECLARATION

    @cute.jit
    def step(self, ch):
        h = ch.acquire(SLOT)
        h.commit()

    @role(warps=0)
    @cute.jit
    def prod(self, ch: PRODUCER_ANNOTATION, ctx):
PRODUCER_BODY
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        for i in cutlass.range(ctx.n, unroll=1):
            h = ch.wait(SLOT)
            h.release()
        return ch

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("NAME")
        self.kernel(out, n).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)
"""

WORK_TEMPLATE = """\
import cutlass
from cutlass import cute
from types import SimpleNamespace
from attn_gym.warpdrive import (
    WarpSpecialized, WorkStream, OneShot, StaticPersistent, AtomicPersistent, ClusterLaunchControl, WorkCursor, role,
)

class Probe(WarpSpecialized):
    def __init__(self):
        self.work = WorkStream(schedule=SCHEDULE)

    @role(warps=0)
    @cute.jit
    def worker(self, work: WorkCursor, ctx):
BODY
        return work

    @cute.kernel
    def kernel(self, out: cute.Tensor, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(out=out, n=bound.total))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, workspace: cute.Tensor, stream):
        bound = self.work.bind((n, 1, 1), workspace, max_ctas=cutlass.Int32(1))
        self.kernel.set_name_prefix("NAME")
        self.kernel(out, bound).launch(**self.launch_config(bound), stream=stream)
"""


def mutations() -> list[Mutation]:
    loop = "for i in cutlass.range(ctx.n, unroll=1):\n"
    valid = loop + "    h = ch.acquire(SLOT)\n    h.commit()"
    corpus = [
        Mutation("typo_annotation", "E07", valid, kind="typo", message="does not resolve"),
        Mutation("wrong_role", "E21", valid, kind="wrong_role", message="producer is declared"),
        Mutation(
            "two_contexts", "E12", valid, kind="two_contexts", message="all receive the context"
        ),
        Mutation("wrong_context_type", "E15", valid, kind="wrong_context_type", message="context"),
        Mutation("tuple_context", "E16", valid, kind="tuple_context", message="no attributes"),
        Mutation("valid", "R2", valid, error=None),
        Mutation(
            "drop_commit", "E58", loop + "    h = ch.acquire(SLOT)", message="handles never closed"
        ),
        Mutation("double_commit", "E56", valid + "\n    h.commit()", message="closed twice"),
        Mutation(
            "wrong_close", "E55", valid.replace("h.commit()", "h.release()"), message="release()"
        ),
        Mutation(
            "cross_loop",
            "E57",
            "h = ch.acquire(SLOT)\n" + loop + "    h.commit()",
            message="different dynamic region",
        ),
        Mutation(
            "cross_branch",
            "E57",
            loop + "    h = ch.acquire(SLOT)\n    if ctx.n > 0:\n        h.commit()",
            message="different dynamic region",
        ),
        Mutation(
            "with_double_close",
            "E56",
            loop + "    with ch.produce(SLOT) as h:\n        h.commit()",
            message="closed twice",
        ),
        Mutation(
            "wrong_bound_slot",
            "E62",
            valid.replace("SLOT", "1"),
            kind="bound",
            message="bound to slot 0",
        ),
        Mutation(
            "ring_slot", "E59", valid.replace("SLOT", "0"), kind="ring", message="takes no slot"
        ),
        Mutation(
            "dynamic_slot",
            "E60",
            valid.replace("SLOT", "i"),
            kind="slots",
            message="Python int slot",
        ),
        Mutation(
            "negative_slot", "E61", valid.replace("SLOT", "-1"), kind="slots", message="got -1"
        ),
        Mutation(
            "out_of_range_slot", "E61", valid.replace("SLOT", "31"), kind="slots", message="got 31"
        ),
        Mutation(
            "valid_token",
            "R2",
            loop
            + "    t = ch.try_acquire(SLOT)\n    h = ch.acquire(SLOT, token=t)\n    h.commit()",
            kind="slots",
            error=None,
        ),
        Mutation(
            "token_across_loop",
            "E127",
            "t = ch.try_acquire(SLOT)\n"
            + loop
            + "    h = ch.acquire(SLOT, token=t)\n    h.commit()",
            kind="slots",
            message="different dynamic region",
        ),
        Mutation(
            "token_reused",
            "E127",
            loop
            + "    t = ch.try_acquire(SLOT)\n"
            + "    h = ch.acquire(SLOT, token=t)\n    h.commit()\n" * 2,
            kind="slots",
            message="stale token",
        ),
        Mutation(
            "valid_peek",
            "R2",
            loop + "    ch.peek(SLOT)\n    h = ch.wait(SLOT)\n    h.release()",
            kind="consumer",
            error=None,
        ),
        Mutation(
            "preview_write",
            "E128",
            loop + "    ch.peek(SLOT).view(write=True)\n    h = ch.wait(SLOT)\n    h.release()",
            kind="consumer",
            message="read-only borrow",
        ),
        Mutation(
            "preview_unconsumed",
            "E128",
            loop + "    h = ch.wait(SLOT)\n    h.release()\n    ch.peek(SLOT)",
            kind="consumer",
            message="previews no later wait consumes",
        ),
        Mutation(
            "skip_join",
            "E131",
            loop + "    ch.skip(1)\n    h = ch.wait(SLOT)\n    h.release()",
            kind="consumer",
            message="only consumers of an exclusive ring",
        ),
        Mutation(
            "helper_discards_end", "U10", loop + "    self.step(ch)", error=None, obligation=True
        ),
        Mutation(
            "unbalanced_epochs",
            "U1",
            valid.replace("ctx.n,", "ctx.n + 1,"),
            error=None,
            obligation=True,
        ),
        Mutation(
            "lane_predicated_pair",
            "U1",
            "if cute.arch.lane_idx() == 0:\n" + textwrap.indent(valid, "    "),
            error=None,
            obligation=True,
        ),
        Mutation(
            "unchecked_no_reason",
            "E77",
            'with unchecked("unclosed", reason=""):\n    ch.acquire(SLOT)',
            error=ValueError,
            message="non-empty reason",
        ),
        Mutation(
            "missing_advance",
            "E71",
            "m, n, l = work.item",
            kind="work",
            message="never calls advance",
        ),
        Mutation(
            "double_advance",
            "E72",
            "while work.valid:\n    work.advance()\n    work.advance()",
            kind="work",
            message="traced 2 times",
        ),
        Mutation(
            "conditional_advance",
            "E73",
            "while work.valid:\n    if ctx.n > 0:\n        work.advance()",
            kind="work",
            message="directly in that loop",
        ),
        Mutation(
            "nested_loop_advance",
            "E73",
            "while work.valid:\n    for i in cutlass.range(ctx.n, unroll=1):\n        work.advance()",
            kind="work",
            message="directly in that loop",
        ),
        Mutation(
            "compound_condition",
            "G1",
            "count = cutlass.Int32(0)\nwhile work.valid & (count < ctx.n):\n    count += 1\n    work.advance()",
            kind="work",
            message="condition",
        ),
        Mutation(
            "advance_before_work",
            "G1",
            "while work.valid:\n    work.advance()\n    m, n, l = work.item\n    ctx.out[0] = m",
            kind="work",
            message="after advance",
        ),
        Mutation(
            "valid_work",
            "R12",
            "while work.valid:\n    m, n, l = work.item\n    ctx.out[0] = m\n    work.advance()",
            kind="work",
            error=None,
        ),
    ]
    for mutation in corpus[6:12]:
        if mutation.error is not None:
            consumer_body = (
                mutation.body.replace("acquire", "wait")
                .replace("commit", "release")
                .replace("produce", "consume")
            )
            if mutation.name == "wrong_close":
                consumer_body = consumer_body.replace("h.release()", "h.commit()")
            corpus.append(
                Mutation(
                    "consumer_" + mutation.name,
                    mutation.contract,
                    consumer_body,
                    kind="consumer",
                    message=mutation.message.replace("release()", "commit()"),
                )
            )
    for pair, leader, group in ((False, False, "ONE"), (True, True, "TWO"), (True, False, "ONE")):
        for copies in (1, 2):
            for scale in (0.5, 1, 2):
                config = TmaMutation(pair, leader, group, int(8192 * copies * scale), copies)
                corpus.append(
                    Mutation(
                        f"tma_{pair}_{leader}_{copies}_{scale}",
                        "E81/R11",
                        "",
                        kind="tma",
                        error=None if scale == 1 else ProtocolError,
                        message="bytes",
                        tma=config,
                    )
                )
        corpus.append(
            Mutation(
                f"tma_opaque_{pair}_{leader}",
                "U6",
                "",
                kind="tma",
                error=None,
                obligation=True,
                tma=TmaMutation(pair, leader, group, opaque=True),
            )
        )
    for leader, group in ((True, "ONE"), (False, "TWO")):
        corpus.append(
            Mutation(
                f"tma_bad_group_{leader}",
                "E80",
                "",
                kind="tma",
                message="CtaGroup",
                tma=TmaMutation(True, leader, group),
            )
        )
    return corpus


def source_for(mutation: Mutation, seed: int, depth: int = 2) -> str:
    name = f"warpdrive_mutation_{seed}_{mutation.name}"
    if mutation.kind == "work":
        schedule = (
            "OneShot()",
            "StaticPersistent()",
            "AtomicPersistent(stages=1)",
            "ClusterLaunchControl(stages=2)",
        )[seed % 4]
        return (
            WORK_TEMPLATE.replace("BODY", textwrap.indent(mutation.body, "        "))
            .replace("NAME", name)
            .replace("SCHEDULE", schedule)
        )
    fixed = mutation.kind in ("slots", "bound")
    if mutation.kind == "bound":
        depth = 2
    declaration = (
        f"Slots({depth}, Threads, Threads)" if fixed else f"Ring({depth}, Threads, Threads)"
    )
    body = mutation.body.replace("SLOT", "0" if fixed else "")
    source = CHANNEL_TEMPLATE
    if mutation.kind == "consumer":
        old = "        for i in cutlass.range(ctx.n, unroll=1):\n            h = ch.wait(SLOT)\n            h.release()"
        assert source.count(old) == 1
        source = source.replace(old, textwrap.indent(body, "        "))
        body = "for i in cutlass.range(ctx.n, unroll=1):\n    h = ch.acquire()\n    h.commit()"
    source = source.replace("PRODUCER_BODY", textwrap.indent(body, "        "))
    if mutation.kind == "bound":
        source = source.replace("@role(warps=0)", "@role(warps=[range(1), range(1, 2)])")
        source = source.replace("@role(warps=1)", "@role(warps=2)")
    if mutation.kind == "typo":
        source = source.replace("PRODUCER_ANNOTATION", '"Consumr"')
    elif mutation.kind == "wrong_role":
        declaration = declaration[:-1] + ', producer_role="other")'
    elif mutation.kind == "two_contexts":
        source = source.replace("PRODUCER_ANNOTATION, ctx):", "PRODUCER_ANNOTATION, ctx, other):")
    elif mutation.kind == "wrong_context_type":
        source = source.replace(
            "class Probe",
            "from dataclasses import dataclass\n@dataclass(frozen=True)\nclass ExpectedCtx:\n    n: cutlass.Int32\n\nclass Probe",
        )
        source = source.replace(
            "PRODUCER_ANNOTATION, ctx):", "PRODUCER_ANNOTATION, ctx: ExpectedCtx):"
        )
    elif mutation.kind == "tuple_context":
        source = source.replace("SimpleNamespace(out=out, n=n)", "(out, n)")
    return (
        source.replace("DECLARATION", declaration)
        .replace(
            "PRODUCER_ANNOTATION", "Producer[Instance]" if mutation.kind == "bound" else "Producer"
        )
        .replace("SLOT", "0" if fixed else "")
        .replace("NAME", name)
    )


def compile_mutation(mutation: Mutation, seed: int, depth: int = 2):
    """Compile fresh source without a runtime tensor, device query, or launch invocation."""
    if mutation.tma is not None:
        return compile_tma_mutation(mutation.tma)
    source = source_for(mutation, seed, depth)
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    root = Path(os.environ.get("WARPDRIVE_FUZZ_MUTATIONS_DIR", MUTATIONS_DIR))
    root.mkdir(parents=True, exist_ok=True)
    source_file = root / f"mutation_{digest}.py"
    if not source_file.exists():  # atomic: concurrent workers may write the same digest
        tmp = root / f".mutation_{digest}.{os.getpid()}.tmp"
        tmp.write_text(source)
        os.replace(tmp, source_file)
    module_name = f"warpdrive_mutation_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
        args = [out, cutlass.Int32(1)]
        if mutation.kind == "work":
            args.append(make_fake_compact_tensor(cutlass.Int32, (2,), assumed_align=4))
        return compile_tvm_ffi(module.Probe(), *args, name=module_name)
    finally:
        del sys.modules[module_name]


def is_expected_error(exc: Exception, error: type[Exception], message: str) -> bool:
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


def run_mutation_fuzz(seed: int, budget: int) -> FuzzReport:
    if budget < 1:
        raise ValueError("budget must be positive")
    rng = random.Random(seed)
    corpus = mutations()
    rng.shuffle(corpus)
    report = FuzzReport(seed)
    for i in range(budget):
        mutation = corpus[i % len(corpus)]
        depth = rng.choice((2, 3, 4))
        report.cases += 1
        try:
            result = compile_mutation(mutation, seed + i, depth)
            if mutation.tma is not None and mutation.error is None:
                wanted = "unverified" if mutation.tma.opaque else "verified CtaGroup"
                if wanted not in result.describe():
                    raise AssertionError(f"TMA status must report {wanted}")
        except Exception as exc:  # noqa: BLE001 — compiler diagnostics are fuzz outcomes.
            if mutation.error is not None and is_expected_error(
                exc, mutation.error, mutation.message
            ):
                report.rejected += 1
            else:
                report.findings.append(
                    Finding(
                        seed,
                        mutation.name,
                        mutation.contract,
                        "unexpected exception",
                        f"{type(exc).__name__}: {exc}",
                    )
                )
        else:
            if mutation.error is not None:
                report.findings.append(
                    Finding(
                        seed,
                        mutation.name,
                        mutation.contract,
                        "missing rejection",
                        f"expected {mutation.error.__name__}: {mutation.message}",
                    )
                )
            elif mutation.obligation:
                report.obligations += 1
            else:
                report.accepted += 1
    return report
