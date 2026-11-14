"""Static check of the tcgen05 thread-sync fences and teardown drain in a kernel's PTX.

Besides the standard memory-consistency mechanisms, PTX orders an asynchronous tcgen05
operation (``mma``, ``cp``, ``shift``, ``ld``, ``st``) of one thread against another thread's
through a thread synchronization with ``tcgen05.fence::before_thread_sync`` between the operation
and the signal, and ``tcgen05.fence::after_thread_sync`` between the wait and the operation (PTX
ISA 9.7.18.6.3 and 9.7.18.6.4); warpdrive uses this route. ``check_tcgen05_sync`` follows every
intraprocedural control-flow path of each kernel in a PTX module and reports:

- ``after``: a tcgen05 operation (or ``tcgen05.dealloc``) reachable from a synchronization (an
  mbarrier wait or probe, a barrier sync) without an ``after_thread_sync`` in between;
- ``before``: a signal (an mbarrier or barrier arrive, a barrier sync, a release store) reachable
  from a tcgen05 operation without a ``before_thread_sync`` in between. A ``tcgen05.commit``
  performs the fence for the MMAs, copies and shifts it tracks, not for loads and stores;
- ``drain`` (with ``teardown_barrier``): the TMEM teardown rendezvous reachable from a
  ``tcgen05.ld``/``st`` without its ``tcgen05.wait::ld``/``wait::st``.

The check is path-insensitive: it does not evaluate branch conditions or guard predicates, so a
path through code that never runs on one thread (for example from one role's body into
another's) counts too, and a guarded instruction may or may not run: a guarded fence, commit or
wait completes nothing. A ``tcgen05`` instruction other than the fences, ``commit``, the waits
and allocation counts as an asynchronous operation, one a commit fences only if it is an
``mma``, ``cp`` or ``shift``. Calls (other than to ``__assertfail``, which
does not return) and indirect branches raise ``ValueError``: the check has no call summaries.
So does a module it cannot split into kernels: an unparsed function header, unbalanced braces,
or no ``.entry`` with a body. Kernels are found independent of line layout.
Acquire loads (flag spins) are not waits here, and only the teardown's completion of TMEM loads
and stores is checked, not an in-loop handoff's (the alias and fence diagnostics own those).
``bar.warp.sync`` (a warp's lanes converge) and ``griddepcontrol.wait`` (an earlier grid
finished) are neither signals nor waits here. ``capture_ptx`` dumps the PTX of the CuTeDSL
compiles inside a block.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

RULES = ("after", "before", "drain")

_SYNCHRONOUS = tuple(
    f"tcgen05.{name}"
    for name in (
        "fence::before_thread_sync",
        "fence::after_thread_sync",
        "commit",
        "wait::ld",
        "wait::st",
        "alloc",
        "dealloc",
        "relinquish_alloc_permit",
    )
)
_WAITS = (
    "mbarrier.try_wait",
    "mbarrier.test_wait",
    "barrier.cluster.wait",
)
_SIGNALS = (
    "mbarrier.arrive",
    "cp.async.mbarrier.arrive",
    "barrier.cluster.arrive",
)
_BARRIER_ARRIVE = re.compile(r"^(?:bar|barrier)(?:\.cta)?\.arrive\b")
_BARRIER_SYNC = re.compile(r"^(?:bar|barrier)(?:\.cta)?\.(?:sync|red)\b")
_RELEASE = re.compile(r"^(?:st|red|atom)\..*\.release\b|^(?:st|red|atom)\.release\b")
_COMMITTED = ("tcgen05.mma", "tcgen05.cp", "tcgen05.shift")  # what a tcgen05.commit tracks
_NORETURN = ("__assertfail",)  # a failed device assert traps
_LABEL = re.compile(r"^([A-Za-z_$][\w$]*):(?!:)\s*(.*)$")
_LINE_END = re.compile(r"^(?:[A-Za-z_$][\w$]*:(?!:)\s*$|\.(?:loc|file)\b)")


@dataclass(frozen=True)
class Finding:
    """``rule`` (one of ``RULES``) broken at PTX line ``line`` (``text``), reached from the
    operation at ``origin`` (``origin_text``). ``source``/``origin_source``: the ``file:line``
    of each from the PTX ``.loc`` directives (compile with ``CUTE_DSL_LINEINFO=1``), else ""."""

    kernel: str
    rule: str
    line: int
    text: str
    origin: int
    origin_text: str
    source: str = ""
    origin_source: str = ""

    def describe(self) -> str:
        what = {
            "after": "tcgen05 operation after a synchronization without "
            "tcgen05.fence::after_thread_sync",
            "before": "signal after a tcgen05 operation without tcgen05.fence::before_thread_sync",
            "drain": "TMEM teardown rendezvous with an uncompleted tcgen05.ld/st",
        }[self.rule]
        here = f" ({self.source})" if self.source else ""
        there = f" ({self.origin_source})" if self.origin_source else ""
        return (
            f"{self.kernel}: {what}: line {self.line}{here} `{self.text[:80]}`, reached from "
            f"line {self.origin}{there} `{self.origin_text[:80]}`"
        )


@dataclass
class _Instr:
    line: int
    text: str  # the instruction without its guard predicate
    opcode: str
    guarded: bool
    scopes: tuple[int, ...]  # enclosing brace scopes, innermost last
    source: str = ""  # file:line of the latest .loc directive


def _statements(lines: list[str], first: int):
    """(line number, kind, text, scope chain) of each label and instruction. A brace opens a
    scope at the start of a statement; inside one (``{%r1, %r2}``) it is an operand vector."""
    scopes, counter = [0], 0
    text, line, depth = "", first, 0  # the statement so far, its first line, vector depth

    def flush():
        stmt = text.strip()
        label = _LABEL.match(stmt)
        if label:
            yield line, "label", label.group(1), tuple(scopes)
            stmt = label.group(2).strip()
        if stmt:
            yield line, "instr", " ".join(stmt.split()), tuple(scopes)

    for number, raw in enumerate(lines, first):
        for ch in raw.split("//", 1)[0] + "\n":
            if not text.strip():
                text, line = text if ch.isspace() else "", number
                if ch == "{":
                    counter += 1
                    scopes.append(counter)
                    continue
                if ch == "}":
                    scopes.pop()
                    continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if ch == ";" and depth == 0:
                yield from flush()
                text = ""
            elif ch == "\n" and depth == 0 and _LINE_END.match(text.strip() or "-"):
                yield from flush()  # a label or a ;-less directive (.loc, .file) ends its line
                text = ""
            else:
                text += ch


_FUNCTION = re.compile(r"(?<![\w$.])\.(?:entry|func)\b")
_HEADER = re.compile(r"\.(entry|func)\s*(?:\([^()]*\)\s*)?([A-Za-z_$%][\w$]*)\s*\(")
_COMMENTS_AND_STRINGS = re.compile(r'(")(?:[^"\\\n]|\\.)*"|//[^\n]*')
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_BRACES = re.compile(r"[{};]")


def _kernels(ptx: str) -> Iterator[tuple[str, int, list[str]]]:
    """(name, first line number, body lines) of each ``.entry``/``.func`` with a body.

    Module-level statements end at ``;`` or a body's closing brace, wherever lines break: a
    header may span lines and a body may share its braces' lines. A function header that does
    not parse, unbalanced braces, or a module without an ``.entry`` body raise."""
    # Drop comments and string contents (.file paths, pragmas), keeping every line break.
    text = _COMMENTS_AND_STRINGS.sub(r"\1\1", ptx)
    if "/*" in text:
        text = _BLOCK_COMMENT.sub(lambda m: "\n" * m.group().count("\n"), text)
    start, opened, depth, head, has_entry = 0, 0, 0, None, False
    for token in _BRACES.finditer(text):
        at, char = token.start(), token.group()
        if depth:
            depth += {"{": 1, "}": -1}.get(char, 0)
            if depth == 0 and head:  # a body closes; after an initializer the statement goes on
                has_entry = has_entry or head.group(1) == "entry"
                yield head.group(2), _line(text, opened), text[opened + 1 : at].split("\n")
                start = at + 1
        elif char == ";":
            start = at + 1
        elif char == "{":
            depth, opened, head = 1, at, _header(text, start, at)
        else:
            raise ValueError(f"PTX line {_line(text, at)}: unbalanced closing brace")
    if depth:
        raise ValueError(f"PTX line {_line(text, opened)}: unclosed brace")
    if not has_entry:
        raise ValueError("the PTX module has no .entry with a body")


def _header(text: str, start: int, end: int) -> re.Match | None:
    """The ``.entry``/``.func`` header in ``text[start:end]`` (None if it names neither)."""
    keyword = _FUNCTION.search(text, start, end)
    if not keyword:
        return None
    header = _HEADER.match(text, keyword.start(), end)
    if not header:
        raise ValueError(f"PTX line {_line(text, keyword.start())}: unparsed function header")
    return header


def _line(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _parse(body: list[str], first: int, files: dict[str, str]):
    instrs, labels, source = [], {}, ""
    for number, kind, text, scopes in _statements(body, first):
        if kind == "label":
            labels[(scopes[-1], text)] = len(instrs)
            continue
        loc = re.match(r"\.loc\s+(\d+)\s+(\d+)", text)
        if loc:
            path = files.get(loc.group(1), loc.group(1))
            source = f"{path.rsplit('/', 1)[-1]}:{loc.group(2)}"
            continue
        if text.startswith("."):  # a declaration (.reg, .shared, .pragma, ...)
            continue
        guarded = text.startswith("@")
        if guarded:
            text = text.split(None, 1)[1] if " " in text or "\t" in text else ""
        opcode = text.split(None, 1)[0] if text else ""
        instrs.append(_Instr(number, text, opcode, guarded, scopes, source))
    return instrs, labels


def _successors(instrs, labels) -> list[list[int]]:
    succ = []
    for i, ins in enumerate(instrs):
        nxt = [i + 1] if i + 1 < len(instrs) else []
        op = ins.opcode
        if op in ("bra", "bra.uni"):
            target = ins.text.split(None, 1)[1].strip()
            index = next(
                (labels[(s, target)] for s in reversed(ins.scopes) if (s, target) in labels),
                None,
            )
            if index is None:
                raise ValueError(f"PTX line {ins.line}: branch to unknown label {target!r}")
            succ.append([index, *nxt] if ins.guarded else [index])
        elif op.startswith("call") and _callee(ins) not in _NORETURN:
            raise ValueError(f"PTX line {ins.line}: calls are not supported (no call summaries)")
        elif op in ("ret", "exit", "trap") or op.startswith("call"):
            succ.append(nxt if ins.guarded else [])
        elif op.startswith("brx"):
            raise ValueError(f"PTX line {ins.line}: indirect branches are not supported")
        else:
            succ.append(nxt)
    return succ


def _callee(ins: _Instr) -> str:
    """The function a ``call`` names: its first operand outside parentheses ("" if none)."""
    operands = re.sub(r"\([^)]*\)", "", ins.text.partition(" ")[2])
    return next((name for name in (o.strip() for o in operands.split(",")) if name), "")


def _barrier_id(text: str) -> int | None:
    operands = text.split(None, 1)[1] if " " in text or "\t" in text else ""
    first = operands.split(",", 1)[0].strip()
    return int(first) if first.isdigit() else None


# Flags of the dataflow state, each with the index of the instruction that set it (or None). _PENDING_MMA holds
# the operations a tcgen05.commit fences (_COMMITTED), _PENDING_LDST every other one.
_SYNCED, _PENDING_LDST, _PENDING_MMA, _LOADED, _STORED = range(5)


def _transfer(ins: _Instr, index: int, state: tuple, teardown_barrier) -> tuple[tuple, list]:
    """The state after ``ins`` (instruction ``index``), and the (rule, origin index) of each rule
    ``ins`` breaks."""
    state, broken = list(state), []
    op = ins.opcode
    # A tcgen05 instruction outside the synchronous ones counts as an asynchronous operation.
    is_async = op.startswith("tcgen05.") and not op.startswith(_SYNCHRONOUS)
    is_sync = _BARRIER_SYNC.match(op) is not None
    signals = (
        is_sync
        or op.startswith(_SIGNALS)
        or _BARRIER_ARRIVE.match(op) is not None
        or _RELEASE.match(op) is not None
    )
    if (is_async or op.startswith("tcgen05.dealloc")) and state[_SYNCED] is not None:
        broken.append(("after", state[_SYNCED]))
    pending = [i for i in (state[_PENDING_LDST], state[_PENDING_MMA]) if i is not None]
    if signals and pending:
        broken.append(("before", min(pending)))
    outstanding = [i for i in (state[_LOADED], state[_STORED]) if i is not None]
    teardown = teardown_barrier is not None and _barrier_id(ins.text) == teardown_barrier
    if is_sync and teardown and outstanding:
        broken.append(("drain", min(outstanding)))
    if op == "tcgen05.fence::after_thread_sync":
        state[_SYNCED] = None
    elif op == "tcgen05.fence::before_thread_sync":
        state[_PENDING_LDST] = state[_PENDING_MMA] = None
    elif op.startswith("tcgen05.commit"):
        state[_PENDING_MMA] = None
    elif op.startswith("tcgen05.wait::ld"):
        state[_LOADED] = None
    elif op.startswith("tcgen05.wait::st"):
        state[_STORED] = None
    if is_async:
        pending = _PENDING_MMA if op.startswith(_COMMITTED) else _PENDING_LDST
        state[pending] = index if state[pending] is None else state[pending]
        if op.startswith(("tcgen05.ld", "tcgen05.st")):
            flag = _LOADED if op.startswith("tcgen05.ld") else _STORED
            state[flag] = index if state[flag] is None else state[flag]
    if is_sync or op.startswith(_WAITS):
        state[_SYNCED] = index if state[_SYNCED] is None else state[_SYNCED]
    return tuple(state), broken


def _join(a: tuple | None, b: tuple) -> tuple:
    if a is None:
        return b
    return tuple(
        x if y is None else y if x is None else min(x, y) for x, y in zip(a, b, strict=True)
    )


def _check_kernel(
    name: str, body: list[str], first: int, teardown_barrier, files: dict[str, str]
) -> list[Finding]:
    """A forward may-analysis to a fixed point: each flag holds the earliest instruction that
    set it on some path reaching the instruction."""
    instrs, labels = _parse(body, first, files)
    if not instrs:
        return []
    succ = _successors(instrs, labels)
    states: list[tuple | None] = [None] * len(instrs)
    states[0] = (None,) * 5
    work, found = [0], {}
    while work:
        i = work.pop()
        out, broken = _transfer(instrs[i], i, states[i], teardown_barrier)
        if instrs[i].guarded:  # its predicate may be false: the state also passes unchanged
            out = _join(out, states[i])
        for rule, origin in broken:  # keep the earliest origin of each (instruction, rule)
            found[i, rule] = min(origin, found.get((i, rule), origin))
        for j in succ[i]:
            if j == len(instrs):  # a label after the last instruction
                continue
            joined = _join(states[j], out)
            if joined != states[j]:
                states[j] = joined
                work.append(j)
    findings = []
    for (i, rule), origin in sorted(found.items()):
        ins, src = instrs[i], instrs[origin]
        findings.append(
            Finding(name, rule, ins.line, ins.text, src.line, src.text, ins.source, src.source)
        )
    return findings


def check_tcgen05_sync(ptx: str, *, teardown_barrier: int | None = None) -> list[Finding]:
    """Findings of every kernel in the PTX module ``ptx`` (see the module docstring).
    ``teardown_barrier``: the named-barrier ID of the TMEM teardown rendezvous, to check that
    every path reaching it completed its TMEM loads and stores. A module without an ``.entry``
    body, or one whose functions do not parse, raises ``ValueError``."""
    files = dict(re.findall(r'^\s*\.file\s+(\d+)\s+"([^"]+)"', ptx, flags=re.MULTILINE))
    return [
        finding
        for name, first, body in _kernels(ptx)
        for finding in _check_kernel(name, body, first, teardown_barrier, files)
    ]


@contextmanager
def capture_ptx(directory: Path | str) -> Iterator[Path]:
    """Dump the PTX of every CuTeDSL compile inside the block into ``directory`` (created),
    as ``<n>.sm_*.ptx``. Compile through ``compile_fn.__wrapped__(...)``: a cached kernel
    produces no PTX."""
    from cutlass.base_dsl.compiler import Compiler

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    original = Compiler.compile
    count = [0]

    def compile_with_ptx(self, module, pipeline, *args, **kwargs):
        if "cute-to-nvvm{" not in pipeline:
            raise RuntimeError("PTX capture requires the CuTeDSL cute-to-nvvm pipeline")
        option = f"dump-ptx-path='{out.resolve() / str(count[0])}' "
        count[0] += 1
        pipeline = re.sub(r"dump-ptx-path='[^']*'", "", pipeline)
        pipeline = pipeline.replace("cute-to-nvvm{", "cute-to-nvvm{" + option, 1)
        return original(self, module, pipeline, *args, **kwargs)

    Compiler.compile = compile_with_ptx
    try:
        yield out
    finally:
        Compiler.compile = original


def read_ptx(path: Path | str) -> str:
    """A dumped PTX file's text (CuTeDSL pads it with NUL bytes)."""
    return Path(path).read_bytes().rstrip(b"\x00").decode()
