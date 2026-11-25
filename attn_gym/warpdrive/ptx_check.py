"""Static checks of a kernel's PTX: tcgen05 thread-sync fences, the teardown drain, bulk copies.

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
finished) are neither signals nor waits here.

``check_bulk_copies`` checks the bulk copies whose source is the CTA's shared memory (a TMA
store, a ``cp.async.bulk`` or ``cp.reduce.async.bulk`` out of it), which a role body issues by
hand, and reports:

- ``proxy``: such a copy reachable from the same thread's shared-memory store (a shared ``st``,
  ``atom`` or ``red``, ``stmatrix``) without a ``fence.proxy.async`` in between. The copy reads
  through the async proxy, which a generic store is not ordered before. The stored and the
  copied buffers are not compared, and neither a store by another thread nor one through a
  generic address (``st`` with no state space) is seen;
- ``exit``: a return reachable from such a copy that completes through a bulk group, without
  its ``cp.async.bulk.commit_group`` and a later ``cp.async.bulk.wait_group[.read] 0``: the copy
  may still read shared memory when the thread returns, and PTX states no drain there. The
  entry's return is the only boundary seen: a role inlined before it is not one.

These rules leave out the paths that a branch or guard on the thread ID rules out (see
``_threads``): a kernel dispatches its roles on the warp index, and a role tests one condition
twice (a store under ``if warp == 0``, its wait under another), so a path through every branch
crosses roles and pairs a store with a skipped wait. A condition on anything else (data,
``elect.sync``, a predicate register written twice) is taken both ways. A kernel without a
one-dimensional ``.reqntid`` is checked on every path.

``capture_ptx`` dumps the PTX of the CuTeDSL compiles inside a block.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path

RULES = ("after", "before", "drain", "proxy", "exit")

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
            "proxy": "bulk copy reads shared memory after this thread's store without "
            "fence.proxy.async",
            "exit": "return while a bulk copy may still read shared memory (no commit_group "
            "and wait_group 0)",
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
    operands: tuple[str, ...]
    guard: tuple[str, bool] | None  # the predicate and the value it runs the instruction on
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


def _kernels(ptx: str) -> Iterator[tuple[str, int, list[str], str]]:
    """(name, first line number, body lines, directives between the header and the body) of
    each ``.entry``/``.func`` with a body.

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
                body = text[opened + 1 : at].split("\n")
                yield head.group(2), _line(text, opened), body, text[head.end() : opened]
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


_GUARD = re.compile(r"@\s*(!?)\s*(\S+)\s+(.*)")
_RETURNS = ("ret", "ret.uni", "exit")
_INTEGER = re.compile(r"(-?)(0[xX][0-9a-fA-F]+|0[bB][01]+|0[0-7]*|[1-9]\d*)U?$")


def _integer(text: str) -> int | None:
    """The value of a PTX integer literal (decimal, hexadecimal, octal or binary), else None."""
    literal = _INTEGER.match(text)
    if not literal:
        return None
    digits = literal.group(2)
    prefixed = {"0x": 16, "0b": 2}.get(digits[:2].lower())
    value = int(digits[2:], prefixed) if prefixed else int(digits, 8 if digits[0] == "0" else 10)
    return -value if literal.group(1) else value


def _operands(text: str) -> tuple[str, ...]:
    """An instruction's operands, split at the commas outside vectors and addresses."""
    out, depth, current = [], 0, ""
    for ch in text.partition(" ")[2]:
        depth += (ch in "{[(") - (ch in "}])")
        if ch == "," and depth == 0:
            out.append(current.strip())
            current = ""
        else:
            current += ch
    return (*out, current.strip()) if current.strip() else tuple(out)


def _parse(body: list[str], first: int, files: dict[str, str]):
    """The instructions and labels of a kernel body. Raises for a call or an indirect branch."""
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
        guard = None
        if text.startswith("@"):
            negated, predicate, text = _GUARD.match(text + " ").groups()
            guard, text = (predicate, not negated), text.strip()
        opcode = text.split(None, 1)[0] if text else ""
        instrs.append(_Instr(number, text, opcode, _operands(text), guard, scopes, source))
    last = instrs[-1] if instrs else None
    falls = last and (last.guard or last.opcode not in (*_RETURNS, "trap", "bra", "bra.uni"))
    if falls or len(instrs) in labels.values():
        # Control that runs off the end of the body, or branches to a label there, returns.
        line, source = (last.line, last.source) if last else (first, "")
        instrs.append(_Instr(line, "ret", "ret", (), None, (0,), source))
    for i in range(len(instrs)):
        _next(instrs, labels, i, None)
    return instrs, labels


def _next(instrs, labels, i: int, runs: bool | None) -> list[int]:
    """The instructions that can follow instruction ``i``, which runs, is skipped by its guard
    (``runs=False``) or either (``None``)."""
    ins, op = instrs[i], instrs[i].opcode
    nxt = [i + 1] if i + 1 < len(instrs) else []
    if op in ("bra", "bra.uni"):
        target = ins.text.split(None, 1)[1].strip()
        index = next(
            (labels[(s, target)] for s in reversed(ins.scopes) if (s, target) in labels),
            None,
        )
        if index is None:
            raise ValueError(f"PTX line {ins.line}: branch to unknown label {target!r}")
        return nxt if runs is False else [index] if runs else [index, *nxt]
    if op.startswith("call") and _callee(ins) not in _NORETURN:
        raise ValueError(f"PTX line {ins.line}: calls are not supported (no call summaries)")
    if op.startswith("brx"):
        raise ValueError(f"PTX line {ins.line}: indirect branches are not supported")
    if op in (*_RETURNS, "trap") or op.startswith("call"):
        return [] if runs else nxt
    return nxt


def _callee(ins: _Instr) -> str:
    """The function a ``call`` names: its first operand outside parentheses ("" if none)."""
    operands = re.sub(r"\([^)]*\)", "", ins.text.partition(" ")[2])
    return next((name for name in (o.strip() for o in operands.split(",")) if name), "")


def _first_int(ins: _Instr) -> int | None:
    """An instruction's first operand when it is a literal (a barrier ID, a group count)."""
    return _integer(ins.operands[0]) if ins.operands else None


# Flags of the dataflow state, each with the index of the instruction that set it (or None). _PENDING_MMA holds
# the operations a tcgen05.commit fences (_COMMITTED), _PENDING_LDST every other one.
_SYNCED, _PENDING_LDST, _PENDING_MMA, _LOADED, _STORED = range(5)


def _transfer(ins: _Instr, index: int, state: tuple, *, teardown_barrier) -> tuple[tuple, list]:
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
    teardown = teardown_barrier is not None and _first_int(ins) == teardown_barrier
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


def _may_analysis(instrs, labels, decided: dict[int, bool], transfer, flags: int) -> dict:
    """A forward may-analysis to a fixed point: each of ``flags`` flags holds the earliest
    instruction that set it on some path reaching the instruction. ``decided``: whether each
    guarded instruction runs, where that is known. Returns the earliest origin of each
    (instruction, rule) that ``transfer`` reports broken."""
    states: list[tuple | None] = [None] * len(instrs)
    states[0] = (None,) * flags
    work, found = [0], {}
    while work:
        i = work.pop()
        runs = decided.get(i) if instrs[i].guard else True
        out, broken = (states[i], []) if runs is False else transfer(instrs[i], i, states[i])
        if runs is None:  # its predicate may be false: the state also passes unchanged
            out = _join(out, states[i])
        for rule, origin in broken:
            found[i, rule] = min(origin, found.get((i, rule), origin))
        for j in _next(instrs, labels, i, runs):
            joined = _join(states[j], out)
            if joined != states[j]:
                states[j] = joined
                work.append(j)
    return found


def _findings(name: str, instrs, found: dict) -> list[Finding]:
    findings = []
    for (i, rule), origin in sorted(found.items()):
        ins, src = instrs[i], instrs[origin]
        findings.append(
            Finding(name, rule, ins.line, ins.text, src.line, src.text, ins.source, src.source)
        )
    return findings


def _files(ptx: str) -> dict[str, str]:
    return dict(re.findall(r'^\s*\.file\s+(\d+)\s+"([^"]+)"', ptx, flags=re.MULTILINE))


def check_tcgen05_sync(ptx: str, *, teardown_barrier: int | None = None) -> list[Finding]:
    """Thread-sync and drain findings of every kernel in the PTX module ``ptx`` (see the module
    docstring). ``teardown_barrier``: the named-barrier ID of the TMEM teardown rendezvous, to
    check that every path reaching it completed its TMEM loads and stores. A module without an
    ``.entry`` body, or one whose functions do not parse, raises ``ValueError``."""
    files, findings = _files(ptx), []
    transfer = partial(_transfer, teardown_barrier=teardown_barrier)
    for name, first, body, _ in _kernels(ptx):
        instrs, labels = _parse(body, first, files)
        if instrs:
            findings += _findings(name, instrs, _may_analysis(instrs, labels, {}, transfer, 5))
    return findings


# ---- the paths of one thread -----------------------------------------------------------
#
# A register whose value follows from the thread ID alone (the warp index, a lane test, a
# role's warp range) is a constant of each thread, so a branch or guard on it is decided.

_LANES = range(32)
_BLOCK = re.compile(r"\.reqntid\s+([^\n]*)")
_TYPE = re.compile(r"\.(?:([bus])(8|16|32|64)|pred)$")
_REGISTER = re.compile(r"(?<![\w$.%])[%A-Za-z_$][\w$.]*")
_COMPARE = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lo": lambda a, b: a < b,
    "ls": lambda a, b: a <= b,
    "hi": lambda a, b: a > b,
    "hs": lambda a, b: a >= b,
}
_BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul.lo": lambda a, b: a * b,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "min": min,
    "max": max,
}


def _results(ins: _Instr) -> list[str]:
    """The registers ``ins`` may write: those of its operands before the first address, or of its
    first operand (a vector's elements, the ``d|p`` pair of a sync op). A component (``%v.x``)
    stands for its vector register. For an instruction that writes none, this only leaves a
    register it reads unknown."""
    address = next((n for n, o in enumerate(ins.operands) if o.startswith("[")), 1)
    names = (r for o in ins.operands[:address] for r in _REGISTER.findall(o))
    return [r.partition(".")[0] for r in names]


def _evaluate(ins: _Instr, values: dict, special: dict) -> tuple[int, ...] | None:
    """The value in each lane that ``ins`` writes, when it follows from the thread ID: integer
    moves, arithmetic, comparisons and selects of such values, and a shuffle of one lane's."""
    typed = _TYPE.search(ins.opcode)

    def operand(o: str) -> tuple[int, ...] | None:
        literal = _integer(o)  # a vector component (a dotted name) is never a known value
        known = None if "." in o else values.get(o)
        return special.get(o) or (known if literal is None else (literal,) * 32)

    args = [operand(o) for o in ins.operands[1:]]
    if not typed or not args or any(a is None for a in args):
        return None
    bits = int(typed.group(2) or 1)
    mask, signed = (1 << bits) - 1, typed.group(1) == "s"

    def number(v: int) -> int:
        v &= mask
        return v - (1 << bits) if signed and v >> (bits - 1) else v

    def lanes(fn) -> tuple[int, ...]:
        return tuple(int(fn(*(number(a[lane]) for a in args))) & mask for lane in _LANES)

    name = ins.opcode[: typed.start()]
    match name.split(".", 1)[0], len(args):
        case "mov", 1:
            return lanes(lambda a: a)
        case "not", 1:
            return lanes(lambda a: ~a)
        case "shl" | "shr", 2:  # the count is unsigned 32-bit whatever the type, and clamped
            counts = [min(bits, b & 0xFFFFFFFF) for b in args[1]]
            shift = int.__lshift__ if name == "shl" else int.__rshift__  # arithmetic if signed
            return tuple(shift(number(a), b) & mask for a, b in zip(args[0], counts, strict=True))
        case "setp", 2 if name[5:] in _COMPARE:
            return lanes(_COMPARE[name[5:]])
        case "selp", 3:
            return tuple((a if p else b) & mask for a, b, p in zip(*args, strict=True))
        case "shfl", 4 if name == "shfl.sync.idx":
            source, index, clamp, members = args
            if len(set(index)) == 1 and 0 <= index[0] < 32 and set(clamp) == {31}:
                if set(members) <= {-1, 0xFFFFFFFF}:  # every lane reads lane ``index``
                    return (source[index[0]],) * 32
        case _, 2 if name in _BINARY:
            return lanes(_BINARY[name])
    return None


def _lane_values(instrs, labels, warp: int) -> dict[str, tuple[int, ...] | None]:
    """The 32 lane values of each register that ``warp`` writes, None where they do not follow
    from the thread ID. A register holds one value if every instruction that the warp can run
    and that writes it gives that value, so the walk repeats until the values, and with them
    the instructions the warp can run, stop changing."""
    special = {"%tid.x": tuple(warp * 32 + lane for lane in _LANES), "%laneid": tuple(_LANES)}
    values: dict[str, tuple[int, ...] | None] = {}

    def runs(ins: _Instr) -> bool | None:
        if ins.guard is None:
            return True
        lanes = values.get(ins.guard[0])
        decisions = {bool(v) == ins.guard[1] for v in lanes} if lanes else {True, False}
        return decisions.pop() if len(decisions) == 1 else None

    while True:
        before = dict(values)
        seen, work = set(), [0]
        while work:
            i = work.pop()
            if i in seen:
                continue
            seen.add(i)
            ins, ran = instrs[i], runs(instrs[i])
            if ran is not False:
                results = _results(ins)
                # A second result (the complement in setp's p|q, shfl's in-range predicate) and
                # a vector's elements hold other values than the one modelled.
                single = results == list(ins.operands[:1])
                value = _evaluate(ins, values, special) if single else None
                for register in results:
                    same = register not in values or values[register] == value
                    values[register] = value if same else None
            work += _next(instrs, labels, i, ran)
        if values == before:
            return values


def _threads(instrs, labels, directives: str) -> Iterator[dict[int, bool]]:
    """Which guarded instructions run, for each thread of the kernel that differs in it: the
    ``decided`` of ``_may_analysis``. ``%tid.x`` gives the warp and lane only in the
    one-dimensional block of a ``.reqntid`` (``.maxntid`` bounds the thread count, not each
    dimension); any other kernel, and a last partial warp, get one thread with nothing decided."""
    block = _BLOCK.search(directives)
    dims = [_integer(d.strip()) for d in block.group(1).split(",")] if block else []
    if not dims or None in dims or any(d != 1 for d in dims[1:]):
        yield {}
        return
    warps, rest = divmod(dims[0], 32)
    if rest:
        yield {}
    seen = set()
    for warp in range(warps):
        values = _lane_values(instrs, labels, warp)
        guards = [
            (i, values[ins.guard[0]], ins.guard[1])
            for i, ins in enumerate(instrs)
            if ins.guard and values.get(ins.guard[0])
        ]
        for lane in _LANES:
            decided = tuple((i, bool(v[lane]) == runs_on) for i, v, runs_on in guards)
            if decided not in seen:
                seen.add(decided)
                yield dict(decided)


# ---- bulk copies out of shared memory --------------------------------------------------

# The CTA's shared memory is the second, source, state space of the copy.
_READS_SHARED = re.compile(
    r"cp(?:\.reduce)?\.async\.bulk\..*(?:global|shared::cluster)\.shared::cta"
)
_SHARED_STORE = re.compile(
    r"stmatrix\.|(?:st|atom|red)\.(?:(?!async\.)[\w:]+\.)*?shared(?:::\w+)?(?:\.|$)"
)
# The earliest shared store since a proxy fence, bulk read since a commit, and committed bulk
# read since a wait for every group.
_WRITTEN, _ISSUED, _GROUPED = range(3)


def _bulk_transfer(ins: _Instr, index: int, state: tuple) -> tuple[tuple, list]:
    state, broken = list(state), []
    op = ins.opcode
    if _READS_SHARED.match(op):
        if state[_WRITTEN] is not None:
            broken.append(("proxy", state[_WRITTEN]))
        if "bulk_group" in op and state[_ISSUED] is None:  # else it completes on an mbarrier
            state[_ISSUED] = index
    elif op == "cp.async.bulk.commit_group":
        reading = [i for i in (state[_ISSUED], state[_GROUPED]) if i is not None]
        state[_ISSUED], state[_GROUPED] = None, min(reading, default=None)
    elif op.startswith("cp.async.bulk.wait_group") and _first_int(ins) == 0:
        state[_GROUPED] = None
    elif op == "fence.proxy.async" or op.startswith("fence.proxy.async.shared::"):
        state[_WRITTEN] = None
    elif _SHARED_STORE.match(op):
        state[_WRITTEN] = index if state[_WRITTEN] is None else state[_WRITTEN]
    elif op in _RETURNS:
        reading = [i for i in (state[_ISSUED], state[_GROUPED]) if i is not None]
        if reading:
            broken.append(("exit", min(reading)))
    return tuple(state), broken


def check_bulk_copies(ptx: str) -> list[Finding]:
    """``proxy`` and ``exit`` findings of every kernel in the PTX module ``ptx`` (see the module
    docstring), on the paths each of its threads can take. Raises like ``check_tcgen05_sync``."""
    files, findings = _files(ptx), []
    for name, first, body, directives in _kernels(ptx):
        instrs, labels = _parse(body, first, files)
        found: dict = {}
        for decided in _threads(instrs, labels, directives) if instrs else ():
            for key, origin in _may_analysis(instrs, labels, decided, _bulk_transfer, 3).items():
                found[key] = min(origin, found.get(key, origin))
        findings += _findings(name, instrs, found)
    return findings


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
