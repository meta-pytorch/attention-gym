"""Liveness checks on the event graph: deadlock and epoch balance.

Roles run fixed op orders and never choose between channels, so one execution order is as good
as any: a dependency cycle in the event graph of one input deadlocks every run with that input,
and an acyclic graph whose waits all have their signals completes. For each assignment of the
values a kernel's loops and branches depend on (``events.py``), the check unrolls and links the
roles and looks for

* imbalance: a channel whose fills and takes disagree (a joining consumer taking fewer epochs
  than were filled, round-robin or exclusive owners not covering each epoch once). An
  assignment that does not balance is taken as one the kernel never sees. If none balances,
  the kernel is unbalanced when its counts are static (no runtime value, one work item), and
  unverified otherwise: a caller may pass only counts the samples miss (``n == 4``);
* deadlock: a cycle, or a wait whose signals never run.

A kernel without leaves (static trip counts, one work item per CTA) is verified outright.
Otherwise the verdict names the counts checked. Anything the trace cannot see (raw roles, raw
state, data-dependent loops, ops inside ``unchecked("liveness")``) makes it ``unverified``;
that is never an error, and neither is a failure of the analysis itself.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from functools import reduce

from attn_gym.warpdrive.events import (
    DEBUG,
    MAX_NODES,
    TAKES,
    ChannelView,
    LaneDivergent,
    Unit,
    Unknown,
    dependencies,
    find_cycle,
    leaves,
    link,
    program,
    tid_dependent,
    unroll,
)
from attn_gym.warpdrive.protocol import ProtocolError

MODES = ("strict", "warn", "off")
MAX_ASSIGNMENTS = 27
OVERRIDE = (
    'Override with unchecked("liveness", reason=...) around the ops, or liveness_check="warn" '
    'or "off" on the kernel.'
)


@dataclass
class Liveness:
    """``verdict``: "verified", "no deadlock found" (``detail``: the counts it assumes) or
    "unverified" (``detail``: why); ``findings``: definite errors."""

    verdict: str
    detail: str = ""
    findings: list = field(default_factory=list)

    def line(self) -> str:
        if self.findings:
            return "liveness: error (see the error message)"
        if self.verdict == "verified":
            return "liveness: verified"
        word = "assumes " if self.verdict == "no deadlock found" else ""
        return f"liveness: {self.verdict} ({word}{self.detail})"


def check(kernel, roles: list) -> Liveness:
    try:
        return _check(kernel, roles)
    except Exception as e:  # a model gap must never fail a compile
        if DEBUG:
            raise
        return Liveness("unverified", f"analysis failed: {type(e).__name__}: {e}"[:300])


def report(result: Liveness | None, mode: str, kernel_name: str) -> None:
    if result is None or not result.findings:
        return
    message = f"{kernel_name}: {result.findings[0]} {OVERRIDE}"
    if mode == "strict":
        raise ProtocolError(message)
    warnings.warn(f"warpdrive: {message}", stacklevel=3)


def _check(kernel, roles: list) -> Liveness:
    raw = next((r.name for r in roles if r.raw), None)
    if raw is not None:
        return Liveness("unverified", f"raw role {raw}")
    external = {n for n, ch in kernel._ws_chans.items() if ch.external is not None}
    for name, ch in kernel._ws_chans.items():
        for key in ch.override or {}:
            if key != "tx" and getattr(ch.scope, key) != getattr(ch.derived_scope, key):
                return Liveness("unverified", f"{name} overrides its {key}")
    for r in roles:
        if r.unread:
            return Liveness("unverified", r.unread)
        for ev in r.events:
            if ev.opted_out and ev.kind != "access":
                return Liveness("unverified", f'unchecked("liveness") at {ev.site}')
            if ev.kind == "op" and ev.data["chan"] in external:
                return Liveness("unverified", f"raw code drives a side of {ev.data['chan']}")
    chans = {n: ChannelView.of(ch) for n, ch in kernel._ws_chans.items() if n not in external}
    named = {
        nb.name
        for ch in kernel._ws_chans.values()
        for nb in (*ch.full_barriers, *ch.empty_barriers)
    }
    bars = {name: nb.threads for name, nb in kernel._ws_named_bars.items()}
    turnstiles = {name: [r.name for r in ts.parties] for name, ts in kernel._ws_turns.items()}
    units = [
        Unit(r, rank, warps)
        for r in roles
        for rank in r.ctas
        for warps in ([(w,) for w in r.warps] if tid_dependent(r) else [r.warps])
    ]
    found: set = set()
    for r in roles:
        for region in r.regions.values():
            for e in (*(region.bounds or ()), region.cond):
                leaves(e, found)
    names = sorted(found, key=repr)
    # Leaves that share a value (n // 2 and n % 2) cannot take any values independently: an
    # assignment may be one no input gives, so its findings are only "unverified".
    related = _related(names)
    deepest = max([ch.n for ch in chans.values()] + [1])
    values = sorted({1, 2, deepest + 1})
    persistent = any(st.schedule.persistent for st in kernel._ws_streams.values())
    items = values if persistent else [1]
    if len(items) * len(values) ** len(names) <= MAX_ASSIGNMENTS:
        combos = itertools.product(items, *([values] * len(names)))
    else:  # every leaf equal, at each value and item count
        combos = ((m, *[v] * len(names)) for m in items for v in values)
    balanced, unbalanced, late = False, None, False
    for m, *vals in combos:
        env = dict(zip(names, vals))
        try:
            ops = {u: unroll(u, env, m, chans, named) for u in units}
            if sum(map(len, ops.values())) > MAX_NODES:
                raise Unknown(f"the roles run over {MAX_NODES} ops")
        except LaneDivergent as e:
            return Liveness("unverified", findings=[str(e)])
        except Unknown as e:
            return Liveness("unverified", str(e))
        problem = _balance(ops, chans)
        if problem:
            unbalanced = unbalanced or (problem, _counts(m, vals))
            continue
        balanced = True
        try:
            graph = link(ops, chans, bars, turnstiles, early=False)
            if find_cycle(graph) is None and not graph.starved:
                finding = _generations(graph, chans, _counts(m, vals))
                if finding:
                    return _found(finding, related)
                continue
            early = link(ops, chans, bars, turnstiles, early=True)
        except Unknown as e:
            return Liveness("unverified", str(e))
        cycle = find_cycle(early)
        where = _counts(m, vals)
        if cycle is not None:
            return _found(_cycle(early, cycle, where), related)
        if early.starved:
            i, why = early.starved[0]
            n = early.nodes[i]
            return _found(f"deadlock: {_node(n)} waits forever: {why}{where}.", related)
        late = True
    if not balanced and unbalanced:
        problem, where = unbalanced
        if names or persistent:  # the kernel may only run counts the samples miss
            finding = _imbalance(roles, chans)
            if finding:
                return Liveness("unverified", findings=[finding])
            return Liveness("unverified", f"no checked value balances: {problem}{where}")
        return _found(f"unbalanced epochs: {problem}{where}.", related)
    if late:
        return Liveness(
            "unverified", "a cycle closes if TMA fills or raw signals wait for their close"
        )
    if not names and not persistent:
        return Liveness("verified")
    assumed = []
    if persistent:
        assumed.append(f"{', '.join(map(str, items))} work items per CTA")
    if names:
        assumed.append(f"loop bounds and conditions of {len(names)} runtime values in {values}")
    if unbalanced:
        assumed.append("values that unbalance a channel never occur")
    return Liveness("no deadlock found", "; ".join(assumed))


class _Opaque(Exception):
    """A count that is not affine in the leaves."""


def _times(a: dict, b: dict) -> dict:
    if set(a) - {1} and set(b) - {1}:
        raise _Opaque  # a product of two runtime values
    c, x = (a.get(1, 0), b) if not set(a) - {1} else (b.get(1, 0), a)
    return {k: c * v for k, v in x.items() if c * v}


def _trips(lower, upper) -> dict:
    """A counted loop's trip count: exact for constant bounds, else one symbol per pair of
    bounds, so only loops with the same bounds cancel. A bound must be a constant or a kernel
    argument, which every role reads alike: arithmetic may wrap at a width the expression does
    not record, and special registers, loads and loop indices differ between roles."""
    if lower[0] == upper[0] == "c":
        return {1: max(0, upper[1] - lower[1])}
    if not {lower[0], upper[0]} <= {"c", "arg"}:
        raise _Opaque
    return {("trips", lower, upper): 1}


def _plus(a: dict, b: dict, sign: int = 1) -> dict:
    out = {k: a.get(k, 0) + sign * b.get(k, 0) for k in {*a, *b}}
    return {k: v for k, v in out.items() if v}


def _imbalance(roles: list, chans: dict) -> str | None:
    """A channel whose fills and takes differ by the same nonzero count for every input: both
    counts are sums of trip counts of the same counted loops (step 1) and differ only in the
    constant. Anything else is left to sampling."""
    for name, chan in chans.items():
        remote = any(tuple(ts) != (r,) for r, ts in (*chan.full.items(), *chan.empty.items()))
        if chan.fixed or remote or len({r.ctas for r in roles}) > 1:
            continue
        totals: dict = {}  # (producer side?, role) -> count
        try:
            for role in roles:
                _count(role, program(role), {1: 1}, name, totals)
            sides = []
            for producer in (True, False):
                counts = [totals.get((producer, r), {}) for r, _ in chan.bound[producer]]
                if chan.join[producer] and any(x != counts[0] for x in counts):
                    raise _Opaque
                sides.append(counts[0] if chan.join[producer] else reduce(_plus, counts, {}))
        except _Opaque:
            continue
        diff = _plus(*sides, -1)
        if set(diff) == {1}:
            return (
                f"unbalanced epochs: the fills and takes of {name} differ by {abs(diff[1])} for "
                "every input (both counts are affine in the same runtime values)."
            )
    return None


def _count(role, body, scale: dict | None, name: str, totals: dict) -> None:
    """Add the fills and takes of channel ``name`` in ``body``, each run ``scale`` times (None:
    a region the counts do not model)."""
    for c in body:
        if isinstance(c, tuple):
            r = role.regions.get(c[0])
            inner = None
            if scale is not None and r is not None and (r.bounds or ())[2:] == (("c", 1),):
                try:
                    inner = _times(scale, _trips(*r.bounds[:2]))
                except _Opaque:
                    pass
            _count(role, c[1], inner, name, totals)
            continue
        d = c.data
        if c.kind in ("open", "close") and d["handle"].chan.name == name:
            if not d["handle"].producer or (c.kind == "open" and not d.get("once")):
                continue
            opaque = c.kind == "open" or d.get("once") or d.get("retire") or not d["signal"]
        elif c.kind == "op" and d["chan"] == name and d["op"] in ("wait", "skip", "state"):
            opaque = d["op"] != "wait" or d.get("span") or d.get("count") != 1
        else:
            continue
        if opaque or scale is None:
            raise _Opaque
        key = (c.kind != "op", role.name)
        totals[key] = _plus(totals.get(key, {}), scale)


def _related(names: list) -> bool:
    """Whether two leaves share a value, so no assignment of them is known to be possible."""

    seen: set = set()
    for name in names:
        mine = set(dependencies(name))
        if mine & seen:
            return True
        seen |= mine
    return False


def _found(finding: str, related: bool) -> Liveness:
    """A definite finding, unless the assignment it needs may be impossible."""
    if related:
        return Liveness(
            "unverified", f"{finding} Some runtime values are related, so this may not occur"
        )
    return Liveness("unverified", findings=[finding])


def _counts(m: int, vals: list) -> str:
    parts = [f"{m} work item{'s' * (m != 1)} per CTA"] if m != 1 else []
    if vals:
        parts.append(f"runtime values {vals}")
    return f" (with {', '.join(parts)})" if parts else ""


def _balance(ops: dict, chans: dict) -> str | None:
    """The first channel whose fills and takes disagree in this assignment."""
    fills, takes = {}, {}
    for unit, nodes in ops.items():
        for n in nodes:
            if n.chan not in chans:
                continue
            if n.kind == "commit":
                for t in chans[n.chan].targets(unit, True):
                    key = (n.chan, n.slot, t)
                    fills.setdefault(key, {}).setdefault(unit.role.name, set()).add(n.epoch)
            elif n.kind == "wait":
                key = (n.chan, n.slot, unit.rank)
                takes.setdefault(key, {}).setdefault(unit.role.name, set()).add(n.epoch)
    for key in sorted(set(fills) | set(takes), key=repr):
        name, slot, _ = key
        chan = chans[name]
        label = name if slot is None else f"{name}[{slot}]"
        by = fills.get(key, {})
        filled = set().union(*by.values())
        if chan.join[True] and len({frozenset(e) for e in by.values()}) > 1:
            return f"producers of {label} fill different epochs"
        got = takes.get(key, {})
        if chan.join[False]:
            for role, epochs in got.items():
                if epochs != filled:
                    return f"{role} takes {len(epochs)} epochs of {label}, {len(filled)} filled"
            continue
        taken = [e for epochs in got.values() for e in epochs]
        if sorted(taken) != sorted(filled):
            return f"the consumers of {label} take {len(taken)} epochs, {len(filled)} filled"
    return None


def _generations(graph, chans: dict, where: str) -> str | None:
    """Check that every barrier stays within one generation of its waiters.

    A parity wait for generation g is exact only once generation g - 1 completed, and an
    arrival counts toward the generation in progress. So every arrival of generation g - 1
    must happen before each wait of generation g (mbarrier rings) and before each arrival of
    generation g (any barrier). This one rule is what round-robin stage counts, exclusive-ring
    rendezvous, turnstiles over named stages and reused named barriers each guarantee.
    """
    nodes = graph.nodes
    index = {n: i for i, n in enumerate(nodes)}
    before, deferred = _happens_before(graph, chans)
    signals: dict = {}  # (chan, slot, kind, epoch, rank) -> node indices
    for i, n in enumerate(nodes):
        if n.kind in ("commit", "release") and n.chan in chans:
            for t in chans[n.chan].targets(n.unit, n.kind == "commit"):
                signals.setdefault((n.chan, n.slot, n.kind, n.epoch, t), []).append(i)

    def ordered(i, j) -> bool:
        return bool(before[j] >> i & 1)

    def entry(j):
        """Where a sync or blocking take arrives: after its last program-order predecessor
        that is not an asynchronous signal (at least its unit's start)."""
        j = graph.prev[j]
        while j in deferred:
            j = graph.prev[j]
        return j

    def precedes(a, b) -> bool:
        return a == b or ordered(a, b)

    def unsure(i, point) -> bool:
        """Whether signal ``i`` may precede ``point`` at a placement the trace does not see: a
        TMA or raw signal can arrive any time after its take."""
        early = nodes[i].early
        return early is not None and precedes(index[early], point)

    gens: dict = {}  # barrier -> generation -> [(node, its arrival point)]

    def join(barrier, g, node, point):
        gens.setdefault(barrier, {}).setdefault(g, []).append((node, point))

    for j, n in enumerate(nodes):
        chan = chans.get(n.chan)
        if chan is None:
            if n.kind in ("sync", "arrive"):  # a NamedBarrier generation: each user's g-th use
                join((n.chan, n.unit.rank), n.epoch, j, j if n.kind == "arrive" else entry(j))
            continue
        lap = 1 if chan.fixed else chan.n
        stage = (n.chan, n.slot, n.unit.rank)
        if chan.named:  # a full and an empty bar.sync per stage: arrivals and syncs both count
            if n.kind in ("commit", "wait"):
                join((*stage, "full"), n.epoch, j, j if n.kind == "commit" else entry(j))
            elif n.kind in ("release", "acquire"):
                g = n.epoch + lap if n.kind == "release" else n.epoch
                join((*stage, "empty"), g, j, j if n.kind == "release" else entry(j))
            continue
        if n.kind not in ("acquire", "wait", "peek"):
            continue
        # Only what happens before the wait starts counts: its own wait edges assume it exact.
        acquire = n.kind == "acquire"
        kind, prev = ("release", n.epoch - 2 * lap) if acquire else ("commit", n.epoch - lap)
        at = entry(j)
        for i in signals.get((n.chan, n.slot, kind, prev, n.unit.rank), []):
            if not precedes(i, at):
                if unsure(i, at):
                    raise Unknown(f"{_node(n)} is exact only if a TMA or raw signal comes early")
                return (
                    f"parity wait{where}: {_node(n)} is exact only after {_node(nodes[i])}, "
                    "the stage's previous phase, but nothing orders that before the wait, so it "
                    "can pass on the old phase. Order it (a NamedBarrier rendezvous, more stages "
                    "or fewer round-robin instances)."
                )
    for barrier, by_gen in gens.items():
        named = chans.get(barrier[0])
        step = 1 if named is None else 1 if named.fixed else named.n
        for g, members in by_gen.items():
            for old, before_point in by_gen.get(g - step, []):
                for new, point in members:
                    if not precedes(before_point, point):
                        if unsure(old, point):
                            raise Unknown(
                                f"{_node(nodes[new])} is ordered only if a TMA or raw signal "
                                "comes early"
                            )
                        return _overrun(nodes[old], nodes[new], where)
    return None


def _overrun(old, new, where: str) -> str:
    return (
        f"barrier overrun{where}: {_node(new)} can arrive before {_node(old)}, an arrival of the "
        "previous generation, so the barrier may complete without its other side. Order them "
        "(a Turnstile, a sync between arrivals, or a stage per instance)."
    )


def _happens_before(graph, chans: dict) -> tuple[list[int], set]:
    """Bitsets: ``before[j]`` has bit ``i`` when node ``i`` happens before node ``j``; and the
    asynchronous signals.

    Program order does not run past an asynchronous signal (a TMA fill or a tcgen05.commit
    completes later), so its successors are not ordered after it."""
    nodes, preds = graph.nodes, graph.preds
    deferred = {
        i
        for i, n in enumerate(nodes)
        if n.kind in ("commit", "release")
        and n.chan in chans
        and (n.kind == "commit", n.unit.role.name) in chans[n.chan].deferred
    }

    def settle(i):  # the last op of i's unit at or before i that is not asynchronous
        while i in deferred:
            i = graph.prev[i]
        return i

    # Only a wait for an asynchronous signal is ordered after it; program order and barrier
    # arrivals run on from the op before it.
    edges = [
        [
            i
            if i not in deferred
            or (nodes[j].chan == nodes[i].chan and nodes[j].kind in (*TAKES, "peek", "tail"))
            else settle(i)
            for i in ps
        ]
        for j, ps in enumerate(preds)
    ]
    before = [0] * len(nodes)
    done = [False] * len(nodes)
    for start in range(len(nodes)):
        stack = [start]
        while stack:
            j = stack[-1]
            if done[j]:
                stack.pop()
                continue
            todo = [i for i in edges[j] if not done[i]]
            if todo:
                stack.extend(todo)
                continue
            bits = 0
            for i in edges[j]:
                bits |= before[i] | (1 << i)
            before[j], done[j] = bits, True
            stack.pop()
    return before, deferred


def _node(n) -> str:
    verb = "commit(wait=True)" if n.kind == "rendezvous" else f"{n.kind}()"
    what = f"{n.chan}.{verb}"
    if n.kind in ("acquire", "wait", "commit", "release", "peek"):
        what += f" of epoch {n.epoch}"
    return f"{n.unit.name}: {what} at {n.site}"


def _cycle(graph, cycle: list, where: str) -> str:
    # find_cycle walks predecessors; print the cycle as "a waits for b".
    steps = [_node(graph.nodes[i]) for i in cycle]
    return f"deadlock{where}: " + " waits for ".join([*steps, steps[0]]) + "."
