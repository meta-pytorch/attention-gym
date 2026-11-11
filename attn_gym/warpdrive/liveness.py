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

from attn_gym.warpdrive.events import (
    DEBUG,
    MAX_NODES,
    ChannelView,
    LaneDivergent,
    Unit,
    Unknown,
    dependencies,
    find_cycle,
    leaves,
    link,
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
            if ev.kind == "open" and ev.data["handle"].closed_at is None:
                return Liveness("unverified", f"{ev.data['handle'].what} at {ev.site} stays open")
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
