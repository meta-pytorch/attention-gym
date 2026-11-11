"""Liveness checks on the event graph: deadlock and epoch balance.

For a fixed input, each role has a fixed op order. A cycle in the modeled dependencies
prevents progress; an acyclic graph completes if every wait has its signals and user work
terminates. The check unrolls and links the roles (``events.py``) to find

* imbalance: a channel whose fills and takes disagree (a joining consumer taking fewer epochs
  than were filled, round-robin or exclusive owners not covering each epoch once). An
  assignment that does not balance is taken as one the kernel never sees. If none balances,
  static counts (no runtime value, one work item) establish an imbalance. Runtime counts stay
  unverified unless symbolic counts prove a constant difference; callers may use counts the
  samples miss (``n == 4``).
* deadlock: a cycle, or a wait whose signals never run.

A passing graph with static trip counts and one work item per CTA is ``verified`` within
the model. Otherwise ``no deadlock found`` names the sampled counts and assumptions.
Unreadable protocols and analysis failures are ``unverified``, not errors.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import warnings
from dataclasses import dataclass, field
from functools import reduce

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
    "unverified" (``detail``: why); ``findings``: definite errors; ``min_stages``: ring ->
    (fewest stages with which the same checks pass, declared stages)."""

    verdict: str
    detail: str = ""
    findings: list = field(default_factory=list)
    min_stages: dict = field(default_factory=dict)

    def line(self) -> str:
        if self.findings:
            return "liveness: error (see the reported finding)"
        word = "assumes " if self.verdict == "no deadlock found" else ""
        line = f"liveness: {self.verdict}" + (f" ({word}{self.detail})" if self.detail else "")
        if self.min_stages:
            stages = ", ".join(f"{n} {k} (declared {d})" for n, (k, d) in self.min_stages.items())
            line += f"\n  min stages for liveness: {stages}"
        return line


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
    balanced, unbalanced, late, runs = False, None, False, []
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
                runs.append(ops)
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
            "unverified", "progress depends on TMA fills or raw signals preceding the handle close"
        )
    stages = _min_stages(runs, chans, bars, turnstiles)
    if not names and not persistent:
        return Liveness("verified", min_stages=stages)
    assumed = []
    if persistent:
        assumed.append(f"{', '.join(map(str, items))} work items per CTA")
    if names:
        assumed.append(f"loop/branch inputs: count {len(names)}, samples {values}")
    if unbalanced:
        assumed.append("values that unbalance a channel never occur")
    return Liveness("no deadlock found", "; ".join(assumed), min_stages=stages)


def _min_stages(runs: list, chans: dict, bars: dict, turnstiles: dict) -> dict:
    """Find each eligible ring's minimum for the checked runs, holding other depths fixed.

    Depths must be multiples of both sides' round-robin counts. More stages cannot block
    progress in this graph, so try all minima together before searching rings separately.
    """
    # A rotating named ring's depth follows its turnstile, not a multiple of its instances.
    rings = {
        n: c
        for n, c in chans.items()
        if not c.fixed and c.n > 1 and not (c.named and max(c.turns.values()) > 1)
    }
    if not rings:
        return {}
    least = {n: math.lcm(*c.turns.values()) for n, c in rings.items()}

    def passes(stages: dict) -> bool:
        trial = {**chans, **{n: dataclasses.replace(chans[n], n=k) for n, k in stages.items()}}
        return all(_passes(ops, trial, bars, turnstiles) for ops in runs)

    if passes(least):
        return {n: (least[n], c.n) for n, c in rings.items()}
    out = {}
    for name, chan in rings.items():
        k = least[name]
        n = next((n for n in range(k, chan.n, k) if passes({name: n})), chan.n)
        out[name] = (n, chan.n)
    return out


def _passes(ops: dict, chans: dict, bars: dict, turnstiles: dict) -> bool:
    try:
        graph = link(ops, chans, bars, turnstiles, early=False)
        return find_cycle(graph) is None and not graph.starved
    except Unknown:
        return False


class _Opaque(Exception):
    """A count outside the symbolic trip-count model."""


def _times(a: dict, b: dict) -> dict:
    if set(a) - {1} and set(b) - {1}:
        raise _Opaque  # a product of two runtime values
    c, x = (a.get(1, 0), b) if not set(a) - {1} else (b.get(1, 0), a)
    return {k: c * v for k, v in x.items() if c * v}


def _trips(lower, upper) -> dict:
    """Count constant bounds exactly; cancel runtime loops only when their bounds match.

    Only constants and kernel arguments are shared safely across roles. Arithmetic can wrap
    at an unrecorded width; special registers, loads and loop indices can differ.
    """
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
                "every input; match the fill and take counts."
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
            "unverified", f"{finding} Related runtime values may make this assignment impossible"
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
