"""The event graph: every role's traced channel and barrier ops, unrolled and linked.

Each role traces once. ``expr.py`` reads its loop bounds and branch conditions; ``unroll``
expands its events for one assignment of runtime values and work-item count. ``link`` adds:

* program order within a role, per CTA it runs in (and per warp, when a branch depends on the
  thread index);
* data: the fill of epoch e by each producer that takes it, then the waits of epoch e;
* reuse: each consumer's release of epoch e before acquire of e + n (e + 1 for ``Slots``);
  the first n ring acquires use the initial capacity;
* named barriers: each user's e-th arrival before that epoch's syncs return;
  a named channel's waits are such syncs;
* turnstiles: each pass before the next seat's turn.

Probes and try tokens add no edges: they never change which epoch a wait takes. A preview
waits for the epoch the next wait takes without advancing. A span takes its epoch at
``span()``, waits in its loop's first trip and signals in the last (or after the loop).

TMA fills complete at the copies, not the commit; ``signal=False`` leaves signaling to raw
code. The graph places these signals at the close to establish progress, or at the take
before reporting a deadlock. Raw roles, raw ``end.state`` access, dynamic skip counts and loops or
branches ``expr.py`` cannot read leave the program unknown (``Unknown``).
"""

from __future__ import annotations

import itertools
import operator
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

from cutlass._mlir import ir

from attn_gym.warpdrive.expr import Reader

TAKES = ("acquire", "wait")
WARP = 32
# Raise analysis failures instead of reporting them unverified (the test suite sets this).
DEBUG = os.environ.get("WARPDRIVE_LIVENESS_DEBUG") == "1"
# Ops one unit may unroll to, and nodes one assignment may link: happens-before is quadratic.
MAX_NODES = 10_000


class Unknown(Exception):
    """The trace does not determine what a role runs."""


@dataclass(frozen=True)
class Region:
    """A dynamic region of a role body. ``kind``: "item" (the work loop), "for", "while",
    "if" or "unknown"; ``arm``: which region of an ``scf.if`` (0: then, 1: else)."""

    kind: str
    arm: int = 0
    bounds: tuple | None = None  # (lower, upper, step) of a for loop; None if unknown
    cond: object = None  # an if's condition; None if unknown
    why: str = ""  # what an unknown bound or condition depends on


@dataclass
class RoleEvents:
    """One role instance's events and decoded regions for alias and liveness checks."""

    name: str
    warps: tuple[int, ...]
    raw: bool
    events: list
    item_key: int | None
    regions: dict = field(default_factory=dict)  # region key -> Region
    ctas: tuple[int, ...] = (0,)
    unread: str = ""  # why reading the regions failed

    @classmethod
    def from_trace(cls, trace, ri, ctas=(0,), regions: bool = True) -> RoleEvents:
        """``regions``: read loop bounds and branch conditions (the liveness check's input)."""
        item = trace.block_key(trace.item_block) if trace.item_block is not None else None
        read, unread = {}, ""
        if regions and not ri.raw:
            try:
                read = _regions(trace, item)
            except Exception as e:  # a model gap must never fail a compile
                if DEBUG:
                    raise
                unread = f"reading the loops of {ri.name} failed: {e!r}"[:300]
        events = list(trace.events)
        return cls(ri.name, tuple(ri.warps), ri.raw, events, item, read, ctas, unread)


def _regions(trace, item) -> dict:
    """Read the regions of ``trace`` while its blocks are live."""
    # Reading operands through CuTeDSL's value casters can build ops; build them in a scratch
    # module, never in the kernel, and erase them while the kernel values they use still exist.
    scratch = ir.Module.create()
    try:
        with ir.InsertionPoint(scratch.body):
            return _read_regions(trace, item)
    finally:
        for op in reversed(list(scratch.body.operations)):
            op.erase()


def _read_regions(trace, item) -> dict:
    reader = Reader(trace.block_key, trace.item_values)
    out = {}
    for block, key, _ in trace._blocks:
        try:
            op = block.owner.operation
            name = op.name
            arm = next(i for i, r in enumerate(op.regions) if any(b == block for b in r.blocks))
        except Exception:  # noqa: BLE001 - an unreadable region is an unknown one
            out[key] = Region("unknown")
            continue
        if key == item:
            out[key] = Region("item")
        elif name == "scf.for":
            bounds, why = [], ""
            for v in list(op.operands)[:3]:
                bounds.append(reader.read(v))
                why = why or reader.why
            known = None not in bounds
            out[key] = Region("for", bounds=tuple(bounds) if known else None, why=why)
        elif name == "scf.if":
            cond = reader.read(op.operands[0])
            out[key] = Region("if", arm, cond=cond, why=reader.why)
        else:
            out[key] = Region("while" if name == "scf.while" else "unknown", why=name)
    return out


# --------------------------------------------------------------------------- expressions

DIVISIONS = ("div", "floordiv", "ceildiv", "rem", "divu", "ceildivu", "remu")
# arith.cmpi predicates: eq ne slt sle sgt sge ult ule ugt uge. An unsigned compare sees a
# negative value as huge.
UNSIGNED = 2**64
COMPARE = (operator.eq, operator.ne, *(operator.lt, operator.le, operator.gt, operator.ge) * 2)


def _div(a, b):  # truncating, as arith.divsi
    q = abs(a) // abs(b) if b else 0
    return q if (a < 0) == (b < 0) else -q


def _trunc(bits, a):  # an i1 is 0 or 1; wider integers are signed
    a %= 2**bits
    return a - 2**bits if bits > 1 and a >= 2 ** (bits - 1) else a


def _sext(bits, a):
    return -(a % 2) if bits == 1 else _trunc(bits, a)


ARITH = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _div,
    "floordiv": lambda a, b: a // b if b else 0,
    "ceildiv": lambda a, b: -(-a // b) if b else 0,
    "rem": lambda a, b: a - b * _div(a, b),
    "min": min,
    "max": max,
    "and": lambda a, b: a & b,
    "or": lambda a, b: a | b,
    "xor": lambda a, b: a ^ b,
    "zext": lambda bits, a: a % 2**bits,
    "sext": _sext,
    "trunc": _trunc,
}
STRUCTURAL = {*ARITH, "select", "cmp", "divu", "ceildivu", "remu", "minu", "maxu"}


def dependencies(e) -> Iterator[tuple]:
    """Nonconstant subexpressions, including ``e`` itself."""
    if isinstance(e, tuple) and e and e[0] != "c":
        yield e
        for arg in e[1:]:
            yield from dependencies(arg)


def leaf(e) -> bool:
    """Whether to sample ``e`` rather than compute it. Sample runtime leaves and divisions
    directly so loop trip counts, not just their operands, vary."""
    if e[0] in ("c", "iv", "tid", "rank", "uni"):
        return False
    if e[0] in DIVISIONS:
        return not any(d[0] in ("iv", "tid", "rank") for d in dependencies(e))
    return e[0] not in STRUCTURAL


def leaves(e, out: set) -> None:
    if not isinstance(e, tuple) or not e:
        return
    if leaf(e):
        out.add(e)
        return
    for x in e[1:]:
        leaves(x, out)


def evaluate(e, env: dict) -> int:
    kind = e[0]
    if kind == "c":
        return e[1]
    if e in env:
        return env[e]
    if kind == "cmp":
        a, b = evaluate(e[2], env), evaluate(e[3], env)
        if e[1] >= 6:
            a, b = a % UNSIGNED, b % UNSIGNED
        return int(COMPARE[e[1]](a, b))
    if kind == "select":
        return evaluate(e[2], env) if evaluate(e[1], env) else evaluate(e[3], env)
    if kind in ARITH:
        return ARITH[kind](evaluate(e[1], env), evaluate(e[2], env))
    if kind in STRUCTURAL:  # unsigned: the signed op, while no operand is negative
        a, b = evaluate(e[1], env), evaluate(e[2], env)
        if a < 0 or b < 0:
            raise Unknown(f"an unsigned {kind} of a negative value")
        return ARITH[kind.removesuffix("u")](a, b)
    raise Unknown(f"no value for {kind}")


# --------------------------------------------------------------------------- unrolling


@dataclass(frozen=True, eq=False)
class Unit:
    """A role instance in one CTA rank, or one of its warps when its branches split them."""

    role: RoleEvents
    rank: int
    warps: tuple[int, ...]

    @property
    def name(self) -> str:
        warps = "" if self.warps == self.role.warps else f" warp {self.warps[0]}"
        return f"{self.role.name}{warps}" + (f" (rank {self.rank})" if self.rank else "")


@dataclass(eq=False)
class Node:
    """A single execution of a traced op; repeated executions remain distinct nodes.

    ``kind`` names the op; a span begins with acquire/wait, and ``rendezvous`` is
    commit(wait=True). ``chan`` names the channel, barrier or turnstile.
    """

    unit: Unit
    kind: str
    chan: str
    slot: object
    epoch: int
    site: str
    early: Node | None = None  # the take where a TMA or raw signal may already have happened
    threads: int | None = None  # a barrier's thread count


def tid_dependent(role: RoleEvents) -> bool:
    return any(d[0] == "tid" for r in role.regions.values() for d in dependencies(r.cond))


def program(role: RoleEvents) -> list:
    """The role's events nested under their regions: ``[event | (key, children)]``."""
    root: list = []
    stack: list = [(None, root)]
    for ev in role.events:
        if ev.kind not in ("op", "open", "close", "barrier", "barrier_arrive", "barrier_native"):
            continue
        keys = [k for k, _ in ev.where]
        depth = 0
        while depth < len(stack) - 1 and depth < len(keys) and stack[depth + 1][0] == keys[depth]:
            depth += 1
        del stack[depth + 1 :]
        for key in keys[depth:]:
            node = (key, [])
            stack[-1][1].append(node)
            stack.append(node)
        stack[-1][1].append(ev)
    return root


def unroll(unit: Unit, env: dict, items: int, chans: dict, named: set) -> list[Node]:
    """The ops ``unit`` runs for one assignment, in order."""
    role, out = unit.role, []
    nxt: dict = {}  # (chan, slot, producer) -> next epoch
    taken: dict = {}  # (chan, slot, producer, order) -> epoch of that traced take's last run
    uses: dict = {}  # barrier -> uses so far
    lanes = range(unit.warps[0] * WARP, unit.warps[0] * WARP + WARP)
    steps = 0

    def end(chan, slot, producer):
        key = (chan.name, slot, producer)
        if key not in nxt:
            nxt[key] = chan.start(role.name, producer)
        return key

    def op(ev):
        d = ev.data
        chan = chans.get(d["chan"])
        if chan is None:  # a turnstile seat
            if d["op"] in ("turn", "pass"):
                out.append(Node(unit, d["op"], d["chan"], None, 0, ev.site))
            return
        slot = d["slot"] if chan.fixed else None
        if d["op"] == "state":
            raise Unknown(f"{role.name} reads or assigns {chan.name}.state at {ev.site}")
        if d["op"] == "skip":
            if d["count"] is None:
                raise Unknown(f"{role.name} skips a dynamic count of {chan.name} at {ev.site}")
            nxt[end(chan, None, False)] += d["count"]
        elif d["op"] in TAKES:
            producer = d["op"] == "acquire"
            key = end(chan, slot, producer)
            taken[(chan.name, slot, producer, d["order"])] = nxt[key]
            if not d.get("span"):
                out.append(Node(unit, d["op"], chan.name, slot, nxt[key], ev.site))
            nxt[key] += 1 if chan.fixed else chan.turns[producer]
        elif d["op"] in ("peek", "tail"):
            epoch = nxt[end(chan, slot, d["op"] == "tail")]
            out.append(Node(unit, d["op"], chan.name, slot, epoch, ev.site))
        elif d["op"] == "rendezvous":  # commit(wait=True): also syncs on the full barrier
            epoch = taken[(chan.name, slot, True, d["order"])]
            out.append(Node(unit, "rendezvous", chan.name, slot, epoch, ev.site))

    def close(ev, env, region):
        rec, chan = ev.data["handle"], chans.get(ev.data["handle"].chan.name)
        once = ev.data.get("once")
        if chan is None or ev.data.get("retire"):
            return
        if once is not None and not env.get((once, region), True):
            return
        slot = rec.slot if chan.fixed else None
        epoch = taken.get((chan.name, slot, rec.producer, rec.order))
        if epoch is None:
            raise Unknown(f"{role.name}: {rec.what} closes an epoch the unroll did not take")
        take = "acquire" if rec.producer else "wait"
        if ev.kind == "open":  # a span's wait, in its loop's first trip
            out.append(Node(unit, take, chan.name, slot, epoch, ev.site))
            return
        kind = "commit" if rec.producer else "release"
        node = Node(unit, kind, chan.name, slot, epoch, ev.site)
        if not ev.data["signal"] or (rec.producer and chan.tma):
            node.early = next(
                (
                    n
                    for n in reversed(out)
                    if n.kind == take
                    and n.chan == chan.name
                    and n.slot == slot
                    and n.epoch == epoch
                ),
                None,
            )
        out.append(node)

    def barrier(ev):
        b = ev.data.get("barrier")
        name = b.name if b is not None else f"named_barrier({ev.data.get('id')})"
        if ev.kind == "barrier_native" or (b is None and ev.data.get("id") is None):
            raise Unknown(f"{role.name} uses barrier {name} through raw code at {ev.site}")
        if name in named:  # a named channel's own barrier: the channel accounts for it
            return
        g = uses.get(name, 0)
        uses[name] = g + 1
        kind = "sync" if ev.kind == "barrier" else "arrive"
        out.append(Node(unit, kind, name, None, g, ev.site, threads=ev.data.get("threads")))

    def walk(children, env, region):
        nonlocal steps
        for c in children:
            steps += 1
            if steps > MAX_NODES:
                raise Unknown(f"{role.name} runs over {MAX_NODES} ops")
            if not isinstance(c, tuple):
                if c.kind == "op":
                    op(c)
                elif c.kind in ("open", "close"):
                    if c.kind == "close" or c.data.get("once") == "first":
                        close(c, env, region)
                else:
                    barrier(c)
                continue
            key, body = c
            r = role.regions.get(key, Region("unknown"))
            if r.kind == "item":
                for _ in range(items):
                    walk(body, env, key)
            elif r.kind == "for":
                if r.bounds is None:
                    raise Unknown(
                        f"{role.name}: the trip count of the loop at {_site(body)} "
                        f"depends on {r.why or 'an unknown value'}"
                    )
                lo, hi, step = (evaluate(b, env) for b in r.bounds)
                trips = range(lo, hi, step) if step > 0 else range(0)
                for n, i in enumerate(trips):
                    trip = {("iv", key): i, ("first", key): n == 0}
                    walk(body, {**env, **trip, ("last", key): n == len(trips) - 1}, key)
            elif r.kind == "if":
                if r.cond is None:
                    raise Unknown(
                        f"{role.name}: the branch at {_site(body)} depends on "
                        f"{r.why or 'an unknown value'}"
                    )
                taken_arm = {bool(evaluate(r.cond, {**env, **_tid(lane)})) for lane in lanes}
                if len(taken_arm) > 1:
                    arrives = _arrives(body, role.name, chans)
                    if arrives == "all":
                        raise LaneDivergent(
                            f"{role.name}: a branch that depends on the lane excludes required "
                            "Threads/Warps or named-barrier arrivals; predicate the data access, "
                            "not the handoff."
                        )
                    if arrives:
                        raise Unknown(
                            f"{role.name}: a branch that depends on the lane at {_site(body)} "
                            "encloses an elected signal or per-lane channel state"
                        )
                    taken_arm = {True}  # some lanes wait or probe: the warp still waits
                if taken_arm.pop() == (r.arm == 0):
                    walk(body, env, region)
            elif r.kind == "while":
                raise Unknown(
                    f"{role.name}: the trip count of the while loop at {_site(body)} is unknown"
                )
            else:
                raise Unknown(
                    f"{role.name}: the region at {_site(body)} ({r.why or 'unreadable'}) is unknown"
                )

    walk(program(role), {**env, ("rank", "nvvm.read.ptx.sreg.cluster.ctarank"): unit.rank}, None)
    return out


def _arrives(body, role: str, chans: dict) -> str | None:
    """How a region's ops depend on the lanes that run them: "all" if one is an arrival every
    thread makes (a ``Threads`` or ``Warps`` signal, a named barrier or named channel sync);
    "some" if one is an elected arrival (a TMA acquire arms the barrier, a TMA or Umma signal)
    or per-lane state the unroll does not model (a skip, a native barrier id)."""
    found = None
    for c in body:
        if isinstance(c, tuple):
            kind = _arrives(c[1], role, chans)
        elif c.kind in ("barrier", "barrier_arrive"):
            kind = "all"
        elif c.kind == "barrier_native":
            kind = "some"
        elif c.kind == "close":
            rec = c.data["handle"]
            chan = chans.get(rec.chan.name)
            if chan is None or not c.data["signal"] or c.data.get("retire"):
                continue
            kind = "some" if not chan.named and (rec.producer, role) in chan.deferred else "all"
        elif c.kind == "op":
            chan, op = chans.get(c.data["chan"]), c.data["op"]
            if chan is None:
                continue
            if op == "rendezvous" or (op in TAKES and chan.named):
                kind = "all"
            elif op == "skip" or (op == "acquire" and chan.tma):
                kind = "some"
            else:
                continue
        else:
            continue
        if kind == "all":
            return kind
        found = found or kind
    return found


def _site(body) -> str:
    """The first traced op's site in a region."""
    for c in body:
        if isinstance(c, tuple):
            return _site(c[1])
        return c.site
    return "?"


class LaneDivergent(Unknown):
    """Only some lanes arrive at a channel handoff or barrier."""


def _tid(t):
    return {("tid", "nvvm.read.ptx.sreg.tid.x"): t, ("tid", "nvvm.read.ptx.sreg.laneid"): t % WARP}


# --------------------------------------------------------------------------- channels


@dataclass
class ChannelView:
    """What linking needs from a channel declaration."""

    name: str
    n: int
    fixed: bool
    named: bool
    tma: bool  # a TMA producer: the copies, not the commit, fill the stage
    join: dict  # producer side? -> every bound instance takes each epoch
    bound: dict  # producer side? -> [(role name, slot or None)]
    turns: dict  # producer side? -> instances taking turns (1 if they join)
    full: dict  # rank -> ranks whose full barrier its producer signal reaches
    empty: dict  # rank -> ranks whose empty barrier its consumer signal reaches
    # (producer side?, role name) whose signal is one elected arrival that completes
    # asynchronously (TMA, tcgen05.commit): nothing the role runs afterwards is ordered after it.
    deferred: frozenset = frozenset()

    @classmethod
    def of(cls, ch) -> ChannelView:
        from attn_gym.warpdrive.channels import Sharing, Tma, Umma

        sides = {True: ("producer", ch.producers), False: ("consumer", ch.consumers)}
        scope = ch.scope
        return cls(
            ch.name,
            ch.n,
            ch.fixed,
            ch.named,
            isinstance(ch.producer, Tma),
            {p: ch.sharing(s) is Sharing.JOIN for p, (s, _) in sides.items()},
            {p: [(ri.name, slot) for ri, slot in b] for p, (_, b) in sides.items()},
            {p: ch.turns(s) for p, (s, _) in sides.items()},
            dict(getattr(scope, "full_targets", None) or {}),
            dict(getattr(scope, "empty_targets", None) or {}),
            frozenset(
                (side == "producer", ri.name)
                for side, bound in sides.values()
                for ri, _ in bound
                if (ch.producer if side == "producer" else ch.mechanism(side, ri.role)) is Umma
                or (side == "producer" and isinstance(ch.producer, Tma))
            ),
        )

    def start(self, role: str, producer: bool) -> int:
        """The first epoch ``role`` takes: its turn among round-robin instances."""
        names = [r for r, _ in self.bound[producer]]
        return names.index(role) if self.turns[producer] > 1 and role in names else 0

    def targets(self, unit: Unit, fill: bool) -> tuple:
        table = self.full if fill else self.empty
        return tuple(table.get(unit.rank, (unit.rank,)))

    def signallers(self, producer: bool, slot, rank: int, src: list, counts: dict) -> int:
        """How many units signal one epoch at ``rank``: every bound unit whose signal reaches
        it when the side joins, else the units of the instance that took the epoch."""
        table = self.full if producer else self.empty
        ranks = [r for r, ts in table.items() if rank in ts] or [rank]
        bound = [r for r, s in self.bound[producer] if s is None or slot is None or s == slot]
        if not self.join[producer]:
            bound = sorted({n.unit.role.name for n in src}) or bound[:1]
        return sum(counts.get((r, k), 0) for r in bound for k in ranks) or 1


# --------------------------------------------------------------------------- linking


@dataclass
class Graph:
    nodes: list
    preds: list  # node index -> predecessor node indices
    starved: list  # (node index, why) of waits whose signals never run


def link(units: dict, chans: dict, bars: dict, turnstiles: dict, early: bool) -> Graph:
    """Join each unit's ops (``units``: Unit -> nodes). ``early`` links a TMA or raw signal at
    its take instead of its close."""
    nodes = [n for ns in units.values() for n in ns]
    index = {n: i for i, n in enumerate(nodes)}
    counts: dict = {}  # (role, rank) -> its units (more than one when split by warp)
    for u in units:
        counts[(u.role.name, u.rank)] = counts.get((u.role.name, u.rank), 0) + 1
    preds = [[] for _ in nodes]
    starved = []
    for ns in units.values():
        for a, b in itertools.pairwise(ns):
            preds[index[b]].append(index[a])

    def at(n):
        return index[n.early] if early and n.early is not None else index[n]

    signals: dict = {}  # (chan, slot, kind, epoch, target rank) -> signal nodes
    waits: dict = {}  # (chan, slot, producer side) -> wait nodes
    for n in nodes:
        chan = chans.get(n.chan)
        if chan is None:
            continue
        if n.kind in ("commit", "release"):
            for t in chan.targets(n.unit, n.kind == "commit"):
                signals.setdefault((n.chan, n.slot, n.kind, n.epoch, t), []).append(n)
        elif n.kind in (*TAKES, "peek", "tail"):
            producer = n.kind in ("acquire", "tail")
            waits.setdefault((n.chan, n.slot, producer), []).append(n)
    for (name, slot, producer), ws in waits.items():
        chan = chans[name]
        lap = 1 if chan.fixed else chan.n
        for w in ws:
            i, rank = index[w], w.unit.rank
            if w.kind == "tail":
                need = range(max(0, w.epoch - lap), w.epoch)
            else:
                need = [w.epoch - lap] if producer else [w.epoch]
            kind = "release" if producer else "commit"
            for e in need:
                if e < 0:
                    continue  # initial capacity
                src = signals.get((name, slot, kind, e, rank), [])
                expected = chan.signallers(not producer, slot, rank, src, counts)
                if len(src) < expected:
                    side = "consumers" if producer else "producers"
                    starved.append((i, f"{len(src)} of {expected} {side} signal epoch {e}"))
                preds[i] += [at(s) for s in src]
    # A named channel's bar.sync, and commit(wait=True), return once every party arrived.
    peers: dict = {}  # (chan, slot, epoch, rank) -> the consumers' syncs
    for (name, slot, producer), ws in waits.items():
        if chans[name].named and not producer:
            for w in ws:
                peers.setdefault((name, slot, w.epoch, w.unit.rank), []).append(w)
    for i, n in enumerate(nodes):
        chan = chans.get(n.chan)
        if chan is None or not chan.named or n.kind not in ("wait", "rendezvous"):
            continue
        rank = n.unit.rank
        others = peers.get((n.chan, n.slot, n.epoch, rank), [])
        expected = chan.signallers(False, n.slot, rank, others, counts)
        if len(others) < expected:
            starved.append((i, f"{len(others)} of {expected} consumers reach epoch {n.epoch}"))
        preds[i] += [j for o in others if o is not n for j in _entry(nodes, o, index)]
        if n.kind == "rendezvous":
            preds[i] += [at(s) for s in signals.get((n.chan, n.slot, "commit", n.epoch, rank), [])]
    _barriers(nodes, index, preds, starved, bars)
    _turnstiles(nodes, index, preds, starved, turnstiles)
    return Graph(nodes, preds, starved)


def _barriers(nodes, index, preds, starved, bars) -> None:
    """Order each epoch's arrivals before its syncs return."""
    gens: dict = {}
    users: dict = {}
    for n in nodes:
        if n.kind in ("sync", "arrive"):
            gens.setdefault((n.chan, n.unit.rank, n.epoch), []).append(n)
            users.setdefault((n.chan, n.unit.rank), set()).add(n.unit)
    for (name, rank, g), members in gens.items():
        everyone = users[(name, rank)]
        threads = bars.get(name, members[0].threads)
        if sum(WARP * len(u.warps) for u in everyone) != threads:
            raise Unknown(f"barrier {name}: its users do not account for its {threads} threads")
        missing = everyone - {m.unit for m in members}
        for m in members:
            if m.kind != "sync":
                continue
            i = index[m]
            if missing:
                who = ", ".join(sorted(u.name for u in missing))
                starved.append((i, f"{name} epoch {g} never completes: {who} do not arrive"))
            for o in members:
                if o is not m:
                    preds[i] += [index[o]] if o.kind == "arrive" else _entry(nodes, o, index)


def _entry(nodes, n, index) -> list:
    """A sync has arrived once its unit reaches it: its program predecessor."""
    i = index[n]
    return [i - 1] if i and nodes[i - 1].unit == n.unit else []


def _turnstiles(nodes, index, preds, starved, turnstiles) -> None:
    for name, parties in turnstiles.items():
        k, count = len(parties), {}
        passes = {}
        for n in nodes:
            if n.chan == name and n.kind in ("turn", "pass"):
                j = parties.index(n.unit.role.name)
                m = count.get((n.unit.rank, j, n.kind), 0)
                count[(n.unit.rank, j, n.kind)] = m + 1
                if n.kind == "pass":
                    passes[(n.unit.rank, m * k + j)] = index[n]
                else:
                    n.epoch = m * k + j
        for n in nodes:
            if n.chan == name and n.kind == "turn" and n.epoch > 0:
                src = passes.get((n.unit.rank, n.epoch - 1))
                if src is None:
                    starved.append((index[n], f"turn {n.epoch} of {name} is never passed"))
                else:
                    preds[index[n]].append(src)


def find_cycle(graph: Graph) -> list | None:
    """Node indices of a dependency cycle, or None."""
    color = [0] * len(graph.nodes)
    for start in range(len(graph.nodes)):
        if color[start]:
            continue
        color[start] = 1
        path, stack = [start], [iter(graph.preds[start])]
        while stack:
            p = next(stack[-1], None)
            if p is None:
                color[path.pop()] = 2
                stack.pop()
            elif color[p] == 1:
                return path[path.index(p) :]
            elif color[p] == 0:
                color[p] = 1
                path.append(p)
                stack.append(iter(graph.preds[p]))
    return None
