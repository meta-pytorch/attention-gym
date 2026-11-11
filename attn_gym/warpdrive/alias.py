"""Find potential hazards in declared SMEM/TMEM overlaps; never prove safety.

The diagnostic models single-CTA kernels and accesses annotated with ``tmem.view/access``
or ``handle.view/access``. Footprints cover regions or buffer stages, optionally narrowed
by TMEM lanes and columns per warp. A conflict needs overlapping footprints, a write and
two distinct declared names.

Same-channel/slot handles order whole-region handoffs only when accesses complete before
the closing signal. Other pairs use a happens-before graph: channel edges match epochs,
loops unroll twice, and persistent kernels cover two work items (one otherwise).
``RULES`` holds the PTX citations. Raw ordering instructions are legal but invisible.

Reports distinguish:

* ``potential hazard``: a conflicting pair with an established missing edge;
* ``unverified``: an invisible access or edge, a cluster, or unknown epoch correspondence.
  Prologues, peeled iterations, exclusive-ring skips and round-robin instances can make
  traced epoch counts incomparable;
* ``no hazard found``: no hazard among annotated accesses, not a proof.

``alias_check="warn"`` adds the report to ``describe()`` and warns once about findings.
``"strict"`` raises on potential hazards; ``"off"`` skips analysis.
``unchecked("alias", reason=...)`` excludes enclosed annotations. Annotations emit no
IR; ordering wrappers emit only the wrapped instruction. See ``docs/warpdrive/storage.md``.
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, field
from functools import cache
from itertools import combinations

from cutlass import cute

from attn_gym.warpdrive.events import RoleEvents
from attn_gym.warpdrive.protocol import ProtocolError, current_trace

MODES = ("warn", "strict", "off")
PTX = "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html"


@dataclass(frozen=True)
class Rule:
    statement: str
    citation: str


# PTX rules used by the diagnostic; keep citations and semantics together.
RULES = {
    "pipelined": Rule(
        "Same-warp tcgen05 pairs execute in issue order only when pipelined: mma A read -> mma D "
        "write (same cta_group), mma D write -> mma D write (same accumulator), st -> ld, "
        "st -> st, ld -> st. mma D write -> mma A read is not pipelined; different warps never "
        "are.",
        f"PTX ISA 9.5 9.7.18.6.2 Pipelined tcgen05 Instructions "
        f"({PTX}#tcgen05-memory-consistency-model-pipelined-instructions)",
    ),
    "commit": Rule(
        "tcgen05.commit tracks the earlier tcgen05.mma/cp/shift of the issuing thread with the "
        "same cta_group, not tcgen05.ld/st, TMA or other threads' operations.",
        f"PTX ISA 9.5 9.7.18.12.1 tcgen05.commit ({PTX}#tcgen-async-sync-operations-commit) and "
        f"9.7.18.6.2.1 ({PTX}#tcgen05-memory-consistency-model-pipelined-instructions-implicit)",
    ),
    "wait_group_read": Rule(
        "cp.async.bulk.wait_group.read N returns once all but the N most recent bulk groups of "
        "the executing thread finished reading their sources; other threads learn it only "
        "through a later synchronization by that thread.",
        f"PTX ISA 9.5 9.7.10.28.6.2 cp.async.bulk.wait_group "
        f"({PTX}#data-movement-and-conversion-instructions-cp-async-bulk-wait-group)",
    ),
    "proxy": Rule(
        "Generic-proxy and async-proxy (TMA, tcgen05.mma operand) accesses to the same SMEM need "
        "fence.proxy.async in the generic thread before the synchronization.",
        f"PTX ISA 9.5 9.7.10.28.2 Async Proxy ({PTX}#async-proxy) and 9.7.18.6.5 "
        f"({PTX}#tcgen05-memory-consistency-model-smem-access)",
    ),
    "tmem_store": Rule(
        "tcgen05.st completes only at tcgen05.wait::st; a thread handing stored columns to "
        "another thread or to tcgen05.mma waits (warpdrive's tmem_store fence) before it "
        "signals.",
        f"PTX ISA 9.5 9.7.18.6.2.1.2 and 9.7.18.6.3 "
        f"({PTX}#tcgen05-memory-consistency-model-inter-thread-sync)",
    ),
    "complete_tx": Rule(
        "A TMA load's complete-tx publishes only that copy's bytes on its barrier; it does not "
        "order the issuing thread's other memory operations.",
        f"PTX ISA 9.5 8.9.1.1 Asynchronous Operations ({PTX}#program-order-async-operations)",
    ),
    "channel": Rule(
        "A managed signal (commit/release) followed by the matching wait/acquire of the same "
        "channel, slot and epoch orders what the signaller completed before it; the consumers "
        "of one fan-out epoch (and the producers of one fan-in epoch) are mutually unordered.",
        "docs/warpdrive/channels.md (Stages and phases; Counts come from the roles); "
        "PTX ISA 9.5 9.7.15.16.16 mbarrier.arrive "
        f"({PTX}#parallel-synchronization-and-communication-instructions-mbarrier-arrive)",
    ),
    "barrier": Rule(
        "Warps of one role are ordered within a handle only by a barrier over the whole role "
        "between the two accesses (record it with named_barrier). Different warps are never "
        "pipelined; PTX also requires tcgen05.wait::ld/st and tcgen05.fence::before/"
        "after_thread_sync around that sync, which the diagnostic assumes rather than checks.",
        f"PTX ISA 9.5 9.7.18.6.2 and 9.7.18.6.3 "
        f"({PTX}#tcgen05-memory-consistency-model-inter-thread-sync)",
    ),
    "stages": Rule(
        "A ring hands over one stage at a time: overlapping buffers bound to one channel must "
        "overlap only in the same stage.",
        "docs/warpdrive/channels.md (Stages and phases)",
    ),
}

# Threads signals follow each thread's accesses. Warps signals follow bar.warp.sync,
# which orders the warp's lane accesses.
THREAD_SIGNALS = ("Threads", "Warps")
HAZARD = "potential hazard"
RAW_ONLY = "only a raw signal (signal=False) would order them, and it is not visible"
UNVERIFIED = "unverified"
# Unknown epoch correspondence is a modeling gap, not a hazard finding (_order).
EPOCHS = "epochs"


# Recording wrappers: the same instruction as the cute.arch call, plus a trace event


def _record(kind: str, /, **data):
    trace = current_trace()
    if trace is not None:
        trace.event(kind, **data)


def fence_async_shared():
    """``cute.arch.fence_view_async_shared()`` (fence.proxy.async), visible to the diagnostic."""
    cute.arch.fence_view_async_shared()
    _record("fence", kind="smem_async")


def bulk_commit_group():
    """``cute.arch.cp_async_bulk_commit_group()``, visible to the diagnostic."""
    cute.arch.cp_async_bulk_commit_group()
    _record("bulk_commit")


def bulk_wait_group(n: int, *, read: bool = True):
    """``cute.arch.cp_async_bulk_wait_group(n, read=read)``, visible to the diagnostic."""
    cute.arch.cp_async_bulk_wait_group(n, read=read)
    _record("bulk_wait", n=n, read=read)


def named_barrier(barrier_id: int, number_of_threads: int):
    """Emit ``cute.arch.barrier`` and record it for the alias diagnostic.

    Only a barrier over the whole calling role (``number_of_threads == 32 * warps``)
    orders its warps. ``NamedBarrier.sync()`` records the same event with an allocated ID.
    A hand-picked ID owned by the layer, a named channel or a ``NamedBarrier`` raises.
    """
    cute.arch.barrier(barrier_id=barrier_id, number_of_threads=number_of_threads)
    # Static IDs can be checked for collisions with allocated barriers.
    _record(
        "barrier", threads=number_of_threads, id=barrier_id if type(barrier_id) is int else None
    )


# Per-role trace data


@dataclass
class Access:
    role: RoleEvents
    index: int  # event index in role.events
    storage: tuple[str, str]  # ("tmem" | "smem", name)
    slot: int | None
    read: bool
    write: bool
    agent: str
    parts: list  # [(warp | None, lanes | None, span)]
    handles: tuple  # open HandleRecords
    site: str
    handle: object = None  # owns the SMEM stage view

    @classmethod
    def from_event(cls, kernel, role: RoleEvents, index: int) -> Access:
        ev = role.events[index]
        d = ev.data
        return cls(
            role,
            index,
            d["storage"],
            d["slot"],
            d["read"],
            d["write"],
            d["agent"],
            _parts(kernel, ev, role),
            ev.open,
            ev.site,
            d.get("handle"),
        )

    def describe(self) -> str:
        mode = "read/write" if self.read and self.write else "write" if self.write else "read"
        slot = "" if self.slot is None else f"[{self.slot}]"
        return f"{self.role.name} {mode} {self.storage[1]}{slot} by {self.agent} @ {self.site}"


@dataclass
class Finding:
    severity: str
    text: str
    rule: str

    def describe(self) -> str:
        return f"{self.severity}: {self.text} [{self.rule}]"


@dataclass
class AliasResult:
    names: tuple[str, str]
    kind: str  # "tmem" | "smem"
    pairs: int = 0  # conflicting annotated access pairs examined
    findings: list = field(default_factory=list)
    unverified: list = field(default_factory=list)  # reasons

    @property
    def status(self) -> str:
        if any(f.severity == HAZARD for f in self.findings):
            return HAZARD
        if self.unverified or self.findings:
            return UNVERIFIED
        return "no hazard found"

    def describe(self) -> str:
        head = f"  {self.names[0]} ~ {self.names[1]} ({self.kind})  {self.status}"
        if self.status == "no hazard found":
            head += f" among {self.pairs} conflicting annotated access pair(s)"
        lines = [head]
        lines += [f"      {f.describe()}" for f in self.findings]
        lines += [f"      unverified: {r}" for r in dict.fromkeys(self.unverified)]
        return "\n".join(lines)


@dataclass
class AliasReport:
    results: list

    def describe(self) -> str:
        if not self.results:
            return ""
        head = "Alias diagnostic (bug finder: reports potential hazards; never proves safety)"
        return "\n".join([head, *(r.describe() for r in self.results)])

    @property
    def hazards(self) -> list:
        return [r for r in self.results if r.status == HAZARD]

    @property
    def flagged(self) -> list:
        """Aliases named by warnings: potential hazards or edges only raw signals could supply."""
        return [r for r in self.results if r.findings]


# Declared overlaps and footprints


def _overlap(a: range | None, b: range | None) -> bool:
    return a is None or b is None or (a.start < b.stop and b.start < a.stop)


def declared_pairs(kernel) -> list[tuple[str, tuple[str, str]]]:
    pairs = []
    plan = kernel.tmem_plan
    if plan is not None:
        for x, y in combinations(plan.regions.values(), 2):
            if x.alias_of is None and y.alias_of is None:
                continue  # non-alias regions are disjoint
            spans = [
                range(plan.col(r.name, j), plan.col(r.name, j) + r.cols)
                for r in (x, y)
                for j in range(r.slots)
            ]
            xs, ys = spans[: x.slots], spans[x.slots :]
            if any(_overlap(p, q) for p in xs for q in ys):
                pairs.append(("tmem", (y.name, x.name)))
    for (xn, x), (yn, y) in combinations(kernel._ws_smem.items(), 2):
        if x.buffer.root is y.buffer.root and _overlap(
            range(x.nbytes) if x.nbytes is not None else None,
            range(y.nbytes) if y.nbytes is not None else None,
        ):
            pairs.append(("smem", (yn, xn)))
    return pairs


def _span(value, warp: int) -> range:
    return value(warp) if callable(value) else value


def _parts(kernel, ev, role: RoleEvents) -> list:
    kind, name = ev.data["storage"]
    if kind == "tmem":
        plan = kernel.tmem_plan
        base, width = plan.col(name, ev.data["slot"]), plan.regions[name].cols
        parts = []
        warps = ev.data["warps"] or role.warps
        if not set(warps) <= set(role.warps):
            raise ProtocolError(
                f"{role.name}: access to {name} @ {ev.site} names warps {warps}, outside the "
                f"role's warps {role.warps}"
            )
        for w in warps:
            lanes = ev.data["lanes"]
            if lanes is None:
                lanes = (
                    range(128)
                    if ev.data["agent"] == "umma"
                    else range(32 * (w % 4), 32 * (w % 4) + 32)
                )
            cols = ev.data["cols"]
            cols = range(width) if cols is None else _span(cols, w)
            lanes = _span(lanes, w)
            if cols.start < 0 or cols.stop > width or lanes.start < 0 or lanes.stop > 128:
                raise ProtocolError(
                    f"{role.name}: access to {name} @ {ev.site}: warp {w}'s footprint (lanes "
                    f"{lanes}, cols {cols}) leaves the region (128 lanes, {width} cols)"
                )
            parts.append((w, lanes, range(base + cols.start, base + cols.stop)))
        return parts
    res = kernel._ws_smem[name]
    # None for a dynamic ring stage or a layout without static stage spans.
    span = res.stage_span(ev.data["slot"]) if ev.data["slot"] is not None else None
    return [(None, None, range(*span) if span is not None else None)]


def _conflicts(a: Access, b: Access) -> list[tuple]:
    """Warp pairs whose footprints overlap (``(None, None)`` for SMEM)."""
    return [
        (wa, wb)
        for wa, la, sa in a.parts
        for wb, lb, sb in b.parts
        if _overlap(la, lb) and _overlap(sa, sb)
    ]


# Completion of one access at a signal (the first hop of an ordering edge)


def _between(role: RoleEvents, seq: list[int], lo: int, hi: int):
    """Events of ``role`` at positions ``lo < p < hi`` of ``seq`` (event indices)."""
    return [role.events[seq[p]] for p in range(lo + 1, hi)]


def _bulk_done(role: RoleEvents, seq: list[int], pos_a: int, pos_e: int) -> bool:
    """Check whether a bulk wait before ``pos_e`` covers the TMA store's group at ``pos_a``."""
    groups = 0  # commits since the store, which joins the first of these groups
    for p in range(pos_a + 1, pos_e):
        ev = role.events[seq[p]]
        if ev.kind == "bulk_commit":
            groups += 1
        elif ev.kind == "bulk_wait" and groups and groups > ev.data["n"]:
            return True
    return False


def _completion(a: Access, seq: list[int], pos_a: int, pos_e: int, other: str) -> Finding | None:
    """Return the missing completion edge, or None if ``a`` is published by the signal.

    ``pos_a`` indexes the access and ``pos_e`` its candidate close in ``seq``.
    ``other`` is the conflicting agent that would synchronize with that signal.
    """
    close = a.role.events[seq[pos_e]]
    rec = close.data["handle"]
    chan = rec.chan
    side = "producer" if rec.producer else "consumer"
    mech = chan.mechanism(side, rec.role)
    mech_name = getattr(mech, "__name__", type(mech).__name__)
    fence = chan.fences(side, rec.role)
    where = f"{a.describe()} before {chan.name} {close.site}"
    if not close.data["signal"]:
        return Finding(UNVERIFIED, f"{where}: raw signal (signal=False) is not visible", "channel")
    if a.agent == "tma" and a.write:
        if rec is a.handle and rec.producer and mech_name == "Tma":
            return None
        return Finding(
            HAZARD,
            f"{where}: a TMA load is published only by the transaction count of the handle "
            f"whose barrier it uses, not by this {mech_name} {side} signal",
            "complete_tx",
        )
    if a.agent == "tma":
        if mech_name not in THREAD_SIGNALS:
            return Finding(
                HAZARD,
                f"{where}: nothing waits for the TMA store's source reads",
                "wait_group_read",
            )
        if _bulk_done(a.role, seq, pos_a, pos_e):
            return None
        return Finding(
            UNVERIFIED,
            f"{where}: no recorded bulk_wait_group(read=True) covers the store before the signal",
            "wait_group_read",
        )
    if a.agent == "umma":
        if mech_name == "Umma":
            return None
        return Finding(
            HAZARD,
            f"{where}: tcgen05.mma completes only through tcgen05.commit, not this {mech_name} "
            f"{side} signal",
            "commit",
        )
    if mech_name not in THREAD_SIGNALS:
        return Finding(
            HAZARD,
            f"{where}: a {mech_name} {side} signal does not publish this thread's accesses",
            "commit" if mech_name == "Umma" else "complete_tx",
        )
    if a.storage[0] == "tmem" and a.write and "tmem_store" not in fence:
        return Finding(
            UNVERIFIED,
            f"{where}: tcgen05.st is not waited (channel tmem_store fence) before the signal "
            f"that hands its columns to a later {other} access",
            "tmem_store",
        )
    if other in ("tma", "umma") and a.storage[0] == "smem":
        fences = {e.data["kind"] for e in _between(a.role, seq, pos_a, pos_e) if e.kind == "fence"}
        if "smem_async" not in fence and "smem_async" not in fences:
            return Finding(
                UNVERIFIED,
                f"{where}: no fence.proxy.async between a generic access and a later {other} "
                "access (channel smem_async fence or fence_async_shared())",
                "proxy",
            )
    return None


# Same-thread ordering inside one handle period


def _pipelined(first: Access, second: Access) -> bool:
    if first.agent == second.agent == "threads":
        return True  # SMEM program order; TMEM st -> ld, st -> st, ld -> st
    # Conflicts are between distinct declared names, never the same accumulator.
    return (
        first.storage[0] == "tmem"
        and first.agent == second.agent == "umma"
        and first.read
        and not first.write  # A-operand read -> D write
    )


def _same_period(first: Access, second: Access, warp_pairs, between=None) -> Finding | None:
    """Check program order within one thread set.

    ``between`` contains the separating events; by default it uses one handle's trace order.
    """
    if between is None:
        between = first.role.events[first.index + 1 : second.index]
    barrier = any(
        e.kind == "barrier" and e.data["threads"] == 32 * len(first.role.warps) for e in between
    )
    for wa, wb in warp_pairs:
        if wa != wb:  # TMEM warp indices, or (None, None) for SMEM
            if not barrier:
                return Finding(
                    UNVERIFIED,
                    f"{first.describe()} and {second.describe()}: warps {wa} and {wb} touch the "
                    "same TMEM lanes and columns in one handle with no recorded named_barrier "
                    "between them",
                    "barrier",
                )
        elif not _pipelined(first, second):
            return Finding(
                HAZARD,
                f"{first.describe()} then {second.describe()}: not a pipelined pair in one "
                "thread, and no completion wait between them",
                "pipelined",
            )
    return None


# Whole-region handoff through one channel


def _handoff_handles(acc: Access) -> list:
    """Return handles that hand over all of ``acc``'s storage.

    A stage view uses its own channel's handle. Other accesses can use any open Slots or
    one-stage Ring handle. A deeper Ring orders epochs e and e + n, not consecutive epochs,
    so it can hand over only storage indexed by its stage.
    """
    if acc.handle is not None:
        return [acc.handle]
    # Round-robin epochs interleave roles, so this program-order handoff cannot match them.
    # _Graph._link uses guessed edges instead.
    return [
        h
        for h in acc.handles
        if not h.chan.internal and not h.chan.rotates and (h.chan.fixed or h.chan.n == 1)
    ]


def _handle_on(acc: Access, chan, slot):
    return next((h for h in _handoff_handles(acc) if h.chan is chan and h.slot == slot), None)


def _close_after(role: RoleEvents, index: int, rec) -> int | None:
    for i in range(index + 1, len(role.events)):
        ev = role.events[i]
        if ev.kind == "close" and ev.data["handle"] is rec:
            return i
    return None


def _open_before(role: RoleEvents, index: int, rec) -> int:
    for i in range(index - 1, -1, -1):
        ev = role.events[i]
        if ev.kind == "open" and ev.data["handle"] is rec:
            return i
    return -1


def _handoff(a: Access, b: Access, chan, slot, warp_pairs) -> Finding | None:
    ha, hb = _handle_on(a, chan, slot), _handle_on(b, chan, slot)
    if a.role is b.role:
        # Same-role handles separate accesses only when the first closes before the second
        # opens. Overlapping lifetimes share one period.
        first, hf, second, hs = (a, ha, b, hb) if a.index < b.index else (b, hb, a, ha)
        close = _close_after(first.role, first.index, hf)
        separated = (
            hf is not hs
            and close is not None
            and close < _open_before(second.role, second.index, hs)
        )
        if not separated:
            found = _same_period(first, second, warp_pairs)
            if found is not None:
                return found
    elif ha.producer == hb.producer and not (chan.exclusive and not ha.producer):
        # Exclusive consumers own separate epochs. Producer reuse orders them once each
        # access completes before its handle closes.
        side = "producers" if ha.producer else "consumers"
        return Finding(
            HAZARD,
            f"{a.describe()} and {b.describe()}: {side} of one {chan.name} epoch are unordered",
            "channel",
        )
    # Other epochs: each access must complete before its own handle's close signals.
    for x, hx, other in ((a, ha, b), (b, hb, a)):
        seq = list(range(len(x.role.events)))
        close = _close_after(x.role, x.index, hx)
        if close is None:
            return Finding(
                UNVERIFIED, f"{x.describe()}: its {chan.name} handle never closes", "channel"
            )
        found = _completion(x, seq, x.index, close, other.agent)
        if found is not None:
            return found
    return None


# Cross-channel happens-before graph


UNROLL_LIMIT = 100_000  # events per role after unrolling (each loop level doubles it)
SEARCH_LIMIT = 5_000_000  # graph states visited per compile


class _TooLarge(Exception):
    pass


def _unroll(role: RoleEvents, items: int) -> list[tuple[int, int]]:
    """Return ``(event index, item)`` pairs in program order.

    Dynamic loops repeat twice, the work loop ``items`` times, and branches once.
    Prologue events use item -1; epilogue events use item ``items``. A span's open
    (``once="first"``) and in-loop close (``once="last"``) appear only in their loop's first
    and last repetition.
    """
    root: list = []
    stack: list = [(None, root)]
    for i, ev in enumerate(role.events):
        keys = [k for k, _ in ev.where]
        depth = 0
        while depth < len(stack) - 1 and depth < len(keys) and stack[depth + 1][0] == keys[depth]:
            depth += 1
        del stack[depth + 1 :]
        for key, kind in ev.where[depth:]:
            node = (key, kind, [])
            stack[-1][1].append(node)
            stack.append((key, node[2]))
        stack[-1][1].append(i)
    out: list = []

    def emit(children, item, first=True, last=True):
        for c in children:
            if len(out) > UNROLL_LIMIT:
                raise _TooLarge(f"role {role.name} unrolls past {UNROLL_LIMIT} events")
            if isinstance(c, int):
                once = role.events[c].data.get("once")
                if once is None or (first if once == "first" else last):
                    out.append((c, item))
            elif c[0] == role.item_key:
                for it in range(items):
                    emit(c[2], it, it == 0, it == items - 1)
            else:
                repetitions = 2 if c[1] == "loop" else 1
                for r in range(repetitions):
                    emit(c[2], item, r == 0, r == repetitions - 1)

    item = -1
    for c in root:
        if not isinstance(c, int) and c[0] == role.item_key:
            emit([c], item)
            item = items
        else:
            emit([c], item)
    return out


class _Graph:
    def __init__(self, roles: list[RoleEvents], items: int):
        self.roles = roles
        self.seq = {id(r): _unroll(r, items) for r in roles}
        self.pos = {}  # (role id, event index) -> positions
        for r in roles:
            for p, (e, _) in enumerate(self.seq[id(r)]):
                self.pos.setdefault((id(r), e), []).append(p)
        self.edges: dict = {}  # (role id, pos) -> [(role id, pos)]
        # signal=False leaves signaling to raw code. Pairs ordered only by these edges
        # are unverified, not potential hazards.
        self.raw_edges: dict = {}
        # Unmatched epochs: try each close against opens in this item and the next, then
        # prune contradictions with established order. Pairs needing these edges have unknown
        # correspondence and remain unverified.
        self.guess_edges: dict = {}  # (role id, pos) -> [(role, pos, reason)]
        self.steps = 0  # search work, bounded by SEARCH_LIMIT
        self.cache: dict = {}  # (access, position, other agent) -> (reachable, failures, any)
        self._link(items)
        self._prune_guesses()

    def _ops(self, role, want):
        """Group positions by ``(channel, slot, producer)`` and work item.

        ``want`` selects opens or closes; retired stages publish nothing and are excluded.
        """
        out: dict = {}
        for p, (e, item) in enumerate(self.seq[id(role)]):
            ev = role.events[e]
            if ev.kind != want:
                continue
            rec = ev.data["handle"]
            if ev.data.get("retire"):  # a retired stage publishes nothing
                continue
            if rec.chan.internal:
                continue
            key = (rec.chan.name, rec.slot, rec.producer)
            out.setdefault(key, {}).setdefault(item, []).append(p)
        return out

    def _link(self, items: int):
        opens = {id(r): self._ops(r, "open") for r in self.roles}
        closes = {id(r): self._ops(r, "close") for r in self.roles}
        chans = {
            ev.data["handle"].chan.name: ev.data["handle"].chan
            for r in self.roles
            for ev in r.events
            if ev.kind == "open"
        }
        span = range(-1, items + 1)
        for src in self.roles:
            for (name, slot, producer), by_item in closes[id(src)].items():
                chan = chans[name]
                for dst in self.roles:
                    if dst is src:
                        continue
                    waits = opens[id(dst)].get((name, slot, not producer))
                    if not waits:
                        continue
                    label = name if slot is None else f"{name}[{slot}]"
                    if chan.exclusive or chan.rotates:
                        # Runtime skips choose exclusive owners; round-robin instances
                        # interleave epochs j, j + k, ... . Traced counts cannot match either.
                        why = (
                            f"{label}: exclusive ring, epoch owners follow runtime skips"
                            if chan.exclusive
                            else f"{label}: round-robin ring, epochs alternate between instances"
                        )
                        for item in span:
                            self._guess(src, by_item.get(item, []), dst, waits, item, why)
                    elif producer:  # publication: commit k -> wait k of the same item
                        for item in span:
                            cs, ws = by_item.get(item, []), waits.get(item, [])
                            if len(cs) != len(ws) and cs:
                                # Only the last wait is known to follow every traced commit.
                                # Other pairings need guesses.
                                why = _mismatch(
                                    label, src, by_item, "commits", dst, waits, "waits"
                                )
                                self._guess(src, cs, dst, waits, item, why)
                            if not ws:
                                continue
                            for k, c in enumerate(cs):
                                w = ws[k] if len(cs) == len(ws) else ws[-1]
                                self._edge(src, c, dst, w)
                    else:
                        self._reuse(src, by_item, dst, waits, chan, label, span)

    def _reuse(self, src, releases, dst, acquires, chan, label, span):
        """Link release of epoch e to acquire of e + stages (one stage per Slots slot)."""
        equal = all(len(releases.get(i, [])) == len(acquires.get(i, [])) for i in span)
        flat_r = [p for i in span for p in releases.get(i, [])]
        flat_a = [p for i in span for p in acquires.get(i, [])]
        if equal:
            # A ring is exact only when no channel op sits in an inner loop.
            if not chan.fixed and any(
                len(releases.get(i, []))
                != len({self.seq[id(src)][p][0] for p in releases.get(i, [])})
                for i in span
            ):
                return
            distance = 1 if chan.fixed else chan.n
            for release, acquire in zip(flat_r, flat_a[distance:]):
                self._edge(src, release, dst, acquire)
            return
        # Prologues and peeled iterations leave correspondence unknown. For Slots, the next
        # item's first acquire still follows every release; other pairings need guesses.
        why = _mismatch(label, src, releases, "releases", dst, acquires, "acquires")
        for i in span:
            self._guess(src, releases.get(i, []), dst, acquires, i, why)
            nxt = acquires.get(i + 1, [])
            for r in releases.get(i, []) if chan.fixed and nxt else []:
                self._edge(src, r, dst, nxt[0])

    def _edge(self, src, p, dst, q):
        close = src.events[self.seq[id(src)][p][0]]
        edges = self.edges if close.data["signal"] else self.raw_edges
        edges.setdefault((id(src), p), []).append((dst, q))

    def _guess(self, src, closes, dst, opens, item, why):
        """Every close of ``item`` may precede any matching open of that item or the next."""
        targets = opens.get(item, []) + opens.get(item + 1, [])
        for p in closes:
            self.guess_edges.setdefault((id(src), p), []).extend((dst, q, why) for q in targets)

    def _prune_guesses(self):
        """Drop guesses that contradict established or raw ordering.

        An open that precedes a close cannot be that close's matching wait.
        """
        after: dict = {}  # (role id, pos) -> what the established order puts after it
        for key, guesses in list(self.guess_edges.items()):
            kept = []
            for dst, q, why in guesses:
                if (id(dst), q) not in after:
                    after[id(dst), q] = self._forward(id(dst), q)
                if key not in after[id(dst), q]:
                    kept.append((dst, q, why))
            if kept:
                self.guess_edges[key] = kept
            else:
                del self.guess_edges[key]

    def _next_search_state(self, todo: deque[tuple]) -> tuple:
        self.steps += 1
        if self.steps > SEARCH_LIMIT:
            raise _TooLarge(f"the ordering search passed {SEARCH_LIMIT} steps")
        return todo.popleft()

    def _channel_edges(self, rid: int, p: int, raw: bool) -> list[tuple[RoleEvents, int]]:
        edges = self.edges.get((rid, p), [])
        return edges + self.raw_edges.get((rid, p), []) if raw else edges

    def _forward(self, rid, p) -> set:
        """Return instances ``(role id, position)`` reachable by established or raw ordering."""
        seen, todo = {(rid, p)}, deque([(rid, p)])
        while todo:
            rid, p = self._next_search_state(todo)
            nxt = [(rid, p + 1)] if p + 1 < len(self.seq[rid]) else []
            nxt += [(id(dst), q) for dst, q in self._channel_edges(rid, p, raw=True)]
            for state in nxt:
                if state not in seen:
                    seen.add(state)
                    todo.append(state)
        return seen

    def reachable(self, a: Access, pos_a: int, check, raw: bool = False) -> set:
        """Return instances reachable after a completed signal leaves ``a``'s role.

        ``check(pos_e)`` must hold at that first signal. ``raw=True`` also treats
        ``signal=False`` closes as signals. Instances are ``(role id, position)`` pairs.
        """
        start = (id(a.role), pos_a, False)
        seen, todo = {start}, deque([start])
        while todo:
            rid, p, left = self._next_search_state(todo)
            nxt = []
            if p + 1 < len(self.seq[rid]):
                nxt.append((rid, p + 1, left))
            for dst, q in self._channel_edges(rid, p, raw):
                if not left and not check(p):
                    continue
                nxt.append((id(dst), q, True))
            for state in nxt:
                if state not in seen:
                    seen.add(state)
                    todo.append(state)
        return {(rid, p) for rid, p, left in seen if left}

    def guessed(self, a: Access, pos_a: int):
        """Return a function giving the guessed edges needed to reach each target.

        The search ignores completion and treats ``signal=False`` closes as signals. The
        result is a tuple of guess reasons, or None if no path exists. A guessed path means
        the missing edge may be unknown epoch correspondence, not an established gap.
        Returning to ``a``'s role at or before ``pos_a`` would form a cycle, so those states
        are excluded.
        """
        start = (id(a.role), pos_a, False)
        parent: dict = {start: None}
        todo = deque([start])
        while todo:
            rid, p, left = self._next_search_state(todo)
            nxt = []
            if p + 1 < len(self.seq[rid]):
                nxt.append(((rid, p + 1, left), None))
            for dst, q in self._channel_edges(rid, p, raw=True):
                nxt.append(((id(dst), q, True), None))
            for dst, q, why in self.guess_edges.get((rid, p), []):
                nxt.append(((id(dst), q, True), why))
            for state, why in nxt:
                if state[0] == start[0] and state[1] <= pos_a:
                    continue
                if state not in parent:
                    parent[state] = ((rid, p, left), why)
                    todo.append(state)

        def reasons(target):
            state = (*target, True)
            if state not in parent:
                return None
            used = []
            while parent[state] is not None:
                state, why = parent[state]
                if why is not None:
                    used.append(why)
            return tuple(dict.fromkeys(reversed(used)))

        return reasons


def _mismatch(label, src, src_ops, src_what, dst, dst_ops, dst_what) -> str:
    """Describe unmatched channel epochs using the roles' per-item counts."""

    def counts(ops):
        return "/".join(str(len(ops.get(i, []))) for i in sorted(src_ops.keys() | dst_ops.keys()))

    return (
        f"{label}: {src.name} {src_what} {counts(src_ops)} vs {dst.name} {dst_what} "
        f"{counts(dst_ops)} per traced work item"
    )


def _prefer(first: Finding | None, found: Finding) -> Finding:
    """Keep the first unverified finding unless only it has unknown epoch correspondence."""
    if first is None or (first.rule == EPOCHS and found.rule != EPOCHS):
        return found
    return first


def _cross_channel(graph: _Graph, a: Access, b: Access) -> Finding | None:
    """Check that every unrolled instance pair is ordered in either direction.

    Any potential hazard takes precedence over an earlier unverified finding.
    """
    first = None
    for pa in graph.pos.get((id(a.role), a.index), []):
        for pb in graph.pos.get((id(b.role), b.index), []):
            found = None
            for x, px, y, py in ((a, pa, b, pb), (b, pb, a, pa)):
                ordered, failures = _graph_order(graph, x, px, y, py)
                if ordered:
                    break
                # A direction only a raw signal might order makes the pair unverified at worst;
                # an unknown epoch correspondence yields to any other failure.
                if failures and (
                    found is None
                    or (RAW_ONLY in failures[0].text and found.rule != EPOCHS)
                    or (found.rule == EPOCHS and failures[0].rule != EPOCHS)
                ):
                    found = failures[0]
            else:
                items = (graph.seq[id(a.role)][pa][1], graph.seq[id(b.role)][pb][1])
                # A never-closed handle's own signal might be the missing edge.
                found = (
                    _unclosed(graph, a, b)
                    or found
                    or Finding(
                        HAZARD,
                        f"{a.describe()} (work item {items[0]}) and {b.describe()} (work item "
                        f"{items[1]}): no channel path orders them",
                        "channel",
                    )
                )
                if found.severity == HAZARD:
                    return found
                first = _prefer(first, found)
    return first


def _unclosed(graph: _Graph, a: Access, b: Access) -> Finding | None:
    """Find an unclosed handle that could supply the missing channel edge.

    ``unchecked("unclosed")`` leaves any replacement signal invisible, making the pair
    unverified. Slots match by slot; dynamic ring stages match any handle on that ring.
    """
    stages = {(x.handle.chan.name, x.handle.slot) for x in (a, b) if x.handle is not None}
    for role in graph.roles:
        for ev in role.events:
            rec = ev.data.get("handle") if ev.kind == "open" else None
            if rec is not None and rec.closed_at is None and (rec.chan.name, rec.slot) in stages:
                return Finding(
                    UNVERIFIED,
                    f"{a.describe()} and {b.describe()}: {rec.chan.name} handle opened at "
                    f"{ev.site} never closes, so what signals it is not visible",
                    "channel",
                )
    return None


def _reach(graph: _Graph, x: Access, px: int, other: str) -> tuple:
    """Cache reachability from instance ``px`` of ``x`` for a conflicting ``other`` agent.

    Returns ordered targets, completion failures, targets reachable without completion,
    targets reachable with raw signals, and a function for any further epoch guesses
    (``_Graph.guessed``).
    """
    key = (id(x), px, other)
    if key not in graph.cache:
        seq = [e for e, _ in graph.seq[id(x.role)]]
        failures: list = []
        verdicts: dict = {}

        def check(pos_e):
            if pos_e not in verdicts:
                verdicts[pos_e] = _completion(x, seq, px, pos_e, other)
                if verdicts[pos_e] is not None:
                    failures.append(verdicts[pos_e])
            return verdicts[pos_e] is None

        def raw_or_check(pos_e):
            return not x.role.events[seq[pos_e]].data["signal"] or check(pos_e)

        ordered = graph.reachable(x, px, check)
        anyway = graph.reachable(x, px, lambda pos_e: True) if failures else ordered
        raw = graph.reachable(x, px, raw_or_check, raw=True) if graph.raw_edges else ordered
        search: list = []  # the guess search, run on the first target not reached otherwise

        def guessed(target):
            if not search:
                search.append(graph.guessed(x, px) if graph.guess_edges else lambda t: None)
            return search[0](target)

        graph.cache[key] = (ordered, failures, anyway, raw, guessed)
    return graph.cache[key]


def _graph_order(graph: _Graph, x: Access, px: int, y: Access, py: int) -> tuple[bool, list]:
    """Check whether instance ``px`` of ``x`` orders instance ``py`` of ``y``.

    Also return completion failures, a possible raw signal or required epoch guesses.
    """
    ordered, failures, anyway, raw, guessed = _reach(graph, x, px, y.agent)
    target = (id(y.role), py)
    if target in ordered:
        return True, []
    if target in raw:
        return False, [
            Finding(
                UNVERIFIED,
                f"{x.describe()} before {y.describe()}: {RAW_ONLY}",
                "channel",
            )
        ]
    if target in anyway:
        return False, failures
    reasons = guessed(target)
    if reasons is None:
        return False, []
    if not reasons:  # a raw edge plus a failed completion: no guess involved
        return False, failures or [
            Finding(UNVERIFIED, f"{x.describe()} before {y.describe()}: {RAW_ONLY}", "channel")
        ]
    return False, [
        Finding(
            UNVERIFIED,
            f"{x.describe()} before {y.describe()}: ordered only through channel epochs the "
            f"trace does not match ({'; '.join(reasons)}): unknown epoch correspondence "
            "(prologues, peeled iterations, exclusive-ring skips or round-robin turns change the "
            "traced counts), not an established missing edge",
            EPOCHS,
        )
    ]


def _same_role(graph: _Graph, a: Access, b: Access, warp_pairs) -> Finding | None:
    """Check all same-role instance pairs, including loop-carried pairs.

    Pipelined same-thread operations use program order; different warps need a recorded
    role barrier. Otherwise, the earlier access must complete before a round trip through
    another role. Any potential hazard takes precedence over an unverified finding.
    """
    seq = graph.seq[id(a.role)]
    first = None
    for pa in graph.pos.get((id(a.role), a.index), []):
        for pb in graph.pos.get((id(b.role), b.index), []):
            x, px, y, py = (a, pa, b, pb) if pa < pb else (b, pb, a, pa)
            between = [a.role.events[e] for e, _ in seq[px + 1 : py]]
            found = _same_period(x, y, warp_pairs, between)
            if found is None:
                continue
            ordered, failures = _graph_order(graph, x, px, y, py)
            if not ordered:
                soft = failures and (RAW_ONLY in failures[0].text or failures[0].rule == EPOCHS)
                found = failures[0] if failures and (found.severity != HAZARD or soft) else found
                if found.severity == HAZARD:
                    found = _unclosed(graph, a, b) or found
                if found.severity == HAZARD:
                    return found
                first = _prefer(first, found)
    return first


def _stage_alignment(kernel, x: str, y: str) -> Finding | str | None:
    rx, ry = kernel._ws_smem[x], kernel._ws_smem[y]
    for chan in set(rx.channels) & set(ry.channels):
        if rx.stage_span(0) is None or ry.stage_span(0) is None:
            return f"channel {chan}: stage spans of {x} and {y} are not static and disjoint"
        for s in range(rx.stages):
            for t in range(ry.stages):
                if s != t and _overlap(range(*rx.stage_span(s)), range(*ry.stage_span(t))):
                    return Finding(
                        HAZARD,
                        f"channel {chan}: stage {s} of {x} overlaps stage {t} of {y}",
                        "stages",
                    )
    return None


def check(kernel, roles: list[RoleEvents]) -> AliasReport:
    """Run the diagnostic over the roles traced in one compile."""
    results = []
    pairs = declared_pairs(kernel)
    if not pairs:
        return AliasReport(results)
    accesses: list[Access] = []
    for role in roles:
        for i, ev in enumerate(role.events):
            if ev.kind == "access" and not ev.opted_out:
                accesses.append(Access.from_event(kernel, role, i))
    streams = getattr(kernel, "_ws_streams", {})
    items = 2 if any(st.schedule.persistent for st in streams.values()) else 1

    @cache
    def get_graph() -> _Graph | str:
        try:
            return _Graph([r for r in roles if not r.raw], items)
        except _TooLarge as exc:
            return f"{exc}; not modeled"

    for kind, (x, y) in pairs:
        res = AliasResult((x, y), kind)
        results.append(res)
        if kernel.cluster.size > 1:
            res.unverified.append(
                f"{type(kernel.cluster).__name__} kernels are not modeled: the diagnostic "
                "covers single-CTA kernels"
            )
            continue
        mine = {n: [a for a in accesses if a.storage == (kind, n)] for n in (x, y)}
        for n in (x, y):
            if not mine[n]:
                res.unverified.append(f"no annotated access to {n}")
            for role in roles:
                raw = [
                    e for e in role.events if e.kind == "raw" and e.data["storage"] == (kind, n)
                ]
                if raw and not any(a.role is role for a in mine[n]):
                    res.unverified.append(
                        f"{role.name} acquires {n} without annotating its accesses @ {raw[0].site}"
                    )
        for a in mine[x] + mine[y]:
            if a.role.raw:
                res.unverified.append(
                    f"raw role {a.role.name}: its synchronization is not visible"
                )
        if kind == "smem":
            found = _stage_alignment(kernel, x, y)
            if isinstance(found, str):
                res.unverified.append(found)
            elif found is not None:
                res.findings.append(found)
        for a in mine[x]:
            for b in mine[y]:
                if not (a.write or b.write) or a.role.raw or b.role.raw:
                    continue
                warp_pairs = _conflicts(a, b)
                if not warp_pairs:
                    continue
                res.pairs += 1
                found = _order(a, b, warp_pairs, get_graph)
                if isinstance(found, str):
                    res.unverified.append(found)
                elif found is not None and all(f.text != found.text for f in res.findings):
                    res.findings.append(found)
    return AliasReport(results)


def _order(a: Access, b: Access, warp_pairs, get_graph) -> Finding | str | None:
    """Return a missing edge, an unverified reason (str), or None for ordered accesses.

    One ordering channel shared by both accesses is enough.
    """
    found = []
    ring_stage = a.storage[0] == "smem" and (a.slot is None or b.slot is None)
    # A handle around a loop hides loop-carried pairs on static storage, so same-role
    # accesses need the unrolled graph. Ring stages still use their own handles: each handle
    # covers one epoch.
    for ha in _handoff_handles(a) if a.role is not b.role or ring_stage else ():
        if _handle_on(b, ha.chan, ha.slot) is not None:
            why = _handoff(a, b, ha.chan, ha.slot, warp_pairs)
            if why is None:
                return None
            found.append(why)
    if found and ring_stage:
        return next((f for f in found if f.severity == HAZARD), found[0])
    if ring_stage:
        return f"{a.describe()} and {b.describe()}: ring-stage accesses on different channels"
    # A different channel, later signal or round trip may order these instances even when
    # their shared channel does not.
    graph = get_graph()
    if isinstance(graph, str):
        return graph
    try:
        found = (
            _same_role(graph, a, b, warp_pairs)
            if a.role is b.role
            else _cross_channel(graph, a, b)
        )
    except _TooLarge as exc:
        return f"{a.describe()} and {b.describe()}: {exc}; not modeled"
    if found is not None and found.rule == EPOCHS:
        return found.text  # a modeling gap, like an unmodeled cluster: not a finding
    return found


def report(rep: AliasReport, mode: str, kernel_name: str):
    """Warn about findings, or raise on potential hazards in strict mode."""
    if mode == "strict" and rep.hazards:
        raise ProtocolError(f"warpdrive alias diagnostic (strict):\n{rep.describe()}")
    flagged = rep.flagged
    if flagged:
        names = ", ".join(f"{r.names[0]}~{r.names[1]}" for r in flagged)
        first = flagged[0].findings[0].describe()
        warnings.warn(
            f"warpdrive: alias diagnostic flagged {names} in {kernel_name} (no error raised; see "
            f"describe()): {first}",
            stacklevel=3,
        )
