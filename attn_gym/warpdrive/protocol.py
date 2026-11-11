"""Trace-time handle lifetimes, state lineage, protocol logs and check opt-outs."""

from __future__ import annotations

import contextlib
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import FrameType

import cutlass
from cutlass._mlir import ir

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_CONTEXTLIB = os.path.abspath(contextlib.__file__)  # frames of @contextmanager wrappers
# CuTeDSL calls into this package when it carries a value through a dynamic region.
_CUTEDSL_DIR = os.path.dirname(os.path.abspath(cutlass.__file__)) + os.sep


class ProtocolError(Exception):
    """A role protocol violation detected while declaring or tracing a kernel."""


# Per-site opt-outs: ``with unchecked(check, reason=...)``.
#   tma       copies through barrier pointers obtained inside the block skip atom/byte checks
#   fence     accesses annotated inside the block are left out of the fence check (fences.py)
#   liveness  channel and barrier ops inside the block are invisible to the liveness check,
#             which reports the kernel unverified instead of checking them
# Use a span across a counted loop. ``signal=False`` omits the signal, not the handle close.
# Leave alias accesses unannotated, carry a Token, or use raw barriers.
CHECKS = ("tma", "fence", "liveness")


@dataclass
class OptOut:
    """A check opt-out with its reason, source site and suppressed findings."""

    check: str
    reason: str
    site: str
    role: str | None = None
    suppressed: list = field(default_factory=list)

    def describe(self) -> str:
        what = "; ".join(sorted(set(self.suppressed))) or "nothing (stale opt-out?)"
        who = f"{self.role}: " if self.role else ""
        return f"unchecked {self.check} @ {self.site}: {who}{self.reason!r} -> {what}"


_ACTIVE: list[OptOut] = []  # innermost last; tracing is single-threaded
_SEEN: list[OptOut] = []  # every opt-out entered since the last take_opt_outs()
_ROLE: list = []  # the RoleTrace of the role being traced (alias.py's ordering wrappers)


@contextmanager
def tracing_role(trace):
    """Set the current role trace for the block."""
    _ROLE.append(trace)
    try:
        yield
    finally:
        _ROLE.remove(trace)


def current_trace():
    """The RoleTrace of the role being traced, or None outside a managed role body."""
    return _ROLE[-1] if _ROLE else None


@contextmanager
def unchecked(check: str, *, reason: str):
    """Opt out of one trace check without changing synchronization.

    ``reason`` must be non-empty; ``describe()`` records it with the site and suppressed checks.
    ``CHECKS`` lists accepted kinds. Other checks stay active;
    see ``docs/warpdrive/dropping-down.md``.
    """
    if check not in CHECKS:
        raise ValueError(f"unknown check {check!r}; one of {CHECKS}")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("unchecked(...) needs a non-empty reason")
    opt = OptOut(check, reason, user_site())
    _SEEN.append(opt)
    _ACTIVE.append(opt)
    try:
        yield opt
    finally:
        _ACTIVE.remove(opt)


def active_opt_out(check: str) -> OptOut | None:
    return next((o for o in reversed(_ACTIVE) if o.check == check), None)


def take_opt_outs() -> list[OptOut]:
    """Opt-outs entered since the last call (the kernel collects them once per trace)."""
    seen = list(_SEEN)
    _SEEN.clear()
    return seen


def _user_frame(frame: FrameType | None = None) -> FrameType | None:
    """The innermost frame outside this package and CuTeDSL, or None."""
    frame = sys._getframe(1) if frame is None else frame
    while frame is not None:
        path = os.path.abspath(frame.f_code.co_filename)
        if (
            os.path.dirname(path) != _PACKAGE_DIR
            and path != _CONTEXTLIB
            and not path.startswith(_CUTEDSL_DIR)
        ):
            return frame
        frame = frame.f_back
    return None


def _function(frame) -> str | None:
    """Return the user function containing a generated loop or branch body.

    CuTeDSL names these nested functions like ``fill.<locals>.while_region_0``.
    """
    if frame is None:
        return None
    return frame.f_code.co_qualname.split(".<locals>.")[0].rsplit(".", 1)[-1]


def _frame_site(frame) -> str:
    if frame is None:
        return "?"
    return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"


def user_site() -> str:
    """``file:line`` of the innermost caller outside this package."""
    return _frame_site(_user_frame())


def current_block():
    return ir.InsertionPoint.current.block


def _ancestors(block, root) -> list:
    """``block`` and its enclosing blocks up to ``root`` (innermost first)."""
    chain = []
    try:
        while block is not None:
            chain.append(block)
            if block == root:
                break
            block = block.owner.operation.block
    except (AttributeError, ValueError, RuntimeError):
        pass
    return chain


@dataclass
class _Copy:
    seen: int  # the lineage's advance count when this copy was made or last advanced
    site: str
    function: str | None  # the user function that made it
    callee: str | None  # its bound or standalone spelling, for schematic reassignment advice


class Lineage:
    """Track copies of a loop-carried endpoint or cursor and reject stale phases.

    CuTeDSL rebuilds values when tracing a loop or branch body and its result. A copy is
    stale once a later-created copy has advanced further. This catches a caller dropping
    a helper's updated value, including in a loop body that does not carry it. Branches
    trace in order, so creating a copy without advancing it must not stale older copies:
    the else branch, a read-only helper and an unused ``views()`` all rely on this.

    Phase reads, advances and role returns are checked. Barrier-pointer accessors are not;
    saved snapshots may still supply barriers. A snapshot rebuilt by a region gets a new
    copy ID, so the check cannot detect its stale state.
    """

    def __init__(self):
        self._advances = 0  # of any copy
        self._copies: list[_Copy] = []  # in creation order

    def add(self) -> int:
        """Register a new copy; returns its id."""
        frame = _user_frame()
        function = _function(frame)
        callee = function
        caller = frame
        # A generated loop body may omit self; check its enclosing helper, not the outer role.
        while caller is not None and _function(caller) == function:
            if "self" in caller.f_locals:
                callee = f"self.{function}"
                break
            caller = _user_frame(caller.f_back) if caller.f_back is not None else None
        self._copies.append(_Copy(self._advances, _frame_site(frame), function, callee))
        return len(self._copies) - 1

    def advanced(self, copy: int) -> None:
        """Record an advance, raw-state access or phase assignment by ``copy``."""
        self._advances += 1
        self._copies[copy].seen = self._advances

    def check(
        self, copy: int, trace: RoleTrace, name: str, kind: str, returned_by: str | None = None
    ) -> None:
        """Raise if ``copy`` of ``name`` is stale.

        ``returned_by`` names the returning role at role exit; otherwise this is a use site.
        """
        seen = self._copies[copy].seen
        hit = next((c for c in self._copies[copy + 1 :] if c.seen > seen), None)
        if hit is None:
            return
        if returned_by is not None:
            where, function = "returned at role exit", returned_by
        else:
            frame = _user_frame()
            where, function = f"used at {_frame_site(frame)}", _function(frame)
        if hit.function is not None and hit.function != function:
            fix = (
                f"assign the helper's returned {kind} to the argument you passed "
                f"(schematically: `{name} = {hit.callee}({name}, ...)`; "
                "use your actual argument or alias)"
            )
        else:
            fix = f"use the {kind} the loop or branch carried, not a copy saved before it"
        raise ProtocolError(
            f"{trace.role_name}: the {name} {kind} {where} is stale: a newer copy made at "
            f"{hit.site} (in {hit.function}) has advanced past it, so its phases are lost; "
            f"{fix}"
        )


@dataclass
class Event:
    """A channel, handle, access or ordering event shared by the trace checks.

    ``where`` lists enclosing dynamic regions outermost first as ``(block key, kind)``:
    ``loop`` for ``scf.for``/``scf.while``, ``if`` or ``other``. ``open`` holds the active
    handle records. ``opted_out`` marks an access inside ``unchecked("fence")`` or a channel or
    barrier op inside ``unchecked("liveness")``.
    """

    kind: str
    data: dict
    site: str
    where: tuple
    open: tuple
    opted_out: bool = False


def _region_kind(block) -> str:
    try:
        name = block.owner.operation.name
    except (AttributeError, ValueError, RuntimeError):
        return "other"
    if name in ("scf.for", "scf.while"):
        return "loop"
    return "if" if name == "scf.if" else "other"


class ProtocolLog:
    """Each role instance's channel-op sequence, read from its events.

    Loops trace once, so their ops appear once. ``depth`` counts dynamic regions below the
    role body: 0 for the body, 1 for one loop or branch.
    """

    def __init__(self):
        self.ops: dict[str, list[tuple]] = {}
        self.tmem: dict[str, set] = {}

    def add(self, trace: RoleTrace):
        """Log ``trace``'s channel ops."""
        for ev in trace.channel_ops():
            d = ev.data
            op = d["op"] if d["count"] == 1 else f"{d['op']} {d['count'] or 'n'}"
            line = (d["chan"], op, d["slot"], len(ev.where), ev.site)
            self.ops.setdefault(trace.role_name, []).append(line)

    def describe(self) -> str:
        lines = []
        for role_name, ops in self.ops.items():
            lines.append(f"role {role_name}:")
            for chan, op, slot, depth, site in ops:
                where = "" if slot is None else f"[{slot}]"
                lines.append(f"  {'  ' * depth}{chan + where:<9} {op:<8} @ {site}")
        if self.tmem:
            lines.append("tmem access:")
            for role_name, regions in self.tmem.items():
                lines.append(f"  {role_name}: {', '.join(sorted(regions))}")
        return "\n".join(lines)


@dataclass(eq=False)
class Lifetime:
    """A handle or span until close, or a preview until its adopting wait.

    ``RoleTrace.finish`` checks for resources left live at role exit. Handles close in their
    opening region because tracing sees each region once and does not carry handle copies.
    Spans may close in their counted loop's last trip. Previews may cross regions: the end's
    next wait adopts the outstanding preview.
    """

    what: str
    site: str
    block: object  # the region that took it
    region_bound: bool = True
    closed_at: str | None = None
    closed_op: str | None = None


class RoleTrace:
    """Live-resource and event bookkeeping for one traced role instance."""

    def __init__(self, role_name: str, log: ProtocolLog | None, ranges=None, waits: bool = True):
        self.role_name = role_name
        self.log = log
        self.ranges = ranges  # the trace backend's ranges of this role (tracing.py), or None
        self.trace_waits = ranges is not None and waits
        self.item_block = None  # the work-item loop body (set by the WorkCursor's advance())
        self.live: list[Lifetime] = []  # in the order they were taken
        self.checks: list = []  # end-of-role trace checks (e.g. work cursors)
        self.tma_bars: list = []  # TMA-channel full-barrier pointers handed out (tma.py)
        # One record for the trace checks: channel ops, handle lifetimes, accesses and ordering.
        self.events: list[Event] = []
        self.open_turns: set[str] = set()  # turnstiles whose turn this role holds here
        self._blocks: list = []  # (block, key, kind) of every region an event was recorded in
        self.root = current_block()

    def where(self) -> tuple:
        """The enclosing dynamic regions of the current insertion point, outermost first."""
        out = []
        for block in reversed(self.blocks()[:-1]):
            entry = next((e for e in self._blocks if e[0] == block), None)
            if entry is None:
                entry = (block, len(self._blocks), _region_kind(block))
                self._blocks.append(entry)
            out.append(entry[1:])
        return tuple(out)

    def block_key(self, block) -> int | None:
        """The key ``where()`` gave ``block`` (None if no event was recorded inside it)."""
        return next((key for b, key, _ in self._blocks if b == block), None)

    def event(self, kind: str, /, **data):  # positional: an event may carry a "kind" field
        site = user_site()
        opted_out = False
        if kind == "access":
            opted_out = self.opt_out("fence", f"{data['storage'][1]} access at {site}")
        elif kind in ("op", "barrier", "barrier_arrive"):
            what = data.get("chan") or getattr(data.get("barrier"), "name", "named barrier")
            opted_out = self.opt_out("liveness", f"{what} at {site}")
        ev = Event(kind, data, site, self.where(), tuple(self.open), opted_out)
        self.events.append(ev)
        return ev

    def opt_out(self, check: str, what: str) -> bool:
        """True (and recorded) when an ``unchecked(check)`` block is active here."""
        opt = active_opt_out(check)
        if opt is None:
            return False
        opt.role = opt.role or self.role_name
        opt.suppressed.append(what)
        return True

    def blocks(self) -> list:
        """The current block and its enclosing blocks up to the role body, innermost first."""
        return _ancestors(current_block(), self.root)

    def record(
        self,
        chan,
        op,
        slot,
        count: int | None = 1,
        order: int | None = None,
        span: bool = False,
    ):
        """Record a channel op. ``count``: a skip's epochs (None if dynamic), else 1;
        ``order``: the end's acquires, waits and skips before it (an acquire or wait), or the
        closing handle's (a close); ``span``: an acquire or wait that takes a span's epoch
        without waiting."""
        self.event("op", chan=chan, op=op, slot=slot, count=count, order=order, span=span)

    def channel_ops(self) -> list[Event]:
        return [ev for ev in self.events if ev.kind == "op"]

    def _item_key(self) -> int | None:
        return None if self.item_block is None else self.block_key(self.item_block)

    def uses_per_item(self) -> dict:
        """``{channel: n}``: acquires/waits per work item, 0 when inside an inner loop."""
        uses: dict = {}
        item = self._item_key()
        if item is None:
            return uses
        for ev in self.channel_ops():
            chan, keys = ev.data["chan"], [key for key, _ in ev.where]
            if ev.data["op"] not in ("acquire", "wait") or item not in keys:
                continue
            depth = len(keys) - 1 - keys.index(item)
            if depth == 0 and uses.get(chan) != 0:
                uses[chan] = uses.get(chan, 0) + 1
            elif depth:  # nested inside the item loop: a dynamic count per item
                uses[chan] = 0
        return uses

    def epoch_sequence(self, chan: str) -> tuple[bool | None, list[tuple[str, int]]] | str:
        """Return ``(per_item, [(op, n)])`` for a channel, or why it is not static.

        Acquires, retires, waits and skips must run directly in the role body or work-item
        loop, not both. No other dynamic region or raw ``end.state`` access is allowed.
        ``n`` is a skip's epoch count, a retire's acquire order, or 1. ``per_item`` is True
        inside the work loop, False outside, and None when there are no ops.
        """
        seq, regions = [], set()
        item = self._item_key()
        for ev in self.channel_ops():
            d = ev.data
            op, count, order = d["op"], d["count"], d["order"]
            if d["chan"] != chan or op not in ("acquire", "retire", "wait", "skip", "state"):
                continue
            if op == "state":
                return f"{self.role_name} reads or assigns end.state"
            inner = ev.where  # the enclosing regions below the role body
            in_item = item is not None and len(inner) == 1 and inner[0][0] == item
            if inner and not in_item:
                return f"{self.role_name}'s {op} runs inside a dynamic loop or branch"
            if count is None:
                return f"{self.role_name}'s skip count is dynamic"
            regions.add(bool(inner))
            seq.append((op, order if op == "retire" else count))
        if len(regions) > 1:
            return f"{self.role_name} uses it both inside and outside the work-item loop"
        return (regions.pop() if regions else None), seq

    @property
    def open(self) -> list:
        """The open handles and spans, in the order they opened."""
        return [r for r in self.live if r.region_bound]

    def opened(self, res: Lifetime):
        self.live.append(res)

    def closed(self, res: Lifetime, op: str, *, last_trip: bool = False, hint: str = ""):
        """Record ``op`` ending ``res``. A region-bound resource ending in another region leaves
        both regions' live sets changed. ``last_trip``: a span signalling in its loop's last
        trip, which ends it once per take (``Span`` checks the loop)."""
        site = user_site()
        if res.closed_at is not None:
            raise ProtocolError(
                f"{self.role_name}: {res.what} closed twice: {res.closed_op} at "
                f"{res.closed_at}, then {op} at {site}"
            )
        if res.region_bound and not last_trip and current_block() != res.block:
            raise ProtocolError(
                f"{self.role_name}: {res.what} opened at {res.site} but {op} at "
                f"{site} is in a different dynamic region (loop or branch). Close it in the "
                f"region that opened it{hint}"
            )
        res.closed_at, res.closed_op = site, op
        self.live.remove(res)

    def finish(self):
        for check in self.checks:
            check()
        handles = "; ".join(f"{r.what} opened at {r.site}" for r in self.open)
        if handles:
            raise ProtocolError(
                f"{self.role_name}: handles never closed: {handles}. Close each handle in its "
                "opening region."
            )
        if self.live:
            items = "; ".join(f"{r.what} at {r.site}" for r in self.live)
            raise ProtocolError(
                f"{self.role_name}: previews no later wait consumes: {items}. Add a wait() "
                "after the peek in trace order and release its handle; peel the last iteration "
                "if the peek reads the next iteration's epoch."
            )
