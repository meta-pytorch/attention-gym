"""Trace-time protocol bookkeeping: handle lifetimes, the protocol log, per-site check opt-outs,
and errors."""

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


# Checks a role can opt out of at one site (``with unchecked(check, reason=...)``):
#   unclosed      handles opened inside the block may stay open at role exit
#   cross_region  closes inside the block may sit in a different loop/branch than their open
#   tma           TMA copies into barriers taken inside the block skip the atom and byte checks
#   alias         accesses annotated inside the block are left out of the alias diagnostic
#   token         wait/acquire(token=...) inside the block trust the readiness they are given: a
#                 Token of any epoch or region, or a raw Boolean (an mbarrier try_wait result)
CHECKS = ("unclosed", "cross_region", "tma", "alias", "token")


@dataclass
class OptOut:
    """One ``unchecked(...)`` block: which check, why, where, and what it let through."""

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
    """Make ``trace`` the current role trace while its body is traced."""
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
    """Turn off one trace-time check for the code inside the block. ``reason`` is required;
    ``describe()`` lists the opt-out with its source line and what it let through. Every other
    check stays on, here and elsewhere. Use it for an ordering argument the checker cannot see,
    for example a handle deliberately closed in the next loop iteration."""
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
    """The user function of ``frame``: CuTeDSL traces loop and branch bodies as functions
    nested in it (``fill.<locals>.while_region_0``)."""
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
    """The copies of one loop-carried DSL value (a channel end or work cursor), to reject a
    copy whose phases are out of date.

    CuTeDSL traces each loop or branch body once and carries a value through it by rebuilding
    it: the body gets a copy and the region's result is a new copy after the region. A copy
    is stale once a copy made after it has seen more advances: the copy a caller still holds
    after dropping the result of a helper that advanced the value in a loop or branch,
    including inside a loop body that does not carry the value itself. Branch bodies are
    traced in order and the else branch gets its copy after the then branch, so a copy
    nothing advanced since (a read-only helper's result, an unused ``views()``) supersedes
    nothing.

    Checked: uses that read or advance the phases, and the role's return. Not checked:
    barrier-pointer accessors, so a saved snapshot may still hand out barriers. A snapshot a
    region rebuilds gets a copy as new as the live one, so returning or using that copy is not
    caught.
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
        """Copy ``copy`` advanced, handed out its raw state, or was assigned new phases."""
        self._advances += 1
        self._copies[copy].seen = self._advances

    def check(
        self, copy: int, trace: RoleTrace, name: str, kind: str, returned_by: str | None = None
    ) -> None:
        """Raise if copy ``copy`` of ``name`` is stale. ``returned_by``: the role function
        returning it at role exit (else a use here)."""
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
    """One traced event for the alias diagnostic (``alias.py``): a handle open/close, an
    annotated or raw storage access, or a recorded ordering instruction.

    ``where``: the enclosing dynamic regions, outermost first, as ``(block key, kind)`` with kind
    ``loop`` (``scf.for``/``scf.while`` body), ``if`` or ``other``; ``open``: the handle records
    open at the event; ``opted_out``: inside ``unchecked("alias")``."""

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
    """Per role instance: the static sequence of channel ops recorded while tracing.

    Dynamic loops are traced once, so an op inside a loop appears once. ``depth`` is the MLIR
    block nesting depth relative to the role body (0 = role body, 1 = inside one loop or if).
    """

    def __init__(self):
        self.ops: dict[str, list[tuple]] = {}
        self.tmem: dict[str, set] = {}

    def record(self, role_name, chan, op, slot, depth, site):
        self.ops.setdefault(role_name, []).append((chan, op, slot, depth, site))

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


class RoleTrace:
    """Open-handle bookkeeping for one traced role instance."""

    def __init__(self, role_name: str, log: ProtocolLog | None, ranges=None, waits: bool = True):
        self.role_name = role_name
        self.log = log
        self.ranges = ranges  # the trace backend's ranges of this role (tracing.py), or None
        self.trace_waits = ranges is not None and waits  # and ranges around blocking waits
        # (channel, op, enclosing blocks innermost first, epochs, order) per recorded op:
        # epochs is a skip's count (None if dynamic), else 1; order is a close's handle order
        # (HandleRecord).
        self.ops: list = []
        self.item_block = None  # the work-item loop body (set by the WorkCursor's advance())
        self.open: list = []
        self.checks: list = []  # end-of-role trace checks (e.g. work cursors)
        self.tma_bars: list = []  # TMA-channel full-barrier pointers handed out (tma.py)
        self.events: list[Event] = []  # alias-diagnostic events, in trace order
        self.open_turns: set[str] = set()  # turnstiles whose turn this role holds here
        self._blocks: list = []  # (block, key, kind) of every region an event was recorded in
        self.root = current_block()

    def where(self) -> tuple:
        """The enclosing dynamic regions of the current insertion point, outermost first."""
        out = []
        for block in reversed(_ancestors(current_block(), self.root)[:-1]):
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
            opted_out = self.opt_out("alias", f"{data['storage'][1]} access at {site}")
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

    def depth(self) -> int:
        block = current_block()
        try:
            blocks = _ancestors(block, self.root)
            return len(blocks) - 1 if blocks and blocks[-1] == self.root else 0
        except (AttributeError, ValueError, RuntimeError):
            return 0

    def record(self, chan, op, slot, count: int | None = 1, order: int | None = None):
        # Blocks are captured while live: a region the tracer later discards must not be
        # walked afterwards (dereferencing an erased block crashes).
        self.ops.append((chan, op, _ancestors(current_block(), self.root), count, order))
        if self.log is not None:
            logged = op if count == 1 else f"{op} {'n' if count is None else count}"
            self.log.record(self.role_name, chan, logged, slot, self.depth(), user_site())

    def uses_per_item(self) -> dict:
        """``{channel: n}``: acquires/waits per work item, 0 when inside an inner loop."""
        uses: dict = {}
        if self.item_block is None:
            return uses
        for chan, op, blocks, *_ in self.ops:
            if op not in ("acquire", "wait") or self.item_block not in blocks:
                continue
            depth = blocks.index(self.item_block)
            if depth == 0 and uses.get(chan) != 0:
                uses[chan] = uses.get(chan, 0) + 1
            elif depth:  # nested inside the item loop: a dynamic count per item
                uses[chan] = 0
        return uses

    def epoch_sequence(self, chan: str) -> tuple[bool | None, list[tuple[str, int]]] | str:
        """``(per_item, [(op, n)])`` of this role's acquires, retires, waits and skips on
        ``chan`` when none sits in a dynamic loop or branch other than the work-item loop and
        the role leaves ``end.state`` to the layer: ``n``
        is a skip's epochs, a retire's acquire (its handle's order), else 1; ``per_item`` says
        whether they run once per work item (inside that loop) or once per launch (None without
        ops). Otherwise why the sequence is not static."""
        seq, regions = [], set()
        for c, op, blocks, count, order in self.ops:
            if c != chan or op not in ("acquire", "retire", "wait", "skip", "state"):
                continue
            if op == "state":
                return f"{self.role_name} reads or assigns end.state"
            inner = blocks[:-1]  # the enclosing blocks below the role body
            in_item = (
                self.item_block is not None and len(inner) == 1 and inner[0] == self.item_block
            )
            if inner and not in_item:
                return f"{self.role_name}'s {op} runs inside a dynamic loop or branch"
            if count is None:
                return f"{self.role_name}'s skip count is dynamic"
            regions.add(bool(inner))
            seq.append((op, order if op == "retire" else count))
        if len(regions) > 1:
            return f"{self.role_name} uses it both inside and outside the work-item loop"
        return (regions.pop() if regions else None), seq

    def opened(self, handle):
        if (opt := active_opt_out("unclosed")) is not None:
            handle.unclosed_ok = opt
        self.open.append(handle)

    def closed(self, handle, op: str):
        site = user_site()
        if handle.closed_at is not None:
            raise ProtocolError(
                f"{self.role_name}: {handle.what} closed twice: {handle.closed_op} at "
                f"{handle.closed_at}, then {op} at {site}"
            )
        cross = current_block() != handle.block
        if cross and not self.opt_out("cross_region", f"{handle.what} {op} at {site}"):
            raise ProtocolError(
                f"{self.role_name}: {handle.what} opened at {handle.site} but {op} at "
                f"{site} is in a different dynamic region (loop or branch), so it would run a "
                "different number of times than the acquire; close handles in the region "
                "that opened them"
            )
        handle.closed_at, handle.closed_op = site, op
        self.open.remove(handle)

    def finish(self):
        for check in self.checks:
            check()
        leaks = []
        for h in self.open:
            if h.unclosed_ok is None:
                leaks.append(h)
                continue
            h.unclosed_ok.role = h.unclosed_ok.role or self.role_name
            h.unclosed_ok.suppressed.append(f"{h.what} opened at {h.site} left open")
        if leaks:
            items = "; ".join(f"{h.what} opened at {h.site}" for h in leaks)
            raise ProtocolError(f"{self.role_name}: handles never closed: {items}")
