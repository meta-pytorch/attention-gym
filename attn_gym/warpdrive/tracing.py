"""Device-side role and wait ranges for IKET and globaltimer.

Only profiler device ops are imported: imported modules enter kernel compile keys.
See ``docs/warpdrive/tracing.md`` (Tracing backends) for setup, capture and decoding.

Range names are stable because profiler pipeline joins use them:

* ``<role>`` spans each work item, from cursor start to ``advance()``, or the whole role
  without a ``WorkStream``. Instance suffixes such as ``softmax[0]`` are kept. CTA identity
  is recorded separately, including for pairs. Globaltimer records the work index as payload;
  IKET work-item ranges have no payload.
* ``<channel>.acquire``, ``.wait``, ``.peek`` and ``.tail`` span blocking mbarrier waits
  inside the role range. ``<turnstile>.turn`` and ``<named barrier>.sync`` do the same.
* ``with self.trace_region("name", payload=...):`` adds a user range inside a role.

``pipeline_plan.plan_from_warpdrive(kernel, cta=...)`` exports a per-CTA plan with matching
role and region names, channel resources, data edges and reuse edges. Its iteration is a
work item. Both backends feed ``pipeline.join``, ``analyze_pipeline`` and
``write_pipeline_perfetto`` without manual annotations; wait ranges become measured waits.
"""

from __future__ import annotations

from contextlib import contextmanager

import cutlass
from cutlass import Int32, Int64, cute
from cutlass.cute.experimental import iket

from attn_gym.warpdrive.profiler.ops import event_stop, read_globaltimer
from attn_gym.warpdrive.protocol import ProtocolError

BACKENDS = ("iket", "globaltimer")
MAX_DEPTH = 8  # nested globaltimer ranges open at once in one role


class Ranges:
    """A role's trace-time range stack; backends record payloads at push or pop.

    Both backends require LIFO closes. ``open_range``/``close_range`` reject mismatches,
    such as advancing a work cursor inside ``trace_region``. Device state is defined at
    role start and never rebound, so this object need not carry values through dynamic
    control flow, even when a push or pop is treated as a write (as with ``RoleContext``).
    """

    def __init__(self):
        self.names: list[str] = []  # the open ranges, innermost last
        # A shared-body role (``roles.md``): its runtime instance index and, per range name the
        # body records, the name each instance records instead (``{"cg0[*]": ["cg0[0]", ...]}``).
        self.instance = None
        self.aliases: dict[str, list[str]] = {}

    def share(self, instance, aliases: dict[str, list[str]]) -> None:
        """Record ``aliases[name][instance]`` in place of each aliased range ``name``."""
        self.instance, self.aliases = instance, aliases

    def open_range(self, name: str) -> int:
        """Record a push; return its depth."""
        self.names.append(name)
        return len(self.names) - 1

    def close_range(self, name: str) -> int:
        """Record the pop of ``name``; return its depth."""
        if not self.names or self.names[-1] != name:
            innermost = repr(self.names[-1]) if self.names else "no range"
            raise ProtocolError(
                f"trace range {name!r} closed while {innermost} is the innermost open one: "
                "ranges must nest (a work cursor advanced inside a trace_region?)"
            )
        self.names.pop()
        return len(self.names)

    def __extract_mlir_values__(self):
        return []

    def __new_from_mlir_values__(self, values):
        return self


class IketRanges(Ranges):
    """One role's IKET ranges."""

    def push(self, name: str, payload=None) -> None:
        self.open_range(name)
        names = self.aliases.get(name)
        if names is None:
            _iket_push(name, payload)
            return
        for j in range(len(names)):  # IKET names are static strings
            _iket_push_if(self.instance == j, names[j], payload)

    def pop(self, name: str, payload=None) -> None:
        """IKET records a range's payload at its push (work items push none)."""
        self.close_range(name)
        iket.range_pop()


def _iket_push(name: str, payload) -> None:
    if payload is None:
        iket.range_push(name)
    else:
        iket.range_push(name, payload)


@cute.jit
def _iket_push_if(cond, name: cutlass.Constexpr, payload):
    if cond:
        _iket_push(name, payload)


class IketTrace:
    """Stateless IKET backend."""

    def role(self, ordinal: int | Int32, first_warp: int | Int32) -> IketRanges:
        return IketRanges()


class GlobaltimerTrace:
    """A CTA's profile buffer, SMEM start times and trace-ordered range tags.

    ``run_roles`` constructs this before dispatching roles.
    """

    def __init__(self, buffer, starts, roles: int, events_per_role: int):
        # Materialize before dispatch so the iterator dominates every role's uses.
        self.buffer = cute.make_tensor(buffer.iterator, buffer.layout)
        self.starts = starts  # Int64 SMEM (roles, MAX_DEPTH): see GlobaltimerRanges
        self.events_per_role = events_per_role
        self.max_events_per_unit = roles * events_per_role
        bx, by, bz = cute.arch.block_idx()
        gx, gy, _ = cute.arch.grid_dim()
        unit = Int32(bx + gx * (by + gy * bz))
        # A buffer with fewer units than the grid has CTAs records the first ones only.
        slice_size = 1 + 4 * self.max_events_per_unit
        recorded = (Int64(unit) + 1) * slice_size <= Int64(cute.size(self.buffer))
        self.unit = cutlass.select_(recorded, unit, Int32(-1))  # -1: not recorded
        self.tags: list[str] = []

    def tag(self, name: str) -> int:
        if name not in self.tags:
            self.tags.append(name)
        return self.tags.index(name)

    def role(self, ordinal: int | Int32, first_warp: int | Int32) -> GlobaltimerRanges:
        return GlobaltimerRanges(self, ordinal, first_warp)


class GlobaltimerRanges(Ranges):
    """A role's globaltimer ranges, recorded by lane 0 of its first warp.

    Push/pop use IKET's LIFO model. Start times use static-depth SMEM slots rather than
    registers held across the range. Nothing is carried through loops, including a work
    range that opens in one iteration and closes in the next. Payloads record at pop.
    Closed ranges take successive slots in the role's share of the CTA buffer; excess
    ranges are dropped.
    """

    def __init__(self, trace: GlobaltimerTrace, ordinal: int | Int32, first_warp: int | Int32):
        """``ordinal``, ``first_warp``: the role instance's; runtime values in a shared body."""
        super().__init__()
        self.trace = trace
        self.first_warp = first_warp  # the record's tid, like IKET's warp of a range
        self.base = ordinal * trace.events_per_role
        self.starts = trace.starts[ordinal, None]
        self.count = cute.make_rmem_tensor((1,), Int32)  # records so far
        self.count[0] = Int32(0)

    def recorder(self):
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        return (warp == self.first_warp) & (cute.arch.lane_idx() == 0)

    def push(self, name: str, payload=None) -> None:
        depth = self.open_range(name)
        if depth == MAX_DEPTH:
            raise ProtocolError(f"more than {MAX_DEPTH} nested trace ranges open at {name!r}")
        start_range(self.starts, depth, self.recorder())

    def pop(self, name: str, payload=None) -> None:
        depth = self.close_range(name)
        trace = self.trace
        slot = self.count[0]
        self.count[0] = slot + 1
        end_range(
            trace.buffer,
            trace.unit,
            slot,
            self.starts,
            depth,
            self._tag(name),
            self.first_warp,
            self.base,
            trace.events_per_role,
            trace.max_events_per_unit,
            payload,
            self.recorder(),
        )

    def _tag(self, name: str):
        names = self.aliases.get(name)
        if names is None:
            return self.trace.tag(name)
        tag = Int32(self.trace.tag(names[0]))
        for j in range(1, len(names)):
            tag = cutlass.select_(self.instance == j, Int32(self.trace.tag(names[j])), tag)
        return tag


@cute.jit
def start_range(starts, depth: cutlass.Constexpr, recorder):
    if recorder:
        starts[depth] = read_globaltimer()


@cute.jit
def end_range(
    buffer,
    unit,
    slot,
    starts,
    depth: cutlass.Constexpr,
    tag,
    tid,
    base,
    capacity: cutlass.Constexpr,
    max_events: cutlass.Constexpr,
    payload,
    recorder,
):
    if recorder & (unit >= 0) & (slot < capacity):
        event_stop(
            buffer,
            unit,
            slot + base,
            starts[depth],
            Int32(tag),
            Int32(tid),
            Int32(max_events),
            payload,
        )


@cute.jit
def push_if(ranges, valid, name: cutlass.Constexpr):
    """Open the next work item's range only when the cursor holds one."""
    if valid:
        ranges.push(name)


@cute.jit
def pop_if(ranges, still_open, name: cutlass.Constexpr, payload):
    """Close a work item's range that a loop exit other than ``valid`` left open."""
    if still_open:
        ranges.pop(name, payload)


@contextmanager
def wait_range(trace, name: str):
    """Trace a blocking wait, or do nothing when tracing or wait ranges are disabled."""
    if trace is None or not trace.trace_waits:
        yield
        return
    trace.ranges.push(name)
    yield
    trace.ranges.pop(name)
