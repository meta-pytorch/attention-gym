"""Device-side role and channel ranges for IKET and globaltimer.

See ``docs/warpdrive/tracing.md`` ("Tracing backends") for backend setup, capture and decoding.
This module imports only the profiler's device ops because kernel compile keys hash
imported modules.

Range names are stable (the profiler's pipeline join relies on the role and wait names):

* ``<role>``: one range per work item of a role that loops over a ``WorkStream`` (from the
  cursor's start to its ``advance()``), else one range over the whole role. Role instances keep
  their suffix (``softmax[0]``). The CTA of a pair kernel is not in the name: both backends record
  the CTA of every range, and ``plan_from_warpdrive(kernel, cta=...)`` builds the plan of one CTA.
  Globaltimer work-item ranges carry their linear index as the payload; IKET work-item
  ranges carry no payload.
* ``<channel>.acquire`` / ``<channel>.wait`` / ``<channel>.peek`` / ``<channel>.tail``: the
  blocking mbarrier wait of a producer acquire, a consumer wait or preview, and a producer
  tail, nested inside the role range, so waiting is separated from working. ``<turnstile>.turn``
  and ``<named barrier>.sync`` likewise.
* user regions: ``with self.trace_region("name", payload=...):`` inside a role.

``pipeline_plan.plan_from_warpdrive`` exports the layer's structure in the profiler's neutral model
(``attn_gym.profiler.pipeline``): one ``Role`` and one ``Region`` per role
instance, both named like the role range, one ``Resource`` per channel, and per channel a data
dependency producer -> consumer plus a reuse dependency back. The plan's iteration is the work
item, so ``pipeline.join`` of either backend's ranges, ``analyze_pipeline``, and
``write_pipeline_perfetto`` analyze a capture of the kernel with no hand annotations; the wait
ranges become the regions' measured waits.
"""

from __future__ import annotations

from contextlib import contextmanager

import cutlass
from cutlass import Int32, Int64, cute
from cutlass.cute.experimental import iket

from attn_gym.profiler.ops import event_stop, read_globaltimer
from attn_gym.warpdrive.protocol import ProtocolError

BACKENDS = ("iket", "globaltimer")
MAX_DEPTH = 8  # nested globaltimer ranges open at once in one role


class Ranges:
    """A role's ranges: ``push(name, payload)`` and ``pop(name, payload)`` (each backend
    records the payload at one of them).

    Both backends pair a pop with the latest push, so the traced pops must close the traced
    pushes in reverse order: ``open_range``/``close_range`` keep that stack at trace time and
    reject a pop that would close another range (a work cursor advanced inside a
    ``trace_region``, say). The object is opaque to CuTeDSL control flow: a push or pop inside
    a dynamic ``if`` reads as a write of it, but its device values are defined at role start and
    never rebound, so carrying them is unnecessary (as ``kernel.RoleContext``)."""

    def __init__(self):
        self.names: list[str] = []  # the open ranges, innermost last

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
    """One role's IKET push/pop ranges."""

    def push(self, name: str, payload=None) -> None:
        self.open_range(name)
        if payload is None:
            iket.range_push(name)
        else:
            iket.range_push(name, payload)

    def pop(self, name: str, payload=None) -> None:
        """IKET records a range's payload at its push (work items push none)."""
        self.close_range(name)
        iket.range_pop()


class IketTrace:
    """The IKET backend's kernel-wide state: none."""

    def role(self, ordinal: int, first_warp: int) -> IketRanges:
        return IketRanges()


class GlobaltimerTrace:
    """A CTA's globaltimer trace, set up in ``run_roles`` before the roles dispatch: the profile
    buffer, this CTA's profiling unit, each role's open range starts in SMEM, and the kernel's
    tag table (range names in trace order)."""

    def __init__(self, buffer, starts, roles: int, events_per_role: int):
        # Materialized here: a lazily created iterator first used inside one role's branch
        # would not dominate the other roles' uses.
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

    def role(self, ordinal: int, first_warp: int) -> GlobaltimerRanges:
        return GlobaltimerRanges(self, ordinal, first_warp)


class GlobaltimerRanges(Ranges):
    """One role instance's globaltimer ranges, recorded by lane 0 of its first warp.

    Push/pop follow IKET's stack model. The traced depth of a push is static, so each open
    range's start time sits at that depth in the role's SMEM state: a range holds no register
    while the code it spans runs, which matters in kernels at the register limit, and nothing
    is carried through loops (a work item's range opens in one iteration and closes in the
    next). The payload is recorded at the pop, where the caller still holds it. A range takes
    its static slot when it closes: the role's next one, in its share of the unit's slots;
    ranges past the share are dropped."""

    def __init__(self, trace: GlobaltimerTrace, ordinal: int, first_warp: int):
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
            trace.tag(name),
            self.first_warp,
            self.base,
            trace.events_per_role,
            trace.max_events_per_unit,
            payload,
            self.recorder(),
        )


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
    tag: cutlass.Constexpr,
    tid: cutlass.Constexpr,
    base: cutlass.Constexpr,
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
    """A range around one blocking wait (nothing when tracing or wait ranges are off)."""
    if trace is None or not trace.trace_waits:
        yield
        return
    trace.ranges.push(name)
    yield
    trace.ranges.pop(name)
