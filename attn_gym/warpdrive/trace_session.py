"""Capture and decode a warpdrive globaltimer trace and write its Perfetto layout.

Kernel modules and package ``__init__`` do not import this module: decoding must not enter
kernel compile keys.
"""

from __future__ import annotations

import os
import warnings
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import torch

from attn_gym.warpdrive.profiler.host import (
    ProfileBuf,
    ProfileSession,
    TagTable,
    decode_ranges,
    profile_session,
)
from attn_gym.warpdrive.profiler.perfetto import write_ranges
from attn_gym.warpdrive.profiler.ranges import Range, percentile
from attn_gym.warpdrive.protocol import ProtocolError

# Strong references keep baked device addresses valid until process exit.
_buffers: dict[tuple[int, int], torch.Tensor] = {}
_buffer_lock = Lock()


def owned_buffer(kernel) -> ProfileBuf:
    """Resolve the current device's process-lifetime buffer before compile-key hashing."""
    if getattr(kernel, "_ws_trace_origin_pid", os.getpid()) != os.getpid():
        raise ProtocolError(
            "globaltimer traced builds cannot use compile_many/precompile_many: "
            "compile in the launching process (the device address is process-local)"
        )
    device = torch.cuda.current_device()
    num_ctas = getattr(kernel, "_ws_trace_ctas", None)
    if num_ctas is None:
        num_ctas = torch.cuda.get_device_properties(device).multi_processor_count
    kernel._collect()
    capacity = len(kernel._ws_roles) * kernel.trace_events_per_role
    size = num_ctas * (1 + 4 * capacity)
    with _buffer_lock:
        key = (device, size)
        if key not in _buffers:
            _buffers[key] = torch.zeros(size, dtype=torch.int64, device=f"cuda:{device}")
        tensor = _buffers[key]
    return ProfileBuf(tensor, capacity, num_ctas, "CTA")


def trace_session(
    kernel, num_ctas: int | None = None, *, trace_path: str | Path | None = None, device=None
) -> AbstractContextManager[ProfileSession]:
    """Capture one launch of a compiled kernel with ``enable_trace("globaltimer")``.

    A layer-owned buffer needs no launch argument or ``num_ctas``. For an explicit buffer,
    pass ``session.prof.tensor`` to the launch and supply ``num_ctas`` to size the capture.
    After exit, ``session.ranges`` contains names, ``cta=(CTA,)``,
    ``warp`` (the role's first warp) and payloads (work index or user-region payload).
    ``trace_path`` optionally writes these ranges through ``write_trace``.

    Warns if any role fills its ``events_per_role`` capacity in a CTA; later ranges drop.
    """
    if kernel.trace_backend != "globaltimer":
        raise ProtocolError(
            f"trace_session needs enable_trace('globaltimer'), not {kernel.trace_backend!r}"
        )
    if not getattr(kernel, "_ws_traced", False):
        raise ProtocolError(
            "trace_session needs a compiled kernel: range names come from its trace"
        )
    return capture(kernel, num_ctas, trace_path, device)


@contextmanager
def capture(kernel, num_ctas: int | None, trace_path, device) -> Iterator[ProfileSession]:
    roles = kernel._ws_roles
    capacity = kernel.trace_events_per_role
    if kernel._ws_trace_owned:
        prof = owned_buffer(kernel)
        if kernel._ws_trace_address != (
            prof.tensor.device.index,
            prof.tensor.data_ptr(),
            prof.tensor.numel(),
        ):
            raise ProtocolError(
                "traced kernel buffer belongs to another device or size; recompile on this device"
            )
        if num_ctas is not None and num_ctas != prof.num_units:
            raise ValueError("num_ctas must match enable_trace(ctas=...) for a layer-owned buffer")
        if device is not None:
            requested = torch.device(device)
            if requested.type == "cuda" and requested.index is None:
                requested = torch.device("cuda", torch.cuda.current_device())
            if requested != prof.tensor.device:
                raise ValueError("device must match the compiled trace buffer device")
        prof.reset()
        session = ProfileSession(prof, TagTable(kernel._ws_trace_tags))
        yield session
        torch.cuda.synchronize(prof.tensor.device)
        session.ranges = decode_ranges(prof, session.tags)
    else:
        if num_ctas is None:
            raise ValueError("num_ctas is required for an explicit trace_buffer")
        with profile_session(
            max_events_per_unit=len(roles) * capacity,
            num_units=(num_ctas, "CTA"),
            tag_names=kernel._ws_trace_tags,
            device=device,
        ) as session:
            yield session
    lanes = {r.first_warp: r.name for r in roles}
    counts = Counter((item.cta, item.warp) for item in session.ranges)
    full = sorted({lanes[warp] for (_, warp), n in counts.items() if n >= capacity})
    if full:
        warnings.warn(
            f"warpdrive trace: role(s) {full} filled their {capacity} records in some CTA; "
            "later ranges were dropped (raise enable_trace(events_per_role=...))",
            stacklevel=3,
        )
    if trace_path is not None:
        write_trace(trace_path, kernel, session.ranges)


def write_trace(path: str | Path, kernel, ranges: list[Range], **options) -> Path:
    """Write a globaltimer trace with one process per CTA and one named track per role.

    ``options`` (``name``, ``args``, ``flows``) pass through to ``write_ranges``.
    """
    lanes = {r.first_warp: r.name for r in kernel._ws_roles}
    return write_ranges(
        path,
        ranges,
        process=lambda item: f"CTA {item.cta[0]}",
        thread=lambda item: lanes[item.warp],
        **options,
    )


WAIT_OPERATIONS = frozenset({"acquire", "wait", "peek", "tail", "turn", "sync"})


@dataclass(frozen=True)
class RoleSummary:
    """Durations summed across CTAs on one role's recording warp."""

    name: str
    warp: int
    items: int
    p50_ns: float
    p90_ns: float
    item_ns: int
    wait_ns: int

    @property
    def wait_share(self) -> float:
        return self.wait_ns / self.item_ns if self.item_ns else 0.0

    @property
    def work_share(self) -> float:
        return 1.0 - self.wait_share if self.item_ns else 0.0


@dataclass(frozen=True)
class RoleReport:
    """Structured role statistics with a human-readable nanosecond/share table."""

    roles: tuple[RoleSummary, ...]

    def __str__(self) -> str:
        lines = ["role                 warp  items    p50 ns    p90 ns     wait     work"]
        lines.extend(
            f"{r.name:<20} {r.warp:4d} {r.items:6d} {r.p50_ns:9.1f} {r.p90_ns:9.1f} "
            f"{r.wait_share:8.1%} {r.work_share:8.1%}"
            for r in self.roles
        )
        return "\n".join(lines)


def role_summary(ranges: list[Range], op=None) -> RoleReport:
    """Summarize role/work-item ranges and contained waits on the same CTA and warp.

    Without an op, outermost non-wait range names identify roles. Wait time sums the outermost
    wait ranges inside the role's items, so the wait share is at most one. Missing or dropped
    ranges are not reconstructed. A raw role's whole-body range counts as one item.
    """

    def is_wait(item: Range) -> bool:
        return item.name.rpartition(".")[2] in WAIT_OPERATIONS

    tracks: dict[tuple[tuple[int, ...], int], list[Range]] = defaultdict(list)
    for item in ranges:
        if item.warp is not None:
            tracks[item.cta, item.warp].append(item)
    for track in tracks.values():
        track.sort(key=lambda item: (item.start_ns, -item.end_ns))

    if op is not None:
        op._collect()
        identities = [(r.name, r.first_warp) for r in op._ws_roles]
    else:
        inferred = set()
        for (_, warp), track in tracks.items():
            outer_end = None
            for item in track:
                if outer_end is None or item.end_ns > outer_end:
                    if not is_wait(item):
                        inferred.add((item.name, warp))
                    outer_end = item.end_ns
        identities = sorted(inferred, key=lambda identity: (identity[1], identity[0]))
    summaries = []
    for name, warp in identities:
        durations = []
        waits = 0
        for (_, track_warp), track in tracks.items():
            if track_warp != warp:
                continue
            items = [r for r in track if r.name == name]
            durations.extend(r.duration_ns for r in items)
            starts = [r.start_ns for r in items]
            # Role work items are sequential on a track. The last item starting before a
            # wait is its only possible owner; unrelated CTAs never enter this search. A wait
            # nested in a counted wait (a channel wait inside a turn) adds no time.
            waited_until = None
            for item in track:
                if not is_wait(item) or (waited_until is not None and item.end_ns <= waited_until):
                    continue
                index = bisect_right(starts, item.start_ns) - 1
                if index >= 0 and item.end_ns <= items[index].end_ns:
                    waits += item.duration_ns
                    waited_until = item.end_ns
        summaries.append(
            RoleSummary(
                name,
                warp,
                len(durations),
                percentile(durations, 0.5) if durations else 0.0,
                percentile(durations, 0.9) if durations else 0.0,
                sum(durations),
                waits,
            )
        )
    return RoleReport(tuple(summaries))
