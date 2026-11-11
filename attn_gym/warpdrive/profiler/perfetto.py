"""The profiler's Perfetto writer: native TrackEvent traces of ranges and pipeline captures.

``TraceBuilder`` holds processes, named tracks, slices, flows, instants and counters, all in
integer nanoseconds. Slices that cross on one track rather than nest move to sibling
"overlap" lanes, which Perfetto merges back under the track's name. ``write_ranges`` lays out
decoded ``Range``s and ``pipeline.perfetto.write_pipeline_perfetto`` a joined capture, both
with this builder. The ``perfetto`` package is imported only when a trace is built.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from attn_gym.warpdrive.profiler.ranges import Range

__all__ = ["TraceBuilder", "stable_id", "trace_file", "write_ranges"]

SEQUENCE_ID = 2001


def stable_id(*parts: object) -> int:
    """Return a deterministic nonzero uint64 identifier."""
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") or 1


def trace_file(path: str | Path) -> Path:
    """Return the native trace path for ``path``.

    ``.pftrace`` replaces a ``.json``/``.json.gz`` suffix and is appended to any other suffix
    except ``.pftrace``/``.perfetto-trace``.
    """
    path = Path(path)
    if path.suffix in {".pftrace", ".perfetto-trace"}:
        return path
    if path.suffixes[-2:] == [".json", ".gz"]:
        return path.with_suffix("").with_suffix(".pftrace")
    if path.suffix == ".json":
        return path.with_suffix(".pftrace")
    return Path(f"{path}.pftrace")


@dataclass
class _Slice:
    start_ns: int
    end_ns: int
    name: str
    args: Mapping[str, object]
    flows_out: tuple[int, ...]
    flows_in: tuple[int, ...]


@dataclass
class _Track:
    parent: int
    name: str
    rank: int
    counter: bool = False
    slices: list[_Slice] = field(default_factory=list)
    instants: list[tuple[int, str, Mapping[str, object]]] = field(default_factory=list)
    samples: list[tuple[int, int | float]] = field(default_factory=list)


def nesting_lanes(slices: list[_Slice]) -> list[list[_Slice]]:
    """Split one track's slices into lanes on which every two slices nest or are disjoint.

    Slices are placed by start, longer first. A lane takes a slice that starts after its open
    slices end or fits inside its innermost open one.
    """
    lanes: list[list[_Slice]] = []
    open_ends: list[list[int]] = []
    for item in sorted(slices, key=lambda item: (item.start_ns, -item.end_ns)):
        for lane, ends in zip(lanes, open_ends, strict=True):
            while ends and ends[-1] <= item.start_ns:
                ends.pop()
            if not ends or item.end_ns <= ends[-1]:
                lane.append(item)
                ends.append(item.end_ns)
                break
        else:
            lanes.append([item])
            open_ends.append([item.end_ns])
    return lanes


def lane_markers(lane: list[_Slice]) -> list[tuple[int, _Slice | None]]:
    """Begin (slice) and end (None) markers of one lane in emission order.

    Timestamps never decrease. A slice ends before one starting at its end begins, and an empty
    slice ends right after its own begin.
    """
    markers: list[tuple[int, _Slice | None]] = []
    open_slices: list[_Slice] = []
    for item in lane:
        while open_slices and open_slices[-1].end_ns <= item.start_ns:
            markers.append((open_slices.pop().end_ns, None))
        markers.append((item.start_ns, item))
        open_slices.append(item)
    markers.extend((item.end_ns, None) for item in reversed(open_slices))
    return markers


def add_annotation(event, name: str, value: object) -> None:
    """Attach one scalar or string debug annotation to a TrackEvent."""
    annotation = event.debug_annotations.add()
    annotation.name = name
    if isinstance(value, bool):
        annotation.bool_value = value
    elif isinstance(value, int):
        annotation.int_value = value
    elif isinstance(value, float):
        annotation.double_value = value
    else:
        annotation.string_value = str(value)


class TraceBuilder:
    """Collect a trace's tracks and events; ``write`` serializes them as TrackEvent packets."""

    def __init__(self) -> None:
        self.processes: dict[int, tuple[str, int]] = {}
        self.tracks: dict[int, _Track] = {}

    def process(self, key: object, name: str, *, rank: int | None = None) -> int:
        """Add (or return) the process for ``key``; processes sort by ``rank``, then order."""
        uuid = stable_id("process", key)
        self.processes.setdefault(uuid, (name, len(self.processes) if rank is None else rank))
        return uuid

    def track(
        self,
        process: int,
        key: object,
        name: str,
        *,
        rank: int | None = None,
        counter: bool = False,
    ) -> int:
        """Add (or return) a track of ``process``; tracks sort by ``rank``, then order."""
        uuid = stable_id("track", process, key)
        if uuid not in self.tracks:
            siblings = sum(track.parent == process for track in self.tracks.values())
            self.tracks[uuid] = _Track(process, name, siblings if rank is None else rank, counter)
        return uuid

    def slice(
        self,
        track: int,
        start_ns: int,
        end_ns: int,
        name: str,
        *,
        args: Mapping[str, object] | None = None,
        flows_out: Iterable[int] = (),
        flows_in: Iterable[int] = (),
    ) -> None:
        """Add one slice; ``flows_out``/``flows_in`` are the ids of flows leaving/ending at it."""
        if end_ns < start_ns:
            raise ValueError(f"slice {name!r} ends at {end_ns} before its start {start_ns}")
        self.tracks[track].slices.append(
            _Slice(start_ns, end_ns, name, args or {}, tuple(flows_out), tuple(flows_in))
        )

    def instant(
        self, track: int, timestamp_ns: int, name: str, args: Mapping[str, object]
    ) -> None:
        self.tracks[track].instants.append((timestamp_ns, name, args))

    def counter(self, track: int, timestamp_ns: int, value: float) -> None:
        self.tracks[track].samples.append((timestamp_ns, value))

    def serialize(self) -> bytes:
        """Build the trace: descriptors first, then events in timestamp order."""
        try:
            from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import (
                TrackDescriptor,
                TrackEvent,
            )
            from perfetto.trace_builder.proto_builder import TraceProtoBuilder
        except ImportError as error:
            raise ImportError("writing Perfetto traces needs the 'perfetto' package") from error

        builder = TraceProtoBuilder()
        for uuid, (name, rank) in self.processes.items():
            descriptor = builder.add_packet().track_descriptor
            descriptor.uuid = uuid
            descriptor.process.pid = uuid % (2**31)
            descriptor.process.process_name = name
            descriptor.child_ordering = TrackDescriptor.EXPLICIT
            descriptor.sibling_order_rank = rank

        events: list[tuple[int, int, object]] = []  # (timestamp, lane track uuid, payload)
        for track_uuid, track in self.tracks.items():
            lanes = nesting_lanes(track.slices) or [[]]
            for lane_index, lane in enumerate(lanes):
                uuid = track_uuid if lane_index == 0 else stable_id(track_uuid, lane_index)
                descriptor = builder.add_packet().track_descriptor
                descriptor.uuid = uuid
                descriptor.parent_uuid = track.parent
                descriptor.name = track.name if lane_index == 0 else f"{track.name} overlap"
                descriptor.sibling_order_rank = track.rank * 100 + lane_index
                if track.counter:
                    descriptor.counter.SetInParent()
                if len(lanes) > 1:
                    descriptor.sibling_merge_behavior = (
                        TrackDescriptor.SIBLING_MERGE_BEHAVIOR_BY_SIBLING_MERGE_KEY
                    )
                    descriptor.sibling_merge_key = str(track_uuid)
                events.extend((ts, uuid, marker) for ts, marker in lane_markers(lane))
            events.extend(
                (ts, track_uuid, ("instant", name, args)) for ts, name, args in track.instants
            )
            events.extend((ts, track_uuid, ("counter", value)) for ts, value in track.samples)

        for timestamp_ns, uuid, payload in sorted(events, key=lambda event: event[0]):
            packet = builder.add_packet()
            packet.timestamp = timestamp_ns
            packet.trusted_packet_sequence_id = SEQUENCE_ID
            event = packet.track_event
            event.track_uuid = uuid
            if payload is None:
                event.type = TrackEvent.TYPE_SLICE_END
            elif isinstance(payload, _Slice):
                event.type = TrackEvent.TYPE_SLICE_BEGIN
                event.name = payload.name
                event.flow_ids.extend(payload.flows_out)
                event.terminating_flow_ids.extend(payload.flows_in)
                for name, value in payload.args.items():
                    add_annotation(event, name, value)
            elif payload[0] == "instant":
                event.type = TrackEvent.TYPE_INSTANT
                event.name = payload[1]
                for name, value in payload[2].items():
                    add_annotation(event, name, value)
            else:
                event.type = TrackEvent.TYPE_COUNTER
                if isinstance(payload[1], int):
                    event.counter_value = payload[1]
                else:
                    event.double_counter_value = payload[1]
        return builder.serialize()

    def write(self, path: str | Path) -> Path:
        """Write the trace to ``trace_file(path)`` and return that path."""
        output = trace_file(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(self.serialize())
        return output


def cta_label(item: Range) -> str:
    return "CTA " + ",".join(map(str, item.cta))


def warp_or_name(item: Range) -> str:
    return item.name if item.warp is None else f"warp {item.warp}"


def write_ranges(
    path: str | Path,
    ranges: Iterable[Range],
    *,
    process: Callable[[Range], str] = cta_label,
    thread: Callable[[Range], str] = warp_or_name,
    name: Callable[[Range], str] | None = None,
    args: Callable[[Range], Mapping[str, object]] | None = None,
    flows: Iterable[tuple[Range, Range]] = (),
) -> Path:
    """Write ``ranges`` as a Perfetto trace, in ns from the earliest start; return its path.

    ``process`` and ``thread`` label the process and track of each range (default: one
    process per CTA, one track per warp, or per range name for records without a warp).
    ``name`` names its slice (default ``Range.name``) and ``args`` adds slice arguments to the
    range's own (CTA, warp, slot, payload, start and duration). ``flows`` are
    ``(source, target)`` pairs drawn as arrows.
    """
    ranges = list(ranges)
    origin = min((item.start_ns for item in ranges), default=0)
    flow_out: dict[Range, list[int]] = defaultdict(list)
    flow_in: dict[Range, list[int]] = defaultdict(list)
    for index, (source, target) in enumerate(flows):
        flow_id = stable_id("flow", index)
        flow_out[source].append(flow_id)
        flow_in[target].append(flow_id)

    builder = TraceBuilder()
    # Processes in CTA order, tracks by warp (records without one last) then label.
    process_rank: dict[str, tuple] = {}
    track_rank: dict[tuple[str, str], tuple] = {}
    for item in ranges:
        process_label, thread_label = process(item), thread(item)
        process_rank[process_label] = min(process_rank.get(process_label, item.cta), item.cta)
        warp = (item.warp is None, item.warp or 0, thread_label)
        key = (process_label, thread_label)
        track_rank[key] = min(track_rank.get(key, warp), warp)
    process_ids = {
        label: builder.process(label, label, rank=rank)
        for rank, label in enumerate(sorted(process_rank, key=process_rank.__getitem__))
    }
    track_ids = {
        key: builder.track(process_ids[key[0]], key[1], key[1], rank=rank)
        for rank, key in enumerate(sorted(track_rank, key=lambda key: track_rank[key]))
    }
    for item in ranges:
        item_args: dict[str, object] = {
            "cta": ",".join(map(str, item.cta)),
            "start_ns": item.start_ns,
            "duration_ns": item.duration_ns,
        }
        for field_name in ("warp", "slot", "payload"):
            if getattr(item, field_name) is not None:
                item_args[field_name] = getattr(item, field_name)
        if args is not None:
            item_args.update(args(item))
        builder.slice(
            track_ids[(process(item), thread(item))],
            item.start_ns - origin,
            item.end_ns - origin,
            item.name if name is None else name(item),
            args=item_args,
            flows_out=flow_out.get(item, ()),
            flows_in=flow_in.get(item, ()),
        )
    return builder.write(path)
