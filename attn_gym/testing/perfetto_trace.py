"""Read a native Perfetto trace back in tests: its slices, each with its process and track."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent


@dataclass(frozen=True)
class Slice:
    process: str
    track: str
    name: str
    start_ns: int
    end_ns: int
    args: dict
    flows_out: tuple[int, ...]
    flows_in: tuple[int, ...]


def annotation_value(annotation):
    return getattr(annotation, annotation.WhichOneof("value"))


def read_slices(path: str | Path) -> list[Slice]:
    """Every slice of the trace at ``path``, in begin order; raises on an end without a begin
    or a slice left open, so a passing read also checks the begin/end nesting."""
    trace = Trace()
    trace.ParseFromString(Path(path).read_bytes())
    names: dict[int, str] = {}
    parents: dict[int, int] = {}
    for packet in trace.packet:
        if packet.HasField("track_descriptor"):
            descriptor = packet.track_descriptor
            if descriptor.HasField("process"):
                names[descriptor.uuid] = descriptor.process.process_name
            else:
                names[descriptor.uuid] = descriptor.name
                parents[descriptor.uuid] = descriptor.parent_uuid
    open_slices: dict[int, list[tuple[int, TrackEvent]]] = {}
    slices = []
    for packet in trace.packet:
        event = packet.track_event
        if not packet.HasField("track_event"):
            continue
        stack = open_slices.setdefault(event.track_uuid, [])
        if event.type == TrackEvent.TYPE_SLICE_BEGIN:
            stack.append((packet.timestamp, event))
        elif event.type == TrackEvent.TYPE_SLICE_END:
            assert stack, f"slice end without a begin on {names[event.track_uuid]!r}"
            start, begin = stack.pop()
            slices.append(
                Slice(
                    names[parents[event.track_uuid]],
                    names[event.track_uuid],
                    begin.name,
                    start,
                    packet.timestamp,
                    {item.name: annotation_value(item) for item in begin.debug_annotations},
                    tuple(begin.flow_ids),
                    tuple(begin.terminating_flow_ids),
                )
            )
    assert not any(open_slices.values()), "a slice was left open"
    return sorted(slices, key=lambda item: (item.start_ns, -item.end_ns))
