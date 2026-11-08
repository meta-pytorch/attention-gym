"""Host side of the profiler: allocate profile buffers and decode their records into ranges.

``profile_session`` wraps both for a block of launches. The buffer layouts are described in
``ops.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch

from attn_gym.profiler.ops import PAYLOAD_FLAG
from attn_gym.profiler.perfetto import write_ranges
from attn_gym.profiler.ranges import Range

__all__ = [
    "ProfileBuf",
    "ProfileSession",
    "TagTable",
    "allocate_profile_buffer",
    "decode_ranges",
    "profile_session",
]


@dataclass
class ProfileBuf:
    """Profile buffer wrapper.

    Attributes:
        tensor: The underlying torch.int64 tensor; pass it to the kernel.
        max_events_per_unit: Record slots per profiling unit.
        num_units: Number of profiling units (e.g., CTAs).
        unit_name: Name for units in trace output (e.g., "CTA", "Block").
        compact: ``True`` for 1xi64 compact records (slot 0 of each unit holds a 64-bit
            anchor); ``False`` for 4xi64 full records (see ``ops.py``).
    """

    tensor: torch.Tensor
    max_events_per_unit: int
    num_units: int
    unit_name: str = "Unit"
    compact: bool = False

    @property
    def slice_size(self) -> int:
        """Size of each unit's buffer slice in int64 elements."""
        if self.compact:
            return 1 + self.max_events_per_unit
        return 1 + 4 * self.max_events_per_unit

    def reset(self) -> None:
        """Zero the buffer, e.g. between a warmup launch and the recorded one."""
        self.tensor.zero_()


class TagTable:
    """Mapping between tag names and the integer IDs kernels record.

    Example:
        tag_table = TagTable(["produce", "consume", "wait_lock"])
        produce_tag = tag_table.id("produce")
    """

    def __init__(self, tag_names: list[str]):
        self._names = list(tag_names)
        self._name_to_id = {name: i for i, name in enumerate(self._names)}

    def id(self, name: str) -> int:
        """The integer ID of ``name`` (KeyError if it is not in the table)."""
        return self._name_to_id[name]

    def name(self, tag_id: int) -> str:
        """The name of ``tag_id``, or ``unknown_<id>`` for an ID outside the table."""
        if 0 <= tag_id < len(self._names):
            return self._names[tag_id]
        return f"unknown_{tag_id}"

    @property
    def names(self) -> list[str]:
        """List of all tag names in order."""
        return list(self._names)

    def __len__(self) -> int:
        return len(self._names)


def allocate_profile_buffer(
    max_events_per_unit: int,
    num_units: int | tuple[int, str],
    device: torch.device | str | None = None,
    compact: bool = False,
) -> ProfileBuf:
    """Allocate a zeroed profile buffer.

    Each profiling unit (CTA, warp, ...) owns a slice of ``max_events_per_unit`` static record
    slots; no atomics are needed on the recording path.

    Args:
        max_events_per_unit: Record slots per unit.
        num_units: ``int`` (count, labeled ``"Unit"``) or ``(count, name)`` for trace labels.
        device: Device to allocate on. Defaults to the current CUDA device.
        compact: ``True`` selects compact records (slice size ``1 + max_events_per_unit``),
            ``False`` full records (slice size ``1 + 4 * max_events_per_unit``).
    """
    if isinstance(num_units, tuple):
        num_units_count, unit_name = num_units
    else:
        num_units_count, unit_name = num_units, "Unit"
    slice_size = 1 + (1 if compact else 4) * max_events_per_unit
    return ProfileBuf(
        tensor=torch.zeros(
            num_units_count * slice_size, dtype=torch.int64, device=device or "cuda"
        ),
        max_events_per_unit=max_events_per_unit,
        num_units=num_units_count,
        unit_name=unit_name,
        compact=compact,
    )


def decode_ranges(buf: ProfileBuf, tag_table: TagTable) -> list[Range]:
    """Decode every recorded slot of ``buf`` into a ``Range``, unit by unit, slot by slot.

    A range's ``cta`` is ``(unit,)`` and its ``slot`` the record's slot; full records also
    carry their warp and payload. Empty slots are skipped.
    """
    if buf.compact:
        return _decode_compact(buf, tag_table)
    return _decode_full(buf, tag_table)


def _decode_full(buf: ProfileBuf, tag_table: TagTable) -> list[Range]:
    grid = buf.tensor.cpu().view(buf.num_units, buf.slice_size)[:, 1:]
    records = grid.reshape(buf.num_units, buf.max_events_per_unit, 4)
    start_ns, dur_ns, tag_words, warps = records.unbind(dim=2)
    units, slots = torch.nonzero((start_ns != 0) | (dur_ns != 0), as_tuple=True)
    ranges = []
    for unit, slot, start, dur, word, warp in zip(
        units.tolist(),
        slots.tolist(),
        start_ns[units, slots].tolist(),
        dur_ns[units, slots].tolist(),
        tag_words[units, slots].tolist(),
        warps[units, slots].tolist(),
        strict=True,
    ):
        tag, payload = word, None
        if word & PAYLOAD_FLAG:
            tag, payload = word & (PAYLOAD_FLAG - 1), word >> 32
        ranges.append(Range(tag_table.name(tag), start, start + dur, (unit,), warp, slot, payload))
    return ranges


def _decode_compact(buf: ProfileBuf, tag_table: TagTable) -> list[Range]:
    """Decode compact records, rebasing each start on its unit's 64-bit anchor."""
    mask32 = (1 << 32) - 1
    mask24 = (1 << 24) - 1

    grid = buf.tensor.cpu().view(buf.num_units, buf.slice_size)
    anchors = grid[:, 0]
    records = grid[:, 1:]

    valid_anchor = anchors != 0
    unanchored = (~valid_anchor & (records != 0).any(dim=1)).nonzero(as_tuple=True)[0]
    if len(unanchored):
        counts = (records[unanchored] != 0).sum(dim=1).tolist()
        raise RuntimeError(
            f"Compact-mode units {unanchored.tolist()} hold {', '.join(map(str, counts))} "
            "record(s) but no anchor: warp 0 must run compact_anchor_init (or "
            "compact_prepare_smem) with the same record predicate as the events"
        )

    ts_lo = records & mask32
    dur_ns = (records >> 32) & mask24
    tags = (records >> 56) & 0xFF
    # A start within 2^31 ns of its unit's anchor, before it (another warp read its timer
    # first) or after it (across a 2^32 ns boundary), is the anchor plus the signed
    # 32-bit difference of their low halves.
    delta = (ts_lo - (anchors & mask32).unsqueeze(1)) & mask32
    delta = delta - ((delta >> 31) << 32)
    start_ns = anchors.unsqueeze(1) + delta

    units, slots = torch.nonzero(records != 0, as_tuple=True)
    return [
        Range(tag_table.name(tag), start, start + dur, (unit,), slot=slot)
        for unit, slot, start, dur, tag in zip(
            units.tolist(),
            slots.tolist(),
            start_ns[units, slots].tolist(),
            dur_ns[units, slots].tolist(),
            tags[units, slots].tolist(),
            strict=True,
        )
    ]


def check_counts(ranges: list[Range], num_units: int, expected_per_unit: int) -> None:
    """Raise when any profiling unit decoded a count other than ``expected_per_unit``."""
    counts = [0] * num_units
    for item in ranges:
        counts[item.cta[0]] += 1
    mismatches = [
        f"unit {unit}: {count}" for unit, count in enumerate(counts) if count != expected_per_unit
    ]
    if mismatches:
        raise RuntimeError(
            f"Expected {expected_per_unit} events per profiling unit; got {', '.join(mismatches)}"
        )


@dataclass
class ProfileSession:
    """A profiling context's buffer and tags; ``ranges`` is set when it exits."""

    prof: ProfileBuf
    tags: TagTable
    ranges: list[Range] = field(default_factory=list)


@contextmanager
def profile_session(
    max_events_per_unit: int,
    num_units: int | tuple[int, str],
    tag_names: list[str],
    trace_path: str | Path | None = None,
    device: torch.device | str | None = None,
    compact: bool = False,
    expected_events_per_unit: int | None = None,
) -> Iterator[ProfileSession]:
    """Allocate a profile buffer for the block's launches; decode it when the block exits.

    Example:
        with profile_session(64, (num_ctas, "CTA"), ["produce"], compact=True) as session:
            kernel(x, session.prof.tensor, session.prof.max_events_per_unit,
                   session.tags.id("produce"))
        summarize_by_tag(session.ranges)

    Args:
        max_events_per_unit: Record slots per unit.
        num_units: ``int`` or ``(count, name)``; the name labels the trace's processes.
        tag_names: The session's tag names; kernels record their ``TagTable`` IDs.
        trace_path: Optional Perfetto trace path, written with one process per unit and one
            track per warp (or per tag for compact records). For another layout, call
            ``perfetto.write_ranges`` on ``session.ranges``.
        device: Device to allocate on.
        compact: Compact (1xi64) rather than full (4xi64) records.
        expected_events_per_unit: Optional exact decoded count required of every unit.
            With sparse sampling, allocate buffer units only for sampled units.

    Yields:
        ProfileSession, whose ``ranges`` are set after the block exits.
    """
    if expected_events_per_unit is not None and expected_events_per_unit < 0:
        raise ValueError("expected_events_per_unit must be non-negative")

    prof = allocate_profile_buffer(max_events_per_unit, num_units, device, compact)
    session = ProfileSession(prof=prof, tags=TagTable(tag_names))
    yield session

    torch.cuda.synchronize()
    session.ranges = decode_ranges(prof, session.tags)
    if expected_events_per_unit is not None:
        check_counts(session.ranges, prof.num_units, expected_events_per_unit)
    if trace_path is not None:
        write_ranges(trace_path, session.ranges, process=lambda r: f"{prof.unit_name} {r.cta[0]}")
