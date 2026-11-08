"""The profiler's measurement model, ``Range``, and duration/overlap statistics over ranges.

Both recorders produce ranges: ``host.decode_ranges`` from a ``%globaltimer`` profile buffer
and ``iket.parse_iket`` from an IKET capture. ``pipeline.measure.join`` joins either to a
pipeline plan; ``perfetto.write_ranges`` draws them.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

__all__ = [
    "OverlapSummary",
    "Range",
    "RangeSummary",
    "overlap_between",
    "summarize_by_tag",
]


@dataclass(frozen=True)
class Range:
    """One measured interval of one named region.

    ``cta`` is where it ran: IKET's ``(x, y, z)`` CTA id, or ``(unit,)`` for a profile buffer
    record, whose unit is the CTA or whatever else the kernel chose. ``warp`` is the recording
    warp (IKET's warp id, a full record's warp), or None for compact records. ``slot`` is a
    profile record's static slot in its unit; ``payload`` is an optional runtime int32 recorded
    with the range, such as a work item's index.
    """

    name: str
    start_ns: int
    end_ns: int
    cta: tuple[int, ...]
    warp: int | None = None
    slot: int | None = None
    payload: int | None = None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns


@dataclass(frozen=True)
class RangeSummary:
    """Aggregate durations for one range name."""

    count: int
    total_ns: int
    mean_ns: float
    min_ns: int
    p50_ns: float
    p95_ns: float
    max_ns: int


@dataclass(frozen=True)
class OverlapSummary:
    """Intersection of two named range streams."""

    overlap_ns: int
    left_total_ns: int
    right_total_ns: int
    left_fraction: float
    right_fraction: float


def percentile(values: list[int], quantile: float) -> float:
    """Return the linearly interpolated ``quantile`` of non-empty integer values."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between 0 and 1")
    if not values:
        raise ValueError("percentile requires at least one value")

    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_by_tag(ranges: list[Range]) -> dict[str, RangeSummary]:
    """Summarize range durations by name."""
    durations_by_name: dict[str, list[int]] = defaultdict(list)
    for item in ranges:
        durations_by_name[item.name].append(item.duration_ns)

    return {
        name: RangeSummary(
            count=len(durations),
            total_ns=sum(durations),
            mean_ns=sum(durations) / len(durations),
            min_ns=min(durations),
            p50_ns=percentile(durations, 0.50),
            p95_ns=percentile(durations, 0.95),
            max_ns=max(durations),
        )
        for name, durations in durations_by_name.items()
    }


def merge_intervals(ranges: list[Range]) -> list[tuple[int, int]]:
    """Merge overlapping range intervals into a sorted disjoint list."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted((item.start_ns, item.end_ns) for item in ranges):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return merged


def interval_total(intervals: list[tuple[int, int]]) -> int:
    """Return the total duration of disjoint intervals."""
    return sum(end - start for start, end in intervals)


def interval_overlap(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> int:
    """Return the intersection duration of two sorted disjoint interval lists."""
    left_idx = 0
    right_idx = 0
    overlap_ns = 0
    while left_idx < len(left) and right_idx < len(right):
        left_start, left_end = left[left_idx]
        right_start, right_end = right[right_idx]
        overlap_ns += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_idx += 1
        else:
            right_idx += 1
    return overlap_ns


def overlap_between(
    ranges: list[Range],
    left: str,
    right: str,
    *,
    same_cta: bool = True,
) -> OverlapSummary:
    """Measure time where ranges named ``left`` and ``right`` are simultaneously active.

    With ``same_cta`` (the default), overlap is measured within each CTA, so ranges on
    different CTAs never count as overlapping. Durations sum the per-CTA interval unions and
    fractions divide those sums, which hides variance across CTAs; inspect CTAs separately
    when it matters.
    """
    groups: dict[tuple[int, ...] | None, list[Range]] = defaultdict(list)
    for item in ranges:
        groups[item.cta if same_cta else None].append(item)

    overlap_ns = 0
    left_total_ns = 0
    right_total_ns = 0
    for group in groups.values():
        left_intervals = merge_intervals([item for item in group if item.name == left])
        right_intervals = merge_intervals([item for item in group if item.name == right])
        left_total_ns += interval_total(left_intervals)
        right_total_ns += interval_total(right_intervals)
        overlap_ns += interval_overlap(left_intervals, right_intervals)

    return OverlapSummary(
        overlap_ns=overlap_ns,
        left_total_ns=left_total_ns,
        right_total_ns=right_total_ns,
        left_fraction=overlap_ns / left_total_ns if left_total_ns else 0.0,
        right_fraction=overlap_ns / right_total_ns if right_total_ns else 0.0,
    )
