"""Join measured ranges (IKET or globaltimer) to a scheduled pipeline plan."""

from __future__ import annotations

import bisect
import statistics
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from attn_gym.profiler.pipeline.plan import Timeline
from attn_gym.profiler.ranges import Range

__all__ = ["WAIT_OPS", "MeasuredCapture", "MeasuredRegion", "join"]

# A range named ``<x>.<op>`` with one of these ops, nested in a plan region's range on the same
# warp, measures a blocking wait inside that region: warpdrive's channel, turnstile and named
# barrier wait ranges (``enable_trace(waits=True)``).
WAIT_OPS = frozenset({"acquire", "wait", "peek", "tail", "turn", "sync"})


@dataclass(frozen=True)
class MeasuredRegion:
    """Envelope of one role region's ranges in one iteration, over the warps that recorded it.

    ``wait_ns``: the measured blocking wait inside the region (the largest per-warp sum of its
    nested wait ranges), or None when the role's waits were not traced.
    """

    name: str
    role: str
    iteration: int
    start_ns: int
    end_ns: int
    median_duration_ns: float
    sample_count: int
    wait_ns: int | None = None


@dataclass(frozen=True)
class MeasuredCapture:
    """One CTA's measured region timeline joined to a plan."""

    cta: tuple[int, ...]
    origin_ns: int
    duration_ns: int
    regions: tuple[MeasuredRegion, ...]


def is_wait(name: str) -> bool:
    return name.rpartition(".")[2] in WAIT_OPS and "." in name


def join(
    ranges: Iterable[Range],
    timeline: Timeline,
    *,
    cta: tuple[int, ...] | None = None,
) -> MeasuredCapture:
    """Assign one CTA's ranges to the timeline's (region, iteration)s and aggregate them.

    ``cta`` defaults to the lowest CTA that recorded a plan region. A warp's n-th range of a
    region is iteration n, so every recording warp must record each region of its role exactly
    once per iteration; ranges without a warp (compact records) count as one recorder per
    role. Wait ranges (``WAIT_OPS``) nested in a region's range on its warp set the region's
    ``wait_ns``; other ranges are ignored.
    """
    role_by_region = {item.region.name: item.region.role for item in timeline.regions}
    roles = {role.name: role for role in timeline.roles}
    ranges = list(ranges)
    if cta is None:
        ctas = [item.cta for item in ranges if item.name in role_by_region]
        if not ctas:
            raise ValueError("no range is named after a plan region")
        cta = min(ctas)
    ranges = [item for item in ranges if item.cta == cta]

    by_recorder: dict[object, list[Range]] = defaultdict(list)
    waits_by_warp: dict[int, list[Range]] = defaultdict(list)
    for item in ranges:
        role_name = role_by_region.get(item.name)
        if role_name is None:
            if item.warp is not None and is_wait(item.name):
                waits_by_warp[item.warp].append(item)
            continue
        role = roles[role_name]
        if item.warp is not None and not role.warp_start <= item.warp < role.warp_end:
            raise ValueError(
                f"range {item.name!r} appeared on warp {item.warp}, outside role {role_name!r}"
            )
        by_recorder[item.warp if item.warp is not None else role_name].append(item)

    regions_by_role: dict[str, list[str]] = defaultdict(list)
    for name, role_name in role_by_region.items():
        regions_by_role[role_name].append(name)
    samples: dict[tuple[str, int], list[tuple[int, int, int | None]]] = defaultdict(list)
    waited_roles: set[str] = set()
    for recorder, recorded in sorted(by_recorder.items(), key=lambda item: str(item[0])):
        recorded.sort(key=lambda item: item.start_ns)
        occurrences: dict[str, int] = defaultdict(int)
        keys = []
        for item in recorded:
            keys.append((item.name, occurrences[item.name]))
            occurrences[item.name] += 1
        role_name = role_by_region[recorded[0].name]
        for name in regions_by_role[role_name]:
            if occurrences[name] != timeline.iterations:
                who = f"warp {recorder}" if isinstance(recorder, int) else f"role {recorder!r}"
                raise ValueError(
                    f"{who} recorded {name!r} {occurrences[name]} times; the timeline has "
                    f"{timeline.iterations} iterations (schedule(iterations="
                    f"{occurrences[name]}) for a capture of that many)"
                )
        waited = nested_wait_ns(recorded, waits_by_warp.get(recorder, []))
        if any(wait is not None for wait in waited):
            waited_roles.add(role_name)
        for key, item, wait in zip(keys, recorded, waited, strict=True):
            samples[key].append((item.start_ns, item.end_ns, wait))

    expected = {item.key for item in timeline.regions}
    missing = sorted(expected - samples.keys())
    if missing:
        raise ValueError(f"capture of CTA {cta} is missing plan regions: {missing}")

    measured = []
    for (name, iteration), envelope in samples.items():
        role_name = role_by_region[name]
        measured.append(
            MeasuredRegion(
                name=name,
                role=role_name,
                iteration=iteration,
                start_ns=min(start for start, _, _ in envelope),
                end_ns=max(end for _, end, _ in envelope),
                median_duration_ns=statistics.median(end - start for start, end, _ in envelope),
                sample_count=len(envelope),
                wait_ns=(
                    max(wait or 0 for *_, wait in envelope) if role_name in waited_roles else None
                ),
            )
        )
    measured.sort(key=lambda item: (item.start_ns, item.role, item.iteration, item.name))
    origin_ns = min(item.start_ns for item in measured)
    return MeasuredCapture(
        cta=cta,
        origin_ns=origin_ns,
        duration_ns=max(item.end_ns for item in measured) - origin_ns,
        regions=tuple(measured),
    )


def nested_wait_ns(regions: list[Range], waits: list[Range]) -> list[int | None]:
    """For each of one warp's region ranges (sorted by start), the summed duration of the
    wait ranges nested in it, or None when none is."""
    totals: list[int | None] = [None] * len(regions)
    starts = [item.start_ns for item in regions]
    for wait in waits:
        index = bisect.bisect_right(starts, wait.start_ns) - 1
        if index >= 0 and wait.end_ns <= regions[index].end_ns:
            totals[index] = (totals[index] or 0) + wait.duration_ns
    return totals
