"""Decode an IKET capture (``run-iket``'s JSON trace) into ranges."""

from __future__ import annotations

import json
from pathlib import Path

from attn_gym.warpdrive.profiler.ranges import Range

__all__ = ["parse_iket"]


def parse_iket(path: str | Path) -> list[Range]:
    """Return the ranges of a one-launch IKET capture, with their CTA, warp and payload."""
    capture = json.loads(Path(path).read_text())
    launches = capture.get("launches", [])
    if len(launches) != 1:
        raise ValueError(f"Expected one IKET launch, found {len(launches)}")
    names = capture["stringTable"]
    locations = capture["locationTable"]
    ranges = []
    for item in launches[0].get("ranges", []):
        location = locations[item["warpLocIdxs"][0]]
        ranges.append(
            Range(
                names[item["rangeNameIdx"]],
                item["startTs"],
                item["endTs"],
                tuple(location["ctaId"]),
                location["warpId"],
                payload=item.get("payloadVal"),
            )
        )
    return ranges
