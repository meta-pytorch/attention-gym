"""Render a static schedule as a self-contained HTML page.

The caller converts ``WorkStream.assignment`` results to ``Item`` objects.
``rows[cta]`` lists their positions in execution order. The page draws a timeline if all items
have predicted spans, otherwise a CTA-by-slot grid, and a work-space map if all items have 2-D
coordinates. Drag the timeline to zoom and click a CTA to focus it across views. Predicted
and measured spans can be compared on one axis; hover shows dependencies and timing deltas.
A focused CTA expands into its measured ``Track``s (one per role, for example). Colours
follow item kind or list position.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from attn_gym.warpdrive.html_page import asset, page

COLOR_BY = ("kind", "slot")
# Reserve red for problems, not item kinds.
PALETTE = (
    "#3f7d5c",
    "#3b689c",
    "#a26f22",
    "#8a5a9e",
    "#3f8a8a",
    "#b0567a",
    "#6b7a3a",
    "#5a6ea8",
    "#8a6a4a",
    "#6a7a6e",
)


@dataclass(frozen=True)
class Item:
    """A work item with display metadata and optional timing/dependencies.

    ``label`` is newline-separated hover text. ``kind`` selects a colour; ``shade`` in
    [0, 1] lightens it. ``start``/``end`` use the page's ``unit``. ``deps`` are producer
    positions in ``items``; ``coord`` is ``(row, column)`` in the work-space map.
    """

    label: str
    kind: str = ""
    shade: float = 0.0
    start: float | None = None
    end: float | None = None
    deps: tuple[int, ...] = ()
    coord: tuple[int, int] | None = None


@dataclass(frozen=True)
class Track:
    """One measured lane drawn under its CTA's timeline row when the CTA is focused.

    ``spans`` are ``(name, start, end, item)`` in the page's ``unit``; ``item`` is the position
    in ``items`` the span belongs to (it takes that item's colour and hover), or None.
    """

    name: str
    spans: Sequence[tuple[str, float, float, int | None]]


def schedule_html(
    title: str,
    subtitle: str,
    items: Sequence[Item],
    rows: Sequence[Sequence[int]],
    *,
    unit: str = "us",
    critical: Sequence[int] = (),
    measured: Mapping[int, tuple[int, float, float]] | None = None,
    tracks: Mapping[int, Sequence[Track]] | None = None,
    color_by: str = "kind",
    kinds: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> str:
    """Build a schedule page with every item assigned to exactly one CTA.

    ``rows[cta]`` gives positions in ``items``, in run order. ``critical`` lists the
    predicted critical path in order. ``measured`` maps item positions to
    ``(cta, start, end)`` from a capture of the same static table, in the same units and
    relative to a comparable origin as the prediction. The viewer can show either timing
    source or both; spans within each CTA and timing source must not overlap. ``critical``,
    ``measured`` and ``tracks`` require all items to have predicted spans. ``tracks[cta]`` are that CTA's measured lanes, drawn on
    the timeline under its row while it is focused. ``notes`` appear below the figures for
    provenance.
    ``kinds`` sets legend/colour order; unlisted kinds follow in order of appearance.
    """
    if color_by not in COLOR_BY:
        raise ValueError(f"color_by must be one of {COLOR_BY}, got {color_by!r}")
    placed = sorted(i for row in rows for i in row)
    if placed != list(range(len(items))):
        raise ValueError("rows must place every item exactly once")
    tracks = {cta: lanes for cta, lanes in (tracks or {}).items() if lanes}
    spans = [span for lanes in tracks.values() for lane in lanes for span in lane.spans]
    named = [
        *critical,
        *(measured or ()),
        *(d for item in items for d in item.deps),
        *(item for *_, item in spans if item is not None),
    ]
    if any(not 0 <= i < len(items) for i in named):
        raise ValueError("deps, critical, measured and tracks name items by position in items")
    if any(not 0 <= cta < len(rows) for cta, _, _ in (measured or {}).values()):
        raise ValueError("a measured span's CTA is not one of rows")
    if any(not 0 <= cta < len(rows) for cta in tracks):
        raise ValueError("a track's CTA is not one of rows")
    kinds = list(dict.fromkeys([*kinds, *(item.kind for item in items)]))
    timed = bool(items) and all(item.start is not None and item.end is not None for item in items)
    if (critical or measured or tracks) and not timed:
        raise ValueError(
            "critical, measured and tracks are drawn on the timeline: give items spans"
        )
    data = {
        "kinds": kinds,
        "colors": [PALETTE[i % len(PALETTE)] for i in range(len(kinds))],
        "label": [item.label for item in items],
        "kind": [kinds.index(item.kind) for item in items],
        "shade": [round(item.shade, 3) for item in items],
        "start": [item.start for item in items] if timed else None,
        "end": [item.end for item in items] if timed else None,
        "deps": [list(item.deps) for item in items] if any(i.deps for i in items) else None,
        "coord": [list(item.coord) for item in items]
        if items and all(item.coord is not None for item in items)
        else None,
        "rows": [list(row) for row in rows],
        "unit": unit,
        "critical": list(critical),
        "measured": {str(k): list(v) for k, v in measured.items()} if measured else None,
        "tracks": {
            str(cta): [
                [
                    lane.name,
                    [
                        [n, s, e, -1 if i is None else i]
                        for n, s, e, i in sorted(lane.spans, key=lambda span: span[1])
                    ],
                ]
                for lane in lanes
            ]
            for cta, lanes in tracks.items()
        }
        or None,
        "colorBy": color_by,
    }
    about = []
    if timed:
        about.append(
            "Predicted: the host scheduler's cost model (estimated time per item plus a fixed "
            "wait per dependency). An estimate, not a measurement."
        )
    if measured:
        about.append(
            "Measured: timestamps recorded on the GPU during a real launch of this same "
            "schedule. This is the real timing."
        )
    about.append("Each row is one CTA (thread block); each box is one work item.")
    if tracks:
        about.append("Tracks: a focused CTA's measured ranges, drawn below its work items.")
    notes_html = "".join(f"<li>{html.escape(line)}</li>" for line in [*about, *notes])
    colors = {"kind": "Kind", "slot": "Order"}
    body = f"""<div class="toolbar">
<span class="group" id="timing-controls" hidden><span class="label">Show</span>
<span class="seg" id="timing" role="group" aria-label="Timing">
<button type="button" data-v="predicted" title="Scheduler estimate">Predicted</button>
<button type="button" data-v="measured" title="Real launch">Measured</button>
<button type="button" data-v="both" title="Two lanes per CTA: predicted on top">Both</button>
</span></span>
<span class="group"><span class="label">Colour</span>
<span class="seg" id="color-by" role="group" aria-label="Colour by">{
        "".join(
            f'<button type="button" data-v="{c}">{label}</button>' for c, label in colors.items()
        )
    }</span></span>
<button type="button" id="critical" hidden title="Highlight the predicted critical path">
Critical path</button>
<select id="add-cta" aria-label="Add a CTA" hidden></select>
<button type="button" id="reset-view" hidden>Reset view</button>
<span class="spacer"></span>
<span class="readout" id="readout" aria-live="polite"></span>
</div>
<div class="legend" id="legend"></div>
<div class="path" id="path" hidden></div>
<div id="views"></div>
<details class="about"><summary>What am I looking at?</summary><ul>{notes_html}</ul></details>
<div id="tip" role="status" hidden></div>"""
    return page(title, subtitle, body, asset("schedule.js"), data, asset("schedule.css"))
