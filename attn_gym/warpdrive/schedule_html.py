"""Self-contained HTML view of a static schedule: which CTA runs which work items, in order.

The input is what ``WorkStream.assignment`` returns, turned into ``Item``s by the caller (who
knows what an item means): ``rows[cta]`` lists the items CTA ``cta`` runs, by position in
``items``. The page draws, from the same data:

* a **CTA x slot grid** (one cell per item, slot = its position in the CTA's list), where wave
  quantization shows as a ragged last column;
* a **timeline** when items carry a predicted ``start``/``end`` (a latency model): one row
  per CTA, with each item's dependency arrows (``deps``) on hover and the ``critical`` path
  on demand, and, when given, ``measured`` spans of a capture instead of the prediction;
* a **work-space map** when items carry a 2-D ``coord`` (a GEMM's output tiles).

Hovering an item in any view outlines its CTA's items in all of them. Cells are coloured by
item kind, by slot, or by CTA.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from attn_gym.warpdrive.html_page import asset, page

COLOR_BY = ("kind", "slot", "cta")
# Kind colours, muted to sit on the paper palette (red is left for problems).
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
    """One work item. ``label``: hover text (lines split on newlines). ``kind``: colour key.
    ``shade`` in [0, 1] lightens the kind colour (a layer index, say). ``start``/``end``:
    predicted span, in the page's ``unit``. ``deps``: positions of producer items, drawn as
    arrows into this one. ``coord``: ``(row, column)`` in the work-space map."""

    label: str
    kind: str = ""
    shade: float = 0.0
    start: float | None = None
    end: float | None = None
    deps: tuple[int, ...] = ()
    coord: tuple[int, int] | None = None


def schedule_html(
    title: str,
    subtitle: str,
    items: Sequence[Item],
    rows: Sequence[Sequence[int]],
    *,
    unit: str = "us",
    critical: Sequence[int] = (),
    measured: Mapping[int, tuple[int, float, float]] | None = None,
    color_by: str = "kind",
    kinds: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> str:
    """The page. ``rows[cta]``: positions in ``items``, in run order. ``critical``: items of
    the predicted critical path, in order. ``measured``: ``{item: (cta, start, end)}`` from a
    capture, drawn on the timeline in place of the prediction when toggled. ``notes``: lines
    under the figures (provenance). ``kinds``: legend and colour order (kinds not listed
    follow in order of appearance)."""
    if color_by not in COLOR_BY:
        raise ValueError(f"color_by must be one of {COLOR_BY}, got {color_by!r}")
    placed = sorted(i for row in rows for i in row)
    if placed != list(range(len(items))):
        raise ValueError("rows must place every item exactly once")
    named = [*critical, *(measured or ()), *(d for item in items for d in item.deps)]
    if any(not 0 <= i < len(items) for i in named):
        raise ValueError("deps, critical and measured name items by position in items")
    if any(not 0 <= cta < len(rows) for cta, _, _ in (measured or {}).values()):
        raise ValueError("a measured span's CTA is not one of rows")
    kinds = list(dict.fromkeys([*kinds, *(item.kind for item in items)]))
    timed = bool(items) and all(item.start is not None and item.end is not None for item in items)
    if (critical or measured) and not timed:
        raise ValueError("critical and measured are drawn on the timeline: give items spans")
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
        "colorBy": color_by,
    }
    notes_html = "".join(f"<li>{html.escape(line)}</li>" for line in notes)
    body = f"""<div class="controls">
<span class="figlabel">colour by</span>
<span class="seg" id="color-by">{
        "".join(f'<button type="button" data-v="{c}">{c}</button>' for c in COLOR_BY)
    }</span>
<button type="button" id="critical" hidden>critical path</button>
<button type="button" id="measured" hidden>measured</button>
<span class="readout" id="readout"></span>
</div>
<div class="legend" id="legend"></div>
<div id="views"></div>
<div class="notes"><section><p class="figlabel">notes</p><ul>{notes_html}</ul></section></div>
<div id="tip" role="status" hidden></div>"""
    return page(title, subtitle, body, asset("schedule.js"), data, asset("schedule.css"))
