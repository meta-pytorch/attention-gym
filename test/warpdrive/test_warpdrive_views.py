"""Test self-contained schedule pages built from WorkStream.assignment.

Every item appears once. GEMM views cover their work spaces; the megakernel view draws
list_schedule's latency model and a traced overlay.
"""

import json
import math
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

import itertools

from attn_gym.warpdrive.examples import qwen_megakernel as mk
from attn_gym.warpdrive.examples.qwen_megakernel import viz as mkv
from attn_gym.warpdrive.examples.schedule_views import (
    grouped_gemm_view,
    grouped_tiles_decode,
    persistent_gemm_view,
)
from attn_gym.warpdrive.profiler import Range
from attn_gym.warpdrive.schedule_html import Item, Track, schedule_html
from attn_gym.warpdrive.trace_session import role_tracks


def page_data(text: str) -> dict:
    """The page's embedded data, after checking it references nothing outside itself."""
    assert not re.search(r"""(src|href)=|<link|@import|url\(""", text)
    assert text.count("<script") == 2 and text.count("</script>") == 2  # the data, the code
    return json.loads(
        re.search(r'<script type="application/json" id="data">(.*?)</script>', text, re.DOTALL)[1]
    )


def test_schedule_page_carries_each_item_once():
    items = [Item("a", "x", start=0, end=1), Item("b", "y", start=1, end=3, deps=(0,))]
    data = page_data(schedule_html("t", "s", items, [[0], [1]], critical=[0, 1]))
    assert data["rows"] == [[0], [1]] and data["label"] == ["a", "b"]
    assert data["start"] == [0, 1] and data["deps"] == [[], [0]] and data["coord"] is None
    with pytest.raises(ValueError, match="exactly once"):
        schedule_html("t", "s", items, [[0, 1], [1]])
    with pytest.raises(ValueError, match="exactly once"):
        schedule_html("t", "s", items, [[0]])
    with pytest.raises(ValueError, match="by position in items"):
        schedule_html("t", "s", [*items, Item("c", deps=(5,))], [[0], [1, 2]])
    with pytest.raises(ValueError, match="CTA is not one of rows"):
        schedule_html("t", "s", items, [[0], [1]], measured={0: (2, 0.0, 1.0)})
    with pytest.raises(ValueError, match="give items spans"):
        schedule_html("t", "s", [Item("a"), Item("b")], [[0], [1]], measured={0: (0, 0.0, 1.0)})


def test_role_tracks_nest_regions_by_depth():
    """Depth counts enclosing ranges on the same CTA and warp; a region with its work item's
    exact span nests inside it whatever the record order; other CTAs and unknown warps do
    not interfere."""
    roles = [SimpleNamespace(name="compute", first_warp=0, warps=range(4))]
    kernel = SimpleNamespace(_ws_roles=roles)
    ranges = [
        Range("poll", 0, 4_000, (0,), warp=0, payload=0),  # same span as its work item
        Range("compute", 0, 4_000, (0,), warp=0, payload=0),
        Range("stream", 1_000, 3_000, (0,), warp=0, payload=0),
        Range("stage", 1_500, 2_000, (0,), warp=0, payload=0),
        Range("compute", 4_000, 5_000, (0,), warp=0, payload=1),  # starts as the first ends
        Range("compute", 500, 900, (1,), warp=0, payload=2),
        Range("compact", 0, 9_000, (0,), warp=None),
    ]
    assert role_tracks(ranges, kernel, 0) == [
        [0, "compute w0-3", "compute", 0, 0.0, 4.0],
        [0, "compute +1", "poll", 0, 0.0, 4.0],
        [0, "compute +2", "stream", 0, 1.0, 3.0],
        [0, "compute +3", "stage", 0, 1.5, 2.0],
        [0, "compute w0-3", "compute", 1, 4.0, 5.0],
        [1, "compute w0-3", "compute", 2, 0.5, 0.9],
    ]


def test_tracks_are_sorted_per_lane_and_name_drawn_items():
    items = [Item("a", "x", start=0, end=1), Item("b", "y", start=1, end=3)]
    lane = Track("compute", [("poll", 2.0, 2.5, 1), ("compute", 0.0, 1.0, None)])
    data = page_data(schedule_html("t", "s", items, [[0], [1]], tracks={1: [lane], 0: []}))
    assert data["tracks"] == {
        "1": [["compute", [["compute", 0.0, 1.0, -1], ["poll", 2.0, 2.5, 1]]]]
    }
    assert page_data(schedule_html("t", "s", items, [[0], [1]]))["tracks"] is None
    with pytest.raises(ValueError, match="track's CTA is not one of rows"):
        schedule_html("t", "s", items, [[0], [1]], tracks={2: [lane]})
    with pytest.raises(ValueError, match="by position in items"):
        schedule_html("t", "s", items, [[0], [1]], tracks={0: [Track("c", [("x", 0, 1, 7)])]})
    with pytest.raises(ValueError, match="give items spans"):
        schedule_html("t", "s", [Item("a"), Item("b")], [[0], [1]], tracks={0: [lane]})


def test_schedule_preserves_both_timing_sources_and_partial_capture():
    items = [
        Item("first", start=0, end=2, coord=(0, 0)),
        Item("second", start=3, end=5, coord=(0, 1)),
    ]
    data = page_data(
        schedule_html(
            "Comparison", "One static table", items, [[0, 1]], measured={1: (0, 4.5, 8.0)}
        )
    )
    assert data["start"] == [0, 3] and data["end"] == [2, 5]
    assert data["measured"] == {"1": [0, 4.5, 8.0]}
    assert data["coord"] == [[0, 0], [0, 1]] and data["unit"] == "us"
    assert data["rows"] == [[0, 1]]


def test_labels_cannot_break_out_of_the_page_data():
    label = 'a</script><!--<script>alert(1)</script>"'
    data = page_data(schedule_html("t", "s", [Item(label)], [[0]]))
    assert data["label"] == [label]


def test_persistent_gemm_view_covers_every_tile_once():
    data = page_data(persistent_gemm_view(1000, 3000, 37))
    tiles = {tuple(c) for c in data["coord"]}
    assert len(tiles) == len(data["coord"]) == math.ceil(1000 / 128) * math.ceil(3000 / 256)
    counts = [len(row) for row in data["rows"]]
    assert len(counts) == 37 and max(counts) - min(counts) <= 1  # static: one wave apart


def test_grouped_host_decode_covers_each_group_exactly():
    """Kept items tile each group's rows once per n tile; the padded indices are dropped."""
    sizes, n_tiles, tile_m = [300, 0, 128, 1, 517], 3, 128
    offsets = [sum(sizes[: g + 1]) for g in range(len(sizes))]
    decode = grouped_tiles_decode(offsets, offsets[-1], n_tiles, tile_m)
    padded = (math.ceil(offsets[-1] / tile_m) + len(sizes)) * n_tiles
    rows = {}
    for index in range(padded):
        keep, item = decode(index)
        if keep:
            group, row0, count, n = item
            rows.setdefault((group, n), []).extend(range(row0, row0 + count))
    for g, size in enumerate(sizes):
        for n in range(n_tiles):
            assert rows.get((g, n), []) == list(range(offsets[g] - size, offsets[g]))
    data = page_data(grouped_gemm_view(sizes, n_tiles * 256, 16))
    assert len(data["label"]) == sum(math.ceil(size / tile_m) for size in sizes) * n_tiles


def test_megakernel_model_and_traced_overlay():
    """The table runs list_schedule's CTAs (megakernel_view asserts it per CTA); rows wait
    for their gates and their CTA, and the critical path runs from 0 to the makespan.
    Compute work items become [table row, CTA, start, end] in us from the first compute start; load
    items and regions are not spans but go on their role's tracks, regions one level down.
    The overlay lands on the rows' drawn items."""
    ctas, splits = 61, 4
    op = mk.QwenDecodeMegakernel(splits=splits)
    rows = mk.instruction_rows(op)
    pred = mk.list_schedule(rows, ctas, mk.row_cost(op, ctas))
    for p in pred:
        assert all(p.start >= pred[g].end + mk.DEPENDENCY_LATENCY - 1e-9 for g in p.gates)
        if p.after is None:
            assert p.start == 0
        else:  # the row that set its start: a gate plus the latency, or its CTA's previous row
            latency = mk.DEPENDENCY_LATENCY if p.after in p.gates else 0.0
            assert p.start == pred[p.after].end + latency
    for cta in range(ctas):
        spans = [(p.start, p.end) for p in pred if p.cta == cta]
        assert all(a[1] <= b[0] + 1e-9 for a, b in itertools.pairwise(spans))
    path = mkv.critical_path(pred)
    assert pred[path[0]].start == 0 and pred[path[-1]].end == max(p.end for p in pred)

    roles = [
        SimpleNamespace(name="compute", first_warp=0, warps=range(8)),
        SimpleNamespace(name="load", first_warp=8, warps=range(8, 9)),
    ]
    runner = SimpleNamespace(max_ctas=ctas, op=SimpleNamespace(splits=splits, _ws_roles=roles))
    events = [
        Range("compute", 5_000, 7_000, (1,), warp=0, payload=1),
        Range("compute", 1_000, 2_500, (0,), warp=0, payload=0),
        Range("load", 900, 1_000, (0,), warp=8, payload=0),
        Range("poll", 1_200, 1_500, (0,), warp=0, payload=0),
        Range("stream", 1_500, 2_500, (0,), warp=0, payload=0),
    ]
    data = mkv.overlay(events, runner)
    assert data == {
        "unit": "us",
        "ctas": ctas,
        "splits": splits,
        "spans": [[0, 0, 0.0, 1.5], [1, 1, 4.0, 6.0]],
        "tracks": [
            [0, "compute w0-7", "compute", 0, 0.0, 1.5],
            [0, "compute +1", "poll", 0, 0.2, 0.5],
            [0, "compute +1", "stream", 0, 0.5, 1.5],
            [0, "load w8", "load", 0, -0.1, 0.0],
            [1, "compute w0-7", "compute", 1, 4.0, 6.0],
        ],
    }
    page = page_data(mkv.megakernel_view(ctas, splits, data))
    assert page["critical"] == path and sum(map(len, page["rows"])) == len(rows)
    assert len(page["measured"]) == 2
    op = page["kinds"][page["kind"][0]]  # work items are renamed after their instruction
    assert page["tracks"]["0"] == [
        ["compute w0-7", [[op, 0.0, 1.5, 0]]],
        ["compute +1", [["poll", 0.2, 0.5, 0], ["stream", 0.5, 1.5, 0]]],
        ["load w8", [[op, -0.1, 0.0, 0]]],
    ]
    data["spans"][1][1] = 2  # table row 1 runs on CTA 1: a capture of another table
    with pytest.raises(ValueError, match="not of this static table"):
        mkv.megakernel_view(ctas, splits, data)
    del data["tracks"]  # a capture from before tracks existed
    data["spans"][1][1] = 1
    assert page_data(mkv.megakernel_view(ctas, splits, data))["tracks"] is None
    with pytest.raises(ValueError, match="no compute work-item ranges"):
        mkv.overlay(events[2:], runner)
