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
from attn_gym.warpdrive.examples import qwen_megakernel_viz as mkv
from attn_gym.warpdrive.examples.schedule_views import (
    grouped_gemm_view,
    grouped_tiles_decode,
    persistent_gemm_view,
)
from attn_gym.warpdrive.profiler import Range
from attn_gym.warpdrive.schedule_html import Item, schedule_html


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
    Compute work items become [table row, CTA, start, end] in us from the first start; load
    items and regions are not spans. The overlay lands on the rows' drawn items."""
    ctas, splits = 61, 4
    rows = mk.instruction_rows(mk.QwenDecodeMegakernel(splits=splits))
    pred = mk.list_schedule(rows, ctas)
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

    runner = SimpleNamespace(max_ctas=ctas, op=SimpleNamespace(splits=splits))
    events = [
        Range("compute", 5_000, 7_000, (1,), payload=1),
        Range("compute", 1_000, 2_500, (0,), payload=0),
        Range("load", 900, 1_000, (0,), warp=8, payload=0),
        Range("poll", 1_200, 1_500, (0,), payload=0),
    ]
    data = mkv.overlay(events, runner)
    assert data == {
        "unit": "us",
        "ctas": ctas,
        "splits": splits,
        "spans": [[0, 0, 0.0, 1.5], [1, 1, 4.0, 6.0]],
    }
    page = page_data(mkv.megakernel_view(ctas, splits, data))
    assert page["critical"] == path and sum(map(len, page["rows"])) == len(rows)
    assert len(page["measured"]) == 2
    data["spans"][1][1] = 2  # table row 1 runs on CTA 1: a capture of another table
    with pytest.raises(ValueError, match="not of this static table"):
        mkv.megakernel_view(ctas, splits, data)
    with pytest.raises(ValueError, match="no compute work-item ranges"):
        mkv.overlay(events[2:], runner)
