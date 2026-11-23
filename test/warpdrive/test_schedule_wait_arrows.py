"""Measured dependency anchors on schedule pages (no GPU required)."""

import json
import re

import pytest

from attn_gym.warpdrive.schedule_html import Item, Track, schedule_html


@pytest.mark.parametrize(
    ("producer_end", "deps", "lane", "spans", "expected"),
    [
        (159.52, (0,), "compute +1", [("poll", 157.76, 159.84, 1)], {"1": 159.84}),
        (159.52, (0,), "compute +1", [("stream", 157.76, 159.84, 1)], {}),
        (159.52, (0,), "load +1", [("poll", 157.76, 159.84, 1)], {}),
        (160.00, (0,), "compute +1", [("poll", 157.76, 159.84, 1)], {}),
        (159.52, (0, 2), "compute +1", [("poll", 157.76, 159.84, 1)], {}),
        (159.52, (0,), "compute +1", [("poll", 157.76, 159.84, 1), ("poll", 160, 161, 1)], {}),
    ],
)
def test_measured_input_wait_anchor(producer_end, deps, lane, spans, expected):
    items = [
        Item("DOWN", start=0, end=1),
        Item("QKV", start=1, end=2, deps=deps),
        Item("other", start=0, end=1),
    ]
    page = schedule_html(
        "test",
        "",
        items,
        [[0, 2], [1]],
        measured={0: (0, 156.832, producer_end), 1: (1, 157.44, 161.6)},
        tracks={1: [Track(lane, spans)]},
        input_wait=("compute +1", "poll"),
    )
    data = json.loads(
        re.search(r'<script type="application/json" id="data">(.*?)</script>', page, re.DOTALL)[1]
    )
    assert data["waitEnd"] == expected
