"""role_summary over IKET or globaltimer ranges."""

from types import SimpleNamespace

from attn_gym.warpdrive.profiler.ranges import Range
from attn_gym.warpdrive.trace_session import role_summary


def test_role_summary_containment():
    ranges = [
        Range("load", 0, 100, (0,), 4),
        Range("load", 100, 300, (0,), 4),
        Range("ab.acquire", 10, 30, (0,), 4),
        Range("turn.turn", 110, 150, (0,), 4),
        Range("inner.sync", 120, 130, (0,), 4),
        Range("user", 50, 70, (0,), 4),
        Range("ab.wait", 10, 30, (1,), 4),  # unrelated CTA
        Range("ab.peek", 10, 30, (0,), 5),  # unrelated warp
        Range("ab.tail", 290, 310, (0,), 4),  # not contained
    ]
    report = role_summary(ranges)
    (row,) = report.roles
    assert (row.name, row.warp, row.items) == ("load", 4, 2)
    # inner.sync sits inside turn.turn: counted once, through the turn.
    assert (row.p50_ns, row.p90_ns, row.item_ns, row.wait_ns) == (150, 190, 300, 60)
    assert row.wait_share == 60 / 300
    assert row.work_share == 1 - 60 / 300
    assert "p90 ns" in str(report) and "20.0%" in str(report)
    op = SimpleNamespace(
        _collect=lambda: None,
        _ws_roles=[
            SimpleNamespace(name="load", first_warp=4),
            SimpleNamespace(name="idle", first_warp=6),
        ],
    )
    empty = role_summary([], op).roles
    assert all(r.items == r.wait_ns == r.item_ns == r.wait_share == 0 for r in empty)
    assert role_summary([]).roles == ()


def test_role_summary_multiple_ctas():
    ranges = [
        r
        for cta in range(64)
        for r in (
            Range("load", 0, 100, (cta,), 4),
            Range("ab.acquire", 10, 30, (cta,), 4),
            Range("load", 100, 300, (cta,), 4),
            Range("ab.wait", 110, 150, (cta,), 4),
        )
    ]
    (row,) = role_summary(ranges).roles
    assert (row.items, row.item_ns, row.wait_ns) == (128, 64 * 300, 64 * 60)
    assert row.wait_share == 0.2
