"""The Qwen3 megakernel's static schedule, drawn per CTA (``schedule_html``).

``build_table`` assigns the step's instructions to CTAs by list schedule over a latency model
(``list_schedule``: ``ROW_COST`` per row, ``DEPENDENCY_LATENCY`` per producer group) and lays
them out ``[slot][cta]`` for ``StaticPersistent`` (``assign_static``). The page shows each
CTA's rows coloured by op and shaded by layer, their predicted start and end, the rows that
gate them (the last row of each producer group) and the predicted critical path. CTA rows
come from ``WorkStream.assignment``, so they are the rows the kernel runs.

``--timeline`` overlays a capture: a JSON file
``{"unit": "us", "ctas": 152, "splits": 16, "spans": [[table_row, cta, start, end], ...]}``,
times from any common origin; ``ctas`` and ``splits`` are optional and checked against the
drawn table. ``overlay`` builds one from the compute work items of a ``trace="globaltimer"``
launch (``qwen_megakernel_trace``)::

    runner = QwenMegakernel(weights, max_len, trace="globaltimer")
    runner(weights, token, pos, *caches)  # compile, warm, and set the step's token and position
    with trace_session(runner) as session:
        runner.launch(*caches)
    Path("capture.json").write_text(json.dumps(overlay(session.ranges, runner)))

Draw it::

    python -m attn_gym.warpdrive.examples.qwen_megakernel_viz OUT.html \\
        [--ctas 152] [--splits 16] [--timeline capture.json]
"""

import argparse
import json
from pathlib import Path

from attn_gym.warpdrive.examples import qwen_megakernel as mk
from attn_gym.warpdrive.examples.qwen_megakernel_trace import item_ranges
from attn_gym.warpdrive.profiler import Range
from attn_gym.warpdrive.schedule_html import Item, schedule_html


def critical_path(pred: list[mk.RowSpan]) -> list[int]:
    """Rows from launch to the last-finishing row, each the one that set the next's start."""
    i = max(range(len(pred)), key=lambda r: pred[r].end)
    path = []
    while i is not None:
        path.append(i)
        i = pred[i].after
    return path[::-1]


def _label(words, deps, table_row: int) -> str:
    kind, layer, a, b = words
    match kind:
        case mk.QKV:
            what = f"QKV layer {layer} · rows {a * mk.ROWS}-{a * mk.ROWS + mk.ROWS - 1}"
        case mk.ATTN:
            what = f"ATTN layer {layer} · kv head {a}, split {b}"
        case mk.OPROJ | mk.DOWN:
            what = f"{mk.OP_NAMES[kind]} layer {layer} · rows {a * mk.ROWS}.. · k split {b}"
        case mk.GATEUP:
            what = f"GATEUP layer {layer} · tile {a}"
        case _:
            what = f"LMHEAD · row tiles {a}-{a + b - 1}"
    waits = ", ".join(" ".join(map(str, d)) for d in deps) or "nothing"
    return f"{what}\nwaits for: {waits}\ntable row {table_row}"


def megakernel_view(ctas: int, splits: int = 16, measured: dict | None = None) -> str:
    op = mk.QwenDecodeMegakernel(splits=splits)
    rows, table = mk.instruction_rows(op), mk.build_table(op, ctas)
    pred = mk.list_schedule(rows, ctas)
    by_cta: list[list[int]] = [[] for _ in range(ctas)]
    for i, p in enumerate(pred):
        by_cta[p.cta].append(i)
    # The kernel's view: StaticPersistent over the [slot][cta] table, padding rows dropped.
    host = table.tolist()
    assigned = op.work.assignment(table, max_ctas=ctas, decode=lambda t: (host[t][0] >= 0, t))
    table_row, table_cta = {}, {}
    for cta, items in enumerate(assigned):
        assert len(items) == len(by_cta[cta]), "build_table disagrees with list_schedule"
        for (t, _), i in zip(items, by_cta[cta]):
            assert host[t][:4] == rows[i][0], "build_table disagrees with list_schedule"
            table_row[t], table_cta[t] = i, cta
    items = [
        Item(
            _label(words, deps, t),
            mk.OP_NAMES[words[0]],
            shade=words[1] / (mk.LAYERS - 1) if words[0] != mk.LMHEAD else 0.0,
            start=pred[i].start,
            end=pred[i].end,
            deps=pred[i].gates,
        )
        for t, i in sorted(table_row.items(), key=lambda kv: kv[1])
        for words, deps, _ in [rows[i]]
    ]
    spans = None
    if measured is not None:
        # A capture of another table (atomic schedule, an ops subset) would land on the wrong
        # rows. Reject spans whose CTA disagrees with this table's assignment.
        for t, cta, _, _ in measured["spans"]:
            if table_cta.get(t) != cta:
                raise ValueError(
                    f"measured table row {t} ran on CTA {cta}, but this table runs it on "
                    f"{table_cta.get(t)}: the capture is not of this static table"
                )
        spans = {table_row[t]: (cta, s, e) for t, cta, s, e in measured["spans"]}
    path = critical_path(pred)
    makespan = max(p.end for p in pred)
    return schedule_html(
        f"Qwen3-0.6B decode megakernel: {len(rows)} rows on {ctas} CTAs",
        f"{op.work.describe()} · splits {splits} · predicted makespan {makespan:.1f} us",
        items,
        by_cta,
        critical=path,
        measured=spans,
        kinds=mk.OP_NAMES,
        notes=[
            "Predicted spans are list_schedule's model: ROW_COST "
            + ", ".join(f"{mk.OP_NAMES[k]} {c} us" for k, c in mk.ROW_COST.items())
            + f", plus {mk.DEPENDENCY_LATENCY} us per producer group. Illustrative, not "
            "measured.",
            (
                "Arrows on hover: the last row of each producer group the row waits for (the one "
                "that gates it in the model). Critical path: rows that each set the next's start."
            ),
            (
                "Shade: layer (darker = earlier). CTA rows are WorkStream.assignment over the "
                "[slot][cta] table, as StaticPersistent runs them."
            ),
            *(
                [f"Measured overlay: {len(spans)} spans from the --timeline capture."]
                if spans
                else []
            ),
        ],
    )


def overlay(ranges: list[Range], runner: mk.QwenMegakernel) -> dict:
    """The ``--timeline`` JSON of one traced launch: each table row's compute work item on its
    CTA, in us from the earliest start."""
    items = sorted(item_ranges(ranges).items())
    if not items:
        raise ValueError(
            "no compute work-item ranges: trace a launch of a trace='globaltimer' runner"
        )
    t0 = min(e.start_ns for _, e in items)
    spans = [[row, e.cta[0], (e.start_ns - t0) / 1e3, (e.end_ns - t0) / 1e3] for row, e in items]
    return {"unit": "us", "ctas": runner.max_ctas, "splits": runner.op.splits, "spans": spans}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ctas", type=int, default=152, help="resident CTAs (GB300: 152 SMs)")
    parser.add_argument("--splits", type=int, default=16)
    parser.add_argument("--timeline", type=Path, help="measured spans (see the docstring)")
    args = parser.parse_args()
    measured = json.loads(args.timeline.read_text()) if args.timeline else None
    for key in ("ctas", "splits"):  # the capture must be of the drawn table
        if measured and measured.get(key, getattr(args, key)) != getattr(args, key):
            parser.error(
                f"the capture ran with {key} = {measured[key]}; pass --{key} {measured[key]}"
            )
    args.out.write_text(megakernel_view(args.ctas, args.splits, measured))
    print(args.out)


if __name__ == "__main__":
    main()
