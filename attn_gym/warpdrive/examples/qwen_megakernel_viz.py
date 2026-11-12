"""The Qwen3 megakernel's static schedule, drawn per CTA (``schedule_html``).

``build_table`` assigns the step's instructions to CTAs by list schedule over a latency model
(``list_schedule``: ``ROW_COST`` per row, ``DEPENDENCY_LATENCY`` per producer group) and lays
them out ``[slot][cta]`` for ``StaticPersistent`` (``assign_static``). The page shows each
CTA's rows coloured by op and shaded by layer, their predicted start and end, the rows that
gate them (the last row of each producer group) and the predicted critical path. CTA rows
come from ``WorkStream.assignment``, so they are the rows the kernel runs.

``--timeline`` overlays a capture: a JSON file
``{"unit": "us", "ctas": 152, "splits": 16, "spans": [[table_row, cta, start, end], ...],
"tracks": [[cta, track, name, table_row, start, end], ...]}``, times from any common origin;
``ctas`` and ``splits`` are optional and checked against the drawn table. ``tracks`` (optional)
are every recorded range by role, drawn under a CTA when it is clicked. ``overlay`` builds the
file from a ``trace="globaltimer"`` launch (``qwen_megakernel_trace``)::

    runner = QwenMegakernel(weights, max_len, trace="globaltimer")
    runner(weights, token, pos, *caches)  # compile, warm, and set the step's token and position
    with trace_session(runner) as session:
        runner.launch(*caches)
    Path("capture.json").write_text(json.dumps(overlay(session.ranges, runner)))

Draw it::

    python -m attn_gym.warpdrive.examples.qwen_megakernel_viz OUT.html \\
        [--model Qwen/Qwen3-32B] [--ctas 152] [--splits 16] [--timeline capture.json]
"""

import argparse
import json
from pathlib import Path

from attn_gym.warpdrive.examples import qwen_megakernel as mk
from attn_gym.warpdrive.examples.qwen_megakernel_ref import QwenConfig
from attn_gym.warpdrive.profiler import Range
from attn_gym.warpdrive.schedule_html import Item, Track, schedule_html
from attn_gym.warpdrive.trace_session import role_tracks, work_item_ranges, work_item_spans


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


def megakernel_view(
    ctas: int,
    splits: int | None = None,
    measured: dict | None = None,
    model: QwenConfig = mk.QWEN3_0_6B,
    name: str = "Qwen3-0.6B",
) -> str:
    op = mk.QwenDecodeMegakernel(splits=splits, model=model)
    splits = op.splits
    rows, table = mk.instruction_rows(op), mk.build_table(op, ctas)
    cost = mk.row_cost(op)
    pred = mk.list_schedule(rows, ctas, cost)
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
            shade=words[1] / (op.layers - 1) if words[0] != mk.LMHEAD else 0.0,
            start=pred[i].start,
            end=pred[i].end,
            deps=pred[i].gates,
        )
        for t, i in sorted(table_row.items(), key=lambda kv: kv[1])
        for words, deps, _ in [rows[i]]
    ]
    spans, tracks = None, None
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
        tracks = measured_tracks(measured.get("tracks", ()), table_row, host)
    path = critical_path(pred)
    return schedule_html(
        f"{name} decode step",
        f"{len(rows):,} work items · {ctas} CTAs · {splits} KV splits",
        items,
        by_cta,
        critical=path,
        measured=spans,
        tracks=tracks,
        kinds=mk.OP_NAMES,
        notes=[
            (
                "Colours are operations; lighter shades are later layers. Arrows on hover point "
                "to the items it waits for."
            ),
            "Predicted cost per item (us): "
            + ", ".join(f"{mk.OP_NAMES[k]} {c:.1f}" for k, c in cost.items())
            + f", plus {mk.DEPENDENCY_LATENCY:g} us per dependency.",
        ],
    )


def measured_tracks(
    rows, table_row: dict[int, int], table: list[list[int]]
) -> dict[int, list[Track]]:
    """``schedule_html`` tracks from the capture's ``[cta, track, name, table_row, start, end]``
    rows. A span whose payload is a drawn table row takes that row's item; a work item (named
    after its role) is renamed after the row's instruction."""
    lanes: dict[int, dict[str, list]] = {}
    for cta, track, name, t, start, end in rows:
        item = table_row.get(t)
        if item is not None and name in ("compute", "load"):
            name = mk.OP_NAMES[table[t][0]]
        lanes.setdefault(cta, {}).setdefault(track, []).append((name, start, end, item))
    return {
        cta: [Track(name, spans) for name, spans in by_name.items()]
        for cta, by_name in lanes.items()
    }


def overlay(ranges: list[Range], runner: mk.QwenMegakernel) -> dict:
    """The ``--timeline`` JSON of one traced launch, in us from the earliest compute start: each
    table row's compute work item on its CTA, every range on its role's track, and the table's
    CTA and split counts."""
    spans = work_item_spans(ranges, "compute")
    origin = min(item.start_ns for item in work_item_ranges(ranges, "compute").values())
    return {
        "unit": "us",
        "ctas": runner.max_ctas,
        "splits": runner.op.splits,
        "spans": spans,
        "tracks": role_tracks(ranges, runner.op, origin),
    }


def load_config(path_or_repo: str) -> QwenConfig:
    """The shape in a checkpoint's ``config.json``: a directory, or a Hub repo (fetches only
    that file)."""
    path = Path(path_or_repo) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(path_or_repo, "config.json"))
    return QwenConfig.from_hf(json.loads(path.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ctas", type=int, default=152, help="resident CTAs (GB300: 152 SMs)")
    parser.add_argument("--splits", type=int, help="KV splits (default: the kernel's)")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-0.6B", help="HF repo or checkpoint directory (its config)"
    )
    parser.add_argument("--timeline", type=Path, help="measured spans (see the docstring)")
    args = parser.parse_args()
    model = load_config(args.model)
    if args.splits is None:
        args.splits = mk.QwenDecodeMegakernel(model=model).splits
    measured = json.loads(args.timeline.read_text()) if args.timeline else None
    for key in ("ctas", "splits"):  # the capture must be of the drawn table
        if measured and measured.get(key, getattr(args, key)) != getattr(args, key):
            parser.error(
                f"the capture ran with {key} = {measured[key]}; pass --{key} {measured[key]}"
            )
    name = Path(args.model).name
    args.out.write_text(megakernel_view(args.ctas, args.splits, measured, model, name))
    print(args.out)


if __name__ == "__main__":
    main()
