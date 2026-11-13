"""Where a Qwen megakernel step spends its time: a globaltimer trace decoded per table row.

``QwenMegakernel(..., trace="globaltimer")`` records, per CTA, every compute and load work item
and the compute role's ``trace_region``s, all with the table row as payload: ``poll`` (waiting
for the row's inputs), ``stream`` (a GEMV tile's weight stages), ``attn.blocks`` (an attention
row's KV blocks) and ``reduce.poll`` (the last down split fetching the other partials)::

    with trace_session(runner, trace_path="step.pftrace") as session:
        runner(weights, token, pos, *caches)  # one launch
    phase_spans(session.ranges, runner.table)  # {"QKV": 1.9, ...} us per layer

The Perfetto trace has one process per CTA and one thread per role. Work items are named after
their instruction (``QKV``, ``load QKV``, ...) with ``layer``/``a``/``b`` as args, and an arrow
runs from each input group's last-finishing producer row to the row that waited for it. Host
only; the kernel module does not import this.
"""

from __future__ import annotations

import math
import statistics
from contextlib import contextmanager

import torch

from attn_gym.warpdrive.examples.qwen_megakernel import (
    ATTN,
    DOWN,
    GATEUP,
    LMHEAD,
    OP_NAMES,
    OPROJ,
    QKV,
    REDUCE,
    ROUTER,
    QwenDecodeMegakernel,
    row_groups,
)
from attn_gym.warpdrive.profiler import Range
from attn_gym.warpdrive.trace_session import trace_session as kernel_session
from attn_gym.warpdrive.trace_session import work_item_ranges, write_trace

# The instruction kind each phase waits for; QKV and the LM head wait for the previous layer's
# down.
PRODUCER = {
    QKV: DOWN, ATTN: QKV, OPROJ: ATTN, ROUTER: OPROJ, GATEUP: OPROJ, DOWN: GATEUP, LMHEAD: DOWN,
    REDUCE: DOWN,
}  # fmt: skip
MOE_PRODUCER = {**PRODUCER, GATEUP: ROUTER}  # a MoE table's expert rows wait for the router
# Batched MoE: REDUCE rows publish the residual QKV and the LM head read.
MOE_BATCH_PRODUCER = {**MOE_PRODUCER, QKV: REDUCE, LMHEAD: REDUCE}


@contextmanager
def trace_session(runner, *, trace_path=None, device=None):
    """``trace_session.trace_session`` for one launch of a ``trace="globaltimer"`` runner: inside
    the block the runner records into the session's buffer. ``trace_path``: also write the
    Perfetto trace (``write_step_trace``)."""
    with kernel_session(runner.op, runner.max_ctas, device=device) as session:
        previous, runner.trace_buffer = runner.trace_buffer, session.prof.tensor
        try:
            yield session
        finally:
            runner.trace_buffer = previous
    if trace_path is not None:
        write_step_trace(trace_path, runner, session.ranges)


def write_step_trace(path, runner, ranges: list[Range]):
    """Write the Perfetto trace: work items named after their instruction with the row's words
    as args, and an arrow from the last-finishing row of each producer group to the row that
    waited for it."""
    rows = runner.table.tolist()

    def name(item: Range) -> str:
        if item.payload is None or item.name not in ("compute", "load"):
            return item.name
        op = OP_NAMES[rows[item.payload][0]]
        return op if item.name == "compute" else f"load {op}"

    def args(item: Range) -> dict:
        if item.payload is None or item.name not in ("compute", "load"):
            return {}
        op, layer, a, b, _ = rows[item.payload]
        return {"op": OP_NAMES[op], "layer": layer, "a": a, "b": b}

    computed = work_item_ranges(ranges, "compute")
    done = {row: item.end_ns for row, item in computed.items()}
    flows = [
        (computed[source], computed[target])
        for edges in input_edges(runner.op, rows, done).values()
        for source, target in edges
    ]
    return write_trace(path, runner.op, ranges, name=name, args=args, flows=flows)


def input_edges(
    kernel: QwenDecodeMegakernel, rows: list[list[int]], done: dict[int, float]
) -> dict[tuple, list]:
    """``{(producer op name, consumer op name): [(producer row, consumer row)]}``: for every row
    of ``kernel``'s table and each group it waits for, the group's last-finishing row.
    ``done`` maps row -> end time."""
    groups: dict[tuple, list[int]] = {}
    for row, (op, layer, a, b, _) in enumerate(rows):
        if op >= 0:
            groups.setdefault(row_groups(kernel, op, layer, a, b)[1], []).append(row)
    edges: dict[tuple, list] = {}
    for row, (op, layer, a, b, _) in enumerate(rows):
        if op < 0 or row not in done:
            continue
        for group in row_groups(kernel, op, layer, a, b)[0]:
            finished = [r for r in groups.get(group, ()) if r in done]
            if finished:
                last = max(finished, key=done.__getitem__)
                key = (OP_NAMES[rows[last][0]], OP_NAMES[op])
                edges.setdefault(key, []).append((last, row))
    return edges


def phase_spans(ranges: list[Range], table: torch.Tensor) -> dict[str, float]:
    """Mean us per layer, by instruction kind, from the producer phase's last row done to this
    phase's last row done. QKV counts from layer 1, whose producer is the previous layer's
    down; a MoE table's GATEUP counts from its ROUTER."""
    rows = table.tolist()
    producers = PRODUCER
    if any(row[0] == ROUTER for row in rows):
        producers = MOE_PRODUCER
    if any(row[0] == REDUCE for row in rows):
        producers = MOE_BATCH_PRODUCER
    last: dict[tuple[int, int], int] = {}
    for row, item in work_item_ranges(ranges, "compute").items():
        key = tuple(rows[row][:2])  # (op, layer); the LM head's layer is the layer count
        last[key] = max(last.get(key, 0), item.end_ns)
    spans: dict[str, list[int]] = {}
    for (op, layer), end in last.items():
        producer_layer = layer - 1 if op in (QKV, LMHEAD) else layer
        producer = last.get((producers[op], producer_layer))
        if producer is not None:
            spans.setdefault(OP_NAMES[op], []).append(end - producer)
    return {name: sum(v) / len(v) / 1e3 for name, v in spans.items()}


def input_waits(ranges: list[Range], table: torch.Tensor) -> dict[str, float]:
    """Median us, by instruction kind, from a row's start to its inputs ready (the end of its
    first ``poll``; a REDUCE row's ``reduce.poll``). Rows that reuse the previous row's GEMV input
    poll nothing and are not counted."""
    rows = table.tolist()
    starts = {row: item.start_ns for row, item in work_item_ranges(ranges, "compute").items()}
    ready: dict[int, float] = {}
    for item in ranges:
        if item.payload is None:
            continue
        reduce_row = 0 <= item.payload < len(rows) and rows[item.payload][0] == REDUCE
        if item.name == "poll" or (item.name == "reduce.poll" and reduce_row):
            ready[item.payload] = min(ready.get(item.payload, math.inf), item.end_ns)
    waits: dict[str, list[int]] = {}
    for row, end in ready.items():
        if row in starts:
            waits.setdefault(OP_NAMES[rows[row][0]], []).append(end - starts[row])
    return {name: statistics.median(v) / 1e3 for name, v in waits.items()}
