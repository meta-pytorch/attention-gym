"""Where a Qwen megakernel step spends its time: a globaltimer trace decoded per table row.

``QwenMegakernel(..., trace="globaltimer")`` records, in every CTA, each work item of the
compute and load roles (payload = its table row) and the compute role's ``trace_region``s, with
the row as payload: ``poll`` (waiting for the row's inputs; it ends when they are ready),
``stream`` (a GEMV tile's weight stages), ``attn.blocks`` (an attention row's KV blocks) and
``reduce.poll`` (the last down split fetching the others' partials)::

    with trace_session(runner, trace_path="step.pftrace") as session:
        runner(weights, token, pos, *caches)  # one launch
    phase_spans(session.ranges, runner.table)  # {"QKV": 1.9, ...} us per layer

The Perfetto trace has one process per CTA (SM) and one thread per role; work items are named
after their instruction (``QKV``, ``load QKV``, ...) with ``layer``/``a``/``b`` in their args,
and an arrow runs from each input group's last-finishing producer row to the row that waited
for it, across CTAs. This module is host-only; the kernel module does not import it.
"""

from __future__ import annotations

import math
import statistics
from contextlib import contextmanager

import torch

from attn_gym.profiler import Range
from attn_gym.warpdrive.examples.qwen_megakernel import (
    ATTN,
    DOWN,
    GATEUP,
    LAYERS,
    LMHEAD,
    OP_NAMES,
    OPROJ,
    QKV,
    row_groups,
)
from attn_gym.warpdrive.trace_session import trace_session as kernel_session
from attn_gym.warpdrive.trace_session import write_trace

# The instruction kind each phase waits for (QKV: the previous layer's down; the LM head: the
# last layer's).
PRODUCER = {QKV: DOWN, ATTN: QKV, OPROJ: ATTN, GATEUP: OPROJ, DOWN: GATEUP, LMHEAD: DOWN}


@contextmanager
def trace_session(runner, *, trace_path=None, device=None):
    """``trace_session.trace_session`` for one launch of a ``trace="globaltimer"`` runner: the
    runner records into the session's buffer inside the block. ``trace_path``: also write the
    step's Perfetto trace (``write_step_trace``)."""
    with kernel_session(runner.op, runner.max_ctas, device=device) as session:
        previous, runner.trace_buffer = runner.trace_buffer, session.prof.tensor
        try:
            yield session
        finally:
            runner.trace_buffer = previous
    if trace_path is not None:
        write_step_trace(trace_path, runner, session.ranges)


def write_step_trace(path, runner, ranges: list[Range]):
    """The step's trace: each work item named after its instruction, with its row's words in
    its args, and an arrow from the last-finishing producer row of every group a row waits
    for to that row."""
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

    computed = item_ranges(ranges)
    done = {row: item.end_ns for row, item in computed.items()}
    flows = [
        (computed[source], computed[target])
        for edges in input_edges(rows, done).values()
        for source, target in edges
    ]
    return write_trace(path, runner.op, ranges, name=name, args=args, flows=flows)


def input_edges(rows: list[list[int]], done: dict[int, float]) -> dict[tuple, list]:
    """``{(producer name, consumer name): [(producer row, consumer row)]}``: for every row and
    each group it waits for, the group's row that finished last (``done``: row -> end time)."""
    groups: dict[tuple, list[int]] = {}
    for row, (op, layer, a, b, _) in enumerate(rows):
        if op >= 0:
            groups.setdefault(row_groups(op, layer, a, b)[1], []).append(row)
    edges: dict[tuple, list] = {}
    for row, (op, layer, a, b, _) in enumerate(rows):
        if op < 0 or row not in done:
            continue
        for group in row_groups(op, layer, a, b)[0]:
            finished = [r for r in groups.get(group, ()) if r in done]
            if finished:
                last = max(finished, key=done.__getitem__)
                key = (OP_NAMES[rows[last][0]], OP_NAMES[op])
                edges.setdefault(key, []).append((last, row))
    return edges


def item_ranges(ranges: list[Range], role: str = "compute") -> dict[int, Range]:
    """Each table row's work-item range of ``role``, by row."""
    return {r.payload: r for r in ranges if r.name == role and r.payload is not None}


def phase_spans(ranges: list[Range], table: torch.Tensor) -> dict[str, float]:
    """Mean us per layer of each instruction kind from its producer phase's last row done to its
    own last row done (QKV from layer 1, whose producer is the previous layer's down)."""
    rows = table.tolist()
    last: dict[tuple[int, int], int] = {}
    for row, item in item_ranges(ranges).items():
        op, layer = rows[row][:2]
        key = (op, LAYERS if op == LMHEAD else layer)
        last[key] = max(last.get(key, 0), item.end_ns)
    spans: dict[str, list[int]] = {}
    for (op, layer), end in last.items():
        producer_layer = layer - 1 if op in (QKV, LMHEAD) else layer
        producer = last.get((PRODUCER[op], producer_layer))
        if producer is not None:
            spans.setdefault(OP_NAMES[op], []).append(end - producer)
    return {name: sum(v) / len(v) / 1e3 for name, v in spans.items()}


def input_waits(ranges: list[Range], table: torch.Tensor) -> dict[str, float]:
    """Median us per instruction kind from a row's start to its inputs being ready (the end of
    its first ``poll``). Rows that reuse the previous row's GEMV input poll nothing."""
    rows = table.tolist()
    starts = {row: item.start_ns for row, item in item_ranges(ranges).items()}
    ready: dict[int, float] = {}
    for item in ranges:
        if item.name == "poll" and item.payload is not None:
            ready[item.payload] = min(ready.get(item.payload, math.inf), item.end_ns)
    waits: dict[str, list[int]] = {}
    for row, end in ready.items():
        if row in starts:
            waits.setdefault(OP_NAMES[rows[row][0]], []).append(end - starts[row])
    return {name: statistics.median(v) / 1e3 for name, v in waits.items()}
