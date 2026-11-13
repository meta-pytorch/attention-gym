"""Host side of the Qwen3 decode megakernel's instruction table: the rows of a step, their
dependency groups, the row-cost model and the static list-schedule assignment to CTAs."""

import collections
import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from attn_gym.warpdrive import StaticPersistent
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    ALL_OPS,
    ATTN,
    BLOCK,
    DOWN,
    DOWN_SPLITS,
    EXPERT_GROUPS,
    GATEUP,
    HEAD_DIM,
    LM_TILES,
    LMHEAD,
    NORM,
    O_SPLITS,
    OP_NAMES,
    OPROJ,
    QKV,
    REDUCE,
    ROUTER,
    WORDS,
    QwenDecodeMegakernel,
)
from attn_gym.warpdrive.examples.qwen_megakernel.tc import QwenDecodeMegakernelTc


# Static-assignment latency model, in us. Ordinary rows cost ``base`` plus ``per_token`` for
# each token after the first, plus the cost of their 16 KB weight stages. Stage time is
# max(stage_min, stage_hbm * min(concurrent_rows, ctas) / ctas), multiplied by
# 1 + ring * max(0, 8 - stages). Attention rows cost base[ATTN] plus ``attn_kv`` per KV block
# of their split at CONTEXT tokens. Batched MoE expert rows use the expected cost under uniform
# routing, with skipped rows costing zero. Fitted to globaltimer captures of compute work items,
# excluding polls (GB200, ctx 1024: Qwen3-0.6B to 14B and 30B-A3B; CUDA cores at batch one,
# tcgen05 at batches 1 to 64), predicting step times within ~15% on those captures.
# Producer waits add DEPENDENCY_LATENCY; batch-one rows reusing their CTA's input, REUSED_INPUT.
@dataclass(frozen=True)
class RowCostModel:
    base: dict[int, float]
    per_token: dict[int, float]
    stage_hbm: float
    stage_min: float
    ring: float
    attn_kv: float
    moe_gateup: float  # batch-one MoE gate/up: each consumer routes and gathers expert weights
    expert_gateup: float  # batched MoE: an active expert's gate/up row


ROW_COST_MODELS = {
    "cuda_core": RowCostModel(
        base={QKV: 1.09, ATTN: 1.53, OPROJ: 1.10, GATEUP: 0.57, DOWN: 0.34, LMHEAD: 1.98,
              ROUTER: 1.17, REDUCE: 0.0, NORM: 0.0},
        per_token={},
        stage_hbm=0.0, stage_min=0.27, ring=0.0, attn_kv=1.47, moe_gateup=3.18, expert_gateup=0.0,
    ),
    "tcgen05": RowCostModel(
        base={QKV: 1.02, ATTN: 2.53, OPROJ: 1.23, GATEUP: 1.58, DOWN: 0.48, LMHEAD: 3.17,
              ROUTER: 1.19, REDUCE: 0.16, NORM: 2.0},
        per_token={QKV: 0.44, OPROJ: 0.34, GATEUP: 0.37, DOWN: 0.20, LMHEAD: 1.29, ROUTER: 1.87,
                   REDUCE: 0.06},
        stage_hbm=0.30, stage_min=0.19, ring=0.07, attn_kv=0.83, moe_gateup=8.68,
        expert_gateup=22.06,
    ),
}  # fmt: skip
CONTEXT = 1024  # the context the table is balanced for: one table serves every length
DEPENDENCY_LATENCY = 1.0
# A batch-one GEMV row whose input its CTA's previous row already wrote skips the prologue
# (including MoE routing, so moe_gateup): it costs its weight stages plus REUSED_INPUT.
REUSED_INPUT = 0.2
GEMV_KINDS = (QKV, OPROJ, GATEUP, DOWN, ROUTER, LMHEAD)


def weight_stages(op: QwenDecodeMegakernel) -> dict[int, int]:
    """16 KB weight stages one row of each kind streams; attention's depend on the context."""
    k = op.hidden_k_tiles
    return {
        QKV: k,
        ATTN: 0,
        OPROJ: op.oproj_k_tiles,
        GATEUP: k,
        DOWN: op.down_k_tiles,
        LMHEAD: LM_TILES * k,
        ROUTER: k,
        REDUCE: 0,
        NORM: 0,
        **(
            {kind: math.prod(op.expert_tiles(kind)) for kind in (GATEUP, DOWN)}
            if op.moe_batched
            else {}
        ),
    }


def row_cost(op: QwenDecodeMegakernel, ctas: int) -> dict[int, float]:
    """Each row kind's cost under ``ROW_COST_MODELS`` on ``ctas`` CTAs."""
    return row_costs(op, ctas)[0]


def row_costs(op: QwenDecodeMegakernel, ctas: int) -> tuple[dict[int, float], dict[int, float]]:
    """``row_cost``, and each kind's cost when the row reuses its CTA's GEMV input (batch one)."""
    model = ROW_COST_MODELS["tcgen05" if isinstance(op, QwenDecodeMegakernelTc) else "cuda_core"]
    stages, batch = weight_stages(op), op.batch
    concurrent = collections.Counter(
        words[0] for words, _, _ in instruction_rows(op) if words[1] == 0 or words[0] == LMHEAD
    )  # rows per layer of each kind
    ring = 1 + model.ring * max(0, 8 - op.stages)

    def streaming(kind, rows):
        share = model.stage_hbm * min(rows, ctas) / ctas
        return stages[kind] * max(model.stage_min, share) * ring

    blocks = math.ceil(math.ceil(CONTEXT / BLOCK) / op.splits)  # per attention split
    cost, reused = {}, {}
    for kind in range(len(OP_NAMES)):
        extra = model.per_token.get(kind, 0.0) * (batch - 1)
        if kind == ATTN:
            cost[kind] = model.base[ATTN] + model.attn_kv * blocks
        elif op.moe_batched and kind in (GATEUP, DOWN):
            active = 1 - (1 - op.top_k / op.experts) ** batch  # an expert gets a token
            row = streaming(kind, active * concurrent[kind])
            cost[kind] = active * (row + (model.expert_gateup if kind == GATEUP else 0.0))
        else:
            cost[kind] = model.base[kind] + extra + streaming(kind, concurrent[kind])
            reused[kind] = REUSED_INPUT + streaming(kind, concurrent[kind])
            if op.moe and kind == GATEUP:
                cost[kind] += model.moe_gateup
    return cost, reused


def input_key(op: QwenDecodeMegakernel, words: list[int]) -> tuple[tuple | None, bool]:
    """``(key, fresh)``: the GEMV input a batch-one row reads (``next_input``'s key; None outside
    batch-one GEMV rows), and whether it writes that input even when its CTA holds it (row 0 of
    layer 0's QKV publishes h_in[0], with ``sums_parts`` row 0 of the h_mid op publishes h_mid).
    A row reuses the input of its CTA's previous GEMV row with the same key."""
    kind, layer, a, b = words
    if op.batch > 1 or op.streams_inputs or kind not in GEMV_KINDS:
        return None, True
    mid = ROUTER if op.moe else GATEUP
    fresh = a == 0 and ((kind == QKV and layer == 0) or (op.sums_parts and kind == mid))
    return (kind, layer, b if kind in (OPROJ, DOWN) else 0), fresh


# One instruction: table words (op, layer, a, b), the producer groups it waits for, and the
# group its completion counts toward (``row_groups``).
Row = tuple[list[int], list[tuple], tuple]


def instruction_rows(op: QwenDecodeMegakernel, ops: tuple[int, ...] = ALL_OPS) -> list[Row]:
    """The step's instructions in dependency order. ``ops`` keeps a subset (single-instruction
    tests)."""
    rows = []

    def add(kind, layer, a, b):
        if kind in ops:
            rows.append(([kind, layer, a, b], *row_groups(op, kind, layer, a, b)))

    for layer in range(op.layers):
        if op.streams_inputs:  # NORM rows: one token each
            for t in range(op.batch):
                add(NORM, layer, 0, t)
        for t in range(op.qkv_rows // op.rows):
            add(QKV, layer, t, 0)
        for token in range(op.batch):
            for group in range(op.attn_groups):
                for split in range(op.splits):
                    add(ATTN, layer, group, token * op.splits + split)
        for ks in range(O_SPLITS):
            for t in range(op.hidden // op.rows):
                add(OPROJ, layer, t, ks)
        if op.moe:
            for t in range(op.experts // op.rows):
                add(ROUTER, layer, t, 0)
        if op.moe_batched:  # EXPERT_GROUPS gate/up and down rows per expert, then the reduce
            for expert in range(op.experts):
                for group in range(EXPERT_GROUPS):
                    add(GATEUP, layer, expert, group)
            for expert in range(op.experts):
                for group in range(EXPERT_GROUPS):
                    add(DOWN, layer, expert, group)
            for t in range(op.hidden // op.rows):
                add(REDUCE, layer, t, 0)
            continue
        if op.shares_mlp:
            for t in range(op.batch):
                add(NORM, layer, 1, t)
        for t in range(2 * op.inter // op.rows):
            add(GATEUP, layer, t, 0)
        for ks in range(DOWN_SPLITS):
            for t in range(op.hidden // op.rows):
                add(DOWN, layer, t, ks)
    # The LM head's layer word is ``layers``: it reads h_in[layers].
    lm_rows = op.vocab // op.rows
    if op.streams_inputs:
        for t in range(op.batch):
            add(NORM, op.layers, 0, t)
    for t in range(0, lm_rows, LM_TILES):
        add(LMHEAD, op.layers, t, min(LM_TILES, lm_rows - t))
    return rows


def build_table(
    op: QwenDecodeMegakernel, ctas: int, ops: tuple[int, ...] = ALL_OPS
) -> torch.Tensor:
    """The ``instruction_rows`` table: in dependency order for the atomic schedule, assigned
    to ``ctas`` CTAs for the static one."""
    rows = instruction_rows(op, ops)
    if isinstance(op.work.schedule, StaticPersistent):
        table = assign_static(op, rows, ctas)
    else:
        table = [words for words, _, _ in rows]
    return torch.tensor([words + [i] for i, words in enumerate(table)], dtype=torch.int32)


def row_groups(
    op: QwenDecodeMegakernel, kind: int, layer: int, a: int, b: int
) -> tuple[list[tuple], tuple]:
    """The producer groups instruction ``(kind, layer, a, b)`` waits for, and the group its
    completion counts toward. A group is done when all of its rows are. Rows reading shared
    normalized inputs wait for NORM rows instead of the residual's producers."""
    deps, group = _row_groups(op, kind, layer, a, b)
    if (op.streams_inputs and kind in (QKV, LMHEAD)) or (op.shares_mlp and kind == GATEUP):
        deps = [("norm", layer, int(kind == GATEUP))]
    return deps, group


def _row_groups(op: QwenDecodeMegakernel, kind: int, layer: int, a: int, b: int):
    residual = "reduce" if op.moe_batched else "down"  # who publishes h_in[layer + 1]
    if kind == NORM:  # a: 0 normalizes h_in[layer] (the LM head's at layer == layers), 1 h_mid
        deps = [("oproj", layer)] if a else [(residual, layer - 1)] if layer else []
        return deps, ("norm", layer, a)
    if kind == QKV:
        head = a * op.rows // HEAD_DIM
        if head < op.q_heads:
            kvh = head // op.group_size
        else:
            kvh = (head - op.q_heads) % op.kv_heads
        return [(residual, layer - 1)] if layer else [], ("qkv", layer, kvh)
    if kind == ATTN:
        return [("qkv", layer, op.kv_head(a))], ("attn", layer, a)
    if kind == OPROJ:
        first = b * op.heads_per_split // op.per_row
        last = ((b + 1) * op.heads_per_split - 1) // op.per_row
        return [("attn", layer, g) for g in range(first, last + 1)], ("oproj", layer)
    if kind == ROUTER:
        return [("oproj", layer)], ("router", layer)
    if op.moe_batched and kind in (GATEUP, DOWN, REDUCE):
        producer = {GATEUP: "router", DOWN: "gateup", REDUCE: "down"}[kind]
        return [(producer, layer)], (OP_NAMES[kind].lower(), layer)
    if kind == GATEUP:
        # MoE expert rows read h_mid (o_proj's last split publishes it) and the router's logits.
        per_split = 2 * op.inter // op.rows // DOWN_SPLITS
        producer = ("router", layer) if op.moe else ("oproj", layer)
        return [producer], ("gateup", layer, a // per_split)
    if kind == DOWN:
        return [("gateup", layer, b)], ("down", layer)
    return [(residual, layer - 1)], ("lmhead",)


class RowSpan(NamedTuple):
    """Where and when the ``row_costs`` model runs a row (us from launch), and what it waited on."""

    cta: int
    start: float
    end: float
    gates: tuple[int, ...]  # the last-finishing row of each producer group it waits for
    after: int | None  # the row that set its start: its latest gate, or its CTA's previous row


def list_schedule(op: QwenDecodeMegakernel, rows: list[Row], ctas: int) -> list[RowSpan]:
    """Greedy list schedule of dependency-ordered rows onto ``ctas`` CTAs under ``op``'s
    ``row_costs``: each row goes to the CTA that finishes it first, starting once every producer
    group it waits for has finished plus ``DEPENDENCY_LATENCY``. A row costs less on the CTA whose
    previous GEMV row wrote the same input (``input_key``)."""
    cost, reused = row_costs(op, ctas)
    last: dict[tuple, int] = {}  # producer group -> its last-finishing row so far
    tail: list[int | None] = [None] * ctas  # each CTA's previous row
    free = [0.0] * ctas  # when each CTA's previous row ends
    held: list[tuple | None] = [None] * ctas  # each CTA's GEMV input key
    spans: list[RowSpan] = []
    for i, (words, deps, group) in enumerate(rows):
        gates = tuple(last[d] for d in deps if d in last)
        gate = max(gates, key=lambda g: spans[g].end, default=None)
        ready = spans[gate].end + DEPENDENCY_LATENCY if gate is not None else 0.0
        key, fresh = input_key(op, words)
        costs = [
            reused[words[0]]
            if key is not None and not fresh and held[c] == key
            else cost[words[0]]
            for c in range(ctas)
        ]
        cta = min(range(ctas), key=lambda c: (max(free[c], ready) + costs[c], c))
        start = max(free[cta], ready)
        after = gate if gate is not None and ready >= free[cta] else tail[cta]
        spans.append(RowSpan(cta, start, start + costs[cta], gates, after))
        tail[cta], free[cta] = i, spans[i].end
        if words[0] in GEMV_KINDS:
            held[cta] = key
        if group not in last or spans[i].end >= spans[last[group]].end:
            last[group] = i
    return spans


def assign_static(op: QwenDecodeMegakernel, rows: list[Row], ctas: int) -> list[list[int]]:
    """``list_schedule``'s per-CTA row lists laid out ``[slot][cta]`` for ``StaticPersistent``
    (CTA b runs rows b, b + ctas, ...); short lists are padded with dropped rows.

    Each list keeps the dependency order, so with every CTA resident the earliest unfinished
    row always has its producers finished: no deadlock."""
    lists: list[list[list[int]]] = [[] for _ in range(ctas)]
    for (words, _, _), span in zip(rows, list_schedule(op, rows, ctas)):
        lists[span.cta].append(words)
    slots = max(len(rows) for rows in lists)
    padding = [-1] * (WORDS - 1)
    return [
        lists[cta][slot] if slot < len(lists[cta]) else padding
        for slot in range(slots)
        for cta in range(ctas)
    ]
