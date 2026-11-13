"""warpdrive piece chain (``piece_chain``): the chain prologue's piece table, work tables and
chunk row table, and the fp32 state chain.

The tables are compared exactly with a Python model (bitonic network included) and the state
chain with an fp64 recurrence; both run on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import math

import pytest
import torch
from test_wd_work_tables import (
    FIELDS,
    SENTINEL,
    chunk_rows_reference,
    cu_of,
    lpt,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive piece chain targets SM100 or SM103",
)


def pc():
    from attn_gym.linear._delta_rule.warpdrive import piece_chain

    return piece_chain


# ---- Python model of the piece table and its work tables ---------------------------------------


def piece_table_reference(cu, pieces, heads, b_t):
    """``(cu_pieces, main_rows, summary_rows)`` of the chain prologue."""
    cu = cu.tolist()
    seqs = len(cu) - 1
    wave = seqs * pieces
    total_chunks = math.ceil((cu[-1] - cu[0]) / b_t)
    span = max(math.ceil(total_chunks / wave), 1)

    def count(length, span):
        return max(math.ceil(math.ceil(length / b_t) / span), 1)

    lens = [cu[b + 1] - cu[b] for b in range(seqs)]
    if seqs > 1 and sum(count(n, span) for n in lens) > wave:
        span = max(math.ceil(total_chunks / (wave - seqs + 1)), 1)
    cu_pieces = [None] * (wave + 1)
    main, summary = [], []
    running_main = running_summary = 0
    for b, length in enumerate(lens):
        chunks = math.ceil(length / b_t)
        own = max(math.ceil(chunks / count(length, span)), 1)
        span_tokens = own * b_t
        filled = math.ceil(length / span_tokens)
        last = filled - 1 if filled > 1 else 0
        main.append(running_main)
        summary.append(running_summary)
        for j in range(last + 1):
            cu_pieces[running_main // heads + j] = cu[b] + min(j * span_tokens, length)
        running_main += (last + 1) * heads
        running_summary += (last + 1) * heads if last > 0 else 0
    for slot in range(running_main // heads, wave + 1):
        cu_pieces[slot] = cu[-1]
    return cu_pieces, main + [running_main], summary + [running_summary]


def chain_items_reference(cu_pieces, row_base, slot_rows, heads, b_t):
    """Work rows of a piece table: ``row_base`` numbers this table's rows, ``slot_rows`` (the
    main table's bases) the flat slots."""
    rows = []
    for item in range(row_base[-1]):
        seq = max(b for b in range(len(row_base) - 1) if row_base[b] <= item)
        slot_in_seq, head = divmod(item - row_base[seq], heads)
        last = (row_base[seq + 1] - row_base[seq]) // heads - 1
        slot = slot_rows[seq] // heads + slot_in_seq
        start, end = cu_pieces[slot], cu_pieces[slot + 1]
        nc = math.ceil((end - start) / b_t)
        final = seq if slot_in_seq == last else -1
        dstate = seq if slot_in_seq == 0 else -1
        rows.append([slot, head, 0, nc, 0, nc, start, end, final, dstate])
    return [rows[p] for p in lpt([r[5] for r in rows])]


def chain_buffers(seqs, pieces, heads):
    """The driver's piece-table workspace (``drivers.plan.chain_buffers`` layout)."""
    layout = pc().piece_table_layout(seqs, pieces, heads)
    table = torch.full((layout.nbytes // 4,), SENTINEL, dtype=torch.int32, device="cuda")

    def words(offset, count):
        return table[offset // 4 : offset // 4 + count]

    items = seqs * pieces * heads
    return table, {
        "cu_pieces": words(layout.cu_pieces, seqs * pieces + 1),
        "main_rows": words(layout.main_rows, seqs + 1),
        "summary_rows": words(layout.summary_rows, seqs + 1),
        "main_count": words(layout.main_count, 1),
        "summary_count": words(layout.summary_count, 1),
        "work_items": torch.full((items, FIELDS), SENTINEL, dtype=torch.int32, device="cuda"),
        "work_items_summary": torch.full(
            (layout.item_rows, FIELDS), SENTINEL, dtype=torch.int32, device="cuda"
        ),
        "scheduler": torch.full((6,), SENTINEL, dtype=torch.int32, device="cuda"),
    }


def run_chain(lens, pieces, heads, b_t, rows_b_t=0, *, summary=True, use_int64_offsets=False):
    cu = cu_of(lens)
    table, buf = chain_buffers(len(lens), pieces, heads)
    if not summary:
        buf["work_items_summary"] = None
    rows = row_count = None
    if rows_b_t:
        rows = torch.full((sum(lens) // rows_b_t + len(lens) * pieces + 2, 4), SENTINEL)
        rows = rows.to(torch.int32).cuda()
        row_count = torch.zeros(1, dtype=torch.int32, device="cuda")
    pc().build_chain_tables(
        cu,
        buf,
        pieces=pieces,
        heads_out=heads,
        b_t=b_t,
        rows=rows,
        row_count=row_count,
        rows_b_t=rows_b_t,
        use_int64_offsets=use_int64_offsets,
    )
    return cu, table, buf, rows, row_count


# (lens, pieces, heads, b_t)
CHAIN_CASES = [
    ([32768], 4, 32, 64),
    ([1, 63, 64, 65], 8, 4, 64),
    ([20000, 0, 3000, 70, 0, 9000], 3, 8, 64),
    ([5000] * 9, 16, 1, 16),
    ([(i * 911) % 3001 for i in range(40)], 3, 2, 16),
    ([(i * 53) % 700 for i in range(1100)], 2, 1, 64),  # more sequences than threads
]


@pytest.mark.parametrize(("lens", "pieces", "heads", "b_t"), CHAIN_CASES)
def test_chain_tables_match_model(lens, pieces, heads, b_t):
    cu, _, buf, rows, row_count = run_chain(lens, pieces, heads, b_t, rows_b_t=b_t)
    cu_pieces, main, summary = piece_table_reference(cu, pieces, heads, b_t)
    assert buf["cu_pieces"].tolist() == cu_pieces
    assert buf["main_rows"].tolist() == main
    assert buf["summary_rows"].tolist() == summary
    n_main, n_summary = main[-1], summary[-1]
    main_items = chain_items_reference(cu_pieces, main, main, heads, b_t)
    summary_items = chain_items_reference(cu_pieces, summary, main, heads, b_t)
    assert buf["work_items"][:n_main].tolist() == main_items
    assert buf["work_items_summary"][:n_summary].tolist() == summary_items
    assert buf["scheduler"].tolist() == [0] * 6
    expect_rows = chunk_rows_reference(torch.tensor(cu_pieces), b_t)
    assert row_count.item() == len(expect_rows)
    assert rows[: len(expect_rows)].tolist() == expect_rows


def test_chain_tables_without_summary_leave_its_table_alone():
    _, _, buf, _, _ = run_chain([20000, 3000], 4, 2, 64, summary=False)
    assert buf["main_rows"][-1].item() > 0
    assert (buf["work_items"][buf["main_rows"][-1].item() :] == SENTINEL).all()


def test_chain_tables_int64_extents_agree():
    """The int64-extent ABI (for oversized tensors) builds the same tables."""
    a = run_chain([20000, 0, 3000], 4, 2, 64, rows_b_t=64)
    b = run_chain([20000, 0, 3000], 4, 2, 64, rows_b_t=64, use_int64_offsets=True)
    assert torch.equal(a[1], b[1]) and torch.equal(a[3], b[3]) and torch.equal(a[4], b[4])
    for key in ("work_items", "work_items_summary", "scheduler"):
        assert torch.equal(a[2][key], b[2][key]), key


def test_chain_tables_reuse_one_compile_across_runtime_shapes():
    """Sequences, pieces, heads and tokens are runtime values."""
    pc().compile_chain_tables.cache_clear()
    cases = [([32768], 4, 32), ([1, 2, 3], 16, 1), ([700] * 300, 2, 3)]
    for lens, pieces, heads in cases:
        run_chain(lens, pieces, heads, 64, rows_b_t=64)
    info = pc().compile_chain_tables.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(cases), info


# ---- state chain -------------------------------------------------------------------------------


def state_case(lens, pieces, heads, dim_v, dim_k, seed_dtype, routed, seed=0):
    cu = cu_of(lens)
    _, buf = chain_buffers(len(lens), pieces, heads)
    pc().build_chain_tables(cu, buf, pieces=pieces, heads_out=heads, b_t=64)
    slots = len(lens) * pieces
    gen = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(slots, heads, dim_v, dim_k, device="cuda", generator=gen)
    # Transitions near a contraction keep the chained states bounded.
    eye = torch.eye(dim_k, device="cuda")
    m = 0.7 * eye + 0.3 * torch.randn(slots, heads, dim_k, dim_k, device="cuda", generator=gen) / (
        dim_k**0.5
    )
    seed_state = indices = None
    if seed_dtype is not None:
        rows = len(lens) + 2 if routed else len(lens)
        seed_state = torch.randn(rows, heads, dim_v, dim_k, device="cuda", generator=gen)
        seed_state = seed_state.to(seed_dtype)
        if routed:
            indices = torch.randperm(rows, device="cuda", generator=gen)[: len(lens)]
            indices = indices.to(torch.int32)
    return buf["main_rows"], h, m, seed_state, indices


def state_chain_reference(main_rows, h, m, seed_state, indices, transpose):
    heads = h.shape[1]
    main_rows = main_rows.tolist()
    x = torch.full_like(h, float("nan"), dtype=torch.float64)
    for seq in range(len(main_rows) - 1):
        base = main_rows[seq] // heads
        count = max((main_rows[seq + 1] - main_rows[seq]) // heads, 1)
        if seed_state is None:
            cur = torch.zeros_like(h[0], dtype=torch.float64)
        else:
            cur = seed_state[seq if indices is None else indices[seq].item()].double()
        order = range(count) if not transpose else reversed(range(count))
        for step in order:
            slot = base + step
            x[slot] = cur
            if count > 1:
                mj = m[slot].double()
                cur = cur @ (mj.transpose(-1, -2) if transpose else mj) + h[slot].double()
    return x


def run_state_chain(main_rows, h, m, seed_state, indices, transpose, rows=None):
    x = torch.full_like(h, float("nan"))
    pc().state_chain(
        h, m, x, main_rows, seed=seed_state, seed_indices=indices, transpose=transpose, rows=rows
    )
    return x


# (lens, pieces, heads, dim_v, dim_k)
STATE_CASES = [
    ([32768], 4, 2, 128, 128),
    ([20000, 0, 3000, 70, 9000], 3, 3, 64, 128),
    ([4096, 4096], 6, 2, 128, 64),
    ([100, 5000, 64], 5, 1, 64, 64),
]


@pytest.mark.parametrize(("lens", "pieces", "heads", "dim_v", "dim_k"), STATE_CASES)
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize(
    ("seed_dtype", "routed"),
    [(None, False), (torch.float32, False), (torch.bfloat16, True)],
)
def test_state_chain_matches_fp64(
    lens, pieces, heads, dim_v, dim_k, transpose, seed_dtype, routed
):
    main_rows, h, m, seed_state, indices = state_case(
        lens, pieces, heads, dim_v, dim_k, seed_dtype, routed
    )
    ref = state_chain_reference(main_rows, h, m, seed_state, indices, transpose)
    filled = ~torch.isnan(ref[:, 0, 0, 0])
    for rows in {r for r in (dim_v // 16, dim_v // 8) if r >= pc().CHAIN_WARPS}:
        x = run_state_chain(main_rows, h, m, seed_state, indices, transpose, rows)
        torch.testing.assert_close(x[filled].double(), ref[filled], atol=1e-4, rtol=1e-4)
        assert torch.isnan(x[~filled]).all()


def test_state_chain_reuses_one_compile_across_runtime_shapes():
    pc().compile_state_chain.cache_clear()
    cases = [([32768], 4, 2), ([1, 2, 3, 4], 3, 5), ([900] * 20, 2, 1)]
    for lens, pieces, heads in cases:
        args = state_case(lens, pieces, heads, 128, 128, torch.float32, False)
        run_state_chain(*args, transpose=False, rows=8)
    info = pc().compile_state_chain.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(cases), info
