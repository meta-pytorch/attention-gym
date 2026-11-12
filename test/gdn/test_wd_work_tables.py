"""warpdrive work-item tables (``work_tables``): the uncut and staged LPT order, the chunk row
table and the split-K planner.

The order and row tables are compared exactly with a Python model of the algorithm (the
bitonic network included, since ties keep its order); the split planner's rows are checked
against the forgetting-horizon contract from fp64 gate sums. Both run on the ``[linear]``
CuTeDSL floor.
"""

from __future__ import annotations

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive work tables target SM100 or SM103",
)

FIELDS = 10
CAPACITY = 4096
INT32_MIN = -(2**31)
SENTINEL = -7


def wt():
    from attn_gym.linear._delta_rule.warpdrive import work_tables

    return work_tables


def cu_of(lens):
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()


def num_sms():
    return torch.cuda.get_device_properties(0).multi_processor_count


# ---- Python model of the order phase -----------------------------------------------------------


def bitonic_order(keys: list[int]) -> list[int]:
    """Positions of ``keys`` after the kernel's descending bitonic network (ties keep the
    network's order, not a stable one)."""
    n = len(keys)
    pad = 1 << max(0, (n - 1).bit_length())
    key = list(keys) + [INT32_MIN] * (pad - n)
    idx = list(range(pad))
    k = 2
    while k <= pad:
        j = k // 2
        while j > 0:
            for i in range(pad):
                partner = i ^ j
                if partner > i:
                    up = (i // k) % 2 == 0
                    if (up and key[i] < key[partner]) or (not up and key[i] > key[partner]):
                        key[i], key[partner] = key[partner], key[i]
                        idx[i], idx[partner] = idx[partner], idx[i]
            j //= 2
        k *= 2
    return idx[:n]


def lpt(keys: list[int], num_ctas: int = 0) -> list[int]:
    """The order phase's permutation of rows with sort ``keys``."""
    n = len(keys)
    if n > CAPACITY or n <= num_ctas or len(set(keys)) <= 1:
        return list(range(n))
    return bitonic_order(keys)


def uncut_reference(cu, heads, b_t, *, num_ctas=0, routes=None, seeds=None):
    """``(count, rows)`` of the uncut table."""
    cu = cu.tolist()
    n_batch = len(cu) - 1
    if n_batch <= CAPACITY:
        kept = [
            b
            for b in range(n_batch)
            if cu[b + 1] > cu[b] or (routes is not None and routes[b] > 0 and seeds[b] == 0)
        ]
    else:
        kept = list(range(n_batch))
    sources = [kept[i // heads] * heads + i % heads for i in range(len(kept) * heads)]

    def chunks(src):
        b = src // heads
        return math.ceil((cu[b + 1] - cu[b]) / b_t)

    rows = []
    for p in lpt([chunks(s) for s in sources], num_ctas):
        src = sources[p]
        b, h = divmod(src, heads)
        nc = chunks(src)
        rows.append([b, h, 0, nc, 0, nc, cu[b], cu[b + 1], b, b])
    return len(sources), rows


def staged_reference(staging, count, num_ctas=0):
    rows = staging[:count].tolist()
    return [rows[p] for p in lpt([r[5] - r[4] for r in rows], num_ctas)]


def chunk_rows_reference(cu, b_t):
    cu = cu.tolist()
    rows = []
    for b in range(len(cu) - 1):
        for c in range(math.ceil((cu[b + 1] - cu[b]) / b_t)):
            rows.append([c, b, cu[b], cu[b + 1]])
    return rows


def table(rows):
    return torch.full((rows, FIELDS), SENTINEL, dtype=torch.int32, device="cuda")


def run_uncut(cu, heads, b_t, **kwargs):
    items = table(max(1, (cu.numel() - 1) * heads))
    count = torch.full((1,), SENTINEL, dtype=torch.int32, device="cuda")
    sched = torch.full((4,), SENTINEL, dtype=torch.int32, device="cuda")
    wt().build_work_table(cu, items, count, n_heads_out=heads, b_t=b_t, scheduler=sched, **kwargs)
    return items, count, sched


UNCUT_CASES = [
    ([1], 1, 64),
    ([63, 64, 65], 4, 64),
    ([0, 300, 0, 4096, 17, 0], 32, 64),
    ([5, 0, 0, 77, 1000, 16, 15], 3, 16),
    ([0, 0, 0], 2, 64),
    ([64] * 37, 8, 64),  # every key equal: copied through in sequence order
    ([(i * 37) % 501 for i in range(1500)], 2, 64),  # compaction over two ballot rounds
    ([(i * 7) % 3 for i in range(4100)], 1, 16),  # past the sort capacity: uncompacted
]


@pytest.mark.parametrize(("lens", "heads", "b_t"), UNCUT_CASES)
def test_uncut_table_matches_model(lens, heads, b_t):
    cu = cu_of(lens)
    items, count, sched = run_uncut(cu, heads, b_t)
    n, rows = uncut_reference(cu, heads, b_t)
    assert count.item() == n
    assert items[:n].tolist() == rows
    assert (items[n:] == SENTINEL).all()
    assert sched.tolist() == [0] * 4


def test_uncut_table_copies_through_up_to_one_item_per_cta():
    """``num_ctas`` rows or fewer run one per CTA, so they keep sequence order unsorted."""
    cu = cu_of([5, 300, 70, 1000])
    items, count, _ = run_uncut(cu, 4, 16, num_ctas=16)
    n, rows = uncut_reference(cu, 4, 16, num_ctas=16)
    assert count.item() == n == 16
    assert items[:n].tolist() == rows
    assert rows[0][0] == 0  # unsorted


def test_paged_uncut_table_keeps_fresh_empty_routes():
    """An empty sequence stays in the table only when its paged slot is fresh (to clear it)."""
    lens = [0, 0, 0, 0, 16, 0]
    routes = torch.tensor([1, 2, 0, -1, 3, 4], dtype=torch.int32, device="cuda")
    seeds = torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.uint8, device="cuda")
    cu = cu_of(lens)
    items, count, _ = run_uncut(cu, 2, 16, state_indices=routes, has_initial_state=seeds)
    n, rows = uncut_reference(cu, 2, 16, routes=routes.tolist(), seeds=seeds.tolist())
    assert n == 6 and count.item() == n
    assert items[:n].tolist() == rows
    assert sorted({r[0] for r in rows}) == [0, 4, 5]


def test_staged_table_orders_split_rows():
    """Staged rows (the split walk's) sort by warmed span, longest first."""
    gen = torch.Generator().manual_seed(0)
    n = 700
    start = torch.randint(0, 50, (n,), generator=gen)
    span = torch.randint(1, 40, (n,), generator=gen)
    staging = torch.zeros(n + 5, FIELDS, dtype=torch.int32)
    staging[:n, 0] = torch.arange(n)
    staging[:n, 4] = start
    staging[:n, 5] = start + span
    staging[:n, 6:] = torch.randint(-1, 9, (n, 4), generator=gen)
    staging = staging.cuda()
    count = torch.tensor([n], dtype=torch.int32, device="cuda")
    items = table(n + 5)
    sched = torch.full((2,), SENTINEL, dtype=torch.int32, device="cuda")
    wt().build_work_table(
        cu_of([1]), items, count, n_heads_out=1, b_t=64, staging=staging, scheduler=sched
    )
    assert count.item() == n
    assert items[:n].tolist() == staged_reference(staging, n)
    assert sched.tolist() == [0, 0]


@pytest.mark.parametrize("b_t", [64, 16])
def test_chunk_row_table_matches_model(b_t):
    lens = [1, 0, 63, 64, 65, 3000, 0, *([17] * 40)]
    cu = cu_of(lens)
    expect = chunk_rows_reference(cu, b_t)
    rows = torch.full((len(expect) + 3, 4), SENTINEL, dtype=torch.int32, device="cuda")
    row_count = torch.zeros(1, dtype=torch.int32, device="cuda")
    items, _, _ = run_uncut(cu, 2, b_t, rows=rows, row_count=row_count, rows_b_t=b_t)
    assert row_count.item() == len(expect)
    assert rows[: len(expect)].tolist() == expect
    assert (rows[len(expect) :] == SENTINEL).all()
    n, ref = uncut_reference(cu, 2, b_t)
    assert items[:n].tolist() == ref


def test_uncut_table_int64_extents_agree():
    """The int64-extent ABI (for oversized tensors) builds the same table."""
    cu = cu_of([0, 300, 0, 4096, 17])
    a = run_uncut(cu, 4, 64)
    b = run_uncut(cu, 4, 64, use_int64_offsets=True)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_work_table_reuses_one_compile_across_runtime_shapes():
    """Sequence count, head count, token count and table sizes are runtime values."""
    wt().compile_work_table.cache_clear()
    cases = [([1], 1), ([64, 0, 65], 7), ([3] * 2000, 2), ([4096, 1], 64)]
    for lens, heads in cases:
        run_uncut(cu_of(lens), heads, 64)
    info = wt().compile_work_table.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(cases), info


# ---- split planner -----------------------------------------------------------------------------


def split_case(lens, heads, channels, gate_dtype, decay, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    shape = (sum(lens), heads, channels) if channels else (sum(lens), heads)
    gate = (-decay * torch.rand(shape, device="cuda", generator=gen)).to(gate_dtype)
    return gate, cu_of(lens)


def run_split(gate, cu, b_t, *, use_int64_offsets=False):
    w = wt()
    batch, heads = cu.numel() - 1, gate.shape[1]
    tokens = gate.shape[0]
    ideal = w.compute_ideal_chunks(tokens, heads, num_sms(), b_t)
    rows = w.max_work_items(tokens, batch, heads, ideal, b_t, num_sms())
    staging = table(rows)
    scratch = torch.full(
        (w.chunk_scratch_rows(tokens, batch, b_t), heads), float("nan"), device="cuda"
    )
    count = torch.full((1,), SENTINEL, dtype=torch.int32, device="cuda")
    facts = w.build_split_table(
        gate,
        cu,
        scratch,
        staging,
        count,
        n_tiles=batch * heads,
        ideal_chunks=ideal,
        num_sms=num_sms(),
        b_t=b_t,
        use_int64_offsets=use_int64_offsets,
    )
    return staging, count, scratch, facts


def chunk_values(gate, cu, b_t):
    """fp64 per-chunk clamped log2 decay ``[HO]`` per (sequence, chunk): the channel max of
    the per-channel sums for channel gates."""
    log2 = torch.clamp(gate.double() / math.log(2.0), max=0.0)
    values = {}
    cu = cu.tolist()
    for b in range(len(cu) - 1):
        for c in range(math.ceil((cu[b + 1] - cu[b]) / b_t)):
            s = log2[cu[b] + c * b_t : min(cu[b + 1], cu[b] + (c + 1) * b_t)].sum(0)
            values[b, c] = s.amax(-1) if gate.ndim == 3 else s
    return values


def check_split_contract(staging, count, scratch, gate, cu, b_t, facts):
    """Rows tile every (sequence, head) exactly; each warmup saturates the forgetting horizon;
    scanned chunk values match the fp64 sums."""
    n = count.item()
    rows = staging[:n].tolist()
    values = chunk_values(gate, cu, b_t)
    cu_list = cu.tolist()
    thresh = facts.log2_threshold
    by_tile = {}
    for r in rows:
        by_tile.setdefault((r[0], r[1]), []).append(r)
    expect_tiles = {
        (b, h)
        for b in range(len(cu_list) - 1)
        for h in range(gate.shape[1])
        if cu_list[b + 1] > cu_list[b]
    }
    assert set(by_tile) == expect_tiles
    for (b, h), items in by_tile.items():
        nc = math.ceil((cu_list[b + 1] - cu_list[b]) / b_t)
        items.sort(key=lambda r: r[2])
        assert [r[2] for r in items] == [0] + [r[3] for r in items[:-1]]
        assert items[-1][3] == nc
        for r in items:
            write_start, write_end, compute_start, compute_end = r[2:6]
            assert r[6:8] == [cu_list[b], cu_list[b + 1]]
            assert 0 <= compute_start <= write_start < write_end <= compute_end <= nc
            assert r[8] == (b if write_end == nc else -1)
            assert r[9] == (b if write_start == 0 else -1)
            before = sum(values[b, c][h].item() for c in range(compute_start, write_start))
            after = sum(values[b, c][h].item() for c in range(write_end, compute_end))
            if write_start > 0:
                assert before <= thresh + 1e-3
            if write_end < nc:
                assert after <= thresh + 1e-3
    written = ~torch.isnan(scratch[:-1])
    for b in range(len(cu_list) - 1):
        base = cu_list[b] // b_t + b
        for c in range(math.ceil((cu_list[b + 1] - cu_list[b]) / b_t)):
            mask = written[base + c]
            if mask.any():
                torch.testing.assert_close(
                    scratch[base + c][mask].double(), values[b, c][mask], atol=1e-4, rtol=1e-5
                )
    return len(by_tile) < n  # whether anything was cut


# (lens, heads, channels, gate dtype, decay, b_t)
SPLIT_CASES = [
    ([32768], 4, 0, torch.float32, 0.05, 64),
    ([32768], 4, 0, torch.float32, 0.0005, 64),  # too little decay: nothing saturates
    ([20000, 64, 3000, 0, 65, 9000], 2, 0, torch.float32, 0.1, 64),
    ([16384, 1], 1, 128, torch.float32, 0.3, 16),
    ([16384], 32, 64, torch.float32, 0.3, 16),  # spans past the scan window: windowed reads
    ([8192, 4000, 17], 2, 64, torch.float32, 0.3, 16),
    ([12000], 2, 128, torch.bfloat16, 0.3, 16),
    ([200] * 64, 2, 0, torch.float32, 0.2, 64),  # many tiles: no cuts
]


@pytest.mark.parametrize(("lens", "heads", "channels", "gate_dtype", "decay", "b_t"), SPLIT_CASES)
def test_split_rows_meet_the_forgetting_horizon(lens, heads, channels, gate_dtype, decay, b_t):
    gate, cu = split_case(lens, heads, channels, gate_dtype, decay)
    staging, count, scratch, facts = run_split(gate, cu, b_t)
    check_split_contract(staging, count, scratch, gate, cu, b_t, facts)


@pytest.mark.parametrize(("channels", "b_t"), [(0, 64), (128, 16)])
def test_split_cuts_long_decaying_sequences(channels, b_t):
    gate, cu = split_case([32768], 4, channels, torch.float32, 0.3 if channels else 0.05)
    staging, count, scratch, facts = run_split(gate, cu, b_t)
    assert check_split_contract(staging, count, scratch, gate, cu, b_t, facts)


@pytest.mark.parametrize("lens", [[32, 0, 16, 0], [0, 0, 0]])
def test_split_table_omits_zero_chunk_sequences(lens):
    """Interior, trailing and all-empty sequences enter no split row: a zero-chunk
    row would run an empty item on a persistent CTA."""
    gate, cu = split_case(lens, 2, 0, torch.float32, 0.1)
    staging, count, _, _ = run_split(gate, cu, 16)
    nonempty = [b for b, n in enumerate(lens) if n]
    assert count.item() == 2 * len(nonempty)
    rows = staging[: count.item()].tolist()
    assert sorted(r[0] for r in rows) == sorted(nonempty * 2)
    assert all(r[5] > r[4] for r in rows)


def test_scalar_split_scan_respects_head_stride():
    """A head slice of a wider projection scans like its compact copy: the interleaved
    columns, which would saturate every horizon at once, are never read."""
    tokens, heads = 4096, 2
    storage = torch.full((tokens, 2 * heads), -37.0, device="cuda")
    gate = storage[:, ::2]
    gate[:, 0], gate[:, 1] = -0.1, -0.2
    cu = cu_of([tokens])
    compact = run_split(gate.contiguous(), cu, 64)
    strided = run_split(gate, cu, 64)
    n = compact[1].item()
    assert n > heads  # the scan cut the sequence
    assert strided[1].item() == n
    assert sorted(strided[0][:n].tolist()) == sorted(compact[0][:n].tolist())
    assert torch.equal(strided[2].view(torch.int32), compact[2].view(torch.int32))


def test_split_int64_extents_agree():
    """The int64-extent ABI (for oversized tensors) emits the same rows."""
    gate, cu = split_case([9000, 3000], 2, 0, torch.float32, 0.1)
    a = run_split(gate, cu, 64)
    b = run_split(gate, cu, 64, use_int64_offsets=True)
    assert a[1].item() == b[1].item()
    n = a[1].item()
    assert sorted(a[0][:n].tolist()) == sorted(b[0][:n].tolist())
    assert torch.equal(a[2].view(torch.int32), b[2].view(torch.int32))


def test_split_table_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequences and heads are runtime values of one specialization per scan-row
    bucket."""
    w = wt()
    w.compile_split_table.cache_clear()
    cases = [([20000, 50], 2), ([3000, 3000, 7000], 3), ([12345], 2), ([900] * 9, 5)]
    buckets = set()
    for lens, heads in cases:
        gate, cu = split_case(lens, heads, 0, torch.float32, 0.1)
        buckets.add(run_split(gate, cu, 64)[3].static.scan_rows)
    info = w.compile_split_table.cache_info()
    assert info.currsize == len(buckets), info
    assert info.hits + info.misses == len(cases), info
