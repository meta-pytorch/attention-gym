"""Warpdrive GDN backward piece summaries (``frost.gdn_bprop_summary``) against an fp64 reference.

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive GDN bprop summary requires SM100 or SM103",
)

B_T = 64


def _inputs(lens, heads, key_heads, d_k, d_v, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)

    def unit(*shape):
        x = torch.randn(*shape, device="cuda", generator=gen)
        return torch.nn.functional.normalize(x, dim=-1).to(dtype)

    q, k = unit(tokens, key_heads, d_k), unit(tokens, key_heads, d_k)
    do = torch.randn(tokens, heads, d_v, device="cuda", generator=gen).to(dtype)
    gate = -0.05 * torch.rand(tokens, heads, device="cuda", generator=gen)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return q, k, do, gate, beta, cu


def _work_table(cu, heads, order, capacity, chunk_ranges=None):
    """One row per (piece, head) in ``order``, the chain prologue's fields (whole pieces, unless
    ``chunk_ranges[item]`` names chunks ``[write_start, compute_end)``), then garbage rows up to
    ``capacity`` that a launch must not run."""
    starts, ends = cu[:-1].tolist(), cu[1:].tolist()
    rows = []
    for item in order:
        piece, head = divmod(item, heads)
        chunks = (ends[piece] - starts[piece] + B_T - 1) // B_T
        begin, end = (chunk_ranges or {}).get(item, (0, chunks))
        rows.append([piece, head, begin, end, begin, end, starts[piece], ends[piece], -1, -1])
    rows += [[10**6, 10**6, 0, 10**6, 0, 10**6, 0, 10**6, -1, -1]] * (capacity - len(rows))
    table = torch.tensor(rows, dtype=torch.int32, device="cuda")
    return table, torch.tensor([len(order)], dtype=torch.int32, device="cuda")


def _run(q, k, do, gate, beta, cu, order=None, extra_rows=0, chunk_ranges=None):
    from attn_gym.linear._delta_rule.frost import gdn_bprop_summary, gdn_tinv

    pieces, heads = cu.numel() - 1, gate.shape[1]
    rows = gdn_tinv.tinv_rows(int(cu[-1]), pieces)
    tinv = torch.empty(rows, heads, B_T, B_T, dtype=k.dtype, device="cuda")
    row_table = torch.empty(rows, 4, dtype=torch.int32, device="cuda")
    row_count = torch.empty(1, dtype=torch.int32, device="cuda")
    gdn_tinv.gdn_tinv(k, gate, beta, cu, tinv, row_table, row_count)
    order = range(pieces * heads) if order is None else order
    table, count = _work_table(cu, heads, order, len(order) + extra_rows, chunk_ranges)
    state_g = torch.full((pieces, heads, do.shape[2], k.shape[2]), float("nan"), device="cuda")
    scale = k.shape[2] ** -0.5
    gdn_bprop_summary.gdn_bprop_summary(q, k, gate, do, tinv, state_g, table, count, scale=scale)
    return state_g


def _reference(q, k, do, gate, beta, cu):
    """fp64 gradient of ``sum(o * dO)`` over each piece w.r.t. its entering state ``S`` (``[V,
    K]``), with ``S <- a S; S <- S + beta (-S k) k^T`` and ``o = scale * S q`` per token: the
    reverse recurrence from a zero gradient at the piece end (the output is linear in ``S``, and
    ``V`` only adds a constant)."""
    heads = gate.shape[1]
    group = heads // k.shape[1]
    q64, k64 = (t.double().repeat_interleave(group, dim=1) for t in (q, k))
    a, b, do64 = gate.double().exp(), beta.double(), do.double()
    scale = k.shape[2] ** -0.5
    grads = []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
        # Reverse sweep of the adjoint: G_t = dO_t q_t^T scale + ... (exact for a linear map).
        g = torch.zeros(heads, do.shape[2], k.shape[2], dtype=torch.float64, device="cuda")
        for t in reversed(range(start, end)):
            kt, qt = k64[t], q64[t]
            g = g + scale * do64[t][:, :, None] * qt[:, None, :]
            # Adjoint of S' = S (I - beta k k^T): G <- G (I - beta k k^T); then of S <- a S.
            gk = torch.einsum("hvk,hk->hv", g, kt)
            g = g - b[t, :, None, None] * gk[:, :, None] * kt[:, None, :]
            g = g * a[t, :, None, None]
        grads.append(g)
    return torch.stack(grads)


def _assert_close(got, ref, dtype):
    # 16-bit operands and per-chunk 16-bit restaging bound the error relative to the gradient.
    tol = 2e-2 if dtype is torch.bfloat16 else 2e-3
    err = (got.double() - ref).abs().amax() / ref.abs().amax()
    assert err < tol, f"max error {err:.3g} of the gradient's scale (tolerance {tol})"


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "d_k", "d_v", "dtype"),
    [
        # Partial, exact and one-past chunks (rows past a piece end); GQA 2.
        ((1, 63, 64, 65, 130, 700), 4, 2, 128, 128, torch.bfloat16),
        ((100, 257), 2, 1, 64, 64, torch.float16),
        # Mixed head dims: M = 64 accumulators for d_v = 64, 64-column D16 for d_k = 64.
        ((300, 77), 3, 3, 128, 64, torch.bfloat16),
        ((150, 90), 2, 2, 64, 128, torch.float16),
        # Many chunks per item: every ring wraps its stages and phases.
        ((2000,), 2, 2, 128, 128, torch.bfloat16),
    ],
    ids=["boundaries-gqa2", "fp16-d64", "dk128-dv64", "dk64-dv128", "ring-wrap"],
)
def test_gdn_bprop_summary_matches_reference(lens, heads, key_heads, d_k, d_v, dtype):
    args = _inputs(lens, heads, key_heads, d_k, d_v, dtype)
    _assert_close(_run(*args), _reference(*args), dtype)


def test_gdn_bprop_summary_runs_listed_items_in_any_order():
    """Rows past ``work_count`` never run, an empty piece writes ``G = 0``, and a permuted table
    writes the same bits."""
    args = _inputs((70, 0, 200, 5), 2, 1, 128, 128, torch.bfloat16, seed=1)
    in_order = _run(*args)
    permuted = _run(*args, order=[7, 2, 0, 5, 3, 6, 1, 4], extra_rows=3)
    assert torch.equal(permuted, in_order)
    assert torch.count_nonzero(permuted[1]) == 0
    _assert_close(permuted, _reference(*args), torch.bfloat16)


def test_gdn_bprop_summary_runs_chunk_ranges_of_split_tables():
    """A split-K row runs chunks ``compute_end - 1`` down to ``write_start`` from a zero
    gradient, and only a row reaching chunk 0 stores."""
    lens, heads = (300, 200), 2
    args = _inputs(lens, heads, 1, 128, 128, torch.bfloat16, seed=3)
    # Piece 0 (5 chunks) from chunk 3 down to 0; piece 1 (4 chunks) stops at chunk 1.
    ranges = {0: (0, 4), 1: (0, 4), 2: (1, 4), 3: (1, 3)}
    state_g = _run(*args, chunk_ranges=ranges)
    q, k, do, gate, beta, _ = args
    head = torch.tensor([0, 4 * B_T], dtype=torch.int32, device="cuda")
    _assert_close(state_g[:1], _reference(q, k, do, gate, beta, head), torch.bfloat16)
    assert state_g[1].isnan().all()


def test_gdn_bprop_summary_pieces_ignore_the_next_pieces_rows():
    """The piece's loads stop at its end, so finite values of any size in the next piece's Q,
    K and dO rows leave the piece's summary bit for bit unchanged."""
    lens = (100, 64)
    q, k, do, gate, beta, cu = _inputs(lens, 2, 2, 128, 128, torch.float16, seed=2)
    clean = _run(q, k, do, gate, beta, cu)
    nxt = slice(lens[0], None)
    q[nxt], do[nxt] = -q[nxt], 30 * do[nxt]
    k[nxt] = k[nxt].flip(-1)
    poisoned = _run(q, k, do, gate, beta, cu)
    assert torch.equal(poisoned[0], clean[0])


def test_gdn_bprop_summary_reuses_one_compile_across_runtime_shapes():
    """Tokens, pieces, heads and the work-table capacity stay runtime values of one
    specialization."""
    from attn_gym.linear._delta_rule.frost import gdn_bprop_summary

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16)]
    gdn_bprop_summary.compile_gdn_bprop_summary.cache_clear()
    for lens, heads, key_heads in shapes:
        _run(*_inputs(lens, heads, key_heads, 128, 128, torch.bfloat16), extra_rows=heads)
    info = gdn_bprop_summary.compile_gdn_bprop_summary.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
