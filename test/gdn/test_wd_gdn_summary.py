"""Warpdrive GDN piece summaries (``gdn_summary``) against an fp64 recurrent reference.

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive GDN summary requires an SM100-family GPU",
)

B_T = 64


def _inputs(lens, heads, key_heads, d_k, d_v, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)
    k = torch.randn(tokens, key_heads, d_k, device="cuda", generator=gen)
    k = torch.nn.functional.normalize(k, dim=-1).to(dtype)
    v = torch.randn(tokens, heads, d_v, device="cuda", generator=gen).to(dtype)
    # Mild decay keeps the M of a 2000-token piece (the product of its gates) in fp32 range.
    gate = -0.02 * torch.rand(tokens, heads, device="cuda", generator=gen)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return k, v, gate, beta, cu


def _work_table(cu, heads, order, capacity, chunk_ranges=None):
    """One row per (piece, head) in ``order``, the chain prologue's fields (whole pieces, unless
    ``chunk_ranges[item]`` names chunks ``[begin, end)``), then garbage rows up to ``capacity``
    that a launch must not run."""
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


def _summary_operands(k, gate, beta, cu, order=None, extra_rows=0, chunk_ranges=None):
    """``gdn_tinv``'s tiles and a work table over every (piece, head) of ``cu``."""
    from attn_gym.linear._delta_rule.warpdrive import gdn_tinv

    pieces, heads = cu.numel() - 1, gate.shape[1]
    rows = gdn_tinv.tinv_rows(int(cu[-1]), pieces)
    tinv = torch.empty(rows, heads, B_T, B_T, dtype=k.dtype, device="cuda")
    row_table = torch.empty(rows, 4, dtype=torch.int32, device="cuda")
    row_count = torch.empty(1, dtype=torch.int32, device="cuda")
    gdn_tinv.gdn_tinv(k, gate, beta, cu, tinv, row_table, row_count)
    order = range(pieces * heads) if order is None else order
    return tinv, *_work_table(cu, heads, order, len(order) + extra_rows, chunk_ranges)


def _run(k, v, gate, beta, cu, order=None, extra_rows=0, chunk_ranges=None):
    from attn_gym.linear._delta_rule.warpdrive import gdn_summary

    pieces, heads = cu.numel() - 1, gate.shape[1]
    tinv, table, count = _summary_operands(k, gate, beta, cu, order, extra_rows, chunk_ranges)
    d_k, d_v = k.shape[2], v.shape[2]
    state_h = torch.full((pieces, heads, d_v, d_k), float("nan"), device="cuda")
    state_m = torch.full((pieces, heads, d_k, d_k), float("nan"), device="cuda")
    gdn_summary.gdn_summary(k, v, gate, tinv, state_h, state_m, table, count)
    return state_h, state_m


def _reference(k, v, gate, beta, cu):
    """fp64 per-piece recurrence ``S <- a S; S <- S + beta (v - S k) k^T`` (``S [V, K]``): H from
    a zero state, and M (``X_end = X_start @ M + H``) from the identity with ``v = 0``."""
    heads = gate.shape[1]
    group = heads // k.shape[1]
    k64 = k.double().repeat_interleave(group, dim=1)  # [T, H, K]
    v64, a, b = v.double(), gate.double().exp(), beta.double()
    d_k = k.shape[2]
    hs, ms = [], []
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
        s_h = torch.zeros(heads, v.shape[2], d_k, dtype=torch.float64, device="cuda")
        s_m = torch.eye(d_k, dtype=torch.float64, device="cuda").repeat(heads, 1, 1)
        for t in range(start, end):
            kt = k64[t]  # [H, K]
            for s, vt in ((s_h, v64[t]), (s_m, torch.zeros_like(kt))):
                s.mul_(a[t, :, None, None])
                err = vt - torch.einsum("hvk,hk->hv", s, kt)
                s.add_(b[t, :, None, None] * err[:, :, None] * kt[:, None, :])
        hs.append(s_h)
        ms.append(s_m)
    return torch.stack(hs), torch.stack(ms)


def _assert_close(got, ref, dtype, what):
    # bf16/fp16 operands and per-chunk 16-bit state staging bound the error relative to the state.
    tol = 3e-2 if dtype is torch.bfloat16 else 2e-3
    err = (got.double() - ref).abs().amax() / ref.abs().amax()
    assert err < tol, f"{what}: max error {err:.3g} of the state's scale (tolerance {tol})"


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "d_k", "d_v", "dtype"),
    [
        # Partial, exact and one-past chunks (zeroed Y past a piece end); GQA 2.
        ((1, 63, 64, 65, 130, 700), 4, 2, 128, 128, torch.bfloat16),
        # Mixed head dims: d_k = 64 makes chain M an M = 64 GEMM, d_v = 128 chain H M = 128.
        ((100, 257), 2, 1, 64, 128, torch.float16),
        ((300, 77), 3, 3, 128, 64, torch.bfloat16),
        # Many chunks per item: every ring wraps its stages and phases.
        ((2000,), 2, 2, 64, 64, torch.bfloat16),
    ],
    ids=["boundaries-gqa2", "fp16-dk64-dv128", "dk128-dv64", "ring-wrap"],
)
def test_gdn_summary_matches_reference(lens, heads, key_heads, d_k, d_v, dtype):
    k, v, gate, beta, cu = _inputs(lens, heads, key_heads, d_k, d_v, dtype)
    state_h, state_m = _run(k, v, gate, beta, cu)
    ref_h, ref_m = _reference(k, v, gate, beta, cu)
    _assert_close(state_h, ref_h, dtype, "H")
    _assert_close(state_m, ref_m, dtype, "M")


def test_gdn_summary_runs_listed_items_in_any_order():
    """Items run in table order (the chain prologue sorts them); rows past ``work_count`` never
    run, an empty piece writes ``H = 0`` and ``M = I``, and a permuted table writes the same bits."""
    lens, heads = (70, 0, 200, 5), 2
    k, v, gate, beta, cu = _inputs(lens, heads, 1, 128, 128, torch.bfloat16, seed=1)
    in_order = _run(k, v, gate, beta, cu)
    order = [7, 2, 0, 5, 3, 6, 1, 4]
    permuted = _run(k, v, gate, beta, cu, order=order, extra_rows=3)
    for got, ref in zip(permuted, in_order, strict=True):
        assert torch.equal(got, ref)
    state_h, state_m = permuted
    assert torch.count_nonzero(state_h[1]) == 0
    assert torch.equal(state_m[1], torch.eye(128, device="cuda").expand(heads, 128, 128))
    ref_h, ref_m = _reference(k, v, gate, beta, cu)
    _assert_close(state_h, ref_h, torch.bfloat16, "H")
    _assert_close(state_m, ref_m, torch.bfloat16, "M")


def test_gdn_summary_writes_into_views_of_a_packed_map():
    """The states may be the two halves of one packed ``[H; M]`` buffer, with the same bits as
    separate outputs; a misaligned, broadcast or self-overlapping view is rejected before launch."""
    from attn_gym.linear._delta_rule.warpdrive import gdn_summary

    lens, heads = (70, 130), 2
    k, v, gate, beta, cu = _inputs(lens, heads, 1, 128, 128, torch.bfloat16, seed=4)
    pieces = len(lens)
    tinv, table, count = _summary_operands(k, gate, beta, cu)
    expected = _run(k, v, gate, beta, cu)
    packed = torch.full((pieces, heads, 256, 128), float("nan"), device="cuda")
    gdn_summary.gdn_summary(k, v, gate, tinv, packed[:, :, :128], packed[:, :, 128:], table, count)
    for got, want in zip(packed.split(128, dim=2), expected, strict=True):
        assert torch.equal(got, want)
    wide = torch.empty(pieces, heads, 128, 256, device="cuda")
    broadcast = torch.empty(1, heads, 128, 128, device="cuda").expand(pieces, -1, -1, -1)
    # Aligned, d_k-divisible strides whose heads share rows.
    overlapping = torch.empty(pieces * heads * 128 * 128, device="cuda").as_strided(
        (pieces, heads, 128, 128), (heads * 128 * 128, 128, 128, 1)
    )
    for bad in (wide[..., 4:132], broadcast, overlapping):
        with pytest.raises(ValueError, match="summary states"):
            gdn_summary.gdn_summary(k, v, gate, tinv, bad, expected[1], table, count)


def test_gdn_summary_runs_chunk_ranges_of_split_tables():
    """A split-K row runs chunks ``[begin, end)``: both chains start from zero past chunk 0 (no
    identity seed, so M stays zero), and only a row reaching its piece's last chunk stores."""
    lens, heads = (300, 200), 2
    k, v, gate, beta, cu = _inputs(lens, heads, 1, 128, 128, torch.bfloat16, seed=3)
    # Piece 0 (5 chunks) from chunk 1 to its end; piece 1 (4 chunks) stops before its end.
    ranges = {0: (1, 5), 1: (1, 5), 2: (0, 2), 3: (0, 3)}
    state_h, state_m = _run(k, v, gate, beta, cu, chunk_ranges=ranges)
    tail = torch.tensor([B_T, lens[0]], dtype=torch.int32, device="cuda")
    ref_h, _ = _reference(k, v, gate, beta, tail)
    _assert_close(state_h[:1], ref_h, torch.bfloat16, "H")
    assert torch.count_nonzero(state_m[0]) == 0
    assert state_h[1].isnan().all() and state_m[1].isnan().all()


def test_gdn_summary_pieces_ignore_the_next_pieces_values():
    """The next piece's V rows must not reach a piece's summary, even when non-finite: the
    piece's loads stop at its end."""
    lens, heads = (100, 64), 2
    k, v, gate, beta, cu = _inputs(lens, heads, 2, 128, 128, torch.float16, seed=2)
    clean = _run(k, v, gate, beta, cu)
    v[lens[0]] = float("inf")
    poisoned = _run(k, v, gate, beta, cu)
    for got, ref in zip(poisoned, clean, strict=True):
        assert torch.equal(got[0], ref[0])


def test_gdn_summary_reuses_one_compile_across_runtime_shapes():
    """Tokens, pieces, heads and the work-table capacity stay runtime values of one
    specialization."""
    from attn_gym.linear._delta_rule.warpdrive import gdn_summary

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16)]
    gdn_summary.compile_gdn_summary.cache_clear()
    for lens, heads, key_heads in shapes:
        _run(*_inputs(lens, heads, key_heads, 128, 128, torch.bfloat16), extra_rows=heads)
    info = gdn_summary.compile_gdn_summary.cache_info()  # one specialization, looked up per launch
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
