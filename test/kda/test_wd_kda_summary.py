"""Warpdrive KDA piece summaries (``kda_summary``) against the fp64 recurrent KDA oracle.

``H`` is a piece's final state from a zero seed and ``M`` its transition: the final state of
the same recurrence from the identity with ``v = 0`` (``X_end = X_start @ M + H``).

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

from attn_gym.testing.kda import assert_matches_low_precision_reference, kda_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive KDA summary requires SM100 or SM103",
)

B_T = 16


def _inputs(
    lens,
    heads,
    key_heads,
    d_k,
    d_v,
    dtype,
    *,
    value_heads=None,
    gate_dtype=torch.float32,
    beta_dtype=torch.float32,
    decay=0.3,
    seed=0,
):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    k = torch.nn.functional.normalize(normal(tokens, key_heads, d_k), dim=-1).to(dtype)
    v = normal(tokens, value_heads or heads, d_v).to(dtype)
    gate = -decay * torch.rand(tokens, heads, d_k, device="cuda", generator=gen)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return k, v, gate.to(gate_dtype), beta.to(beta_dtype), cu


def _work_table(cu, heads, chunk_ranges=None, extra_rows=0):
    """One row per (piece, head), whole pieces unless ``chunk_ranges[(piece, head)]`` names
    chunks ``[begin, end)``, longest first as the chain prologue orders them, then garbage rows
    past ``work_count`` that a launch must not run."""
    starts, ends = cu[:-1].tolist(), cu[1:].tolist()
    rows = []
    for piece, (start, end) in enumerate(zip(starts, ends, strict=True)):
        chunks = -(-(end - start) // B_T)
        for head in range(heads):
            begin, stop = (chunk_ranges or {}).get((piece, head), (0, chunks))
            rows.append([piece, head, begin, stop, begin, stop, start, end, -1, -1])
    rows.sort(key=lambda r: -(r[3] - r[4]))
    rows += [[10**6, 10**6, 0, 10**6, 0, 10**6, 0, 10**6, -1, -1]] * extra_rows
    table = torch.tensor(rows, dtype=torch.int32, device="cuda")
    count = torch.tensor([len(rows) - extra_rows], dtype=torch.int32, device="cuda")
    return table, count


def _run(k, v, gate, beta, cu, chunk_ranges=None, extra_rows=0):
    from attn_gym.linear._delta_rule.warpdrive import kda_summary

    pieces, heads, d_k, d_v = cu.numel() - 1, gate.shape[1], k.shape[2], v.shape[2]
    table, count = _work_table(cu, heads, chunk_ranges, extra_rows)
    state_h = torch.full((pieces, heads, d_v, d_k), float("nan"), device="cuda")
    state_m = torch.full((pieces, heads, d_k, d_k), float("nan"), device="cuda")
    kda_summary.kda_summary(k, v, gate, beta, state_h, state_m, table, count)
    return state_h, state_m


def _reference(k, v, gate, beta, cu, dtype):
    """``(H, M)`` per piece from the recurrent oracle in ``dtype`` (k / v heads expanded)."""
    heads, d_k = gate.shape[1], k.shape[2]
    pieces = cu.numel() - 1
    k = k.repeat_interleave(heads // k.shape[1], dim=1).to(dtype)
    v = v.repeat_interleave(heads // v.shape[1], dim=1).to(dtype)
    g, b = gate.to(dtype), beta.to(dtype)
    _, h = kda_reference(k[None], k[None], v[None], g[None], b[None], cu_seqlens=cu)
    eye = torch.eye(d_k, dtype=dtype, device="cuda").expand(pieces, heads, d_k, d_k)
    zero_v = torch.zeros_like(k)
    _, m = kda_reference(k[None], k[None], zero_v[None], g[None], b[None], eye, cu_seqlens=cu)
    return h, m


def _assert_matches(got, k, v, gate, beta, cu, dtype):
    hi = _reference(k, v, gate, beta, cu, torch.float64)
    lo = _reference(k, v, gate, beta, cu, torch.float32)
    for name, actual, high, low in zip("HM", got, hi, lo, strict=True):
        assert_matches_low_precision_reference(actual, high, low, name, source_dtype=dtype)


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "d_k", "d_v", "dtype", "options"),
    [
        # Partial, exact and one-past chunks around B_T = 16.
        ((1, 15, 16, 17, 63, 64, 65, 129), 4, 4, 128, 128, torch.bfloat16, {}),
        # Grouped key and value heads, fp16, a 16-bit gate and beta.
        (
            (300, 77),
            8,
            2,
            128,
            128,
            torch.float16,
            {"value_heads": 4, "gate_dtype": torch.float16, "beta_dtype": torch.float16},
        ),
        # Mixed head dims: chain M is a d_k-row GEMM, chain H a d_v-row one.
        ((500, 31), 4, 4, 64, 64, torch.bfloat16, {}),
        ((200, 9), 2, 2, 128, 64, torch.bfloat16, {"gate_dtype": torch.bfloat16}),
        ((200, 9), 2, 1, 64, 128, torch.float16, {"beta_dtype": torch.bfloat16}),
        # Many chunks per item (every ring wraps its stages and phases) and more items than
        # CTAs (the load warp claims the rest from the scheduler counter).
        ((2000,) + (7,) * 40 + (300,), 4, 2, 128, 128, torch.bfloat16, {"decay": 0.02}),
    ],
    ids=["boundaries", "gqa-fp16-16bit-gate-beta", "d64", "dk128-dv64", "dk64-dv128", "many"],
)
def test_kda_summary_matches_reference(lens, heads, key_heads, d_k, d_v, dtype, options):
    k, v, gate, beta, cu = _inputs(lens, heads, key_heads, d_k, d_v, dtype, **options)
    _assert_matches(_run(k, v, gate, beta, cu, extra_rows=3), k, v, gate, beta, cu, dtype)


def test_kda_summary_empty_pieces_write_zero_and_identity():
    """An empty piece writes ``H = 0`` and ``M = I``; its neighbours match the reference."""
    lens, heads = (70, 0, 200, 0, 5), 2
    k, v, gate, beta, cu = _inputs(lens, heads, 1, 128, 128, torch.bfloat16, seed=1)
    state_h, state_m = _run(k, v, gate, beta, cu)
    for piece in (1, 3):
        assert torch.count_nonzero(state_h[piece]) == 0
        assert torch.equal(state_m[piece], torch.eye(128, device="cuda").expand(heads, 128, 128))
    _assert_matches((state_h, state_m), k, v, gate, beta, cu, torch.bfloat16)


def test_kda_summary_runs_chunk_ranges_of_split_tables():
    """An item from chunk ``begin > 0`` is unseeded: both chains start from zero (M stays zero),
    and only an item reaching its piece's last chunk stores."""
    lens, heads = (300, 200), 2
    k, v, gate, beta, cu = _inputs(lens, heads, 1, 128, 128, torch.bfloat16, seed=3)
    # Piece 0 (19 chunks) from chunk 1 to its end; piece 1 (13 chunks) stops before its end.
    ranges = {(0, 0): (1, 19), (0, 1): (1, 19), (1, 0): (0, 5), (1, 1): (2, 12)}
    state_h, state_m = _run(k, v, gate, beta, cu, chunk_ranges=ranges)
    tail = torch.tensor([B_T, lens[0]], dtype=torch.int32, device="cuda")
    hi, _ = _reference(k, v, gate, beta, tail, torch.float64)
    lo, _ = _reference(k, v, gate, beta, tail, torch.float32)
    assert_matches_low_precision_reference(state_h[:1], hi, lo, "H", source_dtype=torch.bfloat16)
    assert torch.count_nonzero(state_m[0]) == 0
    assert state_h[1].isnan().all() and state_m[1].isnan().all()


def test_kda_summary_ignores_the_next_piece_past_a_ragged_tail():
    """Non-finite values in the next piece must not reach a piece's summary: its loads stop
    at its end."""
    lens, heads = (37, 100), 4
    k, v, gate, beta, cu = _inputs(lens, heads, 4, 128, 128, torch.bfloat16, seed=2)
    clean = _run(k, v, gate, beta, cu)
    for t in (k, v, gate, beta):
        t[lens[0] :] = float("nan")
    poisoned = _run(k, v, gate, beta, cu)
    for got, ref in zip(poisoned, clean, strict=True):
        assert torch.equal(got[0], ref[0])


def test_kda_summary_reuses_one_compile_across_runtime_shapes():
    """Tokens, pieces, head counts and work rows stay runtime values of one specialization."""
    from attn_gym.linear._delta_rule.warpdrive import kda_summary

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16), ((16,) * 20, 1, 1)]
    kda_summary.compile_kda_summary.cache_clear()
    for lens, heads, key_heads in shapes:
        _run(*_inputs(lens, heads, key_heads, 128, 128, torch.bfloat16), extra_rows=heads)
    info = kda_summary.compile_kda_summary.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
