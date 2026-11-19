"""Warpdrive KDA backward summaries (``kda_bprop_summary``) against an fp64 reference.

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA bprop summary requires an SM100-family GPU",
)

B_T = 16


def _inputs(lens, heads, key_heads, d_k, d_v, dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)

    def unit(*shape):
        x = torch.randn(*shape, device="cuda", generator=gen)
        return torch.nn.functional.normalize(x, dim=-1).to(dtype)

    q, k = unit(tokens, key_heads, d_k), unit(tokens, key_heads, d_k)
    do = torch.randn(tokens, heads, d_v, device="cuda", generator=gen).to(dtype)
    gate = -0.1 * torch.rand(tokens, heads, d_k, device="cuda", generator=gen)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return q, k, gate, beta, do, cu


def _work_table(cu, heads, order, capacity, chunk_ranges=None):
    """One row per (sequence, head) in ``order``, the split-K table's fields (whole sequences,
    unless ``chunk_ranges[item]`` names chunks ``[write_start, compute_end)``), then garbage
    rows up to ``capacity`` that a launch must not run."""
    starts, ends = cu[:-1].tolist(), cu[1:].tolist()
    rows = []
    for item in order:
        seq, head = divmod(item, heads)
        chunks = (ends[seq] - starts[seq] + B_T - 1) // B_T
        begin, end = (chunk_ranges or {}).get(item, (0, chunks))
        rows.append([seq, head, begin, end, begin, end, starts[seq], ends[seq], -1, -1])
    rows += [[10**6, 10**6, 0, 10**6, 0, 10**6, 0, 10**6, -1, -1]] * (capacity - len(rows))
    table = torch.tensor(rows, dtype=torch.int32, device="cuda")
    return table, torch.tensor([len(order)], dtype=torch.int32, device="cuda")


def _run(q, k, gate, beta, do, cu, d_final=None, order=None, extra_rows=0, chunk_ranges=None):
    from attn_gym.linear._delta_rule.warpdrive import kda_bprop_summary

    seqs, heads = cu.numel() - 1, gate.shape[1]
    order = range(seqs * heads) if order is None else order
    table, count = _work_table(cu, heads, order, len(order) + extra_rows, chunk_ranges)
    out = torch.full((seqs, heads, do.shape[2], k.shape[2]), float("nan"), device="cuda")
    kda_bprop_summary.kda_bprop_summary(
        q, k, gate, beta, do, out, table, count, scale=k.shape[2] ** -0.5, d_final_state=d_final
    )
    return out


def _reference(q, k, gate, beta, do, cu, d_final=None):
    """fp64 gradient of ``sum(o * dO) + <S_end, d_final>`` over each sequence w.r.t. its entering
    state ``S`` (``[V, K]``), with ``S <- S diag(exp(gate)); S <- S + beta (v - S k) k^T`` and
    ``o = scale * S q`` per token: the reverse adjoint sweep from ``d_final`` (or zero)."""
    heads = gate.shape[1]
    group = heads // k.shape[1]
    q64, k64 = (t.double().repeat_interleave(group, dim=1) for t in (q, k))
    a, b, do64 = gate.double().exp(), beta.double(), do.double()
    scale = k.shape[2] ** -0.5
    grads = []
    for seq, (start, end) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True)):
        g = torch.zeros(heads, do.shape[2], k.shape[2], dtype=torch.float64, device="cuda")
        if d_final is not None:
            g = d_final[seq].double().clone()
        for t in reversed(range(start, end)):
            kt, qt = k64[t], q64[t]
            g = g + scale * do64[t][:, :, None] * qt[:, None, :]
            gk = torch.einsum("hvk,hk->hv", g, kt)
            g = g - b[t, :, None, None] * gk[:, :, None] * kt[:, None, :]
            g = g * a[t, :, None, :]
        grads.append(g)
    return torch.stack(grads)


def _assert_close(got, ref, dtype):
    # 16-bit operands and per-chunk 16-bit restaging bound the error relative to the gradient.
    tol = 2e-2 if dtype is torch.bfloat16 else 3e-3
    err = (got.double() - ref).abs().amax() / ref.abs().amax()
    assert err < tol, f"max error {err:.3g} of the gradient's scale (tolerance {tol})"


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "d_k", "d_v", "dtype"),
    [
        # Partial, exact and one-past chunks (rows past a sequence end); GQA 2.
        ((1, 15, 16, 17, 63, 64, 65, 300), 4, 2, 128, 128, torch.bfloat16),
        ((100, 257), 2, 1, 64, 64, torch.float16),
        # Mixed head dims: M = 64 accumulators for d_v = 64, 64-column D16 for d_k = 64.
        ((300, 77), 3, 3, 128, 64, torch.bfloat16),
        ((150, 90), 2, 2, 64, 128, torch.float16),
        # Many chunks per item: every ring wraps its stages and phases.
        ((2000,), 2, 2, 128, 128, torch.bfloat16),
    ],
    ids=["boundaries-gqa2", "fp16-d64", "dk128-dv64", "dk64-dv128", "ring-wrap"],
)
@pytest.mark.parametrize("seeded", [False, True], ids=["zero-exit", "seeded"])
def test_kda_bprop_summary_matches_reference(lens, heads, key_heads, d_k, d_v, dtype, seeded):
    args = _inputs(lens, heads, key_heads, d_k, d_v, dtype)
    d_final = None
    if seeded:
        gen = torch.Generator(device="cuda").manual_seed(7)
        shape = (len(lens), heads, d_v, d_k)
        d_final = 0.1 * torch.randn(shape, device="cuda", generator=gen)
    _assert_close(_run(*args, d_final=d_final), _reference(*args, d_final=d_final), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("gate_value", [-1.0, -5.25])
@pytest.mark.parametrize("seeded", [False, True], ids=["zero-exit", "seeded"])
def test_kda_bprop_summary_model_range_gates(dtype, gate_value, seeded):
    """Uniform model-range gates: fp16 forms KK and A from tf32 decay operands and dU from the
    unscaled cotangent and K_restore, whose fp16 factors would under- and overflow."""
    lens = (70, 33)
    q, k, gate, beta, do, cu = _inputs(lens, 2, 2, 128, 128, dtype, seed=4)
    gate = torch.full_like(gate, gate_value)
    d_final = None
    if seeded:
        gen = torch.Generator(device="cuda").manual_seed(8)
        d_final = 0.1 * torch.randn(len(lens), 2, 128, 128, device="cuda", generator=gen)
    args = (q, k, gate, beta, do, cu)
    _assert_close(_run(*args, d_final=d_final), _reference(*args, d_final=d_final), dtype)


@pytest.mark.parametrize("seeded", [False, True], ids=["zero-exit", "seeded"])
def test_kda_bprop_summary_runs_listed_items_in_any_order(seeded):
    """Rows past ``work_count`` never run, an empty sequence writes its exit cotangent, and a
    permuted table writes the same bits."""
    lens, heads = (70, 0, 200, 5), 2
    args = _inputs(lens, heads, 1, 128, 128, torch.bfloat16, seed=1)
    d_final = None
    if seeded:
        d_final = torch.randn(len(lens), heads, 128, 128, device="cuda")
    in_order = _run(*args, d_final=d_final)
    permuted = _run(*args, d_final=d_final, order=[7, 2, 0, 5, 3, 6, 1, 4], extra_rows=3)
    assert torch.equal(permuted, in_order)
    expected_empty = torch.zeros_like(permuted[1]) if d_final is None else d_final[1]
    assert torch.equal(permuted[1], expected_empty)
    _assert_close(permuted, _reference(*args, d_final=d_final), torch.bfloat16)


def test_kda_bprop_summary_runs_chunk_ranges_of_split_tables():
    """A split-K row runs chunks ``compute_end - 1`` down to ``write_start`` from a zero exit
    cotangent, and only a row reaching chunk 0 stores."""
    lens, heads = (80, 64), 2
    args = _inputs(lens, heads, 1, 128, 128, torch.bfloat16, seed=3)
    # Sequence 0 (5 chunks) from chunk 3 down to 0; sequence 1 (4 chunks) stops at chunk 1.
    ranges = {0: (0, 4), 1: (0, 4), 2: (1, 4), 3: (1, 3)}
    out = _run(*args, chunk_ranges=ranges)
    q, k, gate, beta, do, _ = args
    head = torch.tensor([0, 4 * B_T], dtype=torch.int32, device="cuda")
    _assert_close(out[:1], _reference(q, k, gate, beta, do, head), torch.bfloat16)
    assert out[1].isnan().all()


def test_kda_bprop_summary_sequences_ignore_the_next_sequences_rows():
    """The sequence's loads stop at its end, so finite values of any size in the next
    sequence's Q, K, gate and dO rows leave its summary bit for bit unchanged."""
    lens = (100, 64)
    q, k, gate, beta, do, cu = _inputs(lens, 2, 2, 128, 128, torch.float16, seed=2)
    clean = _run(q, k, gate, beta, do, cu)
    nxt = slice(lens[0], None)
    q[nxt], do[nxt], gate[nxt], beta[nxt] = -q[nxt], 30 * do[nxt], 4 * gate[nxt], 1 - beta[nxt]
    k[nxt] = k[nxt].flip(-1)
    poisoned = _run(q, k, gate, beta, do, cu)
    assert torch.equal(poisoned[0], clean[0])


def test_kda_bprop_summary_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequences, heads and the work-table capacity stay runtime values of one
    specialization."""
    from attn_gym.linear._delta_rule.warpdrive import kda_bprop_summary

    shapes = [((100,), 2, 1), ((16, 64, 1), 6, 3), ((5000, 1, 17), 16, 16)]
    kda_bprop_summary.compile_kda_bprop_summary.cache_clear()
    for lens, heads, key_heads in shapes:
        _run(*_inputs(lens, heads, key_heads, 128, 128, torch.bfloat16), extra_rows=heads)
    info = kda_bprop_summary.compile_kda_bprop_summary.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
