"""Warpdrive KDA checkpoint-series recompute (``kda_recompute``) against an fp64 token
recurrence.

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive KDA recompute requires SM100 or SM103",
)

B_T = 16
POISON = 0x7FC1  # a 16-bit NaN pattern for checkpoint rows the kernel must not write


def _inputs(lens, heads, key_heads, d_k, d_v, dtype, gate_dtype, state_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    k = torch.nn.functional.normalize(normal(tokens, key_heads, d_k), dim=-1).to(dtype)
    v = normal(tokens, key_heads, d_v).to(dtype)
    gate = (-0.3 * torch.rand(tokens, heads, d_k, device="cuda", generator=gen)).to(gate_dtype)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    state = None
    if state_dtype is not None:
        state = 0.1 * normal(len(lens), heads, d_v, d_k)
        state = state.to(state_dtype)
    return k, v, gate, beta, cu, state


def _chunks(length):
    return (length + B_T - 1) // B_T


def _table(lens, heads, cu, warmup=None):
    """Work table: one uncut item per (sequence, head), or with ``warmup = (write_start,
    compute_start)`` a split item that writes chunks from ``write_start`` after recomputing
    from ``compute_start``, clipped to the sequence. As in the split tables, a warmup is at
    least one chunk (``compute_start < write_start``) unless ``write_start`` is 0."""
    rows = []
    for b, length in enumerate(lens):
        n = _chunks(length)
        if n == 0:
            continue
        ws = 0 if warmup is None else min(warmup[0], n - 1)
        cs = 0 if ws == 0 else min(warmup[1], ws - 1)
        for h in range(heads):
            rows.append([b, h, ws, n, cs, n, cu[b].item(), cu[b + 1].item(), -1, -1])
    return torch.tensor(rows, dtype=torch.int32, device="cuda").reshape(-1, 10)


def _row(cu, b, c):
    """The closed-form checkpoint row of chunk ``c`` of sequence ``b``."""
    return cu[b].item() // B_T + b + c


def _run(k, v, gate, beta, cu, state, items):
    from attn_gym.linear._delta_rule.warpdrive import kda_recompute

    heads, d_k, d_v = gate.shape[1], k.shape[2], v.shape[2]
    rows = kda_recompute.checkpoint_rows(k.shape[0], cu.numel() - 1)
    ckpt = torch.empty(rows, heads, d_v, d_k, dtype=k.dtype, device="cuda")
    ckpt.view(torch.int16).fill_(POISON)
    n_items = torch.tensor([items.shape[0]], dtype=torch.int32, device="cuda")
    kda_recompute.kda_recompute(k, v, gate, beta, items, n_items, ckpt, initial_state=state)
    return ckpt


def _reference(k, v, gate, beta, lens, heads, state, starts=None):
    """fp64 KDA recurrence ``S <- S diag(exp(g)); S += beta (v - S k) k^T`` per token; returns
    the state entering every chunk, ``{(sequence, chunk): [heads, d_v, d_k]}``. ``starts[b]``
    restarts the recurrence from zero at that chunk (a split item's warmup)."""
    out, start = {}, 0
    group = heads // k.shape[1]
    k64 = k.double().repeat_interleave(group, dim=1)
    v64 = v.double().repeat_interleave(group, dim=1)
    decay = gate.double().exp()
    beta64 = beta.double()
    for b, length in enumerate(lens):
        s = torch.zeros(heads, v.shape[2], k.shape[2], dtype=torch.float64, device="cuda")
        if state is not None:
            s = state[b].double().clone()
        first = 0 if starts is None else starts[b]
        for c in range(_chunks(length)):
            if c == first and c > 0:
                s = torch.zeros_like(s)
            out[(b, c)] = s.clone()
            for t in range(start + c * B_T, min(start + (c + 1) * B_T, start + length)):
                s = s * decay[t][:, None, :]
                kt = k64[t]
                delta = beta64[t][:, None] * (v64[t] - torch.einsum("hvk,hk->hv", s, kt))
                s = s + delta[:, :, None] * kt[:, None, :]
        start += length
    return out


def _check(ckpt, ref, cu, dtype, written=None):
    """Written rows match the reference within the io rounding of growing states; every other
    row keeps its poison."""
    hit = torch.zeros(ckpt.shape[0], dtype=torch.bool, device="cuda")
    for (b, c), want in ref.items():
        if written is not None and not written(b, c):
            continue
        row = _row(cu, b, c)
        hit[row] = True
        got = ckpt[row].double()
        tol = 4 * torch.finfo(dtype).eps * max(want.abs().max().item(), 1.0) + 2e-3
        err = (got - want).abs().max().item()
        assert err <= tol, f"sequence {b} chunk {c}: error {err:.3e} > {tol:.3e}"
    untouched = ckpt[~hit].view(torch.int16)
    assert (untouched == POISON).all(), "rows outside the checkpoint map were written"


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "d", "dtype", "gate_dtype", "state_dtype"),
    [
        # Partial, exact and one-past chunks around B_T = 16 and the 6-stage raw ring.
        ((1, 15, 16, 17, 63, 64, 65, 129), 4, 4, 128, torch.bfloat16, torch.float32, None),
        # Grouped k/v heads, fp16, an fp32 initial state.
        ((300, 77), 8, 2, 128, torch.float16, torch.float32, torch.float32),
        # d = 64 (one key half, M = 64 value tiles), a bf16 state and a 16-bit gate.
        ((500, 31), 4, 4, 64, torch.bfloat16, torch.bfloat16, torch.bfloat16),
        # Unequal key and value widths; a 16-bit gate over 128 key channels.
        ((200, 33), 4, 2, (128, 64), torch.bfloat16, torch.bfloat16, torch.float32),
        ((200, 33), 4, 4, (64, 128), torch.float16, torch.float32, None),
        # Many chunks per CTA: every ring wraps its stages and phases; more items than CTAs.
        ((1500, 900) + (7,) * 70, 3, 1, 128, torch.bfloat16, torch.float32, torch.float32),
    ],
    ids=[
        "boundaries",
        "fp16-gqa-state",
        "d64-bf16state-gate16",
        "dk128-dv64-gate16",
        "dk64-dv128",
        "ring-wrap",
    ],
)
def test_kda_recompute_matches_reference(
    lens, heads, key_heads, d, dtype, gate_dtype, state_dtype
):
    d_k, d_v = d if isinstance(d, tuple) else (d, d)
    k, v, gate, beta, cu, state = _inputs(
        lens, heads, key_heads, d_k, d_v, dtype, gate_dtype, state_dtype
    )
    ckpt = _run(k, v, gate, beta, cu, state, _table(lens, heads, cu))
    _check(ckpt, _reference(k, v, gate.float(), beta, lens, heads, state), cu, dtype)


@pytest.mark.parametrize(
    ("lens", "warmup", "with_state"),
    [
        ((700, 40, 333), (9, 3), False),
        # A warmup from chunk 0 seeds from the initial state without storing chunk 0's row.
        ((700, 40, 333), (9, 0), True),
        # More split items than CTAs: the checkpoint row counters run across items.
        ((150, 3, 40) * 20, (4, 1), True),
    ],
    ids=["zero-warmup", "seeded-warmup", "many-items"],
)
def test_kda_recompute_split_items(lens, warmup, with_state):
    """Split items write from ``write_start`` after a warmup from ``compute_start`` (from zero
    unless it starts at chunk 0); earlier rows stay unwritten (the item before owns them)."""
    k, v, gate, beta, cu, state = _inputs(
        lens, 4, 4, 128, 128, torch.bfloat16, torch.float32, torch.float32 if with_state else None
    )
    table = _table(lens, 4, cu, warmup)
    ckpt = _run(k, v, gate, beta, cu, state, table)
    starts = [table[table[:, 0] == b][0, 4].item() for b in range(len(lens))]
    ws_of = [table[table[:, 0] == b][0, 2].item() for b in range(len(lens))]
    ref = _reference(k, v, gate, beta, lens, 4, state, starts)
    _check(ckpt, ref, cu, torch.bfloat16, written=lambda b, c: c >= ws_of[b])


def test_kda_recompute_ignores_the_next_sequence_past_a_ragged_tail():
    """Non-finite values in the next sequence must not reach a ragged sequence: its loads
    stop at its end."""
    lens = (37, 100)
    k, v, gate, beta, cu, _ = _inputs(lens, 4, 4, 128, 128, torch.bfloat16, torch.float32, None)
    table = _table(lens, 4, cu)
    clean = _run(k, v, gate, beta, cu, None, table)
    for t in (k, v, gate, beta):
        t[37:] = float("nan")
    poisoned = _run(k, v, gate, beta, cu, None, table)
    rows = [_row(cu, 0, c) for c in range(_chunks(37))]
    assert torch.equal(poisoned[rows], clean[rows])


def test_kda_recompute_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequence count, head counts, work rows and checkpoint rows stay runtime
    values."""
    from attn_gym.linear._delta_rule.warpdrive import kda_recompute

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16), ((16,) * 20, 1, 1)]
    kda_recompute.compile_kda_recompute.cache_clear()
    for lens, heads, key_heads in shapes:
        k, v, gate, beta, cu, _ = _inputs(
            lens, heads, key_heads, 128, 128, torch.bfloat16, torch.float32, None
        )
        _run(k, v, gate, beta, cu, None, _table(lens, heads, cu))
    info = kda_recompute.compile_kda_recompute.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
