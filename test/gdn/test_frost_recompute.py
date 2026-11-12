"""Warpdrive GDN checkpoint-series recompute (``frost.gdn_recompute``) against an fp64 token
recurrence.

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive GDN recompute requires SM100 or SM103",
)

B_T = 64
POISON = 0x7FC1  # a 16-bit NaN pattern for checkpoint rows the kernel must not write


def _inputs(lens, heads, key_heads, d_k, d_v, dtype, state_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)
    k = torch.randn(tokens, key_heads, d_k, device="cuda", generator=gen)
    k = torch.nn.functional.normalize(k, dim=-1).to(dtype)
    v = torch.randn(tokens, heads, d_v, device="cuda", generator=gen).to(dtype)
    gate = -0.3 * torch.rand(tokens, heads, device="cuda", generator=gen)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    state = None
    if state_dtype is not None:
        state = 0.1 * torch.randn(len(lens), heads, d_v, d_k, device="cuda", generator=gen)
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


def _run(k, v, gate, beta, cu, lens, state, items, max_ctas=None):
    from attn_gym.linear._delta_rule.frost import gdn_recompute, gdn_tinv

    heads, d_k, d_v = gate.shape[1], k.shape[2], v.shape[2]
    rows = gdn_tinv.tinv_rows(sum(lens), len(lens))
    tinv = torch.empty(rows, heads, B_T, B_T, dtype=k.dtype, device="cuda")
    table = torch.empty(rows, 4, dtype=torch.int32, device="cuda")
    count = torch.empty(1, dtype=torch.int32, device="cuda")
    gdn_tinv.gdn_tinv(k, gate, beta, cu, tinv, table, count)
    ckpt_rows = gdn_recompute.checkpoint_rows(sum(lens), len(lens)) + 1  # + a row past the series
    ckpt = torch.empty(ckpt_rows, heads, d_v, d_k, dtype=k.dtype, device="cuda")
    ckpt.view(torch.int16).fill_(POISON)
    n_items = torch.tensor([items.shape[0]], dtype=torch.int32, device="cuda")
    gdn_recompute.gdn_recompute(
        k, v, gate, tinv, items, n_items, ckpt, state_in=state, max_ctas=max_ctas
    )
    return ckpt


def _reference(k, v, gate, beta, lens, heads, state, starts=None):
    """fp64 gated delta rule ``S = a S (I - b k k^T) + b v k^T`` per token; returns the state
    entering every chunk, ``{(sequence, chunk): [heads, d_v, d_k]}``. ``starts[b]`` restarts
    the recurrence from zero at that chunk (a split item's warmup)."""
    out, start = {}, 0
    k_group, v_group = heads // k.shape[1], heads // v.shape[1]
    for b, length in enumerate(lens):
        first = 0 if starts is None else starts[b]
        s = torch.zeros(heads, v.shape[2], k.shape[2], dtype=torch.float64, device="cuda")
        if state is not None and first == 0:
            s = state[b].double().clone()
        for t in range(first * B_T, length):
            if t % B_T == 0:
                out[b, t // B_T] = s.clone()
            kt = k[start + t].double().repeat_interleave(k_group, 0)
            vt = v[start + t].double().repeat_interleave(v_group, 0)
            s = torch.exp(gate[start + t].double())[:, None, None] * s
            pred = torch.einsum("hvk,hk->hv", s, kt)
            s = s + torch.einsum("hv,hk->hvk", beta[start + t].double()[:, None] * (vt - pred), kt)
        start += length
    return out


def _row(lens, b, c):
    """Checkpoint row of chunk ``c`` of sequence ``b``: ``cu[b] // 64 + b + c``."""
    return sum(lens[:b]) // B_T + b + c


def _check(ckpt, ref, lens, dtype, skip_before=None):
    """Compare every checkpoint row with ``ref``; rows before ``skip_before[b]`` (a split
    item's warmup) must stay unwritten."""
    atol = 3e-2 if dtype is torch.bfloat16 else 1e-2
    for b, length in enumerate(lens):
        for c in range(_chunks(length)):
            row = _row(lens, b, c)
            if skip_before is not None and c < skip_before[b]:
                assert (ckpt[row].view(torch.int16) == POISON).all(), f"row {row} was written"
            else:
                expected = ref[b, c]
                got = ckpt[row].double()
                scale = max(expected.abs().max().item(), 1.0)
                torch.testing.assert_close(
                    got, expected, atol=atol * scale, rtol=0, msg=f"seq {b} chunk {c}"
                )


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "dims", "dtype", "state_dtype"),
    [
        # Partial, exact and one-past chunks; GQA 2; zero chunk-0 checkpoints.
        ((1, 63, 64, 65, 128, 129), 4, 2, (128, 128), torch.bfloat16, None),
        # fp32 initial state seeds the series; d_k = d_v = 64 (M = 64 accumulators).
        ((300, 77), 4, 1, (64, 64), torch.float16, torch.float32),
        # bf16 initial state; d_v narrower than d_k.
        ((150, 64), 2, 2, (128, 64), torch.bfloat16, torch.bfloat16),
        # More items than CTAs (scheduler claims) and many short sequences.
        ((7,) * 40 + (200,), 12, 3, (128, 128), torch.bfloat16, None),
        # One long item per head: every channel ring wraps its stages and phases.
        ((3000,), 2, 2, (128, 128), torch.float16, torch.float32),
    ],
    ids=["boundaries-gqa2", "fp16-d64-init", "dv64-bf16-init", "many-items", "ring-wrap"],
)
def test_gdn_recompute_matches_reference(lens, heads, key_heads, dims, dtype, state_dtype):
    k, v, gate, beta, cu, state = _inputs(lens, heads, key_heads, *dims, dtype, state_dtype)
    ckpt = _run(k, v, gate, beta, cu, lens, state, _table(lens, heads, cu))
    _check(ckpt, _reference(k, v, gate, beta, lens, heads, state), lens, dtype)
    assert (ckpt[-1].view(torch.int16) == POISON).all(), "wrote past the series"


def test_gdn_recompute_split_items_start_from_zero_and_write_only_their_range():
    """A split item recomputes from a zero state at ``compute_start`` (the warmup) and stores
    only the checkpoints of ``[write_start, write_end)``."""
    lens, heads = (700, 300, 130), 2
    k, v, gate, beta, cu, state = _inputs(lens, heads, heads, 128, 128, torch.bfloat16, None)
    write_start, compute_start = 4, 2
    items = _table(lens, heads, cu, warmup=(write_start, compute_start))
    ckpt = _run(k, v, gate, beta, cu, lens, state, items)
    starts = items[::heads, 4].tolist()
    ref = _reference(k, v, gate, beta, lens, heads, None, starts=starts)
    _check(ckpt, ref, lens, torch.bfloat16, skip_before=items[::heads, 2].tolist())


@pytest.mark.parametrize(
    ("dims", "dtype", "state_dtype"),
    [
        ((64, 64), torch.bfloat16, None),
        ((128, 128), torch.float16, torch.float32),
        ((128, 64), torch.bfloat16, torch.bfloat16),
    ],
    ids=["d64-zero-seeded", "fp16-fp32-state", "dv64-bf16-state"],
)
def test_gdn_recompute_one_cta_carries_signal_phases_across_mixed_items(dims, dtype, state_dtype):
    """One CTA runs every item in turn: one-chunk, odd and even chunk counts, empty rows and
    split warmups. The one-way TMEM signals' parities carry across items, and each item's
    final state wait must drain KV before the next item seeds; resetting a parity at an item
    boundary or dropping that drain breaks the series. The single-CTA result must also equal
    a launch with one CTA per item bit for bit."""
    lens = (64, 0, 65, 321, 129, 0, 63, 257)
    compute_starts = (0, 0, 0, 1, 0, 0, 0, 2)
    write_starts = (0, 0, 0, 2, 0, 0, 0, 3)
    heads = 1
    k, v, gate, beta, cu, state = _inputs(lens, heads, heads, *dims, dtype, state_dtype)
    rows = [
        [b, 0, ws, _chunks(n), cs, _chunks(n), cu[b].item(), cu[b + 1].item(), -1, -1]
        for b, (n, cs, ws) in enumerate(zip(lens, compute_starts, write_starts, strict=True))
    ]
    items = torch.tensor(rows, dtype=torch.int32, device="cuda")
    one = _run(k, v, gate, beta, cu, lens, state, items, max_ctas=1)
    ref = _reference(k, v, gate, beta, lens, heads, state, starts=compute_starts)
    _check(one, ref, lens, dtype, skip_before=write_starts)
    spread = _run(k, v, gate, beta, cu, lens, state, items, max_ctas=len(lens))
    assert torch.equal(one.view(torch.int16), spread.view(torch.int16))


def test_gdn_recompute_checkpoint_rows_are_closed_form():
    """Every sequence's chunk-0 checkpoint (its initial state) lands at row ``cu[b] // 64 + b``
    and the padding rows between sequences stay unwritten."""
    gen = torch.Generator().manual_seed(0)
    lens = torch.randint(1, 140, (300,), generator=gen).tolist()
    k, v, gate, beta, cu, state = _inputs(lens, 1, 1, 64, 64, torch.bfloat16, torch.float32)
    ckpt = _run(k, v, gate, beta, cu, lens, state, _table(lens, 1, cu))
    written = {_row(lens, b, c) for b, n in enumerate(lens) for c in range(_chunks(n))}
    first = torch.tensor([_row(lens, b, 0) for b in range(len(lens))], device="cuda")
    torch.testing.assert_close(ckpt[first], state.to(torch.bfloat16), atol=0, rtol=0)
    padding = sorted(set(range(ckpt.shape[0])) - written)
    assert (ckpt[padding].view(torch.int16) == POISON).all(), "a padding row was written"


def test_gdn_recompute_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequences, heads, work rows and checkpoint rows stay runtime values."""
    from attn_gym.linear._delta_rule.frost import gdn_recompute

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16), ((65,) * 33, 3, 1)]
    gdn_recompute.compile_gdn_recompute.cache_clear()
    for lens, heads, key_heads in shapes:
        k, v, gate, beta, cu, state = _inputs(
            lens, heads, key_heads, 128, 128, torch.bfloat16, None
        )
        _run(k, v, gate, beta, cu, lens, state, _table(lens, heads, cu))
    info = gdn_recompute.compile_gdn_recompute.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
