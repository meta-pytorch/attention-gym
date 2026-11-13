"""The warpdrive KDA backward (kda_bprop) against an fp64 recurrent oracle."""

import itertools

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    cumulative_sequence_offsets,
    kda_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive KDA bprop requires SM100 or SM103",
)

B_T = 16


def _inputs(lengths, heads, k_heads, d_k, d_v, dtype, gate_dtype, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lengths)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    q = F.normalize(randn(tokens, heads, d_k), dim=-1).to(dtype)
    k = F.normalize(randn(tokens, k_heads, d_k), dim=-1).to(dtype)
    v = (randn(tokens, heads, d_v) * 0.5).to(dtype)
    gate = (-torch.rand(tokens, heads, d_k, device="cuda", generator=generator) * 0.2).to(
        gate_dtype
    )
    beta = torch.rand(tokens, heads, device="cuda", generator=generator)
    do = randn(tokens, heads, d_v).to(dtype)
    return q, k, v, gate, beta, do


def _checkpoints(q, k, v, gate, beta, offsets, initial_state):
    """The fp64 state entering every 16-token chunk, at the closed-form padded rows
    ``cu[b] // 16 + b + c``; every other row is NaN so a misaddressed read shows."""
    tokens, heads, d_k = q.shape
    d_v = v.shape[-1]
    k = k.double().repeat_interleave(heads // k.shape[1], 1)
    v, g, beta = v.double(), gate.double() * LOG2_E, beta.double()
    rows = tokens // B_T + len(offsets) - 1
    out = torch.full((rows, heads, d_v, d_k), float("nan"), device=q.device, dtype=q.dtype)
    bounds = offsets.tolist()
    for b, (start, end) in enumerate(itertools.pairwise(bounds)):
        state = torch.zeros(heads, d_v, d_k, dtype=torch.float64, device=q.device)
        if initial_state is not None:
            state = initial_state[b].double().clone()
        for t in range(start, end):
            if (t - start) % B_T == 0:
                out[start // B_T + b + (t - start) // B_T] = state.to(q.dtype)
            state = state * g[t].exp2()[:, None, :]
            delta = (v[t] - torch.einsum("hvk,hk->hv", state, k[t])) * beta[t][:, None]
            state = state + torch.einsum("hv,hk->hvk", delta, k[t])
    return out


def _reference(q, k, v, gate, beta, do, offsets, initial_state, d_final_state, scale):
    """fp64 and fp32 gradients w.r.t. per-output-head q, k, v (the kernel's outputs), gate,
    beta and the initial state."""
    heads = gate.shape[1]
    results = []
    for precision in (torch.float64, torch.float32):
        leaves = [
            t.to(precision).repeat_interleave(heads // t.shape[1], 1).requires_grad_()
            for t in (q, k, v)
        ]
        leaves += [t.to(precision).requires_grad_() for t in (gate, beta)]
        state = None
        if initial_state is not None:
            state = initial_state.to(precision).requires_grad_()
        out, final = kda_reference(
            *(t[None] for t in leaves),
            state,
            cu_seqlens=offsets,
            scale=scale,
            output_final_state=d_final_state is not None,
        )
        outputs, cotangents = [out], [do.to(precision)[None]]
        if d_final_state is not None:
            outputs.append(final)
            cotangents.append(d_final_state.to(precision))
        targets = leaves if state is None else [*leaves, state]
        grads = torch.autograd.grad(outputs, targets, cotangents)
        results.append(grads)
    return results


def _run(
    lengths,
    heads=2,
    k_heads=2,
    d_k=128,
    d_v=128,
    dtype=torch.bfloat16,
    gate_dtype=torch.float32,
    initial=False,
    exit_grad=False,
    seed=0,
):
    from attn_gym.linear._delta_rule.warpdrive.kda_bprop import kda_bprop

    offsets = cumulative_sequence_offsets(lengths)
    q, k, v, gate, beta, do = _inputs(lengths, heads, k_heads, d_k, d_v, dtype, gate_dtype, seed)
    n = len(lengths)
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    s0 = dsf = None
    if initial:
        s0 = torch.randn(n, heads, d_v, d_k, device="cuda", generator=generator) * 0.05
    if exit_grad:
        dsf = torch.randn(n, heads, d_v, d_k, device="cuda", generator=generator) * 0.05
    checkpoints = _checkpoints(q, k, v, gate, beta, offsets, s0)
    d_initial = None if s0 is None else torch.full_like(s0, float("nan"))
    grads = kda_bprop(
        q,
        k,
        v,
        gate,
        beta,
        do,
        checkpoints,
        offsets,
        scale=d_k**-0.5,
        use_initial_state=s0 is not None,
        d_final_state=dsf,
        d_initial_state=d_initial,
    )
    return (q, k, v, gate, beta, do, offsets, s0, dsf), (*grads, d_initial)


@pytest.mark.parametrize(
    "config",
    [
        {"lengths": [1, 15, 16, 17, 63, 64, 65, 130]},
        {"lengths": [200, 33, 7], "d_k": 64, "dtype": torch.float16, "initial": True, "seed": 1},
        {
            "lengths": [96, 50],
            "d_k": 64,
            "d_v": 64,
            "exit_grad": True,
            "seed": 2,
            "gate_dtype": torch.bfloat16,
        },
        {"lengths": [77, 160], "heads": 4, "k_heads": 1, "d_v": 64, "initial": True, "seed": 3},
        {"lengths": [5] * 20 + [300], "initial": True, "exit_grad": True, "seed": 4},
    ],
    ids=["ragged", "d64_128_state", "d64_exit_bf16_gate", "grouped_k", "many_short"],
)
def test_matches_fp64_recurrence(config):
    """Gradients from checkpoints entering every chunk: ragged tails, more items than CTAs,
    both channel layouts (d_k 64 / 128), both value widths, entry state and exit cotangent."""
    (q, k, v, gate, beta, do, offsets, s0, dsf), got = _run(**config)
    high, low = _reference(q, k, v, gate, beta, do, offsets, s0, dsf, q.shape[-1] ** -0.5)
    names = ("dq", "dk", "dv", "dgate", "dbeta", "d_initial_state")
    for name, actual, hi, lo in zip(names, got, high, low, strict=False):
        if actual is None:
            continue
        assert_matches_low_precision_reference(actual, hi, lo, name, source_dtype=q.dtype)


def test_neighbour_past_a_ragged_tail_is_isolated():
    """Non-finite values in the next sequence must not reach the earlier sequence's
    gradients: its loads stop at its end."""
    from attn_gym.linear._delta_rule.warpdrive.kda_bprop import kda_bprop

    lengths = [37, 40]
    offsets = cumulative_sequence_offsets(lengths)
    q, k, v, gate, beta, do = _inputs(lengths, 2, 2, 128, 128, torch.bfloat16, torch.float32, 5)
    checkpoints = _checkpoints(q, k, v, gate, beta, offsets, None)
    clean = kda_bprop(q, k, v, gate, beta, do, checkpoints, offsets, scale=0.1,
                      use_initial_state=False)  # fmt: skip
    for t in (q, k, v, gate, beta, do):
        t[lengths[0] :] = float("nan")
    poisoned = kda_bprop(q, k, v, gate, beta, do, checkpoints, offsets, scale=0.1,
                         use_initial_state=False)  # fmt: skip
    for a, b in zip(clean, poisoned, strict=True):
        assert torch.equal(a[: lengths[0]], b[: lengths[0]])


def test_one_compile_across_runtime_shapes():
    """Tokens, sequences, heads, checkpoint rows and work rows are runtime values: one static
    configuration compiles once (sizes 1, multiples of 16 and others)."""
    from attn_gym.linear._delta_rule.warpdrive.kda_bprop import compile_kda_bprop

    shapes = ([1], [16, 32], [17, 48, 5], [64] * 3 + [9])
    _run(shapes[0])
    before = compile_kda_bprop.cache_info()
    for i, lengths in enumerate(shapes):
        _run(lengths, heads=2 + i % 2, k_heads=2 + i % 2, seed=i)
    after = compile_kda_bprop.cache_info()
    assert after.currsize == before.currsize
    assert after.hits + after.misses - before.hits - before.misses == len(shapes)
