"""The KDA warpdrive driver backward with grouped q/k/v heads (fewer heads than the gate)."""

import pytest
import torch
import torch.nn.functional as F

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    cumulative_sequence_offsets,
    kda_reference,
)

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")
pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="warpdrive KDA requires an SM100-family GPU",
)

HEADS = 4
DIM = 128


def _inputs(lengths, q_heads, k_heads, v_heads, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lengths)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    q = F.normalize(randn(tokens, q_heads, DIM), dim=-1).bfloat16()
    k = F.normalize(randn(tokens, k_heads, DIM), dim=-1).bfloat16()
    v = (randn(tokens, v_heads, DIM) * 0.5).bfloat16()
    gate = -torch.rand(tokens, HEADS, DIM, device="cuda", generator=generator) * 0.1
    beta = torch.rand(tokens, HEADS, device="cuda", generator=generator)
    do = randn(tokens, HEADS, DIM).bfloat16()
    return q, k, v, gate, beta, do


@pytest.mark.parametrize("scheme", ["uncut", "chain", "split"])
@pytest.mark.parametrize("heads", [(4, 2, 2), (2, 1, 4)], ids=["kv_grouped", "qk_grouped"])
def test_grouped_head_gradients_sum_over_groups(monkeypatch, scheme, heads):
    """dq/dk/dv of a grouped head are the sums over the output heads that read it."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda as driver

    num_sm = torch.cuda.get_device_properties().multi_processor_count
    pieces = 2 if scheme == "chain" else 0
    monkeypatch.setattr(
        driver.BackwardPlan, "build", lambda *args: driver.BackwardPlan(pieces, num_sm)
    )
    lengths = [300, 77]
    offsets = cumulative_sequence_offsets(lengths)
    q, k, v, gate, beta, do = _inputs(lengths, *heads, seed=7)
    grads = driver.kda_backward(
        q, k, v, gate, beta, do, offsets, scale=DIM**-0.5, split=scheme == "split"
    )
    for got, t in zip(grads[:3], (q, k, v), strict=True):
        assert got.shape == t.shape and got.dtype == t.dtype
    references = []
    for precision in (torch.float64, torch.float32):
        leaves = [t.to(precision).requires_grad_() for t in (q, k, v, gate, beta)]
        expanded = [t.repeat_interleave(HEADS // t.shape[1], 1) for t in leaves[:3]]
        out, _ = kda_reference(
            *(t[None] for t in (*expanded, *leaves[3:])),
            cu_seqlens=offsets,
            scale=DIM**-0.5,
            output_final_state=False,
        )
        references.append(torch.autograd.grad(out, leaves, do.to(precision)[None]))
    names = ("dq", "dk", "dv", "dgate", "dbeta")
    for name, got, high, low in zip(names, grads[:5], *references, strict=True):
        assert_matches_low_precision_reference(got, high, low, f"{scheme} {name}")
