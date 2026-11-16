"""Public scalar-GDN warpdrive coverage at ``K = V = 64``.

Every plan the drivers can choose at D=64 (uncut, exact chain, approximate split, and the
automatic choice, which must not take the 128-only d_v split) must match the eager reference on
output, final state, and all six gradients, with grouped heads, empty sequences, and partial tails.
The paged route must match the non-paged warpdrive call bitwise.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, paged_chunk_gdn
from attn_gym.testing import make_gdn_test_inputs, strided_state_pool
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_rms_matches_low_precision_reference,
    clone_kda_inputs,
)

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive GDN path requires an SM100-family GPU",
)

WARPDRIVE = {"backend": "warpdrive"}
DIM = 64
# A one-token sequence, an empty one, partial and complete 64-token chunks.
LENGTHS = (65, 0, 1, 127, 64)
# Forward pieces and backward pieces; ``None`` keeps the automatic plan.
SCHEMES = {"automatic": None, "uncut": (0, 0), "chain": (2, 2), "split": (0, 0)}


def _pin_plans(monkeypatch, scheme: str) -> None:
    from attn_gym.linear._delta_rule.warpdrive.drivers import gdn

    if SCHEMES[scheme] is None:
        build = gdn.ForwardPlan.build

        def checked(*args):
            plan = build(*args)
            assert plan.tiles_per_head == 1, "the d_v split serves only D=128"
            return plan

        monkeypatch.setattr(gdn.ForwardPlan, "build", checked)
        return
    forward_pieces, backward_pieces = SCHEMES[scheme]
    num_sm = torch.cuda.get_device_properties().multi_processor_count
    monkeypatch.setattr(
        gdn.ForwardPlan, "build", lambda *_: gdn.ForwardPlan(forward_pieces, 1, num_sm)
    )
    monkeypatch.setattr(
        gdn.BackwardPlan, "build", lambda *_: gdn.BackwardPlan(backward_pieces, num_sm)
    )


def _results(inputs, cu_seqlens, cotangents, stateful: bool, **kwargs):
    operands = inputs if stateful else inputs[:5]
    output, final_state = chunk_gdn(
        *operands, cu_seqlens=cu_seqlens, output_final_state=stateful, **kwargs
    )
    results = (output,) if final_state is None else (output, final_state)
    cotangents = tuple(c.to(r.dtype) for c, r in zip(cotangents, results, strict=False))
    return (*results, *torch.autograd.grad(results, operands, cotangents))


@pytest.mark.parametrize(
    ("scheme", "dtype", "key_heads"),
    [
        ("automatic", torch.bfloat16, 1),
        ("automatic", torch.float16, 2),
        ("uncut", torch.bfloat16, 1),
        ("chain", torch.bfloat16, 1),
        ("split", torch.bfloat16, 2),
    ],
)
def test_d64_forward_backward_matches_reference(monkeypatch, scheme, dtype, key_heads) -> None:
    _pin_plans(monkeypatch, scheme)
    stateful = scheme != "split"  # split schedules require a no-state call
    *inputs, cu_seqlens = make_gdn_test_inputs(
        LENGTHS, key_heads=key_heads, value_heads=2, key_dim=DIM, dtype=dtype, seed=641
    )
    generator = torch.Generator(device="cuda").manual_seed(643)
    cotangents = tuple(
        torch.randn(t.shape, device="cuda", generator=generator) for t in (inputs[2], inputs[5])
    )

    options = {"split_forward": True, "split_backward": True} if scheme == "split" else {}
    actual = _results(
        clone_kda_inputs(inputs, requires_grad=True),
        cu_seqlens,
        cotangents,
        stateful,
        kernel_options={**WARPDRIVE, **options},
    )
    low, high = (
        _results(
            clone_kda_inputs(inputs, dtype=precision, requires_grad=True),
            cu_seqlens,
            cotangents,
            stateful,
            impl="reference",
        )
        for precision in (torch.float32, torch.float64)
    )
    names = ("output", "state") if stateful else ("output",)
    names += ("dq", "dk", "dv", "dgate", "dbeta", "dstate")[: len(actual) - len(names)]
    for name, got, high_tensor, low_tensor in zip(names, actual, high, low, strict=True):
        assert_matches_low_precision_reference(
            got, high_tensor, low_tensor, f"{scheme} {name}", source_dtype=dtype
        )
        assert_rms_matches_low_precision_reference(
            got, high_tensor, low_tensor, f"{scheme} {name}", source_dtype=dtype
        )


def test_d64_paged_matches_stateful_chunk() -> None:
    """Resumed, fresh, empty, and null routes advance a D=64 pool like the stateful chunk call."""
    lengths = (65, 0, 0, 127, 64)
    q, k, value, gate, beta, _, cu_seqlens = make_gdn_test_inputs(
        lengths, key_heads=1, value_heads=2, key_dim=DIM, seed=647
    )
    state_indices = torch.tensor([4, 2, 5, 0, 1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True, True, False], device="cuda")
    storage, pool = strided_state_pool(6, 2, DIM, DIM, prefix=0, suffix=32)
    expected_pool = storage.clone()[:, : pool[0].numel()].view_as(pool)

    initial_state = torch.zeros(len(lengths), 2, DIM, DIM, device="cuda")
    initial_state[0], initial_state[2] = expected_pool[4], expected_pool[5]
    expected, final_state = chunk_gdn(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        kernel_options=WARPDRIVE,
    )
    null_rows = slice(sum(lengths[:3]), sum(lengths[:4]))
    expected[:, null_rows] = 0
    for sequence, slot in ((0, 4), (1, 2), (2, 5), (4, 1)):
        expected_pool[slot] = final_state[sequence]

    with torch.no_grad():
        output = paged_chunk_gdn(
            q,
            k,
            value,
            gate,
            beta,
            pool,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
            kernel_options=WARPDRIVE,
        )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)
