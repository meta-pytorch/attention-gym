"""Regression coverage for shared indexer inputs and selection error budgets."""

import pytest
import torch

from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.testing.indexer import (
    assert_indexer_selection,
    assert_selection_regret_within,
    make_indexer_test_inputs,
)


@pytest.mark.parametrize("heads", [1, 3])
def test_indexer_inputs_are_reproducible_without_changing_global_rng(heads):
    """The shared factory preserves its seed and signed weights without global RNG mutation."""
    state = torch.random.get_rng_state()
    inputs = make_indexer_test_inputs(17, heads, 8, torch.bfloat16, device="cpu")
    repeated = make_indexer_test_inputs(17, heads, 8, torch.bfloat16, device="cpu")
    assert torch.equal(torch.random.get_rng_state(), state)
    for actual, expected in zip(inputs, repeated):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (inputs[2][..., 0] >= 0.25).all()
    if heads > 1:
        assert (inputs[2][..., 1] <= -0.25).all()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal,topk", [(False, 5), (True, 5), (True, 0)])
def test_selection_oracle_handles_different_row_magnitudes(dtype, causal, topk):
    """Check normalized errors with six orders of score magnitude, padding and empty output."""
    q, k, weights = make_indexer_test_inputs(17, 3, 8, dtype, device="cpu")
    q.mul_(torch.logspace(-3, 3, 17).view(1, 17, 1, 1))
    actual = lightning_indexer(q, k, weights, topk, causal=causal, impl="reference")
    assert_indexer_selection(actual.roll(1, dims=-1), q, k, weights, topk, causal)


def test_relative_rms_detects_distributed_errors_hidden_by_absolute_mean():
    """The old maximum/absolute-mean gates pass this case; relative RMS must reject it."""
    actual = torch.tensor([[0.0, 0, 0, 0], [0.02, 0.02, 0.02, 0.02]], dtype=torch.float64)
    reference = torch.tensor([[0.0, 0, 0, 0], [0.02, 0, 0, 0]], dtype=torch.float64)
    magnitude = torch.tensor([[1e8], [1.0]], dtype=torch.float64)
    rounding_factor = 1e-3
    allowance = rounding_factor * magnitude
    assert (actual.amax(-1, keepdim=True) <= reference.amax(-1, keepdim=True) + allowance).all()
    assert actual.mean() <= reference.mean() + allowance.mean()
    with pytest.raises(AssertionError, match="relative RMS"):
        assert_selection_regret_within(
            actual,
            reference,
            magnitude,
            torch.ones_like(actual, dtype=torch.bool),
            rounding_factor,
        )


def test_relative_rms_ignores_padding_position():
    """Different padding locations do not change aggregate reference regret."""
    assert_selection_regret_within(
        torch.tensor([[0.0, 0.02]]),
        torch.tensor([[0.02, 0.0]]),
        torch.ones(1, 1),
        torch.tensor([[False, True]]),
        0.0,
    )


def test_zero_score_regret_requires_exact_zero():
    """Zero-magnitude rows keep an exact error budget rather than dividing by zero."""
    zeros = torch.zeros(1, 2, dtype=torch.float64)
    magnitude = torch.zeros(1, 1, dtype=torch.float64)
    valid = torch.ones_like(zeros, dtype=torch.bool)
    assert_selection_regret_within(zeros, zeros, magnitude, valid, 1e-3)
    with pytest.raises(AssertionError, match="pointwise"):
        assert_selection_regret_within(zeros + 1e-12, zeros, magnitude, valid, 1e-3)
