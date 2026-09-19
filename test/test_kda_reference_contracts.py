"""Independent max/RMS guards and the full-precision KDA measuring stick."""

import pytest
import torch

from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_rms_matches_low_precision_reference,
    kda_reference,
)


@pytest.mark.parametrize("corruption", ["isolated", "widespread"])
def test_max_and_rms_catch_different_errors(corruption):
    high = torch.ones(1024, dtype=torch.float64)
    if corruption == "widespread":
        high[0] = 1024
    actual = high.clone()
    if corruption == "isolated":
        actual[0] += 0.1
        passing, failing = (
            assert_rms_matches_low_precision_reference,
            assert_matches_low_precision_reference,
        )
    else:
        actual[1:] += 1
        passing, failing = (
            assert_matches_low_precision_reference,
            assert_rms_matches_low_precision_reference,
        )
    passing(actual, high, high, "negative control")
    with pytest.raises(AssertionError):
        failing(actual, high, high, "negative control")


def test_rms_envelope_accounts_for_reference_error():
    high = torch.tensor([1.0, 2.0], dtype=torch.float64)
    actual = high + 0.4
    assert_rms_matches_low_precision_reference(actual, high, high + 0.25, "rounded reference")
    with pytest.raises(AssertionError, match="RMS error"):
        assert_rms_matches_low_precision_reference(actual, high, high, "exact reference")


@pytest.mark.parametrize(
    "check",
    [assert_matches_low_precision_reference, assert_rms_matches_low_precision_reference],
)
@pytest.mark.parametrize("corruption", [1e-30, float("inf"), float("nan")])
def test_accuracy_envelopes_keep_zero_and_finite_contracts(check, corruption):
    zero = torch.zeros(4, dtype=torch.float64)
    check(zero, zero, zero, "zero")
    actual = zero.clone()
    actual[0] = corruption
    with pytest.raises(AssertionError):
        check(actual, zero, zero, "corrupt zero")


def test_recurrent_reference_preserves_fp64_gate_bits():
    """A one-token carried state exposes a hidden FP32 gate cast in a supposed FP64 oracle."""
    one = torch.ones(1, 1, 1, 1, dtype=torch.float64)
    zero = torch.zeros_like(one)
    beta = torch.zeros(1, 1, 1, dtype=torch.float64)
    gate = torch.full_like(one, -5 + 2**-30, requires_grad=True)
    output, state = kda_reference(one, zero, zero, gate, beta, one, scale=1.0)
    assert state is not None
    expected = gate.exp()
    torch.testing.assert_close(output, expected, rtol=1e-14, atol=0)
    torch.testing.assert_close(state, expected, rtol=1e-14, atol=0)
    (gradient,) = torch.autograd.grad((output, state), gate, (one, one))
    torch.testing.assert_close(gradient, 2 * expected, rtol=1e-14, atol=0)
    rounded, _ = kda_reference(
        one, zero, zero, gate.detach().float().double(), beta, one, scale=1.0
    )
    assert not torch.equal(output, rounded)
