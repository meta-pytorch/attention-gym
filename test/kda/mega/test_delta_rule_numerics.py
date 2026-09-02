"""Numerical contracts shared by the optimized GDN and KDA implementations."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip(
    "cutlass.experimental",
    reason="the Mega numerical tests require nvidia-cutlass-dsl>=4.7",
)

from attn_gym.linear import chunk_gdn, chunk_kda, recurrent_gdn
from attn_gym.linear.kda.impl.mega_ops import chunk_mega_packed_local_bwd_op
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    clone_kda_inputs,
    kda_reference,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the Mega numerical tests require SM100 or SM103",
)

_MEGA = {"backend": "mega"}
_GRADIENT_NAMES = ("dq", "dk", "dv", "dgate", "dbeta")
# On GB200, healthy GDN/KDA paths measure at most 0.82 BF16 eps; the superseded
# KDA rounding path measures 1.97--2.13 eps for dGate. This limit retains at least
# 1.5x headroom from both ranges without encoding a different constant per result.
_RMS_EPS_LIMIT = 1.25


def _d_output_like(output: torch.Tensor, seed: int) -> torch.Tensor:
    """Generate a deterministic BF16 cotangent for one output."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(output.shape, dtype=torch.bfloat16, device="cuda", generator=generator)


def _gdn_result(
    inputs: tuple[torch.Tensor, ...],
    d_output: torch.Tensor,
    *,
    precision: torch.dtype | None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run GDN through Mega or its eager recurrence and differentiate the output."""
    leaves = clone_kda_inputs(inputs, dtype=precision)
    if precision is None:
        output = chunk_gdn(*leaves, impl="fused", kernel_options=_MEGA)[0]
    else:
        output = recurrent_gdn(*leaves, impl="reference")[0]
    return output, torch.autograd.grad(output, leaves, d_output.to(output.dtype))


def _kda_result(
    inputs: tuple[torch.Tensor, ...],
    d_output: torch.Tensor,
    *,
    precision: torch.dtype,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Run the eager KDA recurrence and differentiate its output."""
    leaves = clone_kda_inputs(inputs, dtype=precision)
    output = kda_reference(*leaves, output_final_state=False)[0]
    return output, torch.autograd.grad(output, leaves, d_output.to(output.dtype))


def _assert_value(
    actual: torch.Tensor,
    high_precision: torch.Tensor,
    low_precision: torch.Tensor,
    name: str,
) -> None:
    """Check localized error and aggregate error measured in BF16 epsilon."""
    assert_matches_low_precision_reference(actual, high_precision, low_precision, name)
    assert_relative_rms_within(
        actual,
        high_precision,
        name,
        max_eps=_RMS_EPS_LIMIT,
        source_dtype=torch.bfloat16,
    )


def _assert_result(
    actual: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
    low_precision: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
    high_precision: tuple[torch.Tensor, tuple[torch.Tensor, ...]],
) -> None:
    """Check an output and all gradients against the low- and high-precision oracles."""
    _assert_value(actual[0], high_precision[0], low_precision[0], "output")
    for name, value, high, low in zip(
        _GRADIENT_NAMES,
        actual[1],
        high_precision[1],
        low_precision[1],
        strict=True,
    ):
        _assert_value(value, high, low, name)


@pytest.mark.parametrize("family", ("gdn", "kda"))
def test_mega_exact_orthogonal_writes(family: str) -> None:
    """Orthogonal unit writes must cross all chunk boundaries without rounding."""
    identity = torch.eye(128, device="cuda", dtype=torch.bfloat16)
    k = identity[None, :, None, :]
    q = identity.roll(64, dims=0)[None, :, None, :]
    value = identity.roll(1, dims=-1)[None, :, None, :]
    beta = torch.ones(1, 128, 1, device="cuda")
    gate = torch.zeros_like(q, dtype=torch.float32) if family == "kda" else torch.zeros_like(beta)

    if family == "gdn":
        output, state = chunk_gdn(
            q,
            k,
            value,
            gate,
            beta,
            scale=1.0,
            output_final_state=True,
            impl="fused",
            kernel_options=_MEGA,
        )
    else:
        output, state = chunk_kda(
            q,
            k,
            value,
            gate,
            beta,
            scale=1.0,
            output_final_state=True,
            kernel_options=_MEGA,
        )

    expected_output = value.roll(64, dims=1)
    expected_output[:, :64] = 0
    assert state is not None
    assert torch.equal(output, expected_output)
    assert torch.equal(state, value[0, :, 0].T[None, None].to(state.dtype))


@pytest.mark.parametrize("seed", (888, 889, 890))
def test_gdn_mega_model_range_backward(seed: int) -> None:
    """GDN output and gradients must remain within the BF16 reference error budget."""
    inputs = make_gdn_test_inputs(
        128,
        key_heads=2,
        value_heads=2,
        seed=seed,
        requires_grad=True,
    )[:5]
    d_output = _d_output_like(inputs[2], seed + 1)
    actual = _gdn_result(inputs, d_output, precision=None)
    low_precision = _gdn_result(inputs, d_output, precision=torch.float32)
    high_precision = _gdn_result(inputs, d_output, precision=torch.float64)
    _assert_result(actual, low_precision, high_precision)


@pytest.mark.parametrize("seed", (888, 889, 890))
def test_kda_mega_forward_and_public_backward_model_range(seed: int) -> None:
    """Mega forward and the public fallback backward must meet the BF16 error budget."""
    inputs = make_kda_test_inputs(128, heads=2, seed=seed, requires_grad=True)
    d_output = _d_output_like(inputs[2], seed + 1)
    actual_inputs = clone_kda_inputs(inputs)
    output = chunk_kda(*actual_inputs, output_final_state=False, kernel_options=_MEGA)[0]
    actual = output, torch.autograd.grad(output, actual_inputs, d_output)
    low_precision = _kda_result(inputs, d_output, precision=torch.float32)
    high_precision = _kda_result(inputs, d_output, precision=torch.float64)
    _assert_result(actual, low_precision, high_precision)


@pytest.mark.parametrize("seed", (888, 889, 890))
def test_kda_mega_raw_backward_precision(seed: int) -> None:
    """Raw Mega backward must stay within the BF16 reference error budget."""
    inputs = make_kda_test_inputs(128, heads=2, seed=seed, requires_grad=True)
    d_output = _d_output_like(inputs[2], seed + 1)
    cu_seqlens = torch.tensor([0, 128], device="cuda", dtype=torch.int32)
    gradients = chunk_mega_packed_local_bwd_op(
        *inputs,
        d_output,
        cu_seqlens,
        False,
        128**-0.5,
    )
    low_precision = _kda_result(inputs, d_output, precision=torch.float32)[1]
    high_precision = _kda_result(inputs, d_output, precision=torch.float64)[1]
    for name, value, high, low in zip(
        _GRADIENT_NAMES,
        gradients,
        high_precision,
        low_precision,
        strict=True,
    ):
        _assert_value(value, high, low, f"seed={seed} {name}")
