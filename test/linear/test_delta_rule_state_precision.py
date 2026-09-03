"""Dynamic-range contracts for the FP32 recurrent state of the optimized delta-rule paths.

The fused GDN and KDA implementations carry the recurrent state and its cotangent in FP32
between chunks, but stage the chunk-entry state and the per-chunk state cotangent as
low-precision MMA operands and tapes. These tests pin what survives that policy: BF16 shares
the FP32 exponent range and must transport adversarially large states and tiny final-state
cotangents; FP16 cannot, which the documented ``xfail`` entries record.
"""

from __future__ import annotations

import math

import pytest
import torch

from attn_gym.linear import chunk_gdn, chunk_kda, recurrent_gdn
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    kda_reference,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the fused delta-rule paths require CUDA capability 8.0 or newer",
)


def _mega_available() -> bool:
    try:
        import cutlass.experimental  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


_MEGA_SKIP = pytest.mark.skipif(
    not _mega_available(),
    reason="the Mega backend requires nvidia-cutlass-dsl>=4.7 on SM100/SM103",
)
# Documented limitations: only the contract assertions may fail, and they must keep failing.
_FP16_STATE_RANGE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="FP16 execution stages chunk-entry states as FP16 MMA operands; 65536 overflows",
)
_FP16_COTANGENT_RANGE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="FP16 execution stages state cotangents as FP16 tapes; 2^-25 flushes to zero",
)
_MEGA_KDA_CARRY = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Mega KDA decays the carried state through a Q/K/V-dtype diagonal MMA, not in FP32",
)
# The FP32 carry may differ from the eager reference only by log2-domain gate conversions.
_FP32_CARRY_RTOL = 1e-5

_FAMILIES = ("gdn", "kda")
_DTYPES = (pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16"))
_BACKENDS = (
    pytest.param(None, id="fused"),
    pytest.param({"backend": "mega"}, id="mega", marks=_MEGA_SKIP),
)


def _zero_inputs(
    family: str, dtype: torch.dtype, tokens: int, gate_value: float
) -> tuple[torch.Tensor, ...]:
    """Zero Q/K/V with a constant natural-log gate, isolating the state-carry dataflow."""
    shape = (1, tokens, 1, 128)
    q, k, v = (torch.zeros(shape, device="cuda", dtype=dtype) for _ in range(3))
    gate_shape = shape if family == "kda" else shape[:-1]
    gate = torch.full(gate_shape, gate_value, device="cuda")
    beta = torch.full(shape[:-1], 0.5, device="cuda")
    return q, k, v, gate, beta


def _run(
    family: str,
    kernel_options: dict[str, str] | None,
    inputs: tuple[torch.Tensor, ...],
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the optimized path or the eager FP32 reference when ``kernel_options`` is ``"ref"``."""
    if kernel_options == "ref":
        if family == "gdn":
            return recurrent_gdn(*inputs, initial_state, impl="reference", output_final_state=True)
        return kda_reference(*inputs, initial_state, output_final_state=True)
    fn = chunk_gdn if family == "gdn" else chunk_kda
    output, state = fn(
        *inputs,
        initial_state,
        impl="fused",
        output_final_state=True,
        kernel_options=kernel_options,
    )
    assert state is not None
    return output, state


def _assert_fp32_carry(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    """Check an FP32-carried result pointwise and in aggregate against the FP32 reference."""
    torch.testing.assert_close(actual, expected, rtol=_FP32_CARRY_RTOL, atol=0.0)
    assert_relative_rms_within(
        actual,
        expected,
        name,
        max_eps=_FP32_CARRY_RTOL / torch.finfo(torch.float32).eps,
        source_dtype=torch.float32,
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kernel_options", _BACKENDS)
@pytest.mark.parametrize("family", _FAMILIES)
def test_large_state_decays_in_fp32(
    family: str, kernel_options: dict[str, str] | None, dtype: torch.dtype, request
) -> None:
    """An FP32 state element above the FP16 range must decay to its FP32 value without NaNs."""
    if dtype is torch.float16:
        request.applymarker(_FP16_STATE_RANGE)
    elif family == "kda" and kernel_options is not None:
        request.applymarker(_MEGA_KDA_CARRY)
    inputs = _zero_inputs(family, dtype, tokens=1, gate_value=-4.0)
    initial_state = torch.zeros(1, 1, 128, 128, device="cuda")
    initial_state[0, 0, 0, 0] = 65536.0

    output, state = _run(family, kernel_options, inputs, initial_state)
    _, expected = _run(family, "ref", inputs, initial_state)

    assert torch.equal(output, torch.zeros_like(output))
    assert expected[0, 0, 0, 0].item() == pytest.approx(65536.0 * math.exp(-4.0))
    _assert_fp32_carry(state, expected, "final_state")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("kernel_options", _BACKENDS)
@pytest.mark.parametrize("family", _FAMILIES)
def test_tiny_final_state_cotangent_reaches_gate_gradient(
    family: str, kernel_options: dict[str, str] | None, dtype: torch.dtype, request
) -> None:
    """A final-state cotangent below the FP16 range must still produce the gate gradient."""
    if dtype is torch.float16:
        request.applymarker(_FP16_COTANGENT_RANGE)
    inputs = _zero_inputs(family, dtype, tokens=64, gate_value=-0.01)
    initial_state = torch.ones(1, 1, 128, 128, device="cuda")
    d_final_state = torch.full_like(initial_state, 2.0**-25)

    def gradients(options):
        gate = inputs[3].clone().requires_grad_(True)
        state_leaf = initial_state.clone().requires_grad_(True)
        _, state = _run(family, options, (*inputs[:3], gate, inputs[4]), state_leaf)
        return torch.autograd.grad((state,), (gate, state_leaf), (d_final_state,))

    d_gate, d_initial_state = gradients(kernel_options)
    expected_d_gate, expected_d_initial_state = gradients("ref")

    assert (expected_d_gate != 0).all()
    # The gate term reads the Q/K/V-dtype state checkpoint, so it carries source-dtype rounding.
    assert_matches_low_precision_reference(
        d_gate, expected_d_gate, expected_d_gate, "dgate", source_dtype=dtype
    )
    assert_relative_rms_within(d_gate, expected_d_gate, "dgate", max_eps=1.25, source_dtype=dtype)
    _assert_fp32_carry(d_initial_state, expected_d_initial_state, "d_initial_state")
