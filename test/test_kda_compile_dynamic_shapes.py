"""Strict compilation regressions for dynamic KDA shapes and scalar arguments.

One compiled callable must serve varying packed sequence counts, head geometry, and
``lower_bound`` values with eager-parity outputs; shape-derived Triton launch arguments
must stay representable once Dynamo promotes dimensions to dynamic.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton")

from attn_gym.linear import bounded_gate_cumsum, chunk_kda
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TOKENS = 128


def _gate_inputs(batch: int = 1, heads: int = 2, head_dim: int = 128):
    """Create valid dense bounded-gate operands for one shape."""
    torch.manual_seed(0)
    raw = (0.5 * torch.randn(batch, TOKENS, heads, head_dim, device="cuda")).to(torch.bfloat16)
    a_log = 0.25 * torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = 0.25 * torch.randn(heads, head_dim, device="cuda", dtype=torch.float32)
    return raw, a_log, dt_bias


def _offsets(sequence_count: int) -> torch.Tensor:
    """Split the token capacity into near-equal packed sequence boundaries."""
    step = TOKENS // sequence_count
    lengths = [step] * (sequence_count - 1) + [TOKENS - step * (sequence_count - 1)]
    return cumulative_sequence_offsets(lengths)


def test_compiled_gate_accepts_varying_packed_sequence_count():
    """One compiled callable must serve differing packed sequence counts."""
    raw, a_log, dt_bias = _gate_inputs()
    compiled = torch.compile(bounded_gate_cumsum, fullgraph=True, dynamic=True)
    for index, sequence_count in enumerate((3, 4, 5)):
        cu_seqlens = _offsets(sequence_count)
        expected = bounded_gate_cumsum(raw, a_log, dt_bias, cu_seqlens=cu_seqlens)
        # The dynamic graph from earlier calls must be reused, not respecialized.
        with torch._dynamo.config.patch(error_on_recompile=index > 1):
            output = compiled(raw, a_log, dt_bias, cu_seqlens=cu_seqlens)
        torch.testing.assert_close(output, expected)


def test_compiled_gate_accepts_varying_head_geometry():
    """One compiled callable must serve differing batch, head count, and head dim."""
    compiled = torch.compile(bounded_gate_cumsum, fullgraph=True, dynamic=True)
    for batch, heads, head_dim in ((1, 2, 128), (2, 3, 96), (1, 5, 64)):
        raw, a_log, dt_bias = _gate_inputs(batch=batch, heads=heads, head_dim=head_dim)
        expected = bounded_gate_cumsum(raw, a_log, dt_bias)
        output = compiled(raw, a_log, dt_bias)
        torch.testing.assert_close(output, expected)


def test_compiled_gate_backward_survives_varying_lower_bound():
    """Varying the lower_bound float must keep backward compiling with eager parity.

    The values only need to be distinct so Dynamo stops specializing on the float; they
    were -10.0 and -5.0 until the causal gate reference capped the supported range at
    about -5.915. See NOTE [Gate range ceiling].
    """
    raw, a_log, dt_bias = _gate_inputs()
    compiled = torch.compile(bounded_gate_cumsum, fullgraph=True, dynamic=True)
    for lower_bound in (-5.0, -3.25):
        expected = bounded_gate_cumsum(raw, a_log, dt_bias, lower_bound=lower_bound)
        output = compiled(raw, a_log, dt_bias, lower_bound=lower_bound)
        torch.testing.assert_close(output, expected)

    expected_inputs = tuple(tensor.detach().requires_grad_() for tensor in (raw, a_log, dt_bias))
    actual_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in expected_inputs)
    expected = bounded_gate_cumsum(*expected_inputs, lower_bound=-5.0)
    actual = compiled(*actual_inputs, lower_bound=-5.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    cotangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=0, atol=0)
    # A_log and dt_bias gradients reduce over tokens; compiled reduction order differs.
    for actual_gradient, expected_gradient in zip(
        actual_gradients[1:], expected_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-5, atol=7e-4)


def test_compiled_chunk_kda_accepts_varying_packed_sequence_count():
    """Varying sequence counts through the composed core, which shares the scheduler."""
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("the CuTe KDA core requires CUDA capability 10.0 or newer")
    inputs = make_kda_test_inputs(TOKENS)
    compiled = torch.compile(chunk_kda, fullgraph=True, dynamic=True)
    for sequence_count in (2, 4):
        cu_seqlens = _offsets(sequence_count)
        expected, expected_state = chunk_kda(*inputs, cu_seqlens=cu_seqlens)
        actual, actual_state = compiled(*inputs, cu_seqlens=cu_seqlens)
        assert expected_state is actual_state is None
        torch.testing.assert_close(actual, expected)
