"""Per-token KDA gate semantics and internal chunk-scan coverage."""

from __future__ import annotations

import math
from functools import partial

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

from attn_gym.linear.kda import bound_gate, chunk_kda
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.constants import LOG2_E, MAX_GATE_LOWER_BOUND_MAGNITUDE
from attn_gym.linear.kda.naive import chunk_cumsum_ref
from attn_gym.linear.kda.ops import (
    _bound_gate_bwd_op,
    _bound_gate_fwd_op,
    _plain_gate_scan_op,
)
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA gate ops require CUDA")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_bound_gate_returns_fp32_natural_log_decay(dtype: torch.dtype):
    """Pin the example gate formula and dtype independently of input precision."""
    torch.manual_seed(3)
    raw_gate = torch.randn(1, 17, 2, 128, device="cuda", dtype=dtype)
    a_log = torch.randn(2, device="cuda")
    dt_bias = torch.randn(2, 128, device="cuda")

    actual = bound_gate(raw_gate, a_log, dt_bias, lower_bound=-3.25, impl="reference")
    expected = -3.25 * torch.sigmoid(a_log.exp().view(1, 1, 2, 1) * (raw_gate.float() + dt_bias))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    assert torch.isfinite(
        bound_gate(raw_gate, a_log, dt_bias, lower_bound=-6.0, impl="reference")
    ).all()
    for invalid_lower_bound in (1.0, float("-inf")):
        with pytest.raises(ValueError, match="lower_bound"):
            bound_gate(raw_gate, a_log, dt_bias, lower_bound=invalid_lower_bound)


def test_bound_gate_defaults_to_fused():
    torch.manual_seed(11)
    inputs = (
        torch.randn(1, 17, 2, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(2, device="cuda"),
        torch.randn(2, 128, device="cuda"),
    )
    actual = bound_gate(*inputs)
    expected = bound_gate(*inputs, impl="fused")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_bound_gate_fullgraph_backward():
    """Compile the example pointwise transform and preserve all three gradients."""
    torch.manual_seed(13)
    inputs = (
        torch.randn(1, 31, 2, 128, device="cuda", dtype=torch.bfloat16),
        torch.randn(2, device="cuda"),
        torch.randn(2, 128, device="cuda"),
    )
    expected_inputs = tuple(value.requires_grad_() for value in inputs)
    actual_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    expected = bound_gate(*expected_inputs, lower_bound=-3.25, impl="reference")
    actual = torch.compile(partial(bound_gate, impl="reference"), fullgraph=True)(
        *actual_inputs, lower_bound=-3.25
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    cotangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
    for index, (actual_gradient, expected_gradient) in enumerate(
        zip(actual_gradients, expected_gradients, strict=True)
    ):
        tolerance = {"rtol": 1e-2, "atol": 5e-4} if index == 0 else {"rtol": 1e-4, "atol": 1e-4}
        torch.testing.assert_close(actual_gradient, expected_gradient, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fastmath", [False, True])
def test_bound_gate_fused_backward_matches_pytorch(
    dtype: torch.dtype,
    fastmath: bool,
):
    """Match optimized forward/backward on partial token and head groups."""
    torch.manual_seed(23)
    inputs = (
        torch.randn(2, 65, 3, 128, device="cuda", dtype=dtype, requires_grad=True),
        torch.randn(3, device="cuda", requires_grad=True),
        torch.randn(3, 128, device="cuda", requires_grad=True),
    )
    actual_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    expected = bound_gate(*inputs, lower_bound=-3.25, impl="reference")
    actual = bound_gate(
        *actual_inputs,
        lower_bound=-3.25,
        fastmath=fastmath,
        impl="fused",
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    cotangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
    tolerances = (
        {"rtol": 1e-2, "atol": 8e-3},
        {"rtol": 3e-3, "atol": 5e-4},
        {"rtol": 3e-3, "atol": 5e-5},
    )
    for actual_gradient, expected_gradient, tolerance in zip(
        actual_gradients, expected_gradients, tolerances, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, **tolerance)


def test_bound_gate_fused_backward_fullgraph_dynamic_tokens():
    """Reuse one fullgraph callable across batch and partial-token shapes."""
    torch.compiler.reset()
    torch.manual_seed(29)
    with torch._dynamo.config.patch(error_on_recompile=True):
        compiled = torch.compile(bound_gate, fullgraph=True, dynamic=True)
        for batch, tokens in ((2, 65), (3, 97)):
            expected_inputs = (
                torch.randn(
                    batch,
                    tokens,
                    3,
                    128,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                ),
                torch.randn(3, device="cuda", requires_grad=True),
                torch.randn(3, 128, device="cuda", requires_grad=True),
            )
            actual_inputs = tuple(
                value.detach().clone().requires_grad_() for value in expected_inputs
            )
            expected = bound_gate(*expected_inputs, impl="reference")
            actual = compiled(*actual_inputs)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

            cotangent = torch.randn_like(expected)
            expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
            actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
            for actual_gradient, expected_gradient in zip(
                actual_gradients, expected_gradients, strict=True
            ):
                torch.testing.assert_close(
                    actual_gradient.float(),
                    expected_gradient.float(),
                    rtol=3e-3,
                    atol=8e-3,
                )


@pytest.mark.parametrize("layout", ["last_dim_strided", "outer_strided", "misaligned"])
def test_bound_gate_supports_general_layouts(layout: str):
    """Support dynamic outer strides and normalize only a noncontiguous last mode."""
    torch.manual_seed(37)
    shape = (1, 33, 3, 128)
    match layout:
        case "last_dim_strided":
            raw_gate = torch.randn(1, 33, 3, 256, device="cuda", dtype=torch.bfloat16)[..., ::2]
            dt_bias = torch.randn(3, 256, device="cuda")[..., ::2]
            a_log = torch.randn(3, device="cuda")
        case "outer_strided":
            raw_gate = torch.randn(1, 33, 3, 2, 128, device="cuda", dtype=torch.bfloat16)[
                ..., 0, :
            ]
            dt_bias = torch.randn(3, 2, 128, device="cuda")[:, 0, :]
            a_log = torch.randn(3, device="cuda")
            assert raw_gate.stride(-1) == 1 and not raw_gate.is_contiguous()
            assert dt_bias.stride(-1) == 1 and not dt_bias.is_contiguous()
        case "misaligned":
            raw_storage = torch.randn(math.prod(shape) + 1, device="cuda", dtype=torch.bfloat16)
            raw_gate = raw_storage[1:].view(shape)
            bias_storage = torch.randn(3 * 128 + 1, device="cuda")
            dt_bias = bias_storage[1:].view(3, 128)
            a_log_storage = torch.randn(4, device="cuda")
            a_log = a_log_storage[1:]
            assert raw_gate.is_contiguous() and raw_gate.data_ptr() % 16
            assert dt_bias.is_contiguous() and dt_bias.data_ptr() % 16
            assert a_log.is_contiguous() and a_log.data_ptr() % 16
    expected_inputs = tuple(
        value.detach().clone().requires_grad_() for value in (raw_gate, a_log, dt_bias)
    )
    actual_inputs = tuple(value.requires_grad_() for value in (raw_gate, a_log, dt_bias))
    expected = bound_gate(*expected_inputs, impl="reference")
    actual = bound_gate(*actual_inputs, fastmath=True, impl="fused")
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    if layout == "last_dim_strided":
        cotangent = torch.randn(1, 33, 3, 256, device="cuda")[..., ::2]
    elif layout == "outer_strided":
        cotangent = torch.randn(1, 33, 3, 2, 128, device="cuda")[:, :, :, 0, :]
    else:
        cotangent_storage = torch.randn(expected.numel() + 1, device="cuda")
        cotangent = cotangent_storage[1:].view_as(expected)
        assert cotangent.is_contiguous() and cotangent.data_ptr() % 16
    expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient.float(), expected_gradient.float(), rtol=3e-3, atol=3e-1
        )


@pytest.mark.parametrize(("batch", "tokens"), [(1, 65536), (65536, 1)])
def test_bound_gate_supports_large_grid_axes(batch: int, tokens: int):
    """Avoid CUDA grid y/z limits for long sequences and large batches."""
    torch.manual_seed(41)
    inputs = (
        torch.randn(
            batch, tokens, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        ),
        torch.randn(1, device="cuda", requires_grad=True),
        torch.randn(1, 128, device="cuda", requires_grad=True),
    )
    expected_inputs = tuple(value.detach().clone().requires_grad_() for value in inputs)
    actual = bound_gate(*inputs, impl="fused")
    expected = bound_gate(*expected_inputs, impl="reference")
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient.float(), expected_gradient.float(), rtol=3e-3, atol=3e-1
        )


def test_bound_gate_operator_registration():
    """Validate both private operators' schema and fake tensor contracts."""
    torch.manual_seed(31)
    shape = (1, 65, 2, 128)
    raw_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    a_log = torch.randn(2, device="cuda")
    dt_bias = torch.randn(2, 128, device="cuda")
    d_gate = torch.randn(shape, device="cuda")
    utilities = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    torch.library.opcheck(
        _bound_gate_fwd_op,
        (raw_gate, a_log, dt_bias, -5.0, False),
        test_utils=utilities,
    )
    torch.library.opcheck(
        _bound_gate_bwd_op,
        (raw_gate, a_log, dt_bias, d_gate, -5.0, False),
        test_utils=utilities,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the fused chunk KDA core requires CUDA capability 10.0 or newer",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_chunk_gate_range_boundary_is_finite(dtype):
    """Exercise the strongest documented per-token decay accepted by the fused rebase."""
    torch.manual_seed(17)
    shape = (1, 64, 1, 128)
    inputs = (
        F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype),
        F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype),
        torch.randn(shape, device="cuda", dtype=dtype),
        torch.full(shape, -MAX_GATE_LOWER_BOUND_MAGNITUDE, device="cuda"),
        torch.rand(shape[:3], device="cuda"),
    )
    inputs = tuple(tensor.requires_grad_() for tensor in inputs)

    output, _ = chunk_kda(*inputs, autotune=False)
    gradients = torch.autograd.grad(output.float().square().mean(), inputs)
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_plain_gate_scan_accepts_strided_dense_input():
    """Read arbitrary strides and return the compact private representation."""
    torch.manual_seed(5)
    storage = torch.empty(1, 128, 2, 256, device="cuda").uniform_(0.5, 1.0).log()
    gate = storage[..., ::2]
    assert not gate.is_contiguous()

    actual = _plain_gate_scan_op(gate, None, None, False)
    expected = chunk_cumsum_ref(gate, 64, scale=LOG2_E)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    cotangent = torch.randn_like(gate)
    actual_gradient = _plain_gate_scan_op(cotangent, None, None, True)
    expected_gradient = chunk_cumsum_ref(cotangent, 64, reverse=True, scale=LOG2_E)
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-5)


def test_plain_gate_scan_resets_packed_boundaries_and_zeros_inactive_gradient():
    """Reset packed scans and keep inactive-capacity gradients out of reductions."""
    lengths = [65, 0, 63]
    active_tokens = sum(lengths)
    capacity = active_tokens + 17
    offsets = cumulative_sequence_offsets(lengths)
    metadata = prepare_ragged_chunk_metadata(offsets, capacity, 64)
    torch.manual_seed(7)
    gate = torch.empty(1, capacity, 2, 128, device="cuda").uniform_(0.5, 1.0).log()

    actual = _plain_gate_scan_op(gate, offsets, metadata.chunk_offsets, False)
    expected = chunk_cumsum_ref(
        gate[:, :active_tokens],
        64,
        scale=LOG2_E,
        cu_seqlens=offsets,
    )
    torch.testing.assert_close(actual[:, :active_tokens], expected, rtol=2e-5, atol=2e-5)

    cotangent = torch.randn_like(gate)
    actual_gradient = _plain_gate_scan_op(
        cotangent,
        offsets,
        metadata.chunk_offsets,
        True,
    )
    expected_gradient = chunk_cumsum_ref(
        cotangent[:, :active_tokens],
        64,
        reverse=True,
        scale=LOG2_E,
        cu_seqlens=offsets,
    )
    torch.testing.assert_close(
        actual_gradient[:, :active_tokens], expected_gradient, rtol=2e-5, atol=2e-5
    )
    assert not actual_gradient[:, active_tokens:].any()


def test_plain_gate_scan_registration():
    """Validate the unified dense/ragged and forward/reverse operator contract."""
    torch.manual_seed(11)
    gate = torch.empty(1, 64, 1, 128, device="cuda").uniform_(0.5, 1.0).log()
    offsets = cumulative_sequence_offsets([33, 31])
    metadata = prepare_ragged_chunk_metadata(offsets, gate.shape[1], 64)
    cases = (
        (gate, None, None, False),
        (gate, None, None, True),
        (gate, offsets, metadata.chunk_offsets, False),
        (gate, offsets, metadata.chunk_offsets, True),
    )
    for args in cases:
        torch.library.opcheck(
            _plain_gate_scan_op,
            args,
            test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
            rtol=2e-5,
            atol=2e-5,
        )
