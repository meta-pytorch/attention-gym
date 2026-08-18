# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for the shipped bounded-gate forward and backward paths."""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("cutlass")
triton = pytest.importorskip("triton")

from attn_gym._backends.cute import tensor_supports_tma
from attn_gym.linear.kda.bwd.cute.gate_bwd_fused import (
    FusedGateBwdOp,
    FusedGateBwdOutput,
    _compile_fused_gate_bwd,
    _fused_gate_bwd_op,
    fused_gate_bwd,
)
from attn_gym.linear.kda.fwd.triton.gate_fwd import (
    MAX_GATE_LOWER_BOUND_MAGNITUDE,
    bounded_gate_cumsum,
)
from attn_gym.linear.kda.naive import fused_gate_bwd_ref

TMA_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)

pytestmark = pytest.mark.skipif(
    not TMA_AVAILABLE,
    reason="the fused CuTeDSL KDA kernel requires TMA on CUDA capability 9.0 or newer",
)


def _inputs(tokens: int, heads: int = 16, head_dim: int = 128, batch: int = 2):
    """Create nonuniform production-dtype inputs from one deterministic seed."""
    torch.manual_seed(1234 + batch + tokens + head_dim)
    g = (0.5 * torch.randn(batch, tokens, heads, head_dim, device="cuda")).to(torch.bfloat16)
    A_log = 0.25 * torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = 0.25 * torch.randn(heads, head_dim, device="cuda", dtype=torch.float32)
    d_cumulative = torch.randn(batch, tokens, heads, head_dim, device="cuda", dtype=torch.float32)
    return g, A_log, dt_bias, d_cumulative


def _forward_reference(g, A_log, dt_bias, lower_bound=-5.0, chunk_size=64):
    """Apply the shipped bounded gate and chunk-local prefix sum in PyTorch."""
    gate_input = g.float() + dt_bias
    gate = lower_bound * torch.sigmoid(A_log.exp().view(1, 1, -1, 1) * gate_input)
    return torch.cat(
        [chunk.cumsum(1) * math.log2(math.e) for chunk in gate.split(chunk_size, dim=1)],
        dim=1,
    )


def _autograd_reference(
    g,
    A_log,
    dt_bias,
    d_cumulative,
    lower_bound=-5.0,
    chunk_size=64,
):
    """Differentiate the unfused forward gate map and chunked prefix sum."""
    g_ref = g.float().detach().requires_grad_()
    A_log_ref = A_log.detach().requires_grad_()
    dt_bias_ref = dt_bias.detach().requires_grad_()
    cumulative = _forward_reference(
        g_ref,
        A_log_ref,
        dt_bias_ref,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
    )
    return torch.autograd.grad(
        cumulative,
        (g_ref, A_log_ref, dt_bias_ref),
        d_cumulative,
    )


def _fused_reference(inputs, lower_bound=-5.0, chunk_size=64):
    """Run the reusable batched PyTorch reference."""
    g, A_log, dt_bias, d_cumulative = inputs
    return fused_gate_bwd_ref(
        g,
        A_log,
        dt_bias,
        d_cumulative,
        lower_bound,
        math.log2(math.e),
        chunk_size,
    )


def _expected_output(inputs, lower_bound=-5.0, chunk_size=64):
    """Build the kernel output ABI and the final dA_log from the canonical reference."""
    dg, dA_log, d_dt_bias = _fused_reference(inputs, lower_bound, chunk_size)
    z = inputs[0].float() + inputs[2]
    dA_partial = torch.stack(
        [
            (dg_chunk * z_chunk).sum((1, 3))
            for dg_chunk, z_chunk in zip(
                dg.split(chunk_size, dim=1),
                z.split(chunk_size, dim=1),
                strict=True,
            )
        ],
        dim=1,
    )
    return FusedGateBwdOutput(dg, dA_partial, d_dt_bias), dA_log


def _assert_bf16_within_one_ulp(actual, expected):
    """Require BF16 ``actual`` to be the rounded FP32 reference or an adjacent BF16 value."""
    assert actual.dtype == torch.bfloat16
    rounded = expected.to(torch.bfloat16)
    assert actual.isfinite().all() and rounded.isfinite().all()
    lower = torch.nextafter(rounded, torch.full_like(rounded, -torch.inf))
    upper = torch.nextafter(rounded, torch.full_like(rounded, torch.inf))
    assert ((actual >= lower) & (actual <= upper)).all()


def _assert_output_close(actual, expected):
    """Compare the gate gradient and both parameter gradients."""
    _assert_bf16_within_one_ulp(actual.dg, expected.dg)
    torch.testing.assert_close(
        actual.dA_partial,
        expected.dA_partial,
        rtol=5e-5,
        atol=5e-4,
    )
    torch.testing.assert_close(actual.d_dt_bias, expected.d_dt_bias, rtol=5e-5, atol=7e-4)


@pytest.fixture(scope="module", autouse=True)
def isolated_cute_cache(tmp_path_factory):
    """Keep compiled test artifacts out of the user cache."""
    cache_dir = tmp_path_factory.mktemp("kda-fused-gate-cache")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(cache_dir))
        _compile_fused_gate_bwd.cache_clear()
        yield cache_dir
        _compile_fused_gate_bwd.cache_clear()


@pytest.mark.parametrize(
    ("chunk_size", "head_dim", "schedule"),
    (
        (6, 32, (2, 2)),
        (7, 32, (1, 1)),
        (7, 128, (1, 2)),
        (64, 128, (4, 2)),
    ),
)
def test_tma_schedule_preserves_chunk_boundaries(chunk_size, head_dim, schedule):
    """Keep the aligned two-stage and misaligned one-stage regression cases."""
    op = FusedGateBwdOp(
        heads=16,
        head_dim=head_dim,
        chunk_size=chunk_size,
        lower_bound=-5.0,
        fastmath=False,
    )
    assert (op.tokens_per_stage, op.stages) == schedule


@pytest.mark.parametrize("chunk_size", (7, 64))
def test_fused_gate_bwd_reference_matches_forward_autograd(chunk_size):
    """Check the reusable reference against direct differentiation of the forward graph."""
    inputs = _inputs(17, head_dim=64)
    expected = _autograd_reference(*inputs, lower_bound=-3.25, chunk_size=chunk_size)
    actual = _fused_reference(inputs, lower_bound=-3.25, chunk_size=chunk_size)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-5, atol=3e-5)


def test_bounded_gate_cumsum_rejects_gate_range_beyond_rebase_budget():
    """Reject decay the intra-chunk rebase cannot represent instead of returning NaNs.

    Past the ceiling the core silently produced non-finite output, so the boundary has to
    fail fast. The ceiling is derived from the subchunk span and the FP32 exponent range;
    see NOTE [Gate range ceiling].
    """
    raw_gate = torch.zeros(1, 64, 2, 128, dtype=torch.bfloat16, device="cuda")
    A_log = torch.zeros(2, dtype=torch.float32, device="cuda")
    dt_bias = torch.zeros(2, 128, dtype=torch.float32, device="cuda")
    ceiling = MAX_GATE_LOWER_BOUND_MAGNITUDE

    # The shipped default must stay comfortably inside the supported range.
    bounded_gate_cumsum(raw_gate, A_log, dt_bias, lower_bound=-5.0)
    bounded_gate_cumsum(raw_gate, A_log, dt_bias, lower_bound=-(ceiling - 1e-3))

    # One chained comparison covers every rejection reason, so they share a message:
    # past the ceiling, non-finite, and positive.
    rejected_values = (
        -(ceiling + 1e-3),
        -16.0,
        -32.0,
        float("nan"),
        float("-inf"),
        1.0,
    )
    for rejected in rejected_values:
        with pytest.raises(ValueError, match="lower_bound must lie in"):
            bounded_gate_cumsum(raw_gate, A_log, dt_bias, lower_bound=rejected)


@pytest.mark.parametrize(
    "shape,match",
    [
        ((1, 17, 2, 16), "multiple of 32"),
        ((0, 17, 2, 32), "nonempty B, T, and H"),
        ((1, 0, 2, 32), "nonempty B, T, and H"),
        ((1, 17, 0, 32), "nonempty B, T, and H"),
    ],
)
def test_bounded_gate_cumsum_rejects_backward_unsupported_shape(shape, match):
    raw_gate = torch.empty(shape, dtype=torch.bfloat16, device="cuda")
    A_log = torch.empty(shape[2], dtype=torch.float32, device="cuda")
    dt_bias = torch.empty(shape[2:], dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match=match):
        bounded_gate_cumsum(raw_gate, A_log, dt_bias)


@pytest.mark.parametrize(
    ("batch", "tokens", "head_dim", "chunk_size"),
    ((2, 65, 128, 64), (1, 17, 32, 8)),
)
def test_bounded_gate_cumsum_autograd_boundary(batch, tokens, head_dim, chunk_size):
    """Exercise batched, strided, tail, and nondefault public gate routes."""
    g, A_log, dt_bias, d_cumulative = _inputs(
        tokens,
        heads=2,
        head_dim=head_dim,
        batch=batch,
    )
    g_storage = torch.empty(*g.shape[:-1], 2 * g.shape[-1], dtype=g.dtype, device="cuda")
    A_storage = torch.empty(2 * A_log.shape[0], dtype=A_log.dtype, device="cuda")
    bias_storage = torch.empty(
        *dt_bias.shape[:-1], 2 * dt_bias.shape[-1], dtype=dt_bias.dtype, device="cuda"
    )
    g = g_storage[..., ::2].copy_(g).requires_grad_()
    A_log = A_storage[::2].copy_(A_log).requires_grad_()
    dt_bias = bias_storage[..., ::2].copy_(dt_bias).requires_grad_()
    inputs = (g, A_log, dt_bias)
    assert not all(tensor.is_contiguous() for tensor in inputs)
    output = bounded_gate_cumsum(*inputs, chunk_size=chunk_size, lower_bound=-3.25)
    expected_output = _forward_reference(
        g,
        A_log,
        dt_bias,
        lower_bound=-3.25,
        chunk_size=chunk_size,
    )
    actual = torch.autograd.grad(output, inputs, d_cumulative)
    expected = _autograd_reference(
        g,
        A_log,
        dt_bias,
        d_cumulative,
        lower_bound=-3.25,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(output, expected_output, rtol=2e-5, atol=2e-5)
    _assert_bf16_within_one_ulp(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=5e-5, atol=5e-4)
    torch.testing.assert_close(actual[2], expected[2], rtol=5e-5, atol=7e-4)


def test_bounded_gate_cumsum_compile_matches_eager():
    """Keep compiled gate forward and backward equivalent to eager execution."""
    g, A_log, dt_bias, d_cumulative = _inputs(1024, heads=2, batch=1)
    expected_inputs = tuple(tensor.detach().requires_grad_() for tensor in (g, A_log, dt_bias))
    actual_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in expected_inputs)

    def operation(g, A_log, dt_bias):
        return bounded_gate_cumsum(g, A_log, dt_bias, lower_bound=-3.25)

    expected = operation(*expected_inputs)
    actual = torch.compile(operation, fullgraph=True)(*actual_inputs)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, d_cumulative)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, d_cumulative)

    # The gate gradient leaves the kernel unchanged, while Inductor may reduce the
    # parameter gradients in a different order than eager.
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(
        actual_gradients[1:], expected_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-5, atol=7e-4)


@pytest.mark.parametrize(
    ("tokens", "head_dim", "lower_bound", "chunk_size"),
    (
        (64, 128, -5.0, 64),
        (65, 128, -5.0, 64),
        (17, 32, -2.5, 1),
        (17, 32, -2.5, 2),
        (17, 32, -3.25, 7),
        (17, 64, -3.25, 7),
        (129, 64, -3.25, 128),
        (65, 1024, -4.0, 64),
    ),
)
def test_fused_gate_bwd_matches_reference(tokens, head_dim, lower_bound, chunk_size):
    """Cover TMA tails and independent static semantic specializations."""
    inputs = _inputs(tokens, head_dim=head_dim)
    expected, expected_dA = _expected_output(inputs, lower_bound, chunk_size)

    actual = fused_gate_bwd(*inputs, lower_bound=lower_bound, chunk_size=chunk_size)

    _assert_output_close(actual, expected)
    torch.testing.assert_close(actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4)


def test_fused_gate_bwd_fastmath_specialization_is_correct():
    """Keep the non-default exponential mode numerically valid."""
    inputs = _inputs(65, heads=17)
    expected, expected_dA = _expected_output(inputs, -4.75)

    actual = fused_gate_bwd(*inputs, lower_bound=-4.75, fastmath=True)

    _assert_output_close(actual, expected)
    torch.testing.assert_close(actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4)


def test_fused_gate_bwd_op_registration():
    """Exercise a non-default TMA schedule through the private operator schema."""
    inputs = _inputs(65, head_dim=64)
    torch.library.opcheck(
        _fused_gate_bwd_op,
        (*inputs, 7, -3.25, False),
    )


def test_fused_gate_bwd_requires_batch_dimension():
    """Keep batch explicit at the public kernel boundary."""
    g, A_log, dt_bias, d_cumulative = _inputs(17, head_dim=64)
    with pytest.raises(ValueError, match=r"shape \(B, T, H, D\)"):
        fused_gate_bwd(g[0], A_log, dt_bias, d_cumulative[0])


def test_fused_gate_bwd_reads_aligned_strided_rows_directly():
    """Avoid compacting packed projections whose trailing rows satisfy TMA alignment."""
    batch, tokens, heads, head_dim = 2, 17, 16, 64
    torch.manual_seed(10)
    g = torch.randn(batch, tokens, 3, heads, head_dim, device="cuda", dtype=torch.bfloat16)[
        :, :, 0
    ]
    d_cumulative = torch.randn(
        batch, tokens, 2, heads, head_dim, device="cuda", dtype=torch.float32
    )[:, :, 0]
    A_log = torch.randn(2 * heads, device="cuda", dtype=torch.float32)[::2]
    dt_bias = torch.randn(heads, 2 * head_dim, device="cuda", dtype=torch.float32)[:, ::2]
    inputs = (g, A_log, dt_bias, d_cumulative)
    assert tensor_supports_tma(g)
    assert tensor_supports_tma(d_cumulative)
    assert not A_log.is_contiguous()
    assert not dt_bias.is_contiguous()

    expected_dg, expected_dA, expected_dt_bias = _fused_reference(inputs, -3.25, 7)
    compiled = torch.compile(fused_gate_bwd, fullgraph=True, dynamic=True)
    actual = compiled(*inputs, lower_bound=-3.25, chunk_size=7)
    _assert_bf16_within_one_ulp(actual.dg, expected_dg)
    torch.testing.assert_close(actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4)
    torch.testing.assert_close(actual.d_dt_bias, expected_dt_bias, rtol=5e-5, atol=7e-4)


def test_fused_gate_bwd_compacts_unsupported_inner_stride():
    """Retain a safe fallback when the feature dimension is not contiguous."""
    _g, A_log, dt_bias, d_cumulative = _inputs(17, head_dim=64)
    g = torch.randn(2, 17, 16, 128, device="cuda", dtype=torch.bfloat16)[..., ::2]
    assert not tensor_supports_tma(g)

    inputs = (g, A_log, dt_bias, d_cumulative)
    expected_dg, expected_dA, expected_dt_bias = _fused_reference(inputs, -3.25, 7)
    actual = fused_gate_bwd(*inputs, lower_bound=-3.25, chunk_size=7)
    _assert_bf16_within_one_ulp(actual.dg, expected_dg)
    torch.testing.assert_close(actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4)
    torch.testing.assert_close(actual.d_dt_bias, expected_dt_bias, rtol=5e-5, atol=7e-4)


def test_fused_gate_bwd_fullgraph_dynamic():
    """Run strict dynamic fullgraph capture across changing batch and output extents."""
    compiled = torch.compile(fused_gate_bwd, fullgraph=True, dynamic=True)
    for batch, tokens in ((1, 63), (3, 65)):
        inputs = _inputs(tokens, head_dim=64, batch=batch)
        expected_dg, expected_dA, expected_dt_bias = _fused_reference(inputs, -3.25, 7)
        actual = compiled(
            *inputs,
            lower_bound=-3.25,
            chunk_size=7,
            fastmath=False,
        )
        _assert_bf16_within_one_ulp(actual.dg, expected_dg)
        torch.testing.assert_close(
            actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4
        )
        torch.testing.assert_close(actual.d_dt_bias, expected_dt_bias, rtol=5e-5, atol=7e-4)


def test_fused_gate_bwd_cuda_graph_replay():
    """Capture only after compilation and reuse the static output buffers on replay."""
    inputs = _inputs(65, head_dim=64)
    fused_gate_bwd(*inputs, lower_bound=-3.25, chunk_size=7)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_gate_bwd(*inputs, lower_bound=-3.25, chunk_size=7)
    captured_dg = actual.dg.clone()

    inputs[-1].mul_(-0.5).add_(0.25)
    expected_dg, expected_dA, expected_dt_bias = _fused_reference(inputs, -3.25, 7)
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(actual.dg, captured_dg)
    _assert_bf16_within_one_ulp(actual.dg, expected_dg)
    torch.testing.assert_close(actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4)
    torch.testing.assert_close(actual.d_dt_bias, expected_dt_bias, rtol=5e-5, atol=7e-4)


def test_fused_gate_bwd_rejects_invalid_static_scalars():
    """Reject invalid compile-time semantics before entering the compiler."""
    inputs = _inputs(17)
    for chunk_size in (0, True):
        with pytest.raises(ValueError, match="positive int"):
            fused_gate_bwd(*inputs, chunk_size=chunk_size)
    with pytest.raises(ValueError, match="finite"):
        fused_gate_bwd(*inputs, lower_bound=float("nan"))
    with pytest.raises(TypeError, match="fastmath must be bool"):
        fused_gate_bwd(*inputs, fastmath=1)


def test_fused_gate_bwd_rejects_higher_order_autograd():
    """Make the low-level backward leaf's differentiation contract explicit."""
    g, A_log, dt_bias, d_cumulative = _inputs(17)
    A_log.requires_grad_()
    with pytest.raises(RuntimeError, match="higher-order autograd"):
        fused_gate_bwd(g, A_log, dt_bias, d_cumulative)
