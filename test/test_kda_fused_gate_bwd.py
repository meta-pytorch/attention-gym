# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for fused CuTeDSL KDA reverse-cumsum and gate backward."""

from __future__ import annotations

import math
from enum import IntEnum

import pytest
import torch

pytest.importorskip("cutlass")
triton = pytest.importorskip("triton")

from attn_gym.linear.kda.bwd.cute.gate_bwd_fused import (
    FusedGateBwdOp,
    FusedGateBwdOutput,
    WarpRole,
    _compile_fused_gate_bwd,
    _fused_gate_bwd_custom_op,
    fused_gate_bwd,
)
from attn_gym.linear.kda.bwd.triton.cumsum import chunk_local_cumsum_vector_kernel
from attn_gym.linear.kda.bwd.triton.gate_bwd import kda_gate_bwd_kernel
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
    z = g_ref + dt_bias_ref
    gate = lower_bound * torch.sigmoid(A_log_ref.exp().view(1, 1, -1, 1) * z)
    cumulative = torch.cat(
        [chunk.cumsum(1) * math.log2(math.e) for chunk in gate.split(chunk_size, dim=1)],
        dim=1,
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
    """Build the partial-output ABI from the canonical backward reference."""
    dg, dA_log, d_dt_bias = _fused_reference(inputs, lower_bound, chunk_size)
    z = inputs[0].float() + inputs[2]
    partials = torch.stack(
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
    return FusedGateBwdOutput(dg, partials), dA_log, d_dt_bias


def _assert_output_close(actual, expected):
    """Compare the two tensors produced by the partial-output ABI."""
    torch.testing.assert_close(actual.dg, expected.dg, rtol=3e-5, atol=4e-5)
    torch.testing.assert_close(
        actual.dA_partial,
        expected.dA_partial,
        rtol=5e-5,
        atol=5e-4,
    )


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
        (1, 128, (1, 1)),
        (2, 128, (2, 1)),
        (3, 128, (1, 2)),
        (6, 32, (2, 2)),
        (7, 128, (1, 2)),
        (7, 32, (1, 1)),
        (7, 96, (1, 1)),
        (32, 128, (4, 2)),
        (64, 128, (4, 2)),
        (128, 128, (4, 2)),
    ),
)
def test_tma_schedule_preserves_chunk_boundaries(chunk_size, head_dim, schedule):
    """Use the widest aligned staged tile that divides each static chunk."""
    op = FusedGateBwdOp(heads=16, head_dim=head_dim, chunk_size=chunk_size)
    assert (op.tokens_per_stage, op.stages) == schedule


def test_public_kernel_types_and_defaults():
    """Keep kernel construction and warp protocol names explicit and reusable."""
    op = FusedGateBwdOp(heads=16, head_dim=128)
    assert (op.chunk_size, op.lower_bound, op.fastmath) == (64, -5.0, False)
    assert issubclass(WarpRole, IntEnum)
    assert WarpRole.TMA_PRODUCER == 0


@pytest.mark.parametrize("chunk_size", (1, 2, 7, 32, 64, 128))
def test_fused_gate_bwd_reference_matches_forward_autograd(chunk_size):
    """Check the reusable reference against direct differentiation of the forward graph."""
    inputs = _inputs(17, head_dim=64)
    expected = _autograd_reference(*inputs, lower_bound=-3.25, chunk_size=chunk_size)
    actual = _fused_reference(inputs, lower_bound=-3.25, chunk_size=chunk_size)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-5, atol=3e-5)
    torch.testing.assert_close(actual[2], expected[2], rtol=2e-5, atol=3e-5)


@pytest.mark.parametrize(
    ("tokens", "head_dim", "lower_bound", "chunk_size"),
    (
        (1, 128, -5.0, 64),
        (63, 128, -5.0, 64),
        (64, 128, -5.0, 64),
        (65, 128, -5.0, 64),
        (65, 32, -2.5, 1),
        (65, 32, -2.5, 2),
        (17, 32, -3.25, 7),
        (17, 64, -3.25, 7),
        (65, 64, -3.25, 32),
        (129, 64, -3.25, 128),
        (65, 1024, -4.0, 64),
    ),
)
def test_fused_gate_bwd_matches_reference(tokens, head_dim, lower_bound, chunk_size):
    """Cover TMA tails and independent static semantic specializations."""
    inputs = _inputs(tokens, head_dim=head_dim)
    expected, expected_dA, expected_dt_bias = _expected_output(
        inputs,
        lower_bound,
        chunk_size,
    )

    actual = fused_gate_bwd(*inputs, lower_bound=lower_bound, chunk_size=chunk_size)

    _assert_output_close(actual, expected)
    torch.testing.assert_close(actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4)
    torch.testing.assert_close(actual.dg.sum((0, 1)), expected_dt_bias, rtol=5e-5, atol=7e-4)


@pytest.mark.parametrize("fastmath", (False, True))
def test_fused_gate_bwd_fastmath_specializations_are_correct(fastmath):
    """Keep both explicit exponential modes numerically valid."""
    inputs = _inputs(65, heads=17)
    expected, expected_dA, _expected_dt_bias = _expected_output(inputs, -4.75)

    actual = fused_gate_bwd(*inputs, lower_bound=-4.75, fastmath=fastmath)

    _assert_output_close(actual, expected)
    torch.testing.assert_close(actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4)


def test_fastmath_has_a_distinct_warm_cache_specialization(
    isolated_cute_cache,
    monkeypatch,
):
    """Encode the math mode in persistent artifacts and reuse a warm entry."""
    inputs = _inputs(33, heads=19)
    lower_bound = -4.625
    before = set(isolated_cute_cache.rglob("*.o"))

    outputs = [
        fused_gate_bwd(*inputs, lower_bound=lower_bound, fastmath=fastmath)
        for fastmath in (False, True)
    ]
    assert len(set(isolated_cute_cache.rglob("*.o")) - before) == 2
    _assert_output_close(outputs[0], _expected_output(inputs, lower_bound)[0])

    _compile_fused_gate_bwd.cache_clear()

    def reject_compiler_process(*_args, **_kwargs):
        raise AssertionError("warm launch unexpectedly started a compiler process")

    monkeypatch.setattr("attn_gym._backends.cute.compile.subprocess.run", reject_compiler_process)
    warm = fused_gate_bwd(*inputs, lower_bound=lower_bound, fastmath=False)
    torch.testing.assert_close(warm, outputs[0])


def test_fused_gate_bwd_matches_triton_composition():
    """Preserve the exact deployed reverse-cumsum plus gate-backward contract."""
    inputs = _inputs(65, batch=1)
    batched_g, A_log, dt_bias, batched_d_cumulative = inputs
    g = batched_g[0]
    d_cumulative = batched_d_cumulative[0]
    tokens, heads, head_dim = g.shape
    d_gate = torch.empty_like(d_cumulative)
    triton_dg = torch.empty_like(d_cumulative)
    cumsum_chunks = triton.cdiv(tokens, 64)
    gate_chunks = triton.cdiv(tokens, 32)
    triton_dA_partial = torch.empty(gate_chunks, heads, device="cuda", dtype=torch.float32)

    cumsum_kernel = chunk_local_cumsum_vector_kernel.fn.fn
    cumsum_kernel[(triton.cdiv(head_dim, 32), cumsum_chunks, heads)](
        d_cumulative,
        d_gate,
        math.log2(math.e),
        None,
        None,
        tokens,
        d_cumulative.stride(0),
        d_gate.stride(0),
        S_STRIDES=d_cumulative.stride(),
        O_STRIDES=d_gate.stride(),
        B=1,
        H=heads,
        S=head_dim,
        BT=64,
        BS=32,
        REVERSE=True,
        HAS_SCALE=True,
        IS_VARLEN=False,
        num_warps=4,
    )
    gate_kernel = kda_gate_bwd_kernel.fn.fn
    gate_kernel[(gate_chunks, heads)](
        g,
        A_log,
        dt_bias,
        d_gate,
        triton_dg,
        triton_dA_partial,
        -5.0,
        tokens,
        G_STRIDES=g.stride(),
        A_LOG_STRIDES=A_log.stride(),
        DT_BIAS_STRIDES=dt_bias.stride(),
        DYG_STRIDES=d_gate.stride(),
        DG_STRIDES=triton_dg.stride(),
        DA_STRIDES=triton_dA_partial.stride(),
        H=heads,
        D=head_dim,
        BT=32,
        BD=head_dim,
        HAS_BIAS=True,
        USE_LOWER_BOUND=True,
        num_warps=4,
        num_stages=3,
    )

    actual = fused_gate_bwd(*inputs)
    torch.testing.assert_close(actual.dg[0], triton_dg, rtol=1e-4, atol=1e-3)
    torch.testing.assert_close(
        actual.dA_partial.sum((0, 1)),
        triton_dA_partial.sum(0),
        rtol=2e-4,
        atol=2e-2,
    )


def test_fused_gate_bwd_custom_op_registration():
    """Exercise a non-default TMA schedule through the private operator schema."""
    inputs = _inputs(65, head_dim=64)
    torch.library.opcheck(
        _fused_gate_bwd_custom_op,
        (*inputs, 7, -3.25, False),
    )


def test_fused_gate_bwd_requires_batch_dimension():
    """Keep batch explicit at the public kernel boundary."""
    g, A_log, dt_bias, d_cumulative = _inputs(17, head_dim=64)
    with pytest.raises(ValueError, match=r"shape \(B, T, H, D\)"):
        fused_gate_bwd(g[0], A_log, dt_bias, d_cumulative[0])


def test_fused_gate_bwd_custom_op_rejects_noncontiguous_input():
    """Enforce the compact fake ABI at the private operator boundary."""
    _g, A_log, dt_bias, d_cumulative = _inputs(17, head_dim=64)
    strided_g = torch.empty(2, 17, 16, 128, device="cuda", dtype=torch.bfloat16)[..., ::2]
    with pytest.raises(ValueError, match="requires contiguous"):
        _fused_gate_bwd_custom_op(
            strided_g,
            A_log,
            dt_bias,
            d_cumulative,
            32,
            -3.25,
            False,
        )


@pytest.mark.parametrize(
    ("head_dim", "chunk_size", "fastmath"),
    ((64, 7, False), (128, 64, False), (128, 64, True)),
)
def test_fused_gate_bwd_fullgraph_dynamic(head_dim, chunk_size, fastmath):
    """Run strict dynamic fullgraph capture for two TMA schedules.

    Crossing a chunk boundary changes the chunk ``dA_partial`` extent, so
    Inductor may compile a second graph; this checks graph completeness rather
    than promising one graph for two different output shapes.
    """
    compiled = torch.compile(fused_gate_bwd, fullgraph=True, dynamic=True)
    for batch, tokens in ((1, 63), (3, 65)):
        inputs = _inputs(tokens, head_dim=head_dim, batch=batch)
        expected_dg, expected_dA, _expected_dt_bias = _fused_reference(
            inputs,
            -3.25,
            chunk_size,
        )
        actual = compiled(
            *inputs,
            lower_bound=-3.25,
            chunk_size=chunk_size,
            fastmath=fastmath,
        )
        torch.testing.assert_close(actual.dg, expected_dg, rtol=3e-5, atol=4e-5)
        torch.testing.assert_close(
            actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4
        )


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
    expected_dg, expected_dA, _expected_dt_bias = _fused_reference(inputs, -3.25, 7)
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(actual.dg, captured_dg)
    torch.testing.assert_close(actual.dg, expected_dg, rtol=3e-5, atol=4e-5)
    torch.testing.assert_close(actual.dA_partial.sum((0, 1)), expected_dA, rtol=5e-5, atol=7e-4)


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
