"""Public autograd and framework integration for scalar-GDN Mega."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 GDN path requires nvidia-cutlass-dsl>=4.7",
)

from attn_gym.linear import chunk_gdn as public_chunk_gdn
from attn_gym.linear.gdn.impl.mega_ops import (
    chunk_gdn_mega_packed_bwd_op,
    chunk_gdn_mega_packed_bwd_with_state_op,
    chunk_gdn_mega_packed_fwd_op,
    chunk_gdn_mega_packed_fwd_with_initial_state_op,
    chunk_gdn_mega_packed_fwd_with_state_op,
)
from attn_gym.linear.kda.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.kda import assert_matches_low_precision_reference, clone_kda_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 GDN path requires SM100 or SM103",
)

_MEGA_KERNEL_OPTIONS = {"backend": "mega"}


def chunk_gdn(*args, **kwargs):
    """Route fused calls in this module through the Mega backend."""
    if kwargs.get("impl") == "fused":
        kwargs.setdefault("kernel_options", _MEGA_KERNEL_OPTIONS)
    return public_chunk_gdn(*args, **kwargs)


def reference_results(
    inputs: tuple[torch.Tensor, ...], precision: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the packed eager GDN oracle with final state in one precision."""
    q, k, value, gate, beta, state, cu_seqlens = inputs
    output, final_state = chunk_gdn(
        q.to(precision),
        k.to(precision),
        value.to(precision),
        gate.to(precision),
        beta.to(precision),
        state.to(precision),
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="reference",
    )
    assert final_state is not None
    return output, final_state


def test_public_gdn_mega_packed_forward_and_six_gradients() -> None:
    """Public fused execution must match eager output, state, and every gradient."""
    inputs = make_gdn_test_inputs(
        (65, 0, 63),
        key_heads=1,
        value_heads=2,
        gate_pattern="isolated_negative_twenty",
        requires_grad=True,
        seed=307,
    )
    low = reference_results(inputs, torch.float32)
    high = reference_results(inputs, torch.float64)
    actual = chunk_gdn(
        *inputs[:6],
        cu_seqlens=inputs[6],
        output_final_state=True,
        impl="fused",
    )
    assert actual[1] is not None
    for name, actual_tensor, high_tensor, low_tensor in zip(
        ("output", "state"), actual, high, low, strict=True
    ):
        assert_matches_low_precision_reference(
            actual_tensor,
            high_tensor,
            low_tensor,
            name,
            source_dtype=inputs[0].dtype,
        )

    torch.manual_seed(311)
    cotangents = (torch.randn_like(actual[0]), torch.randn_like(actual[1]))
    actual_gradients = torch.autograd.grad(actual, inputs[:6], cotangents)
    low_targets = tuple(tensor.detach().float().requires_grad_() for tensor in inputs[:6])
    high_targets = tuple(tensor.detach().double().requires_grad_() for tensor in inputs[:6])
    low_result = chunk_gdn(
        *low_targets,
        cu_seqlens=inputs[6],
        output_final_state=True,
        impl="reference",
    )
    high_result = chunk_gdn(
        *high_targets,
        cu_seqlens=inputs[6],
        output_final_state=True,
        impl="reference",
    )
    low_gradients = torch.autograd.grad(
        low_result,
        low_targets,
        tuple(cotangent.float() for cotangent in cotangents),
    )
    high_gradients = torch.autograd.grad(
        high_result,
        high_targets,
        tuple(cotangent.double() for cotangent in cotangents),
    )
    for name, actual_gradient, high_gradient, low_gradient in zip(
        ("dq", "dk", "dv", "dgate", "dbeta", "dstate"),
        actual_gradients,
        high_gradients,
        low_gradients,
        strict=True,
    ):
        assert_matches_low_precision_reference(
            actual_gradient,
            high_gradient,
            low_gradient,
            name,
            source_dtype=inputs[0].dtype,
        )


def test_public_gdn_mega_grouped_h4_h12_forward_backward() -> None:
    """A non-power-of-two Q/K sharing group must reduce gradients correctly."""
    inputs = make_gdn_test_inputs(
        (65,), key_heads=4, value_heads=12, dtype=torch.bfloat16, requires_grad=True, seed=310
    )
    actual = chunk_gdn(*inputs[:5], impl="fused")[0]
    low_targets = tuple(tensor.detach().float().requires_grad_() for tensor in inputs[:5])
    high_targets = tuple(tensor.detach().double().requires_grad_() for tensor in inputs[:5])
    low = chunk_gdn(*low_targets, impl="reference")[0]
    high = chunk_gdn(*high_targets, impl="reference")[0]
    assert_matches_low_precision_reference(
        actual, high, low, "H4/H12 output", source_dtype=inputs[0].dtype
    )
    d_output = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, inputs[:5], d_output)
    low_gradients = torch.autograd.grad(low, low_targets, d_output.float())
    high_gradients = torch.autograd.grad(high, high_targets, d_output.double())
    for name, actual_gradient, high_gradient, low_gradient in zip(
        ("dq", "dk", "dv", "dgate", "dbeta"),
        actual_gradients,
        high_gradients,
        low_gradients,
        strict=True,
    ):
        assert_matches_low_precision_reference(
            actual_gradient,
            high_gradient,
            low_gradient,
            f"H4/H12 {name}",
            source_dtype=inputs[0].dtype,
        )


def test_public_gdn_mega_h64_custom_scale_forward_backward() -> None:
    """Production head count and a non-default scale must match the eager oracle."""
    inputs = make_gdn_test_inputs(
        (64,), key_heads=64, value_heads=64, dtype=torch.bfloat16, requires_grad=True, seed=312
    )
    scale = 0.125
    actual = chunk_gdn(*inputs[:5], scale=scale, impl="fused")[0]
    low_targets = tuple(tensor.detach().float().requires_grad_() for tensor in inputs[:5])
    high_targets = tuple(tensor.detach().double().requires_grad_() for tensor in inputs[:5])
    low = chunk_gdn(*low_targets, scale=scale, impl="reference")[0]
    high = chunk_gdn(*high_targets, scale=scale, impl="reference")[0]
    assert_matches_low_precision_reference(
        actual, high, low, "H64 custom-scale output", source_dtype=inputs[0].dtype
    )
    d_output = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, inputs[:5], d_output)
    low_gradients = torch.autograd.grad(low, low_targets, d_output.float())
    high_gradients = torch.autograd.grad(high, high_targets, d_output.double())
    for name, actual_gradient, high_gradient, low_gradient in zip(
        ("dq", "dk", "dv", "dgate", "dbeta"),
        actual_gradients,
        high_gradients,
        low_gradients,
        strict=True,
    ):
        assert_matches_low_precision_reference(
            actual_gradient,
            high_gradient,
            low_gradient,
            f"H64 custom-scale {name}",
            source_dtype=inputs[0].dtype,
        )


def test_public_gdn_mega_dense_batch_lowers_to_packed() -> None:
    """Dense B>1 calls must preserve reference outputs, states, and gradients."""
    packed = make_gdn_test_inputs(
        (65, 65), key_heads=1, value_heads=2, dtype=torch.float16, requires_grad=True, seed=313
    )
    dense = tuple(tensor.reshape(2, 65, *tensor.shape[2:]) for tensor in packed[:5])
    state = packed[5]
    expected = chunk_gdn(*dense, state, output_final_state=True, impl="reference")
    actual = chunk_gdn(*dense, state, output_final_state=True, impl="fused")
    assert actual[1] is not None and expected[1] is not None
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-2, atol=2e-2)

    d_output = torch.randn_like(actual[0])
    d_state = torch.randn_like(actual[1])
    actual_gradients = torch.autograd.grad(actual, (*dense, state), (d_output, d_state))
    expected_gradients = torch.autograd.grad(expected, (*dense, state), (d_output, d_state))
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-2, atol=5e-2)


def test_gdn_mega_raw_operator_registration() -> None:
    """Every fixed-arity raw schema must agree with runtime and fake metadata."""
    q, k, value, gate, beta, state, cu_seqlens = make_gdn_test_inputs(
        (65, 63), key_heads=1, value_heads=2, dtype=torch.bfloat16, seed=317
    )
    d_output = torch.randn_like(value)
    d_state = torch.randn_like(state)
    chunk_offsets = prepare_ragged_chunk_metadata(cu_seqlens, q.shape[1], 64).chunk_offsets
    scale = 128**-0.5
    test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    torch.library.opcheck(
        chunk_gdn_mega_packed_fwd_op,
        (q, k, value, gate, beta, cu_seqlens, chunk_offsets, scale),
        test_utils=test_utils,
    )
    torch.library.opcheck(
        chunk_gdn_mega_packed_fwd_with_initial_state_op,
        (q, k, value, gate, beta, state, cu_seqlens, chunk_offsets, scale),
        test_utils=test_utils,
    )
    torch.library.opcheck(
        chunk_gdn_mega_packed_fwd_with_state_op,
        (q, k, value, gate, beta, state, cu_seqlens, chunk_offsets, scale),
        test_utils=test_utils,
    )
    torch.library.opcheck(
        chunk_gdn_mega_packed_bwd_op,
        (q, k, value, gate, beta, d_output, cu_seqlens, scale),
        test_utils=test_utils,
    )
    for state_cotangent in (None, d_state):
        torch.library.opcheck(
            chunk_gdn_mega_packed_bwd_with_state_op,
            (
                q,
                k,
                value,
                gate,
                beta,
                d_output,
                state,
                state_cotangent,
                cu_seqlens,
                scale,
            ),
            test_utils=test_utils,
        )


def test_gdn_mega_backward_fake_preserves_leading_strides() -> None:
    """Backward fakes must expose the exact layouts returned by the CUDA wrapper."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    q, k, value, gate, beta, _state, cu_seqlens = make_gdn_test_inputs(
        (65,), key_heads=2, value_heads=2, dtype=torch.bfloat16, seed=319
    )
    inputs = tuple(
        torch.as_strided(
            tensor,
            tensor.shape,
            (tensor.stride(0) + 16 // tensor.element_size(), *tensor.stride()[1:]),
        )
        for tensor in (q, k, value, gate, beta)
    )
    d_output = torch.randn_like(value)
    arguments = (*inputs, d_output, cu_seqlens)
    actual = chunk_gdn_mega_packed_bwd_op(*arguments, 128**-0.5)
    with FakeTensorMode() as mode:
        fake_arguments = tuple(mode.from_tensor(tensor) for tensor in arguments)
        fake = chunk_gdn_mega_packed_bwd_op(*fake_arguments, 128**-0.5)

    expected_strides = tuple(tensor.stride() for tensor in inputs)
    assert tuple(tensor.stride() for tensor in actual) == expected_strides
    assert tuple(tensor.stride() for tensor in fake) == expected_strides


@pytest.mark.parametrize("offsets", ([1, 3], [0, 4, 3], [0, 9]))
def test_public_gdn_mega_rejects_invalid_packed_offset_values(offsets: list[int]) -> None:
    """Device-side boundary validation must fail before invalid TMA descriptors execute."""
    code = f"""
import torch
from attn_gym.linear import chunk_gdn
shape = (1, 8, 2, 128)
q = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
gate = -torch.rand(shape[:3], device='cuda')
beta = torch.rand(shape[:3], device='cuda')
cu = torch.tensor({offsets!r}, dtype=torch.int32, device='cuda')
chunk_gdn(
    q,
    k,
    v,
    gate,
    beta,
    cu_seqlens=cu,
    impl='fused',
    kernel_options={{"backend": "mega"}},
)
torch.cuda.synchronize()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
            env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        )
    except subprocess.TimeoutExpired:
        pytest.fail("invalid packed offsets hung instead of failing validation")
    assert result.returncode != 0, "invalid packed offsets reached the Mega kernel"


def test_public_gdn_mega_casts_low_precision_gate_and_beta_with_gradients() -> None:
    """FP32 gate normalization must preserve low-precision output and gradient accuracy."""
    q, k, value, gate, beta, _state, _cu_seqlens = make_gdn_test_inputs(
        (65,), key_heads=2, value_heads=2, dtype=torch.bfloat16, seed=327
    )
    quantized = tuple(
        tensor.detach().to(torch.bfloat16).requires_grad_() for tensor in (q, k, value, gate, beta)
    )
    actual_inputs = clone_kda_inputs(quantized)
    source_inputs = clone_kda_inputs(quantized)
    high_inputs = clone_kda_inputs(quantized, dtype=torch.float64)
    actual_output = chunk_gdn(*actual_inputs, impl="fused")[0]
    source_output = chunk_gdn(*source_inputs, impl="reference")[0]
    high_output = chunk_gdn(*high_inputs, impl="reference")[0]
    d_output = torch.randn_like(actual_output)
    actual_gradients = torch.autograd.grad(actual_output, actual_inputs, d_output)
    source_gradients = torch.autograd.grad(source_output, source_inputs, d_output)
    high_gradients = torch.autograd.grad(high_output, high_inputs, d_output.double())

    assert_matches_low_precision_reference(
        actual_output,
        high_output,
        source_output,
        "low-precision gate output",
        source_dtype=torch.bfloat16,
    )
    for name, actual, high, source in zip(
        ("dq", "dk", "dv", "dgate", "dbeta"),
        actual_gradients,
        high_gradients,
        source_gradients,
        strict=True,
    ):
        assert actual.dtype == torch.bfloat16
        assert_matches_low_precision_reference(
            actual,
            high,
            source,
            f"low-precision gate {name}",
            source_dtype=torch.bfloat16,
        )


def test_public_gdn_mega_no_state_backward_initializes_cuda_device() -> None:
    """The custom-autograd callback must restore a distinct caller CUDA device."""
    if torch.cuda.device_count() < 2:
        pytest.skip("CUDA device restoration requires at least two visible GPUs")
    inputs = make_gdn_test_inputs((65,), key_heads=2, value_heads=2, requires_grad=True, seed=329)
    output, final_state = chunk_gdn(*inputs[:5], impl="fused")
    assert final_state is None
    input_device = inputs[0].device.index
    assert input_device is not None
    caller_device = 1 if input_device == 0 else 0
    with torch.cuda.device(caller_device):
        gradients = torch.autograd.grad(output, inputs[:5], torch.randn_like(output))
        assert torch.cuda.current_device() == caller_device
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("layout", ["misaligned", "inner_stride"])
def test_public_compiled_gdn_normalizes_unsupported_q_layout(layout: str) -> None:
    """Compiled public execution must match eager normalization for supported tensor values."""
    q, k, value, gate, beta, _state, _cu_seqlens = make_gdn_test_inputs(
        (65,), key_heads=2, value_heads=2, dtype=torch.bfloat16, seed=330
    )
    if layout == "misaligned":
        storage = torch.empty(q.numel() + 1, dtype=q.dtype, device=q.device)
        candidate_q = storage[1:].view_as(q).copy_(q)
    else:
        storage = torch.empty(q.numel() * 2, dtype=q.dtype, device=q.device)
        candidate_q = torch.as_strided(
            storage,
            q.shape,
            (q.stride(0) * 2, q.stride(1) * 2, q.stride(2) * 2, 2),
        ).copy_(q)

    @torch.compile(fullgraph=True)
    def compiled(query, key, values, gates, betas):
        return chunk_gdn(query, key, values, gates, betas, impl="fused")[0]

    expected = chunk_gdn(candidate_q, k, value, gate, beta, impl="fused")[0]
    actual = compiled(candidate_q, k, value, gate, beta)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_public_gdn_mega_fullgraph_forward_backward() -> None:
    """Strict compilation must retain the public stateful autograd contract."""
    inputs = make_gdn_test_inputs(
        (65, 0, 63), key_heads=1, value_heads=2, requires_grad=True, seed=331
    )

    @torch.compile(fullgraph=True)
    def compiled(q, k, value, gate, beta, state, cu_seqlens):
        return chunk_gdn(
            q,
            k,
            value,
            gate,
            beta,
            state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            impl="fused",
        )

    actual_inputs = (*clone_kda_inputs(inputs[:-1]), inputs[-1])
    expected = chunk_gdn(*inputs[:6], cu_seqlens=inputs[6], output_final_state=True, impl="fused")
    actual = compiled(*actual_inputs)
    assert expected[1] is not None and actual[1] is not None
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    cotangents = (torch.randn_like(actual[0]), torch.randn_like(actual[1]))
    expected_gradients = torch.autograd.grad(expected, inputs[:6], cotangents)
    actual_gradients = torch.autograd.grad(actual, actual_inputs[:6], cotangents)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["no_state", "initial_only", "final_from_zero"])
def test_public_gdn_mega_fullgraph_output_arities(mode: str) -> None:
    """Strict compilation must support every fixed-arity forward operator route."""
    inputs = make_gdn_test_inputs((65,), key_heads=2, value_heads=2, seed=333)

    @torch.compile(fullgraph=True)
    def compiled(q, k, value, gate, beta, state):
        if mode == "no_state":
            return chunk_gdn(q, k, value, gate, beta, impl="fused")[0]
        if mode == "initial_only":
            return chunk_gdn(q, k, value, gate, beta, state, impl="fused")[0]
        return chunk_gdn(q, k, value, gate, beta, output_final_state=True, impl="fused")

    state = inputs[5]
    if mode == "no_state":
        expected = chunk_gdn(*inputs[:5], impl="fused")[0]
    elif mode == "initial_only":
        expected = chunk_gdn(*inputs[:5], state, impl="fused")[0]
    else:
        expected = chunk_gdn(*inputs[:5], output_final_state=True, impl="fused")
    actual = compiled(*inputs[:5], state)
    if isinstance(actual, tuple):
        for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_public_gdn_mega_dense_batch_fullgraph() -> None:
    """Dense B>1 packing and output reshaping must remain inside one strict graph."""
    packed = make_gdn_test_inputs(
        (65, 65), key_heads=1, value_heads=2, dtype=torch.bfloat16, seed=335
    )
    dense = tuple(tensor.reshape(2, 65, *tensor.shape[2:]) for tensor in packed[:5])

    @torch.compile(fullgraph=True)
    def compiled(q, k, value, gate, beta, state):
        return chunk_gdn(
            q,
            k,
            value,
            gate,
            beta,
            state,
            output_final_state=True,
            impl="fused",
        )

    expected = chunk_gdn(*dense, packed[5], output_final_state=True, impl="fused")
    actual = compiled(*dense, packed[5])
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        assert actual_tensor is not None and expected_tensor is not None
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)


def test_public_gdn_mega_dynamic_packed_sequence_count() -> None:
    """One dynamic strict graph must accept changing packed sequence counts."""
    inputs = make_gdn_test_inputs((128,), key_heads=2, value_heads=2, seed=336)

    @torch.compile(fullgraph=True, dynamic=True)
    def compiled(q, k, value, gate, beta, cu_seqlens):
        return chunk_gdn(q, k, value, gate, beta, cu_seqlens=cu_seqlens, impl="fused")[0]

    for lengths in ((64, 64), (31, 0, 33, 64)):
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")
        expected = chunk_gdn(*inputs[:5], cu_seqlens=cu_seqlens, impl="fused")[0]
        actual = compiled(*inputs[:5], cu_seqlens)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_public_gdn_mega_dynamic_dense_tokens() -> None:
    """One dynamic fullgraph callable must reuse the dense lowering across BT64 boundaries."""

    @torch.compile(fullgraph=True, dynamic=True)
    def compiled(q, k, value, gate, beta):
        return chunk_gdn(q, k, value, gate, beta, impl="fused")[0]

    for tokens in (63, 65, 129):
        inputs = make_gdn_test_inputs(
            (tokens,), key_heads=2, value_heads=2, dtype=torch.bfloat16, seed=337 + tokens
        )
        expected = chunk_gdn(*inputs[:5], impl="fused")[0]
        actual = compiled(*inputs[:5])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    os.environ.get("ATTN_GYM_RUN_STRESS_TESTS") != "1",
    reason="set ATTN_GYM_RUN_STRESS_TESTS=1 to run repeated-launch stress tests",
)
def test_public_gdn_mega_repeated_stateful_launches_cross_wave_boundary() -> None:
    """Persistent CTAs must preserve barrier phases across their fourth logical item."""
    inputs = make_gdn_test_inputs(
        (128,) * 29,
        key_heads=8,
        value_heads=16,
        requires_grad=False,
        seed=353,
    )
    q, k, value, gate, beta, state, cu_seqlens = inputs
    with torch.no_grad():
        for _ in range(1000):
            chunk_gdn(
                q,
                k,
                value,
                gate,
                beta,
                state,
                cu_seqlens=cu_seqlens,
                output_final_state=True,
                impl="fused",
            )
            torch.cuda.synchronize()


@pytest.mark.skipif(
    os.environ.get("ATTN_GYM_RUN_STRESS_TESTS") != "1",
    reason="set ATTN_GYM_RUN_STRESS_TESTS=1 to run repeated-launch stress tests",
)
def test_public_gdn_mega_repeated_backward_crosses_wave_boundary() -> None:
    """Recompute and backward barriers must survive persistent CTA reuse."""
    inputs = make_gdn_test_inputs(
        (128,) * 29,
        key_heads=8,
        value_heads=16,
        requires_grad=True,
        seed=419,
    )
    q, k, value, gate, beta, state, cu_seqlens = inputs
    d_output = torch.randn_like(value)
    d_state = torch.randn_like(state)
    for _ in range(500):
        output, final_state = chunk_gdn(
            q,
            k,
            value,
            gate,
            beta,
            state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            impl="fused",
        )
        assert final_state is not None
        torch.autograd.grad(
            (output, final_state),
            (q, k, value, gate, beta, state),
            (d_output, d_state),
        )
        torch.cuda.synchronize()


def test_public_gdn_mega_cuda_graph_replay() -> None:
    """Compiled packed stateful forward/backward must capture and replay bitwise."""
    inputs = make_gdn_test_inputs(
        (65, 0, 63), key_heads=2, value_heads=2, requires_grad=True, seed=347
    )

    @torch.compile(fullgraph=True)
    def compiled_forward(q, k, value, gate, beta, state, cu_seqlens):
        return chunk_gdn(
            q,
            k,
            value,
            gate,
            beta,
            state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            impl="fused",
        )

    def step(q, k, value, gate, beta, state, cu_seqlens, d_output, d_state):
        output, final_state = compiled_forward(q, k, value, gate, beta, state, cu_seqlens)
        assert final_state is not None
        return torch.autograd.grad(
            (output, final_state),
            (q, k, value, gate, beta, state),
            (d_output, d_state),
        )

    d_output = torch.randn_like(inputs[2])
    d_state = torch.randn_like(inputs[5])
    expected = step(*inputs, d_output, d_state)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        with torch.cuda.graph(graph):
            captured = step(*inputs, d_output, d_state)
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(False)
    graph.replay()
    torch.cuda.synchronize()
    for actual_gradient, expected_gradient in zip(captured, expected, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
