import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import (
    GatedDeltaRuleOutput,
    Impl,
    chunk_gdn,
    recurrent_gdn,
)

REFERENCE_CASES = [(recurrent_gdn, {}), (chunk_gdn, {"chunk_size": 4})]


def make_inputs(sequence: int) -> tuple[torch.Tensor, ...]:
    """Create stable gated delta rule inputs with normalized keys."""
    torch.manual_seed(0)
    batch, heads, key_dimension, value_dimension = 2, 3, 4, 5
    query = torch.randn(batch, heads, sequence, key_dimension)
    key = F.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(batch, heads, sequence, value_dimension)
    gate = F.logsigmoid(torch.randn(batch, heads, sequence))
    beta = torch.sigmoid(torch.randn(batch, heads, sequence))
    initial_state = torch.randn(batch, heads, key_dimension, value_dimension)
    return query, key, value, gate, beta, initial_state


def run_with_state(function, inputs, kwargs):
    """Run one GDN form with its initial and final recurrent state."""
    return function(*inputs[:5], initial_state=inputs[5], return_final_state=True, **kwargs)


@pytest.mark.parametrize("sequence,chunk_size", [(1, 4), (7, 4), (8, 4), (17, 8)])
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_chunk_matches_recurrent(sequence, chunk_size, use_initial_state):
    inputs = make_inputs(sequence)
    initial_state = inputs[-1] if use_initial_state else None
    recurrent = recurrent_gdn(
        *inputs[:-1],
        initial_state=initial_state,
        return_final_state=True,
    )
    chunked = chunk_gdn(
        *inputs[:-1],
        initial_state=initial_state,
        return_final_state=True,
        chunk_size=chunk_size,
    )

    assert isinstance(chunked, GatedDeltaRuleOutput)
    torch.testing.assert_close(chunked.output, recurrent.output, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(chunked.final_state, recurrent.final_state, atol=1e-6, rtol=1e-5)


def test_segmented_recurrent_execution_matches_full_sequence():
    inputs = make_inputs(sequence=9)
    full = recurrent_gdn(*inputs[:-1], return_final_state=True)
    first = recurrent_gdn(
        *(tensor[:, :, :4] for tensor in inputs[:-1]),
        return_final_state=True,
    )
    second = recurrent_gdn(
        *(tensor[:, :, 4:] for tensor in inputs[:-1]),
        initial_state=first.final_state,
        return_final_state=True,
    )

    torch.testing.assert_close(torch.cat((first.output, second.output), dim=2), full.output)
    torch.testing.assert_close(second.final_state, full.final_state)


def test_recurrent_execution_is_batch_invariant():
    inputs = make_inputs(sequence=5)
    batched = recurrent_gdn(*inputs[:-1], return_final_state=True)

    for batch_index in range(inputs[0].shape[0]):
        single = recurrent_gdn(
            *(tensor[batch_index : batch_index + 1] for tensor in inputs[:-1]),
            return_final_state=True,
        )
        torch.testing.assert_close(single.output[0], batched.output[batch_index])
        torch.testing.assert_close(single.final_state[0], batched.final_state[batch_index])


def test_chunk_gradients_match_recurrent():
    inputs = make_inputs(sequence=7)[:-1]
    gradients = []
    for function, kwargs in REFERENCE_CASES:
        differentiable_inputs = [value.clone().requires_grad_() for value in inputs]
        result = function(
            *differentiable_inputs,
            return_final_state=True,
            **kwargs,
        )
        gradients.append(
            torch.autograd.grad(
                result.output.square().mean() + result.final_state.square().mean(),
                differentiable_inputs,
            )
        )

    for chunk_gradient, recurrent_gradient in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(chunk_gradient, recurrent_gradient, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("function,kwargs", REFERENCE_CASES)
@pytest.mark.parametrize("qkv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp32_gate", [False, True])
def test_low_precision_compute_and_gradient_dtypes(function, kwargs, qkv_dtype, fp32_gate):
    query, key, value, gate, beta, initial_state = make_inputs(sequence=7)
    gate_dtype = torch.float32 if fp32_gate else qkv_dtype
    inputs = [
        query.to(qkv_dtype),
        key.to(qkv_dtype),
        value.to(qkv_dtype),
        gate.to(gate_dtype),
        beta.to(gate_dtype),
        initial_state,
    ]
    inputs = [tensor.requires_grad_() for tensor in inputs]
    expected_inputs = [tensor.detach().float() for tensor in inputs]

    result = run_with_state(function, inputs, kwargs)
    expected = run_with_state(function, expected_inputs, kwargs)
    assert result.output.dtype == qkv_dtype
    assert result.final_state.dtype == torch.float32
    torch.testing.assert_close(
        result.output.float(),
        expected.output,
        rtol=torch.finfo(qkv_dtype).eps,
        atol=torch.finfo(qkv_dtype).eps,
    )
    torch.testing.assert_close(result.final_state, expected.final_state, rtol=0, atol=0)

    gradients = torch.autograd.grad(
        result.output.float().square().mean() + result.final_state.square().mean(), inputs
    )
    assert tuple(gradient.dtype for gradient in gradients) == (
        qkv_dtype,
        qkv_dtype,
        qkv_dtype,
        gate_dtype,
        gate_dtype,
        torch.float32,
    )


@pytest.mark.parametrize(
    "function,kwargs,device_type,autocast_dtype",
    [
        (recurrent_gdn, {}, "cpu", torch.bfloat16),
        (chunk_gdn, {"chunk_size": 4}, "cpu", torch.bfloat16),
        (recurrent_gdn, {}, "cuda", torch.float16),
        (recurrent_gdn, {}, "cuda", torch.bfloat16),
    ],
)
def test_reference_disables_autocast(function, kwargs, device_type, autocast_dtype):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA autocast requires CUDA")

    expected_inputs = [
        tensor.to(device_type).requires_grad_() for tensor in make_inputs(sequence=7)
    ]
    actual_inputs = [tensor.detach().clone().requires_grad_() for tensor in expected_inputs]
    expected = run_with_state(function, expected_inputs, kwargs)
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        actual = run_with_state(function, actual_inputs, kwargs)

    torch.testing.assert_close(actual.output, expected.output, rtol=0, atol=0)
    torch.testing.assert_close(actual.final_state, expected.final_state, rtol=0, atol=0)
    expected_gradients = torch.autograd.grad(
        expected.output.square().mean() + expected.final_state.square().mean(), expected_inputs
    )
    actual_gradients = torch.autograd.grad(
        actual.output.square().mean() + actual.final_state.square().mean(), actual_inputs
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("function", [chunk_gdn, recurrent_gdn])
def test_low_precision_default_state_is_fp32(function):
    query, key, value, gate, beta, _initial_state = make_inputs(sequence=3)
    result = function(
        query.bfloat16(),
        key.bfloat16(),
        value.bfloat16(),
        gate,
        beta,
        return_final_state=True,
    )
    assert result.output.dtype == torch.bfloat16
    assert result.final_state.dtype == torch.float32


@pytest.mark.parametrize("function", [chunk_gdn, recurrent_gdn])
def test_fp64_preserves_compute_and_state_dtype(function):
    inputs = [tensor.double() for tensor in make_inputs(sequence=3)]
    result = function(*inputs[:5], initial_state=inputs[5], return_final_state=True)
    assert result.output.dtype == torch.float64
    assert result.final_state.dtype == torch.float64


@pytest.mark.parametrize("function", [chunk_gdn, recurrent_gdn])
def test_impl_accepts_enum_and_string(function):
    inputs = make_inputs(sequence=2)
    from_enum = function(*inputs[:-1], impl=Impl.REFERENCE)
    from_string = function(*inputs[:-1], impl="reference")
    torch.testing.assert_close(from_enum.output, from_string.output)

    with pytest.raises(ValueError, match="'fused', 'reference'"):
        function(*inputs[:-1], impl="eager")
    with pytest.raises(NotImplementedError, match="impl='fused'"):
        function(*inputs[:-1], impl=Impl.FUSED)


def test_invalid_chunk_size_fails_clearly():
    inputs = make_inputs(sequence=2)
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        chunk_gdn(*inputs[:-1], chunk_size=0)


@pytest.mark.parametrize("function", [chunk_gdn, recurrent_gdn])
def test_invalid_initial_state_shape_fails_clearly(function):
    inputs = make_inputs(sequence=2)
    with pytest.raises(ValueError, match="initial_state must have shape"):
        function(*inputs[:-1], initial_state=inputs[-1][..., :-1])


def test_mismatched_qkv_dtypes_fail_clearly():
    inputs = list(make_inputs(sequence=2))
    inputs[2] = inputs[2].double()
    with pytest.raises(ValueError, match="query, key, and value must have the same dtype"):
        recurrent_gdn(*inputs[:-1])


def test_gate_and_beta_accept_independent_floating_dtypes():
    query, key, value, gate, beta, _initial_state = make_inputs(sequence=2)
    result = recurrent_gdn(
        query.bfloat16(),
        key.bfloat16(),
        value.bfloat16(),
        gate.half(),
        beta.double(),
        return_final_state=True,
    )
    assert result.output.dtype == torch.bfloat16
    assert result.final_state.dtype == torch.float32


def test_invalid_initial_state_dtype_fails_clearly():
    query, key, value, gate, beta, initial_state = make_inputs(sequence=2)
    with pytest.raises(ValueError, match="initial_state must have dtype torch.float32"):
        recurrent_gdn(
            query.bfloat16(),
            key.bfloat16(),
            value.bfloat16(),
            gate,
            beta,
            initial_state=initial_state.bfloat16(),
        )


def test_mismatched_devices_fail_clearly():
    inputs = list(make_inputs(sequence=2))
    inputs[2] = inputs[2].to("meta")

    with pytest.raises(ValueError, match="all inputs must be on the same device"):
        recurrent_gdn(*inputs[:-1])


def test_nonfloating_inputs_fail_clearly():
    inputs = list(make_inputs(sequence=2))
    inputs[4] = inputs[4].long()

    with pytest.raises(ValueError, match="all inputs must have floating-point dtypes"):
        recurrent_gdn(*inputs[:-1])
