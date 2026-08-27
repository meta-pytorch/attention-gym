import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import Impl, chunk_gdn, recurrent_gdn
from attn_gym.testing import cumulative_sequence_offsets

REFERENCE_CASES = [recurrent_gdn, chunk_gdn]


def make_inputs(sequence: int) -> tuple[torch.Tensor, ...]:
    """Create stable gated delta rule inputs with normalized keys."""
    torch.manual_seed(0)
    batch, heads, key_dimension, value_dimension = 2, 3, 4, 5
    query = torch.randn(batch, sequence, heads, key_dimension)
    key = F.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(batch, sequence, heads, value_dimension)
    gate = F.logsigmoid(torch.randn(batch, sequence, heads))
    beta = torch.sigmoid(torch.randn(batch, sequence, heads))
    initial_state = torch.randn(batch, heads, key_dimension, value_dimension)
    return query, key, value, gate, beta, initial_state


def run_with_state(function, inputs):
    """Run one GDN form with its initial and final recurrent state."""
    return function(*inputs, output_final_state=True)


@pytest.mark.parametrize("sequence", [1, 7, 64, 73])
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_chunk_matches_recurrent(sequence, use_initial_state):
    inputs = make_inputs(sequence)
    initial_state = inputs[-1] if use_initial_state else None
    recurrent_output, recurrent_state = recurrent_gdn(
        *inputs[:-1],
        initial_state,
        output_final_state=True,
    )
    chunked_output, chunked_state = chunk_gdn(
        *inputs[:-1],
        initial_state,
        output_final_state=True,
    )

    torch.testing.assert_close(chunked_output, recurrent_output, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(chunked_state, recurrent_state, atol=1e-6, rtol=1e-5)


def test_segmented_recurrent_execution_matches_full_sequence():
    inputs = make_inputs(sequence=9)
    full_output, full_state = recurrent_gdn(*inputs[:-1], output_final_state=True)
    first_output, first_state = recurrent_gdn(
        *(tensor[:, :4] for tensor in inputs[:-1]),
        output_final_state=True,
    )
    second_output, second_state = recurrent_gdn(
        *(tensor[:, 4:] for tensor in inputs[:-1]),
        first_state,
        output_final_state=True,
    )

    torch.testing.assert_close(torch.cat((first_output, second_output), dim=1), full_output)
    torch.testing.assert_close(second_state, full_state)


@pytest.mark.parametrize("function", REFERENCE_CASES)
def test_packed_matches_independent_sequences(function):
    q, k, v, gate, beta, _state = make_inputs(sequence=8)
    q, k, v, gate, beta = (tensor[:1] for tensor in (q, k, v, gate, beta))
    cu_seqlens = cumulative_sequence_offsets([3, 0, 4], device="cpu")
    initial_state = torch.randn(3, q.shape[2], q.shape[3], v.shape[-1])

    output, final_state = function(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    expected_output = torch.zeros_like(v)
    expected_state = initial_state.clone()
    for sequence, (begin, end) in enumerate(((0, 3), (3, 3), (3, 7))):
        if begin == end:
            continue
        span = slice(begin, end)
        span_output, span_state = function(
            q[:, span],
            k[:, span],
            v[:, span],
            gate[:, span],
            beta[:, span],
            initial_state[sequence : sequence + 1],
            output_final_state=True,
        )
        expected_output[:, span] = span_output
        expected_state[sequence] = span_state[0]

    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(final_state, expected_state)


@pytest.mark.parametrize("offsets", [[1, 3], [0, 4, 3], [0, 9]])
def test_packed_rejects_invalid_offset_values(offsets):
    inputs = make_inputs(sequence=8)
    with pytest.raises(ValueError, match="start at zero, be nondecreasing"):
        recurrent_gdn(
            *(tensor[:1] for tensor in inputs[:-1]),
            cu_seqlens=torch.tensor(offsets, dtype=torch.int32),
        )


def test_recurrent_execution_is_batch_invariant():
    inputs = make_inputs(sequence=5)
    batched_output, batched_state = recurrent_gdn(*inputs[:-1], output_final_state=True)

    for batch_index in range(inputs[0].shape[0]):
        single_output, single_state = recurrent_gdn(
            *(tensor[batch_index : batch_index + 1] for tensor in inputs[:-1]),
            output_final_state=True,
        )
        torch.testing.assert_close(single_output[0], batched_output[batch_index])
        torch.testing.assert_close(single_state[0], batched_state[batch_index])


def test_chunk_gradients_match_recurrent():
    inputs = make_inputs(sequence=73)[:-1]
    gradients = []
    for function in REFERENCE_CASES:
        differentiable_inputs = [value.clone().requires_grad_() for value in inputs]
        result = function(
            *differentiable_inputs,
            output_final_state=True,
        )
        output, final_state = result
        gradients.append(
            torch.autograd.grad(
                output.square().mean() + final_state.square().mean(),
                differentiable_inputs,
            )
        )

    for chunk_gradient, recurrent_gradient in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(chunk_gradient, recurrent_gradient, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("function", REFERENCE_CASES)
@pytest.mark.parametrize("qkv_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp32_gate", [False, True])
def test_low_precision_compute_and_gradient_dtypes(function, qkv_dtype, fp32_gate):
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

    output, final_state = run_with_state(function, inputs)
    expected_output, expected_state = run_with_state(function, expected_inputs)
    assert output.dtype == qkv_dtype
    assert final_state.dtype == torch.float32
    torch.testing.assert_close(
        output.float(),
        expected_output,
        rtol=torch.finfo(qkv_dtype).eps,
        atol=torch.finfo(qkv_dtype).eps,
    )
    torch.testing.assert_close(final_state, expected_state, rtol=0, atol=0)

    gradients = torch.autograd.grad(
        output.float().square().mean() + final_state.square().mean(), inputs
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
    "function,device_type,autocast_dtype",
    [
        (recurrent_gdn, "cpu", torch.bfloat16),
        (chunk_gdn, "cpu", torch.bfloat16),
        (recurrent_gdn, "cuda", torch.float16),
        (recurrent_gdn, "cuda", torch.bfloat16),
    ],
)
def test_reference_disables_autocast(function, device_type, autocast_dtype):
    if device_type == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA autocast requires CUDA")

    expected_inputs = [
        tensor.to(device_type).requires_grad_() for tensor in make_inputs(sequence=7)
    ]
    actual_inputs = [tensor.detach().clone().requires_grad_() for tensor in expected_inputs]
    expected_output, expected_state = run_with_state(function, expected_inputs)
    with torch.autocast(device_type=device_type, dtype=autocast_dtype):
        actual_output, actual_state = run_with_state(function, actual_inputs)

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    expected_gradients = torch.autograd.grad(
        expected_output.square().mean() + expected_state.square().mean(), expected_inputs
    )
    actual_gradients = torch.autograd.grad(
        actual_output.square().mean() + actual_state.square().mean(), actual_inputs
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("function", [chunk_gdn, recurrent_gdn])
def test_low_precision_default_state_is_fp32(function):
    query, key, value, gate, beta, _initial_state = make_inputs(sequence=3)
    output, final_state = function(
        query.bfloat16(),
        key.bfloat16(),
        value.bfloat16(),
        gate,
        beta,
        output_final_state=True,
    )
    assert output.dtype == torch.bfloat16
    assert final_state.dtype == torch.float32


@pytest.mark.parametrize("function", [chunk_gdn, recurrent_gdn])
def test_fp64_preserves_compute_and_state_dtype(function):
    inputs = [tensor.double() for tensor in make_inputs(sequence=3)]
    output, final_state = function(*inputs, output_final_state=True)
    assert output.dtype == torch.float64
    assert final_state.dtype == torch.float64


@pytest.mark.parametrize("function", [chunk_gdn, recurrent_gdn])
def test_impl_accepts_enum_and_string(function):
    inputs = make_inputs(sequence=2)
    from_enum = function(*inputs[:-1], impl=Impl.REFERENCE)
    from_string = function(*inputs[:-1], impl="reference")
    torch.testing.assert_close(from_enum[0], from_string[0])
    assert type(from_enum) is tuple and from_enum[1] is None

    with pytest.raises(ValueError, match="'fused', 'reference'"):
        function(*inputs[:-1], impl="eager")
    error = NotImplementedError if function is chunk_gdn else ValueError
    message = "impl='fused'" if function is chunk_gdn else "requires CUDA tensors"
    with pytest.raises(error, match=message):
        function(*inputs[:-1], impl=Impl.FUSED)


@pytest.mark.parametrize("function", [chunk_gdn, recurrent_gdn])
def test_invalid_initial_state_shape_fails_clearly(function):
    inputs = make_inputs(sequence=2)
    with pytest.raises(ValueError, match="initial_state must have shape"):
        function(*inputs[:-1], initial_state=inputs[-1][..., :-1])


def test_recurrent_paged_validates_mode_contract():
    q, k, v, gate, beta, _state = make_inputs(sequence=2)
    state_cache = torch.randn(4, q.shape[2], v.shape[-1], q.shape[-1])
    state_indices = torch.tensor([1, 2], dtype=torch.int32)
    has_initial_state = torch.ones(2, dtype=torch.bool)

    with pytest.raises(ValueError, match="requires initial_state"):
        recurrent_gdn(q, k, v, gate, beta, state_indices=state_indices)
    with pytest.raises(ValueError, match="drop output_final_state"):
        recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices=state_indices,
            output_final_state=True,
        )
    with pytest.raises(ValueError, match="requires impl='fused'"):
        recurrent_gdn(q, k, v, gate, beta, state_cache, state_indices=state_indices)
    with pytest.raises(ValueError, match="requires state_indices"):
        recurrent_gdn(q, k, v, gate, beta, has_initial_state=has_initial_state)


def test_mismatched_qkv_dtypes_fail_clearly():
    inputs = list(make_inputs(sequence=2))
    inputs[2] = inputs[2].double()
    with pytest.raises(ValueError, match="q, k, and v must have the same dtype"):
        recurrent_gdn(*inputs[:-1])


def test_gate_and_beta_accept_independent_floating_dtypes():
    query, key, value, gate, beta, _initial_state = make_inputs(sequence=2)
    output, final_state = recurrent_gdn(
        query.bfloat16(),
        key.bfloat16(),
        value.bfloat16(),
        gate.half(),
        beta.double(),
        output_final_state=True,
    )
    assert output.dtype == torch.bfloat16
    assert final_state.dtype == torch.float32


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
