import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import GatedDeltaRuleOutput, gated_delta_rule


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


@pytest.mark.parametrize("sequence,chunk_size", [(1, 4), (7, 4), (8, 4), (17, 8)])
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_chunked_matches_recurrent(sequence, chunk_size, use_initial_state):
    inputs = make_inputs(sequence)
    initial_state = inputs[-1] if use_initial_state else None
    recurrent = gated_delta_rule(
        *inputs[:-1],
        initial_state=initial_state,
        return_final_state=True,
        mode="recurrent",
    )
    chunked = gated_delta_rule(
        *inputs[:-1],
        initial_state=initial_state,
        return_final_state=True,
        mode="chunked",
        chunk_size=chunk_size,
    )

    assert isinstance(chunked, GatedDeltaRuleOutput)
    torch.testing.assert_close(chunked.output, recurrent.output, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(chunked.final_state, recurrent.final_state, atol=1e-6, rtol=1e-5)


def test_segmented_recurrent_execution_matches_full_sequence():
    inputs = make_inputs(sequence=9)
    full = gated_delta_rule(*inputs[:-1], return_final_state=True, mode="recurrent")
    first = gated_delta_rule(
        *(tensor[:, :, :4] for tensor in inputs[:-1]),
        return_final_state=True,
        mode="recurrent",
    )
    second = gated_delta_rule(
        *(tensor[:, :, 4:] for tensor in inputs[:-1]),
        initial_state=first.final_state,
        return_final_state=True,
        mode="recurrent",
    )

    torch.testing.assert_close(torch.cat((first.output, second.output), dim=2), full.output)
    torch.testing.assert_close(second.final_state, full.final_state)


def test_recurrent_execution_is_batch_invariant():
    inputs = make_inputs(sequence=5)
    batched = gated_delta_rule(*inputs[:-1], return_final_state=True, mode="recurrent")

    for batch_index in range(inputs[0].shape[0]):
        single = gated_delta_rule(
            *(tensor[batch_index : batch_index + 1] for tensor in inputs[:-1]),
            return_final_state=True,
            mode="recurrent",
        )
        torch.testing.assert_close(single.output[0], batched.output[batch_index])
        torch.testing.assert_close(single.final_state[0], batched.final_state[batch_index])


def test_chunked_gradients_match_recurrent():
    inputs = make_inputs(sequence=7)[:-1]
    gradients = []
    for mode in ("recurrent", "chunked"):
        differentiable_inputs = [value.clone().requires_grad_() for value in inputs]
        result = gated_delta_rule(
            *differentiable_inputs,
            return_final_state=True,
            mode=mode,
            chunk_size=4,
        )
        gradients.append(
            torch.autograd.grad(
                result.output.square().mean() + result.final_state.square().mean(),
                differentiable_inputs,
            )
        )

    for chunked_gradient, recurrent_gradient in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(chunked_gradient, recurrent_gradient, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("sequence,expected_mode", [(1, "recurrent"), (7, "chunked")])
def test_auto_mode_matches_documented_policy(sequence, expected_mode):
    inputs = make_inputs(sequence)
    automatic = gated_delta_rule(*inputs[:-1])
    explicit = gated_delta_rule(*inputs[:-1], mode=expected_mode)
    torch.testing.assert_close(automatic.output, explicit.output)
    assert automatic.final_state is None


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"mode": "parallel"}, "Unsupported mode"),
        ({"backend": "triton"}, "Unsupported backend"),
        ({"chunk_size": 0}, "chunk_size must be greater than zero"),
    ],
)
def test_invalid_options_fail_clearly(kwargs, match):
    inputs = make_inputs(sequence=2)
    with pytest.raises(ValueError, match=match):
        gated_delta_rule(*inputs[:-1], **kwargs)


def test_invalid_initial_state_shape_fails_clearly():
    inputs = make_inputs(sequence=2)
    with pytest.raises(ValueError, match="initial_state must have shape"):
        gated_delta_rule(*inputs[:-1], initial_state=inputs[-1][..., :-1])


@pytest.mark.parametrize("mismatch_initial_state", [False, True])
def test_mismatched_dtypes_fail_clearly(mismatch_initial_state):
    inputs = list(make_inputs(sequence=2))
    if mismatch_initial_state:
        inputs[-1] = inputs[-1].double()
        kwargs = {"initial_state": inputs[-1]}
    else:
        inputs[2] = inputs[2].double()
        kwargs = {}

    with pytest.raises(ValueError, match="all inputs must have the same dtype"):
        gated_delta_rule(*inputs[:-1], **kwargs)


def test_mismatched_devices_fail_clearly():
    inputs = list(make_inputs(sequence=2))
    inputs[2] = inputs[2].to("meta")

    with pytest.raises(ValueError, match="all inputs must be on the same device"):
        gated_delta_rule(*inputs[:-1])


def test_nonfloating_inputs_fail_clearly():
    inputs = list(make_inputs(sequence=2))
    inputs[4] = inputs[4].long()

    with pytest.raises(ValueError, match="all inputs must have floating-point dtypes"):
        gated_delta_rule(*inputs[:-1])
