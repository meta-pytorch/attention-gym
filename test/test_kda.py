import math

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear.kda.naive import (
    _naive_chunk_kda_from_cumulative,
    chunk_cumsum_ref,
    naive_chunk_kda,
    naive_recurrent_kda,
)


def make_inputs(seq_len: int) -> tuple[torch.Tensor, ...]:
    """Create stable KDA delta-rule inputs with normalized keys and a per-channel gate."""
    torch.manual_seed(0)
    batch, heads, key_dim, value_dim = 2, 3, 4, 5
    q = torch.randn(batch, seq_len, heads, key_dim)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(batch, seq_len, heads, value_dim)
    g = F.logsigmoid(torch.randn(batch, seq_len, heads, key_dim))  # per-channel (diagonal) gate
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads))
    initial_state = torch.randn(batch, heads, value_dim, key_dim)
    return q, k, v, g, beta, initial_state


@pytest.mark.parametrize("seq_len,chunk_size", [(1, 4), (7, 4), (8, 4), (17, 8)])
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_naive_chunk_matches_recurrent(seq_len, chunk_size, use_initial_state):
    inputs = make_inputs(seq_len)
    initial_state = inputs[-1] if use_initial_state else None
    recurrent_output, recurrent_state = naive_recurrent_kda(
        *inputs[:-1], initial_state=initial_state, output_final_state=True
    )
    chunk_output, chunk_state = naive_chunk_kda(
        *inputs[:-1],
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(chunk_output, recurrent_output, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(chunk_state, recurrent_state, atol=1e-5, rtol=1e-4)

    cumulative_output, cumulative_state = _naive_chunk_kda_from_cumulative(
        *inputs[:3],
        chunk_cumsum_ref(inputs[3], chunk_size),
        inputs[4],
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    torch.testing.assert_close(cumulative_output, recurrent_output, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(cumulative_state, recurrent_state, atol=1e-5, rtol=1e-4)


def test_naive_chunk_accumulates_low_precision_gate_in_fp32():
    q, k, v, g, beta, initial_state = make_inputs(seq_len=64)
    g = g.to(torch.bfloat16)
    expected = _naive_chunk_kda_from_cumulative(
        q,
        k,
        v,
        chunk_cumsum_ref(g.float(), 64),
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
    )
    actual = naive_chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_naive_chunk_is_stable_for_kda_gate_range():
    torch.manual_seed(3)
    q = torch.randn(1, 65, 1, 4)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn_like(q)
    g = -5.0 * torch.sigmoid(torch.randn_like(q)) * math.log2(math.e)
    beta = torch.sigmoid(torch.randn(1, 65, 1))
    initial_state = torch.randn(1, 1, 4, 4)
    results = []

    for function in (naive_recurrent_kda, naive_chunk_kda):
        inputs = [tensor.clone().requires_grad_() for tensor in (q, k, v, g, beta, initial_state)]
        kwargs = {"chunk_size": 64} if function is naive_chunk_kda else {}
        output, state = function(
            *inputs[:5],
            initial_state=inputs[5],
            output_final_state=True,
            **kwargs,
        )
        gradients = torch.autograd.grad(output.square().mean() + state.square().mean(), inputs)
        results.append((output, state, gradients))

    expected, expected_state, expected_gradients = results[0]
    actual, actual_state, actual_gradients = results[1]
    assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
    assert all(torch.isfinite(gradient).all() for gradient in actual_gradients)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-4, atol=2e-5)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-4, atol=2e-5)


def test_naive_chunk_from_cumulative_validates_shapes():
    q, k, v, g, beta, _state = make_inputs(seq_len=7)
    cumulative_g = chunk_cumsum_ref(g, 4)
    with pytest.raises(ValueError, match="k must have shape"):
        _naive_chunk_kda_from_cumulative(q, k[:, :-1], v, cumulative_g, beta, chunk_size=4)
    with pytest.raises(ValueError, match="beta must have shape"):
        _naive_chunk_kda_from_cumulative(q, k, v, cumulative_g, beta.transpose(1, 2), chunk_size=4)
    with pytest.raises(ValueError, match="initial_state must have shape"):
        _naive_chunk_kda_from_cumulative(
            q,
            k,
            v,
            cumulative_g,
            beta,
            initial_state=torch.zeros(1, 3, 4, 5),
            chunk_size=4,
        )


def test_naive_chunk_gradients_match_recurrent():
    inputs = make_inputs(seq_len=7)
    gradients = []
    for function in (naive_recurrent_kda, naive_chunk_kda):
        differentiable_inputs = [value.clone().requires_grad_() for value in inputs]
        kwargs = {"chunk_size": 4} if function is naive_chunk_kda else {}
        output, state = function(
            *differentiable_inputs[:5],
            initial_state=differentiable_inputs[5],
            output_final_state=True,
            **kwargs,
        )
        gradients.append(
            torch.autograd.grad(
                output.square().mean() + state.square().mean(), differentiable_inputs
            )
        )

    for actual, recurrent in zip(gradients[1], gradients[0], strict=True):
        torch.testing.assert_close(actual, recurrent, atol=1e-5, rtol=1e-4)
