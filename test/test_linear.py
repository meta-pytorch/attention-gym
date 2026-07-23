import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import (
    fused_recurrent_gated_delta_rule,
    naive_chunk_gated_delta_rule,
    naive_recurrent_gated_delta_rule,
)


def make_inputs(seq_len: int) -> tuple[torch.Tensor, ...]:
    """Create stable gated-delta-rule inputs with normalized keys."""
    torch.manual_seed(0)
    batch, heads, key_dim, value_dim = 2, 3, 4, 5
    q = torch.randn(batch, seq_len, heads, key_dim)
    k = F.normalize(torch.randn_like(q), dim=-1)
    v = torch.randn(batch, seq_len, heads, value_dim)
    g = F.logsigmoid(torch.randn(batch, seq_len, heads))
    beta = torch.sigmoid(torch.randn(batch, seq_len, heads))
    initial_state = torch.randn(batch, heads, key_dim, value_dim)
    return q, k, v, g, beta, initial_state


@pytest.mark.parametrize("seq_len,chunk_size", [(1, 4), (7, 4), (8, 4), (17, 8)])
@pytest.mark.parametrize("use_initial_state", [False, True])
def test_naive_chunk_matches_recurrent(seq_len, chunk_size, use_initial_state):
    inputs = make_inputs(seq_len)
    initial_state = inputs[-1] if use_initial_state else None
    recurrent_output, recurrent_state = naive_recurrent_gated_delta_rule(
        *inputs[:-1], initial_state=initial_state, output_final_state=True
    )
    chunk_output, chunk_state = naive_chunk_gated_delta_rule(
        *inputs[:-1],
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(chunk_output, recurrent_output, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(chunk_state, recurrent_state, atol=1e-6, rtol=1e-5)


def test_naive_chunk_gradients_match_recurrent():
    inputs = make_inputs(seq_len=7)[:-1]
    gradients = []
    for function in (naive_recurrent_gated_delta_rule, naive_chunk_gated_delta_rule):
        differentiable_inputs = [value.clone().requires_grad_() for value in inputs]
        kwargs = {"chunk_size": 4} if function is naive_chunk_gated_delta_rule else {}
        output, state = function(*differentiable_inputs, output_final_state=True, **kwargs)
        gradients.append(
            torch.autograd.grad(
                output.square().mean() + state.square().mean(), differentiable_inputs
            )
        )

    for chunk_gradient, recurrent_gradient in zip(gradients[1], gradients[0]):
        torch.testing.assert_close(chunk_gradient, recurrent_gradient, atol=1e-6, rtol=1e-5)


def test_fused_recurrent_stub_is_explicit():
    inputs = make_inputs(seq_len=1)
    with pytest.raises(NotImplementedError, match="Triton kernel not implemented"):
        fused_recurrent_gated_delta_rule(*inputs[:-1])
