"""CPU coverage for the public selected-attention metadata contract."""

import pytest
import torch

from attn_gym.sparse.selected_attention import AuxRequest, selected_attention


@pytest.fixture
def inputs() -> dict[str, torch.Tensor]:
    """Build a small, nontrivial shared-KV attention input."""
    generator = torch.Generator().manual_seed(42)
    return {
        "query": torch.randn(1, 2, 4, 3, generator=generator),
        "local_kv": torch.randn(1, 1, 4, 3, generator=generator),
        "sparse_kv": torch.randn(1, 1, 2, 3, generator=generator),
        "kv_indices": torch.tensor([[[0], [1], [-1], [0]]]),
        "attention_sink": torch.randn(2, generator=generator),
        "doc_ids": torch.tensor([[0, 0, 1, 1]]),
    }


@pytest.mark.parametrize(
    "name", ["query", "local_kv", "sparse_kv", "kv_indices", "attention_sink", "doc_ids"]
)
def test_non_tensor_input(inputs, name):
    """Malformed inputs report their own name rather than incidental attribute errors."""
    inputs[name] = []
    with pytest.raises(TypeError, match=rf"{name} must be a torch.Tensor"):
        selected_attention(**inputs, backend="eager")


@pytest.mark.parametrize(
    "name", ["query", "local_kv", "sparse_kv", "kv_indices", "attention_sink", "doc_ids"]
)
@pytest.mark.parametrize("rank", [0, 1, 5])
def test_invalid_rank(inputs, name, rank):
    """Rank errors are raised before indexing a malformed tensor's shape."""
    inputs[name] = torch.zeros((1,) * rank, dtype=inputs[name].dtype)
    with pytest.raises(ValueError, match=rf"{name} must have shape"):
        selected_attention(**inputs, backend="eager")


@pytest.mark.parametrize("mode", ["", "invalid", None])
def test_invalid_mode(inputs, mode):
    """Only the two documented modes are accepted."""
    with pytest.raises(ValueError, match="mode must be 'auto' or 'chunked'"):
        selected_attention(**inputs, mode=mode, backend="eager")


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("mode", ["auto", "chunked"])
def test_public_reference_fullgraph(inputs, dynamic, mode):
    """Public validation supports fullgraph reference outputs, LSE, and gradients."""
    differentiable = tuple(tensor for tensor in inputs.values() if tensor.is_floating_point())
    for tensor in differentiable:
        tensor.requires_grad_()
    compiled = torch.compile(selected_attention, backend="eager", fullgraph=True, dynamic=dynamic)
    expected, expected_aux = selected_attention(
        **inputs, backend="eager", return_aux=AuxRequest(lse=True)
    )
    actual, actual_aux = compiled(
        **inputs, backend="eager", mode=mode, return_aux=AuxRequest(lse=True)
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_aux.lse, expected_aux.lse)
    expected_gradients = torch.autograd.grad(expected.square().sum(), differentiable)
    actual_gradients = torch.autograd.grad(actual.square().sum(), differentiable)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("compile_call", [False, True])
@pytest.mark.parametrize("window", [-1, True, 1.5])
def test_invalid_window(inputs, compile_call, window):
    """Metadata validation is not bypassed while Dynamo traces the public API."""
    call = (
        torch.compile(selected_attention, backend="eager") if compile_call else selected_attention
    )
    error = ValueError if window == -1 else TypeError
    with pytest.raises(error, match="sliding_window_size must"):
        call(**inputs, sliding_window_size=window, backend="eager")
