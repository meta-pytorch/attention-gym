"""CPU coverage for the public gather-attention metadata contract."""

import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, Impl, gather_attn


@pytest.fixture
def inputs() -> dict[str, torch.Tensor]:
    """Build a small, nontrivial shared-KV attention input."""
    generator = torch.Generator().manual_seed(42)
    return {
        "query": torch.randn(1, 2, 4, 3, generator=generator),
        "local_kv": torch.randn(1, 1, 4, 3, generator=generator),
        "sparse_kv": torch.randn(1, 1, 2, 3, generator=generator),
        "kv_indices": torch.tensor([[[0], [0], [-1], [0]]]),
        "attention_sink": torch.randn(2, generator=generator),
        "cu_seqlens": torch.tensor([0, 2, 4], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 1, 2], dtype=torch.int32),
    }


@pytest.mark.parametrize(
    "name",
    [
        "query",
        "local_kv",
        "sparse_kv",
        "kv_indices",
        "attention_sink",
        "cu_seqlens",
        "cu_seqlens_k",
    ],
)
def test_non_tensor_input(inputs, name):
    """Malformed inputs report their own name rather than incidental attribute errors."""
    inputs[name] = []
    with pytest.raises(TypeError, match=rf"{name} must be a torch.Tensor"):
        gather_attn(**inputs, impl=Impl.REFERENCE)


@pytest.mark.parametrize(
    "name",
    [
        "query",
        "local_kv",
        "sparse_kv",
        "kv_indices",
        "attention_sink",
        "cu_seqlens",
        "cu_seqlens_k",
    ],
)
@pytest.mark.parametrize("rank", [0, 1, 5])
def test_invalid_rank(inputs, name, rank):
    """Rank errors are raised before indexing a malformed tensor's shape."""
    inputs[name] = torch.zeros((1,) * rank, dtype=inputs[name].dtype)
    with pytest.raises(ValueError, match=rf"{name} must have shape"):
        gather_attn(**inputs, impl=Impl.REFERENCE)


@pytest.mark.parametrize("impl", ["", "eager", "invalid", None])
def test_invalid_impl(inputs, impl):
    with pytest.raises(ValueError, match="unknown impl"):
        gather_attn(**inputs, impl=impl)


@pytest.mark.parametrize("impl", [Impl.FUSED, "fused"])
def test_fused_requires_cuda(inputs, impl):
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        gather_attn(**inputs, impl=impl)


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "auto"},
        {"backend": "eager"},
        {"backend": None},
        {"other": "triton"},
        {"backend": "triton", "other": "x"},
    ],
)
def test_invalid_kernel_options(inputs, options):
    with pytest.raises(ValueError, match="unsupported gather_attn kernel options"):
        gather_attn(**inputs, kernel_options=options)
    with pytest.raises(ValueError, match="kernel_options are not supported"):
        gather_attn(**inputs, impl=Impl.REFERENCE, kernel_options=options)


@pytest.mark.parametrize("backend", ["cute", "triton"])
def test_reference_rejects_backend_options(inputs, backend):
    with pytest.raises(ValueError, match="kernel_options are not supported"):
        gather_attn(**inputs, impl=Impl.REFERENCE, kernel_options={"backend": backend})


@pytest.mark.usefixtures("fresh_compile_cache")
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("impl", [Impl.REFERENCE, "reference"])
def test_public_reference_fullgraph(inputs, dynamic, impl):
    """Public validation supports fullgraph reference outputs, LSE, and gradients."""
    differentiable = tuple(tensor for tensor in inputs.values() if tensor.is_floating_point())
    for tensor in differentiable:
        tensor.requires_grad_()
    compiled = torch.compile(gather_attn, backend="eager", fullgraph=True, dynamic=dynamic)
    expected, expected_aux = gather_attn(
        **inputs, impl=Impl.REFERENCE, return_aux=AuxRequest(lse=True)
    )
    actual, actual_aux = compiled(**inputs, impl=impl, return_aux=AuxRequest(lse=True))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_aux.lse, expected_aux.lse)
    expected_gradients = torch.autograd.grad(expected.square().sum(), differentiable)
    actual_gradients = torch.autograd.grad(actual.square().sum(), differentiable)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.usefixtures("fresh_compile_cache")
@pytest.mark.parametrize("compile_call", [False, True])
@pytest.mark.parametrize("window", [-1, True, 1.5])
def test_invalid_window(inputs, compile_call, window):
    """Metadata validation is not bypassed while Dynamo traces the public API."""
    call = torch.compile(gather_attn, backend="eager") if compile_call else gather_attn
    error = ValueError if window == -1 else TypeError
    with pytest.raises(error, match="sliding_window_size must"):
        call(**inputs, sliding_window_size=window, impl=Impl.REFERENCE)
