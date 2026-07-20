import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse import compressed_sparse_attention

MAX_ABS_ERROR = 1e-2
DIFFERENTIABLE_INPUTS = (0, 2, 3, 4, 5, 6, 7, 8, 16, 18, 19)


def make_inputs(
    share_kv: bool,
    dtype: torch.dtype,
    *,
    batch: int = 2,
    heads: int = 4,
    sequence_length: int = 19,
    head_dim: int = 32,
    index_heads: int = 2,
    index_dim: int = 16,
    compression_rate: int = 4,
    topk: int = 3,
    window: int = 5,
    rope_dims: int = 8,
    scale: float = 0.2,
):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(123)

    def randn(*shape, value_scale=scale):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator) * value_scale

    def query(*shape):
        return F.normalize(randn(*shape), dim=-1)

    kv_heads = 1 if share_kv else heads
    index_kv_heads = 1 if share_kv else index_heads
    return (
        query(batch, heads, sequence_length, head_dim),
        query(batch, index_heads, sequence_length, index_dim),
        randn(batch, kv_heads, sequence_length, head_dim),
        randn(batch, kv_heads, sequence_length, head_dim),
        randn(batch, kv_heads, sequence_length, head_dim),
        randn(batch, kv_heads, sequence_length, head_dim),
        randn(batch, kv_heads, sequence_length, head_dim),
        randn(compression_rate, head_dim),
        randn(compression_rate, head_dim),
        randn(batch, sequence_length, index_heads),
        randn(batch, index_kv_heads, sequence_length, index_dim),
        randn(batch, index_kv_heads, sequence_length, index_dim),
        randn(batch, index_kv_heads, sequence_length, index_dim),
        randn(batch, index_kv_heads, sequence_length, index_dim),
        randn(compression_rate, index_dim),
        randn(compression_rate, index_dim),
        1.0 + randn(head_dim, value_scale=0.05),
        1.0 + randn(index_dim, value_scale=0.05),
        1.0 + randn(head_dim, value_scale=0.05),
        randn(heads),
        compression_rate,
        topk,
        window,
        rope_dims,
        share_kv,
    )


def assert_matches_reference(inputs):
    with torch.inference_mode():
        expected = compressed_sparse_attention(*inputs, backend="eager")
        actual = compressed_sparse_attention(*inputs, backend="triton")

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    max_abs_error = (actual.float() - expected.float()).abs().max().item()
    assert max_abs_error <= MAX_ABS_ERROR


def _with_gradients(inputs):
    return tuple(
        value.detach().clone().requires_grad_(index in DIFFERENTIABLE_INPUTS)
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(inputs)
    )


def _gradients(inputs, backend, grad_output):
    output = compressed_sparse_attention(*inputs, backend=backend)
    targets = tuple(inputs[index] for index in DIFFERENTIABLE_INPUTS)
    return torch.autograd.grad(output, targets, grad_outputs=grad_output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_matches_reference_within_one_e_minus_two(share_kv, dtype):
    assert_matches_reference(make_inputs(share_kv, dtype))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_matches_reference_across_query_and_window_tiles():
    inputs = make_inputs(
        True,
        torch.float32,
        batch=1,
        heads=2,
        sequence_length=137,
        head_dim=48,
        index_heads=2,
        index_dim=24,
        compression_rate=8,
        topk=64,
        window=131,
        rope_dims=16,
    )
    assert_matches_reference(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_handles_no_selected_or_local_keys():
    inputs = make_inputs(
        False,
        torch.float32,
        batch=1,
        heads=2,
        sequence_length=17,
        compression_rate=32,
        topk=0,
        window=0,
    )
    assert_matches_reference(inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_accepts_mixed_shared_and_expanded_heads():
    inputs = list(make_inputs(True, torch.float32, batch=1, heads=2, index_heads=2))
    inputs[4] = inputs[4].expand(-1, 2, -1, -1).contiguous()
    inputs[12] = inputs[12].expand(-1, 2, -1, -1).contiguous()
    assert_matches_reference(tuple(inputs))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("share_kv", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_triton_backward_matches_reference(share_kv, dtype):
    inputs = make_inputs(
        share_kv,
        dtype,
        batch=1,
        heads=2,
        sequence_length=17,
        head_dim=32,
        index_heads=2,
        index_dim=16,
        compression_rate=4,
        topk=3,
        window=5,
        rope_dims=8,
    )
    generator = torch.Generator(device="cuda").manual_seed(321)
    grad_output = torch.randn(
        inputs[0].shape,
        device="cuda",
        dtype=dtype,
        generator=generator,
    ) * 0.01
    expected = _gradients(_with_gradients(inputs), "eager", grad_output)
    actual = _gradients(_with_gradients(inputs), "triton", grad_output)

    for index, (actual_gradient, expected_gradient) in enumerate(zip(actual, expected)):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-2, rtol=1e-2), (
            f"gradient for input {DIFFERENTIABLE_INPUTS[index]} differs; "
            f"max abs error "
            f"{(actual_gradient.float() - expected_gradient.float()).abs().max().item():.6g}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("topk", "window"),
    [
        pytest.param(0, 0, id="sink-only"),
        pytest.param(3, 0, id="compressed-only"),
        pytest.param(0, 5, id="local-only"),
    ],
)
def test_triton_backward_selection_edge_cases(topk, window):
    inputs = make_inputs(
        True,
        torch.float32,
        batch=1,
        heads=2,
        sequence_length=17,
        head_dim=32,
        index_heads=2,
        index_dim=16,
        compression_rate=4,
        topk=topk,
        window=window,
        rope_dims=8,
    )
    grad_output = torch.randn_like(inputs[0]) * 0.01
    expected = _gradients(_with_gradients(inputs), "eager", grad_output)
    actual = _gradients(_with_gradients(inputs), "triton", grad_output)

    for actual_gradient, expected_gradient in zip(actual, expected):
        assert torch.allclose(actual_gradient, expected_gradient, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_triton_backward_leaves_discrete_indexer_inputs_without_gradients():
    inputs = make_inputs(
        True,
        torch.float32,
        batch=1,
        heads=2,
        sequence_length=17,
        head_dim=32,
        index_heads=2,
        index_dim=16,
        compression_rate=4,
        topk=3,
        window=5,
        rope_dims=8,
    )
    inputs = tuple(
        value.detach().clone().requires_grad_(True)
        if isinstance(value, torch.Tensor)
        else value
        for value in inputs
    )
    output = compressed_sparse_attention(*inputs, backend="triton")
    output.backward(torch.randn_like(output) * 0.01)

    differentiable = set(DIFFERENTIABLE_INPUTS)
    for index, value in enumerate(inputs[:20]):
        assert (value.grad is not None) == (index in differentiable)
