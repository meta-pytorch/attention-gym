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


def differentiable_copy(inputs):
    return tuple(
        value.detach().clone().requires_grad_(index in DIFFERENTIABLE_INPUTS)
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(inputs)
    )


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
    base = make_inputs(
        share_kv,
        dtype,
        batch=1,
        heads=2,
        sequence_length=65,
        head_dim=32,
        index_heads=2,
        index_dim=16,
        compression_rate=8,
        topk=3,
        window=33,
        rope_dims=8,
    )
    reference_inputs = differentiable_copy(base)
    triton_inputs = differentiable_copy(base)
    reference_targets = tuple(reference_inputs[index] for index in DIFFERENTIABLE_INPUTS)
    triton_targets = tuple(triton_inputs[index] for index in DIFFERENTIABLE_INPUTS)

    expected = compressed_sparse_attention(*reference_inputs, backend="eager")
    actual = compressed_sparse_attention(*triton_inputs, backend="triton")
    generator = torch.Generator(device=actual.device).manual_seed(456)
    grad_output = (
        torch.randn(
            actual.shape,
            device=actual.device,
            dtype=actual.dtype,
            generator=generator,
        )
        * 0.01
    )
    expected_gradients = torch.autograd.grad(expected, reference_targets, grad_output)
    actual_gradients = torch.autograd.grad(actual, triton_targets, grad_output)

    for input_index, actual_gradient, expected_gradient in zip(
        DIFFERENTIABLE_INPUTS,
        actual_gradients,
        expected_gradients,
    ):
        error = (actual_gradient.float() - expected_gradient.float()).abs().max().item()
        assert error <= MAX_ABS_ERROR, f"input {input_index} max gradient error {error}"
