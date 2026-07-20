import argparse
import math

import pytest
import torch

from attn_gym.sparse import compressed_sparse_attention

pytest.importorskip("flash_attn.cute.interface")

from attn_gym.sparse.compressed_sparse_attention.cute import (
    _radix_topk_indices,
    _require_sm100,
)
from attn_gym.sparse.compressed_sparse_attention.cute.kernels import (
    compile_index_scores,
    compile_index_topk,
    compile_selected_gather,
    cute_dtype,
)
from benchmarks.sparse.compressed_sparse_attention.benchmark_compressed_sparse_attention_triton import (
    make_inputs,
)


MAX_ABS_ERROR = 1e-2


def _inputs(dtype: str, **overrides):
    configuration = dict(
        batch=1,
        heads=128,
        sequence_length=128,
        head_dim=512,
        index_heads=4,
        index_dim=64,
        compression_rate=32,
        topk=8,
        window=128,
        rope_dims=64,
        dtype=dtype,
        share_kv=True,
        seed=123,
    )
    configuration.update(overrides)
    return make_inputs(argparse.Namespace(**configuration))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_cute_d512_matches_reference_within_one_e_minus_two(dtype):
    inputs = _inputs(dtype)
    with torch.inference_mode():
        expected = compressed_sparse_attention(*inputs, backend="eager")
        actual = compressed_sparse_attention(*inputs, backend="cute")

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert (actual.float() - expected.float()).abs().max().item() <= MAX_ABS_ERROR


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
def test_cute_batch4_bf16_matches_reference_within_one_e_minus_two():
    inputs = _inputs("bfloat16", batch=4)
    with torch.inference_mode():
        expected = compressed_sparse_attention(*inputs, backend="eager")
        actual = compressed_sparse_attention(*inputs, backend="cute")

    assert (actual.float() - expected.float()).abs().max().item() <= MAX_ABS_ERROR


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param(
            dict(
                heads=129,
                sequence_length=65,
                index_heads=3,
                index_dim=320,
                compression_rate=17,
                topk=9,
                window=33,
                rope_dims=64,
            ),
            id="arbitrary-shapes-and-head-tail",
        ),
        pytest.param(
            dict(
                heads=128,
                sequence_length=65,
                index_heads=3,
                index_dim=320,
                compression_rate=17,
                topk=9,
                window=33,
                rope_dims=320,
            ),
            id="rope-tail-crosses-both-mma-splits",
        ),
        pytest.param(
            dict(sequence_length=64, compression_rate=16, topk=0, window=0),
            id="sink-only",
        ),
    ],
)
def test_cute_generalized_shapes_match_reference(overrides):
    inputs = _inputs("float16", **overrides)
    with torch.inference_mode():
        expected = compressed_sparse_attention(*inputs, backend="eager")
        actual = compressed_sparse_attention(*inputs, backend="cute")

    assert actual.shape == expected.shape
    assert (actual.float() - expected.float()).abs().max().item() <= MAX_ABS_ERROR


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
@pytest.mark.parametrize(
    ("topk", "tied_scores", "index_heads", "index_dim"),
    [
        pytest.param(64, False, 4, 64, id="topk64-common-index-shape"),
        pytest.param(96, False, 4, 64, id="topk96-common-index-shape"),
        pytest.param(96, True, 4, 64, id="topk96-tied-scores"),
        pytest.param(64, False, 3, 320, id="topk64-general-index-shape"),
    ],
)
def test_cute_radix_topk_matches_insertion_selection(
    topk, tied_scores, index_heads, index_dim
):
    batch = 1
    sequence = 800
    compression_rate = 8
    window = 64
    rope_dims = 64
    num_blocks = math.ceil(sequence / compression_rate)
    gather_length = math.ceil((topk + window) / 128) * 128
    dtype = torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(123)
    q_i = torch.randn(
        batch,
        index_heads,
        sequence,
        index_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    compressed_indices = torch.randn(
        batch,
        num_blocks,
        1,
        index_dim,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    weights = torch.randn(
        batch,
        sequence,
        index_heads,
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    if tied_scores:
        weights.zero_()
    cos = torch.randn(
        sequence,
        rope_dims // 2,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    sin = torch.randn_like(cos)
    score_keys = torch.empty(
        batch * sequence,
        num_blocks,
        device="cuda",
        dtype=torch.float32,
    )
    completed_lengths = torch.empty(
        batch * sequence,
        device="cuda",
        dtype=torch.int32,
    )
    actual = torch.empty(
        batch,
        sequence,
        gather_length,
        device="cuda",
        dtype=torch.int32,
    )
    expected = torch.empty_like(actual)

    score_kernel = compile_index_scores(
        cute_dtype(q_i),
        batch,
        sequence,
        index_heads,
        index_dim,
        num_blocks,
        compression_rate,
        rope_dims,
    )
    score_kernel(
        q_i,
        compressed_indices,
        weights,
        cos,
        sin,
        score_keys,
        completed_lengths,
    )
    selected_indices = _radix_topk_indices(
        score_keys,
        completed_lengths,
        topk,
    )
    compile_selected_gather(
        batch,
        sequence,
        num_blocks,
        topk,
        window,
        gather_length,
    )(selected_indices, actual)
    compile_index_topk(
        cute_dtype(q_i),
        batch,
        sequence,
        index_heads,
        index_dim,
        num_blocks,
        compression_rate,
        topk,
        window,
        rope_dims,
        gather_length,
    )(q_i, compressed_indices, weights, cos, sin, expected)

    assert torch.equal(
        actual[:, :, :topk].sort(dim=-1).values,
        expected[:, :, :topk].sort(dim=-1).values,
    )
    assert torch.equal(actual[:, :, topk:], expected[:, :, topk:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_cute_backward_matches_reference_within_one_e_minus_two(dtype):
    base = _inputs(
        dtype,
        heads=8,
        sequence_length=35,
        index_heads=4,
        compression_rate=16,
        topk=2,
        window=16,
    )

    def differentiable_copy():
        return tuple(
            value.detach().clone().requires_grad_(True)
            if isinstance(value, torch.Tensor)
            else value
            for value in base
        )

    reference_inputs = differentiable_copy()
    cute_inputs = differentiable_copy()
    expected = compressed_sparse_attention(*reference_inputs, backend="eager")
    actual = compressed_sparse_attention(*cute_inputs, backend="cute")
    saved = actual.grad_fn.saved_tensors
    input_pointers = {value.data_ptr() for value in cute_inputs[:20]}
    assert len(saved) == 20
    assert all(value.data_ptr() in input_pointers for value in saved)
    generator = torch.Generator(device=actual.device).manual_seed(456)
    grad_output = torch.randn(
        actual.shape, device=actual.device, dtype=actual.dtype, generator=generator
    ) * 0.01
    expected.backward(grad_output)
    actual.backward(grad_output)

    differentiable = {0, 2, 3, 4, 5, 6, 7, 8, 16, 18, 19}
    for index, (cute_input, reference_input) in enumerate(
        zip(cute_inputs[:20], reference_inputs[:20])
    ):
        if index in differentiable:
            assert cute_input.grad is not None
            assert reference_input.grad is not None
            error = (cute_input.grad.float() - reference_input.grad.float()).abs().max()
            assert error.item() <= MAX_ABS_ERROR, f"input {index} max error {error.item()}"
        else:
            assert cute_input.grad is None
            assert reference_input.grad is None


def test_cute_rejects_non_sm100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 3))
    with pytest.raises(RuntimeError, match="SM100 exclusively"):
        _require_sm100(torch.device("cuda"))
