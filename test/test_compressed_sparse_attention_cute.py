import argparse
import importlib
import math

import pytest
import torch
import torch.nn.functional as F

from attn_gym.sparse.compressed_sparse_attention.api import compressed_sparse_attention

pytest.importorskip("flash_attn.cute.interface")

cute_backend = importlib.import_module("attn_gym.sparse.compressed_sparse_attention.cute")

_DSA_PACKED_WORKSPACE_BYTES = cute_backend._DSA_PACKED_WORKSPACE_BYTES
_dsa_head_chunk = cute_backend._dsa_head_chunk
_dsa_tile_shape = cute_backend._dsa_tile_shape
_dsa_workspace_bytes = cute_backend._dsa_workspace_bytes
_require_sm100 = cute_backend._require_sm100


MAX_ABS_ERROR = 3e-2
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

CUTE_PATHOLOGICAL_SHAPES = [
    pytest.param(
        dict(
            heads=1,
            sequence_length=1,
            index_heads=1,
            index_dim=64,
            compression_rate=1,
            topk=1,
            window=1,
            rope_dims=2,
        ),
        id="minimum-valid-shape",
    ),
    pytest.param(
        dict(
            heads=127,
            sequence_length=31,
            index_heads=3,
            index_dim=192,
            compression_rate=32,
            topk=9,
            window=64,
            rope_dims=2,
        ),
        id="head-and-compression-tile-minus-one",
    ),
    pytest.param(
        dict(
            heads=128,
            sequence_length=32,
            index_heads=1,
            index_dim=64,
            compression_rate=32,
            topk=9,
            window=32,
            rope_dims=32,
        ),
        id="head-and-compression-tile-exact",
    ),
    pytest.param(
        dict(
            heads=129,
            sequence_length=33,
            index_heads=5,
            index_dim=320,
            compression_rate=32,
            topk=9,
            window=33,
            rope_dims=64,
        ),
        id="head-and-compression-tile-plus-one",
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
        dict(
            heads=1,
            sequence_length=127,
            index_heads=2,
            index_dim=64,
            compression_rate=2,
            topk=63,
            window=64,
            rope_dims=64,
        ),
        id="sequence-selected-width-and-radix-threshold-minus-one",
    ),
    pytest.param(
        dict(
            heads=1,
            sequence_length=128,
            index_heads=2,
            index_dim=64,
            compression_rate=2,
            topk=64,
            window=64,
            rope_dims=64,
        ),
        id="sequence-selected-width-and-radix-threshold-exact",
    ),
    pytest.param(
        dict(
            heads=1,
            sequence_length=129,
            index_heads=2,
            index_dim=64,
            compression_rate=2,
            topk=64,
            window=65,
            rope_dims=64,
        ),
        id="sequence-and-selected-width-plus-one",
    ),
    pytest.param(
        dict(
            batch=3,
            heads=5,
            sequence_length=17,
            index_heads=5,
            index_dim=512,
            compression_rate=17,
            topk=99,
            window=99,
            rope_dims=512,
        ),
        id="full-rope-and-clamped-selection",
    ),
    pytest.param(
        dict(sequence_length=64, compression_rate=16, topk=0, window=0),
        id="sink-only",
    ),
    pytest.param(dict(topk=0, window=33), id="local-only"),
    pytest.param(dict(topk=9, window=0), id="compressed-only"),
]


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor | int | bool, ...]:
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def randn(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator) * scale

    def query(*shape: int) -> torch.Tensor:
        return F.normalize(randn(*shape), dim=-1)

    kv_heads = 1 if args.share_kv else args.heads
    index_kv_heads = 1 if args.share_kv else args.index_heads
    return (
        query(args.batch, args.heads, args.sequence_length, args.head_dim),
        query(args.batch, args.index_heads, args.sequence_length, args.index_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.compression_rate, args.head_dim),
        randn(args.compression_rate, args.head_dim),
        randn(args.batch, args.sequence_length, args.index_heads),
        randn(args.batch, index_kv_heads, args.sequence_length, args.index_dim),
        randn(args.batch, index_kv_heads, args.sequence_length, args.index_dim),
        randn(args.batch, index_kv_heads, args.sequence_length, args.index_dim),
        randn(args.batch, index_kv_heads, args.sequence_length, args.index_dim),
        randn(args.compression_rate, args.index_dim),
        randn(args.compression_rate, args.index_dim),
        1.0 + randn(args.head_dim, scale=0.05),
        1.0 + randn(args.index_dim, scale=0.05),
        1.0 + randn(args.head_dim, scale=0.05),
        randn(args.heads),
        args.compression_rate,
        args.topk,
        args.window,
        args.rope_dims,
        args.share_kv,
    )


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
def test_cute_d512_matches_reference():
    inputs = _inputs("bfloat16")
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
def test_cute_batch4_bf16_matches_reference():
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
    CUTE_PATHOLOGICAL_SHAPES,
)
def test_cute_generalized_shapes_match_reference(overrides):
    inputs = _inputs("bfloat16", **overrides)
    with torch.inference_mode():
        expected = compressed_sparse_attention(*inputs, backend="eager")
        actual = compressed_sparse_attention(*inputs, backend="cute")

    assert actual.shape == expected.shape
    assert (actual.float() - expected.float()).abs().max().item() <= MAX_ABS_ERROR


def assert_backward_matches_reference(base):
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
    grad_output = (
        torch.randn(actual.shape, device=actual.device, dtype=actual.dtype, generator=generator)
        * 0.01
    )
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
@pytest.mark.parametrize(
    "workspace_budget",
    [None, 4 * 1024**2],
    ids=["single-tile", "token-tiled"],
)
def test_cute_backward_matches_reference(workspace_budget, monkeypatch):
    if workspace_budget is not None:
        monkeypatch.setattr(cute_backend, "_DSA_PACKED_WORKSPACE_BYTES", workspace_budget)
    base = _inputs(
        "bfloat16",
        heads=64,
        sequence_length=35,
        index_heads=4,
        compression_rate=16,
        topk=2,
        window=16,
    )
    assert_backward_matches_reference(base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
@pytest.mark.parametrize("heads", [127, 129])
def test_cute_backward_handles_pathological_head_counts(heads):
    base = _inputs(
        "bfloat16",
        heads=heads,
        sequence_length=17,
        index_heads=1,
        index_dim=64,
        compression_rate=8,
        topk=3,
        window=9,
        rope_dims=64,
    )
    assert_backward_matches_reference(base)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0),
    reason="the CuTe backend targets SM100 exclusively",
)
def test_cute_rejects_fp16():
    with pytest.raises(TypeError, match="bfloat16 inputs only"):
        compressed_sparse_attention(*_inputs("float16"), backend="cute")


def test_dsa_head_chunk_respects_workspace_budget_below_64_heads():
    tokens = 8192
    dim = 512
    total_kv = tokens + math.ceil(tokens / 32)
    chunk = _dsa_head_chunk(tokens, dim, 128, total_kv)

    assert 0 < chunk < 64
    assert _dsa_workspace_bytes(tokens, dim, chunk, total_kv) <= (_DSA_PACKED_WORKSPACE_BYTES)
    assert _dsa_workspace_bytes(tokens, dim, chunk + 1, total_kv) > (_DSA_PACKED_WORKSPACE_BYTES)

    head_tile, token_tile = _dsa_tile_shape(tokens, dim, 128, total_kv)
    assert head_tile >= 64
    assert 0 < token_tile < tokens
    tiled_bytes = _dsa_workspace_bytes(token_tile, dim, head_tile, total_kv)
    assert tiled_bytes <= _DSA_PACKED_WORKSPACE_BYTES


def test_cute_rejects_non_sm100(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (10, 3))
    with pytest.raises(RuntimeError, match="SM100 exclusively"):
        _require_sm100(torch.device("cuda"))
