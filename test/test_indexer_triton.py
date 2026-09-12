"""Exercise TMA index selection without requiring the optional CuTeDSL backend."""

import pytest
import torch

from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.testing.indexer import assert_indexer_selection, make_indexer_test_inputs


@pytest.fixture(autouse=True)
def require_hopper() -> None:
    """Skip before importing Triton on hosts without a supported NVIDIA GPU."""
    if torch.version.hip or not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer required")
    pytest.importorskip("triton")
    pytest.importorskip("triton.tools.tensor_descriptor")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,heads,dim,topk,causal",
    [
        (65, 3, 16, 0, True),
        (65, 5, 48, 1, False),
        (129, 17, 128, 37, True),
        (129, 3, 48, 37, False),
        (257, 5, 16, 129, False),
        (513, 3, 128, 512, True),
        (1, 2, 8, 1, True),
        (65, 256, 256, 37, False),
    ],
)
def test_triton_selection(dtype, tokens, heads, dim, topk, causal):
    """Cover partial tiles, multi-tile selection and all supported score dtypes."""
    q, k, weights = make_indexer_test_inputs(tokens, heads, dim, dtype)
    actual = lightning_indexer(
        q, k, weights, topk, causal=causal, impl="fused", kernel_options={"backend": "triton"}
    )
    assert_indexer_selection(actual, q, k, weights, topk, causal)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_triton_selection_across_score_magnitudes(dtype):
    """Exercise relative-RMS budgets when query rows span six orders of magnitude."""
    q, k, weights = make_indexer_test_inputs(65, 3, 48, dtype)
    q.mul_(torch.logspace(-3, 3, 65, device=q.device).view(1, 65, 1, 1))
    actual = lightning_indexer(q, k, weights, 37, kernel_options={"backend": "triton"})
    assert_indexer_selection(actual, q, k, weights, 37, False)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_zero_weight_ties(dtype, causal):
    """Ties preserve validity, counts, uniqueness and padding without fixing order."""
    q, k, weights = make_indexer_test_inputs(129, 3, 48, dtype)
    weights.zero_()
    actual = lightning_indexer(
        q, k, weights, 37, causal=causal, impl="fused", kernel_options={"backend": "triton"}
    )
    assert_indexer_selection(actual, q, k, weights, 37, causal)


@pytest.mark.parametrize("causal", [False, True])
def test_triton_strided_inputs(causal):
    """Aligned noncompact outer strides and nonunit weight strides are supported."""
    q, k, weights = make_indexer_test_inputs(130, 6, 48, torch.bfloat16)
    q, k, weights = q[:, ::2, ::2], k[:, ::2], weights[:, ::2, ::2]
    assert all(not tensor.is_contiguous() for tensor in (q, k, weights))
    actual = lightning_indexer(
        q, k, weights, 37, causal=causal, impl="fused", kernel_options={"backend": "triton"}
    )
    assert_indexer_selection(actual, q, k, weights, 37, causal)
    from attn_gym.sparse.indexer.ops import _indexer_op

    torch.library.opcheck(_indexer_op, (q, k, weights, 37, causal, 1, "triton"))


@pytest.mark.parametrize(
    "layout,tokens,heads,dim,topk,dtype,error,message",
    [
        (None, 65, 3, 16, 1, torch.float32, TypeError, "FP16 or BF16"),
        (None, 65, 3, 18, 1, torch.bfloat16, ValueError, "H <= 256"),
        (None, 65, 3, 264, 1, torch.bfloat16, ValueError, "H <= 256"),
        (None, 65, 257, 16, 1, torch.bfloat16, ValueError, "H <= 256"),
        ("q", 65, 3, 16, 1, torch.bfloat16, ValueError, "contiguous last dimension"),
        ("k", 65, 3, 16, 1, torch.bfloat16, ValueError, "contiguous last dimension"),
        ("outer", 65, 3, 17, 1, torch.bfloat16, ValueError, "16-byte outer strides"),
    ],
    ids=[
        "dtype",
        "dim_unaligned",
        "dim_large",
        "heads",
        "q_layout",
        "k_layout",
        "outer_stride",
    ],
)
def test_triton_unsupported_inputs(layout, tokens, heads, dim, topk, dtype, error, message):
    """Reject unsupported arithmetic and TMA layouts at the public boundary."""
    q, k, weights = make_indexer_test_inputs(tokens, heads, dim, dtype)
    match layout:
        case "q":
            q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
        case "k":
            k = k.transpose(-1, -2).contiguous().transpose(-1, -2)
        case "outer":
            q, k = q[..., :16], k[..., :16]
    with pytest.raises(error, match=message):
        lightning_indexer(q, k, weights, topk, impl="fused", kernel_options={"backend": "triton"})


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("topk", [0, 37])
def test_triton_opcheck(dtype, requires_grad, topk):
    """Validate schema, autograd, fake tensors and dynamic AOT dispatch."""
    from attn_gym.sparse.indexer.ops import _indexer_op

    inputs = make_indexer_test_inputs(65, 3, 48, dtype)
    for tensor in inputs:
        tensor.requires_grad_(requires_grad)
    torch.library.opcheck(_indexer_op, (*inputs, topk, True, 1, "triton"))
    actual = lightning_indexer(
        *inputs, topk, causal=True, impl="fused", kernel_options={"backend": "triton"}
    )
    assert not actual.requires_grad
    assert actual.grad_fn is None


@pytest.mark.parametrize("topk", [0, 37])
def test_triton_dynamic_fullgraph(topk):
    """Reuse the compiled public callable across odd sequence lengths."""
    compiled = torch.compile(lightning_indexer, fullgraph=True, dynamic=True)
    for tokens in (65, 129):
        inputs = make_indexer_test_inputs(tokens, 3, 48, torch.bfloat16)
        expected = lightning_indexer(
            *inputs, topk, causal=True, impl="fused", kernel_options={"backend": "triton"}
        )
        actual = compiled(
            *inputs, topk, causal=True, impl="fused", kernel_options={"backend": "triton"}
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_triton_wide_offsets(monkeypatch):
    """Exercise the wide specialization even when all actual offsets fit int32."""
    from attn_gym.sparse.indexer.impl import triton as implementation

    inputs = make_indexer_test_inputs(65, 3, 48, torch.bfloat16)
    assert not implementation.requires_int64_offsets(*inputs)
    expected = lightning_indexer(
        *inputs, 37, causal=True, impl="fused", kernel_options={"backend": "triton"}
    )
    monkeypatch.setattr(implementation, "requires_int64_offsets", lambda *tensors: True)
    actual = lightning_indexer(
        *inputs, 37, causal=True, impl="fused", kernel_options={"backend": "triton"}
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("operand", [0, 1])
def test_triton_misaligned_base(operand):
    """A compact tensor with an offset of one element is not a valid TMA base."""
    inputs = list(make_indexer_test_inputs(65, 3, 48, torch.bfloat16))
    tensor = inputs[operand]
    storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
    inputs[operand] = storage[1:].view(tensor.shape).copy_(tensor)
    with pytest.raises(ValueError, match="16-byte-aligned Q/K"):
        lightning_indexer(*inputs, 37, impl="fused", kernel_options={"backend": "triton"})
