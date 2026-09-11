"""Exercise TMA index selection without requiring the optional CuTeDSL backend."""

import math

import pytest
import torch

from attn_gym.sparse.indexer import lightning_indexer


@pytest.fixture(autouse=True)
def require_hopper() -> None:
    """Skip before importing Triton on hosts without a supported NVIDIA GPU."""
    if torch.version.hip or not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer required")
    pytest.importorskip("triton")
    pytest.importorskip("triton.tools.tensor_descriptor")


def make_inputs(
    tokens: int, heads: int, dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use quantized random inputs and guarantee both signs of head weights."""
    torch.manual_seed(77)
    q = torch.randn(2, tokens, heads, dim, device="cuda", dtype=dtype)
    k = torch.randn(2, tokens, dim, device="cuda", dtype=dtype)
    weights = torch.randn(2, tokens, heads, device="cuda", dtype=dtype)
    weights[..., 0] = weights[..., 0].abs() + 0.25
    weights[..., 1] = -weights[..., 1].abs() - 0.25
    return q, k, weights


def assert_selection(
    actual: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool,
) -> None:
    """Check index invariants and FP64 boundary regret against eager's regret.

    The allowance bounds FP32 dot, weighting, head reduction and scaling errors
    using absolute products (so cancellation cannot shrink the budget). Twice
    that score error bounds a selection boundary swap; no FP16/BF16 output
    rounding is allowed because the kernel retains scores in FP32.
    """
    batch, tokens, heads, dim = q.shape
    assert actual.shape == (batch, tokens, topk)
    assert actual.dtype == torch.int32
    assert actual.device == q.device
    assert not actual.requires_grad
    valid = actual >= 0
    assert not ((actual < -1) | (actual >= tokens)).any()
    row = torch.arange(tokens, device=q.device).view(1, tokens, 1)
    counts = (row + 1).clamp_max(topk) if causal else torch.full_like(row, topk)
    assert torch.equal(valid.sum(-1, keepdim=True), counts.expand(batch, -1, -1))
    if causal:
        assert not (valid & (actual > row)).any()
    ordered = actual.sort(-1).values
    assert not ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any()
    if topk == 0:
        return

    q64, k64, w64 = q.double(), k.double(), weights.double()
    dots = q64.permute(0, 2, 1, 3) @ k64.transpose(-1, -2).unsqueeze(1)
    scores = (dots.relu() * w64.transpose(1, 2).unsqueeze(-1)).sum(1)
    scores /= math.sqrt(heads * dim)
    assert torch.isfinite(scores).all()
    absolute_dots = q64.abs().permute(0, 2, 1, 3) @ k64.abs().transpose(-1, -2).unsqueeze(1)
    magnitude = (absolute_dots * w64.abs().transpose(1, 2).unsqueeze(-1)).sum(1)
    magnitude /= math.sqrt(heads * dim)
    if causal:
        future = torch.arange(tokens, device=q.device).view(1, 1, tokens) > row
        scores.masked_fill_(future, -torch.inf)
        magnitude.masked_fill_(future, 0)
    boundary = scores.sort(-1, descending=True).values.gather(
        -1, (counts - 1).expand(batch, -1, -1)
    )
    eager = lightning_indexer(q, k, weights, topk, causal=causal, impl="reference")
    actual_scores = scores.gather(-1, actual.long().clamp_min(0))
    eager_scores = scores.gather(-1, eager.long().clamp_min(0))
    actual_error = (boundary - actual_scores).clamp_min(0).masked_fill(~valid, 0)
    eager_error = (boundary - eager_scores).clamp_min(0).masked_fill(eager < 0, 0)
    reduction_eps = (dim + heads + 2) * torch.finfo(torch.float32).eps
    allowance = 2 * reduction_eps / (1 - reduction_eps) * magnitude.amax(-1, keepdim=True)
    assert (
        actual_error.amax(-1, keepdim=True) <= eager_error.amax(-1, keepdim=True) + allowance
    ).all()
    assert actual_error.mean() <= eager_error.mean() + allowance.mean()


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
    q, k, weights = make_inputs(tokens, heads, dim, dtype)
    actual = lightning_indexer(
        q, k, weights, topk, causal=causal, impl="fused", kernel_options={"backend": "triton"}
    )
    assert_selection(actual, q, k, weights, topk, causal)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal", [False, True])
def test_triton_zero_weight_ties(dtype, causal):
    """Exact ties select ascending indices and leave causal tail slots as -1."""
    q, k, weights = make_inputs(129, 3, 48, dtype)
    weights.zero_()
    actual = lightning_indexer(
        q, k, weights, 37, causal=causal, impl="fused", kernel_options={"backend": "triton"}
    )
    expected = torch.arange(37, device=q.device, dtype=torch.int32).expand(2, 129, -1)
    if causal:
        expected = expected.masked_fill(
            expected > torch.arange(129, device=q.device).view(1, 129, 1), -1
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("causal", [False, True])
def test_triton_strided_inputs(causal):
    """Aligned noncompact outer strides and nonunit weight strides are supported."""
    q, k, weights = make_inputs(130, 6, 48, torch.bfloat16)
    q, k, weights = q[:, ::2, ::2], k[:, ::2], weights[:, ::2, ::2]
    assert all(not tensor.is_contiguous() for tensor in (q, k, weights))
    actual = lightning_indexer(
        q, k, weights, 37, causal=causal, impl="fused", kernel_options={"backend": "triton"}
    )
    assert_selection(actual, q, k, weights, 37, causal)


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "dim_unaligned",
        "dim_large",
        "heads",
        "topk",
        "q_layout",
        "k_layout",
        "outer_stride",
    ],
)
def test_triton_unsupported_inputs(case):
    """Reject unsupported arithmetic and TMA layouts at the public boundary."""
    dim = {"dim_unaligned": 18, "dim_large": 264, "outer_stride": 17}.get(case, 16)
    q, k, weights = make_inputs(
        513 if case == "topk" else 65, 257 if case == "heads" else 3, dim, torch.bfloat16
    )
    topk = 513 if case == "topk" else 1
    match case:
        case "dtype":
            q, k, weights = q.float(), k.float(), weights.float()
        case "q_layout":
            q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
        case "k_layout":
            k = k.transpose(-1, -2).contiguous().transpose(-1, -2)
        case "outer_stride":
            q, k = q[..., :16], k[..., :16]
    error = TypeError if case == "dtype" else ValueError
    message = (
        "FP16 or BF16"
        if case == "dtype"
        else "contiguous last dimension|16-byte outer strides"
        if case in ("q_layout", "k_layout", "outer_stride")
        else "H <= 256"
    )
    with pytest.raises(error, match=message):
        lightning_indexer(q, k, weights, topk, impl="fused", kernel_options={"backend": "triton"})


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("topk", [0, 37])
def test_triton_opcheck(dtype, requires_grad, topk):
    """Validate schema, autograd, fake tensors and dynamic AOT dispatch."""
    from attn_gym.sparse.indexer.ops import _indexer_op

    inputs = make_inputs(65, 3, 48, dtype)
    for tensor in inputs:
        tensor.requires_grad_(requires_grad)
    torch.library.opcheck(_indexer_op, (*inputs, topk, True, "triton"))
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
        inputs = make_inputs(tokens, 3, 48, torch.bfloat16)
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

    inputs = make_inputs(65, 3, 48, torch.bfloat16)
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
    inputs = list(make_inputs(65, 3, 48, torch.bfloat16))
    tensor = inputs[operand]
    storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
    inputs[operand] = storage[1:].view(tensor.shape).copy_(tensor)
    with pytest.raises(ValueError, match="16-byte-aligned Q/K"):
        lightning_indexer(*inputs, 37, impl="fused", kernel_options={"backend": "triton"})


@pytest.mark.parametrize("causal", [False, True])
def test_triton_cuda_graph(causal):
    """Replay captured selection after changing static inputs, not just once."""
    inputs = make_inputs(129, 3, 48, torch.float16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            lightning_indexer(
                *inputs, 37, causal=causal, impl="fused", kernel_options={"backend": "triton"}
            )
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = lightning_indexer(
            *inputs, 37, causal=causal, impl="fused", kernel_options={"backend": "triton"}
        )
    for _ in range(2):
        for tensor in inputs:
            tensor.normal_()
        graph.replay()
        expected = lightning_indexer(
            *inputs, 37, causal=causal, impl="fused", kernel_options={"backend": "triton"}
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
