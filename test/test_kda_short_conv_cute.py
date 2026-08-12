"""Correctness and integration tests for the CuTeDSL KDA short convolution."""

from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

from attn_gym.linear.kda.short_conv.cute import (
    ShortConvConfig,
    _backward_custom_op,
    _candidate_configs,
    _configured_backward_custom_op,
    _configured_forward_custom_op,
    _forward_custom_op,
    cute_causal_conv1d_silu,
    tune_causal_conv1d_silu,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL short convolution requires CUDA capability 10.0 or newer",
)


def _inputs(tokens: int = 17, channels: int = 12, width: int = 4, batch: int = 1):
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    return x.requires_grad_(), weight.requires_grad_()


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    padded = F.pad(x.transpose(1, 2), (weight.shape[1] - 1, 0))
    return F.silu(F.conv1d(padded, weight[:, None], groups=x.shape[-1])).transpose(1, 2)


def _packed_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Concatenate independent convolutions for every nonempty packed sequence."""
    offsets = cu_seqlens.cpu().tolist()
    return torch.cat(
        [_reference(x[:, start:end], weight) for start, end in pairwise(offsets) if start != end],
        dim=1,
    )


@pytest.mark.parametrize("width", [1, 3, 4, 5])
def test_short_conv_forward_and_backward_match_pytorch(width: int):
    """Check generic widths and partial first and last time tiles."""
    torch.manual_seed(0)
    x, weight = _inputs(width=width)
    grad_output = torch.randn_like(x)

    actual = cute_causal_conv1d_silu(x, weight)
    expected = _reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=2e-1)


@pytest.mark.parametrize("channels", [5, 6])
def test_short_conv_defaults_support_any_positive_channel_count(channels: int):
    """Select a compatible packed channel width without requiring an explicit config."""
    torch.manual_seed(1)
    x, weight = _inputs(tokens=19, channels=channels)
    grad_output = torch.randn_like(x)

    actual = cute_causal_conv1d_silu(x, weight)
    expected = _reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=3e-1)

    compiled = torch.compile(cute_causal_conv1d_silu, fullgraph=True)
    torch.testing.assert_close(compiled(x, weight), expected, rtol=2e-2, atol=2e-2)

    for kind in ("forward", "input_gradient", "weight_gradient"):
        candidates = _candidate_configs(kind, channels)
        assert candidates
        assert all(channels % config.channels_per_thread == 0 for config in candidates)


@pytest.mark.parametrize("width", [3, 4])
def test_short_conv_batched_forward_and_backward_match_pytorch(width: int):
    """Keep batches independent across generic and optimized convolution widths."""
    torch.manual_seed(1)
    x, weight = _inputs(tokens=19, channels=12, width=width, batch=3)
    grad_output = torch.randn_like(x)
    weight_config = ShortConvConfig(128, 4, 8)

    actual = cute_causal_conv1d_silu(x, weight, weight_grad_config=weight_config)
    expected = _reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=3e-1)


@pytest.mark.parametrize("width", [1, 3, 4, 5])
def test_short_conv_packed_forward_and_backward_match_independent_sequences(width: int):
    """Reset convolution and gradient dependencies at every packed boundary."""
    torch.manual_seed(2)
    x, weight = _inputs(tokens=67, channels=12, width=width)
    cu_seqlens = torch.tensor(
        [0, 1, 3, 8, 8, 17, 31, 67],
        device="cuda",
        dtype=torch.int32,
    )
    grad_output = torch.randn_like(x)

    actual = cute_causal_conv1d_silu(x, weight, cu_seqlens=cu_seqlens)
    expected = _packed_reference(x, weight, cu_seqlens)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=3e-1)


def test_short_conv_accepts_misaligned_contiguous_storage():
    """Materialize alignment inside the opaque launcher when a view starts off-boundary."""
    tokens, channels = 17, 12
    x_storage = torch.randn(1 * tokens * channels + 1, device="cuda", dtype=torch.bfloat16)
    weight_storage = torch.randn(channels * 4 + 1, device="cuda", dtype=torch.bfloat16)
    x = x_storage[1:].view(1, tokens, channels).requires_grad_()
    weight = weight_storage[1:].view(channels, 4).requires_grad_()
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    assert weight.is_contiguous() and weight.data_ptr() % 16 != 0

    actual = cute_causal_conv1d_silu(x, weight)
    expected = _reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_short_conv_explicit_config_and_tuning_flow():
    """Route explicit schedules through the shared compile-and-benchmark tuner."""
    x, weight = _inputs(channels=6, width=3)
    grad_output = torch.randn_like(x)
    forward = ShortConvConfig(128, 2, 8)
    input_gradient = ShortConvConfig(128, 2, 10)
    weight_gradient = ShortConvConfig(128, 2, 128)
    selected = tune_causal_conv1d_silu(
        x,
        weight,
        grad_output,
        forward_configs=(forward,),
        input_grad_configs=(input_gradient,),
        weight_grad_configs=(weight_gradient,),
        parallel_compile=False,
    )
    assert selected.forward == forward
    assert selected.input_gradient == input_gradient
    assert selected.weight_gradient == weight_gradient
    actual = cute_causal_conv1d_silu(
        x,
        weight,
        forward_config=selected.forward,
        input_grad_config=selected.input_gradient,
        weight_grad_config=selected.weight_gradient,
    )
    expected = _reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    compiled = torch.compile(
        lambda x, weight: cute_causal_conv1d_silu(
            x,
            weight,
            forward_config=selected.forward,
            input_grad_config=selected.input_gradient,
            weight_grad_config=selected.weight_gradient,
        ),
        fullgraph=True,
    )
    torch.testing.assert_close(compiled(x, weight), expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("packed", [False, True])
def test_short_conv_custom_op_registration(packed: bool):
    """Exercise dense and packed schemas, fake implementations, and autograd."""
    x, weight = _inputs(batch=1 if packed else 2)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 2, 7, 17], device="cuda", dtype=torch.int32) if packed else None
    torch.library.opcheck(_forward_custom_op, (x, weight, cu_seqlens))
    torch.library.opcheck(
        _backward_custom_op,
        (x, weight, grad_output, cu_seqlens),
        test_utils=("test_schema", "test_faketensor"),
    )
    if packed:
        configs = (128, 4, 8, 128, 4, 10, 128, 4, 128)
        torch.library.opcheck(
            _configured_forward_custom_op,
            (x, weight, cu_seqlens, *configs),
        )
        torch.library.opcheck(
            _configured_backward_custom_op,
            (x, weight, grad_output, cu_seqlens, *configs[3:]),
            test_utils=("test_schema", "test_faketensor"),
        )


def test_short_conv_fullgraph_forward_and_backward():
    """Keep batched opaque operators inside a strict compiled graph."""
    x, weight = _inputs(batch=2)
    grad_output = torch.randn_like(x)

    expected_output = cute_causal_conv1d_silu(x, weight)
    expected = torch.autograd.grad(expected_output, (x, weight), grad_output)
    compiled = torch.compile(cute_causal_conv1d_silu, fullgraph=True)
    actual_output = compiled(x, weight)
    actual = torch.autograd.grad(actual_output, (x, weight), grad_output)
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=3e-2, atol=2e-1)


def test_short_conv_packed_fullgraph_forward_and_backward():
    """Keep the configured packed operation inside a strict compiled graph."""
    x, weight = _inputs(tokens=31)
    cu_seqlens = torch.tensor([0, 2, 9, 17, 31], device="cuda", dtype=torch.int32)
    grad_output = torch.randn_like(x)
    configs = {
        "forward_config": ShortConvConfig(128, 4, 8),
        "input_grad_config": ShortConvConfig(128, 4, 10),
        "weight_grad_config": ShortConvConfig(128, 4, 128),
    }

    expected_output = cute_causal_conv1d_silu(x, weight, cu_seqlens=cu_seqlens, **configs)
    expected = torch.autograd.grad(expected_output, (x, weight), grad_output)
    compiled = torch.compile(cute_causal_conv1d_silu, fullgraph=True)
    actual_output = compiled(x, weight, cu_seqlens=cu_seqlens, **configs)
    actual = torch.autograd.grad(actual_output, (x, weight), grad_output)
    torch.testing.assert_close(actual[0], expected[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=3e-2, atol=2e-1)


def test_short_conv_packed_offsets_obey_autograd_versioning():
    """Reject sequence-boundary mutation between forward and backward."""
    x, weight = _inputs(tokens=31)
    cu_seqlens = torch.tensor([0, 3, 11, 31], device="cuda", dtype=torch.int32)
    output = cute_causal_conv1d_silu(x, weight, cu_seqlens=cu_seqlens)

    with torch.no_grad():
        cu_seqlens[1] = 4
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output, (x, weight), torch.randn_like(output))


def test_short_conv_cuda_graph_replay():
    """Capture the compiled launchers and replay with changed input values."""
    x, weight = _inputs()
    grad_output = torch.randn_like(x)
    _forward_custom_op(x, weight)
    _backward_custom_op(x, weight, grad_output)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _forward_custom_op(x, weight)
        captured_gradients = _backward_custom_op(x, weight, grad_output)

    graph.replay()
    first_output = captured_output.clone()
    first_gradients = tuple(gradient.clone() for gradient in captured_gradients)
    with torch.no_grad():
        x.add_(0.25)
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(captured_output, first_output)
    assert not torch.equal(captured_gradients[0], first_gradients[0])
    expected_output = _forward_custom_op(x, weight)
    expected_gradients = _backward_custom_op(x, weight, grad_output)
    torch.testing.assert_close(captured_output, expected_output)
    torch.testing.assert_close(captured_gradients[0], expected_gradients[0])
    torch.testing.assert_close(captured_gradients[1], expected_gradients[1])


def test_short_conv_packed_cuda_graph_replays_changed_boundaries():
    """Read packed boundaries from device memory again on every graph replay."""
    x, weight = _inputs(tokens=31)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 0, 11, 31, 31], device="cuda", dtype=torch.int32)
    _forward_custom_op(x, weight, cu_seqlens)
    _backward_custom_op(x, weight, grad_output, cu_seqlens)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _forward_custom_op(x, weight, cu_seqlens)
        captured_gradients = _backward_custom_op(x, weight, grad_output, cu_seqlens)

    graph.replay()
    first_output = captured_output.clone()
    with torch.no_grad():
        cu_seqlens.copy_(torch.tensor([0, 0, 8, 31, 31], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(captured_output, first_output)
    expected_output = _forward_custom_op(x, weight, cu_seqlens)
    expected_gradients = _backward_custom_op(x, weight, grad_output, cu_seqlens)
    torch.testing.assert_close(captured_output, expected_output)
    torch.testing.assert_close(captured_gradients[0], expected_gradients[0])
    torch.testing.assert_close(captured_gradients[1], expected_gradients[1])
