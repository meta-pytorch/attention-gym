"""Correctness and integration tests for the CuTeDSL KDA short convolution."""

from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

import attn_gym.linear.kda.short_conv.cute as cute_backend
from attn_gym.linear.kda.short_conv.cute import (
    ShortConvConfig,
    ShortConvTunedConfig,
    _backward_custom_op,
    _candidate_configs,
    _configured_backward_custom_op,
    _configured_forward_custom_op,
    _forward_custom_op,
    cute_causal_conv1d_silu,
    tune_causal_conv1d_silu,
)
from examples.kda_training import KDAAttention

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL short convolution requires CUDA capability 10.0 or newer",
)


def _inputs(
    tokens: int = 17,
    channels: int = 12,
    width: int = 4,
    batch: int = 1,
    dtype: torch.dtype = torch.bfloat16,
):
    x = torch.randn(batch, tokens, channels, device="cuda", dtype=dtype)
    weight = torch.randn(channels, width, device="cuda", dtype=dtype)
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


def _packed_state_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply stateful convolution independently to each packed sequence."""
    width = weight.shape[1]
    outputs = []
    final_states = []
    for sequence, (start, end) in enumerate(pairwise(cu_seqlens.cpu().tolist())):
        extended = torch.cat((initial_state[sequence : sequence + 1], x[:, start:end]), dim=1)
        if start != end:
            output = F.conv1d(
                extended.transpose(1, 2),
                weight[:, None],
                groups=x.shape[-1],
            ).transpose(1, 2)
            outputs.append(F.silu(output))
        final_states.append(extended[:, -(width - 1) :] if width > 1 else extended[:, :0])
    return torch.cat(outputs, dim=1), torch.cat(final_states, dim=0)


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


def test_short_conv_dtype_defaults_follow_measured_storage_traffic():
    """Use the measured dtype- and layout-specific gradient schedules."""
    fp16 = ShortConvTunedConfig.default(torch.float16)
    bf16 = ShortConvTunedConfig.default()
    fp32 = ShortConvTunedConfig.default(torch.float32)

    assert fp16.forward == ShortConvConfig(128, 4, 16)
    assert fp16.input_gradient == ShortConvConfig(128, 2, 32)
    assert fp16.weight_gradient == ShortConvConfig(128, 2, 64)
    assert bf16.input_gradient == ShortConvConfig(128, 2, 32)
    assert bf16.weight_gradient == ShortConvConfig(128, 2, 64)
    assert ShortConvTunedConfig.default(stateful=True) == bf16
    assert ShortConvTunedConfig.default(torch.float16, stateful=True) == fp16
    packed_fp16 = ShortConvTunedConfig.default(torch.float16, packed=True)
    packed = ShortConvTunedConfig.default(packed=True)
    packed_fp32 = ShortConvTunedConfig.default(torch.float32, packed=True)
    assert packed_fp16.input_gradient == ShortConvConfig(128, 4, 16)
    assert packed_fp16.weight_gradient == ShortConvConfig(128, 2, 32)
    assert packed.input_gradient == ShortConvConfig(128, 4, 16)
    assert packed.weight_gradient == ShortConvConfig(128, 2, 32)
    assert packed_fp32.input_gradient == ShortConvConfig(128, 2, 16)
    assert packed_fp32.weight_gradient == ShortConvConfig(128, 4, 32)
    packed_stateful = ShortConvTunedConfig.default(packed=True, stateful=True)
    assert packed_stateful.input_gradient == packed.input_gradient
    assert packed_stateful.weight_gradient == packed.weight_gradient
    assert ShortConvTunedConfig.default(
        torch.float16, packed=True, stateful=True
    ) == ShortConvTunedConfig.default(torch.float16, packed=True)
    assert ShortConvTunedConfig.default(
        torch.float32, packed=True, stateful=True
    ) == ShortConvTunedConfig.default(torch.float32, packed=True)
    assert fp32.forward == ShortConvConfig(128, 4, 4)
    assert fp32.input_gradient == ShortConvConfig(128, 2, 12)
    assert fp32.weight_gradient == ShortConvConfig(128, 2, 160)
    assert ShortConvTunedConfig.default(torch.float32, stateful=True) == fp32

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        defaults = ShortConvTunedConfig.default(dtype, packed=True, stateful=True)
        assert defaults.input_gradient in _candidate_configs(
            "input_gradient", 512, dtype, packed=True, stateful=True
        )
        assert defaults.weight_gradient in _candidate_configs(
            "weight_gradient", 512, dtype, packed=True, stateful=True
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_short_conv_packed_defaults_select_tma(dtype: torch.dtype):
    """Keep the production packed defaults on the staged implementation."""
    defaults = ShortConvTunedConfig.default(dtype, packed=True, stateful=True)
    descriptor = cute_backend.SHORT_CONV_DTYPES[dtype]
    assert cute_backend.supports_tma(
        cute_backend.CausalConv1dSiluInputGradientTma,
        defaults.input_gradient,
        cute_backend.tuned_config(descriptor, packed=True).input_gradient,
        384,
        512,
        4,
        7,
        packed_supported=True,
    )
    assert cute_backend.supports_tma(
        cute_backend.CausalConv1dSiluWeightGradientPartialsTma,
        defaults.weight_gradient,
        cute_backend.tuned_config(descriptor, packed=True).weight_gradient,
        384,
        512,
        4,
        7,
        packed_supported=True,
    )


@pytest.mark.parametrize(
    ("dtype", "rounding_allowance", "rtol", "atol"),
    [
        (torch.float16, 1e-3, 3e-3, 4e-3),
        (torch.bfloat16, 8e-3, 3e-2, 3e-2),
        (torch.float32, 5e-6, 2e-5, 2e-5),
    ],
)
def test_short_conv_supported_dtypes_match_high_precision_reference(
    dtype: torch.dtype,
    rounding_allowance: float,
    rtol: float,
    atol: float,
):
    """Specialize storage while retaining FP32 convolution accumulation."""
    torch.manual_seed(5)
    x, weight = _inputs(tokens=17, channels=12, width=4, batch=2, dtype=dtype)
    initial_state = torch.randn(
        2,
        3,
        12,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    grad_output = torch.randn_like(x)

    actual = cute_causal_conv1d_silu(x, weight, initial_state=initial_state)
    reference_input = torch.cat((initial_state, x), dim=1)
    expected = F.silu(
        F.conv1d(reference_input.transpose(1, 2), weight[:, None], groups=12)
    ).transpose(1, 2)

    high_x = x.detach().double().requires_grad_()
    high_weight = weight.detach().double().requires_grad_()
    high_state = initial_state.detach().double().requires_grad_()
    high_input = torch.cat((high_state, high_x), dim=1)
    high_precision = F.silu(
        F.conv1d(high_input.transpose(1, 2), high_weight[:, None], groups=12)
    ).transpose(1, 2)

    actual_gradients = torch.autograd.grad(
        actual,
        (x, weight, initial_state),
        grad_output,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (x, weight, initial_state),
        grad_output,
    )
    high_gradients = torch.autograd.grad(
        high_precision,
        (high_x, high_weight, high_state),
        grad_output.double(),
    )

    def relative_l2(value: torch.Tensor, reference: torch.Tensor) -> float:
        return ((value.double() - reference).norm() / reference.norm()).item()

    actual_error = relative_l2(actual, high_precision)
    expected_error = relative_l2(expected, high_precision)
    assert actual_error <= expected_error + rounding_allowance
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for actual_gradient, expected_gradient, high_gradient in zip(
        actual_gradients,
        expected_gradients,
        high_gradients,
        strict=True,
    ):
        actual_error = relative_l2(actual_gradient, high_gradient)
        expected_error = relative_l2(expected_gradient, high_gradient)
        assert actual_error <= expected_error + rounding_allowance
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=rtol, atol=atol)


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [3, 4, 5])
def test_short_conv_tma_backward_matches_batched_reference(
    width: int,
    dtype: torch.dtype,
):
    """Exercise aligned dense batches through the selected TMA schedules."""
    torch.manual_seed(7)
    x, weight = _inputs(tokens=384, channels=256, width=width, batch=2, dtype=dtype)
    grad_output = torch.randn_like(x)

    actual = cute_causal_conv1d_silu(x, weight)
    expected = _reference(x, weight)
    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    match dtype:
        case torch.float16 | torch.bfloat16:
            fallback_input = ShortConvConfig(128, 1, 28)
            fallback_weight = ShortConvConfig(128, 4, 128)
        case torch.float32:
            fallback_input = ShortConvConfig(128, 2, 12)
            fallback_weight = ShortConvConfig(128, 4, 32)
        case _:
            raise AssertionError(f"unexpected dtype {dtype}")
    fallback_gradients = cute_backend._launch_backward(
        x,
        weight,
        grad_output,
        fallback_input,
        fallback_weight,
    )

    tolerance = 1e-4 if dtype == torch.float32 else 3e-2
    weight_atol = 2e-4 if dtype == torch.float32 else 2e-1
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        actual_gradients[0],
        expected_gradients[0],
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        actual_gradients[1],
        expected_gradients[1],
        rtol=tolerance,
        atol=weight_atol,
    )
    # dx keeps the same per-token reduction tree. dw changes partial boundaries and therefore
    # follows the dtype tolerance rather than a bitwise contract.
    assert torch.equal(actual_gradients[0], fallback_gradients[0])
    torch.testing.assert_close(
        actual_gradients[1],
        fallback_gradients[1],
        rtol=tolerance,
        atol=weight_atol,
    )


def test_short_conv_tma_input_gradient_wider_than_time_tile():
    """Keep each CTA's tail writes inside its owned tile when width exceeds that tile."""
    torch.manual_seed(11)
    x, weight = _inputs(tokens=64, channels=256, width=40, batch=2)
    grad_output = torch.randn_like(x)
    expected = _reference(x, weight)
    (expected_dx,) = torch.autograd.grad(expected, (x,), grad_output)

    defaults = ShortConvTunedConfig.default()
    actual_dx = cute_backend._launch_backward(
        x,
        weight,
        grad_output,
        defaults.input_gradient,
        ShortConvConfig(128, 4, 128),
    )[0]

    torch.testing.assert_close(actual_dx, expected_dx, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_short_conv_packed_tma_backward_matches_fallback(dtype: torch.dtype):
    """Stage physical tiles while resetting both gradients at arbitrary boundaries."""
    torch.manual_seed(8)
    x, weight = _inputs(tokens=384, channels=512, width=4, dtype=dtype)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor(
        [0, 0, 3, 17, 18, 129, 257, 384],
        device="cuda",
        dtype=torch.int32,
    )

    actual = cute_causal_conv1d_silu(x, weight, cu_seqlens=cu_seqlens)
    expected = _packed_reference(x, weight, cu_seqlens)
    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    fallback_gradients = cute_backend._launch_backward(
        x,
        weight,
        grad_output,
        ShortConvConfig(128, 4, 10),
        ShortConvConfig(64, 4, 128),
        cu_seqlens,
    )

    tolerance = 1e-4 if dtype == torch.float32 else 3e-2
    weight_atol = 2e-4 if dtype == torch.float32 else 2e-1
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(
        actual_gradients[0], expected_gradients[0], rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        actual_gradients[1], expected_gradients[1], rtol=tolerance, atol=weight_atol
    )
    torch.testing.assert_close(
        actual_gradients[0], fallback_gradients[0], rtol=tolerance, atol=tolerance
    )
    torch.testing.assert_close(
        actual_gradients[1], fallback_gradients[1], rtol=tolerance, atol=weight_atol
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_short_conv_dense_stateful_tma_backward_matches_reference(dtype: torch.dtype):
    """Use caller history in aligned dense TMA input- and weight-gradient tiles."""
    torch.manual_seed(12)
    x, weight = _inputs(tokens=384, channels=256, width=4, batch=2, dtype=dtype)
    initial_state = torch.randn(2, 3, 256, device="cuda", dtype=dtype, requires_grad=True)
    reference_state = initial_state.detach().clone().requires_grad_()
    grad_output = torch.randn_like(x)

    actual = cute_causal_conv1d_silu(x, weight, initial_state=initial_state)
    expected_input = torch.cat((reference_state, x), dim=1)
    expected = F.silu(
        F.conv1d(expected_input.transpose(1, 2), weight[:, None], groups=256)
    ).transpose(1, 2)
    actual_gradients = torch.autograd.grad(actual, (x, weight, initial_state), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight, reference_state), grad_output)

    tolerance = 1e-4 if dtype == torch.float32 else 3e-2
    weight_atol = 2e-4 if dtype == torch.float32 else 2e-1
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for index, (actual_gradient, expected_gradient) in enumerate(
        zip(actual_gradients, expected_gradients, strict=True)
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=tolerance,
            atol=weight_atol if index == 1 else tolerance,
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_short_conv_packed_stateful_tma_backward_matches_reference_and_fallback(
    dtype: torch.dtype,
):
    """Seed every packed TMA convolution window from caller-owned sequence history."""
    torch.manual_seed(9)
    x, weight = _inputs(tokens=384, channels=512, width=4, dtype=dtype)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor(
        [0, 0, 3, 17, 18, 129, 257, 384],
        device="cuda",
        dtype=torch.int32,
    )
    initial_state = torch.randn(
        cu_seqlens.shape[0] - 1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    reference_state = initial_state.detach().clone().requires_grad_()

    actual = cute_causal_conv1d_silu(
        x,
        weight,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
    )
    expected, _ = _packed_state_reference(x, weight, cu_seqlens, reference_state)
    actual_gradients = torch.autograd.grad(
        actual,
        (x, weight, initial_state),
        grad_output,
    )
    expected_gradients = torch.autograd.grad(
        expected,
        (x, weight, reference_state),
        grad_output,
    )
    fallback_gradients = cute_backend._launch_backward(
        x,
        weight,
        grad_output,
        ShortConvConfig(128, 4, 10),
        ShortConvConfig(64, 4, 128),
        cu_seqlens,
        initial_state,
    )

    tolerance = 1e-4 if dtype == torch.float32 else 3e-2
    weight_atol = 2e-4 if dtype == torch.float32 else 2e-1
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    for index, (actual_gradient, expected_gradient, fallback_gradient) in enumerate(
        zip(actual_gradients, expected_gradients, fallback_gradients, strict=True)
    ):
        atol = weight_atol if index == 1 else tolerance
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=tolerance,
            atol=atol,
        )
        torch.testing.assert_close(
            actual_gradient,
            fallback_gradient,
            rtol=tolerance,
            atol=atol,
        )


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


def test_short_conv_packed_final_state_uses_implicit_zero_history():
    """Return packed history for empty and short sequences without an initial state."""
    x, weight = _inputs(tokens=7, channels=12, width=5)
    cu_seqlens = torch.tensor([0, 0, 1, 3, 7], device="cuda", dtype=torch.int32)
    actual, actual_final = cute_causal_conv1d_silu(
        x,
        weight,
        cu_seqlens=cu_seqlens,
        return_final_state=True,
    )
    zero_state = x.new_zeros(cu_seqlens.shape[0] - 1, weight.shape[1] - 1, x.shape[2])
    expected, expected_final = _packed_state_reference(x, weight, cu_seqlens, zero_state)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_final, expected_final)


@pytest.mark.parametrize("width", [1, 3, 5])
def test_short_conv_packed_state_forward_and_backward(width: int):
    """Compose packed boundaries with differentiable per-sequence history."""
    torch.manual_seed(4)
    x, weight = _inputs(tokens=17, channels=12, width=width)
    cu_seqlens = torch.tensor([0, 0, 1, 4, 4, 17], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        cu_seqlens.shape[0] - 1,
        width - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    expected_state = initial_state.detach().clone().requires_grad_()

    actual, actual_final = cute_causal_conv1d_silu(
        x,
        weight,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        return_final_state=True,
    )
    expected, expected_final = _packed_state_reference(
        x,
        weight,
        cu_seqlens,
        expected_state,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_final, expected_final)

    grad_output = torch.randn_like(actual)
    grad_final = torch.randn_like(actual_final)
    actual_gradients = torch.autograd.grad(
        (actual, actual_final),
        (x, weight, initial_state),
        (grad_output, grad_final),
    )
    expected_gradients = torch.autograd.grad(
        (expected, expected_final),
        (x, weight, expected_state),
        (grad_output, grad_final),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=3e-1)


@pytest.mark.parametrize(
    ("width", "tokens", "with_initial_state"),
    [(1, 2, True), (3, 5, True), (5, 2, True), (5, 2, False)],
)
def test_kda_example_fused_short_conv_supports_generic_width_and_state(
    width: int,
    tokens: int,
    with_initial_state: bool,
):
    """Keep the example on CuTeDSL when adapting optional convolution history."""
    torch.manual_seed(3)
    model = KDAAttention(
        hidden_size=128,
        num_heads=1,
        head_dim=128,
        short_conv_kernel_size=width,
        backend="fused",
        device="cuda",
    ).to(torch.bfloat16)
    channels = model.qkv_conv1d.in_channels
    qkv = torch.randn(2, tokens, channels, device="cuda", dtype=torch.bfloat16)
    qkv_actual = qkv.clone().requires_grad_()
    qkv_expected = qkv.clone().requires_grad_()
    state = torch.randn(
        2,
        width - 1,
        channels,
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_actual = state.clone().requires_grad_() if with_initial_state else None
    state_expected = state.clone().requires_grad_() if with_initial_state else None
    weight_actual = model.qkv_conv1d.weight
    weight_expected = weight_actual.detach().clone().requires_grad_()

    actual, actual_final = model.short_convolution(
        qkv_actual,
        state_actual,
        return_final_state=True,
    )
    reference_state = (
        qkv_expected.new_zeros(2, width - 1, channels)
        if state_expected is None
        else state_expected
    )
    reference_input = torch.cat((reference_state, qkv_expected), dim=1)
    expected = F.silu(
        F.conv1d(
            reference_input.transpose(1, 2),
            weight_expected,
            groups=channels,
        ).transpose(1, 2)
    )
    expected_final = reference_input[:, tokens:].clone()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_final, expected_final)

    grad_output = torch.randn_like(actual)
    grad_final = torch.randn_like(actual_final)
    actual_inputs = (qkv_actual, weight_actual)
    expected_inputs = (qkv_expected, weight_expected)
    if with_initial_state:
        actual_inputs += (state_actual,)
        expected_inputs += (state_expected,)
    actual_gradients = torch.autograd.grad(
        (actual, actual_final),
        actual_inputs,
        (grad_output, grad_final),
    )
    expected_gradients = torch.autograd.grad(
        (expected, expected_final),
        expected_inputs,
        (grad_output, grad_final),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=3e-1)


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


@pytest.mark.parametrize("width", [1, 3])
def test_short_conv_explicit_config_and_tuning_flow(width: int):
    """Route stateful schedules through the shared compile-and-benchmark tuner."""
    x, weight = _inputs(channels=6, width=width)
    grad_output = torch.randn_like(x)
    initial_state = torch.randn(
        1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    forward = ShortConvConfig(128, 2, 8)
    input_gradient = ShortConvConfig(128, 2, 10)
    weight_gradient = ShortConvConfig(128, 2, 128)
    selected = tune_causal_conv1d_silu(
        x,
        weight,
        grad_output,
        initial_state=initial_state,
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
        initial_state=initial_state,
        forward_config=selected.forward,
        input_grad_config=selected.input_gradient,
        weight_grad_config=selected.weight_gradient,
    )
    extended = torch.cat((initial_state, x), dim=1)
    expected = F.silu(
        F.conv1d(extended.transpose(1, 2), weight[:, None], groups=x.shape[-1])
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    compiled = torch.compile(
        lambda x, weight, initial_state: cute_causal_conv1d_silu(
            x,
            weight,
            initial_state=initial_state,
            forward_config=selected.forward,
            input_grad_config=selected.input_gradient,
            weight_grad_config=selected.weight_gradient,
        ),
        fullgraph=True,
    )
    compiled_output = compiled(x, weight, initial_state)
    torch.testing.assert_close(compiled_output, actual, rtol=0, atol=0)

    actual_gradients = torch.autograd.grad(actual, (x, weight, initial_state), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight, initial_state), grad_output)
    compiled_gradients = torch.autograd.grad(
        compiled_output,
        (x, weight, initial_state),
        grad_output,
    )
    for gradients in (actual_gradients, compiled_gradients):
        for gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(gradient, expected_gradient, rtol=3e-2, atol=3e-1)


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
            (x, weight, cu_seqlens, None, *configs),
        )
        torch.library.opcheck(
            _configured_backward_custom_op,
            (x, weight, grad_output, cu_seqlens, None, False, *configs[3:]),
            test_utils=("test_schema", "test_faketensor"),
        )


@pytest.mark.parametrize("packed", [False, True])
def test_short_conv_stateful_custom_op_registration(packed: bool):
    """Exercise schemas, fake tensors, and autograd for caller-provided history."""
    x, weight = _inputs(batch=1 if packed else 2)
    cu_seqlens = torch.tensor([0, 2, 7, 17], device="cuda", dtype=torch.int32) if packed else None
    num_sequences = cu_seqlens.shape[0] - 1 if packed else x.shape[0]
    initial_state = torch.randn(
        num_sequences,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn_like(x)
    torch.library.opcheck(_forward_custom_op, (x, weight, cu_seqlens, initial_state))
    config = (128, 4, 10, 128, 4, 128)
    torch.library.opcheck(
        _configured_backward_custom_op,
        (x, weight, grad_output, cu_seqlens, initial_state, True, *config),
        test_utils=("test_schema", "test_faketensor"),
    )


def test_short_conv_constant_history_skips_its_gradient_kernel(monkeypatch):
    """Avoid launching the history-gradient kernel for nondifferentiable state."""
    x, weight = _inputs()
    initial_state = torch.randn(
        x.shape[0],
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
    )

    def fail_compile(*args, **kwargs):
        raise AssertionError("initial-state-gradient kernel should not be compiled")

    monkeypatch.setattr(cute_backend, "_compile_initial_state_gradient", fail_compile)
    output = cute_causal_conv1d_silu(x, weight, initial_state=initial_state)
    torch.autograd.grad(output, (x, weight), torch.randn_like(output))


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_short_conv_stateful_fullgraph_forward_and_backward(dtype: torch.dtype):
    """Compile every storage dtype's initial- and final-state contract as one graph."""
    x, weight = _inputs(tokens=2, width=5, batch=2, dtype=dtype)
    initial_state = torch.randn(
        2,
        4,
        x.shape[2],
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    grad_output = torch.randn_like(x)
    grad_final = torch.randn_like(initial_state)

    def operation(x, weight, initial_state):
        return cute_causal_conv1d_silu(
            x,
            weight,
            initial_state=initial_state,
            return_final_state=True,
        )

    expected_output, expected_final = operation(x, weight, initial_state)
    expected = torch.autograd.grad(
        (expected_output, expected_final),
        (x, weight, initial_state),
        (grad_output, grad_final),
    )
    actual_output, actual_final = torch.compile(operation, fullgraph=True)(
        x, weight, initial_state
    )
    actual = torch.autograd.grad(
        (actual_output, actual_final),
        (x, weight, initial_state),
        (grad_output, grad_final),
    )
    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_final, expected_final)
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


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


def test_short_conv_packed_stateful_fullgraph_forward_and_backward():
    """Compile packed initial and final state through the public operation."""
    x, weight = _inputs(tokens=17, width=5)
    cu_seqlens = torch.tensor([0, 0, 2, 7, 17], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        cu_seqlens.shape[0] - 1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn_like(x)
    grad_final = torch.randn_like(initial_state)

    def operation(x, weight, initial_state, cu_seqlens):
        return cute_causal_conv1d_silu(
            x,
            weight,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            return_final_state=True,
        )

    expected_output, expected_final = operation(x, weight, initial_state, cu_seqlens)
    expected = torch.autograd.grad(
        (expected_output, expected_final),
        (x, weight, initial_state),
        (grad_output, grad_final),
    )
    actual_output, actual_final = torch.compile(operation, fullgraph=True)(
        x,
        weight,
        initial_state,
        cu_seqlens,
    )
    actual = torch.autograd.grad(
        (actual_output, actual_final),
        (x, weight, initial_state),
        (grad_output, grad_final),
    )
    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_final, expected_final)
    for actual_gradient, expected_gradient in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_short_conv_width_one_preserves_empty_state_gradients():
    """Return shaped zero gradients for explicit empty convolution state."""
    x, weight = _inputs(tokens=7, width=1, batch=2)
    initial_state = torch.empty(
        2,
        0,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = cute_causal_conv1d_silu(x, weight, initial_state=initial_state)
    (grad_initial_state,) = torch.autograd.grad(output.sum(), (initial_state,))
    assert grad_initial_state.shape == initial_state.shape

    packed_x, packed_weight = _inputs(tokens=7, width=1)
    cu_seqlens = torch.tensor([0, 0, 2, 7], device="cuda", dtype=torch.int32)
    _, final_state = cute_causal_conv1d_silu(
        packed_x,
        packed_weight,
        cu_seqlens=cu_seqlens,
        return_final_state=True,
    )
    (grad_x,) = torch.autograd.grad(final_state, (packed_x,), torch.empty_like(final_state))
    torch.testing.assert_close(grad_x, torch.zeros_like(packed_x), rtol=0, atol=0)


def test_short_conv_final_state_preserves_unused_initial_state_gradient():
    """Return a shaped zero state gradient when new input fully replaces history."""
    x, weight = _inputs(tokens=7, width=3)
    initial_state = torch.randn(
        1,
        2,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    _, final_state = cute_causal_conv1d_silu(
        x,
        weight,
        initial_state=initial_state,
        return_final_state=True,
    )
    (grad_initial_state,) = torch.autograd.grad(
        final_state,
        (initial_state,),
        torch.randn_like(final_state),
    )
    torch.testing.assert_close(
        grad_initial_state,
        torch.zeros_like(initial_state),
        rtol=0,
        atol=0,
    )


def test_short_conv_initial_state_obeys_autograd_versioning():
    """Reject dense-history mutation between forward and backward."""
    x, weight = _inputs()
    initial_state = torch.randn(
        1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = cute_causal_conv1d_silu(x, weight, initial_state=initial_state)

    with torch.no_grad():
        initial_state.add_(0.25)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output, (x, weight), torch.randn_like(output))


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


def test_short_conv_stateful_cuda_graph_replays_changed_history():
    """Read caller-owned dense history again on every graph replay."""
    x, weight = _inputs()
    initial_state = torch.randn(
        1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
    )
    grad_output = torch.randn_like(x)
    config = (128, 4, 10, 128, 4, 128)
    _forward_custom_op(x, weight, None, initial_state)
    _configured_backward_custom_op(x, weight, grad_output, None, initial_state, True, *config)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _forward_custom_op(x, weight, None, initial_state)
        captured_gradients = _configured_backward_custom_op(
            x, weight, grad_output, None, initial_state, True, *config
        )

    graph.replay()
    first_output = captured_output.clone()
    first_gradients = tuple(gradient.clone() for gradient in captured_gradients)
    with torch.no_grad():
        initial_state.add_(0.25)
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(captured_output, first_output)
    assert not torch.equal(captured_gradients[0], first_gradients[0])
    expected_output = _forward_custom_op(x, weight, None, initial_state)
    expected_gradients = _configured_backward_custom_op(
        x, weight, grad_output, None, initial_state, True, *config
    )
    torch.testing.assert_close(captured_output, expected_output)
    for actual_gradient, expected_gradient in zip(
        captured_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_short_conv_packed_stateful_cuda_graph_replays_boundaries_and_history():
    """Replay the packed-state custom operators with changed device metadata."""
    x, weight = _inputs(tokens=31)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 0, 11, 31, 31], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        cu_seqlens.shape[0] - 1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
    )
    config = (128, 4, 10, 128, 4, 128)
    _forward_custom_op(x, weight, cu_seqlens, initial_state)
    _configured_backward_custom_op(
        x, weight, grad_output, cu_seqlens, initial_state, True, *config
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _forward_custom_op(x, weight, cu_seqlens, initial_state)
        captured_gradients = _configured_backward_custom_op(
            x, weight, grad_output, cu_seqlens, initial_state, True, *config
        )

    with torch.no_grad():
        initial_state.add_(0.25)
        cu_seqlens.copy_(torch.tensor([0, 0, 8, 31, 31], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected_output = _forward_custom_op(x, weight, cu_seqlens, initial_state)
    expected_gradients = _configured_backward_custom_op(
        x, weight, grad_output, cu_seqlens, initial_state, True, *config
    )
    torch.testing.assert_close(captured_output, expected_output)
    for actual_gradient, expected_gradient in zip(
        captured_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_short_conv_public_packed_final_state_cuda_graph_replay():
    """Replay the complete packed initial/final-state public operation."""
    x, weight = _inputs(tokens=31)
    cu_seqlens = torch.tensor([0, 0, 11, 31, 31], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        cu_seqlens.shape[0] - 1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
    )

    def operation():
        return cute_causal_conv1d_silu(
            x,
            weight,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            return_final_state=True,
        )

    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_final = operation()

    with torch.no_grad():
        initial_state.add_(0.25)
        cu_seqlens.copy_(torch.tensor([0, 0, 8, 31, 31], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected_output, expected_final = operation()
    torch.testing.assert_close(captured_output, expected_output)
    torch.testing.assert_close(captured_final, expected_final)


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


def test_short_conv_packed_tma_cuda_graph_replays_changed_boundaries():
    """Reread arbitrary packed boundaries in TMA backward during graph replay."""
    x, weight = _inputs(tokens=384, channels=512)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor(
        [0, 0, 3, 17, 129, 257, 384],
        device="cuda",
        dtype=torch.int32,
    )
    _backward_custom_op(x, weight, grad_output, cu_seqlens)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_gradients = _backward_custom_op(x, weight, grad_output, cu_seqlens)

    with torch.no_grad():
        cu_seqlens.copy_(
            torch.tensor([0, 1, 8, 31, 130, 300, 384], device="cuda", dtype=torch.int32)
        )
    graph.replay()
    torch.cuda.synchronize()

    expected_gradients = _backward_custom_op(x, weight, grad_output, cu_seqlens)
    torch.testing.assert_close(captured_gradients[0], expected_gradients[0])
    torch.testing.assert_close(captured_gradients[1], expected_gradients[1])


def test_short_conv_packed_stateful_tma_cuda_graph_replays_boundaries_and_history():
    """Reread arbitrary boundaries and caller history in packed TMA backward replay."""
    x, weight = _inputs(tokens=384, channels=512)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor(
        [0, 0, 3, 17, 129, 257, 384],
        device="cuda",
        dtype=torch.int32,
    )
    initial_state = torch.randn(
        cu_seqlens.shape[0] - 1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=x.dtype,
    )
    defaults = ShortConvTunedConfig.default(x.dtype, packed=True, stateful=True)
    config = (
        defaults.input_gradient.threads,
        defaults.input_gradient.channels_per_thread,
        defaults.input_gradient.times_per_block,
        defaults.weight_gradient.threads,
        defaults.weight_gradient.channels_per_thread,
        defaults.weight_gradient.times_per_block,
    )
    _configured_backward_custom_op(
        x, weight, grad_output, cu_seqlens, initial_state, True, *config
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_gradients = _configured_backward_custom_op(
            x, weight, grad_output, cu_seqlens, initial_state, True, *config
        )

    first_gradients = tuple(gradient.clone() for gradient in captured_gradients)
    with torch.no_grad():
        initial_state.add_(0.25)
        cu_seqlens.copy_(
            torch.tensor([0, 1, 8, 31, 130, 300, 384], device="cuda", dtype=torch.int32)
        )
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(captured_gradients[0], first_gradients[0])
    assert not torch.equal(captured_gradients[1], first_gradients[1])
    expected_gradients = _configured_backward_custom_op(
        x, weight, grad_output, cu_seqlens, initial_state, True, *config
    )
    for actual_gradient, expected_gradient in zip(
        captured_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
