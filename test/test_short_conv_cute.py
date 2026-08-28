"""Correctness and integration tests for the CuTeDSL short convolution."""

import sys
from itertools import pairwise
from types import FunctionType

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

import attn_gym.linear.short_conv.cute as cute_backend
from attn_gym.linear.short_conv import activations
from attn_gym.linear.short_conv.cute import (
    ShortConvConfig,
    ShortConvTunedConfig,
    _candidate_configs,
    causal_conv1d,
    causal_conv1d_decode,
    tune_causal_conv1d,
)
from attn_gym.linear.short_conv.ops import (
    short_conv_backward_op as _backward_op,
)
from attn_gym.linear.short_conv.ops import (
    short_conv_configured_backward_op as _configured_backward_op,
)
from attn_gym.linear.short_conv.ops import (
    short_conv_configured_backward_with_state_grad_op as _configured_backward_with_state_grad_op,
)
from attn_gym.linear.short_conv.ops import (
    short_conv_configured_decode_op as _configured_decode_op,
)
from attn_gym.linear.short_conv.ops import (
    short_conv_configured_forward_op as _configured_forward_op,
)
from attn_gym.linear.short_conv.ops import (
    short_conv_decode_op as _decode_op,
)
from attn_gym.linear.short_conv.ops import (
    short_conv_forward_op as _forward_op,
)


def test_kda_backward_compatibility_exports():
    """The moved names stay importable from attn_gym.linear.kda and stay identical."""
    from attn_gym.linear import kda, short_conv

    for name in ("causal_conv1d", "causal_conv1d_decode", "register_activation"):
        assert getattr(kda, name) is getattr(short_conv, name)
        # Wildcard imports resolved these names before the move and must keep doing so.
        assert name in kda.__all__


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="the CuTeDSL short convolution requires CUDA capability 9.0 or newer",
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


def _plain_conv_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    padded = F.pad(x.transpose(1, 2), (weight.shape[1] - 1, 0))
    return F.conv1d(padded, weight[:, None], groups=x.shape[-1]).transpose(1, 2)


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.silu(_plain_conv_reference(x, weight))


def test_short_conv_forced_int64_matches_default(monkeypatch):
    """Widen forward and every training gradient without changing bits."""

    def run():
        torch.manual_seed(43)
        x, weight = _inputs(tokens=64, channels=512, batch=2)
        initial_state = torch.randn(
            2,
            weight.shape[1] - 1,
            x.shape[2],
            device="cuda",
            dtype=x.dtype,
            requires_grad=True,
        )
        output = causal_conv1d(
            x,
            weight,
            activation="silu",
            initial_state=initial_state,
        )
        gradients = torch.autograd.grad(output.square().sum(), (x, weight, initial_state))
        return output, *gradients

    baseline = run()
    monkeypatch.setattr(cute_backend, "requires_int64_abi", lambda *tensors: True)
    forced = run()
    for expected, actual in zip(baseline, forced, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


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
    output = torch.cat(outputs, dim=1) if outputs else x[:, :0]
    return output, torch.cat(final_states, dim=0)


def _misaligned_inputs(tokens: int = 17, channels: int = 12, width: int = 4):
    """Build contiguous views whose storage starts off a 16-byte boundary."""
    x_storage = torch.randn(tokens * channels + 1, device="cuda", dtype=torch.bfloat16)
    weight_storage = torch.randn(channels * width + 1, device="cuda", dtype=torch.bfloat16)
    x = x_storage[1:].view(1, tokens, channels).requires_grad_()
    weight = weight_storage[1:].view(channels, width).requires_grad_()
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    assert weight.is_contiguous() and weight.data_ptr() % 16 != 0
    return x, weight


def _decode_conv(x, weight, activation="silu"):
    """Walk the same sequence through the one-token step, one slot per batch row.

    Repeated decode steps from a zero history equal a single dense call, so both
    entry points share every reference below. The update is inference-only, hence
    the detach.
    """
    x, weight = x.detach(), weight.detach()
    state = x.new_zeros(x.shape[0], weight.shape[1] - 1, x.shape[2])
    return torch.stack(
        [
            causal_conv1d_decode(x[:, t].contiguous(), weight, state, activation=activation)
            for t in range(x.shape[1])
        ],
        dim=1,
    )


def _assert_conv_matches(run, x, weight, activation="silu", rtol=2e-2, atol=2e-2):
    """Compare an entry point against the reference and return both tensors."""
    actual = run(x, weight, activation=activation)
    convolve = _reference if activation == "silu" else _plain_conv_reference
    expected = convolve(x, weight)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return actual, expected


@pytest.mark.parametrize("width", [1, 4, 5])
def test_short_conv_forward_and_backward_match_pytorch(width: int):
    """Check generic widths and partial first and last time tiles."""
    torch.manual_seed(0)
    x, weight = _inputs(width=width)
    grad_output = torch.randn_like(x)

    actual, expected = _assert_conv_matches(causal_conv1d, x, weight)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=2e-1)


@pytest.mark.parametrize(
    ("sequences", "width", "activation", "channels_per_thread"),
    [
        (8, 4, "silu", 1),
        (9, 4, "silu", 2),
        (8, 5, "silu", 4),
        (8, 4, None, 4),
    ],
)
def test_short_conv_hopper_decode_defaults(
    sequences: int,
    width: int,
    activation: str | None,
    channels_per_thread: int,
):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper-specific decode schedule")
    x = torch.empty(sequences, 6144, device="cuda", dtype=torch.bfloat16)
    config = cute_backend._default_decode_config(x, width, activation)
    assert config == ShortConvConfig(128, channels_per_thread, 16)


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
    packed_fp16 = ShortConvTunedConfig.default(torch.float16, packed=True)
    packed = ShortConvTunedConfig.default(packed=True)
    packed_fp32 = ShortConvTunedConfig.default(torch.float32, packed=True)
    assert packed_fp16.input_gradient == ShortConvConfig(128, 4, 16)
    assert packed_fp16.weight_gradient == ShortConvConfig(128, 2, 32)
    assert packed.input_gradient == ShortConvConfig(128, 4, 16)
    assert packed.weight_gradient == ShortConvConfig(128, 2, 32)
    assert packed_fp32.input_gradient == ShortConvConfig(128, 2, 16)
    assert packed_fp32.weight_gradient == ShortConvConfig(128, 4, 32)
    assert fp32.forward == ShortConvConfig(128, 4, 4)
    assert fp32.input_gradient == ShortConvConfig(128, 2, 12)
    assert fp32.weight_gradient == ShortConvConfig(128, 2, 160)

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        defaults = ShortConvTunedConfig.default(dtype, packed=True)
        assert defaults.input_gradient in _candidate_configs(
            "input_gradient", 512, dtype, packed=True
        )
        assert defaults.weight_gradient in _candidate_configs(
            "weight_gradient", 512, dtype, packed=True
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_short_conv_packed_defaults_select_tma(dtype: torch.dtype):
    """Keep the production packed defaults on the staged implementation."""
    defaults = ShortConvTunedConfig.default(dtype, packed=True)
    descriptor = cute_backend.SHORT_CONV_DTYPES[dtype]
    assert cute_backend.supports_tma(
        cute_backend.CausalConv1dSiluInputGradientTma,
        defaults.input_gradient,
        cute_backend.tuned_config(descriptor, packed=True).input_gradient,
        512,
        4,
    )
    assert cute_backend.supports_tma(
        cute_backend.CausalConv1dSiluWeightGradientPartialsTma,
        defaults.weight_gradient,
        cute_backend.tuned_config(descriptor, packed=True).weight_gradient,
        512,
        4,
    )


def test_short_conv_tma_rejects_partial_channel_tiles():
    """Make full channel tiles an explicit invariant of direct TMA construction."""
    config = ShortConvTunedConfig.default(torch.bfloat16, packed=True).input_gradient
    with pytest.raises(AssertionError, match="must be divisible by the channel tile"):
        cute_backend.CausalConv1dSiluInputGradientTma(
            1,
            384,
            513,
            4,
            config,
            cute_backend.SHORT_CONV_DTYPES[torch.bfloat16],
            activations._silu_derivative,
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

    actual = causal_conv1d(x, weight, activation="silu", initial_state=initial_state)
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

    actual, expected = _assert_conv_matches(causal_conv1d, x, weight)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=3e-1)

    compiled = torch.compile(causal_conv1d, fullgraph=True)
    torch.testing.assert_close(
        compiled(x, weight, activation="silu"), expected, rtol=2e-2, atol=2e-2
    )

    for kind in ("forward", "input_gradient", "weight_gradient"):
        candidates = _candidate_configs(kind, channels)
        assert candidates
        assert all(channels % config.channels_per_thread == 0 for config in candidates)


@pytest.mark.parametrize(
    ("width", "dtype"),
    [(3, torch.bfloat16), (4, torch.float32), (5, torch.float16)],
)
@pytest.mark.parametrize("tokens", [384, 379])
def test_short_conv_tma_backward_matches_batched_reference(
    width: int,
    dtype: torch.dtype,
    tokens: int,
):
    """Exercise dense batches, including partial trailing TMA boxes, through TMA schedules."""
    torch.manual_seed(7)
    x, weight = _inputs(tokens=tokens, channels=256, width=width, batch=2, dtype=dtype)
    grad_output = torch.randn_like(x)

    actual = causal_conv1d(x, weight, activation="silu")
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
        activation="silu",
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
        activation="silu",
    )[0]

    torch.testing.assert_close(actual_dx, expected_dx, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("tokens", [384, 379])
def test_short_conv_packed_tma_backward_matches_fallback(tokens: int):
    """Stage physical tiles while resetting both gradients at arbitrary boundaries."""
    torch.manual_seed(8)
    x, weight = _inputs(tokens=tokens, channels=512, width=4)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor(
        [0, 0, 3, 17, 18, 129, 257, tokens],
        device="cuda",
        dtype=torch.int32,
    )

    actual = causal_conv1d(x, weight, activation="silu", cu_seqlens=cu_seqlens)
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
        activation="silu",
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=2e-1)

    # One BF16 ULP of headroom: partial trailing time blocks change dw's
    # FP32 partial-reduction boundaries between the TMA and fallback trees.
    torch.testing.assert_close(actual_gradients[0], fallback_gradients[0], rtol=1e-2, atol=1e-4)
    torch.testing.assert_close(actual_gradients[1], fallback_gradients[1], rtol=1e-2, atol=1e-4)


def test_short_conv_dense_stateful_tma_backward_matches_reference():
    """Use caller history in aligned dense TMA input- and weight-gradient tiles."""
    torch.manual_seed(12)
    x, weight = _inputs(tokens=384, channels=256, width=4, batch=2)
    initial_state = torch.randn(2, 3, 256, device="cuda", dtype=x.dtype, requires_grad=True)
    reference_state = initial_state.detach().clone().requires_grad_()
    grad_output = torch.randn_like(x)

    actual = causal_conv1d(x, weight, activation="silu", initial_state=initial_state)
    expected_input = torch.cat((reference_state, x), dim=1)
    expected = F.silu(
        F.conv1d(expected_input.transpose(1, 2), weight[:, None], groups=256)
    ).transpose(1, 2)
    actual_gradients = torch.autograd.grad(actual, (x, weight, initial_state), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight, reference_state), grad_output)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    for index, (actual_gradient, expected_gradient) in enumerate(
        zip(actual_gradients, expected_gradients, strict=True)
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=3e-2,
            atol=2e-1 if index == 1 else 3e-2,
        )


def test_short_conv_packed_stateful_tma_backward_matches_reference_and_fallback():
    """Seed every packed TMA convolution window from caller-owned sequence history."""
    torch.manual_seed(9)
    x, weight = _inputs(tokens=384, channels=512, width=4)
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
        dtype=x.dtype,
        requires_grad=True,
    )
    reference_state = initial_state.detach().clone().requires_grad_()

    actual = causal_conv1d(
        x,
        weight,
        activation="silu",
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
        activation="silu",
    )

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    for index, (actual_gradient, expected_gradient, fallback_gradient) in enumerate(
        zip(actual_gradients, expected_gradients, fallback_gradients, strict=True)
    ):
        atol = 2e-1 if index == 1 else 3e-2
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=atol)
        torch.testing.assert_close(actual_gradient, fallback_gradient, rtol=3e-2, atol=atol)


def test_short_conv_packed_forward_active_endpoint_is_bitwise_exact():
    """Skip inactive capacity without changing any active forward value."""
    tokens, active_tokens = 257, 63
    x, weight = _inputs(tokens=tokens, channels=512, width=4)
    cu_seqlens = torch.tensor(
        [0, 7, active_tokens, active_tokens],
        device="cuda",
        dtype=torch.int32,
    )
    expected = causal_conv1d(
        x[:, :active_tokens].detach().clone(),
        weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
    )
    with torch.no_grad():
        x[:, active_tokens:].fill_(float("nan"))
    actual = causal_conv1d(x, weight, activation="silu", cu_seqlens=cu_seqlens)
    torch.testing.assert_close(actual[:, :active_tokens], expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("stateful", "tokens", "channels", "width", "boundaries"),
    [
        (False, 67, 12, 5, [0, 0, 3, 17, 53]),
        (True, 67, 12, 5, [0, 0, 3, 17, 53]),
        (False, 384, 512, 4, [0, 0, 3, 17, 301]),
        (False, 384, 512, 4, [0, 0, 3, 17, 256]),
        (True, 67, 12, 5, [0, 0, 0, 0, 0]),
        (False, 384, 512, 4, [0, 301]),
        (False, 384, 512, 4, [0, 384]),
        (True, 384, 512, 4, [0, 384]),
    ],
    ids=[
        "generic-stateless",
        "generic-stateful",
        "tma",
        "tma-boundary",
        "all-inactive-stateful",
        "tma-single-sequence",
        "tma-single-sequence-full-stateless",
        "tma-single-sequence-full-stateful",
    ],
)
def test_short_conv_dynamic_active_endpoint_ignores_nan_suffix(
    stateful: bool,
    tokens: int,
    channels: int,
    width: int,
    boundaries: list[int],
):
    """Exclude inactive physical capacity from every defined value and gradient."""
    torch.manual_seed(23)
    active_tokens = boundaries[-1]
    x, weight = _inputs(tokens=tokens, channels=channels, width=width)
    with torch.no_grad():
        x[:, active_tokens:].fill_(float("nan"))
    cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    initial_state = None
    if stateful:
        initial_state = torch.randn(
            cu_seqlens.shape[0] - 1,
            width - 1,
            channels,
            device="cuda",
            dtype=x.dtype,
            requires_grad=True,
        )

    expected_x = x[:, :active_tokens].detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected_state = (
        torch.zeros(
            cu_seqlens.shape[0] - 1,
            width - 1,
            channels,
            device="cuda",
            dtype=x.dtype,
        )
        if initial_state is None
        else initial_state.detach().clone().requires_grad_()
    )
    expected_output, expected_final = _packed_state_reference(
        expected_x,
        expected_weight,
        cu_seqlens,
        expected_state,
    )
    actual_output, actual_final = causal_conv1d(
        x,
        weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        return_final_state=True,
    )
    torch.testing.assert_close(
        actual_output[:, :active_tokens], expected_output, rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(actual_final, expected_final)

    grad_output = torch.randn_like(x)
    grad_output[:, active_tokens:].fill_(float("nan"))
    grad_final = torch.randn_like(actual_final)
    actual_inputs = (x, weight) if initial_state is None else (x, weight, initial_state)
    expected_inputs = (
        (expected_x, expected_weight)
        if initial_state is None
        else (expected_x, expected_weight, expected_state)
    )
    actual_gradients = torch.autograd.grad(
        (actual_output, actual_final),
        actual_inputs,
        (grad_output, grad_final),
    )
    expected_gradients = torch.autograd.grad(
        (expected_output, expected_final),
        expected_inputs,
        (grad_output[:, :active_tokens], grad_final),
        allow_unused=True,
    )
    expected_gradients = tuple(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(expected_inputs, expected_gradients, strict=True)
    )
    torch.testing.assert_close(
        actual_gradients[0][:, :active_tokens],
        expected_gradients[0],
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(
        actual_gradients[1],
        expected_gradients[1],
        rtol=3e-2,
        atol=2e-1,
    )
    if stateful:
        torch.testing.assert_close(
            actual_gradients[2],
            expected_gradients[2],
            rtol=3e-2,
            atol=3e-2,
        )


def test_short_conv_batched_forward_and_backward_match_pytorch():
    """Keep batches independent across the optimized convolution width."""
    width = 4
    torch.manual_seed(1)
    x, weight = _inputs(tokens=19, channels=12, width=width, batch=3)
    grad_output = torch.randn_like(x)
    weight_config = ShortConvConfig(128, 4, 8)

    actual = causal_conv1d(x, weight, activation="silu", weight_grad_config=weight_config)
    expected = _reference(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=3e-1)


@pytest.mark.parametrize("width", [1, 5])
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

    actual = causal_conv1d(x, weight, activation="silu", cu_seqlens=cu_seqlens)
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
    actual, actual_final = causal_conv1d(
        x,
        weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
        return_final_state=True,
    )
    zero_state = x.new_zeros(cu_seqlens.shape[0] - 1, weight.shape[1] - 1, x.shape[2])
    expected, expected_final = _packed_state_reference(x, weight, cu_seqlens, zero_state)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_final, expected_final)


@pytest.mark.parametrize("width", [1, 5])
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

    actual, actual_final = causal_conv1d(
        x,
        weight,
        activation="silu",
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


def test_short_conv_accepts_misaligned_contiguous_storage():
    """Materialize alignment inside the opaque launcher when a view starts off-boundary."""
    x, weight = _misaligned_inputs()

    _assert_conv_matches(causal_conv1d, x, weight)


def test_short_conv_explicit_config_and_tuning_flow():
    """Route stateful schedules through the shared compile-and-benchmark tuner."""
    width = 3
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
    selected = tune_causal_conv1d(
        x,
        weight,
        grad_output,
        activation="silu",
        initial_state=initial_state,
        forward_configs=(forward,),
        input_grad_configs=(input_gradient,),
        weight_grad_configs=(weight_gradient,),
        parallel_compile=False,
    )
    assert selected.forward == forward
    assert selected.input_gradient == input_gradient
    assert selected.weight_gradient == weight_gradient
    actual = causal_conv1d(
        x,
        weight,
        activation="silu",
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
        lambda x, weight, initial_state: causal_conv1d(
            x,
            weight,
            activation="silu",
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
    # The raw ops have no autograd kernel; differentiable use goes through the
    # autograd.Function wrappers, so opcheck runs on detached inputs.
    x, weight = _inputs(batch=1 if packed else 2)
    x, weight = x.detach(), weight.detach()
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 2, 7, 17], device="cuda", dtype=torch.int32) if packed else None
    torch.library.opcheck(_forward_op, (x, weight, cu_seqlens))
    torch.library.opcheck(
        _backward_op,
        (x, weight, grad_output, cu_seqlens),
        test_utils=("test_schema", "test_faketensor"),
    )
    if packed:
        configs = (128, 4, 8, 128, 4, 10, 128, 4, 128)
        torch.library.opcheck(
            _configured_forward_op,
            (x, weight, cu_seqlens, None, *configs),
        )
        torch.library.opcheck(
            _configured_backward_op,
            (x, weight, grad_output, cu_seqlens, None, *configs[3:], False),
            test_utils=("test_schema", "test_faketensor"),
        )


def test_short_conv_stateful_custom_op_registration():
    """Exercise packed schemas, fake tensors, and autograd with caller-provided history."""
    x, weight = _inputs()
    x, weight = x.detach(), weight.detach()
    cu_seqlens = torch.tensor([0, 2, 7, 17], device="cuda", dtype=torch.int32)
    num_sequences = cu_seqlens.shape[0] - 1
    initial_state = torch.randn(
        num_sequences,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
    )
    grad_output = torch.randn_like(x)
    torch.library.opcheck(_forward_op, (x, weight, cu_seqlens, initial_state))
    config = (128, 4, 10, 128, 4, 128, False)
    torch.library.opcheck(
        _configured_backward_op,
        (x, weight, grad_output, cu_seqlens, initial_state, *config),
        test_utils=("test_schema", "test_faketensor"),
    )
    torch.library.opcheck(
        _configured_backward_with_state_grad_op,
        (x, weight, grad_output, cu_seqlens, initial_state, *config),
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
    output = causal_conv1d(x, weight, activation="silu", initial_state=initial_state)
    torch.autograd.grad(output, (x, weight), torch.randn_like(output))


def test_short_conv_fullgraph_forward_and_backward():
    """Keep batched opaque operators inside a strict compiled graph."""
    x, weight = _inputs(batch=2)
    grad_output = torch.randn_like(x)

    expected_output = causal_conv1d(x, weight, activation="silu")
    expected = torch.autograd.grad(expected_output, (x, weight), grad_output)
    compiled = torch.compile(causal_conv1d, fullgraph=True)
    actual_output = compiled(x, weight, activation="silu")
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
        return causal_conv1d(
            x,
            weight,
            activation="silu",
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
    output = causal_conv1d(x, weight, activation="silu", initial_state=initial_state)
    (grad_initial_state,) = torch.autograd.grad(output.sum(), (initial_state,))
    assert grad_initial_state.shape == initial_state.shape

    packed_x, packed_weight = _inputs(tokens=7, width=1)
    cu_seqlens = torch.tensor([0, 0, 2, 7], device="cuda", dtype=torch.int32)
    _, final_state = causal_conv1d(
        packed_x,
        packed_weight,
        activation="silu",
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
    _, final_state = causal_conv1d(
        x,
        weight,
        activation="silu",
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
    output = causal_conv1d(x, weight, activation="silu", initial_state=initial_state)

    with torch.no_grad():
        initial_state.add_(0.25)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output, (x, weight), torch.randn_like(output))


def test_short_conv_packed_offsets_obey_autograd_versioning():
    """Reject sequence-boundary mutation between forward and backward."""
    x, weight = _inputs(tokens=31)
    cu_seqlens = torch.tensor([0, 3, 11, 31], device="cuda", dtype=torch.int32)
    output = causal_conv1d(x, weight, activation="silu", cu_seqlens=cu_seqlens)

    with torch.no_grad():
        cu_seqlens[1] = 4
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(output, (x, weight), torch.randn_like(output))


def test_short_conv_cuda_graph_replay():
    """Capture the compiled launchers and replay with changed input values."""
    x, weight = _inputs()
    grad_output = torch.randn_like(x)
    _forward_op(x, weight, activation="silu")
    _backward_op(x, weight, grad_output, activation="silu")
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _forward_op(x, weight, activation="silu")
        captured_gradients = _backward_op(x, weight, grad_output, activation="silu")

    graph.replay()
    first_output = captured_output.clone()
    first_gradients = tuple(gradient.clone() for gradient in captured_gradients)
    with torch.no_grad():
        x.add_(0.25)
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(captured_output, first_output)
    assert not torch.equal(captured_gradients[0], first_gradients[0])
    expected_output = _forward_op(x, weight, activation="silu")
    expected_gradients = _backward_op(x, weight, grad_output, activation="silu")
    torch.testing.assert_close(captured_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(captured_gradients[0], expected_gradients[0], rtol=0, atol=0)
    torch.testing.assert_close(captured_gradients[1], expected_gradients[1], rtol=0, atol=0)


def test_short_conv_persistent_tma_input_gradient_replays_phase_wrap():
    """Carry TMA pipeline phases across worker tiles and changing graph endpoints."""
    defaults = ShortConvTunedConfig.default(torch.bfloat16, packed=True)
    input_config = defaults.input_gradient
    weight_config = defaults.weight_gradient
    device = torch.device("cuda", torch.cuda.current_device())
    channels = input_config.threads * input_config.channels_per_thread
    workers = cute_backend._persistent_tma_dx_workers(
        32_768,
        channels,
        input_config,
        device,
    )
    tokens = (workers + 3) * input_config.times_per_block
    x, weight = _inputs(tokens=tokens, channels=channels, width=4)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor(
        [0, tokens // 4, tokens // 2, tokens, tokens],
        device="cuda",
        dtype=torch.int32,
    )
    base_args = (
        input_config.threads,
        input_config.channels_per_thread,
        input_config.times_per_block,
        weight_config.threads,
        weight_config.channels_per_thread,
        weight_config.times_per_block,
    )
    persistent_args = (*base_args, True)
    static_args = (*base_args, False)

    _configured_backward_op(
        x,
        weight,
        grad_output,
        cu_seqlens,
        None,
        *persistent_args,
        activation="silu",
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        persistent_dx, persistent_dw = _configured_backward_op(
            x,
            weight,
            grad_output,
            cu_seqlens,
            None,
            *persistent_args,
            activation="silu",
        )

    second_worker_tile = workers * input_config.times_per_block
    replay_cases = (
        (
            (workers + 1) * input_config.times_per_block - 3,
            [0, 7, second_worker_tile - 1, second_worker_tile + 5],
        ),
        (7, [0, 0, 3, 7]),
    )
    for active_tokens, prefix in replay_cases:
        with torch.no_grad():
            cu_seqlens.copy_(
                torch.tensor([*prefix, active_tokens], device="cuda", dtype=torch.int32)
            )
            x[:, active_tokens:].fill_(float("nan"))
            grad_output[:, active_tokens:].zero_()
        graph.replay()
        torch.cuda.synchronize()
        static_dx, static_dw = _configured_backward_op(
            x,
            weight,
            grad_output,
            cu_seqlens,
            None,
            *static_args,
            activation="silu",
        )
        torch.testing.assert_close(
            persistent_dx[:, :active_tokens],
            static_dx[:, :active_tokens],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(persistent_dw, static_dw, rtol=0, atol=0)


def test_short_conv_public_persistent_tma_input_gradient_matches_static():
    """Compile stateful persistent TMA dx without changing any defined result."""
    tokens, channels = 1024, 1536
    static_x, static_weight = _inputs(tokens=tokens, channels=channels, width=4)
    persistent_x = static_x.detach().clone().requires_grad_()
    persistent_weight = static_weight.detach().clone().requires_grad_()
    grad_output = torch.randn_like(static_x)
    cu_seqlens = torch.arange(0, tokens + 1, tokens // 16, device="cuda", dtype=torch.int32)
    static_state = torch.randn(
        cu_seqlens.shape[0] - 1,
        static_weight.shape[1] - 1,
        channels,
        device="cuda",
        dtype=static_x.dtype,
        requires_grad=True,
    )
    persistent_state = static_state.detach().clone().requires_grad_()

    static_output = causal_conv1d(
        static_x,
        static_weight,
        activation="silu",
        cu_seqlens=cu_seqlens,
        initial_state=static_state,
    )
    compiled = torch.compile(
        lambda x, weight, state: causal_conv1d(
            x,
            weight,
            activation="silu",
            cu_seqlens=cu_seqlens,
            initial_state=state,
            persistent_tma_input_gradient=True,
        ),
        fullgraph=True,
    )
    persistent_output = compiled(persistent_x, persistent_weight, persistent_state)
    static_gradients = torch.autograd.grad(
        static_output,
        (static_x, static_weight, static_state),
        grad_output,
    )
    persistent_gradients = torch.autograd.grad(
        persistent_output,
        (persistent_x, persistent_weight, persistent_state),
        grad_output,
    )
    torch.testing.assert_close(persistent_output, static_output, rtol=0, atol=0)
    for actual, expected in zip(persistent_gradients, static_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_short_conv_persistent_tma_input_gradient_validates_route():
    """Reject the explicit option when no packed TMA input-gradient route exists."""
    x, weight = _inputs(tokens=32, channels=12)
    with pytest.raises(ValueError, match="require packed cu_seqlens"):
        causal_conv1d(x, weight, persistent_tma_input_gradient=True)

    cu_seqlens = torch.tensor([0, 32], device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="require the staged input-gradient"):
        causal_conv1d(
            x,
            weight,
            cu_seqlens=cu_seqlens,
            persistent_tma_input_gradient=True,
        )


def test_short_conv_packed_stateful_cuda_graph_replays_boundaries_and_history():
    """Replay fallback kernels after changing boundaries, active length, and history."""
    x, weight = _inputs(tokens=31)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor([0, 0, 11, 27, 27], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        cu_seqlens.shape[0] - 1,
        weight.shape[1] - 1,
        x.shape[2],
        device="cuda",
        dtype=torch.bfloat16,
    )
    config = (128, 4, 10, 128, 4, 128, False)
    _forward_op(x, weight, cu_seqlens, initial_state, activation="silu")
    _configured_backward_with_state_grad_op(
        x, weight, grad_output, cu_seqlens, initial_state, *config, activation="silu"
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _forward_op(x, weight, cu_seqlens, initial_state, activation="silu")
        captured_gradients = _configured_backward_with_state_grad_op(
            x, weight, grad_output, cu_seqlens, initial_state, *config, activation="silu"
        )

    active_tokens = 23
    with torch.no_grad():
        initial_state.add_(0.25)
        cu_seqlens.copy_(
            torch.tensor([0, 0, 8, active_tokens, active_tokens], device="cuda", dtype=torch.int32)
        )
        x[:, active_tokens:].fill_(float("nan"))
        grad_output[:, active_tokens:].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    expected_output = _forward_op(x, weight, cu_seqlens, initial_state, activation="silu")
    expected_gradients = _configured_backward_with_state_grad_op(
        x, weight, grad_output, cu_seqlens, initial_state, *config, activation="silu"
    )
    torch.testing.assert_close(
        captured_output[:, :active_tokens], expected_output[:, :active_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        captured_gradients[0][:, :active_tokens],
        expected_gradients[0][:, :active_tokens],
        rtol=0,
        atol=0,
    )
    for actual_gradient, expected_gradient in zip(
        captured_gradients[1:],
        expected_gradients[1:],
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_short_conv_packed_stateful_tma_cuda_graph_replays_boundaries_and_history():
    """Reread boundaries, active length, and caller history in packed TMA replay."""
    x, weight = _inputs(tokens=384, channels=512)
    grad_output = torch.randn_like(x)
    cu_seqlens = torch.tensor(
        [0, 0, 3, 17, 129, 257, 301],
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
    defaults = ShortConvTunedConfig.default(x.dtype, packed=True)
    config = (
        defaults.input_gradient.threads,
        defaults.input_gradient.channels_per_thread,
        defaults.input_gradient.times_per_block,
        defaults.weight_gradient.threads,
        defaults.weight_gradient.channels_per_thread,
        defaults.weight_gradient.times_per_block,
        False,
    )
    _configured_backward_with_state_grad_op(
        x, weight, grad_output, cu_seqlens, initial_state, *config, activation="silu"
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_gradients = _configured_backward_with_state_grad_op(
            x, weight, grad_output, cu_seqlens, initial_state, *config, activation="silu"
        )

    first_gradients = tuple(gradient.clone() for gradient in captured_gradients)
    active_tokens = 279
    with torch.no_grad():
        initial_state.add_(0.25)
        cu_seqlens.copy_(
            torch.tensor(
                [0, 1, 8, 31, 130, 250, active_tokens],
                device="cuda",
                dtype=torch.int32,
            )
        )
        x[:, active_tokens:].fill_(float("nan"))
        grad_output[:, active_tokens:].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(
        captured_gradients[0][:, :active_tokens], first_gradients[0][:, :active_tokens]
    )
    assert not torch.equal(captured_gradients[1], first_gradients[1])
    expected_gradients = _configured_backward_with_state_grad_op(
        x, weight, grad_output, cu_seqlens, initial_state, *config, activation="silu"
    )
    torch.testing.assert_close(
        captured_gradients[0][:, :active_tokens],
        expected_gradients[0][:, :active_tokens],
        rtol=0,
        atol=0,
    )
    for actual_gradient, expected_gradient in zip(
        captured_gradients[1:],
        expected_gradients[1:],
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_short_conv_identity_activation_matches_plain_conv():
    """Run the None-activation kernels against an activation-free reference."""
    torch.manual_seed(5)
    x, weight = _inputs()
    grad_output = torch.randn_like(x)

    actual, expected = _assert_conv_matches(causal_conv1d, x, weight, None)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=3e-1)


def test_short_conv_identity_activation_packed_stateful():
    """Check the identity derivative through the packed stateful gradient kernels."""
    torch.manual_seed(6)
    x, weight = _inputs(width=4)
    cu_seqlens = torch.tensor([0, 2, 7, 17], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(
        3, weight.shape[1] - 1, x.shape[2], device="cuda", dtype=torch.bfloat16
    ).requires_grad_()
    grad_output = torch.randn_like(x)

    actual = causal_conv1d(x, weight, cu_seqlens=cu_seqlens, initial_state=initial_state)
    outputs = []
    for sequence, (start, end) in enumerate(pairwise(cu_seqlens.cpu().tolist())):
        extended = torch.cat((initial_state[sequence : sequence + 1], x[:, start:end]), dim=1)
        outputs.append(
            F.conv1d(extended.transpose(1, 2), weight[:, None], groups=x.shape[-1]).transpose(1, 2)
        )
    expected = torch.cat(outputs, dim=1)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    leaves = (x, weight, initial_state)
    actual_gradients = torch.autograd.grad(actual, leaves, grad_output)
    expected_gradients = torch.autograd.grad(expected, leaves, grad_output)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=3e-1)


_ACTIVATION_GLOBAL_SCALE = 2.0


def _globally_scaled_activation(value):
    return value * _ACTIVATION_GLOBAL_SCALE


def _globally_scaled_derivative(value):
    return _ACTIVATION_GLOBAL_SCALE


def _tanh_activation(value):
    from cutlass import cute

    return cute.math.tanh(value, fastmath=True)


def _tanh_activation_derivative(value):
    from cutlass import cute

    tanh_value = cute.math.tanh(value, fastmath=True)
    return 1.0 - tanh_value * tanh_value


def test_short_conv_registered_custom_activation():
    """Fuse a user-registered CuTeDSL activation and validate both gradients."""
    activations.register_activation("tanh", _tanh_activation, _tanh_activation_derivative)
    torch.manual_seed(7)
    x, weight = _inputs()
    grad_output = torch.randn_like(x)

    actual = causal_conv1d(x, weight, activation="tanh")
    expected = torch.tanh(_plain_conv_reference(x, weight))
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    actual_gradients = torch.autograd.grad(actual, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected, (x, weight), grad_output)
    torch.testing.assert_close(actual_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=3e-2, atol=3e-1)


def _make_scaled(scale: float):
    def scaled_activation(value):
        return value * scale

    def scaled_derivative(value):
        return scale

    return scaled_activation, scaled_derivative


def test_short_conv_activation_registration_contract():
    """Reject unknown names, invalid names, and silent re-registration."""
    x, weight = _inputs()
    with pytest.raises(ValueError, match="unknown activation"):
        causal_conv1d(x, weight, activation="missing")
    with pytest.raises(ValueError, match="nonempty string"):
        activations.register_activation("", _tanh_activation, _tanh_activation_derivative)
    # Distinct function objects with identical sources may reuse a name; different
    # sources must not, so a stale cached kernel can never be served silently.
    equivalent_forward = FunctionType(_tanh_activation.__code__, _tanh_activation.__globals__)
    equivalent_derivative = FunctionType(
        _tanh_activation_derivative.__code__, _tanh_activation_derivative.__globals__
    )
    activations.register_activation("tanh-contract", _tanh_activation, _tanh_activation_derivative)
    activations.register_activation("tanh-contract", equivalent_forward, equivalent_derivative)
    # The no-op keeps the original, importable callables in the registry.
    assert activations._ACTIVATIONS["tanh-contract"].forward is _tanh_activation
    # Closure re-registration is a no-op only for equal captured values.
    activations.register_activation("scaled-contract", *_make_scaled(2.0))
    activations.register_activation("scaled-contract", *_make_scaled(2.0))
    with pytest.raises(ValueError, match="different implementation"):
        activations.register_activation("scaled-contract", *_make_scaled(3.0))
    # Functions without retrievable source have no stable cache identity.
    namespace: dict = {}
    exec("def sourceless(value):\n    return value", namespace)  # noqa: S102 -- sourceless fixture
    with pytest.raises(ValueError, match="no stable cache identity"):
        activations.register_activation(
            "sourceless", namespace["sourceless"], namespace["sourceless"]
        )
    with pytest.raises(ValueError, match="different implementation"):
        activations.register_activation(
            "tanh-contract", activations._silu, activations._silu_derivative
        )


def test_short_conv_closure_activations_key_on_captured_values():
    """Two factory instances with different captures compile different kernels."""
    activations.register_activation("scaled-two", *_make_scaled(2.0))
    activations.register_activation("scaled-three", *_make_scaled(3.0))
    torch.manual_seed(9)
    x, weight = _inputs(channels=6, width=3)
    grad_output = torch.randn_like(x)

    doubled = causal_conv1d(x, weight, activation="scaled-two")
    tripled = causal_conv1d(x, weight, activation="scaled-three")
    expected = _plain_conv_reference(x, weight)
    torch.testing.assert_close(doubled, expected * 2.0, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(tripled, expected * 3.0, rtol=2e-2, atol=2e-2)

    doubled_gradients = torch.autograd.grad(doubled, (x, weight), grad_output)
    expected_gradients = torch.autograd.grad(expected * 2.0, (x, weight), grad_output)
    torch.testing.assert_close(doubled_gradients[0], expected_gradients[0], rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(doubled_gradients[1], expected_gradients[1], rtol=3e-2, atol=3e-1)


def test_short_conv_global_mutation_recompiles():
    """Mutating a referenced module global must recompile, not reuse stale kernels."""
    module = sys.modules[__name__]
    activations.register_activation(
        "global-scaled", _globally_scaled_activation, _globally_scaled_derivative
    )
    torch.manual_seed(10)
    x, weight = _inputs(channels=6, width=3)
    expected = _plain_conv_reference(x, weight)

    original = module._ACTIVATION_GLOBAL_SCALE
    try:
        doubled = causal_conv1d(x, weight, activation="global-scaled")
        torch.testing.assert_close(doubled, expected * original, rtol=2e-2, atol=2e-2)
        module._ACTIVATION_GLOBAL_SCALE = original + 1.0
        rescaled = causal_conv1d(x, weight, activation="global-scaled")
        torch.testing.assert_close(rescaled, expected * (original + 1.0), rtol=2e-2, atol=2e-2)
    finally:
        module._ACTIVATION_GLOBAL_SCALE = original


def test_tune_compiles_serially_for_script_defined_activations(monkeypatch):
    """__main__ and closure callables cannot cross the compiler-process boundary."""
    captured = []

    def fake_tune(configs, compile_fn, launch, *, compile_call=None, parallel_compile=True):
        captured.append(parallel_compile)
        return next(iter(configs))

    monkeypatch.setattr(cute_backend, "tune", fake_tune)
    script_forward = FunctionType(_tanh_activation.__code__, _tanh_activation.__globals__)
    script_derivative = FunctionType(
        _tanh_activation_derivative.__code__, _tanh_activation_derivative.__globals__
    )
    script_forward.__module__ = script_derivative.__module__ = "__main__"
    activations.register_activation("script-scoped", script_forward, script_derivative)

    x, weight = _inputs(channels=6, width=3)
    config = ShortConvConfig(128, 2, 8)
    selected = tune_causal_conv1d(
        x,
        weight,
        torch.randn_like(x),
        activation="script-scoped",
        forward_configs=(config,),
        input_grad_configs=(config,),
        weight_grad_configs=(config,),
    )
    assert selected == ShortConvTunedConfig(config, config, config)
    assert captured == [False, False, False]

    captured.clear()
    activations.register_activation("closure-scoped", *_make_scaled(4.0))
    tune_causal_conv1d(
        x,
        weight,
        torch.randn_like(x),
        activation="closure-scoped",
        forward_configs=(config,),
        input_grad_configs=(config,),
        weight_grad_configs=(config,),
    )
    assert captured == [False, False, False]


def _defaulted_activation(value, options={"scale": 2.0}):  # noqa: B006 -- mutable default fixture
    return value * options["scale"]


def test_function_cache_key_covers_defaults_and_nested_functions():
    """Pin the key surface: mutable defaults change keys; nested fns stay serial."""
    from attn_gym._backends.cute import function_cache_key

    before = function_cache_key(_defaulted_activation)
    _defaulted_activation.__defaults__[0]["scale"] = 3.0
    try:
        assert function_cache_key(_defaulted_activation) != before
    finally:
        _defaulted_activation.__defaults__[0]["scale"] = 2.0

    def nested_activation(value, scale=2.0):
        return value * scale

    # No closure, real module, but <locals>: must not claim process crossing.
    activation = activations.Activation("nested", nested_activation, nested_activation)
    assert not activation.crosses_process_boundary


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float16, 2e-2, 2e-2), (torch.bfloat16, 2e-2, 2e-2), (torch.float32, 2e-5, 2e-5)],
)
def test_short_conv_decode_supported_dtypes(dtype: torch.dtype, rtol: float, atol: float):
    """Specialize storage while retaining FP32 convolution accumulation."""
    torch.manual_seed(5)
    x, weight = _inputs(tokens=9, channels=12, width=4, batch=2, dtype=dtype)
    _assert_conv_matches(_decode_conv, x, weight, rtol=rtol, atol=atol)


@pytest.mark.parametrize("channels", [5, 6])
def test_short_conv_decode_supports_any_positive_channel_count(channels: int):
    """Select a compatible channel width without requiring an explicit config."""
    torch.manual_seed(1)
    x, weight = _inputs(tokens=19, channels=channels)
    _assert_conv_matches(_decode_conv, x, weight)


def test_short_conv_decode_accepts_misaligned_contiguous_storage():
    """Materialize alignment inside the opaque launcher for off-boundary views."""
    x, weight = _misaligned_inputs()
    _assert_conv_matches(_decode_conv, x, weight)


def test_short_conv_decode_registered_custom_activation():
    """Fuse a user-registered CuTeDSL activation into the one-token step."""
    activations.register_activation("tanh", _tanh_activation, _tanh_activation_derivative)
    torch.manual_seed(7)
    x, weight = _inputs()
    actual = _decode_conv(x, weight, "tanh")
    expected = torch.tanh(_plain_conv_reference(x, weight))
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_short_conv_decode_activation_registration_contract():
    """Reject unknown activation names on the one-token path."""
    x, weight = (tensor.detach() for tensor in _inputs())
    with pytest.raises(ValueError, match="unknown activation"):
        causal_conv1d_decode(x[:, 0], weight, x.new_zeros(1, 3, x.shape[2]), activation="missing")


def test_short_conv_decode_fullgraph_forward(paged_short_conv_inputs):
    """Keep the opaque forward and the in-place history write in a strict graph."""
    torch.manual_seed(4)
    x, weight, state, slots = paged_short_conv_inputs()
    expected = causal_conv1d_decode(x, weight, state.clone(), state_indices=slots)
    compiled = torch.compile(causal_conv1d_decode, fullgraph=True)
    actual = compiled(x, weight, state, state_indices=slots)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_short_conv_decode_configured_fullgraph_forward(paged_short_conv_inputs):
    """Keep explicit scalar schedules behind the configured opaque operator."""
    torch.manual_seed(6)
    x, weight, state, slots = paged_short_conv_inputs(channels=6)
    config = ShortConvConfig(64, 2, 8)
    expected_state = state.clone()
    expected = causal_conv1d_decode(
        x,
        weight,
        expected_state,
        state_indices=slots,
        forward_config=config,
    )
    compiled = torch.compile(causal_conv1d_decode, fullgraph=True)
    actual = compiled(
        x,
        weight,
        state,
        state_indices=slots,
        forward_config=config,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_short_conv_decode_configured_dynamic_shapes(paged_short_conv_inputs):
    """Reuse one configured full graph across different sequence counts."""
    config = ShortConvConfig(64, 2, 8)
    compiled = torch.compile(causal_conv1d_decode, fullgraph=True, dynamic=True)
    for sequences in (2, 4):
        x, weight, state, slots = paged_short_conv_inputs(
            sequences=sequences,
            channels=6,
        )
        expected_state = state.clone()
        actual_state = state.clone()
        expected = causal_conv1d_decode(
            x,
            weight,
            expected_state,
            state_indices=slots,
            forward_config=config,
        )
        actual = compiled(
            x,
            weight,
            actual_state,
            state_indices=slots,
            forward_config=config,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


def test_short_conv_hopper_default_schedule_matches_reference(paged_short_conv_inputs):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper-specific decode schedule")
    x, weight, state, slots = paged_short_conv_inputs(sequences=64, channels=3072, slots=65)
    initial_state = state.clone()
    history = torch.cat([initial_state[slots.long()], x.unsqueeze(1)], dim=1)

    actual = causal_conv1d_decode(x, weight, state, activation="silu", state_indices=slots)

    torch.testing.assert_close(actual, _reference(history, weight)[:, -1], rtol=2e-2, atol=2e-2)


def test_short_conv_decode_advances_only_the_named_slots(paged_short_conv_inputs):
    """Address a paged pool by slot and leave the other rows untouched."""
    torch.manual_seed(1)
    x, weight, state, slots = paged_short_conv_inputs()
    initial_state = state.clone()
    history = torch.cat([initial_state[slots.long()], x.unsqueeze(1)], dim=1)
    expected_final = initial_state.clone()
    expected_final[slots.long()] = history[:, 1:]

    actual = causal_conv1d_decode(x, weight, state, activation="silu", state_indices=slots)

    torch.testing.assert_close(actual, _reference(history, weight)[:, -1], rtol=2e-2, atol=2e-2)
    # Absence of a write, not a numeric claim: unnamed rows must be bitwise intact.
    torch.testing.assert_close(state, expected_final, rtol=0, atol=0)


def test_short_conv_decode_advances_strided_state_slots(paged_short_conv_inputs):
    """Address slots in a page-padded state pool without touching the padding."""
    torch.manual_seed(2)
    x, weight, compact_state, slots = paged_short_conv_inputs()
    num_slots, state_rows, channels = compact_state.shape
    num_state_elements = state_rows * channels
    alignment_elements = 16 // compact_state.element_size()
    padding = alignment_elements - num_state_elements % alignment_elements
    storage = torch.randn(
        num_slots,
        num_state_elements + padding,
        device=compact_state.device,
        dtype=compact_state.dtype,
    )
    state = storage[:, :num_state_elements].view(num_slots, state_rows, channels)
    state.copy_(compact_state)
    initial_storage = storage.clone()
    history = torch.cat([state[slots.long()].clone(), x.unsqueeze(1)], dim=1)

    actual = causal_conv1d_decode(x, weight, state, activation="silu", state_indices=slots)

    expected_state = initial_storage[:, :num_state_elements].view_as(state)
    expected_state[slots.long()] = history[:, 1:]
    torch.testing.assert_close(actual, _reference(history, weight)[:, -1], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(
        storage[:, num_state_elements:],
        initial_storage[:, num_state_elements:],
        rtol=0,
        atol=0,
    )


def test_short_conv_decode_ignores_padding_slots(paged_short_conv_inputs):
    x, weight, state, slots = paged_short_conv_inputs()
    slots.copy_(torch.tensor([3, 0, -1], device="cuda", dtype=torch.int32))
    initial_state = state.clone()

    output = causal_conv1d_decode(x, weight, state, activation="silu", state_indices=slots)

    history = torch.cat([initial_state[3:4], x[:1].unsqueeze(1)], dim=1)
    torch.testing.assert_close(
        output[:1], _reference(history, weight)[:, -1], rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(output[1:], torch.zeros_like(output[1:]), rtol=0, atol=0)
    torch.testing.assert_close(state[0], initial_state[0], rtol=0, atol=0)


def test_short_conv_decode_custom_op_registration(paged_short_conv_inputs):
    """Exercise default and configured mutation schemas with fake and dynamic tensors."""
    x, weight, state, slots = paged_short_conv_inputs(channels=6)
    torch.library.opcheck(_decode_op, (x, weight, state.clone(), slots))
    torch.library.opcheck(
        _configured_decode_op,
        (x, weight, state.clone(), slots, 64, 2, 8),
    )


def test_short_conv_decode_validates_inputs_and_config(paged_short_conv_inputs):
    """Reject malformed tensor contracts and channel-incompatible schedules."""
    x, weight, state, slots = paged_short_conv_inputs(channels=6)
    with pytest.raises(ValueError, match="positive sequence and channel"):
        causal_conv1d_decode(x[:0], weight, state, state_indices=slots[:0])
    with pytest.raises(ValueError, match="weight must have shape"):
        causal_conv1d_decode(x, weight[:-1], state, state_indices=slots)
    with pytest.raises(ValueError, match="contiguous CUDA FP16, BF16, or FP32"):
        causal_conv1d_decode(x.double(), weight, state, state_indices=slots)
    with pytest.raises(ValueError, match="weight must match x dtype"):
        causal_conv1d_decode(x, weight.cpu(), state, state_indices=slots)
    with pytest.raises(ValueError, match="state must match x dtype"):
        causal_conv1d_decode(x, weight, state.cpu(), state_indices=slots)
    with pytest.raises(ValueError, match="C must be divisible"):
        causal_conv1d_decode(
            x,
            weight,
            state,
            state_indices=slots,
            forward_config=ShortConvConfig(64, 4, 8),
        )


def test_short_conv_decode_cuda_graph_replay(paged_short_conv_inputs):
    """Capture the one-token step and replay it from a reset history."""
    torch.manual_seed(8)
    x, weight, state, slots = paged_short_conv_inputs()
    initial_state = state.clone()
    causal_conv1d_decode(x, weight, state, activation="silu", state_indices=slots)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    state.copy_(initial_state)
    with torch.cuda.graph(graph):
        captured = causal_conv1d_decode(x, weight, state, activation="silu", state_indices=slots)

    # Replay advances the pool in place, so both runs must start from the same history.
    x.add_(0.25)
    state.copy_(initial_state)
    graph.replay()
    torch.cuda.synchronize()
    replayed, replayed_state = captured.clone(), state.clone()

    state.copy_(initial_state)
    expected = causal_conv1d_decode(x, weight, state, activation="silu", state_indices=slots)
    torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
    torch.testing.assert_close(replayed_state, state, rtol=0, atol=0)
