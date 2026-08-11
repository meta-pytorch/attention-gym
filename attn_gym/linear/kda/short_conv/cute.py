# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL causal depthwise convolution with a first-order backward.

The implementation is specialized for contiguous BF16 ``[1, T, C]`` tensors.
Each thread owns a compile-time number of adjacent channels. Forward stages its input window in
packed BF16 registers, while backward computes BF16 input gradients and FP32
weight-gradient partials followed by a Torch reduction.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import cutlass
import torch
from cutlass import BFloat16, Float32, cute

from attn_gym._backends.cute import ceildiv, compile_tvm_ffi, jit_cache, tune


@dataclass(frozen=True)
class ShortConvConfig:
    """Compile-time launch and register-tiling specialization."""

    threads: int
    channels_per_thread: int
    times_per_block: int


@dataclass(frozen=True)
class ShortConvTunedConfig:
    """Selected specializations for forward and both first-order gradients."""

    forward: ShortConvConfig
    input_gradient: ShortConvConfig
    weight_gradient: ShortConvConfig


class CausalConv1dSiluForward:
    """Compute causal depthwise convolution followed by SiLU."""

    # These schedule values are kernel specialization parameters. Register-tensor
    # shapes, vector partitioning, unrolled loops, and the block mapping all need
    # them while CuTeDSL compiles this kernel; they are not runtime operator state.
    default_config = ShortConvConfig(threads=128, channels_per_thread=4, times_per_block=8)

    def __init__(
        self,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
    ):
        self.tokens = tokens
        self.channels = channels
        self.width = width
        self.threads = config.threads
        self.channels_per_thread = config.channels_per_thread
        self.times_per_block = config.times_per_block

    def get_name(self) -> str:
        """Return the stable compiled-artifact name."""
        return (
            f"short_conv_fwd_bf16_t{self.tokens}_c{self.channels}_w{self.width}"
            f"_th{self.threads}_bt{self.times_per_block}_v{self.channels_per_thread}"
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, weight: cute.Tensor, output: cute.Tensor):
        """Compute one packed channel group and eight output tokens."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, _ = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread
        time_start = time_block * self.times_per_block

        if channel < self.channels:
            weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    weights[channel_offset, tap] = Float32(weight[channel + channel_offset, tap])

            x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
            output_groups = cute.zipped_divide(output, (1, self.channels_per_thread))
            inputs = cute.make_rmem_tensor(
                (self.channels_per_thread, self.times_per_block + self.width - 1),
                BFloat16,
            )
            inputs.fill(BFloat16(0.0))
            for input_offset in cutlass.range_constexpr(self.times_per_block + self.width - 1):
                input_time = time_start + input_offset - (self.width - 1)
                if input_time >= 0 and input_time < self.tokens:
                    inputs[(None, input_offset)].store(
                        x_groups[((0, None), (input_time, channel_group))].load()
                    )

            for time_offset in cutlass.range_constexpr(self.times_per_block):
                time = time_start + time_offset
                if time < self.tokens:
                    if cutlass.const_expr(self.width == 4):
                        # Keep the hot K3 specialization as one expression tree.
                        # CuTeDSL currently emits extra register moves for the
                        # generic TensorSSA reduction below at this width.
                        value = (
                            inputs[(None, time_offset + 3)].load().to(Float32)
                            * weights[(None, 3)].load()
                            + inputs[(None, time_offset + 2)].load().to(Float32)
                            * weights[(None, 2)].load()
                            + inputs[(None, time_offset + 1)].load().to(Float32)
                            * weights[(None, 1)].load()
                            + inputs[(None, time_offset)].load().to(Float32)
                            * weights[(None, 0)].load()
                        )
                    else:
                        products = cute.make_rmem_tensor(
                            (self.channels_per_thread, self.width), Float32
                        )
                        for tap in cutlass.range_constexpr(self.width):
                            products[(None, tap)].store(
                                inputs[(None, time_offset + tap)].load().to(Float32)
                                * weights[(None, tap)].load()
                            )
                        value = products.load().reduce(
                            cute.ReductionOp.ADD,
                            Float32(0.0),
                            reduction_profile=(None, 1),
                        )
                    half = value * 0.5
                    output_groups[((0, None), (time, channel_group))].store(
                        (half * cute.math.tanh(half, fastmath=True) + half).to(BFloat16)
                    )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        stream,
    ):
        """Launch the configured forward specialization."""
        self.kernel(x, weight, output, _name_prefix=self.get_name()).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.ceil_div(self.tokens, self.times_per_block),
                1,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluInputGradient:
    """Recompute the preactivation and compute the SiLU input gradient."""

    default_config = ShortConvConfig(threads=128, channels_per_thread=4, times_per_block=10)

    def __init__(
        self,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
    ):
        self.tokens = tokens
        self.channels = channels
        self.width = width
        self.threads = config.threads
        self.channels_per_thread = config.channels_per_thread
        self.times_per_block = config.times_per_block

    def get_name(self) -> str:
        """Return the stable compiled-artifact name."""
        return (
            f"short_conv_dx_bf16_t{self.tokens}_c{self.channels}_w{self.width}"
            f"_th{self.threads}_bt{self.times_per_block}_v{self.channels_per_thread}"
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        grad_x: cute.Tensor,
    ):
        """Compute one packed channel group and ten input-gradient tokens."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, _ = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread
        time_start = time_block * self.times_per_block

        if channel < self.channels:
            x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
            dy_groups = cute.zipped_divide(grad_output, (1, self.channels_per_thread))
            dx_groups = cute.zipped_divide(grad_x, (1, self.channels_per_thread))
            weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    weights[channel_offset, tap] = Float32(weight[channel + channel_offset, tap])

            inputs = cute.make_rmem_tensor(
                (self.channels_per_thread, self.times_per_block + 2 * (self.width - 1)),
                BFloat16,
            )
            inputs.fill(BFloat16(0.0))
            for input_offset in cutlass.range_constexpr(
                self.times_per_block + 2 * (self.width - 1)
            ):
                input_time = time_start + input_offset - (self.width - 1)
                if input_time >= 0 and input_time < self.tokens:
                    inputs[(None, input_offset)].store(
                        x_groups[((0, None), (input_time, channel_group))].load()
                    )

            grad_z = cute.make_rmem_tensor(
                (self.channels_per_thread, self.times_per_block + self.width - 1),
                Float32,
            )
            grad_z.fill(Float32(0.0))
            for output_offset in cutlass.range_constexpr(self.times_per_block + self.width - 1):
                output_time = time_start + output_offset
                if output_time < self.tokens:
                    products = cute.make_rmem_tensor(
                        (self.channels_per_thread, self.width), Float32
                    )
                    for tap in cutlass.range_constexpr(self.width):
                        products[(None, tap)].store(
                            inputs[(None, output_offset + tap)].load().to(Float32)
                            * weights[(None, tap)].load()
                        )
                    value = products.load().reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        reduction_profile=(None, 1),
                    )
                    half = value * 0.5
                    tanh_half = cute.math.tanh(half, fastmath=True)
                    derivative = (tanh_half + 1.0) * 0.5 + half * (
                        1.0 - tanh_half * tanh_half
                    ) * 0.5
                    incoming = (
                        dy_groups[((0, None), (output_time, channel_group))].load().to(Float32)
                    )
                    grad_z[(None, output_offset)].store(incoming * derivative)

            for time_offset in cutlass.range_constexpr(self.times_per_block):
                time = time_start + time_offset
                if time < self.tokens:
                    products = cute.make_rmem_tensor(
                        (self.channels_per_thread, self.width), Float32
                    )
                    for future_offset in cutlass.range_constexpr(self.width):
                        products[(None, future_offset)].store(
                            grad_z[(None, time_offset + future_offset)].load()
                            * weights[(None, self.width - 1 - future_offset)].load()
                        )
                    value = products.load().reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        reduction_profile=(None, 1),
                    )
                    dx_groups[((0, None), (time, channel_group))].store(value.to(BFloat16))

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        grad_x: cute.Tensor,
        stream,
    ):
        """Launch the configured input-gradient specialization."""
        self.kernel(
            x,
            weight,
            grad_output,
            grad_x,
            _name_prefix=self.get_name(),
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.ceil_div(self.tokens, self.times_per_block),
                1,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluWeightGradientPartials:
    """Compute FP32 time-tile partial sums for the weight gradient."""

    default_config = ShortConvConfig(threads=128, channels_per_thread=4, times_per_block=128)

    def __init__(
        self,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
    ):
        self.tokens = tokens
        self.channels = channels
        self.width = width
        self.threads = config.threads
        self.channels_per_thread = config.channels_per_thread
        self.times_per_block = config.times_per_block

    def get_name(self) -> str:
        """Return the stable compiled-artifact name."""
        return (
            f"short_conv_dw_bf16_t{self.tokens}_c{self.channels}_w{self.width}"
            f"_th{self.threads}_bt{self.times_per_block}_v{self.channels_per_thread}"
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        partials: cute.Tensor,
    ):
        """Compute one FP32 weight-gradient partial per tap and owned channel."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, _ = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread
        time_start = time_block * self.times_per_block

        if channel < self.channels:
            x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
            dy_groups = cute.zipped_divide(grad_output, (1, self.channels_per_thread))
            weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            accumulators = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            accumulators.fill(Float32(0.0))
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    weights[channel_offset, tap] = Float32(weight[channel + channel_offset, tap])

            for time_offset in cutlass.range(self.times_per_block, unroll_full=True):
                time = time_start + time_offset
                if time < self.tokens:
                    input_taps = cute.make_rmem_tensor(
                        (self.channels_per_thread, self.width), Float32
                    )
                    input_taps.fill(Float32(0.0))
                    for tap in cutlass.range_constexpr(self.width):
                        input_time = time + tap - (self.width - 1)
                        if input_time >= 0:
                            input_taps[(None, tap)].store(
                                x_groups[((0, None), (input_time, channel_group))]
                                .load()
                                .to(Float32)
                            )
                    products = input_taps.load() * weights.load()
                    value = products.reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        reduction_profile=(None, 1),
                    )
                    half = value * 0.5
                    tanh_half = cute.math.tanh(half, fastmath=True)
                    derivative = (tanh_half + 1.0) * 0.5 + half * (
                        1.0 - tanh_half * tanh_half
                    ) * 0.5
                    incoming = dy_groups[((0, None), (time, channel_group))].load().to(Float32)
                    grad_z = incoming * derivative
                    for tap in cutlass.range_constexpr(self.width):
                        accumulators[(None, tap)].store(
                            accumulators[(None, tap)].load()
                            + grad_z * input_taps[(None, tap)].load()
                        )

            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    partials[time_block, channel + channel_offset, tap] = accumulators[
                        channel_offset, tap
                    ]

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        partials: cute.Tensor,
        stream,
    ):
        """Launch the configured weight-gradient specialization."""
        self.kernel(
            x,
            weight,
            grad_output,
            partials,
            _name_prefix=self.get_name(),
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.ceil_div(self.tokens, self.times_per_block),
                1,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


# The TVM-FFI compact-tensor ABI currently specializes T and C through these
# fake shapes. The kernel's bounds arithmetic does not otherwise require them
# to be constexpr; a future dynamic-shape launcher could pass those extents at
# runtime without relaxing the schedule parameters above.
def _fake_bf16_matrix(rows: int, columns: int):
    """Create a row-major BF16 fake tensor for compilation."""
    return cute.runtime.make_fake_compact_tensor(
        BFloat16,
        (rows, columns),
        stride_order=(1, 0),
        assumed_align=16,
    )


@jit_cache
def _compile_forward(tokens: int, channels: int, width: int, config: ShortConvConfig):
    """Compile one static forward specialization."""
    operation = CausalConv1dSiluForward(tokens, channels, width, config)
    return compile_tvm_ffi(
        operation,
        _fake_bf16_matrix(tokens, channels),
        _fake_bf16_matrix(channels, width),
        _fake_bf16_matrix(tokens, channels),
    )


@jit_cache
def _compile_input_gradient(
    tokens: int,
    channels: int,
    width: int,
    config: ShortConvConfig,
):
    """Compile one static input-gradient specialization."""
    operation = CausalConv1dSiluInputGradient(tokens, channels, width, config)
    return compile_tvm_ffi(
        operation,
        _fake_bf16_matrix(tokens, channels),
        _fake_bf16_matrix(channels, width),
        _fake_bf16_matrix(tokens, channels),
        _fake_bf16_matrix(tokens, channels),
    )


@jit_cache
def _compile_weight_gradient(
    tokens: int,
    channels: int,
    width: int,
    config: ShortConvConfig,
):
    """Compile one static weight-gradient specialization."""
    num_time_blocks = ceildiv(tokens, config.times_per_block)
    partials = cute.runtime.make_fake_compact_tensor(
        Float32,
        (num_time_blocks, channels, width),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    operation = CausalConv1dSiluWeightGradientPartials(tokens, channels, width, config)
    return compile_tvm_ffi(
        operation,
        _fake_bf16_matrix(tokens, channels),
        _fake_bf16_matrix(channels, width),
        _fake_bf16_matrix(tokens, channels),
        partials,
    )


def _validate_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    """Validate the public compile-time-width CuTeDSL tensor contract."""
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(f"x must have shape [1, T, C], got {tuple(x.shape)}")
    if x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(f"x must have positive T and C dimensions, got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA BF16 tensor")
    if weight.ndim != 2 or weight.shape[0] != x.shape[2] or weight.shape[1] < 1:
        raise ValueError(
            f"weight must have shape [C, W] with W positive, got {tuple(weight.shape)}"
        )
    if weight.dtype != torch.bfloat16 or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("weight must be contiguous CUDA BF16 on x.device")


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize the uncommon contiguous view that misses the launcher ABI alignment."""
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _launch_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: ShortConvConfig,
) -> torch.Tensor:
    """Allocate and launch the compiled forward specialization."""
    x, weight = _aligned(x), _aligned(weight)
    tokens, channels = x.shape[1:]
    width = weight.shape[1]
    output = torch.empty_like(x)
    compiled = _compile_forward(
        tokens,
        channels,
        width,
        config,
    )
    compiled(x.view(tokens, channels), weight, output.view(tokens, channels))
    return output


def _launch_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    input_config: ShortConvConfig,
    weight_config: ShortConvConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch configured gradient kernels and reduce FP32 partials."""
    x, weight = _aligned(x), _aligned(weight)
    tokens, channels = x.shape[1:]
    width = weight.shape[1]
    grad_output = _aligned(grad_output.contiguous())
    grad_x = torch.empty_like(x)
    _compile_input_gradient(
        tokens,
        channels,
        width,
        input_config,
    )(
        x.view(tokens, channels),
        weight,
        grad_output.view(tokens, channels),
        grad_x.view(tokens, channels),
    )

    num_time_blocks = ceildiv(tokens, weight_config.times_per_block)
    partials = torch.empty(
        (num_time_blocks, channels, width),
        dtype=torch.float32,
        device=x.device,
    )
    _compile_weight_gradient(tokens, channels, width, weight_config)(
        x.view(tokens, channels),
        weight,
        grad_output.view(tokens, channels),
        partials,
    )
    return grad_x, partials.sum(dim=0).to(torch.bfloat16)


def _candidate_configs(kind: str, channels: int) -> tuple[ShortConvConfig, ...]:
    """Return the focused schedule space used by the explicit tuning flow."""
    if kind == "forward":
        candidates = (
            ShortConvConfig(64, 4, 8),
            ShortConvConfig(128, 2, 8),
            ShortConvConfig(128, 4, 4),
            ShortConvConfig(128, 4, 8),
            ShortConvConfig(128, 4, 16),
            ShortConvConfig(256, 4, 8),
        )
    elif kind == "input_gradient":
        candidates = (
            ShortConvConfig(64, 4, 8),
            ShortConvConfig(128, 2, 8),
            ShortConvConfig(128, 4, 8),
            ShortConvConfig(128, 4, 10),
            ShortConvConfig(128, 4, 12),
            ShortConvConfig(256, 4, 8),
        )
    elif kind == "weight_gradient":
        candidates = (
            ShortConvConfig(64, 4, 128),
            ShortConvConfig(128, 2, 64),
            ShortConvConfig(128, 4, 32),
            ShortConvConfig(128, 4, 64),
            ShortConvConfig(128, 4, 128),
            ShortConvConfig(256, 4, 128),
        )
    else:  # pragma: no cover - internal callers use the literals above.
        raise ValueError(f"unknown short-convolution kernel kind {kind!r}")
    return tuple(config for config in candidates if channels % config.channels_per_thread == 0)


def tune_causal_conv1d_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    forward_configs: Iterable[ShortConvConfig] | None = None,
    input_grad_configs: Iterable[ShortConvConfig] | None = None,
    weight_grad_configs: Iterable[ShortConvConfig] | None = None,
    parallel_compile: bool = True,
) -> ShortConvTunedConfig:
    """Compile and benchmark forward, input-gradient, and weight-gradient schedules.

    Width is an operation parameter and is specialized but not autotuned. The
    returned configs can be passed directly to :func:`cute_causal_conv1d_silu`.
    Tuning uses the shared CuTeDSL ``tune`` flow: cached variants compile in
    parallel and execute sequentially under the vetted Inductor GPU benchmarker.
    """
    _validate_inputs(x, weight)
    if grad_output.shape != x.shape or grad_output.dtype != x.dtype:
        raise ValueError("grad_output must match x shape and dtype")
    if grad_output.device != x.device or not grad_output.is_contiguous():
        raise ValueError("grad_output must be contiguous on x.device")

    x, weight, grad_output = _aligned(x), _aligned(weight), _aligned(grad_output)
    tokens, channels = x.shape[1:]
    width = weight.shape[1]
    x_matrix = x.view(tokens, channels)
    grad_matrix = grad_output.view(tokens, channels)

    forward_candidates = tuple(
        _candidate_configs("forward", channels) if forward_configs is None else forward_configs
    )
    for config in forward_candidates:
        _validate_config(config, channels, "forward_configs")
    forward_output = torch.empty_like(x).view(tokens, channels)
    forward = tune(
        forward_candidates,
        _compile_forward,
        lambda compiled, _config: compiled(x_matrix, weight, forward_output),
        compile_call=lambda config: (tokens, channels, width, config),
        parallel_compile=parallel_compile,
    )

    input_candidates = tuple(
        _candidate_configs("input_gradient", channels)
        if input_grad_configs is None
        else input_grad_configs
    )
    for config in input_candidates:
        _validate_config(config, channels, "input_grad_configs")
    grad_x = torch.empty_like(x).view(tokens, channels)
    input_gradient = tune(
        input_candidates,
        _compile_input_gradient,
        lambda compiled, _config: compiled(x_matrix, weight, grad_matrix, grad_x),
        compile_call=lambda config: (tokens, channels, width, config),
        parallel_compile=parallel_compile,
    )

    weight_candidates = tuple(
        _candidate_configs("weight_gradient", channels)
        if weight_grad_configs is None
        else weight_grad_configs
    )
    for config in weight_candidates:
        _validate_config(config, channels, "weight_grad_configs")
    partials = {
        config: torch.empty(
            ceildiv(tokens, config.times_per_block),
            channels,
            width,
            dtype=torch.float32,
            device=x.device,
        )
        for config in weight_candidates
    }
    weight_gradient = tune(
        weight_candidates,
        _compile_weight_gradient,
        lambda compiled, config: compiled(
            x_matrix,
            weight,
            grad_matrix,
            partials[config],
        ),
        compile_call=lambda config: (tokens, channels, width, config),
        parallel_compile=parallel_compile,
    )
    return ShortConvTunedConfig(forward, input_gradient, weight_gradient)


def _config(threads: int, channels_per_thread: int, times_per_block: int) -> ShortConvConfig:
    """Reconstruct a compile-time config from custom-op scalar arguments."""
    return ShortConvConfig(threads, channels_per_thread, times_per_block)


@torch.library.custom_op("attn_gym::_cute_short_conv_fwd", mutates_args=())
def _forward_custom_op(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Launch the tuned K3 defaults through the lean production schema."""
    return _launch_forward(x, weight, CausalConv1dSiluForward.default_config)


@_forward_custom_op.register_fake
def _default_forward_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    del weight
    return torch.empty_like(x)


@torch.library.custom_op("attn_gym::_cute_short_conv_bwd", mutates_args=())
def _backward_custom_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the tuned K3 backward defaults through the lean schema."""
    return _launch_backward(
        x,
        weight,
        grad_output,
        CausalConv1dSiluInputGradient.default_config,
        CausalConv1dSiluWeightGradientPartials.default_config,
    )


@_backward_custom_op.register_fake
def _default_backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_output
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_default_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


@torch.autograd.function.once_differentiable
def _default_backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, weight = ctx.saved_tensors
    return _backward_custom_op(x, weight, grad_output)


_forward_custom_op.register_autograd(
    _default_backward,
    setup_context=_setup_default_context,
)


@torch.library.custom_op("attn_gym::_cute_short_conv_configured_fwd", mutates_args=())
def _configured_forward_custom_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_threads: int,
    forward_channels: int,
    forward_times: int,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
) -> torch.Tensor:
    """Keep the configured CuTeDSL forward launcher behind an opaque operator."""
    del input_threads, input_channels, input_times, weight_threads, weight_channels, weight_times
    return _launch_forward(
        x,
        weight,
        _config(forward_threads, forward_channels, forward_times),
    )


@_configured_forward_custom_op.register_fake
def _forward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    forward_threads: int,
    forward_channels: int,
    forward_times: int,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
) -> torch.Tensor:
    """Describe forward output metadata without invoking the compiler."""
    del (
        weight,
        forward_threads,
        forward_channels,
        forward_times,
        input_threads,
        input_channels,
        input_times,
        weight_threads,
        weight_channels,
        weight_times,
    )
    return torch.empty_like(x)


@torch.library.custom_op("attn_gym::_cute_short_conv_configured_bwd", mutates_args=())
def _configured_backward_custom_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep both configured first-order backward launchers opaque."""
    return _launch_backward(
        x,
        weight,
        grad_output,
        _config(input_threads, input_channels, input_times),
        _config(weight_threads, weight_channels, weight_times),
    )


@_configured_backward_custom_op.register_fake
def _backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe backward output metadata without invoking the compiler."""
    del (
        grad_output,
        input_threads,
        input_channels,
        input_times,
        weight_threads,
        weight_channels,
        weight_times,
    )
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_context(ctx, inputs, output) -> None:
    """Save inputs and backward specializations for preactivation recomputation."""
    del output
    (
        x,
        weight,
        _forward_threads,
        _forward_channels,
        _forward_times,
        ctx.input_threads,
        ctx.input_channels,
        ctx.input_times,
        ctx.weight_threads,
        ctx.weight_channels,
        ctx.weight_times,
    ) = inputs
    ctx.save_for_backward(x, weight)


@torch.autograd.function.once_differentiable
def _backward(ctx, grad_output: torch.Tensor):
    """Dispatch the registered first-order backward custom operator."""
    x, weight = ctx.saved_tensors
    grad_x, grad_weight = _configured_backward_custom_op(
        x,
        weight,
        grad_output,
        ctx.input_threads,
        ctx.input_channels,
        ctx.input_times,
        ctx.weight_threads,
        ctx.weight_channels,
        ctx.weight_times,
    )
    return grad_x, grad_weight, None, None, None, None, None, None, None, None, None


_configured_forward_custom_op.register_autograd(_backward, setup_context=_setup_context)


def _validate_config(config: ShortConvConfig, channels: int, name: str) -> None:
    """Reject launch configurations that cannot form safe packed channel groups."""
    if config.threads < 32 or config.threads > 1024 or config.threads % 32 != 0:
        raise ValueError(f"{name}.threads must be a warp multiple in [32, 1024]")
    if config.channels_per_thread < 1 or channels % config.channels_per_thread != 0:
        raise ValueError(f"C must be divisible by positive {name}.channels_per_thread")
    if config.times_per_block < 1:
        raise ValueError(f"{name}.times_per_block must be positive")


def cute_causal_conv1d_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    forward_config: ShortConvConfig | None = None,
    input_grad_config: ShortConvConfig | None = None,
    weight_grad_config: ShortConvConfig | None = None,
) -> torch.Tensor:
    """Apply causal depthwise convolution and SiLU with CuTeDSL.

    Width is inferred from ``weight.shape[1]`` and compile-time specialized.
    Schedule fields control register shapes, vector partitioning, unrolled loops,
    and block mapping, so changing one compiles and caches a distinct kernel.

    Args:
        x: Contiguous CUDA BF16 input with shape ``[1, T, C]``.
        weight: Contiguous CUDA BF16 depthwise weights with shape ``[C, W]``.
        forward_config: Optional forward schedule specialization.
        input_grad_config: Optional input-gradient schedule specialization.
        weight_grad_config: Optional weight-gradient schedule specialization.

    Returns:
        A contiguous CUDA BF16 tensor with the same shape as ``x``.
    """
    _validate_inputs(x, weight)
    if forward_config is None and input_grad_config is None and weight_grad_config is None:
        for name, config in (
            ("forward_config", CausalConv1dSiluForward.default_config),
            ("input_grad_config", CausalConv1dSiluInputGradient.default_config),
            ("weight_grad_config", CausalConv1dSiluWeightGradientPartials.default_config),
        ):
            _validate_config(config, x.shape[2], name)
        return _forward_custom_op(x, weight)

    forward = forward_config or CausalConv1dSiluForward.default_config
    input_grad = input_grad_config or CausalConv1dSiluInputGradient.default_config
    weight_grad = weight_grad_config or CausalConv1dSiluWeightGradientPartials.default_config
    for name, config in (
        ("forward_config", forward),
        ("input_grad_config", input_grad),
        ("weight_grad_config", weight_grad),
    ):
        _validate_config(config, x.shape[2], name)
    return _configured_forward_custom_op(
        x,
        weight,
        forward.threads,
        forward.channels_per_thread,
        forward.times_per_block,
        input_grad.threads,
        input_grad.channels_per_thread,
        input_grad.times_per_block,
        weight_grad.threads,
        weight_grad.channels_per_thread,
        weight_grad.times_per_block,
    )


__all__ = [
    "ShortConvConfig",
    "ShortConvTunedConfig",
    "cute_causal_conv1d_silu",
    "tune_causal_conv1d_silu",
]
