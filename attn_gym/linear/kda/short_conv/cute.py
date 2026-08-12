# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL causal depthwise convolution with a first-order backward.

The implementation accepts contiguous FP16, BF16, or FP32 ``[B, T, C]`` tensors and
treats every batch row as an independent sequence. Dense inputs may supply their
preceding ``W - 1`` input positions as functional state. An optional CUDA
``cu_seqlens`` tensor instead delimits independent sequences packed into ``[1, T, C]``
without requiring a maximum sequence length or auxiliary schedule. Each thread owns a
compile-time number of adjacent channels. Forward stages its input window in registers,
while backward computes input gradients and FP32 weight-gradient partials followed by
a Torch reduction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import cutlass
import torch
from cutlass import BFloat16, Float16, Float32, Int32, cute

from attn_gym._backends.cute import ceildiv, compile_tvm_ffi, jit_cache, tune
from attn_gym._backends.cute.device import upper_bound


@dataclass(frozen=True)
class ShortConvDType:
    """Map a Torch storage dtype to its compile-time CuTeDSL type and artifact tag."""

    cute_type: type[cutlass.Numeric]
    name: str


SHORT_CONV_DTYPES = {
    torch.float16: ShortConvDType(Float16, "fp16"),
    torch.bfloat16: ShortConvDType(BFloat16, "bf16"),
    torch.float32: ShortConvDType(Float32, "fp32"),
}


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

    @classmethod
    def default(cls, dtype: torch.dtype = torch.bfloat16) -> ShortConvTunedConfig:
        """Return measured GB300 defaults for one storage dtype."""
        if dtype == torch.float16:
            return cls(
                ShortConvConfig(128, 4, 16),
                ShortConvConfig(128, 4, 6),
                ShortConvConfig(128, 4, 128),
            )
        if dtype == torch.float32:
            return cls(
                ShortConvConfig(128, 4, 4),
                ShortConvConfig(128, 2, 12),
                ShortConvConfig(128, 4, 32),
            )
        if dtype != torch.bfloat16:
            raise ValueError(f"unsupported short-convolution dtype {dtype}")
        return cls(
            ShortConvConfig(128, 4, 16),
            ShortConvConfig(128, 4, 10),
            ShortConvConfig(128, 4, 128),
        )


@cute.jit
def _silu(value):
    """Apply SiLU to an FP32 register tensor."""
    half = value * 0.5
    return half * cute.math.tanh(half, fastmath=True) + half


@cute.jit
def _silu_derivative(value):
    """Evaluate the SiLU derivative from its preactivation."""
    half = value * 0.5
    tanh_half = cute.math.tanh(half, fastmath=True)
    return (tanh_half + 1.0) * 0.5 + half * (1.0 - tanh_half * tanh_half) * 0.5


@cute.jit
def sequence_bounds(cu_seqlens: cute.Tensor, time: Int32):
    """Find the packed sequence containing a physical token position."""
    if cutlass.const_expr(cute.size(cu_seqlens) == 2):
        sequence = Int32(0)
        sequence_start = Int32(cu_seqlens[0])
        sequence_end = Int32(cu_seqlens[1])
    else:
        num_offsets = Int32(cute.size(cu_seqlens))
        sequence = upper_bound(cu_seqlens, time, Int32(1), num_offsets) - 1
        if sequence >= num_offsets - 1:
            sequence = num_offsets - 2
        sequence_start = Int32(cu_seqlens[sequence])
        sequence_end = Int32(cu_seqlens[sequence + 1])
    if sequence_start < 0:
        sequence_start = Int32(0)
    return sequence, sequence_start, sequence_end


@cute.jit
def tile_sequence_bounds(
    cu_seqlens: cute.Tensor | None,
    time: Int32,
    batch: Int32,
    tokens: cutlass.Constexpr,
):
    """Initialize sequence metadata for a dense or packed physical tile."""
    if cutlass.const_expr(cu_seqlens is None):
        return batch, Int32(0), Int32(tokens)
    return sequence_bounds(cu_seqlens, time)


@cute.jit
def advance_sequence_bounds(
    cu_seqlens: cute.Tensor,
    sequence: Int32,
    sequence_start: Int32,
    sequence_end: Int32,
    time: Int32,
):
    """Advance packed sequence metadata when time crosses a boundary."""
    if time >= sequence_end:
        sequence, sequence_start, sequence_end = sequence_bounds(cu_seqlens, time)
    return sequence, sequence_start, sequence_end


@cute.jit
def unrolled_dot(
    inputs: cute.Tensor,
    weights: cute.Tensor,
    input_offset: cutlass.Constexpr,
    width: cutlass.Constexpr,
):
    """Build a descending compile-time convolution dot product."""
    last_tap = width - 1
    value = (
        inputs[(None, input_offset + last_tap)].load().to(Float32)
        * weights[(None, last_tap)].load()
    )
    for step in cutlass.range_constexpr(width - 1):
        tap = width - 2 - step
        value = (
            value
            + inputs[(None, input_offset + tap)].load().to(Float32) * weights[(None, tap)].load()
        )
    return value


@cute.jit
def load_history(
    initial_state: cute.Tensor,
    sequence: Int32,
    input_time: Int32,
    sequence_start: Int32,
    channel_group: Int32,
    width: cutlass.Constexpr,
):
    """Load one vector from a sequence's caller-provided causal history."""
    state_offset = width - 1 + input_time - sequence_start
    return (
        initial_state[
            (
                (0, None),
                (sequence * (width - 1) + state_offset, channel_group),
            )
        ]
        .load()
        .to(Float32)
    )


@cute.jit
def history_dot(
    inputs: cute.Tensor,
    weights: cute.Tensor,
    initial_state: cute.Tensor,
    sequence: Int32,
    output_time: Int32,
    sequence_start: Int32,
    input_offset: cutlass.Constexpr,
    channel_group: Int32,
    channels_per_thread: cutlass.Constexpr,
    width: cutlass.Constexpr,
):
    """Evaluate an early convolution dot product from staged input and history."""
    products = cute.make_rmem_tensor((channels_per_thread, width), Float32)
    for tap in cutlass.range_constexpr(width):
        input_time = output_time + tap - (width - 1)
        input_value = cute.make_rmem_tensor((channels_per_thread,), Float32)
        if input_time >= sequence_start:
            input_value.store(inputs[(None, input_offset + tap)].load().to(Float32))
        else:
            input_value.store(
                load_history(
                    initial_state,
                    sequence,
                    input_time,
                    sequence_start,
                    channel_group,
                    width,
                )
            )
        products[(None, tap)].store(input_value.load() * weights[(None, tap)].load())
    return products.load().reduce(
        cute.ReductionOp.ADD,
        Float32(0.0),
        reduction_profile=(None, 1),
    )


class ShortConvKernel:
    """Own the static problem shape, schedule, and artifact naming shared by each kernel."""

    kernel_kind: ClassVar[str]
    sequence_axis: ClassVar[str] = "b"
    time_tiled: ClassVar[bool] = True

    def __init__(
        self,
        sequences: int,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
        dtype: ShortConvDType,
    ):
        self.sequences = sequences
        self.tokens = tokens
        self.channels = channels
        self.width = width
        self.threads = config.threads
        self.channels_per_thread = config.channels_per_thread
        self.times_per_block = config.times_per_block
        self.dtype = dtype

    def get_name(self) -> str:
        """Return the stable compiled-artifact name."""
        name = (
            f"short_conv_{self.kernel_kind}_{self.dtype.name}_{self.sequence_axis}{self.sequences}"
            f"_t{self.tokens}_c{self.channels}_w{self.width}_th{self.threads}"
        )
        if self.time_tiled:
            name += f"_bt{self.times_per_block}"
        return f"{name}_v{self.channels_per_thread}"


class CausalConv1dSiluForward(ShortConvKernel):
    """Compute causal depthwise convolution followed by a compile-time activation."""

    kernel_kind = "fwd"

    def __init__(
        self,
        batches: int,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
        dtype: ShortConvDType,
        activation,
    ):
        super().__init__(batches, tokens, channels, width, config, dtype)
        self.batches = batches
        self.activation = activation

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
    ):
        """Compute one packed channel group and eight output tokens."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, batch = cute.arch.block_idx()
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
            if cutlass.const_expr(initial_state is not None):
                initial_groups = cute.zipped_divide(initial_state, (1, self.channels_per_thread))
            inputs = cute.make_rmem_tensor(
                (self.channels_per_thread, self.times_per_block + self.width - 1),
                self.dtype.cute_type,
            )
            inputs.fill(self.dtype.cute_type(0.0))
            for input_offset in cutlass.range_constexpr(self.times_per_block + self.width - 1):
                input_time = time_start + input_offset - (self.width - 1)
                if input_time >= 0 and input_time < self.tokens:
                    inputs[(None, input_offset)].store(
                        x_groups[
                            ((0, None), (batch * self.tokens + input_time, channel_group))
                        ].load()
                    )

            tile_sequence, tile_sequence_start, tile_sequence_end = tile_sequence_bounds(
                cu_seqlens,
                Int32(time_start),
                Int32(batch),
                self.tokens,
            )

            for time_offset in cutlass.range_constexpr(self.times_per_block):
                time = time_start + time_offset
                sequence = tile_sequence
                sequence_start = tile_sequence_start
                if time < self.tokens:
                    if cutlass.const_expr(cu_seqlens is not None):
                        sequence, sequence_start, _ = advance_sequence_bounds(
                            cu_seqlens,
                            tile_sequence,
                            tile_sequence_start,
                            tile_sequence_end,
                            Int32(time),
                        )
                    else:
                        sequence_start = Int32(0)

                    if cutlass.const_expr(initial_state is not None):
                        value = unrolled_dot(inputs, weights, time_offset, self.width)
                        if time - (self.width - 1) < sequence_start:
                            value = history_dot(
                                inputs,
                                weights,
                                initial_groups,
                                sequence,
                                Int32(time),
                                sequence_start,
                                time_offset,
                                channel_group,
                                self.channels_per_thread,
                                self.width,
                            )
                    elif cutlass.const_expr(cu_seqlens is None):
                        value = unrolled_dot(inputs, weights, time_offset, self.width)
                    else:
                        last_tap = self.width - 1
                        value = (
                            inputs[(None, time_offset + last_tap)].load().to(Float32)
                            * weights[(None, last_tap)].load()
                        )
                        if time - last_tap >= sequence_start:
                            value = unrolled_dot(inputs, weights, time_offset, self.width)
                        else:
                            for step in cutlass.range_constexpr(self.width - 1):
                                tap = self.width - 2 - step
                                if time + tap - last_tap >= sequence_start:
                                    value = (
                                        value
                                        + inputs[(None, time_offset + tap)].load().to(Float32)
                                        * weights[(None, tap)].load()
                                    )
                    output_groups[((0, None), (batch * self.tokens + time, channel_group))].store(
                        self.activation(value).to(self.dtype.cute_type)
                    )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        stream,
    ):
        """Launch the configured forward specialization."""
        self.kernel(
            x,
            weight,
            output,
            cu_seqlens,
            initial_state,
            _name_prefix=self.get_name(),
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.ceil_div(self.tokens, self.times_per_block),
                self.batches,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluInputGradient(ShortConvKernel):
    """Recompute the preactivation and apply a compile-time activation derivative."""

    kernel_kind = "dx"

    def __init__(
        self,
        batches: int,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
        dtype: ShortConvDType,
        d_activation,
    ):
        super().__init__(batches, tokens, channels, width, config, dtype)
        self.batches = batches
        self.d_activation = d_activation

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        grad_x: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
    ):
        """Compute one packed channel group and ten input-gradient tokens."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, batch = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread
        time_start = time_block * self.times_per_block

        if channel < self.channels:
            x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
            dy_groups = cute.zipped_divide(grad_output, (1, self.channels_per_thread))
            dx_groups = cute.zipped_divide(grad_x, (1, self.channels_per_thread))
            if cutlass.const_expr(initial_state is not None):
                initial_groups = cute.zipped_divide(initial_state, (1, self.channels_per_thread))
            weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    weights[channel_offset, tap] = Float32(weight[channel + channel_offset, tap])

            inputs = cute.make_rmem_tensor(
                (self.channels_per_thread, self.times_per_block + 2 * (self.width - 1)),
                self.dtype.cute_type,
            )
            inputs.fill(self.dtype.cute_type(0.0))
            for input_offset in cutlass.range_constexpr(
                self.times_per_block + 2 * (self.width - 1)
            ):
                input_time = time_start + input_offset - (self.width - 1)
                if input_time >= 0 and input_time < self.tokens:
                    inputs[(None, input_offset)].store(
                        x_groups[
                            ((0, None), (batch * self.tokens + input_time, channel_group))
                        ].load()
                    )

            grad_z = cute.make_rmem_tensor(
                (self.channels_per_thread, self.times_per_block + self.width - 1),
                Float32,
            )
            grad_z.fill(Float32(0.0))
            output_sequence, output_sequence_start, output_sequence_end = tile_sequence_bounds(
                cu_seqlens,
                Int32(time_start),
                Int32(batch),
                self.tokens,
            )
            for output_offset in cutlass.range_constexpr(self.times_per_block + self.width - 1):
                output_time = time_start + output_offset
                if output_time < self.tokens:
                    if cutlass.const_expr(cu_seqlens is not None):
                        (
                            output_sequence,
                            output_sequence_start,
                            output_sequence_end,
                        ) = advance_sequence_bounds(
                            cu_seqlens,
                            output_sequence,
                            output_sequence_start,
                            output_sequence_end,
                            Int32(output_time),
                        )
                    if cutlass.const_expr(initial_state is not None):
                        value = unrolled_dot(inputs, weights, output_offset, self.width)
                        if output_time - (self.width - 1) < output_sequence_start:
                            value = history_dot(
                                inputs,
                                weights,
                                initial_groups,
                                output_sequence,
                                Int32(output_time),
                                output_sequence_start,
                                output_offset,
                                channel_group,
                                self.channels_per_thread,
                                self.width,
                            )
                    else:
                        products = cute.make_rmem_tensor(
                            (self.channels_per_thread, self.width), Float32
                        )
                        products.fill(Float32(0.0))
                        for tap in cutlass.range_constexpr(self.width):
                            input_time = output_time + tap - (self.width - 1)
                            if (
                                cutlass.const_expr(cu_seqlens is None)
                                or input_time >= output_sequence_start
                            ):
                                products[(None, tap)].store(
                                    inputs[(None, output_offset + tap)].load().to(Float32)
                                    * weights[(None, tap)].load()
                                )
                        value = products.load().reduce(
                            cute.ReductionOp.ADD,
                            Float32(0.0),
                            reduction_profile=(None, 1),
                        )
                    derivative = self.d_activation(value)
                    incoming = (
                        dy_groups[((0, None), (batch * self.tokens + output_time, channel_group))]
                        .load()
                        .to(Float32)
                    )
                    grad_z[(None, output_offset)].store(incoming * derivative)

            input_sequence, input_sequence_start, input_sequence_end = tile_sequence_bounds(
                cu_seqlens,
                Int32(time_start),
                Int32(batch),
                self.tokens,
            )
            for time_offset in cutlass.range_constexpr(self.times_per_block):
                time = time_start + time_offset
                if time < self.tokens:
                    if cutlass.const_expr(cu_seqlens is not None):
                        (
                            input_sequence,
                            input_sequence_start,
                            input_sequence_end,
                        ) = advance_sequence_bounds(
                            cu_seqlens,
                            input_sequence,
                            input_sequence_start,
                            input_sequence_end,
                            Int32(time),
                        )
                    products = cute.make_rmem_tensor(
                        (self.channels_per_thread, self.width), Float32
                    )
                    products.fill(Float32(0.0))
                    for future_offset in cutlass.range_constexpr(self.width):
                        if cutlass.const_expr(cu_seqlens is None):
                            products[(None, future_offset)].store(
                                grad_z[(None, time_offset + future_offset)].load()
                                * weights[(None, self.width - 1 - future_offset)].load()
                            )
                        else:
                            if time + future_offset < input_sequence_end:
                                products[(None, future_offset)].store(
                                    grad_z[(None, time_offset + future_offset)].load()
                                    * weights[(None, self.width - 1 - future_offset)].load()
                                )
                    value = products.load().reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        reduction_profile=(None, 1),
                    )
                    dx_groups[((0, None), (batch * self.tokens + time, channel_group))].store(
                        value.to(self.dtype.cute_type)
                    )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        grad_x: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        stream,
    ):
        """Launch the configured input-gradient specialization."""
        self.kernel(
            x,
            weight,
            grad_output,
            grad_x,
            cu_seqlens,
            initial_state,
            _name_prefix=self.get_name(),
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.ceil_div(self.tokens, self.times_per_block),
                self.batches,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluWeightGradientPartials(ShortConvKernel):
    """Compute FP32 batch/time-tile partial sums for the weight gradient."""

    kernel_kind = "dw"

    def __init__(
        self,
        batches: int,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
        dtype: ShortConvDType,
        d_activation,
    ):
        super().__init__(batches, tokens, channels, width, config, dtype)
        self.batches = batches
        self.d_activation = d_activation

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        partials: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
    ):
        """Compute one FP32 weight-gradient partial per tap and owned channel."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, batch = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread
        time_start = time_block * self.times_per_block

        if channel < self.channels:
            x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
            dy_groups = cute.zipped_divide(grad_output, (1, self.channels_per_thread))
            if cutlass.const_expr(initial_state is not None):
                initial_groups = cute.zipped_divide(initial_state, (1, self.channels_per_thread))
            weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            accumulators = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            accumulators.fill(Float32(0.0))
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    weights[channel_offset, tap] = Float32(weight[channel + channel_offset, tap])

            sequence, sequence_start, sequence_end = tile_sequence_bounds(
                cu_seqlens,
                Int32(time_start),
                Int32(batch),
                self.tokens,
            )
            for time_offset in cutlass.range(self.times_per_block, unroll_full=True):
                time = time_start + time_offset
                if time < self.tokens:
                    if cutlass.const_expr(cu_seqlens is not None):
                        sequence, sequence_start, sequence_end = advance_sequence_bounds(
                            cu_seqlens,
                            sequence,
                            sequence_start,
                            sequence_end,
                            Int32(time),
                        )
                    input_taps = cute.make_rmem_tensor(
                        (self.channels_per_thread, self.width), Float32
                    )
                    input_taps.fill(Float32(0.0))
                    for tap in cutlass.range_constexpr(self.width):
                        input_time = time + tap - (self.width - 1)
                        if input_time >= sequence_start:
                            input_taps[(None, tap)].store(
                                x_groups[
                                    (
                                        (0, None),
                                        (batch * self.tokens + input_time, channel_group),
                                    )
                                ]
                                .load()
                                .to(Float32)
                            )
                        elif cutlass.const_expr(initial_state is not None):
                            input_taps[(None, tap)].store(
                                load_history(
                                    initial_groups,
                                    sequence,
                                    input_time,
                                    sequence_start,
                                    channel_group,
                                    self.width,
                                )
                            )
                    value = (input_taps.load() * weights.load()).reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        reduction_profile=(None, 1),
                    )
                    derivative = self.d_activation(value)
                    incoming = (
                        dy_groups[((0, None), (batch * self.tokens + time, channel_group))]
                        .load()
                        .to(Float32)
                    )
                    grad_z = incoming * derivative
                    for tap in cutlass.range_constexpr(self.width):
                        accumulators[(None, tap)].store(
                            accumulators[(None, tap)].load()
                            + grad_z * input_taps[(None, tap)].load()
                        )

            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    partials[batch, time_block, channel + channel_offset, tap] = accumulators[
                        channel_offset, tap
                    ]

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        partials: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        stream,
    ):
        """Launch the configured weight-gradient specialization."""
        self.kernel(
            x,
            weight,
            grad_output,
            partials,
            cu_seqlens,
            initial_state,
            _name_prefix=self.get_name(),
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.ceil_div(self.tokens, self.times_per_block),
                self.batches,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluInitialStateGradient(ShortConvKernel):
    """Differentiate causal history using its triangular output dependency.

    History position ``s`` contributes to output positions ``o <= s`` through
    weight tap ``s - o``. Each block computes that short sum for one sequence,
    history position, and channel group.
    """

    kernel_kind = "dstate"
    sequence_axis = "n"
    time_tiled = False

    def __init__(
        self,
        num_sequences: int,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
        dtype: ShortConvDType,
        d_activation,
    ):
        super().__init__(num_sequences, tokens, channels, width, config, dtype)
        self.num_sequences = num_sequences
        self.d_activation = d_activation

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        initial_state: cute.Tensor,
        grad_output: cute.Tensor,
        grad_initial_state: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
    ):
        """Differentiate one history position and packed channel group."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, history_offset, sequence = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread

        if cutlass.const_expr(cu_seqlens is None):
            sequence_start = sequence * self.tokens
            sequence_end = sequence_start + self.tokens
        else:
            sequence_start = Int32(cu_seqlens[sequence])
            sequence_end = Int32(cu_seqlens[sequence + 1])
        sequence_length = sequence_end - sequence_start

        if channel < self.channels:
            x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
            state_groups = cute.zipped_divide(initial_state, (1, self.channels_per_thread))
            dy_groups = cute.zipped_divide(grad_output, (1, self.channels_per_thread))
            dstate_groups = cute.zipped_divide(grad_initial_state, (1, self.channels_per_thread))
            weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    weights[channel_offset, tap] = Float32(weight[channel + channel_offset, tap])

            gradient = cute.make_rmem_tensor((self.channels_per_thread,), Float32)
            gradient.fill(Float32(0.0))
            for output_offset in cutlass.range_constexpr(self.width - 1):
                if output_offset <= history_offset and output_offset < sequence_length:
                    products = cute.make_rmem_tensor(
                        (self.channels_per_thread, self.width), Float32
                    )
                    for tap in cutlass.range_constexpr(self.width):
                        input_offset = output_offset + tap - (self.width - 1)
                        input_value = cute.make_rmem_tensor((self.channels_per_thread,), Float32)
                        if input_offset >= 0:
                            input_value.store(
                                x_groups[
                                    (
                                        (0, None),
                                        (sequence_start + input_offset, channel_group),
                                    )
                                ]
                                .load()
                                .to(Float32)
                            )
                        else:
                            input_value.store(
                                load_history(
                                    state_groups,
                                    Int32(sequence),
                                    input_offset,
                                    Int32(0),
                                    channel_group,
                                    self.width,
                                )
                            )
                        products[(None, tap)].store(
                            input_value.load() * weights[(None, tap)].load()
                        )
                    value = products.load().reduce(
                        cute.ReductionOp.ADD,
                        Float32(0.0),
                        reduction_profile=(None, 1),
                    )
                    output_gradient = (
                        dy_groups[
                            (
                                (0, None),
                                (sequence_start + output_offset, channel_group),
                            )
                        ]
                        .load()
                        .to(Float32)
                    )
                    weight_tap = history_offset - output_offset
                    gradient.store(
                        gradient.load()
                        + output_gradient
                        * self.d_activation(value)
                        * weights[(None, weight_tap)].load()
                    )
            dstate_groups[
                (
                    (0, None),
                    (sequence * (self.width - 1) + history_offset, channel_group),
                )
            ].store(gradient.load().to(self.dtype.cute_type))

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        initial_state: cute.Tensor,
        grad_output: cute.Tensor,
        grad_initial_state: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        stream,
    ):
        """Launch the initial-state gradient specialization."""
        self.kernel(
            x,
            weight,
            initial_state,
            grad_output,
            grad_initial_state,
            cu_seqlens,
            _name_prefix=self.get_name(),
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                self.width - 1,
                self.num_sequences,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


# The TVM-FFI compact-tensor ABI currently specializes T and C through these
# fake shapes. The kernel's bounds arithmetic does not otherwise require them
# to be constexpr; a future dynamic-shape launcher could pass those extents at
# runtime without relaxing the schedule parameters above.
def _fake_matrix(dtype: ShortConvDType, rows: int, columns: int):
    """Create a row-major fake tensor for one storage-dtype specialization."""
    return cute.runtime.make_fake_compact_tensor(
        dtype.cute_type,
        (rows, columns),
        stride_order=(1, 0),
        assumed_align=16,
    )


def _fake_cu_seqlens(num_sequences: int | None):
    """Create packed offsets for compilation or preserve the dense specialization."""
    if num_sequences is None:
        return None
    return cute.runtime.make_fake_compact_tensor(
        Int32,
        (num_sequences + 1,),
        stride_order=(0,),
        assumed_align=4,
    )


def _fake_initial_state(
    dtype: ShortConvDType,
    num_sequences: int,
    channels: int,
    width: int,
    has_initial_state: bool,
):
    """Create per-sequence history storage or preserve the zero-history specialization."""
    if not has_initial_state:
        return None
    return _fake_matrix(dtype, num_sequences * (width - 1), channels)


@jit_cache
def _compile_forward(
    batches: int,
    tokens: int,
    channels: int,
    width: int,
    dtype: ShortConvDType,
    config: ShortConvConfig,
    num_sequences: int | None = None,
    has_initial_state: bool = False,
):
    """Compile one static forward specialization."""
    operation = CausalConv1dSiluForward(batches, tokens, channels, width, config, dtype, _silu)
    return compile_tvm_ffi(
        operation,
        _fake_matrix(dtype, batches * tokens, channels),
        _fake_matrix(dtype, channels, width),
        _fake_matrix(dtype, batches * tokens, channels),
        _fake_cu_seqlens(num_sequences),
        _fake_initial_state(
            dtype,
            batches if num_sequences is None else num_sequences,
            channels,
            width,
            has_initial_state,
        ),
    )


@jit_cache
def _compile_input_gradient(
    batches: int,
    tokens: int,
    channels: int,
    width: int,
    dtype: ShortConvDType,
    config: ShortConvConfig,
    num_sequences: int | None = None,
    has_initial_state: bool = False,
):
    """Compile one static input-gradient specialization."""
    operation = CausalConv1dSiluInputGradient(
        batches, tokens, channels, width, config, dtype, _silu_derivative
    )
    return compile_tvm_ffi(
        operation,
        _fake_matrix(dtype, batches * tokens, channels),
        _fake_matrix(dtype, channels, width),
        _fake_matrix(dtype, batches * tokens, channels),
        _fake_matrix(dtype, batches * tokens, channels),
        _fake_cu_seqlens(num_sequences),
        _fake_initial_state(
            dtype,
            batches if num_sequences is None else num_sequences,
            channels,
            width,
            has_initial_state,
        ),
    )


@jit_cache
def _compile_weight_gradient(
    batches: int,
    tokens: int,
    channels: int,
    width: int,
    dtype: ShortConvDType,
    config: ShortConvConfig,
    num_sequences: int | None = None,
    has_initial_state: bool = False,
):
    """Compile one static weight-gradient specialization."""
    num_time_blocks = ceildiv(tokens, config.times_per_block)
    partials = cute.runtime.make_fake_compact_tensor(
        Float32,
        (batches, num_time_blocks, channels, width),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    operation = CausalConv1dSiluWeightGradientPartials(
        batches, tokens, channels, width, config, dtype, _silu_derivative
    )
    return compile_tvm_ffi(
        operation,
        _fake_matrix(dtype, batches * tokens, channels),
        _fake_matrix(dtype, channels, width),
        _fake_matrix(dtype, batches * tokens, channels),
        partials,
        _fake_cu_seqlens(num_sequences),
        _fake_initial_state(
            dtype,
            batches if num_sequences is None else num_sequences,
            channels,
            width,
            has_initial_state,
        ),
    )


@jit_cache
def _compile_initial_state_gradient(
    batches: int,
    tokens: int,
    channels: int,
    width: int,
    dtype: ShortConvDType,
    threads: int,
    channels_per_thread: int,
    num_sequences: int | None = None,
):
    """Compile one initial-state gradient specialization."""
    state_sequences = batches if num_sequences is None else num_sequences
    config = ShortConvConfig(threads, channels_per_thread, times_per_block=1)
    operation = CausalConv1dSiluInitialStateGradient(
        state_sequences, tokens, channels, width, config, dtype, _silu_derivative
    )
    return compile_tvm_ffi(
        operation,
        _fake_matrix(dtype, batches * tokens, channels),
        _fake_matrix(dtype, channels, width),
        _fake_matrix(dtype, state_sequences * (width - 1), channels),
        _fake_matrix(dtype, batches * tokens, channels),
        _fake_matrix(dtype, state_sequences * (width - 1), channels),
        _fake_cu_seqlens(num_sequences),
    )


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> None:
    """Validate the public compile-time-width CuTeDSL tensor contract."""
    if x.ndim != 3:
        raise ValueError(f"x must have shape [B, T, C], got {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] < 1:
        raise ValueError(f"x must have positive B, T, and C dimensions, got {tuple(x.shape)}")
    if x.dtype not in SHORT_CONV_DTYPES or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA FP16, BF16, or FP32 tensor")
    if weight.ndim != 2 or weight.shape[0] != x.shape[2] or weight.shape[1] < 1:
        raise ValueError(
            f"weight must have shape [C, W] with W positive, got {tuple(weight.shape)}"
        )
    if weight.dtype != x.dtype or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("weight must match x dtype and be contiguous on x.device")
    if cu_seqlens is not None:
        if x.shape[0] != 1:
            raise ValueError("packed cu_seqlens require x to have batch size 1")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
        if (
            cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != x.device
            or not cu_seqlens.is_contiguous()
        ):
            raise ValueError("cu_seqlens must be contiguous CUDA int32 on x.device")
    if initial_state is not None:
        state_sequences = x.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
        expected_state = (state_sequences, weight.shape[1] - 1, x.shape[2])
        if initial_state.shape != expected_state:
            raise ValueError(
                f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
            )
        if (
            initial_state.dtype != x.dtype
            or initial_state.device != x.device
            or not initial_state.is_contiguous()
        ):
            raise ValueError("initial_state must match x dtype and be contiguous on x.device")


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize the uncommon contiguous view that misses the launcher ABI alignment."""
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _launch_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: ShortConvConfig,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Allocate and launch the compiled forward specialization."""
    x, weight = _aligned(x), _aligned(weight)
    if initial_state is not None:
        initial_state = _aligned(initial_state)
    batches, tokens, channels = x.shape
    width = weight.shape[1]
    dtype = SHORT_CONV_DTYPES[x.dtype]
    output = torch.empty_like(x)
    compiled = _compile_forward(
        batches,
        tokens,
        channels,
        width,
        dtype,
        config,
        None if cu_seqlens is None else cu_seqlens.shape[0] - 1,
        initial_state is not None,
    )
    compiled(
        x.view(batches * tokens, channels),
        weight,
        output.view(batches * tokens, channels),
        cu_seqlens,
        None if initial_state is None else initial_state.flatten(0, 1),
    )
    return output


def _launch_backward(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    input_config: ShortConvConfig,
    weight_config: ShortConvConfig,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    compute_initial_state_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Launch configured gradient kernels and reduce FP32 partials."""
    x, weight = _aligned(x), _aligned(weight)
    if initial_state is not None:
        initial_state = _aligned(initial_state)
    batches, tokens, channels = x.shape
    width = weight.shape[1]
    dtype = SHORT_CONV_DTYPES[x.dtype]
    num_sequences = None if cu_seqlens is None else cu_seqlens.shape[0] - 1
    grad_output = _aligned(grad_output.contiguous())
    grad_x = torch.empty_like(x)
    _compile_input_gradient(
        batches,
        tokens,
        channels,
        width,
        dtype,
        input_config,
        num_sequences,
        initial_state is not None,
    )(
        x.view(batches * tokens, channels),
        weight,
        grad_output.view(batches * tokens, channels),
        grad_x.view(batches * tokens, channels),
        cu_seqlens,
        None if initial_state is None else initial_state.flatten(0, 1),
    )

    num_time_blocks = ceildiv(tokens, weight_config.times_per_block)
    if initial_state is None or not compute_initial_state_grad:
        grad_initial_state = None
    elif width == 1:
        grad_initial_state = torch.zeros_like(initial_state)
    else:
        grad_initial_state = torch.empty_like(initial_state)
    partials = torch.empty(
        (batches, num_time_blocks, channels, width),
        dtype=torch.float32,
        device=x.device,
    )
    _compile_weight_gradient(
        batches,
        tokens,
        channels,
        width,
        dtype,
        weight_config,
        num_sequences,
        initial_state is not None,
    )(
        x.view(batches * tokens, channels),
        weight,
        grad_output.view(batches * tokens, channels),
        partials,
        cu_seqlens,
        None if initial_state is None else initial_state.flatten(0, 1),
    )
    if grad_initial_state is not None and width > 1:
        _compile_initial_state_gradient(
            batches,
            tokens,
            channels,
            width,
            dtype,
            input_config.threads,
            input_config.channels_per_thread,
            num_sequences,
        )(
            x.view(batches * tokens, channels),
            weight,
            initial_state.flatten(0, 1),
            grad_output.view(batches * tokens, channels),
            grad_initial_state.flatten(0, 1),
            cu_seqlens,
        )
    return grad_x, partials.sum(dim=(0, 1)).to(x.dtype), grad_initial_state


def _compatible_config(config: ShortConvConfig, channels: int) -> ShortConvConfig:
    """Adapt the packed channel width while preserving the tuned schedule."""
    channels_per_thread = 1
    for divisor in range(config.channels_per_thread, 1, -1):
        if config.channels_per_thread % divisor == 0 and channels % divisor == 0:
            channels_per_thread = divisor
            break
    return ShortConvConfig(config.threads, channels_per_thread, config.times_per_block)


def _candidate_configs(
    kind: str,
    channels: int,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ShortConvConfig, ...]:
    """Return the focused schedule space used by the explicit tuning flow."""
    defaults = ShortConvTunedConfig.default(dtype)
    if kind == "forward":
        default = defaults.forward
        candidates = (
            ShortConvConfig(64, 4, 8),
            ShortConvConfig(128, 2, 8),
            ShortConvConfig(128, 4, 4),
            ShortConvConfig(128, 4, 8),
            ShortConvConfig(128, 4, 16),
            ShortConvConfig(256, 4, 8),
        )
    elif kind == "input_gradient":
        default = defaults.input_gradient
        candidates = (
            ShortConvConfig(64, 4, 8),
            ShortConvConfig(128, 2, 8),
            ShortConvConfig(128, 4, 6),
            ShortConvConfig(128, 4, 8),
            ShortConvConfig(128, 4, 10),
            ShortConvConfig(128, 4, 12),
            ShortConvConfig(256, 4, 8),
        )
    elif kind == "weight_gradient":
        default = defaults.weight_gradient
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
    candidates = (*candidates, _compatible_config(default, channels))
    return tuple(
        dict.fromkeys(
            config for config in candidates if channels % config.channels_per_thread == 0
        )
    )


def tune_causal_conv1d_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    forward_configs: Iterable[ShortConvConfig] | None = None,
    input_grad_configs: Iterable[ShortConvConfig] | None = None,
    weight_grad_configs: Iterable[ShortConvConfig] | None = None,
    parallel_compile: bool = True,
) -> ShortConvTunedConfig:
    """Compile and benchmark forward, input-gradient, and weight-gradient schedules.

    Width is an operation parameter and is specialized but not autotuned. Packed
    offsets and initial state use the same inputs accepted by the public operation.
    The returned configs can be passed directly to :func:`cute_causal_conv1d_silu`.
    Tuning uses the shared CuTeDSL ``tune`` flow: cached variants compile in
    parallel and execute sequentially under the vetted Inductor GPU benchmarker.
    """
    _validate_inputs(x, weight, cu_seqlens, initial_state)
    if grad_output.shape != x.shape or grad_output.dtype != x.dtype:
        raise ValueError("grad_output must match x shape and dtype")
    if grad_output.device != x.device or not grad_output.is_contiguous():
        raise ValueError("grad_output must be contiguous on x.device")

    x, weight, grad_output = _aligned(x), _aligned(weight), _aligned(grad_output)
    if initial_state is not None:
        initial_state = _aligned(initial_state)
    batches, tokens, channels = x.shape
    width = weight.shape[1]
    dtype = SHORT_CONV_DTYPES[x.dtype]
    num_sequences = None if cu_seqlens is None else cu_seqlens.shape[0] - 1
    x_matrix = x.view(batches * tokens, channels)
    grad_matrix = grad_output.view(batches * tokens, channels)
    kernel_initial_state = None if width == 1 else initial_state
    state_matrix = None if kernel_initial_state is None else kernel_initial_state.flatten(0, 1)

    forward_candidates = tuple(
        _candidate_configs("forward", channels, x.dtype)
        if forward_configs is None
        else forward_configs
    )
    for config in forward_candidates:
        _validate_config(config, channels, "forward_configs")
    forward_output = torch.empty_like(x).view(batches * tokens, channels)
    forward = tune(
        forward_candidates,
        _compile_forward,
        lambda compiled, _config: compiled(
            x_matrix,
            weight,
            forward_output,
            cu_seqlens,
            state_matrix,
        ),
        compile_call=lambda config: (
            batches,
            tokens,
            channels,
            width,
            dtype,
            config,
            num_sequences,
            kernel_initial_state is not None,
        ),
        parallel_compile=parallel_compile,
    )

    input_candidates = tuple(
        _candidate_configs("input_gradient", channels, x.dtype)
        if input_grad_configs is None
        else input_grad_configs
    )
    for config in input_candidates:
        _validate_config(config, channels, "input_grad_configs")
    grad_x = torch.empty_like(x).view(batches * tokens, channels)
    input_gradient = tune(
        input_candidates,
        _compile_input_gradient,
        lambda compiled, _config: compiled(
            x_matrix, weight, grad_matrix, grad_x, cu_seqlens, state_matrix
        ),
        compile_call=lambda config: (
            batches,
            tokens,
            channels,
            width,
            dtype,
            config,
            num_sequences,
            kernel_initial_state is not None,
        ),
        parallel_compile=parallel_compile,
    )

    weight_candidates = tuple(
        _candidate_configs("weight_gradient", channels, x.dtype)
        if weight_grad_configs is None
        else weight_grad_configs
    )
    for config in weight_candidates:
        _validate_config(config, channels, "weight_grad_configs")
    partials = {
        config: torch.empty(
            batches,
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
            cu_seqlens,
            state_matrix,
        ),
        compile_call=lambda config: (
            batches,
            tokens,
            channels,
            width,
            dtype,
            config,
            num_sequences,
            kernel_initial_state is not None,
        ),
        parallel_compile=parallel_compile,
    )
    return ShortConvTunedConfig(forward, input_gradient, weight_gradient)


def _config(threads: int, channels_per_thread: int, times_per_block: int) -> ShortConvConfig:
    """Reconstruct a compile-time config from custom-op scalar arguments."""
    return ShortConvConfig(threads, channels_per_thread, times_per_block)


@torch.library.custom_op("attn_gym::_cute_short_conv_fwd", mutates_args=())
def _forward_custom_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the tuned K3 defaults through the lean production schema."""
    return _launch_forward(
        x,
        weight,
        ShortConvTunedConfig.default(x.dtype).forward,
        cu_seqlens,
        initial_state,
    )


@_forward_custom_op.register_fake
def _default_forward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    del weight, cu_seqlens, initial_state
    return torch.empty_like(x)


@torch.library.custom_op("attn_gym::_cute_short_conv_bwd", mutates_args=())
def _backward_custom_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the tuned K3 backward defaults through the lean schema."""
    defaults = ShortConvTunedConfig.default(x.dtype)
    grad_x, grad_weight, _ = _launch_backward(
        x,
        weight,
        grad_output,
        defaults.input_gradient,
        defaults.weight_gradient,
        cu_seqlens,
    )
    return grad_x, grad_weight


@_backward_custom_op.register_fake
def _default_backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_output, cu_seqlens
    return torch.empty_like(x), torch.empty_like(weight)


def _setup_default_context(ctx, inputs, output) -> None:
    del output
    ctx.save_for_backward(*inputs)


@torch.autograd.function.once_differentiable
def _default_backward(ctx, grad_output: torch.Tensor):
    x, weight, cu_seqlens, initial_state = ctx.saved_tensors
    if initial_state is None:
        grad_x, grad_weight = _backward_custom_op(x, weight, grad_output, cu_seqlens)
        grad_initial_state = None
    else:
        defaults = ShortConvTunedConfig.default(x.dtype)
        input_config = defaults.input_gradient
        weight_config = defaults.weight_gradient
        grad_x, grad_weight, grad_initial_state = _configured_backward_custom_op(
            x,
            weight,
            grad_output,
            cu_seqlens,
            initial_state,
            ctx.needs_input_grad[3],
            input_config.threads,
            input_config.channels_per_thread,
            input_config.times_per_block,
            weight_config.threads,
            weight_config.channels_per_thread,
            weight_config.times_per_block,
        )
        if not ctx.needs_input_grad[3]:
            grad_initial_state = None
    return grad_x, grad_weight, None, grad_initial_state


_forward_custom_op.register_autograd(
    _default_backward,
    setup_context=_setup_default_context,
)


@torch.library.custom_op("attn_gym::_cute_short_conv_configured_fwd", mutates_args=())
def _configured_forward_custom_op(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
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
        cu_seqlens,
        initial_state,
    )


@_configured_forward_custom_op.register_fake
def _forward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
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
        cu_seqlens,
        initial_state,
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
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    compute_initial_state_grad: bool,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the configured first-order backward launchers opaque."""
    grad_x, grad_weight, grad_initial_state = _launch_backward(
        x,
        weight,
        grad_output,
        _config(input_threads, input_channels, input_times),
        _config(weight_threads, weight_channels, weight_times),
        cu_seqlens,
        initial_state,
        compute_initial_state_grad,
    )
    if grad_initial_state is None:
        grad_initial_state = x.new_empty(0)
    return grad_x, grad_weight, grad_initial_state


@_configured_backward_custom_op.register_fake
def _backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    compute_initial_state_grad: bool,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe backward output metadata without invoking the compiler."""
    del (
        grad_output,
        cu_seqlens,
        input_threads,
        input_channels,
        input_times,
        weight_threads,
        weight_channels,
        weight_times,
    )
    grad_initial_state = (
        x.new_empty(0)
        if initial_state is None or not compute_initial_state_grad
        else torch.empty_like(initial_state)
    )
    return torch.empty_like(x), torch.empty_like(weight), grad_initial_state


def _setup_context(ctx, inputs, output) -> None:
    """Save inputs and backward specializations for preactivation recomputation."""
    del output
    (
        x,
        weight,
        cu_seqlens,
        initial_state,
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
    ctx.save_for_backward(x, weight, cu_seqlens, initial_state)


@torch.autograd.function.once_differentiable
def _backward(ctx, grad_output: torch.Tensor):
    """Dispatch the registered first-order backward custom operator."""
    x, weight, cu_seqlens, initial_state = ctx.saved_tensors
    grad_x, grad_weight, grad_initial_state = _configured_backward_custom_op(
        x,
        weight,
        grad_output,
        cu_seqlens,
        initial_state,
        ctx.needs_input_grad[3],
        ctx.input_threads,
        ctx.input_channels,
        ctx.input_times,
        ctx.weight_threads,
        ctx.weight_channels,
        ctx.weight_times,
    )
    if not ctx.needs_input_grad[3]:
        grad_initial_state = None
    return (
        grad_x,
        grad_weight,
        None,
        grad_initial_state,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


_configured_forward_custom_op.register_autograd(_backward, setup_context=_setup_context)


def _validate_config(config: ShortConvConfig, channels: int, name: str) -> None:
    """Reject launch configurations that cannot form safe packed channel groups."""
    if config.threads < 32 or config.threads > 1024 or config.threads % 32 != 0:
        raise ValueError(f"{name}.threads must be a warp multiple in [32, 1024]")
    if config.channels_per_thread < 1 or channels % config.channels_per_thread != 0:
        raise ValueError(f"C must be divisible by positive {name}.channels_per_thread")
    if config.times_per_block < 1:
        raise ValueError(f"{name}.times_per_block must be positive")


def dense_final_conv_state(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_length: int,
) -> torch.Tensor:
    """Return the final dense history for a positive state length."""
    if x.shape[1] >= state_length:
        if initial_state is None:
            return x[:, -state_length:].clone()
        # Preserve the shaped zero gradient of replaced caller history.
        return x[:, -state_length:] + initial_state[:, :0].sum()
    prefix = (
        x.new_zeros(x.shape[0], state_length - x.shape[1], x.shape[2])
        if initial_state is None
        else initial_state[:, x.shape[1] :]
    )
    return torch.cat((prefix, x), dim=1)


def packed_final_conv_state(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_length: int,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Gather final histories from packed sequences without host reads."""
    starts = cu_seqlens[:-1].to(torch.int64)
    ends = cu_seqlens[1:].to(torch.int64)
    offsets = torch.arange(state_length, device=x.device)
    positions = ends[:, None] - state_length + offsets
    from_input = positions >= starts[:, None]
    input_values = x[0, positions.clamp(0, x.shape[1] - 1)]
    if initial_state is None:
        state_values = torch.zeros_like(input_values)
    else:
        state_positions = (offsets + (ends - starts)[:, None]).clamp(0, state_length - 1)
        state_values = initial_state.gather(
            1,
            state_positions[:, :, None].expand(-1, -1, x.shape[2]),
        )
    return torch.where(from_input[:, :, None], input_values, state_values)


def final_conv_state(
    x: torch.Tensor,
    initial_state: torch.Tensor | None,
    state_length: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return each sequence's final short history without copying the complete input."""
    if state_length == 0:
        num_sequences = x.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
        final_state = x[:, :0].expand(num_sequences, -1, -1).clone()
        return final_state if initial_state is None else final_state + initial_state
    if cu_seqlens is None:
        return dense_final_conv_state(x, initial_state, state_length)
    return packed_final_conv_state(x, initial_state, state_length, cu_seqlens)


def cute_causal_conv1d_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    forward_config: ShortConvConfig | None = None,
    input_grad_config: ShortConvConfig | None = None,
    weight_grad_config: ShortConvConfig | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply causal depthwise convolution and SiLU with CuTeDSL.

    Width is inferred from ``weight.shape[1]`` and compile-time specialized.
    Schedule fields control register shapes, vector partitioning, unrolled loops,
    and block mapping, so changing one compiles and caches a distinct kernel.

    Args:
        x: Contiguous CUDA FP16, BF16, or FP32 input with shape ``[B, T, C]``.
            Each batch row is convolved as an independent sequence.
        weight: Contiguous depthwise weights with shape ``[C, W]`` matching
            ``x`` dtype and device.
        cu_seqlens: Optional contiguous CUDA int32 offsets delimiting independent
            sequences in a packed ``[1, T, C]`` input. Offsets must be nondecreasing,
            begin at zero, and end at ``T``; repeated offsets represent empty padding
            slots for static-shape CUDA Graph replay.
        initial_state: Optional causal history with shape ``[N, W - 1, C]``, where
            ``N`` is the dense batch size or the number of packed sequences. Absence
            is equivalent to an all-zero history.
        return_final_state: Return the final ``W - 1`` input positions with the output.
        forward_config: Optional forward schedule specialization.
        input_grad_config: Optional input-gradient schedule specialization.
        weight_grad_config: Optional weight-gradient schedule specialization.

    Returns:
        A contiguous tensor with the same shape, dtype, and device as ``x``.
        When ``return_final_state`` is true, also returns ``[N, W - 1, C]`` history.
    """
    _validate_inputs(x, weight, cu_seqlens, initial_state)
    channels = x.shape[2]
    kernel_initial_state = None if weight.shape[1] == 1 else initial_state
    defaults = ShortConvTunedConfig.default(x.dtype)
    default_forward = defaults.forward
    default_input_grad = defaults.input_gradient
    default_weight_grad = defaults.weight_gradient
    forward = forward_config or _compatible_config(default_forward, channels)
    input_grad = input_grad_config or _compatible_config(default_input_grad, channels)
    weight_grad = weight_grad_config or _compatible_config(default_weight_grad, channels)
    for name, config in (
        ("forward_config", forward),
        ("input_grad_config", input_grad),
        ("weight_grad_config", weight_grad),
    ):
        _validate_config(config, channels, name)
    if (
        forward_config is None
        and input_grad_config is None
        and weight_grad_config is None
        and forward == default_forward
        and input_grad == default_input_grad
        and weight_grad == default_weight_grad
    ):
        output = _forward_custom_op(x, weight, cu_seqlens, kernel_initial_state)
    else:
        output = _configured_forward_custom_op(
            x,
            weight,
            cu_seqlens,
            kernel_initial_state,
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
    if initial_state is not None and weight.shape[1] == 1:
        output = output + initial_state.sum()
    if not return_final_state:
        return output
    return output, final_conv_state(
        x,
        initial_state,
        weight.shape[1] - 1,
        cu_seqlens,
    )


__all__ = [
    "ShortConvConfig",
    "ShortConvTunedConfig",
    "cute_causal_conv1d_silu",
    "tune_causal_conv1d_silu",
]
