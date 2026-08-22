# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL causal depthwise convolution with a first-order backward.

The implementation accepts contiguous FP16, BF16, or FP32 ``[B, T, C]`` tensors and
treats every batch row as an independent sequence. Dense inputs may supply their
preceding ``W - 1`` input positions as functional state. An optional CUDA
``cu_seqlens`` tensor instead delimits independent sequences packed into ``[1, T, C]``.
Its final offset is the dynamic active endpoint and may be smaller than the physical
capacity ``T``. Each thread owns a compile-time number of adjacent channels. Forward
stages its input window in registers, while backward computes input gradients and FP32
weight-gradient partials followed by a Torch reduction.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import cutlass
import torch
from cutlass import BFloat16, Float16, Float32, Int32, Int64, cute, pipeline
from cutlass.cute.nvgpu import cpasync

from attn_gym._backends.cute import (
    ceildiv,
    compile_tvm_ffi,
    get_device_properties,
    jit_cache,
    tune,
)
from attn_gym._backends.cute.device import upper_bound
from attn_gym.linear.kda import ops as kda_ops
from attn_gym.linear.kda.short_conv.activations import Activation, resolve_activation

_forward_op = kda_ops.short_conv_forward_op
_backward_op = kda_ops.short_conv_backward_op
_decode_op = kda_ops.short_conv_decode_op
_configured_forward_op = kda_ops.short_conv_configured_forward_op
_configured_backward_op = kda_ops.short_conv_configured_backward_op
_configured_backward_with_state_grad_op = kda_ops.short_conv_configured_backward_with_state_grad_op
_configured_decode_op = kda_ops.short_conv_configured_decode_op


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


# NOTE [Short-convolution time scheduling]
# Match the shared scheduler's concrete launch convention: zero time workers means a
# static capacity grid whose inactive CTAs return from a device check; a positive count
# means a bounded grid that strides over active tiles. The measured short-convolution
# policy uses eight CTAs per SM across the complete channel/time grid.
_SHORT_CONV_CTAS_PER_SM = 8


def _persistent_time_workers(
    tokens: int,
    channels: int,
    config: ShortConvConfig,
    device: torch.device,
    *,
    persistent_eligible: bool,
) -> int:
    """Return a bounded packed time grid, or zero for static execution."""
    if not persistent_eligible:
        return 0
    capacity = ceildiv(tokens, config.times_per_block)
    channel_blocks = ceildiv(channels, config.threads * config.channels_per_thread)
    device_workers = get_device_properties(device).multi_processor_count * _SHORT_CONV_CTAS_PER_SM
    workers = ceildiv(device_workers, channel_blocks)
    return workers if workers < capacity else 0


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
    def default(
        cls,
        dtype: torch.dtype = torch.bfloat16,
        *,
        packed: bool = False,
    ) -> ShortConvTunedConfig:
        """Return measured GB300 defaults for one storage dtype and layout mode."""
        match dtype:
            case torch.float16 | torch.bfloat16:
                forward = ShortConvConfig(128, 4, 16)
                gradients = (
                    (ShortConvConfig(128, 4, 16), ShortConvConfig(128, 2, 32))
                    if packed
                    else (ShortConvConfig(128, 2, 32), ShortConvConfig(128, 2, 64))
                )
            case torch.float32:
                forward = ShortConvConfig(128, 4, 4)
                gradients = (
                    (ShortConvConfig(128, 2, 16), ShortConvConfig(128, 4, 32))
                    if packed
                    else (ShortConvConfig(128, 2, 12), ShortConvConfig(128, 2, 160))
                )
            case _:
                raise ValueError(f"unsupported short-convolution dtype {dtype}")
        return cls(forward, *gradients)


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
    """Advance packed sequence metadata when time crosses a boundary.

    NOTE [Boundary trigger forms]: offsets are monotone with ``end <= tokens``
    (device-asserted by the scheduler; ``sequence_bounds`` is a binary search).
    Kernels resolve bounds at tile entry and walk ``time`` with unit stride, so
    every boundary is hit by equality: side-effect-free refreshes may use ``>=``
    (self-healing) or ``==`` (skips the inactive-tail re-lookup); once-only side
    effects such as the input-gradient terminal flush MUST use ``==``. Any
    traversal-stride change must revisit every ``== sequence_end`` trigger. The
    ``skip_sequence_boundaries`` constexpr fast path is perf-load-bearing
    (GB300, C=12288: 17-19% for full-capacity replays; a runtime bool loses
    28-33%).
    """
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
    tma_stage_tokens: ClassVar[int] = 0

    def __init__(
        self,
        sequences: int,
        tokens: int,
        channels: int,
        width: int,
        config: ShortConvConfig,
        dtype: ShortConvDType,
        time_workers: int = 0,
    ):
        self.sequences = sequences
        self.tokens = tokens
        self.channels = channels
        self.width = width
        self.threads = config.threads
        self.channels_per_thread = config.channels_per_thread
        self.times_per_block = config.times_per_block
        self.dtype = dtype
        self.time_workers = time_workers

    def get_name(self) -> str:
        """Return the stable compiled-artifact name."""
        name = (
            f"short_conv_{self.kernel_kind}_{self.dtype.name}_{self.sequence_axis}{self.sequences}"
            f"_t{self.tokens}_c{self.channels}_w{self.width}_th{self.threads}"
        )
        if self.time_tiled:
            name += f"_bt{self.times_per_block}"
        if self.time_workers:
            name += f"_execution_persistent_tw{self.time_workers}"
        name += f"_v{self.channels_per_thread}"
        if self.tma_stage_tokens:
            name += f"_tma{self.tma_stage_tokens}"
        return name


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
        time_workers: int = 0,
    ):
        super().__init__(batches, tokens, channels, width, config, dtype, time_workers)
        self.batches = batches
        self.activation = activation

    @cute.jit
    def run_tile(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        channel_group: Int32,
        channel: Int32,
        time_block: Int32,
        batch: Int32,
        active_endpoint: Int32,
    ):
        """Compute one independent physical time tile for a channel group."""
        time_start = time_block * self.times_per_block
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
            if input_time >= 0 and input_time < active_endpoint:
                inputs[(None, input_offset)].store(
                    x_groups[((0, None), (batch * self.tokens + input_time, channel_group))].load()
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
            if time < active_endpoint:
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

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
    ):
        """Compute one channel group over static or persistent time tiles."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, worker, batch = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread
        if channel < self.channels:
            if cutlass.const_expr(cu_seqlens is not None):
                active_endpoint = Int32(cu_seqlens[cute.size(cu_seqlens) - 1])
                active_time_blocks = (
                    active_endpoint + self.times_per_block - 1
                ) // self.times_per_block
                if cutlass.const_expr(self.time_workers == 0):
                    if active_endpoint == self.tokens:
                        self.run_tile(
                            x,
                            weight,
                            output,
                            cu_seqlens,
                            initial_state,
                            Int32(channel_group),
                            Int32(channel),
                            Int32(worker),
                            Int32(batch),
                            Int32(self.tokens),
                        )
                    elif worker < active_time_blocks:
                        self.run_tile(
                            x,
                            weight,
                            output,
                            cu_seqlens,
                            initial_state,
                            Int32(channel_group),
                            Int32(channel),
                            Int32(worker),
                            Int32(batch),
                            active_endpoint,
                        )
                else:
                    for time_block in cutlass.range(worker, active_time_blocks, self.time_workers):
                        self.run_tile(
                            x,
                            weight,
                            output,
                            cu_seqlens,
                            initial_state,
                            Int32(channel_group),
                            Int32(channel),
                            Int32(time_block),
                            Int32(batch),
                            active_endpoint,
                        )
            else:
                self.run_tile(
                    x,
                    weight,
                    output,
                    cu_seqlens,
                    initial_state,
                    Int32(channel_group),
                    Int32(channel),
                    Int32(worker),
                    Int32(batch),
                    Int32(self.tokens),
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
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            x,
            weight,
            output,
            cu_seqlens,
            initial_state,
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                (
                    self.time_workers
                    if cutlass.const_expr(self.time_workers > 0)
                    else cute.ceil_div(self.tokens, self.times_per_block)
                ),
                self.batches,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluDecode(ShortConvKernel):
    """Advance a paged causal history by one token per sequence, activation fused."""

    kernel_kind = "decode"
    sequence_axis = "n"
    time_tiled = False

    def __init__(
        self,
        channels: int,
        width: int,
        config: ShortConvConfig,
        dtype: ShortConvDType,
        activation,
    ):
        assert channels % config.channels_per_thread == 0, (
            f"decode channels ({channels}) must be divisible by channels_per_thread "
            f"({config.channels_per_thread})"
        )
        super().__init__(0, 1, channels, width, config, dtype)
        self.activation = activation

    def get_name(self) -> str:
        """Name the artifact without the runtime-bound sequence axis."""
        return (
            f"short_conv_{self.kernel_kind}_{self.dtype.name}_c{self.channels}"
            f"_w{self.width}_th{self.threads}_v{self.channels_per_thread}"
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        state: cute.Tensor,
        state_indices: cute.Tensor | None,
    ):
        """Advance one packed channel group of one sequence's history."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, sequence, _ = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread

        if channel < self.channels:
            slot = Int32(sequence)
            active = cutlass.Boolean(True)
            if cutlass.const_expr(state_indices is not None):
                slot = state_indices[sequence]
                active = cutlass.Boolean(slot > 0)

            x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
            output_groups = cute.zipped_divide(output, (1, self.channels_per_thread))
            state_groups = cute.zipped_divide(state, (1, self.channels_per_thread))

            if active:
                weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
                for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                    for tap in cutlass.range_constexpr(self.width):
                        weights[channel_offset, tap] = Float32(
                            weight[channel + channel_offset, tap]
                        )

                history_base = slot * (self.width - 1)
                taps = cute.make_rmem_tensor(
                    (self.channels_per_thread, self.width), self.dtype.cute_type
                )
                for row in cutlass.range_constexpr(self.width - 1):
                    taps[(None, row)].store(
                        state_groups[((0, None), (history_base + row, channel_group))].load()
                    )
                taps[(None, self.width - 1)].store(
                    x_groups[((0, None), (sequence, channel_group))].load()
                )

                output_groups[((0, None), (sequence, channel_group))].store(
                    self.activation(unrolled_dot(taps, weights, 0, self.width)).to(
                        self.dtype.cute_type
                    )
                )
                for row in cutlass.range_constexpr(self.width - 1):
                    state_groups[((0, None), (history_base + row, channel_group))].store(
                        taps[(None, row + 1)].load()
                    )
            else:
                padding = cute.make_rmem_tensor((self.channels_per_thread,), self.dtype.cute_type)
                padding.fill(self.dtype.cute_type(0.0))
                output_groups[((0, None), (sequence, channel_group))].store(padding.load())

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        state: cute.Tensor,
        state_indices: cute.Tensor | None,
        stream,
    ):
        """Launch the configured decode specialization."""
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(x, weight, output, state, state_indices).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.size(x, mode=[0]),
                1,
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
        time_workers: int = 0,
    ):
        super().__init__(batches, tokens, channels, width, config, dtype, time_workers)
        self.batches = batches
        self.d_activation = d_activation

    @cute.jit
    def run_tile(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        grad_x: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        channel_group: Int32,
        channel: Int32,
        time_block: Int32,
        batch: Int32,
        active_endpoint: Int32,
    ):
        """Compute one independent physical time tile for a channel group."""
        time_start = time_block * self.times_per_block
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
        for input_offset in cutlass.range_constexpr(self.times_per_block + 2 * (self.width - 1)):
            input_time = time_start + input_offset - (self.width - 1)
            if input_time >= 0 and input_time < active_endpoint:
                inputs[(None, input_offset)].store(
                    x_groups[((0, None), (batch * self.tokens + input_time, channel_group))].load()
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
            if output_time < active_endpoint:
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
            if time < active_endpoint:
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
                products = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
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
        """Compute one channel group over static or persistent time tiles."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, worker, batch = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread
        if channel < self.channels:
            if cutlass.const_expr(cu_seqlens is not None):
                active_endpoint = Int32(cu_seqlens[cute.size(cu_seqlens) - 1])
                active_time_blocks = (
                    active_endpoint + self.times_per_block - 1
                ) // self.times_per_block
                if cutlass.const_expr(self.time_workers == 0):
                    if active_endpoint == self.tokens:
                        self.run_tile(
                            x,
                            weight,
                            grad_output,
                            grad_x,
                            cu_seqlens,
                            initial_state,
                            Int32(channel_group),
                            Int32(channel),
                            Int32(worker),
                            Int32(batch),
                            Int32(self.tokens),
                        )
                    elif worker < active_time_blocks:
                        self.run_tile(
                            x,
                            weight,
                            grad_output,
                            grad_x,
                            cu_seqlens,
                            initial_state,
                            Int32(channel_group),
                            Int32(channel),
                            Int32(worker),
                            Int32(batch),
                            active_endpoint,
                        )
                else:
                    for time_block in cutlass.range(worker, active_time_blocks, self.time_workers):
                        self.run_tile(
                            x,
                            weight,
                            grad_output,
                            grad_x,
                            cu_seqlens,
                            initial_state,
                            Int32(channel_group),
                            Int32(channel),
                            Int32(time_block),
                            Int32(batch),
                            active_endpoint,
                        )
            else:
                self.run_tile(
                    x,
                    weight,
                    grad_output,
                    grad_x,
                    cu_seqlens,
                    initial_state,
                    Int32(channel_group),
                    Int32(channel),
                    Int32(worker),
                    Int32(batch),
                    Int32(self.tokens),
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
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            x,
            weight,
            grad_output,
            grad_x,
            cu_seqlens,
            initial_state,
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                (
                    self.time_workers
                    if cutlass.const_expr(self.time_workers > 0)
                    else cute.ceil_div(self.tokens, self.times_per_block)
                ),
                self.batches,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluWeightGradientPartials(ShortConvKernel):
    """Compute FP32 worker partial sums for the weight gradient."""

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
        time_workers: int = 0,
    ):
        super().__init__(batches, tokens, channels, width, config, dtype, time_workers)
        self.batches = batches
        self.d_activation = d_activation

    @cute.jit
    def accumulate_tile(
        self,
        x: cute.Tensor,
        grad_output: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        weights: cute.Tensor,
        accumulators: cute.Tensor,
        channel_group: Int32,
        time_block: Int32,
        batch: Int32,
        active_endpoint: Int32,
    ):
        """Accumulate one logical time tile into a worker-local FP32 partial."""
        time_start = time_block * self.times_per_block
        x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
        dy_groups = cute.zipped_divide(grad_output, (1, self.channels_per_thread))
        if cutlass.const_expr(initial_state is not None):
            initial_groups = cute.zipped_divide(initial_state, (1, self.channels_per_thread))
        sequence, sequence_start, sequence_end = tile_sequence_bounds(
            cu_seqlens,
            Int32(time_start),
            Int32(batch),
            self.tokens,
        )
        for time_offset in cutlass.range(self.times_per_block, unroll_full=True):
            time = time_start + time_offset
            if time < active_endpoint:
                active_time = True
                if cutlass.const_expr(cu_seqlens is not None):
                    sequence, sequence_start, sequence_end = advance_sequence_bounds(
                        cu_seqlens,
                        sequence,
                        sequence_start,
                        sequence_end,
                        Int32(time),
                    )
                    active_time = time < sequence_end
                if active_time:
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
                    incoming = (
                        dy_groups[((0, None), (batch * self.tokens + time, channel_group))]
                        .load()
                        .to(Float32)
                    )
                    grad_z = incoming * self.d_activation(value)
                    for tap in cutlass.range_constexpr(self.width):
                        accumulators[(None, tap)].store(
                            accumulators[(None, tap)].load()
                            + grad_z * input_taps[(None, tap)].load()
                        )

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
        """Accumulate static tiles or a strided active-tile list into one worker partial."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, worker, batch = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread

        if channel < self.channels:
            weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            accumulators = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
            accumulators.fill(Float32(0.0))
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    weights[channel_offset, tap] = Float32(weight[channel + channel_offset, tap])

            if cutlass.const_expr(cu_seqlens is not None):
                active_endpoint = Int32(cu_seqlens[cute.size(cu_seqlens) - 1])
                active_time_blocks = (
                    active_endpoint + self.times_per_block - 1
                ) // self.times_per_block
                if cutlass.const_expr(self.time_workers == 0):
                    if active_endpoint == self.tokens:
                        self.accumulate_tile(
                            x,
                            grad_output,
                            cu_seqlens,
                            initial_state,
                            weights,
                            accumulators,
                            Int32(channel_group),
                            Int32(worker),
                            Int32(batch),
                            Int32(self.tokens),
                        )
                    elif worker < active_time_blocks:
                        self.accumulate_tile(
                            x,
                            grad_output,
                            cu_seqlens,
                            initial_state,
                            weights,
                            accumulators,
                            Int32(channel_group),
                            Int32(worker),
                            Int32(batch),
                            active_endpoint,
                        )
                else:
                    for time_block in cutlass.range(worker, active_time_blocks, self.time_workers):
                        self.accumulate_tile(
                            x,
                            grad_output,
                            cu_seqlens,
                            initial_state,
                            weights,
                            accumulators,
                            Int32(channel_group),
                            Int32(time_block),
                            Int32(batch),
                            active_endpoint,
                        )
            else:
                self.accumulate_tile(
                    x,
                    grad_output,
                    cu_seqlens,
                    initial_state,
                    weights,
                    accumulators,
                    Int32(channel_group),
                    Int32(worker),
                    Int32(batch),
                    Int32(self.tokens),
                )

            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width):
                    partials[batch, worker, channel + channel_offset, tap] = accumulators[
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
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            x,
            weight,
            grad_output,
            partials,
            cu_seqlens,
            initial_state,
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                (
                    self.time_workers
                    if cutlass.const_expr(self.time_workers > 0)
                    else cute.ceil_div(self.tokens, self.times_per_block)
                ),
                self.batches,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class ShortConvTmaKernel:
    """Share the dense two-input TMA stage layout and issue protocol."""

    stages = 2
    tma_stage_tokens = 8

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
        channel_tile = config.threads * config.channels_per_thread
        assert channels % channel_tile == 0, (
            f"TMA channels ({channels}) must be divisible by the channel tile ({channel_tile})"
        )
        super().__init__(batches, tokens, channels, width, config, dtype, d_activation)

    def staged_layout(self):
        channels_per_block = self.threads * self.channels_per_thread
        return cute.make_layout(
            (self.tma_stage_tokens, channels_per_block, self.stages),
            stride=(
                channels_per_block,
                1,
                self.tma_stage_tokens * channels_per_block,
            ),
        )

    @cute.jit
    def load_input_taps(self, history: cute.Tensor, current: cute.Tensor):
        """Append the current per-channel fragment to the FP32 convolution window."""
        input_taps = cute.make_rmem_tensor(
            (self.channels_per_thread, self.width),
            Float32,
        )
        if cutlass.const_expr(self.dtype.name == "fp32"):
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width - 1):
                    input_taps[channel_offset, tap] = history[channel_offset, tap].to(Float32)
                input_taps[channel_offset, self.width - 1] = current[channel_offset].to(Float32)
        else:
            for tap in cutlass.range_constexpr(self.width - 1):
                input_taps[(None, tap)].store(history[(None, tap)].load().to(Float32))
            input_taps[(None, self.width - 1)].store(current.load().to(Float32))
        return input_taps

    @cute.jit
    def advance_history(self, history: cute.Tensor, current: cute.Tensor):
        """Shift a raw-input register window and append one channel fragment."""
        if cutlass.const_expr(self.dtype.name == "fp32"):
            for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                for tap in cutlass.range_constexpr(self.width - 2):
                    history[channel_offset, tap] = history[channel_offset, tap + 1]
                history[channel_offset, self.width - 2] = current[channel_offset]
        else:
            for tap in cutlass.range_constexpr(self.width - 2):
                history[(None, tap)].store(history[(None, tap + 1)].load())
            history[(None, self.width - 2)].store(current.load())

    @cute.jit
    def initialize_history(
        self,
        history: cute.Tensor,
        x: cute.Tensor,
        initial_state: cute.Tensor | None,
        sequence: Int32,
        sequence_start: Int32,
        output_time: Int32,
        batch: Int32,
        channel_group: Int32,
    ):
        """Initialize a convolution window from physical input, state, or zeros."""
        history.fill(self.dtype.cute_type(0.0))
        x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
        initial_groups = initial_state
        if cutlass.const_expr(initial_state is not None):
            initial_groups = cute.zipped_divide(initial_state, (1, self.channels_per_thread))
        for history_offset in cutlass.range_constexpr(self.width - 1):
            history_time = output_time + history_offset - (self.width - 1)
            if history_time >= sequence_start:
                history[(None, history_offset)].store(
                    x_groups[
                        ((0, None), (batch * self.tokens + history_time, channel_group))
                    ].load()
                )
            elif cutlass.const_expr(initial_state is not None):
                history[(None, history_offset)].store(
                    load_history(
                        initial_groups,
                        sequence,
                        history_time,
                        sequence_start,
                        channel_group,
                        self.width,
                    ).to(self.dtype.cute_type)
                )

    @cute.jit
    def make_pipeline(
        self,
        tidx: Int32,
        batch: Int32,
        tma_atom_x: cute.CopyAtom,
        tma_tensor_x: cute.Tensor,
        tma_atom_dy: cute.CopyAtom,
        tma_tensor_dy: cute.Tensor,
    ):
        """Allocate and partition the shared two-input TMA pipeline."""
        channels_per_block = self.threads * self.channels_per_thread
        smem = cutlass.utils.SmemAllocator()
        barriers = smem.allocate_array(Int64, self.stages * 2)
        sX = smem.allocate_tensor(
            self.dtype.cute_type,
            self.staged_layout(),
            byte_alignment=128,
        )
        sD = smem.allocate_tensor(
            self.dtype.cute_type,
            self.staged_layout(),
            byte_alignment=128,
        )
        tile = cute.make_layout(
            (self.tma_stage_tokens, channels_per_block),
            stride=(channels_per_block, 1),
        )
        tile_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.threads // 32,
            ),
            tx_count=2 * cute.size_in_bytes(self.dtype.cute_type, tile),
            barrier_storage=barriers,
            tidx=tidx,
        )

        cta_tiler = (self.tma_stage_tokens, channels_per_block)
        gX_tiles = cute.local_tile(tma_tensor_x, cta_tiler, (None, None, batch))
        gD_tiles = cute.local_tile(tma_tensor_dy, cta_tiler, (None, None, batch))
        tXsX, tXgX = cpasync.tma_partition(
            tma_atom_x,
            0,
            cute.make_layout(1),
            cute.group_modes(sX, 0, 2),
            cute.group_modes(gX_tiles, 0, 2),
        )
        tDsD, tDgD = cpasync.tma_partition(
            tma_atom_dy,
            0,
            cute.make_layout(1),
            cute.group_modes(sD, 0, 2),
            cute.group_modes(gD_tiles, 0, 2),
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.stages,
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.stages,
        )
        return (
            tile_pipeline,
            producer_state,
            consumer_state,
            sX,
            sD,
            tXsX,
            tXgX,
            tDsD,
            tDgD,
        )

    @cute.jit
    def make_tma_atoms(self, x: cute.Tensor, grad_output: cute.Tensor):
        """Build matching per-batch TMA load atoms for input and output-gradient tiles."""
        staged = self.staged_layout()
        cta_tiler = (
            self.tma_stage_tokens,
            self.threads * self.channels_per_thread,
        )
        # Keep the batch mode separate so a partial trailing time box clamps
        # (zero-fills) inside its own batch instead of straddling batch rows.
        batched = cute.make_layout(
            (self.tokens, self.channels, self.batches),
            stride=(self.channels, 1, self.tokens * self.channels),
        )
        load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_x, tma_tensor_x = cpasync.make_tiled_tma_atom(
            load_op,
            cute.make_tensor(x.iterator, batched),
            staged,
            cta_tiler,
        )
        tma_atom_dy, tma_tensor_dy = cpasync.make_tiled_tma_atom(
            load_op,
            cute.make_tensor(grad_output.iterator, batched),
            staged,
            cta_tiler,
        )
        return tma_atom_x, tma_tensor_x, tma_atom_dy, tma_tensor_dy

    @cute.jit
    def issue_stage(
        self,
        tile_pipeline: pipeline.PipelineTmaAsync,
        producer_state: pipeline.PipelineState,
        tma_atom_x: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        tma_atom_dy: cute.CopyAtom,
        tDgD: cute.Tensor,
        tDsD: cute.Tensor,
        time_tile: Int32,
        channel_tile: Int32,
    ):
        tile_pipeline.producer_acquire(producer_state)
        barrier = tile_pipeline.producer_get_barrier(producer_state)
        cute.copy(
            tma_atom_x,
            tXgX[(None, time_tile, channel_tile)],
            tXsX[(None, producer_state.index)],
            tma_bar_ptr=barrier,
        )
        cute.copy(
            tma_atom_dy,
            tDgD[(None, time_tile, channel_tile)],
            tDsD[(None, producer_state.index)],
            tma_bar_ptr=barrier,
        )


class CausalConv1dSiluInputGradientTma(
    ShortConvTmaKernel,
    CausalConv1dSiluInputGradient,
):
    """Stream boundary-aware input gradients from TMA-staged physical tiles."""

    @cute.jit
    def emit_input_gradient(
        self,
        grad_z: cute.Tensor,
        weights: cute.Tensor,
        grad_x_groups: cute.Tensor,
        batch: Int32,
        channel_group: Int32,
        output_time: Int32,
        time_start: Int32,
        sequence_start: Int32,
        packed: cutlass.Constexpr,
    ):
        """Reduce one completed input gradient without crossing sequence or CTA ownership."""
        products = cute.make_rmem_tensor(
            (self.channels_per_thread, self.width),
            Float32,
        )
        for future_offset in cutlass.range_constexpr(self.width):
            products[(None, future_offset)].store(
                grad_z[(None, future_offset)].load()
                * weights[(None, self.width - 1 - future_offset)].load()
            )
        dx_value = products.load().reduce(
            cute.ReductionOp.ADD,
            Float32(0.0),
            reduction_profile=(None, 1),
        )
        input_time = output_time - (self.width - 1)
        owns_input = input_time >= time_start and input_time < self.tokens
        if cutlass.const_expr(packed):
            owns_input = owns_input and input_time >= sequence_start
        if owns_input:
            grad_x_groups[((0, None), (batch * self.tokens + input_time, channel_group))].store(
                dx_value.to(self.dtype.cute_type)
            )

    @cute.jit
    def flush_mainloop_sequence(
        self,
        grad_z: cute.Tensor,
        weights: cute.Tensor,
        grad_x_groups: cute.Tensor,
        batch: Int32,
        channel_group: Int32,
        boundary_time: Int32,
        time_start: Int32,
        sequence_start: Int32,
    ):
        """Complete a sequence ending inside the CTA's main output tile."""
        for flush in cutlass.range_constexpr(self.width - 1):
            for tap in cutlass.range_constexpr(self.width - 1):
                grad_z[(None, tap)].store(grad_z[(None, tap + 1)].load())
            grad_z[(None, self.width - 1)].fill(Float32(0.0))
            self.emit_input_gradient(
                grad_z,
                weights,
                grad_x_groups,
                batch,
                channel_group,
                boundary_time + flush,
                time_start,
                sequence_start,
                True,
            )

    @cute.jit
    def advance_token(
        self,
        current: cute.Tensor,
        incoming: cute.Tensor,
        weights: cute.Tensor,
        history: cute.Tensor,
        grad_z: cute.Tensor,
        grad_x_groups: cute.Tensor,
        batch: Int32,
        channel_group: Int32,
        output_time: Int32,
        time_start: Int32,
        sequence_start: Int32,
        packed: cutlass.Constexpr,
        emit_gradient: cutlass.Constexpr,
    ):
        """Advance one convolution window and optionally emit its completed input gradient."""
        input_taps = self.load_input_taps(history, current)
        value = (input_taps.load() * weights.load()).reduce(
            cute.ReductionOp.ADD,
            Float32(0.0),
            reduction_profile=(None, 1),
        )
        for tap in cutlass.range_constexpr(self.width - 1):
            grad_z[(None, tap)].store(grad_z[(None, tap + 1)].load())
        grad_z[(None, self.width - 1)].store(incoming.load() * self.d_activation(value))

        if cutlass.const_expr(emit_gradient):
            self.emit_input_gradient(
                grad_z,
                weights,
                grad_x_groups,
                batch,
                channel_group,
                output_time,
                time_start,
                sequence_start,
                packed,
            )

        self.advance_history(history, current)

    @cute.jit
    def run_mainloop(
        self,
        tile_pipeline: pipeline.PipelineTmaAsync,
        producer_state: pipeline.PipelineState,
        consumer_state: pipeline.PipelineState,
        sX: cute.Tensor,
        sD: cute.Tensor,
        tma_atom_x: cute.CopyAtom,
        tXgX: cute.Tensor,
        tXsX: cute.Tensor,
        tma_atom_dy: cute.CopyAtom,
        tDgD: cute.Tensor,
        tDsD: cute.Tensor,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        grad_x: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        skip_sequence_boundaries: cutlass.Constexpr,
    ):
        """Run the staged recurrence with an optional single-sequence fast path."""
        tidx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        channel_group = channel_block * self.threads + tidx
        time_start = time_block * self.times_per_block
        stage_tokens = self.tma_stage_tokens
        subtiles = self.times_per_block // stage_tokens
        first_time_tile = Int32(time_start // stage_tokens)

        x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
        dy_groups = cute.zipped_divide(grad_output, (1, self.channels_per_thread))
        dx_groups = cute.zipped_divide(grad_x, (1, self.channels_per_thread))
        sX_groups = cute.zipped_divide(sX, (1, self.channels_per_thread, 1))
        sD_groups = cute.zipped_divide(sD, (1, self.channels_per_thread, 1))
        weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
        history = cute.make_rmem_tensor(
            (self.channels_per_thread, self.width - 1),
            self.dtype.cute_type,
        )
        grad_z = cute.make_rmem_tensor(
            (self.channels_per_thread, self.width),
            Float32,
        )
        history.fill(self.dtype.cute_type(0.0))
        grad_z.fill(Float32(0.0))
        sequence, sequence_start, sequence_end = tile_sequence_bounds(
            cu_seqlens,
            Int32(time_start),
            Int32(batch),
            self.tokens,
        )
        weight_tile = cute.local_tile(
            weight,
            (self.channels_per_thread, self.width),
            (channel_group, 0),
        )
        weights.store(weight_tile.load().to(Float32))
        if cutlass.const_expr(initial_state is None):
            for history_offset in cutlass.range_constexpr(self.width - 1):
                history_time = time_start + history_offset - (self.width - 1)
                if history_time >= sequence_start:
                    history[(None, history_offset)].store(
                        x_groups[
                            ((0, None), (batch * self.tokens + history_time, channel_group))
                        ].load()
                    )
        else:
            self.initialize_history(
                history,
                x,
                initial_state,
                sequence,
                sequence_start,
                Int32(time_start),
                Int32(batch),
                Int32(channel_group),
            )

        for subtile in cutlass.range_constexpr(subtiles):
            tile_pipeline.consumer_wait(consumer_state)
            stage = consumer_state.index
            for slot in cutlass.range_constexpr(stage_tokens):
                output_offset = subtile * stage_tokens + slot
                output_time = time_start + output_offset
                current = cute.make_rmem_tensor(
                    (self.channels_per_thread,),
                    self.dtype.cute_type,
                )
                incoming = cute.make_rmem_tensor(
                    (self.channels_per_thread,),
                    Float32,
                )
                current.fill(self.dtype.cute_type(0.0))
                incoming.fill(Float32(0.0))
                if output_time < self.tokens:
                    if cutlass.const_expr(cu_seqlens is not None):  # noqa: SIM102
                        if cutlass.const_expr(not skip_sequence_boundaries):  # noqa: SIM102
                            # Once-only flush: must be "=="; see NOTE [Boundary trigger forms].
                            if output_time == sequence_end:
                                previous_sequence_start = sequence_start
                                sequence, sequence_start, sequence_end = sequence_bounds(
                                    cu_seqlens,
                                    Int32(output_time),
                                )
                                self.flush_mainloop_sequence(
                                    grad_z,
                                    weights,
                                    dx_groups,
                                    batch,
                                    channel_group,
                                    Int32(output_time),
                                    Int32(time_start),
                                    previous_sequence_start,
                                )
                                if output_time < sequence_end:
                                    if cutlass.const_expr(initial_state is None):
                                        history.fill(self.dtype.cute_type(0.0))
                                    else:
                                        self.initialize_history(
                                            history,
                                            x,
                                            initial_state,
                                            sequence,
                                            sequence_start,
                                            Int32(output_time),
                                            Int32(batch),
                                            Int32(channel_group),
                                        )
                                else:
                                    # Keep inactive recurrence steps from overwriting the flushed prefix.
                                    sequence_start = Int32(output_time)
                                grad_z.fill(Float32(0.0))
                    current.store(sX_groups[((0, None, 0), (slot, tidx, stage))].load())
                    incoming.store(
                        sD_groups[((0, None, 0), (slot, tidx, stage))].load().to(Float32)
                    )
                self.advance_token(
                    current,
                    incoming,
                    weights,
                    history,
                    grad_z,
                    dx_groups,
                    batch,
                    channel_group,
                    output_time,
                    Int32(time_start),
                    sequence_start,
                    cu_seqlens is not None,
                    output_offset >= self.width - 1,
                )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tile_pipeline.consumer_release(consumer_state)
            consumer_state.advance()
            if cutlass.const_expr(subtile + self.stages < subtiles) and warp_idx == 0:
                self.issue_stage(
                    tile_pipeline,
                    producer_state,
                    tma_atom_x,
                    tXgX,
                    tXsX,
                    tma_atom_dy,
                    tDgD,
                    tDsD,
                    first_time_tile + Int32(subtile + self.stages),
                    Int32(channel_block),
                )
                producer_state.advance()

        # A boundary in the lookahead belongs to the next CTA; zero-fill this CTA's tail.
        for tail in cutlass.range_constexpr(self.width - 1):
            output_offset = self.times_per_block + tail
            output_time = time_start + output_offset
            current = cute.make_rmem_tensor(
                (self.channels_per_thread,),
                self.dtype.cute_type,
            )
            incoming = cute.make_rmem_tensor(
                (self.channels_per_thread,),
                Float32,
            )
            current.fill(self.dtype.cute_type(0.0))
            incoming.fill(Float32(0.0))
            has_input = output_time < self.tokens
            if cutlass.const_expr(cu_seqlens is not None):
                has_input = has_input and output_time < sequence_end
            if has_input:
                current.store(
                    x_groups[
                        ((0, None), (batch * self.tokens + output_time, channel_group))
                    ].load()
                )
                incoming.store(
                    dy_groups[((0, None), (batch * self.tokens + output_time, channel_group))]
                    .load()
                    .to(Float32)
                )
            self.advance_token(
                current,
                incoming,
                weights,
                history,
                grad_z,
                dx_groups,
                batch,
                channel_group,
                output_time,
                Int32(time_start),
                sequence_start,
                cu_seqlens is not None,
                True,
            )
        return producer_state, consumer_state

    @cute.kernel
    def tma_kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        grad_x: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        tma_atom_x: cute.CopyAtom,
        tma_tensor_x: cute.Tensor,
        tma_atom_dy: cute.CopyAtom,
        tma_tensor_dy: cute.Tensor,
    ):
        """Compute boundary-aware input gradients while retaining one convolution window."""
        tidx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        time_start = time_block * self.times_per_block
        stage_tokens = self.tma_stage_tokens

        (
            tile_pipeline,
            producer_state,
            consumer_state,
            sX,
            sD,
            tXsX,
            tXgX,
            tDsD,
            tDgD,
        ) = self.make_pipeline(
            tidx,
            Int32(batch),
            tma_atom_x,
            tma_tensor_x,
            tma_atom_dy,
            tma_tensor_dy,
        )
        first_time_tile = Int32(time_start // stage_tokens)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_x)
            cpasync.prefetch_descriptor(tma_atom_dy)
            for issued in cutlass.range_constexpr(self.stages):
                self.issue_stage(
                    tile_pipeline,
                    producer_state,
                    tma_atom_x,
                    tXgX,
                    tXsX,
                    tma_atom_dy,
                    tDgD,
                    tDsD,
                    first_time_tile + Int32(issued),
                    Int32(channel_block),
                )
                producer_state.advance()

        if cutlass.const_expr(cu_seqlens is None):
            producer_state, consumer_state = self.run_mainloop(
                tile_pipeline,
                producer_state,
                consumer_state,
                sX,
                sD,
                tma_atom_x,
                tXgX,
                tXsX,
                tma_atom_dy,
                tDgD,
                tDsD,
                x,
                weight,
                grad_output,
                grad_x,
                cu_seqlens,
                initial_state,
                False,
            )
        elif cutlass.const_expr(cute.size(cu_seqlens) == 2):
            active_endpoint = Int32(cu_seqlens[1])
            if active_endpoint == self.tokens:
                producer_state, consumer_state = self.run_mainloop(
                    tile_pipeline,
                    producer_state,
                    consumer_state,
                    sX,
                    sD,
                    tma_atom_x,
                    tXgX,
                    tXsX,
                    tma_atom_dy,
                    tDgD,
                    tDsD,
                    x,
                    weight,
                    grad_output,
                    grad_x,
                    cu_seqlens,
                    initial_state,
                    True,
                )
            else:
                producer_state, consumer_state = self.run_mainloop(
                    tile_pipeline,
                    producer_state,
                    consumer_state,
                    sX,
                    sD,
                    tma_atom_x,
                    tXgX,
                    tXsX,
                    tma_atom_dy,
                    tDgD,
                    tDsD,
                    x,
                    weight,
                    grad_output,
                    grad_x,
                    cu_seqlens,
                    initial_state,
                    False,
                )
        else:
            producer_state, consumer_state = self.run_mainloop(
                tile_pipeline,
                producer_state,
                consumer_state,
                sX,
                sD,
                tma_atom_x,
                tXgX,
                tXsX,
                tma_atom_dy,
                tDgD,
                tDsD,
                x,
                weight,
                grad_output,
                grad_x,
                cu_seqlens,
                initial_state,
                False,
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
        """Build TMA descriptors and launch the boundary-aware streaming specialization."""
        tma_atom_x, tma_tensor_x, tma_atom_dy, tma_tensor_dy = self.make_tma_atoms(
            x,
            grad_output,
        )
        self.tma_kernel.set_name_prefix(self.get_name())
        self.tma_kernel(
            x,
            weight,
            grad_output,
            grad_x,
            cu_seqlens,
            initial_state,
            tma_atom_x,
            tma_tensor_x,
            tma_atom_dy,
            tma_tensor_dy,
        ).launch(
            grid=(
                self.channels // (self.threads * self.channels_per_thread),
                cute.ceil_div(self.tokens, self.times_per_block),
                self.batches,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluWeightGradientPartialsTma(
    ShortConvTmaKernel,
    CausalConv1dSiluWeightGradientPartials,
):
    """Stage physical input and output-gradient tiles through a two-stage TMA pipeline."""

    @cute.kernel
    def tma_kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        grad_output: cute.Tensor,
        partials: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        initial_state: cute.Tensor | None,
        tma_atom_x: cute.CopyAtom,
        tma_tensor_x: cute.Tensor,
        tma_atom_dy: cute.CopyAtom,
        tma_tensor_dy: cute.Tensor,
    ):
        """Compute boundary-aware weight-gradient partials from asynchronously staged tiles."""
        tidx, _, _ = cute.arch.thread_idx()
        channel_block, time_block, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        channel_group = channel_block * self.threads + tidx
        channel = channel_group * self.channels_per_thread
        time_start = time_block * self.times_per_block
        stage_tokens = self.tma_stage_tokens
        subtiles = self.times_per_block // stage_tokens

        (
            tile_pipeline,
            producer_state,
            consumer_state,
            sX,
            sD,
            tXsX,
            tXgX,
            tDsD,
            tDgD,
        ) = self.make_pipeline(
            tidx,
            Int32(batch),
            tma_atom_x,
            tma_tensor_x,
            tma_atom_dy,
            tma_tensor_dy,
        )
        first_time_tile = Int32(time_start // stage_tokens)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_x)
            cpasync.prefetch_descriptor(tma_atom_dy)
            for issued in cutlass.range_constexpr(self.stages):
                self.issue_stage(
                    tile_pipeline,
                    producer_state,
                    tma_atom_x,
                    tXgX,
                    tXsX,
                    tma_atom_dy,
                    tDgD,
                    tDsD,
                    first_time_tile + Int32(issued),
                    Int32(channel_block),
                )
                producer_state.advance()

        x_groups = cute.zipped_divide(x, (1, self.channels_per_thread))
        sX_groups = cute.zipped_divide(sX, (1, self.channels_per_thread, 1))
        sD_groups = cute.zipped_divide(sD, (1, self.channels_per_thread, 1))
        weights = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
        accumulators = cute.make_rmem_tensor((self.channels_per_thread, self.width), Float32)
        history = cute.make_rmem_tensor(
            (self.channels_per_thread, self.width - 1),
            self.dtype.cute_type,
        )
        accumulators.fill(Float32(0.0))
        history.fill(self.dtype.cute_type(0.0))
        sequence, sequence_start, sequence_end = tile_sequence_bounds(
            cu_seqlens,
            Int32(time_start),
            Int32(batch),
            self.tokens,
        )
        weight_tile = cute.local_tile(
            weight,
            (self.channels_per_thread, self.width),
            (channel_group, 0),
        )
        weights.store(weight_tile.load().to(Float32))
        if cutlass.const_expr(initial_state is None):
            for history_offset in cutlass.range_constexpr(self.width - 1):
                history_time = time_start + history_offset - (self.width - 1)
                if history_time >= sequence_start:
                    if cutlass.const_expr(self.dtype.name == "fp32"):
                        for channel_offset in cutlass.range_constexpr(self.channels_per_thread):
                            history[channel_offset, history_offset] = x[
                                batch * self.tokens + history_time,
                                channel + channel_offset,
                            ]
                    else:
                        history[(None, history_offset)].store(
                            x_groups[
                                (
                                    (0, None),
                                    (batch * self.tokens + history_time, channel_group),
                                )
                            ].load()
                        )
        else:
            self.initialize_history(
                history,
                x,
                initial_state,
                sequence,
                sequence_start,
                Int32(time_start),
                Int32(batch),
                Int32(channel_group),
            )

        for subtile in cutlass.range_constexpr(subtiles):
            tile_pipeline.consumer_wait(consumer_state)
            stage = consumer_state.index
            for slot in cutlass.range_constexpr(stage_tokens):
                time = time_start + subtile * stage_tokens + slot
                if time < self.tokens:
                    active_time = True
                    if cutlass.const_expr(cu_seqlens is not None):
                        # "==" skips the inactive-tail re-lookup; legal because the
                        # walk is unit-stride -- see NOTE [Boundary trigger forms].
                        if time == sequence_end:
                            sequence, sequence_start, sequence_end = sequence_bounds(
                                cu_seqlens,
                                Int32(time),
                            )
                            if time < sequence_end:
                                if cutlass.const_expr(initial_state is None):
                                    history.fill(self.dtype.cute_type(0.0))
                                else:
                                    self.initialize_history(
                                        history,
                                        x,
                                        initial_state,
                                        sequence,
                                        sequence_start,
                                        Int32(time),
                                        Int32(batch),
                                        Int32(channel_group),
                                    )
                        active_time = time < sequence_end
                    if active_time:
                        current = cute.make_rmem_tensor(
                            (self.channels_per_thread,),
                            self.dtype.cute_type,
                        )
                        if cutlass.const_expr(self.dtype.name == "fp32"):
                            for channel_offset in cutlass.range_constexpr(
                                self.channels_per_thread
                            ):
                                current[channel_offset] = sX[
                                    slot,
                                    tidx * self.channels_per_thread + channel_offset,
                                    stage,
                                ]
                        else:
                            current.store(sX_groups[((0, None, 0), (slot, tidx, stage))].load())
                        input_taps = self.load_input_taps(history, current)
                        value = (input_taps.load() * weights.load()).reduce(
                            cute.ReductionOp.ADD,
                            Float32(0.0),
                            reduction_profile=(None, 1),
                        )
                        incoming = cute.make_rmem_tensor(
                            (self.channels_per_thread,),
                            Float32,
                        )
                        if cutlass.const_expr(self.dtype.name == "fp32"):
                            for channel_offset in cutlass.range_constexpr(
                                self.channels_per_thread
                            ):
                                incoming[channel_offset] = sD[
                                    slot,
                                    tidx * self.channels_per_thread + channel_offset,
                                    stage,
                                ].to(Float32)
                        else:
                            incoming.store(
                                sD_groups[((0, None, 0), (slot, tidx, stage))].load().to(Float32)
                            )
                        grad_z = incoming.load() * self.d_activation(value)
                        for tap in cutlass.range_constexpr(self.width):
                            accumulators[(None, tap)].store(
                                accumulators[(None, tap)].load()
                                + grad_z * input_taps[(None, tap)].load()
                            )
                        self.advance_history(history, current)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tile_pipeline.consumer_release(consumer_state)
            consumer_state.advance()

            if cutlass.const_expr(subtile + self.stages < subtiles) and warp_idx == 0:
                self.issue_stage(
                    tile_pipeline,
                    producer_state,
                    tma_atom_x,
                    tXgX,
                    tXsX,
                    tma_atom_dy,
                    tDgD,
                    tDsD,
                    first_time_tile + Int32(subtile + self.stages),
                    Int32(channel_block),
                )
                producer_state.advance()

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
        """Build TMA descriptors and launch the boundary-aware staged specialization."""
        tma_atom_x, tma_tensor_x, tma_atom_dy, tma_tensor_dy = self.make_tma_atoms(
            x,
            grad_output,
        )
        self.tma_kernel.set_name_prefix(self.get_name())
        self.tma_kernel(
            x,
            weight,
            grad_output,
            partials,
            cu_seqlens,
            initial_state,
            tma_atom_x,
            tma_tensor_x,
            tma_atom_dy,
            tma_tensor_dy,
        ).launch(
            grid=(
                self.channels // (self.threads * self.channels_per_thread),
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
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            x,
            weight,
            initial_state,
            grad_output,
            grad_initial_state,
            cu_seqlens,
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
    num_sequences: int | None,
    has_initial_state: bool,
    activation: Activation,
    time_workers: int,
):
    """Compile one dense, packed-static, or packed-persistent forward specialization."""
    activation_fn = activation.forward
    operation = CausalConv1dSiluForward(
        batches,
        tokens,
        channels,
        width,
        config,
        dtype,
        activation_fn,
        time_workers,
    )
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


def tuned_config(dtype: ShortConvDType, *, packed: bool = False) -> ShortConvTunedConfig:
    """Resolve measured defaults from a compile-time storage descriptor."""
    match dtype.name:
        case "fp16":
            return ShortConvTunedConfig.default(torch.float16, packed=packed)
        case "bf16":
            return ShortConvTunedConfig.default(torch.bfloat16, packed=packed)
        case "fp32":
            return ShortConvTunedConfig.default(torch.float32, packed=packed)
        case _:
            raise ValueError(f"unsupported short-convolution dtype {dtype.name}")


def supports_tma(
    operation_type: type[ShortConvTmaKernel],
    config: ShortConvConfig,
    selected_config: ShortConvConfig | None,
    channels: int,
    width: int,
) -> bool:
    """Return whether the TMA gradient kernels support this static schedule.

    TMA zero-fills out-of-bounds token rows, so any token count works; only the
    schedule's channel and stage divisibility matter.
    """
    return (
        config == selected_config
        and width > 1
        and channels % (config.threads * config.channels_per_thread) == 0
        and config.times_per_block % operation_type.tma_stage_tokens == 0
    )


def _input_gradient_uses_tma(
    dtype: ShortConvDType,
    config: ShortConvConfig,
    num_sequences: int | None,
    channels: int,
    width: int,
) -> bool:
    """Select the unchanged staged input-gradient path when its schedule supports it."""
    return supports_tma(
        CausalConv1dSiluInputGradientTma,
        config,
        None
        if dtype.name == "fp32" and num_sequences is None
        else tuned_config(dtype, packed=num_sequences is not None).input_gradient,
        channels,
        width,
    )


def _weight_gradient_uses_tma(
    dtype: ShortConvDType,
    config: ShortConvConfig,
    num_sequences: int | None,
    channels: int,
    width: int,
) -> bool:
    """Select the unchanged staged weight-gradient path when its schedule supports it."""
    return supports_tma(
        CausalConv1dSiluWeightGradientPartialsTma,
        config,
        tuned_config(dtype, packed=num_sequences is not None).weight_gradient,
        channels,
        width,
    )


@jit_cache
def _compile_input_gradient(
    batches: int,
    tokens: int,
    channels: int,
    width: int,
    dtype: ShortConvDType,
    config: ShortConvConfig,
    num_sequences: int | None,
    has_initial_state: bool,
    activation: Activation,
    time_workers: int,
):
    """Compile one static, TMA, or packed-persistent input-gradient specialization."""
    derivative = activation.derivative
    use_tma = _input_gradient_uses_tma(dtype, config, num_sequences, channels, width)
    if use_tma:
        assert time_workers == 0
        operation = CausalConv1dSiluInputGradientTma(
            batches, tokens, channels, width, config, dtype, derivative
        )
    else:
        operation = CausalConv1dSiluInputGradient(
            batches,
            tokens,
            channels,
            width,
            config,
            dtype,
            derivative,
            time_workers,
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
    num_sequences: int | None,
    has_initial_state: bool,
    activation: Activation,
    time_workers: int,
):
    """Compile one static, TMA, or packed-persistent weight-gradient specialization."""
    derivative = activation.derivative
    use_tma = _weight_gradient_uses_tma(dtype, config, num_sequences, channels, width)
    num_partials = time_workers or ceildiv(tokens, config.times_per_block)
    partials = cute.runtime.make_fake_compact_tensor(
        Float32,
        (batches, num_partials, channels, width),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    if use_tma:
        assert time_workers == 0
        operation = CausalConv1dSiluWeightGradientPartialsTma(
            batches, tokens, channels, width, config, dtype, derivative
        )
    else:
        operation = CausalConv1dSiluWeightGradientPartials(
            batches,
            tokens,
            channels,
            width,
            config,
            dtype,
            derivative,
            time_workers,
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
    num_sequences: int | None,
    activation: Activation,
):
    """Compile one initial-state gradient specialization."""
    derivative = activation.derivative
    state_sequences = batches if num_sequences is None else num_sequences
    config = ShortConvConfig(threads, channels_per_thread, times_per_block=1)
    operation = CausalConv1dSiluInitialStateGradient(
        state_sequences, tokens, channels, width, config, dtype, derivative
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


def _validate_decode_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
) -> None:
    """Validate the one-token in-place decode tensor contract."""
    if x.ndim != 2:
        raise ValueError(f"x must have shape [num_sequences, C], got {tuple(x.shape)}")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(
            f"x must have positive sequence and channel dimensions, got {tuple(x.shape)}"
        )
    if x.dtype not in SHORT_CONV_DTYPES or not x.is_cuda or not x.is_contiguous():
        raise ValueError("x must be a contiguous CUDA FP16, BF16, or FP32 tensor")
    if weight.ndim != 2 or weight.shape[0] != x.shape[1] or weight.shape[1] < 2:
        raise ValueError(
            f"weight must have shape [{x.shape[1]}, W] with W >= 2, got {tuple(weight.shape)}"
        )
    if weight.dtype != x.dtype or weight.device != x.device or not weight.is_contiguous():
        raise ValueError("weight must match x dtype and be contiguous on x.device")

    sequences, channels = x.shape
    slots = sequences if state_indices is None else (state.shape[0] if state.ndim == 3 else -1)
    expected_state = (slots, weight.shape[1] - 1, channels)
    if slots < 1 or state.ndim != 3 or state.shape != expected_state:
        raise ValueError(f"state must have shape {expected_state}, got {tuple(state.shape)}")
    if state.dtype != x.dtype or state.device != x.device or not state.is_contiguous():
        raise ValueError("state must match x dtype and be contiguous on x.device")

    if state_indices is not None and (
        tuple(state_indices.shape) != (sequences,)
        or state_indices.dtype != torch.int32
        or state_indices.device != x.device
        or not state_indices.is_contiguous()
    ):
        raise ValueError(
            f"state_indices must be contiguous int32 with shape ({sequences},) on x.device"
        )


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize the uncommon contiguous view that misses the launcher ABI alignment."""
    return tensor if tensor.data_ptr() % 16 == 0 else tensor.clone()


def _launch_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    config: ShortConvConfig,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    *,
    activation: str | None,
) -> torch.Tensor:
    """Allocate and launch the compiled forward specialization."""
    resolved_activation = resolve_activation(activation)
    x, weight = _aligned(x), _aligned(weight)
    if initial_state is not None:
        initial_state = _aligned(initial_state)
    batches, tokens, channels = x.shape
    width = weight.shape[1]
    dtype = SHORT_CONV_DTYPES[x.dtype]
    output = torch.empty_like(x)
    num_sequences = None if cu_seqlens is None else cu_seqlens.shape[0] - 1
    compiled = _compile_forward(
        batches,
        tokens,
        channels,
        width,
        dtype,
        config,
        num_sequences,
        initial_state is not None,
        resolved_activation,
        _persistent_time_workers(
            tokens,
            channels,
            config,
            x.device,
            persistent_eligible=num_sequences is not None,
        ),
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
    *,
    activation: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Launch configured gradient kernels and reduce FP32 partials."""
    resolved_activation = resolve_activation(activation)
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
        resolved_activation,
        _persistent_time_workers(
            tokens,
            channels,
            input_config,
            x.device,
            persistent_eligible=num_sequences is not None
            and not _input_gradient_uses_tma(dtype, input_config, num_sequences, channels, width),
        ),
    )(
        x.view(batches * tokens, channels),
        weight,
        grad_output.view(batches * tokens, channels),
        grad_x.view(batches * tokens, channels),
        cu_seqlens,
        None if initial_state is None else initial_state.flatten(0, 1),
    )

    weight_workers = _persistent_time_workers(
        tokens,
        channels,
        weight_config,
        x.device,
        persistent_eligible=num_sequences is not None
        and not _weight_gradient_uses_tma(dtype, weight_config, num_sequences, channels, width),
    )
    num_partials = weight_workers or ceildiv(tokens, weight_config.times_per_block)
    if initial_state is None or not compute_initial_state_grad:
        grad_initial_state = None
    elif width == 1:
        grad_initial_state = torch.zeros_like(initial_state)
    else:
        grad_initial_state = torch.empty_like(initial_state)
    partials = torch.empty(
        (batches, num_partials, channels, width),
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
        resolved_activation,
        weight_workers,
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
            resolved_activation,
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
    *,
    packed: bool = False,
) -> tuple[ShortConvConfig, ...]:
    """Return the focused schedule space used by the explicit tuning flow."""
    defaults = ShortConvTunedConfig.default(dtype, packed=packed)
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


def tune_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    *,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    forward_configs: Iterable[ShortConvConfig] | None = None,
    input_grad_configs: Iterable[ShortConvConfig] | None = None,
    weight_grad_configs: Iterable[ShortConvConfig] | None = None,
    parallel_compile: bool = True,
) -> ShortConvTunedConfig:
    """Compile and benchmark forward, input-gradient, and weight-gradient schedules.

    Width is an operation parameter and is specialized but not autotuned. Packed
    offsets, activation, and initial state use the same inputs accepted by the public
    operation. The returned configs can be passed directly to :func:`causal_conv1d`.
    Tuning uses the shared CuTeDSL ``tune`` flow: cached variants compile in
    parallel and execute sequentially under the vetted Inductor GPU benchmarker.
    """
    resolved_activation = resolve_activation(activation)
    # Script-defined (__main__) activations cannot be pickled into the fresh
    # compiler driver; compile those candidates in this process instead.
    parallel_compile = parallel_compile and resolved_activation.crosses_process_boundary
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
    packed = cu_seqlens is not None
    num_sequences = None if cu_seqlens is None else cu_seqlens.shape[0] - 1
    x_matrix = x.view(batches * tokens, channels)
    grad_matrix = grad_output.view(batches * tokens, channels)
    kernel_initial_state = None if width == 1 else initial_state
    state_matrix = None if kernel_initial_state is None else kernel_initial_state.flatten(0, 1)

    forward_candidates = tuple(
        _candidate_configs("forward", channels, x.dtype, packed=packed)
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
            resolved_activation,
            _persistent_time_workers(
                tokens,
                channels,
                config,
                x.device,
                persistent_eligible=num_sequences is not None,
            ),
        ),
        parallel_compile=parallel_compile,
    )

    input_candidates = tuple(
        _candidate_configs("input_gradient", channels, x.dtype, packed=packed)
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
            resolved_activation,
            _persistent_time_workers(
                tokens,
                channels,
                config,
                x.device,
                persistent_eligible=num_sequences is not None
                and not _input_gradient_uses_tma(dtype, config, num_sequences, channels, width),
            ),
        ),
        parallel_compile=parallel_compile,
    )

    weight_candidates = tuple(
        _candidate_configs("weight_gradient", channels, x.dtype, packed=packed)
        if weight_grad_configs is None
        else weight_grad_configs
    )
    for config in weight_candidates:
        _validate_config(config, channels, "weight_grad_configs")
    weight_workers = {
        config: _persistent_time_workers(
            tokens,
            channels,
            config,
            x.device,
            persistent_eligible=num_sequences is not None
            and not _weight_gradient_uses_tma(dtype, config, num_sequences, channels, width),
        )
        for config in weight_candidates
    }
    partials = {
        config: torch.empty(
            batches,
            workers or ceildiv(tokens, config.times_per_block),
            channels,
            width,
            dtype=torch.float32,
            device=x.device,
        )
        for config, workers in weight_workers.items()
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
            resolved_activation,
            weight_workers[config],
        ),
        parallel_compile=parallel_compile,
    )
    return ShortConvTunedConfig(forward, input_gradient, weight_gradient)


def _config(threads: int, channels_per_thread: int, times_per_block: int) -> ShortConvConfig:
    """Reconstruct a compile-time config from registered-operator scalar arguments."""
    return ShortConvConfig(threads, channels_per_thread, times_per_block)


def _fake_dynamic_rows(dtype: ShortConvDType, columns: int):
    """Create a row-major fake tensor whose row count binds at launch."""
    return cute.runtime.make_fake_compact_tensor(
        dtype.cute_type,
        (cute.sym_int32(), columns),
        stride_order=(1, 0),
        assumed_align=16,
    )


def _fake_state_indices(paged: bool):
    """Create runtime-bound slot indices or preserve the unpaged specialization."""
    if not paged:
        return None
    return cute.runtime.make_fake_compact_tensor(
        Int32,
        (cute.sym_int32(),),
        stride_order=(0,),
        assumed_align=4,
    )


@jit_cache
def _compile_decode(
    channels: int,
    width: int,
    dtype: ShortConvDType,
    config: ShortConvConfig,
    paged: bool,
    activation: Activation,
):
    """Compile one decode specialization, shared across batch sizes and pool depths."""
    operation = CausalConv1dSiluDecode(channels, width, config, dtype, activation.forward)
    return compile_tvm_ffi(
        operation,
        _fake_dynamic_rows(dtype, channels),
        _fake_matrix(dtype, channels, width),
        _fake_dynamic_rows(dtype, channels),
        _fake_dynamic_rows(dtype, channels),
        _fake_state_indices(paged),
    )


def _launch_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    config: ShortConvConfig,
    state_indices: torch.Tensor | None,
    *,
    activation: str | None,
) -> torch.Tensor:
    """Allocate the output and launch the compiled decode specialization."""
    resolved_activation = resolve_activation(activation)
    x, weight = _aligned(x), _aligned(weight)
    if state.data_ptr() % 16 != 0:
        raise ValueError("state storage must be 16-byte aligned for the in-place advance")
    channels = x.shape[1]
    width = weight.shape[1]
    _validate_config(config, channels, "forward_config")
    dtype = SHORT_CONV_DTYPES[x.dtype]
    output = torch.empty_like(x)
    compiled = _compile_decode(
        channels,
        width,
        dtype,
        config,
        state_indices is not None,
        resolved_activation,
    )
    compiled(x, weight, output, state.flatten(0, 1), state_indices)
    return output


def _cute_short_conv_fwd_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Launch the tuned K3 defaults through the lean production schema."""
    return _launch_forward(
        x,
        weight,
        ShortConvTunedConfig.default(x.dtype).forward,
        cu_seqlens,
        initial_state,
        activation=activation,
    )


torch.library.impl("attn_gym::_cute_short_conv_fwd", "CUDA", _cute_short_conv_fwd_cuda)


@torch.library.register_fake("attn_gym::_cute_short_conv_fwd")
def _default_forward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    del weight, cu_seqlens, initial_state, activation
    return torch.empty_like(x)


def _cute_short_conv_decode_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Launch the tuned forward defaults through the decode schema."""
    _validate_decode_inputs(x, weight, state, state_indices)
    config = _compatible_config(ShortConvTunedConfig.default(x.dtype).forward, x.shape[1])
    return _launch_decode(
        x,
        weight,
        state,
        config,
        state_indices,
        activation=activation,
    )


torch.library.impl("attn_gym::_cute_short_conv_decode", "CUDA", _cute_short_conv_decode_cuda)


@torch.library.register_fake("attn_gym::_cute_short_conv_decode")
def _decode_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    del weight, state, state_indices, activation
    return torch.empty_like(x)


def _cute_short_conv_configured_decode_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    forward_threads: int,
    forward_channels: int,
    forward_times: int,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Keep configured decode compilation and launch work behind an opaque operator."""
    _validate_decode_inputs(x, weight, state, state_indices)
    return _launch_decode(
        x,
        weight,
        state,
        _config(forward_threads, forward_channels, forward_times),
        state_indices,
        activation=activation,
    )


torch.library.impl(
    "attn_gym::_cute_short_conv_configured_decode",
    "CUDA",
    _cute_short_conv_configured_decode_cuda,
)


@torch.library.register_fake("attn_gym::_cute_short_conv_configured_decode")
def _configured_decode_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    forward_threads: int,
    forward_channels: int,
    forward_times: int,
    *,
    activation: str | None = None,
) -> torch.Tensor:
    del (
        weight,
        state,
        state_indices,
        forward_threads,
        forward_channels,
        forward_times,
        activation,
    )
    return torch.empty_like(x)


def _cute_short_conv_bwd_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the tuned K3 backward defaults through the lean schema."""
    defaults = ShortConvTunedConfig.default(x.dtype, packed=cu_seqlens is not None)
    grad_x, grad_weight, _ = _launch_backward(
        x,
        weight,
        grad_output,
        defaults.input_gradient,
        defaults.weight_gradient,
        cu_seqlens,
        activation=activation,
    )
    return grad_x, grad_weight


torch.library.impl("attn_gym::_cute_short_conv_bwd", "CUDA", _cute_short_conv_bwd_cuda)


@torch.library.register_fake("attn_gym::_cute_short_conv_bwd")
def _default_backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    *,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_output, cu_seqlens, activation
    return torch.empty_like(x), torch.empty_like(weight)


def _cute_short_conv_configured_fwd_cuda(
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
    *,
    activation: str | None = None,
) -> torch.Tensor:
    """Keep the configured CuTeDSL forward launcher behind an opaque operator."""
    del input_threads, input_channels, input_times, weight_threads, weight_channels, weight_times
    return _launch_forward(
        x,
        weight,
        _config(forward_threads, forward_channels, forward_times),
        cu_seqlens,
        initial_state,
        activation=activation,
    )


torch.library.impl(
    "attn_gym::_cute_short_conv_configured_fwd", "CUDA", _cute_short_conv_configured_fwd_cuda
)


@torch.library.register_fake("attn_gym::_cute_short_conv_configured_fwd")
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
    *,
    activation: str | None = None,
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
        activation,
    )
    return torch.empty_like(x)


def _cute_short_conv_configured_bwd_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
    *,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep the configured first-order backward launchers opaque.

    ``initial_state`` remains an optional input for preactivation recomputation,
    but its gradient is never computed; use the ``_with_state_grad`` variant for that.
    """
    grad_x, grad_weight, _ = _launch_backward(
        x,
        weight,
        grad_output,
        _config(input_threads, input_channels, input_times),
        _config(weight_threads, weight_channels, weight_times),
        cu_seqlens,
        initial_state,
        compute_initial_state_grad=False,
        activation=activation,
    )
    return grad_x, grad_weight


def _cute_short_conv_configured_bwd_with_state_grad_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
    *,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the configured first-order backward including the state gradient."""
    grad_x, grad_weight, grad_initial_state = _launch_backward(
        x,
        weight,
        grad_output,
        _config(input_threads, input_channels, input_times),
        _config(weight_threads, weight_channels, weight_times),
        cu_seqlens,
        initial_state,
        compute_initial_state_grad=True,
        activation=activation,
    )
    return grad_x, grad_weight, grad_initial_state


torch.library.impl(
    "attn_gym::_cute_short_conv_configured_bwd", "CUDA", _cute_short_conv_configured_bwd_cuda
)
torch.library.impl(
    "attn_gym::_cute_short_conv_configured_bwd_with_state_grad",
    "CUDA",
    _cute_short_conv_configured_bwd_with_state_grad_cuda,
)


@torch.library.register_fake("attn_gym::_cute_short_conv_configured_bwd")
def _backward_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
    *,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe backward output metadata without invoking the compiler."""
    del (
        grad_output,
        cu_seqlens,
        initial_state,
        input_threads,
        input_channels,
        input_times,
        weight_threads,
        weight_channels,
        weight_times,
        activation,
    )
    return torch.empty_like(x), torch.empty_like(weight)


@torch.library.register_fake("attn_gym::_cute_short_conv_configured_bwd_with_state_grad")
def _backward_with_state_grad_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    initial_state: torch.Tensor,
    input_threads: int,
    input_channels: int,
    input_times: int,
    weight_threads: int,
    weight_channels: int,
    weight_times: int,
    *,
    activation: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe stateful backward output metadata without invoking the compiler."""
    del (
        grad_output,
        cu_seqlens,
        input_threads,
        input_channels,
        input_times,
        weight_threads,
        weight_channels,
        weight_times,
        activation,
    )
    return torch.empty_like(x), torch.empty_like(weight), torch.empty_like(initial_state)


class _ShortConv(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        activation: str | None,
    ) -> torch.Tensor:
        output = kda_ops.short_conv_forward_op(
            x, weight, cu_seqlens, initial_state, activation=activation
        )
        ctx.save_for_backward(x, weight, cu_seqlens, initial_state)
        ctx.activation = activation
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        x, weight, cu_seqlens, initial_state = ctx.saved_tensors
        activation = ctx.activation
        if initial_state is None:
            grad_x, grad_weight = kda_ops.short_conv_backward_op(
                x, weight, grad_output, cu_seqlens, activation=activation
            )
            grad_initial_state = None
        else:
            defaults = ShortConvTunedConfig.default(
                x.dtype,
                packed=cu_seqlens is not None,
            )
            input_config = defaults.input_gradient
            weight_config = defaults.weight_gradient
            configs = (
                input_config.threads,
                input_config.channels_per_thread,
                input_config.times_per_block,
                weight_config.threads,
                weight_config.channels_per_thread,
                weight_config.times_per_block,
            )
            if ctx.needs_input_grad[3]:
                grad_x, grad_weight, grad_initial_state = (
                    kda_ops.short_conv_configured_backward_with_state_grad_op(
                        x,
                        weight,
                        grad_output,
                        cu_seqlens,
                        initial_state,
                        *configs,
                        activation=activation,
                    )
                )
            else:
                grad_x, grad_weight = kda_ops.short_conv_configured_backward_op(
                    x,
                    weight,
                    grad_output,
                    cu_seqlens,
                    initial_state,
                    *configs,
                    activation=activation,
                )
                grad_initial_state = None
        return grad_x, grad_weight, None, grad_initial_state, None


class _ConfiguredShortConv(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        initial_state: torch.Tensor | None,
        activation: str | None,
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
        output = kda_ops.short_conv_configured_forward_op(
            x,
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
            activation=activation,
        )
        # Save inputs and backward specializations for preactivation recomputation.
        ctx.save_for_backward(x, weight, cu_seqlens, initial_state)
        ctx.activation = activation
        ctx.input_threads = input_threads
        ctx.input_channels = input_channels
        ctx.input_times = input_times
        ctx.weight_threads = weight_threads
        ctx.weight_channels = weight_channels
        ctx.weight_times = weight_times
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        """Dispatch the registered first-order backward operator."""
        x, weight, cu_seqlens, initial_state = ctx.saved_tensors
        configs = (
            ctx.input_threads,
            ctx.input_channels,
            ctx.input_times,
            ctx.weight_threads,
            ctx.weight_channels,
            ctx.weight_times,
        )
        if initial_state is not None and ctx.needs_input_grad[3]:
            grad_x, grad_weight, grad_initial_state = (
                kda_ops.short_conv_configured_backward_with_state_grad_op(
                    x,
                    weight,
                    grad_output,
                    cu_seqlens,
                    initial_state,
                    *configs,
                    activation=ctx.activation,
                )
            )
        else:
            grad_x, grad_weight = kda_ops.short_conv_configured_backward_op(
                x,
                weight,
                grad_output,
                cu_seqlens,
                initial_state,
                *configs,
                activation=ctx.activation,
            )
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
            None,
        )


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


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    forward_config: ShortConvConfig | None = None,
    input_grad_config: ShortConvConfig | None = None,
    weight_grad_config: ShortConvConfig | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Apply causal depthwise convolution with an optional fused activation.

    Width is inferred from ``weight.shape[1]`` and compile-time specialized.
    Schedule fields control register shapes, vector partitioning, unrolled loops,
    and block mapping, so changing one compiles and caches a distinct kernel.

    Args:
        x: Contiguous CUDA FP16, BF16, or FP32 input with shape ``[B, T, C]``.
            Each batch row is convolved as an independent sequence.
        weight: Contiguous depthwise weights with shape ``[C, W]`` matching
            ``x`` dtype and device.
        activation: Optional activation fused into the convolution epilogue and
            recomputed by the backward kernels. ``None`` applies no activation;
            ``"silu"`` is built in and :func:`register_activation` adds custom
            names. The name is an ordinary string argument, so compiled graphs
            specialize on it like any other static scalar.
        cu_seqlens: Optional contiguous CUDA int32 offsets delimiting independent
            sequences in a packed ``[1, T, C]`` input. Offsets must be nondecreasing,
            begin at zero, and end at an active endpoint ``L <= T``; repeated offsets
            represent empty padding slots for static-shape CUDA Graph replay. Only
            output positions before ``L`` are defined.
        initial_state: Optional causal history with shape ``[N, W - 1, C]``, where
            ``N`` is the dense batch size or the number of packed sequences. Absence
            is equivalent to an all-zero history.
        return_final_state: Return the final ``W - 1`` input positions with the output.
        forward_config: Optional forward schedule specialization.
        input_grad_config: Optional input-gradient schedule specialization.
        weight_grad_config: Optional weight-gradient schedule specialization.

    Returns:
        A contiguous tensor with the same shape, dtype, and device as ``x``. For packed
        input, only output and input-gradient positions in ``[0, L)`` are defined;
        parameter, state, and final-state values exclude the inactive suffix. When
        ``return_final_state`` is true, also returns ``[N, W - 1, C]`` history.
    """
    resolve_activation(activation)
    _validate_inputs(x, weight, cu_seqlens, initial_state)
    channels = x.shape[2]
    kernel_initial_state = None if weight.shape[1] == 1 else initial_state
    defaults = ShortConvTunedConfig.default(
        x.dtype,
        packed=cu_seqlens is not None,
    )
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
        output = _ShortConv.apply(x, weight, cu_seqlens, kernel_initial_state, activation)
    else:
        output = _ConfiguredShortConv.apply(
            x,
            weight,
            cu_seqlens,
            kernel_initial_state,
            activation,
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


def causal_conv1d_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    *,
    activation: str | None = None,
    state_indices: torch.Tensor | None = None,
    forward_config: ShortConvConfig | None = None,
) -> torch.Tensor:
    """Advance the convolution by one token per sequence over a paged history.

    The decode counterpart to :func:`causal_conv1d`, fused: one program reads a slot's
    history, convolves the new token in, and writes the shifted history back, so no
    gather or scatter surrounds the call.

    Args:
        x: ``[num_sequences, C]`` input, one token per sequence, matching ``weight`` and
            ``state`` in dtype.
        weight: Contiguous depthwise weights ``[C, W]``. ``W >= 2``; a width-1
            convolution carries no history and needs no update.
        state: Causal history ``[num_slots, W - 1, C]`` holding each slot's trailing
            ``W - 1`` inputs, oldest first -- the layout :func:`causal_conv1d` takes as
            ``initial_state`` and returns as its final state. Advanced **in place**: the
            rows shift left and ``x`` becomes the newest. Its storage must be contiguous
            and 16-byte aligned, since the launcher cannot realign it without losing the
            update.
        activation: Any name registered with :func:`register_activation`, or ``None``.
        state_indices: Optional contiguous int32 slot indices, one per sequence, selecting
            rows of a paged ``state`` pool. Without them, sequence ``i`` uses slot ``i``.
            Positive slots must be distinct. Non-positive entries produce zero output and
            leave the pool untouched.
        forward_config: Optional schedule specialization.

    Returns:
        The activated output ``[num_sequences, C]``, same dtype as ``x``.

    Inference-only: the single-token step has no backward, so gradient-tracking inputs
    are rejected when autograd is enabled.
    """
    resolve_activation(activation)
    _validate_decode_inputs(x, weight, state, state_indices)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (x, weight, state)):
        raise RuntimeError(
            "causal_conv1d_decode is inference-only and has no backward; use "
            "causal_conv1d for training or call under torch.no_grad()"
        )
    default = ShortConvTunedConfig.default(x.dtype).forward
    config = _compatible_config(default, x.shape[1]) if forward_config is None else forward_config
    _validate_config(config, x.shape[1], "forward_config")
    if forward_config is None and config == default:
        return kda_ops.short_conv_decode_op(x, weight, state, state_indices, activation=activation)
    return kda_ops.short_conv_configured_decode_op(
        x,
        weight,
        state,
        state_indices,
        config.threads,
        config.channels_per_thread,
        config.times_per_block,
        activation=activation,
    )


__all__ = [
    "ShortConvConfig",
    "ShortConvTunedConfig",
    "causal_conv1d",
    "causal_conv1d_decode",
    "tune_causal_conv1d",
]
