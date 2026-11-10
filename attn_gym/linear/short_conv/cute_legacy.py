# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Previous decode, paged state-update and initial-state-gradient kernels with hand-built grids.

Verbatim copies (artifact names prefixed ``legacy_``) kept only for the bitwise parity test and
the A/B timing of the scheduler-driven kernels in ``cute.py``; removed with that test.
"""

from __future__ import annotations

import cutlass
from cutlass import Float32, Int32, cute

from attn_gym.linear.short_conv.activations import Activation
from attn_gym.linear.short_conv.cute import (
    ShortConvConfig,
    ShortConvDType,
    ShortConvGradientKernel,
    ShortConvKernel,
    _fake_decode_state,
    _fake_dynamic_rows,
    _fake_has_initial_state,
    _fake_indices,
    _fake_input,
    _fake_matrix,
    load_history,
    unrolled_dot,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache


class CausalConv1dSiluDecode(ShortConvKernel):
    """Advance a paged causal history by one token per sequence, activation fused."""

    kernel_kind = "legacy_decode"
    time_tiled = False

    def __init__(
        self,
        channels: int,
        width: int,
        config: ShortConvConfig,
        dtype: ShortConvDType,
        activation,
        use_has_initial_state: bool = False,
    ):
        assert channels % config.channels_per_thread == 0, (
            f"decode channels ({channels}) must be divisible by channels_per_thread "
            f"({config.channels_per_thread})"
        )
        super().__init__(channels, width, config, dtype)
        self.activation = activation
        self.use_has_initial_state = use_has_initial_state

    def get_name(self) -> str:
        """Name the artifact without the runtime-bound sequence axis."""
        return f"{super().get_name()}_his{int(self.use_has_initial_state)}"

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        state: cute.Tensor,
        state_indices: cute.Tensor | None,
        has_initial_state: cute.Tensor | None,
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
            state_groups = cute.zipped_divide(state, (1, 1, self.channels_per_thread))

            if active:
                weights = self.load_weights(weight, channel)

                taps = cute.make_rmem_tensor(
                    (self.channels_per_thread, self.width), self.dtype.cute_type
                )
                if cutlass.const_expr(has_initial_state is None):  # noqa: SIM114
                    for row in cutlass.range_constexpr(self.width - 1):
                        taps[(None, row)].store(
                            state_groups[((0, 0, None), (slot, row, channel_group))].load()
                        )
                elif has_initial_state[sequence]:
                    for row in cutlass.range_constexpr(self.width - 1):
                        taps[(None, row)].store(
                            state_groups[((0, 0, None), (slot, row, channel_group))].load()
                        )
                else:
                    # Fresh slots may contain NaNs: never load their old history.
                    taps.fill(self.dtype.cute_type(0.0))
                taps[(None, self.width - 1)].store(
                    x_groups[((0, None), (sequence, channel_group))].load()
                )

                output_groups[((0, None), (sequence, channel_group))].store(
                    self.activation(
                        unrolled_dot(taps, weights, 0, self.width, explicit_fma=True)
                    ).to(self.dtype.cute_type)
                )
                for row in cutlass.range_constexpr(self.width - 1):
                    state_groups[((0, 0, None), (slot, row, channel_group))].store(
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
        has_initial_state: cute.Tensor | None,
        stream,
    ):
        """Launch the configured decode specialization."""
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(x, weight, output, state, state_indices, has_initial_state).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.size(x, mode=[0]),
                1,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dPagedStateUpdate(ShortConvKernel):
    """Write each sequence's final causal history directly into its selected slot."""

    kernel_kind = "legacy_paged_state"
    time_tiled = False

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        state: cute.Tensor,
        state_indices: cute.Tensor,
        has_initial_state: cute.Tensor | None,
        cu_seqlens: cute.Tensor | None,
        tokens: Int32,
    ):
        """Advance one selected history slot without a compact staging tensor."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, sequence, _ = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread

        if channel < self.channels:
            slot = state_indices[sequence]
            if slot > 0:
                if cutlass.const_expr(cu_seqlens is None):
                    batch = Int32(sequence)
                    sequence_start = Int32(0)
                    sequence_end = tokens
                else:
                    batch = Int32(0)
                    sequence_start = Int32(cu_seqlens[sequence])
                    sequence_end = Int32(cu_seqlens[sequence + 1])
                sequence_length = sequence_end - sequence_start
                state_groups = cute.zipped_divide(state, (1, 1, self.channels_per_thread))

                for row in cutlass.range_constexpr(self.width - 1):
                    values = cute.make_rmem_tensor(
                        (self.channels_per_thread,), self.dtype.cute_type
                    )
                    values.fill(self.dtype.cute_type(0.0))
                    input_time = sequence_end - (self.width - 1) + row
                    if input_time >= sequence_start:
                        values.store(self.input_group(x, batch, input_time, channel_group).load())
                    # For short sequences the source row is ``row + sequence_length``,
                    # so ascending stores cannot overwrite a value before it is read.
                    elif cutlass.const_expr(has_initial_state is None):  # noqa: SIM114
                        values.store(
                            state_groups[
                                ((0, 0, None), (slot, row + sequence_length, channel_group))
                            ].load()
                        )
                    elif has_initial_state[sequence]:
                        values.store(
                            state_groups[
                                ((0, 0, None), (slot, row + sequence_length, channel_group))
                            ].load()
                        )
                    state_groups[((0, 0, None), (slot, row, channel_group))].store(values.load())

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        state: cute.Tensor,
        state_indices: cute.Tensor,
        has_initial_state: cute.Tensor | None,
        cu_seqlens: cute.Tensor | None,
        tokens: Int32,
        stream,
    ):
        """Launch one state-update task per sequence and channel group."""
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(x, state, state_indices, has_initial_state, cu_seqlens, tokens).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                cute.size(state_indices),
                1,
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


class CausalConv1dSiluInitialStateGradient(ShortConvGradientKernel):
    """Differentiate causal history using its triangular output dependency.

    History position ``s`` contributes to output positions ``o <= s`` through
    weight tap ``s - o``. Each block computes that short sum for one sequence,
    history position, and channel group.
    """

    kernel_kind = "legacy_dstate"
    time_tiled = False

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        initial_state: cute.Tensor,
        grad_output: cute.Tensor,
        grad_initial_state: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        tokens: Int32,
    ):
        """Differentiate one history position and packed channel group."""
        thread_idx, _, _ = cute.arch.thread_idx()
        channel_block, history_offset, sequence = cute.arch.block_idx()
        channel_group = channel_block * self.threads + thread_idx
        channel = channel_group * self.channels_per_thread

        if cutlass.const_expr(cu_seqlens is None):
            batch = Int32(sequence)
            sequence_start = Int32(0)
            sequence_end = tokens
        else:
            batch = Int32(0)
            sequence_start = Int32(cu_seqlens[sequence])
            sequence_end = Int32(cu_seqlens[sequence + 1])
        sequence_length = sequence_end - sequence_start

        if channel < self.channels:
            state_groups = cute.zipped_divide(initial_state, (1, self.channels_per_thread))
            dy_groups = cute.zipped_divide(grad_output, (1, self.channels_per_thread))
            dstate_groups = cute.zipped_divide(grad_initial_state, (1, self.channels_per_thread))
            weights = self.load_weights(weight, channel)

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
                                self.input_group(
                                    x, batch, sequence_start + input_offset, channel_group
                                )
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
                        self.row_group(
                            dy_groups, batch, sequence_start + output_offset, tokens, channel_group
                        )
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
                    (
                        self.upcast_offset(sequence) * (self.width - 1)
                        + self.upcast_offset(history_offset),
                        channel_group,
                    ),
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
        tokens: Int32,
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
            tokens,
        ).launch(
            grid=(
                cute.ceil_div(self.channels, self.threads * self.channels_per_thread),
                self.width - 1,
                initial_state.shape[0] // (self.width - 1),
            ),
            block=(self.threads, 1, 1),
            stream=stream,
        )


@jit_cache
def _compile_initial_state_gradient(
    channels: int,
    width: int,
    dtype: ShortConvDType,
    threads: int,
    channels_per_thread: int,
    input_alignment: int,
    packed: bool,
    activation: Activation,
    use_int64_offsets: bool = False,
):
    """Compile one initial-state gradient specialization."""
    derivative = activation.derivative
    config = ShortConvConfig(threads, channels_per_thread, times_per_block=1)
    operation = CausalConv1dSiluInitialStateGradient(
        channels,
        width,
        config,
        dtype,
        derivative,
        use_int64_offsets,
    )
    return compile_tvm_ffi(
        operation,
        _fake_input(dtype, channels, input_alignment, use_int64_offsets),
        _fake_matrix(dtype, channels, width),
        _fake_dynamic_rows(dtype, channels, use_int64_offsets),
        _fake_dynamic_rows(dtype, channels, use_int64_offsets),
        _fake_dynamic_rows(dtype, channels, use_int64_offsets),
        _fake_indices(packed),
        Int32(0),
    )


@jit_cache
def _compile_paged_state_update(
    channels: int,
    width: int,
    dtype: ShortConvDType,
    config: ShortConvConfig,
    input_alignment: int,
    packed: bool,
    use_has_initial_state: bool,
    use_int64_offsets: bool = False,
):
    """Compile paged state updates with runtime token and sequence counts and input strides."""
    operation = CausalConv1dPagedStateUpdate(
        channels,
        width,
        config,
        dtype,
        use_int64_offsets,
    )
    return compile_tvm_ffi(
        operation,
        _fake_input(dtype, channels, input_alignment, use_int64_offsets),
        _fake_decode_state(dtype, width, channels),
        _fake_indices(True),
        _fake_has_initial_state(use_has_initial_state),
        _fake_indices(packed),
        Int32(0),
    )


@jit_cache
def _compile_decode(
    channels: int,
    width: int,
    dtype: ShortConvDType,
    config: ShortConvConfig,
    input_alignment: int,
    paged: bool,
    activation: Activation,
    use_has_initial_state: bool = False,
):
    """Compile one decode specialization, shared across batch sizes and pool depths."""
    operation = CausalConv1dSiluDecode(
        channels, width, config, dtype, activation.forward, use_has_initial_state
    )
    return compile_tvm_ffi(
        operation,
        _fake_input(dtype, channels, input_alignment, outer_modes=1),
        _fake_matrix(dtype, channels, width),
        _fake_dynamic_rows(dtype, channels),
        _fake_decode_state(dtype, width, channels),
        _fake_indices(paged),
        _fake_has_initial_state(use_has_initial_state),
    )
