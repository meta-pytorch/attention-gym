# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch-only private operator contracts for the fused short-convolution backend.

Schemas live here so they exist before graph capture. CUDA implementations and
fake registrations live in ``attn_gym.linear.short_conv.cute`` and import the
optional CuTeDSL backend only when the dispatcher executes the operator.
"""

from __future__ import annotations

import torch

torch.library.define(
    "attn_gym::_cute_short_conv_fwd",
    "(Tensor x, Tensor weight, Tensor? cu_seqlens=None, Tensor? initial_state=None,"
    " *, str? activation=None) -> Tensor",
)
torch.library.define(
    "attn_gym::_cute_short_conv_decode",
    "(Tensor x, Tensor weight, Tensor(a!) state, Tensor? state_indices,"
    " *, str? activation=None) -> Tensor",
)
torch.library.define(
    "attn_gym::_cute_short_conv_paged_fwd",
    "(Tensor x, Tensor weight, Tensor(a!) state, Tensor state_indices,"
    " Tensor? has_initial_state, Tensor? cu_seqlens, *, str? activation=None) -> Tensor",
)
torch.library.define(
    "attn_gym::_cute_short_conv_configured_decode",
    "(Tensor x, Tensor weight, Tensor(a!) state, Tensor? state_indices,"
    " int forward_threads, int forward_channels, int forward_times,"
    " *, str? activation=None) -> Tensor",
)
torch.library.define(
    "attn_gym::_cute_short_conv_bwd",
    "(Tensor x, Tensor weight, Tensor grad_output, Tensor? cu_seqlens=None,"
    " *, str? activation=None) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::_cute_short_conv_configured_fwd",
    "(Tensor x, Tensor weight, Tensor? cu_seqlens, Tensor? initial_state,"
    " int forward_threads, int forward_channels, int forward_times,"
    " int input_threads, int input_channels, int input_times,"
    " int weight_threads, int weight_channels, int weight_times,"
    " *, str? activation=None) -> Tensor",
)

_SHORT_CONV_CONFIGURED_BWD_ARGS = (
    "(Tensor x, Tensor weight, Tensor grad_output, Tensor? cu_seqlens, {initial_state},"
    " int input_threads, int input_channels, int input_times,"
    " int weight_threads, int weight_channels, int weight_times,"
    " bool persistent_tma_input_gradient, *, str? activation=None)"
)
torch.library.define(
    "attn_gym::_cute_short_conv_configured_bwd",
    _SHORT_CONV_CONFIGURED_BWD_ARGS.format(initial_state="Tensor? initial_state")
    + " -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::_cute_short_conv_configured_bwd_with_state_grad",
    _SHORT_CONV_CONFIGURED_BWD_ARGS.format(initial_state="Tensor initial_state")
    + " -> (Tensor, Tensor, Tensor)",
)

short_conv_forward_op = torch.ops.attn_gym._cute_short_conv_fwd.default
short_conv_backward_op = torch.ops.attn_gym._cute_short_conv_bwd.default
short_conv_decode_op = torch.ops.attn_gym._cute_short_conv_decode.default
short_conv_paged_forward_op = torch.ops.attn_gym._cute_short_conv_paged_fwd.default
short_conv_configured_forward_op = torch.ops.attn_gym._cute_short_conv_configured_fwd.default
short_conv_configured_backward_op = torch.ops.attn_gym._cute_short_conv_configured_bwd.default
short_conv_configured_backward_with_state_grad_op = (
    torch.ops.attn_gym._cute_short_conv_configured_bwd_with_state_grad.default
)
short_conv_configured_decode_op = torch.ops.attn_gym._cute_short_conv_configured_decode.default

__all__ = [
    "short_conv_backward_op",
    "short_conv_configured_backward_op",
    "short_conv_configured_backward_with_state_grad_op",
    "short_conv_configured_decode_op",
    "short_conv_configured_forward_op",
    "short_conv_decode_op",
    "short_conv_forward_op",
    "short_conv_paged_forward_op",
]
