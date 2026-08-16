# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimized KDA short-convolution operations."""

from attn_gym.linear.kda.short_conv.activations import register_activation
from attn_gym.linear.kda.short_conv.cute import (
    ShortConvConfig,
    ShortConvTunedConfig,
    causal_conv1d,
    tune_causal_conv1d,
)

__all__ = [
    "ShortConvConfig",
    "ShortConvTunedConfig",
    "causal_conv1d",
    "register_activation",
    "tune_causal_conv1d",
]
