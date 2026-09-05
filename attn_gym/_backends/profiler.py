# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch-only profiler helpers, importable without any kernel backend."""

from __future__ import annotations

import contextlib

import torch


def profiler_range(name: str):
    """A named profiler range, free when no torch profiler is active (~4us each)."""
    if torch.autograd.profiler._is_profiler_enabled:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()


__all__ = ["profiler_range"]
