# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL helpers for packed (ragged) sequence boundary tensors."""

from __future__ import annotations

from cutlass import Int32, cute


@cute.jit
def load_ragged_token_count(cu_seqlens: cute.Tensor):
    """Load the terminal packed offset containing the runtime active token count."""
    return Int32(cu_seqlens[cute.size(cu_seqlens) - 1])
