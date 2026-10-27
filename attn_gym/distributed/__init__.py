# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Communication building blocks for context-parallel attention."""

from attn_gym.distributed.copy_engine import (
    DEFAULT_TIMEOUT_MS,
    CopyEngineAllGather,
    CopyEngineChannel,
    nvlink_domain,
    same_nvlink_domain,
)

__all__ = [
    "DEFAULT_TIMEOUT_MS",
    "CopyEngineAllGather",
    "CopyEngineChannel",
    "nvlink_domain",
    "same_nvlink_domain",
]
