# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selective K/V block exchange for context-parallel attention.

`BlockExchangePlan.from_assembled_blocks` lays out which blocks move between ranks.
`CopyEngineBlockExchange` runs a plan with copy-engine pushes into PyTorch symmetric memory.
"""

from attn_gym.distributed.selective_gather.copy_engine import CopyEngineBlockExchange
from attn_gym.distributed.selective_gather.plan import (
    BlockExchangePlan,
    BlockLocation,
)

__all__ = [
    "BlockExchangePlan",
    "BlockLocation",
    "CopyEngineBlockExchange",
]
