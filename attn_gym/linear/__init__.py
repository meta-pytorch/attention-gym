# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear attention operations."""

import importlib
from typing import TYPE_CHECKING

from attn_gym.linear.gdn import GatedDeltaRuleOutput, chunk_gdn, recurrent_gdn
from attn_gym.linear.kda import (
    active_token_mask,
    chunk_kda,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
    recurrent_kda,
    recurrent_kda_decode,
)
from attn_gym.linear.types import Impl

# Note: Lazy Imports
# Backend-backed names load on first use, keeping reference imports torch-only.
# Missing backends raise an actionable ImportError.
if TYPE_CHECKING:
    from attn_gym.linear.kda import (
        bounded_gate_cumsum,
        causal_conv1d,
        l2norm,
        register_activation,
    )

KDA_OPS = [
    "bounded_gate_cumsum",
    "chunk_kda",
    "recurrent_kda",
    "recurrent_kda_decode",
]

GDN_OPS = [
    "GatedDeltaRuleOutput",
    "chunk_gdn",
    "recurrent_gdn",
]

# Model-agnostic building blocks; they currently ship from the KDA module.
GENERIC_OPS = [
    "Impl",
    "active_token_mask",
    "causal_conv1d",
    "l2norm",
    "mask_inactive_token_gradients",
    "mask_inactive_tokens",
    "register_activation",
]

__all__ = GDN_OPS + GENERIC_OPS + KDA_OPS  # noqa: PLE0605 -- built from the op groups above


def __getattr__(name: str):
    # Names in __all__ that are not bound eagerly above resolve through
    # attn_gym.linear.kda; see the lazy-imports note.
    if name in __all__:
        return getattr(importlib.import_module("attn_gym.linear.kda"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
