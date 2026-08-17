# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Linear attention operations: ``from attn_gym.linear import ...``."""

import importlib
from typing import TYPE_CHECKING

from attn_gym.linear.gdn import GatedDeltaRuleOutput, gated_delta_rule
from attn_gym.linear.kda import (
    active_token_mask,
    bounded_gate_cumsum,
    l2norm,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
    naive_chunk_kda,
    naive_chunk_kda_from_cumulative,
    naive_recurrent_kda,
    recurrent_kda,
)

# Note: Lazy Imports
# The CuTeDSL kernels are an optional dependency (`pip install attn-gym[linear]`);
# torch- and triton-backed names import eagerly. Names backed by CuTeDSL resolve
# lazily through PEP 562 module `__getattr__` (the standard scientific-python
# lazy-import mechanism) so the base package imports without the extra, and a
# missing backend raises an ImportError naming the install command.
if TYPE_CHECKING:
    from attn_gym.linear.kda import causal_conv1d, chunk_kda, register_activation

KDA_OPS = [
    "bounded_gate_cumsum",
    "chunk_kda",
    "naive_chunk_kda",
    "naive_chunk_kda_from_cumulative",
    "naive_recurrent_kda",
    "recurrent_kda",
]

GDN_OPS = [
    "GatedDeltaRuleOutput",
    "gated_delta_rule",
]

# Model-agnostic building blocks; they currently ship from the KDA module.
GENERIC_OPS = [
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
