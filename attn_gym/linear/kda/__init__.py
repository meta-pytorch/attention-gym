# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA (Kimi Delta Attention) operations.

``chunk_kda`` and ``recurrent_kda`` dispatch on ``impl``; ``paged_chunk_kda``
provides the inference-only mutable-cache path. Fused backends load lazily on first call,
and the naive oracles in
``attn_gym.linear.kda.naive`` serve ``impl="reference"``.
"""

import importlib

from attn_gym.linear.kda.api import (
    KernelOptions,
    chunk_kda,
    paged_chunk_kda,
    recurrent_kda,
    recurrent_kda_decode,
)
from attn_gym.linear.kda.constants import MAX_GATE_LOWER_BOUND_MAGNITUDE
from attn_gym.linear.kda.gate import bound_gate
from attn_gym.linear.kda.masking import (
    active_token_mask,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
)

# Note: Lazy Imports (see attn_gym/linear/__init__.py)
_BACKEND_EXPORTS = {
    "l2norm": "attn_gym.linear.kda.fwd.triton.l2norm_fwd",
    # Staged primitives around the affine state boundary (stages.py) and the all-gather recipe
    # over them. Ownership plans live in attn_gym.linear.context_parallel.
    "ChunkKDASaved": "attn_gym.linear.kda.stages",
    "chunk_kda_prepare": "attn_gym.linear.kda.stages",
    "chunk_kda_prepare_backward": "attn_gym.linear.kda.stages",
    "context_parallel_kda": "attn_gym.linear.kda.context_parallel",
}
# Backward compat: these moved to attn_gym.linear.short_conv, which owns their
# lazy resolution and error message; import them from attn_gym.linear instead.
_SHORT_CONV_BC = {"causal_conv1d", "causal_conv1d_decode", "register_activation"}


def __getattr__(name: str):
    if name in _SHORT_CONV_BC:
        return getattr(importlib.import_module("attn_gym.linear.short_conv"), name)
    module_name = _BACKEND_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"{name} requires the optional CUDA kernel backends: pip install attn-gym[linear]"
        ) from error
    return getattr(module, name)


__all__ = sorted(  # noqa: PLE0605 -- backend exports resolve lazily
    [
        "KernelOptions",
        "MAX_GATE_LOWER_BOUND_MAGNITUDE",
        "active_token_mask",
        "bound_gate",
        "chunk_kda",
        "mask_inactive_token_gradients",
        "mask_inactive_tokens",
        "paged_chunk_kda",
        "recurrent_kda",
        "recurrent_kda_decode",
        *_BACKEND_EXPORTS,
        *_SHORT_CONV_BC,
    ]
)
