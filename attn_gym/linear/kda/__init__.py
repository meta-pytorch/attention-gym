# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA (Kimi Delta Attention) operations.

``chunk_kda`` and ``recurrent_kda`` dispatch on ``impl``: fused backends load
lazily on first fused call, and the naive oracles in
``attn_gym.linear.kda.naive`` serve ``impl="reference"``.
"""

import importlib

from attn_gym.linear.kda.api import chunk_kda, recurrent_kda
from attn_gym.linear.kda.masking import (
    active_token_mask,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
)

# Note: Lazy Imports (see attn_gym/linear/__init__.py)
_BACKEND_EXPORTS = {
    "bounded_gate_cumsum": "attn_gym.linear.kda.fwd.triton.gate_fwd",
    "causal_conv1d": "attn_gym.linear.kda.short_conv",
    "l2norm": "attn_gym.linear.kda.fwd.triton.l2norm_fwd",
    "register_activation": "attn_gym.linear.kda.short_conv",
}


def __getattr__(name: str):
    module_name = _BACKEND_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        if name in {"causal_conv1d", "register_activation"}:
            message = f"{name} requires the optional CuTeDSL backend: pip install attn-gym[linear]"
        else:
            message = f"{name} requires CUDA with Triton support"
        raise ImportError(message) from error
    return getattr(module, name)


__all__ = sorted(  # noqa: PLE0605 -- backend exports resolve lazily
    [
        "active_token_mask",
        "chunk_kda",
        "mask_inactive_token_gradients",
        "mask_inactive_tokens",
        "recurrent_kda",
        *_BACKEND_EXPORTS,
    ]
)
