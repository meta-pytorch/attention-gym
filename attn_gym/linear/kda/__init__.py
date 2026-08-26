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
    "causal_conv1d": "attn_gym.linear.kda.short_conv",
    "causal_conv1d_decode": "attn_gym.linear.kda.short_conv",
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
        if name in {"causal_conv1d", "causal_conv1d_decode", "register_activation"}:
            message = f"{name} requires the optional CuTeDSL backend: pip install attn-gym[linear]"
        else:
            message = f"{name} requires CUDA with Triton support"
        raise ImportError(message) from error
    return getattr(module, name)


__all__ = sorted(  # noqa: PLE0605 -- backend exports resolve lazily
    [
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
    ]
)
