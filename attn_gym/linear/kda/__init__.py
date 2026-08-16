# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA (Kimi Delta Attention) references and optimized kernels."""

import importlib

from attn_gym.linear.kda.fwd.triton.gate_fwd import bounded_gate_cumsum
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm
from attn_gym.linear.kda.masking import (
    active_token_mask,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
)
from attn_gym.linear.kda.naive import (
    naive_chunk_kda,
    naive_chunk_kda_from_cumulative,
    naive_recurrent_kda,
)

# Note: Lazy Imports (see attn_gym/linear/__init__.py)
_CUTEDSL_EXPORTS = {
    "causal_conv1d": "attn_gym.linear.kda.short_conv",
    "chunk_kda": "attn_gym.linear.kda.fwd.cute",
    "register_activation": "attn_gym.linear.kda.short_conv",
}


def __getattr__(name: str):
    module_name = _CUTEDSL_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ImportError(
            f"{name} requires the optional CuTeDSL backend: pip install attn-gym[linear]"
        ) from error
    return getattr(module, name)


__all__ = sorted(  # noqa: PLE0605 -- the lazy half comes from _CUTEDSL_EXPORTS
    [
        "active_token_mask",
        "bounded_gate_cumsum",
        "l2norm",
        "mask_inactive_token_gradients",
        "mask_inactive_tokens",
        "naive_chunk_kda",
        "naive_chunk_kda_from_cumulative",
        "naive_recurrent_kda",
        *_CUTEDSL_EXPORTS,
    ]
)
