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

from attn_gym.linear._lazy import lazy_exports
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
# Backward compat: these moved to attn_gym.linear.short_conv, whose own lazy resolution supplies
# the error message; import them from attn_gym.linear instead.
_SHORT_CONV_BC = {
    name: "attn_gym.linear.short_conv"
    for name in ("causal_conv1d", "causal_conv1d_decode", "register_activation")
}
__getattr__ = lazy_exports(
    __name__, _BACKEND_EXPORTS | _SHORT_CONV_BC, requirement="CUDA kernel backends"
)


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
