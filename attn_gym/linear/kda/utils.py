# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
#
# Portions of this file are derived from flash-linear-attention
# (https://github.com/fla-org/flash-linear-attention) and are licensed under
# the MIT license; for the full list of FLA contributors, visit
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# The remaining portions are licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Consolidated utilities for the KDA kernels (forward + backward). Merges the
# former ``common/utils.py`` (Meta) and ``fla_ops/utils.py`` (FLA) into a single
# module. Targets NVIDIA GPUs only.

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton import utils as triton_utils
from attn_gym.linear.kda.constants import is_sm100_kda_capability

autotune_cache_kwargs = triton_utils.autotune_cache_kwargs

IS_GATHER_SUPPORTED = hasattr(triton.language, "gather")


@lru_cache(maxsize=8)
def is_sm100_kda_target(device: torch.device) -> bool:
    """Return whether a physical device should use the SM100-specific KDA route."""
    properties = torch.cuda.get_device_properties(device)
    return is_sm100_kda_capability((properties.major, properties.minor))


def _uses_pre_ampere_triton_target() -> bool:
    try:
        backend = triton.runtime.driver.active.get_current_target().backend
    except Exception:
        return False
    return backend == "cuda" and torch.cuda.get_device_capability(0)[0] < 8


if _uses_pre_ampere_triton_target():
    # Make old cards happy, since triton will use tf32 by default.
    os.environ["TRITON_F32_DEFAULT"] = "ieee"

triton_utils.configure_triton_allocator()


# ---------------------------------------------------------------------------
# Triton math ops
# ---------------------------------------------------------------------------
exp2 = triton_utils.exp2


@triton.jit
def masked_exp2(value, mask, FASTMATH: tl.constexpr = True):
    """Mask exponent inputs and select approximate or precise exponentiation."""
    exponent = tl.where(mask, value, 0.0)
    return tl.where(mask, exp2(exponent, FASTMATH), 0.0)


if not IS_GATHER_SUPPORTED:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        return None
else:
    gather = tl.gather  # type: ignore
