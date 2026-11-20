"""Route fused delta-rule stages between legacy CuTeDSL kernels and portable Triton kernels.

The branched stages of the fused ``chunk_kda`` backward have portable Triton implementations that
run on every supported CUDA architecture. On SM100-family GPUs they default to legacy CuTeDSL
kernels (the ``kda/bwd/cute`` modules, not the warpdrive backend). Setting
``ATTN_GYM_LEGACY_CUTE=0`` routes them to Triton there too, so the portable pipeline can be
validated and measured on Blackwell before the legacy kernels are deleted. Callers read the
variable on every launch inside registered operators, so tracing never specializes on it.
"""

from __future__ import annotations

import os

import torch

from attn_gym._backends.arch import runs_sm100_kernels
from attn_gym._backends.cute.utils import get_device_properties

LEGACY_CUTE_ENV = "ATTN_GYM_LEGACY_CUTE"


def uses_legacy_cute(device: torch.device) -> bool:
    """Return whether ``device`` runs the legacy SM100-family CuTeDSL stages.

    Raises:
        ValueError: If ``ATTN_GYM_LEGACY_CUTE`` is set to anything but ``0`` or ``1``.
    """
    match os.environ.get(LEGACY_CUTE_ENV, "1"):
        case "0":
            return False
        case "1":
            properties = get_device_properties(device)
            return runs_sm100_kernels((properties.major, properties.minor))
        case value:
            raise ValueError(f"{LEGACY_CUTE_ENV} must be 0 or 1, got {value!r}")


__all__ = ["LEGACY_CUTE_ENV", "uses_legacy_cute"]
