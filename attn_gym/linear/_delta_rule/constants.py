"""Log-base conversion, chunk geometry, and targets shared by delta-rule implementations."""

import math

LN2 = math.log(2.0)
LOG2_E = math.log2(math.e)
DEFAULT_CHUNK_SIZE = 64
SM100_KDA_CAPABILITIES = frozenset(((10, 0), (10, 3)))


def is_sm100_kda_capability(capability: tuple[int, int] | None) -> bool:
    """Return whether a CUDA capability supports the SM100-specific delta-rule kernels."""
    return capability in SM100_KDA_CAPABILITIES
