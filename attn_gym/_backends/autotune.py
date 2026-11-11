"""When a public operation may pick kernel configurations by timing."""

import torch


def should_autotune(requested: bool) -> bool:
    """Return ``requested`` unless deterministic algorithms are enabled.

    Candidates can differ in floating-point order, so timing-based selection can change bits
    across processes and winner-cache states. Deterministic algorithms pin configurations as
    with ``autotune=False``. Operations resolve this policy and pass a bool to tuners and kernels.
    """
    return requested and not torch.are_deterministic_algorithms_enabled()
