"""When a public operation may pick kernel configurations by timing."""

from collections.abc import Mapping

import torch


def resolve_autotune(kernel_options: Mapping[str, object], *, default: bool = True) -> bool:
    """Return the validated ``kernel_options["autotune"]`` request, or ``default`` if absent."""
    autotune = kernel_options.get("autotune", default)
    if type(autotune) is not bool:
        raise TypeError("kernel_options['autotune'] must be a bool")
    return autotune


def resolve_tuning_options(
    kernel_options: Mapping[str, object] | None, *, op_name: str, default: bool = True
) -> bool:
    """Validate ``kernel_options`` of an operation whose only option is ``autotune``."""
    if kernel_options is None:
        return default
    unknown = kernel_options.keys() - {"autotune"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported {op_name} kernel options: {names}")
    return resolve_autotune(kernel_options, default=default)


def should_autotune(requested: bool) -> bool:
    """Return ``requested`` unless deterministic algorithms are enabled.

    Candidates can differ in floating-point order, so timing-based selection can change bits
    across processes and winner-cache states. Deterministic algorithms pin configurations as
    with ``kernel_options={"autotune": False}``. Operations resolve this policy and pass a bool
    to tuners and kernels.
    """
    return requested and not torch.are_deterministic_algorithms_enabled()
