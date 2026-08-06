"""Triton-specific tensor capability checks."""

import torch


def can_use_tma(tensor: torch.Tensor) -> bool:
    """Return whether a tensor can be accessed with a TMA tensor descriptor."""
    if not tensor.is_cuda or torch.cuda.get_device_capability(tensor.device)[0] < 9:
        return False

    element_size = tensor.element_size()
    storage_is_aligned = (
        torch.compiler.is_compiling() or tensor.storage_offset() * element_size % 16 == 0
    )
    return (
        tensor.shape[-1] <= 256
        and tensor.stride(-1) == 1
        and storage_is_aligned
        and all(stride * element_size % 16 == 0 for stride in tensor.stride()[:-1])
    )
