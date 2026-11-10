"""Routing to the Triton grouped schedule, shared by the public API and the Triton backend."""

from collections.abc import Sequence

import torch


def uses_grouped_schedule(heads: int, kv_heads: int, separate_values: bool) -> bool:
    """Whether inputs need the Triton grouped schedule: separate values or 1 < kv_heads < heads.

    Backend choice, the forward schedule, the forward's query map, and the backward launcher
    all route on this one predicate so they cannot disagree.
    """
    return separate_values or kv_heads not in (1, heads)


def require_grouped_schedule(
    query: torch.Tensor, kv_tensors: Sequence[torch.Tensor], *, sliding_window_size: int
) -> None:
    """Raise unless the grouped Blackwell schedule can run these inputs.

    Reads metadata only, so the public API checks before dispatch (also while tracing) and the
    forward checks again for direct backend callers. ``kv_tensors`` holds every key and value
    pool the kernels gather from in place; the backend makes the query contiguous itself.
    Head dimensions above 256 are excluded because separate K and V tiles then exceed shared
    memory, and the dense tiles' 128-row operands need datacenter Blackwell's shared memory.
    """
    head_dim = query.shape[-1]
    major, minor = torch.cuda.get_device_capability(query.device)
    violation = None
    if major != 10:
        violation = f"SM100 or SM103, got SM{major}{minor}"
    elif query.dtype != torch.bfloat16:
        violation = f"bfloat16 inputs, got {query.dtype}"
    elif any(tensor.stride(-1) != 1 for tensor in kv_tensors):
        violation = "unit inner strides"
    elif head_dim % 16 or head_dim > 256:
        violation = f"head_dim <= 256 and divisible by 16, got {head_dim}"
    elif sliding_window_size > 2048:
        violation = f"sliding_window_size <= 2048, got {sliding_window_size}"
    if violation is not None:
        raise ValueError(
            "Triton gather_attn supports separate values and grouped KV heads only with "
            f"{violation}; use impl='reference'"
        )
