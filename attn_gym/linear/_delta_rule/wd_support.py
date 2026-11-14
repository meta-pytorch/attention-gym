# SPDX-License-Identifier: BSD-3-Clause
"""What the warpdrive GDN/KDA kernels support, defined once for kernels, drivers and adapters.

Outside the ``warpdrive`` package and importing only torch, so ``import attn_gym.linear`` can
read it without loading any kernel module.
"""

import torch

# Head dimensions every GDN/KDA kernel compiles for (keys and values alike).
HEAD_DIMS = (64, 128)
# Q/K/V dtypes the kernels read, and the names their compile keys use.
IO_DTYPE_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16"}


def standalone_dtype(op, name: str) -> str:
    """The dtype ``op``'s own compile function reads from its constructor. A bundle that launches
    ``op`` from its own host keys the dtype itself and leaves the parameter None."""
    value = getattr(op, name)
    if value is None:
        raise ValueError(f"compiling {type(op).__name__} on its own needs {name}= at construction")
    return value
