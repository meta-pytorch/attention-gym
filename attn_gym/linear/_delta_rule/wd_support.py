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
