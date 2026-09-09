# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: only the softplus behind the GDN gate remains; the KDA
# kernels define their own L2-norm floor.

"""Element-wise device helpers shared by the FROST LA kernels."""

import cutlass
from cutlass import cute


@cute.jit
def softplus(x: cutlass.Float32) -> cutlass.Float32:
    """log(1 + exp(x)) with the linear tail (x > 20 returns x: exp saturates
    fp32 there and log1p(exp(x)) == x to fp32 precision)."""
    result = x
    if x < cutlass.Float32(20.0):
        result = cute.math.log(cutlass.Float32(1.0) + cute.math.exp(x, fastmath=True), fastmath=True)
    return result
