# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side CuTeDSL helpers imported only by CuTe kernel modules."""

from __future__ import annotations

from cutlass import Float32, Int32, cute


@cute.jit
def upper_bound(
    values: cute.Tensor,
    query: Int32,
    begin: Int32,
    end: Int32,
) -> Int32:
    """Return the first index in ``[begin, end)`` whose value exceeds ``query``."""
    low = begin
    high = end
    while low < high:
        middle = (low + high) // 2
        if Int32(values[middle]) <= query:
            low = middle + 1
        else:
            high = middle
    return low


@cute.jit
def cta_reduce_sum(value: Float32, warp_partials: cute.Tensor) -> Float32:
    """Reduce one FP32 value per thread across a CTA.

    ``warp_partials`` is caller-owned shared-memory scratch with one FP32 slot
    per full warp in a one-dimensional CTA. All CTA threads must call this
    helper. The returned value is only meaningful to thread zero. This is a
    terminal reduction: synchronize the CTA before reusing the scratch.
    """
    tidx, _, _ = cute.arch.thread_idx()
    lane = tidx % 32
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    num_warps = cute.size(warp_partials)

    warp_sum = cute.arch.warp_reduction_sum(value)
    if lane == 0:
        warp_partials[warp] = warp_sum
    cute.arch.sync_threads()

    cta_sum = Float32(0.0)
    if warp == 0:
        if lane < num_warps:
            cta_sum = warp_partials[lane]
        cta_sum = cute.arch.warp_reduction_sum(cta_sum)
    return cta_sum


__all__ = ["cta_reduce_sum", "upper_bound"]
