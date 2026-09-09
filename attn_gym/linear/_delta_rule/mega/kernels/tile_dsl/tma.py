# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: multicast, bulk-copy, cp.async tile loads, and static
# descriptor paths unused by the vendored kernels were removed. Tiles always address a runtime
# descriptor that the caller acquires once with :func:`tma_tensormap_acquire`.


import cutlass
from cutlass import cute
from cutlass.experimental import primitives as nvvm

from .handles import smem_data_ptr


@cute.jit
def tma_load_tile(smem_tile, gmem_slice, mbar):
    """Issue the TMA loads of one tile from the electing thread."""
    coord_d = gmem_slice.coord_d
    outer_coords = tuple(gmem_slice.coords[1:])
    for i in cutlass.range_constexpr(smem_tile.tma_loads_per_tile):
        d = coord_d + cutlass.Int32(i * smem_tile.tma_granu_elems)
        smem_chunk = smem_data_ptr(smem_tile.shifted(i * smem_tile.tma_subtile_stride_elems).base)
        if nvvm.elect_sync():
            nvvm.cp_async_bulk_tensor_shared_cta_global(
                smem_chunk,
                gmem_slice.desc_ptr,
                [d] + list(outer_coords),
                mbar,
            )


@cute.jit
def tma_store_tile(smem_tile, gmem_slice):
    """Issue the TMA stores of one tile; the caller elects the issuing thread."""
    coord_d = gmem_slice.coord_d
    outer_coords = tuple(gmem_slice.coords[1:])
    for i in cutlass.range_constexpr(smem_tile.tma_loads_per_tile):
        d = coord_d + cutlass.Int32(i * smem_tile.tma_granu_elems)
        smem_chunk = smem_data_ptr(smem_tile.shifted(i * smem_tile.tma_subtile_stride_elems).base)
        nvvm.cp_async_bulk_tensor_global_shared_cta(
            gmem_slice.desc_ptr,
            smem_chunk,
            tuple([d] + list(outer_coords)),
        )


@cute.jit
def tma_store_commit():
    nvvm.cp_async_bulk_commit_group()


@cute.jit
def tma_store_wait(num_remaining: int = 0):
    nvvm.cp_async_bulk_wait_group(num_remaining, read=True)


@cute.jit
def tma_tensormap_acquire(desc_ptr):
    """Issue a single ``fence.proxy.tensormap::generic.acquire.gpu`` over a
    runtime TMA descriptor.

    A runtime descriptor written on the host by a descriptor-builder kernel
    (via ``tensormap_replace``) is visible to the TMA proxy only after this
    GENERIC->TENSORMAP acquire.  Such descriptors are built once (not rewritten
    per work-tile), so a single acquire per consumer CTA (or once per
    persistent-loop tile) suffices.
    """
    nvvm.fence_proxy_acquire(
        nvvm.MemScope.GPU,
        desc_ptr,
        128,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )
