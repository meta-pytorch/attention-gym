# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe. Multicast, bulk-copy, cp.async tile loads, and static
# descriptor paths unused by the vendored kernels were removed. Tiles always address a runtime
# descriptor that the caller acquires once with :func:`tma_tensormap_acquire`. Store group
# commit/wait use the cute.arch wrappers; the tensor copies stay on the raw-descriptor primitives.


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
    cute.arch.cp_async_bulk_commit_group()


@cute.jit
def tma_store_wait(num_remaining: int = 0):
    cute.arch.cp_async_bulk_wait_group(num_remaining, read=True)


@cute.jit
def cp_async_commit():
    nvvm.cp_async_commit_group()


@cute.jit
def cp_async_wait(num_remaining: cutlass.Constexpr[int] = 0):
    nvvm.cp_async_wait_group(num_remaining)


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


def ptx_type_suffix(dtype) -> str:
    """PTX type suffix for a 32-bit global ld/st: ``f32`` for floats, ``b32`` for bit patterns."""
    return "f32" if dtype == cutlass.Float32 else "b32"


def ld_global(addr, dtype):
    """32-bit global load: one register of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.global.{ptx_type_suffix(dtype)} $0, [$1];",
        write_only_types=[dtype],
        read_only_args=[addr],
    )


def ld_global_v2(addr, dtype):
    """64-bit global load: two 32-bit registers of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.global.v2.{ptx_type_suffix(dtype)} {{$0, $1}}, [$2];",
        write_only_types=[dtype] * 2,
        read_only_args=[addr],
    )


def ld_global_v4(addr, dtype):
    """128-bit global load: four 32-bit registers of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.global.v4.{ptx_type_suffix(dtype)} {{$0, $1, $2, $3}}, [$4];",
        write_only_types=[dtype] * 4,
        read_only_args=[addr],
    )


def st_global(addr, value, dtype):
    """32-bit global store: one register of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.global.{ptx_type_suffix(dtype)} [$0], $1;",
        read_only_args=[addr, value],
    )


def st_global_v2(addr, values, dtype):
    """64-bit global store: two 32-bit registers of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.global.v2.{ptx_type_suffix(dtype)} [$0], {{$1, $2}};",
        read_only_args=[addr, values[0], values[1]],
    )


def st_global_v4(addr, values, dtype):
    """128-bit global store: four 32-bit registers of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.global.v4.{ptx_type_suffix(dtype)} [$0], {{$1, $2, $3, $4}};",
        read_only_args=[addr, values[0], values[1], values[2], values[3]],
    )


def ld_shared_v2(addr, dtype):
    """64-bit shared load: two 32-bit registers of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.shared.v2.{ptx_type_suffix(dtype)} {{$0, $1}}, [$2];",
        write_only_types=[dtype] * 2,
        read_only_args=[addr],
    )


def ld_shared_v4(addr, dtype):
    """128-bit shared load: four 32-bit registers of ``dtype`` from ``addr``."""
    return nvvm.inline_ptx(
        f"ld.shared.v4.{ptx_type_suffix(dtype)} {{$0, $1, $2, $3}}, [$4];",
        write_only_types=[dtype] * 4,
        read_only_args=[addr],
    )


def st_shared_v2(addr, values, dtype):
    """64-bit shared store: two 32-bit registers of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.shared.v2.{ptx_type_suffix(dtype)} [$0], {{$1, $2}};",
        read_only_args=[addr, values[0], values[1]],
    )


def st_shared_v4(addr, values, dtype):
    """128-bit shared store: four 32-bit registers of ``dtype`` to ``addr``."""
    nvvm.inline_ptx(
        f"st.shared.v4.{ptx_type_suffix(dtype)} [$0], {{$1, $2, $3, $4}};",
        read_only_args=[addr, values[0], values[1], values[2], values[3]],
    )
