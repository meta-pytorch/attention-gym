# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute

from .swizzle import swizzle_xor_128b


def _advance(x, n):
    if hasattr(x, "subview"):
        return x.subview(n)
    return x + n


def _bulk_copy_ptr(x):
    if hasattr(x, "llvm_ptr"):
        return x.llvm_ptr
    if hasattr(x, "ir_value") and callable(x.ir_value):
        return x.ir_value()
    return x


@cute.jit
def cp_async_bulk_shared_cluster_shared_cta(dst_mem, src_mem, mbar, size, *, pred=None):
    if cutlass.const_expr(pred is None):
        nvvm.cp_async_bulk_shared_cluster_shared_cta(dst_mem, src_mem, mbar, size)
    else:
        nvvm.inline_ptx(
            "cp.async.bulk.shared::cluster.shared::cta" ".mbarrier::complete_tx::bytes " "[{$r0}], [{$r1}], {$r2}, [{$r3}];",
            read_only_args=[
                _bulk_copy_ptr(dst_mem),
                _bulk_copy_ptr(src_mem),
                cutlass.Int32(size),
                _bulk_copy_ptr(mbar),
            ],
            predicate=pred,
        )


@cute.jit
def tma_load_tile(
    smem_tile,
    gmem_slice,
    mbar,
    *,
    cta_group: int = 1,
    mcast_mask=None,
    acquire: cutlass.Constexpr[bool] = True,
    l2_cache_hint=None,
):
    num_iters = smem_tile.tma_loads_per_tile
    granu_elems = smem_tile.tma_granu_elems
    sub_stride = smem_tile.tma_subtile_stride_elems
    if cutlass.const_expr(gmem_slice.desc_ptr is not None):
        tma_desc_ptr = gmem_slice.desc_ptr
        if cutlass.const_expr(acquire):
            nvvm.fence_proxy_acquire(
                nvvm.MemScope.GPU,
                tma_desc_ptr,
                128,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    else:
        tma_desc_ptr = gmem_slice.tma_desc.get_ptr()
    coord_d = gmem_slice.coord_d
    outer_coords = tuple(gmem_slice.coords[1:])
    for i in cutlass.range_constexpr(num_iters):
        d = coord_d + cutlass.Int32(i * granu_elems)
        smem_chunk = smem_tile.base.subview(i * sub_stride)
        if nvvm.elect_sync():
            coords = [d] + list(outer_coords)
            if cutlass.const_expr(cta_group == 1):
                nvvm.cp_async_bulk_tensor_shared_cta_global(
                    smem_chunk,
                    tma_desc_ptr,
                    coords,
                    mbar,
                    l2_cache_hint=l2_cache_hint,
                )
            else:
                nvvm.cp_async_bulk_tensor_shared_cluster_global(
                    smem_chunk,
                    tma_desc_ptr,
                    coords,
                    mbar,
                    [],
                    multicast_mask=mcast_mask,
                    group=nvvm.CTAGroup.CTA_2,
                    l2_cache_hint=l2_cache_hint,
                )


@cute.jit
def tma_store_tile(smem_tile, gmem_slice, *, acquire: cutlass.Constexpr[bool] = True):
    num_iters = smem_tile.tma_loads_per_tile
    granu_elems = smem_tile.tma_granu_elems
    sub_stride = smem_tile.tma_subtile_stride_elems
    coord_d = gmem_slice.coord_d
    outer_coords = tuple(gmem_slice.coords[1:])
    if cutlass.const_expr(gmem_slice.desc_ptr is not None):
        tma_desc_ptr = gmem_slice.desc_ptr
        if cutlass.const_expr(acquire):
            nvvm.fence_proxy_acquire(
                nvvm.MemScope.GPU,
                tma_desc_ptr,
                128,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    else:
        tma_desc_ptr = gmem_slice.tma_desc.get_ptr()
    for i in cutlass.range_constexpr(num_iters):
        d = coord_d + cutlass.Int32(i * granu_elems)
        smem_chunk = smem_tile.base.subview(i * sub_stride)
        nvvm.cp_async_bulk_tensor_global_shared_cta(
            tma_desc_ptr,
            smem_chunk,
            tuple([d] + list(outer_coords)),
        )


@cute.jit
def bulk_copy(smem_dst, gmem_src, n_bytes, mbar):
    if nvvm.elect_sync():
        nvvm.cp_async_bulk_shared_cluster_global(
            smem_dst,
            gmem_src,
            mbar,
            n_bytes,
        )


@cute.jit
def bulk_copy_multicast(smem_dst, gmem_src, n_bytes, mbar, mcast_mask):
    if nvvm.elect_sync():
        cluster_dst = nvvm.mapa(smem_dst, cutlass.Int32(0), addrspace=7)
        nvvm.cp_async_bulk_shared_cluster_global(
            cluster_dst,
            gmem_src,
            mbar,
            n_bytes,
            multicast_mask=mcast_mask,
        )


@cute.jit
def tma_store_commit():
    nvvm.cp_async_bulk_commit_group()


@cute.jit
def tma_store_wait(num_remaining: int = 0):
    nvvm.cp_async_bulk_wait_group(num_remaining, read=True)


@cute.jit
def cp_async_commit():
    nvvm.cp_async_commit_group()


@cute.jit
def cp_async_wait(num_remaining: cutlass.Constexpr[int] = 0):
    nvvm.cp_async_wait_group(num_remaining)


@cute.jit
def load_tile(
    smem_dst,
    gmem_src,
    total_elems: cutlass.Constexpr[int],
    tidx,
    *,
    num_threads: cutlass.Constexpr[int],
    elems_per_copy: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
    cache: nvvm.LoadCacheModifier = nvvm.LoadCacheModifier.CG,
):
    bytes_per_copy = elems_per_copy * elem_bytes
    if cutlass.const_expr(bytes_per_copy not in (4, 8, 16)):
        raise ValueError(f"load_tile: elems_per_copy*elem_bytes must be 4/8/16, got " f"{elems_per_copy}*{elem_bytes}={bytes_per_copy}")
    chunk_elems = num_threads * elems_per_copy
    if cutlass.const_expr(total_elems % chunk_elems != 0):
        raise ValueError(
            f"load_tile: total_elems ({total_elems}) must be a multiple of " f"num_threads*elems_per_copy ({num_threads}*{elems_per_copy}=" f"{chunk_elems})"
        )
    n_iters = total_elems // chunk_elems
    base_off = tidx * elems_per_copy
    for i in cutlass.range_constexpr(n_iters):
        off = base_off + i * chunk_elems
        nvvm.cp_async_shared_global(
            _advance(smem_dst, off),
            _advance(gmem_src, off),
            bytes_per_copy,
            cache,
        )


@cute.jit
def load_tile_2d(
    smem_dst,
    gmem_src,
    rows: cutlass.Constexpr[int],
    elems_per_row: cutlass.Constexpr[int],
    gmem_row_stride_elems,
    tidx,
    *,
    num_threads: cutlass.Constexpr[int],
    elems_per_copy: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int],
    cache: nvvm.LoadCacheModifier = nvvm.LoadCacheModifier.CG,
    swizzle: cutlass.Constexpr[bool] = False,
    cp_size_bytes=None,
    valid_rows=None,
    valid_cols=None,
    row_base=None,
    col_base=None,
):
    bytes_per_copy = elems_per_copy * elem_bytes
    if cutlass.const_expr(bytes_per_copy not in (4, 8, 16)):
        raise ValueError(f"load_tile_2d: elems_per_copy*elem_bytes must be 4/8/16, got " f"{elems_per_copy}*{elem_bytes}={bytes_per_copy}")
    if cutlass.const_expr(elems_per_row % elems_per_copy != 0):
        raise ValueError(f"load_tile_2d: elems_per_row ({elems_per_row}) must be a multiple " f"of elems_per_copy ({elems_per_copy})")
    chunks_per_row = elems_per_row // elems_per_copy
    total_chunks = rows * chunks_per_row
    if cutlass.const_expr(total_chunks % num_threads != 0):
        raise ValueError(f"load_tile_2d: total_chunks ({rows}*{chunks_per_row}={total_chunks}) " f"must be a multiple of num_threads ({num_threads})")
    if cutlass.const_expr(cp_size_bytes is None):
        cp_size_bytes = bytes_per_copy
    predicate_active = cutlass.const_expr((valid_rows is not None) or (valid_cols is not None))
    n_iters = total_chunks // num_threads
    for i in cutlass.range_constexpr(n_iters):
        chunk_idx = i * num_threads + tidx
        row = chunk_idx // chunks_per_row
        col_elem = (chunk_idx % chunks_per_row) * elems_per_copy
        src = _advance(gmem_src, row * gmem_row_stride_elems + col_elem)
        if cutlass.const_expr(swizzle):
            smem_col = swizzle_xor_128b(row, col_elem, elem_bytes=elem_bytes)
        else:
            smem_col = col_elem
        dst = smem_dst.subview(row * elems_per_row + smem_col)
        if cutlass.const_expr(predicate_active):
            pred = cutlass.Int32(1)
            if cutlass.const_expr(valid_rows is not None):
                row_abs = row if row_base is None else row + row_base
                pred = pred * cutlass.Int32(row_abs < valid_rows)
            if cutlass.const_expr(valid_cols is not None):
                col_abs_end = col_elem + cutlass.Int32(elems_per_copy) if col_base is None else col_elem + col_base + cutlass.Int32(elems_per_copy)
                pred = pred * cutlass.Int32(col_abs_end <= valid_cols)
            cp_size_final = cp_size_bytes * pred
        else:
            cp_size_final = cp_size_bytes
        nvvm.cp_async_shared_global(dst, src, bytes_per_copy, cache, cp_size=cp_size_final)


@cute.jit
def tma_tensormap_acquire(desc_ptr):
    """Issue a single ``fence.proxy.tensormap::generic.acquire.gpu`` over a
    runtime TMA descriptor.

    A runtime descriptor written on the host by a descriptor-builder kernel
    (via ``tensormap_replace``) is visible to the TMA proxy only after this
    GENERIC->TENSORMAP acquire.  Such descriptors are built once (not rewritten
    per work-tile), so a single acquire per consumer CTA (or once per
    persistent-loop tile) suffices; call this once and pass ``acquire=False`` to
    the per-tile ``tma_load_tile`` / ``tma_store_tile`` wrappers to skip the
    redundant per-call fences.
    """
    nvvm.fence_proxy_acquire(
        nvvm.MemScope.GPU,
        desc_ptr,
        128,
        from_proxy=nvvm.Proxy.GENERIC,
        to_proxy=nvvm.Proxy.TENSORMAP,
    )
