# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ragged TMA loads and stores: per-sequence clipping with descriptors based at each tensor.

TMA clips at tensor edges, not sequence boundaries. A packed ``(tokens, *rest)`` tensor viewed
as ``(min(tokens, edge), *rest, select)`` (``tail_view``), with the first and last modes both at
the token stride and the base at the tensor, clips a box of ``valid <= edge`` rows ending at
token ``end >= edge`` at that end: the box starts ``edge - valid`` rows into the first mode, so
its rows past ``valid`` fall outside it. Loads zero-fill those rows and stores drop them, as
cuDNN's per-sequence descriptors do, so no sequence reads or writes another's tokens.

A box ending before token ``edge`` (only a sequence ending inside the tensor's first ``edge``
tokens has one) clips only at token ``edge``. Its item gets a per-sequence descriptor, as upstream:
the issuing warp copies the tensor's descriptor into a global slot and replaces its token and
select extents with ``end`` and 1. Every TMA issues through a descriptor address chosen once per
item (``sequence_descriptors``): that slot, or the kernel parameter holding the shared
descriptor, so each site keeps one TMA path. Measured alternatives (GB300): a per-box branch
slowed the load warps (8x4096 H32: GDN prefill +7%, GDN bprop +4%, KDA bprop +10%); a second,
per-item copy of the load loops cost up to 11% (KDA bprop at d=64); rewriting the whole
descriptor (``update_tma_descriptor``) in the cold branch cost 1-3% on the GDN chain kernels;
and loading such boxes one row per TMA put 4096 TMAs on one tinv CTA (73 -> 233 us).

Keep descriptor bases at the tensor. A base one box before it (the former ``edge_view``) made
gdn_bprop's Q/K/V/dO loads fault on GB300 (Xid 31, MMU ``FAULT_PDE ACCESS_TYPE_VIRT_READ`` at
exactly ``tensor - 64`` rows: a TMA load reads its descriptor's base page even when no box
element lies there) whenever a tensor started its mapping, a caching-allocator segment start or
a ``cuMemMap`` mapping with unmapped VA before it; the same tensors 1 MiB into a larger buffer
loaded fine. A base ``2**30`` rows early (the grouped-GEMM trick) raised illegal addresses on
loads outright, and on stores (``UTMASTG``, "Warp Out of range Address") whenever another
context on the GPU hit a device-side assert while the kernel ran: kda_bprop's dq/dk/dv
stores faulted in 8 of 11 full-suite runs that included the warpdrive invalid-offset subprocess
tests and in 0 of 11 without them.
"""

import re
from typing import NamedTuple

import cutlass
import cutlass._mlir.dialects.cute as _cute_ir
import cutlass._mlir.dialects.cute_nvgpu as _cute_nvgpu_ir
import torch
from cutlass import cute
from cutlass.cute.arch import inline_ptx
from cutlass.cute.nvgpu import cpasync

from attn_gym.warpdrive.cache import make_fake_strided_tensor

DESCRIPTOR_WORDS = 16  # int64 words of one 128-byte TMA descriptor


@cute.jit
def tail_view(tensor: cute.Tensor, edge: cutlass.Constexpr[int]) -> cute.Tensor:
    """``(tokens, *rest)`` -> ``(min(tokens, edge), *rest, max(tokens - edge + 1, 1))``, the
    first and last modes at the token stride and the base at the tensor. ``tail_rows`` puts its
    clipping edge at a sequence end ``end >= edge`` (rows past it zero-fill or drop, the slack
    past the last sequence included); a box ending before token ``edge`` clips only there."""
    rows_stride = tensor.layout.stride[0]
    tokens = tensor.shape[0]
    shape = (cutlass.min(tokens, edge), *tensor.shape[1:], cutlass.max(tokens - edge + 1, 1))
    stride = (rows_stride, *tensor.layout.stride[1:], rows_stride)
    return cute.make_tensor(tensor.iterator, cute.make_layout(shape, stride=stride))


@cute.jit
def tail_rows(view: cute.Tensor, start, end, edge: cutlass.Constexpr[int]):
    """The rows from token ``start`` of a ``tail_view`` TMA coordinate tensor, as a tensor whose
    row ``i`` is token ``start + i``: rows at or past token ``max(end, edge)`` lie past the edge."""
    rank = cute.rank(view)
    select = cutlass.max(end - edge, 0)
    rows = view[(None,) * (rank - 1) + (select,)]
    return cute.domain_offset((start - select,) + (0,) * (rank - 2), rows)


class RaggedTma(NamedTuple):
    """TMA boxes of ``edge`` rows of a packed ``(token, cols..., head)`` tensor, clipped at each
    box's sequence end. ``atom`` and ``view`` are the ``tail_view`` descriptor and its TMA
    coordinates and ``slots`` the ``(edge, keys, DESCRIPTOR_WORDS)`` int64 global slots of the
    per-sequence descriptors (``slot_region``). Build with ``make_ragged_tma``."""

    atom: cute.CopyAtom
    view: cute.Tensor
    slots: cute.Tensor


def tma_slot_rows(tensors: int, edge: int, keys: int) -> int:
    """Rows of the ``(rows, DESCRIPTOR_WORDS)`` int64 slot buffer of a kernel with ``tensors``
    ragged tensors and ``keys`` item keys."""
    return tensors * edge * keys


def new_tma_slots(rows: int, device) -> torch.Tensor:
    """A slot buffer of ``rows`` descriptors (uninitialized: every slot is written before use).
    Kernels launched one after another may share one."""
    return torch.empty(max(rows, 1), DESCRIPTOR_WORDS, dtype=torch.int64, device=device)


def fake_tma_slots():
    """The compile-time stand-in of a slot buffer."""
    return make_fake_strided_tensor(
        cutlass.Int64,
        (cute.sym_int(), DESCRIPTOR_WORDS),
        stride_divisibility=DESCRIPTOR_WORDS,
        assumed_align=DESCRIPTOR_WORDS * 8,
    )


@cute.jit
def slot_region(
    slots: cute.Tensor, index: cutlass.Constexpr[int], edge: cutlass.Constexpr[int], keys
) -> cute.Tensor:
    """Ragged tensor ``index``'s ``(edge, keys, DESCRIPTOR_WORDS)`` region of a
    ``(rows, DESCRIPTOR_WORDS)`` slot buffer (``tma_slot_rows`` rows)."""
    shape = (edge, keys, DESCRIPTOR_WORDS)
    stride = (keys * DESCRIPTOR_WORDS, DESCRIPTOR_WORDS, 1)
    region = index * edge * keys * DESCRIPTOR_WORDS
    return cute.make_tensor(slots.iterator + region, cute.make_layout(shape, stride=stride))


@cute.jit
def make_ragged_tma(
    op,
    tensor: cute.Tensor,
    box_layout,
    edge: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    slots: cute.Tensor,
) -> RaggedTma:
    """The ``RaggedTma`` of a ``(token, cols..., head)`` tensor moving ``(edge, cols)`` SMEM
    boxes of layout ``box_layout`` with TMA op ``op``; ``slots`` from ``slot_region``."""
    atom, view = cpasync.make_tiled_tma_atom(op, tail_view(tensor, edge), box_layout, (edge, cols))
    return RaggedTma(atom, view, slots)


def _descriptor_type(space: cute.AddressSpace):
    # 64 bytes: the alignment of a descriptor kernel parameter (slots align to 128)
    return _cute_ir.PtrType.get(_cute_nvgpu_ir.TmaDescriptorTiledType.get(), space, 64)


@cutlass.dsl_user_op
def _shared_descriptor(rt: RaggedTma, *, loc=None, ip=None) -> cutlass.Int64:
    """The generic address of ``rt``'s shared descriptor (a kernel parameter)."""
    exec_atom = _cute_nvgpu_ir.atom_make_exec_tma(rt.atom._trait.value, loc=loc, ip=ip)
    ptr_type = _descriptor_type(cute.AddressSpace.generic)
    return _cute_nvgpu_ir.get_tma_desc_addr(ptr_type, exec_atom, loc=loc, ip=ip).toint()


@cutlass.dsl_user_op
def _descriptor_at(addr, space: cute.AddressSpace, *, loc=None, ip=None):
    """The descriptor pointer of address ``addr`` in ``space``."""
    value = cutlass.Int64(addr).ir_value(loc=loc, ip=ip)
    aligned = _cute_ir.assume(_cute_ir.ConstrainedIntType.get(64, 64), value, loc=loc, ip=ip)
    return _cute_ir.inttoptr(_descriptor_type(space), aligned, loc=loc, ip=ip)


def _clip_modes(rt: RaggedTma) -> tuple[int, int]:
    """The descriptor dimensions of ``rt``'s ``tail_view`` token and select modes (its first and
    last), read from the atom's ``tma_gbasis``: dimension ``i`` moves along mode ``k`` for a
    basis ``1@k``."""
    gbasis = re.search(r'tma_gbasis = <"\([^)]*\):\(([^)]*)\)">', str(rt.atom._trait.value.type))
    basis = gbasis.group(1).split(",") if gbasis else []
    last = cute.rank(rt.view) - 1
    if "1@0" not in basis or f"1@{last}" not in basis:
        raise NotImplementedError(f"ragged TMA modes not separate descriptor dims: {basis}")
    return basis.index("1@0"), basis.index(f"1@{last}")


@cute.jit
def _write_descriptors(rts: tuple, end, key, edge: cutlass.Constexpr[int]) -> tuple:
    """Each tensor's ``tensor[:end]`` tail-view descriptor in its slot ``(end, key)`` (``end <
    edge``: the shared one with token extent ``end`` and select extent 1); returns the slots'
    addresses."""
    gmem = cute.AddressSpace.gmem
    descs = [_descriptor_at(rt.slots[end, key, None].iterator.toint(), gmem) for rt in rts]
    with cute.arch.elect_one():
        for rt, desc in zip(rts, descs):
            tokens, select = _clip_modes(rt)
            cpasync.copy_tensormap(rt.atom, desc)
            replace = "tensormap.replace.tile.global_dim.global.b1024.b32 [$0], {}, $1;"
            inline_ptx(replace.format(tokens), read_only_args=[desc.toint(), end])
            inline_ptx(replace.format(select), read_only_args=[desc.toint(), cutlass.Int32(1)])
    cute.arch.sync_warp()
    inline_ptx("fence.proxy.tensormap::generic.release.cta;")
    for desc in descs:
        inline_ptx(
            "fence.proxy.tensormap::generic.acquire.cta [$0], 128;",
            read_only_args=[desc.toint()],
        )
    return tuple(desc.toint() for desc in descs)


@cute.jit
def sequence_descriptors(
    rts: tuple, end, key, edge: cutlass.Constexpr[int], moves_rows=True
) -> tuple:
    """One descriptor pointer per ``RaggedTma`` of ``rts`` for the ``ragged_load``/
    ``ragged_store`` calls of an item of the sequence ending at token ``end``: the shared
    descriptors unless the item ``moves_rows`` and ``end < edge``; then each tensor's
    descriptor clipped at ``end``, written to its slot ``(end, key)``. ``key`` must be unique
    among the launch's items that move rows of sequences ending at ``end``. All lanes of the
    warp that issues the item's TMAs call it, once per item.

    Ordering: one lane copies each tensor's descriptor into its slot and replaces the two
    extents (generic-proxy writes in program order), then the warp synchronizes; one
    tensormap-proxy release fence and each lane's acquire fence on every slot precede, in program
    order, every TMA the warp issues through them. A slot is written once per launch: only an
    item with rows of a sequence ending before token ``edge`` writes one, such a sequence has one
    chunk and so one such item per ``key``, and the item writes each slot once and reads it only
    from this warp. CTA scope suffices because those TMAs are this CTA's. Kernels sharing a slot
    buffer run one after another (stream order, or PDL with ``griddepcontrol.wait`` before any
    item, which waits for the previous grid and its TMAs to complete), so no slot is rewritten
    while a TMA reads it. One release fence covers all of an item's descriptors: each fence costs
    about a microsecond on the issuing warp."""
    addrs = tuple(_shared_descriptor(rt) for rt in rts)
    if moves_rows and end < edge:
        addrs = _write_descriptors(rts, end, key, edge)
    return tuple(descriptor_at(addr) for addr in addrs)


def descriptor_at(addr):
    """The descriptor pointer, for ``ragged_load``/``ragged_store``'s ``desc``, at the generic
    integer address ``addr`` (an entry of ``sequence_descriptors`` kept as an integer)."""
    return _descriptor_at(addr, cute.AddressSpace.generic)


@cute.jit
def _partition(rt: RaggedTma, smem: cute.Tensor, start, valid, head, col_box, edge, cols):
    rows = tail_rows(rt.view, start, start + valid, edge)[None, None, head]
    gmem = cute.local_tile(rows, (edge, cols), (0, col_box))
    return cpasync.tma_partition(
        rt.atom, 0, cute.make_layout(1), cute.group_modes(smem, 0, 2), cute.group_modes(gmem, 0, 2)
    )


@cute.jit
def ragged_load(
    rt: RaggedTma,
    dst: cute.Tensor,
    start,
    valid,
    head,
    col_box,
    barrier,
    edge: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    desc=None,
):
    """TMA column box ``col_box`` of the tokens ``[start, start + valid)`` of head ``head`` into
    the ``(edge, cols)`` SMEM box ``dst``, completing on ``barrier``; rows ``>= valid`` load as
    zeros. ``desc``: the item's ``sequence_descriptors`` entry for ``rt``; None issues through
    the shared descriptor, which clips a box ending before token ``edge`` only there."""
    s, g = _partition(rt, dst, start, valid, head, col_box, edge, cols)
    if cutlass.const_expr(desc is None):
        cute.copy(rt.atom, g, s, tma_bar_ptr=barrier)
    else:
        cute.copy(rt.atom, g, s, tma_bar_ptr=barrier, tma_desc_ptr=desc)


@cute.jit
def ragged_store(
    rt: RaggedTma,
    src: cute.Tensor,
    start,
    valid,
    head,
    col_box,
    edge: cutlass.Constexpr[int],
    cols: cutlass.Constexpr[int],
    desc=None,
):
    """TMA-store the ``(edge, cols)`` SMEM box ``src`` to column box ``col_box`` of the tokens
    ``[start, start + valid)`` of head ``head``; rows ``>= valid`` drop. ``desc`` as in
    ``ragged_load``."""
    s, g = _partition(rt, src, start, valid, head, col_box, edge, cols)
    if cutlass.const_expr(desc is None):
        cute.copy(rt.atom, s, g)
    else:
        cute.copy(rt.atom, s, g, tma_desc_ptr=desc)
