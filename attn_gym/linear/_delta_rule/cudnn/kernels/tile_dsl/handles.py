# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: block-scale, multi-CTA, and static-descriptor handles unused
# by the vendored kernels were removed.


from dataclasses import dataclass, replace

import cutlass
from cutlass import cute
from cutlass.experimental import primitives as nvvm


@cute.jit
def smem_data_ptr(storage):
    """Return the raw shared-memory pointer for an Array or Tensor."""
    if cutlass.const_expr(hasattr(storage, "iterator")):
        pointer = storage.iterator.raw_ptr()
    else:
        pointer = storage.data_ptr()
    return pointer


@dataclass(frozen=True)
class MmaDesc:
    """Static geometry of one tcgen05 MMA: operand byte strides between K steps."""

    M: int
    N: int
    K: int
    bpe_a: int
    bpe_b: int
    tile_k_hw: int = 64
    btranspose: bool = False
    atranspose: bool = False
    cta_group: int = 1
    idesc: object = None
    kind: object = None

    def __post_init__(self):
        if self.cta_group != 1:
            raise ValueError("the vendored kernels only support cta_group=1")

    @property
    def num_k_steps(self):
        return self.K // self.tile_k_hw

    @staticmethod
    def _swz_from_inner(inner_bytes: int) -> int:
        if inner_bytes % 128 == 0:
            return 128
        if inner_bytes % 64 == 0:
            return 64
        return 32

    @property
    def swz_a_bytes(self):
        inner = self.M if self.atranspose else self.K
        return MmaDesc._swz_from_inner(inner * self.bpe_a)

    @property
    def swz_b_bytes(self):
        inner = self.N if self.btranspose else self.K
        return MmaDesc._swz_from_inner(inner * self.bpe_b)

    @property
    def smem_advance_A_intra(self):
        if self.atranspose:
            return self.tile_k_hw * self.swz_a_bytes
        return self.tile_k_hw * self.bpe_a

    @property
    def smem_advance_B_intra(self):
        if self.btranspose:
            return self.tile_k_hw * self.swz_b_bytes
        return self.tile_k_hw * self.bpe_b

    @property
    def smem_subtile_A(self):
        if self.atranspose:
            return self.K * self.swz_a_bytes
        return self.swz_a_bytes * self.M

    @property
    def smem_subtile_B(self):
        if self.btranspose:
            return self.K * self.swz_b_bytes
        return self.swz_b_bytes * self.N

    @property
    def sps_A(self):
        if self.atranspose:
            return self.K // self.tile_k_hw
        return (self.swz_a_bytes // self.bpe_a) // self.tile_k_hw

    @property
    def sps_B(self):
        if self.btranspose:
            return self.K // self.tile_k_hw
        return (self.swz_b_bytes // self.bpe_b) // self.tile_k_hw

    @property
    def num_subtiles_B(self):
        return self.num_k_steps // self.sps_B

    @property
    def tmem_advance_A(self):
        return self.tile_k_hw * self.bpe_a // 4


@dataclass(frozen=True)
class SmemTile:
    base: object
    elems_per_stage: int
    leading_byte_offset: int
    stride_byte_offset: int
    layout: int
    tma_loads_per_tile: int = 1
    tma_granu_elems: int = 0
    tma_subtile_stride_elems: int = 0
    stages: int = 1

    def _offset_base(self, offset):
        if hasattr(self.base, "iterator"):
            return cute.domain_offset((offset,), self.base)
        if hasattr(self.base, "subview"):
            return self.base.subview(offset)
        if hasattr(self.base, "data_ptr"):
            return self.base.data_ptr() + offset
        return self.base + offset

    def __getitem__(self, stage):
        return replace(
            self,
            base=self._offset_base(stage * self.elems_per_stage),
            stages=1,
        )

    def shifted(self, off_elems):
        return replace(self, base=self._offset_base(off_elems))

    def desc(self):
        if hasattr(self.base, "data_ptr"):
            base = self.base.data_ptr()
        elif hasattr(self.base, "iterator"):
            base = self.base.iterator.raw_ptr()
        else:
            base = self.base
        return nvvm.Tcgen05SmemDesc.build(
            base,
            leading_byte_offset=self.leading_byte_offset,
            stride_byte_offset=self.stride_byte_offset,
            layout=self.layout,
        )


@dataclass(frozen=True)
class GmemTileTmaSlice:
    """Runtime TMA-descriptor pointer plus the load/store coordinates."""

    coords: tuple
    desc_ptr: object

    @property
    def coord_d(self):
        return self.coords[0]


def tma_slice_runtime_desc(desc_ptr, *coords):
    return GmemTileTmaSlice(coords=tuple(coords), desc_ptr=desc_ptr)
