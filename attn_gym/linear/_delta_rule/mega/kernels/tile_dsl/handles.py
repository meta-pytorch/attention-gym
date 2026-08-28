# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT


from dataclasses import dataclass, replace


@dataclass(frozen=True)
class MmaDesc:
    M: int
    N: int
    K: int
    bpe_a: int
    bpe_b: int
    tile_k_hw: int = 64
    btranspose: bool = False
    atranspose: bool = False
    k_subtile: int = -1
    cta_group: int = 1
    idesc: object = None
    kind: object = None
    is_block_scale: bool = False
    sf_blocks_per_step: int = 0
    sf_cycle: int = 0
    scale_vec_size: object = None

    def __post_init__(self):
        if self.k_subtile < 0:
            ks = self.K if self.btranspose else (128 // self.bpe_a if (self.K * self.bpe_a) % 128 == 0 else 64 // self.bpe_a)
            object.__setattr__(self, "k_subtile", ks)

    @property
    def m_per_cta(self):
        return self.M // self.cta_group

    @property
    def n_per_cta(self):
        return self.N // self.cta_group

    @property
    def num_k_steps(self):
        return self.K // self.tile_k_hw

    @property
    def steps_per_subtile(self):
        return self.k_subtile // self.tile_k_hw

    @property
    def num_subtiles(self):
        return self.num_k_steps // self.steps_per_subtile

    @staticmethod
    def _swz_from_inner(inner_bytes: int) -> int:
        if inner_bytes % 128 == 0:
            return 128
        if inner_bytes % 64 == 0:
            return 64
        return 32

    @property
    def swz_a_bytes(self):
        inner = self.m_per_cta if self.atranspose else self.K
        return MmaDesc._swz_from_inner(inner * self.bpe_a)

    @property
    def swz_b_bytes(self):
        inner = self.n_per_cta if self.btranspose else self.K
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
        return self.swz_a_bytes * self.m_per_cta

    @property
    def smem_subtile_B(self):
        if self.btranspose:
            return self.K * self.swz_b_bytes
        return self.swz_b_bytes * self.n_per_cta

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
    def num_subtiles_A(self):
        return self.num_k_steps // self.sps_A

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

    def __getitem__(self, stage):
        off = stage * self.elems_per_stage
        base = self.base.subview(off) if hasattr(self.base, "subview") else self.base + off
        return replace(self, base=base, stages=1)

    def shifted(self, off_elems):
        base = self.base.subview(off_elems) if hasattr(self.base, "subview") else self.base + off_elems
        return replace(self, base=base)

    def desc(self):
        from cutlass.experimental import primitives as prims

        return prims.Tcgen05SmemDesc.build(
            self.base,
            leading_byte_offset=self.leading_byte_offset,
            stride_byte_offset=self.stride_byte_offset,
            layout=self.layout,
        )


@dataclass(frozen=True)
class GmemTileTma:
    tma_desc: object

    def __call__(self, *coords, coord_0=None):
        if not coords:
            raise ValueError("GmemTileTma needs at least the innermost coord")
        if coord_0 is not None:
            if len(coords) != 4:
                raise ValueError(f"GmemTileTma 5-D form (coord_0=…) expects 4 positional " f"coords; got {len(coords)}")
            return GmemTileTmaSlice(
                tma_desc=self.tma_desc,
                coords=(coord_0,) + tuple(coords),
            )
        if not 2 <= len(coords) <= 5:
            raise ValueError(f"GmemTileTma supports rank 2..5; got {len(coords)} coords")
        return GmemTileTmaSlice(
            tma_desc=self.tma_desc,
            coords=tuple(coords),
        )


@dataclass(frozen=True)
class GmemTileTmaSlice:
    tma_desc: object
    coords: tuple
    desc_ptr: object = None

    @property
    def rank(self):
        return len(self.coords)

    @property
    def coord_d(self):
        return self.coords[0]

    def with_coord_d(self, new_d):
        return GmemTileTmaSlice(
            tma_desc=self.tma_desc,
            coords=(new_d,) + tuple(self.coords[1:]),
            desc_ptr=self.desc_ptr,
        )


def tma_slice_runtime_desc(desc_ptr, *coords):
    return GmemTileTmaSlice(tma_desc=None, coords=tuple(coords), desc_ptr=desc_ptr)


@dataclass(frozen=True)
class GmemTileLinear:
    base: object
    stride_b: int
    stride_h: int
    stride_tile: int

    def addr(self, batch, head, tile_idx):
        return self.base + batch * self.stride_b + head * self.stride_h + tile_idx * self.stride_tile
