# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: cluster, grid-dependency, and predicated arrive paths unused
# by the vendored single-CTA kernels were removed.


import enum
from dataclasses import dataclass, replace
from typing import NamedTuple

import cutlass
from cutlass import cute
from cutlass.experimental import primitives as nvvm

WAIT_TIMEOUT = 1


class PipelineState(NamedTuple):
    idx: object
    phase: object

    @classmethod
    def start(cls, phase: int = 0):
        return cls(idx=cutlass.Int32(0), phase=cutlass.Int32(phase))


def advance(state, stages):
    if stages < 1:
        raise ValueError(f"PipelineState.advance requires stages >= 1, got {stages}")
    incr = state.idx + cutlass.Int32(1)
    stages_i = cutlass.Int32(stages)
    new_idx = incr % stages_i
    flip = incr // stages_i
    new_phase = state.phase ^ flip
    return PipelineState(idx=new_idx, phase=new_phase)


@cute.jit
def wait(mb, phase):
    while not nvvm.mbarrier_try_wait_parity(mb, phase, time_limit=WAIT_TIMEOUT):
        pass


@cute.jit
def arrive(mb):
    nvvm.mbarrier_arrive(mb)


@cute.jit
def arrive_expect_tx(mb, n_bytes):
    nvvm.mbarrier_arrive_expect_tx(mb, n_bytes)


@cute.jit
def commit_mma(mb):
    nvvm.tcgen05_commit(mb, group=nvvm.CTAGroup.CTA_1)


class Producer(enum.IntEnum):
    THREAD = 0
    TMA_LOAD = 1
    MMA_COMMIT = 2


@dataclass(frozen=True)
class MBarrier:
    base_ptr: object
    stages: cutlass.Constexpr[int]
    init_count: cutlass.Constexpr[object]
    producer: cutlass.Constexpr[int] = int(Producer.THREAD)
    stage_idx: object = 0

    def __getitem__(self, i):
        return replace(self, stage_idx=i)

    @property
    def smem_ptr(self):
        if isinstance(self.stage_idx, int) and self.stage_idx == 0:
            return self.base_ptr
        return self.base_ptr.subview(self.stage_idx)

    def init(self, override_count=None):
        if override_count is not None:
            count = override_count
        elif isinstance(self.init_count, (tuple, list)):
            if not isinstance(self.stage_idx, int):
                raise TypeError(
                    "MBarrier with tuple init_count requires a Python-int stage_idx "
                    f"(via [py_int]); got {type(self.stage_idx).__name__}."
                )
            count = int(self.init_count[self.stage_idx])
        else:
            count = int(self.init_count)
        nvvm.mbarrier_init(self.smem_ptr, count)

    def wait(self, phase):
        wait(self.smem_ptr, phase)

    def arrive(self, *, n_bytes=None, cta_group=None):
        """Arrive with the producer-specific op.

        ``cta_group`` is accepted for source compatibility with the upstream MMA-commit call
        sites; the vendored kernels only run single-CTA MMAs.
        """
        if cutlass.const_expr(self.producer == int(Producer.THREAD)):
            arrive(self.smem_ptr)
        elif cutlass.const_expr(self.producer == int(Producer.TMA_LOAD)):
            if n_bytes is None:
                raise TypeError("MBarrier(producer=TMA_LOAD).arrive() requires n_bytes=")
            arrive_expect_tx(self.smem_ptr, n_bytes)
        else:
            if cta_group is not None and cta_group != 1:
                raise ValueError("the vendored kernels only support cta_group=1")
            commit_mma(self.smem_ptr)
