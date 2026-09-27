# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe. Cluster, peer, and predicated arrive paths unused by the
# vendored single-CTA kernels were removed; arrive, expect-tx, init, and tcgen05.commit use the
# cute.arch wrappers. The mbarrier waits stay on the upstream primitives.


import enum
from dataclasses import dataclass, replace
from typing import NamedTuple

import cutlass
from cutlass import cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.cute.nvgpu import tcgen05
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
def wait_try(mb, phase):
    """Untimed ``mbarrier.try_wait.parity`` loop (no suspend hint, no inline PTX)."""
    while not nvvm.mbarrier_wait_parity(mb, phase, nvvm.MBarrierWait.TRY):
        pass


@cute.jit
def wait(mb, phase, spin: cutlass.Constexpr[bool] = False):
    """Spin on ``mb`` until its phase parity differs from ``phase``.

    ``spin=False`` is the DSL wrapper's ``mbarrier.try_wait.parity`` with the ``time_limit``
    suspend hint. ``spin=True`` is the hint-less C++ ``MBarrier::wait`` as inline PTX (the public
    wrapper always emits the ``.timelimit`` form). Both keep ``.acquire.cta`` ordering and the same
    termination condition; only the retry mechanics differ. Upstream measured the trade-off on
    sm_107a (spinning every wait loses on the linear-attention forward kernels); on sm_100a ptxas
    lowers the hint-less form to one divergent ``SYNCS.PHASECHK`` plus a branch.

    The inline-PTX labels are scoped to the ``{ }`` block, so the fixed names are legal at every
    instantiation. Do not pass ``predicate=`` (the DSL can lower it onto the last operand's value);
    branch around the loop instead.
    """
    if cutlass.const_expr(spin):
        nvvm.inline_ptx(
            "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
            "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [{$r0}], {$r1};\n\t"
            "@P1 bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}",
            read_only_args=[mb, cutlass.Int32(phase)],
        )
    else:
        while not nvvm.mbarrier_try_wait_parity(mb, phase, time_limit=WAIT_TIMEOUT):
            pass


@cute.jit
def arrive(mb):
    cute.arch.mbarrier_arrive(mb.data_ptr())


@cute.jit
def arrive_expect_tx(mb, n_bytes):
    cute.arch.mbarrier_arrive_and_expect_tx(mb.data_ptr(), n_bytes)


@cute.jit
def commit_mma(mb):
    tcgen05.commit(mb.data_ptr())


@cute.jit
def wait_on_dependent_grids():
    inline_ptx(
        "griddepcontrol.wait;",
        write_only_types=[],
        read_only_args=[],
    )


@cute.jit
def launch_dependent_grids():
    inline_ptx(
        "griddepcontrol.launch_dependents;",
        write_only_types=[],
        read_only_args=[],
    )


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
    try_wait: cutlass.Constexpr[bool] = False
    spin: cutlass.Constexpr[bool] = False
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
        cute.arch.mbarrier_init(self.smem_ptr.data_ptr(), count)

    def wait(self, phase):
        # A barrier declared with try_wait=True waits through the DSL wrapper's untimed try_wait
        # loop (the linear attention kernels: one SYNCS.PHASECHK + branch on sm100, no inline
        # PTX); spin=True selects the hint-less inline-PTX spin; the default is the timed form.
        if cutlass.const_expr(self.try_wait):
            wait_try(self.smem_ptr, phase)
        else:
            wait(self.smem_ptr, phase, spin=self.spin)

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
