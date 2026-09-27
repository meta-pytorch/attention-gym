# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.


import enum
from dataclasses import dataclass, field, replace
from typing import Callable, NamedTuple, Union

from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm
import cutlass
import cutlass.cute as cute

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
    """Untimed ``mbarrier.try_wait.parity`` loop through the DSL wrapper (no suspend hint, no inline PTX)."""
    while not nvvm.mbarrier_wait_parity(mb, phase, nvvm.MBarrierWait.TRY):
        pass


@cute.jit
def wait(mb, phase, spin: cutlass.Constexpr[bool] = False):
    """Spin on ``mb`` until its phase parity differs from ``phase``.

    ``spin=False`` (the default; an untouched call site compiles to the same cubin as before this kwarg existed) is the DSL
    wrapper's ``mbarrier.try_wait.parity`` with the ``time_limit`` suspend hint.  ``spin=True`` is the C++ reference's
    hint-less ``MBarrier::wait`` as inline PTX (the public wrapper always emits the 4-operand ``.timelimit`` form, so the
    hint-less one has to be spelled by hand).  Both arms keep ``.acquire.cta`` ordering, the same parity test and the same
    termination condition (spin until the phase flips; neither form has a real time limit, so ``timeout`` stays the hang
    detector).  Only the retry mechanics differ, and the difference is a PER-CALL-SITE choice, not a library default.

    MEASURED (host sm_107a trace-compiles at S=8K dense; perf on the Rubin node locked at 2376 MHz, A/B/A x3 with a control
    pair, CUPTI medians, TFLOPS ratios against develop): on sm_107a the default lowers per lane, ``SYNCS.PHASECHK.TRANS64.TRYWAIT
    P0 / @!P0 NANOSLEEP.SYNCS 0x1 / @!P0 SYNCS.PHASECHK / @!P0 BRA`` plus a ``BSSY`` reconverge (the warp is parked in hardware
    and woken by the phase event), while ``spin=True`` lowers to the uniform ``USYNCS.PHASECHK.TRANS64.TRYWAIT UP0 /
    BRA.U !UP0`` pair (two uniform instructions, no sleep; with every wait spun the d512 fp8 prefill goes 110 -> 8 divergent
    SYNCS.PHASECHK, 57 -> 6 NANOSLEEP, 57 -> 108 USYNCS.PHASECHK, 0 spills either way).  Spinning the per-KV-iteration RING
    waits gains +4.9 % on d128 bf16 @2K, +3.5 % on d512 fp8 @8K, +9.4 % on d192x128 mxfp8 @32K and +1.1..+4.3 % on d128
    mxfp8 / d192x128 f16 / d192x128 fp8 / d512 f16 (those four measured together with the predicated credit arrive), but
    LOSES -5.8 % @2K and -2.5 % @32K on d128 per-tensor fp8 and -2.2 % @32K on d512 mxfp8 (the loss follows the hint-less
    first check on the ring waits: keeping the sleeping retry after a hint-less first check, or spinning only the whole-tile
    idle waits, does not remove it), and spinning every wait is a loss on the linear-attention forward kernels (KDA -2.2 %
    @32K, GDP -5.0 % @8K).  On sm_100a ptxas lowers the hint-less form to ONE divergent ``SYNCS.PHASECHK`` plus a branch and
    emits no ``USYNCS.PHASECHK`` at all, so none of the sm_107a reasoning transfers and the sm100 kernels stay on the default.
    Hence the sm107 prefill kernels that gain opt their ring waits in through one module constant (``SPIN_RING_WAITS``);
    the waits a warp parks in for a whole tile (scheduler credits and CLC responses, the TMA-STG's O-ready wait,
    ``tmem_dealloc``, the end-of-kernel drains) and every other consumer keep the default.

    The inline-PTX labels are scoped to the ``{ }`` block, so the fixed names are legal at every instantiation.  No
    ``predicate=`` is passed (the DSL can lower one onto the last operand's VALUE, see ``arrive_expect_tx``); if this loop
    is ever predicated, branch around it instead.
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
    nvvm.mbarrier_arrive(mb)


@cute.jit
def arrive_expect_tx(mb, n_bytes, pred=None):
    if cutlass.const_expr(pred is None):
        nvvm.mbarrier_arrive_expect_tx(mb, n_bytes)
    else:
        # A branch round the native op, NOT nvvm.inline_ptx(predicate=...): the
        # DSL can lower that predicate to the last operand's *value* rather than
        # a predicate register, emitting PTX like
        #     @512 mbarrier.arrive.expect_tx.release.cta.shared.b64 %rd271, [%r202], 512;
        # which ptxas rejects as a syntax error, surfaced only as the generic
        # "NVVM backend compilation failed".  The two forms are equivalent here --
        # a thread whose predicate is false does not arrive either way.
        if pred:
            nvvm.mbarrier_arrive_expect_tx(mb, n_bytes)


@cute.jit
def cga_arrive():
    nvvm.barrier_cluster_arrive_relaxed_aligned()


@cute.jit
def cga_wait():
    nvvm.barrier_cluster_wait_aligned()


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


@cute.jit
def mbar_arrive_on_peer(mb, peer_cta_id, pred=None):
    peer_mb = nvvm.mapa(mb, peer_cta_id)
    if cutlass.const_expr(pred is None):
        nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CLUSTER, relaxed=True)
    else:
        nvvm.inline_ptx(
            "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [{$r0}];",
            read_only_args=[peer_mb.ir_value()],
            predicate=pred,
        )


@cute.jit
def arrive_on_leader(mb, leader_cta_id, cta_group: int):
    if cutlass.const_expr(cta_group == 1):
        nvvm.mbarrier_arrive(mb)
    else:
        peer_mb = nvvm.mapa(mb, leader_cta_id)
        nvvm.mbarrier_arrive(peer_mb, scope=nvvm.MemScope.CLUSTER, relaxed=True)


@cute.jit
def commit_mma(mb, mcast_mask, cta_group: int, pred=None):
    if cutlass.const_expr(pred is None):
        if cutlass.const_expr(cta_group == 1):
            nvvm.tcgen05_commit(mb, group=nvvm.CTAGroup.CTA_1)
        else:
            nvvm.tcgen05_commit(
                mb,
                multicast_mask=mcast_mask,
                group=nvvm.CTAGroup.CTA_2,
            )
    else:
        # Branch round the native ops rather than nvvm.inline_ptx(predicate=...);
        # see arrive_expect_tx above.  The multicast form has a second reason: a
        # constant-folded mask reaches the asm operand as an immediate, and the
        # emitted "tcgen05.commit...multicast::cluster.b64 [%r661], 3;" is
        # rejected by ptxas ("Arguments mismatch") because ctaMask must be a
        # register.  That is the integer twin of the float hazard opaque_f32_zero
        # documents.  The native op takes the mask as a value and keeps it in a
        # register, so both problems go away.
        if pred:
            if cutlass.const_expr(cta_group == 1):
                nvvm.tcgen05_commit(mb, group=nvvm.CTAGroup.CTA_1)
            else:
                nvvm.tcgen05_commit(mb, multicast_mask=mcast_mask, group=nvvm.CTAGroup.CTA_2)


class Producer(enum.IntEnum):
    THREAD = 0
    TMA_LOAD = 1
    MMA_COMMIT = 2
    LEADER = 3


class Scope(enum.IntEnum):
    LOCAL = 0
    LEADER = 1


_IntOrCountFn = Union[int, Callable[[int], int]]


@dataclass(frozen=True)
class MBarrier:
    base_ptr: object
    stages: cutlass.Constexpr[int]
    init_count: cutlass.Constexpr[object]
    producer: cutlass.Constexpr[int] = int(Producer.THREAD)
    scope: cutlass.Constexpr[int] = int(Scope.LOCAL)
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
                raise TypeError("MBarrier with tuple init_count requires a Python-int " f"stage_idx (via [py_int]); got " f"{type(self.stage_idx).__name__}.")
            count = int(self.init_count[self.stage_idx])
        else:
            count = int(self.init_count)
        nvvm.mbarrier_init(self.smem_ptr, count)

    def wait(self, phase, spin: bool = False):
        # spin=True opts THIS call site into the hint-less uniform spin (see wait()); a barrier declared with spin=True opts
        # every wait on it in; the default is the sleeping form.  A barrier declared with try_wait=True waits through the
        # DSL wrapper's untimed try_wait loop instead (the linear attention kernels: one SYNCS.PHASECHK + branch on sm100,
        # no inline PTX, so their cubins are unchanged).
        if cutlass.const_expr(self.try_wait):
            wait_try(self.smem_ptr, phase)
        else:
            wait(self.smem_ptr, phase, spin=spin or self.spin)

    def arrive(
        self,
        *,
        n_bytes=None,
        mcast_mask=None,
        cta_group=None,
        leader_cta_id=None,
        pred=None,
    ):
        if cutlass.const_expr(self.producer == int(Producer.THREAD)):
            if cutlass.const_expr(pred is not None):
                raise TypeError(
                    "MBarrier(producer=THREAD).arrive() does NOT support pred= "
                    "(silently over-arrives). Keep the `if nvvm.elect_sync():` "
                    "branch around the plain arrive; only TMA_LOAD (expect_tx) "
                    "and MMA_COMMIT (commit) arrives have a predicated path."
                )
            arrive(self.smem_ptr)

        elif cutlass.const_expr(self.producer == int(Producer.TMA_LOAD)):
            if n_bytes is None:
                raise TypeError("MBarrier(producer=TMA_LOAD).arrive() requires n_bytes=")
            arrive_expect_tx(self.smem_ptr, n_bytes, pred=pred)

        elif cutlass.const_expr(self.producer == int(Producer.MMA_COMMIT)):
            if cta_group is None:
                raise TypeError("MBarrier(producer=MMA_COMMIT).arrive() requires " "cta_group= (Python compile-time int).")
            commit_mma(self.smem_ptr, mcast_mask, cta_group, pred=pred)

        else:
            if cta_group is None or leader_cta_id is None:
                raise TypeError("MBarrier(producer=LEADER).arrive() requires " "cta_group= AND leader_cta_id=.")
            if cutlass.const_expr(pred is not None):
                raise TypeError(
                    "MBarrier(producer=LEADER).arrive() does NOT support pred=. "
                    "Use arrive_on_peer(pred=) for a predicated cross-CTA arrive, "
                    "or keep the `if elect_sync():` branch."
                )
            arrive_on_leader(self.smem_ptr, leader_cta_id, cta_group)

    def arrive_on_peer(self, peer_cta_id, pred=None):
        mbar_arrive_on_peer(self.smem_ptr, peer_cta_id, pred=pred)
