"""Microbenchmarks of warpdrive's fixed costs against hand-written CuTeDSL twins.

Each probe exists twice: once with warpdrive channels and roles, once with raw ``cute.arch``
barriers in a hand-written kernel that does the same work.

``--suite handoff`` (lane 0 of the timed warps writes ``clock64`` and ``globaltimer`` stamps):

* ``pingpong``: one round trip per iteration between two warps (``a -> b`` on ``fwd``,
  ``b -> a`` on ``back``), reported as cycles per one-way handoff. Kinds: ``threads``
  (mbarrier, Threads both sides), ``named`` (``bar.arrive``/``bar.sync``), ``umma``
  (``tcgen05.commit`` both directions), ``tma`` (a 512-byte TMA load completes the full barrier)
  and ``round_robin`` (two consumer instances alternate).
* ``stream``: ``a`` produces, ``b`` consumes, no work: cycles per item through an ``n``-stage
  ring (steady-state throughput).

The raw twins take a wait style: ``wait`` is ``cute.arch.mbarrier_wait`` (what the layer emits),
``try`` is one hintless ``mbarrier.try_wait`` before it (CUTLASS's try-wait token, as FA4 uses).
Timing the difference between ``n`` and ``2n`` iterations cancels loop entry and exit.

``--suite fixed``: ``EmptyRoles``, the decode example's 16-warp role layout, rings (48
mbarriers), TMEM allocation and OneShot work stream with empty role bodies, against
``RawEmptyRoles`` allocating TMEM from warp 0 or warp 1. CUPTI kernel durations of graph
replays (16 and 148 CTAs) and per-warp ``clock64`` phases: entry, pre-sync, role start, role
end and kernel end, as medians of cycles after each CTA's first entry stamp.

Results and the changes they ruled out are in ``docs/warpdrive/examples.md`` (warpdrive fixed
costs).

Run under ``gpu-run``:
``python -m attn_gym.warpdrive.examples.bench_overhead --output results.json``.
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Int32, Int64, cute, pipeline
from cutlass._mlir.dialects import nvvm
from cutlass.cute.nvgpu import cpasync

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.compat import (
    FastDivmodDivisor,
    SmemAllocator,
    TmemAllocator,
)
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Consumer,
    Instance,
    OneShot,
    Producer,
    Ring,
    Sharing,
    SmemPlan,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkMap,
    WorkStream,
    advance_state,
    role,
)

TMA_ELEMS = 256  # one TMA box: 512 bytes of bf16
TMA_BYTES = TMA_ELEMS * 2
HANDOFF_KINDS = ("threads", "named", "umma", "tma", "round_robin")
STREAM_KINDS = ("threads", "named", "umma", "tma")
WAITS = ("wait", "try")
# stamps: clock start/end, globaltimer start/end of the timed interval
NUM_STAMPS = 4


@dataclass(frozen=True)
class ProbeCtx:
    stamps: cute.Tensor
    n: Int32
    tma: cute.CopyAtom | None
    gsrc: cute.Tensor


@cute.jit
def stamp(stamps, at: cutlass.Constexpr, value):
    if cute.arch.lane_idx() == 0:
        stamps[at] = Int64(value)


def tma_load(ctx: ProbeCtx, dst, bar):
    """One 512-byte TMA load of ``gsrc`` into ``dst`` (a ``(TMA_ELEMS,)`` SMEM tensor)."""
    s, g = cpasync.tma_partition(
        ctx.tma,
        0,
        cute.make_layout(1),
        cute.group_modes(dst, 0, 1),
        cute.zipped_divide(ctx.gsrc, (TMA_ELEMS,)),
    )
    cute.copy(ctx.tma, g[None, 0], s, tma_bar_ptr=bar)


def fwd_producer(kind: str):
    return {"umma": Umma, "tma": Tma(TMA_BYTES)}.get(kind, Threads)


class ProbeBase(WarpSpecialized):
    """Launch plumbing shared by the warpdrive probes (one CTA)."""

    def get_name(self) -> str:
        return f"warpdrive_overhead_{type(self).__name__.lower()}_{self.kind}_n{self.stages}"

    def tma_buffer(self, kind: str):
        """The TMA kind's staged destination: a 128-byte aligned plan buffer."""
        if kind != "tma":
            return None
        self.smem_plan = SmemPlan()
        return self.smem_plan.buffer("T", BFloat16, align=128)

    @cute.kernel
    def kernel(self, stamps, n, tma, gsrc):
        layouts = {}
        if cutlass.const_expr(self.kind == "tma"):
            layouts = {"T": cute.make_layout((TMA_ELEMS, self.stages))}
        self.begin_roles(layouts=layouts)
        self.run_roles(ProbeCtx(stamps, n, tma, gsrc))

    @cute.jit
    def __call__(self, stamps: cute.Tensor, src: cute.Tensor, n: Int32, stream: cuda.CUstream):
        tma, gsrc = make_tma(src)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(stamps, n, tma, gsrc).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


def make_tma(src):
    return cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), src, cute.make_layout(TMA_ELEMS), (TMA_ELEMS,)
    )


class PingPong(ProbeBase):
    """``a`` (warp 0) -> ``b`` (warp 1) on ``fwd``, back on ``back``; ``ctx.n`` round trips."""

    def __init__(self, kind: str = "threads", stages: int = 1):
        self.kind, self.stages = kind, stages
        sync = "named" if kind == "named" else "mbarrier"
        buffer = self.tma_buffer(kind)
        back_consumer = Umma if kind == "umma" else Threads
        # The consumer never reads the stage: no release fence, like the hand-written twins.
        self.fwd = Ring(
            stages, fwd_producer(kind), Threads, sync=sync, buffer=buffer, release_fence=()
        )
        self.back = Ring(stages, Threads, back_consumer, sync=sync)

    @role(warps=0)
    @cute.jit
    def a(self, ctx: ProbeCtx, *, fwd: Producer, back: Consumer):
        t0, g0 = cute.arch.clock64(), cute.arch.globaltimer()
        for _ in cutlass.range(ctx.n, unroll=1):
            h = fwd.acquire()
            if cutlass.const_expr(self.kind == "tma"):
                tma_load(ctx, h.data, h.barrier)
            h.commit()
            with back.consume():
                pass
        t1, g1 = cute.arch.clock64(), cute.arch.globaltimer()
        stamp(ctx.stamps, 0, t0)
        stamp(ctx.stamps, 1, t1)
        stamp(ctx.stamps, 2, g0)
        stamp(ctx.stamps, 3, g1)
        return fwd, back

    @role(warps=1)
    @cute.jit
    def b(self, ctx: ProbeCtx, *, fwd: Consumer, back: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with fwd.consume():
                pass
            with back.produce():
                pass
        return fwd, back


class RoundRobinPingPong(ProbeBase):
    """``a`` hands epoch i to ``b[i % 2]``, which hands it back; two-instance round robin."""

    def __init__(self, kind: str = "round_robin", stages: int = 2):
        self.kind, self.stages = kind, stages
        self.fwd = Ring(stages, Threads, Threads, consumer_sharing=Sharing.ROUND_ROBIN)
        self.back = Ring(stages, Threads, Threads, producer_sharing=Sharing.ROUND_ROBIN)

    @role(warps=0)
    @cute.jit
    def a(self, ctx: ProbeCtx, *, fwd: Producer, back: Consumer):
        t0, g0 = cute.arch.clock64(), cute.arch.globaltimer()
        for _ in cutlass.range(ctx.n, unroll=1):
            with fwd.produce():
                pass
            with back.consume():
                pass
        t1, g1 = cute.arch.clock64(), cute.arch.globaltimer()
        stamp(ctx.stamps, 0, t0)
        stamp(ctx.stamps, 1, t1)
        stamp(ctx.stamps, 2, g0)
        stamp(ctx.stamps, 3, g1)
        return fwd, back

    @role(warps=[range(1, 2), range(2, 3)])
    @cute.jit
    def b(self, ctx: ProbeCtx, *, stage: Instance, fwd: Consumer, back: Producer):
        for _ in cutlass.range(stage, ctx.n, 2, unroll=1):
            with fwd.consume():
                pass
            with back.produce():
                pass
        return fwd, back


class Stream(ProbeBase):
    """``a`` produces ``ctx.n`` items into an ``stages``-stage ring, ``b`` consumes them."""

    def __init__(self, kind: str = "threads", stages: int = 4):
        self.kind, self.stages = kind, stages
        sync = "named" if kind == "named" else "mbarrier"
        buffer = self.tma_buffer(kind)
        # The consumer never reads the stage: no release fence, like the hand-written twins.
        self.fwd = Ring(
            stages, fwd_producer(kind), Threads, sync=sync, buffer=buffer, release_fence=()
        )

    @role(warps=0)
    @cute.jit
    def a(self, ctx: ProbeCtx, *, fwd: Producer):
        t0, g0 = cute.arch.clock64(), cute.arch.globaltimer()
        for _ in cutlass.range(ctx.n, unroll=1):
            h = fwd.acquire()
            if cutlass.const_expr(self.kind == "tma"):
                tma_load(ctx, h.data, h.barrier)
            h.commit()
        stamp(ctx.stamps, 0, t0)
        stamp(ctx.stamps, 2, g0)
        return fwd

    @role(warps=1)
    @cute.jit
    def b(self, ctx: ProbeCtx, *, fwd: Consumer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with fwd.consume():
                pass
        t1, g1 = cute.arch.clock64(), cute.arch.globaltimer()
        stamp(ctx.stamps, 1, t1)
        stamp(ctx.stamps, 3, g1)
        return fwd


# --------------------------------------------------------------------------- #
# Hand-written twins
# --------------------------------------------------------------------------- #


@cute.jit
def raw_wait(style: cutlass.Constexpr, bar, phase):
    """mbarrier wait in one of ``WAITS`` styles."""
    if cutlass.const_expr(style == "try"):
        if not cute.arch.mbarrier_try_wait(bar, phase):
            cute.arch.mbarrier_wait(bar, phase)
    else:
        cute.arch.mbarrier_wait(bar, phase)


def raw_arrive(kind: str, bar):
    """``tcgen05.commit`` from one elected lane for ``umma``; a per-thread arrive otherwise."""
    if cutlass.const_expr(kind == "umma"):
        with cute.arch.elect_one():
            cute.nvgpu.tcgen05.commit(bar)
    else:
        cute.arch.mbarrier_arrive(bar)


def named_sync(barrier_id, threads: int):
    """Unaligned ``barrier.cta.sync``, as the layer's named channels emit."""
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(), thread_count=Int32(threads).ir_value(), aligned=False
    )


def named_arrive(barrier_id, threads: int):
    cute.arch.barrier_arrive(barrier_id=barrier_id, number_of_threads=threads, aligned=False)


class RawProbe:
    """A hand-written twin of ``PingPong`` (``mode="pingpong"``), ``RoundRobinPingPong``
    (``kind="round_robin"``) or ``Stream`` (``mode="stream"``)."""

    def __init__(self, kind: str, stages: int, mode: str, wait: str):
        self.kind, self.stages, self.mode = kind, stages, mode
        self.wait_style = wait
        self.num_warps = 3 if kind == "round_robin" else 2

    def get_name(self) -> str:
        return f"raw_overhead_{self.mode}_{self.kind}_n{self.stages}_{self.wait_style}"

    @cute.jit
    def __call__(self, stamps: cute.Tensor, src: cute.Tensor, n: Int32, stream: cuda.CUstream):
        tma, gsrc = make_tma(src)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(stamps, n, tma, gsrc).launch(
            grid=(1, 1, 1), block=(32 * self.num_warps, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, stamps, n, tma, gsrc):
        ctx = ProbeCtx(stamps, n, tma, gsrc)
        stages, kind = self.stages, self.kind
        smem = SmemAllocator()
        bars = smem.allocate_array(Int64, 4 * stages)
        buf = smem.allocate_tensor(BFloat16, cute.make_layout((TMA_ELEMS, stages)), 128)
        full_f, empty_f, full_b, empty_b = (bars + i * stages for i in range(4))
        threads = 32
        fwd_full = 1 if kind in ("umma", "tma") else threads
        back_empty = 1 if kind == "umma" else threads
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(stages):
                    cute.arch.mbarrier_init(full_f + i, fwd_full)
                    cute.arch.mbarrier_init(empty_f + i, threads)
                    cute.arch.mbarrier_init(full_b + i, threads)
                    cute.arch.mbarrier_init(empty_b + i, back_empty)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        if cutlass.const_expr(kind == "named"):
            self.named_roles(ctx, warp)
        elif cutlass.const_expr(kind == "round_robin"):
            self.round_robin_roles(ctx, warp, full_f, empty_f, full_b, empty_b)
        else:
            self.mbarrier_roles(ctx, warp, buf, full_f, empty_f, full_b, empty_b)

    @cute.jit
    def mbarrier_roles(self, ctx, warp, buf, full_f, empty_f, full_b, empty_b):
        kind, stages, style = self.kind, self.stages, self.wait_style
        prod = pipeline.PipelineUserType.Producer
        cons = pipeline.PipelineUserType.Consumer
        if warp == 0:
            f = pipeline.make_pipeline_state(prod, stages)
            b = pipeline.make_pipeline_state(cons, stages)
            t0, g0 = cute.arch.clock64(), cute.arch.globaltimer()
            for _ in cutlass.range(ctx.n, unroll=1):
                raw_wait(style, empty_f + f.index, f.phase)
                if cutlass.const_expr(kind == "tma"):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full_f + f.index, TMA_BYTES)
                    tma_load(ctx, buf[None, f.index], full_f + f.index)
                else:
                    raw_arrive(kind, full_f + f.index)
                f.advance()
                if cutlass.const_expr(self.mode == "pingpong"):
                    raw_wait(style, full_b + b.index, b.phase)
                    raw_arrive(kind, empty_b + b.index)
                    b.advance()
            t1, g1 = cute.arch.clock64(), cute.arch.globaltimer()
            stamp(ctx.stamps, 0, t0)
            stamp(ctx.stamps, 2, g0)
            if cutlass.const_expr(self.mode == "pingpong"):
                stamp(ctx.stamps, 1, t1)
                stamp(ctx.stamps, 3, g1)
        if warp == 1:
            f = pipeline.make_pipeline_state(cons, stages)
            b = pipeline.make_pipeline_state(prod, stages)
            for _ in cutlass.range(ctx.n, unroll=1):
                raw_wait(style, full_f + f.index, f.phase)
                cute.arch.mbarrier_arrive(empty_f + f.index)
                f.advance()
                if cutlass.const_expr(self.mode == "pingpong"):
                    raw_wait(style, empty_b + b.index, b.phase)
                    cute.arch.mbarrier_arrive(full_b + b.index)
                    b.advance()
            if cutlass.const_expr(self.mode == "stream"):
                t1, g1 = cute.arch.clock64(), cute.arch.globaltimer()
                stamp(ctx.stamps, 1, t1)
                stamp(ctx.stamps, 3, g1)

    @cute.jit
    def round_robin_roles(self, ctx, warp, full_f, empty_f, full_b, empty_b):
        stages, style = self.stages, self.wait_style
        prod = pipeline.PipelineUserType.Producer
        cons = pipeline.PipelineUserType.Consumer
        if warp == 0:
            f = pipeline.make_pipeline_state(prod, stages)
            b = pipeline.make_pipeline_state(cons, stages)
            t0, g0 = cute.arch.clock64(), cute.arch.globaltimer()
            for _ in cutlass.range(ctx.n, unroll=1):
                raw_wait(style, empty_f + f.index, f.phase)
                cute.arch.mbarrier_arrive(full_f + f.index)
                f.advance()
                raw_wait(style, full_b + b.index, b.phase)
                cute.arch.mbarrier_arrive(empty_b + b.index)
                b.advance()
            t1, g1 = cute.arch.clock64(), cute.arch.globaltimer()
            stamp(ctx.stamps, 0, t0)
            stamp(ctx.stamps, 1, t1)
            stamp(ctx.stamps, 2, g0)
            stamp(ctx.stamps, 3, g1)
        else:
            turn = warp - 1
            f = pipeline.make_pipeline_state(cons, stages)
            b = pipeline.make_pipeline_state(prod, stages)
            if turn == 1:
                f.advance()
                b.advance()
            for _ in cutlass.range(turn, ctx.n, 2, unroll=1):
                raw_wait(style, full_f + f.index, f.phase)
                cute.arch.mbarrier_arrive(empty_f + f.index)
                f = advance_state(f, 2)
                raw_wait(style, empty_b + b.index, b.phase)
                cute.arch.mbarrier_arrive(full_b + b.index)
                b = advance_state(b, 2)

    @cute.jit
    def named_roles(self, ctx, warp):
        # Barrier IDs 1-4 per stage: fwd full/empty, back full/empty; 64 threads each.
        stages = self.stages
        both = 64
        if warp == 0:
            if cutlass.const_expr(self.mode == "pingpong"):
                for s in cutlass.range_constexpr(stages):
                    named_arrive(1 + 4 * s + 3, both)  # seed back empty
            t0, g0 = cute.arch.clock64(), cute.arch.globaltimer()
            for i in cutlass.range(ctx.n, unroll=1):
                s = i % stages
                named_sync(1 + 4 * s + 1, both)  # fwd empty
                named_arrive(1 + 4 * s, both)  # fwd full
                if cutlass.const_expr(self.mode == "pingpong"):
                    named_sync(1 + 4 * s + 2, both)  # back full
                    named_arrive(1 + 4 * s + 3, both)  # back empty
            t1, g1 = cute.arch.clock64(), cute.arch.globaltimer()
            stamp(ctx.stamps, 0, t0)
            stamp(ctx.stamps, 2, g0)
            if cutlass.const_expr(self.mode == "pingpong"):
                stamp(ctx.stamps, 1, t1)
                stamp(ctx.stamps, 3, g1)
        if warp == 1:
            for s in cutlass.range_constexpr(stages):
                named_arrive(1 + 4 * s + 1, both)  # seed fwd empty
            for i in cutlass.range(ctx.n, unroll=1):
                s = i % stages
                named_sync(1 + 4 * s, both)
                named_arrive(1 + 4 * s + 1, both)
                if cutlass.const_expr(self.mode == "pingpong"):
                    named_sync(1 + 4 * s + 3, both)
                    named_arrive(1 + 4 * s + 2, both)
            if cutlass.const_expr(self.mode == "stream"):
                t1, g1 = cute.arch.clock64(), cute.arch.globaltimer()
                stamp(ctx.stamps, 1, t1)
                stamp(ctx.stamps, 3, g1)


# --------------------------------------------------------------------------- #
# Fixed costs: a decode-shaped kernel whose roles do no work
# --------------------------------------------------------------------------- #

# The decode example's mbarrier rings (q, k, v, s, p, o, epi): 24 stages, 48 barriers.
FIXED_RINGS = (1, 7, 5, 4, 4, 2, 1)
FIXED_WARPS = 16
TMEM_THREADS = 448  # warps 0, 1 and 4-15
# stamps per warp: entry, pre-sync, role start, role end, kernel end
FIXED_SLOTS = 5


@dataclass(frozen=True)
class GridItems(WorkMap):
    """``(i % a, (i // a) % b, i // (a * b))`` through launch-built fast divisors, as the decode
    example's split mapping does."""

    words = 3

    def bind(self, space):
        a, b, c = space
        return (a, b, c, FastDivmodDivisor(a), FastDivmodDivisor(b))

    def count64(self, space):
        a, b, c, *_ = space
        return Int64(a) * b * c

    def decode(self, index, space, state):
        return True, grid_item(index, space[3], space[4]), state

    def get_key(self) -> str:
        return "grid_items"


def grid_item(index, div_a, div_b):
    rest, i = divmod(index, div_a)
    k, j = divmod(rest, div_b)
    return i, j, k


@cute.jit
def stamp_warp(stamps, slot: cutlass.Constexpr):
    """Every lane stores the same clock: a lane-divergent store in the prologue is avoided."""
    cta = cute.arch.block_idx()[0]
    warp = cute.arch.warp_idx()
    stamps[(cta * FIXED_WARPS + warp) * FIXED_SLOTS + slot] = cute.arch.clock64()


@dataclass(frozen=True)
class FixedCtx:
    stamps: cute.Tensor
    out: cute.Tensor


@cute.jit
def touch(ctx: FixedCtx, item):
    """Keep the decoded item live: a never-taken store."""
    i, j, k = item
    if i + j + k < 0:
        ctx.out[0] = i


class EmptyRoles(WarpSpecialized):
    """The decode example's role layout, rings, TMEM allocation and OneShot work stream, with
    empty role bodies: what remains is warpdrive's prologue, dispatch, cursor and teardown."""

    def __init__(self):
        for i, n in enumerate(FIXED_RINGS):
            setattr(self, f"c{i}", Ring(n, Threads, Threads))
        self.tmem_plan = TmemPlan().region("acc", 256)
        self.work = WorkStream(mapping=GridItems(), schedule=OneShot())

    def get_name(self) -> str:
        return "warpdrive_overhead_empty_roles"

    @role(warps=0)
    @cute.jit
    def a(
        self,
        ctx: FixedCtx,
        *,
        c0: Producer,
        c1: Producer,
        c2: Producer,
        c3: Producer,
        c4: Producer,
        c5: Producer,
        c6: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        stamp_warp(ctx.stamps, 2)
        while work.valid:
            touch(ctx, work.item)
            work.advance()
        stamp_warp(ctx.stamps, 3)
        return c0, c1, c2, c3, c4, c5, c6, work

    @role(warps=1)
    @cute.jit
    def b(
        self,
        ctx: FixedCtx,
        *,
        c0: Consumer,
        c1: Consumer,
        c2: Consumer,
        c3: Consumer,
        c4: Consumer,
        c5: Consumer,
        c6: Consumer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        stamp_warp(ctx.stamps, 2)
        while work.valid:
            touch(ctx, work.item)
            work.advance()
        stamp_warp(ctx.stamps, 3)
        return c0, c1, c2, c3, c4, c5, c6, work

    @role(warps=[range(2, 3), range(3, 4)])
    @cute.jit
    def c(self, ctx: FixedCtx, *, work: WorkCursor):
        stamp_warp(ctx.stamps, 2)
        while work.valid:
            touch(ctx, work.item)
            work.advance()
        stamp_warp(ctx.stamps, 3)
        return work

    @role(warps=[range(4, 8), range(8, 12)])
    @cute.jit
    def d(self, ctx: FixedCtx, *, work: WorkCursor, tmem: Tmem):
        stamp_warp(ctx.stamps, 2)
        while work.valid:
            touch(ctx, work.item)
            work.advance()
        stamp_warp(ctx.stamps, 3)
        return work

    @role(warps=range(12, 16))
    @cute.jit
    def e(self, ctx: FixedCtx, *, work: WorkCursor, tmem: Tmem):
        stamp_warp(ctx.stamps, 2)
        while work.valid:
            touch(ctx, work.item)
            work.advance()
        stamp_warp(ctx.stamps, 3)
        return work

    @cute.kernel
    def kernel(self, stamps, out, bound):
        stamp_warp(stamps, 0)
        self.begin_roles(work=bound)
        stamp_warp(stamps, 1)
        self.run_roles(FixedCtx(stamps, out))
        stamp_warp(stamps, 4)

    @cute.jit
    def __call__(self, stamps: cute.Tensor, out: cute.Tensor, ctas: Int32, stream: cuda.CUstream):
        bound = self.work.bind((ctas, Int32(1), Int32(1)))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(stamps, out, bound).launch(**self.launch_config(bound), stream=stream)


class RawEmptyRoles:
    """``EmptyRoles`` by hand, FA4-style: 48 mbarriers initialized by warp 0, TMEM allocated by
    ``alloc_warp``, one CTA barrier, one item decode, a branch on the warp, then the TMEM users'
    teardown barrier and dealloc."""

    def __init__(self, alloc_warp: int = 0):
        self.alloc_warp = alloc_warp

    def get_name(self) -> str:
        return f"raw_overhead_empty_roles_w{self.alloc_warp}"

    @cute.jit
    def __call__(self, stamps: cute.Tensor, out: cute.Tensor, ctas: Int32, stream: cuda.CUstream):
        div_a, div_b = FastDivmodDivisor(ctas), FastDivmodDivisor(Int32(1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(stamps, out, div_a, div_b).launch(
            grid=(ctas, 1, 1), block=(32 * FIXED_WARPS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, stamps, out, div_a, div_b):
        ctx = FixedCtx(stamps, out)
        stamp_warp(stamps, 0)
        smem = SmemAllocator()
        bars = smem.allocate_array(Int64, 2 * sum(FIXED_RINGS))
        holder = smem.allocate_array(Int32, 1)
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(2 * sum(FIXED_RINGS)):
                    cute.arch.mbarrier_init(bars + i, 32)
        tmem = TmemAllocator(
            holder,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=512),
            allocator_warp_id=self.alloc_warp,
        )
        tmem.allocate(256)
        tmem.relinquish_alloc_permit()
        cute.arch.mbarrier_init_fence()
        stamp_warp(stamps, 1)
        cute.arch.sync_threads()
        item = grid_item(cute.arch.block_idx()[0], div_a, div_b)
        uses_tmem = (warp < 2) | (warp >= 4)
        if (warp >= 2) & (warp < 4):
            self.role(ctx, item, None)
        else:
            self.role(ctx, item, tmem.retrieve_ptr(cutlass.Float32))
        if uses_tmem:
            cute.arch.barrier(barrier_id=2, number_of_threads=TMEM_THREADS)
            if warp == self.alloc_warp:
                tmem.free(tmem.retrieve_ptr(cutlass.Float32), 256)
        stamp_warp(stamps, 4)

    @cute.jit
    def role(self, ctx, item, tmem_ptr):
        stamp_warp(ctx.stamps, 2)
        touch(ctx, item)
        stamp_warp(ctx.stamps, 3)


@jit_cache
def compile_empty_roles(impl: str):
    op = EmptyRoles() if impl == "warpdrive" else RawEmptyRoles(int(impl.removeprefix("raw_w")))
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(Int64, (cute.sym_int(),), assumed_align=8),
        make_fake_compact_tensor(Int32, (1,), assumed_align=4),
        Int32(1),
        name=op.get_name(),
    )


def make_probe(mode: str, kind: str, stages: int) -> ProbeBase:
    if mode == "stream":
        return Stream(kind, stages)
    if kind == "round_robin":
        return RoundRobinPingPong(kind, stages)
    return PingPong(kind, stages)


@jit_cache
def compile_warpdrive_probe(op: ProbeBase):
    return _compile(op)


@jit_cache
def compile_raw_probe(kind: str, stages: int, mode: str, wait: str):
    return _compile(RawProbe(kind, stages, mode, wait))


def _compile(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(Int64, (NUM_STAMPS,), assumed_align=8),
        make_fake_compact_tensor(BFloat16, (TMA_ELEMS,)),
        Int32(1),
        name=op.get_name(),
    )


def compiled(impl: str, mode: str, kind: str, stages: int):
    if impl == "warpdrive":
        return compile_warpdrive_probe(make_probe(mode, kind, stages))
    return compile_raw_probe(kind, stages, mode, impl)


def run_probe(fn, n: int) -> dict:
    """One launch of ``n`` iterations: cycles and ns of the timed interval."""
    stamps = torch.zeros(NUM_STAMPS, device="cuda", dtype=torch.int64)
    src = torch.zeros(TMA_ELEMS, device="cuda", dtype=torch.bfloat16)
    fn(stamps, src, n)
    torch.cuda.synchronize()
    s = stamps.tolist()
    return {"cycles": s[1] - s[0], "ns": s[3] - s[2]}


def handoff_cost(fn, mode: str, n: int) -> dict:
    """Per one-way handoff (pingpong) or per item (stream), from the difference between ``n`` and
    ``2n`` iterations so loop entry and exit cancel."""
    per = 2 if mode == "pingpong" else 1
    lo, hi = run_probe(fn, n), run_probe(fn, 2 * n)
    return {k: (hi[k] - lo[k]) / (per * n) for k in ("cycles", "ns")}


FIXED_IMPLS = ("warpdrive", "raw_w0", "raw_w1")
FIXED_PHASES = ("entry", "presync", "role start", "role end", "kernel end")


def fixed_stamps(fn, ctas: int) -> list[float]:
    """Median cycles from each CTA's first warp entry to each phase (median over warps)."""
    stamps = torch.zeros(ctas * FIXED_WARPS * FIXED_SLOTS, device="cuda", dtype=torch.int64)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    fn(stamps, out, ctas)
    torch.cuda.synchronize()
    st = stamps.view(ctas, FIXED_WARPS, FIXED_SLOTS).double()
    rel = st - st[:, :, 0].min(dim=1).values[:, None, None]
    return [float(rel[:, :, k].median()) for k in range(FIXED_SLOTS)]


def kernel_times_us(fn, ctas: int, reps: int) -> float:
    """Median CUPTI duration of ``reps`` graph-replayed launches."""
    stamps = torch.zeros(ctas * FIXED_WARPS * FIXED_SLOTS, device="cuda", dtype=torch.int64)
    out = torch.zeros(1, device="cuda", dtype=torch.int32)
    fn(stamps, out, ctas)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn(stamps, out, ctas)
    graph.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    times = [
        ev.device_time
        for ev in prof.events()
        if ev.device_type == torch.autograd.DeviceType.CUDA and "overhead" in ev.name
    ]
    assert len(times) == reps, len(times)
    return statistics.median(times)


def run_fixed(rounds: int, reps: int) -> list[dict]:
    fns = {impl: compile_empty_roles(impl) for impl in FIXED_IMPLS}
    rows = []
    for ctas in (16, 148):
        for r in range(rounds):
            order = FIXED_IMPLS if r % 2 == 0 else FIXED_IMPLS[::-1]
            for impl in order:
                rows.append(
                    {
                        "suite": "fixed",
                        "round": r,
                        "impl": impl,
                        "ctas": ctas,
                        "kernel_us": kernel_times_us(fns[impl], ctas, reps),
                        "phases_cycles": fixed_stamps(fns[impl], ctas),
                    }
                )
    print(f"{'ctas':>4} {'impl':10} {'kernel us':>9} " + "".join(f"{p:>11}" for p in FIXED_PHASES))
    for ctas in (16, 148):
        for impl in FIXED_IMPLS:
            rs = [r for r in rows if r["ctas"] == ctas and r["impl"] == impl]
            us = statistics.median(r["kernel_us"] for r in rs)
            phases = [
                statistics.median(r["phases_cycles"][k] for r in rs) for k in range(FIXED_SLOTS)
            ]
            print(f"{ctas:>4} {impl:10} {us:9.3f} " + "".join(f"{x:11.0f}" for x in phases))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--suite", choices=("handoff", "fixed", "all"), default="all")
    parser.add_argument("--n", type=int, default=4096, help="iterations per launch")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--reps", type=int, default=200, help="graph launches per fixed round")
    parser.add_argument("--output", help="write JSON results here")
    args = parser.parse_args()
    if torch.cuda.get_device_capability()[0] != 10:
        sys.exit("needs an SM100-family GPU")
    rows = []
    if args.suite in ("fixed", "all"):
        rows += run_fixed(args.rounds, args.reps)
    if args.suite in ("handoff", "all"):
        rows += run_handoff(args.n, args.rounds)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"device": torch.cuda.get_device_name(), "rows": rows}, f, indent=1)


def run_handoff(n: int, rounds: int) -> list[dict]:
    rows = []
    cases = [("pingpong", k, 1 if k != "round_robin" else 2) for k in HANDOFF_KINDS]
    cases += [("stream", k, s) for k in STREAM_KINDS for s in (1, 4)]
    impls = ("warpdrive", *WAITS)
    fns = {(i, *c): compiled(i, *c) for c in cases for i in impls}
    for fn in fns.values():  # warm every variant before timing
        run_probe(fn, 64)
    for r in range(rounds):
        order = impls if r % 2 == 0 else impls[::-1]
        for case in cases:
            for impl in order:
                res = handoff_cost(fns[(impl, *case)], case[0], n)
                rows.append(
                    {
                        "suite": "handoff",
                        "round": r,
                        "impl": impl,
                        "mode": case[0],
                        "kind": case[1],
                        "stages": case[2],
                        **res,
                    }
                )
    summary = {}
    for row in rows:
        key = (row["mode"], row["kind"], row["stages"], row["impl"])
        summary.setdefault(key, []).append(row)
    print(f"{'mode':9} {'kind':12} {'n':>2} {'impl':10} {'cycles':>8} {'ns':>7}")
    for (mode, kind, stages, impl), rs in summary.items():
        cyc = statistics.median(r["cycles"] for r in rs)
        ns = statistics.median(r["ns"] for r in rs)
        print(f"{mode:9} {kind:12} {stages:>2} {impl:10} {cyc:8.1f} {ns:7.1f}")
    return rows


if __name__ == "__main__":
    main()
