"""Mix managed and raw synchronization on the same barriers.

The managed ``producer`` (warp 0) writes epoch ``e``'s words ``e * 64 + t`` into ``buf``
and commits each ``ring`` stage. The raw ``consumer`` (warps 1-2) waits through
``self.barrier_ptr("ring")`` with its own ``PipelineState``. All 64 readers meet at ``read``
before thread 0 releases the stage; ``override={"empty_counts": 1}`` declares that one arrival.

The raw ``watcher`` (warp 3) waits for ``done.arrive()`` with native ``bar.sync`` on
``done.id``, then reduces the per-thread results into ``out``. ``describe()`` reports the
external empty count and ``done`` participation as unverified: the native waiter supplies
32 of its 96 threads. See ``docs/warpdrive/dropping-down.md``.
"""

from types import SimpleNamespace

import cutlass
import torch
from cutlass import cute, pipeline

from attn_gym.warpdrive import (
    Local,
    NamedBarrier,
    Producer,
    Ring,
    SmemPlan,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor

WORDS = 64  # one word per consumer thread per stage


class RawInterop(WarpSpecialized):
    def __init__(self, stages: int = 2):
        self.stages = stages
        self.ring = Ring(stages, external="consumer", override={"empty_counts": 1})
        self.read = NamedBarrier(threads=64)  # the consumer's threads
        self.done = NamedBarrier(threads=96)  # the consumer arrives, the watcher syncs
        self.smem_plan = SmemPlan()
        self.smem_plan.buffer("buf", cutlass.Int32)  # (stage, word)
        self.smem_plan.buffer("results", cutlass.Int32)  # (wrong words | sum, thread)

    @role(warps=0)
    @cute.jit
    def producer(self, ctx, *, ring: Producer):
        lane = cute.arch.lane_idx()
        for e in cutlass.range(ctx.n, unroll=1):
            with ring.produce() as h:
                ctx.buf[h.index, lane] = e * WORDS + lane
                ctx.buf[h.index, lane + 32] = e * WORDS + lane + 32
        return ring

    @role(warps=range(1, 3), raw=True)
    @cute.jit
    def consumer(self, ctx, me: Local):
        t = me.tid
        bad, total = cutlass.Int32(0), cutlass.Int32(0)
        full = self.barrier_ptr("ring")  # full barriers at [0, n), empty at [n, 2n)
        empty = full + self.stages
        state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stages)
        for e in cutlass.range(ctx.n, unroll=1):
            cute.arch.mbarrier_wait(full + state.index, state.phase)
            word = ctx.buf[state.index, t]
            bad += word % WORDS != t  # this thread's word; the sum checks the epochs
            total += word
            self.read.sync()  # every thread has read the stage before the one release
            if t == 0:
                cute.arch.mbarrier_arrive(empty + state.index)
            state.advance()
        ctx.results[0, t] = bad
        ctx.results[1, t] = total
        self.done.arrive()  # publishes the results to the watcher

    @role(warps=3, raw=True)
    @cute.jit
    def watcher(self, ctx):
        cute.arch.barrier(barrier_id=self.done.id, number_of_threads=self.done.threads)
        lane = cute.arch.lane_idx()
        if lane < 2:
            acc = cutlass.Int32(0)
            for t in cutlass.range(WORDS, unroll=1):
                acc += ctx.results[lane, t]
            ctx.out[lane] = acc

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        layouts = {
            "buf": cute.make_layout((self.stages, WORDS)),
            "results": cute.make_layout((2, WORDS)),
        }
        self.begin_roles(layouts=layouts)
        ctx = SimpleNamespace(out=out, n=n, buf=self.buffer("buf"), results=self.buffer("results"))
        self.run_roles(ctx)

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream):
        self.kernel.set_name_prefix("warpdrive_raw_interop")
        self.kernel(out, n).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_raw_interop(op: RawInterop):
    out = make_fake_compact_tensor(cutlass.Int32, (2,), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name="warpdrive_raw_interop")


def raw_interop(n: int, stages: int = 2) -> torch.Tensor:
    """``[wrong words, sum of every word]`` after ``n`` epochs through the ring."""
    out = torch.full((2,), -1, device="cuda", dtype=torch.int32)
    compile_raw_interop(RawInterop(stages))(out, n)
    return out
