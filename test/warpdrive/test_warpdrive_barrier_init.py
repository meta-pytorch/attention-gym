"""Deep channels initialize every full/empty barrier with its declared arrival count."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Consumer,
    Producer,
    Ring,
    Threads,
    WarpSpecialized,
    role,
)


class DeepRing(WarpSpecialized):
    def __init__(self, stages: int):
        self.ch = Ring(stages, Threads, Threads, payload=(cutlass.Int32, (32,)))

    @role(warps=4)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce() as h:
                h.data[lane] = i * 32 + lane
        return ch

    @role(warps=range(4))
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        lane = cute.arch.lane_idx()
        acc = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume() as h:
                acc = acc + h.data[lane]
        ctx.out[cute.arch.thread_idx()[0]] = acc
        return ch

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_deep_ring")
        self.kernel(out, n).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_deep_ring(stages: int):
    return compile_tvm_ffi(
        DeepRing(stages),
        make_fake_compact_tensor(cutlass.Int32, (128,), assumed_align=4),
        cutlass.Int32(1),
        name="warpdrive_deep_ring",
    )


@pytest.mark.parametrize("stages", [1, 33])
def test_deep_ring_arrival_counts_and_wraparound(stages):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("warpdrive requires a Blackwell GPU")
    # The deep case crosses a warp's lane count in both the 32-arrival full barriers and
    # the 128-arrival empty barriers. Repeated launches and wraps exercise every barrier.
    n = 2 * stages + 3
    fn = compile_deep_ring(stages)
    out = torch.empty(128, device="cuda", dtype=torch.int32)
    lane = torch.arange(128, dtype=torch.int32) % 32
    expected = n * (n - 1) // 2 * 32 + n * lane
    for _ in range(3):
        out.fill_(-1)
        fn(out, n)
        torch.testing.assert_close(out.cpu(), expected)
