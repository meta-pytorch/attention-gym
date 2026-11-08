"""``Warps`` channel sides: one elected arrival per warp, after the warp's lanes synchronize."""

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
    CtaPair,
    Producer,
    Ring,
    Slots,
    Threads,
    Umma,
    Warps,
    WarpSpecialized,
    role,
)

N = 128  # payload words: one per producer thread


@cute.jit
def spin(cycles):
    start = cute.arch.clock64()
    while cute.arch.clock64() - start < cycles:
        pass


class WarpsToy(WarpSpecialized):
    """Four producer warps fill a two-stage payload ring, one word per thread; two consumer
    warps check every word of each stage. Warp 3 and the higher lanes of each warp write late,
    and consumer warp 5 reads late, so a stage published before every lane of every producer
    warp has written, or released before every consumer lane has read, shows up as stale words
    or as an overwrite. (A miscounted barrier hangs rather than miscompares, and the compiler
    reconverges a warp before the arrive even without ``bar.warp.sync``, which the memory model
    still requires; neither makes a runtime negative control.)"""

    def __init__(self):
        self.ch = Ring(2, Warps, Warps, payload=(cutlass.Int32, (N,)))

    @role(warps=range(4))
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        tid = cute.arch.thread_idx()[0] % N
        late = (tid // 32 == 3) * ctx.delay + cute.arch.lane_idx() * 64
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce() as h:
                spin(late)
                h.data[tid] = i * N + tid
        return ch

    @role(warps=range(4, 6))
    @cute.jit
    def cons(self, ctx, *, ch: Consumer):
        tid = cute.arch.thread_idx()[0] % 64
        late = (tid // 32 == 1) * ctx.delay
        bad = cutlass.Int32(0)
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.consume() as h:
                spin(late)
                for e in cutlass.range_constexpr(N // 64):
                    word = tid + 64 * e
                    bad = bad + (h.data[word] != i * N + word)
        ctx.out[tid] = bad
        return ch

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32, delay: cutlass.Int64):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n, delay=delay))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, delay: cutlass.Int64, stream):
        self.kernel.set_name_prefix("warpdrive_warps_toy")
        self.kernel(out, n, delay).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (64,), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), cutlass.Int64(0), name="warpdrive_warps_toy")


def test_warps_counts_are_per_warp():
    op = WarpsToy()
    op.describe()
    sc = op.ch.scope
    assert sc.full_counts == {0: [4, 4]} and sc.empty_counts == {0: [2, 2]}
    assert "signal    Warps -> Warps" in op.describe()


@pytest.mark.parametrize("n", [1, 5, 64])
def test_warps_publish_and_release_whole_warps(requires_blackwell, n):
    out = torch.full((64,), -1, device="cuda", dtype=torch.int32)
    compile_toy(WarpsToy())(out, n, 20_000)
    torch.cuda.synchronize()
    assert out.tolist() == [0] * 64


class PairWarpsToy(WarpSpecialized):
    """Two-warp producers in both CTAs of a pair join on the leader's full barrier; the leader
    commits back (multicast) and two-warp sinks in both CTAs release it."""

    def __init__(self, mech=Warps):
        self.cluster = CtaPair()
        self.mech = mech
        self.ch = Slots(1, mech, Umma)
        self.back = Ring(2, Umma, mech)

    @role(warps=range(2))
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.produce(0):
                pass
        return ch

    @role(warps=2, cta="leader")
    @cute.jit
    def mma(self, ctx, *, ch: Consumer, back: Producer):
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(0), back.produce():
                pass
        return ch, back

    @role(warps=range(4, 6))
    @cute.jit
    def sink(self, ctx, *, back: Consumer):
        count = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with back.consume():
                count = count + 1
        rank = self.cluster.rank()
        if cute.arch.thread_idx()[0] % 64 == 0:
            ctx.out[rank] = count
        return back

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_pair_warps_toy")
        self.kernel(out, n).launch(**self.launch_config(grid=(2, 1, 1)), stream=stream)


@jit_cache
def compile_pair_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (2,), assumed_align=4)
    return compile_tvm_ffi(op, out, cutlass.Int32(1), name="warpdrive_pair_warps_toy")


@pytest.mark.parametrize("mech, per_warp", [(Warps, 1), (Threads, 32)], ids=["warps", "threads"])
def test_pair_counts_scale_with_the_signal(mech, per_warp):
    op = PairWarpsToy(mech)
    op.describe()
    # Both CTAs' producer warps arrive on the leader; both CTAs' sinks release the leader.
    assert op.ch.scope.full_counts[0] == [2 * 2 * per_warp]
    assert op.back.scope.empty_counts[0] == [2 * 2 * per_warp] * 2


@pytest.mark.parametrize("n", [1, 7])
def test_pair_warps_round_trip(requires_blackwell, n):
    out = torch.zeros(2, device="cuda", dtype=torch.int32)
    compile_pair_toy(PairWarpsToy())(out, n)
    torch.cuda.synchronize()
    assert out.tolist() == [n, n]


def test_named_channels_reject_warps():
    with pytest.raises(ValueError, match="both sides must be Threads"):
        Ring(1, Warps, Threads, sync="named")
