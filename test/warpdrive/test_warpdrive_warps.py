"""``Warps`` channel sides: one elected arrival per warp, after the warp's lanes synchronize."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

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
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor

N = 128  # payload words: one per producer thread


@cute.jit
def spin(cycles):
    start = cute.arch.clock64()
    while cute.arch.clock64() - start < cycles:
        pass


class WarpsToy(WarpSpecialized):
    """Four producer warps fill a two-stage ring; two consumer warps check every word.

    Warp 3 and each warp's higher lanes write late; consumer warp 5 reads late. Publishing
    before every producer lane writes, or releasing before every consumer lane reads,
    causes stale words or overwrites.

    A wrong barrier count hangs rather than miscompares. The compiler reconverges before
    arrival even without bar.warp.sync, but the memory model still requires that sync.
    Neither omission is a runtime negative control.
    """

    def __init__(self, wait: str = "suspend"):
        self.ch = Ring(2, Warps, Warps, payload=(cutlass.Int32, (N,)), wait=wait)

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


@pytest.mark.parametrize("wait", ["suspend", "spin"])
def test_warps_publish_and_release_whole_warps(requires_blackwell, wait):
    fn = compile_toy(WarpsToy(wait))
    for n in (1, 64):
        out = torch.full((64,), -1, device="cuda", dtype=torch.int32)
        fn(out, n, 20_000)
        torch.cuda.synchronize()
        assert out.tolist() == [0] * 64, n


def test_channel_wait_mode_is_declared_and_described():
    assert "wait spin" in WarpsToy("spin").describe()
    assert "wait spin" not in WarpsToy().describe()
    with pytest.raises(ValueError, match="wait must be one of"):
        Ring(2, Threads, Threads, wait="poll")
    with pytest.raises(ValueError, match="named channels have none"):
        Ring(2, Threads, Threads, sync="named", wait="spin")


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
    text = op.describe()
    # Both CTAs' producer warps arrive on the leader; both CTAs' sinks release the leader.
    assert op.ch.scope.full_counts[0] == [2 * 2 * per_warp]
    assert op.back.scope.empty_counts[0] == [2 * 2 * per_warp] * 2
    assert f"signal    {mech.__name__} -> Umma" in text, text


def test_pair_warps_round_trip(requires_blackwell):
    out = torch.zeros(2, device="cuda", dtype=torch.int32)
    compile_pair_toy(PairWarpsToy())(out, 7)
    torch.cuda.synchronize()
    assert out.tolist() == [7, 7]


def test_named_channels_reject_warps():
    with pytest.raises(ValueError, match="both sides must be Threads"):
        Ring(1, Warps, Threads, sync="named")
