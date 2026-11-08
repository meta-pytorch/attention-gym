"""Independent fake-tensor TMA copy templates for group/byte-accounting mutations."""

from dataclasses import dataclass
from types import SimpleNamespace

import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    Producer,
    Ring,
    SingleCta,
    Threads,
    Tma,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, make_fake_compact_tensor

TILE_BYTES = 64 * 64 * 2


@dataclass(frozen=True)
class TmaMutation:
    pair: bool = False
    leader: bool = False
    group: str = "ONE"
    tx: int = TILE_BYTES
    copies: int = 1
    opaque: bool = False


class TmaProbe(WarpSpecialized):
    def __init__(self, config: TmaMutation):
        self.config = config
        self.cluster = CtaPair() if config.pair else SingleCta()
        self.ch = Ring(2, Tma(config.tx), Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        for i in cutlass.range(ctx.n, unroll=1):
            h = ch.acquire()
            for j in cutlass.range_constexpr(self.config.copies):
                if cutlass.const_expr(self.config.opaque):
                    bar = cute.make_ptr(
                        cutlass.Int64, h.barrier.toint(), cute.AddressSpace.smem, assumed_align=8
                    )
                else:
                    bar = h.barrier
                cute.copy(ctx.tma, ctx.tg[None, 0], ctx.ts[None, h.index], tma_bar_ptr=bar)
            h.commit()
        return ch

    @cute.jit
    def use(self, ch, ctx):
        for i in cutlass.range(ctx.n, unroll=1):
            h = ch.wait()
            h.release()
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        ch = self.use(ch, ctx)
        return ch

    @cute.kernel
    def kernel(self, tma, tg, n: cutlass.Int32, layout):
        smem = self.begin_roles()
        s = smem.allocate_tensor(cutlass.BFloat16, layout, 128)
        ts, tgp = cpasync.tma_partition(
            tma, 0, cute.make_layout(1), cute.group_modes(s, 0, 2), cute.group_modes(tg, 0, 2)
        )
        self.run_roles(SimpleNamespace(tma=tma, tg=tgp, ts=ts, n=n))

    @cute.jit
    def __call__(self, src: cute.Tensor, n: cutlass.Int32, stream):
        group = tcgen05.CtaGroup.TWO if self.config.group == "TWO" else tcgen05.CtaGroup.ONE
        layout = cute.make_ordered_layout((64, 64, 2), order=(1, 0, 2))
        tma, tg = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(group), src, cute.select(layout, mode=[0, 1]), (64, 64)
        )
        tiles = cute.local_tile(tg, (64, 64), (None, 0))
        self.kernel.set_name_prefix("warpdrive_tma_mutation")
        self.kernel(tma, tiles, n, layout).launch(
            **self.launch_config(grid=(self.cluster.size, 1, 1)), stream=stream
        )


class TmaLeaderProbe(TmaProbe):
    @role(warps=1, cta="leader")
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        ch = self.use(ch, ctx)
        return ch


def compile_tma_mutation(config: TmaMutation):
    op = TmaLeaderProbe(config) if config.leader else TmaProbe(config)
    src = make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(divisibility=64), 64))
    compile_tvm_ffi(op, src, cutlass.Int32(1), name="warpdrive_tma_mutation")
    return op
