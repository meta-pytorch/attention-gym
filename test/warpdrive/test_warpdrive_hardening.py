"""Hardening of the warp-role layer: the automatic TMA atom/byte check, escape hatches (raw state,
raw signals, raw roles, user-driven barriers, overrides), per-site check opt-outs, and a GEMM that
mixes managed and raw synchronization."""

import re
import warnings
from dataclasses import dataclass, fields
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute, pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Barriers,
    Consumer,
    CtaPair,
    Producer,
    ProtocolError,
    Ring,
    SingleCta,
    Slots,
    StaticPersistent,
    Threads,
    Tma,
    Tmem,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
    unchecked,
)
from attn_gym.warpdrive.examples.gemm import Gemm, GemmParams

TILE_ROWS = 64  # one 64x64 bf16 TMA tile per stage
TILE_BYTES = TILE_ROWS * 64 * 2


def _raises_protocol(match: str, fn):
    with pytest.raises(Exception, match=match) as info:
        fn()
    assert isinstance(info.value, ProtocolError) or "ProtocolError" in repr(info.value)
    return info.value


# --------------------------------------------------------------------------- #
# TMA toy: a load role copies 64x64 tiles into a 2-stage ring; the consumer records the first
# element of each tile (tile i holds the value i + 1). ``consumer_cta`` places the consumer: in a
# pair, "leader" makes the channel pair-scoped (both CTAs' bytes complete on the leader), "all"
# keeps it CTA-local. ``group`` picks the atom's CtaGroup; ``tx`` the declared bytes.
# --------------------------------------------------------------------------- #


class _TmaToy(WarpSpecialized):
    def __init__(self, cluster, group="ONE", tx=TILE_BYTES, mode="plain"):
        self.cluster = cluster
        self.group, self.mode = group, mode
        self.ld = Ring(2, Tma(tx), Threads)
        self.key = f"{cluster.get_key()}_{self.consumer_cta}_{group.lower()}_{tx}_{mode}"

    @role(warps=0)
    @cute.jit
    def load(self, ld: Producer, ctx):
        for i in cutlass.range(ctx.n, unroll=1):
            h = ld.acquire()
            if cutlass.const_expr(self.mode == "opaque"):
                # The barrier reaches the copy through a pointer the layer did not hand out.
                bar = cute.make_ptr(
                    cutlass.Int64, h.barrier.toint(), cute.AddressSpace.smem, assumed_align=8
                )
                cute.copy(
                    ctx.tma, ctx.tg[None, i % ctx.tiles], ctx.ts[None, h.index], tma_bar_ptr=bar
                )
            elif cutlass.const_expr(self.mode in ("half_opaque", "mixed_end")):
                # Two copies of the tile per stage (Tma(2 x tile)); only one through h.barrier.
                src, dst = ctx.tg[None, i % ctx.tiles], ctx.ts[None, h.index]
                cute.copy(ctx.tma, src, dst, tma_bar_ptr=h.barrier)
                if cutlass.const_expr(self.mode == "half_opaque"):
                    other = cute.make_ptr(
                        cutlass.Int64, h.barrier.toint(), cute.AddressSpace.smem, assumed_align=8
                    )
                else:
                    other = ld.full_barrier(h.index)
                cute.copy(ctx.tma, src, dst, tma_bar_ptr=other)
            elif cutlass.const_expr(self.mode == "nested"):
                if ctx.n > 0:  # a dynamic branch: the copy's execution count is not static
                    cute.copy(
                        ctx.tma,
                        ctx.tg[None, i % ctx.tiles],
                        ctx.ts[None, h.index],
                        tma_bar_ptr=h.barrier,
                    )
            elif cutlass.const_expr(self.mode == "rawptr"):
                bar = self.barrier_ptr("ld") + h.index  # the full barrier, not handed out
                cute.copy(
                    ctx.tma, ctx.tg[None, i % ctx.tiles], ctx.ts[None, h.index], tma_bar_ptr=bar
                )
            elif cutlass.const_expr(self.mode == "bulk"):
                # A cp.async.bulk (non-tensor) copy of the whole tile, like FA4 bwd's LSE rows;
                # bulk copies do not elect a lane themselves.
                with cute.arch.elect_one():
                    cute.copy(ctx.bulk, *self.flat(ctx, i, h), mbar_ptr=h.barrier)
            elif cutlass.const_expr(self.mode == "bulk_lane0"):
                if cute.arch.lane_idx() == 0:  # a generic branch, not elect_one(): not counted
                    cute.copy(ctx.bulk, *self.flat(ctx, i, h), mbar_ptr=h.barrier)
            elif cutlass.const_expr(self.mode == "bulk_direct"):
                # Every lane of the warp issues the copy: 32 x the bytes (never launched).
                cute.copy(ctx.bulk, *self.flat(ctx, i, h), mbar_ptr=h.barrier)
            elif cutlass.const_expr(self.mode == "optout"):
                with unchecked("tma", reason="toy: atom verified by hand"):
                    bar = h.barrier
                cute.copy(
                    ctx.tma, ctx.tg[None, i % ctx.tiles], ctx.ts[None, h.index], tma_bar_ptr=bar
                )
            else:
                cute.copy(
                    ctx.tma,
                    ctx.tg[None, i % ctx.tiles],
                    ctx.ts[None, h.index],
                    tma_bar_ptr=h.barrier,
                )
            h.commit()
        return ld

    @cute.jit
    def flat(self, ctx, i, h):
        """The tile and its stage as flat vectors (a bulk copy moves contiguous bytes)."""
        flat = cute.make_layout(TILE_ROWS * 64)
        src = cute.make_tensor(ctx.g[None, None, i % ctx.tiles].iterator, flat)
        dst = cute.make_tensor(ctx.s[None, None, h.index].iterator, flat)
        return src, dst

    @cute.jit
    def consume(self, ld, ctx):
        rank = self.cluster.rank()  # warp-uniform: not inside the lane-0 branch
        for i in cutlass.range(ctx.n, unroll=1):
            with ld.consume() as h:
                if cute.arch.lane_idx() == 0:
                    first = ctx.s[0, 0, h.index].to(cutlass.Float32)
                    last = ctx.s[TILE_ROWS - 1, 63, h.index].to(cutlass.Float32)
                    ctx.out[rank * ctx.n + i] = first + 1000.0 * (last - first)
        return ld

    @cute.kernel
    def kernel(self, tma, tg, g, out, n: cutlass.Int32, tiles: cutlass.Int32, s_layout):
        smem = self.begin_roles()
        s = smem.allocate_tensor(cutlass.BFloat16, s_layout, 128)
        ts, tgp = cpasync.tma_partition(
            tma, 0, cute.make_layout(1), cute.group_modes(s, 0, 2), cute.group_modes(tg, 0, 2)
        )
        bulk = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), cutlass.BFloat16)
        ctx = SimpleNamespace(
            tma=tma, tg=tgp, ts=ts, s=s, g=g, bulk=bulk, out=out, n=n, tiles=tiles
        )
        self.run_roles(ctx)

    @cute.jit
    def __call__(self, src: cute.Tensor, out: cute.Tensor, n: cutlass.Int32, stream):
        group = tcgen05.CtaGroup.TWO if self.group == "TWO" else tcgen05.CtaGroup.ONE
        s_layout = cute.make_ordered_layout((TILE_ROWS, 64, 2), order=(1, 0, 2))
        tma, tg = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(group),
            src,
            cute.select(s_layout, mode=[0, 1]),
            (TILE_ROWS, 64),
        )
        tiles = cute.local_tile(tg, (TILE_ROWS, 64), (None, 0))
        g = cute.local_tile(src, (TILE_ROWS, 64), (None, 0))  # plain (rows, cols, tile) view
        self.kernel(tma, tiles, g, out, n, src.shape[0] // TILE_ROWS, s_layout).launch(
            **self.launch_config(grid=(self.cluster.size, 1, 1)), stream=stream
        )


class _TmaToyAll(_TmaToy):
    consumer_cta = "all"

    @role(warps=1)
    @cute.jit
    def use(self, ld: Consumer, ctx):
        ld = self.consume(ld, ctx)
        return ld


class _TmaToyLeader(_TmaToy):
    consumer_cta = "leader"

    @role(warps=1, cta="leader")
    @cute.jit
    def use(self, ld: Consumer, ctx):
        ld = self.consume(ld, ctx)
        return ld


def make_tma_toy(cluster, consumer_cta="all", group="ONE", tx=TILE_BYTES, mode="plain"):
    cls = _TmaToyLeader if consumer_cta == "leader" else _TmaToyAll
    return cls(cluster, group, tx, mode)


@jit_cache
def compile_tma_toy(op):
    src = make_fake_compact_tensor(cutlass.BFloat16, (cute.sym_int(divisibility=64), 64))
    out = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, src, out, cutlass.Int32(1), name=f"warpdrive_tma_toy_{op.key}")


def run_tma_toy(op, n=7, tiles=3):
    rows = torch.arange(tiles * TILE_ROWS, device="cuda") // TILE_ROWS + 1
    src = rows.view(-1, 1).expand(-1, 64).to(torch.bfloat16).contiguous()
    src[torch.arange(tiles, device="cuda") * TILE_ROWS + TILE_ROWS - 1, 63] += 0.5  # last elem
    out = torch.zeros(op.cluster.size * n, device="cuda", dtype=torch.float32)
    compile_tma_toy(op)(src, out, n)
    torch.cuda.synchronize()
    expect = [i % tiles + 1 + 500.0 for i in range(n)]
    readers = 1 if op.consumer_cta == "leader" else op.cluster.size
    assert out.view(op.cluster.size, n)[:readers].tolist() == [expect] * readers


@pytest.mark.parametrize(
    "cluster, consumer_cta, group",
    [(SingleCta(), "all", "ONE"), (CtaPair(), "leader", "TWO"), (CtaPair(), "all", "ONE")],
    ids=["single", "pair_scoped", "pair_local"],
)
def test_tma_check_accepts_matching_atoms(requires_blackwell, cluster, consumer_cta, group):
    op = make_tma_toy(cluster, consumer_cta, group)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a verified kernel emits no "unverified" warning
        run_tma_toy(op)
    report = op._ws_tma["ld"]
    assert report.groups == {group} and report.stage_bytes == {TILE_BYTES}
    assert not report.unverified
    text = op.describe()
    assert f"tma       verified CtaGroup.{group}, {TILE_BYTES} B/stage per CTA @ " in text


@pytest.mark.parametrize(
    "cluster, consumer_cta, group, message",
    [
        (CtaPair(), "leader", "ONE", r"needs CtaGroup\.TWO: ld is pair-scoped.*leader hangs"),
        (CtaPair(), "all", "TWO", r"needs CtaGroup\.ONE: ld is CTA-local"),
    ],
    ids=["one_atom_on_pair_channel", "two_atom_on_local_channel"],
)
def test_tma_check_rejects_mismatched_atom(cluster, consumer_cta, group, message):
    """Compile-time errors for kernels that would hang on the GPU (they are never launched)."""
    err = _raises_protocol(
        message, lambda: compile_tma_toy(make_tma_toy(cluster, consumer_cta, group))
    )
    assert "test_warpdrive_hardening.py:" in str(err)  # points at the user's copy line


@pytest.mark.parametrize("mode", ["plain", "bulk"])
@pytest.mark.parametrize("tx", [TILE_BYTES // 2, 2 * TILE_BYTES])
def test_tma_check_rejects_wrong_tx_bytes(tx, mode):
    effect = "completes before every copy lands" if tx < TILE_BYTES else "never completes"
    _raises_protocol(
        rf"deliver {TILE_BYTES} bytes, but the layer arms {tx} .*{effect}.*Tma\({TILE_BYTES}\)",
        lambda: compile_tma_toy(make_tma_toy(SingleCta(), tx=tx, mode=mode)),
    )


def test_tma_check_verifies_bulk_copies(requires_blackwell):
    """A ``cp.async.bulk`` load (``CopyBulkG2SOp``, ``mbar_ptr=``) inside ``elect_one()`` is
    counted like a tensor TMA load: its bytes complete on the issuing CTA's barrier."""
    op = make_tma_toy(SingleCta(), mode="bulk")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run_tma_toy(op)
    report = op._ws_tma["ld"]
    assert report.groups == {"bulk"} and report.stage_bytes == {TILE_BYTES}
    assert not report.unverified
    assert f"tma       verified cp.async.bulk, {TILE_BYTES} B/stage per CTA @ " in op.describe()


def test_tma_check_rejects_bulk_copy_on_pair_scoped_channel():
    """The leader arms both CTAs' bytes, but a bulk copy from the peer completes on the peer's
    own barrier: rejected like a CtaGroup.ONE tensor load."""
    _raises_protocol(
        r"a cp\.async\.bulk copy, which completes on the issuing CTA's barrier, but ld needs "
        r"CtaGroup\.TWO: ld is pair-scoped.*use a tensor TMA atom",
        lambda: compile_tma_toy(make_tma_toy(CtaPair(), "leader", "TWO", mode="bulk")),
    )


def test_tma_check_rejects_wrong_tx_bytes_in_pair():
    """Pair-scoped: the leader arms 2 x Tma(tx); each CTA must copy tx."""
    _raises_protocol(
        rf"deliver {TILE_BYTES} bytes per CTA, but the layer arms {2 * TILE_BYTES} per CTA",
        lambda: compile_tma_toy(make_tma_toy(CtaPair(), "leader", "TWO", tx=2 * TILE_BYTES)),
    )


def test_tma_check_fails_open_on_opaque_barrier(requires_blackwell):
    """A barrier routed through a pointer the layer did not hand out cannot be analyzed: the
    kernel compiles and runs, describe() reports it unverified, and one warning is emitted."""
    op = make_tma_toy(SingleCta(), mode="opaque")
    with pytest.warns(UserWarning, match=r"TMA check could not verify channel\(s\) \['ld'\]"):
        run_tma_toy(op)
    assert "tma       unverified (barrier taken at test_warpdrive_hardening.py:" in (op.describe())


@pytest.mark.parametrize(
    "mode, tx, reason",
    [
        ("half_opaque", 2 * TILE_BYTES, r"barrier taken at .* reaches no recognized TMA copy"),
        ("mixed_end", 2 * TILE_BYTES, r"bytes of ld\[i\] acquire at .*end\.full_barrier\(\)"),
        ("nested", TILE_BYTES, r"bytes of ld\[i\] acquire at .*nested loop/branch"),
        ("bulk_lane0", TILE_BYTES, r"bytes of ld\[i\] acquire at .*nested loop/branch"),
        ("bulk_direct", TILE_BYTES, r"bulk copy at .* sits outside elect_one\(\), so every lane"),
        ("rawptr", TILE_BYTES, r"no TMA copy completes on a barrier taken from the layer"),
    ],
)
def test_tma_check_fails_open_on_partial_accounting(mode, tx, reason):
    """Incomplete byte accounting marks the channel unverified instead of raising (the kernels
    are correct; the check just cannot see every copy)."""
    op = make_tma_toy(SingleCta(), tx=tx, mode=mode)
    with pytest.warns(UserWarning, match=r"could not verify channel\(s\) \['ld'\]"):
        compile_tma_toy(op)
    assert any(re.search(reason, why) for why in op._ws_tma["ld"].unverified)
    assert "tma       unverified (" in op.describe()


def test_tma_rawptr_kernel_runs(requires_blackwell):
    with pytest.warns(UserWarning, match="could not verify"):
        run_tma_toy(make_tma_toy(SingleCta(), mode="rawptr"))


def test_tma_opt_out_is_listed(requires_blackwell):
    op = make_tma_toy(SingleCta(), mode="optout")
    run_tma_toy(op)
    text = op.describe()
    assert "Unchecked opt-outs\n  unchecked tma @ test_warpdrive_hardening.py:" in text
    assert "load: 'toy: atom verified by hand' -> ld barrier at" in text
    assert "tma       opted out (unchecked)" in text


# --------------------------------------------------------------------------- #
# Raw state and raw signals on managed ends
# --------------------------------------------------------------------------- #


class RawStateToy(WarpSpecialized):
    """The producer signals through the raw full barrier (``commit(signal=False)``); the
    consumer logs the end's next index/phase and each handle's index/phase."""

    def __init__(self):
        self.ch = Ring(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        for _ in cutlass.range(ctx.n, unroll=1):
            h = ch.acquire()
            cute.arch.mbarrier_arrive(h.full_barrier)  # count: 32 threads, as derived
            h.commit(signal=False)
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        for i in cutlass.range(ctx.n, unroll=1):
            nxt_index, nxt_phase = ch.index, ch.phase()
            h = ch.wait()
            if cute.arch.lane_idx() == 0:
                ctx.out[4 * i + 0] = nxt_index
                ctx.out[4 * i + 1] = nxt_phase
                ctx.out[4 * i + 2] = h.index
                ctx.out[4 * i + 3] = h.phase
            h.release()
        return ch

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(out, n).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def compile_raw_state_toy(op: RawStateToy):
    fake = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, fake, cutlass.Int32(1), name="warpdrive_raw_state")


def test_raw_state_and_raw_signal(requires_blackwell):
    n = 5
    out = torch.full((4 * n,), -1, device="cuda", dtype=torch.int32)
    compile_raw_state_toy(RawStateToy())(out, n)
    torch.cuda.synchronize()
    rows = out.view(n, 4).tolist()
    assert rows == [[i % 2, i // 2 % 2, i % 2, i // 2 % 2] for i in range(n)]


# --------------------------------------------------------------------------- #
# Per-site opt-outs
# --------------------------------------------------------------------------- #


class OptOutToy(WarpSpecialized):
    """Compiled only: ``mode`` leaves a handle open or closes it in another region."""

    def __init__(self, mode: str, opt_out: bool):
        self.mode, self.opt_out = mode, opt_out
        self.ch = Slots(1, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        check = "unclosed" if self.mode == "leak" else "cross_region"
        if cutlass.const_expr(self.mode == "leak"):
            if cutlass.const_expr(self.opt_out):
                with unchecked(check, reason="toy: left open on purpose"):
                    ch.acquire(0)
            else:
                ch.acquire(0)
        else:
            h = ch.acquire(0)
            for _ in cutlass.range(ctx.n, unroll=1):
                if cutlass.const_expr(self.opt_out):
                    with unchecked(check, reason="toy: closed in the loop on purpose"):
                        h.commit()
                else:
                    h.commit()
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        return ch

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel(out, n).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


@jit_cache
def _compile_opt_out_toy(op):
    fake = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    name = f"warpdrive_opt_out_{op.mode}_{int(op.opt_out)}"
    return compile_tvm_ffi(op, fake, cutlass.Int32(1), name=name)


@pytest.mark.parametrize(
    "mode, error, listed",
    [
        ("leak", r"handles never closed: ch\[0\] acquire", r"left open"),
        ("cross", r"different dynamic region", r"ch\[0\] acquire commit at"),
    ],
)
def test_opt_out_scopes_one_site(mode, error, listed):
    _raises_protocol(error, lambda: _compile_opt_out_toy(OptOutToy(mode, False)))
    op = OptOutToy(mode, True)
    _compile_opt_out_toy(op)
    text = op.describe()
    assert "unchecked " in text and "test_warpdrive_hardening.py:" in text
    assert re.search(listed, text), text


def test_opt_out_needs_a_reason():
    with pytest.raises(ValueError, match="non-empty reason"), unchecked("unclosed", reason=" "):
        pass
    with pytest.raises(ValueError, match="unknown check"), unchecked("arrivals", reason="x"):
        pass


# --------------------------------------------------------------------------- #
# Host-side declarations: raw roles, Barriers, overrides
# --------------------------------------------------------------------------- #


class _RawWithEnd(WarpSpecialized):
    def __init__(self):
        self.ch = Slots(1, Threads, Threads)

    @role(warps=0, raw=True)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        pass


class _RawInPersistent(WarpSpecialized):
    def __init__(self):
        self.work = WorkStream(schedule=StaticPersistent())
        self.bars = Barriers(2, counts=[32, 1])

    @role(warps=0)
    @cute.jit
    def work(self, work: WorkCursor, ctx):
        return work

    @role(warps=range(4, 8), regs=200, raw=True)
    @cute.jit
    def side(self, bars: Barriers, ctx):
        pass


def test_raw_role_declarations():
    with pytest.raises(ProtocolError, match="raw role prod: parameter ch is Producer"):
        _RawWithEnd()._collect()
    op = _RawInPersistent()
    text = op.describe()  # raw roles need not loop over the stream
    side = next(r for r in op._ws_roles if r.name == "side")
    assert side.raw and side.warps == range(4, 8) and side.regs == 200
    assert op.num_warps == 8  # padded to whole warpgroups
    assert "  4-7    side        200  all  raw\n" in text
    assert "  bars  2 mbarriers  layer-initialized [32, 1]  user-driven (unchecked)" in text
    with pytest.raises(ValueError, match="Barriers counts must be 2 positive ints"):
        Barriers(2, counts=[1])


class _Overrides(WarpSpecialized):
    def __init__(self, **override):
        self.ab = Ring(2, Tma(4096), Umma, override=override or None)
        self.acc = Ring(1, Umma, Threads, remote_scope="cluster")

    @role(warps=0)
    @cute.jit
    def load(self, ab: Producer, ctx):
        return ab

    @role(warps=1)
    @cute.jit
    def mma(self, ab: Consumer, acc: Producer, ctx):
        return ab, acc

    @role(warps=range(4, 8))
    @cute.jit
    def epi(self, acc: Consumer, ctx):
        return acc


def test_overrides_are_shown_against_derived_values():
    op = _Overrides(tx=4096, empty_counts=[1, 4])
    text = op.describe()
    # Effective counts, then each override as derived -> effective.
    assert "empty     [1, 4]\n" in text
    assert "override  tx 4096 -> 4096\n" in text
    assert "override  empty_counts 1 x2 -> [1, 4]  (differs from derived)\n" in text
    assert "override  remote_scope cta -> cluster  (differs from derived)" in text
    assert op.ab.scope.empty_counts == {0: [1, 4]} and op.ab.derived_scope.empty_counts == {
        0: [1, 1]
    }
    assert op.ab.scope.tx == 4096
    with pytest.raises(ValueError, match="unknown override keys"):
        Ring(2, Threads, Threads, override={"arrivals": 1})
    with pytest.raises(ProtocolError, match="needs 2 positive counts"):
        _Overrides(full_counts=[1, 2, 3]).describe()


class _PairMasks(WarpSpecialized):
    def __init__(self, **override):
        self.cluster = CtaPair()
        self.back = Ring(2, Umma, Threads, override=override or None)

    @role(warps=0, cta="leader")
    @cute.jit
    def mma(self, back: Producer, ctx):
        return back

    @role(warps=1)
    @cute.jit
    def sink(self, back: Consumer, ctx):
        return back


def test_pair_mask_override_wins():
    op = _PairMasks(full_mask=0b01)
    text = op.describe()
    assert op.back.derived_scope.full_mask == 0b11 and op.back.scope.full_mask == 0b01
    assert op.back.scope.empty_mask == 0b01  # derived: the leader-only producer's CTA
    assert "scope     pair-scoped, commit_mask 0b01, remote_scope cta\n" in text
    assert "override  full_mask 0b11 -> 0b01  (differs from derived)" in text


# --------------------------------------------------------------------------- #
# Mixed managed/raw GEMM: the ab edge stays managed but the MMA releases it with a raw
# tcgen05.commit; the acc edge is user-driven Barriers behind cutlass.pipeline.PipelineUmmaAsync;
# the epilogue is a raw role.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MixedGemmParams(GemmParams):
    """Gemm's parameters plus the user-built accumulator pipeline; Gemm's load role still takes
    ``params: GemmParams``, a base of this type."""

    acc_pipe: pipeline.PipelineUmmaAsync


class MixedGemm(Gemm):
    def __init__(self):
        super().__init__(ab_stages=3, acc_stages=1)
        del self.acc  # replaced by a user-driven edge
        self.ab = Ring(
            3,
            Tma(sum(self.tile[:2]) * self.tile[2] * 2),
            Umma,
            override={"empty_counts": 1},
            buffer=self.ab.buffer,
        )
        self.acc_bars = Barriers(2)  # full + empty of a 1-stage PipelineUmmaAsync

    def get_name(self) -> str:
        return "warpdrive_mixed_gemm"

    @role(warps=5, regs=40)
    @cute.jit
    def mma(self, ab: Consumer, tmem: Tmem, ctx: MixedGemmParams):
        tiled_mma = ctx.tiled_mma
        tCtAcc = cute.make_tensor(tmem.ptr("acc"), ctx.acc_layout)
        acc_producer, _ = ctx.acc_pipe.make_participants()
        a = acc_producer.acquire_and_advance()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        ab = ab.views(A=ctx.tCrA, B=ctx.tCrB)
        for k in cutlass.range(ctx.num_k_tiles, unroll=1):
            s = ab.wait()
            for kb in cutlass.range_constexpr(cute.size(ctx.tCrA.tensor, mode=[2])):
                cute.gemm(
                    tiled_mma,
                    tCtAcc[None, None, None, a.index],
                    s.view("A", None, None, kb),
                    s.view("B", None, None, kb),
                    tCtAcc[None, None, None, a.index],
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            with cute.arch.elect_one():  # the layer's Umma release, spelled out
                tcgen05.commit(s.empty_barrier)
            s.release(signal=False)
        a.commit()
        return ab

    @role(warps=range(4), regs=160, raw=True)
    @cute.jit
    def epilogue(self, tmem: Tmem, ctx: MixedGemmParams):
        tidx, _, _ = cute.arch.thread_idx()
        t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32),
            cute.zipped_divide(ctx.acc_fake, ctx.epi_tiler)[None, 0],
        ).get_slice(tidx)
        # A raw role takes no work cursor; single_item() reads the OneShot item the way the cursor
        # does (blockIdx on the tile grid, decoded on the 1-D fallback). The wrapper never
        # launches an empty space, so the item is valid.
        _, (m, n, _) = self.work.single_item()
        tDgC = t2r.partition_D(cute.zipped_divide(ctx.tCgC[None, None, None, m, n], ctx.epi_tiler))
        tDcC = t2r.partition_D(cute.zipped_divide(ctx.tCcC[None, None, None, m, n], ctx.epi_tiler))
        rAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, cutlass.Float32)
        rC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, self.dtype)
        tAcc = cute.make_tensor(tmem.ptr("acc"), ctx.acc_layout)
        _, acc_consumer = ctx.acc_pipe.make_participants()
        a = acc_consumer.wait_and_advance()
        tDtC = t2r.partition_S(cute.zipped_divide(tAcc[None, None, None, a.index], ctx.epi_tiler))
        for i in cutlass.range_constexpr(cute.size(tDtC, mode=[2])):
            cute.copy(t2r, tDtC[None, None, i], rAcc)
            rC.store(rAcc.load().to(self.dtype))
            g, crd = tDgC[None, None, i], tDcC[None, None, i]
            for j in cutlass.range_constexpr(cute.size(rC)):
                if cute.elem_less(crd[j], ctx.mC.shape):
                    g[j] = rC[j]
        a.release()

    def run_roles(self, ctx):
        # User code between begin_roles() and run_roles(): build the raw pipeline on the
        # reserved barriers; the layer's init fence and CTA sync cover its initialization.
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            barrier_storage=self.barrier_ptr("acc_bars"),
            defer_sync=True,
        )
        values = {f.name: getattr(ctx, f.name) for f in fields(ctx)}
        super().run_roles(MixedGemmParams(**values, acc_pipe=acc_pipe))


_MIXED = MixedGemm()


@pytest.mark.parametrize("m, n, k", [(128, 256, 64), (300, 520, 136), (1000, 1000, 1000)])
def test_mixed_managed_and_raw_gemm(requires_blackwell, m, n, k):
    from attn_gym.warpdrive.examples.gemm import compile_gemm

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    op = _MIXED  # one instance: the compile cache traces it once per process
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    compile_gemm(op)(a, b, out)
    assert (out.float() - ref).abs().max().item() <= 2 * torch_err + 1e-3
    text = op.describe()
    assert "  0-3    epilogue    160  all  raw\n" in text
    assert "  acc_bars  2 mbarriers  user-initialized  user-driven (unchecked)" in text
    assert "override  empty_counts 1 x3 -> 1 x3\n" in text
    assert "differs from derived" not in text
    report = op._ws_tma["ab"]
    assert report.groups == {"ONE"} and report.stage_bytes == {49152}


def test_raw_role_runs_under_its_budget_branch(monkeypatch, traced_compiles):
    """setmaxnreg is applied once per budget branch (a run of adjacent warpgroups with one
    budget, under a warpgroup-uniform condition); a raw role is dispatched inside its branch
    like any role (it does not adjust registers itself), and every IKET role range opens after
    its budget's adjustment (compile only)."""
    from attn_gym.warpdrive import tracing
    from attn_gym.warpdrive.examples.gemm import compile_gemm

    events = []
    for fn in ("setmaxregister_increase", "setmaxregister_decrease"):
        real = getattr(cute.arch, fn)
        monkeypatch.setattr(
            cute.arch, fn, lambda regs, *a, _real=real, **kw: (events.append(regs), _real(regs))
        )
    iket = tracing.iket

    def push(name):
        events.append(name)
        return iket.range_push(name)

    monkeypatch.setattr(
        tracing, "iket", SimpleNamespace(range_push=push, range_pop=iket.range_pop)
    )
    op = MixedGemm().enable_iket(waits=False)
    compile_gemm.cache_clear()  # trace even if another test compiled the same key
    compile_gemm(op)
    budgets = [regs for regs, _, _ in op._ws_budget_groups]
    assert [e for e in events if isinstance(e, int)] == budgets  # once per branch, raw included
    regs_of = {r.name: r.regs for r in op._ws_roles}
    assert regs_of["epilogue"] == 160
    current, seen = None, set()
    for e in events:
        if isinstance(e, int):
            current = e
        elif e in regs_of:
            assert current == regs_of[e], (e, events)
            seen.add(e)
    assert seen == set(regs_of)
