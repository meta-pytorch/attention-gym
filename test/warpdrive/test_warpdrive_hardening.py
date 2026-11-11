"""Test TMA checks and boundaries between managed and raw synchronization.

Cover raw state, signals, roles, barriers, overrides, and per-site opt-outs, plus a
mixed managed/raw GEMM and the raw-interop example.
"""

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

from attn_gym.testing.warpdrive import raises_protocol
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
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.gemm import Gemm, GemmParams
from attn_gym.warpdrive.examples.raw_interop import (
    WORDS,
    RawInterop,
    compile_raw_interop,
    raw_interop,
)

TILE_ROWS = 64  # one 64x64 bf16 TMA tile per stage
TILE_BYTES = TILE_ROWS * 64 * 2
DELAY = 200_000  # cycles: far longer than one try_acquire probe may suspend


@cute.jit
def spin(cycles):
    start = cute.arch.clock64()
    while cute.arch.clock64() - start < cycles:
        pass


# TMA toy: a load role copies 64x64 tiles into a 2-stage ring; the consumer records the first
# element of each tile (tile i holds the value i + 1), each after spinning ``delay`` cycles.
# ``consumer_cta`` places the consumer: in a pair, "leader" makes the channel pair-scoped (both
# CTAs' bytes complete on the leader), "all" keeps it CTA-local. ``group`` picks the atom's
# CtaGroup; ``tx`` the declared bytes; ``mode`` how the load role issues its copies.


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
            if cutlass.const_expr(self.mode == "token"):
                token = ld.try_acquire()
                h = ld.acquire(token=token)
            else:
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
            if cutlass.const_expr(self.mode == "token"):
                if cute.arch.lane_idx() == 0:  # the probe's readiness, after the tiles
                    ctx.out[ctx.n + i] = cutlass.Float32(cutlass.Int32(token.ready))
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
            spin(ctx.delay)
            with ld.consume() as h:
                if cute.arch.lane_idx() == 0:
                    first = ctx.s[0, 0, h.index].to(cutlass.Float32)
                    last = ctx.s[TILE_ROWS - 1, 63, h.index].to(cutlass.Float32)
                    ctx.out[rank * ctx.n + i] = first + 1000.0 * (last - first)
        return ld

    @cute.kernel
    def kernel(self, tma, tg, g, out, n: cutlass.Int32, tiles: cutlass.Int32, delay, s_layout):
        smem = self.begin_roles()
        s = smem.allocate_tensor(cutlass.BFloat16, s_layout, 128)
        ts, tgp = cpasync.tma_partition(
            tma, 0, cute.make_layout(1), cute.group_modes(s, 0, 2), cute.group_modes(tg, 0, 2)
        )
        bulk = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), cutlass.BFloat16)
        ctx = SimpleNamespace(
            tma=tma, tg=tgp, ts=ts, s=s, g=g, bulk=bulk, out=out, n=n, tiles=tiles, delay=delay
        )
        self.run_roles(ctx)

    @cute.jit
    def __call__(self, src, out, n: cutlass.Int32, delay: cutlass.Int64, stream):
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
        self.kernel(tma, tiles, g, out, n, src.shape[0] // TILE_ROWS, delay, s_layout).launch(
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
    name = f"warpdrive_tma_toy_{op.key}"
    return compile_tvm_ffi(op, src, out, cutlass.Int32(1), cutlass.Int64(0), name=name)


def run_tma_toy(op, n=7, tiles=3, delay=0):
    """Checks every reader's tiles; returns what follows them (token mode: probe results)."""
    rows = torch.arange(tiles * TILE_ROWS, device="cuda") // TILE_ROWS + 1
    src = rows.view(-1, 1).expand(-1, 64).to(torch.bfloat16).contiguous()
    src[torch.arange(tiles, device="cuda") * TILE_ROWS + TILE_ROWS - 1, 63] += 0.5  # last elem
    out = torch.zeros((op.cluster.size + 1) * n, device="cuda", dtype=torch.float32)
    compile_tma_toy(op)(src, out, n, delay)
    torch.cuda.synchronize()
    expect = [i % tiles + 1 + 500.0 for i in range(n)]
    readers = 1 if op.consumer_cta == "leader" else op.cluster.size
    assert out[: readers * n].view(readers, n).tolist() == [expect] * readers
    return out[op.cluster.size * n :].tolist()


@pytest.mark.parametrize(
    "cluster, consumer_cta, group, mode, verified",
    [
        (SingleCta(), "all", "ONE", "plain", "CtaGroup.ONE"),
        (CtaPair(), "leader", "TWO", "plain", "CtaGroup.TWO"),
        (CtaPair(), "all", "ONE", "plain", "CtaGroup.ONE"),
        # A cp.async.bulk load (mbar_ptr=) inside elect_one() is counted like a tensor TMA
        # load: its bytes complete on the issuing CTA's barrier.
        (SingleCta(), "all", "ONE", "bulk", "cp.async.bulk"),
    ],
    ids=["single", "pair_scoped", "pair_local", "bulk"],
)
def test_tma_check_accepts_matching_atoms(
    requires_blackwell, cluster, consumer_cta, group, mode, verified
):
    op = make_tma_toy(cluster, consumer_cta, group, mode=mode)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a verified kernel emits no "unverified" warning
        run_tma_toy(op)
    assert f"tma       verified {verified}, {TILE_BYTES} B/stage per CTA @ " in op.describe()


def test_tma_try_acquire_arms_once(requires_blackwell):
    """A token-based acquire must arm the TMA barrier exactly once.

    With a delayed reader, try_acquire probes fail after the first two stages. Arming twice
    hangs the full barrier; omitting the arm lets a reader see an incomplete copy.
    """
    op = make_tma_toy(SingleCta(), mode="token")
    run_tma_toy(op)
    assert run_tma_toy(op, delay=DELAY) == [1.0, 1.0] + [0.0] * 5


PAIR = {"cluster": CtaPair(), "consumer_cta": "leader", "group": "TWO"}
SINGLE = {"cluster": SingleCta()}


@pytest.mark.parametrize(
    "toy, message",
    [
        ({**PAIR, "group": "ONE"}, r"needs CtaGroup\.TWO: ld is pair-scoped.*leader hangs"),
        ({**PAIR, "consumer_cta": "all"}, r"needs CtaGroup\.ONE: ld is CTA-local"),
        # The leader arms both CTAs' bytes, but a bulk copy from the peer completes on the
        # peer's own barrier: rejected like a CtaGroup.ONE tensor load.
        (
            {**PAIR, "mode": "bulk"},
            (
                r"a cp\.async\.bulk copy, which completes on the issuing CTA's barrier, but ld "
                r"needs CtaGroup\.TWO: ld is pair-scoped.*use a tensor TMA atom"
            ),
        ),
        (
            {**SINGLE, "tx": TILE_BYTES // 2},
            (
                rf"deliver {TILE_BYTES} bytes, but the layer arms {TILE_BYTES // 2} .*completes "
                rf"before every copy lands.*Tma\({TILE_BYTES}\)"
            ),
        ),
        (
            {**SINGLE, "tx": 2 * TILE_BYTES, "mode": "bulk"},
            rf"deliver {TILE_BYTES} bytes, but the layer arms {2 * TILE_BYTES} .*never completes",
        ),
        # Pair-scoped: the leader arms 2 x Tma(tx); each CTA must copy tx.
        (
            {**PAIR, "tx": 2 * TILE_BYTES},
            rf"deliver {TILE_BYTES} bytes per CTA, but the layer arms {2 * TILE_BYTES} per CTA",
        ),
    ],
    ids=[
        "one_atom_on_pair",
        "two_atom_on_local",
        "bulk_on_pair",
        "short_tx",
        "long_tx",
        "pair_tx",
    ],
)
def test_tma_check_rejects_hanging_kernels(toy, message):
    """Compile-time errors for kernels that would hang on the GPU (they are never launched),
    pointing at the user's copy line."""
    op = make_tma_toy(**toy)
    raises_protocol(rf"(?s)test_warpdrive_hardening\.py:.*{message}", lambda: compile_tma_toy(op))


def test_tma_check_fails_open_on_opaque_barrier(requires_blackwell):
    """A barrier pointer not issued by the layer is unverified, not rejected.

    The kernel still runs, describe() reports it unverified, and one warning is emitted.
    """
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
    """Incomplete byte accounting is unverified, not an error.

    These kernels are correct, but the check cannot see every copy.
    """
    op = make_tma_toy(SingleCta(), tx=tx, mode=mode)
    assert op.checks().channels["ld"].tma == "not checked yet"
    with pytest.warns(UserWarning, match=r"could not verify channel\(s\) \['ld'\]"):
        compile_tma_toy(op)
    assert op.checks().channels["ld"].tma == "unverified"
    assert re.search(rf"tma       unverified \(.*{reason}", op.describe()), op.describe()


def test_tma_opt_out_is_listed(requires_blackwell):
    op = make_tma_toy(SingleCta(), mode="optout")
    run_tma_toy(op)
    text = op.describe()
    assert "Unchecked opt-outs\n  unchecked tma @ test_warpdrive_hardening.py:" in text
    assert "load: 'toy: atom verified by hand' -> ld barrier at" in text
    assert op.checks().channels["ld"].tma == "opted out"


# Raw state and raw signals on managed ends


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


# Per-site opt-outs


class OptOutToy(WarpSpecialized):
    """Compiled only: ``mode`` leaves a handle open or closes it in another region."""

    liveness_check = "off"

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
    raises_protocol(error, lambda: _compile_opt_out_toy(OptOutToy(mode, False)))
    op = OptOutToy(mode, True)
    _compile_opt_out_toy(op)
    text = op.describe()
    assert "unchecked " in text and "test_warpdrive_hardening.py:" in text
    assert re.search(listed, text), text


# Host-side declarations: raw roles, Barriers, overrides


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
    op = _RawInPersistent()
    text = op.describe()  # raw roles need not loop over the stream
    assert op.num_warps == 8  # padded to whole warpgroups
    assert "  4-7    side        200  all  raw\n" in text
    assert "  bars  2 mbarriers  layer-initialized [32, 1]  user-driven (unchecked)" in text


class RawCountToy(WarpSpecialized):
    """One raw warp arrives lane by lane on a layer-initialized ``Barriers(1, counts=32)`` and
    records after each arrival whether phase 0 has completed."""

    def __init__(self):
        self.bar = Barriers(1, counts=32)

    @role(warps=0, raw=True)
    @cute.jit
    def lanes(self, ctx, *, bar: Barriers):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(32, unroll=1):
            if lane == i:
                cute.arch.mbarrier_arrive(bar)
            cute.arch.sync_warp()
            if lane == i:
                ctx.out[i] = cutlass.Int32(cute.arch.mbarrier_test_wait(bar, 0))

    @cute.kernel
    def kernel(self, out: cute.Tensor):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out))

    @cute.jit
    def __call__(self, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(out).launch(**self.launch_config(grid=(1, 1, 1)), stream=stream)


def test_layer_initialized_barriers_complete_after_their_count(requires_blackwell):
    fake = make_fake_compact_tensor(cutlass.Int32, (32,), assumed_align=4)
    fn = compile_tvm_ffi(RawCountToy(), fake, name="warpdrive_raw_count")
    out = torch.full((32,), -1, device="cuda", dtype=torch.int32)
    fn(out)
    assert out.tolist() == [0] * 31 + [1]


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
    assert op.back.scope.empty_mask == 0b01  # derived: the leader-only producer's CTA
    assert "scope     pair-scoped, commit_mask 0b01, remote_scope cta\n" in text
    assert "override  full_mask 0b11 -> 0b01  (differs from derived)" in text


@pytest.mark.parametrize(
    "make, error, message",
    [
        (lambda: _RawWithEnd().describe(), ProtocolError, "raw role prod: parameter ch is Prod"),
        (lambda: Barriers(2, counts=[1]), ValueError, "Barriers counts must be 2 positive ints"),
        (lambda: Ring(2, Threads, Threads, override={"x": 1}), ValueError, "unknown override"),
        (lambda: _Overrides(full_counts=[1, 2, 3]).describe(), ProtocolError, "2 positive cou"),
        (lambda: unchecked("unclosed", reason=" ").__enter__(), ValueError, "non-empty reason"),
        (lambda: unchecked("arrivals", reason="x").__enter__(), ValueError, "unknown check"),
    ],
    ids=["raw_role_end", "barrier_counts", "override_key", "override_counts", "reason", "check"],
)
def test_escape_hatch_declaration_errors(make, error, message):
    with pytest.raises(error, match=message):
        make()


# Mixed managed/raw GEMM: the ab edge stays managed but the MMA releases it with a raw
# tcgen05.commit; the acc edge is user-driven Barriers behind cutlass.pipeline.PipelineUmmaAsync;
# the epilogue is a raw role.


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

    @role(warps=5, regs=48)
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


def test_mixed_managed_and_raw_gemm(requires_blackwell):
    from attn_gym.warpdrive.examples.gemm import compile_gemm

    op = MixedGemm()
    fn = compile_gemm(op)
    torch.manual_seed(0)
    for m, n, k in [(300, 520, 136), (1000, 1000, 1000)]:  # ragged; many tiles and ring wraps
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(a.float(), b.float())
        torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
        out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        fn(a, b, out)
        assert (out.float() - ref).abs().max().item() <= 2 * torch_err + 1e-3
    text = op.describe()
    assert "  0-3    epilogue    160  all  raw\n" in text
    assert "  acc_bars  2 mbarriers  user-initialized  user-driven (unchecked)" in text
    assert "override  empty_counts 1 x3 -> 1 x3\n" in text
    assert "differs from derived" not in text
    assert "tma       verified CtaGroup.ONE, 49152 B/stage per CTA @ " in text


# The raw-interop example: a managed producer, a raw consumer that drives the ring's external
# consumer side by pointer under an overridden empty count, and a raw role that waits natively
# on a NamedBarrier's ID.


def test_raw_interop_describe_marks_what_the_raw_code_leaves_unverified():
    op = RawInterop()
    compile_raw_interop(op)
    text = op.describe()
    assert "ring  Ring(2)  producer -> raw code\n" in text, text
    assert "empty     1 x2\n" in text, text
    assert "external  consumer side driven by raw code; empty counts are the override's" in text
    row = next(line for line in text.splitlines() if line.strip().startswith("done "))
    assert "arrive by consumer; native id by watcher" in row, row
    assert "unverified: roles calling it have 64 of 96 threads" in row, row


@pytest.mark.parametrize("stages", [1, 2, 3])
def test_raw_interop_consumer_sees_every_epoch(requires_blackwell, stages):
    for n in (0, 1, 7, 64, 7):  # the repeat starts from fresh barriers
        bad, total = raw_interop(n, stages).tolist()
        assert bad == 0 and total == int(torch.arange(n * WORDS, dtype=torch.int64).sum()), n
