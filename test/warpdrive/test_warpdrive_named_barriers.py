"""Test named-barrier ID allocation, counts, collisions, and role handoffs.

A runtime toy uses a one-shot flag between roles and a sync within one role.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute
from probe_kernels import probe_kernel

from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    NamedBarrier,
    Producer,
    Ring,
    Threads,
    WarpSpecialized,
    named_barrier,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor


class FlagToy(WarpSpecialized):
    """publisher (warp 0, raw) -> done -> waiters (warps 1-2), then pair_sync inside the waiters.

    The publisher stalls for ``delay`` cycles, writes ``value + lane`` to SMEM and arrives on
    the one-shot ``done`` flag; the waiters sync on it, so they read the value rather than the
    sentinel. Waiter warp 0 then stalls and doubles the value into a second SMEM row before the
    waiters' ``pair_sync``; waiter warp 1 reads that row after it. ``mode="native"`` makes the
    waiters sync on ``done`` with a native ``bar.sync`` on ``done.id``; other modes inject
    errors."""

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.done = NamedBarrier(threads=96)
        short = mode in ("short_users", "host_id")
        self.pair_sync = NamedBarrier(threads=96 if short else 64)
        if mode == "foreign":
            other = FlagToy()
            other.describe()  # allocates other.done the same ID as self.done
            self.borrowed = (other.done,)

    @role(warps=0, raw=True)
    @cute.jit
    def publisher(self, ctx):
        lane = cute.arch.lane_idx()
        start = cute.arch.clock64()
        while cute.arch.clock64() - start < ctx.delay:
            pass
        ctx.smem[0, lane] = ctx.value + lane
        self.done.arrive()
        if cutlass.const_expr(self.mode == "collision"):
            named_barrier(self.done.id, 96)
        if cutlass.const_expr(self.mode == "out_of_range"):
            named_barrier(16, 32)
        if cutlass.const_expr(self.mode == "foreign"):
            self.borrowed[0].sync()

    @role(warps=range(1, 3))
    @cute.jit
    def waiters(self, ctx):
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 1
        if cutlass.const_expr(self.mode == "native"):
            cute.arch.barrier(barrier_id=self.done.id, number_of_threads=self.done.threads)
        else:
            self.done.sync()
        value = ctx.smem[0, lane]
        if warp == 0:
            start = cute.arch.clock64()
            while cute.arch.clock64() - start < ctx.delay:
                pass
            ctx.smem[1, lane] = value * 2
        self.pair_sync.sync()
        if warp == 1:
            ctx.out[lane, 0] = value
            ctx.out[lane, 1] = ctx.smem[1, lane]

    @cute.kernel
    def kernel(self, out, value: cutlass.Int32, delay: cutlass.Int64):
        smem = self.begin_roles()
        buf = smem.allocate_tensor(cutlass.Int32, cute.make_layout((2, 32), stride=(32, 1)), 16)
        tidx = cute.arch.thread_idx()[0]
        if tidx < 64:
            buf[tidx // 32, tidx % 32] = -1  # run_roles' CTA sync publishes the sentinel
        self.run_roles(SimpleNamespace(out=out, smem=buf, value=value, delay=delay))

    @cute.jit
    def __call__(self, out: cute.Tensor, value: cutlass.Int32, delay: cutlass.Int64, stream):
        if cutlass.const_expr(self.mode == "host_id"):
            print(self.pair_sync.id)  # host code: no native call joins the barrier
        self.kernel.set_name_prefix(f"warpdrive_named_barrier_toy_{self.mode}")
        self.kernel(out, value, delay).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_flag_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (32, 2), assumed_align=4)
    return compile_tvm_ffi(
        op, out, cutlass.Int32(0), cutlass.Int64(0), name=f"named_barrier_toy_{op.mode}"
    )


@pytest.mark.parametrize("mode", ["ok", "native"])
def test_one_shot_flag_and_intra_role_sync(requires_blackwell, mode):
    """Delayed writes become visible after the one-shot flag and intra-role sync.

    Native sync on the managed flag's ID must also observe the publisher's write.
    """
    value, delay = 1000, 20000
    out = torch.full((32, 2), -5, device="cuda", dtype=torch.int32)
    compile_flag_toy(FlagToy(mode))(out, value, delay)
    expected = torch.arange(32, device="cuda", dtype=torch.int32) + value
    torch.testing.assert_close(out[:, 0], expected)
    torch.testing.assert_close(out[:, 1], expected * 2)


def test_describe_lists_named_barriers_and_their_users():
    """A compile lists each barrier's users; a second instance served from the cache restores
    them from the trace record."""
    expected = [
        "done id 3 96 threads arrive by publisher; sync by waiters",
        "pair_sync id 4 64 threads sync by waiters",
    ]
    for op in (FlagToy(), FlagToy()):
        assert "users known after a compile" in op.describe()
        compile_flag_toy(op)
        rows = [" ".join(line.split()) for line in op.describe().splitlines()]
        assert all(row in rows for row in expected)
    assert compile_flag_toy.cache_info().hits >= 1


def test_native_id_users_leave_the_count_unverified():
    """Native waiters complete managed arrivals on ``done.id``.

    Only 32 of 96 threads are traced, so the barrier is unverified, not rejected.
    """
    op = FlagToy("native")
    compile_flag_toy(op)
    row = next(line for line in op.describe().splitlines() if line.strip().startswith("done "))
    assert "arrive by publisher; native id by waiters" in row
    assert "unverified: roles calling it have 32 of 96 threads" in row


SHORT_USERS = (
    r"NamedBarrier pair_sync counts 96 threads, but the roles using it \(waiters\) have only 64; "
    r"it can never complete"
)


@pytest.mark.parametrize(
    "mode, message",
    [
        ("collision", r"named_barrier\(3\) at .* uses a barrier ID owned by NamedBarrier done"),
        ("out_of_range", r"named_barrier\(16\) at .*: barrier IDs are 0-15"),
        ("foreign", r"publisher: NamedBarrier done at .* belongs to another kernel"),
        ("short_users", SHORT_USERS),
        ("host_id", SHORT_USERS),  # a host-side .id read joins no barrier
    ],
)
def test_named_barrier_trace_errors(mode, message):
    _raises_protocol(message, lambda: compile_flag_toy(FlagToy(mode)))


# Allocation and declaration checks (host only)


@role(warps=0)
@cute.jit
def _prod(self, ctx, *, a: Producer):
    return a


@role(warps=1)
@cute.jit
def _cons(self, ctx, *, a: Consumer):
    return a


PAIR = {"prod": _prod, "cons": _cons}


def test_named_barriers_allocate_upward_and_leave_channel_ids():
    ring = {"a": lambda: Ring(2, Threads, Threads, sync="named")}
    bars = {"x": lambda: NamedBarrier(threads=64), "y": lambda: NamedBarrier(threads=32)}
    plain = probe_kernel(ring, PAIR)
    op = probe_kernel({**ring, **bars}, PAIR)
    assert op.named_barrier_ids == plain.named_barrier_ids == {"a": ([12, 13], [14, 15])}
    op.describe()
    assert (op.x.id, op.y.id) == (3, 4)
    # The channel's slots are NamedBarriers too, counting both sides' threads.
    slots = [(nb.name, nb.id, nb.threads) for nb in (*op.a.full_barriers, *op.a.empty_barriers)]
    assert slots == [("a.full[0]", 12, 64), ("a.full[1]", 13, 64)] + [
        ("a.empty[0]", 14, 64),
        ("a.empty[1]", 15, 64),
    ]


def test_named_barrier_ids_are_exhaustible():
    # 6 channel stages take 12 IDs; two NamedBarriers make 14 of the 13 free IDs (3-15).
    attrs = {
        "a": lambda: Ring(6, Threads, Threads, sync="named"),
        "x": lambda: NamedBarrier(threads=64),
        "y": lambda: NamedBarrier(threads=64),
    }
    _raises_protocol("need 14 barrier IDs; 13 are free", probe_kernel(attrs, PAIR).describe)
    del attrs["y"]
    probe_kernel(attrs, PAIR).describe()  # 13 fit exactly


@pytest.mark.parametrize("threads", [0, 48, 2048, 64.0, True])
def test_named_barrier_counts_whole_warps(threads):
    with pytest.raises(ValueError, match="int multiple of 32 in"):
        NamedBarrier(threads=threads)


SHARED = NamedBarrier(threads=64)


@pytest.mark.parametrize(
    "attrs, message",
    [
        ({"x": lambda: NamedBarrier(threads=96)}, "counts 96 threads, but the kernel launches 64"),
        (
            {"x": lambda: SHARED, "y": lambda: SHARED},
            "attributes 'x' and 'y' use the same declaration",
        ),
    ],
    ids=["count_exceeds_kernel", "shared_declaration"],
)
def test_named_barrier_declaration_errors(attrs, message):
    op = probe_kernel({"a": lambda: Ring(1, Threads, Threads), **attrs}, PAIR)
    _raises_protocol(message, op.describe)


def test_named_barrier_id_needs_a_kernel():
    _raises_protocol("has no ID yet", lambda: NamedBarrier(threads=64).id)


class PairProbe(WarpSpecialized):
    """A CTA pair whose ``x`` is synced by a leader-only warp and a warp in both CTAs."""

    cluster = CtaPair()

    def __init__(self, threads: int, native: bool = False):
        self.native = native
        self.x = NamedBarrier(threads=threads)

    @role(warps=0, cta="leader")
    @cute.jit
    def lead(self, ctx):
        self.x.sync()

    @role(warps=1)
    @cute.jit
    def both(self, ctx):
        if cutlass.const_expr(self.native):
            cute.arch.barrier(barrier_id=self.x.id, number_of_threads=self.x.threads)
        else:
            self.x.sync()

    @cute.kernel
    def kernel(self, out):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out))

    @cute.jit
    def __call__(self, out: cute.Tensor, stream):
        self.kernel.set_name_prefix(
            f"warpdrive_named_barrier_pair_{self.x.threads}_{int(self.native)}"
        )
        self.kernel(out).launch(**self.launch_config(grid=(2, 1, 1)), stream=stream)


@jit_cache
def compile_pair_probe(op):
    out = make_fake_compact_tensor(cutlass.Int32, (1,), assumed_align=4)
    return compile_tvm_ffi(op, out, name=f"named_barrier_pair_{op.x.threads}_{int(op.native)}")


@pytest.mark.parametrize(
    "native, message",
    [
        # The peer CTA runs only ``both``: 32 threads can never complete a 64-thread barrier
        # there, though the leader CTA has 64.
        (
            False,
            r"x counts 64 threads, but the roles using it in CTA rank 1 \(both\) have only 32",
        ),
        # A native ID user leaves missing arrivals unverified, but no native call can complete
        # a 64-thread barrier in a CTA where only 32 threads run any role.
        (True, r"x counts 64 threads, but only 32 run the roles in CTA rank 1; it can never"),
    ],
    ids=["managed", "native"],
)
def test_named_barrier_users_are_counted_per_cta(native, message):
    _raises_protocol(message, lambda: compile_pair_probe(PairProbe(64, native=native)))
    compile_pair_probe(PairProbe(32))
