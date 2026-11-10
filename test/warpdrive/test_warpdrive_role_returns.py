"""Explicit role results keep final phases out of Python frame introspection."""

import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from attn_gym.warpdrive import (
    ClusterLaunchControl,
    Consumer,
    CtaPair,
    Producer,
    ProtocolError,
    Slots,
    StaticPersistent,
    Threads,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    PersistentGemmParams,
    compile_persistent_gemm,
)


class ReturnedProducer(WarpSpecialized):
    def __init__(self, mode="renamed"):
        self.mode = mode
        self.cluster = CtaPair()
        self.ch = Slots(1, Threads, Umma)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        live = ch
        ch = None  # a role may rename its argument; teardown follows the returned end
        for _ in cutlass.range(ctx.n, unroll=1):
            with live.produce(0):
                pass
        if cutlass.const_expr(self.mode == "missing"):
            return None
        if cutlass.const_expr(self.mode == "duplicate"):
            return live, live
        if cutlass.const_expr(self.mode == "wrong_type"):
            return 7
        return live

    @role(warps=1, cta="leader")
    @cute.jit
    def consume(self, ch: Consumer, ctx):
        count = cutlass.Int32(0)
        for _ in cutlass.range(ctx.n, unroll=1):
            with ch.consume(0):
                count += 1
        if cute.arch.lane_idx() == 0:
            ctx.out[0] = count
        if cutlass.const_expr(self.mode == "missing_consumer"):
            return None
        return ch

    @cute.kernel
    def kernel(self, out: cute.Tensor, n: cutlass.Int32):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out, n=n))

    @cute.jit
    def __call__(self, out: cute.Tensor, n: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(f"warpdrive_{type(self).__name__.lower()}_{self.mode}")
        self.kernel(out, n).launch(**self.launch_config(grid=(2, 1, 1)), stream=stream)


@jit_cache
def compile_producer(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (2,), assumed_align=4),
        cutlass.Int32(1),
        name=f"warpdrive_{type(op).__name__.lower()}_{op.mode}",
    )


def test_returned_renamed_producer_drains_final_state(requires_blackwell):
    fn = compile_producer(ReturnedProducer())
    for n in (0, 1, 9, 64):
        out = torch.full((2,), -1, device="cuda", dtype=torch.int32)
        fn(out, n)
        assert out[0].item() == n


@pytest.mark.parametrize(
    "mode, message",
    [
        ("missing", "return all warpdrive state arguments; missing ch"),
        ("missing_consumer", "return all warpdrive state arguments; missing ch"),
        ("duplicate", "returned the same state argument"),
        ("wrong_type", "return channel endpoints, work cursors and turnstile seats"),
    ],
)
def test_bad_role_result_rejected_before_launch(requires_blackwell, mode, message):
    with pytest.raises(Exception, match=message):
        compile_producer(ReturnedProducer(mode))


@cute.jit
def fill_slot(ch, n):
    for _ in cutlass.range(n, unroll=1):
        with ch.produce(0):
            pass
    return ch


class HelperProducer(ReturnedProducer):
    """A helper fills a slot in a dynamic loop and returns the advanced endpoint.

    Dropping its result leaves the caller at the pre-loop phases.
    """

    @cute.jit
    def fill(self, ch, n):
        for _ in cutlass.range(n, unroll=1):
            with ch.produce(0):
                pass
        return ch

    @cute.jit
    def read_phase(self, ch, ctx):
        for _ in cutlass.range(ctx.n, unroll=1):
            if cute.arch.lane_idx() == 0:
                ctx.out[1] = ch.phase(0)
        return ch

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        if cutlass.const_expr(self.mode == "read_only"):
            self.read_phase(ch, ctx)  # reads only: dropping its result loses nothing
            ch = self.fill(ch, ctx.n)
        if cutlass.const_expr(self.mode == "saved_pointer"):
            snapshot = ch
            ch = self.fill(ch, ctx.n)
            if cute.arch.lane_idx() == 0:
                ctx.out[1] = snapshot.empty_barrier(0).toint() - snapshot.full_barrier(0).toint()
        if cutlass.const_expr(self.mode == "dropped"):
            self.fill(ch, ctx.n)
        if cutlass.const_expr(self.mode == "dropped_free"):
            fill_slot(ch, ctx.n)
        if cutlass.const_expr(self.mode == "dropped_then_used"):
            self.fill(ch, ctx.n - 1)
            with ch.produce(0):
                pass
        if cutlass.const_expr(self.mode == "dropped_in_loop"):
            for _ in cutlass.range(ctx.n, unroll=1):
                self.fill(ch, 1)  # the loop does not even carry ch: each pass restarts
        return ch


@pytest.mark.parametrize("mode", ["read_only", "saved_pointer"])
def test_helper_end_assigned_or_read_only(requires_blackwell, mode):
    fn = compile_producer(HelperProducer(mode))
    for n in (0, 1, 9):
        out = torch.full((2,), -1, device="cuda", dtype=torch.int32)
        fn(out, n)
        assert out[0].item() == n
        if mode == "saved_pointer":
            assert out[1].item() == 8  # one slot: empty barrier is one 64-bit word after full


@pytest.mark.parametrize(
    "mode, message",
    [
        ("dropped", r"ch end returned at role exit is stale"),
        ("dropped_free", r"ch end returned at role exit is stale"),
        ("dropped_then_used", r"ch end used at test_warpdrive_role_returns\.py:\d+ is stale"),
        ("dropped_in_loop", r"ch end returned at role exit is stale"),
    ],
)
def test_dropped_helper_end_rejected(requires_blackwell, mode, message):
    with pytest.raises(ProtocolError, match=message) as info:
        compile_producer(HelperProducer(mode))
    hint = str(info.value)
    callee = "fill_slot" if mode == "dropped_free" else "self.fill"
    assert "assign the helper's returned end to the argument you passed" in hint
    assert f"schematically: `ch = {callee}(ch, ...)`" in hint
    assert "use your actual argument or alias" in hint


class ReturnedCursor(WarpSpecialized):
    def __init__(self, schedule, *, missing=False, drained_in_helper=False):
        self.work = WorkStream(schedule=schedule)
        self.missing = missing
        self.drained_in_helper = drained_in_helper

    @role(warps=0)
    @cute.jit
    def count(self, work: WorkCursor, ctx):
        live = work
        work = None
        if cutlass.const_expr(self.drained_in_helper):
            self.drain(live, ctx)
            return live
        while live.valid:
            i, _, _ = live.item
            if cute.arch.lane_idx() == 0:
                ctx.out[i] = i + 1
            live = self.next_item(live)
        if cutlass.const_expr(self.missing):
            return None
        return live

    @cute.jit
    def drain(self, work, ctx):
        while work.valid:
            i, _, _ = work.item
            if cute.arch.lane_idx() == 0:
                ctx.out[i] = i + 1
            work.advance()
        return work

    @cute.jit
    def next_item(self, work):
        work.advance()
        return work

    @cute.kernel
    def kernel(self, out: cute.Tensor, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(out=out))

    @cute.jit
    def __call__(self, out: cute.Tensor, workspace: cute.Tensor, stream: cuda.CUstream):
        bound = self.work.bind((out.shape[0], 1, 1), workspace, 2)
        self.kernel.set_name_prefix("warpdrive_returned_cursor")
        self.kernel(out, bound).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_cursor(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        name="warpdrive_explicit_cursor",
    )


@pytest.mark.parametrize(
    "schedule", [StaticPersistent(), ClusterLaunchControl()], ids=["static", "clc"]
)
def test_returned_cursor_from_helper(requires_blackwell, schedule):
    op = ReturnedCursor(schedule)
    fn = compile_cursor(op)
    workspace, _ = op.work.launch_args("cuda")
    for n in (1, 19, 33):
        out = torch.zeros(n, device="cuda", dtype=torch.int32)
        fn(out, workspace)
        torch.testing.assert_close(out, torch.arange(1, n + 1, device="cuda", dtype=torch.int32))


def test_dropped_helper_cursor_rejected(requires_blackwell):
    op = ReturnedCursor(StaticPersistent(), drained_in_helper=True)
    with pytest.raises(ProtocolError, match="work cursor returned at role exit is stale") as info:
        compile_cursor(op)
    hint = str(info.value)
    assert "schematically: `work = self.drain(work, ...)`" in hint
    assert "use your actual argument or alias" in hint


@pytest.mark.parametrize("iket", [False, True])
def test_cursor_return_required_independently_of_tracing(requires_blackwell, iket):
    op = ReturnedCursor(StaticPersistent(), missing=True).enable_iket(iket)
    with pytest.raises(Exception, match="return all warpdrive state arguments; missing work"):
        compile_cursor(op)


class ReorderedPersistentGemm(PersistentGemm):
    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: PersistentGemmParams, ab: Producer, work: WorkCursor):
        ab, work = PersistentGemm.load(self, params, ab=ab, work=work)
        return work, ab  # state identity, not tuple position, controls cleanup


def test_reordered_state_drives_pair_producer_tail(requires_blackwell):
    # One cluster processes 16 output tiles x 4 K tiles: repeated wraps and a partial last
    # round of the six-stage AB ring. The returned AB state is actually used by its pair tail.
    torch.manual_seed(0)
    a = torch.randn(900, 200, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(1000, 200, device="cuda", dtype=torch.bfloat16)
    outputs = []
    for cls in (PersistentGemm, ReorderedPersistentGemm):
        op = cls(cluster=CtaPair(), schedule=StaticPersistent())
        fn = compile_persistent_gemm(op)
        workspace, _ = op.work.launch_args(a.device)
        out = torch.full((900, 1000), float("nan"), device="cuda", dtype=torch.bfloat16)
        fn(a, b, out, workspace, 2)
        torch.cuda.synchronize()
        outputs.append(out)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_no_python_profiler_needed(requires_blackwell, monkeypatch, traced_compiles):
    def forbidden(*args):
        pytest.fail("role dispatch must not inspect Python frames through setprofile")

    compile_producer.cache_clear()  # trace even if another test compiled the same keys
    compile_cursor.cache_clear()
    monkeypatch.setattr(sys, "setprofile", forbidden)
    compile_producer(ReturnedProducer())
    compile_cursor(ReturnedCursor(StaticPersistent()).enable_iket())
