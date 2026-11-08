"""Optional channel storage: SmemPlan buffers bound to channels, stage projection through
handles, registered partitions (StageView), overlaps, and the declaration/trace-time errors."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import cute

from attn_gym.testing.warpdrive import raises_protocol as _raises_protocol
from attn_gym.warpdrive import (
    Consumer,
    Producer,
    ProtocolError,
    Ring,
    SmemPlan,
    StageView,
    StaticPersistent,
    Threads,
    Tma,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor

STAGES = 3


# Storage toy: warp 0 writes epoch i's value into the ring's X stage through ``h.data``; warp 1
# reads it back through a registered (4, 8) partition of X, through the whole buffer at
# ``h.index``, and through Y, a Float32 buffer overlapping X's bytes. ``mode`` injects errors.


class StorageToy(WarpSpecialized):
    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.smem_plan = SmemPlan()
        x = self.smem_plan.buffer("X", cutlass.Int32, align=128)
        y = self.smem_plan.buffer("Y", cutlass.Float32, overlaps=x)
        self.ch = Ring(STAGES, Threads, Threads, buffer={"X": x, "Y": y})

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, ctx):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.produce() as h:
                if cutlass.const_expr(self.mode == "unnamed_view"):
                    h.view()[lane] = 0  # two buffers: the name is required
                h.data["X"][lane] = i * 32 + lane
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, ctx):
        lane = cute.arch.lane_idx()
        x = ctx.x  # (32, STAGES), column-major
        grid = cute.make_tensor(x.iterator, cute.make_layout((4, 8, STAGES), stride=(8, 1, 32)))
        if cutlass.const_expr(self.mode == "bad_extent"):
            grid = cute.make_tensor(x.iterator, cute.make_layout((4, 8, 2), stride=(8, 1, 32)))
        name = "Z" if self.mode == "unknown_view" else "X"
        ch = ch.views(**{name: StageView(grid, axis=2)})
        for i in cutlass.range(ctx.n, unroll=1):
            with ch.consume() as h:
                ctx.out[i * 32 + lane, 0] = h.view("X")[lane // 8, lane % 8]
                ctx.out[i * 32 + lane, 1] = x[lane, h.index]
                ctx.outf[i * 32 + lane] = h.view("Y")[lane]
        return ch

    @cute.kernel
    def kernel(self, out, outf, n: cutlass.Int32, x_layout, y_layout):
        layouts = {"X": x_layout, "Y": y_layout}
        if cutlass.const_expr(self.mode == "missing_layout"):
            layouts = {"X": x_layout}
        self.begin_roles(layouts=layouts)
        self.run_roles(SimpleNamespace(out=out, outf=outf, n=n, x=self.buffer("X")))

    @cute.jit
    def __call__(self, out: cute.Tensor, outf: cute.Tensor, n: cutlass.Int32, stream):
        stages = 2 if self.mode == "short_layout" else STAGES
        x_layout = cute.make_layout((32, stages))
        y_rows = 64 if self.mode == "overlap_too_big" else 32
        y_layout = cute.make_layout((y_rows, stages))
        self.kernel.set_name_prefix("warpdrive_storage_toy")
        self.kernel(out, outf, n, x_layout, y_layout).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_storage_toy(op):
    out = make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(), 2), assumed_align=4)
    outf = make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),), assumed_align=4)
    return compile_tvm_ffi(op, out, outf, cutlass.Int32(1), name=f"storage_toy_{op.mode}")


def test_buffer_views_project_each_handles_stage(requires_blackwell):
    """Canonical views, a registered partition, the whole buffer at h.index, and an
    overlapping buffer all address the stage the handle owns, across ring wraps."""
    op = StorageToy()
    n = 2 * STAGES + 1
    out = torch.full((n * 32, 2), -1, device="cuda", dtype=torch.int32)
    outf = torch.zeros(n * 32, device="cuda", dtype=torch.float32)
    compile_storage_toy(op)(out, outf, n)
    expected = torch.arange(n * 32, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(out[:, 0], expected)
    torch.testing.assert_close(out[:, 1], expected)
    assert torch.equal(outf.view(torch.int32), expected)  # Y reads X's bytes
    text = op.describe()
    assert "storage   buffer X, Y" in text
    assert "SMEM plan: 384 B in 1 buffer(s)" in text
    assert "X  Int32    384 B  3 stages (mode -1)  align 128   -> ch" in text
    assert "Y  Float32  384 B  3 stages (mode -1)  overlaps X  -> ch" in text


@pytest.mark.parametrize(
    "mode, message",
    [
        ("short_layout", r"buffer 'X': stage mode -1 of X's layout has 2 stages, but ch has 3"),
        ("bad_extent", r"ch\.views\(X=\.\.\.\): the view's stage mode 2 has 2 stages"),
        ("unknown_view", r"the channel has no buffer 'Z'"),
        ("unnamed_view", r"channel ch has buffers \['X', 'Y'\]; name the one to view"),
        ("missing_layout", r"needs a layout for each smem_plan buffer \['X', 'Y'\]"),
        ("overlap_too_big", r"SMEM buffer Y \(768 B\) overlaps X, which has only 384 B"),
    ],
)
def test_storage_trace_time_errors(mode, message):
    _raises_protocol(message, lambda: compile_storage_toy(StorageToy(mode)))


def test_storage_declaration_errors():
    plan = SmemPlan()
    a = plan.buffer("A", cutlass.BFloat16, align=1024)
    with pytest.raises(ValueError, match="already declared"):
        plan.buffer("A", cutlass.BFloat16)
    with pytest.raises(ValueError, match="power-of-two"):
        plan.buffer("B", cutlass.BFloat16, align=48)
    with pytest.raises(ValueError, match="needs 2048-byte alignment"):
        plan.buffer("B", cutlass.BFloat16, align=2048, overlaps=a)
    with pytest.raises(ProtocolError, match="overlaps unknown buffer 'C'"):
        plan.buffer("B", cutlass.BFloat16, overlaps="C")
    with pytest.raises(ProtocolError, match="not in this plan"):
        plan.buffer("B", cutlass.BFloat16, overlaps=SmemPlan().buffer("A", cutlass.BFloat16))
    with pytest.raises(ValueError, match="payload= or buffer=, not both"):
        Ring(2, payload=(cutlass.Int32, (4,)), buffer=a)
    with pytest.raises(ValueError, match="SmemBuffer or a non-empty"):
        Ring(2, buffer={"A": "not a buffer"})


def test_channel_buffer_must_belong_to_the_kernels_plan():
    class Foreign(StorageToy):
        def __init__(self):
            super().__init__()
            self.ch = Ring(STAGES, buffer=SmemPlan().buffer("X", cutlass.Int32))

    with pytest.raises(ProtocolError, match="is not a buffer of this kernel's smem_plan"):
        Foreign().describe()


def test_buffer_keeps_remote_scope_and_pair_channels():
    """Unlike a payload, a buffer implies no DSMEM publication: the default remote scope stays
    "cta", and pair-scoped channels may bind one (the 2-CTA persistent GEMM's A/B ring)."""
    from attn_gym.warpdrive import CtaPair
    from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm

    assert Ring(2, buffer=SmemPlan().buffer("A", cutlass.Int32)).remote_scope == "cta"
    # A payload that thread arrivals publish derives a cluster-scope release; one a Tma producer
    # completes by bytes (the cluster work ring) leaves only slot hand-backs, at CTA scope.
    payload = (cutlass.Int32, (4,))
    assert Ring(2, Threads, Threads, payload=payload).remote_scope == "cluster"
    assert Ring(2, Tma(16), Threads, payload=payload).remote_scope == "cta"
    text = PersistentGemm(cluster=CtaPair()).describe()
    ab = text[text.index("  ab ") : text.index("  acc ")]
    assert "pair-scoped" in ab and "storage   buffer A, B" in ab


def test_fa4_o_over_q_runs_correctly(requires_blackwell):
    """FA4 with sO placed over sQ (one work item per CTA) matches the reference."""
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import (
        LOG2_E,
        FlashAttentionForward,
        compile_fa4_fwd,
    )

    torch.manual_seed(0)
    b, h, sq, sk = 2, 3, 640, 512
    q, k, v = (
        torch.randn(b, s, h, 128, device="cuda", dtype=torch.bfloat16) for s in (sq, sk, sk)
    )
    op = FlashAttentionForward(overlap_o_q=True)
    out, lse = torch.empty_like(q), torch.empty(b, h, sq, device="cuda", dtype=torch.float32)
    workspace, sms = op.work.launch_args(q.device)
    compile_fa4_fwd(op)(q, k, v, out, lse, 128**-0.5 * LOG2_E, workspace, sms)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)
    assert (out.float() - ref).abs().max().item() <= 2**-7 * ref.abs().max().item() + 4e-3
    assert "O  BFloat16  65536 B  2 stages (mode -1)  overlaps Q" in op.describe()
    with pytest.raises(ValueError, match="overlap_o_q needs one work item per CTA"):
        FlashAttentionForward(overlap_o_q=True, schedule=StaticPersistent())
