"""Test ``ConsumerEnd.consume_unread``: a joining consumer passes over an epoch it does not read.

Two consumer roles join every epoch of a ring with a declared release fence; ``skipper`` reads
only even epochs and passes over the odd ones. The pass waits for the epoch and releases it
without the fence (nothing was read), so the ring keeps cycling and both roles still see the
right values. The reader holds each stage before reading it, so a pass that released without
waiting would let the producer overwrite the stage first.
"""

import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass._mlir._mlir_libs._cutlass_ir import _aot_support

from attn_gym.warpdrive import Consumer, Producer, Ring, Threads, WarpSpecialized, role
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor

EPOCHS = 12  # wraps the 2-stage ring six times
DELAY = 20000  # cycles the reader holds each stage before reading it


class Skipper(WarpSpecialized):
    """``mode``: how ``skipper`` passes over odd epochs: "unread" (``consume_unread``),
    "consume" (``with ch.consume(): pass``) or "peek" (a preview, then ``consume_unread``)."""

    def __init__(self, mode: str):
        self.mode = mode
        self.ch = Ring(2, Threads, Threads, release_fence="smem_async", payload=(Int32, (32,)))

    @role(warps=0)
    @cute.jit
    def prod(self, ctx, *, ch: Producer):
        lane = cute.arch.lane_idx()
        for e in cutlass.range(EPOCHS, unroll=1):
            with ch.produce() as h:
                h.data[lane] = e * 100 + lane
        return ch

    @role(warps=1)
    @cute.jit
    def reader(self, ctx, *, ch: Consumer):
        lane = cute.arch.lane_idx()
        for e in cutlass.range(EPOCHS, unroll=1):
            with ch.consume() as h:
                # Read late: a pass that arrived early lets the producer refill this stage.
                start = cute.arch.clock64()
                while cute.arch.clock64() - start < DELAY:
                    pass
                ctx.out[0, e, lane] = h.data[lane]
        return ch

    @role(warps=2)
    @cute.jit
    def skipper(self, ctx, *, ch: Consumer):
        lane = cute.arch.lane_idx()
        for p in cutlass.range(EPOCHS // 2, unroll=1):
            with ch.consume() as h:
                ctx.out[1, 2 * p, lane] = h.data[lane]
            if cutlass.const_expr(self.mode == "consume"):
                with ch.consume():
                    pass
            else:
                if cutlass.const_expr(self.mode == "peek"):
                    ctx.out[2, 2 * p + 1, lane] = ch.peek().view()[lane]
                ch.consume_unread()
        return ch

    @cute.kernel
    def kernel(self, out):
        self.begin_roles()
        self.run_roles(SimpleNamespace(out=out))

    @cute.jit
    def __call__(self, out: cute.Tensor, stream):
        self.kernel.set_name_prefix(f"warpdrive_consume_unread_{self.mode}")
        self.kernel(out).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


def _fake_out():
    return make_fake_compact_tensor(Int32, (3, EPOCHS, 32), assumed_align=4)


@jit_cache
def compile_skipper(op):
    return compile_tvm_ffi(op, _fake_out(), name=f"consume_unread_{op.mode}")


@pytest.mark.parametrize("mode", ["unread", "consume", "peek"])
def test_consume_unread_passes_over_epochs(requires_blackwell, mode):
    out = torch.full((3, EPOCHS, 32), -1, dtype=torch.int32, device="cuda")
    compile_skipper(Skipper(mode))(out)
    expected = torch.arange(EPOCHS, dtype=torch.int32)[:, None] * 100 + torch.arange(32)
    assert torch.equal(out[0].cpu(), expected)
    assert torch.equal(out[1, ::2].cpu(), expected[::2])
    assert (out[1, 1::2] == -1).all()
    if mode == "peek":
        assert torch.equal(out[2, 1::2].cpu(), expected[1::2])


def test_consume_unread_drops_only_its_own_fence(requires_blackwell, tmp_path):
    """The pass emits no release fence; the reading releases keep theirs, and so does a pass
    over a previewed epoch (the preview read it)."""
    disassembler = shutil.which("nvdisasm")
    if disassembler is None:
        pytest.skip("the SASS check requires nvdisasm")
    fences = {}
    for mode in ("unread", "consume", "peek"):
        compiled = cute.compile(Skipper(mode), _fake_out(), cuda.CUstream(0))
        obj = bytes(
            _aot_support.export_module_to_bytes(
                compiled.ir_module, format="o", opt_level=3, enable_pic=True
            )
        )
        cubin = tmp_path / f"skipper_{mode}.cubin"
        cubin.write_bytes(obj[[m.start() for m in re.finditer(b"\x7fELF", obj)][1] :])
        sass = subprocess.check_output([disassembler, str(cubin)], text=True)
        fences[mode] = sass.count("FENCE.VIEW.ASYNC.S")
    assert fences["unread"] == fences["consume"] - 1 == fences["peek"] - 1


def test_consume_unread_is_traced_as_a_wait_and_release():
    op = Skipper("unread")
    op.trace_protocol = True
    compile_skipper.__wrapped__(op)
    ops = [(chan, op_name) for chan, op_name, *_ in op._ws_log.ops["skipper"]]
    assert ops == [("ch", "wait"), ("ch", "release")] * 2
