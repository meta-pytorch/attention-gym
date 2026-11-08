"""Keyword-only role dependencies use the same name/annotation binding and state returns."""

from dataclasses import dataclass

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Barriers,
    Consumer,
    Instance,
    Producer,
    ProtocolError,
    RoleConfig,
    Slots,
    Threads,
    WarpSpecialized,
    role,
)


@dataclass(frozen=True)
class CopyParams:
    out: cute.Tensor
    steps: cutlass.Int32


class KeywordCopy(WarpSpecialized):
    def __init__(self):
        self.roles = {"load": RoleConfig([range(1), range(1, 2)])}
        self.ch = Slots(2, Threads, Threads, payload=(cutlass.Int32, (32,)))

    @role(warps=0)
    @cute.jit
    def load(self, params: CopyParams, *, stage: Instance, ch: Producer[Instance]):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(params.steps, unroll=1):
            with ch.produce(stage) as h:
                h.data[lane] = (stage + 1) * (i + 1) + lane
        return ch

    @role(warps=2)
    @cute.jit
    def consume(self, ch: Consumer, *, params: CopyParams):
        # Deliberately mix positional state with keyword-only params: binding follows the
        # declared signature, not a mandatory position for either kind of argument.
        lane = cute.arch.lane_idx()
        for _ in cutlass.range(params.steps, unroll=1):
            for stage in cutlass.range_constexpr(2):
                with ch.consume(stage) as h:
                    params.out[stage * 32 + lane] += h.data[lane]
        return ch

    @cute.kernel
    def kernel(self, out: cute.Tensor, steps: cutlass.Int32):
        self.begin_roles()
        self.run_roles(CopyParams(out, steps))

    @cute.jit
    def __call__(self, out: cute.Tensor, steps: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_keyword_copy")
        self.kernel(out, steps).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


class KeywordRaw(WarpSpecialized):
    def __init__(self):
        self.scratch = Barriers(1, counts=1)

    @role(warps=0, raw=True)
    @cute.jit
    def worker(self, params: CopyParams, *, scratch: Barriers):
        lane = cute.arch.lane_idx()
        params.out[lane] = params.steps + lane

    @cute.kernel
    def kernel(self, out: cute.Tensor, steps: cutlass.Int32):
        self.begin_roles()
        self.run_roles(CopyParams(out, steps))

    @cute.jit
    def __call__(self, out: cute.Tensor, steps: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix("warpdrive_keyword_raw")
        self.kernel(out, steps).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_copy(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4),
        cutlass.Int32(1),
        name="warpdrive_keyword_roles",
    )


@pytest.mark.parametrize("n", [1, 7])
def test_keyword_dependencies_and_params(requires_blackwell, n):
    out = torch.zeros(64, device="cuda", dtype=torch.int32)
    compile_copy(KeywordCopy())(out, n)
    lane = torch.arange(32, device="cuda", dtype=torch.int32)
    tri = n * (n + 1) // 2
    torch.testing.assert_close(out[:32], tri + n * lane)
    torch.testing.assert_close(out[32:], 2 * tri + n * lane)


def test_raw_keyword_dependency(requires_blackwell):
    out = torch.empty(32, device="cuda", dtype=torch.int32)
    compile_copy(KeywordRaw())(out, 3)
    torch.testing.assert_close(out, torch.arange(32, device="cuda", dtype=torch.int32) + 3)


def test_variadic_role_parameters_rejected():
    class VarArgs(WarpSpecialized):
        @role(warps=0)
        @cute.jit
        def worker(self, *args):
            pass

    class VarKwargs(WarpSpecialized):
        @role(warps=0)
        @cute.jit
        def worker(self, **kwargs):
            pass

    for cls in (VarArgs, VarKwargs):
        with pytest.raises(ProtocolError, match="variadic parameter .* cannot be injected"):
            cls().describe()
