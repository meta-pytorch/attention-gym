"""Measure kernel runtime with and without compact profiling instrumentation.

Run with:
    python examples/profiler/measure_profiler_overhead.py

The same kernel is compiled twice: ``enabled=False`` specializes the profiling region away, so
the unprofiled build emits no timer reads or stores.
"""

import cutlass
import torch
from cutlass import Int32, cute

from attn_gym.utils import benchmark_cuda_function_in_microseconds
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.profiler import (
    allocate_profile_buffer,
    compact_anchor_init,
    compact_profile_region,
)

TAG_KERNEL = 0
NUM_BLOCKS = 256
THREADS_PER_BLOCK = 128
ELEMENTS = NUM_BLOCKS * THREADS_PER_BLOCK
MAX_EVENTS_PER_UNIT = 1


class ConstantKernel:
    """Write a constant; with profiling, warp 0 of each block records the store as one region."""

    def __init__(self, enable_profiling: bool):
        self.enable_profiling = enable_profiling

    def __cute_cache_key__(self) -> bool:
        return self.enable_profiling

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
        enable_profiling: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        idx = bidx * bdim + tidx

        if cutlass.const_expr(enable_profiling):
            compact_anchor_init(prof_buf, bidx, max_events_per_unit)
        with compact_profile_region(
            prof_buf,
            max_events_per_unit,
            Int32(TAG_KERNEL),
            bidx,
            Int32(0),
            enabled=enable_profiling,
        ):
            if idx < cute.size(output):
                output[idx] = 1.0

    @cute.jit
    def __call__(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
        stream,
    ):
        self.kernel.set_name_prefix("constant_kernel")
        self.kernel(output, prof_buf, max_events_per_unit, self.enable_profiling).launch(
            grid=(NUM_BLOCKS, 1, 1), block=(THREADS_PER_BLOCK, 1, 1), stream=stream
        )


@jit_cache
def compile_constant_kernel(op: ConstantKernel):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),)),
        make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),)),
        Int32(1),
        name="constant_kernel",
    )


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return

    device = torch.device("cuda")
    output = torch.empty(ELEMENTS, dtype=torch.float32, device=device)
    prof_buf = allocate_profile_buffer(
        max_events_per_unit=MAX_EVENTS_PER_UNIT,
        num_units=(NUM_BLOCKS, "Block"),
        device=device,
        compact=True,
    )

    times_us = {}
    for enabled in (False, True):
        compiled = compile_constant_kernel(ConstantKernel(enable_profiling=enabled))
        times_us[enabled] = benchmark_cuda_function_in_microseconds(
            compiled, output, prof_buf.tensor, prof_buf.max_events_per_unit
        )

    overhead_avg_pct = 100.0 * (times_us[True] - times_us[False]) / times_us[False]
    print(f"Unprofiled avg: {times_us[False]:.3f} us")
    print(f"Profiled avg:   {times_us[True]:.3f} us")
    print(f"Overhead avg:   {overhead_avg_pct:.2f}%")


if __name__ == "__main__":
    main()
