"""Record producer and consumer warp ranges in static slots.

Run with:
    python examples/profiler/warp_specialized_static.py [--trace PATH]
"""

import argparse
from pathlib import Path

import cutlass
import torch
from cutlass import Float32, Int32, cute

from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.profiler import (
    compact_anchor_init,
    compact_profile_region,
    profile_session,
    summarize_by_tag,
)

TAG_PRODUCER = 0
TAG_CONSUMER = 1
NUM_BLOCKS = 4
THREADS_PER_BLOCK = 64
MAX_EVENTS_PER_UNIT = 4
PRODUCER_WORK_ITERS = 200_000
CONSUMER_WORK_ITERS = 120_000


class WarpSpecializedStaticProfile:
    """Profile producer and consumer warps with static event indices."""

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
    ):
        bidx, _, _ = cute.arch.block_idx()
        bdim, _, _ = cute.arch.block_dim()
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()

        block_base = bidx * bdim
        compact_anchor_init(prof_buf, bidx, max_events_per_unit)

        if warp_idx == 0:
            with compact_profile_region(
                prof_buf,
                max_events_per_unit,
                Int32(TAG_PRODUCER),
                bidx,
                Int32(0),
                target_warp=Int32(0),
            ):
                idx = block_base + lane_idx
                if idx < cute.size(output):
                    # Extra writes make the producer interval measurable.

                    for i in cutlass.range(bidx * 2, unroll=-1):
                        # Vary the work by block.
                        output[idx + i] = Float32(i)
                    for i in cutlass.range(Int32(PRODUCER_WORK_ITERS)):
                        output[idx + i] = Float32(i)

        if warp_idx == 1:
            with compact_profile_region(
                prof_buf,
                max_events_per_unit,
                Int32(TAG_CONSUMER),
                bidx,
                Int32(1),
                target_warp=Int32(1),
            ):
                idx = block_base + Int32(32) + lane_idx
                if idx < cute.size(output):
                    # A shorter consumer interval makes the overlap visible.

                    for i in cutlass.range(bidx * 2, unroll=-1):
                        # Vary the work by block.
                        output[idx + i] = Float32(i)
                    for i in cutlass.range(Int32(CONSUMER_WORK_ITERS)):
                        output[CONSUMER_WORK_ITERS + idx + i] = Float32(i)

    @cute.jit
    def __call__(
        self,
        output: cute.Tensor,
        prof_buf: cute.Tensor,
        max_events_per_unit: cutlass.Int32,
        stream,
    ):
        self.kernel.set_name_prefix("warp_specialized_static")
        self.kernel(output, prof_buf, max_events_per_unit).launch(
            grid=(NUM_BLOCKS, 1, 1), block=(THREADS_PER_BLOCK, 1, 1), stream=stream
        )


@jit_cache
def compile_warp_static(op: WarpSpecializedStaticProfile):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Float32, (cute.sym_int(),)),
        make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),)),
        Int32(1),
        name="warp_specialized_static",
    )


def run():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--trace", type=Path, default=Path("profiler_traces/profiler_warp_static_trace.pftrace")
    )
    trace_path = parser.parse_args().trace
    if not torch.cuda.is_available():
        print("CUDA is not available. This example requires a GPU.")
        return

    device = torch.device("cuda")
    output = torch.zeros(
        NUM_BLOCKS * THREADS_PER_BLOCK * (PRODUCER_WORK_ITERS + CONSUMER_WORK_ITERS),
        dtype=torch.float32,
        device=device,
    )

    trace_path.parent.mkdir(parents=True, exist_ok=True)

    with profile_session(
        max_events_per_unit=MAX_EVENTS_PER_UNIT,
        num_units=(NUM_BLOCKS, "CTA"),
        tag_names=["producer", "consumer"],
        trace_path=str(trace_path),
        device=device,
        compact=True,
        expected_events_per_unit=2,
    ) as session:
        compiled = compile_warp_static(WarpSpecializedStaticProfile())
        compiled(output, session.prof.tensor, session.prof.max_events_per_unit)

    for name, summary in summarize_by_tag(session.ranges).items():
        print(f"{name}: count={summary.count}, median={summary.p50_ns:.0f} ns")

    print(f"Trace: {trace_path}")


if __name__ == "__main__":
    run()
