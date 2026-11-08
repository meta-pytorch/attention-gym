"""Use intra-kernel traces to diagnose and fix a serialized warp pipeline.

Warp 0 loads one tile from global memory into shared memory. Warp 1 waits for that
tile, applies a dependent FMA chain, and stores it. With one shared-memory stage,
the roles run in lockstep. With two stages, the producer can load tile ``i + 1``
while the consumer processes tile ``i``.

Run with:
    python examples/profiler/cute_profiler_buffering.py [--trace-dir DIR]

The script checks correctness, reports unprofiled latency, summarizes the sampled
CTA's compact shared-memory records, joins them to the kernel's pipeline plan (stage reuse
as a ``num_stages``-tile dependency), and writes one Perfetto trace per variant with the
plan's dependency arrows and the stage ring's occupancy.
"""

import argparse
import statistics
from collections.abc import Callable
from enum import IntEnum
from pathlib import Path

import cutlass
import torch
from cutlass import Float32, Int32, cute, pipeline

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.compat import SmemAllocator
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.profiler import (
    compact_flush_smem_to_gmem,
    compact_prepare_smem,
    compact_profile_region,
    overlap_between,
    profile_session,
    summarize_by_tag,
)
from attn_gym.profiler.pipeline import (
    Dependency,
    PipelinePlan,
    Region,
    Resource,
    Role,
    analyze_pipeline,
    join,
    write_pipeline_perfetto,
)
from attn_gym.utils import benchmark_cuda_function_in_microseconds

NUM_BLOCKS = 32
TILES_PER_CTA = 256
VALUES_PER_LANE = 4
COMPUTE_ITERS = 64
THREADS = 64
WARP_SIZE = 32
PRODUCER_WARP = 0
CONSUMER_WARP = 1
PROFILED_CTA = 0
PROFILE_UNIT = 0
EVENTS_PER_TILE = 4
TILE_ELEMENTS = WARP_SIZE * VALUES_PER_LANE
MAX_PROFILE_EVENTS = EVENTS_PER_TILE * TILES_PER_CTA


class ProfileTag(IntEnum):
    PRODUCER_ACQUIRE = 0
    PRODUCER_LOAD = 1
    CONSUMER_WAIT = 2
    CONSUMER_COMPUTE_STORE = 3


PROFILE_TAGS = tuple(tag.name.lower() for tag in ProfileTag)


class WarpSpecializedTransform:
    """Transform tiles with distinct producer and consumer warps."""

    def __init__(self, num_stages: int, enable_profiling: bool = False):
        if num_stages not in (1, 2):
            raise ValueError("num_stages must be 1 or 2")
        self.num_stages = num_stages
        self.enable_profiling = enable_profiling

    def shared_storage(self):
        """Return shared storage without profiling state in the production specialization."""
        if self.enable_profiling:

            @cute.struct
            class SharedStorage:
                barriers: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages], 8
                ]
                tiles: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_stages * TILE_ELEMENTS],
                    128,
                ]
                profile: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Int64, 1 + MAX_PROFILE_EVENTS], 8
                ]

        else:

            @cute.struct
            class SharedStorage:
                barriers: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages], 8
                ]
                tiles: cute.struct.Align[
                    cute.struct.MemRange[cutlass.Float32, self.num_stages * TILE_ELEMENTS],
                    128,
                ]

        return SharedStorage

    def __cute_cache_key__(self) -> tuple[int, bool]:
        return self.num_stages, self.enable_profiling

    def profile_scope(
        self,
        profile_records,
        tag: ProfileTag,
        event_idx,
        target_warp: int,
        record,
        enable_profiling: cutlass.Constexpr,
    ):
        """Create one compact shared-memory profiling region."""
        return compact_profile_region(
            profile_records,
            Int32(MAX_PROFILE_EVENTS),
            Int32(tag),
            Int32(PROFILE_UNIT),
            event_idx,
            target_warp=Int32(target_warp),
            enabled=enable_profiling,
            record=record,
            smem=True,
        )

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        profile_gmem: cute.Tensor,
        shared_storage: cutlass.Constexpr,
        enable_profiling: cutlass.Constexpr,
    ):
        smem = SmemAllocator()
        storage = smem.allocate(shared_storage)
        staged = storage.tiles.get_tensor(
            cute.make_ordered_layout(
                (self.num_stages, VALUES_PER_LANE, WARP_SIZE), order=(2, 1, 0)
            )
        )
        barriers = storage.barriers.data_ptr()

        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        record = block_idx == PROFILED_CTA

        if cutlass.const_expr(enable_profiling):
            profile_records = storage.profile.get_tensor(
                cute.make_layout((1 + MAX_PROFILE_EVENTS,))
            )
            compact_prepare_smem(profile_records, Int32(MAX_PROFILE_EVENTS), record)
        else:
            profile_records = None

        producer, consumer = pipeline.PipelineAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, WARP_SIZE),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, WARP_SIZE),
            barrier_storage=barriers,
        ).make_participants()

        block_base = block_idx * TILES_PER_CTA * TILE_ELEMENTS

        if warp_idx == PRODUCER_WARP:
            for tile_idx in cutlass.range(TILES_PER_CTA, unroll=1):
                event_base = tile_idx * Int32(EVENTS_PER_TILE)
                with self.profile_scope(
                    profile_records,
                    ProfileTag.PRODUCER_ACQUIRE,
                    event_base,
                    PRODUCER_WARP,
                    record,
                    enable_profiling,
                ):
                    stage = producer.acquire_and_advance()
                with self.profile_scope(
                    profile_records,
                    ProfileTag.PRODUCER_LOAD,
                    event_base + Int32(1),
                    PRODUCER_WARP,
                    record,
                    enable_profiling,
                ):
                    tile_base = block_base + tile_idx * TILE_ELEMENTS
                    for value_idx in cutlass.range_constexpr(VALUES_PER_LANE):
                        staged[(stage.index, value_idx, lane_idx)] = src[
                            tile_base + value_idx * WARP_SIZE + lane_idx
                        ]
                    stage.commit()
            producer.tail()

        if warp_idx == CONSUMER_WARP:
            for tile_idx in cutlass.range(TILES_PER_CTA, unroll=1):
                event_base = tile_idx * Int32(EVENTS_PER_TILE)
                with self.profile_scope(
                    profile_records,
                    ProfileTag.CONSUMER_WAIT,
                    event_base + Int32(2),
                    CONSUMER_WARP,
                    record,
                    enable_profiling,
                ):
                    stage = consumer.wait_and_advance()
                with self.profile_scope(
                    profile_records,
                    ProfileTag.CONSUMER_COMPUTE_STORE,
                    event_base + Int32(3),
                    CONSUMER_WARP,
                    record,
                    enable_profiling,
                ):
                    tile_base = block_base + tile_idx * TILE_ELEMENTS
                    for value_idx in cutlass.range_constexpr(VALUES_PER_LANE):
                        value = staged[(stage.index, value_idx, lane_idx)]
                        for _ in cutlass.range_constexpr(COMPUTE_ITERS):
                            value = value * Float32(1.0001) + Float32(0.0001)
                        dst[tile_base + value_idx * WARP_SIZE + lane_idx] = value
                    stage.release()

        if cutlass.const_expr(enable_profiling):
            assert profile_records is not None
            compact_flush_smem_to_gmem(
                profile_records,
                profile_gmem,
                Int32(PROFILE_UNIT),
                Int32(MAX_PROFILE_EVENTS),
                record,
            )

    @cute.jit
    def __call__(self, src: cute.Tensor, dst: cute.Tensor, profile_gmem: cute.Tensor, stream):
        """Launch; the unprofiled build never reads ``profile_gmem``."""
        self.kernel.set_name_prefix(f"warp_transform_{self.num_stages}stage")
        self.kernel(
            src,
            dst,
            profile_gmem,
            self.shared_storage(),
            self.enable_profiling,
        ).launch(grid=(NUM_BLOCKS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    def compile(self, src: torch.Tensor, dst: torch.Tensor, profile_gmem: torch.Tensor | None):
        """Compile (or load) the specialization and return a no-allocation launch callable.
        Without profiling, ``src`` stands in for the profile buffer the kernel does not read."""
        compiled = compile_transform(self)
        profile = profile_gmem if profile_gmem is not None else src
        return lambda: compiled(src, dst, profile)


@jit_cache
def compile_transform(op: WarpSpecializedTransform):
    """One compile per stage count and profiling mode; the element count is a runtime value."""
    numel = cute.sym_int(divisibility=TILE_ELEMENTS)
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(Float32, (numel,)),
        make_fake_compact_tensor(Float32, (numel,)),
        make_fake_compact_tensor(
            cutlass.Int64 if op.enable_profiling else Float32, (cute.sym_int(),)
        ),
        name="warp_transform",
    )


def buffering_plan(num_stages: int) -> PipelinePlan:
    """The kernel's pipeline: the consumer waits for a loaded tile, and the producer acquires
    a stage the consumer released ``num_stages`` tiles earlier."""
    plan = PipelinePlan(f"{num_stages}-stage CTA {PROFILED_CTA}", iteration_name="tile")
    plan.add_role(Role("producer", "Producer warp", PRODUCER_WARP, PRODUCER_WARP + 1))
    plan.add_role(Role("consumer", "Consumer warp", CONSUMER_WARP, CONSUMER_WARP + 1))
    plan.add_resource(Resource("tile", "tile stages", num_stages))
    for tag in ProfileTag:
        role = "producer" if tag <= ProfileTag.PRODUCER_LOAD else "consumer"
        plan.add_region(Region(PROFILE_TAGS[tag], role, PROFILE_TAGS[tag], tag % 2, 1))
    plan.add_dependency(Dependency("producer_load", "consumer_wait", "tile"))
    plan.add_dependency(
        Dependency(
            "consumer_compute_store", "producer_acquire", "tile", distance=num_stages, kind="reuse"
        )
    )
    return plan


def benchmark_unprofiled(
    launches: dict[int, Callable[[], object]],
) -> dict[int, list[float]]:
    """Collect interleaved timing rounds to reduce order and clock bias."""
    samples = {stages: [] for stages in launches}
    for round_idx in range(5):
        order = (1, 2) if round_idx % 2 == 0 else (2, 1)
        for stages in order:
            samples[stages].append(benchmark_cuda_function_in_microseconds(launches[stages]))
    return samples


def capture_profile(
    num_stages: int, src: torch.Tensor, dst: torch.Tensor, trace_path: Path
) -> dict[str, float]:
    """Capture one warmed profiled launch, join it to the kernel's pipeline plan, write the
    trace, and return summary statistics."""
    op = WarpSpecializedTransform(num_stages=num_stages, enable_profiling=True)
    with profile_session(
        max_events_per_unit=MAX_PROFILE_EVENTS,
        num_units=(1, "CTA"),
        tag_names=list(PROFILE_TAGS),
        compact=True,
        expected_events_per_unit=MAX_PROFILE_EVENTS,
    ) as session:
        launch = op.compile(src, dst, session.prof.tensor)
        for _ in range(3):
            launch()
        torch.cuda.synchronize()
        session.prof.reset()
        launch()

    timeline = buffering_plan(num_stages).schedule(iterations=TILES_PER_CTA)
    measured = join(session.ranges, timeline)
    analysis = analyze_pipeline(timeline, measured)
    write_pipeline_perfetto(trace_path, timeline, measured, analysis=analysis)

    summaries = summarize_by_tag(session.ranges)
    overlap = overlap_between(session.ranges, "producer_load", "consumer_compute_store")
    (ring,) = analysis.rings
    return {
        **{name: summaries[name].p50_ns for name in PROFILE_TAGS},
        "producer_load_overlap_pct": 100.0 * overlap.left_fraction,
        "stage_reuse_wait_ns": ring.reuse_wait_ns,
    }


def main() -> None:
    """Compare lockstep and double-buffered kernels, then write both traces."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace-dir", type=Path, default=Path("profiler_traces"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    numel = NUM_BLOCKS * TILES_PER_CTA * TILE_ELEMENTS
    src = torch.linspace(-1.0, 1.0, numel, device="cuda", dtype=torch.float32)
    outputs = {stages: torch.empty_like(src) for stages in (1, 2)}
    launches = {}

    for stages in (1, 2):
        op = WarpSpecializedTransform(num_stages=stages)
        launches[stages] = op.compile(src, outputs[stages], None)
        launches[stages]()
    torch.cuda.synchronize()
    torch.testing.assert_close(outputs[1], outputs[2], rtol=0, atol=0)
    expected = src
    for _ in range(COMPUTE_ITERS):
        expected = expected * 1.0001 + 0.0001
    rounding_tolerance = COMPUTE_ITERS * torch.finfo(torch.float32).eps
    torch.testing.assert_close(
        outputs[1], expected, rtol=rounding_tolerance, atol=rounding_tolerance
    )

    timing_samples = benchmark_unprofiled(launches)
    timings = {stages: statistics.median(samples) for stages, samples in timing_samples.items()}

    trace_dir = args.trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    summaries = {
        stages: capture_profile(
            stages,
            src,
            outputs[stages],
            trace_dir / f"buffering_{stages}_stage.pftrace",
        )
        for stages in (1, 2)
    }

    print("production timing (fixed-pointer, warm-buffer, five interleaved rounds)")
    for stages in (1, 2):
        samples = timing_samples[stages]
        print(
            f"  stages={stages}: median={timings[stages]:.3f} us "
            f"range=[{min(samples):.3f}, {max(samples):.3f}]"
        )
    print(f"  speedup: {timings[1] / timings[2]:.3f}x")
    print("profile summary for sampled CTA 0 (median ns across tiles)")
    for stages in (1, 2):
        values = summaries[stages]
        print(
            f"  stages={stages}: acquire={values['producer_acquire']:.0f}, "
            f"load={values['producer_load']:.0f}, wait={values['consumer_wait']:.0f}, "
            f"compute_store={values['consumer_compute_store']:.0f}, "
            f"load_overlap={values['producer_load_overlap_pct']:.1f}%, "
            f"stage_reuse_wait={values['stage_reuse_wait_ns']:.0f}"
        )
        print(f"    trace: {trace_dir / f'buffering_{stages}_stage.pftrace'}")


if __name__ == "__main__":
    main()
