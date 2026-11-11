"""The profiler's standalone examples compile, compute the right values and record ranges."""

import pytest
import torch

pytest.importorskip("cutlass.cute")
pytest.importorskip("perfetto")

from attn_gym.warpdrive.profiler import allocate_profile_buffer
from examples.profiler import (
    cute_profiler_buffering as buffering,
)
from examples.profiler import (
    measure_profiler_overhead as overhead,
)
from examples.profiler import (
    warp_specialized_static as static,
)


def test_buffering_computes_and_records_both_stage_counts(requires_blackwell, tmp_path):
    """Both stage counts compute the FMA chain and record every range.

    How much the loads overlap the consumer is a measurement: the example prints it.
    """
    numel = buffering.NUM_BLOCKS * buffering.TILES_PER_CTA * buffering.TILE_ELEMENTS
    src = torch.linspace(-1.0, 1.0, numel, device="cuda")
    expected = src.clone()
    for _ in range(buffering.COMPUTE_ITERS):
        expected = expected * 1.0001 + 0.0001
    tolerance = buffering.COMPUTE_ITERS * torch.finfo(torch.float32).eps
    for stages in (1, 2):
        dst = torch.empty_like(src)
        buffering.WarpSpecializedTransform(stages).compile(src, dst, None)()
        torch.testing.assert_close(dst, expected, rtol=tolerance, atol=tolerance)
        trace = tmp_path / f"{stages}.pftrace"
        summary = buffering.capture_profile(stages, src, dst, trace)
        assert trace.stat().st_size > 0
        assert set(buffering.PROFILE_TAGS) < set(summary)
        assert all(value >= 0 for value in summary.values())


@pytest.mark.parametrize("enabled", [False, True])
def test_overhead_kernel_records_only_when_enabled(requires_blackwell, enabled):
    """The unprofiled specialization writes no records; the profiled one fills every slot."""
    output = torch.zeros(overhead.ELEMENTS, device="cuda")
    prof = allocate_profile_buffer(1, (overhead.NUM_BLOCKS, "Block"), compact=True)
    compiled = overhead.compile_constant_kernel(overhead.ConstantKernel(enable_profiling=enabled))
    compiled(output, prof.tensor, prof.max_events_per_unit)
    torch.cuda.synchronize()
    assert bool((output == 1).all())
    assert bool((prof.tensor != 0).all()) if enabled else not prof.tensor.any()


def test_static_warp_roles_record_one_range_each(requires_blackwell, tmp_path, monkeypatch):
    """Each CTA records one range per role, and the run writes a trace."""
    trace = tmp_path / "static.pftrace"
    monkeypatch.setattr("sys.argv", ["warp_specialized_static.py", "--trace", str(trace)])
    recorded = []
    monkeypatch.setattr(static, "summarize_by_tag", lambda ranges: recorded.extend(ranges) or {})
    static.run()
    assert trace.stat().st_size > 0
    assert sorted((r.name, r.cta) for r in recorded) == sorted(
        (tag, (cta,)) for tag in ("producer", "consumer") for cta in range(static.NUM_BLOCKS)
    )
