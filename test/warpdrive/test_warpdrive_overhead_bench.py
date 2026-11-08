import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive.examples import bench_overhead as bench

HANDOFF_CASES = [("pingpong", k, 2 if k == "round_robin" else 1) for k in bench.HANDOFF_KINDS]
STREAM_CASES = [("stream", k, s) for k in bench.STREAM_KINDS for s in (1, 4)]


@pytest.mark.parametrize("impl", ["warpdrive", *bench.WAITS])
@pytest.mark.parametrize("mode, kind, stages", HANDOFF_CASES + STREAM_CASES)
def test_handoff_probes_finish_and_time_their_iterations(
    requires_blackwell, impl, mode, kind, stages
):
    """Each probe runs 7 and 70 iterations to completion (both past every ring's stage count,
    so stages are reused), and more iterations take longer."""
    fn = bench.compiled(impl, mode, kind, stages)
    short, long = bench.run_probe(fn, 7), bench.run_probe(fn, 70)
    assert 0 < short["cycles"] < long["cycles"]
    assert 0 < long["ns"]


def test_empty_roles_twins_record_ordered_phases(requires_blackwell):
    """Every warp of the warpdrive kernel and of its hand-written twins passes entry, the
    prologue, its role and the kernel end, in that order, on a grid larger than one wave."""
    for impl in bench.FIXED_IMPLS:
        fn = bench.compile_empty_roles(impl)
        ctas = 300
        stamps = torch.zeros(
            ctas * bench.FIXED_WARPS * bench.FIXED_SLOTS, device="cuda", dtype=torch.int64
        )
        out = torch.zeros(1, device="cuda", dtype=torch.int32)
        fn(stamps, out, ctas)
        torch.cuda.synchronize()
        st = stamps.view(ctas, bench.FIXED_WARPS, bench.FIXED_SLOTS)
        assert (st > 0).all(), impl
        assert (st[..., 1:] >= st[..., :-1]).all(), impl
        assert out.item() == 0  # the decoded items are never negative
