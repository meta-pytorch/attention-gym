"""Indexer benchmark CLI, stage accounting, and partial roofline contracts."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from benchmarks.sparse import indexer_benchmark
from benchmarks.sparse.paged_indexer_benchmark import (
    partial_bounds,
    stage_work,
    summarize_stage_events,
)


def test_benchmark_rejects_explicit_auto(monkeypatch):
    """Automatic dispatch is requested by omission, not an explicit backend value."""
    monkeypatch.setattr(sys, "argv", ["indexer_benchmark.py", "--backend", "auto"])
    with pytest.raises(SystemExit) as exc:
        indexer_benchmark.parse_args()
    assert exc.value.code == 2


def test_benchmark_omitted_backend_uses_device_selection(monkeypatch):
    """The default CLI leaves the backend override unset."""
    monkeypatch.setattr(sys, "argv", ["indexer_benchmark.py"])
    assert indexer_benchmark.parse_args().backend == [None]


@pytest.mark.parametrize("dependency", [None, ModuleType("transformer_nuggets.utils.benchmark")])
def test_benchmark_help_without_compatible_nuggets(monkeypatch, capsys, dependency):
    """Help remains available when Nuggets is absent or lacks graph sample statistics."""
    monkeypatch.setitem(sys.modules, "transformer_nuggets.utils.benchmark", dependency)
    monkeypatch.setattr(sys, "argv", ["indexer_benchmark.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        indexer_benchmark.main()
    assert exc.value.code == 0
    assert "uv pip install" in capsys.readouterr().out


@pytest.mark.parametrize("dependency", [None, ModuleType("transformer_nuggets.utils.benchmark")])
def test_benchmark_reports_missing_or_old_nuggets(monkeypatch, dependency):
    """An unavailable statistics helper reports a compatible installation before GPU setup."""
    monkeypatch.setitem(sys.modules, "transformer_nuggets.utils.benchmark", dependency)
    monkeypatch.setattr(sys, "argv", ["indexer_benchmark.py"])
    with pytest.raises(SystemExit, match="Install the compatible revision"):
        indexer_benchmark.main()


@pytest.mark.parametrize("selector", ["indexer_topk_k512", "_select"])
def test_paged_stage_profile(selector):
    events = [
        SimpleNamespace(
            name=name,
            device_type=torch.autograd.DeviceType.CUDA,
            device_time_total=duration,
        )
        for name, duration in [
            ("_score_tma", 5),
            (selector, 10),
            ("_score_tma", 7),
            (selector, 14),
        ]
    ]
    events.append(
        SimpleNamespace(
            name="attn_gym::_paged_indexer",
            device_type=torch.autograd.DeviceType.CPU,
            device_time_total=1000,
        )
    )
    stages = summarize_stage_events(events * 3, iterations=3)
    for stage, duration in [("scoring", 12), ("selection", 24)]:
        assert stages[stage]["profiled_mean_us"] == duration
        assert stages[stage]["launches_per_call"] == 2


@pytest.mark.parametrize("iterations", [1, 2])
def test_paged_profile_rejects_incomplete_coverage(iterations):
    events = [
        SimpleNamespace(
            name="new_selection_kernel",
            device_type=torch.autograd.DeviceType.CUDA,
            device_time_total=10,
        )
    ]
    if iterations == 2:
        with pytest.raises(RuntimeError, match="Incomplete profiler"):
            summarize_stage_events(events, iterations=iterations)
    else:
        roofline = pytest.importorskip("transformer_nuggets.roofline")
        stages = summarize_stage_events(events, iterations=iterations)
        assert stages["other"]["profiled_mean_us"] == 10
        with pytest.raises(RuntimeError, match="coverage mismatch"):
            partial_bounds(
                stage_work(1, 512, 4, 128, 512, 64),
                stages,
                roofline.RooflineSpec("B200", 2250, 8000, 1),
            )


@pytest.mark.parametrize("candidates,zero_weights", [(32768, False), (32768, True), (256, False)])
def test_paged_partial_roofline(candidates, zero_weights):
    roofline = pytest.importorskip("transformer_nuggets.roofline")
    work = stage_work(64, candidates, 4, 128, 512, 64, capacity=2**22, zero_weights=zero_weights)
    assert work["selection"].logical_flops is None
    assert work["selection"].write_bytes == 64 * 512 * 4
    if zero_weights or candidates <= 512:
        assert work["scoring"].logical_flops == 0
        assert work["scoring"].read_bytes < 4096
        assert work["selection"].read_bytes < 4096
    else:
        assert work["scoring"].write_bytes == work["selection"].read_bytes
    stages = {
        "scoring": {"profiled_mean_us": 90, "launches_per_call": 1},
        "selection": {"profiled_mean_us": 70, "launches_per_call": 1},
    }
    result = partial_bounds(work, stages, roofline.RooflineSpec("B200", 2250, 8000, 1))
    assert result["full_roofline_percent"] is None
    assert result["partial_serial_floor_us"] > result["stages"]["scoring"]["partial_floor_us"]
    assert result["stages"]["selection"]["compute_floor_us"] is None
    for stage in result["stages"].values():
        assert "achieved_roofline_percent" not in stage
