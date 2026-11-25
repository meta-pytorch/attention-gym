"""Indexer benchmark CLI, replay stage accounting, and logical traffic scenarios."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from attn_gym.testing.indexer import make_paged_indexer_test_inputs
from benchmarks.sparse import indexer_benchmark, paged_indexer_benchmark
from benchmarks.sparse.paged_indexer_benchmark import (
    dense_selection,
    model_scenario,
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
    stages = summarize_stage_events(events * 3, 3, ["_score_tma", selector] * 2)
    for stage, duration in [("scoring", 12), ("selection", 24)]:
        assert stages[stage]["profiled_mean_us"] == duration
        assert stages[stage]["launches_per_call"] == 2


@pytest.mark.parametrize(
    "graph,observed,error",
    [
        (["_score", "_select"] * 32, ["_score", "_select"] * 2, "count mismatch"),
        (["_score", "_select"], ["_score", "_select", "_select"], "count mismatch"),
        (["new_kernel"], ["new_kernel"] * 2, "Unrecognized"),
    ],
)
def test_paged_profile_rejects_missing_or_unattributed_launches(graph, observed, error):
    events = [
        SimpleNamespace(name=n, device_type=torch.autograd.DeviceType.CUDA, device_time_total=1)
        for n in observed
    ]
    with pytest.raises(RuntimeError, match=error):
        summarize_stage_events(events, 2, graph)


@pytest.mark.parametrize(
    "live,capacity,zero,cute,score_io,selector_read",
    [
        # BF16: 4*128 Q + 1024*128 K + 4 weights. INT32: 16 page IDs + 2 visibility inputs.
        (1024, 1024, False, False, (263248, 4096 + 8), 4096 + 4),
        (1024, 1024, False, True, (263248, 4096 + 8), 4096 + 8),
        (1024, 2**22, True, False, (16, 8), 4),  # Weights read, no scores written.
        (256, 2**22, False, True, (8, 8), 8),  # Live pool fits, but scoring writes bounds.
        (256, 256, False, False, None, 8),  # No bounds buffer; selector computes visibility.
    ],
)
def test_paged_logical_stage_io(live, capacity, zero, cute, score_io, selector_read):
    pytest.importorskip("transformer_nuggets.roofline")
    work = stage_work(
        1, live, 4, 128, 512, 64, capacity=capacity, zero_weights=zero, cute_selection=cute
    )
    assert work["selection"].logical_flops is None
    assert work["selection"].write_bytes == 512 * 4
    assert work["selection"].read_bytes == selector_read
    if score_io is None:
        assert "scoring" not in work
    else:
        assert (work["scoring"].read_bytes, work["scoring"].write_bytes) == score_io
        assert (work["scoring"].logical_flops == 0) == (zero or live <= 512)


def test_paged_hbm_and_launch_scenario_is_not_warm_replay_latency():
    roofline = pytest.importorskip("transformer_nuggets.roofline")
    work = stage_work(1, 1024, 4, 128, 512, 64, capacity=1024, cute_selection=False)
    stages = {
        "scoring": {"profiled_mean_us": 1, "launches_per_call": 1},
        "selection": {"profiled_mean_us": 1, "launches_per_call": 1},
        "initialization": {"profiled_mean_us": 1, "launches_per_call": 3},
    }
    # At 1 GB/s, charging every logical byte to HBM exceeds the warm profile time.
    result = model_scenario(work, stages, roofline.RooflineSpec("scenario", 1000, 1, 10))
    scoring = result["stages"]["scoring"]
    selection = result["stages"]["selection"]
    assert scoring["stage_estimate_us"] == work["scoring"].logical_bytes / 1000
    assert selection["stage_estimate_us"] == selection["assumed_launch_cost_us"] == 10
    assert result["modeled_serial_estimate_us"] == scoring["stage_estimate_us"] + 10
    assert selection["useful_compute_at_peak_us"] is None
    assert result["unmodeled_profiled_stages"] == ["initialization"]
    assert result["full_roofline_percent"] is None


@pytest.mark.parametrize("zero,expected", [(True, [0, 1, 2, 3]), (False, [0, 2, 7, 63])])
def test_paged_dense_deterministic_ties_and_ascending_ids(
    deterministic_algorithms, zero, expected
):
    args = make_paged_indexer_test_inputs((1,), (256,), device="cpu", heads=1, head_dim=1)
    args[0].fill_(1)
    args[1].zero_()
    args[1][0, [63, 7, 2], 0] = torch.tensor([3, 2, 1], dtype=args[1].dtype)
    args[2].fill_(0 if zero else 1)
    assert dense_selection(args, 4, 64).tolist() == [expected]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip, reason="requires CUDA")
def test_paged_deterministic_replay_profiles_fills(
    deterministic_algorithms, monkeypatch, tmp_path
):
    pytest.importorskip("transformer_nuggets.roofline")
    pytest.importorskip("cuda.bindings.driver")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("requires SM80+")
    output = tmp_path / "case.json"
    monkeypatch.setattr(torch.utils.deterministic, "fill_uninitialized_memory", True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--batch=1",
            "--candidates=64",
            "--topk=4",
            "--backend=triton",
            "--deterministic",
            "--weights=zero",
            "--warmup=1",
            "--rep=2",
            "--profile-rep=2",
            "--peak-memory-gbps=8000",
            "--peak-compute-tflops=2250",
            f"--output={output}",
        ],
    )
    paged_indexer_benchmark.main()
    (result,) = json.loads(output.read_text())["results"]
    stages = result["profile"]
    assert stages["initialization"]["launches_per_call"] > 0
    assert stages["initialization"]["profiled_mean_us"] > 0
    total = sum(s["profiled_mean_us"] for s in stages.values())
    assert result["topk_share_of_profiled_kernel_time_percent"] == pytest.approx(
        100 * stages["selection"]["profiled_mean_us"] / total
    )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip, reason="requires CUDA")
def test_paged_ncu_rejects_invalid_ids_before_profiling(deterministic_algorithms, monkeypatch):
    pytest.importorskip("transformer_nuggets.roofline")

    def invalid_selection(query, *args, topk, **kwargs):
        return query.new_full((query.size(0), topk), -1, dtype=torch.int32)

    monkeypatch.setattr(paged_indexer_benchmark, "paged_lightning_indexer", invalid_selection)
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench", "--ncu", "--deterministic", "--batch=1", "--candidates=64", "--topk=4"],
    )
    with pytest.raises(AssertionError):
        paged_indexer_benchmark.main()
