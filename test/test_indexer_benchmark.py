"""Indexer benchmark CLI, replay stage accounting, and logical traffic scenarios."""

import json
import sys
from argparse import Namespace
from contextlib import contextmanager
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


def test_benchmark_rejects_triton_gvr2(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["indexer_benchmark.py", "--backend", "triton", "--selector", "gvr2"]
    )
    with pytest.raises(SystemExit) as exc:
        indexer_benchmark.parse_args()
    assert exc.value.code == 2
    assert "GVR2 selector requires the CuTe backend" in capsys.readouterr().err


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
    help_text = capsys.readouterr().out
    assert "--sequence-length" in help_text
    assert "uv pip install" not in help_text


@pytest.mark.parametrize("dependency", [None, ModuleType("transformer_nuggets.utils.benchmark")])
def test_benchmark_without_compatible_nuggets_reaches_cuda_setup(monkeypatch, dependency):
    """Missing or old Nuggets must not block execution before the CUDA requirement."""
    monkeypatch.setitem(sys.modules, "transformer_nuggets.utils.benchmark", dependency)
    monkeypatch.setattr(sys, "argv", ["indexer_benchmark.py"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        indexer_benchmark.main()


def test_benchmark_prints_median_ms_for_supported_routes(monkeypatch, capsys):
    samples = [1000.0, 2000.0, 9000.0]
    monkeypatch.setattr(
        sys,
        "argv",
        ["indexer_benchmark.py", "--backend", "cute", "triton", "--selector", "default", "gvr2"],
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "mock CUDA GPU")
    monkeypatch.setattr(indexer_benchmark, "make_inputs", lambda args: (None,) * 5)
    monkeypatch.setattr(indexer_benchmark, "benchmark_graph", lambda *args, **kwargs: samples)
    indexer_benchmark.main()
    output = capsys.readouterr().out
    assert "[fused/cute/gvr2] forward: 2.000 ms" in output
    assert "[fused/triton/default] forward: 2.000 ms" in output
    assert "[fused/triton/gvr2] skipped" in output
    assert "[fused/triton/gvr2] forward" not in output


@pytest.mark.parametrize(
    "tokens,ratio,causal",
    [(17, 1, False), (17, 3, True), (131072, 4, True)],
)
def test_compressed_useful_flops(tokens, ratio, causal):
    args = Namespace(
        batch=2,
        sequence_length=tokens,
        heads=64,
        head_dim=128,
        compress_ratio=ratio,
        causal=causal,
    )
    pairs = (
        sum((row + 1) // ratio for row in range(tokens)) if causal else tokens * (tokens // ratio)
    )
    assert (
        indexer_benchmark.useful_flops(args) == 2 * args.batch * args.heads * args.head_dim * pairs
    )


@pytest.mark.parametrize("warmup", [0, 2])
def test_timing_events_are_inside_capture(monkeypatch, warmup):
    """CPU-only protocol gate: replay/synchronization must not sit between host event records."""
    state = {"capturing": False, "released": False, "calls": 0, "replays": 0, "records": 0}

    class Output:
        def __init__(self):
            self.captured = state["capturing"]

        def __del__(self):
            if self.captured:
                state["released"] = True

    class Event:
        def __init__(self, *, enable_timing, external):
            assert enable_timing and external

        def record(self):
            assert state["capturing"]
            state["records"] += 1

        def elapsed_time(self, other):
            assert not state["capturing"]
            return 0.25

    class Graph:
        def replay(self):
            assert not state["capturing"] and not state["released"]
            state["replays"] += 1

    @contextmanager
    def capture(graph):
        state["capturing"] = True
        yield
        state["capturing"] = False

    def work():
        state["calls"] += 1
        return Output()

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    assert indexer_benchmark.benchmark_graph(work, warmup=warmup, samples=3) == [250.0] * 3
    assert {key: value for key, value in state.items() if key != "released"} == {
        "capturing": False,
        "calls": warmup + 1,
        "replays": warmup + 3,
        "records": 2,
    }


@pytest.mark.parametrize("backends", [["triton"], ["cute", "triton"]])
def test_mxfp8_benchmark_rejects_triton_before_running(monkeypatch, capsys, backends):
    monkeypatch.setattr(sys, "argv", ["bench", "--dtype", "mxfp8", "--backend", *backends])
    with pytest.raises(SystemExit) as exc:
        indexer_benchmark.parse_args()
    assert exc.value.code == 2
    assert "fused MXFP8 scoring requires CuTe" in capsys.readouterr().err


def test_mxfp8_benchmark_defaults_to_supported_shape(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "--dtype", "mxfp8"])
    args = indexer_benchmark.parse_args()
    assert args.heads in (32, 64) and args.head_dim == 128


def test_mxfp8_benchmark_shape_depends_on_implementation(monkeypatch, capsys):
    argv = ["bench", "--dtype", "mxfp8", "--heads", "3", "--head-dim", "64"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        indexer_benchmark.parse_args()
    assert exc.value.code == 2
    assert "--heads 32 or 64 and --head-dim 128" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*argv, "--impl", "reference"])
    args = indexer_benchmark.parse_args()
    assert (args.heads, args.head_dim) == (3, 64)


def test_mxfp8_benchmark_quantization():
    generator = torch.Generator().manual_seed(17)
    original = torch.randn(2, 3, 128, generator=generator)
    original[0, :, :32] = 0
    original[1, :, 32:64] *= 16
    data, scales = indexer_benchmark.quantize_mxfp8(original)
    assert data.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float8_e8m0fnu
    assert scales.shape == (2, 3, 4)
    dequant = (data.float().unflatten(-1, (4, 32)) * scales.float().unsqueeze(-1)).flatten(-2)
    assert torch.isfinite(dequant).all()
    assert torch.equal(dequant[0, :, :32], original[0, :, :32])
    # E4M3 rounding: half an ULP for normals, half the subnormal quantum otherwise.
    allowance = 0.5 * torch.finfo(torch.float8_e4m3fn).eps * original.abs()
    allowance += scales.float().repeat_interleave(32, dim=-1) * 2**-10
    assert ((dequant - original).abs() <= allowance).all()


@pytest.mark.parametrize(
    "scorer,selector",
    [
        ("paged_score_kernel", "paged_radix_topk_kernel"),
        ("paged_score_tma_kernel", "indexer_topk_k512"),
        ("paged_warpdrive_score_bfloat16_h4_d128_p64_pos0_kernel", "indexer_topk_k512"),
    ],
)
def test_paged_stage_profile(scorer, selector):
    events = [
        SimpleNamespace(
            name=name,
            device_type=torch.autograd.DeviceType.CUDA,
            device_time_total=duration,
        )
        for name, duration in [
            (scorer, 5),
            (selector, 10),
            (scorer, 7),
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
    stages = summarize_stage_events(events * 3, 3, [scorer, selector] * 2)
    for stage, duration in [("scoring", 12), ("selection", 24)]:
        assert stages[stage]["profiled_mean_us"] == duration
        assert stages[stage]["launches_per_call"] == 2


@pytest.mark.parametrize(
    "graph,observed,error",
    [
        (["paged_score_kernel"] * 32, ["paged_score_kernel"] * 2, "count mismatch"),
        (["indexer_topk_k512"], ["indexer_topk_k512"] * 3, "count mismatch"),
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
        1, live, 4, 128, 512, 64, capacity=capacity, skip_zero_scores=zero, cute_selection=cute
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


@pytest.mark.parametrize("backend,capacity", [("triton", 64), ("cute", 65536)])
@pytest.mark.skipif(not torch.cuda.is_available() or torch.version.hip, reason="requires CUDA")
def test_paged_deterministic_replay_profiles_fills(
    deterministic_algorithms, monkeypatch, tmp_path, backend, capacity
):
    pytest.importorskip("transformer_nuggets.roofline")
    pytest.importorskip("cuda.bindings.driver")
    from attn_gym._backends.arch import cuda_runs_sm100_kernels

    if torch.cuda.get_device_capability()[0] < 8 or (
        backend == "cute" and not cuda_runs_sm100_kernels()
    ):
        pytest.skip("requires SM80+ for Triton or SM100-family for Warpdrive")
    if backend == "cute":
        pytest.importorskip("cutlass.cute")
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
            f"--backend={backend}",
            f"--capacity={capacity}",
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
    # Warpdrive scores the zero-weight row; only Triton skips writing/reading scores.
    scoring = result["model_scenario"]["stages"]["scoring"]
    assert scoring["logical_flops"] == (64 * 4 * (2 * 128 + 3) if backend == "cute" else 0)
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
