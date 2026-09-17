"""Benchmark CLI, dependency independence, and graph timing contracts."""

import sys
from argparse import Namespace
from contextlib import contextmanager
from types import ModuleType

import pytest
import torch

from benchmarks.sparse import indexer_benchmark


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


@pytest.mark.parametrize("samples", [[3000.0, 1000.0, 2000.0], [3000.0, 1000.0]])
def test_benchmark_prints_median_ms(monkeypatch, capsys, samples):
    monkeypatch.setattr(sys, "argv", ["indexer_benchmark.py"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "mock CUDA GPU")
    monkeypatch.setattr(indexer_benchmark, "make_inputs", lambda args: (None,) * 5)
    monkeypatch.setattr(indexer_benchmark, "benchmark_graph", lambda *args, **kwargs: samples)
    indexer_benchmark.main()
    assert "forward: 2.000 ms" in capsys.readouterr().out


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
    assert state == {
        "capturing": False,
        "released": True,
        "calls": warmup + 1,
        "replays": warmup + 3,
        "records": 2,
    }


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
