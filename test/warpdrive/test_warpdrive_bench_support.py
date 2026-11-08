"""CUDA-graph timing statistics shared by the warpdrive example benchmarks."""

import pytest
import torch

from attn_gym.warpdrive.examples.bench_support import (
    CudaBenchmarkStats,
    benchmark_cuda_graph_stats,
)


@pytest.mark.parametrize("warmup_iters", [0, 2])
def test_benchmark_cuda_graph_stats_does_not_capture(monkeypatch, warmup_iters):
    calls = []
    timings_ms = iter([0.010, 0.012, 0.011])

    class FakeGraph:
        def replay(self):
            calls.append("replay")

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            calls.append("record")

        def elapsed_time(self, end):
            assert end is not self
            calls.append("elapsed")
            return next(timings_ms)

    def unexpected_capture(*args, **kwargs):
        pytest.fail("An existing graph must not be captured again")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", unexpected_capture)
    monkeypatch.setattr(torch.cuda, "graph", unexpected_capture)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))

    stats = benchmark_cuda_graph_stats(
        FakeGraph(),
        num_iters=3,
        warmup_iters=warmup_iters,
        confidence=0.90,
        n_resamples=50,
        seed=7,
    )

    expected_calls = ["replay"] * warmup_iters + ["sync"]
    expected_calls += ["record", "replay", "record", "sync", "elapsed"] * 3
    assert calls == expected_calls
    assert stats == CudaBenchmarkStats.from_samples(
        [10.0, 12.0, 11.0], confidence=0.90, n_resamples=50, seed=7
    )


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"num_iters": 0}, "num_iters must be positive"),
        ({"num_iters": -1}, "num_iters must be positive"),
        ({"warmup_iters": -1}, "warmup_iters must be non-negative"),
    ],
)
def test_benchmark_cuda_graph_stats_rejects_invalid_counts(kwargs, message):
    with pytest.raises(ValueError, match=message):
        benchmark_cuda_graph_stats(object(), **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("side_stream", [False, True])
def test_benchmark_existing_cuda_graph_with_updated_inputs(monkeypatch, side_stream):
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        torch.add(x, 3, out=out)
    torch.cuda.current_stream().wait_stream(capture_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        torch.add(x, 3, out=out)

    def unexpected_capture(*args, **kwargs):
        pytest.fail("Timing must reuse the existing graph")

    monkeypatch.setattr(torch.cuda, "CUDAGraph", unexpected_capture)
    monkeypatch.setattr(torch.cuda, "graph", unexpected_capture)
    stream = torch.cuda.Stream() if side_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for offset in (0, 5):
            x.add_(offset)
            stats = benchmark_cuda_graph_stats(graph, num_iters=5, warmup_iters=2)
            torch.testing.assert_close(out, x + 3)
            assert len(stats.samples_us) == 5
            assert all(sample > 0 for sample in stats.samples_us)
            assert stats.p05_us <= stats.median_us <= stats.p95_us
    torch.cuda.current_stream().wait_stream(stream)
