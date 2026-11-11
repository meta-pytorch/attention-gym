"""Replay, timing statistics and provenance shared by the examples' CUDA-graph benchmarks."""

import hashlib
import importlib.metadata
import os
import subprocess
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch


def settle(graphs: Iterable[torch.cuda.CUDAGraph], seconds: float) -> None:
    """Replay the graphs in turn for ``seconds`` so DVFS settles before the first timed round."""
    graphs = list(graphs)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()


def replay_timed(
    graph: torch.cuda.CUDAGraph, reps: int, warmup: int
) -> tuple[list[float], dict[str, list[float]]]:
    """Return whole-replay CUDA-event times and per-kernel CUPTI times, in microseconds.

    Run ``warmup`` untimed replays, then ``reps`` measured replays. Kernel samples are
    grouped by name.
    """
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for _ in range(warmup):
        graph.replay()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for s, e in zip(starts, ends):
            s.record()
            graph.replay()
            e.record()
        torch.cuda.synchronize()
    kernels = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.device_time > 0:
            kernels.setdefault(ev.name, []).append(ev.device_time)
    return [s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends)], kernels


def source_metadata(source: Path) -> dict:
    """Return the source path, checkout commit and dirty flag, and current file SHA-256.

    The hash includes uncommitted edits. Commit and dirty are None without Git metadata.
    """
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(source.parent), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "source": str(source),
        "commit": commit,
        "dirty": dirty,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }


def provenance(sources: dict[str, Path], arguments: dict) -> dict:
    """Record package versions, GPU reservation, source metadata and benchmark arguments."""
    return {
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "triton", "nvidia-cutlass-dsl")
        },
        "gpu": torch.cuda.get_device_name(),
        "reserved_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        **{name: source_metadata(path) for name, path in sources.items()},
        "arguments": arguments,
    }


@dataclass(frozen=True)
class CudaBenchmarkStats:
    samples_us: tuple[float, ...]
    quantiles_us: tuple[float, float, float]

    @staticmethod
    def quantile(samples: Sequence[float], q: float) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"q must be in [0, 1], got {q}")
        if len(samples) == 0:
            raise ValueError("samples must be non-empty")

        ordered = sorted(float(sample) for sample in samples)
        if len(ordered) == 1:
            return ordered[0]

        position = (len(ordered) - 1) * q
        lower_idx = int(position)
        upper_idx = min(lower_idx + 1, len(ordered) - 1)
        if lower_idx == upper_idx:
            return ordered[lower_idx]

        weight = position - lower_idx
        lower = ordered[lower_idx]
        upper = ordered[upper_idx]
        return lower + (upper - lower) * weight

    @classmethod
    def from_samples(cls, samples_us: Sequence[float]) -> "CudaBenchmarkStats":
        samples = tuple(float(sample) for sample in samples_us)
        quantiles_us = (
            cls.quantile(samples, 0.05),
            cls.quantile(samples, 0.50),
            cls.quantile(samples, 0.95),
        )
        return cls(samples_us=samples, quantiles_us=quantiles_us)

    @property
    def p05_us(self) -> float:
        return self.quantiles_us[0]

    @property
    def median_us(self) -> float:
        return self.quantiles_us[1]

    @property
    def p95_us(self) -> float:
        return self.quantiles_us[2]


def benchmark_cuda_graph_stats(
    graph: torch.cuda.CUDAGraph,
    *,
    num_iters: int = 100,
    warmup_iters: int = 20,
) -> CudaBenchmarkStats:
    """Time an already-captured CUDA graph without capturing or invoking its Python body.

    Return replay samples and quantiles in microseconds. Capture, input updates and
    correctness checks are untimed.
    Fixed buffers give a reused-buffer workload. CUDA-event intervals include GPU work
    and can include host submission gaps; they are not kernel-only times.

    The caller owns the graph and buffers, selects its CUDA device, and orders input
    updates before replay on the current stream. Warmup and timing both execute captured
    mutations. The helper synchronizes the current device after warmup and each timed replay.

    Args:
        graph: An already-captured graph, ready to replay on the current device.
        num_iters: Number of measured replays; must be positive.
        warmup_iters: Number of untimed replays; may be zero.
    """
    if num_iters <= 0:
        raise ValueError("num_iters must be positive")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")

    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples_us = []
    for _ in range(num_iters):
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1e3)
    return CudaBenchmarkStats.from_samples(samples_us)
