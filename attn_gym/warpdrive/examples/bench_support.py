"""Replay, timing statistics and provenance shared by the examples' CUDA-graph benchmarks."""

import hashlib
import importlib.metadata
import os
import random
import statistics
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
    median_us: float
    median_ci_us: tuple[float, float]
    quantiles_us: tuple[float, float, float]
    confidence: float

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
    def bootstrap_median_confidence_interval(
        cls,
        samples: Sequence[float],
        confidence: float = 0.95,
        n_resamples: int = 1000,
        seed: int = 0,
    ) -> tuple[float, float]:
        """Estimate a percentile bootstrap confidence interval for the sample median.

        Resample timings with replacement and take quantiles of the resampled medians.
        No parametric distribution is assumed. Samples must come from one stable,
        warmed-up workload and be approximately exchangeable: thermal drift, autotuning
        and startup effects invalidate that assumption.

        Small samples give unstable intervals; single-digit counts are weak, and low
        tens still need caution. Check that adding or removing a few steady-state
        samples does not move the median much.
        """
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if len(samples) == 0:
            raise ValueError("samples must be non-empty")
        if len(samples) == 1:
            value = float(samples[0])
            return value, value

        rng = random.Random(seed)
        estimates = [
            statistics.median(rng.choices(samples, k=len(samples)))
            for _ in range(max(1, n_resamples))
        ]
        alpha = 1.0 - confidence
        return cls.quantile(estimates, alpha / 2), cls.quantile(estimates, 1.0 - alpha / 2)

    @classmethod
    def from_samples(
        cls,
        samples_us: Sequence[float],
        confidence: float = 0.95,
        n_resamples: int = 1000,
        seed: int = 0,
    ) -> "CudaBenchmarkStats":
        samples = tuple(float(sample) for sample in samples_us)
        quantiles_us = (
            cls.quantile(samples, 0.05),
            cls.quantile(samples, 0.50),
            cls.quantile(samples, 0.95),
        )
        return cls(
            samples_us=samples,
            median_us=quantiles_us[1],
            median_ci_us=cls.bootstrap_median_confidence_interval(
                samples,
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed,
            ),
            quantiles_us=quantiles_us,
            confidence=confidence,
        )

    @property
    def p05_us(self) -> float:
        return self.quantiles_us[0]

    @property
    def p50_us(self) -> float:
        return self.quantiles_us[1]

    @property
    def p95_us(self) -> float:
        return self.quantiles_us[2]


def _time_cuda_graph_replay_samples_us(
    graph: torch.cuda.CUDAGraph,
    *,
    num_iters: int,
    warmup_iters: int,
) -> list[float]:
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
    return samples_us


def benchmark_cuda_graph_stats(
    graph: torch.cuda.CUDAGraph,
    *,
    num_iters: int = 100,
    warmup_iters: int = 20,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> CudaBenchmarkStats:
    """Time an already-captured CUDA graph without capturing or invoking its Python body.

    Return replay samples and quantiles in microseconds, plus a bootstrap confidence
    interval for the median. Capture, input updates and correctness checks are untimed.
    Fixed buffers give a reused-buffer workload. CUDA-event intervals include GPU work
    and can include host submission gaps; they are not kernel-only times.

    The caller owns the graph and buffers, selects its CUDA device, and orders input
    updates before replay on the current stream. Warmup and timing both execute captured
    mutations. The helper synchronizes the current device after warmup and each timed replay.

    Args:
        graph: An already-captured graph, ready to replay on the current device.
        num_iters: Number of measured replays; must be positive.
        warmup_iters: Number of untimed replays; may be zero.
        confidence: Bootstrap confidence level in (0, 1).
        n_resamples: Number of bootstrap resamples.
        seed: Seed for bootstrap resampling, not the captured workload.
    """
    samples_us = _time_cuda_graph_replay_samples_us(
        graph, num_iters=num_iters, warmup_iters=warmup_iters
    )
    return CudaBenchmarkStats.from_samples(
        samples_us, confidence=confidence, n_resamples=n_resamples, seed=seed
    )
