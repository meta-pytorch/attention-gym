"""Numerical checks, profiling, and benchmark reporting for delta-rule examples.

Batch construction, the training step, and capture lifetime are shown in the base training
example. These helpers take tensors or a runnable step; they never import the examples.
"""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from attn_gym.testing.profiling import record_distributed_profile


def assert_context_parallel_matches_reference(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    parameter_gradients: Iterable[tuple[torch.Tensor, torch.Tensor]],
    compute_dtype: torch.dtype,
) -> None:
    """Check outputs and world-summed parameter gradients at the input-precision budget."""
    assert_close = partial(
        torch.testing.assert_close,
        atol=torch.finfo(compute_dtype).eps,
        rtol=torch.finfo(compute_dtype).eps,
    )
    for actual, expected in pairs:
        assert_close(actual, expected)
    for actual, expected in parameter_gradients:
        reduced = actual.detach().clone()
        dist.all_reduce(reduced)
        assert_close(reduced, expected)


def profile_training_step(
    step: Callable[[], object],
    profile_path: Path,
    device: torch.device,
    warmup_steps: int,
) -> None:
    """Profile one eager step and report memory without owning its model or inputs."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)
    merged_path = record_distributed_profile(
        step,
        profile_path,
        "iteration",
        device,
        warmup_steps=warmup_steps,
    )
    step_peak = torch.cuda.max_memory_allocated(device) - resident
    print(
        f"rank {dist.get_rank()}: step peak {step_peak / 2**30:.2f} GiB above "
        f"{resident / 2**30:.2f} GiB resident",
        flush=True,
    )
    if merged_path is not None:
        print(f"profile={merged_path}", flush=True)


def measure_training_step(
    step: Callable[[], object],
    device: torch.device,
    *,
    steps: int,
    warmup_steps: int,
    local_tokens: int,
) -> dict[str, object]:
    """Measure steady-state CUDA event times and memory while the caller keeps inputs alive."""
    for _ in range(warmup_steps):
        step()
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)
    resident = torch.cuda.memory_allocated(device)

    step_ms: list[float] = []
    for _ in range(steps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        step()
        end.record()
        end.synchronize()
        step_ms.append(start.elapsed_time(end))
    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    return {
        "rank": dist.get_rank(),
        "host": os.uname().nodename,
        "local_tokens": local_tokens,
        "step_ms": step_ms,
        "resident_bytes": resident,
        "peak_bytes": peak,
        "step_peak_bytes": peak - resident,
        "reserved_bytes": reserved,
    }


def write_benchmark_report(
    local: dict[str, object],
    model: torch.nn.Module,
    device: torch.device,
    *,
    steps: int,
    warmup_steps: int,
    cuda_graph: bool,
    sequence_lengths: tuple[int, ...],
    partition: str,
) -> None:
    """Gather per-rank measurements and write the existing scaling-report schema on rank zero."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    if rank != 0:
        return
    ranks = [entry for entry in gathered if entry is not None]
    slowest = [max(entry["step_ms"][index] for entry in ranks) for index in range(steps)]
    mean = sum(slowest) / steps
    std = (sum((value - mean) ** 2 for value in slowest) / max(steps - 1, 1)) ** 0.5
    tokens = sum(sequence_lengths)
    variant = model.variant
    core_backend = (model.kernel_options or {}).get("backend", "fused")
    compute_dtype = str(model.compute_dtype).removeprefix("torch.")
    report = {
        "tokens": tokens,
        "sequence_lengths": list(sequence_lengths),
        "partition": partition,
        "hidden_size": model.hidden_size,
        "heads": model.num_heads,
        "head_dim": model.head_dim,
        "variant": variant,
        "compute_dtype": compute_dtype,
        "short_conv_kernel_size": model.qkv_conv1d.kernel_size[0],
        "kda_backend": core_backend,
        "device_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "world_size": world_size,
        "mode": "cuda_graph" if cuda_graph else "eager",
        "steps": steps,
        "warmup_steps": warmup_steps,
        "step_ms_mean": mean,
        "step_ms_std": std,
        "step_ms_min": min(slowest),
        "tokens_per_s": tokens / (mean / 1e3),
        "step_peak_gib_max": max(entry["step_peak_bytes"] for entry in ranks) / 2**30,
        "resident_gib_max": max(entry["resident_bytes"] for entry in ranks) / 2**30,
        "peak_gib_max": max(entry["peak_bytes"] for entry in ranks) / 2**30,
        # Graph replays allocate from the graph's private pool, which
        # memory_allocated does not see; reserved is the footprint that matters there.
        "reserved_gib_max": max(entry["reserved_bytes"] for entry in ranks) / 2**30,
        "ranks": ranks,
    }
    output_path = Path(
        "data",
        f"{variant}_cp_scaling_{report['mode']}_{partition}_{core_backend}"
        f"_{compute_dtype}_w{world_size}_t{tokens}_h{model.num_heads}_c{model.hidden_size}.json",
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=1))
    print(
        f"benchmark: W={world_size} tokens={tokens} "
        f"({min(entry['local_tokens'] for entry in ranks)}.."
        f"{max(entry['local_tokens'] for entry in ranks)}/rank) "
        f"{report['mode']} step {mean:.2f}±{std:.2f} ms  {report['tokens_per_s'] / 1e6:.2f} Mtok/s  "
        f"peak {report['peak_gib_max']:.2f} GiB allocated (step +{report['step_peak_gib_max']:.2f}), "
        f"{report['reserved_gib_max']:.2f} GiB reserved -> {output_path}",
        flush=True,
    )
