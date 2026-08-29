"""Profiling helpers for distributed example validation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from importlib import import_module
from pathlib import Path

import torch
import torch.distributed as dist


@contextmanager
def kernel_stage(name: str, annotate: bool, *, backward: bool = True) -> Iterator[None]:
    """Label eager profiler ranges and optionally annotate captured CUDA Graph kernels."""
    annotation = nullcontext()
    if annotate:
        from torch.cuda.graph_annotations import mark_kernels

        annotation = mark_kernels(name, backward=backward)
    with torch.profiler.record_function(name), annotation:
        yield


@contextmanager
def profile_trace(path: Path) -> Iterator[torch.profiler.profile]:
    """Record one native Perfetto trace per rank."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        profiler = import_module("transformer_nuggets.utils.benchmark").profiler
    except ImportError as error:
        raise RuntimeError(
            "profiling requires transformer-nuggets with native Perfetto support"
        ) from error

    with profiler(
        path.with_suffix(".pftrace"),
        record_shapes=True,
        trace_format="track_event",
    ) as active_profiler:
        yield active_profiler


def record_distributed_profile(
    step: Callable[[], object],
    path: Path,
    label: str,
    device: torch.device,
) -> Path | None:
    """Profile one synchronized world-group step and merge its native rank traces."""
    try:
        merge_traces = import_module("transformer_nuggets.utils.merge_traces").merge_traces
    except ImportError as error:
        raise RuntimeError(
            "distributed profiling requires transformer-nuggets with native Perfetto support"
        ) from error

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dist.barrier()
    with profile_trace(path) as active_profiler:
        # Profiler initialization is rank-local, so align again after every profiler is active.
        dist.barrier()
        torch.cuda.synchronize(device)
        with torch.profiler.record_function(f"cp/rank_{rank}/{label}"):
            step()
        torch.cuda.synchronize(device)
        active_profiler.step()
    dist.barrier()

    merged_path = None
    if rank == 0:
        rank_paths = [
            path.with_name(f"{path.stem}_rank_{index}.pftrace") for index in range(world_size)
        ]
        merged_path = path.with_name(f"{path.stem}_merged.pftrace")
        merge_traces(
            [str(rank_path) for rank_path in rank_paths],
            str(merged_path),
            labels=[f"Rank {index} · GPU {index}" for index in range(world_size)],
            align_timestamps=False,
        )
    dist.barrier()
    return merged_path


__all__ = ["kernel_stage", "profile_trace", "record_distributed_profile"]
