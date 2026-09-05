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
def profile_trace(path: Path, *, warmup: int = 0) -> Iterator[torch.profiler.profile]:
    """Record one native Perfetto trace per rank, discarding ``warmup`` scheduled steps."""
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
        warmup=warmup,
    ) as active_profiler:
        yield active_profiler


def record_distributed_profile(
    step: Callable[[], object],
    path: Path,
    label: str,
    device: torch.device,
    *,
    warmup_steps: int = 0,
) -> Path | None:
    """Profile one synchronized world-group step and merge its native rank traces.

    ``warmup_steps`` extra steps run first with the profiler attached but are
    dropped from the trace, so the recorded step is steady state: kernels,
    NCCL communicators, allocator, and CUPTI are all hot, and the ranks
    re-align on a barrier right before it. Without warmup the trace mostly
    shows launch skew (ranks waiting inside the first collective).
    """
    try:
        merge_traces = import_module("transformer_nuggets.utils.merge_traces").merge_traces
    except ImportError as error:
        raise RuntimeError(
            "distributed profiling requires transformer-nuggets with native Perfetto support"
        ) from error

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    dist.barrier()
    with profile_trace(path, warmup=warmup_steps) as active_profiler:
        for index in range(warmup_steps):
            step()
            if index == warmup_steps - 1:
                # Still inside the discarded warmup phase: pay the re-alignment here so
                # neither the barrier nor the skew it absorbs appears in the trace.
                torch.cuda.synchronize(device)
                dist.barrier()
            active_profiler.step()
        if warmup_steps == 0:
            # Profiler initialization is rank-local, so align again after every profiler is active.
            dist.barrier()
        torch.cuda.synchronize(device)
        with torch.profiler.record_function(f"cp/rank_{rank}/{label}"):
            step()
        torch.cuda.synchronize(device)
        active_profiler.step()
    dist.barrier()

    # Ranks on other nodes may not share a filesystem with rank 0: ship the trace
    # bytes over the process group so the merge only needs rank 0's local disk.
    rank_paths = [
        path.with_name(f"{path.stem}_rank_{index}.pftrace") for index in range(world_size)
    ]
    gathered: list[bytes | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(rank_paths[rank].read_bytes(), gathered, dst=0)

    merged_path = None
    if rank == 0:
        assert gathered is not None
        for rank_path, trace in zip(rank_paths, gathered, strict=True):
            if trace is not None and not rank_path.exists():
                rank_path.write_bytes(trace)
        merged_path = path.with_name(f"{path.stem}_merged.pftrace")
        merge_traces(
            [str(rank_path) for rank_path in rank_paths],
            str(merged_path),
            labels=[f"Rank {index} · GPU {index}" for index in range(world_size)],
            align_timestamps=True,
        )
    dist.barrier()
    return merged_path


__all__ = ["kernel_stage", "profile_trace", "record_distributed_profile"]
