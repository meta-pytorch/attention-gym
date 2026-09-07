"""Optional profiler ranges, CUDA Graph annotations, and example trace export."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from functools import wraps
from importlib import import_module
from inspect import signature
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist

TraceFormat = Literal["track_event", "chrome_json"]


def trace_suffix(trace_format: TraceFormat) -> str:
    """File suffix transformer-nuggets writes for ``trace_format``."""
    return ".pftrace" if trace_format == "track_event" else ".json.gz"


def record_function(enabled: bool, name: str):
    """Create a profiler range only when profiling is requested."""
    return torch.profiler.record_function(name) if enabled else nullcontext()


def graph_annotations_available() -> bool:
    """Check optional CUDA Graph annotation support without making it an import requirement."""
    try:
        from torch.cuda.graph_annotations import is_available
    except ImportError:
        return False
    return is_available()


def mark_kernels(*args: Any, **kwargs: Any):
    """Load optional CUDA Graph annotations only when they are requested."""
    from torch.cuda.graph_annotations import mark_kernels as annotate

    return annotate(*args, **kwargs)


def annotate_kernels(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Label a module method when ``module.enable_graph_annotations`` is true.

    The label may reference attributes through ``{module.attribute}``; disabled annotations
    bypass both label formatting and the optional CUDA Graph annotation API.
    """

    def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(function)
        def annotated(self: Any, *args: Any, **kwargs: Any) -> Any:
            if not self.enable_graph_annotations:
                return function(self, *args, **kwargs)
            with mark_kernels(name.format(module=self)):
                return function(self, *args, **kwargs)

        return annotated

    return decorate


@contextmanager
def kernel_stage(name: str, annotate: bool, *, backward: bool = True) -> Iterator[None]:
    """Label eager profiler ranges and optionally annotate captured CUDA Graph kernels."""
    annotation = nullcontext()
    if annotate:
        annotation = mark_kernels(name, backward=backward)
    with torch.profiler.record_function(name), annotation:
        yield


@contextmanager
def profile_trace(
    path: Path, *, warmup: int = 0, trace_format: TraceFormat = "track_event"
) -> Iterator[torch.profiler.profile]:
    """Record one trace per rank, discarding ``warmup`` scheduled steps.

    ``track_event`` writes native Perfetto; ``chrome_json`` writes gzipped Kineto JSON, which
    keeps the shape metadata ``annotate-roofline`` needs (a later Perfetto merge drops it).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        profiler = import_module("transformer_nuggets.utils.benchmark").profiler
    except ImportError as error:
        raise RuntimeError(
            "profiling requires transformer-nuggets with native Perfetto support"
        ) from error

    if "trace_format" not in signature(profiler).parameters:
        raise RuntimeError(
            "installed transformer-nuggets lacks native Perfetto support; "
            "install it from https://github.com/drisspg/transformer_nuggets"
        )

    with profiler(
        path.with_name(path.stem + trace_suffix(trace_format)),
        record_shapes=True,
        trace_format=trace_format,
        gzip_trace=trace_format == "chrome_json",
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
    trace_format: TraceFormat = "track_event",
) -> Path | None:
    """Profile one synchronized world-group step and merge its rank traces.

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
    with profile_trace(path, warmup=warmup_steps, trace_format=trace_format) as active_profiler:
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
    suffix = trace_suffix(trace_format)
    rank_paths = [
        path.with_name(f"{path.stem}_rank_{index}{suffix}") for index in range(world_size)
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
        # Native traces preserve their clock timestamps; JSON-style re-zeroing is unsupported.
        merge_traces(
            [str(rank_path) for rank_path in rank_paths],
            str(merged_path),
            labels=[f"Rank {index} · GPU {index}" for index in range(world_size)],
        )
    dist.barrier()
    return merged_path


__all__ = [
    "TraceFormat",
    "annotate_kernels",
    "graph_annotations_available",
    "kernel_stage",
    "profile_trace",
    "record_distributed_profile",
    "record_function",
]
