"""Graph-safe chunk scheduling for packed variable-length KDA inputs.

Coordinate system
-----------------

- **Sequence**: packed token range ``[cu_seqlens[i], cu_seqlens[i + 1])``.
- **Local chunk**: chunk number within one sequence. Local chunk 0 begins at the
  sequence start, not at a multiple of ``chunk_size`` in the packed buffer.
- **Global chunk**: a dense index over all active local chunks from every
  sequence. Kernels use this as their common chunk coordinate.
- **Chunk capacity**: the fixed CUDA Graph upper bound on global chunks. Global
  chunk slots beyond the runtime active count contain no work on that replay.

Scheduling views
----------------

``GridScheduler`` supports three launch geometries; ``subtask`` is terminology
for only the flattened one:

1. **Flat-task scheduling** (Triton): a kernel supplies only its number of
   kernel-specific work coordinates per chunk. The scheduler sees that count,
   not whether the coordinate means a head, channel tile, or
   ``(head, output-tile)`` pair. The kernel flattens and later decodes::

       task = global_chunk * subtasks_per_chunk + subtask
       global_chunk, subtask = divmod(task, subtasks_per_chunk)

   A persistent worker CTA processes tasks
   ``w, w + num_workers, w + 2 * num_workers, ...``. Here a subtask is CTA-level
   parallel work, not an inner loop.

2. **Chunk-axis scheduling** (CuTeDSL): the scheduler controls only the chunk
   grid axis. Pair, head, and tile coordinates remain explicit grid axes, and
   each CTA strides over chunks for one fixed combination of those coordinates.
   This path neither passes nor decodes a ``subtasks_per_chunk`` value.

3. **Sequence-axis scheduling**: recurrence kernels persist the sequence-head
   axis while retaining the value tile as an explicit grid axis. Kernels derive
   the active sequence extent from ``cu_seqlens`` during replay.

The value three in the worked example below is therefore illustrative only for
flat-task scheduling; it is not derived from ``cu_seqlens`` and does not describe
the CuTeDSL grid.

Shared routing metadata
-----------------------

``chunk_offsets`` has the same fixed ``[N + 1]`` shape as ``cu_seqlens``. During
CUDA Graph capture its storage and shape become fixed, but one device operation
rebuilds its values from the current ``cu_seqlens`` on every replay::

    chunk_count[i] = ceil_div(cu_seqlens[i + 1] - cu_seqlens[i], chunk_size)
    chunk_offsets = exclusive_prefix_sum(chunk_count)

The fused ``chunk_kda`` implementation prepares this prefix once per call, passes the same
``RaggedChunkMetadata`` to its internal gate scan and core stages, and saves the same
``cu_seqlens``/``chunk_offsets`` storage for backward. Each consumer still binary-searches
``chunk_offsets`` to map a global chunk to its sequence and local chunk, but it does not
recompute every sequence's chunk count and prefix. This avoids duplicate metadata launches and
guarantees one global-chunk coordinate system across the implementation.

Worked example
--------------

For ``cu_seqlens=[0, 5, 5, 12]`` and ``chunk_size=4``:

- sequence 0 has length 5 and therefore 2 chunks;
- sequence 1 is empty and therefore has 0 chunks;
- sequence 2 has length 7 and therefore 2 chunks.

The counts are ``[2, 0, 2]``. Their exclusive prefix sum is
``chunk_offsets=[0, 2, 2, 4]``: sequence 0 owns global chunks ``[0, 2)``, sequence
1 owns none, and sequence 2 owns ``[2, 4)``. Global chunk 2 is therefore local
chunk 0 of sequence 2 and begins at packed token 5. If a flat-scheduled example
kernel has three subtasks per chunk, task 7 maps to
``(global_chunk, subtask) = divmod(7, 3) = (2, 1)``.

Flat STATIC scheduling launches every capacity task. Chunk-axis STATIC
scheduling launches the full chunk-capacity axis while retaining the other grid
axes. Inactive CTAs return after reading the active count. PERSISTENT scheduling
bounds the flat, chunk, or sequence axis and strides only over runtime active work.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import NamedTuple

import torch
import triton
import triton.language as tl

# The op-wrapped metadata builder is re-exported so every caller routes the offsets
# launch through the registered scheduler op instead of tracing it directly.
from attn_gym.linear.kda.chunk_schedule import (
    RaggedChunkMetadata,
    ResolvedSchedule,
    ScheduleKind,
    ScheduleRequest,
    chunk_capacity,
    prepare_ragged_chunk_metadata,
    validate_schedule_request,
)

# Default persistent worker cap; individual kernels can override it where measured.
PERSISTENT_CTAS_PER_SM = 4

# Auto policy: static launches pay only launch time for padding CTAs, so a
# few waves of them are cheaper than the persistent stride loop's overhead.
# Switch only after the capacity grid exceeds this many bounded-worker waves:
# with 100 persistent workers, capacities through 400 tasks use the static grid.
PERSISTENT_AUTO_WAVES = 4


@functools.cache
def _multiprocessor_count_for_index(device_index: int) -> int:
    """Cache the SM count for one concrete CUDA device."""
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _multiprocessor_count(device: torch.device) -> int:
    """Resolve unindexed CUDA devices before consulting the SM-count cache."""
    device_index = torch.cuda.current_device() if device.index is None else device.index
    return _multiprocessor_count_for_index(device_index)


@dataclass(frozen=True)
class GridScheduler:
    """Map ragged work onto launch grids for Triton and CuTeDSL kernels.

    The runtime work count depends on sequence lengths, but CUDA Graph
    capture fixes the launch shape at ``metadata.capacity``. Static kernels
    launch one CTA per capacity task; inactive CTAs read the active count and
    return. Persistent kernels launch a bounded machine-derived worker grid and
    stride over the active flat-task list, so capacity padding adds neither CTAs
    nor loop iterations. Workers are independent and require no inter-CTA
    synchronization; their count is an occupancy choice, not a co-residency
    requirement.
    """

    metadata: RaggedChunkMetadata
    ctas_per_sm: int = PERSISTENT_CTAS_PER_SM

    @staticmethod
    def _resolve(
        request: ScheduleRequest,
        eligible: bool,
        capacity_tasks: int,
        workers: int,
        requirement: str,
    ) -> ResolvedSchedule:
        """Translate caller policy into one concrete schedule."""
        validate_schedule_request(request)
        if request is ScheduleRequest.PERSISTENT and not eligible:
            raise ValueError(f"persistent scheduling requires {requirement}")
        if capacity_tasks == 0:
            return ResolvedSchedule(ScheduleKind.STATIC, workers, capacity_tasks)
        if request is ScheduleRequest.AUTO:
            persistent = eligible and capacity_tasks > PERSISTENT_AUTO_WAVES * workers
        else:
            persistent = request is ScheduleRequest.PERSISTENT
        return ResolvedSchedule(
            ScheduleKind.PERSISTENT if persistent else ScheduleKind.STATIC,
            workers,
            capacity_tasks,
        )

    def num_workers(self, subtasks_per_chunk: int, device: torch.device | str) -> int:
        """Return the bounded worker count for a flat ``(chunk, subtask)`` list."""
        sm_count = _multiprocessor_count(torch.device(device))
        return min(self.metadata.capacity * subtasks_per_chunk, sm_count * self.ctas_per_sm)

    def num_chunk_workers(self, device: torch.device | str) -> int:
        """Determine the persistent grid size along the chunk axis for this device.

        CuTeDSL kernels retain their natural grid dimensions for the remaining
        axes (pair, head, ...), while this helper sizes only the chunk axis. These
        small CTAs are latency bound, so measured performance improves as chunk
        parallelism increases; cap that axis at one worker per SM, independently
        of the number of subtasks per chunk.
        """
        sm_count = _multiprocessor_count(torch.device(device))
        if self.metadata.capacity == 0:
            return 0
        # Power-of-two quantization keeps the TVM-FFI compile cache small when
        # eager callers vary the capacity below the SM count.
        return min(1 << (self.metadata.capacity - 1).bit_length(), sm_count)

    def resolve_flat(
        self,
        request: ScheduleRequest,
        subtasks_per_chunk: int,
        device: torch.device | str,
        *,
        eligible: bool = True,
        requirement: str = "a persistent kernel for this input layout",
    ) -> ResolvedSchedule:
        """Resolve scheduling for a flattened ``(chunk, subtask)`` work list."""
        workers = self.num_workers(subtasks_per_chunk, device)
        return self._resolve(
            request,
            eligible,
            self.metadata.capacity * subtasks_per_chunk,
            workers,
            requirement,
        )

    def resolve_sequences(
        self,
        request: ScheduleRequest,
        subtasks_per_sequence: int,
        device: torch.device | str,
    ) -> ResolvedSchedule:
        """Resolve scheduling for flattened sequence-level recurrence work."""
        num_sequences = self.metadata.cu_seqlens.shape[0] - 1
        sm_count = _multiprocessor_count(torch.device(device))
        workers = min(num_sequences * subtasks_per_sequence, sm_count * self.ctas_per_sm)
        return self._resolve(
            request,
            True,
            num_sequences * subtasks_per_sequence,
            workers,
            "a sequence-persistent kernel",
        )

    def resolve_chunk(
        self,
        request: ScheduleRequest,
        device: torch.device | str,
    ) -> ResolvedSchedule:
        """Resolve scheduling when only the chunk axis is persistent."""
        workers = self.num_chunk_workers(device)
        return self._resolve(request, True, self.metadata.capacity, workers, "a ragged kernel")


class ChunkWork(NamedTuple):
    """One sequence-local chunk decoded from a global logical work index."""

    global_chunk: int
    sequence: int
    local_chunk: int
    token_start: int
    valid_tokens: int


def chunk_work_oracle(cu_seqlens: list[int], chunk_size: int) -> list[ChunkWork]:
    """Decode all logical chunks on the CPU for tests and scheduler validation."""
    if len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if cu_seqlens[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(begin > end for begin, end in itertools.pairwise(cu_seqlens)):
        raise ValueError("cu_seqlens must be monotonic")

    work: list[ChunkWork] = []
    for sequence, (begin, end) in enumerate(itertools.pairwise(cu_seqlens)):
        for token_start in range(begin, end, chunk_size):
            work.append(
                ChunkWork(
                    global_chunk=len(work),
                    sequence=sequence,
                    local_chunk=(token_start - begin) // chunk_size,
                    token_start=token_start,
                    valid_tokens=min(chunk_size, end - token_start),
                )
            )
    return work


@triton.jit(debug=True, do_not_specialize=["num_sequences"])
def _prepare_ragged_chunk_offsets_kernel(
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    tokens: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Validate boundaries and build the chunk-count prefix in one CTA."""
    sequence = tl.arange(0, BLOCK)
    sequence_mask = sequence < num_sequences
    begin = tl.load(cu_seqlens + sequence, mask=sequence_mask, other=0)
    end = tl.load(cu_seqlens + sequence + 1, mask=sequence_mask, other=0)
    valid = (begin >= 0) & (begin <= end) & (end <= tokens) & ((sequence != 0) | (begin == 0))
    tl.device_assert(valid, "invalid packed cu_seqlens", mask=sequence_mask)

    count = tl.where(sequence_mask & valid, (end - begin + chunk_size - 1) // chunk_size, 0)
    inclusive_offset = tl.cumsum(count, axis=0)
    tl.store(chunk_offsets, 0)
    tl.store(chunk_offsets + sequence + 1, inclusive_offset, mask=sequence_mask)


@triton.jit
def load_ragged_chunk_count(chunk_offsets, num_sequences):
    """Load the terminal prefix-sum entry containing the active chunk count."""
    return tl.load(chunk_offsets + num_sequences).to(tl.int64)


@triton.jit
def load_ragged_task_count(chunk_offsets, num_sequences, subtasks_per_chunk):
    """Return the runtime number of active flattened chunk tasks.

    ``num_sequences`` selects only the terminal prefix-sum entry; scheduling does
    not stride over sequences. If each chunk has three subtasks, flat tasks 0--2
    belong to global chunk 0 and tasks 3--5 belong to global chunk 1. Worker ``w``
    handles ``w, w + num_workers, ...`` up to this returned bound.
    """
    return load_ragged_chunk_count(chunk_offsets, num_sequences) * subtasks_per_chunk


@triton.jit
def load_ragged_sequence_extent(cu_seqlens, num_sequences: tl.constexpr):
    """Return one past the last sequence slot that may contain tokens."""
    active_tokens = tl.load(cu_seqlens + num_sequences)
    low = 0
    high = num_sequences
    while low < high:
        middle = (low + high) // 2
        if tl.load(cu_seqlens + middle) < active_tokens:
            low = middle + 1
        else:
            high = middle
    return low


@triton.jit
def decode_ragged_task(task, subtasks_per_chunk):
    """Decode one widened flat task into ``(global_chunk, subtask)``."""
    return task // subtasks_per_chunk, task % subtasks_per_chunk


@triton.jit
def load_ragged_chunk_work(
    cu_seqlens,
    chunk_offsets,
    global_chunk,
    num_sequences,
    chunk_size: tl.constexpr,
):
    """Binary-search one known-active global chunk's sequence boundaries.

    Returns its sequence index, sequence-relative chunk index, packed token start,
    and valid token count. Callers must reject ``global_chunk >= chunk_offsets[-1]``
    before invoking this helper.
    """
    low = 0
    high = num_sequences + 1
    while low < high:
        middle = (low + high) // 2
        offset = tl.load(chunk_offsets + middle)
        if offset <= global_chunk:
            low = middle + 1
        else:
            high = middle

    sequence = low - 1
    sequence_offset = tl.load(chunk_offsets + sequence)
    local_chunk = global_chunk - sequence_offset
    begin = tl.load(cu_seqlens + sequence)
    end = tl.load(cu_seqlens + sequence + 1)
    token_start = begin + local_chunk * chunk_size
    valid_tokens = tl.minimum(chunk_size, end - token_start)
    return sequence, local_chunk, token_start, valid_tokens


@triton.jit(do_not_specialize=["num_sequences"])
def _decode_ragged_chunk_work_kernel(
    cu_seqlens,
    chunk_offsets,
    work,
    num_sequences,
    chunk_size: tl.constexpr,
):
    global_chunk = tl.program_id(0)
    active_chunks = load_ragged_chunk_count(chunk_offsets, num_sequences)
    output = work + global_chunk * 5

    if global_chunk < active_chunks:
        sequence, local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            global_chunk,
            num_sequences,
            chunk_size,
        )

        tl.store(output, global_chunk)
        tl.store(output + 1, sequence)
        tl.store(output + 2, local_chunk)
        tl.store(output + 3, token_start)
        tl.store(output + 4, valid_tokens)
    else:
        for field in tl.static_range(5):
            tl.store(output + field, -1)


def _prepare_ragged_chunk_offsets(
    cu_seqlens: torch.Tensor,
    tokens: int,
    chunk_size: int,
) -> torch.Tensor:
    """Build packed chunk offsets for the torch-only registration wrapper."""
    num_sequences = cu_seqlens.shape[0] - 1
    chunk_offsets = torch.empty_like(cu_seqlens)
    block = triton.next_power_of_2(num_sequences)
    _prepare_ragged_chunk_offsets_kernel[(1,)](
        cu_seqlens,
        chunk_offsets,
        num_sequences=num_sequences,
        tokens=tokens,
        chunk_size=chunk_size,
        BLOCK=block,
        num_warps=min(8, max(1, block // 32)),
    )
    return chunk_offsets


def decode_ragged_chunk_work(metadata: RaggedChunkMetadata) -> torch.Tensor:
    """Materialize scheduler decisions for diagnostics and mapping tests."""
    work = torch.empty(
        (metadata.capacity, 5),
        dtype=torch.int32,
        device=metadata.cu_seqlens.device,
    )
    if metadata.capacity:
        _decode_ragged_chunk_work_kernel[(metadata.capacity,)](
            metadata.cu_seqlens,
            metadata.chunk_offsets,
            work,
            num_sequences=metadata.cu_seqlens.shape[0] - 1,
            chunk_size=metadata.chunk_size,
            num_warps=1,
        )
    return work


__all__ = [
    "ChunkWork",
    "GridScheduler",
    "RaggedChunkMetadata",
    "chunk_capacity",
    "chunk_work_oracle",
    "decode_ragged_chunk_work",
    "decode_ragged_task",
    "load_ragged_chunk_count",
    "load_ragged_chunk_work",
    "load_ragged_sequence_extent",
    "load_ragged_task_count",
    "prepare_ragged_chunk_metadata",
]
