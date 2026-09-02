"""Tests for graph-safe packed KDA logical chunk scheduling."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl

from attn_gym.linear.kda.chunk_scheduler import (
    PERSISTENT_AUTO_WAVES,
    PERSISTENT_CTAS_PER_SM,
    GridScheduler,
    RaggedChunkMetadata,
    ResolvedSchedule,
    ScheduleKind,
    ScheduleRequest,
    chunk_capacity,
    chunk_work_oracle,
    decode_ragged_chunk_work,
    decode_ragged_task,
    load_ragged_chunk_count,
    load_ragged_task_count,
    prepare_ragged_chunk_metadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the KDA chunk scheduler requires CUDA",
)


@triton.jit
def _load_ragged_task_count_kernel(chunk_offsets, output, subtasks_per_chunk):
    tl.store(output, load_ragged_task_count(chunk_offsets, 0, subtasks_per_chunk))


@triton.jit
def _decode_ragged_task_kernel(task, subtasks_per_chunk, output):
    global_chunk, subtask = decode_ragged_task(task, subtasks_per_chunk)
    tl.store(output, global_chunk)
    tl.store(output + 1, subtask)


@triton.jit
def _load_ragged_chunk_count_kernel(chunk_offsets, output):
    tl.store(output, load_ragged_chunk_count(chunk_offsets, 0))


def _expected_tensor(lengths: list[int], chunk_size: int) -> tuple[torch.Tensor, list[int]]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    rows = [tuple(work) for work in chunk_work_oracle(offsets, chunk_size)]
    return torch.tensor(rows, dtype=torch.int32), offsets


@pytest.mark.parametrize(
    "lengths",
    [
        [1],
        [63],
        [64],
        [65],
        [127],
        [128],
        [129],
        [65, 63],
        [0, 1, 0, 63, 64, 65, 0],
        [1] * 257,
        [65] * 1024,
        [4097, 1, 511, 0, 65],
    ],
)
def test_ragged_chunk_scheduler_matches_cpu_oracle(lengths):
    expected, offsets = _expected_tensor(lengths, 64)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, offsets[-1], 64)

    assert metadata.chunk_size == 64
    assert metadata.capacity == chunk_capacity(offsets[-1], len(lengths), 64)
    assert metadata.chunk_offsets[-1].item() == expected.shape[0]
    _assert_work_matches(decode_ragged_chunk_work(metadata), expected)


@pytest.mark.parametrize("chunk_size", [16, 32, 64])
def test_ragged_chunk_scheduler_preserves_chunk_size(chunk_size):
    lengths = [chunk_size + 1, chunk_size - 1]
    expected, offsets = _expected_tensor(lengths, chunk_size)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, offsets[-1], chunk_size)

    assert metadata.chunk_size == chunk_size
    torch.testing.assert_close(decode_ragged_chunk_work(metadata).cpu(), expected)


def _assert_work_matches(work: torch.Tensor, expected: torch.Tensor) -> None:
    """Active rows match the CPU oracle; the capacity filler rows are -1."""
    active_chunks = expected.shape[0]
    torch.testing.assert_close(work[:active_chunks].cpu(), expected)
    filler = work[active_chunks:].cpu()
    torch.testing.assert_close(filler, torch.full_like(filler, -1))


@pytest.mark.parametrize(
    ("initial_offsets", "replay_offsets", "replay_lengths"),
    (
        pytest.param([0, 65, 128], [0, 1, 128], [1, 127], id="boundaries"),
        pytest.param([0, 64, 128], [0, 32, 65], [32, 33], id="active-token-count"),
    ),
)
def test_ragged_chunk_scheduler_cuda_graph_replay(initial_offsets, replay_offsets, replay_lengths):
    """Reread boundaries and the active endpoint from device memory on replay."""
    cu_seqlens = torch.tensor(initial_offsets, device="cuda", dtype=torch.int32)
    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, initial_offsets[-1], 64)
    decode_ragged_chunk_work(warm_metadata)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, initial_offsets[-1], 64)
        work = decode_ragged_chunk_work(metadata)

    cu_seqlens.copy_(torch.tensor(replay_offsets, device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected, _ = _expected_tensor(replay_lengths, 64)
    assert metadata.chunk_offsets[-1].item() == expected.shape[0]
    _assert_work_matches(work, expected)


def test_ragged_chunk_scheduler_fullgraph():
    cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)

    def operation(offsets):
        metadata = prepare_ragged_chunk_metadata(offsets, 128, 64)
        return metadata.chunk_offsets, decode_ragged_chunk_work(metadata)

    expected_offsets, expected_work = operation(cu_seqlens)
    actual_offsets, actual_work = torch.compile(operation, fullgraph=True)(cu_seqlens)
    torch.testing.assert_close(actual_offsets, expected_offsets)
    torch.testing.assert_close(actual_work, expected_work)


def test_ragged_chunk_capacity_is_tight_for_empty_sequences():
    assert chunk_capacity(0, 1024, 64) == 0
    assert chunk_capacity(1, 1024, 64) == 1
    assert chunk_capacity(65, 64, 64) == 64
    assert chunk_capacity(128, 2, 64) == 3


def test_ragged_chunk_scheduler_accepts_zero_capacity():
    cu_seqlens = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 0, 64)
    actual = decode_ragged_chunk_work(metadata)

    assert metadata.capacity == 0
    assert actual.shape == (0, 5)


def test_ragged_chunk_scheduler_accepts_more_than_4096_sequences():
    num_sequences = 4097
    cu_seqlens = torch.arange(num_sequences + 1, device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, num_sequences, 64)
    actual = decode_ragged_chunk_work(metadata)

    assert metadata.capacity == num_sequences
    torch.testing.assert_close(metadata.chunk_offsets, cu_seqlens)
    torch.testing.assert_close(actual[:, 0], cu_seqlens[:-1])
    torch.testing.assert_close(actual[:, 1], cu_seqlens[:-1])
    torch.testing.assert_close(actual[:, 2], torch.zeros_like(cu_seqlens[:-1]))
    torch.testing.assert_close(actual[:, 3], cu_seqlens[:-1])
    torch.testing.assert_close(actual[:, 4], torch.ones_like(cu_seqlens[:-1]))


@pytest.mark.parametrize(
    "boundaries,tokens",
    [
        ([1, 64], 64),
        ([0, 65, 64], 64),
        ([0, -1, 64], 64),
        ([0, 65], 64),
    ],
)
def test_ragged_chunk_scheduler_rejects_invalid_boundaries(boundaries, tokens):
    source = f"""
import os
import torch
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata

try:
    offsets = torch.tensor({boundaries!r}, device="cuda", dtype=torch.int32)
    prepare_ragged_chunk_metadata(offsets, {tokens}, 64)
    torch.cuda.synchronize()
except RuntimeError as error:
    if "device-side assert" in str(error).lower():
        os._exit(0)
    os._exit(2)
os._exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).parents[1],
        env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _decode_ragged_chunk_work_cute(*args, **kwargs):
    pytest.importorskip("cutlass")
    from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import (
        decode_ragged_chunk_work_cute,
    )

    return decode_ragged_chunk_work_cute(*args, **kwargs)


requires_cute = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL KDA scheduler test requires an SM100 or SM103 GPU",
)


@pytest.mark.parametrize(
    "lengths",
    [
        [65, 63],
        [0, 1, 0, 63, 64, 65, 0],
        [1] * 257,
        [4097, 1, 511, 0, 65],
    ],
)
@requires_cute
def test_cute_chunk_scheduler_broadcast_matches_oracle(lengths):
    expected, offsets = _expected_tensor(lengths, 64)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, offsets[-1], 64)
    actual = _decode_ragged_chunk_work_cute(
        cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
    ).cpu()

    for warp in range(actual.shape[1]):
        torch.testing.assert_close(actual[: expected.shape[0], warp], expected)
        torch.testing.assert_close(
            actual[expected.shape[0] :, warp],
            torch.full_like(actual[expected.shape[0] :, warp], -1),
        )


@requires_cute
def test_cute_chunk_scheduler_accepts_zero_capacity():
    cu_seqlens = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 0, 64)
    actual = _decode_ragged_chunk_work_cute(
        cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
    )

    assert actual.shape == (0, 4, 5)


@requires_cute
def test_cute_chunk_scheduler_cuda_graph_replays_boundaries():
    _, offsets = _expected_tensor([65, 63], 64)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    _decode_ragged_chunk_work_cute(
        cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual = _decode_ragged_chunk_work_cute(
            cu_seqlens,
            captured_metadata.chunk_offsets,
            captured_metadata.capacity,
        )

    cu_seqlens.copy_(torch.tensor([0, 1, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected, _ = _expected_tensor([1, 127], 64)
    for warp in range(actual.shape[1]):
        torch.testing.assert_close(actual[: expected.shape[0], warp].cpu(), expected)
        torch.testing.assert_close(
            actual[expected.shape[0] :, warp].cpu(),
            torch.full_like(actual[expected.shape[0] :, warp].cpu(), -1),
        )


def test_grid_scheduler_persistent_grid_is_machine_derived():
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    metadata = RaggedChunkMetadata(None, None, capacity=1 << 20, chunk_size=64)
    scheduler = GridScheduler(metadata)
    assert scheduler.num_workers(6, "cuda") == sm_count * PERSISTENT_CTAS_PER_SM
    # Tiny capacity never over-launches workers, and zero capacity launches nothing.
    assert GridScheduler(metadata._replace(capacity=2)).num_workers(3, "cuda") == 6
    assert GridScheduler(metadata._replace(capacity=0)).num_workers(3, "cuda") == 0


def test_grid_scheduler_sequence_workers_use_sequence_capacity():
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    cu_seqlens = torch.empty(513, device="cuda", dtype=torch.int32)
    metadata = RaggedChunkMetadata(cu_seqlens, None, capacity=1, chunk_size=64)
    scheduler = GridScheduler(metadata, ctas_per_sm=1)

    resolved = scheduler.resolve_sequences(ScheduleRequest.PERSISTENT, 6, "cuda")
    assert resolved == ResolvedSchedule(
        ScheduleKind.PERSISTENT,
        min(512 * 6, sm_count),
        512 * 6,
    )


def test_grid_scheduler_chunk_workers_cap_at_sm_count():
    sm_count = torch.cuda.get_device_properties("cuda").multi_processor_count
    metadata = RaggedChunkMetadata(None, None, capacity=1 << 20, chunk_size=64)
    scheduler = GridScheduler(metadata)
    assert scheduler.num_chunk_workers("cuda") == sm_count
    # Sub-SM capacities quantize up to the next power of two to bound the
    # TVM-FFI compile cache; extra workers idle in the stride loop.
    assert GridScheduler(metadata._replace(capacity=3)).num_chunk_workers("cuda") == 4
    assert GridScheduler(metadata._replace(capacity=0)).num_chunk_workers("cuda") == 0


def test_persistent_task_count_promotes_before_multiplication():
    chunk_offsets = torch.tensor([1_500_000_000], device="cuda", dtype=torch.int32)
    output = torch.empty((), device="cuda", dtype=torch.int64)
    _load_ragged_task_count_kernel[(1,)](chunk_offsets, output, 2)
    assert output.item() == 3_000_000_000


def test_ragged_task_helpers_load_and_decode_flat_work():
    chunk_offsets = torch.tensor([4], device="cuda", dtype=torch.int32)
    count = torch.empty((), device="cuda", dtype=torch.int64)
    decoded = torch.empty(2, device="cuda", dtype=torch.int64)

    _load_ragged_chunk_count_kernel[(1,)](chunk_offsets, count)
    _decode_ragged_task_kernel[(1,)](3_000_000_001, 3, decoded)

    assert count.item() == 4
    assert decoded.tolist() == [1_000_000_000, 1]


@pytest.mark.parametrize(
    ("schedule_request", "eligible", "capacity_tasks", "expected_kind", "raises"),
    (
        (ScheduleRequest.AUTO, True, 400, ScheduleKind.STATIC, False),
        (ScheduleRequest.AUTO, True, 401, ScheduleKind.PERSISTENT, False),
        (ScheduleRequest.AUTO, False, 401, ScheduleKind.STATIC, False),
        (ScheduleRequest.STATIC, True, 401, ScheduleKind.STATIC, False),
        (ScheduleRequest.STATIC, False, 401, ScheduleKind.STATIC, False),
        (ScheduleRequest.PERSISTENT, True, 1, ScheduleKind.PERSISTENT, False),
        (ScheduleRequest.PERSISTENT, True, 0, ScheduleKind.STATIC, False),
        (ScheduleRequest.PERSISTENT, False, 401, None, True),
        (ScheduleRequest.PERSISTENT, False, 0, None, True),
    ),
)
def test_grid_scheduler_resolves_flat_plan_contract(
    monkeypatch,
    schedule_request,
    eligible,
    capacity_tasks,
    expected_kind,
    raises,
):
    workers = 100
    assert PERSISTENT_AUTO_WAVES * workers == 400
    metadata = RaggedChunkMetadata(None, None, capacity_tasks, 64)
    scheduler = GridScheduler(metadata)
    monkeypatch.setattr(GridScheduler, "num_workers", lambda self, subtasks, device: workers)

    if raises:
        with pytest.raises(ValueError, match="persistent scheduling requires"):
            scheduler.resolve_flat(schedule_request, 1, "cuda", eligible=eligible)
        return

    resolved = scheduler.resolve_flat(schedule_request, 1, "cuda", eligible=eligible)
    assert resolved == ResolvedSchedule(expected_kind, workers, capacity_tasks)


def test_grid_scheduler_rejects_legacy_boolean_requests(monkeypatch):
    metadata = RaggedChunkMetadata(None, None, capacity=1, chunk_size=64)
    monkeypatch.setattr(GridScheduler, "num_workers", lambda self, subtasks, device: 1)

    with pytest.raises(TypeError, match="ScheduleRequest"):
        GridScheduler(metadata).resolve_flat(True, 1, "cuda")


@pytest.mark.parametrize(
    ("schedule_request", "capacity", "expected_kind"),
    (
        (ScheduleRequest.AUTO, 256, ScheduleKind.STATIC),
        (ScheduleRequest.AUTO, 257, ScheduleKind.PERSISTENT),
        (ScheduleRequest.STATIC, 512, ScheduleKind.STATIC),
        (ScheduleRequest.PERSISTENT, 512, ScheduleKind.PERSISTENT),
        (ScheduleRequest.PERSISTENT, 0, ScheduleKind.STATIC),
    ),
)
def test_grid_scheduler_chunk_plan_uses_the_same_resolved_type(
    monkeypatch, schedule_request, capacity, expected_kind
):
    metadata = RaggedChunkMetadata(None, None, capacity=capacity, chunk_size=64)
    monkeypatch.setattr(
        GridScheduler,
        "num_chunk_workers",
        lambda self, device: 0 if self.metadata.capacity == 0 else 64,
    )

    resolved = GridScheduler(metadata).resolve_chunk(schedule_request, "cuda")

    expected_workers = 0 if capacity == 0 else 64
    assert resolved == ResolvedSchedule(expected_kind, expected_workers, capacity)
