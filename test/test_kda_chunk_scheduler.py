"""Tests for graph-safe packed KDA logical chunk scheduling."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import (
    chunk_capacity,
    chunk_work_oracle,
    decode_ragged_chunk_work,
    prepare_ragged_chunk_metadata,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the KDA chunk scheduler requires CUDA",
)


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
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL KDA scheduler test requires CUDA capability 10.0 or newer",
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
