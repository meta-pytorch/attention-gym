"""Tests for graph-safe packed KDA logical chunk scheduling."""

from __future__ import annotations

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
    actual = decode_ragged_chunk_work(metadata, 64).cpu()

    active_chunks = expected.shape[0]
    assert metadata.capacity == chunk_capacity(offsets[-1], len(lengths), 64)
    assert metadata.chunk_offsets[-1].item() == active_chunks
    torch.testing.assert_close(actual[:active_chunks], expected)
    torch.testing.assert_close(
        actual[active_chunks:],
        torch.full_like(actual[active_chunks:], -1),
    )


def test_ragged_chunk_scheduler_cuda_graph_replays_boundaries():
    cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    warm_work = decode_ragged_chunk_work(metadata, 64)
    del warm_work
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        work = decode_ragged_chunk_work(captured_metadata, 64)

    cu_seqlens.copy_(torch.tensor([0, 1, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected, _ = _expected_tensor([1, 127], 64)
    active_chunks = expected.shape[0]
    assert captured_metadata.chunk_offsets[-1].item() == active_chunks
    torch.testing.assert_close(work[:active_chunks].cpu(), expected)
    torch.testing.assert_close(
        work[active_chunks:].cpu(),
        torch.full_like(work[active_chunks:].cpu(), -1),
    )


def test_ragged_chunk_scheduler_fullgraph():
    cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)

    def operation(offsets):
        metadata = prepare_ragged_chunk_metadata(offsets, 128, 64)
        return metadata.chunk_offsets, decode_ragged_chunk_work(metadata, 64)

    expected_offsets, expected_work = operation(cu_seqlens)
    actual_offsets, actual_work = torch.compile(operation, fullgraph=True)(cu_seqlens)
    torch.testing.assert_close(actual_offsets, expected_offsets)
    torch.testing.assert_close(actual_work, expected_work)


def test_ragged_chunk_scheduler_replays_aligned_to_ragged():
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        work = decode_ragged_chunk_work(metadata, 64)

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected, _ = _expected_tensor([65, 63], 64)
    torch.testing.assert_close(work.cpu(), expected)
