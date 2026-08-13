"""CuTeDSL broadcast tests for the packed KDA chunk scheduler."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.kda.chunk_scheduler import (
    chunk_work_oracle,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import (
    decode_ragged_chunk_work_cute,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL KDA scheduler test requires CUDA capability 10.0 or newer",
)


def _expected(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    expected = torch.tensor(
        [tuple(work) for work in chunk_work_oracle(offsets, 64)],
        dtype=torch.int32,
    )
    return torch.tensor(offsets, device="cuda", dtype=torch.int32), expected


@pytest.mark.parametrize(
    "lengths",
    [
        [65, 63],
        [0, 1, 0, 63, 64, 65, 0],
        [1] * 257,
        [4097, 1, 511, 0, 65],
    ],
)
def test_cute_chunk_scheduler_broadcast_matches_oracle(lengths):
    cu_seqlens, expected = _expected(lengths)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, int(cu_seqlens[-1]), 64)
    actual = decode_ragged_chunk_work_cute(
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


def test_cute_chunk_scheduler_cuda_graph_replays_boundaries():
    cu_seqlens, _ = _expected([65, 63])
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    decode_ragged_chunk_work_cute(cu_seqlens, metadata.chunk_offsets, metadata.capacity)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual = decode_ragged_chunk_work_cute(
            cu_seqlens,
            captured_metadata.chunk_offsets,
            captured_metadata.capacity,
        )

    cu_seqlens.copy_(torch.tensor([0, 1, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    _, expected = _expected([1, 127])
    for warp in range(actual.shape[1]):
        torch.testing.assert_close(actual[: expected.shape[0], warp].cpu(), expected)
        torch.testing.assert_close(
            actual[expected.shape[0] :, warp].cpu(),
            torch.full_like(actual[expected.shape[0] :, warp].cpu(), -1),
        )
