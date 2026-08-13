"""Ragged scheduling tests for the Triton KDA dAv backward stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_dav import chunk_kda_bwd_dav
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.utils import ChunkMetadata, prepare_complete_chunk_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the KDA dAv backward kernel requires CUDA",
)

_CHUNK_SIZE = 64
_SCALE = 128**-0.5


def _offsets(lengths: list[int]) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device="cuda", dtype=torch.int32)


def _inputs(tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    shape = (1, tokens, 1, 128)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    A = torch.randn(1, tokens, 1, _CHUNK_SIZE, device="cuda", dtype=torch.bfloat16) / 8
    do = torch.randn_like(v) / 8
    return v, A, do


def _run_ragged(
    v: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = prepare_ragged_chunk_metadata(_offsets(lengths), v.shape[1], _CHUNK_SIZE)
    return chunk_kda_bwd_dav(v, A, do, _SCALE, metadata=metadata)


def _independent_sequences(
    v: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    dv_parts = []
    dA_parts = []
    token_start = 0
    for length in lengths:
        token_end = token_start + length
        if length:
            dv, dA = _run_ragged(
                v[:, token_start:token_end],
                A[:, token_start:token_end],
                do[:, token_start:token_end],
                [length],
            )
            dv_parts.append(dv)
            dA_parts.append(dA)
        token_start = token_end
    return torch.cat(dv_parts, dim=1), torch.cat(dA_parts, dim=1)


@pytest.mark.parametrize("lengths", [[65, 63], [0, 1, 64, 0, 65]])
def test_ragged_bwd_dav_matches_independent_sequences(lengths):
    inputs = _inputs(sum(lengths))
    actual = _run_ragged(*inputs, lengths)
    expected = _independent_sequences(*inputs, lengths)

    for packed, independent in zip(actual, expected, strict=True):
        torch.testing.assert_close(packed, independent, rtol=0, atol=0)


def test_legacy_chunk_map_path_is_preserved():
    lengths = [64, 0, 64]
    inputs = _inputs(sum(lengths))
    cu_seqlens = _offsets(lengths)
    chunk_indices, num_chunks = prepare_complete_chunk_metadata(
        cu_seqlens,
        sum(lengths),
        _CHUNK_SIZE,
    )
    metadata = ChunkMetadata(cu_seqlens, chunk_indices, num_chunks)

    actual = chunk_kda_bwd_dav(*inputs, _SCALE, metadata=metadata)
    expected = _independent_sequences(*inputs, lengths)
    for packed, independent in zip(actual, expected, strict=True):
        torch.testing.assert_close(packed, independent, rtol=0, atol=0)


def test_ragged_bwd_dav_fullgraph():
    inputs = _inputs(128)
    cu_seqlens = _offsets([65, 63])

    def operation(v, A, do, offsets):
        metadata = prepare_ragged_chunk_metadata(offsets, 128, _CHUNK_SIZE)
        return chunk_kda_bwd_dav(v, A, do, _SCALE, metadata=metadata)

    expected = operation(*inputs, cu_seqlens)
    actual = torch.compile(operation, fullgraph=True)(*inputs, cu_seqlens)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-6, atol=1e-7)


def test_ragged_bwd_dav_replays_aligned_to_ragged():
    inputs = _inputs(128)
    cu_seqlens = _offsets([64, 64])
    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, _CHUNK_SIZE)
    chunk_kda_bwd_dav(*inputs, _SCALE, metadata=warm_metadata)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, _CHUNK_SIZE)
        actual = chunk_kda_bwd_dav(*inputs, _SCALE, metadata=metadata)

    cu_seqlens.copy_(_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    expected = _independent_sequences(*inputs, [65, 63])
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    for replayed, independent in zip(actual, expected, strict=True):
        torch.testing.assert_close(replayed, independent, rtol=0, atol=0)
