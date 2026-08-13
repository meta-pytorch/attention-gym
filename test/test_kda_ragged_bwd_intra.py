"""Ragged scheduler tests for the CuTe KDA intra-chunk backward stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import (
    ChunkKdaBwdIntraConfig,
    chunk_kda_bwd_intra,
)
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.utils import ChunkMetadata, prepare_complete_chunk_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe KDA backward stage requires CUDA capability 10.0 or newer",
)


def _metadata(lengths: list[int]):
    """Build fixed-capacity routing for one packed length distribution."""
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return prepare_ragged_chunk_metadata(offsets, sum(lengths), 64)


def _inputs(tokens: int) -> tuple[torch.Tensor, ...]:
    """Create deterministic standalone inputs for the intra backward stage."""
    torch.manual_seed(29)
    shape = (1, tokens, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    g = -torch.rand(shape, device="cuda")
    beta = torch.rand(1, tokens, 1, device="cuda")
    dAqk = torch.randn(1, tokens, 1, 64, device="cuda") / 16
    dAkk = torch.randn_like(dAqk) / 16
    dq = torch.randn(shape, device="cuda") / 16
    dk = torch.randn_like(dq) / 16
    db = torch.randn_like(beta) / 16
    dg = torch.randn_like(g) / 16
    return q, k, g, beta, dAqk, dAkk, dq, dk, db, dg


def _run(inputs: tuple[torch.Tensor, ...], lengths: list[int]):
    """Run a ragged launch with the full fixed-capacity grid exposed."""
    metadata = _metadata(lengths)
    return chunk_kda_bwd_intra(
        *inputs,
        metadata,
        config=ChunkKdaBwdIntraConfig(metadata.capacity),
    )


def _sequence_local_reference(
    inputs: tuple[torch.Tensor, ...],
    lengths: list[int],
) -> tuple[torch.Tensor, ...]:
    """Concatenate independent per-sequence launches as the routing oracle."""
    output_parts: list[list[torch.Tensor]] = [[], [], [], []]
    begin = 0
    for length in lengths:
        if length == 0:
            continue
        end = begin + length
        local_inputs = tuple(tensor[:, begin:end].clone() for tensor in inputs)
        for parts, output in zip(output_parts, _run(local_inputs, [length]), strict=True):
            parts.append(output)
        begin = end
    return tuple(torch.cat(parts, dim=1) for parts in output_parts)


@pytest.mark.parametrize("lengths", [[65, 63], [0, 1, 64, 0, 65]])
def test_ragged_bwd_intra_matches_sequence_local_launches(lengths):
    inputs = _inputs(sum(lengths))
    metadata = _metadata(lengths)
    active_chunks = metadata.chunk_offsets[-1].item()
    assert active_chunks <= metadata.capacity
    if 0 in lengths:
        assert active_chunks < metadata.capacity

    config = ChunkKdaBwdIntraConfig(metadata.capacity) if 0 in lengths else None
    actual = chunk_kda_bwd_intra(*inputs, metadata, config=config)
    expected = _sequence_local_reference(inputs, lengths)

    for packed, sequence_local in zip(actual, expected, strict=True):
        torch.testing.assert_close(packed, sequence_local, rtol=0, atol=0)


def test_ragged_bwd_intra_accepts_all_empty_sequences():
    inputs = _inputs(0)
    dq, dk, dg, db = _run(inputs, [0, 0])

    assert dq.shape == dk.shape == dg.shape == (1, 0, 1, 128)
    assert dq.dtype == dk.dtype == torch.bfloat16
    assert dg.dtype == db.dtype == torch.float32
    torch.testing.assert_close(db, inputs[8], rtol=0, atol=0)


def test_ragged_bwd_intra_preserves_legacy_routing():
    inputs = _inputs(128)
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    chunk_indices, num_chunks = prepare_complete_chunk_metadata(cu_seqlens, 128, 64)
    legacy = ChunkMetadata(cu_seqlens, chunk_indices, num_chunks)

    expected = chunk_kda_bwd_intra(
        *inputs,
        legacy,
        config=ChunkKdaBwdIntraConfig(2),
    )
    actual = _run(inputs, [64, 64])

    for ragged, complete in zip(actual, expected, strict=True):
        torch.testing.assert_close(ragged, complete, rtol=0, atol=0)


def test_ragged_bwd_intra_replays_aligned_to_partial():
    inputs = _inputs(128)
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    config = ChunkKdaBwdIntraConfig(warm_metadata.capacity)
    chunk_kda_bwd_intra(*inputs, warm_metadata, config=config)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual = chunk_kda_bwd_intra(*inputs, metadata, config=config)

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected = _run(inputs, [65, 63])
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    for captured, eager in zip(actual, expected, strict=True):
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
