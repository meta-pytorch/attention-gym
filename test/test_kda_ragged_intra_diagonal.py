"""Ragged scheduler tests for the KDA intra-chunk diagonal stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
    chunk_kda_fwd_intra_diagonal,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the KDA intra stage requires CUDA",
)


def _inputs(tokens: int):
    torch.manual_seed(7)
    q = torch.randn(1, tokens, 1, 128, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    g = -torch.rand(1, tokens, 1, 128, device="cuda")
    beta = torch.sigmoid(torch.randn(1, tokens, 1, device="cuda"))
    return q, k, g, beta


def _offsets(lengths: list[int]) -> list[int]:
    result = [0]
    for length in lengths:
        result.append(result[-1] + length)
    return result


def _written_aqk_mask(lengths: list[int]) -> torch.Tensor:
    mask = torch.zeros((sum(lengths), 64), dtype=torch.bool)
    sequence_start = 0
    for length in lengths:
        for chunk_start in range(0, length, 64):
            valid = min(64, length - chunk_start)
            for subchunk_start in range(0, valid, 16):
                rows = slice(
                    sequence_start + chunk_start + subchunk_start,
                    sequence_start + chunk_start + min(subchunk_start + 16, valid),
                )
                mask[rows, subchunk_start : subchunk_start + 16] = True
        sequence_start += length
    return mask


def _sequence_local_reference(inputs, lengths: list[int]):
    outputs_aqk = []
    outputs_akk = []
    start = 0
    for length in lengths:
        if length:
            local_inputs = tuple(tensor[:, start : start + length] for tensor in inputs)
            cu_seqlens = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            metadata = prepare_ragged_chunk_metadata(cu_seqlens, length, 64)
            Aqk, Akk = chunk_kda_fwd_intra_diagonal(
                *local_inputs,
                scale=128**-0.5,
                metadata=metadata,
            )
            outputs_aqk.append(Aqk)
            outputs_akk.append(Akk)
        start += length
    return torch.cat(outputs_aqk, dim=1), torch.cat(outputs_akk, dim=1)


@pytest.mark.parametrize("lengths", [[65, 63], [0, 1, 64, 0, 65]])
def test_ragged_intra_diagonal_matches_sequence_local_launches(lengths):
    tokens = sum(lengths)
    inputs = _inputs(tokens)
    cu_seqlens = torch.tensor(_offsets(lengths), device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)

    actual_aqk, actual_akk = chunk_kda_fwd_intra_diagonal(
        *inputs,
        scale=128**-0.5,
        metadata=metadata,
    )
    expected_aqk, expected_akk = _sequence_local_reference(inputs, lengths)

    mask = _written_aqk_mask(lengths).to(device="cuda")
    torch.testing.assert_close(actual_aqk[0, :, 0][mask], expected_aqk[0, :, 0][mask])
    torch.testing.assert_close(actual_akk, expected_akk)


def test_ragged_intra_diagonal_fullgraph():
    inputs = _inputs(128)
    cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)

    def operation(q, k, g, beta, offsets):
        metadata = prepare_ragged_chunk_metadata(offsets, 128, 64)
        return chunk_kda_fwd_intra_diagonal(
            q,
            k,
            g,
            beta,
            scale=128**-0.5,
            metadata=metadata,
        )

    expected_aqk, expected_akk = operation(*inputs, cu_seqlens)
    actual_aqk, actual_akk = torch.compile(operation, fullgraph=True)(*inputs, cu_seqlens)
    mask = _written_aqk_mask([65, 63]).to(device="cuda")
    torch.testing.assert_close(actual_aqk[0, :, 0][mask], expected_aqk[0, :, 0][mask])
    torch.testing.assert_close(actual_akk, expected_akk)


def test_ragged_intra_diagonal_replays_aligned_to_ragged():
    inputs = _inputs(128)
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    chunk_kda_fwd_intra_diagonal(
        *inputs,
        scale=128**-0.5,
        metadata=warm_metadata,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual_aqk, actual_akk = chunk_kda_fwd_intra_diagonal(
            *inputs,
            scale=128**-0.5,
            metadata=metadata,
        )

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected_aqk, expected_akk = _sequence_local_reference(inputs, [65, 63])
    mask = _written_aqk_mask([65, 63]).to(device="cuda")
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    torch.testing.assert_close(actual_aqk[0, :, 0][mask], expected_aqk[0, :, 0][mask])
    torch.testing.assert_close(actual_akk, expected_akk)
