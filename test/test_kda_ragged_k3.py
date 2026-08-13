"""Ragged scheduler tests for the CuTe KDA K3 off-diagonal stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    _chunk_kda_fwd_k3b_ragged_custom_op,
    chunk_kda_fwd_k3b_ragged_cute,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe K3 kernel requires CUDA capability 10.0 or newer",
)


def _offsets(lengths: list[int]) -> list[int]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def _inputs(tokens: int):
    torch.manual_seed(19)
    q = torch.randn(1, tokens, 1, 128, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    g = -torch.rand(1, tokens, 1, 128, device="cuda")
    beta = torch.sigmoid(torch.randn(1, tokens, 1, device="cuda"))
    Aqk = torch.zeros(1, tokens, 1, 64, device="cuda", dtype=torch.bfloat16)
    return q, k, g, beta, Aqk


def _run(inputs, lengths: list[int]):
    tokens = sum(lengths)
    cu_seqlens = torch.tensor(_offsets(lengths), device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    Aqk, AkkOD = chunk_kda_fwd_k3b_ragged_cute(
        *inputs,
        scale=128**-0.5,
        metadata=metadata,
    )
    return metadata, Aqk, AkkOD


def _sequence_local_reference(inputs, lengths: list[int]):
    Aqk_parts = []
    AkkOD_parts = []
    start = 0
    for length in lengths:
        if length:
            local_inputs = tuple(tensor[:, start : start + length].clone() for tensor in inputs)
            metadata, Aqk, AkkOD = _run(local_inputs, [length])
            active_rows = int(metadata.chunk_offsets[-1]) * 6
            Aqk_parts.append(Aqk)
            AkkOD_parts.append(AkkOD[:active_rows])
        start += length
    return torch.cat(Aqk_parts, dim=1), torch.cat(AkkOD_parts)


@pytest.mark.parametrize("lengths", [[65, 63], [0, 1, 64, 0, 65]])
def test_ragged_k3_matches_sequence_local_launches(lengths):
    inputs = _inputs(sum(lengths))
    metadata, actual_aqk, actual_akk_od = _run(inputs, lengths)
    expected_aqk, expected_akk_od = _sequence_local_reference(inputs, lengths)
    active_rows = int(metadata.chunk_offsets[-1]) * 6

    torch.testing.assert_close(actual_aqk, expected_aqk)
    torch.testing.assert_close(actual_akk_od[:active_rows], expected_akk_od)
    torch.testing.assert_close(
        actual_akk_od[active_rows:],
        torch.zeros_like(actual_akk_od[active_rows:]),
    )


@pytest.mark.parametrize("lengths", [[0], [0, 0, 0]])
def test_ragged_k3_accepts_all_empty_sequences(lengths):
    metadata, Aqk, AkkOD = _run(_inputs(0), lengths)

    assert Aqk.shape == (1, 0, 1, 64)
    assert AkkOD.shape == (metadata.capacity * 6, 256)
    torch.testing.assert_close(AkkOD, torch.zeros_like(AkkOD))


def test_ragged_k3_custom_op_and_fullgraph():
    inputs = _inputs(128)
    cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    op_args = (
        *inputs,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        128**-0.5,
        metadata.capacity,
    )
    torch.library.opcheck(_chunk_kda_fwd_k3b_ragged_custom_op, op_args)

    def operation(q, k, g, beta, Aqk, offsets):
        graph_metadata = prepare_ragged_chunk_metadata(offsets, 128, 64)
        return chunk_kda_fwd_k3b_ragged_cute(
            q,
            k,
            g,
            beta,
            Aqk,
            scale=128**-0.5,
            metadata=graph_metadata,
        )

    expected = operation(*inputs, cu_seqlens)
    actual = torch.compile(operation, fullgraph=True)(*inputs, cu_seqlens)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_ragged_k3_replays_aligned_to_ragged():
    inputs = _inputs(128)
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    chunk_kda_fwd_k3b_ragged_cute(
        *inputs,
        scale=128**-0.5,
        metadata=warm_metadata,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual_aqk, actual_akk_od = chunk_kda_fwd_k3b_ragged_cute(
            *inputs,
            scale=128**-0.5,
            metadata=metadata,
        )

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected_aqk, expected_akk_od = _sequence_local_reference(inputs, [65, 63])
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    torch.testing.assert_close(actual_aqk, expected_aqk)
    torch.testing.assert_close(actual_akk_od, expected_akk_od)
