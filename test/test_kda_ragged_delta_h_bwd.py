"""Focused tests for scheduler-routed ragged KDA delta-H backward."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd_v1_dispatch import (
    blackwell_delta_h_bwd_dhu_dispatch,
)
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe delta-H backward stage requires CUDA capability 10.0 or newer",
)


def _offsets(lengths: list[int]) -> torch.Tensor:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device="cuda", dtype=torch.int32)


def _inputs(tokens: int, sequences: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(37)
    shape = (1, tokens, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    w = torch.randn_like(q) / 8
    do = torch.randn_like(q) / 8
    dv = torch.randn_like(q) / 8
    gk = -torch.rand(shape, device="cuda")
    h0 = torch.randn(sequences, 1, 128, 128, device="cuda") / 8
    dht = torch.randn_like(h0) / 8
    return q, k, w, do, dv, gk, h0, dht


def _run_ragged(
    inputs: tuple[torch.Tensor, ...],
    lengths: list[int],
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, w, do, dv, gk, h0, dht = inputs
    offsets = _offsets(lengths) if cu_seqlens is None else cu_seqlens
    metadata = prepare_ragged_chunk_metadata(offsets, q.shape[1], 64)
    return blackwell_delta_h_bwd_dhu_dispatch(
        q,
        k,
        w,
        do,
        dv,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=128**-0.5,
        metadata=metadata,
    )


def test_ragged_delta_h_matches_independent_sequences():
    lengths = [65, 0, 63]
    inputs = _inputs(sum(lengths), len(lengths))
    dh, dh0, dv = _run_ragged(inputs, lengths)

    expected_dh = []
    expected_dh0 = []
    expected_dv = torch.empty_like(dv)
    begin = 0
    for sequence, length in enumerate(lengths):
        end = begin + length
        if length == 0:
            expected_dh0.append(inputs[-1][sequence])
        else:
            sequence_inputs = tuple(value[:, begin:end] for value in inputs[:6]) + (
                inputs[6][sequence : sequence + 1],
                inputs[7][sequence : sequence + 1],
            )
            sequence_dh, sequence_dh0, sequence_dv = _run_ragged(
                sequence_inputs,
                [length],
            )
            expected_dh.append(sequence_dh)
            expected_dh0.append(sequence_dh0[0])
            expected_dv[:, begin:end] = sequence_dv
        begin = end

    active_chunks = sum((length + 63) // 64 for length in lengths)
    assert dh.shape[1] == 4
    torch.testing.assert_close(
        dh[:, :active_chunks],
        torch.cat(expected_dh, dim=1),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(dh0, torch.stack(expected_dh0), rtol=0, atol=0)
    torch.testing.assert_close(dv, expected_dv, rtol=0, atol=0)


def test_ragged_delta_h_preserves_legacy_packed_arguments():
    lengths = [64, 64]
    inputs = _inputs(sum(lengths), len(lengths))
    metadata = prepare_ragged_chunk_metadata(_offsets(lengths), sum(lengths), 64)
    metadata_outputs = blackwell_delta_h_bwd_dhu_dispatch(
        *inputs[:5],
        gk=inputs[5],
        h0=inputs[6],
        dht=inputs[7],
        scale=128**-0.5,
        metadata=metadata,
    )
    legacy_outputs = blackwell_delta_h_bwd_dhu_dispatch(
        *inputs[:5],
        gk=inputs[5],
        h0=inputs[6],
        dht=inputs[7],
        scale=128**-0.5,
        cu_seqlens=metadata.cu_seqlens,
        chunk_offsets=metadata.chunk_offsets,
        num_seqs=metadata.cu_seqlens.new_full((1,), len(lengths)),
        num_chunks=2,
    )

    metadata_dh, metadata_dh0, metadata_dv = metadata_outputs
    legacy_dh, legacy_dh0, legacy_dv = legacy_outputs
    assert metadata_dh.shape[1] == 3
    assert legacy_dh.shape[1] == 2
    torch.testing.assert_close(metadata_dh[:, :2], legacy_dh, rtol=0, atol=0)
    torch.testing.assert_close(metadata_dh0, legacy_dh0, rtol=0, atol=0)
    torch.testing.assert_close(metadata_dv, legacy_dv, rtol=0, atol=0)


def test_ragged_delta_h_replays_aligned_to_tails():
    tokens = 128
    inputs = _inputs(tokens, 2)
    cu_seqlens = _offsets([64, 64])

    _run_ragged(inputs, [64, 64], cu_seqlens)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_dh, captured_dh0, captured_dv = _run_ragged(
            inputs,
            [64, 64],
            cu_seqlens,
        )

    cu_seqlens.copy_(_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    expected_dh, expected_dh0, expected_dv = _run_ragged(inputs, [65, 63])
    assert captured_dh.shape[1] == 3
    torch.testing.assert_close(captured_dh, expected_dh, rtol=0, atol=0)
    torch.testing.assert_close(captured_dh0, expected_dh0, rtol=0, atol=0)
    torch.testing.assert_close(captured_dv, expected_dv, rtol=0, atol=0)
