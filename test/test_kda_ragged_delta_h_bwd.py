"""Focused tests for scheduler-routed ragged KDA delta-H backward."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd_v1 import blackwell_delta_h_bwd_dhu_v1
from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd_v1_dispatch import (
    blackwell_delta_h_bwd_dhu_dispatch,
)
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTe delta-H backward stage requires SM100 or SM103",
)


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
    bv: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, w, do, dv, gk, h0, dht = inputs
    offsets = cumulative_sequence_offsets(lengths) if cu_seqlens is None else cu_seqlens
    metadata = prepare_ragged_chunk_metadata(offsets, q.shape[1], 64)
    options = {
        "gk": gk,
        "h0": h0,
        "dht": dht,
        "scale": 128**-0.5,
        "metadata": metadata,
    }
    if bv is not None:
        return blackwell_delta_h_bwd_dhu_v1(q, k, w, do, dv, bv=bv, **options)
    return blackwell_delta_h_bwd_dhu_dispatch(q, k, w, do, dv, **options)


def test_ragged_delta_h_handles_all_empty_sequences():
    inputs = _inputs(0, 2)
    dh, dh0, dv = _run_ragged(inputs, [0, 0])

    assert dh.shape == (1, 0, 1, 128, 128)
    assert dv.shape == inputs[4].shape
    torch.testing.assert_close(dh0, inputs[-1], rtol=0, atol=0)


def test_ragged_delta_h_rejects_mismatched_chunk_size():
    inputs = _inputs(128, 1)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([128]), 128, 128)

    with pytest.raises(ValueError, match="metadata chunk size"):
        blackwell_delta_h_bwd_dhu_dispatch(
            *inputs[:5],
            gk=inputs[5],
            h0=inputs[6],
            dht=inputs[7],
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


@pytest.mark.parametrize("bv", [16, 32])
@pytest.mark.parametrize(
    "lengths", [[130, 0, 0], [63, 130]], ids=["small-partial", "large-partial"]
)
def test_ragged_delta_h_ignores_nan_poisoned_physical_suffix(bv, lengths):
    physical_tokens = 256
    active_tokens = sum(lengths)
    inputs = _inputs(physical_tokens, len(lengths))
    active_inputs = tuple(value[:, :active_tokens].clone() for value in inputs[:6]) + tuple(
        value.clone() for value in inputs[6:]
    )
    for value in inputs[:6]:
        value[:, active_tokens:].fill_(torch.nan)

    dh, dh0, dv = _run_ragged(inputs, lengths, bv=bv)
    expected_dh, expected_dh0, expected_dv = _run_ragged(active_inputs, lengths, bv=bv)

    active_chunks = sum((length + 63) // 64 for length in lengths)
    torch.testing.assert_close(
        dh[:, :active_chunks],
        expected_dh[:, :active_chunks],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(dh0, expected_dh0, rtol=0, atol=0)
    torch.testing.assert_close(
        dv[:, :active_tokens],
        expected_dv[:, :active_tokens],
        rtol=0,
        atol=0,
    )


def test_ragged_delta_h_preserves_legacy_packed_arguments():
    lengths = [64, 64]
    inputs = _inputs(sum(lengths), len(lengths))
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(lengths), sum(lengths), 64
    )
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
    cu_seqlens = cumulative_sequence_offsets([64, 64])

    _run_ragged(inputs, [64, 64], cu_seqlens)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_dh, captured_dh0, captured_dv = _run_ragged(
            inputs,
            [64, 64],
            cu_seqlens,
        )

    cu_seqlens.copy_(cumulative_sequence_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    expected_dh, expected_dh0, expected_dv = _run_ragged(inputs, [65, 63])
    assert captured_dh.shape[1] == 3
    torch.testing.assert_close(captured_dh, expected_dh, rtol=0, atol=0)
    torch.testing.assert_close(captured_dh0, expected_dh0, rtol=0, atol=0)
    torch.testing.assert_close(captured_dv, expected_dv, rtol=0, atol=0)
