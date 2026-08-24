"""Focused tests for scheduler-routed ragged KDA delta-H backward."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.kda.bwd.cute import chunk_delta_h_bwd as delta_h_module
from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd import (
    _blackwell_delta_h_bwd_dhu_dv_fused_packed,
    blackwell_delta_h_bwd_dhu_dv_fused_dispatch,
    should_bound_sequence_extent,
)
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, prepare_ragged_chunk_metadata
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTe delta-H backward stage requires SM100 or SM103",
)


def _inputs(
    tokens: int,
    sequences: int,
    heads: int = 1,
    lengths: list[int] | None = None,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(37)
    shape = (1, tokens, heads, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    w = torch.randn_like(q) / 8
    do = torch.randn_like(q) / 8
    aqk = torch.randn(1, tokens, heads, 64, device="cuda", dtype=torch.bfloat16) / 8
    begin = 0
    for sequence_length in lengths or [tokens]:
        for offset in range(0, sequence_length, 64):
            length = min(64, sequence_length - offset)
            token = slice(begin + offset, begin + offset + length)
            aqk[:, token, :, :length] *= torch.ones(
                length, length, device="cuda", dtype=torch.bool
            ).tril()[None, :, None, :]
        begin += sequence_length
    gk = -torch.rand(shape, device="cuda")
    h0 = torch.randn(sequences, heads, 128, 128, device="cuda") / 8
    dht = torch.randn_like(h0) / 8
    return q, k, w, do, aqk, gk, h0, dht


def _run_ragged(
    inputs: tuple[torch.Tensor, ...],
    lengths: list[int],
    cu_seqlens: torch.Tensor | None = None,
    bv: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, w, do, aqk, gk, h0, dht = inputs
    offsets = cumulative_sequence_offsets(lengths) if cu_seqlens is None else cu_seqlens
    metadata = prepare_ragged_chunk_metadata(offsets, q.shape[1], 64)
    if bv is None:
        return blackwell_delta_h_bwd_dhu_dv_fused_dispatch(
            q, k, w, do, aqk, gk=gk, h0=h0, dht=dht, scale=128**-0.5, metadata=metadata
        )
    return _blackwell_delta_h_bwd_dhu_dv_fused_packed(
        q,
        k,
        w,
        do,
        aqk,
        metadata,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=128**-0.5,
        bv=bv,
    )


@pytest.mark.parametrize(
    ("tokens", "sequences", "heads", "has_initial_state", "expected"),
    (
        (2048, 31, 16, False, False),
        (2048, 32, 7, False, False),
        (2048, 32, 8, False, True),
        (2049, 32, 8, False, False),
        (2048, 512, 16, True, False),
    ),
)
def test_delta_h_sequence_extent_selector(tokens, sequences, heads, has_initial_state, expected):
    assert (
        should_bound_sequence_extent(tokens, sequences, heads, 64, has_initial_state) is expected
    )


def _run_without_initial_state(
    inputs: tuple[torch.Tensor, ...],
    metadata: RaggedChunkMetadata,
    *,
    bv: int = 32,
    dht: torch.Tensor | None = None,
) -> tuple[torch.Tensor, None, torch.Tensor]:
    """Run dHU without materializing an initial-state gradient."""
    return _blackwell_delta_h_bwd_dhu_dv_fused_packed(
        *inputs[:5],
        metadata,
        gk=inputs[5],
        h0=None,
        dht=dht,
        scale=128**-0.5,
        bv=bv,
    )


def test_ragged_delta_h_without_initial_state_handles_zero_sequence_extent():
    physical_tokens = 128
    num_sequences = 32
    inputs = _inputs(physical_tokens, num_sequences, heads=8)
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets([0] * num_sequences), physical_tokens, 64
    )
    for tensor in inputs[:6]:
        tensor.fill_(torch.nan)

    dh, dh0, dv = _run_without_initial_state(inputs, metadata)
    torch.cuda.synchronize()

    assert dh.shape == (1, metadata.capacity, 8, 128, 128)
    assert dh0 is None
    assert dv.shape == inputs[3].shape


def test_ragged_delta_h_handles_all_empty_sequences():
    inputs = _inputs(0, 2)
    dh, dh0, dv = _run_ragged(inputs, [0, 0])

    assert dh.shape == (1, 0, 1, 128, 128)
    assert dv.shape == inputs[3].shape
    torch.testing.assert_close(dh0, inputs[-1], rtol=0, atol=0)


def test_ragged_delta_h_rejects_mismatched_chunk_size():
    inputs = _inputs(128, 1)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([128]), 128, 128)

    with pytest.raises(ValueError, match="metadata chunk size"):
        blackwell_delta_h_bwd_dhu_dv_fused_dispatch(
            *inputs[:5],
            gk=inputs[5],
            h0=inputs[6],
            dht=inputs[7],
            scale=128**-0.5,
            metadata=metadata,
        )


def test_ragged_delta_h_matches_independent_sequences():
    lengths = [65, 0, 63]
    inputs = _inputs(sum(lengths), len(lengths), lengths=lengths)
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
    "lengths",
    [[130, 0, 0], [63, 130]],
    ids=["trailing-empty-sequences", "tail-then-multichunk"],
)
def test_ragged_delta_h_ignores_nan_poisoned_physical_suffix(bv, lengths):
    physical_tokens = 256
    active_tokens = sum(lengths)
    inputs = _inputs(physical_tokens, len(lengths), lengths=lengths)
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


@pytest.mark.parametrize(
    "replayed_lengths",
    ([65, 63], [128, 0]),
    ids=["tails", "empty-sequence"],
)
def test_ragged_delta_h_replays_aligned_boundaries(replayed_lengths):
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

    cu_seqlens.copy_(cumulative_sequence_offsets(replayed_lengths))
    graph.replay()
    torch.cuda.synchronize()

    expected_dh, expected_dh0, expected_dv = _run_ragged(inputs, replayed_lengths)
    active_chunks = sum((length + 63) // 64 for length in replayed_lengths)
    assert captured_dh.shape[1] == 3
    torch.testing.assert_close(
        captured_dh[:, :active_chunks],
        expected_dh[:, :active_chunks],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(captured_dh0, expected_dh0, rtol=0, atol=0)
    torch.testing.assert_close(captured_dv, expected_dv, rtol=0, atol=0)


def test_ragged_delta_h_sequence_extent_forced_int64_matches_default(monkeypatch):
    lengths = [65, 0, 63] + [0] * 77
    inputs = _inputs(sum(lengths), len(lengths), heads=8, lengths=lengths)
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(lengths), sum(lengths), 64
    )

    expected_dh, expected_dh0, expected_dv = _run_without_initial_state(inputs, metadata)
    monkeypatch.setattr(delta_h_module, "requires_int64_abi", lambda *_tensors: True)
    actual_dh, actual_dh0, actual_dv = _run_without_initial_state(inputs, metadata)
    assert actual_dh0 is None and expected_dh0 is None
    active_chunks = metadata.chunk_offsets[-1].item()
    torch.testing.assert_close(
        actual_dh[:, :active_chunks], expected_dh[:, :active_chunks], rtol=0, atol=0
    )
    torch.testing.assert_close(actual_dv, expected_dv, rtol=0, atol=0)


@pytest.mark.parametrize(
    "replay_lengths",
    (
        pytest.param([65, 0, 63] + [0] * 77, id="sparse-interior-empty"),
        pytest.param([4] * 16 + [3] * 64, id="fully-active"),
    ),
)
@pytest.mark.parametrize("use_dht", [False, True])
@pytest.mark.parametrize("bv", [16, 32])
def test_ragged_delta_h_replay_bounds_sequence_extent_without_initial_state(
    replay_lengths, use_dht, bv
):
    physical_tokens = 256
    num_sequences = 80
    inputs = _inputs(physical_tokens, num_sequences, heads=8)
    q, k, w, do, aqk, gk = inputs[:6]
    cu_seqlens = cumulative_sequence_offsets([4] * 64 + [0] * 16)

    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, physical_tokens, 64)
    _blackwell_delta_h_bwd_dhu_dv_fused_packed(
        q,
        k,
        w,
        do,
        aqk,
        warm_metadata,
        gk=gk,
        h0=None,
        dht=inputs[7] if use_dht else None,
        scale=128**-0.5,
        bv=bv,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, physical_tokens, 64)
        actual_dh, actual_dh0, actual_dv = _blackwell_delta_h_bwd_dhu_dv_fused_packed(
            q,
            k,
            w,
            do,
            aqk,
            metadata,
            gk=gk,
            h0=None,
            dht=inputs[7] if use_dht else None,
            scale=128**-0.5,
            bv=bv,
        )
    assert actual_dh0 is None

    active_tokens = sum(replay_lengths)
    cu_seqlens.copy_(cumulative_sequence_offsets(replay_lengths))
    for tensor in (q, k, w, do, aqk, gk):
        tensor[:, active_tokens:].fill_(torch.nan)
    graph.replay()
    torch.cuda.synchronize()

    sequence_extent = max(index + 1 for index, length in enumerate(replay_lengths) if length)
    compact_inputs = tuple(tensor[:, :active_tokens].clone() for tensor in inputs[:6])
    compact_metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(replay_lengths[:sequence_extent]), active_tokens, 64
    )
    expected_dh, expected_dh0, expected_dv = _blackwell_delta_h_bwd_dhu_dv_fused_packed(
        *compact_inputs[:5],
        compact_metadata,
        gk=compact_inputs[5],
        h0=None,
        dht=inputs[7][:sequence_extent].clone() if use_dht else None,
        scale=128**-0.5,
        bv=bv,
    )
    assert expected_dh0 is None
    active_chunks = compact_metadata.chunk_offsets[-1].item()
    torch.testing.assert_close(
        actual_dh[:, :active_chunks], expected_dh[:, :active_chunks], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_dv[:, :active_tokens], expected_dv[:, :active_tokens], rtol=0, atol=0
    )
