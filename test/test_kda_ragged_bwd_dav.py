"""Ragged scheduling tests for the Triton KDA dAv backward stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_dav import chunk_kda_bwd_dav
from attn_gym.linear.kda.chunk_scheduler import ScheduleRequest, prepare_ragged_chunk_metadata
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the KDA dAv backward kernel requires CUDA",
)

_CHUNK_SIZE = 64
_SCALE = 128**-0.5


def _inputs(tokens: int, heads: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(31)
    shape = (1, tokens, heads, 128)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    A = torch.randn(1, tokens, heads, _CHUNK_SIZE, device="cuda", dtype=torch.bfloat16) / 8
    do = torch.randn_like(v) / 8
    return v, A, do


def _run_ragged(
    v: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    lengths: list[int],
    *,
    schedule: ScheduleRequest = ScheduleRequest.STATIC,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(lengths), v.shape[1], _CHUNK_SIZE
    )
    return chunk_kda_bwd_dav(v, A, do, _SCALE, metadata=metadata, schedule=schedule)


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


def test_ragged_bwd_dav_handles_all_empty_sequences():
    inputs = _inputs(0)
    actual = _run_ragged(*inputs, [0, 0])

    assert actual[0].shape == inputs[0].shape
    assert actual[1].shape == inputs[1].shape


def test_ragged_bwd_dav_rejects_batched_inputs():
    inputs = tuple(tensor.expand(2, *tensor.shape[1:]) for tensor in _inputs(64))
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([64]), 64, _CHUNK_SIZE)

    with pytest.raises(ValueError, match="requires batch size 1"):
        chunk_kda_bwd_dav(*inputs, _SCALE, metadata=metadata)


def test_ragged_bwd_dav_rejects_mismatched_chunk_size():
    v, _, do = _inputs(64)
    A = torch.empty(1, 64, 1, 32, device="cuda", dtype=torch.bfloat16)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([64]), 64, _CHUNK_SIZE)

    with pytest.raises(ValueError, match="metadata chunk size"):
        chunk_kda_bwd_dav(v, A, do, _SCALE, chunk_size=32, metadata=metadata)


def test_ragged_bwd_dav_replays_aligned_to_ragged():
    inputs = _inputs(128)
    cu_seqlens = cumulative_sequence_offsets([64, 64])
    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, _CHUNK_SIZE)
    chunk_kda_bwd_dav(*inputs, _SCALE, metadata=warm_metadata)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, _CHUNK_SIZE)
        actual = chunk_kda_bwd_dav(*inputs, _SCALE, metadata=metadata)

    cu_seqlens.copy_(cumulative_sequence_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    expected = _independent_sequences(*inputs, [65, 63])
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    for replayed, independent in zip(actual, expected, strict=True):
        torch.testing.assert_close(replayed, independent, rtol=0, atol=0)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="persistent scheduling requires the TMA path",
)
@pytest.mark.parametrize("lengths", [[65, 63], [1, 64, 0, 65]])
@pytest.mark.parametrize("heads", [1, 2])
def test_persistent_ragged_bwd_dav_matches_static_over_capacity(lengths, heads):
    """Stay bit-identical to static scheduling when capacity dwarfs active work."""
    tokens = sum(lengths)
    inputs = _inputs(16 * tokens, heads)
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(lengths), inputs[0].shape[1], _CHUNK_SIZE
    )
    assert metadata.capacity >= 8 * metadata.chunk_offsets[-1].item()

    static = _run_ragged(*inputs, lengths)
    persistent = _run_ragged(*inputs, lengths, schedule=ScheduleRequest.PERSISTENT)
    expected = _independent_sequences(*inputs, lengths)

    for actual, reference in zip(persistent, expected, strict=True):
        torch.testing.assert_close(actual[:, :tokens], reference, rtol=0, atol=0)
    for actual, other in zip(persistent, static, strict=True):
        assert torch.equal(actual[:, :tokens], other[:, :tokens])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="persistent scheduling requires the TMA path",
)
def test_persistent_ragged_bwd_dav_handles_zero_capacity():
    inputs = _inputs(0)
    actual = _run_ragged(*inputs, [0, 0], schedule=ScheduleRequest.PERSISTENT)

    assert actual[0].shape == inputs[0].shape
    assert actual[1].shape == inputs[1].shape

    # Zero capacity has nothing to schedule, so even the off-TMA V=64 layout
    # returns instead of raising the persistent-eligibility error.
    v, A, do = inputs
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([0, 0]), 0, _CHUNK_SIZE)
    dv, dA = chunk_kda_bwd_dav(
        v[..., :64],
        A,
        do[..., :64],
        _SCALE,
        metadata=metadata,
        schedule=ScheduleRequest.PERSISTENT,
    )
    assert dv.shape == v[..., :64].shape
    assert dA.shape == A.shape


def test_persistent_ragged_bwd_dav_requires_packed_tma_path():
    v, A, do = _inputs(64)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([64]), 64, _CHUNK_SIZE)

    with pytest.raises(ValueError, match="persistent scheduling requires the packed TMA path"):
        chunk_kda_bwd_dav(
            v[..., :64],
            A,
            do[..., :64],
            _SCALE,
            metadata=metadata,
            schedule=ScheduleRequest.PERSISTENT,
        )

    # Dense launch grids are already exact, so the request is trivially satisfied.
    dense = chunk_kda_bwd_dav(v, A, do, _SCALE)
    dense_persistent = chunk_kda_bwd_dav(v, A, do, _SCALE, schedule=ScheduleRequest.PERSISTENT)
    for actual, other in zip(dense_persistent, dense, strict=True):
        assert torch.equal(actual, other)
