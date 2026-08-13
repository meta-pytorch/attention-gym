"""Tests for ragged KDA inter-chunk state recurrence."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the KDA recurrence kernel requires CUDA",
)


def _offsets(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )


def _run(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor,
    lengths: list[int],
):
    metadata = prepare_ragged_chunk_metadata(_offsets(lengths), k.shape[1], 64)
    return chunk_gated_delta_rule_fwd_h(
        k,
        w,
        u,
        gk,
        initial_state,
        metadata=metadata,
    )


def test_ragged_recurrence_matches_independent_sequences():
    torch.manual_seed(0)
    lengths = [65, 0, 63]
    tokens = sum(lengths)
    shape = (1, tokens, 2, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    w = torch.randn_like(k) / 8
    u = torch.randn_like(k) / 8
    gk = -torch.rand(shape, device="cuda")
    initial_state = torch.randn(3, 2, 128, 128, device="cuda") / 8

    h, v_new, final_state = _run(k, w, u, gk, initial_state, lengths)

    expected_v = torch.empty_like(v_new)
    expected_h = []
    expected_final = []
    begin = 0
    for sequence, length in enumerate(lengths):
        if length == 0:
            expected_final.append(initial_state[sequence])
            continue
        end = begin + length
        sequence_h, sequence_v, sequence_final = _run(
            k[:, begin:end],
            w[:, begin:end],
            u[:, begin:end],
            gk[:, begin:end],
            initial_state[sequence : sequence + 1],
            [length],
        )
        expected_v[:, begin:end] = sequence_v
        expected_h.append(sequence_h)
        expected_final.append(sequence_final[0])
        begin = end

    torch.testing.assert_close(v_new, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(h[:, :3], torch.cat(expected_h, dim=1), rtol=0, atol=0)
    torch.testing.assert_close(final_state, torch.stack(expected_final), rtol=0, atol=0)


def test_ragged_recurrence_replays_aligned_to_ragged():
    torch.manual_seed(1)
    tokens = 128
    shape = (1, tokens, 1, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    w = torch.randn_like(k) / 8
    u = torch.randn_like(k) / 8
    gk = -torch.rand(shape, device="cuda")
    initial_state = torch.randn(2, 1, 128, 128, device="cuda") / 8
    cu_seqlens = _offsets([64, 64])

    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    chunk_gated_delta_rule_fwd_h(k, w, u, gk, initial_state, metadata=warm_metadata)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k,
            w,
            u,
            gk,
            initial_state,
            metadata=metadata,
        )

    cu_seqlens.copy_(_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    expected_h, expected_v, expected_final = _run(
        k,
        w,
        u,
        gk,
        initial_state,
        [65, 63],
    )
    torch.testing.assert_close(v_new, expected_v, rtol=0, atol=0)
    torch.testing.assert_close(h[:, :3], expected_h[:, :3], rtol=0, atol=0)
    torch.testing.assert_close(final_state, expected_final, rtol=0, atol=0)
