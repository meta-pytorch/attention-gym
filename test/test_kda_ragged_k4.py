"""Ragged scheduler tests for the CuTe KDA K4 inverse stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    _chunk_kda_fwd_k4b_ragged_custom_op,
    chunk_kda_fwd_inter_solve_ragged_cute,
    chunk_kda_fwd_k4b_ragged_cute,
)
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe K4 kernel requires CUDA capability 10.0 or newer",
)

_BLOCK_PAIRS = ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2))


def _diagonal_inverses(lengths: list[int]) -> torch.Tensor:
    diagonal = torch.zeros(1, sum(lengths), 1, 16, device="cuda")
    token_start = 0
    for length in lengths:
        rows = torch.arange(length, device="cuda")
        diagonal[0, token_start + rows, 0, rows % 16] = 1
        token_start += length
    return diagonal


def _inputs(lengths: list[int], *, poison_inactive: bool = False):
    offsets = cumulative_sequence_offsets(lengths)
    metadata = prepare_ragged_chunk_metadata(offsets, sum(lengths), 64)
    torch.manual_seed(29)
    offdiagonal = torch.randn(metadata.capacity * 6, 16 * 16, device="cuda") / 32
    if poison_inactive:
        active_rows = sum((length + 63) // 64 for length in lengths) * 6
        offdiagonal[active_rows:] = torch.nan
    return metadata, offdiagonal, _diagonal_inverses(lengths)


def _reference(
    offdiagonal: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    expected = torch.zeros(1, sum(lengths), 1, 64, device="cuda", dtype=torch.bfloat16)
    token_start = 0
    global_chunk = 0
    for length in lengths:
        for local_start in range(0, length, 64):
            valid_tokens = min(64, length - local_start)
            matrix = torch.eye(64, device="cuda")
            for pair, (row_block, column_block) in enumerate(_BLOCK_PAIRS):
                matrix[
                    row_block * 16 : (row_block + 1) * 16,
                    column_block * 16 : (column_block + 1) * 16,
                ] = offdiagonal[global_chunk * 6 + pair].reshape(16, 16)
            inverse = torch.linalg.inv(matrix)
            output_start = token_start + local_start
            expected[0, output_start : output_start + valid_tokens, 0] = inverse[:valid_tokens].to(
                torch.bfloat16
            )
            global_chunk += 1
        token_start += length
    return expected


@pytest.mark.parametrize("lengths", [[65, 63], [0, 1, 64, 0, 65]])
def test_ragged_k4_matches_block_inverse(lengths):
    metadata, offdiagonal, diagonal = _inputs(lengths, poison_inactive=True)
    actual = chunk_kda_fwd_k4b_ragged_cute(offdiagonal, diagonal, metadata)
    expected = _reference(offdiagonal, lengths)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_ragged_k4_inactive_capacity_is_ignored_and_output_is_zero_extended():
    lengths = [64, 64]
    metadata, offdiagonal, diagonal = _inputs(lengths, poison_inactive=True)
    actual = chunk_kda_fwd_k4b_ragged_cute(offdiagonal, diagonal, metadata)
    expected = _reference(offdiagonal, lengths)
    alternate_poison = offdiagonal.clone()
    alternate_poison[12:] = torch.inf
    alternate = chunk_kda_fwd_k4b_ragged_cute(alternate_poison, diagonal, metadata)

    assert metadata.capacity == 3
    assert metadata.chunk_offsets.tolist() == [0, 1, 2]
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, alternate)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    upper = torch.ones(64, 64, dtype=torch.bool, device="cuda").triu(1)
    assert torch.count_nonzero(actual.view(2, 64, 64)[upper.expand(2, -1, -1)]) == 0


def test_ragged_k4_custom_op_and_fullgraph():
    lengths = [65, 63]
    metadata, offdiagonal, diagonal = _inputs(lengths)
    op_args = (
        offdiagonal,
        diagonal,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
    )
    torch.library.opcheck(_chunk_kda_fwd_k4b_ragged_custom_op, op_args)

    def operation(AkkOD, Akkd, offsets):
        graph_metadata = prepare_ragged_chunk_metadata(offsets, 128, 64)
        return chunk_kda_fwd_k4b_ragged_cute(AkkOD, Akkd, graph_metadata)

    expected = operation(offdiagonal, diagonal, metadata.cu_seqlens)
    actual = torch.compile(operation, fullgraph=True)(
        offdiagonal,
        diagonal,
        metadata.cu_seqlens,
    )
    torch.testing.assert_close(actual, expected)


def test_ragged_inter_solve_fullgraph_captures_k3():
    lengths = [65, 63]
    tokens = sum(lengths)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    torch.manual_seed(31)
    q = torch.randn(1, tokens, 1, 128, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    g = -torch.rand(1, tokens, 1, 128, device="cuda")
    beta = torch.sigmoid(torch.randn(1, tokens, 1, device="cuda"))
    diagonal = _diagonal_inverses(lengths)
    Aqk = torch.zeros(1, tokens, 1, 64, device="cuda", dtype=torch.bfloat16)

    def operation(q, k, g, beta, diagonal, Aqk, offsets):
        graph_metadata = prepare_ragged_chunk_metadata(offsets, tokens, 64)
        return chunk_kda_fwd_inter_solve_ragged_cute(
            q,
            k,
            g,
            beta,
            diagonal,
            Aqk,
            128**-0.5,
            graph_metadata,
        )

    expected = operation(q, k, g, beta, diagonal, Aqk.clone(), metadata.cu_seqlens)
    compiled = torch.compile(operation, fullgraph=True)
    actual = compiled(q, k, g, beta, diagonal, Aqk.clone(), metadata.cu_seqlens)
    alternate = compiled(torch.zeros_like(q), k, g, beta, diagonal, Aqk, metadata.cu_seqlens)

    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual[0], alternate[0])


@pytest.mark.parametrize("lengths", [[0], [0, 0, 0]])
def test_ragged_inter_solve_accepts_all_empty_sequences(lengths):
    metadata, _, diagonal = _inputs(lengths)
    q = torch.empty(1, 0, 1, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.empty(1, 0, 1, 128, device="cuda")
    beta = torch.empty(1, 0, 1, device="cuda")
    Aqk = torch.empty(1, 0, 1, 64, device="cuda", dtype=torch.bfloat16)

    actual_aqk, actual_akk = chunk_kda_fwd_inter_solve_ragged_cute(
        q,
        q,
        g,
        beta,
        diagonal,
        Aqk,
        128**-0.5,
        metadata,
    )

    assert actual_aqk.shape == Aqk.shape
    assert actual_akk.shape == (1, 0, 1, 64)


def test_ragged_k4_replays_aligned_to_ragged():
    aligned_lengths = [64, 64]
    metadata, offdiagonal, diagonal = _inputs(aligned_lengths)
    cu_seqlens = metadata.cu_seqlens
    chunk_kda_fwd_k4b_ragged_cute(offdiagonal, diagonal, metadata)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual = chunk_kda_fwd_k4b_ragged_cute(
            offdiagonal,
            diagonal,
            captured_metadata,
        )

    ragged_lengths = [65, 63]
    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    diagonal.copy_(_diagonal_inverses(ragged_lengths))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(offdiagonal, ragged_lengths)
    assert captured_metadata.chunk_offsets.tolist() == [0, 2, 3]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
