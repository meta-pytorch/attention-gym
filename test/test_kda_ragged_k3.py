"""Ragged scheduler tests for the CuTe KDA K3 off-diagonal stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    chunk_kda_fwd_k3b_ragged_cute,
)
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe K3 kernel requires CUDA capability 10.0 or newer",
)

_BLOCK_PAIRS = ((1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2))


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
    cu_seqlens = cumulative_sequence_offsets(lengths)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    Aqk, AkkOD = chunk_kda_fwd_k3b_ragged_cute(
        *inputs,
        scale=128**-0.5,
        metadata=metadata,
    )
    return metadata, Aqk, AkkOD


def _k3_reference(
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    lengths: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K3's sequence-local off-diagonal blocks with PyTorch operations."""
    q, k, g, beta = (tensor[0, :, 0].float() for tensor in inputs[:4])
    expected_aqk = inputs[4].clone()
    active_chunks = sum((length + 63) // 64 for length in lengths)
    expected_akk_od = torch.zeros(active_chunks * 6, 16 * 16, device=q.device)

    sequence_start = 0
    global_chunk = 0
    for length in lengths:
        sequence_end = sequence_start + length
        for chunk_start in range(sequence_start, sequence_end, 64):
            chunk_end = min(chunk_start + 64, sequence_end)
            for pair, (row_block, column_block) in enumerate(_BLOCK_PAIRS):
                row_start = chunk_start + row_block * 16
                column_start = chunk_start + column_block * 16
                column_end = min(column_start + 16, chunk_end)

                expected_aqk[
                    0,
                    column_start:column_end,
                    0,
                    row_block * 16 : (row_block + 1) * 16,
                ] = 0
                if row_start >= chunk_end:
                    continue

                row_end = min(row_start + 16, chunk_end)
                gate_reference = g[row_start]
                row_gate = torch.exp2(g[row_start:row_end] - gate_reference)
                column_gate = torch.exp2(gate_reference - g[column_start:column_end])

                gated_q = (q[row_start:row_end] * row_gate).to(torch.bfloat16).float()
                gated_row_k = (k[row_start:row_end] * row_gate).to(torch.bfloat16).float()
                gated_column_k = (
                    (k[column_start:column_end] * column_gate).to(torch.bfloat16).float()
                )
                expected_aqk[
                    0,
                    row_start:row_end,
                    0,
                    column_block * 16 : (column_block + 1) * 16,
                ] = (gated_q @ gated_column_k.T) * 128**-0.5

                raw_block = expected_akk_od[global_chunk * 6 + pair].view(16, 16)
                raw_block[: row_end - row_start, : column_end - column_start] = beta[
                    row_start:row_end, None
                ] * (gated_row_k @ gated_column_k.T)
            global_chunk += 1
        sequence_start = sequence_end
    return expected_aqk, expected_akk_od


def _assert_matches_reference(
    actual_aqk: torch.Tensor,
    actual_akk_od: torch.Tensor,
    expected_aqk: torch.Tensor,
    expected_akk_od: torch.Tensor,
) -> None:
    """Compare K3 outputs within their BF16-gating and FP32-dot error bounds."""
    # The CuTe fast exp2 may shift a gated operand by one BF16 step. Aqk also
    # rounds its FP32 dot to BF16; AkkOD only adds FP32 reduction-order error.
    # The absolute allowances cover cancellation at the tests' 1/8 input scale
    # and account for Aqk's additional 1/sqrt(128) scale.
    bf16_eps = torch.finfo(torch.bfloat16).eps
    torch.testing.assert_close(actual_aqk, expected_aqk, rtol=2 * bf16_eps, atol=2e-5)
    torch.testing.assert_close(actual_akk_od, expected_akk_od, rtol=bf16_eps, atol=2e-4)


@pytest.mark.parametrize("lengths", [[65, 63], [0, 1, 64, 0, 65]])
def test_ragged_k3_matches_pytorch_reference(lengths):
    inputs = _inputs(sum(lengths))
    input_aqk = inputs[-1].clone()
    metadata, actual_aqk, actual_akk_od = _run(inputs, lengths)
    torch.testing.assert_close(inputs[-1], input_aqk, rtol=0, atol=0)
    assert actual_aqk.data_ptr() != inputs[-1].data_ptr()
    expected_aqk, expected_akk_od = _k3_reference(inputs, lengths)
    active_rows = int(metadata.chunk_offsets[-1]) * 6

    _assert_matches_reference(
        actual_aqk,
        actual_akk_od[:active_rows],
        expected_aqk,
        expected_akk_od,
    )
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


def test_ragged_k3_ignores_poisoned_inactive_capacity():
    """Capacity-only CTAs must not read token storage, even with zero active tokens."""
    tokens = 128
    inputs = tuple(torch.full_like(tensor, float("nan")) for tensor in _inputs(tokens)[:4]) + (
        torch.zeros(1, tokens, 1, 64, device="cuda", dtype=torch.bfloat16),
    )
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([0, 0]), tokens, 64)
    aqk, akk_od = chunk_kda_fwd_k3b_ragged_cute(*inputs, scale=128**-0.5, metadata=metadata)

    assert metadata.capacity > 0
    assert not akk_od.isnan().any()
    assert not aqk.isnan().any()


def test_ragged_k3_rejects_mismatched_metadata_chunk_size():
    inputs = _inputs(128)
    cu_seqlens = cumulative_sequence_offsets([128])
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 32)

    with pytest.raises(ValueError, match="metadata chunk size must match chunk_size=64, got 32"):
        chunk_kda_fwd_k3b_ragged_cute(
            *inputs,
            scale=128**-0.5,
            metadata=metadata,
        )


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

    expected_aqk, expected_akk_od = _k3_reference(inputs, [65, 63])
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    _assert_matches_reference(actual_aqk, actual_akk_od, expected_aqk, expected_akk_od)
