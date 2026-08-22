"""Ragged scheduler tests for the CuTe KDA K4 inverse stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import (
    GridScheduler,
    RaggedChunkMetadata,
    ScheduleKind,
    ScheduleRequest,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    _resolve_ragged_execution,
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


@pytest.mark.parametrize("lengths", [[0], [0, 0, 0]])
def test_ragged_inter_solve_accepts_all_empty_sequences(lengths, monkeypatch):
    requests = []
    resolve_chunk = GridScheduler.resolve_chunk

    def record_resolve(self, request, device):
        requests.append(request)
        return resolve_chunk(self, request, device)

    monkeypatch.setattr(GridScheduler, "resolve_chunk", record_resolve)
    metadata, offdiagonal, diagonal = _inputs(lengths)
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

    forced_akk = chunk_kda_fwd_k4b_ragged_cute(
        offdiagonal,
        diagonal,
        metadata,
        schedule=ScheduleRequest.PERSISTENT,
    )

    assert actual_aqk.shape == Aqk.shape
    assert actual_akk.shape == (1, 0, 1, 64)
    assert forced_akk.shape == actual_akk.shape
    assert requests == [ScheduleRequest.AUTO, ScheduleRequest.PERSISTENT]


@pytest.mark.parametrize(
    ("schedule_request", "capacity", "expected_kind"),
    (
        (ScheduleRequest.AUTO, 300, ScheduleKind.STATIC),
        (ScheduleRequest.AUTO, 301, ScheduleKind.PERSISTENT),
        (ScheduleRequest.STATIC, 301, ScheduleKind.STATIC),
        (ScheduleRequest.PERSISTENT, 1, ScheduleKind.PERSISTENT),
    ),
)
def test_ragged_inter_solve_uses_three_wave_auto_threshold(
    monkeypatch, schedule_request, capacity, expected_kind
):
    monkeypatch.setattr(
        GridScheduler,
        "num_chunk_workers",
        lambda self, device: min(self.metadata.capacity, 100),
    )
    metadata = RaggedChunkMetadata(
        torch.empty(2, device="cuda", dtype=torch.int32),
        torch.empty(2, device="cuda", dtype=torch.int32),
        capacity,
        64,
    )
    resolved = _resolve_ragged_execution(torch.empty(1, device="cuda"), metadata, schedule_request)

    assert resolved.kind is expected_kind
    assert resolved.workers == min(capacity, 100)
    assert resolved.capacity_tasks == capacity


def test_ragged_inter_solve_nonempty_resolves_once(monkeypatch):
    requests = []
    resolve_chunk = GridScheduler.resolve_chunk

    def record_resolve(self, request, device):
        requests.append(request)
        return resolve_chunk(self, request, device)

    monkeypatch.setattr(GridScheduler, "resolve_chunk", record_resolve)
    lengths = [65, 63]
    metadata, _, diagonal = _inputs(lengths)
    torch.manual_seed(37)
    q = torch.randn(1, sum(lengths), 1, 128, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    g = -torch.rand(1, sum(lengths), 1, 128, device="cuda")
    beta = torch.rand(1, sum(lengths), 1, device="cuda")
    Aqk = torch.zeros(1, sum(lengths), 1, 64, device="cuda", dtype=torch.bfloat16)

    actual_aqk, actual_akk = chunk_kda_fwd_inter_solve_ragged_cute(
        q,
        k,
        g,
        beta,
        diagonal,
        Aqk,
        128**-0.5,
        metadata,
    )

    assert actual_aqk.shape == Aqk.shape
    assert actual_akk.shape == Aqk.shape
    assert requests == [ScheduleRequest.AUTO]


@pytest.mark.parametrize("schedule", [ScheduleRequest.STATIC, ScheduleRequest.PERSISTENT])
def test_ragged_k4_replays_aligned_to_ragged(schedule):
    aligned_lengths = [64, 64]
    metadata, offdiagonal, diagonal = _inputs(aligned_lengths)
    cu_seqlens = metadata.cu_seqlens
    chunk_kda_fwd_k4b_ragged_cute(offdiagonal, diagonal, metadata, schedule=schedule)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual = chunk_kda_fwd_k4b_ragged_cute(
            offdiagonal,
            diagonal,
            captured_metadata,
            schedule=schedule,
        )

    ragged_lengths = [65, 63]
    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    diagonal.copy_(_diagonal_inverses(ragged_lengths))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(offdiagonal, ragged_lengths)
    assert captured_metadata.chunk_offsets.tolist() == [0, 2, 3]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_persistent_ragged_k4_matches_static_over_capacity():
    """Stride a fixed chunk-worker grid over active chunks far below capacity."""
    lengths = [65, 63]
    offsets = cumulative_sequence_offsets(lengths)
    capacity_tokens = 16 * sum(lengths)
    metadata = prepare_ragged_chunk_metadata(offsets, capacity_tokens, 64)
    torch.manual_seed(29)
    offdiagonal = torch.randn(metadata.capacity * 6, 16 * 16, device="cuda") / 32
    diagonal = torch.zeros(1, capacity_tokens, 1, 16, device="cuda")
    token_start = 0
    for length in lengths:
        rows = torch.arange(length, device="cuda")
        diagonal[0, token_start + rows, 0, rows % 16] = 1
        token_start += length

    static = chunk_kda_fwd_k4b_ragged_cute(
        offdiagonal, diagonal, metadata, schedule=ScheduleRequest.STATIC
    )
    persistent = chunk_kda_fwd_k4b_ragged_cute(
        offdiagonal, diagonal, metadata, schedule=ScheduleRequest.PERSISTENT
    )

    assert torch.equal(persistent, static)
    active_tokens = sum(lengths)
    torch.testing.assert_close(
        persistent[0, :active_tokens].float(),
        _reference(offdiagonal, lengths)[0].float(),
        rtol=2e-2,
        atol=2e-2,
    )
    # The zero-initialized output past the active tokens must stay untouched.
    assert not persistent[0, active_tokens:].any()


def test_persistent_ragged_k4_strides_multiple_chunks_per_worker(monkeypatch):
    """Force fewer workers than active chunks so CTAs reuse SMEM across iterations."""
    from attn_gym.linear.kda import chunk_scheduler

    monkeypatch.setattr(chunk_scheduler.GridScheduler, "num_chunk_workers", lambda self, device: 2)
    lengths = [65, 63, 130, 70]
    metadata, offdiagonal, diagonal = _inputs(lengths)
    assert metadata.chunk_offsets[-1].item() > 2

    static = chunk_kda_fwd_k4b_ragged_cute(
        offdiagonal, diagonal, metadata, schedule=ScheduleRequest.STATIC
    )
    persistent = chunk_kda_fwd_k4b_ragged_cute(
        offdiagonal, diagonal, metadata, schedule=ScheduleRequest.PERSISTENT
    )

    assert torch.equal(persistent, static)
