"""Tests for ragged KDA inter-chunk state recurrence."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda import chunk_scheduler
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    ScheduleRequest,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.kda.fwd.triton import chunk_delta_h
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import (
    _delta_h_launch,
    _persistent_sequence_workers,
    chunk_gated_delta_rule_fwd_h,
)
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the KDA recurrence kernel requires CUDA capability 10.0",
)


def _run(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor,
    lengths: list[int],
):
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), k.shape[1], 64)
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


def _persistent_overflow_case(
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], RaggedChunkMetadata, list[int]]:
    torch.manual_seed(11)
    lengths = [2, 0] * 40
    tokens = sum(lengths)
    heads = 2
    shape = (1, tokens, heads, 128)
    k = torch.randn(shape, device="cuda", dtype=dtype) / 32
    w = torch.randn_like(k) / 32
    u = torch.randn_like(k) / 32
    gk = -torch.rand(shape, device="cuda")
    initial_state = torch.randn(len(lengths), heads, 128, 128, device="cuda") / 32
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    return (k, w, u, gk, initial_state), metadata, lengths


def _run_persistent_overflow_case(
    inputs: tuple[torch.Tensor, ...],
    metadata: RaggedChunkMetadata,
    schedule: ScheduleRequest,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run one explicit schedule with a materialized final state."""
    final_state = torch.empty_like(inputs[-1])
    h, v_new = _delta_h_launch(
        *inputs,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
        final_state,
        schedule,
    )
    return h, v_new, final_state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_ragged_recurrence_persistent_matches_static_with_active_overflow(dtype):
    inputs, metadata, lengths = _persistent_overflow_case(dtype)
    heads = inputs[0].shape[2]
    sequence_workers = _persistent_sequence_workers(
        metadata,
        inputs[0].shape[1],
        heads,
        value_tiles=2 if dtype != torch.float32 else 4,
        device=inputs[0].device,
        schedule=ScheduleRequest.PERSISTENT,
    )
    sequence_extent = max(index + 1 for index, length in enumerate(lengths) if length)
    assert sequence_workers < sequence_extent * heads

    outputs = [
        _run_persistent_overflow_case(inputs, metadata, schedule)
        for schedule in (ScheduleRequest.STATIC, ScheduleRequest.PERSISTENT)
    ]

    active_chunks = metadata.chunk_offsets[-1].item()
    torch.testing.assert_close(
        outputs[1][0][:, :active_chunks], outputs[0][0][:, :active_chunks], rtol=0, atol=0
    )
    torch.testing.assert_close(outputs[1][1:], outputs[0][1:], rtol=0, atol=0)


def test_ragged_recurrence_persistent_int64_matches_default(monkeypatch):
    inputs, metadata, _ = _persistent_overflow_case(torch.bfloat16)

    expected = _run_persistent_overflow_case(inputs, metadata, ScheduleRequest.PERSISTENT)
    monkeypatch.setattr(chunk_delta_h, "requires_int64_offsets", lambda *_tensors: True)
    actual = _run_persistent_overflow_case(inputs, metadata, ScheduleRequest.PERSISTENT)
    active_chunks = metadata.chunk_offsets[-1].item()
    torch.testing.assert_close(
        actual[0][:, :active_chunks], expected[0][:, :active_chunks], rtol=0, atol=0
    )
    torch.testing.assert_close(actual[1:], expected[1:], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("num_sequences", "heads", "tokens", "schedule", "expected_workers"),
    (
        (31, 16, 31 * 64, ScheduleRequest.AUTO, 0),
        (32, 16, 32 * 64, ScheduleRequest.AUTO, 74),
        (32, 16, 32 * 64 + 1, ScheduleRequest.AUTO, 0),
        (64, 7, 64 * 64, ScheduleRequest.AUTO, 0),
        (64, 8, 64 * 64, ScheduleRequest.AUTO, 74),
        (32, 16, 32 * 64 + 1, ScheduleRequest.PERSISTENT, 74),
        (512, 16, 2048, ScheduleRequest.STATIC, 0),
    ),
)
def test_delta_h_auto_schedule_is_conservative(
    monkeypatch,
    num_sequences,
    heads,
    tokens,
    schedule,
    expected_workers,
):
    monkeypatch.setattr(chunk_scheduler, "_multiprocessor_count", lambda device: 148)
    cu_seqlens = torch.empty(num_sequences + 1, device="cuda", dtype=torch.int32)
    metadata = RaggedChunkMetadata(cu_seqlens, None, capacity=0, chunk_size=64)

    assert (
        _persistent_sequence_workers(
            metadata,
            tokens,
            heads=heads,
            value_tiles=2,
            device=cu_seqlens.device,
            schedule=schedule,
        )
        == expected_workers
    )


def test_ragged_recurrence_accepts_all_empty_sequences():
    shape = (1, 0, 1, 128)
    k, w, u = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    gk = torch.empty(shape, device="cuda")
    initial_state = torch.randn(3, 1, 128, 128, device="cuda")

    h, v_new, final_state = _run(k, w, u, gk, initial_state, [0, 0, 0])

    assert h.shape == (1, 0, 1, 128, 128)
    assert v_new.shape == u.shape
    torch.testing.assert_close(final_state, initial_state, rtol=0, atol=0)


def test_ragged_recurrence_rejects_mismatched_metadata_chunk_size():
    shape = (1, 64, 1, 128)
    inputs = [torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([64]), 64, 32)

    with pytest.raises(ValueError, match="metadata chunk size"):
        chunk_gated_delta_rule_fwd_h(
            *inputs,
            torch.zeros(1, 1, 128, 128, device="cuda"),
            metadata=metadata,
        )


@pytest.mark.parametrize("output_final_state", [False, True])
def test_ragged_recurrence_fullgraph(output_final_state: bool):
    torch.manual_seed(3)
    lengths = [65, 63]
    tokens = sum(lengths)
    shape = (1, tokens, 1, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    w = torch.randn_like(k) / 8
    u = torch.randn_like(k) / 8
    gk = -torch.rand(shape, device="cuda")
    initial_state = torch.randn(2, 1, 128, 128, device="cuda") / 8
    cu_seqlens = cumulative_sequence_offsets(lengths)

    def operation(k, w, u, gk, initial_state, cu_seqlens):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
        return chunk_gated_delta_rule_fwd_h(
            k, w, u, gk, initial_state, metadata=metadata, output_final_state=output_final_state
        )

    expected = operation(k, w, u, gk, initial_state, cu_seqlens)
    actual = torch.compile(operation, fullgraph=True)(k, w, u, gk, initial_state, cu_seqlens)
    if not output_final_state:
        assert actual[2] is None and expected[2] is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ragged_recurrence_replays_aligned_to_ragged():
    torch.manual_seed(1)
    tokens = 128
    shape = (1, tokens, 1, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    w = torch.randn_like(k) / 8
    u = torch.randn_like(k) / 8
    gk = -torch.rand(shape, device="cuda")
    initial_state = torch.randn(2, 1, 128, 128, device="cuda") / 8
    cu_seqlens = cumulative_sequence_offsets([64, 64])

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

    cu_seqlens.copy_(cumulative_sequence_offsets([65, 63]))
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


@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("output_final_state", [False, True])
def test_ragged_recurrence_persistent_cuda_graph_replays_active_sequences(
    use_initial_state: bool,
    output_final_state: bool,
):
    torch.manual_seed(13)
    tokens, heads, num_sequences = 128, 16, 32
    shape = (1, tokens, heads, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 32
    w = torch.randn_like(k) / 32
    u = torch.randn_like(k) / 32
    gk = -torch.rand(shape, device="cuda")
    initial_state = (
        torch.randn(num_sequences, heads, 128, 128, device="cuda") / 32
        if use_initial_state
        else None
    )
    cu_seqlens = cumulative_sequence_offsets([4] * num_sequences)

    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    assert (
        _persistent_sequence_workers(
            warm_metadata,
            tokens,
            heads,
            value_tiles=2,
            device=k.device,
            schedule=ScheduleRequest.AUTO,
        )
        > 0
    )
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
            output_final_state=output_final_state,
        )
    assert (final_state is not None) is output_final_state

    replay_lengths = [8] * 16 + [0] * (num_sequences - 16)
    cu_seqlens.copy_(cumulative_sequence_offsets(replay_lengths))
    graph.replay()
    torch.cuda.synchronize()

    expected_metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    replay_extent = max(index + 1 for index, length in enumerate(replay_lengths) if length)
    sequence_workers = _persistent_sequence_workers(
        expected_metadata,
        tokens,
        heads,
        value_tiles=2,
        device=k.device,
        schedule=ScheduleRequest.AUTO,
    )
    assert sequence_workers < replay_extent * heads
    expected_final = (
        torch.empty(num_sequences, heads, 128, 128, device="cuda", dtype=torch.float32)
        if output_final_state
        else None
    )
    expected_h, expected_v = _delta_h_launch(
        k,
        w,
        u,
        gk,
        initial_state,
        expected_metadata.cu_seqlens,
        expected_metadata.chunk_offsets,
        expected_metadata.capacity,
        expected_final,
        ScheduleRequest.STATIC,
    )
    active_chunks = expected_metadata.chunk_offsets[-1].item()
    torch.testing.assert_close(h[:, :active_chunks], expected_h[:, :active_chunks], rtol=0, atol=0)
    torch.testing.assert_close(v_new, expected_v, rtol=0, atol=0)
    if output_final_state:
        torch.testing.assert_close(final_state, expected_final, rtol=0, atol=0)
    else:
        assert final_state is None and expected_final is None


def test_delta_h_opcheck():
    """Schema/fake consistency for the registered op pair, dense and ragged."""
    from attn_gym.linear.kda.ops import delta_h_op, delta_h_with_state_op

    torch.manual_seed(5)
    lengths = [65, 63]
    tokens = sum(lengths)
    shape = (1, tokens, 1, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    w = torch.randn_like(k) / 8
    u = torch.randn_like(k) / 8
    gk = -torch.rand(shape, device="cuda")
    initial_state = torch.randn(2, 1, 128, 128, device="cuda") / 8
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    ragged_args = (
        k,
        w,
        u,
        gk,
        initial_state,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
    )
    torch.library.opcheck(delta_h_op, ragged_args)
    torch.library.opcheck(delta_h_with_state_op, ragged_args)
    dense_args = (k, w, u, gk, None, None, None, tokens // 64)
    torch.library.opcheck(delta_h_op, dense_args)
    torch.library.opcheck(delta_h_with_state_op, dense_args)
