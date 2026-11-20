"""Replay-relative checkpoints preserve the canonical 64-token KDA chunk schedule."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear import ReplayState, StateCheckpoints, chunk_kda, paged_chunk_kda
from attn_gym.linear.kda.ops import chunk_replay_commit_op
from attn_gym.testing import (
    CHECKPOINT_SENTINEL,
    assert_checkpoints_into_cache_match,
    checkpoint_rows,
    sentinel_state_checkpoints,
)
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

OPTIONS = {"schedule": "static", "autotune": False}


@pytest.fixture
def cuda() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("paged chunk KDA requires CUDA capability 8.0 or newer")


def replay_buffers(counts: list[int]) -> ReplayState:
    return ReplayState(
        *make_kda_test_inputs(64, batch=len(counts), seed=73, gate_scale=0.1),
        torch.tensor(counts, device="cuda", dtype=torch.int32),
    )


def clone_replay(replay: ReplayState) -> ReplayState:
    return ReplayState(*(tensor.clone() for tensor in replay))


def checkpoint_counts(
    lengths: list[int], routes: list[int], flags: list[bool], counts: list[int], interval: int
) -> list[int]:
    return [
        ((counts[max(slot, 0)] if has_state else 0) + length) // interval
        for length, slot, has_state in zip(lengths, routes, flags, strict=True)
    ]


def expected_checkpoints(
    tensors: tuple[torch.Tensor, ...],
    initial_cache: torch.Tensor,
    initial_replay: ReplayState,
    saved: StateCheckpoints,
    lengths: list[int],
    routes: list[int],
    flags: list[bool],
    counts: list[int],
    *,
    packed: bool,
) -> torch.Tensor:
    """Stop independent, replay-prepended chunk calls at each requested boundary."""
    expected = torch.full_like(checkpoint_rows(saved), CHECKPOINT_SENTINEL)
    reservations = checkpoint_counts(lengths, routes, flags, counts, saved.every_n_tokens)
    token_start = checkpoint_start = 0
    for sequence, (length, slot, has_state, reserved) in enumerate(
        zip(lengths, routes, flags, reservations, strict=True)
    ):
        if slot > 0:
            replayed = counts[slot] if has_state else 0
            new_tokens = tuple(
                tensor[:, token_start : token_start + length]
                if packed
                else tensor[sequence : sequence + 1, :length]
                for tensor in tensors
            )
            logical_tokens = tuple(
                torch.cat((cache[slot : slot + 1, :replayed], new), dim=1)
                for cache, new in zip(initial_replay[:-1], new_tokens, strict=True)
            )
            initial = initial_cache[slot : slot + 1].float()
            if not has_state:
                initial = torch.zeros_like(initial)
            for index in range(reserved):
                stop = (index + 1) * saved.every_n_tokens
                _, state = chunk_kda(
                    *(tensor[:, :stop] for tensor in logical_tokens),
                    initial,
                    cu_seqlens=cumulative_sequence_offsets([stop]),
                    output_final_state=True,
                    kernel_options=OPTIONS,
                )
                assert state is not None
                expected[checkpoint_start + index].copy_(state[0])
        token_start += length
        checkpoint_start += reserved
    return expected


def assert_replay_equal(actual: ReplayState, expected: ReplayState) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "mode,state_dtype,interval",
    [
        ("packed", torch.float32, 64),
        ("packed", torch.float32, 192),
        ("dense", torch.float32, 128),
        ("decode", torch.float32, 64),
        ("decode", torch.bfloat16, 64),
        # Intervals above 64 never complete during one-token decode.
        ("decode", torch.float32, 128),
    ],
)
@torch.no_grad()
def test_replay_checkpoints_match_chunk_boundaries(
    cuda: None, interval: int, mode: str, state_dtype: torch.dtype, no_benchmarking: None
) -> None:
    # One token completes count=63; a fresh slot must ignore count=63 and NaN poison.
    # Packed input also covers an empty request and reserved-but-unwritten inactive rows.
    packed = mode == "packed"
    if mode != "decode":
        lengths = (
            [0, 1, 2 * interval - 19 + 7, interval - 1, interval, interval]
            if packed
            else [2 * interval + 7] * 6
        )
        routes = [5, 2, 4, 1, 0, -1]
        flags = [True, True, True, False, True, False]
        counts = [0, 63, 63, 0, 19, 11, 0]
    else:
        # Non-completing requests come first, so checkpoint rows differ from sequence indices.
        lengths = [1] * 6
        routes = [4, 2, 1, 0, 5, -1]
        flags = [True, True, False, True, True, False]
        counts = [0, 63, 63, 0, 62, 63]
    tensors = make_kda_test_inputs(
        sum(lengths) if packed else lengths[0], batch=1 if packed else len(lengths), gate_scale=0.1
    )
    initial_cache = torch.randn(len(counts), 1, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    initial_replay = replay_buffers(counts)
    initial_cache[1].fill_(float("nan"))
    for tensor in initial_replay[:-1]:
        tensor[1].fill_(float("nan"))
    reserved = checkpoint_counts(lengths, routes, flags, counts, interval)
    saved = sentinel_state_checkpoints(reserved, interval, (1, 128, 128))
    expected_states = expected_checkpoints(
        tensors,
        initial_cache,
        initial_replay,
        saved,
        lengths,
        routes,
        flags,
        counts,
        packed=packed,
    )
    expected_cache, actual_cache = initial_cache.clone(), initial_cache.clone()
    expected_replay, actual_replay = clone_replay(initial_replay), clone_replay(initial_replay)
    slots = torch.tensor(routes, device="cuda", dtype=torch.int32)
    has_initial = torch.tensor(flags, device="cuda")
    cu_seqlens = cumulative_sequence_offsets(lengths) if packed else None
    expected_output = paged_chunk_kda(
        *tensors,
        expected_cache,
        slots,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial,
        replay_state=expected_replay,
        kernel_options=OPTIONS,
    )
    output = paged_chunk_kda(
        *tensors,
        actual_cache,
        slots,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial,
        replay_state=actual_replay,
        state_checkpoints=saved,
        kernel_options=OPTIONS,
    )
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0, equal_nan=True)
    assert_replay_equal(actual_replay, expected_replay)
    torch.testing.assert_close(checkpoint_rows(saved), expected_states, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["packed", "decode"])
@torch.no_grad()
def test_replay_checkpoints_written_into_the_state_cache_match_a_separate_pool(
    cuda: None, mode: str, no_benchmarking: None
) -> None:
    decode = mode == "decode"
    lengths = [1, 1, 1] if decode else [45, 173, 70]
    counts = [0, 63, 19, 63, 0, 0, 0, 0]  # Slots 4-7 receive checkpoints.
    tensors = make_kda_test_inputs(
        1 if decode else sum(lengths), batch=3 if decode else 1, gate_scale=0.1
    )
    initial_replay = replay_buffers(counts)
    slots = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    cu_seqlens = None if decode else cumulative_sequence_offsets(lengths)
    cache = torch.randn(len(counts), 1, 128, 128, device="cuda") * 0.01

    def run(cache: torch.Tensor, saved: StateCheckpoints) -> torch.Tensor:
        return paged_chunk_kda(
            *tensors,
            cache,
            slots,
            cu_seqlens=cu_seqlens,
            replay_state=clone_replay(initial_replay),
            state_checkpoints=saved,
            kernel_options=OPTIONS,
        )

    reserved = checkpoint_counts(lengths, [1, 2, 3], [True] * 3, counts, 64)
    reference = sentinel_state_checkpoints(reserved, 64, (1, 128, 128), spare=0)
    block_table = [4, 5] if decode else [4, 0, 5, 6, -1, 7]
    # Every reserved row is produced: counts 63, 19, 63 complete chunks in all three sequences.
    produced = range(len(block_table))
    assert_checkpoints_into_cache_match(run, cache, reference, block_table, produced)


@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("execution", ["fullgraph", "cuda_graph"])
@torch.no_grad()
def test_replay_checkpoints_read_changed_graph_metadata(
    cuda: None, mode: str, execution: str, no_benchmarking: None
) -> None:
    from torch._dynamo.testing import CompileCounterWithBackend

    decode = mode == "decode"
    tensors = make_kda_test_inputs(1 if decode else 384, batch=3 if decode else 1, gate_scale=0.1)
    initial_cache = torch.randn(6, 1, 128, 128, device="cuda") * 0.01
    initial_replay = replay_buffers([0, 63, 19, 0, 17, 63])
    state_cache = initial_cache.clone()
    replay_state = clone_replay(initial_replay)
    slots = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
    has_initial = torch.ones(3, device="cuda", dtype=torch.bool)
    cu_seqlens = None if decode else cumulative_sequence_offsets([45, 173, 166])
    saved = sentinel_state_checkpoints([1, 0, 0] if decode else [1, 3, 2], 64, (1, 128, 128))
    states, offsets = saved.state_pool, saved.cu_checkpoint_offsets

    def run(cache, replay, pool):
        return paged_chunk_kda(
            *tensors,
            cache,
            slots,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial,
            replay_state=replay,
            state_checkpoints=saved._replace(state_pool=pool),
            kernel_options=OPTIONS,
        )

    run(state_cache, replay_state, states)
    state_cache.copy_(initial_cache)
    for destination, source in zip(replay_state, initial_replay, strict=True):
        destination.copy_(source)
    if execution == "fullgraph":
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(run, fullgraph=True, backend=counter)
        compiled(state_cache, replay_state, states)
    else:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(state_cache, replay_state, states)

    for lengths, routes, flags, counts in (
        (
            [1, 1, 1] if decode else [45, 173, 166],
            [2, 4, 5],
            [True, False, True],
            [0, 0, 63, 0, 63, 19],
        ),
        (
            [1, 1, 1] if decode else [0, 129, 255],
            [0, 4, 2],
            [True, False, True],
            [0, 0, 63, 0, 17, 0],
        ),
    ):
        slots.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
        has_initial.copy_(torch.tensor(flags, device="cuda"))
        if cu_seqlens is not None:
            cu_seqlens.copy_(cumulative_sequence_offsets(lengths))
        offsets.copy_(
            cumulative_sequence_offsets(checkpoint_counts(lengths, routes, flags, counts, 64))
        )
        state_cache.copy_(initial_cache)
        for destination, source in zip(replay_state, initial_replay, strict=True):
            destination.copy_(source)
        replay_state.count.copy_(torch.tensor(counts, device="cuda", dtype=torch.int32))
        states.fill_(CHECKPOINT_SENTINEL)
        expected_cache = state_cache.clone()
        expected_replay = clone_replay(replay_state)
        expected_states = expected_checkpoints(
            tensors,
            initial_cache,
            initial_replay,
            saved,
            lengths,
            routes,
            flags,
            counts,
            packed=not decode,
        )
        eager_states = torch.full_like(states, CHECKPOINT_SENTINEL)
        expected_output = run(expected_cache, expected_replay, eager_states)
        eager_rows = checkpoint_rows(saved._replace(state_pool=eager_states))
        torch.testing.assert_close(eager_rows, expected_states, rtol=0, atol=0)
        if execution == "fullgraph":
            output = compiled(state_cache, replay_state, states)
        else:
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(state_cache, expected_cache, rtol=0, atol=0)
        assert_replay_equal(replay_state, expected_replay)
        torch.testing.assert_close(checkpoint_rows(saved), expected_states, rtol=0, atol=0)
        assert torch.all(states[0] == CHECKPOINT_SENTINEL)
    if execution == "fullgraph":
        assert counter.frame_count == 1


@pytest.mark.parametrize("wide_offsets", [False, True])
@torch.no_grad()
def test_replay_checkpoint_commit_operator_registration(
    cuda: None, monkeypatch: pytest.MonkeyPatch, wide_offsets: bool
) -> None:
    if wide_offsets:
        from attn_gym.linear.kda.fwd.triton import paged_replay

        monkeypatch.setattr(paged_replay, "requires_int64_offsets", lambda *args: True)
    final_state = torch.randn(4, 1, 128, 128, device="cuda")
    slots = torch.tensor([2, 3, 0, 4], device="cuda", dtype=torch.int32)
    flags = torch.tensor([True, True, True, False], device="cuda")
    active_counts = torch.tensor([63, 17, 0, 0], device="cuda", dtype=torch.int32)
    state_cache = torch.randn(6, 1, 128, 128, device="cuda")
    replay_count = torch.zeros(6, device="cuda", dtype=torch.int32)
    saved = sentinel_state_checkpoints([1, 0, 0, 0], 64, (1, 128, 128))
    args = (final_state, slots, flags, active_counts, state_cache, replay_count, *saved)
    if not wide_offsets:
        torch.library.opcheck(chunk_replay_commit_op, args)
        # Block-table entries alias the state cache: slot 5 is not routed.
        alias = saved._replace(
            state_pool=state_cache,
            block_table=torch.tensor([5], device="cuda", dtype=torch.int32),
        )
        torch.library.opcheck(chunk_replay_commit_op, (*args[:6], *alias))
        saved.state_pool.fill_(CHECKPOINT_SENTINEL)
    chunk_replay_commit_op(*args)
    rows = checkpoint_rows(saved)
    torch.testing.assert_close(rows[0], final_state[0], rtol=0, atol=0)
    assert torch.all(rows[1:] == CHECKPOINT_SENTINEL)
    assert torch.all(saved.state_pool[0] == CHECKPOINT_SENTINEL)


@pytest.mark.parametrize("invalid", ["dtype", "offsets", "block_table", "pool"])
def test_raw_replay_checkpoint_commit_rejects_invalid_metadata_without_device_execution(
    invalid: str,
) -> None:
    from attn_gym.linear.kda.fwd.triton.paged_replay import commit_paged_chunk_replay

    final_state = torch.empty(2, 1, 128, 128)
    slots = torch.tensor([1, 2], dtype=torch.int32)
    active_counts = torch.tensor([63, 63], dtype=torch.int32)
    state_cache = torch.empty(3, 1, 128, 128)
    replay_count = torch.zeros(3, dtype=torch.int32)
    pool = torch.empty(3, 1, 128, 128)
    offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
    block_table = torch.tensor([1, 2], dtype=torch.int32)
    interval = 64
    match invalid:
        case "dtype":
            pool = pool.half()
        case "offsets":
            offsets = None
        case "block_table":
            block_table = None
        case "pool":
            pool = None
    with pytest.raises((TypeError, ValueError), match="checkpoint"):
        commit_paged_chunk_replay(
            final_state,
            slots,
            None,
            active_counts,
            state_cache,
            replay_count,
            pool,
            offsets,
            interval,
            block_table,
        )


@torch.no_grad()
def test_replay_checkpoint_commit_writes_a_pool_with_any_unit_head_stride(cuda: None) -> None:
    # A contiguous [slots, V, K] buffer viewed as [slots, 1, V, K] by transposition has head
    # stride slots * V * K; the head index is always zero, so the pool is still dense per slot.
    final = torch.randn(2, 1, 128, 128, device="cuda")
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    counts = torch.full((2,), 63, device="cuda", dtype=torch.int32)
    cache = torch.zeros(3, 1, 128, 128, device="cuda")
    replay_count = torch.zeros(3, device="cuda", dtype=torch.int32)
    pool = torch.full((4, 128, 128), CHECKPOINT_SENTINEL, device="cuda").unsqueeze(0)
    pool = pool.transpose(0, 1)
    assert pool.is_contiguous() and pool.stride(1) == 4 * 128 * 128
    block_table = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    saved = StateCheckpoints(pool, cumulative_sequence_offsets([1, 1]), 64, block_table)
    chunk_replay_commit_op(final, slots, None, counts, cache, replay_count, *saved)
    torch.testing.assert_close(pool[[3, 1]], final, rtol=0, atol=0)
    assert torch.all(pool[[0, 2]] == CHECKPOINT_SENTINEL)


@torch.no_grad()
def test_replay_checkpoint_commit_reuses_runtime_intervals(cuda: None, triton_compiles):
    for sequences, interval in ((9, 64), (16, 128), (17, 1024)):
        final = torch.arange(sequences * 128 * 128, device="cuda", dtype=torch.float32).reshape(
            sequences, 1, 128, 128
        )
        cache = torch.zeros(sequences + 1, 1, 128, 128, device="cuda")
        slots = torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32)
        counts = torch.full((sequences,), 63, device="cuda", dtype=torch.int32)
        replay_count = torch.zeros(sequences + 1, device="cuda", dtype=torch.int32)
        saved = sentinel_state_checkpoints(
            [int(interval == 64)] * sequences, interval, (1, 128, 128), spare=0
        )
        chunk_replay_commit_op(final, slots, None, counts, cache, replay_count, *saved)
        if interval == 64:
            torch.testing.assert_close(checkpoint_rows(saved), final, rtol=0, atol=0)
        else:
            assert torch.all(saved.state_pool == CHECKPOINT_SENTINEL)
    assert triton_compiles["_commit_state_kernel"] == 1


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3,
    reason="the executed-overflow regression allocates about 8.6 GiB",
)
@torch.no_grad()
def test_replay_checkpoint_commit_executes_offsets_past_int32() -> None:
    # The last request's first element sits past INT32_MAX: its program must widen before
    # multiplying by the block size, or its route, count and checkpoint row lookups wrap.
    sequences = 2**31 // (128 * 128) + 1
    final = torch.zeros(sequences, 1, 128, 128, device="cuda")
    final[-1].normal_()
    slots = torch.zeros(sequences, device="cuda", dtype=torch.int32)
    slots[-1] = 1
    counts = torch.zeros(sequences, device="cuda", dtype=torch.int32)
    counts[-1] = 63
    cache = torch.zeros(2, 1, 128, 128, device="cuda")
    replay_count = torch.full((2,), 63, device="cuda", dtype=torch.int32)
    saved = sentinel_state_checkpoints([0] * (sequences - 1) + [1], 64, (1, 128, 128), spare=0)
    chunk_replay_commit_op(final, slots, None, counts, cache, replay_count, *saved)
    torch.testing.assert_close(cache[1], final[-1], rtol=0, atol=0)
    torch.testing.assert_close(checkpoint_rows(saved)[0], final[-1], rtol=0, atol=0)
    assert replay_count[1].item() == 0

    # Reuse the allocation as the checkpoint pool for a one-request commit: only the store into
    # the last slot, at element offset 2**31, needs 64-bit addressing.
    source = torch.randn(1, 1, 128, 128, device="cuda")
    replay_count.fill_(63)
    into_last_slot = StateCheckpoints(
        final,
        cumulative_sequence_offsets([1]),
        64,
        torch.tensor([sequences - 1], device="cuda", dtype=torch.int32),
    )
    chunk_replay_commit_op(
        source, slots[-1:], None, counts[-1:], cache, replay_count, *into_last_slot
    )
    torch.testing.assert_close(final[-1], source[0], rtol=0, atol=0)


@pytest.mark.parametrize("prefix", [150, 128], ids=["mid-chunk", "chunk-boundary"])
@torch.no_grad()
def test_prefix_cache_example_restores_checkpoint_and_replay_tail(cuda: None, prefix: int) -> None:
    from examples.linear.kda_prefix_cache_checkpoints import restore_shared_prefix

    assert restore_shared_prefix(prefix) == 128
