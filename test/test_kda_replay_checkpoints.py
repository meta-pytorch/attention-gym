"""Selected positions retain KDA's canonical 64-token replay boundaries."""

from __future__ import annotations

import math
import subprocess
import sys

import pytest
import torch

from attn_gym.linear import (
    ReplayState,
    StateCheckpoints,
    chunk_kda,
    copy_replay_tails,
    paged_chunk_kda,
)
from attn_gym.linear.kda.ops import chunk_replay_commit_op
from attn_gym.testing import (
    CHECKPOINT_SENTINEL,
    assert_checkpoints_into_cache_match,
    sentinel_state_checkpoints,
)
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

OPTIONS = {"schedule": "static", "autotune": False}


@pytest.fixture
def cuda() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("paged chunk KDA requires CUDA capability 8.0 or newer")


def int32(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, device="cuda", dtype=torch.int32)


def replay_buffers(counts: list[int]) -> ReplayState:
    return ReplayState(
        *make_kda_test_inputs(64, batch=len(counts), seed=73, gate_scale=0.1), int32(counts)
    )


def clone_replay(replay: ReplayState) -> ReplayState:
    return ReplayState(*(tensor.clone() for tensor in replay))


def assert_replay_equal(actual: ReplayState, expected: ReplayState) -> None:
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0, equal_nan=True)


def expected_checkpoints(
    tensors: tuple[torch.Tensor, ...],
    initial_cache: torch.Tensor,
    initial_replay: ReplayState,
    saved: StateCheckpoints,
    lengths: list[int],
    routes: list[int],
    flags: list[bool],
    counts: list[int],
    boundaries: list[int],
    *,
    packed: bool,
) -> torch.Tensor:
    """Stop independent, replay-prepended chunk calls at the expected boundaries."""
    expected = torch.full_like(saved.state_pool, CHECKPOINT_SENTINEL)
    token_start = 0
    for sequence, (length, slot, has_state, stop) in enumerate(
        zip(lengths, routes, flags, boundaries, strict=True)
    ):
        if stop:
            replayed = counts[slot] if has_state else 0
            new_tokens = tuple(
                tensor[:, token_start : token_start + length]
                if packed
                else tensor[sequence : sequence + 1, :length]
                for tensor in tensors
            )
            logical_tokens = tuple(
                torch.cat((cache[slot : slot + 1, :replayed], new), dim=1)[:, :stop]
                for cache, new in zip(initial_replay[:-1], new_tokens, strict=True)
            )
            initial = initial_cache[slot : slot + 1].float()
            _, state = chunk_kda(
                *logical_tokens,
                initial if has_state else torch.zeros_like(initial),
                cu_seqlens=cumulative_sequence_offsets([stop]),
                output_final_state=True,
                kernel_options=OPTIONS,
            )
            assert state is not None
            expected[saved.checkpoint_slots[sequence]].copy_(state[0])
        token_start += length
    return expected


@pytest.mark.parametrize(
    "mode,state_dtype",
    [
        ("packed", torch.float32),
        ("dense", torch.float32),
        ("decode", torch.float32),
        ("decode", torch.bfloat16),
    ],
)
@torch.no_grad()
def test_replay_checkpoints_match_chunk_boundaries(
    cuda: None, mode: str, state_dtype: torch.dtype, no_benchmarking: None
) -> None:
    packed = mode == "packed"
    if mode != "decode":
        lengths = [0, 1, 180, 155, 70, 64, 64] if packed else [180] * 7
        routes = [5, 2, 4, 1, 3, 0, -1]
        flags = [True, True, True, False, True, True, False]
        counts = [0, 63, 63, 25, 19, 11]
        # 19 replayed + 118 new selects boundary 128 (not the final boundary 192).
        # One past the input must skip instead of rounding down to an existing boundary.
        positions = [3, 1, 118, 73, lengths[4] + 1, 64, 64]
        boundaries = [0, 64, 128, 64, 0, 0, 0]
    else:
        lengths = [1] * 7
        routes = [4, 2, 1, 0, 5, -1, 3]
        flags = [True, True, False, True, True, False, True]
        counts = [0, 63, 63, 63, 62, 63]
        positions = [1, 1, 1, 1, 2, 1, 0]
        boundaries = [0, 64, 0, 0, 0, 0, 0]
    tensors = make_kda_test_inputs(
        sum(lengths) if packed else lengths[0], batch=1 if packed else len(lengths), gate_scale=0.1
    )
    cache = torch.randn(len(counts), 1, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    replay = replay_buffers(counts)
    # A fresh slot must not consume its stale state or replay tail.
    cache[1].fill_(float("nan"))
    for tensor in replay[:-1]:
        tensor[1].fill_(float("nan"))
    saved = sentinel_state_checkpoints(positions, (1, 128, 128))
    expected_states = expected_checkpoints(
        tensors, cache, replay, saved, lengths, routes, flags, counts, boundaries, packed=packed
    )
    kwargs = {
        "cu_seqlens": cumulative_sequence_offsets(lengths) if packed else None,
        "has_initial_state": torch.tensor(flags, device="cuda"),
        "kernel_options": OPTIONS,
    }
    slots = int32(routes)
    expected_cache, expected_replay = cache.clone(), clone_replay(replay)
    expected_output = paged_chunk_kda(
        *tensors, expected_cache, slots, replay_state=expected_replay, **kwargs
    )
    output = paged_chunk_kda(
        *tensors, cache, slots, replay_state=replay, state_checkpoints=saved, **kwargs
    )
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0, equal_nan=True)
    assert_replay_equal(replay, expected_replay)
    torch.testing.assert_close(saved.state_pool, expected_states, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["packed", "decode"])
@torch.no_grad()
def test_replay_checkpoints_written_into_the_state_cache_match_a_separate_pool(
    cuda: None, mode: str, no_benchmarking: None
) -> None:
    decode = mode == "decode"
    lengths = [1, 1, 1] if decode else [45, 173, 70]
    positions = [1, 1, 1] if decode else [2, 129, 69]
    tensors = make_kda_test_inputs(
        1 if decode else sum(lengths), batch=3 if decode else 1, gate_scale=0.1
    )
    initial_replay = replay_buffers([0, 63, 19, 63, 0, 0, 0])
    slots = int32([1, 2, 3])
    cu_seqlens = None if decode else cumulative_sequence_offsets(lengths)
    cache = torch.randn(7, 1, 128, 128, device="cuda") * 0.01

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

    reference = sentinel_state_checkpoints(positions, (1, 128, 128))
    assert_checkpoints_into_cache_match(
        run, cache, reference, [6, 4, -1], [0, 2] if decode else [0, 1, 2]
    )


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
    slots = int32([1, 2, 3])
    has_initial = torch.ones(3, device="cuda", dtype=torch.bool)
    cu_seqlens = None if decode else cumulative_sequence_offsets([45, 173, 166])
    saved = sentinel_state_checkpoints([1, 1, 1] if decode else [40, 140, 72], (1, 128, 128))

    def run(cache, replay, pool):
        return paged_chunk_kda(
            *tensors,
            cache,
            slots,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial,
            replay_state=replay,
            state_checkpoints=StateCheckpoints(
                pool, saved.checkpoint_positions, saved.checkpoint_slots
            ),
            kernel_options=OPTIONS,
        )

    run(state_cache, replay_state, saved.state_pool)
    if execution == "fullgraph":
        counter = CompileCounterWithBackend("inductor")
        compiled = torch.compile(run, fullgraph=True, backend=counter)
        compiled(state_cache, replay_state, saved.state_pool)
    else:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(state_cache, replay_state, saved.state_pool)

    for lengths, routes, flags, counts, positions, destinations, boundaries in (
        (
            [1, 1, 1] if decode else [45, 173, 166],
            [2, 4, 5],
            [True, False, True],
            [0, 0, 63, 0, 63, 19],
            [1, 1, 1] if decode else [44, 77, 111],
            [4, 3, 2],
            [64, 0, 0] if decode else [64, 64, 128],
        ),
        (
            [1, 1, 1] if decode else [0, 129, 255],
            [0, 4, 2],
            [True, False, True],
            [0, 0, 63, 0, 17, 0],
            [1, 1, 2] if decode else [1, 128, 12],
            [3, 2, 1],
            [0, 0, 0] if decode else [0, 128, 64],
        ),
    ):
        slots.copy_(int32(routes))
        has_initial.copy_(torch.tensor(flags, device="cuda"))
        if cu_seqlens is not None:
            cu_seqlens.copy_(cumulative_sequence_offsets(lengths))
        saved.checkpoint_positions.copy_(int32(positions))
        saved.checkpoint_slots.copy_(int32(destinations))
        saved.state_pool.fill_(CHECKPOINT_SENTINEL)
        state_cache.copy_(initial_cache)
        for destination, source in zip(replay_state, initial_replay, strict=True):
            destination.copy_(source)
        replay_state.count.copy_(int32(counts))
        expected_cache, expected_replay = state_cache.clone(), clone_replay(replay_state)
        expected_states = expected_checkpoints(
            tensors,
            initial_cache,
            initial_replay,
            saved,
            lengths,
            routes,
            flags,
            counts,
            boundaries,
            packed=not decode,
        )
        expected_output = run(expected_cache, expected_replay, torch.empty_like(saved.state_pool))
        if execution == "fullgraph":
            output = compiled(state_cache, replay_state, saved.state_pool)
        else:
            graph.replay()
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(state_cache, expected_cache, rtol=0, atol=0)
        assert_replay_equal(replay_state, expected_replay)
        torch.testing.assert_close(saved.state_pool, expected_states, rtol=0, atol=0)
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
    slots = int32([2, 3, 0, 4])
    flags = torch.tensor([True, True, True, False], device="cuda")
    active_counts = int32([63, 17, 0, 0])
    state_cache = torch.randn(6, 1, 128, 128, device="cuda")
    replay_count = torch.zeros(6, device="cuda", dtype=torch.int32)
    saved = sentinel_state_checkpoints([1] * 4, (1, 128, 128))
    args = (final_state, slots, flags, active_counts, state_cache, replay_count)
    if not wide_offsets:
        torch.library.opcheck(chunk_replay_commit_op, (*args, *saved))
        # Slot 5 is not routed and can receive the checkpoint in the state cache itself.
        alias = StateCheckpoints(state_cache, saved.checkpoint_positions, int32([5, 0, 0, 0]))
        torch.library.opcheck(chunk_replay_commit_op, (*args, *alias))
        saved.state_pool.fill_(CHECKPOINT_SENTINEL)
    chunk_replay_commit_op(*args, *saved)
    torch.testing.assert_close(saved.state_pool[1], final_state[0], rtol=0, atol=0)
    assert torch.all(saved.state_pool[2:] == CHECKPOINT_SENTINEL)
    assert torch.all(saved.state_pool[0] == CHECKPOINT_SENTINEL)


@pytest.mark.parametrize("invalid", ["dtype", "positions", "slots", "pool"])
def test_raw_replay_checkpoint_commit_rejects_invalid_metadata_without_device_execution(
    invalid: str,
) -> None:
    from attn_gym.linear.kda.fwd.triton.paged_replay import commit_paged_chunk_replay

    final_state = torch.empty(2, 1, 128, 128)
    slots = torch.tensor([1, 2], dtype=torch.int32)
    active_counts = torch.tensor([63, 63], dtype=torch.int32)
    state_cache = torch.empty(3, 1, 128, 128)
    replay_count = torch.zeros(3, dtype=torch.int32)
    pool = torch.empty_like(
        state_cache, dtype=torch.float16 if invalid == "dtype" else torch.float32
    )
    with pytest.raises((TypeError, ValueError), match="checkpoint"):
        commit_paged_chunk_replay(
            final_state,
            slots,
            None,
            active_counts,
            state_cache,
            replay_count,
            None if invalid == "pool" else pool,
            None if invalid == "positions" else slots,
            None if invalid == "slots" else slots,
        )


@torch.no_grad()
def test_replay_checkpoint_commit_writes_a_pool_with_any_unit_head_stride(cuda: None) -> None:
    # A singleton head may have arbitrary stride even though the pool is dense per slot.
    final = torch.randn(2, 1, 128, 128, device="cuda")
    slots = int32([1, 2])
    counts = int32([63, 63])
    cache = torch.zeros(3, 1, 128, 128, device="cuda")
    replay_count = torch.zeros(3, device="cuda", dtype=torch.int32)
    pool = (
        torch.full((4, 128, 128), CHECKPOINT_SENTINEL, device="cuda").unsqueeze(0).transpose(0, 1)
    )
    assert pool.is_contiguous() and pool.stride(1) == 4 * 128 * 128
    saved = StateCheckpoints(pool, int32([1, 1]), int32([3, 1]))
    chunk_replay_commit_op(final, slots, None, counts, cache, replay_count, *saved)
    torch.testing.assert_close(pool[[3, 1]], final, rtol=0, atol=0)
    assert torch.all(pool[[0, 2]] == CHECKPOINT_SENTINEL)


@torch.no_grad()
def test_replay_checkpoint_commit_reuses_runtime_positions(cuda: None, triton_compiles):
    for sequences, position in ((9, 1), (16, 0), (17, 1024)):
        final = torch.arange(sequences * 128 * 128, device="cuda", dtype=torch.float32).reshape(
            sequences, 1, 128, 128
        )
        cache = torch.zeros(sequences + 1, 1, 128, 128, device="cuda")
        slots = torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32)
        counts = torch.full_like(slots, 63)
        replay_count = torch.zeros(sequences + 1, device="cuda", dtype=torch.int32)
        saved = sentinel_state_checkpoints([position] * sequences, (1, 128, 128), spare=0)
        chunk_replay_commit_op(final, slots, None, counts, cache, replay_count, *saved)
        if position == 1:
            torch.testing.assert_close(saved.state_pool[1:], final, rtol=0, atol=0)
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
    # multiplying by the block size, or its route, count and checkpoint slot lookups wrap.
    sequences = 2**31 // (128 * 128) + 1
    final = torch.zeros(sequences, 1, 128, 128, device="cuda")
    final[-1].normal_()
    slots = torch.zeros(sequences, device="cuda", dtype=torch.int32)
    slots[-1] = 1
    counts = torch.zeros_like(slots)
    counts[-1] = 63
    cache = torch.zeros(2, 1, 128, 128, device="cuda")
    replay_count = int32([63, 63])
    saved = StateCheckpoints(
        torch.full_like(cache, CHECKPOINT_SENTINEL), slots.clone(), slots.clone()
    )
    chunk_replay_commit_op(final, slots, None, counts, cache, replay_count, *saved)
    torch.testing.assert_close(cache[1], final[-1], rtol=0, atol=0)
    torch.testing.assert_close(saved.state_pool[1], final[-1], rtol=0, atol=0)
    assert replay_count[1].item() == 0

    # Reuse the allocation as the checkpoint pool: only its store needs 64-bit addressing.
    source = torch.randn(1, 1, 128, 128, device="cuda")
    replay_count.fill_(63)
    into_last_slot = StateCheckpoints(final, int32([1]), int32([sequences - 1]))
    chunk_replay_commit_op(
        source, slots[-1:], None, counts[-1:], cache, replay_count, *into_last_slot
    )
    torch.testing.assert_close(final[-1], source[0], rtol=0, atol=0)


@pytest.mark.parametrize("execution", ["cuda_graph", "fullgraph"])
def test_batched_copy_replay_tails_reads_new_graph_metadata(cuda: None, execution: str) -> None:
    from torch._dynamo.testing import CompileCounterWithBackend

    lengths = [70, 90, 4, 9]
    offsets = cumulative_sequence_offsets(lengths)
    layout = ReplayState.slot_layout(2, 96, value_dim=40)
    # Head-major projections: one sequence's token features need not be flattenable as a row.
    inputs = tuple(
        torch.randn(1, shape[1], sum(lengths), *shape[2:], device="cuda", dtype=dtype).transpose(
            1, 2
        )
        for shape, dtype in layout[:-1]
    )
    # Token-innermost FP32 K exercises the feature stride and the BF16 replay-storage cast.
    k = torch.randn(1, 2, 96, sum(lengths), device="cuda").movedim(-1, 1)
    inputs = (inputs[0], k, *inputs[2:])
    # Include a guard row at each end of every slot. Comparing the entire allocation catches
    # padding writes, writes to unselected slots, and writes past a short tail.
    pages = tuple(
        torch.full((8, 66, *shape[1:]), -7, device="cuda", dtype=dtype)
        for shape, dtype in layout[:-1]
    )
    counts = torch.full((8, 2), 37, device="cuda", dtype=torch.int32)
    replay = ReplayState(*(p[:, 1:65] for p in pages), counts[:, :1])
    positions, slots, sources = (int32([0] * 4) for _ in range(3))
    has_initial = torch.ones(4, device="cuda", dtype=torch.bool)

    def copy() -> None:
        copy_replay_tails(
            replay,
            inputs,
            cu_seqlens=offsets,
            state_indices=sources,
            has_initial_state=has_initial,
            checkpoint_positions=positions,
            checkpoint_slots=slots,
        )

    if execution == "fullgraph":
        counter = CompileCounterWithBackend("inductor")
        run = torch.compile(copy, fullgraph=True, backend=counter)
        run()
    else:
        copy()  # compile before capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copy()
        run = graph.replay

    for seq_lengths, pos, dest, routes, flags, source_counts, writes in (
        # Mid-chunk, fresh (stale count) exactly on a boundary, unselected, no boundary.
        (
            lengths,
            [50, 64, 4, 9],
            [3, 1, -1, 2],
            [6, 5, 7, 4],
            [True, False, True, False],
            [0, 17, 41, 37, 63, 37, 19, 63],
            [(3, 45, 5), (1, 134, 0)],
        ),
        # Changed offsets: past-end, long tail spanning several tiles, one token after 63, null.
        (
            [67, 92, 4, 10],
            [68, 90, 1, 9],
            [1, 2, 3, 0],
            [5, 6, 4, 7],
            [True] * 4,
            [0, 17, 41, 37, 63, 0, 37, 63],
            [(2, 94, 63), (3, 160, 0)],
        ),
        # Neither a nonpositive source nor a nonpositive position publishes a checkpoint.
        (
            [67, 92, 4, 10],
            [65, 70, 0, 9],
            [1, 3, 2, 5],
            [0, -1, 4, 6],
            [True] * 4,
            [0, 17, 41, 37, 63, 0, 60, 63],
            [(5, 167, 5)],
        ),
    ):
        offsets.copy_(cumulative_sequence_offsets(seq_lengths))
        positions.copy_(int32(pos))
        slots.copy_(int32(dest))
        sources.copy_(int32(routes))
        has_initial.copy_(torch.tensor(flags, device="cuda"))
        counts[:, 0].copy_(int32(source_counts))
        expected = [page.clone() for page in pages]
        expected_counts = counts.clone()
        for slot, first, tail in writes:
            expected_counts[slot, 0] = tail
            for page, source in zip(expected, inputs, strict=True):
                page[slot, 1 : 1 + tail].copy_(source[0, first : first + tail])
        run()
        for actual, reference in zip((*pages, counts), (*expected, expected_counts), strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    if execution == "fullgraph":
        assert counter.frame_count == 1


@pytest.mark.parametrize("layout_kind", ["interleaved", "slabs"])
def test_copy_replay_tails_into_mixed_dtype_pages(cuda: None, layout_kind: str) -> None:
    # Serving engines can carve all five buffers and the count out of the same slot page.
    # Exercise actual shared storage under CUDA capture (fullgraph uses separate allocations).
    layout = ReplayState.slot_layout(1, 128)
    sizes = [math.prod(shape) * dtype.itemsize for shape, dtype in layout]
    storage = torch.zeros(4, sum(sizes) + 12, device="cuda", dtype=torch.uint8)

    def views(pages):
        fields = []
        start = 0
        for (shape, dtype), size in zip(layout, sizes, strict=True):
            if layout_kind == "slabs":
                field = pages.view(-1)[4 * start : 4 * (start + size)]
            else:
                field = pages[:, start : start + size]
            fields.append(field.view(dtype).view(4, *shape))
            start += size
        return ReplayState(*fields)

    replay = views(storage)
    replay.count[1] = 63
    inputs = make_kda_test_inputs(3)
    metadata = {
        "cu_seqlens": int32([0, 3]),
        "state_indices": int32([1]),
        "checkpoint_positions": int32([3]),
        "checkpoint_slots": int32([2]),
    }
    expected = storage.clone()
    reference = views(expected)
    reference.count[2] = 2
    for cache, tokens in zip(reference[:-1], inputs, strict=True):
        cache[2, :2].copy_(tokens[0, 1:3])
    copy_replay_tails(replay, inputs, **metadata)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_replay_tails(replay, inputs, **metadata)
    for field in replay:
        field[2].zero_()
    graph.replay()
    torch.testing.assert_close(storage, expected, rtol=0, atol=0)


def test_copy_replay_tails_rejects_nonzero_first_offset(cuda: None) -> None:
    # A device assert poisons the CUDA context, so run the malformed call in its own process.
    code = """
import torch
from attn_gym.linear import ReplayState, copy_replay_tails
from attn_gym.testing.kda import make_kda_test_inputs

def ints(values):
    return torch.tensor(values, device="cuda", dtype=torch.int32)

replay = ReplayState(*make_kda_test_inputs(64, batch=3), ints([0, 0, 0]))
copy_replay_tails(replay, make_kda_test_inputs(66), cu_seqlens=ints([1, 66]),
                  state_indices=ints([1]), checkpoint_positions=ints([65]),
                  checkpoint_slots=ints([2]))
torch.cuda.synchronize()
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=60
    )
    assert run.returncode != 0 and "invalid packed cu_seqlens" in run.stderr, run.stderr


@pytest.mark.parametrize("token_innermost", [False, True])
def test_copy_replay_tails_operator_registration(cuda: None, token_innermost: bool) -> None:
    inputs = make_kda_test_inputs(5)
    if token_innermost:
        inputs = tuple(t.movedim(1, -1).contiguous().movedim(-1, 1) for t in inputs)
    replay = replay_buffers([0, 61, 62, 0, 0])
    torch.library.opcheck(
        torch.ops.attn_gym.kda_copy_replay_tails.default,
        (*inputs, *replay, int32([0, 3, 5]), int32([2, 1]), None, int32([3, 2]), int32([4, 3])),
    )


def test_copy_replay_tails_requires_inference_before_compilation(cuda: None) -> None:
    inputs = make_kda_test_inputs(3, requires_grad=True)
    replay = replay_buffers([0, 63, 0])
    metadata = {
        "cu_seqlens": int32([0, 3]),
        "state_indices": int32([1]),
        "checkpoint_positions": int32([3]),
        "checkpoint_slots": int32([2]),
    }

    def copy():
        copy_replay_tails(replay, inputs, **metadata)

    compiled = torch.compile(copy, backend="aot_eager", fullgraph=True)
    with pytest.raises(RuntimeError, match="inference-only"):
        copy()
    with pytest.raises(torch._dynamo.exc.Unsupported, match="[Ee]xception"):
        compiled()
    assert replay.count[2].item() == 0
    with torch.no_grad():
        compiled()
    assert replay.count[2].item() == 2


def test_copy_replay_tails_rejects_positions_aliasing_mutated_count(cuda: None) -> None:
    replay = replay_buffers([0, 63, 2])
    with pytest.raises(ValueError, match="alias|overlap"):
        copy_replay_tails(
            replay,
            make_kda_test_inputs(2),
            cu_seqlens=int32([0, 2]),
            state_indices=int32([1]),
            checkpoint_positions=replay.count[2:],
            checkpoint_slots=int32([2]),
        )


def test_copy_replay_tails_reuses_binary_across_packed_shapes(cuda: None, triton_compiles) -> None:
    # Cross 1, divisible-by-16 and odd classes for sequence count and packed token capacity.
    for lengths, capacity in (([1], 3), ([2] * 16, 48), ([2] * 16 + [3], 35)):
        sequences = len(lengths)
        # The feature stride scales with total tokens instead of being a model constant.
        inputs = tuple(
            t.movedim(1, -1).contiguous().movedim(-1, 1)
            for t in make_kda_test_inputs(sum(lengths))
        )
        replay = replay_buffers([0] + [63] * sequences + [37] * (capacity - sequences - 1))
        sources = torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32)
        destinations = sources + sequences
        torch.ops.attn_gym.kda_copy_replay_tails.default(
            *inputs,
            *replay,
            cumulative_sequence_offsets(lengths),
            sources,
            None,
            int32(lengths),
            destinations,
        )
        torch.testing.assert_close(replay.count[destinations], int32([n - 1 for n in lengths]))
    assert triton_compiles.total() == 1


@pytest.mark.parametrize("prefix", [150, 128], ids=["mid-chunk", "chunk-boundary"])
@torch.no_grad()
def test_prefix_cache_example_restores_checkpoint_and_replay_tail(cuda: None, prefix: int) -> None:
    from examples.linear.kda_prefix_cache_checkpoints import restore_shared_prefix

    assert restore_shared_prefix(prefix) == 128
