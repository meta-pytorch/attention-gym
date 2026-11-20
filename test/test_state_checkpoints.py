"""Caller-owned checkpoints for inference-only paged delta-rule chunks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym.linear import (
    StateCheckpoints,
    chunk_gdn,
    chunk_kda,
    paged_chunk_gdn,
    paged_chunk_kda,
)
from attn_gym.linear._delta_rule.chunk_ops import _plain_gate_scan_op
from attn_gym.linear._delta_rule.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.testing import (
    CHECKPOINT_SENTINEL,
    checkpoint_rows,
    sentinel_state_checkpoints,
)
from attn_gym.testing.gdn import make_gdn_test_inputs
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

Rule = Literal["gdn", "kda"]


@pytest.fixture(params=["gdn", "kda"])
def rule(request: pytest.FixtureRequest) -> Rule:
    return request.param


@pytest.fixture
def cuda() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("paged chunk kernels require CUDA capability 8.0 or newer")


def operations(rule: Rule) -> tuple[Callable, Callable]:
    return (paged_chunk_gdn, chunk_gdn) if rule == "gdn" else (paged_chunk_kda, chunk_kda)


def options(rule: Rule) -> dict[str, object]:
    # Disable autotuning and use KDA's static factor/output schedules.
    return {"autotune": False, **({"schedule": "static"} if rule == "kda" else {})}


def inputs(rule: Rule, tokens: int, *, batch: int = 1) -> tuple[torch.Tensor, ...]:
    if rule == "kda":
        return make_kda_test_inputs(tokens, batch=batch, heads=2, gate_scale=0.1)
    return make_gdn_test_inputs(
        tokens, batch=batch, key_heads=2, value_heads=2, gate_pattern="near_zero"
    )[:5]


def checkpoints(lengths: list[int], interval: int, *, spare: int = 2) -> StateCheckpoints:
    counts = [length // interval for length in lengths]
    return sentinel_state_checkpoints(counts, interval, (2, 128, 128), spare=spare)


@pytest.mark.parametrize(
    "interval,state_dtype", [(64, torch.float32), (128, torch.bfloat16)], ids=["i64", "i128-bf16"]
)
@torch.no_grad()
def test_checkpoints_match_truncated_chunks_and_preserve_paged_results(
    rule: Rule, cuda: None, interval: int, state_dtype: torch.dtype
) -> None:
    # Covers an empty sequence, an incomplete interval, aligned final states, null routes that
    # reserve unwritten rows, a sequence that reserves a row more than it produces, and FP32
    # snapshots taken before BF16 pool rounding.
    paged, chunk = operations(rule)
    lengths = [0, interval - 1, 2 * interval, interval + 7, interval, interval]
    tensors = inputs(rule, sum(lengths))
    cu_seqlens = cumulative_sequence_offsets(lengths)
    routes = [5, 2, 4, 1, 0, -1]
    initial_flags = [False, True, True, False, True, False]
    slots = torch.tensor(routes, dtype=torch.int32, device="cuda")
    has_initial = torch.tensor(initial_flags, device="cuda")
    initial_cache = torch.randn(7, 2, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    # A fresh route must not read even nonfinite stale contents.
    initial_cache[1].fill_(float("nan"))
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()
    reserved = [length // interval for length in lengths]
    reserved[2] += 1
    saved = sentinel_state_checkpoints(reserved, interval, (2, 128, 128))
    expected_states = checkpoint_rows(saved).clone()
    kwargs = {"cu_seqlens": cu_seqlens, "has_initial_state": has_initial}

    expected_output = paged(
        *tensors, expected_cache, slots, **kwargs, kernel_options=options(rule)
    )
    actual_output = paged(
        *tensors,
        actual_cache,
        slots,
        **kwargs,
        state_checkpoints=saved,
        kernel_options=options(rule),
    )
    assert isinstance(actual_output, torch.Tensor)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)

    # Compare each post-boundary state to the same chunk arithmetic stopped at that boundary.
    token_start = checkpoint_start = 0
    for length, slot, has_state, rows in zip(
        lengths, routes, initial_flags, reserved, strict=True
    ):
        if slot > 0:
            initial = initial_cache[slot : slot + 1].float()
            if not has_state:
                initial = torch.zeros_like(initial)
            for index, stop in enumerate(range(interval, length + 1, interval)):
                prefix = tuple(tensor[:, token_start : token_start + stop] for tensor in tensors)
                _, final_state = chunk(
                    *prefix,
                    initial,
                    cu_seqlens=cumulative_sequence_offsets([stop]),
                    output_final_state=True,
                    kernel_options=options(rule),
                )
                expected_states[checkpoint_start + index].copy_(final_state[0])
        token_start += length
        checkpoint_start += rows

    torch.testing.assert_close(checkpoint_rows(saved), expected_states, rtol=0, atol=0)
    assert torch.all(saved.state_pool[0] == CHECKPOINT_SENTINEL)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_checkpoints_publish_prefix_states_into_the_cache(
    rule: Rule, cuda: None, state_dtype: torch.dtype, triton_compiles
) -> None:
    paged, chunk = operations(rule)
    lengths = [200, 130]
    tensors = inputs(rule, sum(lengths))
    cu_seqlens = cumulative_sequence_offsets(lengths)
    initial_cache = torch.randn(10, 2, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    # Rows: three for sequence 0 (after 64, 128, 192 tokens), two for sequence 1. They start at
    # offset 1, as in a padded block table: entry 0 names slot 4, which no sequence owns and
    # which must stay untouched. Entries alias the cache in unrouted slots and are runtime
    # data: the second request compiles nothing new.
    offsets = cumulative_sequence_offsets([3, 2]) + 1
    compiles = []
    for block_table, published in (
        ([4, 0, 7, 0, 9, 0], {7: (0, 128), 9: (200, 264)}),
        ([4, 0, 0, 9, 0, 7], {9: (0, 192), 7: (200, 328)}),
    ):
        cache = initial_cache.clone()
        paged(
            *tensors,
            cache,
            slots,
            cu_seqlens=cu_seqlens,
            state_checkpoints=StateCheckpoints(
                cache,
                offsets,
                64,
                torch.tensor(block_table, device="cuda", dtype=torch.int32),
            ),
            kernel_options=options(rule),
        )
        for slot, (start, stop) in published.items():
            route = 1 if start == 0 else 2
            _, expected = chunk(
                *(tensor[:, start:stop] for tensor in tensors),
                initial_cache[route : route + 1].float(),
                cu_seqlens=cumulative_sequence_offsets([stop - start]),
                output_final_state=True,
                kernel_options=options(rule),
            )
            torch.testing.assert_close(cache[slot], expected[0].to(state_dtype), rtol=0, atol=0)
        for slot in set(range(10)) - {1, 2, *published}:
            torch.testing.assert_close(cache[slot], initial_cache[slot], rtol=0, atol=0)
        compiles.append(sum(triton_compiles.values()))
    assert compiles[0] == compiles[1]


@torch.no_grad()
def test_dense_batch_checkpoints_include_aligned_final_state(rule: Rule, cuda: None) -> None:
    paged, chunk = operations(rule)
    tensors = inputs(rule, 128, batch=2)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    initial_cache = torch.randn(5, 2, 128, 128, device="cuda") * 0.01
    state_cache = initial_cache.clone()
    saved = checkpoints([128, 128], 128)
    paged(*tensors, state_cache, slots, state_checkpoints=saved, kernel_options=options(rule))
    _, expected = chunk(
        *tensors,
        initial_cache[slots.long()].clone(),
        output_final_state=True,
        kernel_options=options(rule),
    )
    rows = checkpoint_rows(saved)
    torch.testing.assert_close(rows[:2], expected, rtol=0, atol=0)
    torch.testing.assert_close(rows[:2], state_cache[slots.long()], rtol=0, atol=0)
    assert torch.all(rows[2:] == CHECKPOINT_SENTINEL)


@pytest.mark.parametrize("execution", ["fullgraph", "cuda_graph"])
@torch.no_grad()
def test_checkpoint_graph_replay_reads_changed_offsets_and_routes(
    rule: Rule, cuda: None, execution: str
) -> None:
    paged, _ = operations(rule)
    tensors = inputs(rule, 256)
    cu_seqlens = cumulative_sequence_offsets([128, 128])
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    has_initial = torch.tensor([True, False], device="cuda")
    initial_cache = torch.randn(5, 2, 128, 128, device="cuda") * 0.01
    state_cache = initial_cache.clone()
    saved = checkpoints([128, 128], 128)

    def run(cache, pool, sequence_offsets, checkpoint_offsets, routes, initial_flags):
        return paged(
            *tensors,
            cache,
            routes,
            cu_seqlens=sequence_offsets,
            has_initial_state=initial_flags,
            state_checkpoints=StateCheckpoints(pool, checkpoint_offsets, 128, saved.block_table),
            kernel_options=options(rule),
        )

    args = (
        state_cache,
        saved.state_pool,
        cu_seqlens,
        saved.cu_checkpoint_offsets,
        slots,
        has_initial,
    )
    run(*args)
    if execution == "fullgraph":
        replay = torch.compile(run, fullgraph=True)
        replay(*args)
    else:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(*args)

    # Both iterations reuse the captured tensor addresses. The second gives the
    # second sequence two checkpoints and makes the first route null.
    for lengths, routes, flags in (
        ([64, 192], [2, 4], [False, True]),
        ([0, 256], [0, 1], [True, False]),
    ):
        cu_seqlens.copy_(cumulative_sequence_offsets(lengths))
        slots.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
        has_initial.copy_(torch.tensor(flags, device="cuda"))
        saved.cu_checkpoint_offsets.copy_(
            cumulative_sequence_offsets([length // 128 for length in lengths])
        )
        state_cache.copy_(initial_cache)
        saved.state_pool.fill_(CHECKPOINT_SENTINEL)
        expected_cache = initial_cache.clone()
        expected_states = torch.full_like(saved.state_pool, CHECKPOINT_SENTINEL)
        expected_output = run(
            expected_cache,
            expected_states,
            cu_seqlens,
            saved.cu_checkpoint_offsets,
            slots,
            has_initial,
        )
        if execution == "fullgraph":
            output = replay(*args)
        else:
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(state_cache, expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(saved.state_pool, expected_states, rtol=0, atol=0)


def checkpoint_operator_inputs(rule: Rule) -> tuple[torch._ops.OpOverload, tuple]:
    q, k, v, gate, beta = inputs(rule, 128)
    # A noncontiguous value tensor checks that the fake output matches the real layout.
    v = v.movedim(1, -1).contiguous().movedim(-1, 1)
    cu_seqlens = cumulative_sequence_offsets([64, 64])
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    gate = _plain_gate_scan_op(
        gate.unsqueeze(-1) if rule == "gdn" else gate,
        cu_seqlens,
        metadata.chunk_offsets,
        False,
    )
    if rule == "gdn":
        gate = gate.squeeze(-1)
    state_cache = torch.randn(5, 2, 128, 128, device="cuda") * 0.01
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    has_initial = torch.tensor([True, False], device="cuda")
    args = (
        q,
        k,
        v,
        gate,
        beta,
        state_cache,
        slots,
        has_initial,
        cu_seqlens,
        metadata.chunk_offsets,
    )
    if rule == "gdn":
        from attn_gym.linear.gdn.ops import chunk_fwd_packed_paged_op

        op = chunk_fwd_packed_paged_op
        args += (metadata.capacity, 128**-0.5, False)
    else:
        from attn_gym.linear.kda.ops import chunk_fwd_ragged_paged_op

        op = chunk_fwd_ragged_paged_op
        args += (False, "static")
    return op, (*args, *checkpoints([64, 64], 64))


@pytest.mark.parametrize("into_state_cache", [False, True])
def test_checkpoint_paged_operator_registration(
    rule: Rule, cuda: None, into_state_cache: bool
) -> None:
    op, args = checkpoint_operator_inputs(rule)
    if into_state_cache:
        # Block-table entries alias the advanced pool: slots 2 and 4 are not routed.
        block_table = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
        args = (*args[:-4], args[5], args[-3], args[-2], block_table)
    torch.library.opcheck(op, args)


@pytest.mark.parametrize("invalid", ["dtype", "missing_block_table"])
def test_raw_checkpoint_operator_validates_before_launch(
    rule: Rule, cuda: None, invalid: str
) -> None:
    op, args = checkpoint_operator_inputs(rule)
    pool, offsets, interval, block_table = args[-4:]
    if invalid == "dtype":
        pool = pool.half()
    else:
        block_table = None
    with pytest.raises((TypeError, ValueError), match="checkpoint"):
        op(*args[:-4], pool, offsets, interval, block_table)


def fake_paged_call(
    rule: Rule,
    checkpoints: Callable[[StateCheckpoints], StateCheckpoints] | None = None,
    *,
    query_requires_grad: bool = False,
    **kwargs,
):
    """Run a public paged call on fake tensors, so only host-side validation executes."""
    paged, _ = operations(rule)
    with FakeTensorMode():
        q, k, v = (
            torch.empty(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        )
        q.requires_grad_(query_requires_grad)
        gate = torch.empty(q.shape if rule == "kda" else q.shape[:-1], device="cuda")
        beta = torch.empty(q.shape[:-1], device="cuda")
        cache = torch.empty(5, 2, 128, 128, device="cuda")
        slots = torch.empty(2, device="cuda", dtype=torch.int32)
        valid = StateCheckpoints(
            torch.empty(4, 2, 128, 128, device="cuda"),
            torch.empty(3, device="cuda", dtype=torch.int32),
            64,
            torch.empty(4, device="cuda", dtype=torch.int32),
        )
        return paged(
            q,
            k,
            v,
            gate,
            beta,
            cache,
            slots,
            cu_seqlens=torch.empty(3, device="cuda", dtype=torch.int32),
            state_checkpoints=valid if checkpoints is None else checkpoints(valid),
            **kwargs,
        )


# The validator is shared, so its full error matrix runs once; other entry points check one case.
INVALID_METADATA = {
    "zero_interval": lambda c: c._replace(every_n_tokens=0),
    "negative_interval": lambda c: c._replace(every_n_tokens=-64),
    "unaligned_interval": lambda c: c._replace(every_n_tokens=65),
    "bool_interval": lambda c: c._replace(every_n_tokens=True),
    "float_interval": lambda c: c._replace(every_n_tokens=64.0),
    "pool_dtype": lambda c: c._replace(state_pool=c.state_pool.half()),
    "pool_shape": lambda c: c._replace(state_pool=c.state_pool[:, :1]),
    "pool_slot_layout": lambda c: c._replace(state_pool=c.state_pool.transpose(-1, -2)),
    "pool_device": lambda c: c._replace(state_pool=c.state_pool.cpu()),
    "offset_dtype": lambda c: c._replace(cu_checkpoint_offsets=c.cu_checkpoint_offsets.long()),
    "offset_shape": lambda c: c._replace(cu_checkpoint_offsets=c.cu_checkpoint_offsets[:2]),
    "offset_stride": lambda c: c._replace(
        cu_checkpoint_offsets=torch.empty(6, device="cuda", dtype=torch.int32)[::2]
    ),
    "offset_device": lambda c: c._replace(cu_checkpoint_offsets=c.cu_checkpoint_offsets.cpu()),
    "autograd": lambda c: c._replace(state_pool=c.state_pool.requires_grad_()),
    "block_table_dtype": lambda c: c._replace(
        block_table=torch.empty(4, device="cuda", dtype=torch.int64)
    ),
    "block_table_shape": lambda c: c._replace(
        block_table=torch.empty(2, 2, device="cuda", dtype=torch.int32)
    ),
    "block_table_stride": lambda c: c._replace(
        block_table=torch.empty(8, device="cuda", dtype=torch.int32)[::2]
    ),
    "block_table_device": lambda c: c._replace(block_table=c.block_table.cpu()),
}


@pytest.mark.parametrize("invalid", list(INVALID_METADATA))
def test_checkpoint_metadata_validation_without_device_execution(invalid: str) -> None:
    with pytest.raises((ValueError, TypeError, RuntimeError), match="checkpoint|every_n_tokens"):
        fake_paged_call("gdn", INVALID_METADATA[invalid], kernel_options=options("gdn"))


def test_checkpoint_calls_reject_operand_autograd(rule: Rule) -> None:
    with pytest.raises(RuntimeError, match="inference.only"):
        fake_paged_call(rule, query_requires_grad=True, kernel_options=options(rule))


def test_kda_validates_checkpoint_metadata() -> None:
    with pytest.raises(TypeError, match="state_checkpoints"):
        fake_paged_call("kda", INVALID_METADATA["pool_dtype"], kernel_options=options("kda"))


def test_warpdrive_rejects_checkpoints_before_backend_dispatch(rule: Rule) -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        fake_paged_call(rule, kernel_options={"backend": "warpdrive"})


@torch.no_grad()
def test_checkpoint_intervals_and_shapes_reuse_recurrence(rule: Rule, cuda: None, triton_compiles):
    paged, _ = operations(rule)
    # Token intervals are multiples of 64, so one variant serves every interval below 2**31.
    for sequences, length, interval, spare in (
        (1, 65, 64, 0),
        (16, 1025, 1024, 16),
        (3, 193, 192, 3),
    ):
        tensors = inputs(rule, sequences * length)
        lengths = [length] * sequences
        cache = torch.zeros(sequences + 1, 2, 128, 128, device="cuda")
        slots = torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32)
        saved = checkpoints(lengths, interval, spare=spare)
        paged(
            *tensors,
            cache,
            slots,
            cu_seqlens=cumulative_sequence_offsets(lengths),
            state_checkpoints=saved,
            kernel_options=options(rule),
        )
        active = sequences * (length // interval)
        rows = checkpoint_rows(saved)
        assert torch.isfinite(rows[:active]).all()
        assert torch.all(rows[:active] != CHECKPOINT_SENTINEL)
        assert torch.all(rows[active:] == CHECKPOINT_SENTINEL)
    assert triton_compiles["chunk_delta_h_kernel"] == 1


@pytest.mark.parametrize("wide_offsets", [False, True])
@torch.no_grad()
def test_gdn_d64_grouped_checkpoints(cuda: None, monkeypatch, wide_offsets: bool) -> None:
    import importlib

    recurrence = importlib.import_module("attn_gym.linear.kda.fwd.triton.chunk_delta_h")
    if wide_offsets:
        monkeypatch.setattr(recurrence, "requires_int64_offsets", lambda *args: True)
    tensors = make_gdn_test_inputs(128, key_dim=64, key_heads=1, value_heads=2)[:5]
    cache = torch.zeros(2, 2, 64, 64, device="cuda")
    saved = sentinel_state_checkpoints([2], 64, (2, 64, 64), spare=0)
    paged_chunk_gdn(
        *tensors,
        cache,
        torch.tensor([1], device="cuda", dtype=torch.int32),
        state_checkpoints=saved,
        kernel_options={"autotune": False},
    )
    for row, stop in enumerate((64, 128)):
        _, expected = chunk_gdn(
            *(tensor[:, :stop] for tensor in tensors),
            cu_seqlens=cumulative_sequence_offsets([stop]),
            output_final_state=True,
            kernel_options={"autotune": False},
        )
        torch.testing.assert_close(checkpoint_rows(saved)[row], expected[0], rtol=0, atol=0)


@torch.no_grad()
def test_persistent_recurrence_writes_every_sequence_checkpoint(
    rule: Rule, cuda: None, triton_compiles
):
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("persistent recurrence requires Hopper or newer")
    paged, _ = operations(rule)
    sequences, heads, length = 128, 8, 64
    tensors = (
        make_kda_test_inputs(sequences * length, heads=heads)
        if rule == "kda"
        else make_gdn_test_inputs(sequences * length, key_heads=heads, value_heads=heads)[:5]
    )
    cache = torch.zeros(sequences + 1, heads, 128, 128, device="cuda")
    saved = sentinel_state_checkpoints([1] * sequences, length, (heads, 128, 128), spare=0)
    paged(
        *tensors,
        cache,
        torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32),
        cu_seqlens=cumulative_sequence_offsets([length] * sequences),
        state_checkpoints=saved,
        kernel_options={"autotune": False},
    )
    torch.testing.assert_close(checkpoint_rows(saved), cache[1:], rtol=0, atol=0)
    assert triton_compiles["chunk_delta_h_persistent_kernel"] == 1
