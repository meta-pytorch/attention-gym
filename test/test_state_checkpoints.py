"""Caller-owned checkpoints for inference-only paged delta-rule chunks."""

from __future__ import annotations

import importlib
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
from attn_gym.testing import CHECKPOINT_SENTINEL, sentinel_state_checkpoints, strided_state_pool
from attn_gym.testing.gdn import make_gdn_test_inputs
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

Rule = Literal["gdn", "kda"]
# Keep the Triton registration and specialization probes on their intended backend.
pytestmark = pytest.mark.usefixtures("fused_chunk_default")


@pytest.fixture(params=["gdn", "kda"])
def rule(request: pytest.FixtureRequest) -> Rule:
    return request.param


@pytest.fixture
def cuda() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("paged chunk kernels require CUDA capability 8.0 or newer")


def operations(rule: Rule) -> tuple[Callable, Callable]:
    return (paged_chunk_gdn, chunk_gdn) if rule == "gdn" else (paged_chunk_kda, chunk_kda)


def options(rule: Rule, backend: str = "fused") -> dict[str, object]:
    if backend == "warpdrive":
        # Truncated single-sequence references must round like the original pack.
        return {"backend": backend, "batch_invariant": True}
    # Disable autotuning and use KDA's static factor/output schedules.
    return {
        "backend": backend,
        "autotune": False,
        **({"schedule": "static"} if rule == "kda" else {}),
    }


def inputs(
    rule: Rule, tokens: int, *, batch: int = 1, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, ...]:
    if rule == "kda":
        return make_kda_test_inputs(tokens, batch=batch, heads=2, gate_scale=0.1, dtype=dtype)
    return make_gdn_test_inputs(
        tokens, batch=batch, key_heads=2, value_heads=2, gate_pattern="near_zero", dtype=dtype
    )[:5]


def ints(values: list[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def checkpoints(
    positions: list[int],
    slots: list[int],
    pool_size: int,
) -> StateCheckpoints:
    pool = torch.full((pool_size, 2, 128, 128), CHECKPOINT_SENTINEL, device="cuda")
    return StateCheckpoints(pool, ints(positions), ints(slots))


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_checkpoints_match_truncated_chunks_and_preserve_paged_results(
    rule: Rule, cuda: None, state_dtype: torch.dtype, checkpoint_backend: str
) -> None:
    opts = options(rule, checkpoint_backend)
    paged, chunk = operations(rule)
    # Only sequences 2 and 3 qualify. The others cover empty and short sequences, null
    # routes, unaligned and nonpositive positions, and nonpositive checkpoint slots.
    lengths = [0, 120, 128, 135, 128, 128, 128, 128, 128, 128, 128]
    routes = [1, 2, 3, 4, 0, -1, 5, 6, 7, 8, 9]
    initial_flags = [True, True, True, False, True, False, True, True, True, True, True]
    saved = checkpoints(
        # 80 is a warpdrive KDA BT16 step but not a public checkpoint boundary.
        [64, 128, 128, 64, 64, 64, 80, 0, -64, 64, 64],
        [1, 2, 7, 6, 5, 4, 3, 8, 9, 0, -1],
        pool_size=11,
    )
    tensors = inputs(rule, sum(lengths))
    cu_seqlens = cumulative_sequence_offsets(lengths)
    slots = ints(routes)
    has_initial = torch.tensor(initial_flags, device="cuda")
    initial_cache = torch.randn(10, 2, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    # A fresh route must not read even nonfinite stale contents.
    initial_cache[4].fill_(float("nan"))
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()
    expected_states = saved.state_pool.clone()
    kwargs = {"cu_seqlens": cu_seqlens, "has_initial_state": has_initial}

    expected_output = paged(*tensors, expected_cache, slots, **kwargs, kernel_options=opts)
    actual_output = paged(
        *tensors,
        actual_cache,
        slots,
        **kwargs,
        state_checkpoints=saved,
        kernel_options=opts,
    )
    assert isinstance(actual_output, torch.Tensor)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)

    for seq, stop, target in ((2, 128, 7), (3, 64, 6)):
        start = sum(lengths[:seq])
        route = routes[seq]
        initial = initial_cache[route : route + 1].float()
        _, final_state = chunk(
            *(tensor[:, start : start + stop] for tensor in tensors),
            initial if initial_flags[seq] else torch.zeros_like(initial),
            cu_seqlens=cumulative_sequence_offsets([stop]),
            output_final_state=True,
            kernel_options=opts,
        )
        expected_states[target].copy_(final_state[0])
    torch.testing.assert_close(saved.state_pool, expected_states, rtol=0, atol=0)


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_checkpoints_publish_prefix_states_into_the_cache(
    rule: Rule, cuda: None, state_dtype: torch.dtype, checkpoint_backend: str
) -> None:
    opts = options(rule, checkpoint_backend)
    paged, chunk = operations(rule)
    lengths = [200, 130]
    tensors = inputs(rule, sum(lengths))
    cu_seqlens = cumulative_sequence_offsets(lengths)
    initial_cache = torch.randn(10, 2, 128, 128, device="cuda", dtype=state_dtype) * 0.01
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)

    def publish(cache, positions, destinations):
        return paged(
            *tensors,
            cache,
            slots,
            cu_seqlens=cu_seqlens,
            state_checkpoints=StateCheckpoints(cache, positions, destinations),
            kernel_options=opts,
        )

    expected_cache = initial_cache.clone()
    expected_output = paged(
        *tensors, expected_cache, slots, cu_seqlens=cu_seqlens, kernel_options=opts
    )
    # Pass the slot-strided cache only once: the compiled call must preserve both writes.
    for run, positions, destinations in (
        (publish, [128, 64], [7, 9]),
        (torch.compile(publish, fullgraph=True), [192, 128], [9, 7]),
    ):
        storage, cache = strided_state_pool(10, 2, 128, 128, dtype=state_dtype)
        storage.fill_(CHECKPOINT_SENTINEL)
        cache.copy_(initial_cache)
        output = run(cache, ints(positions), ints(destinations))
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        torch.testing.assert_close(cache[slots], expected_cache[slots], rtol=0, atol=0)
        for seq, (position, slot) in enumerate(zip(positions, destinations, strict=True)):
            start = sum(lengths[:seq])
            stop = start + position
            route = seq + 1
            _, expected = chunk(
                *(tensor[:, start:stop] for tensor in tensors),
                initial_cache[route : route + 1].float(),
                cu_seqlens=cumulative_sequence_offsets([stop - start]),
                output_final_state=True,
                kernel_options=opts,
            )
            torch.testing.assert_close(cache[slot], expected[0].to(state_dtype), rtol=0, atol=0)
        for slot in set(range(10)) - {1, 2, *destinations}:
            torch.testing.assert_close(cache[slot], initial_cache[slot], rtol=0, atol=0)
        assert torch.all(storage[:, :11] == CHECKPOINT_SENTINEL)
        assert torch.all(storage[:, -17:] == CHECKPOINT_SENTINEL)


@torch.no_grad()
def test_dense_batch_checkpoints_include_aligned_final_state(
    rule: Rule, cuda: None, checkpoint_backend: str
) -> None:
    opts = options(rule, checkpoint_backend)
    paged, chunk = operations(rule)
    # Fused paged KDA casts Q/K/V to BF16; warpdrive preserves FP16 in both calls.
    dtype = torch.float16 if checkpoint_backend == "warpdrive" else torch.bfloat16
    tensors = inputs(rule, 128, batch=2, dtype=dtype)
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    initial_cache = torch.randn(5, 2, 128, 128, device="cuda") * 0.01
    state_cache = initial_cache.clone()
    saved = checkpoints([128, 128], [2, 1], pool_size=4)
    paged(*tensors, state_cache, slots, state_checkpoints=saved, kernel_options=opts)
    _, expected = chunk(
        *tensors,
        initial_cache[slots.long()].clone(),
        output_final_state=True,
        kernel_options=opts,
    )
    rows = saved.state_pool[[2, 1]]
    torch.testing.assert_close(rows, expected, rtol=0, atol=0)
    torch.testing.assert_close(rows, state_cache[slots.long()], rtol=0, atol=0)
    assert torch.all(saved.state_pool[[0, 3]] == CHECKPOINT_SENTINEL)


@pytest.mark.parametrize("execution", ["fullgraph", "cuda_graph"])
@torch.no_grad()
def test_checkpoint_graph_replay_reads_changed_positions_slots_and_routes(
    rule: Rule, cuda: None, execution: str, checkpoint_backend: str
) -> None:
    opts = options(rule, checkpoint_backend)
    paged, _ = operations(rule)
    tensors = inputs(rule, 256)
    cu_seqlens = cumulative_sequence_offsets([128, 128])
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    has_initial = torch.tensor([True, False], device="cuda")
    initial_cache = torch.randn(5, 2, 128, 128, device="cuda") * 0.01
    state_cache = initial_cache.clone()
    saved = checkpoints([128, 128], [2, 1], pool_size=4)

    def run(cache, pool, positions, destinations, sequence_offsets, routes, initial_flags):
        return paged(
            *tensors,
            cache,
            routes,
            cu_seqlens=sequence_offsets,
            has_initial_state=initial_flags,
            state_checkpoints=StateCheckpoints(pool, positions, destinations),
            kernel_options=opts,
        )

    args = (state_cache, *saved, cu_seqlens, slots, has_initial)
    run(*args)
    if execution == "fullgraph":
        replay = torch.compile(run, fullgraph=True)
        replay(*args)
    else:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(*args)

    # Reuse the captured tensor addresses, moving the checkpoint to a different boundary
    # and destination and then making one route null.
    for lengths, routes, flags, positions, destinations in (
        ([64, 192], [2, 4], [False, True], [64, 128], [1, 3]),
        ([0, 256], [0, 1], [True, False], [64, 192], [1, 2]),
    ):
        cu_seqlens.copy_(cumulative_sequence_offsets(lengths))
        slots.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
        has_initial.copy_(torch.tensor(flags, device="cuda"))
        saved.checkpoint_positions.copy_(ints(positions))
        saved.checkpoint_slots.copy_(ints(destinations))
        state_cache.copy_(initial_cache)
        saved.state_pool.fill_(CHECKPOINT_SENTINEL)
        expected_cache = initial_cache.clone()
        expected_states = torch.full_like(saved.state_pool, CHECKPOINT_SENTINEL)
        expected_output = run(
            expected_cache,
            expected_states,
            saved.checkpoint_positions,
            saved.checkpoint_slots,
            cu_seqlens,
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


def checkpoint_operator_inputs(
    rule: Rule, backend: str = "fused"
) -> tuple[torch._ops.OpOverload, tuple]:
    q, k, v, gate, beta = inputs(rule, 128)
    # A noncontiguous value tensor checks that the fake output matches the real layout.
    v = v.movedim(1, -1).contiguous().movedim(-1, 1)
    cu_seqlens = cumulative_sequence_offsets([64, 64])
    state_cache = torch.randn(5, 2, 128, 128, device="cuda") * 0.01
    slots = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    has_initial = torch.tensor([True, False], device="cuda")
    saved = checkpoints([64, 64], [2, 4], pool_size=5)
    if backend == "warpdrive":
        args = (q, k, v, gate, beta, state_cache, slots, has_initial, cu_seqlens, 128**-0.5)
        if rule == "gdn":
            from attn_gym.linear.gdn.impl.warpdrive_ops import (
                chunk_gdn_warpdrive_packed_fwd_paged_op as op,
            )
        else:
            from attn_gym.linear.kda.impl.warpdrive_ops import (
                chunk_warpdrive_packed_fwd_paged_op as op,
            )

            args += ("bf16",)
        return op, (*args, False, *saved)

    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    gate = _plain_gate_scan_op(
        gate.unsqueeze(-1) if rule == "gdn" else gate,
        cu_seqlens,
        metadata.chunk_offsets,
        False,
    )
    if rule == "gdn":
        gate = gate.squeeze(-1)
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
    return op, (*args, *saved)


@pytest.mark.parametrize("into_state_cache", [False, True])
def test_checkpoint_paged_operator_registration(
    rule: Rule, cuda: None, into_state_cache: bool, checkpoint_backend: str
) -> None:
    op, args = checkpoint_operator_inputs(rule, checkpoint_backend)
    if into_state_cache:
        # Destinations 2 and 4 are not routed by this call.
        args = (*args[:-3], args[5], *args[-2:])
    torch.library.opcheck(op, args)


@pytest.mark.parametrize(
    "invalid", ["dtype", "checkpoint_states", "checkpoint_positions", "checkpoint_slots"]
)
def test_raw_checkpoint_operator_validates_before_launch(
    rule: Rule, cuda: None, invalid: str, checkpoint_backend: str
) -> None:
    op, args = checkpoint_operator_inputs(rule, checkpoint_backend)
    pool, positions, slots = args[-3:]
    kwargs = {
        "checkpoint_states": pool,
        "checkpoint_positions": positions,
        "checkpoint_slots": slots,
    }
    if invalid == "dtype":
        kwargs["checkpoint_states"] = pool.half()
    else:
        kwargs[invalid] = None
    with pytest.raises((TypeError, ValueError), match="checkpoint"):
        op(*args[:-3], **kwargs)


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
            torch.empty(2, device="cuda", dtype=torch.int32),
            torch.empty(2, device="cuda", dtype=torch.int32),
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
    "pool_dtype": lambda c: c._replace(state_pool=c.state_pool.half()),
    "pool_shape": lambda c: c._replace(state_pool=c.state_pool[:, :1]),
    "pool_slot_layout": lambda c: c._replace(state_pool=c.state_pool.transpose(-1, -2)),
    "pool_device": lambda c: c._replace(state_pool=c.state_pool.cpu()),
    "autograd": lambda c: c._replace(state_pool=c.state_pool.requires_grad_()),
}


@pytest.mark.parametrize("field", ["checkpoint_positions", "checkpoint_slots"])
@pytest.mark.parametrize("invalid", ["dtype", "short", "rank", "stride", "device"])
def test_checkpoint_vectors_must_be_contiguous_int32_per_sequence(
    field: str, invalid: str
) -> None:
    def bad_vector(c: StateCheckpoints) -> StateCheckpoints:
        vector = getattr(c, field)
        match invalid:
            case "dtype":
                vector = vector.long()
            case "short":
                vector = vector[:1]
            case "rank":
                vector = vector.view(1, 2)
            case "stride":
                vector = vector.new_empty(4)[::2]
            case "device":
                vector = vector.cpu()
        return c._replace(**{field: vector})

    with pytest.raises((ValueError, TypeError), match=field):
        fake_paged_call("gdn", bad_vector, kernel_options=options("gdn"))


@pytest.mark.parametrize("invalid", list(INVALID_METADATA))
def test_checkpoint_metadata_validation_without_device_execution(invalid: str) -> None:
    with pytest.raises((ValueError, TypeError, RuntimeError), match="checkpoint"):
        fake_paged_call("gdn", INVALID_METADATA[invalid], kernel_options=options("gdn"))


def test_checkpoint_calls_reject_operand_autograd(rule: Rule) -> None:
    with pytest.raises(RuntimeError, match="inference.only"):
        fake_paged_call(rule, query_requires_grad=True, kernel_options=options(rule))


def test_kda_validates_checkpoint_metadata() -> None:
    with pytest.raises(TypeError, match="state_checkpoints"):
        fake_paged_call("kda", INVALID_METADATA["pool_dtype"], kernel_options=options("kda"))


@pytest.mark.parametrize("plan", ["auto_plan", "pinned_plan", "full_width"])
@torch.no_grad()
def test_warpdrive_checkpoint_restores_selected_prefix(
    rule: Rule, requires_blackwell: None, plan: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("cutlass")
    paged, _ = operations(rule)
    opts = {"backend": "warpdrive", "batch_invariant": plan == "pinned_plan"}
    if rule == "kda" and plan == "full_width":
        from attn_gym.linear._delta_rule.warpdrive.bundles import kda_warmup_forward

        monkeypatch.setattr(kda_warmup_forward, "requires_int64_abi", lambda *args: True)
    length, stop = 197, 128
    tensors = inputs(rule, length)
    # With two heads, this many empty sequences fills the grid without extra token work.
    # Keeping the same pack for the truncated oracle retains the full 128-wide value tile.
    empty = (
        [0] * (torch.cuda.get_device_properties().multi_processor_count // 2)
        if plan == "full_width"
        else []
    )

    def run(pool, slot, start, end, saved=None):
        return paged(
            *(x[:, start:end] for x in tensors),
            pool,
            ints([slot, *empty]),
            cu_seqlens=cumulative_sequence_offsets([end - start, *empty]),
            state_checkpoints=saved,
            kernel_options=opts,
        )

    # A separate FP32 checkpoint must not be rounded through the BF16 running pool.
    cache = torch.randn(3, 2, 128, 128, device="cuda", dtype=torch.bfloat16) * 0.01
    storage, pool = strided_state_pool(4, 2, 128, 128)
    storage.fill_(CHECKPOINT_SENTINEL)
    saved = StateCheckpoints(pool, ints([stop, *empty]), ints([2, *empty]))
    baseline = cache.clone()
    uninterrupted = run(baseline, 1, 0, length)
    actual = cache.clone()
    output = run(actual, 1, 0, length, saved)
    assert torch.equal(output, uninterrupted)
    assert torch.equal(actual, baseline)
    prefix = cache.float()
    run(prefix, 1, 0, stop)
    assert torch.equal(saved.state_pool[2], prefix[1])
    assert torch.all(saved.state_pool[[0, 1, 3]] == CHECKPOINT_SENTINEL)
    assert torch.all(storage[:, :11] == CHECKPOINT_SENTINEL)
    assert torch.all(storage[:, -17:] == CHECKPOINT_SENTINEL)
    suffix = run(saved.state_pool, 2, stop, length)
    assert torch.equal(suffix, uninterrupted[:, stop:])
    assert torch.equal(saved.state_pool[2].bfloat16(), baseline[1])


@torch.no_grad()
def test_checkpoint_positions_and_shapes_reuse_recurrence(
    rule: Rule,
    cuda: None,
    triton_compiles,
    checkpoint_backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    paged, _ = operations(rule)
    opts = options(rule, checkpoint_backend)
    if checkpoint_backend == "warpdrive":
        from attn_gym.linear._delta_rule.warpdrive.drivers.plan import ForwardPlan

        # Even the batch-invariant plan can split V on an underfilled grid. Count binaries
        # for one tile width, independently of the SM count of the device under test.
        monkeypatch.setattr(ForwardPlan, "uncut", lambda _n, _h, _v, sms: ForwardPlan(0, 1, sms))
        compile_fn = importlib.import_module(
            f"attn_gym.linear._delta_rule.warpdrive.bundles.{rule}_warmup_forward"
        )._compile_warmup_forward
        compile_fn.cache_clear()
    for sequences, length, position, spare in (
        (1, 65, 64, 0),
        (16, 1025, 1024, 16),
        (3, 193, 192, 3),
    ):
        tensors = inputs(rule, sequences * length)
        lengths = [length] * sequences
        cache = torch.zeros(sequences + 1, 2, 128, 128, device="cuda")
        slots = torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32)
        saved = sentinel_state_checkpoints([position] * sequences, (2, 128, 128), spare=spare)
        paged(
            *tensors,
            cache,
            slots,
            cu_seqlens=cumulative_sequence_offsets(lengths),
            state_checkpoints=saved,
            kernel_options=opts,
        )
        rows = saved.state_pool[1 : sequences + 1]
        assert torch.isfinite(rows).all()
        assert torch.all(rows != CHECKPOINT_SENTINEL)
        assert torch.all(saved.state_pool[sequences + 1 :] == CHECKPOINT_SENTINEL)
        assert torch.all(saved.state_pool[0] == CHECKPOINT_SENTINEL)
    if checkpoint_backend == "warpdrive":
        info = compile_fn.cache_info()
        assert info.currsize == 1
        assert info.hits + info.misses == 3
    else:
        assert triton_compiles["chunk_delta_h_kernel"] == 1


@pytest.mark.parametrize("wide_offsets", [False, True])
@torch.no_grad()
def test_gdn_d64_grouped_checkpoints(
    cuda: None, monkeypatch, wide_offsets: bool, checkpoint_backend: str
) -> None:
    opts = options("gdn", checkpoint_backend)
    if wide_offsets:
        if checkpoint_backend == "warpdrive":
            from attn_gym.linear._delta_rule.warpdrive.bundles import gdn_warmup_forward

            monkeypatch.setattr(gdn_warmup_forward, "requires_int64_abi", lambda *args: True)
        else:
            recurrence = importlib.import_module("attn_gym.linear.kda.fwd.triton.chunk_delta_h")
            monkeypatch.setattr(recurrence, "requires_int64_offsets", lambda *args: True)
    tensors = make_gdn_test_inputs(128, key_dim=64, key_heads=1, value_heads=2)[:5]
    cache = torch.zeros(2, 2, 64, 64, device="cuda")
    saved = sentinel_state_checkpoints([64], (2, 64, 64), spare=0)
    paged_chunk_gdn(
        *tensors,
        cache,
        torch.tensor([1], device="cuda", dtype=torch.int32),
        state_checkpoints=saved,
        kernel_options=opts,
    )
    _, expected = chunk_gdn(
        *(tensor[:, :64] for tensor in tensors),
        cu_seqlens=cumulative_sequence_offsets([64]),
        output_final_state=True,
        kernel_options=opts,
    )
    torch.testing.assert_close(saved.state_pool[1], expected[0], rtol=0, atol=0)


@torch.no_grad()
def test_persistent_recurrence_writes_only_selected_checkpoints(
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
    positions = [length, length - 1, length * 2, 0] * (sequences // 4)
    saved = sentinel_state_checkpoints(positions, (heads, 128, 128), spare=0)
    paged(
        *tensors,
        cache,
        torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32),
        cu_seqlens=cumulative_sequence_offsets([length] * sequences),
        state_checkpoints=saved,
        kernel_options={"autotune": False},
    )
    expected = torch.full_like(saved.state_pool, CHECKPOINT_SENTINEL)
    expected[1::4] = cache[1::4]
    torch.testing.assert_close(saved.state_pool, expected, rtol=0, atol=0)
    assert triton_compiles["chunk_delta_h_persistent_kernel"] == 1
