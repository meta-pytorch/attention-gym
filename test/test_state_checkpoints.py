"""Caller-owned checkpoints for inference-only paged delta-rule chunks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym.linear import (
    ReplayState,
    StateCheckpoints,
    chunk_gdn,
    chunk_kda,
    paged_chunk_gdn,
    paged_chunk_kda,
)
from attn_gym.linear._delta_rule.chunk_ops import _plain_gate_scan_op
from attn_gym.linear._delta_rule.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.testing.gdn import make_gdn_test_inputs
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

Rule = Literal["gdn", "kda"]
SENTINEL = 12345.0


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
    states = torch.full((sum(counts) + spare, 2, 128, 128), SENTINEL, device="cuda")
    return StateCheckpoints(states, cumulative_sequence_offsets(counts), interval)


def checkpoint_metadata_inputs(
    rule: Rule,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, StateCheckpoints]:
    q, k, v = (torch.empty(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    gate = torch.empty(q.shape if rule == "kda" else q.shape[:-1], device="cuda")
    beta = torch.empty(q.shape[:-1], device="cuda")
    cache = torch.empty(5, 2, 128, 128, device="cuda")
    slots = torch.empty(2, device="cuda", dtype=torch.int32)
    cu_seqlens = torch.empty(3, device="cuda", dtype=torch.int32)
    saved = StateCheckpoints(
        torch.empty(4, 2, 128, 128, device="cuda"),
        torch.empty(3, device="cuda", dtype=torch.int32),
        64,
    )
    return (q, k, v, gate, beta, cache, slots), cu_seqlens, saved


@pytest.mark.parametrize("interval", [64, 128])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_checkpoints_match_truncated_chunks_and_preserve_paged_results(
    rule: Rule, cuda: None, interval: int, state_dtype: torch.dtype
) -> None:
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
    saved = checkpoints(lengths, interval)
    expected_states = saved.states.clone()

    expected_output = paged(
        *tensors,
        expected_cache,
        slots,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial,
        kernel_options=options(rule),
    )
    actual_output = paged(
        *tensors,
        actual_cache,
        slots,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial,
        state_checkpoints=saved,
        kernel_options=options(rule),
    )
    assert isinstance(actual_output, torch.Tensor)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)

    # Compare each post-boundary state to the same chunk arithmetic stopped at that boundary.
    token_start = checkpoint_start = 0
    for length, slot, has_state in zip(lengths, routes, initial_flags, strict=True):
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
                assert final_state is not None
                expected_states[checkpoint_start + index].copy_(final_state[0])
        token_start += length
        checkpoint_start += length // interval

    torch.testing.assert_close(saved.states, expected_states, rtol=0, atol=0)


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
    assert expected is not None
    torch.testing.assert_close(saved.states[:2], expected, rtol=0, atol=0)
    torch.testing.assert_close(saved.states[:2], state_cache[slots.long()], rtol=0, atol=0)
    assert torch.all(saved.states[2:] == SENTINEL)


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

    def run(cache, states, sequence_offsets, checkpoint_offsets, routes, initial_flags):
        return paged(
            *tensors,
            cache,
            routes,
            cu_seqlens=sequence_offsets,
            has_initial_state=initial_flags,
            state_checkpoints=StateCheckpoints(states, checkpoint_offsets, 128),
            kernel_options=options(rule),
        )

    args = (state_cache, saved.states, cu_seqlens, saved.cu_checkpoint_offsets, slots, has_initial)
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
        saved.states.fill_(SENTINEL)
        expected_cache = initial_cache.clone()
        expected_states = torch.full_like(saved.states, SENTINEL)
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
        torch.testing.assert_close(saved.states, expected_states, rtol=0, atol=0)


def checkpoint_operator_inputs(
    rule: Rule, permuted_value: bool = False
) -> tuple[torch._ops.OpOverload, tuple]:
    q, k, v, gate, beta = inputs(rule, 128)
    if permuted_value:
        v = v.movedim(1, -1).contiguous().movedim(-1, 1)
        assert not v.is_contiguous()
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
    saved = checkpoints([64, 64], 64)
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
    return op, (*args, saved.states, saved.cu_checkpoint_offsets, saved.every_n_tokens)


@pytest.mark.parametrize("permuted_value", [False, True])
def test_checkpoint_paged_operator_registration(
    rule: Rule, cuda: None, permuted_value: bool
) -> None:
    op, args = checkpoint_operator_inputs(rule, permuted_value)
    torch.library.opcheck(op, args)


@pytest.mark.parametrize("invalid", ["dtype", "stride", "shape", "offsets", "interval"])
def test_raw_checkpoint_operator_rejects_invalid_metadata(
    rule: Rule, cuda: None, invalid: str
) -> None:
    op, args = checkpoint_operator_inputs(rule)
    states, offsets, interval = args[-3:]
    match invalid:
        case "dtype":
            states = states.bfloat16()
        case "stride":
            states = states.transpose(-1, -2)
        case "shape":
            states = torch.empty(4, 2, 128, 256, device="cuda")
        case "offsets":
            offsets = None
        case "interval":
            interval = 65
    with pytest.raises((TypeError, ValueError), match="checkpoint"):
        op(*args[:-3], states, offsets, interval)


@pytest.mark.parametrize(
    "invalid",
    [
        "zero_interval",
        "negative_interval",
        "unaligned_interval",
        "bool_interval",
        "float_interval",
        "state_dtype",
        "state_shape",
        "state_stride",
        "state_device",
        "offset_dtype",
        "offset_shape",
        "offset_stride",
        "offset_device",
    ],
)
def test_checkpoint_metadata_validation_without_device_execution(rule: Rule, invalid: str) -> None:
    paged, _ = operations(rule)
    with FakeTensorMode():
        args, cu_seqlens, saved = checkpoint_metadata_inputs(rule)
        states, offsets, interval = saved
        match invalid:
            case "zero_interval":
                interval = 0
            case "negative_interval":
                interval = -64
            case "unaligned_interval":
                interval = 65
            case "bool_interval":
                interval = True
            case "float_interval":
                interval = 64.0
            case "state_dtype":
                states = states.bfloat16()
            case "state_shape":
                states = states[:, :1]
            case "state_stride":
                states = states.transpose(-1, -2)
            case "state_device":
                states = states.cpu()
            case "offset_dtype":
                offsets = offsets.long()
            case "offset_shape":
                offsets = offsets[:2]
            case "offset_stride":
                offsets = torch.empty(6, device="cuda", dtype=torch.int32)[::2]
            case "offset_device":
                offsets = offsets.cpu()
        with pytest.raises((ValueError, TypeError), match="checkpoint|every_n_tokens"):
            paged(
                *args,
                cu_seqlens=cu_seqlens,
                state_checkpoints=StateCheckpoints(states, offsets, interval),
                kernel_options=options(rule),
            )


@pytest.mark.parametrize("grad_input", ["q", "states"])
def test_checkpoint_calls_reject_autograd(rule: Rule, cuda: None, grad_input: str) -> None:
    paged, _ = operations(rule)
    tensors = inputs(rule, 64)
    state_cache = torch.zeros(3, 2, 128, 128, device="cuda")
    slots = torch.tensor([1], device="cuda", dtype=torch.int32)
    saved = checkpoints([64], 64)
    (tensors[0] if grad_input == "q" else saved.states).requires_grad_()
    with pytest.raises(RuntimeError, match="inference.only"):
        paged(
            *tensors,
            state_cache,
            slots,
            state_checkpoints=saved,
            kernel_options=options(rule),
        )


def test_cudnn_rejects_checkpoints_before_backend_dispatch(rule: Rule) -> None:
    paged, _ = operations(rule)
    with FakeTensorMode():
        args, cu_seqlens, saved = checkpoint_metadata_inputs(rule)
        with pytest.raises(ValueError, match="checkpoint"):
            paged(
                *args,
                cu_seqlens=cu_seqlens,
                state_checkpoints=saved,
                kernel_options={"backend": "cudnn"},
            )


def test_kda_replay_rejects_checkpoints() -> None:
    with FakeTensorMode():
        args, cu_seqlens, saved = checkpoint_metadata_inputs("kda")
        replay = ReplayState(
            *(
                torch.empty((5, *shape), dtype=dtype, device="cuda")
                for shape, dtype in ReplayState.slot_layout(2, 128)
            )
        )
        with pytest.raises(ValueError, match="checkpoint"):
            paged_chunk_kda(
                *args,
                cu_seqlens=cu_seqlens,
                state_checkpoints=saved,
                replay_state=replay,
                kernel_options=options("kda"),
            )


@torch.no_grad()
def test_checkpoint_intervals_and_shapes_reuse_recurrence(rule: Rule, cuda: None, triton_compiles):
    paged, _ = operations(rule)
    # Interval in chunks crosses Triton's ==1 / divisible-by-16 / other classes.
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
        assert torch.isfinite(saved.states[:active]).all()
        assert torch.all(saved.states[:active] != SENTINEL)
        assert torch.all(saved.states[active:] == SENTINEL)
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
    saved = StateCheckpoints(
        torch.empty(2, 2, 64, 64, device="cuda"), cumulative_sequence_offsets([2]), 64
    )
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
        torch.testing.assert_close(saved.states[row], expected[0], rtol=0, atol=0)


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
    saved = StateCheckpoints(
        torch.full((sequences, heads, 128, 128), SENTINEL, device="cuda"),
        cumulative_sequence_offsets([1] * sequences),
        length,
    )
    paged(
        *tensors,
        cache,
        torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32),
        cu_seqlens=cumulative_sequence_offsets([length] * sequences),
        state_checkpoints=saved,
        kernel_options={"autotune": False},
    )
    torch.testing.assert_close(saved.states, cache[1:], rtol=0, atol=0)
    assert triton_compiles["chunk_delta_h_persistent_kernel"] == 1
