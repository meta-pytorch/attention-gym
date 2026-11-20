"""Per-sequence checkpoints of the paged GDN recurrent accumulator."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym.linear import StateCheckpoints, recurrent_gdn
from attn_gym.linear.gdn.ops import recurrent_fwd_paged_op
from attn_gym.testing import (
    CHECKPOINT_SENTINEL,
    assert_checkpoints_into_cache_match,
    cumulative_sequence_offsets,
    sentinel_state_checkpoints,
    strided_state_pool,
)
from attn_gym.testing.gdn import make_gdn_test_inputs


@pytest.fixture
def cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("the fused recurrent scan requires CUDA")


def test_debug_rejects_unaligned_checkpoint_only_for_a_selected_slot(cuda: None) -> None:
    # A failing device assert invalidates its CUDA context. Exercise both cases in one child.
    code = """
import torch
from attn_gym.linear import StateCheckpoints, recurrent_gdn
from attn_gym.testing.gdn import make_gdn_test_inputs

inputs = make_gdn_test_inputs(128, key_dim=32, value_dim=17)[:5]
cache = torch.zeros(3, inputs[2].shape[2], 17, 32, device="cuda")
route = torch.tensor([1], dtype=torch.int32, device="cuda")
request = StateCheckpoints(cache, torch.full_like(route, 65), torch.full_like(route, -1))
with torch.inference_mode():
    for destination in (-1, 2):
        request.checkpoint_slots.fill_(destination)
        recurrent_gdn(*inputs, cache, state_indices=route, state_checkpoints=request,
                      kernel_options={"autotune": False})
        torch.cuda.synchronize()
        print(f"completed {destination}", flush=True)
"""
    run = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
        env={**os.environ, "TRITON_DEBUG": "1"},
    )
    assert "completed -1" in run.stdout, run.stderr
    assert run.returncode != 0 and "invalid state checkpoint position" in run.stderr, run.stderr


def assert_bits(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def metadata_inputs() -> tuple[tuple[torch.Tensor, ...], torch.Tensor, StateCheckpoints]:
    q, k = (torch.empty(1, 128, 1, 32, device="cuda") for _ in range(2))
    v = torch.empty(1, 128, 2, 17, device="cuda")
    gate, beta = (torch.empty(v.shape[:-1], device="cuda") for _ in range(2))
    cache = torch.empty(3, 2, 17, 32, device="cuda")
    slots = torch.empty(1, device="cuda", dtype=torch.int32)
    checkpoints = StateCheckpoints(
        torch.empty(2, 2, 17, 32, device="cuda"),
        torch.empty_like(slots),
        torch.empty_like(slots),
    )
    return (q, k, v, gate, beta, cache), slots, checkpoints


@pytest.mark.parametrize(
    "impl,paged,match",
    [
        ("fused", False, "state_checkpoints.*state_indices"),
        ("reference", False, "state_checkpoints.*state_indices"),
        ("reference", True, "impl='fused'"),
    ],
)
def test_unsupported_checkpoint_modes_rejected(impl: str, paged: bool, match: str) -> None:
    with FakeTensorMode():
        args, slots, checkpoints = metadata_inputs()
        indices = {"state_indices": slots} if paged else {}
        with pytest.raises(ValueError, match=match):
            recurrent_gdn(
                *(args if paged else args[:5]),
                **indices,
                state_checkpoints=checkpoints,
                impl=impl,
            )


def test_public_metadata_validated_without_device_execution() -> None:
    with FakeTensorMode():
        args, slots, saved = metadata_inputs()
        with pytest.raises(TypeError, match="state_checkpoints"):
            recurrent_gdn(
                *args,
                state_indices=slots,
                state_checkpoints=saved._replace(state_pool=saved.state_pool.half()),
            )


def test_operand_autograd_rejected_without_device_execution() -> None:
    with FakeTensorMode():
        args, slots, saved = metadata_inputs()
        args[0].requires_grad_()
        with pytest.raises(RuntimeError, match="inference.only"):
            recurrent_gdn(*args, state_indices=slots, state_checkpoints=saved)


@pytest.mark.parametrize("invalid", ["dtype", "missing_checkpoint_slots"])
def test_raw_metadata_validated_without_device_execution(invalid: str) -> None:
    # Exercise the real raw-op implementation: validation must precede kernel dispatch.
    from attn_gym.linear.gdn.impl.fused import _gdn_recurrent_fwd_paged_cuda

    with FakeTensorMode():
        args, slots, (pool, positions, destinations) = metadata_inputs()
        if invalid == "dtype":
            pool = pool.half()
        else:
            destinations = None
        with pytest.raises((ValueError, TypeError), match="checkpoint"):
            _gdn_recurrent_fwd_paged_cuda(
                *args, slots, None, None, 32**-0.5, False, pool, positions, destinations
            )


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "key_dim,value_dim,dtype",
    [(32, 17, torch.float16), (128, 128, torch.bfloat16)],
    ids=["odd-dims", "d128"],
)
@torch.no_grad()
def test_checkpoints_match_prefix_stops_and_preserve_results(
    cuda: None,
    batch_invariant: bool,
    state_dtype: torch.dtype,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
) -> None:
    # Grouped heads, a strided pool, empty and incomplete sequences, null routes, and
    # fresh routes whose stale slots hold NaN.
    lengths = [0, 63, 200, 135, 64, 64, 129, 128, 65]
    positions = [64, 64, 128, 64, 64, 64, 65, 192, 0]
    routes = [5, 2, 4, 1, 0, -1, 6, 7, 3]
    initial_flags = [False, True, True, False, True, False, True, True, False]
    heads = 3
    tensors = make_gdn_test_inputs(
        sum(lengths),
        key_heads=1,
        value_heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        dtype=dtype,
        gate_pattern="near_zero",
    )[:5]
    cu_seqlens = cumulative_sequence_offsets(lengths)
    slots = torch.tensor(routes, device="cuda", dtype=torch.int32)
    has_initial = torch.tensor(initial_flags, device="cuda")
    storage, actual_cache = strided_state_pool(8, heads, key_dim, value_dim, dtype=state_dtype)
    actual_cache[1].fill_(float("nan"))
    actual_cache[5].fill_(float("nan"))
    initial_cache = actual_cache.clone()
    # Ordinary scans need not be bitwise invariant to a different storage layout.
    expected_cache = storage.clone().as_strided(
        actual_cache.shape, actual_cache.stride(), actual_cache.storage_offset()
    )
    saved = sentinel_state_checkpoints(positions, (heads, value_dim, key_dim))
    expected_states = saved.state_pool.clone()
    kwargs = {
        "cu_seqlens": cu_seqlens,
        "state_indices": slots,
        "has_initial_state": has_initial,
        "kernel_options": {"batch_invariant": batch_invariant},
    }
    expected_output, expected_final = recurrent_gdn(*tensors, expected_cache, **kwargs)
    output, final = recurrent_gdn(*tensors, actual_cache, state_checkpoints=saved, **kwargs)
    assert final is expected_final is None
    assert_bits(output, expected_output)
    assert_bits(actual_cache, expected_cache)

    # Keep physical T, sequence count and pool strides fixed. Batch-invariant scans
    # permit an FP32 oracle pool; ordinary scans can change rounding with the pool dtype.
    oracle_dtype = torch.float32 if batch_invariant else state_dtype
    # Only sequences 2 and 3 request an exact, reachable boundary on a live route.
    for sequence in (2, 3):
        prefix_offsets = cu_seqlens.clone()
        prefix_offsets[sequence + 1 :] = sum(lengths[:sequence]) + positions[sequence]
        prefix_cache = storage.to(dtype=oracle_dtype, copy=True).as_strided(
            actual_cache.shape, actual_cache.stride(), actual_cache.storage_offset()
        )
        prefix_cache.copy_(initial_cache)
        recurrent_gdn(*tensors, prefix_cache, **{**kwargs, "cu_seqlens": prefix_offsets})
        expected_states[sequence + 1].copy_(prefix_cache[routes[sequence]])
    if not batch_invariant and state_dtype == torch.bfloat16:
        assert_bits(saved.state_pool.bfloat16(), expected_states.bfloat16())
    else:
        assert_bits(saved.state_pool, expected_states)


@pytest.mark.parametrize("batch_invariant", [False, True])
@torch.no_grad()
def test_dense_checkpoints_capture_fp32_before_pool_rounding(
    cuda: None, batch_invariant: bool
) -> None:
    tensors = make_gdn_test_inputs(
        128, batch=2, key_dim=32, value_dim=17, dtype=torch.float16, gate_pattern="near_zero"
    )[:5]
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    cache = torch.randn(5, 2, 17, 32, device="cuda", dtype=torch.bfloat16) * 0.01
    initial = cache.clone()
    saved = sentinel_state_checkpoints([64, 128], (2, 17, 32))
    options = {"batch_invariant": batch_invariant}
    recurrent_gdn(
        *tensors, cache, state_indices=slots, state_checkpoints=saved, kernel_options=options
    )
    expected = torch.full_like(saved.state_pool, CHECKPOINT_SENTINEL)
    for sequence, stop in enumerate((64, 128)):
        prefix_cache = initial.float()
        recurrent_gdn(
            *(tensor[:, :stop] for tensor in tensors),
            prefix_cache,
            state_indices=slots,
            kernel_options=options,
        )
        expected[sequence + 1].copy_(prefix_cache[slots[sequence]])
    assert_bits(saved.state_pool, expected)
    states = saved.state_pool[1:3]
    assert torch.any(states != states.bfloat16().float())


@pytest.mark.parametrize("batch_invariant", [False, True])
@torch.no_grad()
def test_checkpoints_written_into_a_strided_bf16_pool_match_a_separate_pool(
    cuda: None, batch_invariant: bool
) -> None:
    lengths = [129, 64, 200, 64]
    tensors = make_gdn_test_inputs(sum(lengths), key_dim=32, value_dim=17)[:5]
    cu_seqlens = cumulative_sequence_offsets(lengths)
    slots = torch.tensor([1, -1, 2, 4], device="cuda", dtype=torch.int32)
    _storage, cache = strided_state_pool(8, 2, 32, 17, dtype=torch.bfloat16)

    def run(cache: torch.Tensor, saved: StateCheckpoints) -> torch.Tensor:
        output, _ = recurrent_gdn(
            *tensors,
            cache,
            state_indices=slots,
            cu_seqlens=cu_seqlens,
            state_checkpoints=saved,
            kernel_options={"batch_invariant": batch_invariant},
        )
        return output

    reference = sentinel_state_checkpoints([128, 64, 192, 64], (2, 17, 32))
    assert_checkpoints_into_cache_match(run, cache, reference, [6, 7, 0, -1], [0, 2, 3])


@torch.no_grad()
def test_empty_sequences_skip_checkpoints(cuda: None) -> None:
    tensors = make_gdn_test_inputs(1, key_dim=32, value_dim=17)[:5]
    initial = torch.randn(5, 2, 17, 32, device="cuda")
    cache = initial.clone()
    saved = sentinel_state_checkpoints([0, 64, 0, 64], (2, 17, 32))
    recurrent_gdn(
        *tensors,
        cache,
        state_indices=torch.tensor([1, 2, 0, -1], device="cuda", dtype=torch.int32),
        has_initial_state=torch.tensor([False, True, False, True], device="cuda"),
        cu_seqlens=cumulative_sequence_offsets([0, 0, 0, 0]),
        state_checkpoints=saved,
    )
    initial[1].zero_()
    assert_bits(cache, initial)
    assert torch.all(saved.state_pool == CHECKPOINT_SENTINEL)


@pytest.mark.parametrize("into_state_cache", [False, True])
def test_checkpoint_operator_registration(cuda: None, into_state_cache: bool) -> None:
    tensors = make_gdn_test_inputs(128, key_dim=32, value_dim=17, gate_pattern="near_zero")[:5]
    cache = torch.randn(5, 2, 17, 32, device="cuda", dtype=torch.bfloat16) * 0.01
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    flags = torch.tensor([True, False], device="cuda")
    sequence_offsets = cumulative_sequence_offsets([64, 64])
    saved = sentinel_state_checkpoints([64, 64], (2, 17, 32))
    if into_state_cache:
        destinations = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
        saved = saved._replace(state_pool=cache, checkpoint_slots=destinations)
    torch.library.opcheck(
        recurrent_fwd_paged_op,
        (*tensors, cache, slots, flags, sequence_offsets, 32**-0.5, True, *saved),
    )


@pytest.mark.parametrize("execution", ["fullgraph", "cuda_graph"])
@torch.no_grad()
def test_graph_replay_reads_updated_routes_positions_and_slots(cuda: None, execution: str) -> None:
    tensors = make_gdn_test_inputs(256, key_dim=32, value_dim=17)[:5]
    initial = torch.randn(8, 2, 17, 32, device="cuda") * 0.01
    cache = initial.clone()
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    flags = torch.tensor([True, False], device="cuda")
    sequence_offsets = cumulative_sequence_offsets([128, 128])
    saved = StateCheckpoints(
        cache,
        torch.tensor([64, 128], device="cuda", dtype=torch.int32),
        torch.tensor([5, 6], device="cuda", dtype=torch.int32),
    )

    def run(cache: torch.Tensor) -> torch.Tensor:
        output, _ = recurrent_gdn(
            *tensors,
            cache,
            state_indices=slots,
            has_initial_state=flags,
            cu_seqlens=sequence_offsets,
            state_checkpoints=saved._replace(state_pool=cache),
            kernel_options={"batch_invariant": True},
        )
        return output

    run(cache)
    if execution == "fullgraph":
        compiled = torch.compile(run, fullgraph=True)
        compiled(cache)
    else:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(cache)
    for lengths, routes, initial_flags, positions, destinations in (
        ([64, 192], [2, 3], [False, True], [64, 192], [0, 7]),
        ([0, 256], [0, 1], [True, False], [0, 64], [-1, 4]),
    ):
        sequence_offsets.copy_(cumulative_sequence_offsets(lengths))
        slots.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
        flags.copy_(torch.tensor(initial_flags, device="cuda"))
        saved.checkpoint_positions.copy_(torch.tensor(positions, device="cuda", dtype=torch.int32))
        saved.checkpoint_slots.copy_(torch.tensor(destinations, device="cuda", dtype=torch.int32))
        cache.copy_(initial)
        expected_cache = initial.clone()
        expected = run(expected_cache)
        if execution == "fullgraph":
            output = compiled(cache)
        else:
            graph.replay()
        assert_bits(output, expected)
        assert_bits(cache, expected_cache)


@torch.no_grad()
def test_fullgraph_dynamic_shapes_preserve_checkpoint_mutation(cuda: None) -> None:
    def run(q, k, v, gate, beta, cache, slots, sequence_offsets, pool, positions, destinations):
        return recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            cache,
            state_indices=slots,
            cu_seqlens=sequence_offsets,
            state_checkpoints=StateCheckpoints(pool, positions, destinations),
            kernel_options={"batch_invariant": True},
        )

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for sequences, length in ((3, 65), (16, 129), (5, 193)):
        tensors = make_gdn_test_inputs(sequences * length, key_dim=32, value_dim=17)[:5]
        lengths = [length] * sequences
        slots = torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32)
        sequence_offsets = cumulative_sequence_offsets(lengths)
        saved = sentinel_state_checkpoints([length - 1] * sequences, (2, 17, 32))
        cache = torch.randn(sequences + 1, 2, 17, 32, device="cuda") * 0.01
        expected_cache = cache.clone()
        expected_states = saved.state_pool.clone()
        expected, _ = run(
            *tensors,
            expected_cache,
            slots,
            sequence_offsets,
            *saved._replace(state_pool=expected_states),
        )
        output, final = compiled(*tensors, cache, slots, sequence_offsets, *saved)
        assert final is None
        assert_bits(output, expected)
        assert_bits(cache, expected_cache)
        assert_bits(saved.state_pool, expected_states)


@torch.no_grad()
def test_runtime_positions_and_shapes_reuse_scan(cuda: None, triton_compiles) -> None:
    # Sizes cover divisible-by-16 and other sequence/token/capacity counts.
    for sequences, length, position, spare in (
        (9, 65, 64, 0),
        (16, 1025, 1024, 16),
        (17, 193, 192, 3),
    ):
        tensors = make_gdn_test_inputs(sequences * length, key_dim=32, value_dim=17)[:5]
        lengths = [length] * sequences
        saved = sentinel_state_checkpoints([position] * sequences, (2, 17, 32), spare=spare)
        recurrent_gdn(
            *tensors,
            torch.zeros(sequences + 1, 2, 17, 32, device="cuda"),
            state_indices=torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32),
            cu_seqlens=cumulative_sequence_offsets(lengths),
            state_checkpoints=saved,
            kernel_options={"batch_invariant": True},
        )
        produced = saved.state_pool[1 : 1 + sequences]
        assert torch.isfinite(produced).all()
        assert torch.all(produced != CHECKPOINT_SENTINEL)
        assert torch.all(saved.state_pool[0] == CHECKPOINT_SENTINEL)
        assert torch.all(saved.state_pool[1 + sequences :] == CHECKPOINT_SENTINEL)
    assert triton_compiles["_recurrent_delta_rule_fwd_kernel"] == 1
