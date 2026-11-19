"""Periodic FP32 snapshots of the paged GDN recurrent accumulator."""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym.linear import StateCheckpoints, recurrent_gdn
from attn_gym.linear.gdn.ops import recurrent_fwd_paged_op
from attn_gym.testing import (
    CHECKPOINT_SENTINEL,
    assert_pool_checkpoints_match_tape,
    cumulative_sequence_offsets,
    sentinel_state_checkpoints,
    strided_state_pool,
)
from attn_gym.testing.gdn import make_gdn_test_inputs


@pytest.fixture
def cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("the fused recurrent scan requires CUDA")


def snapshots(
    lengths: list[int],
    interval: int,
    heads: int = 2,
    value_dim: int = 17,
    key_dim: int = 32,
    *,
    spare: int = 2,
) -> StateCheckpoints:
    counts = [length // interval for length in lengths]
    return sentinel_state_checkpoints(counts, interval, (heads, value_dim, key_dim), spare=spare)


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
        torch.empty(2, device="cuda", dtype=torch.int32),
        64,
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
                state_checkpoints=saved._replace(states=saved.states.bfloat16()),
            )


def test_operand_autograd_rejected_without_device_execution() -> None:
    with FakeTensorMode():
        args, slots, saved = metadata_inputs()
        args[0].requires_grad_()
        with pytest.raises(RuntimeError, match="inference.only"):
            recurrent_gdn(*args, state_indices=slots, state_checkpoints=saved)


@pytest.mark.parametrize("invalid", ["dtype", "missing_offsets"])
def test_raw_metadata_validated_without_device_execution(invalid: str) -> None:
    # Exercise the real raw-op implementation: validation must precede kernel dispatch.
    from attn_gym.linear.gdn.impl.fused import _gdn_recurrent_fwd_paged_cuda

    with FakeTensorMode():
        args, slots, (states, offsets, interval, _) = metadata_inputs()
        if invalid == "dtype":
            states = states.bfloat16()
        else:
            offsets = None
        with pytest.raises((ValueError, TypeError), match="checkpoint"):
            _gdn_recurrent_fwd_paged_cuda(
                *args, slots, None, None, 32**-0.5, False, states, offsets, interval
            )


@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "key_dim,value_dim,dtype,interval",
    [(32, 17, torch.float16, 128), (128, 128, torch.bfloat16, 64)],
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
    interval: int,
) -> None:
    # Grouped heads, a strided pool, empty and incomplete sequences, null routes, and
    # fresh routes whose stale slots hold NaN.
    lengths = [0, interval - 1, 2 * interval, interval + 7, interval, interval, 0]
    routes = [5, 2, 4, 1, 0, -1, 6]
    initial_flags = [False, True, True, False, True, False, True]
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
    saved = snapshots(lengths, interval, heads, value_dim, key_dim)
    expected_states = saved.states.clone()
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
    origin = checkpoint_start = 0
    for sequence, (length, slot) in enumerate(zip(lengths, routes, strict=True)):
        if slot > 0:
            for checkpoint, stop in enumerate(range(interval, length + 1, interval)):
                prefix_offsets = cu_seqlens.clone()
                prefix_offsets[sequence + 1 :] = origin + stop
                prefix_cache = storage.to(dtype=oracle_dtype, copy=True).as_strided(
                    actual_cache.shape, actual_cache.stride(), actual_cache.storage_offset()
                )
                prefix_cache.copy_(initial_cache)
                recurrent_gdn(*tensors, prefix_cache, **{**kwargs, "cu_seqlens": prefix_offsets})
                expected_states[checkpoint_start + checkpoint].copy_(prefix_cache[slot])
        origin += length
        checkpoint_start += length // interval
    if not batch_invariant and state_dtype == torch.bfloat16:
        assert_bits(saved.states.bfloat16(), expected_states.bfloat16())
    else:
        assert_bits(saved.states, expected_states)


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
    saved = snapshots([128, 128], 64)
    options = {"batch_invariant": batch_invariant}
    recurrent_gdn(
        *tensors, cache, state_indices=slots, state_checkpoints=saved, kernel_options=options
    )
    for checkpoint, stop in enumerate((64, 128)):
        prefix_cache = initial.float()
        recurrent_gdn(
            *(tensor[:, :stop] for tensor in tensors),
            prefix_cache,
            state_indices=slots,
            kernel_options=options,
        )
        assert_bits(saved.states[checkpoint:4:2], prefix_cache[slots.long()])
    assert torch.any(saved.states[:4] != saved.states[:4].bfloat16().float())
    assert torch.all(saved.states[4:] == CHECKPOINT_SENTINEL)


@pytest.mark.parametrize("batch_invariant", [False, True])
@torch.no_grad()
def test_checkpoints_written_into_a_strided_bf16_pool_match_the_tape(
    cuda: None, batch_invariant: bool
) -> None:
    lengths = [129, 64, 200]
    tensors = make_gdn_test_inputs(sum(lengths), key_dim=32, value_dim=17)[:5]
    cu_seqlens = cumulative_sequence_offsets(lengths)
    slots = torch.tensor([1, -1, 2], device="cuda", dtype=torch.int32)
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

    # Rows: 2 for the first sequence plus 1 spare, 1 reserved by the null route, 3 for the last.
    tape = sentinel_state_checkpoints([3, 1, 3], 64, (2, 17, 32), spare=0)
    assert_pool_checkpoints_match_tape(run, cache, tape, [3, 0, 6, 4, 5, -1, 7])


@torch.no_grad()
@pytest.mark.parametrize("execution", ["fullgraph", "cuda_graph"])
def test_graph_replay_reads_updated_destinations_in_the_state_cache(
    cuda: None, execution: str
) -> None:
    tensors = make_gdn_test_inputs(256, key_dim=32, value_dim=17)[:5]
    initial = torch.randn(8, 2, 17, 32, device="cuda") * 0.01
    cache = initial.clone()
    slots = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    destinations = torch.tensor([3, 4, 5, 6], device="cuda", dtype=torch.int32)
    sequence_offsets = cumulative_sequence_offsets([128, 128])
    checkpoint_offsets = cumulative_sequence_offsets([2, 2])

    def run(cache: torch.Tensor, destinations: torch.Tensor) -> torch.Tensor:
        output, _ = recurrent_gdn(
            *tensors,
            cache,
            state_indices=slots,
            cu_seqlens=sequence_offsets,
            state_checkpoints=StateCheckpoints(cache, checkpoint_offsets, 64, destinations),
            kernel_options={"batch_invariant": True},
        )
        return output

    run(cache, destinations)
    if execution == "fullgraph":
        compiled = torch.compile(run, fullgraph=True)
        compiled(cache, destinations)
    else:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(cache, destinations)
    for selected in ([3, 0, 4, -1], [7, 6, 5, 4]):
        destinations.copy_(torch.tensor(selected, device="cuda", dtype=torch.int32))
        cache.copy_(initial)
        expected_cache = initial.clone()
        expected = run(expected_cache, destinations.clone())
        if execution == "fullgraph":
            output = compiled(cache, destinations)
        else:
            graph.replay()
        assert_bits(output, expected)
        assert_bits(cache, expected_cache)


@torch.no_grad()
def test_empty_sequences_and_zero_capacity_checkpoints(cuda: None) -> None:
    tensors = make_gdn_test_inputs(1, key_dim=32, value_dim=17)[:5]
    initial = torch.randn(5, 2, 17, 32, device="cuda")
    cache = initial.clone()
    recurrent_gdn(
        *tensors,
        cache,
        state_indices=torch.tensor([1, 2, 0, -1], device="cuda", dtype=torch.int32),
        has_initial_state=torch.tensor([False, True, False, True], device="cuda"),
        cu_seqlens=cumulative_sequence_offsets([0, 0, 0, 0]),
        state_checkpoints=snapshots([0, 0, 0, 0], 64, spare=0),
    )
    initial[1].zero_()
    assert_bits(cache, initial)


@pytest.mark.parametrize("into_state_cache", [False, True])
def test_checkpoint_operator_registration(cuda: None, into_state_cache: bool) -> None:
    tensors = make_gdn_test_inputs(128, key_dim=32, value_dim=17, gate_pattern="near_zero")[:5]
    cache = torch.randn(5, 2, 17, 32, device="cuda", dtype=torch.bfloat16) * 0.01
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    flags = torch.tensor([True, False], device="cuda")
    sequence_offsets = cumulative_sequence_offsets([64, 64])
    saved = snapshots([64, 64], 64, spare=0)
    if into_state_cache:
        # Destinations alias the advanced pool: slots 2 and 4 are not routed.
        destinations = torch.tensor([2, 4], device="cuda", dtype=torch.int32)
        saved = saved._replace(states=cache, destinations=destinations)
    torch.library.opcheck(
        recurrent_fwd_paged_op,
        (*tensors, cache, slots, flags, sequence_offsets, 32**-0.5, True, *saved),
    )


@torch.no_grad()
def test_cuda_graph_replay_reads_updated_routes_and_offsets(cuda: None) -> None:
    tensors = make_gdn_test_inputs(256, key_dim=32, value_dim=17)[:5]
    initial = torch.randn(5, 2, 17, 32, device="cuda") * 0.01
    cache = initial.clone()
    slots = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    flags = torch.tensor([True, False], device="cuda")
    sequence_offsets = cumulative_sequence_offsets([128, 128])
    saved = snapshots([128, 128], 128)

    def run(cache, states):
        return recurrent_gdn(
            *tensors,
            cache,
            state_indices=slots,
            has_initial_state=flags,
            cu_seqlens=sequence_offsets,
            state_checkpoints=StateCheckpoints(states, saved.cu_checkpoint_offsets, 128),
            kernel_options={"batch_invariant": True},
        )

    run(cache, saved.states)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = run(cache, saved.states)

    for lengths, routes, initial_flags in (
        ([64, 192], [2, 4], [False, True]),
        ([0, 256], [0, 1], [True, False]),
    ):
        sequence_offsets.copy_(cumulative_sequence_offsets(lengths))
        saved.cu_checkpoint_offsets.copy_(
            cumulative_sequence_offsets([length // 128 for length in lengths])
        )
        slots.copy_(torch.tensor(routes, device="cuda", dtype=torch.int32))
        flags.copy_(torch.tensor(initial_flags, device="cuda"))
        cache.copy_(initial)
        saved.states.fill_(CHECKPOINT_SENTINEL)
        expected_cache = initial.clone()
        expected_states = torch.full_like(saved.states, CHECKPOINT_SENTINEL)
        expected, _ = run(expected_cache, expected_states)
        graph.replay()
        assert_bits(output, expected)
        assert_bits(cache, expected_cache)
        assert_bits(saved.states, expected_states)


@torch.no_grad()
def test_fullgraph_dynamic_shapes_preserve_checkpoint_mutation(cuda: None) -> None:
    def run(q, k, v, gate, beta, cache, slots, sequence_offsets, states, checkpoint_offsets):
        return recurrent_gdn(
            q,
            k,
            v,
            gate,
            beta,
            cache,
            state_indices=slots,
            cu_seqlens=sequence_offsets,
            state_checkpoints=StateCheckpoints(states, checkpoint_offsets, 64),
            kernel_options={"batch_invariant": True},
        )

    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    for sequences, length in ((3, 65), (16, 129), (5, 193)):
        tensors = make_gdn_test_inputs(sequences * length, key_dim=32, value_dim=17)[:5]
        lengths = [length] * sequences
        slots = torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32)
        sequence_offsets = cumulative_sequence_offsets(lengths)
        saved = snapshots(lengths, 64)
        cache = torch.randn(sequences + 1, 2, 17, 32, device="cuda") * 0.01
        expected_cache = cache.clone()
        expected_states = saved.states.clone()
        offsets = saved.cu_checkpoint_offsets
        expected, _ = run(
            *tensors, expected_cache, slots, sequence_offsets, expected_states, offsets
        )
        output, final = compiled(*tensors, cache, slots, sequence_offsets, saved.states, offsets)
        assert final is None
        assert_bits(output, expected)
        assert_bits(cache, expected_cache)
        assert_bits(saved.states, expected_states)


@torch.no_grad()
def test_runtime_intervals_and_shapes_reuse_scan(cuda: None, triton_compiles) -> None:
    # Sizes cover divisible-by-16 and other sequence/token/capacity counts.
    for sequences, length, interval, spare in (
        (9, 65, 64, 0),
        (16, 1025, 1024, 16),
        (17, 193, 192, 3),
    ):
        tensors = make_gdn_test_inputs(sequences * length, key_dim=32, value_dim=17)[:5]
        lengths = [length] * sequences
        saved = snapshots(lengths, interval, spare=spare)
        recurrent_gdn(
            *tensors,
            torch.zeros(sequences + 1, 2, 17, 32, device="cuda"),
            state_indices=torch.arange(1, sequences + 1, device="cuda", dtype=torch.int32),
            cu_seqlens=cumulative_sequence_offsets(lengths),
            state_checkpoints=saved,
            kernel_options={"batch_invariant": True},
        )
        assert torch.isfinite(saved.states[:sequences]).all()
        assert torch.all(saved.states[:sequences] != CHECKPOINT_SENTINEL)
        assert torch.all(saved.states[sequences:] == CHECKPOINT_SENTINEL)
    assert triton_compiles["_recurrent_delta_rule_fwd_kernel"] == 1


@torch.no_grad()
def test_interval_larger_than_int32_does_not_wrap(cuda: None) -> None:
    saved = snapshots([64], 2**32 + 64)
    recurrent_gdn(
        *make_gdn_test_inputs(64, key_dim=32, value_dim=17)[:5],
        torch.zeros(2, 2, 17, 32, device="cuda"),
        state_indices=torch.tensor([1], device="cuda", dtype=torch.int32),
        state_checkpoints=saved,
    )
    assert torch.all(saved.states == CHECKPOINT_SENTINEL)
