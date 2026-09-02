"""End-to-end behavior and gradient tests for public ragged CuTe KDA."""

from __future__ import annotations

import inspect

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear import chunk_kda
from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.linear.kda.naive import naive_chunk_kda
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the fused KDA core requires CUDA capability 8.0 or newer",
)


def _run_gradients(
    inputs,
    initial_state,
    offsets,
    output_grad,
    state_grad,
    operation=chunk_kda,
    *,
    output_final_state=True,
):
    output, final_state = operation(
        *inputs,
        initial_state,
        cu_seqlens=offsets,
        output_final_state=output_final_state,
    )
    losses = []
    if output_grad is not None:
        losses.append((output.float() * output_grad).sum())
    if state_grad is not None:
        assert final_state is not None
        losses.append((final_state * state_grad).sum())
    targets = (*inputs, *((initial_state,) if initial_state is not None else ()))
    return output, final_state, torch.autograd.grad(sum(losses), targets)


def _run_naive_gradients(
    inputs,
    initial_state,
    lengths,
    output_grad,
    state_grad,
    dtype,
    *,
    output_final_state,
):
    """Run sequence-local naive KDA in one precision for a numerical measuring stick."""
    reference_inputs = tuple(value.detach().to(dtype).requires_grad_() for value in inputs)
    reference_state = (
        None if initial_state is None else initial_state.detach().to(dtype).requires_grad_()
    )
    outputs = []
    states = []
    token_start = 0
    for sequence, length in enumerate(lengths):
        if length == 0:
            if output_final_state:
                states.append(reference_state[sequence])
            continue
        token_slice = slice(token_start, token_start + length)
        sequence_state = (
            None if reference_state is None else reference_state[sequence : sequence + 1]
        )
        output, final_state = naive_chunk_kda(
            *(value[:, token_slice] for value in reference_inputs[:3]),
            reference_inputs[3][:, token_slice] * LOG2_E,
            reference_inputs[4][:, token_slice],
            initial_state=sequence_state,
            output_final_state=output_final_state,
            chunk_size=64,
        )
        outputs.append(output)
        if output_final_state:
            states.append(final_state[0])
        token_start += length
    reference_output = torch.cat(outputs, dim=1)
    reference_final_state = torch.stack(states) if output_final_state else None
    losses = []
    if output_grad is not None:
        losses.append((reference_output * output_grad.to(dtype)).sum())
    if state_grad is not None:
        losses.append((reference_final_state * state_grad.to(dtype)).sum())
    targets = (*reference_inputs, *((reference_state,) if reference_state is not None else ()))
    gradients = torch.autograd.grad(sum(losses), targets, allow_unused=True)
    gradients = tuple(
        torch.zeros_like(target) if gradient is None else gradient
        for target, gradient in zip(targets, gradients, strict=True)
    )
    return reference_output, reference_final_state, gradients


def _assert_run_close(actual, expected, active_tokens=None):
    """Compare runs exactly on outputs/state; token tensors only over the active prefix."""
    prefix = slice(None) if active_tokens is None else slice(None, active_tokens)
    actual_output, actual_state, actual_gradients = actual
    expected_output, expected_state, expected_gradients = expected
    torch.testing.assert_close(actual_output[:, prefix], expected_output, rtol=0, atol=0)
    if expected_state is None:
        assert actual_state is None
    else:
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        if active_tokens is not None and actual_gradient.shape != expected_gradient.shape:
            actual_gradient = actual_gradient[:, prefix]
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=3e-2)


def test_public_ragged_preserves_state_for_all_empty_sequences():
    """Ignore poisoned physical capacity when every logical sequence is empty."""
    inputs = tuple(
        torch.full_like(tensor, float("nan")) for tensor in make_kda_test_inputs(64, seed=29)
    )
    initial_state = torch.randn(2, 1, 128, 128, device="cuda") / 8

    _output, final_state = chunk_kda(
        *inputs,
        initial_state,
        cu_seqlens=cumulative_sequence_offsets([0, 0]),
        output_final_state=True,
    )

    torch.testing.assert_close(final_state, initial_state, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["outer_strided", "misaligned"])
def test_public_packed_backward_normalizes_cotangents(layout):
    lengths = [256, 256]
    inputs = make_kda_test_inputs(sum(lengths), heads=16, seed=37, requires_grad=True)
    actual_inputs = clone_kda_inputs(inputs)
    initial_state = torch.randn(2, 16, 128, 128, device="cuda", requires_grad=True)
    actual_state = initial_state.detach().clone().requires_grad_()
    offsets = cumulative_sequence_offsets(lengths)
    expected_output, expected_final_state = chunk_kda(
        *inputs,
        initial_state,
        cu_seqlens=offsets,
        output_final_state=True,
        autotune=False,
    )
    actual_output, actual_final_state = chunk_kda(
        *actual_inputs,
        actual_state,
        cu_seqlens=offsets,
        output_final_state=True,
        autotune=False,
    )
    assert expected_final_state is not None and actual_final_state is not None
    output_grad = torch.randn_like(expected_output)
    state_grad = torch.randn_like(expected_final_state)
    match layout:
        case "outer_strided":
            output_storage = torch.empty(
                expected_output.shape[0],
                2 * expected_output.shape[1],
                *expected_output.shape[2:],
                device="cuda",
                dtype=expected_output.dtype,
            )
            actual_output_grad = output_storage[:, ::2]
            state_storage = torch.empty(
                *state_grad.shape[:-1],
                2,
                state_grad.shape[-1],
                device="cuda",
                dtype=state_grad.dtype,
            )
            actual_state_grad = state_storage[..., 0, :]
            assert not actual_output_grad.is_contiguous()
            assert not actual_state_grad.is_contiguous()
        case "misaligned":
            output_storage = torch.empty(
                output_grad.numel() + 1,
                device="cuda",
                dtype=output_grad.dtype,
            )
            actual_output_grad = output_storage[1:].view_as(output_grad)
            state_storage = torch.empty(
                state_grad.numel() + 1,
                device="cuda",
                dtype=state_grad.dtype,
            )
            actual_state_grad = state_storage[1:].view_as(state_grad)
            assert actual_output_grad.data_ptr() % 16
            assert actual_state_grad.data_ptr() % 16
    actual_output_grad.copy_(output_grad)
    actual_state_grad.copy_(state_grad)

    expected = torch.autograd.grad(
        (expected_output, expected_final_state),
        (*inputs, initial_state),
        (output_grad, state_grad),
    )
    actual = torch.autograd.grad(
        (actual_output, actual_final_state),
        (*actual_inputs, actual_state),
        (actual_output_grad, actual_state_grad),
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_public_ragged_backward_matches_independent_sequences():
    lengths = [65, 0, 63]
    tokens = sum(lengths)
    packed_inputs = make_kda_test_inputs(tokens, requires_grad=True)
    initial_state = (torch.randn(3, 1, 128, 128, device="cuda") / 8).requires_grad_()
    output_grad = torch.randn(1, tokens, 1, 128, device="cuda")
    state_grad = torch.randn_like(initial_state)

    output, final_state, gradients = _run_gradients(
        packed_inputs,
        initial_state,
        cumulative_sequence_offsets(lengths),
        output_grad,
        state_grad,
    )

    expected_output = torch.empty_like(output)
    expected_state = []
    expected_gradients = [torch.empty_like(value) for value in packed_inputs]
    expected_state_gradient = torch.empty_like(initial_state)
    begin = 0
    for sequence, length in enumerate(lengths):
        if length == 0:
            expected_state.append(initial_state[sequence].detach())
            expected_state_gradient[sequence] = state_grad[sequence]
            continue
        end = begin + length
        sequence_inputs = tuple(
            value[:, begin:end].detach().clone().requires_grad_() for value in packed_inputs
        )
        sequence_state = initial_state[sequence : sequence + 1].detach().clone().requires_grad_()
        sequence_output, sequence_final, sequence_gradients = _run_gradients(
            sequence_inputs,
            sequence_state,
            cumulative_sequence_offsets([length]),
            output_grad[:, begin:end],
            state_grad[sequence : sequence + 1],
        )
        expected_output[:, begin:end] = sequence_output
        expected_state.append(sequence_final[0])
        for expected, gradient in zip(expected_gradients, sequence_gradients[:-1]):
            expected[:, begin:end] = gradient
        expected_state_gradient[sequence] = sequence_gradients[-1][0]
        begin = end

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(final_state, torch.stack(expected_state), rtol=0, atol=0)
    for actual, expected in zip(gradients[:-1], expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(gradients[-1], expected_state_gradient, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    ("has_initial_state", "output_final_state", "cotangents"),
    [
        (False, False, "output"),
        (True, True, "output"),
        (True, True, "state"),
        (True, True, "both"),
    ],
)
def test_public_ragged_backward_matches_naive_reference(
    has_initial_state,
    output_final_state,
    cotangents,
):
    lengths = [65, 0, 63]
    tokens = sum(lengths)
    actual_inputs = make_kda_test_inputs(tokens, requires_grad=True)
    actual_state = (
        (torch.randn(3, 1, 128, 128, device="cuda") / 8).requires_grad_()
        if has_initial_state
        else None
    )
    output_grad = None if cotangents == "state" else torch.randn(1, tokens, 1, 128, device="cuda")
    state_grad = torch.randn_like(actual_state) if cotangents in ("state", "both") else None

    actual = _run_gradients(
        actual_inputs,
        actual_state,
        cumulative_sequence_offsets(lengths),
        output_grad,
        state_grad,
        output_final_state=output_final_state,
    )
    low_precision = _run_naive_gradients(
        actual_inputs,
        actual_state,
        lengths,
        output_grad,
        state_grad,
        torch.float32,
        output_final_state=output_final_state,
    )
    high_precision = _run_naive_gradients(
        actual_inputs,
        actual_state,
        lengths,
        output_grad,
        state_grad,
        torch.float64,
        output_final_state=output_final_state,
    )

    names = ["output"]
    actual_values = [actual[0]]
    low_precision_values = [low_precision[0]]
    high_precision_values = [high_precision[0]]
    if output_final_state:
        names.append("final_state")
        actual_values.append(actual[1])
        low_precision_values.append(low_precision[1])
        high_precision_values.append(high_precision[1])
    names.extend(("dq", "dk", "dv", "dg", "db"))
    if has_initial_state:
        names.append("d_initial_state")
    actual_values.extend(actual[2])
    low_precision_values.extend(low_precision[2])
    high_precision_values.extend(high_precision[2])
    for name, value, high_precision_value, low_precision_value in zip(
        names,
        actual_values,
        high_precision_values,
        low_precision_values,
        strict=True,
    ):
        assert_matches_low_precision_reference(
            value,
            high_precision_value,
            low_precision_value,
            name,
        )


@pytest.mark.parametrize(
    ("has_initial_state", "output_final_state", "cotangents"),
    [
        (False, False, "output"),
        (True, True, "output"),
        (True, True, "state"),
        (True, True, "both"),
    ],
)
def test_public_ragged_forward_and_backward_fullgraph(
    has_initial_state,
    output_final_state,
    cotangents,
):
    """Compile normal and optional-cotangent public ragged autograd paths."""
    lengths = [65, 0, 63]
    inputs = make_kda_test_inputs(sum(lengths), requires_grad=True)
    compiled_inputs = clone_kda_inputs(inputs)
    initial_state = (
        (torch.randn(3, 1, 128, 128, device="cuda") / 8).requires_grad_()
        if has_initial_state
        else None
    )
    compiled_state = (
        None if initial_state is None else initial_state.detach().clone().requires_grad_()
    )
    output_grad = (
        None if cotangents == "state" else torch.randn(1, sum(lengths), 1, 128, device="cuda")
    )
    state_grad = torch.randn_like(initial_state) if cotangents in ("state", "both") else None
    offsets = cumulative_sequence_offsets(lengths)

    expected = _run_gradients(
        inputs,
        initial_state,
        offsets,
        output_grad,
        state_grad,
        output_final_state=output_final_state,
    )
    actual = _run_gradients(
        compiled_inputs,
        compiled_state,
        offsets,
        output_grad,
        state_grad,
        operation=torch.compile(chunk_kda, fullgraph=True),
        output_final_state=output_final_state,
    )

    _assert_run_close(actual, expected)


def test_public_ragged_forward_and_backward_cuda_graph_replay():
    """Replay changing L and M under fixed token and sequence capacities."""
    inputs = make_kda_test_inputs(128, requires_grad=True)
    initial_state = (torch.randn(4, 1, 128, 128, device="cuda") / 8).requires_grad_()
    output_grad = torch.randn(1, 128, 1, 128, device="cuda")
    state_grad = torch.randn_like(initial_state)
    # Capture N=4 state slots with M=2 nonempty sequences. Repeating the active
    # endpoint encodes the unused sequence-capacity tail without changing shape.
    offsets = cumulative_sequence_offsets([64, 64, 0, 0])

    _run_gradients(
        clone_kda_inputs(inputs),
        initial_state.detach().clone().requires_grad_(),
        offsets,
        output_grad,
        state_grad,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _run_gradients(inputs, initial_state, offsets, output_grad, state_grad)

    active_lengths = [17, 15, 33]
    active_tokens = sum(active_lengths)
    offsets.copy_(cumulative_sequence_offsets([*active_lengths, 0]))
    with torch.no_grad():
        for tensor in inputs:
            tensor[:, active_tokens:].fill_(float("nan"))
        # The loss multiplies the full physical buffer, so the suffix cotangent
        # must stay finite; the suffix inputs need not.
        output_grad[:, active_tokens:].zero_()
    graph.replay()
    torch.cuda.synchronize()

    expected = _run_gradients(
        tuple(tensor[:, :active_tokens].detach().clone().requires_grad_() for tensor in inputs),
        initial_state.detach().clone().requires_grad_(),
        cumulative_sequence_offsets([*active_lengths, 0]),
        output_grad[:, :active_tokens],
        state_grad,
    )
    _assert_run_close(actual, expected, active_tokens)


@pytest.mark.parametrize(
    ("captured_lengths", "active_lengths"),
    [
        pytest.param([64, 64, 64, 64], [65, 64, 0, 0], id="fewer-tokens-and-sequences"),
        pytest.param([64, 64], [128, 0], id="full-capacity-fewer-sequences"),
    ],
)
def test_public_ragged_cuda_graph_replay_with_fewer_sequences(captured_lengths, active_lengths):
    """Capture N sequences at capacity T, then replay M<N by zero-padding the offsets tail.

    The replay contract: a graph captured for T tokens across N sequences stays valid for any
    L<=T tokens across M<=N sequences when the caller repeats the active endpoint in the
    cu_seqlens tail. Empty tail slots must pass their initial state (and its cotangent)
    through unchanged.
    """
    capacity = sum(captured_lengths)
    sequences = len(captured_lengths)
    active_sequences = sum(1 for length in active_lengths if length > 0)
    active_tokens = sum(active_lengths)
    inputs = make_kda_test_inputs(capacity, requires_grad=True)
    initial_state = (torch.randn(sequences, 1, 128, 128, device="cuda") / 8).requires_grad_()
    output_grad = torch.randn(1, capacity, 1, 128, device="cuda")
    state_grad = torch.randn_like(initial_state)
    offsets = cumulative_sequence_offsets(captured_lengths)

    _run_gradients(
        clone_kda_inputs(inputs),
        initial_state.detach().clone().requires_grad_(),
        offsets,
        output_grad,
        state_grad,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, final_state, gradients = _run_gradients(
            inputs, initial_state, offsets, output_grad, state_grad
        )

    offsets.copy_(cumulative_sequence_offsets(active_lengths))
    with torch.no_grad():
        for tensor in inputs:
            tensor[:, active_tokens:].fill_(float("nan"))
        output_grad[:, active_tokens:].zero_()
    graph.replay()
    torch.cuda.synchronize()

    expected_output, expected_state, expected_gradients = _run_gradients(
        tuple(tensor[:, :active_tokens].detach().clone().requires_grad_() for tensor in inputs),
        initial_state[:active_sequences].detach().clone().requires_grad_(),
        cumulative_sequence_offsets(active_lengths[:active_sequences]),
        output_grad[:, :active_tokens],
        state_grad[:active_sequences],
    )

    torch.testing.assert_close(output[:, :active_tokens], expected_output, rtol=0, atol=0)
    torch.testing.assert_close(final_state[:active_sequences], expected_state, rtol=0, atol=0)
    torch.testing.assert_close(
        final_state[active_sequences:], initial_state[active_sequences:], rtol=0, atol=0
    )
    for index, (actual_gradient, expected_gradient) in enumerate(
        zip(gradients[:-1], expected_gradients[:-1], strict=True)
    ):
        torch.testing.assert_close(
            actual_gradient[:, :active_tokens],
            expected_gradient,
            rtol=3e-2,
            atol=3e-2,
            msg=lambda message, index=index: f"input gradient {index}: {message}",
        )
    torch.testing.assert_close(
        gradients[-1][:active_sequences], expected_gradients[-1], rtol=3e-2, atol=3e-2
    )
    torch.testing.assert_close(
        gradients[-1][active_sequences:], state_grad[active_sequences:], rtol=0, atol=0
    )


def test_dense_tail_batch_gradients_match_explicit_packed_lowering():
    batch, tokens = 2, 65
    dense_inputs = make_kda_test_inputs(tokens, batch=batch, requires_grad=True)
    packed_inputs = tuple(
        value.detach().reshape(1, batch * tokens, *value.shape[2:]).clone().requires_grad_()
        for value in dense_inputs
    )
    dense_state = (torch.randn(batch, 1, 128, 128, device="cuda") / 8).requires_grad_()
    packed_state = dense_state.detach().clone().requires_grad_()
    output_grad = torch.randn(batch, tokens, 1, 128, device="cuda")
    state_grad = torch.randn_like(dense_state)

    dense = _run_gradients(dense_inputs, dense_state, None, output_grad, state_grad)
    packed = _run_gradients(
        packed_inputs,
        packed_state,
        cumulative_sequence_offsets([tokens, tokens]),
        output_grad.reshape(1, batch * tokens, 1, 128),
        state_grad,
    )

    dense_output, dense_final_state, dense_gradients = dense
    packed_output, packed_final_state, packed_gradients = packed
    torch.testing.assert_close(
        dense_output, packed_output.reshape_as(dense_output), rtol=0, atol=0
    )
    torch.testing.assert_close(dense_final_state, packed_final_state, rtol=0, atol=0)
    for dense_gradient, packed_gradient in zip(
        dense_gradients[:-1], packed_gradients[:-1], strict=True
    ):
        torch.testing.assert_close(
            dense_gradient,
            packed_gradient.reshape_as(dense_gradient),
            rtol=3e-2,
            atol=3e-2,
        )
    torch.testing.assert_close(dense_gradients[-1], packed_gradients[-1], rtol=3e-2, atol=3e-2)


def test_public_auto_persistent_matches_static_composition(monkeypatch):
    """Route the complete public forward/backward through AUTO persistent plans."""
    from attn_gym.linear.kda import chunk_scheduler
    from attn_gym.linear.kda.chunk_schedule import ScheduleKind

    capacity = 4096
    active_lengths = [321, 0, 63, 128, 488]
    active_tokens = sum(active_lengths)
    inputs = make_kda_test_inputs(capacity, requires_grad=True)
    output_grad = torch.randn_like(inputs[0])
    with torch.no_grad():
        for tensor in inputs:
            tensor[:, active_tokens:].fill_(float("nan"))
        output_grad[:, active_tokens:].zero_()
    offsets = cumulative_sequence_offsets(active_lengths)

    kinds = []
    resolve_flat = chunk_scheduler.GridScheduler.resolve_flat
    resolve_chunk = chunk_scheduler.GridScheduler.resolve_chunk

    def record_flat(self, *args, **kwargs):
        resolved = resolve_flat(self, *args, **kwargs)
        kinds.append(resolved.kind)
        return resolved

    def record_chunk(self, *args, **kwargs):
        resolved = resolve_chunk(self, *args, **kwargs)
        kinds.append(resolved.kind)
        return resolved

    monkeypatch.setattr(chunk_scheduler.GridScheduler, "resolve_flat", record_flat)
    monkeypatch.setattr(chunk_scheduler.GridScheduler, "resolve_chunk", record_chunk)
    monkeypatch.setattr(chunk_scheduler, "PERSISTENT_AUTO_WAVES", 1 << 30)
    static = _run_gradients(
        clone_kda_inputs(inputs),
        None,
        offsets,
        output_grad,
        None,
        output_final_state=False,
    )
    assert set(kinds) == {ScheduleKind.STATIC}

    kinds.clear()
    monkeypatch.setattr(chunk_scheduler, "PERSISTENT_AUTO_WAVES", 0)
    persistent = _run_gradients(
        clone_kda_inputs(inputs),
        None,
        offsets,
        output_grad,
        None,
        output_final_state=False,
    )
    assert ScheduleKind.PERSISTENT in kinds
    torch.testing.assert_close(
        persistent[0][:, :active_tokens], static[0][:, :active_tokens], rtol=0, atol=0
    )
    assert persistent[1] is static[1] is None
    for persistent_gradient, static_gradient in zip(persistent[2], static[2], strict=True):
        torch.testing.assert_close(
            persistent_gradient[:, :active_tokens],
            static_gradient[:, :active_tokens],
            rtol=3e-2,
            atol=3e-2,
        )


def test_public_api_has_no_scheduling_knob():
    parameters = inspect.signature(chunk_kda).parameters
    assert "persistent" not in parameters
    assert "schedule" not in parameters
