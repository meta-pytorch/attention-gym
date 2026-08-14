"""Custom-op contract tests for composed ragged KDA autograd."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear import naive_chunk_kda_from_cumulative
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    _chunk_kda_bwd_op,
    _chunk_kda_bwd_with_state_grad_op,
    _chunk_kda_fwd_ragged_op,
    _chunk_kda_fwd_ragged_with_state_op,
    _ChunkKDARagged,
)
from attn_gym.testing.kda import (
    clone_kda_inputs,
    cumulative_sequence_offsets,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTe KDA core requires CUDA capability 10.0 or 10.3",
)


def test_ragged_custom_op_registrations():
    inputs = make_kda_test_inputs(128, requires_grad=True)
    initial_state = (torch.randn(2, 1, 128, 128, device="cuda") / 8).requires_grad_()
    cu_seqlens = cumulative_sequence_offsets([65, 63])
    forward_args = (
        *(value.detach() for value in inputs),
        initial_state.detach(),
        cu_seqlens,
    )
    torch.library.opcheck(
        _chunk_kda_fwd_ragged_with_state_op,
        forward_args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    torch.library.opcheck(
        _chunk_kda_fwd_ragged_op,
        forward_args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    with torch.no_grad():
        output, state, Aqk, Akk, chunk_offsets = _chunk_kda_fwd_ragged_with_state_op(*forward_args)
    torch.library.opcheck(
        _chunk_kda_bwd_with_state_grad_op,
        (
            *(value.detach() for value in inputs),
            Aqk,
            Akk,
            cu_seqlens,
            chunk_offsets,
            chunk_offsets.new_empty(()),
            torch.randn_like(output),
            torch.randn_like(state),
            initial_state.detach(),
            True,
            False,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    no_state_args = (*(value.detach() for value in inputs), None, cu_seqlens)
    with torch.no_grad():
        output, Aqk, Akk, chunk_offsets = _chunk_kda_fwd_ragged_op(*no_state_args)
    torch.library.opcheck(
        _chunk_kda_bwd_op,
        (
            *(value.detach() for value in inputs),
            Aqk,
            Akk,
            cu_seqlens,
            chunk_offsets,
            chunk_offsets[-1:],
            torch.randn_like(output),
            None,
            None,
            True,
            False,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )


def _run_composed_gradients(
    operation,
    inputs,
    initial_state,
    cu_seqlens,
    output_final_state,
    output_grad,
    state_grad,
):
    result = operation(*inputs, initial_state, cu_seqlens)
    if output_final_state:
        output, final_state = result
        # A state-only loss exercises the optional d_output=None path.
        loss = (final_state * state_grad).sum()
    else:
        output, final_state = result, None
        # Omitting final state exercises d_final_state=None.
        loss = (output.float() * output_grad).sum()
    targets = (*inputs, *((initial_state,) if initial_state is not None else ()))
    gradients = torch.autograd.grad(loss, targets, allow_unused=True)
    gradients = tuple(
        torch.zeros_like(target) if gradient is None else gradient
        for target, gradient in zip(targets, gradients, strict=True)
    )
    return output, final_state, gradients


def _run_naive_gradients(
    inputs,
    initial_state,
    lengths,
    output_final_state,
    output_grad,
    state_grad,
):
    outputs = []
    states = []
    token_start = 0
    for sequence, length in enumerate(lengths):
        if length == 0:
            if output_final_state:
                states.append(initial_state[sequence])
            continue
        token_slice = slice(token_start, token_start + length)
        sequence_state = None if initial_state is None else initial_state[sequence : sequence + 1]
        output, final_state = naive_chunk_kda_from_cumulative(
            *(value[:, token_slice].float() for value in inputs[:4]),
            inputs[4][:, token_slice].float(),
            initial_state=sequence_state,
            output_final_state=output_final_state,
            chunk_size=64,
        )
        outputs.append(output)
        if output_final_state:
            states.append(final_state[0])
        token_start += length

    output = torch.cat(outputs, dim=1)
    final_state = torch.stack(states) if output_final_state else None
    loss = (final_state * state_grad).sum() if output_final_state else (output * output_grad).sum()
    targets = (*inputs, *((initial_state,) if initial_state is not None else ()))
    gradients = torch.autograd.grad(loss, targets, allow_unused=True)
    gradients = tuple(
        torch.zeros_like(target) if gradient is None else gradient
        for target, gradient in zip(targets, gradients, strict=True)
    )
    return output, final_state, gradients


def _assert_run_close(actual, expected):
    actual_output, actual_state, actual_gradients = actual
    expected_output, expected_state, expected_gradients = expected
    torch.testing.assert_close(
        actual_output.float(), expected_output.float(), rtol=2e-2, atol=2e-3
    )
    if actual_state is not None:
        torch.testing.assert_close(actual_state, expected_state, rtol=2e-2, atol=3e-3)
    for gradient, reference in zip(actual_gradients, expected_gradients, strict=True):
        tolerance = 3e-2 + 3e-2 * reference.float().abs().max()
        assert (gradient.float() - reference.float()).abs().max() <= tolerance


@pytest.mark.parametrize(
    ("has_initial_state", "output_final_state"),
    [(False, False), (True, True)],
)
def test_ragged_autograd_composition_matches_reference_and_fullgraph(
    has_initial_state,
    output_final_state,
):
    lengths = [65, 0, 63]
    tokens = sum(lengths)
    inputs = make_kda_test_inputs(tokens, requires_grad=True)
    initial_state = (
        (torch.randn(3, 1, 128, 128, device="cuda") / 8).requires_grad_()
        if has_initial_state
        else None
    )
    cu_seqlens = cumulative_sequence_offsets(lengths)
    output_grad = torch.randn(1, tokens, 1, 128, device="cuda")
    state_grad = (
        torch.randn_like(initial_state)
        if initial_state is not None and output_final_state
        else None
    )

    def operation(q, k, v, cumulative_gate, beta, state, offsets):
        return _ChunkKDARagged.apply(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            state,
            offsets,
            output_final_state,
            False,
        )

    actual = _run_composed_gradients(
        operation,
        inputs,
        initial_state,
        cu_seqlens,
        output_final_state,
        output_grad,
        state_grad,
    )
    reference_inputs = clone_kda_inputs(inputs)
    reference_state = (
        None if initial_state is None else initial_state.detach().clone().requires_grad_()
    )
    expected = _run_naive_gradients(
        reference_inputs,
        reference_state,
        lengths,
        output_final_state,
        output_grad,
        state_grad,
    )
    _assert_run_close(actual, expected)

    compiled = torch.compile(operation, fullgraph=True)
    compiled_inputs = clone_kda_inputs(inputs)
    compiled_state = (
        None if initial_state is None else initial_state.detach().clone().requires_grad_()
    )
    compiled_result = _run_composed_gradients(
        compiled,
        compiled_inputs,
        compiled_state,
        cu_seqlens,
        output_final_state,
        output_grad,
        state_grad,
    )
    _assert_run_close(compiled_result, expected)
