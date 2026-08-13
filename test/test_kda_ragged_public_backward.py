"""End-to-end gradient tests for public ragged CuTe KDA."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import chunk_kda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe KDA core requires CUDA capability 10.0 or newer",
)


def _offsets(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )


def _inputs(tokens: int, batch: int = 1):
    torch.manual_seed(41)
    shape = (batch, tokens, 1, 128)
    values = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        -torch.rand(shape, device="cuda"),
        torch.rand(batch, tokens, 1, device="cuda"),
    ]
    return tuple(value.requires_grad_() for value in values)


def _clone_inputs(inputs):
    return tuple(value.detach().clone().requires_grad_() for value in inputs)


def _run_gradients(
    inputs,
    initial_state,
    offsets,
    output_grad,
    state_grad,
    operation=chunk_kda,
):
    output, final_state = operation(
        *inputs,
        initial_state,
        cu_seqlens=offsets,
        output_final_state=True,
    )
    loss = (output.float() * output_grad).sum() + (final_state * state_grad).sum()
    return output, final_state, torch.autograd.grad(loss, (*inputs, initial_state))


def _assert_run_close(actual, expected):
    actual_output, actual_state, actual_gradients = actual
    expected_output, expected_state, expected_gradients = expected
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=3e-2)


def test_public_ragged_backward_matches_independent_sequences():
    lengths = [65, 0, 63]
    tokens = sum(lengths)
    packed_inputs = _inputs(tokens)
    initial_state = (torch.randn(3, 1, 128, 128, device="cuda") / 8).requires_grad_()
    output_grad = torch.randn(1, tokens, 1, 128, device="cuda")
    state_grad = torch.randn_like(initial_state)

    output, final_state, gradients = _run_gradients(
        packed_inputs,
        initial_state,
        _offsets(lengths),
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
            _offsets([length]),
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


def test_public_ragged_forward_and_backward_fullgraph():
    lengths = [65, 63]
    inputs = _inputs(sum(lengths))
    compiled_inputs = _clone_inputs(inputs)
    initial_state = (torch.randn(2, 1, 128, 128, device="cuda") / 8).requires_grad_()
    compiled_state = initial_state.detach().clone().requires_grad_()
    output_grad = torch.randn(1, sum(lengths), 1, 128, device="cuda")
    state_grad = torch.randn_like(initial_state)
    offsets = _offsets(lengths)

    expected = _run_gradients(inputs, initial_state, offsets, output_grad, state_grad)
    compiled = torch.compile(chunk_kda, fullgraph=True)
    actual = _run_gradients(
        compiled_inputs,
        compiled_state,
        offsets,
        output_grad,
        state_grad,
        operation=compiled,
    )

    _assert_run_close(actual, expected)


def test_public_ragged_forward_and_backward_cuda_graph_replay():
    inputs = _inputs(128)
    initial_state = (torch.randn(2, 1, 128, 128, device="cuda") / 8).requires_grad_()
    output_grad = torch.randn(1, 128, 1, 128, device="cuda")
    state_grad = torch.randn_like(initial_state)
    offsets = _offsets([64, 64])

    _run_gradients(
        _clone_inputs(inputs),
        initial_state.detach().clone().requires_grad_(),
        offsets,
        output_grad,
        state_grad,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _run_gradients(inputs, initial_state, offsets, output_grad, state_grad)

    offsets.copy_(_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    expected = _run_gradients(
        _clone_inputs(inputs),
        initial_state.detach().clone().requires_grad_(),
        _offsets([65, 63]),
        output_grad,
        state_grad,
    )
    _assert_run_close(actual, expected)


def test_dense_tail_batch_gradients_match_explicit_packed_lowering():
    batch, tokens = 2, 65
    dense_inputs = _inputs(tokens, batch=batch)
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
        _offsets([tokens, tokens]),
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
