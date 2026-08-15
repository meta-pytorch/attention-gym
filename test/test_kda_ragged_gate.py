# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness and graph-safety tests for Triton ragged KDA gate prefixes."""

from __future__ import annotations

import pytest
import torch
from torch._inductor import config as inductor_config

pytest.importorskip("triton")

from attn_gym.linear.kda.fwd.triton.gate_fwd import (
    _bounded_gate_cumsum_ragged_bwd_op,
    _bounded_gate_cumsum_ragged_fwd_op,
    bounded_gate_cumsum,
)
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

LOWER_BOUND = -3.25
RCP_LN2 = 1.4426950216


def _inputs(tokens: int, heads: int = 2, head_dim: int = 128, batch: int = 1):
    """Create nonuniform production-dtype inputs from one deterministic seed."""
    torch.manual_seed(1234 + batch + tokens + head_dim)
    shape = (batch, tokens, heads, head_dim)
    raw_gate = (0.5 * torch.randn(shape, device="cuda")).to(torch.bfloat16)
    A_log = 0.25 * torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = 0.25 * torch.randn(heads, head_dim, device="cuda", dtype=torch.float32)
    d_cumulative = torch.randn(shape, device="cuda", dtype=torch.float32)
    return raw_gate, A_log, dt_bias, d_cumulative


def _poison_inactive_suffix(
    raw_gate: torch.Tensor,
    d_cumulative: torch.Tensor,
    active_tokens: int,
) -> None:
    """Fill the undefined physical suffix with values that expose accidental reads."""
    with torch.no_grad():
        raw_gate[:, active_tokens:].fill_(torch.nan)
        d_cumulative[:, active_tokens:].fill_(torch.nan)


def _active_reference(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lengths: list[int],
    chunk_size: int,
) -> torch.Tensor:
    """Evaluate only the logical sequence-local chunks in the active prefix."""
    chunks = []
    token_start = 0
    decay = A_log.exp()[None, None, :, None]
    bias = dt_bias[None, None, :, :]
    for length in lengths:
        sequence_end = token_start + length
        for chunk_start in range(token_start, sequence_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, sequence_end)
            gate_input = raw_gate[:, chunk_start:chunk_end].float() + bias
            gate = LOWER_BOUND * torch.sigmoid(decay * gate_input)
            chunks.append(gate.cumsum(1) * RCP_LN2)
        token_start = sequence_end
    return torch.cat(chunks, dim=1)


@pytest.mark.parametrize("padding", (0, 17), ids=["full", "padded"])
@pytest.mark.parametrize(
    ("head_dim", "chunk_size"),
    (
        (128, 64),
        (96, 32),
        (1024, 64),
    ),
)
def test_bounded_gate_cumsum_ragged_matches_active_reference(head_dim, chunk_size, padding):
    """Ignore poisoned storage beyond the dynamic terminal offset in forward and backward."""
    lengths = [chunk_size + 1, 0, chunk_size - 1]
    active_tokens = sum(lengths)
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(
        active_tokens + padding,
        head_dim=head_dim,
    )
    _poison_inactive_suffix(raw_gate, d_cumulative, active_tokens)
    actual_inputs = tuple(
        tensor.detach().requires_grad_() for tensor in (raw_gate, A_log, dt_bias)
    )
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in actual_inputs)
    cu_seqlens = cumulative_sequence_offsets(lengths)

    actual = bounded_gate_cumsum(
        *actual_inputs,
        chunk_size=chunk_size,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
    )
    actual_gradients = torch.autograd.grad(actual, actual_inputs, d_cumulative)
    expected = _active_reference(*reference_inputs, lengths, chunk_size)
    expected_gradients = torch.autograd.grad(
        expected,
        reference_inputs,
        d_cumulative[:, :active_tokens],
    )

    torch.testing.assert_close(actual[:, :active_tokens], expected, rtol=1e-6, atol=8e-5)
    torch.testing.assert_close(
        actual_gradients[0][:, :active_tokens],
        expected_gradients[0][:, :active_tokens],
        rtol=4 * torch.finfo(torch.bfloat16).eps,
        atol=1e-6,
    )
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=5e-5, atol=7e-4)
    torch.testing.assert_close(actual_gradients[2], expected_gradients[2], rtol=5e-5, atol=7e-4)

    # The padded run must also match a physically sliced kernel run bit-for-bit.
    sliced_gate = actual_inputs[0][:, :active_tokens].detach().clone().requires_grad_()
    sliced_params = tuple(tensor.detach().clone().requires_grad_() for tensor in actual_inputs[1:])
    sliced = bounded_gate_cumsum(
        sliced_gate,
        *sliced_params,
        chunk_size=chunk_size,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
    )
    sliced_gradients = torch.autograd.grad(
        sliced,
        (sliced_gate, *sliced_params),
        d_cumulative[:, :active_tokens],
    )
    torch.testing.assert_close(actual[:, :active_tokens], sliced, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_gradients[0][:, :active_tokens], sliced_gradients[0], rtol=0, atol=0
    )
    for actual_gradient, sliced_gradient in zip(
        actual_gradients[1:], sliced_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual_gradient, sliced_gradient, rtol=0, atol=0)


def test_bounded_gate_cumsum_ragged_zero_active_tokens():
    """Produce zero parameter gradients when every physical token is inactive."""
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(64)
    _poison_inactive_suffix(raw_gate, d_cumulative, 0)
    inputs = tuple(tensor.detach().requires_grad_() for tensor in (raw_gate, A_log, dt_bias))
    cu_seqlens = cumulative_sequence_offsets([0, 0])

    output = bounded_gate_cumsum(*inputs, lower_bound=LOWER_BOUND, cu_seqlens=cu_seqlens)
    gradients = torch.autograd.grad(output, inputs, d_cumulative)

    assert output.shape == raw_gate.shape
    torch.testing.assert_close(gradients[1], torch.zeros_like(A_log), rtol=0, atol=0)
    torch.testing.assert_close(gradients[2], torch.zeros_like(dt_bias), rtol=0, atol=0)


def test_bounded_gate_cumsum_ragged_op_registration():
    """Validate functional schemas, fake tensors, and AOT dispatch for both compositions."""
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(128)
    cu_seqlens = cumulative_sequence_offsets([65, 0, 63])
    forward_args = (
        raw_gate.detach(),
        A_log.detach(),
        dt_bias.detach(),
        cu_seqlens,
        64,
        LOWER_BOUND,
    )
    # Autograd is intentionally owned by _BoundedGateCumsum rather than the raw operators.
    test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    torch.library.opcheck(
        _bounded_gate_cumsum_ragged_fwd_op,
        forward_args,
        test_utils=test_utils,
    )

    with torch.no_grad():
        _output, chunk_offsets = _bounded_gate_cumsum_ragged_fwd_op(*forward_args)
    torch.library.opcheck(
        _bounded_gate_cumsum_ragged_bwd_op,
        (
            raw_gate.detach(),
            A_log.detach(),
            dt_bias.detach(),
            d_cumulative.detach(),
            cu_seqlens,
            chunk_offsets,
            64,
            LOWER_BOUND,
            False,
        ),
        test_utils=test_utils,
    )


def test_bounded_gate_cumsum_ragged_reduce_overhead():
    """Require strict Inductor CUDA Graph capture with a poisoned inactive suffix."""
    lengths = [65, 0, 63]
    active_tokens = sum(lengths)
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(active_tokens + 17)
    _poison_inactive_suffix(raw_gate, d_cumulative, active_tokens)
    reference_inputs = tuple(
        tensor.detach().requires_grad_() for tensor in (raw_gate, A_log, dt_bias)
    )
    compiled_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in reference_inputs
    )
    cu_seqlens = cumulative_sequence_offsets(lengths)

    def operation(raw_gate, A_log, dt_bias, offsets):
        return bounded_gate_cumsum(
            raw_gate,
            A_log,
            dt_bias,
            lower_bound=LOWER_BOUND,
            cu_seqlens=offsets,
        )

    expected = _active_reference(*reference_inputs, lengths, 64)
    expected_gradients = torch.autograd.grad(
        expected,
        reference_inputs,
        d_cumulative[:, :active_tokens],
    )
    with inductor_config.patch("triton.cudagraph_or_error", True):
        actual = torch.compile(operation, fullgraph=True, mode="reduce-overhead")(
            *compiled_inputs, cu_seqlens
        )
        actual_gradients = torch.autograd.grad(actual, compiled_inputs, d_cumulative)

    torch.testing.assert_close(actual[:, :active_tokens], expected, rtol=1e-6, atol=8e-5)
    torch.testing.assert_close(
        actual_gradients[0][:, :active_tokens],
        expected_gradients[0][:, :active_tokens],
        rtol=torch.finfo(torch.bfloat16).eps,
        atol=1e-6,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients[1:], expected_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-5, atol=7e-4)


def test_bounded_gate_cumsum_ragged_cuda_graph_replay():
    """Replay one graph as the active endpoint shrinks and then grows back to capacity."""
    initial_lengths = [64, 0, 64]
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(145)
    fresh_gate = raw_gate.clone()
    fresh_d_cumulative = d_cumulative.clone()
    _poison_inactive_suffix(raw_gate, d_cumulative, sum(initial_lengths))
    inputs = tuple(tensor.detach().requires_grad_() for tensor in (raw_gate, A_log, dt_bias))
    cu_seqlens = cumulative_sequence_offsets(initial_lengths)

    def run(current_inputs, offsets):
        output = bounded_gate_cumsum(
            *current_inputs,
            lower_bound=LOWER_BOUND,
            cu_seqlens=offsets,
        )
        return output, torch.autograd.grad(output, current_inputs, d_cumulative)

    run(inputs, cu_seqlens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_gradients = run(inputs, cu_seqlens)

    for replay_lengths in ([65, 0, 32], [72, 1, 72]):
        active_tokens = sum(replay_lengths)
        with torch.no_grad():
            inputs[0].copy_(fresh_gate)
            d_cumulative.copy_(fresh_d_cumulative)
        _poison_inactive_suffix(inputs[0], d_cumulative, active_tokens)
        cu_seqlens.copy_(cumulative_sequence_offsets(replay_lengths))
        graph.replay()
        torch.cuda.synchronize()

        # Replay must match a fresh eager kernel run on the same storage bit-for-bit.
        eager_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
        eager, eager_gradients = run(eager_inputs, cu_seqlens)
        torch.testing.assert_close(
            actual[:, :active_tokens], eager[:, :active_tokens], rtol=0, atol=0
        )
        torch.testing.assert_close(
            actual_gradients[0][:, :active_tokens],
            eager_gradients[0][:, :active_tokens],
            rtol=0,
            atol=0,
        )
        for actual_gradient, eager_gradient in zip(
            actual_gradients[1:], eager_gradients[1:], strict=True
        ):
            torch.testing.assert_close(actual_gradient, eager_gradient, rtol=0, atol=0)

        reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
        expected = _active_reference(*reference_inputs, replay_lengths, 64)
        expected_gradients = torch.autograd.grad(
            expected,
            reference_inputs,
            d_cumulative[:, :active_tokens],
        )
        torch.testing.assert_close(actual[:, :active_tokens], expected, rtol=1e-6, atol=8e-5)
        torch.testing.assert_close(
            actual_gradients[0][:, :active_tokens],
            expected_gradients[0][:, :active_tokens],
            rtol=torch.finfo(torch.bfloat16).eps,
            atol=1e-6,
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients[1:], expected_gradients[1:], strict=True
        ):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-5, atol=7e-4)


@pytest.mark.parametrize(
    ("kind", "match"),
    (
        ("shape", "shape"),
        ("dtype", "CUDA int32"),
        ("device", "CUDA int32"),
        ("contiguous", "CUDA int32"),
    ),
)
def test_bounded_gate_cumsum_rejects_invalid_ragged_metadata(kind, match):
    raw_gate, A_log, dt_bias, _ = _inputs(64)
    if kind == "shape":
        cu_seqlens = torch.tensor([[0, 64]], device="cuda", dtype=torch.int32)
    elif kind == "dtype":
        cu_seqlens = torch.tensor([0, 64], device="cuda", dtype=torch.int64)
    elif kind == "device":
        cu_seqlens = torch.tensor([0, 64], dtype=torch.int32)
    else:
        cu_seqlens = torch.tensor([0, 1, 64, 65], device="cuda", dtype=torch.int32)[::2]
    with pytest.raises(ValueError, match=match):
        bounded_gate_cumsum(raw_gate, A_log, dt_bias, cu_seqlens=cu_seqlens)


def test_bounded_gate_cumsum_rejects_packed_batch_and_fastmath():
    raw_gate, A_log, dt_bias, _ = _inputs(64, batch=2)
    with pytest.raises(ValueError, match="batch size one"):
        bounded_gate_cumsum(
            raw_gate,
            A_log,
            dt_bias,
            cu_seqlens=cumulative_sequence_offsets([64]),
        )
    with pytest.raises(ValueError, match="does not support fastmath"):
        bounded_gate_cumsum(
            raw_gate[:1],
            A_log,
            dt_bias,
            cu_seqlens=cumulative_sequence_offsets([64]),
            fastmath=True,
        )
