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


def _inputs(tokens: int, heads: int = 2, head_dim: int = 128, batch: int = 1):
    """Create nonuniform production-dtype inputs from one deterministic seed."""
    torch.manual_seed(1234 + batch + tokens + head_dim)
    shape = (batch, tokens, heads, head_dim)
    raw_gate = (0.5 * torch.randn(shape, device="cuda")).to(torch.bfloat16)
    A_log = 0.25 * torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = 0.25 * torch.randn(heads, head_dim, device="cuda", dtype=torch.float32)
    d_cumulative = torch.randn(shape, device="cuda", dtype=torch.float32)
    return raw_gate, A_log, dt_bias, d_cumulative


@pytest.mark.parametrize(
    ("head_dim", "chunk_size"),
    (
        (128, 64),
        (96, 32),
        (1024, 64),
    ),
)
def test_bounded_gate_cumsum_ragged_matches_independent_sequences(head_dim, chunk_size):
    """Reset forward and all first-order gradients at ragged sequence boundaries."""
    lengths = [chunk_size + 1, 0, chunk_size - 1]
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(sum(lengths), head_dim=head_dim)
    packed_inputs = tuple(
        tensor.detach().requires_grad_() for tensor in (raw_gate, A_log, dt_bias)
    )
    independent_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in packed_inputs
    )

    actual = bounded_gate_cumsum(
        *packed_inputs,
        chunk_size=chunk_size,
        lower_bound=-3.25,
        cu_seqlens=cumulative_sequence_offsets(lengths),
    )
    actual_gradients = torch.autograd.grad(actual, packed_inputs, d_cumulative)

    independent_outputs = []
    token_start = 0
    for length in lengths:
        if length == 0:
            continue
        token_end = token_start + length
        independent_outputs.append(
            bounded_gate_cumsum(
                independent_inputs[0][:, token_start:token_end],
                independent_inputs[1],
                independent_inputs[2],
                chunk_size=chunk_size,
                lower_bound=-3.25,
            )
        )
        token_start = token_end
    expected = torch.cat(independent_outputs, dim=1)
    expected_gradients = torch.autograd.grad(expected, independent_inputs, d_cumulative)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_gradients[0],
        expected_gradients[0],
        rtol=4 * torch.finfo(torch.bfloat16).eps,
        atol=0,
    )
    torch.testing.assert_close(actual_gradients[1], expected_gradients[1], rtol=5e-5, atol=7e-4)
    torch.testing.assert_close(actual_gradients[2], expected_gradients[2], rtol=5e-5, atol=7e-4)


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
        -3.25,
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
            -3.25,
            False,
        ),
        test_utils=test_utils,
    )


def test_bounded_gate_cumsum_ragged_reduce_overhead():
    """Require Inductor CUDA Graph capture for packed forward and backward."""
    lengths = [65, 0, 63]
    inputs = _inputs(sum(lengths))
    eager_inputs = tuple(tensor.detach().requires_grad_() for tensor in inputs[:3])
    compiled_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs[:3])
    d_cumulative = inputs[3]
    cu_seqlens = cumulative_sequence_offsets(lengths)

    def operation(raw_gate, A_log, dt_bias, offsets):
        return bounded_gate_cumsum(
            raw_gate,
            A_log,
            dt_bias,
            lower_bound=-3.25,
            cu_seqlens=offsets,
        )

    expected = operation(*eager_inputs, cu_seqlens)
    expected_gradients = torch.autograd.grad(expected, eager_inputs, d_cumulative)
    with inductor_config.patch("triton.cudagraph_or_error", True):
        actual = torch.compile(operation, fullgraph=True, mode="reduce-overhead")(
            *compiled_inputs, cu_seqlens
        )
        actual_gradients = torch.autograd.grad(actual, compiled_inputs, d_cumulative)

    torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-5)
    torch.testing.assert_close(
        actual_gradients[0],
        expected_gradients[0],
        rtol=torch.finfo(torch.bfloat16).eps,
        atol=0,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients[1:], expected_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-5, atol=7e-4)


def test_bounded_gate_cumsum_ragged_cuda_graph_replay():
    """Replay one static graph after aligned offsets become ragged with an empty sequence."""
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(128)
    inputs = tuple(tensor.detach().requires_grad_() for tensor in (raw_gate, A_log, dt_bias))
    cu_seqlens = cumulative_sequence_offsets([64, 0, 64])

    def run(current_inputs, offsets):
        output = bounded_gate_cumsum(
            *current_inputs,
            lower_bound=-3.25,
            cu_seqlens=offsets,
        )
        return output, torch.autograd.grad(output, current_inputs, d_cumulative)

    run(inputs, cu_seqlens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_gradients = run(inputs, cu_seqlens)

    cu_seqlens.copy_(cumulative_sequence_offsets([65, 0, 63]))
    graph.replay()
    torch.cuda.synchronize()

    expected_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    expected, expected_gradients = run(expected_inputs, cumulative_sequence_offsets([65, 0, 63]))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


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
