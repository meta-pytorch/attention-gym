# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness and graph-safety tests for Triton ragged KDA gate prefixes."""

from __future__ import annotations

import importlib

import pytest
import torch
from torch._inductor import config as inductor_config

pytest.importorskip("triton")

from attn_gym.linear.kda.bwd.triton.gate_bwd import kda_gate_bwd_ragged
from attn_gym.linear.kda.chunk_scheduler import ScheduleRequest, prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.triton.gate_fwd import (
    _bounded_gate_cumsum_ragged_bwd_op,
    _bounded_gate_cumsum_ragged_fwd_op,
    bounded_gate_cumsum,
    bounded_gate_cumsum_ragged,
)
from attn_gym.linear.kda.utils import RCP_LN2
from attn_gym.testing.kda import clone_kda_inputs, cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

LOWER_BOUND = -3.25
BF16_EPS = torch.finfo(torch.bfloat16).eps


def _inputs(tokens: int, heads: int = 2, head_dim: int = 128, batch: int = 1):
    """Create nonuniform production-dtype inputs from one deterministic seed."""
    torch.manual_seed(1234 + batch + tokens + head_dim)
    shape = (batch, tokens, heads, head_dim)
    raw_gate = (0.5 * torch.randn(shape, device="cuda")).to(torch.bfloat16)
    A_log = 0.25 * torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = 0.25 * torch.randn(heads, head_dim, device="cuda", dtype=torch.float32)
    d_cumulative = torch.randn(shape, device="cuda", dtype=torch.float32)
    return raw_gate, A_log, dt_bias, d_cumulative


def _leaves(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Reuse the given storage as differentiable leaves."""
    return tuple(tensor.detach().requires_grad_() for tensor in tensors)


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


Run = tuple[torch.Tensor, tuple[torch.Tensor, ...]]


def _run(
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor,
    d_cumulative: torch.Tensor,
    chunk_size: int = 64,
) -> Run:
    """Run the ragged gate kernel and differentiate every leaf against one cotangent."""
    output = bounded_gate_cumsum(
        *inputs,
        chunk_size=chunk_size,
        lower_bound=LOWER_BOUND,
        cu_seqlens=cu_seqlens,
    )
    return output, torch.autograd.grad(output, inputs, d_cumulative)


def _reference_run(
    inputs: tuple[torch.Tensor, ...],
    lengths: list[int],
    d_cumulative: torch.Tensor,
    chunk_size: int = 64,
) -> Run:
    """Evaluate the Python reference and its gradients on the active prefix."""
    expected = _active_reference(*inputs, lengths, chunk_size)
    return expected, torch.autograd.grad(expected, inputs, d_cumulative[:, : sum(lengths)])


def _assert_matches_reference(actual: Run, expected: Run, active_tokens: int, *, gate_rtol):
    """Compare a kernel run against the Python reference over the active prefix."""
    actual_output, actual_gradients = actual
    expected_output, expected_gradients = expected
    torch.testing.assert_close(
        actual_output[:, :active_tokens], expected_output, rtol=1e-6, atol=8e-5
    )
    torch.testing.assert_close(
        actual_gradients[0][:, :active_tokens],
        expected_gradients[0][:, :active_tokens],
        rtol=gate_rtol,
        atol=1e-6,
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients[1:], expected_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=5e-5, atol=7e-4)


def _assert_bitwise_match(actual: Run, other: Run, active_tokens: int) -> None:
    """Require bit-for-bit agreement between two kernel runs over the active prefix."""
    actual_output, actual_gradients = actual
    other_output, other_gradients = other
    torch.testing.assert_close(
        actual_output[:, :active_tokens], other_output[:, :active_tokens], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_gradients[0][:, :active_tokens],
        other_gradients[0][:, :active_tokens],
        rtol=0,
        atol=0,
    )
    for actual_gradient, other_gradient in zip(
        actual_gradients[1:], other_gradients[1:], strict=True
    ):
        torch.testing.assert_close(actual_gradient, other_gradient, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("head_dim", "chunk_size"),
    (
        (128, 64),
        (96, 32),
        (1024, 64),
    ),
)
def test_bounded_gate_cumsum_ragged_matches_active_reference(head_dim, chunk_size):
    """Ignore poisoned storage beyond the dynamic terminal offset in forward and backward."""
    lengths = [chunk_size + 1, 0, chunk_size - 1]
    active_tokens = sum(lengths)
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(active_tokens + 17, head_dim=head_dim)
    _poison_inactive_suffix(raw_gate, d_cumulative, active_tokens)
    inputs = _leaves(raw_gate, A_log, dt_bias)
    reference_inputs = clone_kda_inputs(inputs)
    cu_seqlens = cumulative_sequence_offsets(lengths)

    actual = _run(inputs, cu_seqlens, d_cumulative, chunk_size)
    expected = _reference_run(reference_inputs, lengths, d_cumulative, chunk_size)
    _assert_matches_reference(actual, expected, active_tokens, gate_rtol=4 * BF16_EPS)

    # The padded run must also match an exactly sized kernel run bit-for-bit, which
    # transitively pins the full-capacity (no padding) numeric path as well.
    sliced_inputs = (
        inputs[0][:, :active_tokens].detach().clone().requires_grad_(),
        *clone_kda_inputs(inputs[1:]),
    )
    sliced = _run(sliced_inputs, cu_seqlens, d_cumulative[:, :active_tokens], chunk_size)
    _assert_bitwise_match(actual, sliced, active_tokens)


def test_bounded_gate_cumsum_ragged_zero_active_tokens():
    """Produce zero parameter gradients when every physical token is inactive."""
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(64)
    _poison_inactive_suffix(raw_gate, d_cumulative, 0)
    inputs = _leaves(raw_gate, A_log, dt_bias)

    output, gradients = _run(inputs, cumulative_sequence_offsets([0, 0]), d_cumulative)

    assert output.shape == raw_gate.shape
    torch.testing.assert_close(gradients[1], torch.zeros_like(A_log), rtol=0, atol=0)
    torch.testing.assert_close(gradients[2], torch.zeros_like(dt_bias), rtol=0, atol=0)


def test_bounded_gate_cumsum_ragged_op_registration():
    """Validate functional schemas, fake tensors, and AOT dispatch for both compositions."""
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(128)
    cu_seqlens = cumulative_sequence_offsets([65, 0, 63])
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, raw_gate.shape[1], 64)
    forward_args = (
        raw_gate.detach(),
        A_log.detach(),
        dt_bias.detach(),
        cu_seqlens,
        metadata.chunk_offsets,
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

    torch.library.opcheck(
        _bounded_gate_cumsum_ragged_bwd_op,
        (
            raw_gate.detach(),
            A_log.detach(),
            dt_bias.detach(),
            d_cumulative.detach(),
            cu_seqlens,
            metadata.chunk_offsets,
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
    reference_inputs = _leaves(raw_gate, A_log, dt_bias)
    compiled_inputs = clone_kda_inputs(reference_inputs)
    cu_seqlens = cumulative_sequence_offsets(lengths)

    def operation(raw_gate, A_log, dt_bias, offsets):
        return bounded_gate_cumsum(
            raw_gate,
            A_log,
            dt_bias,
            lower_bound=LOWER_BOUND,
            cu_seqlens=offsets,
        )

    expected = _reference_run(reference_inputs, lengths, d_cumulative)
    with inductor_config.patch("triton.cudagraph_or_error", True):
        actual = torch.compile(operation, fullgraph=True, mode="reduce-overhead")(
            *compiled_inputs, cu_seqlens
        )
        actual_gradients = torch.autograd.grad(actual, compiled_inputs, d_cumulative)

    _assert_matches_reference(
        (actual, actual_gradients), expected, active_tokens, gate_rtol=BF16_EPS
    )


def test_bounded_gate_cumsum_ragged_cuda_graph_replay():
    """Replay one graph as the active endpoint shrinks and then grows back to capacity."""
    initial_lengths = [64, 0, 64]
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(145)
    fresh_gate = raw_gate.clone()
    fresh_d_cumulative = d_cumulative.clone()
    _poison_inactive_suffix(raw_gate, d_cumulative, sum(initial_lengths))
    inputs = _leaves(raw_gate, A_log, dt_bias)
    cu_seqlens = cumulative_sequence_offsets(initial_lengths)

    _run(inputs, cu_seqlens, d_cumulative)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _run(inputs, cu_seqlens, d_cumulative)

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
        eager = _run(clone_kda_inputs(inputs), cu_seqlens, d_cumulative)
        _assert_bitwise_match(actual, eager, active_tokens)

        reference_inputs = clone_kda_inputs(inputs)
        expected = _reference_run(reference_inputs, replay_lengths, d_cumulative)
        _assert_matches_reference(actual, expected, active_tokens, gate_rtol=BF16_EPS)


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


def _gate_bwd(
    inputs: tuple[torch.Tensor, ...],
    d_cumulative: torch.Tensor,
    metadata,
    *,
    schedule: ScheduleRequest,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate the packed gate directly, bypassing the autograd composition."""
    return kda_gate_bwd_ragged(
        *inputs,
        d_cumulative,
        metadata,
        lower_bound=LOWER_BOUND,
        scale=RCP_LN2,
        schedule=schedule,
    )


@pytest.mark.parametrize("head_dim", (128, 1024))
def test_kda_gate_bwd_ragged_persistent_matches_static_over_capacity(head_dim):
    """Stay bit-identical to static scheduling when capacity dwarfs active work."""
    lengths = [65, 0, 63]
    active_tokens = sum(lengths)
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(16 * active_tokens, head_dim=head_dim)
    _poison_inactive_suffix(raw_gate, d_cumulative, active_tokens)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, raw_gate.shape[1], 64)
    assert metadata.capacity >= 8 * metadata.chunk_offsets[-1].item()

    inputs = (raw_gate, A_log, dt_bias)
    static = _gate_bwd(inputs, d_cumulative, metadata, schedule=ScheduleRequest.STATIC)
    persistent = _gate_bwd(inputs, d_cumulative, metadata, schedule=ScheduleRequest.PERSISTENT)

    assert torch.equal(persistent[0][:, :active_tokens], static[0][:, :active_tokens])
    for actual, other in zip(persistent[1:], static[1:], strict=True):
        assert torch.equal(actual, other)

    leaves = _leaves(*inputs)
    expected_output = _active_reference(*leaves, lengths, 64)
    expected = torch.autograd.grad(expected_output, leaves, d_cumulative[:, :active_tokens])
    torch.testing.assert_close(
        persistent[0][:, :active_tokens], expected[0][:, :active_tokens], rtol=BF16_EPS, atol=1e-6
    )
    for actual, reference in zip(persistent[1:], expected[1:], strict=True):
        torch.testing.assert_close(actual, reference, rtol=5e-5, atol=7e-4)


def test_kda_gate_bwd_ragged_persistent_handles_zero_capacity():
    """Reduce capacity-sized partials to zero parameter gradients with nothing launched."""
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(0)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([0]), 0, 64)
    assert metadata.capacity == 0

    dg, dA_log, ddt_bias = _gate_bwd(
        (raw_gate, A_log, dt_bias),
        d_cumulative,
        metadata,
        schedule=ScheduleRequest.PERSISTENT,
    )

    assert dg.shape == raw_gate.shape
    torch.testing.assert_close(dA_log, torch.zeros_like(A_log), rtol=0, atol=0)
    torch.testing.assert_close(ddt_bias, torch.zeros_like(dt_bias), rtol=0, atol=0)


@pytest.mark.parametrize("head_dim", (96, 128, 1024))
def test_bounded_gate_cumsum_persistent_matches_static_over_capacity(head_dim):
    """Stride a fixed worker grid over the active chunks far below capacity."""
    lengths = [65, 0, 63]
    active_tokens = sum(lengths)
    capacity_tokens = 16 * active_tokens
    raw_gate, A_log, dt_bias, _ = _inputs(capacity_tokens, head_dim=head_dim)
    with torch.no_grad():
        raw_gate[:, active_tokens:].fill_(torch.nan)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, capacity_tokens, 64)
    assert metadata.capacity >= 8 * ((active_tokens + 63) // 64)

    static = bounded_gate_cumsum_ragged(
        raw_gate,
        A_log,
        dt_bias,
        metadata,
        lower_bound=LOWER_BOUND,
        schedule=ScheduleRequest.STATIC,
    )
    persistent = bounded_gate_cumsum_ragged(
        raw_gate,
        A_log,
        dt_bias,
        metadata,
        lower_bound=LOWER_BOUND,
        schedule=ScheduleRequest.PERSISTENT,
    )
    expected = _active_reference(raw_gate, A_log, dt_bias, lengths, 64)

    assert torch.equal(persistent[:, :active_tokens], static[:, :active_tokens])
    torch.testing.assert_close(persistent[:, :active_tokens], expected, rtol=1e-6, atol=8e-5)


def test_bounded_gate_cumsum_persistent_handles_zero_capacity():
    """Skip the launch entirely when the schedule holds no chunks."""
    raw_gate, A_log, dt_bias, _ = _inputs(0)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([0]), 0, 64)
    assert metadata.capacity == 0

    output = bounded_gate_cumsum_ragged(
        raw_gate,
        A_log,
        dt_bias,
        metadata,
        lower_bound=LOWER_BOUND,
        schedule=ScheduleRequest.PERSISTENT,
    )

    assert output.shape == raw_gate.shape


def test_kda_gate_bwd_ragged_persistent_strides_multiple_tasks_per_worker(monkeypatch):
    """Force fewer workers than tasks so the stride loop iterates repeatedly."""
    from attn_gym.linear.kda import chunk_scheduler

    monkeypatch.setattr(chunk_scheduler.GridScheduler, "num_workers", lambda self, s, d: 2)
    lengths = [65, 0, 63, 130]
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(sum(lengths))
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(lengths), raw_gate.shape[1], 64
    )

    static = _gate_bwd(
        (raw_gate, A_log, dt_bias),
        d_cumulative,
        metadata,
        schedule=ScheduleRequest.STATIC,
    )
    persistent = _gate_bwd(
        (raw_gate, A_log, dt_bias),
        d_cumulative,
        metadata,
        schedule=ScheduleRequest.PERSISTENT,
    )
    for actual, other in zip(persistent, static, strict=True):
        assert torch.equal(actual, other)


def test_kda_gate_bwd_offset_width_specializations_match(monkeypatch):
    """Keep the normal int32 and forced int64 gate-backward paths equivalent."""
    module = importlib.import_module("attn_gym.linear.kda.bwd.triton.gate_bwd")
    raw_gate, A_log, dt_bias, d_cumulative = _inputs(64, head_dim=96)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([33, 31]), 64, 64)

    def run():
        return module.kda_gate_bwd_ragged(
            raw_gate,
            A_log,
            dt_bias,
            d_cumulative,
            metadata,
            lower_bound=LOWER_BOUND,
            scale=RCP_LN2,
        )

    monkeypatch.setattr(module, "requires_int64_offsets", lambda *_tensors: False)
    outputs32 = run()
    monkeypatch.setattr(module, "requires_int64_offsets", lambda *_tensors: True)
    outputs64 = run()

    for output64, output32 in zip(outputs64, outputs32, strict=True):
        torch.testing.assert_close(output64, output32, rtol=0, atol=0)
