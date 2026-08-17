# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness and integration tests for the fused KDA recurrence."""

from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import recurrent_kda
from attn_gym.linear.kda.fwd.triton.recurrent import (
    _recurrent_fwd_no_state_op,
    _recurrent_fwd_op,
)
from attn_gym.linear.kda.naive import naive_recurrent_kda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="recurrent_kda requires CUDA"
)

BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _inputs(
    batch: int = 2,
    tokens: int = 37,
    heads: int = 2,
    key_dim: int = 64,
    value_dim: int = 64,
    dtype: torch.dtype = torch.float32,
    initial_state: bool = False,
    seed: int = 0,
):
    torch.manual_seed(seed)
    # KDA L2-normalizes q and k before the core; unnormalized keys make the
    # delta-rule recurrence exponentially unstable and useless for comparison.
    q = F.normalize(torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(batch, tokens, heads, value_dim, device="cuda", dtype=dtype)
    # Realistic bounded log2 decays keep the recurrence stable over the scan.
    gate = -torch.rand(batch, tokens, heads, key_dim, device="cuda") * 3.0
    beta = torch.rand(batch, tokens, heads, device="cuda")
    state = torch.randn(batch, heads, key_dim, value_dim, device="cuda") if initial_state else None
    return q, k, v, gate, beta, state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_initial_state", [False, True])
@pytest.mark.parametrize("tokens", [1, 37])
def test_recurrent_matches_naive_dense(dtype: torch.dtype, use_initial_state: bool, tokens: int):
    """Match the reference scan on dense batches, including single-token decode."""
    q, k, v, gate, beta, state = _inputs(
        tokens=tokens, dtype=dtype, initial_state=use_initial_state
    )

    output, final_state = recurrent_kda(q, k, v, gate, beta, state, output_final_state=True)
    expected, expected_state = naive_recurrent_kda(
        q.float(),
        k.float(),
        v.float(),
        gate,
        beta,
        initial_state=state,
        output_final_state=True,
    )

    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    assert output.dtype == q.dtype
    assert final_state is not None and final_state.dtype == torch.float32
    torch.testing.assert_close(output.float(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(final_state, expected_state, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize(("key_dim", "value_dim"), [(80, 48), (128, 128)])
def test_recurrent_matches_naive_non_power_of_two(key_dim: int, value_dim: int):
    """Mask partial key and value blocks correctly."""
    q, k, v, gate, beta, _ = _inputs(key_dim=key_dim, value_dim=value_dim, seed=1)

    output, final_state = recurrent_kda(q, k, v, gate, beta, output_final_state=True)
    expected, expected_state = naive_recurrent_kda(q, k, v, gate, beta, output_final_state=True)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_state, expected_state, rtol=1e-5, atol=1e-5)


def test_recurrent_packed_capacity_and_empty_slots():
    """Pass empty-slot state through and ignore rows past the terminal offset."""
    q, k, v, gate, beta, _ = _inputs(batch=1, tokens=32, seed=3)
    cu_seqlens = torch.tensor([0, 0, 11, 27, 27], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(4, q.shape[2], q.shape[3], v.shape[-1], device="cuda")

    output, final_state = recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    assert final_state is not None
    # Empty padding slots preserve their incoming state bitwise.
    torch.testing.assert_close(final_state[0], initial_state[0], rtol=0, atol=0)
    torch.testing.assert_close(final_state[3], initial_state[3], rtol=0, atol=0)
    for sequence, (start, end) in enumerate(pairwise(cu_seqlens.cpu().tolist())):
        if start == end:
            continue
        expected, expected_state = naive_recurrent_kda(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            gate[:, start:end],
            beta[:, start:end],
            initial_state=initial_state[sequence : sequence + 1],
            output_final_state=True,
        )
        torch.testing.assert_close(output[:, start:end], expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            final_state[sequence : sequence + 1], expected_state, rtol=1e-5, atol=1e-5
        )


@pytest.mark.skipif(not BLACKWELL, reason="chunk_kda requires CUDA capability 10.0")
def test_recurrent_agrees_with_chunked_core():
    """Cross-check the decode scan against the training core on shared inputs."""
    from attn_gym.linear import bounded_gate_cumsum, chunk_kda

    batch, tokens, heads, head_dim = 2, 128, 2, 128
    q, k, v, _, beta = _inputs(
        batch=batch,
        tokens=tokens,
        heads=heads,
        key_dim=head_dim,
        value_dim=head_dim,
        dtype=torch.bfloat16,
        seed=4,
    )[:5]
    raw_gate = torch.randn(batch, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device="cuda")
    dt_bias = torch.zeros(heads, head_dim, device="cuda")

    per_token_gate = bounded_gate_cumsum(raw_gate, a_log, dt_bias, chunk_size=1)
    cumulative_gate = bounded_gate_cumsum(raw_gate, a_log, dt_bias, chunk_size=64)

    recurrent_output, recurrent_state = recurrent_kda(
        q, k, v, per_token_gate, beta, output_final_state=True
    )
    chunked_output, chunked_state = chunk_kda(
        q, k, v, cumulative_gate, beta, output_final_state=True
    )
    torch.testing.assert_close(
        recurrent_output.float(), chunked_output.float(), rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(recurrent_state, chunked_state, rtol=5e-2, atol=5e-2)


def test_recurrent_validates_public_contract():
    """Reject malformed inputs at the public boundary before a kernel launch."""
    q, k, v, gate, beta, _ = _inputs(tokens=4)
    with pytest.raises(ValueError, match="k must have shape"):
        recurrent_kda(q, k[:, :-1], v, gate, beta)
    with pytest.raises(ValueError, match="gate must have shape"):
        recurrent_kda(q, k, v, gate[..., :-1], beta)
    with pytest.raises(ValueError, match="beta must have shape"):
        recurrent_kda(q, k, v, gate, beta[:, :, :-1])
    with pytest.raises(ValueError, match="initial_state must have shape"):
        recurrent_kda(q, k, v, gate, beta, q.new_zeros(1, 1, 1, 1))
    with pytest.raises(ValueError, match="batch size one"):
        recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            cu_seqlens=torch.tensor([0, 4], device="cuda", dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="num_sequences"):
        recurrent_kda(
            q[:1],
            k[:1],
            v[:1],
            gate[:1],
            beta[:1],
            cu_seqlens=torch.tensor([0], device="cuda", dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="contiguous int32"):
        recurrent_kda(
            q[:1],
            k[:1],
            v[:1],
            gate[:1],
            beta[:1],
            cu_seqlens=torch.tensor([0, 4], device="cuda", dtype=torch.int64),
        )
    with pytest.raises(ValueError, match="requires K in"):
        big = q.new_zeros(1, 4, 2, 512)
        recurrent_kda(big, big, big, big, q.new_zeros(1, 4, 2))
    with pytest.raises(TypeError, match="inputs must use one of"):
        recurrent_kda(q.double(), k.double(), v.double(), gate.double(), beta.double())


@pytest.mark.parametrize("operand", range(6))
def test_recurrent_rejects_gradient_tracking(operand: int):
    """State a clear inference-only contract for every gradient-tracking operand."""
    tensors = list(_inputs(tokens=4, initial_state=True))
    tensors[operand] = tensors[operand].float().requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        recurrent_kda(*tensors)
    with torch.no_grad():
        output, _ = recurrent_kda(*tensors)
    assert not output.requires_grad


@pytest.mark.parametrize("packed", [False, True])
def test_recurrent_custom_op_registration(packed: bool):
    """Exercise the schema and fake implementation for both modes."""
    batch = 1 if packed else 2
    q, k, v, gate, beta, _ = _inputs(batch=batch, tokens=17)
    cu_seqlens = torch.tensor([0, 2, 7, 17], device="cuda", dtype=torch.int32) if packed else None
    num_sequences = 3 if packed else batch
    state = torch.randn(num_sequences, q.shape[2], q.shape[3], v.shape[-1], device="cuda")
    torch.library.opcheck(_recurrent_fwd_op, (q, k, v, gate, beta, state, cu_seqlens))
    torch.library.opcheck(_recurrent_fwd_no_state_op, (q, k, v, gate, beta, state, cu_seqlens))


@pytest.mark.parametrize("output_final_state", [False, True])
def test_recurrent_fullgraph_compile(output_final_state: bool):
    """Compile both optional-state branches of the public operation."""
    q, k, v, gate, beta, state = _inputs(initial_state=True)

    expected, expected_state = recurrent_kda(
        q, k, v, gate, beta, state, output_final_state=output_final_state
    )
    compiled = torch.compile(recurrent_kda, fullgraph=True)
    output, final_state = compiled(
        q, k, v, gate, beta, state, output_final_state=output_final_state
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    if output_final_state:
        torch.testing.assert_close(final_state, expected_state, rtol=0, atol=0)
    else:
        assert final_state is None and expected_state is None


def test_recurrent_cuda_graph_replay():
    """Replay fixed shapes with mutated boundaries, values, and history."""
    q, k, v, gate, beta, _ = _inputs(batch=1, tokens=32, seed=5)
    cu_seqlens = torch.tensor([0, 11, 27, 32], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(3, q.shape[2], q.shape[3], v.shape[-1], device="cuda")
    _recurrent_fwd_op(q, k, v, gate, beta, initial_state, cu_seqlens)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_state = _recurrent_fwd_op(
            q, k, v, gate, beta, initial_state, cu_seqlens
        )

    active_tokens = 23
    with torch.no_grad():
        initial_state.add_(0.25)
        cu_seqlens.copy_(
            torch.tensor([0, 8, active_tokens, active_tokens], device="cuda", dtype=torch.int32)
        )
        q[:, active_tokens:].fill_(float("nan"))
        v[:, active_tokens:].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    expected_output, expected_state = _recurrent_fwd_op(
        q, k, v, gate, beta, initial_state, cu_seqlens
    )
    torch.testing.assert_close(
        captured_output[:, :active_tokens],
        expected_output[:, :active_tokens],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(captured_state, expected_state, rtol=0, atol=0)


def test_recurrent_launches_beyond_grid_y_limit():
    """Flat 1-D launches must survive sequence-head counts above 65,535."""
    batch, heads, head_dim = 2200, 32, 16
    assert batch * heads > 65_535
    q, k, v, gate, beta, _ = _inputs(
        batch=batch, tokens=1, heads=heads, key_dim=head_dim, value_dim=head_dim, seed=6
    )
    output, _ = recurrent_kda(q, k, v, gate, beta)
    expected, _ = naive_recurrent_kda(q, k, v, gate, beta)
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def test_naive_recurrent_packed_matches_public_contract():
    """The pure reference honors packed semantics: empty slots and capacity tails."""
    q, k, v, gate, beta, _ = _inputs(batch=1, tokens=32, seed=7)
    cu_seqlens = torch.tensor([0, 0, 11, 27, 27], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(4, q.shape[2], q.shape[3], v.shape[-1], device="cuda")

    output, final_state = naive_recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    assert output.shape == v.shape and final_state is not None
    torch.testing.assert_close(final_state[0], initial_state[0], rtol=0, atol=0)
    torch.testing.assert_close(final_state[3], initial_state[3], rtol=0, atol=0)
    torch.testing.assert_close(output[:, 27:], torch.zeros_like(output[:, 27:]), rtol=0, atol=0)
    fused, fused_state = recurrent_kda(
        q, k, v, gate, beta, initial_state, cu_seqlens=cu_seqlens, output_final_state=True
    )
    torch.testing.assert_close(output[:, :27], fused[:, :27], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final_state, fused_state, rtol=1e-5, atol=1e-5)
