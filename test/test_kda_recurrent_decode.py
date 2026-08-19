# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness and integration tests for fused serving-oriented KDA decode."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import recurrent_kda_decode
from attn_gym.linear.kda.naive import naive_recurrent_kda
from attn_gym.linear.kda.ops import recurrent_decode_op

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="recurrent_kda_decode requires CUDA"
)


def _strided_state_pool(
    num_slots: int,
    heads: int,
    key_dim: int,
    value_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix, suffix = 11, 17
    state_elements = heads * key_dim * value_dim
    storage = torch.randn(
        num_slots, prefix + state_elements + suffix, device="cuda", dtype=torch.float32
    )
    state = storage[:, prefix : prefix + state_elements].view(num_slots, heads, key_dim, value_dim)
    assert not state.is_contiguous()
    return storage, state


def _decode_inputs(
    *,
    batch: int = 3,
    heads: int = 2,
    key_dim: int = 64,
    value_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    torch.manual_seed(seed)
    packed_qkv = torch.randn(
        batch,
        heads,
        2 * key_dim + value_dim,
        device="cuda",
        dtype=dtype,
    ).flatten(1)
    raw_gate = torch.randn(1, batch, heads, key_dim, device="cuda", dtype=dtype)
    raw_beta = torch.randn(1, batch, heads, device="cuda", dtype=dtype)
    A_log = 0.1 * torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = 0.1 * torch.randn(heads, key_dim, device="cuda", dtype=torch.float32)
    storage, state_cache = _strided_state_pool(7, heads, key_dim, value_dim)
    state_indices = torch.tensor([5, 1, 3], device="cuda", dtype=torch.int32)[:batch]
    return (
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        storage,
        state_cache,
        state_indices,
    )


def _reference_decode(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    lower_bound: float | None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = packed_qkv.shape[0]
    heads, key_dim, value_dim = state_cache.shape[1:]
    per_head = packed_qkv.view(batch, heads, 2 * key_dim + value_dim)
    q = per_head[..., :key_dim].unsqueeze(1).float()
    k = per_head[..., key_dim : 2 * key_dim].unsqueeze(1).float()
    v = per_head[..., 2 * key_dim :].unsqueeze(1)
    q *= torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k *= torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)

    gate_input = raw_gate.float() + dt_bias[None, None]
    decay = A_log.exp()[None, None, :, None]
    if lower_bound is None:
        gate = -decay * F.softplus(gate_input)
    else:
        gate = lower_bound * torch.sigmoid(decay * gate_input)
    gate *= math.log2(math.e)
    beta = raw_beta.float().sigmoid()

    active = state_indices > 0
    output = packed_qkv.new_zeros(1, batch, heads, value_dim)
    expected_cache = state_cache.clone()
    if active.any():
        active_indices = state_indices[active].long()
        active_output, active_state = naive_recurrent_kda(
            q[active],
            k[active],
            v[active],
            gate[:, active].transpose(0, 1),
            beta[:, active].transpose(0, 1),
            scale=scale,
            initial_state=state_cache[active_indices],
            output_final_state=True,
        )
        output[:, active] = active_output.transpose(0, 1).to(output.dtype)
        expected_cache[active_indices] = active_state
    return output, expected_cache


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("lower_bound", [-5.0, None])
@pytest.mark.parametrize(("key_dim", "value_dim"), [(64, 64), (80, 48)])
def test_recurrent_decode_matches_reference(
    dtype: torch.dtype,
    lower_bound: float | None,
    key_dim: int,
    value_dim: int,
):
    inputs = _decode_inputs(dtype=dtype, key_dim=key_dim, value_dim=value_dim)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    before = state_cache.clone()

    output = recurrent_kda_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        lower_bound=lower_bound,
    )
    expected, expected_cache = _reference_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        before,
        state_indices,
        lower_bound,
    )

    tolerance = 1e-5 if dtype == torch.float32 else 3e-2
    assert output.shape == expected.shape and output.dtype == dtype
    torch.testing.assert_close(output.float(), expected.float(), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(state_cache, expected_cache, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("padding_slot", [0, -1])
def test_recurrent_decode_padding_slot_is_ignored(padding_slot: int):
    inputs = list(_decode_inputs())
    inputs[-1] = torch.tensor([5, padding_slot, 3], device="cuda", dtype=torch.int32)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    before = state_cache.clone()

    output = recurrent_kda_decode(
        packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_cache, state_indices
    )

    torch.testing.assert_close(output[:, 1], torch.zeros_like(output[:, 1]), rtol=0, atol=0)
    torch.testing.assert_close(state_cache[0], before[0], rtol=0, atol=0)


def test_recurrent_decode_validates_contract():
    inputs = list(_decode_inputs(batch=2))
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    with pytest.raises(ValueError, match="packed_qkv must have shape"):
        recurrent_kda_decode(
            packed_qkv[:, :-1],
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
        )
    with pytest.raises(ValueError, match="raw_gate must have shape"):
        recurrent_kda_decode(
            packed_qkv,
            raw_gate[..., :-1],
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
        )
    with pytest.raises(TypeError, match="state_cache must use float32"):
        recurrent_kda_decode(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache.bfloat16(),
            state_indices,
        )
    with pytest.raises(ValueError, match="finite and negative"):
        recurrent_kda_decode(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
            lower_bound=1.0,
        )


def test_recurrent_decode_rejects_gradient_tracking():
    inputs = list(_decode_inputs(batch=2))
    inputs[0] = inputs[0].requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        recurrent_kda_decode(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[6], inputs[7]
        )


@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_recurrent_decode_custom_op_registration(lower_bound: float | None):
    inputs = _decode_inputs(batch=2)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    torch.library.opcheck(
        recurrent_decode_op,
        (
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
            0.0 if lower_bound is None else lower_bound,
            lower_bound is not None,
            state_cache.shape[2] ** -0.5,
        ),
    )


@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_recurrent_decode_fullgraph_compile(lower_bound: float | None):
    inputs = _decode_inputs(batch=2)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, eager_cache, state_indices = inputs
    _, compiled_cache = _strided_state_pool(
        eager_cache.shape[0], eager_cache.shape[1], eager_cache.shape[2], eager_cache.shape[3]
    )
    compiled_cache.copy_(eager_cache)

    expected = recurrent_kda_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        eager_cache,
        state_indices,
        lower_bound=lower_bound,
    )
    compiled = torch.compile(recurrent_kda_decode, fullgraph=True)
    output = compiled(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        compiled_cache,
        state_indices,
        lower_bound=lower_bound,
    )

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled_cache, eager_cache, rtol=0, atol=0)


def test_recurrent_decode_cuda_graph_replay():
    inputs = _decode_inputs(batch=3, seed=4)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, storage, state_cache, state_indices = inputs
    recurrent_kda_decode(
        packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_cache, state_indices
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = recurrent_kda_decode(
            packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_cache, state_indices
        )

    with torch.no_grad():
        state_cache.add_(0.25)
        state_indices.copy_(torch.tensor([6, 0, 2], device="cuda", dtype=torch.int32))
        packed_qkv.add_(0.1)
        raw_gate.mul_(0.9)
        raw_beta.add_(0.2)
    expected_storage = storage.clone()
    state_elements = state_cache[0].numel()
    expected_cache = expected_storage[:, 11 : 11 + state_elements].view_as(state_cache)
    expected = recurrent_kda_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        expected_cache,
        state_indices,
    )

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)
