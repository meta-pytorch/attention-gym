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
from attn_gym.testing import strided_state_pool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="recurrent_kda_decode requires CUDA"
)


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
    q = torch.randn(batch, heads, key_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(batch, heads, value_dim, device="cuda", dtype=dtype)
    packed_qkv = torch.cat((q.flatten(1), k.flatten(1), v.flatten(1)), dim=1)
    raw_gate = torch.randn(1, batch, heads, key_dim, device="cuda", dtype=dtype)
    raw_beta = torch.randn(1, batch, heads, device="cuda", dtype=dtype)
    A_log = 0.1 * torch.randn(heads, device="cuda", dtype=torch.float32)
    dt_bias = 0.1 * torch.randn(heads, key_dim, device="cuda", dtype=torch.float32)
    storage, state_cache = strided_state_pool(7, heads, key_dim, value_dim)
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
    gate_transform: str = "bounded",
    lower_bound: float = -5.0,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch = packed_qkv.shape[0]
    heads, value_dim, key_dim = state_cache.shape[1:]
    q_flat, k_flat, v_flat = packed_qkv.split(
        (heads * key_dim, heads * key_dim, heads * value_dim), dim=1
    )
    q = q_flat.view(batch, heads, key_dim).unsqueeze(1).float()
    k = k_flat.view(batch, heads, key_dim).unsqueeze(1).float()
    v = v_flat.view(batch, heads, value_dim).unsqueeze(1)
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)

    gate_input = raw_gate.float() + dt_bias[None, None]
    decay = A_log.exp()[None, None, :, None]
    if gate_transform == "bounded":
        gate = lower_bound * torch.sigmoid(decay * gate_input)
    else:
        assert gate_transform == "softplus"
        gate = -decay * F.softplus(gate_input)
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


@pytest.mark.parametrize(
    ("dtype", "gate_transform", "lower_bound", "key_dim", "value_dim"),
    [
        (torch.float16, "bounded", -5.0, 64, 64),
        (torch.float16, "softplus", 1.0, 80, 48),
        (torch.bfloat16, "bounded", -5.0, 80, 48),
        (torch.bfloat16, "softplus", float("nan"), 64, 64),
        (torch.float32, "bounded", 0.0, 64, 64),
        (torch.float32, "softplus", -5.0, 80, 48),
    ],
)
def test_recurrent_decode_matches_reference(
    dtype: torch.dtype,
    gate_transform: str,
    lower_bound: float,
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
        gate_transform=gate_transform,
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
        gate_transform,
        lower_bound,
    )

    tolerance = 1e-5 if dtype == torch.float32 else 3e-2
    assert output.shape == expected.shape and output.dtype == dtype
    torch.testing.assert_close(output.float(), expected.float(), rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(state_cache, expected_cache, rtol=tolerance, atol=tolerance)


def test_hopper_vector_gate_schedule_matches_reference():
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Hopper-specific decode schedule")
    inputs = _decode_inputs(batch=1, heads=16, key_dim=128, value_dim=128)
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
    )
    expected, expected_cache = _reference_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        before,
        state_indices,
    )

    torch.testing.assert_close(output.float(), expected.float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(state_cache, expected_cache, rtol=3e-2, atol=3e-2)


def test_decode_launcher_grouped_vector_gate_matches_expanded():
    """Grouped vector-gate decode shares only q/k; the gate stays per value head.

    The public KDA decode pins equal head counts, so this pins the shared launcher's
    multi-value attention addressing directly against explicit q/k expansion.
    """
    from attn_gym.linear._delta_rule.decode import (
        GateTransform,
        launch_recurrent_delta_rule_decode,
    )
    from attn_gym.linear._delta_rule.recurrent import GateKind

    torch.manual_seed(0)
    batch, key_heads, heads, dim = 3, 2, 6, 32
    groups = heads // key_heads
    q = torch.randn(batch, key_heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn(batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    # Independent per-value-head decays: grouping must not collapse state capacity.
    raw_gate = torch.randn(1, batch, heads, dim, device="cuda", dtype=torch.bfloat16)
    raw_beta = torch.randn(1, batch, heads, device="cuda", dtype=torch.bfloat16)
    A_log = 0.1 * torch.randn(heads, device="cuda")
    dt_bias = 0.1 * torch.randn(heads, dim, device="cuda")
    state_indices = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)

    def run(query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = torch.cat((query.flatten(1), key.flatten(1), v.flatten(1)), dim=1)
        pool = torch.randn(4, heads, dim, dim, device="cuda", dtype=torch.float32)
        torch.manual_seed(7)
        pool.normal_()
        output = packed.new_empty(batch, heads, dim)
        launch_recurrent_delta_rule_decode(
            packed,
            raw_gate[0],
            raw_beta[0],
            A_log,
            dt_bias,
            pool,
            state_indices,
            output,
            gate_kind=GateKind.VECTOR,
            gate_transform=GateTransform.SOFTPLUS,
            key_heads=query.shape[1],
            lower_bound=0.0,
            scale=dim**-0.5,
            has_initial_state=None,
            op_name="test",
        )
        return output, pool

    grouped_output, grouped_pool = run(q, k)
    expanded_output, expanded_pool = run(
        q.repeat_interleave(groups, dim=1), k.repeat_interleave(groups, dim=1)
    )

    torch.testing.assert_close(grouped_output, expanded_output, rtol=0, atol=0)
    torch.testing.assert_close(grouped_pool, expanded_pool, rtol=0, atol=0)


def test_recurrent_decode_fresh_slots_start_from_zero():
    """has_initial_state=False treats slot contents as garbage and overwrites them."""
    inputs = _decode_inputs()
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    has_initial_state = torch.tensor([True, False, True], device="cuda")
    zeroed = state_cache.clone()
    zeroed[state_indices[1].long()] = 0.0

    output = recurrent_kda_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        has_initial_state=has_initial_state,
    )
    expected, expected_cache = _reference_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        zeroed,
        state_indices,
    )

    torch.testing.assert_close(output.float(), expected.float(), rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(state_cache, expected_cache, rtol=1e-5, atol=1e-5)


def test_recurrent_decode_softplus_matches_negative_tail_reference():
    inputs = _decode_inputs(batch=1, heads=1, key_dim=16, value_dim=8, dtype=torch.float32)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    raw_gate.fill_(-18.0)
    raw_beta.fill_(-100.0)
    A_log.fill_(18.0)
    dt_bias.zero_()
    before = state_cache.clone()
    expected, expected_cache = _reference_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        before,
        state_indices,
        "softplus",
    )

    output = recurrent_kda_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        gate_transform="softplus",
    )

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_cache, expected_cache, rtol=1e-5, atol=1e-5)


def test_recurrent_decode_padding_slots_are_ignored():
    inputs = list(_decode_inputs())
    inputs[-1] = torch.tensor([5, 0, -1], device="cuda", dtype=torch.int32)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    before = state_cache.clone()

    output = recurrent_kda_decode(
        packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_cache, state_indices
    )

    torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]), rtol=0, atol=0)
    torch.testing.assert_close(state_cache[0], before[0], rtol=0, atol=0)


def test_recurrent_decode_key_dim_boundary():
    inputs = _decode_inputs(batch=1, heads=1, key_dim=256, value_dim=16, dtype=torch.float32)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    before = state_cache.clone()
    expected, expected_cache = _reference_decode(
        packed_qkv, raw_gate, raw_beta, A_log, dt_bias, before, state_indices
    )

    output = recurrent_kda_decode(
        packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_cache, state_indices
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state_cache, expected_cache, rtol=1e-5, atol=1e-5)

    too_large = _decode_inputs(batch=1, heads=1, key_dim=257, value_dim=16)
    with pytest.raises(ValueError, match=r"K in \[1, 256\]"):
        recurrent_kda_decode(
            too_large[0],
            too_large[1],
            too_large[2],
            too_large[3],
            too_large[4],
            too_large[6],
            too_large[7],
        )


def test_recurrent_decode_validates_contract():
    inputs = list(_decode_inputs(batch=1, heads=1, key_dim=16, value_dim=8))
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
    with pytest.raises(ValueError, match="finite and nonpositive"):
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
    for name, invalid_A_log, invalid_dt_bias in (
        ("A_log", A_log.bfloat16(), dt_bias),
        ("dt_bias", A_log, dt_bias.bfloat16()),
    ):
        with pytest.raises(ValueError, match=rf"{name} must be contiguous float32"):
            recurrent_kda_decode(
                packed_qkv,
                raw_gate,
                raw_beta,
                invalid_A_log,
                invalid_dt_bias,
                state_cache,
                state_indices,
            )
    with pytest.raises(ValueError, match="gate_transform must be one of"):
        recurrent_kda_decode(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
            gate_transform="unknown",
        )
    with pytest.raises(ValueError, match="out must have shape"):
        recurrent_kda_decode(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
            out=packed_qkv.new_empty(1, 1, state_cache.shape[1], state_cache.shape[2] - 1),
        )
    with pytest.raises(TypeError, match="out must use packed_qkv.dtype"):
        recurrent_kda_decode(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
            out=torch.empty(
                1,
                1,
                state_cache.shape[1],
                state_cache.shape[2],
                device="cuda",
                dtype=torch.float32,
            ),
        )


def test_recurrent_decode_rejects_gradient_tracking():
    inputs = list(_decode_inputs(batch=1, heads=1, key_dim=16, value_dim=8))
    inputs[0] = inputs[0].requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        recurrent_kda_decode(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[6], inputs[7]
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_recurrent_decode_rejects_aliasing_out(compiled: bool):
    inputs = _decode_inputs(batch=2, heads=2, key_dim=16, value_dim=8, dtype=torch.float32)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    out = packed_qkv.flatten()[: 2 * 2 * 8].view(1, 2, 2, 8)
    function = (
        torch.compile(recurrent_kda_decode, fullgraph=True) if compiled else recurrent_kda_decode
    )

    with pytest.raises(ValueError, match="out must not alias"):
        function(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
            out=out,
        )

    if compiled:
        # Inductor rejects dtype-view input mutations before our alias check can run.
        return
    fresh_out = packed_qkv.new_empty(1, 2, 2, 8)
    has_initial_state = fresh_out.view(torch.bool).flatten()[:2]
    with pytest.raises(ValueError, match="out must not alias"):
        function(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
            has_initial_state=has_initial_state,
            out=fresh_out,
        )


def test_recurrent_decode_rejects_state_aliasing_read_only_input():
    batch, heads, key_dim, value_dim, slots = 1, 1, 16, 8, 7
    state_elements = heads * value_dim * key_dim
    storage = torch.randn(slots * state_elements, device="cuda", dtype=torch.float32)
    state_cache = storage.view(slots, heads, value_dim, key_dim)
    channels = heads * (2 * key_dim + value_dim)
    packed_qkv = storage[5 * state_elements : 5 * state_elements + channels].view(batch, channels)
    raw_gate = torch.randn(1, batch, heads, key_dim, device="cuda")
    raw_beta = torch.randn(1, batch, heads, device="cuda")
    A_log = torch.randn(heads, device="cuda")
    dt_bias = torch.randn(heads, key_dim, device="cuda")
    state_indices = torch.tensor([5], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="state_cache must not alias"):
        recurrent_kda_decode(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
        )


def test_recurrent_decode_custom_op_registration():
    inputs = _decode_inputs(batch=1, heads=1, key_dim=16, value_dim=8)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
    out = packed_qkv.new_empty(1, 1, state_cache.shape[1], state_cache.shape[2])
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
            None,
            out,
            -5.0,
            True,
            state_cache.shape[3] ** -0.5,
        ),
    )


@pytest.mark.parametrize(("gate_transform", "use_out"), [("bounded", False), ("softplus", True)])
def test_recurrent_decode_fullgraph_compile(gate_transform: str, use_out: bool):
    inputs = _decode_inputs(batch=1, heads=1, key_dim=16, value_dim=8)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, eager_cache, state_indices = inputs
    lower_bound = 0.0 if gate_transform == "bounded" else float("nan")
    _, compiled_cache = strided_state_pool(
        eager_cache.shape[0], eager_cache.shape[1], eager_cache.shape[3], eager_cache.shape[2]
    )
    compiled_cache.copy_(eager_cache)
    output_shape = (1, 1, eager_cache.shape[1], eager_cache.shape[2])
    eager_out = packed_qkv.new_full(output_shape, torch.nan) if use_out else None
    compiled_out = torch.full_like(eager_out, torch.nan) if eager_out is not None else None
    # Exercises the optional-tensor plumbing and USE_HAS_INITIAL_STATE under compilation.
    has_initial_state = torch.ones(1, device="cuda", dtype=torch.bool)

    expected = recurrent_kda_decode(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        eager_cache,
        state_indices,
        has_initial_state=has_initial_state,
        gate_transform=gate_transform,
        lower_bound=lower_bound,
        out=eager_out,
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
        has_initial_state=has_initial_state,
        gate_transform=gate_transform,
        lower_bound=lower_bound,
        out=compiled_out,
    )

    if compiled_out is not None:
        assert expected is eager_out
        assert output is compiled_out
        assert not torch.isnan(output).any()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled_cache, eager_cache, rtol=0, atol=0)


def test_recurrent_decode_fullgraph_dynamic_batch():
    torch.compiler.reset()
    with torch._dynamo.config.patch(error_on_recompile=True):
        compiled = torch.compile(recurrent_kda_decode, fullgraph=True, dynamic=True)
        for batch in (2, 3):
            inputs = _decode_inputs(
                batch=batch,
                heads=4,
                key_dim=16,
                value_dim=8,
                dtype=torch.float32,
            )
            packed_qkv, raw_gate, raw_beta, A_log, dt_bias, _, state_cache, state_indices = inputs
            eager_cache = state_cache.clone()
            compiled_cache = state_cache.clone()
            expected = recurrent_kda_decode(
                packed_qkv,
                raw_gate,
                raw_beta,
                A_log,
                dt_bias,
                eager_cache,
                state_indices,
                scale=0.25,
            )
            actual = compiled(
                packed_qkv,
                raw_gate,
                raw_beta,
                A_log,
                dt_bias,
                compiled_cache,
                state_indices,
                scale=0.25,
            )

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(compiled_cache, eager_cache, rtol=0, atol=0)


def test_recurrent_decode_cuda_graph_replay():
    inputs = _decode_inputs(batch=3, seed=4)
    packed_qkv, raw_gate, raw_beta, A_log, dt_bias, storage, state_cache, state_indices = inputs
    out = packed_qkv.new_empty(1, 3, state_cache.shape[1], state_cache.shape[2])
    recurrent_kda_decode(
        packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_cache, state_indices, out=out
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = recurrent_kda_decode(
            packed_qkv,
            raw_gate,
            raw_beta,
            A_log,
            dt_bias,
            state_cache,
            state_indices,
            out=out,
        )
    assert captured_output is out

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
