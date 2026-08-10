"""Tests for the trainable single-device KDA module example."""

from __future__ import annotations

import copy

import pytest
import torch

from examples.kda_training import KDAAttention


def test_reference_kda_module_trains_on_cpu():
    torch.manual_seed(0)
    model = KDAAttention(
        hidden_size=16,
        num_heads=2,
        head_dim=4,
        chunk_size=4,
    )
    hidden_states = torch.randn(2, 7, 16, requires_grad=True)
    output = model(hidden_states, return_final_state=True)

    assert output.hidden_states.shape == hidden_states.shape
    assert output.final_state is not None
    assert output.final_state.shape == (2, 2, 4, 4)

    loss = output.hidden_states.square().mean() + output.final_state.square().mean()
    loss.backward()
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_low_precision_module_keeps_fp32_recurrent_state():
    model = KDAAttention(16, 2, 4, compute_dtype=torch.bfloat16)
    hidden_states = torch.randn(1, 5, 16, dtype=torch.bfloat16, requires_grad=True)
    output = model(hidden_states, return_final_state=True)

    assert output.hidden_states.dtype == torch.bfloat16
    assert output.final_state is not None and output.final_state.dtype == torch.float32
    (output.hidden_states.float().square().mean() + output.final_state.square().mean()).backward()
    assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()


def test_default_gate_range_and_state_carry_remain_finite():
    torch.manual_seed(2)
    model = KDAAttention(hidden_size=8, num_heads=1, head_dim=4, chunk_size=64)
    hidden_states = torch.randn(1, 65, 8)

    full = model(hidden_states, return_final_state=True)
    first = model(hidden_states[:, :33], return_final_state=True)
    second = model(
        hidden_states[:, 33:],
        initial_state=first.final_state,
        return_final_state=True,
    )

    assert full.final_state is not None and second.final_state is not None
    assert torch.isfinite(full.hidden_states).all()
    assert torch.isfinite(full.final_state).all()
    torch.testing.assert_close(
        torch.cat((first.hidden_states, second.hidden_states), dim=1),
        full.hidden_states,
        rtol=2e-4,
        atol=2e-5,
    )
    torch.testing.assert_close(second.final_state, full.final_state, rtol=2e-4, atol=2e-5)

    (full.hidden_states.square().mean() + full.final_state.square().mean()).backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="the fused CuTeDSL gate backward requires CUDA capability 9.0 or newer",
)
def test_cute_gate_backward_matches_module_autograd(tmp_path, monkeypatch):
    pytest.importorskip("cutlass")
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path / "cache"))
    torch.manual_seed(1)
    reference = KDAAttention(
        hidden_size=64,
        num_heads=2,
        head_dim=32,
        chunk_size=7,
        gate_backward="torch",
        compute_dtype=torch.bfloat16,
        device="cuda",
    )
    cute = copy.deepcopy(reference)
    cute.gate_backward = "cute"
    reference_input = torch.randn(2, 9, 64, device="cuda", requires_grad=True)
    cute_input = reference_input.detach().clone().requires_grad_()

    def forward_and_backward(model, hidden_states):
        output = model(hidden_states, return_final_state=True)
        assert output.final_state is not None
        loss = output.hidden_states.float().square().mean()
        loss = loss + output.final_state.float().square().mean()
        loss.backward()
        gradients = {name: parameter.grad for name, parameter in model.named_parameters()}
        return output, gradients

    expected, expected_gradients = forward_and_backward(reference, reference_input)
    actual, actual_gradients = forward_and_backward(cute, cute_input)

    torch.testing.assert_close(actual.hidden_states, expected.hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(actual.final_state, expected.final_state, rtol=0, atol=0)
    torch.testing.assert_close(cute_input.grad, reference_input.grad, rtol=3e-3, atol=3e-3)
    for name in expected_gradients:
        torch.testing.assert_close(
            actual_gradients[name],
            expected_gradients[name],
            rtol=4e-3,
            atol=4e-3,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_cute_gate_backward_rejects_unsupported_specializations():
    with pytest.raises(ValueError, match="head_dim divisible by 32"):
        KDAAttention(16, 2, 8, gate_backward="cute")
    with pytest.raises(ValueError, match="compute_dtype=torch.bfloat16"):
        KDAAttention(32, 1, 32, gate_backward="cute", compute_dtype=torch.float32)
