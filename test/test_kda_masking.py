"""Tests for fixed-capacity packed KDA masking helpers."""

import pytest
import torch

from attn_gym.linear.kda import (
    active_token_mask,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fixed-capacity KDA masking requires CUDA",
)


def test_mask_inactive_tokens_fullgraph_autograd():
    values = torch.arange(8, device="cuda", dtype=torch.float32).reshape(1, 4, 2)
    values[:, 2:] = float("nan")
    values.requires_grad_()
    cu_seqlens = torch.tensor([0, 2], device="cuda", dtype=torch.int32)

    def operation(x, offsets):
        return mask_inactive_tokens(x, active_token_mask(x, offsets))

    actual = torch.compile(operation, fullgraph=True)(values, cu_seqlens)
    (gradient,) = torch.autograd.grad(actual.sum(), values)

    expected = torch.tensor([[[0, 1], [2, 3], [0, 0], [0, 0]]], device="cuda", dtype=torch.float32)
    expected_gradient = torch.tensor(
        [[[1, 1], [1, 1], [0, 0], [0, 0]]], device="cuda", dtype=torch.float32
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(gradient, expected_gradient)


def test_mask_inactive_tokens_cuda_graph_replays_endpoint_down_then_up():
    values = torch.arange(10, device="cuda", dtype=torch.float32).reshape(1, 5, 2)
    cu_seqlens = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    active_mask = active_token_mask(values, cu_seqlens)
    mask_inactive_tokens(values, active_mask)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        active_mask = active_token_mask(values, cu_seqlens)
        actual = mask_inactive_tokens(values, active_mask)

    for endpoint in (2, 3):
        cu_seqlens.copy_(torch.tensor([0, endpoint], device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()

        expected = values.clone()
        expected[:, endpoint:] = 0
        torch.testing.assert_close(actual, expected)


def test_reusable_active_token_mask_sanitizes_values_and_derivatives():
    values = torch.arange(10, device="cuda", dtype=torch.float32).reshape(1, 5, 2)
    values[:, 3:] = float("nan")
    values.requires_grad_()
    incoming = torch.arange(10, device="cuda", dtype=torch.float32).reshape_as(values)
    incoming[:, 3:] = float("nan")
    cu_seqlens = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    active_mask = active_token_mask(values, cu_seqlens)

    sanitized = mask_inactive_tokens(values, active_mask)
    protected = mask_inactive_token_gradients(values, active_mask)
    (gradient,) = torch.autograd.grad(protected, values, incoming)
    _, tangent = torch.func.jvp(
        lambda x: mask_inactive_token_gradients(x, active_mask),
        (values.detach(),),
        (torch.ones_like(values),),
    )

    torch.testing.assert_close(protected, values, equal_nan=True)
    torch.testing.assert_close(sanitized[:, :3], values[:, :3])
    assert not sanitized[:, 3:].any()
    torch.testing.assert_close(gradient[:, :3], incoming[:, :3])
    assert not gradient[:, 3:].any()
    assert tangent[:, :3].all()
    assert not tangent[:, 3:].any()


def test_inactive_token_gradient_barrier_supports_functional_fullgraph():
    values = torch.randn(1, 5, 3, device="cuda")
    incoming = torch.randn_like(values)
    incoming[:, 2:] = float("nan")
    cu_seqlens = torch.tensor([0, 2], device="cuda", dtype=torch.int32)

    def loss(x, offsets, cotangent):
        active_mask = active_token_mask(x, offsets)
        protected = mask_inactive_token_gradients(x, active_mask)
        return (protected * cotangent).sum()

    actual = torch.compile(torch.func.grad(loss), fullgraph=True)(values, cu_seqlens, incoming)

    torch.testing.assert_close(actual[:, :2], incoming[:, :2])
    assert not actual[:, 2:].any()


def test_inactive_token_gradient_barrier_replays_endpoint_boundaries():
    values = torch.randn(1, 5, 2, device="cuda", requires_grad=True)
    incoming = torch.randn_like(values)
    baseline_incoming = incoming.clone()
    cu_seqlens = torch.tensor([0, 4], device="cuda", dtype=torch.int32)

    def run():
        active_mask = active_token_mask(values, cu_seqlens)
        protected = mask_inactive_token_gradients(values, active_mask)
        return torch.autograd.grad(protected, values, incoming)[0]

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gradient = run()

    for endpoint in (0, 2, 5):
        cu_seqlens.copy_(torch.tensor([0, endpoint], device="cuda", dtype=torch.int32))
        incoming.copy_(baseline_incoming)
        incoming[:, endpoint:].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(gradient[:, :endpoint], incoming[:, :endpoint])
        assert not gradient[:, endpoint:].any()


def test_active_token_mask_validates_packed_metadata():
    values = torch.ones((1, 4, 2), device="cuda")
    cu_seqlens = torch.tensor([0, 4], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match=r"packed shape \[1, T, \.\.\.\]"):
        active_token_mask(values.expand(2, -1, -1), cu_seqlens)
    with pytest.raises(ValueError, match=r"shape \[num_sequences \+ 1\]"):
        active_token_mask(values, cu_seqlens[:1])
    with pytest.raises(ValueError, match="dtype torch.int32"):
        active_token_mask(values, cu_seqlens.to(torch.int64))
    with pytest.raises(ValueError, match="same device"):
        active_token_mask(values, cu_seqlens.cpu())
    with pytest.raises(ValueError, match="contiguous"):
        active_token_mask(
            values,
            torch.tensor([0, 0, 4, 4], device="cuda", dtype=torch.int32)[::2],
        )


def test_reusable_mask_shape_and_noop():
    values = torch.ones((1, 4, 2), device="cuda")

    active_mask = torch.ones(4, device="cuda", dtype=torch.bool)
    for operation in (mask_inactive_tokens, mask_inactive_token_gradients):
        assert operation(values, None) is values
        with pytest.raises(ValueError, match=r"packed shape \[1, T, \.\.\.\]"):
            operation(values.expand(2, -1, -1), active_mask)
        with pytest.raises(ValueError, match=r"shape \[4\]"):
            operation(values, active_mask[:1])
