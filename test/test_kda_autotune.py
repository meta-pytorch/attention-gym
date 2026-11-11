"""``chunk_kda`` launches pinned kernel configurations under deterministic algorithms."""

import pytest
import torch

from attn_gym.linear import KernelOptions, chunk_kda
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _forward(
    kernel_options: KernelOptions | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    inputs = make_kda_test_inputs(192, heads=2, sigmoid_beta=True, requires_grad=True)
    output, _ = chunk_kda(*inputs, kernel_options=kernel_options)
    return output, inputs


def _gradients(output: torch.Tensor, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return torch.autograd.grad(output.float().square().mean(), inputs)


def test_deterministic_mode_gives_the_bits_of_autotune_false(no_benchmarking, request):
    pinned_output, pinned_inputs = _forward({"autotune": False})
    pinned_gradients = _gradients(pinned_output, pinned_inputs)
    request.getfixturevalue("deterministic_algorithms")
    output, inputs = _forward()
    assert torch.equal(output, pinned_output)
    for gradient, pinned in zip(_gradients(output, inputs), pinned_gradients, strict=True):
        assert torch.equal(gradient, pinned)


def test_backward_follows_deterministic_mode_enabled_after_the_forward(isolated_winners, request):
    output, inputs = _forward()
    request.getfixturevalue("no_benchmarking")
    request.getfixturevalue("deterministic_algorithms")
    assert all(torch.isfinite(gradient).all() for gradient in _gradients(output, inputs))
