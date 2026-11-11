"""``chunk_gdn`` launches fixed kernel configurations when it may not autotune."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear import TuningOptions, chunk_gdn, paged_chunk_gdn
from attn_gym.testing import make_gdn_test_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TOKENS = 192


def chunk_outputs_and_gradients(
    head_dim: int, packed: bool, kernel_options: TuningOptions | None = None
) -> tuple[torch.Tensor, ...]:
    """Return the output, final state, and input gradients of one fused training step."""
    *inputs, cu_seqlens = make_gdn_test_inputs(
        (70, TOKENS - 70) if packed else TOKENS,
        key_dim=head_dim,
        sigmoid_beta=True,
        requires_grad=True,
    )
    output, final_state = chunk_gdn(
        *inputs, cu_seqlens=cu_seqlens, output_final_state=True, kernel_options=kernel_options
    )
    loss = output.float().square().mean() + final_state.square().mean()
    return output, final_state, *torch.autograd.grad(loss, inputs)


def assert_matches_tuned(actual: tuple[torch.Tensor, ...], tuned: tuple[torch.Tensor, ...]):
    """Configurations differ only in floating-point order, so results agree to rounding."""
    assert len(actual) == len(tuned)
    for index, (pinned, expected) in enumerate(zip(actual, tuned)):
        error = (pinned.float() - expected.float()).norm() / expected.float().norm()
        assert error < 2e-2, f"result {index} differs from the autotuned call by {error:.3g}"


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_chunk_gdn_without_autotune_never_benchmarks(
    head_dim: int, packed: bool, isolated_winners: None, request: pytest.FixtureRequest
):
    tuned = chunk_outputs_and_gradients(head_dim, packed)
    request.getfixturevalue("no_benchmarking")
    pinned = chunk_outputs_and_gradients(head_dim, packed, {"autotune": False})
    assert_matches_tuned(pinned, tuned)
    repeated = chunk_outputs_and_gradients(head_dim, packed, {"autotune": False})
    assert all(torch.equal(first, second) for first, second in zip(pinned, repeated))


@pytest.mark.parametrize("head_dim", [64, 128])
def test_deterministic_mode_gives_the_bits_of_autotune_false(
    head_dim: int, no_benchmarking: None, request: pytest.FixtureRequest
):
    pinned = chunk_outputs_and_gradients(head_dim, True, {"autotune": False})
    request.getfixturevalue("deterministic_algorithms")
    actual = chunk_outputs_and_gradients(head_dim, packed=True)
    assert all(torch.equal(result, expected) for result, expected in zip(actual, pinned))


def test_paged_chunk_gdn_without_autotune_never_benchmarks(no_benchmarking: None):
    *inputs, state, _cu_seqlens = make_gdn_test_inputs(TOKENS, sigmoid_beta=True)
    state_cache = torch.cat([torch.zeros_like(state), state])
    state_indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    expected = chunk_gdn(
        *inputs, state, output_final_state=True, kernel_options={"autotune": False}
    )
    with torch.no_grad():
        output = paged_chunk_gdn(
            *inputs, state_cache, state_indices, kernel_options={"autotune": False}
        )
    assert_matches_tuned((output, state_cache[1:]), expected)
