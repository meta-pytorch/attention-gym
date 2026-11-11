"""Pinned gather_attn launches: no timing-picked kernel configurations (issue #777)."""

import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, gather_attn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# (heads, kv_heads, head_dim, topk, window): the TMA, shared and grouped Triton schedules.
SCHEDULES = {
    "tma": (16, 1, 256, 16, 16),
    "shared": (16, 1, 64, 32, 128),
    "grouped": (16, 4, 128, 32, 128),
}


def _forward_and_backward(shape, kernel_options) -> list[torch.Tensor]:
    heads, kv_heads, head_dim, topk, window = shape
    tokens, ratio = 256, 4
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*size):
        return torch.randn(*size, device="cuda", dtype=torch.bfloat16, generator=generator)

    leaves = [
        randn(1, heads, tokens, head_dim).requires_grad_(),
        randn(1, kv_heads, tokens, head_dim).requires_grad_(),
        randn(1, kv_heads, tokens // ratio, head_dim).requires_grad_(),
    ]
    indices = torch.randint(
        0, tokens // ratio, (1, tokens, topk), device="cuda", generator=generator
    )
    visible = (torch.arange(tokens, device="cuda") // ratio)[None, :, None]
    indices = torch.where(indices < visible, indices, -1).int()
    output, aux = gather_attn(
        *leaves,
        indices,
        attention_sink=torch.zeros(heads, device="cuda"),
        sliding_window_size=window,
        return_aux=AuxRequest(lse=True),
        kernel_options=kernel_options,
    )
    output.backward(randn(*output.shape))
    return [output.detach(), aux.lse, *(leaf.grad for leaf in leaves)]


@pytest.mark.parametrize("shape", SCHEDULES.values(), ids=SCHEDULES)
def test_autotune_false_and_deterministic_mode_launch_the_same_pinned_forward(
    shape, no_benchmarking, request
):
    """Both modes skip benchmarking in forward and backward; forward output and LSE match bits."""
    output, lse, *_ = _forward_and_backward(shape, {"backend": "triton", "autotune": False})
    request.getfixturevalue("deterministic_algorithms")
    # Deterministic mode also switches backward schedules, so only the forward is comparable.
    deterministic_output, deterministic_lse, *_ = _forward_and_backward(shape, None)
    assert torch.equal(deterministic_output, output)
    assert torch.equal(deterministic_lse, lse)


def test_autotune_option_must_be_a_bool():
    query = torch.randn(1, 2, 8, 16)
    with pytest.raises(ValueError, match="unsupported gather_attn kernel options"):
        gather_attn(
            query,
            query,
            query,
            torch.zeros(1, 8, 2, dtype=torch.int32),
            kernel_options={"autotune": 1},
        )
