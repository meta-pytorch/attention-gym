"""Triton gather attention on GPUs with 99 KiB of shared memory per block (SM86/SM89).

A10G CI failed with ``OutOfResources: shared memory, Required: 131072, Hardware limit: 101376``
because the head-parallel kernels used fixed tiles sized for Hopper/Blackwell. These tests run on
any CUDA GPU: one compiles every launch for SM86 without running it, the other runs the
small-budget tiles on the local GPU against the eager reference.
"""

import pytest
import torch

from attn_gym.sparse.gather_attn import Impl, gather_attn
from attn_gym.testing.triton_budget import compile_only_for_target, emulate_device_dispatch

SM86_CAPABILITY = (8, 6)
SM86_MAX_SHARED_MEMORY = 101376


def make_inputs(
    heads: int,
    head_dim: int,
    topk: int,
    dtype: torch.dtype = torch.bfloat16,
    seq_len: int = 256,
    compress_ratio: int = 4,
) -> dict[str, torch.Tensor]:
    """Build DeepSeek-V4-style shared-KV inputs: one KV head, FP32 sink, causal top-k."""
    generator = torch.Generator(device="cuda").manual_seed(2026)
    sparse_seq_len = seq_len // compress_ratio

    def randn(*shape: int, dtype: torch.dtype = dtype) -> torch.Tensor:
        return torch.randn(
            *shape, device="cuda", dtype=dtype, generator=generator, requires_grad=True
        )

    positions = torch.arange(seq_len, device="cuda")[:, None]
    visible = torch.clamp((positions + 1) // compress_ratio, min=1)
    offsets = torch.randint(sparse_seq_len, (seq_len, topk), device="cuda", generator=generator)
    kv_indices = torch.where(offsets < visible, offsets, offsets % visible)
    return {
        "query": randn(1, heads, seq_len, head_dim),
        "local_kv": randn(1, 1, seq_len, head_dim),
        "sparse_kv": randn(1, 1, sparse_seq_len, head_dim),
        "kv_indices": kv_indices[None].to(torch.int32),
        "attention_sink": randn(heads, dtype=torch.float32),
    }


# (heads, head_dim, window, topk): torchtitan's DeepSeek-V4 debug model (CSA/HCA top-k and SWA
# with an empty top-k), a pipelined local window, the flash model's D=512 (top-k reduced to
# keep the fully unrolled top-k loop cheap to compile), and smaller heads.
BUDGET_CASES = [
    (16, 256, 16, 16),
    (16, 256, 16, 0),
    (8, 256, 128, 16),
    (32, 512, 128, 16),
    (16, 128, 128, 64),
    (16, 64, 128, 64),
]


@pytest.mark.parametrize("heads,head_dim,window,topk", BUDGET_CASES)
def test_sm86_launches_fit_shared_memory(heads, head_dim, window, topk):
    """Every launch has a config within SM86's 101376-byte per-block limit."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton")
    inputs = make_inputs(heads, head_dim, topk)
    with compile_only_for_target(SM86_CAPABILITY, SM86_MAX_SHARED_MEMORY) as launches:
        output = gather_attn(
            **inputs, sliding_window_size=window, kernel_options={"backend": "triton"}
        )
        output.sum().backward()

    kernels = {launch.kernel for launch in launches}
    assert {"_gather_attn_fwd", "_gather_attn_bwd_dq", "_gather_attn_bwd_dlocal_kv"} <= kernels
    assert ("_gather_attn_bwd_dsparse_kv" in kernels) == (topk > 0)
    too_large = {
        launch.kernel: min(launch.shared)
        for launch in launches
        if min(launch.shared) > SM86_MAX_SHARED_MEMORY
    }
    assert not too_large


@pytest.mark.parametrize(
    "head_dim,window,dtype",
    [(256, 64, torch.bfloat16), (128, 64, torch.bfloat16), (256, 64, torch.float32)],
)
def test_sm86_tiles_match_reference(head_dim, window, dtype):
    """The tiles selected for a 99 KiB budget compute the same forward and gradients."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton")
    inputs = make_inputs(4, head_dim, topk=8, dtype=dtype)
    reference_inputs = {
        name: value.detach().double().requires_grad_() if value.is_floating_point() else value
        for name, value in inputs.items()
    }
    expected = gather_attn(**reference_inputs, sliding_window_size=window, impl=Impl.REFERENCE)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)

    # Take the SM86 host path (no TMA, no Blackwell schedule) on the local GPU.
    with emulate_device_dispatch(SM86_CAPABILITY, SM86_MAX_SHARED_MEMORY):
        output = gather_attn(
            **inputs, sliding_window_size=window, kernel_options={"backend": "triton"}
        )
        output.backward(grad_output.to(output.dtype))

    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(output.double(), expected, atol=tolerance, rtol=tolerance)
    for name in ("query", "local_kv", "sparse_kv", "attention_sink"):
        torch.testing.assert_close(
            inputs[name].grad.double(),
            reference_inputs[name].grad,
            atol=tolerance * reference_inputs[name].grad.abs().max().item(),
            rtol=tolerance,
            msg=lambda message, name=name: f"{name} gradient: {message}",
        )
