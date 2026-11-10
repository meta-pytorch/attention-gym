"""Triton gather attention within per-GPU on-chip memory: SM86 shared memory, Blackwell TMEM.

A10G CI failed with ``OutOfResources: shared memory, Required: 131072, Hardware limit: 101376``
because the head-parallel kernels used fixed tiles sized for Hopper/Blackwell. The SM86 tests run
on any CUDA GPU: one compiles every launch for SM86 without running it, the other runs the
small-budget tiles on the local GPU against the eager reference.

On Blackwell, FP32 D=256 selected a forward tile that fit shared memory but not tensor memory
(``Required: 576, Hardware limit: 512``). The SM100 compile-only test checks both budgets for
every launch; the Blackwell reference test runs every production tuning candidate.
"""

import pytest
import torch

from attn_gym.sparse.gather_attn import Impl, gather_attn
from attn_gym.testing.gather_attn import as_fp64, assert_numerics, skip_unsupported
from attn_gym.testing.gather_attn import make_inputs as make_mla_inputs
from attn_gym.testing.triton_budget import (
    Launch,
    compile_only_for_target,
    emulate_device_dispatch,
)

SM86_CAPABILITY = (8, 6)
SM86_MAX_SHARED_MEMORY = 101376
# Consumer Blackwell has SM86's per-block limit, not datacenter Blackwell's.
SM120_CAPABILITY = (12, 0)
SM100_CAPABILITY = (10, 0)
SM100_MAX_SHARED_MEMORY = 232448
SM100_TENSOR_MEMORY_COLUMNS = 512


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


def over_shared_memory(launches: list[Launch], limit: int) -> dict[str, int]:
    """Map kernels whose configs all exceed ``limit`` to their minimum shared-memory bytes."""
    return {launch.kernel: min(launch.shared) for launch in launches if min(launch.shared) > limit}


# (heads, head_dim, window, topk, forward kernel): torchtitan's DeepSeek-V4 debug model (CSA/HCA
# top-k and SWA with an empty top-k), a pipelined local window, the flash model's D=512, and
# smaller heads. BF16 with 16+ heads takes the shared-KV schedule unless D=256; fewer heads
# keep the per-head kernels.
BUDGET_CASES = [
    (16, 256, 16, 16, "_gather_attn_fwd"),
    (16, 256, 16, 0, "_gather_attn_fwd"),
    (8, 256, 128, 16, "_gather_attn_fwd"),
    (32, 512, 128, 16, "_gather_attn_fwd_shared"),
    (16, 128, 128, 64, "_gather_attn_fwd_shared"),
    (16, 64, 128, 64, "_gather_attn_fwd_shared"),
    (8, 512, 128, 16, "_gather_attn_fwd"),
    (8, 128, 128, 64, "_gather_attn_fwd"),
    (8, 64, 128, 64, "_gather_attn_fwd"),
]


@pytest.mark.parametrize("heads,head_dim,window,topk,forward_kernel", BUDGET_CASES)
def test_sm86_launches_fit_shared_memory(heads, head_dim, window, topk, forward_kernel):
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
    assert forward_kernel in kernels
    for stage in ("bwd_dq", "bwd_dlocal_kv"):
        assert any(kernel.startswith(f"_gather_attn_{stage}") for kernel in kernels), stage
    assert any("bwd_dsparse_kv" in kernel for kernel in kernels) == (topk > 0)
    assert not over_shared_memory(launches, SM86_MAX_SHARED_MEMORY)


@pytest.mark.parametrize(
    "heads,latent_dim,rope_dim",
    [(96, 512, 0), (96, 512, 64), (96, 512, 128), (128, 128, 128)],
    ids=["mla-bare", "mla-paired", "mla-wide-rope", "widest-head-tile"],
)
@pytest.mark.parametrize("capability", [SM86_CAPABILITY, SM120_CAPABILITY], ids=["sm86", "sm120"])
def test_shared_kv_launches_fit_99_kib_shared_memory(capability, heads, latent_dim, rope_dim):
    """Each shared-KV forward/backward launch has a config within 99 KiB of shared memory.

    The cases cover absorbed MLA (96 heads, latent 512, RoPE 64), the widest rows, and the
    largest head tile. Previously, datacenter-sized key tiles needed 163840 bytes on SM120,
    and other GPUs rejected RoPE pairs.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton")
    inputs = make_mla_inputs(
        heads=heads,
        latent_dim=latent_dim,
        rope_dim=rope_dim,
        seq_len=64,
        topk=64,
        with_docs=False,
    )
    with compile_only_for_target(capability, SM86_MAX_SHARED_MEMORY) as launches:
        output = gather_attn(
            **inputs.kwargs(),
            sliding_window_size=32,
            scale=0.0625,
            kernel_options={"backend": "triton"},
        )
        output.sum().backward()

    assert "_gather_attn_fwd_shared" in {launch.kernel for launch in launches}
    assert not over_shared_memory(launches, SM86_MAX_SHARED_MEMORY)


@pytest.mark.parametrize("rope_dim", [0, 64], ids=["bare", "paired"])
def test_absorbed_mla_small_key_tiles_match_reference(rope_dim):
    """Key tiles selected for a 99 KiB budget match reference outputs and gradients."""
    skip_unsupported("triton")
    # 40 selections end in a partial tile for both the 16- and 32-key tiles.
    inputs = make_mla_inputs(heads=96, rope_dim=rope_dim, topk=40)
    reference_inputs = as_fp64(inputs)
    options = {"sliding_window_size": 7, "scale": 0.0625}
    expected = gather_attn(**reference_inputs.kwargs(), **options, impl=Impl.REFERENCE)
    grad_output = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, reference_inputs.differentiable(), grad_output)

    with emulate_device_dispatch(SM86_CAPABILITY, SM86_MAX_SHARED_MEMORY):
        output = gather_attn(**inputs.kwargs(), **options, kernel_options={"backend": "triton"})
        grads = torch.autograd.grad(output, inputs.differentiable(), grad_output.to(output.dtype))

    assert_numerics(output, expected, "output")
    for index, (grad, expected_grad) in enumerate(zip(grads, expected_grads, strict=True)):
        assert_numerics(grad, expected_grad, f"gradient {index}")


@pytest.mark.parametrize("head_dim", [384, 512])
@pytest.mark.parametrize("capability", [SM86_CAPABILITY, SM120_CAPABILITY], ids=["sm86", "sm120"])
def test_fp32_wide_head_launches_fit_99_kib_shared_memory(capability, head_dim):
    """Every launch has a config within the 101376-byte per-block shared-memory limit.

    Compile-only for FP32 head_dim 384/512 with a local window.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton")
    inputs = make_inputs(4, head_dim, topk=16, dtype=torch.float32)
    with compile_only_for_target(capability, SM86_MAX_SHARED_MEMORY) as launches:
        output = gather_attn(
            **inputs, sliding_window_size=64, kernel_options={"backend": "triton"}
        )
        output.sum().backward()

    assert {"_gather_attn_bwd_dq", "_gather_attn_bwd_dlocal_kv"} <= {
        launch.kernel for launch in launches
    }
    assert not over_shared_memory(launches, SM86_MAX_SHARED_MEMORY)


@pytest.mark.parametrize(
    "head_dim,window,dtype",
    [
        (256, 64, torch.bfloat16),
        (128, 64, torch.bfloat16),
        (256, 64, torch.float32),
        (512, 64, torch.float32),
    ],
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

    # Take the SM86 host path (no TMA; four heads keep the per-head kernels) on the local GPU.
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


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_sm100_launches_fit_shared_and_tensor_memory(dtype, head_dim):
    """Every generic-schedule launch has a config within SM100 shared and tensor memory."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for Triton")
    # Four query heads take the generic (TMA when D <= 256) schedule, not the shared-KV one.
    inputs = make_inputs(4, head_dim, topk=8, dtype=dtype)
    with compile_only_for_target(SM100_CAPABILITY, SM100_MAX_SHARED_MEMORY) as launches:
        output = gather_attn(
            **inputs, sliding_window_size=64, kernel_options={"backend": "triton"}
        )
        output.sum().backward()

    kernels = {launch.kernel for launch in launches}
    assert {"_gather_attn_bwd_dq", "_gather_attn_bwd_dsparse_kv"} <= kernels
    too_large = {
        launch.kernel: list(zip(launch.shared, launch.tmem))
        for launch in launches
        if not launch.fits(SM100_MAX_SHARED_MEMORY, SM100_TENSOR_MEMORY_COLUMNS)
    }
    assert not too_large


@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_blackwell_tiles_match_reference(dtype, head_dim):
    """Tiles and tuned configs selected on Blackwell compute the reference forward and grads."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (10, 11):
        pytest.skip("Blackwell (SM10x/SM11x) GPU required")
    inputs = make_inputs(4, head_dim, topk=8, dtype=dtype, seq_len=128)
    reference_inputs = {
        name: value.detach().double().requires_grad_() if value.is_floating_point() else value
        for name, value in inputs.items()
    }
    expected = gather_attn(**reference_inputs, sliding_window_size=64, impl=Impl.REFERENCE)
    grad_output = torch.randn_like(expected)
    expected.backward(grad_output)

    output = gather_attn(**inputs, sliding_window_size=64, kernel_options={"backend": "triton"})
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
