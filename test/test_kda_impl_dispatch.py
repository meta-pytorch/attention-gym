# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implementation-selection tests for the public KDA operations."""

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import Impl, KernelOptions, chunk_kda, recurrent_kda
from attn_gym.linear._delta_rule.validation import validate_paged_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="KDA ops require CUDA")

BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


def _inputs(
    batch: int = 2,
    tokens: int = 37,
    heads: int = 2,
    head_dim: int = 64,
    value_dim: int | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    device: str = "cuda",
):
    torch.manual_seed(seed)

    def normalized() -> torch.Tensor:
        raw = torch.randn(batch, tokens, heads, head_dim, device=device)
        return F.normalize(raw, dim=-1).to(dtype)

    q, k = normalized(), normalized()
    value_dim = head_dim if value_dim is None else value_dim
    v = torch.randn(batch, tokens, heads, value_dim, device=device, dtype=dtype)
    gate = -torch.rand(batch, tokens, heads, head_dim, device=device) * 3.0
    beta = torch.rand(batch, tokens, heads, device=device)
    return q, k, v, gate, beta


def test_impl_accepts_enum_and_string():
    """Coerce both spellings and reject unknown selectors with the valid set."""
    q, k, v, gate, beta = _inputs(tokens=8)

    from_enum, _ = recurrent_kda(q, k, v, gate, beta, impl=Impl.REFERENCE)
    from_string, _ = recurrent_kda(q, k, v, gate, beta, impl="reference")
    torch.testing.assert_close(from_enum, from_string, rtol=0, atol=0)

    with pytest.raises(ValueError, match="'fused', 'reference'"):
        recurrent_kda(q, k, v, gate, beta, impl="naive")


def test_recurrent_accepts_reserved_autotune_flag():
    """Keep the public selector signatures aligned while recurrent uses a fixed policy."""
    q, k, v, gate, beta = _inputs(tokens=8)

    tuned, _ = recurrent_kda(q, k, v, gate, beta, autotune=True, impl="reference")
    fixed, _ = recurrent_kda(q, k, v, gate, beta, autotune=False, impl="reference")
    torch.testing.assert_close(tuned, fixed, rtol=0, atol=0)


def test_recurrent_reference_matches_fused():
    """Agree across implementations on dense batches."""
    q, k, v, gate, beta = _inputs()
    initial_state = torch.randn(2, 2, 64, 64, device="cuda")

    fused, fused_state = recurrent_kda(
        q, k, v, gate, beta, initial_state, output_final_state=True, impl="fused"
    )
    reference, reference_state = recurrent_kda(
        q, k, v, gate, beta, initial_state, output_final_state=True, impl="reference"
    )
    torch.testing.assert_close(fused, reference, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_state, reference_state, rtol=1e-5, atol=1e-5)


def test_recurrent_reference_packed_capacity_and_empty_slots():
    """Match the fused packed contract, including padding slots and tails."""
    q, k, v, gate, beta = _inputs(batch=1, tokens=32, seed=1)
    cu_seqlens = torch.tensor([0, 0, 11, 27, 27], device="cuda", dtype=torch.int32)
    initial_state = torch.randn(4, 2, 64, 64, device="cuda")

    with torch.no_grad():
        fused, fused_state = recurrent_kda(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            impl="fused",
        )
    reference, reference_state = recurrent_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="reference",
    )
    active = cu_seqlens[-1].item()
    torch.testing.assert_close(fused[:, :active], reference[:, :active], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_state, reference_state, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(reference_state[0], initial_state[0], rtol=0, atol=0)


def test_reference_impls_run_on_cpu():
    """The reference contract promises any hardware; pin the CPU boundary."""
    q, k, v, gate, beta = _inputs(tokens=70, device="cpu")
    output, state = recurrent_kda(q, k, v, gate, beta, output_final_state=True, impl="reference")
    chunk_output, chunk_state = chunk_kda(
        q, k, v, gate, beta, output_final_state=True, impl="reference"
    )
    for tensor in (output, state, chunk_output, chunk_state):
        assert tensor.device.type == "cpu"
    torch.testing.assert_close(chunk_output, output, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(chunk_state, state, rtol=1e-4, atol=1e-4)


def test_recurrent_reference_is_differentiable():
    """Keep a training-capable escape hatch where the fused scan has none."""
    q, k, v, gate, beta = _inputs(tokens=8)
    q, v = q.requires_grad_(), v.requires_grad_()

    output, _ = recurrent_kda(q, k, v, gate, beta, impl="reference")
    output.sum().backward()
    assert q.grad is not None and v.grad is not None


@pytest.mark.skipif(not BLACKWELL, reason="fused chunk_kda requires CUDA capability 10.0")
def test_chunk_reference_matches_fused():
    """Agree across implementations through the per-token gate contract."""
    q, k, v, gate, beta = _inputs(tokens=100, head_dim=128, dtype=torch.bfloat16, seed=2)
    fused, fused_state = chunk_kda(q, k, v, gate, beta, output_final_state=True, impl="fused")
    reference, reference_state = chunk_kda(
        q, k, v, gate, beta, output_final_state=True, impl="reference"
    )
    assert reference.dtype == q.dtype and reference_state.dtype == torch.float32
    torch.testing.assert_close(fused.float(), reference.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(fused_state, reference_state, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not BLACKWELL, reason="fused chunk_kda requires CUDA capability 10.0")
def test_chunk_reference_packed_matches_fused():
    """Match the fused ragged path per sequence, including empty slots."""
    q, k, v, gate, beta = _inputs(batch=1, tokens=150, head_dim=128, dtype=torch.bfloat16, seed=3)
    cu_seqlens = torch.tensor([0, 0, 70, 150], device="cuda", dtype=torch.int32)
    fused, fused_state = chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="fused",
    )
    reference, reference_state = chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="reference",
    )
    torch.testing.assert_close(fused.float(), reference.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(fused_state, reference_state, rtol=2e-2, atol=2e-2)


def test_chunk_reference_supports_any_head_dim():
    """Lift the fused K=V=128 constraint without changing the contract."""
    q, k, v, gate, beta = _inputs(tokens=70, head_dim=48, value_dim=32, seed=4)
    output, state = chunk_kda(q, k, v, gate, beta, output_final_state=True, impl="reference")
    assert output.shape == v.shape and state.shape == (2, 2, 32, 48)
    assert state.is_contiguous()
    validate_paged_state(
        q,
        v,
        state,
        None,
        torch.arange(q.shape[0], dtype=torch.int32, device=q.device),
    )
    with pytest.raises(ValueError, match="128"):
        chunk_kda(q, k, v, gate, beta, impl="fused")


def test_chunk_kernel_options_plumbing():
    """Expose a typed extension point without silently accepting unknown options."""
    q, k, v, gate, beta = _inputs(tokens=8)
    options: KernelOptions = {}
    expected, _ = chunk_kda(q, k, v, gate, beta, impl="reference")
    actual, _ = chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        impl="reference",
        kernel_options=options,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="no chunk_kda kernel options"):
        chunk_kda(
            q,
            k,
            v,
            gate,
            beta,
            kernel_options={"BACKEND": "MEGA"},  # type: ignore[typeddict-unknown-key]
        )


def test_chunk_reference_rejects_fastmath():
    """Keep fused-only knobs from silently changing meaning."""
    q, k, v, gate, beta = _inputs(tokens=8)
    with pytest.raises(ValueError, match="fastmath"):
        chunk_kda(q, k, v, gate, beta, fastmath=True, impl="reference")


@pytest.mark.skipif(not BLACKWELL, reason="fused chunk_kda requires CUDA capability 10.0")
def test_chunk_autotune_flag_is_deterministic_and_accurate():
    """autotune=False pins fixed heuristic configs without changing the math contract."""
    q, k, v, gate, beta = _inputs(tokens=128, head_dim=128, dtype=torch.bfloat16, seed=5)
    q, v = q.requires_grad_(), v.requires_grad_()

    def run():
        output, state = chunk_kda(q, k, v, gate, beta, output_final_state=True, autotune=False)
        grads = torch.autograd.grad(output.float().square().mean() + state.square().mean(), (q, v))
        return output, state, *grads

    first = run()
    second = run()
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    tuned_output, tuned_state = chunk_kda(
        q.detach(), k, v.detach(), gate, beta, output_final_state=True
    )
    torch.testing.assert_close(first[0], tuned_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(first[1], tuned_state, rtol=2e-2, atol=2e-2)


def test_reference_stays_fp32_under_autocast():
    """The FP32-oracle contract must survive an active AMP region."""
    q, k, v, gate, beta = _inputs(tokens=64)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output, state = recurrent_kda(
            q, k, v, gate, beta, output_final_state=True, impl="reference"
        )
        chunk_output, _ = chunk_kda(q, k, v, gate, beta, impl="reference")
    expected, expected_state = recurrent_kda(
        q, k, v, gate, beta, output_final_state=True, impl="reference"
    )
    chunk_expected, _ = chunk_kda(q, k, v, gate, beta, impl="reference")
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    torch.testing.assert_close(chunk_output, chunk_expected, rtol=0, atol=0)
