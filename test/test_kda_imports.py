"""Cold-import and registration tests for the public KDA boundary."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

from attn_gym.linear.kda import bound_gate


def run_fresh_python(source: str) -> None:
    """Run a repository import scenario without this pytest process's module state."""
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=True,
        text=True,
    )


def test_reference_api_imports_without_optional_kernel_dependencies():
    """Keep the public reference path usable with torch as its only dependency."""
    run_fresh_python(
        """
        import builtins
        import torch

        original_import = builtins.__import__

        def reject_optional_backends(name, *args, **kwargs):
            if name.split('.', 1)[0] in {'cutlass', 'triton'}:
                raise ModuleNotFoundError(name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = reject_optional_backends

        import attn_gym.linear as linear

        from attn_gym.linear import (
            active_token_mask,
            chunk_kda,
            mask_inactive_token_gradients,
            mask_inactive_tokens,
            recurrent_kda,
            recurrent_kda_decode,
        )
        from attn_gym.linear.kda import bound_gate

        raw_gate = torch.randn(1, 3, 1, 2, requires_grad=True)
        a_log = torch.randn(1, requires_grad=True)
        dt_bias = torch.randn(1, 2, requires_grad=True)
        gate = bound_gate(raw_gate, a_log, dt_bias, lower_bound=-3.25)
        expected_gate = -3.25 * torch.sigmoid(
            a_log.exp().view(1, 1, 1, 1) * (raw_gate.float() + dt_bias)
        )
        assert gate.dtype == torch.float32
        torch.testing.assert_close(gate, expected_gate, rtol=0, atol=0)
        torch.autograd.grad(gate.sum(), (raw_gate, a_log, dt_bias))

        packed = torch.randn(1, 3, 2)
        offsets = torch.tensor([0, 2], dtype=torch.int32)
        mask = active_token_mask(packed, offsets)
        mask_inactive_tokens(packed, mask)
        mask_inactive_token_gradients(packed, mask)

        shape = (1, 3, 1, 2)
        q = torch.randn(shape)
        k = torch.randn(shape)
        v = torch.randn(shape)
        beta = torch.rand(shape[:3])
        gate = -torch.rand(shape)
        assert not hasattr(linear, 'bounded_gate')
        assert not hasattr(linear, 'bounded_gate_cumsum')
        recurrent_kda(q, k, v, gate, beta, impl='reference')
        chunk_kda(q, k, v, gate, beta, impl='reference')
        assert callable(recurrent_kda_decode)
        """
    )


@pytest.mark.parametrize("lower_bound", [1.0, float("-inf")])
def test_compiled_bound_gate_rejects_invalid_lower_bound(lower_bound: float):
    """Keep compiled and eager scalar validation consistent."""
    raw_gate = torch.randn(1, 3, 1, 2)
    a_log = torch.randn(1)
    dt_bias = torch.randn(1, 2)
    compiled = torch.compile(bound_gate, fullgraph=True)

    with pytest.raises(torch._dynamo.exc.Unsupported, match="lower_bound"):
        compiled(raw_gate, a_log, dt_bias, lower_bound=lower_bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fallback requires a CUDA device")
def test_masking_cuda_falls_back_without_triton():
    """Keep CUDA masking functional when the optional Triton package is absent."""
    run_fresh_python(
        """
        import builtins
        import torch

        values = torch.tensor([[[1.0], [2.0]]], device='cuda')
        active_mask = torch.tensor([True, False], device='cuda')
        original_import = builtins.__import__

        def reject_triton(name, *args, **kwargs):
            if name.split('.', 1)[0] == 'triton':
                raise ModuleNotFoundError(name=name)
            return original_import(name, *args, **kwargs)

        builtins.__import__ = reject_triton

        from attn_gym.linear import mask_inactive_tokens

        actual = mask_inactive_tokens(values, active_mask)
        torch.testing.assert_close(actual, torch.tensor([[[1.0], [0.0]]], device='cuda'))
        """
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="cold fused chunk compilation requires CUDA capability 10.0 or newer",
)
def test_chunk_kda_cold_public_fullgraph_compile():
    """Register operator contracts before the first fused call enters Dynamo."""
    pytest.importorskip("cutlass")
    run_fresh_python(
        """
        import torch
        import torch.nn.functional as F

        from attn_gym.linear import chunk_kda
        shape = (1, 64, 1, 128)
        q = F.normalize(torch.randn(shape, device='cuda'), dim=-1).to(torch.bfloat16)
        k = F.normalize(torch.randn(shape, device='cuda'), dim=-1).to(torch.bfloat16)
        v = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        gate = -torch.rand(shape, device='cuda')
        beta = torch.rand(shape[:3], device='cuda')

        compiled = torch.compile(chunk_kda, fullgraph=True)
        compiled(q, k, v, gate, beta, autotune=False)
        torch.cuda.synchronize()
        """
    )
