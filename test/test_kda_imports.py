"""Cold-import and registration tests for the public KDA boundary."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch


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
        )

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
        """
    )


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
