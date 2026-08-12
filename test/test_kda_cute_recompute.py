"""Tests for the CuTeDSL KDA recompute kernel."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.utils import ChunkMetadata, prepare_chunk_indices

pytest.importorskip("cutlass")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL KDA recompute kernel requires CUDA capability 10.0 or newer",
)


def test_recompute_ignores_upper_triangle_and_reuses_compilation(tmp_path, monkeypatch):
    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import (
        _compile_recompute_w_u,
        recompute_w_u_fwd,
    )

    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path / "cache"))
    _compile_recompute_w_u.cache_clear()
    torch.manual_seed(4)
    shape = (1, 64, 1, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    beta = torch.rand(1, 64, 1, device="cuda")
    A = torch.randn(1, 64, 64, device="cuda", dtype=torch.bfloat16)
    A = A.tril().masked_fill(torch.ones_like(A, dtype=torch.bool).triu(1), torch.nan).unsqueeze(2)
    cu_seqlens = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    chunk_indices = prepare_chunk_indices(cu_seqlens, 64)
    metadata = ChunkMetadata(
        cu_seqlens,
        chunk_indices,
        torch.tensor(1, device="cuda", dtype=torch.int32),
    )

    def run_recompute():
        return recompute_w_u_fwd(
            k,
            v,
            beta,
            A,
            metadata=metadata,
        )

    w, u, _, _ = run_recompute()
    first_cache_info = _compile_recompute_w_u.cache_info()
    repeated_w, repeated_u, _, _ = run_recompute()
    second_cache_info = _compile_recompute_w_u.cache_info()
    assert second_cache_info.hits == first_cache_info.hits + 1
    assert second_cache_info.currsize == first_cache_info.currsize
    torch.testing.assert_close(repeated_w, w, rtol=0, atol=0)
    torch.testing.assert_close(repeated_u, u, rtol=0, atol=0)

    A = A.nan_to_num().float().transpose(1, 2)
    beta = beta.transpose(1, 2)[..., None]
    expected_w = (A @ (k.transpose(1, 2).float() * beta)).transpose(1, 2)
    expected_u = (A @ (v.transpose(1, 2).float() * beta)).transpose(1, 2)
    torch.testing.assert_close(w.float(), expected_w, rtol=2e-2, atol=0.2)
    torch.testing.assert_close(u.float(), expected_u, rtol=2e-2, atol=0.2)
