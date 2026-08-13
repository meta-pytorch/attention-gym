"""Tests for the CuTeDSL KDA recompute kernel."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
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


def test_recompute_uses_optional_q_and_gate_arguments():
    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd

    torch.manual_seed(5)
    shape = (1, 64, 1, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn_like(k)
    gk = -torch.rand(shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, 64, 1, device="cuda")
    A = torch.randn(1, 64, 64, device="cuda", dtype=torch.bfloat16).tril().unsqueeze(2)
    cu_seqlens = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    metadata = ChunkMetadata(
        cu_seqlens,
        prepare_chunk_indices(cu_seqlens, 64),
        torch.tensor(1, device="cuda", dtype=torch.int32),
    )

    plain_w, plain_u, plain_qg, plain_kg = recompute_w_u_fwd(k, v, beta, A, metadata=metadata)
    q_only_w, q_only_u, q_only_qg, q_only_kg = recompute_w_u_fwd(
        k, v, beta, A, metadata=metadata, q=q
    )
    gate_w, gate_u, gate_qg, gate_kg = recompute_w_u_fwd(k, v, beta, A, metadata=metadata, gk=gk)
    both_w, both_u, both_qg, both_kg = recompute_w_u_fwd(
        k, v, beta, A, metadata=metadata, q=q, gk=gk
    )

    assert plain_qg is plain_kg is q_only_qg is q_only_kg is None
    assert gate_qg is None
    assert gate_kg is not None and both_qg is not None and both_kg is not None
    torch.testing.assert_close(q_only_w, plain_w, rtol=0, atol=0)
    torch.testing.assert_close(q_only_u, plain_u, rtol=0, atol=0)
    torch.testing.assert_close(gate_w, both_w, rtol=0, atol=0)
    torch.testing.assert_close(gate_u, both_u, rtol=0, atol=0)
    torch.testing.assert_close(gate_kg, both_kg, rtol=0, atol=0)

    expected_qg = q.float() * torch.exp2(gk)
    expected_kg = k.float() * torch.exp2(gk[:, -1:] - gk)
    weighted_k = k.transpose(1, 2).float() * beta.transpose(1, 2)[..., None]
    weighted_k *= torch.exp2(gk.transpose(1, 2))
    expected_w = (A.float().transpose(1, 2) @ weighted_k).transpose(1, 2)
    torch.testing.assert_close(both_qg.float(), expected_qg, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(both_kg.float(), expected_kg, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(both_w.float(), expected_w, rtol=2e-2, atol=0.2)


def test_recompute_rejects_mismatched_ragged_chunk_size():
    """Reject offsets whose logical chunks do not match the fixed CuTe specialization."""
    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd

    shape = (1, 64, 1, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    beta = torch.rand(1, 64, 1, device="cuda")
    A = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16).tril()
    cu_seqlens = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 64, 32)

    with pytest.raises(ValueError, match="metadata chunk size must match chunk_size=64"):
        recompute_w_u_fwd(k, v, beta, A, metadata)
