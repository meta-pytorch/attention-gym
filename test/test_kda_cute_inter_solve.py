"""Tests for the cached CuTeDSL KDA inter-solve kernels."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize(
    ("head_dim", "chunk_size", "subchunk_size", "message"),
    (
        (64, 64, 16, "head_dim=128"),
        (128, 32, 16, "chunk_size=64"),
        (128, 64, 8, "subchunk_size=16"),
    ),
)
def test_inter_solve_rejects_unsupported_specializations(
    head_dim,
    chunk_size,
    subchunk_size,
    message,
):
    pytest.importorskip("cutlass")
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        _validate_specialization,
    )

    with pytest.raises(AssertionError, match=message):
        _validate_specialization(head_dim, chunk_size, subchunk_size)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL KDA inter-solve requires CUDA capability 10.0 or newer",
)
def test_inter_solve_reuses_compiled_specializations(tmp_path, monkeypatch):
    pytest.importorskip("cutlass")
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        _compile_k3b,
        _compile_k4b,
        chunk_kda_fwd_inter_solve_cute,
    )

    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path / "cache"))
    _compile_k3b.cache_clear()
    _compile_k4b.cache_clear()

    torch.manual_seed(0)
    shape = (1, 128, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    cumulative_gate = (torch.randn(shape, device="cuda") * 0.01).cumsum(1)
    beta = torch.rand(1, 128, 1, device="cuda")
    diagonal_inverse = torch.randn(1, 128, 1, 16, device="cuda") * 0.01
    cu_seqlens = torch.tensor([0, 128], device="cuda", dtype=torch.int64)
    chunk_indices = torch.tensor([[0, 0], [0, 1]], device="cuda", dtype=torch.int64)

    def run_inter_solve():
        Akk = torch.full((1, 128, 1, 64), torch.nan, device="cuda", dtype=q.dtype)
        return chunk_kda_fwd_inter_solve_cute(
            q,
            k,
            cumulative_gate,
            beta,
            diagonal_inverse,
            128**-0.5,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            Akk=Akk,
        )

    run_inter_solve()
    torch.cuda.synchronize()
    first_info = (_compile_k3b.cache_info(), _compile_k4b.cache_info())
    outputs = run_inter_solve()
    torch.cuda.synchronize()
    second_info = (_compile_k3b.cache_info(), _compile_k4b.cache_info())

    assert tuple(output.shape for output in outputs) == ((1, 128, 1, 64),) * 2
    Akk = outputs[1].view(1, 2, 64, 1, 64).permute(0, 1, 3, 2, 4)
    upper = torch.ones(64, 64, dtype=torch.bool, device="cuda").triu(1)
    assert torch.count_nonzero(Akk[..., upper]) == 0
    assert all(
        second.hits == first.hits + 1 and second.currsize == first.currsize
        for first, second in zip(first_info, second_info, strict=True)
    )
