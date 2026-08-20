"""Tests for the CuTeDSL KDA recompute kernel."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    ScheduleRequest,
    prepare_ragged_chunk_metadata,
)

pytest.importorskip("cutlass")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL KDA recompute kernel requires CUDA capability 10.0 or newer",
)


def test_recompute_ignores_upper_triangle_and_reuses_compilation(tmp_path, monkeypatch):
    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import (
        _compile_recompute_w_u,
        _recompute_w_u_fwd_cute,
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

    def run_recompute():
        return recompute_w_u_fwd(k, v, beta, A, metadata=None)

    # Compile-cache reuse contract lives on the CuTe kernel, which the public
    # API no longer reaches; call the internal entry to keep it compiling.
    w_cute, u_cute, _, _ = _recompute_w_u_fwd_cute(k, v, beta, A, metadata=None)
    first_cache_info = _compile_recompute_w_u.cache_info()
    repeated_w, repeated_u, _, _ = _recompute_w_u_fwd_cute(k, v, beta, A, metadata=None)
    second_cache_info = _compile_recompute_w_u.cache_info()
    assert second_cache_info.hits == first_cache_info.hits + 1
    assert second_cache_info.currsize == first_cache_info.currsize
    torch.testing.assert_close(repeated_w, w_cute, rtol=0, atol=0)
    torch.testing.assert_close(repeated_u, u_cute, rtol=0, atol=0)

    w, u, _, _ = run_recompute()

    A = A.nan_to_num().float().transpose(1, 2)
    beta = beta.transpose(1, 2)[..., None]
    expected_w = (A @ (k.transpose(1, 2).float() * beta)).transpose(1, 2)
    expected_u = (A @ (v.transpose(1, 2).float() * beta)).transpose(1, 2)
    for actual_w, actual_u in ((w, u), (w_cute, u_cute)):
        torch.testing.assert_close(actual_w.float(), expected_w, rtol=2e-2, atol=0.2)
        torch.testing.assert_close(actual_u.float(), expected_u, rtol=2e-2, atol=0.2)


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

    plain_w, plain_u, plain_qg, plain_kg = recompute_w_u_fwd(k, v, beta, A, metadata=None)
    q_only_w, q_only_u, q_only_qg, q_only_kg = recompute_w_u_fwd(k, v, beta, A, metadata=None, q=q)
    gate_w, gate_u, gate_qg, gate_kg = recompute_w_u_fwd(k, v, beta, A, metadata=None, gk=gk)
    both_w, both_u, both_qg, both_kg = recompute_w_u_fwd(k, v, beta, A, metadata=None, q=q, gk=gk)

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


def test_recompute_fake_tensors_reach_no_launch():
    """Fake tracing must stop at output metadata on every dispatch path."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd

    with FakeTensorMode():
        k = torch.empty(1, 128, 2, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.empty_like(k)
        q = torch.empty_like(k)
        gk = torch.empty(1, 128, 2, 128, device="cuda", dtype=torch.float32)
        beta = torch.empty(1, 128, 2, device="cuda")
        A = torch.empty(1, 128, 2, 64, device="cuda", dtype=torch.bfloat16)
        cu_seqlens = torch.empty(2, device="cuda", dtype=torch.int32)
        chunk_offsets = torch.empty(2, device="cuda", dtype=torch.int32)
        ragged = RaggedChunkMetadata(cu_seqlens, chunk_offsets, capacity=2, chunk_size=64)
        for metadata in (None, ragged):
            w, u, qg, kg = recompute_w_u_fwd(k, v, beta, A, metadata=metadata, q=q, gk=gk)
            assert w.shape == k.shape and u.shape == v.shape
            assert qg.shape == k.shape and kg.shape == k.shape


def test_recompute_triton_matches_cute_on_ragged_partial_chunks():
    """Pin the default path against the CuTe kernel on tails and empty sequences."""
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import recompute_w_u_fwd

    torch.manual_seed(6)
    lengths = [65, 0, 63, 129, 1]
    total = sum(lengths)
    offsets = torch.tensor([0, 65, 65, 128, 257, 258], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(offsets, total, 64)
    k = torch.randn(1, total, 2, 128, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn_like(k) / 8
    q = torch.randn_like(k) / 8
    gk = -torch.rand(1, total, 2, 128, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, total, 2, device="cuda")
    # A content is arbitrary: both paths apply identical inclusive-tril masking.
    A = torch.randn(1, total, 2, 64, device="cuda", dtype=torch.bfloat16) / 8

    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import _recompute_w_u_fwd_cute

    w, u, qg, kg = recompute_w_u_fwd(k, v, beta, A, metadata=metadata, q=q, gk=gk)
    w_c, u_c, qg_c, kg_c = _recompute_w_u_fwd_cute(
        k, v, beta, A, metadata=metadata, q=q, gk=gk, dot_precision="tf32"
    )
    # qg/kg are pointwise and unaffected by dot precision: bitwise.
    torch.testing.assert_close(qg, qg_c, rtol=0, atol=0)
    torch.testing.assert_close(kg, kg_c, rtol=0, atol=0)
    # w/u differ only by bf16-vs-tf32 operand rounding.
    torch.testing.assert_close(w.float(), w_c.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(u.float(), u_c.float(), rtol=2e-2, atol=2e-2)


def test_recompute_precision_modes_match_cute_and_tighten_error():
    """tf32/tf32x3 run real higher-precision dots, not silently-degraded bf16."""
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import (
        _recompute_w_u_fwd_cute,
        recompute_w_u_fwd,
    )

    torch.manual_seed(7)
    total = 257  # tail chunk included
    offsets = torch.tensor([0, 129, 257], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(offsets, total, 64)
    k = torch.randn(1, total, 2, 128, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn_like(k) / 8
    beta = torch.rand(1, total, 2, device="cuda")
    A = torch.randn(1, total, 2, 64, device="cuda", dtype=torch.float32) / 8

    reference_u = None
    for precision in ("bf16", "tf32", "tf32x3"):
        w, u, _, _ = recompute_w_u_fwd(k, v, beta, A, metadata=metadata, dot_precision=precision)
        w_c, u_c, _, _ = _recompute_w_u_fwd_cute(
            k, v, beta, A, metadata=metadata, dot_precision=precision
        )
        torch.testing.assert_close(w.float(), w_c.float(), rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(u.float(), u_c.float(), rtol=5e-3, atol=5e-3)
        if reference_u is None:
            # Chunkwise fp32 reference: A holds per-chunk [rows, 64] blocks.
            reference_u = torch.zeros_like(v, dtype=torch.float32)
            bounds = offsets.tolist()
            for start, end in pairwise(bounds):
                for cs in range(start, end, 64):
                    ce = min(cs + 64, end)
                    n = ce - cs
                    a_blk = A[0, cs:ce, :, :n].float().permute(1, 0, 2).tril()
                    v_blk = (v[0, cs:ce].float() * beta[0, cs:ce, :, None]).permute(1, 0, 2)
                    reference_u[0, cs:ce] = (a_blk @ v_blk).permute(1, 0, 2)
        errors = (u.float() - reference_u).abs().max().item()
        if precision == "bf16":
            bf16_error = errors
        elif precision == "tf32x3":
            assert errors < bf16_error, (
                f"tf32x3 U error {errors} should beat bf16 error {bf16_error}"
            )

    with pytest.raises(ValueError, match="tf32x3"):
        recompute_w_u_fwd(k, v, beta, A.bfloat16(), metadata=metadata, dot_precision="tf32x3")


def test_recompute_supports_grouped_value_heads():
    """H_V > H_K maps each value head onto its key-head group for k/gk."""
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.linear.kda.fwd.cute.recompute_w_u_fwd import (
        _recompute_w_u_fwd_cute,
        recompute_w_u_fwd,
    )

    torch.manual_seed(8)
    total, hk, hv = 130, 2, 4
    offsets = torch.tensor([0, 65, 130], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(offsets, total, 64)
    k = torch.randn(1, total, hk, 128, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn(1, total, hv, 128, device="cuda", dtype=torch.bfloat16) / 8
    q = torch.randn(1, total, hv, 128, device="cuda", dtype=torch.bfloat16) / 8
    gk = -torch.rand(1, total, hk, 128, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, total, hv, device="cuda")
    A = torch.randn(1, total, hv, 64, device="cuda", dtype=torch.bfloat16) / 8

    w, u, qg, kg = recompute_w_u_fwd(k, v, beta, A, metadata=metadata, q=q, gk=gk)
    w_c, u_c, qg_c, kg_c = _recompute_w_u_fwd_cute(k, v, beta, A, metadata=metadata, q=q, gk=gk)
    assert w.shape == (1, total, hv, 128) and kg.shape == (1, total, hv, 128)
    torch.testing.assert_close(qg, qg_c, rtol=0, atol=0)
    torch.testing.assert_close(kg, kg_c, rtol=0, atol=0)
    torch.testing.assert_close(w.float(), w_c.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(u.float(), u_c.float(), rtol=2e-2, atol=2e-2)


def _chunkwise_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    gk: torch.Tensor,
    bounds: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate W/U in fp32 over the active sequence-local 64-token chunks."""
    w = torch.zeros_like(k, dtype=torch.float32)
    u = torch.zeros_like(v, dtype=torch.float32)
    for start, end in pairwise(bounds):
        for chunk_start in range(start, end, 64):
            chunk_end = min(chunk_start + 64, end)
            rows = chunk_end - chunk_start
            a = A[0, chunk_start:chunk_end, :, :rows].float().permute(1, 0, 2).tril()
            scaled_beta = beta[0, chunk_start:chunk_end, :, None].float()
            gate = torch.exp2(gk[0, chunk_start:chunk_end].float())
            kb = (k[0, chunk_start:chunk_end].float() * scaled_beta * gate).permute(1, 0, 2)
            vb = (v[0, chunk_start:chunk_end].float() * scaled_beta).permute(1, 0, 2)
            w[0, chunk_start:chunk_end] = (a @ kb).permute(1, 0, 2)
            u[0, chunk_start:chunk_end] = (a @ vb).permute(1, 0, 2)
    return w, u


def test_recompute_persistent_matches_static_over_capacity():
    """Persistent scheduling must be bit-identical to static scheduling."""
    from attn_gym.linear.kda.fwd.triton.recompute_w_u import recompute_w_u_fwd_triton

    torch.manual_seed(9)
    lengths = [65, 0, 63]
    active = sum(lengths)
    capacity_tokens = 16 * active
    bounds = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    offsets = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(offsets, capacity_tokens, 64)
    assert metadata.capacity >= 8 * ((active + 63) // 64)

    heads = 2
    k = torch.randn(1, capacity_tokens, heads, 128, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn_like(k) / 8
    q = torch.randn_like(k) / 8
    gk = -torch.rand(1, capacity_tokens, heads, 128, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, capacity_tokens, heads, device="cuda")
    A = torch.randn(1, capacity_tokens, heads, 64, device="cuda", dtype=torch.float32) / 8

    # Pin the config so bit-equality does not depend on two independent
    # autotune winners choosing the same accumulation shape.
    static = recompute_w_u_fwd_triton(k, v, beta, A, metadata, q=q, gk=gk, autotune=False)
    persistent = recompute_w_u_fwd_triton(
        k,
        v,
        beta,
        A,
        metadata,
        q=q,
        gk=gk,
        autotune=False,
        schedule=ScheduleRequest.PERSISTENT,
    )
    for static_tensor, persistent_tensor in zip(static, persistent, strict=True):
        assert torch.equal(persistent_tensor[:, :active], static_tensor[:, :active])

    expected_w, expected_u = _chunkwise_reference(k, v, beta, A, gk, bounds)
    torch.testing.assert_close(
        persistent[0][:, :active].float(), expected_w[:, :active], rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        persistent[1][:, :active].float(), expected_u[:, :active], rtol=2e-2, atol=2e-2
    )


def test_recompute_persistent_is_noop_for_dense():
    """Dense launch grids are already exact, so the request is trivially satisfied."""
    from attn_gym.linear.kda.fwd.triton.recompute_w_u import recompute_w_u_fwd_triton

    torch.manual_seed(5)
    k = torch.randn(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn_like(k) / 8
    beta = torch.rand(1, 64, 1, device="cuda")
    A = torch.randn(1, 64, 1, 64, device="cuda", dtype=torch.bfloat16).tril() / 8

    dense = recompute_w_u_fwd_triton(k, v, beta, A, None)
    dense_persistent = recompute_w_u_fwd_triton(
        k, v, beta, A, None, schedule=ScheduleRequest.PERSISTENT
    )
    for actual, other in zip(dense_persistent, dense, strict=True):
        assert (actual is None and other is None) or torch.equal(actual, other)
