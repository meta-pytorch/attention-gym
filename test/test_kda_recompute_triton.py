"""Direct tests for the Triton KDA W/U recompute kernel's optional inputs and modes."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.triton.recompute_w_u import recompute_w_u_fwd_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the Triton KDA recompute kernel requires an SM90 or newer GPU",
)


def _reference(k, v, beta, A, offsets, gk=None, q=None, chunk_size=64):
    """Chunkwise fp32 reference with grouped value heads mapped onto key heads."""
    hk, hv = k.shape[2], v.shape[2]
    group = hv // hk
    k = k.float().repeat_interleave(group, dim=2)
    gk = None if gk is None else gk.float().repeat_interleave(group, dim=2)
    v, beta, A = v.float(), beta.float(), A.float()
    w, u = torch.zeros_like(k), torch.zeros_like(v)
    kg = None if gk is None else torch.zeros_like(k)
    for start, end in pairwise(offsets.tolist()):
        for cs in range(start, end, chunk_size):
            ce = min(cs + chunk_size, end)
            a_blk = A[0, cs:ce, :, : ce - cs].permute(1, 0, 2).tril()
            kb = k[0, cs:ce] * beta[0, cs:ce, :, None]
            if gk is not None:
                kb = kb * torch.exp2(gk[0, cs:ce])
                kg[0, cs:ce] = k[0, cs:ce] * torch.exp2(gk[0, ce - 1 : ce] - gk[0, cs:ce])
            w[0, cs:ce] = (a_blk @ kb.permute(1, 0, 2)).permute(1, 0, 2)
            vb = (v[0, cs:ce] * beta[0, cs:ce, :, None]).permute(1, 0, 2)
            u[0, cs:ce] = (a_blk @ vb).permute(1, 0, 2)
    qg = None if q is None or gk is None else q.float() * torch.exp2(gk)
    return w, u, qg, kg


def test_optional_q_and_gate_arguments():
    """qg needs q and gk, kg needs gk, and q alone changes nothing."""
    torch.manual_seed(5)
    shape = (1, 64, 1, 128)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    q = torch.randn_like(k)
    gk = -torch.rand(shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, 64, 1, device="cuda")
    A = torch.randn(1, 64, 64, device="cuda", dtype=torch.bfloat16).tril().unsqueeze(2)

    plain = recompute_w_u_fwd_triton(k, v, beta, A)
    q_only = recompute_w_u_fwd_triton(k, v, beta, A, q=q)
    gate = recompute_w_u_fwd_triton(k, v, beta, A, gk=gk)
    both = recompute_w_u_fwd_triton(k, v, beta, A, q=q, gk=gk)

    assert plain[2] is plain[3] is q_only[2] is q_only[3] is gate[2] is None
    assert gate[3] is not None and both[2] is not None and both[3] is not None
    for actual, expected in zip(q_only[:2], plain[:2], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(gate[:2] + gate[3:], both[:2] + both[3:], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    offsets = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    ref_w, ref_u, ref_qg, ref_kg = _reference(k, v, beta, A, offsets, gk=gk, q=q)
    torch.testing.assert_close(both[0].float(), ref_w, rtol=2e-2, atol=0.2)
    torch.testing.assert_close(both[1].float(), ref_u, rtol=2e-2, atol=0.2)
    torch.testing.assert_close(both[2].float(), ref_qg, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(both[3].float(), ref_kg, rtol=1e-2, atol=1e-2)


def test_precision_modes_tighten_error_and_reject_half_a():
    """tf32x3 beats bf16 against an fp32 reference and requires fp32 A."""
    torch.manual_seed(7)
    total = 257  # tail chunk included
    offsets = torch.tensor([0, 129, 257], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(offsets, total, 64)
    k = torch.randn(1, total, 2, 128, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn_like(k) / 8
    beta = torch.rand(1, total, 2, device="cuda")
    A = torch.randn(1, total, 2, 64, device="cuda", dtype=torch.float32) / 8
    _, ref_u, _, _ = _reference(k, v, beta, A, offsets)

    errors = {}
    for precision in ("bf16", "tf32", "tf32x3"):
        _, u, _, _ = recompute_w_u_fwd_triton(k, v, beta, A, metadata, dot_precision=precision)
        errors[precision] = (u.float() - ref_u).abs().max().item()
    assert errors["tf32x3"] < errors["bf16"], errors

    with pytest.raises(ValueError, match="tf32x3"):
        recompute_w_u_fwd_triton(k, v, beta, A.bfloat16(), metadata, dot_precision="tf32x3")


def test_grouped_value_heads_map_onto_key_heads():
    """H_V > H_K reads k/gk from key head ``value_head // (H_V // H_K)``."""
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

    w, u, qg, kg = recompute_w_u_fwd_triton(k, v, beta, A, metadata, q=q, gk=gk)
    ref_w, ref_u, ref_qg, ref_kg = _reference(k, v, beta, A, offsets, gk=gk, q=q)
    assert w.shape == kg.shape == (1, total, hv, 128)
    torch.testing.assert_close(qg.float(), ref_qg, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(kg.float(), ref_kg, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(w.float(), ref_w, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(u.float(), ref_u, rtol=2e-2, atol=2e-2)
