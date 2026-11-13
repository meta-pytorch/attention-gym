"""Warpdrive GDN T pass (``gdn_tinv``) against an fp64 chunk-inverse reference.

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive GDN T pass requires an SM100-family GPU",
)

B_T = 64


def _inputs(lens, heads, key_heads, d_k, dtype, beta_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)
    k = torch.randn(tokens, key_heads, d_k, device="cuda", generator=gen)
    k = torch.nn.functional.normalize(k, dim=-1).to(dtype)
    gate = -0.3 * torch.rand(tokens, heads, device="cuda", generator=gen)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen).to(beta_dtype)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return k, gate, beta, cu


def _reference(k, gate, beta, lens, heads):
    """fp64 ``(I + strict_lower(M))^-1 * beta[j]`` per chunk, ``M = KK^T * decay * beta[i]``;
    returns ``{tile row: (tile [heads, n, n], n valid rows)}``."""
    group = heads // k.shape[1]
    out, start = {}, 0
    for b, length in enumerate(lens):
        for c in range((length + B_T - 1) // B_T):
            n = min(B_T, length - c * B_T)
            t = slice(start + c * B_T, start + c * B_T + n)
            kk = k[t].double().repeat_interleave(group, dim=1).transpose(0, 1)  # [h, n, d]
            g = gate[t].double().cumsum(0).T  # [h, n]
            bt = beta[t].double().T
            m = kk @ kk.transpose(1, 2) * torch.exp(g[:, :, None] - g[:, None, :])
            m = torch.tril(m * bt[:, :, None], -1)
            eye = torch.eye(n, dtype=torch.float64, device="cuda")
            out[start // B_T + b + c] = (torch.linalg.inv(eye + m) * bt[:, None, :], n)
        start += length
    return out


def _run(k, gate, beta, cu, lens, heads):
    from attn_gym.linear._delta_rule.warpdrive import gdn_tinv

    rows = gdn_tinv.tinv_rows(sum(lens), len(lens))
    tinv = torch.full((rows, heads, B_T, B_T), float("nan"), dtype=k.dtype, device="cuda")
    table = torch.full((rows, 4), -1, dtype=torch.int32, device="cuda")
    count = torch.zeros(1, dtype=torch.int32, device="cuda")
    gdn_tinv.gdn_tinv(k, gate, beta, cu, tinv, table, count)
    return tinv, table, count


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "d_k", "dtype", "beta_dtype"),
    [
        # Partial, exact and one-past chunks; GQA 2.
        ((1, 63, 64, 65, 128, 129), 4, 2, 128, torch.bfloat16, torch.float32),
        ((300, 77), 8, 2, 64, torch.float16, torch.float16),
        # More than 32 sequences (row-table warp scan carries) and an odd tile count (a lone
        # last pair member).
        ((7,) * 40 + (200,), 3, 1, 128, torch.bfloat16, torch.bfloat16),
        # Many pairs per CTA: every channel ring wraps its stages and phases.
        ((16384, 3000), 16, 4, 64, torch.bfloat16, torch.float32),
    ],
    ids=["boundaries-gqa2", "fp16-dk64-gqa4", "many-seqs-odd-tiles", "ring-wrap"],
)
def test_gdn_tinv_matches_reference(lens, heads, key_heads, d_k, dtype, beta_dtype):
    k, gate, beta, cu = _inputs(lens, heads, key_heads, d_k, dtype, beta_dtype)
    tinv, table, count = _run(k, gate, beta, cu, lens, heads)

    expected_rows = [
        (c, b, start, start + length)
        for b, (start, length) in enumerate(zip(cu[:-1].tolist(), lens, strict=True))
        for c in range((length + B_T - 1) // B_T)
    ]
    assert count.item() == len(expected_rows)
    assert table[: len(expected_rows)].tolist() == [list(r) for r in expected_rows]

    atol = 1e-2 if dtype is torch.bfloat16 else 2e-3
    for row, (ref, n) in _reference(k, gate, beta, lens, heads).items():
        got = tinv[row, :, :n, :n].double()
        torch.testing.assert_close(got, ref, atol=atol, rtol=0, msg=f"tile row {row}")


def test_gdn_tinv_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequence count and head counts stay runtime values of one specialization."""
    from attn_gym.linear._delta_rule.warpdrive import gdn_tinv

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16)]
    gdn_tinv.compile_gdn_tinv.cache_clear()
    for lens, heads, key_heads in shapes:
        _run(*_inputs(lens, heads, key_heads, 128, torch.bfloat16, torch.float32), lens, heads)
    info = gdn_tinv.compile_gdn_tinv.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
