"""Warpdrive KDA prep (``kda_prep``) against its record formulas evaluated in torch.

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import math

import pytest
import torch

from attn_gym.testing.kda import assert_matches_low_precision_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive KDA prep requires SM100 or SM103",
)

B_T = 16


def _inputs(lens, heads, q_heads, key_heads, d, dtype, gate_dtype, beta_dtype, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    q = torch.nn.functional.normalize(normal(tokens, q_heads, d), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(normal(tokens, key_heads, d), dim=-1).to(dtype)
    gate = (-0.3 * torch.rand(tokens, heads, d, device="cuda", generator=gen)).to(gate_dtype)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen).to(beta_dtype)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return q, k, gate, beta, cu


def _run(q, k, gate, beta, cu):
    from attn_gym.linear._delta_rule.warpdrive import kda_prep

    return kda_prep.kda_prep(q, k, gate, beta, cu)


def _chunks(x, rows, heads):
    """``x [T, h, ...]`` gathered per valid row-table entry as ``[R, heads, 16, ...]`` (heads
    expanded from ``h``), zero past each sequence end."""
    chunk, _, start, end = rows.long().unbind(1)
    tok = start[:, None] + chunk[:, None] * B_T + torch.arange(B_T, device=x.device)
    valid = tok < end[:, None]
    g = x[tok.clamp(max=x.shape[0] - 1)].double()
    g = g.repeat_interleave(heads // x.shape[1], dim=2)
    g = torch.where(valid.view(*valid.shape, *([1] * (g.dim() - 2))), g, 0.0)
    return g.transpose(1, 2)  # [R, heads, 16, ...]


def _reference(q, k, gate, beta, rows, round_dtype=None):
    """Records of each valid row from the module docstring's formulas in fp64; ``round_dtype``
    rounds the operands the kernel stores in 16 bits at the same points (the low-precision
    reference of the error budget)."""

    def rnd(x):
        return x if round_dtype is None else x.to(round_dtype).double()

    heads = gate.shape[1]
    qc, kc = _chunks(q, rows, heads), _chunks(k, rows, heads)
    g = torch.cumsum(_chunks(gate, rows, heads) / math.log(2), dim=2)
    b = _chunks(beta[..., None], rows, heads)[..., 0]  # [R, H, 16]
    k_decay = rnd(kc * torch.exp2(g))
    q_decay = rnd(qc * torch.exp2(g))
    k_inv = rnd(kc * torch.exp2(-g))
    k_restore = rnd(kc * torch.exp2(g[:, :, -1:] - g))
    eye = torch.eye(B_T, dtype=torch.float64, device=q.device)
    lower = torch.tril(torch.ones(B_T, B_T, dtype=torch.float64, device=q.device))
    kk = k_decay @ k_inv.transpose(-1, -2)
    L = b[..., :, None] * kk * (lower - eye)
    inv_beta = rnd(torch.linalg.inv(eye + L) * b[..., None, :])
    a_left = rnd((q_decay @ k_inv.transpose(-1, -2)) * lower)
    return {
        "k_decay": k_decay,
        "q_decay": q_decay,
        "t": inv_beta.transpose(-1, -2) @ k_restore,
        "a": a_left @ inv_beta,
        "diag": torch.exp2(g[:, :, -1]),
    }


def _decode(out, dtype):
    """The kernel's records at the valid rows, with ``a`` unpacked from its SW32 image."""
    rows = out["rows"][: int(out["row_count"][0])]
    tile_row = (rows[:, 2] // B_T + rows[:, 1] + rows[:, 0]).long()
    rec = {name: out[name][tile_row] for name in ("k_decay", "q_decay", "t", "diag")}
    row = torch.arange(B_T)[:, None]
    col = torch.arange(B_T)[None, :]
    swz = ((col // 8) ^ ((row >> 2) & 1)) * 8 + col % 8
    flat = out["a"][tile_row].view(dtype)  # [R, H, 256] 16-bit SW32 image
    rec["a"] = flat[..., (row * B_T + swz).flatten().cuda()].unflatten(-1, (B_T, B_T))
    return rows, rec


CASES = {
    # Partial, exact and one-past chunks around B_T = 16.
    "boundaries": ((1, 15, 16, 17, 63, 64, 65, 129), 4, 4, 4, 128, torch.bfloat16, "f32"),
    # Grouped q and k heads, fp16.
    "fp16-gqa": ((300, 77), 8, 4, 2, 128, torch.float16, "f32"),
    # d = 64: one key box, 16 channels per warp.
    "d64": ((500, 31), 4, 4, 4, 64, torch.bfloat16, "f32"),
    # 16-bit gate and beta (the gate exchanges through its own fp32 tile).
    "bf16-gate-beta": ((200, 9), 2, 2, 2, 128, torch.bfloat16, "io"),
    "fp16-gate-beta": ((300, 5), 4, 4, 4, 128, torch.float16, "io"),
    # Several items per CTA: the two-stage raw ring wraps its stages and phases.
    "ring-wrap": ((6000, 3000) + (7,) * 40, 4, 4, 1, 128, torch.bfloat16, "f32"),
}


@pytest.mark.parametrize("case", CASES, ids=list(CASES))
def test_kda_prep_matches_reference(case):
    lens, heads, q_heads, key_heads, d, dtype, aux = CASES[case]
    aux_dtype = torch.float32 if aux == "f32" else dtype
    q, k, gate, beta, cu = _inputs(lens, heads, q_heads, key_heads, d, dtype, aux_dtype, aux_dtype)
    out = _run(q, k, gate, beta, cu)
    rows, rec = _decode(out, dtype)
    hi = _reference(q, k, gate, beta, rows)
    lo = _reference(q, k, gate, beta, rows, round_dtype=dtype)
    for name in ("k_decay", "q_decay", "t", "a", "diag"):
        assert_matches_low_precision_reference(
            rec[name], hi[name], lo[name], name, source_dtype=dtype
        )


def test_kda_prep_ignores_the_next_sequence_past_a_ragged_tail():
    """Non-finite values in the next sequence must not reach a ragged sequence's records: its
    loads stop at its end."""
    q, k, gate, beta, cu = _inputs(
        (37, 100), 4, 4, 4, 128, torch.bfloat16, torch.float32, torch.float32
    )
    clean = _run(q, k, gate, beta, cu)
    for t in (q, k, gate, beta):
        t[37:] = float("nan")
    poisoned = _run(q, k, gate, beta, cu)
    rows = 37 // B_T + 1  # the first sequence's record rows
    for name in ("k_decay", "q_decay", "t", "a", "diag"):
        assert torch.equal(poisoned[name][:rows], clean[name][:rows]), name


def test_kda_prep_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequence count, head counts and record rows stay runtime values."""
    from attn_gym.linear._delta_rule.warpdrive import kda_prep

    shapes = [
        ((100,), 2, 2, 1),
        ((64, 64, 1), 6, 3, 3),
        ((5000, 1, 17), 16, 16, 16),
        ((16,) * 20, 1, 1, 1),
    ]
    kda_prep.compile_kda_prep.cache_clear()
    for lens, heads, q_heads, key_heads in shapes:
        f32 = torch.float32
        _run(*_inputs(lens, heads, q_heads, key_heads, 128, torch.bfloat16, f32, f32))
    info = kda_prep.compile_kda_prep.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
