"""Warpdrive prep-fed KDA prefill (``frost.kda_prep_prefill``) against the fp64 recurrent oracle.

The prep records are built here in torch from the upstream ``kda_prep_f16`` definitions, so the
test needs no prep kernel. Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import math

import pytest
import torch

from attn_gym.testing.kda import assert_matches_low_precision_reference, kda_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive KDA prep prefill requires SM100 or SM103",
)

B_T = 16


def _inputs(lens, heads, key_heads, d, dtype, d_v=128, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    q = torch.nn.functional.normalize(normal(tokens, heads, d), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(normal(tokens, key_heads, d), dim=-1).to(dtype)
    v = normal(tokens, heads, d_v).to(dtype)
    gate = -0.3 * torch.rand(tokens, heads, d, device="cuda", generator=gen)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return q, k, v, gate, beta, cu


def _swizzle32(rows, cols):
    """Element offset of ``(row, col)`` in a 32-byte swizzled 16 x 16 16-bit tile."""
    return rows * B_T + ((cols // 8) ^ ((rows >> 2) & 1)) * 8 + cols % 8


def _records(q, k, gate, beta, cu):
    """``(k_decay, q_decay, t, a, diag)`` of ``kda_prep_f16`` in fp32 math, rows past a sequence
    end zero (sequence ``b`` chunk ``j`` at row ``cu[b] // 16 + b + j``)."""
    dtype, heads, d = q.dtype, gate.shape[1], q.shape[-1]
    k = k.repeat_interleave(heads // k.shape[1], dim=1)
    num_seqs = cu.shape[0] - 1
    rows = q.shape[0] // B_T + num_seqs
    out = [torch.zeros(rows, heads, B_T, d, dtype=dtype, device="cuda") for _ in range(3)]
    a_tiles = torch.zeros(rows, heads, B_T * B_T, dtype=dtype, device="cuda")
    diag = torch.zeros(rows, heads, d, device="cuda")
    r, c = torch.meshgrid(torch.arange(B_T), torch.arange(B_T), indexing="ij")
    a_index = _swizzle32(r, c).flatten().cuda()
    eye = torch.eye(B_T, device="cuda")
    for b in range(num_seqs):
        start, end = int(cu[b]), int(cu[b + 1])
        chunks = math.ceil((end - start) / B_T)
        if not chunks:
            continue
        pad = chunks * B_T - (end - start)

        def chunked(x, s=start, e=end, p=pad, n=chunks):
            x = torch.nn.functional.pad(x[s:e].float(), (0, 0) * (x.dim() - 1) + (0, p))
            return x.reshape(n, B_T, *x.shape[1:]).transpose(1, 2)  # (chunk, head, t, ...)

        qc, kc, gc, bc = chunked(q), chunked(k), chunked(gate), chunked(beta)
        g = torch.cumsum(gc * math.log2(math.e), dim=2)
        k_decay = kc * torch.exp2(g)
        k_inv = kc * torch.exp2(-g)
        k_restore = kc * torch.exp2(g[:, :, -1:] - g)
        lower = torch.tril(k_decay @ k_inv.transpose(-1, -2), -1) * bc[..., None]
        inv_beta = torch.linalg.inv(eye + lower) * bc[..., None, :]
        a_full = torch.tril(qc * torch.exp2(g) @ k_inv.transpose(-1, -2)) @ inv_beta
        row0 = start // B_T + b
        rows_b = slice(row0, row0 + chunks)
        out[0][rows_b] = k_decay.to(dtype)
        out[1][rows_b] = (qc * torch.exp2(g)).to(dtype)
        out[2][rows_b] = (inv_beta.transpose(-1, -2) @ k_restore).to(dtype)
        a_tiles[rows_b, :, a_index] = a_full.flatten(-2).to(dtype)
        diag[rows_b] = torch.exp2(g[:, :, -1])
    return (*out, a_tiles.view(torch.int32), diag)


def _reference(q, k, v, gate, beta, cu, state, dtype):
    """The packed recurrent oracle in ``dtype`` (k heads expanded to the gate heads)."""
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    args = (t[None].to(dtype) for t in (q, k, v, gate, beta))
    o, final = kda_reference(
        *args,
        None if state is None else state.to(dtype),
        cu_seqlens=cu,
        scale=q.shape[-1] ** -0.5,
        output_final_state=True,
    )
    return o[0], final


def _run(v, records, cu, scale, state=None, final=False):
    from attn_gym.linear._delta_rule.frost import kda_prep_prefill

    return kda_prep_prefill.kda_prep_prefill(
        v, *records, cu, scale=scale, initial_state=state, output_final_state=final
    )


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "d", "dtype", "state_dtype"),
    [
        # Partial, exact and one-past chunks around B_T = 16 and the 8-stage raw ring.
        ((1, 15, 16, 17, 63, 64, 65, 129), 4, 4, 128, torch.bfloat16, None),
        # Grouped keys, fp16, carried fp32 state in and out.
        ((300, 77), 8, 2, 128, torch.float16, torch.float32),
        # d_k = 64 (one key half) with a bf16 state.
        ((500, 31), 4, 4, 64, torch.bfloat16, torch.bfloat16),
        # Many chunks per CTA: every ring wraps its stages and phases; more items than CTAs.
        ((3000, 1000) + (7,) * 40, 2, 1, 128, torch.bfloat16, torch.float32),
    ],
    ids=["boundaries", "fp16-gqa-state", "d64-bf16-state", "ring-wrap"],
)
def test_kda_prep_prefill_matches_reference(lens, heads, key_heads, d, dtype, state_dtype):
    q, k, v, gate, beta, cu = _inputs(lens, heads, key_heads, d, dtype)
    state = None
    if state_dtype is not None:
        gen = torch.Generator(device="cuda").manual_seed(7)
        state = 0.1 * torch.randn(len(lens), heads, 128, d, device="cuda", generator=gen)
        state = state.to(state_dtype)
    o, final = _run(v, _records(q, k, gate, beta, cu), cu, d**-0.5, state, state is not None)
    hi_o, hi_s = _reference(q, k, v, gate, beta, cu, state, torch.float64)
    lo_o, lo_s = _reference(q, k, v, gate, beta, cu, state, torch.float32)
    assert_matches_low_precision_reference(o, hi_o, lo_o, "o", source_dtype=dtype)
    if state is not None:
        assert_matches_low_precision_reference(
            final.float(), hi_s, lo_s, "final", source_dtype=dtype
        )


@pytest.mark.parametrize("with_state", [False, True], ids=["fresh", "seeded"])
def test_kda_prep_prefill_reuses_ctas_across_empty_and_unseeded_items(with_state):
    """More items than CTAs with empty sequences between them: each CTA runs several items, so
    an unseeded item starts while its predecessor's last O GEMM may still read the U operand,
    and empty items must leave every barrier phase untouched; final states come out with and
    without a seed (empty sequences pass theirs through)."""
    lens = (0, 50, 0, 33, 16, 0, 200, 1, 0, 17) * 2
    heads = 8
    q, k, v, gate, beta, cu = _inputs(lens, heads, heads, 128, torch.bfloat16, seed=3)
    state = None
    if with_state:
        gen = torch.Generator(device="cuda").manual_seed(9)
        state = 0.1 * torch.randn(len(lens), heads, 128, 128, device="cuda", generator=gen)
    o, final = _run(v, _records(q, k, gate, beta, cu), cu, 128**-0.5, state, final=True)
    hi_o, hi_s = _reference(q, k, v, gate, beta, cu, state, torch.float64)
    lo_o, lo_s = _reference(q, k, v, gate, beta, cu, state, torch.float32)
    assert_matches_low_precision_reference(o, hi_o, lo_o, "o", source_dtype=torch.bfloat16)
    assert_matches_low_precision_reference(
        final.float(), hi_s, lo_s, "final", source_dtype=torch.bfloat16
    )


def test_kda_prep_prefill_ignores_the_next_sequence_past_a_ragged_tail():
    """Non-finite V in the next sequence must not reach a ragged sequence: its loads stop at
    its end."""
    q, k, v, gate, beta, cu = _inputs((37, 100), 4, 4, 128, torch.bfloat16)
    records = _records(q, k, gate, beta, cu)
    clean, _ = _run(v, records, cu, 128**-0.5)
    v[37:] = float("nan")
    poisoned, _ = _run(v, records, cu, 128**-0.5)
    assert torch.equal(poisoned[:37], clean[:37])


def test_kda_prep_prefill_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequence count, head counts, record rows and work rows stay runtime values."""
    from attn_gym.linear._delta_rule.frost import kda_prep_prefill

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16), ((16,) * 20, 1, 1)]
    kda_prep_prefill.compile_kda_prep_prefill.cache_clear()
    for lens, heads, key_heads in shapes:
        q, k, v, gate, beta, cu = _inputs(lens, heads, key_heads, 128, torch.bfloat16)
        _run(v, _records(q, k, gate, beta, cu), cu, 128**-0.5)
    info = kda_prep_prefill.compile_kda_prep_prefill.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
