"""Warpdrive KDA prefill forward (``kda_prefill``) against the fp64 recurrent KDA oracle.

Runs on the ``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.testing.kda import assert_matches_low_precision_reference, kda_reference

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA prefill requires an SM100-family GPU",
)


def _inputs(lens, heads, key_heads, d, dtype, gate_dtype=torch.float32, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    q = torch.nn.functional.normalize(normal(tokens, heads, d), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(normal(tokens, key_heads, d), dim=-1).to(dtype)
    v = normal(tokens, heads, d).to(dtype)
    gate = (-0.3 * torch.rand(tokens, heads, d, device="cuda", generator=gen)).to(gate_dtype)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    return q, k, v, gate, beta, cu


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


def _run(q, k, v, gate, beta, cu, state=None, final=False, tiles_per_head=1):
    from attn_gym.linear._delta_rule.warpdrive import kda_prefill

    return kda_prefill.kda_prefill(
        q,
        k,
        v,
        gate,
        beta,
        cu,
        scale=q.shape[-1] ** -0.5,
        initial_state=state,
        output_final_state=final,
        tiles_per_head=tiles_per_head,
    )


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "d", "dtype", "with_state", "tiles", "gate_dtype"),
    [
        # Partial, exact and one-past chunks around B_T = 16 and the 8-stage raw ring.
        ((1, 15, 16, 17, 63, 64, 65, 129), 4, 4, 128, torch.bfloat16, False, 1, torch.float32),
        # Grouped keys, fp16, carried state in and out.
        ((300, 77), 8, 2, 128, torch.float16, True, 1, torch.float32),
        # d = 64 (one key half, M = 64 value tiles) with state.
        ((500, 31), 4, 4, 64, torch.bfloat16, True, 1, torch.float32),
        # The d_v split (two 64-column value tiles per head) and a 16-bit gate.
        ((200, 9), 2, 2, 128, torch.bfloat16, True, 2, torch.bfloat16),
        # Many chunks per CTA: every ring wraps its stages and phases; more items than CTAs.
        ((3000, 1000) + (7,) * 40, 2, 1, 128, torch.bfloat16, False, 1, torch.float32),
    ],
    ids=["boundaries", "fp16-gqa-state", "d64-state", "dv-split-bf16gate", "ring-wrap"],
)
def test_kda_prefill_matches_reference(
    lens, heads, key_heads, d, dtype, with_state, tiles, gate_dtype
):
    q, k, v, gate, beta, cu = _inputs(lens, heads, key_heads, d, dtype, gate_dtype)
    state = None
    if with_state:
        gen = torch.Generator(device="cuda").manual_seed(7)
        state = 0.1 * torch.randn(len(lens), heads, d, d, device="cuda", generator=gen)
    o, final = _run(q, k, v, gate, beta, cu, state, final=with_state, tiles_per_head=tiles)
    hi_o, hi_s = _reference(q, k, v, gate.float(), beta, cu, state, torch.float64)
    lo_o, lo_s = _reference(q, k, v, gate.float(), beta, cu, state, torch.float32)
    assert_matches_low_precision_reference(o, hi_o, lo_o, "o", source_dtype=dtype)
    if with_state:
        assert_matches_low_precision_reference(final, hi_s, lo_s, "final", source_dtype=dtype)


def test_kda_prefill_ignores_the_next_sequence_past_a_ragged_tail():
    """Non-finite values in the next sequence must not reach a ragged sequence: its loads
    stop at its end."""
    q, k, v, gate, beta, cu = _inputs((37, 100), 4, 4, 128, torch.bfloat16)
    clean, _ = _run(q, k, v, gate, beta, cu)
    for t in (q, k, v, gate):
        t[37:] = float("nan")
    poisoned, _ = _run(q, k, v, gate, beta, cu)
    assert torch.equal(poisoned[:37], clean[:37])


def test_kda_prefill_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequence count, head counts and work rows stay runtime values."""
    from attn_gym.linear._delta_rule.warpdrive import kda_prefill

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16), ((16,) * 20, 1, 1)]
    kda_prefill.compile_kda_prefill.cache_clear()
    for lens, heads, key_heads in shapes:
        _run(*_inputs(lens, heads, key_heads, 128, torch.bfloat16))
    info = kda_prefill.compile_kda_prefill.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
