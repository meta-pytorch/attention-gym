"""Warpdrive GDN prefill (``gdn_prefill``) against the eager GDN reference.

Runs on the ``[linear]`` CuTeDSL floor. The standalone wrapper builds the uncut work table on the
host; the launch bundles (``bundles``) launch the same kernel after the work-table
prologue in ``work_tables``.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive GDN prefill requires SM100 or SM103",
)

D = 128


def _inputs(lens, heads, key_heads, dtype, initial_state, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lens)

    def unit(*shape):
        x = torch.randn(*shape, device="cuda", generator=gen)
        return torch.nn.functional.normalize(x, dim=-1).to(dtype)

    q, k = unit(tokens, key_heads, D), unit(tokens, key_heads, D)
    v = torch.randn(tokens, heads, D, device="cuda", generator=gen).to(dtype)
    gate = -0.3 * torch.rand(tokens, heads, device="cuda", generator=gen)
    beta = torch.rand(tokens, heads, device="cuda", generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32).cuda()
    state = None
    if initial_state:
        state = 0.1 * torch.randn(len(lens), heads, D, D, device="cuda", generator=gen)
    return q, k, v, gate, beta, cu, state


def _reference(q, k, v, gate, beta, cu, state, final, precision):
    from attn_gym.linear import chunk_gdn

    return chunk_gdn(
        *(t[None].to(precision) for t in (q, k, v, gate, beta)),
        None if state is None else state.to(precision),
        cu_seqlens=cu,
        output_final_state=final,
        impl="reference",
    )


def _check(lens, heads, key_heads, dtype, initial_state, final, **kw):
    from attn_gym.linear._delta_rule.warpdrive.gdn_prefill import gdn_prefill
    from attn_gym.testing.kda import assert_matches_low_precision_reference

    q, k, v, gate, beta, cu, state = _inputs(lens, heads, key_heads, dtype, initial_state)
    out, out_state = gdn_prefill(
        q,
        k,
        v,
        gate,
        beta,
        cu,
        scale=D**-0.5,
        initial_state=state,
        output_final_state=final,
        **kw,
    )
    torch.cuda.synchronize()
    high = _reference(q, k, v, gate, beta, cu, state, final, torch.float64)
    low = _reference(q, k, v, gate, beta, cu, state, final, torch.float32)
    assert_matches_low_precision_reference(out, high[0][0], low[0][0], "o", source_dtype=dtype)
    if final:
        assert_matches_low_precision_reference(
            out_state, high[1], low[1], "final state", source_dtype=dtype
        )


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "dtype", "initial_state", "final", "tiles_per_head"),
    [
        # Partial, exact and one-past chunks with GQA 2, both state ends.
        ((1, 63, 64, 65, 128, 129), 4, 2, torch.bfloat16, True, True, 1),
        # No state: the first chunk runs without the KS/QS GEMMs.
        ((300, 77, 64), 4, 4, torch.float16, False, True, 1),
        # 160 items on at most 148 CTAs (ticket claims), odd chunk counts (a lone last pair
        # member) and the KQ/tinv/A rings wrapping across items.
        ((7,) * 39 + (1000,), 4, 1, torch.bfloat16, True, True, 1),
        # The d_v split: two 64-column tiles per head.
        ((1, 65, 700), 4, 2, torch.float16, True, True, 2),
        # Fewer tokens than one chunk in the whole batch (K/V boxes clip at the tensor end).
        ((1, 5, 20), 2, 1, torch.bfloat16, True, True, 1),
    ],
    ids=["boundaries-gqa2", "no-state", "many-items", "dv-split", "tiny-batch"],
)
def test_gdn_prefill_compute_matches_reference(
    lens, heads, key_heads, dtype, initial_state, final, tiles_per_head
):
    _check(lens, heads, key_heads, dtype, initial_state, final, tiles_per_head=tiles_per_head)


@pytest.mark.parametrize(
    ("lens", "heads", "key_heads", "dtype", "final"),
    [
        ((1, 63, 64, 65, 129), 4, 2, torch.bfloat16, True),
        ((2000, 7), 2, 1, torch.float16, False),
    ],
    ids=["boundaries-gqa2", "fp16-no-final"],
)
def test_gdn_prefill_gmem_matches_reference(lens, heads, key_heads, dtype, final):
    """The chain variant, fed by the warpdrive T pass's chunk-inverse tiles."""
    _check(lens, heads, key_heads, dtype, True, final, tinv_source="gmem")


def test_gdn_prefill_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequences, heads and table rows stay runtime values of one specialization."""
    from attn_gym.linear._delta_rule.warpdrive import gdn_prefill

    shapes = [((100,), 2, 1), ((64, 64, 1), 6, 3), ((5000, 1, 17), 16, 16)]
    gdn_prefill.compile_gdn_prefill.cache_clear()
    for lens, heads, key_heads in shapes:
        q, k, v, gate, beta, cu, state = _inputs(lens, heads, key_heads, torch.bfloat16, True)
        gdn_prefill.gdn_prefill(
            q, k, v, gate, beta, cu, scale=D**-0.5, initial_state=state, output_final_state=True
        )
    torch.cuda.synchronize()
    info = gdn_prefill.compile_gdn_prefill.cache_info()
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
