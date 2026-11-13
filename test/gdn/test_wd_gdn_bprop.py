"""Warpdrive GDN backward (``gdn_bprop``) against fp64 autograd of eager GDN.

The checkpoints come from the warpdrive T pass and recompute, so the whole chain runs on the
``[linear]`` CuTeDSL floor.
"""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear import chunk_gdn
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.kda import assert_matches_low_precision_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive GDN backward requires SM100 or SM103",
)

B_T = 64
NAMES = ("dq", "dk", "dv", "dgate", "dbeta", "d_initial_state")


def _table(lens, heads, cu):
    """Uncut work table: one item per (sequence, head) over all of its chunks, writing its
    state gradient to row ``b`` of ``dstate0``."""
    rows = []
    for b, length in enumerate(lens):
        n = (length + B_T - 1) // B_T
        starts = cu[b].item(), cu[b + 1].item()
        rows += [[b, h, 0, n, 0, n, *starts, b, b] for h in range(heads) if n]
    return torch.tensor(rows, dtype=torch.int32, device="cuda").reshape(-1, 10)


def _wd_backward(q, k, v, gate, beta, do, cu, lens, state, d_final, scale, num_sm=None, op=None):
    """T pass -> checkpoint recompute -> bprop for packed ``[T, H, D]`` operands; dq/dk are
    reduced over each key head's value-head group as the drivers do."""
    from attn_gym.linear._delta_rule.warpdrive import gdn_bprop, gdn_recompute, gdn_tinv

    tokens, heads, d = v.shape
    beta = beta.float()
    rows = gdn_tinv.tinv_rows(tokens, len(lens))
    tinv = torch.empty(rows, heads, B_T, B_T, dtype=k.dtype, device="cuda")
    tinv_table = torch.empty(rows, 4, dtype=torch.int32, device="cuda")
    tinv_count = torch.empty(1, dtype=torch.int32, device="cuda")
    gdn_tinv.gdn_tinv(k, gate, beta, cu, tinv, tinv_table, tinv_count)

    items = _table(lens, heads, cu)
    count = torch.tensor([items.shape[0]], dtype=torch.int32, device="cuda")
    ckpt = torch.empty(
        gdn_recompute.checkpoint_rows(tokens, len(lens)), heads, d, d, dtype=k.dtype, device="cuda"
    )
    gdn_recompute.gdn_recompute(k, v, gate, tinv, items, count, ckpt, state_in=state)

    dq, dk = (torch.empty(tokens, heads, d, dtype=q.dtype, device="cuda") for _ in range(2))
    dv = torch.empty_like(v)
    dgate, dbeta = (torch.empty(tokens, heads, device="cuda") for _ in range(2))
    dstate0 = None
    if state is not None:
        dstate0 = torch.zeros_like(state) if d_final is None else d_final.clone()
    gdn_bprop.gdn_bprop(
        q, k, v, gate, beta, do, ckpt, items, count, dq, dk, dv, dgate, dbeta,
        scale=scale, use_initial_state=state is not None, dstate_in=d_final, dstate0=dstate0,
        num_sm=num_sm, op=op,
    )  # fmt: skip
    group = heads // q.shape[1]
    dq, dk = (x.float().unflatten(1, (-1, group)).sum(2).to(q.dtype) for x in (dq, dk))
    return dq, dk, dv, dgate, dbeta, dstate0


def _reference(inputs, do, state, d_final, scale, precision):
    """Autograd of eager GDN at ``precision`` (None: the source dtypes)."""
    q, k, v, gate, beta, cu = inputs
    cast = (lambda x: x) if precision is None else (lambda x: x.to(precision))
    operands = [cast(x.detach()).requires_grad_() for x in (q, k, v, gate, beta)]
    s = None if state is None else cast(state.detach()).requires_grad_()
    out, final = chunk_gdn(
        *(x[None] for x in operands),
        s,
        cu_seqlens=cu,
        output_final_state=d_final is not None,
        scale=scale,
        impl="reference",
    )
    outputs, cotangents = [out], [cast(do)[None]]
    if d_final is not None:
        outputs.append(final)
        cotangents.append(d_final.to(final.dtype))
    targets = operands if s is None else [*operands, s]
    return torch.autograd.grad(outputs, targets, cotangents)


@pytest.mark.parametrize(
    ("lens", "dtype", "d", "key_heads", "heads", "state_mode", "num_sm"),
    [
        # Partial, exact and one-past chunks with an odd chunk count; GQA 2.
        ((1, 63, 64, 65, 129), torch.bfloat16, 128, 1, 2, "none", None),
        # fp16, d 64, initial state (chunk 0 reads its checkpoint).
        ((65, 200), torch.float16, 64, 2, 2, "initial", None),
        # Initial state and a final-state cotangent seeding dS; GQA 2, d 64.
        ((63, 191), torch.bfloat16, 64, 1, 2, "cotangent", None),
        # fp16, d 128, final-state cotangent.
        ((130, 64), torch.float16, 128, 2, 4, "cotangent", None),
        # More items than CTAs: scheduler claims and the pinned partial last wave.
        ((7,) * 20 + (300,), torch.bfloat16, 128, 2, 4, "none", 5),
        # One long item: every ring wraps its stages and phases.
        ((1500,), torch.bfloat16, 128, 1, 1, "initial", None),
    ],
    ids=["bounds-gqa2", "fp16-d64-init", "d64-cotangent-gqa2", "fp16-cotangent", "claims", "long"],
)
def test_gdn_bprop_matches_reference(lens, dtype, d, key_heads, heads, state_mode, num_sm):
    q, k, v, gate, beta, state, cu = make_gdn_test_inputs(
        lens, key_heads=key_heads, value_heads=heads, key_dim=d, dtype=dtype, seed=len(lens)
    )
    q, k, v, gate, beta = (x[0] for x in (q, k, v, gate, beta))
    state = None if state_mode == "none" else state
    gen = torch.Generator(device="cuda").manual_seed(1)
    do = torch.randn(v.shape, device="cuda", generator=gen).to(dtype)
    d_final = None
    if state_mode == "cotangent":
        d_final = torch.randn(state.shape, device="cuda", generator=gen)
    scale = d**-0.5
    inputs = (q, k, v, gate, beta, cu)

    actual = _wd_backward(q, k, v, gate, beta, do, cu, lens, state, d_final, scale, num_sm)
    high = _reference(inputs, do, state, d_final, scale, torch.float64)
    source = _reference(inputs, do, state, d_final, scale, None)
    for name, got, ref_high, ref_source in zip(NAMES, actual, high, source, strict=False):
        assert_matches_low_precision_reference(got, ref_high, ref_source, name, source_dtype=dtype)


def test_gdn_bprop_column_fold_waits_for_slow_reader():
    """cg0 folds the dGate column partials through sKK scratch that warp 0 overwrites with the
    next chunk's L tile; a reader warp that falls behind must still read the partials. The
    delay changes no arithmetic, so every output must match the undelayed kernel bit for bit."""
    from attn_gym.linear._delta_rule.warpdrive import gdn_bprop

    lens = (300, 129)
    q, k, v, gate, beta, _, cu = make_gdn_test_inputs(lens, value_heads=2, seed=3)
    q, k, v, gate, beta = (x[0] for x in (q, k, v, gate, beta))
    do = torch.randn_like(v)
    args = (q, k, v, gate, beta, do, cu, lens, None, None, 128**-0.5)
    delayed = gdn_bprop.GdnBprop("bf16", 128, False, False, False, column_fold_delay=20000)
    expected = _wd_backward(*args)
    actual = _wd_backward(*args, op=delayed)
    for name, got, want in zip(NAMES[:5], actual[:5], expected[:5], strict=True):
        assert torch.equal(got, want), name


@pytest.mark.parametrize("state_mode", ["none", "cotangent"])
def test_gdn_bprop_short_first_sequence_ignores_next_rows(state_mode):
    """A sequence ending before token 64 (where tail views clip only at token 64, so the item
    gets its own descriptor) must give the gradients of the sequence run alone, bit for bit."""
    lens = (5, 300)
    q, k, v, gate, beta, state, cu = make_gdn_test_inputs(lens, value_heads=2, seed=5)
    q, k, v, gate, beta = (x[0] for x in (q, k, v, gate, beta))
    gen = torch.Generator(device="cuda").manual_seed(2)
    do = torch.randn(v.shape, device="cuda", generator=gen).to(v.dtype)
    d_final = None
    if state_mode == "cotangent":
        d_final = torch.randn(state.shape, device="cuda", generator=gen)
    else:
        state = None
    packed = _wd_backward(q, k, v, gate, beta, do, cu, lens, state, d_final, 128**-0.5)
    n = lens[0]
    alone = _wd_backward(
        *(x[:n] for x in (q, k, v, gate, beta, do)),
        cu[:2],
        lens[:1],
        None if state is None else state[:1],
        None if d_final is None else d_final[:1],
        128**-0.5,
    )
    for name, got, want in zip(NAMES[:5], packed[:5], alone[:5], strict=True):
        assert torch.equal(got[:n], want), name
    if state is not None:
        assert torch.equal(packed[5][:1], alone[5]), NAMES[5]


def test_gdn_bprop_reuses_one_compile_across_runtime_shapes():
    """Tokens, sequence count, head counts, item count and grid stay runtime values of one
    specialization."""
    from attn_gym.linear._delta_rule.warpdrive import gdn_bprop

    shapes = [((100,), 1, 2, None), ((1, 64, 65), 2, 6, 3), ((7,) * 33 + (640,), 4, 4, 16)]
    gdn_bprop.compile_gdn_bprop.cache_clear()
    gdn_bprop._compiled.cache_clear()
    for lens, key_heads, heads, num_sm in shapes:
        q, k, v, gate, beta, _, cu = make_gdn_test_inputs(
            lens, key_heads=key_heads, value_heads=heads, dtype=torch.bfloat16
        )
        q, k, v, gate, beta = (x[0] for x in (q, k, v, gate, beta))
        _wd_backward(q, k, v, gate, beta, torch.randn_like(v), cu, lens, None, None, 0.1, num_sm)
    assert gdn_bprop.compile_gdn_bprop.cache_info().currsize == 1
    info = gdn_bprop._compiled.cache_info()  # the wrapper's per-configuration entry
    assert info.currsize == 1, info
    assert info.hits + info.misses == len(shapes), info
