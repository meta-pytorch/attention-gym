"""The FA4 SM100 backward ports: dK/dV bit for bit against FA4's 1-CTA backward (the 1-CTA
kernel) or FA4's default 2-CTA backward (``cluster=CtaPair()``) on the same inputs, dQ to the TMA
reduce-add order, all three against an fp32 reference, and the wrapper's input contract. Small
shapes; the benchmark covers the large ones."""

import re
import warnings

import pytest
import torch

pytest.importorskip("cutlass.cute")
pytest.importorskip("flash_attn.cute")

from flash_attn.cute.interface import _flash_attn_bwd, _flash_attn_fwd

from attn_gym.warpdrive import (
    Cluster,
    CtaPair,
    RoleConfig,
    StaticPersistent,
    WorkStream,
)
from attn_gym.warpdrive.examples.fa4_bwd import (
    FlashAttentionBackward,
    FlashAttentionBackward2Cta,
    compile_fa4_bwd,
    fa4_bwd,
    fa4_config,
)

D = 128


def inputs(b, h, sq, sk, scale, causal, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(b, sq, h, D, device="cuda", dtype=torch.bfloat16)
    k, v = (torch.randn(b, sk, h, D, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    dout = torch.randn_like(q)
    out, lse = _flash_attn_fwd(q, k, v, softmax_scale=scale, causal=causal, return_lse=True)[:2]
    return q, k, v, out, dout, lse


def reference(q, k, v, dout, scale, causal):
    """fp32 attention backward through autograd (rows without keys give zero gradients)."""
    qf, kf, vf = (t.float().transpose(1, 2).requires_grad_() for t in (q, k, v))
    s = qf @ kf.transpose(-1, -2) * scale
    if causal:
        sq, sk = s.shape[-2:]
        keep = torch.ones(sq, sk, dtype=torch.bool, device=s.device).tril(sk - sq)
        s = s.masked_fill(~keep, float("-inf"))
    (s.softmax(-1).nan_to_num() @ vf).backward(dout.float().transpose(1, 2))
    return [t.grad.transpose(1, 2) for t in (qf, kf, vf)]


SHAPES = [
    (1, 2, 128, 128),  # one Q tile: no pipelined loop iteration; the pair's peer has no keys
    (1, 2, 384, 256),  # Q ring wraps; two key tiles
    (2, 3, 200, 330),  # ragged Q and K tails
    (1, 2, 1000, 64),  # one partial key tile, eight Q tiles
    (1, 2, 640, 520),  # three CTA pairs; the last one's peer has no keys
    (1, 2, 640, 450),  # two CTA pairs; the last one's peer key tile is partial
]
CLUSTERS = [None, CtaPair()]


def cluster_id(cluster):
    return "pair" if cluster else "single"


@pytest.mark.parametrize("cluster", CLUSTERS, ids=cluster_id)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_matches_fa4_and_fp32(requires_blackwell, shape, causal, cluster):
    scale = 0.3  # not the default 1/sqrt(d)
    q, k, v, out, dout, lse = inputs(*shape, scale, causal)
    ours = fa4_bwd(q, k, v, out, dout, lse, scale, causal=causal, cluster=cluster)
    # The 1-CTA kernel reproduces FA4's disable_2cta configuration, the pair FA4's default.
    config = fa4_config(q, k, causal, disable_2cta=cluster is None)
    theirs = _flash_attn_bwd(q, k, v, out, dout, lse, scale, causal, config=config)[:3]
    ref = reference(q, k, v, dout, scale, causal)
    assert torch.equal(ours[1], theirs[1]) and torch.equal(ours[2], theirs[2])
    for name, a, b, r in zip(("dq", "dk", "dv"), ours, theirs, ref):
        assert torch.isfinite(a).all(), name
        err, fa4_err = ((t.float() - r).abs().max().item() for t in (a, b))
        # dQ adds the tiles' partial sums in TMA reduce order; within FA4's own error.
        assert err <= 1.1 * fa4_err + 2e-3 * r.abs().max().item(), (name, err, fa4_err)


@pytest.mark.parametrize("cluster", CLUSTERS, ids=cluster_id)
def test_repeated_launches_and_shapes(requires_blackwell, cluster):
    """One compiled kernel across shapes; dK/dV are deterministic, dQ is not bitwise (the
    reduce-adds of different key tiles race), but stays within reduce-order rounding."""
    scale = D**-0.5
    fa4_bwd(*inputs(1, 1, 128, 128, scale, False), scale, cluster=cluster)
    before = compile_fa4_bwd.cache_info()
    for shape in [(1, 2, 256, 384), (2, 1, 130, 129)]:
        args = inputs(*shape, scale, False)
        first, again = (fa4_bwd(*args, scale, cluster=cluster) for _ in range(2))
        assert torch.equal(first[1], again[1]) and torch.equal(first[2], again[2])
        assert torch.allclose(first[0].float(), again[0].float(), atol=2e-3, rtol=0)
    after = compile_fa4_bwd.cache_info()
    assert after.currsize == before.currsize
    assert after.hits + after.misses == before.hits + before.misses + 4


def test_one_dq_staging_buffer(requires_blackwell):
    """One dQ staging buffer computes the same dK/dV bits (dQ: the same tile sums)."""
    scale = D**-0.5
    args = inputs(2, 2, 300, 260, scale, False)
    base, knob = fa4_bwd(*args, scale), fa4_bwd(*args, scale, dq_stages=1)
    assert torch.equal(base[1], knob[1]) and torch.equal(base[2], knob[2])
    assert torch.allclose(base[0].float(), knob[0].float(), atol=2e-3, rtol=0)
    with pytest.raises(ValueError, match="dq_stages"):
        FlashAttentionBackward(dq_stages=4)


# Every role off its default warps, without budgets: reduce leaves warp 0 and starts mid
# warpgroup, compute (warps 6-13) straddles three warpgroups.
MOVED_ROLES = {
    "load": RoleConfig(0),
    "mma": RoleConfig(1),
    "reduce": RoleConfig(range(2, 6)),
    "compute": RoleConfig(range(6, 14)),
}


@pytest.mark.parametrize("cluster", CLUSTERS, ids=cluster_id)
def test_moved_roles_match_defaults(requires_blackwell, cluster):
    """The TMEM rows follow the hardware warp quadrant and the other indices the role, so the
    moved kernel computes the default's dK/dV bits (dQ: the same tile sums)."""
    scale = D**-0.5
    args = inputs(1, 2, 384, 256, scale, True)
    roles = MOVED_ROLES
    if cluster is not None:
        roles = {**roles, "mma": RoleConfig(1, cta="leader"), "relay": RoleConfig(14)}
    base = fa4_bwd(*args, scale, causal=True, cluster=cluster)
    moved = fa4_bwd(*args, scale, causal=True, cluster=cluster, roles=roles)
    assert torch.equal(base[1], moved[1]) and torch.equal(base[2], moved[2])
    assert torch.allclose(base[0].float(), moved[0].float(), atol=2e-3, rtol=0)


def test_describe_shows_the_derived_protocol():
    op = FlashAttentionBackward()
    text = op.describe()
    for line in (
        "0-3    reduce      152",
        "4-11   compute     136",
        "tmem P   [0, 64) (inside S)",
        "tmem dQ  [256, 384) (inside dP)",
    ):
        assert line in text, text
    assert op.s.scope.empty_counts == {0: [8]} and op.dq.scope.empty_counts == {0: [4]}
    causal = FlashAttentionBackward(causal=True).describe()
    assert "0-3    reduce      136" in causal and "4-11   compute     144" in causal


def test_describe_shows_the_pair_protocol():
    """The pair: FA4's 2-CTA register split, the leader-only MMA, dQ over S's tail, and the dS
    handoff joining both CTAs' compute warps and relay warps on the leader's barrier."""
    op = FlashAttentionBackward2Cta(causal=True)
    text = op.describe()
    for line in (
        "0-3    reduce      136",
        "4-11   compute     136",
        "12     mma         104  leader",
        "14     relay       104  all",
        "tmem dQ  [64, 128) (inside S)",
        "xchg  1 mbarriers",
    ):
        assert line in text, text
    assert op.ds.scope.full_counts[0] == [2 * (8 + 1)]  # compute and relay warps of both CTAs
    assert op.s.scope.empty_counts[0] == [16] and op.dq.scope.empty_counts[0] == [8]
    assert op.k.scope.tx == 2 * 128 * D * 2  # the leader arms both CTAs' halves


def test_pair_traced_diagnostics():
    """The pair compiles without warpdrive warnings and every TMA load is verified (the alias
    diagnostic does not model CTA pairs)."""
    op = FlashAttentionBackward2Cta()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_fa4_bwd(op)
    assert [str(w.message) for w in caught if "warpdrive" in str(w.message)] == []
    for channel in ("k", "kt", "v", "q", "qt", "do"):
        assert op._ws_tma[channel].status().startswith("verified CtaGroup.TWO"), channel
    for channel in ("lse", "dpsum"):
        assert op._ws_tma[channel].status().startswith("verified cp.async.bulk, 512 B/stage")


def test_traced_diagnostics_are_quiet():
    """The port compiles without warpdrive diagnostic warnings: the LSE/dPsum bulk copies are
    verified, and the TMEM alias pairs the MMA's peeled last tile leaves unmatched are
    reported as unknown epoch correspondence (unverified), not as potential hazards."""
    op = FlashAttentionBackward()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_fa4_bwd(op)
    assert [str(w.message) for w in caught if "warpdrive" in str(w.message)] == []
    text = op.describe()
    for channel in ("lse", "dpsum"):
        assert op._ws_tma[channel].status().startswith("verified cp.async.bulk, 512 B/stage")
    alias = text[text.index("Alias diagnostic") :]
    assert "potential hazard" not in alias.split("\n", 1)[1]
    for pair in ("P ~ S", "dS ~ dP", "dQ ~ dP", "dQ ~ dS"):
        assert re.search(rf"\n  {re.escape(pair)} \(tmem\)  unverified\n", alias), alias
    assert "unknown epoch correspondence" in alias
    assert "[tmem_store]" not in alias  # the ds commit now declares both fences


def test_rejects_unsupported_inputs(requires_blackwell):
    q, k, v, out, dout, lse = inputs(1, 2, 128, 128, D**-0.5, False)
    with pytest.raises(ValueError, match="expected"):
        fa4_bwd(q[..., :64], k[..., :64], v[..., :64], out[..., :64], dout[..., :64], lse)
    with pytest.raises(ValueError, match="no GQA"):
        fa4_bwd(q, k[:, :, :1], v[:, :, :1], out, dout, lse)
    with pytest.raises(TypeError, match="bf16"):
        fa4_bwd(q.half(), k, v, out, dout, lse)
    with pytest.raises(ValueError, match="contiguous"):
        fa4_bwd(q.transpose(1, 2).contiguous().transpose(1, 2), k, v, out, dout, lse)
    with pytest.raises(ValueError, match="lse"):
        fa4_bwd(q, k, v, out, dout, lse.transpose(1, 2))
    with pytest.raises(ValueError, match="dq_stages"):
        fa4_bwd(q, k, v, out, dout, lse, cluster=CtaPair(), dq_stages=2)
    with pytest.raises(ValueError, match="one CTA or a CTA pair"):
        fa4_bwd(q, k, v, out, dout, lse, cluster=Cluster(2, 2, pair=True))
    with pytest.raises(ValueError, match="one work item"):
        op = FlashAttentionBackward()
        op.work = WorkStream(schedule=StaticPersistent())
        compile_fa4_bwd(op)
    empty = torch.empty(1, 0, 2, D, device="cuda", dtype=torch.bfloat16)
    dq, dk, dv = fa4_bwd(empty, k, v, empty, empty, lse[..., :0])
    assert dq.shape == empty.shape and not dk.any() and not dv.any()
