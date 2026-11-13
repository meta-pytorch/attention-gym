"""Host-path contracts of the warpdrive GDN driver: what a warm call may skip.

Counts calls instead of timing them, so the checks cannot flake on a loaded host.
"""

import collections

import pytest
import torch

from attn_gym.linear import chunk_gdn
from attn_gym.linear._delta_rule.warpdrive import contract
from attn_gym.linear._delta_rule.warpdrive.bundles import gdn_warmup_backward
from attn_gym.linear._delta_rule.warpdrive.drivers import gdn as gdn_driver
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.warpdrive.cache import jit as cute_cache

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive backend requires SM100 or SM103",
)

_WARPDRIVE = {"backend": "warpdrive"}


def _gdn_step(lengths, kernel_options=_WARPDRIVE):
    q, k, value, gate, beta, _, cu_seqlens = make_gdn_test_inputs(
        lengths, key_heads=2, value_heads=2, seed=907
    )
    leaves = tuple(t.requires_grad_() for t in (q, k, value, gate, beta))
    output, _ = chunk_gdn(*leaves, cu_seqlens=cu_seqlens, kernel_options=kernel_options)
    return (output, *torch.autograd.grad(output, leaves, torch.ones_like(output)))


def _count(monkeypatch, calls, module, name):
    original = getattr(module, name)

    def counted(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)


def test_warm_gdn_call_validates_once_and_skips_full_cache_key(monkeypatch):
    lengths = (97, 0, 161)
    _gdn_step(lengths)  # compile or load every launch
    gdn_warmup_backward._validate_launch.cache_clear()
    calls = collections.Counter()
    _count(monkeypatch, calls, contract, "validate_gdn_bwd_operands")
    # jit_cache's full argument-key builds, the path a warm launch must skip.
    _count(monkeypatch, calls, cute_cache, "_fast_key")
    _count(monkeypatch, calls, cute_cache, "_make_runtime_key")

    _gdn_step(lengths)
    # A new metadata signature runs the backward bundle checks exactly once.
    assert calls["validate_gdn_bwd_operands"] == 1
    calls.clear()
    _gdn_step(lengths)
    assert calls == {}


def test_gdn_chain_changing_shapes_matches_default(monkeypatch):
    """The chain launches are keyed without the batch shape; a later shape must still get its
    own plan, piece geometry and validation instead of a replay of the previous one."""
    monkeypatch.setattr(gdn_driver, "MIN_CHAIN_TOKENS_PER_PIECE_FWD", 0)
    monkeypatch.setattr(gdn_driver, "MIN_CHAIN_TOKENS_PER_PIECE_BWD", 0)
    device = torch.device("cuda")
    # 8, then 4 pieces per sequence, then the first shape again.
    for lengths in ((2048,), (900, 1148), (2048,)):
        tokens = sum(lengths)
        assert gdn_driver.ForwardPlan.build(tokens, len(lengths), 2, 128, device).chain
        assert gdn_driver.BackwardPlan.build(tokens, len(lengths), 2, device).chain
        actual = _gdn_step(lengths)
        expected = _gdn_step(lengths, kernel_options=None)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
