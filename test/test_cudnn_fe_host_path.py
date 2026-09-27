"""Host-path contracts of the cuDNN v1.30 GDN/KDA drivers: what a warm call may skip.

Counts calls instead of timing them, so the checks cannot flake on a loaded host.
"""

import collections

import pytest
import torch

pytest.importorskip("cutlass.experimental")

from attn_gym._backends.cute import cache as cute_cache
from attn_gym.linear import chunk_gdn, chunk_kda
from attn_gym.linear._delta_rule.cudnn_fe import gdn as fe_gdn
from attn_gym.linear._delta_rule.cudnn_fe.kernel import (
    gdn_bprop_f16,
    gdn_warmup_backward_f16,
    kda_prefill_f16,
    kda_warmup_forward_f16,
)
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the cuDNN v1.30 kernels require SM100 or SM103",
)

_CUDNN = {"backend": "cudnn"}


def _gdn_step(lengths, kernel_options=_CUDNN):
    q, k, value, gate, beta, _, cu_seqlens = make_gdn_test_inputs(
        lengths, key_heads=2, value_heads=2, seed=907
    )
    leaves = tuple(t.requires_grad_() for t in (q, k, value, gate, beta))
    output, _ = chunk_gdn(*leaves, cu_seqlens=cu_seqlens, kernel_options=kernel_options)
    return (output, *torch.autograd.grad(output, leaves, torch.ones_like(output)))


def _kda_step(lengths):
    leaves = make_kda_test_inputs(
        sum(lengths), heads=2, seed=909, normalize_qk=True, requires_grad=True
    )
    offsets = cumulative_sequence_offsets(lengths)
    output, _ = chunk_kda(*leaves, cu_seqlens=offsets, kernel_options=_CUDNN)
    torch.autograd.grad(output, leaves, torch.ones_like(output))


def _count(monkeypatch, calls, module, name):
    original = getattr(module, name)

    def counted(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)


def _counting_full_cache_keys(monkeypatch, calls):
    """Count jit_cache's full argument-key builds, the path a warm launch must skip."""
    _count(monkeypatch, calls, cute_cache, "_fast_key")
    _count(monkeypatch, calls, cute_cache, "_make_runtime_key")


def test_warm_gdn_call_validates_once_and_skips_full_cache_key(monkeypatch):
    lengths = (97, 0, 161)
    _gdn_step(lengths)  # compile or load every launch
    gdn_warmup_backward_f16._validate_launch.cache_clear()
    calls = collections.Counter()
    _count(monkeypatch, calls, gdn_bprop_f16, "validate_bwd_operands")
    _counting_full_cache_keys(monkeypatch, calls)

    _gdn_step(lengths)
    # A new metadata signature runs the backward bundle checks exactly once.
    assert calls["validate_bwd_operands"] == 1
    calls.clear()
    _gdn_step(lengths)
    assert calls == {}


def test_warm_kda_call_validates_once_and_skips_full_cache_key(monkeypatch):
    lengths = (97, 0, 161)
    _kda_step(lengths)
    kda_warmup_forward_f16._validate_launch.cache_clear()
    calls = collections.Counter()
    _count(monkeypatch, calls, kda_prefill_f16, "validate_forward_operands")
    _counting_full_cache_keys(monkeypatch, calls)

    _kda_step(lengths)
    assert calls["validate_forward_operands"] == 1
    calls.clear()
    _kda_step(lengths)
    assert calls == {}


def test_gdn_chain_changing_shapes_matches_default(monkeypatch):
    """The chain launches are keyed without the batch shape; a later shape must still get its
    own plan, piece geometry and validation instead of a replay of the previous one."""
    monkeypatch.setattr(fe_gdn, "MIN_CHAIN_TOKENS_PER_PIECE_FWD", 0)
    monkeypatch.setattr(fe_gdn, "MIN_CHAIN_TOKENS_PER_PIECE_BWD", 0)
    device = torch.device("cuda")
    # 8, then 4 pieces per sequence, then the first shape again.
    for lengths in ((2048,), (900, 1148), (2048,)):
        tokens = sum(lengths)
        assert fe_gdn.ForwardPlan.build(tokens, len(lengths), 2, 128, device).chain
        assert fe_gdn.BackwardPlan.build(tokens, len(lengths), 2, device).chain
        actual = _gdn_step(lengths)
        expected = _gdn_step(lengths, kernel_options=None)
        for got, want in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
