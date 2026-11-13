"""Host-path contracts of the warpdrive KDA driver: what a warm call may skip.

Counts calls instead of timing them, so the checks cannot flake on a loaded host.
"""

import collections

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_kda
from attn_gym.linear._delta_rule.warpdrive.bundles import kda_warmup_forward
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs
from attn_gym.warpdrive.cache import jit as cute_cache

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive backend requires an SM100-family GPU",
)


def _kda_step(lengths):
    leaves = make_kda_test_inputs(
        sum(lengths), heads=2, seed=909, normalize_qk=True, requires_grad=True
    )
    offsets = cumulative_sequence_offsets(lengths)
    output, _ = chunk_kda(*leaves, cu_seqlens=offsets, kernel_options={"backend": "warpdrive"})
    torch.autograd.grad(output, leaves, torch.ones_like(output))


def _count(monkeypatch, calls, module, name):
    original = getattr(module, name)

    def counted(*args, **kwargs):
        calls[name] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, counted)


def test_warm_kda_call_validates_once_and_skips_full_cache_key(monkeypatch):
    lengths = (97, 0, 161)
    _kda_step(lengths)  # compile or load every launch
    kda_warmup_forward._validate_launch.cache_clear()
    calls = collections.Counter()
    _count(monkeypatch, calls, kda_warmup_forward, "validate_kda_forward_operands")
    # jit_cache's full argument-key builds, the path a warm launch must skip.
    _count(monkeypatch, calls, cute_cache, "_fast_key")
    _count(monkeypatch, calls, cute_cache, "_make_runtime_key")

    _kda_step(lengths)
    # A new metadata signature runs the forward bundle checks exactly once.
    assert calls["validate_kda_forward_operands"] == 1
    calls.clear()
    _kda_step(lengths)
    assert calls == {}
