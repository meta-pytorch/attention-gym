"""Fixtures shared by the linear-attention tests."""

import importlib

import pytest


@pytest.fixture
def mamba2_first_candidate(isolated_winners, monkeypatch):
    """Tune every Mamba2 Triton kernel over its first candidate only, from empty winner caches.

    One candidate makes compile counts deterministic; the empty caches keep winners that other
    tests chose over every candidate from launching other configurations. Each tuner's first
    candidate uses ``BP=32``, which ``prune_p_tiles`` keeps for every head size.
    """
    from attn_gym._backends.triton.tune import TritonTuner

    for name in ("chunk_fwd", "chunk_bwd"):
        module = importlib.import_module(f"attn_gym.linear.mamba2.triton.{name}")
        for tuner in vars(module).values():
            if isinstance(tuner, TritonTuner):
                monkeypatch.setattr(tuner, "_configs", tuner._configs[:1])
