"""Shared pytest fixtures."""

from collections.abc import Callable, Iterator

import pytest
import torch


@pytest.fixture
def paged_short_conv_inputs() -> Callable[..., tuple[torch.Tensor, ...]]:
    """Build one-token inputs and a paged short-convolution history pool."""

    def make(
        sequences: int = 3,
        channels: int = 12,
        width: int = 4,
        slots: int = 7,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if slots <= sequences:
            raise ValueError("slots must exceed the sequence count")
        x = torch.randn(sequences, channels, device="cuda", dtype=dtype)
        weight = torch.randn(channels, width, device="cuda", dtype=dtype)
        state = torch.randn(slots, width - 1, channels, device="cuda", dtype=dtype)
        state_indices = torch.randperm(slots - 1, device="cuda")[:sequences].to(torch.int32) + 1
        return x, weight, state, state_indices

    return make


@pytest.fixture
def selected_attention_single_config(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Use one valid Triton schedule unless a test explicitly covers autotuning."""
    if (
        not torch.cuda.is_available()
        or request.node.get_closest_marker("full_autotune") is not None
    ):
        yield
        return

    from triton.runtime.autotuner import Autotuner

    from attn_gym.sparse.selected_attention.impl.triton import backward, forward, shared_backward

    tuners = tuple(
        {
            id(value): value
            for module in (forward, backward, shared_backward)
            for value in vars(module).values()
            if isinstance(value, Autotuner)
        }.values()
    )
    original_caches = {id(tuner): tuner.cache.copy() for tuner in tuners}
    for tuner in tuners:
        original_pruner = tuner.early_config_prune

        def first_valid_config(configs, named_args, _pruner=original_pruner, **kwargs):
            valid_configs = _pruner(configs, named_args, **kwargs) if _pruner else configs
            return valid_configs[:1]

        monkeypatch.setattr(tuner, "early_config_prune", first_valid_config)

    try:
        yield
    finally:
        # A single pruned config is inserted into Autotuner.cache without disk caching.
        # Restore genuine entries so reduced test choices cannot affect later callers.
        for tuner in tuners:
            tuner.cache.clear()
            tuner.cache.update(original_caches[id(tuner)])
