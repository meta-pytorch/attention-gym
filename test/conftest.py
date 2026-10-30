"""Shared pytest fixtures."""

import os
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

# Show measured values when shared test helpers' assertions fail.
pytest.register_assert_rewrite("attn_gym.testing")

# Both caches are read at import or CUDA-init time, so set them before any test module loads;
# subprocess tests inherit them. Explicit user settings win.
# Cached CuTeDSL objects are finalized by the driver on load and memoized in its ComputeCache.
# One suite run writes ~1 GiB there; the smaller default limit evicts warm entries, and each
# xdist worker then re-finalizes them (up to ~30 s per GDN backward test).
os.environ.setdefault("CUDA_CACHE_MAXSIZE", str(4 << 30))
# FA4 otherwise recompiles every specialization in every process; its disk cache is keyed by
# the FA4 source fingerprint, so editing a flash-attention checkout still invalidates it.
os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")

import torch


def pytest_configure(config: pytest.Config) -> None:
    """Refuse to test one checkout's sources through another worktree's editable install."""
    import attn_gym

    package_root = Path(attn_gym.__file__).resolve().parent.parent
    repo_root = Path(__file__).resolve().parent.parent
    if package_root == repo_root or "site-packages" in package_root.parts:
        return
    raise pytest.UsageError(
        f"attn_gym imports from {package_root}, not this checkout {repo_root}. The active "
        "environment's editable install (or a `.venv` symlinked to another worktree) points at "
        "a sibling checkout; follow .agents/skills/worktree-env-setup/SKILL.md to create this "
        "worktree's own .venv."
    )


@pytest.fixture(autouse=True)
def _seed_global_rng() -> None:
    """Make unseeded draws independent of which tests an xdist worker ran first.

    Otherwise adding, removing, or reparametrizing any test reshuffles other tests' data.
    """
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def _reset_shared_compile_wrapper() -> Iterator[None]:
    """Give each test its own recompile budget for compiled partials and op objects.

    `torch.compile` wraps callables without a Python frame in `wrap_inline`, whose `inner` code
    object is shared process-wide, so unrelated tests on one xdist worker otherwise exhaust its
    recompile limit together and fail depending on test distribution.
    """
    yield
    from torch._C._dynamo.eval_frame import reset_code
    from torch._dynamo.external_utils import wrap_inline

    reset_code(wrap_inline(len).__code__)


@pytest.fixture
def fresh_compile_cache() -> Iterator[None]:
    """Keep independent compile tests from sharing a function's recompilation budget."""
    torch.compiler.reset()
    try:
        yield
    finally:
        torch.compiler.reset()


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
def gather_attn_single_config(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Use one valid Triton schedule unless a test explicitly covers autotuning.

    The schedule is each tuner's first candidate, or, for a test marked
    ``tuner_config(index)``, the candidate at that index (modulo the tuner's valid candidates).
    """
    if (
        not torch.cuda.is_available()
        or request.node.get_closest_marker("full_autotune") is not None
    ):
        yield
        return
    marker = request.node.get_closest_marker("tuner_config")
    index = marker.args[0] if marker else 0

    import importlib

    from attn_gym._backends.triton.tune import TritonTuner
    from attn_gym.sparse.gather_attn.impl.triton import backward, forward, shared_backward

    tuners = {
        id(value): value
        for module in (forward, backward, shared_backward)
        for value in vars(module).values()
        if isinstance(value, TritonTuner)
    }.values()
    for tuner in tuners:
        original_pruner = tuner._prune

        def pinned_config(configs, args, _pruner=original_pruner):
            valid = _pruner(configs, args) if _pruner else configs
            # An empty list still reaches the tuner's own "no configs" error.
            return valid[index % len(valid) :][:1] if valid else valid

        monkeypatch.setattr(tuner, "_prune", pinned_config)
    # The single-candidate winner would otherwise stay in the in-process memo, which is not
    # keyed by the candidate set, and skip tuning for later callers.
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")
    monkeypatch.setattr(tune_module, "_WINNERS", dict(tune_module._WINNERS))
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", dict(tune_module._WINNERS_FAST))
    yield


@pytest.fixture
def nan_filled_empty(monkeypatch):
    """Fill every fresh floating-point tensor with NaN, the worst case ``torch.empty`` allows.

    The caching allocator mostly recycles blocks, so a test cannot rely on real garbage: filling
    at the allocation entry points the kernels use makes every unwritten element a NaN. The
    grouped gather schedule's ``uninitialized_empty`` buffers also get integer poison.
    """

    def nan_filled(allocate):
        def allocate_nan_filled(*args, **kwargs):
            tensor = allocate(*args, **kwargs)
            return tensor.fill_(torch.nan) if tensor.is_floating_point() else tensor

        return allocate_nan_filled

    for owner, name in ((torch, "empty"), (torch, "empty_like"), (torch.Tensor, "new_empty")):
        monkeypatch.setattr(owner, name, nan_filled(getattr(owner, name)))
    if not torch.cuda.is_available():
        return
    from attn_gym.sparse.gather_attn.impl.triton import (
        backward,
        forward,
        key_major_backward,
        primitives,
    )

    def poisoned(allocate):
        """Integers too: gather attention's index buffers bypass ``torch.empty``."""

        def allocate_poisoned(*args, **kwargs):
            tensor = allocate(*args, **kwargs)
            if tensor.is_floating_point():
                return tensor.fill_(torch.nan)
            return tensor.fill_(torch.iinfo(tensor.dtype).max)

        return allocate_poisoned

    # The grouped gather schedule allocates buffers it fully writes without torch.empty.
    for module in (backward, forward, key_major_backward, primitives):
        monkeypatch.setattr(
            module, "uninitialized_empty", poisoned(primitives.uninitialized_empty)
        )
