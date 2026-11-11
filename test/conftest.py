"""Shared pytest fixtures."""

import os
import sys
from collections import Counter
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


def _configure_worker_parallelism(config: pytest.Config) -> None:
    if sys.platform != "linux" or not hasattr(config, "workerinput"):
        return
    workers = config.workerinput["workercount"]
    if workers <= 1:
        return
    cpu_budget = max(1, len(os.sched_getaffinity(0)) // workers)
    # Limit compiler parallelism per xdist worker unless explicitly overridden.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", str(min(4, cpu_budget)))
    if "OMP_NUM_THREADS" in os.environ or "MKL_NUM_THREADS" in os.environ:
        return
    threads = min(torch.get_num_threads(), cpu_budget)
    # Propagate the budget to spawned ranks too, and handle plugins that imported torch early.
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)


def pytest_configure(config: pytest.Config) -> None:
    """Configure worker budgets and reject imports from another checkout."""
    _configure_worker_parallelism(config)
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
def requires_blackwell() -> None:
    """Skip unless an SM100-family GPU (the warpdrive target) is present."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("warpdrive tests require SM100-family GPUs")


@pytest.fixture
def traced_compiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this test's compiles out of the jit_cache disk cache (``CUTE_DSL_NO_CACHE``).

    For tests that assert on tracing itself (IKET ranges emitted, monkeypatched trace hooks).
    A cached warpdrive kernel restores what its trace reported (describe(), alias findings, TMA
    reports, warnings), but nothing is traced. The in-memory cache still applies: clear the
    compile function (``cache_clear()``) when an earlier test in the process may have compiled
    the same key.
    """
    monkeypatch.setenv("CUTE_DSL_NO_CACHE", "1")


@pytest.fixture
def fresh_compile_cache() -> Iterator[None]:
    """Keep independent compile tests from sharing a function's recompilation budget."""
    torch.compiler.reset()
    try:
        yield
    finally:
        torch.compiler.reset()


@pytest.fixture
def isolated_winners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Use empty ``run_tunable`` winner caches in memory and on disk."""
    from attn_gym._backends import tune as tune_module

    # Only winners move: compiled kernels stay in the shared cache.
    monkeypatch.setattr(tune_module, "get_cache_path", lambda: tmp_path)
    monkeypatch.setattr(tune_module, "_WINNERS", {})
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", {})


@pytest.fixture
def no_benchmarking(
    isolated_winners: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject ``run_tunable`` tuning.

    Drop ``run_tunable`` winners so cached results cannot hide a tuned launch.
    """
    from attn_gym._backends import tune as tune_module

    monkeypatch.setattr(tune_module, "get_cache_path", lambda: tmp_path / "no_benchmarking")
    tune_module._WINNERS.clear()
    tune_module._WINNERS_FAST.clear()

    def fail(*_args, **_kwargs):
        raise AssertionError("a pinned launch must not benchmark")

    monkeypatch.setattr(tune_module, "tune", fail)


@pytest.fixture
def deterministic_algorithms() -> Iterator[None]:
    """Run the test under ``torch.use_deterministic_algorithms(True)``."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.fixture
def triton_compiles(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """Count Triton binaries built for each kernel name while the test runs.

    The hook fires on every in-process kernel-cache miss, including binaries loaded from Triton's
    disk cache, so a count reflects specializations rather than cold compiles. Those in-process
    caches outlive a test, so the fixture first empties them for every loaded ``attn_gym`` kernel;
    otherwise a specialization an earlier test built in the same worker would go uncounted.
    """
    import triton
    from triton.runtime.autotuner import Heuristics
    from triton.runtime.jit import JITFunction

    for name, module in list(sys.modules.items()):
        if module is None or name.partition(".")[0] != "attn_gym":
            continue
        for kernel in list(vars(module).values()):
            while isinstance(kernel, Heuristics):
                kernel = kernel.fn
            if isinstance(kernel, JITFunction):
                kernel.device_caches.clear()

    compiles: Counter[str] = Counter()

    def count(**kwargs) -> None:
        compiles[kwargs["fn"].name] += 1

    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", count)
    return compiles


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

    from attn_gym._backends import tune as tune_module
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
    monkeypatch.setattr(tune_module, "_WINNERS", dict(tune_module._WINNERS))
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", dict(tune_module._WINNERS_FAST))
    yield


@pytest.fixture
def nan_filled_empty(monkeypatch):
    """Fill every fresh floating-point tensor with NaN, the worst case ``torch.empty`` allows.

    The caching allocator mostly recycles blocks, so a test cannot rely on real garbage: filling
    at the allocation entry points the kernels use makes every unwritten element a NaN.
    """

    def nan_filled(allocate):
        def allocate_nan_filled(*args, **kwargs):
            tensor = allocate(*args, **kwargs)
            return tensor.fill_(torch.nan) if tensor.is_floating_point() else tensor

        return allocate_nan_filled

    for owner, name in ((torch, "empty"), (torch, "empty_like"), (torch.Tensor, "new_empty")):
        monkeypatch.setattr(owner, name, nan_filled(getattr(owner, name)))
