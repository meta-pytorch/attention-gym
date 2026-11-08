"""Runtime fuzzing of warpdrive's guarantees to user kernels (quick mode).

Generated valid kernels (``warpdrive_fuzz/gen.py``) run in isolated child processes against an
exact host model (``oracle.py``); host properties of the scheduler and of derived counts run
against independent recomputations (``sched.py``); synthetic layer bugs (``runtime_mutants.py``)
check that the oracle notices each failure class. Long campaigns: ``tools/warpdrive/fuzz_warpdrive.py``.
"""

import random

import pytest
import torch

pytest.importorskip("cutlass.cute")

from cutlass import cute

from attn_gym.warpdrive import (
    AtomicPersistent,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from warpdrive_fuzz import gen, runtime, runtime_mutants, sched

pytestmark = pytest.mark.fuzz

CASE_TIMEOUT = 120.0  # seconds per generated kernel: compile + 2 launches + checks
MUTANT_TIMEOUT = 30.0  # a mutant that hangs is detected; don't wait longer than this


def _covering_seeds(pool: range, limit: int, **force) -> list[int]:
    """Greedy seeds covering the generator's knob classes, skipping the >65535-CTA grids (their
    host check is slow; the long campaign runs them)."""
    specs = {s: gen.features(gen.generate(s, **force)) for s in pool}
    specs = {s: f for s, f in specs.items() if "grid_yz_edge" not in f}
    chosen, covered = [], set()
    while len(chosen) < limit:
        best = max(specs, key=lambda s: (len(specs[s] - covered), -s))
        if not specs[best] - covered:
            break
        chosen.append(best)
        covered |= specs.pop(best)
    return sorted(chosen)


QUICK_SEEDS = _covering_seeds(range(300), 32)
CLUSTERS = {"2x2p": (2, 2, True), "2x2": (2, 2, False)}
QUICK_SEEDS += [s for s in range(300, 400) if s not in QUICK_SEEDS][
    : max(0, 16 - len(QUICK_SEEDS))
]


def _explain(spec, res) -> str:
    lines = [
        f"seed {spec.seed} key {res.key}: {res.status}",
        f"source: {res.source}",
        f"rerun: python tools/warpdrive/fuzz_warpdrive.py runtime --seeds {spec.seed}"
        + (
            f" --cluster {'x'.join(map(str, spec.cluster[:2]))}{'p' * spec.pair}"
            if spec.cluster
            else ""
        ),
        *res.failures[:20],
        res.detail[-3000:],
    ]
    return "\n".join(lines)


@pytest.mark.parametrize("seed", QUICK_SEEDS)
def test_generated_kernel_matches_model(requires_blackwell, tmp_path_factory, seed):
    """Every payload epoch reaches every consumer once, in channel order and intact; each kept
    work item reaches each placed role once with identical per-CTA sequences; launches are
    bitwise repeatable for deterministic schedules; nothing hangs."""
    spec = gen.generate(seed)
    res = runtime.run_isolated(spec, tmp_path_factory.getbasetemp() / "wdfuzz", CASE_TIMEOUT)
    assert res.status == runtime.PASS, _explain(spec, res)


CLUSTER_SEEDS = [
    (name, seed)
    for name, shape in CLUSTERS.items()
    for seed in _covering_seeds(range(100), 4 if shape[2] else 2, cluster=shape)
]


@pytest.mark.parametrize("cluster,seed", CLUSTER_SEEDS)
def test_generated_cluster_kernel_matches_model(
    requires_blackwell, tmp_path_factory, cluster, seed
):
    """The same model on clusters of two MMA pairs (and four CTAs without pairs): roles in all,
    leader or rank-0 CTAs, channels scoped to each pair or the whole cluster, and the rank-0
    scheduler publishing to all four CTAs."""
    spec = gen.generate(seed, cluster=CLUSTERS[cluster])
    res = runtime.run_isolated(spec, tmp_path_factory.getbasetemp() / "wdfuzz", CASE_TIMEOUT)
    assert res.status == runtime.PASS, _explain(spec, res)


# Mutants the oracle must catch, with the seeds (exercising the mutated feature) to try. Known
# blind spots of a value-based oracle are excluded: the early-release work-ring mutant (its
# read-after-release window is too narrow to hit without delays inside the layer) and dropped
# pair tails (a late remote tcgen05.commit into an exited CTA changes no logged value).
BLIND_SPOTS = ("receive_early_release", "no_pair_tail")
DETECTABLE = [m for m in runtime_mutants.MUTANTS if m not in BLIND_SPOTS]


# Mutants whose failure is a device fault rather than a mismatch or hang: extra mbarrier
# arrivals trap.
FAULTING = ("signal_false_signals",)


def _detected(mutant: str, res) -> bool:
    if res.status in (runtime.FAIL, runtime.HANG):
        return True
    fault = res.status in (runtime.ERROR, runtime.CRASH) and "CUDA error" in res.detail
    return fault and mutant in FAULTING


@pytest.mark.parametrize("mutant", DETECTABLE)
def test_oracle_detects_layer_mutant(requires_blackwell, tmp_path_factory, mutant):
    feature = runtime_mutants.MUTANTS[mutant][1]
    seeds = [s for s in range(300) if feature in gen.features(gen.generate(s))]
    workdir = tmp_path_factory.getbasetemp() / f"wdfuzz_mutant_{mutant}"
    env = {"WDFUZZ_MUTANT": mutant, "CUTE_DSL_DISABLE_FILE_CACHING": "1"}
    seen = []
    for seed in seeds[:4]:
        res = runtime.run_isolated(gen.generate(seed), workdir, MUTANT_TIMEOUT, env=env)
        seen.append((seed, res.status))
        if _detected(mutant, res):
            return
    pytest.fail(f"mutant {mutant} survived seeds {seen}")


# --------------------------------------------------------------------------- #
# Host properties (no GPU)
# --------------------------------------------------------------------------- #


def test_mma_operands_are_exact_integer_matmuls():
    """The MMA subgraph's device check (``runtime_device.mma_check``) recomputes each
    accumulator from the operand formulas; that equals the matmul of the bf16 source tensors
    the harness uploads, exactly (small integers, fp32 accumulation)."""
    from warpdrive_fuzz import runtime_device

    spec = next(s for s in map(gen.generate, range(300)) if s.mma is not None)
    a = runtime.mma_operand(spec, "mma_a", "cpu")
    b = runtime.mma_operand(spec, "mma_b", "cpu")
    ab, _ = gen.mma_channels(spec)
    rows, n, k = gen.MMA_M, spec.mma.n, spec.mma.k
    for e in range(3):
        tiles = [(e * ab.mult + j) % ab.tma_tiles for j in range(ab.mult)]
        got = sum(
            a[t * rows : (t + 1) * rows].float() @ b[t * n : (t + 1) * n].float().T for t in tiles
        )
        r = torch.arange(rows).view(-1, 1, 1)
        c = torch.arange(n).view(1, -1, 1)
        kk = torch.arange(k).view(1, 1, -1)
        want = sum(
            (
                runtime_device.mma_a(t, r, kk, spec.mma.salt)
                * runtime_device.mma_b(t, c, kk, spec.mma.salt)
            ).sum(-1)
            for t in tiles
        )
        assert torch.equal(got, want.float())


def test_raster_decode_matches_documented_order():
    rng = random.Random(0)
    failures = []
    for _ in range(400):
        failures += sched.check_raster(sched.random_stream(rng))
    assert not failures, failures[:5]


def test_launch_grid_matches_contract():
    rng = random.Random(1)
    failures = []
    for _ in range(400):
        stream = sched.random_stream(rng)
        for cluster in (1, 2, *CLUSTERS.values(), (4, 1, True), (2, 4, False)):
            failures += sched.check_grid(stream, cluster)
    for space in sched.grid_boundary_spaces():
        for cs in (1, 2, 4, 8):
            failures += sched.check_tile_grid_dims(space, cs)
    assert not failures, failures[:5]


@pytest.mark.parametrize("chunk", range(4))
def test_declared_scopes_match_independent_model(tmp_path_factory, chunk):
    """Counts, targets, masks, tx, tails, scheduler placement, register padding and barrier
    initialization of generated kernels, against a recomputation from the declared graph."""
    workdir = tmp_path_factory.getbasetemp() / "wdfuzz_decl"
    failures = []
    for seed in range(chunk * 40, chunk * 40 + 40):
        spec = gen.generate(seed)
        failures += [f"seed {seed}: {f}" for f in sched.check_declared(spec, workdir).failures]
    assert not failures, failures[:10]


@pytest.mark.parametrize("cluster", [*CLUSTERS.values(), (4, 1, True), (2, 1, True)])
def test_declared_cluster_scopes_match_independent_model(tmp_path_factory, cluster):
    """The same recomputation on generated cluster kernels: per-rank counts, per-pair and
    whole-cluster targets, per-rank commit masks, and the rank-0 scheduler's ring."""
    workdir = tmp_path_factory.getbasetemp() / "wdfuzz_decl"
    failures = []
    for seed in range(40):
        spec = gen.generate(seed, cluster=cluster)
        failures += [f"seed {seed}: {f}" for f in sched.check_declared(spec, workdir).failures]
    assert not failures, failures[:10]


# --------------------------------------------------------------------------- #
# Minimized regressions of runtime-fuzzer findings (test/warpdrive_fuzz/FINDINGS.md)
# --------------------------------------------------------------------------- #


class _SchedulerInDefaultBudgetGroup(WarpSpecialized):
    """F-A1: warpgroup 0 is declared with the default budget (regs=None) and warp 1 is a hole,
    so the atomic scheduler lands there; another warpgroup sets regs=160."""

    def __init__(self):
        self.work = WorkStream(schedule=AtomicPersistent())

    @role(warps=0)
    @cute.jit
    def a(self, work: WorkCursor, ctx):
        while work.valid:
            work.advance()
        return work

    @role(warps=range(4, 8), regs=160)
    @cute.jit
    def b(self, work: WorkCursor, ctx):
        while work.valid:
            work.advance()
        return work


def test_scheduler_takes_its_warpgroups_default_budget():
    """F-A1: once raised ``warpgroup 0 has mixed register budgets {'a': None,
    'work.scheduler': 160}``; the scheduler now inherits its warpgroup's (default) budget."""
    op = _SchedulerInDefaultBudgetGroup()
    text = op.describe()
    assert "scheduler warp 1 broadcasts" in text
    rows = {name: regs for _, name, regs in sched.described_roles(text)}
    assert rows["work.scheduler"] == rows["a"]
