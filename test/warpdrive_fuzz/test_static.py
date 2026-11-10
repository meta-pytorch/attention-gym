"""Small regressions from declaration and compile-only mutation fuzzing."""

import pytest

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive import TmemPlan
from warpdrive_fuzz.mutate import compile_mutation, mutations, run_mutation_fuzz
from warpdrive_fuzz.static import FuzzReport, declaration_cases, exercise_case, run_static_fuzz

DECLARATION_CASES = declaration_cases(0)


@pytest.mark.parametrize("case", DECLARATION_CASES, ids=[c.name for c in DECLARATION_CASES])
def test_declaration_case_verdict(case):
    """Reject invalid probes with the expected exception and message; accept valid ones."""
    report = FuzzReport(0)
    exercise_case(report, case)
    assert not report.findings, report.findings


def test_tmem_rejects_duplicate_names_before_plan_corruption():
    plan = TmemPlan().region("a", 32, 2).alias("b", of="a", offset=8, cols=16)
    with pytest.raises(ValueError, match="already declared"):
        plan.region("a", 16)
    with pytest.raises(ValueError, match="already declared"):
        plan.alias("a", of="a", offset=0, cols=16)
    assert plan.used_cols == 64
    assert plan.col("b", 1) == 40


@pytest.mark.parametrize(
    "name,message", [("compound_condition", "condition"), ("advance_before_work", "after advance")]
)
@pytest.mark.parametrize("seed", [0, 2], ids=["oneshot", "atomic"])
def test_tile_cursor_rejects_undrained_condition_and_next_item_reads(name, message, seed):
    """A local and a broadcast schedule; the cursor check is shared by all four."""
    mutation = next(m for m in mutations() if m.name == name)
    with pytest.raises(Exception, match=message) as info:
        compile_mutation(mutation, seed=seed)
    assert "ProtocolError" in repr(info.value)


@pytest.mark.fuzz
@pytest.mark.parametrize("seed", [7, 19])
def test_declaration_fuzz_quick(seed):
    """Run the corpus in seeded order, including random valid declarations.

    Seed 0 is covered by test_declaration_case_verdict.
    """
    budget = len(DECLARATION_CASES)
    report = run_static_fuzz(seed, budget=budget)
    assert not report.findings, report.findings
    assert report.cases == report.accepted + report.rejected == budget
    assert report.accepted > 0 and report.rejected > 0


@pytest.mark.fuzz
@pytest.mark.parametrize("seed", [0, 7, 19])
def test_mutation_fuzz_quick(seed):
    report = run_mutation_fuzz(seed, budget=12)
    assert not report.findings, report.findings
    assert report.cases == report.accepted + report.rejected + report.obligations == 12
