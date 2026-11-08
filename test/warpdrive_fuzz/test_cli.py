"""Campaign accounting must include running jobs that finish after the admission deadline."""

import json
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass.cute")

from tools.warpdrive import fuzz_warpdrive
from warpdrive_fuzz import gen, runtime


def campaign_args(tmp_path, seeds="0,1"):
    return SimpleNamespace(
        out=str(tmp_path),
        jobs=2,
        seeds=seeds,
        pair=None,
        cluster="",
        stream=None,
        empty=False,
        timeout=1,
        wrapper="",
        hours=1,
        quiet=True,
    )


def stub_specs(monkeypatch):
    monkeypatch.setattr(
        gen,
        "generate",
        lambda seed, **kwargs: SimpleNamespace(seed=seed, pair=False, cluster=(), stream=None),
    )
    monkeypatch.setattr(gen, "features", lambda spec: set())


def test_deadline_collects_running_failures(monkeypatch, tmp_path, capsys):
    stub_specs(monkeypatch)
    started = Barrier(2)
    release_late = Event()
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls > 1:
            release_late.set()
            return 3601.0
        return 0.0

    def run(spec, *args, **kwargs):
        started.wait(timeout=5)  # both jobs running: the second cannot be cancelled
        if spec.seed == 1:
            assert release_late.wait(timeout=5)
            return runtime.CaseResult(1, "late", runtime.FAIL, failures=["late failure"])
        return runtime.CaseResult(0, "first", runtime.PASS)

    monkeypatch.setattr(fuzz_warpdrive.time, "time", clock)
    monkeypatch.setattr(runtime, "run_isolated", run)
    assert fuzz_warpdrive.runtime_campaign(campaign_args(tmp_path)) == 1
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert {row["seed"]: row["status"] for row in rows} == {0: "pass", 1: "fail"}
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["pass"] == summary["fail"] == 1


def test_deadline_accounts_for_cancelled_and_completed_seeds(monkeypatch, tmp_path, capsys):
    stub_specs(monkeypatch)
    started = Barrier(2)
    release_late = Event()
    clock_calls = 0

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls > 1:
            release_late.set()
            return 3601.0
        return 0.0

    def run(spec, *args, **kwargs):
        if spec.seed < 2:
            started.wait(timeout=5)
        if spec.seed:
            assert release_late.wait(timeout=5)
        return runtime.CaseResult(spec.seed, str(spec.seed), runtime.PASS)

    monkeypatch.setattr(fuzz_warpdrive.time, "time", clock)
    monkeypatch.setattr(runtime, "run_isolated", run)
    assert fuzz_warpdrive.runtime_campaign(campaign_args(tmp_path, seeds="0:100")) == 0
    rows = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert sorted(row["seed"] for row in rows) == list(range(100))
    assert all(row["status"] in (runtime.PASS, "cancelled") for row in rows)
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    for status in (runtime.PASS, "cancelled"):
        assert summary.get(status, 0) == sum(row["status"] == status for row in rows)


def test_known_failure_is_not_a_success(monkeypatch, tmp_path):
    stub_specs(monkeypatch)
    monkeypatch.setattr(
        runtime,
        "run_isolated",
        lambda spec, *args, **kwargs: runtime.CaseResult(
            spec.seed,
            "known",
            runtime.REJECT,
            detail="mixed register budgets {'a': None, 'work.scheduler': 160}",
        ),
    )
    assert fuzz_warpdrive.runtime_campaign(campaign_args(tmp_path, seeds="0")) == 1


def test_empty_corpus_always_has_an_empty_extent():
    seen_negative = False
    for seed in range(100):
        spec = gen.generate(seed, empty=True)
        assert spec.stream is not None
        assert min(spec.stream.extents) <= 0, (seed, spec.stream.extents)
        seen_negative |= min(spec.stream.extents) < 0
    assert seen_negative  # negative extents also mean an empty space (R17)
