"""Checks for the public context-parallel fold benchmark."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("cutlass")

from examples.linear.benchmark_context_parallel_folds import (
    CapturedPhase,
    assert_exact_parity,
    gpu_telemetry,
    summarize,
)


def measurements(rounds):
    return [
        {"round": index, "label": label, "phase": phase, "step_ms": times}
        for index, times in enumerate(rounds)
        for label in ("independent", "root")
        for phase in ("forward", "backward")
    ]


@pytest.mark.parametrize("phase", ["forward", "backward"])
def test_benchmark_rejects_uncaptured_baseline(monkeypatch, phase):
    captured = {
        label: {
            name: CapturedPhase(Mock(spec=torch.cuda.CUDAGraph), (torch.ones(1),), 1)
            for name in ("forward", "backward")
        }
        for label in ("independent", "root")
    }
    captured["independent"][phase] = captured["independent"][phase]._replace(fold_calls=0)
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op: None)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    with pytest.raises(AssertionError, match=f"independent {phase} fold was not captured"):
        assert_exact_parity(captured, torch.device("cpu"))


def test_telemetry_selects_visible_device_by_uuid(monkeypatch):
    device = torch.device("cuda", 1)
    uuid = UUID("d921a7dc-6a26-9503-4a7e-a4ad0c0e8e4a")
    expected = f"GPU-{uuid}"

    def properties(selected):
        assert selected == device
        return SimpleNamespace(uuid=uuid)

    def query(command, *, text):
        assert command[0] == "nvidia-smi"
        assert f"--id={expected}" in command
        assert text
        return f"{expected}, 2070 MHz, 200 W, 35\n"

    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(subprocess, "check_output", query)
    selected, readings = gpu_telemetry(device)
    assert selected == expected
    assert readings.startswith(f"{expected},")


def test_summary_uses_slowest_rank_per_iteration():
    ranks = [measurements([[9.0, 1.0, 1.0]]), measurements([[1.0, 9.0, 1.0]])]
    assert set(summarize(ranks, rounds=1).values()) == {9.0}


def test_summary_takes_median_of_round_medians():
    rounds = [[1.0, 2.0, 100.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
    ranks = [measurements(rounds), measurements(rounds)]
    assert set(summarize(ranks, rounds=3).values()) == {4.0}
