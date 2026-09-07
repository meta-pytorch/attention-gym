"""Check CP command-line/data wiring without launching CUDA kernels or process groups."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("cutlass")
typer = pytest.importorskip("typer")
from typer.testing import CliRunner

from examples.linear import delta_rule_context_parallel as cp
from examples.linear.delta_rule_training import packed_sequence_metadata


@pytest.mark.parametrize("explicit_lengths", [False, True])
@pytest.mark.parametrize("legacy_names", [False, True])
def test_cli_uses_shared_packed_data_and_familiar_options(
    monkeypatch, explicit_lengths, legacy_names
):
    monkeypatch.setattr(cp, "distributed_device", lambda: nullcontext(torch.device("cpu")))
    monkeypatch.setattr(cp.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: 3)
    run = Mock()
    make_batch = Mock(wraps=cp.make_context_parallel_batch)
    monkeypatch.setattr(cp, "run_training_step", run)
    monkeypatch.setattr(cp, "make_context_parallel_batch", make_batch)
    app = typer.Typer()
    app.command()(cp.main)
    args = [
        "--batch-size",
        "4",
        "--tokens",
        "31",
        "--hidden-size",
        "12",
        "--heads" if legacy_names else "--num-heads",
        "3",
        "--kda-backend" if legacy_names else "--core-backend",
        "fused",
        "--partition",
        "zigzag",
        "--no-validate",
    ]
    if explicit_lengths:
        args += ["--sequence-lengths", "5,12,14"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    run.assert_called_once()
    model, hidden, target, routing, terminal_index, loss_scale = run.call_args.args
    if explicit_lengths:
        lengths, offsets = (5, 12, 14), (0, 5, 17, 31)
    else:
        torch.manual_seed(0)
        lengths, offsets = packed_sequence_metadata(4, 31)
    plan, actual_offsets, hidden_size, device = make_batch.call_args.args
    assert actual_offsets == offsets
    assert not make_batch.call_args.kwargs["validate"]
    assert routing.cp_rank == 1
    assert len(plan.fragments) == 2
    assert hidden.shape == target.shape == (1, routing.tokens, hidden_size)
    assert loss_scale == 1.0 / (sum(lengths) * hidden_size)
    assert terminal_index.tolist() == list(plan.terminal)
    assert isinstance(model, cp.ContextParallelDeltaRuleAttention)
    assert isinstance(model, cp.DeltaRuleAttention)
    assert model.num_heads == 3 and model.hidden_size == 12
    assert model.backend == "fused" and device.type == "cpu"


@pytest.mark.parametrize("mode", ["eager-profile", "graph", "graph-profile", "benchmark"])
def test_cli_shows_each_execution_mode(monkeypatch, mode):
    monkeypatch.setattr(cp, "distributed_device", lambda: nullcontext(torch.device("cpu")))
    monkeypatch.setattr(cp.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(cp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(cp, "run_training_step", Mock())
    monkeypatch.setattr(cp, "graph_annotations_available", lambda: False)
    graph = Mock()
    capture = Mock(return_value=nullcontext(graph))
    eager_profile = Mock()
    replay_profile = Mock(return_value=None)
    benchmark = Mock()
    monkeypatch.setattr(cp, "capture_training_graph", capture)
    monkeypatch.setattr(cp, "profile_eager_step", eager_profile)
    monkeypatch.setattr(cp, "record_distributed_profile", replay_profile)
    monkeypatch.setattr(cp, "run_benchmark", benchmark)
    options = {
        "eager-profile": ["--profile"],
        "graph": ["--cuda-graph"],
        "graph-profile": ["--cuda-graph", "--profile"],
        "benchmark": ["--benchmark-steps", "2", "--cuda-graph", "--profile"],
    }
    app = typer.Typer()
    app.command()(cp.main)
    result = CliRunner().invoke(
        app,
        [
            "--sequence-lengths",
            "17,13,31",
            "--hidden-size",
            "12",
            "--num-heads",
            "1",
            "--no-validate",
            *options[mode],
        ],
    )
    assert result.exit_code == 0, result.output
    assert capture.call_count == int(mode in ("graph", "graph-profile"))
    assert eager_profile.call_count == int(mode == "eager-profile")
    assert replay_profile.call_count == int(mode == "graph-profile")
    assert benchmark.call_count == int(mode == "benchmark")
    if capture.called:
        assert capture.call_args.kwargs == {"validate": True, "annotations": False}
    if replay_profile.called:
        assert replay_profile.call_args.args[0] == graph.replay
    if benchmark.called:
        assert benchmark.call_args.kwargs["cuda_graph"]
        assert benchmark.call_args.kwargs["steps"] == 2
        assert benchmark.call_args.kwargs["sequence_lengths"] == (17, 13, 31)
