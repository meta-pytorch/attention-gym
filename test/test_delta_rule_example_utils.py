"""CPU checks for the data and process-lifetime helpers used by the CP example."""

from contextlib import nullcontext
from unittest.mock import Mock

import pytest
import torch

from attn_gym.linear.context_parallel import ContextParallelPlan
from attn_gym.testing import delta_rule


@pytest.fixture
def training():
    """Load example recipes only when their optional fused dependencies are installed."""
    pytest.importorskip("cutlass")
    pytest.importorskip("typer")
    from examples.linear import delta_rule_training

    return delta_rule_training


def test_reference_checks_reduce_copies_of_parameter_gradients(monkeypatch):
    gradient = torch.tensor([0.25, -0.5, 2.0], requires_grad=True)
    original = gradient.detach().clone()
    reduce = Mock(side_effect=lambda value: value.mul_(2))
    monkeypatch.setattr(delta_rule.dist, "all_reduce", reduce)
    output = torch.tensor([1.0, -2.0])
    delta_rule.assert_context_parallel_matches_reference(
        [(output, output.clone())], [(gradient, original * 2)], torch.bfloat16
    )
    reduce.assert_called_once()
    assert reduce.call_args.args[0].data_ptr() != gradient.data_ptr()
    torch.testing.assert_close(gradient, original)


@pytest.mark.parametrize("mismatch", ["output", "parameter"])
def test_reference_checks_reject_mismatches(monkeypatch, mismatch):
    monkeypatch.setattr(delta_rule.dist, "all_reduce", lambda value: None)
    actual = torch.tensor([1.0, -2.0])
    expected = actual + 1.0
    pairs = [(actual, expected)] if mismatch == "output" else [(actual, actual)]
    gradients = [(actual, expected)] if mismatch == "parameter" else [(actual, actual)]
    with pytest.raises(AssertionError):
        delta_rule.assert_context_parallel_matches_reference(pairs, gradients, torch.bfloat16)


@pytest.mark.parametrize(
    "fragments",
    [
        pytest.param([[(0, 15)], [(15, 31)]], id="contiguous-2"),
        pytest.param([[(0, 10)], [(10, 20)], [(20, 31)]], id="contiguous-3"),
        pytest.param([[(0, 7), (23, 31)], [(7, 15), (15, 23)]], id="zigzag-2"),
        pytest.param(
            [[(0, 5), (25, 31)], [(5, 10), (20, 25)], [(10, 15), (15, 20)]], id="zigzag-3"
        ),
    ],
)
@pytest.mark.parametrize("validate", [False, True])
def test_batch_shards_the_same_global_stream(fragments, validate, training):
    offsets = (0, 5, 17, 31)
    world_size = len(fragments)
    plans = [
        ContextParallelPlan.from_fragments(offsets, fragments, rank) for rank in range(world_size)
    ]
    reference = training.make_context_parallel_batch(
        plans[0], offsets, 4, torch.device("cpu"), conv_history=3, validate=True
    )
    rng_state = torch.get_rng_state()
    batches = [
        training.make_context_parallel_batch(
            plan, offsets, 4, torch.device("cpu"), conv_history=3, validate=validate
        )
        for plan in plans
    ]
    # Batch construction has its own data seeds and must not perturb model initialization.
    torch.testing.assert_close(torch.get_rng_state(), rng_state)
    all_ids = torch.cat([batch.token_ids for batch in batches]).sort().values
    torch.testing.assert_close(all_ids, torch.arange(offsets[-1]))
    for rank, (plan, batch) in enumerate(zip(plans, batches, strict=True)):
        torch.testing.assert_close(batch.local_hidden, reference.global_hidden[:, batch.token_ids])
        torch.testing.assert_close(batch.local_target, reference.global_target[:, batch.token_ids])
        assert (batch.global_hidden is not None) == validate
        assert (batch.global_target is not None) == validate
        assert batch.local_hidden.is_leaf and batch.local_hidden.requires_grad
        assert not batch.local_target.requires_grad
        assert batch.routing.cp_rank == rank
        assert batch.loss_scale == 1.0 / (offsets[-1] * 4)
        assert batch.terminal_index.tolist() == list(plan.terminal)
        assert batch.terminal_sequences.tolist() == [
            plan.subsequences[index].sequence for index in plan.terminal
        ]
        assert batch.routing.tail_sources.shape[-1] == 3


@pytest.mark.parametrize("fail", [False, True])
def test_distributed_device_cleans_up_on_caller_exit(monkeypatch, fail, training):
    calls = Mock()
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "set_device", calls.set_device)
    monkeypatch.setattr(torch.cuda, "synchronize", calls.synchronize)
    monkeypatch.setattr(training.dist, "init_process_group", calls.init)
    monkeypatch.setattr(training.dist, "destroy_process_group", calls.destroy)
    error = pytest.raises(RuntimeError, match="caller failed") if fail else nullcontext()
    with error, training.distributed_device() as device:
        assert device == torch.device("cuda", 1)
        calls.init.assert_called_once_with("nccl", device_id=device)
        calls.destroy.assert_not_called()
        if fail:
            raise RuntimeError("caller failed")
    calls.synchronize.assert_called_once_with(device)
    calls.destroy.assert_called_once_with()
    assert [entry[0] for entry in calls.mock_calls] == [
        "set_device",
        "init",
        "synchronize",
        "destroy",
    ]
