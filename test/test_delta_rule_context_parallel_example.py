"""Exercise per-batch routing on one complete delta-rule module with two NCCL ranks."""

from __future__ import annotations

from dataclasses import fields
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

pytest.importorskip("cutlass")
pytest.importorskip("typer")

from attn_gym.linear.context_parallel import ContextParallelPlan, ContextParallelRouting
from examples.delta_rule_context_parallel import ContextParallelDeltaRuleAttention
from examples.delta_rule_training import DeltaRuleAttention, DeltaRuleAttentionOutput

pytestmark = [
    pytest.mark.skipif(
        torch.cuda.device_count() < 2
        or any(
            torch.cuda.get_device_capability(i) < (9, 0)
            for i in range(min(2, torch.cuda.device_count()))
        ),
        reason="the complete CP delta-rule module needs two Hopper-or-newer GPUs and NCCL",
    ),
    pytest.mark.xdist_group("two-gpu"),
]

# Both layouts own 64 tokens per rank, but change document boundaries and fragment ownership.
LAYOUTS = (
    ((0, 17, 83, 128), [[(0, 64)], [(64, 128)]]),
    ((0, 29, 91, 128), [[(0, 32), (96, 128)], [(32, 64), (64, 96)]]),
)
CAPS = {"slots": 2, "max_subsequences": 4, "conv_history": 3}


def _gradients(
    model: DeltaRuleAttention,
    result: DeltaRuleAttentionOutput,
    hidden: torch.Tensor,
    target: torch.Tensor,
    terminal: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Include both state losses, masking nonterminal and padding rows without dynamic shapes."""
    assert result.final_state is not None and result.final_conv_state is not None
    loss = F.mse_loss(result.hidden_states.float(), target, reduction="sum") / target.shape[-1]
    loss = loss + 1e-4 * (result.final_state.square() * terminal[:, None, None, None]).sum()
    loss = loss + 1e-4 * (result.final_conv_state.float().square() * terminal[:, None, None]).sum()
    return torch.autograd.grad(loss, (hidden, *model.parameters()))


def _step(
    model: ContextParallelDeltaRuleAttention,
    hidden: torch.Tensor,
    target: torch.Tensor,
    routing: ContextParallelRouting,
) -> tuple[DeltaRuleAttentionOutput, tuple[torch.Tensor, ...]]:
    """Run the public per-call routing ABI, including full-module backward."""
    result = model(hidden, routing=routing, return_final_state=True)
    return result, _gradients(model, result, hidden, target, routing.terminal)


def _assert_matches_reference(
    model: ContextParallelDeltaRuleAttention,
    reference: DeltaRuleAttention,
    plan: ContextParallelPlan,
    global_hidden: torch.Tensor,
    global_target: torch.Tensor,
    offsets: tuple[int, ...],
    actual: tuple[DeltaRuleAttentionOutput, tuple[torch.Tensor, ...]],
) -> None:
    """Check outputs, both endpoint states, input gradients, and every reduced parameter gradient."""
    hidden = global_hidden.detach().clone().requires_grad_()
    expected = reference(
        hidden,
        cu_seqlens=torch.tensor(offsets, device=hidden.device, dtype=torch.int32),
        return_final_state=True,
    )
    expected_grads = _gradients(
        reference, expected, hidden, global_target, hidden.new_ones(len(offsets) - 1)
    )
    result, grads = actual
    ids = plan.global_token_ids(hidden.device)
    terminal = torch.tensor(plan.terminal, device=hidden.device, dtype=torch.long)
    sequences = torch.tensor(
        [plan.subsequences[index].sequence for index in plan.terminal],
        device=hidden.device,
        dtype=torch.long,
    )
    pairs = [
        ("output", result.hidden_states, expected.hidden_states[:, ids]),
        ("input gradient", grads[0], expected_grads[0][:, ids]),
        ("final state", result.final_state[terminal], expected.final_state[sequences]),
        ("conv state", result.final_conv_state[terminal], expected.final_conv_state[sequences]),
    ]
    for (name, _), grad, expected_grad in zip(
        model.named_parameters(), grads[1:], expected_grads[1:], strict=True
    ):
        reduced = grad.detach().clone()
        dist.all_reduce(reduced, group=model.cp_group)
        pairs.append((name, reduced, expected_grad))
    # Match the full-module example's pointwise budget: partitioning changes low-precision rounding.
    eps = torch.finfo(model.compute_dtype).eps
    for name, value, expected_value in pairs:
        torch.testing.assert_close(value, expected_value, atol=eps, rtol=eps, msg=name)


def _rank_main(
    rank: int, rendezvous: str, dtype: torch.dtype, capture: bool, variant: str
) -> None:
    """Keep one model across layouts; captured runs update only caller-owned tensor buffers."""
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        device_id=device,
        timeout=timedelta(seconds=180),
    )
    graph = None
    try:
        # Keep parameter autograd nodes on the same stream for warmup, capture, and replay.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.manual_seed(7)
            options = {
                "hidden_size": 32,
                "num_heads": 1,
                "head_dim": 128,
                "variant": variant,
                "backend": "fused",
                "compute_dtype": dtype,
                "device": device,
            }
            model = ContextParallelDeltaRuleAttention(**options, group=dist.group.WORLD)
            reference = DeltaRuleAttention(**options)
            reference.load_state_dict(model.state_dict())
            global_hidden = torch.randn(1, 128, 32, device=device)
            global_target = torch.randn_like(global_hidden)
            offsets, fragments = LAYOUTS[0]
            plan = ContextParallelPlan.from_fragments(offsets, fragments, rank)
            routing = plan.routing(device, **CAPS)
            ids = plan.global_token_ids(device)
            hidden = global_hidden[:, ids].clone().requires_grad_()
            target = global_target[:, ids].clone()
            with pytest.raises(ValueError, match="routing conv_history"):
                model(hidden, routing=plan.routing(device))
            if capture:
                for _ in range(2):
                    _step(model, hidden, target, routing)
                stream.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = _step(model, hidden, target, routing)

            # Return to the original routing too: no invocation may retain the preceding batch.
            for offsets, fragments in (*LAYOUTS, LAYOUTS[0]):
                plan = ContextParallelPlan.from_fragments(offsets, fragments, rank)
                new_routing = plan.routing(device, **CAPS)
                ids = plan.global_token_ids(device)
                with torch.no_grad():
                    hidden.copy_(global_hidden[:, ids])
                    target.copy_(global_target[:, ids])
                if capture:
                    for field in fields(routing):
                        destination = getattr(routing, field.name)
                        source = getattr(new_routing, field.name)
                        if isinstance(destination, torch.Tensor):
                            destination.copy_(source)
                        else:
                            assert destination == source
                    graph.replay()
                    actual = captured
                else:
                    actual = _step(model, hidden, target, new_routing)
                stream.synchronize()
                # Routing is a per-call input; the module must not retain it between batches.
                assert not any(isinstance(v, ContextParallelRouting) for v in vars(model).values())
                _assert_matches_reference(
                    model, reference, plan, global_hidden, global_target, offsets, actual
                )
                with torch.no_grad():
                    output_only = model(hidden, routing=new_routing)
                assert output_only.final_state is None and output_only.final_conv_state is None
                torch.testing.assert_close(output_only.hidden_states, actual[0].hidden_states)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize(device)
    finally:
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("variant", "dtype", "capture"),
    [
        pytest.param("kda", torch.bfloat16, False, id="kda-bf16-eager"),
        pytest.param("kda", torch.bfloat16, True, id="kda-bf16-cuda-graph"),
        pytest.param("kda", torch.float16, False, id="kda-fp16-eager"),
        pytest.param("kda", torch.float16, True, id="kda-fp16-cuda-graph"),
        # context_parallel_gdn is covered at kernel level; here GDN only adds gate plumbing.
        pytest.param("gdn", torch.bfloat16, True, id="gdn-bf16-cuda-graph"),
        pytest.param("gdn", torch.float16, False, id="gdn-fp16-eager"),
    ],
)
def test_same_module_uses_each_batch_routing(
    tmp_path: Path, variant: str, dtype: torch.dtype, capture: bool
) -> None:
    """One model must follow two incompatible layouts, including in-place graph metadata replay."""
    mp.spawn(
        _rank_main,
        args=((tmp_path / "nccl-init").as_uri(), dtype, capture, variant),
        nprocs=2,
        join=True,
    )
