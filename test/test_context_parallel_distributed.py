"""Two-GPU NCCL check of the public context-parallel recipe against the unsharded ops.

The single-GPU stages tests drive ``prepare``/``run`` and the fold helpers directly; this test
runs ``context_parallel_kda``/``context_parallel_gdn`` themselves, so the autograd Function, the
all-gather, and the ``d_final_state=None`` backward path are exercised end to end.
"""

from __future__ import annotations

import dataclasses
import os
import socket
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two visible CUDA devices"
)

CU_SEQLENS = (0, 40, 232, 384)
HEADS, HEAD_DIM = 2, 128
TOKENS = CU_SEQLENS[-1]

# Fragment tables over two ranks. Zig-zag is the training layout; "uneven" gives rank 0 three
# fragments of one sequence (a chain through three slots on one rank) and rank 1 two, so rank 1
# gathers a padding slot; "empty-document" adds a zero-length sequence to the stream.
TABLES = {
    "zigzag": (CU_SEQLENS, [[(0, 96), (288, 384)], [(96, 192), (192, 288)]]),
    "uneven": (CU_SEQLENS, [[(0, 64), (128, 192), (256, 384)], [(64, 128), (192, 256)]]),
    "empty-document": ((0, 40, 40, 100, 384), [[(0, 96), (288, 384)], [(96, 192), (192, 288)]]),
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _rank_main(
    rank: int, world: int, port: int, op_name: str, final_state_loss: bool, table: str
) -> None:
    from attn_gym.linear.context_parallel import ContextParallelPlan
    from attn_gym.linear.gdn import chunk_gdn, context_parallel_gdn
    from attn_gym.linear.kda import chunk_kda, context_parallel_kda

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    try:
        cu_seqlens, fragments = TABLES[table]
        tokens = TOKENS
        generator = torch.Generator(device=device).manual_seed(0)
        randn = partial(torch.randn, device=device, generator=generator)
        q = torch.nn.functional.normalize(randn(1, tokens, HEADS, HEAD_DIM), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(randn(1, tokens, HEADS, HEAD_DIM), dim=-1).bfloat16()
        v = randn(1, tokens, HEADS, HEAD_DIM).bfloat16()
        gate_shape = (1, tokens, HEADS) + ((HEAD_DIM,) if op_name == "kda" else ())
        gate = -torch.rand(gate_shape, device=device, generator=generator)
        beta = torch.rand(1, tokens, HEADS, device=device, generator=generator)
        d_output = randn(1, tokens, HEADS, HEAD_DIM).bfloat16()
        d_final = randn(len(cu_seqlens) - 1, HEADS, HEAD_DIM, HEAD_DIM)

        # Unsharded oracle with the same loss structure.
        inputs = tuple(t.clone().requires_grad_() for t in (q, k, v, gate, beta))
        chunk = chunk_kda if op_name == "kda" else chunk_gdn
        offsets = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        output, final = chunk(*inputs, cu_seqlens=offsets, output_final_state=True)
        outputs = (output, final) if final_state_loss else (output,)
        cotangents = (d_output, d_final) if final_state_loss else (d_output,)
        expected_grads = torch.autograd.grad(outputs, inputs, grad_outputs=cotangents)

        plan = ContextParallelPlan.from_fragments(cu_seqlens, fragments, rank)
        ids = plan.global_token_ids(device)
        local = tuple(t[:, ids].clone().requires_grad_() for t in (q, k, v, gate, beta))
        cp = context_parallel_kda if op_name == "kda" else context_parallel_gdn
        local_output, exit_states = cp(
            *local, routing=plan.routing(device), group=dist.group.WORLD
        )
        local_outputs = (local_output,)
        local_cotangents = [d_output[:, ids]]
        if final_state_loss:
            d_exit = torch.zeros_like(exit_states)
            for index in plan.terminal:
                d_exit[index] = d_final[plan.subsequences[index].sequence]
            local_outputs = (local_output, exit_states)
            local_cotangents.append(d_exit)
        grads = torch.autograd.grad(local_outputs, local, grad_outputs=local_cotangents)

        eps = torch.finfo(torch.bfloat16).eps
        for actual, expected in zip(
            (local_output, *grads), (output[:, ids], *(g[:, ids] for g in expected_grads))
        ):
            expected = expected.float()
            torch.testing.assert_close(
                actual.float(),
                expected,
                atol=eps * max(expected.abs().max().item(), 1.0),
                rtol=eps,
            )
        for index in plan.terminal:
            torch.testing.assert_close(
                exit_states[index], final[plan.subsequences[index].sequence], atol=eps, rtol=eps
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("op_name", ["kda", "gdn"])
@pytest.mark.parametrize("final_state_loss", [True, False], ids=["with-final", "output-only"])
@pytest.mark.parametrize("table", list(TABLES))
def test_context_parallel_matches_unsharded_op_on_two_ranks(op_name, final_state_loss, table):
    mp.spawn(
        _rank_main, args=(2, _free_port(), op_name, final_state_loss, table), nprocs=2, join=True
    )


# Three document layouts of one 384-token stream: the fragment table (zig-zag over two ranks)
# never changes, so one captured graph must serve all of them.
LAYOUTS = ((0, 40, 232, 384), (0, 100, 160, 180, 384), (0, 384))
MAX_SUBSEQUENCES = 6


def _replay_main(rank: int, world: int, port: int, op_name: str) -> None:
    from attn_gym.linear.context_parallel import ContextParallelPlan
    from attn_gym.linear.gdn import chunk_gdn, context_parallel_gdn
    from attn_gym.linear.kda import chunk_kda, context_parallel_kda

    chunk = chunk_kda if op_name == "kda" else chunk_gdn
    cp = context_parallel_kda if op_name == "kda" else context_parallel_gdn

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    graph = None
    try:
        tokens = LAYOUTS[0][-1]
        block = tokens // (2 * world)
        fragments = [
            [(r * block, (r + 1) * block), ((2 * world - 1 - r) * block, (2 * world - r) * block)]
            for r in range(world)
        ]
        generator = torch.Generator(device=device).manual_seed(1)
        randn = partial(torch.randn, device=device, generator=generator)

        gate_shape = (1, tokens, HEADS) + ((HEAD_DIM,) if op_name == "kda" else ())

        def sample():
            q = torch.nn.functional.normalize(randn(1, tokens, HEADS, HEAD_DIM), dim=-1).bfloat16()
            k = torch.nn.functional.normalize(randn(1, tokens, HEADS, HEAD_DIM), dim=-1).bfloat16()
            v = randn(1, tokens, HEADS, HEAD_DIM).bfloat16()
            gate = -torch.rand(gate_shape, device=device, generator=generator)
            beta = torch.rand(1, tokens, HEADS, device=device, generator=generator)
            d_output = randn(1, tokens, HEADS, HEAD_DIM).bfloat16()
            # One cotangent per possible document; the layout decides which rows are used.
            d_final = randn(MAX_SUBSEQUENCES, HEADS, HEAD_DIM, HEAD_DIM)
            return (q, k, v, gate, beta), d_output, d_final

        # Static buffers the graph reads: the span inputs, the routing tensors, the cotangents.
        # The exit-state cotangent is per span segment and zero except on terminal rows, which
        # is how a caller masks the final-state loss to true document ends under replay.
        plan = ContextParallelPlan.from_fragments(LAYOUTS[0], fragments, rank)
        routing = plan.routing(device, max_subsequences=MAX_SUBSEQUENCES)
        ids = plan.global_token_ids(device)  # the same for every layout: the table is fixed
        (q, k, v, gate, beta), d_output, _ = sample()
        local = [t[:, ids].clone().requires_grad_() for t in (q, k, v, gate, beta)]
        local_d_output = d_output[:, ids].clone()
        local_d_exit = torch.zeros(MAX_SUBSEQUENCES, HEADS, HEAD_DIM, HEAD_DIM, device=device)

        def step():
            output, exit_states = cp(*local, routing=routing, group=dist.group.WORLD)
            assert exit_states.shape[0] == MAX_SUBSEQUENCES
            grads = torch.autograd.grad(
                (output, exit_states), local, grad_outputs=(local_d_output, local_d_exit)
            )
            return output, exit_states, grads

        # Warm up on a side stream (compiles and allocates), then capture once.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        dist.barrier()

        eps = torch.finfo(torch.bfloat16).eps
        for layout in LAYOUTS[::-1]:  # start away from the captured layout
            # A new batch: fresh inputs and a new document layout of the same fragment table.
            (q, k, v, gate, beta), d_output, d_final = sample()
            for buffer, source in zip(local, (q, k, v, gate, beta), strict=True):
                buffer.data.copy_(source[:, ids])
            local_d_output.copy_(d_output[:, ids])
            new_plan = ContextParallelPlan.from_fragments(layout, fragments, rank)
            new_routing = new_plan.routing(device, max_subsequences=MAX_SUBSEQUENCES)
            for field in dataclasses.fields(new_routing):
                value = getattr(new_routing, field.name)
                if isinstance(value, torch.Tensor):
                    getattr(routing, field.name).copy_(value)
            local_d_exit.zero_()
            for index in new_plan.terminal:
                local_d_exit[index] = d_final[new_plan.subsequences[index].sequence]
            graph.replay()
            torch.cuda.synchronize()

            inputs = tuple(t.clone().requires_grad_() for t in (q, k, v, gate, beta))
            offsets = torch.tensor(layout, dtype=torch.int32, device=device)
            output, final = chunk(*inputs, cu_seqlens=offsets, output_final_state=True)
            expected_grads = torch.autograd.grad(
                (output, final),
                inputs,
                grad_outputs=(d_output, d_final[: final.shape[0]]),
            )
            actual_output, exit_states, actual_grads = captured
            for actual, expected in zip(
                (actual_output, *actual_grads),
                (output[:, ids], *(g[:, ids] for g in expected_grads)),
                strict=True,
            ):
                expected = expected.float()
                torch.testing.assert_close(
                    actual.float(),
                    expected,
                    atol=eps * max(expected.abs().max().item(), 1.0),
                    rtol=eps,
                )
            for index in new_plan.terminal:
                torch.testing.assert_close(
                    exit_states[index],
                    final[new_plan.subsequences[index].sequence],
                    atol=eps,
                    rtol=eps,
                )
    finally:
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


@pytest.mark.parametrize("op_name", ["kda", "gdn"])
def test_one_captured_graph_replays_across_document_layouts(op_name):
    mp.spawn(_replay_main, args=(2, _free_port(), op_name), nprocs=2, join=True)
