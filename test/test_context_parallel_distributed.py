"""Two-GPU NCCL check of the public context-parallel recipe against the unsharded ops.

The single-GPU stages tests drive ``prepare``/``run`` and the fold helpers directly; this test
runs ``context_parallel_kda``/``context_parallel_gdn`` themselves, so the autograd Function, the
all-gather, and the ``d_final_state=None`` backward path are exercised end to end.
"""

from __future__ import annotations

import dataclasses
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("cutlass")

from attn_gym.linear.context_parallel import ContextParallelPlan, context_parallel_conv_history
from attn_gym.linear.gdn import chunk_gdn, context_parallel_gdn
from attn_gym.linear.kda import chunk_kda, context_parallel_kda
from attn_gym.testing.kda import assert_relative_rms_within

# Every case spawns two ranks that pin cuda:0 and cuda:1, so xdist must not run them concurrently.
pytestmark = [
    pytest.mark.skipif(
        torch.cuda.device_count() < 2
        or any(torch.cuda.get_device_capability(i) < (8, 0) for i in range(2)),
        reason="needs two CUDA devices of capability 8.0 or newer",
    ),
    pytest.mark.xdist_group("two-gpu"),
]

CU_SEQLENS = (0, 40, 232, 384)
HEADS, HEAD_DIM = 2, 128
# The unsharded op and its context-parallel binding.
OPS = {"kda": (chunk_kda, context_parallel_kda), "gdn": (chunk_gdn, context_parallel_gdn)}

# Fragment tables over two cp ranks. Zig-zag is the training layout; "uneven" is one 384-token
# sequence where cp rank 0 owns three fragments and cp rank 1 two, so its chain alternates ranks
# through all five slots and cp rank 1 gathers a padding slot; "empty-document" adds a
# zero-length sequence to the stream.
TABLES = {
    "zigzag": (CU_SEQLENS, [[(0, 96), (288, 384)], [(96, 192), (192, 288)]]),
    "uneven": ((0, 384), [[(0, 64), (128, 192), (256, 384)], [(64, 128), (192, 256)]]),
    "empty-document": ((0, 40, 40, 100, 384), [[(0, 96), (288, 384)], [(96, 192), (192, 288)]]),
}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@contextmanager
def _process_group(cp_rank: int, world: int, port: int) -> Iterator[torch.device]:
    device = torch.device("cuda", cp_rank)
    torch.cuda.set_device(device)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=cp_rank, world_size=world, device_id=device)
    try:
        yield device
    finally:
        dist.destroy_process_group()


def _sample(op_name: str, tokens: int, documents: int, generator: torch.Generator):
    """Global operands and cotangents; both ranks seed the same generator, so they agree."""
    device = generator.device
    randn = partial(torch.randn, device=device, generator=generator)
    q = torch.nn.functional.normalize(randn(1, tokens, HEADS, HEAD_DIM), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(randn(1, tokens, HEADS, HEAD_DIM), dim=-1).bfloat16()
    v = randn(1, tokens, HEADS, HEAD_DIM).bfloat16()
    gate_shape = (1, tokens, HEADS) + ((HEAD_DIM,) if op_name == "kda" else ())
    gate = -torch.rand(gate_shape, device=device, generator=generator)
    beta = torch.rand(1, tokens, HEADS, device=device, generator=generator)
    d_output = randn(1, tokens, HEADS, HEAD_DIM).bfloat16()
    d_final = randn(documents, HEADS, HEAD_DIM, HEAD_DIM)
    return (q, k, v, gate, beta), d_output, d_final


def _exit_cotangent(plan: ContextParallelPlan, d_final: torch.Tensor, rows: int) -> torch.Tensor:
    """Per-segment exit-state cotangent: the document's loss on terminal rows, zero elsewhere."""
    d_exit = d_final.new_zeros(rows, *d_final.shape[1:])
    for index in plan.terminal:
        d_exit[index] = d_final[plan.subsequences[index].sequence]
    return d_exit


def _assert_matches_unsharded(
    plan: ContextParallelPlan,
    ids: torch.Tensor,
    sharded: tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor],
    unsharded: tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor],
) -> None:
    """``sharded = (output, grads, exit_states)`` on the span; ``unsharded`` the global oracle.

    Pointwise within bf16 eps of the reference's magnitude, and relative RMS within 2 eps.
    """
    output, grads, exit_states = sharded
    expected_output, expected_grads, final = unsharded
    eps = torch.finfo(torch.bfloat16).eps
    for name, actual, expected in zip(
        ("output", "dq", "dk", "dv", "dgate", "dbeta"),
        (output, *grads),
        (expected_output[:, ids], *(g[:, ids] for g in expected_grads)),
        strict=True,
    ):
        actual, expected = actual.float(), expected.float()
        torch.testing.assert_close(
            actual, expected, atol=eps * max(expected.abs().max().item(), 1.0), rtol=eps
        )
        assert_relative_rms_within(actual, expected, name, max_eps=2.0)
    for index in plan.terminal:
        expected = final[plan.subsequences[index].sequence]
        torch.testing.assert_close(exit_states[index], expected, atol=eps, rtol=eps)
        assert_relative_rms_within(exit_states[index], expected, "final state", max_eps=2.0)


def _rank_main(
    cp_rank: int, world: int, port: int, op_name: str, final_state_loss: bool, table: str
) -> None:
    chunk, cp = OPS[op_name]
    cu_seqlens, fragments = TABLES[table]
    with _process_group(cp_rank, world, port) as device:
        generator = torch.Generator(device=device).manual_seed(0)
        operands, d_output, d_final = _sample(
            op_name, cu_seqlens[-1], len(cu_seqlens) - 1, generator
        )

        # Unsharded oracle with the same loss structure.
        inputs = tuple(t.clone().requires_grad_() for t in operands)
        offsets = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)
        output, final = chunk(*inputs, cu_seqlens=offsets, output_final_state=True)
        outputs = (output, final) if final_state_loss else (output,)
        cotangents = (d_output, d_final) if final_state_loss else (d_output,)
        expected_grads = torch.autograd.grad(outputs, inputs, grad_outputs=cotangents)

        plan = ContextParallelPlan.from_fragments(cu_seqlens, fragments, cp_rank)
        ids = plan.global_token_ids(device)
        local = tuple(t[:, ids].clone().requires_grad_() for t in operands)
        local_output, exit_states = cp(
            *local, routing=plan.routing(device), group=dist.group.WORLD
        )
        local_outputs = (local_output,)
        local_cotangents = [d_output[:, ids]]
        if final_state_loss:
            local_outputs = (local_output, exit_states)
            local_cotangents.append(_exit_cotangent(plan, d_final, exit_states.shape[0]))
        grads = torch.autograd.grad(local_outputs, local, grad_outputs=local_cotangents)
        _assert_matches_unsharded(
            plan, ids, (local_output, grads, exit_states), (output, expected_grads, final)
        )


@pytest.mark.parametrize("op_name", list(OPS))
@pytest.mark.parametrize("final_state_loss", [True, False], ids=["with-final", "output-only"])
@pytest.mark.parametrize("table", list(TABLES))
def test_context_parallel_matches_unsharded_op_on_two_ranks(op_name, final_state_loss, table):
    mp.spawn(
        _rank_main, args=(2, _free_port(), op_name, final_state_loss, table), nprocs=2, join=True
    )


# One 384-token stream under three document layouts and three fragment tables (zig-zag, contiguous,
# three fragments per rank). Each rank always owns 192 tokens, so with the routing caps below one
# captured graph must serve every (layout, table) pair.
LAYOUTS = ((0, 40, 232, 384), (0, 100, 160, 180, 384), (0, 384))
REPLAY_TABLES = (
    [[(0, 96), (288, 384)], [(96, 192), (192, 288)]],
    [[(0, 192)], [(192, 384)]],
    [[(0, 64), (128, 192), (256, 320)], [(64, 128), (192, 256), (320, 384)]],
)
SLOTS, MAX_SUBSEQUENCES = 3, 6
# The short-convolution halo is captured in the same graph on its own [1, T, C] activations.
CONV_HISTORY, CHANNELS = 3, 8


def _conv_history_oracle(
    layout: tuple[int, ...],
    fragments: list[list[tuple[int, int]]],
    cp_rank: int,
    qkv: torch.Tensor,
    d_histories: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read each local segment's preceding tokens straight off the global stream.

    Also returns the global gradient of every rank's history loss: each rank's cotangent scattered
    onto the tokens its histories read, which the collective's backward must deliver to the owner.
    """
    history = qkv.new_zeros(MAX_SUBSEQUENCES, CONV_HISTORY, qkv.shape[-1])
    d_qkv = torch.zeros_like(qkv)
    for owner, d_history in enumerate(d_histories):
        plan = ContextParallelPlan.from_fragments(layout, fragments, owner)
        for index, piece in enumerate(plan.subsequences):
            available = min(CONV_HISTORY, piece.start - layout[piece.sequence])
            preceding = slice(piece.start - available, piece.start)
            if owner == cp_rank:
                history[index, CONV_HISTORY - available :] = qkv[0, preceding]
            d_qkv[0, preceding] += d_history[index, CONV_HISTORY - available :]
    return history, d_qkv


def _replay_main(cp_rank: int, world: int, port: int, op_name: str) -> None:
    chunk, cp = OPS[op_name]
    assert world == 2, "the tables above are written for two ranks"
    tokens = LAYOUTS[0][-1]
    with _process_group(cp_rank, world, port) as device:
        generator = torch.Generator(device=device).manual_seed(1)

        def sample():
            # One cotangent per possible document; the layout decides which rows are used. Every
            # rank draws every rank's history cotangent to build the oracle gradient.
            operands, d_output, d_final = _sample(op_name, tokens, MAX_SUBSEQUENCES, generator)
            qkv = torch.randn(1, tokens, CHANNELS, device=device, generator=generator)
            d_histories = torch.randn(
                world, MAX_SUBSEQUENCES, CONV_HISTORY, CHANNELS, device=device, generator=generator
            )
            return operands, d_output, d_final, qkv, d_histories

        # Static buffers the graph reads: the span inputs, the routing tensors, the cotangents.
        # The exit-state cotangent is per span segment and zero except on terminal rows, which
        # is how a caller masks the final-state loss to true document ends under replay.
        caps = {"slots": SLOTS, "max_subsequences": MAX_SUBSEQUENCES, "conv_history": CONV_HISTORY}
        plan = ContextParallelPlan.from_fragments(LAYOUTS[0], REPLAY_TABLES[0], cp_rank)
        routing = plan.routing(device, **caps)
        ids = plan.global_token_ids(device)
        operands, d_output, _, qkv, d_histories = sample()
        local = [t[:, ids].clone().requires_grad_() for t in (*operands, qkv)]
        local_d_output = d_output[:, ids].clone()
        local_d_exit = torch.zeros(MAX_SUBSEQUENCES, HEADS, HEAD_DIM, HEAD_DIM, device=device)
        local_d_history = d_histories[cp_rank].clone()

        def step():
            *span, local_qkv = local
            output, exit_states = cp(*span, routing=routing, group=dist.group.WORLD)
            assert exit_states.shape[0] == MAX_SUBSEQUENCES
            grads = torch.autograd.grad(
                (output, exit_states), span, grad_outputs=(local_d_output, local_d_exit)
            )
            history = context_parallel_conv_history(local_qkv, routing, dist.group.WORLD)
            (d_qkv,) = torch.autograd.grad(history, local_qkv, grad_outputs=local_d_history)
            return output, exit_states, grads, history, d_qkv

        # Warm up on a side stream (compiles and allocates), then capture once.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                captured = step()
            dist.barrier()

            # Start away from the captured pair; every step changes both documents and table.
            for layout, fragments in zip(LAYOUTS[::-1], REPLAY_TABLES[::-1], strict=True):
                new_plan = ContextParallelPlan.from_fragments(layout, fragments, cp_rank)
                ids = new_plan.global_token_ids(device)
                operands, d_output, d_final, qkv, d_histories = sample()
                with torch.no_grad():
                    for buffer, source in zip(local, (*operands, qkv), strict=True):
                        buffer.copy_(source[:, ids])
                local_d_output.copy_(d_output[:, ids])
                local_d_history.copy_(d_histories[cp_rank])
                new_routing = new_plan.routing(device, **caps)
                for field in dataclasses.fields(new_routing):
                    value = getattr(new_routing, field.name)
                    if isinstance(value, torch.Tensor):
                        getattr(routing, field.name).copy_(value)
                local_d_exit.copy_(_exit_cotangent(new_plan, d_final, MAX_SUBSEQUENCES))
                graph.replay()
                torch.cuda.synchronize()

                inputs = tuple(t.clone().requires_grad_() for t in operands)
                offsets = torch.tensor(layout, dtype=torch.int32, device=device)
                output, final = chunk(*inputs, cu_seqlens=offsets, output_final_state=True)
                expected_grads = torch.autograd.grad(
                    (output, final), inputs, grad_outputs=(d_output, d_final[: final.shape[0]])
                )
                actual_output, exit_states, actual_grads, history, d_qkv = captured
                _assert_matches_unsharded(
                    new_plan,
                    ids,
                    (actual_output, actual_grads, exit_states),
                    (output, expected_grads, final),
                )
                expected_history, expected_d_qkv = _conv_history_oracle(
                    layout, fragments, cp_rank, qkv, d_histories
                )
                torch.testing.assert_close(history, expected_history)
                torch.testing.assert_close(d_qkv, expected_d_qkv[:, ids])
        finally:
            graph.reset()


@pytest.mark.parametrize("op_name", list(OPS))
def test_one_captured_graph_replays_across_document_layouts(op_name):
    mp.spawn(_replay_main, args=(2, _free_port(), op_name), nprocs=2, join=True)
