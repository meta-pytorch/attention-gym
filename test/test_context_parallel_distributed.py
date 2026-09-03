"""Two-GPU NCCL check of the public context-parallel recipe against the unsharded ops.

The single-GPU stages tests drive ``prepare``/``run`` and the fold helpers directly; this test
runs ``context_parallel_kda``/``context_parallel_gdn`` themselves, so the autograd Function, the
all-gather, and the ``d_final_state=None`` backward path are exercised end to end.
"""

from __future__ import annotations

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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _rank_main(rank: int, world: int, port: int, op_name: str, final_state_loss: bool) -> None:
    from attn_gym.linear.context_parallel import ContextParallelPlan
    from attn_gym.linear.gdn import chunk_gdn, context_parallel_gdn
    from attn_gym.linear.kda import chunk_kda, context_parallel_kda

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device)
    try:
        tokens = CU_SEQLENS[-1]
        generator = torch.Generator(device=device).manual_seed(0)
        randn = partial(torch.randn, device=device, generator=generator)
        q = torch.nn.functional.normalize(randn(1, tokens, HEADS, HEAD_DIM), dim=-1).bfloat16()
        k = torch.nn.functional.normalize(randn(1, tokens, HEADS, HEAD_DIM), dim=-1).bfloat16()
        v = randn(1, tokens, HEADS, HEAD_DIM).bfloat16()
        gate_shape = (1, tokens, HEADS) + ((HEAD_DIM,) if op_name == "kda" else ())
        gate = -torch.rand(gate_shape, device=device, generator=generator)
        beta = torch.rand(1, tokens, HEADS, device=device, generator=generator)
        d_output = randn(1, tokens, HEADS, HEAD_DIM).bfloat16()
        d_final = randn(len(CU_SEQLENS) - 1, HEADS, HEAD_DIM, HEAD_DIM)

        # Unsharded oracle with the same loss structure.
        inputs = tuple(t.clone().requires_grad_() for t in (q, k, v, gate, beta))
        chunk = chunk_kda if op_name == "kda" else chunk_gdn
        offsets = torch.tensor(CU_SEQLENS, dtype=torch.int32, device=device)
        output, final = chunk(*inputs, cu_seqlens=offsets, output_final_state=True)
        outputs = (output, final) if final_state_loss else (output,)
        cotangents = (d_output, d_final) if final_state_loss else (d_output,)
        expected_grads = torch.autograd.grad(outputs, inputs, grad_outputs=cotangents)

        # Zig-zag: rank r owns blocks r and 2 * world - 1 - r of 2 * world equal blocks.
        block = tokens // (2 * world)
        fragments = [
            [(r * block, (r + 1) * block), ((2 * world - 1 - r) * block, (2 * world - r) * block)]
            for r in range(world)
        ]
        plan = ContextParallelPlan.from_fragments(CU_SEQLENS, fragments, rank)
        ids = plan.global_token_ids(device)
        local = tuple(t[:, ids].clone().requires_grad_() for t in (q, k, v, gate, beta))
        cp = context_parallel_kda if op_name == "kda" else context_parallel_gdn
        local_output, exit_states = cp(
            *local,
            cu_seqlens=torch.tensor(plan.cu_seqlens, dtype=torch.int32, device=device),
            plan=plan,
            group=dist.group.WORLD,
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
def test_context_parallel_matches_unsharded_op_on_two_ranks(op_name, final_state_loss):
    mp.spawn(_rank_main, args=(2, _free_port(), op_name, final_state_loss), nprocs=2, join=True)
