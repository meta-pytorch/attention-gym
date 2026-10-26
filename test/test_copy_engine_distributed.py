"""Check the copy-engine all-gather and ring channel against NCCL references on two GPUs."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.multiprocessing as mp

from attn_gym.distributed import CopyEngineAllGather, CopyEngineChannel

pytestmark = [
    pytest.mark.skipif(
        torch.cuda.device_count() < 2, reason="needs two GPUs for symmetric memory"
    ),
    pytest.mark.xdist_group("two-gpu"),
]

WORLD = 2
PART_SHAPE = (1, 3, 16, 8)  # (batch, heads, sequence shard, head dim); gathered along dim 2


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init(rank: int, port: int) -> torch.device:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("nccl", rank=rank, world_size=WORLD, device_id=device)
    return device


def _parts(seed: int, device: torch.device) -> list[torch.Tensor]:
    """Rank-distinct K and V shards."""
    gen = torch.Generator(device=device).manual_seed(100 * seed + dist.get_rank())
    return [torch.randn(PART_SHAPE, device=device, generator=gen) for _ in range(2)]


def _reference_gather(parts: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(parts)  # (2, *PART_SHAPE)
    gathered = funcol.all_gather_single(stacked, 0, dist.group.WORLD)
    return (
        funcol.wait_tensor(gathered)
        .unflatten(0, (WORLD, 2))
        .permute(1, 2, 3, 0, 4, 5)
        .flatten(3, 4)
    )


def _all_gather_worker(rank: int, port: int) -> None:
    device = _init(rank, port)
    try:
        gather = CopyEngineAllGather(dist.group.WORLD, 2, PART_SHAPE, 2, torch.float32, device)
        # Repeated calls reuse the buffer: each must see this call's shards, not a stale one.
        for seed in range(3):
            parts = [x.requires_grad_(True) for x in _parts(seed, device)]
            gather.start(parts)
            out = gather.wait(parts)
            ref_parts = [x.detach().clone().requires_grad_(True) for x in parts]
            ref = _reference_gather(ref_parts)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
            gen = torch.Generator(device=device).manual_seed(seed)
            weight = torch.randn(out.shape, device=device, generator=gen)
            (out * weight).sum().backward()
            (ref * weight).sum().backward()
            for x, r in zip(parts, ref_parts, strict=True):
                torch.testing.assert_close(x.grad, r.grad)

        # A captured gather must move the values present at replay time.
        static = _parts(10, device)
        gather.start(static)
        gather.wait(static)
        graph = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            gather.start(static)
            gather.wait(static)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            gather.start(static)
            captured_out = gather.wait(static)
        for seed in (11, 12):
            for dst, src in zip(static, _parts(seed, device), strict=True):
                dst.copy_(src)
            graph.replay()
            torch.testing.assert_close(captured_out, _reference_gather(static), rtol=0, atol=0)
        graph.reset()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _channel_worker(rank: int, port: int) -> None:
    device = _init(rank, port)
    try:
        nxt, prv = (rank + 1) % WORLD, (rank - 1) % WORLD
        channel = CopyEngineChannel(
            dist.group.WORLD, prv, nxt, 2, PART_SHAPE, torch.float32, device
        )
        # Rotate a token through more steps than there are slots, so slots are reused.
        current = _parts(0, device)
        expected_owner = rank
        for step in range(5):
            channel.send(current, step)
            received = channel.recv(step + 1).clone()
            channel.release(step + 1)
            expected_owner = (expected_owner - 1) % WORLD
            gen = torch.Generator(device=device).manual_seed(expected_owner)
            want = [torch.randn(PART_SHAPE, device=device, generator=gen) for _ in range(2)]
            torch.testing.assert_close(received, torch.stack(want), rtol=0, atol=0)
            current = list(received)
        channel.join()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "worker", [_all_gather_worker, _channel_worker], ids=["all-gather", "channel"]
)
def test_copy_engine_matches_nccl(worker) -> None:
    mp.spawn(worker, args=(_free_port(),), nprocs=WORLD, join=True)
