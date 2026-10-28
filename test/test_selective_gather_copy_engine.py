"""Check the copy-engine block exchange against an all-gather reference on four GPUs."""

import datetime
import os
import random
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.checkpoint

from attn_gym.distributed.selective_gather import (
    BlockExchangePlan,
    BlockLocation,
    CopyEngineBlockExchange,
)
from attn_gym.testing.selective_gather import sliding_window_assembled_blocks

WORLD = 4


def _one_peer_domain() -> bool:
    return torch.cuda.device_count() >= WORLD and all(
        torch.cuda.can_device_access_peer(i, j)
        for i in range(WORLD)
        for j in range(WORLD)
        if i != j
    )


pytestmark = [
    pytest.mark.skipif(not _one_peer_domain(), reason="needs four GPUs with peer access"),
    pytest.mark.xdist_group("two-gpu"),
]

BLOCK_TOKENS = 3
BLOCKS_PER_RANK = 4
HEADS, K_DIM, V_DIM = 2, 8, 5
DTYPE = torch.bfloat16
LAYERS = 3


def _init(rank: int, store_path: str) -> torch.device:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    # A file store avoids racing other tests for a free TCP port.
    dist.init_process_group(
        "nccl", init_method=f"file://{store_path}", rank=rank, world_size=WORLD, device_id=device
    )
    return device


def _layout(name: str) -> list[list[BlockLocation]]:
    """Assembled blocks of every rank for a named pattern."""
    if name == "random":
        # Dense random reads: blocks reach several consumers, so the owner adds several
        # contributions per block.
        rng = random.Random(0)
        return [
            [
                BlockLocation(owner, block)
                for owner in range(WORLD)
                for block in range(BLOCKS_PER_RANK)
                if owner == consumer or rng.random() < 0.6
            ]
            for consumer in range(WORLD)
        ]
    window = {"sliding": 2, "head-tail": 1, "head-tail-full-chunk": 2}[name]
    return sliding_window_assembled_blocks(
        WORLD, BLOCKS_PER_RANK, window, head_tail=name != "sliding"
    )


def _gather(tensor: torch.Tensor) -> list[torch.Tensor]:
    out = tensor.new_empty((WORLD, *tensor.shape))
    dist.all_gather_into_tensor(out, tensor.contiguous())
    return list(out)


def _blocks(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(-1, BLOCK_TOKENS, *tensor.shape[1:])


def _reference_forward(parts, assembled_blocks, rank):
    """Assembled blocks filled from every rank's local blocks, and the indices they fill."""
    outs = []
    for part in parts:
        gathered = [_blocks(t) for t in _gather(part)]
        outs.append([gathered[owner][block] for owner, block in assembled_blocks[rank]])
    return outs, range(len(assembled_blocks[rank]))


def _reference_backward(grads, assembled_blocks, rank):
    """Owned gradient, then each consumer's gradient in send-slot (consumer) order."""
    positions = [{x: i for i, x in enumerate(blocks)} for blocks in assembled_blocks]
    consumers = [
        [c for c in range(WORLD) if c != rank and BlockLocation(rank, block) in positions[c]]
        for block in range(BLOCKS_PER_RANK)
    ]
    many = any(len(c) > 1 for c in consumers)
    result = []
    for grad in grads:
        gathered = [_blocks(t) for t in _gather(grad)]
        blocks = []
        for block in range(BLOCKS_PER_RANK):
            location = BlockLocation(rank, block)
            total = gathered[rank][positions[rank][location]]
            total = total.float() if many else total
            for consumer in consumers[block]:
                total = total + gathered[consumer][positions[consumer][location]]
            blocks.append(total.to(DTYPE))
        result.append(torch.stack(blocks).flatten(0, 1))
    return result


def _inputs(seed: int, device: torch.device):
    gen = torch.Generator(device=device).manual_seed(100 * seed + dist.get_rank())
    tokens = BLOCK_TOKENS * BLOCKS_PER_RANK
    k = torch.randn((tokens, HEADS, K_DIM), device=device, generator=gen).to(DTYPE)
    v = torch.randn((tokens, HEADS, V_DIM), device=device, generator=gen).to(DTYPE)
    return k.requires_grad_(True), v.requires_grad_(True)


def _exchange(name: str, rank: int, device: torch.device, group=None):
    assembled_blocks = _layout(name)
    plan = BlockExchangePlan.from_assembled_blocks(
        assembled_blocks, rank=rank, blocks_per_rank=BLOCKS_PER_RANK, block_tokens=BLOCK_TOKENS
    )
    exchange = CopyEngineBlockExchange(
        group or dist.group.WORLD,
        plan,
        k_token_shape=(HEADS, K_DIM),
        v_token_shape=(HEADS, V_DIM),
        dtype=DTYPE,
        device=device,
    )
    return exchange, assembled_blocks


def _apply(exchange, k, v, mode: str):
    if mode == "remat":
        # Checkpointing saves only the received blocks and recomputes the assembly.
        remote_k, remote_v = exchange.exchange(k, v)
        return torch.utils.checkpoint.checkpoint(
            exchange.assemble, k, v, remote_k, remote_v, use_reentrant=False
        )
    return exchange.exchange_and_assemble(k, v)


def _check(out, want, mapped):
    for got, ref in zip(out, want, strict=True):
        got_blocks = _blocks(got)
        for block in mapped:
            torch.testing.assert_close(got_blocks[block], ref[block], rtol=0, atol=0)


def _training_order_worker(rank: int, store_path: str, mode: str) -> None:
    device = _init(rank, store_path)
    try:
        for name in ("sliding", "head-tail", "head-tail-full-chunk", "random"):
            exchange, assembled_blocks = _exchange(name, rank, device)
            # Every layer's forward, then every backward in reverse. Layers share the landing
            # zones, so each write must wait until its reader freed them. Rank 1 reads slowly.
            # References first: no collective may resynchronize the ranks between calls.
            layers = [_inputs(seed, device) for seed in range(LAYERS)]
            gen = torch.Generator(device=device).manual_seed(rank)
            refs, grads = [], []
            for k, v in layers:
                refs.append(_reference_forward((k.detach(), v.detach()), assembled_blocks, rank))
            outs = []
            for k, v in layers:
                outs.append(_apply(exchange, k, v, mode))
                if rank == 1:
                    torch.cuda._sleep(20_000_000)
            for out_k, out_v in outs:
                grads.append(
                    (
                        torch.randn(out_k.shape, device=device, generator=gen).to(DTYPE),
                        torch.randn(out_v.shape, device=device, generator=gen).to(DTYPE),
                    )
                )
            for (out_k, out_v), grad in zip(reversed(outs), reversed(grads), strict=True):
                torch.autograd.backward((out_k, out_v), grad)
                if rank == 2:
                    torch.cuda._sleep(20_000_000)
            for out, (want, mapped) in zip(outs, refs, strict=True):
                _check(out, want, mapped)
            for (k, v), grad in zip(layers, grads, strict=True):
                want_k, want_v = _reference_backward(grad, assembled_blocks, rank)
                torch.testing.assert_close(k.grad, want_k, rtol=0, atol=0)
                torch.testing.assert_close(v.grad, want_v, rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _graph_worker(rank: int, store_path: str, mode: str) -> None:
    device = _init(rank, store_path)
    try:
        exchange, assembled_blocks = _exchange("head-tail", rank, device)
        k, v = (t.detach().requires_grad_(True) for t in _inputs(0, device))
        with torch.no_grad():
            out_k, out_v = exchange.exchange_and_assemble(k, v)
        grad_k, grad_v = torch.zeros_like(out_k), torch.zeros_like(out_v)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                out_k, out_v = _apply(exchange, k, v, mode)
                torch.autograd.grad((out_k, out_v), (k, v), (grad_k, grad_v))
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out_k, out_v = _apply(exchange, k, v, mode)
            dk, dv = torch.autograd.grad((out_k, out_v), (k, v), (grad_k, grad_v))
        gen = torch.Generator(device=device).manual_seed(rank)
        for seed in (1, 2, 3):
            new_k, new_v = _inputs(seed, device)
            with torch.no_grad():
                k.copy_(new_k), v.copy_(new_v)
                grad_k.copy_(torch.randn(grad_k.shape, device=device, generator=gen))
                grad_v.copy_(torch.randn(grad_v.shape, device=device, generator=gen))
            graph.replay()
            want, mapped = _reference_forward((k.detach(), v.detach()), assembled_blocks, rank)
            _check((out_k, out_v), want, mapped)
            want_k, want_v = _reference_backward((grad_k, grad_v), assembled_blocks, rank)
            torch.testing.assert_close(dk, want_k, rtol=0, atol=0)
            torch.testing.assert_close(dv, want_v, rtol=0, atol=0)
        graph.reset()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["fused", "remat"])
@pytest.mark.parametrize(
    "worker", [_training_order_worker, _graph_worker], ids=["training-order", "cuda-graph"]
)
def test_copy_engine_block_exchange_matches_reference(worker, mode: str, tmp_path: Path) -> None:
    _spawn(worker, str(tmp_path / "store"), mode, seconds=300)


def _desync_worker(rank: int, store_path: str, log_dir: str) -> None:
    """Rank 1 exchanges while the others never do: it must trap after the timeout, not hang."""
    device = _init(rank, store_path)
    # The exchange takes its signal timeout from the group's NCCL timeout.
    group = dist.new_group(list(range(WORLD)), timeout=datetime.timedelta(seconds=1))
    exchange, _ = _exchange("sliding", rank, device, group=group)
    dist.barrier()
    done = Path(log_dir) / "rank1.done"
    if rank != 1:
        # Stay alive, and out of the exchange, until rank 1 has timed out or the test gives up.
        deadline = time.monotonic() + 120
        while not done.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        os._exit(0)
    # The kernel reports through device printf, which writes to this process's fd 1.
    os.dup2(os.open(Path(log_dir) / "rank1.log", os.O_WRONLY | os.O_CREAT), 1)
    k, v = _inputs(0, device)
    start = time.monotonic()
    try:
        exchange.exchange_and_assemble(k, v)
        torch.cuda.synchronize()
    except RuntimeError:
        code = 0 if 1 <= time.monotonic() - start < 60 else 2
    else:
        code = 3  # the wait returned although no owner signalled
    done.touch()
    os._exit(code)


def test_desynced_ranks_trap_instead_of_hanging(tmp_path: Path) -> None:
    _spawn(_desync_worker, str(tmp_path / "store"), str(tmp_path), seconds=120)
    message = (tmp_path / "rank1.log").read_text()
    assert "SymmetricMemory::wait_signal: rank 1 failed to receive signal from rank 0" in message


def _spawn(worker, *args, seconds: int) -> None:
    """Run ``worker`` on every rank, and kill them all if they do not finish in time."""
    context = mp.spawn(worker, args=args, nprocs=WORLD, join=False)
    deadline = time.monotonic() + seconds
    try:
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, f"spawned ranks did not finish within {seconds} s"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
