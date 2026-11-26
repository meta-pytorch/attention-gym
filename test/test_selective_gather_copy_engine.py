"""Check the copy-engine block exchange against an all-gather reference on up to four GPUs."""

import os
import random
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.multiprocessing as mp
import torch.utils.checkpoint

from attn_gym.distributed.selective_gather import (
    BlockExchangePlan,
    BlockLocation,
    CopyEngineBlockExchange,
    build_sliding_window_plan,
)
from attn_gym.testing.selective_gather import local_ranges

WORLD = min(torch.cuda.device_count(), 4)


def _one_peer_domain() -> bool:
    return WORLD >= 2 and all(
        torch.cuda.can_device_access_peer(i, j)
        for i in range(WORLD)
        for j in range(WORLD)
        if i != j
    )


multi_gpu = pytest.mark.skipif(not _one_peer_domain(), reason="needs two GPUs with peer access")
pytestmark = pytest.mark.xdist_group("two-gpu")

BLOCK_TOKENS = 3
BLOCKS_PER_RANK = 4
HEADS, K_DIM, V_DIM = 2, 8, 5
DTYPE = torch.bfloat16
LAYERS = 3


def _init(rank: int, store_path: str, symm_mem_backend: str | None = None) -> torch.device:
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    # A file store avoids racing other tests for a free TCP port.
    dist.init_process_group(
        "nccl", init_method=f"file://{store_path}", rank=rank, world_size=WORLD, device_id=device
    )
    # None keeps the default backend. FSDP's symmetric-memory communication selects NCCL for
    # the whole process.
    if symm_mem_backend is not None:
        symm_mem.set_backend(symm_mem_backend)
    return device


def _plans(name: str) -> list[BlockExchangePlan]:
    """Every rank's plan for a named pattern."""
    if name == "random":
        # Dense random reads: blocks reach several consumers, so the owner adds several
        # contributions per block.
        rng = random.Random(0)
        assembled_blocks = [
            [
                BlockLocation(owner, block)
                for owner in range(WORLD)
                for block in range(BLOCKS_PER_RANK)
                if owner == consumer or rng.random() < 0.6
            ]
            for consumer in range(WORLD)
        ]
        return [
            BlockExchangePlan.from_assembled_blocks(
                assembled_blocks,
                rank=rank,
                blocks_per_rank=BLOCKS_PER_RANK,
                block_tokens=BLOCK_TOKENS,
            )
            for rank in range(WORLD)
        ]
    # A window of window_blocks * BLOCK_TOKENS + 1 tokens reaches exactly window_blocks blocks
    # before each chunk.
    window_blocks = {"sliding": 2, "head-tail": 1, "head-tail-full-chunk": 2}[name]
    ranges = local_ranges(
        WORLD * BLOCKS_PER_RANK * BLOCK_TOKENS, WORLD, head_tail=name != "sliding"
    )
    return [
        build_sliding_window_plan(
            ranges,
            rank=rank,
            block_tokens=BLOCK_TOKENS,
            sliding_window_size=window_blocks * BLOCK_TOKENS + 1,
        )[0]
        for rank in range(WORLD)
    ]


def _assembled_blocks(plans: list[BlockExchangePlan]) -> list[list[BlockLocation]]:
    """Each rank's assembled blocks in slot order, read from every rank's plan."""
    layouts = []
    for rank, plan in enumerate(plans):
        slots = {d: BlockLocation(rank, b) for b, d in enumerate(plan.local_destination_blocks)}
        for owner, owner_plan in enumerate(plans):
            for i in range(plan.recv_counts[owner]):
                block = owner_plan.send_source_blocks[owner_plan.send_offsets[rank] + i]
                slot = plan.remote_destination_blocks[plan.recv_offsets[owner] + i]
                slots[slot] = BlockLocation(owner, block)
        layouts.append([slots[slot] for slot in range(len(slots))])
    return layouts


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


def _exchange(name: str, rank: int, device: torch.device, **kwargs):
    plans = _plans(name)
    exchange = CopyEngineBlockExchange(
        dist.group.WORLD,
        plans[rank],
        k_token_shape=(HEADS, K_DIM),
        v_token_shape=(HEADS, V_DIM),
        dtype=DTYPE,
        device=device,
        **kwargs,
    )
    return exchange, _assembled_blocks(plans)


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


def _training_order_worker(
    rank: int, store_path: str, mode: str, symm_mem_backend: str | None
) -> None:
    device = _init(rank, store_path, symm_mem_backend)
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


def _graph_worker(rank: int, store_path: str, mode: str, symm_mem_backend: str | None) -> None:
    device = _init(rank, store_path, symm_mem_backend)
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


@multi_gpu
@pytest.mark.parametrize("symm_mem_backend", [None, "NCCL"], ids=["default", "nccl"])
@pytest.mark.parametrize("mode", ["fused", "remat"])
@pytest.mark.parametrize(
    "worker", [_training_order_worker, _graph_worker], ids=["training-order", "cuda-graph"]
)
def test_copy_engine_block_exchange_matches_reference(
    worker, mode: str, symm_mem_backend: str | None, tmp_path: Path
) -> None:
    _spawn(worker, str(tmp_path / "store"), mode, symm_mem_backend, seconds=300)


def _desync_worker(rank: int, store_path: str, log_dir: str) -> None:
    """Rank 1 exchanges while the others never do: it must trap after the timeout, not hang."""
    device = _init(rank, store_path)
    exchange, _ = _exchange("sliding", rank, device, timeout_ms=1000)
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
        code = 3  # the barrier returned although the other ranks never joined
    done.touch()
    os._exit(code)


@multi_gpu
def test_desynced_ranks_trap_instead_of_hanging(tmp_path: Path) -> None:
    _spawn(_desync_worker, str(tmp_path / "store"), str(tmp_path), seconds=120)
    message = (tmp_path / "rank1.log").read_text()
    assert "SymmetricMemory::barrier: rank 1 failed" in message


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("mode", ["fused", "remat"])
def test_fake_process_group_runs_local_work(mode: str) -> None:
    """Under a fake process group, local blocks still land and backward runs."""
    from torch.testing._internal.distributed.fake_pg import FakeStore

    rank, world_size = 1, 4
    dist.init_process_group("fake", rank=rank, world_size=world_size, store=FakeStore())
    try:
        ranges = local_ranges(
            world_size * BLOCKS_PER_RANK * BLOCK_TOKENS, world_size, head_tail=True
        )
        plan, _ = build_sliding_window_plan(
            ranges, rank=rank, block_tokens=BLOCK_TOKENS, sliding_window_size=BLOCK_TOKENS + 1
        )
        exchange = CopyEngineBlockExchange(
            dist.group.WORLD,
            plan,
            k_token_shape=(HEADS, K_DIM),
            v_token_shape=(HEADS, V_DIM),
            dtype=DTYPE,
            device=torch.device("cuda"),
        )
        k, v = _inputs(0, torch.device("cuda"))
        out = _apply(exchange, k, v, mode)
        for got, local in zip(out, (k, v), strict=True):
            got_blocks = _blocks(got)
            for block, destination in enumerate(plan.local_destination_blocks):
                torch.testing.assert_close(
                    got_blocks[destination], _blocks(local)[block], rtol=0, atol=0
                )
        torch.autograd.backward(out, [torch.ones_like(t) for t in out])
        assert k.grad.isfinite().all() and v.grad.isfinite().all()
    finally:
        dist.destroy_process_group()
