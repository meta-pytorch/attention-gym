# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Copy-engine transfers into symmetric memory.

PyTorch symmetric memory maps each rank's buffer into its peers' address spaces, so a plain
``Tensor.copy_`` from a local tensor into a peer's buffer is a copy-engine transfer over NVLink:
the payload moves without occupying SMs (only small signal kernels launch), and it is not limited by
NCCL's per-peer channel count. It still shares memory and NVLink bandwidth with other work. On GB300 a contiguous push reaches ~820 GB/s, versus ~330 GB/s for a
default NCCL send/recv and ~430 GB/s for an NCCL all-gather.

Every rank in the group must share one NVLink domain: one node, or a multi-node NVLink system
such as GB300 NVL72. Copies are issued on a side stream; only contiguous source/destination
pairs are copied, because PyTorch lowers strided device copies to an SM kernel.

Ranks synchronize through the symmetric-memory signal pad, one flag per (peer, channel):
``READY`` (sender -> receiver: the data has landed) and ``FREE`` (receiver -> sender: the buffer
may be overwritten). All signals are queued on CUDA streams, so both classes can be captured in a
CUDA graph and replayed.
"""

import math
import os
from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

READY, FREE = 0, 1


def _channel(slot: int, signal: int) -> int:
    return 2 * slot + signal


class CopyEngineChannel:
    """Double-buffered pushes to ``dst`` and receives from ``src`` (ranks within ``group``).

    Each rank owns two receive slots of shape ``(num_parts, *part_shape)`` that ``src`` writes
    alternately, one per step. A receiver frees a slot only after its compute (and any onward
    push from that slot) is queued, so a slot is never overwritten while it is being read.

    Per step ``i``: ``send(parts, i)`` pushes into the slot ``dst`` reads at ``i + 1``,
    ``recv(i)`` waits for and returns this rank's slot for ``i``, and ``release(i)`` hands it
    back. Call ``join()`` before work that must see every pending push finished (for example the
    end of a CUDA graph capture).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        src: int,
        dst: int,
        num_parts: int,
        part_shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.src, self.dst = src, dst
        shape = (2, num_parts, *part_shape)
        self.slots = symm_mem.empty(shape, dtype=dtype, device=device)
        self.handle = symm_mem.rendezvous(self.slots, group.group_name)
        self.peer_slots = self.handle.get_buffer(dst, shape, dtype)
        self.stream = torch.cuda.Stream(device)
        for slot in range(2):
            self.handle.put_signal(src, _channel(slot, FREE))

    def send(self, parts: Sequence[torch.Tensor], step: int) -> None:
        slot = (step + 1) % 2
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.no_grad():
            self.handle.wait_signal(self.dst, _channel(slot, FREE))
            for peer_part, part in zip(self.peer_slots[slot], parts, strict=True):
                peer_part.copy_(part)
            self.handle.put_signal(self.dst, _channel(slot, READY))
        for part in parts:
            part.record_stream(self.stream)

    def recv(self, step: int) -> torch.Tensor:
        slot = step % 2
        self.handle.wait_signal(self.src, _channel(slot, READY))
        return self.slots[slot]

    def release(self, step: int) -> None:
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.handle.put_signal(self.src, _channel(step % 2, FREE))

    def join(self) -> None:
        torch.cuda.current_stream().wait_stream(self.stream)


class CopyEngineAllGather:
    """All-gather into a persistent symmetric buffer, written directly in the consumer's layout.

    Each rank contributes ``num_parts`` tensors of ``part_shape`` (for example K and V). The
    output has shape ``(num_parts, *part_shape)`` with ``part_shape[dim]`` multiplied by the world
    size, and rank ``r``'s shard of every part lands at offset ``r * part_shape[dim]`` along
    ``dim``: there is no gathered-then-reordered intermediate. Every rank pushes its shard into
    every peer's buffer (one contiguous copy per row in front of ``dim``), so the side stream does
    ``P * num_parts * prod(part_shape[:dim])`` copies.

    ``start(parts)`` issues the pushes; ``wait(parts)`` returns the gathered buffer. When any part
    requires grad, ``wait`` is differentiable: its backward reduce-scatters the gradient over NCCL.
    Every ``start`` first tells peers this rank's buffer is free, then waits for each peer's free
    signal before writing into it. The signal is queued behind everything already on the current
    stream (including any backward that reads the previous gather), so one gather may be
    outstanding at a time, and every call issues the same signals, so a captured ``start`` replays
    correctly whether or not the gather ran eagerly first.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_parts: int,
        part_shape: Sequence[int],
        dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.group = group
        self.rank, self.world_size = group.rank(), group.size()
        self.part_shape = tuple(part_shape)
        self.dim = dim
        self.rows = math.prod(part_shape[:dim])
        self.row_len = math.prod(part_shape[dim:])
        out_shape = (num_parts, self.rows, self.world_size * self.row_len)
        self.flat = symm_mem.empty(out_shape, dtype=dtype, device=device)
        self.handle = symm_mem.rendezvous(self.flat, group.group_name)
        self.peer_flat = [
            self.flat if peer == self.rank else self.handle.get_buffer(peer, out_shape, dtype)
            for peer in range(self.world_size)
        ]
        gathered_part = list(part_shape)
        gathered_part[dim] *= self.world_size
        self.buffer = self.flat.view(num_parts, *gathered_part)
        self.stream = torch.cuda.Stream(device)

    def peers(self) -> list[int]:
        return [(self.rank + i) % self.world_size for i in range(1, self.world_size)]

    def start(self, parts: Sequence[torch.Tensor]) -> None:
        lo, hi = self.rank * self.row_len, (self.rank + 1) * self.row_len
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream), torch.no_grad():
            # Everything that read the previous gather is queued before this point.
            for peer in self.peers():
                self.handle.put_signal(peer, _channel(0, FREE))
            for peer in [self.rank, *self.peers()]:
                if peer != self.rank:
                    self.handle.wait_signal(peer, _channel(0, FREE))
                for dst, part in zip(self.peer_flat[peer], parts, strict=True):
                    for row, src in zip(dst, part.reshape(self.rows, self.row_len), strict=True):
                        row[lo:hi].copy_(src)
                if peer != self.rank:
                    self.handle.put_signal(peer, _channel(0, READY))
        for part in parts:
            part.record_stream(self.stream)

    def wait(self, parts: Sequence[torch.Tensor]) -> torch.Tensor:
        if torch.is_grad_enabled() and any(part.requires_grad for part in parts):
            return _GatherWait.apply(self, *parts)
        return self._wait()

    def _wait(self) -> torch.Tensor:
        torch.cuda.current_stream().wait_stream(self.stream)
        for peer in self.peers():
            self.handle.wait_signal(peer, _channel(0, READY))
        return self.buffer

    def reduce_scatter(self, grad: torch.Tensor) -> list[torch.Tensor]:
        """Sum ``grad`` (shaped like the output) over ranks; return this rank's shard per part."""
        by_rank = grad.reshape(-1, self.rows, self.world_size, self.row_len).permute(2, 0, 1, 3)
        shard = grad.new_empty(by_rank.shape[1:])
        dist.reduce_scatter_tensor(shard, by_rank.contiguous(), group=self.group)
        return [part.view(self.part_shape) for part in shard]


class _GatherWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gather: CopyEngineAllGather, *parts: torch.Tensor) -> torch.Tensor:
        ctx.gather = gather
        # Alias, not the buffer object itself, so autograd metadata stays off the shared buffer.
        return gather._wait().detach()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, *ctx.gather.reduce_scatter(grad)


def single_node() -> bool:
    """True when every rank of the world group runs on this node (torchrun's ``LOCAL_WORLD_SIZE``).

    Multi-node NVLink domains (NVL72) also support copy-engine transfers, but there is no cheap
    portable query for them, so callers opt in explicitly there.
    """
    return int(os.environ.get("LOCAL_WORLD_SIZE", "-1")) == dist.get_world_size()
