# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run a fixed block exchange plan with copy-engine pushes into symmetric memory.

Each copy moves one run of consecutive blocks, so every transfer is a contiguous ``copy_``
and runs on a copy engine. Peers synchronize through the symmetric-memory signal pad, one
flag per (peer, signal): an owner waits for ``FREE`` from a consumer before writing into its
landing zone, and sends ``READY`` after the writes. A consumer sends ``FREE`` back after it
has copied the blocks out. Every rank in the group must share one NVLink domain.

The plan is host data, so the copies are fixed when the exchange is created: a CUDA graph
replays the same transfers with the current K/V values.
"""

# T: local tokens
# A: assembled tokens
# H: heads
# K: key head dimensions
# V: value head dimensions
# R: received tokens

from collections.abc import Sequence

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from .plan import BlockExchangePlan

READY, FREE = 0, 1


def _build_copy_runs(pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Merge ``(source_block, destination_block)`` pairs into ``(source, destination, count)``
    runs of consecutive blocks, keeping the pair order."""
    runs: list[tuple[int, int, int]] = []
    for source, destination in pairs:
        if (
            runs
            and runs[-1][0] + runs[-1][2] == source
            and runs[-1][1] + runs[-1][2] == destination
        ):
            runs[-1] = (runs[-1][0], runs[-1][1], runs[-1][2] + 1)
        else:
            runs.append((source, destination, 1))
    return runs


def _execute_copy_runs(
    destination: torch.Tensor, source: torch.Tensor, runs: Sequence[tuple[int, int, int]]
) -> None:
    for source_block, destination_block, count in runs:
        destination[destination_block : destination_block + count].copy_(
            source[source_block : source_block + count]
        )


class _LandingZone:
    """One symmetric buffer of ``num_blocks`` K blocks and V blocks that peers write into."""

    def __init__(
        self,
        group: dist.ProcessGroup,
        num_blocks: int,
        k_block_numel: int,
        v_block_numel: int,
        dtype: torch.dtype,
        device: torch.device,
        peers: Sequence[int],
    ):
        self.num_blocks, self.k_block_numel = num_blocks, k_block_numel
        self.v_block_numel = v_block_numel
        numel = num_blocks * (k_block_numel + v_block_numel)
        self.buffer = symm_mem.empty(numel, dtype=dtype, device=device)
        self.handle = symm_mem.rendezvous(self.buffer, group.group_name)
        self.k, self.v = self._split(self.buffer)
        self.peer_k, self.peer_v = {}, {}
        for peer in peers:
            peer_buffer = self.handle.get_buffer(peer, (numel,), dtype)
            self.peer_k[peer], self.peer_v[peer] = self._split(peer_buffer)

    def _split(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k_numel = self.num_blocks * self.k_block_numel
        return (
            flat[:k_numel].view(self.num_blocks, self.k_block_numel),
            flat[k_numel:].view(self.num_blocks, self.v_block_numel),
        )


class CopyEngineBlockExchange:
    """Exchange and assemble K/V blocks for a fixed `BlockExchangePlan` on copy engines.

    Owners push the planned blocks into consumers' landing zones in symmetric memory, and each
    rank copies its local and received blocks into its assembled K/V. Local block ``b`` lands at
    assembled block ``local_destination_blocks[b]`` and receive slot ``s`` at
    ``remote_destination_blocks[s]``. Assembled blocks that no local or received block maps to
    are left uninitialized; the attention mask must not read them. Backward sends each received
    block's gradient to its owner, which adds the contributions to its own gradient in
    send-slot order. When any block has more than one remote contributor, the sums accumulate
    in float32.

    Landing zones hold the largest per-rank receive and send counts, because symmetric memory
    needs the same size on every rank. One instance can serve every layer with the same plan
    and shapes, one call at a time; every rank must make the same calls in the same order.
    Create it before any CUDA graph capture, and keep it alive while graphs that use it are
    captured or replayed: freeing symmetric memory during capture invalidates the capture.
    Signal waits give up after the group's NCCL timeout: a timed-out wait reports the rank and
    signal and traps, so a rank that falls out of step fails instead of hanging.

    Args:
        group (dist.ProcessGroup): Process group of the exchange. Every rank must share one
            NVLink domain.
        plan (BlockExchangePlan): This rank's plan. Its rank and world size must match
            ``group``.
        k_token_shape (Sequence[int]): Shape of one K token, for example
            ``(heads, k_head_dim)``.
        v_token_shape (Sequence[int]): Shape of one V token.
        dtype (torch.dtype): K/V dtype.
        device (torch.device): This rank's CUDA device.

    Example:
        Without activation checkpointing, exchange and assemble in one op::

            exchange = CopyEngineBlockExchange(
                group, plan, k_token_shape=(heads, k_dim), v_token_shape=(heads, v_dim),
                dtype=torch.bfloat16, device=k.device,
            )
            assembled_k, assembled_v = exchange.exchange_and_assemble(k, v)

        Under activation checkpointing, save only the received blocks::

            remote_k, remote_v = exchange.exchange(k, v)
            assembled_k, assembled_v = checkpoint(
                exchange.assemble, k, v, remote_k, remote_v, use_reentrant=False
            )
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        plan: BlockExchangePlan,
        *,
        k_token_shape: Sequence[int],
        v_token_shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device,
    ):
        if (group.rank(), group.size()) != (plan.rank, plan.world_size):
            raise ValueError(
                f"Plan is for rank {plan.rank} of {plan.world_size}, but the group has rank "
                f"{group.rank()} of {group.size()}."
            )
        self.plan, self.dtype = plan, dtype
        timeout = group._get_backend(device).options._timeout
        self.timeout_ms = int(timeout.total_seconds() * 1000)
        self.block_tokens = plan.block_tokens
        self.k_token_shape, self.v_token_shape = tuple(k_token_shape), tuple(v_token_shape)
        k_block_numel = plan.block_tokens * torch.Size(k_token_shape).numel()
        v_block_numel = plan.block_tokens * torch.Size(v_token_shape).numel()

        send_blocks = plan.send_source_blocks
        send_offsets, send_counts = plan.send_offsets, plan.send_counts
        recv_offsets, recv_counts = plan.recv_offsets, plan.recv_counts
        peer_recv_offsets, peer_send_offsets = plan.peer_recv_offsets, plan.peer_send_offsets
        local_destinations = plan.local_destination_blocks
        remote_destinations = plan.remote_destination_blocks
        self.num_recv = len(remote_destinations)
        self.consumers = [p for p, count in enumerate(send_counts) if count]
        self.owners = [p for p, count in enumerate(recv_counts) if count]

        # Forward: local blocks to the assembled K/V; send slots to each consumer's receive
        # slots; receive slots to the assembled K/V.
        self.local_copy_runs = _build_copy_runs(list(enumerate(local_destinations)))
        self.push_copy_runs = {
            p: _build_copy_runs(
                [
                    (send_blocks[send_offsets[p] + i], peer_recv_offsets[p] + i)
                    for i in range(send_counts[p])
                ]
            )
            for p in self.consumers
        }
        self.recv_copy_runs = _build_copy_runs(list(enumerate(remote_destinations)))
        # Backward: assembled gradients of received blocks back to each owner's send slots;
        # owned gradients from the assembled gradient; contributions from send slots in slot
        # order.
        self.grad_push_copy_runs = {
            p: _build_copy_runs(
                [
                    (remote_destinations[recv_offsets[p] + i], peer_send_offsets[p] + i)
                    for i in range(recv_counts[p])
                ]
            )
            for p in self.owners
        }
        self.owned_grad_copy_runs = _build_copy_runs(
            [(d, b) for b, d in enumerate(local_destinations)]
        )
        self.grad_accumulation_runs = _build_copy_runs(
            [(slot, b) for slot, b in enumerate(send_blocks)]
        )
        self.accumulate_float32 = len(set(send_blocks)) < len(send_blocks)

        self.forward_zone = _LandingZone(
            group,
            plan.max_recv_blocks,
            k_block_numel,
            v_block_numel,
            dtype,
            device,
            self.consumers,
        )
        self.backward_zone = _LandingZone(
            group,
            plan.max_send_blocks,
            k_block_numel,
            v_block_numel,
            dtype,
            device,
            self.owners,
        )
        self.stream = torch.cuda.Stream(device)
        # Both landing zones start empty: tell every writer it may write.
        for owner in self.owners:
            self.forward_zone.handle.put_signal(owner, FREE, self.timeout_ms)
        for consumer in self.consumers:
            self.backward_zone.handle.put_signal(consumer, FREE, self.timeout_ms)

    def exchange_and_assemble(
        self, k_THK: torch.Tensor, v_THV: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange the planned blocks and return the assembled K/V.

        Use it when attention saves its inputs anyway, for example without activation
        checkpointing. Backward sends each received block's gradient to its owner.

        Args:
            k_THK (torch.Tensor): Local K, ``[blocks_per_rank * block_tokens, *k_token_shape]``.
            v_THV (torch.Tensor): Local V, ``[blocks_per_rank * block_tokens, *v_token_shape]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Assembled K and V,
            ``[num_assembled_blocks * block_tokens, ...]``.
        """
        self._check(k_THK, v_THV)
        return _ExchangeAndAssemble.apply(self, k_THK, v_THV)

    def exchange(
        self, k_THK: torch.Tensor, v_THV: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange the planned blocks and return only the received ones.

        Use it with `assemble` under activation checkpointing: call `exchange` outside the
        checkpoint so its results are saved, and call `assemble` inside it. Backward then
        recomputes only the local assembly, not the communication. `exchange` itself has no
        autograd; `assemble`'s backward sends the gradients to their owners.

        Args:
            k_THK (torch.Tensor): Local K, ``[blocks_per_rank * block_tokens, *k_token_shape]``.
            v_THV (torch.Tensor): Local V, ``[blocks_per_rank * block_tokens, *v_token_shape]``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Received K and V in receive-slot order,
            ``[sum(recv_counts) * block_tokens, ...]``.
        """
        self._check(k_THK, v_THV)
        recv_tokens = self.num_recv * self.block_tokens
        zone = self.forward_zone
        with torch.no_grad():
            self._push(
                zone,
                self._blocks(k_THK, self.k_token_shape),
                self._blocks(v_THV, self.v_token_shape),
                self.push_copy_runs,
            )
            for owner in self.owners:
                zone.handle.wait_signal(owner, READY, self.timeout_ms)
            remote_k_RHK = zone.k[: self.num_recv].clone().view(recv_tokens, *self.k_token_shape)
            remote_v_RHV = zone.v[: self.num_recv].clone().view(recv_tokens, *self.v_token_shape)
            for owner in self.owners:
                zone.handle.put_signal(owner, FREE, self.timeout_ms)
            torch.cuda.current_stream().wait_stream(self.stream)
        return remote_k_RHK, remote_v_RHV

    def assemble(
        self,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        remote_k_RHK: torch.Tensor,
        remote_v_RHV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble local K/V and the results of `exchange` into the assembled K/V.

        Forward is local. Backward is the same collective as `exchange_and_assemble`: it
        returns the full gradient of K/V, and the received blocks get none.

        Args:
            k_THK (torch.Tensor): Local K, as passed to `exchange`.
            v_THV (torch.Tensor): Local V, as passed to `exchange`.
            remote_k_RHK (torch.Tensor): Received K returned by `exchange`.
            remote_v_RHV (torch.Tensor): Received V returned by `exchange`.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Assembled K and V,
            ``[num_assembled_blocks * block_tokens, ...]``.
        """
        self._check(k_THK, v_THV)
        return _Assemble.apply(self, k_THK, v_THV, remote_k_RHK, remote_v_RHV)

    def _check(self, k_THK: torch.Tensor, v_THV: torch.Tensor) -> None:
        local_tokens = self.plan.blocks_per_rank * self.block_tokens
        if k_THK.shape != (local_tokens, *self.k_token_shape) or v_THV.shape != (
            local_tokens,
            *self.v_token_shape,
        ):
            raise ValueError(
                f"Expected K {(local_tokens, *self.k_token_shape)} and V "
                f"{(local_tokens, *self.v_token_shape)}, got {tuple(k_THK.shape)} and "
                f"{tuple(v_THV.shape)}."
            )
        if k_THK.dtype != self.dtype or v_THV.dtype != self.dtype:
            raise ValueError(f"K and V must be {self.dtype}.")
        if not (k_THK.is_contiguous() and v_THV.is_contiguous()):
            raise ValueError("K and V must be contiguous.")

    def _blocks(self, tensor: torch.Tensor, token_shape: tuple[int, ...]) -> torch.Tensor:
        return tensor.view(-1, self.block_tokens * torch.Size(token_shape).numel())

    def _push(
        self,
        zone: _LandingZone,
        k_blocks: torch.Tensor,
        v_blocks: torch.Tensor,
        copy_runs: dict[int, list[tuple[int, int, int]]],
    ) -> None:
        """Write each peer's copy runs into its landing zone on the side stream."""
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for peer, peer_copy_runs in copy_runs.items():
                zone.handle.wait_signal(peer, FREE, self.timeout_ms)
                _execute_copy_runs(zone.peer_k[peer], k_blocks, peer_copy_runs)
                _execute_copy_runs(zone.peer_v[peer], v_blocks, peer_copy_runs)
                zone.handle.put_signal(peer, READY, self.timeout_ms)

    def _assemble_fwd(
        self,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        remote_k_RHK: torch.Tensor,
        remote_v_RHV: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assembled_tokens = self.plan.num_assembled_blocks * self.block_tokens
        assembled_k_AHK = k_THK.new_empty((assembled_tokens, *self.k_token_shape))
        assembled_v_AHV = v_THV.new_empty((assembled_tokens, *self.v_token_shape))
        out_k = self._blocks(assembled_k_AHK, self.k_token_shape)
        out_v = self._blocks(assembled_v_AHV, self.v_token_shape)
        _execute_copy_runs(out_k, self._blocks(k_THK, self.k_token_shape), self.local_copy_runs)
        _execute_copy_runs(out_v, self._blocks(v_THV, self.v_token_shape), self.local_copy_runs)
        _execute_copy_runs(
            out_k, self._blocks(remote_k_RHK, self.k_token_shape), self.recv_copy_runs
        )
        _execute_copy_runs(
            out_v, self._blocks(remote_v_RHV, self.v_token_shape), self.recv_copy_runs
        )
        return assembled_k_AHK, assembled_v_AHV

    def _exchange_and_assemble_fwd(
        self, k_THK: torch.Tensor, v_THV: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assembled_tokens = self.plan.num_assembled_blocks * self.block_tokens
        assembled_k_AHK = k_THK.new_empty((assembled_tokens, *self.k_token_shape))
        assembled_v_AHV = v_THV.new_empty((assembled_tokens, *self.v_token_shape))
        k_blocks = self._blocks(k_THK, self.k_token_shape)
        v_blocks = self._blocks(v_THV, self.v_token_shape)
        out_k = self._blocks(assembled_k_AHK, self.k_token_shape)
        out_v = self._blocks(assembled_v_AHV, self.v_token_shape)
        zone = self.forward_zone
        self._push(zone, k_blocks, v_blocks, self.push_copy_runs)
        # The local copies overlap the pushes.
        _execute_copy_runs(out_k, k_blocks, self.local_copy_runs)
        _execute_copy_runs(out_v, v_blocks, self.local_copy_runs)
        for owner in self.owners:
            zone.handle.wait_signal(owner, READY, self.timeout_ms)
        _execute_copy_runs(out_k, zone.k, self.recv_copy_runs)
        _execute_copy_runs(out_v, zone.v, self.recv_copy_runs)
        for owner in self.owners:
            zone.handle.put_signal(owner, FREE, self.timeout_ms)
        torch.cuda.current_stream().wait_stream(self.stream)
        return assembled_k_AHK, assembled_v_AHV

    def _exchange_and_assemble_bwd(
        self, grad_k_AHK: torch.Tensor, grad_v_AHV: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_tokens = self.plan.blocks_per_rank * self.block_tokens
        grad_k_blocks = self._blocks(grad_k_AHK.contiguous(), self.k_token_shape)
        grad_v_blocks = self._blocks(grad_v_AHV.contiguous(), self.v_token_shape)
        zone = self.backward_zone
        self._push(zone, grad_k_blocks, grad_v_blocks, self.grad_push_copy_runs)
        grad_k_THK = grad_k_AHK.new_empty((local_tokens, *self.k_token_shape))
        grad_v_THV = grad_v_AHV.new_empty((local_tokens, *self.v_token_shape))
        local_k = self._blocks(grad_k_THK, self.k_token_shape)
        local_v = self._blocks(grad_v_THV, self.v_token_shape)
        _execute_copy_runs(local_k, grad_k_blocks, self.owned_grad_copy_runs)
        _execute_copy_runs(local_v, grad_v_blocks, self.owned_grad_copy_runs)
        for consumer in self.consumers:
            zone.handle.wait_signal(consumer, READY, self.timeout_ms)
        if self.accumulate_float32:
            local_k, local_v = local_k.float(), local_v.float()
        for local, landed in ((local_k, zone.k), (local_v, zone.v)):
            for slot, block, count in self.grad_accumulation_runs:
                local[block : block + count] += landed[slot : slot + count]
        for consumer in self.consumers:
            zone.handle.put_signal(consumer, FREE, self.timeout_ms)
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.accumulate_float32:
            grad_k_THK.view_as(local_k).copy_(local_k)
            grad_v_THV.view_as(local_v).copy_(local_v)
        return grad_k_THK, grad_v_THV


class _ExchangeAndAssemble(torch.autograd.Function):
    @staticmethod
    def forward(ctx, exchange: CopyEngineBlockExchange, k_THK, v_THV):
        ctx.exchange = exchange
        return exchange._exchange_and_assemble_fwd(k_THK, v_THV)

    @staticmethod
    def backward(ctx, grad_k_AHK, grad_v_AHV):
        return None, *ctx.exchange._exchange_and_assemble_bwd(grad_k_AHK, grad_v_AHV)


class _Assemble(torch.autograd.Function):
    @staticmethod
    def forward(ctx, exchange: CopyEngineBlockExchange, k_THK, v_THV, remote_k_RHK, remote_v_RHV):
        ctx.exchange = exchange
        return exchange._assemble_fwd(k_THK, v_THV, remote_k_RHK, remote_v_RHV)

    @staticmethod
    def backward(ctx, grad_k_AHK, grad_v_AHV):
        return None, *ctx.exchange._exchange_and_assemble_bwd(grad_k_AHK, grad_v_AHV), None, None
