"""Torch-only packed chunk metadata used by the public fused wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import torch

from attn_gym.linear.kda.ops import prepare_chunk_offsets_op


class ScheduleRequest(Enum):
    """Internal scheduling policy before eligibility and geometry are known."""

    AUTO = "auto"
    STATIC = "static"
    PERSISTENT = "persistent"


class ScheduleKind(Enum):
    """Concrete internal work-distribution strategy for one launch."""

    STATIC = "static"
    PERSISTENT = "persistent"


def validate_schedule_request(request: ScheduleRequest) -> None:
    """Reject legacy booleans and other non-enum internal schedule requests."""
    if not isinstance(request, ScheduleRequest):
        raise TypeError(f"request must be a ScheduleRequest, got {request!r}")


@dataclass(frozen=True)
class ResolvedSchedule:
    """Resolved execution kind and the geometry used to select it.

    ``workers`` is the bounded persistent-grid candidate; STATIC kernels use
    their natural capacity grid. ``capacity_tasks`` is the total number of
    possible logical work slots used by the automatic policy.
    """

    kind: ScheduleKind
    workers: int
    capacity_tasks: int


class RaggedChunkMetadata(NamedTuple):
    """Graph-safe routing for sequence-relative chunks in a packed token buffer.

    Attributes:
        cu_seqlens: Device tensor of shape ``[N + 1]`` containing packed token
            boundaries. Sequence ``i`` occupies
            ``[cu_seqlens[i], cu_seqlens[i + 1])``; repeated boundaries represent
            empty sequences.
        chunk_offsets: Device tensor of shape ``[N + 1]`` containing the exclusive
            prefix sum of per-sequence chunk counts. It is a sequence-boundary
            tensor, not a per-chunk map::

                count[i] = ceil_div(
                    cu_seqlens[i + 1] - cu_seqlens[i], chunk_size
                )
                chunk_offsets[i + 1] = chunk_offsets[i] + count[i]

            Thus ``chunk_offsets[-1]`` is the runtime active chunk count and
            sequence ``i`` owns global chunks in
            ``[chunk_offsets[i], chunk_offsets[i + 1])``. Given a
            ``global_chunk``, kernels find its sequence with
            ``upper_bound(chunk_offsets, global_chunk) - 1``. Upper-bound search
            handles the repeated offsets produced by empty sequences.
        capacity: Static host-side upper bound on the active chunk count for every
            boundary distribution with the same token and sequence tensor shapes.
            CUDA Graph capture fixes launch and allocation shapes, so kernels use
            this capacity while reading the actual active count from
            ``chunk_offsets[-1]`` on the device. Capacity-only CTAs finish without
            accessing token data.
        chunk_size: Logical chunk size used to construct ``chunk_offsets``. Keeping
            it with the offsets prevents consumers from decoding them under a
            different chunking scheme.

    On CUDA Graph replay, values in ``cu_seqlens`` may change from aligned to ragged
    without host synchronization or recapture, provided the physical token capacity
    and sequence tensor shapes remain fixed. The terminal offset is the dynamic active
    token count and may be smaller than the physical capacity.
    """

    cu_seqlens: torch.Tensor
    chunk_offsets: torch.Tensor
    capacity: int
    chunk_size: int

    @classmethod
    def from_offsets(
        cls,
        cu_seqlens: torch.Tensor,
        chunk_offsets: torch.Tensor,
        tokens: int,
        chunk_size: int,
    ) -> RaggedChunkMetadata:
        """Restore metadata from existing offsets without reading or regenerating device values.

        ``tokens`` is the physical token capacity, not the runtime active endpoint. Callers
        retain ownership of optional-input validation at their respective boundaries.
        """
        return cls(
            cu_seqlens,
            chunk_offsets,
            chunk_capacity(tokens, cu_seqlens.shape[0] - 1, chunk_size),
            chunk_size,
        )

    def validate_chunk_size(self, chunk_size: int) -> None:
        """Reject consumers configured for a different logical chunk size."""
        if self.chunk_size != chunk_size:
            raise ValueError(
                f"metadata chunk size must match chunk_size={chunk_size}, got {self.chunk_size}"
            )


def chunk_capacity(tokens: int, num_sequences: int, chunk_size: int) -> int:
    """Return the exact shape-derived launch bound over all valid boundaries.

    At most ``min(tokens, num_sequences)`` sequences can be nonempty. Giving each
    such sequence one token creates one chunk per sequence; concentrating all
    remaining tokens in one sequence maximizes the number of additional chunks.
    """
    if tokens < 0:
        raise ValueError(f"tokens must be nonnegative, got {tokens}")
    if num_sequences < 1:
        raise ValueError(f"num_sequences must be positive, got {num_sequences}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    nonempty = min(tokens, num_sequences)
    if nonempty == 0:
        return 0
    return nonempty + (tokens - nonempty) // chunk_size


def prepare_ragged_chunk_metadata(
    cu_seqlens: torch.Tensor,
    tokens: int,
    chunk_size: int,
) -> RaggedChunkMetadata:
    """Build graph-safe packed chunk metadata through the registered scheduler op."""
    if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
        raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous int32")
    if not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be a CUDA tensor")

    num_sequences = cu_seqlens.shape[0] - 1
    capacity = chunk_capacity(tokens, num_sequences, chunk_size)
    chunk_offsets = prepare_chunk_offsets_op(cu_seqlens, tokens, chunk_size)
    return RaggedChunkMetadata(cu_seqlens, chunk_offsets, capacity, chunk_size)


__all__ = [
    "RaggedChunkMetadata",
    "chunk_capacity",
    "prepare_ragged_chunk_metadata",
]
