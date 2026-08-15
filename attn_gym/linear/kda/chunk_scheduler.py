"""Graph-safe logical chunk scheduling for packed variable-length KDA inputs."""

from __future__ import annotations

import itertools
from typing import NamedTuple

import torch
import triton
import triton.language as tl


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
            ``chunk_offsets[-1]`` on the device. Capacity-only CTAs return before
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

    def validate_chunk_size(self, chunk_size: int) -> None:
        """Reject consumers configured for a different logical chunk size."""
        if self.chunk_size != chunk_size:
            raise ValueError(
                f"metadata chunk size must match chunk_size={chunk_size}, got {self.chunk_size}"
            )


class ChunkWork(NamedTuple):
    """One sequence-local chunk decoded from a global logical work index."""

    global_chunk: int
    sequence: int
    local_chunk: int
    token_start: int
    valid_tokens: int


MAX_NUM_SEQUENCES = 4096


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


def chunk_work_oracle(cu_seqlens: list[int], chunk_size: int) -> list[ChunkWork]:
    """Decode all logical chunks on the CPU for tests and scheduler validation."""
    if len(cu_seqlens) < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if cu_seqlens[0] != 0:
        raise ValueError("cu_seqlens must start at zero")
    if any(begin > end for begin, end in itertools.pairwise(cu_seqlens)):
        raise ValueError("cu_seqlens must be monotonic")

    work: list[ChunkWork] = []
    for sequence, (begin, end) in enumerate(itertools.pairwise(cu_seqlens)):
        for token_start in range(begin, end, chunk_size):
            work.append(
                ChunkWork(
                    global_chunk=len(work),
                    sequence=sequence,
                    local_chunk=(token_start - begin) // chunk_size,
                    token_start=token_start,
                    valid_tokens=min(chunk_size, end - token_start),
                )
            )
    return work


@triton.jit(debug=True, do_not_specialize=["num_sequences"])
def _prepare_ragged_chunk_offsets_kernel(
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    tokens: tl.constexpr,
    chunk_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Validate boundaries and build the chunk-count prefix in one CTA."""
    sequence = tl.arange(0, BLOCK)
    sequence_mask = sequence < num_sequences
    begin = tl.load(cu_seqlens + sequence, mask=sequence_mask, other=0)
    end = tl.load(cu_seqlens + sequence + 1, mask=sequence_mask, other=0)
    valid = (begin >= 0) & (begin <= end) & (end <= tokens) & ((sequence != 0) | (begin == 0))
    tl.device_assert(valid, "invalid packed cu_seqlens", mask=sequence_mask)

    count = tl.where(sequence_mask & valid, (end - begin + chunk_size - 1) // chunk_size, 0)
    inclusive_offset = tl.cumsum(count, axis=0)
    tl.store(chunk_offsets, 0)
    tl.store(chunk_offsets + sequence + 1, inclusive_offset, mask=sequence_mask)


@triton.jit
def load_ragged_chunk_work(
    cu_seqlens,
    chunk_offsets,
    global_chunk,
    num_sequences,
    chunk_size: tl.constexpr,
):
    """Binary-search one known-active global chunk's sequence boundaries.

    Returns its sequence index, sequence-relative chunk index, packed token start,
    and valid token count. Callers must reject ``global_chunk >= chunk_offsets[-1]``
    before invoking this helper.
    """
    low = 0
    high = num_sequences + 1
    while low < high:
        middle = (low + high) // 2
        offset = tl.load(chunk_offsets + middle)
        if offset <= global_chunk:
            low = middle + 1
        else:
            high = middle

    sequence = low - 1
    sequence_offset = tl.load(chunk_offsets + sequence)
    local_chunk = global_chunk - sequence_offset
    begin = tl.load(cu_seqlens + sequence)
    end = tl.load(cu_seqlens + sequence + 1)
    token_start = begin + local_chunk * chunk_size
    valid_tokens = tl.minimum(chunk_size, end - token_start)
    return sequence, local_chunk, token_start, valid_tokens


@triton.jit(do_not_specialize=["num_sequences"])
def _decode_ragged_chunk_work_kernel(
    cu_seqlens,
    chunk_offsets,
    work,
    num_sequences,
    chunk_size: tl.constexpr,
):
    global_chunk = tl.program_id(0)
    active_chunks = tl.load(chunk_offsets + num_sequences)
    output = work + global_chunk * 5

    if global_chunk < active_chunks:
        sequence, local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            global_chunk,
            num_sequences,
            chunk_size,
        )

        tl.store(output, global_chunk)
        tl.store(output + 1, sequence)
        tl.store(output + 2, local_chunk)
        tl.store(output + 3, token_start)
        tl.store(output + 4, valid_tokens)
    else:
        for field in tl.static_range(5):
            tl.store(output + field, -1)


def prepare_ragged_chunk_metadata(
    cu_seqlens: torch.Tensor,
    tokens: int,
    chunk_size: int,
) -> RaggedChunkMetadata:
    """Build fixed-shape, graph-replayable packed chunk offsets on the GPU."""
    if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
        raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous int32")
    if not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be a CUDA tensor")

    num_sequences = cu_seqlens.shape[0] - 1
    if num_sequences > MAX_NUM_SEQUENCES:
        raise ValueError(
            f"the ragged chunk scheduler supports at most {MAX_NUM_SEQUENCES} sequences, "
            f"got {num_sequences}"
        )
    capacity = chunk_capacity(tokens, num_sequences, chunk_size)
    chunk_offsets = torch.empty_like(cu_seqlens)
    block = triton.next_power_of_2(num_sequences)
    _prepare_ragged_chunk_offsets_kernel[(1,)](
        cu_seqlens,
        chunk_offsets,
        num_sequences=num_sequences,
        tokens=tokens,
        chunk_size=chunk_size,
        BLOCK=block,
        num_warps=min(8, max(1, block // 32)),
    )
    return RaggedChunkMetadata(cu_seqlens, chunk_offsets, capacity, chunk_size)


def decode_ragged_chunk_work(metadata: RaggedChunkMetadata) -> torch.Tensor:
    """Materialize scheduler decisions for diagnostics and mapping tests."""
    work = torch.empty(
        (metadata.capacity, 5),
        dtype=torch.int32,
        device=metadata.cu_seqlens.device,
    )
    if metadata.capacity:
        _decode_ragged_chunk_work_kernel[(metadata.capacity,)](
            metadata.cu_seqlens,
            metadata.chunk_offsets,
            work,
            num_sequences=metadata.cu_seqlens.shape[0] - 1,
            chunk_size=metadata.chunk_size,
            num_warps=1,
        )
    return work


__all__ = [
    "MAX_NUM_SEQUENCES",
    "ChunkWork",
    "RaggedChunkMetadata",
    "chunk_capacity",
    "chunk_work_oracle",
    "decode_ragged_chunk_work",
    "load_ragged_chunk_work",
    "prepare_ragged_chunk_metadata",
]
