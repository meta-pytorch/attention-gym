"""Graph-safe logical chunk scheduling for packed variable-length KDA inputs."""

from __future__ import annotations

import itertools
from typing import NamedTuple

import torch
import triton
import triton.language as tl

# The op-wrapped metadata builder is re-exported so every caller routes the offsets
# launch through the registered scheduler op instead of tracing it directly.
from attn_gym.linear.kda.chunk_schedule import (
    RaggedChunkMetadata,
    chunk_capacity,
    prepare_ragged_chunk_metadata,
)


class ChunkWork(NamedTuple):
    """One sequence-local chunk decoded from a global logical work index."""

    global_chunk: int
    sequence: int
    local_chunk: int
    token_start: int
    valid_tokens: int


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


def _prepare_ragged_chunk_offsets(
    cu_seqlens: torch.Tensor,
    tokens: int,
    chunk_size: int,
) -> torch.Tensor:
    """Build packed chunk offsets for the torch-only registration wrapper."""
    num_sequences = cu_seqlens.shape[0] - 1
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
    return chunk_offsets


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
    "ChunkWork",
    "RaggedChunkMetadata",
    "chunk_capacity",
    "chunk_work_oracle",
    "decode_ragged_chunk_work",
    "load_ragged_chunk_work",
    "prepare_ragged_chunk_metadata",
]
