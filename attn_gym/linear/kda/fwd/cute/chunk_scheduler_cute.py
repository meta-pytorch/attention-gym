"""CuTeDSL work decoding for the packed KDA chunk scheduler."""

from __future__ import annotations

import cutlass
from cutlass import Int32, cute

from attn_gym._backends.cute.device import upper_bound
from attn_gym._backends.cute.ragged import load_ragged_token_count


@cute.jit
def load_ragged_chunk_count(chunk_offsets: cute.Tensor):
    """Load the terminal prefix-sum entry containing the active chunk count."""
    num_sequences = Int32(cute.size(chunk_offsets)) - 1
    return Int32(chunk_offsets[num_sequences])


@cute.jit
def load_ragged_sequence_extent(cu_seqlens: cute.Tensor):
    """Return one past the last sequence slot that may contain tokens."""
    num_sequences = Int32(cute.size(cu_seqlens)) - 1
    active_tokens = load_ragged_token_count(cu_seqlens)
    sequence_extent = num_sequences
    if Int32(cu_seqlens[num_sequences - 1]) >= active_tokens:
        sequence_extent = upper_bound(
            cu_seqlens,
            active_tokens - 1,
            Int32(0),
            num_sequences,
        )
    return sequence_extent


@cute.jit
def load_ragged_chunk_work(
    cu_seqlens: cute.Tensor,
    chunk_offsets: cute.Tensor,
    global_chunk: Int32,
    chunk_size: Int32,
):
    """Binary-search one known-active global chunk's sequence boundaries.

    Returns its sequence index, sequence-relative chunk index, packed token start,
    and valid token count. The invoking kernel owns its launch configuration and
    must reject inactive chunks before calling this warp-independent helper.
    """
    num_sequences = Int32(cute.size(chunk_offsets)) - 1
    sequence = (
        upper_bound(
            chunk_offsets,
            global_chunk,
            Int32(0),
            num_sequences + 1,
        )
        - 1
    )
    sequence_offset = Int32(chunk_offsets[sequence])
    local_chunk = global_chunk - sequence_offset
    begin = Int32(cu_seqlens[sequence])
    end = Int32(cu_seqlens[sequence + 1])
    token_start = begin + local_chunk * chunk_size
    valid_tokens = cutlass.min(chunk_size, end - token_start)
    return sequence, local_chunk, token_start, valid_tokens


__all__ = [
    "load_ragged_chunk_count",
    "load_ragged_chunk_work",
    "load_ragged_sequence_extent",
]
