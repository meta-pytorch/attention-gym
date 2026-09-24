"""Triton primitives shared by gather-attention kernel schedules."""

from collections.abc import Sequence
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import _document_ids, ptr_offset


def can_use_shared_kv_schedule(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
    sliding_window_size: int,
) -> bool:
    """Check the shared Blackwell forward/backward schedule's layout and shape limits."""
    _, heads, _, head_dim = query.shape
    return (
        torch.cuda.get_device_capability(query.device)[0] >= 10
        and query.dtype == torch.bfloat16
        and sparse_kv.stride(1) == 0
        and local_kv.stride(1) == 0
        and query.stride(-1) == 1
        and sparse_kv.stride(-1) == 1
        and local_kv.stride(-1) == 1
        and 16 <= heads <= 128
        and head_dim % 16 == 0
        and (head_dim <= 128 or head_dim == 512)
        and sliding_window_size <= 2048
    )


class TileConfig(NamedTuple):
    """Launch parameters of a fixed-config (not autotuned) kernel."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    # Shared memory in BLOCK_D-wide rows: the largest ``metadata.shared / (BLOCK_D *
    # element_size)`` measured for SM86 at D in {64, 128, 256} with Triton 3.8.
    shared_rows: int = 0


def select_tiles(
    candidates: Sequence[TileConfig],
    block_d: int,
    element_size: int,
    device: torch.device,
) -> TileConfig:
    """Return the first candidate whose estimated shared memory fits one block on ``device``.

    Candidates are ordered by preference; the last one is the fallback for tiny budgets.
    """
    budget = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    return next(
        (c for c in candidates if c.shared_rows * block_d * element_size <= budget),
        candidates[-1],
    )


def prune_wide_backward_configs(configs, _named_args, D, **_):
    """Bound full-width D=512 gradient tiles without changing smaller-head tuning."""
    if D != 512:
        return configs
    return [
        config
        for config in configs
        if config.num_warps == 4
        and config.num_stages == 1
        and all(size <= 16 for size in config.kwargs.values())
    ]


@triton.jit
def load_bhsd(
    tensor_ptr,
    strides: tl.constexpr,
    batch,
    head,
    positions,
    offsets_d,
    mask,
):
    """Load a tile from gather attention's BHSD tensors."""
    return tl.load(
        tensor_ptr
        + ptr_offset(
            (batch, head, positions[:, None], offsets_d[None, :]),
            strides,
        ),
        mask=mask,
        other=0.0,
    )


@triton.jit
def load_document_bounds(
    cu_seqlens_ptr,
    positions,
    num_documents,
    sequence_length: tl.constexpr,
    WIDE: tl.constexpr,
):
    """Find packed bounds, treating inactive capacity as one isolated document."""
    document = _document_ids(cu_seqlens_ptr, positions, num_documents, WIDE)
    offset = document.to(tl.int64) if WIDE else document
    # The right-sided search skips empty documents and returns N for tail/padded positions.
    start = tl.load(cu_seqlens_ptr + offset)
    end = tl.load(
        cu_seqlens_ptr + offset + 1,
        mask=document < num_documents,
        other=sequence_length,
    )
    return start, end


@triton.jit
def store_bhsd(
    tensor_ptr,
    value,
    strides: tl.constexpr,
    batch,
    head,
    positions,
    offsets_d,
    mask,
):
    """Store a tile to gather attention's BHSD tensors."""
    tl.store(
        tensor_ptr
        + ptr_offset(
            (batch, head, positions[:, None], offsets_d[None, :]),
            strides,
        ),
        value,
        mask=mask,
    )


@triton.jit
def causal_window_mask(
    query_positions,
    key_positions,
    query_mask,
    key_mask,
    window: tl.constexpr,
):
    """Mask valid gather-attention pairs to the causal local window."""
    return (
        query_mask[:, None]
        & key_mask[None, :]
        & (key_positions[None, :] <= query_positions[:, None])
        & (key_positions[None, :] >= query_positions[:, None] - window + 1)
    )


@triton.jit
def online_softmax_update(accumulator, running_max, running_sum, logits, values):
    """Merge one gather-attention tile into FP32 online-softmax state."""
    tile_max = tl.max(logits, axis=1)
    new_max = tl.maximum(running_max, tile_max)
    # An empty prefix with no sink has max=-inf. Keep that state for later tiles,
    # but avoid -inf - -inf while computing this tile's zero contribution.
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    alpha = tl.exp(running_max - safe_max)
    probabilities = tl.exp(logits - safe_max[:, None])
    accumulator *= alpha[:, None]
    accumulator += tl.dot(probabilities.to(values.dtype), values, input_precision="tf32x3")
    running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
    return accumulator, new_max, running_sum
