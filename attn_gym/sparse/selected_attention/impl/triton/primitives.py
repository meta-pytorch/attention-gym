"""Triton primitives shared by selected-attention kernel schedules."""

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset


def can_use_shared_kv_schedule(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
    sliding_window_size: int,
) -> bool:
    """Check Blackwell shared-KV constraints common to forward and backward.

    Callers enforce the head-dimension limits supported by their kernel.
    """
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
        and sliding_window_size <= 2048
    )


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
    """Load a tile from selected attention's BHSD tensors."""
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
def load_bs(
    tensor_ptr,
    strides: tl.constexpr,
    batch,
    positions,
    mask,
    other: tl.constexpr,
):
    """Load positions from a selected-attention batch-sequence tensor."""
    return tl.load(
        tensor_ptr + ptr_offset((batch, positions), strides),
        mask=mask,
        other=other,
    )


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
    """Store a tile to selected attention's BHSD tensors."""
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
    """Mask valid selected-attention pairs to the causal local window."""
    return (
        query_mask[:, None]
        & key_mask[None, :]
        & (key_positions[None, :] <= query_positions[:, None])
        & (key_positions[None, :] >= query_positions[:, None] - window + 1)
    )


@triton.jit
def online_softmax_update(accumulator, running_max, running_sum, logits, values):
    """Merge one selected-attention tile into FP32 online-softmax state."""
    tile_max = tl.max(logits, axis=1)
    new_max = tl.maximum(running_max, tile_max)
    alpha = tl.exp(running_max - new_max)
    probabilities = tl.exp(logits - new_max[:, None])
    accumulator *= alpha[:, None]
    accumulator += tl.dot(probabilities.to(values.dtype), values, input_precision="tf32x3")
    running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
    return accumulator, new_max, running_sum
