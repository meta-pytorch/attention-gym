# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable Triton KDA K4 assembly from BC16 block factors.

The kernel owns only the BT64 K4 stage: one Triton program loads the six FP32
strictly-lower blocks and four pre-inverted diagonal blocks for a ``(chunk,
head)`` pair, applies the fixed 4x4 block-inverse recurrence, and writes the
causal 64x64 inverse in the KDA output dtype. Dense and packed layouts share the
same block ABI; packed work is decoded entirely from device metadata.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.triton.utils import requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)

_CHUNK_SIZE = 64
_SUBCHUNK_SIZE = 16
_OFFDIAGONAL_BLOCKS = 6
_SUPPORTED_OUTPUT_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def load_diagonal_block(
    Akkd, token_start, head, block, valid_rows, H: tl.constexpr, BC: tl.constexpr
):
    """Load one pre-inverted BC16 diagonal block and enforce its causal shape."""
    row = tl.arange(0, BC)
    column = tl.arange(0, BC)
    token = token_start + block * BC + row[:, None]
    offset = (token * H + head) * BC + column[None, :]
    value = tl.load(Akkd + offset, mask=row[:, None] < valid_rows, other=0.0)
    return tl.where(column[None, :] <= row[:, None], value, 0.0)


@triton.jit
def load_offdiagonal_block(
    AkkOD,
    chunk,
    head,
    block,
    valid_rows,
    valid_columns,
    H: tl.constexpr,
    BC: tl.constexpr,
    OFFDIAGONAL_BLOCKS: tl.constexpr,
):
    """Load one K3-compatible FP32 off-diagonal block with tail masking."""
    row = tl.arange(0, BC)
    column = tl.arange(0, BC)
    block_row = chunk * OFFDIAGONAL_BLOCKS + block
    offset = (block_row * H + head) * (BC * BC) + row[:, None] * BC + column[None, :]
    mask = (row[:, None] < valid_rows) & (column[None, :] < valid_columns)
    return tl.load(AkkOD + offset, mask=mask, other=0.0)


@triton.jit
def narrow_dot(left, right, output_type: tl.constexpr):
    """Run one KDA block product after narrowing both operands to the output dtype."""
    return tl.dot(left.to(output_type), right.to(output_type))


@triton.jit
def store_block(
    output,
    value,
    token_start,
    head,
    row_block,
    column_block,
    valid_rows,
    valid_columns,
    H: tl.constexpr,
    TRIANGULAR: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
):
    """Store one output BC16 block, explicitly zeroing masked columns and upper entries."""
    row = tl.arange(0, BC)
    column = tl.arange(0, BC)
    token = token_start + row_block * BC + row[:, None]
    output_column = column_block * BC + column[None, :]
    offset = (token * H + head) * BT + output_column
    value = tl.where(column[None, :] < valid_columns, value, 0.0)
    if TRIANGULAR:
        value = tl.where(column[None, :] <= row[:, None], value, 0.0)
    tl.store(
        output + offset,
        value.to(output.dtype.element_ty),
        mask=row[:, None] < valid_rows,
    )


@triton.heuristics(
    {
        "IS_RAGGED": lambda args: args["cu_seqlens"] is not None,
        "USE_INT64_OFFSETS": lambda args: requires_int64_offsets(
            args["AkkOD"],
            args["Akkd"],
            args["output"],
            args["cu_seqlens"],
            args["chunk_offsets"],
        ),
    }
)
@triton.jit(do_not_specialize=["T", "num_sequences"])
def chunk_kda_fwd_k4b_triton_kernel(
    AkkOD,
    Akkd,
    output,
    cu_seqlens,
    chunk_offsets,
    T,
    num_sequences,
    H: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    OFFDIAGONAL_BLOCKS: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Assemble one KDA BT64 inverse for each statically scheduled chunk and head."""
    chunk = tl.program_id(0)
    head = tl.program_id(1)
    if IS_RAGGED:
        if chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        _, _, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            chunk,
            num_sequences,
            BT,
        )
        if USE_INT64_OFFSETS:
            chunk = chunk.to(tl.int64)
            head = head.to(tl.int64)
            token_start = token_start.to(tl.int64)
    else:
        if USE_INT64_OFFSETS:
            chunk = chunk.to(tl.int64)
            head = head.to(tl.int64)
            T = T.to(tl.int64)
        token_start = chunk * BT
        valid_tokens = T - token_start

    output_type: tl.constexpr = output.dtype.element_ty
    valid0 = tl.maximum(0, tl.minimum(BC, valid_tokens))
    valid1 = tl.maximum(0, tl.minimum(BC, valid_tokens - BC))
    valid2 = tl.maximum(0, tl.minimum(BC, valid_tokens - 2 * BC))
    valid3 = tl.maximum(0, tl.minimum(BC, valid_tokens - 3 * BC))

    inverse00 = load_diagonal_block(Akkd, token_start, head, 0, valid0, H, BC)
    inverse11 = load_diagonal_block(Akkd, token_start, head, 1, valid1, H, BC)
    inverse22 = load_diagonal_block(Akkd, token_start, head, 2, valid2, H, BC)
    inverse33 = load_diagonal_block(Akkd, token_start, head, 3, valid3, H, BC)

    block10 = load_offdiagonal_block(
        AkkOD, chunk, head, 0, valid1, valid0, H, BC, OFFDIAGONAL_BLOCKS
    )
    block20 = load_offdiagonal_block(
        AkkOD, chunk, head, 1, valid2, valid0, H, BC, OFFDIAGONAL_BLOCKS
    )
    block21 = load_offdiagonal_block(
        AkkOD, chunk, head, 2, valid2, valid1, H, BC, OFFDIAGONAL_BLOCKS
    )
    block30 = load_offdiagonal_block(
        AkkOD, chunk, head, 3, valid3, valid0, H, BC, OFFDIAGONAL_BLOCKS
    )
    block31 = load_offdiagonal_block(
        AkkOD, chunk, head, 4, valid3, valid1, H, BC, OFFDIAGONAL_BLOCKS
    )
    block32 = load_offdiagonal_block(
        AkkOD, chunk, head, 5, valid3, valid2, H, BC, OFFDIAGONAL_BLOCKS
    )

    inverse10 = -narrow_dot(narrow_dot(inverse11, block10, output_type), inverse00, output_type)
    inverse21 = -narrow_dot(narrow_dot(inverse22, block21, output_type), inverse11, output_type)
    inverse32 = -narrow_dot(narrow_dot(inverse33, block32, output_type), inverse22, output_type)
    inverse20 = -narrow_dot(
        inverse22,
        narrow_dot(block20, inverse00, output_type) + narrow_dot(block21, inverse10, output_type),
        output_type,
    )
    inverse31 = -narrow_dot(
        inverse33,
        narrow_dot(block31, inverse11, output_type) + narrow_dot(block32, inverse21, output_type),
        output_type,
    )
    inverse30 = -narrow_dot(
        inverse33,
        narrow_dot(block30, inverse00, output_type)
        + narrow_dot(block31, inverse10, output_type)
        + narrow_dot(block32, inverse20, output_type),
        output_type,
    )

    zero = tl.zeros([BC, BC], dtype=tl.float32)
    store_block(output, inverse00, token_start, head, 0, 0, valid0, valid0, H, True, BT, BC)
    store_block(output, zero, token_start, head, 0, 1, valid0, valid1, H, False, BT, BC)
    store_block(output, zero, token_start, head, 0, 2, valid0, valid2, H, False, BT, BC)
    store_block(output, zero, token_start, head, 0, 3, valid0, valid3, H, False, BT, BC)
    store_block(output, inverse10, token_start, head, 1, 0, valid1, valid0, H, False, BT, BC)
    store_block(output, inverse11, token_start, head, 1, 1, valid1, valid1, H, True, BT, BC)
    store_block(output, zero, token_start, head, 1, 2, valid1, valid2, H, False, BT, BC)
    store_block(output, zero, token_start, head, 1, 3, valid1, valid3, H, False, BT, BC)
    store_block(output, inverse20, token_start, head, 2, 0, valid2, valid0, H, False, BT, BC)
    store_block(output, inverse21, token_start, head, 2, 1, valid2, valid1, H, False, BT, BC)
    store_block(output, inverse22, token_start, head, 2, 2, valid2, valid2, H, True, BT, BC)
    store_block(output, zero, token_start, head, 2, 3, valid2, valid3, H, False, BT, BC)
    store_block(output, inverse30, token_start, head, 3, 0, valid3, valid0, H, False, BT, BC)
    store_block(output, inverse31, token_start, head, 3, 1, valid3, valid1, H, False, BT, BC)
    store_block(output, inverse32, token_start, head, 3, 2, valid3, valid2, H, False, BT, BC)
    store_block(output, inverse33, token_start, head, 3, 3, valid3, valid3, H, True, BT, BC)


def chunk_kda_fwd_k4b_triton(
    AkkOD: torch.Tensor,
    Akkd: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
    *,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Assemble dense or packed BT64 KDA inverse blocks with Triton.

    Args:
        AkkOD: Contiguous FP32 K3 factors with shape ``[capacity * 6, H * 256]``
            and block order ``(10, 20, 21, 30, 31, 32)``.
        Akkd: Contiguous FP32 pre-inverted diagonal factors with shape
            ``[1, T, H, 16]``.
        metadata: Packed chunk routing, or ``None`` for complete dense chunks.
        output_dtype: Output storage dtype, either FP16 or BF16.

    Returns:
        The causal inverse with shape ``[1, T, H, 64]``.
    """
    if Akkd.ndim != 4:
        raise ValueError(f"Akkd must be 4D, got shape {tuple(Akkd.shape)}")
    batch, tokens, heads, diagonal_width = Akkd.shape
    if batch != 1 or diagonal_width != _SUBCHUNK_SIZE or heads < 1:
        raise ValueError(
            "Triton K4 requires Akkd shape [1, T, H, 16] with at least one head, "
            f"got {tuple(Akkd.shape)}"
        )
    if AkkOD.ndim != 2:
        raise ValueError(f"AkkOD must be 2D, got shape {tuple(AkkOD.shape)}")
    if output_dtype not in _SUPPORTED_OUTPUT_DTYPES:
        raise TypeError("Triton K4 output_dtype must be torch.float16 or torch.bfloat16")
    if AkkOD.dtype != torch.float32 or Akkd.dtype != torch.float32:
        raise TypeError("Triton K4 requires FP32 AkkOD and Akkd")
    if AkkOD.device != Akkd.device or not Akkd.is_cuda:
        raise ValueError("Triton K4 requires AkkOD and Akkd on the same CUDA device")
    if not AkkOD.is_contiguous() or not Akkd.is_contiguous():
        raise ValueError("Triton K4 requires contiguous AkkOD and Akkd")

    if metadata is None:
        if tokens % _CHUNK_SIZE:
            raise ValueError(f"dense Triton K4 requires complete BT64 chunks, got T={tokens}")
        capacity = tokens // _CHUNK_SIZE
        cu_seqlens = None
        chunk_offsets = None
        num_sequences = 0
    else:
        metadata.validate_chunk_size(_CHUNK_SIZE)
        cu_seqlens = metadata.cu_seqlens
        chunk_offsets = metadata.chunk_offsets
        capacity = metadata.capacity
        num_sequences = cu_seqlens.shape[0] - 1

    expected_shape = (capacity * _OFFDIAGONAL_BLOCKS, heads * _SUBCHUNK_SIZE**2)
    if AkkOD.shape != expected_shape:
        raise ValueError(f"AkkOD must have shape {expected_shape}, got {tuple(AkkOD.shape)}")

    output = torch.empty(
        (batch, tokens, heads, _CHUNK_SIZE),
        dtype=output_dtype,
        device=Akkd.device,
    )
    if isinstance(Akkd, FakeTensor) or capacity == 0:
        return output

    chunk_kda_fwd_k4b_triton_kernel[(capacity, heads)](
        AkkOD=AkkOD,
        Akkd=Akkd,
        output=output,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=tokens,
        num_sequences=num_sequences,
        H=heads,
        BT=_CHUNK_SIZE,
        BC=_SUBCHUNK_SIZE,
        OFFDIAGONAL_BLOCKS=_OFFDIAGONAL_BLOCKS,
        num_warps=4,
        num_stages=2,
    )
    return output


__all__ = ["chunk_kda_fwd_k4b_triton"]
