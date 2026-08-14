# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Chunk-local cumulative sum along the time axis. In the KDA backward pass the
# per-channel decay gate is accumulated *in reverse* within each chunk (the
# adjoint of a forward prefix-sum is a reverse prefix-sum), so both directions
# are supported.
#
# The kernels use logical (B, T, H[, S]) strides, including for physical
# head-first layouts. The leading batch stride is passed separately so Triton can
# specialize on its alignment without treating its T-dependent value as constexpr;
# inner strides remain constexpr for codegen. Variable-length packing and an
# optional output scale are also supported. Larger token-major scalar gates reuse
# the vector kernel through a zero-copy (B, T, 1, H) view.

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, load_ragged_chunk_work
from attn_gym.linear.kda.utils import (
    autotune_cache_kwargs,
    check_shared_mem,
    input_guard,
    prepare_chunk_indices,
)

BS_LIST = [32, 64] if check_shared_mem() else [16, 32]
# H=8 is near the scalar/vector crossover on GB300; K3's supported shapes start at H=16.
_MIN_VECTOR_HEADS = 16


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in [1, 2, 4, 8]],
    key=["B", "H", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    s_batch_stride,
    o_batch_stride,
    S_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    BT: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    tl.static_assert(not IS_VARLEN or B == 1, "packed varlen requires B == 1")
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    # Resolve the row span of this chunk (constant time for the fixed-length
    # case, or looked up from the packing metadata for varlen batches).
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = 0

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    token = bos + o_t if IS_VARLEN else o_t

    s_batch_offset = i_b * s_batch_stride if B > 1 else 0
    b_s = tl.load(
        s + s_batch_offset + ptr_offset((token, i_h), S_STRIDES),
        mask=m_t,
        other=0.0,
    ).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
    if HAS_SCALE:
        b_o = b_o * scale
    o_batch_offset = i_b * o_batch_stride if B > 1 else 0
    tl.store(
        o + o_batch_offset + ptr_offset((token, i_h), O_STRIDES),
        b_o.to(o.dtype.element_ty),
        mask=m_t,
    )


@triton.heuristics(
    {
        "HAS_SCALE": lambda args: args["scale"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({"BS": bs}, num_warps=w) for bs in BS_LIST for w in [2, 4, 8]],
    # BS changes the scan order; exclude grid-only B so batch changes reuse one tile.
    key=["H", "S", "BT", "IS_VARLEN", "REVERSE"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    s_batch_stride,
    o_batch_stride,
    S_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    tl.static_assert(not IS_VARLEN or B == 1, "packed varlen requires B == 1")
    i_s = tl.program_id(0).to(tl.int64)
    i_t = tl.program_id(1).to(tl.int64)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = eos - bos
    else:
        bos = 0

    o_t = i_t * BT + tl.arange(0, BT)
    o_s = i_s * BS + tl.arange(0, BS)
    m = (o_t[:, None] < T) & (o_s[None, :] < S)
    token = bos + o_t if IS_VARLEN else o_t

    s_batch_offset = i_b * s_batch_stride if B > 1 else 0
    b_s = tl.load(
        s
        + s_batch_offset
        + ptr_offset(
            (token[:, None], i_h, o_s[None, :]),
            S_STRIDES,
        ),
        mask=m,
        other=0.0,
    ).to(tl.float32)
    b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
    if HAS_SCALE:
        b_o = b_o * scale
    o_batch_offset = i_b * o_batch_stride if B > 1 else 0
    tl.store(
        o
        + o_batch_offset
        + ptr_offset(
            (token[:, None], i_h, o_s[None, :]),
            O_STRIDES,
        ),
        b_o.to(o.dtype.element_ty),
        mask=m,
    )


@triton.jit(do_not_specialize=["num_sequences"])
def ragged_chunk_local_cumsum_vector_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    S_STRIDES: tl.constexpr,
    O_STRIDES: tl.constexpr,
    S: tl.constexpr,
    BT: tl.constexpr,
    BS: tl.constexpr,
    REVERSE: tl.constexpr,
):
    """Scan one device-routed sequence-local chunk in FP32."""
    global_chunk = tl.program_id(0)
    i_h = tl.program_id(1).to(tl.int64)
    i_s = tl.program_id(2).to(tl.int64)
    if global_chunk >= tl.load(chunk_offsets + num_sequences):
        return

    _sequence, _local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
        cu_seqlens,
        chunk_offsets,
        global_chunk,
        num_sequences,
        BT,
    )
    token = token_start.to(tl.int64) + tl.arange(0, BT)
    channel = i_s * BS + tl.arange(0, BS)
    mask = (tl.arange(0, BT)[:, None] < valid_tokens) & (channel[None, :] < S)
    values = tl.load(
        s + ptr_offset((token[:, None], i_h, channel[None, :]), S_STRIDES),
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    result = tl.cumsum(values, axis=0, reverse=REVERSE) * scale
    tl.store(
        o + ptr_offset((token[:, None], i_h, channel[None, :]), O_STRIDES),
        result,
        mask=mask,
    )


@input_guard(no_guard_contiguous=True)
def ragged_chunk_local_cumsum_vector(
    g: torch.Tensor,
    metadata: RaggedChunkMetadata,
    *,
    reverse: bool,
    scale: float,
) -> torch.Tensor:
    """Compute a graph-safe FP32 cumsum over device-routed ragged chunks."""
    if g.ndim != 4 or g.shape[0] != 1:
        raise ValueError(f"ragged vector cumsum expects shape [1, T, H, D], got {tuple(g.shape)}")
    if metadata.cu_seqlens.device != g.device:
        raise ValueError("ragged metadata and input must be on the same device")

    output = torch.empty_like(g, dtype=torch.float32)
    _, _, heads, head_dim = g.shape
    block_size = 32
    ragged_chunk_local_cumsum_vector_kernel[
        (metadata.capacity, heads, triton.cdiv(head_dim, block_size))
    ](
        g,
        output,
        scale,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        metadata.cu_seqlens.shape[0] - 1,
        S_STRIDES=g.stride()[1:],
        O_STRIDES=output.stride()[1:],
        S=head_dim,
        BT=metadata.chunk_size,
        BS=block_size,
        REVERSE=reverse,
        num_warps=4,
    )
    return output


@input_guard(no_guard_contiguous=("g",))
def chunk_local_cumsum_scalar(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float32,
    chunk_indices: torch.LongTensor | None = None,
) -> torch.Tensor:
    """Compute chunk-local cumsums for scalar gates.

    Token-major inputs with at least 16 heads use a zero-copy ``(B, T, 1, H)``
    view of the vector kernel, including for strided tensors. Smaller and
    head-first inputs retain the scalar kernel. Packed inputs require batch size one.
    """
    if g.ndim != 3:
        raise ValueError(f"Expected a 3D scalar gate, got shape {tuple(g.shape)}")
    if chunk_size <= 0 or chunk_size & (chunk_size - 1):
        raise ValueError(f"chunk_size must be a positive power of two, got {chunk_size}")
    if cu_seqlens is not None and g.shape[0] != 1:
        raise ValueError("Packed variable-length inputs must have batch size one")

    output = torch.empty_like(
        g,
        dtype=output_dtype if output_dtype is not None else g.dtype,
    )
    source = g.transpose(1, 2) if head_first else g
    destination = output.transpose(1, 2) if head_first else output
    batch, tokens, heads = source.shape

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    chunks = triton.cdiv(tokens, chunk_size) if cu_seqlens is None else len(chunk_indices)

    if not head_first and heads >= _MIN_VECTOR_HEADS:
        source = source.unsqueeze(2)
        destination = destination.unsqueeze(2)

        def grid(meta):
            return (triton.cdiv(heads, meta["BS"]), chunks, batch)

        chunk_local_cumsum_vector_kernel[grid](
            source,
            destination,
            scale,
            cu_seqlens,
            chunk_indices,
            tokens,
            source.stride(0),
            destination.stride(0),
            S_STRIDES=source.stride()[1:],
            O_STRIDES=destination.stride()[1:],
            B=batch,
            H=1,
            S=heads,
            BT=chunk_size,
            REVERSE=reverse,
        )
    else:
        chunk_local_cumsum_scalar_kernel[(chunks, batch * heads)](
            source,
            destination,
            scale,
            cu_seqlens,
            chunk_indices,
            tokens,
            source.stride(0),
            destination.stride(0),
            S_STRIDES=source.stride()[1:],
            O_STRIDES=destination.stride()[1:],
            B=batch,
            H=heads,
            BT=chunk_size,
            REVERSE=reverse,
        )
    return output
