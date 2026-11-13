# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable Triton KDA intra-chunk backward for BT64 and K=128.

The stage differentiates the forward intra factors ``Aqk[i, j] = sum_d q_i k_j 2^(g_i - g_j)``
and ``Akk[i, j] = beta_i sum_d k_i k_j 2^(g_i - g_j)`` (scale is folded into ``dAqk``). It adds
its dq/dk/dbeta/dgate contributions to the running gradients and owns the last dgate write, so
it converts the complete gate gradient to the log2 domain (``* ln 2``).

One program owns one 16-row subchunk of one chunk and head and walks every key channel in
``BK``-wide blocks, so the beta gradient reduces over all channels in registers without
partials. Every gate product is rebased at a row of the owning subchunk:

- earlier subchunks at its first row, ``k_j 2^(g_first - g_j) <= 1``;
- its own rows at its middle row, so both factors stay below ``2^68.3`` at the gate bound;
- later subchunks (transposed terms) at its last valid row, ``q_j 2^(g_j - g_last) <= 1``.

Inside the fused gate contract the in-subchunk factors therefore stay normal. A factor that
reaches an earlier or later subchunk still leaves the normal range once its decay falls below
``2^-126``; the flush-to-zero fastmath ``exp2`` then drops it even where a large operand would
have made the term normal ("Fast math and precision" in ``docs/linear.md``). The earlier-key
row factor ``2^(g_i - g_first)`` is formed as ``2^(g_i - g_mid) * 2^(g_mid - g_first)``:
deriving it directly would flush to zero for the last rows at steep gates while their
in-subchunk terms still matter. Matrix products run in TF32 with FP32 accumulation.
"""

from __future__ import annotations

from typing import Any

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import LN_2, ptr_offset, requires_int64_offsets
from attn_gym.linear._delta_rule.triton.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.linear.kda.utils import exp2

_CHUNK_SIZE = 64
_SUBCHUNK_SIZE = 16
_HEAD_DIM = 128
_IO_TYPES = (torch.float16, torch.bfloat16)


@triton.jit(do_not_specialize=["num_sequences"])
def chunk_kda_bwd_intra_kernel(
    q,
    k,
    g,
    beta,
    d_aqk,
    d_akk,
    dq,
    dk,
    db,
    dg,
    dq_out,
    dk_out,
    db_out,
    dg_out,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    FASTMATH: tl.constexpr,
):
    """Write one subchunk's complete dq/dk/dg rows and its final dbeta."""
    NC: tl.constexpr = BT // BC
    task = tl.program_id(0)
    head = tl.program_id(1)
    global_chunk = task // NC
    subchunk = task % NC
    if IS_RAGGED:
        if global_chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        _, _, token_start, valid = load_ragged_chunk_work(
            cu_seqlens, chunk_offsets, global_chunk, num_sequences, BT
        )
    else:
        token_start = global_chunk * BT
        valid = BT
    row_base = subchunk * BC
    if row_base >= valid:
        return
    if USE_INT64_OFFSETS:
        token_start = token_start.to(tl.int64)
        head = head.to(tl.int64)

    offs = tl.arange(0, BC)
    local_row = row_base + offs
    row_mask = local_row < valid
    token = token_start + local_row
    ref_mid = tl.minimum(row_base + BC // 2, valid - 1)
    ref_last = tl.minimum(row_base + BC, valid) - 1
    subchunks = tl.cdiv(valid, BC)

    # The owning subchunk's dA tiles: rows i against its own columns, and the transposed view
    # (later rows of the same subchunk against row i) for the dk terms.
    a_offset = ptr_offset((token[:, None], head, local_row[None, :]), (H * BT, BT, 1))
    causal = (offs[None, :] <= offs[:, None]) & row_mask[:, None]
    strict = offs[None, :] < offs[:, None]
    diagonal = offs[None, :] == offs[:, None]
    daq_tile = tl.load(d_aqk + a_offset, mask=causal, other=0.0)
    dak_tile = tl.load(d_akk + a_offset, mask=causal, other=0.0)
    daq_diag = tl.sum(tl.where(diagonal, daq_tile, 0.0), axis=1)
    dak_diag = tl.sum(tl.where(diagonal, dak_tile, 0.0), axis=1)
    daq_strict = tl.where(strict, daq_tile, 0.0)
    dak_strict = tl.where(strict, dak_tile, 0.0)
    daq_t = tl.trans(daq_strict)
    dak_t = tl.trans(dak_strict)
    beta_i = tl.load(beta + token * H + head, mask=row_mask, other=0.0)

    db_acc = tl.zeros((BC,), dtype=tl.float32)
    for key_start in tl.static_range(0, K, BK):
        key = key_start + tl.arange(0, BK)
        tile_mask = row_mask[:, None]
        compact = ptr_offset((token[:, None], head, key[None, :]), (H * K, K, 1))
        q_offset = ptr_offset((token[:, None], head, key[None, :]), (q_stride_t, q_stride_h, 1))
        k_offset = ptr_offset((token[:, None], head, key[None, :]), (k_stride_t, k_stride_h, 1))
        g_first = tl.load(g + ptr_offset((token_start + row_base, head, key), (H * K, K, 1)))
        g_mid = tl.load(g + ptr_offset((token_start + ref_mid, head, key), (H * K, K, 1)))
        g_last = tl.load(g + ptr_offset((token_start + ref_last, head, key), (H * K, K, 1)))

        # Column terms from later subchunks, rebased at the owning subchunk's last valid row.
        dk_t = tl.zeros((BC, BK), dtype=tl.float32)
        for j in range(subchunk + 1, subchunks):
            local_query = j * BC + offs
            query_mask = local_query < valid
            query_token = token_start + local_query
            q_j = tl.load(
                q
                + ptr_offset(
                    (query_token[:, None], head, key[None, :]), (q_stride_t, q_stride_h, 1)
                ),
                mask=query_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            k_j = tl.load(
                k
                + ptr_offset(
                    (query_token[:, None], head, key[None, :]), (k_stride_t, k_stride_h, 1)
                ),
                mask=query_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            g_j = tl.load(
                g + ptr_offset((query_token[:, None], head, key[None, :]), (H * K, K, 1)),
                mask=query_mask[:, None],
                other=0.0,
            )
            beta_j = tl.load(beta + query_token * H + head, mask=query_mask, other=0.0)
            decay = tl.where(query_mask[:, None], exp2(g_j - g_last[None, :], FASTMATH), 0.0)
            later_offset = ptr_offset(
                (query_token[:, None], head, local_row[None, :]), (H * BT, BT, 1)
            )
            later_mask = query_mask[:, None] & row_mask[None, :]
            daq = tl.load(d_aqk + later_offset, mask=later_mask, other=0.0)
            dak = tl.load(d_akk + later_offset, mask=later_mask, other=0.0)
            dk_t += tl.dot(tl.trans(daq), q_j * decay)
            dk_t += tl.dot(tl.trans(dak), k_j * beta_j[:, None] * decay)

        # Row terms from earlier subchunks, rebased at the owning subchunk's first row.
        dq_strict = tl.zeros((BC, BK), dtype=tl.float32)
        dk_qk = tl.zeros((BC, BK), dtype=tl.float32)
        for j in range(subchunk):
            key_token = token_start + j * BC + offs
            k_j = tl.load(
                k
                + ptr_offset((key_token[:, None], head, key[None, :]), (k_stride_t, k_stride_h, 1))
            ).to(tl.float32)
            g_j = tl.load(g + ptr_offset((key_token[:, None], head, key[None, :]), (H * K, K, 1)))
            k_gated = k_j * exp2(g_first[None, :] - g_j, FASTMATH)
            prev_offset = ptr_offset(
                (token[:, None], head, (j * BC + offs)[None, :]), (H * BT, BT, 1)
            )
            daq = tl.load(d_aqk + prev_offset, mask=tile_mask, other=0.0)
            dak = tl.load(d_akk + prev_offset, mask=tile_mask, other=0.0)
            dq_strict += tl.dot(daq, k_gated)
            dk_qk += tl.dot(dak, k_gated)

        # The owning subchunk's own terms, rebased at its middle row. Invalid rows take the
        # middle-row gate so every exponent stays finite.
        q_i = tl.load(q + q_offset, mask=tile_mask, other=0.0).to(tl.float32)
        k_i = tl.load(k + k_offset, mask=tile_mask, other=0.0).to(tl.float32)
        g_i = tl.load(g + compact, mask=tile_mask, other=0.0)
        g_i = tl.where(tile_mask, g_i, g_mid[None, :])
        to_mid = exp2(g_mid[None, :] - g_i, FASTMATH)
        from_mid = exp2(g_i - g_mid[None, :], FASTMATH)
        # Combine the scale factors before scaling an accumulator: each factor alone can push
        # a small normal accumulator below the FP32 normal range.
        first_scale = from_mid * exp2(g_mid - g_first, FASTMATH)[None, :]
        last_scale = to_mid * exp2(g_last - g_mid, FASTMATH)[None, :]
        k_mid = tl.where(tile_mask, k_i * to_mid, 0.0)
        dq_strict = dq_strict * first_scale + tl.dot(daq_strict, k_mid) * from_mid
        dk_qk = dk_qk * first_scale + tl.dot(dak_strict, k_mid) * from_mid
        q_from_mid = tl.where(tile_mask, q_i * from_mid, 0.0)
        kb_from_mid = tl.where(tile_mask, k_i * beta_i[:, None] * from_mid, 0.0)
        dk_t = (
            dk_t * last_scale + (tl.dot(daq_t, q_from_mid) + tl.dot(dak_t, kb_from_mid)) * to_mid
        )

        dk_beta = dk_qk * beta_i[:, None]
        dq_new = tl.load(dq + compact, mask=tile_mask, other=0.0)
        dq_new += dq_strict + daq_diag[:, None] * k_i
        tl.store(dq_out + compact, dq_new.to(dq_out.dtype.element_ty), mask=tile_mask)
        dk_new = tl.load(dk + compact, mask=tile_mask, other=0.0)
        dk_new += (
            dk_beta + dk_t + daq_diag[:, None] * q_i + (2.0 * dak_diag * beta_i)[:, None] * k_i
        )
        tl.store(dk_out + compact, dk_new.to(dk_out.dtype.element_ty), mask=tile_mask)
        dg_new = tl.load(dg + compact, mask=tile_mask, other=0.0)
        dg_new += dq_strict * q_i + (dk_beta - dk_t) * k_i
        tl.store(dg_out + compact, dg_new * LN_2, mask=tile_mask)
        db_acc += tl.sum((dk_qk + dak_diag[:, None] * k_i) * k_i, axis=1)

    scalar = token * H + head
    db_in = tl.load(db + scalar, mask=row_mask, other=0.0)
    tl.store(db_out + scalar, db_in + db_acc, mask=row_mask)


def _tuning_key(args: dict[str, Any]) -> tuple[Any, ...]:
    # Tune per power of two of the chunk grid, not per T or document count.
    return (
        args["q"].dtype,
        args["H"],
        triton.next_power_of_2(triton.cdiv(args["dq"].shape[1], args["BT"])),
        args["IS_RAGGED"],
        args["USE_INT64_OFFSETS"],
        args["FASTMATH"],
    )


# Each launch writes fresh output buffers from its inputs, so benchmarking needs no reset.
_intra = TritonTuner(
    chunk_kda_bwd_intra_kernel,
    [
        triton.Config({"BK": 64}, num_warps=2, num_stages=2),
        triton.Config({"BK": 128}, num_warps=2, num_stages=2),
        triton.Config({"BK": 128}, num_warps=4, num_stages=2),
        triton.Config({"BK": 128}, num_warps=2, num_stages=1),
        triton.Config({"BK": 64}, num_warps=4, num_stages=2),
        triton.Config({"BK": 32}, num_warps=1, num_stages=2),
    ],
    key=_tuning_key,
)


def chunk_kda_bwd_intra_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
    *,
    autotune: bool,
    fastmath: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(dq, dk, dg, db)`` after adding the intra-chunk contributions.

    ``dq``/``dk`` return in the Q/K dtype, ``dg`` in FP32 with the ``ln(2)`` factor for
    differentiating ``2**g`` applied to the incoming and intra terms, and ``db`` like the
    incoming ``db``. Q and K need only a contiguous last dimension; the other tensors must be
    contiguous.
    """
    batch, tokens, heads, head_dim = q.shape
    if batch != 1 or head_dim != _HEAD_DIM:
        raise ValueError("the intra-chunk backward requires B=1 and K=128")
    if q.dtype not in _IO_TYPES or k.dtype != q.dtype:
        raise TypeError("q and k must share dtype float16 or bfloat16")
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        raise ValueError("q and k must be contiguous in their last dimension")
    compact = (g, beta, dAqk, dAkk, dq, dk, db, dg)
    if any(not tensor.is_contiguous() for tensor in compact):
        raise ValueError("g, beta, dAqk, dAkk, dq, dk, db, and dg must be contiguous")
    if metadata is not None:
        metadata.validate_chunk_size(_CHUNK_SIZE)
        capacity = metadata.capacity
        cu_seqlens, chunk_offsets = metadata.cu_seqlens, metadata.chunk_offsets
        num_sequences = cu_seqlens.shape[0] - 1
    elif tokens % _CHUNK_SIZE:
        raise ValueError("the intra-chunk backward requires complete 64-token chunks")
    else:
        capacity = tokens // _CHUNK_SIZE
        cu_seqlens = chunk_offsets = None
        num_sequences = 1

    dq_out = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk_out = torch.empty_like(k, memory_format=torch.contiguous_format)
    dg_out = torch.empty_like(g, memory_format=torch.contiguous_format)
    db_out = torch.empty_like(db, memory_format=torch.contiguous_format)
    if tokens == 0:
        db_out.copy_(db)
        return dq_out, dk_out, dg_out, db_out

    grid = (capacity * (_CHUNK_SIZE // _SUBCHUNK_SIZE), heads)
    _intra[grid](
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db,
        dg,
        dq_out,
        dk_out,
        db_out,
        dg_out,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        H=heads,
        K=_HEAD_DIM,
        BT=_CHUNK_SIZE,
        BC=_SUBCHUNK_SIZE,
        IS_RAGGED=metadata is not None,
        USE_INT64_OFFSETS=requires_int64_offsets(
            q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, dq_out, dk_out, db_out, dg_out
        ),
        FASTMATH=fastmath,
        enable_reflect_ftz=fastmath,
        autotune=autotune,
    )
    return dq_out, dk_out, dg_out, db_out


__all__ = ["chunk_kda_bwd_intra_triton"]
