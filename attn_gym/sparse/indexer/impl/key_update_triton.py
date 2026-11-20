"""Triton paged indexer key update: pooling, offset RMSNorm, partial RoPE, and stores."""

import triton
import triton.language as tl

from attn_gym._backends.triton.utils import _document_ids

_POOL = 1
_RING = 2


@triton.jit
def _pooled(
    RawKeys,
    RawCache,
    row,
    start,
    position,
    ring_page,
    ring_capacity,
    lanes,
    mask,
    stride_rt,
    stride_rd,
    stride_cp,
    stride_cs,
    stride_cd,
    RATIO: tl.constexpr,
):
    total = tl.zeros(lanes.shape, tl.float32)
    for shift in tl.static_range(RATIO - 1, -1, -1):
        local = row - shift
        in_step = local >= start
        step_value = tl.load(
            RawKeys + local.to(tl.int64) * stride_rt + lanes * stride_rd,
            mask=mask & in_step,
            other=0.0,
        )
        ring_value = tl.load(
            RawCache
            + ring_page.to(tl.int64) * stride_cp
            + ((position - shift) % ring_capacity).to(tl.int64) * stride_cs
            + lanes * stride_cd,
            mask=mask & ~in_step,
            other=0.0,
        )
        total += tl.where(in_step, step_value, ring_value).to(tl.float32)
    return (total / RATIO).to(RawKeys.dtype.element_ty).to(tl.float32)


@triton.jit(do_not_specialize=["sequences", "ring_capacity", "page_size"])
def _key_update(
    RawKeys,
    KeyCache,
    RawCache,
    BlockTable,
    RawBlockTable,
    CuQ,
    Lengths,
    Weight,
    Rope,
    sequences,
    ring_capacity,
    page_size,
    eps,
    stride_rt,
    stride_rd,
    stride_kp,
    stride_ks,
    stride_kd,
    stride_cp,
    stride_cs,
    stride_cd,
    stride_bs,
    stride_bp,
    stride_qs,
    stride_ropep,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROTARY: tl.constexpr,
    RATIO: tl.constexpr,
    STAGES: tl.constexpr,
):
    row = tl.program_id(0)
    sequence = _document_ids(CuQ, row, sequences, False)
    if sequence >= sequences:
        return
    start = tl.load(CuQ + sequence)
    end = tl.load(CuQ + sequence + 1)
    length = tl.load(Lengths + sequence)
    position = length - end + row
    ring_page = tl.load(RawBlockTable + sequence.to(tl.int64) * stride_qs)
    lanes = tl.arange(0, BLOCK_D)
    mask = lanes < HEAD_DIM
    if STAGES & 1:  # noqa: SIM102 (static, then dynamic)
        if (position + 1) % RATIO == 0:
            half = ROTARY // 2
            partner = tl.where(
                lanes < half, lanes + half, tl.where(lanes < ROTARY, lanes - half, lanes)
            )
            pooled = _pooled(
                RawKeys,
                RawCache,
                row,
                start,
                position,
                ring_page,
                ring_capacity,
                lanes,
                mask,
                stride_rt,
                stride_rd,
                stride_cp,
                stride_cs,
                stride_cd,
                RATIO,
            )
            partner_pooled = _pooled(
                RawKeys,
                RawCache,
                row,
                start,
                position,
                ring_page,
                ring_capacity,
                partner,
                mask,
                stride_rt,
                stride_rd,
                stride_cp,
                stride_cs,
                stride_cd,
                RATIO,
            )
            variance = tl.sum(tl.where(mask, pooled * pooled, 0.0), axis=0) / HEAD_DIM
            inverse = tl.rsqrt(variance + eps)
            out_ty = RawKeys.dtype.element_ty
            weight = tl.load(Weight + lanes, mask=mask, other=0.0).to(tl.float32)
            partner_weight = tl.load(Weight + partner, mask=mask, other=0.0).to(tl.float32)
            normed = ((1.0 + weight) * (pooled * inverse)).to(out_ty).to(tl.float32)
            partner_normed = ((1.0 + partner_weight) * (partner_pooled * inverse)).to(out_ty)
            partner_normed = partner_normed.to(tl.float32)
            rotary = lanes < ROTARY
            rope_row = Rope + (position - RATIO + 1).to(tl.int64) * stride_ropep
            cos = tl.load(rope_row + lanes, mask=rotary, other=1.0).to(out_ty).to(tl.float32)
            sin = tl.load(rope_row + ROTARY + lanes, mask=rotary, other=0.0)
            sin = sin.to(out_ty).to(tl.float32)
            signed = tl.where(lanes < half, -partner_normed, partner_normed)
            direct = (normed * cos).to(out_ty).to(tl.float32)
            crossed = (signed * sin).to(out_ty).to(tl.float32)
            rotated = tl.where(rotary, (direct + crossed).to(out_ty), normed.to(out_ty))
            candidate = position // RATIO
            page = tl.load(
                BlockTable
                + sequence.to(tl.int64) * stride_bs
                + (candidate // page_size).to(tl.int64) * stride_bp
            )
            tl.store(
                KeyCache
                + page.to(tl.int64) * stride_kp
                + (candidate % page_size).to(tl.int64) * stride_ks
                + lanes * stride_kd,
                rotated,
                mask=mask,
            )
    if STAGES & 2:  # noqa: SIM102 (static, then dynamic)
        if position >= length // RATIO * RATIO:
            value = tl.load(RawKeys + row.to(tl.int64) * stride_rt + lanes * stride_rd, mask=mask)
            tl.store(
                RawCache
                + ring_page.to(tl.int64) * stride_cp
                + (position % ring_capacity).to(tl.int64) * stride_cs
                + lanes * stride_cd,
                value,
                mask=mask,
            )


def launch(
    raw_keys,
    key_cache,
    raw_cache,
    block_table,
    raw_block_table,
    cu_seqlens_q,
    seqused_k,
    norm_weight,
    rope_cache,
    compress_ratio,
    eps,
    max_seqlen_q,
):
    """Run one pass when no ring slot can be read and written in the same step.

    A sequence with ``n`` queries reads ring positions at most 3 before its first
    query and writes positions in its final incomplete block, so a read and a write
    land on the same slot only if ``n + 2 >= ring_capacity``. Larger host bounds,
    or an unknown one, read every member before any ring write with two passes.
    Multiply-add contraction is disabled so every product and sum rounds as in the
    eager reference, which makes the stored keys bitwise equal to it.
    """
    tokens, head_dim = raw_keys.shape
    if tokens == 0:
        return
    ring_capacity = raw_cache.shape[1]
    single_pass = max_seqlen_q is not None and max_seqlen_q + 2 < ring_capacity
    for stages in (_POOL | _RING,) if single_pass else (_POOL, _RING):
        _key_update[(tokens,)](
            raw_keys,
            key_cache,
            raw_cache,
            block_table,
            raw_block_table,
            cu_seqlens_q,
            seqused_k,
            norm_weight,
            rope_cache,
            seqused_k.shape[0],
            ring_capacity,
            key_cache.shape[1],
            eps,
            raw_keys.stride(0),
            raw_keys.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            raw_cache.stride(0),
            raw_cache.stride(1),
            raw_cache.stride(2),
            block_table.stride(0),
            block_table.stride(1),
            raw_block_table.stride(0),
            rope_cache.stride(0),
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            ROTARY=rope_cache.shape[1] // 2,
            RATIO=compress_ratio,
            STAGES=stages,
            num_warps=1,
            enable_fp_fusion=False,
        )


__all__ = ["launch"]
