"""Triton forward backend for compressed sparse attention."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _pad_to_block_size(x: torch.Tensor, block_size: int, value: float) -> torch.Tensor:
    pad_length = (-x.shape[-2]) % block_size
    if pad_length == 0:
        return x
    return F.pad(x, (0, 0, 0, pad_length), mode="constant", value=value)


def _compress(
    c_a: torch.Tensor,
    c_b: torch.Tensor,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    b_a: torch.Tensor,
    b_b: torch.Tensor,
    compression_rate: int,
) -> torch.Tensor:
    """Apply the reference compression equations using native PyTorch views."""
    c_a = _pad_to_block_size(c_a, compression_rate, 0.0)
    c_b = _pad_to_block_size(c_b, compression_rate, 0.0)
    z_a = _pad_to_block_size(z_a, compression_rate, float("-inf"))
    z_b = _pad_to_block_size(z_b, compression_rate, float("-inf"))

    c_b = F.pad(c_b, (0, 0, compression_rate, 0), value=0.0)[:, :, :-compression_rate, :]
    z_b = F.pad(z_b, (0, 0, compression_rate, 0), value=float("-inf"))[:, :, :-compression_rate, :]

    batch, heads, padded_sequence, dimension = c_a.shape
    num_blocks = padded_sequence // compression_rate
    blocked_shape = (batch, heads, num_blocks, compression_rate, dimension)
    c_a = c_a.reshape(blocked_shape)
    c_b = c_b.reshape(blocked_shape)
    z_a = z_a.reshape(blocked_shape)
    z_b = z_b.reshape(blocked_shape)

    logits = torch.cat((z_a + b_a, z_b + b_b), dim=-2)
    probabilities = logits.softmax(dim=-2)
    return (
        c_a * probabilities[..., :compression_rate, :]
        + c_b * probabilities[..., compression_rate:, :]
    ).sum(dim=-2)


def _rope_frequencies(rotary_dim: int, device: torch.device) -> torch.Tensor:
    base = 160_000.0
    original_seq_len = 65_536
    factor = 16.0
    beta_fast = 32.0
    beta_slow = 1.0
    frequencies = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32) / rotary_dim)
    )

    def correction_dimension(num_rotations: float) -> float:
        return (
            rotary_dim
            * math.log(original_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    low = max(math.floor(correction_dimension(beta_fast)), 0)
    high = min(math.ceil(correction_dimension(beta_slow)), rotary_dim - 1)
    if low == high:
        high += 0.001
    ramp = (torch.arange(rotary_dim // 2, device=device, dtype=torch.float32) - low) / (high - low)
    smooth = 1 - ramp.clamp(0, 1)
    return frequencies / factor * (1 - smooth) + frequencies * smooth


def _apply_rope(
    x: torch.Tensor,
    rotary_dim: int,
    *,
    positions: torch.Tensor | None = None,
    inverse: bool = False,
) -> torch.Tensor:
    if positions is None:
        positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)
    else:
        positions = positions.to(device=x.device, dtype=torch.float32)
    angles = torch.outer(positions, _rope_frequencies(rotary_dim, x.device))
    frequencies_complex = torch.polar(torch.ones_like(angles), angles)
    if inverse:
        frequencies_complex = frequencies_complex.conj()
    tail = x[..., -rotary_dim:].float().reshape(*x.shape[:-1], rotary_dim // 2, 2)
    tail = torch.view_as_complex(tail)
    frequency_shape = [1] * (x.ndim - 2) + [x.shape[-2], rotary_dim // 2]
    rotated = torch.view_as_real(tail * frequencies_complex.view(*frequency_shape)).flatten(-2)
    return torch.cat((x[..., :-rotary_dim], rotated.to(x.dtype)), dim=-1)


@triton.jit
def _tiled_selected_attention_fwd(
    q_ptr,
    compressed_kv_ptr,
    local_kv_ptr,
    topk_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_ls: tl.constexpr,
    stride_ld: tl.constexpr,
    stride_tb: tl.constexpr,
    stride_ts: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_les: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_LOCAL_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute selected attention without materializing dense logits or probabilities."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D
    query = tl.load(
        q_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=query_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )

    sink = tl.load(sink_ptr + head).to(tl.float32)
    running_max = tl.full((BLOCK_M,), sink, tl.float32)
    running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    completed_blocks = (offsets_m + 1) // COMPRESSION_RATE

    # Selected compressed keys differ per query, so fold them into the online softmax one at a
    # time while retaining the query tile and accumulator in registers.
    for selected_slot in tl.static_range(0, TOPK):
        selected_block = tl.load(
            topk_ptr + batch * stride_tb + offsets_m * stride_ts + selected_slot * stride_tk,
            mask=query_mask,
            other=0,
        )
        valid = (
            query_mask
            & (selected_block >= 0)
            & (selected_block < completed_blocks)
            & (selected_block < N_BLOCKS)
        )
        compressed_value = tl.load(
            compressed_kv_ptr
            + batch * stride_cb
            + head * stride_ch
            + selected_block[:, None] * stride_cn
            + offsets_d[None, :] * stride_cd,
            mask=valid[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        logit = tl.sum(query * compressed_value, axis=1) * SCALE
        logit = tl.where(valid, logit, -float("inf"))
        new_max = tl.maximum(running_max, logit)
        alpha = tl.exp(running_max - new_max)
        probability = tl.exp(logit - new_max)
        accumulator = accumulator * alpha[:, None] + probability[:, None] * compressed_value
        running_sum = running_sum * alpha + probability
        running_max = new_max

    # Adjacent query windows overlap heavily. Process their union in tensor-core-friendly tiles.
    first_local_position = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for local_tile in tl.static_range(0, NUM_LOCAL_TILES):
        offsets_n = first_local_position + local_tile * BLOCK_N + offsets_n_base
        local_mask = (offsets_n >= 0) & (offsets_n < S)
        local_values = tl.load(
            local_kv_ptr
            + batch * stride_lb
            + head * stride_lh
            + offsets_n[:, None] * stride_ls
            + offsets_d[None, :] * stride_ld,
            mask=local_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        logits = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        causal_window_mask = (
            query_mask[:, None]
            & local_mask[None, :]
            & (offsets_n[None, :] <= offsets_m[:, None])
            & (offsets_n[None, :] >= offsets_m[:, None] - WINDOW + 1)
        )
        logits = tl.where(causal_window_mask, logits, -float("inf"))

        tile_max = tl.max(logits, axis=1)
        new_max = tl.maximum(running_max, tile_max)
        alpha = tl.exp(running_max - new_max)
        probabilities = tl.exp(logits - new_max[:, None])
        accumulator *= alpha[:, None]
        accumulator += tl.dot(
            probabilities.to(local_values.dtype), local_values, input_precision="tf32x3"
        )
        running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
        running_max = new_max

    output = accumulator / running_sum[:, None]
    tl.store(
        out_ptr
        + batch * stride_ob
        + head * stride_oh
        + offsets_m[:, None] * stride_os
        + offsets_d[None, :] * stride_od,
        output,
        mask=query_mask[:, None] & dimension_mask[None, :],
    )
    tl.store(
        lse_ptr + batch * stride_leb + head * stride_leh + offsets_m * stride_les,
        running_max + tl.log(running_sum),
        mask=query_mask,
    )


@triton.jit
def _selected_compressed_attention_merge_fwd(
    q_ptr,
    compressed_kv_ptr,
    topk_ptr,
    local_output_ptr,
    local_lse_ptr,
    out_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_tb: tl.constexpr,
    stride_ts: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_les: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Merge vectorized selected-compressed attention into local attention."""
    query_position = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_k = tl.arange(0, BLOCK_K)
    offsets_d = tl.arange(0, BLOCK_D)
    dimension_mask = offsets_d < D
    query = tl.load(
        q_ptr
        + batch * stride_qb
        + head * stride_qh
        + query_position * stride_qs
        + offsets_d * stride_qd,
        mask=dimension_mask,
        other=0.0,
    )
    selected_blocks = tl.load(
        topk_ptr + batch * stride_tb + query_position * stride_ts + offsets_k * stride_tk,
        mask=offsets_k < TOPK,
        other=0,
    )
    completed_blocks = (query_position + 1) // COMPRESSION_RATE
    valid = (
        (offsets_k < TOPK)
        & (selected_blocks >= 0)
        & (selected_blocks < completed_blocks)
        & (selected_blocks < N_BLOCKS)
    )
    selected_values = tl.load(
        compressed_kv_ptr
        + batch * stride_cb
        + head * stride_ch
        + selected_blocks[:, None] * stride_cn
        + offsets_d[None, :] * stride_cd,
        mask=valid[:, None] & dimension_mask[None, :],
        other=0.0,
    )
    logits = tl.sum(selected_values * query[None, :], axis=1) * SCALE
    logits = tl.where(valid, logits, -float("inf"))
    has_compressed = tl.sum(valid.to(tl.int32), axis=0) > 0
    compressed_max = tl.max(logits, axis=0)
    safe_compressed_max = tl.where(has_compressed, compressed_max, 0.0)
    probabilities = tl.where(valid, tl.exp(logits - safe_compressed_max), 0.0)
    compressed_sum = tl.sum(probabilities, axis=0)
    compressed_numerator = tl.sum(
        probabilities[:, None] * selected_values,
        axis=0,
    )

    local_lse = tl.load(
        local_lse_ptr + batch * stride_leb + head * stride_leh + query_position * stride_les
    )
    combined_max = tl.where(
        has_compressed,
        tl.maximum(local_lse, safe_compressed_max),
        local_lse,
    )
    local_scale = tl.exp(local_lse - combined_max)
    compressed_scale = tl.where(
        has_compressed,
        tl.exp(safe_compressed_max - combined_max),
        0.0,
    )
    denominator = local_scale + compressed_scale * compressed_sum
    local_output = tl.load(
        local_output_ptr
        + batch * stride_ob
        + head * stride_oh
        + query_position * stride_os
        + offsets_d * stride_od,
        mask=dimension_mask,
        other=0.0,
    )
    output = (local_output * local_scale + compressed_numerator * compressed_scale) / denominator
    tl.store(
        local_lse_ptr + batch * stride_leb + head * stride_leh + query_position * stride_les,
        combined_max + tl.log(denominator),
    )
    tl.store(
        out_ptr
        + batch * stride_ob
        + head * stride_oh
        + query_position * stride_os
        + offsets_d * stride_od,
        output,
        mask=dimension_mask,
    )


@triton.jit
def _build_selected_block_mask(
    topk_ptr,
    mask_ptr,
    stride_tb: tl.constexpr,
    stride_ts: tl.constexpr,
    stride_tk: tl.constexpr,
    stride_mb: tl.constexpr,
    stride_ms: tl.constexpr,
    stride_mn: tl.constexpr,
    S: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    COMPRESSION_RATE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_position = tl.program_id(0)
    batch = tl.program_id(1)
    offsets_k = tl.arange(0, BLOCK_K)
    selected_blocks = tl.load(
        topk_ptr + batch * stride_tb + query_position * stride_ts + offsets_k * stride_tk,
        mask=offsets_k < TOPK,
        other=0,
    )
    completed_blocks = (query_position + 1) // COMPRESSION_RATE
    valid = (
        (offsets_k < TOPK)
        & (selected_blocks >= 0)
        & (selected_blocks < completed_blocks)
        & (selected_blocks < N_BLOCKS)
    )
    tl.store(
        mask_ptr + batch * stride_mb + query_position * stride_ms + selected_blocks * stride_mn,
        1,
        mask=valid,
    )


@triton.jit
def _local_attention_bwd_dq(
    q_ptr,
    local_kv_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    sink_ptr,
    grad_q_ptr,
    grad_sink_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_ls: tl.constexpr,
    stride_ld: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_les: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_KEY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H
    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D
    matrix_mask = query_mask[:, None] & dimension_mask[None, :]

    query = tl.load(
        q_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=matrix_mask,
        other=0.0,
    )
    output = tl.load(
        output_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=matrix_mask,
        other=0.0,
    )
    grad_output = tl.load(
        grad_output_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=matrix_mask,
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + batch * stride_leb + head * stride_leh + offsets_m * stride_les,
        mask=query_mask,
        other=0.0,
    )
    delta = tl.sum(grad_output * output, axis=1)
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    first_key = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for key_tile in tl.range(0, NUM_KEY_TILES):
        offsets_n = first_key + key_tile * BLOCK_N + offsets_n_base
        key_mask = (offsets_n >= 0) & (offsets_n < S)
        local_values = tl.load(
            local_kv_ptr
            + batch * stride_lb
            + head * stride_lh
            + offsets_n[:, None] * stride_ls
            + offsets_d[None, :] * stride_ld,
            mask=key_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        valid = (
            query_mask[:, None]
            & key_mask[None, :]
            & (offsets_n[None, :] <= offsets_m[:, None])
            & (offsets_n[None, :] >= offsets_m[:, None] - WINDOW + 1)
        )
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(
            grad_output,
            tl.trans(local_values),
            input_precision="tf32x3",
        )
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_query += (
            tl.dot(
                grad_scores.to(local_values.dtype),
                local_values,
                input_precision="tf32x3",
            )
            * SCALE
        )

    tl.store(
        grad_q_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        grad_query,
        mask=matrix_mask,
    )
    sink = tl.load(sink_ptr + head)
    sink_probability = tl.exp(sink - lse)
    sink_gradient = tl.where(query_mask, -sink_probability * delta, 0.0)
    tl.atomic_add(grad_sink_ptr + head, tl.sum(sink_gradient, axis=0))


@triton.jit
def _compressed_attention_bwd_dq(
    q_ptr,
    compressed_kv_ptr,
    selection_mask_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_q_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_mb: tl.constexpr,
    stride_ms: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_les: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_KEY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H
    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D
    matrix_mask = query_mask[:, None] & dimension_mask[None, :]
    query = tl.load(
        q_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=matrix_mask,
        other=0.0,
    )
    output = tl.load(
        output_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=matrix_mask,
        other=0.0,
    )
    grad_output = tl.load(
        grad_output_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=matrix_mask,
        other=0.0,
    )
    lse = tl.load(
        lse_ptr + batch * stride_leb + head * stride_leh + offsets_m * stride_les,
        mask=query_mask,
        other=0.0,
    )
    delta = tl.sum(grad_output * output, axis=1)
    grad_query = tl.load(
        grad_q_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=matrix_mask,
        other=0.0,
    ).to(tl.float32)
    offsets_n_base = tl.arange(0, BLOCK_N)
    for key_tile in tl.range(0, NUM_KEY_TILES):
        offsets_n = key_tile * BLOCK_N + offsets_n_base
        key_mask = offsets_n < N_BLOCKS
        compressed_values = tl.load(
            compressed_kv_ptr
            + batch * stride_cb
            + head * stride_ch
            + offsets_n[:, None] * stride_cn
            + offsets_d[None, :] * stride_cd,
            mask=key_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        selected = tl.load(
            selection_mask_ptr
            + batch * stride_mb
            + offsets_m[:, None] * stride_ms
            + offsets_n[None, :] * stride_mn,
            mask=query_mask[:, None] & key_mask[None, :],
            other=0,
        )
        valid = query_mask[:, None] & key_mask[None, :] & (selected != 0)
        scores = tl.dot(query, tl.trans(compressed_values), input_precision="tf32x3") * SCALE
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(
            grad_output,
            tl.trans(compressed_values),
            input_precision="tf32x3",
        )
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_query += (
            tl.dot(
                grad_scores.to(compressed_values.dtype),
                compressed_values,
                input_precision="tf32x3",
            )
            * SCALE
        )

    tl.store(
        grad_q_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        grad_query,
        mask=matrix_mask,
    )


@triton.jit
def _local_attention_bwd_dkv(
    q_ptr,
    local_kv_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_local_kv_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_ls: tl.constexpr,
    stride_ld: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_les: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H
    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, BLOCK_D)
    key_mask = offsets_n < S
    dimension_mask = offsets_d < D
    local_values = tl.load(
        local_kv_ptr
        + batch * stride_lb
        + head * stride_lh
        + offsets_n[:, None] * stride_ls
        + offsets_d[None, :] * stride_ld,
        mask=key_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )
    grad_values = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    first_query = key_block * BLOCK_N
    offsets_m_base = tl.arange(0, BLOCK_M)
    for query_tile in tl.range(0, NUM_QUERY_TILES):
        offsets_m = first_query + query_tile * BLOCK_M + offsets_m_base
        query_mask = offsets_m < S
        matrix_mask = query_mask[:, None] & dimension_mask[None, :]
        query = tl.load(
            q_ptr
            + batch * stride_qb
            + head * stride_qh
            + offsets_m[:, None] * stride_qs
            + offsets_d[None, :] * stride_qd,
            mask=matrix_mask,
            other=0.0,
        )
        output = tl.load(
            output_ptr
            + batch * stride_qb
            + head * stride_qh
            + offsets_m[:, None] * stride_qs
            + offsets_d[None, :] * stride_qd,
            mask=matrix_mask,
            other=0.0,
        )
        grad_output = tl.load(
            grad_output_ptr
            + batch * stride_qb
            + head * stride_qh
            + offsets_m[:, None] * stride_qs
            + offsets_d[None, :] * stride_qd,
            mask=matrix_mask,
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + batch * stride_leb + head * stride_leh + offsets_m * stride_les,
            mask=query_mask,
            other=0.0,
        )
        delta = tl.sum(grad_output * output, axis=1)
        valid = (
            query_mask[:, None]
            & key_mask[None, :]
            & (offsets_n[None, :] <= offsets_m[:, None])
            & (offsets_n[None, :] >= offsets_m[:, None] - WINDOW + 1)
        )
        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(
            grad_output,
            tl.trans(local_values),
            input_precision="tf32x3",
        )
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_values += tl.dot(
            tl.trans(probabilities.to(grad_output.dtype)),
            grad_output,
            input_precision="tf32x3",
        )
        grad_values += (
            tl.dot(
                tl.trans(grad_scores.to(query.dtype)),
                query,
                input_precision="tf32x3",
            )
            * SCALE
        )

    tl.store(
        grad_local_kv_ptr
        + batch * stride_qb
        + head * stride_qh
        + offsets_n[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        grad_values,
        mask=key_mask[:, None] & dimension_mask[None, :],
    )


@triton.jit
def _compressed_attention_bwd_dkv(
    q_ptr,
    compressed_kv_ptr,
    selection_mask_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_compressed_kv_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_ch: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_cd: tl.constexpr,
    stride_mb: tl.constexpr,
    stride_ms: tl.constexpr,
    stride_mn: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_les: tl.constexpr,
    stride_gb: tl.constexpr,
    stride_gh: tl.constexpr,
    stride_gn: tl.constexpr,
    stride_gd: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    SCALE: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H
    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_d = tl.arange(0, BLOCK_D)
    key_mask = offsets_n < N_BLOCKS
    dimension_mask = offsets_d < D
    compressed_values = tl.load(
        compressed_kv_ptr
        + batch * stride_cb
        + head * stride_ch
        + offsets_n[:, None] * stride_cn
        + offsets_d[None, :] * stride_cd,
        mask=key_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )
    grad_values = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    offsets_m_base = tl.arange(0, BLOCK_M)
    for query_tile in tl.range(0, NUM_QUERY_TILES):
        offsets_m = query_tile * BLOCK_M + offsets_m_base
        query_mask = offsets_m < S
        matrix_mask = query_mask[:, None] & dimension_mask[None, :]
        query = tl.load(
            q_ptr
            + batch * stride_qb
            + head * stride_qh
            + offsets_m[:, None] * stride_qs
            + offsets_d[None, :] * stride_qd,
            mask=matrix_mask,
            other=0.0,
        )
        output = tl.load(
            output_ptr
            + batch * stride_qb
            + head * stride_qh
            + offsets_m[:, None] * stride_qs
            + offsets_d[None, :] * stride_qd,
            mask=matrix_mask,
            other=0.0,
        )
        grad_output = tl.load(
            grad_output_ptr
            + batch * stride_qb
            + head * stride_qh
            + offsets_m[:, None] * stride_qs
            + offsets_d[None, :] * stride_qd,
            mask=matrix_mask,
            other=0.0,
        )
        lse = tl.load(
            lse_ptr + batch * stride_leb + head * stride_leh + offsets_m * stride_les,
            mask=query_mask,
            other=0.0,
        )
        delta = tl.sum(grad_output * output, axis=1)
        selected = tl.load(
            selection_mask_ptr
            + batch * stride_mb
            + offsets_m[:, None] * stride_ms
            + offsets_n[None, :] * stride_mn,
            mask=query_mask[:, None] & key_mask[None, :],
            other=0,
        )
        valid = query_mask[:, None] & key_mask[None, :] & (selected != 0)
        scores = tl.dot(query, tl.trans(compressed_values), input_precision="tf32x3") * SCALE
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(
            grad_output,
            tl.trans(compressed_values),
            input_precision="tf32x3",
        )
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_values += tl.dot(
            tl.trans(probabilities.to(grad_output.dtype)),
            grad_output,
            input_precision="tf32x3",
        )
        grad_values += (
            tl.dot(
                tl.trans(grad_scores.to(query.dtype)),
                query,
                input_precision="tf32x3",
            )
            * SCALE
        )

    tl.store(
        grad_compressed_kv_ptr
        + batch * stride_gb
        + head * stride_gh
        + offsets_n[:, None] * stride_gn
        + offsets_d[None, :] * stride_gd,
        grad_values,
        mask=key_mask[:, None] & dimension_mask[None, :],
    )


def _launch_selected_attention(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    local_kv: torch.Tensor,
    topk_blocks: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
    *,
    _return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch, heads, sequence_length, head_dim = query.shape
    topk = topk_blocks.shape[-1]
    block_m = 64
    block_n = 128
    block_d = max(16, triton.next_power_of_2(head_dim))
    num_local_tiles = (
        triton.cdiv(sliding_window_size + block_m - 1, block_n) if sliding_window_size else 0
    )
    output = torch.empty_like(query)
    local_lse = torch.empty(
        batch,
        heads,
        sequence_length,
        device=query.device,
        dtype=torch.float32,
    )
    _tiled_selected_attention_fwd[(triton.cdiv(sequence_length, block_m), batch * heads)](
        query,
        compressed_kv,
        local_kv,
        topk_blocks,
        attention_sink,
        output,
        local_lse,
        *query.stride(),
        *compressed_kv.stride(),
        *local_kv.stride(),
        *topk_blocks.stride(),
        *output.stride(),
        *local_lse.stride(),
        H=heads,
        S=sequence_length,
        D=head_dim,
        N_BLOCKS=compressed_kv.shape[-2],
        COMPRESSION_RATE=compression_rate,
        TOPK=0,
        WINDOW=sliding_window_size,
        SCALE=1.0 / math.sqrt(head_dim),
        NUM_LOCAL_TILES=num_local_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
    )
    if topk:
        block_k = max(16, triton.next_power_of_2(topk))
        _selected_compressed_attention_merge_fwd[(sequence_length, batch * heads)](
            query,
            compressed_kv,
            topk_blocks,
            output,
            local_lse,
            output,
            *query.stride(),
            *compressed_kv.stride(),
            *topk_blocks.stride(),
            *output.stride(),
            *local_lse.stride(),
            H=heads,
            S=sequence_length,
            D=head_dim,
            N_BLOCKS=compressed_kv.shape[-2],
            COMPRESSION_RATE=compression_rate,
            TOPK=topk,
            SCALE=1.0 / math.sqrt(head_dim),
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
        )
    if _return_lse:
        return output, local_lse
    return output


def _launch_selected_attention_backward(
    query: torch.Tensor,
    compressed_kv: torch.Tensor,
    local_kv: torch.Tensor,
    topk_blocks: torch.Tensor,
    attention_sink: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    compression_rate: int,
    sliding_window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run sparse selected-attention backward kernels."""
    batch, heads, sequence_length, head_dim = query.shape
    num_blocks = compressed_kv.shape[-2]
    topk = topk_blocks.shape[-1]
    block_m = 64
    block_n = 32
    block_d = max(16, triton.next_power_of_2(head_dim))
    scale = 1.0 / math.sqrt(head_dim)
    grad_output = grad_output.contiguous()
    grad_query = torch.empty_like(query)
    grad_local_kv = torch.empty(
        batch,
        heads,
        sequence_length,
        head_dim,
        device=query.device,
        dtype=query.dtype,
    )
    grad_compressed_kv = torch.empty(
        batch,
        heads,
        num_blocks,
        head_dim,
        device=query.device,
        dtype=query.dtype,
    )
    grad_sink_fp32 = torch.zeros(
        heads,
        device=query.device,
        dtype=torch.float32,
    )

    num_local_key_tiles = (
        triton.cdiv(sliding_window_size + block_m - 1, block_n) if sliding_window_size else 0
    )
    num_local_query_tiles = (
        triton.cdiv(sliding_window_size + block_n - 1, block_m) if sliding_window_size else 0
    )
    query_grid = (triton.cdiv(sequence_length, block_m), batch * heads)
    _local_attention_bwd_dq[query_grid](
        query,
        local_kv,
        output,
        grad_output,
        lse,
        attention_sink,
        grad_query,
        grad_sink_fp32,
        *query.stride(),
        *local_kv.stride(),
        *lse.stride(),
        H=heads,
        S=sequence_length,
        D=head_dim,
        WINDOW=sliding_window_size,
        SCALE=scale,
        NUM_KEY_TILES=num_local_key_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
    )
    _local_attention_bwd_dkv[(triton.cdiv(sequence_length, block_n), batch * heads)](
        query,
        local_kv,
        output,
        grad_output,
        lse,
        grad_local_kv,
        *query.stride(),
        *local_kv.stride(),
        *lse.stride(),
        H=heads,
        S=sequence_length,
        D=head_dim,
        WINDOW=sliding_window_size,
        SCALE=scale,
        NUM_QUERY_TILES=num_local_query_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
    )

    if topk:
        selection_mask = torch.zeros(
            batch,
            sequence_length,
            num_blocks,
            device=query.device,
            dtype=torch.uint8,
        )
        block_k = max(16, triton.next_power_of_2(topk))
        _build_selected_block_mask[(sequence_length, batch)](
            topk_blocks,
            selection_mask,
            *topk_blocks.stride(),
            *selection_mask.stride(),
            S=sequence_length,
            N_BLOCKS=num_blocks,
            COMPRESSION_RATE=compression_rate,
            TOPK=topk,
            BLOCK_K=block_k,
            num_warps=4,
        )
        num_compressed_key_tiles = triton.cdiv(num_blocks, block_n)
        _compressed_attention_bwd_dq[query_grid](
            query,
            compressed_kv,
            selection_mask,
            output,
            grad_output,
            lse,
            grad_query,
            *query.stride(),
            *compressed_kv.stride(),
            *selection_mask.stride(),
            *lse.stride(),
            H=heads,
            S=sequence_length,
            D=head_dim,
            N_BLOCKS=num_blocks,
            SCALE=scale,
            NUM_KEY_TILES=num_compressed_key_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=8,
        )
        _compressed_attention_bwd_dkv[(num_compressed_key_tiles, batch * heads)](
            query,
            compressed_kv,
            selection_mask,
            output,
            grad_output,
            lse,
            grad_compressed_kv,
            *query.stride(),
            *compressed_kv.stride(),
            *selection_mask.stride(),
            *lse.stride(),
            *grad_compressed_kv.stride(),
            H=heads,
            S=sequence_length,
            D=head_dim,
            N_BLOCKS=num_blocks,
            SCALE=scale,
            NUM_QUERY_TILES=triton.cdiv(sequence_length, block_m),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=8,
        )
    else:
        grad_compressed_kv.zero_()

    return (
        grad_query,
        grad_compressed_kv,
        grad_local_kv,
        grad_sink_fp32.to(attention_sink.dtype),
    )


class _SelectedAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        compressed_kv: torch.Tensor,
        local_kv: torch.Tensor,
        topk_blocks: torch.Tensor,
        attention_sink: torch.Tensor,
        compression_rate: int,
        sliding_window_size: int,
    ) -> torch.Tensor:
        output, lse = _launch_selected_attention(
            query,
            compressed_kv,
            local_kv,
            topk_blocks,
            attention_sink,
            compression_rate,
            sliding_window_size,
            _return_lse=True,
        )
        ctx.save_for_backward(
            query,
            compressed_kv,
            local_kv,
            topk_blocks,
            attention_sink,
            output,
            lse,
        )
        ctx.compression_rate = compression_rate
        ctx.sliding_window_size = sliding_window_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        query, compressed_kv, local_kv, topk_blocks, attention_sink, output, lse = (
            ctx.saved_tensors
        )
        gradients = _launch_selected_attention_backward(
            query,
            compressed_kv,
            local_kv,
            topk_blocks,
            attention_sink,
            output,
            lse,
            grad_output,
            ctx.compression_rate,
            ctx.sliding_window_size,
        )
        grad_query, grad_compressed_kv, grad_local_kv, grad_sink = gradients
        return (
            grad_query,
            grad_compressed_kv,
            grad_local_kv,
            None,
            grad_sink,
            None,
            None,
        )


def _validate_backend_constraints(
    tensors: tuple[tuple[str, torch.Tensor], ...],
    *,
    backend_name: str = "Triton",
    max_head_dim: int = 256,
    supported_dtypes: tuple[torch.dtype, ...] = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ),
) -> None:
    query = tensors[0][1]
    if query.device.type != "cuda":
        raise ValueError(
            f"The {backend_name} compressed sparse attention backend requires CUDA tensors."
        )
    if query.dtype not in supported_dtypes:
        dtype_names = ", ".join(str(dtype).removeprefix("torch.") for dtype in supported_dtypes)
        raise TypeError(f"The {backend_name} backend supports {dtype_names} tensors.")
    for name, tensor in tensors:
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous for the {backend_name} backend.")
    head_dim = query.shape[-1]
    if head_dim > max_head_dim:
        raise ValueError(
            f"The {backend_name} backend currently supports head_dim <= {max_head_dim}."
        )


def _prepare_attention_inputs(
    Q: torch.Tensor,
    Q_I: torch.Tensor,
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    W_I: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
    *,
    backend_name: str = "Triton",
    max_head_dim: int = 256,
    supported_dtypes: tuple[torch.dtype, ...] = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ),
    expand_attention_kv: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and prepare the four inputs consumed by the selected-attention kernel."""
    tensors = (
        ("Q", Q),
        ("Q_I", Q_I),
        ("KV", KV),
        ("C_a", C_a),
        ("C_b", C_b),
        ("Z_a", Z_a),
        ("Z_b", Z_b),
        ("B_a", B_a),
        ("B_b", B_b),
        ("W_I", W_I),
        ("K_Ia", K_Ia),
        ("K_Ib", K_Ib),
        ("Z_Ia", Z_Ia),
        ("Z_Ib", Z_Ib),
        ("B_Ia", B_Ia),
        ("B_Ib", B_Ib),
        ("KV_norm_weight", KV_norm_weight),
        ("compressed_indices_norm_weight", compressed_indices_norm_weight),
        ("compressed_kv_norm_weight", compressed_kv_norm_weight),
        ("attention_sink", attention_sink),
    )
    _validate_backend_constraints(
        tensors,
        backend_name=backend_name,
        max_head_dim=max_head_dim,
        supported_dtypes=supported_dtypes,
    )

    batch, heads, sequence_length, head_dim = Q.shape
    index_heads, index_dim = Q_I.shape[1], Q_I.shape[3]
    kv_inputs = (KV, C_a, C_b, Z_a, Z_b)
    if share_kv and len({tensor.shape[1] for tensor in kv_inputs}) > 1:
        KV, C_a, C_b, Z_a, Z_b = (tensor.expand(-1, heads, -1, -1) for tensor in kv_inputs)
    index_inputs = (K_Ia, K_Ib, Z_Ia, Z_Ib)
    if share_kv and len({tensor.shape[1] for tensor in index_inputs}) > 1:
        K_Ia, K_Ib, Z_Ia, Z_Ib = (
            tensor.expand(-1, index_heads, -1, -1) for tensor in index_inputs
        )
    compressed_kv = _compress(C_a, C_b, Z_a, Z_b, B_a, B_b, compression_rate)
    compressed_indices = _compress(K_Ia, K_Ib, Z_Ia, Z_Ib, B_Ia, B_Ib, compression_rate)
    num_blocks = compressed_kv.shape[-2]

    query = _apply_rope(Q, rope_dims)
    index_query = _apply_rope(Q_I, rope_dims)
    local_kv = F.rms_norm(KV, (head_dim,), weight=KV_norm_weight)
    local_kv = _apply_rope(local_kv, rope_dims)

    compressed_positions = torch.arange(num_blocks, device=Q.device) * compression_rate
    compressed_indices = F.rms_norm(
        compressed_indices, (index_dim,), weight=compressed_indices_norm_weight
    )
    compressed_indices = _apply_rope(compressed_indices, rope_dims, positions=compressed_positions)
    compressed_kv = F.rms_norm(compressed_kv, (head_dim,), weight=compressed_kv_norm_weight)
    compressed_kv = _apply_rope(compressed_kv, rope_dims, positions=compressed_positions)

    if compressed_indices.shape[1] == 1 and index_heads != 1:
        compressed_indices = compressed_indices.expand(-1, index_heads, -1, -1)
    scores = F.relu(index_query @ compressed_indices.transpose(-2, -1)) / math.sqrt(
        index_dim * index_heads
    )
    scores = (scores * W_I.permute(0, 2, 1).unsqueeze(-1)).sum(dim=1)
    query_positions = torch.arange(sequence_length, device=Q.device)
    block_positions = torch.arange(num_blocks, device=Q.device)
    completed_blocks = (query_positions + 1) // compression_rate
    scores.masked_fill_(
        block_positions[None, None, :] >= completed_blocks[None, :, None],
        float("-inf"),
    )
    topk = min(num_topk_blocks, num_blocks)
    topk_blocks = torch.topk(scores, k=topk, dim=-1).indices

    if expand_attention_kv and local_kv.shape[1] == 1 and heads != 1:
        local_kv = local_kv.expand(-1, heads, -1, -1)
    if expand_attention_kv and compressed_kv.shape[1] == 1 and heads != 1:
        compressed_kv = compressed_kv.expand(-1, heads, -1, -1)
    return query, compressed_kv, local_kv, topk_blocks


def compressed_sparse_attention(
    Q: torch.Tensor,
    Q_I: torch.Tensor,
    KV: torch.Tensor,
    C_a: torch.Tensor,
    C_b: torch.Tensor,
    Z_a: torch.Tensor,
    Z_b: torch.Tensor,
    B_a: torch.Tensor,
    B_b: torch.Tensor,
    W_I: torch.Tensor,
    K_Ia: torch.Tensor,
    K_Ib: torch.Tensor,
    Z_Ia: torch.Tensor,
    Z_Ib: torch.Tensor,
    B_Ia: torch.Tensor,
    B_Ib: torch.Tensor,
    KV_norm_weight: torch.Tensor,
    compressed_indices_norm_weight: torch.Tensor,
    compressed_kv_norm_weight: torch.Tensor,
    attention_sink: torch.Tensor,
    compression_rate: int,
    num_topk_blocks: int,
    sliding_window_size: int,
    rope_dims: int,
    share_kv: bool,
) -> torch.Tensor:
    """Run the hybrid Triton CSA implementation."""
    query, compressed_kv, local_kv, topk_blocks = _prepare_attention_inputs(
        Q,
        Q_I,
        KV,
        C_a,
        C_b,
        Z_a,
        Z_b,
        B_a,
        B_b,
        W_I,
        K_Ia,
        K_Ib,
        Z_Ia,
        Z_Ib,
        B_Ia,
        B_Ib,
        KV_norm_weight,
        compressed_indices_norm_weight,
        compressed_kv_norm_weight,
        attention_sink,
        compression_rate,
        num_topk_blocks,
        sliding_window_size,
        rope_dims,
        share_kv,
    )
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, compressed_kv, local_kv, attention_sink)
    ):
        output = _SelectedAttention.apply(
            query,
            compressed_kv,
            local_kv,
            topk_blocks,
            attention_sink,
            compression_rate,
            sliding_window_size,
        )
    else:
        output = _launch_selected_attention(
            query,
            compressed_kv,
            local_kv,
            topk_blocks,
            attention_sink,
            compression_rate,
            sliding_window_size,
        )
    return _apply_rope(output, rope_dims, inverse=True)


__all__ = ["compressed_sparse_attention"]
