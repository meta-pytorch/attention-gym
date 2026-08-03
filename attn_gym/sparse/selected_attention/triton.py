"""Triton backend for selected attention."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_attention_fwd(
    q_ptr,
    index_kv_ptr,
    local_kv_ptr,
    indices_ptr,
    doc_ids_ptr,
    sink_ptr,
    out_ptr,
    lse_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_ib: tl.constexpr,
    stride_ih: tl.constexpr,
    stride_in: tl.constexpr,
    stride_id: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_ls: tl.constexpr,
    stride_ld: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xs: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_db: tl.constexpr,
    stride_ds: tl.constexpr,
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
    INDEX_SEQ_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    NUM_LOCAL_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Forward pass: online softmax over selected index blocks + local sliding window."""
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

    # Load doc_ids for query positions if needed
    if HAS_DOC_IDS:
        query_doc_ids = tl.load(
            doc_ids_ptr + batch * stride_db + offsets_m * stride_ds,
            mask=query_mask,
            other=-1,
        )

    sink = tl.load(sink_ptr + head).to(tl.float32)
    running_max = tl.full((BLOCK_M,), sink, tl.float32)
    running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Phase 1: Selected index blocks (each query selects different positions)
    for selected_slot in tl.static_range(0, TOPK):
        selected_idx = tl.load(
            indices_ptr
            + batch * stride_xb
            + offsets_m * stride_xs
            + selected_slot * stride_xk,
            mask=query_mask,
            other=0,
        )
        valid = query_mask & (selected_idx >= 0) & (selected_idx < INDEX_SEQ_LEN)
        index_value = tl.load(
            index_kv_ptr
            + batch * stride_ib
            + head * stride_ih
            + selected_idx[:, None] * stride_in
            + offsets_d[None, :] * stride_id,
            mask=valid[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        logit = tl.sum(query * index_value, axis=1) * SCALE
        logit = tl.where(valid, logit, -float("inf"))
        new_max = tl.maximum(running_max, logit)
        alpha = tl.exp(running_max - new_max)
        probability = tl.exp(logit - new_max)
        accumulator = accumulator * alpha[:, None] + probability[:, None] * index_value
        running_sum = running_sum * alpha + probability
        running_max = new_max

    # Phase 2: Local sliding window (tensor-core-friendly tiles)
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
        if HAS_DOC_IDS:
            key_doc_ids = tl.load(
                doc_ids_ptr + batch * stride_db + offsets_n * stride_ds,
                mask=local_mask,
                other=-2,
            )
            doc_mask = query_doc_ids[:, None] == key_doc_ids[None, :]
            causal_window_mask = causal_window_mask & doc_mask

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
def _selected_attention_bwd_dq(
    q_ptr,
    index_kv_ptr,
    local_kv_ptr,
    indices_ptr,
    doc_ids_ptr,
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
    stride_ib: tl.constexpr,
    stride_ih: tl.constexpr,
    stride_in: tl.constexpr,
    stride_id: tl.constexpr,
    stride_lb: tl.constexpr,
    stride_lh: tl.constexpr,
    stride_ls: tl.constexpr,
    stride_ld: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xs: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_db: tl.constexpr,
    stride_ds: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_les: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    INDEX_SEQ_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    NUM_LOCAL_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward for dQ: iterate over index slots + local window tiles."""
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

    if HAS_DOC_IDS:
        query_doc_ids = tl.load(
            doc_ids_ptr + batch * stride_db + offsets_m * stride_ds,
            mask=query_mask,
            other=-1,
        )

    # Index branch gradient
    for selected_slot in tl.range(0, TOPK):
        selected_idx = tl.load(
            indices_ptr
            + batch * stride_xb
            + offsets_m * stride_xs
            + selected_slot * stride_xk,
            mask=query_mask,
            other=0,
        )
        valid = query_mask & (selected_idx >= 0) & (selected_idx < INDEX_SEQ_LEN)
        index_value = tl.load(
            index_kv_ptr
            + batch * stride_ib
            + head * stride_ih
            + selected_idx[:, None] * stride_in
            + offsets_d[None, :] * stride_id,
            mask=valid[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        scores = tl.sum(query * index_value, axis=1) * SCALE
        probabilities = tl.where(valid, tl.exp(scores - lse), 0.0)
        grad_probs = tl.sum(grad_output * index_value, axis=1)
        grad_scores = probabilities * (grad_probs - delta)
        grad_query += grad_scores[:, None] * index_value * SCALE

    # Local window gradient
    first_key = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for key_tile in tl.range(0, NUM_LOCAL_TILES):
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
        if HAS_DOC_IDS:
            key_doc_ids = tl.load(
                doc_ids_ptr + batch * stride_db + offsets_n * stride_ds,
                mask=key_mask,
                other=-2,
            )
            doc_mask = query_doc_ids[:, None] == key_doc_ids[None, :]
            valid = valid & doc_mask

        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(
            grad_output, tl.trans(local_values), input_precision="tf32x3"
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

    # Sink gradient
    sink = tl.load(sink_ptr + head)
    sink_probability = tl.exp(sink - lse)
    sink_gradient = tl.where(query_mask, -sink_probability * delta, 0.0)
    tl.atomic_add(grad_sink_ptr + head, tl.sum(sink_gradient, axis=0))


@triton.jit
def _selected_attention_bwd_dlocal_kv(
    q_ptr,
    local_kv_ptr,
    doc_ids_ptr,
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
    stride_db: tl.constexpr,
    stride_ds: tl.constexpr,
    stride_leb: tl.constexpr,
    stride_leh: tl.constexpr,
    stride_les: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    NUM_QUERY_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward for dKV (local sliding window)."""
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

    if HAS_DOC_IDS:
        key_doc_ids = tl.load(
            doc_ids_ptr + batch * stride_db + offsets_n * stride_ds,
            mask=key_mask,
            other=-2,
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
        if HAS_DOC_IDS:
            query_doc_ids = tl.load(
                doc_ids_ptr + batch * stride_db + offsets_m * stride_ds,
                mask=query_mask,
                other=-1,
            )
            doc_mask = query_doc_ids[:, None] == key_doc_ids[None, :]
            valid = valid & doc_mask

        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(
            grad_output, tl.trans(local_values), input_precision="tf32x3"
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
        + batch * stride_lb
        + head * stride_lh
        + offsets_n[:, None] * stride_ls
        + offsets_d[None, :] * stride_ld,
        grad_values,
        mask=key_mask[:, None] & dimension_mask[None, :],
    )


@triton.jit
def _selected_attention_bwd_dindex_kv(
    q_ptr,
    index_kv_ptr,
    indices_ptr,
    selected_queries_ptr,
    block_offsets_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_index_kv_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_ib: tl.constexpr,
    stride_ih: tl.constexpr,
    stride_in: tl.constexpr,
    stride_id: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xs: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_sqb: tl.constexpr,
    stride_sqe: tl.constexpr,
    stride_bob: tl.constexpr,
    stride_bon: tl.constexpr,
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
    INDEX_SEQ_LEN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Backward for dindex_kv: use inverted index to gather relevant queries."""
    selected_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m_base = tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    dot_rows = tl.arange(0, 16)
    dimension_mask = offsets_d < D

    index_value = tl.load(
        index_kv_ptr
        + batch * stride_ib
        + head * stride_ih
        + selected_block * stride_in
        + offsets_d * stride_id,
        mask=dimension_mask,
        other=0.0,
    )
    dot_value = tl.where(
        dot_rows[None, :] == 0,
        index_value[:, None],
        0.0,
    )
    grad_value = tl.zeros((16, BLOCK_D), tl.float32)
    entry_start = tl.load(
        block_offsets_ptr + batch * stride_bob + selected_block * stride_bon
    )
    entry_end = tl.load(
        block_offsets_ptr + batch * stride_bob + (selected_block + 1) * stride_bon
    )

    for entry_tile in tl.range(entry_start, entry_end, BLOCK_M):
        entry_offsets = entry_tile + offsets_m_base
        entry_mask = entry_offsets < entry_end
        query_positions = tl.load(
            selected_queries_ptr + batch * stride_sqb + entry_offsets * stride_sqe,
            mask=entry_mask,
            other=0,
        )
        query_mask = entry_mask & (query_positions >= 0) & (query_positions < S)
        matrix_mask = query_mask[:, None] & dimension_mask[None, :]
        query_offsets = (
            batch * stride_qb
            + head * stride_qh
            + query_positions[:, None] * stride_qs
            + offsets_d[None, :] * stride_qd
        )
        query = tl.load(q_ptr + query_offsets, mask=matrix_mask, other=0.0)
        output = tl.load(output_ptr + query_offsets, mask=matrix_mask, other=0.0)
        grad_output = tl.load(
            grad_output_ptr + query_offsets, mask=matrix_mask, other=0.0
        )
        lse = tl.load(
            lse_ptr + batch * stride_leb + head * stride_leh + query_positions * stride_les,
            mask=query_mask,
            other=0.0,
        )
        score_tile = tl.dot(query, dot_value, input_precision="tf32x3")
        scores = tl.sum(score_tile * (dot_rows[None, :] == 0), axis=1) * SCALE
        grad_probability_tile = tl.dot(grad_output, dot_value, input_precision="tf32x3")
        grad_probabilities = tl.sum(
            grad_probability_tile * (dot_rows[None, :] == 0), axis=1
        )
        delta = tl.sum(grad_output * output, axis=1)
        probabilities = tl.where(query_mask, tl.exp(scores - lse), 0.0)
        grad_scores = probabilities * (grad_probabilities - delta)
        combined_weights = tl.cat(probabilities, grad_scores * SCALE, dim=0)
        combined_values = tl.cat(grad_output, query, dim=0)
        combined_weight_tile = tl.where(
            dot_rows[:, None] == 0,
            combined_weights[None, :],
            0.0,
        )
        grad_value = tl.dot(
            combined_weight_tile.to(query.dtype),
            combined_values,
            acc=grad_value,
            input_precision="tf32x3",
        )

    tl.store(
        grad_index_kv_ptr
        + batch * stride_gb
        + head * stride_gh
        + selected_block * stride_gn
        + offsets_d * stride_gd,
        tl.sum(grad_value * (dot_rows[:, None] == 0), axis=0),
        mask=dimension_mask,
    )


def _launch_forward(
    Q: torch.Tensor,
    index_kv: torch.Tensor,
    local_kv: torch.Tensor,
    indices: torch.Tensor,
    attention_sink: torch.Tensor,
    doc_ids: torch.Tensor | None,
    sliding_window_size: int,
    *,
    _return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, head_dim = Q.shape
    topk = indices.shape[-1]
    index_seq_len = index_kv.shape[2]
    block_m = 64
    block_n = 128
    block_d = max(16, triton.next_power_of_2(head_dim))
    num_local_tiles = (
        triton.cdiv(sliding_window_size + block_m - 1, block_n) if sliding_window_size else 0
    )

    output = torch.empty_like(Q)
    lse = torch.empty(batch, heads, seq_len, device=Q.device, dtype=torch.float32)

    has_doc_ids = doc_ids is not None
    if doc_ids is None:
        doc_ids = torch.empty(batch, seq_len, device=Q.device, dtype=torch.int32)
        doc_strides = (0, 0)
    else:
        doc_strides = doc_ids.stride()

    _selected_attention_fwd[(triton.cdiv(seq_len, block_m), batch * heads)](
        Q,
        index_kv,
        local_kv,
        indices,
        doc_ids,
        attention_sink,
        output,
        lse,
        *Q.stride(),
        *index_kv.stride(),
        *local_kv.stride(),
        *indices.stride(),
        *doc_strides,
        *output.stride(),
        *lse.stride(),
        H=heads,
        S=seq_len,
        D=head_dim,
        INDEX_SEQ_LEN=index_seq_len,
        TOPK=topk,
        WINDOW=sliding_window_size,
        SCALE=1.0 / math.sqrt(head_dim),
        HAS_DOC_IDS=has_doc_ids,
        NUM_LOCAL_TILES=num_local_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
    )
    if _return_lse:
        return output, lse
    return output


def _build_index_query_map(
    indices: torch.Tensor,
    index_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert query-major indices into block-major query lists (CSR format)."""
    batch, seq_len, topk = indices.shape
    if topk == 0:
        selected_queries = torch.empty(batch, 0, device=indices.device, dtype=torch.int32)
        block_offsets = torch.zeros(
            batch, index_seq_len + 1, device=indices.device, dtype=torch.int32
        )
        return selected_queries, block_offsets

    flat_indices = indices.flatten(1)
    sorted_indices, sorted_entries = torch.sort(flat_indices, dim=-1)
    selected_queries = torch.div(sorted_entries, topk, rounding_mode="floor").to(torch.int32)
    block_ids = torch.arange(
        index_seq_len + 1, device=indices.device, dtype=sorted_indices.dtype
    )
    block_ids = block_ids.unsqueeze(0).expand(batch, -1).contiguous()
    block_offsets = torch.searchsorted(sorted_indices, block_ids).to(torch.int32)
    return selected_queries.contiguous(), block_offsets.contiguous()


def _launch_backward(
    Q: torch.Tensor,
    index_kv: torch.Tensor,
    local_kv: torch.Tensor,
    indices: torch.Tensor,
    selected_queries: torch.Tensor,
    block_offsets: torch.Tensor,
    attention_sink: torch.Tensor,
    doc_ids: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    sliding_window_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch backward kernels for selected attention."""
    batch, heads, seq_len, head_dim = Q.shape
    index_seq_len = index_kv.shape[2]
    topk = indices.shape[-1]
    block_m = 64
    block_n = 32
    block_d = max(16, triton.next_power_of_2(head_dim))
    scale = 1.0 / math.sqrt(head_dim)
    grad_output = grad_output.contiguous()

    grad_Q = torch.empty_like(Q)
    grad_local_kv = torch.empty_like(local_kv)
    grad_sink_fp32 = torch.zeros(heads, device=Q.device, dtype=torch.float32)

    has_doc_ids = doc_ids is not None
    if doc_ids is None:
        doc_ids_t = torch.empty(batch, seq_len, device=Q.device, dtype=torch.int32)
        doc_strides = (0, 0)
    else:
        doc_ids_t = doc_ids
        doc_strides = doc_ids.stride()

    num_local_key_tiles = (
        triton.cdiv(sliding_window_size + block_m - 1, block_n) if sliding_window_size else 0
    )
    num_local_query_tiles = (
        triton.cdiv(sliding_window_size + block_n - 1, block_m) if sliding_window_size else 0
    )

    # dQ kernel
    _selected_attention_bwd_dq[(triton.cdiv(seq_len, block_m), batch * heads)](
        Q,
        index_kv,
        local_kv,
        indices,
        doc_ids_t,
        output,
        grad_output,
        lse,
        attention_sink,
        grad_Q,
        grad_sink_fp32,
        *Q.stride(),
        *index_kv.stride(),
        *local_kv.stride(),
        *indices.stride(),
        *doc_strides,
        *lse.stride(),
        H=heads,
        S=seq_len,
        D=head_dim,
        INDEX_SEQ_LEN=index_seq_len,
        TOPK=topk,
        WINDOW=sliding_window_size,
        SCALE=scale,
        HAS_DOC_IDS=has_doc_ids,
        NUM_LOCAL_TILES=num_local_key_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
    )

    # dKV (local) kernel
    _selected_attention_bwd_dlocal_kv[(triton.cdiv(seq_len, block_n), batch * heads)](
        Q,
        local_kv,
        doc_ids_t,
        output,
        grad_output,
        lse,
        grad_local_kv,
        *Q.stride(),
        *local_kv.stride(),
        *doc_strides,
        *lse.stride(),
        H=heads,
        S=seq_len,
        D=head_dim,
        WINDOW=sliding_window_size,
        SCALE=scale,
        HAS_DOC_IDS=has_doc_ids,
        NUM_QUERY_TILES=num_local_query_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=8,
    )

    # dindex_kv kernel
    if topk > 0:
        grad_index_kv = torch.empty_like(index_kv)
        compressed_block_m = 32 if head_dim <= 128 else 16
        _selected_attention_bwd_dindex_kv[(index_seq_len, batch * heads)](
            Q,
            index_kv,
            indices,
            selected_queries,
            block_offsets,
            output,
            grad_output,
            lse,
            grad_index_kv,
            *Q.stride(),
            *index_kv.stride(),
            *indices.stride(),
            *selected_queries.stride(),
            *block_offsets.stride(),
            *lse.stride(),
            *grad_index_kv.stride(),
            H=heads,
            S=seq_len,
            D=head_dim,
            INDEX_SEQ_LEN=index_seq_len,
            SCALE=scale,
            BLOCK_M=compressed_block_m,
            BLOCK_D=block_d,
            num_warps=8,
        )
    else:
        grad_index_kv = torch.zeros_like(index_kv)

    return grad_Q, grad_index_kv, grad_local_kv, grad_sink_fp32.to(attention_sink.dtype)


class _SelectedAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        index_kv: torch.Tensor,
        local_kv: torch.Tensor,
        indices: torch.Tensor,
        attention_sink: torch.Tensor,
        doc_ids: torch.Tensor | None,
        sliding_window_size: int,
    ) -> torch.Tensor:
        output, lse = _launch_forward(
            Q, index_kv, local_kv, indices, attention_sink, doc_ids,
            sliding_window_size, _return_lse=True,
        )
        selected_queries, block_offsets = _build_index_query_map(
            indices, index_kv.shape[2]
        )
        ctx.save_for_backward(
            Q, index_kv, local_kv, indices, selected_queries, block_offsets,
            attention_sink, output, lse,
        )
        ctx.doc_ids = doc_ids
        ctx.sliding_window_size = sliding_window_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            Q, index_kv, local_kv, indices, selected_queries, block_offsets,
            attention_sink, output, lse,
        ) = ctx.saved_tensors
        grad_Q, grad_index_kv, grad_local_kv, grad_sink = _launch_backward(
            Q, index_kv, local_kv, indices, selected_queries, block_offsets,
            attention_sink, ctx.doc_ids, output, lse, grad_output,
            ctx.sliding_window_size,
        )
        return grad_Q, grad_index_kv, grad_local_kv, None, grad_sink, None, None


def selected_attention(
    Q: torch.Tensor,
    KV: torch.Tensor,
    index_kv: torch.Tensor,
    indices: torch.Tensor,
    attention_sink: torch.Tensor,
    doc_ids: torch.Tensor | None,
    sliding_window_size: int,
    share_kv: bool = True,
) -> torch.Tensor:
    """Triton implementation of selected attention.

    Args:
        Q: (batch, heads, seq_len, head_dim) — queries.
        KV: (batch, 1 or heads, seq_len, head_dim) — local sliding-window key-values.
        index_kv: (batch, 1 or heads, index_seq_len, head_dim) — indexed key-values.
        indices: (batch, seq_len, topk) — which index_kv positions each query attends to.
        attention_sink: (heads,) — learned per-head sink weight.
        doc_ids: (batch, seq_len) or None — document IDs for packing isolation.
        sliding_window_size: size of the causal sliding window.
        share_kv: if True, expand single-head KV to all heads.

    Returns:
        Attention output with same shape as Q.
    """
    b, h, s, head_dim = Q.shape

    # Validate CUDA
    if Q.device.type != "cuda":
        raise ValueError("The Triton selected attention backend requires CUDA tensors.")

    # Expand shared KV heads
    if share_kv:
        KV = KV.expand(-1, h, -1, -1).contiguous()
        index_kv = index_kv.expand(-1, h, -1, -1).contiguous()
    else:
        KV = KV.contiguous()
        index_kv = index_kv.contiguous()

    Q = Q.contiguous()
    indices = indices.contiguous()
    if doc_ids is not None:
        doc_ids = doc_ids.contiguous()

    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (Q, KV, index_kv, attention_sink)
    ):
        return _SelectedAttentionFunction.apply(
            Q, index_kv, KV, indices, attention_sink, doc_ids, sliding_window_size
        )
    return _launch_forward(
        Q, index_kv, KV, indices, attention_sink, doc_ids, sliding_window_size
    )
