"""Backward kernels and launcher for Triton selected attention."""

import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import can_use_tma

from .primitives import (
    can_use_shared_kv_schedule,
    causal_window_mask,
    load_bhsd,
    load_bs,
    store_bhsd,
)
from .shared_backward import (
    _selected_attention_bwd_dq_shared,
    _selected_attention_bwd_dsparse_kv_shared,
    _selected_attention_bwd_dsparse_kv_shared_atomic,
)


@triton.jit
def _selected_attention_bwd_dq(
    query_ptr,
    sparse_kv_ptr,
    local_kv_ptr,
    kv_indices_ptr,
    doc_ids_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    attention_sink_ptr,
    grad_query_ptr,
    grad_sink_ptr,
    QUERY_STRIDES: tl.constexpr,
    SPARSE_KV_STRIDES: tl.constexpr,
    LOCAL_KV_STRIDES: tl.constexpr,
    KV_INDICES_STRIDES: tl.constexpr,
    DOC_IDS_STRIDES: tl.constexpr,
    LSE_STRIDES: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    SPARSE_SEQ_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    NUM_LOCAL_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Accumulate query and sink gradients over both attention branches."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D
    matrix_mask = query_mask[:, None] & dimension_mask[None, :]

    query = load_bhsd(query_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask)
    output = load_bhsd(output_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask)
    grad_output = load_bhsd(
        grad_output_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask
    )
    lse = tl.load(
        lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
        mask=query_mask,
        other=0.0,
    )
    delta = tl.sum(grad_output * output, axis=1)
    grad_query = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    if HAS_DOC_IDS:
        query_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_m, query_mask, -1)

    for selected_slot in tl.range(0, TOPK):
        selected_idx = tl.load(
            kv_indices_ptr
            + batch * KV_INDICES_STRIDES[0]
            + offsets_m * KV_INDICES_STRIDES[1]
            + selected_slot * KV_INDICES_STRIDES[2],
            mask=query_mask,
            other=0,
        )
        valid = query_mask & (selected_idx >= 0) & (selected_idx < SPARSE_SEQ_LEN)
        sparse_value = load_bhsd(
            sparse_kv_ptr,
            SPARSE_KV_STRIDES,
            batch,
            head,
            selected_idx,
            offsets_d,
            valid[:, None] & dimension_mask[None, :],
        )
        scores = tl.sum(query * sparse_value, axis=1) * SCALE
        probabilities = tl.where(valid, tl.exp(scores - lse), 0.0)
        grad_probs = tl.sum(grad_output * sparse_value, axis=1)
        grad_scores = probabilities * (grad_probs - delta)
        grad_query += grad_scores[:, None] * sparse_value * SCALE

    first_key = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for key_tile in tl.range(0, NUM_LOCAL_TILES):
        local_start = first_key + key_tile * BLOCK_N
        offsets_n = local_start + offsets_n_base
        key_mask = (offsets_n >= 0) & (offsets_n < S)
        local_values = load_bhsd(
            local_kv_ptr,
            LOCAL_KV_STRIDES,
            batch,
            head,
            offsets_n,
            offsets_d,
            key_mask[:, None] & dimension_mask[None, :],
        )
        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, key_mask, WINDOW)
        if HAS_DOC_IDS:
            key_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_n, key_mask, -2)
            valid &= query_doc_ids[:, None] == key_doc_ids[None, :]

        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(local_values), input_precision="tf32x3")
        grad_scores = probabilities * (grad_probabilities - delta[:, None])
        grad_query += (
            tl.dot(
                grad_scores.to(local_values.dtype),
                local_values,
                input_precision="tf32x3",
            )
            * SCALE
        )

    store_bhsd(
        grad_query_ptr,
        grad_query,
        QUERY_STRIDES,
        batch,
        head,
        offsets_m,
        offsets_d,
        matrix_mask,
    )

    # Sink gradient
    sink = tl.load(attention_sink_ptr + head)
    sink_probability = tl.exp(sink - lse)
    sink_gradient = tl.where(query_mask, -sink_probability * delta, 0.0)
    tl.atomic_add(grad_sink_ptr + head, tl.sum(sink_gradient, axis=0))


@triton.jit
def _selected_attention_bwd_dlocal_kv(
    query_ptr,
    local_kv_ptr,
    doc_ids_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_local_kv_ptr,
    QUERY_STRIDES: tl.constexpr,
    LOCAL_KV_STRIDES: tl.constexpr,
    GRAD_LOCAL_KV_STRIDES: tl.constexpr,
    DOC_IDS_STRIDES: tl.constexpr,
    LSE_STRIDES: tl.constexpr,
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

    local_values = load_bhsd(
        local_kv_ptr,
        LOCAL_KV_STRIDES,
        batch,
        head,
        offsets_n,
        offsets_d,
        key_mask[:, None] & dimension_mask[None, :],
    )

    if HAS_DOC_IDS:
        key_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_n, key_mask, -2)

    grad_values = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    first_query = key_block * BLOCK_N
    offsets_m_base = tl.arange(0, BLOCK_M)

    for query_tile in tl.range(0, NUM_QUERY_TILES):
        query_start = first_query + query_tile * BLOCK_M
        offsets_m = query_start + offsets_m_base
        query_mask = offsets_m < S
        matrix_mask = query_mask[:, None] & dimension_mask[None, :]
        query = load_bhsd(query_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask)
        output = load_bhsd(
            output_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask
        )
        grad_output = load_bhsd(
            grad_output_ptr, QUERY_STRIDES, batch, head, offsets_m, offsets_d, matrix_mask
        )
        lse = tl.load(
            lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
            mask=query_mask,
            other=0.0,
        )
        delta = tl.sum(grad_output * output, axis=1)
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, key_mask, WINDOW)
        if HAS_DOC_IDS:
            query_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_m, query_mask, -1)
            valid &= query_doc_ids[:, None] == key_doc_ids[None, :]

        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        probabilities = tl.exp(scores - lse[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(local_values), input_precision="tf32x3")
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

    store_bhsd(
        grad_local_kv_ptr,
        grad_values,
        GRAD_LOCAL_KV_STRIDES,
        batch,
        head,
        offsets_n,
        offsets_d,
        key_mask[:, None] & dimension_mask[None, :],
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (4, 8)
        for num_stages in (1, 3)
    ],
    key=["H", "S", "D", "WINDOW", "HAS_DOC_IDS"],
    cache_results=True,
)
@triton.jit
def _selected_attention_bwd_dlocal_kv_tma(
    query_desc,
    local_desc,
    doc_ids_ptr,
    output_desc,
    grad_output_desc,
    lse_ptr,
    grad_local_desc,
    DOC_IDS_STRIDES: tl.constexpr,
    LSE_STRIDES: tl.constexpr,
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
    """TMA backward for local-KV gradients."""
    key_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_n = key_block * BLOCK_N + tl.arange(0, BLOCK_N)
    key_mask = offsets_n < S
    local_values = tl.reshape(
        local_desc.load([batch, head, key_block * BLOCK_N, 0]),
        (BLOCK_N, BLOCK_D),
    )

    if HAS_DOC_IDS:
        key_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_n, key_mask, -2)

    grad_values = tl.zeros((BLOCK_N, BLOCK_D), tl.float32)
    first_query = key_block * BLOCK_N
    offsets_m_base = tl.arange(0, BLOCK_M)

    for query_tile in tl.range(0, NUM_QUERY_TILES):
        query_start = first_query + query_tile * BLOCK_M
        offsets_m = query_start + offsets_m_base
        query_mask = offsets_m < S
        query = tl.reshape(
            query_desc.load([batch, head, query_start, 0]),
            (BLOCK_M, BLOCK_D),
        )
        output = tl.reshape(
            output_desc.load([batch, head, query_start, 0]),
            (BLOCK_M, BLOCK_D),
        )
        grad_output = tl.reshape(
            grad_output_desc.load([batch, head, query_start, 0]),
            (BLOCK_M, BLOCK_D),
        )
        lse = tl.load(
            lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
            mask=query_mask,
            other=0.0,
        )
        delta = tl.sum(grad_output * output, axis=1)
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, key_mask, WINDOW)
        if HAS_DOC_IDS:
            query_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_m, query_mask, -1)
            valid &= query_doc_ids[:, None] == key_doc_ids[None, :]

        scores = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        probabilities = tl.where(valid, tl.exp(scores - lse[:, None]), 0.0)
        grad_probabilities = tl.dot(grad_output, tl.trans(local_values), input_precision="tf32x3")
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

    grad_local_desc.store(
        [batch, head, key_block * BLOCK_N, 0],
        tl.reshape(grad_values, (1, 1, BLOCK_N, BLOCK_D)),
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": block_m},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_m in (16, 32, 64)
        for num_warps in (4, 8)
        for num_stages in (1, 3)
    ],
    key=["H", "S", "D", "SPARSE_SEQ_LEN", "TOPK"],
    cache_results=True,
)
@triton.jit
def _selected_attention_bwd_dsparse_kv(
    query_ptr,
    sparse_kv_ptr,
    selected_queries_ptr,
    block_offsets_ptr,
    output_ptr,
    grad_output_ptr,
    lse_ptr,
    grad_sparse_kv_ptr,
    QUERY_STRIDES: tl.constexpr,
    SPARSE_KV_STRIDES: tl.constexpr,
    SELECTED_QUERIES_STRIDES: tl.constexpr,
    BLOCK_OFFSETS_STRIDES: tl.constexpr,
    LSE_STRIDES: tl.constexpr,
    GRAD_SPARSE_KV_STRIDES: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    SPARSE_SEQ_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute sparse-KV gradients from an inverted index of selecting queries."""
    sparse_index = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m_base = tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    dot_rows = tl.arange(0, 16)
    dimension_mask = offsets_d < D

    sparse_value = tl.load(
        sparse_kv_ptr
        + batch * SPARSE_KV_STRIDES[0]
        + head * SPARSE_KV_STRIDES[1]
        + sparse_index * SPARSE_KV_STRIDES[2]
        + offsets_d * SPARSE_KV_STRIDES[3],
        mask=dimension_mask,
        other=0.0,
    )
    dot_value = tl.where(
        dot_rows[None, :] == 0,
        sparse_value[:, None],
        0.0,
    )
    grad_value = tl.zeros((16, BLOCK_D), tl.float32)
    entry_start = tl.load(
        block_offsets_ptr
        + batch * BLOCK_OFFSETS_STRIDES[0]
        + sparse_index * BLOCK_OFFSETS_STRIDES[1]
    )
    entry_end = tl.load(
        block_offsets_ptr
        + batch * BLOCK_OFFSETS_STRIDES[0]
        + (sparse_index + 1) * BLOCK_OFFSETS_STRIDES[1]
    )

    for entry_tile in tl.range(entry_start, entry_end, BLOCK_M):
        entry_offsets = entry_tile + offsets_m_base
        entry_mask = entry_offsets < entry_end
        query_positions = tl.load(
            selected_queries_ptr
            + batch * SELECTED_QUERIES_STRIDES[0]
            + entry_offsets * SELECTED_QUERIES_STRIDES[1],
            mask=entry_mask,
            other=0,
        )
        query_mask = entry_mask & (query_positions >= 0) & (query_positions < S)
        matrix_mask = query_mask[:, None] & dimension_mask[None, :]
        query = load_bhsd(
            query_ptr, QUERY_STRIDES, batch, head, query_positions, offsets_d, matrix_mask
        )
        output = load_bhsd(
            output_ptr, QUERY_STRIDES, batch, head, query_positions, offsets_d, matrix_mask
        )
        grad_output = load_bhsd(
            grad_output_ptr, QUERY_STRIDES, batch, head, query_positions, offsets_d, matrix_mask
        )
        lse = tl.load(
            lse_ptr
            + batch * LSE_STRIDES[0]
            + head * LSE_STRIDES[1]
            + query_positions * LSE_STRIDES[2],
            mask=query_mask,
            other=0.0,
        )
        score_tile = tl.dot(query, dot_value, input_precision="tf32x3")
        scores = tl.sum(score_tile * (dot_rows[None, :] == 0), axis=1) * SCALE
        grad_probability_tile = tl.dot(grad_output, dot_value, input_precision="tf32x3")
        grad_probabilities = tl.sum(grad_probability_tile * (dot_rows[None, :] == 0), axis=1)
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
        grad_sparse_kv_ptr
        + batch * GRAD_SPARSE_KV_STRIDES[0]
        + head * GRAD_SPARSE_KV_STRIDES[1]
        + sparse_index * GRAD_SPARSE_KV_STRIDES[2]
        + offsets_d * GRAD_SPARSE_KV_STRIDES[3],
        tl.sum(grad_value * (dot_rows[:, None] == 0), axis=0),
        mask=dimension_mask,
    )


def _build_index_query_map(
    kv_indices: torch.Tensor,
    sparse_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert query-major indices into block-major query lists (CSR format)."""
    batch, _seq_len, topk = kv_indices.shape
    if topk == 0:
        selected_queries = torch.empty(batch, 0, device=kv_indices.device, dtype=torch.int32)
        block_offsets = torch.zeros(
            batch, sparse_seq_len + 1, device=kv_indices.device, dtype=torch.int32
        )
        return selected_queries, block_offsets

    sorted_indices, sorted_entries = torch.sort(kv_indices.flatten(1), dim=-1)
    selected_queries = torch.div(sorted_entries, topk, rounding_mode="floor").to(torch.int32)
    block_ids = torch.arange(
        sparse_seq_len + 1, device=kv_indices.device, dtype=sorted_indices.dtype
    )
    block_ids = block_ids.unsqueeze(0).expand(batch, -1).contiguous()
    block_offsets = torch.searchsorted(sorted_indices, block_ids).to(torch.int32)
    return selected_queries.contiguous(), block_offsets.contiguous()


def _launch_backward(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    selected_queries: torch.Tensor,
    block_offsets: torch.Tensor,
    attention_sink: torch.Tensor,
    doc_ids: torch.Tensor | None,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    sliding_window_size: int,
    share_kv: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch backward kernels for selected attention."""
    batch, heads, seq_len, head_dim = query.shape
    sparse_seq_len = sparse_kv.shape[2]
    topk = kv_indices.shape[-1]
    block_m = 64
    block_n = 32
    block_d = max(16, triton.next_power_of_2(head_dim))
    scale = 1.0 / math.sqrt(head_dim)
    grad_output = grad_output.contiguous()

    grad_query = torch.empty_like(query)
    # Expanded shared KV has a zero head stride, but its per-head gradients
    # need distinct storage before ExpandBackward sums them.
    grad_local_kv = torch.empty(local_kv.shape, device=local_kv.device, dtype=local_kv.dtype)

    has_doc_ids = doc_ids is not None
    doc_ids = query if doc_ids is None else doc_ids
    use_tma = can_use_tma(query) and can_use_tma(local_kv) and can_use_tma(grad_output)

    num_local_query_tiles = (
        triton.cdiv(sliding_window_size + block_n - 1, block_m) if sliding_window_size else 0
    )

    # Zero head strides share values; share_kv also guarantees autograd will sum dKV heads.
    use_shared_schedule = (
        share_kv
        and can_use_shared_kv_schedule(query, sparse_kv, local_kv, sliding_window_size)
        and head_dim <= 128
    )
    if use_shared_schedule:
        block_h = min(32, triton.next_power_of_2(heads))
        block_k = max(16, min(64, triton.next_power_of_2(topk)))
        grad_sink_partials = torch.empty(
            batch, heads, seq_len, device=query.device, dtype=torch.float32
        )
        _selected_attention_bwd_dq_shared[(seq_len, batch, triton.cdiv(heads, block_h))](
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            doc_ids,
            output,
            grad_output,
            lse,
            attention_sink,
            grad_query,
            grad_sink_partials,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            DOC_IDS_STRIDES=doc_ids.stride(),
            LSE_STRIDES=lse.stride(),
            GRAD_SINK_PARTIALS_STRIDES=grad_sink_partials.stride(),
            B=batch,
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            WINDOW=sliding_window_size,
            SCALE=scale,
            HAS_DOC_IDS=has_doc_ids,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
        )
        grad_sink_fp32 = grad_sink_partials.sum(dim=(0, 2))
    else:
        num_local_key_tiles = (
            triton.cdiv(sliding_window_size + block_m - 1, block_n) if sliding_window_size else 0
        )
        grad_sink_fp32 = torch.zeros(heads, device=query.device, dtype=torch.float32)
        _selected_attention_bwd_dq[(triton.cdiv(seq_len, block_m), batch * heads)](
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            doc_ids,
            output,
            grad_output,
            lse,
            attention_sink,
            grad_query,
            grad_sink_fp32,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            DOC_IDS_STRIDES=doc_ids.stride(),
            LSE_STRIDES=lse.stride(),
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            WINDOW=sliding_window_size,
            SCALE=scale,
            HAS_DOC_IDS=has_doc_ids,
            NUM_LOCAL_TILES=num_local_key_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=8,
            num_stages=3,
        )

    local_grid = (triton.cdiv(seq_len, block_n), batch * heads)
    if use_tma:
        query_desc = TensorDescriptor.from_tensor(query, [1, 1, block_m, block_d])
        local_desc = TensorDescriptor.from_tensor(local_kv, [1, 1, block_n, block_d])
        output_desc = TensorDescriptor.from_tensor(output, [1, 1, block_m, block_d])
        grad_output_desc = TensorDescriptor.from_tensor(grad_output, [1, 1, block_m, block_d])
        grad_local_desc = TensorDescriptor.from_tensor(grad_local_kv, [1, 1, block_n, block_d])
        _selected_attention_bwd_dlocal_kv_tma[local_grid](
            query_desc,
            local_desc,
            doc_ids,
            output_desc,
            grad_output_desc,
            lse,
            grad_local_desc,
            DOC_IDS_STRIDES=doc_ids.stride(),
            LSE_STRIDES=lse.stride(),
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
        )
    else:
        _selected_attention_bwd_dlocal_kv[local_grid](
            query,
            local_kv,
            doc_ids,
            output,
            grad_output,
            lse,
            grad_local_kv,
            QUERY_STRIDES=query.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            GRAD_LOCAL_KV_STRIDES=grad_local_kv.stride(),
            DOC_IDS_STRIDES=doc_ids.stride(),
            LSE_STRIDES=lse.stride(),
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
            num_stages=3,
        )

    if topk == 0:
        kv_heads = 1 if share_kv else heads
        grad_sparse_kv = torch.zeros(
            batch,
            kv_heads,
            sparse_seq_len,
            head_dim,
            device=sparse_kv.device,
            dtype=sparse_kv.dtype,
        )
    elif use_shared_schedule and not torch.are_deterministic_algorithms_enabled():
        grad_sparse_kv_fp32 = torch.zeros(
            batch,
            1,
            sparse_seq_len,
            head_dim,
            device=sparse_kv.device,
            dtype=torch.float32,
        )
        sparse_grid = lambda meta: (
            seq_len,
            batch,
            triton.cdiv(heads, meta["BLOCK_H"]) * triton.cdiv(topk, meta["BLOCK_K"]),
        )
        _selected_attention_bwd_dsparse_kv_shared_atomic[sparse_grid](
            query,
            sparse_kv,
            kv_indices,
            output,
            grad_output,
            lse,
            grad_sparse_kv_fp32,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            LSE_STRIDES=lse.stride(),
            GRAD_SPARSE_KV_STRIDES=grad_sparse_kv_fp32.stride(),
            B=batch,
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            SCALE=scale,
            BLOCK_D=block_d,
        )
        grad_sparse_kv = grad_sparse_kv_fp32.to(sparse_kv.dtype)
    else:
        if use_shared_schedule:
            grad_sparse_kv_partials = torch.zeros(
                sparse_kv.shape,
                device=sparse_kv.device,
                dtype=sparse_kv.dtype,
            )
            sparse_kernel = _selected_attention_bwd_dsparse_kv_shared
            sparse_grid = lambda meta: (
                sparse_seq_len,
                batch,
                triton.cdiv(heads, meta["BLOCK_H"]),
            )
        else:
            grad_sparse_kv_partials = torch.empty(
                sparse_kv.shape,
                device=sparse_kv.device,
                dtype=sparse_kv.dtype,
            )
            sparse_kernel = _selected_attention_bwd_dsparse_kv
            sparse_grid = (sparse_seq_len, batch * heads)

        sparse_kernel[sparse_grid](
            query,
            sparse_kv,
            selected_queries,
            block_offsets,
            output,
            grad_output,
            lse,
            grad_sparse_kv_partials,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            SELECTED_QUERIES_STRIDES=selected_queries.stride(),
            BLOCK_OFFSETS_STRIDES=block_offsets.stride(),
            LSE_STRIDES=lse.stride(),
            GRAD_SPARSE_KV_STRIDES=grad_sparse_kv_partials.stride(),
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            SCALE=scale,
            BLOCK_D=block_d,
        )
        grad_sparse_kv = (
            grad_sparse_kv_partials.sum(dim=1, keepdim=True)
            if share_kv
            else grad_sparse_kv_partials
        )

    if share_kv:
        grad_local_kv = grad_local_kv.sum(dim=1, keepdim=True)

    return grad_query, grad_sparse_kv, grad_local_kv, grad_sink_fp32.to(attention_sink.dtype)
