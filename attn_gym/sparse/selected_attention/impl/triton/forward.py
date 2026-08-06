"""Forward kernels and launcher for Triton selected attention."""

import math

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import can_use_tma, ptr_offset

from .primitives import causal_window_mask, load_bhsd, load_bs, online_softmax_update, store_bhsd


@triton.jit
def _selected_attention_fwd(
    query_ptr,
    sparse_kv_ptr,
    local_kv_ptr,
    kv_indices_ptr,
    doc_ids_ptr,
    attention_sink_ptr,
    output_ptr,
    lse_ptr,
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
    """Apply online softmax over selected sparse entries and the local window."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D

    query = load_bhsd(
        query_ptr,
        QUERY_STRIDES,
        batch,
        head,
        offsets_m,
        offsets_d,
        query_mask[:, None] & dimension_mask[None, :],
    )

    if HAS_DOC_IDS:
        query_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_m, query_mask, -1)

    sink = tl.load(attention_sink_ptr + head).to(tl.float32)
    running_max = tl.full((BLOCK_M,), sink, tl.float32)
    running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    # Each query has its own set of selected sparse entries.
    for selected_slot in tl.static_range(0, TOPK):
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
        logit = tl.sum(query * sparse_value, axis=1) * SCALE
        logit = tl.where(valid, logit, -float("inf"))
        new_max = tl.maximum(running_max, logit)
        alpha = tl.exp(running_max - new_max)
        probability = tl.exp(logit - new_max)
        accumulator = accumulator * alpha[:, None] + probability[:, None] * sparse_value
        running_sum = running_sum * alpha + probability
        running_max = new_max

    # The local window is processed in tensor-core-friendly tiles.
    first_local_position = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for local_tile in tl.static_range(0, NUM_LOCAL_TILES):
        local_start = first_local_position + local_tile * BLOCK_N
        offsets_n = local_start + offsets_n_base
        local_mask = (offsets_n >= 0) & (offsets_n < S)
        local_values = load_bhsd(
            local_kv_ptr,
            LOCAL_KV_STRIDES,
            batch,
            head,
            offsets_n,
            offsets_d,
            local_mask[:, None] & dimension_mask[None, :],
        )
        logits = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, local_mask, WINDOW)
        if HAS_DOC_IDS:
            key_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_n, local_mask, -2)
            valid &= query_doc_ids[:, None] == key_doc_ids[None, :]

        logits = tl.where(valid, logits, -float("inf"))
        accumulator, running_max, running_sum = online_softmax_update(
            accumulator, running_max, running_sum, logits, local_values
        )

    output = accumulator / running_sum[:, None]
    store_bhsd(
        output_ptr,
        output,
        QUERY_STRIDES,
        batch,
        head,
        offsets_m,
        offsets_d,
        query_mask[:, None] & dimension_mask[None, :],
    )
    tl.store(
        lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
        running_max + tl.log(running_sum),
        mask=query_mask,
    )


def prune_shared_forward_configs(configs, _named_args, D, **_):
    """Avoid local tiles that exceed shared memory for wide head dimensions."""
    if D <= 128:
        return configs
    return [config for config in configs if config.kwargs["BLOCK_N"] == 64]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": block_n}, num_warps=num_warps, num_stages=1)
        for block_n in (64, 128, 256)
        for num_warps in (4, 8)
    ],
    key=["B", "H", "S", "D", "SPARSE_SEQ_LEN", "TOPK", "WINDOW", "HAS_DOC_IDS"],
    prune_configs_by={"early_config_prune": prune_shared_forward_configs},
    cache_results=True,
)
@triton.jit
def _selected_attention_fwd_shared(
    query_ptr,
    sparse_kv_ptr,
    local_kv_ptr,
    kv_indices_ptr,
    doc_ids_ptr,
    attention_sink_ptr,
    output_ptr,
    lse_ptr,
    QUERY_STRIDES: tl.constexpr,
    SPARSE_KV_STRIDES: tl.constexpr,
    LOCAL_KV_STRIDES: tl.constexpr,
    KV_INDICES_STRIDES: tl.constexpr,
    DOC_IDS_STRIDES: tl.constexpr,
    LSE_STRIDES: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    SPARSE_SEQ_LEN: tl.constexpr,
    TOPK: tl.constexpr,
    WINDOW: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_DOC_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Process one sequence position across heads to reuse shared KV tiles."""
    sequence = tl.program_id(0)
    batch = tl.program_id(1)
    head_block = tl.program_id(2)

    offsets_h = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offsets_d = tl.arange(0, BLOCK_D)
    head_mask = offsets_h < H
    dimension_mask = offsets_d < D

    query = tl.load(
        query_ptr
        + ptr_offset(
            (batch, offsets_h[:, None], sequence, offsets_d[None, :]),
            QUERY_STRIDES,
        ),
        mask=head_mask[:, None] & dimension_mask[None, :],
        other=0.0,
    )
    sink = tl.load(attention_sink_ptr + offsets_h, mask=head_mask, other=0.0).to(tl.float32)
    running_max = sink
    running_sum = tl.full((BLOCK_H,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)

    if TOPK:
        # Keep on-chip storage bounded when the selected set contains hundreds of entries.
        offsets_k = tl.arange(0, BLOCK_K)
        for selected_start in tl.range(0, TOPK, BLOCK_K, num_stages=2):
            selected_offsets = selected_start + offsets_k
            selected_idx = tl.load(
                kv_indices_ptr
                + ptr_offset((batch, sequence, selected_offsets), KV_INDICES_STRIDES),
                mask=selected_offsets < TOPK,
                other=-1,
            )
            selected_valid = (
                (selected_offsets < TOPK) & (selected_idx >= 0) & (selected_idx < SPARSE_SEQ_LEN)
            )
            selected_idx = tl.where(selected_valid, selected_idx, 0)
            sparse_values = tl.load(
                sparse_kv_ptr
                + ptr_offset(
                    (batch, 0, selected_idx[:, None], offsets_d[None, :]),
                    SPARSE_KV_STRIDES,
                ),
                mask=selected_valid[:, None] & dimension_mask[None, :],
                other=0.0,
            )
            logits = tl.dot(query, tl.trans(sparse_values), input_precision="tf32x3") * SCALE
            logits = tl.where(head_mask[:, None] & selected_valid[None, :], logits, -float("inf"))
            accumulator, running_max, running_sum = online_softmax_update(
                accumulator, running_max, running_sum, logits, sparse_values
            )

    if HAS_DOC_IDS:
        query_doc_id = tl.load(doc_ids_ptr + ptr_offset((batch, sequence), DOC_IDS_STRIDES))

    offsets_n_base = tl.arange(0, BLOCK_N)
    first_local_position = sequence - WINDOW + 1
    for local_start in tl.range(0, WINDOW, BLOCK_N, num_stages=2):
        offsets_n = first_local_position + local_start + offsets_n_base
        local_valid = (offsets_n >= 0) & (offsets_n <= sequence) & (offsets_n < S)
        local_values = tl.load(
            local_kv_ptr
            + ptr_offset(
                (batch, 0, offsets_n[:, None], offsets_d[None, :]),
                LOCAL_KV_STRIDES,
            ),
            mask=local_valid[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        if HAS_DOC_IDS:
            key_doc_ids = tl.load(
                doc_ids_ptr + ptr_offset((batch, offsets_n), DOC_IDS_STRIDES),
                mask=local_valid,
                other=-1,
            )
            local_valid &= key_doc_ids == query_doc_id

        logits = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        logits = tl.where(head_mask[:, None] & local_valid[None, :], logits, -float("inf"))
        accumulator, running_max, running_sum = online_softmax_update(
            accumulator, running_max, running_sum, logits, local_values
        )

    tl.store(
        output_ptr
        + ptr_offset(
            (batch, offsets_h[:, None], sequence, offsets_d[None, :]),
            QUERY_STRIDES,
        ),
        accumulator / running_sum[:, None],
        mask=head_mask[:, None] & dimension_mask[None, :],
    )
    tl.store(
        lse_ptr + ptr_offset((batch, offsets_h, sequence), LSE_STRIDES),
        running_max + tl.log(running_sum),
        mask=head_mask,
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in (4, 8)
        for num_stages in (1, 3)
    ],
    key=["H", "S", "D", "SPARSE_SEQ_LEN", "TOPK", "WINDOW", "HAS_DOC_IDS"],
    cache_results=True,
)
@triton.jit
def _selected_attention_fwd_tma(
    query_desc,
    sparse_kv_ptr,
    local_desc,
    kv_indices_ptr,
    doc_ids_ptr,
    attention_sink_ptr,
    output_desc,
    lse_ptr,
    SPARSE_KV_STRIDES: tl.constexpr,
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
    """TMA forward using host-created descriptors for dense tiles."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    head = batch_head % H
    batch = batch_head // H

    offsets_m = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_d = tl.arange(0, BLOCK_D)
    query_mask = offsets_m < S
    dimension_mask = offsets_d < D
    query = tl.reshape(
        query_desc.load([batch, head, query_block * BLOCK_M, 0]),
        (BLOCK_M, BLOCK_D),
    )

    if HAS_DOC_IDS:
        query_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_m, query_mask, -1)

    sink = tl.load(attention_sink_ptr + head).to(tl.float32)
    running_max = tl.full((BLOCK_M,), sink, tl.float32)
    running_sum = tl.full((BLOCK_M,), 1.0, tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for selected_slot in tl.static_range(0, TOPK):
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
        logit = tl.sum(query * sparse_value, axis=1) * SCALE
        logit = tl.where(valid, logit, -float("inf"))
        new_max = tl.maximum(running_max, logit)
        alpha = tl.exp(running_max - new_max)
        probability = tl.exp(logit - new_max)
        accumulator = accumulator * alpha[:, None] + probability[:, None] * sparse_value
        running_sum = running_sum * alpha + probability
        running_max = new_max

    first_local_position = query_block * BLOCK_M - WINDOW + 1
    offsets_n_base = tl.arange(0, BLOCK_N)
    for local_tile in tl.static_range(0, NUM_LOCAL_TILES):
        local_start = first_local_position + local_tile * BLOCK_N
        offsets_n = local_start + offsets_n_base
        local_mask = (offsets_n >= 0) & (offsets_n < S)
        local_values = tl.reshape(
            local_desc.load([batch, head, local_start, 0]),
            (BLOCK_N, BLOCK_D),
        )
        logits = tl.dot(query, tl.trans(local_values), input_precision="tf32x3") * SCALE
        valid = causal_window_mask(offsets_m, offsets_n, query_mask, local_mask, WINDOW)
        if HAS_DOC_IDS:
            key_doc_ids = load_bs(doc_ids_ptr, DOC_IDS_STRIDES, batch, offsets_n, local_mask, -2)
            valid &= query_doc_ids[:, None] == key_doc_ids[None, :]

        logits = tl.where(valid, logits, -float("inf"))
        accumulator, running_max, running_sum = online_softmax_update(
            accumulator, running_max, running_sum, logits, local_values
        )

    output_desc.store(
        [batch, head, query_block * BLOCK_M, 0],
        tl.reshape(accumulator / running_sum[:, None], (1, 1, BLOCK_M, BLOCK_D)),
    )
    tl.store(
        lse_ptr + batch * LSE_STRIDES[0] + head * LSE_STRIDES[1] + offsets_m * LSE_STRIDES[2],
        running_max + tl.log(running_sum),
        mask=query_mask,
    )


def _launch_forward(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    doc_ids: torch.Tensor | None,
    sliding_window_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the forward kernel and return its output and log-sum-exp state."""
    batch, heads, seq_len, head_dim = query.shape
    topk = kv_indices.shape[-1]
    sparse_seq_len = sparse_kv.shape[2]
    block_d = max(16, triton.next_power_of_2(head_dim))
    output = torch.empty_like(query)
    lse = torch.empty(batch, heads, seq_len, device=query.device, dtype=torch.float32)
    has_doc_ids = doc_ids is not None
    doc_ids = query if doc_ids is None else doc_ids

    # This head-major schedule is tuned for shared KV on Blackwell.
    if (
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
    ):
        # A smaller head tile keeps D=512 accumulators within Blackwell's resources.
        block_h = (
            triton.next_power_of_2(heads)
            if head_dim <= 128
            else min(32, triton.next_power_of_2(heads))
        )
        block_k = max(16, min(64, triton.next_power_of_2(topk))) if topk else 16
        _selected_attention_fwd_shared[(seq_len, batch, triton.cdiv(heads, block_h))](
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            doc_ids,
            attention_sink,
            output,
            lse,
            QUERY_STRIDES=query.stride(),
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            LOCAL_KV_STRIDES=local_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            DOC_IDS_STRIDES=doc_ids.stride(),
            LSE_STRIDES=lse.stride(),
            B=batch,
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            WINDOW=sliding_window_size,
            SCALE=1.0 / math.sqrt(head_dim),
            HAS_DOC_IDS=has_doc_ids,
            BLOCK_H=block_h,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
        )
        return output, lse

    block_m = 64
    block_n = 128
    num_local_tiles = (
        triton.cdiv(sliding_window_size + block_m - 1, block_n) if sliding_window_size else 0
    )
    grid = (triton.cdiv(seq_len, block_m), batch * heads)
    if can_use_tma(query) and can_use_tma(local_kv):
        query_desc = TensorDescriptor.from_tensor(query, [1, 1, block_m, block_d])
        local_desc = TensorDescriptor.from_tensor(local_kv, [1, 1, block_n, block_d])
        output_desc = TensorDescriptor.from_tensor(output, [1, 1, block_m, block_d])
        _selected_attention_fwd_tma[grid](
            query_desc,
            sparse_kv,
            local_desc,
            kv_indices,
            doc_ids,
            attention_sink,
            output_desc,
            lse,
            SPARSE_KV_STRIDES=sparse_kv.stride(),
            KV_INDICES_STRIDES=kv_indices.stride(),
            DOC_IDS_STRIDES=doc_ids.stride(),
            LSE_STRIDES=lse.stride(),
            H=heads,
            S=seq_len,
            D=head_dim,
            SPARSE_SEQ_LEN=sparse_seq_len,
            TOPK=topk,
            WINDOW=sliding_window_size,
            SCALE=1.0 / math.sqrt(head_dim),
            HAS_DOC_IDS=has_doc_ids,
            NUM_LOCAL_TILES=num_local_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )
    else:
        _selected_attention_fwd[grid](
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            doc_ids,
            attention_sink,
            output,
            lse,
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
            SCALE=1.0 / math.sqrt(head_dim),
            HAS_DOC_IDS=has_doc_ids,
            NUM_LOCAL_TILES=num_local_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            num_warps=8,
            num_stages=3,
        )
    return output, lse
