from typing import Literal

import torch
from torch import Tensor


def _validate_inputs(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor,
    doc_ids: Tensor,
    sliding_window_size: Tensor,
    share_kv: bool,
) -> None:
    if type(sliding_window_size) is not int:
        raise TypeError(
            f"sliding_window_size must be a Python int, got {type(sliding_window_size).__name__}."
        )

    tensors = {
        "query": query,
        "local_kv": local_kv,
        "sparse_kv": sparse_kv,
        "kv_indices": kv_indices,
        "attention_sink": attention_sink,
    }



    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")



    assert isinstance(query, torch.Tensor)
    assert isinstance(local_kv, torch.Tensor)
    assert isinstance(sparse_kv, torch.Tensor)
    assert isinstance(attention_sink, torch.Tensor)

    assert local_kv.dtype == query.dtype, f"local_kv must have the same dtype as query, but got {local_kv.dtype} and {query.dtype}."
    assert sparse_kv.dtype == query.dtype, f"sparse_kv must have the same dtype as query, but got {sparse_kv.dtype} and {query.dtype}."
    assert attention_sink.dtype == query.dtype, f"attention_sink must have the same dtype as query, but got {attention_sink.dtype} and {query.dtype}."

    assert local_kv.device == query.device, f"local_kv must be on the same device as query, but got {local_kv.device} and {query.device}."
    assert sparse_kv.device == query.device, f"sparse_kv must be on the same device as query, but got {sparse_kv.device} and {query.device}."
    assert attention_sink.device == query.device, f"attention_sink must be on the same device as query, but got {attention_sink.device} and {query.device}."
    if doc_ids is not None:
        assert doc_ids.device == query.device, f"doc_ids must be on the same device as query, but got {doc_ids.device} and {query.device}."


    if query.ndim != 4:
        raise ValueError("query must have shape [batch, heads, sequence_length, head_dim].")
    batch, heads, sequence_length, head_dim = query.shape
    if min(batch, heads, sequence_length, head_dim) <= 0:
        raise ValueError("query dimensions must all be positive.")
    if not query.is_floating_point():
        raise TypeError("Selected attention inputs must have a floating-point dtype.")

    if sliding_window_size < 0:
        raise ValueError("sliding_window_size must be non-negative.")

    expected_kv_heads = 1 if share_kv else heads
    if (
        local_kv.ndim != 4
        or local_kv.shape[0] != batch
        or local_kv.shape[1] != expected_kv_heads
        or local_kv.shape[2] != sequence_length
        or local_kv.shape[3] != head_dim
    ):
        expected_h = "1 or heads" if share_kv else "heads"
        raise ValueError(
            f"local_kv must have shape [batch, {expected_h}, sequence_length, head_dim], "
            f"got {list(local_kv.shape)}."
        )
    sparse_seq_len = sparse_kv.shape[2]

    if (
        sparse_kv.ndim != 4
        or sparse_kv.shape[0] != batch
        or sparse_kv.shape[1] != expected_kv_heads
        or sparse_kv.shape[3] != head_dim
    ):
        expected_h = "1 or heads" if share_kv else "heads"
        raise ValueError(
            f"sparse_kv must have shape [batch, {expected_h}, sequence_length, head_dim], "
            f"got {list(sparse_kv.shape)}."
        )

    if (
        kv_indices.ndim != 3
        or kv_indices.shape[0] != batch
        or kv_indices.shape[1] != sequence_length
    ):
        raise ValueError(
            f"kv_indices must have shape [batch, sequence_length, num_topk], "
            f"got {list(kv_indices.shape)}."
        )
    if kv_indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"kv_indices must be an integer tensor (int32 or int64), got {kv_indices.dtype}."
        )
    if kv_indices.device != query.device:
        raise ValueError(f"kv_indices must be on {query.device}, got {kv_indices.device}.")
    num_topk = kv_indices.shape[2]
    if num_topk > sparse_seq_len:
        raise ValueError(
            f"kv_indices num_topk ({num_topk}) must not exceed "
            f"sparse_kv sequence length ({sparse_seq_len})."
        )

    if attention_sink.shape != (heads,):
        raise ValueError(
            f"attention_sink must have shape [{heads}], got {list(attention_sink.shape)}."
        )

    # --- doc_ids (optional) ---
    if doc_ids is not None:
        if not isinstance(doc_ids, torch.Tensor):
            raise TypeError(
                f"doc_ids must be a torch.Tensor or None, got {type(doc_ids).__name__}."
            )
        if doc_ids.ndim != 2 or doc_ids.shape[0] != batch or doc_ids.shape[1] != sequence_length:
            raise ValueError(
                f"doc_ids must have shape [batch, sequence_length], " f"got {list(doc_ids.shape)}."
            )


def selected_attention(
    query,
    local_kv,
    sparse_kv,
    kv_indices,
    attention_sink,
    doc_ids=None,
    sliding_window_size: int = 512,
    backend: str = "triton",
    mode: str = "auto",
):
    """
    Performs selected attention
        Each query attends to the previous sliding_window_size elements in the local_kv tensor
        As well the positions in sparse_kv pointed to by kv_indices
        Only one softmax is applied, and it covers the sparse_kv values as well as the sliding
        window values
    Args:
        query: query, shaped like (batch_size, num_heads, sequence_length, head_dim)

        local_kv: Key and Value for the sliding window branch,
            represented as a shared tensor, (batch_size, 1, sequence_length, head_dim)
            Or represented as (batch_size, num_heads, sequence_length, head_dim)

        sparse_kv: KV candidate pool for the indexing branch, shape of (batch, 1, X, head_dim)
            Or shaped as (batch_size, num_heads, X, head_dim)
            where X is any integer

        kv_indices: Which entries to select from sparse_kv.
            Shape of (batch, sequence_length, num_topk_blocks), integer tensor
            query[i, j] will attend to all sparse_kv[i, kv_indices[k]] for all k < num_topk_blocks

        attention_sink: tensor in shape of (num_heads, ), learnable per head weight that occupies
            denominator of softmax

        doc_ids: Integer tensor in shape of (batch_size, sequence_length) or None.
            Looks something like [0, 0, 0, 1, 1, 2, 2, 2, 2], where all tokens with the same id
            can causally attend to each other
            If doc_ids[i, j] = doc_ids[i, j-y], then query[i, j] can causally attend to
            local_kv[i, j-y]
            Should be monotonically increasing on the sequence axis
            If doc_ids is None, all tokens on the same sequence axis will be assumed to be in the
            same document.
            Only applies to the sliding window branch.
            It's the caller's responsibility to make sure kv_indices don't cross document boundaries

        sliding_window_size: Integer, size of sliding window

        backend: one of eager, triton, or cute, controls which backend executes this code

        mode: Currently only chunked is supported; auto defaults to chunked
    Returns:
        Tensor in shape of (batch_size, num_heads, sequence_length, head_dim)
    """
    share_kv = (sparse_kv.shape[1] == 1)
    if not torch.compiler.is_compiling():
        _validate_inputs(
            query, local_kv, sparse_kv, kv_indices, attention_sink, doc_ids,
            sliding_window_size, share_kv
        )

    match backend:
        case "eager":
            from .impl import reference

            return reference.selected_attention(
                query,
                local_kv,
                sparse_kv,
                kv_indices,
                attention_sink,
                doc_ids,
                sliding_window_size,
                share_kv,
            )
        case "triton":
            from .impl import triton as triton_backend

            return triton_backend.selected_attention(
                query,
                local_kv,
                sparse_kv,
                kv_indices,
                attention_sink,
                doc_ids,
                sliding_window_size,
                share_kv,
            )
        case _:
            raise NotImplementedError(f"Backend {backend!r} is not supported yet.")
