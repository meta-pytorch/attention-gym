from typing import Literal

import torch


Backend = Literal["eager", "triton", "cute"]
Mode = Literal["auto", "chunked", "recurrent"]


def _validate_inputs(
    Q: object,
    KV: object,
    index_kv: object,
    indices: object,
    attention_sink: object,
    doc_ids: object,
    sliding_window_size: object,
    share_kv: object,
) -> None:
    if type(sliding_window_size) is not int:
        raise TypeError(
            f"sliding_window_size must be a Python int, got {type(sliding_window_size).__name__}."
        )
    if type(share_kv) is not bool:
        raise TypeError(f"share_kv must be a Python bool, got {type(share_kv).__name__}.")

    tensors = {
        "Q": Q,
        "KV": KV,
        "index_kv": index_kv,
        "indices": indices,
        "attention_sink": attention_sink,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    assert isinstance(Q, torch.Tensor)
    assert isinstance(KV, torch.Tensor)
    assert isinstance(index_kv, torch.Tensor)
    assert isinstance(indices, torch.Tensor)
    assert isinstance(attention_sink, torch.Tensor)

    if Q.ndim != 4:
        raise ValueError("Q must have shape [batch, heads, sequence_length, head_dim].")
    batch, heads, sequence_length, head_dim = Q.shape
    if min(batch, heads, sequence_length, head_dim) <= 0:
        raise ValueError("Q dimensions must all be positive.")
    if not Q.is_floating_point():
        raise TypeError("Selected attention inputs must have a floating-point dtype.")

    if sliding_window_size < 0:
        raise ValueError("sliding_window_size must be non-negative.")

    expected_kv_heads = 1 if share_kv else heads
    if (
        KV.ndim != 4
        or KV.shape[0] != batch
        or KV.shape[1] != expected_kv_heads
        or KV.shape[2] != sequence_length
        or KV.shape[3] != head_dim
    ):
        expected_h = "1 or heads" if share_kv else "heads"
        raise ValueError(
            f"KV must have shape [batch, {expected_h}, sequence_length, head_dim], "
            f"got {list(KV.shape)}."
        )
    index_seq_len = index_kv.shape[2]

    if indices.ndim != 3 or indices.shape[0] != batch or indices.shape[1] != sequence_length:
        raise ValueError(
            f"indices must have shape [batch, sequence_length, num_topk], "
            f"got {list(indices.shape)}."
        )
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"indices must be an integer tensor (int32 or int64), got {indices.dtype}."
        )
    if indices.device != Q.device:
        raise ValueError(f"indices must be on {Q.device}, got {indices.device}.")
    num_topk = indices.shape[2]
    if num_topk > index_seq_len:
        raise ValueError(
            f"indices num_topk ({num_topk}) must not exceed "
            f"index_kv sequence length ({index_seq_len})."
        )

    if attention_sink.shape != (heads,):
        raise ValueError(
            f"attention_sink must have shape [{heads}], got {list(attention_sink.shape)}."
        )

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
    Q,
    KV,
    index_kv,
    indices,
    attention_sink,
    doc_ids=None,
    sliding_window_size: int = 512,
    share_kv: bool = True,
    backend: Backend = "eager",
    mode: Mode = "auto",
):
    """
    Args:
        Q: query, shaped like (batch_size, num_heads, sequence_length, head_dim)

        KV: Key and Value for the sliding window branch,
            represented as a shared tensor, (batch_size, 1, sequence_length, head_dim) if share_kv = False
            Otherwise represented as (batch_size, num_heads, sequence_length, head_dim)

        index_kv: KV for the indexing branch, shape of (batch, 1, X, head_dim) if share_kv = False
            Otherwise represented as (batch_size, num_heads, X, head_dim)
            where X is any number greater than

        indices: Which indices to attend to. Shape of (batch, num_heads, num_topk_blocks), integer tensor
            If None, index_kv will be ignored

        attention_sink: tensor in shape of (num_heads, ), learnable per head weight that occupies denominator of softmax

        doc_ids: Integer tensor in shape of (batch_size, sequence_length) or None.
            Looks something like [0, 0, 0, 1, 1, 2, 2, 2, 2], where all tokens with the same id can causally attend to each other
            If doc_ids[i, j] = doc_ids[i, j-y], then Q[i, j] can causally attend to KV[i, j-y]
            Should be monotonically increasing on the sequence axis
            If doc_ids is None, all tokens on the same sequence axis will be assumed to be in the same document.

        sliding_window_size: Integer, size of sliding window

        share_kv: bool, true iff all query heads attend to the same KV head

        backend: one of eager, triton, or cute, controls which backend executes this code

        mode: Currently only chunked is supported; auto defaults to chunked
    Returns:
        Tensor in shape of (batch_size, num_heads, sequence_length, head_dim)
    """
    _validate_inputs(
        Q, KV, index_kv, indices, attention_sink, doc_ids, sliding_window_size, share_kv
    )

    match backend:
        case "eager":
            from . import reference

            return reference.selected_attention(
                Q,
                KV,
                index_kv,
                indices,
                attention_sink,
                doc_ids,
                sliding_window_size,
                share_kv,
            )
        case "triton":
            from . import triton as triton_backend

            return triton_backend.selected_attention(
                Q,
                KV,
                index_kv,
                indices,
                attention_sink,
                doc_ids,
                sliding_window_size,
                share_kv,
            )
        case _:
            raise NotImplementedError(f"Backend {backend!r} is not supported yet.")
