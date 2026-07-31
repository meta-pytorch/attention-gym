"""Torch-only compressed sparse attention reference implementation."""

import math

import torch
import torch.nn.functional as F


def make_block_mask(query_length, num_blocks, compression_rate, device, dtype):
    query_positions = torch.arange(query_length, device=device)
    block_positions = torch.arange(num_blocks, device=device)
    completed_blocks = (query_positions + 1) // compression_rate
    bool_mask = block_positions[None, :] < completed_blocks[:, None]
    mask = torch.zeros(bool_mask.shape, device=bool_mask.device, dtype=dtype)
    return mask.masked_fill(~bool_mask, float("-inf"))


def make_sliding_window_mask(query_length, window_size, device, dtype):
    query_positions = torch.arange(query_length, device=device)[:, None]
    key_positions = torch.arange(query_length, device=device)[None, :]
    valid = (key_positions <= query_positions) & (
        key_positions >= query_positions - window_size + 1
    )
    return torch.zeros(
        (query_length, query_length),
        device=device,
        dtype=dtype,
    ).masked_fill(~valid, float("-inf"))


def sink_softmax(x, sink, dim):
    sink = sink[None, :, None, None]
    maximums = torch.max(x, dim=dim, keepdim=True).values
    maximums = torch.maximum(maximums, sink)
    x = x - maximums
    sink = sink - maximums
    x = torch.exp(x)
    return x / (torch.sum(x, dim, keepdim=True) + torch.exp(sink))


import torch


def make_packed_mask(
    doc_ids: torch.Tensor,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Args:
        doc_ids: Integer tensor of shape [batch, sequence].
                 Tokens in the same document have the same ID.

    Returns:
        Additive attention mask of shape [batch, 1, sequence, sequence]
    """
    batch_size, seq_len = doc_ids.shape
    device = doc_ids.device

    # [B, S, 1] == [B, 1, S] -> [B, S_query, S_key]
    same_document = doc_ids[:, :, None] == doc_ids[:, None, :]

    mask = torch.full(
        (batch_size, seq_len, seq_len),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    mask.masked_fill_(allowed, 0.0)

    # Head-broadcasting dimension.
    return mask[:, None, :, :]

def selected_attention(
    Q,
    KV,
    index_kv,
    indices, 
    attention_sink,
    doc_ids,
    sliding_window_size: int,
    share_kv: bool = True,
):
    '''
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
    '''
    device = Q.device
    dtype = Q.dtype
    b, h, s, head_dim = Q.shape
    index_sequence_length = index_kv.shape[2]
    if share_kv:
        KV = KV.expand(-1, h, -1, -1)
        if indices is None:
            index_kv = index_kv.expand(-1, h, -1, -1)
    if indices is not None:
        # We have s queries that each potentially attend to index_sequence_length elements
        topk_mask = torch.full((b, s, index_sequence_length), float("-inf"), device=device, dtype=dtype)
        topk_mask.scatter_(dim=-1, index=indices, value=0.0)
    
    
    SWA_mask = make_sliding_window_mask(s, sliding_window_size, device, dtype).unsqueeze(0)
    SWA_mask = SWA_mask.expand(b, -1, -1)

    if doc_ids is not None:
        packing_mask = make_packed_mask(doc_ids, dtype = dtype)
    

    if indices is not None:
        attention_kv = torch.cat([index_kv, KV], dim=-2)
        attention_mask = torch.cat([topk_mask, SWA_mask], dim=-1).unsqueeze(1)
    else:
        attention_kv = KV
        attention_mask = SWA_mask
    scale = head_dim**0.5

    P = sink_softmax(
        torch.matmul(Q, torch.permute(attention_kv, (0, 1, 3, 2))) / scale + attention_mask,
        attention_sink,
        dim=-1,
    )
    attn_output = P @ attention_kv
    return attn_output



b, h, s, head_dim = 3, 5, 14, 17
Q = torch.randn(b, h, s, head_dim)
KV = torch.randn(b, h,  s, head_dim)
idx_kv = torch.randn(b, h, s //2, head_dim)

_, indices = torch.topk(
        torch.randn(b, s, s//2),
        k=2,
        dim=-1,
    )

attention_sink = torch.randn(h)
doc_ids = None
sliding_window_size = 3
share_kv = True
selected_attention(
    Q,
    KV,
    idx_kv,
    indices, 
    attention_sink,
    doc_ids,
    sliding_window_size,
    share_kv,
)