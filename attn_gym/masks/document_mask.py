"""Generates a document causal attention mask based on a document ID tensor"""

import random
from typing import List, Union

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature, noop_mask
from attn_gym.masks import causal_mask


def _offsets_to_doc_ids_tensor(offsets):
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def length_to_offsets(lengths: List[int], device: Union[str, torch.device]) -> Tensor:
    """Converts a list of lengths to a list of offsets.

    Args:
        lengths: A list of lengths.

    """
    offsets = [0]
    offsets.extend(lengths)
    offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def generate_doc_mask_mod(mask_mod: _mask_mod_signature, offsets: Tensor) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        offsets: This tensor should be of shape(num_documents + 1)
            this should contain the cumulative counts of document tokens.
            e.g. if you have 3 documents of length 2, 4, 3 then
            offsets = [0, 2, 6, 9]

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_mask_mod(b, h, q_idx, kv_idx):
        same_doc = document_id[q_idx] == document_id[kv_idx]
        q_logical = q_idx - offsets[document_id[q_idx]]
        kv_logical = kv_idx - offsets[document_id[kv_idx]]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask

    return doc_mask_mod


def generate_packed_causal_doc_mask_mod(offsets: Tensor) -> _mask_mod_signature:
    """Generates a causal document mask for packed sequences sharing one offset layout.

    Args:
        offsets: Cumulative document token counts with shape ``(num_documents + 1,)``.

    Note:
        This is equivalent to ``generate_doc_mask_mod(causal_mask, offsets)`` for packed
        causal attention, but expresses each query row as one contiguous KV interval.
    """
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def doc_causal_mask_mod(b, h, q_idx, kv_idx):
        doc_start = offsets[document_id[q_idx]]
        return (kv_idx >= doc_start) & causal_mask(b, h, q_idx, kv_idx)

    return doc_causal_mask_mod


def generate_random_lengths(total_length, num_documents):
    # Initialize all lengths to 1 to ensure each document has at least one token
    lengths = [1] * num_documents
    remaining_length = total_length - num_documents

    # Randomly distribute the remaining length
    for _ in range(remaining_length):
        index = random.randint(0, num_documents - 1)
        lengths[index] += 1

    return lengths


def main(device: str = "cpu", causal: bool = True):
    """Visualize the attention scores of document causal mask mod.

    Args:
        device (str): Device to use for computation. Defaults to "cpu".
    """
    from attn_gym import visualize_attention_scores

    random.seed(0)

    max_seq_len, doc_count = 21, 4
    B, H, SEQ_LEN, HEAD_DIM = 1, 1, max_seq_len, 8

    lengths = generate_random_lengths(max_seq_len, doc_count)
    offsets = length_to_offsets(lengths, device)

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    if causal:
        base_mask_mod = causal_mask
    else:
        base_mask_mod = noop_mask
    document_causal_mask = generate_doc_mask_mod(base_mask_mod, offsets)

    visualize_attention_scores(
        query,
        key,
        mask_mod=document_causal_mask,
        device=device,
        name="document_causal_mask",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")

    CLI(main)
