"""Shared-prefix masks for packed prompt/continuation documents."""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature

from attn_gym.masks.causal import causal_mask
from attn_gym.masks.document_mask import _offsets_to_doc_ids_tensor, length_to_offsets


def generate_shared_prefix_mask_mod(
    offsets: Tensor, prefix_document_id: Tensor
) -> _mask_mod_signature:
    """Generates a packed mask for shared-prefix between multiple responses.

    Args:
        offsets: Cumulative document offsets with shape ``(num_documents + 1,)``.
        prefix_document_id: Maps each document to its prefix document. Prefix
            documents should map to themselves.

    Note:
        This represents each logical sample as ``prefix_document + continuation_document``
        without physically duplicating prefix tokens. Prefix tokens are causal. Continuation documents attend to their
        full prefix document and causally within themselves. Attention stays within prefix + documents
    """
    assert offsets.ndim == 1
    assert prefix_document_id.ndim == 1
    assert prefix_document_id.device == offsets.device
    assert prefix_document_id.numel() + 1 == offsets.numel()
    document_id = _offsets_to_doc_ids_tensor(offsets)

    def shared_prefix_mask_mod(b, h, q_idx, kv_idx):
        q_document = document_id[q_idx]
        q_document_start = offsets[q_document]
        q_prefix_document = prefix_document_id[q_document]
        q_prefix_start = offsets[q_prefix_document]
        q_prefix_end = offsets[q_prefix_document + 1]
        same_document_causal = (kv_idx >= q_document_start) & causal_mask(b, h, q_idx, kv_idx)
        prefix_document = (
            (q_document != q_prefix_document)
            & (kv_idx >= q_prefix_start)
            & (kv_idx < q_prefix_end)
        )
        return same_document_causal | prefix_document

    shared_prefix_mask_mod.__name__ = "shared_prefix_mask_mod"
    return shared_prefix_mask_mod


def main(device: str = "cpu"):
    """Visualize grouped shared-prefix masking."""
    from attn_gym import visualize_attention_scores

    document_lengths = [3, 2, 2, 2, 3, 2, 2, 2]
    document_offsets = length_to_offsets(document_lengths, device)
    prefix_document_id = torch.tensor([0, 0, 0, 0, 4, 4, 4, 4], device=device)
    mask_mod = generate_shared_prefix_mask_mod(document_offsets, prefix_document_id)

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, sum(document_lengths), 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    visualize_attention_scores(
        query,
        key,
        mask_mod=mask_mod,
        device=device,
        name="shared_prefix_mask",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")

    CLI(main)
