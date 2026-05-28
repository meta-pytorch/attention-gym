import pytest
import torch

from attn_gym.masks.causal import causal_mask
from attn_gym.masks.document_mask import (
    generate_doc_mask_mod,
    generate_packed_causal_doc_mask_mod,
    length_to_offsets,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("lengths", [[3, 2, 5], [1, 4, 1, 7], [6]])
def test_packed_causal_doc_mask_matches_generic_causal_doc_mask(device, lengths):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    offsets = length_to_offsets(lengths, device=device)
    seq_len = offsets[-1].item()
    q_idx = torch.arange(seq_len, device=device, dtype=torch.int32).view(seq_len, 1)
    kv_idx = torch.arange(seq_len, device=device, dtype=torch.int32).view(1, seq_len)

    generic_mask_mod = generate_doc_mask_mod(causal_mask, offsets)
    packed_causal_mask_mod = generate_packed_causal_doc_mask_mod(offsets)

    assert torch.equal(
        packed_causal_mask_mod(0, 0, q_idx, kv_idx),
        generic_mask_mod(0, 0, q_idx, kv_idx),
    )
