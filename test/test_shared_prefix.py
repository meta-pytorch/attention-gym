import math

import pytest
import torch

from attn_gym.masks.document_mask import length_to_offsets
from attn_gym.masks.shared_prefix import generate_shared_prefix_mask_mod


def _dense_attention(q, k, v, mask):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared_prefix_matches_duplicated_causal_samples_cuda():
    torch.manual_seed(0)
    device = "cuda"

    prefix_response_docs = [
        (range(0, 3), range(3, 5)),
        (range(0, 3), range(5, 7)),
        (range(0, 3), range(7, 9)),
        (range(9, 12), range(12, 14)),
        (range(9, 12), range(14, 16)),
        (range(9, 12), range(16, 18)),
    ]
    document_lengths = [3, 2, 2, 2, 3, 2, 2, 2]
    offsets = length_to_offsets(document_lengths, device)
    prefix_document_id = torch.tensor([0, 0, 0, 0, 4, 4, 4, 4], device=device)
    mask_mod = generate_shared_prefix_mask_mod(offsets, prefix_document_id)

    H, S, D = 2, sum(document_lengths), 8
    q = torch.randn(H, S, D, device=device)
    k = torch.randn(H, S, D, device=device)
    v = torch.randn(H, S, D, device=device)

    q_idx = torch.arange(S, device=device).view(-1, 1)
    kv_idx = torch.arange(S, device=device).view(1, -1)
    shared_prefix_mask = mask_mod(torch.zeros_like(q_idx), torch.zeros_like(q_idx), q_idx, kv_idx)
    shared_prefix_out = _dense_attention(q, k, v, shared_prefix_mask)

    causal_mask = torch.ones(5, 5, dtype=torch.bool, device=device).tril()
    for prefix_indices, response_indices in prefix_response_docs:
        logical_indices = list(prefix_indices) + list(response_indices)
        duplicated_out = _dense_attention(
            q[:, logical_indices, :],
            k[:, logical_indices, :],
            v[:, logical_indices, :],
            causal_mask,
        )
        torch.testing.assert_close(shared_prefix_out[:, logical_indices, :], duplicated_out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_q8_row_matches_prefix_a_plus_response_a2_causal_sample_cuda():
    device = "cuda"
    document_lengths = [3, 2, 2, 2, 3, 2, 2, 2]
    offsets = length_to_offsets(document_lengths, device)
    prefix_document_id = torch.tensor([0, 0, 0, 0, 4, 4, 4, 4], device=device)
    mask_mod = generate_shared_prefix_mask_mod(offsets, prefix_document_id)

    q_idx = torch.tensor([[8]], device=device)
    kv_idx = torch.arange(sum(document_lengths), device=device).view(1, -1)
    q8_mask = mask_mod(torch.zeros_like(q_idx), torch.zeros_like(q_idx), q_idx, kv_idx).squeeze(0)

    expected = torch.zeros(sum(document_lengths), dtype=torch.bool, device=device)
    expected[[0, 1, 2, 7, 8]] = True

    assert torch.equal(q8_mask, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_shared_prefix_rejects_prefix_document_id_shape_mismatch_cuda():
    device = "cuda"
    offsets = length_to_offsets([3, 2, 2], device)

    with pytest.raises(AssertionError):
        generate_shared_prefix_mask_mod(offsets, torch.tensor([0, 0], device=device))
