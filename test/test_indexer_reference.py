"""Tests for the indexer reference implementation."""

import torch

from attn_gym.sparse.indexer import index


def test_basic_correctness():
    """Tiny case with hardcoded expected winner.

    q[0,0,0,:] = [1, 0], one head, two candidates:
      k[0,0,:] = [1, 0]  -> dot = 1 -> relu = 1
      k[0,1,:] = [0, 1]  -> dot = 0 -> relu = 0
    score = w * relu(dot) / sqrt(H*D) -> candidate 0 wins.
    """
    q = torch.tensor([[[[1.0, 0.0]],
                        [[0.0, 1.0]]]])      # [1,2,1,2]
    k = torch.tensor([[[1.0, 0.0],
                        [0.0, 1.0]]])         # [1,2,2]
    w = torch.tensor([[[1.0], [1.0]]])        # [1,2,1]

    actual = index(q, k, w, topk=1)

    assert actual.shape == (1, 2, 1)
    # t=0: dot with k0=1, k1=0 -> winner is 0
    assert actual[0, 0, 0].item() == 0
    # t=1: dot with k0=0, k1=1 -> winner is 1
    assert actual[0, 1, 0].item() == 1


def test_causal_masking():
    """Causal: no index exceeds query position; overflow slots get -1 sentinels."""
    B, T, H, D = 1, 4, 2, 3
    topk = 3

    # With topk=3, query at t=0 has 1 valid candidate, t=1 has 2, t=2 has 3.
    # Overflow slots should be -1.
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, D)
    w = torch.ones(B, T, H)

    actual = index(q, k, w, topk, causal=True)

    for t in range(T):
        row = actual[0, t, :]  # [topk]
        valid = row[row >= 0]
        sentinels = row[row < 0]

        # Valid indices must not exceed query position
        assert (valid <= t).all(), f"t={t}: index exceeds causal bound"
        # Valid indices must be unique
        assert valid.unique().shape == valid.shape, f"t={t}: duplicate valid indices"

        # Number of valid candidates at position t is min(t+1, topk)
        expected_valid = min(t + 1, topk)
        assert len(valid) == expected_valid, (
            f"t={t}: expected {expected_valid} valid, got {len(valid)}"
        )
        # Remaining slots are -1
        assert (sentinels == -1).all(), f"t={t}: non -1 sentinel"


def test_output_dtype_and_shape():
    """Output must be int32 with shape [B, T, topk]."""
    B, T, H, D, K = 2, 6, 4, 8, 3
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, D)
    w = torch.randn(B, T, H)

    out = index(q, k, w, K)

    assert out.dtype == torch.int32
    assert out.shape == (B, T, K)


def test_mode_auto_matches_prefill():
    """mode="auto" and mode="prefill" produce identical results (auto defaults to prefill)."""
    B, T, H, D, K = 2, 6, 4, 8, 3
    torch.manual_seed(0)
    q = torch.randn(B, T, H, D)
    k = torch.randn(B, T, D)
    w = torch.randn(B, T, H)

    out_auto = index(q, k, w, K, mode="auto")
    out_prefill = index(q, k, w, K, mode="prefill")
    out_default = index(q, k, w, K)

    assert torch.equal(out_auto, out_prefill)
    assert torch.equal(out_auto, out_default)
