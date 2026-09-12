"""Tests for the indexer reference implementation."""

import pytest
import torch

from attn_gym.sparse.indexer import lightning_indexer


def test_basic_correctness():
    """Tiny case with hardcoded expected winner.

    q[0,0,0,:] = [1, 0], one head, two candidates:
      k[0,0,:] = [1, 0]  -> dot = 1 -> relu = 1
      k[0,1,:] = [0, 1]  -> dot = 0 -> relu = 0
    score = w * relu(dot) / sqrt(H*D) -> candidate 0 wins.
    """
    q = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])  # [1,2,1,2]
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # [1,2,2]
    w = torch.tensor([[[1.0], [1.0]]])  # [1,2,1]

    actual = lightning_indexer(q, k, w, topk=1, impl="reference")

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

    actual = lightning_indexer(q, k, w, topk, causal=True, impl="reference")

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

    out = lightning_indexer(q, k, w, K, impl="reference")

    assert out.dtype == torch.int32
    assert out.shape == (B, T, K)


def test_compressed_candidates_causal_masking():
    """Each candidate summarizes compress_ratio tokens; a row sees only completed windows.

    With T=10 and compress_ratio=4 there are S=2 candidates: tokens 0..3 and 4..7.
    Tokens 8..9 form no candidate. Query t sees (t + 1) // 4 candidates.
    """
    torch.manual_seed(0)
    T, ratio, topk = 10, 4, 2
    q = torch.randn(1, T, 2, 8, dtype=torch.float64)
    k = torch.randn(1, T // ratio, 8, dtype=torch.float64)
    w = torch.ones(1, T, 2, dtype=torch.float64)

    actual = lightning_indexer(q, k, w, topk, causal=True, compress_ratio=ratio, impl="reference")

    for t in range(T):
        row = actual[0, t]
        visible = (t + 1) // ratio
        valid = row[row >= 0]
        assert len(valid) == min(visible, topk), f"t={t}"
        assert (valid < visible).all(), f"t={t}: candidate not yet complete"
        assert (row == -1).sum() == topk - len(valid), f"t={t}"
    assert (actual[0, :3] == -1).all()
    assert set(actual[0, 7].tolist()) == {0, 1}


def test_compressed_candidates_rank_within_visible_prefix():
    """Selection ranks by score among the visible candidates, not merely masks them."""
    torch.manual_seed(1)
    T, ratio = 13, 4
    q = torch.randn(1, T, 2, 8, dtype=torch.float64)
    k = torch.randn(1, T // ratio, 8, dtype=torch.float64)
    w = torch.randn(1, T, 2, dtype=torch.float64).abs() + 0.25
    scores = (torch.einsum("bthd,bsd->bths", q, k).relu() * w.unsqueeze(-1)).sum(2)[0]

    actual = lightning_indexer(q, k, w, 1, causal=True, compress_ratio=ratio, impl="reference")

    assert actual[0, 7, 0] == scores[7, :2].argmax()
    assert actual[0, 11, 0] == scores[11, :3].argmax()
    assert actual[0, 12, 0] == scores[12, :3].argmax()


@pytest.mark.parametrize(
    "kwargs,candidates,error,message",
    [
        ({"causal": True, "compress_ratio": 4}, 3, ValueError, "S = T // compress_ratio"),
        ({"causal": True, "compress_ratio": 3}, 2, ValueError, "S = T // compress_ratio"),
        ({"compress_ratio": 4}, 2, ValueError, "pass causal=True"),
        ({"causal": True, "compress_ratio": 0}, 2, ValueError, "must be positive"),
        ({"causal": True, "compress_ratio": 4.0}, 2, TypeError, "Python int"),
        ({"causal": True}, 2, ValueError, "S = T // compress_ratio"),
    ],
    ids=["too_many", "not_ratio", "noncausal", "zero", "float", "square_required"],
)
def test_compress_ratio_validation(kwargs, candidates, error, message):
    """Reject mismatched candidate counts, noncausal ratios, and non-int ratios."""
    q = torch.randn(1, 10, 2, 8)
    k = torch.randn(1, candidates, 8)
    w = torch.randn(1, 10, 2)
    with pytest.raises(error, match=message):
        lightning_indexer(q, k, w, 1, impl="reference", **kwargs)
