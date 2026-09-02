"""Tests for the CuTeDSL (SM100) indexer backend.

Validates that the cute backend produces the same Top-K index sets as the
eager reference, and that FP64 scores at kernel-selected positions are at
or above the FP64 Top-K boundary within reduction tolerance.
"""

import math

import pytest
import torch

from attn_gym.sparse.indexer import index


def _skip_no_sm100():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for CuTe backend")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 (compute capability 10.0) required for CuTe backend")


def _reference_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Compute the indexer score matrix in the input dtype."""
    batch, queries, heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(heads * head_dim)
    dots = torch.einsum("bthd,bsd->bths", q, k)
    return (torch.relu(dots) * weights.unsqueeze(-1)).sum(dim=2) * scale


def _validate_indices(
    actual: torch.Tensor,
    reference_scores: torch.Tensor,
    topk: int,
    causal: bool,
) -> None:
    """Assert index sets match, tolerating ties at the Top-K boundary."""
    device = actual.device
    batch, queries = actual.shape[:2]
    candidates = reference_scores.shape[-1]

    actual_i64 = actual.to(torch.int64)
    valid = actual_i64 >= 0

    # Range check
    assert not torch.any(actual_i64 < -1), "index below -1"
    assert not torch.any(actual_i64 >= candidates), "index >= candidates"

    # Causal check
    if causal:
        row = torch.arange(queries, device=device).view(1, queries, 1)
        assert not torch.any(valid & (actual_i64 > row)), "causal violation"

    # Valid count check
    if causal:
        row = torch.arange(queries, device=device).view(1, queries, 1)
        expected_valid = torch.minimum(
            row + 1, torch.full_like(row, topk)
        ).expand(batch, -1, -1)
    else:
        expected_valid = torch.full(
            (batch, queries, 1), topk, device=device
        )
    assert torch.equal(valid.sum(-1, keepdim=True), expected_valid), "wrong valid count"

    # No duplicate valid indices
    sorted_actual = actual_i64.sort(-1).values
    dup = (sorted_actual[..., 1:] == sorted_actual[..., :-1]) & (
        sorted_actual[..., 1:] >= 0
    )
    assert not torch.any(dup), "duplicate valid index"

    # Compute reference topk from scores
    if causal:
        mask = torch.ones(queries, candidates, device=device, dtype=torch.bool).triu_(1)
        masked_scores = reference_scores.clone()
        masked_scores.masked_fill_(mask, float("-inf"))
    else:
        masked_scores = reference_scores

    ref_values, ref_indices = masked_scores.cpu().topk(topk, dim=-1, sorted=True)
    ref_values, ref_indices = ref_values.to(device), ref_indices.to(device)
    if causal:
        row = torch.arange(queries, device=device).view(1, queries, 1)
        ref_indices.masked_fill_(ref_indices > row, -1)

    sorted_ref = ref_indices.to(torch.int64).sort(-1).values
    exact_rows = (sorted_actual == sorted_ref).all(-1)
    if exact_rows.all():
        return

    # Causal rows with less than topk indices should just have all the indices
    if causal:
        row = torch.arange(queries, device=device).view(1, queries, 1)
        full_rows = expected_valid.squeeze(-1) == topk
        assert not torch.any((~exact_rows) & (~full_rows)), (
            "mismatch in a causal row with fewer than K valid keys"
        )

    # Basically, the idea is that the selected rows should be within tolerance of min(topk(cutoff))
    safe_indices = actual_i64.clamp_min(0)
    selected_values = masked_scores.gather(-1, safe_indices)
    selected_values = selected_values.masked_fill(~valid, float("inf"))
    min_selected = selected_values.min(-1).values
    kth_reference = ref_values[..., -1]
    boundary_gap = (kth_reference - min_selected).clamp_min(0)
    scale = torch.maximum(kth_reference.abs(), torch.ones_like(kth_reference))
    tolerance = 5.0e-4 * scale
    assert not torch.any((boundary_gap > tolerance) & (~exact_rows)), (
        f"mismatch not at boundary (max gap {boundary_gap.max().item():.6g})"
    )


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize(
    "batch,queries,heads,head_dim,topk",
    [
        (2, 128, 64, 128, 32),
        (2, 128, 128, 128, 128),
        (2, 64, 66, 128, 37),
        (2, 256, 128, 128, 64),
    ],
    ids=[
        "base",
        "topk_eq_seq",
        "odd_heads_odd_topk",
        "long_seq",
    ],
)
def test_cute_matches_eager(batch, queries, heads, head_dim, topk, causal):
    """Cute backend index set matches eager, up to boundary ties."""
    _skip_no_sm100()

    torch.manual_seed(2026)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn(batch, queries, heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, queries, head_dim, device=device, dtype=dtype)
    w = torch.randn(batch, queries, heads, device=device, dtype=dtype)

    actual = index(q, k, w, topk, causal=causal, backend="cute")
    scores = _reference_scores(q.float(), k.float(), w.float())

    assert actual.dtype == torch.int32
    assert actual.shape == (batch, queries, topk)
    _validate_indices(actual, scores, topk, causal)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "batch,queries,heads,head_dim,topk",
    [
        (2, 128, 64, 128, 32),
        (2, 128, 128, 128, 128),
        (2, 64, 66, 128, 37),
        (2, 65, 64, 128, 16),
        (2, 127, 64, 128, 16),
        (2, 129, 64, 128, 16),
        (2, 128, 64, 16, 32),
        (2, 128, 64, 32, 32),
        (2, 128, 64, 48, 32),
        (2, 128, 64, 64, 32),
        (2, 128, 64, 96, 32),
        (2, 128, 64, 128, 0),
        (2, 128, 64, 128, 1),
        (2, 128, 64, 128, 127),
        (2, 256, 64, 128, 129),
        (2, 512, 64, 128, 512),
    ],
    ids=[
        "base",
        "topk_eq_seq",
        "odd_heads_odd_topk",
        "candidates_65",
        "candidates_127",
        "candidates_129",
        "head_dim_16",
        "head_dim_32",
        "head_dim_48",
        "head_dim_64",
        "head_dim_96",
        "topk_0",
        "topk_1",
        "topk_127",
        "topk_129",
        "topk_512",
    ],
)
def test_cute_topk_scores_vs_fp64(batch, queries, heads, head_dim, topk, causal, dtype):
    """Every kernel-selected score is at or above the FP64 Top-K boundary.

    Inputs are generated in the tested low-precision dtype then promoted to
    FP64, so all paths see the same quantized values and the FP64 baseline
    isolates arithmetic error only. The tolerance accounts for the reduction
    chain: D (dot product) and H (weighted head sum).
    """
    _skip_no_sm100()

    torch.manual_seed(77)
    device = torch.device("cuda")

    q_lp = torch.randn(batch, queries, heads, head_dim, device=device, dtype=dtype)
    k_lp = torch.randn(batch, queries, head_dim, device=device, dtype=dtype)
    w_lp = torch.randn(batch, queries, heads, device=device, dtype=dtype)

    if topk == 0:
        cute_indices = index(q_lp, k_lp, w_lp, topk, causal=causal, backend="cute")
        assert cute_indices.shape == (batch, queries, 0)
        return

    scores_64 = _reference_scores(q_lp.double(), k_lp.double(), w_lp.double())

    # Apply causal mask
    if causal:
        causal_mask = torch.ones(
            queries, queries, device=device, dtype=torch.bool
        ).triu_(1)
        scores_64.masked_fill_(causal_mask, float("-inf"))

    # FP64 Top-K boundary: the Kth-largest score per row
    ref_topk_values, _ = scores_64.cpu().topk(topk, dim=-1, sorted=True)
    ref_topk_values = ref_topk_values.to(device)
    # [B, T] — the minimum score in the FP64 top-K set
    boundary = ref_topk_values[..., -1]

    # Get cute-selected indices
    cute_indices = index(q_lp, k_lp, w_lp, topk, causal=causal, backend="cute")
    valid = cute_indices >= 0
    safe_indices = cute_indices.to(torch.int64).clamp_min(0)

    # FP64 scores at the kernel-selected positions
    kernel_scores_64 = scores_64.gather(-1, safe_indices)  # [B, T, topk]

    # Tolerance from the reduction chain: dot(D) then sum(H)
    accumulation_eps = (
        (math.sqrt(head_dim) + math.sqrt(heads)) * torch.finfo(torch.float32).eps
    )
    scale = torch.maximum(boundary.abs(), torch.ones_like(boundary))
    tolerance = accumulation_eps * scale  # [B, T]

    # Every valid kernel-selected score must be >= boundary - tolerance
    min_allowed = (boundary - tolerance).unsqueeze(-1)  # [B, T, 1]
    violations = valid & (kernel_scores_64 < min_allowed)
    assert not torch.any(violations), (
        f"kernel selected a candidate below the FP64 boundary "
        f"(worst gap: {(min_allowed.expand_as(kernel_scores_64)[violations] - kernel_scores_64[violations]).max().item():.6g})"
    )
