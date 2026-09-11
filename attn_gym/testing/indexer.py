"""Shared operands and score-selection error checks for lightning indexers."""

import math

import torch

from attn_gym.sparse.indexer import lightning_indexer


def make_indexer_test_inputs(
    tokens: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    batch: int = 2,
    device: str | torch.device = "cuda",
    seed: int = 77,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate reproducible quantized operands with both signs of head weights."""
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch, tokens, heads, head_dim, device=device, dtype=dtype, generator=generator
    )
    k = torch.randn(batch, tokens, head_dim, device=device, dtype=dtype, generator=generator)
    weights = torch.randn(batch, tokens, heads, device=device, dtype=dtype, generator=generator)
    weights[..., 0] = weights[..., 0].abs() + 0.25
    if heads > 1:
        weights[..., 1] = -weights[..., 1].abs() - 0.25
    return q, k, weights


def assert_selection_regret_within(
    actual: torch.Tensor,
    reference: torch.Tensor,
    magnitude: torch.Tensor,
    valid: torch.Tensor,
    rounding_factor: float,
) -> None:
    """Bound maximum regret and relative RMS against reference rounding error.

    Normalize each query by its absolute scoring magnitude before aggregation,
    so high-magnitude rows cannot hide relative errors in low-magnitude rows.
    Padded entries must already have zero regret, but their positions need not
    match between outputs. Zero-score rows keep an exact zero pointwise budget
    when the reference regret is zero.
    """
    allowance = rounding_factor * magnitude
    assert (actual.amax(-1, keepdim=True) <= reference.amax(-1, keepdim=True) + allowance).all(), (
        "selection regret exceeds the pointwise error budget"
    )
    assert actual.mean() <= reference.mean() + allowance.mean(), (
        "selection regret exceeds the mean absolute error budget"
    )
    scale = torch.where(magnitude > 0, magnitude, 1)
    count = valid.sum()
    relative_rms = ((actual / scale).square().sum() / count).sqrt()
    reference_rms = ((reference / scale).square().sum() / count).sqrt()
    assert relative_rms <= reference_rms + rounding_factor, (
        f"selection relative RMS {relative_rms.item():.6g} exceeds "
        f"reference {reference_rms.item():.6g} + rounding allowance {rounding_factor:.6g}"
    )


def assert_indexer_selection(
    actual: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool,
) -> None:
    """Check index invariants and FP64 boundary regret against eager's regret.

    Absolute products bound FP32 dot, weighting, head reduction and scaling
    errors without cancellation shrinking the budget. Twice that score error
    bounds a selection boundary swap; scores have no FP16/BF16 output rounding.
    """
    batch, tokens, heads, dim = q.shape
    assert actual.shape == (batch, tokens, topk)
    assert actual.dtype == torch.int32
    assert actual.device == q.device
    assert actual.is_contiguous()
    assert not actual.requires_grad
    valid = actual >= 0
    assert not ((actual < -1) | (actual >= tokens)).any()
    row = torch.arange(tokens, device=q.device).view(1, tokens, 1)
    counts = (row + 1).clamp_max(topk) if causal else torch.full_like(row, topk)
    assert torch.equal(valid.sum(-1, keepdim=True), counts.expand(batch, -1, -1))
    if causal:
        assert not (valid & (actual > row)).any()
    ordered = actual.sort(-1).values
    assert not ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any()
    if topk == 0:
        return

    q64, k64, w64 = q.double(), k.double(), weights.double()
    dots = q64.permute(0, 2, 1, 3) @ k64.transpose(-1, -2).unsqueeze(1)
    scores = (dots.relu() * w64.transpose(1, 2).unsqueeze(-1)).sum(1)
    scores /= math.sqrt(heads * dim)
    assert torch.isfinite(scores).all()
    absolute_dots = q64.abs().permute(0, 2, 1, 3) @ k64.abs().transpose(-1, -2).unsqueeze(1)
    magnitude = (absolute_dots * w64.abs().transpose(1, 2).unsqueeze(-1)).sum(1)
    magnitude /= math.sqrt(heads * dim)
    if causal:
        future = torch.arange(tokens, device=q.device).view(1, 1, tokens) > row
        scores.masked_fill_(future, -torch.inf)
        magnitude.masked_fill_(future, 0)
    boundary = scores.sort(-1, descending=True).values.gather(
        -1, (counts - 1).expand(batch, -1, -1)
    )
    eager = lightning_indexer(q, k, weights, topk, causal=causal, impl="reference")
    actual_scores = scores.gather(-1, actual.long().clamp_min(0))
    eager_scores = scores.gather(-1, eager.long().clamp_min(0))
    actual_error = (boundary - actual_scores).clamp_min(0).masked_fill(~valid, 0)
    eager_error = (boundary - eager_scores).clamp_min(0).masked_fill(eager < 0, 0)
    reduction_eps = (dim + heads + 2) * torch.finfo(torch.float32).eps
    assert_selection_regret_within(
        actual_error,
        eager_error,
        magnitude.amax(-1, keepdim=True),
        valid,
        2 * reduction_eps / (1 - reduction_eps),
    )
