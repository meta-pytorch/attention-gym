"""Shared operands and score-selection error checks for lightning indexers."""

import math
from typing import NamedTuple

import torch

from attn_gym.sparse.indexer.impl.reference import score_rows

# Bound each [B, H, query_tile, S] FP64 temporary, not the logical test shape.
_SCORE_TILE_BYTES = 64 * 1024 * 1024


def make_paged_indexer_test_inputs(
    query_lengths=(1, 3, 0, 2),
    key_lengths=(4099, 37, 0, 514),
    *,
    heads=4,
    head_dim=128,
    page_size=64,
    compress_ratio=4,
    dtype=torch.bfloat16,
    device="cuda",
    tail=0,
    seed=77,
):
    """Paged operands with shuffled physical pages and unused page-table entries."""
    generator = torch.Generator(device=device).manual_seed(seed)
    sequences = len(query_lengths)
    pages_per_sequence = max(
        1, (max(key_lengths, default=0) // compress_ratio + page_size - 1) // page_size
    )
    pages = sequences * pages_per_sequence
    tokens = sum(query_lengths) + tail
    q = torch.randn(tokens, heads, head_dim, device=device, dtype=dtype, generator=generator)
    k = torch.randn(pages, page_size, head_dim, device=device, dtype=dtype, generator=generator)
    w = torch.randn(tokens, heads, device=device, dtype=dtype, generator=generator)
    table = torch.randperm(pages, device=device, generator=generator).int().reshape(sequences, -1)
    for sequence, length in enumerate(key_lengths):
        used = (length // compress_ratio + page_size - 1) // page_size
        table[sequence, used:] = -1
    cu = torch.tensor([0, *query_lengths], dtype=torch.int32, device=device).cumsum(0).int()
    lengths = torch.tensor(key_lengths, dtype=torch.int32, device=device)
    return q, k, w, table, cu, lengths


@torch.no_grad()
def indexer_reference_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Compute input-dtype reference scores with bounded query-tile intermediates.

    The returned [B, T, S] matrix is still quadratic; query tiling avoids retaining
    the much larger [B, T, H, S] dot/ReLU/product buffers. Keep the full head and
    channel reductions and their original scaling order.
    """
    batch, queries, heads = q.shape[:3]
    candidates = k.shape[1]
    bytes_per_row = max(1, batch * heads * candidates * q.element_size())
    tile_rows = max(1, _SCORE_TILE_BYTES // bytes_per_row)
    scores = q.new_empty(batch, queries, candidates)
    for start in range(0, queries, tile_rows):
        rows = slice(start, start + tile_rows)
        scores[:, rows] = score_rows(q[:, rows], k, weights[:, rows])
    return scores


def make_indexer_test_inputs(
    tokens: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    *,
    batch: int = 2,
    compress_ratio: int = 1,
    device: str | torch.device = "cuda",
    seed: int = 77,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate reproducible quantized operands with both signs of head weights."""
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        batch, tokens, heads, head_dim, device=device, dtype=dtype, generator=generator
    )
    candidates = tokens // compress_ratio
    k = torch.randn(batch, candidates, head_dim, device=device, dtype=dtype, generator=generator)
    weights = torch.randn(batch, tokens, heads, device=device, dtype=dtype, generator=generator)
    weights[..., 0] = weights[..., 0].abs() + 0.25
    if heads > 1:
        weights[..., 1] = -weights[..., 1].abs() - 0.25
    return q, k, weights


class ScaledIndexerInputs(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    weights: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor


def make_indexer_mxfp8_test_inputs(
    tokens: int,
    heads: int,
    head_dim: int,
    weights_dtype: torch.dtype,
    *,
    batch: int = 2,
    compress_ratio: int = 1,
    device: str | torch.device = "cuda",
    seed: int = 77,
) -> ScaledIndexerInputs:
    """E4M3 data and E8M0 scales varying across queries, heads, keys and D groups."""
    q, k, weights = make_indexer_test_inputs(
        tokens,
        heads,
        head_dim,
        torch.bfloat16,
        batch=batch,
        compress_ratio=compress_ratio,
        device=device,
        seed=seed,
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    if weights_dtype == torch.float32:
        weights = torch.randn(
            weights.shape, device=device, dtype=weights_dtype, generator=generator
        )
    q_exponents = torch.randint(
        -3, 3, (*q.shape[:-1], head_dim // 32), device=device, generator=generator
    )
    k_exponents = torch.randint(
        -3, 3, (*k.shape[:-1], head_dim // 32), device=device, generator=generator
    )
    q_scale = torch.pow(2.0, q_exponents).to(torch.float8_e8m0fnu)
    k_scale = torch.pow(2.0, k_exponents).to(torch.float8_e8m0fnu)
    # MX scales have no zero encoding; zero data groups retain positive scales.
    q[:, 0, :, :32] = 0
    k[:, :1, :32] = 0
    return ScaledIndexerInputs(
        q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), weights, q_scale, k_scale
    )


def assert_mxfp8_score_row(
    actual: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
) -> None:
    """Compare one live score row to independent FP64 dequantization and block-scaled eager."""
    heads, dim = q.shape
    q64 = q.double() * q_scale.double().repeat_interleave(32, -1)
    k64 = k.double() * k_scale.double().repeat_interleave(32, -1)
    w64 = weights.double()[:, None]
    norm = math.sqrt(heads * dim)
    expected = ((q64 @ k64.T).relu() * w64).sum(0) / norm
    magnitude = ((q64.abs() @ k64.abs().T) * w64.abs()).sum(0) / norm
    eager = score_rows(q[None].float(), k.float(), weights[None].float(), q_scale[None], k_scale)[
        0
    ]
    assert torch.isfinite(eager).all(), "nonfinite FP32 score baseline"
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    error = (actual.double() - expected).abs()
    # D dot terms, H/4 head-reduction chain, scale updates and epilogue rounding.
    eps = torch.finfo(torch.float32).eps
    n_eps = (dim + heads // 4 + 8) * eps
    assert (error <= n_eps / (1 - n_eps) * magnitude).all(), (error.max(), magnitude.max())
    # Relative RMS versus eager plus an FP32 reduction/rounding allowance. Compare
    # norms before dividing by the common denominator, which can be zero for empty signal.
    eager_error = eager.double() - expected
    allowance = (math.sqrt(dim) + math.sqrt(heads) + 4) * eps
    assert error.norm() <= eager_error.norm() + allowance * expected.norm()


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


def _selection_errors(
    actual: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    visible: torch.Tensor,
    causal: bool,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score one query tile while keeping the full candidate and head reductions."""
    _, _, heads, dim = q.shape
    q64, k64, w64 = q.double(), k.double(), weights.double()
    if q_scale is not None:
        q64 = (q64.unflatten(-1, (dim // 32, 32)) * q_scale.double().unsqueeze(-1)).flatten(-2)
    if k_scale is not None:
        k64 = (k64.unflatten(-1, (dim // 32, 32)) * k_scale.double().unsqueeze(-1)).flatten(-2)
    dots = q64.permute(0, 2, 1, 3) @ k64.transpose(-1, -2).unsqueeze(1)
    scores = (dots.relu() * w64.transpose(1, 2).unsqueeze(-1)).sum(1)
    scores /= math.sqrt(heads * dim)
    assert torch.isfinite(scores).all()
    # sum_h |w_h| (|q_h| . |k_s|) == (sum_h |w_h| |q_h|) . |k_s|: fold heads before the product.
    magnitude = (w64.abs().unsqueeze(-1) * q64.abs()).sum(2) @ k64.abs().transpose(-1, -2)
    magnitude /= math.sqrt(heads * dim)

    # Compare eager and fused rounding errors against the independent FP64 scores above.
    accum_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    eager_scores = score_rows(
        q.to(accum_dtype), k.to(accum_dtype), weights.to(accum_dtype), q_scale, k_scale
    )
    if causal:
        future = torch.arange(k.shape[1], device=q.device).view(1, 1, -1) >= visible
        scores.masked_fill_(future, -torch.inf)
        magnitude.masked_fill_(future, 0)
        eager_scores.masked_fill_(future, -torch.inf)
    width = min(actual.shape[-1], k.shape[1])
    best = scores.topk(width).values
    eager = eager_scores.topk(width).indices

    def regret(ids: torch.Tensor) -> torch.Tensor:
        selected = scores.gather(-1, ids.long().clamp_min(0)).masked_fill(ids < 0, -torch.inf)
        chosen = selected.sort(descending=True).values[..., :width]
        return (best - chosen).masked_fill(best == -torch.inf, 0)

    return regret(actual), regret(eager), magnitude.amax(-1, keepdim=True)


@torch.no_grad()
def assert_indexer_selection(
    actual: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int = 1,
    *,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> None:
    """Check index invariants and sorted FP64 Top-K score gaps against eager's gaps.

    Absolute products bound FP32 dot, weighting, head reduction and scaling
    errors without cancellation shrinking the budget. Twice that score error
    bounds a selection boundary swap; scores have no FP16/BF16 output rounding.
    """
    batch, tokens, heads, dim = q.shape
    candidates = k.shape[1]
    assert candidates == tokens // compress_ratio
    assert actual.shape == (batch, tokens, topk)
    assert actual.dtype == torch.int32
    assert actual.device == q.device
    assert actual.is_contiguous()
    assert not actual.requires_grad
    valid = actual >= 0
    assert not ((actual < -1) | (actual >= candidates)).any()
    row = torch.arange(tokens, device=q.device).view(1, tokens, 1)
    visible = (row + 1) // compress_ratio
    width = min(topk, candidates)
    counts = visible.clamp_max(width) if causal else torch.full_like(row, width)
    assert torch.equal(valid.sum(-1, keepdim=True), counts.expand(batch, -1, -1))
    if causal:
        assert not (valid & (actual >= visible)).any()
    ordered = actual.sort(-1).values
    assert not ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any()
    if width == 0:
        return

    tile_rows = max(1, _SCORE_TILE_BYTES // (batch * heads * candidates * 8))
    actual_error = torch.empty_like(actual[..., :width], dtype=torch.float64)
    eager_error = torch.empty_like(actual_error)
    magnitude = torch.empty((batch, tokens, 1), device=q.device, dtype=torch.float64)
    for start in range(0, tokens, tile_rows):
        rows = slice(start, start + tile_rows)
        actual_error[:, rows], eager_error[:, rows], magnitude[:, rows] = _selection_errors(
            actual[:, rows],
            q[:, rows],
            k,
            weights[:, rows],
            visible[:, rows],
            causal,
            None if q_scale is None else q_scale[:, rows],
            k_scale,
        )
    scale_ops = int(q_scale is not None) + int(k_scale is not None)
    reduction_eps = (dim + heads + 2 + scale_ops) * torch.finfo(torch.float32).eps
    # Retain only O(B*T*topk) regrets, not O(B*H*T*S) scores. Check once over all
    # rows: per-tile mean/RMS budgets would change acceptance, especially at the tail.
    assert_selection_regret_within(
        actual_error,
        eager_error,
        magnitude,
        valid,
        2 * reduction_eps / (1 - reduction_eps),
    )


def assert_paged_indexer_selection(args, actual, topk, ratio=4, positions=None):
    """Check paged IDs, causal visibility, padding, and FP64 selection regret."""
    q, cache, w, table, cu, lengths = args
    assert actual.shape == (q.shape[0], topk)
    assert actual.dtype == torch.int32 and actual.is_contiguous()
    assert not actual.requires_grad
    torch.testing.assert_close(actual[int(cu[-1]) :], torch.full_like(actual[int(cu[-1]) :], -1))
    for sequence, length in enumerate(lengths.tolist()):
        begin, end = cu[sequence : sequence + 2].tolist()
        for row in range(begin, end):
            position = (
                length - (end - begin) + row - begin if positions is None else int(positions[row])
            )
            count = min(max(0, (position + 1) // ratio), length // ratio)
            selected = actual[row]
            valid = selected >= 0
            assert int(valid.sum()) == min(topk, count)
            assert (selected[valid] < count).all() and (selected >= -1).all()
            assert selected[valid].unique().numel() == valid.sum()
            assert not valid[min(topk, count) :].any()
            if count == 0 or topk == 0:
                continue
            logical = torch.arange(count, device=q.device)
            keys = cache[
                table[sequence, logical // cache.shape[1]].long(), logical % cache.shape[1]
            ]
            dots = q[row].double() @ keys.double().T
            scores = (dots.relu() * w[row].double()[:, None]).sum(0)
            best = scores.topk(min(topk, count)).values
            chosen = scores[selected[valid].long()].sort(descending=True).values
            error = (best - chosen)[None, :]
            magnitude = (
                ((q[row].double().abs() @ keys.double().abs().T) * w[row].double().abs()[:, None])
                .sum(0)
                .max()
                .reshape(1, 1)
            )
            eps = (q.shape[1] + q.shape[2] + 2) * torch.finfo(torch.float32).eps
            assert_selection_regret_within(
                error,
                torch.zeros_like(error),
                magnitude,
                torch.ones_like(error, dtype=torch.bool),
                2 * eps / (1 - eps),
            )
