"""Bounded oracle regressions against the original, unchunked scoring formulas."""

import math
from unittest.mock import Mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.testing import indexer


def unchunked_errors(
    actual: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, tokens, heads, dim = q.shape
    q64, k64, w64 = q.double(), k.double(), weights.double()
    dots = q64.permute(0, 2, 1, 3) @ k64.transpose(-1, -2).unsqueeze(1)
    scores = (dots.relu() * w64.transpose(1, 2).unsqueeze(-1)).sum(1)
    scores /= math.sqrt(heads * dim)
    absolute_dots = q64.abs().permute(0, 2, 1, 3) @ k64.abs().transpose(-1, -2).unsqueeze(1)
    magnitude = (absolute_dots * w64.abs().transpose(1, 2).unsqueeze(-1)).sum(1)
    magnitude /= math.sqrt(heads * dim)
    visible = (torch.arange(tokens, device=q.device).view(1, tokens, 1) + 1) // compress_ratio
    counts = visible.clamp_max(topk) if causal else torch.full_like(visible, topk)
    if causal:
        future = torch.arange(k.shape[1], device=q.device).view(1, 1, -1) >= visible
        scores.masked_fill_(future, -torch.inf)
        magnitude.masked_fill_(future, 0)
    boundary = scores.sort(-1, descending=True).values.gather(
        -1, (counts - 1).clamp_min(0).expand(batch, -1, -1)
    )
    eager = lightning_indexer(
        q, k, weights, topk, causal=causal, compress_ratio=compress_ratio, impl="reference"
    )
    actual_scores = scores.gather(-1, actual.long().clamp_min(0))
    eager_scores = scores.gather(-1, eager.long().clamp_min(0))
    return (
        (boundary - actual_scores).clamp_min(0).masked_fill(actual < 0, 0),
        (boundary - eager_scores).clamp_min(0).masked_fill(eager < 0, 0),
        magnitude.amax(-1, keepdim=True),
    )


DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    ),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "causal,ratio,topk",
    [(False, 1, 0), (False, 1, 1), (False, 1, 17), (True, 1, 5), (True, 3, 1), (True, 3, 5)],
)
@pytest.mark.parametrize("ties", [False, True])
def test_tiled_oracle_matches_unchunked(monkeypatch, device, dtype, causal, ratio, topk, ties):
    inputs = indexer.make_indexer_test_inputs(17, 3, 8, dtype, device=device, compress_ratio=ratio)
    q, k, _ = inputs
    q.mul_(torch.logspace(-3, 3, 17, device=device).view(1, 17, 1, 1))
    if ties:
        k.zero_()
    actual = lightning_indexer(
        *inputs, topk, causal=causal, compress_ratio=ratio, impl="reference"
    ).roll(1, dims=-1)
    # Four full tiles and a one-row tail; compression windows cross tile boundaries.
    monkeypatch.setattr(indexer, "_SCORE_TILE_BYTES", 2 * 3 * 4 * k.shape[1] * 8)
    check = Mock(wraps=indexer.assert_selection_regret_within)
    monkeypatch.setattr(indexer, "assert_selection_regret_within", check)
    indexer.assert_indexer_selection(actual, *inputs, topk, causal, ratio)
    if topk == 0:
        check.assert_not_called()
        return
    check.assert_called_once()
    expected = unchunked_errors(actual, *inputs, topk, causal, ratio)
    for result, reference in zip(check.call_args.args[:3], expected):
        torch.testing.assert_close(result, reference, atol=1e-10, rtol=1e-12)
    assert torch.equal(check.call_args.args[3], actual >= 0)
    eps = (8 + 3 + 2) * torch.finfo(torch.float32).eps
    assert check.call_args.args[4] == 2 * eps / (1 - eps)


@pytest.mark.parametrize("device", DEVICES)
def test_all_scoring_matmuls_are_tiled(monkeypatch, device):
    inputs = indexer.make_indexer_test_inputs(17, 3, 8, torch.bfloat16, device=device)
    actual = lightning_indexer(*inputs, 5, impl="reference")
    elements = 2 * 3 * 4 * 17
    monkeypatch.setattr(indexer, "_SCORE_TILE_BYTES", elements * 8)
    products = []

    class BoundedMatmul(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if func == torch.ops.aten.bmm.default:
                assert result.numel() <= elements
                products.append((result.dtype, result.numel()))
            return result

    with BoundedMatmul():
        indexer.assert_indexer_selection(actual, *inputs, 5, False)
    assert len(products) == 15  # FP64 dot/magnitude and FP32 eager, for each of five tiles.
    assert {dtype for dtype, _ in products} == {torch.float32, torch.float64}
    assert products[-1][1] == elements // 4


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("causal,ratio", [(False, 1), (True, 1), (True, 3)])
@pytest.mark.parametrize("corruption", ["regret", "duplicate", "padding", "range", "future"])
def test_tiled_oracle_rejects_corrupt_selection(monkeypatch, device, causal, ratio, corruption):
    if corruption == "future" and not causal:
        pytest.skip("dense selection has no future candidates")
    q = torch.ones(1, 17, 1, 1, device=device)
    k = torch.arange(17 // ratio, dtype=q.dtype, device=device).view(1, -1, 1)
    weights = torch.ones(1, 17, 1, device=device)
    actual = lightning_indexer(
        q, k, weights, 2, causal=causal, compress_ratio=ratio, impl="reference"
    )
    monkeypatch.setattr(indexer, "_SCORE_TILE_BYTES", 4 * k.shape[1] * 8)
    match corruption:
        case "regret":
            actual[:, -1, 0] = 0  # Valid and unique, but far below the selection boundary.
        case "duplicate":
            actual[:, -1, 0] = actual[:, -1, 1]
        case "padding":
            actual[:, -1, 0] = -1
        case "range":
            actual[:, -1, 0] = k.shape[1]
        case "future":
            actual[:, ratio - 1, 0] = k.shape[1] - 1
    with pytest.raises(AssertionError):
        indexer.assert_indexer_selection(actual, q, k, weights, 2, causal, ratio)


def test_tiled_oracle_keeps_global_mean_and_rms(monkeypatch):
    """A failing tail tile can pass the original global budget; check all entries once."""
    q = torch.ones(1, 5, 1, 1)
    k = torch.ones(1, 5, 1)
    weights = torch.ones(1, 5, 1)
    actual = lightning_indexer(q, k, weights, 2, impl="reference")
    eps = 4 * torch.finfo(torch.float32).eps
    factor = 2 * eps / (1 - eps)
    # Last row's mean/RMS (2*factor) fail alone, but earlier rows' eager
    # regret makes both global budgets pass.
    errors = torch.tensor([[[0.0, 0.0]] * 4 + [[2.0, 2.0]]], dtype=torch.float64) * factor
    baseline = torch.tensor([[[2.0, 0.0]] * 5], dtype=torch.float64) * factor
    baseline[:, -1] = torch.tensor([1.0, 0.0], dtype=torch.float64) * factor
    magnitude = torch.ones(1, 5, 1, dtype=torch.float64)
    valid = torch.ones_like(actual, dtype=torch.bool)
    indexer.assert_selection_regret_within(errors, baseline, magnitude, valid, factor)
    with pytest.raises(AssertionError, match="mean absolute"):
        indexer.assert_selection_regret_within(
            errors[:, -1:], baseline[:, -1:], magnitude[:, -1:], valid[:, -1:], factor
        )
    # Supply known per-row regrets, exercising assembly and the real global checker.
    monkeypatch.setattr(indexer, "_SCORE_TILE_BYTES", 4 * 5 * 8)
    tiles = iter(
        (
            (errors[:, :4], baseline[:, :4], magnitude[:, :4]),
            (errors[:, 4:], baseline[:, 4:], magnitude[:, 4:]),
        )
    )
    monkeypatch.setattr(indexer, "_selection_errors", lambda *args: next(tiles))
    check = Mock(wraps=indexer.assert_selection_regret_within)
    monkeypatch.setattr(indexer, "assert_selection_regret_within", check)
    indexer.assert_indexer_selection(actual, q, k, weights, 2, False)
    check.assert_called_once()
    for result, expected in zip(check.call_args.args[:4], (errors, baseline, magnitude, valid)):
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
