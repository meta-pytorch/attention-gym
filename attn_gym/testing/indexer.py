"""Shared operands and score-selection error checks for lightning indexers."""

import math
from typing import NamedTuple

import torch

from attn_gym.sparse.indexer import lightning_indexer


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
    k[:, 0, :32] = 0
    return ScaledIndexerInputs(
        q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), weights, q_scale, k_scale
    )


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
    compress_ratio: int = 1,
    *,
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> None:
    """Check index invariants and FP64 boundary regret against eager's regret.

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
    counts = visible.clamp_max(topk) if causal else torch.full_like(row, topk)
    assert torch.equal(valid.sum(-1, keepdim=True), counts.expand(batch, -1, -1))
    if causal:
        assert not (valid & (actual >= visible)).any()
    ordered = actual.sort(-1).values
    assert not ((ordered[..., 1:] == ordered[..., :-1]) & (ordered[..., 1:] >= 0)).any()
    if topk == 0:
        return

    q64, k64, w64 = q.double(), k.double(), weights.double()
    if q_scale is not None:
        q64 = (q64.unflatten(-1, (dim // 32, 32)) * q_scale.double().unsqueeze(-1)).flatten(-2)
    if k_scale is not None:
        k64 = (k64.unflatten(-1, (dim // 32, 32)) * k_scale.double().unsqueeze(-1)).flatten(-2)
    # Keep exhaustive rows/keys without materializing [B,H,T,S] FP64 intermediates.
    scores = torch.empty((batch, tokens, candidates), device=q.device, dtype=torch.float64)
    magnitude = torch.empty_like(scores)
    for start in range(0, tokens, 128):
        query = q64[:, start : start + 128].permute(0, 2, 1, 3)
        weight = w64[:, start : start + 128].transpose(1, 2).unsqueeze(-1)
        scores[:, start : start + 128] = (
            (query @ k64.transpose(-1, -2).unsqueeze(1)).relu() * weight
        ).sum(1)
        magnitude[:, start : start + 128] = (
            (query.abs() @ k64.abs().transpose(-1, -2).unsqueeze(1)) * weight.abs()
        ).sum(1)
    scores /= math.sqrt(heads * dim)
    magnitude /= math.sqrt(heads * dim)
    assert torch.isfinite(scores).all()
    if causal:
        future = torch.arange(candidates, device=q.device).view(1, 1, candidates) >= visible
        scores.masked_fill_(future, -torch.inf)
        magnitude.masked_fill_(future, 0)
    boundary = scores.sort(-1, descending=True).values.gather(
        -1, (counts - 1).clamp_min(0).expand(batch, -1, -1)
    )
    eager = lightning_indexer(
        q,
        k,
        weights,
        topk,
        causal=causal,
        compress_ratio=compress_ratio,
        impl="reference",
        q_scale=q_scale,
        k_scale=k_scale,
    )
    actual_scores = scores.gather(-1, actual.long().clamp_min(0))
    eager_scores = scores.gather(-1, eager.long().clamp_min(0))
    actual_error = (boundary - actual_scores).clamp_min(0).masked_fill(~valid, 0)
    eager_error = (boundary - eager_scores).clamp_min(0).masked_fill(eager < 0, 0)
    scale_ops = int(q_scale is not None) + int(k_scale is not None)
    reduction_eps = (dim + heads + 2 + scale_ops) * torch.finfo(torch.float32).eps
    assert_selection_regret_within(
        actual_error,
        eager_error,
        magnitude.amax(-1, keepdim=True),
        valid,
        2 * reduction_eps / (1 - reduction_eps),
    )


def assert_indexer_topk_values(indices: torch.Tensor, rows: torch.Tensor) -> None:
    """Selection preserves input values: FP64 selected value multisets must be exact."""
    indices = indices.long()
    assert ((indices >= 0) & (indices < rows.shape[-1])).all()
    ordered = indices.sort(-1).values
    assert not (ordered[..., 1:] == ordered[..., :-1]).any()
    rows64 = rows.double()
    actual = rows64.gather(-1, indices).sort(-1).values
    expected = rows64.topk(indices.shape[-1], sorted=False).values.sort(-1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def indexer_gvr2_sample_positions(candidates: int, topk: int, threads: int = 512) -> torch.Tensor:
    """Mirror ``IndexerGVR2TopKKernel.sample_plan`` for a 16-byte aligned row.

    Returns every element index the kernel samples (float4 pairs), so a test can
    place values on exactly the sampled positions and force a sampling outcome.
    """
    quads = candidates // 4
    aim = 11 * topk // 8 if topk >= 1024 else 3 * topk // 2
    aim = max(min(aim, 4096 // 2), topk)
    aim = min(max(aim, int(math.sqrt(6 * candidates) + 0.5)), 4096 // 2)
    factor = 64 if topk >= 1024 else 32
    selected = min(max(factor * candidates // aim, 256), candidates // 2)
    half = max(quads >> 1, 1)
    pairs = min(max(selected >> 3, 1), half, threads)
    stride = max(half // pairs, 1)
    sample_threads = 0 if quads < 4 else min(half // stride, threads)
    quad = torch.arange(sample_threads) * stride * 2
    return (quad[:, None] * 4 + torch.arange(8)[None, :]).flatten().cuda()


def compile_indexer_topk(
    topk: int, causal: bool = False, ratio: int = 1, wide: bool = False, radix: bool = False
):
    """Compile the production selector ABI on the current device for direct tests."""
    from attn_gym._backends.cute.target import (
        detect_compile_target,
        get_compile_target,
        set_compile_target,
    )
    from attn_gym.sparse.indexer.impl.cute import _compile_topk

    previous = get_compile_target()
    try:
        set_compile_target(detect_compile_target(torch.cuda.current_device()))
        return _compile_topk(topk, causal, ratio, wide, "default" if radix else "gvr2")
    finally:
        set_compile_target(previous)
