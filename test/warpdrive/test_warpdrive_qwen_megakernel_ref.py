"""Offline, batch-one checkpoint integration checks for the decode oracle."""

from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass.cute")  # the bench module imports the megakernel

from attn_gym.warpdrive.examples.bench_qwen_megakernel import (
    decode_step_fast,
    prepare_fast_weights,
)
from attn_gym.warpdrive.examples.qwen_megakernel_ref import (
    compare_logits,
    decode_step_ref,
    load_qwen3_weights,
    make_decode_state,
    validate_against_hf,
)


@pytest.fixture(scope="module")
def checkpoint() -> Path:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for checkpoint integration")
    hub = pytest.importorskip("huggingface_hub")
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    model = hub.try_to_load_from_cache("Qwen/Qwen3-0.6B", "model.safetensors")
    config = hub.try_to_load_from_cache("Qwen/Qwen3-0.6B", "config.json")
    if not isinstance(model, str) or not isinstance(config, str):
        pytest.skip("Qwen3-0.6B is not in the local HF cache")
    return Path(model).parent


def test_decode_matches_hf_and_writes_only_current_cache_position(checkpoint: Path) -> None:
    weights = load_qwen3_weights(str(checkpoint))
    metrics = validate_against_hf(weights, str(checkpoint), ctx_len=64, steps=2)
    assert metrics["greedy_match"]
    caches, token, pos = make_decode_state(weights, 8, 10, seed=7)
    prefix = tuple(cache[:, :, :pos].clone() for cache in caches)
    for cache in caches:
        cache[:, :, pos:].fill_(float("nan"))
    logits = decode_step_ref(weights, token, pos, *caches)
    assert logits.shape == (weights.config.vocab_size,)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    for cache, before in zip(caches, prefix):
        torch.testing.assert_close(cache[:, :, :pos], before, rtol=0, atol=0)
        assert torch.isfinite(cache[:, :, pos]).all()
        assert torch.isnan(cache[:, :, pos + 1]).all()
    assert weights.lm_head is weights.embed
    packed = prepare_fast_weights(weights, 10)
    layer_caches = tuple(tuple(cache.unbind(0)) for cache in caches)
    fast_logits = decode_step_fast(packed, token, pos, *layer_caches)
    high_weights = weights.to(torch.float32)
    high_logits = decode_step_ref(high_weights, token, pos, *caches, dtype=torch.float32)
    assert compare_logits(fast_logits, logits, high_logits)["passed"]
    for cache, before in zip(caches, prefix):
        torch.testing.assert_close(cache[:, :, :pos], before, rtol=0, atol=0)
        assert torch.isnan(cache[:, :, pos + 1]).all()


def test_error_budget_rejects_wrong_token_and_nonfinite_logits() -> None:
    high = torch.tensor([1.0, 2.0, 3.0])
    bf16 = torch.tensor([1.01, 2.02, 3.01])
    assert compare_logits(bf16, bf16, high)["passed"]
    assert not compare_logits(high.flip(0), bf16, high)["passed"]
    assert not compare_logits(torch.full_like(high, float("nan")), bf16, high)["passed"]
