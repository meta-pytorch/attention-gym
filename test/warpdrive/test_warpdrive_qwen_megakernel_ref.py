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
    prompt_tokens,
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
    high_logits = decode_step_ref(weights, token, pos, *caches, dtype=torch.float32)
    assert compare_logits(fast_logits, logits, high_logits)["passed"]
    for cache, before in zip(caches, prefix):
        torch.testing.assert_close(cache[:, :, :pos], before, rtol=0, atol=0)
        assert torch.isnan(cache[:, :, pos + 1]).all()


def test_moe_decode_matches_hf(tmp_path: Path) -> None:
    """A small random Qwen3-MoE, saved by HF and read back by load_qwen3_weights (per-expert
    tensors, router, untied head, transformers 5 config names): prefill and decode logits agree
    with Qwen3MoeForCausalLM to bf16 noise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    torch.manual_seed(0)
    config = transformers.Qwen3MoeConfig(
        hidden_size=512, num_hidden_layers=3, num_attention_heads=8, num_key_value_heads=2,
        head_dim=128, intermediate_size=1024, moe_intermediate_size=256, num_experts=16,
        num_experts_per_tok=4, norm_topk_prob=True, vocab_size=2048, tie_word_embeddings=False,
    )  # fmt: skip
    model = transformers.Qwen3MoeForCausalLM(config).to(torch.bfloat16).cuda().eval()
    model.save_pretrained(tmp_path)
    weights = load_qwen3_weights(str(tmp_path))
    assert weights.config.num_experts == 16 and weights.router.shape == (3, 16, 512)
    assert weights.gate.shape == (3, 16, 256, 512) and weights.down.shape == (3, 16, 512, 256)
    ctx = 40
    with torch.no_grad():
        output = model(prompt_tokens(weights, ctx, 0)[None], use_cache=True, logits_to_keep=1)
        caches, token, pos = make_decode_state(weights, ctx, ctx + 4, seed=0)
        assert token == output.logits[0, -1].argmax()
        for step in range(3):
            ours = decode_step_ref(weights, token, pos + step, *caches)
            output = model(token.reshape(1, 1), past_key_values=output.past_key_values)
            theirs = output.logits[0, -1].float()
            torch.testing.assert_close(ours, theirs, atol=0.05, rtol=0.02)
            assert ours.argmax() == theirs.argmax()
            token = theirs.argmax()


def test_error_budget_rejects_wrong_token_and_nonfinite_logits() -> None:
    high = torch.tensor([1.0, 2.0, 3.0])
    bf16 = torch.tensor([1.01, 2.02, 3.01])
    assert compare_logits(bf16, bf16, high)["passed"]
    assert not compare_logits(high.flip(0), bf16, high)["passed"]
    assert not compare_logits(torch.full_like(high, float("nan")), bf16, high)["passed"]


def test_compare_logits_accepts_either_token_of_a_near_tie():
    """Eager and the test pick different tokens 0.01 apart in the oracle (well within eager's
    error): the gate passes. A token the oracle scores far below eager's fails."""
    high = torch.tensor([8.0, 7.99, 2.0, 1.0], dtype=torch.float64)
    eager = torch.tensor([7.97, 8.0, 2.05, 0.95])
    test = torch.tensor([8.01, 7.98, 2.0, 1.04])
    metrics = compare_logits(test, eager, high)
    assert not metrics["argmax_match"] and metrics["argmax_ok"] and metrics["passed"], metrics
    wrong = torch.tensor([7.9, 7.95, 8.05, 1.0])
    assert not compare_logits(wrong, eager, high)["argmax_ok"]
