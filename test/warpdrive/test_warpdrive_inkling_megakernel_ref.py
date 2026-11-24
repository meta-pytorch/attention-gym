"""The Inkling 2 Mini oracle against HF's ``Inkling2LoopForCausalLM`` on a small random model,
and the capturable torch step (``fast``) against the oracle with the declared expert numerics of
a quantized copy (``inkling_tiny.quantized_tiny_checkpoint``).

Checkpoint-free: a random tiny model (``inkling_tiny``) is saved in the original checkpoint
naming and loaded by both implementations.
"""

import pytest
import torch
from inkling_tiny import quantized_tiny_checkpoint, tiny_models, transformers_loop

from attn_gym.warpdrive.examples.inkling_megakernel.fast import decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    CONV_RING,
    InklingCache,
    forward_tokens,
    load_inkling_weights,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
transformers_loop()  # every test here compares with HF

PROMPT, STEPS = 24, CONV_RING + 4


@pytest.fixture(scope="module")
def tiny_path(tmp_path_factory):
    return tmp_path_factory.mktemp("tiny_inkling")


@pytest.fixture(scope="module")
def models(tiny_path):
    return tiny_models(tiny_path)


@pytest.fixture(scope="module")
def declared(models, tiny_path, tmp_path_factory):
    """The tiny model with quantized experts and their declared numerics (bf16 block values)."""
    path = quantized_tiny_checkpoint(tiny_path, tmp_path_factory.mktemp("q") / "tiny", 6)
    return load_inkling_weights(path)


def prompt(weights):
    generator = torch.Generator().manual_seed(1)
    return torch.randint(weights.config.vocab_size, (PROMPT,), generator=generator).cuda()


@torch.no_grad()
def test_prefill_and_decode_match_hf_bitwise(models):
    """Prefill, then greedy decode past the sliding window, the relative extent, the log-scaling
    floor and a wrap of the convolution rings: every logit equals HF's (eager attention)."""
    hf, weights = models
    tokens = prompt(weights)
    cache = InklingCache.empty(weights.config, max_len=64)
    output = hf(tokens[None], use_cache=True)
    logits = forward_tokens(weights, tokens, 0, cache, all_logits=True)
    torch.testing.assert_close(logits, output.logits[0], atol=0, rtol=0)
    for step in range(STEPS):
        token = output.logits[0, -1].argmax()
        output = hf(token.reshape(1, 1), cache=output.past_key_values, use_cache=True)
        logits = forward_tokens(weights, token, PROMPT + step, cache)
        torch.testing.assert_close(logits, output.logits[0, -1], atol=0, rtol=0, msg=f"{step}")


@torch.no_grad()
def test_fast_decode_step_tracks_oracle_and_captures(declared):
    """``fast.decode_step`` differs from the oracle only in summation order and the double
    rounding of NVFP4 GEMM outputs: same greedy tokens and close logits over the window, extent
    and ring boundaries. Its CUDA-graph replay equals its eager run bit for bit."""
    weights = declared
    assert all(w.declared_experts for w in weights.layers)
    tokens = prompt(weights)
    oracle = InklingCache.empty(weights.config, max_len=64)
    fast = InklingCache.empty(weights.config, max_len=64)
    forward_tokens(weights, tokens, 0, oracle)
    forward_tokens(weights, tokens, 0, fast)
    token, pos = tokens[-1:].clone(), torch.zeros(1, device="cuda", dtype=torch.int32)
    for step in range(STEPS):
        expected = forward_tokens(weights, token, PROMPT + step, oracle)
        pos.fill_(PROMPT + step)
        logits = decode_step(weights, token, pos, fast)
        torch.testing.assert_close(logits, expected, atol=0.1, rtol=0, msg=f"step {step}")
        assert logits.argmax() == expected.argmax()
        token = expected.argmax().reshape(1)

    # Replay the last step: same inputs, same caches (it rewrites the same rows).
    eager = decode_step(weights, token, pos, fast)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = decode_step(weights, token, pos, fast)
    graph.replay()
    torch.testing.assert_close(captured, eager, atol=0, rtol=0)
