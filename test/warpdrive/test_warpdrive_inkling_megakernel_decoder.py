"""The Inkling 2 Mini decode step as megakernel launches only (``megakernel.InklingDecoder``):
the head launch (final block residual, norm, LM head, softcap) against the oracle on random
weights at the real shapes, and on the checkpoint, with its experts as stored, a greedy decode
against the oracle on their bf16 dequantization, compile reuse across positions and a CUDA graph
replay bit-identical to eager launches."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

from inkling_checkpoint import TOKENIZER, WEIGHTS, checkpoint_path
from test_warpdrive_inkling_megakernel_visit import (
    CONFIG,
    assert_caches_equal,
    capture,
    clone,
    restore,
)

from attn_gym.warpdrive.examples.inkling_megakernel import quant, ref
from attn_gym.warpdrive.examples.inkling_megakernel.head_kernel import compile_head_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import (
    InklingDecoder,
    InklingHeadMegakernel,
    experts_format,
)
from attn_gym.warpdrive.examples.inkling_megakernel.msg_kernel import compile_msg_kernel
from attn_gym.warpdrive.examples.inkling_megakernel.visit_kernel import compile_visit_kernel
from attn_gym.warpdrive.examples.megakernel.formats import MXFP8, NVFP4

pytestmark = pytest.mark.usefixtures("requires_blackwell", "release_cuda_memory")

COMPILES = (compile_visit_kernel, compile_msg_kernel, compile_head_kernel)
PROMPT = "What is the capital of France? One sentence."
STEPS = 16
# Logits differ from the oracle's where a bf16 rounding of an LM-head output or of the
# normalized final state flips: one bf16 ulp of a logit (2 ** -3 at |logit| in [16, 32)), at
# most, compressed by the softcap.
HEAD_ATOL = 0.13
# A decode step's logits stay within a few units of the oracle's: the caches carry a step's
# rounding differences into later steps, the experts' quantized inputs (declared numerics)
# amplify them (a flipped rounding can move a code, a block scale or a row's amax), and where a
# router's 8th and 9th keys nearly tie (margins of 1e-5 occur within tens of steps) rounding
# picks another expert at one visit. Measured 1.4 to 4.1 over 40 steps (HF's numerics: about 1).
STEP_ATOL = 6.0
# Best-two logit margins under which the oracle's token is a tie: one bf16 ulp of a logit in
# [16, 32) is 0.125 (the oracle itself ties exactly at times, broken by the lower index).
TIE = 0.25


@torch.no_grad()
def test_head_random_weights():
    """The head launch against ``ref.hidden_states``' last block residual and norm and
    ``ref.logits_of`` on a random LM head at the real vocabulary (its last tile partial), with
    logits spread over the softcap's range; 1 to 4 blocks."""
    config, h = CONFIG, CONFIG.hidden_size
    gen = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, device="cuda", generator=gen) * scale

    weights = SimpleNamespace(
        config=config,
        lm_head=randn(config.vocab_size, h, scale=20 * h**-0.5).bfloat16(),
        final_query=randn(h),
        final_norm=(1 + randn(h, scale=0.1)).bfloat16(),
    )
    runner = InklingHeadMegakernel(weights)
    for count in (4, 1, 3):
        states = [(randn(1, h) * 2).bfloat16() for _ in range(count + 1)]
        blocks, partial = states[:-1], states[-1]
        logits = runner(blocks, partial)
        x = ref.block_residual(weights.final_query[None], blocks, partial)[:, 0]
        x = ref.rms_norm(x, weights.final_norm, config.rms_eps).to(torch.bfloat16)
        expected = ref.logits_of(weights, x)[0]
        errors = (logits - expected).abs()
        print(f"{count} blocks: max {errors.max():.4f}, {(errors > 0).float().mean():.4f} differ")
        assert errors.max() <= HEAD_ATOL
        assert logits.argmax() == expected.argmax()


def prompt_ids(path) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    return ref.chat_prompt_ids(tokenizer, [{"role": "user", "content": PROMPT}], "low")


@pytest.mark.xdist_group("inkling-checkpoint")
@torch.no_grad()
def test_checkpoint_greedy_decode():
    """``STEPS`` greedy steps after a chat prompt's prefill on Inkling 2 Mini with its experts as
    stored (NVFP4 layers 0-19, MXFP8 20-23), the decoder fed the oracle's token (``ref`` on their
    bf16 dequantization, the declared numerics) into its own cache: it picks the oracle's
    token every step, unless the oracle's best two logits nearly tie (``TIE``) and it picks the
    other, with close logits (``STEP_ATOL``); positions compile nothing after the first step; one
    step captured in a CUDA graph replays bit-identically to the same step's eager launches,
    logits and caches. Skips unless the checkpoint is cached; never downloads."""
    path = checkpoint_path(WEIGHTS + TOKENIZER)
    weights = quant.load_quantized_inkling_weights(path)
    assert [experts_format(w) for w in weights.layers] == [NVFP4] * 20 + [MXFP8] * 4
    dense = quant.dequantized_weights(weights)
    ids = torch.tensor(prompt_ids(path), device="cuda")
    theirs = ref.InklingCache.empty(weights.config, max_len=256)
    token = ref.forward_tokens(dense, ids, 0, theirs).argmax().reshape(1)
    mine = clone(theirs)
    decoder = InklingDecoder(weights)
    pos = torch.tensor([ids.numel()], device="cuda", dtype=torch.int32)
    counts = None
    for step in range(STEPS):
        expected = ref.forward_tokens(dense, token, ids.numel() + step, theirs)
        logits = decoder(token, pos, mine)
        error = (logits - expected).abs().max().item()
        (best, second), (first, other) = expected.topk(2)
        print(f"step {step}: token {int(first)}, logit max error {error:.4f}")
        assert logits.argmax() == first or (logits.argmax() == other and best - second < TIE)
        assert error <= STEP_ATOL, step
        if counts is None:
            counts = [f.cache_info().currsize for f in COMPILES]
        token = first.reshape(1)
        pos += 1
    assert [f.cache_info().currsize for f in COMPILES] == counts

    initial = clone(mine)
    graph, logits = capture(lambda: decoder(token, pos, mine))
    restore(mine, initial)
    eager = decoder(token, pos, mine).clone()
    eager_cache = clone(mine)
    restore(mine, initial)
    logits.zero_()
    graph.replay()
    assert torch.equal(logits, eager)
    assert_caches_equal(mine, eager_cache)
