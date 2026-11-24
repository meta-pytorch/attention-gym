"""Inkling 2 Mini's device-side speculative step (``examples.megakernel.spec.SpecStep``) and its
chat backends: the random tiny target and drafter check that the step, eager and replayed from a
CUDA graph, decodes what ``dspark.speculative_generate``'s host loop decodes with the same
components; the checkpoint tests (skipped unless cached) run the chat's ``spec-torch`` and
``megakernel-spec`` backends against the host loop. ``test_warpdrive_megakernel_spec`` tests the
step's acceptance on scripted components.
"""

from pathlib import Path

import pytest
import torch
from inkling_checkpoint import TOKENIZER, WEIGHTS, checkpoint_path

# The tiny drafter and the tiny target of the oracle's HF parity tests (only tests that take
# tiny_target need transformers' Inkling2Loop).
from test_warpdrive_inkling_dspark import (  # noqa: F401
    PROMPT,
    TINY_VISITS,
    tiny_drafter_dir,
    tiny_target,
)
from test_warpdrive_megakernel_spec import step_runner

from attn_gym.warpdrive.examples.inkling_megakernel import quant
from attn_gym.warpdrive.examples.inkling_megakernel.chat import (
    SPEC_TOKENS,
    MegakernelSpec,
    SpecTorch,
)
from attn_gym.warpdrive.examples.inkling_megakernel.dspark import (
    DSparkDrafter,
    load_dspark_weights,
    spec_max_len,
    speculative_generate,
    target_forward,
)
from attn_gym.warpdrive.examples.inkling_megakernel.ref import InklingCache, load_inkling_weights
from attn_gym.warpdrive.examples.inkling_megakernel.spec import (
    DecoderTargetBlock,
    TorchTargetBlock,
    block_forward,
)
from attn_gym.warpdrive.examples.megakernel.chat import SYNC_EVERY, Chat
from attn_gym.warpdrive.examples.megakernel.spec import SpecStep, prefill_blocks

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.usefixtures("release_cuda_memory"),
]


@torch.no_grad()
def device_decode(weights, prompt, drafter, max_len: int, tokens: int, max_new: int, graph):
    """Speculative greedy decode with ``SpecStep`` (``spec.TorchTargetBlock`` of ``tokens``) after
    the oracle's prefill of ``prompt``, one step per run, eager or replayed from a CUDA graph,
    until ``max_new`` tokens are out: the tokens, each step's accepted drafts and the cache."""
    cache = InklingCache.empty(weights.config, max_len)
    logits, features = target_forward(weights, prompt, 0, cache, TINY_VISITS, False)
    drafter.commit(features, torch.arange(prompt.numel(), device="cuda"))
    step = SpecStep(TorchTargetBlock(weights, cache, tokens, TINY_VISITS), drafter)
    token = logits.argmax().reshape(1)
    pos = torch.tensor([prompt.numel() - 1], device="cuda", dtype=torch.int32)
    row = torch.zeros(1, 1 + tokens, device="cuda", dtype=torch.long)
    live = (token, pos, *vars(cache).values(), drafter.cache.k, drafter.cache.v, drafter.f_rnn)
    saved = [t.clone() for t in live]

    def reset():
        for tensor, value in zip(live, saved):
            tensor.copy_(value)

    run = step_runner(step, token, pos, row, graph, reset)
    out, accepted = [token.item()], []
    while len(out) < max_new:  # a step starts only while fewer than max_new tokens are out
        run()
        count, *committed = row[0].tolist()
        out += committed[:count]
        accepted.append(count - 1)
    return out[:max_new], accepted, cache


@torch.no_grad()
def test_spec_step_decodes_the_host_loop(tiny_target, tiny_drafter_dir):  # noqa: F811
    """On the random tiny target and drafter, the device step at the chat's default ``k``, eager
    and replayed from a CUDA graph, commits the tokens and accepted counts of
    ``speculative_generate``'s host loop with the same verify forward (``spec.block_forward``)
    and drafter, and writes the same caches."""
    weights, prompt = tiny_target
    tokens = SPEC_TOKENS
    max_new = 24
    max_len = spec_max_len(PROMPT, max_new, tokens)  # attention reads whole caches
    draft = load_dspark_weights(weights, tiny_drafter_dir)
    draft_len = spec_max_len(PROMPT, max_new, draft.config.block_size)

    def drafter():
        return DSparkDrafter(draft, draft_len, tokens - 1, static=True)

    def verify(block, pos, cache):
        positions = torch.arange(pos, pos + block.numel(), device="cuda")
        return block_forward(weights, block, positions, cache, TINY_VISITS)

    expected = speculative_generate(
        weights, drafter(), prompt, max_new, visits=TINY_VISITS, verify=verify
    )
    eager = device_decode(weights, prompt, drafter(), max_len, tokens, max_new, graph=False)
    graph = device_decode(weights, prompt, drafter(), max_len, tokens, max_new, graph=True)
    assert eager[:2] == (expected.tokens, expected.accepted)
    assert graph[:2] == eager[:2]
    for name, tensor in vars(eager[2]).items():
        assert torch.equal(getattr(graph[2], name), tensor), name


def reply_of(tokens: list[int], stops: set[int], room: int) -> list[int]:
    """The chat's reply for decoded ``tokens``: those before the first stop (possibly none), at
    most ``room``."""
    end = next((i for i, t in enumerate(tokens) if t in stops), len(tokens))
    return tokens[: min(end, room)]


class StubChatModel:
    """A ``chat.ChatModel`` around ``backend`` whose conversation renders to ``ids``."""

    def __init__(self, backend, ids: list[int], stops: set[int], max_len: int):
        self.backend, self.ids, self.stops, self.max_len = backend, ids, stops, max_len
        self.tokenizer, self.label = None, "spec"

    def prompt_ids(self, messages):
        return self.ids

    def prefill(self, ids, start):
        return self.backend.prefill(ids, start)

    def reply_text(self, ids):
        return ""

    def assistant_message(self, ids):
        return {"role": "assistant", "ids": list(ids)}


@pytest.fixture(scope="module")
def checkpoint():
    """The checkpoint directory, a chat prompt (low reasoning effort) and the stop tokens. Each
    test loads the weights it needs, so they are freed after it."""
    path = checkpoint_path((*WEIGHTS, *TOKENIZER, "mtp.safetensors"))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    messages = [{"role": "user", "content": "Write a Python function that reverses a string."}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, reasoning_effort="low"
    )
    stops = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
    stops = {s for s in stops if isinstance(s, int) and s >= 0}
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return Path(path), ids, stops


@pytest.mark.xdist_group("inkling-checkpoint")
@torch.no_grad()
def test_checkpoint_spec_chat_decodes_the_host_loop(checkpoint):
    """The chat's ``spec-torch`` backend (k = 4, CUDA graphs of two steps per replay) replies
    with ``speculative_generate``'s tokens for the same verify forward and drafter, cut before
    the first stop token (the chat never shows one) or at the caches' room, and the shipped
    drafter is accepted (a broken feature, commit or accept wiring drafts ~nothing)."""
    path, ids, stops = checkpoint
    target = load_inkling_weights(path)
    max_len = len(ids) + 64
    backend = SpecTorch(target, InklingCache.empty(target.config, max_len), path)
    tokens = backend.tokens_per_step
    chat = Chat(StubChatModel(backend, ids, stops, max_len))
    chat.reply("", None, 0.0, lambda *_: None)
    reply = chat.messages[-1]["ids"]

    def verify(block, pos, cache):
        positions = torch.arange(pos, pos + block.numel(), device="cuda")
        return block_forward(target, block, positions, cache)

    drafter = DSparkDrafter(
        backend.drafter.weights, backend.drafter.cache.max_len, tokens - 1, static=True
    )
    prompt = torch.tensor(ids, device="cuda")
    max_new = max_len - len(ids) - (tokens - 1)  # the same target cache length
    expected = speculative_generate(target, drafter, prompt, max_new, stops, verify=verify)
    assert reply == reply_of(expected.tokens, stops, max_len - len(ids) - SYNC_EVERY - 1)
    assert expected.tokens_per_step > 1.6, expected.accepted


@pytest.mark.xdist_group("inkling-checkpoint")
@pytest.mark.usefixtures("requires_blackwell")
@torch.no_grad()
def test_checkpoint_megakernel_spec_chat_decodes_the_host_loop(checkpoint):
    """The chat's ``megakernel-spec`` backend (k = 4, CUDA graphs of two steps per replay)
    replies with ``speculative_generate``'s tokens for the same prefill (``prefill_blocks``),
    verify block (``InklingDecoder.decode_block``) and draft launches, run eagerly, cut as
    ``reply_of``; its block rows are causal (the prefill's padding moves no earlier row) and
    the drafter is accepted."""
    pytest.importorskip("cutlass")
    from attn_gym.warpdrive.examples.inkling_megakernel.draft import MegakernelDrafter
    from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import InklingDecoder

    path, ids, stops = checkpoint
    target = quant.load_quantized_inkling_weights(path)
    max_len = -(-(len(ids) + 96) // 64) * 64
    backend = MegakernelSpec(target, InklingCache.empty(target.config, max_len), path)
    tokens = backend.tokens_per_step
    chat = Chat(StubChatModel(backend, ids, stops, max_len))
    chat.reply("", None, 0.0, lambda *_: None)
    reply = chat.messages[-1]["ids"]

    decoder = InklingDecoder(target, block_tokens=tokens)
    draft = backend.drafter.weights
    drafter = MegakernelDrafter(draft, backend.drafter.max_len, tokens - 1)

    def verify(block, pos, cache):
        return decoder.decode_block(block, torch.tensor([pos], device="cuda"), cache, True)

    def prefill(prompt, length):
        assert length == max_len
        cache = InklingCache.empty(target.config, length)
        block = DecoderTargetBlock(decoder, cache)
        # Causal block rows: the last token's row and features ignore the tokens after it.
        first = torch.tensor([0], device="cuda")
        logits, features = (t.clone() for t in block(prompt[:tokens], first))
        other = prompt[:tokens].clone()
        other[-1] += 1
        changed, changed_features = block(other, first)
        assert torch.equal(changed[:-1], logits[:-1])
        assert torch.equal(changed_features[:-1], features[:-1])
        return cache, prefill_blocks(block, drafter.kernels, prompt, 0)

    prompt = torch.tensor(ids, device="cuda")
    max_new = max_len - len(ids) - (tokens - 1)  # the same target cache length
    expected = speculative_generate(
        target, drafter, prompt, max_new, stops, verify=verify, prefill=prefill
    )
    assert reply == reply_of(expected.tokens, stops, max_len - len(ids) - SYNC_EVERY - 1)
    assert expected.tokens_per_step > 1.6, expected.accepted
