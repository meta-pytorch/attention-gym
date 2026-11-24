"""Inkling chat options and history: the cache capacities a backend takes, and a parsed reply
re-rendering to the tokens the model generated."""

import pytest
import typer
from inkling_checkpoint import MODEL

from attn_gym.warpdrive.examples.inkling_megakernel.chat import (
    assistant_message,
    check_options,
    reasoning_effort,
    reply_parts,
)
from attn_gym.warpdrive.examples.megakernel.chat import ChatOptions


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer is not in the local Hugging Face cache")


def test_reply_parts_stream():
    text = "<|message_model|><|content_thinking|>hm<|end_message|><|message_model|><|content_text|>Par"
    assert reply_parts(text) == [("thinking", "hm", True), ("text", "Par", False)]


def test_parsed_reply_rerenders_to_its_tokens(tokenizer):
    user = [{"role": "user", "content": "Capital of France?"}]
    prompt = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    reply = (
        "<|message_model|><|content_thinking|>Simple question.<|end_message|>"
        "<|message_model|><|content_text|>Paris.<|end_message|>"
    )
    stop = "<|content_model_end_sampling|>"  # the stop token: sampled, never fed
    ids = tokenizer(prompt + reply, add_special_tokens=False).input_ids
    message = assistant_message(reply)
    assert message == {
        "role": "assistant",
        "content": "Paris.",
        "reasoning_content": "Simple question.",
    }
    history = tokenizer.apply_chat_template(user + [message], tokenize=False)
    rendered = tokenizer(history, add_special_tokens=False).input_ids
    assert rendered[: len(ids)] == ids
    assert history[len(prompt + reply) :].startswith(stop)


@pytest.mark.parametrize(
    ("given", "effort"), [(None, "low"), ("high", "high"), ("0.5", 0.5), ("0", 0.0)]
)
def test_reasoning_effort_accepts_levels_and_numbers(given, effort):
    assert reasoning_effort(ChatOptions("reference", reasoning_effort=given)) == effort


@pytest.mark.parametrize("given", ["", "nan", "1", "medium-ish"])
def test_reasoning_effort_rejects_other_values(given):
    with pytest.raises(typer.BadParameter):
        reasoning_effort(ChatOptions("reference", reasoning_effort=given))


def test_megakernel_backends_take_kv_block_capacities():
    check_options(ChatOptions("reference", max_len=1000))
    check_options(ChatOptions("megakernel", max_len=1024))
    with pytest.raises(typer.BadParameter, match="multiple of 64"):
        check_options(ChatOptions("megakernel-bf16", max_len=1000))
