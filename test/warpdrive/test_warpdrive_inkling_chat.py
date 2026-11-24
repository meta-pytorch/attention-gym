"""Inkling chat options and history: the files, cache capacities and decoding modes a backend
takes, and a parsed reply re-rendering to the tokens the model generated."""

import pytest
import typer
from inkling_checkpoint import MODEL

from attn_gym.warpdrive.examples.inkling_megakernel.chat import (
    SPEC_TOKENS,
    assistant_message,
    check_options,
    checkpoint_dir,
    reasoning_effort,
    reply_parts,
    verify_tokens,
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
    for backend in ("megakernel-bf16", "megakernel-spec"):
        with pytest.raises(typer.BadParameter, match="multiple of 64"):
            check_options(ChatOptions(backend, max_len=1000))


@pytest.mark.parametrize(("backend", "drafter"), [("megakernel", False), ("spec-torch", True)])
def test_checkpoint_download_takes_the_backends_files(monkeypatch, tmp_path, backend, drafter):
    """A Hub download fetches the drafter with the target, in one snapshot, only for a
    speculative backend; a local directory is used as it is, without the Hub."""
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    def snapshot_download(repo, allow_patterns, tqdm_class=None):
        calls.append((repo, allow_patterns))
        return str(tmp_path)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    options = ChatOptions(backend)
    assert checkpoint_dir(MODEL, options=options) == tmp_path
    [(repo, patterns)] = calls
    assert repo == MODEL and "model*.safetensors" in patterns
    assert ("mtp.safetensors" in patterns) == drafter
    assert checkpoint_dir(tmp_path, options=options) == tmp_path
    assert len(calls) == 1


def test_speculative_backends_reject_sampling_at_startup():
    check_options(ChatOptions("megakernel", temperature=0.7))
    check_options(ChatOptions("spec-torch", temperature=0.0))
    with pytest.raises(typer.BadParameter, match="greedily"):
        check_options(ChatOptions("spec-torch", temperature=0.7))


@pytest.mark.parametrize(
    ("backend", "given", "k"),
    [("spec-torch", None, SPEC_TOKENS), ("spec-torch", 8, 8), ("megakernel-spec", 2, 2)],
)
def test_verify_tokens_pick_the_block(backend, given, k):
    options = ChatOptions(backend, verify_tokens=given)
    check_options(options)
    assert verify_tokens(options) == k


@pytest.mark.parametrize(
    ("backend", "given"),
    [("spec-torch", 1), ("spec-torch", 9), ("megakernel-spec", 5), ("megakernel", 4)],
)
def test_verify_tokens_rejects_other_blocks_and_backends(backend, given):
    with pytest.raises(typer.BadParameter, match="--verify-tokens"):
        check_options(ChatOptions(backend, verify_tokens=given))
