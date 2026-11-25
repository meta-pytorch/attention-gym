"""Terminal output is append-only even when an unmarked answer precedes thinking."""

from itertools import pairwise

import pytest

from attn_gym.warpdrive.examples.inkling_megakernel.chat import InklingChatModel
from attn_gym.warpdrive.examples.megakernel.chat import ChatOptions


class Tokenizer:
    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return "".join(ids)


@pytest.mark.parametrize(
    "options", [ChatOptions("reference"), ChatOptions("reference", show_thinking=True)]
)
def test_streaming_render_preserves_unmarked_text_and_hides_only_thinking(options):
    model = object.__new__(InklingChatModel)
    model.tokenizer = Tokenizer()
    model.show_thinking = options.show_thinking
    tokens = [
        "Hello! ",
        "<|message_model|>",
        "<|content_thinking|>",
        "private",
        "<|end_message|>",
        "<|message_model|>",
        "<|content_text|>",
        "World.",
        "<|end_message|>",
    ]
    outputs = [model.reply_text(tokens[:end]) for end in range(len(tokens) + 1)]
    assert outputs[1] == "Hello! "
    assert outputs[-1] == (
        "Hello! [thinking]private[/thinking]\nWorld." if options.show_thinking else "Hello! World."
    )
    assert all(after.startswith(before) for before, after in pairwise(outputs))
    assert model.assistant_message(tokens[:1]) == {"role": "assistant", "content": "Hello! "}
    assert model.assistant_message(tokens) == {
        "role": "assistant",
        "content": "Hello! World.",
        "reasoning_content": "private",
    }
