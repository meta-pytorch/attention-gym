"""Inkling 2 Mini in the megakernel chat (``attn_gym.warpdrive.examples.megakernel.chat``).

    python -m attn_gym.warpdrive.examples.inkling_megakernel.chat --backend reference --prompt "Hi"

Prefill runs the PyTorch oracle (``quant.forward_tokens``: ``ref.forward_tokens``, one forward
pass into the caches, dequantizing only the routed experts of a quantized model), except on the
megakernel backends, which feed prompts of up to ``ORACLE_PREFILL`` tokens through their decode
step one token at a time, so that the caches come from the same kernels as the reply. The oracle
prefill is host-bound: about 2.1 s plus 0.9 ms per token, against 3.4 ms per decode step (GB300,
quantized experts, 2026-11-23), so it wins only past about 850 tokens. Backends:

- ``reference``: the oracle, one eager step at a time (a host sync per step: no CUDA graph);
- ``torch``: ``fast.decode_step`` (the oracle's dataflow on capturable kernels) in CUDA graphs;
- ``compile``: the same step under torch.compile (minutes to compile cold), in CUDA graphs;
- ``megakernel`` (default): the step as warpdrive megakernel launches only
  (``megakernel.InklingDecoder``: one per visit, one per core pass's messages, one for the LM
  head), in CUDA graphs, streaming the checkpoint's experts as stored (NVFP4 and MXFP8,
  ``quant.load_quantized_inkling_weights``: 19 GB of weights). It needs an SM100-family GPU
  and CuTeDSL; elsewhere use ``torch``;
- ``megakernel-bf16``: the same launches on the experts' bf16 block values (49 GB), with the
  same tokens bit for bit;
- ``spec-torch``: greedy speculative decoding with the checkpoint's DSpark drafter
  (``examples.megakernel.spec.SpecStep``: draft, a ``k``-token verify through ``fast.py``'s kernels
  and acceptance on the device), in CUDA graphs; ``--verify-tokens`` picks the verify block ``k``
  (2 to 8, default ``SPEC_TOKENS``). Greedy only; its prefill (the oracle's) also records the
  drafter's context.

The other backends load bf16 experts. Every backend computes the experts' numerics the
checkpoint declares (``ref.ExpertQuant``: quantized inputs, as vLLM's SM100 kernels), not HF's.

Replies are sequences of ``<|message_model|><|content_X|>...<|end_message|>`` parts, ``X`` being
``thinking`` or ``text``, closed by ``<|content_model_end_sampling|>``; the reply shows thinking
parts between ``[thinking]`` markers and keeps both for the next turn's history.
"""

import re
from pathlib import Path

import torch
import typer

from attn_gym.warpdrive.examples.inkling_megakernel import quant, ref
from attn_gym.warpdrive.examples.inkling_megakernel.dspark import (
    DSparkDrafter,
    load_dspark_weights,
    target_forward,
)
from attn_gym.warpdrive.examples.inkling_megakernel.fast import decode_step
from attn_gym.warpdrive.examples.inkling_megakernel.ref import (
    CONV_RING,
    InklingCache,
    forward_tokens,
    load_inkling_weights,
)
from attn_gym.warpdrive.examples.inkling_megakernel.spec import TorchTargetBlock
from attn_gym.warpdrive.examples.megakernel.chat import SYNC_EVERY, ChatOptions, run
from attn_gym.warpdrive.examples.megakernel.constants import BLOCK
from attn_gym.warpdrive.examples.megakernel.spec import SpecStep

__all__ = ["DEFAULT_BACKEND", "OPTIONS", "check_options", "checkpoint_dir", "load_chat_model"]

DEFAULT_EFFORT = "low"
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
PART = re.compile(
    r"(?:<\|message_model\|>)?<\|content_(thinking|text)\|>(.*?)(<\|end_message\|>|$)", re.DOTALL
)


class Backend:
    """A decode backend over ``weights`` and their ``cache``, built from the checkpoint directory
    ``checkpoint`` (speculative backends read their drafter there). ``prefill`` feeds a prompt
    through the oracle unless the backend overrides it."""

    capturable = False
    quantized = False  # whether the backend takes the checkpoint's experts as stored
    max_len_multiple = 1  # the cache capacities it takes are multiples of this

    def __init__(self, weights, cache: InklingCache, checkpoint: Path):
        self.weights, self.cache = weights, cache

    @classmethod
    def from_options(cls, weights, cache: InklingCache, checkpoint: Path, options: ChatOptions):
        """The backend for the chat's ``options``."""
        return cls(weights, cache, checkpoint)

    def prefill(self, ids: list[int], start: int) -> torch.Tensor:
        """Feed ``ids`` at positions ``start ..`` into the caches with one oracle forward pass
        (``quant.forward_tokens``); the last token's logits."""
        tokens = torch.tensor(ids, device="cuda")
        return quant.forward_tokens(self.weights, tokens, start, self.cache)


class Reference(Backend):
    """One eager decode step of the oracle."""

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return forward_tokens(self.weights, token, int(pos.item()), self.cache)


class Torch(Backend):
    """``fast.decode_step``, captured by the chat's CUDA graphs."""

    capturable = True
    compiled = False  # whether the step runs under torch.compile

    def __init__(self, weights, cache: InklingCache, checkpoint: Path):
        super().__init__(weights, cache, checkpoint)
        self.fn = torch.compile(decode_step) if self.compiled else decode_step

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.fn(self.weights, token, pos, self.cache)


# Prompt tokens past which the oracle prefill beats one decode step per token: 22 / 500 / 2000
# tokens took 0.08 / 1.70 / 6.82 s through the decode step and 2.13 / 3.64 / 3.98 s through the
# oracle (GB300, quantized experts, 2026-11-23).
ORACLE_PREFILL = 850


class Megakernel(Backend):
    """``megakernel.InklingDecoder``: the step as megakernel launches, captured by the chat's
    CUDA graphs."""

    capturable = True
    quantized = True
    max_len_multiple = BLOCK  # the KV block

    def __init__(self, weights, cache: InklingCache, checkpoint: Path):
        from attn_gym.warpdrive.examples.inkling_megakernel.megakernel import InklingDecoder

        super().__init__(weights, cache, checkpoint)
        self.decoder = InklingDecoder(weights)

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.decoder(token, pos, self.cache)

    def prefill(self, ids: list[int], start: int) -> torch.Tensor:
        """Feed ``ids`` at positions ``start ..`` through the decode step, one eager step per
        token, or through the oracle when that is faster (more than ``ORACLE_PREFILL`` tokens);
        the last token's logits."""
        if len(ids) > ORACLE_PREFILL:
            return super().prefill(ids, start)
        tokens = torch.tensor(ids, device="cuda")
        pos = torch.empty(1, device="cuda", dtype=torch.int32)
        for i in range(len(ids)):
            pos.fill_(start + i)
            logits = self.decoder(tokens[i : i + 1], pos, self.cache)
        return logits


class MegakernelBf16(Megakernel):
    """``Megakernel`` on the experts dequantized to bf16."""

    quantized = False


class Compile(Torch):
    """``Torch`` under torch.compile."""

    compiled = True


SPEC_TOKENS = 4  # the default verify block k


class Speculative(Backend):
    """Greedy speculative decoding (``chat.MultiTokenBackend``):
    ``examples.megakernel.spec.SpecStep`` over a target block of ``tokens_per_step`` tokens and the
    checkpoint's DSpark drafter, captured by the chat's CUDA graphs. Subclasses build ``self.spec``
    (the ``SpecStep``) and a prefill that also commits the prompt's drafter features."""

    capturable = True
    verify_tokens = range(2, SYNC_EVERY + 1)  # the k it takes (a replay feeds SYNC_EVERY)

    def __init__(
        self, weights, cache: InklingCache, checkpoint: Path, tokens_per_step: int = SPEC_TOKENS
    ):
        super().__init__(weights, cache, checkpoint)
        self.tokens_per_step = tokens_per_step

    @classmethod
    def from_options(cls, weights, cache: InklingCache, checkpoint: Path, options: ChatOptions):
        return cls(weights, cache, checkpoint, verify_tokens(options))

    def drafter_max_len(self, cache: InklingCache, draft) -> int:
        """Each block's anchor sits at a position the chat keeps inside the target's caches; the
        drafter writes block_size positions from it."""
        return cache.max_len + draft.config.block_size

    def decode(self, token: torch.Tensor, pos: torch.Tensor, out: torch.Tensor) -> None:
        self.spec(token, pos, out)


class SpecTorch(Speculative):
    """``Speculative`` over a ``spec.TorchTargetBlock`` and the torch drafter (static:
    capturable)."""

    def __init__(
        self, weights, cache: InklingCache, checkpoint: Path, tokens_per_step: int = SPEC_TOKENS
    ):
        super().__init__(weights, cache, checkpoint, tokens_per_step)
        draft = load_dspark_weights(weights, checkpoint)
        self.visits = draft.config.target_visits
        max_len = self.drafter_max_len(cache, draft)
        self.drafter = DSparkDrafter(draft, max_len, self.tokens_per_step - 1, static=True)
        target = TorchTargetBlock(weights, cache, self.tokens_per_step, self.visits)
        self.spec = SpecStep(target, self.drafter)

    def prefill(self, ids: list[int], start: int) -> torch.Tensor:
        """The oracle's prefill of ``ids`` at ``start ..``, also committing their drafter
        features; the last token's logits."""
        tokens = torch.tensor(ids, device="cuda")
        logits, features = target_forward(
            self.weights, tokens, start, self.cache, self.visits, all_logits=False
        )
        self.drafter.commit(features, torch.arange(start, start + len(ids), device="cuda"))
        return logits


BACKENDS = {
    "reference": Reference,
    "torch": Torch,
    "compile": Compile,
    "megakernel": Megakernel,
    "megakernel-bf16": MegakernelBf16,
    "spec-torch": SpecTorch,
}
DEFAULT_BACKEND = "megakernel"
OPTIONS = frozenset({"reasoning_effort", "verify_tokens"})


def reply_parts(text: str) -> list[tuple[str, str, bool]]:
    """``(kind, text, closed)`` parts of a reply decoded with its special tokens, kind
    ``thinking`` or ``text``; only the last part may be unclosed (still streaming)."""
    return [(match[1], match[2], bool(match[3])) for match in PART.finditer(text)]


def assistant_message(text: str) -> dict:
    """The chat-template message of a reply decoded with its special tokens. The template keeps
    one thinking and one text part per message, so a reply with several parts of a kind
    re-renders them joined (and thinking first); the chat then rebuilds its caches. Otherwise the
    rendering reproduces the reply's tokens and the next turn reuses their cache entries."""
    parts = reply_parts(text)
    message = {"role": "assistant", "content": "".join(t for k, t, _ in parts if k == "text")}
    thinking = "".join(t for k, t, _ in parts if k == "thinking")
    if thinking:
        message["reasoning_content"] = thinking
    return message


class InklingChatModel:
    def __init__(self, checkpoint: Path, options: ChatOptions):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        backend = BACKENDS[options.backend]
        load = quant.load_quantized_inkling_weights if backend.quantized else load_inkling_weights
        self.weights = load(checkpoint)
        # The chat feeds up to SYNC_EVERY positions past a stop token (a speculative step's
        # rejected ones included), then resumes at the stop token's position: its convolutions
        # read the kernel - 1 earlier ring rows, which that over-run must not have overwritten.
        if SYNC_EVERY + self.weights.config.conv_kernel - 1 > CONV_RING:
            raise ValueError(f"CONV_RING {CONV_RING} is too short for SYNC_EVERY {SYNC_EVERY}")
        self.max_len = options.max_len
        self.cache = InklingCache.empty(self.weights.config, options.max_len)
        self.backend = backend.from_options(self.weights, self.cache, checkpoint, options)
        self.label = f"the {options.backend} backend"
        self.effort = reasoning_effort(options)
        stops = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        self.stops = {s for s in stops if isinstance(s, int) and s >= 0}

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, reasoning_effort=self.effort
        )
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def prefill(self, ids: list[int], start: int) -> torch.Tensor:
        return self.backend.prefill(ids, start)

    def parts(self, ids: list[int]) -> list[tuple[str, str, bool]]:
        return reply_parts(self.tokenizer.decode(ids, skip_special_tokens=False))

    def reply_text(self, ids: list[int]) -> str:
        """Thinking between markers; the closing marker waits for the part's end, so each
        text extends the previous one (the chat streams the difference)."""
        shown = []
        for kind, text, closed in self.parts(ids):
            if kind == "thinking":
                text = f"[thinking]{text}" + ("[/thinking]\n" if closed else "")
            shown.append(text)
        return "".join(shown)

    def assistant_message(self, ids: list[int]) -> dict:
        return assistant_message(self.tokenizer.decode(ids, skip_special_tokens=False))


def checkpoint_dir(path_or_repo: str | Path, tqdm_class=None, *, options: ChatOptions) -> Path:
    """``ref.checkpoint_dir`` of the files the backend reads: the speculative ones' drafter
    (``mtp.safetensors``) comes from the target's snapshot."""
    drafter = issubclass(BACKENDS[options.backend], Speculative)
    return ref.checkpoint_dir(path_or_repo, tqdm_class, drafter=drafter)


def check_options(options: ChatOptions):
    if options.backend not in BACKENDS:
        raise typer.BadParameter(f"Inkling backends: {sorted(BACKENDS)}, got {options.backend!r}")
    backend = BACKENDS[options.backend]
    if issubclass(backend, Speculative):
        if options.temperature > 0:
            raise typer.BadParameter("speculative backends decode greedily only (--temperature 0)")
        verify_tokens(options)
    elif options.verify_tokens is not None:
        raise typer.BadParameter("--verify-tokens applies to the speculative backends")
    if options.max_len % backend.max_len_multiple:
        raise typer.BadParameter(
            f"--max-len must be a multiple of {backend.max_len_multiple} on {options.backend}"
        )
    reasoning_effort(options)


def verify_tokens(options: ChatOptions) -> int:
    """A speculative backend's verify block ``k``: ``--verify-tokens`` (``SPEC_TOKENS`` when not
    given), one of the backend's ``verify_tokens``."""
    k = SPEC_TOKENS if options.verify_tokens is None else options.verify_tokens
    allowed = BACKENDS[options.backend].verify_tokens
    if k not in allowed:
        raise typer.BadParameter(f"--verify-tokens on {options.backend}: one of {list(allowed)}")
    return k


def reasoning_effort(options: ChatOptions) -> str | float:
    """The chat template's ``reasoning_effort``: a level name or a number in [0, 0.99]
    (``DEFAULT_EFFORT`` when not given)."""
    effort = DEFAULT_EFFORT if options.reasoning_effort is None else options.reasoning_effort
    if effort in EFFORTS:
        return effort
    try:
        value = float(effort)
    except ValueError:
        value = -1.0
    if not 0 <= value <= 0.99:
        raise typer.BadParameter(f"--reasoning-effort: one of {EFFORTS} or 0-0.99")
    return value


def load_chat_model(checkpoint: Path, options: ChatOptions) -> InklingChatModel:
    return InklingChatModel(checkpoint, options)


if __name__ == "__main__":
    run("thinkingmachines/Inkling-2-Mini")
