"""Qwen3 in the megakernel chat (``attn_gym.warpdrive.examples.megakernel.chat``), decoded by
the megakernel (``--engine`` cuda_core or tcgen05) or torch.compile.

    python -m attn_gym.warpdrive.examples.qwen_megakernel.chat --model Qwen/Qwen3-8B
    python -m attn_gym.warpdrive.examples.qwen_megakernel.chat --backend compile --prompt "Hi"

Prefill runs the PyTorch reference (one forward pass into the KV cache). The ``compile`` backend
is torch.compile of the batched PyTorch step (``bench.decode_step_fast_batch``) at batch one.
Both backends are CUDA-graph capturable.
"""

import json
from dataclasses import replace
from pathlib import Path

import torch
import typer

from attn_gym.warpdrive.examples.megakernel.chat import ChatOptions, run
from attn_gym.warpdrive.examples.qwen_megakernel import QwenMegakernel, ref
from attn_gym.warpdrive.examples.qwen_megakernel.bench import (
    decode_step_fast_batch,
    prepare_fast_weights,
)
from attn_gym.warpdrive.examples.qwen_megakernel.ref import (
    YarnScaling,
    forward_tokens,
    load_qwen3_weights,
)

__all__ = ["DEFAULT_BACKEND", "OPTIONS", "check_options", "checkpoint_dir", "load_chat_model"]

# Qwen3's 40,960-position checkpoints were pretrained on 32,768 tokens: Qwen's YaRN setup
# scales from there. Checkpoints with longer native contexts (Qwen3-2507) scale from theirs.
QWEN3_POSITIONS, QWEN3_PRETRAINED_CONTEXT = 40960, 32768


class Megakernel:
    """One decode step on ``QwenMegakernel``: ``step`` reads the token and position from device
    buffers, writes K/V at the position and returns the logits."""

    capturable = True

    def __init__(self, weights, max_len: int, caches, engine: str):
        self.runner, self.caches = QwenMegakernel(weights, max_len, engine=engine), caches
        self.runner(weights, 0, 1, *caches)  # compile

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        self.runner.step_info[0].copy_(token[0])
        self.runner.step_info[1].copy_(pos[0])
        self.runner.launch(*self.caches)
        return self.runner.logits


class Compiled:
    """One decode step of ``torch.compile(decode_step_fast_batch)`` at batch one: static shapes
    (SDPA over the whole cache with a length mask), so it compiles once and graph-captures."""

    capturable = True

    def __init__(self, weights, max_len: int, caches, engine: str):
        self.packed = prepare_fast_weights(weights, max_len)
        layers = weights.config.num_layers
        self.views = [tuple(c[layer][None] for layer in range(layers)) for c in caches]
        self.fn = torch.compile(decode_step_fast_batch)
        self.step(torch.zeros(1, device="cuda", dtype=torch.long), torch.ones(1, device="cuda"))

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.fn(self.packed, token, pos.long(), *self.views)[0]


BACKENDS = {"megakernel": Megakernel, "compile": Compiled}
DEFAULT_BACKEND = "megakernel"
OPTIONS = frozenset({"engine", "thinking", "yarn_factor"})
DEFAULT_ENGINE = "cuda_core"


class QwenChatModel:
    def __init__(self, checkpoint: Path, options: ChatOptions):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.weights = load_qwen3_weights(
            checkpoint, interleave_gate_up=options.backend == "megakernel"
        )
        config = self.weights.config
        hf_config = json.loads((checkpoint / "config.json").read_text())
        native = hf_config["max_position_embeddings"]
        original = QWEN3_PRETRAINED_CONTEXT if native == QWEN3_POSITIONS else native
        max_len, yarn_factor = options.max_len, options.yarn_factor
        if yarn_factor is None:  # only past the native context, as Qwen recommends
            long = max_len > native and config.rope_scaling is None
            yarn_factor = max_len / original if long else None
        if yarn_factor is not None:
            params = {"factor": yarn_factor, "original_max_position_embeddings": original}
            yarn = YarnScaling.from_hf(params, original) if yarn_factor else None
            config = replace(config, rope_scaling=yarn)
            self.weights = replace(self.weights, config=config)
        shape = (config.num_layers, config.num_kv_heads, max_len, config.head_dim)
        # Zeroed: the compiled step's masked SDPA reads rows past the position (0 * NaN is NaN).
        self.caches = [torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in "kv"]
        engine = DEFAULT_ENGINE if options.engine is None else options.engine
        self.backend = BACKENDS[options.backend](self.weights, max_len, self.caches, engine)
        self.max_len, self.thinking = max_len, bool(options.thinking)
        stops = [self.tokenizer.eos_token_id]
        stops += [self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self.stops = {s for s in stops if isinstance(s, int) and s >= 0}
        self.label = (
            f"the {engine} megakernel" if options.backend == "megakernel" else "torch.compile"
        )
        if config.rope_scaling:
            self.label += f", YaRN x{config.rope_scaling.factor:g}"

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=self.thinking
        )
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def prefill(self, ids: list[int], start: int) -> torch.Tensor:
        tokens = torch.tensor(ids, device="cuda")
        return forward_tokens(self.weights, tokens, start, *self.caches, dtype=torch.bfloat16)

    def reply_text(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def assistant_message(self, ids: list[int]) -> dict:
        return {"role": "assistant", "content": self.reply_text(ids)}


def checkpoint_dir(path_or_repo: str | Path, tqdm_class=None, *, options: ChatOptions) -> Path:
    """``ref.checkpoint_dir``: every backend reads the same files."""
    return ref.checkpoint_dir(path_or_repo, tqdm_class)


def check_options(options: ChatOptions):
    if options.backend not in BACKENDS:
        raise typer.BadParameter(f"Qwen backends: {sorted(BACKENDS)}, got {options.backend!r}")
    if options.yarn_factor and options.yarn_factor < 1:
        raise typer.BadParameter(
            f"--yarn-factor must be 0 or at least 1, got {options.yarn_factor}"
        )


def load_chat_model(checkpoint: Path, options: ChatOptions) -> QwenChatModel:
    return QwenChatModel(checkpoint, options)


if __name__ == "__main__":
    run("Qwen/Qwen3-0.6B")
