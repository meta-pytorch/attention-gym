"""Chat with a Qwen3 checkpoint decoded by the megakernel (or torch.compile), showing output
tokens per second.

    python -m attn_gym.warpdrive.examples.qwen_megakernel.chat --model Qwen/Qwen3-8B
    python -m attn_gym.warpdrive.examples.qwen_megakernel.chat --backend compile --prompt "Hi"

Each turn prefills the new prompt tokens with the PyTorch reference (one forward pass into the
KV cache), then decodes on the GPU with the chosen backend: ``megakernel`` (``--engine`` cuda_core or
tcgen05) or ``compile`` (torch.compile of the batched PyTorch step, ``bench.decode_step_fast_batch``,
at batch one). Either runs inside the same CUDA graph of SYNC_EVERY decode steps.
The conversation's KV cache carries over between turns; when the chat template re-renders an
earlier turn differently the cache is rebuilt from the start.
"""

import time
from dataclasses import dataclass

import torch
import typer
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from attn_gym.warpdrive.examples.qwen_megakernel import QwenMegakernel
from attn_gym.warpdrive.examples.qwen_megakernel.bench import (
    decode_step_fast_batch,
    prepare_fast_weights,
)
from attn_gym.warpdrive.examples.qwen_megakernel.ref import forward_tokens, load_qwen3_weights

SYNC_EVERY = 8  # decode steps between host reads of the sampled tokens

app = typer.Typer(add_completion=False, help=__doc__.splitlines()[0])
console = Console()


@dataclass
class Turn:
    text: str
    prompt_tokens: int
    prefill_s: float
    tokens: int
    decode_s: float


class Megakernel:
    """One decode step on ``QwenMegakernel``: ``step`` reads the token and position from device
    buffers, writes K/V at the position and returns the logits."""

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

    def __init__(self, weights, max_len: int, caches, engine: str):
        self.packed = prepare_fast_weights(weights, max_len)
        layers = weights.config.num_layers
        self.views = [tuple(c[layer][None] for layer in range(layers)) for c in caches]
        self.fn = torch.compile(decode_step_fast_batch)
        self.step(torch.zeros(1, device="cuda", dtype=torch.long), torch.ones(1, device="cuda"))

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.fn(self.packed, token, pos.long(), *self.views)[0]


BACKENDS = {"megakernel": Megakernel, "compile": Compiled}


class Chat:
    """A conversation whose KV cache holds every token fed so far (positions 0 .. len - 1)."""

    def __init__(self, model: str, backend: str, engine: str, max_len: int, thinking: bool):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.weights = load_qwen3_weights(model)
        config = self.weights.config
        shape = (config.num_layers, config.num_kv_heads, max_len, config.head_dim)
        # Zeroed: the compiled step's masked SDPA reads rows past the position (0 * NaN is NaN).
        self.caches = [torch.zeros(shape, device="cuda", dtype=torch.bfloat16) for _ in "kv"]
        self.backend = BACKENDS[backend](self.weights, max_len, self.caches, engine)
        self.max_len, self.thinking = max_len, thinking
        self.messages: list[dict] = []
        self.cached: list[int] = []  # token ids whose K/V are in the caches
        stops = [self.tokenizer.eos_token_id]
        stops += [self.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        self.stops = {s for s in stops if isinstance(s, int) and s >= 0}
        self.pos = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.graphs: dict[float, tuple] = {}  # temperature -> (graph, token in, tokens out)
        torch.cuda.synchronize()

    def prompt_ids(self) -> list[int]:
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def prefill(self, ids: list[int]) -> torch.Tensor:
        """Feed the tokens past the cached prefix; returns the last one's logits."""
        if ids[: len(self.cached)] != self.cached or len(ids) == len(self.cached):
            self.cached = []  # the template re-rendered history: start over
        new = ids[len(self.cached) :]
        tokens = torch.tensor(new, device="cuda")
        logits = forward_tokens(
            self.weights, tokens, len(self.cached), *self.caches, dtype=torch.bfloat16
        )
        self.cached = list(ids)
        return logits

    def decode_graph(self, temperature: float):
        """A CUDA graph of SYNC_EVERY decode steps: each feeds the token in ``token`` at the next
        position (``self.pos`` + 1, on the device), runs the backend's step, samples the next token
        into ``token`` and records it in ``out``. One replay decodes SYNC_EVERY tokens."""
        if temperature not in self.graphs:
            token = torch.zeros(1, device="cuda", dtype=torch.long)
            out = torch.zeros(SYNC_EVERY, device="cuda", dtype=torch.long)

            def steps():
                for i in range(SYNC_EVERY):
                    self.pos.add_(1)
                    logits = self.backend.step(token, self.pos)
                    token.copy_(self.sample(logits, temperature))
                    out[i].copy_(token[0])

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):  # warm up outside the capture
                self.pos.fill_(0)
                steps()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                steps()
            self.graphs[temperature] = (graph, token, out)
        return self.graphs[temperature]

    def sample(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """The next token as a one-element device tensor: no host sync."""
        if temperature <= 0:
            return logits.argmax().reshape(1)
        return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)

    def reply(self, prompt: str, max_new_tokens: int, temperature: float, on_token) -> Turn:
        self.messages.append({"role": "user", "content": prompt})
        ids = self.prompt_ids()
        if len(ids) + max_new_tokens >= self.max_len:
            raise typer.BadParameter(f"the conversation would pass max_len = {self.max_len}")
        graph, fed, sampled = self.decode_graph(temperature)  # before prefill: warm-up writes K/V
        start = time.perf_counter()
        token = self.sample(self.prefill(ids), temperature)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - start
        # One graph replay decodes SYNC_EVERY tokens on the GPU; the host then reads them to
        # stream text and look for a stop token. Steps past a stop are discarded (their K/V rows
        # are overwritten by the next prefill).
        pos = len(self.cached)
        if pos + max_new_tokens + SYNC_EVERY >= self.max_len:
            raise typer.BadParameter(f"the conversation would pass max_len = {self.max_len}")
        fed.copy_(token)
        self.pos.fill_(pos - 1)  # the graph adds 1 before each step
        out: list[int] = []
        replays = 0
        done = False
        start = time.perf_counter()
        while not done:
            first = int(fed.item()) if not replays else None  # the prefill's token leads
            graph.replay()
            replays += 1
            tokens = ([first] if first is not None else []) + sampled.tolist()
            for t in tokens:
                if t in self.stops or len(out) == max_new_tokens:
                    done = True
                    break
                out.append(t)
            on_token(out, time.perf_counter() - start)
        decode_s = time.perf_counter() - start
        self.cached += out[: replays * SYNC_EVERY]  # the last replay's final token was not fed
        text = self.tokenizer.decode(out, skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": text})
        return Turn(text, len(ids), prefill_s, len(out), decode_s)


def stats_line(tokens: int, seconds: float, prompt_tokens: int, prefill_s: float) -> Text:
    rate = tokens / seconds if seconds > 0 else 0.0
    line = Text()
    line.append(f"{rate:,.0f} tok/s", style="bold green")
    line.append(f"  {tokens} tokens in {seconds:.2f} s", style="dim")
    if tokens:
        line.append(f"  {1e3 * seconds / tokens:.2f} ms/token", style="dim")
    line.append(f"  prefill {prompt_tokens} tokens in {1e3 * prefill_s:.0f} ms", style="dim")
    return line


def run_turn(chat: Chat, prompt: str, max_new_tokens: int, temperature: float):
    answer = Text()
    with Live(console=console, refresh_per_second=15, transient=False) as live:

        def on_token(out, elapsed):
            answer.plain = chat.tokenizer.decode(out, skip_special_tokens=True)
            live.update(Group(answer, stats_line(len(out), elapsed, 0, 0.0)))

        turn = chat.reply(prompt, max_new_tokens, temperature, on_token)
        live.update(
            Group(
                Text(turn.text),
                stats_line(turn.tokens, turn.decode_s, turn.prompt_tokens, turn.prefill_s),
            )
        )


@app.command()
def main(
    model: str = typer.Option("Qwen/Qwen3-0.6B", help="HF repo or checkpoint directory"),
    backend: str = typer.Option("megakernel", help="megakernel or compile (torch.compile)"),
    engine: str = typer.Option("cuda_core", help="megakernel engine: cuda_core or tcgen05"),
    max_len: int = typer.Option(4096, help="KV cache capacity (tokens)"),
    max_new_tokens: int = typer.Option(512, help="tokens per reply"),
    temperature: float = typer.Option(0.0, help="0: greedy"),
    thinking: bool = typer.Option(False, help="Qwen3 thinking mode"),
    prompt: str | None = typer.Option(None, help="answer one prompt and exit"),
):
    if backend not in BACKENDS:
        raise typer.BadParameter(f"backend must be one of {sorted(BACKENDS)}")
    label = f"the {engine} megakernel" if backend == "megakernel" else "torch.compile"
    with console.status(f"loading {model} and compiling {label}"):
        chat = Chat(model, backend, engine, max_len, thinking)
    if prompt is not None:
        run_turn(chat, prompt, max_new_tokens, temperature)
        return
    console.print(f"[bold]{model}[/] on {label}. Empty line or Ctrl-D to quit.")
    while True:
        try:
            text = console.input("[bold cyan]you>[/] ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text.strip():
            break
        run_turn(chat, text, max_new_tokens, temperature)


if __name__ == "__main__":
    app()
