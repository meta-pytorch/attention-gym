"""Chat with a checkpoint decoded on the GPU by a megakernel (or a baseline backend), showing
output tokens per second.

    python -m attn_gym.warpdrive.examples.megakernel.chat --model Qwen/Qwen3-8B
    python -m attn_gym.warpdrive.examples.megakernel.chat --model Qwen/Qwen3-0.6B --backend compile
    python -m attn_gym.warpdrive.examples.megakernel.chat --model thinkingmachines/Inkling-2-Mini \
        --backend reference --prompt "Hi"

The checkpoint's ``config.json`` selects the model family (``FAMILIES``): Qwen3 dense and MoE
(``qwen_megakernel.chat``) or Inkling 2 Mini (``inkling_megakernel.chat``). A family builds a
``ChatModel``: its chat template, a prefill (one forward pass of the new prompt tokens into its
caches) and a ``DecodeBackend`` that decodes one token per step, or a ``MultiTokenBackend`` that
commits a variable number of tokens per step (speculative decoding).

Each turn prefills the new prompt tokens, then decodes. A backend whose step is CUDA-graph
capturable decodes inside one graph of steps feeding up to SYNC_EVERY positions per replay,
sampling on the device; the host reads the tokens (and each step's count) once per replay and
streams their text while the next replay decodes. Other backends run the same steps eagerly.
The conversation's caches carry over between turns; when the chat template re-renders an
earlier turn differently, the caches are rebuilt from the start.
"""

import importlib
import json
import math
import time
from dataclasses import dataclass, fields
from functools import partial
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.text import Text

SYNC_EVERY = 8  # positions a replay may feed between host reads of the sampled tokens

# config.json model_type -> the family's module: ``DEFAULT_BACKEND``, ``OPTIONS`` (the
# family-specific ``ChatOptions`` fields it reads), ``check_options(options)``,
# ``checkpoint_dir(model, tqdm_class, options=options)`` (the local directory of the files the
# backend reads, downloaded if needed) and ``load_chat_model(checkpoint, options)``, which builds
# the ChatModel.
FAMILIES = {
    "qwen3": "attn_gym.warpdrive.examples.qwen_megakernel.chat",
    "qwen3_moe": "attn_gym.warpdrive.examples.qwen_megakernel.chat",
    "inkling2_loop": "attn_gym.warpdrive.examples.inkling_megakernel.chat",
}

app = typer.Typer(add_completion=False, help=__doc__.splitlines()[0])
console = Console()


class DecodeBackend(Protocol):
    """One decode step: feeds ``token`` (one-element device tensor) at position ``pos`` (one-element
    int32 device tensor), writes the model's caches there, and returns fp32 logits ``(vocab,)``.
    ``capturable``: the step reads both on the device and syncs nothing, so it can run inside a
    CUDA graph."""

    capturable: bool

    def step(self, token: torch.Tensor, pos: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class MultiTokenBackend(Protocol):
    """A greedy step that commits ``1 .. tokens_per_step`` tokens (speculative decoding):
    ``decode`` feeds ``token`` at position ``pos + 1`` (``pos``: the last fed position, as
    ``DecodeBackend``'s callers keep it), then writes how many tokens it committed, ``count``,
    into ``out[0]`` and them into ``out[1 : 1 + count]`` (``out``: ``(1 + tokens_per_step,)``
    int64), the last one into ``token`` (not fed yet), and advances ``pos`` by ``count``. It
    may write the caches at up to ``tokens_per_step`` positions from ``pos + 1``.
    ``tokens_per_step`` is at most SYNC_EVERY; ``capturable`` as ``DecodeBackend``'s."""

    capturable: bool
    tokens_per_step: int

    def decode(self, token: torch.Tensor, pos: torch.Tensor, out: torch.Tensor) -> None: ...


class ChatModel(Protocol):
    """A model family's side of the chat."""

    backend: DecodeBackend | MultiTokenBackend
    label: str  # the backend as shown to the user
    max_len: int  # cache capacity in tokens
    stops: set[int]  # token ids that end a reply (not fed)

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        """The conversation rendered by the chat template, ready for the model's reply."""

    def prefill(self, ids: list[int], start: int) -> torch.Tensor:
        """Feed ``ids`` at positions ``start ..`` into the caches; the last token's logits."""

    def reply_text(self, ids: list[int]) -> str:
        """The text shown for a partial or complete reply."""

    def assistant_message(self, ids: list[int]) -> dict:
        """The reply as a chat-template message, for the next turn's history."""


@dataclass(frozen=True)
class ChatOptions:
    """Settings from the command line. Family-specific ones default to None, meaning not given:
    each family applies its defaults, and ``check_options`` rejects those it does not read."""

    backend: str
    max_len: int = 4096
    temperature: float = 0.0  # 0: greedy
    engine: str | None = None  # Qwen megakernel GEMV engine
    thinking: bool | None = None  # Qwen3 thinking mode
    yarn_factor: float | None = None  # Qwen3 YaRN
    reasoning_effort: str | None = None  # Inkling
    verify_tokens: int | None = None  # Inkling speculative verify block


def download(model: str, checkpoint_dir) -> Path:
    """The local checkpoint directory of ``model`` (resolved by the family's ``checkpoint_dir``),
    drawing any Hub download as one progress bar of bytes written to disk. huggingface_hub's own
    tqdm bars (files fetched, network bytes and bytes written) would be three interleaved lines,
    and one line per redraw under a rich status. ``HF_HUB_DISABLE_PROGRESS_BARS`` and
    non-terminal output hide the bar. A local checkpoint path skips both."""
    if Path(model).exists():
        return checkpoint_dir(model)
    from huggingface_hub.utils import are_progress_bars_disabled
    from tqdm.std import tqdm  # huggingface_hub's dependency, needed only to download

    progress = Progress(
        TextColumn(f"downloading [bold]{model}[/]"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
        disable=not console.is_terminal or are_progress_bars_disabled(),
    )

    class DiskBytes(tqdm):
        """Draws the snapshot's bytes-written bar on ``progress`` and hides the others: the
        file-count bar (unit "it") and the network-bytes bar, whose total is a guess."""

        def __init__(self, *args, **kwargs):
            kwargs.update(gui=True, disable=False)  # gui: count but print nothing
            super().__init__(*args, **kwargs)
            shown = self.unit == "B" and self.desc != "Downloading bytes"
            self.task = progress.add_task("", total=self.total or None) if shown else None

        def display(self, *_, **__):  # tqdm's refresh draws through here
            if self.task is not None:
                progress.update(self.task, completed=self.n, total=self.total or None)

        def clear(self, *_, **__):  # tqdm's clear writes to the terminal
            pass

    with progress:
        return checkpoint_dir(model, tqdm_class=DiskBytes)


def family_module(model: str):
    """The family module of a Hub repo or checkpoint directory, read from its ``config.json``
    (fetched alone, before any weights)."""
    path = Path(model)
    if path.is_file():
        path = path.parent
    if path.is_dir():
        config = path / "config.json"
    else:
        from huggingface_hub import hf_hub_download

        config = Path(hf_hub_download(model, "config.json"))
    model_type = json.loads(config.read_text()).get("model_type")
    if model_type not in FAMILIES:
        raise typer.BadParameter(
            f"unsupported model_type {model_type!r}; known: {sorted(FAMILIES)}"
        )
    return importlib.import_module(FAMILIES[model_type])


def check_options(family, options: ChatOptions):
    """Reject the family-specific options that ``family`` does not read (``OPTIONS``), then apply
    the family's own checks."""
    family_specific = [f.name for f in fields(ChatOptions) if f.default is None]
    given = [name for name in family_specific if getattr(options, name) is not None]
    if foreign := [f"--{name.replace('_', '-')}" for name in given if name not in family.OPTIONS]:
        raise typer.BadParameter(f"options of another model family: {', '.join(foreign)}")
    family.check_options(options)


class CacheFull(typer.BadParameter):
    """A turn whose prompt and reply would not fit the caches."""


@dataclass
class Turn:
    text: str
    prompt_tokens: int  # tokens prefilled this turn (past the cached prefix)
    prefill_s: float
    tokens: int
    decode_s: float


class Chat:
    """A conversation whose caches hold every token fed so far (positions 0 .. len - 1)."""

    def __init__(self, model: ChatModel):
        self.model = model
        self.messages: list[dict] = []
        self.cached: list[int] = []  # token ids whose cache entries are written
        self.pos = torch.zeros(1, device="cuda", dtype=torch.int32)
        self.graphs: dict[float, tuple] = {}  # temperature -> (graph, token in, rows out)

    @torch.no_grad()
    def prefill(self, ids: list[int]) -> tuple[torch.Tensor, int]:
        """Feed the tokens past the cached prefix; returns the last one's logits and how many
        tokens were fed."""
        if ids[: len(self.cached)] != self.cached or len(ids) == len(self.cached):
            self.cached = []  # the template re-rendered history: start over
        new = ids[len(self.cached) :]
        logits = self.model.prefill(new, len(self.cached))
        self.cached = list(ids)
        return logits, len(new)

    def steps(self, token: torch.Tensor, out: torch.Tensor, temperature: float):
        """One run's decode steps: each feeds ``token`` at the next position (``self.pos`` + 1,
        on the device) and puts the next token into ``token``, recording the tokens it committed
        in its row of ``out`` (``[count, tokens ...]``; one sampled token in a one-token step,
        whose count column stays 1)."""
        backend = self.model.backend
        multi = isinstance(backend, MultiTokenBackend)
        for row in out:
            if multi:
                backend.decode(token, self.pos, row)
                continue
            self.pos.add_(1)
            logits = backend.step(token, self.pos)
            token.copy_(sample(logits, temperature))
            row[1].copy_(token[0])

    @torch.no_grad()
    def decoder(self, temperature: float):
        """``(run, token in, rows out)``: ``run()`` decodes from ``token`` by SYNC_EVERY //
        tokens_per_step steps (rows ``[count, tokens ...]``), so that it feeds at most
        SYNC_EVERY positions, replaying one CUDA graph when the backend is capturable."""
        if temperature in self.graphs:
            return self.graphs[temperature]
        backend = self.model.backend
        per_step = self.tokens_per_step(temperature)
        token = torch.zeros(1, device="cuda", dtype=torch.long)
        out = torch.ones(SYNC_EVERY // per_step, 1 + per_step, device="cuda", dtype=torch.long)
        run = lambda: self.steps(token, out, temperature)
        if backend.capturable:
            # Warm-up and capture run real steps at positions 1 .. SYNC_EVERY (at most): the
            # cached prefix is gone, so the next prefill rebuilds from position 0.
            self.cached = []
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):  # warm up outside the capture
                self.pos.fill_(0)
                run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            run = graph.replay
        self.graphs[temperature] = (run, token, out)
        return self.graphs[temperature]

    def tokens_per_step(self, temperature: float) -> int:
        """The backend's tokens per step, checked against SYNC_EVERY and ``temperature``."""
        backend = self.model.backend
        per_step = backend.tokens_per_step if isinstance(backend, MultiTokenBackend) else 1
        if not 1 <= per_step <= SYNC_EVERY:
            raise ValueError(f"tokens_per_step must be in 1 .. {SYNC_EVERY}, got {per_step}")
        if per_step > 1 and temperature > 0:
            raise typer.BadParameter("multi-token (speculative) backends decode greedily only")
        return per_step

    def reply(self, prompt: str, max_new_tokens: int | None, temperature: float, on_token) -> Turn:
        """Answer ``prompt``, streaming through ``on_token(ids, elapsed)``. ``max_new_tokens``:
        None decodes until a stop token or full caches. A rejected turn leaves the history
        unchanged."""
        model = self.model
        check_temperature(temperature)
        self.tokens_per_step(temperature)
        self.messages.append({"role": "user", "content": prompt})
        ids = model.prompt_ids(self.messages)
        # A run feeds up to SYNC_EVERY positions, writing cache rows past the last kept token;
        # this much room keeps every row inside the caches.
        room = model.max_len - len(ids) - SYNC_EVERY - 1
        max_new_tokens = room if max_new_tokens is None else max_new_tokens
        if not 0 < max_new_tokens <= room:
            self.messages.pop()
            raise CacheFull(f"the conversation would pass max_len = {model.max_len}")
        run, fed, rows = self.decoder(temperature)  # before prefill: warm-up writes caches
        start = time.perf_counter()
        logits, prefilled = self.prefill(ids)
        token = sample(logits, temperature)
        torch.cuda.synchronize()
        prefill_s = time.perf_counter() - start
        # One run decodes up to SYNC_EVERY tokens on the GPU; the host then reads them, looks for
        # a stop token and streams text. A replayed graph decodes the next run meanwhile; an
        # eager run blocks the host, so the text streams first. Steps past a stop are discarded
        # (their cache rows are overwritten by the next prefill).
        stream_first = not model.backend.capturable
        fed.copy_(token)
        self.pos.fill_(len(self.cached) - 1)  # each step feeds at pos + 1
        out: list[int] = []
        tokens = [int(fed.item())]  # the prefill's token leads
        start = time.perf_counter()
        run()
        fed_count, done = 0, False  # tokens the runs fed
        while not done:
            for count, *committed in rows.tolist():  # waits for the run
                tokens += committed[:count]
                fed_count += count  # a step feeds as many tokens as it commits
            for t in tokens:
                if t in model.stops or len(out) == max_new_tokens:
                    done = True
                    break
                out.append(t)
            tokens = []
            if stream_first:
                on_token(out, time.perf_counter() - start)
            if not done:  # its buffers were read: the next run may overwrite them
                run()
            if not stream_first:
                on_token(out, time.perf_counter() - start)
        decode_s = time.perf_counter() - start
        self.cached += out[:fed_count]  # the last run's final token was not fed
        self.messages.append(model.assistant_message(out))
        return Turn(model.reply_text(out), prefilled, prefill_s, len(out), decode_s)

    def restart(self):
        self.messages, self.cached = [], []


def check_temperature(temperature: float):
    """A sampling temperature is finite; 0 (or less) decodes greedily."""
    if not math.isfinite(temperature):
        raise typer.BadParameter(f"--temperature must be finite, got {temperature}")


def sample(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """The next token as a one-element device tensor: no host sync."""
    if temperature <= 0:
        return logits.argmax().reshape(1)
    return torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1)


def stats_line(tokens: int, seconds: float, prompt_tokens: int, prefill_s: float) -> Text:
    rate = tokens / seconds if seconds > 0 else 0.0
    line = Text()
    line.append(f"{rate:,.0f} tok/s", style="bold green")
    line.append(f"  {tokens} tokens in {seconds:.2f} s", style="dim")
    if tokens:
        line.append(f"  {1e3 * seconds / tokens:.2f} ms/token", style="dim")
    line.append(f"  prefill {prompt_tokens} tokens in {1e3 * prefill_s:.0f} ms", style="dim")
    return line


def run_turn(chat: Chat, prompt: str, max_new_tokens: int | None, temperature: float):
    """Stream the reply as plain text (appended, so replies taller than the terminal keep
    scrolling), then its stats."""
    shown = 0  # characters of the reply already written

    def write(text: str):
        nonlocal shown
        console.out(text[shown:], end="", highlight=False)
        console.file.flush()
        shown = len(text)

    def on_token(out, elapsed):
        text = chat.model.reply_text(out)
        if not text.endswith("\ufffd"):  # wait for the rest of a multi-byte character
            write(text)

    turn = chat.reply(prompt, max_new_tokens, temperature, on_token)
    write(turn.text)
    console.print()
    console.print(stats_line(turn.tokens, turn.decode_s, turn.prompt_tokens, turn.prefill_s))


@app.command()
def main(
    model: str = typer.Option("Qwen/Qwen3-0.6B", help="HF repo or checkpoint directory"),
    backend: str | None = typer.Option(
        None,
        help="decode backend of the model's family (an unknown name lists them). Default: the "
        "family's DEFAULT_BACKEND.",
    ),
    engine: str | None = typer.Option(
        None, help="Qwen megakernel engine: cuda_core (default) or tcgen05"
    ),
    max_len: int = typer.Option(4096, help="cache capacity (tokens)"),
    yarn_factor: float | None = typer.Option(
        None,
        help="Qwen3 YaRN RoPE factor over the pretraining context (32K for Qwen3; 4 gives 128K, "
        "as Qwen documents). Default: max_len / 32K when max_len passes the checkpoint's "
        "max_position_embeddings and it has no RoPE scaling of its own; 0 turns YaRN off.",
    ),
    max_new_tokens: int | None = typer.Option(
        None, help="cap on tokens per reply; default: until a stop token or full caches"
    ),
    temperature: float = typer.Option(0.0, help="0: greedy"),
    thinking: bool | None = typer.Option(None, help="Qwen3 thinking mode (default off)"),
    reasoning_effort: str | None = typer.Option(
        None,
        help="Inkling thinking effort: none, minimal, low (default), medium, high, xhigh, max "
        "or 0-0.99",
    ),
    verify_tokens: int | None = typer.Option(
        None, help="Inkling speculative backends' verify block k (default 4)"
    ),
    prompt: str | None = typer.Option(None, help="answer one prompt and exit"),
):
    check_temperature(temperature)
    family = family_module(model)
    backend = family.DEFAULT_BACKEND if backend is None else backend
    options = ChatOptions(
        backend=backend,
        engine=engine,
        max_len=max_len,
        temperature=temperature,
        thinking=thinking,
        yarn_factor=yarn_factor,
        reasoning_effort=reasoning_effort,
        verify_tokens=verify_tokens,
    )
    check_options(family, options)
    # Before the spinner (it garbles bars); the backend chooses the checkpoint files.
    checkpoint = download(model, partial(family.checkpoint_dir, options=options))
    with console.status(f"loading {model} and preparing the {backend} backend"):
        chat_model = family.load_chat_model(checkpoint, options)
        torch.cuda.synchronize()
    chat = Chat(chat_model)
    if prompt is not None:
        run_turn(chat, prompt, max_new_tokens, temperature)
        return
    console.print(
        f"[bold]{model}[/] on {chat_model.label}, {max_len} tokens. Empty line or Ctrl-D to quit."
    )
    while True:
        try:
            text = console.input("[bold cyan]you>[/] ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text.strip():
            break
        try:
            run_turn(chat, text, max_new_tokens, temperature)
        except CacheFull:
            console.print(
                f"[yellow]The conversation filled max_len = {max_len}: starting over.[/]"
            )
            chat.restart()
            try:
                run_turn(chat, text, max_new_tokens, temperature)
            except CacheFull as error:
                console.print(f"[red]{error}[/]")


def run(default_model: str):
    """The CLI with another default ``--model`` (a family module's ``python -m`` entry)."""
    typer.main.get_command(app).main(default_map={"model": default_model})


if __name__ == "__main__":
    app()
