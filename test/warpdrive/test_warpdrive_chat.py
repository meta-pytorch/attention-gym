"""The shared chat driver with a fake model: stop tokens, incremental prefill, the cache
rebuild after a new CUDA graph's warm-up overwrote cached positions, streaming, multi-token
steps, the rejected temperatures, options of another model family, and the CLI's family
selection on a local checkpoint or its error for a model it cannot read."""

import importlib
import importlib.util
import json
import re
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import typer
from typer.testing import CliRunner

from attn_gym.warpdrive.cache import CacheStorageError
from attn_gym.warpdrive.examples.megakernel import chat as chat_module
from attn_gym.warpdrive.examples.megakernel.chat import (
    FAMILIES,
    SYNC_EVERY,
    CacheFull,
    Chat,
    ChatOptions,
    app,
    check_options,
    compile_kernels,
    prepare,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
# The family modules import their megakernel runners.
needs_cutedsl = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="the families' runners need CuTeDSL"
)

VOCAB, STOP = 64, 63


class CountingBackend:
    """Capturable: the next token is ``(pos + 1) % STOP`` until position ``stop_at``, then STOP.
    ``stop_at`` is a device tensor, so changing it in place reaches captured graphs."""

    capturable = True

    def __init__(self, stop_at: int):
        self.stop_at = torch.tensor(stop_at, device="cuda")

    def kernels(self):
        return []

    def step(self, token, pos):
        nxt = torch.where(pos >= self.stop_at, STOP, (pos.long() + 1) % STOP)
        return F.one_hot(nxt.long(), VOCAB)[0].float()


class FakeModel:
    def __init__(self, stop_at: int):
        self.backend = CountingBackend(stop_at)
        self.label, self.max_len, self.stops = "fake", 1024, {STOP}
        self.prefills: list[tuple[int, int]] = []  # (start, count)

    def prompt_ids(self, messages):
        ids = []
        for message in messages:
            ids += [1, 2, 3] if message["role"] == "user" else message["ids"]
        return ids

    def prefill(self, ids, start):
        self.prefills.append((start, len(ids)))
        last = start + len(ids) - 1
        return F.one_hot(torch.tensor((last + 1) % STOP, device="cuda"), VOCAB).float()

    def reply_text(self, ids):
        return " ".join(map(str, ids))

    def assistant_message(self, ids):
        return {"role": "assistant", "ids": list(ids)}


@needs_cuda
def test_stop_incremental_prefill_and_graph_rebuild():
    model = FakeModel(stop_at=12)
    chat = Chat(model)
    turn = chat.reply("hi", None, 0.0, lambda *_: None)
    # Prompt positions 0-2; the prefill samples 3, then steps feed 3, 4, ... until the stop.
    assert turn.text.split() == [str(t) for t in range(3, 13)]
    assert model.prefills == [(0, 3)]

    model.backend.stop_at.fill_(20)
    turn = chat.reply("again", None, 0.0, lambda *_: None)
    assert model.prefills[-1] == (13, 3), "same graph: only the new prompt is prefilled"
    assert turn.text.split() == [str(t) for t in range(16, 21)]

    chat.reply("hot", 3, 0.5, lambda *_: None)  # a new graph: warm-up ran at positions 1-8
    assert model.prefills[-1][0] == 0, "the cached prefix must be rebuilt after warm-up"


class EagerCountingBackend(CountingBackend):
    """``CountingBackend`` run eagerly, counting its steps on the host."""

    capturable = False

    def __init__(self, stop_at: int):
        super().__init__(stop_at)
        self.steps = 0

    def step(self, token, pos):
        self.steps += 1
        return super().step(token, pos)


@needs_cuda
def test_eager_backend_streams_each_run_before_the_next():
    """An eager run blocks the host, so each run's tokens stream before the next run starts."""
    model = FakeModel(stop_at=100)
    model.backend = backend = EagerCountingBackend(100)
    streamed = []
    chat = Chat(model)
    chat.reply(
        "hi", 2 * SYNC_EVERY + 4, 0.0, lambda out, _: streamed.append((len(out), backend.steps))
    )
    # The prefill's token, then a run's SYNC_EVERY tokens per call.
    assert streamed == [(1 + r * SYNC_EVERY, r * SYNC_EVERY) for r in (1, 2)] + [
        (2 * SYNC_EVERY + 4, 3 * SYNC_EVERY)
    ]


class SpecCountingBackend(CountingBackend):
    """``CountingBackend``'s tokens committed by a multi-token step: ``counts[i]`` tokens at
    call ``i`` (cyclic), recording the positions each call writes (``tokens_per_step`` of them)."""

    tokens_per_step = 3  # does not divide SYNC_EVERY: runs of SYNC_EVERY // 3 steps

    def __init__(self, stop_at: int, counts: list[int]):
        super().__init__(stop_at)
        self.counts = torch.tensor(counts, device="cuda")
        self.calls = torch.zeros((), device="cuda", dtype=torch.long)
        self.written = torch.zeros(1024, device="cuda", dtype=torch.bool)

    def decode(self, token, pos, out):
        count = self.counts.index_select(0, (self.calls % self.counts.numel()).reshape(1))[0]
        self.calls += 1
        fed = pos.long() + 1 + torch.arange(self.tokens_per_step, device="cuda")
        self.written[fed] = True
        nxt = torch.where(fed >= self.stop_at, STOP, (fed + 1) % STOP)
        out[0].copy_(count)
        out[1:].copy_(nxt)
        token.copy_(nxt.gather(0, count.reshape(1) - 1))
        pos.add_(count)


@needs_cuda
def test_multi_token_steps_commit_variable_counts():
    """A multi-token backend's variable counts reach the reply in order, stops and the token cap
    cut inside a step's tokens, the next prefill starts after the fed tokens, and a run writes no
    cache row SYNC_EVERY or more positions past the kept tokens."""
    model = FakeModel(stop_at=7)
    model.backend = backend = SpecCountingBackend(7, [1, 3, 2])
    chat = Chat(model)
    turn = chat.reply("hi", None, 0.0, lambda *_: None)  # stops in a run's first step
    assert turn.text.split() == [str(t) for t in range(3, 8)]
    assert len(chat.cached) == 8
    assert not backend.written[len(chat.cached) + SYNC_EVERY :].any()

    backend.stop_at.fill_(100)
    backend.calls.zero_()
    turn = chat.reply("again", 4, 0.0, lambda *_: None)  # the first run commits 1 + 3 tokens
    assert model.prefills[-1] == (8, 3)
    assert turn.text.split() == ["11", "12", "13", "14"]
    chat.reply("more", 2, 0.0, lambda *_: None)
    assert model.prefills[-1] == (15, 3), "the reply's 4 tokens were all fed"

    history = list(chat.messages)
    with pytest.raises(typer.BadParameter, match="greedily"):
        chat.reply("hot", 3, 0.5, lambda *_: None)
    assert chat.messages == history, "a rejected turn must leave the history unchanged"


class StubCompile:
    """A ``jit_cache`` compile function of kernel names: the kernels on disk (``cached``) and
    those this process compiled (``compiled_here``), which a disk cache keeps."""

    def __init__(self, cached: set[str], disk: bool = True):
        self.cached, self.disk, self.compiled_here = set(cached), disk, []

    def __call__(self, kernel: str):
        self.compiled_here.append(kernel)
        if self.disk:
            self.cached.add(kernel)

    def is_cached(self, kernel: str) -> bool:
        return kernel in self.cached

    def disk_cache_enabled(self) -> bool:
        return self.disk


def batch_compiler(monkeypatch, compile_fn: StubCompile, error: Exception | None = None):
    """The kernels the chat hands ``precompile_calls``, which stores them, or with ``error``
    stores the first and raises it."""
    batched = []

    def precompile_calls(calls, workers, progress):
        kernels = [kernel for _, kernel in calls]
        batched.extend(kernels)
        compile_fn.cached |= set(kernels[:1] if error else kernels)
        if error:
            raise error

    monkeypatch.setattr(chat_module, "precompile_calls", precompile_calls)
    return batched


@pytest.mark.parametrize(
    ("cached", "disk", "batched", "here"),
    [
        ({"c"}, True, ["a", "b"], []),  # several cold kernels compile in parallel
        ({"b", "c"}, True, [], ["a"]),  # one compiles here: no compiler process to start
        ({"c"}, False, [], ["a", "b", "c"]),  # without the disk cache all compile here
    ],
)
def test_compile_kernels_compiles_what_the_disk_cache_lacks(
    monkeypatch, cached, disk, batched, here
):
    compile_fn = StubCompile(cached, disk)
    sent = batch_compiler(monkeypatch, compile_fn)
    compile_kernels([(compile_fn, kernel) for kernel in "abc"])
    assert (sent, compile_fn.compiled_here) == (batched, here)


def test_compile_kernels_compiles_here_what_a_failed_disk_cache_left(monkeypatch):
    """A batch that could not use the disk cache leaves its other kernels to this process."""
    compile_fn = StubCompile(set())
    batch_compiler(monkeypatch, compile_fn, CacheStorageError("unwritable"))
    compile_kernels([(compile_fn, "a"), (compile_fn, "b")])
    assert compile_fn.compiled_here == ["b"]


def test_compile_kernels_raises_a_compile_failure(monkeypatch):
    compile_fn = StubCompile(set())
    batch_compiler(monkeypatch, compile_fn, RuntimeError("bad kernel"))
    with pytest.raises(RuntimeError, match="bad kernel"):
        compile_kernels([(compile_fn, "a"), (compile_fn, "b")])
    assert compile_fn.compiled_here == []


@needs_cuda
def test_prepare_compiles_cold_kernels_then_captures(capsys):
    """Before the first turn, ``prepare`` compiles only the backend's kernels missing from the
    disk cache, saying nothing when every one is cached, then captures the decode: the first
    reply prefills from position 0 and captures nothing more."""
    compile_fn = StubCompile({"cached"})
    model = FakeModel(stop_at=12)
    model.backend.kernels = lambda: [(compile_fn, "cached"), (compile_fn, "cold")]
    chat = Chat(model)
    prepare(chat, 0.0)
    assert compile_fn.compiled_here == ["cold"]
    assert "compiled 1 kernel " in capsys.readouterr().out
    assert list(chat.graphs) == [0.0]

    prepare(Chat(model), 0.0)
    assert compile_fn.compiled_here == ["cold"] and capsys.readouterr().out == ""
    graph = chat.graphs[0.0]
    chat.reply("hi", None, 0.0, lambda *_: None)
    assert model.prefills == [(0, 3)] and chat.graphs == {0.0: graph}


class RecordingBackend(CountingBackend):
    """``CountingBackend`` marking the positions its steps write."""

    def __init__(self, stop_at: int):
        super().__init__(stop_at)
        self.written = torch.zeros(1024, device="cuda", dtype=torch.bool)

    def step(self, token, pos):
        self.written[pos.long()] = True
        return super().step(token, pos)


@needs_cuda
def test_chat_caches_must_hold_the_warm_up_run():
    """The warm-up run writes positions 1 .. SYNC_EVERY: a chat whose caches end before that is
    rejected before any step; one whose caches hold just that writes nothing past them."""
    model = FakeModel(stop_at=100)
    model.backend = backend = RecordingBackend(100)
    model.max_len = SYNC_EVERY
    with pytest.raises(CacheFull):
        Chat(model)
    model.max_len = SYNC_EVERY + 1
    prepare(Chat(model), 0.0)
    assert backend.written.nonzero().flatten().tolist() == list(range(1, SYNC_EVERY + 1))


@needs_cuda
def test_reply_rejects_nonfinite_temperature():
    model = FakeModel(stop_at=12)
    chat = Chat(model)
    chat.reply("hi", None, 0.0, lambda *_: None)
    history = list(chat.messages)
    for temperature in (float("nan"), float("inf")):
        with pytest.raises(typer.BadParameter, match="finite"):
            chat.reply("hot", 3, temperature, lambda *_: None)
    assert chat.messages == history, "a rejected turn must leave the history unchanged"


@pytest.mark.parametrize("temperature", ["nan", "inf"])
def test_cli_rejects_nonfinite_temperature_before_loading(temperature):
    """The default model is never resolved: the option fails first."""
    result = CliRunner().invoke(app, ["--temperature", temperature])
    assert result.exit_code == 2 and "finite" in result.output


def test_family_options_reject_another_familys():
    """A family-specific option the family does not read fails before its own checks run; the
    options it reads pass through to them."""
    checked = []
    family = SimpleNamespace(OPTIONS=frozenset({"engine"}), check_options=checked.append)
    options = ChatOptions("megakernel", engine="tcgen05")
    check_options(family, options)
    assert checked == [options]
    with pytest.raises(typer.BadParameter, match="--thinking, --yarn-factor"):
        check_options(family, ChatOptions("megakernel", thinking=False, yarn_factor=2.0))
    assert len(checked) == 1


@needs_cutedsl
def test_families_declare_the_options_they_read():
    """Every family-specific option belongs to some family, and a family lists only such
    options: a misspelled or missing ``OPTIONS`` entry would reject the family's own flag."""
    specific = {f.name for f in fields(ChatOptions) if f.default is None}
    declared = [importlib.import_module(m).OPTIONS for m in set(FAMILIES.values())]
    assert all(options <= specific for options in declared)
    assert set().union(*declared) == specific


class Loaded(Exception):
    """Raised by a stubbed ``load_chat_model``: the CLI got as far as loading."""


@needs_cutedsl
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], ChatOptions("megakernel")),
        (
            ["--backend", "compile", "--engine", "tcgen05", "--thinking", "--max-len", "8192"],
            ChatOptions("compile", max_len=8192, engine="tcgen05", thinking=True),
        ),
    ],
)
def test_cli_loads_a_local_qwen_checkpoint(tmp_path, monkeypatch, args, expected):
    """A checkpoint directory's ``config.json`` selects the Qwen family, whose default backend
    and the given options reach its loader with that directory, without Hub access."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    family = importlib.import_module(FAMILIES["qwen3"])
    loaded = {}

    def load_chat_model(checkpoint, options):
        loaded.update(checkpoint=checkpoint, options=options)
        raise Loaded

    monkeypatch.setattr(family, "load_chat_model", load_chat_model)
    result = CliRunner().invoke(app, ["--model", str(tmp_path), *args])
    assert isinstance(result.exception, Loaded), result.output
    assert loaded == {"checkpoint": tmp_path, "options": expected}


@needs_cutedsl
def test_cli_rejects_an_unknown_qwen_backend_before_loading(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    result = CliRunner().invoke(app, ["--model", str(tmp_path), "--backend", "nope"])
    assert result.exit_code == 2 and "Qwen backends" in result.output


def cli_error(args: list[str]) -> str:
    """The CLI's usage error for ``args``, its panel's borders and line wraps removed."""
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    return " ".join(re.sub("[│╭╮╰╯─]", " ", result.output).split())


def assert_names_supported_models(message: str):
    assert "Invalid value for --model" in message
    for repo in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-30B-A3B", "thinkingmachines/Inkling-2-Mini"):
        assert repo in message
    assert "local checkpoint directory" in message


@pytest.mark.parametrize("error", ["RemoteEntryNotFoundError", "LocalEntryNotFoundError"])
def test_cli_rejects_a_model_the_hub_cannot_serve(monkeypatch, error):
    """A Hub failure fetching config.json is a usage error naming the supported models, not a
    traceback: a missing repo or file (``--model help``) says so, any other failure (offline)
    keeps the Hub's message."""
    huggingface_hub = pytest.importorskip("huggingface_hub")
    exception = getattr(importlib.import_module("huggingface_hub.errors"), error)

    def hf_hub_download(repo_id, filename):
        message = f"{repo_id}/{filename}"
        if issubclass(exception, huggingface_hub.errors.HfHubHTTPError):
            httpx = pytest.importorskip("httpx2")
            request = httpx.Request("GET", f"https://huggingface.co/{message}")
            raise exception(message, response=httpx.Response(404, request=request))
        raise exception(message)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    message = cli_error(["--model", "help"])
    if error == "RemoteEntryNotFoundError":
        expected = (
            f"'help' is neither a checkpoint directory nor a Hub repo with a config.json ({error})"
        )
    else:
        expected = "could not fetch 'help''s config.json: help/config.json"
    assert expected in message
    assert_names_supported_models(message)


@pytest.mark.parametrize(
    "config", [None, "{not json", "[]", "null", '{"model_type": []}', '{"model_type": "llama"}']
)
def test_cli_rejects_a_checkpoint_directory_it_cannot_read(tmp_path, config):
    """A directory without a config.json, with an unreadable one, one that is not an object with
    a model_type string, or of an unsupported model type."""
    if config is not None:
        (tmp_path / "config.json").write_text(config)
    assert_names_supported_models(cli_error(["--model", str(tmp_path)]))
