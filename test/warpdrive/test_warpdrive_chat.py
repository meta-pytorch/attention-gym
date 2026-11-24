"""The shared chat driver with a fake model: stop tokens, incremental prefill, the cache
rebuild after a new CUDA graph's warm-up overwrote cached positions, streaming, multi-token
steps, the rejected temperatures, options of another model family, and the CLI's family
selection on a local checkpoint."""

import importlib
import importlib.util
import json
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import typer
from typer.testing import CliRunner

from attn_gym.warpdrive.examples.megakernel.chat import (
    FAMILIES,
    SYNC_EVERY,
    Chat,
    ChatOptions,
    app,
    check_options,
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
