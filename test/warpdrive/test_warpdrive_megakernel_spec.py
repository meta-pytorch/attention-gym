"""The speculative step (``megakernel.spec.SpecStep``) on a scripted target and drafter: the
device acceptance against a host reference for every accepted count, eager and replayed from a
CUDA graph, and ``prefill_blocks`` committing each prompt position."""

import pytest
import torch
import torch.nn.functional as F

from attn_gym.warpdrive.examples.megakernel.spec import SpecStep, prefill_blocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 97


def target_next(tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """The scripted target's greedy next token after ``tokens`` at ``positions``."""
    return (tokens * 5 + positions * 3 + 1) % VOCAB


class ScriptedTarget:
    """A ``TargetBlock`` whose argmax is ``target_next`` and whose features hold each row's
    position."""

    def __init__(self, tokens: int):
        self.tokens = tokens

    def __call__(self, tokens, pos):
        positions = pos.reshape(1).long() + torch.arange(self.tokens, device="cuda")
        logits = F.one_hot(target_next(tokens, positions), VOCAB).float()
        return logits, positions.view(-1, 1, 1).expand(-1, 4, 2).float()


class ScriptedDrafter:
    """A ``DraftModel`` whose proposal ``i`` is accepted up to ``schedule[i]`` drafts (cyclic):
    it drafts ``target_next``'s chain and perturbs that draft. Counts how often each position is
    committed and records the features of the most recent committed row."""

    def __init__(self, tokens: int, schedule: list[int], max_len: int):
        self.num_drafts = tokens - 1
        self.schedule = torch.tensor(schedule, device="cuda")
        self.calls = torch.zeros((), device="cuda", dtype=torch.long)
        self.committed = torch.zeros(max_len, device="cuda", dtype=torch.long)
        self.last = torch.zeros((), device="cuda")

    def propose(self, anchor, pos):
        accept = self.schedule.index_select(0, (self.calls % self.schedule.numel()).reshape(1))
        self.calls += 1
        drafts, token = [], anchor.reshape(1)
        for i in range(self.num_drafts):
            token = target_next(token, pos.reshape(1).long() + i)
            token = torch.where(accept == i, (token + 1) % VOCAB, token)
            drafts.append(token)
        return torch.cat(drafts)

    def commit(self, features, positions, last=None):
        last = torch.tensor(positions.numel() - 1, device="cuda") if last is None else last
        rows = torch.arange(positions.numel(), device="cuda") <= last
        self.committed.index_add_(0, positions, rows.long())
        self.last.copy_(features.index_select(0, last.reshape(1))[0, 0, 0])


def reference_steps(anchor: int, pos: int, schedule: list[int], steps: int):
    """Host reference of ``steps`` speculative steps from ``anchor`` at ``pos + 1``: the counts
    and the committed tokens, the greedy chain of ``target_next``."""
    counts = [schedule[i % len(schedule)] + 1 for i in range(steps)]
    tokens = [anchor]
    for _ in range(sum(counts)):
        tokens.append((tokens[-1] * 5 + (pos + len(tokens)) * 3 + 1) % VOCAB)
    return counts, tokens[1:]


def step_runner(step, token, pos, rows: torch.Tensor, graph: bool, reset):
    """A callable running one ``step`` per row of ``rows``: eagerly, or replaying a CUDA graph of
    them captured after a warm-up, after which ``reset()`` restores the state it advanced."""

    def run():
        for row in rows:
            step(token, pos, row)

    if not graph:
        return run
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    cuda_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cuda_graph):
        run()
    reset()
    return cuda_graph.replay


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
@torch.no_grad()
def test_device_accept_matches_host_reference(graph):
    """Every accepted count 0 .. k - 1: each step reports its count and the committed tokens
    (the target's greedy chain), advances the position by the count, leaves the bonus as the
    anchor and commits each fed position once, the most recent one as the drafter's context."""
    tokens = 4
    schedule = list(range(tokens))[::-1]
    steps, start_pos, anchor = 2 * tokens, 40, 11
    drafter = ScriptedDrafter(tokens, schedule, max_len=256)
    step = SpecStep(ScriptedTarget(tokens), drafter)
    token = torch.tensor([anchor], device="cuda")
    pos = torch.tensor([start_pos], device="cuda", dtype=torch.int32)
    rows = torch.zeros(steps, 1 + tokens, device="cuda", dtype=torch.long)

    def reset():
        token.fill_(anchor)
        pos.fill_(start_pos)
        drafter.calls.zero_()
        drafter.committed.zero_()

    step_runner(step, token, pos, rows, graph, reset)()
    counts, expected = reference_steps(anchor, start_pos, schedule, steps)
    got = rows.tolist()
    assert [row[0] for row in got] == counts
    assert [t for count, *row in got for t in row[:count]] == expected
    assert pos.item() == start_pos + sum(counts)
    assert token.item() == expected[-1]
    fed = range(start_pos + 1, start_pos + 1 + sum(counts))
    committed = drafter.committed.nonzero().flatten().tolist()
    assert committed == list(fed) and drafter.committed.max().item() == 1
    assert drafter.last.item() == fed[-1]


@pytest.mark.parametrize("prompt", [3, 9])
@torch.no_grad()
def test_prefill_blocks_commits_each_prompt_position(prompt):
    """Blocks of 4 over prompts shorter than and past a block (padded last block):
    the last token's logits, each prompt position committed once and none past it, and the last
    one as the drafter's context."""
    start, tokens = 30, 4
    drafter = ScriptedDrafter(tokens, [0], max_len=64)
    ids = torch.arange(7, 7 + prompt, device="cuda")
    logits = prefill_blocks(ScriptedTarget(tokens), drafter, ids, start)
    last = start + prompt - 1
    assert logits.argmax().item() == target_next(ids[-1], torch.tensor(last)).item()
    committed = drafter.committed.nonzero().flatten().tolist()
    assert committed == list(range(start, last + 1)) and drafter.committed.max().item() == 1
    assert drafter.last.item() == last
