"""A differentiated warpdrive KDA forward runs the kernels its backward recomputes.

Small packs share a decay/factor prep pass between the d_v split's two value tiles when the
forward is inference-only. The backward recomputes the per-tile formulation, which rounds the
chunk states apart from the shared prep, so a forward that records a graph runs per-tile.
``kernel_options["exact_recompute"]`` overrides the inference, for ``chunk_kda`` and for the
staged handles that context parallelism differentiates inside its own autograd function.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_kda

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA path requires an SM100-family GPU",
)

WARPDRIVE = {"backend": "warpdrive"}
# Shapes whose inference forward shares the prep on a GB300 (152 SMs): the first pairs it with a
# chain backward, the second with the warmup backward.
PREP_SHAPES = [pytest.param(2048, 8, id="1x2048xH8"), pytest.param(272, 2, id="1x272xH2")]
PRECISIONS = [
    *(pytest.param(torch.bfloat16, value, id=value) for value in ("bf16", "tf32+bf16", "tf32")),
    pytest.param(torch.float16, None, id="fp16"),
]


def _inputs(tokens, heads, dtype=torch.bfloat16, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    shape = (1, tokens, heads, 128)
    q, k = (F.normalize(normal(*shape), dim=-1).to(dtype) for _ in range(2))
    v = normal(*shape).to(dtype)
    gate = -5 * torch.rand(*shape, device="cuda", generator=gen)
    beta = torch.rand(*shape[:3], device="cuda", generator=gen)
    return q, k, v, gate, beta


def _leaves(inputs, requires_grad=True):
    return [tensor.detach().clone().requires_grad_(requires_grad) for tensor in inputs]


@pytest.mark.parametrize("dtype, value", PRECISIONS)
@pytest.mark.parametrize("tokens, heads", PREP_SHAPES)
def test_handoff_gradients_match_one_call(tokens, heads, dtype, value) -> None:
    """The last 16 tokens' outputs and gradients of one differentiated call equal those of a
    call started from the final state of a differentiated call over the first ``T - 16``: the
    backward's checkpoints are the states the forward ran. The first shape's chain backward
    composes piece summaries, which round apart from the uncut forward; these gates decay that
    difference away before the last chunk."""
    options = WARPDRIVE if value is None else {**WARPDRIVE, "intra_chunk_dtypes": value}
    inputs = _inputs(tokens, heads, dtype)
    cut = tokens - 16
    d_output = torch.randn_like(inputs[2])

    leaves = _leaves(inputs)
    output, _ = chunk_kda(*leaves, kernel_options=options)
    expected = torch.autograd.grad(output, leaves, d_output)

    head = _leaves(t[:, :cut] for t in inputs)
    _, handoff = chunk_kda(*head, output_final_state=True, kernel_options=options)
    tail = _leaves(t[:, cut:] for t in inputs)
    tail_output, _ = chunk_kda(*tail, initial_state=handoff.detach(), kernel_options=options)
    actual = torch.autograd.grad(tail_output, tail, d_output[:, cut:].contiguous())

    assert torch.equal(tail_output, output[:, cut:])
    for name, got, want in zip(("dq", "dk", "dv", "dgate", "dbeta"), actual, expected):
        assert torch.equal(got, want[:, cut:]), name


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_only_inference_forwards_share_the_prep(compiled, monkeypatch) -> None:
    """Under grad the forward returns the per-tile bits; under ``no_grad``, ``inference_mode``
    or without inputs that require grad it keeps the shared prep's. ``torch.compile`` decides
    while tracing."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    inputs = _inputs(2048, 8)
    run = torch.compile(chunk_kda, fullgraph=True) if compiled else chunk_kda

    def forward(requires_grad=True):
        return run(*_leaves(inputs, requires_grad), kernel_options=WARPDRIVE)[0].detach()

    training = forward()
    without_grad_inputs = forward(requires_grad=False)
    with torch.no_grad():
        no_grad = forward()
    with torch.inference_mode():
        inference = chunk_kda(*inputs, kernel_options=WARPDRIVE)[0]
    monkeypatch.setattr(kda, "PREP_TILE_FRACTION", 0.0)
    with torch.no_grad():
        per_tile = chunk_kda(*inputs, kernel_options=WARPDRIVE)[0]

    assert torch.equal(training, per_tile)
    assert not torch.equal(no_grad, per_tile)
    assert torch.equal(without_grad_inputs, no_grad)
    assert torch.equal(inference, no_grad)


def _per_tile(monkeypatch):
    """Make every later warpdrive KDA forward run per-tile, as the backward recomputes."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    monkeypatch.setattr(kda, "PREP_TILE_FRACTION", 0.0)


def test_exact_recompute_overrides_the_grad_mode_inference(monkeypatch) -> None:
    """``True`` runs per-tile under ``no_grad``; ``False`` shares the prep under grad."""
    inputs = _inputs(2048, 8)

    def forward(exact_recompute=None, requires_grad=True):
        options = dict(WARPDRIVE)
        if exact_recompute is not None:
            options["exact_recompute"] = exact_recompute
        return chunk_kda(*_leaves(inputs, requires_grad), kernel_options=options)[0].detach()

    exact_under_no_grad = forward(True, requires_grad=False)
    inexact_under_grad = forward(False)
    with torch.no_grad():
        shared = forward()
    _per_tile(monkeypatch)
    with torch.no_grad():
        per_tile = forward()

    assert not torch.equal(shared, per_tile)
    assert torch.equal(exact_under_no_grad, per_tile)
    assert torch.equal(inexact_under_grad, shared)


# One sequence whose staged summaries run uncut on a GB300, so ``run`` takes the public op's
# plan, and whose inference forward shares the prep.
STAGED_SHAPE = (256, 8)


def _staged_run(inputs, kernel_options=WARPDRIVE):
    """Prepare and run one staged forward from a zero entry state under ``no_grad``, as one CP
    rank does inside its autograd function."""
    from attn_gym.linear.kda.stages import chunk_kda_prepare

    with torch.no_grad():
        prepared = chunk_kda_prepare(*inputs, kernel_options=kernel_options)
        assert prepared.pieces is None
        entry = torch.zeros(1, inputs[0].shape[2], 128, 128, device="cuda")
        return prepared.run(entry, output_final_state=True)


def test_staged_run_infers_exact_recompute_at_prepare(monkeypatch) -> None:
    """A handle prepared under autograd runs per-tile; ``False`` keeps the shared prep and
    ``True`` drops it, whatever the grad mode."""
    from attn_gym.linear.kda.stages import chunk_kda_prepare

    inputs = _inputs(*STAGED_SHAPE)
    entry = torch.zeros(1, inputs[0].shape[2], 128, 128, device="cuda")

    def run(exact_recompute=None):
        options = dict(WARPDRIVE)
        if exact_recompute is not None:
            options["exact_recompute"] = exact_recompute
        prepared = chunk_kda_prepare(*_leaves(inputs), kernel_options=options)
        assert prepared.pieces is None
        with torch.no_grad():
            return prepared.run(entry, output_final_state=True)

    inferred_under_grad = run()
    inexact_under_grad = run(False)
    exact_under_no_grad = _staged_run(inputs, {**WARPDRIVE, "exact_recompute": True})
    shared = _staged_run(inputs)
    _per_tile(monkeypatch)
    per_tile = _staged_run(inputs)

    assert not torch.equal(shared[0], per_tile[0])
    for got, want in (
        (inferred_under_grad, per_tile),
        (inexact_under_grad, shared),
        (exact_under_no_grad, per_tile),
    ):
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_context_parallel_kda_runs_the_exact_forward(monkeypatch) -> None:
    """One rank owning the whole document: its staged forward runs inside an autograd function,
    where grad mode is off, so ``context_parallel_kda`` must ask for the per-tile kernels."""
    import attn_gym.linear.context_parallel as cp_impl
    from attn_gym.linear.context_parallel import ContextParallelPlan
    from attn_gym.linear.kda import context_parallel_kda

    tokens, _ = STAGED_SHAPE
    inputs = _inputs(*STAGED_SHAPE)
    monkeypatch.setattr(cp_impl.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(cp_impl.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(cp_impl, "_all_gather_slots", lambda slots, group: slots.unsqueeze(0))
    routing = ContextParallelPlan.from_fragments((0, tokens), [[(0, tokens)]], 0).routing("cuda")
    output, _ = context_parallel_kda(
        *_leaves(inputs), routing=routing, group=None, kernel_options=WARPDRIVE
    )
    shared = _staged_run(inputs)
    _per_tile(monkeypatch)
    per_tile = _staged_run(inputs)

    assert not torch.equal(shared[0], per_tile[0])
    assert torch.equal(output.detach(), per_tile[0])
