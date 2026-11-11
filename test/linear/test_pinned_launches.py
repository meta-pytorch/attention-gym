"""Linear operations launch pinned kernel configurations when they may not autotune."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("cutlass")

import attn_gym.linear.gdn.context_parallel as gdn_context_parallel
import attn_gym.linear.kda.context_parallel as kda_context_parallel
from attn_gym.linear import (
    TuningOptions,
    chunk_gdn,
    l2norm,
    paged_chunk_gdn,
    paged_chunk_kda,
    recurrent_gdn,
    recurrent_kda,
)
from attn_gym.linear.context_parallel import StagedOp
from attn_gym.linear.gdn import context_parallel_gdn
from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
from attn_gym.linear.kda import context_parallel_kda
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.gdn import make_gdn_test_inputs
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
    reason="fused delta-rule kernels require CUDA capability 8.0+",
)

TOKENS = 192
Tensors = tuple[torch.Tensor, ...]
StagesFactory = Callable[[TuningOptions | None], StagedOp]


def kda_inputs() -> Tensors:
    return make_kda_test_inputs(TOKENS, heads=2, normalize_qk=True)


def gdn_inputs() -> Tensors:
    return make_gdn_test_inputs(TOKENS, key_heads=2)[:5]


def recurrent_scan(scan: Callable, inputs: Callable[[], Tensors], options) -> Tensors:
    return scan(*inputs(), output_final_state=True, kernel_options=options)


def paged_chunk(operation: Callable, inputs: Callable[[], Tensors], options) -> Tensors:
    """Return the output and the state pool the call advanced in place."""
    q, k, v, gate, beta = inputs()
    # Slot zero is the padding slot; the sequence starts from the zero state in slot one.
    state_cache = torch.zeros(2, v.shape[2], v.shape[3], q.shape[3], device=q.device)
    state_indices = torch.tensor([1], device=q.device, dtype=torch.int32)
    with torch.no_grad():
        output = operation(q, k, v, gate, beta, state_cache, state_indices, kernel_options=options)
    return output, state_cache


def l2norm_output_and_gradient(options: TuningOptions | None) -> Tensors:
    x = kda_inputs()[0].requires_grad_()
    output = l2norm(x, kernel_options=options)
    return output, *torch.autograd.grad(output.float().square().sum(), x)


def direct_stages(prepare: Callable, prepare_backward: Callable) -> StagesFactory:
    return lambda options: StagedOp(
        partial(prepare, kernel_options=options),
        partial(prepare_backward, kernel_options=options),
    )


def context_parallel_stages(module, operation: Callable) -> StagesFactory:
    """The stages ``operation`` hands to the context-parallel recipe, without its collective."""

    def stages(options: TuningOptions | None) -> StagedOp:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(module, "context_parallel_chunk", lambda stages, *_, **__: stages)
            return operation(*[None] * 5, routing=None, group=None, kernel_options=options)

    return stages


def staged_forward(stages: StagedOp, inputs: Callable[[], Tensors]):
    """Return the forward handle and its output and final state."""
    cu_seqlens = cumulative_sequence_offsets([70, TOKENS - 70])
    prepared = stages.prepare(*inputs(), cu_seqlens=cu_seqlens)
    return prepared, *prepared.run(None, output_final_state=True)


def staged_step(
    stages: StagesFactory, inputs: Callable[[], Tensors], options: TuningOptions | None
) -> Tensors:
    """Run one forward and backward through both staged handles."""
    staged = stages(options)
    prepared, output, final_state = staged_forward(staged, inputs)
    grads = staged.prepare_backward(prepared.saved, output, None, scale=prepared.scale)
    return output, final_state, *grads.run(final_state)[:5]


STAGES = {
    "chunk_kda_prepare": (
        direct_stages(chunk_kda_prepare, chunk_kda_prepare_backward),
        kda_inputs,
    ),
    "chunk_gdn_prepare": (
        direct_stages(chunk_gdn_prepare, chunk_gdn_prepare_backward),
        gdn_inputs,
    ),
    "context_parallel_kda": (
        context_parallel_stages(kda_context_parallel, context_parallel_kda),
        kda_inputs,
    ),
    "context_parallel_gdn": (
        context_parallel_stages(gdn_context_parallel, context_parallel_gdn),
        gdn_inputs,
    ),
}
OPERATIONS = {
    "recurrent_kda": partial(recurrent_scan, recurrent_kda, kda_inputs),
    "recurrent_gdn": partial(recurrent_scan, recurrent_gdn, gdn_inputs),
    "paged_chunk_kda": partial(paged_chunk, paged_chunk_kda, kda_inputs),
    "paged_chunk_gdn": partial(paged_chunk, paged_chunk_gdn, gdn_inputs),
    "l2norm": l2norm_output_and_gradient,
    **{name: partial(staged_step, *case) for name, case in STAGES.items()},
}


@pytest.mark.parametrize("operation", OPERATIONS.values(), ids=OPERATIONS)
def test_deterministic_mode_gives_the_bits_of_autotune_false(operation, no_benchmarking, request):
    pinned = operation({"autotune": False})
    request.getfixturevalue("deterministic_algorithms")
    actual = operation(None)
    assert len(actual) == len(pinned)
    for index, (result, expected) in enumerate(zip(actual, pinned, strict=True)):
        assert torch.equal(result, expected), f"result {index} differs"


@pytest.mark.parametrize("name", ["chunk_kda_prepare", "chunk_gdn_prepare"])
def test_staged_runs_follow_deterministic_mode_enabled_after_prepare(
    name, isolated_winners, request
):
    stages, inputs = STAGES[name]
    staged = stages(None)
    prepared, output, final_state = staged_forward(staged, inputs)
    grads = staged.prepare_backward(prepared.saved, output, None, scale=prepared.scale)
    request.getfixturevalue("no_benchmarking")
    request.getfixturevalue("deterministic_algorithms")
    prepared.run(None, output_final_state=True)
    grads.run(final_state)


def l2norm_loss() -> tuple[torch.Tensor, Tensors]:
    x = kda_inputs()[0].requires_grad_()
    return l2norm(x).float().square().sum(), (x,)


def chunk_gdn_loss() -> tuple[torch.Tensor, Tensors]:
    # K = 64 takes the Triton backward on every architecture.
    inputs = make_gdn_test_inputs(TOKENS, key_dim=64, sigmoid_beta=True, requires_grad=True)[:5]
    output, _ = chunk_gdn(*inputs)
    return output.float().square().mean(), inputs


@pytest.mark.parametrize("forward", [l2norm_loss, chunk_gdn_loss], ids=["l2norm", "chunk_gdn"])
def test_backward_follows_deterministic_mode_enabled_after_the_forward(
    forward, isolated_winners, request
):
    loss, inputs = forward()
    request.getfixturevalue("no_benchmarking")
    request.getfixturevalue("deterministic_algorithms")
    assert all(torch.isfinite(gradient).all() for gradient in torch.autograd.grad(loss, inputs))


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the persistent ragged kernels require the TMA path",
)
def test_staged_kda_backward_accepts_the_forward_kernel_options():
    """Both stages accept the forward options; the pinned schedules agree bitwise here."""
    stages, inputs = STAGES["chunk_kda_prepare"]
    expected = staged_step(stages, inputs, {"autotune": False})
    actual = staged_step(stages, inputs, {"autotune": False, "schedule": "persistent"})
    for result, reference in zip(actual, expected, strict=True):
        assert torch.equal(result, reference)
