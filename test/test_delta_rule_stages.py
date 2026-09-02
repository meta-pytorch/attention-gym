"""Staged KDA/GDN primitives and the context-parallel recipe simulated in one process."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.context_parallel import (
    ContextParallelPlan,
    compose_entry_states,
    compose_exit_cotangents,
    grad_summary_slots,
    summary_slots,
)
from attn_gym.linear.gdn import chunk_gdn
from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
from attn_gym.linear.kda import chunk_kda
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.linear.state_summary import compose_summaries, merge_state, neutral_summary

HEAD_DIM = 128

requires_kda_target = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
    reason="fused delta-rule kernels require CUDA capability 8.0+",
)


class Op(NamedTuple):
    """One delta-rule variant: its public op plus staged entry points and input generator."""

    chunk: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    reference: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    prepare: Callable[..., object]
    prepare_backward: Callable[..., object]
    key_heads: int
    value_heads: int
    vector_gate: bool  # KDA gates are per channel [1, T, HV, K]; GDN gates are per head [1, T, HV]

    def make_inputs(self, tokens: int, seed: int, dtype: torch.dtype = torch.bfloat16):
        """Unit-norm Q/K, random V, negative log gates, and beta in (0, 1)."""
        generator = torch.Generator(device="cuda").manual_seed(seed)
        randn = partial(torch.randn, device="cuda", generator=generator)
        q = torch.nn.functional.normalize(randn(1, tokens, self.key_heads, HEAD_DIM), dim=-1)
        k = torch.nn.functional.normalize(randn(1, tokens, self.key_heads, HEAD_DIM), dim=-1)
        v = randn(1, tokens, self.value_heads, HEAD_DIM)
        gate_shape = (1, tokens, self.value_heads) + ((HEAD_DIM,) if self.vector_gate else ())
        gate = -torch.rand(gate_shape, device="cuda", generator=generator)
        beta = torch.rand(1, tokens, self.value_heads, device="cuda", generator=generator)
        return q.to(dtype), k.to(dtype), v.to(dtype), gate, beta


KDA = Op(
    chunk_kda,
    partial(chunk_kda, impl="reference"),
    chunk_kda_prepare,
    chunk_kda_prepare_backward,
    key_heads=2,
    value_heads=2,
    vector_gate=True,
)
GDN = Op(
    partial(chunk_gdn, impl="fused"),
    partial(chunk_gdn, impl="reference"),
    chunk_gdn_prepare,
    chunk_gdn_prepare_backward,
    key_heads=2,
    value_heads=2,
    vector_gate=False,
)
op_param = pytest.mark.parametrize(
    "op",
    [
        pytest.param(KDA, id="kda"),
        pytest.param(GDN, id="gdn"),
        pytest.param(GDN._replace(key_heads=1), id="gdn-gqa"),
    ],
)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> None:
    # Sharding changes accumulation order and low-precision outputs round at the tensor's
    # magnitude, so allow one input-precision ulp of the largest expected value.
    eps = torch.finfo(dtype).eps
    expected = expected.float()
    torch.testing.assert_close(
        actual.float(), expected, atol=eps * max(expected.abs().max().item(), 1.0), rtol=eps
    )


def relative_rms_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative RMS error against an FP32 oracle."""
    return ((actual.float() - reference).square().mean() / reference.square().mean()).sqrt().item()


def test_summary_algebra_composes_like_sequential_merges():
    generator = torch.Generator().manual_seed(0)
    first = torch.randn(2, 6, 4, generator=generator)
    then = torch.randn(2, 6, 4, generator=generator)
    state = torch.randn(2, 2, 4, generator=generator)

    sequential = merge_state(merge_state(state, first), then)
    composed = compose_summaries(first, then)
    torch.testing.assert_close(merge_state(state, composed), sequential)
    # Spelled out: (A0, B0) then (A1, B1) is (A0 @ A1, B0 @ A1 + B1), packed [bias; transition].
    bias, transition = first[:, :2], first[:, 2:]
    torch.testing.assert_close(
        composed, torch.cat((bias @ then[:, 2:] + then[:, :2], transition @ then[:, 2:]), dim=1)
    )

    identity = neutral_summary(2, 2, 4, device="cpu")
    torch.testing.assert_close(merge_state(state, identity), state)
    torch.testing.assert_close(compose_summaries(identity, first), first)
    torch.testing.assert_close(compose_summaries(first, identity), first)


@requires_kda_target
@op_param
def test_prepare_run_matches_chunk_op_with_packed_initial_state(op):
    """``prepare`` + ``run`` is the public op's own kernel sequence, so results are bitwise equal."""
    q, k, v, gate, beta = op.make_inputs(320, seed=1)
    cu_seqlens = torch.tensor([0, 100, 320], dtype=torch.int32, device="cuda")
    initial_state = torch.randn(2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")

    expected, expected_state = op.chunk(
        q, k, v, gate, beta, initial_state, cu_seqlens=cu_seqlens, output_final_state=True
    )
    prepared = op.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
    output, final_state = prepared.run(initial_state, output_final_state=True)

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(final_state, expected_state, atol=0, rtol=0)


@requires_kda_target
@op_param
@pytest.mark.parametrize("tokens", [128, 70], ids=["complete-chunks", "partial-tail"])
def test_prepare_run_matches_chunk_op_unpacked_with_strided_qkv(op, tokens):
    """Unpacked inputs and last-dim-strided Q/K/V follow the same normalization as the op."""
    q, k, v, gate, beta = op.make_inputs(tokens, seed=6)
    q, k, v = (torch.stack((tensor, tensor), dim=-1)[..., 0] for tensor in (q, k, v))
    assert q.stride(-1) == 2

    # The fused ops accept either layout and the stages normalize strided inputs themselves; the
    # reference gets compact copies so the same call also serves ops that reject strides.
    expected, _ = op.chunk(q.contiguous(), k.contiguous(), v.contiguous(), gate, beta)
    output, final_state = op.prepare(q, k, v, gate, beta).run()

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert final_state is None


@requires_kda_target
@op_param
def test_state_summary_matches_zero_and_identity_probes(op):
    """``final(H) = H @ A + B`` for every H, so zero and identity probes recover ``[B; A]``."""
    q, k, v, gate, beta = op.make_inputs(200, seed=2)
    cu_seqlens = torch.tensor([0, 72, 200], dtype=torch.int32, device="cuda")
    prepared = op.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)

    zero = torch.zeros(2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    identity = torch.eye(HEAD_DIM, device="cuda").expand_as(zero).contiguous()
    _, bias = op.chunk(q, k, v, gate, beta, zero, cu_seqlens=cu_seqlens, output_final_state=True)
    _, shifted = op.chunk(
        q, k, v, gate, beta, identity, cu_seqlens=cu_seqlens, output_final_state=True
    )
    for index, (start, stop) in enumerate(((0, 72), (72, 200))):
        summary = prepared.state_summary(start, stop)
        assert summary.shape == (op.value_heads, 2 * HEAD_DIM, HEAD_DIM)
        assert_close(summary[:, :HEAD_DIM], bias[index], torch.bfloat16)
        assert_close(summary[:, HEAD_DIM:], shifted[index] - bias[index], torch.bfloat16)


@requires_kda_target
def test_state_summary_rejects_ranges_outside_the_stream():
    q, k, v, gate, beta = KDA.make_inputs(128, seed=3)
    prepared = chunk_kda_prepare(q, k, v, gate, beta)
    with pytest.raises(ValueError, match="summary range"):
        prepared.state_summary(64, 200)
    with pytest.raises(ValueError, match="summary range"):
        prepared.state_summary(64, 64)


class RankResult(NamedTuple):
    plan: ContextParallelPlan
    token_ids: torch.Tensor
    output: torch.Tensor
    final_state: torch.Tensor
    grads: tuple[torch.Tensor, ...]  # dq, dk, dv, dgate, dbeta


def simulate_context_parallel(
    op: Op, global_cu_seqlens, token_ranges, inputs, d_output, d_final_state
) -> list[RankResult]:
    """Run the recipe for every rank in one process, replacing each all-gather with a stack."""
    q, _, v, _, _ = inputs
    device = q.device
    ranks = range(len(token_ranges))
    plans = [
        ContextParallelPlan.from_token_ranges(global_cu_seqlens, token_ranges, rank)
        for rank in ranks
    ]
    token_ids = [plan.global_token_ids(device) for plan in plans]
    prepared = [
        op.prepare(
            *(tensor[:, token_ids[r]] for tensor in inputs),
            cu_seqlens=torch.tensor(plans[r].cu_seqlens, dtype=torch.int32, device=device),
        )
        for r in ranks
    ]
    neutral = neutral_summary(v.shape[2], v.shape[-1], q.shape[-1], device=device)

    gathered = torch.stack([summary_slots(prepared[r], plans[r], neutral) for r in ranks])
    initial_states = [compose_entry_states(gathered, plans[r]) for r in ranks]
    forward = [prepared[r].run(initial_states[r], output_final_state=True) for r in ranks]

    # The loss's state cotangent reaches only the fragments that end their sequence.
    d_exit_states = [
        torch.stack(
            [
                d_final_state[fragment.sequence]
                if index in plan.terminal
                else torch.zeros_like(d_final_state[0])
                for index, fragment in enumerate(plan.fragments)
            ]
        )
        for plan in plans
    ]
    grads = [
        op.prepare_backward(prepared[r].saved, d_output[:, token_ids[r]], initial_states[r])
        for r in ranks
    ]
    gathered = torch.stack(
        [grad_summary_slots(grads[r], d_exit_states[r], plans[r], neutral) for r in ranks]
    )
    input_grads = [
        grads[r].run(compose_exit_cotangents(gathered, d_exit_states[r], plans[r]))[:5]
        for r in ranks
    ]
    return [RankResult(plans[r], token_ids[r], *forward[r], input_grads[r]) for r in ranks]


# Sequences [0, 40), [40, 232), [232, 384); each entry lists one rank's global token ranges.
CU_SEQLENS = (0, 40, 232, 384)
LAYOUTS = [
    pytest.param([[(0, 192)], [(192, 384)]], id="contiguous-2"),
    pytest.param([[(0, 96), (288, 384)], [(96, 192), (192, 288)]], id="zigzag-2"),
    pytest.param(
        [[(0, 64), (320, 384)], [(64, 128), (256, 320)], [(128, 192), (192, 256)]], id="zigzag-3"
    ),
    # Uneven ownership: rank 0 holds one short piece, rank 1 the rest in reversed order.
    pytest.param([[(96, 160)], [(160, 384), (0, 96)]], id="uneven-reordered"),
]


@requires_kda_target
@op_param
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_simulated_context_parallel_matches_unsharded_op(op, layout, dtype):
    q, k, v, gate, beta = op.make_inputs(CU_SEQLENS[-1], seed=4, dtype=dtype)
    inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in (q, k, v, gate, beta))
    offsets = torch.tensor(CU_SEQLENS, dtype=torch.int32, device="cuda")

    output, final_state = op.chunk(*inputs, cu_seqlens=offsets, output_final_state=True)
    generator = torch.Generator(device="cuda").manual_seed(5)
    d_output = torch.randn(output.shape, device="cuda", generator=generator).to(output.dtype)
    d_final_state = torch.randn(final_state.shape, device="cuda", generator=generator)
    expected_grads = torch.autograd.grad(
        (output, final_state), inputs, grad_outputs=(d_output, d_final_state)
    )

    # Sharding must not cost accuracy: measure both paths against the FP32 eager oracle.
    f32 = tuple(tensor.detach().float().requires_grad_() for tensor in (q, k, v, gate, beta))
    ref_output, ref_final = op.reference(*f32, cu_seqlens=offsets, output_final_state=True)
    ref_grads = torch.autograd.grad(
        (ref_output, ref_final), f32, grad_outputs=(d_output.float(), d_final_state)
    )

    results = simulate_context_parallel(
        op, CU_SEQLENS, layout, (q, k, v, gate, beta), d_output, d_final_state
    )
    for result in results:
        ids = result.token_ids
        assert_close(result.output, output[:, ids], dtype)
        for index in result.plan.terminal:
            sequence = result.plan.fragments[index].sequence
            assert_close(result.final_state[index], final_state[sequence], dtype)
        for actual, expected, reference in zip(
            result.grads, expected_grads, ref_grads, strict=True
        ):
            assert_close(actual, expected[:, ids], dtype)
            sharded = relative_rms_error(actual, reference[:, ids])
            unsharded = relative_rms_error(expected[:, ids], reference[:, ids])
            # Small fragments give noisy per-rank statistics; a systematic precision loss in the
            # summary path shows up as a multiple, not a fraction, of the unsharded error.
            assert sharded <= 1.5 * unsharded + 1e-6, (sharded, unsharded)
