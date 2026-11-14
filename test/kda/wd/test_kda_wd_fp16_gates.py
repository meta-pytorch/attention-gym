"""FP16 warpdrive KDA across the model gate range, against FP64 and the fused FP16 backend.

FP16 cannot hold the in-chunk decay factors 2^g and 2^-g (their ratio reaches 2^121 at the
gate bound), so the FP16 kernels form the in-chunk products from TF32 decay operands.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_kda
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    cumulative_sequence_offsets,
    kda_reference,
)

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA path requires an SM100-family GPU",
)

LENGTHS = [200, 65, 1, 127]
HEADS = 2
D = 128
# Uniform per-token gates up to the gradient guideline, and Kimi-style bounded sigmoid gates.
GATES = [-0.1, -0.5, -2.0, -5.25, "sigmoid"]
FUSED = {"backend": "fused", "autotune": False}


@pytest.fixture(params=["uncut", "dv_split", "chain"])
def plan(request, monkeypatch) -> str:
    """Pin the forward and backward plans: the prefill, the shared-prep d_v split (prep and
    prep-prefill) or a four-piece chain (summary and prefill)."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    pieces, tiles = {"uncut": (0, 1), "dv_split": (0, 2), "chain": (4, 1)}[request.param]
    forward, backward = kda.ForwardPlan.build, kda.BackwardPlan.build
    monkeypatch.setattr(
        kda.ForwardPlan,
        "build",
        lambda *args: replace(forward(*args), pieces=pieces, tiles_per_head=tiles),
    )
    monkeypatch.setattr(
        kda.BackwardPlan, "build", lambda *args: replace(backward(*args), pieces=pieces)
    )
    return request.param


def _inputs(gate: float | str, stateful: bool, seed: int = 0) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, sum(LENGTHS), HEADS, D)

    def normal(*size):
        return torch.randn(*size, device="cuda", generator=generator)

    q = F.normalize(normal(*shape), dim=-1).half()
    k = F.normalize(normal(*shape), dim=-1).half()
    v = normal(*shape).half()
    if gate == "sigmoid":
        g = -5.45 * torch.sigmoid(2 * normal(*shape))
    else:
        g = torch.full(shape, gate, device="cuda")
    beta = torch.sigmoid(normal(*shape[:3]))
    state = normal(len(LENGTHS), HEADS, D, D) / 8 if stateful else None
    return q, k, v, g, beta, state


@pytest.mark.parametrize("stateful", [False, True], ids=["no-state", "state"])
@pytest.mark.parametrize("gate", GATES, ids=str)
def test_fp16_forward_matches_fp64_across_the_gate_range(plan, gate, stateful) -> None:
    q, k, v, g, beta, state = _inputs(gate, stateful)
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)

    def run(**options):
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None if state is None else state.clone(),
            cu_seqlens=cu_seqlens,
            output_final_state=stateful,
            **options,
        )

    with torch.no_grad():
        output, final_state = run(kernel_options={"backend": "warpdrive"})
        fused_output, fused_final = run(kernel_options=FUSED)
        expected, expected_final = kda_reference(
            *(t.double() for t in (q, k, v, g, beta)),
            initial_state=None if state is None else state.double(),
            cu_seqlens=cu_seqlens,
            output_final_state=stateful,
        )
    assert_relative_rms_within(output, expected, "output", max_eps=1.0, source_dtype=torch.half)
    assert_matches_low_precision_reference(
        output, expected, fused_output, "output", source_dtype=torch.half
    )
    if stateful:
        assert_matches_low_precision_reference(
            final_state, expected_final, fused_final, "final state", source_dtype=torch.half
        )
