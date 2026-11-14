"""warpdrive paged prefill on a shape the chunk planner chains.

The piece chain has no paged routing, so a paged call at a chained shape must run the unchained
plan, d_v split included, not a chain plan stripped of its pieces. The result matches the
non-paged chunk call (which chains) within the plan tolerance.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, chunk_kda, paged_chunk_gdn, paged_chunk_kda
from attn_gym.linear._delta_rule.warpdrive.drivers import gdn, kda
from attn_gym.testing import cumulative_sequence_offsets, make_gdn_test_inputs
from attn_gym.testing.kda import assert_relative_rms_within, make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(), reason="warpdrive requires an SM100-family GPU"
)

WARPDRIVE = {"backend": "warpdrive"}
TOKENS, HEADS = 4096, 2


def _inputs(family):
    if family == "gdn":
        return make_gdn_test_inputs((TOKENS,), key_heads=1, value_heads=HEADS)[:5]
    q, k, value, gate, beta = make_kda_test_inputs(TOKENS, heads=HEADS)[:5]
    return q, k, value, gate, beta


@pytest.mark.parametrize("family", ["gdn", "kda"])
def test_paged_prefill_runs_the_uncut_plan_on_a_chained_shape(monkeypatch, family) -> None:
    driver = gdn if family == "gdn" else kda
    device = torch.device("cuda")
    if not driver.ForwardPlan.build(TOKENS, 1, HEADS, 128, device).chain:
        pytest.skip("the planner does not chain this shape on this GPU")
    num_sm = torch.cuda.get_device_properties(device).multi_processor_count
    assert driver.ForwardPlan.uncut(1, HEADS, 128, num_sm).tiles_per_head == 2

    plans = []
    uncut = driver.ForwardPlan.uncut.__func__

    def recording(cls, *args):
        plans.append(uncut(cls, *args))
        return plans[-1]

    monkeypatch.setattr(driver.ForwardPlan, "uncut", classmethod(recording))
    q, k, value, gate, beta = _inputs(family)
    offsets = cumulative_sequence_offsets([TOKENS])
    pool = torch.zeros(2, HEADS, 128, 128, device=device)  # slot 0 is the null route
    paged, chunk = (
        (paged_chunk_gdn, chunk_gdn) if family == "gdn" else (paged_chunk_kda, chunk_kda)
    )
    with torch.no_grad():
        output = paged(
            q,
            k,
            value,
            gate,
            beta,
            pool,
            torch.ones(1, dtype=torch.int32, device=device),
            cu_seqlens=offsets,
            has_initial_state=torch.zeros(1, dtype=torch.bool, device=device),
            kernel_options=WARPDRIVE,
        )
        expected, final = chunk(
            q,
            k,
            value,
            gate,
            beta,
            cu_seqlens=offsets,
            output_final_state=True,
            kernel_options=WARPDRIVE,
        )
    assert [(p.pieces, p.tiles_per_head) for p in plans] == [(0, 2)]
    assert_relative_rms_within(output, expected, f"{family} output", max_eps=1e-1 / 2**-8)
    assert_relative_rms_within(pool[1:], final, f"{family} final state", max_eps=1e-1 / 2**-8)
