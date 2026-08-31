"""Private forward validation for the CuTeDSL 4.7 scalar-GDN kernel."""

from __future__ import annotations

from typing import Literal

import pytest
import torch

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 GDN path requires nvidia-cutlass-dsl>=4.7",
)

from attn_gym.linear import chunk_gdn
from attn_gym.linear._delta_rule.mega.gdn_forward import run_forward
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.gdn import GatePattern
from attn_gym.testing.kda import assert_matches_low_precision_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 GDN path requires SM100 or SM103",
)

StateMode = Literal["none", "initial", "final", "initial_final"]


_FORWARD_CASES = (
    pytest.param(
        (1,),
        torch.float16,
        2,
        "mild",
        "none",
        id="t1-fp16-equal-mild-no-state",
    ),
    pytest.param(
        (63,),
        torch.bfloat16,
        1,
        "near_zero",
        "initial",
        id="t63-bf16-grouped-near-zero-initial",
    ),
    pytest.param(
        (64,),
        torch.bfloat16,
        2,
        "model_softplus",
        "final",
        id="t64-bf16-equal-model-gate-final",
    ),
    pytest.param(
        (65,),
        torch.float16,
        1,
        "isolated_negative_twenty",
        "initial_final",
        id="t65-fp16-grouped-isolated-spike-initial-final",
    ),
    pytest.param(
        (127,),
        torch.bfloat16,
        1,
        "mild",
        "initial_final",
        id="t127-bf16-grouped-mild-initial-final",
    ),
    pytest.param(
        (128,),
        torch.float16,
        2,
        "mild",
        "initial_final",
        id="t128-fp16-equal-mild-initial-final",
    ),
    pytest.param(
        (129,),
        torch.bfloat16,
        1,
        "uniform_negative_twenty",
        "none",
        id="t129-bf16-grouped-uniform-negative-twenty-no-state",
    ),
)


def reference_forward(
    inputs: tuple[torch.Tensor, ...],
    precision: torch.dtype,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the public eager GDN oracle at one operand precision."""
    q, k, value, gate, beta, _state, cu_seqlens = inputs
    return chunk_gdn(
        q.to(precision),
        k.to(precision),
        value.to(precision),
        gate.to(precision),
        beta.to(precision),
        None if initial_state is None else initial_state.to(precision),
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
        impl="reference",
    )


def assert_forward_matches_references(
    inputs: tuple[torch.Tensor, ...], initial_state: torch.Tensor | None, output_final_state: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compare the private launcher with source-precision and FP64 eager GDN."""
    source_output, source_state = reference_forward(
        inputs, torch.float32, initial_state, output_final_state
    )
    high_output, high_state = reference_forward(
        inputs, torch.float64, initial_state, output_final_state
    )
    actual_output, actual_state = run_forward(
        *inputs[:5],
        inputs[6],
        initial_state,
        scale=None,
        output_final_state=output_final_state,
    )
    assert_matches_low_precision_reference(
        actual_output,
        high_output,
        source_output,
        "output",
        source_dtype=inputs[0].dtype,
    )
    if output_final_state:
        assert actual_state is not None
        assert source_state is not None
        assert high_state is not None
        assert_matches_low_precision_reference(
            actual_state,
            high_state,
            source_state,
            "final state",
            source_dtype=inputs[0].dtype,
        )
    else:
        assert actual_state is None
    return actual_output, actual_state


@pytest.mark.parametrize(
    ("lengths", "dtype", "key_heads", "gate_pattern", "state_mode"), _FORWARD_CASES
)
def test_gdn_mega_forward_matches_curated_boundary_matrix(
    lengths: tuple[int, ...],
    dtype: torch.dtype,
    key_heads: int,
    gate_pattern: GatePattern,
    state_mode: StateMode,
) -> None:
    """Check every BT64 boundary with a non-Cartesian dtype, head, gate, and state matrix."""
    inputs = make_gdn_test_inputs(
        lengths,
        key_heads=key_heads,
        value_heads=2,
        gate_pattern=gate_pattern,
        dtype=dtype,
        seed=lengths[0] + key_heads,
    )
    initial_state = inputs[5] if state_mode in ("initial", "initial_final") else None
    output_final_state = state_mode in ("final", "initial_final")
    assert_forward_matches_references(inputs, initial_state, output_final_state)


def test_gdn_mega_packed_second_sequence_uses_its_own_descriptors() -> None:
    """Regression for sequence-relative TMA descriptors after the first packed sequence."""
    inputs = make_gdn_test_inputs(
        (1, 65, 0),
        key_heads=1,
        value_heads=2,
        gate_pattern="mild",
        dtype=torch.bfloat16,
        seed=101,
    )
    source_output, source_state = reference_forward(inputs, torch.float32, inputs[5], True)
    high_output, high_state = reference_forward(inputs, torch.float64, inputs[5], True)
    actual_output, actual_state = run_forward(
        *inputs[:5],
        inputs[6],
        inputs[5],
        scale=None,
        output_final_state=True,
    )
    assert actual_state is not None
    assert source_state is not None
    assert high_state is not None

    second_sequence = slice(1, 66)
    assert_matches_low_precision_reference(
        actual_output[:, second_sequence],
        high_output[:, second_sequence],
        source_output[:, second_sequence],
        "second packed sequence output",
        source_dtype=inputs[0].dtype,
    )
    assert_matches_low_precision_reference(
        actual_state[1],
        high_state[1],
        source_state[1],
        "second packed sequence final state",
        source_dtype=inputs[0].dtype,
    )
    torch.testing.assert_close(actual_state[2], inputs[5][2], rtol=0, atol=0)
