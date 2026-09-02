"""Private backward validation for the CuTeDSL 4.7 scalar-GDN kernels."""

from __future__ import annotations

from random import Random
from typing import Literal

import pytest
import torch

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 GDN path requires nvidia-cutlass-dsl>=4.7",
)

from attn_gym.linear import chunk_gdn
from attn_gym.linear._delta_rule.mega.gdn_backward import chunk_gdn_bwd_mega_packed
from attn_gym.linear._delta_rule.mega.gdn_forward import run_forward
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.gdn import GatePattern
from attn_gym.testing.kda import assert_matches_low_precision_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 GDN path requires SM100 or SM103",
)

BackwardStateMode = Literal["none", "initial", "cotangent"]


_BACKWARD_CASES = (
    pytest.param(
        (65,),
        torch.bfloat16,
        1,
        "isolated_negative_twenty",
        "none",
        id="t65-bf16-grouped-isolated-spike",
    ),
    pytest.param(
        (128,),
        torch.float16,
        2,
        "model_softplus",
        "initial",
        id="t128-fp16-equal-model-gate-initial",
    ),
    pytest.param(
        (63, 65, 0),
        torch.bfloat16,
        1,
        "near_zero",
        "cotangent",
        id="packed-bf16-grouped-near-zero-state-cotangent",
    ),
)

_FUZZ_CASES = (
    pytest.param(7, torch.bfloat16, 1, "mild", id="seed7-bf16-grouped-mild"),
    pytest.param(29, torch.float16, 2, "model_softplus", id="seed29-fp16-equal-model-gate"),
)


def reference_forward(
    inputs: tuple[torch.Tensor, ...],
    precision: torch.dtype,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate stateful public eager GDN at source or FP64 operand precision."""
    q, k, value, gate, beta, _state, cu_seqlens = inputs
    output, final_state = chunk_gdn(
        q.to(precision),
        k.to(precision),
        value.to(precision),
        gate.to(precision),
        beta.to(precision),
        initial_state.to(precision),
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="reference",
    )
    assert final_state is not None
    return output, final_state


def reference_gradients(
    inputs: tuple[torch.Tensor, ...],
    d_output: torch.Tensor,
    precision: torch.dtype | None,
    initial_state: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, ...]:
    """Differentiate public eager GDN at source precision or independently in FP64."""
    q, k, value, gate, beta, _state, cu_seqlens = inputs
    operands = tuple(
        tensor.detach().to(tensor.dtype if precision is None else precision).requires_grad_()
        for tensor in (q, k, value, gate, beta)
    )
    state = (
        None
        if initial_state is None
        else initial_state.detach()
        .to(initial_state.dtype if precision is None else precision)
        .requires_grad_()
    )
    targets = operands if state is None else (*operands, state)
    output, final_state = chunk_gdn(
        *operands,
        state,
        cu_seqlens=cu_seqlens,
        output_final_state=d_final_state is not None,
        scale=scale,
        impl="reference",
    )
    outputs = (output,) if final_state is None else (output, final_state)
    cotangents = (d_output.to(output.dtype if precision is None else precision),)
    if final_state is not None:
        assert d_final_state is not None
        cotangents += (d_final_state.to(final_state.dtype),)
    return torch.autograd.grad(outputs, targets, cotangents)


def make_cotangents(
    value: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    seed: int,
    state_cotangent: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Create deterministic output and optional final-state cotangents."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    d_output = torch.randn(
        value.shape, dtype=value.dtype, device=value.device, generator=generator
    )
    if not state_cotangent:
        return d_output, None
    assert initial_state is not None
    d_final_state = torch.randn(
        initial_state.shape,
        dtype=initial_state.dtype,
        device=initial_state.device,
        generator=generator,
    )
    return d_output, d_final_state


def assert_gradients_match_references(
    inputs: tuple[torch.Tensor, ...],
    d_output: torch.Tensor,
    initial_state: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Compare every private backward result with source-precision and FP64 eager GDN."""
    source_gradients = reference_gradients(inputs, d_output, None, initial_state, d_final_state)
    high_gradients = reference_gradients(
        inputs, d_output, torch.float64, initial_state, d_final_state
    )
    actual_gradients = chunk_gdn_bwd_mega_packed(
        *inputs[:5],
        d_output,
        inputs[6],
        initial_state,
        d_final_state,
    )
    names = ("dq", "dk", "dv", "dgate", "dbeta")
    if initial_state is not None:
        names += ("d_initial_state",)

    for name, actual, high, source in zip(
        names, actual_gradients[: len(names)], high_gradients, source_gradients, strict=True
    ):
        assert actual is not None
        assert_matches_low_precision_reference(
            actual,
            high,
            source,
            name,
            source_dtype=inputs[0].dtype,
        )
    if initial_state is None:
        assert actual_gradients[-1] is None
    return actual_gradients


@pytest.mark.parametrize(
    ("lengths", "dtype", "key_heads", "gate_pattern", "state_mode"), _BACKWARD_CASES
)
def test_gdn_mega_backward_matches_curated_matrix(
    lengths: tuple[int, ...],
    dtype: torch.dtype,
    key_heads: int,
    gate_pattern: GatePattern,
    state_mode: BackwardStateMode,
) -> None:
    """Check grouped/equal heads, BT64 boundaries, gates, and all state-gradient modes."""
    inputs = make_gdn_test_inputs(
        lengths,
        key_heads=key_heads,
        value_heads=2,
        gate_pattern=gate_pattern,
        dtype=dtype,
        seed=sum(lengths) + key_heads,
    )
    initial_state = None if state_mode == "none" else inputs[5]
    d_output, d_final_state = make_cotangents(
        inputs[2],
        initial_state,
        seed=sum(lengths) + 47,
        state_cotangent=state_mode == "cotangent",
    )
    assert_gradients_match_references(inputs, d_output, initial_state, d_final_state)


@pytest.mark.parametrize(
    ("dtype", "key_heads"),
    (
        pytest.param(torch.bfloat16, 1, id="bf16-grouped"),
        pytest.param(torch.float16, 2, id="fp16-equal"),
    ),
)
def test_gdn_mega_backward_uniform_negative_twenty_is_finite(
    dtype: torch.dtype, key_heads: int
) -> None:
    """Ensure the production backward stays finite for uniformly extreme scalar gates."""
    inputs = make_gdn_test_inputs(
        (129,),
        key_heads=key_heads,
        value_heads=2,
        gate_pattern="uniform_negative_twenty",
        dtype=dtype,
        seed=131 + key_heads,
    )
    d_output, _ = make_cotangents(inputs[2], None, seed=137 + key_heads, state_cotangent=False)
    gradients = chunk_gdn_bwd_mega_packed(*inputs[:5], d_output, inputs[6])[:5]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
@pytest.mark.parametrize("beta_value", (0.0, 1e-12, 1e-10, 1e-8))
def test_gdn_mega_backward_preserves_small_beta_gradient(
    dtype: torch.dtype, beta_value: float
) -> None:
    """dBeta must remain defined when the write gate is zero or very small."""
    shape = (1, 2, 1, 128)
    q = torch.zeros(shape, device="cuda", dtype=dtype)
    k = torch.zeros_like(q)
    value = torch.zeros_like(q)
    q[..., 0] = 1
    k[..., 0] = 1
    value[0, 0, 0, 0] = 1
    value[0, 1, 0, 0] = 2  # Makes both the residual and strict-lower dBeta terms nonzero.
    gate = torch.zeros(shape[:-1], device="cuda")
    beta = torch.tensor([[[1.0], [beta_value]]], device="cuda")
    d_output = torch.zeros_like(value)
    d_output[0, 1, 0, 0] = 1
    cu_seqlens = torch.tensor([0, 2], device="cuda", dtype=torch.int32)
    inputs = (q, k, value, gate, beta, None, cu_seqlens)

    gradients = chunk_gdn_bwd_mega_packed(*inputs[:5], d_output, cu_seqlens, scale=1.0)
    expected_gradients = reference_gradients(
        inputs, d_output, torch.float64, None, None, scale=1.0
    )

    for actual, expected in zip(gradients[:5], expected_gradients, strict=True):
        assert torch.equal(actual, expected.to(actual.dtype))
    assert gradients[4][0, 1, 0].item() == 1.0


def test_gdn_mega_backward_mixed_small_beta_matches_reference() -> None:
    """Direct dBeta must survive zero and small beta across persistent chunks."""
    inputs = make_gdn_test_inputs(
        (64, 65, 0),
        key_heads=1,
        value_heads=2,
        dtype=torch.bfloat16,
        seed=151,
    )
    beta = inputs[4]
    beta[:, ::4] = 0.0
    beta[:, 1::4] = 1e-12
    beta[:, 2::4] = 1e-8
    d_output, d_final_state = make_cotangents(inputs[2], inputs[5], seed=157, state_cotangent=True)

    assert_gradients_match_references(inputs, d_output, inputs[5], d_final_state)


def test_gdn_mega_backward_rejects_mismatched_output_gradient_dtype() -> None:
    q, k, value, gate, beta, _state, cu_seqlens = make_gdn_test_inputs(
        (64,), key_heads=1, value_heads=1, dtype=torch.bfloat16, seed=143
    )
    with pytest.raises(TypeError, match="d_output must use q.dtype"):
        chunk_gdn_bwd_mega_packed(
            q,
            k,
            value,
            gate,
            beta,
            torch.randn_like(value, dtype=torch.float32),
            cu_seqlens,
        )


def test_gdn_mega_backward_checkpoint_uses_v_by_k_state_at_token_64() -> None:
    """Regression for loading the BT64 entering-state checkpoint with transposed V/K axes."""
    inputs = make_gdn_test_inputs((65,), key_heads=2, value_heads=2, dtype=torch.float16, seed=149)
    q, k, value, gate, beta, _state, cu_seqlens = inputs
    key_index, value_index = 11, 17
    q.zero_()
    k.zero_()
    value.zero_()
    gate.zero_()
    beta.zero_()
    k[0, 0, 0, key_index] = 1
    value[0, 0, 0, value_index] = 1
    beta[0, 0, 0] = 1
    q[0, 64, 0, key_index] = 1
    d_output = torch.zeros_like(value)
    d_output[0, 64, 0, value_index] = 1

    source_gradients = reference_gradients(inputs, d_output, None, None, None)
    high_gradients = reference_gradients(inputs, d_output, torch.float64, None, None)
    actual_gradients = chunk_gdn_bwd_mega_packed(q, k, value, gate, beta, d_output, cu_seqlens)

    # The checkpoint entering token 64 has S[V=17, K=11] != S[K=11, V=17].
    assert_matches_low_precision_reference(
        actual_gradients[0][0, 64, 0],
        high_gradients[0][0, 64, 0],
        source_gradients[0][0, 64, 0],
        "dQ at the first token after the BT64 checkpoint",
        source_dtype=q.dtype,
    )


def random_packed_lengths(seed: int) -> tuple[int, ...]:
    """Generate a bounded deterministic multi-sequence shape with one empty sequence."""
    random = Random(seed)
    lengths = [random.randrange(1, 130) for _ in range(3)]
    lengths.insert(random.randrange(4), 0)
    return tuple(lengths)


@pytest.mark.parametrize(("seed", "dtype", "key_heads", "gate_pattern"), _FUZZ_CASES)
def test_gdn_mega_packed_differential_fuzz(
    seed: int, dtype: torch.dtype, key_heads: int, gate_pattern: GatePattern
) -> None:
    """Differentially fuzz packed launcher paths against source-precision and FP64 eager GDN."""
    inputs = make_gdn_test_inputs(
        random_packed_lengths(seed),
        key_heads=key_heads,
        value_heads=2,
        gate_pattern=gate_pattern,
        dtype=dtype,
        seed=seed,
    )
    d_output, d_final_state = make_cotangents(
        inputs[2], inputs[5], seed=seed + 1, state_cotangent=True
    )
    source_output, source_state = reference_forward(inputs, torch.float32, inputs[5])
    high_output, high_state = reference_forward(inputs, torch.float64, inputs[5])
    actual_output, actual_state = run_forward(
        *inputs[:5], inputs[6], inputs[5], scale=None, output_final_state=True
    )
    assert actual_state is not None
    assert_matches_low_precision_reference(
        actual_output,
        high_output,
        source_output,
        "fuzz output",
        source_dtype=dtype,
    )
    assert_matches_low_precision_reference(
        actual_state,
        high_state,
        source_state,
        "fuzz final state",
        source_dtype=dtype,
    )
    assert_gradients_match_references(inputs, d_output, inputs[5], d_final_state)
