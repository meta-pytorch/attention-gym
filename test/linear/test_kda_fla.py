"""The ``backend="fla"`` kernel option agrees with the reference implementation.

The adapter's job is to hand flash-linear-attention exactly the operands this
package's contract defines -- q/k pre-normalized, ``gate`` the bound natural-log
decay, ``beta`` already sigmoided -- and to reconcile the one convention the two
libraries do not share: the recurrent state's last two axes are transposed
relative to each other. Every check below fails if either half is dropped.
"""

import pytest
import torch

from attn_gym.linear import Impl, chunk_kda
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    make_kda_test_inputs,
)

fla = pytest.importorskip("fla", reason="backend='fla' needs flash-linear-attention")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fla kernels run on CUDA"
)

TOKENS = 128
HEADS = 4


FLA = {"backend": "fla"}


def _run(inputs, *, backend=None, impl=Impl.FUSED, **kwargs):
    query, key, value, gate, beta = inputs
    options = {"backend": backend} if backend else None
    return chunk_kda(query, key, value, gate, beta, impl=impl, kernel_options=options, **kwargs)


def test_forward_matches_reference():
    inputs = make_kda_test_inputs(TOKENS, batch=2, heads=HEADS)
    fla_output, _ = _run(inputs, backend="fla")
    reference_output, _ = _run(inputs, impl=Impl.REFERENCE)
    high_precision, _ = _run(tuple(value.float() for value in inputs), impl=Impl.REFERENCE)
    assert_matches_low_precision_reference(
        fla_output, high_precision, reference_output, "backend=fla forward"
    )


def test_final_state_matches_reference_after_the_axis_flip():
    """The state axes, not just its values.

    Without the flip in the adapter the two states still have the same shape,
    so only comparing them elementwise catches it.
    """
    inputs = make_kda_test_inputs(TOKENS, batch=2, heads=HEADS)
    _, fla_state = _run(inputs, backend="fla", output_final_state=True)
    _, reference_state = _run(inputs, impl=Impl.REFERENCE, output_final_state=True)
    assert fla_state.shape == reference_state.shape
    torch.testing.assert_close(fla_state, reference_state, rtol=2e-2, atol=2e-2)
    flipped_error = (fla_state - reference_state.transpose(-1, -2)).abs().max()
    aligned_error = (fla_state - reference_state).abs().max()
    assert aligned_error < flipped_error, (
        "the state agrees with the transposed reference as closely as with the "
        "reference itself, so this test cannot see the axis flip"
    )


def test_state_handoff_between_impls():
    """A state one impl produced resumes the other, which is what CP composition needs."""
    inputs = make_kda_test_inputs(2 * TOKENS, batch=1, heads=HEADS)
    first = tuple(value[:, :TOKENS] for value in inputs)
    second = tuple(value[:, TOKENS:] for value in inputs)

    _, reference_state = _run(first, impl=Impl.REFERENCE, output_final_state=True)
    resumed, _ = _run(second, backend="fla", initial_state=reference_state)
    whole, _ = _run(inputs, impl=Impl.REFERENCE)

    high_precision, _ = _run(tuple(value.float() for value in inputs), impl=Impl.REFERENCE)
    assert_matches_low_precision_reference(
        resumed,
        high_precision[:, TOKENS:],
        whole[:, TOKENS:],
        "backend=fla resumed from a reference state",
    )


def test_gradients_match_reference():
    inputs = make_kda_test_inputs(TOKENS, batch=1, heads=HEADS, requires_grad=True)
    reference_inputs = clone_kda_inputs(inputs)

    fla_output, _ = _run(inputs, backend="fla")
    reference_output, _ = _run(reference_inputs, impl=Impl.REFERENCE)
    fla_output.sum().backward()
    reference_output.sum().backward()

    for name, actual, expected in zip(("q", "k", "v", "gate", "beta"), inputs, reference_inputs):
        assert actual.grad is not None, f"{name}: backend='fla' produced no gradient"
        torch.testing.assert_close(
            actual.grad, expected.grad, rtol=2e-2, atol=2e-2, msg=f"grad {name}"
        )


def test_varlen_matches_reference():
    """Packed documents through ``cu_seqlens``: the boundaries must reset the recurrence."""
    lengths = [48, 80]
    inputs = make_kda_test_inputs(sum(lengths), batch=1, heads=HEADS)
    cu_seqlens = cumulative_sequence_offsets(lengths).to(inputs[0].device)

    fla_output, _ = _run(inputs, backend="fla", cu_seqlens=cu_seqlens)
    reference_output, _ = _run(inputs, impl=Impl.REFERENCE, cu_seqlens=cu_seqlens)
    high_precision, _ = _run(
        tuple(value.float() for value in inputs),
        impl=Impl.REFERENCE,
        cu_seqlens=cu_seqlens,
    )
    assert_matches_low_precision_reference(
        fla_output, high_precision, reference_output, "backend=fla varlen forward"
    )


def test_backend_is_rejected_with_the_reference_impl():
    """Kernel options select a kernel, so they mean nothing without kernels."""
    inputs = make_kda_test_inputs(32, batch=1, heads=1)
    with pytest.raises(ValueError, match="kernel_options"):
        _run(inputs, backend="fla", impl=Impl.REFERENCE)
