"""Native SM90 KDA forward, backward, and compiler integration tests."""

from __future__ import annotations

import importlib
from functools import partial

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear import Impl, chunk_kda, paged_chunk_kda
from attn_gym.linear._delta_rule.reference import packed_delta_rule_reference
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_delta_h_triton import (
    chunk_kda_bwd_delta_h_triton,
)
from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    _chunk_kda_bwd_with_state_grad_op,
    _chunk_kda_fwd_with_state_op,
)
from attn_gym.linear.kda.naive import chunk_cumsum_ref, naive_chunk_kda
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    make_kda_test_inputs,
    strided_state_pool,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9,
    reason="the native Hopper KDA route requires CUDA capability 9.x",
)


def _high_precision_reference(
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate the public natural-log gate contract without its FP32 cast."""
    q, k, v, gate, beta, initial_state = inputs
    dense_op = partial(naive_chunk_kda, chunk_size=64)
    if cu_seqlens is None:
        return dense_op(
            q,
            k,
            v,
            gate * LOG2_E,
            beta,
            initial_state=initial_state,
            scale=128**-0.5,
            output_final_state=True,
        )
    return packed_delta_rule_reference(
        dense_op,
        q,
        k,
        v,
        gate * LOG2_E,
        beta,
        initial_state,
        cu_seqlens,
        True,
        scale=128**-0.5,
    )


def _training_inputs(
    dtype: torch.dtype,
    lengths: list[int] | None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    """Create one dense or packed stateful training case from quantized inputs."""
    tokens = 128 if lengths is None else sum(lengths)
    inputs = make_kda_test_inputs(
        tokens,
        batch=1,
        heads=2,
        dtype=dtype,
        normalize_qk=True,
        gate_scale=0.5,
        requires_grad=True,
    )
    sequences = 1 if lengths is None else len(lengths)
    initial_state = (
        torch.randn(sequences, 2, 128, 128, device="cuda", dtype=torch.float32) / 32
    ).requires_grad_()
    cu_seqlens = None if lengths is None else cumulative_sequence_offsets(lengths)
    return (*inputs, initial_state), cu_seqlens


def _assert_training_matches_reference(
    actual_inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | None,
    dtype: torch.dtype,
    *,
    fastmath: bool = False,
) -> None:
    """Bound one fused training case by its low-precision eager error."""
    reference_inputs = clone_kda_inputs(actual_inputs)
    high_inputs = clone_kda_inputs(actual_inputs, dtype=torch.float64)
    actual_output, actual_state = chunk_kda(
        *actual_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        fastmath=fastmath,
        autotune=False,
    )
    reference_output, reference_state = chunk_kda(
        *reference_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl=Impl.REFERENCE,
    )
    high_output, high_state = _high_precision_reference(high_inputs, cu_seqlens)
    assert actual_state is not None and reference_state is not None and high_state is not None
    assert_matches_low_precision_reference(
        actual_output,
        high_output,
        reference_output,
        "Hopper chunk KDA output",
        source_dtype=dtype,
    )
    assert_matches_low_precision_reference(
        actual_state,
        high_state,
        reference_state,
        "Hopper chunk KDA final state",
        source_dtype=dtype,
    )

    d_output = torch.randn_like(actual_output)
    d_state = torch.randn_like(actual_state)
    actual_gradients = torch.autograd.grad(
        (actual_output, actual_state), actual_inputs, (d_output, d_state)
    )
    reference_gradients = torch.autograd.grad(
        (reference_output, reference_state), reference_inputs, (d_output, d_state)
    )
    high_gradients = torch.autograd.grad(
        (high_output, high_state), high_inputs, (d_output.double(), d_state.double())
    )
    for name, actual, high, reference in zip(
        ("q", "k", "v", "gate", "beta", "initial_state"),
        actual_gradients,
        high_gradients,
        reference_gradients,
        strict=True,
    ):
        assert_matches_low_precision_reference(
            actual,
            high,
            reference,
            f"Hopper chunk KDA gradient {name}",
            source_dtype=dtype,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lengths", [None, [65, 0, 63]], ids=["dense", "packed"])
def test_hopper_training_matches_low_precision_reference(
    dtype: torch.dtype,
    lengths: list[int] | None,
):
    """Cover ordinary dense and packed Hopper training."""
    inputs, cu_seqlens = _training_inputs(dtype, lengths)
    _assert_training_matches_reference(inputs, cu_seqlens, dtype)


def test_hopper_fastmath_backward_matches_reference(monkeypatch):
    """Exercise and verify fastmath plumbing in the native SM90 WY backward."""
    import attn_gym.linear.kda.bwd.cute.chunk_kda_bwd as backward_module

    original = backward_module.chunk_kda_bwd_wy_triton
    received_fastmath: list[bool] = []

    def capture_fastmath(*args, **kwargs):
        received_fastmath.append(kwargs["fastmath"])
        return original(*args, **kwargs)

    monkeypatch.setattr(backward_module, "chunk_kda_bwd_wy_triton", capture_fastmath)
    inputs, cu_seqlens = _training_inputs(torch.bfloat16, [65, 63])
    _assert_training_matches_reference(inputs, cu_seqlens, torch.bfloat16, fastmath=True)
    assert received_fastmath == [True]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("lengths", [None, [65, 63]], ids=["dense", "packed"])
def test_hopper_bc16_rebase_matches_reference_at_lower_bound_five(
    dtype: torch.dtype,
    lengths: list[int] | None,
):
    """Pin the BC16 rebase numerics at the model's default gate lower bound."""
    inputs, cu_seqlens = _training_inputs(dtype, lengths)
    gate = torch.full_like(inputs[3], -5.0, requires_grad=True)
    actual_inputs = (*inputs[:3], gate, *inputs[4:])
    reference_inputs = clone_kda_inputs(actual_inputs)
    actual_output, actual_state = chunk_kda(
        *actual_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        autotune=False,
    )
    reference_output, reference_state = chunk_kda(
        *reference_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl=Impl.REFERENCE,
    )
    assert actual_state is not None and reference_state is not None
    atol = 6e-3 if dtype is torch.bfloat16 else 5e-4
    torch.testing.assert_close(actual_output, reference_output, rtol=3e-2, atol=atol)
    torch.testing.assert_close(actual_state, reference_state, rtol=3e-2, atol=atol)

    d_output = torch.randn_like(actual_output)
    d_state = torch.randn_like(actual_state)
    actual_gradients = torch.autograd.grad(
        (actual_output, actual_state), actual_inputs, (d_output, d_state)
    )
    reference_gradients = torch.autograd.grad(
        (reference_output, reference_state), reference_inputs, (d_output, d_state)
    )
    for actual, reference in zip(actual_gradients, reference_gradients, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, reference, rtol=3e-2, atol=atol)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hopper_delta_h_masks_upper_aqk(dtype: torch.dtype):
    """Ignore poisoned upper-triangular Aqk entries in the reverse recurrence."""
    torch.manual_seed(97)
    shape = (1, 64, 1, 128)
    qg = torch.randn(shape, device="cuda", dtype=dtype) / 8
    kg = torch.randn_like(qg) / 8
    w = torch.randn_like(qg) / 8
    d_output = torch.randn_like(qg) / 8
    gate = -torch.rand(shape, device="cuda")
    aqk = torch.randn(1, 64, 1, 64, device="cuda", dtype=dtype) / 8
    upper = torch.ones(64, 64, dtype=torch.bool, device="cuda").triu(1)
    aqk[0, :, 0].masked_fill_(upper, 0)
    poisoned = aqk.clone()
    poisoned[0, :, 0].masked_fill_(upper, float("nan"))

    def run(factors: torch.Tensor):
        return chunk_kda_bwd_delta_h_triton(
            qg,
            kg,
            w,
            d_output,
            factors,
            gk=gate,
            initial_state=None,
            d_final_state=None,
            scale=128**-0.5,
            metadata=None,
        )

    expected = run(aqk)
    actual = run(poisoned)
    for poisoned_output, neutral_output in zip(actual, expected, strict=True):
        if poisoned_output is None:
            assert neutral_output is None
        else:
            assert neutral_output is not None
            torch.testing.assert_close(poisoned_output, neutral_output, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hopper_final_state_only_gradient(dtype: torch.dtype):
    """Exercise a final-state cotangent when the output cotangent is absent."""
    actual_inputs, cu_seqlens = _training_inputs(dtype, [65, 0, 63])
    reference_inputs = clone_kda_inputs(actual_inputs)
    high_inputs = clone_kda_inputs(actual_inputs, dtype=torch.float64)
    _, actual_state = chunk_kda(
        *actual_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        autotune=False,
    )
    _, reference_state = chunk_kda(
        *reference_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl=Impl.REFERENCE,
    )
    _, high_state = _high_precision_reference(high_inputs, cu_seqlens)
    assert actual_state is not None and reference_state is not None and high_state is not None
    d_state = torch.randn_like(actual_state)
    actual_gradients = torch.autograd.grad(actual_state, actual_inputs[1:], d_state)
    reference_gradients = torch.autograd.grad(reference_state, reference_inputs[1:], d_state)
    high_gradients = torch.autograd.grad(high_state, high_inputs[1:], d_state.double())
    for name, actual, high, reference in zip(
        ("k", "v", "gate", "beta", "initial_state"),
        actual_gradients,
        high_gradients,
        reference_gradients,
        strict=True,
    ):
        assert_matches_low_precision_reference(
            actual,
            high,
            reference,
            f"Hopper final-state-only gradient {name}",
            source_dtype=dtype,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hopper_fullgraph_forward_backward(dtype: torch.dtype):
    """Keep the public Hopper route inside one strict compiled training graph."""
    eager_inputs, cu_seqlens = _training_inputs(dtype, [65, 63])
    compiled_inputs = clone_kda_inputs(eager_inputs)

    def operation(*args):
        return chunk_kda(
            *args,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            autotune=False,
        )

    expected_output, expected_state = operation(*eager_inputs)
    actual_output, actual_state = torch.compile(operation, fullgraph=True)(*compiled_inputs)
    assert expected_state is not None and actual_state is not None
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)

    d_output = torch.randn_like(expected_output)
    d_state = torch.randn_like(expected_state)
    expected_gradients = torch.autograd.grad(
        (expected_output, expected_state), eager_inputs, (d_output, d_state)
    )
    actual_gradients = torch.autograd.grad(
        (actual_output, actual_state), compiled_inputs, (d_output, d_state)
    )
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hopper_raw_operator_registration():
    """Validate the unchanged opaque schemas against the real SM90 launchers."""
    inputs, _cu_seqlens = _training_inputs(torch.bfloat16, None)
    q, k, v, gate, beta, initial_state = inputs
    cumulative_gate = chunk_cumsum_ref(gate * LOG2_E, 64)
    forward_args = (
        q.detach(),
        k.detach(),
        v.detach(),
        cumulative_gate.detach(),
        beta.detach(),
        initial_state.detach(),
        128**-0.5,
        False,
    )
    torch.library.opcheck(
        _chunk_kda_fwd_with_state_op,
        forward_args,
        rtol=2e-2,
        atol=2e-3,
    )
    with torch.no_grad():
        _output, state, aqk, akk = _chunk_kda_fwd_with_state_op(*forward_args)
    torch.library.opcheck(
        _chunk_kda_bwd_with_state_grad_op,
        (
            q.detach(),
            k.detach(),
            v.detach(),
            cumulative_gate.detach(),
            beta.detach(),
            aqk,
            akk,
            None,
            None,
            None,
            torch.randn_like(state),
            initial_state.detach(),
            128**-0.5,
            False,
            False,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )


def test_hopper_outer_strided_cotangents():
    """Normalize supported output/state cotangent layouts at the SM90 boundary."""
    actual_inputs, cu_seqlens = _training_inputs(torch.bfloat16, [65, 63])
    expected_inputs = clone_kda_inputs(actual_inputs)
    actual_output, actual_state = chunk_kda(
        *actual_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        autotune=False,
    )
    expected_output, expected_state = chunk_kda(
        *expected_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        autotune=False,
    )
    assert actual_state is not None and expected_state is not None

    d_output = torch.randn(
        actual_output.shape[0],
        actual_output.shape[2],
        actual_output.shape[1],
        actual_output.shape[3],
        device="cuda",
        dtype=actual_output.dtype,
    ).permute(0, 2, 1, 3)
    state_storage, d_state = strided_state_pool(2, 2, 128, 128, prefix=0)
    assert not d_output.is_contiguous() and not d_state.is_contiguous()
    actual_gradients = torch.autograd.grad(
        (actual_output, actual_state), actual_inputs, (d_output, d_state)
    )
    expected_gradients = torch.autograd.grad(
        (expected_output, expected_state),
        expected_inputs,
        (d_output.contiguous(), d_state.contiguous()),
    )
    del state_storage
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hopper_cuda_graph_replays_packed_boundaries_and_backward():
    """Replay changed packed boundaries through the complete SM90 training route."""
    inputs, _ = _training_inputs(torch.bfloat16, [64, 64])
    cu_seqlens = cumulative_sequence_offsets([64, 64])
    warm_output, warm_state = chunk_kda(
        *inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        autotune=False,
    )
    assert warm_state is not None
    torch.autograd.grad(warm_output.float().sum() + warm_state.sum(), inputs)
    torch.cuda.synchronize()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, state = chunk_kda(
                *inputs,
                cu_seqlens=cu_seqlens,
                output_final_state=True,
                autotune=False,
            )
            assert state is not None
            gradients = torch.autograd.grad(output.float().sum() + state.sum(), inputs)
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(False)

    with torch.no_grad():
        inputs[2].mul_(0.5)
        cu_seqlens.copy_(cumulative_sequence_offsets([65, 63]))
    expected_inputs = clone_kda_inputs(inputs)
    expected_output, expected_state = chunk_kda(
        *expected_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        autotune=False,
    )
    assert expected_state is not None
    expected_gradients = torch.autograd.grad(
        expected_output.float().sum() + expected_state.sum(), expected_inputs
    )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    for actual, expected in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hopper_int64_specializations_match_default(monkeypatch):
    """Force every new wide-address path and preserve public outputs and gradients."""
    packed_offsets = cumulative_sequence_offsets([65, 63])

    def run(cu_seqlens: torch.Tensor | None):
        inputs = make_kda_test_inputs(
            128,
            batch=1,
            heads=1,
            dtype=torch.bfloat16,
            normalize_qk=True,
            requires_grad=True,
        )
        output, _ = chunk_kda(
            *inputs,
            cu_seqlens=cu_seqlens,
            autotune=False,
        )
        gradients = torch.autograd.grad(output.float().square().sum(), inputs)
        return output, *gradients

    expected = (run(None), run(packed_offsets))
    for module_name, attribute in (
        (
            "attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop",
            "requires_int64_offsets",
        ),
        ("attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_k3_triton", "requires_int64_offsets"),
        ("attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_k4_triton", "requires_int64_offsets"),
        ("attn_gym.linear.kda.fwd.triton.recompute_w_u", "requires_int64_offsets"),
        ("attn_gym.linear.kda.fwd.triton.chunk_delta_h", "requires_int64_offsets"),
        ("attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o", "requires_int64_offsets"),
        ("attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk", "requires_int64_offsets"),
        (
            "attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_delta_h_triton",
            "requires_int64_offsets",
        ),
        (
            "attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_wy_triton",
            "requires_int64_offsets",
        ),
        ("attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra", "requires_int64_abi"),
    ):
        monkeypatch.setattr(importlib.import_module(module_name), attribute, lambda *_args: True)

    actual = (run(None), run(packed_offsets))
    for wide_case, normal_case in zip(actual, expected, strict=True):
        for wide, normal in zip(wide_case, normal_case, strict=True):
            torch.testing.assert_close(wide, normal, rtol=0, atol=0)


def test_paged_chunk_kda_uses_pre_blackwell_route():
    """Advance paged state through the same portable factors as ordinary prefill."""
    inputs = tuple(
        tensor.detach() for tensor in make_kda_test_inputs(64, batch=1, heads=1, normalize_qk=True)
    )
    state_cache = torch.randn(2, 1, 128, 128, device="cuda")
    expected_cache = state_cache.clone()
    state_indices = torch.ones(1, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        expected_output, expected_state = chunk_kda(
            *inputs,
            expected_cache[state_indices.long()].clone(),
            output_final_state=True,
            autotune=False,
        )
        output = paged_chunk_kda(*inputs, state_cache, state_indices, autotune=False)
    assert expected_state is not None
    expected_cache[state_indices.long()] = expected_state
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state_cache, expected_cache, rtol=0, atol=0)
