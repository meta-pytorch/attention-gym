"""Composition tests for the optimized CuTeDSL KDA forward pipeline."""

from __future__ import annotations

import functools
import gc
import importlib
import math

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear.kda.constants import LOG2_E, MAX_GATE_LOWER_BOUND_MAGNITUDE
from attn_gym.linear.kda.naive import chunk_cumsum_ref, naive_chunk_kda

pytest.importorskip("cutlass")

from attn_gym.linear import chunk_kda, paged_chunk_kda
from attn_gym.linear.kda.bwd.cute import chunk_kda_bwd as _chunk_kda_bwd_module
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import (
    ChunkKdaBwdIntraConfig,
    chunk_kda_bwd_intra,
)
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
    ChunkKdaBwdWyDqkgConfig,
    chunk_kda_bwd_wy_dqkg,
)
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    _chunk_kda_bwd_op,
    _chunk_kda_bwd_with_state_grad_op,
    _chunk_kda_fwd_op,
    _chunk_kda_fwd_ragged_paged_op,
    _chunk_kda_fwd_with_state_op,
)
from attn_gym.testing import strided_state_pool

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL KDA forward pipeline requires SM100/SM103",
)

_DEFAULT_SCALE = 128**-0.5


def _inputs(
    *,
    batch: int = 1,
    tokens: int = 128,
    heads: int = 1,
    initial_state: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    head_dim = 128
    shape = (batch, tokens, heads, head_dim)
    q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    gate = -torch.rand(shape, device="cuda") * math.log(2.0)
    beta = torch.rand(batch, tokens, heads, device="cuda")
    tensors = [q, k, v, gate, beta]
    if initial_state:
        tensors.append(torch.randn(batch, heads, head_dim, head_dim, device="cuda") * 0.01)
    return tuple(tensor.requires_grad_() for tensor in tensors)


def _clone_inputs(inputs):
    return tuple(tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in inputs)


def _packed_qkv_tensor(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Place Q, K, and V in one packed projection tensor."""
    batch, tokens, heads, head_dim = q.shape
    qkv = q.new_empty(batch, tokens, 3, heads, head_dim)
    for index, tensor in enumerate((q, k, v)):
        qkv[:, :, index].copy_(tensor.detach())
    qkv.requires_grad_(q.requires_grad or k.requires_grad or v.requires_grad)
    return qkv


def _strided_qkv_views(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sibling Q/K/V views of one packed projection."""
    return _packed_qkv_tensor(q, k, v).unbind(2)


def _unaligned_view(tensor: torch.Tensor) -> torch.Tensor:
    """Preserve compact logical strides while shifting the base by one element."""
    storage = tensor.new_empty(tensor.numel() + 1)
    view = storage[1:].view(tensor.shape)
    view.copy_(tensor.detach())
    storage.requires_grad_(tensor.requires_grad)
    return view


def _head_major_view(tensor: torch.Tensor) -> torch.Tensor:
    """Create a dense head-major view whose generated outputs must not inherit its strides."""
    storage = tensor.detach().permute(0, 2, 1, 3).contiguous()
    storage.requires_grad_(tensor.requires_grad)
    return storage.permute(0, 2, 1, 3)


def _assert_golden(
    actual: torch.Tensor,
    golden: torch.Tensor,
    reference: torch.Tensor,
    dtype: torch.dtype,
    name: str,
) -> None:
    """Bound kernel error by the reference error and operand precision."""
    golden64 = golden.to(torch.float64)
    band = torch.finfo(dtype).eps * golden64.abs().max().item()
    actual_error = (actual.to(torch.float64) - golden64).abs().max().item()
    reference_error = (reference.to(torch.float64) - golden64).abs().max().item()
    budget = 2.0 * reference_error + 2.0 * band
    assert actual_error <= budget, (
        f"{name}: kernel error {actual_error:.3e} exceeds budget {budget:.3e} "
        f"(reference error {reference_error:.3e}, band {band:.3e}, dtype {dtype})"
    )


def _packed_reference(
    inputs: tuple[torch.Tensor, ...],
    spans: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate packed sequences independently with the naive implementation."""
    q, k, v, gate, beta, initial_state = inputs
    outputs = []
    states = []
    for sequence, (start, end) in enumerate(spans):
        output, state = naive_chunk_kda(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            gate[:, start:end] * LOG2_E,
            beta[:, start:end],
            initial_state=initial_state[sequence : sequence + 1],
            output_final_state=True,
            chunk_size=64,
        )
        outputs.append(output)
        assert state is not None
        states.append(state)
    return torch.cat(outputs, dim=1), torch.cat(states)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_private_chunk_kda_forward_matches_reference(dtype: torch.dtype):
    """Exercise native 16-bit forward factors without the public dtype normalization."""
    torch.manual_seed(2)
    q, k, v, gate, beta = _inputs(tokens=64, dtype=dtype)
    cumulative_gate = chunk_cumsum_ref(gate * LOG2_E, 64)
    actual, aqk, akk = _chunk_kda_fwd_op(
        q, k, v, cumulative_gate, beta, None, _DEFAULT_SCALE, False
    )
    golden, _ = naive_chunk_kda(
        q.double(),
        k.double(),
        v.double(),
        (gate * LOG2_E).double(),
        beta.double(),
        chunk_size=64,
    )
    reference, _ = naive_chunk_kda(q, k, v, gate * LOG2_E, beta, chunk_size=64)

    assert actual.dtype == aqk.dtype == akk.dtype == dtype
    _assert_golden(actual, golden, reference, dtype, f"private forward {dtype}")


def test_private_fp16_forward_factors_are_finite_at_gate_limit():
    """Keep the diagonal gate factorization within FP16 range at the public bound."""
    torch.manual_seed(37)
    q, k, v, _gate, beta = _inputs(tokens=64, dtype=torch.float16)
    gate = torch.full_like(q, -MAX_GATE_LOWER_BOUND_MAGNITUDE, dtype=torch.float32)
    cumulative_gate = chunk_cumsum_ref(gate * LOG2_E, 64)

    output, aqk, akk = _chunk_kda_fwd_op(
        q, k, v, cumulative_gate, beta, None, _DEFAULT_SCALE, False
    )

    for tensor in (output, aqk, akk):
        assert tensor.dtype == torch.float16
        assert torch.isfinite(tensor).all()


def test_optimized_chunk_kda_matches_reference():
    """Check forward values and the isolated final-state cotangent path."""
    torch.manual_seed(3)
    inputs = _inputs(initial_state=True)
    q, k, v, gate, beta, initial_state = inputs

    output, state = chunk_kda(*inputs, output_final_state=True)
    assert state is not None
    assert output.dtype == torch.bfloat16
    assert state.dtype == torch.float32
    expected, expected_state = naive_chunk_kda(
        q.float(),
        k.float(),
        v.float(),
        gate * LOG2_E,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
    )
    assert expected_state is not None
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(state, expected_state, rtol=2e-2, atol=2e-3)

    d_state = torch.randn_like(state)
    state_inputs = (k, v, gate, beta, initial_state)
    actual_gradients = torch.autograd.grad(state, state_inputs, d_state)
    expected_gradients = torch.autograd.grad(expected_state, state_inputs, d_state)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        error = (actual_gradient.float() - expected_gradient).abs().max()
        tolerance = 5e-3 + 5e-3 * expected_gradient.abs().max()
        assert error <= tolerance


@pytest.mark.parametrize(("tokens", "packed"), [(128, False), (65, True)])
def test_optimized_chunk_kda_fp16_training_matches_reference(tokens, packed):
    """Exercise native FP16 dense and packed training with state gradients."""
    torch.manual_seed(101)
    inputs = _inputs(tokens=tokens, initial_state=True, dtype=torch.float16)
    actual_inputs = _clone_inputs(inputs)
    expected_inputs = _clone_inputs(inputs)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32) if packed else None

    actual, actual_state = chunk_kda(
        *actual_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        autotune=False,
    )
    expected, expected_state = chunk_kda(
        *expected_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="reference",
    )
    assert actual_state is not None and expected_state is not None
    actual_loss = actual.float().square().mean() + actual_state.square().mean()
    expected_loss = expected.float().square().mean() + expected_state.square().mean()
    actual_gradients = torch.autograd.grad(actual_loss, actual_inputs)
    expected_gradients = torch.autograd.grad(expected_loss, expected_inputs)

    assert actual.dtype == torch.float16
    assert actual_state.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=1e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=3e-2, atol=1e-3)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        assert torch.isfinite(actual_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=1e-3)


def test_optimized_chunk_kda_accepts_strided_packed_qkv_views():
    """Read Q/K/V directly from one packed projection in forward and backward."""
    torch.manual_seed(29)
    q, k, v, gate, beta = _inputs(tokens=128, heads=2)
    q, k, v = _strided_qkv_views(q, k, v)
    expected_stride = (128 * 3 * 2 * 128, 3 * 2 * 128, 128, 1)
    assert all(tensor.stride() == expected_stride for tensor in (q, k, v))
    assert not any(tensor.is_contiguous() for tensor in (q, k, v))
    actual_inputs = (q, k, v, gate, beta)
    expected_inputs = _clone_inputs(actual_inputs)
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    actual_output, actual_state = chunk_kda(
        *actual_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    expected_output, expected_state = chunk_kda(
        *expected_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    assert actual_state is not None and expected_state is not None
    torch.testing.assert_close(actual_output, expected_output)
    torch.testing.assert_close(actual_state, expected_state)

    d_output = torch.randn_like(actual_output)
    d_state = torch.randn_like(actual_state)
    actual_gradients = torch.autograd.grad(
        (actual_output, actual_state), actual_inputs, (d_output, d_state)
    )
    expected_gradients = torch.autograd.grad(
        (expected_output, expected_state), expected_inputs, (d_output, d_state)
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_optimized_chunk_kda_accumulates_into_packed_qkv_backing():
    """Map compact kernel gradients back through the packed QKV views."""
    torch.manual_seed(41)
    q, k, v, gate, beta = _inputs(tokens=128, heads=2)
    actual_qkv = _packed_qkv_tensor(q, k, v)
    expected_qkv = actual_qkv.detach().clone().requires_grad_()
    actual_q, actual_k, actual_v = actual_qkv.unbind(2)
    expected_q, expected_k, expected_v = (tensor.contiguous() for tensor in expected_qkv.unbind(2))
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    actual, _ = chunk_kda(
        actual_q,
        actual_k,
        actual_v,
        gate,
        beta,
        cu_seqlens=cu_seqlens,
    )
    expected, _ = chunk_kda(
        expected_q,
        expected_k,
        expected_v,
        gate,
        beta,
        cu_seqlens=cu_seqlens,
    )
    torch.testing.assert_close(actual, expected)
    d_output = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, actual_qkv, d_output)[0]
    expected_gradient = torch.autograd.grad(expected, expected_qkv, d_output)[0]
    assert actual_gradient.is_contiguous()
    torch.testing.assert_close(actual_gradient, expected_gradient)


def test_optimized_chunk_kda_compacts_unsupported_head_major_qkv_views():
    """Keep compact internal output ABIs when inputs use a head-major layout."""
    torch.manual_seed(37)
    expected_inputs = _inputs(tokens=128, heads=2)
    actual_inputs = list(_clone_inputs(expected_inputs))
    actual_inputs[:3] = tuple(_head_major_view(tensor) for tensor in actual_inputs[:3])
    actual_inputs = tuple(actual_inputs)
    assert all(tensor.stride(-1) == 1 for tensor in actual_inputs[:3])
    assert all(tensor.stride(-2) != tensor.shape[-1] for tensor in actual_inputs[:3])

    expected, _ = chunk_kda(*expected_inputs)
    actual, _ = chunk_kda(*actual_inputs)
    torch.testing.assert_close(actual, expected)
    d_output = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, d_output)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, d_output)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_optimized_chunk_kda_packed_matches_independent_sequences():
    """Keep packed sequence boundaries first-class through forward and backward."""
    torch.manual_seed(13)
    inputs = _inputs(tokens=192, heads=2, initial_state=True)
    q, k, v, gate, beta, _initial_state = inputs
    initial_state = torch.randn(2, 2, 128, 128, device="cuda", requires_grad=True) * 0.01
    inputs = (q, k, v, gate, beta, initial_state)
    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)

    output, state = chunk_kda(
        *inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    assert state is not None and state.shape == (2, 2, 128, 128)

    spans = ((0, 64), (64, 192))
    reference_inputs = (q.float(), k.float(), v.float(), gate, beta, initial_state)
    expected_output, expected_state = _packed_reference(reference_inputs, spans)
    golden_inputs = tuple(tensor.detach().double().requires_grad_() for tensor in inputs)
    golden_output, golden_state = _packed_reference(golden_inputs, spans)

    _assert_golden(output, golden_output, expected_output, torch.bfloat16, "packed output")
    _assert_golden(state, golden_state, expected_state, torch.bfloat16, "packed state")
    d_output = torch.randn_like(output)
    d_state = torch.randn_like(state)
    actual_gradients = torch.autograd.grad((output, state), inputs, (d_output, d_state))
    expected_gradients = torch.autograd.grad(
        (expected_output, expected_state),
        inputs,
        (d_output.float(), d_state),
    )
    golden_gradients = torch.autograd.grad(
        (golden_output, golden_state),
        golden_inputs,
        (d_output.double(), d_state.double()),
    )
    for index, (actual_gradient, golden_gradient, expected_gradient) in enumerate(
        zip(actual_gradients, golden_gradients, expected_gradients, strict=True)
    ):
        _assert_golden(
            actual_gradient,
            golden_gradient,
            expected_gradient,
            torch.bfloat16,
            f"packed input gradient {index}",
        )


def test_paged_chunk_kda_updates_strided_pool():
    """Read and advance routed cache slots without touching page padding."""
    torch.manual_seed(29)
    inputs = tuple(tensor.detach() for tensor in _inputs(tokens=192, heads=2))
    q, k, v, cumulative_gate, beta = inputs
    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)
    slots = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    storage, pool = strided_state_pool(6, q.shape[2], 128, 128, prefix=0)
    expected_storage = storage.clone()
    state_elements = pool[0].numel()
    expected_pool = expected_storage[:, :state_elements].view_as(pool)

    expected_output, expected_state = chunk_kda(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        expected_pool[slots.long()],
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    assert expected_state is not None
    expected_pool[slots.long()] = expected_state

    output = paged_chunk_kda(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        pool,
        slots,
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


def test_paged_chunk_kda_matches_pytorch():
    torch.manual_seed(30)
    q, k, v, cumulative_gate, beta = (tensor.detach() for tensor in _inputs(batch=2, tokens=64))
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state_cache = torch.randn(4, 1, 128, 128, device="cuda")
    expected_cache = state_cache.clone()
    initial_state = expected_cache[state_indices.long()].clone()

    expected_output, expected_state = naive_chunk_kda(
        q.float(),
        k.float(),
        v.float(),
        cumulative_gate.float() * LOG2_E,
        beta,
        initial_state=initial_state,
        output_final_state=True,
    )
    output = paged_chunk_kda(q, k, v, cumulative_gate, beta, state_cache, state_indices)
    assert expected_state is not None
    expected_cache[state_indices.long()] = expected_state

    torch.testing.assert_close(output.float(), expected_output, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(state_cache, expected_cache, rtol=2e-2, atol=2e-3)


def test_paged_chunk_kda_ignores_padding_slots():
    """Zero padding outputs and leave non-positive cache routes untouched."""
    torch.manual_seed(31)
    inputs = tuple(tensor.detach() for tensor in _inputs(tokens=128, heads=2))
    q, k, v, cumulative_gate, beta = inputs
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    slots = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    storage, pool = strided_state_pool(5, q.shape[2], 128, 128, prefix=0)
    before = storage.clone()

    output = paged_chunk_kda(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        pool,
        slots,
        cu_seqlens=cu_seqlens,
    )

    expected, expected_state = chunk_kda(
        q[:, :64],
        k[:, :64],
        v[:, :64],
        cumulative_gate[:, :64],
        beta[:, :64],
        before[3:4, : pool[0].numel()].view(1, q.shape[2], 128, 128),
        output_final_state=True,
    )
    assert expected_state is not None
    torch.testing.assert_close(output[:, :64], expected, rtol=0, atol=0)
    torch.testing.assert_close(output[:, 64:], torch.zeros_like(output[:, 64:]), rtol=0, atol=0)
    torch.testing.assert_close(pool[3], expected_state[0], rtol=0, atol=0)
    torch.testing.assert_close(storage[0], before[0], rtol=0, atol=0)


def test_paged_chunk_kda_zero_initializes_new_slots():
    torch.manual_seed(47)
    inputs = tuple(tensor.detach() for tensor in _inputs(tokens=128, heads=2))
    q, k, v, cumulative_gate, beta = inputs
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    slots = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device="cuda")
    storage, pool = strided_state_pool(5, q.shape[2], 128, 128, prefix=0)
    before = storage.clone()

    output = paged_chunk_kda(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        pool,
        slots,
        cu_seqlens=cu_seqlens,
        has_initial_state=has_initial_state,
    )
    state_elements = pool[0].numel()
    original_slot = before[4, :state_elements].view_as(pool[4])
    expected_initial = torch.stack((original_slot, torch.zeros_like(pool[2])))
    expected_output, expected_state = chunk_kda(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        expected_initial,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )

    assert expected_state is not None
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(pool[2], expected_state[1], rtol=0, atol=0)
    torch.testing.assert_close(storage[0], before[0], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("batch", "tokens", "explicit_offsets", "expected_route"),
    [
        (1, 128, None, "dense"),
        (2, 128, None, "ragged"),
        (1, 65, None, "ragged"),
        (1, 128, [0, 64, 128], "ragged"),
    ],
)
def test_chunk_kda_selects_direct_dense_or_ragged_route(
    monkeypatch, batch, tokens, explicit_offsets, expected_route
):
    """Keep complete single sequences on the direct launcher without a mode object."""
    module = importlib.import_module("attn_gym.linear.kda.impl.fused")
    routes = []

    def dense_forward(q, _k, v, _gate, _beta, _state, _scale, _tune):
        routes.append("dense")
        factors = q.new_empty((*q.shape[:3], 64))
        return torch.empty_like(v), factors, factors

    def ragged_forward(q, _k, v, _gate, _beta, _state, _cu_seqlens, _chunk_offsets, _scale, _tune):
        routes.append("ragged")
        factors = q.new_empty((*q.shape[:3], 64))
        return torch.empty_like(v), factors, factors

    monkeypatch.setattr(module, "chunk_fwd_op", dense_forward)
    monkeypatch.setattr(module, "chunk_fwd_ragged_op", ragged_forward)
    inputs = tuple(tensor.detach() for tensor in _inputs(batch=batch, tokens=tokens))
    cu_seqlens = (
        None
        if explicit_offsets is None
        else torch.tensor(explicit_offsets, device="cuda", dtype=torch.int32)
    )

    output, state = chunk_kda(*inputs, cu_seqlens=cu_seqlens)

    assert output.shape == inputs[0].shape
    assert state is None
    assert routes == [expected_route]


def test_optimized_chunk_kda_dense_batch_matches_equal_length_packing():
    """Lower a dense batch to independent equal-length packed sequences."""
    torch.manual_seed(43)
    dense_inputs = _inputs(batch=2, tokens=128, heads=2, initial_state=True)
    packed_inputs = _clone_inputs(dense_inputs)
    q, k, v, gate, beta, initial_state = packed_inputs
    packed_shape = (1, 256, 2, 128)
    packed_inputs = (
        q.reshape(packed_shape),
        k.reshape(packed_shape),
        v.reshape(packed_shape),
        gate.reshape(packed_shape),
        beta.reshape(1, 256, 2),
        initial_state,
    )
    cu_seqlens = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)

    dense_output, dense_state = chunk_kda(*dense_inputs, output_final_state=True)
    packed_output, packed_state = chunk_kda(
        *packed_inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    assert dense_state is not None and packed_state is not None
    torch.testing.assert_close(
        dense_output.reshape_as(packed_output), packed_output, rtol=0, atol=0
    )
    torch.testing.assert_close(dense_state, packed_state, rtol=0, atol=0)

    d_output = torch.randn_like(dense_output)
    d_state = torch.randn_like(dense_state)
    dense_gradients = torch.autograd.grad(
        (dense_output, dense_state), dense_inputs, (d_output, d_state)
    )
    packed_gradients = torch.autograd.grad(
        (packed_output, packed_state),
        packed_inputs,
        (d_output.reshape_as(packed_output), d_state),
    )
    for dense_gradient, packed_gradient in zip(dense_gradients, packed_gradients, strict=True):
        torch.testing.assert_close(
            dense_gradient.reshape_as(packed_gradient), packed_gradient, rtol=0, atol=0
        )


def test_chunk_kda_single_sequence_metadata_matches_dense_path():
    """Treat caller-provided ``[0, T]`` metadata as the dense degenerate case."""
    torch.manual_seed(17)
    dense_inputs = _inputs(tokens=128, heads=2, initial_state=True)
    explicit_inputs = _clone_inputs(dense_inputs)

    dense_output, dense_state = chunk_kda(*dense_inputs, output_final_state=True)
    explicit_output, explicit_state = chunk_kda(
        *explicit_inputs,
        cu_seqlens=torch.tensor([0, 128], device="cuda", dtype=torch.int32),
        output_final_state=True,
    )
    assert dense_state is not None and explicit_state is not None
    torch.testing.assert_close(explicit_output, dense_output, rtol=0, atol=0)
    torch.testing.assert_close(explicit_state, dense_state, rtol=0, atol=0)

    d_output = torch.randn_like(dense_output)
    d_state = torch.randn_like(dense_state)
    dense_gradients = torch.autograd.grad(
        (dense_output, dense_state), dense_inputs, (d_output, d_state)
    )
    explicit_gradients = torch.autograd.grad(
        (explicit_output, explicit_state), explicit_inputs, (d_output, d_state)
    )
    # Forward values and most gradients stay bitwise across the two lowerings.
    # The exception is the FP32 gradient at the internal chunk-scan boundary:
    # it is the only output assembled by software FP32 reduction chains (dq/dk/dv come
    # from MMA accumulators with hardware-fixed order), and the dense versus ragged
    # constexpr specializations of the fused wy/dqkg kernel schedule those
    # chains differently, so it accumulates a few rounding steps near zero and
    # gets an absolute bound instead of bit equality.
    gate_index = 3
    for index, (explicit_gradient, dense_gradient) in enumerate(
        zip(explicit_gradients, dense_gradients, strict=True)
    ):
        atol = 1e-8 if index == gate_index else 0.0
        torch.testing.assert_close(explicit_gradient, dense_gradient, rtol=0, atol=atol)


def test_optimized_chunk_kda_autograd_without_initial_state():
    """Exercise the isolated output cotangent path with identical BF16-rounded values."""
    torch.manual_seed(4)
    inputs = _inputs(heads=2)
    q, k, v, gate, beta = inputs
    d_output = torch.randn_like(q)

    output, state = chunk_kda(*inputs)
    assert state is None
    actual_gradients = torch.autograd.grad(output, inputs, d_output)
    reference_output, _ = naive_chunk_kda(
        q.float(),
        k.float(),
        v.float(),
        gate * LOG2_E,
        beta,
        chunk_size=64,
    )
    expected_gradients = torch.autograd.grad(reference_output, inputs, d_output.float())
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        error = (actual_gradient.float() - expected_gradient).abs().max()
        tolerance = 5e-3 + 5e-3 * expected_gradient.abs().max()
        assert error <= tolerance


def test_kda_tensor_descriptor_and_pointer_paths_match(monkeypatch):
    """Keep the Blackwell TMA path bitwise equivalent to its pointer fallback."""
    torch.manual_seed(41)
    descriptor_inputs = _inputs(heads=2)
    d_output = torch.randn_like(descriptor_inputs[0])

    descriptor_output, state = chunk_kda(*descriptor_inputs)
    assert state is None
    descriptor_gradients = torch.autograd.grad(descriptor_output, descriptor_inputs, d_output)

    for module_name, attribute in (
        ("attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o", "_can_use_tensor_descriptors"),
        ("attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk", "can_use_tensor_descriptors"),
    ):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, attribute, lambda *_tensors: False)

    pointer_inputs = _clone_inputs(descriptor_inputs)
    pointer_output, state = chunk_kda(*pointer_inputs)
    assert state is None
    pointer_gradients = torch.autograd.grad(pointer_output, pointer_inputs, d_output)

    torch.testing.assert_close(pointer_output, descriptor_output, rtol=0, atol=0)
    for pointer_gradient, descriptor_gradient in zip(
        pointer_gradients,
        descriptor_gradients,
        strict=True,
    ):
        torch.testing.assert_close(pointer_gradient, descriptor_gradient, rtol=0, atol=0)


def test_kda_offset_width_specializations_match(monkeypatch):
    """Keep the normal int32 and large-storage int64 specializations equivalent."""
    torch.manual_seed(40)
    inputs = _inputs(heads=2)
    d_output = torch.randn_like(inputs[0])

    for module_name, attribute in (
        ("attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o", "_can_use_tensor_descriptors"),
        ("attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk", "can_use_tensor_descriptors"),
    ):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, attribute, lambda *_tensors: False)

    output32, state = chunk_kda(*inputs)
    assert state is None
    gradients32 = torch.autograd.grad(output32, inputs, d_output)

    for module_name in (
        "attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop",
        "attn_gym.linear.kda.fwd.triton.chunk_delta_h",
        "attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o",
        "attn_gym.linear.kda.fwd.triton.recompute_w_u",
        "attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk",
    ):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "requires_int64_offsets", lambda *_tensors: True)

    inputs64 = _clone_inputs(inputs)
    output64, state = chunk_kda(*inputs64)
    assert state is None
    gradients64 = torch.autograd.grad(output64, inputs64, d_output)

    torch.testing.assert_close(output64, output32, rtol=0, atol=0)
    for gradient64, gradient32 in zip(gradients64, gradients32, strict=True):
        torch.testing.assert_close(gradient64, gradient32, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("attribute", "kernel", "config_type"),
    (
        ("chunk_kda_bwd_intra", chunk_kda_bwd_intra, ChunkKdaBwdIntraConfig),
        ("chunk_kda_bwd_wy_dqkg", chunk_kda_bwd_wy_dqkg, ChunkKdaBwdWyDqkgConfig),
    ),
)
def test_backward_tuning_configs(monkeypatch, attribute, kernel, config_type):
    """Keep each explicit and tuned schedule equivalent to its default."""
    torch.manual_seed(5)
    inputs = _inputs(tokens=320, heads=2)
    d_output = torch.randn_like(inputs[0])
    candidates = tuple(config_type(value) for value in (1, 2))
    launches = (
        kernel,
        *(functools.partial(kernel, config=config) for config in candidates),
        functools.partial(kernel, autotune=True),
        functools.partial(kernel, autotune=True, configs=candidates),
    )

    gradients = []
    for launch in launches:
        monkeypatch.setattr(_chunk_kda_bwd_module, attribute, launch)
        cloned_inputs = _clone_inputs(inputs)
        output, _ = chunk_kda(*cloned_inputs)
        gradients.append(torch.autograd.grad(output, cloned_inputs, d_output))

    for result in gradients[1:]:
        for candidate, expected in zip(result, gradients[0], strict=True):
            torch.testing.assert_close(candidate, expected, rtol=0, atol=0)


def test_delta_h_dispatch_counts_packed_sequences(monkeypatch):
    """Choose BV from logical rather than physical packed batch size."""
    dispatch = importlib.import_module("attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd")
    captured = {}
    result = tuple(torch.empty(0, device="cuda") for _ in range(3))

    def fake_delta_h(*args, bv, **kwargs):
        captured["bv"] = bv
        captured["metadata"] = args[5]
        return result

    class DeviceProperties:
        multi_processor_count = 100

    monkeypatch.setattr(dispatch, "_blackwell_delta_h_bwd_dhu_dv_fused_packed", fake_delta_h)
    monkeypatch.setattr(
        dispatch,
        "get_device_properties",
        lambda _device: DeviceProperties(),
    )
    tensor = torch.empty(1, 7, 2, 128, device="cuda")
    cu_seqlens = torch.arange(8, dtype=torch.int32, device="cuda")
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, tensor.shape[1], 64)
    actual = dispatch.blackwell_delta_h_bwd_dhu_dv_fused_dispatch(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        metadata=metadata,
    )

    assert actual is result
    assert captured["bv"] == 32
    assert captured["metadata"] is metadata


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunk_kda_op_registration(dtype):
    """Validate schema, fake tensors, and AOT dispatch for both raw forward operators."""
    torch.manual_seed(5)
    q, k, v, gate, beta, initial_state = _inputs(initial_state=True, dtype=dtype)
    cumulative_gate = chunk_cumsum_ref(gate * LOG2_E, 64)
    q, k, v = _strided_qkv_views(q, k, v)
    # The raw operators are intentionally not differentiable; autograd lives in the
    # _ChunkKDA autograd.Function, covered by the gradient tests above.
    args = (
        q.detach(),
        k.detach(),
        v.detach(),
        cumulative_gate.detach(),
        beta.detach(),
        initial_state.detach(),
        _DEFAULT_SCALE,
        True,
    )
    torch.library.opcheck(_chunk_kda_fwd_op, args, rtol=2e-2, atol=2e-3)
    torch.library.opcheck(_chunk_kda_fwd_with_state_op, args, rtol=2e-2, atol=2e-3)


def test_chunk_kda_paged_op_registration():
    """Validate the mutating paged schema, fake tensor, and AOT behavior."""
    torch.manual_seed(37)
    q, k, v, cumulative_gate, beta = (tensor.detach() for tensor in _inputs(tokens=128, heads=2))
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, q.shape[1], 64)
    _storage, pool = strided_state_pool(5, q.shape[2], 128, 128, prefix=0)
    slots = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device="cuda")

    torch.library.opcheck(
        _chunk_kda_fwd_ragged_paged_op,
        (
            q,
            k,
            v,
            cumulative_gate,
            beta,
            pool,
            slots,
            has_initial_state,
            cu_seqlens,
            metadata.chunk_offsets,
            True,
        ),
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunk_kda_backward_op_registration(dtype):
    """Validate the intentionally first-order backward operator registrations."""
    torch.manual_seed(6)
    q, k, v, gate, beta, initial_state = _inputs(initial_state=True, dtype=dtype)
    cumulative_gate = chunk_cumsum_ref(gate * LOG2_E, 64)
    with torch.no_grad():
        _output, state, Aqk, Akk = _chunk_kda_fwd_with_state_op(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            _DEFAULT_SCALE,
            True,
        )
    torch.library.opcheck(
        _chunk_kda_bwd_op,
        (
            q.detach(),
            k.detach(),
            v.detach(),
            cumulative_gate.detach(),
            beta.detach(),
            Aqk,
            Akk,
            None,
            None,
            None,
            torch.randn_like(state),
            None,
            _DEFAULT_SCALE,
            False,
            True,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )
    torch.library.opcheck(
        _chunk_kda_bwd_with_state_grad_op,
        (
            q.detach(),
            k.detach(),
            v.detach(),
            cumulative_gate.detach(),
            beta.detach(),
            Aqk,
            Akk,
            None,
            None,
            None,
            torch.randn_like(state),
            initial_state.detach(),
            _DEFAULT_SCALE,
            False,
            True,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )


@pytest.mark.parametrize(
    "dtype,initial_state,output_final_state",
    [
        (torch.float32, False, False),
        (torch.bfloat16, False, True),
        (torch.bfloat16, True, False),
        (torch.bfloat16, True, True),
        (torch.float16, True, True),
    ],
    ids=["fp32-output", "no-initial", "no-final", "initial-and-final", "fp16"],
)
def test_chunk_kda_fullgraph_forward_and_backward(dtype, initial_state, output_final_state):
    """Capture the public operation and its registered backward as one strict graph."""
    torch.manual_seed(7)
    eager_inputs = _inputs(initial_state=initial_state, dtype=dtype)
    compiled_inputs = _clone_inputs(eager_inputs)

    def operation(*args):
        return chunk_kda(*args, output_final_state=output_final_state)

    expected_output, expected_state = operation(*eager_inputs)
    actual_output, actual_state = torch.compile(operation, fullgraph=True)(*compiled_inputs)
    torch.testing.assert_close(actual_output, expected_output)
    if output_final_state:
        assert actual_state is not None and expected_state is not None
        torch.testing.assert_close(actual_state, expected_state)
    else:
        assert actual_state is None and expected_state is None

    d_output = torch.randn_like(expected_output)
    expected_results = (
        (expected_output,) if expected_state is None else (expected_output, expected_state)
    )
    actual_results = (actual_output,) if actual_state is None else (actual_output, actual_state)
    cotangents = (
        (d_output,) if expected_state is None else (d_output, torch.randn_like(expected_state))
    )
    expected_gradients = torch.autograd.grad(expected_results, eager_inputs, cotangents)
    actual_gradients = torch.autograd.grad(actual_results, compiled_inputs, cotangents)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_chunk_kda_shape_packed_batch_fullgraph_forward_and_backward():
    """Compile generated dense-batch boundaries through graph-safe packed scheduling."""
    torch.manual_seed(47)
    eager_inputs = _inputs(batch=2, tokens=128)
    compiled_inputs = _clone_inputs(eager_inputs)

    def operation(*args):
        return chunk_kda(*args)[0]

    expected = operation(*eager_inputs)
    actual = torch.compile(operation, fullgraph=True)(*compiled_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    d_output = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, eager_inputs, d_output)
    actual_gradients = torch.autograd.grad(actual, compiled_inputs, d_output)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_chunk_kda_packed_fullgraph_forward_and_backward():
    """Keep packed metadata inside the strict compiled forward and backward graph."""
    torch.manual_seed(17)
    eager_inputs = _inputs(tokens=128, heads=2)
    compiled_inputs = list(_clone_inputs(eager_inputs))
    compiled_inputs[:3] = _strided_qkv_views(*eager_inputs[:3])
    compiled_inputs = tuple(compiled_inputs)
    assert not any(tensor.is_contiguous() for tensor in compiled_inputs[:3])
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    def operation(*args):
        return chunk_kda(*args, cu_seqlens=cu_seqlens)[0]

    expected = operation(*eager_inputs)
    actual = torch.compile(operation, fullgraph=True)(*compiled_inputs)
    torch.testing.assert_close(actual, expected)
    d_output = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, eager_inputs, d_output)
    actual_gradients = torch.autograd.grad(actual, compiled_inputs, d_output)
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_paged_chunk_kda_fullgraph_compile():
    """Compile the public paged path while preserving in-place cache updates."""
    torch.manual_seed(41)
    inputs = tuple(tensor.detach() for tensor in _inputs(tokens=192, heads=2))
    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)
    slots = torch.tensor([5, 2], device="cuda", dtype=torch.int32)
    eager_storage, eager_pool = strided_state_pool(6, inputs[0].shape[2], 128, 128, prefix=0)
    compiled_storage, compiled_pool = strided_state_pool(6, inputs[0].shape[2], 128, 128, prefix=0)
    compiled_storage.copy_(eager_storage)

    def operation(state_pool):
        return paged_chunk_kda(
            *inputs,
            state_pool,
            slots,
            cu_seqlens=cu_seqlens,
        )

    expected = operation(eager_pool)
    output = torch.compile(operation, fullgraph=True)(compiled_pool)

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled_storage, eager_storage, rtol=0, atol=0)


def test_chunk_kda_fullgraph_unaligned_qkv_fallback():
    """Apply the same runtime alignment fallback in eager and compiled execution."""
    torch.manual_seed(31)
    expected_inputs = _inputs(heads=2)
    actual_inputs = list(_clone_inputs(expected_inputs))
    actual_inputs[:3] = tuple(_unaligned_view(tensor) for tensor in actual_inputs[:3])
    actual_inputs = tuple(actual_inputs)
    assert all(tensor.data_ptr() % 16 == 2 for tensor in actual_inputs[:3])

    def operation(*args):
        return chunk_kda(*args)[0]

    expected = operation(*expected_inputs)
    actual = torch.compile(operation, fullgraph=True)(*actual_inputs)
    torch.testing.assert_close(actual, expected)
    d_output = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, d_output)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, d_output)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_chunk_kda_reduce_overhead_backward():
    """Keep CUDA Graph warmup allocations local to the compiled invocation."""
    inputs = _inputs(heads=2)

    def operation(*args):
        output, _ = chunk_kda(*args)
        return output.float().square().mean()

    loss = torch.compile(operation, fullgraph=True, mode="reduce-overhead")(*inputs)
    gradients = torch.autograd.grad(loss, inputs)
    assert all(gradient.isfinite().all() for gradient in gradients)


def test_chunk_kda_rejects_higher_order_autograd():
    """Keep the composed backward explicitly first-order."""
    q, k, v, gate, beta = _inputs()
    output, _ = chunk_kda(q, k, v, gate, beta)
    gradient = torch.autograd.grad(output.float().sum(), q, create_graph=True)[0]
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(gradient.float().sum(), q)


def test_chunk_kda_cuda_graph_replay():
    """Capture and replay the composed forward and backward after warmup."""
    torch.manual_seed(7)
    inputs = _inputs(initial_state=True)
    warm_output, warm_state = chunk_kda(*inputs, output_final_state=True)
    assert warm_state is not None
    warm_gradients = torch.autograd.grad(warm_output.float().sum() + warm_state.sum(), inputs)
    del warm_output, warm_state, warm_gradients
    gc.collect()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, state = chunk_kda(*inputs, output_final_state=True)
        assert state is not None
        gradients = torch.autograd.grad(output.float().sum() + state.sum(), inputs)
    captured_state = state.clone()

    with torch.no_grad():
        inputs[2].mul_(0.5)
    expected_inputs = _clone_inputs(inputs)
    expected_output, expected_state = chunk_kda(*expected_inputs, output_final_state=True)
    assert expected_state is not None
    expected_gradients = torch.autograd.grad(
        expected_output.float().sum() + expected_state.sum(), expected_inputs
    )
    graph.replay()
    torch.cuda.synchronize()

    assert not torch.equal(state, captured_state)
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(state, expected_state)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_chunk_kda_packed_cuda_graph_replays_boundaries_and_backward(dtype):
    """Replay packed metadata and gradients without capture-time host transfers."""
    torch.manual_seed(23)
    inputs = _inputs(tokens=192, heads=2, dtype=dtype)
    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)
    warm_output, _ = chunk_kda(*inputs, cu_seqlens=cu_seqlens)
    warm_gradients = torch.autograd.grad(warm_output.float().sum(), inputs)
    del warm_output, warm_gradients
    gc.collect()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, _ = chunk_kda(*inputs, cu_seqlens=cu_seqlens)
        gradients = torch.autograd.grad(output.float().sum(), inputs)

    with torch.no_grad():
        inputs[2].mul_(0.5)
        cu_seqlens.copy_(torch.tensor([0, 128, 192], device="cuda", dtype=torch.int32))
    expected_inputs = _clone_inputs(inputs)
    expected_output, _ = chunk_kda(*expected_inputs, cu_seqlens=cu_seqlens)
    expected_gradients = torch.autograd.grad(expected_output.float().sum(), expected_inputs)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected_output)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_paged_chunk_kda_cuda_graph_replay():
    """Replay changed cache routing, values, and packed boundaries."""
    torch.manual_seed(43)
    inputs = tuple(tensor.detach() for tensor in _inputs(tokens=192, heads=2))
    q, k, v, cumulative_gate, beta = inputs
    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)
    slots = torch.tensor([5, 2], device="cuda", dtype=torch.int32)
    storage, pool = strided_state_pool(7, q.shape[2], 128, 128, prefix=0)
    paged_chunk_kda(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        pool,
        slots,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = paged_chunk_kda(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            pool,
            slots,
            cu_seqlens=cu_seqlens,
        )

    with torch.no_grad():
        storage.add_(0.25)
        slots.copy_(torch.tensor([6, 1], device="cuda", dtype=torch.int32))
        cu_seqlens.copy_(torch.tensor([0, 128, 192], device="cuda", dtype=torch.int32))
        v.mul_(0.9)
    expected_storage = storage.clone()
    state_elements = pool[0].numel()
    expected_pool = expected_storage[:, :state_elements].view_as(pool)
    expected = paged_chunk_kda(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        expected_pool,
        slots,
        cu_seqlens=cu_seqlens,
    )

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


@pytest.mark.parametrize("cut", [5, 8, 37])
def test_chunk_kda_forward_prefix_ignores_future_tokens(cut):
    """Perturbing only future tokens must leave causal-prefix output bitwise unchanged.

    The intra-chunk rebase splits ``2^{g_i - g_j}`` into two separately rounded factors,
    so a reference row drawn from the future lets a later token perturb an earlier
    result even though the reference cancels in exact arithmetic. The cuts land inside a
    16-row subchunk, where a midpoint reference moves as the suffix changes. See
    NOTE [Causal gate reference].
    """
    torch.manual_seed(11)
    tokens, heads = 128, 2
    shape = (1, tokens, heads, 128)
    query = F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(torch.bfloat16)
    key = F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(torch.bfloat16)
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    beta = torch.rand(1, tokens, heads, device="cuda")
    increments = -torch.rand(shape, device="cuda")

    def forward(tensors, gate_increments):
        """Run the public core from per-token gate increments."""
        output, _ = chunk_kda(*tensors, gate_increments, beta)
        return output

    baseline = forward((query, key, value), increments)

    # Rebuild the suffix of every operand, including the gate the reference is drawn from.
    variant = [tensor.clone() for tensor in (query, key, value)]
    for tensor in variant:
        tensor[:, cut:] = torch.randn_like(tensor[:, cut:])
    variant_increments = increments.clone()
    variant_increments[:, cut:] = -torch.rand_like(variant_increments[:, cut:])
    perturbed = forward(tuple(variant), variant_increments)

    assert not torch.equal(baseline[:, cut:], perturbed[:, cut:]), (
        "the suffix perturbation did not take effect, so the test proves nothing"
    )
    torch.testing.assert_close(perturbed[:, :cut], baseline[:, :cut], rtol=0, atol=0)


def test_chunk_kda_validates_public_contract():
    """Reject malformed inputs at the opaque boundary before a kernel launch."""
    q, k, v, gate, beta = (tensor.detach() for tensor in _inputs())

    with pytest.raises(ValueError, match="k must have shape"):
        chunk_kda(q, k[:, :-1], v, gate, beta)
    with pytest.raises(ValueError, match="beta must have shape"):
        chunk_kda(q, k, v, gate, beta[:, :-1])
    with pytest.raises(ValueError, match="initial_state must have shape"):
        chunk_kda(q, k, v, gate, beta, q.new_empty(1, 1, 64, 128))
    with pytest.raises(ValueError, match="nonempty"):
        chunk_kda(q[:, :0], k[:, :0], v[:, :0], gate[:, :0], beta[:, :0])
    with pytest.raises(ValueError, match="same device"):
        chunk_kda(q, k, v, gate, beta.cpu())
    with pytest.raises(TypeError, match="inputs must use one of"):
        chunk_kda(q.double(), k, v, gate, beta)


def test_paged_chunk_kda_validates_public_contract():
    q, k, v, gate, beta = (tensor.detach() for tensor in _inputs())
    _storage, pool = strided_state_pool(3, q.shape[2], 128, 128, prefix=0)
    slots = torch.tensor([2], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="paged state pool must have shape"):
        paged_chunk_kda(q, k, v, gate, beta, pool[:, :, :-1], slots)
    with pytest.raises(ValueError, match="state pool must be on q.device"):
        paged_chunk_kda(q, k, v, gate, beta, pool.cpu(), slots)
    with pytest.raises(ValueError, match="state_indices must be"):
        paged_chunk_kda(q, k, v, gate, beta, pool, slots.long())
    with pytest.raises(RuntimeError, match="inference-only"):
        paged_chunk_kda(q.requires_grad_(), k, v, gate, beta, pool, slots)
