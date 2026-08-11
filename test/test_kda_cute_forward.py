"""Composition tests for the optimized CuTeDSL KDA forward pipeline."""

from __future__ import annotations

import gc
import importlib

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import naive_chunk_kda_from_cumulative
from attn_gym.linear.kda.naive import chunk_cumsum_ref

pytest.importorskip("cutlass")

from attn_gym.linear.kda.fwd.cute import chunk_kda
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    _chunk_kda_bwd_custom_op,
    _chunk_kda_fwd_custom_op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL KDA forward pipeline requires CUDA capability 10.0 or newer",
)


def _inputs(*, heads: int = 1, initial_state: bool = False, dtype: torch.dtype = torch.bfloat16):
    batch, tokens, head_dim = 1, 128, 128
    shape = (batch, tokens, heads, head_dim)
    q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    cumulative_gate = chunk_cumsum_ref(-torch.rand(shape, device="cuda"), 64)
    beta = torch.rand(batch, tokens, heads, device="cuda")
    tensors = [q, k, v, cumulative_gate, beta]
    if initial_state:
        tensors.append(torch.randn(batch, heads, head_dim, head_dim, device="cuda") * 0.01)
    return tuple(tensor.requires_grad_() for tensor in tensors)


def _clone_inputs(inputs):
    return tuple(tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in inputs)


def test_optimized_chunk_kda_matches_reference():
    """Check forward values and the isolated final-state cotangent path."""
    torch.manual_seed(3)
    inputs = _inputs(initial_state=True)
    q, k, v, cumulative_gate, beta, initial_state = inputs

    output, state = chunk_kda(*inputs, output_final_state=True)
    assert state is not None
    assert output.dtype == torch.bfloat16
    assert state.dtype == torch.float32
    expected, expected_state = naive_chunk_kda_from_cumulative(
        q.float(),
        k.float(),
        v.float(),
        cumulative_gate,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
    )
    assert expected_state is not None
    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(state, expected_state, rtol=2e-2, atol=2e-3)

    d_state = torch.randn_like(state)
    state_inputs = (k, v, cumulative_gate, beta, initial_state)
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


def test_optimized_chunk_kda_autograd_without_initial_state():
    """Exercise the isolated output cotangent path with identical BF16-rounded values."""
    torch.manual_seed(4)
    inputs = _inputs(heads=2)
    q, k, v, cumulative_gate, beta = inputs
    d_output = torch.randn_like(q)

    output, state = chunk_kda(*inputs)
    assert state is None
    actual_gradients = torch.autograd.grad(output, inputs, d_output)
    reference_output, _ = naive_chunk_kda_from_cumulative(
        q.float(),
        k.float(),
        v.float(),
        cumulative_gate,
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

    for module_name in (
        "attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o",
        "attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_dav",
    ):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "_can_use_tensor_descriptors", lambda *_tensors: False)

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

    for module_name in (
        "attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o",
        "attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_dav",
    ):
        module = importlib.import_module(module_name)
        monkeypatch.setattr(module, "_can_use_tensor_descriptors", lambda *_tensors: False)

    output32, state = chunk_kda(*inputs)
    assert state is None
    gradients32 = torch.autograd.grad(output32, inputs, d_output)

    for module_name in (
        "attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop",
        "attn_gym.linear.kda.fwd.triton.chunk_delta_h",
        "attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o",
        "attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_dav",
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


def test_chunk_kda_custom_op_registration():
    """Validate schema, fake tensors, autograd registration, and AOT dispatch."""
    torch.manual_seed(5)
    q, k, v, cumulative_gate, beta, initial_state = _inputs(initial_state=True)
    torch.library.opcheck(
        _chunk_kda_fwd_custom_op,
        (
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            True,
            False,
            False,
        ),
        rtol=2e-2,
        atol=2e-3,
    )


def test_chunk_kda_backward_custom_op_registration():
    """Validate the intentionally first-order backward operator registration."""
    torch.manual_seed(6)
    q, k, v, cumulative_gate, beta, initial_state = _inputs(initial_state=True)
    with torch.no_grad():
        _output, state, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks = _chunk_kda_fwd_custom_op(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            True,
            False,
            False,
        )
    torch.library.opcheck(
        _chunk_kda_bwd_custom_op,
        (
            q.detach(),
            k.detach(),
            v.detach(),
            cumulative_gate.detach(),
            beta.detach(),
            Aqk,
            Akk,
            cu_seqlens,
            chunk_indices,
            num_chunks,
            None,
            torch.randn_like(state),
            initial_state.detach(),
            False,
            False,
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
    ],
    ids=["fp32-output", "no-initial", "no-final", "initial-and-final"],
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
    q, k, v, cumulative_gate, beta = _inputs()
    output, _ = chunk_kda(q, k, v, cumulative_gate, beta)
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


def test_chunk_kda_validates_public_contract():
    """Reject malformed inputs at the opaque boundary before a kernel launch."""
    q, k, v, cumulative_gate, beta = (tensor.detach() for tensor in _inputs())

    with pytest.raises(ValueError, match="k must have shape"):
        chunk_kda(q, k[:, :-1], v, cumulative_gate, beta)
    with pytest.raises(ValueError, match="beta must have shape"):
        chunk_kda(q, k, v, cumulative_gate, beta[:, :-1])
    with pytest.raises(ValueError, match="initial_state must have shape"):
        chunk_kda(q, k, v, cumulative_gate, beta, q.new_empty(1, 1, 64, 128))
    with pytest.raises(ValueError, match="nonempty token"):
        chunk_kda(q[:, :0], k[:, :0], v[:, :0], cumulative_gate[:, :0], beta[:, :0])
    with pytest.raises(ValueError, match="same device"):
        chunk_kda(q, k, v, cumulative_gate, beta.cpu())
    with pytest.raises(TypeError, match="inputs must use one of"):
        chunk_kda(q.double(), k, v, cumulative_gate, beta)
