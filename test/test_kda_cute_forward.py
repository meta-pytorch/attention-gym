"""Composition tests for the optimized CuTeDSL KDA forward pipeline."""

from __future__ import annotations

import functools
import gc
import importlib

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import naive_chunk_kda_from_cumulative
from attn_gym.linear.kda.naive import chunk_cumsum_ref

pytest.importorskip("cutlass")

from attn_gym.linear.kda.bwd.cute import chunk_kda_bwd as _chunk_kda_bwd_module
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import (
    ChunkKdaBwdIntraConfig,
    chunk_kda_bwd_intra,
)
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
    ChunkKdaBwdWyDqkgConfig,
    chunk_kda_bwd_wy_dqkg,
)
from attn_gym.linear.kda.fwd.cute import chunk_kda
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    _chunk_kda_bwd_custom_op,
    _chunk_kda_fwd_custom_op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTeDSL KDA forward pipeline requires CUDA capability 10.0 or newer",
)


def _inputs(
    *,
    tokens: int = 128,
    heads: int = 1,
    initial_state: bool = False,
    dtype: torch.dtype = torch.bfloat16,
):
    batch, head_dim = 1, 128
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
    q, k, v, cumulative_gate, beta, initial_state = inputs
    outputs = []
    states = []
    for sequence, (start, end) in enumerate(spans):
        output, state = naive_chunk_kda_from_cumulative(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            cumulative_gate[:, start:end],
            beta[:, start:end],
            initial_state=initial_state[sequence : sequence + 1],
            output_final_state=True,
            chunk_size=64,
        )
        outputs.append(output)
        assert state is not None
        states.append(state)
    return torch.cat(outputs, dim=1), torch.cat(states)


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


def test_optimized_chunk_kda_packed_matches_independent_sequences():
    """Keep packed sequence boundaries first-class through forward and backward."""
    torch.manual_seed(13)
    inputs = _inputs(tokens=192, heads=2, initial_state=True)
    q, k, v, cumulative_gate, beta, _initial_state = inputs
    initial_state = torch.randn(2, 2, 128, 128, device="cuda", requires_grad=True) * 0.01
    inputs = (q, k, v, cumulative_gate, beta, initial_state)
    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)

    output, state = chunk_kda(
        *inputs,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    assert state is not None and state.shape == (2, 2, 128, 128)

    spans = ((0, 64), (64, 192))
    reference_inputs = (q.float(), k.float(), v.float(), cumulative_gate, beta, initial_state)
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
    for explicit_gradient, dense_gradient in zip(explicit_gradients, dense_gradients, strict=True):
        torch.testing.assert_close(explicit_gradient, dense_gradient, rtol=0, atol=0)


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
        functools.partial(kernel, tune=True),
        functools.partial(kernel, tune=True, configs=candidates),
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
    dispatch = importlib.import_module(
        "attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd_v1_dispatch"
    )
    captured = {}
    result = tuple(torch.empty(0, device="cuda") for _ in range(3))

    def fake_delta_h(**kwargs):
        captured["bv"] = kwargs["bv"]
        return result

    class DeviceProperties:
        multi_processor_count = 100

    monkeypatch.setattr(dispatch, "blackwell_delta_h_bwd_dhu_v1", fake_delta_h)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: DeviceProperties())
    tensor = torch.empty(1, 1, 2, 128, device="cuda")
    cu_seqlens = torch.arange(8, dtype=torch.int32, device="cuda")
    actual = dispatch.blackwell_delta_h_bwd_dhu_dispatch(
        tensor,
        tensor,
        tensor,
        tensor,
        tensor,
        cu_seqlens=cu_seqlens,
    )

    assert actual is result
    assert captured["bv"] == 32


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
            None,
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
            None,
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


def test_chunk_kda_packed_fullgraph_forward_and_backward():
    """Keep packed metadata inside the strict compiled forward and backward graph."""
    torch.manual_seed(17)
    eager_inputs = _inputs(tokens=128, heads=2)
    compiled_inputs = _clone_inputs(eager_inputs)
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


def test_chunk_kda_packed_cuda_graph_replays_boundaries_and_backward():
    """Replay packed metadata and gradients without capture-time host transfers."""
    torch.manual_seed(23)
    inputs = _inputs(tokens=192, heads=2)
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
