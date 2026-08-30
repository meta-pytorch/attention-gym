"""Numerical, registration, and compilation coverage for fused chunk GDN."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import chunk_gdn
from attn_gym.linear.gdn.bwd.triton.chunk_gdn_bwd_intra import (
    chunk_gdn_bwd_intra_dense,
)
from attn_gym.linear.gdn.ops import (
    chunk_bwd_op,
    chunk_bwd_with_state_grad_op,
    chunk_fwd_op,
    chunk_fwd_packed_op,
    chunk_fwd_packed_with_state_op,
    chunk_fwd_with_state_op,
)
from attn_gym.linear.kda.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.ops import _plain_gate_scan_op
from attn_gym.testing.kda import assert_matches_low_precision_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="fused chunk GDN requires CUDA capability 10.0 or newer",
)


def make_inputs(
    *,
    batch: int = 1,
    tokens: int = 64,
    key_heads: int = 2,
    value_heads: int = 2,
    gate_kind: str = "mild",
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Create deterministic quantized inputs with a selected unbounded-gate regime."""
    torch.manual_seed(23)
    qk_shape = (batch, tokens, key_heads, 128)
    value_shape = (batch, tokens, value_heads, 128)
    q = F.normalize(torch.randn(qk_shape, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn(qk_shape, device="cuda"), dim=-1).to(dtype)
    v = torch.randn(value_shape, device="cuda", dtype=dtype)
    match gate_kind:
        case "mild":
            gate = -F.softplus(torch.randn(value_shape[:3], device="cuda"))
        case "spikes":
            gate = torch.full(value_shape[:3], -0.1, device="cuda")
            gate[:, 7::16] = -20.0
        case "unbounded":
            gate = torch.full(value_shape[:3], -20.0, device="cuda")
        case _:
            raise ValueError(f"unknown gate kind {gate_kind!r}")
    beta = torch.sigmoid(torch.randn(value_shape[:3], device="cuda"))
    state = torch.randn(batch, value_heads, 128, 128, device="cuda")
    tensors = (q, k, v, gate, beta, state)
    if requires_grad:
        tensors = tuple(tensor.requires_grad_() for tensor in tensors)
    return tensors


def misaligned_like(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into contiguous storage beginning one element past an aligned allocation."""
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    result = storage[1:].view(tensor.shape)
    result.copy_(tensor)
    assert result.is_contiguous() and result.data_ptr() % 16 != 0
    return result


def run_with_gradients(
    inputs: tuple[torch.Tensor, ...],
    impl: str,
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """Run output/state and all input gradients under one shared scalar loss."""
    output, state = chunk_gdn(
        *inputs[:5],
        inputs[5],
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl=impl,
    )
    assert state is not None
    gradients = torch.autograd.grad(
        output.float().square().mean() + state.float().square().mean(), inputs
    )
    return output, state, *gradients


@pytest.mark.parametrize(
    ("batch", "tokens", "key_heads", "value_heads", "gate_kind", "lengths"),
    [
        (1, 64, 2, 2, "mild", None),
        (1, 65, 1, 4, "spikes", None),
        (2, 65, 1, 4, "mild", None),
        (1, 64, 1, 1, "unbounded", None),
        (1, 192, 1, 4, "spikes", [65, 0, 127]),
    ],
)
def test_fused_chunk_matches_low_precision_reference(
    batch: int,
    tokens: int,
    key_heads: int,
    value_heads: int,
    gate_kind: str,
    lengths: list[int] | None,
):
    """Bound fused forward and all gradients by the eager low-precision error."""
    inputs = make_inputs(
        batch=batch,
        tokens=tokens,
        key_heads=key_heads,
        value_heads=value_heads,
        gate_kind=gate_kind,
        requires_grad=True,
    )
    if lengths is not None:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        inputs = (
            *inputs[:5],
            torch.randn(
                len(lengths),
                value_heads,
                128,
                128,
                device="cuda",
                requires_grad=True,
            ),
        )
    else:
        cu_seqlens = None

    fused_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    golden_inputs = tuple(tensor.detach().double().requires_grad_() for tensor in inputs)
    actual = run_with_gradients(fused_inputs, "fused", cu_seqlens)
    expected = run_with_gradients(reference_inputs, "reference", cu_seqlens)
    golden = run_with_gradients(golden_inputs, "reference", cu_seqlens)

    for name, result, high_precision, reference in zip(
        ("output", "state", "dq", "dk", "dv", "dgate", "dbeta", "dstate"),
        actual,
        golden,
        expected,
        strict=True,
    ):
        if high_precision.abs().max().item() < 1e-12:
            assert torch.isfinite(result).all()
            assert (result.double() - high_precision).abs().max().item() <= 1e-12
        else:
            assert_matches_low_precision_reference(result, high_precision, reference, name)

    if lengths is not None:
        torch.testing.assert_close(actual[1][1], inputs[5][1], rtol=0, atol=0)


def test_fp16_fused_chunk_matches_reference():
    """Validate the advertised FP16 forward and gradient path."""
    inputs = make_inputs(tokens=64, dtype=torch.float16, requires_grad=True)
    fused_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    golden_inputs = tuple(tensor.detach().double().requires_grad_() for tensor in inputs)
    actual = run_with_gradients(fused_inputs, "fused")
    expected = run_with_gradients(reference_inputs, "reference")
    golden = run_with_gradients(golden_inputs, "reference")
    for name, result, high_precision, reference in zip(
        ("output", "state", "dq", "dk", "dv", "dgate", "dbeta", "dstate"),
        actual,
        golden,
        expected,
        strict=True,
    ):
        if name == "dstate":
            # dstate crosses both the forward state restage and reverse-state recurrence;
            # account for both low-precision boundaries while retaining a normalized guard.
            high = high_precision.double()
            actual_error = (result.double() - high).abs().max().item()
            reference_error = (reference.double() - high).abs().max().item()
            rounding = torch.finfo(torch.float16).eps * high.abs().max().item()
            assert actual_error <= 4 * (reference_error + rounding)
            relative_rms = torch.linalg.vector_norm(
                result.double() - high
            ) / torch.linalg.vector_norm(high).clamp_min(1e-30)
            assert relative_rms.item() <= 5e-2
        else:
            assert_matches_low_precision_reference(
                result,
                high_precision,
                reference,
                name,
                source_dtype=torch.float16,
            )


def test_packed_tail_ignores_nan_capacity_slack():
    """Never evaluate or consume storage beyond the terminal packed offset."""
    q, k, v, gate, beta, _state = make_inputs(tokens=192)
    state = torch.randn(2, v.shape[2], 128, 128, device="cuda")
    active_tokens = 127
    for tensor in (q, k, v, gate, beta):
        tensor[:, active_tokens:] = torch.nan
    cu_seqlens = torch.tensor([0, 65, active_tokens], device="cuda", dtype=torch.int32)
    actual_output, actual_state = chunk_gdn(
        q,
        k,
        v,
        gate,
        beta,
        state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="fused",
    )
    expected_output, expected_state = chunk_gdn(
        q,
        k,
        v,
        gate,
        beta,
        state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="reference",
    )
    assert actual_state is not None and expected_state is not None
    assert torch.isfinite(actual_output[:, :active_tokens]).all()
    torch.testing.assert_close(
        actual_output[:, :active_tokens],
        expected_output[:, :active_tokens],
        rtol=2e-2,
        atol=2e-3,
    )
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-2, atol=2e-3)


def test_packed_backward_zeros_capacity_slack():
    """Return exact zero gradients for physical rows beyond the terminal packed offset."""
    inputs = make_inputs(tokens=192, requires_grad=True)
    state = torch.randn(2, inputs[2].shape[2], 128, 128, device="cuda", requires_grad=True)
    leaves = (*inputs[:5], state)
    active_tokens = 127
    cu_seqlens = torch.tensor([0, 65, active_tokens], device="cuda", dtype=torch.int32)
    output, final_state = chunk_gdn(
        *leaves[:5],
        leaves[5],
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="fused",
    )
    assert final_state is not None
    gradients = torch.autograd.grad(
        output[:, :active_tokens].float().square().mean() + final_state.float().square().mean(),
        leaves,
    )
    for gradient in gradients[:5]:
        torch.testing.assert_close(
            gradient[:, active_tokens:],
            torch.zeros_like(gradient[:, active_tokens:]),
            rtol=0,
            atol=0,
        )


def test_scalar_intra_masks_poisoned_akk_diagonal_and_upper():
    """Treat only strict-lower dAkk entries as meaningful before any arithmetic."""
    q, k, _v, _gate, beta, _state = make_inputs(tokens=64, key_heads=1, value_heads=1)
    cumulative_gate = (-20.0 * torch.arange(1, 65, device="cuda")).view(1, 64, 1)
    d_aqk = torch.randn(1, 64, 1, 64, device="cuda")
    d_akk = torch.randn_like(d_aqk)
    row = torch.arange(64, device="cuda")
    strict = row[:, None] > row[None, :]
    clean_d_akk = torch.where(strict[None, :, None, :], d_akk, 0.0)
    poisoned_d_akk = torch.where(strict[None, :, None, :], d_akk, torch.nan)
    d_gate_raw = torch.zeros_like(q, dtype=torch.float32)
    expected = chunk_gdn_bwd_intra_dense(
        q, k, cumulative_gate, beta, d_aqk, clean_d_akk, d_gate_raw
    )
    actual = chunk_gdn_bwd_intra_dense(
        q, k, cumulative_gate, beta, d_aqk, poisoned_d_akk, d_gate_raw
    )
    for result, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_misaligned_contiguous_inputs_and_cotangent():
    """Normalize valid storage-offset views at the opaque kernel boundary."""
    q, k, v, gate, beta, state = make_inputs(tokens=64)
    misaligned_inputs = (
        misaligned_like(q).requires_grad_(),
        k.requires_grad_(),
        v.requires_grad_(),
        gate.requires_grad_(),
        misaligned_like(beta).requires_grad_(),
        misaligned_like(state).requires_grad_(),
    )
    output, final_state = chunk_gdn(
        *misaligned_inputs[:5],
        misaligned_inputs[5],
        output_final_state=True,
        impl="fused",
    )
    assert final_state is not None
    gradients = torch.autograd.grad(
        output.float().square().mean() + final_state.float().square().mean(),
        misaligned_inputs,
    )
    assert torch.isfinite(output).all()
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    args, _output, state_output, inverse = raw_args()
    misaligned_d_output = misaligned_like(torch.randn_like(args[2]))
    backward_args = (
        *args[:5],
        inverse,
        misaligned_d_output,
        torch.randn_like(state_output),
        args[5],
        None,
        None,
        args[6],
    )
    outputs = chunk_bwd_with_state_grad_op(*backward_args)
    assert all(torch.isfinite(result).all() for result in outputs)


def test_no_state_output_only_and_state_only_gradients():
    """Exercise no-state/no-final-state and a missing output cotangent."""
    inputs = make_inputs(tokens=64, key_heads=1, value_heads=4, requires_grad=True)
    output, state = chunk_gdn(*inputs[:5], impl="fused")
    assert state is None
    output_gradients = torch.autograd.grad(output.float().square().mean(), inputs[:5])
    assert all(torch.isfinite(gradient).all() for gradient in output_gradients)

    state_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    _output, final_state = chunk_gdn(
        *state_inputs[:5],
        state_inputs[5],
        output_final_state=True,
        impl="fused",
    )
    assert final_state is not None
    state_gradients = torch.autograd.grad(final_state.square().mean(), state_inputs)
    assert all(torch.isfinite(gradient).all() for gradient in state_gradients)


def test_int64_layouts_fail_before_kernel_launch(monkeypatch):
    """Reject wide layouts until every new Triton kernel has an i64 specialization."""
    import attn_gym.linear.gdn.impl.chunk as chunk_impl

    monkeypatch.setattr(chunk_impl, "requires_int64_offsets", lambda *args: True)
    inputs = make_inputs(tokens=64)
    with pytest.raises(ValueError, match="int64 tensor offsets"):
        chunk_gdn(*inputs[:5], impl="fused")


def test_raw_ops_reject_pre_blackwell_devices(monkeypatch):
    """Keep capability validation inside the real registered CUDA launcher."""
    import attn_gym.linear.gdn.impl.chunk as chunk_impl

    args, _output, _state, _inverse = raw_args()
    monkeypatch.setattr(
        chunk_impl,
        "get_device_properties",
        lambda device: SimpleNamespace(major=9),
    )
    with pytest.raises(ValueError, match="capability 10.0"):
        chunk_fwd_with_state_op(*args)


def raw_args(tokens: int = 64, heads: int = 2):
    """Construct detached raw-op arguments and their forward tapes."""
    q, k, v, gate, beta, state = make_inputs(tokens=tokens, key_heads=heads, value_heads=heads)
    cumulative = _plain_gate_scan_op(gate.unsqueeze(-1), None, None, False).squeeze(-1)
    args = (q, k, v, cumulative, beta, state, 128**-0.5)
    with torch.no_grad():
        output, final_state, inverse = chunk_fwd_with_state_op(*args)
    return args, output, final_state, inverse


def test_dense_raw_operator_registration():
    """Validate dense forward/backward schemas, fakes, and AOT dispatch."""
    args, output, final_state, inverse = raw_args()
    torch.library.opcheck(chunk_fwd_op, args)
    torch.library.opcheck(chunk_fwd_with_state_op, args)
    backward_args = (
        *args[:5],
        inverse,
        torch.randn_like(output),
        torch.randn_like(final_state),
        args[5],
        None,
        None,
        args[6],
    )
    utilities = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    torch.library.opcheck(chunk_bwd_op, backward_args, test_utils=utilities)
    torch.library.opcheck(chunk_bwd_with_state_grad_op, backward_args, test_utils=utilities)


def test_packed_raw_operator_registration():
    """Validate fixed-capacity packed forward/backward registrations."""
    q, k, v, gate, beta, _state = make_inputs(tokens=128)
    state = torch.randn(2, v.shape[2], 128, 128, device="cuda")
    cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, q.shape[1], 64)
    cumulative = _plain_gate_scan_op(
        gate.unsqueeze(-1), cu_seqlens, metadata.chunk_offsets, False
    ).squeeze(-1)
    args = (
        q,
        k,
        v,
        cumulative,
        beta,
        state,
        cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
        128**-0.5,
    )
    torch.library.opcheck(chunk_fwd_packed_op, args)
    torch.library.opcheck(chunk_fwd_packed_with_state_op, args)
    with torch.no_grad():
        output, final_state, inverse = chunk_fwd_packed_with_state_op(*args)
    backward_args = (
        *args[:5],
        inverse,
        torch.randn_like(output),
        torch.randn_like(final_state),
        state,
        cu_seqlens,
        metadata.chunk_offsets,
        128**-0.5,
    )
    utilities = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    torch.library.opcheck(chunk_bwd_op, backward_args, test_utils=utilities)
    torch.library.opcheck(
        chunk_bwd_with_state_grad_op,
        backward_args,
        test_utils=utilities,
    )


@pytest.mark.parametrize("packed", [False, True])
def test_public_fullgraph_forward_backward(packed: bool):
    """Compile the documented fused operation with strict forward/backward capture."""
    tokens = 128 if packed else 64
    inputs = make_inputs(tokens=tokens, key_heads=1, value_heads=4, requires_grad=True)
    cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32) if packed else None
    if packed:
        inputs = (
            *inputs[:5],
            torch.randn(2, 4, 128, 128, device="cuda", requires_grad=True),
        )
    expected_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    expected = run_with_gradients(expected_inputs, "fused", cu_seqlens)
    compiled = torch.compile(
        partial(chunk_gdn, impl="fused", output_final_state=True),
        fullgraph=True,
    )
    output, state = compiled(*inputs[:5], inputs[5], cu_seqlens=cu_seqlens)
    assert state is not None
    gradients = torch.autograd.grad(
        output.float().square().mean() + state.float().square().mean(), inputs
    )
    actual = output, state, *gradients
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_public_dynamic_tokens_forward_backward():
    """Reuse one dynamic fullgraph callable across complete dense chunk counts."""
    compiled = torch.compile(
        partial(chunk_gdn, impl="fused", output_final_state=True),
        fullgraph=True,
        dynamic=True,
    )
    for tokens in (64, 128):
        inputs = make_inputs(tokens=tokens, key_heads=1, value_heads=4, requires_grad=True)
        output, state = compiled(*inputs[:5], inputs[5])
        assert state is not None
        gradients = torch.autograd.grad(
            output.float().square().mean() + state.float().square().mean(), inputs
        )
        assert torch.isfinite(output).all()
        assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_backward_cuda_graph_replay():
    """Capture and replay the training backward with fixed saved tensors."""
    inputs = make_inputs(tokens=64, key_heads=1, value_heads=4, requires_grad=True)
    output, state = chunk_gdn(
        *inputs[:5],
        inputs[5],
        output_final_state=True,
        impl="fused",
    )
    assert state is not None
    d_output = torch.randn_like(output)
    d_state = torch.randn_like(state)

    def backward():
        return torch.autograd.grad(
            (output, state),
            inputs,
            grad_outputs=(d_output, d_state),
            retain_graph=True,
        )

    backward()
    torch.cuda.synchronize()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backward()
    graph.replay()
    torch.cuda.synchronize()
    actual = tuple(gradient.clone() for gradient in captured)
    expected = backward()
    torch.cuda.synchronize()
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_packed_cuda_graph_replays_boundaries():
    """Reread packed boundaries and chunk offsets on CUDA Graph replay."""
    q, k, v, gate, beta, _state = make_inputs(tokens=128)
    state = torch.randn(2, v.shape[2], 128, 128, device="cuda")
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    def run():
        return chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            impl="fused",
        )

    for _ in range(2):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    actual = captured[0].clone(), captured[1].clone()
    expected = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
