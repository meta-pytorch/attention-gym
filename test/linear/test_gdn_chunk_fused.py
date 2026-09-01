"""Numerical, registration, and compilation coverage for fused chunk GDN."""

from __future__ import annotations

from functools import partial
from types import SimpleNamespace

import pytest
import torch

from attn_gym._backends.cute import normalize_tma_rows
from attn_gym.linear import (
    active_token_mask,
    chunk_gdn,
    mask_inactive_token_gradients,
    paged_chunk_gdn,
    recurrent_gdn,
)
from attn_gym.linear.gdn.bwd.triton.chunk_gdn_bwd_intra import (
    chunk_gdn_bwd_intra_dense,
)
from attn_gym.linear.gdn.ops import (
    chunk_bwd_op,
    chunk_bwd_with_state_grad_op,
    chunk_fwd_op,
    chunk_fwd_packed_op,
    chunk_fwd_packed_paged_op,
    chunk_fwd_packed_with_state_op,
    chunk_fwd_with_state_op,
)
from attn_gym.linear.kda.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.ops import _plain_gate_scan_op
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    cumulative_sequence_offsets,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="fused chunk GDN requires CUDA capability 9.0 or newer",
)
requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="paged chunk GDN requires CUDA capability 10.0 or newer",
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
    """Adapt the shared GDN fixture to this file's dense-input shorthand."""
    gate_pattern = {
        "mild": "softplus",
        "spikes": "periodic_negative_twenty",
        "unbounded": "uniform_negative_twenty",
    }[gate_kind]
    *inputs, cu_seqlens = make_gdn_test_inputs(
        tokens,
        batch=batch,
        key_heads=key_heads,
        value_heads=value_heads,
        gate_pattern=gate_pattern,
        dtype=dtype,
        seed=23,
        value_scale=1.0,
        state_scale=1.0,
        sigmoid_beta=True,
        requires_grad=requires_grad,
    )
    assert cu_seqlens is None
    return tuple(inputs)


def misaligned_like(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into contiguous storage beginning one element past an aligned allocation."""
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    result = storage[1:].view(tensor.shape)
    result.copy_(tensor)
    assert result.is_contiguous() and result.data_ptr() % 16 != 0
    return result


def token_strided_like(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into aligned storage with padding between otherwise compact token rows."""
    batch, tokens, heads, dim = tensor.shape
    storage = torch.empty(batch, tokens, heads * dim + 8, dtype=tensor.dtype, device=tensor.device)
    result = storage[..., : heads * dim].view(tensor.shape)
    result.copy_(tensor)
    assert result.stride() == (tokens * (heads * dim + 8), heads * dim + 8, dim, 1)
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


def run_final_state_gradients(
    inputs: tuple[torch.Tensor, ...],
    impl: str,
) -> tuple[torch.Tensor, ...]:
    """Run a no-initial-state final-state loss and return its five input gradients."""
    _output, state = chunk_gdn(*inputs, output_final_state=True, impl=impl)
    assert state is not None
    gradients = torch.autograd.grad(state.float().square().mean(), inputs, allow_unused=True)
    return state, *(
        torch.zeros_like(value) if gradient is None else gradient
        for value, gradient in zip(inputs, gradients, strict=True)
    )


def force_portable_backward(monkeypatch) -> None:
    """Force the Hopper Triton backward while retaining the current GPU forward."""
    import attn_gym.linear.gdn.impl.chunk as chunk_impl

    monkeypatch.setattr(
        chunk_impl,
        "get_device_properties",
        lambda device: SimpleNamespace(major=9),
    )


@pytest.mark.parametrize(
    ("batch", "tokens", "key_heads", "value_heads", "gate_kind", "lengths", "portable"),
    [
        (1, 64, 2, 2, "mild", None, False),
        (1, 65, 1, 4, "spikes", None, False),
        (2, 65, 1, 4, "mild", None, False),
        (1, 64, 1, 1, "unbounded", None, True),
        (1, 64, 1, 64, "mild", None, True),
        (1, 192, 1, 4, "spikes", [65, 0, 127], True),
    ],
)
def test_fused_chunk_matches_low_precision_reference(
    batch: int,
    tokens: int,
    key_heads: int,
    value_heads: int,
    gate_kind: str,
    lengths: list[int] | None,
    portable: bool,
    monkeypatch,
):
    """Bound Blackwell and Hopper-route gradients by the eager low-precision error."""
    if portable:
        force_portable_backward(monkeypatch)
    inputs = make_inputs(
        batch=batch,
        tokens=tokens,
        key_heads=key_heads,
        value_heads=value_heads,
        gate_kind=gate_kind,
        requires_grad=True,
    )
    if lengths is not None:
        cu_seqlens = cumulative_sequence_offsets(lengths)
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
            # The portable gate VJP subtracts two independently reduced FP32 terms.
            # Uniform -20 gates underflow the true result to zero, leaving a sub-nanounit
            # cancellation residue that is still far below low-precision significance.
            zero_atol = 1e-9 if portable and name == "dgate" else 1e-12
            assert (result.double() - high_precision).abs().max().item() <= zero_atol
        else:
            assert_matches_low_precision_reference(result, high_precision, reference, name)

    if lengths is not None:
        torch.testing.assert_close(actual[1][1], inputs[5][1], rtol=0, atol=0)


@pytest.mark.parametrize("portable", [False, True])
def test_fp16_fused_chunk_matches_reference(portable: bool, monkeypatch):
    """Validate FP16 forward and gradients on both backward implementations."""
    if portable:
        force_portable_backward(monkeypatch)
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


def test_chunk_state_continues_in_recurrent_decode():
    """Carry the fused training/prefill state directly into Hopper decode."""
    q, k, v, gate, beta, initial_state = make_inputs(
        tokens=65,
        key_heads=1,
        value_heads=4,
    )
    with torch.no_grad():
        _prefill_output, prefill_state = chunk_gdn(
            q[:, :64],
            k[:, :64],
            v[:, :64],
            gate[:, :64],
            beta[:, :64],
            initial_state,
            output_final_state=True,
            impl="fused",
        )
        assert prefill_state is not None
        expected_output, expected_state = recurrent_gdn(
            q[:, 64:],
            k[:, 64:],
            v[:, 64:],
            gate[:, 64:],
            beta[:, 64:],
            prefill_state,
            output_final_state=True,
            impl="reference",
        )
        actual_output, actual_state = recurrent_gdn(
            q[:, 64:],
            k[:, 64:],
            v[:, 64:],
            gate[:, 64:],
            beta[:, 64:],
            prefill_state,
            output_final_state=True,
            autotune=False,
            impl="fused",
        )

    torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(actual_state, expected_state, rtol=2e-2, atol=2e-3)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() >= (10, 0),
    reason="pre-Blackwell capability guard only",
)
def test_paged_chunk_rejects_pre_blackwell():
    """Keep the Blackwell-only paged prefill boundary explicit on Hopper."""
    q, k, v, gate, beta, _state = make_inputs(tokens=64, key_heads=1, value_heads=2)
    state_cache = torch.randn(3, 2, 128, 128, device="cuda")
    state_indices = torch.tensor([1], device="cuda", dtype=torch.int32)
    with torch.no_grad(), pytest.raises(ValueError, match="CUDA capability 10.0"):
        paged_chunk_gdn(q, k, v, gate, beta, state_cache, state_indices)


@requires_blackwell
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_paged_chunk_rejects_unsupported_qkv_dtype(dtype: torch.dtype):
    q, k, v, gate, beta, _state = make_inputs(tokens=64, key_heads=1, value_heads=2)
    state_cache = torch.randn(3, 2, 128, 128, device="cuda")
    state_indices = torch.tensor([1], device="cuda", dtype=torch.int32)

    with torch.no_grad(), pytest.raises(TypeError, match="matching float16 or bfloat16 QKV"):
        paged_chunk_gdn(
            q.to(dtype),
            k.to(dtype),
            v.to(dtype),
            gate,
            beta,
            state_cache,
            state_indices,
        )


@requires_blackwell
def test_paged_chunk_raw_operator_registration():
    """Validate mutation, fake output layout, and dynamic AOT dispatch."""
    q, k, v, gate, beta, _state = make_inputs(tokens=128, key_heads=1, value_heads=2)
    v = token_strided_like(v)
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, q.shape[1], 64)
    cumulative = _plain_gate_scan_op(
        gate.unsqueeze(-1), cu_seqlens, metadata.chunk_offsets, False
    ).squeeze(-1)
    state_cache = torch.randn(4, 2, 128, 128, device="cuda")
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device="cuda")

    torch.library.opcheck(
        chunk_fwd_packed_paged_op,
        (
            q,
            k,
            v,
            cumulative,
            beta,
            state_cache,
            state_indices,
            has_initial_state,
            cu_seqlens,
            metadata.chunk_offsets,
            metadata.capacity,
            128**-0.5,
        ),
    )


@requires_blackwell
def test_paged_chunk_matches_gather_scatter():
    """Advance selected slots directly without copying state through the caller."""
    q, k, v, gate, beta, _state = make_inputs(
        tokens=192,
        key_heads=1,
        value_heads=4,
    )
    q, k, v = (token_strided_like(tensor) for tensor in (q, k, v))
    assert all(normalize_tma_rows(tensor).data_ptr() == tensor.data_ptr() for tensor in (q, k, v))
    cu_seqlens = torch.tensor([0, 65, 192], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([4, 2], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device="cuda")
    state_elements = 4 * 128 * 128
    storage = torch.randn(6, state_elements + 17, device="cuda")
    state_cache = storage[:, :state_elements].view(6, 4, 128, 128)
    expected_storage = storage.clone()
    expected_cache = expected_storage[:, :state_elements].view_as(state_cache)
    expected_initial_state = torch.stack((expected_cache[4], torch.zeros_like(expected_cache[2])))

    expected_output, expected_state = chunk_gdn(
        q,
        k,
        v,
        gate,
        beta,
        expected_initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        impl="fused",
    )
    assert expected_state is not None
    expected_cache[state_indices.long()] = expected_state

    with torch.no_grad():
        actual_output = paged_chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


@requires_blackwell
def test_paged_chunk_dense_batch_matches_gather_scatter():
    q, k, v, gate, beta, _state = make_inputs(
        batch=2,
        tokens=65,
        key_heads=1,
        value_heads=2,
    )
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    state_cache = torch.randn(5, 2, 128, 128, device="cuda")
    expected_cache = state_cache.clone()

    expected_output, expected_state = chunk_gdn(
        q,
        k,
        v,
        gate,
        beta,
        expected_cache[state_indices.long()].clone(),
        output_final_state=True,
        impl="fused",
    )
    assert expected_state is not None
    expected_cache[state_indices.long()] = expected_state

    with torch.no_grad():
        actual_output = paged_chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices,
        )

    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(state_cache, expected_cache, rtol=0, atol=0)


@requires_blackwell
def test_paged_chunk_state_continues_in_recurrent_decode():
    """Use one paged pool directly across chunk prefill and recurrent decode."""
    q, k, v, gate, beta, _state = make_inputs(tokens=65, key_heads=1, value_heads=2)
    state_indices = torch.tensor([3], device="cuda", dtype=torch.int32)
    prefill_offsets = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    decode_offsets = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    initial_cache = torch.randn(5, 2, 128, 128, device="cuda")
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()

    with torch.no_grad():
        _expected_prefill, expected_state = chunk_gdn(
            q[:, :64],
            k[:, :64],
            v[:, :64],
            gate[:, :64],
            beta[:, :64],
            expected_cache[state_indices.long()].clone(),
            cu_seqlens=prefill_offsets,
            output_final_state=True,
            impl="fused",
        )
        assert expected_state is not None
        expected_cache[state_indices.long()] = expected_state
        expected_decode, _ = recurrent_gdn(
            q[:, 64:],
            k[:, 64:],
            v[:, 64:],
            gate[:, 64:],
            beta[:, 64:],
            expected_cache,
            cu_seqlens=decode_offsets,
            state_indices=state_indices,
        )

        paged_chunk_gdn(
            q[:, :64],
            k[:, :64],
            v[:, :64],
            gate[:, :64],
            beta[:, :64],
            actual_cache,
            state_indices,
            cu_seqlens=prefill_offsets,
        )
        actual_decode, _ = recurrent_gdn(
            q[:, 64:],
            k[:, 64:],
            v[:, 64:],
            gate[:, 64:],
            beta[:, 64:],
            actual_cache,
            cu_seqlens=decode_offsets,
            state_indices=state_indices,
        )

    torch.testing.assert_close(actual_decode, expected_decode, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)


@requires_blackwell
def test_paged_chunk_handles_padding_and_empty_fresh_slots():
    """Null routes stay untouched while an empty newly assigned slot is cleared."""
    q, k, v, gate, beta, _state = make_inputs(tokens=64, key_heads=1, value_heads=2)
    cu_seqlens = torch.tensor([0, 0, 64], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 0], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([False, True], device="cuda")
    state_cache = torch.randn(5, 2, 128, 128, device="cuda")
    original_cache = state_cache.clone()

    with torch.no_grad():
        output = paged_chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )

    torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
    torch.testing.assert_close(state_cache[3], torch.zeros_like(state_cache[3]), rtol=0, atol=0)
    torch.testing.assert_close(
        state_cache[[0, 1, 2, 4]], original_cache[[0, 1, 2, 4]], rtol=0, atol=0
    )


@requires_blackwell
def test_paged_chunk_fullgraph_compile():
    """Keep paged mutation opaque and correctly aliased under fullgraph compilation."""
    q, k, v, gate, beta, _state = make_inputs(tokens=65, key_heads=1, value_heads=2)
    cu_seqlens = torch.tensor([0, 31, 65], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device="cuda")
    initial_cache = torch.randn(5, 2, 128, 128, device="cuda")
    expected_cache = initial_cache.clone()
    actual_cache = initial_cache.clone()

    def run(state_cache):
        return paged_chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
        )

    with torch.no_grad():
        expected = run(expected_cache)
        actual = torch.compile(run, fullgraph=True)(actual_cache)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_cache, expected_cache, rtol=0, atol=0)


@requires_blackwell
def test_paged_chunk_cuda_graph_replay():
    """Replay changed cache routing, values, and packed boundaries."""
    q, k, v, gate, beta, _state = make_inputs(tokens=192, key_heads=1, value_heads=2)
    cu_seqlens = torch.tensor([0, 64, 192], device="cuda", dtype=torch.int32)
    state_indices = torch.tensor([5, 2], device="cuda", dtype=torch.int32)
    state_elements = 2 * 128 * 128
    storage = torch.randn(7, state_elements + 17, device="cuda")
    state_cache = storage[:, :state_elements].view(7, 2, 128, 128)
    with torch.no_grad():
        paged_chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices,
            cu_seqlens=cu_seqlens,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        output = paged_chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            state_cache,
            state_indices,
            cu_seqlens=cu_seqlens,
        )

    with torch.no_grad():
        storage.add_(0.25)
        state_indices.copy_(torch.tensor([6, 1], device="cuda", dtype=torch.int32))
        cu_seqlens.copy_(torch.tensor([0, 128, 192], device="cuda", dtype=torch.int32))
        v.mul_(0.9)
        expected_storage = storage.clone()
        expected_cache = expected_storage[:, :state_elements].view_as(state_cache)
        expected = paged_chunk_gdn(
            q,
            k,
            v,
            gate,
            beta,
            expected_cache,
            state_indices,
            cu_seqlens=cu_seqlens,
        )

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


@pytest.mark.parametrize("portable", [False, True])
def test_packed_caller_masks_inactive_capacity_gradients(portable: bool, monkeypatch):
    """Block unspecified inactive gradients at the caller-to-ragged boundary."""
    if portable:
        force_portable_backward(monkeypatch)
    inputs = make_inputs(tokens=192, requires_grad=True)
    state = torch.randn(2, inputs[2].shape[2], 128, 128, device="cuda", requires_grad=True)
    leaves = (*inputs[:5], state)
    active_tokens = 127
    cu_seqlens = torch.tensor([0, 65, active_tokens], device="cuda", dtype=torch.int32)
    active_mask = active_token_mask(inputs[0], cu_seqlens)
    masked_inputs = tuple(
        mask_inactive_token_gradients(tensor, active_mask) for tensor in inputs[:5]
    )
    output, final_state = chunk_gdn(
        *masked_inputs,
        state,
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


@pytest.mark.parametrize("tokens", [64, 65])
def test_aligned_outer_strided_qkv_matches_compact(tokens: int):
    """Preserve aligned token-strided QKV without changing forward or backward results."""
    inputs = make_inputs(tokens=tokens, key_heads=1, value_heads=4, requires_grad=True)
    expected_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    strided_inputs = (
        *(token_strided_like(tensor).requires_grad_() for tensor in inputs[:3]),
        *(tensor.detach().clone().requires_grad_() for tensor in inputs[3:]),
    )
    assert all(
        normalize_tma_rows(tensor).data_ptr() == tensor.data_ptr() for tensor in strided_inputs[:3]
    )

    expected = run_with_gradients(expected_inputs, "fused")
    actual = run_with_gradients(strided_inputs, "fused")
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-3)


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


@pytest.mark.parametrize(("batch", "tokens"), [(1, 65), (2, 129)])
def test_fp16_no_initial_state_final_state_gradients(batch: int, tokens: int):
    """Preserve small FP32 state cotangents across a partial FP16 chunk."""
    inputs = make_inputs(
        batch=batch,
        tokens=tokens,
        key_heads=1,
        value_heads=1,
        dtype=torch.float16,
        requires_grad=True,
    )[:5]

    fused_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    golden_inputs = tuple(tensor.detach().double().requires_grad_() for tensor in inputs)
    actual = run_final_state_gradients(fused_inputs, "fused")
    expected = run_final_state_gradients(reference_inputs, "reference")
    golden = run_final_state_gradients(golden_inputs, "reference")
    for name, result, high_precision, reference in zip(
        ("state", "dq", "dk", "dv", "dgate", "dbeta"),
        actual,
        golden,
        expected,
        strict=True,
    ):
        if high_precision.abs().max().item() < 1e-12:
            torch.testing.assert_close(result.double(), high_precision, rtol=0, atol=1e-12)
        else:
            assert_matches_low_precision_reference(
                result,
                high_precision,
                reference,
                name,
                source_dtype=torch.float16,
            )


def test_int64_layouts_fail_before_kernel_launch(monkeypatch):
    """Reject wide layouts until every new Triton kernel has an i64 specialization."""
    import attn_gym.linear.gdn.impl.chunk as chunk_impl

    monkeypatch.setattr(chunk_impl, "requires_int64_offsets", lambda *args: True)
    inputs = make_inputs(tokens=64)
    with pytest.raises(ValueError, match="int64 tensor offsets"):
        chunk_gdn(*inputs[:5], impl="fused")


def test_raw_ops_reject_pre_hopper_devices(monkeypatch):
    """Keep capability validation inside the real registered CUDA launcher."""
    import attn_gym.linear.gdn.impl.chunk as chunk_impl

    args, _output, _state, _inverse = raw_args()
    monkeypatch.setattr(
        chunk_impl,
        "get_device_properties",
        lambda device: SimpleNamespace(major=8),
    )
    with pytest.raises(ValueError, match="capability 9.0"):
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
@pytest.mark.parametrize("portable", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_public_fullgraph_forward_backward(
    packed: bool,
    portable: bool,
    dtype: torch.dtype,
    monkeypatch,
):
    """Compile both backward implementations with strict forward/backward capture."""
    if portable:
        force_portable_backward(monkeypatch)
    tokens = 128 if packed else 64
    inputs = make_inputs(
        tokens=tokens,
        key_heads=1,
        value_heads=4,
        dtype=dtype,
        requires_grad=True,
    )
    cu_seqlens = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32) if packed else None
    if packed:
        inputs = (
            *inputs[:5],
            torch.randn(2, 4, 128, 128, device="cuda", requires_grad=True),
        )
    expected_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    expected = run_with_gradients(expected_inputs, "fused", cu_seqlens)
    compiled = torch.compile(
        partial(
            chunk_gdn,
            impl="fused",
            output_final_state=True,
        ),
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
    """Reuse one dynamic callable across the packed-tail and complete-chunk routes."""
    compiled = torch.compile(
        partial(
            chunk_gdn,
            impl="fused",
            output_final_state=True,
        ),
        fullgraph=True,
        dynamic=True,
    )
    for tokens in (63, 64, 65):
        inputs = make_inputs(tokens=tokens, key_heads=1, value_heads=4, requires_grad=True)
        reference_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
        expected = run_with_gradients(reference_inputs, "reference")
        output, state = compiled(*inputs[:5], inputs[5])
        assert state is not None
        gradients = torch.autograd.grad(
            output.float().square().mean() + state.float().square().mean(), inputs
        )
        for result, reference in zip((output, state, *gradients), expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("portable", [False, True])
def test_backward_cuda_graph_replay(portable: bool, monkeypatch):
    """Capture and replay both training backward implementations."""
    if portable:
        force_portable_backward(monkeypatch)
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
