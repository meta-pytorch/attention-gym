"""Training integration for the CuTeDSL 4.7 KDA forward."""

from __future__ import annotations

import math
import os
from itertools import accumulate

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear.kda.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 KDA path requires nvidia-cutlass-dsl>=4.7",
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 KDA path requires SM100 or SM103",
)

D = 128
SCALE = D**-0.5


def _swap_token_head_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Copy `[B,T,H,D]` data into dense TMA-compatible `[B,H,T,D]` storage order."""
    _, tokens, heads, dim = tensor.shape
    storage = torch.empty(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    return torch.as_strided(
        storage,
        tensor.shape,
        (heads * tokens * dim, dim, tokens * dim, 1),
    ).copy_(tensor)


def _swap_state_head_value_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Copy `[N,H,V,K]` data into dense TMA-compatible `[N,V,H,K]` storage order."""
    _, heads, value_dim, key_dim = tensor.shape
    storage = torch.empty(tensor.numel(), dtype=tensor.dtype, device=tensor.device)
    return torch.as_strided(
        storage,
        tensor.shape,
        (heads * value_dim * key_dim, key_dim, heads * key_dim, 1),
    ).copy_(tensor)


def _pad_state_sequence_storage(tensor: torch.Tensor) -> torch.Tensor:
    """Copy `[N,H,V,K]` data into slots with aligned padding between sequences."""
    sequences, heads, value_dim, key_dim = tensor.shape
    sequence_stride = heads * value_dim * key_dim + 16
    storage = torch.empty(
        sequences * sequence_stride,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.as_strided(
        storage,
        tensor.shape,
        (sequence_stride, value_dim * key_dim, key_dim, 1),
    ).copy_(tensor)


def _make_inputs(
    *,
    requires_grad: bool,
    saturated: bool = False,
    heads: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, ...]:
    lengths = [65, 0, 63]
    q, k, value, gate, beta = make_kda_test_inputs(
        sum(lengths),
        heads=heads,
        seed=97,
        gate_scale=math.log(2.0),
        gate_value=-5.0 if saturated else None,
        log_uniform_gate=True,
        sigmoid_beta=True,
        dtype=dtype,
        normalize_qk=True,
        value_scale=1.0,
        requires_grad=requires_grad,
    )
    initial_state = (torch.randn(len(lengths), heads, D, D, device="cuda") / 100).requires_grad_(
        requires_grad
    )
    return (
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cumulative_sequence_offsets(lengths),
    )


def _reference(
    inputs: tuple[torch.Tensor, ...],
    dtype: torch.dtype,
    *,
    scale: float | None = None,
    initial_state: bool = True,
    packed: bool = True,
    output_final_state: bool = True,
) -> tuple[tuple[torch.Tensor, torch.Tensor | None], tuple[torch.Tensor, ...]]:
    """Run one precision of the shared oracle and return its autograd leaves."""
    operands = clone_kda_inputs(inputs[:5], dtype=dtype)
    state = inputs[5].detach().to(dtype).clone().requires_grad_(inputs[5].requires_grad)
    targets = (*operands, *((state,) if initial_state else ()))
    result = kda_reference(
        *operands,
        state if initial_state else None,
        cu_seqlens=inputs[-1] if packed else None,
        scale=scale,
        output_final_state=output_final_state,
    )
    return result, targets


def _references(
    inputs: tuple[torch.Tensor, ...],
    **kwargs,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, ...],
    tuple[torch.Tensor, ...],
]:
    """Run matched FP64 and FP32 references from the same quantized inputs."""
    low_precision, low_targets = _reference(inputs, torch.float32, **kwargs)
    high_precision, high_targets = _reference(inputs, torch.float64, **kwargs)
    return high_precision, low_precision, high_targets, low_targets


def _reference_gradients(
    result: tuple[torch.Tensor, torch.Tensor | None],
    targets: tuple[torch.Tensor, ...],
    cotangents: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Differentiate an eager reference using cotangents in its compute precision."""
    outputs = tuple(output for output in result if output is not None)
    return torch.autograd.grad(
        outputs,
        targets,
        tuple(cotangent.to(output.dtype) for cotangent, output in zip(cotangents, outputs)),
    )


def _assert_reference(
    actual: torch.Tensor,
    high_precision: torch.Tensor,
    low_precision: torch.Tensor,
    name: str,
    source_dtype: torch.dtype,
) -> None:
    """Apply the shared low-precision error budget with the operand dtype."""
    assert_matches_low_precision_reference(
        actual,
        high_precision,
        low_precision,
        name,
        source_dtype=source_dtype,
    )


def _candidate(*inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    from attn_gym.linear import chunk_kda

    q, k, value, gate, beta, initial_state, cu_seqlens = inputs
    output, final_state = chunk_kda(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        kernel_options={"backend": "mega"},
    )
    assert final_state is not None
    return output, final_state


def _candidate_initial_state_no_final(*inputs: torch.Tensor) -> torch.Tensor:
    from attn_gym.linear import chunk_kda

    q, k, value, gate, beta, initial_state, cu_seqlens = inputs
    output, final_state = chunk_kda(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=False,
        kernel_options={"backend": "mega"},
    )
    assert final_state is None
    return output


def _candidate_no_state(*inputs: torch.Tensor) -> torch.Tensor:
    from attn_gym.linear import chunk_kda

    q, k, value, gate, beta, _, cu_seqlens = inputs
    output, _ = chunk_kda(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens=cu_seqlens,
        kernel_options={"backend": "mega"},
    )
    return output


def _candidate_dense(*inputs: torch.Tensor) -> torch.Tensor:
    from attn_gym.linear import chunk_kda

    output, _ = chunk_kda(*inputs[:5], kernel_options={"backend": "mega"})
    return output


def _candidate_dense_split(*inputs: torch.Tensor) -> torch.Tensor:
    from attn_gym.linear import chunk_kda

    output, _ = chunk_kda(
        *inputs[:5],
        kernel_options={"backend": "mega", "split_backward": True},
    )
    return output


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mega_training_fullgraph_and_six_gradients(dtype: torch.dtype) -> None:
    actual_inputs = _make_inputs(requires_grad=True, dtype=dtype)
    high_precision, low_precision, high_targets, low_targets = _references(actual_inputs)
    actual = torch.compile(_candidate, fullgraph=True)(*actual_inputs)
    assert actual[0].dtype == dtype
    assert actual[1].dtype == torch.float32
    for name, actual_tensor, high_tensor, low_tensor in zip(
        ("output", "final_state"), actual, high_precision, low_precision, strict=True
    ):
        assert high_tensor is not None and low_tensor is not None
        _assert_reference(actual_tensor, high_tensor, low_tensor, name, dtype)

    cotangents = (torch.randn_like(actual[0]), torch.randn_like(actual[1]))
    high_grads = _reference_gradients(high_precision, high_targets, cotangents)
    low_grads = _reference_gradients(low_precision, low_targets, cotangents)
    actual_grads = torch.autograd.grad(actual, actual_inputs[:-1], cotangents)
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        assert actual_grad.dtype == (dtype if index < 3 else torch.float32)
        _assert_reference(actual_grad, high_grad, low_grad, f"gradient {index}", dtype)


@pytest.mark.parametrize(
    ("state_layout", "cotangent_layout"),
    [
        (_swap_state_head_value_storage, _pad_state_sequence_storage),
        (_pad_state_sequence_storage, _swap_state_head_value_storage),
    ],
    ids=["permuted-head-value", "padded-sequence"],
)
def test_mega_outer_strided_initial_state_fullgraph_gradients(
    state_layout,
    cotangent_layout,
) -> None:
    inputs = list(_make_inputs(requires_grad=True, heads=2, dtype=torch.bfloat16))
    inputs[5] = state_layout(inputs[5].detach()).requires_grad_()
    actual_inputs = tuple(inputs)
    assert not actual_inputs[5].is_contiguous()

    high_precision, low_precision, high_targets, low_targets = _references(actual_inputs)
    actual = torch.compile(_candidate, fullgraph=True)(*actual_inputs)
    state_cotangent = cotangent_layout(torch.randn_like(actual[1].contiguous()))
    cotangents = (torch.randn_like(actual[0]), state_cotangent)
    high_grads = _reference_gradients(high_precision, high_targets, cotangents)
    low_grads = _reference_gradients(low_precision, low_targets, cotangents)
    actual_grads = torch.autograd.grad(actual, actual_inputs[:-1], cotangents)
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        _assert_reference(
            actual_grad,
            high_grad,
            low_grad,
            f"outer-strided state gradient {index}",
            torch.bfloat16,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mega_initial_state_without_final_state_fullgraph_gradients(dtype: torch.dtype) -> None:
    inputs = _make_inputs(requires_grad=True, dtype=dtype)
    high_precision, low_precision, high_targets, low_targets = _references(
        inputs, output_final_state=False
    )
    actual = torch.compile(_candidate_initial_state_no_final, fullgraph=True)(*inputs)
    high_output, low_output = high_precision[0], low_precision[0]
    _assert_reference(actual, high_output, low_output, "output-only stateful output", dtype)

    d_output = torch.randn_like(actual)
    high_grads = _reference_gradients(high_precision, high_targets, (d_output,))
    low_grads = _reference_gradients(low_precision, low_targets, (d_output,))
    actual_grads = torch.autograd.grad(actual, inputs[:-1], d_output)
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        _assert_reference(
            actual_grad,
            high_grad,
            low_grad,
            f"output-only stateful gradient {index}",
            dtype,
        )


def test_mega_initial_state_without_final_state_selects_output_only_op(monkeypatch) -> None:
    from attn_gym.linear import chunk_kda
    from attn_gym.linear.kda.impl import mega as backend

    selected = []
    inputs = _make_inputs(requires_grad=False)
    initial_state_ptr = inputs[5].data_ptr()
    output_only_op = backend.chunk_mega_packed_fwd_with_initial_state_op
    tensor_clone = torch.Tensor.clone

    def reject_state_clone(tensor, *args, **kwargs):
        if tensor.is_cuda and tensor.data_ptr() == initial_state_ptr:
            pytest.fail("output-only initial-state execution must not clone the state")
        return tensor_clone(tensor, *args, **kwargs)

    def record_output_only(*args):
        selected.append(True)
        return output_only_op(*args)

    monkeypatch.setattr(torch.Tensor, "clone", reject_state_clone)
    monkeypatch.setattr(backend, "chunk_mega_packed_fwd_with_initial_state_op", record_output_only)
    monkeypatch.setattr(
        backend,
        "chunk_mega_packed_fwd_with_state_op",
        lambda *args: pytest.fail("state-returning operator must not run"),
    )
    output, final_state = chunk_kda(
        *inputs[:6],
        cu_seqlens=inputs[-1],
        output_final_state=False,
        kernel_options={"backend": "mega"},
    )
    assert output.shape == inputs[2].shape
    assert final_state is None
    assert selected == [True]


def test_mega_custom_scale_matches_fused_forward_and_gradients() -> None:
    from attn_gym.linear import chunk_kda

    scale = 0.125
    actual_inputs = _make_inputs(requires_grad=True)
    high_precision, low_precision, high_targets, low_targets = _references(
        actual_inputs, scale=scale
    )
    actual = chunk_kda(
        *actual_inputs[:6],
        cu_seqlens=actual_inputs[-1],
        scale=scale,
        output_final_state=True,
        kernel_options={"backend": "mega"},
    )
    for name, actual_tensor, high_tensor, low_tensor in zip(
        ("output", "final_state"), actual, high_precision, low_precision, strict=True
    ):
        assert actual_tensor is not None and high_tensor is not None and low_tensor is not None
        _assert_reference(actual_tensor, high_tensor, low_tensor, name, torch.bfloat16)

    cotangents = (torch.randn_like(actual[0]), torch.randn_like(actual[1]))
    high_grads = _reference_gradients(high_precision, high_targets, cotangents)
    low_grads = _reference_gradients(low_precision, low_targets, cotangents)
    actual_grads = torch.autograd.grad(actual, actual_inputs[:-1], cotangents)
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        _assert_reference(
            actual_grad, high_grad, low_grad, f"custom-scale gradient {index}", torch.bfloat16
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mega_dense_training_fullgraph_and_gradients(dtype: torch.dtype) -> None:
    expected_inputs = list(_make_inputs(requires_grad=True, dtype=dtype))
    actual_inputs = list(_make_inputs(requires_grad=True, dtype=dtype))
    dense_cu = torch.tensor([0, expected_inputs[0].shape[1]], dtype=torch.int32, device="cuda")
    expected_inputs[-1] = dense_cu
    actual_inputs[-1] = dense_cu
    expected_inputs = tuple(expected_inputs)
    actual_inputs = tuple(actual_inputs)
    expected = _candidate_no_state(*expected_inputs)
    actual = torch.compile(_candidate_dense, fullgraph=True)(*actual_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    high_precision, low_precision, high_targets, low_targets = _references(
        actual_inputs,
        initial_state=False,
        packed=False,
        output_final_state=False,
    )
    _assert_reference(actual, high_precision[0], low_precision[0], "dense output", dtype)
    d_output = torch.randn_like(actual)
    high_grads = _reference_gradients(high_precision, high_targets, (d_output,))
    low_grads = _reference_gradients(low_precision, low_targets, (d_output,))
    actual_grads = torch.autograd.grad(actual, actual_inputs[:5], d_output)
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        _assert_reference(actual_grad, high_grad, low_grad, f"dense gradient {index}", dtype)


def test_mega_saturated_gate_six_gradients_are_finite() -> None:
    actual_inputs = _make_inputs(requires_grad=True, saturated=True)
    high_precision, low_precision, high_targets, low_targets = _references(actual_inputs)
    actual = _candidate(*actual_inputs)
    for name, actual_tensor, high_tensor, low_tensor in zip(
        ("output", "final_state"), actual, high_precision, low_precision, strict=True
    ):
        assert high_tensor is not None and low_tensor is not None
        _assert_reference(
            actual_tensor, high_tensor, low_tensor, f"saturated {name}", torch.bfloat16
        )

    cotangents = (torch.randn_like(actual[0]), torch.randn_like(actual[1]))
    high_grads = _reference_gradients(high_precision, high_targets, cotangents)
    low_grads = _reference_gradients(low_precision, low_targets, cotangents)
    actual_grads = torch.autograd.grad(actual, actual_inputs[:-1], cotangents)
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        _assert_reference(
            actual_grad,
            high_grad,
            low_grad,
            f"saturated gradient {index}",
            torch.bfloat16,
        )


def test_mega_local_backward_split_matches_exact_gradients() -> None:
    from attn_gym.linear.kda.impl.mega_ops import chunk_mega_packed_local_bwd_op

    tokens, heads = 2048, 1
    for mode in ("contracting", "no_forgetting"):
        torch.manual_seed(107)
        shape = (1, tokens, heads, D)
        q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16().requires_grad_()
        k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16().requires_grad_()
        value = torch.randn(shape, device="cuda", dtype=torch.bfloat16).requires_grad_()
        gate = (
            torch.empty(shape, device="cuda").uniform_(0.5, 1.0).log()
            if mode == "contracting"
            else torch.full(shape, -1e-5, device="cuda")
        ).requires_grad_()
        beta = torch.sigmoid(torch.randn(1, tokens, heads, device="cuda")).requires_grad_()
        d_output = torch.randn_like(value)
        inputs = (
            q,
            k,
            value,
            gate,
            beta,
            torch.empty(0, device="cuda"),
            torch.empty(0, dtype=torch.int32, device="cuda"),
        )
        high_precision, low_precision, high_targets, low_targets = _references(
            inputs,
            initial_state=False,
            packed=False,
            output_final_state=False,
        )
        high_grads = _reference_gradients(high_precision, high_targets, (d_output,))
        low_grads = _reference_gradients(low_precision, low_targets, (d_output,))
        actual = chunk_mega_packed_local_bwd_op(
            q.detach(),
            k.detach(),
            value.detach(),
            gate.detach(),
            beta.detach(),
            d_output,
            cumulative_sequence_offsets([tokens]),
            True,
            SCALE,
        )
        for index, (actual_grad, high_grad, low_grad) in enumerate(
            zip(actual, high_grads, low_grads, strict=True)
        ):
            _assert_reference(
                actual_grad,
                high_grad,
                low_grad,
                f"{mode} local gradient {index}",
                torch.bfloat16,
            )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [1, 64])
def test_mega_packed_local_backward_matches_exact_gradients(
    heads: int, dtype: torch.dtype
) -> None:
    from attn_gym.linear.kda.impl.mega_ops import chunk_mega_packed_local_bwd_op

    actual_inputs = _make_inputs(requires_grad=True, heads=heads, dtype=dtype)
    high_precision, low_precision, high_targets, low_targets = _references(
        actual_inputs,
        initial_state=False,
        output_final_state=False,
    )
    d_output = torch.randn_like(actual_inputs[2])
    high_grads = _reference_gradients(high_precision, high_targets, (d_output,))
    low_grads = _reference_gradients(low_precision, low_targets, (d_output,))
    actual = chunk_mega_packed_local_bwd_op(
        *(tensor.detach() for tensor in actual_inputs[:5]),
        d_output,
        actual_inputs[-1],
        True,
        SCALE,
    )
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual, high_grads, low_grads, strict=True)
    ):
        _assert_reference(actual_grad, high_grad, low_grad, f"packed gradient {index}", dtype)


def test_mega_public_packed_unsplit_local_backward_matches_exact_gradients(monkeypatch) -> None:
    from attn_gym.linear.kda.impl import mega as backend

    monkeypatch.setattr(backend, "_PACKED_LOCAL_BACKWARD_MIN_TOKENS", 1)
    monkeypatch.setattr(backend, "_LOCAL_BACKWARD_MIN_HEADS", 1)
    selected = []
    local_backward_op = backend.chunk_mega_packed_local_bwd_op

    def record_local_backward(*args):
        selected.append(args[-2])
        return local_backward_op(*args)

    monkeypatch.setattr(backend, "chunk_mega_packed_local_bwd_op", record_local_backward)
    inputs = _make_inputs(requires_grad=True)
    high_precision, low_precision, high_targets, low_targets = _references(
        inputs,
        initial_state=False,
        output_final_state=False,
    )
    actual = _candidate_no_state(*inputs)
    d_output = torch.randn_like(actual)
    high_grads = _reference_gradients(high_precision, high_targets, (d_output,))
    low_grads = _reference_gradients(low_precision, low_targets, (d_output,))
    actual_grads = torch.autograd.grad(actual, inputs[:5], d_output)
    assert selected == [False]
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        _assert_reference(
            actual_grad,
            high_grad,
            low_grad,
            f"public packed unsplit gradient {index}",
            torch.bfloat16,
        )


def test_mega_kernel_options_are_strict() -> None:
    from attn_gym.linear.kda.validation import ResolvedKernelOptions, resolve_kernel_options

    fused = ResolvedKernelOptions("fused", split_backward=False, split_forward=False)
    assert resolve_kernel_options(None) == fused
    assert resolve_kernel_options({}) == fused
    assert resolve_kernel_options({"backend": "mega", "split_backward": True}) == (
        ResolvedKernelOptions("mega", split_backward=True, split_forward=False)
    )
    assert resolve_kernel_options({"backend": "mega", "split_forward": True}) == (
        ResolvedKernelOptions("mega", split_backward=False, split_forward=True)
    )
    with pytest.raises(ValueError, match="unsupported chunk_kda kernel options"):
        resolve_kernel_options({"unknown": True})
    with pytest.raises(ValueError, match="must be 'fused' or 'mega'"):
        resolve_kernel_options({"backend": "triton"})
    for name in ("split_backward", "split_forward"):
        with pytest.raises(TypeError, match="must be a bool"):
            resolve_kernel_options({name: 1})
        with pytest.raises(ValueError, match="requires.*mega"):
            resolve_kernel_options({name: True})


@pytest.mark.parametrize("option", ["split_backward", "split_forward"])
def test_mega_split_schedules_reject_stateful_calls(option: str) -> None:
    from attn_gym.linear import chunk_kda

    inputs = _make_inputs(requires_grad=False)
    with pytest.raises(ValueError, match=f"{option} currently requires a no-state call"):
        chunk_kda(
            *inputs[:6],
            cu_seqlens=inputs[-1],
            output_final_state=True,
            kernel_options={"backend": "mega", option: True},
        )


def _split_forward_pair(
    monkeypatch, dtype: torch.dtype, **gate: float
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, int]:
    """Unsplit and forgetting-horizon Mega forwards of one long dense stream.

    Returns the inputs, both outputs, and the number of work items the split schedule emitted.
    """
    from attn_gym.linear import chunk_kda
    from attn_gym.linear._delta_rule.mega import forward, schedule

    inputs = make_kda_test_inputs(
        8192, seed=107, normalize_qk=True, sigmoid_beta=True, dtype=dtype, **gate
    )
    unsplit, _ = chunk_kda(*inputs, kernel_options={"backend": "mega"})
    schedules = []

    def recording_schedule(*args, **kwargs):
        schedules.append(schedule.prepare_mega_schedule(*args, **kwargs))
        return schedules[-1]

    monkeypatch.setattr(forward, "prepare_mega_schedule", recording_schedule)
    split, _ = chunk_kda(*inputs, kernel_options={"backend": "mega", "split_forward": True})
    torch.cuda.synchronize()
    (recorded,) = schedules
    return inputs, unsplit, split, int(recorded.work_count.item())


def test_mega_split_forward_places_no_cuts_when_the_gate_never_forgets(monkeypatch) -> None:
    _, unsplit, split, work_items = _split_forward_pair(
        monkeypatch, torch.bfloat16, gate_value=-1e-5
    )
    assert work_items == 1
    torch.testing.assert_close(split, unsplit, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mega_split_forward_matches_reference_on_a_contracting_gate(monkeypatch, dtype) -> None:
    """Cuts land only where the gate has saturated, so the result stays within the budget."""
    inputs, _, split, work_items = _split_forward_pair(
        monkeypatch, dtype, gate_scale=math.log(2.0)
    )
    assert work_items > 1, "the contracting gate must actually be cut"
    high, _ = kda_reference(
        *clone_kda_inputs(inputs, dtype=torch.float64), output_final_state=False
    )
    low, _ = kda_reference(
        *clone_kda_inputs(inputs, dtype=torch.float32), output_final_state=False
    )
    _assert_reference(split, high, low, "split forward output", dtype)


def test_mega_cpu_inputs_fail_before_dispatch() -> None:
    from attn_gym.linear import chunk_kda

    shape = (1, 64, 1, D)
    q, k, value = (torch.zeros(shape) for _ in range(3))
    gate = torch.zeros(shape)
    beta = torch.zeros(shape[:3])
    with pytest.raises(ValueError, match="Mega KDA backend requires CUDA tensors"):
        chunk_kda(q, k, value, gate, beta, kernel_options={"backend": "mega"})


def test_mega_dense_and_packed_split_share_auto_scheduler(monkeypatch) -> None:
    from attn_gym.linear import chunk_kda
    from attn_gym.linear._delta_rule.mega import schedule
    from attn_gym.linear.kda.impl import mega as backend

    monkeypatch.setattr(backend, "_DENSE_LOCAL_BACKWARD_MIN_TOKENS", 1)
    monkeypatch.setattr(backend, "_PACKED_LOCAL_BACKWARD_MIN_TOKENS", 1)
    monkeypatch.setattr(backend, "_LOCAL_BACKWARD_MIN_HEADS", 1)
    local_backward = backend.chunk_mega_packed_local_bwd_op
    compute_ideal_chunks = schedule.compute_ideal_chunks
    selected = []
    geometries = []

    def record_split(*args):
        selected.append(args[-2])
        return local_backward(*args)

    def record_geometry(*args):
        geometries.append(args)
        return compute_ideal_chunks(*args)

    monkeypatch.setattr(backend, "chunk_mega_packed_local_bwd_op", record_split)
    monkeypatch.setattr(schedule, "compute_ideal_chunks", record_geometry)

    exact_inputs = _make_inputs(requires_grad=True)
    exact_output = _candidate_dense(*exact_inputs)
    torch.autograd.grad(exact_output, exact_inputs[:5], torch.randn_like(exact_output))

    dense_inputs = _make_inputs(requires_grad=True)
    dense_output = _candidate_dense_split(*dense_inputs)
    torch.autograd.grad(dense_output, dense_inputs[:5], torch.randn_like(dense_output))

    packed_inputs = _make_inputs(requires_grad=True)
    dense_cu_seqlens = cumulative_sequence_offsets([packed_inputs[0].shape[1]])
    packed_output, _ = chunk_kda(
        *packed_inputs[:5],
        cu_seqlens=dense_cu_seqlens,
        kernel_options={"backend": "mega", "split_backward": True},
    )
    torch.autograd.grad(packed_output, packed_inputs[:5], torch.randn_like(packed_output))

    assert selected == [True, True]
    assert len(geometries) == 2 and geometries[0] == geometries[1]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mega_local_backward_fullgraph(monkeypatch, dtype: torch.dtype) -> None:
    from attn_gym.linear.kda.impl import mega as backend

    monkeypatch.setattr(backend, "_DENSE_LOCAL_BACKWARD_MIN_TOKENS", 1)
    monkeypatch.setattr(backend, "_LOCAL_BACKWARD_MIN_HEADS", 1)
    expected_inputs = list(_make_inputs(requires_grad=True, dtype=dtype))
    actual_inputs = list(_make_inputs(requires_grad=True, dtype=dtype))
    dense_cu = torch.tensor([0, expected_inputs[0].shape[1]], dtype=torch.int32, device="cuda")
    expected_inputs[-1] = dense_cu
    actual_inputs[-1] = dense_cu
    expected_inputs = tuple(expected_inputs)
    actual_inputs = tuple(actual_inputs)
    expected = _candidate_no_state(*expected_inputs)
    compiled = torch.compile(_candidate_dense_split, fullgraph=True)
    actual = compiled(*actual_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    high_precision, low_precision, high_targets, low_targets = _references(
        actual_inputs,
        initial_state=False,
        packed=False,
        output_final_state=False,
    )
    d_output = torch.randn_like(actual)
    high_grads = _reference_gradients(high_precision, high_targets, (d_output,))
    low_grads = _reference_gradients(low_precision, low_targets, (d_output,))
    actual_grads = torch.autograd.grad(actual, actual_inputs[:5], d_output)
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        _assert_reference(
            actual_grad, high_grad, low_grad, f"local fullgraph gradient {index}", dtype
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mega_composed_state_only_backward(dtype: torch.dtype) -> None:
    actual_inputs = _make_inputs(requires_grad=True, dtype=dtype)
    high_precision, low_precision, high_targets, low_targets = _references(actual_inputs)
    actual = torch.compile(_candidate, fullgraph=True)(*actual_inputs)
    state_cotangent = torch.ones_like(actual[1])
    high_grads = _reference_gradients(
        high_precision, high_targets, (torch.zeros_like(actual[0]), state_cotangent)
    )
    low_grads = _reference_gradients(
        low_precision, low_targets, (torch.zeros_like(actual[0]), state_cotangent)
    )
    actual_grads = torch.autograd.grad(actual[1], actual_inputs[:-1], state_cotangent)
    for index, (actual_grad, high_grad, low_grad) in enumerate(
        zip(actual_grads, high_grads, low_grads, strict=True)
    ):
        _assert_reference(actual_grad, high_grad, low_grad, f"state-only gradient {index}", dtype)


def test_mega_dense_tail_rejected_at_public_boundary() -> None:
    from attn_gym.linear import chunk_kda

    inputs = _make_inputs(requires_grad=False)
    dense_tail = tuple(tensor[:, :65].contiguous() for tensor in inputs[:5])
    with pytest.raises(ValueError, match="T divisible by 64"):
        chunk_kda(*dense_tail, kernel_options={"backend": "mega"})


def test_mega_rejects_mismatched_value_and_beta_shapes() -> None:
    from attn_gym.linear import chunk_kda

    inputs = _make_inputs(requires_grad=False)
    with pytest.raises(ValueError, match="v must have shape"):
        chunk_kda(
            inputs[0],
            inputs[1],
            inputs[2].repeat(2, 1, 1, 1),
            *inputs[3:5],
            kernel_options={"backend": "mega"},
        )
    with pytest.raises(ValueError, match="beta must have shape"):
        chunk_kda(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4].repeat(2, 1, 1),
            kernel_options={"backend": "mega"},
        )


@pytest.mark.skipif(
    os.environ.get("ATTN_GYM_RUN_STRESS_TESTS") != "1",
    reason="set ATTN_GYM_RUN_STRESS_TESTS=1 to run repeated-launch stress tests",
)
def test_mega_repeated_backward_with_empty_sequences() -> None:
    """Empty work items must not advance the dstate handshake phase."""
    from attn_gym.linear import chunk_kda

    lengths = tuple(0 if index % 2 == 0 else 128 for index in range(29))
    q, k, value, gate, beta = make_kda_test_inputs(
        sum(lengths),
        heads=16,
        seed=421,
        normalize_qk=True,
        requires_grad=True,
    )
    state = (torch.randn(29, 16, D, D, device="cuda") / 100).requires_grad_()
    cu_seqlens = cumulative_sequence_offsets(lengths)
    d_output = torch.randn_like(value)
    d_state = torch.randn_like(state)
    for _ in range(500):
        output, final_state = chunk_kda(
            q,
            k,
            value,
            gate,
            beta,
            state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            kernel_options={"backend": "mega"},
        )
        assert final_state is not None
        torch.autograd.grad(
            (output, final_state),
            (q, k, value, gate, beta, state),
            (d_output, d_state),
        )
        torch.cuda.synchronize()


def test_mega_packed_h64_exact_tail_with_trailing_empty_is_finite() -> None:
    from attn_gym.linear import chunk_kda
    from attn_gym.linear.kda import bound_gate, l2norm

    lengths = (
        33,
        159,
        4,
        2,
        431,
        3,
        161,
        1006,
        2,
        737,
        488,
        3732,
        72,
        254,
        1106,
        2,
        0,
    )
    tokens, heads = sum(lengths), 64
    shape = (1, tokens, heads, D)
    torch.manual_seed(2090)

    def tensor(*dims: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(*dims, device="cuda") * scale).bfloat16().requires_grad_()

    q = tensor(*shape)
    k = tensor(*shape)
    value = tensor(*shape)
    gate = tensor(*shape, scale=0.5)
    beta_logits = tensor(*shape[:3])
    a_log = torch.zeros(heads, device="cuda", requires_grad=True)
    dt_bias = torch.zeros(heads, D, device="cuda", requires_grad=True)
    cu_seqlens = torch.tensor(
        (0, *accumulate(lengths)),
        device="cuda",
        dtype=torch.int32,
    )

    output, _ = chunk_kda(
        l2norm(q, cu_seqlens=cu_seqlens),
        l2norm(k, cu_seqlens=cu_seqlens),
        value,
        bound_gate(gate, a_log, dt_bias, lower_bound=-5.0, impl="fused"),
        beta_logits.float().sigmoid(),
        cu_seqlens=cu_seqlens,
        kernel_options={"backend": "mega"},
    )
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()

    output.float().sum().backward()
    for tensor in (q, k, value, gate, beta_logits, a_log, dt_bias):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_mega_packed_tail_isolated_from_next_sequence() -> None:
    from attn_gym.linear import chunk_kda

    torch.manual_seed(2111)
    tokens, heads = 17, 1
    shape = (1, tokens, heads, D)
    q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
    value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    gate = torch.full(shape, -0.5, device="cuda")
    beta = torch.full((1, tokens, heads), 0.5, device="cuda")
    k[:, 1:] = 1e20
    cu_seqlens = torch.tensor([0, 1, tokens], dtype=torch.int32, device="cuda")

    expected, _ = chunk_kda(
        q[:, :1],
        k[:, :1],
        value[:, :1],
        gate[:, :1],
        beta[:, :1],
        autotune=False,
    )
    actual, _ = chunk_kda(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens=cu_seqlens,
        kernel_options={"backend": "mega"},
    )
    torch.testing.assert_close(actual[:, :1], expected, rtol=2e-2, atol=2e-2)

    final_sequence_only = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    final_actual, _ = chunk_kda(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens=final_sequence_only,
        kernel_options={"backend": "mega"},
    )
    torch.testing.assert_close(final_actual[:, :1], expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("outer_strided", [False, True])
def test_mega_forward_op_registration(dtype: torch.dtype, outer_strided: bool) -> None:
    from attn_gym.linear.kda.impl.mega_ops import (
        chunk_mega_dense_training_fwd_op,
        chunk_mega_packed_fwd_op,
        chunk_mega_packed_fwd_with_initial_state_op,
        chunk_mega_packed_fwd_with_state_op,
        chunk_mega_packed_local_bwd_op,
        chunk_mega_packed_training_fwd_op,
        plain_gate_bwd_dense_cute_op,
    )

    inputs = _make_inputs(requires_grad=False, heads=2, dtype=dtype)
    if outer_strided:
        q, k, value, gate = (_swap_token_head_storage(tensor) for tensor in inputs[:4])
        initial_state = _swap_state_head_value_storage(inputs[5])
        inputs = (q, k, value, gate, inputs[4], initial_state, inputs[6])
    dense_cu_seqlens = cumulative_sequence_offsets([inputs[0].shape[1]])
    test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    torch.library.opcheck(
        chunk_mega_packed_fwd_op,
        (*inputs[:5], inputs[-1], False, SCALE),
        test_utils=test_utils,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.library.opcheck(
        chunk_mega_dense_training_fwd_op,
        (*inputs[:5], dense_cu_seqlens, False, SCALE),
        test_utils=test_utils,
        rtol=2e-2,
        atol=2e-2,
    )
    d_output = torch.randn_like(inputs[2])
    local_gradients = chunk_mega_packed_local_bwd_op(
        *inputs[:5], d_output, inputs[-1], False, SCALE
    )
    expected_strides = tuple(
        torch.empty_like(tensor[0]).unsqueeze(0).stride() for tensor in inputs[:5]
    )
    assert tuple(gradient.stride() for gradient in local_gradients) == expected_strides
    torch.library.opcheck(
        chunk_mega_packed_local_bwd_op,
        (*inputs[:5], d_output, inputs[-1], False, SCALE),
        test_utils=test_utils,
        rtol=2e-2,
        atol=2e-2,
    )
    backward_metadata = prepare_ragged_chunk_metadata(inputs[-1], inputs[0].shape[1], 64)
    torch.library.opcheck(
        chunk_mega_packed_training_fwd_op,
        (
            *inputs[:5],
            backward_metadata.cu_seqlens,
            backward_metadata.chunk_offsets,
            False,
            SCALE,
        ),
        test_utils=test_utils,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.library.opcheck(
        chunk_mega_packed_fwd_with_initial_state_op,
        (*inputs, SCALE),
        test_utils=test_utils,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.library.opcheck(
        chunk_mega_packed_fwd_with_state_op,
        (*inputs, SCALE),
        test_utils=test_utils,
        rtol=2e-2,
        atol=2e-2,
    )
    torch.library.opcheck(
        plain_gate_bwd_dense_cute_op,
        (torch.randn(1, 128, 1, D, device="cuda"),),
        test_utils=test_utils,
        rtol=2e-2,
        atol=2e-2,
    )


def test_mega_initial_state_forward_op_fullgraph() -> None:
    from attn_gym.linear.kda.impl.mega_ops import chunk_mega_packed_fwd_with_initial_state_op

    inputs = _make_inputs(requires_grad=False)
    args = (*inputs, SCALE)
    expected = chunk_mega_packed_fwd_with_initial_state_op(*args)
    actual = torch.compile(
        chunk_mega_packed_fwd_with_initial_state_op,
        fullgraph=True,
    )(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("candidate", [_candidate_dense, _candidate_no_state])
def test_mega_multistream_fullgraph_cuda_graph_replay(candidate) -> None:
    inputs = _make_inputs(requires_grad=False)
    compiled = torch.compile(candidate, fullgraph=True)
    expected = compiled(*inputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


def test_mega_tma_validation_routes_oversized_singleton_stride_to_int64() -> None:
    from attn_gym._backends.cute.utils import requires_int64_abi
    from attn_gym.linear._delta_rule.mega.kernels.compat import validate_tma_tensor

    tensor = torch.empty_strided(
        (1, 1, D),
        (D, 2**31 + 8, 1),
        dtype=torch.bfloat16,
        device="cuda",
    )
    validate_tma_tensor("tensor", tensor)
    assert requires_int64_abi(tensor)


def test_mega_forced_int64_forward_backward_matches_int32(monkeypatch) -> None:
    from attn_gym.linear._delta_rule.mega.kernels import (
        kda_bprop_f16,
        kda_prefill_f16,
        kda_recompute_f16,
    )
    from attn_gym.linear.kda.impl import mega as backend

    monkeypatch.setattr(backend, "_PACKED_LOCAL_BACKWARD_MIN_TOKENS", 1)
    monkeypatch.setattr(backend, "_LOCAL_BACKWARD_MIN_HEADS", 1)
    expected_inputs = _make_inputs(requires_grad=True, dtype=torch.float16)
    actual_inputs = _make_inputs(requires_grad=True, dtype=torch.float16)
    d_output = torch.randn_like(expected_inputs[2])

    expected = _candidate_no_state(*expected_inputs)
    expected_grads = torch.autograd.grad(expected, expected_inputs[:5], d_output)
    for module in (kda_prefill_f16, kda_recompute_f16, kda_bprop_f16):
        monkeypatch.setattr(module, "requires_int64_abi", lambda *_: True)
    actual = _candidate_no_state(*actual_inputs)
    actual_grads = torch.autograd.grad(actual, actual_inputs[:5], d_output)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_mega_plain_gate_backward_forced_int64_matches_int32(monkeypatch) -> None:
    from attn_gym.linear._delta_rule.mega.kernels import kda_plain_gate_bwd

    d_cumulative = torch.randn(1, 64, 1, D, device="cuda")
    expected = kda_plain_gate_bwd.plain_gate_cumsum_dense_bwd_cute(d_cumulative)
    monkeypatch.setattr(kda_plain_gate_bwd, "requires_int64_abi", lambda *_: True)
    actual = kda_plain_gate_bwd.plain_gate_cumsum_dense_bwd_cute(d_cumulative)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mega_fullgraph_cuda_graph_replay() -> None:
    inputs = _make_inputs(requires_grad=False)
    compiled = torch.compile(_candidate, fullgraph=True)
    expected = compiled(*inputs)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(*inputs)
    graph.replay()
    torch.cuda.synchronize()
    for captured_tensor, expected_tensor in zip(captured, expected, strict=True):
        torch.testing.assert_close(captured_tensor, expected_tensor, rtol=0, atol=0)
