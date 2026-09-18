"""Backend cache and training regressions for runtime packed token/sequence counts."""

from __future__ import annotations

import itertools

import pytest
import torch

pytest.importorskip("triton")

from attn_gym.linear import chunk_gdn, chunk_kda
from attn_gym.linear._delta_rule.triton.chunk_scheduler import (
    ScheduleRequest,
    _prepare_ragged_chunk_offsets_kernel,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.gdn.fwd.triton.chunk_gdn_fwd_recurrence import (
    chunk_gdn_fwd_recurrence_packed,
)
from attn_gym.linear.kda.fwd.triton import chunk_delta_h
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires CUDA capability 8.0 or newer",
)


def _cache(kernel):
    # This is Triton's backend kernel cache, not Dynamo's graph cache. Clearing it
    # makes the assertion independent of test order (disk cache hits still populate it).
    return kernel.device_caches[torch.cuda.current_device()][0]


def _expected_offsets(lengths: list[int]) -> torch.Tensor:
    counts = [(length + 63) // 64 for length in lengths]
    return torch.tensor([0, *itertools.accumulate(counts)], dtype=torch.int32, device="cuda")


def test_metadata_reuses_backend_cache_across_independent_t_and_n():
    cache = _cache(_prepare_ragged_chunk_offsets_kernel)
    cache.clear()
    # N=3 and N=4 share the necessary next-power-of-two vector class. T also
    # crosses a divisibility boundary, independently of N and device contents.
    for tokens, lengths in ((128, [65, 0, 31]), (257, [1, 64, 0]), (257, [0, 65, 1, 0])):
        metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
        torch.testing.assert_close(metadata.chunk_offsets, _expected_offsets(lengths))
        assert len(cache) == 1, "exact T/N created another metadata backend specialization"

    lengths = [0, 65, 1, 0, 1]
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), 257, 64)
    torch.testing.assert_close(metadata.chunk_offsets, _expected_offsets(lengths))
    assert len(cache) == 2, "a different vector class must remain separately compiled"


def _inputs(tokens: int, sequences: int, scalar_gate: bool, dtype: torch.dtype):
    torch.manual_seed(113)
    shape = (1, tokens, 8, 128)
    k, w, u = [torch.randn(shape, device="cuda", dtype=dtype) / 32 for _ in range(3)]
    gate = -torch.rand(shape[:3] if scalar_gate else shape, device="cuda")
    state = torch.randn(sequences, 8, 128, 128, device="cuda") / 32
    return k, w, u, gate, state


def _recurrence(inputs, metadata, scalar_gate: bool):
    if scalar_gate:
        return chunk_gdn_fwd_recurrence_packed(*inputs, metadata)
    return chunk_gated_delta_rule_fwd_h(*inputs, metadata=metadata)


def _assert_persistent(inputs, metadata) -> None:
    workers = chunk_delta_h._persistent_sequence_workers(
        metadata,
        inputs[0].shape[1],
        heads=8,
        value_tiles=4 if inputs[0].dtype == torch.float32 else 2,
        device=inputs[0].device,
        schedule=ScheduleRequest.AUTO,
    )
    assert 0 < workers < (metadata.cu_seqlens.numel() - 1) * 8


@pytest.mark.parametrize("scalar_gate", [False, True], ids=["kda", "gdn"])
def test_persistent_reuses_backend_cache_across_independent_t_and_n(scalar_gate):
    kernels = (
        chunk_delta_h.chunk_delta_h_kernel_k128_wsp,
        chunk_delta_h.chunk_delta_h_kernel_k128_persistent,
    )
    for kernel in kernels:
        _cache(kernel).clear()
    for tokens, sequences in ((128, 64), (257, 64), (257, 63)):
        inputs = _inputs(tokens, sequences, scalar_gate, torch.bfloat16)
        lengths = [2] * (sequences - 1) + [0]
        metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
        _assert_persistent(inputs, metadata)
        _recurrence(inputs, metadata, scalar_gate)
        torch.cuda.synchronize()
        for kernel in kernels:
            assert len(_cache(kernel)) == 1, (
                f"exact T/N created another {kernel.__name__} backend specialization"
            )


def _recurrence_reference(inputs, lengths: list[int], dtype: torch.dtype):
    k, w, u, gate, initial = inputs
    k, w, u, gate = (value.to(dtype) for value in (k, w, u, gate))
    values, states, chunks = [], [], []
    begin = 0
    for sequence, length in enumerate(lengths):
        state = initial[sequence].to(dtype).transpose(-1, -2)
        for offset in range(0, length, 64):
            end = begin + min(offset + 64, length)
            token = slice(begin + offset, end)
            chunks.append(state)
            delta = u[0, token].transpose(0, 1) - w[0, token].transpose(0, 1) @ state
            values.append(delta.transpose(0, 1))
            decay = gate[0, end - 1].exp2()
            decay = decay[:, None, None] if gate.ndim == 3 else decay[:, :, None]
            state = state * decay + k[0, token].permute(1, 2, 0) @ delta
        states.append(state.transpose(-1, -2))
        begin += length
    heads, dim = k.shape[2:]
    h = torch.stack(chunks)[None] if chunks else k.new_empty(1, 0, heads, dim, dim)
    v = torch.cat(values)[None] if values else u[:, :0]
    return h, v, torch.stack(states)


def _tf32_recurrence_error_bounds(inputs, lengths: list[int], reference):
    """Propagate roundoff around the already-computed FP64 states and deltas."""
    k, w, u, gate, initial = (value.double() for value in inputs)
    history, deltas, _ = reference
    states = iter(history[0])
    eps = torch.finfo(torch.float32).eps
    # TF32 retains ten fraction bits. Use a full ulp to cover truncation as well
    # as nearest rounding, and gamma_128 for the largest FP32 dot reduction.
    operand_eps = 2.0**-10
    dot_error = 2 * operand_eps + operand_eps**2 + 128 * eps / (1 - 128 * eps)
    chunk_errors, value_errors, state_errors = [], [], []
    begin = 0
    for sequence, length in enumerate(lengths):
        state_error = torch.zeros_like(initial[sequence].transpose(-1, -2))
        for offset in range(0, length, 64):
            end = begin + min(offset + 64, length)
            token = slice(begin + offset, end)
            kw = w[0, token].transpose(0, 1)
            kk = k[0, token].permute(1, 2, 0)
            uu = u[0, token].transpose(0, 1)
            chunk_errors.append(state_error)
            state_bound = next(states).abs() + state_error
            projected_bound = kw.abs() @ state_bound
            delta = deltas[0, token].transpose(0, 1)
            delta_error = kw.abs() @ state_error + dot_error * projected_bound
            delta_error += eps * (uu.abs() + projected_bound)
            value_errors.append(delta_error.transpose(0, 1))
            decay = gate[0, end - 1].exp2()
            decay = decay[:, None, None] if gate.ndim == 3 else decay[:, :, None]
            update_bound = kk.abs() @ (delta.abs() + delta_error)
            # Four eps for exp2, one for decay multiplication, one for the
            # final add; the dot uses its own operand/reduction error above.
            state_error = (
                decay * state_error
                + 6 * eps * decay * state_bound
                + kk.abs() @ delta_error
                + (dot_error + eps) * update_bound
            )
        state_errors.append(state_error.transpose(-1, -2))
        begin += length
    h = torch.stack(chunk_errors)[None] if chunk_errors else torch.empty_like(history)
    v = torch.cat(value_errors)[None] if value_errors else torch.empty_like(deltas)
    return h, v, torch.stack(state_errors)


@pytest.mark.parametrize("scalar_gate", [False, True], ids=["kda", "gdn"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_persistent_graph_updates_boundaries_and_empty_states(scalar_gate, dtype):
    inputs = _inputs(257, 64, scalar_gate, dtype)
    boundaries = cumulative_sequence_offsets([4] * 64)

    def operation():
        metadata = prepare_ragged_chunk_metadata(boundaries, 257, 64)
        return metadata.chunk_offsets, _recurrence(inputs, metadata, scalar_gate)

    metadata = prepare_ragged_chunk_metadata(boundaries, 257, 64)
    _assert_persistent(inputs, metadata)
    operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        offsets, actual = operation()

    # Includes a full descriptor chunk, a partial tail, interior empties, active
    # work beyond the primary wave, trailing empties, and the all-empty replay.
    for lengths in ([65, 0, 3] + [2] * 29 + [0] * 32, [0] * 64):
        boundaries.copy_(cumulative_sequence_offsets(lengths))
        active = sum(lengths)
        for value in inputs[:4]:
            value[:, active:] = float("nan")
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(offsets, _expected_offsets(lengths))
        high = _recurrence_reference(inputs, lengths, torch.float64)
        result = (actual[0][:, : high[0].shape[1]], actual[1][:, :active], actual[2])
        if dtype == torch.float32:
            bounds = _tf32_recurrence_error_bounds(inputs, lengths, high)
            for value, fp64, bound in zip(result, high, bounds, strict=True):
                assert torch.isfinite(value).all()
                assert torch.all((value.double() - fp64).abs() <= 2 * bound)
        else:
            low = _recurrence_reference(inputs, lengths, torch.float32)
            for name, value, fp64, eager in zip(
                ("h", "v_new", "state"), result, high, low, strict=True
            ):
                if value.numel():
                    assert_matches_low_precision_reference(
                        value, fp64, eager, name, source_dtype=dtype
                    )
        # TF32 has FP16's ten fraction bits, despite FP32 input/output storage.
        precision = torch.float16 if dtype == torch.float32 else dtype
        for name, value, fp64 in zip(("h", "v_new", "state"), result, high, strict=True):
            if value.numel():
                assert_relative_rms_within(value, fp64, name, max_eps=1.25, source_dtype=precision)
        empty = torch.tensor([length == 0 for length in lengths], device="cuda")
        torch.testing.assert_close(actual[2][empty], inputs[-1][empty], rtol=0, atol=0)


def _training_inputs(tokens: int, sequences: int, scalar_gate: bool):
    inputs = make_kda_test_inputs(
        tokens, heads=8, normalize_qk=True, gate_scale=0.5, requires_grad=True
    )
    if scalar_gate:
        inputs = (*inputs[:3], inputs[3][..., 0].detach().contiguous().requires_grad_(), inputs[4])
    state = (torch.randn(sequences, 8, 128, 128, device="cuda") / 32).requires_grad_()
    return *inputs, state


@pytest.mark.parametrize("scalar_gate", [False, True], ids=["kda", "gdn"])
@pytest.mark.parametrize("execution", ["compiled", "graph"])
def test_public_training_persistent_recompute_gradients(
    scalar_gate, execution, monkeypatch, fresh_compile_cache
):
    torch.manual_seed(19)
    inputs = _training_inputs(256, 64, scalar_gate)
    boundaries = cumulative_sequence_offsets([4] * 64)
    op = chunk_gdn if scalar_gate else chunk_kda
    options = {} if scalar_gate else {"autotune": False}
    selected_workers = []
    original = chunk_delta_h._persistent_sequence_workers

    def record_workers(*args, **kwargs):
        workers = original(*args, **kwargs)
        selected_workers.append(workers)
        return workers

    monkeypatch.setattr(chunk_delta_h, "_persistent_sequence_workers", record_workers)

    def operation(q, k, v, gate, beta, state, offsets):
        return op(
            q, k, v, gate, beta, state, cu_seqlens=offsets, output_final_state=True, **options
        )

    function = (
        torch.compile(operation, fullgraph=True, dynamic=True)
        if execution == "compiled"
        else operation
    )
    grad_output = torch.randn_like(inputs[2])
    grad_state = torch.randn_like(inputs[-1])

    def training():
        output, final = function(*inputs, boundaries)
        gradients = torch.autograd.grad((output, final), inputs, (grad_output, grad_state))
        return output, final, *gradients

    training()
    assert len(selected_workers) >= 2 and all(worker > 0 for worker in selected_workers), (
        "public forward and backward recomputation must both select persistent recurrence"
    )
    torch.cuda.synchronize()
    if execution == "graph":
        previous_override = torch._C._override_stale_capture_stream()
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = training()
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(previous_override)

    cases = [(256, [4] * 64), (256, [0, 5, 3] + [2] * 29 + [0] * 32)]
    if execution == "compiled":
        cases.extend(((321, [4] * 64), (321, [4] * 63)))
    for tokens, lengths in cases:
        if tokens != inputs[0].shape[1] or len(lengths) != inputs[-1].shape[0]:
            inputs = _training_inputs(tokens, len(lengths), scalar_gate)
            boundaries = cumulative_sequence_offsets(lengths)
            grad_output = torch.randn_like(inputs[2])
            grad_state = torch.randn_like(inputs[-1])
        else:
            boundaries.copy_(cumulative_sequence_offsets(lengths))
        active = sum(lengths)
        with torch.no_grad():
            inputs[-1].mul_(0.75)
            for value in inputs[:5]:
                value[:, active:] = float("nan")
            grad_output[:, active:] = 0
        if execution == "graph":
            graph.replay()
            torch.cuda.synchronize()
            actual = captured
        else:
            actual = training()
        references = []
        for dtype in (None, torch.float64):
            ref_inputs = clone_kda_inputs(inputs, dtype=dtype)
            q, k, v, gate, beta, state = ref_inputs
            if scalar_gate:
                gate = gate[..., None].expand_as(k)
            out, final = kda_reference(q, k, v, gate, beta, state, cu_seqlens=boundaries)
            grads = torch.autograd.grad(
                (out, final), ref_inputs, (grad_output.to(out.dtype), grad_state.to(final.dtype))
            )
            references.append((out, final, *grads))
        # Inactive output and token-gradient storage is unspecified by these APIs.
        for name, value, low, high in zip(
            ("output", "state", "dq", "dk", "dv", "dg", "db", "dh0"),
            actual,
            *references,
            strict=True,
        ):
            if name not in ("state", "dh0"):
                value, low, high = (tensor[:, :active] for tensor in (value, low, high))
            assert_matches_low_precision_reference(value, high, low, name)
            assert_relative_rms_within(value, high, name, max_eps=1.25)
