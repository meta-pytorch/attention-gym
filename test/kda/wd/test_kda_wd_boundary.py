"""Public-boundary parity of warpdrive chunk KDA with the fused backend.

Dense batches, dtypes, and layouts that the fused backend accepts must also run on warpdrive.
Each case checks one exact oracle (the same warpdrive call on canonical operands: packed rows,
the fused dtype policy's casts, compact layouts) and stays within the plan tolerance of the
fused backend.
"""

from __future__ import annotations

import math

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym._backends.cute import tensor_supports_contiguous_dim, tensor_supports_tma
from attn_gym.linear import chunk_kda, paged_chunk_kda
from attn_gym.testing.kda import (
    assert_matches_fused,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA path requires an SM100-family GPU",
)

WARPDRIVE = {"backend": "warpdrive"}
FUSED = {"backend": "fused", "autotune": False}
HEADS = 2


def run(inputs, kernel_options, cotangent_seed=5, **kwargs):
    """Return the output, final state, and gradients of every input."""
    output, final_state = chunk_kda(
        *inputs, output_final_state=True, kernel_options=kernel_options, **kwargs
    )
    generator = torch.Generator(device="cuda").manual_seed(cotangent_seed)
    cotangents = tuple(
        torch.randn(t.shape, device="cuda", generator=generator).to(t.dtype)
        for t in (output, final_state)
    )
    return (output, final_state, *torch.autograd.grad((output, final_state), inputs, cotangents))


def make_inputs(tokens: int, *, batch: int = 1, sequences: int | None = None, seed: int = 11):
    """KDA operands and a small FP32 entry state, one per dense row or packed sequence.

    Gates stay in ``[-ln 2, 0]``: warpdrive FP16 does not support model-range gates.
    """
    q, k, value, gate, beta = make_kda_test_inputs(
        tokens,
        batch=batch,
        heads=HEADS,
        seed=seed,
        gate_scale=math.log(2.0),
        normalize_qk=True,
        sigmoid_beta=True,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    state = torch.randn(sequences or batch, HEADS, 128, 128, device="cuda", generator=generator)
    return q, k, value, gate, beta, state / 100


def test_dense_batch_matches_packed_rows_and_fused() -> None:
    batch, tokens = 3, 100
    dense = make_inputs(tokens, batch=batch)
    actual = run(clone_kda_inputs(dense, requires_grad=True), WARPDRIVE)

    packed = tuple(t.reshape(1, batch * tokens, *t.shape[2:]) for t in dense[:5])
    offsets = cumulative_sequence_offsets([tokens] * batch)
    expected = run(
        clone_kda_inputs((*packed, dense[5]), requires_grad=True), WARPDRIVE, cu_seqlens=offsets
    )
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(got, want.reshape(got.shape), rtol=0, atol=0, msg=str(index))

    fused = run(clone_kda_inputs(dense, requires_grad=True), FUSED)
    for index, (got, want) in enumerate(zip(actual, fused, strict=True)):
        assert_matches_fused(got, want, f"dense batch result {index}")


def _cast(tensors, dtypes):
    return tuple(t.to(dtype) for t, dtype in zip(tensors, dtypes, strict=True))


_BF16, _FP16, _FP32 = torch.bfloat16, torch.float16, torch.float32


@pytest.mark.parametrize(
    "caller_dtypes",
    [
        pytest.param((_FP32, _FP32, _FP32, _FP32, _FP32, _FP32), id="fp32-qkv"),
        pytest.param((_BF16, _FP32, _FP16, _FP32, _FP32, _FP32), id="mixed-qkv"),
        pytest.param((_BF16, _BF16, _BF16, _BF16, _BF16, _FP32), id="bf16-gate-beta"),
        pytest.param((_FP16, _FP16, _FP16, _FP16, _FP16, _FP32), id="fp16-gate-beta"),
        pytest.param((_BF16, _BF16, _BF16, _FP32, _FP32, _BF16), id="bf16-state"),
        pytest.param((_FP16, _FP16, _FP16, _FP32, _FP32, _FP16), id="fp16-state"),
    ],
)
def test_dtype_policy_matches_cast_operands_and_fused(caller_dtypes) -> None:
    """Q/K/V that share FP16/BF16 run as given, others in BF16; gate, beta, and state in FP32.
    The output keeps ``q.dtype`` and every gradient its operand's dtype."""
    lengths = (70, 0, 58)
    base = make_inputs(sum(lengths), sequences=len(lengths))
    offsets = cumulative_sequence_offsets(lengths)
    caller = clone_kda_inputs(_cast(base, caller_dtypes), requires_grad=True)
    qkv_dtype = (
        caller_dtypes[0]
        if caller_dtypes[0] in (_FP16, _BF16) and len(set(caller_dtypes[:3])) == 1
        else _BF16
    )
    canonical = clone_kda_inputs(
        _cast(caller, (qkv_dtype,) * 3 + (_FP32,) * 3), requires_grad=True
    )

    actual = run(caller, WARPDRIVE, cu_seqlens=offsets)
    expected = run(canonical, WARPDRIVE, cu_seqlens=offsets)
    assert actual[0].dtype == caller_dtypes[0]
    assert actual[1].dtype == _FP32
    torch.testing.assert_close(actual[0], expected[0].to(caller_dtypes[0]), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    for index, (got, want, dtype) in enumerate(
        zip(actual[2:], expected[2:], caller_dtypes, strict=True)
    ):
        assert got.dtype == dtype, f"gradient {index}"
        torch.testing.assert_close(got, want.to(dtype), rtol=0, atol=0, msg=f"gradient {index}")

    fused = run(clone_kda_inputs(caller, requires_grad=True), FUSED, cu_seqlens=offsets)
    for index, (got, want) in enumerate(zip(actual, fused, strict=True)):
        assert_matches_fused(got, want, f"{caller_dtypes} result {index}")


def _misaligned(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into storage offset by one element, so the base breaks every vector alignment."""
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    return storage[1:].view_as(tensor).copy_(tensor)


def _strided_inner(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into storage whose innermost mode has stride two."""
    shape = (*tensor.shape[:-1], 2 * tensor.shape[-1])
    storage = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    return storage[..., ::2].copy_(tensor)


@pytest.mark.parametrize("stateful", [False, True], ids=["no-state", "state"])
def test_unaligned_layouts_match_compact_operands(stateful: bool) -> None:
    """Operands the launchers cannot address are copied rather than rejected."""
    lengths = (70, 0, 58)
    compact = make_inputs(sum(lengths), sequences=len(lengths))
    offsets = cumulative_sequence_offsets(lengths)
    offset_storage = torch.empty(offsets.numel() + 1, dtype=offsets.dtype, device="cuda")
    misaligned_offsets = offset_storage[1:].copy_(offsets)
    assert misaligned_offsets.data_ptr() % 8
    q, k, value, gate, beta, state = compact
    unaligned = (
        _misaligned(q),
        _strided_inner(k),
        _misaligned(value),
        _strided_inner(gate),
        _strided_inner(beta),
        _misaligned(state),
    )
    if not stateful:
        compact, unaligned = compact[:5], unaligned[:5]

    def call(inputs, cu_seqlens):
        output, final_state = chunk_kda(
            *inputs,
            cu_seqlens=cu_seqlens,
            output_final_state=stateful,
            kernel_options=WARPDRIVE,
        )
        results = (output,) if final_state is None else (output, final_state)
        cotangents = tuple(torch.ones_like(result) for result in results)
        return (*results, *torch.autograd.grad(results, inputs, cotangents))

    assert not any(tensor_supports_tma(t) for t in (*unaligned[:4], *unaligned[5:]))
    assert not tensor_supports_contiguous_dim(unaligned[4], alignment_bytes=4)
    # The unaligned copies are already independent leaves; cloning would compact them.
    actual = call(tuple(t.requires_grad_() for t in unaligned), misaligned_offsets)
    expected = call(clone_kda_inputs(compact, requires_grad=True), offsets)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=str(index))


@pytest.mark.parametrize("layout", ["strided-key", "transposed"])
def test_noncontiguous_entry_state_trains_like_compact_state(layout: str) -> None:
    compact = make_inputs(65)
    state = (
        _strided_inner(compact[-1]) if layout == "strided-key" else compact[-1].transpose(-1, -2)
    ).requires_grad_()
    assert state.dtype == torch.float32 and state.stride(-1) != 1
    caller = (*clone_kda_inputs(compact[:5], requires_grad=True), state)
    canonical = clone_kda_inputs((*caller[:5], state.contiguous()), requires_grad=True)

    actual = run(caller, WARPDRIVE)
    expected = run(canonical, WARPDRIVE)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=str(index))

    fused = run(clone_kda_inputs(canonical, requires_grad=True), FUSED)
    for index, (got, want) in enumerate(zip(actual, fused, strict=True)):
        assert_matches_fused(got, want, f"{layout} result {index}")


def _feature_major(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into dense storage whose last mode is outermost, so it needs a normalizing copy."""
    order = (0, tensor.ndim - 1, *range(1, tensor.ndim - 1))
    inverse = tuple(order.index(dim) for dim in range(tensor.ndim))
    return tensor.permute(order).contiguous().permute(inverse)


def test_normalized_operands_return_the_fake_layouts() -> None:
    """Results computed from copied operands get back the strides the fake kernels declare."""
    from attn_gym.linear.kda.impl.warpdrive_ops import (
        chunk_warpdrive_packed_fwd_op,
        chunk_warpdrive_packed_fwd_with_state_op,
        chunk_warpdrive_packed_local_bwd_op,
    )

    lengths = (70, 0, 58)
    *tensors, state = make_inputs(sum(lengths), sequences=len(lengths))
    tensors = tuple(_feature_major(t) for t in tensors)
    state = _feature_major(state)
    assert not any(tensor_supports_tma(t) for t in (*tensors[:4], state))
    offsets = cumulative_sequence_offsets(lengths)
    scale = 128**-0.5
    test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    for op, args in (
        (chunk_warpdrive_packed_fwd_op, (*tensors, offsets, False, scale)),
        (chunk_warpdrive_packed_fwd_with_state_op, (*tensors, state, offsets, scale)),
        (
            chunk_warpdrive_packed_local_bwd_op,
            (*tensors, _feature_major(torch.randn_like(tensors[2])), offsets, False, scale),
        ),
    ):
        torch.library.opcheck(op, args, test_utils=test_utils, rtol=2e-2, atol=2e-2)


def test_layout_restore_allocates_only_to_copy() -> None:
    """A result already in the fake layout comes back as is, with no extra allocation; one in
    another layout is copied into the declared strides."""
    from attn_gym.linear.kda.impl.warpdrive_ops import _restore_layout

    value = make_inputs(128)[2]
    misaligned = _misaligned(value)  # contiguous, so the compact copy's result already matches
    compact = misaligned.clone()
    result = torch.randn_like(compact)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    assert _restore_layout(result, compact, misaligned) is result
    assert torch.cuda.max_memory_allocated() == before  # not even a transient buffer

    permuted = _feature_major(value)
    restored = _restore_layout(result, permuted.contiguous(), permuted)
    assert restored.stride() == torch.empty_like(permuted[0]).unsqueeze(0).stride()
    torch.testing.assert_close(restored, result, rtol=0, atol=0)


def test_aligned_operands_reach_the_launcher_without_copies(monkeypatch) -> None:
    from attn_gym.linear._delta_rule.warpdrive.launchers import forward

    q, k, value, gate, beta, _ = make_inputs(128)
    offsets = cumulative_sequence_offsets([64, 64])
    launched = []
    run_forward = forward.run_forward

    def recording(*args, **kwargs):
        launched.append(tuple(tensor.data_ptr() for tensor in args[:6]))
        return run_forward(*args, **kwargs)

    monkeypatch.setattr(forward, "run_forward", recording)
    chunk_kda(q, k, value, gate, beta, cu_seqlens=offsets, kernel_options=WARPDRIVE)
    assert launched == [tuple(t.data_ptr() for t in (q, k, value, gate, beta, offsets))]


def test_paged_dense_batch_and_dtypes_match_canonical_call() -> None:
    """The paged forward applies the same batch packing and dtype policy as chunk_kda."""
    batch, tokens = 2, 80
    q, k, value, gate, beta, state = make_inputs(tokens, batch=batch)
    pool = torch.zeros(batch + 1, HEADS, 128, 128, device="cuda")
    pool[1:] = state
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    caller = (q.float(), k.float(), value.float(), gate.bfloat16(), beta.bfloat16())
    canonical = (
        *(t.reshape(1, batch * tokens, *t.shape[2:]) for t in (q, k, value)),
        *(t.float().reshape(1, batch * tokens, *t.shape[2:]) for t in caller[3:]),
    )
    with torch.no_grad():
        actual_pool = pool.clone()
        actual = paged_chunk_kda(*caller, actual_pool, indices, kernel_options=WARPDRIVE)
        expected_pool = pool.clone()
        expected = paged_chunk_kda(
            *canonical,
            expected_pool,
            indices,
            cu_seqlens=cumulative_sequence_offsets([tokens] * batch),
            kernel_options=WARPDRIVE,
        )
        fused_pool = pool.clone()
        fused = paged_chunk_kda(*caller, fused_pool, indices, kernel_options=FUSED)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float().reshape(actual.shape), rtol=0, atol=0)
    torch.testing.assert_close(actual_pool, expected_pool, rtol=0, atol=0)
    assert_matches_fused(actual, fused, "paged output")
    assert_matches_fused(actual_pool, fused_pool, "paged pool")
