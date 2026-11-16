"""Public-boundary parity of warpdrive chunk GDN with the fused backend.

Entry-state dtypes, dense batches, and layouts that the fused backend accepts must also run on
warpdrive. Each case checks one exact oracle (the same warpdrive call on canonical operands)
and stays within the plan tolerance of the fused backend.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym._backends.cute import tensor_supports_contiguous_dim, tensor_supports_tma
from attn_gym.linear import chunk_gdn
from attn_gym.testing import make_gdn_test_inputs
from attn_gym.testing.kda import assert_matches_fused, clone_kda_inputs

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive GDN path requires an SM100-family GPU",
)

WARPDRIVE = {"backend": "warpdrive"}
FUSED = {"backend": "fused", "autotune": False}


def run(inputs, kernel_options, **kwargs):
    """Return the output, final state, and gradients of every input."""
    output, final_state = chunk_gdn(
        *inputs, output_final_state=True, kernel_options=kernel_options, **kwargs
    )
    generator = torch.Generator(device="cuda").manual_seed(5)
    cotangents = tuple(
        torch.randn(t.shape, device="cuda", generator=generator).to(t.dtype)
        for t in (output, final_state)
    )
    return (output, final_state, *torch.autograd.grad((output, final_state), inputs, cotangents))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_bf16_initial_state_matches_fp32_state_and_fused(dtype: torch.dtype) -> None:
    """A BF16 entry state runs in FP32 and its cotangent comes back in BF16."""
    *operands, offsets = make_gdn_test_inputs((70, 0, 58), key_heads=1, value_heads=2, dtype=dtype)
    caller = clone_kda_inputs((*operands[:5], operands[5].bfloat16()), requires_grad=True)
    canonical = clone_kda_inputs((*caller[:5], caller[5].float()), requires_grad=True)

    actual = run(caller, WARPDRIVE, cu_seqlens=offsets)
    expected = run(canonical, WARPDRIVE, cu_seqlens=offsets)
    assert actual[-1].dtype == torch.bfloat16
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(got, want.to(got.dtype), rtol=0, atol=0, msg=str(index))

    fused = run(clone_kda_inputs(caller, requires_grad=True), FUSED, cu_seqlens=offsets)
    assert fused[-1].dtype == torch.bfloat16
    for index, (got, want) in enumerate(zip(actual, fused, strict=True)):
        assert_matches_fused(got, want, f"result {index}")


def test_dense_batch_with_state_matches_packed_rows_and_fused() -> None:
    batch, tokens = 3, 100
    dense = make_gdn_test_inputs(tokens, batch=batch, key_heads=1, value_heads=2)[:6]
    actual = run(clone_kda_inputs(dense, requires_grad=True), WARPDRIVE)

    packed = tuple(t.reshape(1, batch * tokens, *t.shape[2:]) for t in dense[:5])
    offsets = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * tokens
    expected = run(
        clone_kda_inputs((*packed, dense[5]), requires_grad=True), WARPDRIVE, cu_seqlens=offsets
    )
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(got, want.reshape(got.shape), rtol=0, atol=0, msg=str(index))

    fused = run(clone_kda_inputs(dense, requires_grad=True), FUSED)
    for index, (got, want) in enumerate(zip(actual, fused, strict=True)):
        assert_matches_fused(got, want, f"dense batch result {index}")


def _misaligned(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into storage offset by one element, so the base breaks every vector alignment."""
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    return storage[1:].view_as(tensor).copy_(tensor)


def _strided_inner(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into storage whose innermost mode has stride two."""
    shape = (*tensor.shape[:-1], 2 * tensor.shape[-1])
    storage = torch.empty(shape, dtype=tensor.dtype, device=tensor.device)
    return storage[..., ::2].copy_(tensor)


def test_unaligned_layouts_match_compact_operands() -> None:
    *compact, offsets = make_gdn_test_inputs((70, 0, 58), key_heads=1, value_heads=2)
    offset_storage = torch.empty(offsets.numel() + 1, dtype=offsets.dtype, device="cuda")
    misaligned_offsets = offset_storage[1:].copy_(offsets)
    q, k, value, gate, beta, state = compact
    unaligned = (
        _misaligned(q),
        _strided_inner(k),
        _misaligned(value),
        _strided_inner(gate),
        _strided_inner(beta),
        _misaligned(state),
    )
    assert not any(tensor_supports_tma(t) for t in (*unaligned[:3], unaligned[5]))
    assert not any(tensor_supports_contiguous_dim(t, alignment_bytes=4) for t in unaligned[3:5])
    # The unaligned copies are already independent leaves; cloning would compact them.
    actual = run(
        tuple(t.requires_grad_() for t in unaligned), WARPDRIVE, cu_seqlens=misaligned_offsets
    )
    expected = run(clone_kda_inputs(compact, requires_grad=True), WARPDRIVE, cu_seqlens=offsets)
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        torch.testing.assert_close(got, want, rtol=0, atol=0, msg=str(index))
