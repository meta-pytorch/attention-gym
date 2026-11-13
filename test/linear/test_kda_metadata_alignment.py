"""Packed KDA metadata may be a contiguous slice of a larger allocation."""

import pytest
import torch

from attn_gym.linear import chunk_kda
from attn_gym.testing.kda import clone_kda_inputs, make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the fused KDA core requires CUDA capability 8.0 or newer",
)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize(
    ("backend", "byte_offset"), [("fused", 4), ("fused", 16), ("warpdrive", 16)]
)
def test_stateful_backward_accepts_sliced_cu_seqlens(
    backend: str, byte_offset: int, compiled: bool
):
    if backend == "warpdrive":
        pytest.importorskip("cutlass.experimental")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("the cuDNN backend requires SM100/SM103")

    inputs = make_kda_test_inputs(100, heads=2, seed=0, requires_grad=True)
    state = (torch.randn(2, 2, 128, 128, device="cuda") / 8).requires_grad_()
    offsets = torch.tensor([0, 60, 100], dtype=torch.int32, device="cuda")
    storage = torch.empty(64, dtype=torch.int32, device="cuda")
    shift = ((byte_offset - storage.data_ptr()) % 128) // storage.element_size()
    sliced_offsets = storage[shift : shift + offsets.numel()]
    sliced_offsets.copy_(offsets)
    assert sliced_offsets.is_contiguous() and sliced_offsets.data_ptr() % 128 == byte_offset

    operation = torch.compile(chunk_kda, fullgraph=True, dynamic=True) if compiled else chunk_kda

    def run(values, initial_state, cu_seqlens):
        output, final_state = operation(
            *values,
            initial_state,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
            kernel_options={"backend": backend, "autotune": False},
        )
        gradients = torch.autograd.grad(
            output.float().sum() + final_state.sum(), (*values, initial_state)
        )
        return output, final_state, *gradients

    expected = run(inputs, state, offsets)
    actual = run(clone_kda_inputs(inputs), state.detach().clone().requires_grad_(), sliced_offsets)
    for result, reference in zip(actual, expected, strict=True):
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
