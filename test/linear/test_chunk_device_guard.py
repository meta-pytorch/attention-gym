"""Chunk attention follows its operands' device, not the thread's current CUDA device."""

import pytest
import torch
import torch.nn.functional as F

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, chunk_kda

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="non-current-device regression requires at least two CUDA GPUs",
)


@pytest.mark.parametrize("family", ["gdn", "kda"])
@pytest.mark.parametrize("backend", ["fused", "warpdrive"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_forward_and_backward_follow_tensor_device(
    family: str, backend: str, packed: bool, compiled: bool
):
    with torch.cuda.device(1):
        if torch.cuda.get_device_capability() < (8, 0):
            pytest.skip("the fused chunk cores require CUDA capability 8.0 or newer")
        if backend == "warpdrive":
            pytest.importorskip("cutlass.experimental")
            if not cuda_runs_sm100_kernels():
                pytest.skip("the cuDNN backend requires an SM100-family GPU")
        generator = torch.Generator(device="cuda:1").manual_seed(0)
        tokens = 100 if packed else 128
        shape = (1, tokens, 2, 128)
        q, k = (
            F.normalize(
                torch.randn(shape, device="cuda:1", generator=generator), dim=-1
            ).bfloat16()
            for _ in range(2)
        )
        v = torch.randn(shape, device="cuda:1", generator=generator).bfloat16()
        gate_shape = shape if family == "kda" else shape[:-1]
        gate = -torch.rand(gate_shape, device="cuda:1", generator=generator)
        beta = torch.rand(shape[:-1], device="cuda:1", generator=generator)
        state = (
            torch.randn(2 if packed else 1, 2, 128, 128, device="cuda:1", generator=generator) / 8
        )
        offsets = (
            torch.tensor([0, 60, 100], dtype=torch.int32, device="cuda:1") if packed else None
        )

    operation = chunk_kda if family == "kda" else chunk_gdn
    if compiled:
        operation = torch.compile(operation, fullgraph=True, dynamic=True)

    def run():
        inputs = tuple(
            tensor.detach().clone().requires_grad_() for tensor in (q, k, v, gate, beta)
        )
        initial_state = state.detach().clone().requires_grad_()
        output, final_state = operation(
            *inputs,
            initial_state,
            cu_seqlens=offsets,
            output_final_state=True,
            kernel_options={"backend": backend, "autotune": False},
        )
        gradients = torch.autograd.grad(
            output.float().sum() + final_state.sum(), (*inputs, initial_state)
        )
        return output, final_state, *gradients

    with torch.cuda.device(1):
        expected = run()
    with torch.cuda.device(0):
        torch.empty(1, device="cuda:0")  # Keep both devices' CUDA contexts live.
        actual = run()
        torch.cuda.synchronize(1)
        assert torch.cuda.current_device() == 0
    for result, reference in zip(actual, expected, strict=True):
        assert result.device == torch.device("cuda:1")
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
