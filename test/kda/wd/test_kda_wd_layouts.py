"""Outer-stride coverage for the KDA warpdrive backend's public forward and backward."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym.linear import chunk_kda
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the warpdrive KDA path requires SM100 or SM103",
)


def pad_rows(tensor: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """Copy a tensor into rows padded by `padding` elements: 16 bytes for fp32, not 32."""
    storage = torch.empty(
        *tensor.shape[:-1], tensor.shape[-1] + padding, dtype=tensor.dtype, device=tensor.device
    )
    return storage[..., : tensor.shape[-1]].copy_(tensor)


def _run(q, k, v, gate, beta, initial_state, cu_seqlens, d_output, d_final_state):
    inputs = tuple(t.detach().requires_grad_() for t in (q, k, v, gate, beta, initial_state))
    output, final_state = chunk_kda(
        *inputs[:5],
        initial_state=inputs[5],
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        kernel_options={"backend": "warpdrive"},
    )
    grads = torch.autograd.grad((output, final_state), inputs, (d_output, d_final_state))
    return (output, final_state, *grads)


def test_kda_wd_16_byte_strided_fp32_operands_match_compact() -> None:
    """An fp32 gate and final-state cotangent whose rows are padded to a 16-byte multiple are
    legal operands; forward and backward must equal the compact layout bit for bit."""
    lengths = (65, 0, 63)
    q, k, v, gate, beta = make_kda_test_inputs(
        sum(lengths), heads=2, seed=331, normalize_qk=True, dtype=torch.bfloat16
    )
    assert gate.dtype == torch.float32
    torch.manual_seed(337)
    cu_seqlens = torch.tensor([0, 65, 65, 128], dtype=torch.int32, device="cuda")
    initial_state = torch.randn(len(lengths), 2, 128, 128, device="cuda") / 100
    d_output = torch.randn_like(v)
    d_final_state = torch.randn_like(initial_state)

    expected = _run(q, k, v, gate, beta, initial_state, cu_seqlens, d_output, d_final_state)
    actual = _run(
        q, k, v, pad_rows(gate), beta, initial_state, cu_seqlens, d_output, pad_rows(d_final_state)
    )
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
