"""Empirical gate-range cases that remain accurate with the FP32 in-chunk scan."""

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import chunk_gdn, recurrent_gdn
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_rms_matches_low_precision_reference,
    cumulative_sequence_offsets,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the fused GDN core requires CUDA capability 8.0 or newer",
)


@pytest.mark.parametrize("backend", ["fused", "warpdrive"])
@pytest.mark.parametrize("pattern", ["isolated_minus_1e4", "uniform_minus_1e30"])
@torch.no_grad()
def test_supported_large_negative_gates_match_recurrent_reference(backend: str, pattern: str):
    if backend == "warpdrive":
        pytest.importorskip("cutlass.experimental")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("the cuDNN backend requires SM100/SM103")

    lengths = [200, 65, 0, 127]
    shape = (1, sum(lengths), 2, 128)
    generator = torch.Generator(device="cuda").manual_seed(21)
    q, k = (
        F.normalize(torch.randn(shape, device="cuda", generator=generator), dim=-1).bfloat16()
        for _ in range(2)
    )
    v = (torch.randn(shape, device="cuda", generator=generator) * 0.5).bfloat16()
    gate = -torch.rand(shape[:-1], device="cuda", generator=generator)
    beta = torch.rand(shape[:-1], device="cuda", generator=generator)
    if pattern == "isolated_minus_1e4":
        gate[:, ::37] = -1e4
    else:
        gate.fill_(-1e30)
    offsets = cumulative_sequence_offsets(lengths)
    inputs = (q, k, v, gate, beta)

    actual = chunk_gdn(
        *inputs,
        cu_seqlens=offsets,
        output_final_state=True,
        kernel_options={"backend": backend, "autotune": False},
    )
    high_precision = recurrent_gdn(
        *(tensor.double() for tensor in inputs),
        cu_seqlens=offsets,
        output_final_state=True,
        impl="reference",
    )
    low_precision = recurrent_gdn(
        *inputs,
        cu_seqlens=offsets,
        output_final_state=True,
        impl="reference",
    )
    # The shared error budget includes eager error and two BF16 rounding bands; it is not
    # fitted to these gate magnitudes. Check both the worst element and aggregate error.
    for name, result, high, low in zip(
        ("output", "final_state"), actual, high_precision, low_precision, strict=True
    ):
        assert_matches_low_precision_reference(result, high, low, name)
        assert_rms_matches_low_precision_reference(result, high, low, name)
