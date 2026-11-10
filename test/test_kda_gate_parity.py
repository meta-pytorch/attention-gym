"""Bitwise parity of the scheduler-driven gate-transform kernels with their previous versions.

The port changes only how a CTA finds its work item, so the forward gate and the three backward
outputs must match the legacy kernels bit for bit. Removed together with the legacy modules.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.kda.bwd.cute import gate_bwd, gate_bwd_legacy
from attn_gym.linear.kda.fwd.cute import gate_fwd, gate_fwd_legacy
from attn_gym.linear.types import GateTransform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the CuTe gate kernels require CUDA capability >= 9.0",
)


def _inputs(batch, tokens, heads, dtype, layout):
    generator = torch.Generator(device="cuda").manual_seed(7 * batch + 3 * tokens + heads)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, device="cuda", dtype=dtype, generator=generator)

    # Scaled by 4: steep and saturated gates as well as ordinary ones.
    if layout == "strided":  # a head slice of a wider buffer: the general (dynamic-stride) ABI
        raw = randn(batch, tokens, heads + 2, 128, dtype=dtype).mul_(4)[:, :, 1:-1]
        d_gate = randn(batch, tokens, heads + 2, 128)[:, :, 1:-1]
        assert not raw.is_contiguous() and not d_gate.is_contiguous()
    else:
        raw = randn(batch, tokens, heads, 128, dtype=dtype).mul_(4)
        d_gate = randn(batch, tokens, heads, 128)
    return raw, randn(heads), randn(heads, 128), d_gate


def _record_launches(monkeypatch, module, name) -> list[torch.Tensor]:
    """Record the ``raw_gate`` every kernel compiled by ``module.name`` is launched on."""
    launched = []
    compile_fn = getattr(module, name)

    def compile_and_record(*args, **kwargs):
        compiled = compile_fn(*args, **kwargs)

        def launch(raw_gate, *tensors):
            launched.append(raw_gate)
            return compiled(raw_gate, *tensors)

        return launch

    monkeypatch.setattr(module, name, compile_and_record)
    return launched


@pytest.mark.parametrize(
    ("batch", "tokens", "heads", "dtype", "layout"),
    [
        pytest.param(1, 1, 1, torch.bfloat16, "compact", id="minimum"),
        pytest.param(2, 65, 3, torch.bfloat16, "compact", id="partial-tile-and-head-group"),
        pytest.param(3, 97, 9, torch.float16, "strided", id="fp16-strided"),
        pytest.param(2, 33, 16, torch.float32, "compact", id="fp32-one-stage"),
        pytest.param(5, 4096, 8, torch.bfloat16, "compact", id="long"),
        pytest.param(65535, 2, 1, torch.bfloat16, "compact", id="last-tile-grid-batch"),
        pytest.param(65536, 1, 1, torch.bfloat16, "compact", id="first-flat-batch"),
        pytest.param(65537, 33, 2, torch.float16, "strided", id="flat-strided"),
    ],
)
@pytest.mark.parametrize(
    ("transform", "lower_bound"),
    [(GateTransform.SOFTPLUS, 0.0), (GateTransform.BOUNDED, -5.0)],
    ids=["softplus", "bounded"],
)
@pytest.mark.parametrize("fastmath", [False, True], ids=["exact", "fastmath"])
@pytest.mark.parametrize("int64", [False, True], ids=["int32", "int64"])
def test_gate_kernels_match_legacy_bitwise(
    monkeypatch, batch, tokens, heads, dtype, layout, transform, lower_bound, fastmath, int64
):
    if int64:
        if tokens > 1024:
            pytest.skip("the wide ABI is covered on the other shapes")
        for module in (gate_fwd, gate_fwd_legacy, gate_bwd, gate_bwd_legacy):
            monkeypatch.setattr(module, "requires_int64_abi", lambda *_tensors: True)
    raw, a_log, bias, d_gate = _inputs(batch, tokens, heads, dtype, layout)
    launched = [
        _record_launches(monkeypatch, module, f"_compile_gate_transform_{direction}")
        for direction, modules in (
            ("fwd", (gate_fwd_legacy, gate_fwd)),
            ("bwd", (gate_bwd_legacy, gate_bwd)),
        )
        for module in modules
    ]
    args = (raw, a_log, bias, lower_bound, fastmath, transform)
    expected = gate_fwd_legacy._gate_transform_fwd_cuda(*args)
    actual = gate_fwd._gate_transform_fwd_cuda(*args)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True, msg="gate")
    args = (raw, a_log, bias, d_gate, lower_bound, fastmath, transform)
    expected_grads = gate_bwd_legacy._gate_transform_bwd_cuda(*args)
    actual_grads = gate_bwd._gate_transform_bwd_cuda(*args)
    for name, got, want in zip(
        ("d_raw_gate", "dA_log", "d_dt_bias"), actual_grads, expected_grads, strict=True
    ):
        torch.testing.assert_close(got, want, rtol=0, atol=0, equal_nan=True, msg=name)
    # Each of the four kernels read the gate in place: a copy would hide the strided launch.
    assert [[r.stride() for r in raws] for raws in launched] == [[raw.stride()]] * 4
