"""The fused GDN/KDA training paths and gate transforms run without CuTeDSL installed."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the fused delta-rule pipelines require CUDA capability 8.0 or newer",
)

# Runs in a fresh interpreter whose CuTeDSL and TVM-FFI imports fail, like an install without the
# ``[linear]`` extra. Every operation must take the portable Triton route on every architecture.
_SCRIPT = textwrap.dedent(
    """
    import sys

    for name in ("cutlass", "tvm_ffi"):
        sys.modules[name] = None

    import torch

    from attn_gym.linear import chunk_gdn, chunk_kda, gate_transform
    from attn_gym.linear.gdn import context_parallel_gdn
    from attn_gym.linear.kda import context_parallel_kda
    from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
    from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
    from attn_gym.testing.gdn import make_gdn_test_inputs
    from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs


    def check(name, operation, inputs):
        fused_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
        reference_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
        fused = operation(fused_inputs, "fused")
        reference = operation(reference_inputs, "reference")
        cotangent = torch.randn_like(reference, dtype=torch.float32)
        actual = (fused, *torch.autograd.grad(fused, fused_inputs, cotangent.to(fused.dtype)))
        expected = (
            reference,
            *torch.autograd.grad(reference, reference_inputs, cotangent.to(reference.dtype)),
        )
        for index, (value, golden) in enumerate(zip(actual, expected, strict=True)):
            error = (value.double() - golden.double()).norm() / golden.double().norm()
            assert error < 2e-2, f"{name}[{index}]: relative L2 error {error:.3e}"


    raw_gate = torch.randn(2, 65, 3, 128, device="cuda", dtype=torch.bfloat16)
    gate_parameters = (torch.randn(3, device="cuda"), torch.randn(3, 128, device="cuda"))
    for kind, lower_bound in (("bounded", -5.0), ("softplus", None)):
        check(
            f"gate_transform[{kind}]",
            lambda tensors, impl, kind=kind, lower_bound=lower_bound: gate_transform(
                *tensors, kind=kind, lower_bound=lower_bound, impl=impl
            ),
            (raw_gate, *gate_parameters),
        )

    lengths = [65, 0, 63]
    for cu_seqlens in (None, cumulative_sequence_offsets(lengths)):
        tokens = 128
        gdn_inputs = make_gdn_test_inputs(tokens, batch=1, value_heads=2, dtype=torch.bfloat16)[:5]
        check(
            "chunk_gdn",
            lambda tensors, impl: chunk_gdn(*tensors, cu_seqlens=cu_seqlens, impl=impl)[0],
            gdn_inputs,
        )
        kda_inputs = make_kda_test_inputs(tokens, heads=2, normalize_qk=True)
        check(
            "chunk_kda",
            lambda tensors, impl: chunk_kda(*tensors, cu_seqlens=cu_seqlens, impl=impl)[0],
            kda_inputs,
        )
        for operation, prepare, prepare_backward, inputs in (
            (chunk_kda, chunk_kda_prepare, chunk_kda_prepare_backward, kda_inputs),
            (chunk_gdn, chunk_gdn_prepare, chunk_gdn_prepare_backward, gdn_inputs),
        ):
            inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
            sequences = 1 if cu_seqlens is None else len(lengths)
            state = torch.randn(sequences, 2, 128, 128, device="cuda").requires_grad_()
            options = {"autotune": False}
            output, final = operation(
                *inputs, state, cu_seqlens=cu_seqlens, output_final_state=True,
                kernel_options=options,
            )
            staged = prepare(*inputs, cu_seqlens=cu_seqlens, kernel_options=options)
            actual_output, actual_final = staged.run(state, output_final_state=True)
            torch.testing.assert_close(actual_output, output, rtol=0, atol=0)
            torch.testing.assert_close(actual_final, final, rtol=0, atol=0)
            bounds = (
                torch.tensor([[0, tokens]], device="cuda", dtype=torch.int32)
                if cu_seqlens is None
                else torch.stack((cu_seqlens[:-1], cu_seqlens[1:]), dim=1)
            )
            forward_summary = staged.state_summaries(bounds)
            dout, dfinal = torch.randn_like(output), torch.randn_like(final)
            expected = torch.autograd.grad((output, final), (*inputs, state), (dout, dfinal))
            backward = prepare_backward(
                staged.saved, dout, state, scale=staged.scale, kernel_options=options,
            )
            reverse_summary = backward.state_grad_summaries(bounds)
            actual = backward.run(dfinal)
            for value, golden in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, golden, rtol=0, atol=0)
            assert forward_summary.shape == reverse_summary.shape == (sequences, 2, 256, 128)
            assert torch.isfinite(forward_summary).all() and torch.isfinite(reverse_summary).all()
    print("ok")
    """
)


def test_fused_training_runs_without_cutedsl():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        check=False,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    assert result.stdout.strip().endswith("ok")
