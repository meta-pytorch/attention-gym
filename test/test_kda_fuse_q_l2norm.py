# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The forward-only fused q L2 norm must match the explicit normalize-then-run path."""

import pytest
import torch

from attn_gym.linear.kda import bounded_gate_cumsum, l2norm

pytest.importorskip("cutlass", reason="fuse_q_l2norm requires the CuTeDSL backend")
from attn_gym.linear.kda import chunk_kda  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the optimized KDA core requires CUDA capability 10.0",
)

_GRAD_OPERANDS = ("q", "k", "v", "cumulative_gate", "beta", "initial_state")


def _inputs(seq_lens: list[int], heads: int):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    total, dim = sum(seq_lens), 128

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=device).bfloat16()

    q, k, v = rand(1, total, heads, dim), rand(1, total, heads, dim), rand(1, total, heads, dim)
    # Stress the in-kernel norm: widely varying row scales plus an all-zero row
    # (which must hit the eps floor identically to the standalone l2norm pass).
    scales = torch.logspace(-3, 3, total, device=device).view(1, total, 1, 1)
    q = (q.float() * scales).bfloat16()
    q[0, total // 2] = 0.0
    raw_gate = rand(1, total, heads, dim)
    a_log = torch.randn(heads, generator=generator, device=device)
    dt_bias = torch.randn(heads, dim, generator=generator, device=device)
    beta = torch.rand(1, total, heads, generator=generator, device=device, dtype=torch.float32)
    cu_seqlens = None
    if len(seq_lens) > 1:
        cu_seqlens = torch.tensor(
            [0, *torch.tensor(seq_lens).cumsum(0).tolist()], device=device, dtype=torch.int32
        )
    cumulative_gate = bounded_gate_cumsum(raw_gate, a_log, dt_bias, cu_seqlens=cu_seqlens)
    initial_state = torch.randn(
        len(seq_lens), heads, dim, dim, generator=generator, device=device, dtype=torch.float32
    )
    return q, k, v, cumulative_gate, beta, cu_seqlens, initial_state


@pytest.mark.parametrize("seq_lens", [[512], [192, 65, 255]], ids=["dense", "ragged"])
def test_fuse_q_l2norm_matches_explicit_normalization(seq_lens: list[int]) -> None:
    q, k, v, cumulative_gate, beta, cu_seqlens, initial_state = _inputs(seq_lens, heads=4)
    kn = l2norm(k)
    expected, expected_state = chunk_kda(
        l2norm(q),
        kn,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    actual, actual_state = chunk_kda(
        q,
        kn,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        fuse_q_l2norm=True,
    )
    # The fused path defers the row scale past the bf16 gram rounding, so
    # outputs agree to bf16 resolution; the state path never touches q's norm.
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(actual_state, expected_state, atol=0.0, rtol=0.0)


def test_fuse_q_l2norm_rejects_compile() -> None:
    q, k, v, cumulative_gate, beta, _, _ = _inputs([256], heads=2)
    kn = l2norm(k)

    compiled = torch.compile(
        lambda: chunk_kda(q, kn, v, cumulative_gate, beta, fuse_q_l2norm=True),
        fullgraph=True,
    )
    # NotImplementedError and dynamo's trace-time wrapper both derive from
    # RuntimeError; only the eager-only contract message matters.
    with pytest.raises(RuntimeError, match="eager-only"):
        compiled()


@pytest.mark.parametrize("operand", _GRAD_OPERANDS)
def test_fuse_q_l2norm_rejects_gradients(operand: str) -> None:
    q, k, v, cumulative_gate, beta, _, initial_state = _inputs([256], heads=2)
    tensors = {
        "q": q,
        "k": l2norm(k),
        "v": v,
        "cumulative_gate": cumulative_gate,
        "beta": beta,
        "initial_state": initial_state,
    }
    tensors[operand] = tensors[operand].clone().requires_grad_(True)

    def run():
        return chunk_kda(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            tensors["cumulative_gate"],
            tensors["beta"],
            tensors["initial_state"],
            fuse_q_l2norm=True,
        )

    with pytest.raises(RuntimeError, match="forward-only"):
        run()
    with torch.no_grad():
        output, _ = run()
    assert not output.requires_grad
