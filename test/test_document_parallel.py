# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-aligned fragments give the unsharded bits at any CP degree, with no collectives."""

from __future__ import annotations

import math
from itertools import accumulate

import pytest
import torch

from attn_gym.linear.context_parallel import ContextParallelPlan
from attn_gym.linear.document_parallel import check_document_aligned
from attn_gym.linear.kda import chunk_kda, document_parallel_kda
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

LENGTHS = (17, 65, 129, 257, 513)
NAMES = ("output", "dq", "dk", "dv", "dgate", "dbeta")
# Every fragment table cuts only at document boundaries; the CP degree and shape vary freely.
TABLES = {
    "cp1": [[(0, 981)]],
    "cp2-contiguous": [[(0, 211)], [(211, 981)]],
    "cp2-interleaved": [[(0, 17), (82, 211), (468, 981)], [(17, 82), (211, 468)]],
    "cp3-uneven": [[(0, 17)], [(17, 468)], [(468, 981)]],
    "cp5-one-each": [[(a, b)] for a, b in zip((0, 17, 82, 211, 468), (17, 82, 211, 468, 981))],
}


def _step(inputs, grad, cu_seqlens, kernel_options):
    leaves = tuple(t.detach().requires_grad_() for t in inputs)
    output = document_parallel_kda(*leaves, cu_seqlens=cu_seqlens, kernel_options=kernel_options)
    return (output, *torch.autograd.grad(output, leaves, grad))


def _bits_differ(actual: torch.Tensor, expected: torch.Tensor) -> int:
    return int((actual.view(torch.int32) != expected.view(torch.int32)).sum())


@pytest.mark.parametrize("backend", ["fused", "mega"])
@pytest.mark.parametrize("gate", ["strong", "mild"])
def test_document_aligned_fragments_reproduce_the_unsharded_bits(backend, gate):
    """Every document-aligned table, at every CP degree, equals the single-span run bitwise."""
    if backend == "mega":
        pytest.importorskip("cutlass.experimental")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("Mega needs SM100/103")
    kernel_options = {"backend": backend} if backend != "fused" else None
    cu = (0, *accumulate(LENGTHS))
    torch.manual_seed(7)
    inputs = make_kda_test_inputs(
        cu[-1],
        heads=2,
        seed=7,
        normalize_qk=True,
        sigmoid_beta=True,
        gate_scale=math.log(2) if gate == "strong" else 0.02,
        log_uniform_gate=gate == "strong",
    )
    grad = torch.randn_like(inputs[2])
    device = inputs[0].device
    expected = _step(
        inputs, grad, torch.tensor(cu, dtype=torch.int32, device=device), kernel_options
    )
    for name, fragments in TABLES.items():
        check_document_aligned(cu, fragments)
        for rank in range(len(fragments)):
            plan = ContextParallelPlan.from_fragments(cu, fragments, rank)
            ids = plan.global_token_ids(device)
            actual = _step(
                tuple(t[:, ids].contiguous() for t in inputs),
                grad[:, ids].contiguous(),
                plan.routing(device).cu_seqlens,
                kernel_options,
            )
            mismatches = {
                n: _bits_differ(a, e[:, ids])
                for n, a, e in zip(NAMES, actual, expected, strict=True)
            }
            assert all(v == 0 for v in mismatches.values()), (name, rank, mismatches)


def test_fragments_that_split_a_document_are_rejected():
    cu = (0, *accumulate(LENGTHS))
    check_document_aligned(cu, [[(0, 211)], [(211, 981)]])
    with pytest.raises(ValueError, match="splits a document"):
        check_document_aligned(cu, [[(0, 300)], [(300, 981)]])


def test_split_schedules_are_rejected():
    inputs = make_kda_test_inputs(128, heads=1, seed=1)
    with pytest.raises(ValueError, match="split_forward/split_backward"):
        document_parallel_kda(
            *inputs, cu_seqlens=None, kernel_options={"backend": "mega", "split_forward": True}
        )


@pytest.mark.parametrize("backend", ["fused", "mega"])
def test_matches_the_public_op(backend):
    """Fused: the public ``chunk_kda`` bits. Mega: its forward bits; the backward is Mega's own
    (the public op still runs the fused backward on a Mega forward), so gradients are close."""
    if backend == "mega":
        pytest.importorskip("cutlass.experimental")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("Mega needs SM100/103")
    kernel_options = {"backend": backend} if backend != "fused" else None
    inputs = make_kda_test_inputs(512, heads=2, seed=3, normalize_qk=True, sigmoid_beta=True)
    grad = torch.randn_like(inputs[2])
    actual = _step(inputs, grad, None, kernel_options)
    leaves = tuple(t.detach().requires_grad_() for t in inputs)
    output, _ = chunk_kda(*leaves, autotune=False, kernel_options=kernel_options)
    expected = (output, *torch.autograd.grad(output, leaves, grad))
    assert _bits_differ(actual[0], expected[0]) == 0
    for name, a, e in zip(NAMES[1:], actual[1:], expected[1:], strict=True):
        if backend == "fused":
            assert _bits_differ(a, e) == 0, name
        else:
            torch.testing.assert_close(a, e, atol=2e-2, rtol=2e-2, msg=name)
