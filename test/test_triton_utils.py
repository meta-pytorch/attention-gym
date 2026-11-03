# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for utilities shared by hand-written Triton kernels."""

import pytest
import torch

triton = pytest.importorskip("triton")
tl = pytest.importorskip("triton.language")
triton_utils = pytest.importorskip("attn_gym._backends.triton.utils")
fixed_tree_sum = triton_utils.fixed_tree_sum
ptr_offset = triton_utils.ptr_offset
requires_int64_offsets = triton_utils.requires_int64_offsets
storage_cosize = triton_utils.storage_cosize


@triton.jit
def _store_ptr_offset(output, index, stride: tl.constexpr, use_int64_offsets: tl.constexpr):
    if use_int64_offsets:
        index = tl.cast(index, tl.int64)
    tl.store(output, ptr_offset((index,), (stride,)))


def test_storage_cosize():
    """Match CuTe co-size semantics for common nonnegative-strided layouts."""
    assert storage_cosize((), ()) == 1
    assert storage_cosize((0, 3), (3, 1)) == 0
    assert storage_cosize((2, 3), (3, 1)) == 6
    assert storage_cosize((2, 3), (8, 2)) == 13
    assert storage_cosize((4, 5), (0, 1)) == 5

    with pytest.raises(ValueError, match="equal length"):
        storage_cosize((2, 3), (1,))
    with pytest.raises(ValueError, match="nonnegative"):
        storage_cosize((-1,), (1,))
    with pytest.raises(ValueError, match="nonnegative"):
        storage_cosize((1,), (-1,))


def test_requires_int64_offsets_uses_relative_storage_cosize():
    """Use the strict int32 threshold from tensor shape and stride metadata."""
    at_limit = torch.empty_strided((2,), (2**31 - 1,), device="meta")
    over_limit = torch.empty_strided((2,), (2**31,), device="meta")

    assert storage_cosize(at_limit.shape, at_limit.stride()) == 2**31
    assert storage_cosize(over_limit.shape, over_limit.stride()) == 2**31 + 1
    assert not requires_int64_offsets()
    assert not requires_int64_offsets(None, at_limit)
    assert requires_int64_offsets(at_limit, None, over_limit)


def test_requires_int64_offsets_in_compiled_backward():
    """Metadata stride inspection must survive Dynamo's backward tracing retries."""

    class OffsetCheckedIdentity(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, metadata):
            ctx.save_for_backward(metadata)
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            (metadata,) = ctx.saved_tensors
            assert not requires_int64_offsets(metadata)
            return grad, None

    x = torch.arange(4.0, requires_grad=True)
    metadata = torch.tensor([0, 4], dtype=torch.int32)
    compiled = torch.compile(OffsetCheckedIdentity.apply, fullgraph=True, backend="eager")
    output = compiled(x, metadata)
    (grad,) = torch.autograd.grad(output.sum(), (x,))
    torch.testing.assert_close(output, x)
    torch.testing.assert_close(grad, torch.ones_like(x))


@pytest.mark.parametrize("index", [700_001, -700_001])
def test_ptr_offset_preserves_caller_selected_integer_width(index):
    """Let callers retain int32 offsets or conditionally promote them to int64."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton kernels")

    stride = 4096
    output = torch.empty((), device="cuda", dtype=torch.int64)

    _store_ptr_offset[(1,)](output, index, stride=stride, use_int64_offsets=False)
    wrapped_int32 = ((index * stride + 2**31) % 2**32) - 2**31
    assert output.item() == wrapped_int32

    _store_ptr_offset[(1,)](output, index, stride=stride, use_int64_offsets=True)
    assert output.item() == index * stride


@triton.jit
def _store_runtime_stride_offset(output, index, stride):
    tl.store(output, ptr_offset((index,), (stride,)))


def test_ptr_offset_accepts_runtime_strides():
    """Stride values may be runtime scalars; only the tuple arity is static."""
    output = torch.empty(1, device="cuda", dtype=torch.int64)
    for stride in (3, 12288):
        _store_runtime_stride_offset[(1,)](output, 7, stride)
        assert output.item() == 7 * stride


@triton.jit
def _store_fixed_tree_sum(x, output, M: tl.constexpr, N: tl.constexpr, W: tl.constexpr):
    o_m = tl.arange(0, M)[:, None, None]
    o_n = tl.arange(0, N)[None, :, None]
    o_w = tl.arange(0, W)[None, None, :]
    block = tl.load(x + (o_m * N + o_n) * W + o_w)
    tl.store(
        output + tl.arange(0, M)[:, None] * N + tl.arange(0, N)[None, :], fixed_tree_sum(block)
    )


def _adjacent_pair_sum(x: torch.Tensor) -> torch.Tensor:
    while x.shape[-1] > 1:
        x = x.unflatten(-1, (-1, 2)).sum(-1)
    return x.squeeze(-1)


@pytest.mark.parametrize("width", [1, 2, 64, 128, 512])
def test_fixed_tree_sum_matches_adjacent_pairs_under_every_layout(width):
    """Sum adjacent pairs to the same bits for every tile shape and warp count."""
    torch.manual_seed(0)
    # Spread magnitudes so a different association order would change the rounded sum.
    x = torch.randn(16, 16, width, device="cuda") * torch.logspace(-8, 8, width, device="cuda")
    expected = _adjacent_pair_sum(x)
    for m, n in ((16, 16), (4, 16), (1, 16), (1, 1)):
        for num_warps in (1, 4, 8):
            output = torch.empty(16, 16, device="cuda")
            for i in range(0, 16, m):
                for j in range(0, 16, n):
                    _store_fixed_tree_sum[(1,)](
                        x[i : i + m, j : j + n].contiguous(),
                        output[i : i + m, j : j + n],
                        M=m,
                        N=n,
                        W=width,
                        num_warps=num_warps,
                    )
            assert torch.equal(output, expected), (m, n, num_warps)
