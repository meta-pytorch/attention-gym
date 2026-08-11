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
