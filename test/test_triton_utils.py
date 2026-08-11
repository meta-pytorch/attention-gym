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
storage_cosize = triton_utils.storage_cosize


@triton.jit
def _store_ptr_offset(output, index, stride: tl.constexpr):
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


def test_ptr_offset_uses_signed_64_bit_arithmetic():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Triton kernels")

    index = 700_001
    stride = 4096
    output = torch.empty((), device="cuda", dtype=torch.int64)
    _store_ptr_offset[(1,)](output, index, stride=stride)

    assert output.item() == index * stride
    assert output.item() > 2**31

    negative_index = -index
    _store_ptr_offset[(1,)](output, negative_index, stride=stride)

    assert output.item() == negative_index * stride
    assert output.item() < -(2**31)
