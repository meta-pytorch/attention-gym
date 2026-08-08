# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for utilities shared by hand-written Triton kernels."""

import pytest

triton_utils = pytest.importorskip("attn_gym._backends.triton.utils")
storage_cosize = triton_utils.storage_cosize


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
