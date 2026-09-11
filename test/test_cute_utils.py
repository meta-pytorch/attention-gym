"""Shared CuTeDSL tensor-layout utility tests."""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.cute.utils import (
    make_fake_strided_tensor,
    tensor_supports_contiguous_dim,
)


def test_tensor_supports_contiguous_dim_tracks_slice_alignment():
    """Separate logical contiguity from the alignment promised to codegen."""
    compact = torch.empty(2, 3, 128)
    assert tensor_supports_contiguous_dim(compact, alignment_bytes=16)

    misaligned_storage = torch.empty(compact.numel() + 1)
    misaligned = misaligned_storage[1:].view_as(compact)
    assert misaligned.is_contiguous()
    assert tensor_supports_contiguous_dim(misaligned, alignment_bytes=4)
    assert not tensor_supports_contiguous_dim(misaligned, alignment_bytes=16)

    outer_strided = torch.empty(2, 3, 2, 128)[:, :, 0, :]
    assert not outer_strided.is_contiguous()
    assert tensor_supports_contiguous_dim(outer_strided, alignment_bytes=16)

    last_dim_strided = torch.empty(2, 3, 256)[..., ::2]
    assert not tensor_supports_contiguous_dim(last_dim_strided, alignment_bytes=4)


def test_make_fake_strided_tensor_keeps_only_one_static_stride():
    """Encode the common dynamic-outer-stride, contiguous-inner-mode ABI."""
    from cutlass import Float16, cute

    fake = make_fake_strided_tensor(
        Float16,
        (cute.sym_int(), 3, 128),
        stride_divisibility=8,
        use_int64_strides=False,
    )
    assert fake.stride[-1] == 1
    assert fake._assumed_align == 16
    assert "div=8" in str(fake)


@pytest.mark.parametrize("use_int64_strides", [False, True])
def test_make_fake_strided_tensor_all_dynamic(use_int64_strides):
    """Scalar-load tensors need no unit stride or stronger-than-element base alignment."""
    from cutlass import Float16, cute

    fake = make_fake_strided_tensor(
        Float16,
        (cute.sym_int(), 3, 128),
        contiguous_dim=None,
        use_int64_strides=use_int64_strides,
    )
    assert fake._assumed_align == 2
    assert all(stride.width == (64 if use_int64_strides else 32) for stride in fake.stride)
    assert all(stride.divisibility == 1 for stride in fake.stride)
