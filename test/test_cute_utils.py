"""Shared CuTeDSL tensor-layout utility tests."""

from __future__ import annotations

import pytest
import torch

from attn_gym._backends.cute.utils import layout_overlaps, tensor_supports_contiguous_dim
from attn_gym.warpdrive.cache import make_fake_strided_tensor


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


@pytest.mark.parametrize(
    ("shape", "strides", "overlaps"),
    [
        ((2, 3), (3, 1), False),  # contiguous
        ((4, 5), (0, 1), True),  # an expanded dimension
        ((3, 4), (2, 1), True),  # rows step less than they span
        ((2, 3), (1, 2), False),  # column-major
        ((1, 4), (0, 1), False),  # size-1 dimensions never step
        ((0, 4), (0, 1), False),  # empty
        ((), (), False),
        # Sufficient, not necessary: injective (0, 2, 4, 3, 5, 7) but nested-reach fails.
        ((3, 2), (2, 3), True),
    ],
)
def test_layout_overlaps(shape, strides, overlaps):
    assert layout_overlaps(shape, strides) is overlaps


def test_views_into_a_packed_map_do_not_overlap():
    """Either half of a packed [C; R] map, and a head slice of it, writes each element once."""
    packed = torch.empty(5, 3, 128 + 64, 64)
    for view in (packed[:, :, :128], packed[:, :, 128:], packed[:, 1:, 128:]):
        assert not layout_overlaps(view.shape, view.stride())
    broadcast = packed[:, :1].expand(5, 3, -1, -1)
    assert layout_overlaps(broadcast.shape, broadcast.stride())
