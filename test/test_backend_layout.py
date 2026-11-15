"""Strided-layout arithmetic shared by the backends, importable without either toolchain."""

import pytest
import torch

from attn_gym._backends.layout import layout_overlaps


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
