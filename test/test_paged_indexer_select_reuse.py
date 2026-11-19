"""The portable paged selector compiles once across modes that only affect scoring."""

import pytest
import torch

from attn_gym.sparse import paged_lightning_indexer
from attn_gym.testing.indexer import (
    assert_paged_indexer_selection,
    make_paged_indexer_test_inputs,
)

CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
pytestmark = pytest.mark.skipif(not CUDA, reason="SM80+ required")

CANDIDATES = 256  # Fixed capacity keeps the selector's BLOCK bucket constant.


def test_bounded_selection_reuses_one_binary_across_ratio_and_positions(triton_compiles):
    """Scored rows select from device bounds, so ratio and positions add no selector variants.

    Unscored pools (capacity <= Top-K) still read the metadata and specialize on it.
    """
    for topk, ratios in ((7, (1, 4, 7)), (512, (1, 4))):
        for ratio in ratios:
            for explicit in (False, True):
                args = make_paged_indexer_test_inputs(
                    (1, 2), (ratio * CANDIDATES,) * 2, compress_ratio=ratio
                )
                positions = (
                    torch.tensor([5, -1, ratio * CANDIDATES - 1], device="cuda").int()
                    if explicit
                    else None
                )
                actual = paged_lightning_indexer(
                    *args,
                    topk=topk,
                    compress_ratio=ratio,
                    query_positions=positions,
                    kernel_options={"backend": "triton"},
                )
                assert_paged_indexer_selection(args, actual, topk, ratio, positions)
    # One bounded binary plus one direct binary per (ratio, explicit) mode.
    assert triton_compiles["paged_radix_topk_kernel"] == 1 + 2 * 2
