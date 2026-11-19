"""Paged indexer scoring within the 99 KiB per-block shared memory of SM86, SM89, and SM120.

The pointer-gather scoring kernel used fixed 128-candidate tiles. With 128 heads and D=256, its
hoisted query and key tiles needed 131072 bytes, so SM86/SM89 (and SM120 without TMA) could not
launch it. The compile-only test builds every launch for those targets without running it; the
reference test runs the selected tiles on the local GPU.
"""

import pytest
import torch

from attn_gym.sparse import paged_lightning_indexer
from attn_gym.testing.indexer import (
    assert_paged_indexer_selection,
    make_paged_indexer_test_inputs,
)
from attn_gym.testing.triton_budget import compile_only_for_target, emulate_device_dispatch

CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
pytestmark = pytest.mark.skipif(not CUDA, reason="SM80+ required")

MAX_SHARED_MEMORY = 101376
TARGETS = [
    pytest.param((8, 6), id="sm86"),
    pytest.param((8, 9), id="sm89"),
    pytest.param((12, 0), id="sm120"),
]
# (heads, head_dim, dtype): the widest accepted tile, a non-power-of-two shape padded to it,
# and the widest rows with fewer heads.
SHAPES = [
    pytest.param(128, 256, torch.float16, id="h128-d256-fp16"),
    pytest.param(128, 256, torch.bfloat16, id="h128-d256-bf16"),
    pytest.param(100, 200, torch.float16, id="h100-d200-fp16"),
    pytest.param(64, 256, torch.bfloat16, id="h64-d256-bf16"),
]


@pytest.mark.parametrize("heads,head_dim,dtype", SHAPES)
@pytest.mark.parametrize("page_size", [13, 64])
@pytest.mark.parametrize("capability", TARGETS)
def test_scoring_fits_99_kib_shared_memory(capability, page_size, heads, head_dim, dtype):
    """Every launch compiles within 99 KiB of shared memory. Compile-only; nothing runs.

    Page 13 takes the pointer-gather kernel everywhere; page 64 takes TMA on SM120.
    """
    args = make_paged_indexer_test_inputs(
        heads=heads, head_dim=head_dim, page_size=page_size, dtype=dtype
    )
    with compile_only_for_target(capability, MAX_SHARED_MEMORY) as launches:
        paged_lightning_indexer(*args, topk=7, kernel_options={"backend": "triton"})

    expected = "_score_tma" if capability >= (9, 0) and page_size % 32 == 0 else "_score"
    assert expected in {launch.kernel for launch in launches}
    too_large = {
        launch.kernel: launch.shared for launch in launches if not launch.fits(MAX_SHARED_MEMORY)
    }
    assert not too_large


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("permuted", [False, True], ids=["contiguous", "permuted"])
def test_small_budget_tiles_match_reference(dtype, permuted):
    """Scoring tiles chosen for SM86 select the reference Top-K on the local GPU.

    Visible counts 1024, 9, and 128 cover full, partial, and sub-tile candidate ranges.
    """
    args = list(
        make_paged_indexer_test_inputs(heads=128, head_dim=256, page_size=13, dtype=dtype, tail=1)
    )
    if permuted:
        # Strided head dimensions take the general-stride loads.
        for i in (0, 1):
            args[i] = args[i].transpose(0, -1).contiguous().transpose(0, -1)
    with emulate_device_dispatch((8, 6), MAX_SHARED_MEMORY):
        actual = paged_lightning_indexer(*args, topk=7, kernel_options={"backend": "triton"})
    assert_paged_indexer_selection(args, actual, 7)
