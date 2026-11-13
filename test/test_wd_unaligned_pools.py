"""Paged pools whose rows are not 16-byte aligned run on warpdrive with the aligned pool's bits.

Serving engines carve recurrent state out of packed byte pages (``strided_state_pool``), so a
pool base or slot stride may be aligned only to its element size. The prefills narrow their
state row loads and stores to that alignment and leave everything else unchanged.
"""

from __future__ import annotations

import importlib

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import paged_chunk_gdn, paged_chunk_kda
from attn_gym.linear._delta_rule.paged_state import state_row_alignment
from attn_gym.linear._delta_rule.warpdrive.drivers.kda import PREP_TILE_FRACTION
from attn_gym.testing import cumulative_sequence_offsets, make_gdn_test_inputs, strided_state_pool
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="warpdrive requires an SM100-family GPU",
)

PAGED = {"gdn": paged_chunk_gdn, "kda": paged_chunk_kda}
# Each prefill kernel's state path: the GDN prefill with and without the d_v split and at
# K=V=64, the KDA prep-fed prefill (d_v split with prep), and the KDA prefill with and without
# the d_v split.
PLANS = [
    ("gdn", "split"),
    ("gdn", "uncut"),
    ("gdn", "d64"),
    ("kda", "prep"),
    ("kda", "split"),
    ("kda", "uncut"),
]
# (prefix, suffix) padding of each slot in elements: the default vLLM-style page (every FP32 row
# 12 bytes past a 16-byte boundary, BF16 rows at 6 or 14), and a slot stride that walks the rows
# of slots 0-7 through every offset (FP32: 4, 0, 12, 8, ...; BF16: 2, 8, 14, 4, 10, 0, 6, 12).
LAYOUTS = {"vllm_page": (11, 17), "slot_stride": (1, 2)}


def _inputs(variant: str, lengths: tuple[int, ...], dim: int = 128) -> tuple[torch.Tensor, ...]:
    if variant == "gdn":
        return make_gdn_test_inputs(lengths, key_heads=1, value_heads=2, key_dim=dim, seed=73)[:5]
    return make_kda_test_inputs(sum(lengths), heads=2, seed=73)


def _sequence_count(variant: str, plan: str, minimum: int) -> int:
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if plan == "split" and variant == "kda":
        return int(PREP_TILE_FRACTION * num_sms / 2) + 1
    if plan == "uncut":
        return num_sms // 2 + 1
    return minimum


def _paged(variant, inputs, pool, routes, cu_seqlens, mask):
    with torch.no_grad():
        return PAGED[variant](
            *inputs,
            pool,
            routes,
            cu_seqlens=cu_seqlens,
            has_initial_state=mask,
            kernel_options={"backend": "warpdrive"},
        )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("pool_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("variant", "plan"), PLANS)
def test_unaligned_pool_matches_aligned_pool(
    variant: str, plan: str, pool_dtype: torch.dtype, layout: str
) -> None:
    """Resumed, fresh, empty and null routes give the aligned pool's outputs and slot bits, and
    the page padding around each slot stays untouched."""
    num_sequences = _sequence_count(variant, plan, 9)
    lengths = (17, 0, 0, 19, 40, 0, 33, 21, 64) + (0,) * (num_sequences - 9)
    dim = 64 if plan == "d64" else 128
    inputs = _inputs(variant, lengths, dim)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    routes = torch.tensor(
        [3, 2, 4, 0, 5, -1, 1, 6, 7] + [0] * (num_sequences - 9), device="cuda", dtype=torch.int32
    )
    mask = torch.ones(num_sequences, device="cuda", dtype=torch.bool)
    mask[1] = mask[6] = False
    prefix, suffix = LAYOUTS[layout]
    storage, pool = strided_state_pool(
        9, 2, dim, dim, prefix=prefix, suffix=suffix, dtype=pool_dtype
    )
    storage.mul_(0.01)
    assert state_row_alignment(pool) < 16
    original_storage = storage.clone()
    aligned = pool.clone()

    expected = _paged(variant, inputs, aligned, routes, cu_seqlens, mask)
    actual = _paged(variant, inputs, pool, routes, cu_seqlens, mask)

    assert torch.equal(actual, expected)
    assert torch.equal(pool, aligned)
    padding = slice(0, prefix), slice(storage.shape[1] - suffix, None)
    for columns in padding:
        assert torch.equal(storage[:, columns], original_storage[:, columns])


@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_pool_alignment_specializes_once_per_alignment(variant: str) -> None:
    """A pool's alignment selects the compiled state access; its address and token counts do
    not."""
    bundle = importlib.import_module(
        f"attn_gym.linear._delta_rule.warpdrive.bundles.{variant}_warmup_forward"
    )
    compile_fn = bundle._compile_warmup_forward
    compile_fn.cache_clear()
    routes = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    pools = [
        strided_state_pool(3, 2, 128, 128, prefix=prefix, suffix=suffix)
        for prefix, suffix in ((0, 32), (11, 17), (3, 25))
    ]
    assert [state_row_alignment(pool) for _storage, pool in pools] == [16, 4, 4]
    for (_storage, pool), compiled in zip(pools, (1, 2, 2), strict=True):
        for tokens in (1, 16, 17):
            _paged(
                variant,
                _inputs(variant, (tokens, 0)),
                pool,
                routes,
                cumulative_sequence_offsets((tokens, 0)),
                None,
            )
        assert compile_fn.cache_info().currsize == compiled
