"""BF16 pools use FP32 recurrence and round state to BF16 on each writeback."""

from __future__ import annotations

import importlib

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import paged_chunk_gdn, paged_chunk_kda
from attn_gym.linear._delta_rule.warpdrive.drivers.kda import PREP_TILE_FRACTION
from attn_gym.testing import cumulative_sequence_offsets, make_gdn_test_inputs, strided_state_pool
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="warpdrive requires an SM100-family GPU",
)

PAGED = {"gdn": paged_chunk_gdn, "kda": paged_chunk_kda}
PLANS = [("gdn", "split"), ("gdn", "uncut"), ("kda", "prep"), ("kda", "split"), ("kda", "uncut")]


def _inputs(variant: str, lengths: tuple[int, ...]) -> tuple[torch.Tensor, ...]:
    if variant == "gdn":
        return make_gdn_test_inputs(lengths, key_heads=1, value_heads=2, seed=71)[:5]
    return make_kda_test_inputs(sum(lengths), heads=2, seed=71)


def _sequence_count(variant: str, plan: str, minimum: int) -> int:
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    if plan == "split" and variant == "kda":
        return int(PREP_TILE_FRACTION * num_sms / 2) + 1
    if plan == "uncut":
        return num_sms // 2 + 1
    return minimum


@pytest.mark.parametrize(("variant", "plan"), PLANS)
def test_bf16_pool_replays_changed_routes_without_copying(
    variant: str, plan: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All prefill plans use the original pool at the compiled launch boundary.

    Compare each replay to the same FP32-pool recurrence seeded with the exact BF16 values,
    then round its writeback once. Changing routes and the fresh mask also exercises fresh,
    resumed, empty and null work without recapturing or resetting the BF16 history.
    """
    num_sequences = _sequence_count(variant, plan, 7)
    lengths = (17, 0, 0, 19, 23, 0, 0) + (0,) * (num_sequences - 7)
    inputs = _inputs(variant, lengths)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    routes = torch.tensor(
        [3, 2, 4, 0, -1, 1, 0] + [0] * (num_sequences - 7), device="cuda", dtype=torch.int32
    )
    mask = torch.ones(num_sequences, device="cuda", dtype=torch.bool)
    mask[1] = False
    storage, pool = strided_state_pool(6, 2, 128, 128, prefix=8, suffix=8, dtype=torch.bfloat16)
    original_storage = storage.clone()
    driver = importlib.import_module(f"attn_gym.linear._delta_rule.warpdrive.drivers.{variant}")
    run = driver.run_warmup_forward
    launches = []

    def check_pool(*args, **kwargs):
        state_in, state_out = kwargs["state_in"], kwargs["state_out"]
        assert state_in is state_out
        if state_in.dtype == torch.bfloat16:
            assert state_in.data_ptr() == pool.data_ptr()
            assert state_in.stride() == pool.stride()
            launches.append(state_in.data_ptr())
        return run(*args, **kwargs)

    monkeypatch.setattr(driver, "run_warmup_forward", check_pool)

    def paged(state_pool: torch.Tensor) -> torch.Tensor:
        return PAGED[variant](
            *inputs,
            state_pool,
            routes,
            cu_seqlens=cu_seqlens,
            has_initial_state=mask,
            kernel_options={"backend": "warpdrive"},
        )

    with torch.no_grad():
        # Compile/warm both dtype specializations before capture.
        paged(pool.float())
        paged(pool)
        storage.copy_(original_storage)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = paged(pool)
        storage.copy_(original_storage)
        for resumed in (True, False, True):
            mask[0] = resumed
            mask[1] = not resumed
            routes[3], routes[4] = (5, 0) if resumed else (-2, 5)
            fp32_pool = pool.float()
            expected_output = paged(fp32_pool)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected_output, rtol=0, atol=0)
            torch.testing.assert_close(pool, fp32_pool.bfloat16(), rtol=0, atol=0)
            torch.testing.assert_close(storage[:, :8], original_storage[:, :8], rtol=0, atol=0)
            torch.testing.assert_close(storage[:, -8:], original_storage[:, -8:], rtol=0, atol=0)
    assert len(launches) == 2  # warmup and capture; replay issues no host-side pool operations


@pytest.mark.parametrize(("variant", "plan"), PLANS)
def test_bf16_pool_writeback_rounds_ties_like_fused(variant: str, plan: str) -> None:
    """A one-token update produces BF16 halfway cases in column zero.

    K=2**-7 in column zero, V=1+2**-7, beta=1/2, no decay. The rounded residual is 1/2,
    so the FP32 state adds 2**-8: ties round down in even rows and up in odd rows.
    """
    num_sequences = _sequence_count(variant, plan, 1)
    q = torch.zeros(1, 1, 2, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    k[..., 0] = 2**-7
    value = torch.full_like(q, 1 + 2**-7)
    gate = (
        torch.zeros_like(q, dtype=torch.float32)
        if variant == "kda"
        else torch.zeros(1, 1, 2, device="cuda")
    )
    beta = torch.full((1, 1, 2), 0.5, device="cuda")
    cu_seqlens = cumulative_sequence_offsets((1,) + (0,) * (num_sequences - 1))
    routes = torch.zeros(num_sequences, device="cuda", dtype=torch.int32)
    routes[0] = 1
    seed = torch.ones(2, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    seed[1, :, 1::2, 0] = 1 + 2**-7
    expected = seed.clone()
    expected[1, :, 1::2, 0] = 1 + 2**-6
    with torch.no_grad():
        for backend in ("fused", "warpdrive"):
            pool = seed.clone()
            output = PAGED[variant](
                q,
                k,
                value,
                gate,
                beta,
                pool,
                routes,
                cu_seqlens=cu_seqlens,
                kernel_options={"backend": backend, "autotune": False},
            )
            torch.testing.assert_close(output, torch.zeros_like(value), rtol=0, atol=0)
            torch.testing.assert_close(pool, expected, rtol=0, atol=0)


@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_bf16_pool_all_empty_matches_fused(variant: str) -> None:
    """Empty intervals clear fresh routes and preserve resumed/null routes.

    The public API requires nonempty input capacity; rows past the zero terminal offset are
    unspecified and are deliberately not compared.
    """
    inputs = _inputs(variant, (1, 0, 0, 0))
    routes = torch.tensor([1, 2, 0, -1], device="cuda", dtype=torch.int32)
    mask = torch.tensor([False, True, False, False], device="cuda")
    pool = torch.randn(3, 2, 128, 128, device="cuda", dtype=torch.bfloat16)
    pool[1] = torch.nan
    expected = pool.clone()
    expected[1] = 0.0
    with torch.no_grad():
        for backend in ("fused", "warpdrive"):
            actual = pool.clone()
            output = PAGED[variant](
                *inputs,
                actual,
                routes,
                cu_seqlens=cumulative_sequence_offsets((0, 0, 0, 0)),
                has_initial_state=mask,
                kernel_options={"backend": backend, "autotune": False},
            )
            assert output.shape == inputs[2].shape
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("variant", ["gdn", "kda"])
def test_pool_dtype_specializes_but_token_counts_reuse_the_kernel(variant: str) -> None:
    """FP32 and BF16 require two compiled ABIs, not one specialization per token count."""
    bundle = importlib.import_module(
        f"attn_gym.linear._delta_rule.warpdrive.bundles.{variant}_warmup_forward"
    )
    compile_fn = bundle._compile_warmup_forward
    compile_fn.cache_clear()
    routes = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        for dtype in (torch.float32, torch.bfloat16):
            pool = torch.zeros(3, 2, 128, 128, device="cuda", dtype=dtype)
            for tokens in (1, 16, 17, 65):
                PAGED[variant](
                    *_inputs(variant, (tokens, 0)),
                    pool,
                    routes,
                    cu_seqlens=cumulative_sequence_offsets((tokens, 0)),
                    kernel_options={"backend": "warpdrive"},
                )
            assert compile_fn.cache_info().currsize == (1 if dtype == torch.float32 else 2)
    info = compile_fn.cache_info()
    assert info.hits + info.misses == 8
