"""Warpdrive structure of the fused WY/dQKG backward: compile reuse and checked protocol."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.bwd.cute import chunk_kda_bwd_wy_dqkg_fused as wy
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    bwd_wy_dqkg_reference,
    cumulative_sequence_offsets,
)
from attn_gym.warpdrive.cache import detect_compile_target, get_compile_target, set_compile_target

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused CuTe WY backward requires CUDA capability 10.0 or 10.3",
)


def _inputs(tokens, capacity, heads=2):
    torch.manual_seed(tokens)

    def rand16(*shape):
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8

    shape = (1, tokens, heads, 128)
    return (
        *(rand16(*shape) for _ in range(4)),  # q, k, v, v_new
        -torch.rand(shape, device="cuda"),  # g
        torch.rand(1, tokens, heads, device="cuda"),  # beta
        rand16(1, tokens, heads, 64),  # A
        rand16(1, capacity, heads, 128, 128),  # h
        rand16(*shape),  # do
        rand16(1, capacity, heads, 128, 128),  # dh
        rand16(*shape),  # dv
    )


def test_wy_compiles_once_per_layout_across_runtime_shapes():
    """Token, chunk and document counts and the persistent-grid schedule are launch values:
    one dense and one ragged kernel serve them all, including both tuning configs."""
    configs = (wy.ChunkKdaBwdWyDqkgConfig(1), wy.ChunkKdaBwdWyDqkgConfig(2))
    calls = 0
    wy._compile_chunk_kda_bwd_wy_dqkg.cache_clear()
    try:
        for tokens in (64, 128, 192, 1088):
            for config in configs:
                inputs = _inputs(tokens, tokens // 64)
                wy.chunk_kda_bwd_wy_dqkg(*inputs, scale=128**-0.5, config=config)
                calls += 1
        for lengths in ([1], [64, 0, 17], [16] * 17, [65, 300, 2]):
            tokens = sum(lengths)
            metadata = prepare_ragged_chunk_metadata(
                cumulative_sequence_offsets(lengths), tokens, 64
            )
            for config in configs:
                inputs = _inputs(tokens, metadata.capacity)
                wy.chunk_kda_bwd_wy_dqkg(*inputs, metadata, scale=128**-0.5, config=config)
                calls += 1
        torch.cuda.synchronize()
        info = wy._compile_chunk_kda_bwd_wy_dqkg.cache_info()
        assert info.currsize == 2
        assert info.hits + info.misses == calls
    finally:
        wy._compile_chunk_kda_bwd_wy_dqkg.cache_clear()


NAMES = ("dq", "dk", "dv2", "dg", "db", "dA")


@pytest.mark.parametrize("lengths", [None, [4096, 1000, 65, 0, 7]], ids=["dense", "packed"])
def test_wy_persistent_units_match_across_grid_sizes(monkeypatch, lengths):
    """Several units per CTA (managed ring and accumulator-span phases carried across units) give
    the same result for every persistent grid: a full wave, two waves, and three CTAs."""
    heads = 8
    tokens = 4096 if lengths is None else sum(lengths)
    metadata = None
    capacity = tokens // 64
    if lengths is not None:
        metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
        capacity = metadata.capacity
    inputs = _inputs(tokens, capacity, heads)

    def run(waves):
        config = wy.ChunkKdaBwdWyDqkgConfig(waves)
        return wy.chunk_kda_bwd_wy_dqkg(*inputs, metadata, scale=128**-0.5, config=config)

    expected = run(1)
    results = [run(2)]

    class ThreeSms:
        multi_processor_count = 3

    monkeypatch.setattr(wy, "get_device_properties", lambda _device: ThreeSms)
    results.append(run(1))
    for result in results:
        for name, actual, reference in zip(NAMES, result, expected, strict=True):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg=name)
    if lengths is None:
        golden = bwd_wy_dqkg_reference(*(t.double() for t in inputs), 128**-0.5)
        low = bwd_wy_dqkg_reference(*inputs, 128**-0.5)
        # The reference returns db before dg; the kernel returns dg before db.
        expected = (*expected[:3], expected[4], expected[3], expected[5])
        for name, actual, high, reference in zip(NAMES, expected, golden, low, strict=True):
            assert_matches_low_precision_reference(actual, high, reference, name)


def test_wy_rejects_empty_persistent_grid():
    inputs = _inputs(64, 1)
    with pytest.raises(ValueError, match="grid_waves must be positive"):
        wy.chunk_kda_bwd_wy_dqkg(*inputs, scale=128**-0.5, config=wy.ChunkKdaBwdWyDqkgConfig(0))


@pytest.mark.parametrize(
    ("ragged", "scalar_gate"), [(False, False), (True, True)], ids=["dense", "ragged-scalar"]
)
def test_wy_protocol_checks(monkeypatch, traced_compiles, ragged, scalar_gate):
    """Every TMA ring verifies its copies; h/dh and v are joined by the MMA warp's commit and
    every CUDA thread, which fence their generic reads before the refill."""
    entries = []
    compile_tvm_ffi = wy.compile_tvm_ffi

    def compile_and_keep(op, *args, **kwargs):
        entries.append(op)
        return compile_tvm_ffi(op, *args, **kwargs)

    monkeypatch.setattr(wy, "compile_tvm_ffi", compile_and_keep)
    previous = get_compile_target()
    wy._compile_chunk_kda_bwd_wy_dqkg.cache_clear()
    try:
        set_compile_target(detect_compile_target(torch.cuda.current_device()))
        wy._compile_chunk_kda_bwd_wy_dqkg(
            heads=2,
            head_dim=128,
            chunk_size=64,
            io_dtype=torch.bfloat16,
            scale=128**-0.5,
            fastmath=True,
            ragged=ragged,
            scalar_gate=scalar_gate,
        )
    finally:
        set_compile_target(previous)
        wy._compile_chunk_kda_bwd_wy_dqkg.cache_clear()
    (op,) = entries
    text = op.describe()
    rings = {name: c.tma for name, c in op.checks().channels.items() if c.tma is not None}
    expected = ["a", "do", "dv", "vnew", "h", "v", "k", "q"] + ([] if scalar_gate else ["g"])
    assert sorted(rings) == sorted(expected), text
    assert set(rings.values()) == {"verified"}, text
    for name, tx in (("h", 32768), ("v", 8192)):
        assert getattr(op, name).scope.empty_counts[0] == [1 + 256] * 2, text
        assert f"signal    Tma({tx} B) -> {{mma: Umma, cuda: Threads}}" in text, text
    assert "release_fence smem_async (cuda)" in text, text
    assert op.g.scope.empty_counts[0] == [8 + 2], text  # CUDA and aux warps
    # The accumulators held across the V loop are managed: one commit, every CUDA thread.
    for name in ("dq_acc", "dw_acc", "dk_acc", "dA_acc"):
        scope = getattr(op, name).scope
        assert (scope.full_counts[0], scope.empty_counts[0]) == ([1], [256]), text
    assert "unchecked" not in text.replace("user-driven (unchecked)", ""), text
    assert "potential hazard" not in text, text
