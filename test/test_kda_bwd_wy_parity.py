"""Bitwise parity of the warpdrive WY/dQKG backward kernel with its pre-warpdrive version.

The port changes only synchronization, storage and dispatch, so every output (dq, dk, dv2, dg,
db, dA) must match the legacy kernel bit for bit. Removed together with the legacy module.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.bwd.cute import chunk_kda_bwd_wy_dqkg_fused as warpdrive
from attn_gym.linear.kda.bwd.cute import chunk_kda_bwd_wy_dqkg_fused_legacy as legacy
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused CuTe WY backward requires CUDA capability 10.0 or 10.3",
)

NAMES = ("dq", "dk", "dv2", "dg", "db", "dA")


def _inputs(tokens, capacity, key_heads, heads, dtype, scalar_gate, qkv_view, seed):
    torch.manual_seed(seed)

    def rand16(*shape):
        return torch.randn(shape, device="cuda", dtype=dtype) / 8

    if qkv_view:  # unbound views into one packed projection, as the model passes them
        qkv = rand16(1, tokens, 2 * key_heads + heads, 128)
        q, k, v = qkv.split([key_heads, key_heads, heads], dim=2)
    else:
        q, k = rand16(1, tokens, key_heads, 128), rand16(1, tokens, key_heads, 128)
        v = rand16(1, tokens, heads, 128)
    gate_shape = (1, tokens, heads) if scalar_gate else (1, tokens, heads, 128)
    return (
        q,
        k,
        v,
        rand16(1, tokens, heads, 128),  # v_new
        -4 * torch.rand(gate_shape, device="cuda"),  # g
        torch.rand(1, tokens, heads, device="cuda"),  # beta
        rand16(1, tokens, heads, 64),  # A
        rand16(1, capacity, heads, 128, 128),  # h
        rand16(1, tokens, heads, 128),  # do
        rand16(1, capacity, heads, 128, 128),  # dh
        rand16(1, tokens, heads, 128),  # dv
    )


def _check(inputs, metadata, monkeypatch, *, grid_waves=None, int64=False):
    if int64:
        for module in (warpdrive, legacy):
            monkeypatch.setattr(module, "requires_int64_abi", lambda *_tensors: True)

    def run(module):
        config = None if grid_waves is None else module.ChunkKdaBwdWyDqkgConfig(grid_waves)
        return module.chunk_kda_bwd_wy_dqkg(
            *inputs, metadata, scale=128**-0.5, fastmath=True, config=config
        )

    expected = run(legacy)
    actual = run(warpdrive)
    for name, a, e in zip(NAMES, actual, expected, strict=True):
        torch.testing.assert_close(a, e, rtol=0, atol=0, equal_nan=True, msg=name)


# (tokens, key heads, value heads, dtype, scalar gate, QKV views, grid waves, int64 ABI)
DENSE = [
    pytest.param(64, 1, 1, torch.bfloat16, False, False, None, False, id="one-chunk"),
    # 64 chunks x 2 heads: every ring wraps many times, several units per CTA.
    pytest.param(4096, 2, 2, torch.bfloat16, False, False, None, False, id="wrap"),
    pytest.param(1024, 16, 32, torch.bfloat16, False, True, None, False, id="items-gqa-view"),
    pytest.param(512, 4, 64, torch.float16, False, False, 2, False, id="fp16-two-waves"),
    pytest.param(192, 2, 6, torch.bfloat16, True, False, None, False, id="scalar-gate-gqa"),
    pytest.param(256, 2, 4, torch.bfloat16, False, True, None, True, id="int64-view"),
    pytest.param(320, 3, 3, torch.float16, True, False, 2, True, id="int64-scalar"),
]


@pytest.mark.parametrize(
    ("tokens", "key_heads", "heads", "dtype", "scalar_gate", "qkv_view", "waves", "int64"), DENSE
)
def test_dense_wy_matches_legacy(
    monkeypatch, tokens, key_heads, heads, dtype, scalar_gate, qkv_view, waves, int64
):
    inputs = _inputs(tokens, tokens // 64, key_heads, heads, dtype, scalar_gate, qkv_view, tokens)
    _check(inputs, None, monkeypatch, grid_waves=waves, int64=int64)


# (lengths, key heads, value heads, dtype, scalar gate, grid waves, int64 ABI)
PACKED = [
    pytest.param([65, 0, 63], 1, 1, torch.bfloat16, False, None, False, id="tails"),
    pytest.param([1], 1, 2, torch.bfloat16, False, None, False, id="one-token"),
    pytest.param([4096, 5, 300], 2, 2, torch.bfloat16, False, None, False, id="wrap"),
    pytest.param([17, 64, 0, 129] * 10, 4, 8, torch.float16, False, 2, False, id="items"),
    pytest.param([65, 100, 0, 1], 2, 4, torch.bfloat16, True, None, False, id="scalar-gate"),
    pytest.param([130, 61, 2], 2, 2, torch.bfloat16, False, None, True, id="int64"),
]


@pytest.mark.parametrize(
    ("lengths", "key_heads", "heads", "dtype", "scalar_gate", "waves", "int64"), PACKED
)
def test_packed_wy_matches_legacy(
    monkeypatch, lengths, key_heads, heads, dtype, scalar_gate, waves, int64
):
    tokens = sum(lengths)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    inputs = list(
        _inputs(tokens, metadata.capacity, key_heads, heads, dtype, scalar_gate, False, tokens)
    )
    # Inactive capacity is never read: poison it in both runs.
    for index in (7, 9):
        inputs[index][:, metadata.chunk_offsets[-1].item() :] = torch.nan
    _check(tuple(inputs), metadata, monkeypatch, grid_waves=waves, int64=int64)
