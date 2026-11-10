"""Bitwise parity of the warpdrive delta-H backward kernel with its pre-warpdrive version.

The port changes only synchronization, storage and work decoding, so every output the kernels
define (dense dh, dh0 and dV; packed active chunk states, dh0 and active dV rows) must match
the legacy kernel bit for bit. Removed together with the legacy module.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.bwd.cute import chunk_delta_h_bwd as warpdrive
from attn_gym.linear.kda.bwd.cute import chunk_delta_h_bwd_legacy as legacy
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTe delta-H backward stage requires SM100 or SM103",
)


def _inputs(tokens, heads, sequences, lengths, dtype, gate, strided_state, seed):
    torch.manual_seed(seed)
    shape = (1, tokens, heads, 128)
    q = torch.randn(shape, device="cuda", dtype=dtype) / 8
    k = torch.randn_like(q) / 8
    w = torch.randn_like(q) / 8
    do = torch.randn_like(q) / 8
    aqk = torch.randn(1, tokens, heads, 64, device="cuda", dtype=dtype) / 8
    if gate == "vector":
        gk = -torch.rand(shape, device="cuda")
    elif gate == "scalar":
        gk = -3 * torch.rand(shape[:3], device="cuda")
    else:
        gk = None
    state_shape = (sequences, heads, 128, 128)
    if strided_state:
        # A padded key mode: valid for the kernel, but not the compact 128-byte-aligned ABI.
        h0 = torch.randn(sequences, heads, 128, 136, device="cuda")[..., :128] / 8
        dht = torch.randn(sequences, heads, 128, 136, device="cuda")[..., :128] / 8
    else:
        h0 = torch.randn(state_shape, device="cuda") / 8
        dht = torch.randn_like(h0) / 8
    return q, k, w, do, aqk, gk, h0, dht


def _assert_equal(actual, expected, name):
    if expected is None:
        assert actual is None, name
        return
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True, msg=name)


def _force_int64(monkeypatch):
    for module in (warpdrive, legacy):
        monkeypatch.setattr(module, "requires_int64_abi", lambda *_tensors: True)


# (tokens, heads, bv, gate, use_h0, use_dht, dtype, int64 ABI, strided state)
DENSE = [
    pytest.param(64, 3, 16, "vector", True, True, torch.bfloat16, False, False, id="one-chunk"),
    # 64 chunks wrap the 6-stage k ring and every 2-stage ring many times.
    pytest.param(4096, 3, 16, "vector", True, True, torch.bfloat16, False, False, id="wrap"),
    pytest.param(4096, 2, 32, "scalar", True, False, torch.bfloat16, False, False, id="wrap-bv32"),
    # 64 heads x 8 value tiles: several persistent items per CTA, odd counts on some CTAs.
    pytest.param(256, 64, 16, "vector", False, True, torch.bfloat16, False, False, id="items"),
    pytest.param(192, 64, 32, None, True, True, torch.bfloat16, False, False, id="items-bv32"),
    pytest.param(192, 5, 32, "vector", False, False, torch.float16, False, False, id="fp16"),
    pytest.param(128, 3, 16, "scalar", True, True, torch.bfloat16, True, False, id="int64"),
    pytest.param(128, 3, 32, "vector", True, True, torch.float16, True, True, id="int64-strided"),
    pytest.param(320, 4, 16, None, True, True, torch.bfloat16, False, True, id="strided"),
]


@pytest.mark.parametrize(
    ("tokens", "heads", "bv", "gate", "use_h0", "use_dht", "dtype", "int64", "strided"), DENSE
)
def test_dense_delta_h_matches_legacy(
    monkeypatch, tokens, heads, bv, gate, use_h0, use_dht, dtype, int64, strided
):
    q, k, w, do, aqk, gk, h0, dht = _inputs(tokens, heads, 1, None, dtype, gate, strided, tokens)
    if int64:
        _force_int64(monkeypatch)

    def run(module):
        return module.blackwell_delta_h_bwd_dhu_dv_fused(
            q,
            k,
            w,
            do,
            aqk,
            gk=gk,
            h0=h0 if use_h0 else None,
            dht=dht if use_dht else None,
            scale=128**-0.5,
            bv=bv,
        )

    expected = run(legacy)
    actual = run(warpdrive)
    for name, a, e in zip(("dh", "dh0", "dv"), actual, expected, strict=True):
        _assert_equal(a, e, name)


# (lengths, physical tokens or None, heads, bv, gate, use_h0, use_dht, dtype, int64)
PACKED = [
    pytest.param(
        [65, 0, 63], None, 3, 16, "vector", True, True, torch.bfloat16, False, id="tails"
    ),
    pytest.param([1], None, 2, 32, "vector", True, True, torch.bfloat16, False, id="one-token"),
    pytest.param([0, 0], None, 2, 16, "vector", True, True, torch.bfloat16, False, id="all-empty"),
    pytest.param(
        [130, 0, 0], 256, 3, 32, "scalar", True, False, torch.float16, False, id="nan-suffix"
    ),
    pytest.param(
        [4096, 5, 300], None, 2, 16, "vector", False, True, torch.bfloat16, False, id="wrap"
    ),
    pytest.param(
        [17, 64, 0, 129] * 10, None, 8, 32, None, True, True, torch.bfloat16, False, id="items"
    ),
    pytest.param([65, 0, 63], None, 3, 32, "scalar", True, True, torch.bfloat16, True, id="int64"),
    # No initial state, >= 32 sequences, >= 8 heads and short documents: the bounded extent.
    pytest.param(
        [65, 0, 63] + [0] * 77,
        256,
        8,
        16,
        "vector",
        False,
        True,
        torch.bfloat16,
        False,
        id="bound-sparse",
    ),
    pytest.param(
        [4] * 16 + [3] * 64,
        None,
        8,
        32,
        "vector",
        False,
        False,
        torch.bfloat16,
        False,
        id="bound-full",
    ),
    pytest.param(
        [8] * 40 + [0] * 24,
        512,
        8,
        16,
        "scalar",
        False,
        True,
        torch.float16,
        True,
        id="bound-int64",
    ),
    # Long documents keep the unbounded path at the same compile key.
    pytest.param(
        [66] * 32, None, 8, 16, "vector", False, False, torch.bfloat16, False, id="unbounded"
    ),
]


@pytest.mark.parametrize(
    ("lengths", "physical", "heads", "bv", "gate", "use_h0", "use_dht", "dtype", "int64"),
    PACKED,
)
def test_packed_delta_h_matches_legacy(
    monkeypatch, lengths, physical, heads, bv, gate, use_h0, use_dht, dtype, int64
):
    active_tokens = sum(lengths)
    tokens = physical or active_tokens
    sequences = len(lengths)
    q, k, w, do, aqk, gk, h0, dht = _inputs(
        tokens, heads, sequences, lengths, dtype, gate, False, sequences + tokens
    )
    for tensor in (q, k, w, do, aqk, gk):
        if tensor is not None:
            tensor[:, active_tokens:].fill_(torch.nan)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    if int64:
        _force_int64(monkeypatch)

    def run(module):
        return module._blackwell_delta_h_bwd_dhu_dv_fused_packed(
            q,
            k,
            w,
            do,
            aqk,
            metadata,
            gk=gk,
            h0=h0 if use_h0 else None,
            dht=dht if use_dht else None,
            scale=128**-0.5,
            bv=bv,
        )

    expected = run(legacy)
    actual = run(warpdrive)
    active_chunks = sum((length + 63) // 64 for length in lengths)
    _assert_equal(actual[0][:, :active_chunks], expected[0][:, :active_chunks], "dh")
    _assert_equal(actual[1], expected[1], "dh0")
    _assert_equal(actual[2][:, :active_tokens], expected[2][:, :active_tokens], "dv")
