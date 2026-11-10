# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FP64 numerics of the KDA intra-chunk forward engine's raw outputs.

The engine gates its operands with the flush-to-zero ``ex2`` and a shifted strip rebase (module
docstring of ``chunk_kda_fwd_intra_engine``), so its outputs are not bitwise-equal to any earlier
kernel. These tests hold the raw Aqk, AkkOD and Akkd of a launch to an FP64 evaluation of the
same pairwise sums, ``2^(g_i - g_j)`` taken directly, entry by entry, and check that every entry
the kernel must not write keeps its sentinel.
"""

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

from attn_gym._backends.cute import normalize_compact_tensor
from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.linear.kda.fwd.cute import chunk_kda_fwd_intra_engine as engine
from attn_gym.linear.kda.naive import chunk_cumsum_ref
from attn_gym.testing.kda import cumulative_sequence_offsets
from attn_gym.warpdrive.cache import detect_compile_target, get_compile_target, set_compile_target

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="KDA intra engine requires SM100 or SM103",
)

BT, BC, D = 64, 16, 128
SENTINEL = -1024.0  # entries a kernel never writes keep this value (exact in BF16)
SCALE = D**-0.5
GATE_BOUND = 5.9  # nats per token, inside the fused path's documented [-5.914, 0]
# Per-entry tolerance: two BF16 operand roundings per term plus the BF16 Aqk store stay within
# 3 * 2^-8 of the sum of the terms' magnitudes, so RTOL leaves a 1.33x margin. The floor sits far
# below a visible term and above what the rebase may drop (decays below 2^-62 for |q_d| <= 1).
RTOL, FLOOR = 2**-6, 2**-56

# (heads, forced int64 ABI, launched CTAs). None launches the device's SM count, where small
# inputs leave most CTAs without a unit; 3 makes every CTA run many units with uneven counts.
CONFIGS = [(2, False, None), (3, True, 3), (16, False, 3), (1, True, None)]
CONFIG_IDS = ["h2", "h3_i64_grid3", "h16_grid3", "h1_i64"]
# Packed sequence lengths and trailing inactive token capacity.
PACKED = [
    ([1], 0),
    ([64], 0),
    ([0, 0], 128),  # no active chunk, nonzero capacity
    ([65, 0, 63, 200, 1, 64], 0),
    ([0, 700, 0, 17, 129], 150),
    ([64] * 37, 0),
    ([1500, 1, 900], 70),
]
DENSE_TOKENS = [64, 128, 64 * 37]
REGIMES = ["typical", "steep", "fast"]
NAMES = ("Aqk", "AkkOD", "Akkd")


def _gate(regime, shape, generator):
    """Per-token natural-log gates."""
    u = torch.rand(shape, device="cuda", generator=generator)
    if regime == "typical":  # log-uniform in [-1, 0]
        return (u * (1 - math.exp(-1)) + math.exp(-1)).log()
    if regime == "steep":  # log-uniform in [-GATE_BOUND, 0]
        return (u * (1 - math.exp(-GATE_BOUND)) + math.exp(-GATE_BOUND)).log()
    return -(GATE_BOUND - 0.7 * u)  # fast: every token decays by 5.2-5.9 nats


def _cumulative(gate, lengths, tokens):
    """Chunk-local cumulative log2 gates; inactive capacity keeps its raw values."""
    offsets = None if lengths is None else cumulative_sequence_offsets(lengths)
    g = gate.float() * LOG2_E
    active = chunk_cumsum_ref(g, BT, cu_seqlens=offsets)
    return torch.cat((active, g[:, active.shape[1] : tokens]), dim=1)


def _inputs(tokens, heads, regime, lengths, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, tokens, heads, D)
    q = F.normalize(torch.randn(shape, device="cuda", generator=generator), dim=-1)
    k = F.normalize(torch.randn(shape, device="cuda", generator=generator), dim=-1)
    g = _cumulative(_gate(regime, shape, generator), lengths, tokens)
    beta = torch.rand(1, tokens, heads, device="cuda", generator=generator)
    return q.to(torch.bfloat16), k.to(torch.bfloat16), g, beta


def _packed_projection(q, k):
    """Sibling Q/K views of one packed projection: head and token strides three tiles wide."""
    packed = q.new_empty(*q.shape[:3], 3, q.shape[3])
    packed[..., 0, :].copy_(q)
    packed[..., 1, :].copy_(k)
    return packed.flatten(3)[..., :D], packed.flatten(3)[..., D : 2 * D]


def _metadata(lengths, tokens):
    if lengths is None:
        return None
    return prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, BT)


def _launch(config, q, k, g, beta, metadata, compile_fn=None):
    """The three raw outputs of one launch, each from a sentinel fill."""
    heads, use_int64, grid = config
    compile_fn = compile_fn or engine._compile_intra_engine_fwd
    tokens = q.shape[1]
    capacity = tokens // BT if metadata is None else metadata.capacity
    aqk = torch.full((tokens, heads, BT), SENTINEL, device="cuda", dtype=torch.bfloat16)
    akkod = torch.full((capacity * 6, heads * 256), SENTINEL, device="cuda")
    akkd = torch.full((tokens, heads, BC), SENTINEL, device="cuda")
    previous = get_compile_target()
    target = detect_compile_target(q.device.index)
    try:
        set_compile_target(target if grid is None else replace(target, sm_count=grid))
        kernel = compile_fn(heads, D, metadata is not None, use_int64)
    finally:
        set_compile_target(previous)
    kernel(
        q[0],
        k[0],
        normalize_compact_tensor(g[0]),
        normalize_compact_tensor(beta[0], alignment_bytes=8),
        aqk,
        akkod,
        akkd,
        SCALE,
        None if metadata is None else metadata.cu_seqlens,
        None if metadata is None else metadata.chunk_offsets,
        tokens,
        heads,
    )
    return aqk, akkod, akkd


def _chunks(lengths, tokens):
    """(chunk index, first token, valid rows) of every active chunk, in the kernel's order."""
    if lengths is None:
        return [(c, c * BT, BT) for c in range(tokens // BT)]
    chunks, start = [], 0
    for length in lengths:
        for offset in range(0, length, BT):
            chunks.append((len(chunks), start + offset, min(BT, length - offset)))
        start += length
    return chunks


def _reference(q, k, g, beta, lengths, capacity):
    """FP64 raw outputs, SENTINEL where the kernel must not write.

    ``Aqk[t, :, c] = scale * sum_d q[t] k[s] 2^(g[t] - g[s])`` for ``s = chunk start + c <= t``
    and 0 for the rest of an active row. Akk has ``beta[t]`` instead of ``scale``, ``k[t]``
    instead of ``q[t]``, and ``s < t``: AkkOD stores its six strictly lower 16x16 blocks per
    chunk, Akkd the strictly lower part of the diagonal blocks (row-major, one row per token).
    """
    tokens, heads = q.shape[1], q.shape[2]
    aqk = torch.full((tokens, heads, BT), SENTINEL, device="cuda", dtype=torch.float64)
    akkod = torch.full((capacity * 6, heads, BC, BC), SENTINEL, device="cuda", dtype=torch.float64)
    akkd = torch.full((tokens, heads, BC), SENTINEL, device="cuda", dtype=torch.float64)
    for chunk, start, n in _chunks(lengths, tokens):
        rows = slice(start, start + n)
        qc, kc, gc, bc = (t[0, rows].double() for t in (q, k, g, beta))
        causal = torch.ones(n, n, device="cuda", dtype=torch.bool).tril()
        # 2^(g_i - g_j) for j <= i; the masked j > i side is clamped before exp2 to stay finite.
        decay = (gc[:, None] - gc[None, :]).clamp_max(0).exp2() * causal[..., None, None]
        kd = torch.einsum("jhd,ijhd->ihjd", kc, decay)
        a = torch.zeros(n, heads, BT, device="cuda", dtype=torch.float64)
        a[..., :n] = SCALE * torch.einsum("ihd,ihjd->ihj", qc, kd)
        aqk[rows] = a
        kk = torch.zeros(BT, heads, BT, device="cuda", dtype=torch.float64)
        strict = causal.tril(-1)[:, None, :]
        kk[:n, :, :n] = bc[..., None] * torch.einsum("ihd,ihjd->ihj", kc, kd) * strict
        blocks = kk.view(4, BC, heads, 4, BC)  # (row block, row, head, col block, col)
        pairs = [(ri, ci) for ri in range(4) for ci in range(ri)]
        akkod[chunk * 6 : chunk * 6 + 6] = torch.stack(
            [blocks[ri, :, :, ci].transpose(0, 1) for ri, ci in pairs]
        )
        diag = torch.stack([blocks[b, :, :, b] for b in range(4)]).view(BT, heads, BC)
        akkd[rows] = diag[:n]
    return aqk, akkod.view(capacity * 6, heads * 256), akkd


def _off_entries(got, ref, bound):
    """Written entries outside ``bound`` (NaN counts as outside); asserts the sentinel layout."""
    written = ref != SENTINEL
    assert torch.equal(got != SENTINEL, written), "written entries differ from the contract"
    err = (got.double() - ref).abs()
    return written & ~(err <= bound), err


def _case(config, tokens, lengths, regime, seed, strided=False):
    """Inputs, FP64 outputs, and the FP64 sums of the terms' magnitudes."""
    q, k, g, beta = _inputs(tokens, config[0], regime, lengths, seed)
    if strided:
        q, k = _packed_projection(q, k)
    metadata = _metadata(lengths, tokens)
    capacity = tokens // BT if metadata is None else metadata.capacity
    expected = _reference(q, k, g, beta, lengths, capacity)
    magnitude = _reference(q.abs(), k.abs(), g, beta, lengths, capacity)
    return (q, k, g, beta, metadata), expected, magnitude


def _check(config, tokens, lengths, regime, seed, strided=False):
    inputs, expected, magnitude = _case(config, tokens, lengths, regime, seed, strided)
    actual = _launch(config, *inputs)
    for name, got, ref, size in zip(NAMES, actual, expected, magnitude, strict=True):
        off, err = _off_entries(got, ref, size * RTOL + FLOOR)
        assert not off.any(), (
            f"{name} ({regime}): {int(off.sum())} off, worst {err[off].max():.2e}"
        )


def _cases(config):
    """(tokens, lengths, regime, seed, strided) of the dense and packed layouts."""
    for index, tokens in enumerate(DENSE_TOKENS):
        yield tokens, None, REGIMES[index % 3], tokens, False
    yield 64 * 5, None, "steep", 5, True
    for index, (lengths, inactive) in enumerate(PACKED):
        tokens = sum(lengths) + inactive
        yield tokens, lengths, REGIMES[index % 3], index, False
        if sum(lengths) > 300:
            yield tokens, lengths, "fast", index, True


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_matches_fp64(config):
    for case in _cases(config):
        _check(config, *case)


@pytest.mark.parametrize("config", CONFIGS, ids=CONFIG_IDS)
def test_error_no_worse_than_legacy(config):
    """Against FP64, no output is less accurate than the pre-warpdrive kernel's exact build.

    Removed together with the legacy module. Rounding differs entry by entry, so the bound is
    5% of the legacy rel-L2; steep gates where the legacy kernel loses terms come out far below.
    """
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine_legacy import (
        compile_legacy_intra_engine_fwd,
    )

    def legacy(*args):
        return compile_legacy_intra_engine_fwd(*args, False)

    def rel_l2(got, ref):
        written = ref != SENTINEL
        return ((got.double() - ref)[written].norm() / ref[written].norm().clamp_min(1e-30)).item()

    for case in _cases(config):
        inputs, expected, _ = _case(config, *case)
        new = _launch(config, *inputs)
        old = _launch(config, *inputs, compile_fn=legacy)
        for name, got, base, ref in zip(NAMES, new, old, expected, strict=True):
            if (ref != SENTINEL).any():
                assert rel_l2(got, ref) <= 1.05 * rel_l2(base, ref), (case[:3], name)


def test_many_units_per_cta_match_fp64():
    """Seven units per CTA on the device's own grid: T = 4096, H = 16 over 152 CTAs on GB300."""
    _check((16, False, None), 4096, None, "steep", seed=1)
    _check((16, False, None), 4096, [1000, 0, 37, 2000, 64, 995], "fast", seed=2)


# (q, k) entry magnitudes: unit, and either side at 2^62, just inside the 2^63 operand range.
AMPLITUDES = {"unit": (1.0, 1.0), "large_q": (2.0**62, 1.0), "large_k": (1.0, 2.0**62)}


@pytest.mark.parametrize("amplitude", AMPLITUDES, ids=list(AMPLITUDES))
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
def test_steep_decay_is_exact_to_bf16(packed, amplitude):
    """Every pair's decay survives the rebase up to the gate bound, down to a 2^-60 floor.

    Head ``h`` has one active channel (q, k along e_0) and a constant per-token gate, so each
    entry is a single term ``q_0 k_0 2^(g_i - g_j)`` that no other channel can mask and the error
    bound is elementwise. At 5.9 nats per token a strip spans 127.7 log2 units, so rebasing at the
    strip's first row alone puts its last row's factor at 2^-127.7: the flush-to-zero ``ex2``
    drops that row's own diagonal term, and the exact ``ex2`` leaves it a BF16 subnormal with two
    significant bits once scaled by q. The next strip's rows fall below the BF16 range and lose
    ``exp(-5.9)``-size terms in both. With ``k`` at 2^62 that rebase also overflows BF16.
    """
    gates = torch.tensor([-0.5, -3.0, -5.5, -GATE_BOUND, -5.914], device="cuda")
    q_amp, k_amp = AMPLITUDES[amplitude]
    lengths = [70, 0, 1, 130, 33] if packed else None
    tokens = sum(lengths) + 24 if packed else 192
    heads = gates.numel()
    q = torch.zeros(1, tokens, heads, D, device="cuda", dtype=torch.bfloat16)
    k = q.clone()
    q[..., 0] = q_amp
    k[..., 0] = k_amp
    gate = gates.view(1, 1, heads, 1).expand(1, tokens, heads, D)
    g = _cumulative(gate, lengths, tokens)
    beta = torch.ones(1, tokens, heads, device="cuda")
    metadata = _metadata(lengths, tokens)
    capacity = tokens // BT if metadata is None else metadata.capacity
    expected = _reference(q, k, g, beta, lengths, capacity)
    actual = _launch((heads, False, None), q, k, g, beta, metadata)
    for name, got, ref in zip(NAMES, actual, expected, strict=True):
        written = ref != SENTINEL
        floor = ref[written].abs().max() * 2**-60
        off, err = _off_entries(got, ref, ref.abs() * RTOL + floor)
        assert not off.any(), f"{name}: {int(off.sum())} entries off, worst {err[off].max():.3e}"
