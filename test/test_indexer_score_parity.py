"""Bitwise parity of the warpdrive CuTe score kernel with its pre-warpdrive version.

The port changes only synchronization and work decoding, so every written slab entry, and
every entry left untouched, must match the legacy kernels bit for bit. Removed together with
the legacy modules.
"""

import math

import pytest
import torch

from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.sparse.indexer.impl import cute as impl
from attn_gym.testing.indexer import make_indexer_test_inputs

CUTE_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability() in (
    (10, 0),
    (10, 3),
)
pytestmark = pytest.mark.skipif(not CUTE_SUPPORTED, reason="SM100 or SM103 required")

# (dtype, forced int64 ABI, contiguous weight heads, strided Q/K/weights)
ABIS = [
    (torch.bfloat16, False, True, False),
    (torch.float16, False, False, True),
    (torch.bfloat16, True, False, False),
    (torch.float16, True, True, True),
]
# (causal, compress_ratio, packed documents)
MODES = [(False, 1, False), (True, 1, False), (True, 4, False), (True, 1, True), (True, 4, True)]
SENTINEL = -12345.0  # entries a kernel never writes keep this value


def _legacy_compile():
    from attn_gym.sparse.indexer.impl.cute_score_legacy import compile_legacy_scores

    return compile_legacy_scores


def _strided(q, k, weights):
    """Independent outer strides and padded Q/K rows."""
    batch, tokens, heads, dim = q.shape
    q = torch.empty(batch, tokens + 1, heads + 2, dim + 16, device=q.device, dtype=q.dtype)[
        :, :tokens, :heads, :dim
    ].copy_(q)
    k = k.transpose(0, 1).contiguous().transpose(0, 1)
    weights = torch.empty(batch, tokens + 1, heads + 3, device=q.device, dtype=q.dtype)[
        :, :tokens, :heads
    ].copy_(weights)
    return q, k, weights


def _strided_heads(weights):
    """The same weights with head stride 2."""
    batch, tokens, heads = weights.shape
    out = torch.empty(batch, tokens, 2 * heads, device=weights.device, dtype=weights.dtype)
    return out[..., ::2].copy_(weights)


def _packed_bounds(tokens, ratio, causal):
    """Documents of odd, empty, and multi-tile lengths, then trailing inactive capacity."""
    from attn_gym.sparse.indexer.impl.triton import prepare_candidate_bounds

    live = tokens - 5 if tokens > 5 else tokens
    q_offsets, k_offsets = [0], [0]
    for length in (1, 0, 3, 130, 0, 17, 700, tokens):
        q_offsets.append(min(live, q_offsets[-1] + length))
        k_offsets.append(k_offsets[-1] + (q_offsets[-1] - q_offsets[-2]) // ratio)
    cu_q = torch.tensor(q_offsets, dtype=torch.int32, device="cuda")
    cu_k = torch.tensor(k_offsets, dtype=torch.int32, device="cuda")
    return prepare_candidate_bounds(cu_q, cu_k, tokens, causal, ratio), k_offsets[-1]


def _slabs(compile_fn, q, k, weights, args, bounds, capacity):
    """Every slab of one call, as the launcher iterates them, each from a sentinel fill."""
    import cutlass

    from attn_gym.warpdrive.cache import (
        detect_compile_target,
        get_compile_target,
        set_compile_target,
    )

    _, heads, head_dim, _, _, use_int64, _ = args
    previous = get_compile_target()
    try:
        set_compile_target(detect_compile_target(q.device.index))
        kernel = compile_fn(*args, has_candidate_bounds=bounds is not None)
    finally:
        set_compile_target(previous)
    integer = cutlass.Int64 if use_int64 else cutlass.Int32
    scale = cutlass.Float32(1.0 / math.sqrt(heads * head_dim))
    extra = () if bounds is None else (bounds,)
    total = q.shape[0] * ((q.shape[1] + 1) // 2)
    slabs = []
    for start in range(0, total, capacity):
        # One row past the remaining pairs, which the kernels must leave untouched.
        scores = torch.full((capacity + 1, 2, k.shape[1]), SENTINEL, device="cuda")
        live = scores[: min(capacity, total - start) + 1]
        kernel(q, k, weights, live, integer(start), scale, *extra)
        slabs.append(scores)
    return torch.stack(slabs)


def _check(heads, head_dim, mode, abi, batch, tokens, candidates, capacity) -> bool:
    """Assert bitwise slab equality; return whether the case wrote any entry."""
    causal, ratio, packed = mode
    dtype, use_int64, contiguous_weight_heads, strided = abi
    bounds = None
    if packed:
        batch = 1
        bounds, candidates = _packed_bounds(tokens, ratio, causal)
    elif ratio > 1:
        candidates = tokens // ratio
    q, _, weights = make_indexer_test_inputs(tokens, heads, head_dim, dtype, batch=batch)
    k = torch.randn(batch, max(candidates, 1), head_dim, device="cuda", dtype=dtype)
    if strided:
        q, k, weights = _strided(q, k, weights)
    if not contiguous_weight_heads:
        weights = _strided_heads(weights)
    assert (weights.stride(-1) == 1) == contiguous_weight_heads
    args = (dtype, heads, head_dim, causal, ratio, use_int64, contiguous_weight_heads)
    expected = _slabs(_legacy_compile(), q, k, weights, args, bounds, capacity)
    actual = _slabs(impl._compile_scores, q, k, weights, args, bounds, capacity)
    mismatch = actual.view(torch.int32) != expected.view(torch.int32)
    assert not mismatch.any(), f"{int(mismatch.sum())} of {mismatch.numel()} slab entries differ"
    return bool((expected != SENTINEL).any())


# (batch, T, S, slab capacity in pairs): one query, one tile, and a 6-tile ring wrap whose
# slabs end mid-batch.
SHAPES = [(1, 1, 1, 4), (3, 17, 17, 4), (2, 777, 650, 300)]
ABI_IDS = ["bf16", "fp16_strided_wh0", "bf16_i64_wh0", "fp16_i64_strided"]
MODE_IDS = ["dense", "causal", "causal_r4", "packed", "packed_r4"]


@pytest.mark.parametrize("abi", ABIS, ids=ABI_IDS)
@pytest.mark.parametrize("mode", MODES, ids=MODE_IDS)
@pytest.mark.parametrize("heads", [32, 64])
def test_packed_kernel_slab_parity(heads, mode, abi):
    wrote = [_check(heads, 128, mode, abi, *shape) for shape in SHAPES]
    assert any(wrote)


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("heads,head_dim", [(64, 128), (32, 128)])
@pytest.mark.parametrize("packed", [False, True])
def test_public_indices_match_legacy(monkeypatch, deterministic, heads, head_dim, packed):
    """Deterministic indices are identical; otherwise each row selects the same set."""
    tokens, topk = 1029, 64
    q, k, weights = make_indexer_test_inputs(
        tokens, heads, head_dim, torch.bfloat16, batch=1 if packed else 2
    )
    kwargs = {"causal": True, "kernel_options": {"backend": "cute"}}
    if packed:
        lengths = [0, 300, 1, 600, 128]
        offsets = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda")
        kwargs |= {"cu_seqlens": offsets.int(), "cu_seqlens_k": offsets.int()}

    def run():
        previous = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(deterministic)
        try:
            return lightning_indexer(q, k, weights, topk, **kwargs)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)

    actual = run()
    monkeypatch.setattr(impl, "_compile_scores", _legacy_compile())
    expected = run()
    if deterministic:
        assert torch.equal(actual, expected)
    else:
        assert torch.equal(actual.sort(-1).values, expected.sort(-1).values)
