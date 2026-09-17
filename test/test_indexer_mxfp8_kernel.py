"""Direct MXFP8 score-kernel tests, independent of the public indexer/selector wiring."""

import math

import pytest
import torch

pytest.importorskip("cutlass.cute.nvgpu.tcgen05", reason="CuTeDSL 4.7+ required")

import cutlass

from attn_gym._backends.cute.target import (
    detect_compile_target,
    get_compile_target,
    set_compile_target,
)
from attn_gym._backends.cute.utils import (
    requires_int64_abi,
    tensor_supports_contiguous_dim,
    tensor_supports_tma,
)
from attn_gym.sparse.indexer.impl.cute import _compile_mxfp8_scores


@pytest.fixture(autouse=True)
def _supported_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("MXFP8 score kernel requires SM100 or SM103")
    previous = get_compile_target()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    set_compile_target(detect_compile_target(torch.cuda.current_device()))
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32 = tf32
    set_compile_target(previous)


def _inputs(batch: int, tokens: int, heads: int, ratio: int, weight_dtype: torch.dtype):
    torch.manual_seed(819)
    q = (torch.randn(batch, tokens, heads, 128, device="cuda") * 2).to(torch.float8_e4m3fn)
    k = (torch.randn(batch, tokens // ratio, 128, device="cuda") * 2).to(torch.float8_e4m3fn)
    weights = torch.randn(batch, tokens, heads, device="cuda", dtype=weight_dtype)
    q_scale = torch.randint(123, 132, (batch, tokens, heads, 4), device="cuda", dtype=torch.uint8)
    k_scale = torch.randint(
        123, 132, (batch, tokens // ratio, 4), device="cuda", dtype=torch.uint8
    )
    # Zero DATA groups, not invalid E8M0 zero scales.
    q.view(torch.uint8)[:, ::3, ::7, :32] = 0
    k.view(torch.uint8)[:, ::7, 64:96] = 0
    q_scale = q_scale.view(torch.float8_e8m0fnu)
    k_scale = k_scale.view(torch.float8_e8m0fnu)
    return q, k, weights, q_scale, k_scale


def _assert_slab(scores, inputs, pair_start: int, causal: bool, ratio: int):
    """FP64 dequantized oracle plus FP32 eager and a pointwise rounding-error bound."""
    q, k, weights, q_scale, k_scale = inputs
    batch, tokens, heads, dim = q.shape
    pairs_per_batch = (tokens + 1) // 2
    eps = torch.finfo(torch.float32).eps
    # D dot additions, H/4 reduction chain, scaling and final reduction/rounding.
    operations = dim + heads // 4 + 8
    gamma = operations * eps / (1 - operations * eps)
    for local_pair in range(scores.shape[0]):
        global_pair = pair_start + local_pair
        b, pair = divmod(global_pair, pairs_per_batch)
        for qi in range(2):
            query = pair * 2 + qi
            valid = min((query + 1) // ratio, k.shape[1]) if causal else k.shape[1]
            actual = scores[local_pair, qi]
            if b >= batch or query >= tokens:
                assert torch.isnan(actual).all()
                continue
            assert torch.isnan(actual[valid:]).all(), "wrote a masked score"
            if valid == 0:
                continue
            q64 = q[b, query].double() * q_scale[b, query].double().repeat_interleave(32, -1)
            k64 = k[b, :valid].double() * k_scale[b, :valid].double().repeat_interleave(32, -1)
            w64 = weights[b, query].double()
            dots = q64 @ k64.T
            expected = (dots.relu() * w64[:, None]).sum(0) / math.sqrt(heads * dim)
            eager = ((q64.float() @ k64.float().T).relu() * w64.float()[:, None]).sum(0)
            eager /= math.sqrt(heads * dim)
            magnitude = ((q64.abs() @ k64.abs().T) * w64.abs()[:, None]).sum(0)
            magnitude /= math.sqrt(heads * dim)
            actual = actual[:valid].double()
            assert torch.isfinite(actual).all()
            error = (actual - expected).abs()
            assert torch.all(error <= gamma * magnitude), (error.max(), magnitude.max())
            eager_error = (eager.double() - expected).abs().mean()
            rounding_allowance = (math.sqrt(dim) + math.sqrt(heads) + 4) * eps
            assert error.mean() <= eager_error + rounding_allowance * expected.abs().mean()


def _run_slab(inputs, pair_start=0, pairs=5, *, causal=True, ratio=4, wide=False):
    q, k, weights = inputs[:3]
    fn = _compile_mxfp8_scores(
        weights.dtype,
        q.shape[2],
        128,
        causal,
        ratio,
        wide,
        weights.stride(-1) == 1,
        tensor_supports_contiguous_dim(inputs[3], alignment_bytes=4),
        tensor_supports_contiguous_dim(inputs[4], alignment_bytes=4),
    )
    scores = torch.full((pairs, 2, k.shape[1]), float("nan"), device="cuda")
    integer = cutlass.Int64 if wide else cutlass.Int32
    fn(*inputs, scores, integer(pair_start), cutlass.Float32(1 / math.sqrt(q.shape[2] * 128)))
    _assert_slab(scores, inputs, pair_start, causal, ratio)
    return scores


def test_mxfp8_score_4k_smoke():
    inputs = _inputs(1, 4096, 64, 4, torch.float32)
    for start in (0, 511, 2044):
        _run_slab(inputs, start)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens,ratio,causal", [(1, 1, False), (133, 1, True), (523, 4, True)])
def test_mxfp8_score_tails(heads, weight_dtype, tokens, ratio, causal):
    inputs = _inputs(2, tokens, heads, ratio, weight_dtype)
    starts = (0, max(0, (tokens + 1) // 2 - 2), 2 * ((tokens + 1) // 2) - 1)
    for start in starts:
        _run_slab(inputs, start, causal=causal, ratio=ratio)


@pytest.mark.parametrize("heads,weight_dtype", [(32, torch.float32), (64, torch.bfloat16)])
def test_mxfp8_score_strides_and_int64(heads, weight_dtype):
    q, k, w, qs, ks = _inputs(2, 527, heads, 4, weight_dtype)
    q = q.transpose(1, 2).contiguous().transpose(1, 2)
    k = k.transpose(0, 1).contiguous().transpose(0, 1)
    w = w.transpose(1, 2).contiguous().transpose(1, 2)
    qs = qs.transpose(1, 2).contiguous().transpose(1, 2)
    # Byte-aligned pointers and nonunit group strides are legal for both scale tensors.
    qs = torch.empty(*qs.shape[:-1], 9, device="cuda", dtype=qs.dtype)[..., 1::2].copy_(qs)
    ks = torch.empty(2, ks.shape[1], 9, device="cuda", dtype=ks.dtype)[..., 1::2].copy_(ks)
    assert tensor_supports_tma(q) and tensor_supports_tma(k)
    inputs = q, k, w, qs, ks
    normal = _run_slab(inputs, 261)
    wide = _run_slab(inputs, 261, wide=True)
    torch.testing.assert_close(normal, wide, rtol=0, atol=0, equal_nan=True)
    # Oversized singleton strides must also fit the fake signature, even if unreachable.
    singleton = tuple(t[:1].as_strided(t[:1].shape, (1 << 33, *t.stride()[1:])) for t in inputs)
    assert requires_int64_abi(*singleton)
    _run_slab(singleton, 261, wide=True)


def test_mxfp8_score_scale_magnitudes():
    inputs = list(_inputs(1, 137, 64, 1, torch.float32))
    for values in inputs[:2]:
        values.copy_((values.float() * 32).clamp(-448, 448))
    qs, ks = inputs[3:]
    qs.view(torch.uint8).copy_(
        (torch.arange(qs.numel(), device="cuda").reshape(qs.shape) % 41 + 107).to(torch.uint8)
    )
    ks.view(torch.uint8).copy_(
        (torch.arange(ks.numel(), device="cuda").reshape(ks.shape) % 61 + 97).to(torch.uint8)
    )
    _run_slab(tuple(inputs), 60, causal=False, ratio=1)


@pytest.mark.parametrize("heads", [32, 64])
def test_mxfp8_score_graph_replay(heads):
    # Seventeen candidate tiles wrap both the acc3/SF ring and the K4 SMEM ring
    # at least four times, including the odd final query and the next batch.
    inputs = _inputs(2, 8207, heads, 4, torch.float32)
    q, k, weights, qs, ks = inputs
    fn = _compile_mxfp8_scores(weights.dtype, heads, 128, True, 4, False, True, True, True)
    scores = torch.full((7, 2, k.shape[1]), float("nan"), device="cuda")
    start = 4099
    args = (*inputs, scores, cutlass.Int32(start), cutlass.Float32(1 / math.sqrt(heads * 128)))
    fn(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn(*args)
    for i in range(8):
        q.view(torch.uint8).bitwise_xor_(128)
        k.copy_(k.view(torch.uint8).roll(32, -1).view(k.dtype))
        qs.view(torch.uint8).add_(1 if i % 2 else -1)
        ks.copy_(ks.view(torch.uint8).roll(1, -1).view(ks.dtype))
        weights.neg_()
        graph.replay()
        _assert_slab(scores, inputs, start, True, 4)


@pytest.mark.parametrize("heads", [32, 64])
def test_mxfp8_score_relu_large_finite_dot(heads):
    inputs = _inputs(1, 3, heads, 1, torch.float32)
    q, k, weights, qs, ks = inputs
    q.copy_(torch.ones(q.shape, device="cuda"))
    k.copy_(torch.ones(k.shape, device="cuda"))
    qs.view(torch.uint8).fill_(187)
    ks.view(torch.uint8).fill_(187)
    weights.fill_(2.0**-64)
    # Dot = 128 * 2**60 * 2**60 = 2**127 is finite, but x + abs(x)
    # overflows. The weighted reduction and normalized result remain finite.
    _run_slab(inputs, 0, pairs=2, causal=False, ratio=1)


@pytest.mark.parametrize("heads", [32, 64])
def test_mxfp8_score_group_scale_sentinel(heads):
    inputs = list(_inputs(1, 3, heads, 1, torch.float32))
    q, k, weights, qs, ks = inputs
    q.copy_(torch.ones(q.shape, device="cuda"))
    k.copy_(torch.ones(k.shape, device="cuda"))
    weights.fill_(1)
    # Independent row/group exponents distinguish the hardware SF swizzle from row-major.
    qs.view(torch.uint8).copy_(
        (torch.arange(qs.numel(), device="cuda").reshape(qs.shape) % 7 + 124).to(torch.uint8)
    )
    ks.view(torch.uint8).copy_(
        torch.tensor(
            [[128, 125, 127, 130], [126, 129, 124, 128], [129, 124, 128, 126]],
            device="cuda",
            dtype=torch.uint8,
        )
    )
    snapshots = [x.view(torch.uint8).clone() for x in inputs]
    _run_slab(tuple(inputs), 0, pairs=2, causal=False, ratio=1)
    for value, snapshot in zip(inputs, snapshots, strict=True):
        assert torch.equal(value.view(torch.uint8), snapshot)


@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize(
    "packed_q,packed_k", [(False, False), (False, True), (True, False), (True, True)]
)
def test_mxfp8_score_packed_scales(heads, packed_q, packed_k):
    inputs = list(_inputs(2, 527, heads, 4, torch.float32))
    for index, packed in ((3, packed_q), (4, packed_k)):
        scale = inputs[index]
        if packed:
            # Four-byte aligned outer-strided rows are still eligible for packed loads.
            storage = torch.empty(*scale.shape[:-1], 8, device="cuda", dtype=scale.dtype)
            inputs[index] = storage[..., :4].copy_(scale)
        else:
            # A unit group stride alone is not sufficient: offset one is misaligned.
            storage = torch.empty(scale.numel() + 1, device="cuda", dtype=scale.dtype)
            inputs[index] = storage[1:].view(scale.shape).copy_(scale)
        assert tensor_supports_contiguous_dim(inputs[index], alignment_bytes=4) is packed
    _run_slab(tuple(inputs), 261, wide=True)
