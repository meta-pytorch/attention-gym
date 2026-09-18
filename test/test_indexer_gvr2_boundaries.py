"""Adversarial arithmetic and scratch-boundary regressions for both GVR2 CTA variants."""

import math
import struct

import pytest
import torch

pytest.importorskip("cutlass.cute")
import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from attn_gym._backends.cute import compile_tvm_ffi
from attn_gym._backends.cute.target import (
    detect_compile_target,
    get_compile_target,
    set_compile_target,
)
from attn_gym.sparse.indexer.impl.cute_topk_gvr2 import (
    IndexerGVR2TopKKernel,
    _bin_index,
    _bin_scale,
)
from attn_gym.testing.indexer import (
    assert_indexer_topk_values as _assert_rows,
)
from attn_gym.testing.indexer import (
    compile_indexer_topk as _compile_kernel,
)
from attn_gym.testing.indexer import (
    indexer_gvr2_sample_positions as _sample_positions,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="SM100 or SM103 required",
)


def _threads(rows):
    return 256 if rows > IndexerGVR2TopKKernel.wide_row_limit else 512


def _check(scores, topk):
    rows, candidates = scores.shape
    output = torch.full((1, rows, topk), -99, device="cuda", dtype=torch.int32)
    _compile_kernel(topk)(scores.view(rows // 2, 2, candidates), output, cutlass.Int32(0))
    _assert_rows(output[0], scores)
    return output[0]


@pytest.mark.parametrize("rows", [2, 300])
def test_mixed_sign_narrowing_endpoint_overflow(rows):
    """The selected bucket crosses INT32_MAX before intersection with the old interval."""
    candidates, topk = 1024, 37
    sampled = _sample_positions(candidates, topk, _threads(rows))
    expected_samples = (
        torch.arange(64, device="cuda")[:, None] * 16 + torch.arange(8, device="cuda")
    ).flatten()
    assert torch.equal(sampled, expected_samples)
    largest = torch.finfo(torch.float32).max
    row = torch.full((candidates,), -largest, device="cuda")
    row[sampled] = -100
    row[sampled[:20]] = -99
    free = torch.ones(candidates, device="cuda", dtype=torch.bool)
    free[sampled] = False
    free = free.nonzero().flatten()
    row[free[:360]] = -1.0001
    row[free[360:400]] = largest

    # All 400 outliers saturate the same affine bin (>288, so it must narrow).
    low = (~struct.unpack("<i", struct.pack("<f", -1.0001))[0]) ^ -(1 << 31)
    high = struct.unpack("<i", struct.pack("<f", largest))[0]
    shift = max(0, (high - low).bit_length() - 8)
    bucket = (high - low) >> shift
    assert low < 0 < high and shift == 24 and bucket == 191
    assert low + ((bucket + 1) << shift) - 1 > 2**31 - 1
    assert ((row > -96.015625).sum() == 400).item()
    scores = row.expand(rows, -1).contiguous()
    selected = _check(scores, topk)
    assert (scores.gather(1, selected.long()) == largest).all()


@pytest.mark.parametrize("rows", [2, 300])
def test_subnormal_affine_window(rows):
    """Tiny nonzero samples and unsampled subnormal winners survive affine binning."""
    candidates, topk = 1024, 37
    sampled = _sample_positions(candidates, topk, _threads(rows))
    tiny = 2.0**-130
    row = torch.full((candidates,), -tiny, device="cuda")
    row[sampled] = 0
    row[sampled[::2]] = tiny
    free = torch.ones(candidates, device="cuda", dtype=torch.bool)
    free[sampled] = False
    free = free.nonzero().flatten()
    # Build on CPU in FP64, then transfer exact FP32 bit patterns without GPU FTZ arithmetic.
    winners = ((2**19 + torch.arange(1, 401, dtype=torch.float64)) * 2.0**-149).float().cuda()
    assert winners.unique().numel() == 400
    row[free[:400]] = winners
    _check(row.expand(rows, -1).contiguous(), topk)


@pytest.mark.parametrize("rows", [2, 300])
@pytest.mark.parametrize("survivors", [4096, 4097])
@pytest.mark.parametrize("equal", [False, True])
def test_staging_capacity_with_oversized_crossing(rows, survivors, equal):
    """Exact staging capacity and overflow retain counts even when crossing scratch is too small."""
    candidates, topk = 16384, 37
    sampled = _sample_positions(candidates, topk, _threads(rows))
    row = torch.full((candidates,), -torch.finfo(torch.float32).max, device="cuda")
    row[sampled] = -100
    row[sampled[:20]] = -99
    free = torch.ones(candidates, device="cuda", dtype=torch.bool)
    free[sampled] = False
    free = free.nonzero().flatten()
    crossing = survivors - sampled.numel()
    assert crossing > 1024
    row[free[:crossing]] = 1 if equal else torch.linspace(-1, 1, crossing, device="cuda")
    assert (row >= -100).sum().item() == survivors
    assert (row > -96.015625).sum().item() == crossing
    _check(row.expand(rows, -1).contiguous(), topk)


@pytest.mark.parametrize("rows", [2, 300])
@pytest.mark.parametrize("count", [288, 289, 1024, 1025])
def test_crossing_boundary_both_variants(rows, count):
    candidates, topk = 16384, 37
    sampled = _sample_positions(candidates, topk, _threads(rows))
    row = torch.full((candidates,), -100.0, device="cuda")
    row[sampled[:10]] = 2
    free = torch.ones(candidates, device="cuda", dtype=torch.bool)
    free[sampled] = False
    row[free.nonzero().flatten()[:count]] = (
        0.5 + torch.arange(1, count + 1, device="cuda") * 2.0**-24
    )
    _check(row.expand(rows, -1).contiguous(), topk)


@pytest.mark.parametrize("rows", [2, 300])
def test_extreme_changed_input_graph_both_variants(rows):
    candidates, topk = 1024, 37
    scores = torch.zeros((rows, candidates), device="cuda")
    output = torch.empty((1, rows, topk), dtype=torch.int32, device="cuda")
    kernel = _compile_kernel(topk)
    kernel(scores.view(rows // 2, 2, candidates), output, cutlass.Int32(0))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(scores.view(rows // 2, 2, candidates), output, cutlass.Int32(0))
    sampled = _sample_positions(candidates, topk, _threads(rows))
    for seed in (9, 10, 11):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        scores.normal_(generator=generator)
        if seed == 10:
            scores[..., sampled] = -1.7e38
            scores[..., sampled[:20]] = -5e37
            scores[..., 1016:] = torch.finfo(torch.float32).max
        elif seed == 11:
            scores.zero_()
            scores[..., ::3] = -0.0
        output.fill_(-99)
        graph.replay()
        _assert_rows(output[0], scores)


class _ArithmeticProbe:
    def __init__(self, threads):
        self.operation = IndexerGVR2TopKKernel(37, False, block_threads=threads)

    def get_name(self):
        return f"gvr2_arithmetic_probe_{self.operation.block_threads}"

    @cute.jit
    def __call__(
        self,
        values: cute.Tensor,
        lengths: cute.Tensor,
        scales: cute.Tensor,
        bins: cute.Tensor,
        plans: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(values, lengths, scales, bins, plans).launch(
            grid=(1, 1, 1), block=(32, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        values: cute.Tensor,
        lengths: cute.Tensor,
        scales: cute.Tensor,
        bins: cute.Tensor,
        plans: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx < lengths.shape[0]:
            scale = _bin_scale(values[tidx, 2])
            scales[tidx] = scale
            bins[tidx] = _bin_index(values[tidx, 0], values[tidx, 1], scale)
            plan = self.operation.sample_plan(lengths[tidx], lengths[tidx] // 4)
            for item in cutlass.range_constexpr(4):
                plans[tidx, item] = plan[item]


@pytest.mark.parametrize("threads", [256, 512])
def test_binning_and_large_length_arithmetic(threads):
    """Exercise helper arithmetic directly without allocating rows hundreds of millions long."""
    lengths = torch.tensor(
        [1024, 357913941, 357913942, 2**30, 2**31 - 1], device="cuda", dtype=torch.int32
    )
    largest = torch.finfo(torch.float32).max
    values = torch.tensor(
        [
            [0, 0, 0],
            [2.0**-130, 0, 2.0**-130],
            [largest, -largest, float("inf")],
            [largest, largest, 0],
            [1, 0, 1],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    scales = torch.empty(5, device="cuda")
    bins = torch.empty(5, device="cuda", dtype=torch.int32)
    plans = torch.empty((5, 4), device="cuda", dtype=torch.int32)
    fake = lambda dtype, shape: cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=16,
        use_32bit_stride=True,
    )
    previous = get_compile_target()
    try:
        set_compile_target(detect_compile_target(torch.cuda.current_device()))
        fn = compile_tvm_ffi(
            _ArithmeticProbe(threads),
            fake(cutlass.Float32, (5, 3)),
            fake(cutlass.Int32, (5,)),
            fake(cutlass.Float32, (5,)),
            fake(cutlass.Int32, (5,)),
            fake(cutlass.Int32, (5, 4)),
        )
    finally:
        set_compile_target(previous)
    fn(values, lengths, scales, bins, plans)
    assert (torch.isfinite(scales) & (scales > 0)).all()
    assert bins.tolist() == [0, 0, 255, 0, 255]
    for length, actual in zip(lengths.tolist(), plans.tolist()):
        aim = min(max(55, int(math.sqrt(6 * length) + 0.5)), 2048)
        selected = min(max(32 * length // aim, 256), length // 2)
        half = max((length // 4) // 2, 1)
        pairs = min(max(selected // 8, 1), half, threads)
        stride = max(half // pairs, 1)
        sample_threads = min(half // stride, threads)
        assert actual == [
            sample_threads,
            stride,
            max(aim * sample_threads * 8 // length, 1),
            max(37 * sample_threads * 8 // length, 1),
        ]
