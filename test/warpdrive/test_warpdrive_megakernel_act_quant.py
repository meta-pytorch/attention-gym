"""The megakernels' device activation quantizers (``examples.megakernel.act_quant``) against the
Inkling oracle's (``inkling_megakernel.act_quant``): codes, scale bytes, block values and the
row amax (``input_rows``: random, outliers, 60 decades of magnitude, every E2M1 rounding tie
and 4.0/6.0 candidate tie, zeros, subnormal scales, infinities and NaNs), and
``quantize_rows``' per-row amax, load, store and block distribution over several rows."""

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cutlass import BFloat16, Float32, cute
from cutlass.cute.runtime import from_dlpack

from attn_gym.warpdrive.compat import SmemAllocator
from attn_gym.warpdrive.examples.inkling_megakernel.act_quant import (
    E2M1_MAGNITUDES,
    mxfp8_block_values,
    mxfp8_quantize,
    nvfp4_block_values,
    nvfp4_quantize,
)
from attn_gym.warpdrive.examples.megakernel.act_quant import (
    abs_f32,
    f32_op,
    mxfp8_block,
    nvfp4_block,
    nvfp4_globals,
    quantize_chunk,
    quantize_rows,
)
from attn_gym.warpdrive.examples.megakernel.constants import COMPUTE_THREADS
from attn_gym.warpdrive.examples.megakernel.formats import BLOCK_COLUMNS, MXFP8, NVFP4
from attn_gym.warpdrive.examples.megakernel.regmath import red_layout

pytestmark = pytest.mark.usefixtures("requires_blackwell", "release_cuda_memory")

THREADS = 128
ROWS_AT_ONCE = 3  # rows per quantize_rows call: several rows per thread round, a ragged count
COLUMNS = 2304  # Inkling 2 Mini's hidden size: the gate/up input row
E2M1_TIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def input_rows(columns: int) -> torch.Tensor:
    """Random rows of several magnitudes and outliers, then rows of amax 1536 (encoder and
    decoder 1) whose blocks of maximum 6 meet every E2M1 rounding tie exactly or tie the 4.0- and
    6.0-based candidates, then edge rows: zeros and signed zeros, a tiny amax (subnormal decoder
    and e4m3 block scales), huge values, infinities and NaNs, a block far below the row maximum,
    and exact powers of two at and just above MXFP8 scale boundaries."""
    gen = torch.Generator().manual_seed(columns)
    random = [
        torch.randn(16, columns, generator=gen),
        torch.randn(16, columns, generator=gen) ** 3 * 5,
        torch.randn(33, columns, generator=gen) * torch.logspace(-30, 30, 33)[:, None],
    ]
    choices = torch.tensor(E2M1_TIES + E2M1_MAGNITUDES)
    ties = choices[torch.randint(len(choices), (16, columns), generator=gen)]
    ties = ties * (torch.randint(2, (16, columns), generator=gen) * 2 - 1)
    ties[:, ::16] = 6.0
    ties[:, 0] = 1536.0
    x = torch.randn(12, columns, generator=gen)
    x[0] = 0
    x[1] = -0.0
    x[2, :16] = 0
    x[3] = 1e-38
    x[4] *= 1e-36  # amax near 1536 * 2^-126: the decoder scale is subnormal
    x[5] *= 3e37
    x[6, 5] = float("inf")
    x[6, 40] = -float("inf")
    x[7, 7] = float("nan")
    x[7, 32:64] = float("nan")
    x[8, :16] *= 1e-6  # block scales of 6.0 and 4.0 in the e4m3 subnormals
    x[9] = 448.0 * 2.0 ** torch.arange(columns).remainder(9).sub(4)
    x[10] = x[9] * (1 + 2**-7)  # just above: MXFP8 scales round up
    x[11, ::2] = -x[11, 1::2]
    return torch.cat([*random, ties, x])


@cute.jit
def row_amax(x: cute.Tensor, row, columns: cutlass.Constexpr):
    """The row's maximum ``|x|`` from 0 (NaNs dropped), as the kernels' CTA maximum."""
    m = Float32(0.0)
    for k in cutlass.range(columns, unroll=8):
        m = f32_op("max", m, abs_f32(Float32(x[row, k])))
    return m


@cute.kernel
def blocks_kernel(fmt: cutlass.Constexpr, x: cute.Tensor, values: cute.Tensor,
                  codes: cute.Tensor, scales: cute.Tensor, amax: cute.Tensor):  # fmt: skip
    n = BLOCK_COLUMNS[fmt]
    columns = cute.size(x, mode=[1])
    blocks = columns // n
    i = cute.arch.block_idx()[0] * THREADS + cute.arch.thread_idx()[0]
    if i < cute.size(x, mode=[0]) * blocks:
        row, block = i // blocks, i % blocks
        v = cute.make_rmem_tensor((n,), Float32)
        for k in cutlass.range_constexpr(n):
            v[k] = Float32(x[row, block * n + k])
        if cutlass.const_expr(fmt == NVFP4):
            m = row_amax(x, row, x.shape[1])
            words, code, byte = nvfp4_block(v, nvfp4_globals(m))
            if block == 0:
                amax[row] = m
            for j in cutlass.range_constexpr(2):  # 8 code bytes as 2 words
                word = code[4 * j]
                for k in cutlass.range_constexpr(1, 4):
                    word = word | (code[4 * j + k] << (8 * k))
                codes[i, j] = word
        else:
            words, code, byte = mxfp8_block(v)
            for j in cutlass.range_constexpr(8):  # 16 e4m3 byte pairs as 8 words
                codes[i, j] = code[2 * j] | (code[2 * j + 1] << 16)
        for j in cutlass.range_constexpr(n // 2):
            values[i, j] = words[j]
        scales[i] = byte


@cute.jit
def run_blocks(fmt: cutlass.Constexpr, x: cute.Tensor, values: cute.Tensor, codes: cute.Tensor,
               scales: cute.Tensor, amax: cute.Tensor):  # fmt: skip
    grid = (-(-cute.size(values, mode=[0]) // THREADS), 1, 1)
    blocks_kernel(fmt, x, values, codes, scales, amax).launch(grid=grid, block=(THREADS, 1, 1))


@cute.kernel
def chunks_kernel(fmt: cutlass.Constexpr, x: cute.Tensor, out: cute.Tensor):
    """``quantize_chunk`` of every 8-column chunk of ``x`` (thread i: chunk i, rows after each
    other), as an activation fetch holds them."""
    rows, columns = x.shape
    chunks = columns // 8
    c = cute.arch.block_idx()[0] * THREADS + cute.arch.thread_idx()[0]
    row = cutlass.min(c // chunks, rows - 1)
    live = c < rows * chunks
    v = cute.make_rmem_tensor((8,), Float32)
    for e in cutlass.range_constexpr(8):
        v[e] = Float32(0.0)
        if live:
            v[e] = Float32(x[row, (c - row * chunks) * 8 + e])
    words = quantize_chunk(fmt, [v[e] for e in range(8)], row_amax(x, row, x.shape[1]))
    if live:
        for j in cutlass.range_constexpr(4):
            out[c, j] = words[j]


@cute.jit
def run_chunks(fmt: cutlass.Constexpr, x: cute.Tensor, out: cute.Tensor):
    grid = (-(-cute.size(out, mode=[0]) // THREADS), 1, 1)
    chunks_kernel(fmt, x, out).launch(grid=grid, block=(THREADS, 1, 1))


class CtaBarrier:
    """``quantize_rows``' ``csync`` for a kernel whose block is the compute threads."""

    def sync(self):
        cute.arch.barrier()


@cute.kernel
def rows_kernel(fmt: cutlass.Constexpr, x: cute.Tensor, out: cute.Tensor):
    """``quantize_rows`` over every row of ``x`` at once (``ROWS_AT_ONCE`` a CTA, the last
    CTA's partly past the end) through shared memory, into ``out``."""
    rows, columns = x.shape
    tid = cute.arch.thread_idx()[0]
    smem = SmemAllocator()
    xs = smem.allocate_tensor(
        BFloat16, cute.make_layout((ROWS_AT_ONCE, columns), stride=(columns, 1)), 16
    )
    red = smem.allocate_tensor(Float32, red_layout(1), 16)
    first = cute.arch.block_idx()[0] * ROWS_AT_ONCE
    count = cutlass.min(rows - first, ROWS_AT_ONCE)
    for i in cutlass.range(-(-ROWS_AT_ONCE * columns // COMPUTE_THREADS), unroll=1):
        k = i * COMPUTE_THREADS + tid
        if k < count * columns:
            xs[k // columns, k % columns] = x[first + k // columns, k % columns]
    cute.arch.barrier()
    quantize_rows(fmt, xs, 0, 0, ROWS_AT_ONCE, count, columns, red, 0, CtaBarrier(), tid,
                  tid // 32, tid % 32)  # fmt: skip
    cute.arch.barrier()
    for i in cutlass.range(-(-ROWS_AT_ONCE * columns // COMPUTE_THREADS), unroll=1):
        k = i * COMPUTE_THREADS + tid
        if k < count * columns:
            out[first + k // columns, k % columns] = xs[k // columns, k % columns]


@cute.jit
def run_rows(fmt: cutlass.Constexpr, x: cute.Tensor, out: cute.Tensor):
    grid = (-(-cute.size(x, mode=[0]) // ROWS_AT_ONCE), 1, 1)
    rows_kernel(fmt, x, out).launch(grid=grid, block=(COMPUTE_THREADS, 1, 1))


def device_blocks(fmt: str, x: torch.Tensor) -> tuple:
    """``(values, codes, scales, amax)`` of ``blocks_kernel`` per block of ``x``."""
    rows, columns = x.shape
    n = BLOCK_COLUMNS[fmt]
    count = rows * columns // n
    values = torch.empty(count, n // 2, device="cuda", dtype=torch.int32)
    codes = torch.empty(
        count, n // 8 if fmt == NVFP4 else n // 4, device="cuda", dtype=torch.int32
    )
    scales = torch.empty(count, device="cuda", dtype=torch.int32)
    amax = torch.empty(rows, device="cuda")
    args = [from_dlpack(t) for t in (x, values, codes, scales, amax)]
    cute.compile(run_blocks, fmt, *args)(*args)
    values = values.view(torch.bfloat16).view(rows, columns)
    return values, codes.view(torch.uint8).view(rows, -1), scales.view(rows, -1), amax


@pytest.mark.parametrize("fmt", [NVFP4, MXFP8])
def test_device_quantizers_match_the_oracle(fmt):
    """``input_rows`` at the gate/up input width: codes, scale bytes, block values and (NVFP4)
    the amax bit for bit, and ``quantize_rows`` (a block a thread) and ``quantize_chunk`` (a
    block over 2 or 4 lanes) give the same block values."""
    x = input_rows(COLUMNS).cuda().bfloat16().contiguous()
    values, codes, scales, amax = device_blocks(fmt, x)
    if fmt == NVFP4:
        q = nvfp4_quantize(x)
        expected = (nvfp4_block_values(q.values, q.scales), q.values, q.scales.view(torch.uint8))
        assert torch.equal(amax.view(torch.int32), q.amax.view(torch.int32))
    else:
        e4m3, exponents = mxfp8_quantize(x)
        expected = (mxfp8_block_values(e4m3, exponents), e4m3.view(torch.uint8), exponents)
    assert torch.equal(codes, expected[1])
    assert torch.equal(scales.to(torch.uint8), expected[2])
    assert torch.equal(values.view(torch.int16), expected[0].view(torch.int16))
    out = torch.full_like(x, float("nan"))
    args = [from_dlpack(t) for t in (x, out)]
    cute.compile(run_rows, fmt, *args)(*args)
    assert torch.equal(out.view(torch.int16), expected[0].view(torch.int16))
    out = torch.empty(x.numel() // 8, 4, device="cuda", dtype=torch.int32)
    args = [from_dlpack(t) for t in (x, out)]
    cute.compile(run_chunks, fmt, *args)(*args)
    out = out.view(torch.bfloat16).view_as(x)
    assert torch.equal(out.view(torch.int16), expected[0].view(torch.int16))
