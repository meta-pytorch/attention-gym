"""Expert activation quantization on the device: vLLM's SM100 NVFP4 and MXFP8 activation
quantizers (``vllm/models/inkling2/nvidia/ops/{nvfp4,mxfp8}.py``, main bf151b7e) bit for bit,
as ``inkling_megakernel.act_quant`` transcribes them in PyTorch. Activations become their
exact bf16 block values (NVFP4 ``e2m1 x e4m3``, MXFP8 ``e4m3 x 2^(e - 127)``), so the bf16 GEMVs
(``gemv``, ``quant_gemv``, ``multi_gemv``) form the quantized products exactly (their fp32 sums
round in each engine's order); NVFP4
scales the fp32 dot products by the row's global scale afterwards (``row_scale``). Rows in
shared memory quantize with ``quantize_rows`` (a thread a block), activations as a fetch holds
them (8 columns a lane) with ``quantize_chunk`` (a block over 2 or 4 lanes, in registers).

- NVFP4 (16-column blocks): the row's ``amax`` (the CTA maximum of ``|x|``, ``cta_max``)
  gives ``encoder = min(1536 / amax, FP32_MAX)``, ``decoder = 1 / encoder`` and ``outer = amax *
  fp32(1 / 1536)`` (``nvfp4_globals``). Per block with ``m = max |x|``: the scales ``s6 =
  e4m3(min(m * (encoder * fp32(1 / 6)), 448))`` and ``s4`` with 0.25 for 1/6, each with values
  ``e2m1(x * min(1 / (s * decoder), FP32_MAX))`` (``cvt.rn.satfinite``, to nearest even,
  saturating at 6) and error ``sum |(e2m1 * s) * outer - x|`` as a pairwise tree over adjacent
  columns; ``s4`` only when its error is strictly smaller.
- MXFP8 (32-column blocks): ``raw = max |x| * fp32(1 / 448)`` rounds up to the power of two
  ``2^(e - 127)`` (the exponent byte ``e``: 0 when ``raw <= 2^-127``, at most 254, 255 for
  NaN), and the values ``x * 2^(127 - e)``, clamped to +-448, round to e4m3; ``e`` 254 and 255
  quantize with 0 (Triton's ``rcp_rn`` flushes ``2^-127``), ``e`` 0 with ``2^127``.

Every rounding step is an explicit ``.rn`` instruction (``mul.rn``, ``add.rn``, ``div.rn``) that
ptxas does not contract into FMAs, as vLLM compiles its quantizers (``enable_fp_fusion=False``);
CuTeDSL's own arithmetic emits ``mul.f32``/``add.f32``, which ptxas may contract. Maxima use
``max.f32``, which drops NaNs as Triton's do (a block's maximum is NaN only when all its values
are).
"""

import cutlass
from cutlass import BFloat16, Float32, Int32, cute
from cutlass._mlir.extras import types as T_
from cutlass._mlir_helpers.arith import bitcast
from cutlass.cutlass_dsl import T, dsl_user_op

from attn_gym.warpdrive.examples.megakernel.constants import COMPUTE_THREADS, COMPUTE_WARPS
from attn_gym.warpdrive.examples.megakernel.formats import BLOCK_COLUMNS, NVFP4
from attn_gym.warpdrive.examples.megakernel.quant_gemv import inline_asm
from attn_gym.warpdrive.examples.megakernel.regmath import bf16x8_to_f32, fp32

FP32_MAX = 3.4028234663852886e38
E4M3_MAX = 448.0
NVFP4_GLOBAL = 1536.0  # the largest E2M1 value 6 times the e4m3 scale 256


@dsl_user_op
def f32_op(op: str, a, b, *, loc=None, ip=None) -> Float32:
    """``op`` (``mul.rn``, ``add.rn``, ``sub.rn``, ``div.rn``, ``max``, ``min``) of two fp32
    values as one PTX instruction."""
    out = inline_asm(
        [T.f32()], [Float32(a), Float32(b)], f"{op}.f32 $0, $1, $2;", "=f,f,f", loc, ip
    )
    return Float32(out[0])


@dsl_user_op
def abs_f32(a, *, loc=None, ip=None) -> Float32:
    return Float32(inline_asm([T.f32()], [Float32(a)], "abs.f32 $0, $1;", "=f,f", loc, ip)[0])


@dsl_user_op
def e2m1x2(even, odd, *, loc=None, ip=None) -> tuple:
    """``(values, codes)`` of two fp32 values rounded to E2M1 (``cvt.rn.satfinite``): their
    values as a bf16x2 word and their codes as a byte, ``even`` in the low half and nibble."""
    out = inline_asm(
        [T.i32(), T.i32()],
        [Float32(even), Float32(odd)],
        "{\n"
        ".reg .b8 c;\n"
        "cvt.rn.satfinite.e2m1x2.f32 c, $3, $2;\n"  # the first operand fills the high nibble
        "cvt.rn.bf16x2.e2m1x2 $0, c;\n"
        "cvt.u32.u8 $1, c;\n"
        "}",
        "=r,=r,f,f",
        loc,
        ip,
    )
    return Int32(out[0]), Int32(out[1])


@dsl_user_op
def e4m3x2(lo, hi, *, loc=None, ip=None) -> tuple:
    """``(values, bytes)`` of two fp32 values rounded to e4m3 (``cvt.rn.satfinite``): their
    values as a bf16x2 word (exact) and their two bytes, ``lo`` in the low half and byte."""
    out = inline_asm(
        [T.i32(), T.i32()],
        [Float32(lo), Float32(hi)],
        "{\n"
        ".reg .b16 h;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h, $3, $2;\n"
        "cvt.rn.bf16x2.e4m3x2 $0, h;\n"
        "cvt.u32.u16 $1, h;\n"
        "}",
        "=r,=r,f,f",
        loc,
        ip,
    )
    return Int32(out[0]), Int32(out[1])


@dsl_user_op
def mul_bf16x2(a, b, *, loc=None, ip=None) -> Int32:
    out = inline_asm(
        [T.i32()], [Int32(a), Int32(b)], "mul.rn.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip
    )
    return Int32(out[0])


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None) -> Int32:
    """``lo`` and ``hi`` (fp32) rounded to bf16 and packed, ``lo`` in the low half."""
    out = inline_asm([T.i32()], [Float32(lo), Float32(hi)], "cvt.rn.bf16x2.f32 $0, $2, $1;",
                     "=r,f,f", loc, ip)  # fmt: skip
    return Int32(out[0])


def as_float(bits) -> Float32:
    return Float32(bitcast(Int32(bits).ir_value(), T_.f32()))


def as_bits(x) -> Int32:
    return Int32(bitcast(Float32(x).ir_value(), T_.i32()))


def halves(word):
    """The fp32 values of a bf16x2 word's low and high halves."""
    return as_float(word << 16), as_float(word & Int32(-65536))


def block_max(x, n: int, lanes: int = 1) -> Float32:
    """``max |x|`` over ``n`` fp32 values per lane and the ``lanes`` neighbouring lanes of an
    aligned group, NaNs dropped unless all are NaN."""
    m = Float32(float("nan"))
    for i in range(n):
        m = f32_op("max", m, abs_f32(x[i]))
    return across_lanes("max", m, lanes)


def across_lanes(op: str, v, lanes: int) -> Float32:
    """``op`` (``max``, ``add.rn``) of ``v`` over an aligned group of ``lanes`` lanes, as a
    butterfly: for ``add.rn`` the pairwise tree over the lanes in order (fp32 addition
    commutes, so every lane gets the same sum). Every lane of the warp must call it."""
    for step in range(lanes.bit_length() - 1):
        v = f32_op(op, v, cute.arch.shuffle_sync_bfly(v, offset=1 << step))
    return v


def nvfp4_globals(amax) -> tuple:
    """``(encoder / 6, encoder / 4, decoder, outer)`` of a row with maximum ``amax``: the
    encoder pre-multiplied by each candidate's factor, as vLLM's per-block products are."""
    encoder = f32_op("min", f32_op("div.rn", Float32(NVFP4_GLOBAL), amax), Float32(FP32_MAX))
    decoder = f32_op("div.rn", Float32(1.0), encoder)
    return (
        f32_op("mul.rn", encoder, Float32(fp32(1.0 / 6.0))),
        f32_op("mul.rn", encoder, Float32(0.25)),
        decoder,
        f32_op("mul.rn", amax, Float32(fp32(1.0 / NVFP4_GLOBAL))),
    )


def pairwise_sum(values: list) -> Float32:
    """fp32 sum of ``values`` (a power of two of them) as a pairwise tree over adjacent ones."""
    while len(values) > 1:
        values = [f32_op("add.rn", values[i], values[i + 1]) for i in range(0, len(values), 2)]
    return values[0]


def nvfp4_candidate(x, n: int, lanes: int, scale, decoder, outer) -> tuple:
    """``(values, codes, error)`` of a block's ``n`` fp32 values ``x`` per lane quantized with
    block scale ``scale`` (fp32 value of an e4m3): ``n / 2`` bf16x2 words of E2M1 values and
    code bytes, and the block's pairwise-tree error sum over its ``lanes`` lanes."""
    inverse = f32_op(
        "min", f32_op("div.rn", Float32(1.0), f32_op("mul.rn", scale, decoder)), Float32(FP32_MAX)
    )
    words, codes, errors = [], [], []
    for j in range(n // 2):
        a, b = x[2 * j], x[2 * j + 1]
        word, code = e2m1x2(f32_op("mul.rn", a, inverse), f32_op("mul.rn", b, inverse))
        words.append(word)
        codes.append(code)
        for q, v in zip(halves(word), (a, b)):
            back = f32_op("mul.rn", f32_op("mul.rn", q, scale), outer)
            errors.append(abs_f32(f32_op("sub.rn", back, v)))
    return words, codes, across_lanes("add.rn", pairwise_sum(errors), lanes)


def nvfp4_block(x, globals_, lanes: int = 1) -> tuple:
    """``(values, codes, scale byte)`` of a 16-column block of fp32 values (bf16 activations)
    quantized to NVFP4 with the row's ``nvfp4_globals``: ``x`` holds the lane's ``16 / lanes``
    consecutive columns, the block's ``lanes`` neighbouring lanes of an aligned group in column
    order (every lane of the warp must call it when ``lanes`` > 1). Returns the lane's bf16x2
    words of exact block values ``e2m1 x scale`` and code bytes (the even column in the low
    nibble), and the block's e4m3 scale byte."""
    enc6, enc4, decoder, outer = globals_
    n = 16 // lanes
    m = block_max(x, n, lanes)
    c6 = f32_op("min", f32_op("mul.rn", m, enc6), Float32(E4M3_MAX))
    c4 = f32_op("min", f32_op("mul.rn", m, enc4), Float32(E4M3_MAX))
    scale_word, scale_bytes = e4m3x2(c6, c4)
    s6, s4 = halves(scale_word)
    words6, codes6, error6 = nvfp4_candidate(x, n, lanes, s6, decoder, outer)
    words4, codes4, error4 = nvfp4_candidate(x, n, lanes, s4, decoder, outer)
    four = error4 < error6
    # The chosen scale's bf16 bits in both halves: the block values are exact bf16 products.
    half = cutlass.select_(four, scale_word >> 16, scale_word & 0xFFFF) & 0xFFFF
    scale2 = half | (half << 16)
    values = [mul_bf16x2(cutlass.select_(four, w4, w6), scale2) for w6, w4 in zip(words6, words4)]
    codes = [cutlass.select_(four, c4, c6) for c6, c4 in zip(codes6, codes4)]
    byte = cutlass.select_(four, scale_bytes >> 8, scale_bytes) & 0xFF
    return values, codes, byte


def mxfp8_block(x, lanes: int = 1) -> tuple:
    """``(values, codes, scale byte)`` of a 32-column block of fp32 values quantized to MXFP8,
    ``x`` the lane's ``32 / lanes`` columns as in ``nvfp4_block``: the lane's bf16x2 words of
    the block values ``e4m3 x 2^(e - 127)`` (rounded to bf16 as vLLM's GEMM operands are: exact
    unless ``e < 3``) and words of e4m3 byte pairs, and the block's exponent byte ``e``."""
    n = 32 // lanes
    raw = f32_op("mul.rn", block_max(x, n, lanes), Float32(fp32(1.0 / E4M3_MAX)))
    bits = as_bits(raw)
    exponent = ((bits + 0x007FFFFF) & 0x7F800000) >> 23  # round up to a power of two
    exponent = cutlass.select_(raw <= Float32(2.0**-127), Int32(0), cutlass.min(exponent, 254))
    exponent = cutlass.select_((bits & 0x7FFFFFFF) > 0x7F800000, Int32(255), exponent)  # NaN
    inverse = as_float((254 - exponent) << 23)  # 2^(127 - e), exact for 1 <= e <= 253
    inverse = cutlass.select_(exponent == 0, Float32(2.0**127), inverse)
    inverse = cutlass.select_(exponent >= 254, Float32(0.0), inverse)
    # 2^(e - 127); e = 0 is the subnormal 2^-127.
    scale = cutlass.select_(exponent == 0, as_float(Int32(0x00400000)), as_float(exponent << 23))
    values, codes = [], []
    for j in range(n // 2):
        pair = []
        for i in range(2):
            v = f32_op("max", f32_op("mul.rn", x[2 * j + i], inverse), Float32(-E4M3_MAX))
            pair.append(f32_op("min", v, Float32(E4M3_MAX)))  # (after max: no NaN left)
        word, code = e4m3x2(pair[0], pair[1])
        lo, hi = halves(word)
        values.append(pack_bf16x2(f32_op("mul.rn", lo, scale), f32_op("mul.rn", hi, scale)))
        codes.append(code)
    return values, codes, exponent


@cute.jit
def cta_max(red, csync, vals, slot: cutlass.Constexpr, warp, lane):
    """Maxima of each of ``vals`` (at most RED_VALUES per-thread fp32 values, NaNs dropped)
    over the compute threads, in every thread: ``regmath.cta_sum``'s protocol (slot ``slot``
    of ``red``, the barrier ``csync``) with ``max.f32``."""
    n = len(vals)
    for i in cutlass.range_constexpr(n):
        v = vals[i]
        for step in cutlass.range_constexpr(5):
            v = f32_op("max", v, cute.arch.shuffle_sync_bfly(v, offset=1 << step))
        if lane == 0:
            red[slot, warp, i] = v
    csync.sync()
    out = []
    for i in cutlass.range_constexpr(n):
        m = red[slot, 0, i]
        for wi in cutlass.range_constexpr(1, COMPUTE_WARPS):
            m = f32_op("max", m, red[slot, wi, i])
        out.append(m)
    return out


def quantize_chunk(fmt: str, x, amax) -> list:
    """The 4 bf16x2 words of block values of a lane's 8 consecutive fp32 activations ``x``
    (columns 8l .. 8l + 7: an activation fetch's layout), half of an NVFP4 block or a quarter
    of an MXFP8 one, the block's other parts in the neighbouring lanes of an aligned group, in
    column order; ``amax``: the row's maximum ``|x|`` (NVFP4). Every lane of the warp must call
    it: a fetch quantizes in registers, without storing and reloading the row."""
    if fmt == NVFP4:
        values, _, _ = nvfp4_block(x, nvfp4_globals(amax), lanes=2)
    else:
        values, _, _ = mxfp8_block(x, lanes=4)
    return values


@cute.jit
def quantize_rows(fmt: cutlass.Constexpr, xs, src, dst, rows: cutlass.Constexpr, count,
                  columns: cutlass.Constexpr, red, slot: cutlass.Constexpr, csync, tid, warp,
                  lane):  # fmt: skip
    """The first ``count`` (at most ``rows``) bf16 rows ``src + r`` of ``xs`` quantized per row
    to ``fmt`` and stored as their block values to rows ``dst + r`` (may be ``src``), by the
    compute threads, ``COMPUTE_THREADS`` blocks a round, each thread holding its blocks across
    the CTA maximum (NVFP4: ``cta_max`` through slot ``slot`` of ``red``). Returns the rows'
    ``amax`` (NVFP4; zeros for MXFP8) in every thread. The caller orders the rows' writes
    before (a barrier) and the block values' readers after (another)."""
    n = BLOCK_COLUMNS[fmt]
    per_row = columns // n
    rounds = -(-rows * per_row // COMPUTE_THREADS)
    x = cute.make_rmem_tensor((rounds, n), Float32)
    x.fill(0.0)  # (blocks past the rows: their masked maxima read zeros)
    maxima = [Float32(0.0)] * rows
    for i in cutlass.range_constexpr(rounds):
        block = i * COMPUTE_THREADS + tid
        r = block // per_row
        if r < count:
            raw = cute.make_rmem_tensor((n,), BFloat16)
            cute.autovec_copy(
                cute.local_tile(xs[src + r, None], (n,), (block - r * per_row,)), raw
            )
            for k in cutlass.range_constexpr(n // 8):
                part = bf16x8_to_f32(cute.local_tile(raw, (8,), (k,)))
                for e in cutlass.range_constexpr(8):
                    x[i, 8 * k + e] = part[e]
        if cutlass.const_expr(fmt == NVFP4):
            m = block_max(x[i, None], n)
            for j in cutlass.range_constexpr(rows):
                hit = (r == j) & (r < count)
                maxima[j] = f32_op("max", maxima[j], cutlass.select_(hit, m, Float32(0.0)))
    if cutlass.const_expr(fmt == NVFP4):
        maxima = cta_max(red, csync, maxima, slot, warp, lane)
    for i in cutlass.range_constexpr(rounds):
        block = i * COMPUTE_THREADS + tid
        r = block // per_row
        if r < count:
            if cutlass.const_expr(fmt == NVFP4):
                amax = maxima[0]
                for j in cutlass.range_constexpr(1, rows):
                    amax = cutlass.select_(r == j, maxima[j], amax)
                values, _, _ = nvfp4_block(x[i, None], nvfp4_globals(amax))
            else:
                values, _, _ = mxfp8_block(x[i, None])
            words = cute.make_rmem_tensor((n // 2,), Int32)
            for j in cutlass.range_constexpr(n // 2):
                words[j] = values[j]
            cute.autovec_copy(
                cute.recast_tensor(words, BFloat16),
                cute.local_tile(xs[dst + r, None], (n,), (block - r * per_row,)),
            )
    return maxima


def row_scale(amax, weight_amax, rcp) -> Float32:
    """vLLM's NVFP4 GEMM output scale ``(amax * weight_amax) * rcp`` (``ops/moe.py:346-353``),
    ``rcp = rcp_rn(1536 * weight_denominator)`` as the host computes it."""
    return f32_op("mul.rn", f32_op("mul.rn", amax, weight_amax), rcp)
