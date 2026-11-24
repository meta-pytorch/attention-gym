"""The one-token GEMV of quantized expert tiles (``formats``; Inkling's loader packs them as
``inkling_megakernel.quant.QuantExperts``): ``quant_tile`` decodes a lane's weight words in
registers to bf16x2 words and accumulates them.

The decode gives the weights' exact block values (NVFP4 ``e2m1 x e4m3`` without the
per-expert global scale, which the caller applies to the dot product; MXFP8 ``e4m3 x
2^(s - 127)``) and ``accumulate_subtile`` adds them in the order of ``gemv.gemv_tile`` (per
lane, even and odd columns in two fp32 accumulators over the subtiles in order;
``quant_tile`` then sums the two and runs ``warp_transpose_reduce``), so for the same input
the dot products are bit-identical to that GEMV on ``QuantExperts.block_values()`` weights.

``quant_tile`` reads a row tile through two rings: the weight ring's stages as 32-bit words
(view ``"QW"``: ``(ROWS, STAGE_BYTES // (4 * ROWS))``, ``stage_subtiles`` 256-column subtiles
each) and one stage of a meta ring for the tile holding its scale words (view ``"S"``: ``(1,
k, 8 * BLOCKS[fmt])`` for ``k >= subtiles`` subtiles, warp w's ``BLOCKS[fmt]`` words each its
4 rows' scale bytes of one block). The launch declares both rings and loads them in its load
role (Inkling's ``mlp_half``).

Per lane, row and 256-column subtile (columns ``8l .. 8l + 7``), one scale byte s:

- NVFP4: one 32-bit word of checkpoint bytes; ``nvfp4_bf16x8`` converts each byte exactly to
  the bf16 pair of its codes' values (``cvt.rn.bf16x2.e2m1x2``) and multiplies by (s, s)
  (``mul.rn.bf16x2``, exact), the scale byte converted once per row (``nvfp4_scale``: ``prmt``
  and ``cvt.rn.bf16x2.e4m3x2``).
- MXFP8: two words of e4m3 bytes; ``mxfp8_bf16x8`` converts each byte pair exactly
  (``cvt.rn.bf16x2.e4m3x2``) and multiplies by the bf16 power of two ``2 ** (s - 127)``
  (``mul.rn.bf16x2``, exact).

Either is 4 ALU (``cvt``) and 4 FMA-pipe (``mul.rn.bf16x2``) instructions per 8 weights plus
the byte or word extraction. bf16x2 weight words accumulate with ``fma_bf16x2``
(``fma.rn.f32.bf16``, SASS ``FHFMA.BF16``): an fp32 FMA of exact bf16 operands at full FMA
rate, with the same results as FFMA on the widened values and no widening instructions for
weights or input. GB300 pipes (ncu ``sm__inst_executed_pipe_*``, 2026-11-22): PRMT, LOP3 and
F2FP (the ``cvt``) run on the ALU pipe at half rate; FHFMA.BF16 and FFMA alternate between the
fmaheavy and fmalite halves of the FMA pipe (full rate); HFMA2/HMUL2.BF16 take both halves
(half rate). In a register-only GB300 microbenchmark at the visit kernel's 2 warps per SMSP, a
single bf16 multiply per pair plus the 8 FHFMA.BF16 took 23.1 cycles per warp and 8 weights,
17.4 without decode (2026-11-22). In the visit kernel an NVFP4 gate/up row tile took about as
long as a bf16 one (2.3 vs 2.5 us in a globaltimer trace of visit 1 at context 1100, with the
earlier HF-numerics decode).
"""

import cutlass
from cutlass import Float32, Int32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from attn_gym.warpdrive.examples.megakernel.constants import ROWS_PER_WARP
from attn_gym.warpdrive.examples.megakernel.formats import BLOCKS, LANE_WORDS, MXFP8, SUBTILE_WORDS
from attn_gym.warpdrive.examples.megakernel.gemv import reduce_tile

# ------------------------------------------------------------------ register decode


def inline_asm(result_types: list, operands: list, body: str, constraints: str, loc, ip):
    """Pure inline PTX with one or more results."""
    values = [v.ir_value(loc=loc, ip=ip) for v in operands]
    single = len(result_types) == 1
    result = llvm.inline_asm(
        result_types[0] if single else llvm.StructType.get_literal(result_types),
        values,
        body,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if single:
        return [result]
    return [llvm.extractvalue(t, result, [i], loc=loc, ip=ip) for i, t in enumerate(result_types)]


@dsl_user_op
def nvfp4_bf16x8(word, scale, *, loc=None, ip=None) -> list:
    """The 8 bf16 block values ``e2m1 x s`` of a word of NVFP4 checkpoint bytes (byte j: the
    codes of columns 2j and 2j + 1, the even one in the low nibble) as 4 bf16x2 words (columns
    2j, 2j + 1 in word j, the even column low), given the block scale as a bf16x2 word (s, s)
    (``nvfp4_scale``): each byte's code values exactly (``cvt``), then one exact bf16 multiply
    (at most 2 + 4 significant bits, magnitudes in [2^-10, 2688])."""
    out = inline_asm(
        [T.i32()] * 4,
        [Int32(word), Int32(scale)],
        "{\n"
        ".reg .b16 h0, h1, h2, h3;\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 t, v;\n"
        "mov.b32 {h0, h1}, $4;\n"  # bytes 0, 2 are the halves' low bytes
        "shr.b32 t, $4, 8;\n"
        "mov.b32 {h2, h3}, t;\n"  # bytes 1, 3
        "cvt.u8.u16 b0, h0;\n"
        "cvt.u8.u16 b1, h2;\n"
        "cvt.u8.u16 b2, h1;\n"
        "cvt.u8.u16 b3, h3;\n"
        "cvt.rn.bf16x2.e2m1x2 v, b0;\n"
        "mul.rn.bf16x2 $0, v, $5;\n"
        "cvt.rn.bf16x2.e2m1x2 v, b1;\n"
        "mul.rn.bf16x2 $1, v, $5;\n"
        "cvt.rn.bf16x2.e2m1x2 v, b2;\n"
        "mul.rn.bf16x2 $2, v, $5;\n"
        "cvt.rn.bf16x2.e2m1x2 v, b3;\n"
        "mul.rn.bf16x2 $3, v, $5;\n"
        "}",
        "=&r,=&r,=&r,=&r,r,r",  # early clobber: $0 is written before $5 is last read
        loc,
        ip,
    )
    return [Int32(v) for v in out]


@dsl_user_op
def mxfp8_bf16x8(word0, word1, scale, *, loc=None, ip=None) -> list:
    """The 8 bf16 weights ``e4m3 * 2 ** (s - 127)`` of two words of e4m3 bytes (byte k of
    ``word0`` is column k, of ``word1`` column 4 + k) as 4 bf16x2 words, given the scale as a
    bf16x2 word ``2 ** (s - 127)`` twice: an exact ``cvt`` to bf16, then an exact bf16
    multiply (the product is a normal number for scale bytes >= ``formats.MIN_MX_SCALE``)."""
    out = inline_asm(
        [T.i32()] * 4,
        [Int32(word0), Int32(word1), Int32(scale)],
        "{\n"
        ".reg .b16 a, b, c, d;\n"
        ".reg .b32 v;\n"
        "mov.b32 {a, b}, $4;\n"
        "mov.b32 {c, d}, $5;\n"
        "cvt.rn.bf16x2.e4m3x2 v, a;\n"
        "mul.rn.bf16x2 $0, v, $6;\n"
        "cvt.rn.bf16x2.e4m3x2 v, b;\n"
        "mul.rn.bf16x2 $1, v, $6;\n"
        "cvt.rn.bf16x2.e4m3x2 v, c;\n"
        "mul.rn.bf16x2 $2, v, $6;\n"
        "cvt.rn.bf16x2.e4m3x2 v, d;\n"
        "mul.rn.bf16x2 $3, v, $6;\n"
        "}",
        "=&r,=&r,=&r,=&r,r,r,r",  # early clobber: $0 is written before $6 is last read
        loc,
        ip,
    )
    return [Int32(v) for v in out]


@dsl_user_op
def fma_bf16x2(acc0, acc1, w, x, *, loc=None, ip=None):
    """``(acc0 + w.lo * x.lo, acc1 + w.hi * x.hi)`` for bf16x2 words, each one fp32 FMA."""
    out = inline_asm(
        [T.f32(), T.f32()],
        [Float32(acc0), Float32(acc1), Int32(w), Int32(x)],
        "{\n"
        ".reg .b16 w0, w1, x0, x1;\n"
        "mov.b32 {w0, w1}, $4;\n"
        "mov.b32 {x0, x1}, $5;\n"
        "fma.rn.f32.bf16 $0, w0, x0, $2;\n"
        "fma.rn.f32.bf16 $1, w1, x1, $3;\n"
        "}",
        "=&f,=&f,f,f,r,r",  # early clobber: $0 is written before $3 is read
        loc,
        ip,
    )
    return Float32(out[0]), Float32(out[1])


@dsl_user_op
def scale_byte(scales, r: int, *, loc=None, ip=None) -> Int32:
    """Byte ``r`` of a lane's scale word (row ``r`` of its 4): one ``prmt``."""
    out = inline_asm(
        [T.i32()], [Int32(scales)], f"prmt.b32 $0, $1, 0, {0x4440 + r:#x};", "=r,r", loc, ip
    )
    return Int32(out[0])


@dsl_user_op
def nvfp4_scale(scales, r: int, *, loc=None, ip=None) -> Int32:
    """bf16x2 (s, s) of e4m3 byte ``r`` of a lane's scale word: the byte twice (``prmt``),
    then an exact ``cvt``."""
    out = inline_asm(
        [T.i32()],
        [Int32(scales)],
        "{\n"
        ".reg .b32 t;\n"
        ".reg .b16 lo, hi;\n"
        f"prmt.b32 t, $1, 0, {0x4400 + 0x11 * r:#x};\n"
        "mov.b32 {lo, hi}, t;\n"
        "cvt.rn.bf16x2.e4m3x2 $0, lo;\n"
        "}",
        "=r,r",
        loc,
        ip,
    )
    return Int32(out[0])


@cute.jit
def mx_scale(scales, r: cutlass.Constexpr) -> Int32:
    """bf16x2 ``2 ** (s - 127)`` twice for byte ``r`` of a lane's scale word (s >= 1)."""
    return scale_byte(scales, r) * Int32(0x00800080)


@cute.jit
def accumulate_subtile(fmt: cutlass.Constexpr, acc, sW, row0, word0, xw, scales, lane):
    """Add one 256-column subtile to ``acc`` (``(ROWS_PER_WARP, 2)`` fp32: the even and odd
    columns of the lane's rows ``row0 ..``). ``sW``: the stage's ``(32, words)`` Int32
    weights, this subtile from word ``word0``; ``xw``: the lane's 8 input columns as 4 bf16x2
    words; ``scales``: the lane's tile-major scale word (its 4 rows' bytes)."""
    n = LANE_WORDS[fmt]
    for r in cutlass.range_constexpr(ROWS_PER_WARP):
        wq = cute.make_rmem_tensor((n,), Int32)
        cute.autovec_copy(cute.local_tile(sW[row0 + r, None], (n,), (word0 // n + lane,)), wq)
        if cutlass.const_expr(fmt == MXFP8):
            words = mxfp8_bf16x8(wq[0], wq[1], mx_scale(scales, r))
        else:
            words = nvfp4_bf16x8(wq[0], nvfp4_scale(scales, r))
        a0, a1 = acc[r, 0], acc[r, 1]
        for j in cutlass.range_constexpr(4):
            a0, a1 = fma_bf16x2(a0, a1, words[j], xw[j])
        acc[r, 0], acc[r, 1] = a0, a1
    return acc


@cute.jit
def quant_tile(fmt: cutlass.Constexpr, w, meta_stage, xs, xrow, stages: cutlass.Constexpr,
               subtiles: cutlass.Constexpr, stage_subtiles: cutlass.Constexpr, warp,
               lane):  # fmt: skip
    """``gemv.gemv_tile`` of a quantized expert row tile against row ``xrow`` of the bf16 input
    rows ``xs``: ``stages`` weight stages of ``stage_subtiles`` 256-column subtiles,
    ``subtiles`` of them real, given the tile's ``meta_stage`` (see the module docstring). The
    same fp32 operations in the same order on the decoded weights, so the same ``(value, row,
    w)`` bit for bit as ``gemv_tile`` on the dequantized weights."""
    blocks = BLOCKS[fmt]
    scales = meta_stage.view("S")
    xwords = cute.recast_tensor(xs[xrow, None], Int32)
    acc = cute.make_rmem_tensor((ROWS_PER_WARP, 2), Float32)
    acc.fill(0.0)
    for st in cutlass.range_constexpr(stages):
        with w.consume() as h:
            stage = h.view("QW")
            for j in cutlass.range_constexpr(stage_subtiles):
                kt = st * stage_subtiles + j
                if cutlass.const_expr(kt < subtiles):
                    xw = cute.make_rmem_tensor((4,), Int32)
                    cute.autovec_copy(cute.local_tile(xwords, (4,), (kt * 32 + lane,)), xw)
                    word = scales[0, kt, warp * blocks + lane // (32 // blocks)]
                    acc = accumulate_subtile(
                        fmt, acc, stage, warp * ROWS_PER_WARP, j * SUBTILE_WORDS[fmt], xw,
                        word, lane,
                    )  # fmt: skip
    v, row = reduce_tile(acc, warp, lane)
    return v, row, w
