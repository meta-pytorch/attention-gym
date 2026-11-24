"""Multi-token GEMVs on tensor cores (Inkling's multi-token launches and DSpark drafter): a
32-row weight tile against up to eight GEMV input rows (tokens) per ``mma.sync.m16n8k16`` (bf16
inputs, fp32 accumulators), the tokens as the MMA's N = 8 columns. Its geometry (``MMA_N``,
``BOX``, ``QBOX``, ``X_PAD``, ``DBUF_FLOATS``) lives in ``constants`` and its TMA boxes in
``ring`` (``load_boxes``, ``box_layout``, ``swizzled_atom``), so that a launch whose module
does not import this one (Inkling's one-token visit) keeps it out of its compile key.

Weights arrive in the weight ring's stages through 128-byte swizzled TMA boxes (``WS``: four
32 x 64 bf16 boxes per stage; ``QWS``: four 32 x 32-word boxes of quantized words;
``ring.box_layout``), whose 16-byte chunk ``k`` of row ``r`` sits at chunk ``k ^ (r % 8)``, so
the eight rows an MMA fragment gathers hit distinct banks. The compute warps read them through
the same swizzled views.

Warp w takes rows ``16 (w % 2) ..`` of the tile (MMA row g at tile row ``ROW_ORDERS[fmt][g]``,
and ``+ 8``) and the 64-column quarter ``w // 2`` of every 256 columns (8 units of 8 columns).
Lane (g, c) = (lane // 4, lane % 4) takes units ``2 c`` and ``2 c + 1`` of each quarter: the
16 columns of one NVFP4 scale block (an MXFP8 block holds 32), so the lane converts its scale
once per row for both. A unit's 8 columns fill the lane's k-slots of two
MMAs (2c, 2c + 1 and 2c + 8, 2c + 9 are the unit's columns 0-3, then 4-7: the dot product's k
order is free as long as A and B agree), for MMA rows g and g + 8, and of input row g (token
g) for B. A bf16 unit is one 16-byte load; a quantized unit decodes in registers
(``quant_gemv``) to the same four bf16x2 words, the weights' block values exactly, so quantized
experts and bf16 experts holding their block values give bit-identical dot products (the MMA
row a tile row lands in does not change its result). ``finish`` sums the four quarters' partial tiles through shared memory in
quarter order; thread t ends with tile row ``t // 8`` against token ``t % 8``.

The row order keeps the rows one shared-memory phase reads in disjoint banks: bf16 (16-byte
loads, two rows a phase) pairs rows differing in bit 0, so chunks ``2c + i`` swizzle apart;
NVFP4 (8-byte loads of a lane's two words, four rows a phase) takes rows of distinct
``r // 2``; MXFP8 (16-byte loads, two rows a phase) pairs rows differing in bit 2.

Input rows (xs) are padded to a stride of 4 mod 32 words (``constants.X_PAD``) so two tokens'
units in one phase use disjoint banks. Rows past the tokens present point at any valid row; their
columns of the result are discarded.
"""

import cutlass
from cutlass import BFloat16, Float32, Int32, cute
from cutlass.cute.nvgpu.warp import LdMatrix8x8x16bOp
from cutlass.cutlass_dsl import T, dsl_user_op

from attn_gym.warpdrive.examples.megakernel.constants import MMA_N
from attn_gym.warpdrive.examples.megakernel.formats import BF16, BLOCKS, MXFP8, NVFP4
from attn_gym.warpdrive.examples.megakernel.quant_gemv import mx_scale, mxfp8_bf16x8, nvfp4_bf16x8
from attn_gym.warpdrive.examples.megakernel.regmath import inline_asm, load_frag, load_words16

# The tile row of MMA row g (of each 8-row half) per weight format, as the ``(shape, stride)``
# of a layout of g (see the module docstring): bf16 0, 1, .. 7; NVFP4 0, 2, 4, 6, 1, 3, 5, 7;
# MXFP8 0, 4, 1, 5, 2, 6, 3, 7.
ROW_ORDERS = {BF16: (8, 1), NVFP4: ((4, 2), (2, 1)), MXFP8: ((2, 4), (4, 1))}


@dsl_user_op
def mma_bf16(acc, a, b, *, loc=None, ip=None) -> list:
    """``acc += A B`` of one ``mma.sync.m16n8k16`` with bf16 operands and fp32 accumulators:
    ``a`` the lane's 4 A registers, ``b`` its 2 B registers, ``acc`` its 4 accumulators."""
    out = inline_asm(
        [T.f32()] * 4,
        [*(Float32(x) for x in acc), *(Int32(x) for x in a), *(Int32(x) for x in b)],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$8, $9, $10, $11}, {$12, $13}, {$4, $5, $6, $7};",
        "=f,=f,=f,=f,f,f,f,f,r,r,r,r,r,r",
        loc,
        ip,
    )
    return [Float32(v) for v in out]


@dsl_user_op
def e4m3_scale(byte, *, loc=None, ip=None) -> Int32:
    """bf16x2 (s, s) of the e4m3 scale byte ``byte`` (an exact ``cvt``)."""
    out = inline_asm(
        [T.i32()],
        [Int32(byte) * Int32(0x0101)],
        "{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, $1;\ncvt.rn.bf16x2.e4m3x2 $0, lo;\n}",
        "=r,r",
        loc,
        ip,
    )
    return Int32(out[0])


@cute.jit
def ldmatrix_x4(row, trans: cutlass.Constexpr):
    """``ldmatrix.m8n8.x4`` (``trans``: ``.trans``): lane 8 i + r passes ``row``, the 8 bf16 of
    row r of matrix i in shared memory; returns the lane's four words."""
    op = LdMatrix8x8x16bOp(transpose=trans, num_matrices=4)
    frag = cute.make_rmem_tensor((8,), BFloat16)
    cute.copy(cute.make_copy_atom(op, BFloat16), row, frag)
    return cute.recast_tensor(frag, Int32)


@cute.jit
def tile_rows(warp, lane, fmt: cutlass.Constexpr = BF16):
    """The tile rows of the lane's MMA rows g and g + 8 for weights in ``fmt``."""
    shape, stride = ROW_ORDERS[fmt]
    row = cute.crd2idx(lane // 4, cute.make_layout(shape, stride=stride)) + 16 * (warp % 2)
    return row, row + 8


@cute.jit
def token_row(xrows, lane):
    """The input row of the lane's token g (``xrows``: MMA_N rows)."""
    g = lane // 4
    row = Int32(xrows[0])
    for i in cutlass.range_constexpr(1, MMA_N):
        row = cutlass.select_(g == i, Int32(xrows[i]), row)
    return row


@cute.jit
def lane_units(quarter, lane):
    """The lane's two units of each 256 columns (of the 32 8-column units)."""
    return [8 * quarter + 2 * (lane % 4) + i for i in range(2)]


@cute.jit
def mma_unit(acc, a0, a1, b):
    """The two MMAs of one unit: A rows g, g + 8 (``a0``, ``a1``: 4 words each), B ``b``."""
    acc = mma_bf16(acc, (a0[0], a1[0], a0[1], a1[1]), (b[0], b[1]))
    return mma_bf16(acc, (a0[2], a1[2], a0[3], a1[3]), (b[2], b[3]))


@cute.jit
def gemv_bf16(w, xs, xrows, col0, k_tiles: cutlass.Constexpr, warp, lane):
    """The lane's accumulators of one tile of bf16 weights (``k_tiles`` 256-column stages of
    the weight ring ``w``) against the xs rows ``xrows`` from column ``col0``; returns ``(acc,
    w)`` for ``finish``."""
    quarter = warp // 2
    r0, r1 = tile_rows(warp, lane)
    xrow = token_row(xrows, lane)
    acc = [Float32(0.0)] * 4
    for kt in cutlass.range(k_tiles, unroll_full=True):
        with w.consume() as h:
            boxes = h.view("WS")  # (ROWS, BOX, boxes): the quarter's columns are box ``quarter``
            for unit in lane_units(quarter, lane):
                a0 = load_words16(boxes[r0, None, quarter], unit % 8 * 8)
                a1 = load_words16(boxes[r1, None, quarter], unit % 8 * 8)
                b = load_words16(xs[xrow, None], col0 + kt * 256 + unit * 8)
                acc = mma_unit(acc, a0, a1, b)
    return acc, w


@cute.jit
def gemv_quant(fmt: cutlass.Constexpr, w, meta_stage, xs, xrows, stages: cutlass.Constexpr,
               subtiles: cutlass.Constexpr, stage_subtiles: cutlass.Constexpr, warp,
               lane):  # fmt: skip
    """``gemv_bf16`` of a quantized expert row tile, given the row's ``meta`` stage: the stages
    and meta views of ``quant_gemv.quant_tile`` (``stages`` weight stages of ``stage_subtiles``
    256-column subtiles, ``subtiles`` of them real), the words in ``QBOX`` swizzled boxes. Each
    unit decodes to the bf16 weights' words and the MMAs run in ``gemv_bf16``'s order on
    them."""
    blocks = BLOCKS[fmt]
    scales = meta_stage.view("S")
    quarter = warp // 2
    r0, r1 = tile_rows(warp, lane, fmt)
    xrow = token_row(xrows, lane)
    units = lane_units(quarter, lane)
    acc = [Float32(0.0)] * 4
    for st in cutlass.range_constexpr(stages):
        with w.consume() as h:
            boxes = h.view("QWS")  # (ROWS, QBOX, boxes) words
            for j in cutlass.range_constexpr(stage_subtiles):
                kt = st * stage_subtiles + j
                if cutlass.const_expr(kt < subtiles):
                    a = [[], []]  # per unit: rows r0, r1
                    for r in (r0, r1):
                        # Both units share a scale block (NVFP4 16, MXFP8 32 columns).
                        word = scales[0, kt, (r // 4) * blocks + units[0] // (32 // blocks)]
                        byte = (word >> (8 * (r % 4))) & 0xFF
                        if cutlass.const_expr(fmt == MXFP8):
                            # A 32-column unit pair is 4 words: subtile j spans boxes 2 j, 2 j + 1.
                            box = boxes[r, None, 2 * j + units[0] // 16]
                            raw = load_frag(box, 2 * (units[0] % 16), 4)
                            scale = mx_scale(byte)
                            for i in cutlass.range_constexpr(2):
                                a[i].append(mxfp8_bf16x8(raw[2 * i], raw[2 * i + 1], scale))
                        else:
                            raw = load_frag(boxes[r, None, j], units[0], 2)
                            scale = e4m3_scale(byte)
                            for i in cutlass.range_constexpr(2):
                                a[i].append(nvfp4_bf16x8(raw[i], scale))
                    for i in cutlass.range_constexpr(2):
                        b = load_words16(xs[xrow, None], kt * 256 + units[i] * 8)
                        acc = mma_unit(acc, a[i][0], a[i][1], b)
    return acc, w


@cute.jit
def finish(acc, dbuf, csync, warp, lane, tid, fmt: cutlass.Constexpr = BF16):
    """Thread ``tid``'s value of the tile: row ``tid // 8`` against token ``tid % 8``, the sum
    of the four quarters' accumulators in quarter order through ``dbuf`` (DBUF_FLOATS fp32).
    Every compute thread calls it; its barriers also order the buffer's reuse. ``fmt``: the
    GEMV's weight format (its row order)."""
    csync.sync()  # the last tile's readers are done
    store_partials(acc, dbuf, warp, lane, fmt)
    csync.sync()
    return partial_sum(dbuf, tid // MMA_N, tid % MMA_N)


@cute.jit
def store_partials(acc, dbuf, warp, lane, fmt: cutlass.Constexpr = BF16):
    """``finish``'s first half: the lane's accumulators to its quarter's partial tile in
    ``dbuf`` (DBUF_FLOATS fp32), unsynchronized."""
    quarter = warp // 2
    c = lane % 4
    r0, r1 = tile_rows(warp, lane, fmt)
    for k in cutlass.range_constexpr(2):
        dbuf[(quarter * 32 + r0) * MMA_N + 2 * c + k] = acc[k]
        dbuf[(quarter * 32 + r1) * MMA_N + 2 * c + k] = acc[2 + k]


@cute.jit
def partial_sum(dbuf, row, token):
    """``finish``'s second half: tile row ``row`` against ``token``, the four quarters'
    partials in ``dbuf`` summed in quarter order (after a barrier past ``store_partials``)."""
    v = dbuf[row * MMA_N + token]
    for q in cutlass.range_constexpr(1, 4):
        v = v + dbuf[(q * 32 + row) * MMA_N + token]
    return v
