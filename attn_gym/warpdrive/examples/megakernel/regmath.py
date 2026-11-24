"""Register math of the megakernels: bf16 rounding and widening, warp and CTA reductions,
vectorized loads and stores of small fragments, fp32 bit reinterpretation and inline PTX, and
fp32 constants as PyTorch rounds them."""

import operator
import struct

import cutlass
from cutlass import Float32, Int32, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass._mlir_helpers.arith import bitcast
from cutlass.cutlass_dsl import T

from attn_gym.warpdrive.examples.megakernel.constants import COMPUTE_WARPS, RED_VALUES


def as_bits(x) -> Int32:
    """The bits of fp32 ``x``."""
    return Int32(bitcast(Float32(x).ir_value(), T.i32()))


def as_float(bits) -> Float32:
    """The fp32 value of ``bits``."""
    return Float32(bitcast(Int32(bits).ir_value(), T.f32()))


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


@cute.jit
def bf16_round(x):
    """Round to the nearest bf16 (ties to even), kept in fp32. Integer ops instead of F2F: the
    conversion is quarter-rate and the norms round every element."""
    bits = as_bits(x)
    return as_float((bits + 0x7FFF + ((bits >> 16) & 1)) & Int32(-65536))


@cute.jit
def warp_transpose_reduce(acc, lane, n: cutlass.Constexpr):
    """Warp-sum each of ``n`` (a power of two <= 32) per-lane values; lane l ends with the sum
    of value ``l // (32 // n)``. Halving exchanges, then one full reduction: 6 shuffles for
    n = 4 instead of 20."""
    a = []
    for r in cutlass.range_constexpr(n):
        a.append(acc[r])
    halvings = n.bit_length() - 1
    for step in cutlass.range_constexpr(halvings):
        level = 16 >> step
        width = n >> (step + 1)
        hi = (lane & level) != 0
        out = []
        for j in cutlass.range_constexpr(width):
            send = cutlass.select_(hi, a[j], a[j + width])
            keep = cutlass.select_(hi, a[j + width], a[j])
            out.append(keep + cute.arch.shuffle_sync_bfly(send, offset=level))
        a = out
    v = a[0]
    for step in cutlass.range_constexpr(halvings, 5):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=16 >> step)
    return v


@cute.jit
def bf16x8_to_f32(frag):
    """8 bf16 registers as fp32: element 2j = word j << 16, 2j + 1 = word j & 0xffff0000."""
    words = cute.recast_tensor(frag, Uint32)
    bits = cute.make_rmem_tensor((8,), Uint32)
    for j in cutlass.range_constexpr(4):
        bits[2 * j] = words[j] << Uint32(16)
        bits[2 * j + 1] = words[j] & Uint32(0xFFFF0000)
    return cute.recast_tensor(bits, Float32)


def group_reduce(op, v, width: int):
    """``op`` (a binary function) of ``v`` over aligned groups of ``width`` lanes, as a
    butterfly of ascending offsets: for a sum, the pairwise tree over the lanes in order (fp32
    addition commutes, so every lane gets the same sum). Every lane of the warp must call it."""
    for step in range(width.bit_length() - 1):
        v = op(v, cute.arch.shuffle_sync_bfly(v, offset=1 << step))
    return v


def group_sum(v, width: int):
    """Sum over aligned groups of ``width`` lanes."""
    return group_reduce(operator.add, v, width)


def red_layout(slots: int) -> cute.Layout:
    """SMEM layout of ``cta_sum``'s ``slots`` slots: fp32 ``(slots, COMPUTE_WARPS,
    RED_VALUES)``."""
    return cute.make_layout(
        (slots, COMPUTE_WARPS, RED_VALUES), stride=(COMPUTE_WARPS * RED_VALUES, RED_VALUES, 1)
    )


def cta_sum(red, csync, vals, slot: int, warp, lane):
    """Sums of each of ``vals`` (at most RED_VALUES per-thread values) over the compute
    threads, in every thread, through slot ``slot`` of ``red`` (``red_layout``) and the compute
    warps' barrier ``csync``. Before a slot is written again, every thread must pass a barrier
    after its last read (another ``cta_sum`` or ``csync``)."""
    return cta_reduce(operator.add, red, csync, vals, slot, warp, lane, init=0.0)


@cute.jit
def cta_reduce(op, red, csync, vals, slot: cutlass.Constexpr, warp, lane,
               init: cutlass.Constexpr = None):  # fmt: skip
    """``cta_sum`` with the binary function ``op``: each warp's ``group_reduce``, then the
    warps' values in warp order, folded into ``init`` (None: from the first warp's)."""
    n = len(vals)
    for i in cutlass.range_constexpr(n):
        v = group_reduce(op, vals[i], 32)
        if lane == 0:
            red[slot, warp, i] = v
    csync.sync()
    out = []
    for i in cutlass.range_constexpr(n):
        total = red[slot, 0, i]
        if cutlass.const_expr(init is not None):
            total = op(Float32(init), total)
        for wi in cutlass.range_constexpr(1, COMPUTE_WARPS):
            total = op(total, red[slot, wi, i])
        out.append(total)
    return out


@cute.jit
def load_vec(src, offset, n: cutlass.Constexpr):
    """``n`` contiguous elements of a 1-D tensor from ``offset`` (a multiple of n), as fp32."""
    out = cute.make_rmem_tensor((n,), Float32)
    out.store(load_frag(src, offset, n).load().to(Float32))
    return out


@cute.jit
def load_frag(src, offset, n: cutlass.Constexpr):
    """``n`` contiguous elements of a 1-D tensor from ``offset`` (a multiple of n) in one vector
    load, in their type."""
    return load_tile(src, offset // n, n)


@cute.jit
def load_tile(src, tile, n: cutlass.Constexpr):
    """Elements ``n * tile .. n * tile + n - 1`` of a 1-D tensor in one vector load, in their
    type."""
    frag = cute.make_rmem_tensor((n,), src.element_type)
    cute.autovec_copy(cute.local_tile(src, (n,), (tile,)), frag)
    return frag


@cute.jit
def store_vec(dst, offset, frag, n: cutlass.Constexpr):
    cast = cute.make_rmem_tensor((n,), dst.element_type)
    cast.store(frag.load().to(dst.element_type))
    cute.autovec_copy(cast, cute.local_tile(dst, (n,), (offset // n,)))


def fp32(x: float) -> float:
    """``x`` rounded to fp32, as a Python float."""
    return struct.unpack("f", struct.pack("f", x))[0]


def fp32_reciprocal(x: float) -> float:
    """``1 / x`` in fp32: PyTorch divides a CUDA tensor by a Python scalar as a multiplication by
    this reciprocal."""
    return fp32(1.0 / fp32(x))


@cute.jit
def load_words16(row, col):
    """The 4 bf16x2 words of 8 bf16 at ``row[col .. col + 7]`` (``col`` a multiple of 8)."""
    return cute.recast_tensor(load_frag(row, col, 8), Int32)
