"""Register math of the megakernels: bf16 rounding and widening, warp and CTA reductions,
vectorized loads and stores of small fragments, and fp32 constants as PyTorch rounds them."""

import struct

import cutlass
from cutlass import Float32, Int32, Uint32, cute
from cutlass._mlir.extras import types as T
from cutlass._mlir_helpers.arith import bitcast

from attn_gym.warpdrive.examples.megakernel.constants import COMPUTE_WARPS, RED_VALUES


@cute.jit
def bf16_round(x):
    """Round to the nearest bf16 (ties to even), kept in fp32. Integer ops instead of F2F: the
    conversion is quarter-rate and the norms round every element."""
    bits = Int32(bitcast(Float32(x).ir_value(), T.i32()))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & Int32(-65536)
    return Float32(bitcast(bits.ir_value(), T.f32()))


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


@cute.jit
def group_sum(v, width: cutlass.Constexpr):
    """Sum over aligned groups of ``width`` lanes."""
    for step in cutlass.range_constexpr(width.bit_length() - 1):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << step)
    return v


def red_layout(slots: int) -> cute.Layout:
    """SMEM layout of ``cta_sum``'s ``slots`` slots: fp32 ``(slots, COMPUTE_WARPS,
    RED_VALUES)``."""
    return cute.make_layout(
        (slots, COMPUTE_WARPS, RED_VALUES), stride=(COMPUTE_WARPS * RED_VALUES, RED_VALUES, 1)
    )


@cute.jit
def cta_sum(red, csync, vals, slot: cutlass.Constexpr, warp, lane):
    """Sums of each of ``vals`` (at most RED_VALUES per-thread values) over the compute
    threads, in every thread, through slot ``slot`` of ``red`` (``red_layout``) and the compute
    warps' barrier ``csync``. Before a slot is written again, every thread must pass a barrier
    after its last read (another ``cta_sum`` or ``csync``)."""
    n = len(vals)
    for i in cutlass.range_constexpr(n):
        v = group_sum(vals[i], 32)
        if lane == 0:
            red[slot, warp, i] = v
    csync.sync()
    out = []
    for i in cutlass.range_constexpr(n):
        total = Float32(0.0)
        for wi in cutlass.range_constexpr(COMPUTE_WARPS):
            total = total + red[slot, wi, i]
        out.append(total)
    return out


@cute.jit
def load_vec(src, offset, n: cutlass.Constexpr):
    """``n`` contiguous elements of a 1-D tensor from ``offset`` (a multiple of n), as fp32."""
    out = cute.make_rmem_tensor((n,), Float32)
    load_into(out, src, offset, n)
    return out


@cute.jit
def load_into(out, src, offset, n: cutlass.Constexpr):
    frag = cute.make_rmem_tensor((n,), src.element_type)
    cute.autovec_copy(cute.local_tile(src, (n,), (offset // n,)), frag)
    out.store(frag.load().to(Float32))


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
