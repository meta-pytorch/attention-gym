"""Helpers of the flash-decoding attention rows: the KV blocks a split reads, and a lane's
slices of a ``HEAD_DIM`` head (``constants``) in the rows' lane layout, where lane group
``lane // 8`` holds a position and lane ``j = lane % 8`` its dims ``8j .. 8j + 7`` and their
rotate-half partners ``64 + 8j .. 64 + 8j + 7``."""

import cutlass
from cutlass import BFloat16, Float32, Int32, cute

from attn_gym.warpdrive.examples.megakernel.constants import BLOCK
from attn_gym.warpdrive.examples.megakernel.regmath import load_vec


@cute.jit
def split_blocks(pos, split, splits: cutlass.Constexpr):
    """``(blk0, blk1, newest)``: the KV blocks ``[blk0, blk1)`` of cached keys (positions
    ``[0, pos)``) that ``split`` of ``splits`` reads, and the split of the last such block,
    which also scores the new token at ``pos`` (split 0 when no key is cached). Later splits
    are empty."""
    last = (pos + BLOCK - 1) // BLOCK
    per = cutlass.max((last + splits - 1) // splits, Int32(1))
    blk0 = split * per
    blk1 = cutlass.max(cutlass.min(blk0 + per, last), blk0)
    newest = cutlass.max(last - 1, Int32(0)) // per
    return blk0, blk1, newest


@cute.jit
def head_vec(src, base, d0):
    """This lane's dims d0..d0+7 and 64+d0..64+d0+7 of a head starting at ``base``."""
    lo = load_vec(src, base + d0, 8)
    hi = load_vec(src, base + 64 + d0, 8)
    out = cute.make_rmem_tensor((16,), Float32)
    for e in cutlass.range_constexpr(8):
        out[e] = lo[e]
        out[8 + e] = hi[e]
    return out


@cute.jit
def stage_row(stage, t, d0):
    """Dims d0..d0+7 and 64+d0..64+d0+7 of position ``t`` of a KV stage, as fp32."""
    out = cute.make_rmem_tensor((16,), Float32)
    for half in cutlass.range_constexpr(2):
        frag = cute.make_rmem_tensor((1, 8), BFloat16)
        cute.autovec_copy(cute.local_tile(stage, (1, 8), (t, d0 // 8 + 8 * half)), frag)
        vals = frag.load().to(Float32)
        for e in cutlass.range_constexpr(8):
            out[8 * half + e] = vals[0, e]
    return out
