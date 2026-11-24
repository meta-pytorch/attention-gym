"""The CUDA-core GEMV of the megakernels' weight tiles: one token against a 32-row tile.

Warp w owns rows ROWS_PER_WARP * w .. of each 32 x 256 weight stage and lane l its columns
8l .. 8l + 7; even and odd columns accumulate separately in packed FFMA2s. A warp
transpose-reduce then leaves row r's sum on lane (r % ROWS_PER_WARP) * (32 // ROWS_PER_WARP) of
warp r // ROWS_PER_WARP (``owns_row``), the lanes ``owns_pair`` and ``pair_value`` pair up for
row pairs (2j, 2j + 1)."""

import cutlass
from cutlass import BFloat16, Float32, cute

from attn_gym.warpdrive.examples.megakernel.constants import ROWS_PER_WARP
from attn_gym.warpdrive.examples.megakernel.regmath import bf16x8_to_f32, warp_transpose_reduce


@cute.jit
def gemv_tile(w, xs, xrow, k_tiles, warp, lane):
    """``(value, row, w)``: a tile's dot products against row ``xrow`` of the bf16 input rows
    ``xs`` (``k_tiles`` * 256 columns), consuming ``k_tiles`` stages of the weight ring ``w``
    (whose ``"W"`` view is a bf16 ``(ROWS, TILE_K)`` stage): this lane's tile row and its value
    (valid where ``owns_row``), and the advanced ring."""
    xs = xs[xrow, None]
    acc = cute.make_rmem_tensor((ROWS_PER_WARP, 2), Float32)
    acc.fill(0.0)
    # Fully unrolled: the rolled loop kept acc in a 32-byte stack frame.
    for kt in cutlass.range(k_tiles, unroll_full=True):
        with w.consume() as h:
            sW = h.view("W")
            wf = []
            for r in cutlass.range_constexpr(ROWS_PER_WARP):
                frag = cute.make_rmem_tensor((8,), BFloat16)
                row_w = sW[warp * ROWS_PER_WARP + r, None]
                cute.autovec_copy(cute.local_tile(row_w, (8,), (lane,)), frag)
                wf.append(frag)
            xh = cute.make_rmem_tensor((8,), BFloat16)
            cute.autovec_copy(cute.local_tile(xs, (8,), (kt * 32 + lane,)), xh)
            xv = bf16x8_to_f32(xh)
            for r in cutlass.range_constexpr(ROWS_PER_WARP):
                wv = bf16x8_to_f32(wf[r])
                a0, a1 = acc[r, 0], acc[r, 1]
                for e in cutlass.range_constexpr(0, 8, 2):
                    a0, a1 = cute.arch.fma_packed_f32x2(
                        (wv[e], wv[e + 1]), (xv[e], xv[e + 1]), (a0, a1)
                    )
                acc[r, 0], acc[r, 1] = a0, a1
    for r in cutlass.range_constexpr(ROWS_PER_WARP):
        acc[r, 0] = acc[r, 0] + acc[r, 1]
    v = warp_transpose_reduce(acc[None, 0], lane, ROWS_PER_WARP)
    return v, warp * ROWS_PER_WARP + lane // (32 // ROWS_PER_WARP), w


def owns_row(lane):
    """Whether ``lane`` holds its tile row's value after ``gemv_tile``."""
    return lane % (32 // ROWS_PER_WARP) == 0


def owns_pair(lane):
    """Whether ``lane`` owns an even row 2j, whose partner 2j + 1 ``pair_value`` delivers."""
    return lane % (2 * (32 // ROWS_PER_WARP)) == 0


@cute.jit
def pair_value(v):
    """Row 2j + 1's value on the owner of row 2j (every lane must call it)."""
    return cute.arch.shuffle_sync_down(v, offset=32 // ROWS_PER_WARP)
