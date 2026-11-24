"""Device code shared by the DSpark drafter's megakernel launches (``draft_ctx_kernel``,
``draft_backbone_kernel``, ``draft_head_kernel``): the 8-column GEMV on
``examples.megakernel.multi_gemv``'s tensor-core engine, tagged-word fills of its input rows and
the launches' common plumbing.

The drafter runs ``TOKENS`` (8) token columns through dense bf16 matrices: the block's anchor and
masks, or up to 8 newly committed context rows. ``DraftKernel.gemv`` computes one 32-row x
``k_tiles * 256`` tile of such a product per call with
``examples.megakernel.multi_gemv.gemv_bf16``: the weight ring's 16 KB stages arrive as swizzled TMA
boxes (``gemv_atom`` into the ring's ``WS`` view) and the input rows ``X`` are padded by ``X_PAD``
columns. The 4 column quarters' partial tiles go through one of two ``GRED`` buffers per call (one
barrier where ``examples.megakernel.multi_gemv.finish`` takes two around a single buffer) and add
in quarter order (deterministic); thread ``tid`` returns tile row ``tid // 8`` against token
``tid % 8``.

Products are exact and accumulate in fp32 (tensor-core order), so results match a bf16 GEMM
within its rounding, not bitwise.
"""

import cutlass
from cutlass import BFloat16, Float32, cute

from attn_gym.warpdrive import WarpSpecialized
from attn_gym.warpdrive.examples.megakernel.constants import (
    BACKOFF_NS,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    DBUF_FLOATS,
    ROWS,
    TILE_K,
    X_PAD,
)
from attn_gym.warpdrive.examples.megakernel.multi_gemv import (
    gemv_bf16,
    partial_sum,
    store_partials,
)
from attn_gym.warpdrive.examples.megakernel.ring import (
    box_layout,
    swizzled_atom,
    weight_stages,
)
from attn_gym.warpdrive.examples.megakernel.tagged import fetch

TOKENS = 8  # token columns of every draft GEMV: one mma n8 tile
WORDS = 4  # table words: op, a, b, table row
REDUCTIONS = 3  # ``cta_sum`` slots each draft launch allocates
TOKEN_ROWS = tuple(range(TOKENS))  # the X rows of the GEMV's tokens
GRED_FLOATS = (
    2 * DBUF_FLOATS
)  # two buffers of ``examples.megakernel.multi_gemv``'s quarter partial tiles


def x_layout(width: int) -> cute.Layout:
    """``X``: ``TOKENS`` bf16 input rows of ``width`` columns, padded by ``X_PAD``."""
    return cute.make_layout((TOKENS, width), stride=(width + X_PAD, 1))


def gemv_atom(src) -> tuple:
    """TMA ``(atom, tensor)`` of ``src``'s swizzled ``(ROWS, BOX)`` boxes: ``DraftKernel.gemv``
    weights (``examples.megakernel.ring.load_boxes`` into the ring's ``WS`` view)."""
    return swizzled_atom(src, BFloat16)


class DraftKernel(WarpSpecialized):
    """Device helpers of the draft launches, on the package's CTA reductions
    (``regmath.cta_sum``), tagged scratch and epoch (``tagged``). Subclasses set ``hidden``,
    ``stages`` and ``csync`` (the compute warps' barrier), define ``scratch_shapes`` (tagged
    words by buffer, for ``examples.megakernel.launch.Launch``) and declare the weight ring
    through ``ring_buffers`` and ``ring_layouts``; their params carry ``xs`` (``x_layout``),
    ``gred`` (``GRED_FLOATS`` fp32), ``red`` (``red_layout(REDUCTIONS)``) and ``state``
    (launch epoch, finished CTAs)."""

    def ring_buffers(self) -> dict:
        """The weight ring's SMEM buffers, declared first: its linear stages ``W`` (what the
        compute warps read) and their swizzled-box view ``WS`` (``gemv_atom``'s loads)."""
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        return {"W": w, "WS": self.smem_plan.buffer("WS", BFloat16, align=1024, overlaps=w)}

    def ring_layouts(self) -> dict:
        """``ring_buffers``' layouts over ``stages`` 16 KB stages."""
        return {
            "W": weight_stages(self.stages),
            "WS": box_layout(BFloat16, self.stages),
        }

    @cute.jit
    def gemv(self, p, w, xs, k_tiles: cutlass.Constexpr, gbuf, tid, warp, lane, col0=0,
             xrows=TOKEN_ROWS):  # fmt: skip
        """``(value, w)``: one 32-row tile of ``k_tiles`` weight stages against the ``xs`` rows
        ``xrows`` (one per token) from column ``col0`` (see the module docstring). Thread
        ``tid`` gets tile row ``tid // 8`` x token ``tid % 8``. ``gbuf`` (0 or 1) alternates
        per call: every thread passes this call's barrier after its last read of the other
        buffer."""
        w = self.gemv_partials(p, w, xs, k_tiles, gbuf, warp, lane, col0, xrows)
        return self.partial(p, gbuf, tid // TOKENS, tid % TOKENS), w

    @cute.jit
    def gemv_partials(self, p, w, xs, k_tiles: cutlass.Constexpr, gbuf, warp, lane, col0=0,
                      xrows=TOKEN_ROWS):  # fmt: skip
        """``gemv``'s accumulation: the quarters' partial tiles in GRED buffer ``gbuf``, then a
        compute-warp barrier; returns the advanced ring."""
        acc, w = gemv_bf16(w, xs, xrows, col0, k_tiles, warp, lane)
        store_partials(acc, self.gred_buffer(p, gbuf), warp, lane)
        self.csync.sync()
        return w

    @cute.jit
    def partial(self, p, gbuf, row, n):
        """Tile row ``row`` x token ``n`` after ``gemv_partials``: its quarters summed in
        order."""
        return partial_sum(self.gred_buffer(p, gbuf), row, n)

    @cute.jit
    def gred_buffer(self, p, gbuf):
        """GRED buffer ``gbuf``: ``DBUF_FLOATS`` fp32 of
        ``examples.megakernel.multi_gemv``'s partial tiles."""
        return cute.local_tile(p.gred, (DBUF_FLOATS,), (gbuf,))

    @cute.jit
    def wait_items(self, buf, first, count: cutlass.Constexpr, stride, epoch, warp, lane):
        """Warp 0 polls one word (``first + i * stride``) of each of ``count`` producer items,
        with backoff, so the CTA's full fetch that follows finds its words ready without the
        whole CTA spinning on them; then the compute warps synchronize."""
        if warp == 0:
            rounds = -(-count // 32)
            seen = cute.make_rmem_tensor((rounds,), Float32)
            segments = [
                (buf, first + cutlass.min(lane + 32 * i, count - 1) * stride, 1)
                for i in range(rounds)
            ]
            fetch(segments, seen, epoch, uniform=True, backoff=BACKOFF_NS)
        self.csync.sync()

    @cute.jit
    def fetch_rows(self, buf, base, width: cutlass.Constexpr, out, epoch, lane):
        """Warp-uniform fetch of the ``width`` tagged words (a multiple of 128) at
        ``buf[base ..]``: lane l gets columns ``4 l + 128 j .. + 3`` in ``out[4 j ..]``."""
        segments = [(buf, base + 128 * j + 4 * lane, 4) for j in range(width // 128)]
        fetch(segments, out, epoch, uniform=True)

    @cute.jit
    def store_row(self, xs, n, width: cutlass.Constexpr, vals, scale, weights, lane):
        """``xs[n, 4 l + 128 j + e] = bf16(weights[..] * (vals[4 j + e] * scale))`` (``weights``
        None: unweighted) for lane l, the layout of ``fetch_rows``."""
        for j in cutlass.range_constexpr(width // 128):
            for e in cutlass.range_constexpr(4):
                k = 128 * j + 4 * lane + e
                v = vals[4 * j + e] * scale
                if cutlass.const_expr(weights is not None):
                    v = Float32(weights[k]) * v
                xs[n, k] = BFloat16(v)


def check_draft_sizes(hidden: int):
    """The engine's shape contract for the hidden size."""
    if hidden % TILE_K or hidden % COMPUTE_THREADS:
        raise ValueError(f"hidden size must be a multiple of {TILE_K}")
    if ROWS != 32 or COMPUTE_WARPS != 8:
        raise ValueError("the draft GEMV assumes 32-row stages and 8 compute warps")
