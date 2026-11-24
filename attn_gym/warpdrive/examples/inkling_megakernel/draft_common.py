"""Device code shared by the DSpark drafter's megakernel launches (``draft_ctx_kernel``,
``draft_backbone_kernel``, ``draft_head_kernel``): the 8-column GEMV on
``examples.megakernel.multi_gemv``'s tensor-core engine, tagged-word fills of its input rows and
the launches' common plumbing.

The drafter runs ``TOKENS`` (8) token columns through dense bf16 matrices: the block's anchor and
masks, or up to 8 newly committed context rows. ``DraftKernel.gemv`` computes one 32-row x
``k_tiles * 256`` tile of such a product per call with
``examples.megakernel.multi_gemv.gemv_bf16``: the weight ring's 16 KB stages arrive as swizzled TMA
boxes (``ring.swizzled_atom`` into the ring's ``WS`` view) and the input rows ``X`` are padded by
``X_PAD`` columns. The 4 column quarters' partial tiles go through one of two ``GRED`` buffers per
call (one barrier where ``examples.megakernel.multi_gemv.finish`` takes two around a single
buffer) and add in quarter order (deterministic); thread ``tid`` returns tile row ``tid // 8``
against token ``tid % 8``.

Products are exact and accumulate in fp32 (tensor-core order), so results match a bf16 GEMM
within its rounding, not bitwise.
"""

import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute

from attn_gym.warpdrive import (
    NamedBarrier,
    Ring,
    SmemPlan,
    StageView,
    StaticPersistent,
    Tma,
    Warps,
    WarpSpecialized,
    WorkStream,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive.examples.megakernel.constants import (
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    DBUF_FLOATS,
    MMA_N,
    RED_VALUES,
    STAGE_BYTES,
    TILE_K,
    X_PAD,
)
from attn_gym.warpdrive.examples.megakernel.multi_gemv import (
    gemv_bf16,
    partial_sum,
    store_partials,
)
from attn_gym.warpdrive.examples.megakernel.regmath import fp32_reciprocal, group_sum, red_layout
from attn_gym.warpdrive.examples.megakernel.ring import (
    box_layout,
    ring_stages,
    tma_partition,
    weight_stages,
)
from attn_gym.warpdrive.examples.megakernel.table import InstructionTable
from attn_gym.warpdrive.examples.megakernel.tagged import fetch, scratch_offsets, tagged_views

TOKENS = MMA_N  # token columns of every draft GEMV: one mma n8 tile
WORDS = 4  # table words: op, a, b, table row
TOKEN_ROWS = tuple(range(TOKENS))  # the X rows of the GEMV's tokens
GRED_FLOATS = 2 * DBUF_FLOATS  # two buffers of ``multi_gemv``'s quarter partial tiles


def x_layout(width: int) -> cute.Layout:
    """``X``: ``TOKENS`` bf16 input rows of ``width`` columns, padded by ``X_PAD``."""
    return cute.make_layout((TOKENS, width), stride=(width + X_PAD, 1))


def check_draft_sizes(hidden: int):
    """The engine's shape contract for the hidden size."""
    if hidden % TILE_K or hidden % COMPUTE_THREADS:
        raise ValueError(f"hidden size must be a multiple of {TILE_K}")


def compile_draft(op, *fakes):
    """``compile_tvm_ffi`` of a draft launch: the fake tensor arguments ``fakes``, then the trace
    buffer, tagged scratch, state, table, workspace and CTA bound every launch takes last."""
    return compile_tvm_ffi(
        op,
        *fakes,
        make_fake_compact_tensor(Int64, (cute.sym_int(),)),  # trace buffer (globaltimer)
        make_fake_compact_tensor(Int64, (scratch_offsets(op.scratch_shapes)[1],)),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


class DraftKernel(WarpSpecialized):
    """The draft launches' engine and device helpers, on the package's CTA reductions
    (``regmath.cta_sum``), tagged scratch and epoch (``tagged``). A subclass declares its SMEM
    through ``declare_engine``, defines ``scratch_shapes`` (tagged words by buffer, for
    ``examples.megakernel.launch.Launch``), starts its kernel with ``begin_draft`` and launches it
    with ``launch``; its params carry the fields ``begin_draft`` returns."""

    def declare_engine(
        self,
        x_width: int,
        other_bytes: int,
        stages: int | None,
        reductions: int,
        ring_views: tuple[str, ...] = (),
    ):
        """Declares the SMEM every draft launch has, in this order: the weight ring of
        ``stages`` 16 KB stages (default: as many as fit beside the other buffers and the
        subclass's ``other_bytes``) as linear stages ``W`` (what the compute warps read), their
        swizzled-box view ``WS`` (``ring.swizzled_atom``'s loads) and ``ring_views`` (bf16 views
        the subclass lays out), then ``X`` (``x_layout(x_width)``), ``RED`` (``red_layout`` of
        ``reductions`` ``cta_sum`` slots, if any) and ``GRED``; and the ring, the compute warps'
        barrier ``csync`` and the table's work stream."""
        self.x_width, self.reductions = x_width, reductions
        fixed = (
            2 * TOKENS * (x_width + X_PAD)
            + 4 * (reductions * COMPUTE_WARPS * RED_VALUES + GRED_FLOATS)
            + other_bytes
        )
        self.stages = ring_stages(fixed, stages)
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        ring = {"W": w}
        for name in ("WS", *ring_views):
            ring[name] = self.smem_plan.buffer(name, BFloat16, align=1024, overlaps=w)
        self.smem_plan.buffer("X", BFloat16, align=16)
        if reductions:
            self.smem_plan.buffer("RED", Float32, align=16)
        self.smem_plan.buffer("GRED", Float32, align=16)
        self.w = Ring(self.stages, Tma(STAGE_BYTES), Warps, buffer=ring)
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.work = WorkStream(mapping=InstructionTable(words=WORDS), schedule=StaticPersistent())

    def begin_draft(self, bound, layouts: dict, names, tensors, scratch, tma, gtensors, boxes):
        """The kernel's start: ``begin_roles`` with the engine's layouts and the subclass's
        ``layouts``, then ``(trace_buffer, fields, smem)``. ``fields`` are the params every
        launch has: the tensor arguments by ``names`` (rebuilt here, before the roles, so both
        roles' code sees them defined), the tagged scratch views, ``xs``, ``red`` (with
        reductions) and ``gred``, and the TMA atoms ``tma`` with their GMEM tiles ``gmem``
        (``boxes``: each atom's SMEM view and box; the first is ``WS``, as ``sWS``).
        ``smem``: each atom's SMEM partition."""
        engine = {
            "W": weight_stages(self.stages),
            "WS": box_layout(BFloat16, self.stages),
            "X": x_layout(self.x_width),
            "GRED": cute.make_layout((DBUF_FLOATS, 2)),  # two buffers of partial tiles
        }
        if self.reductions:
            engine["RED"] = red_layout(self.reductions)
        self.begin_roles(work=bound, layouts=engine | layouts)
        fields = {k: cute.make_tensor(t.iterator, t.layout) for k, t in zip(names, tensors)}
        trace_buffer = fields.pop("trace_buffer")
        fields |= tagged_views(scratch, self.scratch_shapes)
        parts = [tma_partition(a, self.buffer(name), g, box)
                 for a, g, (name, box) in zip(tma, gtensors, boxes)]  # fmt: skip
        smem = [s for s, _ in parts]
        fields |= {
            "tma": tuple(tma),
            "gmem": tuple(g for _, g in parts),
            "sWS": StageView(smem[0], axis=cute.rank(smem[0]) - 1),
            "xs": self.buffer("X"),
            "gred": self.buffer("GRED"),
        }
        if self.reductions:
            fields["red"] = self.buffer("RED")
        return trace_buffer, fields, smem

    def launch(self, pairs, tensors, scratch, table, workspace, max_ctas, stream):
        """``kernel`` over the TMA ``(atom, tensor)`` ``pairs``, launched cooperatively (its CTAs
        poll each other's tagged words, so the whole grid must be resident)."""
        bound = self.work.bind(table, workspace, max_ctas)
        atoms, gtensors = zip(*pairs)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(atoms, gtensors, scratch, tensors, bound).launch(
            **self.launch_config(bound), stream=stream, cooperative=True
        )

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
        store_partials(acc, p.gred[None, gbuf], warp, lane)
        self.csync.sync()
        return w

    @cute.jit
    def partial(self, p, gbuf, row, n):
        """Tile row ``row`` x token ``n`` after ``gemv_partials``: its quarters summed in
        order."""
        return partial_sum(p.gred[None, gbuf], row, n)

    @cute.jit
    def warp_rinv(self, vals, width):
        """The RMSNorm scale ``rsqrt(mean square + rms_eps)`` of a ``width``-element row whose
        elements the warp's lanes hold (``vals`` each), in every lane."""
        ss = Float32(0.0)
        for i in cutlass.range_constexpr(cute.size(vals)):
            ss = ss + vals[i] * vals[i]
        ss = group_sum(ss, 32)
        return cute.math.rsqrt(ss * fp32_reciprocal(width) + Float32(self.rms_eps))

    @cute.jit
    def fetch_rows(self, buf, base, width: cutlass.Constexpr, out, epoch, lane):
        """Warp-uniform fetch of the ``width`` tagged words (a multiple of 128) at
        ``buf[base ..]``: lane l gets columns ``4 l + 128 j .. + 3`` in ``out[4 j ..]``."""
        segments = [(buf, base + 128 * j + 4 * lane, 4) for j in range(width // 128)]
        fetch(segments, out, epoch, uniform=True)

    @cute.jit
    def store_row(self, xs, n, width: cutlass.Constexpr, vals, scale, weights, lane):
        """``xs[n, 4 l + 128 j + e] = bf16(weights[..] * (vals[4 j + e] * scale))`` for lane l,
        the layout of ``fetch_rows``."""
        for j in cutlass.range_constexpr(width // 128):
            for e in cutlass.range_constexpr(4):
                k = 128 * j + 4 * lane + e
                xs[n, k] = BFloat16(Float32(weights[k]) * (vals[4 * j + e] * scale))
