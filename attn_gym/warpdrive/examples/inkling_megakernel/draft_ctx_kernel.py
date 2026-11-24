"""The DSpark drafter's context K/V of newly committed positions as one persistent warpdrive
launch (``DraftCtxKernel``) and its compile function; ``draft`` holds the host side.

Implements ``dspark.DSparkDrafter.commit`` for the ``n`` (1 .. ``TOKENS``) rows of bf16
``(n, taps, hidden)`` target features at consecutive positions from ``positions[0]`` (read on
the device), ``f_rnn`` from row ``last`` (read on the device)::

    every CTA   per layer it has items of: mix_rmsnorm of the rows' features (fuse_alpha[layer],
                dflare_norm[layer]) -> bf16 X (TOKENS rows; rows >= n zero, not stored)
    KV          32 rows of the layer's [k_proj_ctx; v_proj_ctx] x hidden (``DraftKernel.gemv``)
                v rows: bf16 -> the draft V cache at the rows' positions
                k rows: bf16 -> tagged words; the head's last quarter (dealt after the others)
                gathers its 128 dims, k_norm_ctx, RoPE -> the draft K cache
    FRNN        the Markov head's f_rnn: mix_rmsnorm of row ``last``'s features (fuse_alpha[-1],
                the head's dflare_norm) -> bf16 (hidden,)

Rows at positions outside ``[0, capacity)`` are not stored.

Roles as ``visit_kernel``: warp 8 ``load`` streams 32 x 256 weight stages (as swizzled boxes,
``ring.swizzled_atom``), warps 0-7 ``compute``. A head's last K quarter waits on the other
three's tagged words, so the launch is cooperative and the table keeps every CTA's rows in
dependency order. The last CTA to finish advances the epoch.
"""

from dataclasses import dataclass

import cutlass
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, cute

from attn_gym.warpdrive import Consumer, Local, Producer, StageView, WorkCursor, role
from attn_gym.warpdrive.cache import jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.inkling_megakernel.draft_common import (
    TOKENS,
    DraftKernel,
    check_draft_sizes,
    compile_draft,
)
from attn_gym.warpdrive.examples.megakernel.constants import (
    BOX,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    HEAD_DIM,
    ROWS,
    SMEM_BYTES,
    TILE_K,
)
from attn_gym.warpdrive.examples.megakernel.regmath import (
    bf16_round,
    cta_sum,
    fp32_reciprocal,
    load_frag,
)
from attn_gym.warpdrive.examples.megakernel.ring import load_boxes, swizzled_atom
from attn_gym.warpdrive.examples.megakernel.tagged import advance_epoch, fetch, publish

KV, FRNN = 0, 1  # table ops: a K/V row tile (a = layer, b = tile), the Markov head's f_rnn
COMPUTE_REGS, LOAD_REGS = 232, 40
TENSOR_ARGS = (
    "trace_buffer", "features", "alpha", "dflare", "markov_norm", "k_norm", "rope", "k_cache", "v_cache",
    "f_rnn", "positions", "last", "state",
)  # fmt: skip


@dataclass(frozen=True)
class CtxParams:
    tma: tuple  # TMA atom of the stacked [k_proj_ctx; v_proj_ctx] of every layer
    gmem: tuple  # its tile partition: (TMA, row tiles, k tiles)
    sWS: StageView  # the weight ring's swizzled-box view
    xs: cute.Tensor  # (TOKENS, hidden) bf16 the mixed features of the current layer
    red: cute.Tensor  # ``red_layout(1)`` fp32: markov_context's sum
    gred: cute.Tensor  # (DBUF_FLOATS, 2) fp32 GEMV quarter partials, double-buffered
    k_raw: cute.Tensor  # tagged (layers, kv_heads, HEAD_DIM, TOKENS) raw k projections
    features: cute.Tensor  # (rows * taps, hidden) bf16 committed rows' target features
    alpha: cute.Tensor  # (layers + 1, taps) fp32 softmax(fuse_weight), bf16-rounded
    dflare: cute.Tensor  # (layers, hidden) bf16
    markov_norm: cute.Tensor  # (hidden,) bf16
    k_norm: cute.Tensor  # (layers, HEAD_DIM) bf16 k_norm_ctx
    rope: cute.Tensor  # (max_len, HEAD_DIM) fp32 [cos | sin] of each position
    k_cache: cute.Tensor  # (layers * kv_heads * max_len, HEAD_DIM) bf16
    v_cache: cute.Tensor
    f_rnn: cute.Tensor  # (hidden,) bf16 output
    positions: cute.Tensor  # int64 (rows,) their consecutive positions (element 0 read)
    last: cute.Tensor  # int64 (1,) the row of f_rnn
    state: cute.Tensor  # int32 (2,) launch epoch, finished CTAs


class DraftCtxKernel(DraftKernel):
    """Context K/V of committed rows (see the module docstring). Constructor arguments are the
    drafter's sizes and constants; ``stages``: weight ring depth in 16 KB stages."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        kv_heads: int,
        taps: int,
        rms_eps: float,
        stages: int | None = None,
        smem_bytes: int = SMEM_BYTES,
    ):
        check_draft_sizes(hidden)
        self.hidden, self.layers, self.kv_heads, self.taps = hidden, layers, kv_heads, taps
        self.rms_eps = rms_eps
        self.k_tiles = kv_heads * HEAD_DIM // ROWS  # K row tiles per layer; V tiles follow
        self.layer_tiles = 2 * self.k_tiles
        self.declare_engine(hidden, 0, stages, reductions=1, smem_bytes=smem_bytes)

    @property
    def scratch_shapes(self) -> dict[str, int]:
        return {"k_raw": self.layers * self.kv_heads * HEAD_DIM * TOKENS}

    def lists(self, ctas: int) -> list[list[list[int]]]:
        """Each CTA's rows, dealt round-robin in dependency order: every K quarter but each
        head's last, the V tiles, the heads' last quarters (which gather their heads), then
        FRNN."""
        early = [[KV, layer, t] for layer in range(self.layers) for t in range(self.k_tiles)
                 if t % 4 != 3]  # fmt: skip
        v = [[KV, layer, self.k_tiles + t] for layer in range(self.layers)
             for t in range(self.k_tiles)]  # fmt: skip
        last = [[KV, layer, t] for layer in range(self.layers) for t in range(3, self.k_tiles, 4)]
        rows = early + v + last + [[FRNN, 0, 0]]
        return [rows[c::ctas] for c in range(ctas)]

    def get_name(self) -> str:
        return (
            f"warpdrive_inkling_draft_ctx_h{self.hidden}_l{self.layers}kv{self.kv_heads}"
            f"_t{self.taps}_s{self.stages}{self.trace_key()}"
        )

    @role(warps=COMPUTE_WARPS, regs=LOAD_REGS)
    @cute.jit
    def load(self, p: CtxParams, *, w: Producer, work: WorkCursor):
        w = w.views(WS=p.sWS)
        while work.valid:
            op, a, b, _row = work.item
            if op == KV:
                tile = a * self.layer_tiles + b
                w = load_boxes(w, p.tma[0], p.gmem[0], tile, 0, self.hidden // TILE_K, "WS")
            work.advance()
        return w, work

    @role(warps=range(COMPUTE_WARPS), regs=COMPUTE_REGS)
    @cute.jit
    def compute(self, p: CtxParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        epoch = p.state[0] + 1
        n_new = p.features.shape[0] // self.taps
        pos0 = Int32(p.positions[0])
        max_len = p.k_cache.shape[0] // (self.layers * self.kv_heads)
        current = Int32(-1)
        gbuf = Int32(0)
        row = tid // TOKENS
        n = tid % TOKENS
        while work.valid:
            op, a, b, _row = work.item
            if op == KV:
                if a != current:
                    self.mix(p, a, n_new, warp, lane)
                    current = a
                value, w = self.gemv(p, w, p.xs, self.hidden // TILE_K, gbuf, tid, warp, lane)
                gbuf = 1 - gbuf
                v = bf16_round(value)
                d = (b % 4) * ROWS + row
                if b >= self.k_tiles:
                    kvh = (b - self.k_tiles) // 4
                    if (n < n_new) & (pos0 + n >= 0) & (pos0 + n < max_len):
                        cache_row = (a * self.kv_heads + kvh) * max_len + pos0 + n
                        p.v_cache[cache_row, d] = BFloat16(v)
                else:
                    kvh = b // 4
                    publish(p.k_raw, ((a * self.kv_heads + kvh) * HEAD_DIM + d) * TOKENS + n,
                            v, epoch)  # fmt: skip
                    if b % 4 == 3:
                        self.norm_rope_head(p, a, kvh, pos0, n_new, max_len, epoch, warp, lane)
            else:
                self.markov_context(p, Int32(p.last[0]), tid, warp, lane)
            work.advance()
        advance_epoch(p.state, epoch, tid)
        return w, work

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def mix(self, p: CtxParams, layer, rows, warp, lane):
        """X = ``dspark.mix_features`` of the ``rows`` rows for ``layer`` (rows past them
        zero): warp n mixes row n, lane l columns ``4 l + 128 j ..``; the taps add in order in
        fp32."""
        h = self.hidden
        chunks = h // 128
        vals = cute.make_rmem_tensor((chunks * 4,), Float32)
        row = cutlass.min(warp, rows - 1)  # rows past ``rows``: loaded in bounds, zeroed below
        for tap in cutlass.range_constexpr(self.taps):
            weight = p.alpha[layer, tap]
            for j in cutlass.range_constexpr(chunks):
                frag = load_frag(p.features[row * self.taps + tap, None], 128 * j + 4 * lane, 4)
                for e in cutlass.range_constexpr(4):
                    if cutlass.const_expr(tap == 0):
                        vals[4 * j + e] = Float32(frag[e]) * weight
                    else:
                        vals[4 * j + e] = vals[4 * j + e] + Float32(frag[e]) * weight
        rinv = cutlass.select_(warp < rows, self.warp_rinv(vals, h), Float32(0.0))
        self.store_row(p.xs, warp, h, vals, rinv, p.dflare[layer, None], lane)
        self.csync.sync()

    @cute.jit
    def norm_rope_head(self, p: CtxParams, layer, kvh, pos0, n_new, max_len, epoch, warp, lane):
        """The kv head's 128 k dims of every row from the four quarters' tagged words: warp n
        takes row n, lane l dims l, l + 32 (first halves) and l + 64, l + 96 (their RoPE
        partners); per-head RMSNorm (k_norm_ctx) rounded to bf16, then RoPE at the row's
        position rounded to bf16 (``dspark.head_norm_rope``), stored for rows < n at
        positions in ``[0, max_len)``."""
        base = (layer * self.kv_heads + kvh) * HEAD_DIM
        x = cute.make_rmem_tensor((4,), Float32)
        fetch(
            [(p.k_raw, (base + lane + 32 * i) * TOKENS + warp, 1) for i in range(4)],
            x,
            epoch,
            uniform=True,
        )
        rinv = self.warp_rinv(x, HEAD_DIM)
        for i in cutlass.range_constexpr(4):
            x[i] = bf16_round(Float32(p.k_norm[layer, lane + 32 * i]) * (x[i] * rinv))
        position = pos0 + warp
        if (warp < n_new) & (position >= 0) & (position < max_len):
            cache_row = (layer * self.kv_heads + kvh) * max_len + position
            for i in cutlass.range_constexpr(2):
                c = p.rope[position, lane + 32 * i]
                s = p.rope[position, HEAD_DIM // 2 + lane + 32 * i]
                x1, x2 = x[i], x[i + 2]
                p.k_cache[cache_row, lane + 32 * i] = BFloat16(x1 * c - x2 * s)
                p.k_cache[cache_row, lane + 32 * i + HEAD_DIM // 2] = BFloat16(x2 * c + x1 * s)

    @cute.jit
    def markov_context(self, p: CtxParams, last, tid, warp, lane):
        """f_rnn: ``dspark.mix_features`` of row ``last`` with the last fuse row and the
        Markov head's norm, every compute thread on columns ``r * 256 + tid``."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        vals = cute.make_rmem_tensor((per,), Float32)
        for tap in cutlass.range_constexpr(self.taps):
            weight = p.alpha[self.layers, tap]
            for r in cutlass.range_constexpr(per):
                f = Float32(p.features[last * self.taps + tap, r * COMPUTE_THREADS + tid])
                if cutlass.const_expr(tap == 0):
                    vals[r] = f * weight
                else:
                    vals[r] = vals[r] + f * weight
        ss = Float32(0.0)
        for r in cutlass.range_constexpr(per):
            ss = ss + vals[r] * vals[r]
        ss = cta_sum(p.red, self.csync, [ss], 0, warp, lane)[0]
        rinv = cute.math.rsqrt(ss * fp32_reciprocal(h) + Float32(self.rms_eps))
        for r in cutlass.range_constexpr(per):
            k = r * COMPUTE_THREADS + tid
            p.f_rnn[k] = BFloat16(Float32(p.markov_norm[k]) * (vals[r] * rinv))
        self.csync.sync()

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(self, tma, gtensors, scratch, tensors, bound):
        trace_buffer, fields, _ = self.begin_draft(
            bound, {}, TENSOR_ARGS, tensors, scratch, tma, gtensors, [("WS", (ROWS, BOX))]
        )
        self.run_roles(CtxParams(**fields), trace_buffer=trace_buffer)

    @cute.jit
    def __call__(
        self,
        wkv: cute.Tensor,
        features: cute.Tensor,
        alpha: cute.Tensor,
        dflare: cute.Tensor,
        markov_norm: cute.Tensor,
        k_norm: cute.Tensor,
        rope: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        f_rnn: cute.Tensor,
        positions: cute.Tensor,
        last: cute.Tensor,
        trace_buffer: cute.Tensor,
        scratch: cute.Tensor,
        state: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        tensors = (trace_buffer, features, alpha, dflare, markov_norm, k_norm, rope, k_cache, v_cache, f_rnn,
                   positions, last, state)  # fmt: skip
        self.launch([swizzled_atom(wkv, BFloat16)], tensors, scratch, table, workspace, max_ctas,
                    stream)  # fmt: skip


@jit_cache
def compile_draft_ctx_kernel(op: DraftCtxKernel):
    """One compile serves every position, committed-row count and cache capacity: they are
    read at run time."""
    bf16, f32 = BFloat16, Float32
    h, layers = op.hidden, op.layers
    cache_rows = cute.sym_int()
    return compile_draft(
        op,
        make_fake_compact_tensor(bf16, (layers * 2 * op.kv_heads * HEAD_DIM, h)),  # wkv
        make_fake_compact_tensor(bf16, (cute.sym_int(), h)),  # features (rows * taps, h)
        make_fake_compact_tensor(f32, (layers + 1, op.taps), assumed_align=4),  # alpha
        make_fake_compact_tensor(bf16, (layers, h)),  # dflare norms
        make_fake_compact_tensor(bf16, (h,)),  # markov norm
        make_fake_compact_tensor(bf16, (layers, HEAD_DIM)),  # k_norm_ctx
        make_fake_compact_tensor(f32, (cute.sym_int(), HEAD_DIM)),  # rope
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),  # k cache
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),  # v cache
        make_fake_compact_tensor(bf16, (h,)),  # f_rnn
        make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), assumed_align=8),  # positions
        make_fake_compact_tensor(cutlass.Int64, (1,), assumed_align=8),  # last
    )
