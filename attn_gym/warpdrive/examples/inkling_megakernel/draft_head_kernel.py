"""The DSpark drafter's head as one persistent warpdrive launch (``DraftHeadKernel``) and its
compile function: base logits of the block's mask rows from the target's LM head, then the
Markov head's sequential greedy drafts (``dspark.DSparkDrafter.markov_drafts``); ``draft``
holds the host side.

Exact restructuring: the vocabulary's 32-row tiles are dealt to CTAs once, and each CTA keeps
its tiles for every Markov step. The base logits stay in SMEM when they fit beside the weight
ring; otherwise each CTA retains the same tiles in global scratch. Each step's argmax is a
per-CTA best plus a reduction over the CTAs' bests::

    CTXR     32 rnn_proj rows x [h_i] (split 0: the 7 mask states) or x [f_rnn] (split 1)
             -> fp32 partials of the head's context (tagged)
    RNN s    CTA t (t < 24) for row tile t of rnn_proj, every step: the argmax of step s - 1
             (the anchor at s = 0; the CTAs' bests of step s - 1 reduced), its state update
             from step s - 1's raw gate and candidate (kept in SMEM), then 32 rows x
             [state, w1[prev]] -> raw = context + dot, fp32 (tagged); tile 0 writes draft s - 1
    LMH      32 LM-head rows x the mask states -> bf16 base logits; cache each row's fold bias
             in SMEM, then process the tile's step-0 fold rows (as FOLD)
    ARGMAX   after a CTA's LMH rows: its step-0 best (value, then lowest index) -> tagged
    FOLD s   every tile of the CTA (s >= 1), one item: 32 fold rows x bf16 tanh(bf16(out_s))
             -> bias = bf16(dot + fold_bias), softcap(bf16(base + bias)) rounded to bf16 ->
             the lanes' running bests; then the CTA's best -> tagged
    FINAL    CTA 0: the last draft

The fold rows (rank 256: one stage per tile) run on CUDA cores, as
``examples.megakernel.gemv.gemv_tile`` with the input in registers: warp w owns rows
``4w .. 4w + 3`` of each stage and needs no CTA barrier, so a step streams the CTA's tiles
back to back.

Roles as ``visit_kernel``: warp 8 ``load`` streams 32 x 256 weight stages (the GEMVs' as
swizzled boxes, ``ring.swizzled_atom``; fold rows linear, with an evict-last L2 policy:
every step rereads them), warps 0-7 ``compute`` (``DraftKernel.gemv``).
The launch is cooperative and every CTA's table rows are in dependency order.
"""

import math
from dataclasses import dataclass

import cutlass
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, Int64, cute

from attn_gym.warpdrive import Consumer, Local, Producer, StageView, WorkCursor, role
from attn_gym.warpdrive.cache import jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.inkling_megakernel.draft_common import (
    TOKENS,
    DraftKernel,
    check_draft_sizes,
    compile_draft,
    engine_bytes,
)
from attn_gym.warpdrive.examples.megakernel.constants import (
    BACKOFF_NS,
    BOX,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    EVICT_LAST,
    ROWS,
    ROWS_PER_WARP,
    SMEM_BYTES,
    SMEM_HEADROOM,
    STAGE_BYTES,
    TILE_K,
)
from attn_gym.warpdrive.examples.megakernel.gemv import owns_row, reduce_tile
from attn_gym.warpdrive.examples.megakernel.quant_gemv import fma_bf16x2
from attn_gym.warpdrive.examples.megakernel.regmath import (
    bf16_round,
    fp32_reciprocal,
    load_words16,
)
from attn_gym.warpdrive.examples.megakernel.ring import (
    load_boxes,
    load_tiles,
    swizzled_atom,
    tma_atom,
)
from attn_gym.warpdrive.examples.megakernel.tagged import advance_epoch, fetch, publish

CTXR, RNN, LMHEAD, FOLD, ARGMAX, FINAL = range(6)
MAX_CTAS = 256  # bound of the grid whose per-CTA bests the argmax reads
COMPUTE_REGS, LOAD_REGS = 232, 40
ARG_FLOATS = 2 * COMPUTE_WARPS + 2
TENSOR_ARGS = (
    "trace_buffer", "hidden", "f_rnn", "rnn_bias", "markov_w1", "fold_bias", "token", "drafts", "state",
)  # fmt: skip
# X keys: which input the X rows hold (the RNN's XR overlaps them).
X_HIDDEN, X_FRNN, X_RNN = 0, 1, 16


def no_best() -> tuple:
    """The running best before any row: ``(value, index, largest logit before the softcap)``."""
    return Float32(-math.inf), Int32(0), Float32(-math.inf)


@dataclass(frozen=True)
class HeadParams:
    tma: tuple  # TMA atoms: rnn_proj, LM head, fold weight
    gmem: tuple
    sWS: StageView  # the weight ring's swizzled-box view (GEMV rows)
    sW: StageView  # its linear view (fold rows)
    xs: cute.Tensor  # (TOKENS, hidden) bf16 GEMV inputs
    xr: cute.Tensor  # (1, 2 rank) bf16 the RNN's input row [state, w1[prev]], overlapping xs
    gred: cute.Tensor
    base: cute.Tensor  # (slots or row_tiles, ROWS, drafts) bf16 base logits
    fbias: cute.Tensor  # (slots, ROWS) bf16 the fold bias of its rows
    rnn_state: cute.Tensor  # (rank,) fp32 the Markov state (bf16 values), RNN CTAs
    arg: cute.Tensor  # (ARG_FLOATS,) fp32 argmax reduction scratch
    ctxr: cute.Tensor  # tagged (2, TOKENS, 3 rank) context partials
    raw: cute.Tensor  # tagged (drafts, 3 rank) raw gate | candidate | out per step
    best: cute.Tensor  # tagged (drafts, MAX_CTAS, 2) per-CTA best (value, index) per step
    hidden: cute.Tensor  # (TOKENS, hidden) bf16 mask rows' final states (row drafts.. unread)
    f_rnn: cute.Tensor  # (hidden,) bf16
    rnn_bias: cute.Tensor  # (3 rank,) fp32
    markov_w1: cute.Tensor  # (padded vocab, rank) bf16
    fold_bias: cute.Tensor  # (vocab,) bf16
    token: cute.Tensor  # int64 (1,) the anchor
    drafts: cute.Tensor  # int64 (drafts,) output
    state: cute.Tensor  # int32 (2,) launch epoch, finished CTAs


class DraftHeadKernel(DraftKernel):
    """Base logits and the Markov head's drafts (see the module docstring). Constructor
    arguments are the sizes and constants the kernel reads; ``slots``: LM-head tiles per CTA
    (``BASE`` capacity); ``stages``: weight ring depth in 16 KB stages."""

    def __init__(
        self,
        hidden: int,
        vocab: int,
        rank: int,
        drafts: int,
        softcap: float,
        slots: int,
        stages: int | None = None,
        smem_bytes: int = SMEM_BYTES,
    ):
        check_draft_sizes(hidden)
        if rank != COMPUTE_THREADS or not 1 <= drafts < TOKENS:
            raise ValueError(f"need rank {COMPUTE_THREADS} and 1 <= drafts < {TOKENS}")
        self.hidden, self.vocab, self.rank, self.drafts = hidden, vocab, rank, drafts
        self.softcap, self.slots = softcap, slots
        self.row_tiles = -(-vocab // ROWS)
        self.rnn_tiles = 3 * rank // ROWS
        other_bytes = 4 * (rank + ARG_FLOATS) + 2 * slots * ROWS  # STATE, ARG, FBIAS
        base_bytes = 2 * slots * ROWS * self.drafts
        # fold_tiles needs two live weights. Spill BASE if both stages would not fit.
        self.global_base = (
            engine_bytes(hidden, 0) + other_bytes + base_bytes + 2 * STAGE_BYTES + SMEM_HEADROOM
            > smem_bytes
        )
        if not self.global_base:
            other_bytes += base_bytes
        self.declare_engine(hidden, other_bytes, stages, reductions=0, smem_bytes=smem_bytes)
        self.smem_plan.buffer("XR", BFloat16, align=16, overlaps="X")
        for name in ("STATE", "ARG"):
            self.smem_plan.buffer(name, Float32, align=16)
        if not self.global_base:
            self.smem_plan.buffer("BASE", BFloat16, align=16)
        self.smem_plan.buffer("FBIAS", BFloat16, align=16)

    @property
    def scratch_shapes(self) -> dict[str, int]:
        return {
            **({"base_words": self.row_tiles * ROWS * self.drafts // 4} if self.global_base else {}),
            "ctxr": 2 * TOKENS * 3 * self.rank,
            "raw": self.drafts * 3 * self.rank,
            "best": self.drafts * MAX_CTAS * 2,
        }

    def get_name(self) -> str:
        return (
            f"warpdrive_inkling_draft_head_h{self.hidden}_v{self.vocab}_r{self.rank}"
            f"_d{self.drafts}_b{self.slots}_s{self.stages}{self.trace_key()}"
        )

    def lists(self, ctas: int) -> list[list[list[int]]]:
        """Each CTA's rows in dependency order (module docstring). Vocabulary tile t goes to
        CTA ``(t % ctas + shift) % ctas`` at slot ``t // ctas`` for the base logits and every
        step (a FOLD row's ``a`` is the CTA's first tile); the CTAs with an extra tile are
        past the RNN and CTXR CTAs."""
        if ctas > MAX_CTAS or ctas < self.rnn_tiles:
            raise ValueError(f"need {self.rnn_tiles} to {MAX_CTAS} CTAs, got {ctas}")
        if -(-self.row_tiles // ctas) > self.slots:
            raise ValueError(f"{ctas} CTAs need {-(-self.row_tiles // ctas)} BASE slots")
        lists = [[] for _ in range(ctas)]
        ctxr = [[CTXR, t, s] for s in range(2) for t in range(self.rnn_tiles)]
        for i, row in enumerate(ctxr):
            lists[(self.rnn_tiles + i) % ctas].append(row)
        shift = min(3 * self.rnn_tiles, ctas - self.row_tiles % ctas) % ctas
        for t in range(self.rnn_tiles):
            lists[t].append([RNN, t, 0])
        for t in range(self.row_tiles):
            lists[(t % ctas + shift) % ctas].append([LMHEAD, t, t // ctas])
        for c in range(ctas):
            lists[c].append([ARGMAX, 0, 0])
        for s in range(1, self.drafts):
            for t in range(self.rnn_tiles):
                lists[t].append([RNN, t, s])
            for c in range(ctas):
                lists[c].append([FOLD, (c - shift) % ctas, s])
        lists[0].append([FINAL, 0, self.drafts - 1])
        return lists

    @role(warps=COMPUTE_WARPS, regs=LOAD_REGS)
    @cute.jit
    def load(self, p: HeadParams, *, w: Producer, work: WorkCursor):
        w = w.views(WS=p.sWS, W=p.sW)
        h_tiles = self.hidden // TILE_K
        state_tiles = 2 * self.rank // TILE_K
        while work.valid:
            op, a, b, _row = work.item
            if op == CTXR:
                k0 = state_tiles + b * h_tiles
                w = load_boxes(w, p.tma[0], p.gmem[0], a, k0, h_tiles, "WS")
            elif op == RNN:
                w = load_boxes(w, p.tma[0], p.gmem[0], a, 0, state_tiles, "WS")
            elif op == LMHEAD:
                w = load_boxes(w, p.tma[1], p.gmem[1], a, 0, h_tiles, "WS")
                w = load_tiles(w, p.tma[2], p.gmem[2], a, 0, 1, "W", EVICT_LAST)
            elif op == FOLD:
                ctas = cute.arch.grid_dim()[0]
                for tile in cutlass.range(a, self.row_tiles, ctas, unroll=1):
                    w = load_tiles(w, p.tma[2], p.gmem[2], tile, 0, 1, "W", EVICT_LAST)
            work.advance()
        return w, work

    @role(warps=range(COMPUTE_WARPS), regs=COMPUTE_REGS)
    @cute.jit
    def compute(self, p: HeadParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        epoch = p.state[0] + 1
        rank3 = 3 * self.rank
        xkey = Int32(-1)
        gbuf = Int32(0)
        row = tid // TOKENS
        n = tid % TOKENS
        # Step 0's running best over the CTA's LMHEAD rows, which all precede its ARGMAX row.
        best_v, best_i, best_raw = no_best()
        xf = cute.make_rmem_tensor((4,), Int32)  # this lane's fold input columns, bf16x2
        fold_step = Int32(-1)  # the step whose fold input xf holds (LMH rows: step 0)
        while work.valid:
            op, a, b, _row = work.item
            if op == CTXR:
                if b != xkey:
                    self.fill_x(p, b, tid)
                    xkey = b
                value, w = self.gemv(p, w, p.xs, self.hidden // TILE_K, gbuf, tid, warp, lane)
                gbuf = 1 - gbuf
                index = (b * TOKENS + n) * rank3 + a * ROWS + row
                if (b == 0) & (n < self.drafts) | (b == 1) & (n == 0):
                    publish(p.ctxr, index, value, epoch)
            elif op == RNN:
                self.rnn_input(p, a, b, epoch, tid, warp, lane)
                xkey = X_RNN + b
                value, w = self.gemv(p, w, p.xr, 2 * self.rank // TILE_K, gbuf, tid, warp, lane,
                                     xrows=(0,) * TOKENS)  # fmt: skip
                gbuf = 1 - gbuf
                r = a * ROWS + row
                ctx = cute.make_rmem_tensor((2,), Float32)
                fetch(
                    [(p.ctxr, b * rank3 + r, 1), (p.ctxr, TOKENS * rank3 + r, 1)],
                    ctx,
                    epoch,
                    uniform=True,
                )
                if n == 0:
                    raw = p.rnn_bias[r] + ctx[0] + ctx[1] + value
                    publish(p.raw, b * rank3 + r, raw, epoch)
            elif op == LMHEAD:
                if xkey != X_HIDDEN:
                    self.fill_x(p, Int32(X_HIDDEN), tid)
                    xkey = X_HIDDEN
                value, w = self.gemv(p, w, p.xs, self.hidden // TILE_K, gbuf, tid, warp, lane)
                gbuf = 1 - gbuf
                base_tile = a if cutlass.const_expr(self.global_base) else b
                if n < self.drafts:
                    p.base[base_tile, row, n] = BFloat16(value)
                if n == 0:
                    v = cutlass.min(a * ROWS + row, self.vocab - 1)
                    p.fbias[b, row] = p.fold_bias[v]
                if fold_step != 0:
                    self.fold_x(p, 0, xf, epoch, lane)
                    fold_step = Int32(0)
                self.csync.sync()  # BASE rows are read by other threads' fold epilogues
                with w.consume() as h:
                    acc = self.fold_dot(h.view("W"), xf, warp, lane)
                best_v, best_i, best_raw = self.fold_update(
                    p, acc, a, b, 0, best_v, best_i, best_raw, warp, lane
                )
            elif op == FOLD:
                self.fold_x(p, b, xf, epoch, lane)
                w, step_v, step_i = self.fold_tiles(p, w, a, b, xf, warp, lane)
                self.publish_best(p, b, step_v, step_i, epoch, tid, warp, lane)
            elif op == ARGMAX:  # after the CTA's LMHEAD rows
                self.publish_best(p, b, best_v, best_i, epoch, tid, warp, lane)
            else:
                prev = self.global_best(p, b, epoch, warp, lane)
                if tid == 0:
                    p.drafts[b] = Int64(prev)
            work.advance()
        advance_epoch(p.state, epoch, tid)
        return w, work

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def fill_x(self, p: HeadParams, key, tid):
        """X rows from the mask states (``X_HIDDEN``) or row 0 from f_rnn (``X_FRNN``)."""
        h = self.hidden
        if key == X_HIDDEN:
            for i in cutlass.range_constexpr(TOKENS * h // (8 * COMPUTE_THREADS)):
                chunk = i * COMPUTE_THREADS + tid
                r = chunk // (h // 8)
                col = chunk % (h // 8)
                frag = cute.make_rmem_tensor((8,), BFloat16)
                cute.autovec_copy(cute.local_tile(p.hidden[r, None], (8,), (col,)), frag)
                cute.autovec_copy(frag, cute.local_tile(p.xs[r, None], (8,), (col,)))
        else:
            for r in cutlass.range_constexpr(h // COMPUTE_THREADS):
                k = r * COMPUTE_THREADS + tid
                p.xs[0, k] = p.f_rnn[k]
        self.csync.sync()

    @cute.jit
    def fold_x(self, p: HeadParams, step, xf, epoch, lane):
        """Lane l's fold input columns ``8l .. 8l + 7`` as 4 bf16x2 words: bf16 tanh(bf16(out))
        of step ``step``'s raw, polled by every warp."""
        raw = cute.make_rmem_tensor((8,), Float32)
        base = step * 3 * self.rank + 2 * self.rank + 8 * lane
        fetch([(p.raw, base, 4), (p.raw, base + 4, 4)], raw, epoch, uniform=True)
        x = cute.make_rmem_tensor((8,), BFloat16)
        for e in cutlass.range_constexpr(8):
            x[e] = BFloat16(cute.math.tanh(bf16_round(raw[e])))
        words = cute.recast_tensor(x, Int32)
        for j in cutlass.range_constexpr(4):
            xf[j] = words[j]

    @cute.jit
    def fold_dot(self, sW, xf, warp, lane):
        """One fold stage's dots against the lane's input columns (``xf``: 4 bf16x2 words):
        warp w's rows ``4w ..``, lane l's columns ``8l ..``, exact bf16 products summed in fp32
        (``fma.rn.f32.bf16``), even and odd columns apart; ``fold_update`` reduces them."""
        acc = cute.make_rmem_tensor((ROWS_PER_WARP, 2), Float32)
        acc.fill(0.0)
        for r in cutlass.range_constexpr(ROWS_PER_WARP):
            words = load_words16(sW[warp * ROWS_PER_WARP + r, None], 8 * lane)
            for j in cutlass.range_constexpr(4):
                acc[r, 0], acc[r, 1] = fma_bf16x2(acc[r, 0], acc[r, 1], words[j], xf[j])
        return acc

    @cute.jit
    def fold_update(self, p: HeadParams, acc, tile, slot, step, best_v, best_i, best_raw, warp,
                    lane):  # fmt: skip
        """Fold logits of vocabulary tile ``tile`` (base logits at BASE ``slot``) from
        ``fold_dot``'s accumulators: the owner lanes update their running best (a larger
        value, or the first index: rows ascend per lane) and the largest logit before the
        softcap."""
        value, row = reduce_tile(acc, warp, lane)
        v = tile * ROWS + row
        if owns_row(lane) & (v < self.vocab):
            bias = bf16_round(value + Float32(p.fbias[slot, row]))
            base_tile = tile if cutlass.const_expr(self.global_base) else slot
            logit = bf16_round(Float32(p.base[base_tile, row, step]) + bias)
            # The softcap is monotone and a later row must beat the best strictly, so only a
            # logit above the best's (pre-softcap) can win: skip the tanh otherwise. NaN never
            # wins (nan_to_num(-inf) in the oracle).
            if logit > best_raw:
                cap = self.softcap
                capped = bf16_round(cute.math.tanh(logit * fp32_reciprocal(cap)) * cap)
                best_raw = logit
                if capped > best_v:
                    best_v = capped
                    best_i = v
        return best_v, best_i, best_raw

    @cute.jit
    def fold_tiles(self, p: HeadParams, w, first, step, xf, warp, lane):
        """``(w, value, index)``: the lane's best fold logit of step ``step`` over the rows of
        this CTA's tiles ``first, first + grid, ..``, two stages at a time: their dots and
        reductions are independent chains."""
        ctas = cute.arch.grid_dim()[0]
        slot = Int32(0)
        best_v, best_i, best_raw = no_best()
        for tile in cutlass.range(first, self.row_tiles, 2 * ctas, unroll=1):
            if tile + ctas < self.row_tiles:
                h0 = w.wait()
                h1 = w.wait()
                acc0 = self.fold_dot(h0.view("W"), xf, warp, lane)
                acc1 = self.fold_dot(h1.view("W"), xf, warp, lane)
                h0.release()
                h1.release()
                best_v, best_i, best_raw = self.fold_update(
                    p, acc0, tile, slot, step, best_v, best_i, best_raw, warp, lane
                )
                best_v, best_i, best_raw = self.fold_update(
                    p, acc1, tile + ctas, slot + 1, step, best_v, best_i, best_raw, warp, lane
                )
            else:
                with w.consume() as h:
                    acc0 = self.fold_dot(h.view("W"), xf, warp, lane)
                best_v, best_i, best_raw = self.fold_update(
                    p, acc0, tile, slot, step, best_v, best_i, best_raw, warp, lane
                )
            slot = slot + 2
        return w, best_v, best_i

    @cute.jit
    def rnn_input(self, p: HeadParams, tile, step, epoch, tid, warp, lane):
        """XR = ``[state, w1[prev]]`` of step ``step``: the anchor and a zero state at step
        0; else the previous step's argmax and the state updated (bf16 op by op, as the
        oracle) from its raw gate and candidate. Tile 0's CTA writes the previous draft."""
        r = self.rank
        prev = Int32(p.token[0])
        if step == 0:
            p.rnn_state[tid] = Float32(0.0)
        else:
            gc = cute.make_rmem_tensor((2,), Float32)
            base = (step - 1) * 3 * r
            fetch([(p.raw, base + tid, 1), (p.raw, base + r + tid, 1)], gc, epoch, uniform=True)
            gate = bf16_round(Float32(1.0) / (Float32(1.0) + cute.math.exp(-bf16_round(gc[0]))))
            kept = bf16_round(gate * p.rnn_state[tid])
            fresh = bf16_round(
                bf16_round(Float32(1.0) - gate) * bf16_round(cute.math.tanh(bf16_round(gc[1])))
            )
            p.rnn_state[tid] = bf16_round(kept + fresh)
            prev = self.global_best(p, step - 1, epoch, warp, lane)
            if (tile == 0) & (tid == 0):
                p.drafts[step - 1] = Int64(prev)
        p.xr[0, tid] = BFloat16(p.rnn_state[tid])
        p.xr[0, r + tid] = p.markov_w1[prev, tid]
        self.csync.sync()

    @cute.jit
    def better_of(self, v, i, ov, oi):
        """The better of ``(v, i)`` and ``(ov, oi)``: the larger value, or the lower index on a
        tie."""
        take = (ov > v) | ((ov == v) & (oi < i))
        return Float32(cutlass.select_(take, ov, v)), Int32(cutlass.select_(take, oi, i))

    @cute.jit
    def warp_best(self, v, i):
        """The warp's best ``(value, index)`` in every lane."""
        for k in cutlass.range_constexpr(5):
            ov = Float32(cute.arch.shuffle_sync_bfly(v, offset=1 << k))
            oi = Int32(cute.arch.shuffle_sync_bfly(i, offset=1 << k))
            v, i = self.better_of(v, i, ov, oi)
        return v, i

    @cute.jit
    def publish_best(self, p: HeadParams, step, v, i, epoch, tid, warp, lane):
        """The CTA's best of step ``step`` over its threads' running bests, published."""
        v, i = self.warp_best(v, i)
        if lane == 0:
            p.arg[warp] = v
            p.arg[COMPUTE_WARPS + warp] = Float32(i)
        self.csync.sync()
        if tid == 0:
            bv = p.arg[0]
            bi = Int32(p.arg[COMPUTE_WARPS])
            for k in cutlass.range_constexpr(1, COMPUTE_WARPS):
                bv, bi = self.better_of(bv, bi, p.arg[k], Int32(p.arg[COMPUTE_WARPS + k]))
            cta = cute.arch.block_idx()[0]
            slot = (step * MAX_CTAS + cta) * 2
            publish(p.best, slot, bv, epoch)
            publish(p.best, slot + 1, Float32(bi), epoch)
        self.csync.sync()

    @cute.jit
    def global_best(self, p: HeadParams, step, epoch, warp, lane):
        """Step ``step``'s draft: the best of every CTA's published best (warp 0 polls them
        with backoff), in every compute thread."""
        if warp == 0:
            ctas = cute.arch.grid_dim()[0]
            rounds = MAX_CTAS // 32
            vals = cute.make_rmem_tensor((2 * rounds,), Float32)
            segments = [
                (p.best, (step * MAX_CTAS + cutlass.min(lane + 32 * k, ctas - 1)) * 2, 2)
                for k in range(rounds)
            ]
            fetch(segments, vals, epoch, uniform=True, backoff=BACKOFF_NS)
            v = vals[0]
            i = Int32(vals[1])
            for k in cutlass.range_constexpr(1, rounds):
                v, i = self.better_of(v, i, vals[2 * k], Int32(vals[2 * k + 1]))
            v, i = self.warp_best(v, i)
            if lane == 0:
                p.arg[2 * COMPUTE_WARPS] = Float32(i)
        self.csync.sync()
        return Int32(p.arg[2 * COMPUTE_WARPS])

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(self, tma, gtensors, scratch, tensors, bound):
        layouts = {
            # One row read for every token (their columns are discarded): any hidden size
            # holds it.
            "XR": cute.make_layout((1, 2 * self.rank), stride=(2 * self.rank, 1)),
            "STATE": cute.make_layout((self.rank,)),
            "ARG": cute.make_layout((ARG_FLOATS,)),
            "FBIAS": cute.make_layout((self.slots, ROWS), stride=(ROWS, 1)),
        }
        base_layout = cute.make_layout(
            (self.row_tiles if self.global_base else self.slots, ROWS, self.drafts),
            stride=(ROWS * self.drafts, 1, ROWS),
        )
        if cutlass.const_expr(not self.global_base):
            layouts["BASE"] = base_layout
        boxes = [("WS", (ROWS, BOX))] * 2 + [("W", (ROWS, TILE_K))]
        trace_buffer, fields, smem = self.begin_draft(
            bound, layouts, TENSOR_ARGS, tensors, scratch, tma, gtensors, boxes
        )
        if cutlass.const_expr(self.global_base):
            base = cute.make_tensor(
                cute.recast_ptr(fields.pop("base_words").iterator, dtype=BFloat16), base_layout
            )
        else:
            base = self.buffer("BASE")
        params = HeadParams(
            sW=StageView(smem[2], axis=1),
            xr=self.buffer("XR"),
            base=base,
            fbias=self.buffer("FBIAS"),
            rnn_state=self.buffer("STATE"),
            arg=self.buffer("ARG"),
            **fields,
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def __call__(
        self,
        rnn_proj: cute.Tensor,
        lm_head: cute.Tensor,
        fold_weight: cute.Tensor,
        hidden: cute.Tensor,
        f_rnn: cute.Tensor,
        rnn_bias: cute.Tensor,
        markov_w1: cute.Tensor,
        fold_bias: cute.Tensor,
        token: cute.Tensor,
        drafts: cute.Tensor,
        trace_buffer: cute.Tensor,
        scratch: cute.Tensor,
        state: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        pairs = [swizzled_atom(src, BFloat16) for src in (rnn_proj, lm_head)]
        pairs.append(tma_atom(fold_weight, (ROWS, TILE_K)))
        tensors = (trace_buffer, hidden, f_rnn, rnn_bias, markov_w1, fold_bias, token, drafts,
                   state)  # fmt: skip
        self.launch(pairs, tensors, scratch, table, workspace, max_ctas, stream)


@jit_cache
def compile_draft_head_kernel(op: DraftHeadKernel):
    """One compile serves every step: weights, states and the anchor are arguments."""
    bf16, f32 = BFloat16, Float32
    h, r = op.hidden, op.rank
    return compile_draft(
        op,
        make_fake_compact_tensor(bf16, (3 * r, 2 * r + 2 * h)),  # rnn_proj
        make_fake_compact_tensor(bf16, (op.vocab, h)),  # LM head
        make_fake_compact_tensor(bf16, (op.vocab, r)),  # fold weight
        make_fake_compact_tensor(bf16, (TOKENS, h)),  # mask states
        make_fake_compact_tensor(bf16, (h,)),  # f_rnn
        make_fake_compact_tensor(f32, (3 * r,), assumed_align=4),  # rnn bias
        make_fake_compact_tensor(bf16, (cute.sym_int(), r)),  # markov_w1
        make_fake_compact_tensor(bf16, (op.vocab,), assumed_align=2),  # fold bias
        make_fake_compact_tensor(Int64, (1,), assumed_align=8),  # anchor
        make_fake_compact_tensor(Int64, (op.drafts,), assumed_align=8),  # drafts
    )
