"""The DSpark drafter's backbone over a block as one persistent warpdrive launch
(``DraftBackboneKernel``) and its compile function; ``draft`` holds the host side.

Implements ``dspark.DSparkDrafter.block_states``: the block ``[embed_norm(embed(anchor)),
mask_embed x 7]`` at positions ``pos .. pos + 7`` (anchor and position read on the device)
through the dense layers, then the final norm of the mask rows. Per layer ``L``::

    every CTA   before its first QKV row: X = the layer input, RMSNorm(input_norm) of the
                residual stream (layer 0: computed from the embedding row and mask_embed; later
                layers: polled from the previous layer's DOWN reducers)
    QKV         32 rows of [wq; wk; wv] x a share of hidden -> fp32 partials (tagged)
                (split-K ``qkv_splits``: Inkling 2 Mini's 3)
    ATTN        kv head x KV split: the projections (the     -> unnormalized O, (m, l) per
                partials' bf16 sum), q_norm + RoPE of its      (head, token) row (tagged)
                q heads (24 rows: heads x tokens), k_norm
                + RoPE and v of the block's 8 keys; flash
                partial over the split's 64-key context
                blocks (the window's or every cached key);
                the last split adds the block's own keys
    MERGE       q head: merge the splits                     -> bf16 attention output (tagged)
    OPROJ       32 rows of wo x the attention output,        -> fp32 residual stream (tagged)
                + the residual (fused add-RMSNorm's sum)
    GATEUP      16 gate + 16 up rows x RMSNorm(post_norm)    -> bf16 act (tagged)
                of the stream; act = SiLU(gate) * up
    DOWN        32 rows of w_down x a quarter of act;        -> fp32 partials; the residual
                the last quarter's items reduce + residual     stream (tagged)
    FINAL       final norm of the mask rows                 -> bf16 (TOKENS, hidden)

Every query of the block sits at its last position (vLLM's DRAFT attention): each attends to
the whole block and to the context, all of it (full layers) or the ``window`` positions before
the block (local layers). The block's own K/V are taken from the projections; they are not
written to the draft cache: positions at or after ``pos`` are rewritten by the next commit or
block before any read (``dspark``'s ``test_rejected_block_rows_do_not_leak``).

Attention runs on ``mma.sync`` (``DraftBackboneKernel.attend``): per 64-key block warp w
scores 8 keys for all 32 (head, token) rows, the CTA shares row maxima and the bf16
probabilities (rounded as the oracle's) through SMEM, and warp w accumulates 16 dims of the
value sum.

Roles as ``visit_kernel``; the launch is cooperative and the host deals every CTA's rows in
dependency order (``lists``): the reducers of a split-K phase on CTAs that run no other split.
"""

from dataclasses import dataclass

import cutlass
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, Int64, cute

from attn_gym.warpdrive import (
    Consumer,
    Local,
    NamedBarrier,
    Producer,
    Ring,
    SmemPlan,
    StageView,
    StaticPersistent,
    Tma,
    Warps,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.inkling_megakernel.draft_common import (
    GRED_FLOATS,
    REDUCTIONS,
    TOKENS,
    WORDS,
    DraftKernel,
    check_draft_sizes,
    gemv_atom,
    x_layout,
)
from attn_gym.warpdrive.examples.megakernel.constants import (
    BLOCK,
    BOX,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    HEAD_DIM,
    LOG2E,
    RED_VALUES,
    ROWS,
    STAGE_BYTES,
    TILE_K,
    X_PAD,
)
from attn_gym.warpdrive.examples.megakernel.multi_gemv import mma_bf16
from attn_gym.warpdrive.examples.megakernel.regmath import (
    bf16_round,
    cta_sum,
    fp32,
    fp32_reciprocal,
    group_sum,
    load_words16,
    red_layout,
)
from attn_gym.warpdrive.examples.megakernel.ring import (
    kv_stages,
    load_boxes,
    load_tiles,
    ring_stages,
    tma_atom,
    tma_partition,
)
from attn_gym.warpdrive.examples.megakernel.table import InstructionTable
from attn_gym.warpdrive.examples.megakernel.tagged import (
    advance_epoch,
    fetch,
    publish,
    scratch_offsets,
    tagged_views,
)

QKV, ATTN, MERGE, OPROJ, GATEUP, DOWN, FINAL = range(7)
COMPUTE_REGS, LOAD_REGS = 232, 40
ATTN_ROWS = 32  # an ATTN item's (head, token) rows: two m16 tiles
# Default split-K of the q/k/v projections: Inkling 2 Mini's 160 9-stage row tiles leave 8 of
# 152 CTAs a second one; 480 3-stage items balance.
QKV_SPLITS = 3
QPAD = 8  # bf16 columns padding Q and P rows (conflict-free fragment loads)
TENSOR_ARGS = (
    "trace_buffer", "embed", "embed_norm", "mask_embed", "input_norm", "post_norm", "q_norm", "k_norm",
    "final_norm", "rope", "token", "pos", "hidden", "state",
)  # fmt: skip
# X keys: the input the X rows hold (``DraftBackboneKernel.xkey``).
X_PRE, X_ATT, X_POST, X_ACT = 1, 2, 3, 4


@dataclass(frozen=True)
class BackboneParams:
    tma: tuple  # TMA atoms: wqkv, wo, wgu, w_down, k cache, v cache
    gmem: tuple
    sWS: StageView  # the weight ring's swizzled-box view
    sKV: StageView
    xs: cute.Tensor  # (TOKENS, x_width) bf16 GEMV inputs
    red: cute.Tensor
    gred: cute.Tensor
    qs: cute.Tensor  # (ATTN_ROWS, HEAD_DIM) bf16 an ATTN item's roped q rows (head-major;
    # rows past attn_rows stay zero)
    kb: cute.Tensor  # (2 * TOKENS, HEAD_DIM) bf16 the block's roped keys (rows 8.. zero)
    vb: cute.Tensor  # (2 * TOKENS, HEAD_DIM) bf16 the block's values (rows 8.. zero)
    probs: cute.Tensor  # (ATTN_ROWS, BLOCK) bf16 a block's probabilities
    rmax: cute.Tensor  # (ATTN_ROWS, COMPUTE_WARPS) fp32 per-warp row maxima, then sums
    # Tagged int64 intermediates (``scratch_shapes``).
    qkv: cute.Tensor
    attn_o: cute.Tensor
    attn_ml: cute.Tensor
    attn_out: cute.Tensor
    h_mid: cute.Tensor
    act: cute.Tensor
    d_part: cute.Tensor
    h_pre: cute.Tensor
    embed: cute.Tensor  # (padded vocab, hidden) bf16
    embed_norm: cute.Tensor  # (hidden,) bf16
    mask_embed: cute.Tensor  # (hidden,) bf16
    input_norm: cute.Tensor  # (layers, hidden) bf16
    post_norm: cute.Tensor  # (layers, hidden) bf16
    q_norm: cute.Tensor  # (layers, HEAD_DIM) bf16
    k_norm: cute.Tensor
    final_norm: cute.Tensor  # (hidden,) bf16
    rope: cute.Tensor  # (max_len, HEAD_DIM) fp32 [cos | sin]
    token: cute.Tensor  # int64 (1,) the anchor
    pos: cute.Tensor  # int64 (1,) its position
    hidden: cute.Tensor  # (TOKENS, hidden) bf16 output: row n - 1 is mask n's final state
    state: cute.Tensor  # int32 (2,) launch epoch, finished CTAs


class DraftBackboneKernel(DraftKernel):
    """The block's dense layers (see the module docstring). Constructor arguments are the
    drafter's sizes and constants; ``local`` is the layers with a ``window``; ``splits``: KV
    splits per kv head; ``down_splits`` and ``qkv_splits``: split-K of the down and q/k/v
    projections; ``stages``: weight ring depth in 16 KB stages."""

    def __init__(
        self,
        hidden: int,
        layers: int,
        q_heads: int,
        kv_heads: int,
        intermediate: int,
        rms_eps: float,
        local: tuple[int, ...],
        window: int,
        splits: int = 8,
        down_splits: int = 4,
        qkv_splits: int = QKV_SPLITS,
        stages: int | None = None,
    ):
        check_draft_sizes(hidden)
        if down_splits < 1 or qkv_splits < 1:
            raise ValueError(
                f"need positive down_splits and qkv_splits, got {down_splits}, {qkv_splits}"
            )
        group = q_heads // kv_heads
        if q_heads % kv_heads or group * TOKENS > ATTN_ROWS:
            raise ValueError(
                f"need q heads a multiple of kv heads, at most {ATTN_ROWS // TOKENS} per kv"
            )
        if hidden % (qkv_splits * TILE_K):
            raise ValueError(f"hidden size must be a multiple of {qkv_splits * TILE_K}")
        if (q_heads * HEAD_DIM) % TILE_K or intermediate % (down_splits * TILE_K):
            raise ValueError("q dim and intermediate / down_splits must be multiples of 256")
        if not 1 <= splits <= 32 or window % BLOCK:
            raise ValueError(f"need 1 to 32 KV splits and a window multiple of {BLOCK}")
        self.hidden, self.layers, self.q_heads, self.kv_heads = hidden, layers, q_heads, kv_heads
        self.intermediate, self.rms_eps = intermediate, rms_eps
        self.local, self.window = tuple(local), window
        self.splits, self.down_splits, self.qkv_splits = splits, down_splits, qkv_splits
        self.group = group
        self.q_dim = q_heads * HEAD_DIM
        self.qkv_rows = (q_heads + 2 * kv_heads) * HEAD_DIM
        self.down_width = intermediate // down_splits
        self.x_width = max(hidden, self.q_dim, self.down_width)
        self.attn_rows = group * TOKENS  # (head, token) rows per kv head
        fixed = (
            2 * TOKENS * (self.x_width + X_PAD)
            + 4 * (REDUCTIONS * COMPUTE_WARPS * RED_VALUES + GRED_FLOATS)
            + 2 * ATTN_ROWS * (HEAD_DIM + QPAD + BLOCK + QPAD)
            + 2 * 4 * TOKENS * HEAD_DIM
            + 4 * ATTN_ROWS * COMPUTE_WARPS
        )
        self.stages = ring_stages(fixed, stages)
        self.smem_plan = SmemPlan()
        ring = self.ring_buffers()
        ring["KV"] = self.smem_plan.buffer("KV", BFloat16, align=1024, overlaps=ring["W"])
        self.smem_plan.buffer("X", BFloat16, align=16)
        self.smem_plan.buffer("RED", Float32, align=16)
        self.smem_plan.buffer("GRED", Float32, align=16)
        for name in ("Q", "KB", "VB", "P"):
            self.smem_plan.buffer(name, BFloat16, align=16)
        self.smem_plan.buffer("RMAX", Float32, align=16)
        self.w = Ring(self.stages, Tma(STAGE_BYTES), Warps, buffer=ring)
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.work = WorkStream(mapping=InstructionTable(words=WORDS), schedule=StaticPersistent())

    @property
    def scratch_shapes(self) -> dict[str, int]:
        layers, h = self.layers, self.hidden
        partials = layers * self.kv_heads * self.splits * self.attn_rows
        return {
            "qkv": layers * self.qkv_splits * TOKENS * self.qkv_rows,
            "attn_o": partials * HEAD_DIM,
            "attn_ml": partials * 2,
            "attn_out": layers * TOKENS * self.q_dim,
            "h_mid": layers * TOKENS * h,
            "act": layers * TOKENS * self.intermediate,
            "d_part": layers * self.down_splits * TOKENS * h,
            "h_pre": (layers + 1) * TOKENS * h,
        }

    def get_name(self) -> str:
        return (
            f"warpdrive_inkling_draft_backbone_h{self.hidden}_l{self.layers}"
            f"_q{self.q_heads}kv{self.kv_heads}_i{self.intermediate}_w{self.window}"
            f"_{''.join(map(str, self.local))}_kv{self.splits}_d{self.down_splits}q{self.qkv_splits}"
            f"_s{self.stages}{self.trace_key()}"
        )

    def lists(self, ctas: int) -> list[list[list[int]]]:
        """Each CTA's rows in dependency order (module docstring), every phase dealt
        round-robin from where the previous one stopped. DOWN deals split s's row tiles to
        group s of the CTAs, so a CTA fills its act input once and the last split's reducers
        run on CTAs with no other split's rows."""
        h_tiles = self.hidden // ROWS
        lists = [[] for _ in range(ctas)]
        cursor = 0

        def deal(rows):
            nonlocal cursor
            for row in rows:
                lists[cursor % ctas].append(row)
                cursor += 1

        groups = [range(s * ctas // self.down_splits, (s + 1) * ctas // self.down_splits)
                  for s in range(self.down_splits)]  # fmt: skip
        if min(len(g) for g in groups) == 0:
            raise ValueError(f"{ctas} CTAs cannot hold {self.down_splits} DOWN groups")
        for layer in range(self.layers):
            tiles = self.qkv_rows // ROWS
            deal(
                [
                    [QKV, layer, t * self.qkv_splits + k]
                    for t in range(tiles)
                    for k in range(self.qkv_splits)
                ]
            )
            deal([[ATTN, layer, b] for b in range(self.kv_heads * self.splits)])
            deal([[MERGE, layer, hq] for hq in range(self.q_heads)])
            deal([[OPROJ, layer, t] for t in range(h_tiles)])
            deal([[GATEUP, layer, t] for t in range(self.intermediate // 16)])
            for s, group in enumerate(groups):
                for t in range(h_tiles):
                    lists[group[t % len(group)]].append([DOWN, layer, t * self.down_splits + s])
        deal([[FINAL, 0, 0]])
        return lists

    # ------------------------------------------------------------------ roles

    @role(warps=COMPUTE_WARPS, regs=LOAD_REGS)
    @cute.jit
    def load(self, p: BackboneParams, *, w: Producer, work: WorkCursor):
        w = w.views(WS=p.sWS, KV=p.sKV)
        pos = Int32(p.pos[0])
        # KV blocks per (layer, kv head) of the cache rows (layers * kv_heads * max_len).
        head_blocks = cute.size(p.gmem[4], mode=[1]) // (self.layers * self.kv_heads)
        h_k = self.hidden // TILE_K
        h_tiles = self.hidden // ROWS
        while work.valid:
            op, a, b, _row = work.item
            if op == QKV:
                part = h_k // self.qkv_splits
                tile = a * (self.qkv_rows // ROWS) + b // self.qkv_splits
                w = load_boxes(w, p.tma[0], p.gmem[0], tile, b % self.qkv_splits * part, part,
                               "WS")  # fmt: skip
            elif op == ATTN:
                kvh = b // self.splits
                blk0, blk1 = self.split_blocks(a, pos, b % self.splits)
                row0 = (a * self.kv_heads + kvh) * head_blocks
                for blk in cutlass.range(blk0, blk1, unroll=1):
                    w = load_tiles(w, p.tma[4], p.gmem[4], row0 + blk, 0, 1, "KV")
                    w = load_tiles(w, p.tma[5], p.gmem[5], row0 + blk, 0, 1, "KV")
            elif op == OPROJ:
                w = load_boxes(w, p.tma[1], p.gmem[1], a * h_tiles + b, 0,
                               self.q_dim // TILE_K, "WS")  # fmt: skip
            elif op == GATEUP:
                w = load_boxes(w, p.tma[2], p.gmem[2], a * (self.intermediate // 16) + b, 0,
                               h_k, "WS")  # fmt: skip
            elif op == DOWN:
                t = b // self.down_splits
                s = b % self.down_splits
                k_tiles = self.down_width // TILE_K
                w = load_boxes(w, p.tma[3], p.gmem[3], a * h_tiles + t, s * k_tiles, k_tiles,
                               "WS")  # fmt: skip
            work.advance()
        return w, work

    @role(warps=range(COMPUTE_WARPS), regs=COMPUTE_REGS)
    @cute.jit
    def compute(self, p: BackboneParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        epoch = p.state[0] + 1
        pos = Int32(p.pos[0])
        h = self.hidden
        rinv_e = self.prologue(p, tid, warp, lane)
        xkey = Int32(self.xkey(X_PRE, 0))
        gbuf = Int32(0)
        row = tid // TOKENS
        n = tid % TOKENS
        h_tiles = h // ROWS
        while work.valid:
            op, a, b, _row = work.item
            if op == QKV:
                if xkey != self.xkey(X_PRE, a):
                    self.wait_items(p.h_pre, self.sentinel(a, ROWS - 1), h_tiles, ROWS, epoch,
                                    warp, lane)  # fmt: skip
                    self.fill_norm(p, p.h_pre, a * TOKENS * h, p.input_norm[a, None], epoch,
                                   warp, lane)  # fmt: skip
                    xkey = self.xkey(X_PRE, a)
                ks = b % self.qkv_splits
                value = Float32(0.0)
                width = h // self.qkv_splits
                # One branch per split: the GEMV input starts at a constant (aligned) column.
                for k in cutlass.range_constexpr(self.qkv_splits):
                    if ks == k:
                        value, w = self.gemv(p, w, p.xs, width // TILE_K, gbuf, tid, warp, lane,
                                             k * width)  # fmt: skip
                gbuf = 1 - gbuf
                index = ((a * self.qkv_splits + ks) * TOKENS + n) * self.qkv_rows
                publish(p.qkv, index + b // self.qkv_splits * ROWS + row, value, epoch)
            elif op == ATTN:
                w = self.attention(p, w, a, b // self.splits, b % self.splits, pos, epoch, tid,
                                   warp, lane)  # fmt: skip
            elif op == MERGE:
                self.merge(p, a, b, epoch, warp, lane)
            elif op == OPROJ:
                if xkey != self.xkey(X_ATT, a):
                    first = (a * TOKENS + TOKENS - 1) * self.q_dim + HEAD_DIM - 1
                    self.wait_items(p.attn_out, first, self.q_heads, HEAD_DIM, epoch, warp, lane)
                    self.fill_raw(p, p.attn_out, a * TOKENS * self.q_dim, self.q_dim,
                                  self.q_dim, epoch, warp, lane)  # fmt: skip
                    xkey = self.xkey(X_ATT, a)
                value, w = self.gemv(p, w, p.xs, self.q_dim // TILE_K, gbuf, tid, warp, lane)
                gbuf = 1 - gbuf
                col = b * ROWS + row
                residual = Float32(0.0)
                if a == 0:
                    residual = self.residual0(p, n, col, rinv_e)
                else:
                    residual = bf16_round(self.fetch1(p.h_pre, (a * TOKENS + n) * h + col, epoch))
                publish(p.h_mid, (a * TOKENS + n) * h + col, bf16_round(value) + residual, epoch)
            elif op == GATEUP:
                if xkey != self.xkey(X_POST, a):
                    first = (a * TOKENS + TOKENS - 1) * h + ROWS - 1
                    self.wait_items(p.h_mid, first, h_tiles, ROWS, epoch, warp, lane)
                    self.fill_norm(p, p.h_mid, a * TOKENS * h, p.post_norm[a, None], epoch,
                                   warp, lane)  # fmt: skip
                    xkey = self.xkey(X_POST, a)
                w = self.gemv_partials(p, w, p.xs, h // TILE_K, gbuf, warp, lane)
                if tid < 16 * TOKENS:
                    gate = bf16_round(self.partial(p, gbuf, row, n))
                    up = bf16_round(self.partial(p, gbuf, row + 16, n))
                    sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate))
                    index = (a * TOKENS + n) * self.intermediate + b * 16 + row
                    publish(p.act, index, bf16_round(gate * sig * up), epoch)
                gbuf = 1 - gbuf
            elif op == DOWN:
                t = b // self.down_splits
                s = b % self.down_splits
                if xkey != self.xkey(X_ACT, a, s):
                    tiles = self.down_width // 16
                    first = (a * TOKENS + TOKENS - 1) * self.intermediate + s * self.down_width
                    self.wait_items(p.act, first + 15, tiles, 16, epoch, warp, lane)
                    self.fill_raw(p, p.act, a * TOKENS * self.intermediate + s * self.down_width,
                                  self.intermediate, self.down_width, epoch, warp, lane)  # fmt: skip
                    xkey = self.xkey(X_ACT, a, s)
                value, w = self.gemv(p, w, p.xs, self.down_width // TILE_K, gbuf, tid, warp, lane)
                gbuf = 1 - gbuf
                col = t * ROWS + row
                if s < self.down_splits - 1:
                    index = ((a * self.down_splits + s) * TOKENS + n) * h + col
                    publish(p.d_part, index, value, epoch)
                else:
                    self.reduce_down(p, a, n, col, value, epoch)
            else:
                self.final(p, epoch, warp, lane)
            work.advance()
        advance_epoch(p.state, epoch, tid)
        return w, work

    # ------------------------------------------------------------ helpers

    @cute.jit
    def fetch1(self, buf, index, epoch):
        """One tagged word, polled (every lane of the warp must call it)."""
        out = cute.make_rmem_tensor((1,), Float32)
        fetch([(buf, index, 1)], out, epoch, uniform=True)
        return out[0]

    def xkey(self, kind, layer, split=0):
        """The key of X holding input ``kind`` (``X_PRE`` ..) of ``layer`` (DOWN: of act
        share ``split``), distinct for every kind, layer and split."""
        return (kind * self.down_splits + split) * self.layers + layer

    def sentinel(self, layer, last):
        """The h_pre word of layer input ``layer``'s first DOWN reducer that its last token's
        thread ``last`` publishes: one word per reducer item, ``ROWS`` apart."""
        return (layer * TOKENS + TOKENS - 1) * self.hidden + last

    # ------------------------------------------------------------- X fills

    @cute.jit
    def prologue(self, p: BackboneParams, tid, warp, lane):
        """Layer 0's input in X: RMSNorm(input_norm[0]) of ``[embed_norm(embed(anchor)),
        mask_embed x 7]``. Returns the embedding row's RMS reciprocal (OPROJ recomputes the
        residual from it)."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        inv_h = fp32_reciprocal(h)
        anchor = Int32(p.token[0])
        e = cute.make_rmem_tensor((per,), Float32)
        m = cute.make_rmem_tensor((per,), Float32)
        ss = Float32(0.0)
        ssm = Float32(0.0)
        for r in cutlass.range_constexpr(per):
            k = r * COMPUTE_THREADS + tid
            e[r] = Float32(p.embed[anchor, k])
            m[r] = Float32(p.mask_embed[k])
            ss = ss + e[r] * e[r]
            ssm = ssm + m[r] * m[r]
        sums = cta_sum(p.red, self.csync, [ss, ssm], 0, warp, lane)
        rinv_e = cute.math.rsqrt(sums[0] * inv_h + Float32(self.rms_eps))
        ss0 = Float32(0.0)
        for r in cutlass.range_constexpr(per):
            k = r * COMPUTE_THREADS + tid
            e[r] = bf16_round(Float32(p.embed_norm[k]) * (e[r] * rinv_e))
            ss0 = ss0 + e[r] * e[r]
        ss0 = cta_sum(p.red, self.csync, [ss0], 1, warp, lane)[0]
        rinv0 = cute.math.rsqrt(ss0 * inv_h + Float32(self.rms_eps))
        rinvm = cute.math.rsqrt(sums[1] * inv_h + Float32(self.rms_eps))
        for r in cutlass.range_constexpr(per):
            k = r * COMPUTE_THREADS + tid
            weight = Float32(p.input_norm[0, k])
            p.xs[0, k] = BFloat16(weight * (e[r] * rinv0))
            xm = BFloat16(weight * (m[r] * rinvm))
            for t in cutlass.range_constexpr(1, TOKENS):
                p.xs[t, k] = xm
        # Rows no ATTN item writes stay zero: Q's past the item's rows, KB/VB's past the block.
        for i in cutlass.range_constexpr(ATTN_ROWS * HEAD_DIM // COMPUTE_THREADS):
            e = i * COMPUTE_THREADS + tid
            p.qs[e // HEAD_DIM, e % HEAD_DIM] = BFloat16(0.0)
        for i in cutlass.range_constexpr(TOKENS * HEAD_DIM // COMPUTE_THREADS):
            e = i * COMPUTE_THREADS + tid
            p.kb[TOKENS + e // HEAD_DIM, e % HEAD_DIM] = BFloat16(0.0)
            p.vb[TOKENS + e // HEAD_DIM, e % HEAD_DIM] = BFloat16(0.0)
        self.csync.sync()
        return rinv_e

    @cute.jit
    def residual0(self, p: BackboneParams, n, col, rinv_e):
        """Layer 0's residual (the block's embedding rows) at token ``n``, column ``col``."""
        anchor = Int32(p.token[0])
        e = bf16_round(Float32(p.embed_norm[col]) * (Float32(p.embed[anchor, col]) * rinv_e))
        return cutlass.select_(n == 0, e, Float32(p.mask_embed[col]))

    @cute.jit
    def fill_norm(self, p: BackboneParams, buf, base, weights, epoch, warp, lane):
        """X row n = bf16(weights * RMSNorm(row n of the fp32 stream at ``buf[base ..]``)),
        warp n polling row n (``fetch_rows``' layout)."""
        h = self.hidden
        chunks = h // 128
        vals = cute.make_rmem_tensor((4 * chunks,), Float32)
        self.fetch_rows(buf, base + warp * h, h, vals, epoch, lane)
        ss = Float32(0.0)
        for i in cutlass.range(4 * chunks, unroll_full=True):
            ss = ss + vals[i] * vals[i]
        ss = group_sum(ss, 32)
        rinv = cute.math.rsqrt(ss * fp32_reciprocal(h) + Float32(self.rms_eps))
        self.store_row(p.xs, warp, h, vals, rinv, weights, lane)
        self.csync.sync()

    @cute.jit
    def fill_raw(self, p: BackboneParams, buf, base, stride, width: cutlass.Constexpr, epoch,
                 warp, lane):  # fmt: skip
        """X row n = the ``width`` bf16 values at ``buf[base + n * stride ..]``, in chunks of
        12 vectors per lane (bounds registers)."""
        chunks = width // 128
        for j0 in cutlass.range_constexpr(0, chunks, 12):
            count = min(12, chunks - j0)
            vals = cute.make_rmem_tensor((4 * count,), Float32)
            segments = [(buf, base + warp * stride + 128 * (j0 + j) + 4 * lane, 4)
                        for j in range(count)]  # fmt: skip
            fetch(segments, vals, epoch, uniform=True)
            for j in cutlass.range_constexpr(count):
                for e in cutlass.range_constexpr(4):
                    p.xs[warp, 128 * (j0 + j) + 4 * lane + e] = BFloat16(vals[4 * j + e])
        self.csync.sync()

    # ------------------------------------------------------------- attention

    @cute.jit
    def split_blocks(self, layer, pos, split):
        """``[blk0, blk1)``: the 64-key context blocks of KV split ``split``, an even share of
        the blocks holding the cached positions before ``pos`` (local layers: the last
        ``window`` of them)."""
        lo = Int32(0)
        for li in self.local:
            if layer == li:
                lo = cutlass.max(pos - self.window, Int32(0))
        first = lo // BLOCK
        count = (pos + BLOCK - 1) // BLOCK - first
        blk0 = first + split * count // self.splits
        blk1 = first + (split + 1) * count // self.splits
        return blk0, blk1

    @cute.jit
    def rope_of(self, p: BackboneParams, position, lane):
        """``norm_rope4``'s cos and sin for lane l at ``position``: ``rope[2, 4]``."""
        rope = cute.make_rmem_tensor((2, 4), Float32)
        for e in cutlass.range_constexpr(4):
            i = 4 * (lane % 16) + e
            rope[0, e] = p.rope[position, i]
            rope[1, e] = p.rope[position, HEAD_DIM // 2 + i]
        return rope

    @cute.jit
    def norm_rope4(self, x, weights, rope, lane):
        """Lane l's dims 4 (l % 16) + e of the head (first half for l < 16, second for l >= 16,
        partners 16 lanes apart): per-head RMSNorm rounded to bf16, then RoPE (``rope_of``)
        rounded to bf16 (``dspark.head_norm_rope``). ``x``: 4 fp32 in place."""
        ss = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            ss = ss + x[e] * x[e]
        ss = group_sum(ss, 32)
        rinv = cute.math.rsqrt(ss * fp32_reciprocal(HEAD_DIM) + Float32(self.rms_eps))
        first = lane < 16
        for e in cutlass.range_constexpr(4):
            x[e] = bf16_round(Float32(weights[4 * lane + e]) * (x[e] * rinv))
        for e in cutlass.range_constexpr(4):
            other = cute.arch.shuffle_sync_bfly(x[e], offset=16)
            c, s = rope[0, e], rope[1, e]
            lo = x[e] * c - other * s
            hi = x[e] * c + other * s
            x[e] = bf16_round(cutlass.select_(first, lo, hi))
        return x

    @cute.jit
    def attention(self, p: BackboneParams, w, layer, kvh, split, pos, epoch, tid, warp, lane):
        """Flash partial of kv head ``kvh``'s q heads (rows ``head * TOKENS + token``) over KV
        split ``split`` on tensor cores; a split with no blocks (not the last) publishes an
        empty partial at once. See ``attention_setup``, ``score_block``, ``value_block``."""
        blk0, blk1 = self.split_blocks(layer, pos, split)
        last = split == self.splits - 1
        rows = self.attn_rows
        item = (layer * self.kv_heads + kvh) * self.splits + split
        if (blk0 == blk1) & (not last):
            if tid < rows * 2:
                r = tid // 2
                if tid % 2 == 0:
                    publish(p.attn_ml, (item * rows + r) * 2, Float32(-1e30), epoch)
                else:
                    publish(p.attn_ml, (item * rows + r) * 2 + 1, Float32(0.0), epoch)
            for i in cutlass.range_constexpr(rows * HEAD_DIM // COMPUTE_THREADS):
                publish(p.attn_o, item * rows * HEAD_DIM + i * COMPUTE_THREADS + tid,
                        Float32(0.0), epoch)  # fmt: skip
        else:
            with self.trace_region("attn.wait"):
                for k in cutlass.range_constexpr(self.qkv_splits):
                    last_token = (
                        (layer * self.qkv_splits + k) * TOKENS + TOKENS - 1
                    ) * self.qkv_rows
                    self.wait_items(p.qkv, last_token + ROWS - 1, self.qkv_rows // ROWS, ROWS,
                                    epoch, warp, lane)  # fmt: skip
            w = self.attend(p, w, layer, kvh, blk0, blk1, last, item, pos, epoch, warp, lane)
        return w

    @cute.jit
    def attend(self, p: BackboneParams, w, layer, kvh, blk0, blk1, last, item, pos, epoch, warp,
               lane):  # fmt: skip
        """The flash partial over context blocks ``[blk0, blk1)`` (and the block's own keys
        when ``last``), published. Rows: 2 m16 tiles (rows past ``attn_rows`` are zero q).
        Per 64-key block warp w scores keys ``8w ..`` (``score_block``), the CTA shares the
        rows' block max through RMAX and the bf16 probabilities through P, and warp w
        accumulates dims ``16w ..`` (``value_block``). Thread (g, t) keeps rows g, g + 8 of
        each m16 tile: their running max (all threads agree), its partial sums, and its O."""
        g = lane // 4
        t = lane % 4
        with self.trace_region("attn.setup"):
            qa = self.attention_setup(p, layer, kvh, pos, epoch, warp, lane)
        lo = Int32(0)
        for li in self.local:
            if layer == li:
                lo = pos - self.window
        m = cute.make_rmem_tensor((4,), Float32)  # rows g, g + 8, 16 + g, 24 + g
        lsum = cute.make_rmem_tensor((4,), Float32)
        o = cute.make_rmem_tensor((2, 2, 4), Float32)  # m16 tile, n8 tile, fragment
        m.fill(-1e30)
        lsum.fill(0.0)
        o.fill(0.0)
        for blk in cutlass.range(blk0, blk1, unroll=1):
            with w.consume() as hk, self.trace_region("attn.qk"):
                sc = self.score_block(hk.view("KV"), qa, warp, lane)
            for j in cutlass.range_constexpr(2):
                kp = blk * BLOCK + 8 * warp + 2 * t + j
                valid = (kp >= lo) & (kp < pos)
                for i in cutlass.range_constexpr(4):
                    sc[i, j] = cutlass.select_(valid, sc[i, j], -Float32.inf)
            with self.trace_region("attn.softmax"):
                self.block_softmax(p, sc, m, lsum, o, warp, lane)
            with w.consume() as hv, self.trace_region("attn.pv"):
                base = blk * BLOCK
                o = self.value_block(p, hv.view("KV"), 4, o, warp, lane, (lo - base, pos - base))
        if last:  # the block's own keys: KB/VB rows 0 .. 7 for warp 0's n8 tile
            sc = self.score_block(p.kb, qa, Int32(0), lane)
            for i in cutlass.range_constexpr(4):
                for j in cutlass.range_constexpr(2):
                    sc[i, j] = cutlass.select_(warp == 0, sc[i, j], -Float32.inf)
            self.block_softmax(p, sc, m, lsum, o, warp, lane)
            o = self.value_block(p, p.vb, 1, o, warp, lane)
        # Row sums: over the 4 lanes of a row, then over the warps through RMAX.
        for i in cutlass.range_constexpr(4):
            lsum[i] = group_sum(lsum[i], 4)
            if t == 0:
                p.rmax[16 * (i // 2) + 8 * (i % 2) + g, warp] = lsum[i]
        self.csync.sync()
        rows = self.attn_rows
        for i in cutlass.range_constexpr(4):
            r = 16 * (i // 2) + 8 * (i % 2) + g
            peers = cute.make_rmem_tensor((COMPUTE_WARPS,), Float32)
            cute.autovec_copy(p.rmax[r, None], peers)
            total = Float32(0.0)
            for ww in cutlass.range_constexpr(COMPUTE_WARPS):
                total = total + peers[ww]
            if r < rows:
                for nt in cutlass.range_constexpr(2):
                    d = 16 * warp + 8 * nt + 2 * t
                    for j in cutlass.range_constexpr(2):
                        index = (item * rows + r) * HEAD_DIM + d + j
                        publish(p.attn_o, index, o[i // 2, nt, 2 * (i % 2) + j], epoch)
                if (warp == 0) & (t == 0):
                    publish(p.attn_ml, (item * rows + r) * 2, m[i], epoch)
                    publish(p.attn_ml, (item * rows + r) * 2 + 1, total, epoch)
        # The next ATTN item rewrites Q, KB, VB, P and RMAX.
        self.csync.sync()
        return w

    @cute.jit
    def attention_setup(self, p: BackboneParams, layer, kvh, pos, epoch, warp, lane):
        """Q rows (q_norm, RoPE) and the block's keys (k_norm, RoPE) and values in SMEM, warp
        n on token n; then this thread's q fragments (``score_block``'s A operand) of both m16
        tiles and the 4 32-dim chunks: ``qa[tile, chunk, word]`` (words 0-3: row g, 4-7: row
        g + 8)."""
        g_n = self.group
        n = warp
        position = pos + n
        rope = self.rope_of(p, position, lane)
        # The projections: the sum of the split-K partials, rounded to bf16 (F.linear's).
        per = 4 * g_n + 8  # words per split: the q heads', then k's and v's
        bases = [((layer * self.qkv_splits + k) * TOKENS + n) * self.qkv_rows
                 for k in range(self.qkv_splits)]  # fmt: skip
        k_col = self.q_dim + kvh * HEAD_DIM + 4 * lane
        segments = [
            (p.qkv, base + offset, 4)
            for base in bases
            for offset in [(kvh * g_n + hl) * HEAD_DIM + 4 * lane for hl in range(g_n)]
            + [k_col, k_col + self.kv_heads * HEAD_DIM]
        ]
        parts = cute.make_rmem_tensor((self.qkv_splits * per,), Float32)
        fetch(segments, parts, epoch, uniform=True)
        qv = cute.make_rmem_tensor((4 * g_n,), Float32)
        kv = cute.make_rmem_tensor((8,), Float32)
        for e in cutlass.range_constexpr(4 * g_n):
            qv[e] = parts[e]
            for k in cutlass.range_constexpr(1, self.qkv_splits):
                qv[e] = qv[e] + parts[k * per + e]
        for e in cutlass.range_constexpr(8):
            kv[e] = parts[4 * g_n + e]
            for k in cutlass.range_constexpr(1, self.qkv_splits):
                kv[e] = kv[e] + parts[k * per + 4 * g_n + e]
        for e in cutlass.range_constexpr(4 * g_n):
            qv[e] = bf16_round(qv[e])
        for e in cutlass.range_constexpr(8):
            kv[e] = bf16_round(kv[e])
        for hl in cutlass.range_constexpr(g_n):
            x = cute.make_rmem_tensor((4,), Float32)
            for e in cutlass.range_constexpr(4):
                x[e] = qv[4 * hl + e]
            x = self.norm_rope4(x, p.q_norm[layer, None], rope, lane)
            for e in cutlass.range_constexpr(4):
                p.qs[hl * TOKENS + n, 4 * lane + e] = BFloat16(x[e])
        x = cute.make_rmem_tensor((4,), Float32)
        for e in cutlass.range_constexpr(4):
            x[e] = kv[e]
        x = self.norm_rope4(x, p.k_norm[layer, None], rope, lane)
        for e in cutlass.range_constexpr(4):
            p.kb[n, 4 * lane + e] = BFloat16(x[e])
            p.vb[n, 4 * lane + e] = BFloat16(kv[4 + e])
        self.csync.sync()
        g = lane // 4
        t = lane % 4
        qa = cute.make_rmem_tensor((2, 4, 8), Int32)
        for mt in cutlass.range_constexpr(2):
            for c in cutlass.range_constexpr(HEAD_DIM // 32):
                lo_words = load_words16(p.qs[16 * mt + g, None], 32 * c + 8 * t)
                hi_words = load_words16(p.qs[16 * mt + g + 8, None], 32 * c + 8 * t)
                for e in cutlass.range_constexpr(4):
                    qa[mt, c, e] = lo_words[e]
                    qa[mt, c, 4 + e] = hi_words[e]
        return qa

    @cute.jit
    def score_block(self, keys, qa, warp, lane):
        """Log2-domain scores of keys ``8 warp + 2t, + 1`` (rows of ``keys``) for this thread's
        4 rows: ``sc[row, key]``. q and k load as 16-byte vectors of the same 8 dims, so the
        mma's k order is a permutation of the dims, the same for both operands."""
        g = lane // 4
        t = lane % 4
        acc = cute.make_rmem_tensor((2, 4), Float32)
        acc.fill(0.0)
        for c in cutlass.range_constexpr(HEAD_DIM // 32):
            kw = load_words16(keys[8 * warp + g, None], 32 * c + 8 * t)
            for mt in cutlass.range_constexpr(2):
                a = [acc[mt, e] for e in range(4)]
                a = mma_bf16(a, (qa[mt, c, 0], qa[mt, c, 4], qa[mt, c, 1], qa[mt, c, 5]),
                             (kw[0], kw[1]))  # fmt: skip
                a = mma_bf16(a, (qa[mt, c, 2], qa[mt, c, 6], qa[mt, c, 3], qa[mt, c, 7]),
                             (kw[2], kw[3]))  # fmt: skip
                for e in cutlass.range_constexpr(4):
                    acc[mt, e] = a[e]
        scale = fp32(HEAD_DIM**-0.5 * LOG2E)
        sc = cute.make_rmem_tensor((4, 2), Float32)
        for i in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(2):
                sc[i, j] = acc[i // 2, 2 * (i % 2) + j] * scale
        return sc

    @cute.jit
    def block_softmax(self, p: BackboneParams, sc, m, lsum, o, warp, lane):
        """The CTA's online-softmax step for a block's scores ``sc`` (this thread's 4 rows x
        keys ``8 warp + 2t, + 1``): block max through RMAX, rescale ``lsum`` and ``o``, then
        the bf16 probabilities to P. Ends with the barrier after which P is complete."""
        g = lane // 4
        t = lane % 4
        for i in cutlass.range_constexpr(4):
            bm = cutlass.max(sc[i, 0], sc[i, 1])
            bm = cutlass.max(bm, cute.arch.shuffle_sync_bfly(bm, offset=1))
            bm = cutlass.max(bm, cute.arch.shuffle_sync_bfly(bm, offset=2))
            if t == 0:
                p.rmax[16 * (i // 2) + 8 * (i % 2) + g, warp] = bm
        self.csync.sync()
        for i in cutlass.range_constexpr(4):
            r = 16 * (i // 2) + 8 * (i % 2) + g
            peers = cute.make_rmem_tensor((COMPUTE_WARPS,), Float32)
            cute.autovec_copy(p.rmax[r, None], peers)
            bm = peers[0]
            for ww in cutlass.range_constexpr(1, COMPUTE_WARPS):
                bm = cutlass.max(bm, peers[ww])
            m_new = cutlass.max(m[i], bm)
            alpha = cute.math.exp2(m[i] - m_new)
            m[i] = m_new
            p0 = cute.math.exp2(sc[i, 0] - m_new)
            p1 = cute.math.exp2(sc[i, 1] - m_new)
            lsum[i] = lsum[i] * alpha + p0 + p1
            for nt in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(2):
                    o[i // 2, nt, 2 * (i % 2) + j] = o[i // 2, nt, 2 * (i % 2) + j] * alpha
            pair = cute.make_rmem_tensor((2,), BFloat16)
            pair[0] = BFloat16(p0)
            pair[1] = BFloat16(p1)
            cute.autovec_copy(pair, cute.local_tile(p.probs[r, None], (2,), (4 * warp + t,)))
        self.csync.sync()

    @cute.jit
    def value_block(self, p: BackboneParams, values, ksteps: cutlass.Constexpr, o, warp, lane,
                    valid=None):  # fmt: skip
        """``o += P @ V`` over the first ``16 ksteps`` keys of ``values`` (rows: keys) for dims
        ``16 warp ..`` (two n8 tiles); P as the A operand (32-bit loads), V's key pairs packed
        from 16-bit loads. ``valid``: None (every row) or the rows ``[lo, hi)`` the scores
        kept; the others' probabilities are zero, but a stale cache row may hold NaN or inf
        and 0 * NaN is NaN, so they enter the product as zeros."""
        g = lane // 4
        t = lane % 4
        for ks in cutlass.range_constexpr(ksteps):
            k0 = 16 * ks + 2 * t
            keys = (k0, k0 + 1, k0 + 8, k0 + 9)
            a = cute.make_rmem_tensor((2, 4), Int32)
            for mt in cutlass.range_constexpr(2):
                for half in cutlass.range_constexpr(2):
                    for kk in cutlass.range_constexpr(2):
                        pair = cute.make_rmem_tensor((2,), BFloat16)
                        src = p.probs[16 * mt + 8 * half + g, None]
                        cute.autovec_copy(cute.local_tile(src, (2,), ((k0 + 8 * kk) // 2,)), pair)
                        a[mt, 2 * kk + half] = cute.recast_tensor(pair, Int32)[0]
            for nt in cutlass.range_constexpr(2):
                d = 16 * warp + 8 * nt + g
                vals = cute.make_rmem_tensor((4,), BFloat16)
                for i in cutlass.range_constexpr(4):
                    vals[i] = values[keys[i], d]
                    if cutlass.const_expr(valid is not None):
                        kept = (keys[i] >= valid[0]) & (keys[i] < valid[1])
                        vals[i] = BFloat16(cutlass.select_(kept, Float32(vals[i]), Float32(0.0)))
                bw = cute.recast_tensor(vals, Int32)
                for mt in cutlass.range_constexpr(2):
                    acc = [o[mt, nt, e] for e in range(4)]
                    acc = mma_bf16(acc, (a[mt, 0], a[mt, 1], a[mt, 2], a[mt, 3]), (bw[0], bw[1]))
                    for e in cutlass.range_constexpr(4):
                        o[mt, nt, e] = acc[e]
        return o

    @cute.jit
    def merge(self, p: BackboneParams, layer, hq, epoch, warp, lane):
        """q head ``hq``'s attention output: warp n merges token n's split partials (lane l
        dims 4l .. 4l + 3; lane l fetches split l % splits' (m, l), shared by shuffles, then
        the partial O in groups of 8 splits) -> bf16(sum o_s 2^(m_s - m) / sum l_s 2^(m_s -
        m)), published."""
        splits = self.splits
        kvh = hq // self.group
        r = (hq % self.group) * TOKENS + warp
        rows = self.attn_rows
        item0 = (layer * self.kv_heads + kvh) * splits
        ml = cute.make_rmem_tensor((2,), Float32)
        fetch([(p.attn_ml, ((item0 + lane % splits) * rows + r) * 2, 2)], ml, epoch, uniform=True)
        ms = cute.make_rmem_tensor((splits,), Float32)
        mx = Float32(-1e30)
        for s in cutlass.range_constexpr(splits):
            ms[s] = cute.arch.shuffle_sync(ml[0], offset=s)
            mx = cutlass.max(mx, ms[s])
        den = Float32(0.0)
        for s in cutlass.range_constexpr(splits):
            ms[s] = cute.math.exp2(ms[s] - mx)
            den = den + cute.arch.shuffle_sync(ml[1], offset=s) * ms[s]
        num = cute.make_rmem_tensor((4,), Float32)
        num.fill(0.0)
        group = 8
        for s0 in cutlass.range_constexpr(0, splits, group):
            count = min(group, splits - s0)
            vals = cute.make_rmem_tensor((4 * count,), Float32)
            segments = [(p.attn_o, ((item0 + s0 + s) * rows + r) * HEAD_DIM + 4 * lane, 4)
                        for s in range(count)]  # fmt: skip
            fetch(segments, vals, epoch, uniform=True)
            for s in cutlass.range_constexpr(count):
                for e in cutlass.range_constexpr(4):
                    num[e] = num[e] + vals[4 * s + e] * ms[s0 + s]
        out = (layer * TOKENS + warp) * self.q_dim + hq * HEAD_DIM + 4 * lane
        for e in cutlass.range_constexpr(4):
            publish(p.attn_out, out + e, bf16_round(num[e] / den), epoch)

    # ------------------------------------------------------------- residuals

    @cute.jit
    def reduce_down(self, p: BackboneParams, layer, n, col, own, epoch):
        """The down projection at token ``n``, column ``col``: the splits' partials in order
        (this item's last), bf16, plus the residual bf16(h_mid) -> the next layer's stream."""
        h = self.hidden
        ds = self.down_splits
        vals = cute.make_rmem_tensor((ds,), Float32)
        segments = [
            (p.d_part, ((layer * ds + s) * TOKENS + n) * h + col, 1) for s in range(ds - 1)
        ] + [(p.h_mid, (layer * TOKENS + n) * h + col, 1)]
        fetch(segments, vals, epoch, uniform=True)
        acc = own
        if cutlass.const_expr(ds > 1):  # vals[0] is the residual when the item is the only split
            acc = vals[0]
            for s in cutlass.range_constexpr(1, ds - 1):
                acc = acc + vals[s]
            acc = acc + own
        total = bf16_round(acc) + bf16_round(vals[ds - 1])
        publish(p.h_pre, ((layer + 1) * TOKENS + n) * h + col, total, epoch)

    @cute.jit
    def final(self, p: BackboneParams, epoch, warp, lane):
        """The mask rows' final norm: warp n (n >= 1) -> ``hidden[n - 1]``."""
        h = self.hidden
        h_tiles = h // ROWS
        self.wait_items(p.h_pre, self.sentinel(self.layers, ROWS - 1), h_tiles, ROWS, epoch,
                        warp, lane)  # fmt: skip
        if warp > 0:
            chunks = h // 128
            vals = cute.make_rmem_tensor((4 * chunks,), Float32)
            self.fetch_rows(p.h_pre, (self.layers * TOKENS + warp) * h, h, vals, epoch, lane)
            ss = Float32(0.0)
            for i in cutlass.range(4 * chunks, unroll_full=True):
                ss = ss + vals[i] * vals[i]
            ss = group_sum(ss, 32)
            rinv = cute.math.rsqrt(ss * fp32_reciprocal(h) + Float32(self.rms_eps))
            self.store_row(p.hidden, warp - 1, h, vals, rinv, p.final_norm, lane)

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(self, tma, gtensors, scratch, tensors, bound):
        layouts = self.ring_layouts() | {
            "KV": kv_stages(self.stages),
            "X": x_layout(self.x_width),
            "RED": red_layout(REDUCTIONS),
            "GRED": cute.make_layout((GRED_FLOATS,)),
            "Q": cute.make_layout((ATTN_ROWS, HEAD_DIM), stride=(HEAD_DIM + QPAD, 1)),
            "KB": cute.make_layout((2 * TOKENS, HEAD_DIM), stride=(HEAD_DIM, 1)),
            "VB": cute.make_layout((2 * TOKENS, HEAD_DIM), stride=(HEAD_DIM, 1)),
            "P": cute.make_layout((ATTN_ROWS, BLOCK), stride=(BLOCK + QPAD, 1)),
            "RMAX": cute.make_layout((ATTN_ROWS, COMPUTE_WARPS), stride=(COMPUTE_WARPS, 1)),
        }
        self.begin_roles(work=bound, layouts=layouts)
        tensors = {k: cute.make_tensor(t.iterator, t.layout) for k, t in zip(TENSOR_ARGS, tensors)}
        trace_buffer = tensors.pop("trace_buffer")
        tagged = tagged_views(scratch, self.scratch_shapes)
        boxes = [("WS", (ROWS, BOX))] * 4 + [("KV", (BLOCK, HEAD_DIM))] * 2
        parts = [tma_partition(a, self.buffer(name), g, box)
                 for a, g, (name, box) in zip(tma, gtensors, boxes)]  # fmt: skip
        params = BackboneParams(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            sWS=StageView(parts[0][0], axis=cute.rank(parts[0][0]) - 1),
            sKV=StageView(parts[4][0], axis=1),
            xs=self.buffer("X"),
            red=self.buffer("RED"),
            gred=self.buffer("GRED"),
            qs=self.buffer("Q"),
            kb=self.buffer("KB"),
            vb=self.buffer("VB"),
            probs=self.buffer("P"),
            rmax=self.buffer("RMAX"),
            **tagged,
            **tensors,
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def __call__(
        self,
        wqkv: cute.Tensor,
        wo: cute.Tensor,
        wgu: cute.Tensor,
        wd: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        embed: cute.Tensor,
        embed_norm: cute.Tensor,
        mask_embed: cute.Tensor,
        input_norm: cute.Tensor,
        post_norm: cute.Tensor,
        q_norm: cute.Tensor,
        k_norm: cute.Tensor,
        final_norm: cute.Tensor,
        rope: cute.Tensor,
        token: cute.Tensor,
        pos: cute.Tensor,
        hidden: cute.Tensor,
        trace_buffer: cute.Tensor,
        scratch: cute.Tensor,
        state: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        pairs = [gemv_atom(src) for src in (wqkv, wo, wgu, wd)]
        pairs += [tma_atom(src, (BLOCK, HEAD_DIM)) for src in (k_cache, v_cache)]
        bound = self.work.bind(table, workspace, max_ctas)
        tensors = (trace_buffer, embed, embed_norm, mask_embed, input_norm, post_norm, q_norm, k_norm,
                   final_norm, rope, token, pos, hidden, state)  # fmt: skip
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tuple(a for a, _ in pairs), tuple(g for _, g in pairs), scratch, tensors, bound
        ).launch(**self.launch_config(bound), stream=stream, cooperative=True)


@jit_cache
def compile_draft_backbone_kernel(op: DraftBackboneKernel):
    """One compile serves every position and cache capacity (a multiple of ``BLOCK``): they
    are read at run time."""
    bf16, f32 = BFloat16, Float32
    h, layers = op.hidden, op.layers
    cache_rows = cute.sym_int(divisibility=BLOCK)
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (layers * op.qkv_rows, h)),  # wqkv
        make_fake_compact_tensor(bf16, (layers * h, op.q_dim)),  # wo
        make_fake_compact_tensor(bf16, (layers * 2 * op.intermediate, h)),  # wgu, interleaved
        make_fake_compact_tensor(bf16, (layers * h, op.intermediate)),  # w_down
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),  # k cache
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),  # v cache
        make_fake_compact_tensor(bf16, (cute.sym_int(), h)),  # embedding table
        make_fake_compact_tensor(bf16, (h,)),  # embed norm
        make_fake_compact_tensor(bf16, (h,)),  # mask embed
        make_fake_compact_tensor(bf16, (layers, h)),  # input norms
        make_fake_compact_tensor(bf16, (layers, h)),  # post norms
        make_fake_compact_tensor(bf16, (layers, HEAD_DIM)),  # q norms
        make_fake_compact_tensor(bf16, (layers, HEAD_DIM)),  # k norms
        make_fake_compact_tensor(bf16, (h,)),  # final norm
        make_fake_compact_tensor(f32, (cute.sym_int(divisibility=BLOCK), HEAD_DIM)),  # rope
        make_fake_compact_tensor(Int64, (1,), assumed_align=8),  # anchor
        make_fake_compact_tensor(Int64, (1,), assumed_align=8),  # pos
        make_fake_compact_tensor(bf16, (TOKENS, h)),  # hidden out
        make_fake_compact_tensor(Int64, (cute.sym_int(),)),  # trace buffer (globaltimer)
        make_fake_compact_tensor(Int64, (scratch_offsets(op.scratch_shapes)[1],)),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )
