"""One Inkling 2 Mini visit over ``tokens`` consecutive tokens of one sequence (2 to 8; 8 is
the DSpark verify block) as one persistent warpdrive launch (``InklingMultiVisitKernel``):
``visit_kernel``'s launch with a token block in place of the token, as a subclass that
overrides its rows.

Token t sits at position ``pos + t``. The launch implements ``ref.attention_block`` and
``ref.mlp_block`` on ``(tokens, hidden)`` inputs (``start=None``), causal inside the block::

    CTA t       token t's one-token prologue (block residuals, RMSNorm), published; every
                CTA copies all tokens' rows into xs (query q, token t at row q * tokens + t)
                (``shared_rows``)
    [messages]  MSGDOWN, GATEMIX over the block's tokens, then CTA t gathers token t's
                gated streams and every CTA copies them (``shared_rows``)
    QKV         32-row tiles x hidden x all tokens   -> bf16 projections per token (tagged)
    ATTN        kv head x KV split, all tokens: each item convolves the block's k/v from
                its raw projections once for the group (token t's history reaches back into
                the block and the rings), clips and norms them; scores the group's q heads
                for every token (as MMA rows) against the split's cached keys (positions <
                pos) on tensor cores (``attention_group``) and, in the newest split (the
                first without cached blocks if any), the block's new keys from SMEM with the
                causal mask; flash partials per q head and token;
                the head's last items to finish (counted per kv head) each merge some q
                heads' tokens over the splits and publish the gated o_proj inputs
    OPROJ       row tile x half (up to 4 tokens, if a half's inputs fit a row of xs; else
                quarter) of the q heads, all tokens (inputs fetched merged into ``OIN``, a
                view of xs with rows of the split's width); the last split's item reduces
                all tokens: the post-attention convolution runs over the block through the
                ring
    ROUTER CTAs token t's router input (its MLP prologue, or with messages its gathered MLP
                stream) on each of the 4 CTAs holding a router tile of token t, the residual
                polled after everything else of the prologue; the tile-0 CTA shares it
    [messages]  MSGDOWN, GATEMIX of the MLP gate over the MLP prologue (CTA t computes and
                shares token t's, as the attention's)
    ROUTER      32 router rows x one token (on the CTA's own row); then the CTA copies the
                other tokens' rows
    union       every CTA, before its first expert row: routes every token (a warp each)
                from the published logits and builds the union of their experts, in
                ascending order, as slots with token lists and weights (CTA 0 publishes it)
    GATEUP      union slot u x tile: the expert's 16 gate + 16 up rows x its tokens
    DOWN        DOWN_TILES row tiles x slot u: 32 down rows each x its tokens' activations
                (fetched once per item)
    REDUCE      row tile: each token's top-k down outputs in ascending expert order,
                post-MLP convolution over the block, residual -> output

GEMVs stream each weight tile once for the whole block, on tensor cores with the tokens as the
MMA's N (``examples.megakernel.multi_gemv``); quantized experts give the bf16-dequantized experts'
results bit for bit. A slot past the union's size is skipped by both roles alike (the load warp
reads the size and the slots' experts once). The cache rows of the block's positions are stored by
OPROJ row tile 0's items after acquiring the TMA reads of the KV block holding ``pos``
(``acquire_pos_block``), as in the one-token launch; ``pos + tokens <= max_len`` is the caller's
contract and rows at or past ``max_len`` are not stored. The rings must hold the block plus the
convolution history (``tokens + conv_kernel - 1 <= RING``).

xs (``x_rows`` rows, padded for ``examples.megakernel.multi_gemv``) also holds, during ATTN rows,
the items' head vectors and scores (fp32: ``qkvs_view``, ``scores_view``);
OPROJ inputs alternate between rows ``0 .. tokens`` and ``tokens .. 2 tokens`` of its view
``OIN``; the MLP reads its inputs from rows ``0 .. tokens`` and one expert's activations from
rows ``tokens .. 2 tokens``.
"""

import math

import cutlass
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass._mlir_helpers.arith import bitcast
from cutlass.cutlass_dsl import T

from attn_gym.warpdrive import NamedBarrier
from attn_gym.warpdrive.examples.inkling_megakernel.attn_half import (
    ATTN,
    GATE,
    KDYN,
    O_SPLITS,
    OPROJ,
    QKV,
    REL,
    SINK,
    VDYN,
    AttentionHalf,
    K,
    Q,
    V,
    clip_factor,
)
from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    GATE_RANK,
    GATE_SPLITS,
    GATE_TILES,
    GATEMIX,
    GATES,
    MAX_BLOCKS,
    MIX_FLOATS,
    MIX_STAGES,
    MLP_GATE,
    MSGDOWN,
    RING,
    STATES,
    GateParams,
)
from attn_gym.warpdrive.examples.inkling_megakernel.mlp_half import (
    AMAX_RED,
    DOWN,
    GATEUP,
    ROUTER,
    MlpHalf,
)
from attn_gym.warpdrive.examples.inkling_megakernel.visit_kernel import (
    InklingQuantVisitKernel,
    InklingVisitKernel,
)
from attn_gym.warpdrive.examples.megakernel.act_quant import (
    block_max,
    cta_max,
    f32_op,
    pack_bf16x2,
    quantize_chunk,
    quantize_rows,
)
from attn_gym.warpdrive.examples.megakernel.constants import (
    BACKOFF_NS,
    BLOCK,
    BOX,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    DBUF_FLOATS,
    HEAD_DIM,
    LOG2E,
    MMA_N,
    QBOX,
    ROWS,
    ROWS_PER_WARP,
    TILE_K,
    X_PAD,
)
from attn_gym.warpdrive.examples.megakernel.formats import BF16, NVFP4
from attn_gym.warpdrive.examples.megakernel.multi_gemv import (
    finish,
    gemv_bf16,
    gemv_quant,
    ldmatrix_x4,
    mma_bf16,
)
from attn_gym.warpdrive.examples.megakernel.regmath import (
    bf16_round,
    cta_sum,
    fp32,
    fp32_reciprocal,
    group_sum,
    load_frag,
)
from attn_gym.warpdrive.examples.megakernel.ring import (
    box_layout,
    load_boxes,
    load_tiles,
    swizzled_atom,
    tma_partition,
)
from attn_gym.warpdrive.examples.megakernel.tagged import (
    advance_epoch,
    fetch,
    load_vector,
    load_words,
    publish,
    unpack_tagged,
)

REDUCE = 9  # after the message-gate ops
MAX_TOKENS = 8
UNION_EXPERTS = 4  # union words: the size at 0, slot experts from here, then counts, entries
DOWN_TILES = 6  # down row tiles per DOWN item: one activation fetch and quantization serve them
# ``shared_rows`` kinds.
ATTN_PROLOGUE, MLP_PROLOGUE, ATTN_GATHER, MLP_GATHER = range(4)
SCORE_STRIDE = BLOCK + 8  # floats per token row of SCORES (padded: rows in distinct banks)
LAST_FLAG = 383  # RED word of an ATTN item's last-split flag (``merge_if_last``)
CHOSEN_WORD = 128  # RED word of ``route``'s first chosen-expert word (``chosen_words``)


def ordered_key(x):
    """An Int32 that orders as the fp32 ``x`` does (``x`` not NaN; -0 below +0)."""
    word = Int32(bitcast(Float32(x).ir_value(), T.i32()))
    return word ^ ((word >> 31) & Int32(0x7FFFFFFF))


@cute.jit
def group_max(v, width: cutlass.Constexpr):
    """Maximum over aligned groups of ``width`` lanes (``regmath.group_sum``'s shuffles)."""
    for step in cutlass.range_constexpr(width.bit_length() - 1):
        v = cutlass.max(v, cute.arch.shuffle_sync_bfly(v, offset=1 << step))
    return v


def pick(values: list, i):
    """``values[i]`` for a dynamic ``i`` < ``len(values)`` (a select chain)."""
    out = values[0]
    for j in range(1, len(values)):
        out = cutlass.select_(i == j, values[j], out)
    return out


class InklingMultiVisitKernel(InklingVisitKernel):
    """A visit over ``tokens`` consecutive tokens (see the module docstring); the other
    constructor arguments are ``InklingVisitKernel``'s."""

    def __init__(self, *, tokens: int, **kwargs):
        if not 2 <= tokens <= MAX_TOKENS:
            raise ValueError(f"tokens must be in [2, {MAX_TOKENS}], got {tokens}")
        self.tokens = tokens
        super().__init__(**kwargs)
        # o_proj split-K: halves of the q heads up to 4 tokens (one OPROJ item per CTA; with 8
        # the fetched inputs would not fit the registers) if a half's inputs fit a row of xs
        # (``OIN``), else quarters (whose geometry ``init_attention`` checked).
        halves = tokens <= 4 and self.q_heads // 2 * HEAD_DIM <= self.hidden
        self.o_splits = 2 if halves else O_SPLITS
        self.heads_per_split = self.q_heads // self.o_splits
        self.oproj_k_tiles = self.heads_per_split * HEAD_DIM // TILE_K
        # Compute warps arrive once they built the union, the load warp syncs (once per launch).
        self.union_built = NamedBarrier(threads=COMPUTE_THREADS + 32)
        if tokens + self.conv_kernel - 1 > RING:
            raise ValueError("the rings must hold the block and the convolution history")
        if self.conv_kernel != 4:
            raise ValueError("a lane loads a convolution's 4 taps of its 4 channels as vectors")
        if (self.hidden // ROWS) % DOWN_TILES:
            raise ValueError(f"DOWN items take {DOWN_TILES} row tiles of the hidden size")
        if self.queries * tokens > self.x_rows:
            raise ValueError("the prologue's inputs exceed xs")
        if self.d_rel % 16:
            raise ValueError("the relative bias runs on m16n8k16 MMAs: d_rel must be 16k")
        if self.extent < tokens or (not self.full and self.window < tokens):
            raise ValueError(
                "the block's new keys are scored at every distance below tokens: need a "
                f"window and relative-bias extent of at least {tokens}"
            )
        if self.experts > COMPUTE_THREADS:
            raise ValueError(
                f"the union counts one expert per compute thread and a lane's chosen experts "
                f"in 8 bits: need at most {COMPUTE_THREADS} experts"
            )
        stride = self.hidden + self.x_pad
        if 2 * (self.tokens * self.group_words + 2 * self.score_floats) > self.x_rows * stride:
            raise ValueError("the ATTN items' head vectors and scores exceed xs")

    # ------------------------------------------------------------------ sizes

    @property
    def x_rows(self) -> int:
        return max(self.streams * self.tokens, 2 * self.tokens)

    @property
    def mix_floats(self) -> int:
        return self.tokens * MIX_FLOATS

    qkvs_floats = 4  # unused: the items' head vectors live in xs (``qkvs_view``)

    @property
    def route_rows(self) -> int:
        return self.tokens

    @property
    def amax_floats(self) -> int:
        """AMAX: each token's quantized MLP input's (a DOWN item keeps its rows' in registers:
        the next item may rewrite them before every thread finished reading)."""
        return self.tokens

    @property
    def slots(self) -> int:
        """Union slots: at most every token's top k distinct experts."""
        return min(self.experts, self.tokens * self.top_k)

    @property
    def slot_words(self) -> int:
        """Union words of the slots' experts and of their counts: slots padded to lanes."""
        return -(-self.slots // 32) * 32

    @property
    def entry_stride(self) -> int:
        """Union words of a slot's token entries (4-word aligned)."""
        return 4 if self.tokens <= 4 else 8

    @property
    def union_words(self) -> int:
        return UNION_EXPERTS + 2 * self.slot_words + self.slots * self.entry_stride

    @property
    def extra_smem_bytes(self) -> int:
        return 4 * self.union_words

    x_pad = X_PAD  # (``examples.megakernel.multi_gemv``)

    def extra_buffers(self):
        self.smem_plan.buffer("UNI", Float32, align=16)
        self.smem_plan.buffer("OIN", BFloat16, align=16, overlaps="X")

    def extra_stage_buffers(self, w) -> dict:
        """Swizzled views of the weight stages for the GEMVs' TMA boxes
        (``examples.megakernel.multi_gemv``)."""
        views = {"WS": self.smem_plan.buffer("WS", BFloat16, align=1024, overlaps=w)}
        if self.quantized:
            views["QWS"] = self.smem_plan.buffer("QWS", Int32, align=1024, overlaps=w)
        return views

    def extra_layouts(self) -> dict:
        width = self.heads_per_split * HEAD_DIM
        layouts = {
            "UNI": cute.make_layout((self.union_words,)),
            # Two buffers of the OPROJ inputs, ``tokens`` rows each (``fetch_o_inputs``).
            "OIN": cute.make_layout((2 * self.tokens, width), stride=(width + self.x_pad, 1)),
            "WS": box_layout(BFloat16, self.stages),
        }
        if self.quantized:
            layouts["QWS"] = box_layout(Int32, self.stages)
        return layouts

    def weight_target(self) -> tuple:
        return "WS", (ROWS, BOX)

    def weight_atom(self, src):
        return swizzled_atom(src, BFloat16)

    def mlp_targets(self) -> list:
        targets = [("WS", (ROWS, BOX))]
        if not self.quantized:
            return targets * 3
        targets += [("QWS", (ROWS, QBOX))] * 2
        return targets + [("S", (1, self.meta_subtiles, self.scale_words))] * 2

    def target_atom(self, name: str, src, box: tuple[int, ...]) -> tuple:
        if name in ("WS", "QWS"):
            return swizzled_atom(src, BFloat16 if name == "WS" else Int32)
        return MlpHalf.target_atom(self, name, src, box)

    @cute.jit
    def gate_params(self, sW, tma, gtensors, tagged: dict, tensors: dict):
        if cutlass.const_expr(not self.messages):
            return None
        parts = [
            tma_partition(tma[0], self.buffer("WS"), gtensors[0], (ROWS, BOX)),
            tma_partition(tma[1], sW, gtensors[1], (ROWS, TILE_K)),
        ]
        return GateParams(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            h_part=tagged.pop("h_part"),
            gated=tagged.pop("gated"),
            mhalf=tensors["mhalf"],
            messages=tensors["messages"],
        )

    def get_name(self) -> str:
        return (
            super()
            .get_name()
            .replace("warpdrive_inkling_visit_", f"warpdrive_inkling_t{self.tokens}_")
        )

    def attention_rows(self) -> list[list[int]]:
        rows = [row for row in super().attention_rows() if row[0] == QKV]
        rows += [[ATTN, kvh, s] for s in range(self.splits) for kvh in range(self.kv_heads)]
        return rows + [[OPROJ, t, s] for s in range(self.o_splits) for t in range(self.row_tiles)]

    def attention_scratch_shapes(self) -> dict[str, int]:
        shapes = {k: self.tokens * size for k, size in super().attention_scratch_shapes().items()}
        # The merged o_proj inputs and each q head's count of finished ATTN items.
        nt, h = self.tokens, self.hidden
        return shapes | {
            "o_in": nt * self.q_heads * HEAD_DIM,
            "count": self.q_heads,
            # ``shared_rows``: the attention's and the MLP's input rows (bf16, 4 per word),
            # MIX (fp32, 2 per word) and each token's flags.
            "xin": self.shared_layout(MLP_GATHER + 1)[0] * h // 4,
            "xmix": nt * MIX_FLOATS,
            "xamax": nt,  # each token's quantized router input's amax (``own_rows``)
            "flags": 4 * nt,
        }

    def mlp_scratch_shapes(self) -> dict[str, int]:
        t, k = self.tokens, self.top_k
        return {
            "logits": t * self.experts,
            "union": self.union_words,
            "act": self.slots * t * self.width,
            "down": t * k * self.hidden,
        }

    def gate_scratch_shapes(self) -> dict[str, int]:
        return {name: self.tokens * size for name, size in super().gate_scratch_shapes().items()}

    # ------------------------------------------------------------------ roles

    @cute.jit
    def load_rows(self, p, w, meta, work):
        w = w.views(KV=p.sKV, WS=p.sWS)
        if cutlass.const_expr(self.messages):
            w = w.views(W=p.sW)  # GATEMIX's [F | G] rows
        if cutlass.const_expr(self.quantized):
            w = w.views(QWS=p.sQWS)
            meta = meta.views(S=p.sS)
        pos = Int32(p.pos[0])
        lane = cute.arch.lane_idx()
        union = Int32(-1)  # the union's size once polled
        slot_experts = cute.make_rmem_tensor((self.slot_words // 32,), Int32)
        slot_experts.fill(0)
        while work.valid:
            op, a, b, _row = work.item
            if op < ROUTER:
                w = self.load_attention(p, w, op, a, b, pos)
            elif op == ROUTER:
                w = self.load_stages(
                    w, p.tma_mlp[0], p.gmem_mlp[0], a, 0, self.hidden_k_tiles, "W"
                )
            elif (op == GATEUP) | (op == DOWN):
                if union < 0:
                    with self.trace_region("route"):
                        union = self.read_union(p, slot_experts, lane)
                if b < union:
                    expert = self.slot_expert(slot_experts, b)
                    if op == GATEUP:
                        w, meta = self.load_expert(p, w, meta, op, a, expert)
                    else:
                        for i in cutlass.range_constexpr(DOWN_TILES):
                            w, meta = self.load_expert(p, w, meta, op, a * DOWN_TILES + i, expert)
            if cutlass.const_expr(self.messages):
                if (op == MSGDOWN) | (op == GATEMIX):
                    w = self.load_gate_item(p, w, op, a, b)
            work.advance()
        return w, meta, work

    @cute.jit
    def load_stages(self, w, atom, tiles, row, k0, count: cutlass.Constexpr,
                    buffer: cutlass.Constexpr):  # fmt: skip
        """GEMV weight stages as swizzled boxes into ``buffer``'s swizzled view (``W`` -> ``WS``,
        ``QW`` -> ``QWS``; ``examples.megakernel.ring.load_boxes``)."""
        return load_boxes(w, atom, tiles, row, k0, count, {"W": "WS", "QW": "QWS"}[buffer])

    @cute.jit
    def compute_rows(self, p, w, meta, work, me):
        tid, warp, lane = me.tid, me.warp, me.lane
        epoch = p.state[0] + 1
        pos = Int32(p.pos[0])
        with self.trace_region("normalize"):
            self.shared_rows(p, epoch, tid, warp, lane, ATTN_PROLOGUE)
        self.stage_scales(p, tid)
        obuf = Int32(self.tokens)
        union = Int32(-1)
        mlp = Int32(0)
        gathered = Int32(0)
        while work.valid:
            op, a, b, item = work.item
            if (mlp == 0) & self.is_mlp_row(op, a, b):
                with self.trace_region("mlp.prologue", payload=item):
                    self.mlp_rows(p, op, a, b, epoch, tid, warp, lane, MLP_PROLOGUE)
                mlp = Int32(1)
            if cutlass.const_expr(self.messages):
                w, gathered = self.message_row(
                    p, w, op, a, b, gathered, epoch, tid, warp, lane, item
                )
            if op < ROUTER:
                w, obuf = self.attention_row(
                    p, w, op, a, b, item, obuf, pos, epoch, tid, warp, lane
                )
            elif (op < MSGDOWN) | (op == REDUCE):
                w, meta, union = self.mlp_row_tokens(
                    p, w, meta, op, a, b, union, pos, epoch, tid, warp, lane, item
                )
            work.advance()
        advance_epoch(p.state, epoch, tid)
        return w, meta, work

    @cute.jit
    def is_mlp_row(self, op, a, b):
        return InklingVisitKernel.is_mlp_row(self, op, a, b) | (op == REDUCE)

    @cute.jit
    def message_row(self, p, w, op, a, b, gathered, epoch, tid, warp, lane, item):
        """``InklingVisitKernel.message_row`` with the gathers of ``shared_rows`` and
        ``mlp_rows``."""
        if op == MSGDOWN:
            w = self.msgdown(p, w, a, b, epoch, warp, lane)
        elif op == GATEMIX:
            w = self.gatemix(p, w, a, b, epoch, warp, lane)
        elif (op < ROUTER) & (gathered == 0):
            with self.trace_region("gather", payload=item):
                self.shared_rows(p, epoch, tid, warp, lane, ATTN_GATHER)
            gathered = Int32(1)
        elif (op >= ROUTER) & (gathered < 2):
            with self.trace_region("gather", payload=item):
                self.mlp_rows(p, op, a, b, epoch, tid, warp, lane, MLP_GATHER)
            gathered = Int32(2)
        return w, gathered

    # ------------------------------------------------------------ prologues

    @cute.jit
    def mlp_rows(self, p, op, a, b, epoch, tid, warp, lane, kind: cutlass.Constexpr):
        """Before this CTA's first MLP row ``(op, a, b)``: the MLP's input rows of ``kind``
        (``shared_rows``). The router CTAs compute the rows of ``router_kind``: each its
        router tile's token's (tile 0's CTA also shares them), copying the other tokens' after
        its router GEMV (``mlp_row_tokens``); the other CTAs copy them."""
        if cutlass.const_expr(kind == self.router_kind):
            if op == ROUTER:
                self.own_rows(p, b, a == 0, epoch, tid, warp, lane, kind)
            else:
                self.copy_rows(p, epoch, tid, kind)
        else:
            self.shared_rows(p, epoch, tid, warp, lane, kind)

    def shared_layout(self, kind: int) -> tuple[int, int]:
        """``(first row in xin, rows per token)`` of a ``shared_rows`` kind: the attention
        prologue's rows, the MLP prologue's, the attention gather's, the MLP gather's (kind 4:
        ``(all rows, 0)``)."""
        rows = (self.queries, 1, self.streams if self.messages else 0, 1 if self.messages else 0)
        return self.tokens * sum(rows[:kind]), (rows + (0,))[kind]

    @property
    def router_kind(self) -> int:
        """The ``shared_rows`` kind of the router's input rows: the MLP gather's with message
        gates, else the MLP prologue's. Router CTAs compute their token's rows of this kind
        themselves (``own_rows``) and the other CTAs only copy them."""
        return MLP_GATHER if self.messages else MLP_PROLOGUE

    @cute.jit
    def shared_rows(self, p, epoch, tid, warp, lane, kind: cutlass.Constexpr):
        """The tokens' GEMV input rows of ``kind`` (``shared_layout``; not ``router_kind``:
        ``mlp_rows``), each computed once (``own_rows``: by CTA t < tokens for token t), then
        copied by every CTA (``copy_rows``)."""
        cta = cute.arch.block_idx()[0]
        if cta < self.tokens:
            self.own_rows(p, cta, cutlass.Boolean(True), epoch, tid, warp, lane, kind)
        self.copy_rows(p, epoch, tid, kind)

    def xmix_floats(self, p):
        """``xmix`` as fp32."""
        ptr = cute.recast_ptr(p.xmix.iterator, dtype=Float32)
        return cute.make_tensor(ptr, cute.make_layout((2 * self.tokens * MIX_FLOATS,)))

    @cute.jit
    def own_rows(self, p, t, share, epoch, tid, warp, lane, kind: cutlass.Constexpr):
        """Token ``t``'s rows of ``kind`` (``token_rows``) into this CTA's xs rows (row ``q *
        tokens + t``) and MIX; with ``share``, also to ``xin``/``xmix`` (plain stores, a
        fence, then the token's tagged flag) for ``copy_rows``. At layer 0 the attention
        prologue's owner also stores token t's embedding to h. The router's input rows
        (``router_kind``) are shared quantized (``MlpHalf.quantize_input``'s block values, from
        xs row ``tokens + t``, free until the DOWN rows), with their amax (``xamax``): the
        router tiles read their own token's row as it is, the GATEUP rows the shared ones."""
        nt, h = self.tokens, self.hidden
        first_row, nq = self.shared_layout(kind)
        mixes = kind < ATTN_GATHER and self.messages
        quantize = kind == self.router_kind
        chunks = h // 8  # 16-byte chunks of a row: two words of xin
        with self.trace_region("rows.own"):
            self.token_rows(p, t, epoch, tid, warp, lane, kind)
            if share:
                source = t
                if cutlass.const_expr(quantize):
                    source = nt + t
                    amax = quantize_rows(
                        self.activation_format, p.xs, t, source, 1, 1, h, p.red, AMAX_RED,
                        self.csync, tid, warp, lane,
                    )[0]  # fmt: skip
                    if cutlass.const_expr(self.activation_format == NVFP4):
                        if tid == 0:
                            publish(p.xamax, t, amax, epoch)
                    self.csync.sync()
                for i in cutlass.range_constexpr(-(-nq * chunks // COMPUTE_THREADS)):
                    c = i * COMPUTE_THREADS + tid
                    if c < nq * chunks:
                        q = c // chunks
                        chunk = c - q * chunks
                        row = cute.recast_tensor(p.xs[q * nt + source, None], Int64)
                        index = (first_row + q * nt + t) * chunks + chunk
                        for k in cutlass.range_constexpr(2):
                            p.xin[2 * index + k] = row[2 * chunk + k]
                if cutlass.const_expr(mixes):
                    if tid < MIX_FLOATS:
                        xmix = self.xmix_floats(p)
                        xmix[(kind * nt + t) * MIX_FLOATS + tid] = p.mix[t * MIX_FLOATS + tid]
                self.csync.sync()
                if tid == 0:
                    cute.arch.fence_acq_rel_gpu()
                    publish(p.flags, kind * nt + t, Float32(1.0), epoch)

    @cute.jit
    def copy_rows(self, p, epoch, tid, kind: cutlass.Constexpr):
        """Once warp 0 saw every token's flag of ``kind`` (and fenced: an acquire), every
        CTA copies the published rows into its own xs rows (row ``q * tokens + t``) and MIX
        with gpu-scope loads, and the router input's amax per token into AMAX."""
        nt, h = self.tokens, self.hidden
        first_row, nq = self.shared_layout(kind)
        mixes = kind < ATTN_GATHER and self.messages
        chunks = h // 8
        if tid < 32:
            seen = cute.make_rmem_tensor((1,), Float32)
            fetch([(p.flags, kind * nt + cutlass.min(tid, nt - 1), 1)], seen, epoch, uniform=True, backoff=BACKOFF_NS)  # fmt: skip
            cute.arch.fence_acq_rel_gpu()
        self.csync.sync()
        total = nt * nq * chunks  # the kind's rows are contiguous in xin
        for i in cutlass.range_constexpr(-(-total // COMPUTE_THREADS)):
            c = i * COMPUTE_THREADS + tid
            if c < total:
                xrow = c // chunks
                words = load_vector(p.xin.iterator + 2 * (first_row * chunks + c), 2)
                row = cute.recast_tensor(p.xs[xrow, None], Int64)
                for k in cutlass.range_constexpr(2):
                    row[2 * (c - xrow * chunks) + k] = words[k]
        if cutlass.const_expr(mixes):
            xmix = self.xmix_floats(p).iterator + kind * nt * MIX_FLOATS
            for i in cutlass.range_constexpr(-(-nt * MIX_FLOATS // COMPUTE_THREADS)):
                c = i * COMPUTE_THREADS + tid
                if c < nt * MIX_FLOATS:
                    p.mix[c] = cute.arch.load(xmix + c, Float32, sem="relaxed", scope="gpu")
        if cutlass.const_expr((kind == self.router_kind) & (self.activation_format == NVFP4)):
            if tid < nt:
                amax = cute.make_rmem_tensor((1,), Float32)
                fetch([(p.xamax, tid, 1)], amax, epoch)
                p.amax[tid] = amax[0]
        self.csync.sync()

    @cute.jit
    def token_rows(self, p, t, epoch, tid, warp, lane, kind: cutlass.Constexpr):
        """Token ``t``'s rows of ``shared_rows`` kind ``kind`` into xs rows ``q * tokens + t``
        (and its MIX): the attention's one-token prologue (``AttentionHalf.attention_prologue``;
        at layer 0 storing the embedding to h), the MLP's (``InklingVisitKernel.mlp_prologue``,
        polling the residual), or a gather of its gated streams (``common.gather``)."""
        per = self.hidden // COMPUTE_THREADS
        if cutlass.const_expr(kind >= ATTN_GATHER):
            self.gather_token(p, t, kind == MLP_GATHER, epoch, tid, warp, lane)
        elif cutlass.const_expr(kind == MLP_PROLOGUE):
            self.mlp_token_rows(p, t, epoch, tid, warp, lane)
        else:
            x = self.load_states(p, p.num_blocks, p.h, tid, p.embed_input == 0, token=t)
            if p.embed_input != 0:
                self.embed_token(p, x, MAX_BLOCKS, False, tid, warp, lane, token=t)
                for r in cutlass.range_constexpr(per):
                    p.h[t * self.hidden + r * COMPUTE_THREADS + tid] = BFloat16(x[MAX_BLOCKS, r])
            self.normalize_states(
                p, x, p.num_blocks, p.attn_query, p.attn_norms, self.queries, p.queries, tid,
                warp, lane, token=t,
            )  # fmt: skip

    @cute.jit
    def mlp_token_rows(self, p, t, epoch, tid, warp, lane):
        """Token ``t``'s MLP prologue (``normalize_states`` of the MLP query over the MLP's
        blocks and the polled residual) into xs row t and its MIX, with everything that does
        not need the residual done before polling it: the blocks' loads (at layer 0 the
        recomputed embedding), norms and query dots, the query and norm weights. After the
        residual arrives one reduction gives its square sum and its raw query dot (scaled by
        its inverse norm afterwards, which rounds differently from dotting the normalized
        state), then the block residual and its RMSNorm."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        inv_h = fp32_reciprocal(h)
        weighted = not self.messages
        mix0 = t * MIX_FLOATS
        num_blocks = p.num_blocks + p.boundary
        x = self.load_states(p, num_blocks - p.embed_input, None, tid, token=t)
        if p.embed_input != 0:
            self.embed_token(p, x, 0, False, tid, warp, lane, token=t)
        qv = cute.make_rmem_tensor((per,), Float32)
        weights = cute.make_rmem_tensor((per,), BFloat16)
        for r in cutlass.range_constexpr(per):
            qv[r] = p.mlp_query[0, r * COMPUTE_THREADS + tid]
            if cutlass.const_expr(weighted):
                weights[r] = p.post_norm[0, r * COMPUTE_THREADS + tid]
        squares = []
        for n in cutlass.range_constexpr(MAX_BLOCKS):
            ss = Float32(0.0)
            for r in cutlass.range_constexpr(per):
                ss = ss + x[n, r] * x[n, r]
            squares.append(ss)
        squares = cta_sum(p.red, self.csync, squares, 0, warp, lane)
        rinvs = []
        dots = []
        for n in cutlass.range_constexpr(MAX_BLOCKS):
            rinvs.append(cute.math.rsqrt(squares[n] * inv_h + Float32(1e-6)))
            dot = Float32(0.0)
            for r in cutlass.range_constexpr(per):
                x[n, r] = x[n, r] * rinvs[n]
                dot = dot + qv[r] * x[n, r]
            dots.append(dot)
        dots = cta_sum(p.red, self.csync, dots, 1, warp, lane)
        resid = cute.make_rmem_tensor((per,), Float32)
        segments = [(p.resid, t * h + r * COMPUTE_THREADS + tid, 1) for r in range(per)]
        fetch(segments, resid, epoch, uniform=True, backoff=BACKOFF_NS)
        ss = Float32(0.0)
        raw = Float32(0.0)
        for r in cutlass.range_constexpr(per):
            ss = ss + resid[r] * resid[r]
            raw = raw + qv[r] * resid[r]
        ss, raw = cta_sum(p.red, self.csync, [ss, raw], 2, warp, lane)
        rinvs.append(cute.math.rsqrt(ss * inv_h + Float32(1e-6)))
        for r in cutlass.range_constexpr(per):
            x[MAX_BLOCKS, r] = resid[r] * rinvs[MAX_BLOCKS]
        dots.append(raw * rinvs[MAX_BLOCKS])
        scale = fp32(h**-0.5)
        scores = []
        top = Float32(-Float32.inf)
        for n in cutlass.range_constexpr(STATES):
            sc = dots[n] * scale
            if cutlass.const_expr(n < MAX_BLOCKS):
                sc = cutlass.select_(n < num_blocks, sc, -Float32.inf)
            scores.append(sc)
            top = cutlass.max(top, sc)
        total = Float32(0.0)
        exps = []
        for n in cutlass.range_constexpr(STATES):
            exps.append(cute.math.exp(scores[n] - top))
            total = total + exps[n]
        probs = [exps[n] / total for n in range(STATES)]
        y = cute.make_rmem_tensor((per,), Float32)
        ss = Float32(0.0)
        for r in cutlass.range_constexpr(per):
            acc = Float32(0.0)
            for n in cutlass.range_constexpr(STATES):
                acc = acc + probs[n] * x[n, r]
            y[r] = acc
            ss = ss + acc * acc
        if cutlass.const_expr(self.messages):
            if tid == 0:
                for n in cutlass.range_constexpr(STATES):
                    p.mix[mix0 + STATES + n] = probs[n]  # query 0's weights
                    p.mix[mix0 + n] = rinvs[n]
        # (RED slot 0's readers passed slot 2's barrier.)
        ss = cta_sum(p.red, self.csync, [ss], 0, warp, lane)[0]
        rinv = cute.math.rsqrt(ss * inv_h + Float32(self.rms_eps))
        for r in cutlass.range_constexpr(per):
            k = r * COMPUTE_THREADS + tid
            if cutlass.const_expr(weighted):
                p.xs[t, k] = BFloat16(Float32(weights[r]) * (y[r] * rinv))
            else:
                p.xs[t, k] = BFloat16(y[r] * rinv)
        self.csync.sync()

    # --------------------------------------------------------------- attention

    @property
    def group_rows(self) -> int:
        """An ATTN item's query rows: its kv head's q heads for every token, row ``j * tokens
        + t`` for q head ``j`` of the group and token ``t``."""
        return self.group * self.tokens

    @property
    def m_tiles(self) -> int:
        """The ATTN items' 16-row MMA tiles of query rows."""
        return -(-self.group_rows // 16)

    @property
    def group_words(self) -> int:
        """An ATTN item's head vectors of one token (fp32 words, ``group_offsets``): the
        group's q heads, k, v (HEAD_DIM each), the group's relative states (d_rel each), then
        in full layers the dynamic k and v taps (kernel each)."""
        g = self.group
        return (g + 2) * HEAD_DIM + g * self.d_rel + (2 * self.conv_kernel if self.full else 0)

    def group_offsets(self) -> tuple[int, int, int, int]:
        """The words of k, v, the relative states and the dynamic taps in ``group_words``."""
        g, d = self.group, HEAD_DIM
        return g * d, (g + 1) * d, (g + 2) * d, (g + 2) * d + g * self.d_rel

    def qkvs_view(self, p):
        """The ATTN items' head vectors: fp32, token t's ``group_words`` from word
        ``t * group_words``, at the start of xs (free during ATTN items)."""
        ptr = cute.recast_ptr(p.xs.iterator, dtype=Float32)
        return cute.make_tensor(ptr, cute.make_layout((self.tokens * self.group_words,)))

    def rows_of(self, first):
        """The MMA's input rows of the tokens' consecutive xs rows from ``first`` (the last
        token's repeated past the block)."""
        return [first + min(t, self.tokens - 1) for t in range(MMA_N)]

    def dbuf(self, p):
        """``examples.megakernel.multi_gemv.finish``'s partial tiles, in ATT (free between ATTN
        items)."""
        return cute.make_tensor(p.att.iterator, cute.make_layout(DBUF_FLOATS))

    @cute.jit
    def gemv(self, p, w, xs, rows, col0, k_tiles: cutlass.Constexpr, warp, lane, tid):
        """``(value, w)``: thread ``tid``'s value (``examples.megakernel.multi_gemv.finish``) of
        a bf16 weight tile against the xs ``rows`` (``gemv_bf16``)."""
        acc, w = gemv_bf16(w, xs, rows, col0, k_tiles, warp, lane)
        return finish(acc, self.dbuf(p), self.csync, warp, lane, tid), w

    @cute.jit
    def attention_row(self, p, w, op, a, b, item, obuf, pos, epoch, tid, warp, lane):
        nt = self.tokens
        row, t = tid // MMA_N, tid % MMA_N
        if op == QKV:
            rows = self.rows_of(self.stream_of(b) * nt)
            v, w = self.gemv(p, w, p.xs, rows, 0, self.hidden_k_tiles, warp, lane, tid)
            if t < nt:
                index = t * self.proj_size + self.proj_offset(b) + a * ROWS + row
                publish(p.proj, index, bf16_round(v), epoch)
        elif op == ATTN:
            blk0, blk1, newest = self.split_blocks(pos, b)
            if (blk0 == blk1) & (b != newest):
                self.publish_empty_group(p, a, b, epoch, tid)
            else:
                w = self.attention_group(p, w, a, b, pos, epoch, tid, warp, lane, item)
            with self.trace_region("merge", payload=item):
                self.merge_if_last(p, a, epoch, tid, warp, lane)
        else:
            obuf = nt - obuf
            with self.trace_region("o.fetch", payload=item):
                self.fetch_o_inputs(p, b, obuf, epoch, tid)
                if a == 0:
                    self.acquire_pos_block(p, b, pos, epoch, tid)
                self.csync.sync()
            if a == 0:
                self.store_kv_tokens(p, b, pos, epoch, tid)
            v, w = self.gemv(p, w, self.buffer("OIN"), self.rows_of(obuf), 0, self.oproj_k_tiles,
                             warp, lane, tid)  # fmt: skip
            if t < nt:
                index = (t * self.o_splits + b) * self.hidden + a * ROWS + row
                publish(p.o_part, index, v, epoch)
            if b == self.o_splits - 1:
                with self.trace_region("reduce", payload=item):
                    self.reduce_attention_tokens(p, a, pos, epoch, tid, lane)
        return w, obuf

    @cute.jit
    def load_attention(self, p, w, op, a, b, pos):
        """``AttentionHalf.load_attention`` with an ATTN item per kv head ``a``."""
        if op == ATTN:
            max_len = p.k_cache.shape[0] // self.kv_heads
            blk0, blk1, _newest = self.split_blocks(pos, b)
            row0 = a * (max_len // BLOCK)
            for blk in cutlass.range(blk0, blk1, unroll=1):
                w = load_tiles(w, p.tma_kv[0], p.gmem_kv[0], row0 + blk, 0, 1, "KV")
                w = load_tiles(w, p.tma_kv[1], p.gmem_kv[1], row0 + blk, 0, 1, "KV")
        else:
            w = AttentionHalf.load_attention(self, p, w, op, a, b, pos)
        return w

    @cute.jit
    def split_blocks(self, pos, split):
        """``AttentionHalf.split_blocks``, except that when a split is left without cached
        blocks, the first such one scores the new keys (and writes them) alone, in parallel
        with the splits reading the cache."""
        blk0, blk1, _newest = AttentionHalf.split_blocks(self, pos, split)
        _first, n, per = self.split_span(pos)
        newest = cutlass.min((n + per - 1) // per, Int32(self.splits - 1))
        return blk0, blk1, newest

    @cute.jit
    def split_span(self, pos):
        """``(first, n, per)`` of ``examples.megakernel.attention.split_blocks``: the first KV
        block of the cached keys, their block count and the blocks per split."""
        lo = Int32(0)
        if cutlass.const_expr(not self.full):
            lo = cutlass.max(pos - (self.window - 1), Int32(0))
        first = lo // BLOCK
        n = (pos + BLOCK - 1) // BLOCK - first
        return first, n, cutlass.max((n + self.splits - 1) // self.splits, Int32(1))

    @cute.jit
    def attn_item(self, t, hq, split):
        """Index of token t's partial of q head ``hq`` over KV split ``split``."""
        return (t * self.q_heads + hq) * self.splits + split

    @cute.jit
    def lane_row(self, mt: cutlass.Constexpr, h: cutlass.Constexpr, g):
        """``(row, active, j, t)`` of MMA row ``16 mt + 8 h + g`` of an ATTN item: the query
        row (the last one past the rows), whether it is one, its q head in the group and its
        token."""
        r = 16 * mt + 8 * h + g
        rr = cutlass.min(r, Int32(self.group_rows - 1))
        j, t = self.head_token(rr)
        return rr, r < self.group_rows, j, t

    def head_token(self, r):
        """``(j, t)`` of query row ``r = j * tokens + t`` (``group_rows``)."""
        j = r // self.tokens
        return j, r - j * self.tokens

    @cute.jit
    def publish_empty_group(self, p, kvh, split, epoch, tid):
        """``publish_empty_partial`` for every query row of kv head ``kvh``."""
        rows = self.group_rows
        for k in cutlass.range_constexpr(-(-rows * HEAD_DIM // COMPUTE_THREADS)):
            e = k * COMPUTE_THREADS + tid
            r = e // HEAD_DIM
            d = e - r * HEAD_DIM
            if r < rows:
                j, t = self.head_token(r)
                item = self.attn_item(t, kvh * self.group + j, split)
                publish(p.attn_o, item * HEAD_DIM + d, Float32(0.0), epoch)
                if d == 0:
                    publish(p.attn_ml, 2 * item, Float32(-1e30), epoch)
                    publish(p.attn_ml, 2 * item + 1, Float32(0.0), epoch)

    @cute.jit
    def group_score(self, dot, bias):
        """``score`` of a whole fp32 dot product and its bf16 relative bias."""
        s = bf16_round(bf16_round(dot) * fp32(HEAD_DIM**-0.5))
        return bf16_round(s + bias) * LOG2E

    @property
    def row_pitch(self) -> int:
        """The query rows the ATTN items' MMA tiles span (per-warp maxima per buffer)."""
        return 16 * self.m_tiles

    @property
    def score_floats(self) -> int:
        """One SCORES buffer: the query rows' block scores (rows of ``SCORE_STRIDE``) and each
        warp's per-row maxima."""
        return self.group_rows * SCORE_STRIDE + COMPUTE_WARPS * self.row_pitch

    def scores_view(self, p):
        """``attention_group``'s two SCORES buffers, in xs after the head vectors."""
        ptr = cute.recast_ptr(p.xs.iterator, dtype=Float32) + self.tokens * self.group_words
        return cute.make_tensor(ptr, cute.make_layout((2 * self.score_floats,)))

    @cute.jit
    def rel_words(self, p, words, first, lane, count: cutlass.Constexpr):
        """Into ``words`` (``(count, d_rel // 16, 2)`` Int32): the lane's B words of
        ``bias_tiles`` for the distances ``first + n`` (``n < 8 count``, clamped to the
        extent), ``rel_proj`` rows as bf16 pairs; loaded ahead of their MMAs."""
        g, c = lane // 4, lane % 4
        rel = cute.recast_tensor(p.rel_proj, Int32)
        for i in cutlass.range_constexpr(count):
            d = cutlass.min(cutlass.max(first + 8 * i + g, Int32(0)), Int32(self.extent - 1))
            for s in cutlass.range_constexpr(self.d_rel // 16):
                for k in cutlass.range_constexpr(2):
                    words[i, s, k] = rel[d, 8 * s + c + 4 * k]

    @cute.jit
    def bias_tiles(self, sa, words, count: cutlass.Constexpr) -> list:
        """The relative-bias logits of the lane's query rows at ``rel_words``' distances
        (``ref``'s ``rel @ rel_proj`` on tensor cores: A the rows' relative states ``sa``, B
        ``words``): per 8-distance tile, per m tile the MMA's C fragment."""
        tiles = []
        for i in cutlass.range_constexpr(count):
            per_tile = []
            for mt in cutlass.range_constexpr(self.m_tiles):
                acc = [Float32(0.0)] * 4
                for s in cutlass.range_constexpr(self.d_rel // 16):
                    acc = mma_bf16(acc, [sa[mt, s, k] for k in range(4)],
                                   (words[i, s, 0], words[i, s, 1]))  # fmt: skip
                per_tile.append(acc)
            tiles.append(per_tile)
        return tiles

    @cute.jit
    def pick_bias(self, tiles, mt: cutlass.Constexpr, h: cutlass.Constexpr, column, odd, lane):
        """The bf16 bias of the lane's MMA row ``16 mt + 8 h + g`` at ``column`` of
        ``bias_tiles``' distances (in range; ``odd``: its parity): the C element sits in lane
        ``(g, column % 8 // 2)`` of the lane's group, register ``2 h + column % 2`` of tile
        ``column // 8``. Every lane must call it; the lanes of a group pass the same parity, so
        the source lane selects the register."""
        src = 4 * (lane // 4) + (column % 8) // 2
        value = Float32(0.0)
        for i in cutlass.range_constexpr(len(tiles)):
            v = cutlass.select_(odd, tiles[i][mt][2 * h + 1], tiles[i][mt][2 * h])
            v = cute.arch.shuffle_sync(v, offset=src)
            value = cutlass.select_(column // 8 == i, v, value)
        return bf16_round(value)

    @cute.jit
    def group_column(self, kvh, word):
        """The proj index of word ``word`` of kv head ``kvh``'s head vectors
        (``group_words``)."""
        d, r, c, group = HEAD_DIM, self.d_rel, self.conv_kernel, self.group
        k_word, v_word, rel_word, dyn_word = self.group_offsets()
        off = self.proj_offsets
        column = off[Q] + kvh * group * d + word
        column = cutlass.select_(word >= k_word, off[K] + kvh * d + word - k_word, column)
        column = cutlass.select_(word >= v_word, off[V] + kvh * d + word - v_word, column)
        column = cutlass.select_(word >= rel_word, off[REL] + kvh * group * r + word - rel_word,
                                 column)  # fmt: skip
        if cutlass.const_expr(self.full):
            column = cutlass.select_(word >= dyn_word, off[KDYN] + kvh * c + word - dyn_word,
                                     column)  # fmt: skip
            base = dyn_word + c
            column = cutlass.select_(word >= base, off[VDYN] + kvh * c + word - base, column)
        return column

    @cute.jit
    def prepare_group(self, p, qkvs, kvh, pos, writer, epoch, tid, warp, lane):
        """``prepare_heads`` for kv head ``kvh``'s group and the block: fetch every token's
        projections into ``qkvs``, then units spread over the warps (``group_units``)."""
        nt, words = self.tokens, self.group_words
        unit_inputs = self.group_unit_inputs(p, kvh, pos, warp, lane)
        total = nt * words
        fetches = -(-total // (2 * COMPUTE_THREADS))
        raw = cute.make_rmem_tensor((2 * fetches,), Float32)
        segments = []
        for k in cutlass.range_constexpr(fetches):
            e = cutlass.min(k * 2 * COMPUTE_THREADS + 2 * tid, total - 2)
            t = e // words
            column = t * self.proj_size + self.group_column(kvh, e - t * words)
            segments.append((p.proj, column, 2))
        fetch(segments, raw, epoch)
        # xs may still be read by the previous row's GEMV.
        self.csync.sync()
        for k in cutlass.range_constexpr(fetches):
            e = k * 2 * COMPUTE_THREADS + 2 * tid
            if e < total:
                qkvs[e] = raw[2 * k]
                qkvs[e + 1] = raw[2 * k + 1]
        self.csync.sync()
        self.group_units(p, qkvs, unit_inputs, kvh, pos, writer, warp, lane)
        self.csync.sync()

    def group_unit_rounds(self) -> tuple[int, int]:
        """``group_units``' rounds: of convolutions, of all units."""
        nt = self.tokens
        return -(-2 * nt // COMPUTE_WARPS), -(-self.group_units_total // COMPUTE_WARPS)

    @property
    def group_units_total(self) -> int:
        """``group_units``' units."""
        return (2 + self.group) * self.tokens + self.group

    @cute.jit
    def group_unit_inputs(self, p, kvh, pos, warp, lane):
        """The registers ``group_units`` reads besides ``qkvs``, loaded early: per round, the
        convolution's taps and ring rows (``load_conv_vectors``), or the q (or relative)
        scales in the rows' place."""
        nt, group, kernel = self.tokens, self.group, self.conv_kernel
        _, rounds = self.group_unit_rounds()
        inputs = []
        for rnd in cutlass.range_constexpr(rounds):
            u = rnd * COMPUTE_WARPS + warp
            tap = cute.make_rmem_tensor((4, kernel), p.k_taps.element_type)
            past = cute.make_rmem_tensor((4, kernel - 1), BFloat16)
            if u < 2 * nt:
                if u % 2 == 0:
                    self.load_conv_vectors(p.ring_k, p.k_taps, kvh, pos, lane, tap, past)
                else:
                    self.load_conv_vectors(p.ring_v, p.v_taps, kvh, pos, lane, tap, past)
            elif u < (2 + group) * nt:
                hq = kvh * group + (u - 2 * nt) // nt
                for e in cutlass.range_constexpr(4):
                    past[e, 0] = p.q_scale[hq, 4 * lane + e]
            elif (u < self.group_units_total) & (lane < self.d_rel):
                past[0, 0] = p.rel_scale[kvh * group + u - (2 + group) * nt, lane]
            inputs.append((tap, past))
        return inputs

    @cute.jit
    def load_conv_vectors(self, ring, taps, kvh, pos, lane, tap, past):
        """``load_conv`` in vector loads (every head unit loads its taps: scalar loads would
        flood L2 with requests)."""
        kernel = self.conv_kernel
        c0 = kvh * HEAD_DIM + 4 * lane
        for t in cutlass.range_constexpr(kernel):
            # Full: (kernel, kv_dim), tap t's 4 channels; sliding: (kv_dim, kernel), channel t.
            if cutlass.const_expr(self.full):
                v = load_frag(taps[t, None], c0, 4)
            else:
                v = load_frag(taps[c0 + t, None], 0, 4)
            for e in cutlass.range_constexpr(4):
                if cutlass.const_expr(self.full):
                    tap[e, t] = v[e]
                else:
                    tap[t, e] = v[e]
        for t in cutlass.range_constexpr(kernel - 1):
            row = (pos - 1 - t + RING) % RING
            v = load_frag(ring[row, None], c0, 4)
            for e in cutlass.range_constexpr(4):
                past[e, t] = v[e]

    @cute.jit
    def group_units(self, p, qkvs, inputs, kvh, pos, writer, warp, lane):
        """The group's head vectors in ``qkvs``, as ``(2 + group) tokens + group`` independent
        units, unit u on warp ``u % COMPUTE_WARPS``: the k (u = 2t) and v (2t + 1)
        convolutions of token t (``conv_token``), the q scale/clip of q head j and token t
        (unit ``2 tokens + j tokens + t``), then the relative states of each q head. The
        convolutions read earlier tokens' raw inputs, so their outputs overwrite the raw
        inputs only after a barrier."""
        nt, words, r, group = self.tokens, self.group_words, self.d_rel, self.group
        k_word, v_word, rel_word, dyn_word = self.group_offsets()
        d0 = 4 * lane
        conv_rounds, rounds = self.group_unit_rounds()
        outs = []
        for rnd in cutlass.range_constexpr(rounds):
            u = rnd * COMPUTE_WARPS + warp
            tap, past = inputs[rnd]
            out = cute.make_rmem_tensor((4,), Float32)
            if cutlass.const_expr(rnd < conv_rounds):
                outs.append(out)
            if u < 2 * nt:
                t = u // 2
                if u % 2 == 0:
                    self.conv_token(qkvs, p.ring_k, tap, past, k_word, dyn_word, kvh, pos,
                                    writer, lane, t, out, True)  # fmt: skip
                else:
                    self.conv_token(qkvs, p.ring_v, tap, past, v_word,
                                    dyn_word + self.conv_kernel, kvh, pos, writer, lane, t,
                                    out, False)  # fmt: skip
            elif u < (2 + group) * nt:
                j, t = self.head_token(u - 2 * nt)
                base = t * words + j * HEAD_DIM + d0
                x = cute.make_rmem_tensor((4,), Float32)
                ss = Float32(0.0)
                for e in cutlass.range_constexpr(4):
                    x[e] = qkvs[base + e] * Float32(past[e, 0])
                    ss = ss + x[e] * x[e]
                factor = clip_factor(group_sum(ss, 32), self.qk_clip)
                for e in cutlass.range_constexpr(4):
                    v = bf16_round(x[e] * factor)
                    if cutlass.const_expr(self.full):
                        v = bf16_round(v * self.log_scale(pos + t))
                    qkvs[base + e] = v
            elif u < self.group_units_total:
                j = u - (2 + group) * nt
                scale = cutlass.select_(lane < r, Float32(past[0, 0]), Float32(0.0))
                for t in cutlass.range_constexpr(nt):
                    index = t * words + rel_word + j * r + cutlass.min(lane, Int32(r - 1))
                    x = Float32(0.0)
                    if lane < r:
                        x = qkvs[index] * scale
                    factor = clip_factor(group_sum(x * x, 32), self.rel_clip)
                    v = bf16_round(x * factor)
                    if cutlass.const_expr(self.full):
                        v = bf16_round(v * self.log_scale(pos + t))
                    if lane < r:
                        qkvs[index] = v
        self.csync.sync()
        for rnd in cutlass.range_constexpr(conv_rounds):
            u = rnd * COMPUTE_WARPS + warp
            if u < 2 * nt:
                base = (u // 2) * words + cutlass.select_(u % 2 == 0, k_word, v_word) + d0
                for e in cutlass.range_constexpr(4):
                    qkvs[base + e] = outs[rnd][e]

    @cute.jit
    def conv_token(self, qkvs, ring, tap, past, base, dyn, kvh, pos, writer, lane, t, out,
                   is_key: cutlass.Constexpr):  # fmt: skip
        """One warp: ``conv_head`` of block token ``t`` into ``out`` (lane l on channels 4l ..
        4l + 3), clipped (``is_key``) or normed; its history is the earlier tokens' raw inputs
        in ``qkvs`` and the ring rows before ``pos`` in ``past`` (zero before position 0;
        ``load_conv``'s, with its ``tap``). The writer stores the token's raw input to the
        ring."""
        taps_n = self.conv_kernel
        words = self.group_words
        ss = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            col = base + 4 * lane + e
            y = qkvs[t * words + col]
            hist = [y]  # hist[i]: the input i positions back, zero before position 0
            for i in cutlass.range_constexpr(1, taps_n):
                ring_value = Float32(0.0)
                for k in cutlass.range_constexpr(taps_n - 1):
                    ring_value = cutlass.select_(i - t - 1 == k, Float32(past[e, k]), ring_value)
                ring_value = cutlass.select_(pos + t - i >= 0, ring_value, Float32(0.0))
                block_value = qkvs[cutlass.max(t - i, Int32(0)) * words + col]
                hist.append(cutlass.select_(t - i >= 0, block_value, ring_value))
            conv = Float32(0.0)
            if cutlass.const_expr(self.full):
                for k in cutlass.range_constexpr(taps_n):
                    conv = conv + hist[k] * (Float32(tap[e, k]) + qkvs[t * words + dyn + k])
                z = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(2 * taps_n + 1)))
            else:
                for k in cutlass.range_constexpr(taps_n):
                    conv = conv + tap[e, k] * hist[taps_n - 1 - k]
                z = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(taps_n + 1)))
            if writer:
                ring[(pos + t) % RING, kvh * HEAD_DIM + 4 * lane + e] = BFloat16(y)
            out[e] = z
            ss = ss + z * z
        ss = group_sum(ss, 32)
        if cutlass.const_expr(is_key):
            factor = clip_factor(ss, self.qk_clip)
        else:
            factor = cute.math.rsqrt(ss * fp32_reciprocal(HEAD_DIM) + Float32(self.rms_eps))
        for e in cutlass.range_constexpr(4):
            out[e] = bf16_round(out[e] * factor)

    @cute.jit
    def attention_group(self, p, w, kvh, split, pos, epoch, tid, warp, lane, item):
        """Flash-decoding partials of kv head ``kvh``'s q heads over KV split ``split`` for
        every token, on tensor cores (``mma.sync.m16n8k16``, the query rows ``group_rows``
        as the MMA's M rows in ``m_tiles`` tiles; rows past them are zero): the group's K/V
        blocks are read and its convolutions run once for its q heads.

        Per 64-key block, warp w scores keys ``8 w ..`` against every query row (its K rows
        through ``ldmatrix``; the relative bias from ``bias_tiles`` over the distances its
        keys have from the block's tokens) and writes the scores and their per-row maximum to
        SCORES; after a barrier every warp reads the block's scores, updates the online
        softmax (the same running max and sum in every warp) and accumulates P V for its 16
        dims ``16 w ..`` (V through ``ldmatrix.trans``; P rounds to bf16 as the oracle's
        probabilities do). The scores alternate between two buffers, so one barrier per block
        suffices. In the newest split every warp also scores the block's new keys from
        ``qkvs`` (causal). Each warp publishes its dims of every row's partial; warp 0 the
        running max and sum."""
        nt, words, rows = self.tokens, self.group_words, self.group_rows
        mts, pitch, sc_stride = self.m_tiles, self.row_pitch, SCORE_STRIDE
        k_word, _, rel_word, _ = self.group_offsets()
        g, c = lane // 4, lane % 4
        blk0, blk1, newest = self.split_blocks(pos, split)
        is_newest = split == newest
        qkvs = self.qkvs_view(p)
        scores = self.scores_view(p)
        key0 = 8 * warp
        ksteps = self.d_rel // 16
        # The bias's B words of the first block and the new keys, loaded before the waits.
        block_words = cute.make_rmem_tensor((2, ksteps, 2), Int32)
        new_words = cute.make_rmem_tensor((1, ksteps, 2), Int32)
        self.rel_words(p, block_words, pos - blk0 * BLOCK - key0 - 7, lane, 2)
        self.rel_words(p, new_words, Int32(0), lane, 1)
        with self.trace_region("attn.prepare", payload=item):
            self.prepare_group(p, qkvs, kvh, pos, is_newest, epoch, tid, warp, lane)
        zero = Int32(0)
        qa = cute.make_rmem_tensor((mts, HEAD_DIM // 16, 4), Int32)  # A words of the q rows
        sa = cute.make_rmem_tensor((mts, ksteps, 4), Int32)  # and of their relative states
        for mt in cutlass.range_constexpr(mts):
            for h in cutlass.range_constexpr(2):
                _rr, active, j, t = self.lane_row(mt, h, g)
                first = t * words + 2 * c
                self.a_words(qkvs, qa, mt, h, first + j * HEAD_DIM, active)
                self.a_words(qkvs, sa, mt, h, first + rel_word + j * self.d_rel, active)
        m = cute.make_rmem_tensor((mts, 2), Float32)  # per lane row: running max and sum
        lsum = cute.make_rmem_tensor((mts, 2), Float32)
        m.fill(-1e30)
        lsum.fill(0.0)
        o = cute.make_rmem_tensor((mts, 2, 4), Float32)  # dims 16 w + 8 n .. of rows g, g + 8
        o.fill(0.0)
        maxes = rows * sc_stride  # the per-warp maxima after the scores
        with self.trace_region("attn.blocks", payload=item):
            for blk in cutlass.range(blk0, blk1, unroll=1):
                buf = ((blk - blk0) % 2) * self.score_floats
                kb = blk * BLOCK + key0
                # Distances pos + t - key of the warp's keys: column t + 7 - (key - kb).
                tiles = self.bias_tiles(sa, block_words, 2)
                self.rel_words(p, block_words, pos - kb - BLOCK - 7, lane, 2)  # the next block's
                with w.consume() as h:
                    keys = h.view("KV")[key0 + lane % 8, None]
                    acc = [[Float32(0.0)] * 4 for _ in range(mts)]
                    for kk in cutlass.range_constexpr(HEAD_DIM // 32):
                        b = ldmatrix_x4(cute.local_tile(keys, (8,), (4 * kk + lane // 8,)), False)
                        for mt in cutlass.range_constexpr(mts):
                            for i in cutlass.range_constexpr(2):
                                a = [qa[mt, 2 * kk + i, k] for k in range(4)]
                                acc[mt] = mma_bf16(acc[mt], a, (b[2 * i], b[2 * i + 1]))
                    for mt in cutlass.range_constexpr(mts):
                        for hh in cutlass.range_constexpr(2):
                            rr, active, _j, t = self.lane_row(mt, hh, g)
                            lo = Int32(0)
                            if cutlass.const_expr(not self.full):
                                lo = pos + t - (self.window - 1)
                            mx = -Float32.inf
                            for e in cutlass.range_constexpr(2):
                                kp = kb + 2 * c + e
                                column = t + 7 - 2 * c - e
                                bias = self.pick_bias(tiles, mt, hh, column, (column % 2) == 1,
                                                      lane)  # fmt: skip
                                bias = cutlass.select_(
                                    pos + t - kp < self.extent, bias, Float32(0.0)
                                )
                                sc = self.group_score(acc[mt][2 * hh + e], bias)
                                sc = cutlass.select_((kp < pos) & (kp >= lo) & active, sc,
                                                     -Float32.inf)  # fmt: skip
                                mx = cutlass.max(mx, sc)
                                if active:
                                    scores[buf + rr * sc_stride + key0 + 2 * c + e] = sc
                            mx = group_max(mx, 4)
                            if active & (c == 0):
                                scores[buf + maxes + warp * pitch + rr] = mx
                self.csync.sync()
                pa = cute.make_rmem_tensor((mts, BLOCK // 16, 4), Int32)
                for mt in cutlass.range_constexpr(mts):
                    for hh in cutlass.range_constexpr(2):
                        rr, active, _j, _t = self.lane_row(mt, hh, g)
                        bm = -Float32.inf
                        for i in cutlass.range_constexpr(COMPUTE_WARPS):
                            bm = cutlass.max(bm, scores[buf + maxes + i * pitch + rr])
                        m_new = cutlass.max(m[mt, hh], bm)
                        part = Float32(0.0)
                        for ks in cutlass.range_constexpr(BLOCK // 16):
                            for half in cutlass.range_constexpr(2):
                                k = buf + rr * sc_stride + 16 * ks + 8 * half + 2 * c
                                p0 = cute.math.exp2(scores[k] - m_new)
                                p1 = cute.math.exp2(scores[k + 1] - m_new)
                                part = part + p0 + p1
                                pa[mt, ks, 2 * half + hh] = cutlass.select_(
                                    active, pack_bf16x2(p0, p1), zero
                                )
                        self.rescale(m, lsum, o, mt, hh, m_new, group_sum(part, 4))
                # Rows from pos are stale, and in sliding layers rows before the first
                # token's window are outside every token's, maybe NaN, and 0 * NaN is NaN:
                # their zero probabilities take V from the block's nearest visible row.
                first_row = Int32(0)
                if cutlass.const_expr(not self.full):
                    first_row = cutlass.max(pos - (self.window - 1) - blk * BLOCK, Int32(0))
                last_row = pos - 1 - blk * BLOCK
                with w.consume() as h:
                    values = h.view("KV")
                    for ks in cutlass.range_constexpr(BLOCK // 16):
                        row = 16 * ks + lane % 8 + 8 * ((lane // 8) % 2)
                        row = cutlass.min(cutlass.max(row, first_row), last_row)
                        dims = cute.local_tile(values[row, None], (8,), (2 * warp + lane // 16,))
                        b = ldmatrix_x4(dims, True)
                        for mt in cutlass.range_constexpr(mts):
                            a = [pa[mt, ks, k] for k in range(4)]
                            for n in cutlass.range_constexpr(2):
                                out = mma_bf16([o[mt, n, e] for e in range(4)], a,
                                               (b[2 * n], b[2 * n + 1]))  # fmt: skip
                                for e in cutlass.range_constexpr(4):
                                    o[mt, n, e] = out[e]
            if is_newest:
                self.attend_new_keys(qkvs, qa, sa, new_words, o, m, lsum, warp, lane)
        if is_newest:  # the writer: one item per kv head
            for k in cutlass.range_constexpr(-(-nt * 2 * HEAD_DIM // COMPUTE_THREADS)):
                e = k * COMPUTE_THREADS + tid
                t = e // (2 * HEAD_DIM)
                rest = e - t * 2 * HEAD_DIM
                kind = rest // HEAD_DIM
                d = rest - kind * HEAD_DIM
                if t < nt:
                    value = qkvs[t * words + k_word + kind * HEAD_DIM + d]
                    index = t * 2 * self.kv_dim + kind * self.kv_dim + kvh * HEAD_DIM + d
                    publish(p.kv_new, index, value, epoch)
        for mt in cutlass.range_constexpr(mts):
            for hh in cutlass.range_constexpr(2):
                _rr, active, j, t = self.lane_row(mt, hh, g)
                if active:
                    item_t = self.attn_item(t, kvh * self.group + j, split)
                    for n in cutlass.range_constexpr(2):
                        for e in cutlass.range_constexpr(2):
                            d = 16 * warp + 8 * n + 2 * c + e
                            publish(p.attn_o, item_t * HEAD_DIM + d, o[mt, n, 2 * hh + e], epoch)
                    if (warp == 0) & (c == 0):
                        # Release the TMA reads of the block of pos (rows pos .. are stored
                        # after the merges, ``attn_half``).
                        if (blk0 * BLOCK <= pos) & (pos < blk1 * BLOCK):
                            cute.arch.fence_proxy("async.global")
                            cute.arch.fence_acq_rel_gpu()
                        publish(p.attn_ml, 2 * item_t, m[mt, hh], epoch)
                        publish(p.attn_ml, 2 * item_t + 1, lsum[mt, hh], epoch)
        # The next attention row rewrites the scores and the head vectors.
        self.csync.sync()
        return w

    @cute.jit
    def attend_new_keys(self, qkvs, qa, sa, rel, o, m, lsum, warp, lane):
        """The newest split's block keys (``attention_group``'s layout): every warp scores
        the block's new keys (B words from ``qkvs``, key g's in lane group g) against every
        query row, causally (the bias at distances 0 .. 7 from ``bias_tiles``), folds them
        into its online softmax and accumulates their values for its dims (``o``, ``m`` and
        ``lsum`` update in place). ``rel``: ``rel_words`` of distances 0 .. 7."""
        nt, words = self.tokens, self.group_words
        k_word, v_word, _, _ = self.group_offsets()
        g, c = lane // 4, lane % 4
        zero = Int32(0)
        key = cutlass.min(g, Int32(nt - 1)) * words + k_word
        kb = []
        for ks in cutlass.range_constexpr(HEAD_DIM // 16):
            col = key + 16 * ks + 2 * c
            kb.append((pack_bf16x2(qkvs[col], qkvs[col + 1]),
                       pack_bf16x2(qkvs[col + 8], qkvs[col + 9])))  # fmt: skip
        vb = []
        for n in cutlass.range_constexpr(2):
            d = v_word + 16 * warp + 8 * n + g
            v0 = qkvs[cutlass.min(2 * c, Int32(nt - 1)) * words + d]
            v1 = qkvs[cutlass.min(2 * c + 1, Int32(nt - 1)) * words + d]
            vb.append((pack_bf16x2(v0, v1), zero))
        tiles = self.bias_tiles(sa, rel, 1)
        for mt in cutlass.range_constexpr(self.m_tiles):
            acc = [Float32(0.0)] * 4
            for ks in cutlass.range_constexpr(HEAD_DIM // 16):
                acc = mma_bf16(acc, [qa[mt, ks, k] for k in range(4)], kb[ks])
            pw = [zero, zero]  # P words of rows g, g + 8
            for hh in cutlass.range_constexpr(2):
                _rr, active, _j, t = self.lane_row(mt, hh, g)
                sc = []
                mx = -Float32.inf
                for e in cutlass.range_constexpr(2):
                    key_t = 2 * c + e
                    distance = t - key_t
                    bias = self.pick_bias(tiles, mt, hh, cutlass.max(distance, Int32(0)),
                                          ((t + e) % 2) == 1, lane)  # fmt: skip
                    s = self.group_score(acc[2 * hh + e], bias)
                    sc.append(cutlass.select_((key_t <= t) & active, s, -Float32.inf))
                    mx = cutlass.max(mx, sc[e])
                m_new = cutlass.max(m[mt, hh], group_max(mx, 4))
                p0 = cute.math.exp2(sc[0] - m_new)
                p1 = cute.math.exp2(sc[1] - m_new)
                pw[hh] = pack_bf16x2(p0, p1)
                self.rescale(m, lsum, o, mt, hh, m_new, group_sum(p0 + p1, 4))
            a = (pw[0], pw[1], zero, zero)
            for n in cutlass.range_constexpr(2):
                out = mma_bf16([o[mt, n, e] for e in range(4)], a, vb[n])
                for e in cutlass.range_constexpr(4):
                    o[mt, n, e] = out[e]

    @cute.jit
    def a_words(self, qkvs, words, mt: cutlass.Constexpr, h: cutlass.Constexpr, first, active):
        """``words[mt, s, 2 half + h]`` for every k step s of ``words``: the lane's MMA A
        words (bf16 pairs) of ``attention_group``'s row ``16 mt + 8 h + g``, from the fp32
        vector at ``qkvs`` word ``first`` (the lane's column ``2 c`` included); zero when the
        row is not ``active``."""
        for s in cutlass.range_constexpr(cute.size(words, mode=[1])):
            for half in cutlass.range_constexpr(2):
                col = first + 16 * s + 8 * half
                word = pack_bf16x2(qkvs[col], qkvs[col + 1])
                words[mt, s, 2 * half + h] = cutlass.select_(active, word, Int32(0))

    @cute.jit
    def rescale(self, m, lsum, o, mt: cutlass.Constexpr, hh: cutlass.Constexpr, m_new, part):
        """The online softmax of the lane's MMA row ``16 mt + 8 hh + g``: its running max
        becomes ``m_new``, its sum and P V scale by ``exp2(m - m_new)`` and the sum adds
        ``part`` (the new keys' probabilities against ``m_new``)."""
        alpha = cute.math.exp2(m[mt, hh] - m_new)
        m[mt, hh] = m_new
        lsum[mt, hh] = lsum[mt, hh] * alpha + part
        for n in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(2):
                o[mt, n, 2 * hh + e] = o[mt, n, 2 * hh + e] * alpha

    @property
    def mergers(self) -> int:
        """An ATTN kv head's merging items: its last ``mergers`` splits to finish."""
        return min(self.splits, self.group_rows)

    @cute.jit
    def merge_if_last(self, p, kvh, epoch, tid, warp, lane):
        """After an ATTN item of kv head ``kvh`` published its partials: count it, and if it is
        among the head's last ``mergers`` splits to finish (rank k of them), merge the query
        rows ``k, k + mergers ..`` (one warp each, ``merge_head_token``: it polls every split's
        partials) into ``o_in``: one CTA merging every row would serialize all the partials'
        loads. The count only grows (``splits`` per launch), so in launch epoch e the last
        split sees ``e * splits - 1``. A merging item polls the splits not yet counted, so
        each must run on another CTA: with more items than CTAs only the last one merges."""
        rows = self.group_rows
        items = self.kv_heads * self.splits
        mergers = cutlass.select_(cute.arch.grid_dim()[0] >= items, Int32(self.mergers), Int32(1))
        flag = cute.make_tensor(p.red.iterator, cute.make_layout((cute.size(p.red),)))
        self.csync.sync()
        if tid == 0:
            count = cute.recast_ptr(p.count.iterator, dtype=Int32)
            old = cute.arch.atomic_add(count + 2 * kvh, Int32(1), sem="acq_rel", scope="gpu")
            flag[LAST_FLAG] = Float32(old - (epoch * self.splits - mergers))
        self.csync.sync()
        rank = Int32(flag[LAST_FLAG])
        for rnd in cutlass.range_constexpr(-(-rows // COMPUTE_WARPS)):
            r = rank + mergers * (rnd * COMPUTE_WARPS + warp)
            if (rank >= 0) & (r < rows):
                j, t = self.head_token(r)
                self.merge_head_token(p, kvh * self.group + j, t, epoch, lane)
        self.csync.sync()  # (RED's flag is rewritten by the next item)

    @cute.jit
    def fetch_o_inputs(self, p, split, obuf, epoch, tid):
        """OPROJ split ``split``'s inputs: the merged gated outputs of its q heads for every
        token (``o_in``), into OIN rows ``obuf + t``."""
        nt, width = self.tokens, self.heads_per_split * HEAD_DIM
        groups = nt * width // 4
        rounds = -(-groups // COMPUTE_THREADS)
        vals = cute.make_rmem_tensor((4 * rounds,), Float32)
        segments, cells = [], []
        for r in cutlass.range_constexpr(rounds):
            # Threads past the groups refetch the last one.
            g = cutlass.min(r * COMPUTE_THREADS + tid, groups - 1)
            t = g // (width // 4)
            col = (g - t * (width // 4)) * 4
            index = (t * self.q_heads + split * self.heads_per_split) * HEAD_DIM + col
            segments.append((p.o_in, index, 4))
            cells.append((t, col))
        fetch(segments, vals, epoch, uniform=True)
        oin = self.buffer("OIN")
        for r in cutlass.range_constexpr(rounds):
            if r * COMPUTE_THREADS + tid < groups:
                t, col = cells[r]
                for e in cutlass.range_constexpr(4):
                    oin[obuf + t, col + e] = BFloat16(vals[4 * r + e])

    @cute.jit
    def merge_head_token(self, p, hq, t, epoch, lane):
        """``merge_head`` of token ``t``, published to ``o_in``: lane s < splits holds split
        s's running max and sum, the splits' weights come from warp reductions and one
        broadcast each."""
        splits = self.splits
        d = 4 * lane
        proj0 = t * self.proj_size
        mine = lane < splits
        item = self.attn_item(t, hq, cutlass.min(lane, Int32(splits - 1)))
        segments = [(p.attn_ml, 2 * item, 2)]
        for sp in cutlass.range_constexpr(splits):
            segments.append((p.attn_o, self.attn_item(t, hq, sp) * HEAD_DIM + d, 4))
        segments.append((p.proj, proj0 + self.proj_offsets[GATE] + hq * HEAD_DIM + d, 4))
        segments.append((p.proj, proj0 + self.proj_offsets[SINK] + hq, 1))
        vals = cute.make_rmem_tensor((2 + 4 * splits + 5,), Float32)
        fetch(segments, vals, epoch, uniform=True)
        sink = vals[2 + 4 * splits + 4] * LOG2E
        m = cutlass.max(group_max(cutlass.select_(mine, vals[0], -Float32.inf), 32), sink)
        weight = cute.math.exp2(cutlass.select_(mine, vals[0], -Float32.inf) - m)
        den = group_sum(cutlass.select_(mine, vals[1], Float32(0.0)) * weight, 32)
        den = den + cute.math.exp2(sink - m)
        num = [Float32(0.0)] * 4
        for sp in cutlass.range_constexpr(splits):
            w_sp = cute.arch.shuffle_sync(weight, offset=sp)
            for e in cutlass.range_constexpr(4):
                num[e] = num[e] + vals[2 + 4 * sp + e] * w_sp
        for e in cutlass.range_constexpr(4):
            gate = vals[2 + 4 * splits + e]
            sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate))
            out = bf16_round(num[e] / den)
            index = (t * self.q_heads + hq) * HEAD_DIM + d + e
            publish(p.o_in, index, bf16_round(out * sig), epoch)

    @cute.jit
    def acquire_pos_block(self, p, split, pos, epoch, tid):
        """OPROJ split ``split``'s row tile 0 item, before ``store_kv_tokens``: acquire the TMA
        reads of the KV block holding ``pos`` (its rows from ``pos`` are stored over), which
        the ATTN items that read it released before publishing their running max."""
        heads = self.heads_per_split // self.group
        if pos % BLOCK != 0:
            first, _n, per = self.split_span(pos)
            reader = (pos // BLOCK - first) // per
            if tid < heads:
                hq = (split * heads + tid) * self.group
                seen = cute.make_rmem_tensor((1,), Float32)
                fetch([(p.attn_ml, 2 * self.attn_item(0, hq, reader), 1)], seen, epoch)
                cute.arch.fence_acq_rel_gpu()

    @cute.jit
    def store_kv_tokens(self, p, split, pos, epoch, tid):
        """``store_kv_rows`` for every token of the block (rows ``pos + t``; rows at or past
        ``max_len`` are not stored)."""
        heads = self.heads_per_split // self.group
        max_len = p.k_cache.shape[0] // self.kv_heads
        per_token = heads * HEAD_DIM
        total = self.tokens * per_token
        for i in cutlass.range_constexpr(-(-total // COMPUTE_THREADS)):
            index = i * COMPUTE_THREADS + tid
            t = index // per_token
            rest = index - t * per_token
            if (index < total) & (pos + t < max_len):
                kvh = split * heads + rest // HEAD_DIM
                d = rest % HEAD_DIM
                column = t * 2 * self.kv_dim + kvh * HEAD_DIM + d
                vals = cute.make_rmem_tensor((2,), Float32)
                fetch([(p.kv_new, column, 1), (p.kv_new, column + self.kv_dim, 1)], vals, epoch)
                p.k_cache[kvh * max_len + pos + t, d] = BFloat16(vals[0])
                p.v_cache[kvh * max_len + pos + t, d] = BFloat16(vals[1])

    @cute.jit
    def block_conv(self, ring, taps, col, pos, t, active, y):
        """``post_conv`` over the block: the ``active`` warps write their token's bf16 ``y``
        to the ring row of position ``pos + t`` of column ``col``, then after a barrier each
        warp's bf16 ``(conv + y) / sqrt(kernel + 1)`` at that position, every earlier input
        read from the ring."""
        if active:
            ring[(pos + t) % RING, col] = BFloat16(y)
        self.csync.sync()
        kernel = self.conv_kernel
        q = pos + t
        conv = Float32(0.0)
        for j in cutlass.range_constexpr(kernel - 1):
            src = q - (kernel - 1) + j
            past = Float32(ring[(src + RING) % RING, col])
            conv = conv + taps[col, j] * cutlass.select_(src >= 0, past, Float32(0.0))
        conv = conv + taps[col, kernel - 1] * y
        return bf16_round((conv + y) * fp32_reciprocal(math.sqrt(kernel + 1)))

    @cute.jit
    def reduce_attention_tokens(self, p, a, pos, epoch, tid, lane):
        """``reduce_attention`` of row tile ``a`` for every token: warp t on token t's 32
        columns writes the bf16 o_proj output to the ring, then after a barrier convolves
        over the block."""
        nt = self.tokens
        t = tid // 32
        active = t < nt
        t = cutlass.min(t, Int32(nt - 1))
        col = a * ROWS + lane
        h = Float32(0.0)
        if p.boundary == 0:
            h = Float32(p.h[t * self.hidden + col])
        splits = self.o_splits
        vals = cute.make_rmem_tensor((splits,), Float32)
        segments = [(p.o_part, (t * splits + s) * self.hidden + col, 1) for s in range(splits)]
        fetch(segments, vals, epoch)
        acc = vals[0]
        for s in cutlass.range_constexpr(1, splits):
            acc = acc + vals[s]
        z = self.block_conv(p.ring_attn, p.attn_conv, col, pos, t, active, bf16_round(acc))
        if active:
            publish(
                p.resid, t * self.hidden + col, cutlass.select_(p.boundary != 0, z, h + z), epoch
            )

    # --------------------------------------------------------------------- MLP

    @cute.jit
    def mlp_row_tokens(self, p, w, meta, op, a, b, union, pos, epoch, tid, warp, lane, item):
        """The compute warps' part of MLP row ``(op, a, b)``; ``union``: the union's size once
        this CTA fetched it (-1 before). Returns the rings' consumers and ``union``."""
        if op == ROUTER:
            # Router tile a of token b, on the CTA's own row of it (``router_kind``); then the
            # other tokens' rows, which the CTA's later rows read.
            v, w = self.gemv(p, w, p.xs, [b] * MMA_N, 0, self.hidden_k_tiles, warp, lane, tid)
            if tid % MMA_N == 0:
                publish(p.logits, b * self.experts + a * ROWS + tid // MMA_N, v, epoch)
            self.copy_rows(p, epoch, tid, self.router_kind)
        elif op == REDUCE:
            self.reduce_tokens(p, a, pos, epoch, tid, lane)
        else:
            if union < 0:
                with self.trace_region("union", payload=item):
                    union = self.build_union(p, epoch, tid, warp, lane)
            if b < union:
                w, meta = self.expert_row(p, w, meta, op, a, b, epoch, tid, warp, lane)
        return w, meta, union

    @cute.jit
    def read_union(self, p, slot_experts, lane):
        """The load warp, once the compute warps built the union (``union_built``): its size
        and its slots' experts (lane l holds slot ``32 i + l``'s in ``slot_experts[i]``)."""
        self.union_built.sync()
        for i in cutlass.range_constexpr(self.slot_words // 32):
            slot_experts[i] = Int32(p.uni[UNION_EXPERTS + 32 * i + lane])
        return Int32(p.uni[0])

    @cute.jit
    def slot_expert(self, slot_experts, slot):
        """Slot ``slot``'s expert from ``read_union``'s registers (every lane must call it)."""
        lanes = [cute.arch.shuffle_sync(slot_experts[i], offset=slot % 32)
                 for i in range(self.slot_words // 32)]  # fmt: skip
        return pick(lanes, slot // 32)

    def chosen_words(self, p):
        """``route``'s chosen experts as Int32 words in RED (free outside ATTN items and
        reductions), lane l of token t's warp at word ``t * 32 + l``: bit i set if expert
        ``(experts // 32) l + i`` is chosen, its first slot from bit 8."""
        ptr = cute.recast_ptr(p.red.iterator + CHOSEN_WORD, dtype=Int32)
        return cute.make_tensor(ptr, cute.make_layout((self.tokens * 32,)))

    @cute.jit
    def route(self, p, lane, epoch, token):
        """``MlpHalf.route``'s routing weights of ``token`` into ROUTE row ``token`` (the
        experts go to ``chosen_words`` instead), each of its top-k rounds one warp maximum
        (``redux.sync`` on keys as ordered integers) and a ballot for the lowest lane holding
        it, whose own keys put the lowest expert first among equal keys (as ``route``'s ties
        to the lower index)."""
        n = self.experts // 32  # lane l holds experts n*l .. n*l + n - 1
        k = self.top_k
        logits = cute.make_rmem_tensor((n,), Float32)
        fetch([(p.logits, token * self.experts + n * lane, n)], logits, epoch, uniform=True, backoff=BACKOFF_NS)  # fmt: skip
        keys = cute.make_rmem_tensor((n,), Int32)
        scores = cute.make_rmem_tensor((n,), Float32)
        for i in cutlass.range_constexpr(n):
            scores[i] = Float32(1.0) / (Float32(1.0) + cute.math.exp(-logits[i]))
            keys[i] = ordered_key(scores[i] + p.router_bias[n * lane + i])
        taken = Int32(0)  # bit i: expert n*lane + i is in the top k
        gone = Int32(-(2**31))  # below every key
        for _ in cutlass.range_constexpr(k):
            best, index = keys[0], Int32(0)
            for i in cutlass.range_constexpr(1, n):
                better = keys[i] > best
                best = cutlass.select_(better, keys[i], best)
                index = cutlass.select_(better, Int32(i), index)
            top = cute.arch.warp_redux_sync(best, "max")
            holders = cute.arch.vote_ballot_sync(best == top)
            winner = cute.arch.popc((holders & -holders) - 1)  # the lowest lane holding it
            mine = lane == winner
            for i in cutlass.range_constexpr(n):
                hit = mine & (index == i)
                taken = taken | (Int32(hit) << i)
                keys[i] = cutlass.select_(hit, gone, keys[i])
        # A chosen expert's slot counts the chosen experts below it: an exclusive scan over
        # lanes of their counts, then the lane's own lower bits.
        count = cute.arch.popc(taken)
        below = count
        for step in cutlass.range_constexpr(5):
            up = cute.arch.shuffle_sync_up(below, offset=1 << step)
            below = below + cutlass.select_(lane >= (1 << step), up, Int32(0))
        slot = below - count
        self.chosen_words(p)[token * 32 + lane] = taken | (slot << 8)
        chosen = Float32(0.0)
        for i in cutlass.range_constexpr(n):
            if (taken >> i) & 1:
                p.route[token, k + slot] = scores[i]
                chosen = chosen + scores[i]
                slot = slot + 1
        total = group_sum(chosen, 32)
        cute.arch.sync_warp()
        if lane < k:
            weight = p.route[token, k + lane] / (total + Float32(1e-20)) * self.route_scale
            p.route[token, k + lane] = weight
        cute.arch.sync_warp()

    @cute.jit
    def build_union(self, p, epoch, tid, warp, lane):
        """Every CTA, before its first expert row: warp t routes token t (``route``: its
        weights into ROUTE row t, which the GATEUP rows read); then thread e counts expert e's tokens, the slots
        number the chosen experts in ascending order (a ballot scan), and the CTA builds the
        union in UNI (size, slot experts, counts, entries ``t * top_k + rank`` in token
        order; zeros elsewhere), hands it to the load warp (``union_built``) and returns its
        size. CTA 0 also publishes it (``InklingVisitMegakernel.union_size``). Every CTA
        computes the same union from the same logits."""
        nt, k = self.tokens, self.top_k
        uni = p.uni
        words = self.union_words
        for i in cutlass.range_constexpr(-(-words // COMPUTE_THREADS)):
            index = i * COMPUTE_THREADS + tid
            if index < words:
                uni[index] = Float32(0.0)
        if warp < nt:
            self.route(p, lane, epoch, warp)
        self.csync.sync()
        # Thread e < experts: expert e's word of each token (``chosen_words``).
        n = self.experts // 32
        bit = tid % n
        picks = self.chosen_words(p)
        mine = [picks[t * 32 + cutlass.min(tid // n, Int32(31))] for t in range(nt)]
        count = Int32(0)
        for t in cutlass.range_constexpr(nt):
            count = count + ((mine[t] >> bit) & 1)
        chosen = (count > 0) & (tid < self.experts)
        ballot = cute.arch.vote_ballot_sync(chosen)
        below = cute.arch.popc(ballot & cute.arch.lanemask_lt())
        if lane == 0:
            p.red[0, warp, 0] = Float32(cute.arch.popc(ballot))
        self.csync.sync()
        slot = below
        size = Int32(0)
        for i in cutlass.range_constexpr(COMPUTE_WARPS):
            chosen_i = Int32(p.red[0, i, 0])  # warp i's chosen experts
            slot = slot + cutlass.select_(i < warp, chosen_i, Int32(0))
            size = size + chosen_i
        if chosen:
            uni[UNION_EXPERTS + slot] = Float32(tid)
            uni[UNION_EXPERTS + self.slot_words + slot] = Float32(count)
            entries = UNION_EXPERTS + 2 * self.slot_words + slot * self.entry_stride
            i = Int32(0)
            for t in cutlass.range_constexpr(nt):
                if (mine[t] >> bit) & 1:  # rank: the token's chosen experts below e
                    rank = (mine[t] >> 8) + cute.arch.popc(mine[t] & ((1 << bit) - 1))
                    uni[entries + i] = Float32(t * k + rank)
                    i = i + 1
        if tid == 0:
            uni[0] = Float32(size)
        self.csync.sync()
        self.union_built.arrive()
        if cute.arch.block_idx()[0] == 0:
            for i in cutlass.range_constexpr(-(-words // COMPUTE_THREADS)):
                index = i * COMPUTE_THREADS + tid
                if index < words:
                    publish(p.union, index, uni[index], epoch)
        return size

    @cute.jit
    def expert_row(self, p, w, meta, op, a, slot, epoch, tid, warp, lane):
        """GATEUP tile ``a`` or DOWN row tiles ``a * DOWN_TILES ..`` (one activation fetch) of
        union slot ``slot`` over its tokens."""
        nt, k = self.tokens, self.top_k
        count = Int32(p.uni[UNION_EXPERTS + self.slot_words + slot])
        first = UNION_EXPERTS + 2 * self.slot_words + slot * self.entry_stride
        entries = [Int32(p.uni[first + i]) for i in range(nt)]
        row, t = tid // MMA_N, tid % MMA_N
        expert = Int32(p.uni[UNION_EXPERTS + slot])
        entry = pick(entries, t)  # thread tid's token entry (``t * top_k + rank``)
        if op == GATEUP:
            rows = [entries[min(i, nt - 1)] // k for i in range(MMA_N)]
            v, w, meta = self.expert_gemv(p, w, meta, 0, rows, warp, lane, tid)
            up = cute.arch.shuffle_sync_down(v, offset=MMA_N)  # tile rows alternate gate, up
            if (row % 2 == 0) & (t < count):
                token = entry // k
                g = self.scaled(p, 0, expert, p.amax[token], v)
                u = self.scaled(p, 0, expert, p.amax[token], up)
                weight = p.route[token, k + entry - token * k]  # (``build_union``'s routing)
                act = self.swiglu(bf16_round(g), bf16_round(u)) * weight
                col = (slot * nt + t) * self.width + a * (ROWS // 2) + row // 2
                publish(p.act, col, bf16_round(act), epoch)
        else:
            # The last DOWN row's reads of the activation rows ended before its ``finish``. Each
            # DOWN item quantizes its slot's rows itself, in registers as they are fetched: one
            # item quantizing them once for the others made theirs wait for it (1.3-1.4x
            # slower block steps). The rows' amax stay in registers (thread tid's row t's).
            with self.trace_region("down.act"):
                amax = self.fetch_act_tokens(p, slot, count, epoch, tid, warp, lane)
                row_amax = pick(amax, t)
                self.csync.sync()
            for i in cutlass.range_constexpr(DOWN_TILES):
                v, w, meta = self.expert_gemv(p, w, meta, 1, self.rows_of(nt), warp, lane, tid)
                if t < count:
                    y = self.scaled(p, 1, expert, row_amax, v)
                    index = entry * self.hidden + (a * DOWN_TILES + i) * ROWS + row
                    publish(p.down, index, bf16_round(y), epoch)
        return w, meta

    @cute.jit
    def expert_gemv(self, p, w, meta, kind: cutlass.Constexpr, rows, warp, lane, tid):
        """``(value, w, meta)``: thread ``tid``'s value (``examples.megakernel.multi_gemv.finish``)
        of an expert row tile of ``kind`` (GATEUP 0, DOWN 1) against the xs ``rows``, in the
        experts' format."""
        if cutlass.const_expr(self.quantized):
            with meta.consume() as h:
                acc, w = gemv_quant(
                    self.experts_format, w, h, p.xs, rows, self.expert_stages[kind],
                    self.expert_subtiles[kind], self.stage_subtiles, warp, lane,
                )  # fmt: skip
        else:
            acc, w = gemv_bf16(w, p.xs, rows, 0, self.expert_subtiles[kind], warp, lane)
        v = finish(acc, self.dbuf(p), self.csync, warp, lane, tid, self.experts_format)
        return v, w, meta

    @cute.jit
    def fetch_act_tokens(self, p, slot, count, epoch, tid, warp, lane):
        """Slot ``slot``'s activations of its ``count`` tokens, in one poll, each row quantized
        in registers with its own amax (``act_quant.quantize_chunk``; NVFP4 after the CTA
        maxima of the threads' ``|x|`` per row) into xs rows ``tokens + i`` as block values,
        zeros in the padding the last down stage reads. Returns the rows' amax (every thread;
        zeros for MXFP8); the caller's barrier orders the rows before the GEMVs."""
        nt = self.tokens
        chunks = self.width // 8
        per = (self.down_k_tiles * 256) // 8  # (whole warps)
        total = nt * per
        rounds = -(-total // COMPUTE_THREADS)
        vals = cute.make_rmem_tensor((8 * rounds,), Float32)
        # Per round: the thread's chunk cc of row i, and whether it holds activations (rows i <
        # count <= tokens); the others fetch chunk 0 and read zeros.
        segments, cells = [], []
        for r in cutlass.range_constexpr(rounds):
            c = r * COMPUTE_THREADS + tid
            i = c // per
            cc = c - i * per
            real = (i < count) & (cc < chunks)
            cells.append((i, cc, real))
            index = (slot * nt + cutlass.select_(real, i, Int32(0))) * self.width
            segments.append((p.act, index + cutlass.select_(real, cc, Int32(0)) * 8, 8))
        fetch(segments, vals, epoch)
        partial = [Float32(0.0)] * nt  # the thread's maximum |x| per row
        for r in cutlass.range_constexpr(rounds):
            i, _cc, real = cells[r]
            for e in cutlass.range_constexpr(8):
                vals[8 * r + e] = cutlass.select_(real, vals[8 * r + e], Float32(0.0))
            if cutlass.const_expr(self.activation_format == NVFP4):
                m = block_max([vals[8 * r + e] for e in range(8)], 8)
                for j in cutlass.range_constexpr(nt):
                    partial[j] = f32_op(
                        "max", partial[j], cutlass.select_(i == j, m, Float32(0.0))
                    )
        amax = [Float32(0.0)] * nt
        if cutlass.const_expr(self.activation_format == NVFP4):
            amax = cta_max(p.red, self.csync, partial, AMAX_RED, warp, lane)
        for r in cutlass.range_constexpr(rounds):
            if r * COMPUTE_THREADS + warp * 32 < count * per:  # (the warp's first chunk)
                i, cc, _real = cells[r]
                x = [vals[8 * r + e] for e in range(8)]
                values = quantize_chunk(self.activation_format, x, pick(amax, i))
                if i < count:
                    words = cute.make_rmem_tensor((4,), Int32)
                    for j in cutlass.range_constexpr(4):
                        words[j] = values[j]
                    cute.autovec_copy(
                        cute.recast_tensor(words, BFloat16),
                        cute.local_tile(p.xs[nt + i, None], (8,), (cc,)),
                    )
        return amax

    @cute.jit
    def reduce_tokens(self, p, a, pos, epoch, tid, lane):
        """REDUCE row tile ``a``: warp t on token t's 32 columns: the fp32 sum of its top-k
        bf16 down outputs (the routing weights are in the activations) in ascending expert
        order (``reduce_experts``), written to the ring, then after a barrier the post-MLP
        convolution over the block plus the residual."""
        nt, k = self.tokens, self.top_k
        t = tid // 32
        active = t < nt
        t = cutlass.min(t, Int32(nt - 1))
        col = a * ROWS + lane
        vals = cute.make_rmem_tensor((k + 1,), Float32)
        segments = [(p.down, (t * k + r) * self.hidden + col, 1) for r in range(k)]
        segments.append((p.resid, t * self.hidden + col, 1))
        fetch(segments, vals, epoch)
        acc = Float32(0.0)
        for r in cutlass.range_constexpr(k):
            acc = acc + vals[r]
        z = self.block_conv(p.ring_mlp, p.mlp_conv, col, pos, t, active, bf16_round(acc))
        if active:
            p.out[t * self.hidden + col] = BFloat16(vals[k] + z)

    # ------------------------------------------------------------ message gates

    @cute.jit
    def msgdown(self, p, w, a, split, epoch, warp, lane):
        nt = self.tokens
        tid = warp * 32 + lane
        per = self.hidden // TILE_K // GATE_SPLITS
        rows = self.rows_of(self.gate_input(a // GATE_TILES) * nt)
        v, w = self.gemv(p, w, p.xs, rows, split * per * TILE_K, per, warp, lane, tid)
        row, t = tid // MMA_N, tid % MMA_N
        if t < nt:
            index = (split * GATES * GATE_TILES + a) * ROWS + row
            publish(p.gate.h_part, t * GATE_SPLITS * GATES * GATE_RANK + index, v, epoch)
        return w

    @cute.jit
    def stream_value_token(self, p, s, i, t):
        query = self.gate_input(s)
        value = Float32(0.0)
        if s == MLP_GATE:
            resid, _tag = unpack_tagged(load_words(p.resid, t * self.hidden + i, 1)[0])
            value = self.mixed_state(p, query, p.num_blocks + p.boundary, resid, i, token=t)
        else:
            h = Float32(p.h[t * self.hidden + i])
            value = self.mixed_state(p, query, p.num_blocks, h, i, token=t)
        return value

    @cute.jit
    def gatemix(self, p, w, tile, s, epoch, warp, lane):
        """``common``'s GATEMIX for every token: the tokens' h (bf16 sums of the MSGDOWN
        partials and the message half) go to ATT once, then each stage's F and G rows serve
        every token."""
        nt = self.tokens
        g = p.gate
        j = lane % 8
        row = ROWS_PER_WARP * warp + lane // 8
        tid = warp * 32 + lane
        hsm = cute.make_tensor(p.att.iterator, cute.make_layout((nt * GATE_RANK,)))
        xm = cute.make_rmem_tensor((MIX_STAGES, nt, 2), Float32)
        for stage in cutlass.range_constexpr(MIX_STAGES):
            i = (tile * MIX_STAGES + stage) * ROWS + row
            for t in cutlass.range_constexpr(nt):
                xm[stage, t, 0] = self.stream_value_token(p, s, i, t)
                xm[stage, t, 1] = g.messages[s, t * self.hidden + i]
        per = -(-nt * GATE_RANK // COMPUTE_THREADS)
        stride = GATES * GATE_RANK
        hv = cute.make_rmem_tensor((per * GATE_SPLITS,), Float32)
        segments, cells = [], []
        for v in cutlass.range_constexpr(per):
            # Threads past the entries refetch the last one.
            index = cutlass.min(v * COMPUTE_THREADS + tid, nt * GATE_RANK - 1)
            t = index // GATE_RANK
            c = index - t * GATE_RANK
            cells.append((index, t, c))
            base = t * GATE_SPLITS * stride + s * GATE_RANK + c
            segments += [(g.h_part, base + k * stride, 1) for k in range(GATE_SPLITS)]
        fetch(segments, hv, epoch, uniform=True)
        self.csync.sync()  # ATT's last readers (a GEMV's ``finish``) are done
        for v in cutlass.range_constexpr(per):
            if v * COMPUTE_THREADS + tid < nt * GATE_RANK:
                index, t, c = cells[v]
                acc = hv[GATE_SPLITS * v]
                for k in cutlass.range_constexpr(1, GATE_SPLITS):
                    acc = acc + hv[GATE_SPLITS * v + k]
                hsm[index] = bf16_round(acc + g.mhalf[(s * GATE_RANK + c) * nt + t])
        self.csync.sync()
        for stage in cutlass.range_constexpr(MIX_STAGES):
            i = (tile * MIX_STAGES + stage) * ROWS + row
            forget, gain = [], []
            with w.consume() as h:
                rows = cute.recast_tensor(h.view("W"), Float32)  # (ROWS, 2 * GATE_RANK)
                fr = [rows[row, 8 * j + e] for e in range(8)]
                gr = [rows[row, GATE_RANK + 8 * j + e] for e in range(8)]
                for t in cutlass.range_constexpr(nt):
                    f = Float32(0.0)
                    gn = Float32(0.0)
                    for e in cutlass.range_constexpr(8):
                        hval = hsm[t * GATE_RANK + 8 * j + e]
                        f = f + fr[e] * hval
                        gn = gn + gr[e] * hval
                    forget.append(f)
                    gain.append(gn)
            for t in cutlass.range_constexpr(nt):
                f = group_sum(forget[t], 8)
                gn = group_sum(gain[t], 8)
                if j == 0:
                    x, m = xm[stage, t, 0], xm[stage, t, 1]
                    out = (Float32(1.0) - cute.math.tanh(f)) * x + cute.math.tanh(gn) * m
                    publish(g.gated, (t * GATES + s) * self.hidden + i, out, epoch)
        # The next item rewrites ATT.
        self.csync.sync()
        return w

    @cute.jit
    def gather_token(self, p, t, mlp: cutlass.Constexpr, epoch, tid, warp, lane):
        """``common.gather`` of token ``t``: its gated streams (the MLP's, or the attention's
        q, k and v) RMS-normalized into xs rows ``s * tokens + t``."""
        h, nt = self.hidden, self.tokens
        per = h // COMPUTE_THREADS
        first, n = (MLP_GATE, 1) if mlp else (0, self.streams)
        norms = p.post_norm if mlp else p.attn_norms
        weights = cute.make_rmem_tensor((n, per), BFloat16)
        for s in cutlass.range_constexpr(n):
            for r in cutlass.range_constexpr(per):
                weights[s, r] = norms[s, r * COMPUTE_THREADS + tid]
        vals = cute.make_rmem_tensor((n * per,), Float32)
        segments = [(p.gate.gated, (t * GATES + first + s) * h + r * COMPUTE_THREADS + tid, 1)
                    for s in range(n) for r in range(per)]  # fmt: skip
        fetch(segments, vals, epoch, uniform=True, backoff=BACKOFF_NS)
        squares = []
        for s in cutlass.range_constexpr(n):
            ss = Float32(0.0)
            for r in cutlass.range_constexpr(per):
                ss = ss + vals[s * per + r] * vals[s * per + r]
            squares.append(ss)
        squares = cta_sum(p.red, self.csync, squares, 2, warp, lane)
        for s in cutlass.range_constexpr(n):
            rinv = cute.math.rsqrt(squares[s] * fp32_reciprocal(h) + Float32(self.rms_eps))
            for r in cutlass.range_constexpr(per):
                k = r * COMPUTE_THREADS + tid
                p.xs[s * nt + t, k] = BFloat16(Float32(weights[s, r]) * (vals[s * per + r] * rinv))
        self.csync.sync()


class InklingMultiQuantVisitKernel(InklingMultiVisitKernel, InklingQuantVisitKernel):
    """``InklingMultiVisitKernel`` with quantized experts (roles of
    ``InklingQuantVisitKernel``)."""


def multi_visit_kernel_class(experts_format: str) -> type[InklingMultiVisitKernel]:
    """The multi-token visit kernel class of experts in ``experts_format``."""
    return InklingMultiVisitKernel if experts_format == BF16 else InklingMultiQuantVisitKernel
