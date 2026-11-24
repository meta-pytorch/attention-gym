"""The attention half of an Inkling 2 Mini decode visit: the device code of the visit
launch's attention rows (``AttentionHalf``, a mixin of ``visit_kernel.InklingVisitKernel``).

Batch one, one token. The rows implement ``ref.attention_block`` with ``start=None`` (device
position, the whole cache behind its mask)::

    every CTA   prologue: block residual over [blocks..., h] for each of the layer's queries
                (none at layer 0; there ``embed`` makes h the token's normalized embedding
                row), RMSNorm -> bf16 attention inputs (SMEM)
    [messages]  MSGDOWN, GATEMIX of the query, key and value gates (``common``), then each
                CTA gathers the three gated streams and RMS-normalizes them into the inputs
    QKV         32-row tiles of q, k, v, rel, (full layers: dynamic k/v taps), gate, sink
                x hidden                                    -> bf16 projections (tagged)
    ATTN        q head x KV split: q scale/clip, k/v short  -> unnormalized O, (m, l) (tagged)
                conv, k clip, v RMS norm (redundant per
                item), relative-bias scores, flash partial
                over the split's 64-key blocks; the newest
                split adds the new token; one item per kv
                head writes the ring rows, publishes the
                new K/V row
    OPROJ       row tile x quarter of the q heads: merge    -> fp32 partials (tagged)
                the heads' splits with the sink, sigmoid
                gate, then 32 o_proj rows x 768 columns;
                the last quarter's item reduces its 32
                columns: split-order sum, post-attention
                conv (ring write), residual               -> fp32 residual (tagged: the
                (row tile 0's items also store the new       MLP half's input)
                K/V rows once their heads' items finished)

Sliding layers read the keys in ``(pos - window, pos]`` with static k/v convolutions; full
layers read ``[0, pos]`` with dynamic taps and log scaling of the query and relative state.
Full layers and layers with messages project q, k and v from three input streams. Scores are
rounded to bf16 as the oracle's (``bf16(bf16(bf16(q.k) * scale) + bf16(rel . rel_proj[:,
distance]))``); the softmax runs in the log2 domain over flash partials, so probabilities are
not rounded to bf16 before the value sum as in the oracle: the merge rounds the normalized
output once.

The load warp streams 32 x 256 weight stages and 64 x 128 KV blocks through the weight ring.
The cache row at the position is stored only after every item reading the block that holds it
has finished; those items mask the row's stale contents. That store is a generic write by one
CTA after other CTAs' TMA (async-proxy) reads of the same block, so the partials double as a
release/acquire edge:

    ATTN item, newest split   KV stages consumed (mbarrier waits: its TMA reads completed),
    (the only split reading   barrier, then the thread publishing (m, l): fence.proxy.async
    the block of ``pos``)     .global, fence.acq_rel.gpu, publish
    OPROJ row tile 0 item     its merge warps poll every split's (m, l), fence.acq_rel.gpu,
                              barrier, then every thread stores its cache rows

The ATTN side's proxy fence orders the completed async reads before the generic release; the
OPROJ side's fence makes the polls an acquire before the barrier that precedes the stores.
"""

import math

import cutlass
from cutlass import BFloat16, Float32, Int32, cute

from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    MAX_BLOCKS,
    RING,
    conv_inputs,
    post_conv,
)
from attn_gym.warpdrive.examples.megakernel.attention import head_vec, split_blocks, stage_row
from attn_gym.warpdrive.examples.megakernel.constants import (
    BLOCK,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    HEAD_DIM,
    LOG2E,
    ROWS,
    TILE_K,
)
from attn_gym.warpdrive.examples.megakernel.gemv import gemv_tile, owns_row
from attn_gym.warpdrive.examples.megakernel.regmath import (
    bf16_round,
    cta_sum,
    fp32,
    fp32_reciprocal,
    group_sum,
)
from attn_gym.warpdrive.examples.megakernel.ring import load_tiles, tma_atom
from attn_gym.warpdrive.examples.megakernel.tagged import fetch, publish

QKV, ATTN, OPROJ = range(3)
# Projection matrices of the QKV rows (their b word), in the order the table deals them: what
# attention reads first, the output gate and sink (read by OPROJ) last.
Q, K, V, REL, KDYN, VDYN, GATE, SINK = range(8)
O_SPLITS = 4  # o_proj split-K: a quarter of the q heads each
QKVS_FLOATS = 512  # an attention item's head vectors (see ``AttentionHalf.qkvs_words``)


@cute.jit
def clip_factor(ss, threshold: cutlass.Constexpr):
    """``ref.l2_clip``'s scale for a vector with squared norm ``ss``: PyTorch evaluates
    ``threshold / (norm + 1e-6)`` as ``reciprocal(norm + 1e-6) * threshold``."""
    scale = Float32(1.0) / (cute.math.sqrt(ss) + Float32(1e-6)) * Float32(threshold)
    return cutlass.min(scale, Float32(1.0))


class AttentionHalf:
    """The attention rows of a visit launch (see the module docstring). Its params carry, besides
    ``InklingKernel``'s: ``tma_attn``/``gmem_attn`` (the projection matrices in ``matrices``
    order, then wo), ``tma_kv``/``gmem_kv`` (k and v cache), ``sW``/``sKV`` stage views,
    ``qkvs`` (QKVS_FLOATS fp32), ``att`` (``(COMPUTE_WARPS + 1, HEAD_DIM + 2)`` fp32), the
    tagged ``proj``, ``attn_o``, ``attn_ml``, ``o_part``, ``kv_new`` and ``resid`` and the
    argument tensors of ``visit_kernel.VisitParams``."""

    def init_attention(
        self,
        q_heads: int,
        kv_heads: int,
        d_rel: int,
        rel_width: int,
        extent: int,
        qk_clip: float,
        rel_clip: float,
        splits: int,
        full: bool,
        window: int,
        log_alpha: float,
        log_floor: int,
    ):
        """Model sizes and constants of the attention rows (``self.hidden``, ``conv_kernel``
        and ``messages`` set before). ``full``: a full-attention layer (three block-residual
        queries, dynamic k/v taps, log scaling ``log_alpha``/``log_floor``), else a sliding
        one (``window`` keys). ``extent``: distances with a relative bias; ``rel_width``:
        columns of ``rel_proj``. ``splits``: KV splits per q head (static; pick for the window
        or the contexts decoded)."""
        hidden, conv_kernel = self.hidden, self.conv_kernel
        if hidden % ROWS:
            raise ValueError(f"hidden size must be a multiple of {ROWS}")
        if q_heads % kv_heads or q_heads % O_SPLITS:
            raise ValueError(f"q heads must be a multiple of kv heads and of {O_SPLITS}")
        if q_heads // O_SPLITS > COMPUTE_WARPS or (q_heads // O_SPLITS * HEAD_DIM) % TILE_K:
            raise ValueError("an o_proj split merges one head per warp in whole 256-column tiles")
        if q_heads // O_SPLITS * HEAD_DIM > hidden:
            raise ValueError(
                f"an o_proj split's {q_heads // O_SPLITS} heads x {HEAD_DIM} merge into one "
                f"hidden-size ({hidden}) input row: need q_heads <= {O_SPLITS * hidden // HEAD_DIM}"
            )
        if (q_heads // O_SPLITS) % (q_heads // kv_heads):
            raise ValueError("an o_proj split must cover whole groups of q heads")
        if d_rel % 8 or d_rel > 32 or not 0 < extent <= min(rel_width, hidden):
            raise ValueError("need d_rel a multiple of 8 up to 32 and extent <= rel_width, hidden")
        if full and conv_kernel % 2:
            raise ValueError("full layers fetch dynamic taps in pairs: need an even kernel")
        if not 2 <= conv_kernel <= RING or not 1 <= splits <= 32:
            raise ValueError("need a kernel-2..RING convolution and 1 to 32 splits")
        if full != (window == 0) or full != (log_floor > 0):
            raise ValueError("full layers take log scaling and no window; sliding the reverse")
        self.q_heads, self.kv_heads = q_heads, kv_heads
        self.group = q_heads // kv_heads
        self.d_rel, self.rel_width, self.extent = d_rel, rel_width, extent
        self.qk_clip, self.rel_clip = qk_clip, rel_clip
        self.splits, self.full, self.window = splits, full, window
        self.log_alpha, self.log_floor = log_alpha, log_floor
        self.queries = 3 if full else 1  # block-residual queries of the prologue
        self.streams = 3 if full or self.messages else 1  # q, k, v input streams
        self.kv_dim = kv_heads * HEAD_DIM
        self.heads_per_split = q_heads // O_SPLITS
        self.oproj_k_tiles = self.heads_per_split * HEAD_DIM // TILE_K
        self.row_tiles = hidden // ROWS
        if self.qkvs_words > QKVS_FLOATS:
            raise ValueError("an attention item's head vectors exceed QKVS_FLOATS")
        rows = {Q: q_heads * HEAD_DIM, K: self.kv_dim, V: self.kv_dim, REL: q_heads * d_rel,
                GATE: q_heads * HEAD_DIM, SINK: q_heads}  # fmt: skip
        if full:
            rows.update({KDYN: kv_heads * conv_kernel, VDYN: kv_heads * conv_kernel})
        self.matrices = tuple(sorted(rows))
        self.proj_rows = rows
        self.proj_tiles = {m: -(-r // ROWS) for m, r in rows.items()}
        offsets, offset = {}, 0
        for m in self.matrices:
            offsets[m] = offset
            offset += self.proj_tiles[m] * ROWS
        self.proj_offsets, self.proj_size = offsets, offset

    @property
    def qkvs_words(self) -> int:
        """An attention item's head vectors in QKVS: q, k, v (HEAD_DIM each), the relative
        state (d_rel), then in full layers the dynamic k and v taps (kernel each)."""
        return 3 * HEAD_DIM + self.d_rel + (2 * self.conv_kernel if self.full else 0)

    def attention_scratch_shapes(self) -> dict[str, int]:
        """Tagged int64 words of each attention intermediate; ``resid`` is the half's output."""
        return {
            "proj": self.proj_size,
            "attn_o": self.q_heads * self.splits * HEAD_DIM,
            "attn_ml": self.q_heads * self.splits * 2,
            "o_part": O_SPLITS * self.hidden,
            "kv_new": 2 * self.kv_dim,
            "resid": self.hidden,
        }

    def attention_rows(self) -> list[list[int]]:
        """Table rows ``[op, a, b]`` of the attention in dependency order: QKV tiles
        (attention's projections first, the gate and sink last), ATTN items split-major, then
        OPROJ rows split-major (the last split's rows reduce their tiles)."""
        rows = [[QKV, t, m] for m in self.matrices for t in range(self.proj_tiles[m])]
        rows += [[ATTN, hq, s] for s in range(self.splits) for hq in range(self.q_heads)]
        return rows + [[OPROJ, t, s] for s in range(O_SPLITS) for t in range(self.row_tiles)]

    # ------------------------------------------------------------------ rows

    @cute.jit
    def load_attention(self, p, w, op, a, b, pos):
        """The load warp's stages of attention row ``(op, a, b)``."""
        count = len(self.matrices)
        if op == QKV:
            for i in cutlass.range_constexpr(count):
                if b == self.matrices[i]:
                    w = load_tiles(
                        w, p.tma_attn[i], p.gmem_attn[i], a, 0, self.hidden_k_tiles, "W"
                    )
        elif op == ATTN:
            max_len = p.k_cache.shape[0] // self.kv_heads
            blk0, blk1, _newest = self.split_blocks(pos, b)
            row0 = (a // self.group) * (max_len // BLOCK)
            for blk in cutlass.range(blk0, blk1, unroll=1):
                w = load_tiles(w, p.tma_kv[0], p.gmem_kv[0], row0 + blk, 0, 1, "KV")
                w = load_tiles(w, p.tma_kv[1], p.gmem_kv[1], row0 + blk, 0, 1, "KV")
        else:
            k_tiles = self.oproj_k_tiles
            w = load_tiles(w, p.tma_attn[count], p.gmem_attn[count], a, b * k_tiles, k_tiles, "W")
        return w

    @cute.jit
    def attention_prologue(self, p, tid, warp, lane):
        """Every CTA: the attention inputs ``xs[0 .. queries)`` from the layer's block
        residuals (layer 0: the token's normalized embedding)."""
        # Layer 0 reads no h: CTA 0 may be storing the embedding there.
        x = self.load_states(p, p.num_blocks, p.h, tid, p.embed_input == 0)
        if p.embed_input != 0:
            self.embed_token(p, x, MAX_BLOCKS, True, tid, warp, lane)
        self.normalize_states(
            p, x, p.num_blocks, p.attn_query, p.attn_norms, self.queries, p.queries, tid, warp,
            lane,
        )  # fmt: skip

    @cute.jit
    def attention_row(self, p, w, op, a, b, item, obuf, pos, epoch, tid, warp, lane):
        """The compute warps' part of attention row ``(op, a, b)``. OPROJ rows alternate
        their input buffer ``obuf`` (xs row ``streams`` or ``streams + 1``): every warp passed
        the last merge's barrier after finishing the row before it, the last reader of this
        buffer. Returns the weight ring's consumer and ``obuf``."""
        if op == QKV:
            v, row, w = gemv_tile(w, p.xs, self.stream_of(b), self.hidden_k_tiles, warp, lane)
            if owns_row(lane):
                index = self.proj_offset(b) + a * ROWS + row
                publish(p.proj, index, bf16_round(v), epoch)
        elif op == ATTN:
            blk0, blk1, newest = self.split_blocks(pos, b)
            if (blk0 == blk1) & (b != newest):
                self.publish_empty_partial(p, a, b, epoch, tid)
            else:
                w = self.attention(p, w, a, b, pos, epoch, tid, warp, lane, item)
        else:
            obuf = 2 * self.streams + 1 - obuf
            with self.trace_region("merge", payload=item):
                if warp < self.heads_per_split:
                    hq = b * self.heads_per_split + warp
                    self.merge_head(p, hq, warp, obuf, epoch, lane)
                    if a == 0:  # acquire the partials before ``store_kv_rows`` (module)
                        cute.arch.fence_acq_rel_gpu()
                self.csync.sync()
            if a == 0:
                self.store_kv_rows(p, b, pos, epoch, tid)
            v, row, w = gemv_tile(w, p.xs, obuf, self.oproj_k_tiles, warp, lane)
            if owns_row(lane):
                publish(p.o_part, b * self.hidden + a * ROWS + row, v, epoch)
            if (b == O_SPLITS - 1) & (warp == 0):
                with self.trace_region("reduce", payload=item):
                    self.reduce_attention(p, a * ROWS + lane, pos, epoch)
        return w, obuf

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def embed_token(self, p, x, slot: cutlass.Constexpr, store: cutlass.Constexpr, tid, warp,
                    lane):  # fmt: skip
        """State ``x[slot]`` (``load_states`` left it zero) becomes the token's
        embedding row through the weighted RMSNorm (``ref.hidden_states``' first ``h``); with
        ``store``, CTA 0 also stores it to ``h`` for the launches after this one (CTAs of this
        launch recompute it rather than read it)."""
        per = self.hidden // COMPUTE_THREADS
        row = Int32(p.token[0])
        ss = Float32(0.0)
        for r in cutlass.range_constexpr(per):
            v = Float32(p.embed[row, r * COMPUTE_THREADS + tid])
            x[slot, r] = v
            ss = ss + v * v
        ss = cta_sum(p.red, self.csync, [ss], 0, warp, lane)[0]
        rinv = cute.math.rsqrt(ss * fp32_reciprocal(self.hidden) + Float32(self.rms_eps))
        first = cute.arch.block_idx()[0] == 0
        for r in cutlass.range_constexpr(per):
            k = r * COMPUTE_THREADS + tid
            v = BFloat16(Float32(p.embed_norm[k]) * (x[slot, r] * rinv))
            x[slot, r] = Float32(v)
            if cutlass.const_expr(store):
                if first:
                    p.h[k] = v
        # The prologue's reductions reuse RED slot 0.
        self.csync.sync()

    @cute.jit
    def stream_of(self, matrix):
        """The input stream (xs row) of a projection: with three streams, k (and its dynamic
        taps) come from stream 1, v from stream 2, the rest from stream 0 (the query input)."""
        stream = Int32(0)
        if cutlass.const_expr(self.streams == 3):
            stream = cutlass.select_((matrix == K) | (matrix == KDYN), Int32(1), stream)
            stream = cutlass.select_((matrix == V) | (matrix == VDYN), Int32(2), stream)
        return stream

    @cute.jit
    def proj_offset(self, matrix):
        offset = Int32(0)
        for m in self.matrices:
            offset = cutlass.select_(matrix == m, Int32(self.proj_offsets[m]), offset)
        return offset

    @cute.jit
    def split_blocks(self, pos, split):
        """``(blk0, blk1, newest)`` of ``split`` over the cached keys this layer reads
        (``attention.split_blocks``; sliding layers: the window)."""
        return split_blocks(pos, split, self.splits, None if self.full else self.window)

    @cute.jit
    def log_scale(self, pos):
        """``ref.log_scale`` in fp32 (full layers): ``1 + alpha * log(max((pos + 1) / floor,
        1))``."""
        ratio = Float32(pos + 1) * fp32_reciprocal(self.log_floor)
        return Float32(1.0) + Float32(self.log_alpha) * cute.math.log(
            cutlass.max(ratio, Float32(1.0))
        )

    @cute.jit
    def qkvs_column(self, hq, kvh, word):
        """The proj index of word ``word`` of an item's QKVS (``qkvs_words``)."""
        d, r, c = HEAD_DIM, self.d_rel, self.conv_kernel
        off = self.proj_offsets
        column = off[Q] + hq * d + word
        column = cutlass.select_(word >= d, off[K] + kvh * d + word - d, column)
        column = cutlass.select_(word >= 2 * d, off[V] + kvh * d + word - 2 * d, column)
        column = cutlass.select_(word >= 3 * d, off[REL] + hq * r + word - 3 * d, column)
        if cutlass.const_expr(self.full):
            base = 3 * d + r
            column = cutlass.select_(word >= base, off[KDYN] + kvh * c + word - base, column)
            base = base + c
            column = cutlass.select_(word >= base, off[VDYN] + kvh * c + word - base, column)
        return column

    @cute.jit
    def distances(self, pos, blk0, blk1, newest):
        """``(first, last)``: the distances ``pos - key`` with a relative bias that the split's
        keys (and the new token, at distance 0, in the newest split) can have."""
        first = cutlass.max(pos - blk1 * BLOCK + 1, Int32(0))
        first = cutlass.select_(newest, Int32(0), first)
        last = cutlass.min(pos - blk0 * BLOCK, Int32(self.extent - 1))
        return first, last

    @cute.jit
    def prepare_heads(self, p, hq, kvh, pos, writer, first, last, epoch, tid, warp, lane):
        """Fetch the item's projections into QKVS, then in place (one warp each): the q head's
        scale, L2 clip (and log scaling), the kv head's k and v convolutions (writing their
        ring rows when ``writer``), k's L2 clip and v's RMS norm, and the relative state's
        scale and clip (and log scaling). Then every thread builds the relative bias of
        distances ``[first, last]`` in BIAS (``score``). Every value is rounded to bf16 as the
        oracle's.

        The weights, ring rows and ``rel_proj`` rows each thread needs load before the poll and
        are used after it, so their latency overlaps the wait for QKV (this launch writes ring
        row ``pos`` only, after reading the earlier ones)."""
        kernel, r = self.conv_kernel, self.d_rel
        rounds = -(-self.extent // COMPUTE_THREADS)
        d0 = 4 * lane  # warps 0-2: this lane's 4 dims of the head
        scales = cute.make_rmem_tensor((4,), BFloat16)
        taps = cute.make_rmem_tensor((4, kernel), p.k_taps.element_type)
        past = cute.make_rmem_tensor((4, kernel - 1), BFloat16)
        rel_proj = cute.make_rmem_tensor((rounds, r), BFloat16)
        if warp == 0:
            for e in cutlass.range_constexpr(4):
                scales[e] = p.q_scale[hq, d0 + e]
        elif warp == 1:
            self.load_conv(p.ring_k, p.k_taps, kvh, pos, lane, taps, past)
        elif warp == 2:
            self.load_conv(p.ring_v, p.v_taps, kvh, pos, lane, taps, past)
        elif warp == 3:
            if lane < r:
                scales[0] = p.rel_scale[hq, lane]
        for i in cutlass.range_constexpr(rounds):
            d = first + i * COMPUTE_THREADS + tid
            if d <= last:
                cute.autovec_copy(p.rel_proj[d, None], rel_proj[i, None])
        words = self.qkvs_words
        # Thread t fetches words 2t, 2t + 1; threads past them refetch the last pair.
        word = cutlass.min(2 * tid, words - 2)
        raw = cute.make_rmem_tensor((2,), Float32)
        fetch([(p.proj, self.qkvs_column(hq, kvh, word), 2)], raw, epoch)
        if 2 * tid < words:
            p.qkvs[word] = raw[0]
            p.qkvs[word + 1] = raw[1]
        self.csync.sync()
        ls = Float32(1.0)
        if cutlass.const_expr(self.full):
            ls = self.log_scale(pos)
        if warp == 0:
            x = cute.make_rmem_tensor((4,), Float32)
            ss = Float32(0.0)
            for e in cutlass.range_constexpr(4):
                x[e] = p.qkvs[d0 + e] * Float32(scales[e])
                ss = ss + x[e] * x[e]
            factor = clip_factor(group_sum(ss, 32), self.qk_clip)
            for e in cutlass.range_constexpr(4):
                v = bf16_round(x[e] * factor)
                if cutlass.const_expr(self.full):
                    v = bf16_round(v * ls)
                p.qkvs[d0 + e] = v
        elif warp == 1:
            self.conv_head(p, p.ring_k, HEAD_DIM, 3 * HEAD_DIM + r, kvh, pos, writer, lane,
                           taps, past, True)  # fmt: skip
        elif warp == 2:
            self.conv_head(p, p.ring_v, 2 * HEAD_DIM, 3 * HEAD_DIM + r + kernel, kvh, pos,
                           writer, lane, taps, past, False)  # fmt: skip
        elif warp == 3:
            x = Float32(0.0)
            if lane < r:
                x = p.qkvs[3 * HEAD_DIM + lane] * Float32(scales[0])
            factor = clip_factor(group_sum(x * x, 32), self.rel_clip)
            v = bf16_round(x * factor)
            if cutlass.const_expr(self.full):
                v = bf16_round(v * ls)
            if lane < r:
                p.qkvs[3 * HEAD_DIM + lane] = v
        self.csync.sync()
        rel = cute.make_rmem_tensor((r,), Float32)
        for k in cutlass.range_constexpr(r):
            rel[k] = p.qkvs[3 * HEAD_DIM + k]
        bias = self.bias_table(p)
        for i in cutlass.range_constexpr(rounds):
            d = first + i * COMPUTE_THREADS + tid
            if d <= last:
                acc = Float32(0.0)
                for k in cutlass.range_constexpr(r):
                    acc = acc + rel[k] * Float32(rel_proj[i, k])
                bias[d] = BFloat16(acc)
        self.csync.sync()

    def bias_table(self, p):
        """BIAS: the bf16 relative bias by distance (``extent`` entries). It reuses the first
        o_proj input row of X: every CTA runs its ATTN rows before its OPROJ rows."""
        return p.xs[self.streams, None]

    @cute.jit
    def load_conv(self, ring, taps, kvh, pos, lane, tap, past):
        """One lane's 4 channels of convolution taps (``conv_head``'s layout) and ring rows
        ``pos - 1 - t`` (``past[:, t]``; rows before position 0 are read but unused)."""
        kernel = self.conv_kernel
        for e in cutlass.range_constexpr(4):
            c = kvh * HEAD_DIM + 4 * lane + e
            for t in cutlass.range_constexpr(kernel):
                if cutlass.const_expr(self.full):
                    tap[e, t] = taps[t, c]
                else:
                    tap[e, t] = taps[c, t]
            for t in cutlass.range_constexpr(kernel - 1):
                past[e, t] = ring[(pos - 1 - t + RING) % RING, c]

    @cute.jit
    def conv_head(self, p, ring, base, dyn, kvh, pos, writer, lane, tap, past,
                  is_key: cutlass.Constexpr):  # fmt: skip
        """One warp: the k (``is_key``: then L2 clip) or v (then unweighted RMS norm) head at
        QKVS ``base`` through its short convolution, lane l on dims 4l .. 4l + 3, from
        ``load_conv``'s taps and ring rows. Sliding layers: ``ref.short_conv`` with static
        (channel, kernel) taps, the last on the current token. Full layers:
        ``ref.dynamic_conv``, tap t (t back) is the static (kernel, channel) weight plus the
        head's dynamic tap at QKVS ``dyn + t``."""
        taps_n = self.conv_kernel
        z = cute.make_rmem_tensor((4,), Float32)
        ss = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            y = p.qkvs[base + 4 * lane + e]
            hist = [y]  # hist[t]: the input t positions back, zero before position 0
            for t in cutlass.range_constexpr(1, taps_n):
                hist.append(cutlass.select_(pos - t >= 0, Float32(past[e, t - 1]), Float32(0.0)))
            conv = Float32(0.0)
            if cutlass.const_expr(self.full):
                for t in cutlass.range_constexpr(taps_n):
                    conv = conv + hist[t] * (Float32(tap[e, t]) + p.qkvs[dyn + t])
                out = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(2 * taps_n + 1)))
            else:
                for t in cutlass.range_constexpr(taps_n):
                    conv = conv + tap[e, t] * hist[taps_n - 1 - t]
                out = bf16_round((conv + y) * fp32_reciprocal(math.sqrt(taps_n + 1)))
            if writer:
                ring[pos % RING, kvh * HEAD_DIM + 4 * lane + e] = BFloat16(y)
            z[e] = out
            ss = ss + out * out
        ss = group_sum(ss, 32)
        if cutlass.const_expr(is_key):
            factor = clip_factor(ss, self.qk_clip)
        else:
            factor = cute.math.rsqrt(ss * fp32_reciprocal(HEAD_DIM) + Float32(self.rms_eps))
        for e in cutlass.range_constexpr(4):
            p.qkvs[base + 4 * lane + e] = bf16_round(z[e] * factor)

    @cute.jit
    def score(self, p, dot, distance):
        """A key's softmax logit in log2 units from its 8 lanes' partial q.k ``dot``: the bf16
        score plus the relative bias at ``distance`` (BIAS; zero past the extent), rounded to
        bf16 as ``ref.attention``."""
        index = cutlass.min(cutlass.max(distance, Int32(0)), Int32(self.extent - 1))
        bias = Float32(self.bias_table(p)[index])
        bias = cutlass.select_(distance < self.extent, bias, Float32(0.0))
        s = bf16_round(bf16_round(group_sum(dot, 8)) * fp32(HEAD_DIM**-0.5))
        return bf16_round(s + bias) * LOG2E

    @cute.jit
    def attention(self, p, w, hq, split, pos, epoch, tid, warp, lane, item):
        """Flash-decoding partial of q head ``hq`` over KV split ``split``.

        Warp w scores positions 8w + 4i + g (i = 0, 1) of each 64-position block: lane group
        g = lane // 8 holds a position, lane j = lane % 8 its dims 8j..8j+7 and 64+8j..64+8j+7
        and relative-state entries 2j, 2j + 1. Each warp keeps its own online-softmax state; the
        warps, plus the new token in the newest split, merge through ATT at the end."""
        g = lane // 8
        j = lane % 8
        d0 = 8 * j
        kvh = hq // self.group
        blk0, blk1, newest = self.split_blocks(pos, split)
        is_newest = split == newest
        writer = is_newest & (hq % self.group == 0)  # one item per kv head
        first, last = self.distances(pos, blk0, blk1, is_newest)
        with self.trace_region("attn.prepare", payload=item):
            self.prepare_heads(p, hq, kvh, pos, writer, first, last, epoch, tid, warp, lane)
        q = head_vec(p.qkvs, 0, d0)
        lo = Int32(0)
        if cutlass.const_expr(not self.full):
            lo = pos - (self.window - 1)
        m = Float32(-1e30)  # running max (log2 units), warp-uniform
        lsum = Float32(0.0)  # this lane group's sum of p
        o = cute.make_rmem_tensor((16,), Float32)  # this lane group's sum of p * v
        o.fill(0.0)
        with self.trace_region("attn.blocks", payload=item):
            for blk in cutlass.range(blk0, blk1, unroll=1):
                sc = cute.make_rmem_tensor((2,), Float32)
                with w.consume() as h:
                    sK = h.view("KV")
                    for i in cutlass.range_constexpr(2):
                        pt = 8 * warp + 4 * i + g
                        kp = blk * BLOCK + pt
                        kv = stage_row(sK, pt, d0)
                        dot = Float32(0.0)
                        for e in cutlass.range_constexpr(16):
                            dot = dot + q[e] * kv[e]
                        s = self.score(p, dot, pos - kp)
                        sc[i] = cutlass.select_((kp < pos) & (kp >= lo), s, -Float32.inf)
                bm = cutlass.max(sc[0], sc[1])
                bm = cutlass.max(bm, cute.arch.shuffle_sync_bfly(bm, offset=8))
                bm = cutlass.max(bm, cute.arch.shuffle_sync_bfly(bm, offset=16))
                m_new = cutlass.max(m, bm)
                alpha = cute.math.exp2(m - m_new)
                m = m_new
                pr0 = cute.math.exp2(sc[0] - m_new)
                pr1 = cute.math.exp2(sc[1] - m_new)
                lsum = lsum * alpha + pr0 + pr1
                for e in cutlass.range_constexpr(16):
                    o[e] = o[e] * alpha
                with w.consume() as h:
                    sV = h.view("KV")
                    for i in cutlass.range_constexpr(2):
                        pt = 8 * warp + 4 * i + g
                        kp = blk * BLOCK + pt
                        if (kp < pos) & (kp >= lo):  # masked: p is 0, but a stale v may be NaN
                            vv = stage_row(sV, pt, d0)
                            pr = pr0 if i == 0 else pr1
                            for e in cutlass.range_constexpr(16):
                                o[e] = o[e] + pr * vv[e]
        # The new K/V row goes to the cache only after every item reading the block that holds
        # its position finished (``store_kv_rows``): the writer publishes it.
        if writer & (tid < HEAD_DIM):
            publish(p.kv_new, kvh * HEAD_DIM + tid, p.qkvs[HEAD_DIM + tid], epoch)
            publish(
                p.kv_new, self.kv_dim + kvh * HEAD_DIM + tid, p.qkvs[2 * HEAD_DIM + tid], epoch
            )
        # Merge the four lane groups (they share m), then publish this warp's state to ATT.
        lsum = lsum + cute.arch.shuffle_sync_bfly(lsum, offset=8)
        lsum = lsum + cute.arch.shuffle_sync_bfly(lsum, offset=16)
        for e in cutlass.range_constexpr(16):
            o[e] = o[e] + cute.arch.shuffle_sync_bfly(o[e], offset=8)
            o[e] = o[e] + cute.arch.shuffle_sync_bfly(o[e], offset=16)
        att = p.att
        if g == 0:
            for e in cutlass.range_constexpr(8):
                att[warp, d0 + e] = o[e]
                att[warp, d0 + 64 + e] = o[8 + e]
            if j == 0:
                att[warp, HEAD_DIM] = m
                att[warp, HEAD_DIM + 1] = lsum
        if is_newest:
            if warp == 0:  # entry COMPUTE_WARPS is the new token: m = its score, l = 1, O = v
                k_new = head_vec(p.qkvs, HEAD_DIM, d0)
                v_new = head_vec(p.qkvs, 2 * HEAD_DIM, d0)
                dot = Float32(0.0)
                for e in cutlass.range_constexpr(16):
                    dot = dot + q[e] * k_new[e]
                s = self.score(p, dot, Int32(0))
                if g == 0:
                    for e in cutlass.range_constexpr(8):
                        att[COMPUTE_WARPS, d0 + e] = v_new[e]
                        att[COMPUTE_WARPS, d0 + 64 + e] = v_new[8 + e]
                    if j == 0:
                        att[COMPUTE_WARPS, HEAD_DIM] = s
                        att[COMPUTE_WARPS, HEAD_DIM + 1] = Float32(1.0)
        self.csync.sync()
        if tid < HEAD_DIM:
            self.merge_partials(p, hq, split, is_newest, tid, epoch)
        # The next attention row rewrites ATT and QKVS.
        self.csync.sync()
        return w

    @cute.jit
    def merge_partials(self, p, hq, split, newest, d, epoch):
        """Merge the warps' online-softmax states in ATT (plus the new token's in the newest
        split) into this split's partial O (dim ``d``), m and l, and publish them."""
        att = p.att
        mx = Float32(-1e30)
        for e in cutlass.range_constexpr(COMPUTE_WARPS):
            mx = cutlass.max(mx, att[e, HEAD_DIM])
        if newest:
            mx = cutlass.max(mx, att[COMPUTE_WARPS, HEAD_DIM])
        num = Float32(0.0)
        den = Float32(0.0)
        for e in cutlass.range_constexpr(COMPUTE_WARPS):
            scale = cute.math.exp2(att[e, HEAD_DIM] - mx)
            num = num + att[e, d] * scale
            den = den + att[e, HEAD_DIM + 1] * scale
        if newest:
            scale = cute.math.exp2(att[COMPUTE_WARPS, HEAD_DIM] - mx)
            num = num + att[COMPUTE_WARPS, d] * scale
            den = den + att[COMPUTE_WARPS, HEAD_DIM + 1] * scale
        item = hq * self.splits + split
        publish(p.attn_o, item * HEAD_DIM + d, num, epoch)
        if d == 0:
            if newest:  # release this item's TMA reads of the block of pos (module docstring)
                cute.arch.fence_proxy("async.global")
                cute.arch.fence_acq_rel_gpu()
            publish(p.attn_ml, 2 * item, mx, epoch)
            publish(p.attn_ml, 2 * item + 1, den, epoch)

    @cute.jit
    def publish_empty_partial(self, p, hq, split, epoch, tid):
        """A split with no KV blocks at this position: m = -1e30, l = 0 (zero weight in the
        merge) and O = 0, published at once without waiting for QKV."""
        item = hq * self.splits + split
        if tid < HEAD_DIM:
            publish(p.attn_o, item * HEAD_DIM + tid, Float32(0.0), epoch)
        if tid == 0:
            publish(p.attn_ml, 2 * item, Float32(-1e30), epoch)
            publish(p.attn_ml, 2 * item + 1, Float32(0.0), epoch)

    @cute.jit
    def store_kv_rows(self, p, split, pos, epoch, tid):
        """Cache rows ``pos`` of the kv heads whose q heads o_proj split ``split`` covers, from
        their writers' published K/V, by row tile 0's item of the split after its merge: every
        ATTN item of those heads has then published its partial, after its last KV load, so no
        TMA read of the block holding ``pos`` is in flight (the release/acquire edge of the
        module docstring orders those reads before these stores). ``0 <= pos < max_len`` is
        the caller's contract; a position at or past ``max_len`` stores nothing rather than
        another head's (or the next visit's) row. A negative one is not guarded: it writes
        before the visit's rows here and in the convolution rings (``pos % RING`` truncates)."""
        heads = self.heads_per_split // self.group
        max_len = p.k_cache.shape[0] // self.kv_heads
        for i in cutlass.range_constexpr(-(-heads * HEAD_DIM // COMPUTE_THREADS)):
            index = i * COMPUTE_THREADS + tid
            if (index < heads * HEAD_DIM) & (pos < max_len):
                kvh = split * heads + index // HEAD_DIM
                d = index % HEAD_DIM
                vals = cute.make_rmem_tensor((2,), Float32)
                column = kvh * HEAD_DIM + d
                fetch([(p.kv_new, column, 1), (p.kv_new, self.kv_dim + column, 1)], vals, epoch)
                p.k_cache[kvh * max_len + pos, d] = BFloat16(vals[0])
                p.v_cache[kvh * max_len + pos, d] = BFloat16(vals[1])

    @cute.jit
    def merge_head(self, p, hq, slot, obuf, epoch, lane):
        """One warp: q head ``hq``'s o_proj input at head slot ``slot`` of ``xs[obuf]``, lane l
        on dims 4l .. 4l + 3. The splits' partials merge with the sink logit in the softmax
        denominator; the bf16 output times the sigmoid gate is rounded to bf16, as
        ``ref.attention``. Lane l also fetches split l % splits' (m, l), shared by shuffles,
        so the words are polled warp-uniformly."""
        splits = self.splits
        d = 4 * lane
        segments = [(p.attn_ml, 2 * (hq * splits + lane % splits), 2)]
        for sp in cutlass.range_constexpr(splits):
            segments.append((p.attn_o, (hq * splits + sp) * HEAD_DIM + d, 4))
        segments.append((p.proj, self.proj_offsets[GATE] + hq * HEAD_DIM + d, 4))
        segments.append((p.proj, self.proj_offsets[SINK] + hq, 1))
        vals = cute.make_rmem_tensor((2 + 4 * splits + 5,), Float32)
        fetch(segments, vals, epoch, uniform=True)
        m = vals[2 + 4 * splits + 4] * LOG2E  # the sink
        ms = cute.make_rmem_tensor((splits,), Float32)
        for sp in cutlass.range_constexpr(splits):
            ms[sp] = cute.arch.shuffle_sync(vals[0], offset=sp)
            m = cutlass.max(m, ms[sp])
        den = cute.math.exp2(vals[2 + 4 * splits + 4] * LOG2E - m)
        for sp in cutlass.range_constexpr(splits):
            ms[sp] = cute.math.exp2(ms[sp] - m)  # split sp's scale
            den = den + cute.arch.shuffle_sync(vals[1], offset=sp) * ms[sp]
        for e in cutlass.range_constexpr(4):
            num = Float32(0.0)
            for sp in cutlass.range_constexpr(splits):
                num = num + vals[2 + 4 * sp + e] * ms[sp]
            gate = vals[2 + 4 * splits + e]
            sig = Float32(1.0) / (Float32(1.0) + cute.math.exp(-gate))
            out = bf16_round(num / den)
            p.xs[obuf, slot * HEAD_DIM + d + e] = BFloat16(out * sig)

    @cute.jit
    def reduce_attention(self, p, col, pos, epoch):
        """Residual column ``col``, by one lane of warp 0: the bf16 o_proj output (the fp32 sum
        of the splits' partials in split order), written to the convolution ring at the
        position, then the post-attention convolution ``(conv + y) / sqrt(kernel + 1)`` (bf16)
        plus ``h`` (unless the layer starts a block), published to ``resid``."""
        # The weights, ring rows and input load before the poll (their latency overlaps it).
        weights, past = conv_inputs(p.ring_attn, p.attn_conv, col, pos, self.conv_kernel)
        h = Float32(0.0)
        if p.boundary == 0:  # (layer 0's h may still be in CTA 0's embedding store)
            h = Float32(p.h[col])
        vals = cute.make_rmem_tensor((O_SPLITS,), Float32)
        fetch([(p.o_part, s * self.hidden + col, 1) for s in range(O_SPLITS)], vals, epoch)
        acc = vals[0]
        for s in cutlass.range_constexpr(1, O_SPLITS):
            acc = acc + vals[s]
        z = post_conv(p.ring_attn, weights, past, col, pos, bf16_round(acc), self.conv_kernel)
        publish(p.resid, col, cutlass.select_(p.boundary != 0, z, h + z), epoch)

    # ------------------------------------------------------------- TMA atoms

    @cute.jit
    def attention_atoms(self, by_matrix: dict, wo, k_cache, v_cache):
        """TMA ``(atom, tensor)`` pairs: 32 x 256 weight tiles of the projection matrices in
        ``matrices`` order (sliding layers have no dynamic taps) and of wo, then 64 x 128 KV
        blocks of the caches."""
        sources = [by_matrix[m] for m in self.matrices] + [wo]
        weights = [tma_atom(src, (ROWS, TILE_K)) for src in sources]
        caches = [tma_atom(src, (BLOCK, HEAD_DIM)) for src in (k_cache, v_cache)]
        return weights, caches
