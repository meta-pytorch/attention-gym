"""Device code shared by the Inkling 2 Mini megakernel launches (``visit_kernel`` with its
halves ``attn_half`` and ``mlp_half``, ``msg_kernel``, ``head_kernel``): sizes, tagged-word
polling, CTA reductions, TMA plumbing, the block-residual prologue and the message-gate work
items, and the launches whose items are row tiles of one matrix (``MatrixRowsKernel``).

Message gates (core passes after the first, ``ref.message_gate``) run inside the visit launch
as two kinds of work items before the rows of the half whose input they gate (gates 0-2: the
attention's q, k and v streams; gate 3, ``MLP_GATE``: the MLP's input)::

    MSGDOWN  6 per gate  32 rows of the gate's down projection x a third of the x half of its
                         input (bf16 rms(x), from the half's prologue) -> fp32 partials (tagged)
    GATEMIX  36 per gate 64 elements: h = bf16(partials + message half), then
                         (1 - tanh(F h)) x + tanh(G h) m            -> fp32 stream (tagged)

The message half of each down projection (its columns on bf16 rms(m)) and the messages ``m``
themselves depend only on the previous pass, so ``msg_kernel`` computes them once per pass for
every core layer. The CTAs then gather the gated streams and RMS-normalize them into the GEMV
inputs (``gather``) before their first own row of the half.

Sized so that a half's message items (126 in attention, 42 in the MLP) fit one per CTA:
every GATEMIX item then waits for the last MSGDOWN item, not for another GATEMIX item of its CTA.
In a globaltimer trace of a sliding attention launch with messages (GB300, Inkling 2 Mini's
shapes, context 1100; 2026-11-21) the gather ended ~6.5 us after the prologue (whole-K
MSGDOWN tiles and 32-row GATEMIX items, two on some CTAs: ~9 us).
"""

import math
from dataclasses import dataclass

import cutlass
from cutlass import BFloat16, Float32, Int32, cute

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
from attn_gym.warpdrive.cache import make_fake_compact_tensor
from attn_gym.warpdrive.examples.megakernel.constants import (
    BOX,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    DBUF_FLOATS,
    RED_VALUES,
    ROWS,
    ROWS_PER_WARP,
    STAGE_BYTES,
    TILE_K,
    X_PAD,
)
from attn_gym.warpdrive.examples.megakernel.gemv import gemv_tile, owns_row
from attn_gym.warpdrive.examples.megakernel.regmath import (
    bf16_round,
    cta_sum,
    fp32,
    fp32_reciprocal,
    group_sum,
    red_layout,
)
from attn_gym.warpdrive.examples.megakernel.ring import (
    box_layout,
    load_boxes,
    load_tiles,
    ring_stages,
    swizzled_atom,
    tma_atom,
    tma_partition,
    weight_stages,
)
from attn_gym.warpdrive.examples.megakernel.table import InstructionTable
from attn_gym.warpdrive.examples.megakernel.tagged import (
    fetch,
    publish,
)

WORDS = 4  # table words: op, a, b, table row
MAX_BLOCKS = 4  # completed blocks of a block residual; the partial block is the fifth state
STATES = MAX_BLOCKS + 1
RING = 16  # ref.CONV_RING: positions of the convolution input rings
# cta_sum slots: state norms, query dots, the inputs' norms. The prologue's reductions use
# distinct slots; a later reuse follows the prologue's closing barrier or another cta_sum.
REDUCTIONS = 3
MSGDOWN, GATEMIX = 6, 7  # message-gate ops, after the visit's own ops 0-5
GATES = 4  # message gates per core layer: query, key, value, MLP
MLP_GATE = 3
GATE_RANK = 64  # rows of a gate's down projection: 8 lanes x 8 values in GATEMIX
GATE_TILES = GATE_RANK // ROWS  # MSGDOWN row tiles per gate
GATE_SPLITS = 3  # MSGDOWN split-K: k tiles of the x half per item = hidden / 256 / GATE_SPLITS
MIX_STAGES = 2  # GATEMIX: 32-row stages of [F | G] per item
# MIX: the prologue's state scale factors (STATES), then each query's state weights (STATES).
MIX_FLOATS = STATES * 4


def add_offset(index, base):
    """``index + base``, leaving ``index`` itself when ``base`` is the Python int 0 (a token
    offset of the one-token kernels, which must keep their IR)."""
    return index if isinstance(base, int) and base == 0 else index + base


@cute.jit
def conv_inputs(ring, taps, col, pos, kernel: cutlass.Constexpr):
    """Column ``col``'s inputs of a post-attention or post-MLP short convolution (``post_conv``):
    its fp32 ``(channel, kernel)`` taps and the ``kernel - 1`` ring rows before position
    ``pos``, oldest first (rows before position 0 are read but unused)."""
    weights = cute.make_rmem_tensor((kernel,), Float32)
    past = cute.make_rmem_tensor((kernel - 1,), BFloat16)
    for j in cutlass.range_constexpr(kernel):
        weights[j] = taps[col, j]
    for j in cutlass.range_constexpr(kernel - 1):
        past[j] = ring[(pos - (kernel - 1) + j + RING) % RING, col]
    return weights, past


@cute.jit
def post_conv(ring, weights, past, col, pos, y, kernel: cutlass.Constexpr):
    """``ref.short_conv`` of column ``col`` at position ``pos`` on the bf16 value ``y``, from
    ``conv_inputs``: writes ``y`` to the ring row of ``pos``, then returns the bf16
    ``(conv + y) / sqrt(kernel + 1)``, the last tap on ``y`` and positions before 0 zero."""
    ring[pos % RING, col] = BFloat16(y)
    conv = Float32(0.0)
    for j in cutlass.range_constexpr(kernel - 1):
        src = pos - (kernel - 1) + j  # >= -(kernel - 1): positions before 0 read zero
        conv = conv + weights[j] * cutlass.select_(src >= 0, Float32(past[j]), Float32(0.0))
    conv = conv + weights[kernel - 1] * y
    return bf16_round((conv + y) * fp32_reciprocal(math.sqrt(kernel + 1)))


def fake_gate_args(hidden: int, tokens: int = 1) -> tuple:
    """Fake compile arguments of a core layer's message gates (``gate_atoms``, ``GateParams``):
    the down projections ``(GATES * GATE_RANK, 2 * hidden)`` bf16, the fp32 ``(GATES, hidden,
    2 * GATE_RANK)`` ``[F | G]`` rows viewed as bf16 ``(GATES * hidden, 4 * GATE_RANK)``, the
    message halves ``(GATES * GATE_RANK * tokens,)`` (row-major, token minor) and the messages
    ``(GATES, tokens * hidden)`` (token-major within a gate)."""
    return (
        make_fake_compact_tensor(BFloat16, (GATES * GATE_RANK, 2 * hidden)),
        make_fake_compact_tensor(BFloat16, (GATES * hidden, 4 * GATE_RANK)),
        make_fake_compact_tensor(Float32, (GATES * GATE_RANK * tokens,), assumed_align=4),
        make_fake_compact_tensor(Float32, (GATES, tokens * hidden), assumed_align=4),
    )


def gate_atoms(down, fg, hidden: int, x_atom) -> list:
    """TMA ``(atom, tensor)`` pairs of the down projections' x halves (their first ``hidden``
    columns) as ``x_atom(src)`` loads them (the launch's weight atoms: MSGDOWN streams them as
    GEMV weights) and of 32 x 256 bf16 tiles of the ``[F | G]`` rows (one tile: 32 rows)."""
    x_half = cute.local_tile(down, (GATES * GATE_RANK, hidden), (0, 0))
    return [x_atom(x_half), tma_atom(fg, (ROWS, TILE_K))]


@dataclass(frozen=True)
class GateParams:
    """A launch's message-gate operands for one core layer (``InklingKernel.messages``)."""

    tma: tuple  # TMA atoms: the down projections' x halves, the [F | G] rows
    gmem: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    h_part: cute.Tensor  # tagged int64 (GATE_SPLITS, GATES, GATE_RANK) x-half partial dots
    gated: cute.Tensor  # tagged int64 (GATES * hidden,) gated streams
    mhalf: cute.Tensor  # (GATES * GATE_RANK,) fp32 message-half dots (msg_kernel)
    messages: cute.Tensor  # (GATES, hidden) fp32 messages (msg_kernel)


class InklingKernel(WarpSpecialized):
    """Device helpers of the Inkling launches. Subclasses set ``hidden``, ``rms_eps``,
    ``stages`` (``ring_stages``) and ``csync`` (the compute warps' barrier), and define
    ``scratch_shapes`` (tagged words by buffer, for ``examples.megakernel.launch.Launch``)
    when they publish any; launches with message gates call ``init_messages`` and implement
    ``stream_value``. Their params carry ``xs`` (GEMV inputs), ``red`` (``cta_sum``'s
    ``REDUCTIONS`` slots), ``mix`` (``(MIX_FLOATS,)`` fp32), ``blocks`` and ``gate``
    (``GateParams`` or None)."""

    messages = False
    tokens = 1  # consecutive tokens of one sequence per launch
    x_pad = 0  # bf16 elements after each row of X (multi-token launches skew rows' banks)

    def init_messages(self, messages: bool, gate_query: tuple[int, ...]):
        """All GATES message gates when ``messages``; ``gate_query[s]`` is the prologue query
        whose block residual gate s's x is (the xs row and MIX weights its prologue wrote)."""
        if messages and (
            (self.hidden // TILE_K) % GATE_SPLITS or self.hidden % (ROWS * MIX_STAGES)
        ):
            raise ValueError(
                f"message gates need hidden / {TILE_K} divisible by {GATE_SPLITS} and hidden "
                f"divisible by {ROWS * MIX_STAGES}"
            )
        if len(gate_query) != GATES:
            raise ValueError(f"need a prologue query for each of the {GATES} gates")
        self.messages, self.gate_query = messages, gate_query

    def core_layouts(self, x_rows: int) -> dict:
        """SMEM layouts of the weight stages ``W``, the GEMV inputs ``X`` (``x_rows`` rows) and
        the reductions ``RED``."""
        return {
            "W": weight_stages(self.stages),
            "X": cute.make_layout((x_rows, self.hidden), stride=(self.hidden + self.x_pad, 1)),
            "RED": red_layout(REDUCTIONS),
        }

    @property
    def scratch_shapes(self) -> dict[str, int]:
        """Tagged int64 words of each per-launch intermediate."""
        return {}

    def gate_scratch_shapes(self) -> dict[str, int]:
        """Tagged words of the message items: MSGDOWN partials (split, gate, row) and the gated
        streams (gate, element); none without messages."""
        if not self.messages:
            return {}
        return {"h_part": GATE_SPLITS * GATES * GATE_RANK, "gated": GATES * self.hidden}

    def gate_rows(self, gates: range) -> list[list[int]]:
        """Table rows ``[op, a, b]`` of the message items of ``gates`` in dependency order:
        MSGDOWN (tile, split; tile ``a`` of gate ``a // GATE_TILES``), then GATEMIX (tile,
        gate); none without messages."""
        if not self.messages:
            return []
        tiles = range(gates.start * GATE_TILES, gates.stop * GATE_TILES)
        rows = [[MSGDOWN, t, k] for t in tiles for k in range(GATE_SPLITS)]
        mix_tiles = self.hidden // (ROWS * MIX_STAGES)
        return rows + [[GATEMIX, t, s] for s in gates for t in range(mix_tiles)]

    def stream_value(self, p, s, i):
        """Element ``i`` of gate ``s``'s input stream: its prologue's block residual of query
        ``gate_input(s)``, recomputed in that prologue's operation order (``mixed_state``)."""
        raise NotImplementedError

    @cute.jit
    def load_stages(self, w, atom, tiles, row, k0, count: cutlass.Constexpr,
                    buffer: cutlass.Constexpr):  # fmt: skip
        """The load warp's ``count`` GEMV weight stages (``load_tiles``)."""
        return load_tiles(w, atom, tiles, row, k0, count, buffer)

    @cute.jit
    def load_states(self, p, num_blocks, partial, tid, load_partial=True, token=0):
        """``(STATES, hidden // COMPUTE_THREADS)`` fp32 registers: the first ``num_blocks``
        completed blocks (zeros past them), then ``partial`` (a ``(hidden,)`` tensor; zeros if
        None or ``load_partial`` is false: the caller fills it). Thread t holds elements
        ``r * 256 + t``. ``token``: of a multi-token launch, whose states are ``(tokens *
        hidden,)``, token-major."""
        per = self.hidden // COMPUTE_THREADS
        base = token * self.hidden
        x = cute.make_rmem_tensor((STATES, per), Float32)
        x.fill(0.0)
        for n in cutlass.range_constexpr(MAX_BLOCKS):
            if n < num_blocks:
                for r in cutlass.range_constexpr(per):
                    x[n, r] = Float32(p.blocks[n][add_offset(r * COMPUTE_THREADS + tid, base)])
        if cutlass.const_expr(partial is not None):
            if load_partial:
                for r in cutlass.range_constexpr(per):
                    x[MAX_BLOCKS, r] = Float32(
                        partial[add_offset(r * COMPUTE_THREADS + tid, base)]
                    )
        return x

    @cute.jit
    def normalize_states(
        self,
        p,
        x,
        num_blocks,
        query,
        norms,
        nq: cutlass.Constexpr,
        queries,
        tid,
        warp,
        lane,
        token=0,
    ):
        """``xs[q]`` for each of the ``nq`` queries: the block residual ``y_q`` of fp32 ``query[q]``
        over ``load_states``'s ``x`` (``ref.block_residual`` over ``num_blocks`` blocks;
        ``queries == 0``: the partial block itself), then ``bf16(norms[q] * rms_norm(y_q))``
        (the half's input norm) or, in a launch with message gates, ``bf16(rms_norm(y_q))``
        (the x half of the gates' down inputs); the state scales and weights go to MIX for
        ``mixed_state``. ``token``: of a multi-token launch (``tokens`` > 1): query q's input
        goes to xs row ``q * tokens + token`` and the MIX values to the token's MIX_FLOATS."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        weighted = not self.messages
        mix0 = token * MIX_FLOATS
        # Every load first, so their latencies overlap: the queries and norm weights.
        qv = cute.make_rmem_tensor((nq, per), Float32)
        weights = cute.make_rmem_tensor((nq, per), BFloat16)
        for q in cutlass.range_constexpr(nq):
            for r in cutlass.range_constexpr(per):
                qv[q, r] = query[q, r * COMPUTE_THREADS + tid]
                if cutlass.const_expr(weighted):
                    weights[q, r] = norms[q, r * COMPUTE_THREADS + tid]
        inv_h = fp32_reciprocal(h)
        y = cute.make_rmem_tensor((nq, per), Float32)
        if queries == 0:
            for r in cutlass.range_constexpr(per):
                y[0, r] = x[MAX_BLOCKS, r]
        else:
            squares = []
            for n in cutlass.range_constexpr(STATES):
                ss = Float32(0.0)
                for r in cutlass.range_constexpr(per):
                    ss = ss + x[n, r] * x[n, r]
                squares.append(ss)
            squares = cta_sum(p.red, self.csync, squares, 0, warp, lane)
            # Unweighted RMSNorm of every state (eps 1e-6, as ref.block_residual), then each
            # query's scores against them; empty block slots are masked out of the softmax.
            rinvs = []
            for n in cutlass.range_constexpr(STATES):
                rinvs.append(cute.math.rsqrt(squares[n] * inv_h + Float32(1e-6)))
                for r in cutlass.range_constexpr(per):
                    x[n, r] = x[n, r] * rinvs[n]
            dots = []
            for q in cutlass.range_constexpr(nq):
                for n in cutlass.range_constexpr(STATES):
                    dot = Float32(0.0)
                    for r in cutlass.range_constexpr(per):
                        dot = dot + qv[q, r] * x[n, r]
                    dots.append(dot)
            dots = cta_sum(p.red, self.csync, dots, 1, warp, lane)
            scale = fp32(h**-0.5)
            for q in cutlass.range_constexpr(nq):
                scores = []
                top = Float32(-Float32.inf)
                for n in cutlass.range_constexpr(STATES):
                    s = dots[q * STATES + n] * scale
                    if cutlass.const_expr(n < MAX_BLOCKS):
                        s = cutlass.select_(n < num_blocks, s, -Float32.inf)
                    scores.append(s)
                    top = cutlass.max(top, s)
                total = Float32(0.0)
                exps = []
                for n in cutlass.range_constexpr(STATES):
                    exps.append(cute.math.exp(scores[n] - top))
                    total = total + exps[n]
                probs = [exps[n] / total for n in range(STATES)]
                for r in cutlass.range_constexpr(per):
                    acc = Float32(0.0)
                    for n in cutlass.range_constexpr(STATES):
                        acc = acc + probs[n] * x[n, r]
                    y[q, r] = acc
                if cutlass.const_expr(self.messages):
                    if tid == 0:
                        for n in cutlass.range_constexpr(STATES):
                            p.mix[add_offset(STATES * (q + 1) + n, mix0)] = probs[n]
            if cutlass.const_expr(self.messages):
                if tid == 0:
                    for n in cutlass.range_constexpr(STATES):
                        p.mix[add_offset(n, mix0)] = rinvs[n]
        squares = []
        for q in cutlass.range_constexpr(nq):
            ss = Float32(0.0)
            for r in cutlass.range_constexpr(per):
                ss = ss + y[q, r] * y[q, r]
            squares.append(ss)
        squares = cta_sum(p.red, self.csync, squares, 2, warp, lane)
        for q in cutlass.range_constexpr(nq):
            rinv = cute.math.rsqrt(squares[q] * inv_h + Float32(self.rms_eps))
            row = add_offset(q * self.tokens, token)
            for r in cutlass.range_constexpr(per):
                k = r * COMPUTE_THREADS + tid
                if cutlass.const_expr(weighted):
                    p.xs[row, k] = BFloat16(Float32(weights[q, r]) * (y[q, r] * rinv))
                else:
                    p.xs[row, k] = BFloat16(y[q, r] * rinv)
        self.csync.sync()

    @cute.jit
    def mixed_state(self, p, query, num_blocks, partial, i, token=0):
        """Element ``i`` of ``normalize_states``' block residual of ``query`` over ``num_blocks``
        blocks and the partial block's element ``partial``, recomputed from the states and MIX
        in its operation order (the same fp32 value); ``token`` as ``normalize_states``'."""
        mix0 = token * MIX_FLOATS
        weights = add_offset(STATES * (query + 1), mix0)
        i = add_offset(i, token * self.hidden)
        acc = Float32(0.0)
        for n in cutlass.range_constexpr(MAX_BLOCKS):
            if n < num_blocks:
                acc = acc + p.mix[weights + n] * (
                    Float32(p.blocks[n][i]) * p.mix[add_offset(n, mix0)]
                )
        return acc + p.mix[weights + MAX_BLOCKS] * (partial * p.mix[add_offset(MAX_BLOCKS, mix0)])

    # ------------------------------------------------------------ message gates

    @cute.jit
    def gate_input(self, s):
        """The prologue query (xs row and MIX weights) of gate ``s``."""
        query = Int32(self.gate_query[0])
        for g in cutlass.range_constexpr(1, GATES):
            query = cutlass.select_(s == g, Int32(self.gate_query[g]), query)
        return query

    @cute.jit
    def gate_params(self, sW, tma, gtensors, tagged: dict, tensors: dict):
        """The kernel's ``GateParams`` (None without messages): the gates' TMA partitions over
        the weight stages ``sW``, their tagged buffers (popped from ``tagged``) and the
        ``mhalf``/``messages`` argument tensors."""
        if cutlass.const_expr(not self.messages):
            return None
        parts = [tma_partition(atom, sW, g, (ROWS, TILE_K)) for atom, g in zip(tma, gtensors)]
        return GateParams(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            h_part=tagged.pop("h_part"),
            gated=tagged.pop("gated"),
            mhalf=tensors["mhalf"],
            messages=tensors["messages"],
        )

    @cute.jit
    def load_gate_item(self, p, w, op, a, b):
        """The load warp's stages of a MSGDOWN (tile ``a``, split ``b``) or GATEMIX (tile ``a``
        of gate ``b``) item."""
        g = p.gate
        if op == MSGDOWN:
            per = self.hidden // TILE_K // GATE_SPLITS
            w = self.load_stages(w, g.tma[0], g.gmem[0], a, b * per, per, "W")
        else:
            for stage in cutlass.range_constexpr(MIX_STAGES):
                row = b * (self.hidden // ROWS) + a * MIX_STAGES + stage
                w = load_tiles(w, g.tma[1], g.gmem[1], row, 0, 1, "W")
        return w

    @cute.jit
    def msgdown(self, p, w, a, split, epoch, warp, lane):
        """MSGDOWN tile ``a``, split ``split``: 32 rows of gate ``a // GATE_TILES``'s down
        projection against a third of the x half of its input (``xs`` row of its query),
        published unrounded."""
        per = self.hidden // TILE_K // GATE_SPLITS
        xbuf = self.gate_input(a // GATE_TILES)
        value = Float32(0.0)
        row = Int32(0)
        # One branch per split: the GEMV input starts at a constant (aligned) column.
        for k in cutlass.range_constexpr(GATE_SPLITS):
            if split == k:
                xs = cute.make_tensor(p.xs.iterator + k * per * TILE_K, p.xs.layout)
                value, row, w = gemv_tile(w, xs, xbuf, per, warp, lane)
        if owns_row(lane):
            index = (split * GATES * GATE_TILES + a) * ROWS + row
            publish(p.gate.h_part, index, value, epoch)
        return w

    @cute.jit
    def gatemix(self, p, w, tile, s, epoch, warp, lane):
        """GATEMIX tile ``tile`` of gate ``s``: elements i of its MIX_STAGES * 32 rows,
        ``ref.message_gate``'s ``(1 - tanh(F_i . h)) x_i + tanh(G_i . h) m_i`` with ``h =
        bf16(x-half partials in split order + message half)`` of the gate's down projection.
        In each stage (32 rows of the fp32 ``[F | G]``) warp w takes rows 4w .. 4w + 3, lane
        group lane // 8 a row, lane j = lane % 8 its columns 8j .. 8j + 7 of F and of G. The
        elements' x and m load before the poll for h, so their latency overlaps it."""
        g = p.gate
        j = lane % 8
        row = ROWS_PER_WARP * warp + lane // 8
        xm = cute.make_rmem_tensor((MIX_STAGES, 2), Float32)
        for stage in cutlass.range_constexpr(MIX_STAGES):
            i = (tile * MIX_STAGES + stage) * ROWS + row
            xm[stage, 0] = self.stream_value(p, s, i)
            xm[stage, 1] = g.messages[s, i]
        hv = cute.make_rmem_tensor((GATE_SPLITS * 8,), Float32)
        stride = GATES * GATE_RANK
        segments = [(g.h_part, k * stride + s * GATE_RANK + 8 * j, 8) for k in range(GATE_SPLITS)]
        fetch(segments, hv, epoch, uniform=True)
        for e in cutlass.range_constexpr(8):
            acc = hv[e]
            for k in cutlass.range_constexpr(1, GATE_SPLITS):
                acc = acc + hv[8 * k + e]
            hv[e] = bf16_round(acc + g.mhalf[s * GATE_RANK + 8 * j + e])
        for stage in cutlass.range_constexpr(MIX_STAGES):
            i = (tile * MIX_STAGES + stage) * ROWS + row
            forget = Float32(0.0)
            gain = Float32(0.0)
            with w.consume() as h:
                rows = cute.recast_tensor(h.view("W"), Float32)  # (ROWS, 2 * GATE_RANK)
                for e in cutlass.range_constexpr(8):
                    forget = forget + rows[row, 8 * j + e] * hv[e]
                    gain = gain + rows[row, GATE_RANK + 8 * j + e] * hv[e]
            forget = group_sum(forget, 8)
            gain = group_sum(gain, 8)
            if j == 0:
                x, m = xm[stage, 0], xm[stage, 1]
                out = (Float32(1.0) - cute.math.tanh(forget)) * x + cute.math.tanh(gain) * m
                publish(g.gated, s * self.hidden + i, out, epoch)
        return w

    @cute.jit
    def gather(self, p, first: cutlass.Constexpr, n: cutlass.Constexpr, norms, epoch, tid, warp,
               lane):  # fmt: skip
        """``xs[s] = bf16(norms[s] * rms_norm(gated stream first + s))`` for each of the ``n``
        gates from ``first``: a half's GEMV inputs, once its GATEMIX items have published."""
        h = self.hidden
        per = h // COMPUTE_THREADS
        vals = cute.make_rmem_tensor((n * per,), Float32)
        segments = [
            (p.gate.gated, (first + s) * h + r * COMPUTE_THREADS + tid, 1)
            for s in range(n)
            for r in range(per)
        ]
        weights = cute.make_rmem_tensor((n, per), BFloat16)
        for s in cutlass.range_constexpr(n):
            for r in cutlass.range_constexpr(per):
                weights[s, r] = norms[s, r * COMPUTE_THREADS + tid]
        fetch(segments, vals, epoch, uniform=True)
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
                p.xs[s, k] = BFloat16(Float32(weights[s, r]) * (vals[s * per + r] * rinv))
        self.csync.sync()


class MatrixRowsKernel(InklingKernel):
    """A launch whose items are 32-row tiles (table word ``a``) of one bf16 ``(rows, hidden)``
    matrix against the GEMV input ``xs[0]``: ``msg_kernel`` and ``head_kernel``. Items depend
    on nothing outside the CTA, so the load warp streams every item's stages at once
    (``load_matrix_rows``). Their params carry ``tma``/``gmem`` (the matrix's atom and tile
    partition), ``sW``, ``xs`` (``(1, hidden)``) and ``red`` (``matrix_params``)."""

    def init_matrix_rows(self, hidden: int, rms_eps: float, stages: int | None, tokens: int = 1):
        """``stages``: weight ring depth in 16 KB stages; ``tokens``: GEMV input rows (one per
        token of a multi-token launch)."""
        if hidden % COMPUTE_THREADS or hidden % TILE_K:
            raise ValueError(f"hidden size must be a multiple of {COMPUTE_THREADS} and {TILE_K}")
        self.hidden, self.rms_eps, self.tokens = hidden, rms_eps, tokens
        self.hidden_k_tiles = hidden // TILE_K
        if tokens > 1:  # tensor-core GEMVs (``examples.megakernel.multi_gemv``)
            self.x_pad = X_PAD
        fixed = 2 * tokens * (hidden + self.x_pad) + 4 * REDUCTIONS * COMPUTE_WARPS * RED_VALUES
        fixed += 4 * DBUF_FLOATS if tokens > 1 else 0
        self.stages = ring_stages(fixed, stages)
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        stage_buffers = {"W": w}
        if tokens > 1:
            stage_buffers["WS"] = self.smem_plan.buffer("WS", BFloat16, align=1024, overlaps=w)
            self.smem_plan.buffer("DBUF", Float32, align=16)
        self.smem_plan.buffer("X", BFloat16, align=16)
        self.smem_plan.buffer("RED", Float32, align=16)
        # Compute reads stages with generic loads before TMA refills them; the layer emits the
        # fence.proxy.async on a Warps release of a Tma-filled ring.
        self.w = Ring(self.stages, Tma(STAGE_BYTES), Warps, buffer=stage_buffers)
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.work = WorkStream(mapping=InstructionTable(words=WORDS), schedule=StaticPersistent())

    @cute.jit
    def load_matrix_rows(self, p, w, work):
        """The load role: every item's weight stages (multi-token: swizzled boxes)."""
        if cutlass.const_expr(self.tokens > 1):
            w = w.views(WS=p.sW)
        else:
            w = w.views(W=p.sW)
        while work.valid:
            _op, tile, _b, _row = work.item
            if cutlass.const_expr(self.tokens > 1):
                w = load_boxes(w, p.tma[0], p.gmem[0], tile, 0, self.hidden_k_tiles, "WS")
            else:
                w = load_tiles(w, p.tma[0], p.gmem[0], tile, 0, self.hidden_k_tiles, "W")
            work.advance()
        return w, work

    def matrix_atom(self, src) -> tuple:
        """TMA ``(atom, tensor)`` of the matrix's stages (multi-token: swizzled boxes)."""
        if self.tokens > 1:
            return swizzled_atom(src, BFloat16)
        return tma_atom(src, (ROWS, TILE_K))

    def matrix_layouts(self) -> dict:
        """The SMEM layouts for ``begin_roles``."""
        layouts = self.core_layouts(self.tokens)
        if self.tokens > 1:
            layouts |= {
                "WS": box_layout(BFloat16, self.stages),
                "DBUF": cute.make_layout((DBUF_FLOATS,)),
            }
        return layouts

    def matrix_params(self, atom, gtensor) -> dict:
        """The params fields of the matrix (TMA ``atom`` over ``gtensor``, ``matrix_atom``) and
        the SMEM buffers, after ``begin_roles(layouts=self.matrix_layouts())``; ``dbuf`` is
        ``examples.megakernel.multi_gemv.finish``'s buffer of multi-token launches."""
        if self.tokens > 1:
            smem, gmem = tma_partition(atom, self.buffer("WS"), gtensor, (ROWS, BOX))
        else:
            smem, gmem = tma_partition(atom, self.buffer("W"), gtensor, (ROWS, TILE_K))
        return {
            "tma": (atom,),
            "gmem": (gmem,),
            "sW": StageView(smem, axis=cute.rank(smem) - 1),
            "xs": self.buffer("X"),
            "red": self.buffer("RED"),
            "dbuf": self.buffer("DBUF") if self.tokens > 1 else None,
        }
