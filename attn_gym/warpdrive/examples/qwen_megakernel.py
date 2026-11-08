"""Qwen3-0.6B batch-one decode step as one persistent warpdrive kernel (a "megakernel").

The host lowers a decode step into a table of instructions, one row per work item::

    for layer in 28 layers:
        QKV     128 items   32 rows of [Wq; Wk; Wv] x RMSNorm(h)            -> qkv
        ATTN    8 x S       kv head x KV split: q/k norm, RoPE, flash decode  -> attention partials
        OPROJ   32 x 4      32 rows x 512-column slice of Wo (split K)        -> o partials
        GATEUP  192         16 gate + 16 up rows x RMSNorm(h + o)             -> SiLU(g) * u
        DOWN    32 x 4      32 rows x 768-column slice of Wd (split K)        -> down partials
    LMHEAD  594         8 x 32 rows of the tied embedding x final RMSNorm    -> logits

By default the host also assigns the rows to CTAs (``list_schedule``, over a latency model)
and lays the table out ``[slot][cta]`` for ``StaticPersistent`` (``assign_static``); the atomic
schedule instead claims rows in dependency order. Every CTA's rows are in dependency order,
so with every CTA resident (one per SM; the launch is cooperative, which guarantees it) the
earliest unfinished row can always run.

Cross-CTA dependencies need no counters: every activation is published as a 64-bit word
``(fp32 value, launch epoch)`` with one relaxed store, and a consumer spins on relaxed loads of
the words it needs until each carries the current epoch (NCCL's LL protocol). A word is
single-copy atomic, so a matching tag means its value is this launch's. That costs one write
and one read propagation per dependency, without fences or a separate flag round trip. The
last CTA to finish advances the epoch for the next launch.

Roles of one CTA (9 warps; 10 with the atomic schedule's scheduler warp)::

    warp 8    load     TMA weight / KV-cache tiles into the ``w`` ring (16 KB stages); it never
                       waits on activations, so it streams the next rows' weights while the
                       compute warps wait for their inputs
    warps 0-7 compute  poll the row's tagged inputs (norms etc. into SMEM) -> consume the row's
                       stages -> publish the outputs

Numerics follow HF's bf16 boundaries: activations hold bf16-rounded values, GEMV accumulation
is fp32 and is rounded once per output (split-K partials are summed in fp32 and rounded by
their consumer, down's by its last split), attention scores are rounded like the eager
matmul, and the softmax uses flash-decoding partials in fp32.

Every per-step intermediate has its own (per-layer) buffer, so there are no write-after-read
hazards inside a launch. The new K/V row at ``pos`` is written by the one attention row whose
TMA-loaded block holds it (masked), after those loads completed.

At ctx 1024 on a GB300 one step takes ~453 us (torch.compile + CUDA graph: ~1230-1365 us;
the 1.31 GB of weights and KV at the ~6.14 TB/s read roofline: 213 us;
docs/warpdrive/examples.md). It is latency-bound: about 140 dependent phases of ~1 us input
propagation plus 1-3 us row latency.
``QwenMegakernel(trace="globaltimer")`` records the work items and ``trace_region``s of every CTA
with their table row as the payload (``qwen_megakernel_trace`` decodes them into per-phase spans
and a Perfetto trace); ``trace="iket"`` emits the same ranges for ``run-iket``.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, vector
from cutlass._mlir.extras import types as T
from cutlass._mlir_helpers.arith import bitcast
from cutlass.cute.nvgpu import cpasync

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    AtomicPersistent,
    Consumer,
    Local,
    NamedBarrier,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    SmemPlan,
    StageView,
    StaticPersistent,
    Tma,
    Warps,
    WarpSpecialized,
    WorkCursor,
    WorkMap,
    WorkStream,
    role,
)

# Qwen3-0.6B.
HIDDEN, LAYERS, Q_HEADS, KV_HEADS, HEAD_DIM = 1024, 28, 16, 8, 128
INTER, VOCAB, EPS = 3072, 151936, 1e-6
QKV_ROWS = (Q_HEADS + 2 * KV_HEADS) * HEAD_DIM  # 4096
ATTN_SCALE = HEAD_DIM**-0.5
LOG2E = 1.4426950408889634

# Tiles: a weight stage is 32 rows x 256 columns, a KV stage 64 positions x 128 dims; 16 KB.
ROWS, TILE_K, BLOCK = 32, 256, 64
STAGE_BYTES = ROWS * TILE_K * 2
O_SPLITS, DOWN_SPLITS = 4, 4  # split-K of o_proj (heads 4s..4s+3) and down (768 columns)
# Weight stages per 32-row tile: QKV, GATEUP and LMHEAD read all 1024 columns.
HIDDEN_K_TILES = HIDDEN // TILE_K
OPROJ_K_TILES = Q_HEADS * HEAD_DIM // TILE_K // O_SPLITS
DOWN_K_TILES = INTER // TILE_K // DOWN_SPLITS
LM_TILES = 8  # row tiles per LM-head item
COMPUTE_WARPS = 8
ROWS_PER_WARP = ROWS // COMPUTE_WARPS
COMPUTE_THREADS = 32 * COMPUTE_WARPS

QKV, ATTN, OPROJ, GATEUP, DOWN, LMHEAD = range(6)
ALL_OPS = (QKV, ATTN, OPROJ, GATEUP, DOWN, LMHEAD)
EVICT_FIRST = 0x12F0000000000000  # TMA L2 cache hint (CUTLASS CacheHintSm90::EVICT_FIRST)
WORDS = 5  # op, layer, a, b, table row
OP_NAMES = ("QKV", "ATTN", "OPROJ", "GATEUP", "DOWN", "LMHEAD")
TRACE_EVENTS = 2048  # globaltimer records per role and CTA (a CTA runs ~140 rows)


class ScratchLayout:
    """Offsets (in 64-bit words) of the tagged per-step intermediates."""

    def __init__(self, splits: int):
        self.shapes = {
            "h_in": (LAYERS + 1, HIDDEN),  # layer inputs (residual); [LAYERS]: LM head's
            "qkv": (LAYERS, QKV_ROWS),
            "attn_o": (LAYERS, KV_HEADS, splits, 2, HEAD_DIM),  # unnormalized partial O
            "attn_ml": (LAYERS, KV_HEADS, 2, splits, 2),  # (m, l) per head, split; log2 m
            "o_part": (LAYERS, O_SPLITS, HIDDEN),
            "h_mid": (LAYERS, HIDDEN),  # residual after attention
            "act": (LAYERS, INTER),
            "d_part": (LAYERS, DOWN_SPLITS, HIDDEN),
        }
        self.offsets, offset = {}, 0
        for name, shape in self.shapes.items():
            self.offsets[name] = offset
            offset += (math.prod(shape) + 15) // 16 * 16  # 128-byte aligned
        self.size = offset


@dataclass(frozen=True)
class InstructionTable(WorkMap):
    """Row ``index`` of an int32 ``(rows, WORDS)`` table; rows with a negative op are padding
    and are dropped for every role."""

    words: int = WORDS
    skips = True

    def count64(self, space):
        rows = space.shape[0]
        return rows if isinstance(rows, int) else Int64(rows)

    def decode(self, index, space, state):
        words = tuple(space[index, w] for w in range(self.words))
        return words[0] >= 0, words, state

    def get_key(self) -> str:
        return f"instructions{self.words}"


@dataclass(frozen=True)
class MegakernelParams:
    tma: tuple  # TMA atoms: wqkv, wo, wgu, wd, lm, k cache, v cache
    gmem: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    sW: StageView  # weight stage partition
    sKV: StageView  # KV stage partition
    xs: cute.Tensor  # (2, INTER) bf16 GEMV inputs, double-buffered by item
    red: cute.Tensor  # (COMPUTE_WARPS,) fp32 block reduction
    att: cute.Tensor  # (9, 2 * HEAD_DIM + 4) fp32 per-warp attention states
    qkvs: cute.Tensor  # (4 * HEAD_DIM,) fp32 an attention row's q (2 heads), k and v
    mls: cute.Tensor  # (4, splits, 2) fp32 an o_proj row's heads' (m, l) per KV split
    # Tagged int64 activations, flat row-major (ScratchLayout.shapes; see the *_index helpers).
    h_in: cute.Tensor
    qkv: cute.Tensor
    attn_o: cute.Tensor
    attn_ml: cute.Tensor
    o_part: cute.Tensor
    h_mid: cute.Tensor
    act: cute.Tensor
    d_part: cute.Tensor
    logits: cute.Tensor  # (VOCAB,) fp32
    state: cute.Tensor  # int32 [epoch of the previous launch, CTAs finished]
    step: cute.Tensor  # int32 [token, pos]
    embed: cute.Tensor  # (VOCAB, HIDDEN) bf16
    ln_in: cute.Tensor  # (LAYERS, HIDDEN) bf16
    ln_post: cute.Tensor
    q_norm: cute.Tensor  # (LAYERS, HEAD_DIM) bf16
    k_norm: cute.Tensor
    final_norm: cute.Tensor  # (HIDDEN,) bf16
    rope: cute.Tensor  # (max_len, 2, HEAD_DIM) bf16 cos, sin
    k_cache: cute.Tensor  # (LAYERS * KV_HEADS * max_len, HEAD_DIM) bf16
    v_cache: cute.Tensor


# The argument tensors in argument order: MegakernelParams fields, and the trace buffer.
TENSOR_ARGS = (
    "logits",
    "state",
    "step",
    "trace_buffer",
    "embed",
    "ln_in",
    "ln_post",
    "q_norm",
    "k_norm",
    "final_norm",
    "rope",
    "k_cache",
    "v_cache",
)


# ------------------------------------------------------------------- tagged words


def pack_tagged(value, tag):
    """The 64-bit word (value bits low, tag high)."""
    low = arith.extui(T.i64(), bitcast(Float32(value).ir_value(), T.i32()))
    high = arith.shli(arith.extui(T.i64(), Int32(tag).ir_value()), arith.constant(T.i64(), 32))
    return Int64(arith.ori(low, high))


def unpack_tagged(word):
    bits = Int64(word).ir_value()
    low = arith.trunci(T.i32(), bits)
    high = arith.trunci(T.i32(), arith.shrui(bits, arith.constant(T.i64(), 32)))
    return Float32(bitcast(low, T.f32())), Int32(high)


@cute.jit
def publish(buf, index, value, epoch):
    cute.arch.store(buf.iterator + index, pack_tagged(value, epoch), sem="relaxed", scope="gpu")


def load_vector(ptr, n: int) -> list:
    """``n`` (2 or 4) int64 words from ``ptr`` (``8 * n``-byte aligned) in one relaxed load."""
    vec = cute.arch.load(ptr, ir.VectorType.get([n], Int64.mlir_type), sem="relaxed", scope="gpu")
    return [Int64(vector.extract(vec, dynamic_position=[], static_position=[k])) for k in range(n)]


@cute.jit
def load_words(buf, index, count: cutlass.Constexpr):
    """``count`` tagged words from ``buf[index]`` with relaxed gpu-scope loads: four words
    (32 B), then two (16 B), then a scalar tail. Each strong load costs one L2 request per
    lane. ``index`` must be a multiple of 4 when ``count >= 4`` and even when ``count >= 2``."""
    words = []
    quads = count // 4 * 4
    for e in cutlass.range_constexpr(0, quads, 4):
        words += load_vector(buf.iterator + (index + e), 4)
    for e in cutlass.range_constexpr(quads, count - 1, 2):
        words += load_vector(buf.iterator + (index + e), 2)
    if cutlass.const_expr(count % 2):
        last = buf.iterator + (index + count - 1)
        words.append(cute.arch.load(last, Int64, sem="relaxed", scope="gpu"))
    return words


@cute.jit
def fetch(segments, out, epoch):
    """Spin until every word of ``segments`` carries ``epoch``; their values fill ``out`` in
    order. Segments are ``(buffer, start, count)`` runs aligned as ``load_words`` requires."""
    starts = []
    total = 0
    for s in cutlass.range_constexpr(len(segments)):
        starts.append(total)
        total += segments[s][2]
    ready = Int32(0)
    while ready == 0:
        ready = Int32(1)
        for s in cutlass.range_constexpr(len(segments)):
            buf, start, count = segments[s]
            words = load_words(buf, start, count)
            for e in cutlass.range_constexpr(count):
                value, tag = unpack_tagged(words[e])
                out[starts[s] + e] = value
                ready = ready & Int32(tag == epoch)


def qkv_index(layer, i):
    return layer * QKV_ROWS + i


def hidden_index(layer, i):
    """h_in, h_mid."""
    return layer * HIDDEN + i


def part_index(layer, split, i):
    """o_part, d_part."""
    return (layer * O_SPLITS + split) * HIDDEN + i


def act_index(layer, i):
    return layer * INTER + i


# ------------------------------------------------------------------ register math


@cute.jit
def bf16_round(x):
    """Round to the nearest bf16 (ties to even), kept in fp32. Integer ops run at full rate;
    an F2F conversion is a quarter-rate instruction and the norms round every element."""
    bits = Int32(bitcast(Float32(x).ir_value(), T.i32()))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & Int32(-65536)
    return Float32(bitcast(bits.ir_value(), T.f32()))


@cute.jit
def warp_transpose_reduce(acc, lane, n: cutlass.Constexpr):
    """Sum each of ``n`` (a power of two <= 32) per-lane values over the warp; lane l ends with
    the sum of value ``l // (32 // n)``. Halving exchanges, then a full reduction: 6 shuffles
    for n = 4 instead of 20."""
    a = []
    for r in cutlass.range_constexpr(n):
        a.append(acc[r])
    halvings = n.bit_length() - 1
    for step in cutlass.range_constexpr(halvings):
        level = 16 >> step
        width = n >> (step + 1)
        hi = (lane & level) != 0
        out = []
        for j in cutlass.range_constexpr(width):
            send = cutlass.select_(hi, a[j], a[j + width])
            keep = cutlass.select_(hi, a[j + width], a[j])
            out.append(keep + cute.arch.shuffle_sync_bfly(send, offset=level))
        a = out
    v = a[0]
    for step in cutlass.range_constexpr(halvings, 5):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=16 >> step)
    return v


@cute.jit
def bf16x8_to_f32(frag):
    """8 bf16 registers as fp32: element 2j = word j << 16, 2j + 1 = word j & 0xffff0000."""
    words = cute.recast_tensor(frag, Uint32)
    bits = cute.make_rmem_tensor((8,), Uint32)
    for j in cutlass.range_constexpr(4):
        bits[2 * j] = words[j] << Uint32(16)
        bits[2 * j + 1] = words[j] & Uint32(0xFFFF0000)
    return cute.recast_tensor(bits, Float32)


@cute.jit
def group_sum(v, width: cutlass.Constexpr):
    """Sum over aligned groups of ``width`` lanes."""
    for step in cutlass.range_constexpr(width.bit_length() - 1):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=1 << step)
    return v


@cute.jit
def load_vec(src, offset, n: cutlass.Constexpr):
    """``n`` contiguous elements of a 1-D tensor starting at ``offset`` (a multiple of n),
    as fp32 registers."""
    out = cute.make_rmem_tensor((n,), Float32)
    load_into(out, src, offset, n)
    return out


@cute.jit
def load_into(out, src, offset, n: cutlass.Constexpr):
    frag = cute.make_rmem_tensor((n,), src.element_type)
    cute.autovec_copy(cute.local_tile(src, (n,), (offset // n,)), frag)
    out.store(frag.load().to(Float32))


@cute.jit
def store_vec(dst, offset, frag, n: cutlass.Constexpr):
    cast = cute.make_rmem_tensor((n,), dst.element_type)
    cast.store(frag.load().to(dst.element_type))
    cute.autovec_copy(cast, cute.local_tile(dst, (n,), (offset // n,)))


class QwenDecodeMegakernel(WarpSpecialized):
    def __init__(
        self,
        stages: int = 12,
        splits: int = 16,
        schedule: str = "static",
        roles: dict[str, RoleConfig] | None = None,
    ):
        """``stages``: weight/KV ring depth in 16 KB stages. ``splits``: KV splits per kv head;
        on a 152-SM GB300, 19 (8 x 19 attention items, one per SM) measured faster than 16 at
        ctx 6208-8192 and slower at 1024-6144. The count is static and the position is read on
        the device, so pick it for the contexts a runner decodes; the capacity alone does not
        decide it.
        ``schedule``: ``"static"`` runs a host-assigned per-CTA row list (``StaticPersistent``
        over a ``[slot][cta]`` table; needs every CTA resident), ``"atomic"`` claims rows in
        dependency order (``AtomicPersistent``, one row decoded ahead).
        ``roles``: placement by role name over the ``@role`` defaults."""
        if schedule not in ("static", "atomic"):
            raise ValueError(f"schedule must be 'static' or 'atomic', got {schedule!r}")
        # The GEMV (ROWS_PER_WARP rows per warp), RMSNorm (4 elements per thread) and attention
        # (8 positions per warp of a block) tilings fix the compute role's size.
        compute = (roles or {}).get("compute")
        if compute is not None and [len(r) for r in compute.warps] != [COMPUTE_WARPS]:
            raise ProtocolError(
                f"roles['compute'] must be one instance of {COMPUTE_WARPS} warps, got "
                f"{compute.warps}"
            )
        self.roles = roles
        # combine_attention fetches a head's 2 * splits (m, l) words with 64 threads.
        if not 1 <= splits <= 32:
            raise ValueError(f"splits must be in [1, 32], got {splits}")
        self.stages, self.splits = stages, splits
        self.scratch_layout = ScratchLayout(splits)
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        kv = self.smem_plan.buffer("KV", BFloat16, align=1024, overlaps=w)
        self.smem_plan.buffer("X", BFloat16, align=16)  # GEMV inputs are bf16 values
        self.smem_plan.buffer("RED", Float32, align=16)
        self.smem_plan.buffer("ATT", Float32, align=16)
        self.smem_plan.buffer("QKVS", Float32, align=16)
        self.smem_plan.buffer("MLS", Float32, align=16)
        # Compute reads stages with generic loads before TMA refills them; the layer derives the
        # fence.proxy.async release fence for a Tma-filled ring released by Warps.
        self.w = Ring(stages, Tma(STAGE_BYTES), Warps, buffer={"W": w, "KV": kv})
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.work = WorkStream(
            mapping=InstructionTable(),
            schedule=StaticPersistent() if schedule == "static" else AtomicPersistent(stages=1),
        )

    def get_name(self) -> str:
        return (
            f"warpdrive_qwen_decode_s{self.stages}_kv{self.splits}_"
            f"{self.work.get_key()}{self.trace_key()}"
        )

    # ------------------------------------------------------------------ roles

    @role(warps=COMPUTE_WARPS)
    @cute.jit
    def load(self, p: MegakernelParams, *, w: Producer, work: WorkCursor):
        w = w.views(W=p.sW, KV=p.sKV)
        tma_qkv, tma_o, tma_gu, tma_d, tma_lm, tma_k, tma_v = p.tma
        g_qkv, g_o, g_gu, g_d, g_lm, g_k, g_v = p.gmem
        pos = p.step[1]
        max_len = p.k_cache.shape[0] // (LAYERS * KV_HEADS)
        while work.valid:
            op, layer, a, b, _row = work.item
            if op == ATTN:
                blk0, blk1 = self.split_blocks(pos, b)
                row0 = (layer * KV_HEADS + a) * (max_len // BLOCK)
                for blk in cutlass.range(blk0, blk1, unroll=1):
                    w = self.load_tiles(w, tma_k, g_k, row0 + blk, 0, 1, "KV")
                    w = self.load_tiles(w, tma_v, g_v, row0 + blk, 0, 1, "KV")
            elif op == QKV:
                tile = layer * (QKV_ROWS // ROWS) + a
                w = self.load_tiles(w, tma_qkv, g_qkv, tile, 0, HIDDEN_K_TILES, "W")
            elif op == OPROJ:
                tile = layer * (HIDDEN // ROWS) + a
                w = self.load_tiles(w, tma_o, g_o, tile, b * OPROJ_K_TILES, OPROJ_K_TILES, "W")
            elif op == GATEUP:
                tile = layer * (2 * INTER // ROWS) + a
                w = self.load_tiles(w, tma_gu, g_gu, tile, 0, HIDDEN_K_TILES, "W")
            elif op == DOWN:
                tile = layer * (HIDDEN // ROWS) + a
                w = self.load_tiles(w, tma_d, g_d, tile, b * DOWN_K_TILES, DOWN_K_TILES, "W")
            else:
                for rt in cutlass.range(a, a + b, unroll=1):
                    w = self.load_tiles(w, tma_lm, g_lm, rt, 0, HIDDEN_K_TILES, "W")
            work.advance()
        return w, work

    @cute.jit
    def load_tiles(
        self, w, atom, tiles, row, k0, count: cutlass.Constexpr, buffer: cutlass.Constexpr
    ):
        """``count`` stages of tiles ``(row, k0 + k)``. Weights and KV rows are read once per
        step, so they are marked evict-first in L2."""
        for k in cutlass.range_constexpr(count):
            with w.produce() as h:
                cute.copy(
                    atom,
                    tiles[None, row, k0 + k],
                    h.view(buffer),
                    tma_bar_ptr=h.barrier,
                    cache_policy=Int64(EVICT_FIRST),
                )
        return w

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(self, p: MegakernelParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        ctas = cute.arch.grid_dim()[0]
        pos = p.step[1]
        token = p.step[0]
        epoch = p.state[0] + 1
        # The GEMV input in xs[xbuf] is reused while consecutive rows share it (key); row 0
        # of GATEUP and of layer 0's QKV always runs the prologue, which publishes a residual.
        xkey = Int32(-1)
        xbuf = Int32(0)
        while work.valid:
            op, layer, a, b, row = work.item
            if op == ATTN:
                w = self.attention(p, w, layer, a, b, pos, epoch, tid, warp, lane, row)
            else:
                ks = cutlass.select_((op == OPROJ) | (op == DOWN), b, 0)
                key = (op * LAYERS + layer) * 8 + ks
                publishes = (a == 0) & ((op == GATEUP) | ((op == QKV) & (layer == 0)))
                if (key != xkey) | publishes:
                    xkey = key
                    xbuf = 1 - xbuf
                    self.prologue(p, op, layer, a, ks, token, epoch, xbuf, tid, warp, lane, row)
                    self.csync.sync()
                w = self.gemv(p, w, op, layer, a, b, epoch, xbuf, warp, lane, row)
                if (op == DOWN) & (b == DOWN_SPLITS - 1) & (tid < ROWS):
                    self.reduce_down(p, layer, a * ROWS + tid, epoch, row)
            work.advance()
        # Every CTA read the epoch before it finished; the last one advances it.
        if tid == 0:
            finished = cute.arch.atomic_add(
                p.state.iterator + 1, Int32(1), sem="acq_rel", scope="gpu"
            )
            if finished == ctas - 1:
                p.state[1] = Int32(0)
                p.state[0] = epoch
        return w, work

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def kv_blocks(self, pos):
        """The KV block count of positions ``[0, pos)`` and the most blocks one split reads."""
        blocks = (pos + BLOCK - 1) // BLOCK
        return blocks, (blocks + self.splits - 1) // self.splits

    @cute.jit
    def split_blocks(self, pos, split):
        """KV blocks ``[blk0, blk1)`` of positions ``[0, pos)`` that ``split`` reads."""
        blocks, per = self.kv_blocks(pos)
        blk0 = split * per
        blk1 = cutlass.min(blk0 + per, blocks)
        return blk0, cutlass.max(blk1, blk0)

    @cute.jit
    def prologue(
        self, p: MegakernelParams, op, layer, a, ks, token, epoch, xbuf, tid, warp, lane, row
    ):
        """Fetch the row's inputs and write its GEMV input into ``xs[xbuf]`` (caller syncs).
        The RMSNorm weight is loaded before polling; the ``poll`` region ends when the inputs
        are ready."""
        xs = p.xs[xbuf, None]
        norm_weight = cute.make_rmem_tensor((4,), Float32)
        if op == QKV:
            load_into(norm_weight, p.ln_in[layer, None], 4 * tid, 4)
        elif op == GATEUP:
            load_into(norm_weight, p.ln_post[layer, None], 4 * tid, 4)
        elif op == LMHEAD:
            load_into(norm_weight, p.final_norm, 4 * tid, 4)
        x = cute.make_rmem_tensor((4,), Float32)
        if op == QKV:
            if layer == 0:
                load_into(x, p.embed[token, None], 4 * tid, 4)
                if a == 0:
                    for e in cutlass.range_constexpr(4):
                        publish(p.h_in, hidden_index(0, 4 * tid + e), x[e], epoch)
            else:
                self.published_residual(x, p.h_in, layer, epoch, tid, row)
            self.rms_norm(p, x, norm_weight, xs, tid, warp, lane)
        elif op == GATEUP:
            self.residual(x, p.h_in, p.o_part, layer, epoch, tid, row)
            if a == 0:
                for e in cutlass.range_constexpr(4):
                    publish(p.h_mid, hidden_index(layer, 4 * tid + e), x[e], epoch)
            self.rms_norm(p, x, norm_weight, xs, tid, warp, lane)
        elif op == LMHEAD:
            self.published_residual(x, p.h_in, LAYERS, epoch, tid, row)
            self.rms_norm(p, x, norm_weight, xs, tid, warp, lane)
        elif op == DOWN:
            span = INTER // DOWN_SPLITS
            count = span // COMPUTE_THREADS
            vals = cute.make_rmem_tensor((count,), Float32)
            segments = []
            for i in cutlass.range_constexpr(count):
                index = act_index(layer, ks * span + i * COMPUTE_THREADS + tid)
                segments.append((p.act, index, 1))
            with self.trace_region("poll", payload=row):
                fetch(segments, vals, epoch)
            for i in cutlass.range_constexpr(count):
                xs[i * COMPUTE_THREADS + tid] = BFloat16(vals[i])
        else:
            self.combine_attention(p, layer, ks, epoch, xs, tid, row)

    @cute.jit
    def residual(self, x, base, parts, layer, epoch, tid, row):
        """x = bf16(base + bf16(sum of split-K partials)) for this thread's 4 elements."""
        splits = O_SPLITS  # == DOWN_SPLITS
        vals = cute.make_rmem_tensor((4 * (splits + 1),), Float32)
        segments = [(base, hidden_index(layer, 4 * tid), 4)]
        for k in cutlass.range_constexpr(splits):
            segments.append((parts, part_index(layer, k, 4 * tid), 4))
        with self.trace_region("poll", payload=row):
            fetch(segments, vals, epoch)
        for e in cutlass.range_constexpr(4):
            total = vals[4 + e]
            for k in cutlass.range_constexpr(1, splits):
                total = total + vals[4 * (k + 1) + e]
            x[e] = bf16_round(vals[e] + bf16_round(total))

    @cute.jit
    def published_residual(self, x, buf, layer, epoch, tid, row):
        """This thread's 4 elements of the residual ``buf[layer]``, published finished."""
        with self.trace_region("poll", payload=row):
            fetch([(buf, hidden_index(layer, 4 * tid), 4)], x, epoch)

    @cute.jit
    def reduce_down(self, p: MegakernelParams, layer, col, epoch, row):
        """Publish the next layer's residual column ``col`` = bf16(h_mid + bf16(sum of the
        down partials)), summed in split order as ``residual`` does.

        The last down split runs it after publishing its own partial: that split finishes
        last (its GATEUP rows are the second wave), and QKV and the LM head then fetch 1 word
        per element instead of 5."""
        vals = cute.make_rmem_tensor((DOWN_SPLITS + 1,), Float32)
        segments = [(p.h_mid, hidden_index(layer, col), 1)]
        for k in cutlass.range_constexpr(DOWN_SPLITS):
            segments.append((p.d_part, part_index(layer, k, col), 1))
        with self.trace_region("reduce.poll", payload=row):
            fetch(segments, vals, epoch)
        total = vals[1]
        for k in cutlass.range_constexpr(2, DOWN_SPLITS + 1):
            total = total + vals[k]
        h = bf16_round(vals[0] + bf16_round(total))
        publish(p.h_in, hidden_index(layer + 1, col), h, epoch)

    @cute.jit
    def rms_norm(self, p: MegakernelParams, x, wv, xs, tid, warp, lane):
        """HF RMSNorm of the 1024-vector held 4 per thread: bf16(w * bf16(x * rsqrt))."""
        ss = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            ss = ss + x[e] * x[e]
        ss = group_sum(ss, 32)
        if lane == 0:
            p.red[warp] = ss
        self.csync.sync()
        total = Float32(0.0)
        for i in cutlass.range_constexpr(COMPUTE_WARPS):
            total = total + p.red[i]
        rinv = cute.math.rsqrt(total / HIDDEN + EPS)
        for e in cutlass.range_constexpr(4):
            x[e] = bf16_round(wv[e] * bf16_round(x[e] * rinv))
        store_vec(xs, 4 * tid, x, 4)

    def attn_o_index(self, layer, kvh, split, head, d):
        return (((layer * KV_HEADS + kvh) * self.splits + split) * 2 + head) * HEAD_DIM + d

    def attn_ml_index(self, layer, kvh, head, split, k):
        """k = 0: m (log2 domain), 1: l."""
        return (((layer * KV_HEADS + kvh) * 2 + head) * self.splits + split) * 2 + k

    @cute.jit
    def combine_attention(self, p: MegakernelParams, layer, ks, epoch, xs, tid, row):
        """o_proj input slice ``ks``: q heads 4ks..4ks+3 from their KV-split partials.

        Thread t combines dims d, d + 1 of head t // 64. The heads' (m, l) pairs are fetched
        once into SMEM (by threads t % 64 < 2 * splits) rather than by all 64 threads."""
        hh = tid // 64  # head within the slice
        d = (tid % 64) * 2
        kvh = 2 * ks + hh // 2
        head = hh % 2
        pairs = 2 * self.splits  # (m, l) words per head
        word = cutlass.min(tid % 64, pairs - 1)
        vals = cute.make_rmem_tensor((1 + 2 * self.splits,), Float32)  # ml word, o[d, d + 1]
        segments = [(p.attn_ml, self.attn_ml_index(layer, kvh, head, 0, 0) + word, 1)]
        for sp in cutlass.range_constexpr(self.splits):
            segments.append((p.attn_o, self.attn_o_index(layer, kvh, sp, head, d), 2))
        with self.trace_region("poll", payload=row):
            fetch(segments, vals, epoch)
        if tid % 64 < pairs:
            p.mls[hh * pairs + word] = vals[0]
        self.csync.sync()
        m = Float32(-1e30)
        for sp in cutlass.range_constexpr(self.splits):
            m = cutlass.max(m, p.mls[hh * pairs + 2 * sp])
        den = Float32(0.0)
        num0 = Float32(0.0)
        num1 = Float32(0.0)
        for sp in cutlass.range_constexpr(self.splits):
            scale = cute.math.exp2(p.mls[hh * pairs + 2 * sp] - m)
            den = den + p.mls[hh * pairs + 2 * sp + 1] * scale
            num0 = num0 + vals[1 + 2 * sp] * scale
            num1 = num1 + vals[2 + 2 * sp] * scale
        xs[hh * HEAD_DIM + d] = BFloat16(num0 / den)
        xs[hh * HEAD_DIM + d + 1] = BFloat16(num1 / den)

    @cute.jit
    def gemv(self, p: MegakernelParams, w, op, layer, a, b, epoch, xbuf, warp, lane, row_index):
        """32-row tiles: warp w owns rows 4w..4w+3 of a stage, lane l columns 8l..8l+7."""
        k_tiles = HIDDEN_K_TILES
        if op == OPROJ:
            k_tiles = OPROJ_K_TILES
        if op == DOWN:
            k_tiles = DOWN_K_TILES
        row_tiles = cutlass.select_(op == LMHEAD, b, 1)
        for t in cutlass.range(row_tiles, unroll=1):
            # Even and odd columns accumulate separately, two per packed FFMA2.
            acc = cute.make_rmem_tensor((ROWS_PER_WARP, 2), Float32)
            acc.fill(0.0)
            with self.trace_region("stream", payload=row_index):
                # Unrolled: the rolled loop kept a 32-byte stack frame (12-20 us per step).
                for kt in cutlass.range(k_tiles, unroll_full=True):
                    with w.consume() as h:
                        sW = h.data["W"]
                        wf = []
                        for r in cutlass.range_constexpr(ROWS_PER_WARP):
                            frag = cute.make_rmem_tensor((8,), BFloat16)
                            row_w = sW[warp * ROWS_PER_WARP + r, None]
                            cute.autovec_copy(cute.local_tile(row_w, (8,), (lane,)), frag)
                            wf.append(frag)
                        xh = cute.make_rmem_tensor((8,), BFloat16)
                        cute.autovec_copy(
                            cute.local_tile(p.xs[xbuf, None], (8,), (kt * 32 + lane,)), xh
                        )
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
            row = warp * ROWS_PER_WARP + lane // (32 // ROWS_PER_WARP)  # within the tile
            if op == GATEUP:
                # Rows alternate gate, up: lanes 0-7 hold gate j, lanes 8-15 up j.
                up = cute.arch.shuffle_sync_down(v, offset=8)
                if lane % 16 == 0:
                    g = bf16_round(v)
                    silu = bf16_round(g / (1.0 + cute.math.exp2(-g * LOG2E)))
                    out = bf16_round(silu * bf16_round(up))
                    publish(p.act, act_index(layer, a * (ROWS // 2) + row // 2), out, epoch)
            elif lane % (32 // ROWS_PER_WARP) == 0:
                if op == QKV:
                    publish(p.qkv, qkv_index(layer, a * ROWS + row), bf16_round(v), epoch)
                elif op == OPROJ:
                    publish(p.o_part, part_index(layer, b, a * ROWS + row), v, epoch)
                elif op == DOWN:
                    publish(p.d_part, part_index(layer, b, a * ROWS + row), v, epoch)
                else:
                    p.logits[(a + t) * ROWS + row] = bf16_round(v)
        return w

    @cute.jit
    def attention(
        self, p: MegakernelParams, w, layer, kvh, split, pos, epoch, tid, warp, lane, row_index
    ):
        """Flash-decoding partial of kv head ``kvh``'s two q heads over one KV split.

        The new token's q (2 heads), k and v are fetched once into SMEM. Warp w scores
        positions 8w + 4i + g (i = 0, 1) of each 64-position block; lane group g = lane // 8
        holds a position, lane j = lane % 8 its dims 8j..8j+7 and 64+8j..64+8j+7 (the
        rotate-half pairs). Each warp keeps its own online-softmax state; the warps (and the
        new token, in the last split) are merged through SMEM at the end."""
        g = lane // 8
        j = lane % 8
        d0 = 8 * j
        max_len = p.k_cache.shape[0] // (LAYERS * KV_HEADS)
        # Warps 0, 1, 2 norm and rotate q0, q1, k: lane l owns dims 2l, 2l + 1 and their
        # rotate-half partners 64 + 2l, 65 + 2l. Their weights and RoPE do not depend on other
        # rows, so they load before the fetch.
        norm = cute.make_rmem_tensor((4,), Float32)
        cos = cute.make_rmem_tensor((4,), Float32)
        sin = cute.make_rmem_tensor((4,), Float32)
        norm_weight = p.q_norm[layer, None]
        if warp == 2:
            norm_weight = p.k_norm[layer, None]
        for half in cutlass.range_constexpr(2):
            dim = 2 * lane + 64 * half
            pair = load_vec(norm_weight, dim, 2)
            cs = load_vec(p.rope[pos, 0, None], dim, 2)
            sn = load_vec(p.rope[pos, 1, None], dim, 2)
            for e in cutlass.range_constexpr(2):
                norm[2 * half + e] = pair[e]
                cos[2 * half + e] = cs[e]
                sin[2 * half + e] = sn[e]
        # Thread t fetches words 2t, 2t + 1 of [q heads 2kvh, 2kvh + 1 | k | v].
        column = 2 * kvh * HEAD_DIM + 2 * tid
        column = cutlass.select_(tid >= 128, (Q_HEADS + kvh) * HEAD_DIM + 2 * tid - 256, column)
        k_end = 2 * (HEAD_DIM + HEAD_DIM // 2)  # 384: threads 192.. fetch v
        v_head = Q_HEADS + KV_HEADS + kvh
        column = cutlass.select_(tid >= 192, v_head * HEAD_DIM + 2 * tid - k_end, column)
        raw = cute.make_rmem_tensor((2,), Float32)
        with self.trace_region("poll", payload=row_index):
            fetch([(p.qkv, qkv_index(layer, column), 2)], raw, epoch)
        p.qkvs[2 * tid] = raw[0]
        p.qkvs[2 * tid + 1] = raw[1]
        self.csync.sync()
        if warp < 3:
            self.head_norm_rope(p.qkvs, warp * HEAD_DIM + 2 * lane, norm, cos, sin)
        self.csync.sync()
        q = cute.make_rmem_tensor((2, 16), Float32)
        q0 = self.head_vec(p.qkvs, 0, d0)
        q1 = self.head_vec(p.qkvs, HEAD_DIM, d0)
        k_new = self.head_vec(p.qkvs, 2 * HEAD_DIM, d0)
        v_new = self.head_vec(p.qkvs, 3 * HEAD_DIM, d0)
        for e in cutlass.range_constexpr(16):
            q[0, e] = q0[e]
            q[1, e] = q1[e]
        with self.trace_region("attn.blocks", payload=row_index):
            m = cute.make_rmem_tensor((2,), Float32)  # running max (log2 domain), warp-uniform
            lsum = cute.make_rmem_tensor((2,), Float32)  # this lane group's sum of p
            o = cute.make_rmem_tensor((2, 16), Float32)  # this lane group's sum of p * v
            m.fill(-1e30)
            lsum.fill(0.0)
            o.fill(0.0)
            blk0, blk1 = self.split_blocks(pos, split)
            for blk in cutlass.range(blk0, blk1, unroll=1):
                sc = cute.make_rmem_tensor((2, 2), Float32)  # (head, i)
                with w.consume() as h:
                    sK = h.data["KV"]
                    for i in cutlass.range_constexpr(2):
                        t = 8 * warp + 4 * i + g
                        kv = self.stage_row(sK, t, d0)
                        valid = blk * BLOCK + t < pos
                        for hq in cutlass.range_constexpr(2):
                            dot = Float32(0.0)
                            for e in cutlass.range_constexpr(16):
                                dot = dot + q[hq, e] * kv[e]
                            dot = group_sum(dot, 8)
                            score = bf16_round(bf16_round(dot) * ATTN_SCALE) * LOG2E
                            sc[hq, i] = cutlass.select_(valid, score, -Float32.inf)
                pr = cute.make_rmem_tensor((2, 2), Float32)
                for hq in cutlass.range_constexpr(2):
                    bm = cutlass.max(sc[hq, 0], sc[hq, 1])
                    bm = cutlass.max(bm, cute.arch.shuffle_sync_bfly(bm, offset=8))
                    bm = cutlass.max(bm, cute.arch.shuffle_sync_bfly(bm, offset=16))
                    m_new = cutlass.max(m[hq], bm)
                    alpha = cute.math.exp2(m[hq] - m_new)
                    m[hq] = m_new
                    pr[hq, 0] = cute.math.exp2(sc[hq, 0] - m_new)
                    pr[hq, 1] = cute.math.exp2(sc[hq, 1] - m_new)
                    lsum[hq] = lsum[hq] * alpha + pr[hq, 0] + pr[hq, 1]
                    for e in cutlass.range_constexpr(16):
                        o[hq, e] = o[hq, e] * alpha
                with w.consume() as h:
                    sV = h.data["KV"]
                    for i in cutlass.range_constexpr(2):
                        t = 8 * warp + 4 * i + g
                        if blk * BLOCK + t < pos:  # masked rows may hold non-finite garbage
                            vv = self.stage_row(sV, t, d0)
                            for hq in cutlass.range_constexpr(2):
                                for e in cutlass.range_constexpr(16):
                                    o[hq, e] = o[hq, e] + pr[hq, i] * vv[e]
        # Append the new K/V row at pos. Only the split whose KV block holds pos TMA-loads
        # that row (attention masks it), and it writes after its own loads completed, so no
        # async-proxy read races the generic store. When pos starts a block, no split reads it.
        blocks, per = self.kv_blocks(pos)
        writer = cutlass.select_(pos // BLOCK < blocks, pos // BLOCK // per, self.splits - 1)
        if split == writer:
            if tid < 8:  # warp 0, lane group 0
                row = (layer * KV_HEADS + kvh) * max_len + pos
                for half in cutlass.range_constexpr(2):
                    kf = cute.make_rmem_tensor((8,), Float32)
                    vf = cute.make_rmem_tensor((8,), Float32)
                    for e in cutlass.range_constexpr(8):
                        kf[e] = k_new[half * 8 + e]
                        vf[e] = v_new[half * 8 + e]
                    store_vec(p.k_cache[row, None], d0 + 64 * half, kf, 8)
                    store_vec(p.v_cache[row, None], d0 + 64 * half, vf, 8)
        # Merge the four lane groups (same m), then publish this warp's state.
        for hq in cutlass.range_constexpr(2):
            lsum[hq] = lsum[hq] + cute.arch.shuffle_sync_bfly(lsum[hq], offset=8)
            lsum[hq] = lsum[hq] + cute.arch.shuffle_sync_bfly(lsum[hq], offset=16)
            for e in cutlass.range_constexpr(16):
                o[hq, e] = o[hq, e] + cute.arch.shuffle_sync_bfly(o[hq, e], offset=8)
                o[hq, e] = o[hq, e] + cute.arch.shuffle_sync_bfly(o[hq, e], offset=16)
        att = p.att
        if g == 0:
            for hq in cutlass.range_constexpr(2):
                for e in cutlass.range_constexpr(8):
                    att[warp, hq * HEAD_DIM + d0 + e] = o[hq, e]
                    att[warp, hq * HEAD_DIM + d0 + 64 + e] = o[hq, 8 + e]
            if j == 0:
                for hq in cutlass.range_constexpr(2):
                    att[warp, 2 * HEAD_DIM + hq] = m[hq]
                    att[warp, 2 * HEAD_DIM + 2 + hq] = lsum[hq]
        last = split == self.splits - 1
        if last:
            if warp == 0:  # entry COMPUTE_WARPS, the new token: m = its score, l = 1, O = v
                for hq in cutlass.range_constexpr(2):
                    dot = Float32(0.0)
                    for e in cutlass.range_constexpr(16):
                        dot = dot + q[hq, e] * k_new[e]
                    dot = group_sum(dot, 8)
                    score = bf16_round(bf16_round(dot) * ATTN_SCALE) * LOG2E
                    if g == 0:
                        for e in cutlass.range_constexpr(8):
                            att[COMPUTE_WARPS, hq * HEAD_DIM + d0 + e] = v_new[e]
                            att[COMPUTE_WARPS, hq * HEAD_DIM + d0 + 64 + e] = v_new[8 + e]
                        if j == 0:
                            att[COMPUTE_WARPS, 2 * HEAD_DIM + hq] = score
                            att[COMPUTE_WARPS, 2 * HEAD_DIM + 2 + hq] = Float32(1.0)
        self.csync.sync()
        hq = tid // HEAD_DIM
        d = tid % HEAD_DIM
        # Entry COMPUTE_WARPS (the new token) exists only in the last split.
        mx = Float32(-1e30)
        for e in cutlass.range_constexpr(COMPUTE_WARPS):
            mx = cutlass.max(mx, att[e, 2 * HEAD_DIM + hq])
        if last:
            mx = cutlass.max(mx, att[COMPUTE_WARPS, 2 * HEAD_DIM + hq])
        num = Float32(0.0)
        den = Float32(0.0)
        for e in cutlass.range_constexpr(COMPUTE_WARPS):
            scale = cute.math.exp2(att[e, 2 * HEAD_DIM + hq] - mx)
            num = num + att[e, hq * HEAD_DIM + d] * scale
            den = den + att[e, 2 * HEAD_DIM + 2 + hq] * scale
        if last:
            scale = cute.math.exp2(att[COMPUTE_WARPS, 2 * HEAD_DIM + hq] - mx)
            num = num + att[COMPUTE_WARPS, hq * HEAD_DIM + d] * scale
            den = den + att[COMPUTE_WARPS, 2 * HEAD_DIM + 2 + hq] * scale
        publish(p.attn_o, self.attn_o_index(layer, kvh, split, hq, d), num, epoch)
        if d == 0:
            publish(p.attn_ml, self.attn_ml_index(layer, kvh, hq, split, 0), mx, epoch)
            publish(p.attn_ml, self.attn_ml_index(layer, kvh, hq, split, 1), den, epoch)
        self.csync.sync()  # att and qkvs are rewritten by the next attention row
        return w

    @cute.jit
    def head_vec(self, src, base, d0):
        """This lane's dims d0..d0+7 and 64+d0..64+d0+7 of a head starting at ``base``."""
        lo = load_vec(src, base + d0, 8)
        hi = load_vec(src, base + 64 + d0, 8)
        out = cute.make_rmem_tensor((16,), Float32)
        for e in cutlass.range_constexpr(8):
            out[e] = lo[e]
            out[8 + e] = hi[e]
        return out

    @cute.jit
    def head_norm_rope(self, head, base, weight, cos, sin):
        """HF q/k RMSNorm then rotate-half RoPE of one head in SMEM, in place, by one warp:
        this lane's dims ``base``, ``base + 1`` and their partners ``base + 64``, ``+ 65``."""
        x = cute.make_rmem_tensor((4,), Float32)
        for half in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(2):
                x[2 * half + e] = head[base + 64 * half + e]
        ss = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            ss = ss + x[e] * x[e]
        ss = group_sum(ss, 32)
        rinv = cute.math.rsqrt(ss / HEAD_DIM + EPS)
        for e in cutlass.range_constexpr(2):
            lo = bf16_round(weight[e] * bf16_round(x[e] * rinv))
            hi = bf16_round(weight[2 + e] * bf16_round(x[2 + e] * rinv))
            head[base + e] = bf16_round(bf16_round(lo * cos[e]) + bf16_round(-hi * sin[e]))
            head[base + 64 + e] = bf16_round(
                bf16_round(hi * cos[2 + e]) + bf16_round(lo * sin[2 + e])
            )

    @cute.jit
    def stage_row(self, stage, t, d0):
        """Dims d0..d0+7 and 64+d0..64+d0+7 of position ``t`` of a KV stage, as fp32."""
        out = cute.make_rmem_tensor((16,), Float32)
        for half in cutlass.range_constexpr(2):
            frag = cute.make_rmem_tensor((1, 8), BFloat16)
            cute.autovec_copy(cute.local_tile(stage, (1, 8), (t, d0 // 8 + 8 * half)), frag)
            vals = frag.load().to(Float32)
            for e in cutlass.range_constexpr(8):
                out[8 * half + e] = vals[0, e]
        return out

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(self, tma, gtensors, scratch, tensors, bound):
        layouts = {
            "W": cute.make_layout((ROWS, TILE_K, self.stages), stride=(TILE_K, 1, ROWS * TILE_K)),
            "KV": cute.make_layout(
                (BLOCK, HEAD_DIM, self.stages), stride=(HEAD_DIM, 1, BLOCK * HEAD_DIM)
            ),
            "X": cute.make_layout((2, INTER), stride=(INTER, 1)),
            "RED": cute.make_layout((COMPUTE_WARPS,)),
            "ATT": cute.make_layout(
                (COMPUTE_WARPS + 1, 2 * HEAD_DIM + 4), stride=(2 * HEAD_DIM + 4, 1)
            ),
            "QKVS": cute.make_layout((4 * HEAD_DIM,)),
            "MLS": cute.make_layout((4 * 2 * self.splits,)),
        }
        self.begin_roles(work=bound, layouts=layouts)
        # Materialize the argument tensors here: their iterators/layouts are created lazily on
        # first use, and a first use inside one role's dispatch branch would not dominate the
        # other roles' uses.
        tensors = dict(zip(TENSOR_ARGS, (cute.make_tensor(t.iterator, t.layout) for t in tensors)))
        trace_buffer = tensors.pop("trace_buffer")
        parts = tma_partitions(tma, gtensors, self.buffer("W"), self.buffer("KV"))
        params = MegakernelParams(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            sW=StageView(parts[0][0], axis=1),
            sKV=StageView(parts[5][0], axis=1),
            xs=self.buffer("X"),
            red=self.buffer("RED"),
            att=self.buffer("ATT"),
            qkvs=self.buffer("QKVS"),
            mls=self.buffer("MLS"),
            **tagged_views(scratch, self.scratch_layout),
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
        embed: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        scratch: cute.Tensor,
        logits: cute.Tensor,
        state: cute.Tensor,
        step: cute.Tensor,
        trace_buffer: cute.Tensor,
        ln_in: cute.Tensor,
        ln_post: cute.Tensor,
        q_norm: cute.Tensor,
        k_norm: cute.Tensor,
        final_norm: cute.Tensor,
        rope: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        tma, gtensors = make_tma_atoms((wqkv, wo, wgu, wd, embed), (k_cache, v_cache))
        bound = self.work.bind(table, workspace, max_ctas)
        # In TENSOR_ARGS order.
        tensors = (
            logits,
            state,
            step,
            trace_buffer,
            embed,
            ln_in,
            ln_post,
            q_norm,
            k_norm,
            final_norm,
            rope,
            k_cache,
            v_cache,
        )
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(tma, gtensors, scratch, tensors, bound).launch(
            **self.launch_config(bound), stream=stream, cooperative=True
        )


# Plain-Python helpers for the entry point and prologue: loops in a ``cute.jit`` body are
# staged, so building lists of atoms and views there leaks values out of the loop region.


def make_tma_atoms(weights, caches):
    """TMA atoms and tile tensors: 32 x 256 weight boxes, 64 x 128 KV-cache boxes."""
    op = cpasync.CopyBulkTensorTileG2SOp()
    w_smem = cute.make_layout((ROWS, TILE_K), stride=(TILE_K, 1))
    kv_smem = cute.make_layout((BLOCK, HEAD_DIM), stride=(HEAD_DIM, 1))
    pairs = [cpasync.make_tiled_tma_atom(op, src, w_smem, (ROWS, TILE_K)) for src in weights]
    pairs += [cpasync.make_tiled_tma_atom(op, src, kv_smem, (BLOCK, HEAD_DIM)) for src in caches]
    return tuple(a for a, _ in pairs), tuple(g for _, g in pairs)


def tma_partitions(tma, gtensors, sW, sKV):
    """``(smem, gmem)`` TMA partitions of each atom: ``(TMA, stage)``, ``(TMA, rows, cols)``."""
    parts = []
    for atom, g in zip(tma, gtensors):
        is_kv = len(parts) >= 5
        tile = (BLOCK, HEAD_DIM) if is_kv else (ROWS, TILE_K)
        smem = sKV if is_kv else sW
        tiles = cute.local_tile(g, tile, (None, None))
        parts.append(
            cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(smem, 0, 2),
                cute.group_modes(tiles, 0, 2),
            )
        )
    return parts


def tagged_views(scratch, layout: ScratchLayout):
    """Flat views of each tagged buffer in the int64 scratch."""
    return {
        name: cute.make_tensor(
            scratch.iterator + layout.offsets[name], cute.make_layout((math.prod(shape),))
        )
        for name, shape in layout.shapes.items()
    }


@jit_cache
def compile_megakernel(op: QwenDecodeMegakernel):
    """One compile for every context length and cache capacity (``max_len`` is dynamic)."""
    bf16 = BFloat16
    cache_rows = cute.sym_int(divisibility=BLOCK)
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (LAYERS * QKV_ROWS, HIDDEN)),
        make_fake_compact_tensor(bf16, (LAYERS * HIDDEN, Q_HEADS * HEAD_DIM)),
        make_fake_compact_tensor(bf16, (LAYERS * 2 * INTER, HIDDEN)),
        make_fake_compact_tensor(bf16, (LAYERS * HIDDEN, INTER)),
        make_fake_compact_tensor(bf16, (VOCAB, HIDDEN)),
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(Int64, (op.scratch_layout.size,)),
        make_fake_compact_tensor(Float32, (VOCAB,)),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int64, (cute.sym_int(),)),  # read only by globaltimer builds
        make_fake_compact_tensor(bf16, (LAYERS, HIDDEN)),
        make_fake_compact_tensor(bf16, (LAYERS, HIDDEN)),
        make_fake_compact_tensor(bf16, (LAYERS, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (LAYERS, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (HIDDEN,)),
        make_fake_compact_tensor(bf16, (cute.sym_int(), 2, HEAD_DIM)),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


# ---------------------------------------------------------------------- host side


# Estimated row latencies (us) and dependency latency for the static assignment.
ROW_COST = {QKV: 1.9, ATTN: 3.9, OPROJ: 1.7, GATEUP: 1.9, DOWN: 1.2, LMHEAD: 9.6}
DEPENDENCY_LATENCY = 1.0


# One instruction: its table words (op, layer, a, b), the producer groups it waits for, and the
# group its completion counts toward (``row_groups``).
Row = tuple[list[int], list[tuple], tuple]


def instruction_rows(op: QwenDecodeMegakernel, ops: tuple[int, ...] = ALL_OPS) -> list[Row]:
    """The decode step's instructions in dependency order (``ops`` keeps a subset, for tests of
    single instructions)."""
    rows = []

    def add(kind, layer, a, b):
        if kind in ops:
            rows.append(([kind, layer, a, b], *row_groups(kind, layer, a, b)))

    for layer in range(LAYERS):
        for t in range(QKV_ROWS // ROWS):
            add(QKV, layer, t, 0)
        for kvh in range(KV_HEADS):
            for split in range(op.splits):
                add(ATTN, layer, kvh, split)
        for ks in range(O_SPLITS):
            for t in range(HIDDEN // ROWS):
                add(OPROJ, layer, t, ks)
        for t in range(2 * INTER // ROWS):
            add(GATEUP, layer, t, 0)
        for ks in range(DOWN_SPLITS):
            for t in range(HIDDEN // ROWS):
                add(DOWN, layer, t, ks)
    lm_rows = VOCAB // ROWS
    for t in range(0, lm_rows, LM_TILES):
        add(LMHEAD, 0, t, min(LM_TILES, lm_rows - t))
    return rows


def build_table(
    op: QwenDecodeMegakernel, ctas: int, ops: tuple[int, ...] = ALL_OPS
) -> torch.Tensor:
    """The table of ``instruction_rows``: in dependency order for the atomic schedule; for the
    static one assigned to ``ctas`` CTAs."""
    rows = instruction_rows(op, ops)
    if isinstance(op.work.schedule, StaticPersistent):
        table = assign_static(rows, ctas)
    else:
        table = [words for words, _, _ in rows]
    return torch.tensor([words + [i] for i, words in enumerate(table)], dtype=torch.int32)


def row_groups(kind: int, layer: int, a: int, b: int) -> tuple[list[tuple], tuple]:
    """The producer groups instruction ``(kind, layer, a, b)`` waits for, and the group its
    completion counts toward (a group is done when all of its rows are)."""
    if kind == QKV:
        head = a * ROWS // HEAD_DIM
        kvh = head // 2 if head < Q_HEADS else (head - Q_HEADS) % KV_HEADS
        return [("down", layer - 1)] if layer else [], ("qkv", layer, kvh)
    if kind == ATTN:
        return [("qkv", layer, a)], ("attn", layer, a)
    if kind == OPROJ:
        return [("attn", layer, 2 * b + i) for i in range(2)], ("oproj", layer)
    if kind == GATEUP:
        per_split = 2 * INTER // ROWS // DOWN_SPLITS
        return [("oproj", layer)], ("gateup", layer, a // per_split)
    if kind == DOWN:
        return [("gateup", layer, b)], ("down", layer)
    return [("down", LAYERS - 1)], ("lmhead",)


class RowSpan(NamedTuple):
    """Where and when the ``ROW_COST`` model runs a row (us from the launch), and why then."""

    cta: int
    start: float
    end: float
    gates: tuple[int, ...]  # the last-finishing row of each producer group it waits for
    after: int | None  # the row that set its start: its latest gate, or its CTA's previous row


def list_schedule(rows: list[Row], ctas: int) -> list[RowSpan]:
    """Greedy list scheduling of dependency-ordered rows onto ``ctas`` CTAs with the
    ``ROW_COST`` model: each row goes to the CTA that can start it first, once every producer
    group it waits for has finished plus ``DEPENDENCY_LATENCY``."""
    last: dict[tuple, int] = {}  # producer group -> its last-finishing row so far
    tail: list[int | None] = [None] * ctas  # each CTA's previous row
    free = [0.0] * ctas  # when each CTA's previous row ends
    spans: list[RowSpan] = []
    for i, (words, deps, group) in enumerate(rows):
        gates = tuple(last[d] for d in deps if d in last)
        gate = max(gates, key=lambda g: spans[g].end, default=None)
        ready = spans[gate].end + DEPENDENCY_LATENCY if gate is not None else 0.0
        cta = min(range(ctas), key=lambda c: (max(free[c], ready), c))
        start = max(free[cta], ready)
        after = gate if gate is not None and ready >= free[cta] else tail[cta]
        spans.append(RowSpan(cta, start, start + ROW_COST[words[0]], gates, after))
        tail[cta], free[cta] = i, spans[i].end
        if group not in last or spans[i].end >= spans[last[group]].end:
            last[group] = i
    return spans


def assign_static(rows: list[Row], ctas: int) -> list[list[int]]:
    """``list_schedule``'s per-CTA row lists laid out ``[slot][cta]`` for ``StaticPersistent``
    (CTA b runs rows b, b + ctas, ...); short lists are padded with dropped rows.

    Each CTA's list keeps the dependency order, so with every CTA resident the earliest
    unfinished row always has its producers finished: no deadlock."""
    lists: list[list[list[int]]] = [[] for _ in range(ctas)]
    for (words, _, _), span in zip(rows, list_schedule(rows, ctas)):
        lists[span.cta].append(words)
    slots = max(len(rows) for rows in lists)
    padding = [-1] * (WORDS - 1)
    return [
        lists[cta][slot] if slot < len(lists[cta]) else padding
        for slot in range(slots)
        for cta in range(ctas)
    ]


def rope_table(max_len: int, device) -> torch.Tensor:
    """(max_len, 2, HEAD_DIM) bf16 cos and sin, computed as the HF reference does."""
    inv_freq = 1.0 / (1e6 ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
    freqs = torch.arange(max_len, device=device, dtype=torch.float32)[:, None] * inv_freq
    angles = torch.cat((freqs, freqs), dim=-1)
    return torch.stack((angles.cos(), angles.sin()), dim=1).to(torch.bfloat16).contiguous()


class QwenMegakernel:
    """Packed weights, scratch and the compiled kernel for one cache capacity ``max_len``."""

    def __init__(
        self,
        weights,
        max_len: int,
        ops: tuple[int, ...] = ALL_OPS,
        trace: str | None = None,
        waits: bool = False,
        **config,
    ):
        """``ops``: keep only these instruction kinds (tests). ``trace``: compile with ranges
        for the work items and the ``trace_region``s, recorded by the ``"globaltimer"`` backend
        (``qwen_megakernel_trace.trace_session`` decodes a launch) or ``"iket"`` (for
        ``run-iket``); ``waits`` adds the layer's wait ranges. ``config``: the
        ``QwenDecodeMegakernel`` constructor arguments."""
        if max_len % BLOCK:
            raise ValueError(f"max_len must be a multiple of {BLOCK}, got {max_len}")
        device = weights.embed.device
        self.max_len = max_len
        self.op = QwenDecodeMegakernel(**config)
        self.op.enable_trace(trace, waits=waits, events_per_role=TRACE_EVENTS)
        self.wqkv = torch.cat((weights.q_proj, weights.k_proj, weights.v_proj), dim=1).reshape(
            -1, HIDDEN
        )
        self.wo = weights.o_proj.reshape(-1, Q_HEADS * HEAD_DIM).contiguous()
        # Gate and up rows interleaved: a 32-row tile holds 16 (gate, up) pairs.
        self.wgu = torch.stack((weights.gate, weights.up), dim=2).reshape(-1, HIDDEN)
        self.wd = weights.down.reshape(-1, INTER).contiguous()
        self.weights = weights
        self.rope = rope_table(max_len, device)
        self.scratch = torch.zeros(self.op.scratch_layout.size, device=device, dtype=torch.int64)
        self.logits = torch.zeros(VOCAB, device=device)
        self.state = torch.zeros(2, device=device, dtype=torch.int32)
        self.step_info = torch.zeros(2, device=device, dtype=torch.int32)
        self.workspace, self.max_ctas = self.op.work.launch_args(device)
        self.table = build_table(self.op, self.max_ctas, ops).to(device)
        # The globaltimer profile buffer; trace_session swaps in one for a traced launch.
        self.trace_buffer = torch.zeros(1, device=device, dtype=torch.int64)
        self.fn = compile_megakernel(self.op)

    def tagged(self, name: str) -> torch.Tensor:
        """The int64 words of a tagged buffer, shaped as ``ScratchLayout.shapes``."""
        layout = self.op.scratch_layout
        shape = layout.shapes[name]
        offset = layout.offsets[name]
        return self.scratch[offset : offset + math.prod(shape)].view(shape)

    def values(self, name: str) -> torch.Tensor:
        """A tagged buffer's fp32 values (whatever their tags)."""
        return self.tagged(name).view(torch.int32)[..., 0::2].view(torch.float32)

    def set_values(self, name: str, values: torch.Tensor):
        """Publish ``values`` to a tagged buffer for the next launch (tests feed inputs of
        single instructions this way)."""
        words = self.tagged(name).view(torch.int32)
        words[..., 0::2] = values.float().view(torch.int32)
        words[..., 1::2] = self.state[0] + 1

    def launch(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """One decode step for the token and position in ``step_info``. The caller keeps
        ``0 < pos < max_len`` and the token in the vocabulary (both are read on the device),
        and runs one launch at a time: the scratch and epoch state belong to this runner."""
        shape = (LAYERS, KV_HEADS, self.max_len, HEAD_DIM)
        for cache in (k_cache, v_cache):
            if cache.shape != shape or cache.dtype != torch.bfloat16 or not cache.is_contiguous():
                raise ValueError(f"caches must be contiguous bf16 {shape}, got {cache.shape}")
            if cache.device != self.logits.device or cache.data_ptr() % 16:
                raise ValueError("caches must be 16-byte aligned on the kernel's device (TMA)")
        w = self.weights
        self.fn(
            self.wqkv,
            self.wo,
            self.wgu,
            self.wd,
            w.embed,
            k_cache.view(-1, HEAD_DIM),
            v_cache.view(-1, HEAD_DIM),
            self.scratch,
            self.logits,
            self.state,
            self.step_info,
            self.trace_buffer,
            w.input_ln,
            w.post_ln,
            w.q_norm,
            w.k_norm,
            w.final_norm,
            self.rope,
            self.table,
            self.workspace,
            self.max_ctas,
        )

    def __call__(self, weights, token, pos: int, k_cache, v_cache) -> torch.Tensor:
        """``decode_step_ref``'s signature: writes K/V at ``pos``, returns fp32 logits.
        ``weights`` must be the runner's own: they are packed at construction."""
        if weights is not self.weights:
            raise ValueError("weights must be the ones this QwenMegakernel was built from")
        if not 0 < pos < self.max_len:
            raise ValueError(f"need 0 < pos < max_len = {self.max_len}, got {pos}")
        self.step_info[0] = token
        self.step_info[1] = pos
        self.launch(k_cache, v_cache)
        return self.logits.clone()
