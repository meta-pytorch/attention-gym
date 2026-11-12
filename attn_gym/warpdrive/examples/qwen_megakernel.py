"""Qwen3 batch-one decode step as one persistent warpdrive kernel (a "megakernel").

One file serves the dense Qwen3 family: ``QwenDecodeMegakernel(model=QwenConfig)`` derives
every tiling from the shape and ``check_model`` rejects what they do not cover (0.6B, 1.7B,
8B, 14B and 32B fit; 4B's hidden size does not). The host lowers the step into a table of
instructions, one row per work item (0.6B counts)::

    for layer in 28 layers:
        QKV     128 items   32 rows of [Wq; Wk; Wv] x RMSNorm(h)            -> qkv
        ATTN    8 x S       kv head x KV split: q/k norm, RoPE, flash decode  -> attention partials
        OPROJ   32 x 4      32 rows x 512-column slice of Wo (split K)        -> o partials
        GATEUP  192         16 gate + 16 up rows x RMSNorm(h + o)             -> SiLU(g) * u
        DOWN    32 x 4      32 rows x 768-column slice of Wd (split K)        -> down partials
    LMHEAD  594         8 x 32 rows of the LM head x final RMSNorm            -> logits

An attention row serves two q heads of one kv head (one when the GQA group is odd), so larger
models run several rows per kv head and split; o_proj splits a quarter of the q heads each.

By default the host also assigns rows to CTAs (``list_schedule``, over a latency model) and
lays the table out ``[slot][cta]`` for ``StaticPersistent`` (``assign_static``); the atomic
schedule claims rows in dependency order instead. Each CTA's rows stay in dependency order, so
with every CTA resident (one per SM, guaranteed by the cooperative launch) the earliest
unfinished row can always run.

Cross-CTA dependencies use no counters. Each activation is published as one 64-bit word
``(fp32 value, launch epoch)`` with a relaxed store; a consumer spins on relaxed loads until
every word it needs carries the current epoch (NCCL's LL protocol). The word is single-copy
atomic, so a matching tag means the value is this launch's. One write and one read
propagation per dependency, no fences, no separate flag. The last CTA to finish advances the
epoch for the next launch.

Roles of one CTA (9 warps; 10 with the atomic schedule's scheduler warp)::

    warp 8    load     TMA weight / KV-cache tiles into the ``w`` ring (16 KB stages). It never
                       waits on activations, so it streams the next rows' weights while the
                       compute warps wait for inputs.
    warps 0-7 compute  poll the row's tagged inputs (norms etc. into SMEM) -> consume the row's
                       stages -> publish the outputs

Numerics follow HF's bf16 boundaries: activations hold bf16-rounded values; GEMV accumulation
is fp32, rounded once per output (split-K partials are summed in fp32 and rounded by their
consumer, down's by its last split); attention scores round like the eager matmul; the
softmax merges flash-decoding partials in fp32.

Every per-step intermediate has its own per-layer buffer, so a launch has no write-after-read
hazards. The new K/V row at ``pos`` is written by the one attention row whose TMA-loaded block
holds it (masked), after those loads completed.

The step is latency-bound: about 140 dependent phases, each paying input propagation plus its
row's latency (docs/warpdrive/examples.md). ``QwenMegakernel(trace="globaltimer")``
records every CTA's work items and ``trace_region``s with the table row as payload
(``qwen_megakernel_trace`` decodes them into phase spans and a Perfetto trace);
``trace="iket"`` emits the same ranges for ``run-iket``.
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
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.qwen_megakernel_ref import QwenConfig

# Every size but the head dimension comes from the model (QwenDecodeMegakernel, check_model).
QWEN3_0_6B = QwenConfig()
HEAD_DIM = 128
ATTN_SCALE = HEAD_DIM**-0.5
LOG2E = 1.4426950408889634

# A weight stage is 32 rows x 256 columns, a KV stage 64 positions x 128 dims; both 16 KB.
ROWS, TILE_K, BLOCK = 32, 256, 64
STAGE_BYTES = ROWS * TILE_K * 2
O_SPLITS, DOWN_SPLITS = 4, 4  # split-K of o_proj (a quarter of the q heads each) and down
LM_TILES = 8  # row tiles per LM-head item
MAX_STAGES = 12
# Default KV splits. GB200, CUDA-graph medians in us at ctx 1024 / 8192, splits 4, 8, 16:
# 8B (16 attention row groups) 2916 / 3759, 2937 / 3347, 3033 / 3417; 14B (40 groups)
# 4624 / 5843, 4750 / 5613, 5111 / 5787; 32B (32 groups) 10435 / 13180, 10654 / 12035,
# 11496 / 12509. Fewer splits win short contexts, 8 balances both.
ATTN_ROWS, MIN_SPLITS = 128, 8
SMEM_BYTES = 232448  # SM100 per-block shared memory (227 KB)
SMEM_HEADROOM = 1024
FETCH_ELEMENTS = 8  # residual elements per thread fetched in one poll (bounds registers)
COMPUTE_WARPS = 8
ROWS_PER_WARP = ROWS // COMPUTE_WARPS
COMPUTE_THREADS = 32 * COMPUTE_WARPS

QKV, ATTN, OPROJ, GATEUP, DOWN, LMHEAD = range(6)
ALL_OPS = (QKV, ATTN, OPROJ, GATEUP, DOWN, LMHEAD)
EVICT_FIRST = 0x12F0000000000000  # TMA L2 cache hint (CUTLASS CacheHintSm90::EVICT_FIRST)
WORDS = 5  # op, layer, a, b, table row
OP_NAMES = ("QKV", "ATTN", "OPROJ", "GATEUP", "DOWN", "LMHEAD")
# Globaltimer records per role and CTA: at least TRACE_EVENTS (0.6B: a CTA runs ~140 rows),
# else TRACE_EVENTS_PER_ROW per row of the CTA's share (a row records its work item and regions).
TRACE_EVENTS, TRACE_EVENTS_PER_ROW = 2048, 8


def qkv_rows(model: QwenConfig) -> int:
    """Rows of the fused [Wq; Wk; Wv]."""
    return (model.num_q_heads + 2 * model.num_kv_heads) * HEAD_DIM


def attention_groups(model: QwenConfig) -> tuple[int, int]:
    """``(groups, per_row)``: an attention row serves ``per_row`` q heads of one kv head, 2
    when the GQA group size is even (the kernel's register budget fits no more), else 1; a
    layer has ``groups`` such rows per KV split. Each reads its kv head's KV blocks."""
    per_row = 2 if model.num_q_heads // model.num_kv_heads % 2 == 0 else 1
    return model.num_q_heads // per_row, per_row


class ScratchLayout:
    """Offsets (in 64-bit words) of the tagged per-step intermediates."""

    def __init__(self, model: QwenConfig, splits: int):
        layers, hidden = model.num_layers, model.hidden_size
        groups, per_row = attention_groups(model)
        self.shapes = {
            "h_in": (layers + 1, hidden),  # layer inputs (residual); [layers]: LM head's
            "qkv": (layers, qkv_rows(model)),
            # Per attention row group (per_row q heads of one kv head) and KV split.
            "attn_o": (layers, groups, splits, per_row, HEAD_DIM),  # unnormalized partial O
            "attn_ml": (layers, groups, per_row, splits, 2),  # (m, l) per head, split; log2 m
            "o_part": (layers, O_SPLITS, hidden),
            "h_mid": (layers, hidden),  # residual after attention
            "act": (layers, model.intermediate_size),
            "d_part": (layers, DOWN_SPLITS, hidden),
        }
        self.offsets, offset = {}, 0
        for name, shape in self.shapes.items():
            self.offsets[name] = offset
            offset += (math.prod(shape) + 15) // 16 * 16  # 128-byte aligned
        self.size = offset


@dataclass(frozen=True)
class InstructionTable(WorkMap):
    """Row ``index`` of an int32 ``(rows, WORDS)`` table. A negative op marks a padding row,
    dropped for every role."""

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
    tma: tuple  # TMA atoms: wqkv, wo, wgu, wd, lm head, k cache, v cache
    gmem: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    sW: StageView  # weight stage partition
    sKV: StageView  # KV stage partition
    xs: cute.Tensor  # (2, x_width) bf16 GEMV inputs, double-buffered by item
    red: cute.Tensor  # (COMPUTE_WARPS,) fp32 block reduction
    att: cute.Tensor  # (9, att_stride) fp32 per-warp attention states
    qkvs: cute.Tensor  # ((per_row + 2) * HEAD_DIM,) fp32 an attention row's q heads, k and v
    mls: cute.Tensor  # (heads_per_split, splits, 2) fp32 an o_proj row's (m, l) per KV split
    # Tagged int64 activations, flat row-major (ScratchLayout.shapes; see the *_index helpers).
    h_in: cute.Tensor
    qkv: cute.Tensor
    attn_o: cute.Tensor
    attn_ml: cute.Tensor
    o_part: cute.Tensor
    h_mid: cute.Tensor
    act: cute.Tensor
    d_part: cute.Tensor
    logits: cute.Tensor  # (vocab,) fp32
    state: cute.Tensor  # int32 [epoch of the previous launch, CTAs finished]
    step: cute.Tensor  # int32 [token, pos]
    embed: cute.Tensor  # (vocab, hidden) bf16
    ln_in: cute.Tensor  # (layers, hidden) bf16
    ln_post: cute.Tensor
    q_norm: cute.Tensor  # (layers, HEAD_DIM) bf16
    k_norm: cute.Tensor
    final_norm: cute.Tensor  # (hidden,) bf16
    rope: cute.Tensor  # (max_len, 2, HEAD_DIM) bf16 cos, sin
    k_cache: cute.Tensor  # (layers * kv_heads * max_len, HEAD_DIM) bf16
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
    """``count`` tagged words from ``buf[index]`` as relaxed gpu-scope loads: 32 B vectors, then
    a 16 B vector, then a scalar. Each strong load is one L2 request per lane, so wider is
    cheaper. ``index`` must be a multiple of 4 when ``count >= 4`` and even when ``count >= 2``."""
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
    """Spin until every word of ``segments`` carries ``epoch``; the values fill ``out`` in order.
    A segment is a ``(buffer, start, count)`` run aligned as ``load_words`` requires."""
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


# ------------------------------------------------------------------ register math


@cute.jit
def bf16_round(x):
    """Round to the nearest bf16 (ties to even), kept in fp32. Integer ops instead of F2F: the
    conversion is quarter-rate and the norms round every element."""
    bits = Int32(bitcast(Float32(x).ir_value(), T.i32()))
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & Int32(-65536)
    return Float32(bitcast(bits.ir_value(), T.f32()))


@cute.jit
def warp_transpose_reduce(acc, lane, n: cutlass.Constexpr):
    """Warp-sum each of ``n`` (a power of two <= 32) per-lane values; lane l ends with the sum
    of value ``l // (32 // n)``. Halving exchanges, then one full reduction: 6 shuffles for
    n = 4 instead of 20."""
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
    """``n`` contiguous elements of a 1-D tensor from ``offset`` (a multiple of n), as fp32."""
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
        stages: int | None = None,
        splits: int | None = None,
        schedule: str = "static",
        roles: dict[str, RoleConfig] | None = None,
        model: QwenConfig = QWEN3_0_6B,
    ):
        """``stages``: weight/KV ring depth in 16 KB stages; by default the most that fit in
        shared memory, up to ``MAX_STAGES``.
        ``splits``: KV splits per attention row group (``attention_groups``: 8 for 0.6B,
        so 8 x splits attention rows per layer); by default about ``ATTN_ROWS`` rows per layer,
        at least ``MIN_SPLITS`` (0.6B and 1.7B: 16, larger models: 8). It is static
        while the position is a runtime value, so pick it for the contexts a runner decodes,
        not the capacity. On a 152-SM GB300, 19 (one 0.6B attention row per SM) is faster than
        16 at long contexts and slower at short ones.
        ``schedule``: ``"static"`` runs a host-assigned per-CTA row list (``StaticPersistent``
        over a ``[slot][cta]`` table; needs every CTA resident), ``"atomic"`` claims rows in
        dependency order (``AtomicPersistent``, one row decoded ahead).
        ``roles``: placement by role name over the ``@role`` defaults.
        ``model``: the Qwen3 shape (default Qwen3-0.6B); see ``check_model``."""
        check_model(model)
        self.model = model
        self.layers, self.hidden = model.num_layers, model.hidden_size
        self.inter, self.vocab = model.intermediate_size, model.vocab_size
        self.q_heads, self.kv_heads = model.num_q_heads, model.num_kv_heads
        self.qkv_rows = qkv_rows(model)
        self.attn_groups, self.per_row = attention_groups(model)
        self.group_size = self.q_heads // self.kv_heads  # q heads per kv head
        self.heads_per_split = self.q_heads // O_SPLITS  # q heads in one o_proj split
        # Weight stages per 32-row tile: QKV, GATEUP and LMHEAD read all hidden columns.
        self.hidden_k_tiles = self.hidden // TILE_K
        self.oproj_k_tiles = self.heads_per_split * HEAD_DIM // TILE_K
        self.down_k_tiles = self.inter // TILE_K // DOWN_SPLITS
        # RMSNorm and the residual sums hold the hidden vector this many elements per thread.
        self.per_thread = self.hidden // COMPUTE_THREADS
        # The longest GEMV input: the hidden vector, a down split or an o_proj split.
        self.x_width = max(self.hidden, self.inter // DOWN_SPLITS, self.heads_per_split * HEAD_DIM)
        if schedule not in ("static", "atomic"):
            raise ValueError(f"schedule must be 'static' or 'atomic', got {schedule!r}")
        # The GEMV (ROWS_PER_WARP rows per warp), RMSNorm (per_thread elements per thread) and
        # attention (8 positions per warp per block) tilings fix the compute role's size.
        compute = (roles or {}).get("compute")
        if compute is not None and [len(r) for r in compute.warps] != [COMPUTE_WARPS]:
            raise ProtocolError(
                f"roles['compute'] must be one instance of {COMPUTE_WARPS} warps, got "
                f"{compute.warps}"
            )
        self.roles = roles
        if splits is None:
            splits = max(MIN_SPLITS, ATTN_ROWS // self.attn_groups)
        if not 1 <= splits <= 32:
            raise ValueError(f"splits must be in [1, 32], got {splits}")
        self.splits = splits
        fit = (SMEM_BYTES - self.fixed_smem_bytes()) // STAGE_BYTES
        if stages is None:
            stages = min(MAX_STAGES, fit)
        if not 2 <= stages <= fit:
            raise ValueError(f"stages must be in [2, {fit}] for this model, got {stages}")
        self.stages = stages
        self.scratch_layout = ScratchLayout(model, splits)
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        kv = self.smem_plan.buffer("KV", BFloat16, align=1024, overlaps=w)
        self.smem_plan.buffer("X", BFloat16, align=16)  # inputs are bf16-rounded: lossless
        self.smem_plan.buffer("RED", Float32, align=16)
        self.smem_plan.buffer("ATT", Float32, align=16)
        self.smem_plan.buffer("QKVS", Float32, align=16)
        self.smem_plan.buffer("MLS", Float32, align=16)
        # Compute reads stages with generic loads before TMA refills them. For a Tma-filled ring
        # released by Warps the layer emits the fence.proxy.async on release.
        self.w = Ring(stages, Tma(STAGE_BYTES), Warps, buffer={"W": w, "KV": kv})
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.work = WorkStream(
            mapping=InstructionTable(),
            schedule=StaticPersistent() if schedule == "static" else AtomicPersistent(stages=1),
        )

    def fixed_smem_bytes(self) -> int:
        """Shared memory besides the weight ring, with headroom for barriers and alignment."""
        floats = (
            COMPUTE_WARPS  # RED
            + (COMPUTE_WARPS + 1) * self.att_stride  # ATT
            + (self.per_row + 2) * HEAD_DIM  # QKVS
            + self.heads_per_split * self.splits * 2  # MLS
        )
        return 2 * 2 * self.x_width + 4 * floats + SMEM_HEADROOM

    @property
    def att_stride(self) -> int:
        """Floats per warp state in ATT: O of each q head, then their m, then their l."""
        return self.per_row * (HEAD_DIM + 2)

    def get_name(self) -> str:
        return (
            f"warpdrive_qwen_decode_h{self.hidden}_s{self.stages}_kv{self.splits}_"
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
        max_len = p.k_cache.shape[0] // (self.layers * self.kv_heads)
        while work.valid:
            op, layer, a, b, _row = work.item
            if op == ATTN:
                blk0, blk1 = self.split_blocks(pos, b)
                row0 = (layer * self.kv_heads + self.kv_head(a)) * (max_len // BLOCK)
                for blk in cutlass.range(blk0, blk1, unroll=1):
                    w = self.load_tiles(w, tma_k, g_k, row0 + blk, 0, 1, "KV")
                    w = self.load_tiles(w, tma_v, g_v, row0 + blk, 0, 1, "KV")
            elif op == QKV:
                tile = layer * (self.qkv_rows // ROWS) + a
                w = self.load_tiles(w, tma_qkv, g_qkv, tile, 0, self.hidden_k_tiles, "W")
            elif op == OPROJ:
                tile = layer * (self.hidden // ROWS) + a
                k_tiles = self.oproj_k_tiles
                w = self.load_tiles(w, tma_o, g_o, tile, b * k_tiles, k_tiles, "W")
            elif op == GATEUP:
                tile = layer * (2 * self.inter // ROWS) + a
                w = self.load_tiles(w, tma_gu, g_gu, tile, 0, self.hidden_k_tiles, "W")
            elif op == DOWN:
                tile = layer * (self.hidden // ROWS) + a
                k_tiles = self.down_k_tiles
                w = self.load_tiles(w, tma_d, g_d, tile, b * k_tiles, k_tiles, "W")
            else:
                for rt in cutlass.range(a, a + b, unroll=1):
                    w = self.load_tiles(w, tma_lm, g_lm, rt, 0, self.hidden_k_tiles, "W")
            work.advance()
        return w, work

    @cute.jit
    def load_tiles(
        self, w, atom, tiles, row, k0, count: cutlass.Constexpr, buffer: cutlass.Constexpr
    ):
        """``count`` stages of tiles ``(row, k0 + k)``. Weights and KV rows are read once per
        step, hence evict-first in L2."""
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
        # Consecutive rows with the same key share the GEMV input in xs[xbuf]. Row 0 of GATEUP
        # and of layer 0's QKV always runs the prologue: it publishes the residual.
        xkey = Int32(-1)
        xbuf = Int32(0)
        while work.valid:
            op, layer, a, b, row = work.item
            if op == ATTN:
                w = self.attention(p, w, layer, a, b, pos, epoch, tid, warp, lane, row)
            else:
                ks = cutlass.select_((op == OPROJ) | (op == DOWN), b, 0)
                key = (op * (self.layers + 1) + layer) * 8 + ks
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
        # Every CTA has read the epoch by the time it finishes; the last one advances it.
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
        """Fetch the row's inputs and write its GEMV input to ``xs[xbuf]``; the caller syncs.
        The norm weight loads before the poll; the ``poll`` region ends when the inputs are
        ready."""
        xs = p.xs[xbuf, None]
        n = self.per_thread
        norm_weight = cute.make_rmem_tensor((n,), Float32)
        if op == QKV:
            load_into(norm_weight, p.ln_in[layer, None], n * tid, n)
        elif op == GATEUP:
            load_into(norm_weight, p.ln_post[layer, None], n * tid, n)
        elif op == LMHEAD:
            load_into(norm_weight, p.final_norm, n * tid, n)
        x = cute.make_rmem_tensor((n,), Float32)
        if op == QKV:
            if layer == 0:
                load_into(x, p.embed[token, None], n * tid, n)
                if a == 0:
                    for e in cutlass.range_constexpr(n):
                        publish(p.h_in, self.hidden_index(0, n * tid + e), x[e], epoch)
            else:
                self.published_residual(x, p.h_in, layer, epoch, tid, row)
            self.rms_norm(p, x, norm_weight, xs, tid, warp, lane)
        elif op == GATEUP:
            self.residual(x, p.h_in, p.o_part, layer, epoch, tid, row)
            if a == 0:
                for e in cutlass.range_constexpr(n):
                    publish(p.h_mid, self.hidden_index(layer, n * tid + e), x[e], epoch)
            self.rms_norm(p, x, norm_weight, xs, tid, warp, lane)
        elif op == LMHEAD:
            self.published_residual(x, p.h_in, layer, epoch, tid, row)  # layer == layers
            self.rms_norm(p, x, norm_weight, xs, tid, warp, lane)
        elif op == DOWN:
            span = self.inter // DOWN_SPLITS
            count = span // COMPUTE_THREADS
            vals = cute.make_rmem_tensor((count,), Float32)
            segments = []
            for i in cutlass.range_constexpr(count):
                index = self.act_index(layer, ks * span + i * COMPUTE_THREADS + tid)
                segments.append((p.act, index, 1))
            with self.trace_region("poll", payload=row):
                fetch(segments, vals, epoch)
            for i in cutlass.range_constexpr(count):
                xs[i * COMPUTE_THREADS + tid] = BFloat16(vals[i])
        else:
            self.combine_attention(p, layer, ks, epoch, xs, tid, row)

    def kv_head(self, group):
        """The kv head of attention row group ``group`` (q heads ``group * per_row``, ...)."""
        return group * self.per_row // self.group_size

    def qkv_index(self, layer, i):
        return layer * self.qkv_rows + i

    def hidden_index(self, layer, i):
        """h_in, h_mid."""
        return layer * self.hidden + i

    def part_index(self, layer, split, i):
        """o_part, d_part."""
        return (layer * O_SPLITS + split) * self.hidden + i

    def act_index(self, layer, i):
        return layer * self.inter + i

    @cute.jit
    def residual(self, x, base, parts, layer, epoch, tid, row):
        """x = bf16(base + bf16(sum of split-K partials)) for this thread's elements, polled
        ``FETCH_ELEMENTS`` at a time: all of them at once would spill past 8 per thread."""
        splits = O_SPLITS
        n = self.per_thread
        with self.trace_region("poll", payload=row):
            for c0 in cutlass.range_constexpr(0, n, FETCH_ELEMENTS):
                chunk = min(FETCH_ELEMENTS, n - c0)
                vals = cute.make_rmem_tensor((chunk * (splits + 1),), Float32)
                segments = [(base, self.hidden_index(layer, n * tid + c0), chunk)]
                for k in cutlass.range_constexpr(splits):
                    segments.append((parts, self.part_index(layer, k, n * tid + c0), chunk))
                fetch(segments, vals, epoch)
                for e in cutlass.range_constexpr(chunk):
                    total = vals[chunk + e]
                    for k in cutlass.range_constexpr(1, splits):
                        total = total + vals[chunk * (k + 1) + e]
                    x[c0 + e] = bf16_round(vals[e] + bf16_round(total))

    @cute.jit
    def published_residual(self, x, buf, layer, epoch, tid, row):
        """This thread's elements of the finished residual ``buf[layer]``."""
        n = self.per_thread
        with self.trace_region("poll", payload=row):
            fetch([(buf, self.hidden_index(layer, n * tid), n)], x, epoch)

    @cute.jit
    def reduce_down(self, p: MegakernelParams, layer, col, epoch, row):
        """Publish the next layer's residual column ``col`` = bf16(h_mid + bf16(sum of the
        down partials)), summed in split order like ``residual``.

        The last down split runs it after publishing its own partial: that split finishes
        last (its GATEUP rows are the second wave), and QKV and the LM head then fetch 1 word
        per element instead of 5."""
        vals = cute.make_rmem_tensor((DOWN_SPLITS + 1,), Float32)
        segments = [(p.h_mid, self.hidden_index(layer, col), 1)]
        for k in cutlass.range_constexpr(DOWN_SPLITS):
            segments.append((p.d_part, self.part_index(layer, k, col), 1))
        with self.trace_region("reduce.poll", payload=row):
            fetch(segments, vals, epoch)
        total = vals[1]
        for k in cutlass.range_constexpr(2, DOWN_SPLITS + 1):
            total = total + vals[k]
        h = bf16_round(vals[0] + bf16_round(total))
        publish(p.h_in, self.hidden_index(layer + 1, col), h, epoch)

    @cute.jit
    def rms_norm(self, p: MegakernelParams, x, wv, xs, tid, warp, lane):
        """HF RMSNorm of the hidden vector held ``per_thread`` per thread:
        bf16(w * bf16(x * rsqrt)), to xs."""
        n = self.per_thread
        ss = Float32(0.0)
        for e in cutlass.range_constexpr(n):
            ss = ss + x[e] * x[e]
        ss = group_sum(ss, 32)
        if lane == 0:
            p.red[warp] = ss
        self.csync.sync()
        total = Float32(0.0)
        for i in cutlass.range_constexpr(COMPUTE_WARPS):
            total = total + p.red[i]
        rinv = cute.math.rsqrt(total / self.hidden + self.model.rms_eps)
        for e in cutlass.range_constexpr(n):
            x[e] = bf16_round(wv[e] * bf16_round(x[e] * rinv))
        store_vec(xs, n * tid, x, n)

    def attn_o_index(self, layer, group, split, head, d):
        """``head``: within the attention row group."""
        index = (layer * self.attn_groups + group) * self.splits + split
        return (index * self.per_row + head) * HEAD_DIM + d

    def attn_ml_index(self, layer, group, head, split, k):
        """k = 0: m (log2 domain), 1: l. Consecutive q heads' words are contiguous."""
        index = (layer * self.attn_groups + group) * self.per_row + head
        return (index * self.splits + split) * 2 + k

    @cute.jit
    def combine_attention(self, p: MegakernelParams, layer, ks, epoch, xs, tid, row):
        """o_proj input slice ``ks``: its ``heads_per_split`` q heads merged from their KV-split
        partials.

        Pass c has thread t combine dims 2j, 2j + 1 of the slice, j = t + 256c. The slice's
        (m, l) words are contiguous; the first pass's poll also fetches them into SMEM."""
        pairs = 2 * self.splits  # (m, l) words per head
        ml_words = self.heads_per_split * pairs
        ml_base = self.attn_ml_index(layer, 0, 0, 0, 0) + ks * ml_words
        ml_fetches = (ml_words + COMPUTE_THREADS - 1) // COMPUTE_THREADS
        dim_pairs = self.heads_per_split * HEAD_DIM // 2
        for c in cutlass.range_constexpr((dim_pairs + COMPUTE_THREADS - 1) // COMPUTE_THREADS):
            j = cutlass.min(tid + c * COMPUTE_THREADS, dim_pairs - 1)
            hh = j // (HEAD_DIM // 2)  # head within the slice
            d = (j % (HEAD_DIM // 2)) * 2
            q_head = ks * self.heads_per_split + hh
            group = q_head // self.per_row
            head = q_head % self.per_row
            segments = []
            if cutlass.const_expr(c == 0):
                for i in cutlass.range_constexpr(ml_fetches):
                    word = cutlass.min(tid + i * COMPUTE_THREADS, ml_words - 1)
                    segments.append((p.attn_ml, ml_base + word, 1))
            first = len(segments)
            for sp in cutlass.range_constexpr(self.splits):
                segments.append((p.attn_o, self.attn_o_index(layer, group, sp, head, d), 2))
            vals = cute.make_rmem_tensor((first + 2 * self.splits,), Float32)
            with self.trace_region("poll", payload=row):
                fetch(segments, vals, epoch)
            if cutlass.const_expr(c == 0):
                for i in cutlass.range_constexpr(ml_fetches):
                    if tid + i * COMPUTE_THREADS < ml_words:
                        p.mls[tid + i * COMPUTE_THREADS] = vals[i]
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
                num0 = num0 + vals[first + 2 * sp] * scale
                num1 = num1 + vals[first + 1 + 2 * sp] * scale
            if tid + c * COMPUTE_THREADS < dim_pairs:
                xs[hh * HEAD_DIM + d] = BFloat16(num0 / den)
                xs[hh * HEAD_DIM + d + 1] = BFloat16(num1 / den)

    @cute.jit
    def gemv(self, p: MegakernelParams, w, op, layer, a, b, epoch, xbuf, warp, lane, row_index):
        """GEMV of 32-row tiles: warp w owns rows 4w..4w+3 of a stage, lane l columns 8l..8l+7."""
        k_tiles = self.hidden_k_tiles
        if op == OPROJ:
            k_tiles = self.oproj_k_tiles
        if op == DOWN:
            k_tiles = self.down_k_tiles
        row_tiles = cutlass.select_(op == LMHEAD, b, 1)
        for t in cutlass.range(row_tiles, unroll=1):
            # Even and odd columns accumulate separately, one packed FFMA2 per pair.
            acc = cute.make_rmem_tensor((ROWS_PER_WARP, 2), Float32)
            acc.fill(0.0)
            with self.trace_region("stream", payload=row_index):
                # Fully unrolled: the rolled loop kept acc in a 32-byte stack frame.
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
                # Tile rows alternate gate, up: lanes 0-7 hold gate j, lanes 8-15 up j.
                up = cute.arch.shuffle_sync_down(v, offset=8)
                if lane % 16 == 0:
                    g = bf16_round(v)
                    silu = bf16_round(g / (1.0 + cute.math.exp2(-g * LOG2E)))
                    out = bf16_round(silu * bf16_round(up))
                    publish(p.act, self.act_index(layer, a * (ROWS // 2) + row // 2), out, epoch)
            elif lane % (32 // ROWS_PER_WARP) == 0:
                if op == QKV:
                    publish(p.qkv, self.qkv_index(layer, a * ROWS + row), bf16_round(v), epoch)
                elif op == OPROJ:
                    publish(p.o_part, self.part_index(layer, b, a * ROWS + row), v, epoch)
                elif op == DOWN:
                    publish(p.d_part, self.part_index(layer, b, a * ROWS + row), v, epoch)
                else:
                    p.logits[(a + t) * ROWS + row] = bf16_round(v)
        return w

    @cute.jit
    def attention(
        self, p: MegakernelParams, w, layer, group, split, pos, epoch, tid, warp, lane, row_index
    ):
        """Flash-decoding partial of attention row group ``group`` (``per_row`` q heads of one
        kv head) over one KV split.

        The new token's q heads, k and v are fetched once into SMEM. Warp w scores
        positions 8w + 4i + g (i = 0, 1) of each 64-position block: lane group g = lane // 8
        holds a position, lane j = lane % 8 its dims 8j..8j+7 and 64+8j..64+8j+7 (the
        rotate-half pairs). Each warp keeps its own online-softmax state; the warps, plus the
        new token in the last split, merge through SMEM at the end."""
        g = lane // 8
        j = lane % 8
        d0 = 8 * j
        heads = self.per_row
        kvh = self.kv_head(group)
        max_len = p.k_cache.shape[0] // (self.layers * self.kv_heads)
        # Warps 0 .. heads - 1 norm and rotate the q heads, warp ``heads`` k: lane l owns dims
        # 2l, 2l + 1 and their rotate-half partners 64 + 2l, 65 + 2l. Norm weights and RoPE
        # depend on no other row, so they load before the fetch.
        norm = cute.make_rmem_tensor((4,), Float32)
        cos = cute.make_rmem_tensor((4,), Float32)
        sin = cute.make_rmem_tensor((4,), Float32)
        norm_weight = p.q_norm[layer, None]
        if warp == heads:
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
        # Thread t fetches words 2t, 2t + 1 of [the group's q heads | k kvh | v kvh]. With one
        # q head there are 384 words: threads past them refetch the last pair but do not store.
        q_words = heads * HEAD_DIM
        word = cutlass.min(2 * tid, q_words + 2 * HEAD_DIM - 2)
        column = group * q_words + word
        k_column = (self.q_heads + kvh) * HEAD_DIM + word - q_words
        column = cutlass.select_(word >= q_words, k_column, column)
        v_column = k_column + (self.kv_heads - 1) * HEAD_DIM
        column = cutlass.select_(word >= q_words + HEAD_DIM, v_column, column)
        raw = cute.make_rmem_tensor((2,), Float32)
        with self.trace_region("poll", payload=row_index):
            fetch([(p.qkv, self.qkv_index(layer, column), 2)], raw, epoch)
        if cutlass.const_expr(q_words + 2 * HEAD_DIM < 2 * COMPUTE_THREADS):
            if 2 * tid == word:
                p.qkvs[word] = raw[0]
                p.qkvs[word + 1] = raw[1]
        else:
            p.qkvs[word] = raw[0]
            p.qkvs[word + 1] = raw[1]
        self.csync.sync()
        if warp < heads + 1:
            self.head_norm_rope(p.qkvs, warp * HEAD_DIM + 2 * lane, norm, cos, sin)
        self.csync.sync()
        q = cute.make_rmem_tensor((heads, 16), Float32)
        for hq in cutlass.range_constexpr(heads):
            qh = self.head_vec(p.qkvs, hq * HEAD_DIM, d0)
            for e in cutlass.range_constexpr(16):
                q[hq, e] = qh[e]
        k_new = self.head_vec(p.qkvs, heads * HEAD_DIM, d0)
        v_new = self.head_vec(p.qkvs, (heads + 1) * HEAD_DIM, d0)
        with self.trace_region("attn.blocks", payload=row_index):
            m = cute.make_rmem_tensor((heads,), Float32)  # running max (log2 domain), warp-uniform
            lsum = cute.make_rmem_tensor((heads,), Float32)  # this lane group's sum of p
            o = cute.make_rmem_tensor((heads, 16), Float32)  # this lane group's sum of p * v
            m.fill(-1e30)
            lsum.fill(0.0)
            o.fill(0.0)
            blk0, blk1 = self.split_blocks(pos, split)
            for blk in cutlass.range(blk0, blk1, unroll=1):
                sc = cute.make_rmem_tensor((heads, 2), Float32)  # (head, i)
                with w.consume() as h:
                    sK = h.data["KV"]
                    for i in cutlass.range_constexpr(2):
                        t = 8 * warp + 4 * i + g
                        kv = self.stage_row(sK, t, d0)
                        valid = blk * BLOCK + t < pos
                        for hq in cutlass.range_constexpr(heads):
                            dot = Float32(0.0)
                            for e in cutlass.range_constexpr(16):
                                dot = dot + q[hq, e] * kv[e]
                            dot = group_sum(dot, 8)
                            score = bf16_round(bf16_round(dot) * ATTN_SCALE) * LOG2E
                            sc[hq, i] = cutlass.select_(valid, score, -Float32.inf)
                pr = cute.make_rmem_tensor((heads, 2), Float32)
                for hq in cutlass.range_constexpr(heads):
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
                        if blk * BLOCK + t < pos:  # masked: p is 0 but stale v may be NaN
                            vv = self.stage_row(sV, t, d0)
                            for hq in cutlass.range_constexpr(heads):
                                for e in cutlass.range_constexpr(16):
                                    o[hq, e] = o[hq, e] + pr[hq, i] * vv[e]
        # Append the new K/V row at pos. Only the split whose KV block holds pos TMA-loads that
        # row (masked above), and it stores after its own loads completed, so no async-proxy
        # read races the generic store. When pos starts a new block, no split reads it.
        blocks, per = self.kv_blocks(pos)
        writer = cutlass.select_(pos // BLOCK < blocks, pos // BLOCK // per, self.splits - 1)
        # Of the kv head's groups, the first writes. The others read the block too but mask
        # pos (K scores select -inf, V is skipped), so the racing store's value never matters.
        if (split == writer) & (group * heads % self.group_size == 0):
            if tid < 8:  # warp 0, lane group 0
                row = (layer * self.kv_heads + kvh) * max_len + pos
                for half in cutlass.range_constexpr(2):
                    kf = cute.make_rmem_tensor((8,), Float32)
                    vf = cute.make_rmem_tensor((8,), Float32)
                    for e in cutlass.range_constexpr(8):
                        kf[e] = k_new[half * 8 + e]
                        vf[e] = v_new[half * 8 + e]
                    store_vec(p.k_cache[row, None], d0 + 64 * half, kf, 8)
                    store_vec(p.v_cache[row, None], d0 + 64 * half, vf, 8)
        # Merge the four lane groups (they share m), then publish this warp's state to SMEM.
        for hq in cutlass.range_constexpr(heads):
            lsum[hq] = lsum[hq] + cute.arch.shuffle_sync_bfly(lsum[hq], offset=8)
            lsum[hq] = lsum[hq] + cute.arch.shuffle_sync_bfly(lsum[hq], offset=16)
            for e in cutlass.range_constexpr(16):
                o[hq, e] = o[hq, e] + cute.arch.shuffle_sync_bfly(o[hq, e], offset=8)
                o[hq, e] = o[hq, e] + cute.arch.shuffle_sync_bfly(o[hq, e], offset=16)
        att = p.att
        if g == 0:
            for hq in cutlass.range_constexpr(heads):
                for e in cutlass.range_constexpr(8):
                    att[warp, hq * HEAD_DIM + d0 + e] = o[hq, e]
                    att[warp, hq * HEAD_DIM + d0 + 64 + e] = o[hq, 8 + e]
            if j == 0:
                for hq in cutlass.range_constexpr(heads):
                    att[warp, heads * HEAD_DIM + hq] = m[hq]
                    att[warp, heads * HEAD_DIM + heads + hq] = lsum[hq]
        last = split == self.splits - 1
        if last:
            if warp == 0:  # entry COMPUTE_WARPS is the new token: m = its score, l = 1, O = v
                for hq in cutlass.range_constexpr(heads):
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
                            att[COMPUTE_WARPS, heads * HEAD_DIM + hq] = score
                            att[COMPUTE_WARPS, heads * HEAD_DIM + heads + hq] = Float32(1.0)
        self.csync.sync()
        # Thread t merges dim t % HEAD_DIM of q head t // HEAD_DIM; with one q head, threads
        # HEAD_DIM.. sit out (their head's ATT columns do not exist).
        if cutlass.const_expr(heads * HEAD_DIM < COMPUTE_THREADS):
            if tid < heads * HEAD_DIM:
                self.merge_partials(p, layer, group, split, last, tid, epoch)
        else:
            self.merge_partials(p, layer, group, split, last, tid, epoch)
        self.csync.sync()  # the next attention row rewrites att and qkvs
        return w

    @cute.jit
    def merge_partials(self, p: MegakernelParams, layer, group, split, last, tid, epoch):
        """Merge the warps' online-softmax states in ATT (plus the new token's in the last split)
        into this split's partial O, m and l for dim ``tid % HEAD_DIM`` of q head
        ``tid // HEAD_DIM``, and publish them."""
        heads = self.per_row
        att = p.att
        hq = tid // HEAD_DIM
        d = tid % HEAD_DIM
        # Entry COMPUTE_WARPS (the new token) is written only in the last split.
        mx = Float32(-1e30)
        for e in cutlass.range_constexpr(COMPUTE_WARPS):
            mx = cutlass.max(mx, att[e, heads * HEAD_DIM + hq])
        if last:
            mx = cutlass.max(mx, att[COMPUTE_WARPS, heads * HEAD_DIM + hq])
        num = Float32(0.0)
        den = Float32(0.0)
        for e in cutlass.range_constexpr(COMPUTE_WARPS):
            scale = cute.math.exp2(att[e, heads * HEAD_DIM + hq] - mx)
            num = num + att[e, hq * HEAD_DIM + d] * scale
            den = den + att[e, heads * HEAD_DIM + heads + hq] * scale
        if last:
            scale = cute.math.exp2(att[COMPUTE_WARPS, heads * HEAD_DIM + hq] - mx)
            num = num + att[COMPUTE_WARPS, hq * HEAD_DIM + d] * scale
            den = den + att[COMPUTE_WARPS, heads * HEAD_DIM + heads + hq] * scale
        publish(p.attn_o, self.attn_o_index(layer, group, split, hq, d), num, epoch)
        if d == 0:
            publish(p.attn_ml, self.attn_ml_index(layer, group, hq, split, 0), mx, epoch)
            publish(p.attn_ml, self.attn_ml_index(layer, group, hq, split, 1), den, epoch)

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
        """HF q/k RMSNorm then rotate-half RoPE of one head in SMEM, in place, by one warp.
        This lane owns dims ``base``, ``base + 1`` and their partners ``base + 64``, ``+ 65``."""
        x = cute.make_rmem_tensor((4,), Float32)
        for half in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(2):
                x[2 * half + e] = head[base + 64 * half + e]
        ss = Float32(0.0)
        for e in cutlass.range_constexpr(4):
            ss = ss + x[e] * x[e]
        ss = group_sum(ss, 32)
        rinv = cute.math.rsqrt(ss / HEAD_DIM + self.model.rms_eps)
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
            "X": cute.make_layout((2, self.x_width), stride=(self.x_width, 1)),
            "RED": cute.make_layout((COMPUTE_WARPS,)),
            "ATT": cute.make_layout(
                (COMPUTE_WARPS + 1, self.att_stride), stride=(self.att_stride, 1)
            ),
            "QKVS": cute.make_layout(((self.per_row + 2) * HEAD_DIM,)),
            "MLS": cute.make_layout((self.heads_per_split * 2 * self.splits,)),
        }
        self.begin_roles(work=bound, layouts=layouts)
        # Materialize the argument tensors here, in converged code: iterators and layouts are
        # created lazily at first use, and a first use inside one role's dispatch branch would
        # not dominate the other roles' uses.
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
        lm_head: cute.Tensor,
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
        tma, gtensors = make_tma_atoms((wqkv, wo, wgu, wd, lm_head), (k_cache, v_cache))
        bound = self.work.bind(table, workspace, max_ctas)
        # TENSOR_ARGS order.
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


# Plain Python, not cute.jit: loops in a jit body are staged, so building lists of atoms and
# views there leaks values out of the loop region.


def make_tma_atoms(weights, caches):
    """TMA atoms and tile tensors: 32 x 256 weight boxes, 64 x 128 KV-cache boxes."""
    op = cpasync.CopyBulkTensorTileG2SOp()
    w_smem = cute.make_layout((ROWS, TILE_K), stride=(TILE_K, 1))
    kv_smem = cute.make_layout((BLOCK, HEAD_DIM), stride=(HEAD_DIM, 1))
    pairs = [cpasync.make_tiled_tma_atom(op, src, w_smem, (ROWS, TILE_K)) for src in weights]
    pairs += [cpasync.make_tiled_tma_atom(op, src, kv_smem, (BLOCK, HEAD_DIM)) for src in caches]
    return tuple(a for a, _ in pairs), tuple(g for _, g in pairs)


def tma_partitions(tma, gtensors, sW, sKV):
    """``(smem, gmem)`` TMA partitions per atom: ``(TMA, stage)`` and ``(TMA, rows, cols)``."""
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
    """One compile serves every context length and cache capacity: ``max_len`` is dynamic."""
    bf16 = BFloat16
    cache_rows = cute.sym_int(divisibility=BLOCK)
    layers, hidden, inter, vocab = op.layers, op.hidden, op.inter, op.vocab
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (layers * op.qkv_rows, hidden)),
        make_fake_compact_tensor(bf16, (layers * hidden, op.q_heads * HEAD_DIM)),
        make_fake_compact_tensor(bf16, (layers * 2 * inter, hidden)),
        make_fake_compact_tensor(bf16, (layers * hidden, inter)),
        make_fake_compact_tensor(bf16, (vocab, hidden)),  # embedding
        make_fake_compact_tensor(bf16, (vocab, hidden)),  # LM head (the embedding when tied)
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(Int64, (op.scratch_layout.size,)),
        make_fake_compact_tensor(Float32, (vocab,)),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int64, (cute.sym_int(),)),  # trace buffer; globaltimer only
        make_fake_compact_tensor(bf16, (layers, hidden)),
        make_fake_compact_tensor(bf16, (layers, hidden)),
        make_fake_compact_tensor(bf16, (layers, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (layers, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (hidden,)),
        make_fake_compact_tensor(bf16, (cute.sym_int(), 2, HEAD_DIM)),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


# ---------------------------------------------------------------------- host side


def check_model(model: QwenConfig):
    """Raise unless the kernel's tilings cover ``model``: 128-dim heads, o_proj splits of
    whole 256-column tiles, a hidden size the norms split into 4-element multiples per thread,
    whole 256-column down splits and 32-row LM-head tiles. Qwen3-4B fails the hidden size."""
    if model.head_dim != HEAD_DIM:
        raise ValueError(f"head dim must be {HEAD_DIM}, got {model.head_dim}")
    if model.num_q_heads % model.num_kv_heads:
        raise ValueError("q heads must be a multiple of kv heads")
    if model.num_q_heads % (2 * O_SPLITS):
        raise ValueError(f"q heads must be a multiple of {2 * O_SPLITS}")
    if model.hidden_size % (4 * COMPUTE_THREADS):
        raise ValueError(f"hidden size must be a multiple of {4 * COMPUTE_THREADS}")
    if model.intermediate_size % (DOWN_SPLITS * COMPUTE_THREADS):
        raise ValueError(
            f"intermediate size must be a multiple of {DOWN_SPLITS * COMPUTE_THREADS}"
        )
    if model.vocab_size % ROWS:
        raise ValueError(f"vocabulary size must be a multiple of {ROWS}")


# The static assignment's latency model: row cost by op and input propagation, in us. The costs
# were measured on Qwen3-0.6B; row_cost scales them to other shapes by STAGE_COST per extra
# 16 KB weight stage. That is the HBM share of one SM, not compute: a Qwen3-32B globaltimer
# capture (GB200, ctx 1024) measured median rows at 0.32-0.36 us per stage for every weight-
# streaming kind (0.18 us, the compute-warp time with resident data, predicted 7.0 ms for a
# 10.6 ms step; 0.33 predicts 10.3 ms, and 2.87 / 4.93 ms for 8B / 14B against 2.95 / 4.76).
ROW_COST = {QKV: 1.9, ATTN: 3.9, OPROJ: 1.7, GATEUP: 1.9, DOWN: 1.2, LMHEAD: 9.6}
STAGE_COST = 0.33
DEPENDENCY_LATENCY = 1.0


def weight_stages(op: QwenDecodeMegakernel) -> dict[int, int]:
    """16 KB weight stages one row of each kind streams; attention's depend on the context."""
    k = op.hidden_k_tiles
    return {
        QKV: k,
        ATTN: 0,
        OPROJ: op.oproj_k_tiles,
        GATEUP: k,
        DOWN: op.down_k_tiles,
        LMHEAD: LM_TILES * k,
    }


def row_cost(op: QwenDecodeMegakernel) -> dict[int, float]:
    """``ROW_COST`` plus ``STAGE_COST`` per weight stage beyond Qwen3-0.6B's."""
    stages, base = weight_stages(op), weight_stages(QwenDecodeMegakernel())
    return {kind: ROW_COST[kind] + STAGE_COST * (stages[kind] - base[kind]) for kind in ROW_COST}


# One instruction: table words (op, layer, a, b), the producer groups it waits for, and the
# group its completion counts toward (``row_groups``).
Row = tuple[list[int], list[tuple], tuple]


def instruction_rows(op: QwenDecodeMegakernel, ops: tuple[int, ...] = ALL_OPS) -> list[Row]:
    """The step's instructions in dependency order. ``ops`` keeps a subset (single-instruction
    tests)."""
    rows = []

    def add(kind, layer, a, b):
        if kind in ops:
            rows.append(([kind, layer, a, b], *row_groups(op, kind, layer, a, b)))

    for layer in range(op.layers):
        for t in range(op.qkv_rows // ROWS):
            add(QKV, layer, t, 0)
        for group in range(op.attn_groups):
            for split in range(op.splits):
                add(ATTN, layer, group, split)
        for ks in range(O_SPLITS):
            for t in range(op.hidden // ROWS):
                add(OPROJ, layer, t, ks)
        for t in range(2 * op.inter // ROWS):
            add(GATEUP, layer, t, 0)
        for ks in range(DOWN_SPLITS):
            for t in range(op.hidden // ROWS):
                add(DOWN, layer, t, ks)
    # The LM head's layer word is ``layers``: it reads h_in[layers].
    lm_rows = op.vocab // ROWS
    for t in range(0, lm_rows, LM_TILES):
        add(LMHEAD, op.layers, t, min(LM_TILES, lm_rows - t))
    return rows


def build_table(
    op: QwenDecodeMegakernel, ctas: int, ops: tuple[int, ...] = ALL_OPS
) -> torch.Tensor:
    """The ``instruction_rows`` table: in dependency order for the atomic schedule, assigned
    to ``ctas`` CTAs for the static one."""
    rows = instruction_rows(op, ops)
    if isinstance(op.work.schedule, StaticPersistent):
        table = assign_static(rows, ctas, row_cost(op))
    else:
        table = [words for words, _, _ in rows]
    return torch.tensor([words + [i] for i, words in enumerate(table)], dtype=torch.int32)


def row_groups(
    op: QwenDecodeMegakernel, kind: int, layer: int, a: int, b: int
) -> tuple[list[tuple], tuple]:
    """The producer groups instruction ``(kind, layer, a, b)`` waits for, and the group its
    completion counts toward. A group is done when all of its rows are."""
    if kind == QKV:
        head = a * ROWS // HEAD_DIM
        if head < op.q_heads:
            kvh = head // op.group_size
        else:
            kvh = (head - op.q_heads) % op.kv_heads
        return [("down", layer - 1)] if layer else [], ("qkv", layer, kvh)
    if kind == ATTN:
        return [("qkv", layer, op.kv_head(a))], ("attn", layer, a)
    if kind == OPROJ:
        first = b * op.heads_per_split // op.per_row
        last = ((b + 1) * op.heads_per_split - 1) // op.per_row
        return [("attn", layer, g) for g in range(first, last + 1)], ("oproj", layer)
    if kind == GATEUP:
        per_split = 2 * op.inter // ROWS // DOWN_SPLITS
        return [("oproj", layer)], ("gateup", layer, a // per_split)
    if kind == DOWN:
        return [("gateup", layer, b)], ("down", layer)
    return [("down", layer - 1)], ("lmhead",)


class RowSpan(NamedTuple):
    """Where and when the ``ROW_COST`` model runs a row (us from launch), and what it waited on."""

    cta: int
    start: float
    end: float
    gates: tuple[int, ...]  # the last-finishing row of each producer group it waits for
    after: int | None  # the row that set its start: its latest gate, or its CTA's previous row


def list_schedule(rows: list[Row], ctas: int, cost: dict[int, float] = ROW_COST) -> list[RowSpan]:
    """Greedy list schedule of dependency-ordered rows onto ``ctas`` CTAs under the ``cost``
    model (``row_cost``): each row goes to the CTA that can start it first, once every
    producer group it waits for has finished plus ``DEPENDENCY_LATENCY``."""
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
        spans.append(RowSpan(cta, start, start + cost[words[0]], gates, after))
        tail[cta], free[cta] = i, spans[i].end
        if group not in last or spans[i].end >= spans[last[group]].end:
            last[group] = i
    return spans


def assign_static(rows: list[Row], ctas: int, cost: dict[int, float]) -> list[list[int]]:
    """``list_schedule``'s per-CTA row lists laid out ``[slot][cta]`` for ``StaticPersistent``
    (CTA b runs rows b, b + ctas, ...); short lists are padded with dropped rows.

    Each list keeps the dependency order, so with every CTA resident the earliest unfinished
    row always has its producers finished: no deadlock."""
    lists: list[list[list[int]]] = [[] for _ in range(ctas)]
    for (words, _, _), span in zip(rows, list_schedule(rows, ctas, cost)):
        lists[span.cta].append(words)
    slots = max(len(rows) for rows in lists)
    padding = [-1] * (WORDS - 1)
    return [
        lists[cta][slot] if slot < len(lists[cta]) else padding
        for slot in range(slots)
        for cta in range(ctas)
    ]


def rope_table(max_len: int, device, theta: float = 1e6) -> torch.Tensor:
    """(max_len, 2, HEAD_DIM) bf16 cos and sin, computed as HF does."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
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
        """``ops``: keep only these instruction kinds (tests). ``trace``: compile with ranges for
        the work items and ``trace_region``s, recorded by ``"globaltimer"``
        (``qwen_megakernel_trace.trace_session`` decodes a launch) or ``"iket"`` (``run-iket``);
        ``waits`` adds the layer's wait ranges. ``config``: ``QwenDecodeMegakernel`` constructor
        arguments; the model shape comes from ``weights.config``."""
        if max_len % BLOCK:
            raise ValueError(f"max_len must be a multiple of {BLOCK}, got {max_len}")
        device = weights.embed.device
        self.max_len = max_len
        self.op = op = QwenDecodeMegakernel(model=weights.config, **config)
        self.workspace, self.max_ctas = op.work.launch_args(device)
        self.table = build_table(op, self.max_ctas, ops).to(device)
        rows_per_cta = -(-self.table.shape[0] // self.max_ctas)
        events = max(TRACE_EVENTS, TRACE_EVENTS_PER_ROW * rows_per_cta)
        op.enable_trace(trace, waits=waits, events_per_role=events)
        self.wqkv = torch.cat((weights.q_proj, weights.k_proj, weights.v_proj), dim=1).reshape(
            -1, op.hidden
        )
        self.wo = weights.o_proj.reshape(-1, op.q_heads * HEAD_DIM).contiguous()
        # Gate and up rows interleaved, so a 32-row tile holds 16 (gate, up) pairs.
        self.wgu = torch.stack((weights.gate, weights.up), dim=2).reshape(-1, op.hidden)
        self.wd = weights.down.reshape(-1, op.inter).contiguous()
        self.weights = weights
        self.rope = rope_table(max_len, device, weights.config.rope_theta)
        self.scratch = torch.zeros(op.scratch_layout.size, device=device, dtype=torch.int64)
        self.logits = torch.zeros(op.vocab, device=device)
        self.state = torch.zeros(2, device=device, dtype=torch.int32)
        self.step_info = torch.zeros(2, device=device, dtype=torch.int32)
        # Placeholder; trace_session swaps in the real globaltimer buffer for a traced launch.
        self.trace_buffer = torch.zeros(1, device=device, dtype=torch.int64)
        self.fn = compile_megakernel(self.op)

    def tagged(self, name: str) -> torch.Tensor:
        """The int64 words of a tagged buffer, shaped as ``ScratchLayout.shapes``."""
        layout = self.op.scratch_layout
        shape = layout.shapes[name]
        offset = layout.offsets[name]
        return self.scratch[offset : offset + math.prod(shape)].view(shape)

    def values(self, name: str) -> torch.Tensor:
        """A tagged buffer's fp32 values, ignoring the tags."""
        return self.tagged(name).view(torch.int32)[..., 0::2].view(torch.float32)

    def set_values(self, name: str, values: torch.Tensor):
        """Publish ``values`` to a tagged buffer for the next launch; tests feed single
        instructions this way."""
        words = self.tagged(name).view(torch.int32)
        words[..., 0::2] = values.float().view(torch.int32)
        words[..., 1::2] = self.state[0] + 1

    def launch(self, k_cache: torch.Tensor, v_cache: torch.Tensor):
        """One decode step for the token and position in ``step_info``. Both are read on the
        device, so the caller keeps ``0 < pos < max_len`` and the token in the vocabulary, and
        runs one launch at a time: the scratch and epoch state belong to this runner."""
        shape = (self.op.layers, self.op.kv_heads, self.max_len, HEAD_DIM)
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
            w.lm_head,
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
        ``weights`` must be the runner's own; they were packed at construction."""
        if weights is not self.weights:
            raise ValueError("weights must be the ones this QwenMegakernel was built from")
        if not 0 < pos < self.max_len:
            raise ValueError(f"need 0 < pos < max_len = {self.max_len}, got {pos}")
        self.step_info[0] = token
        self.step_info[1] = pos
        self.launch(k_cache, v_cache)
        return self.logits.clone()
