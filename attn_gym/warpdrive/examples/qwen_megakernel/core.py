"""The Qwen3 decode megakernel's base: scratch layout, tagged-word helpers, the CUDA-core GEMV
engine and ``QwenDecodeMegakernel`` with its compile function. See ``qwen_megakernel``."""

import math
from dataclasses import dataclass

import cutlass
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
from attn_gym.warpdrive.examples.qwen_megakernel.ref import QwenConfig

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
REDUCE_TOKENS = 4  # batched residual reduce: tokens per poll
POLL_WORDS = 32  # prologue: tagged words per thread per poll (bounds registers)
PART_ELEMENTS = 8  # batch-one gate/up: residual elements (5 words each) per poll
EXPERT_GROUPS = 4  # batched MoE: rows per expert, each a quarter of its gate/up or down tiles
COMPUTE_WARPS = 8
ROWS_PER_WARP = ROWS // COMPUTE_WARPS
COMPUTE_THREADS = 32 * COMPUTE_WARPS
# Reason for the compute role's thread-sync fence skips (tcgen05 engines; the CUDA-core one
# issues no tcgen05 ops, so they are no-ops there).
COMPUTE_TMEM_FENCED = (
    "the compute warps touch TMEM only in the engine's acc loads, which the acc wait and release "
    "fence"
)

QKV, ATTN, OPROJ, GATEUP, DOWN, LMHEAD, ROUTER, REDUCE, NORM = range(9)
ALL_OPS = (QKV, ATTN, OPROJ, GATEUP, DOWN, LMHEAD, ROUTER, REDUCE, NORM)
EVICT_FIRST = 0x12F0000000000000  # TMA L2 cache hint (CUTLASS CacheHintSm90::EVICT_FIRST)
WORDS = 5  # op, layer, a, b, table row
OP_NAMES = ("QKV", "ATTN", "OPROJ", "GATEUP", "DOWN", "LMHEAD", "ROUTER", "REDUCE", "NORM")
SHARED_CHUNK = 64  # activation columns per epoch flag: one tcgen05 gate/up tile
# Globaltimer records per role and CTA: at least TRACE_EVENTS (0.6B: a CTA runs ~140 rows),
# else TRACE_EVENTS_PER_ROW per row of the CTA's share (a row records its work item and regions).
TRACE_EVENTS, TRACE_EVENTS_PER_ROW = 2048, 8
WEIGHT_ATOMS = 6  # wqkv, wo, wgu, wd, LM head, router; the KV caches' atoms follow


def qkv_rows(model: QwenConfig) -> int:
    """Rows of the fused [Wq; Wk; Wv]."""
    return (model.num_q_heads + 2 * model.num_kv_heads) * HEAD_DIM


def mlp_width(model: QwenConfig) -> int:
    """Columns of the MLP's activation: the intermediate size, or for a MoE model the routed
    experts' activations side by side (slot s holds expert e_s's, scaled by its weight)."""
    if model.is_moe:
        return model.experts_per_token * model.moe_intermediate_size
    return model.intermediate_size


def attention_groups(model: QwenConfig) -> tuple[int, int]:
    """``(groups, per_row)``: an attention row serves ``per_row`` q heads of one kv head, 2
    when the GQA group size is even (the kernel's register budget fits no more), else 1; a
    layer has ``groups`` such rows per KV split. Each reads its kv head's KV blocks."""
    per_row = 2 if model.num_q_heads // model.num_kv_heads % 2 == 0 else 1
    return model.num_q_heads // per_row, per_row


class ScratchLayout:
    """Offsets (in 64-bit words) of the tagged per-step intermediates."""

    def __init__(self, model: QwenConfig, splits: int, batch: int = 1, shared: bool = False):
        layers, hidden = model.num_layers, model.hidden_size
        groups, per_row = attention_groups(model)
        # Every buffer has a leading token axis (``batch`` sequences decoded together).
        self.shapes = {
            "h_in": (batch, layers + 1, hidden),  # layer inputs (residual); [layers]: LM head's
            "qkv": (batch, layers, qkv_rows(model)),
            # Per attention row group (per_row q heads of one kv head) and KV split.
            "attn_o": (batch, layers, groups, splits, per_row, HEAD_DIM),  # unnormalized O
            "attn_ml": (batch, layers, groups, per_row, splits, 2),  # (m, l); log2 m
            "o_part": (batch, layers, O_SPLITS, hidden),
            # The residual after attention; h_in's per-token stride (layer [layers] unused).
            "h_mid": (batch, layers + 1, hidden),
            "act": (batch, layers, mlp_width(model)),
            "d_part": (batch, layers, DOWN_SPLITS, hidden),
            "router_logits": (batch, layers, model.num_experts or ROWS),  # MoE only
        }
        # Batched MoE: the normalized MLP input per token, each expert's token list (count;
        # entries t * top_k + rank; routing weights) and the per-(token, rank) down outputs.
        moe_batch = model.is_moe and batch > 1
        experts, top_k = model.num_experts or 1, model.experts_per_token or 1
        self.shapes.update({
            "x_mid": (batch, layers, hidden) if moe_batch else (1,),
            "e_count": (layers, experts) if moe_batch else (1,),
            "e_entry": (layers, experts, batch) if moe_batch else (1,),
            "e_weight": (layers, experts, batch) if moe_batch else (1,),
            "e_part": (batch, layers, top_k, hidden) if moe_batch else (1,),
        })  # fmt: skip
        # Shared inputs (QwenDecodeMegakernelTcStream): a flag per token of each normalized input
        # (ln_in and ln_post per layer, then the final norm) and per act column chunk.
        chunks = mlp_width(model) // SHARED_CHUNK
        self.shapes.update({
            "x_flags": (2 * layers + 1, batch) if shared else (1,),
            "act_flags": (layers, chunks) if shared else (1,),
        })  # fmt: skip
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
    tma: tuple  # TMA atoms: wqkv, wo, wgu, wd, lm head, router, k cache, v cache
    gmem: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    sW: StageView  # weight stage partition
    sKV: StageView  # KV stage partition
    xs: cute.Tensor  # (2, x_width) bf16 GEMV inputs, double-buffered by item
    red: cute.Tensor  # (COMPUTE_WARPS,) fp32 block reduction
    att: cute.Tensor  # (9, att_stride) fp32 per-warp attention states
    qkvs: cute.Tensor  # ((per_row + 2) * HEAD_DIM,) fp32 an attention row's q heads, k and v
    route: cute.Tensor  # (2, top_k) fp32 routing weights by rank, double-buffered by item
    # Tagged int64 activations, flat row-major (ScratchLayout.shapes; see the *_index helpers).
    h_in: cute.Tensor
    qkv: cute.Tensor
    attn_o: cute.Tensor
    attn_ml: cute.Tensor
    o_part: cute.Tensor
    h_mid: cute.Tensor
    act: cute.Tensor
    d_part: cute.Tensor
    router_logits: cute.Tensor
    x_mid: cute.Tensor  # batched MoE (ScratchLayout)
    e_count: cute.Tensor
    e_entry: cute.Tensor
    e_weight: cute.Tensor
    e_part: cute.Tensor
    x_flags: cute.Tensor  # shared-input epoch flags (ScratchLayout)
    act_flags: cute.Tensor
    exp: cute.Tensor  # batched MoE SMEM: the routing of a layer, then an expert row's tokens
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
    xscratch: cute.Tensor  # (ctas, x_slot) bf16 per-CTA GEMV inputs (QwenDecodeMegakernelTcStream)
    xshared: cute.Tensor  # (shared_size,) bf16 inputs all CTAs read (QwenDecodeMegakernelTcStream)


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
    "xscratch",
    "xshared",
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


def chunk_width(width: int, threads: int) -> int:
    """The widest chunk (8, 4, 2 or 1 columns) that ``threads`` threads tile ``width`` columns with in
    whole rounds; 8 keeps a GEMV input chunk one 16-byte group of ``x_index``."""
    return next(c for c in (8, 4, 2, 1) if width % (threads * c) == 0)


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


class CudaCoreGemv:
    """The GEMV engine on CUDA cores, for one token: warp w owns rows ROWS_PER_WARP * w .. of each
    32 x 256 weight stage and lane l its columns 8l .. 8l + 7; even and odd columns accumulate
    separately in packed FFMA2s. A warp transpose-reduce then leaves row r's sum on lane
    (r % ROWS_PER_WARP) * (32 // ROWS_PER_WARP) of warp r // ROWS_PER_WARP.

    ``compute_loop`` calls ``input_ready`` per GEMV row, ``gemv`` calls ``tile`` per 32-row tile
    and ``epilogue`` asks ``owns_row``, ``owns_pair`` and ``pair_value``: another engine
    supplies the same five."""

    def input_ready(self, chans, new):
        """Called by every compute thread for each GEMV row.
        ``new`` is true when this CTA wrote and synchronized a new input; shared-input
        readiness is handled by ``xload``."""
        return chans

    @cute.jit
    def tile(self, p: MegakernelParams, chans, env, xbuf, k_tiles, warp, lane):
        """``(values, row, chans)``: one tile's dot products against ``xs[xbuf]`` (one per token:
        this engine runs batch one), this lane's tile row and the advanced channel ends
        (``chans[0]`` is the weight ring)."""
        w = chans[0]
        xs = p.xs[xbuf, None]
        acc = cute.make_rmem_tensor((ROWS_PER_WARP, 2), Float32)
        acc.fill(0.0)
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
                cute.autovec_copy(cute.local_tile(xs, (8,), (kt * 32 + lane,)), xh)
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
        return [v], warp * ROWS_PER_WARP + lane // (32 // ROWS_PER_WARP), (w, *chans[1:])

    def owns_row(self, warp, lane):
        return lane % (32 // ROWS_PER_WARP) == 0

    def owns_pair(self, warp, lane):
        """Owns an even row 2j, whose partner 2j + 1 ``pair_value`` delivers."""
        return lane % (2 * (32 // ROWS_PER_WARP)) == 0

    @cute.jit
    def pair_value(self, v):
        """Row 2j + 1's value on the owner of row 2j (every lane must call it)."""
        return cute.arch.shuffle_sync_down(v, offset=32 // ROWS_PER_WARP)


class QwenDecodeMegakernel(WarpSpecialized):
    rows, tile_k = ROWS, TILE_K  # a weight stage and work-item tile: rows x columns (16 KB)
    x_tail = 0  # bf16 elements after each GEMV input that an engine may read
    x_buffers = 2  # GEMV input buffers: the next row's input is written while one is read
    max_batch = 1
    streams_inputs = False  # GEMV inputs in global memory (QwenDecodeMegakernelTcStream)
    shared_size = 8  # bf16 elements of the shared inputs (a stub unless shared)
    x_slot = 64  # bf16 elements per CTA of the input scratch (a stub unless streaming)

    def __init__(
        self,
        stages: int | None = None,
        splits: int | None = None,
        schedule: str = "static",
        roles: dict[str, RoleConfig] | None = None,
        model: QwenConfig = QWEN3_0_6B,
        batch: int = 1,
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
        ``model``: the Qwen3 shape (default Qwen3-0.6B); see ``check_model``.
        ``batch``: sequences decoded per launch, one token each (up to the engine's
        ``max_batch``; dense models only)."""
        check_model(model)
        if not 1 <= batch <= self.max_batch:
            raise ValueError(
                f"batch must be in [1, {self.max_batch}] for this engine, got {batch}"
            )
        self.batch = batch
        # Batched MoE: rows per (layer, expert) over the tokens routed there (``route_batch``).
        self.moe_batched = model.is_moe and batch > 1
        if self.moe_batched and model.num_experts != self.rows:
            raise ValueError(f"batched MoE needs {self.rows} experts (one router tile)")
        if self.moe_batched and (2 * model.moe_intermediate_size // self.rows) % EXPERT_GROUPS:
            raise ValueError(f"batched MoE needs {EXPERT_GROUPS} groups of gate/up tiles")
        # GEMV input rows stored in SMEM, S = batch: row t, column k at
        # (k // 8) * S * 8 + t * 8 + k % 8. An MMA's B rows past the batch read the next column
        # group (their D columns are discarded); see TcGen05Gemv.
        self.x_rows = batch
        self.model = model
        self.layers, self.hidden = model.num_layers, model.hidden_size
        self.inter, self.vocab = mlp_width(model), model.vocab_size
        # MoE: the gate/up and down GEMVs gather slot s's rows from the routed expert e_s.
        self.moe = model.is_moe
        self.experts, self.top_k = model.num_experts, model.experts_per_token
        self.expert_inter = model.moe_intermediate_size
        self.q_heads, self.kv_heads = model.num_q_heads, model.num_kv_heads
        self.qkv_rows = qkv_rows(model)
        self.attn_groups, self.per_row = attention_groups(model)
        self.group_size = self.q_heads // self.kv_heads  # q heads per kv head
        self.heads_per_split = self.q_heads // O_SPLITS  # q heads in one o_proj split
        # Weight stages per 32-row tile: QKV, GATEUP and LMHEAD read all hidden columns.
        self.hidden_k_tiles = self.hidden // self.tile_k
        self.oproj_k_tiles = self.heads_per_split * HEAD_DIM // self.tile_k
        self.down_k_tiles = self.inter // self.tile_k // DOWN_SPLITS
        # The longest GEMV input: the hidden vector, a down split or an o_proj split.
        self.x_width = max(self.hidden, self.inter // DOWN_SPLITS, self.heads_per_split * HEAD_DIM)
        if self.moe_batched:  # an expert down row's input is one expert's activation slot
            self.x_width = max(self.x_width, self.expert_inter)
        if schedule not in ("static", "atomic"):
            raise ValueError(f"schedule must be 'static' or 'atomic', got {schedule!r}")
        # The GEMV (ROWS_PER_WARP rows per warp), prologue (token_warps per token) and attention
        # (8 positions per warp per block) tilings fix the compute role's size.
        compute = (roles or {}).get("compute")
        if compute is not None and [len(r) for r in compute.warps] != [COMPUTE_WARPS]:
            raise ProtocolError(
                f"roles['compute'] must be one instance of {COMPUTE_WARPS} warps, got "
                f"{compute.warps}"
            )
        self.roles = roles
        if splits is None:
            # A batch multiplies the attention rows: split each sequence's KV that much less.
            splits = max(1, max(MIN_SPLITS, ATTN_ROWS // self.attn_groups) // batch)
        if not 1 <= splits <= 32:
            raise ValueError(f"splits must be in [1, 32], got {splits}")
        self.splits = splits
        fit = (SMEM_BYTES - self.fixed_smem_bytes()) // STAGE_BYTES
        if stages is None:
            stages = min(MAX_STAGES, fit)
        if not 2 <= stages <= fit:
            raise ValueError(f"stages must be in [2, {fit}] for this model, got {stages}")
        self.stages = stages
        self.scratch_layout = ScratchLayout(model, splits, batch, self.streams_inputs)
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        kv = self.smem_plan.buffer("KV", BFloat16, align=1024, overlaps=w)
        self.smem_plan.buffer("X", BFloat16, align=16)  # inputs are bf16-rounded: lossless
        self.smem_plan.buffer("RED", Float32, align=16)
        self.smem_plan.buffer("ATT", Float32, align=16)
        self.smem_plan.buffer("QKVS", Float32, align=16)
        self.smem_plan.buffer("ROUTE", Float32, align=16)
        if self.moe_batched:
            self.smem_plan.buffer("EXP", Float32, align=16)
        # Compute reads stages with generic loads before TMA refills them. For a Tma-filled ring
        # released by Warps the layer emits the fence.proxy.async on release.
        self.w = Ring(stages, Tma(STAGE_BYTES), self.weight_consumers(), buffer={"W": w, "KV": kv})
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.engine = CudaCoreGemv()
        self.work = WorkStream(
            mapping=InstructionTable(),
            schedule=StaticPersistent() if schedule == "static" else AtomicPersistent(stages=1),
        )

    def weight_consumers(self):
        """The weight ring's consumer mechanism (the engine may add an MMA warp)."""
        return Warps

    def fixed_smem_bytes(self) -> int:
        """Shared memory besides the weight ring, with headroom for barriers and alignment."""
        floats = (
            COMPUTE_WARPS  # RED
            + (COMPUTE_WARPS + 1) * self.att_stride  # ATT
            + (self.per_row + 2) * HEAD_DIM  # QKVS
            + self.route_floats  # ROUTE
        )
        if self.moe_batched:
            floats += self.exp_floats
        x_bytes = 2 * self.x_buffers * self.x_stride
        return x_bytes + 4 * floats + SMEM_HEADROOM

    @property
    def exp_floats(self) -> int:
        """EXP: each token's experts and weights by rank, then an expert row's entries, weights
        and token count."""
        return 2 * self.batch * self.top_k + 2 * self.batch + 1

    @property
    def shares_mlp(self) -> bool:
        """Whether dense MLP inputs are shared: gate/up reads NORM outputs, down reads activations."""
        return self.streams_inputs and not self.moe

    @property
    def x_stride(self) -> int:
        """bf16 elements of one GEMV input buffer: x_rows rows of x_width, plus the tail."""
        return self.x_rows * self.x_width + self.x_tail

    def x_index(self, t, k):
        """Element of GEMV input row t, column k (see x_rows)."""
        if cutlass.const_expr(self.x_rows == 1):
            return k
        return (k // 8) * (self.x_rows * 8) + t * 8 + k % 8

    def x_buffer(self, p: MegakernelParams, xbuf):
        return p.xs[xbuf % self.x_buffers, None]

    @property
    def route_floats(self) -> int:
        """ROUTE: the top-k routing weights by rank, double-buffered like the GEMV input.
        Dense models keep a stub."""
        return 2 * self.route_width

    @property
    def route_width(self) -> int:
        return self.top_k if self.moe else 2

    @property
    def att_stride(self) -> int:
        """Floats per warp state in ATT: O of each q head, then their m, then their l."""
        return self.per_row * (HEAD_DIM + 2)

    def get_name(self) -> str:
        return (
            f"warpdrive_qwen_decode_h{self.hidden}{f'_moe{self.experts}' if self.moe else ''}"
            f"_s{self.stages}_kv{self.splits}{f'_b{self.batch}' if self.batch > 1 else ''}_"
            f"{self.work.get_key()}{self.trace_key()}"
        )

    # ------------------------------------------------------------------ roles

    @role(warps=COMPUTE_WARPS)
    @cute.jit
    def load(self, p: MegakernelParams, *, w: Producer, work: WorkCursor):
        w = w.views(W=p.sW, KV=p.sKV)
        tma_qkv, tma_o, tma_gu, tma_d, tma_lm, tma_router, tma_k, tma_v = p.tma
        g_qkv, g_o, g_gu, g_d, g_lm, g_router, g_k, g_v = p.gmem
        max_len = p.k_cache.shape[0] // (self.batch * self.layers * self.kv_heads)
        # MoE: the expert rows to load follow the layer's routing, recomputed once per layer.
        epoch = p.state[0] + 1
        lane = cute.arch.lane_idx()
        chosen = cute.make_rmem_tensor((max(self.top_k, 1),), Int32)
        routed_layer = Int32(-1)
        while work.valid:
            op, layer, a, b, _row = work.item
            if cutlass.const_expr(self.moe and not self.moe_batched):
                if ((op == GATEUP) | (op == DOWN)) & (layer != routed_layer):
                    self.route(p, layer, lane, epoch, chosen)
                    routed_layer = layer
            if cutlass.const_expr(self.moe_batched):
                if (op == GATEUP) | (op == DOWN):
                    if self.expert_count(p, layer, a, epoch) > 0:
                        if op == GATEUP:
                            w = self.load_expert(p, w, GATEUP, layer, a, b)
                        else:
                            w = self.load_expert(p, w, DOWN, layer, a, b)
            if op == ATTN:
                t, split = self.attn_token(b)
                blk0, blk1 = self.split_blocks(p.step[t, 1], split)
                head = (t * self.layers + layer) * self.kv_heads + self.kv_head(a)
                row0 = head * (max_len // BLOCK)
                for blk in cutlass.range(blk0, blk1, unroll=1):
                    w = self.load_tiles(w, tma_k, g_k, row0 + blk, 0, 1, "KV")
                    w = self.load_tiles(w, tma_v, g_v, row0 + blk, 0, 1, "KV")
            elif op == QKV:
                tile = layer * (self.qkv_rows // self.rows) + a
                w = self.load_tiles(w, tma_qkv, g_qkv, tile, 0, self.hidden_k_tiles, "W")
            elif op == OPROJ:
                tile = layer * (self.hidden // self.rows) + a
                k_tiles = self.oproj_k_tiles
                w = self.load_tiles(w, tma_o, g_o, tile, b * k_tiles, k_tiles, "W")
            elif (
                (op == REDUCE)
                | (op == NORM)
                | (((op == GATEUP) | (op == DOWN)) & self.moe_batched)
            ):
                pass  # no weights, or loaded above
            elif op == GATEUP:
                tile = layer * (2 * self.inter // self.rows) + a
                if cutlass.const_expr(self.moe):
                    # Tile a of slot s is tile a % per_slot of expert e_s's gate/up rows.
                    per_slot = 2 * self.expert_inter // self.rows
                    expert = self.expert_of(chosen, a // per_slot)
                    tile = (layer * self.experts + expert) * per_slot + a % per_slot
                w = self.load_tiles(w, tma_gu, g_gu, tile, 0, self.hidden_k_tiles, "W")
            elif op == DOWN:
                tile = layer * (self.hidden // self.rows) + a
                k_tiles = self.down_k_tiles
                if cutlass.const_expr(self.moe):
                    # Column tile kt of the routed activation is tile kt % per_slot of slot
                    # kt // per_slot's expert down projection.
                    per_slot = self.expert_inter // self.tile_k
                    for k in cutlass.range_constexpr(k_tiles):
                        kt = b * k_tiles + k
                        expert = self.expert_of(chosen, kt // per_slot)
                        row = (layer * self.experts + expert) * (self.hidden // self.rows) + a
                        w = self.load_tiles(w, tma_d, g_d, row, kt % per_slot, 1, "W")
                else:
                    w = self.load_tiles(w, tma_d, g_d, tile, b * k_tiles, k_tiles, "W")
            elif op == ROUTER:
                tile = layer * (self.experts // self.rows) + a
                w = self.load_tiles(w, tma_router, g_router, tile, 0, self.hidden_k_tiles, "W")
            else:
                for rt in cutlass.range(a, a + b, unroll=1):
                    w = self.load_tiles(w, tma_lm, g_lm, rt, 0, self.hidden_k_tiles, "W")
            work.advance()
        return w, work

    @cute.jit
    def load_expert(self, p: MegakernelParams, w, kind: cutlass.Constexpr, layer, expert, group):
        """Batched MoE: all of one expert's gate/up or down (``kind``) tiles."""
        atoms, tiles = (p.tma[2], p.gmem[2]) if kind == GATEUP else (p.tma[3], p.gmem[3])
        k_tiles, row_tiles = self.expert_tiles(kind)
        first = ((layer * self.experts + expert) * EXPERT_GROUPS + group) * row_tiles
        for rt in cutlass.range(first, first + row_tiles, unroll=1):
            w = self.load_tiles(w, atoms, tiles, rt, 0, k_tiles, "W")
        return w

    def expert_tiles(self, op):
        """Batched MoE: (k_tiles, row_tiles) of an expert row, group b of EXPERT_GROUPS: a share of
        gate/up's 2 x expert width rows over the hidden columns, or of down's hidden rows over
        the expert width."""
        if op == GATEUP:
            return self.hidden_k_tiles, 2 * self.expert_inter // self.rows // EXPERT_GROUPS
        return self.expert_inter // self.tile_k, self.hidden // self.rows // EXPERT_GROUPS

    @cute.jit
    def expert_count(self, p: MegakernelParams, layer, expert, epoch):
        """Batched MoE: how many tokens the router row sent to ``expert`` in ``layer``."""
        count = cute.make_rmem_tensor((1,), Float32)
        fetch([(p.e_count, layer * self.experts + expert, 1)], count, epoch)
        return Int32(count[0])

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

    @cute.jit
    def route(self, p: MegakernelParams, layer, lane, epoch, chosen, weights=None, t=0):
        """One warp's top-k routing of ``layer``, uniform across the warp: ``chosen`` (top_k Int32)
        gets the experts by rank, largest router logit first, ties to the lower expert index
        (the reference's stable sort). With ``weights`` (top_k floats), lane 0 also writes their
        routing weights: softmax over all experts in fp32, renormalized over the top k when the
        model's ``norm_topk_prob`` is set, and rounded to bf16, as HF does.

        Router logits are bf16-rounded, so a logit and its index pack into one int32 key that
        orders like (logit, -index): k rounds of a 5-shuffle warp max pick the top k."""
        n = self.experts // 32  # lane l holds experts n*l .. n*l + n - 1
        logits = cute.make_rmem_tensor((n,), Float32)
        base = (t * self.layers + layer) * self.experts
        fetch([(p.router_logits, base + n * lane, n)], logits, epoch)
        keys = cute.make_rmem_tensor((n,), Int32)
        for i in cutlass.range_constexpr(n):
            bits = (Int32(bitcast(logits[i].ir_value(), T.i32())) >> 16) & 0xFFFF
            # bf16 bits -> an unsigned order (flip negatives, set the sign of positives), then
            # recentred so a signed compare keeps it.
            order = cutlass.select_((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000)
            keys[i] = ((order - 0x8000) << 16) | (0xFFFF - (n * lane + i))
        chosen_logits = cute.make_rmem_tensor((self.top_k,), Float32)
        for r in cutlass.range_constexpr(self.top_k):
            best = keys[0]
            for i in cutlass.range_constexpr(1, n):
                best = cutlass.max(best, keys[i])
            for step in cutlass.range_constexpr(5):
                best = cutlass.max(best, cute.arch.shuffle_sync_bfly(best, offset=16 >> step))
            chosen[r] = 0xFFFF - (best & 0xFFFF)
            order = (best >> 16) + 0x8000
            bits = cutlass.select_((order & 0x8000) != 0, order & 0x7FFF, (~order) & 0xFFFF)
            chosen_logits[r] = Float32(bitcast((bits << 16).ir_value(), T.f32()))
            for i in cutlass.range_constexpr(n):
                keys[i] = cutlass.select_(keys[i] == best, Int32(-(2**31)), keys[i])
        if cutlass.const_expr(weights is not None):
            top = chosen_logits[0]
            total = Float32(0.0)
            for i in cutlass.range_constexpr(n):
                total = total + cute.math.exp2((logits[i] - top) * LOG2E)
            total = group_sum(total, 32)
            chosen_sum = Float32(0.0)
            probs = cute.make_rmem_tensor((self.top_k,), Float32)
            for r in cutlass.range_constexpr(self.top_k):
                probs[r] = cute.math.exp2((chosen_logits[r] - top) * LOG2E) / total
                chosen_sum = chosen_sum + probs[r]
            if lane == 0:
                for r in cutlass.range_constexpr(self.top_k):
                    if cutlass.const_expr(self.model.norm_topk_prob):
                        weights[r] = bf16_round(probs[r] / chosen_sum)
                    else:
                        weights[r] = bf16_round(probs[r])

    @cute.jit
    def expert_of(self, chosen, slot):
        """The expert at rank ``slot`` of ``route``'s ``chosen`` (a select chain: no
        dynamically indexed registers)."""
        expert = chosen[0]
        for r in cutlass.range_constexpr(1, self.top_k):
            expert = cutlass.select_(slot == r, chosen[r], expert)
        return expert

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(self, p: MegakernelParams, *, w: Consumer, work: WorkCursor, me: Local):
        chans, work = self.compute_loop(p, (w,), work, me)
        return chans[0], work

    @cute.jit
    def next_input(self, op, layer, a, b, xkey, xbuf):
        """``(new, xkey, xbuf)``: whether GEMV row ``(op, layer, a, b)`` writes a new input to
        ``xs[xbuf]`` (alternating) or reuses the previous one. Consecutive rows with the same key
        share it; row 0 of layer 0's QKV, and with ``sums_parts`` of GATEUP (of ROUTER for MoE),
        always runs the prologue: it publishes h_in[0] or h_mid."""
        ks = cutlass.select_((op == OPROJ) | (op == DOWN), b, 0)
        key = (op * (self.layers + 1) + layer) * 8 + ks
        mid = ROUTER if self.moe else GATEUP  # the op whose row 0 publishes h_mid
        publishes = (a == 0) & ((op == QKV) & (layer == 0))
        if cutlass.const_expr(self.sums_parts):
            publishes = publishes | ((a == 0) & (op == mid))
        new = (key != xkey) | publishes
        if cutlass.const_expr(self.moe_batched):
            new = new | (op == GATEUP) | (op == DOWN)  # each expert gathers its own tokens
        if new:
            xkey = key
            xbuf = 1 - xbuf
        return new, xkey, xbuf

    @cute.jit
    def compute_loop(self, p: MegakernelParams, chans, work: WorkCursor, me: Local, env=()):
        """The compute role over its rows; ``chans[0]`` is the weight ring, the rest belong to
        the engine, and ``env`` holds the engine's loop-invariant values (TMEM tensors). Returns
        the channel ends and the cursor."""
        tid, warp, lane = me.tid, me.warp, me.lane
        ctas = cute.arch.grid_dim()[0]
        epoch = p.state[0] + 1
        xkey = Int32(-1)
        xbuf = Int32(0)
        while work.valid:
            op, layer, a, b, row = work.item
            if op == ATTN:
                t, split = self.attn_token(b)
                pos = p.step[t, 1]
                blk0, blk1 = self.split_blocks(pos, split)
                if (blk0 == blk1) & (split != self.splits - 1):
                    self.publish_empty_partial(p, layer, a, split, t, epoch, tid)
                else:
                    w = self.attention(
                        p, chans[0], layer, a, split, t, pos, epoch, tid, warp, lane, row
                    )
                    chans = (w, *chans[1:])
            elif op == REDUCE:
                self.reduce_slots(p, layer, a, epoch, tid, row)
            elif op == NORM:  # only in tables of kernels that share inputs
                if cutlass.const_expr(self.streams_inputs):
                    self.norm_row(p, layer, a, b, epoch, tid, warp, lane, row)
            elif self.row_active(p, op, layer, a, epoch):
                new, xkey, xbuf = self.next_input(op, layer, a, b, xkey, xbuf)
                if cutlass.const_expr(self.streams_inputs):
                    # NORM rows or gate/up produce a shared input; xload waits for its flags.
                    new = new & (self.shared_input(op) == 0)
                if new:
                    ks = cutlass.select_((op == OPROJ) | (op == DOWN), b, 0)
                    self.prologue(p, op, layer, a, ks, epoch, xbuf, tid, warp, lane, row)
                    self.csync.sync(skip_fence="both", reason=COMPUTE_TMEM_FENCED)
                chans = self.engine.input_ready(chans, new)
                chans = self.gemv(p, chans, env, op, layer, a, b, epoch, xbuf, warp, lane, row)
                if cutlass.const_expr(self.moe_batched):
                    if op == ROUTER:
                        self.route_batch(p, layer, epoch, tid, warp, lane)
                # The last split publishes the residual its consumers read: down's next h_in, and
                # o_proj's h_mid unless gate/up sums the partials itself (``sums_parts``).
                last = b == DOWN_SPLITS - 1
                reduces = op == DOWN
                if cutlass.const_expr(self.moe_batched):
                    reduces = op == OPROJ  # b is the expert's tile group; REDUCE rows sum down
                elif cutlass.const_expr(not self.sums_parts):
                    reduces = reduces | (op == OPROJ)
                if cutlass.const_expr(self.batch == 1):
                    if reduces & last & (tid < self.rows):
                        col = a * self.rows + tid
                        self.reduce_residual(p, op, layer, col, 0, 1, epoch, row)
                elif reduces & last:
                    # Every compute thread: column tid % rows, token group tid // rows.
                    col = a * self.rows + tid % self.rows
                    groups = COMPUTE_THREADS // self.rows
                    self.reduce_residual(p, op, layer, col, tid // self.rows, groups, epoch, row)
            work.advance()
        # Every CTA has read the epoch by the time it finishes; the last one advances it.
        if tid == 0:
            finished = cute.arch.atomic_add(
                p.state.iterator + 1, Int32(1), sem="acq_rel", scope="gpu"
            )
            if finished == ctas - 1:
                p.state[1] = Int32(0)
                p.state[0] = epoch
        return chans, work

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def kv_blocks(self, pos):
        """The KV block count of positions ``[0, pos)`` and the most blocks one split reads."""
        blocks = (pos + BLOCK - 1) // BLOCK
        return blocks, (blocks + self.splits - 1) // self.splits

    @cute.jit
    def row_active(self, p: MegakernelParams, op, layer, a, epoch):
        """False only for a batched MoE expert row with no tokens routed to its expert (every
        role skips it alike)."""
        active = True
        if cutlass.const_expr(self.moe_batched):
            if (op == GATEUP) | (op == DOWN):
                active = self.expert_count(p, layer, a, epoch) > 0
        return active

    @cute.jit
    def route_batch(self, p: MegakernelParams, layer, epoch, tid, warp, lane):
        """Batched MoE, in the router row after it published every token's logits: each warp
        routes whole tokens (``route``) into EXP, then thread e lists expert e's tokens in token
        order (count, entries t * top_k + rank, routing weights) for the expert rows."""
        k, batch = self.top_k, self.batch
        for t in cutlass.range(warp, batch, COMPUTE_WARPS, unroll=1):
            chosen = cute.make_rmem_tensor((k,), Int32)
            weights = cute.make_rmem_tensor((k,), Float32)
            self.route(p, layer, lane, epoch, chosen, weights, t)
            if lane == 0:
                for r in cutlass.range_constexpr(k):
                    p.exp[t * k + r] = Float32(chosen[r])
                    p.exp[batch * k + t * k + r] = weights[r]
        self.csync.sync(skip_fence="both", reason=COMPUTE_TMEM_FENCED)
        if tid < self.experts:
            count = Int32(0)
            base = (layer * self.experts + tid) * batch
            for t in cutlass.range(batch, unroll=1):
                for r in cutlass.range_constexpr(k):
                    if Int32(p.exp[t * k + r]) == tid:
                        publish(p.e_entry, base + count, Float32(t * k + r), epoch)
                        publish(p.e_weight, base + count, p.exp[batch * k + t * k + r], epoch)
                        count = count + 1
            publish(p.e_count, layer * self.experts + tid, Float32(count), epoch)
        # EXP is reused by the next row.
        self.csync.sync(skip_fence="before", reason=COMPUTE_TMEM_FENCED)

    @cute.jit
    def reduce_slots(self, p: MegakernelParams, layer, a, epoch, tid, row):
        """Batched MoE: next layer's residual columns a * rows .. of every token,
        bf16(h_mid + bf16(sum of its top_k experts' down outputs, in rank order)); thread tid
        takes column tid % rows of tokens tid // rows, + 2, ..."""
        col = a * self.rows + tid % self.rows
        groups = COMPUTE_THREADS // self.rows
        k = self.top_k
        for t in cutlass.range(tid // self.rows, self.batch, groups, unroll=1):
            vals = cute.make_rmem_tensor((k + 1,), Float32)
            segments = [(p.h_mid, self.hidden_index(t, layer, col), 1)]
            for r in cutlass.range_constexpr(k):
                segments.append((p.e_part, self.slot_index(t, layer, r, col), 1))
            with self.trace_region("reduce.poll", payload=row):
                fetch(segments, vals, epoch)
            total = vals[1]
            for r in cutlass.range_constexpr(2, k + 1):
                total = total + vals[r]
            h = bf16_round(vals[0] + bf16_round(total))
            publish(p.h_in, self.hidden_index(t, layer + 1, col), h, epoch)

    def slot_index(self, t, layer, rank, i):
        """e_part: token t's rank-th expert's down output."""
        return ((t * self.layers + layer) * self.top_k + rank) * self.hidden + i

    @cute.jit
    def attn_token(self, b):
        """An attention row's (token, KV split) from its b word, t * splits + split."""
        t = b // self.splits
        return t, b - t * self.splits

    @cute.jit
    def split_blocks(self, pos, split):
        """KV blocks ``[blk0, blk1)`` of positions ``[0, pos)`` that ``split`` reads."""
        blocks, per = self.kv_blocks(pos)
        blk0 = split * per
        blk1 = cutlass.min(blk0 + per, blocks)
        return blk0, cutlass.max(blk1, blk0)

    @property
    def sums_parts(self) -> bool:
        """At batch one gate/up (the router for MoE) sums h_in and o_proj's partials itself, and
        its row 0 publishes h_mid: one token's five words per element cost less than the hop
        through o_proj's last split. Batched and streamed decoding read h_mid from o_proj."""
        return self.batch == 1 and not self.streams_inputs

    @property
    def token_warps(self) -> int:
        """Compute warps per token in the prologues: all 8 at batch one, then 4, 2 and 1 as the
        batch fills the warps (8 // min(batch, 8) is a power of two)."""
        return COMPUTE_WARPS // min(self.batch, COMPUTE_WARPS)

    @cute.jit
    def prologue(self, p: MegakernelParams, op, layer, a, ks, epoch, xbuf, tid, warp, lane, row):
        """Fetch the row's inputs and write its GEMV input, one row per token, to ``xs[xbuf]``; the
        caller syncs. Each token is built by a group of ``token_warps`` warps: below 8 tokens the
        groups cover the batch in one round, so their block barriers line up."""
        xs = self.x_buffer(p, xbuf)
        if ((op == GATEUP) | (op == DOWN)) & self.moe_batched:
            self.gather_expert_inputs(p, op, layer, a, epoch, xs, tid, warp, lane, row)
        else:
            group_warps = self.token_warps
            threads = 32 * group_warps
            gt = (warp % group_warps) * 32 + lane  # thread within the token's group
            norm_weight = p.final_norm
            if op == QKV:
                norm_weight = p.ln_in[layer, None]
            elif (op == GATEUP) | (op == ROUTER):
                norm_weight = p.ln_post[layer, None]
            span = self.inter // DOWN_SPLITS
            groups = COMPUTE_WARPS // group_warps
            for t0 in cutlass.range(0, self.batch, groups, unroll=1):
                t = t0 + warp // group_warps
                active = t < self.batch
                if op == OPROJ:
                    if active:
                        self.combine_attention(p, layer, ks, t, epoch, xs, gt // 32, lane, row)
                elif op == DOWN:
                    if active:
                        base = self.act_index(t, layer, ks * span)
                        self.poll_input(p.act, base, span, xs, t, gt, threads, epoch, row)
                else:
                    self.normalize(
                        p, op, layer, a, t, active, xs, norm_weight, gt, group_warps, warp, lane,
                        epoch, row,
                    )  # fmt: skip
        if cutlass.const_expr(self.moe and not self.moe_batched):
            if (op == GATEUP) & (warp == 0):
                chosen = cute.make_rmem_tensor((self.top_k,), Int32)
                self.route(p, layer, lane, epoch, chosen, p.route[xbuf, None])

    @cute.jit
    def normalize(
        self,
        p: MegakernelParams,
        op,
        layer,
        a,
        t,
        active,
        xs,
        norm_weight,
        gt,
        group_warps: cutlass.Constexpr,
        warp,
        lane,
        epoch,
        row,
    ):
        """Token t's HF RMSNorm input of a QKV, gate/up, router or LM-head row into ``xs``
        (bf16(w * bf16(x * rsqrt))), by a group of ``group_warps`` warps (``gt``: the thread within
        it; ``active``: the group has a token). QKV and the LM head read h_in, gate/up and the router
        h_mid (with ``sums_parts`` the dense gate/up and MoE router sum it from h_in and o_proj's
        partials, row 0 publishing it); layer 0's QKV reads the embedding, and its row 0
        publishes h_in[0]."""
        threads = 32 * group_warps
        c = chunk_width(self.hidden, threads)
        rounds = self.hidden // (threads * c)
        # A thread's share stays in registers when it fits one poll (always at 8 warps per token,
        # so NORM rows never read back their global output); else it is staged through xs.
        held = rounds * c <= POLL_WORDS
        x = cute.make_rmem_tensor((rounds * c if held else c,), Float32)
        ss = Float32(0.0)
        if active:
            if (op == QKV) & (layer == 0):
                for r in cutlass.range_constexpr(rounds):
                    k0 = (r * threads + gt) * c
                    e0 = r * c if held else 0
                    for e in cutlass.range_constexpr(c):
                        x[e0 + e] = Float32(p.embed[p.step[t, 0], k0 + e])
                        if a == 0:
                            publish(p.h_in, self.hidden_index(t, 0, k0 + e), x[e0 + e], epoch)
                    if cutlass.const_expr(not held):
                        store_vec(xs, self.x_index(t, k0), x, c)
            elif self.sums_input(op):
                self.sum_parts(p, x, xs, t, layer, a, gt, threads, c, rounds, held, epoch, row)
            else:
                buf = p.h_mid if (op == GATEUP) | (op == ROUTER) else p.h_in
                base = self.hidden_index(t, layer, 0)
                if cutlass.const_expr(held):
                    segments = [(buf, base + (r * threads + gt) * c, c) for r in range(rounds)]
                    with self.trace_region("poll", payload=row):
                        fetch(segments, x, epoch)
                else:
                    self.poll_input(buf, base, self.hidden, xs, t, gt, threads, epoch, row)
            for r in cutlass.range_constexpr(rounds):
                if cutlass.const_expr(not held):
                    x = load_vec(xs, self.x_index(t, (r * threads + gt) * c), c)
                e0 = r * c if held else 0
                for e in cutlass.range_constexpr(c):
                    ss = ss + x[e0 + e] * x[e0 + e]
        ss = group_sum(ss, 32)
        if cutlass.const_expr(group_warps > 1):  # across the group's warps (every warp syncs)
            if lane == 0:
                p.red[warp] = ss
            self.csync.sync(skip_fence="both", reason=COMPUTE_TMEM_FENCED)
            first = warp // group_warps * group_warps
            ss = Float32(0.0)
            for i in cutlass.range_constexpr(group_warps):
                ss = ss + p.red[first + i]
        rinv = cute.math.rsqrt(ss / self.hidden + self.model.rms_eps)
        if active:
            for r in cutlass.range_constexpr(rounds):
                k0 = (r * threads + gt) * c
                if cutlass.const_expr(not held):
                    x = load_vec(xs, self.x_index(t, k0), c)
                e0 = r * c if held else 0
                y = cute.make_rmem_tensor((c,), Float32)
                wv = load_vec(norm_weight, k0, c)
                for e in cutlass.range_constexpr(c):
                    y[e] = bf16_round(wv[e] * bf16_round(x[e0 + e] * rinv))
                store_vec(xs, self.x_index(t, k0), y, c)
                if cutlass.const_expr(self.moe_batched):
                    if op == ROUTER:  # the expert rows gather it (gather_expert_inputs)
                        for e in cutlass.range_constexpr(c):
                            index = (t * self.layers + layer) * self.hidden + k0 + e
                            publish(p.x_mid, index, y[e], epoch)

    @cute.jit
    def gather_expert_inputs(
        self, p: MegakernelParams, op, layer, e, epoch, xs, tid, warp, lane, row
    ):
        """Batched MoE expert row: its token list into EXP, then input row i = entry i's token's
        normalized MLP input (gate/up) or that token's rank's activation slot (down). Rows past
        the count are stale; their MMA columns are discarded."""
        k, batch = self.top_k, self.batch
        entries = 2 * batch * k
        count = self.expert_count(p, layer, e, epoch)
        base = (layer * self.experts + e) * batch
        # The previous expert row's epilogue has read its list.
        self.csync.sync(skip_fence="both", reason=COMPUTE_TMEM_FENCED)
        for i in cutlass.range(tid, count, COMPUTE_THREADS, unroll=1):
            vals = cute.make_rmem_tensor((2,), Float32)
            fetch([(p.e_entry, base + i, 1), (p.e_weight, base + i, 1)], vals, epoch)
            p.exp[entries + i] = vals[0]
            p.exp[entries + batch + i] = vals[1]
        if tid == 0:
            p.exp[entries + 2 * batch] = Float32(count)
        self.csync.sync(skip_fence="both", reason=COMPUTE_TMEM_FENCED)
        for i in cutlass.range(warp, count, COMPUTE_WARPS, unroll=1):
            entry = Int32(p.exp[entries + i])
            t = entry // k
            if op == GATEUP:
                src = (t * self.layers + layer) * self.hidden
                self.poll_input(p.x_mid, src, self.hidden, xs, i, lane, 32, epoch, row)
            else:
                src = self.act_index(t, layer, (entry - t * k) * self.expert_inter)
                self.poll_input(p.act, src, self.expert_inter, xs, i, lane, 32, epoch, row)

    @cute.jit
    def sums_input(self, op):
        """Whether ``op``'s prologue sums h_mid from h_in and o_proj's partials (``sums_parts``)."""
        sums = False
        if cutlass.const_expr(self.sums_parts):
            mid = ROUTER if self.moe else GATEUP
            sums = op == mid
        return sums

    @cute.jit
    def sum_parts(
        self,
        p: MegakernelParams,
        x,
        xs,
        t,
        layer,
        a,
        gt,
        threads,
        c,
        rounds,
        held: cutlass.Constexpr,
        epoch,
        row,
    ):
        """bf16(h_in + bf16(sum of o_proj's partials)) for this thread's chunks, summed in split
        order, PART_ELEMENTS elements per poll: into ``x`` when ``held``, else staged chunk by chunk
        in ``xs``. Row 0 publishes it as h_mid."""
        per_poll = max(1, PART_ELEMENTS // c)
        with self.trace_region("poll", payload=row):
            for r0 in cutlass.range_constexpr(0, rounds, per_poll):
                n = min(per_poll, rounds - r0)
                vals = cute.make_rmem_tensor((n * c * (O_SPLITS + 1),), Float32)
                starts = [((r0 + r) * threads + gt) * c for r in range(n)]
                segments = [
                    seg
                    for k0 in starts
                    for seg in [(p.h_in, self.hidden_index(t, layer, k0), c)]
                    + [(p.o_part, self.part_index(t, layer, k, k0), c) for k in range(O_SPLITS)]
                ]
                fetch(segments, vals, epoch)
                for r in cutlass.range_constexpr(n):
                    v0 = r * c * (O_SPLITS + 1)
                    k0 = starts[r]
                    y = cute.make_rmem_tensor((c,), Float32)
                    for e in cutlass.range_constexpr(c):
                        total = vals[v0 + c + e]
                        for k in cutlass.range_constexpr(1, O_SPLITS):
                            total = total + vals[v0 + c * (k + 1) + e]
                        y[e] = bf16_round(vals[v0 + e] + bf16_round(total))
                        if cutlass.const_expr(held):
                            x[(r0 + r) * c + e] = y[e]
                        if a == 0:
                            publish(p.h_mid, self.hidden_index(t, layer, k0 + e), y[e], epoch)
                    if cutlass.const_expr(not held):
                        store_vec(xs, self.x_index(t, k0), y, c)

    @cute.jit
    def poll_input(
        self,
        buf,
        base,
        width: cutlass.Constexpr,
        xs,
        t,
        gt,
        threads: cutlass.Constexpr,
        epoch,
        row,
    ):
        """Columns ``base .. base + width`` of a tagged buffer into GEMV input row t by ``threads``
        threads: thread gt owns chunks ``r * threads + gt`` of ``chunk_width`` columns, polled
        POLL_WORDS words at a time."""
        c = chunk_width(width, threads)
        rounds = width // (threads * c)
        per_poll = max(1, POLL_WORDS // c)
        for r0 in cutlass.range_constexpr(0, rounds, per_poll):
            n = min(per_poll, rounds - r0)
            vals = cute.make_rmem_tensor((c * n,), Float32)
            segments = [(buf, base + ((r0 + r) * threads + gt) * c, c) for r in range(n)]
            with self.trace_region("poll", payload=row):
                fetch(segments, vals, epoch)
            for r in cutlass.range_constexpr(n):
                x = cute.make_rmem_tensor((c,), Float32)
                for e in cutlass.range_constexpr(c):
                    x[e] = vals[c * r + e]
                store_vec(xs, self.x_index(t, ((r0 + r) * threads + gt) * c), x, c)

    @cute.jit
    def combine_attention(self, p: MegakernelParams, layer, ks, t, epoch, xs, warp, lane, row):
        """Token t's o_proj input slice ``ks``, its heads merged from their KV-split partials and
        dealt to the token group's ``token_warps`` warps (``warp``: within the group). A unit is a head,
        or half a head when the group has more warps than heads; lane l of a unit merges dim
        pairs l (and l + 32 for a whole head), one poll per unit. Lane l also fetches split
        l % splits' (m, l) and the warp shares them by shuffles (splits <= 32)."""
        splits = self.splits
        halves = 2 if self.token_warps > self.heads_per_split else 1  # units per head
        per_unit = 2 // halves  # dim halves of a unit
        # Unrolled for one warp per token (few splits, a small body); a runtime loop otherwise, so
        # many splits' unrolled loops are one copy (eight measured 4x slower at 0.6B, 16 splits).
        units = self.heads_per_split * halves
        unroll = units if self.token_warps == 1 else 1
        for u in cutlass.range(warp, units, self.token_warps, unroll=unroll):
            hh = u // halves
            q_head = ks * self.heads_per_split + hh
            group = q_head // self.per_row
            head = q_head % self.per_row
            # This lane's split's (m, l), then the unit's dims (2 lane', 2 lane' + 1) per split.
            ml_base = self.attn_ml_index(t, layer, group, head, 0, 0)
            segments = [(p.attn_ml, ml_base + 2 * (lane % splits), 2)]
            for i in cutlass.range_constexpr(per_unit):
                d = (lane + 32 * (u % halves + i)) * 2
                for sp in cutlass.range_constexpr(splits):
                    index = self.attn_o_index(t, layer, group, sp, head, d)
                    segments.append((p.attn_o, index, 2))
            vals = cute.make_rmem_tensor((2 + 2 * per_unit * splits,), Float32)
            with self.trace_region("poll", payload=row):
                fetch(segments, vals, epoch)
            ms = cute.make_rmem_tensor((splits,), Float32)
            m = Float32(-1e30)
            for sp in cutlass.range_constexpr(splits):
                ms[sp] = cute.arch.shuffle_sync(vals[0], offset=sp)
                m = cutlass.max(m, ms[sp])
            den = Float32(0.0)
            for sp in cutlass.range_constexpr(splits):
                ms[sp] = cute.math.exp2(ms[sp] - m)  # split sp's scale
                den = den + cute.arch.shuffle_sync(vals[1], offset=sp) * ms[sp]
            for i in cutlass.range_constexpr(per_unit):
                d = (lane + 32 * (u % halves + i)) * 2
                num0 = Float32(0.0)
                num1 = Float32(0.0)
                for sp in cutlass.range_constexpr(splits):
                    o = 2 + (i * splits + sp) * 2
                    num0 = num0 + vals[o] * ms[sp]
                    num1 = num1 + vals[o + 1] * ms[sp]
                xs[self.x_index(t, hh * HEAD_DIM + d)] = BFloat16(num0 / den)
                xs[self.x_index(t, hh * HEAD_DIM + d + 1)] = BFloat16(num1 / den)

    def kv_head(self, group):
        """The kv head of attention row group ``group`` (q heads ``group * per_row``, ...)."""
        return group * self.per_row // self.group_size

    def qkv_index(self, t, layer, i):
        return (t * self.layers + layer) * self.qkv_rows + i

    def hidden_index(self, t, layer, i):
        """h_in, h_mid."""
        return (t * (self.layers + 1) + layer) * self.hidden + i

    def part_index(self, t, layer, split, i):
        """o_part, d_part."""
        return ((t * self.layers + layer) * O_SPLITS + split) * self.hidden + i

    def act_index(self, t, layer, i):
        return (t * self.layers + layer) * self.inter + i

    @cute.jit
    def reduce_residual(self, p: MegakernelParams, op, layer, col, group, groups, epoch, row):
        """Publish the next layer's residual column ``col`` = bf16(h_mid + bf16(sum of the
        down partials)), summed in split order like ``residual``.

        The last down split runs it after publishing its own partial: that split finishes
        last (its GATEUP rows are the second wave), and QKV and the LM head then fetch 1 word
        per element instead of 5."""
        base, parts, out, out_layer = p.h_mid, p.d_part, p.h_in, layer + 1
        if op == OPROJ:
            base, parts, out, out_layer = p.h_in, p.o_part, p.h_mid, layer
        if cutlass.const_expr(self.batch == 1):
            self.reduce_tokens(p, base, parts, out, layer, out_layer, col, 0, 1, epoch, row)
        else:
            # Token group ``group`` of ``groups`` takes REDUCE_TOKENS tokens per poll (one round
            # trip for all their words), every ``groups``-th such run.
            step = groups * REDUCE_TOKENS
            for t in cutlass.range(group * REDUCE_TOKENS, self.batch, step, unroll=1):
                self.reduce_tokens(
                    p, base, parts, out, layer, out_layer, col, t, REDUCE_TOKENS, epoch, row
                )

    @cute.jit
    def reduce_tokens(
        self, p, base, parts, out, layer, out_layer, col, t0, count: cutlass.Constexpr, epoch, row
    ):
        """Residual column col of tokens t0 .. t0 + count - 1 (those below the batch)."""
        vals = cute.make_rmem_tensor((count * (DOWN_SPLITS + 1),), Float32)
        segments = []
        for i in cutlass.range_constexpr(count):
            t = cutlass.min(t0 + i, self.batch - 1)
            segments.append((base, self.hidden_index(t, layer, col), 1))
            for k in cutlass.range_constexpr(DOWN_SPLITS):
                segments.append((parts, self.part_index(t, layer, k, col), 1))
        with self.trace_region("reduce.poll", payload=row):
            fetch(segments, vals, epoch)
        for i in cutlass.range_constexpr(count):
            total = vals[i * (DOWN_SPLITS + 1) + 1]
            for k in cutlass.range_constexpr(2, DOWN_SPLITS + 1):
                total = total + vals[i * (DOWN_SPLITS + 1) + k]
            h = bf16_round(vals[i * (DOWN_SPLITS + 1)] + bf16_round(total))
            if t0 + i < self.batch:
                publish(out, self.hidden_index(t0 + i, out_layer, col), h, epoch)

    def attn_o_index(self, t, layer, group, split, head, d):
        """``head``: within the attention row group."""
        index = ((t * self.layers + layer) * self.attn_groups + group) * self.splits + split
        return (index * self.per_row + head) * HEAD_DIM + d

    def attn_ml_index(self, t, layer, group, head, split, k):
        """k = 0: m (log2 domain), 1: l. Consecutive q heads' words are contiguous."""
        index = ((t * self.layers + layer) * self.attn_groups + group) * self.per_row + head
        return (index * self.splits + split) * 2 + k

    @cute.jit
    def gemv(
        self, p: MegakernelParams, chans, env, op, layer, a, b, epoch, xbuf, warp, lane, row_index
    ):
        """The row's 32-row weight tiles through ``self.engine``, each result published by
        ``epilogue`` (the LM head runs ``b`` tiles per row); returns the channel ends."""
        k_tiles, row_tiles = self.gemv_tiles(op, b)
        for t in cutlass.range(row_tiles, unroll=1):
            with self.trace_region("stream", payload=row_index):
                vals, row, chans = self.engine.tile(p, chans, env, xbuf, k_tiles, warp, lane)
            self.epilogue(p, op, layer, a, b, t, vals, row, epoch, xbuf, warp, lane)
            if cutlass.const_expr(self.shares_mlp):
                if op == GATEUP:  # every CTA's down rows bulk-read this act chunk
                    cute.arch.fence_proxy("async.global")
                    self.csync.sync(skip_fence="before", reason=COMPUTE_TMEM_FENCED)
                    if warp * 32 + lane == 0:
                        cute.arch.fence_acq_rel_gpu()
                        chunks = self.inter // SHARED_CHUNK
                        publish(p.act_flags, layer * chunks + a, Float32(0.0), epoch)
        return chans

    @cute.jit
    def gemv_tiles(self, op, b):
        """``(k_tiles, row_tiles)``: weight stages per tile and tiles of a GEMV row."""
        k_tiles = self.hidden_k_tiles
        if op == OPROJ:
            k_tiles = self.oproj_k_tiles
        if op == DOWN:
            k_tiles = self.down_k_tiles
        row_tiles = cutlass.select_(op == LMHEAD, b, 1)
        if cutlass.const_expr(self.moe_batched):
            for kind in (GATEUP, DOWN):
                expert_k, expert_rows = self.expert_tiles(kind)
                k_tiles = cutlass.select_(op == kind, expert_k, k_tiles)
                row_tiles = cutlass.select_(op == kind, expert_rows, row_tiles)
        return k_tiles, row_tiles

    @cute.jit
    def epilogue(
        self, p: MegakernelParams, op, layer, a, b, tile, vals, row, epoch, xbuf, warp, lane
    ):
        """Publish one tile's results: ``vals[t]`` is token t's dot product of tile row ``row``,
        valid on the lanes the engine says own it."""
        if cutlass.const_expr(self.moe_batched):
            if (op == GATEUP) | (op == DOWN):
                # Row b holds group b of the expert's tiles.
                per_group = self.expert_tiles(GATEUP)[1]
                per_group = cutlass.select_(op == DOWN, self.expert_tiles(DOWN)[1], per_group)
                first = b * per_group
                self.expert_epilogue(p, op, layer, first + tile, vals, row, epoch, warp, lane)
        for t in cutlass.range_constexpr(self.batch):
            v = vals[t]
            if cutlass.const_expr(self.moe_batched):
                if (op == GATEUP) | (op == DOWN):
                    op = -1  # published by expert_epilogue
            if op == GATEUP:
                # Tile rows alternate gate, up: the owner of gate row 2j gets up row 2j + 1's.
                up = self.engine.pair_value(v)
                if self.engine.owns_pair(warp, lane):
                    g = bf16_round(v)
                    silu = bf16_round(g / (1.0 + cute.math.exp2(-g * LOG2E)))
                    out = bf16_round(silu * bf16_round(up))
                    if cutlass.const_expr(self.moe):
                        # Fold slot s's routing weight in here, so down sums plain partials.
                        slot = a // (2 * self.expert_inter // self.rows)
                        out = bf16_round(out * p.route[xbuf, slot])
                    col = a * (self.rows // 2) + row // 2
                    if cutlass.const_expr(self.shares_mlp):
                        p.xshared[self.act_offset(layer) + self.x_index(t, col)] = BFloat16(out)
                    else:
                        publish(p.act, self.act_index(t, layer, col), out, epoch)
            elif self.engine.owns_row(warp, lane):
                if op == QKV:
                    index = self.qkv_index(t, layer, a * self.rows + row)
                    publish(p.qkv, index, bf16_round(v), epoch)
                elif op == OPROJ:
                    publish(p.o_part, self.part_index(t, layer, b, a * self.rows + row), v, epoch)
                elif op == DOWN:
                    publish(p.d_part, self.part_index(t, layer, b, a * self.rows + row), v, epoch)
                elif op == ROUTER:
                    index = (t * self.layers + layer) * self.experts + a * self.rows + row
                    publish(p.router_logits, index, bf16_round(v), epoch)
                elif op == LMHEAD:
                    p.logits[t, (a + tile) * self.rows + row] = bf16_round(v)

    @cute.jit
    def expert_epilogue(self, p: MegakernelParams, op, layer, tile, vals, row, epoch, warp, lane):
        """Batched MoE expert row: MMA column i is entry i of the expert's list (EXP): token
        t, rank r. Gate/up publishes t's activation slot r times its routing weight, down t's
        rank-r output."""
        k, batch = self.top_k, self.batch
        entries = 2 * batch * k
        count = Int32(p.exp[entries + 2 * batch])
        for i in cutlass.range_constexpr(batch):
            v = vals[i]
            entry = Int32(p.exp[entries + i])
            t = entry // k
            rank = entry - t * k
            if op == GATEUP:
                up = self.engine.pair_value(v)
                if self.engine.owns_pair(warp, lane) & (i < count):
                    g = bf16_round(v)
                    silu = bf16_round(g / (1.0 + cute.math.exp2(-g * LOG2E)))
                    out = bf16_round(silu * bf16_round(up))
                    out = bf16_round(out * p.exp[entries + batch + i])
                    col = rank * self.expert_inter + tile * (self.rows // 2) + row // 2
                    publish(p.act, self.act_index(t, layer, col), out, epoch)
            elif self.engine.owns_row(warp, lane) & (i < count):
                index = self.slot_index(t, layer, rank, tile * self.rows + row)
                publish(p.e_part, index, v, epoch)

    @cute.jit
    def publish_empty_partial(self, p: MegakernelParams, layer, group, split, t, epoch, tid):
        """An attention row whose split has no KV blocks at this position (short contexts leave
        most splits empty) and does not hold the new token (the last split): its partial is empty,
        m = -1e30 and l = 0 (zero weight in o_proj's merge) and O = 0, published at once without
        waiting for QKV. Such a split never appends the new K/V row: that is the split whose
        block holds pos, or the last split when pos starts a new block."""
        heads = self.per_row
        if tid < heads * HEAD_DIM:
            hq = tid // HEAD_DIM
            d = tid % HEAD_DIM
            publish(p.attn_o, self.attn_o_index(t, layer, group, split, hq, d), 0.0, epoch)
            if d == 0:
                ml = self.attn_ml_index(t, layer, group, hq, split, 0)
                publish(p.attn_ml, ml, Float32(-1e30), epoch)
                publish(p.attn_ml, ml + 1, Float32(0.0), epoch)

    @cute.jit
    def attention(
        self,
        p: MegakernelParams,
        w,
        layer,
        group,
        split,
        t,
        pos,
        epoch,
        tid,
        warp,
        lane,
        row_index,
    ):
        """Flash-decoding partial of token ``t``'s attention row group ``group`` (``per_row`` q
        heads of one kv head) over one KV split.

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
        max_len = p.k_cache.shape[0] // (self.batch * self.layers * self.kv_heads)
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
            fetch([(p.qkv, self.qkv_index(t, layer, column), 2)], raw, epoch)
        if cutlass.const_expr(q_words + 2 * HEAD_DIM < 2 * COMPUTE_THREADS):
            if 2 * tid == word:
                p.qkvs[word] = raw[0]
                p.qkvs[word + 1] = raw[1]
        else:
            p.qkvs[word] = raw[0]
            p.qkvs[word + 1] = raw[1]
        self.csync.sync(skip_fence="both", reason=COMPUTE_TMEM_FENCED)
        if warp < heads + 1:
            self.head_norm_rope(p.qkvs, warp * HEAD_DIM + 2 * lane, norm, cos, sin)
        self.csync.sync(skip_fence="both", reason=COMPUTE_TMEM_FENCED)
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
                with w.consume(skip_fence="both", reason=COMPUTE_TMEM_FENCED) as h:
                    sK = h.data["KV"]
                    for i in cutlass.range_constexpr(2):
                        pt = 8 * warp + 4 * i + g
                        kv = self.stage_row(sK, pt, d0)
                        valid = blk * BLOCK + pt < pos
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
                with w.consume(skip_fence="both", reason=COMPUTE_TMEM_FENCED) as h:
                    sV = h.data["KV"]
                    for i in cutlass.range_constexpr(2):
                        pt = 8 * warp + 4 * i + g
                        if blk * BLOCK + pt < pos:  # masked: p is 0 but stale v may be NaN
                            vv = self.stage_row(sV, pt, d0)
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
                row = ((t * self.layers + layer) * self.kv_heads + kvh) * max_len + pos
                # A batch's caches can pass 2^31 elements: offset the row in 64 bits.
                offset = Int64(row) * HEAD_DIM
                row_layout = cute.make_layout((HEAD_DIM,))
                k_row = cute.make_tensor(p.k_cache.iterator + offset, row_layout)
                v_row = cute.make_tensor(p.v_cache.iterator + offset, row_layout)
                for half in cutlass.range_constexpr(2):
                    kf = cute.make_rmem_tensor((8,), Float32)
                    vf = cute.make_rmem_tensor((8,), Float32)
                    for e in cutlass.range_constexpr(8):
                        kf[e] = k_new[half * 8 + e]
                        vf[e] = v_new[half * 8 + e]
                    store_vec(k_row, d0 + 64 * half, kf, 8)
                    store_vec(v_row, d0 + 64 * half, vf, 8)
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
        self.csync.sync(skip_fence="both", reason=COMPUTE_TMEM_FENCED)
        # Thread t merges dim t % HEAD_DIM of q head t // HEAD_DIM; with one q head, threads
        # HEAD_DIM.. sit out (their head's ATT columns do not exist).
        if cutlass.const_expr(heads * HEAD_DIM < COMPUTE_THREADS):
            if tid < heads * HEAD_DIM:
                self.merge_partials(p, layer, group, split, t, last, tid, epoch)
        else:
            self.merge_partials(p, layer, group, split, t, last, tid, epoch)
        # The next attention row rewrites att and qkvs.
        self.csync.sync(skip_fence="before", reason=COMPUTE_TMEM_FENCED)
        return w

    @cute.jit
    def merge_partials(self, p: MegakernelParams, layer, group, split, t, last, tid, epoch):
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
        publish(p.attn_o, self.attn_o_index(t, layer, group, split, hq, d), num, epoch)
        if d == 0:
            publish(p.attn_ml, self.attn_ml_index(t, layer, group, hq, split, 0), mx, epoch)
            publish(p.attn_ml, self.attn_ml_index(t, layer, group, hq, split, 1), den, epoch)

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
    def kernel(self, tma, gtensors, scratch, tensors, bound, engine_args):
        x_stride = self.x_stride
        layouts = {
            "W": self.weight_stage_layout(),
            "KV": cute.make_layout(
                (BLOCK, HEAD_DIM, self.stages), stride=(HEAD_DIM, 1, BLOCK * HEAD_DIM)
            ),
            "X": cute.make_layout((self.x_buffers, x_stride), stride=(x_stride, 1)),
            "RED": cute.make_layout((COMPUTE_WARPS,)),
            "ATT": cute.make_layout(
                (COMPUTE_WARPS + 1, self.att_stride), stride=(self.att_stride, 1)
            ),
            "QKVS": cute.make_layout(((self.per_row + 2) * HEAD_DIM,)),
            "ROUTE": cute.make_layout((2, self.route_width), stride=(self.route_width, 1)),
        }
        if cutlass.const_expr(self.moe_batched):
            layouts["EXP"] = cute.make_layout((self.exp_floats,))
        layouts.update(self.extra_layouts())
        self.begin_roles(work=bound, layouts=layouts)
        # Materialize the argument tensors here, in converged code: iterators and layouts are
        # created lazily at first use, and a first use inside one role's dispatch branch would
        # not dominate the other roles' uses.
        tensors = dict(zip(TENSOR_ARGS, (cute.make_tensor(t.iterator, t.layout) for t in tensors)))
        trace_buffer = tensors.pop("trace_buffer")
        parts = self.tma_partitions(tma, gtensors, engine_args)
        fields = dict(
            tma=tuple(tma),
            gmem=tuple(part[1] for part in parts),
            sW=StageView(parts[0][0], axis=1),
            sKV=StageView(parts[WEIGHT_ATOMS][0], axis=1),
            xs=self.buffer("X"),
            red=self.buffer("RED"),
            att=self.buffer("ATT"),
            qkvs=self.buffer("QKVS"),
            route=self.buffer("ROUTE"),
            exp=self.buffer("EXP") if self.moe_batched else self.buffer("ROUTE"),
            **tagged_views(scratch, self.scratch_layout),
            **tensors,
        )
        self.run_roles(self.make_params(fields, engine_args), trace_buffer=trace_buffer)

    def weight_stage_layout(self):
        return cute.make_layout((ROWS, TILE_K, self.stages), stride=(TILE_K, 1, ROWS * TILE_K))

    def extra_layouts(self) -> dict:
        """SMEM layouts of an engine's own buffers."""
        return {}

    def tma_partitions(self, tma, gtensors, engine_args):
        return tma_partitions(tma, gtensors, self.buffer("W"), self.buffer("KV"))

    def make_params(self, fields, engine_args):
        return MegakernelParams(**fields)

    def make_tma_atoms(self, weights, caches):
        """TMA atoms, their global tile tensors and the engine's extra kernel arguments."""
        tma, gtensors = make_tma_atoms(weights, caches)
        return tma, gtensors, ()

    @cute.jit
    def __call__(
        self,
        wqkv: cute.Tensor,
        wo: cute.Tensor,
        wgu: cute.Tensor,
        wd: cute.Tensor,
        embed: cute.Tensor,
        lm_head: cute.Tensor,
        router: cute.Tensor,
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
        xscratch: cute.Tensor,
        xshared: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        weights = (wqkv, wo, wgu, wd, lm_head, router)
        tma, gtensors, engine_args = self.make_tma_atoms(weights, (k_cache, v_cache))
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
            xscratch,
            xshared,
        )
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(tma, gtensors, scratch, tensors, bound, engine_args).launch(
            **self.launch_config(bound), stream=stream, cooperative=True
        )


# Plain Python, not cute.jit: loops in a jit body are staged, so building lists of atoms and
# views there leaks values out of the loop region.


def make_tma_atoms(weights, caches, w_smem=None):
    """TMA atoms and tile tensors: weight boxes of ``w_smem`` (default 32 x 256 row-major),
    64 x 128 KV-cache boxes."""
    op = cpasync.CopyBulkTensorTileG2SOp()
    if w_smem is None:
        w_smem = cute.make_layout((ROWS, TILE_K), stride=(TILE_K, 1))
    kv_smem = cute.make_layout((BLOCK, HEAD_DIM), stride=(HEAD_DIM, 1))
    w_tile = cute.product_each(w_smem.shape)
    pairs = [cpasync.make_tiled_tma_atom(op, src, w_smem, w_tile) for src in weights]
    pairs += [cpasync.make_tiled_tma_atom(op, src, kv_smem, (BLOCK, HEAD_DIM)) for src in caches]
    return tuple(a for a, _ in pairs), tuple(g for _, g in pairs)


def tma_partitions(tma, gtensors, sW, sKV, w_tile=(ROWS, TILE_K)):
    """``(smem, gmem)`` TMA partitions per atom: ``(TMA, stage)`` and ``(TMA, rows, cols)``."""
    parts = []
    for atom, g in zip(tma, gtensors):
        is_kv = len(parts) >= WEIGHT_ATOMS
        tile = (BLOCK, HEAD_DIM) if is_kv else w_tile
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
    layers, hidden, vocab = op.layers, op.hidden, op.vocab
    gate_up, down, router = packed_mlp_shapes(op)
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (layers * op.qkv_rows, hidden)),
        make_fake_compact_tensor(bf16, (layers * hidden, op.q_heads * HEAD_DIM)),
        make_fake_compact_tensor(bf16, gate_up),
        make_fake_compact_tensor(bf16, down),
        make_fake_compact_tensor(bf16, (vocab, hidden)),  # embedding
        make_fake_compact_tensor(bf16, (vocab, hidden)),  # LM head (the embedding when tied)
        make_fake_compact_tensor(bf16, router),
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(Int64, (op.scratch_layout.size,)),
        make_fake_compact_tensor(Float32, (op.batch, vocab)),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int32, (op.batch, 2), assumed_align=4),
        make_fake_compact_tensor(Int64, (cute.sym_int(),)),  # trace buffer; globaltimer only
        make_fake_compact_tensor(bf16, (layers, hidden)),
        make_fake_compact_tensor(bf16, (layers, hidden)),
        make_fake_compact_tensor(bf16, (layers, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (layers, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (hidden,)),
        make_fake_compact_tensor(bf16, (cute.sym_int(), 2, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (cute.sym_int(), op.x_slot), assumed_align=16),
        make_fake_compact_tensor(bf16, (op.shared_size,), assumed_align=16),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


# ---------------------------------------------------------------------- host side


def packed_mlp_shapes(op: QwenDecodeMegakernel) -> tuple[tuple, tuple, tuple]:
    """2-D shapes of the packed gate/up (rows interleaved), down and router weights. MoE stacks
    them per (layer, expert); a dense model's router is a one-tile stub."""
    layers, hidden = op.layers, op.hidden
    if op.moe:
        experts, inter = layers * op.experts, op.expert_inter
        return (experts * 2 * inter, hidden), (experts * hidden, inter), (experts, hidden)
    return (layers * 2 * op.inter, hidden), (layers * hidden, op.inter), (op.rows, hidden)


def check_model(model: QwenConfig):
    """Raise unless the kernel's tilings cover ``model``: 128-dim heads, o_proj splits of
    whole 256-column tiles, a hidden size the norms split into 4-element multiples per thread,
    whole 256-column down splits and 32-row LM-head tiles. Qwen3-4B fails the hidden size. MoE:
    32, 64, 128 or 256 experts (a routing warp's lanes each hold a power-of-two share) and
    expert widths of whole 256-column tiles."""
    if model.head_dim != HEAD_DIM:
        raise ValueError(f"head dim must be {HEAD_DIM}, got {model.head_dim}")
    if model.num_q_heads % model.num_kv_heads:
        raise ValueError("q heads must be a multiple of kv heads")
    if model.num_q_heads % (2 * O_SPLITS):
        raise ValueError(f"q heads must be a multiple of {2 * O_SPLITS}")
    if model.hidden_size % (4 * COMPUTE_THREADS):
        raise ValueError(f"hidden size must be a multiple of {4 * COMPUTE_THREADS}")
    if mlp_width(model) % (DOWN_SPLITS * COMPUTE_THREADS):
        raise ValueError(
            f"MLP width {mlp_width(model)} must be a multiple of {DOWN_SPLITS * COMPUTE_THREADS}"
        )
    if model.is_moe:
        if model.num_experts not in (32, 64, 128, 256):
            raise ValueError(f"need 32, 64, 128 or 256 experts, got {model.num_experts}")
        if model.moe_intermediate_size % TILE_K:
            raise ValueError(f"expert width must be a multiple of {TILE_K}")
    if model.vocab_size % ROWS:
        raise ValueError(f"vocabulary size must be a multiple of {ROWS}")
