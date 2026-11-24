"""One Inkling 2 Mini decode visit (attention half, then MoE MLP half) as one persistent
warpdrive launch (``InklingVisitKernel``) and its compile function; ``megakernel`` holds the
host side.

Batch one, one token. The launch implements ``ref.attention_block`` (``start=None``) and then
``ref.mlp_block`` on its residual. Its instruction table holds the attention rows
(``attn_half``), then the MLP rows (``mlp_half``), each half's message-gate rows first
(``common``); every CTA runs its share of both in that order::

    every CTA   attention prologue: block residuals of the layer's queries, RMSNorm
    attention   [MSGDOWN, GATEMIX q/k/v], QKV, ATTN, OPROJ; the last o_proj split's rows
                publish the post-attention residual as tagged words
    every CTA   MLP prologue, before its first MLP row: polls the whole residual, block
                residual over [blocks..., (h at block boundaries), residual], RMSNorm
    MLP         [MSGDOWN, GATEMIX MLP gate], ROUTER, GATEUP, DOWN; the last slot's rows
                write the visit's output

The MLP's input crosses CTAs as tagged words, not through a launch boundary. The load warp
runs ahead of the compute warps by the weight ring's depth: once it has issued its CTA's last
attention stages it streams the CTA's router tiles, then waits for the routing and streams
its experts' rows, while the compute warps still finish attention rows or wait for the
residual. At layer 0 (``embed_input``) every CTA recomputes the embedding row h in both
prologues; CTA 0 also stores it for later launches.

Roles of one CTA (9 warps), as ``qwen_megakernel``: warp 8 ``load`` streams 32 x 256 weight
stages and 64 x 128 KV blocks through one ring, warps 0-7 ``compute`` run ``gemv_tile`` over
the stages and publish tagged words ``(fp32 value, launch epoch)``; consumers spin on relaxed
loads until every word carries the epoch. The host deals the rows to CTAs in dependency order
(``megakernel.build_visit_table``) for ``StaticPersistent``; with every CTA resident, the
earliest unfinished row always has its producers finished (the MLP prologue's producers are
attention rows, which precede every MLP row of every CTA). The launch is cooperative, so the
driver starts it only with all of its one-CTA-per-SM grid resident and rejects a grid that
cannot be (CUDA error 720, e.g. under an MPS or green-context SM limit) instead of letting a
kernel on another stream hold SMs that spinning CTAs wait for. The last CTA to finish advances
the epoch, so one runner's launches may follow each other in a stream or a CUDA graph.

Quantized experts add a second ring, ``meta`` (load warp -> compute warps), for each expert
row's scales (``mlp_half``); ``InklingQuantVisitKernel``'s roles take its ends.

``full`` (sliding or full attention), ``messages``, ``experts_format`` and
``activation_format`` are constructor choices: 4 compiles per pair of formats
(``visit_kernel_class`` picks the class of a weight format); Inkling 2 Mini's layers take
(NVFP4, NVFP4) and (MXFP8, MXFP8).
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
from attn_gym.warpdrive.examples.inkling_megakernel.attn_half import (
    GATE,
    KDYN,
    QKVS_FLOATS,
    REL,
    SINK,
    VDYN,
    AttentionHalf,
    K,
    Q,
    V,
)
from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    GATE_TILES,
    GATEMIX,
    MAX_BLOCKS,
    MIX_FLOATS,
    MLP_GATE,
    MSGDOWN,
    REDUCTIONS,
    RING,
    WORDS,
    GateParams,
    InklingKernel,
    fake_gate_args,
    gate_atoms,
)
from attn_gym.warpdrive.examples.inkling_megakernel.mlp_half import (
    META_STAGES,
    ROUTER,
    STAGE_WORDS,
    MlpHalf,
    fake_expert_args,
)
from attn_gym.warpdrive.examples.megakernel.constants import (
    BACKOFF_NS,
    BLOCK,
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    HEAD_DIM,
    RED_VALUES,
    ROWS,
    STAGE_BYTES,
    TILE_K,
)
from attn_gym.warpdrive.examples.megakernel.formats import BF16, NVFP4
from attn_gym.warpdrive.examples.megakernel.ring import kv_stages, ring_stages, tma_partition
from attn_gym.warpdrive.examples.megakernel.table import InstructionTable
from attn_gym.warpdrive.examples.megakernel.tagged import (
    advance_epoch,
    fetch,
    load_words,
    scratch_offsets,
    tagged_views,
    unpack_tagged,
)

# Register budgets (setmaxnreg per warpgroup; the launch pads to 12 warps, 168 registers each):
# 9 warps would cap every warp at 168, 3 warps sharing a sub-partition's 16384 registers.
COMPUTE_REGS, LOAD_REGS = 232, 40
# VisitParams fields passed to the kernel as one tuple of argument tensors, in this order.
TENSOR_ARGS = (
    "trace_buffer", "state", "pos", "token", "embed", "embed_norm", "h", "attn_query",
    "attn_norms", "q_scale", "rel_scale", "rel_proj", "k_taps", "v_taps", "attn_conv", "ring_k",
    "ring_v", "ring_attn", "k_cache", "v_cache", "mlp_query", "post_norm", "mlp_conv",
    "router_bias", "expert_scales", "ring_mlp", "mhalf", "messages", "out",
)  # fmt: skip


@dataclass(frozen=True)
class VisitParams:
    tma_attn: tuple  # TMA atoms: projection matrices (``matrices`` order), wo
    gmem_attn: tuple  # their tile partitions: (TMA, row tiles, k tiles)
    tma_kv: tuple  # k and v cache
    gmem_kv: tuple
    tma_mlp: tuple  # router, gate/up, down
    gmem_mlp: tuple
    sW: StageView | None  # weight stage partition
    sKV: StageView  # KV stage partition
    sQW: StageView | None  # quantized expert weight stage partition
    sWS: StageView | None  # multi-token launches: swizzled weight stage partitions (bf16,
    sQWS: StageView | None  # quantized; ``multi_kernel``)
    sS: StageView | None  # ``meta`` stage partition: scales
    xs: cute.Tensor  # (x_rows, hidden) bf16 GEMV inputs (see ``InklingVisitKernel.x_rows``)
    red: cute.Tensor  # (REDUCTIONS, COMPUTE_WARPS, RED_VALUES) fp32 prologue reductions
    mix: cute.Tensor  # (MIX_FLOATS,) fp32 the last prologue's block-residual weights (messages)
    qkvs: cute.Tensor  # (QKVS_FLOATS,) fp32 an attention item's head vectors
    att: cute.Tensor  # (COMPUTE_WARPS + 1, HEAD_DIM + 2) fp32 per-warp flash states
    route: cute.Tensor  # (route_rows, 2 * top_k) fp32 routing by slot (``MlpHalf``)
    amax: cute.Tensor  # (amax_floats,) fp32 the quantized expert inputs' amax (``MlpHalf``)
    scales: cute.Tensor  # (2, scale_columns) fp32 ``expert_scales`` in SMEM (``MlpHalf``)
    uni: cute.Tensor | None  # multi-token launches: the expert union (``multi_kernel``)
    # Tagged int64 intermediates (``scratch_shapes``).
    proj: cute.Tensor
    attn_o: cute.Tensor
    attn_ml: cute.Tensor
    o_part: cute.Tensor
    kv_new: cute.Tensor
    resid: cute.Tensor  # (hidden,) the post-attention residual: the MLP half's input
    logits: cute.Tensor
    act: cute.Tensor
    down: cute.Tensor
    gate: GateParams | None  # message gates (``messages`` kernels)
    state: cute.Tensor  # int32 (2,): launch epoch, finished CTAs
    pos: cute.Tensor  # int32 (1,) position of the token
    token: cute.Tensor  # int64 (1,) the token (read when ``embed_input``)
    embed: cute.Tensor  # (vocab rows, hidden) bf16 embedding table
    embed_norm: cute.Tensor  # (hidden,) bf16
    h: cute.Tensor  # (hidden,) bf16 visit input; written by CTA 0 when ``embed_input``
    blocks: tuple  # MAX_BLOCKS (hidden,) bf16: the MLP half's completed blocks (below)
    num_blocks: Int32  # the attention half's blocks: the first num_blocks of ``blocks``
    queries: Int32  # 0: the attention input is h itself (layer 0), 1: block residuals
    boundary: Int32  # 1: the layer starts a block: the residual restarts from the attention
    # output and the MLP half reads h as block num_blocks
    embed_input: Int32  # 1: h is the token's normalized embedding row (layer 0)
    attn_query: cute.Tensor  # (queries, hidden) fp32 block-residual queries
    attn_norms: cute.Tensor  # (streams, hidden) bf16 input RMSNorm weights
    q_scale: cute.Tensor  # (q_heads, HEAD_DIM) bf16
    rel_scale: cute.Tensor  # (q_heads, d_rel) bf16
    rel_proj: cute.Tensor  # (rel_width, d_rel) bf16: ref's rel_proj transposed, a row per distance
    k_taps: cute.Tensor  # sliding: (kv_dim, kernel) fp32; full: (kernel, kv_dim) bf16 static
    v_taps: cute.Tensor
    attn_conv: cute.Tensor  # (hidden, kernel) fp32 post-attention taps
    ring_k: cute.Tensor  # (RING, kv_dim) bf16 the visit's convolution input rings
    ring_v: cute.Tensor
    ring_attn: cute.Tensor  # (RING, hidden) bf16
    k_cache: cute.Tensor  # (kv_heads * max_len, HEAD_DIM) bf16 the visit's keys
    v_cache: cute.Tensor
    mlp_query: cute.Tensor  # (1, hidden) fp32 MLP block-residual query
    post_norm: cute.Tensor  # (1, hidden) bf16 post-attention RMSNorm weight
    mlp_conv: cute.Tensor  # (hidden, kernel) fp32 post-MLP taps, the last on the current token
    router_bias: cute.Tensor  # (experts,) fp32
    # (2, experts + 1) fp32: per matrix (w13, w2) the experts' NVFP4 weight amax, then
    # rcp(1536 * weight denominator) (``megakernel.expert_scales``)
    expert_scales: cute.Tensor
    ring_mlp: cute.Tensor  # (RING, hidden) bf16
    mhalf: cute.Tensor  # (GATES * GATE_RANK,) fp32 message halves (read when ``messages``)
    messages: cute.Tensor  # (GATES, hidden) fp32
    out: cute.Tensor  # (hidden,) bf16 the visit's output
    # Tagged intermediates of multi-token launches only (``multi_kernel``).
    union: cute.Tensor | None = None  # the expert union
    o_in: cute.Tensor | None = None  # the merged o_proj inputs
    count: cute.Tensor | None = None  # finished ATTN items per q head (untagged counters)
    xin: cute.Tensor | None = None  # the shared prologues' input rows (untagged)
    xmix: cute.Tensor | None = None  # their MIX values (untagged)
    xamax: cute.Tensor | None = None  # the router inputs' amax (``multi_kernel.own_rows``)
    flags: cute.Tensor | None = None  # their tokens' flags


class InklingVisitKernel(AttentionHalf, MlpHalf, InklingKernel):
    """One decode visit (see the module docstring). Constructor arguments are the model sizes
    and constants the kernel reads (``AttentionHalf.init_attention``, ``MlpHalf.init_mlp``);
    ``messages``: a core visit after the first pass, whose q, k, v and MLP inputs pass through
    message gates; ``experts_format``: the experts' format and ``activation_format`` the
    quantization of their inputs (``mlp_half``); ``stages``: weight
    ring depth in 16 KB stages."""

    quantized_roles = False  # whether the roles take the ``meta`` ring

    def __init__(
        self,
        hidden: int,
        q_heads: int,
        kv_heads: int,
        d_rel: int,
        rel_width: int,
        extent: int,
        conv_kernel: int,
        rms_eps: float,
        qk_clip: float,
        rel_clip: float,
        splits: int,
        full: bool,
        width: int,
        experts: int,
        top_k: int,
        gate_softcap: float,
        up_softcap: float,
        route_scale: float,
        window: int = 0,
        log_alpha: float = 0.0,
        log_floor: int = 0,
        messages: bool = False,
        experts_format: str = BF16,
        activation_format: str = NVFP4,
        stages: int | None = None,
    ):
        if hidden % COMPUTE_THREADS or hidden % TILE_K:
            raise ValueError(f"hidden size must be a multiple of {COMPUTE_THREADS} and {TILE_K}")
        self.hidden, self.conv_kernel, self.rms_eps = hidden, conv_kernel, rms_eps
        self.hidden_k_tiles = hidden // TILE_K
        # Gates 0-2 read the attention prologue's queries (full layers: one per stream), the
        # MLP gate the MLP prologue's.
        self.init_messages(messages, (0, 1, 2, 0) if full else (0, 0, 0, 0))
        self.init_attention(
            q_heads, kv_heads, d_rel, rel_width, extent, qk_clip, rel_clip, splits, full,
            window, log_alpha, log_floor,
        )  # fmt: skip
        self.init_mlp(
            width, experts, top_k, gate_softcap, up_softcap, route_scale, experts_format,
            activation_format,
        )  # fmt: skip
        if self.quantized != self.quantized_roles:
            raise ValueError(
                f"{experts_format} experts need visit_kernel_class({experts_format!r})"
            )
        fixed = (
            2 * self.x_rows * (hidden + self.x_pad)
            + 4 * (REDUCTIONS * COMPUTE_WARPS * RED_VALUES + self.mix_floats + self.qkvs_floats)
            + 4 * (COMPUTE_WARPS + 1) * (HEAD_DIM + 2)
            + 4 * (self.route_rows * 2 * top_k + self.amax_floats + 2 * self.scale_columns)
            + self.extra_smem_bytes
            + (META_STAGES * self.meta_stage_bytes if self.quantized else 0)
        )
        self.stages = ring_stages(fixed, stages)
        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", BFloat16, align=1024)
        kv = self.smem_plan.buffer("KV", BFloat16, align=1024, overlaps=w)
        stage_buffers = {"W": w, "KV": kv}
        if self.quantized:
            stage_buffers["QW"] = self.smem_plan.buffer("QW", Int32, align=1024, overlaps=w)
        stage_buffers |= self.extra_stage_buffers(w)
        self.smem_plan.buffer("X", BFloat16, align=16)
        self.smem_plan.buffer("RED", Float32, align=16)
        self.smem_plan.buffer("MIX", Float32, align=16)
        self.smem_plan.buffer("QKVS", Float32, align=16)
        self.smem_plan.buffer("ATT", Float32, align=16)
        # The load warp writes its routing while the compute warps may still run attention
        # rows: ROUTE overlaps nothing.
        self.smem_plan.buffer("ROUTE", Float32, align=16)
        self.smem_plan.buffer("AMAX", Float32, align=16)
        self.smem_plan.buffer("SCALES", Float32, align=16)
        self.extra_buffers()
        if self.quantized:
            meta = {"S": self.smem_plan.buffer("S", Int32, align=128)}
        # Compute reads stages with generic loads before TMA refills them; the layer emits the
        # fence.proxy.async on a Warps release of a Tma-filled ring.
        self.w = Ring(self.stages, Tma(STAGE_BYTES), Warps, buffer=stage_buffers)
        if self.quantized:
            self.meta = Ring(META_STAGES, Tma(self.meta_stage_bytes), Warps, buffer=meta)
        self.csync = NamedBarrier(threads=COMPUTE_THREADS)
        self.work = WorkStream(mapping=InstructionTable(words=WORDS), schedule=StaticPersistent())

    @property
    def x_rows(self) -> int:
        """Rows of X: the attention inputs (``streams``), then two o_proj input buffers; the
        MLP half reuses rows 0 (its input, the router's), 1-2 (activation buffers) and
        ``xq_row`` (the quantized input)."""
        return max(self.streams + 2, 4)

    @property
    def xq_row(self) -> int:
        """The X row of the quantized MLP input (``MlpHalf.quantize_input``)."""
        return 3

    # Sizes of the SMEM buffers MIX, QKVS, AMAX (fp32) and ROUTE (rows of 2 * top_k fp32), and
    # any further bytes a subclass declares (``multi_kernel``).
    mix_floats, qkvs_floats, amax_floats = MIX_FLOATS, QKVS_FLOATS, 2
    route_rows, extra_smem_bytes = 1, 0

    @property
    def scale_columns(self) -> int:
        """Columns of the SMEM copy of ``expert_scales`` (``MlpHalf.stage_scales``): all of
        them for NVFP4 activations, else one unused."""
        return self.experts + 1 if self.activation_format == NVFP4 else 1

    @property
    def scratch_shapes(self) -> dict[str, int]:
        """Tagged int64 words of each per-launch intermediate."""
        shapes = self.attention_scratch_shapes() | self.mlp_scratch_shapes()
        return shapes | self.gate_scratch_shapes()

    def extra_buffers(self):
        """Declares a subclass's further SMEM buffers (``extra_smem_bytes``)."""

    def extra_stage_buffers(self, w) -> dict:
        """A subclass's further views of the weight ring's stages (``w``: the stage buffer)."""
        return {}

    def weight_target(self) -> tuple:
        """The SMEM buffer and TMA box of the attention's weight atoms (``weight_atom``)."""
        return "W", (ROWS, TILE_K)

    def extra_layouts(self) -> dict:
        """Layouts of ``extra_buffers``."""
        return {}

    def get_name(self) -> str:
        kind = ("full" if self.full else f"swa{self.window}") + ("_msg" if self.messages else "")
        return (
            f"warpdrive_inkling_visit_{kind}_h{self.hidden}_q{self.q_heads}kv{self.kv_heads}"
            f"_kv{self.splits}_i{self.width}_e{self.experts}k{self.top_k}"
            f"{'' if self.experts_format == BF16 else '_' + self.experts_format}"
            f"{'' if self.activation_format == self.experts_format else '_a' + self.activation_format}"
            f"_s{self.stages}"
            f"{self.trace_key()}"
        )

    # ------------------------------------------------------------------ roles

    @role(warps=COMPUTE_WARPS, regs=LOAD_REGS)
    @cute.jit
    def load(self, p: VisitParams, *, w: Producer, work: WorkCursor):
        w, _, work = self.load_rows(p, w, None, work)
        return w, work

    @role(warps=range(COMPUTE_WARPS), regs=COMPUTE_REGS)
    @cute.jit
    def compute(self, p: VisitParams, *, w: Consumer, work: WorkCursor, me: Local):
        w, _, work = self.compute_rows(p, w, None, work, me)
        return w, work

    @cute.jit
    def load_rows(self, p: VisitParams, w, meta, work):
        """The load warp's rows; ``meta``: the ``meta`` ring's producer (quantized experts),
        else None. Returns the ends."""
        if cutlass.const_expr(self.quantized):
            w = w.views(W=p.sW, KV=p.sKV, QW=p.sQW)
            meta = meta.views(S=p.sS)
        else:
            w = w.views(W=p.sW, KV=p.sKV)
        epoch = p.state[0] + 1
        pos = Int32(p.pos[0])
        lane = cute.arch.lane_idx()
        routed = Int32(0)
        while work.valid:
            op, a, b, _row = work.item
            if op < ROUTER:
                w = self.load_attention(p, w, op, a, b, pos)
            elif op < MSGDOWN:
                w, meta, routed = self.load_mlp(p, w, meta, op, a, b, routed, lane, epoch)
            if cutlass.const_expr(self.messages):
                if op >= MSGDOWN:
                    w = self.load_gate_item(p, w, op, a, b)
            work.advance()
        return w, meta, work

    @cute.jit
    def compute_rows(self, p: VisitParams, w, meta, work, me):
        """The compute warps' rows; ``meta``: the ``meta`` ring's consumer (quantized experts),
        else None. Returns the ends."""
        tid, warp, lane = me.tid, me.warp, me.lane
        epoch = p.state[0] + 1
        pos = Int32(p.pos[0])
        with self.trace_region("normalize"):
            self.attention_prologue(p, tid, warp, lane)
        self.stage_scales(p, tid)
        obuf = Int32(self.streams)
        held = Int32(-1)  # the slot whose activations the current DOWN buffer holds
        abuf = Int32(1)
        quantized = Int32(0)  # 1 once the MLP input is quantized
        mlp = Int32(0)  # 1 once the MLP prologue ran
        gathered = Int32(0)  # messages: 1 after the attention gather, 2 after the MLP's
        while work.valid:
            op, a, b, item = work.item
            if (mlp == 0) & self.is_mlp_row(op, a, b):
                with self.trace_region("mlp.prologue", payload=item):
                    self.mlp_prologue(p, epoch, tid, warp, lane)
                mlp = Int32(1)
            if cutlass.const_expr(self.messages):
                w, gathered = self.message_row(
                    p, w, op, a, b, gathered, epoch, tid, warp, lane, item
                )
            if op < ROUTER:
                w, obuf = self.attention_row(
                    p, w, op, a, b, item, obuf, pos, epoch, tid, warp, lane
                )
            elif op < MSGDOWN:
                w, meta, held, abuf, quantized = self.mlp_row(
                    p, w, meta, op, a, b, held, abuf, quantized, epoch, tid, warp, lane
                )
            work.advance()
        advance_epoch(p.state, epoch, tid)
        return w, meta, work

    # ------------------------------------------------------------ compute helpers

    @cute.jit
    def is_mlp_row(self, op, a, b):
        """Whether row ``(op, a, b)`` belongs to the MLP half: its own rows and the MLP gate's
        MSGDOWN and GATEMIX items."""
        mlp = (op >= ROUTER) & (op < MSGDOWN)
        if cutlass.const_expr(self.messages):
            mlp = mlp | ((op == MSGDOWN) & (a >= MLP_GATE * GATE_TILES))
            mlp = mlp | ((op == GATEMIX) & (b == MLP_GATE))
        return mlp

    @cute.jit
    def message_row(self, p, w, op, a, b, gathered, epoch, tid, warp, lane, item):
        """The compute warps' part of a message-gate row, or before a half's first own row the
        gather of its gated streams into its GEMV inputs (``gathered``: 1 after the attention's,
        2 after the MLP's). Returns the weight ring's consumer and ``gathered``."""
        if op == MSGDOWN:
            w = self.msgdown(p, w, a, b, epoch, warp, lane)
        elif op == GATEMIX:
            w = self.gatemix(p, w, a, b, epoch, warp, lane)
        elif (op < ROUTER) & (gathered == 0):
            with self.trace_region("gather", payload=item):
                self.gather(p, 0, self.streams, p.attn_norms, epoch, tid, warp, lane)
            gathered = Int32(1)
        elif (op >= ROUTER) & (gathered < 2):
            with self.trace_region("gather", payload=item):
                self.gather(p, MLP_GATE, 1, p.post_norm, epoch, tid, warp, lane)
            gathered = Int32(2)
        return w, gathered

    @cute.jit
    def mlp_prologue(self, p: VisitParams, epoch, tid, warp, lane):
        """Every CTA, before its first MLP row: the MLP input ``xs[0]`` from the block residual
        of the MLP query over the MLP's blocks (the attention's, plus h when the layer starts a
        block; at layer 0 the recomputed embedding) and the attention's residual, polled from
        its tagged words. The blocks and embedding load before the wait: warp 0 polls one word
        of each o_proj reducer's row tile, with backoff (a CTA waits here through attention's
        tail: every thread polling every word made attention rows ~1 us slower; GB300,
        2026-11-21), then the CTA fetches the whole residual."""
        per = self.hidden // COMPUTE_THREADS
        num_blocks = p.num_blocks + p.boundary
        # At layer 0 the only block is h, which CTA 0 may be storing: recomputed, not read.
        x = self.load_states(p, num_blocks - p.embed_input, None, tid)
        if p.embed_input != 0:
            self.embed_token(p, x, 0, False, tid, warp, lane)
        if warp == 0:
            rounds = -(-self.row_tiles // 32)
            last = (self.row_tiles - 1) * ROWS
            sentinels = cute.make_rmem_tensor((rounds,), Float32)
            segments = [(p.resid, cutlass.min((lane + 32 * i) * ROWS, last), 1)
                        for i in range(rounds)]  # fmt: skip
            fetch(segments, sentinels, epoch, uniform=True, backoff=BACKOFF_NS)
        self.csync.sync()
        resid = cute.make_rmem_tensor((per,), Float32)
        fetch(
            [(p.resid, r * COMPUTE_THREADS + tid, 1) for r in range(per)],
            resid,
            epoch,
            uniform=True,
        )
        for r in cutlass.range_constexpr(per):
            x[MAX_BLOCKS, r] = resid[r]
        self.normalize_states(
            p, x, num_blocks, p.mlp_query, p.post_norm, 1, Int32(1), tid, warp, lane
        )

    @cute.jit
    def stream_value(self, p: VisitParams, s, i):
        """Element ``i`` of gate ``s``'s input stream: the attention prologue's block residual
        for gates 0-2 (partial block h), the MLP prologue's for the MLP gate (partial block
        the residual, read again from its tagged words, complete once the prologue ran)."""
        query = self.gate_input(s)
        value = Float32(0.0)
        if s == MLP_GATE:
            resid, _tag = unpack_tagged(load_words(p.resid, i, 1)[0])
            value = self.mixed_state(p, query, p.num_blocks + p.boundary, resid, i)
        else:
            value = self.mixed_state(p, query, p.num_blocks, Float32(p.h[i]), i)
        return value

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(self, tma, gtensors, gate_tma, gate_g, scratch, tensors, blocks, bound, flags):
        layouts = (
            self.core_layouts(self.x_rows)
            | {
                "KV": kv_stages(self.stages),
                "MIX": cute.make_layout((self.mix_floats,)),
                "QKVS": cute.make_layout((self.qkvs_floats,)),
                "ATT": cute.make_layout(
                    (COMPUTE_WARPS + 1, HEAD_DIM + 2), stride=(HEAD_DIM + 2, 1)
                ),
                "ROUTE": cute.make_layout(
                    (self.route_rows, 2 * self.top_k), stride=(2 * self.top_k, 1)
                ),
                "AMAX": cute.make_layout((self.amax_floats,)),
                "SCALES": cute.make_layout(
                    (2, self.scale_columns), stride=(self.scale_columns, 1)
                ),
            }
            | self.extra_layouts()
        )
        if cutlass.const_expr(self.quantized):
            layouts["QW"] = cute.make_layout(
                (ROWS, STAGE_WORDS, self.stages), stride=(STAGE_WORDS, 1, ROWS * STAGE_WORDS)
            )
            layouts["S"] = cute.make_layout(
                (1, self.meta_subtiles, self.scale_words, META_STAGES),
                stride=(0, self.scale_words, 1, self.meta_subtiles * self.scale_words),
            )
        self.begin_roles(work=bound, layouts=layouts)
        # Materialize the argument tensors in converged code: a first use inside one role's
        # branch would not dominate the other roles' uses.
        tensors = {k: cute.make_tensor(t.iterator, t.layout) for k, t in zip(TENSOR_ARGS, tensors)}
        trace_buffer = tensors.pop("trace_buffer")
        blocks = tuple(cute.make_tensor(t.iterator, t.layout) for t in blocks)
        tagged = tagged_views(scratch, self.scratch_shapes)
        # tma: the attention weights (matrices, wo), the caches, the MLP's (``mlp_targets``).
        n_attn = len(self.matrices) + 1
        targets = [self.weight_target()] * n_attn + [("KV", (BLOCK, HEAD_DIM))] * 2
        targets += self.mlp_targets()
        parts, views = [], {}
        for atom, g, (name, box) in zip(tma, gtensors, targets):
            part = tma_partition(atom, self.buffer(name), g, box)
            parts.append(part)
            views.setdefault(name, StageView(part[0], axis=cute.rank(part[0]) - 1))
        gate = self.gate_params(self.buffer("W"), gate_tma, gate_g, tagged, tensors)
        if cutlass.const_expr("W" not in views and self.messages):
            # Multi-token launches load only GATEMIX's [F | G] rows through the linear view.
            part = tma_partition(gate_tma[1], self.buffer("W"), gate_g[1], (ROWS, TILE_K))
            views["W"] = StageView(part[0], axis=1)
        num_blocks, queries, boundary, embed_input = flags
        gmem = tuple(part[1] for part in parts)
        params = VisitParams(
            tma_attn=tuple(tma[:n_attn]),
            gmem_attn=gmem[:n_attn],
            tma_kv=tuple(tma[n_attn : n_attn + 2]),
            gmem_kv=gmem[n_attn : n_attn + 2],
            tma_mlp=tuple(tma[n_attn + 2 :]),
            gmem_mlp=gmem[n_attn + 2 :],
            sW=views.get("W"),
            sKV=views["KV"],
            sQW=views.get("QW"),
            sWS=views.get("WS"),
            sQWS=views.get("QWS"),
            sS=views.get("S"),
            xs=self.buffer("X"),
            red=self.buffer("RED"),
            mix=self.buffer("MIX"),
            qkvs=self.buffer("QKVS"),
            att=self.buffer("ATT"),
            route=self.buffer("ROUTE"),
            amax=self.buffer("AMAX"),
            scales=self.buffer("SCALES"),
            uni=self.buffer("UNI") if self.tokens > 1 else None,
            gate=gate,
            blocks=blocks,
            num_blocks=num_blocks,
            queries=queries,
            boundary=boundary,
            embed_input=embed_input,
            **tagged,
            **tensors,
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def __call__(
        self,
        wq: cute.Tensor,
        wk: cute.Tensor,
        wv: cute.Tensor,
        wrel: cute.Tensor,
        k_dynamic: cute.Tensor,
        v_dynamic: cute.Tensor,
        wgate: cute.Tensor,
        wsink: cute.Tensor,
        wo: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        attn_query: cute.Tensor,
        attn_norms: cute.Tensor,
        q_scale: cute.Tensor,
        rel_scale: cute.Tensor,
        rel_proj: cute.Tensor,
        k_taps: cute.Tensor,
        v_taps: cute.Tensor,
        attn_conv: cute.Tensor,
        router: cute.Tensor,
        w13: cute.Tensor,
        w2: cute.Tensor,
        s13: cute.Tensor,
        s2: cute.Tensor,
        router_bias: cute.Tensor,
        expert_scales: cute.Tensor,
        mlp_query: cute.Tensor,
        post_norm: cute.Tensor,
        mlp_conv: cute.Tensor,
        gate_down: cute.Tensor,
        gate_fg: cute.Tensor,
        mhalf: cute.Tensor,
        messages: cute.Tensor,
        embed: cute.Tensor,
        embed_norm: cute.Tensor,
        h: cute.Tensor,
        block0: cute.Tensor,
        block1: cute.Tensor,
        block2: cute.Tensor,
        block3: cute.Tensor,
        num_blocks: Int32,
        queries: Int32,
        boundary: Int32,
        embed_input: Int32,
        ring_k: cute.Tensor,
        ring_v: cute.Tensor,
        ring_attn: cute.Tensor,
        ring_mlp: cute.Tensor,
        token: cute.Tensor,
        pos: cute.Tensor,
        out: cute.Tensor,
        trace_buffer: cute.Tensor,
        scratch: cute.Tensor,
        state: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        by_matrix = {Q: wq, K: wk, V: wv, REL: wrel, KDYN: k_dynamic, VDYN: v_dynamic,
                     GATE: wgate, SINK: wsink}  # fmt: skip
        weights, caches = self.attention_atoms(by_matrix, wo, k_cache, v_cache)
        pairs = weights + caches + self.mlp_atoms(router, w13, w2, s13, s2)
        gate_pairs = []
        if cutlass.const_expr(self.messages):
            gate_pairs = gate_atoms(gate_down, gate_fg, self.hidden, self.weight_atom)
        bound = self.work.bind(table, workspace, max_ctas)
        tensors = (trace_buffer, state, pos, token, embed, embed_norm, h, attn_query, attn_norms,
                   q_scale, rel_scale, rel_proj, k_taps, v_taps, attn_conv, ring_k, ring_v,
                   ring_attn, k_cache, v_cache, mlp_query, post_norm, mlp_conv, router_bias,
                   expert_scales, ring_mlp, mhalf, messages, out)  # fmt: skip
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tuple(a for a, _ in pairs),
            tuple(g for _, g in pairs),
            tuple(a for a, _ in gate_pairs),
            tuple(g for _, g in gate_pairs),
            scratch,
            tensors,
            (block0, block1, block2, block3),
            bound,
            (num_blocks, queries, boundary, embed_input),
        ).launch(**self.launch_config(bound), stream=stream, cooperative=True)


class InklingQuantVisitKernel(InklingVisitKernel):
    """``InklingVisitKernel`` whose roles also take the ``meta`` ring of quantized experts
    (``experts_format`` "nvfp4" or "mxfp8")."""

    quantized_roles = True

    @role(warps=COMPUTE_WARPS, regs=LOAD_REGS)
    @cute.jit
    def load(self, p: VisitParams, *, w: Producer, meta: Producer, work: WorkCursor):
        return self.load_rows(p, w, meta, work)

    @role(warps=range(COMPUTE_WARPS), regs=COMPUTE_REGS)
    @cute.jit
    def compute(self, p: VisitParams, *, w: Consumer, meta: Consumer, work: WorkCursor, me: Local):
        return self.compute_rows(p, w, meta, work, me)


def visit_kernel_class(experts_format: str) -> type[InklingVisitKernel]:
    """The visit kernel class of experts in ``experts_format``."""
    return InklingVisitKernel if experts_format == BF16 else InklingQuantVisitKernel


@jit_cache
def compile_visit_kernel(op: InklingVisitKernel):
    """One compile per layer kind and experts format serves every visit, layer and position:
    weights, caches and states are arguments; the position, token, block count, query,
    boundary and embedding flags and the cache capacity are read at run time."""
    bf16, f32 = BFloat16, Float32
    h, qh, kv_dim = op.hidden, op.q_heads, op.kv_dim
    e = op.experts
    cache_rows = cute.sym_int(divisibility=BLOCK)
    taps = op.kv_heads * op.conv_kernel
    if op.full:  # static taps (kernel, channel), tap 0 the current token
        tap_shape, tap_dtype = (op.conv_kernel, kv_dim), bf16
    else:  # (channel, kernel), the last tap the current token
        tap_shape, tap_dtype = (kv_dim, op.conv_kernel), f32
    # Multi-token launches: tokens rows of every per-token vector, token-major.
    nt = op.tokens
    # Multi-token launches load 4 taps or ring channels in one vector (``load_conv_vectors``),
    # which TVM-FFI checks the alignment of at the call.
    tap_align, ring_align = (2, 2) if nt == 1 else (4 * tap_dtype.width // 8, 8)
    vector = [
        make_fake_compact_tensor(bf16, (nt * h,), assumed_align=2) for _ in range(MAX_BLOCKS)
    ]
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (qh * HEAD_DIM, h)),  # wq
        make_fake_compact_tensor(bf16, (kv_dim, h)),  # wk
        make_fake_compact_tensor(bf16, (kv_dim, h)),  # wv
        make_fake_compact_tensor(bf16, (qh * op.d_rel, h)),  # wrel
        make_fake_compact_tensor(bf16, (taps, h)),  # k_dynamic (unread in sliding layers)
        make_fake_compact_tensor(bf16, (taps, h)),  # v_dynamic
        make_fake_compact_tensor(bf16, (qh * HEAD_DIM, h)),  # wgate
        make_fake_compact_tensor(bf16, (qh, h)),  # wsink
        make_fake_compact_tensor(bf16, (h, qh * HEAD_DIM)),  # wo
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(bf16, (cache_rows, HEAD_DIM)),
        make_fake_compact_tensor(f32, (op.queries, h), assumed_align=4),  # attention queries
        make_fake_compact_tensor(bf16, (op.streams, h), assumed_align=2),  # attention norms
        make_fake_compact_tensor(bf16, (qh, HEAD_DIM), assumed_align=2),
        make_fake_compact_tensor(bf16, (qh, op.d_rel), assumed_align=2),
        make_fake_compact_tensor(bf16, (op.rel_width, op.d_rel), assumed_align=16),
        make_fake_compact_tensor(tap_dtype, tap_shape, assumed_align=tap_align),
        make_fake_compact_tensor(tap_dtype, tap_shape, assumed_align=tap_align),
        make_fake_compact_tensor(f32, (h, op.conv_kernel), assumed_align=4),  # attention conv
        make_fake_compact_tensor(bf16, (e, h)),  # router (bf16-exact copy)
        *fake_expert_args(op),
        make_fake_compact_tensor(f32, (e,), assumed_align=4),  # router bias
        make_fake_compact_tensor(f32, (2, e + 1), assumed_align=4),  # expert scales
        make_fake_compact_tensor(f32, (1, h), assumed_align=4),  # MLP query
        make_fake_compact_tensor(bf16, (1, h), assumed_align=2),  # post norm
        make_fake_compact_tensor(f32, (h, op.conv_kernel), assumed_align=4),  # MLP conv
        *fake_gate_args(h, nt),
        make_fake_compact_tensor(bf16, (cute.sym_int(), h)),  # embedding table
        make_fake_compact_tensor(bf16, (h,), assumed_align=2),  # embedding norm
        make_fake_compact_tensor(bf16, (nt * h,), assumed_align=2),  # h
        *vector,
        Int32(1),
        Int32(1),
        Int32(0),
        Int32(0),
        make_fake_compact_tensor(bf16, (RING, kv_dim), assumed_align=ring_align),
        make_fake_compact_tensor(bf16, (RING, kv_dim), assumed_align=ring_align),
        make_fake_compact_tensor(bf16, (RING, h), assumed_align=2),
        make_fake_compact_tensor(bf16, (RING, h), assumed_align=2),
        make_fake_compact_tensor(Int64, (nt,), assumed_align=8),  # token(s)
        make_fake_compact_tensor(Int32, (1,), assumed_align=4),  # pos (of the first token)
        make_fake_compact_tensor(bf16, (nt * h,), assumed_align=2),  # out
        make_fake_compact_tensor(Int64, (cute.sym_int(),)),  # trace buffer (globaltimer only)
        make_fake_compact_tensor(Int64, (scratch_offsets(op.scratch_shapes)[1],)),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )
