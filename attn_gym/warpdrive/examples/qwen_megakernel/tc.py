"""The tcgen05 swap-AB GEMV engine (``QwenDecodeMegakernelTc``) and streamed inputs
(``QwenDecodeMegakernelTcStream``) for the Qwen3 decode megakernel."""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    AFTER_FENCE_COVERED,
    Consumer,
    Local,
    Producer,
    Ring,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WorkCursor,
    role,
)
from attn_gym.warpdrive.examples.qwen_megakernel.core import (
    ATTN,
    COMPUTE_TMEM_FENCED,
    COMPUTE_WARPS,
    DOWN,
    DOWN_SPLITS,
    GATEUP,
    LMHEAD,
    NORM,
    QKV,
    QWEN3_0_6B,
    REDUCE,
    SHARED_CHUNK,
    STAGE_BYTES,
    MegakernelParams,
    QwenDecodeMegakernel,
    fetch,
    make_tma_atoms,
    mlp_width,
    publish,
    qkv_rows,
    tma_partitions,
)
from attn_gym.warpdrive.examples.qwen_megakernel.ref import QwenConfig

TC_MIN_N = 16  # MMA N >= 16 at M = 128; batch one uses column 0 (the others: shifted x)
TC_ROWS, TC_TILE_K = 128, 64  # a stage: one 128 x 64 SW128 K-major MMA operand (16 KB)
COMPUTE_TMEM_DRAINED = (
    "the compute warps issue no tcgen05.st and complete each acc load at its release"
)


@dataclass(frozen=True)
class TcParams(MegakernelParams):
    tiled_mma: cute.TiledMma
    a_layout: cute.ComposedLayout  # a weight stage as an MMA operand: 128 x 64 SW128 K-major
    acc_layout: cute.Layout  # TMEM accumulators: (MMA, MMA_M, MMA_N, stage)
    acc_fake: cute.Tensor  # one accumulator stage, for building the TMEM load


class TcGen05Gemv:
    """The GEMV engine on tcgen05 tensor cores, swap-AB: D (128 x N) = W x X^T with a 128-row
    weight tile as A and the GEMV inputs (one row per token) as B, accumulated in TMEM by the
    one-warp mma role; N is the batch rounded up to a power of two, at least 16.

    A: a weight stage is one 128 x 64 SW128 K-major operand, so a stage is four K=16 MMAs.
    B: SWIZZLE_NONE K-major over ``xs[xbuf]``, row t, column k at (k // 8) * S * 8 + 8t + k % 8
    with S = batch (``x_index``): core matrices 16 B * S apart. Rows past the batch read into the
    next column group; their D columns are discarded. At batch one the input is the contiguous
    vector the CUDA-core engine reads, row n is it shifted by 8n elements, and only column 0
    is used.
    D: TMEM lane i holds row i. Compute warps 0-3 load the tile: the one on TMEM subpartition
    s (hardware warp index % 4) loads rows 32s .. 32s + 31, lane l holding row 32s + l.

    Channels ``(w, xin, acc)``: ``xin`` hands each GEMV row's input to the mma warp, ``acc`` each
    tile's accumulator back. The compute warps release the weight stages without reading them."""

    def __init__(self, n: int):
        self.n = n  # MMA N: column t of D is token t

    def input_ready(self, chans, new):
        xin = chans[1]
        with xin.produce(skip_fence="before", reason=COMPUTE_TMEM_FENCED):
            pass
        return (chans[0], xin, *chans[2:])

    @cute.jit
    def tile(self, p: MegakernelParams, chans, env, xbuf, k_tiles, warp, lane):
        w, acc = chans[0], chans[2]
        (t_acc,) = env
        for _ in cutlass.range(k_tiles, unroll=1):
            with w.consume(skip_fence="both", reason=COMPUTE_TMEM_FENCED):
                pass
        vals = cute.make_rmem_tensor((self.n,), Float32)
        vals.fill(0.0)
        # A warp reaches TMEM lanes 32 * (its hardware warp % 4) ..: role-local warps 0-3 are
        # four consecutive warps, so they cover the four subpartitions wherever the role sits.
        row = 32 * (cute.arch.warp_idx() % 4) + lane
        with acc.consume() as h:
            if warp < 4:
                t2r = tcgen05.make_tmem_copy(
                    cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(self.n)), Float32),
                    p.acc_fake,
                ).get_slice(row)
                src = t2r.partition_S(t_acc[None, None, None, h.index])
                loaded = cute.make_rmem_tensor(t2r.partition_D(p.acc_fake).shape, Float32)
                cute.copy(t2r, src, loaded)
                for c in cutlass.range_constexpr(self.n):
                    vals[c] = loaded[c]
        return vals, row, (w, chans[1], acc, *chans[3:])

    def owns_row(self, warp, lane):
        return warp < 4

    def owns_pair(self, warp, lane):
        return (warp < 4) & (lane % 2 == 0)

    @cute.jit
    def pair_value(self, v):
        return cute.arch.shuffle_sync_down(v, offset=1)


class QwenDecodeMegakernelTc(QwenDecodeMegakernel):
    """The decode megakernel with the tcgen05 GEMV engine (``TcGen05Gemv``): work items are
    128-row tiles streamed in 128 x 64 stages, an mma warp (warp 9) issues the MMAs, and the
    compute warps prepare inputs and run the epilogues. Attention and the epilogues are the
    CUDA-core kernel's."""

    rows, tile_k = TC_ROWS, TC_TILE_K
    x_buffers = 1  # a row's MMAs finish before the compute warps write the next input
    max_batch = 64

    @property
    def tc_n(self) -> int:
        """MMA N: the batch rounded up to a power of two (the TMEM load's column count), >= 16."""
        return max(TC_MIN_N, 1 << (self.batch - 1).bit_length())

    @property
    def x_tail(self) -> int:
        """B rows past the stored ones read up to 8 elements per row into the next column group."""
        return max(self.tc_n - self.x_rows, 0) * 8 + 8

    def __init__(self, *args, model: QwenConfig = QWEN3_0_6B, **kwargs):
        check_tc_model(model)
        super().__init__(*args, model=model, **kwargs)
        self.engine = TcGen05Gemv(self.tc_n)
        self.xin = Ring(2, Warps, Umma, commit_fence="smem_async")
        self.acc = Ring(2, Umma, Warps, release_fence="tmem_load")
        self.tmem_plan = TmemPlan().region("acc", self.tc_n, 2)

    def weight_consumers(self):
        return {"mma": Umma, "compute": Warps}

    def get_name(self) -> str:
        return super().get_name() + "_tc"

    def weight_stage_layout(self):
        return cute.make_layout(
            (self.rows, self.tile_k, self.stages), stride=(self.tile_k, 1, STAGE_BYTES // 2)
        )

    def tma_partitions(self, tma, gtensors, engine_args):
        _, a_layout = engine_args
        ptr = cute.recast_ptr(self.buffer("W").iterator, a_layout.inner, dtype=BFloat16)
        stages = cute.make_tensor(ptr, self.weight_stage_layout())
        w_tile = (self.rows, self.tile_k)
        return tma_partitions(tma, gtensors, stages, self.buffer("KV"), w_tile)

    def make_tma_atoms(self, weights, caches):
        op = tcgen05.MmaF16BF16Op(
            BFloat16,
            Float32,
            (self.rows, self.tc_n, 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        tile = (self.rows, self.tc_n, self.tile_k)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, tile, BFloat16, 1)
        a_layout = cute.select(a_layout, mode=[0, 1, 2])
        w_smem = cute.make_composed_layout(
            a_layout.inner, 0, cute.make_layout((self.rows, self.tile_k), stride=(self.tile_k, 1))
        )
        tma, gtensors = make_tma_atoms(weights, caches, w_smem)
        return tma, gtensors, (tiled_mma, a_layout)

    def make_params(self, fields, engine_args):
        tiled_mma, a_layout = engine_args
        acc_shape = tiled_mma.partition_shape_C((self.rows, self.tc_n))
        return TcParams(
            **fields,
            tiled_mma=tiled_mma,
            a_layout=a_layout,
            acc_layout=tiled_mma.make_fragment_C(cute.append(acc_shape, 2)).layout,
            acc_fake=tiled_mma.make_fragment_C(acc_shape),
        )

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(
        self,
        p: MegakernelParams,
        *,
        w: Consumer,
        xin: Producer,
        acc: Consumer,
        work: WorkCursor,
        me: Local,
        tmem: Tmem,
    ):
        t_acc = cute.make_tensor(tmem.ptr("acc"), p.acc_layout)
        tmem.skip_teardown_drain(reason=COMPUTE_TMEM_DRAINED)
        chans, work = self.compute_loop(p, (w, xin, acc), work, me, env=(t_acc,))
        return (*chans, work)

    @role(warps=COMPUTE_WARPS + 1)
    @cute.jit
    def mma(
        self,
        p: MegakernelParams,
        *,
        w: Consumer,
        xin: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        w, xin, acc, _, work = self.mma_loop(p, w, xin, acc, None, work, tmem)
        return w, xin, acc, work

    @cute.jit
    def mma_loop(self, p: MegakernelParams, w, xin, acc, xa, work: WorkCursor, tmem: Tmem):
        """Per GEMV row: wait for its input, then per tile accumulate its weight stages into a
        TMEM stage; attention rows' KV stages are released untouched. B is the K tile of the SMEM
        input ``xs[xbuf]``, or with streamed inputs the activation chunk ``xload`` copied into ``xa``."""
        tiled_mma = p.tiled_mma
        t_acc = cute.make_tensor(tmem.ptr("acc"), p.acc_layout)
        tmem.skip_teardown_drain(reason="the MMA warp issues no tcgen05.ld or tcgen05.st")
        w_base = self.buffer("W").iterator
        s_rows = self.x_rows
        # B row n, column k at element (k // 8) * S * 8 + 8n + k % 8 (``x_index``). With one stored
        # row (S = 1) the core matrices overlap and row n is the input shifted by 8n elements.
        b_layout = cute.make_layout(
            (((8, self.tc_n // 8), (8, 2)), 1, 1), stride=(((8, 64), (1, 8 * s_rows)), 0, 0)
        )
        epoch = p.state[0] + 1  # batched MoE: expert rows with no tokens are skipped
        xkey = Int32(-1)
        xbuf = Int32(0)
        while work.valid:
            op, layer, a, b, _row = work.item
            if op == ATTN:
                t, split = self.attn_token(b)
                blk0, blk1 = self.split_blocks(p.step[t, 1], split)
                for _ in cutlass.range(2 * (blk1 - blk0), unroll=1):
                    with w.consume():
                        pass
            elif (op == REDUCE) | (op == NORM):
                pass
            elif self.row_active(p, op, layer, a, epoch):
                _new, xkey, xbuf = self.next_input(op, layer, a, b, xkey, xbuf)
                k_tiles, row_tiles = self.gemv_tiles(op, b)
                with xin.consume():
                    for _t in cutlass.range(row_tiles, unroll=1):
                        with acc.produce() as d:
                            dst = t_acc[None, None, None, d.index]
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            for kt in cutlass.range(k_tiles, unroll=1):
                                # Streamed inputs: the xa wait fences the MMAs.
                                skip = None if xa is None else "after"
                                with w.consume(skip_fence=skip, reason=AFTER_FENCE_COVERED) as h:
                                    a_stage = w_base + h.index * (STAGE_BYTES // 2)
                                    if cutlass.const_expr(xa is None):
                                        col = (xbuf % self.x_buffers) * self.x_stride
                                        x0 = p.xs.iterator + col + kt * self.tile_k * s_rows
                                        tiled_mma = self.stage_mmas(
                                            p, tiled_mma, dst, a_stage, x0, b_layout
                                        )
                                    else:
                                        with xa.consume() as hx:
                                            x0 = (
                                                self.buffer("XA").iterator
                                                + hx.index * self.xa_stage
                                            )
                                            tiled_mma = self.stage_mmas(
                                                p, tiled_mma, dst, a_stage, x0, b_layout
                                            )
            work.advance()
        return w, xin, acc, xa, work

    @cute.jit
    def stage_mmas(self, p: MegakernelParams, tiled_mma, dst, a_stage, x0, b_layout):
        """One weight stage's four K=16 MMAs into ``dst``: A the stage at ``a_stage``, B from ``x0``.
        Returns ``tiled_mma``: its accumulate flag is loop-carried state."""
        ptr = cute.recast_ptr(a_stage, p.a_layout.inner, dtype=BFloat16)
        frag_a = tiled_mma.make_fragment_A(cute.make_tensor(ptr, p.a_layout.outer))
        for kb in cutlass.range_constexpr(self.tile_k // 16):
            sb = cute.make_tensor(x0 + kb * 16 * self.x_rows, b_layout)
            frag_b = tiled_mma.make_fragment_B(sb)
            cute.gemm(tiled_mma, dst, frag_a[None, None, kb], frag_b[None, None, 0], dst)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
        return tiled_mma


# Activation-ring depth is clamped to [XA_MIN_STAGES, XA_MAX_STAGES] within the XA_BYTES SMEM
# budget. Each chunk holds 128 x batch bytes; its stage also includes the B operand's read-past
# tail. A deep ring overlaps chunk copies with the weight stream.
XA_MIN_STAGES, XA_MAX_STAGES, XA_BYTES = 4, 16, 32 * 1024


class TcGen05StreamGemv(TcGen05Gemv):
    """``TcGen05Gemv`` with the inputs in global memory (``QwenDecodeMegakernelTcStream``): a new
    input is fenced for the async proxy and handed to the load warp (``xready``)."""

    @cute.jit
    def input_ready(self, chans, new):
        w, xin, acc, xready = chans
        with xin.produce(skip_fence="before", reason=COMPUTE_TMEM_FENCED):
            pass
        if new:
            # The compute warps wrote the scratch with generic stores; the load warp reads it
            # with bulk copies (async proxy).
            cute.arch.fence_proxy("async.global")
            with xready.produce(skip_fence="before", reason=COMPUTE_TMEM_FENCED):
                pass
        return (w, xin, acc, xready)


class QwenDecodeMegakernelTcStream(QwenDecodeMegakernelTc):
    """``QwenDecodeMegakernelTc`` with GEMV inputs streamed from global memory.
    In the ``x_index`` layout, a 64-column chunk of the batch occupies one contiguous
    128 x batch byte span. A dedicated warp (``xload``, warp 10) bulk-copies these
    chunks into the activation ring (``xa``), which supplies the MMAs' B operand. Shared memory
    holds a few chunks rather than the whole input, leaving room for the weight ring.
    The weight stream remains independent of input readiness.

    Shared inputs (``shared_input``: QKV, the LM head and, for dense models, gate/up and down)
    are built once for all CTAs. NORM rows, one per token, normalize h_in and, for dense models,
    h_mid into ``xshared`` using the batch-one prologue's arithmetic. Dense gate/up epilogues
    write their activations there. NORM rows publish an epoch flag per token; gate/up publishes
    one per 64-column chunk. ``xload`` polls these flags. For dense models, each sequence's
    results are bit-identical to ``QwenDecodeMegakernelTc`` at batch one with equal splits.
    Other rows (o_proj, MoE router and experts) build their inputs in the CTA's ``xscratch``
    slot and hand them to ``xload`` through ``xready``."""

    streams_inputs = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = TcGen05StreamGemv(self.tc_n)
        xa = self.smem_plan.buffer("XA", BFloat16, align=128)
        self.xa = Ring(self.xa_stages, Tma(self.xa_bytes), Umma, buffer={"XA": xa})
        self.xready = Ring(2, Warps, Warps)

    @property
    def x_stride(self) -> int:
        return 8  # the SMEM input buffer is unused

    @property
    def shared_size(self) -> int:
        """Normalized inputs (ln_in, ln_post per layer, the final norm), then each layer's act."""
        return self.act_offset(self.layers)

    def norm_offset(self, slot):
        """Shared input slot 2 * layer (QKV), 2 * layer + 1 (gate/up) or 2 * layers (LM head)."""
        return slot * self.hidden * self.x_rows

    def act_offset(self, layer):
        return self.norm_offset(2 * self.layers + 1) + layer * self.inter * self.x_rows

    @property
    def x_slot(self) -> int:
        """bf16 elements of a CTA's input scratch: x_rows rows of x_width, in whole chunks."""
        return -(-self.x_rows * self.x_width // 64) * 64

    @property
    def xa_bytes(self) -> int:
        """Bytes of one 64-column chunk of the batch's inputs (one bulk copy)."""
        return self.tile_k * self.x_rows * 2

    @property
    def xa_stages(self) -> int:
        fit = XA_BYTES // (2 * self.xa_stage)
        return max(XA_MIN_STAGES, min(XA_MAX_STAGES, fit))

    @property
    def xa_stage(self) -> int:
        """bf16 elements per activation stage: a chunk plus the B operand's read-past tail."""
        return self.tile_k * self.x_rows + self.x_tail

    def fixed_smem_bytes(self) -> int:
        return super().fixed_smem_bytes() + 2 * self.xa_stages * self.xa_stage

    def extra_layouts(self) -> dict:
        return {"XA": cute.make_layout((self.xa_stage, self.xa_stages), stride=(1, self.xa_stage))}

    def get_name(self) -> str:
        return super().get_name() + "_stream"

    def x_buffer(self, p: MegakernelParams, xbuf):
        return p.xscratch[cute.arch.block_idx()[0], None]

    @cute.jit
    def shared_input(self, op):
        shared = Int32(0)
        if (op == QKV) | (op == LMHEAD):
            shared = Int32(1)
        if cutlass.const_expr(self.shares_mlp):
            if (op == GATEUP) | (op == DOWN):
                shared = Int32(1)
        return shared

    @cute.jit
    def norm_row(self, p: MegakernelParams, layer, phase, t, epoch, tid, warp, lane, row):
        """Write token t's normalized input to its shared slot, then publish its epoch flag.
        NORM row (layer, phase, t) builds the QKV input at phase 0, the gate/up input at phase 1,
        or the LM-head input at layer == layers, using the batch-one prologue's arithmetic.
        Layer 0, phase 0 also publishes h_in[0]."""
        slot = 2 * layer + phase
        xs = cute.make_tensor(
            p.xshared.iterator + self.norm_offset(slot),
            cute.make_layout((self.hidden * self.x_rows,)),
        )
        op, norm_weight = GATEUP, p.ln_post[layer, None]
        if layer == self.layers:
            op, norm_weight = LMHEAD, p.final_norm
        elif phase == 0:
            op, norm_weight = QKV, p.ln_in[layer, None]
        self.normalize(
            p, op, layer, 0, t, True, xs, norm_weight, tid, COMPUTE_WARPS, warp, lane, epoch, row
        )
        # Generic stores, then the flag: the consumers' bulk copies read them (async proxy).
        cute.arch.fence_proxy("async.global")
        # Also: every thread has read red before the next row rewrites it.
        self.csync.sync(skip_fence="before", reason=COMPUTE_TMEM_FENCED)
        if tid == 0:
            cute.arch.fence_acq_rel_gpu()
            publish(p.x_flags, slot * self.batch + t, Float32(0.0), epoch)

    @cute.jit
    def wait_flags(self, buf, start, count, epoch, lane):
        """Spin until flags ``buf[start .. start + count)`` carry the epoch (the warp splits them),
        then order the bulk copies after the producers' stores."""
        flag = cute.make_rmem_tensor((1,), Float32)
        for i in cutlass.range(lane, count, 32, unroll=1):
            fetch([(buf, start + i, 1)], flag, epoch)
        cute.arch.sync_warp()
        cute.arch.fence_acq_rel_gpu()
        cute.arch.fence_proxy("async.global")

    @cute.jit
    def issue_chunks(self, xa, atom, src0, k_tiles, row_tiles):
        """Copy the row's input chunks from ``src0`` into the activation ring in weight-stage order.
        Each chunk contains 64 columns across x_rows rows; stage i reads chunk i % k_tiles."""
        elements = self.tile_k * self.x_rows
        for i in cutlass.range(k_tiles * row_tiles, unroll=1):
            src = src0 + (i % k_tiles) * elements
            with xa.produce() as h, cute.arch.elect_one():
                stage = h.view("XA", write=True, by="tma")
                cute.copy(
                    atom,
                    cute.make_tensor(src, cute.make_layout((elements,))),
                    cute.make_tensor(stage.iterator, cute.make_layout((elements,))),
                    mbar_ptr=h.barrier,
                )
        return xa

    @role(warps=COMPUTE_WARPS + 2)
    @cute.jit
    def xload(self, p: MegakernelParams, *, xa: Producer, xready: Consumer, work: WorkCursor):
        """Stream each GEMV row's input chunks in MMA order.
        Shared inputs wait for producer flags; new per-CTA inputs wait for the compute warps'
        handoff. Reused inputs are streamed again. This dedicated warp lets the weight stream
        advance independently of input readiness."""
        epoch = p.state[0] + 1
        lane = cute.arch.lane_idx()
        scratch = p.xscratch.iterator + cute.arch.block_idx()[0] * self.x_slot
        atom = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), BFloat16)
        span = self.inter // DOWN_SPLITS
        xkey = Int32(-1)
        xbuf = Int32(0)
        while work.valid:
            op, layer, a, b, _row = work.item
            if (op != ATTN) & (op != REDUCE) & (op != NORM):
                if self.row_active(p, op, layer, a, epoch):
                    new, xkey, xbuf = self.next_input(op, layer, a, b, xkey, xbuf)
                    k_tiles, row_tiles = self.gemv_tiles(op, b)
                    if self.shared_input(op) != 0:
                        offset = Int32(0)
                        if op == DOWN:
                            chunks = self.inter // SHARED_CHUNK
                            first = layer * chunks + b * (span // SHARED_CHUNK)
                            self.wait_flags(p.act_flags, first, span // SHARED_CHUNK, epoch, lane)
                            offset = self.act_offset(layer) + b * span * self.x_rows
                        else:
                            slot = 2 * layer + cutlass.select_(op == GATEUP, 1, 0)
                            self.wait_flags(p.x_flags, slot * self.batch, self.batch, epoch, lane)
                            offset = self.norm_offset(slot)
                        xa = self.issue_chunks(
                            xa, atom, p.xshared.iterator + offset, k_tiles, row_tiles
                        )
                    else:
                        if new:
                            with xready.consume():
                                pass
                        xa = self.issue_chunks(xa, atom, scratch, k_tiles, row_tiles)
            work.advance()
        return xa, xready, work

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(
        self,
        p: MegakernelParams,
        *,
        w: Consumer,
        xin: Producer,
        acc: Consumer,
        xready: Producer,
        work: WorkCursor,
        me: Local,
        tmem: Tmem,
    ):
        t_acc = cute.make_tensor(tmem.ptr("acc"), p.acc_layout)
        tmem.skip_teardown_drain(reason=COMPUTE_TMEM_DRAINED)
        chans, work = self.compute_loop(p, (w, xin, acc, xready), work, me, env=(t_acc,))
        return (*chans, work)

    @role(warps=COMPUTE_WARPS + 1)
    @cute.jit
    def mma(
        self,
        p: MegakernelParams,
        *,
        w: Consumer,
        xin: Consumer,
        acc: Producer,
        xa: Consumer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        return self.mma_loop(p, w, xin, acc, xa, work, tmem)


def check_tc_model(model: QwenConfig):
    """Raise unless 128-row tiles and 64-column stages cover ``model`` (every Qwen3 shape the
    CUDA-core kernel covers does)."""
    sizes = {
        "q/k/v rows": qkv_rows(model),
        "hidden size": model.hidden_size,
        "gate/up rows": 2 * mlp_width(model),
        "vocabulary size": model.vocab_size,
    }
    if model.is_moe:
        sizes["experts"] = model.num_experts
        sizes["expert gate/up rows"] = 2 * model.moe_intermediate_size
    for what, size in sizes.items():
        if size % TC_ROWS:
            raise ValueError(f"tcgen05 engine: {what} {size} must be a multiple of {TC_ROWS}")
