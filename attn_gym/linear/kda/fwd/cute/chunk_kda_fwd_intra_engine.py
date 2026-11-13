# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SM100 warp-specialized persistent forward intra-chunk engine for KDA.

The engine computes BT64 Aqk, K3-compatible off-diagonal blocks, and raw 16x16 diagonal blocks over
(chunk, head) work units. Compute warps produce strip-rebased BF16 operands in shared memory, one
MMA warp accumulates in TMEM, and a separate Triton launch performs the diagonal Neumann inverses.

The per-channel decay lives inside the key-dimension reduction, so operands are rebased at each
16-token strip: ``2^(g_i - g_j)`` is split into a row factor ``2^(g_i - ref)`` and a key factor
``2^(ref - g_j)`` with ``ref = g_row0 - 64`` (``g`` is the chunk-local cumulative log2 gate,
``row0`` the strip's first row). At the supported gate bound a strip spans less than 128 log2
units, so every key factor and the row factor of every in-strip row lie in ``[2^-64, 2^64]``,
and BF16 operands stay finite while their Q/K entries are below ``2^63`` in magnitude. A row
factor and its gated operand ``q_d 2^(g_i - ref)`` (``k_d`` for Akk) leave the normal range only
below ``2^-126``, so for ``|k_d| >= 2^-62`` a term ``q_d k_d 2^(g_i - g_j)`` can be lost only when
its decay is below ``2^-62 / min(1, |q_d|)``: with model-scale Q/K entries (about 1 or less) the
flush-to-zero ``ex2`` loses nothing visible. The reference is causal and independent of the
sequence length.

Roles and channels of the warpdrive kernel::

    load (warp 8) --raw: 2 stages, TMA Q and K--> compute (warps 0-7) --op: 2 SMEM stages--> mma
                  --rawg: 1 stage, TMA gate---->         ^                                (warp 9)
                                                         +--------- acc: 2 TMEM stages ------+

Per (chunk, head) unit, ``raw``, ``rawg`` and ``acc`` carry one epoch and ``op`` one per 16-token
strip. Every CTA is persistent: CTA ``b`` of ``G`` owns units ``b, b + G, ...`` of the
``active chunks * H`` total, which the kernel reads on the device, so all three roles run the same
number of units.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch

# ---------------------------------------------------------------------------
# Tiny standalone Triton kernel: 16x16 unit-lower Neumann inverses of the
# engine's raw diag blocks (same log-depth factorization as the fwd forloop).
# ---------------------------------------------------------------------------
import triton
import triton.language as tl
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Float32, Int32

from attn_gym._backends.arch import runs_sm100_kernels
from attn_gym._backends.cute import TMA_ALIGNMENT_BYTES, normalize_compact_tensor
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear._delta_rule.triton.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear._delta_rule.triton.chunk_scheduler import (
    load_ragged_chunk_work as _tl_load_ragged_chunk_work,
)
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import load_ragged_chunk_work
from attn_gym.warpdrive import (
    Consumer,
    Local,
    NamedBarrier,
    Producer,
    ProtocolError,
    Ring,
    SmemPlan,
    StageView,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.cache import (
    compile_tvm_ffi,
    get_compile_target,
    jit_cache,
    make_fake_strided_tensor,
)
from attn_gym.warpdrive.compat import LayoutEnum


@triton.jit(do_not_specialize=["num_sequences"])
def _diag_neumann_inverse_kernel(
    akkd,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    H: tl.constexpr,
    BC: tl.constexpr,
    BLOCKS_PER_CHUNK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid, i_h = tl.program_id(0), tl.program_id(1).to(tl.int64)
    chunk = pid // BLOCKS_PER_CHUNK
    sub = pid % BLOCKS_PER_CHUNK
    if IS_VARLEN:
        if chunk >= tl.load(chunk_offsets + num_sequences):
            return
        _, _, token_start, valid = _tl_load_ragged_chunk_work(
            cu_seqlens, chunk_offsets, chunk, num_sequences, BLOCKS_PER_CHUNK * BC
        )
        row0 = token_start + sub * BC
        vsub = tl.minimum(tl.maximum(valid - sub * BC, 0), BC)
    else:
        row0 = chunk * (BLOCKS_PER_CHUNK * BC) + sub * BC
        vsub = BC
    o_r = tl.arange(0, BC)
    o_c = tl.arange(0, BC)
    ptrs = akkd + (row0 + o_r[:, None]).to(tl.int64) * (H * BC) + i_h * BC + o_c[None, :]
    m_r = o_r[:, None] < vsub
    b_a = tl.load(ptrs, mask=m_r, other=0.0)
    m_i = (o_r[:, None] == o_c[None, :]).to(tl.float32)
    b_n = -b_a
    b_p = b_n + m_i
    for _d in tl.static_range(3):
        b_n = tl.sum(b_n[:, :, None] * b_n[None, :, :], 1)
        b_p += tl.sum(b_p[:, :, None] * b_n[None, :, :], 1)
    tl.store(ptrs, b_p, mask=m_r)


def _kda_diag_neumann_inverse(
    akkd: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
    chunk_size: int,
) -> torch.Tensor:
    """Invert the engine's raw 16x16 unit-lower diagonal blocks in place."""
    _, tokens, heads, BC = akkd.shape
    assert BC == 16, f"diag Neumann inverse requires 16x16 blocks, got BC={BC}"
    blocks_per_chunk = chunk_size // BC
    if metadata is None:
        assert tokens % chunk_size == 0
        capacity = tokens // chunk_size
        cu_seqlens = chunk_offsets = None
        num_sequences = 0
    else:
        capacity = metadata.capacity
        cu_seqlens = metadata.cu_seqlens
        chunk_offsets = metadata.chunk_offsets
        num_sequences = cu_seqlens.shape[0] - 1
    _diag_neumann_inverse_kernel[(capacity * blocks_per_chunk, heads)](
        akkd,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        H=heads,
        BC=BC,
        BLOCKS_PER_CHUNK=blocks_per_chunk,
        IS_VARLEN=metadata is not None,
        num_warps=1,
    )
    return akkd


@dataclass(frozen=True)
class IntraFwdParams:
    """Kernel-prologue values the roles read."""

    mma16: cute.TiledMma  # one 16-token strip: [64, 16] += [64, K] @ [16, K].T
    atom_q: cute.CopyAtom
    desc_q: cute.Tensor  # TMA coordinates (T, K, (H, 1)), as desc_k and desc_g
    atom_k: cute.CopyAtom
    desc_k: cute.Tensor
    atom_g: cute.CopyAtom
    desc_g: cute.Tensor
    sQ: cute.Tensor
    sK: cute.Tensor
    sG: cute.Tensor
    sQg: cute.Tensor  # MMA A operands, (64, K, stage)
    sKgq: cute.Tensor
    sKgk: cute.Tensor  # MMA B operand, (16, K, stage)
    sBeta: cute.Tensor
    sAqkSt: cute.Tensor  # FP32 accumulators staged for the stores, (64, BT)
    sAkkSt: cute.Tensor
    acc_layout: cute.Layout  # both stages of one [64, BT] accumulator
    strip_layout: cute.Layout  # one [64, 16] strip of it
    g_beta: cute.Tensor
    g_aqk: cute.Tensor
    g_akkod: cute.Tensor
    g_akkd: cute.Tensor
    scale: Float32
    cu_seqlens: cute.Tensor | None
    chunk_offsets: cute.Tensor | None
    H: Int32
    first_unit: Int32  # this CTA's first (chunk, head) unit
    unit_stride: Int32  # launched CTAs
    num_units: Int32  # units this CTA owns


class KdaIntraFwdEngine(WarpSpecialized):
    """Forward intra stage on the engine architecture.

    Two S-shaped gemms over strip-rebased gated operands:

        Aqk[i, j] = scale * sum_d qg[i, d] * kgk[j, d]   (non-strict causal)
        Akk[i, j] = beta[i] * sum_d kgq[i, d] * kgk[j, d]  (strict lower)

    Warp plan: compute(0-7) production + drain/store, load(8) TMA, mma(9). No role declares a
    register budget: an active 208/48 split is slower on dense shapes. The 16x16 diagonal-block
    Neumann inverses (K4b's Akkd input) run in a separate tiny Triton kernel invoked by the
    Python wrapper.

    The roles run a hand-rolled persistent loop, not a ``WorkStream``. Compute is skewed by one
    unit: it produces unit ``it`` while the MMA of ``it - 1`` runs, then drains ``it - 1``, so it
    makes one more trip than load and mma. The unit count comes from the device
    (``chunk_offsets``), so a CUDA graph replays with new sequence boundaries.
    """

    raw_depth = 2  # q/k TMA prefetch depth (g stays depth-1: smem budget)
    op_depth = 2
    acc_depth = 2
    align = 1024
    rebase_shift = 64.0  # log2 units from a strip's row-start gate down to its rebase reference

    def __init__(
        self,
        num_heads: int,
        chunk_size: int = 64,
        head_dim: int = 128,
        varlen: bool = True,
        use_int64_offsets: bool = False,
    ):
        assert chunk_size == 64 and head_dim == 128
        self.BT = chunk_size
        self.BC = 16
        self.subchunks = self.BT // self.BC
        self.offdiag_blocks = self.subchunks * (self.subchunks - 1) // 2
        self.BK = head_dim
        self.io_type = cutlass.BFloat16
        self.acc_type = cutlass.Float32
        self.num_heads = num_heads
        self.varlen = varlen
        self.use_int64_offsets = use_int64_offsets
        # tcgen05 uses the same physical and logical M64 row count for BT64.
        self.mma_rows = 64
        self.s_tile = (self.mma_rows, self.BT, self.BK)
        self.cta_group = tcgen05.CtaGroup.ONE

        plan = self.smem_plan = SmemPlan()
        io = self.io_type
        raw_q = plan.buffer("Q", io, align=self.align)
        raw_k = plan.buffer("K", io, align=self.align)
        raw_g = plan.buffer("G", Float32, align=self.align)
        # Each operand has two layouts over the same bytes: the MMA's, and the row-major one the
        # compute threads store through.
        operands = {}
        for name in ("Qg", "Kgq", "Kgk"):
            operands[name] = plan.buffer(name, io, align=self.align)
            operands[f"{name}St"] = plan.buffer(f"{name}St", io, overlaps=operands[name])
        tile_bytes = self.BT * self.BK * io.width // 8
        # The compute threads read the raw tiles with ordinary loads, so each release takes the
        # derived fence.proxy.async before the next TMA fill; one arrival per warp.
        self.raw = Ring(
            self.raw_depth, Tma(2 * tile_bytes), Warps, buffer={"Q": raw_q, "K": raw_k}
        )
        self.rawg = Ring(1, Tma(self.BT * self.BK * Float32.width // 8), Warps, buffer=raw_g)
        self.op = Ring(self.op_depth, Threads, Umma, buffer=operands)
        # All 256 compute threads take every acc epoch although only 128 load TMEM: the other
        # half reads the staged copy next, and one role keeps one loop. tcgen05.wait::ld before
        # each release, so MMA cannot overwrite a stage still being read.
        self.acc = Ring(self.acc_depth, Umma, Threads, release_fence="tmem_load")
        # An M64 accumulator occupies half of the 128 TMEM lanes, and the fragment layout puts its
        # second stage in the other half of the same columns (a lane offset, stride 2^20). Both
        # ring stages of each accumulator therefore share one BT-column region, and the roles
        # index the stage through that layout instead of a plan slot.
        self.tmem_plan = TmemPlan().region("aqk", self.BT).region("akk", self.BT)
        # Accumulators staged in SMEM by the first 128 compute threads, read by all 256; then
        # the stores of one unit before the next unit restages.
        self.staged = NamedBarrier(threads=256)

    def get_name(self) -> str:
        return (
            f"kda_intra_engine_fwd_vl{int(self.varlen)}_h{self.num_heads}"
            f"_k{self.BK}_bt{self.BT}_i64{int(self.use_int64_offsets)}"
        )

    @cute.jit
    def upcast(self, value):
        """Promote an address operand before its first overflowing multiply."""
        return cutlass.Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    @cute.jit
    def unit(self, params: IntraFwdParams, it):
        """Chunk and head of this CTA's ``it``-th unit.

        Every role decodes its units here, so they visit the same units in the same order and
        each channel's producer and consumer agree on its epoch count.
        """
        w = params.first_unit + it * params.unit_stride
        return w // params.H, w % params.H

    @cute.jit
    def _decode_chunk(self, params: IntraFwdParams, chunk_idx):
        """Return the physical token start and valid rows for one chunk."""
        if cutlass.const_expr(self.varlen):
            _, _, token_start, valid = load_ragged_chunk_work(
                params.cu_seqlens, params.chunk_offsets, chunk_idx, Int32(self.BT)
            )
        else:
            token_start = chunk_idx * Int32(self.BT)
            valid = Int32(self.BT)
        return token_start, valid

    @cute.jit
    def _gated(self, x, scale, out, active):
        """``out[e] = scale[e] * 2^x[e]`` in the I/O dtype where ``active``, else 0."""
        for e in cutlass.range_constexpr(4):
            v = Float32(0.0)
            if active:
                v = Float32(scale[e]) * cute.math.exp2(x[e], fastmath=True)
            out[e] = self.io_type(v)

    @cute.jit
    def _produce_strip(
        self, op_h, sG, tsg_p, tsk_p, tsq_p, ep_b, cp_f32, cp_bf16, ep_row, ep_col0, valid, sc
    ):
        """Gate and store one 16-token strip's Qg, Kgq (rows >= row0) and Kgk (its 16 rows).

        All exponents are computed from the staged gates before any operand is gated; computing
        them inside the masked loop is slower on dense shapes.
        """
        row0 = sc * 16
        ref = cute.make_rmem_tensor((4,), self.acc_type)
        for e in cutlass.range_constexpr(4):
            ref[e] = sG[(row0, ep_col0 + e)] - Float32(self.rebase_shift)
        blocks = cute.size(tsg_p, mode=[1])
        x_row = cute.make_rmem_tensor((blocks, 4), Float32)
        x_key = cute.make_rmem_tensor((2, 4), Float32)
        # Row blocks 2 * sc and 2 * sc + 1 are the strip's own rows: their gates also give the
        # key exponents.
        for rb in cutlass.range_constexpr(2 * sc, blocks):
            r_g = cute.make_fragment_like(tsg_p[(None, 0, None)], Float32)
            cute.copy(cp_f32, tsg_p[(None, rb, None)], r_g)
            for e in cutlass.range_constexpr(4):
                x_row[rb, e] = r_g[e] - ref[e]
                if rb < 2 * sc + 2:
                    x_key[rb - 2 * sc, e] = ref[e] - r_g[e]

        tqg_p = ep_b.partition_D(op_h.view("QgSt", write=True))
        tkgq_p = ep_b.partition_D(op_h.view("KgqSt", write=True))
        tkgk_p = ep_b.partition_D(op_h.view("KgkSt", write=True))
        for rb in cutlass.range_constexpr(2 * sc, blocks):
            r_k = cute.make_fragment_like(tsk_p[(None, 0, None)], self.io_type)
            r_q = cute.make_fragment_like(r_k, self.io_type)
            cute.copy(cp_bf16, tsk_p[(None, rb, None)], r_k)
            cute.copy(cp_bf16, tsq_p[(None, rb, None)], r_q)
            r_qg = cute.make_fragment_like(r_k, self.io_type)
            r_kq = cute.make_fragment_like(r_k, self.io_type)
            prow = ep_row + rb * 8
            active = prow < valid and prow >= row0
            self._gated(x_row[rb, None], r_q, r_qg, active)
            self._gated(x_row[rb, None], r_k, r_kq, active)
            cute.copy(cp_bf16, r_qg, tqg_p[(None, rb, None)])
            cute.copy(cp_bf16, r_kq, tkgq_p[(None, rb, None)])
        for rb2 in cutlass.range_constexpr(2):
            r_kb = cute.make_fragment_like(tsk_p[(None, 0, None)], self.io_type)
            cute.copy(cp_bf16, tsk_p[(None, sc * 2 + rb2, None)], r_kb)
            r_kk = cute.make_fragment_like(r_kb, self.io_type)
            self._gated(x_key[rb2, None], r_kb, r_kk, row0 + ep_row + rb2 * 8 < valid)
            cute.copy(cp_bf16, r_kk, tkgk_p[(None, rb2, None)])

    # ------------------------------------------------------------------ roles

    @role(warps=range(8))
    @cute.jit
    def compute(
        self,
        params: IntraFwdParams,
        *,
        raw: Consumer,
        rawg: Consumer,
        op: Producer,
        acc: Consumer,
        tmem: Tmem,
        me: Local,
    ):
        """Produce gated operands and drain Aqk/Akk accumulators."""
        if cutlass.const_expr(me.warps != 8):
            raise ProtocolError(
                f"compute: the production and store partitions cover 256 threads, got {me.warps} "
                "warps"
            )
        BT = self.BT
        H, scale = params.H, params.scale
        sBeta, sAqkSt, sAkkSt = params.sBeta, params.sAqkSt, params.sAkkSt
        g_beta, g_aqk, g_akkod, g_akkd = params.g_beta, params.g_aqk, params.g_akkod, params.g_akkd
        n_iters = params.num_units
        # me.tid is already below 256; the modulo tells the compiler, which then folds the dense
        # row predicates of the production loop.
        local_tid = me.tid % me.threads
        # The M64 accumulator has one row pair per thread of a single warpgroup: only rows
        # 0-127 load it.
        tmem_row = me.tmem_row
        t_aqk_acc = cute.make_tensor(tmem.ptr("aqk"), params.acc_layout)
        t_akk_acc = cute.make_tensor(tmem.ptr("akk"), params.acc_layout)

        t2r_atom = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition(BT // 8), tcgen05.Pack.NONE),
            self.acc_type,
        )
        aqk_flat = t_aqk_acc[((None, None), 0, 0, None)]
        tc_s = tcgen05.make_tmem_copy(t2r_atom, aqk_flat[(None, None, 0)])
        sl_s = tc_s.get_slice(tmem_row % 128)
        p_t_aqk = sl_s.partition_S(aqk_flat)
        p_t_akk = sl_s.partition_S(t_akk_acc[((None, None), 0, 0, None)])
        sreg_shape = sl_s.partition_D(cute.make_identity_tensor((self.mma_rows, BT))).shape
        r2s_atom = sm100_utils.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, Float32, self.acc_type, tc_s
        )
        tc_r2s = cute.make_tiled_copy_D(r2s_atom, tc_s)
        thr_r2s = tc_r2s.get_slice(tmem_row % 128)

        ep_thr = cute.make_ordered_layout((8, 32), order=(1, 0))
        ep_val = cute.make_layout((1, 4))
        cp_f32 = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=64),
            ep_thr,
            ep_val,
        )
        cp_bf16 = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_type, num_bits_per_copy=64),
            ep_thr,
            ep_val,
        )
        ep_g = cp_f32.get_slice(local_tid)
        ep_b = cp_bf16.get_slice(local_tid)
        ep_row = local_tid // 32
        ep_col0 = (local_tid % 32) * 4
        # Store-pass copies over the (BT, BT) outputs: 16 rows x 16
        # col-groups x 4 elems = 256 threads.
        st_thr = cute.make_ordered_layout((16, 16), order=(1, 0))
        st_val = cute.make_layout((1, 4))
        cp_st_b = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.io_type, num_bits_per_copy=64),
            st_thr,
            st_val,
        )
        st_b = cp_st_b.get_slice(local_tid)
        st_row_in = local_tid // 16
        st_col0 = (local_tid % 16) * 4

        it = Int32(0)
        has_work = it < n_iters + 1
        while has_work:
            # ---- PRODUCE unit `it` (overlaps MMA/epilogue of `it - 1`) ----
            if it < n_iters:
                chunk_idx, _ = self.unit(params, it)
                _token_start, valid = self._decode_chunk(params, chunk_idx)
                raw_h = raw.wait()
                g_h = rawg.wait()
                sG = g_h.view("G")
                tsg_p = ep_g.partition_S(sG)
                tsk_p = ep_b.partition_S(raw_h.view("K"))
                tsq_p = ep_b.partition_S(raw_h.view("Q"))
                for sc in cutlass.range_constexpr(self.subchunks):
                    op_h = op.acquire()
                    self._produce_strip(
                        op_h,
                        sG,
                        tsg_p,
                        tsk_p,
                        tsq_p,
                        ep_b,
                        cp_f32,
                        cp_bf16,
                        ep_row,
                        ep_col0,
                        valid,
                        sc,
                    )
                    op_h.commit()
                raw_h.release()
                g_h.release()

            # ---- EPILOGUE unit `it - 1` ----
            if it > 0:
                chunk_idx, h_idx = self.unit(params, it - 1)
                token_start, valid = self._decode_chunk(params, chunk_idx)
                token_offset = self.upcast(token_start)
                head_offset = self.upcast(h_idx)
                if local_tid < BT:
                    b_val = Float32(0.0)
                    if local_tid < valid:
                        b_val = g_beta[(token_offset + local_tid, (head_offset, 0))]
                    sBeta[local_tid] = b_val

                acc_h = acc.wait()
                s_reg = cute.make_rmem_tensor(sreg_shape, self.acc_type)
                if tmem_row < 128:
                    cute.copy(tc_s, p_t_aqk[(None, None, None, acc_h.index)], s_reg)
                    cute.copy(tc_r2s, tc_r2s.retile(s_reg), thr_r2s.partition_D(sAqkSt))
                    cute.copy(tc_s, p_t_akk[(None, None, None, acc_h.index)], s_reg)
                    cute.copy(tc_r2s, tc_r2s.retile(s_reg), thr_r2s.partition_D(sAkkSt))
                acc_h.release()
                self.staged.sync(aligned=True)

                gstride_a = cute.assume(H * BT, divby=4)
                gaq = cute.make_tensor(
                    cute.make_ptr(
                        self.io_type,
                        (g_aqk.iterator + token_offset * H * BT + head_offset * BT).toint(),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    cute.make_layout((BT, BT), stride=(gstride_a, 1)),
                )
                taq_o = st_b.partition_D(gaq)
                for rb in cutlass.range_constexpr(cute.size(taq_o, mode=[1])):
                    row = rb * 16 + st_row_in
                    if row < valid:
                        r_a = cute.make_fragment_like(taq_o[(None, 0, None)], self.io_type)
                        for e in cutlass.range_constexpr(cute.size(r_a)):
                            col = st_col0 + e
                            aqk_v = Float32(0.0)
                            if col <= row and col < valid:
                                aqk_v = sAqkSt[(row, col)] * scale
                            r_a[e] = self.io_type(aqk_v)
                        cute.copy(cp_st_b, r_a, taq_o[(None, rb, None)])

                for rb in cutlass.range_constexpr(cute.size(taq_o, mode=[1])):
                    row = rb * 16 + st_row_in
                    beta_r = sBeta[row]
                    ri = row // 16
                    ci_blk = st_col0 // 16
                    if ci_blk < ri:
                        pair = ri * (ri - 1) // 2 + ci_blk
                        od_row = self.upcast(chunk_idx) * self.offdiag_blocks + pair
                        r_od = cute.make_rmem_tensor((4,), Float32)
                        for e in cutlass.range_constexpr(4):
                            r_od[e] = sAkkSt[(row, st_col0 + e)] * beta_r
                        od_raw = (
                            g_akkod.iterator
                            + od_row * (H * 256)
                            + head_offset * 256
                            + (row % 16) * 16
                            + (st_col0 % 16)
                        )
                        od_t = cute.make_tensor(
                            cute.make_ptr(
                                Float32,
                                od_raw.toint(),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout((4,), stride=(1,)),
                        )
                        cute.autovec_copy(r_od, od_t)
                dib = local_tid // 64
                dit = local_tid % 64
                dr = dit % 16
                dc0 = (dit // 16) * 4
                arow = dib * 16 + dr
                if arow < valid:
                    for e in cutlass.range_constexpr(4):
                        dc = dc0 + e
                        a_v = Float32(0.0)
                        if dc < dr:
                            a_v = sAkkSt[(arow, dib * 16 + dc)] * sBeta[arow]
                        g_akkd[(token_offset + arow, dc, (head_offset, 0))] = a_v
                self.staged.sync(aligned=True)

            it = it + 1
            has_work = it < n_iters + 1
        return raw, rawg, op, acc

    @role(warps=8)
    @cute.jit
    def load(self, params: IntraFwdParams, *, raw: Producer, rawg: Producer):
        """Stage raw Q, K, and gate tiles with TMA."""
        BT, BK = self.BT, self.BK
        atom_q, atom_k, atom_g = params.atom_q, params.atom_k, params.atom_g
        cpasync.prefetch_descriptor(atom_q)
        cpasync.prefetch_descriptor(atom_k)
        cpasync.prefetch_descriptor(atom_g)
        it = Int32(0)
        has_work = it < params.num_units
        while has_work:
            chunk_idx, h_idx = self.unit(params, it)
            token_start, _valid = self._decode_chunk(params, chunk_idx)
            token_offset = self.upcast(token_start)
            head_offset = self.upcast(h_idx)
            sSK, gSK = self._part_epi(
                atom_k,
                cute.domain_offset((token_offset, 0, (0, 0)), params.desc_k)[
                    None, None, (head_offset, 0)
                ],
                (BT, BK),
                params.sK,
            )
            sSG, gSG = self._part_epi(
                atom_g,
                cute.domain_offset((token_offset, 0, (0, 0)), params.desc_g)[
                    None, None, (head_offset, 0)
                ],
                (BT, BK),
                params.sG,
            )
            sSQ, gSQ = self._part_epi(
                atom_q,
                cute.domain_offset((token_offset, 0, (0, 0)), params.desc_q)[
                    None, None, (head_offset, 0)
                ],
                (BT, BK),
                params.sQ,
            )
            raw = raw.views(Q=StageView(sSQ, axis=1), K=StageView(sSK, axis=1))
            rawg = rawg.views(G=StageView(sSG, axis=1))
            with raw.produce() as rh:
                cute.copy(
                    atom=atom_q, src=gSQ[(None, 0, 0)], dst=rh.view("Q"), tma_bar_ptr=rh.barrier
                )
                cute.copy(
                    atom=atom_k, src=gSK[(None, 0, 0)], dst=rh.view("K"), tma_bar_ptr=rh.barrier
                )
            with rawg.produce() as gh:
                cute.copy(
                    atom=atom_g, src=gSG[(None, 0, 0)], dst=gh.view("G"), tma_bar_ptr=gh.barrier
                )
            it = it + 1
            has_work = it < params.num_units
        return raw, rawg

    @role(warps=9)
    @cute.jit
    def mma(self, params: IntraFwdParams, *, op: Consumer, acc: Producer, tmem: Tmem):
        """Consume gated operands and produce Aqk/Akk accumulators."""
        mma16 = params.mma16
        t_qg_a = mma16.make_fragment_A(params.sQg)
        op = op.views(
            Qg=StageView(t_qg_a, axis=3),
            Kgq=StageView(mma16.make_fragment_A(params.sKgq), axis=3),
            Kgk=StageView(mma16.make_fragment_B(params.sKgk), axis=3),
        )
        # TMEM addresses pack (lane << 16 | col): an accumulator stage is a lane offset, taken
        # from the fragment layout, and a strip a column offset.
        stage_stride = params.acc_layout.stride[3]
        aqk, akk = tmem.ptr("aqk"), tmem.ptr("akk")
        it = Int32(0)
        has_work = it < params.num_units
        while has_work:
            acc_h = acc.acquire()
            for sc in cutlass.range_constexpr(self.subchunks):
                op_h = op.wait()
                col = acc_h.index * stage_stride + sc * 16
                aqk_c = cute.make_tensor(aqk + col, params.strip_layout)
                akk_c = cute.make_tensor(akk + col, params.strip_layout)
                for kp in cutlass.range(cute.size(t_qg_a, mode=[2]), unroll_full=True):
                    mma16.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma16,
                        aqk_c,
                        op_h.view("Qg", None, None, kp, read=True, by="umma"),
                        op_h.view("Kgk", None, None, kp, read=True, by="umma"),
                        aqk_c,
                    )
                for kp in cutlass.range(cute.size(t_qg_a, mode=[2]), unroll_full=True):
                    mma16.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma16,
                        akk_c,
                        op_h.view("Kgq", None, None, kp, read=True, by="umma"),
                        op_h.view("Kgk", None, None, kp, read=True, by="umma"),
                        akk_c,
                    )
                op_h.release()
            acc_h.commit()
            it = it + 1
            has_work = it < params.num_units
        return op, acc

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        mma_s: cute.TiledMma,
        mma16: cute.TiledMma,
        atom_q: cute.CopyAtom,
        desc_q: cute.Tensor,
        atom_k: cute.CopyAtom,
        desc_k: cute.Tensor,
        atom_g: cute.CopyAtom,
        desc_g: cute.Tensor,
        s_kraw: cute.ComposedLayout,
        s_graw: cute.ComposedLayout,
        s_sa_op: cute.ComposedLayout,
        s_sb_op: cute.ComposedLayout,
        s_op_store: cute.ComposedLayout,
        s_opb_store: cute.ComposedLayout,
        g_beta: cute.Tensor,
        g_aqk: cute.Tensor,
        g_akkod: cute.Tensor,
        g_akkd: cute.Tensor,
        scale: Float32,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        T: Int32,
        H: Int32,
    ):
        """Bind the buffers, decode this CTA's share of the units, then run the roles."""
        BT = self.BT
        smem = self.begin_roles(
            layouts={
                "Q": s_kraw,
                "K": s_kraw,
                "G": s_graw,
                "Qg": s_sa_op,
                "Kgq": s_sa_op,
                "Kgk": s_sb_op,
                "QgSt": s_op_store,
                "KgqSt": s_op_store,
                "KgkSt": s_opb_store,
            }
        )
        st_lay = cute.make_layout((self.mma_rows, BT), stride=(BT, 1))
        acc = mma_s.make_fragment_C(
            cute.append(mma_s.partition_shape_C(self.s_tile[:2]), self.acc_depth)
        )
        # Both stages must fit the plan's BT columns per accumulator.
        assert tcgen05.find_tmem_tensor_col_offset(acc) == BT

        bx = cute.arch.block_idx()[0]
        gdx = cute.arch.grid_dim()[0]
        if cutlass.const_expr(self.varlen):
            num_sequences = Int32(cute.size(chunk_offsets)) - 1
            active_chunks = Int32(chunk_offsets[num_sequences])
        else:
            active_chunks = T // Int32(BT)
        total_work = active_chunks * H
        params = IntraFwdParams(
            mma16=mma16,
            atom_q=atom_q,
            desc_q=desc_q,
            atom_k=atom_k,
            desc_k=desc_k,
            atom_g=atom_g,
            desc_g=desc_g,
            sQ=self.buffer("Q"),
            sK=self.buffer("K"),
            sG=self.buffer("G"),
            sQg=self.buffer("Qg"),
            sKgq=self.buffer("Kgq"),
            sKgk=self.buffer("Kgk"),
            sBeta=smem.allocate_tensor(Float32, cute.make_layout((BT,)), 128),
            sAqkSt=smem.allocate_tensor(Float32, st_lay, self.align),
            sAkkSt=smem.allocate_tensor(Float32, st_lay, self.align),
            acc_layout=acc.layout,
            strip_layout=mma16.make_fragment_C(mma16.partition_shape_C((BT, 16))).layout,
            g_beta=g_beta,
            g_aqk=g_aqk,
            g_akkod=g_akkod,
            g_akkd=g_akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_offsets=chunk_offsets,
            H=H,
            first_unit=bx,
            unit_stride=gdx,
            num_units=(total_work - bx + gdx - 1) // gdx,
        )
        self.run_roles(params)

    @cute.jit
    def _part_epi(self, atom, g_mnl, tile, s_buf):
        g_div = cute.flat_divide(g_mnl, tile)
        sg = cute.group_modes(s_buf, 0, 2)
        gg = cute.group_modes(g_div, 0, 2)
        return cpasync.tma_partition(atom, 0, cute.make_layout(1), sg, gg)

    def _launch_grid(self):
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA compilation requires a CUDA target with an SM count")
        return (sm_count, 1, 1)

    # ------------------------------------------------------------------- host

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,  # [T, H, K] bf16
        k_in: cute.Tensor,  # [T, H, K] bf16
        g_in: cute.Tensor,  # [T, H, K] fp32
        beta_in: cute.Tensor,  # [T, H] fp32
        aqk_in: cute.Tensor,  # [T, H, BT] bf16 (out)
        akkod_in: cute.Tensor,  # [caps*6, H*256] fp32 (out, K3b blocked layout)
        akkd_in: cute.Tensor,  # [T, H, 16] fp32 (out, diag-block inverses)
        scale: Float32,
        cu_seqlens_in: cute.Tensor | None,
        chunk_offsets_in: cute.Tensor | None,
        T: Int32,
        H: Int32,
        stream,
    ):
        BT, BK = self.BT, self.BK

        def tok_view(t, dim):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout(
                    (T, dim, (H, 1)),
                    stride=(
                        t.layout.stride[0],
                        t.layout.stride[2],
                        (t.layout.stride[1], 0),
                    ),
                ),
            )

        g_q = tok_view(q_in, BK)
        g_k = tok_view(k_in, BK)
        g_g = tok_view(g_in, BK)
        g_aqk = tok_view(aqk_in, BT)
        g_akkod = akkod_in  # already flat [caps*6, H*256]
        g_akkd = tok_view(akkd_in, 16)
        g_beta = cute.make_tensor(
            beta_in.iterator, cute.make_layout((T, (H, 1)), stride=(H, (1, 0)))
        )

        mma_s = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_type,
            self.cta_group,
            self.s_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # Per-16-column strip MMA: each strip's operands are rebased at that strip's own
        # reference (see the strip-rebase paragraph in the module docstring).
        strip_tile = (self.mma_rows, 16, BK)
        mma16 = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            self.acc_type,
            self.cta_group,
            strip_tile[:2],
            tcgen05.OperandSource.SMEM,
        )

        tma_ld = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        s_kraw = sm100_utils.make_smem_layout_epi(
            self.io_type, LayoutEnum.ROW_MAJOR, (BT, BK), self.raw_depth
        )
        s_graw = sm100_utils.make_smem_layout_epi(Float32, LayoutEnum.ROW_MAJOR, (BT, BK), 1)
        atom_q, desc_q = cpasync.make_tiled_tma_atom(
            tma_ld, g_q, cute.select(s_kraw, mode=[0, 1]), (BT, BK)
        )
        atom_k, desc_k = cpasync.make_tiled_tma_atom(
            tma_ld, g_k, cute.select(s_kraw, mode=[0, 1]), (BT, BK)
        )
        atom_g, desc_g = cpasync.make_tiled_tma_atom(
            tma_ld, g_g, cute.select(s_graw, mode=[0, 1]), (BT, BK)
        )
        s_sa_op = sm100_utils.make_smem_layout_a(mma16, strip_tile, self.io_type, self.op_depth)
        s_sb_op = sm100_utils.make_smem_layout_b(mma16, strip_tile, self.io_type, self.op_depth)
        s_op_store = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.ROW_MAJOR,
            (self.mma_rows, BK),
            self.op_depth,
        )
        s_opb_store = sm100_utils.make_smem_layout_epi(
            self.io_type, LayoutEnum.ROW_MAJOR, (16, BK), self.op_depth
        )

        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            mma_s,
            mma16,
            atom_q,
            desc_q,
            atom_k,
            desc_k,
            atom_g,
            desc_g,
            s_kraw,
            s_graw,
            s_sa_op,
            s_sb_op,
            s_op_store,
            s_opb_store,
            g_beta,
            g_aqk,
            g_akkod,
            g_akkd,
            scale,
            cu_seqlens_in,
            chunk_offsets_in,
            T,
            H,
        ).launch(**self.launch_config(grid=self._launch_grid()), stream=stream)


@jit_cache
def _compile_intra_engine_fwd(
    H: int,
    head_dim: int,
    varlen: bool,
    use_int64_offsets: bool,
):
    target = get_compile_target()
    if target.device_type != "cuda" or not runs_sm100_kernels(target.effective_capability):
        raise ValueError(f"KDA intra engine requires an SM100-family target; got {target}")
    op = KdaIntraFwdEngine(
        head_dim=head_dim,
        num_heads=H,
        varlen=varlen,
        use_int64_offsets=use_int64_offsets,
    )
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    st, sn = sym_int(), cute.sym_int()

    def token_tensor(dtype, dim):
        return make_fake_compact_tensor(
            dtype, (st, H, dim), stride_order=(2, 1, 0), assumed_align=128
        )

    def qk_tensor():
        return make_fake_strided_tensor(
            cutlass.BFloat16,
            (st, H, head_dim),
            contiguous_dim=2,
            stride_divisibility=TMA_ALIGNMENT_BYTES // 2,
            assumed_align=TMA_ALIGNMENT_BYTES,
            use_int64_strides=use_int64_offsets,
        )

    beta = make_fake_compact_tensor(cutlass.Float32, (st, H), stride_order=(1, 0), assumed_align=8)
    cu = (
        make_fake_compact_tensor(cutlass.Int32, (sn,), stride_order=(0,), assumed_align=4)
        if varlen
        else None
    )
    offs = (
        make_fake_compact_tensor(cutlass.Int32, (sn,), stride_order=(0,), assumed_align=4)
        if varlen
        else None
    )
    return compile_tvm_ffi(
        op,
        qk_tensor(),
        qk_tensor(),
        token_tensor(cutlass.Float32, head_dim),
        beta,
        token_tensor(cutlass.BFloat16, 64),
        make_fake_compact_tensor(
            cutlass.Float32, (sym_int(), H * 256), stride_order=(1, 0), assumed_align=16
        ),
        token_tensor(cutlass.Float32, 16),
        cutlass.Float32(1.0),
        cu,
        offs,
        1,
        H,
        name=op.get_name(),
    )


def kda_intra_engine_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce the BT64 Aqk and inverse factors.

    There is no ``fastmath`` switch: the shifted strip rebase keeps every factor that matters
    normal, so the flush-to-zero ``ex2`` both modes use loses nothing visible (module docstring).
    """
    batch, tokens, heads, head_dim = q.shape
    chunk_size = 64
    assert batch == 1 and head_dim == 128
    if metadata is None:
        assert tokens % chunk_size == 0
        capacity = tokens // chunk_size
        cu_seqlens = chunk_offsets = None
    else:
        metadata.validate_chunk_size(chunk_size)
        capacity = metadata.capacity
        cu_seqlens = metadata.cu_seqlens
        chunk_offsets = metadata.chunk_offsets
    aqk = torch.empty((1, tokens, heads, chunk_size), device=q.device, dtype=torch.bfloat16)
    offdiag_blocks = 6
    # Dense execution writes every row. Ragged capacity slack stays zero for downstream consumers.
    akkod_factory = torch.empty if metadata is None else torch.zeros
    akkod = akkod_factory(
        (capacity * offdiag_blocks, heads * 256), device=q.device, dtype=torch.float32
    )
    akkd = torch.empty((1, tokens, heads, 16), device=q.device, dtype=torch.float32)
    if capacity == 0:
        return aqk, akkod, akkd

    q_flat = q[0]
    k_flat = k[0]
    g_flat = normalize_compact_tensor(g[0])
    beta_flat = normalize_compact_tensor(beta[0], alignment_bytes=8)
    aqk_flat = aqk[0]
    akkd_flat = akkd[0]
    use_int64_offsets = requires_int64_abi(
        q_flat,
        k_flat,
        g_flat,
        beta_flat,
        aqk_flat,
        akkod,
        akkd_flat,
    )
    compiled = _compile_intra_engine_fwd(
        heads,
        head_dim,
        metadata is not None,
        use_int64_offsets,
    )
    compiled(
        q_flat,
        k_flat,
        g_flat,
        beta_flat,
        aqk_flat,
        akkod,
        akkd_flat,
        float(scale),
        cu_seqlens,
        chunk_offsets,
        tokens,
        heads,
    )
    return aqk, akkod, _kda_diag_neumann_inverse(akkd, metadata, chunk_size)
