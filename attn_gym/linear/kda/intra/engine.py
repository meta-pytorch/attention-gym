# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SM100 (tcgen05) intra-chunk engine for KDA.

Warp-specialized persistent kernel over (chunk, head) work units, following the
``BlackwellDeltaHBwdV1`` architecture (high-level tcgen05: trivial tiled MMA,
TmemAllocator, cute.gemm) with wy_dqkg's gated-operand idiom: CUDA warps produce
gated bf16 MMA operands into SMEM via R2S alias views, handed to the MMA warp
through PipelineAsyncUmma.

Backward mode (``mode="bwd"``) computes the full intra-chunk backward, with
exp2 factored as gq[i,d] = 2^(g-gref), gk[j,d] = 2^(gref-g), gref = g[row 0]:

    phase A gemms:  dq  = aq @ kgk^T          dkb = ak @ kgk^T
    phase B gemms:  S   = kgq @ kgk^T(d)      dka = aq^T @ qg^T
                    dkc = (ak*beta)^T @ kgq^T
    epilogue:       dq2 = dq_in + dq*gq
                    dk2 = dk_in + dka*gk + dkb*gq*beta + dkc*gk
                    dg2 = (dg_in + q*dq_s - k*dka_s + k*dkb_s - k*dkc_s) * ln2
                    db2 = db_in + rowsum(ak * S)

Safety status per mode: the forward class (KdaIntraFwdEngine) uses the
per-16-column-strip rebase and is bounded at the training gate range; the
backward class below still uses the single chunk-row-0 reference and OVERFLOWS
at init-realistic gates (see the note below) -- do not wire it into training
until it receives the same strip treatment.

Notes: causality and gk scale
-----------------------------
The per-channel decay 2^(g[i,d]-g[j,d]) sits inside the reduction over d, so it
can only enter the MMA as gated operands split around a reference:
gq = 2^(g-gref) on rows and gk = 2^(gref-g) on columns. gref cancels exactly in
real arithmetic; in floating point the two exp2 roundings are independent, so
outputs depend on gref at the ulp level. gref must therefore be causal (at or
before every query row it serves) and independent of sequence length, or future
gates perturb earlier outputs (~1.5e-5 bf16 measured for the midpoint variant)
and prefix invariance breaks. This engine uses gref = g[chunk row 0]: causal
and length-stable.

The cost is one-sided intermediate scale: gk grows as 2^(drop since row 0), up
to 2^R64 for the full-chunk drop R64. With the training gate bound of -5
nats/token (7.21 log2 units), R64 reaches ~461 while the fp32/bf16 exponent
budget is ~126: a channel sustaining more than ~2 log2 units/token of decay
across a chunk overflows gk to inf (and flushes gq to 0), producing NaNs.
Measured: finite at 1.0 log2/tok, NaN at >=3.6. The shipped kernels are safe by
construction (16-token windows: midpoint <= 2^58, row-0 <= 2^115; K3b's
between-blocks reference keeps both factors <= 1). Clamping cannot fix this
(late-pair true values are O(1) but would compute as 0*inf), and dispatch on
the per-chunk drop is not viable either: at initialization the training
example's own gate parameterization (A_log = dt_bias = 0 -> gate ~
-5*sigmoid(z) ~ -2.5 nats/token) puts the MEDIAN per-chunk per-channel drop at
231 log2 units, with 100% of (chunk, head, channel) triples above the 126-unit
cliff. The engine would NaN on the first training step. The only real fix is
per-row-block operand rebasing (sandwiched block gemms), which must land
before this engine can be wired into training.

Design + tradeoffs: ~/agent_notes/plans/attention_gym_intra_engine_tcgen05.md.
"""

import math

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Float32, Int32, Int64

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import compile_tvm_ffi
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import load_ragged_chunk_work

LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# Tiny standalone Triton kernel: 16x16 unit-lower Neumann inverses of the
# engine's raw diag blocks (same log-depth factorization as the fwd forloop).
# ---------------------------------------------------------------------------
import triton
import triton.language as tl

from attn_gym.linear.kda.chunk_scheduler import (
    load_ragged_chunk_work as _tl_load_ragged_chunk_work,
)


@triton.jit
def _diag_neumann_inverse_kernel(
    akkd,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    H: tl.constexpr,
    BC: tl.constexpr,
):
    pid, i_h = tl.program_id(0), tl.program_id(1).to(tl.int64)
    if pid // 4 >= tl.load(chunk_offsets + num_sequences):
        return
    _, _, token_start, valid = _tl_load_ragged_chunk_work(
        cu_seqlens, chunk_offsets, pid // 4, num_sequences, 4 * BC
    )
    sub = pid % 4
    row0 = token_start + sub * BC
    vsub = tl.minimum(tl.maximum(valid - sub * BC, 0), BC)
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


def _kda_diag_neumann_inverse(akkd: torch.Tensor, metadata: RaggedChunkMetadata) -> torch.Tensor:
    """In-place inv(I + strict(block)) over the engine's raw 16x16 diag blocks.

    The 3-squaring doubling covers exactly degree 15, so BC must be 16.
    """
    _, _, heads, BC = akkd.shape
    assert BC == 16, f"diag Neumann inverse requires 16x16 blocks, got BC={BC}"
    _diag_neumann_inverse_kernel[(metadata.capacity * 4, heads)](
        akkd,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        metadata.cu_seqlens.shape[0] - 1,
        H=heads,
        BC=BC,
        num_warps=1,
    )
    return akkd


class KdaIntraEngine:
    """Warp-specialized SM100 intra-chunk engine (bwd mode)."""

    CUDA_WARP_IDS = (0, 1, 2, 3, 4, 5, 6, 7)
    LOAD_WARP_ID = 8
    MMA_WARP_ID = 9
    IDLE_WARP_IDS = (10, 11)  # pad warp-group 2 so setmaxregister stays aligned
    WARP_SZ = 32
    N_WARPS = 12
    CTA_THREADS = N_WARPS * WARP_SZ  # 384

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim: int = 128,
        num_heads: int | None = None,
        mode: str = "bwd",
        io_type=cutlass.BFloat16,
        acc_type=cutlass.Float32,
        kq_depth: int = 1,
        araw_depth: int = 1,
        grid_waves: int = 1,
    ):
        assert chunk_size == 64 and head_dim == 128
        assert mode == "bwd", "fwd mode not wired yet"
        self.mode = mode
        self.BT = chunk_size
        self.BK = head_dim
        self.io_type = io_type
        self.acc_type = acc_type
        self.num_heads = num_heads

        # Token-contraction gemms (dq/dkb/dka/dkc): (M, N, K) = (BT, BK, BT).
        self.tok_tile = (self.BT, self.BK, self.BT)
        # Channel-contraction gemm (S): (M, N, K) = (BT, BT, BK).
        self.s_tile = (self.BT, self.BT, self.BK)

        self.raw_depth = 1  # g (held through the epilogue)
        self.kq_depth = kq_depth
        self.araw_depth = araw_depth
        self.grid_waves = grid_waves
        self.op_depth = 1
        self.acc_depth = 1

        self.cuda_regs = 224
        self.aux_regs = 48
        self.cluster = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tmem_free_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=self.CTA_THREADS)
        self.cuda_bar = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.WARP_SZ * len(self.CUDA_WARP_IDS)
        )
        self.align = 1024

    def get_name(self) -> str:
        head_tag = f"_h{self.num_heads}" if self.num_heads is not None else ""
        return (
            f"kda_intra_engine_{self.mode}{head_tag}_k{self.BK}_bt{self.BT}"
            f"_kq{self.kq_depth}_ar{self.araw_depth}_w{self.grid_waves}"
        )

    # ------------------------------------------------------------------
    # Host-side setup
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,  # [T, H, K] bf16
        k_in: cute.Tensor,  # [T, H, K] bf16
        g_in: cute.Tensor,  # [T, H, K] fp32
        beta_in: cute.Tensor,  # [T, H] fp32
        daqk_in: cute.Tensor,  # [T, H, BT] fp32
        dakk_in: cute.Tensor,  # [T, H, BT] fp32
        dq_in: cute.Tensor,  # [T, H, K] fp32
        dk_in: cute.Tensor,  # [T, H, K] fp32
        db_in: cute.Tensor,  # [T, H] fp32
        dg_in: cute.Tensor,  # [T, H, K] fp32
        dq2_in: cute.Tensor,  # [T, H, K] bf16 (out)
        dk2_in: cute.Tensor,  # [T, H, K] bf16 (out)
        dg2_in: cute.Tensor,  # [T, H, K] fp32 (out)
        db2_in: cute.Tensor,  # [T, H] fp32 (out)
        cu_seqlens_in: cute.Tensor,
        chunk_offsets_in: cute.Tensor,
        T: Int32,
        H: Int32,
        capacity: Int32,
        stream,
    ):
        BT, BK = self.BT, self.BK

        def tok_view(t, dim):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout((T, dim, (H, 1)), stride=(H * dim, 1, (dim, 0))),
            )

        def vec_view(t):
            return cute.make_tensor(t.iterator, cute.make_layout((T, (H, 1)), stride=(H, (1, 0))))

        g_q = tok_view(q_in, BK)
        g_k = tok_view(k_in, BK)
        g_g = tok_view(g_in, BK)
        g_aq = tok_view(daqk_in, BT)
        g_ak = tok_view(dakk_in, BT)
        g_dqin = tok_view(dq_in, BK)
        g_dkin = tok_view(dk_in, BK)
        g_dgin = tok_view(dg_in, BK)
        g_dq2 = tok_view(dq2_in, BK)
        g_dk2 = tok_view(dk2_in, BK)
        g_dg2 = tok_view(dg2_in, BK)
        g_beta = vec_view(beta_in)
        g_dbin = vec_view(db_in)
        g_db2 = vec_view(db2_in)

        # --- MMAs ---
        # Token-contraction: A (BT, BT) K-major, B (BK, BT) MN-major.
        mma_tok = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            self.acc_type,
            self.cta_group,
            self.tok_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # Transposed token-contraction (dka/dkc): A (BT, BT) MN-major reads the
        # aq/akb buffers as (j, i) without a second copy; B unchanged MN-major.
        mma_tok2 = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.MN,
            tcgen05.OperandMajorMode.MN,
            self.acc_type,
            self.cta_group,
            self.tok_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # Channel-contraction (S): A (BT, BK) K-major, B (BT, BK) K-major.
        mma_s = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_type,
            self.cta_group,
            self.s_tile[:2],
            tcgen05.OperandSource.SMEM,
        )

        tok_c = mma_tok.partition_shape_C(self.tok_tile[:2])
        n_tok = tcgen05.find_tmem_tensor_col_offset(
            mma_tok.make_fragment_C(cute.append(tok_c, self.acc_depth))
        )
        s_c = mma_s.partition_shape_C(self.s_tile[:2])
        s_fk_probe = mma_s.make_fragment_C(cute.append(s_c, self.acc_depth))
        n_s = tcgen05.find_tmem_tensor_col_offset(s_fk_probe)
        # dkc accumulates into dka's columns (identical gk epilogue scale), so
        # all four regions coexist: no tmem reuse, no drain-order guard needed.
        self.tm_dq = 0
        self.tm_dkb = n_tok
        self.tm_dka = 2 * n_tok
        self.tm_s = 3 * n_tok
        raw_tot = 3 * n_tok + n_s
        tot = 1
        for _ in cutlass.range_constexpr(10):
            if cutlass.const_expr(tot < raw_tot):
                tot *= 2
        self.tm_tot = tot

        # --- SMEM layouts ---
        tma_ld = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)

        s_a_op = sm100_utils.make_smem_layout_a(mma_tok, self.tok_tile, self.io_type, 1)
        s_b_op = sm100_utils.make_smem_layout_b(mma_tok, self.tok_tile, self.io_type, 1)
        s_a2_op = sm100_utils.make_smem_layout_a(mma_tok2, self.tok_tile, self.io_type, 1)
        s_b2_op = sm100_utils.make_smem_layout_b(mma_tok2, self.tok_tile, self.io_type, 1)
        s_sa_op = sm100_utils.make_smem_layout_a(mma_s, self.s_tile, self.io_type, 1)
        s_sb_op = sm100_utils.make_smem_layout_b(mma_s, self.s_tile, self.io_type, 1)
        s_a_store = sm100_utils.make_smem_layout_epi(
            self.io_type, utils.LayoutEnum.ROW_MAJOR, (BT, BT), 1
        )
        s_b_store = sm100_utils.make_smem_layout_epi(
            self.io_type, utils.LayoutEnum.ROW_MAJOR, (BT, BK), 1
        )
        s_sab_store = sm100_utils.make_smem_layout_epi(
            self.io_type, utils.LayoutEnum.ROW_MAJOR, (BT, BK), 1
        )

        s_kraw = sm100_utils.make_smem_layout_epi(
            self.io_type, utils.LayoutEnum.ROW_MAJOR, (BT, BK), self.raw_depth
        )
        s_graw = sm100_utils.make_smem_layout_epi(
            Float32, utils.LayoutEnum.ROW_MAJOR, (BT, BK), self.raw_depth
        )
        s_araw = sm100_utils.make_smem_layout_epi(
            Float32, utils.LayoutEnum.ROW_MAJOR, (BT, BT), self.raw_depth
        )

        atom_q, desc_q = cpasync.make_tiled_tma_atom(
            tma_ld, g_q, cute.select(s_kraw, mode=[0, 1]), (BT, BK)
        )
        atom_k, desc_k = cpasync.make_tiled_tma_atom(
            tma_ld, g_k, cute.select(s_kraw, mode=[0, 1]), (BT, BK)
        )
        atom_g, desc_g = cpasync.make_tiled_tma_atom(
            tma_ld, g_g, cute.select(s_graw, mode=[0, 1]), (BT, BK)
        )
        atom_aq, desc_aq = cpasync.make_tiled_tma_atom(
            tma_ld, g_aq, cute.select(s_araw, mode=[0, 1]), (BT, BT)
        )
        atom_ak, desc_ak = cpasync.make_tiled_tma_atom(
            tma_ld, g_ak, cute.select(s_araw, mode=[0, 1]), (BT, BT)
        )

        self.kq_bytes = cute.size_in_bytes(self.io_type, cute.select(s_kraw, mode=[0, 1]))
        self.g_bytes = cute.size_in_bytes(Float32, cute.select(s_graw, mode=[0, 1]))
        self.a_bytes = cute.size_in_bytes(Float32, cute.select(s_araw, mode=[0, 1]))

        @cute.struct
        class Shared:
            bar_kq: cute.struct.MemRange[Int64, self.raw_depth * 2]
            bar_raw: cute.struct.MemRange[Int64, self.raw_depth * 2]
            bar_opA: cute.struct.MemRange[Int64, self.op_depth * 2]
            bar_opB: cute.struct.MemRange[Int64, self.op_depth * 2]
            bar_accA: cute.struct.MemRange[Int64, self.acc_depth * 2]
            bar_accB: cute.struct.MemRange[Int64, self.acc_depth * 2]
            tmem_buf: Int32
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_kraw)], self.align
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_kraw)], self.align
            ]
            sG: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(s_graw)], self.align]
            sAqRaw: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(s_araw)], self.align
            ]
            sAkRaw: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(s_araw)], self.align
            ]
            sBeta: cute.struct.Align[cute.struct.MemRange[Float32, self.BT], 128]
            sBetaRed: cute.struct.Align[cute.struct.MemRange[Float32, 2 * self.BT], 128]
            sAq: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_a_op)], self.align
            ]
            sAk: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_a_op)], self.align
            ]
            sKgk: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_b_op)], self.align
            ]
            sQg: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_b_op)], self.align
            ]
            sKgqB: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_b_op)], self.align
            ]
            sStg1: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_b_store)], self.align
            ]
            sStg2: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_b_store)], self.align
            ]
            sStg3: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_b_store)], self.align
            ]

        self.shared_type = Shared

        self.kernel(
            mma_tok,
            mma_tok2,
            mma_s,
            atom_q,
            desc_q,
            atom_k,
            desc_k,
            atom_g,
            desc_g,
            atom_aq,
            desc_aq,
            atom_ak,
            desc_ak,
            s_kraw,
            s_graw,
            s_araw,
            s_a_op,
            s_b_op,
            s_a2_op,
            s_b2_op,
            s_sa_op,
            s_sb_op,
            s_a_store,
            s_b_store,
            s_sab_store,
            g_q,
            g_k,
            g_beta,
            g_dqin,
            g_dkin,
            g_dgin,
            g_dbin,
            g_dq2,
            g_dk2,
            g_dg2,
            g_db2,
            cu_seqlens_in,
            chunk_offsets_in,
            H,
            capacity,
        ).launch(
            grid=self._launch_grid(),
            block=[self.CTA_THREADS, 1, 1],
            cluster=self.cluster,
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mma_tok: cute.TiledMma,
        mma_tok2: cute.TiledMma,
        mma_s: cute.TiledMma,
        atom_q: cute.CopyAtom,
        desc_q: cute.Tensor,
        atom_k: cute.CopyAtom,
        desc_k: cute.Tensor,
        atom_g: cute.CopyAtom,
        desc_g: cute.Tensor,
        atom_aq: cute.CopyAtom,
        desc_aq: cute.Tensor,
        atom_ak: cute.CopyAtom,
        desc_ak: cute.Tensor,
        s_kraw: cute.ComposedLayout,
        s_graw: cute.ComposedLayout,
        s_araw: cute.ComposedLayout,
        s_a_op: cute.ComposedLayout,
        s_b_op: cute.ComposedLayout,
        s_a2_op: cute.ComposedLayout,
        s_b2_op: cute.ComposedLayout,
        s_sa_op: cute.ComposedLayout,
        s_sb_op: cute.ComposedLayout,
        s_a_store: cute.ComposedLayout,
        s_b_store: cute.ComposedLayout,
        s_sab_store: cute.ComposedLayout,
        g_q: cute.Tensor,
        g_k: cute.Tensor,
        g_beta: cute.Tensor,
        g_dqin: cute.Tensor,
        g_dkin: cute.Tensor,
        g_dgin: cute.Tensor,
        g_dbin: cute.Tensor,
        g_dq2: cute.Tensor,
        g_dk2: cute.Tensor,
        g_dg2: cute.Tensor,
        g_db2: cute.Tensor,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        H: Int32,
        capacity: Int32,
    ):
        BT, BK = self.BT, self.BK
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()

        if warp_id == self.LOAD_WARP_ID:
            cpasync.prefetch_descriptor(atom_q)
            cpasync.prefetch_descriptor(atom_k)
            cpasync.prefetch_descriptor(atom_g)
            cpasync.prefetch_descriptor(atom_aq)
            cpasync.prefetch_descriptor(atom_ak)

        sa = utils.SmemAllocator()
        sm = sa.allocate(self.shared_type)

        n_cuda = self.WARP_SZ * len(self.CUDA_WARP_IDS)

        def tma_pipe(bar, nbytes, depth):
            return pipeline.PipelineTmaAsync.create(
                num_stages=depth,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, len(self.CUDA_WARP_IDS)
                ),
                tx_count=nbytes,
                barrier_storage=bar,
            ).make_participants()

        pkq_P, pkq_C = tma_pipe(sm.bar_kq.data_ptr(), 2 * self.kq_bytes, self.raw_depth)
        raw_bytes = self.g_bytes + 2 * self.a_bytes
        praw_P, praw_C = tma_pipe(sm.bar_raw.data_ptr(), raw_bytes, self.raw_depth)

        def op_pipe(bar):
            return pipeline.PipelineAsyncUmma.create(
                num_stages=self.op_depth,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, n_cuda),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                barrier_storage=bar,
            ).make_participants()

        popA_P, popA_C = op_pipe(sm.bar_opA.data_ptr())
        popB_P, popB_C = op_pipe(sm.bar_opB.data_ptr())

        def acc_pipe(bar):
            return pipeline.PipelineUmmaAsync.create(
                num_stages=self.acc_depth,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, n_cuda),
                barrier_storage=bar,
            ).make_participants()

        paccA_P, paccA_C = acc_pipe(sm.bar_accA.data_ptr())
        paccB_P, paccB_C = acc_pipe(sm.bar_accB.data_ptr())

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.CTA_THREADS)
        tmem = utils.TmemAllocator(
            sm.tmem_buf,
            barrier_for_retrieve=tmem_bar,
            allocator_warp_id=self.LOAD_WARP_ID,
        )
        tmem.allocate(self.tm_tot)
        tmem.wait_for_alloc()
        tp = tmem.retrieve_ptr(self.acc_type)

        # --- SMEM views ---
        sQ = sm.sQ.get_tensor(s_kraw.outer, swizzle=s_kraw.inner)
        sK = sm.sK.get_tensor(s_kraw.outer, swizzle=s_kraw.inner)
        sG = sm.sG.get_tensor(s_graw.outer, swizzle=s_graw.inner)
        sAqRaw = sm.sAqRaw.get_tensor(s_araw.outer, swizzle=s_araw.inner)
        sAkRaw = sm.sAkRaw.get_tensor(s_araw.outer, swizzle=s_araw.inner)
        sBetaRed = sm.sBetaRed.get_tensor(cute.make_layout((2 * BT,)))
        sAq = sm.sAq.get_tensor(s_a_op.outer, swizzle=s_a_op.inner)
        sAqSt = sm.sAq.get_tensor(s_a_store.outer, swizzle=s_a_store.inner)
        sAk = sm.sAk.get_tensor(s_a_op.outer, swizzle=s_a_op.inner)
        sAkSt = sm.sAk.get_tensor(s_a_store.outer, swizzle=s_a_store.inner)
        sAqT = sm.sAq.get_tensor(s_a2_op.outer, swizzle=s_a2_op.inner)
        sAkbT = sm.sAk.get_tensor(s_a2_op.outer, swizzle=s_a2_op.inner)
        sKgk = sm.sKgk.get_tensor(s_b_op.outer, swizzle=s_b_op.inner)
        sKgkSt = sm.sKgk.get_tensor(s_b_store.outer, swizzle=s_b_store.inner)
        sQgSt = sm.sQg.get_tensor(s_b_store.outer, swizzle=s_b_store.inner)
        sKgqBSt = sm.sKgqB.get_tensor(s_b_store.outer, swizzle=s_b_store.inner)
        sKgqA = sm.sKgqB.get_tensor(s_sa_op.outer, swizzle=s_sa_op.inner)
        sKgkS = sm.sKgk.get_tensor(s_sb_op.outer, swizzle=s_sb_op.inner)
        # db products (ak * S) stage through the freed dAqk raw tile.
        sDbSt = sm.sAqRaw.get_tensor(s_araw.outer, swizzle=s_araw.inner)

        # --- MMA fragments ---
        t_aq_a = mma_tok.make_fragment_A(sAq)
        t_ak_a = mma_tok.make_fragment_A(sAk)
        t_aqt_a = mma_tok2.make_fragment_A(sAqT)
        t_akbt_a = mma_tok2.make_fragment_A(sAkbT)
        t_kgk_b = mma_tok.make_fragment_B(sKgk)
        sQgT2 = sm.sQg.get_tensor(s_b2_op.outer, swizzle=s_b2_op.inner)
        sKgqB2 = sm.sKgqB.get_tensor(s_b2_op.outer, swizzle=s_b2_op.inner)
        sStg1 = sm.sStg1.get_tensor(s_b_store.outer, swizzle=s_b_store.inner)
        sStg2 = sm.sStg2.get_tensor(s_b_store.outer, swizzle=s_b_store.inner)
        sStg3 = sm.sStg3.get_tensor(s_b_store.outer, swizzle=s_b_store.inner)
        t_qg_b = mma_tok2.make_fragment_B(sQgT2)
        t_kgqb_b = mma_tok2.make_fragment_B(sKgqB2)
        t_kgqa_a = mma_s.make_fragment_A(sKgqA)
        t_kgks_b = mma_s.make_fragment_B(sKgkS)

        tok_sh = mma_tok.partition_shape_C(self.tok_tile[:2])
        tok_fk = mma_tok.make_fragment_C(cute.append(tok_sh, self.acc_depth))
        t_dq_acc = cute.make_tensor(tp + self.tm_dq, tok_fk.layout)
        t_dkb_acc = cute.make_tensor(tp + self.tm_dkb, tok_fk.layout)
        tok2_sh = mma_tok2.partition_shape_C(self.tok_tile[:2])
        tok2_fk = mma_tok2.make_fragment_C(cute.append(tok2_sh, self.acc_depth))
        t_dka_acc = cute.make_tensor(tp + self.tm_dka, tok2_fk.layout)
        s_sh = mma_s.partition_shape_C(self.s_tile[:2])
        s_fk = mma_s.make_fragment_C(cute.append(s_sh, self.acc_depth))
        t_s_acc = cute.make_tensor(tp + self.tm_s, s_fk.layout)

        # --- Work decode ---
        bx = cute.arch.block_idx()[0]
        gdx = cute.arch.grid_dim()[0]
        num_sequences = Int32(cute.size(chunk_offsets)) - 1
        active_chunks = Int32(chunk_offsets[num_sequences])
        total_work = active_chunks * H
        n_iters = (total_work - bx + gdx - 1) // gdx

        # ///////////////////////////////////////////////////////////////////
        #  CUDA warps
        # ///////////////////////////////////////////////////////////////////
        if warp_id in self.CUDA_WARP_IDS:
            cute.arch.setmaxregister_increase(self.cuda_regs)
            local_tid = tid % n_cuda

            t2r_tok_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(BK // 8), tcgen05.Pack.NONE),
                self.acc_type,
            )
            dq_flat = t_dq_acc[((None, None), 0, 0, None)]
            tc_tok = tcgen05.make_tmem_copy(t2r_tok_atom, dq_flat[(None, None, 0)])
            sl_tok = tc_tok.get_slice(local_tid)
            p_t_dq = sl_tok.partition_S(dq_flat)
            p_t_dkb = sl_tok.partition_S(t_dkb_acc[((None, None), 0, 0, None)])
            p_t_dka = sl_tok.partition_S(t_dka_acc[((None, None), 0, 0, None)])
            id_md = cute.make_identity_tensor((BT, BK))
            coords_md = sl_tok.partition_D(id_md)
            reg_shape = coords_md.shape

            t2r_s_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(BT // 8), tcgen05.Pack.NONE),
                self.acc_type,
            )
            s_flat = t_s_acc[((None, None), 0, 0, None)]
            tc_s = tcgen05.make_tmem_copy(t2r_s_atom, s_flat[(None, None, 0)])
            sl_s = tc_s.get_slice(local_tid)
            p_t_s = sl_s.partition_S(s_flat)
            id_ss = cute.make_identity_tensor((BT, BT))
            coords_ss = sl_s.partition_D(id_ss)
            sreg_shape = coords_ss.shape

            # Vectorized epilogue copies: one thread map (4 rows x 32 col-groups,
            # 4 elems per copy) shared by fp32 128b loads, bf16 64b stores, and
            # fp32 128b stores, so fragments stay coordinate-aligned.
            ep_thr = cute.make_ordered_layout((8, 32), order=(1, 0))
            ep_val = cute.make_layout((1, 4))
            cp_g_f32 = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                ep_thr,
                ep_val,
            )
            cp_g_bf16 = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.io_type, num_bits_per_copy=64
                ),
                ep_thr,
                ep_val,
            )
            ep_g = cp_g_f32.get_slice(local_tid)
            ep_b = cp_g_bf16.get_slice(local_tid)
            ep_row = local_tid // 32  # this thread's row within each 8-row step
            # (BT, BT) A-tile copies: 16 rows x 16 col-groups x 4 elems = 256 thr.
            cp_g_f32h4 = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                cute.make_ordered_layout((16, 16), order=(1, 0)),
                cute.make_layout((1, 4)),
            )
            cp_g_bf16h4 = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.io_type, num_bits_per_copy=64
                ),
                cute.make_ordered_layout((16, 16), order=(1, 0)),
                cute.make_layout((1, 4)),
            )
            ep_gh4 = cp_g_f32h4.get_slice(local_tid)
            ep_bh4 = cp_g_bf16h4.get_slice(local_tid)
            ep_hrow8 = local_tid // 16
            gstride = cute.assume(H * BK, divby=4)

            r2s_atom = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR, self.io_type, self.acc_type, tc_tok
            )
            tc_r2s = cute.make_tiled_copy_D(r2s_atom, tc_tok)
            thr_r2s = tc_r2s.get_slice(local_tid)
            r2s_s_atom = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR, Float32, self.acc_type, tc_s
            )
            tc_r2s_s = cute.make_tiled_copy_D(r2s_s_atom, tc_s)
            thr_r2s_s = tc_r2s_s.get_slice(local_tid)

            it = Int32(0)
            has_work = it < n_iters
            while has_work:
                w = bx + it * gdx
                chunk_idx = w // H
                h_idx = w % H
                _, _, token_start, valid = load_ragged_chunk_work(
                    cu_seqlens, chunk_offsets, chunk_idx, Int32(BT)
                )

                # ---- Phase A operands: aq, ak, kgk ----
                opA_h = popA_P.acquire_and_advance()
                raw_h = praw_C.wait_and_advance()
                kq_h = pkq_C.wait_and_advance()
                # 64-col vectorized reads; masked; forward tiles stored via
                # tiled copies, transposed tiles via scalar stores (reads and
                # value math stay 4-wide).
                ep_hcol0 = (local_tid % 16) * 4
                taqr_p = ep_gh4.partition_S(sAqRaw[(None, None, raw_h.index)])
                takr_p = ep_gh4.partition_S(sAkRaw[(None, None, raw_h.index)])
                taq_p = ep_bh4.partition_D(sAqSt[(None, None, 0)])
                tak_p = ep_bh4.partition_D(sAkSt[(None, None, 0)])
                for rb in cutlass.range_constexpr(cute.size(taqr_p, mode=[1])):
                    prow = ep_hrow8 + rb * 16
                    # beta straight from gmem (L2-hot): the staging barrier is
                    # gone; rows >= valid produce zeros regardless.
                    beta_r = Float32(0.0)
                    if prow < valid:
                        beta_r = g_beta[(token_start + prow, (h_idx, 0))]
                    r_aq = cute.make_fragment_like(taqr_p[(None, 0, None)], Float32)
                    cute.copy(cp_g_f32h4, taqr_p[(None, rb, None)], r_aq)
                    r_ak = cute.make_fragment_like(r_aq, Float32)
                    cute.copy(cp_g_f32h4, takr_p[(None, rb, None)], r_ak)
                    r_a1 = cute.make_fragment_like(taq_p[(None, 0, None)], self.io_type)
                    r_a2 = cute.make_fragment_like(r_a1, self.io_type)
                    for e in cutlass.range_constexpr(cute.size(r_a1)):
                        pcol = ep_hcol0 + e
                        aq_v = Float32(0.0)
                        ak_v = Float32(0.0)
                        if prow < valid and pcol <= prow:
                            aq_v = r_aq[e]
                            ak_v = r_ak[e] * beta_r
                        r_a1[e] = self.io_type(aq_v)
                        r_a2[e] = self.io_type(ak_v)
                    cute.copy(cp_g_bf16h4, r_a1, taq_p[(None, rb, None)])
                    cute.copy(cp_g_bf16h4, r_a2, tak_p[(None, rb, None)])

                ep_col0 = (local_tid % 32) * 4
                grf = cute.make_rmem_tensor((4,), self.acc_type)
                for e in cutlass.range_constexpr(4):
                    grf[e] = sG[(0, ep_col0 + e, raw_h.index)]
                tsg_p = ep_g.partition_S(sG[(None, None, raw_h.index)])
                tsk_p = ep_b.partition_S(sK[(None, None, kq_h.index)])
                tkgk_p = ep_b.partition_D(sKgkSt[(None, None, 0)])
                for rb in cutlass.range_constexpr(cute.size(tsg_p, mode=[1])):
                    prow = ep_row + rb * 8
                    r_g = cute.make_fragment_like(tsg_p[(None, 0, None)], Float32)
                    cute.copy(cp_g_f32, tsg_p[(None, rb, None)], r_g)
                    r_k = cute.make_fragment_like(tsk_p[(None, 0, None)], self.io_type)
                    cute.copy(cp_g_bf16, tsk_p[(None, rb, None)], r_k)
                    r_o = cute.make_fragment_like(r_k, self.io_type)
                    for e in cutlass.range_constexpr(cute.size(r_o)):
                        kgk_v = Float32(0.0)
                        if prow < valid:
                            kgk_v = Float32(r_k[e]) * cute.math.exp2(
                                grf[e] - r_g[e], fastmath=True
                            )
                        r_o[e] = self.io_type(kgk_v)
                    cute.copy(cp_g_bf16, r_o, tkgk_p[(None, rb, None)])
                cute.arch.fence_proxy("async.shared", space="cta")
                opA_h.commit()

                # ---- Phase B operands: qg, kgq (x2 views) ----
                opB_h = popB_P.acquire_and_advance()
                tsq_p = ep_b.partition_S(sQ[(None, None, kq_h.index)])
                tqg_p = ep_b.partition_D(sQgSt[(None, None, 0)])
                tkgqb_p = ep_b.partition_D(sKgqBSt[(None, None, 0)])
                for rb in cutlass.range_constexpr(cute.size(tsg_p, mode=[1])):
                    prow = ep_row + rb * 8
                    r_g = cute.make_fragment_like(tsg_p[(None, 0, None)], Float32)
                    cute.copy(cp_g_f32, tsg_p[(None, rb, None)], r_g)
                    r_k = cute.make_fragment_like(tsk_p[(None, 0, None)], self.io_type)
                    cute.copy(cp_g_bf16, tsk_p[(None, rb, None)], r_k)
                    r_q = cute.make_fragment_like(r_k, self.io_type)
                    cute.copy(cp_g_bf16, tsq_p[(None, rb, None)], r_q)
                    r_oq = cute.make_fragment_like(r_k, self.io_type)
                    r_ok = cute.make_fragment_like(r_k, self.io_type)
                    for e in cutlass.range_constexpr(cute.size(r_oq)):
                        qg_v = Float32(0.0)
                        kgq_v = Float32(0.0)
                        if prow < valid:
                            gq = cute.math.exp2(r_g[e] - grf[e], fastmath=True)
                            qg_v = Float32(r_q[e]) * gq
                            kgq_v = Float32(r_k[e]) * gq
                        r_oq[e] = self.io_type(qg_v)
                        r_ok[e] = self.io_type(kgq_v)
                    cute.copy(cp_g_bf16, r_oq, tqg_p[(None, rb, None)])
                    cute.copy(cp_g_bf16, r_ok, tkgqb_p[(None, rb, None)])
                cute.arch.fence_proxy("async.shared", space="cta")
                opB_h.commit()
                kq_h.release()  # k/q raws done: TMA prefetches n+1 during epilogue

                # ---- Drain all three accumulators as RAW bf16 stages ----
                # All operand buffers are free once accB is committed (the MMA
                # warp released opA at accA-commit and opB at accB-commit), so
                # the raw accs stage into sKgk/sQg/sKgqB and every epilogue
                # multiply happens in the coalesced pass domain below.
                accA_h = paccA_C.wait_and_advance()
                tmp_reg = cute.make_rmem_tensor(reg_shape, self.acc_type)
                tmp_bf16 = cute.make_rmem_tensor(reg_shape, self.io_type)

                if local_tid < 128:
                    cute.copy(tc_tok, p_t_dq[(None, None, None, accA_h.index)], tmp_reg)
                    cute.arch.fence_view_async_tmem_load()
                    tmp_bf16.store(tmp_reg.load().to(self.io_type))
                    cute.copy(
                        tc_r2s,
                        tc_r2s.retile(tmp_bf16),
                        thr_r2s.partition_D(sStg1[(None, None, 0)]),
                    )
                    cute.copy(tc_tok, p_t_dkb[(None, None, None, accA_h.index)], tmp_reg)
                    cute.arch.fence_view_async_tmem_load()
                    tmp_bf16.store(tmp_reg.load().to(self.io_type))
                    cute.copy(
                        tc_r2s,
                        tc_r2s.retile(tmp_bf16),
                        thr_r2s.partition_D(sStg2[(None, None, 0)]),
                    )
                accA_h.release()
                accB_h = paccB_C.wait_and_advance()
                if local_tid < 128:
                    cute.copy(tc_tok, p_t_dka[(None, None, None, accB_h.index)], tmp_reg)
                    cute.arch.fence_view_async_tmem_load()
                    tmp_bf16.store(tmp_reg.load().to(self.io_type))
                    cute.copy(
                        tc_r2s,
                        tc_r2s.retile(tmp_bf16),
                        thr_r2s.partition_D(sStg3[(None, None, 0)]),
                    )
                self.cuda_bar.arrive_and_wait()
                self.cuda_bar.arrive_and_wait()

                # ---- Coalesced epilogue passes (all math here) ----
                gin = self._gchunk(g_dqin.iterator, token_start, h_idx, Float32, H, gstride)
                gout = self._gchunk(g_dq2.iterator, token_start, h_idx, self.io_type, H, gstride)
                gin_k = self._gchunk(g_dkin.iterator, token_start, h_idx, Float32, H, gstride)
                gout_k = self._gchunk(g_dk2.iterator, token_start, h_idx, self.io_type, H, gstride)
                gin_g = self._gchunk(g_dgin.iterator, token_start, h_idx, Float32, H, gstride)
                gout_g = self._gchunk(g_dg2.iterator, token_start, h_idx, Float32, H, gstride)
                tin_q = ep_g.partition_S(gin)
                tout_q = ep_b.partition_D(gout)
                tin_k = ep_g.partition_S(gin_k)
                tout_k = ep_b.partition_D(gout_k)
                tin_g = ep_g.partition_S(gin_g)
                tout_g = ep_g.partition_D(gout_g)
                tsg = ep_g.partition_S(sG[(None, None, raw_h.index)])
                tqg = ep_b.partition_S(sQgSt[(None, None, 0)])
                tkgq = ep_b.partition_S(sKgqBSt[(None, None, 0)])
                tkgk = ep_b.partition_S(sKgkSt[(None, None, 0)])
                tdq = ep_b.partition_S(sStg1[(None, None, 0)])
                tdkb = ep_b.partition_S(sStg2[(None, None, 0)])
                tdka = ep_b.partition_S(sStg3[(None, None, 0)])

                for rb in cutlass.range_constexpr(cute.size(tin_q, mode=[1])):
                    if ep_row + rb * 8 < valid:
                        r_g = cute.make_fragment_like(tsg[(None, 0, None)], Float32)
                        cute.copy(cp_g_f32, tsg[(None, rb, None)], r_g)
                        r_dqr = cute.make_fragment_like(tdq[(None, 0, None)], self.io_type)
                        cute.copy(cp_g_bf16, tdq[(None, rb, None)], r_dqr)
                        r_dkbr = cute.make_fragment_like(tdkb[(None, 0, None)], self.io_type)
                        cute.copy(cp_g_bf16, tdkb[(None, rb, None)], r_dkbr)
                        r_dkar = cute.make_fragment_like(tdka[(None, 0, None)], self.io_type)
                        cute.copy(cp_g_bf16, tdka[(None, rb, None)], r_dkar)
                        r_qg = cute.make_fragment_like(tqg[(None, 0, None)], self.io_type)
                        cute.copy(cp_g_bf16, tqg[(None, rb, None)], r_qg)
                        r_kgq = cute.make_fragment_like(r_qg, self.io_type)
                        cute.copy(cp_g_bf16, tkgq[(None, rb, None)], r_kgq)
                        r_kgk = cute.make_fragment_like(r_qg, self.io_type)
                        cute.copy(cp_g_bf16, tkgk[(None, rb, None)], r_kgk)
                        r_i1 = cute.make_fragment_like(tin_q[(None, 0, None)], Float32)
                        cute.copy(cp_g_f32, tin_q[(None, rb, None)], r_i1)
                        r_i2 = cute.make_fragment_like(tin_k[(None, 0, None)], Float32)
                        cute.copy(cp_g_f32, tin_k[(None, rb, None)], r_i2)
                        r_i3 = cute.make_fragment_like(tin_g[(None, 0, None)], Float32)
                        cute.copy(cp_g_f32, tin_g[(None, rb, None)], r_i3)

                        r_o1 = cute.make_fragment_like(r_qg, self.io_type)
                        r_o2 = cute.make_fragment_like(r_qg, self.io_type)
                        r_o3 = cute.make_fragment_like(r_i3, Float32)
                        for e in cutlass.range_constexpr(cute.size(r_o1)):
                            gq = cute.math.exp2(r_g[e] - grf[e], fastmath=True)
                            dq_r = Float32(r_dqr[e])
                            dkb_r = Float32(r_dkbr[e])
                            dka_r = Float32(r_dkar[e])
                            r_o1[e] = self.io_type(r_i1[e] + dq_r * gq)
                            r_o2[e] = self.io_type(r_i2[e] + dkb_r * gq + dka_r / gq)
                            # dg identity via staged gated operands:
                            # q*dq_s = qg*dq_raw, k*dkb_s = kgq*dkb_raw,
                            # k*dka_s = kgk*dka_raw (gk = 1/gq folded into kgk).
                            r_o3[e] = Float32(
                                (
                                    r_i3[e]
                                    + Float32(r_qg[e]) * dq_r
                                    + Float32(r_kgq[e]) * dkb_r
                                    - Float32(r_kgk[e]) * dka_r
                                )
                                * Float32(LN2)
                            )
                        cute.copy(cp_g_bf16, r_o1, tout_q[(None, rb, None)])
                        cute.copy(cp_g_bf16, r_o2, tout_k[(None, rb, None)])
                        cute.copy(cp_g_f32, r_o3, tout_g[(None, rb, None)])

                # ---- db: rowsum(ak * S) via sAqRaw staging ----
                s_reg = cute.make_rmem_tensor(sreg_shape, self.acc_type)
                if local_tid < 128:
                    cute.copy(tc_s, p_t_s[(None, None, None, accB_h.index)], s_reg)
                    cute.arch.fence_view_async_tmem_load()
                    cute.copy(
                        tc_r2s_s,
                        tc_r2s_s.retile(s_reg),
                        thr_r2s_s.partition_D(sDbSt[(None, None, raw_h.index)]),
                    )
                accB_h.release()
                self.cuda_bar.arrive_and_wait()
                # Row-per-thread masked dot: db[r] = sum_{j<=r} ak[r,j] * S[r,j].
                if local_tid < 2 * BT:
                    dbrow = local_tid % BT
                    dbhalf = local_tid // BT
                    total = Float32(0.0)
                    if dbrow < valid:
                        for j in cutlass.range_constexpr(BT // 2):
                            jj = dbhalf * (BT // 2) + j
                            if jj <= dbrow:
                                total += (
                                    sDbSt[(dbrow, jj, raw_h.index)]
                                    * sAkRaw[(dbrow, jj, raw_h.index)]
                                )
                    sBetaRed[local_tid] = total
                self.cuda_bar.arrive_and_wait()
                if local_tid < BT and local_tid < valid:
                    g_db2[(token_start + local_tid, (h_idx, 0))] = (
                        g_dbin[(token_start + local_tid, (h_idx, 0))]
                        + sBetaRed[local_tid]
                        + sBetaRed[local_tid + BT]
                    )
                self.cuda_bar.arrive_and_wait()
                raw_h.release()

                it = it + 1
                has_work = it < n_iters

        # ///////////////////////////////////////////////////////////////////
        #  LOAD warp
        # ///////////////////////////////////////////////////////////////////
        elif warp_id == self.LOAD_WARP_ID:
            cute.arch.setmaxregister_decrease(self.aux_regs)
            it = Int32(0)
            has_work = it < n_iters
            while has_work:
                w = bx + it * gdx
                chunk_idx = w // H
                h_idx = w % H
                _, _, token_start, valid = load_ragged_chunk_work(
                    cu_seqlens, chunk_offsets, chunk_idx, Int32(BT)
                )

                def off(d, ts=token_start):
                    return cute.domain_offset((ts, 0, (0, 0)), d)

                _, sSAq, gSAq = self._part_epi(
                    atom_aq, off(desc_aq)[None, None, (h_idx, 0)], (BT, BT), sAqRaw
                )
                _, sSAk, gSAk = self._part_epi(
                    atom_ak, off(desc_ak)[None, None, (h_idx, 0)], (BT, BT), sAkRaw
                )
                _, sSK, gSK = self._part_epi(
                    atom_k, off(desc_k)[None, None, (h_idx, 0)], (BT, BK), sK
                )
                _, sSG, gSG = self._part_epi(
                    atom_g, off(desc_g)[None, None, (h_idx, 0)], (BT, BK), sG
                )
                _, sSQ, gSQ = self._part_epi(
                    atom_q, off(desc_q)[None, None, (h_idx, 0)], (BT, BK), sQ
                )

                kqh = pkq_P.acquire_and_advance()
                rh = praw_P.acquire_and_advance()
                cute.copy(
                    atom=atom_aq,
                    src=gSAq[(None, 0, 0)],
                    dst=sSAq[None, rh.index],
                    tma_bar_ptr=rh.barrier,
                )
                cute.copy(
                    atom=atom_ak,
                    src=gSAk[(None, 0, 0)],
                    dst=sSAk[None, rh.index],
                    tma_bar_ptr=rh.barrier,
                )
                cute.copy(
                    atom=atom_k,
                    src=gSK[(None, 0, 0)],
                    dst=sSK[None, kqh.index],
                    tma_bar_ptr=kqh.barrier,
                )
                cute.copy(
                    atom=atom_g,
                    src=gSG[(None, 0, 0)],
                    dst=sSG[None, rh.index],
                    tma_bar_ptr=rh.barrier,
                )
                cute.copy(
                    atom=atom_q,
                    src=gSQ[(None, 0, 0)],
                    dst=sSQ[None, kqh.index],
                    tma_bar_ptr=kqh.barrier,
                )
                it = it + 1
                has_work = it < n_iters

        # ///////////////////////////////////////////////////////////////////
        #  MMA warp
        # ///////////////////////////////////////////////////////////////////
        elif warp_id == self.MMA_WARP_ID:
            cute.arch.setmaxregister_decrease(self.aux_regs)
            it = Int32(0)
            has_work = it < n_iters
            while has_work:
                # Phase A: dq = aq @ kgk^T, dkb = ak @ kgk^T
                opA_h = popA_C.wait_and_advance()
                accA_h = paccA_P.acquire_and_advance()
                accB_h = paccB_P.acquire_and_advance()
                for kp in cutlass.range(cute.size(t_aq_a, mode=[2]), unroll_full=True):
                    mma_tok.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma_tok,
                        t_dq_acc[None, None, None, accA_h.index],
                        t_aq_a[None, None, kp, 0],
                        t_kgk_b[None, None, kp, 0],
                        t_dq_acc[None, None, None, accA_h.index],
                    )
                for kp in cutlass.range(cute.size(t_ak_a, mode=[2]), unroll_full=True):
                    mma_tok.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma_tok,
                        t_dkb_acc[None, None, None, accA_h.index],
                        t_ak_a[None, None, kp, 0],
                        t_kgk_b[None, None, kp, 0],
                        t_dkb_acc[None, None, None, accA_h.index],
                    )
                accA_h.commit()

                # Phase B: dka/dkc, then S.
                opB_h = popB_C.wait_and_advance()
                for kp in cutlass.range(cute.size(t_aqt_a, mode=[2]), unroll_full=True):
                    mma_tok2.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma_tok2,
                        t_dka_acc[None, None, None, accB_h.index],
                        t_aqt_a[None, None, kp, 0],
                        t_qg_b[None, None, kp, 0],
                        t_dka_acc[None, None, None, accB_h.index],
                    )
                for kp in cutlass.range(cute.size(t_akbt_a, mode=[2]), unroll_full=True):
                    mma_tok2.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(True))
                    cute.gemm(
                        mma_tok2,
                        t_dka_acc[None, None, None, accB_h.index],
                        t_akbt_a[None, None, kp, 0],
                        t_kgqb_b[None, None, kp, 0],
                        t_dka_acc[None, None, None, accB_h.index],
                    )
                for kp in cutlass.range(cute.size(t_kgqa_a, mode=[2]), unroll_full=True):
                    mma_s.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                    cute.gemm(
                        mma_s,
                        t_s_acc[None, None, None, accB_h.index],
                        t_kgqa_a[None, None, kp, 0],
                        t_kgks_b[None, None, kp, 0],
                        t_s_acc[None, None, None, accB_h.index],
                    )
                opA_h.release()  # kgk (shared with the S B-operand) now free
                accB_h.commit()
                opB_h.release()

                it = it + 1
                has_work = it < n_iters

        if warp_id in self.IDLE_WARP_IDS:
            cute.arch.setmaxregister_decrease(self.aux_regs)

        tmem.relinquish_alloc_permit()
        self.tmem_free_bar.arrive_and_wait()
        tmem.free(tp)

    @cute.jit
    def _gchunk(self, base_iter, token_start, h_idx, dtype, H, gstride):
        """(BT, BK) gmem tile view at (token_start, head) for vectorized copies."""
        raw = base_iter + token_start * H * self.BK + h_idx * self.BK
        ptr = cute.make_ptr(dtype, raw.toint(), cute.AddressSpace.gmem, assumed_align=16)
        return cute.make_tensor(ptr, cute.make_layout((self.BT, self.BK), stride=(gstride, 1)))

    @cute.jit
    def _part_epi(self, atom, g_mnl, tile, s_buf):
        """Partition for epilogue-style TMA (delta_h idiom)."""
        g_div = cute.flat_divide(g_mnl, tile)
        sg = cute.group_modes(s_buf, 0, 2)
        gg = cute.group_modes(g_div, 0, 2)
        ss, gs = cpasync.tma_partition(atom, 0, cute.make_layout(1), sg, gg)
        return atom, ss, gs

    def _launch_grid(self):
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA compilation requires a CUDA target with an SM count")
        return (sm_count * self.grid_waves, 1, 1)


# ============================================================================
# Compile cache + public wrapper
# ============================================================================


@jit_cache
def _compile_intra_engine_bwd(
    H: int,
    head_dim: int,
    chunk_size: int,
    kq_depth: int = 1,
    araw_depth: int = 1,
    grid_waves: int = 1,
):
    op = KdaIntraEngine(
        chunk_size=chunk_size,
        head_dim=head_dim,
        num_heads=H,
        mode="bwd",
        kq_depth=kq_depth,
        araw_depth=araw_depth,
        grid_waves=grid_waves,
    )
    st, sn = cute.sym_int(), cute.sym_int()

    def fk(dtype, dim):
        return make_fake_compact_tensor(
            dtype, (st, H, dim), stride_order=(2, 1, 0), assumed_align=128
        )

    def fv(dtype):
        return make_fake_compact_tensor(dtype, (st, H), stride_order=(1, 0), assumed_align=8)

    args = [
        fk(cutlass.BFloat16, head_dim),  # q
        fk(cutlass.BFloat16, head_dim),  # k
        fk(cutlass.Float32, head_dim),  # g
        fv(cutlass.Float32),  # beta
        fk(cutlass.Float32, chunk_size),  # dAqk
        fk(cutlass.Float32, chunk_size),  # dAkk
        fk(cutlass.Float32, head_dim),  # dq
        fk(cutlass.Float32, head_dim),  # dk
        fv(cutlass.Float32),  # db
        fk(cutlass.Float32, head_dim),  # dg
        fk(cutlass.BFloat16, head_dim),  # dq2
        fk(cutlass.BFloat16, head_dim),  # dk2
        fk(cutlass.Float32, head_dim),  # dg2
        fv(cutlass.Float32),  # db2
    ]
    cu = make_fake_compact_tensor(cutlass.Int32, (sn,), stride_order=(0,), assumed_align=4)
    offs = make_fake_compact_tensor(cutlass.Int32, (sn,), stride_order=(0,), assumed_align=4)
    return compile_tvm_ffi(op, *args, cu, offs, 1, H, 1, name=op.get_name())


def kda_intra_engine_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    metadata: RaggedChunkMetadata,
    *,
    kq_depth: int = 1,
    araw_depth: int = 1,
    grid_waves: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Engine-backed intra-chunk backward matching chunk_kda_bwd_intra's contract."""
    batch, tokens, heads, head_dim = q.shape
    assert batch == 1 and head_dim == 128
    dq2 = torch.empty_like(q, memory_format=torch.contiguous_format)
    dk2 = torch.empty_like(k, memory_format=torch.contiguous_format)
    dg2 = torch.empty_like(g, memory_format=torch.contiguous_format)
    db2 = torch.empty_like(db, memory_format=torch.contiguous_format)
    compiled = _compile_intra_engine_bwd(
        heads, head_dim, metadata.chunk_size, kq_depth, araw_depth, grid_waves
    )
    compiled(
        q[0].contiguous(),
        k[0].contiguous(),
        g[0].contiguous(),
        beta[0].contiguous(),
        dAqk[0].contiguous(),
        dAkk[0].contiguous(),
        dq[0].contiguous(),
        dk[0].contiguous(),
        db[0].contiguous(),
        dg[0].contiguous(),
        dq2[0],
        dk2[0],
        dg2[0],
        db2[0],
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        tokens,
        heads,
        metadata.capacity,
    )
    return dq2, dk2, dg2, db2


# ============================================================================
# Forward mode: Aqk/Akk on the same engine machinery
# ============================================================================


class KdaIntraFwdEngine:
    """Forward intra stage on the engine architecture (mode="fwd").

    Two S-shaped gemms over the same gated operands the backward engine
    produces (see the bwd class):

        Aqk[i, j] = scale * sum_d qg[i, d] * kgk[j, d]   (non-strict causal)
        Akk[i, j] = beta[i] * sum_d kgq[i, d] * kgk[j, d]  (strict lower)

    Warp plan: CUDA(0-7) production + drain/store, LOAD(8) TMA, MMA(9),
    idle(10-11) padding warp-group 3 for setmaxregister alignment. The 16x16
    diagonal-block Neumann inverses (K4b's Akkd input) run in a separate tiny
    Triton kernel invoked by the Python wrapper.
    """

    CUDA_WARP_IDS = (0, 1, 2, 3, 4, 5, 6, 7)
    LOAD_WARP_ID = 8
    MMA_WARP_ID = 9
    IDLE_WARP_IDS = (10, 11)  # pad warp-group 2 for setmaxregister alignment
    WARP_SZ = 32
    N_WARPS = 12
    CTA_THREADS = N_WARPS * WARP_SZ  # 384

    def __init__(
        self,
        chunk_size: int = 64,
        head_dim: int = 128,
        num_heads: int | None = None,
        io_type=cutlass.BFloat16,
        acc_type=cutlass.Float32,
    ):
        assert chunk_size == 64 and head_dim == 128
        self.BT = chunk_size
        self.BK = head_dim
        self.io_type = io_type
        self.acc_type = acc_type
        self.num_heads = num_heads
        self.s_tile = (self.BT, self.BT, self.BK)
        self.raw_depth = 2  # q/k TMA prefetch depth (g stays depth-1: smem budget)
        self.op_depth = 2
        self.acc_depth = 2
        self.cuda_regs = 208
        self.aux_regs = 48
        self.cluster = (1, 1, 1)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tmem_free_bar = pipeline.NamedBarrier(barrier_id=2, num_threads=self.CTA_THREADS)
        self.cuda_bar = pipeline.NamedBarrier(
            barrier_id=3, num_threads=self.WARP_SZ * len(self.CUDA_WARP_IDS)
        )
        self.align = 1024

    def get_name(self) -> str:
        head_tag = f"_h{self.num_heads}" if self.num_heads is not None else ""
        return f"kda_intra_engine_fwd{head_tag}_k{self.BK}_bt{self.BT}"

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
        cu_seqlens_in: cute.Tensor,
        chunk_offsets_in: cute.Tensor,
        T: Int32,
        H: Int32,
        capacity: Int32,
        stream,
    ):
        BT, BK = self.BT, self.BK

        def tok_view(t, dim):
            return cute.make_tensor(
                t.iterator,
                cute.make_layout((T, dim, (H, 1)), stride=(H * dim, 1, (dim, 0))),
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
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_type,
            self.cta_group,
            self.s_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # Per-16-column strip MMA: each strip's operands are rebased at that
        # strip's own row-start gate, bounding every exp2 to a 16-token range
        # (see "Notes: causality and gk scale" in the module docstring).
        strip_tile = (BT, 16, BK)
        mma16 = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_type,
            self.cta_group,
            strip_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        s_c = mma_s.partition_shape_C(self.s_tile[:2])
        s_fk_probe = mma_s.make_fragment_C(cute.append(s_c, self.acc_depth))
        n_s = tcgen05.find_tmem_tensor_col_offset(s_fk_probe)
        # tmem addresses pack (lane << 16 | col); appended acc stages land at a
        # LANE offset (stride 2^20 for the [64,64] fp32 fragment), not +cols.
        # Take the stage stride from the fragment layout, never from col math.
        self.tm_stage = s_fk_probe.layout.stride[3]
        self.tm_aqk = 0
        self.tm_akk = n_s
        raw_tot = 2 * n_s
        tot = 1
        for _ in cutlass.range_constexpr(10):
            if cutlass.const_expr(tot < raw_tot):
                tot *= 2
        self.tm_tot = tot

        tma_ld = cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        s_kraw = sm100_utils.make_smem_layout_epi(
            self.io_type, utils.LayoutEnum.ROW_MAJOR, (BT, BK), self.raw_depth
        )
        s_graw = sm100_utils.make_smem_layout_epi(Float32, utils.LayoutEnum.ROW_MAJOR, (BT, BK), 1)
        atom_q, desc_q = cpasync.make_tiled_tma_atom(
            tma_ld, g_q, cute.select(s_kraw, mode=[0, 1]), (BT, BK)
        )
        atom_k, desc_k = cpasync.make_tiled_tma_atom(
            tma_ld, g_k, cute.select(s_kraw, mode=[0, 1]), (BT, BK)
        )
        atom_g, desc_g = cpasync.make_tiled_tma_atom(
            tma_ld, g_g, cute.select(s_graw, mode=[0, 1]), (BT, BK)
        )
        self.k_bytes = cute.size_in_bytes(self.io_type, cute.select(s_kraw, mode=[0, 1]))
        self.g_bytes = cute.size_in_bytes(Float32, cute.select(s_graw, mode=[0, 1]))
        s_sa_op = sm100_utils.make_smem_layout_a(mma16, strip_tile, self.io_type, self.op_depth)
        s_sb_op = sm100_utils.make_smem_layout_b(mma16, strip_tile, self.io_type, self.op_depth)
        s_op_store = sm100_utils.make_smem_layout_epi(
            self.io_type, utils.LayoutEnum.ROW_MAJOR, (BT, BK), self.op_depth
        )
        s_opb_store = sm100_utils.make_smem_layout_epi(
            self.io_type, utils.LayoutEnum.ROW_MAJOR, (16, BK), self.op_depth
        )

        @cute.struct
        class SharedF:
            bar_raw: cute.struct.MemRange[Int64, self.raw_depth * 2]
            bar_rawg: cute.struct.MemRange[Int64, 1 * 2]
            bar_op: cute.struct.MemRange[Int64, self.op_depth * 2]
            bar_acc: cute.struct.MemRange[Int64, self.acc_depth * 2]
            tmem_buf: Int32
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_kraw)], self.align
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_kraw)], self.align
            ]
            sG: cute.struct.Align[cute.struct.MemRange[Float32, cute.cosize(s_graw)], self.align]
            sBeta: cute.struct.Align[cute.struct.MemRange[Float32, self.BT], 128]
            sQg: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_sa_op)], self.align
            ]
            sKgq: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_sa_op)], self.align
            ]
            sKgk: cute.struct.Align[
                cute.struct.MemRange[self.io_type, cute.cosize(s_sb_op)], self.align
            ]
            sSt1: cute.struct.Align[cute.struct.MemRange[Float32, self.BT * self.BT], self.align]
            sSt2: cute.struct.Align[cute.struct.MemRange[Float32, self.BT * self.BT], self.align]

        self.shared_type = SharedF
        self.kernel(
            mma_s,
            mma16,
            g_g,
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
            H,
            capacity,
        ).launch(
            grid=self._launch_grid(),
            block=[self.CTA_THREADS, 1, 1],
            cluster=self.cluster,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mma_s: cute.TiledMma,
        mma16: cute.TiledMma,
        g_g: cute.Tensor,
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
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        H: Int32,
        capacity: Int32,
    ):
        BT, BK = self.BT, self.BK
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tid, _, _ = cute.arch.thread_idx()

        sa = utils.SmemAllocator()
        sm = sa.allocate(self.shared_type)
        n_cuda = self.WARP_SZ * len(self.CUDA_WARP_IDS)

        praw_P, praw_C = pipeline.PipelineTmaAsync.create(
            num_stages=self.raw_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.CUDA_WARP_IDS)
            ),
            tx_count=2 * self.k_bytes,
            barrier_storage=sm.bar_raw.data_ptr(),
        ).make_participants()
        prawg_P, prawg_C = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.CUDA_WARP_IDS)
            ),
            tx_count=self.g_bytes,
            barrier_storage=sm.bar_rawg.data_ptr(),
        ).make_participants()
        pop_P, pop_C = pipeline.PipelineAsyncUmma.create(
            num_stages=self.op_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, n_cuda),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            barrier_storage=sm.bar_op.data_ptr(),
        ).make_participants()
        pacc_P, pacc_C = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_depth,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, n_cuda),
            barrier_storage=sm.bar_acc.data_ptr(),
        ).make_participants()

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.CTA_THREADS)
        tmem = utils.TmemAllocator(
            sm.tmem_buf, barrier_for_retrieve=tmem_bar, allocator_warp_id=self.LOAD_WARP_ID
        )
        tmem.allocate(self.tm_tot)
        tmem.wait_for_alloc()
        tp = tmem.retrieve_ptr(self.acc_type)

        sQ = sm.sQ.get_tensor(s_kraw.outer, swizzle=s_kraw.inner)
        sK = sm.sK.get_tensor(s_kraw.outer, swizzle=s_kraw.inner)
        sG = sm.sG.get_tensor(s_graw.outer, swizzle=s_graw.inner)
        sBeta = sm.sBeta.get_tensor(cute.make_layout((BT,)))
        sQg = sm.sQg.get_tensor(s_sa_op.outer, swizzle=s_sa_op.inner)
        sQgSt = sm.sQg.get_tensor(s_op_store.outer, swizzle=s_op_store.inner)
        sKgq = sm.sKgq.get_tensor(s_sa_op.outer, swizzle=s_sa_op.inner)
        sKgqSt = sm.sKgq.get_tensor(s_op_store.outer, swizzle=s_op_store.inner)
        sKgk = sm.sKgk.get_tensor(s_sb_op.outer, swizzle=s_sb_op.inner)
        sKgkSt = sm.sKgk.get_tensor(s_opb_store.outer, swizzle=s_opb_store.inner)
        st_lay = cute.make_layout((BT, BT), stride=(BT, 1))
        sSt1 = sm.sSt1.get_tensor(st_lay)
        sSt2 = sm.sSt2.get_tensor(st_lay)

        t_qg_a = mma16.make_fragment_A(sQg)
        t_kgq_a = mma16.make_fragment_A(sKgq)
        t_kgk_b = mma16.make_fragment_B(sKgk)
        s_f16 = mma16.make_fragment_C(mma16.partition_shape_C((BT, 16)))
        s_sh = mma_s.partition_shape_C(self.s_tile[:2])
        s_fk = mma_s.make_fragment_C(cute.append(s_sh, self.acc_depth))
        t_aqk_acc = cute.make_tensor(tp + self.tm_aqk, s_fk.layout)
        t_akk_acc = cute.make_tensor(tp + self.tm_akk, s_fk.layout)

        bx = cute.arch.block_idx()[0]
        gdx = cute.arch.grid_dim()[0]
        num_sequences = Int32(cute.size(chunk_offsets)) - 1
        active_chunks = Int32(chunk_offsets[num_sequences])
        total_work = active_chunks * H
        n_iters = (total_work - bx + gdx - 1) // gdx

        if warp_id in self.CUDA_WARP_IDS:
            cute.arch.setmaxregister_increase(self.cuda_regs)
            local_tid = tid % n_cuda

            t2r_atom = cute.make_copy_atom(
                tcgen05.Ld16x256bOp(tcgen05.Repetition(BT // 8), tcgen05.Pack.NONE),
                self.acc_type,
            )
            aqk_flat = t_aqk_acc[((None, None), 0, 0, None)]
            tc_s = tcgen05.make_tmem_copy(t2r_atom, aqk_flat[(None, None, 0)])
            sl_s = tc_s.get_slice(local_tid % 128)
            p_t_aqk = sl_s.partition_S(aqk_flat)
            p_t_akk = sl_s.partition_S(t_akk_acc[((None, None), 0, 0, None)])
            id_ss = cute.make_identity_tensor((BT, BT))
            coords_ss = sl_s.partition_D(id_ss)
            sreg_shape = coords_ss.shape
            r2s_atom = sm100_utils.get_smem_store_op(
                utils.LayoutEnum.ROW_MAJOR, Float32, self.acc_type, tc_s
            )
            tc_r2s = cute.make_tiled_copy_D(r2s_atom, tc_s)
            thr_r2s = tc_r2s.get_slice(local_tid % 128)

            ep_thr = cute.make_ordered_layout((8, 32), order=(1, 0))
            ep_val = cute.make_layout((1, 4))
            cp_f32 = cute.make_tiled_copy_tv(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128),
                ep_thr,
                ep_val,
            )
            cp_bf16 = cute.make_tiled_copy_tv(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.io_type, num_bits_per_copy=64
                ),
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
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), self.io_type, num_bits_per_copy=64
                ),
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
                    w = bx + it * gdx
                    chunk_idx = w // H
                    h_idx = w % H
                    _, _, token_start, valid = load_ragged_chunk_work(
                        cu_seqlens, chunk_offsets, chunk_idx, Int32(BT)
                    )
                    raw_h = praw_C.wait_and_advance()
                    g_h = prawg_C.wait_and_advance()
                    tsg_p = ep_g.partition_S(sG[(None, None, g_h.index)])
                    tsk_p = ep_b.partition_S(sK[(None, None, raw_h.index)])
                    tsq_p = ep_b.partition_S(sQ[(None, None, raw_h.index)])
                    for sc in cutlass.range_constexpr(4):
                        row0 = sc * 16
                        op_h = pop_P.acquire_and_advance()
                        grf = cute.make_rmem_tensor((4,), self.acc_type)
                        for e in cutlass.range_constexpr(4):
                            grf[e] = sG[(row0, ep_col0 + e, g_h.index)]
                        tqg_p = ep_b.partition_D(sQgSt[(None, None, op_h.index)])
                        tkgq_p = ep_b.partition_D(sKgqSt[(None, None, op_h.index)])
                        tkgk_p = ep_b.partition_D(sKgkSt[(None, None, op_h.index)])
                        for rb in cutlass.range_constexpr(cute.size(tsg_p, mode=[1])):
                            prow = ep_row + rb * 8
                            if rb * 8 + 8 > row0:
                                r_g = cute.make_fragment_like(tsg_p[(None, 0, None)], Float32)
                                r_k = cute.make_fragment_like(tsk_p[(None, 0, None)], self.io_type)
                                r_q = cute.make_fragment_like(r_k, self.io_type)
                                cute.copy(cp_f32, tsg_p[(None, rb, None)], r_g)
                                cute.copy(cp_bf16, tsk_p[(None, rb, None)], r_k)
                                cute.copy(cp_bf16, tsq_p[(None, rb, None)], r_q)
                                r_qg = cute.make_fragment_like(r_k, self.io_type)
                                r_kq = cute.make_fragment_like(r_k, self.io_type)
                                for e in cutlass.range_constexpr(cute.size(r_qg)):
                                    qg_v = Float32(0.0)
                                    kgq_v = Float32(0.0)
                                    if prow < valid and prow >= row0:
                                        gq = cute.math.exp2(r_g[e] - grf[e], fastmath=True)
                                        qg_v = Float32(r_q[e]) * gq
                                        kgq_v = Float32(r_k[e]) * gq
                                    r_qg[e] = self.io_type(qg_v)
                                    r_kq[e] = self.io_type(kgq_v)
                                cute.copy(cp_bf16, r_qg, tqg_p[(None, rb, None)])
                                cute.copy(cp_bf16, r_kq, tkgq_p[(None, rb, None)])
                        for rb2 in cutlass.range_constexpr(2):
                            srow = row0 + ep_row + rb2 * 8
                            r_gb = cute.make_fragment_like(tsg_p[(None, 0, None)], Float32)
                            r_kb = cute.make_fragment_like(tsk_p[(None, 0, None)], self.io_type)
                            cute.copy(cp_f32, tsg_p[(None, sc * 2 + rb2, None)], r_gb)
                            cute.copy(cp_bf16, tsk_p[(None, sc * 2 + rb2, None)], r_kb)
                            r_kk = cute.make_fragment_like(r_kb, self.io_type)
                            for e in cutlass.range_constexpr(cute.size(r_kk)):
                                kgk_v = Float32(0.0)
                                if srow < valid:
                                    kgk_v = Float32(r_kb[e]) * cute.math.exp2(
                                        grf[e] - r_gb[e], fastmath=True
                                    )
                                r_kk[e] = self.io_type(kgk_v)
                            cute.copy(cp_bf16, r_kk, tkgk_p[(None, rb2, None)])
                        cute.arch.fence_proxy("async.shared", space="cta")
                        op_h.commit()
                    raw_h.release()
                    g_h.release()

                # ---- EPILOGUE unit `it - 1` ----
                if it > 0:
                    ep_it = it - 1
                    w = bx + ep_it * gdx
                    chunk_idx = w // H
                    h_idx = w % H
                    _, _, token_start, valid = load_ragged_chunk_work(
                        cu_seqlens, chunk_offsets, chunk_idx, Int32(BT)
                    )
                    if local_tid < BT:
                        b_val = Float32(0.0)
                        if local_tid < valid:
                            b_val = g_beta[(token_start + local_tid, (h_idx, 0))]
                        sBeta[local_tid] = b_val

                    acc_h = pacc_C.wait_and_advance()
                    s_reg = cute.make_rmem_tensor(sreg_shape, self.acc_type)
                    if local_tid < 128:
                        cute.copy(tc_s, p_t_aqk[(None, None, None, acc_h.index)], s_reg)
                        cute.arch.fence_view_async_tmem_load()
                        cute.copy(tc_r2s, tc_r2s.retile(s_reg), thr_r2s.partition_D(sSt1))
                        cute.copy(tc_s, p_t_akk[(None, None, None, acc_h.index)], s_reg)
                        cute.arch.fence_view_async_tmem_load()
                        cute.copy(tc_r2s, tc_r2s.retile(s_reg), thr_r2s.partition_D(sSt2))
                    acc_h.release()
                    self.cuda_bar.arrive_and_wait()

                    gstride_a = cute.assume(H * BT, divby=4)
                    gaq = cute.make_tensor(
                        cute.make_ptr(
                            self.io_type,
                            (g_aqk.iterator + token_start * H * BT + h_idx * BT).toint(),
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
                                    aqk_v = sSt1[(row, col)] * scale
                                r_a[e] = self.io_type(aqk_v)
                            cute.copy(cp_st_b, r_a, taq_o[(None, rb, None)])

                    # AkkOD: strict off-diagonal 16x16 blocks in K3b's blocked
                    # layout [chunk*6 + pair, h*256 + r16*16 + c16], beta-scaled;
                    # each thread's 4-col run is contiguous in the block row.
                    for rb in cutlass.range_constexpr(cute.size(taq_o, mode=[1])):
                        row = rb * 16 + st_row_in
                        beta_r = sBeta[row]
                        ri = row // 16
                        ci_blk = st_col0 // 16
                        if ci_blk < ri:
                            pair = ri * (ri - 1) // 2 + ci_blk
                            od_row = chunk_idx * 6 + pair
                            r_od = cute.make_rmem_tensor((4,), Float32)
                            for e in cutlass.range_constexpr(4):
                                r_od[e] = sSt2[(row, st_col0 + e)] * beta_r
                            od_raw = (
                                g_akkod.iterator
                                + od_row * (H * 256)
                                + Int32(h_idx) * 256
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
                    # Akkd RAW: beta-scaled strict diag blocks; the 16x16
                    # Neumann inverses run in a tiny standalone Triton kernel
                    # off this kernel's critical path.
                    dib = local_tid // 64  # diag block 0..3
                    dit = local_tid % 64
                    dr = dit % 16
                    dc0 = (dit // 16) * 4
                    arow = dib * 16 + dr
                    if arow < valid:
                        for e in cutlass.range_constexpr(4):
                            dc = dc0 + e
                            a_v = Float32(0.0)
                            if dc < dr:
                                a_v = sSt2[(arow, dib * 16 + dc)] * sBeta[arow]
                            g_akkd[(token_start + arow, dc, (h_idx, 0))] = a_v
                    self.cuda_bar.arrive_and_wait()

                it = it + 1
                has_work = it < n_iters + 1

        elif warp_id == self.LOAD_WARP_ID:
            cute.arch.setmaxregister_decrease(self.aux_regs)
            cpasync.prefetch_descriptor(atom_q)
            cpasync.prefetch_descriptor(atom_k)
            cpasync.prefetch_descriptor(atom_g)
            it = Int32(0)
            has_work = it < n_iters
            while has_work:
                w = bx + it * gdx
                chunk_idx = w // H
                h_idx = w % H
                _, _, token_start, valid = load_ragged_chunk_work(
                    cu_seqlens, chunk_offsets, chunk_idx, Int32(BT)
                )

                def off(d, ts=token_start):
                    return cute.domain_offset((ts, 0, (0, 0)), d)

                _, sSK, gSK = self._part_epi(
                    atom_k, off(desc_k)[None, None, (h_idx, 0)], (BT, BK), sK
                )
                _, sSG, gSG = self._part_epi(
                    atom_g, off(desc_g)[None, None, (h_idx, 0)], (BT, BK), sG
                )
                _, sSQ, gSQ = self._part_epi(
                    atom_q, off(desc_q)[None, None, (h_idx, 0)], (BT, BK), sQ
                )
                rh = praw_P.acquire_and_advance()
                cute.copy(
                    atom=atom_q,
                    src=gSQ[(None, 0, 0)],
                    dst=sSQ[None, rh.index],
                    tma_bar_ptr=rh.barrier,
                )
                cute.copy(
                    atom=atom_k,
                    src=gSK[(None, 0, 0)],
                    dst=sSK[None, rh.index],
                    tma_bar_ptr=rh.barrier,
                )
                gh = prawg_P.acquire_and_advance()
                cute.copy(
                    atom=atom_g,
                    src=gSG[(None, 0, 0)],
                    dst=sSG[None, gh.index],
                    tma_bar_ptr=gh.barrier,
                )
                it = it + 1
                has_work = it < n_iters

        elif warp_id == self.MMA_WARP_ID:
            cute.arch.setmaxregister_decrease(self.aux_regs)
            it = Int32(0)
            has_work = it < n_iters
            while has_work:
                acc_h = pacc_P.acquire_and_advance()
                for sc in cutlass.range_constexpr(4):
                    op_h = pop_C.wait_and_advance()
                    col = acc_h.index * self.tm_stage + sc * 16
                    aqk_c = cute.make_tensor(tp + self.tm_aqk + col, s_f16.layout)
                    akk_c = cute.make_tensor(tp + self.tm_akk + col, s_f16.layout)
                    for kp in cutlass.range(cute.size(t_qg_a, mode=[2]), unroll_full=True):
                        mma16.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            mma16,
                            aqk_c,
                            t_qg_a[None, None, kp, op_h.index],
                            t_kgk_b[None, None, kp, op_h.index],
                            aqk_c,
                        )
                    for kp in cutlass.range(cute.size(t_kgq_a, mode=[2]), unroll_full=True):
                        mma16.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            mma16,
                            akk_c,
                            t_kgq_a[None, None, kp, op_h.index],
                            t_kgk_b[None, None, kp, op_h.index],
                            akk_c,
                        )
                    op_h.release()
                acc_h.commit()
                it = it + 1
                has_work = it < n_iters

        if warp_id in self.IDLE_WARP_IDS:
            cute.arch.setmaxregister_decrease(self.aux_regs)

        tmem.relinquish_alloc_permit()
        self.tmem_free_bar.arrive_and_wait()
        tmem.free(tp)

    @cute.jit
    def _part_epi(self, atom, g_mnl, tile, s_buf):
        g_div = cute.flat_divide(g_mnl, tile)
        sg = cute.group_modes(s_buf, 0, 2)
        gg = cute.group_modes(g_div, 0, 2)
        ss, gs = cpasync.tma_partition(atom, 0, cute.make_layout(1), sg, gg)
        return atom, ss, gs

    @cute.jit
    def _gchunk(self, base_iter, token_start, h_idx, dtype, H, gstride):
        """(BT, BK) gmem tile view at (token_start, head) for vectorized copies."""
        raw = base_iter + token_start * H * self.BK + h_idx * self.BK
        ptr = cute.make_ptr(dtype, raw.toint(), cute.AddressSpace.gmem, assumed_align=16)
        return cute.make_tensor(ptr, cute.make_layout((self.BT, self.BK), stride=(gstride, 1)))

    def _launch_grid(self):
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA compilation requires a CUDA target with an SM count")
        return (sm_count, 1, 1)


@jit_cache
def _compile_intra_engine_fwd(H: int, head_dim: int, chunk_size: int):
    op = KdaIntraFwdEngine(chunk_size=chunk_size, head_dim=head_dim, num_heads=H)
    st, sn = cute.sym_int(), cute.sym_int()

    def fk(dtype, dim):
        return make_fake_compact_tensor(
            dtype, (st, H, dim), stride_order=(2, 1, 0), assumed_align=128
        )

    beta = make_fake_compact_tensor(cutlass.Float32, (st, H), stride_order=(1, 0), assumed_align=8)
    cu = make_fake_compact_tensor(cutlass.Int32, (sn,), stride_order=(0,), assumed_align=4)
    offs = make_fake_compact_tensor(cutlass.Int32, (sn,), stride_order=(0,), assumed_align=4)
    return compile_tvm_ffi(
        op,
        fk(cutlass.BFloat16, head_dim),
        fk(cutlass.BFloat16, head_dim),
        fk(cutlass.Float32, head_dim),
        beta,
        fk(cutlass.BFloat16, chunk_size),
        make_fake_compact_tensor(
            cutlass.Float32, (cute.sym_int(), H * 256), stride_order=(1, 0), assumed_align=16
        ),
        fk(cutlass.Float32, 16),
        cutlass.Float32(1.0),
        cu,
        offs,
        1,
        H,
        1,
        name=op.get_name(),
    )


def kda_intra_engine_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward intra: Aqk (bf16), AkkOD (K3b blocked), Akkd (diag inverses).

    Drop-in producer for the K4b assembly stage, replacing forloop + K3b. The
    kernel emits raw beta-scaled strict diag blocks; their 16x16 Neumann
    inverses run in a tiny standalone Triton launch here, so the returned Akkd
    is exactly what K4b consumes.
    """
    batch, tokens, heads, head_dim = q.shape
    assert batch == 1 and head_dim == 128
    aqk = torch.empty(
        (1, tokens, heads, metadata.chunk_size), device=q.device, dtype=torch.bfloat16
    )
    akkod = torch.zeros((metadata.capacity * 6, heads * 256), device=q.device, dtype=torch.float32)
    akkd = torch.empty((1, tokens, heads, 16), device=q.device, dtype=torch.float32)
    compiled = _compile_intra_engine_fwd(heads, head_dim, metadata.chunk_size)
    compiled(
        q[0].contiguous(),
        k[0].contiguous(),
        g[0].contiguous(),
        beta[0].contiguous(),
        aqk[0],
        akkod,
        akkd[0],
        float(scale),
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        tokens,
        heads,
        metadata.capacity,
    )
    return aqk, akkod, _kda_diag_neumann_inverse(akkd, metadata)
