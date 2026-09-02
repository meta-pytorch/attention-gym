# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SM100 warp-specialized persistent forward intra-chunk engine for KDA.

The engine computes BT64 Aqk, K3-compatible off-diagonal blocks, and raw 16x16 diagonal blocks over
(chunk, head) work units. CUDA warps produce strip-rebased BF16 operands in shared memory, one MMA
warp accumulates in TMEM, and a separate Triton launch performs the diagonal Neumann inverses.

The per-channel decay lives inside the key-dimension reduction, so operands are rebased at each
16-token strip. This keeps the largest factor within the FP32 exponent range at the supported
training gate bound while preserving causal, sequence-length-independent references.
"""

from enum import IntEnum

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch

# ---------------------------------------------------------------------------
# Tiny standalone Triton kernel: 16x16 unit-lower Neumann inverses of the
# engine's raw diag blocks (same log-depth factorization as the fwd forloop).
# ---------------------------------------------------------------------------
import triton
import triton.language as tl
from cutlass import cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Float32, Int32, Int64

from attn_gym._backends.cute import (
    TMA_ALIGNMENT_BYTES,
    make_fake_strided_tensor,
    tensor_supports_contiguous_dim,
)
from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import compile_tvm_ffi, requires_int64_abi
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata
from attn_gym.linear.kda.chunk_scheduler import (
    load_ragged_chunk_work as _tl_load_ragged_chunk_work,
)
from attn_gym.linear.kda.constants import is_sm100_kda_capability
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import load_ragged_chunk_work


@triton.jit
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


class WarpRole(IntEnum):
    """Warp-role boundaries in the persistent intra-chunk engine."""

    CUDA = 0
    LOAD = 8
    MMA = 9
    IDLE = 10
    END = 12


class KdaIntraFwdEngine:
    """Forward intra stage on the engine architecture.

    Two S-shaped gemms over strip-rebased gated operands:

        Aqk[i, j] = scale * sum_d qg[i, d] * kgk[j, d]   (non-strict causal)
        Akk[i, j] = beta[i] * sum_d kgq[i, d] * kgk[j, d]  (strict lower)

    Warp plan: CUDA(0-7) production + drain/store, LOAD(8) TMA, MMA(9),
    idle(10-11) padding warp-group 2 for setmaxregister alignment. The 16x16
    diagonal-block Neumann inverses (K4b's Akkd input) run in a separate tiny
    Triton kernel invoked by the Python wrapper.
    """

    CUDA_WARP_IDS = tuple(range(WarpRole.CUDA, WarpRole.LOAD))
    IDLE_WARP_IDS = tuple(range(WarpRole.IDLE, WarpRole.END))
    WARP_SZ = cute.arch.WARP_SIZE
    N_WARPS = WarpRole.END
    CTA_THREADS = N_WARPS * WARP_SZ
    assert len(CUDA_WARP_IDS) * WARP_SZ == 256
    assert N_WARPS % 4 == 0

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
        return (
            f"kda_intra_engine_fwd_vl{int(self.varlen)}_h{self.num_heads}"
            f"_k{self.BK}_bt{self.BT}_i64{int(self.use_int64_offsets)}"
        )

    @cute.jit
    def upcast(self, value):
        """Promote an address operand before its first overflowing multiply."""
        return cutlass.Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    @cute.jit
    def _decode_chunk(self, chunk_idx, T, cu_seqlens, chunk_offsets):
        """Return the physical token start and valid rows for one chunk."""
        if cutlass.const_expr(self.varlen):
            _, _, token_start, valid = load_ragged_chunk_work(
                cu_seqlens, chunk_offsets, chunk_idx, Int32(self.BT)
            )
        else:
            token_start = chunk_idx * Int32(self.BT)
            valid = Int32(self.BT)
        return token_start, valid

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
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_type,
            self.cta_group,
            self.s_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # Per-16-column strip MMA: each strip's operands are rebased at that
        # strip's own row-start gate, bounding every exp2 to a 16-token range
        # (see the strip-rebase paragraph in the module docstring).
        strip_tile = (self.mma_rows, 16, BK)
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
            self.io_type,
            utils.LayoutEnum.ROW_MAJOR,
            (self.mma_rows, BK),
            self.op_depth,
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
            sAqkSt: cute.struct.Align[
                cute.struct.MemRange[Float32, self.mma_rows * self.BT], self.align
            ]
            sAkkSt: cute.struct.Align[
                cute.struct.MemRange[Float32, self.mma_rows * self.BT], self.align
            ]

        self.shared_type = SharedF
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
        ).launch(
            grid=self._launch_grid(),
            block=[self.CTA_THREADS, 1, 1],
            cluster=self.cluster,
            stream=stream,
        )

    @cute.jit
    def run_cuda(
        self,
        tid,
        n_cuda,
        n_iters,
        bx,
        gdx,
        T,
        H,
        scale,
        praw_C,
        prawg_C,
        pop_P,
        pacc_C,
        sQ,
        sK,
        sG,
        sBeta,
        sQgSt,
        sKgqSt,
        sKgkSt,
        sAqkSt,
        sAkkSt,
        t_aqk_acc,
        t_akk_acc,
        g_beta,
        g_aqk,
        g_akkod,
        g_akkd,
        cu_seqlens,
        chunk_offsets,
    ):
        """Produce gated operands and drain Aqk/Akk accumulators."""
        BT = self.BT
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
        sreg_shape = sl_s.partition_D(cute.make_identity_tensor((self.mma_rows, BT))).shape
        r2s_atom = sm100_utils.get_smem_store_op(
            utils.LayoutEnum.ROW_MAJOR, Float32, self.acc_type, tc_s
        )
        tc_r2s = cute.make_tiled_copy_D(r2s_atom, tc_s)
        thr_r2s = tc_r2s.get_slice(local_tid % 128)

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
                chunk_idx = (bx + it * gdx) // H
                _token_start, valid = self._decode_chunk(chunk_idx, T, cu_seqlens, chunk_offsets)
                raw_h = praw_C.wait_and_advance()
                g_h = prawg_C.wait_and_advance()
                tsg_p = ep_g.partition_S(sG[(None, None, g_h.index)])
                tsk_p = ep_b.partition_S(sK[(None, None, raw_h.index)])
                tsq_p = ep_b.partition_S(sQ[(None, None, raw_h.index)])
                for sc in cutlass.range_constexpr(self.subchunks):
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
                token_start, valid = self._decode_chunk(chunk_idx, T, cu_seqlens, chunk_offsets)
                token_offset = self.upcast(token_start)
                head_offset = self.upcast(h_idx)
                if local_tid < BT:
                    b_val = Float32(0.0)
                    if local_tid < valid:
                        b_val = g_beta[(token_offset + local_tid, (head_offset, 0))]
                    sBeta[local_tid] = b_val

                acc_h = pacc_C.wait_and_advance()
                s_reg = cute.make_rmem_tensor(sreg_shape, self.acc_type)
                if local_tid < 128:
                    cute.copy(tc_s, p_t_aqk[(None, None, None, acc_h.index)], s_reg)
                    cute.arch.fence_view_async_tmem_load()
                    cute.copy(tc_r2s, tc_r2s.retile(s_reg), thr_r2s.partition_D(sAqkSt))
                    cute.copy(tc_s, p_t_akk[(None, None, None, acc_h.index)], s_reg)
                    cute.arch.fence_view_async_tmem_load()
                    cute.copy(tc_r2s, tc_r2s.retile(s_reg), thr_r2s.partition_D(sAkkSt))
                acc_h.release()
                self.cuda_bar.arrive_and_wait()

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
                self.cuda_bar.arrive_and_wait()

            it = it + 1
            has_work = it < n_iters + 1

    @cute.jit
    def run_load(
        self,
        n_iters,
        bx,
        gdx,
        T,
        H,
        atom_q,
        desc_q,
        atom_k,
        desc_k,
        atom_g,
        desc_g,
        sQ,
        sK,
        sG,
        praw_P,
        prawg_P,
        cu_seqlens,
        chunk_offsets,
    ):
        """Stage raw Q, K, and gate tiles with TMA."""
        BT, BK = self.BT, self.BK
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
            token_start, _valid = self._decode_chunk(chunk_idx, T, cu_seqlens, chunk_offsets)
            token_offset = self.upcast(token_start)
            head_offset = self.upcast(h_idx)
            sSK, gSK = self._part_epi(
                atom_k,
                cute.domain_offset((token_offset, 0, (0, 0)), desc_k)[
                    None, None, (head_offset, 0)
                ],
                (BT, BK),
                sK,
            )
            sSG, gSG = self._part_epi(
                atom_g,
                cute.domain_offset((token_offset, 0, (0, 0)), desc_g)[
                    None, None, (head_offset, 0)
                ],
                (BT, BK),
                sG,
            )
            sSQ, gSQ = self._part_epi(
                atom_q,
                cute.domain_offset((token_offset, 0, (0, 0)), desc_q)[
                    None, None, (head_offset, 0)
                ],
                (BT, BK),
                sQ,
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

    @cute.jit
    def run_mma(
        self,
        n_iters,
        mma16,
        t_qg_a,
        t_kgq_a,
        t_kgk_b,
        s_f16,
        tp,
        pacc_P,
        pop_C,
    ):
        """Consume gated operands and produce Aqk/Akk accumulators."""
        cute.arch.setmaxregister_decrease(self.aux_regs)
        it = Int32(0)
        has_work = it < n_iters
        while has_work:
            acc_h = pacc_P.acquire_and_advance()
            for sc in cutlass.range_constexpr(self.subchunks):
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
        BT = self.BT
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
            sm.tmem_buf, barrier_for_retrieve=tmem_bar, allocator_warp_id=WarpRole.LOAD
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
        st_lay = cute.make_layout((self.mma_rows, BT), stride=(BT, 1))
        sAqkSt = sm.sAqkSt.get_tensor(st_lay)
        sAkkSt = sm.sAkkSt.get_tensor(st_lay)

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
        if cutlass.const_expr(self.varlen):
            num_sequences = Int32(cute.size(chunk_offsets)) - 1
            active_chunks = Int32(chunk_offsets[num_sequences])
        else:
            active_chunks = T // Int32(BT)
        total_work = active_chunks * H
        n_iters = (total_work - bx + gdx - 1) // gdx

        if warp_id in self.CUDA_WARP_IDS:
            self.run_cuda(
                tid=tid,
                n_cuda=n_cuda,
                n_iters=n_iters,
                bx=bx,
                gdx=gdx,
                T=T,
                H=H,
                scale=scale,
                praw_C=praw_C,
                prawg_C=prawg_C,
                pop_P=pop_P,
                pacc_C=pacc_C,
                sQ=sQ,
                sK=sK,
                sG=sG,
                sBeta=sBeta,
                sQgSt=sQgSt,
                sKgqSt=sKgqSt,
                sKgkSt=sKgkSt,
                sAqkSt=sAqkSt,
                sAkkSt=sAkkSt,
                t_aqk_acc=t_aqk_acc,
                t_akk_acc=t_akk_acc,
                g_beta=g_beta,
                g_aqk=g_aqk,
                g_akkod=g_akkod,
                g_akkd=g_akkd,
                cu_seqlens=cu_seqlens,
                chunk_offsets=chunk_offsets,
            )
        elif warp_id == WarpRole.LOAD:
            self.run_load(
                n_iters=n_iters,
                bx=bx,
                gdx=gdx,
                T=T,
                H=H,
                atom_q=atom_q,
                desc_q=desc_q,
                atom_k=atom_k,
                desc_k=desc_k,
                atom_g=atom_g,
                desc_g=desc_g,
                sQ=sQ,
                sK=sK,
                sG=sG,
                praw_P=praw_P,
                prawg_P=prawg_P,
                cu_seqlens=cu_seqlens,
                chunk_offsets=chunk_offsets,
            )
        elif warp_id == WarpRole.MMA:
            self.run_mma(
                n_iters=n_iters,
                mma16=mma16,
                t_qg_a=t_qg_a,
                t_kgq_a=t_kgq_a,
                t_kgk_b=t_kgk_b,
                s_f16=s_f16,
                tp=tp,
                pacc_P=pacc_P,
                pop_C=pop_C,
            )

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
        return cpasync.tma_partition(atom, 0, cute.make_layout(1), sg, gg)

    def _launch_grid(self):
        sm_count = get_compile_target().sm_count
        if sm_count is None:
            raise RuntimeError("KDA compilation requires a CUDA target with an SM count")
        return (sm_count, 1, 1)


@jit_cache
def _compile_intra_engine_fwd(
    H: int,
    head_dim: int,
    varlen: bool,
    use_int64_offsets: bool,
):
    target = get_compile_target()
    if target.device_type != "cuda" or not is_sm100_kda_capability(target.effective_capability):
        raise ValueError(f"KDA intra engine requires an SM100 or SM103 target; got {target}")
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


def normalize_compact_tensor(tensor: torch.Tensor, alignment_bytes: int) -> torch.Tensor:
    """Copy compact tensors whose storage does not satisfy the compiled alignment."""
    if tensor.is_contiguous() and tensor_supports_contiguous_dim(
        tensor, alignment_bytes=alignment_bytes
    ):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def kda_intra_engine_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce the BT64 Aqk and inverse factors."""
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
    g_flat = normalize_compact_tensor(g[0], 128)
    beta_flat = normalize_compact_tensor(beta[0], 8)
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
