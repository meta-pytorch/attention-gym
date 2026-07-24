# ruff: noqa
#
# Vendored from NVIDIA/cudnn-frontend's DeepSeek Sparse Attention SM100 backward.
# Upstream package: nvidia-cudnn-frontend 1.25.0
# Original source copyright (c) 2026, Jerry Chen.
# Package copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cuda.bindings.driver as cuda
import math
from typing import Tuple, Type, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, BFloat16, Int64
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils as utils
from cutlass._mlir.dialects import arith, llvm, nvvm, vector


class FlashAttentionDSABackwardSm100:
    arch = 100

    def __init__(self, head_dim: int, head_dim_v: int, block_tile: int):
        self.head_dim = head_dim
        self.head_dim_v = head_dim_v
        self.same_hdim_kv = head_dim == head_dim_v
        self.block_tile = block_tile
        self.QK_mma_tiler = (block_tile, block_tile, head_dim)
        # head_dim_main: 128-aligned portion for the main 4 sub-tiles
        head_dim_main = (head_dim // 128) * 128
        self.head_dim_main = head_dim_main
        self.dOP_mma_tiler = (128, block_tile, block_tile)
        self.dOP_cta_tiler = (head_dim_v, block_tile, block_tile)
        self.dOV_mma_tiler = (block_tile, block_tile, head_dim_v)
        self.KdS_mma_tiler = (128, block_tile, block_tile)
        self.KdS_cta_tiler = (head_dim_main, block_tile, block_tile)
        self.QdS_mma_tiler = (128, block_tile, block_tile)
        self.QdS_cta_tiler = (head_dim_main, block_tile, block_tile)
        self.cluster_shape_mn = (1, 1)

        self.element_dtype = BFloat16
        self.acc_dtype = Float32

        # =============== Sum OdO ================
        self.sum_OdO_max_threads_per_block = 128
        self.sum_OdO_block_q = 16
        self.sum_OdO_num_threads_d = 8
        self.sum_OdO_num_threads_q = (
            self.sum_OdO_max_threads_per_block // self.sum_OdO_num_threads_d
        )
        self.sum_OdO_elem_per_load = 2
        self.dSink_block_q = 256
        self.dSink_num_threads = 32

        # =============== Bwd ====================
        self.load_KV_warp_id = (0, 1, 2, 3)
        self.compute_warp_id = (4, 5, 6, 7)
        self.reduce_warp_id = (8, 9, 10, 11, 12, 13, 14, 15)
        self.mma_warp_id = 16
        self.load_warp_id = 17
        self.empty_warp_id = 18

        self.num_load_KV_warps = 4
        self.num_compute_warps = 4
        self.num_reduce_warps = 8

        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (
            self.num_load_KV_warps + self.num_compute_warps + self.num_reduce_warps + 4
        )

        # self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.num_tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp
            * (self.num_compute_warps + self.num_reduce_warps + 1),
        )
        self.compute_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.num_compute_warps * self.threads_per_warp,
        )
        self.load_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )
        self.load_KV_sync_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.num_load_KV_warps * self.threads_per_warp,
        )
        self.reduce_sync_barrier = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=self.num_reduce_warps * self.threads_per_warp,
        )
        self.t2r_dKV01_done_barrier = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=(self.num_reduce_warps + 1) * self.threads_per_warp,
        )
        self.t2r_dKV4_done_barrier = pipeline.NamedBarrier(
            barrier_id=8,
            num_threads=(self.num_reduce_warps + 1) * self.threads_per_warp,
        )

        self.tmem_S_offset = 0
        self.tmem_dP_offset = 0
        self.tmem_dKV0_offset = self.tmem_dP_offset + block_tile
        self.tmem_dKV1_offset = self.tmem_dKV0_offset + block_tile
        self.tmem_dKV2_offset = self.tmem_dKV0_offset
        self.tmem_dKV3_offset = self.tmem_dKV1_offset
        self.tmem_dQ0_offset = self.tmem_dKV3_offset + block_tile
        self.tmem_dQ1_offset = self.tmem_dQ0_offset + block_tile
        self.tmem_dQ2_offset = self.tmem_dQ1_offset + block_tile
        self.tmem_dQ3_offset = self.tmem_dQ2_offset + block_tile
        self.tmem_dQ4_offset = self.tmem_dQ3_offset + block_tile
        self.tmem_dKV4_offset = self.tmem_dKV0_offset

        self.dQ4_mma_tiler = (64, block_tile, block_tile)
        self.dKV4_mma_tiler = (64, block_tile, block_tile)

        self.num_regs_load_KV = 48
        self.num_regs_compute = 128
        self.num_regs_reduce = 128
        self.num_regs_mma = 40
        self.num_regs_empty = 40
        self.num_regs_load = 40

        self.buffer_align_bytes = 1024
        self.non_tma_align_bytes = 128

    def _setup_attributes(self):
        self.load_mma_QdO_stage = 1
        self.load_mma_K_stage = 1
        # self.load_mma_dO_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1
        self.mma_compute_S_stage = 1
        self.mma_compute_dP_stage = 1
        self.mma_compute_dQ_stage = 1
        self.compute_mma_P_stage = 1
        self.compute_mma_dS_stage = 1
        self.mma_reduce_dKV_stage = 2 if not self.same_hdim_kv else 1
        self.reduce_store_dKV_stage = 1
        self.compute_tmastore_dQ_stage = 1

    @staticmethod
    def _get_workspace_size_LSE_OdO(
        q: int, d: int, h: int, b: int, acc_dtype: Type[cutlass.Numeric]
    ):
        # q is total seqlen, b=1
        d = (d + 7) // 8 * 8  # round up to 8
        q = (q + 7) // 8 * 8  # round up to 8
        h = (h + 63) // 64 * 64  # align the two FP32 workspace regions
        workspace_bytes = 0
        # OdO vector
        workspace_bytes += acc_dtype.width // 8
        # scaled LSE vector
        workspace_bytes += acc_dtype.width // 8
        # Avoid single workspace bytes exceeds 32bit range
        return (b, h, q, workspace_bytes)

    @staticmethod
    def _get_workspace_size_dKV(k: int, d: int, b: int, acc_dtype: Type[cutlass.Numeric]):
        d = (d + 7) // 8 * 8  # round up to 8
        k = (k + 7) // 8 * 8  # round up to 8
        # FP32 versions of dKV
        workspace_bytes = d * acc_dtype.width // 8
        return (b, 1, k, workspace_bytes)

    def get_workspace_tensor(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        total_seqlen_Q: Int32,
        total_seqlen_KV: Int32,
        acc_dtype: Type[cutlass.Numeric],
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor]:
        # problem_shape contains the max seqlen of Q and K
        max_Q, max_K, D, HB = (
            problem_shape[0],
            problem_shape[1],
            problem_shape[2],
            problem_shape[3],
        )
        H, B = cute.size(problem_shape[3][0]), cute.size(problem_shape[3][1])

        D = cute.round_up(D, 8)
        total_seqlen_Q = cute.round_up(total_seqlen_Q, 8)
        workspace_heads = cute.round_up(H, 64)

        acc_bytes = acc_dtype.width // 8
        workspace_head_stride = cute.assume(workspace_heads, divby=64)
        sum_OdO_bytes = cute.assume(
            workspace_head_stride * total_seqlen_Q * acc_bytes,
            divby=acc_bytes * 64,
        )

        sum_OdO_iter = workspace_LSE_OdO.iterator
        scaled_lse_iter = sum_OdO_iter + sum_OdO_bytes
        dKV_acc_iter = workspace_dKV.iterator

        sum_OdO_iter = cute.recast_ptr(sum_OdO_iter, dtype=self.acc_dtype)
        scaled_lse_iter = cute.recast_ptr(scaled_lse_iter, dtype=self.acc_dtype)
        dKV_acc_iter = cute.recast_ptr(dKV_acc_iter, dtype=self.acc_dtype)

        sum_OdO = cute.make_tensor(
            sum_OdO_iter,
            cute.make_layout(
                (H, (total_seqlen_Q, 1)),
                stride=(1, (workspace_head_stride, 0)),
            ),
        )
        scaled_lse = cute.make_tensor(
            scaled_lse_iter,
            cute.make_layout(
                (H, (total_seqlen_Q, 1)),
                stride=(1, (workspace_head_stride, 0)),
            ),
        )
        dKV_acc = cute.make_tensor(
            dKV_acc_iter,
            cute.make_layout((total_seqlen_KV, D, (1, 1)), stride=(D, 1, (0, 0))),
        )

        return sum_OdO, scaled_lse, dKV_acc

    @staticmethod
    def _compute_sum_OdO_grid(
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        block_q: int,
    ) -> Tuple[int, int, int]:
        grid = (
            cute.ceil_div(cute.size(problem_shape[0]), block_q),
            cute.size(problem_shape[3][0]),  # H
            cute.size(problem_shape[3][1]),  # B
        )
        return grid

    @cute.jit
    def __call__(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        mQ: cute.Tensor,
        mKV: cute.Tensor,
        mOut: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mdSink: cute.Tensor,
        workspace_LSE_OdO: cute.Tensor,
        workspace_dKV: cute.Tensor,
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        """
        Forward pass for DeepSeek Sparse Attention.
        """

        # [M, H, D] -> [H, D, (M, 1)]
        mQ = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (mQ.shape[1], mQ.shape[2], (mQ.shape[0], 1)),
                stride=(mQ.stride[1], mQ.stride[2], (mQ.stride[0], 0)),
            ),
        )

        # [N, D] -> [N, D, (1, 1)]
        mKV = cute.make_tensor(
            mKV.iterator,
            cute.make_layout(
                (mKV.shape[0], mKV.shape[1], (1, 1)), stride=(mKV.stride[0], mKV.stride[1], (0, 0))
            ),
        )

        # [M, H, Dv] -> [H, Dv, (M, 1)]
        mOut = cute.make_tensor(
            mOut.iterator,
            cute.make_layout(
                (mOut.shape[1], mOut.shape[2], (mOut.shape[0], 1)),
                stride=(mOut.stride[1], mOut.stride[2], (mOut.stride[0], 0)),
            ),
        )

        # [M, H, Dv] -> [H, Dv, (M, 1)]
        mdO = cute.make_tensor(
            mdO.iterator,
            cute.make_layout(
                (mdO.shape[1], mdO.shape[2], (mdO.shape[0], 1)),
                stride=(mdO.stride[1], mdO.stride[2], (mdO.stride[0], 0)),
            ),
        )
        # [M, H, D] -> [D, H, (M, 1)]
        mdQ = cute.make_tensor(
            mdQ.iterator,
            cute.make_layout(
                (mdQ.shape[2], mdQ.shape[1], (mdQ.shape[0], 1)),
                stride=(mdQ.stride[2], mdQ.stride[1], (mdQ.stride[0], 0)),
            ),
        )
        # [N, D] -> [D, N, (1, 1)]
        mdKV = cute.make_tensor(
            mdKV.iterator,
            cute.make_layout(
                (mdKV.shape[1], mdKV.shape[0], (1, 1)),
                stride=(mdKV.stride[1], mdKV.stride[0], (0, 0)),
            ),
        )

        # [M, H] -> [H, (M, 1)]
        mLSE = cute.make_tensor(
            mLSE.iterator,
            cute.make_layout(
                (mLSE.shape[1], (mLSE.shape[0], 1)), stride=(mLSE.stride[1], (mLSE.stride[0], 0))
            ),
        )

        # [H] -> [H, (1, 1)]
        mdSink = cute.make_tensor(
            mdSink.iterator, cute.make_layout((mdSink.shape[0], (1, 1)), stride=(1, (0, 0)))
        )
        mAttnSink = cute.make_tensor(mAttnSink.iterator, mdSink.layout)

        # [M, TopK] -> [TopK, (M, 1)]
        mTopkIdxs = cute.make_tensor(
            mTopkIdxs.iterator,
            cute.make_layout(
                (mTopkIdxs.shape[1], (mTopkIdxs.shape[0], 1)),
                stride=(mTopkIdxs.stride[1], (mTopkIdxs.stride[0], 0)),
            ),
        )
        # [M] -> [M, (1, 1)] when provided; None means non-compact (use full topk, -1 entries in topk_idxs)
        if cutlass.const_expr(mTopkLength is not None):
            mTopkLength = cute.make_tensor(
                mTopkLength.iterator,
                cute.make_layout(
                    (mTopkLength.shape[0], (1, 1)), stride=(mTopkLength.stride[0], (0, 0))
                ),
            )

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.ONE

        # S = Q @ KV
        QK_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.QK_mma_tiler[:2],
        )

        # dP = dO @ KV
        dOV_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.dOV_mma_tiler[:2],
        )

        # dKV = dO^T @ P
        dOP_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.dOP_mma_tiler[:2],
        )
        # dKV = Q^T @ dS
        QdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.QdS_mma_tiler[:2],
        )
        # dQ = KV @ dS^T
        KdS_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.element_dtype,
            self.element_dtype,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            self.acc_dtype,
            cta_group,
            self.KdS_mma_tiler[:2],
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            # dKV4: Q^T[512:575] @ dS -> (64, 64) output
            dKV4_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.element_dtype,
                self.element_dtype,
                OperandMajorMode.MN,
                OperandMajorMode.K,
                self.acc_dtype,
                cta_group,
                self.dKV4_mma_tiler[:2],
            )
            # dQ4: K[512:575] @ dS^T -> (64, 64) output
            dQ4_tiled_mma = sm100_utils.make_trivial_tiled_mma(
                self.element_dtype,
                self.element_dtype,
                OperandMajorMode.MN,
                OperandMajorMode.MN,
                self.acc_dtype,
                cta_group,
                self.dQ4_mma_tiler[:2],
            )
        else:
            dKV4_tiled_mma = None
            dQ4_tiled_mma = None

        self.cluster_layout_vmnk = cute.make_layout(((1), (1, 1, 1)), stride=((0), (0, 0, 0)))

        Q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            QK_tiled_mma, self.QK_mma_tiler, self.element_dtype, self.load_mma_QdO_stage
        )
        K_smem_layout_staged = sm100_utils.make_smem_layout_b(
            QK_tiled_mma, self.QK_mma_tiler, self.element_dtype, self.load_mma_K_stage
        )
        dO_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dOV_tiled_mma, self.dOV_mma_tiler, self.element_dtype, self.load_mma_QdO_stage
        )
        if cutlass.const_expr(not self.same_hdim_kv):
            V_smem_layout_staged = sm100_utils.make_smem_layout_b(
                dOV_tiled_mma, self.dOV_mma_tiler, self.element_dtype, self.load_mma_K_stage
            )
        else:
            V_smem_layout_staged = K_smem_layout_staged

        dOT_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dOP_tiled_mma, self.dOP_cta_tiler, self.element_dtype, self.load_mma_QdO_stage
        )
        P_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dOP_tiled_mma, self.dOP_mma_tiler, self.element_dtype, self.compute_mma_P_stage
        )
        P_smem_layout_store_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.COL_MAJOR,
            self.QK_mma_tiler[:2],
            self.load_mma_K_stage,
        )
        K_smem_layout_staged_2 = sm100_utils.make_smem_layout_a(
            KdS_tiled_mma, self.KdS_cta_tiler, self.element_dtype, self.load_mma_K_stage
        )
        if cutlass.const_expr(not self.same_hdim_kv):
            # Tail view: partition sK with 64-wide blocks, giving head_dim/64 sub-tiles
            K_tail_smem_layout_staged = sm100_utils.make_smem_layout_a(
                dQ4_tiled_mma,
                (self.head_dim, self.block_tile, self.block_tile),
                self.element_dtype,
                self.load_mma_K_stage,
            )
        else:
            K_tail_smem_layout_staged = None
        dST_smem_layout_staged = sm100_utils.make_smem_layout_b(
            KdS_tiled_mma, self.KdS_mma_tiler, self.element_dtype, self.compute_mma_dS_stage
        )
        QT_smem_layout_staged = sm100_utils.make_smem_layout_a(
            QdS_tiled_mma, self.QdS_cta_tiler, self.element_dtype, self.load_mma_QdO_stage
        )
        if cutlass.const_expr(not self.same_hdim_kv):
            # Tail view: partition sQ with 64-wide blocks
            QT_tail_smem_layout_staged = sm100_utils.make_smem_layout_a(
                dKV4_tiled_mma,
                (self.head_dim, self.block_tile, self.block_tile),
                self.element_dtype,
                self.load_mma_QdO_stage,
            )
        else:
            QT_tail_smem_layout_staged = None
        dS_smem_layout_staged = sm100_utils.make_smem_layout_b(
            QdS_tiled_mma, self.QdS_mma_tiler, self.element_dtype, self.compute_mma_dS_stage
        )
        dS_smem_layout_store_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.COL_MAJOR,
            self.dOV_mma_tiler[:2],
            self.load_mma_K_stage,
        )

        dQ_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.element_dtype,
            utils.LayoutEnum.from_tensor(mdQ),
            (self.KdS_mma_tiler[0], self.KdS_mma_tiler[1]),
            self.mma_compute_dQ_stage,
        )

        dKV_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.acc_dtype,
            utils.LayoutEnum.from_tensor(mdKV),
            (self.dOP_mma_tiler[0], self.dOP_mma_tiler[1] // 2),
            self.mma_reduce_dKV_stage,
        )

        LSE_smem_layout = cute.make_layout((self.QK_mma_tiler[0], self.load_compute_LSE_stage))
        sum_OdO_smem_layout = cute.make_layout(
            (self.QK_mma_tiler[0], self.load_compute_sum_OdO_stage)
        )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        Q_smem_layout = cute.select(Q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            Q_smem_layout,
            self.QK_mma_tiler,
            QK_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        dO_smem_layout = cute.select(dO_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mdO,
            dO_smem_layout,
            self.dOV_mma_tiler,
            dOV_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        dQ_smem_layout = cute.select(dQ_smem_layout_staged, mode=[0, 1])
        tma_atom_dQ, tma_tensor_dQ = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_store_op,
            mdQ,
            dQ_smem_layout,
            (self.KdS_mma_tiler[0], self.KdS_mma_tiler[1]),
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            dQ4_smem_layout_staged = sm100_utils.make_smem_layout_epi(
                self.element_dtype,
                utils.LayoutEnum.from_tensor(mdQ),
                (self.dQ4_mma_tiler[0], self.dQ4_mma_tiler[1]),
                self.mma_compute_dQ_stage,
            )
            dQ4_smem_layout = cute.select(dQ4_smem_layout_staged, mode=[0, 1])
            tma_atom_dQ_64, tma_tensor_dQ_64 = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                mdQ,
                dQ4_smem_layout,
                (self.dQ4_mma_tiler[0], self.dQ4_mma_tiler[1]),
            )
        else:
            dQ4_smem_layout_staged = None
            tma_atom_dQ_64 = None
            tma_tensor_dQ_64 = None

        self.tma_copy_Q_bytes = cute.size_in_bytes(self.element_dtype, Q_smem_layout)
        self.tma_copy_dO_bytes = cute.size_in_bytes(self.element_dtype, dO_smem_layout)
        self.tma_copy_QdO_bytes = self.tma_copy_Q_bytes + self.tma_copy_dO_bytes

        # Compute base smem size (without topk) to determine topk capacity
        def _align_up(x, a):
            return (x + a - 1) // a * a

        _elem_bytes = self.element_dtype.width // 8
        _acc_bytes = self.acc_dtype.width // 8
        _tma_align = self.buffer_align_bytes
        _non_tma_align = self.non_tma_align_bytes

        _offset = 0
        # 10 mbar_ptr fields: MemRange[Int64, stage * 2], all stages = 1, so 2 Int64s = 16 bytes each
        _offset += 10 * 2 * 8  # 160 bytes
        # tmem_holding_buf: Int32
        _offset += 4
        # TMA buffers (1024 alignment): sQ, sK, sdO
        _offset = _align_up(_offset, _tma_align)
        _offset += int(cute.cosize(Q_smem_layout_staged)) * _elem_bytes
        _offset = _align_up(_offset, _tma_align)
        _offset += int(cute.cosize(K_smem_layout_staged)) * _elem_bytes
        _offset = _align_up(_offset, _tma_align)
        _offset += int(cute.cosize(dO_smem_layout_staged)) * _elem_bytes
        # Non-TMA buffers (128 alignment): sP, sdS, sLSE, sSum_OdO
        _offset = _align_up(_offset, _non_tma_align)
        _offset += int(cute.cosize(P_smem_layout_staged)) * _elem_bytes
        _offset = _align_up(_offset, _non_tma_align)
        _offset += int(cute.cosize(dS_smem_layout_staged)) * _elem_bytes
        _offset = _align_up(_offset, _non_tma_align)
        _offset += int(cute.cosize(LSE_smem_layout)) * _acc_bytes
        _offset = _align_up(_offset, _non_tma_align)
        _offset += int(cute.cosize(sum_OdO_smem_layout)) * _acc_bytes
        # Remaining space for topk indices (Int32 = 4 bytes each)
        _max_smem_bytes = 227 * 1024
        self.smem_topk_capacity = (_max_smem_bytes - _offset) // 4

        @cute.struct
        class SharedStorage:
            load_mma_QdO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_QdO_stage * 2]
            load_mma_K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_mma_K_stage * 2]
            load_compute_LSE_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_LSE_stage * 2
            ]
            load_compute_sum_OdO_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.load_compute_sum_OdO_stage * 2
            ]
            mma_compute_S_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_S_stage * 2
            ]
            mma_compute_dP_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dP_stage * 2
            ]
            mma_compute_dQ_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_compute_dQ_stage * 2
            ]
            compute_mma_P_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_P_stage * 2
            ]
            compute_mma_dS_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.compute_mma_dS_stage * 2
            ]
            mma_reduce_dKV_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.mma_reduce_dKV_stage * 2
            ]
            tmem_holding_buf: cutlass.Int32
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(Q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(K_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sdO: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(dO_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sP: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(P_smem_layout_staged)],
                self.non_tma_align_bytes,
            ]
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.element_dtype, cute.cosize(dS_smem_layout_staged)],
                self.non_tma_align_bytes,
            ]
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(LSE_smem_layout)],
                self.non_tma_align_bytes,
            ]
            sSum_OdO: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, cute.cosize(sum_OdO_smem_layout)],
                self.non_tma_align_bytes,
            ]
            sTopkIdxs: cute.struct.MemRange[cutlass.Int32, self.smem_topk_capacity]

        assert SharedStorage.size_in_bytes() <= _max_smem_bytes, (
            f"SharedStorage ({SharedStorage.size_in_bytes()} bytes) exceeds {_max_smem_bytes} bytes (227KB), "
            f"smem_topk_capacity={self.smem_topk_capacity}"
        )
        self.shared_storage = SharedStorage

        sum_OdO, scaled_LSE, mdKV_acc = self.get_workspace_tensor(
            problem_shape,
            workspace_LSE_OdO,
            workspace_dKV,
            mQ.shape[2][0],
            mKV.shape[0],
            self.acc_dtype,
        )
        mdKV_acc = cute.make_tensor(mdKV_acc.iterator, mdKV.layout)

        # ============ Sum OdO ============
        sum_OdO_scale = Float32(-1.0)
        LSE_scale = Float32(-math.log2(math.e))

        sum_OdO_grid = self._compute_sum_OdO_grid(problem_shape, self.sum_OdO_block_q)

        self.sum_OdO(
            mOut,
            mdO,
            sum_OdO,
            mLSE,
            mAttnSink,
            scaled_LSE,
            sum_OdO_scale,
            LSE_scale,
            problem_shape,
        ).launch(
            grid=sum_OdO_grid,
            block=[self.sum_OdO_num_threads_d, self.sum_OdO_num_threads_q, 1],
            cluster=[1, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

        num_head_blocks = cute.ceil_div(problem_shape[3][0], self.block_tile)
        bwd_grid = (problem_shape[0], num_head_blocks, problem_shape[3][1])
        self.bwd(
            problem_shape,
            QK_tiled_mma,
            dOV_tiled_mma,
            dOP_tiled_mma,
            QdS_tiled_mma,
            KdS_tiled_mma,
            dKV4_tiled_mma,
            dQ4_tiled_mma,
            tma_atom_Q,
            tma_tensor_Q,
            tma_atom_dO,
            tma_tensor_dO,
            tma_atom_dQ,
            tma_tensor_dQ,
            tma_atom_dQ_64,
            tma_tensor_dQ_64,
            mKV,
            mdQ,
            mdKV_acc,
            mdSink,
            mAttnSink,
            mTopkIdxs,
            mTopkLength,
            scaled_LSE,
            sum_OdO,
            softmax_scale,
            Q_smem_layout_staged,
            K_smem_layout_staged,
            dO_smem_layout_staged,
            V_smem_layout_staged,
            dOT_smem_layout_staged,
            P_smem_layout_staged,
            P_smem_layout_store_staged,
            K_smem_layout_staged_2,
            K_tail_smem_layout_staged,
            dST_smem_layout_staged,
            QT_smem_layout_staged,
            QT_tail_smem_layout_staged,
            dS_smem_layout_staged,
            dS_smem_layout_store_staged,
            dKV_smem_layout_staged,
            dQ_smem_layout_staged,
            dQ4_smem_layout_staged,
            LSE_smem_layout,
            sum_OdO_smem_layout,
        ).launch(
            grid=bwd_grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[1, 1, 1],
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

        self.block_seq = 8
        self.num_threads_D_convert = 32
        self.num_threads_seq = 256 // self.num_threads_D_convert
        self.convert_elem_per_load = 4

        convert_grid_x = (mKV.shape[0] + self.block_seq - 1) // self.block_seq
        convert_grid = [
            1,
            convert_grid_x,
            1,
        ]
        convert_block = [self.num_threads_D_convert, self.num_threads_seq, 1]
        self.convert(
            mdKV_acc,
            mdKV,
            mKV.shape[0],
        ).launch(
            grid=convert_grid,
            block=convert_block,
            stream=stream,
        )

        if cutlass.const_expr(self.same_hdim_kv):
            dSink_grid = (
                cute.ceil_div(problem_shape[0], self.dSink_block_q),
                problem_shape[3][0],
                problem_shape[3][1],
            )
            self.sum_dSink(
                sum_OdO,
                scaled_LSE,
                mAttnSink,
                mdSink,
                problem_shape,
            ).launch(
                grid=dSink_grid,
                block=[self.dSink_num_threads, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.kernel
    def convert(
        self,
        mdKV_acc: cute.Tensor,
        mdKV: cute.Tensor,  # (D, N, (1, B))
        seqlen: Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        (
            _,
            seq_block_idx,
            batch_idx,
        ) = cute.arch.block_idx()

        seq_id = self.block_seq * seq_block_idx + tidy

        if seq_id < seqlen:
            cur_mdKV_acc_row = mdKV_acc[None, seq_id, (0, batch_idx)]
            cur_mdKV_row = mdKV[None, seq_id, (0, batch_idx)]
            tile_mdKV_acc_row = cute.flat_divide(cur_mdKV_acc_row, (64,))  # (64, D/64)
            tile_mdKV_acc_row = cute.flat_divide(tile_mdKV_acc_row, (32,))  # (32, 2, D/64)
            # Tiles from 128-wide store_dKV: layout = groups of 4 per lane
            num_128_tiles = self.head_dim_main // 64
            for i in cutlass.range(num_128_tiles, unroll_full=True):
                for j in cutlass.range(2, unroll_full=True):
                    cur_tile_mdKV_acc = tile_mdKV_acc_row[tidx, j, i]
                    dim_idx = tidx // 4 + tidx % 4 * 8 + j * 32 + i * 64
                    cur_mdKV_row[dim_idx] = self.element_dtype(cur_tile_mdKV_acc)
            # Last tile from 64-wide store_dKV_64: layout = groups of 2 per lane
            # Layout F (M=64, 16dp): dp_idx//4=k → warp=k//8, lane=k%8
            # pos 2k holds M=warp*16+lane, pos 2k+1 holds M=warp*16+lane+8
            # Unscramble: p=tidx+j*32, k=p//2 → dim = base + (k//8)*16 + k%8 + (p%2)*8
            if cutlass.const_expr(not self.same_hdim_kv):
                for j in cutlass.range(2, unroll_full=True):
                    cur_tile_mdKV_acc = tile_mdKV_acc_row[tidx, j, num_128_tiles]
                    k = tidx // 2 + j * 16
                    dim_idx = self.head_dim_main + (k // 8) * 16 + k % 8 + (tidx % 2) * 8
                    cur_mdKV_row[dim_idx] = self.element_dtype(cur_tile_mdKV_acc)

    @cute.kernel
    def sum_OdO(
        self,
        O: cute.Tensor,
        dO: cute.Tensor,
        sum_OdO: cute.Tensor,
        lse: cute.Tensor,
        attn_sink: cute.Tensor,
        scaled_lse: cute.Tensor,
        sum_OdO_scale: Float32,
        lse_scale: Float32,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, tidy, tidz = cute.arch.thread_idx()

        seqlen_q = problem_shape[0]
        offset = 0

        for idx_q_t in cutlass.range(
            tidy, self.sum_OdO_block_q, self.sum_OdO_num_threads_q, unroll_full=True
        ):
            idx_q = idx_q_t + self.sum_OdO_block_q * bidx
            if idx_q < seqlen_q:
                O_bhq = O[bidy, None, (idx_q + offset, bidz)]
                O_bhq = cute.logical_divide(O_bhq, cute.make_layout(self.sum_OdO_elem_per_load))
                dO_bhq = dO[bidy, None, (idx_q + offset, bidz)]
                dO_bhq = cute.logical_divide(dO_bhq, cute.make_layout(self.sum_OdO_elem_per_load))

                idx_d_start = tidx
                idx_d_step = self.sum_OdO_num_threads_d
                acc = 0.0
                for idx_d in cutlass.range(
                    idx_d_start, O.shape[1] // self.sum_OdO_elem_per_load, idx_d_step
                ):
                    O_frag = O_bhq[None, idx_d].load()
                    dO_frag = dO_bhq[None, idx_d].load()
                    prod_frag = O_frag * dO_frag
                    prod_frag = prod_frag.to(self.acc_dtype)
                    acc += prod_frag.reduce(cute.ReductionOp.ADD, 0.0, reduction_profile=0)

                acc = cute.arch.warp_reduction_sum(
                    acc, threads_in_group=self.sum_OdO_num_threads_d
                )

                if tidx == 0:
                    lse_bhq = lse[bidy, (idx_q + offset, bidz)]
                    sum_OdO_bhq = sum_OdO_scale * acc

                    if cutlass.const_expr(self.same_hdim_kv):
                        attn_sink_bh = attn_sink[bidy, (0, bidz)]

                        log2_e = -lse_scale
                        lse_log2 = lse_bhq * log2_e
                        sink_log2 = attn_sink_bh * log2_e
                        lse_max_log2 = cute.arch.fmax(lse_log2, sink_log2)
                        sum_exp2 = Float32(
                            cute.math.exp2(lse_log2 - lse_max_log2)
                            + cute.math.exp2(sink_log2 - lse_max_log2)
                        )
                        lse_with_sink_log2 = lse_max_log2 + cute.math.log2(sum_exp2)
                        scaled_lse_bhq = -lse_with_sink_log2

                        if lse_bhq == Float32(float("inf")):
                            scaled_lse_bhq = Float32(float("-inf"))
                    else:
                        scaled_lse_bhq = lse_scale * lse_bhq

                    sum_OdO[bidy, (idx_q + offset, bidz)] = sum_OdO_bhq
                    scaled_lse[bidy, (idx_q + offset, bidz)] = scaled_lse_bhq

    @cute.kernel
    def sum_dSink(
        self,
        sum_OdO: cute.Tensor,
        scaled_lse: cute.Tensor,
        attn_sink: cute.Tensor,
        dSink: cute.Tensor,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Tuple[Int32, Int32], Int32]],
    ):
        q_block_idx, head_idx, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        seqlen_q = problem_shape[0]
        q_end = min(seqlen_q, (q_block_idx + 1) * self.dSink_block_q)
        q_idx = q_block_idx * self.dSink_block_q + tidx

        log2_e = Float32(math.log2(math.e))
        sink_log2 = attn_sink[head_idx, (0, batch_idx)] * log2_e
        acc = Float32(0.0)

        while q_idx < q_end:
            p_sink = cute.math.exp2(sink_log2 + scaled_lse[head_idx, (q_idx, batch_idx)])
            acc += p_sink * sum_OdO[head_idx, (q_idx, batch_idx)]
            q_idx += self.dSink_num_threads

        acc = cute.arch.warp_reduction_sum(acc, threads_in_group=self.dSink_num_threads)

        if tidx == 0:
            dSink_ptr = dSink.iterator + cute.crd2idx((head_idx, (0, batch_idx)), dSink.layout)
            cute.arch.atomic_add(dSink_ptr.llvm_ptr, acc)

    @cute.kernel
    def bwd(
        self,
        problem_shape: Tuple[Int32, Int32, Int32, Tuple[Int32, Int32]],
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: Optional[cute.TiledMma],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        tma_tensor_dO: cute.Tensor,
        tma_atom_dQ: cute.CopyAtom,
        tma_tensor_dQ: cute.Tensor,
        tma_atom_dQ_64: Optional[cute.CopyAtom],
        tma_tensor_dQ_64: Optional[cute.Tensor],
        mKV: cute.Tensor,
        mdQ: cute.Tensor,
        mdKV_acc: cute.Tensor,
        mdSink: cute.Tensor,
        mAttnSink: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        mTopkLength: Optional[cute.Tensor],
        mLSE: cute.Tensor,
        mSum_OdO: cute.Tensor,
        scale_softmax: Float32 | float,
        Q_smem_layout_staged: cute.ComposedLayout,
        K_smem_layout_staged: cute.ComposedLayout,
        dO_smem_layout_staged: cute.ComposedLayout,
        V_smem_layout_staged: cute.ComposedLayout,
        dOT_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_staged: cute.ComposedLayout,
        P_smem_layout_store_staged: cute.ComposedLayout,
        K_smem_layout_staged_2: cute.ComposedLayout,
        K_tail_smem_layout_staged: Optional[cute.ComposedLayout],
        dST_smem_layout_staged: cute.ComposedLayout,
        QT_smem_layout_staged: cute.ComposedLayout,
        QT_tail_smem_layout_staged: Optional[cute.ComposedLayout],
        dS_smem_layout_staged: cute.ComposedLayout,
        dS_smem_layout_store_staged: cute.ComposedLayout,
        dKV_smem_layout_staged: cute.ComposedLayout,
        dQ_smem_layout_staged: cute.ComposedLayout,
        dQ4_smem_layout_staged: Optional[cute.ComposedLayout],
        LSE_smem_layout: cute.Layout,
        sum_OdO_smem_layout: cute.Layout,
    ):
        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        max_seqlen_q, max_seqlen_kv, head_dim, (num_heads, batch_size) = problem_shape

        if warp_idx == self.load_warp_id:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_dO)
            cpasync.prefetch_descriptor(tma_atom_dQ)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_mma_QdO_pipeline = self.make_and_init_load_mma_QdO_pipeline(
            storage.load_mma_QdO_mbar_ptr.data_ptr(),
        )
        load_mma_K_pipeline = self.make_and_init_load_mma_K_pipeline(
            storage.load_mma_K_mbar_ptr.data_ptr(),
        )
        load_compute_LSE_pipeline = self.make_and_init_load_compute_LSE_pipeline(
            storage.load_compute_LSE_mbar_ptr.data_ptr(),
        )
        load_compute_sum_OdO_pipeline = self.make_and_init_load_compute_sum_OdO_pipeline(
            storage.load_compute_sum_OdO_mbar_ptr.data_ptr(),
        )
        mma_compute_S_pipeline = self.make_and_init_mma_compute_S_pipeline(
            storage.mma_compute_S_mbar_ptr.data_ptr(),
        )
        mma_compute_dP_pipeline = self.make_and_init_mma_compute_dP_pipeline(
            storage.mma_compute_dP_mbar_ptr.data_ptr(),
        )
        mma_compute_dQ_pipeline = self.make_and_init_mma_compute_dQ_pipeline(
            storage.mma_compute_dQ_mbar_ptr.data_ptr(),
        )
        compute_mma_P_pipeline = self.make_and_init_compute_mma_P_pipeline(
            storage.compute_mma_P_mbar_ptr.data_ptr(),
        )
        compute_mma_dS_pipeline = self.make_and_init_compute_mma_dS_pipeline(
            storage.compute_mma_dS_mbar_ptr.data_ptr(),
        )
        mma_reduce_dKV_pipeline = self.make_and_init_mma_reduce_dKV_pipeline(
            storage.mma_reduce_dKV_mbar_ptr.data_ptr(),
        )
        compute_tmastore_dQ_pipeline = self.make_and_init_compute_tmastore_dQ_pipeline()

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.compute_warp_id[0],
        )

        pipeline.pipeline_init_arrive(is_relaxed=True)

        sQ = storage.sQ.get_tensor(Q_smem_layout_staged.outer, swizzle=Q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(K_smem_layout_staged.outer, swizzle=K_smem_layout_staged.inner)
        sV = storage.sK.get_tensor(V_smem_layout_staged.outer, swizzle=V_smem_layout_staged.inner)
        sP = storage.sP.get_tensor(P_smem_layout_staged.outer, swizzle=P_smem_layout_staged.inner)
        sP_store = storage.sP.get_tensor(
            P_smem_layout_store_staged.outer, swizzle=P_smem_layout_store_staged.inner
        )
        sdO = storage.sdO.get_tensor(
            dO_smem_layout_staged.outer, swizzle=dO_smem_layout_staged.inner
        )
        sdS = storage.sdS.get_tensor(
            dS_smem_layout_staged.outer, swizzle=dS_smem_layout_staged.inner
        )
        sdS_store = storage.sdS.get_tensor(
            dS_smem_layout_store_staged.outer, swizzle=dS_smem_layout_store_staged.inner
        )
        # reuse sK
        sdQ_ptr = cute.recast_ptr(sK.iterator, dQ_smem_layout_staged.inner)
        sdQ = cute.make_tensor(sdQ_ptr, dQ_smem_layout_staged.outer)

        sLSE = storage.sLSE.get_tensor(LSE_smem_layout)
        sSum_OdO = storage.sSum_OdO.get_tensor(sum_OdO_smem_layout)

        sdST_ptr = cute.recast_ptr(sdS.iterator, dST_smem_layout_staged.inner)
        sdST = cute.make_tensor(sdST_ptr, dST_smem_layout_staged.outer)

        sQT_ptr = cute.recast_ptr(sQ.iterator, QT_smem_layout_staged.inner)
        sQT = cute.make_tensor(sQT_ptr, QT_smem_layout_staged.outer)

        sdOT_ptr = cute.recast_ptr(sdO.iterator, dOT_smem_layout_staged.inner)
        sdOT = cute.make_tensor(sdOT_ptr, dOT_smem_layout_staged.outer)

        sK_2_ptr = cute.recast_ptr(sK.iterator, K_smem_layout_staged_2.inner)
        sK_2 = cute.make_tensor(sK_2_ptr, K_smem_layout_staged_2.outer)

        if cutlass.const_expr(not self.same_hdim_kv):
            # sK_tail: view sK storage with 64-wide partitioning, access block 8 (cols 512:575)
            # K_tail_smem_layout_staged partitions head_dim=576 into 64-wide blocks → 9 blocks
            sK_tail_ptr = cute.recast_ptr(sK.iterator, K_tail_smem_layout_staged.inner)
            sK_tail_full = cute.make_tensor(sK_tail_ptr, K_tail_smem_layout_staged.outer)
            sK_tail = sK_tail_full[None, 8, None, None]  # block 8 = cols 512:575

            # sQT_tail: view sQ storage with 64-wide partitioning, access block 8
            sQT_tail_ptr = cute.recast_ptr(sQ.iterator, QT_tail_smem_layout_staged.inner)
            sQT_tail_full = cute.make_tensor(sQT_tail_ptr, QT_tail_smem_layout_staged.outer)
            sQT_tail = sQT_tail_full[None, 8, None, None]  # block 8 = rows 512:575

            # sdQ4: reuse sK for the 64×64 dQ4 epilogue
            sdQ4_ptr = cute.recast_ptr(sK.iterator, dQ4_smem_layout_staged.inner)
            sdQ4 = cute.make_tensor(sdQ4_ptr, dQ4_smem_layout_staged.outer)

        pipeline.pipeline_init_wait()

        if cutlass.const_expr(mTopkLength is not None):
            topk = mTopkLength[token_idx]
        else:
            topk = mTopkIdxs.shape[0]

        tile_count = cute.ceil_div(topk, self.block_tile)

        # Load topk indices into smem cooperatively
        sTopkIdxs = storage.sTopkIdxs.get_tensor(cute.make_layout((self.smem_topk_capacity,)))
        _load_bound = min(topk, self.smem_topk_capacity)
        _idx = tidx
        while _idx < _load_bound:
            sTopkIdxs[_idx] = mTopkIdxs[_idx, (token_idx, batch_idx)]
            _idx += self.threads_per_cta
        self.cta_sync_barrier.arrive_and_wait()

        if warp_idx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            self.load(
                QK_tiled_mma,
                dOV_tiled_mma,
                tma_atom_Q,
                tma_tensor_Q,
                tma_atom_dO,
                tma_tensor_dO,
                mLSE,
                mSum_OdO,
                sQ,
                sdO,
                sLSE,
                sSum_OdO,
                (load_mma_QdO_pipeline, load_compute_LSE_pipeline, load_compute_sum_OdO_pipeline),
            )

        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            (
                tStS,
                tdPtdP,
                tdKVtdKV0,
                tdKVtdKV1,
                tdKVtdKV2,
                tdKVtdKV3,
                tdQtdQ0,
                tdQtdQ1,
                tdQtdQ2,
                tdQtdQ3,
                tdKVtdKV4,
                tdQtdQ4,
            ) = self.get_tmem_tensor(
                QK_tiled_mma,
                dOV_tiled_mma,
                QdS_tiled_mma,
                KdS_tiled_mma,
                dKV4_tiled_mma,
                dQ4_tiled_mma,
                tmem_ptr_base,
            )

            # (MMA, MMA_M, MMA_K, STAGE)
            tSrQ = QK_tiled_mma.make_fragment_A(sQ)
            # (MMA, MMA_N, MMA_K, STAGE)
            tSrK = QK_tiled_mma.make_fragment_B(sK)

            tdKVrQT = QdS_tiled_mma.make_fragment_A(sQT)
            tdKVrdS = QdS_tiled_mma.make_fragment_B(sdS)
            # Awkward, but I don't know a cleaner way to handle the modes for cute.gemm
            tdKVrQT_shape = (
                tdKVrQT.shape[0],
                1,
                tdKVrQT.shape[1],
                tdKVrQT.shape[2],
                tdKVrQT.shape[3],
            )
            tdKVrQT_stride = (
                tdKVrQT.stride[0],
                0,
                tdKVrQT.stride[1],
                tdKVrQT.stride[2],
                tdKVrQT.stride[3],
            )
            tdKVrQT = cute.make_tensor(
                tdKVrQT.iterator, cute.make_layout(tdKVrQT_shape, stride=tdKVrQT_stride)
            )

            tdPrdO = dOV_tiled_mma.make_fragment_A(sdO)
            tdPrV = dOV_tiled_mma.make_fragment_B(sV)

            tdQrK = KdS_tiled_mma.make_fragment_A(sK_2)
            tdQrdST = KdS_tiled_mma.make_fragment_B(sdST)

            tdQrK_shape = (tdQrK.shape[0], 1, tdQrK.shape[1], tdQrK.shape[2], tdQrK.shape[3])
            tdQrK_stride = (tdQrK.stride[0], 0, tdQrK.stride[1], tdQrK.stride[2], tdQrK.stride[3])
            tdQrK = cute.make_tensor(
                tdQrK.iterator, cute.make_layout(tdQrK_shape, stride=tdQrK_stride)
            )

            tdKVrdOT = dOP_tiled_mma.make_fragment_A(sdOT)
            tdKVrP = dOP_tiled_mma.make_fragment_B(sP)
            tdKVrdOT_shape = (
                tdKVrdOT.shape[0],
                1,
                tdKVrdOT.shape[1],
                tdKVrdOT.shape[2],
                tdKVrdOT.shape[3],
            )
            tdKVrdOT_stride = (
                tdKVrdOT.stride[0],
                0,
                tdKVrdOT.stride[1],
                tdKVrdOT.stride[2],
                tdKVrdOT.stride[3],
            )
            tdKVrdOT = cute.make_tensor(
                tdKVrdOT.iterator, cute.make_layout(tdKVrdOT_shape, stride=tdKVrdOT_stride)
            )

            if cutlass.const_expr(not self.same_hdim_kv):
                # dQ4 fragment: sK_tail (64-wide, single M-block) @ dS^T
                # sK_tail has 3 modes after slicing: (tile, K_blocks, stage)
                # make_fragment_A returns 3 modes: (MMA, MMA_K, STAGE)
                # Reshape to 5 modes: (MMA, 1_dummy, 1_M_block, MMA_K, STAGE)
                tdQrK_tail = dQ4_tiled_mma.make_fragment_A(sK_tail)
                tdQrK_tail_shape = (
                    tdQrK_tail.shape[0],
                    1,
                    1,
                    tdQrK_tail.shape[1],
                    tdQrK_tail.shape[2],
                )
                tdQrK_tail_stride = (
                    tdQrK_tail.stride[0],
                    0,
                    0,
                    tdQrK_tail.stride[1],
                    tdQrK_tail.stride[2],
                )
                tdQrK_tail = cute.make_tensor(
                    tdQrK_tail.iterator,
                    cute.make_layout(tdQrK_tail_shape, stride=tdQrK_tail_stride),
                )

                # dKV4 fragment: sQT_tail (64-wide, single M-block) @ dS
                tdKVrQT_tail = dKV4_tiled_mma.make_fragment_A(sQT_tail)
                tdKVrQT_tail_shape = (
                    tdKVrQT_tail.shape[0],
                    1,
                    1,
                    tdKVrQT_tail.shape[1],
                    tdKVrQT_tail.shape[2],
                )
                tdKVrQT_tail_stride = (
                    tdKVrQT_tail.stride[0],
                    0,
                    0,
                    tdKVrQT_tail.stride[1],
                    tdKVrQT_tail.stride[2],
                )
                tdKVrQT_tail = cute.make_tensor(
                    tdKVrQT_tail.iterator,
                    cute.make_layout(tdKVrQT_tail_shape, stride=tdKVrQT_tail_stride),
                )

                tdKVrdS_4 = dKV4_tiled_mma.make_fragment_B(sdS)
            else:
                tdQrK_tail = None
                tdKVrQT_tail = None
                tdKVrdS_4 = None

            self.mma(
                QK_tiled_mma,
                dOV_tiled_mma,
                dOP_tiled_mma,
                QdS_tiled_mma,
                KdS_tiled_mma,
                dKV4_tiled_mma,
                dQ4_tiled_mma,
                tSrQ,
                tSrK,
                tdPrdO,
                tdPrV,
                tdKVrdOT,
                tdKVrP,
                tdQrK,
                tdQrdST,
                tdKVrQT,
                tdKVrdS,
                tdQrK_tail,
                tdKVrQT_tail,
                tdKVrdS_4,
                tStS,
                tdPtdP,
                (tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4),
                (tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4),
                tile_count,
                sdS,
                (
                    load_mma_QdO_pipeline,
                    load_mma_K_pipeline,
                    mma_compute_S_pipeline,
                    mma_compute_dP_pipeline,
                    mma_compute_dQ_pipeline,
                    compute_mma_P_pipeline,
                    compute_mma_dS_pipeline,
                    mma_reduce_dKV_pipeline,
                ),
            )

        elif warp_idx in self.compute_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            if warp_idx == self.compute_warp_id[0]:
                tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            (
                tStS,
                tdPtdP,
                tdKVtdKV0,
                tdKVtdKV1,
                tdKVtdKV2,
                tdKVtdKV3,
                tdQtdQ0,
                tdQtdQ1,
                tdQtdQ2,
                tdQtdQ3,
                tdKVtdKV4,
                tdQtdQ4,
            ) = self.get_tmem_tensor(
                QK_tiled_mma,
                dOV_tiled_mma,
                QdS_tiled_mma,
                KdS_tiled_mma,
                dKV4_tiled_mma,
                dQ4_tiled_mma,
                tmem_ptr_base,
            )

            self.compute(
                tma_atom_dQ,
                tma_tensor_dQ,
                tma_atom_dQ_64,
                tma_tensor_dQ_64,
                dQ4_tiled_mma,
                tStS,
                tdPtdP,
                (tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4),
                sLSE,
                sSum_OdO,
                sP_store,
                sdS,
                sdS_store,
                sdQ,
                sdQ4 if not self.same_hdim_kv else None,
                scale_softmax,
                tile_count,
                (
                    mma_compute_S_pipeline,
                    mma_compute_dP_pipeline,
                    load_compute_LSE_pipeline,
                    load_compute_sum_OdO_pipeline,
                    compute_mma_P_pipeline,
                    compute_mma_dS_pipeline,
                    mma_compute_dQ_pipeline,
                    compute_tmastore_dQ_pipeline,
                ),
            )

            if warp_idx == self.compute_warp_id[0]:
                cute.arch.dealloc_tmem(tmem_ptr_base, self.num_tmem_alloc_cols)

        elif warp_idx in self.reduce_warp_id:
            cute.arch.setmaxregister_increase(self.num_regs_reduce)
            tmem.wait_for_alloc()
            tmem_ptr_base = tmem.retrieve_ptr(self.acc_dtype)

            (
                tStS,
                tdPtdP,
                tdKVtdKV0,
                tdKVtdKV1,
                tdKVtdKV2,
                tdKVtdKV3,
                tdQtdQ0,
                tdQtdQ1,
                tdQtdQ2,
                tdQtdQ3,
                tdKVtdKV4,
                tdQtdQ4,
            ) = self.get_tmem_tensor(
                QK_tiled_mma,
                dOV_tiled_mma,
                QdS_tiled_mma,
                KdS_tiled_mma,
                dKV4_tiled_mma,
                dQ4_tiled_mma,
                tmem_ptr_base,
            )

            self.reduce_dKV(
                (tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4),
                mdKV_acc,
                mTopkIdxs,
                sTopkIdxs,
                max_seqlen_kv,
                tile_count,
                topk,
                mma_reduce_dKV_pipeline,
            )

        elif warp_idx in self.load_KV_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_load_KV)
            self.load_KV(
                mKV,
                mTopkIdxs,
                sTopkIdxs,
                sK,
                tile_count,
                topk,
                load_mma_K_pipeline,
                mTopkLength,
            )

        else:
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

    @cute.jit
    def load(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        tma_atom_Q: cute.CopyAtom,
        tma_tensor_Q: cute.Tensor,
        tma_atom_dO: cute.CopyAtom,
        tma_tensor_dO: cute.Tensor,
        mLSE: cute.Tensor,
        mSum_OdO: cute.Tensor,
        sQ: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        pipelines,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        local_tidx = tidx % self.threads_per_warp

        load_mma_QdO_pipeline, load_compute_LSE_pipeline, load_compute_sum_OdO_pipeline = pipelines

        load_mma_QdO_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_mma_QdO_stage
        )
        load_compute_LSE_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_compute_LSE_stage
        )
        load_compute_sum_OdO_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_compute_sum_OdO_stage
        )

        # (bM, bK, RestM, RestK, (M, B))
        gQ = cute.local_tile(
            tma_tensor_Q,
            cute.select(self.QK_mma_tiler, mode=[0, 2]),
            (None, None, (token_idx, batch_idx)),
        )
        gdO = cute.local_tile(
            tma_tensor_dO,
            cute.select(self.dOV_mma_tiler, mode=[0, 2]),
            (None, None, (token_idx, batch_idx)),
        )

        QK_thr_mma = QK_tiled_mma.get_slice(0)
        tSgQ = QK_thr_mma.partition_A(gQ)
        tQsQ, tQgQ_mkl = cpasync.tma_partition(
            tma_atom_Q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tSgQ, 0, 3),
        )

        dOV_thr_mma = dOV_tiled_mma.get_slice(0)
        tdPgdO = dOV_thr_mma.partition_A(gdO)
        tdPsdO, tdPgdO_mkl = cpasync.tma_partition(
            tma_atom_dO,
            0,
            cute.make_layout(1),
            cute.group_modes(sdO, 0, 3),
            cute.group_modes(tdPgdO, 0, 3),
        )

        # Load Q and dO
        load_mma_QdO_pipeline.producer_acquire(load_mma_QdO_producer_state)
        tma_barrier = load_mma_QdO_pipeline.producer_get_barrier(load_mma_QdO_producer_state)
        cute.copy(
            tma_atom_Q,
            tQgQ_mkl[None, head_block_idx, 0],
            tQsQ[None, load_mma_QdO_producer_state.index],
            tma_bar_ptr=tma_barrier,
        )

        # Load dO
        cute.copy(
            tma_atom_dO,
            tdPgdO_mkl[None, head_block_idx, 0],
            tdPsdO[None, load_mma_QdO_producer_state.index],
            tma_bar_ptr=tma_barrier,
        )
        load_mma_QdO_producer_state.advance()

        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.acc_dtype,
            num_bits_per_copy=64,
        )
        thr_layout = cute.make_layout((32), stride=(1))
        val_layout = cute.make_layout((2), stride=(1))
        async_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, thr_layout, val_layout)
        thr_async_copy = async_tiled_copy.get_slice(local_tidx)

        # (64, 1, M, B)
        gLSE = cute.flat_divide(mLSE, (self.block_tile,))
        gSum_OdO = cute.flat_divide(mSum_OdO, (self.block_tile,))

        # Load LSE
        load_compute_LSE_pipeline.producer_acquire(load_compute_LSE_producer_state)

        gLSE_for_copy = thr_async_copy.partition_S(
            gLSE[None, head_block_idx, (token_idx, batch_idx)]
        )
        sLSE_for_copy = thr_async_copy.partition_D(sLSE)

        cute.copy(
            async_copy_atom,
            gLSE_for_copy[None, 0],
            sLSE_for_copy[None, 0, load_compute_LSE_producer_state.index],
        )
        load_compute_LSE_pipeline.producer_commit(load_compute_LSE_producer_state)
        load_compute_LSE_producer_state.advance()

        # Load Sum_OdO
        load_compute_sum_OdO_pipeline.producer_acquire(load_compute_sum_OdO_producer_state)

        gSum_OdO_for_copy = thr_async_copy.partition_S(
            gSum_OdO[None, head_block_idx, (token_idx, batch_idx)]
        )
        sSum_OdO_for_copy = thr_async_copy.partition_D(sSum_OdO)

        cute.copy(
            async_copy_atom,
            gSum_OdO_for_copy[None, 0],
            sSum_OdO_for_copy[None, 0, load_compute_sum_OdO_producer_state.index],
        )

        load_compute_sum_OdO_pipeline.producer_commit(load_compute_sum_OdO_producer_state)
        load_compute_sum_OdO_producer_state.advance()

    @cute.jit
    def _copy_kv_row(
        self,
        mKV: cute.Tensor,
        topk_idx: Int32,
        batch_idx: Int32,
        tile_sK: cute.Tensor,
        local_tidx: Int32,
        async_copy_atom: cute.CopyAtom,
        async_thr_copy: cute.TiledCopy,
    ):
        gK_row = mKV[topk_idx, None, (0, batch_idx)]
        tile_gK = cute.composition(gK_row, cute.make_layout(tile_sK.shape))
        for j in cutlass.range_constexpr(3):
            group_idx = j * 4 + local_tidx // 8
            if group_idx < self.head_dim // 64:
                cur_gK = tile_gK[None, group_idx]
                cur_sK = tile_sK[None, group_idx]
                tSgK = async_thr_copy.partition_S(cur_gK)
                tSsK = async_thr_copy.partition_D(cur_sK)
                cute.copy(async_copy_atom, tSgK, tSsK)

    @cute.jit
    def _zero_kv_row(
        self,
        tile_sK: cute.Tensor,
        local_tidx: Int32,
    ):
        for j in cutlass.range_constexpr(3):
            group_idx = j * 4 + local_tidx // 8
            if group_idx < self.head_dim // 64:
                cur_sK = tile_sK[None, group_idx]
                cur_sK = cute.flat_divide(cur_sK, (8,))  # (8, 8)
                cur_sK = cur_sK[None, local_tidx % 8]
                cur_sK.fill(0.0)

    @cute.jit
    def _load_kv_rows(
        self,
        mKV: cute.Tensor,
        sK_slice: cute.Tensor,
        rTopkIdx: cute.Tensor,
        tile_index: Int32,
        topk: Int32,
        mTopkLength: Optional[cute.Tensor],
        is_first: bool,
    ):
        """Load one tile of KV rows into sK_slice. Compile-time specialization via is_first."""
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        local_tidx = tidx % self.threads_per_warp
        local_warp_idx = tidx // self.threads_per_warp

        async_copy_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.element_dtype,
            num_bits_per_copy=128,
        )
        thr_layout = cute.make_layout((8,))
        val_layout = cute.make_layout((8,))
        async_tiled_copy = cute.make_tiled_copy_tv(async_copy_atom, thr_layout, val_layout)
        async_thr_copy = async_tiled_copy.get_slice(local_tidx % 8)

        rows_per_warp = self.block_tile // self.num_load_KV_warps
        for i in range(rows_per_warp):
            row = i * self.num_load_KV_warps + local_warp_idx
            idx = tile_index * self.block_tile + row
            tile_sK = sK_slice[row, (None, None)]
            topk_idx = rTopkIdx[i]

            if cutlass.const_expr(mTopkLength is not None):
                if cutlass.const_expr(is_first):
                    if idx < topk:
                        self._copy_kv_row(
                            mKV,
                            topk_idx,
                            batch_idx,
                            tile_sK,
                            local_tidx,
                            async_copy_atom,
                            async_thr_copy,
                        )
                    else:
                        self._zero_kv_row(tile_sK, local_tidx)
                else:
                    self._copy_kv_row(
                        mKV,
                        topk_idx,
                        batch_idx,
                        tile_sK,
                        local_tidx,
                        async_copy_atom,
                        async_thr_copy,
                    )
            else:
                if idx < topk:
                    if topk_idx >= 0:
                        self._copy_kv_row(
                            mKV,
                            topk_idx,
                            batch_idx,
                            tile_sK,
                            local_tidx,
                            async_copy_atom,
                            async_thr_copy,
                        )
                    else:
                        self._zero_kv_row(tile_sK, local_tidx)
                else:
                    self._zero_kv_row(tile_sK, local_tidx)

    @cute.jit
    def load_KV(
        self,
        mKV: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        sTopkIdxs: cute.Tensor,
        sK: cute.Tensor,
        tile_count: Int32,
        topk: Int32,
        load_mma_K_pipeline,
        mTopkLength: Optional[cute.Tensor],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        local_warp_idx = tidx // self.threads_per_warp

        load_mma_K_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.load_mma_K_stage
        )

        tile_index = tile_count - 1

        # ---- First tile (tail, may be partial) ----
        rows_per_warp = self.block_tile // self.num_load_KV_warps
        rTopkIdx = cute.make_rmem_tensor((rows_per_warp,), cutlass.Int32)
        for i in range(rows_per_warp):
            row = i * self.num_load_KV_warps + local_warp_idx
            idx = tile_index * self.block_tile + row
            if idx < self.smem_topk_capacity:
                rTopkIdx[i] = sTopkIdxs[idx]
            else:
                rTopkIdx[i] = mTopkIdxs[idx, (token_idx, batch_idx)]

        load_mma_K_pipeline.producer_acquire(load_mma_K_producer_state)
        sK_slice = sK[(None, None), 0, (None, None), load_mma_K_producer_state.index]
        sK_slice = cute.composition(sK_slice, cute.make_layout((self.block_tile, self.head_dim)))

        self._load_kv_rows(mKV, sK_slice, rTopkIdx, tile_index, topk, mTopkLength, is_first=True)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        self.load_KV_sync_barrier.arrive_and_wait()
        load_mma_K_pipeline.producer_commit(load_mma_K_producer_state)
        load_mma_K_producer_state.advance()
        tile_index -= 1

        # ---- Remaining tiles (all full, right-to-left) ----
        while tile_index >= 0:
            rTopkIdx = cute.make_rmem_tensor((rows_per_warp,), cutlass.Int32)
            for i in range(rows_per_warp):
                row = i * self.num_load_KV_warps + local_warp_idx
                idx = tile_index * self.block_tile + row
                if idx < self.smem_topk_capacity:
                    rTopkIdx[i] = sTopkIdxs[idx]
                else:
                    rTopkIdx[i] = mTopkIdxs[idx, (token_idx, batch_idx)]

            load_mma_K_pipeline.producer_acquire(load_mma_K_producer_state)
            sK_slice = sK[(None, None), 0, (None, None), load_mma_K_producer_state.index]
            sK_slice = cute.composition(
                sK_slice, cute.make_layout((self.block_tile, self.head_dim))
            )

            self._load_kv_rows(
                mKV, sK_slice, rTopkIdx, tile_index, topk, mTopkLength, is_first=False
            )

            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            self.load_KV_sync_barrier.arrive_and_wait()
            load_mma_K_pipeline.producer_commit(load_mma_K_producer_state)
            load_mma_K_producer_state.advance()
            tile_index -= 1

    @cute.jit
    def mma(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        dOP_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: Optional[cute.TiledMma],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tdPrdO: cute.Tensor,
        tdPrV: cute.Tensor,
        tdKVrdOT: cute.Tensor,
        tdKVrP: cute.Tensor,
        tdQrK: cute.Tensor,
        tdQrdST: cute.Tensor,
        tdKVrQT: cute.Tensor,
        tdKVrdS: cute.Tensor,
        tdQrK_tail: Optional[cute.Tensor],
        tdKVrQT_tail: Optional[cute.Tensor],
        tdKVrdS_4: Optional[cute.Tensor],
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdKVtdKV: Tuple,
        tdQtdQ: Tuple,
        tile_count: Int32,
        sdS: cute.Tensor,
        pipelines,
    ):
        (
            load_mma_QdO_pipeline,
            load_mma_K_pipeline,
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            mma_compute_dQ_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_reduce_dKV_pipeline,
        ) = pipelines
        tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4 = tdKVtdKV
        tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4 = tdQtdQ

        tidx, _, _ = cute.arch.thread_idx()
        local_tidx = tidx % self.threads_per_warp
        token_idx, _, batch_idx = cute.arch.block_idx()

        load_mma_QdO_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_mma_QdO_stage
        )
        load_mma_K_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_mma_K_stage
        )
        mma_compute_S_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_S_stage
        )
        mma_compute_dP_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_dP_stage
        )
        mma_compute_dQ_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_compute_dQ_stage
        )
        compute_mma_P_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_P_stage
        )
        compute_mma_dS_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.compute_mma_dS_stage
        )
        mma_reduce_dKV_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.mma_reduce_dKV_stage
        )

        load_mma_QdO_pipeline.consumer_wait(load_mma_QdO_consumer_state)
        mma_compute_dQ_pipeline.producer_acquire(mma_compute_dQ_producer_state)

        tile_index = tile_count - 1
        is_first_mma = True
        while tile_index >= 0:
            load_mma_K_pipeline.consumer_wait(load_mma_K_consumer_state)
            mma_compute_S_pipeline.producer_acquire(mma_compute_S_producer_state)
            # Gemm S = Q @ K
            QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tSrQ, mode=[2]), unroll=4):
                cute.gemm(
                    QK_tiled_mma,
                    tStS,
                    tSrQ[None, None, k_block, load_mma_QdO_consumer_state.index],
                    tSrK[None, None, k_block, load_mma_K_consumer_state.index],
                    tStS,
                )
                QK_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_compute_S_pipeline.producer_commit(mma_compute_S_producer_state)
            mma_compute_S_producer_state.advance()

            # Gemm dP = dO @ V
            mma_compute_dP_pipeline.producer_acquire(mma_compute_dP_producer_state)
            dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdPrdO, mode=[2]), unroll=4):
                cute.gemm(
                    dOV_tiled_mma,
                    tdPtdP,
                    tdPrdO[None, None, k_block, load_mma_QdO_consumer_state.index],
                    tdPrV[None, None, k_block, load_mma_K_consumer_state.index],
                    tdPtdP,
                )
                dOV_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            mma_compute_dP_pipeline.producer_commit(mma_compute_dP_producer_state)
            mma_compute_dP_producer_state.advance()

            # Gemm dKV = dO @ P part1
            compute_mma_P_pipeline.consumer_wait(compute_mma_P_consumer_state)
            mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)

            # dKV0
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV0,
                    tdKVrdOT[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                    tdKVtdKV0,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dKV1
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV1,
                    tdKVrdOT[None, None, 1, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                    tdKVtdKV1,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            compute_mma_dS_pipeline.consumer_wait(compute_mma_dS_consumer_state)

            # Gemm dKV = Q @ dS part1
            # dKV0
            QdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKVtdKV0,
                    tdKVrQT[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKVtdKV0,
                )
            # dKV1
            for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKVtdKV1,
                    tdKVrQT[None, None, 1, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKVtdKV1,
                )

            # Notify to reduce the first part of dKV (dKV0, dKV1)
            mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
            mma_reduce_dKV_producer_state.advance()

            # Gemm dKV4 = Q^T[512:575] @ dS (round 1.5, only GEMM5, no GEMM3)
            # dKV4 skips producer_acquire — barrier1 guarantees TMEM safety.
            # Only producer_commit is needed to notify consumer.
            if cutlass.const_expr(not self.same_hdim_kv):
                # barrier1: wait for reduce warps to finish T2R of dKV0/dKV1
                self.t2r_dKV01_done_barrier.arrive_and_wait()

                dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_block in cutlass.range(0, cute.size(tdKVrdS_4, mode=[2]), unroll=2):
                    cute.gemm(
                        dKV4_tiled_mma,
                        tdKVtdKV4,
                        tdKVrQT_tail[None, None, 0, k_block, load_mma_QdO_consumer_state.index],
                        tdKVrdS_4[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdKVtdKV4,
                    )
                    dKV4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                # Commit dKV4 on pipeline (no acquire needed) to notify consumer
                mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
                mma_reduce_dKV_producer_state.advance()

            # Gemm dQ = K @ dS

            # dQ0
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ0,
                    tdQrK[None, None, 0, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ0,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ1
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ1,
                    tdQrK[None, None, 1, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ1,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ2
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ2,
                    tdQrK[None, None, 2, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ2,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ3
            KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
            for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                cute.gemm(
                    KdS_tiled_mma,
                    tdQtdQ3,
                    tdQrK[None, None, 3, k_block, load_mma_K_consumer_state.index],
                    tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdQtdQ3,
                )
                KdS_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dQ4 (tail 64 cols: K[512:575] @ dS^T)
            if cutlass.const_expr(not self.same_hdim_kv):
                dQ4_tiled_mma.set(tcgen05.Field.ACCUMULATE, not is_first_mma)
                for k_block in cutlass.range(0, cute.size(tdQrdST, mode=[2]), unroll=2):
                    cute.gemm(
                        dQ4_tiled_mma,
                        tdQtdQ4,
                        tdQrK_tail[None, None, 0, k_block, load_mma_K_consumer_state.index],
                        tdQrdST[None, None, k_block, compute_mma_dS_consumer_state.index],
                        tdQtdQ4,
                    )
                    dQ4_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # KV is used
            load_mma_K_pipeline.consumer_release(load_mma_K_consumer_state)
            load_mma_K_consumer_state.advance()

            # Gemm dKV = dO @ P part2
            # Wait for reduce warps to finish T2R of dKV4 from TMEM,
            # since dKV2 shares the same TMEM offset as dKV4/dKV0.
            if cutlass.const_expr(not self.same_hdim_kv):
                self.t2r_dKV4_done_barrier.arrive_and_wait()
            mma_reduce_dKV_pipeline.producer_acquire(mma_reduce_dKV_producer_state)
            # dKV2
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV2,
                    tdKVrdOT[None, None, 2, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                    tdKVtdKV2,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # dKV3
            dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_block in cutlass.range(0, cute.size(tdKVrP, mode=[2]), unroll=2):
                cute.gemm(
                    dOP_tiled_mma,
                    tdKVtdKV3,
                    tdKVrdOT[None, None, 3, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrP[None, None, k_block, compute_mma_P_consumer_state.index],
                    tdKVtdKV3,
                )
                dOP_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

            # P is used
            compute_mma_P_pipeline.consumer_release(compute_mma_P_consumer_state)
            compute_mma_P_consumer_state.advance()

            # Gemm dKV = Q @ dS
            # dKV2
            for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKVtdKV2,
                    tdKVrQT[None, None, 2, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKVtdKV2,
                )
            # dKV3
            for k_block in cutlass.range(0, cute.size(tdKVrdS, mode=[2]), unroll=2):
                cute.gemm(
                    QdS_tiled_mma,
                    tdKVtdKV3,
                    tdKVrQT[None, None, 3, k_block, load_mma_QdO_consumer_state.index],
                    tdKVrdS[None, None, k_block, compute_mma_dS_consumer_state.index],
                    tdKVtdKV3,
                )

            mma_reduce_dKV_pipeline.producer_commit(mma_reduce_dKV_producer_state)
            mma_reduce_dKV_producer_state.advance()

            # dS is used
            compute_mma_dS_pipeline.consumer_release(compute_mma_dS_consumer_state)
            compute_mma_dS_consumer_state.advance()

            is_first_mma = False
            tile_index -= 1

        mma_compute_dQ_pipeline.producer_commit(mma_compute_dQ_producer_state)
        mma_compute_dQ_producer_state.advance()

        # Q and dO is used
        load_mma_QdO_pipeline.consumer_release(load_mma_QdO_consumer_state)
        load_mma_QdO_consumer_state.advance()

    @cute.jit
    def compute(
        self,
        tma_atom_dQ: cute.CopyAtom,
        tma_tensor_dQ: cute.Tensor,
        tma_atom_dQ_64: Optional[cute.CopyAtom],
        tma_tensor_dQ_64: Optional[cute.Tensor],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdQtdQ: Tuple,
        sLSE: cute.Tensor,
        sSum_OdO: cute.Tensor,
        sP_store: cute.Tensor,
        sdS: cute.Tensor,
        sdS_store: cute.Tensor,
        sdQ: cute.Tensor,
        sdQ4: Optional[cute.Tensor],
        scale_softmax: Float32,
        tile_count: Int32,
        pipelines,
    ):
        (
            mma_compute_S_pipeline,
            mma_compute_dP_pipeline,
            load_compute_LSE_pipeline,
            load_compute_sum_OdO_pipeline,
            compute_mma_P_pipeline,
            compute_mma_dS_pipeline,
            mma_compute_dQ_pipeline,
            compute_tmastore_dQ_pipeline,
        ) = pipelines

        tdQtdQ0, tdQtdQ1, tdQtdQ2, tdQtdQ3, tdQtdQ4 = tdQtdQ

        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.compute_warp_id[0] * self.threads_per_warp
        tidx_in_warp = tidx % self.threads_per_warp

        token_idx, head_block_idx, batch_idx = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        mma_compute_S_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_S_stage
        )
        mma_compute_dP_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dP_stage
        )
        mma_compute_dQ_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_compute_dQ_stage
        )
        compute_mma_P_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_mma_P_stage
        )
        compute_mma_dS_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_mma_dS_stage
        )
        load_compute_LSE_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_compute_LSE_stage
        )
        load_compute_sum_OdO_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_compute_sum_OdO_stage
        )
        compute_tmastore_dQ_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.compute_tmastore_dQ_stage
        )
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )

        # (((16,4),64), 1, 1):(((65536,2097152),1),0,0)
        tStS = tStS[(None, None), 0, 0]
        tdPtdP = tdPtdP[(None, None), 0, 0]

        dp_idx = tidx_in_wg % 128
        cS = cute.make_identity_tensor(cute.select(self.QK_mma_tiler, mode=[0, 1]))
        cS = cute.composition(cS, sP_store[None, None, compute_mma_P_producer_state.index].layout)
        cdP = cute.make_identity_tensor(cute.select(self.dOV_mma_tiler, mode=[0, 1]))
        cdP = cute.composition(
            cdP, sdS_store[None, None, compute_mma_dS_producer_state.index].layout
        )

        tiled_t2r_S = tcgen05.make_tmem_copy(tmem_load_atom, tStS)
        tiled_t2r_dP = tcgen05.make_tmem_copy(tmem_load_atom, tdPtdP)
        thr_t2r_S = tiled_t2r_S.get_slice(tidx % 128)
        thr_t2r_dP = tiled_t2r_dP.get_slice(tidx % 128)

        tTR_cS = thr_t2r_S.partition_D(cS)
        tTR_sS = thr_t2r_S.partition_D(sP_store[None, None, compute_mma_P_producer_state.index])
        tTR_rS = cute.make_rmem_tensor(tTR_sS.shape, self.acc_dtype)

        tTR_tS = thr_t2r_S.partition_S(tStS)

        tTR_cdP = thr_t2r_dP.partition_D(cdP)
        tTR_sdP = thr_t2r_dP.partition_D(
            sdS_store[None, None, compute_mma_dS_producer_state.index]
        )
        tTR_rdP = cute.make_rmem_tensor(tTR_sdP.shape, self.acc_dtype)

        tTR_tdP = thr_t2r_dP.partition_S(tdPtdP)

        load_compute_LSE_pipeline.consumer_wait(load_compute_LSE_consumer_state)
        load_compute_sum_OdO_pipeline.consumer_wait(load_compute_sum_OdO_consumer_state)

        tile_index = tile_count - 1
        while tile_index >= 0:
            mma_compute_S_pipeline.consumer_wait(mma_compute_S_consumer_state)
            compute_mma_P_pipeline.producer_acquire(compute_mma_P_producer_state)

            cute.copy(tiled_t2r_S, tTR_tS, tTR_rS)

            log2_e = Float32(math.log2(math.e))
            softmax_scale_log2_e = scale_softmax * log2_e

            for i in cutlass.range(0, cute.size(tTR_rS), 2):
                lse = (
                    sLSE[cute.get(tTR_cS[i], mode=[0]), load_compute_LSE_consumer_state.index],
                    sLSE[cute.get(tTR_cS[i + 1], mode=[0]), load_compute_LSE_consumer_state.index],
                )

                tTR_rS[i], tTR_rS[i + 1] = cute.arch.fma_packed_f32x2(
                    (tTR_rS[i], tTR_rS[i + 1]),
                    (softmax_scale_log2_e, softmax_scale_log2_e),
                    lse,
                )

                tTR_rS[i] = cute.math.exp2(tTR_rS[i], fastmath=True)
                tTR_rS[i + 1] = cute.math.exp2(tTR_rS[i + 1], fastmath=True)

            tTR_rS_f16 = self.quantize(tTR_rS, 4)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()

            # ======= stsm ============
            smem_store_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.element_dtype,
            )
            smem_store_p = cute.make_tiled_copy_D(smem_store_atom, tiled_t2r_S)
            thr_smem_store_p = smem_store_p.get_slice(tidx % 128)
            sP_store_slice = sP_store[None, None, compute_mma_P_producer_state.index]
            tRS_sP = thr_smem_store_p.partition_D(sP_store_slice)
            tRS_rP = cute.make_rmem_tensor(tRS_sP.shape, self.element_dtype)
            tRS_rP.store(smem_store_p.retile(tTR_rS_f16).load())
            cute.copy(smem_store_p, tRS_rP, tRS_sP)

            # Fence for shared memory
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            # Notify for P
            compute_mma_P_pipeline.producer_commit(compute_mma_P_producer_state)
            compute_mma_P_producer_state.advance()

            mma_compute_S_pipeline.consumer_release(mma_compute_S_consumer_state)
            mma_compute_S_consumer_state.advance()

            mma_compute_dP_pipeline.consumer_wait(mma_compute_dP_consumer_state)
            compute_mma_dS_pipeline.producer_acquire(compute_mma_dS_producer_state)

            cute.copy(tiled_t2r_dP, tTR_tdP, tTR_rdP)

            for i in cutlass.range(0, cute.size(tTR_rdP), 2):
                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.add_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]),
                    (
                        sSum_OdO[
                            cute.get(tTR_cdP[i], mode=[0]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                        sSum_OdO[
                            cute.get(tTR_cdP[i + 1], mode=[0]),
                            load_compute_sum_OdO_consumer_state.index,
                        ],
                    ),
                )

                tTR_rdP[i], tTR_rdP[i + 1] = cute.arch.mul_packed_f32x2(
                    (tTR_rdP[i], tTR_rdP[i + 1]), (tTR_rS[i], tTR_rS[i + 1])
                )

            tTR_rdP_f16 = self.quantize(tTR_rdP, 4, scale_softmax)

            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()

            mma_compute_dP_pipeline.consumer_release(mma_compute_dP_consumer_state)
            mma_compute_dP_consumer_state.advance()

            smem_store_ds = cute.make_tiled_copy_D(smem_store_atom, tiled_t2r_dP)
            thr_smem_store_ds = smem_store_ds.get_slice(tidx % 128)
            sdS_store_slice = sdS_store[None, None, compute_mma_dS_producer_state.index]
            tRS_sdS = thr_smem_store_ds.partition_D(sdS_store_slice)
            tRS_rdS = cute.make_rmem_tensor(tRS_sdS.shape, self.element_dtype)
            tRS_rdS.store(smem_store_ds.retile(tTR_rdP_f16).load())
            cute.copy(smem_store_ds, tRS_rdS, tRS_sdS)

            # self.compute_sync_barrier.arrive_and_wait()

            # Fence for shared memory
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )

            compute_mma_dS_pipeline.producer_commit(compute_mma_dS_producer_state)
            compute_mma_dS_producer_state.advance()

            tile_index -= 1

        load_compute_LSE_pipeline.consumer_release(load_compute_LSE_consumer_state)
        load_compute_sum_OdO_pipeline.consumer_release(load_compute_sum_OdO_consumer_state)

        # Store dQ
        tdQtdQ0 = tdQtdQ0[(None, None), 0, 0]
        tdQtdQ1 = tdQtdQ1[(None, None), 0, 0]
        tdQtdQ2 = tdQtdQ2[(None, None), 0, 0]
        tdQtdQ3 = tdQtdQ3[(None, None), 0, 0]

        # (512, 64)
        gdQ = cute.local_tile(
            tma_tensor_dQ,
            cute.select(self.KdS_mma_tiler, mode=[0, 1]),
            (None, None, (token_idx, batch_idx)),
        )
        # (128, 64)
        gdQ0 = gdQ[None, None, 0, head_block_idx]
        gdQ1 = gdQ[None, None, 1, head_block_idx]
        gdQ2 = gdQ[None, None, 2, head_block_idx]
        gdQ3 = gdQ[None, None, 3, head_block_idx]

        # sdQ: ((64,2),(8,8),(1,1))
        sdQ_slice = sdQ[None, None, mma_compute_dQ_consumer_state.index]

        # ((64,2),(8,8),(1,1))
        tdQsdQ0, tdQgdQ0_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice, 0, 2),
            cute.group_modes(gdQ0, 0, 2),
        )
        tdQsdQ1, tdQgdQ1_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice, 0, 2),
            cute.group_modes(gdQ1, 0, 2),
        )
        tdQsdQ2, tdQgdQ2_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice, 0, 2),
            cute.group_modes(gdQ2, 0, 2),
        )
        tdQsdQ3, tdQgdQ3_mkl = cpasync.tma_partition(
            tma_atom_dQ,
            0,
            cute.make_layout(1),
            cute.group_modes(sdQ_slice, 0, 2),
            cute.group_modes(gdQ3, 0, 2),
        )

        if cutlass.const_expr(not self.same_hdim_kv):
            tdQtdQ4 = tdQtdQ4[(None, None), 0, 0]
            gdQ4 = cute.local_tile(
                tma_tensor_dQ_64,
                cute.select(self.dQ4_mma_tiler, mode=[0, 1]),
                (None, None, (token_idx, batch_idx)),
            )
            gdQ4 = gdQ4[None, None, 8, head_block_idx]

            sdQ4_slice = sdQ4[None, None, mma_compute_dQ_consumer_state.index]

            tdQsdQ4, tdQgdQ4_mkl = cpasync.tma_partition(
                tma_atom_dQ_64,
                0,
                cute.make_layout(1),
                cute.group_modes(sdQ4_slice, 0, 2),
                cute.group_modes(gdQ4, 0, 2),
            )

        dp_idx = tidx % 128
        wg_idx = (tidx % (self.num_compute_warps * self.threads_per_warp)) // 128

        mma_compute_dQ_pipeline.consumer_wait(mma_compute_dQ_consumer_state)

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        # Wait in all threads for the acquire to complete
        self.compute_sync_barrier.arrive_and_wait()

        self.store_dQ(
            tma_atom_dQ,
            sdQ_slice,
            tdQsdQ0,
            tdQgdQ0_mkl,
            tdQtdQ0,
            dp_idx,
            warp_idx,
        )

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()

        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()

        self.store_dQ(
            tma_atom_dQ,
            sdQ_slice,
            tdQsdQ1,
            tdQgdQ1_mkl,
            tdQtdQ1,
            dp_idx,
            warp_idx,
        )

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()

        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()

        self.store_dQ(
            tma_atom_dQ,
            sdQ_slice,
            tdQsdQ2,
            tdQgdQ2_mkl,
            tdQtdQ2,
            dp_idx,
            warp_idx,
        )

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()

        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_acquire()
        self.compute_sync_barrier.arrive_and_wait()

        self.store_dQ(
            tma_atom_dQ,
            sdQ_slice,
            tdQsdQ3,
            tdQgdQ3_mkl,
            tdQtdQ3,
            dp_idx,
            warp_idx,
        )

        if warp_idx == self.compute_warp_id[0]:
            compute_tmastore_dQ_pipeline.producer_commit()
        self.compute_sync_barrier.arrive_and_wait()
        compute_tmastore_dQ_producer_state.advance()

        # Store dQ4 (tail 64 cols)
        if cutlass.const_expr(not self.same_hdim_kv):
            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_acquire()
            self.compute_sync_barrier.arrive_and_wait()

            self.store_dQ_64(
                tma_atom_dQ_64,
                sdQ4_slice,
                tdQsdQ4,
                tdQgdQ4_mkl,
                tdQtdQ4,
                dp_idx,
                wg_idx,
                warp_idx,
            )

            if warp_idx == self.compute_warp_id[0]:
                compute_tmastore_dQ_pipeline.producer_commit()
            self.compute_sync_barrier.arrive_and_wait()
            compute_tmastore_dQ_producer_state.advance()

        mma_compute_dQ_pipeline.consumer_release(mma_compute_dQ_consumer_state)
        mma_compute_dQ_consumer_state.advance()

        compute_tmastore_dQ_pipeline.producer_tail()

    @cute.jit
    def reduce_dKV(
        self,
        tdKVtdKV: Tuple,
        mdKV_acc: cute.Tensor,
        mTopkIdxs: cute.Tensor,
        sTopkIdxs: cute.Tensor,
        max_seqlen_kv: Int32,
        tile_count: Int32,
        topk: Int32,
        mma_reduce_dKV_pipeline,
    ):
        tdKVtdKV0, tdKVtdKV1, tdKVtdKV2, tdKVtdKV3, tdKVtdKV4 = tdKVtdKV

        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)

        num_warp_groups = self.num_reduce_warps // 4

        tdKVtdKV0 = tdKVtdKV0[(None, None), 0, 0]
        tdKVtdKV1 = tdKVtdKV1[(None, None), 0, 0]
        tdKVtdKV2 = tdKVtdKV2[(None, None), 0, 0]
        tdKVtdKV3 = tdKVtdKV3[(None, None), 0, 0]
        if cutlass.const_expr(not self.same_hdim_kv):
            tdKVtdKV4 = tdKVtdKV4[(None, None), 0, 0]

        # Set up identity tensor partition once (all dKV sub-tiles share the same layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV0)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)

        # Set up 64-wide identity tensor partition for dKV4
        if cutlass.const_expr(not self.same_hdim_kv):
            cdKV_64 = cute.make_identity_tensor((self.dKV4_mma_tiler[0], self.dKV4_mma_tiler[1]))
            tiled_t2r_dKV_64 = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV4)
            thr_t2r_dKV_64 = tiled_t2r_dKV_64.get_slice(dp_idx)
            tTR_cdKV_64 = self.split_wg(
                thr_t2r_dKV_64.partition_D(cdKV_64), num_warp_groups, wg_idx
            )

        mma_reduce_dKV_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.mma_reduce_dKV_stage
        )

        tile_index = tile_count - 1
        rTopkIdx = cute.make_rmem_tensor((8,), cutlass.Int32)
        rTopkIdx_64 = cute.make_rmem_tensor((8,), cutlass.Int32)
        while tile_index >= 0:
            # Preload topk indices into rmem (shared across all 4 store_dKV calls)
            for i in cutlass.range_constexpr(8):
                coord_base = i * 2 - i % 2
                local_row_idx = cute.get(tTR_cdKV[coord_base], mode=[1])
                global_row_idx = tile_index * self.block_tile + local_row_idx
                if global_row_idx < topk:
                    if global_row_idx < self.smem_topk_capacity:
                        rTopkIdx[i] = sTopkIdxs[global_row_idx]
                    else:
                        rTopkIdx[i] = mTopkIdxs[global_row_idx, (token_idx, batch_idx)]
                else:
                    rTopkIdx[i] = Int32(-1)

            # Preload topk indices for 64-wide dKV4 (different split_wg → different row coords)
            if cutlass.const_expr(not self.same_hdim_kv):
                for i in cutlass.range_constexpr(8):
                    coord_base = i * 2 - i % 2
                    local_row_idx = cute.get(tTR_cdKV_64[coord_base], mode=[1])
                    global_row_idx = tile_index * self.block_tile + local_row_idx
                    if global_row_idx < topk:
                        if global_row_idx < self.smem_topk_capacity:
                            rTopkIdx_64[i] = sTopkIdxs[global_row_idx]
                        else:
                            rTopkIdx_64[i] = mTopkIdxs[global_row_idx, (token_idx, batch_idx)]
                    else:
                        rTopkIdx_64[i] = Int32(-1)

            mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

            if cutlass.const_expr(not self.same_hdim_kv):
                # Split T2R and atomic_add: load dKV0/dKV1 from TMEM to registers,
                # then signal MMA warp that TMEM is free for dKV4 overlap.
                rdKV0 = self.t2r_dKV(tdKVtdKV0)
                rdKV1 = self.t2r_dKV(tdKVtdKV1)
                cute.arch.fence_view_async_tmem_load()
                self.t2r_dKV01_done_barrier.arrive_and_wait()
                self.reduce_dKV_from_reg(mdKV_acc, rdKV0, rTopkIdx, 0)
                self.reduce_dKV_from_reg(mdKV_acc, rdKV1, rTopkIdx, 1)
            else:
                self.store_dKV(mdKV_acc, tdKVtdKV0, rTopkIdx, 0)
                self.store_dKV(mdKV_acc, tdKVtdKV1, rTopkIdx, 1)

            mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
            mma_reduce_dKV_consumer_state.advance()

            # dKV4 reduce (round 1.5): cols 512:575
            # dKV4 reduce: use pipeline for notification, barrier for T2R safety
            if cutlass.const_expr(not self.same_hdim_kv):
                mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

                # T2R dKV4, then signal MMA that TMEM is free for dKV2/dKV3
                rdKV4 = self.t2r_dKV_64(tdKVtdKV4)
                cute.arch.fence_view_async_tmem_load()
                self.t2r_dKV4_done_barrier.arrive_and_wait()
                self.reduce_dKV_64_from_reg(mdKV_acc, rdKV4, rTopkIdx_64)

                mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
                mma_reduce_dKV_consumer_state.advance()

            mma_reduce_dKV_pipeline.consumer_wait(mma_reduce_dKV_consumer_state)

            self.store_dKV(mdKV_acc, tdKVtdKV2, rTopkIdx, 2)
            self.store_dKV(mdKV_acc, tdKVtdKV3, rTopkIdx, 3)

            mma_reduce_dKV_pipeline.consumer_release(mma_reduce_dKV_consumer_state)
            mma_reduce_dKV_consumer_state.advance()

            tile_index -= 1

    @cute.jit
    def t2r_dKV(self, tdKVtdKV: cute.Tensor):
        """T2R: load dKV from TMEM to registers. Caller must fence after all T2R calls."""
        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)

        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r_dKV.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dKV, tTR_tdKV, tTR_rdKV)
        return tTR_rdKV

    @cute.jit
    def reduce_dKV_from_reg(
        self,
        dKV_acc: cute.Tensor,
        tTR_rdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
        sub_tile_idx: int,
    ):
        """Reduce dKV from registers to global memory via atomic_add."""
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128

        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2

            rdKV_frg = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]
            rdKV_frg[2] = tTR_rdKV[coord_base + 16]
            rdKV_frg[3] = tTR_rdKV[coord_base + 18]

            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (128,))  # (128, 4)
                tile_dKV_row = tile_dKV_row[None, sub_tile_idx]
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (4,))  # (4, 32)
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load())

        self.reduce_sync_barrier.arrive_and_wait()

    @cute.jit
    def t2r_dKV_64(self, tdKVtdKV: cute.Tensor):
        """T2R: load 64-wide dKV from TMEM to registers. Caller must fence."""
        tidx, _, _ = cute.arch.thread_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )
        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)

        cdKV = cute.make_identity_tensor((self.dKV4_mma_tiler[0], self.dKV4_mma_tiler[1]))
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r_dKV.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dKV, tTR_tdKV, tTR_rdKV)
        return tTR_rdKV

    @cute.jit
    def reduce_dKV_64_from_reg(
        self,
        dKV_acc: cute.Tensor,
        tTR_rdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
    ):
        """Reduce 64-wide dKV from registers to global memory via atomic_add."""
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128

        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2

            rdKV_frg = cute.make_rmem_tensor((2,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]

            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (64,))  # (64, D/64)
                tile_dKV_row = tile_dKV_row[None, self.head_dim_main // 64]  # last 64-elem tile
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (2,))  # (2, 32)
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load())

        self.reduce_sync_barrier.arrive_and_wait()

    @cute.jit
    def store_dKV(
        self,
        dKV_acc: cute.Tensor,
        tdKVtdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
        sub_tile_idx: int,
    ):
        tidx, _, batch_idx = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)

        cdKV = cute.make_identity_tensor((self.dOP_mma_tiler[0], self.dOP_mma_tiler[1]))
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r_dKV.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dKV, tTR_tdKV, tTR_rdKV)

        cute.arch.fence_view_async_tmem_load()

        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2

            rdKV_frg = cute.make_rmem_tensor((4,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]
            rdKV_frg[2] = tTR_rdKV[coord_base + 16]
            rdKV_frg[3] = tTR_rdKV[coord_base + 18]

            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (128,))  # (128, 4)
                tile_dKV_row = tile_dKV_row[None, sub_tile_idx]
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (4,))  # (4, 32)
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load())

        self.reduce_sync_barrier.arrive_and_wait()

    @cute.jit
    def store_dKV_64(
        self,
        dKV_acc: cute.Tensor,
        tdKVtdKV: cute.Tensor,
        rTopkIdx: cute.Tensor,
    ):
        tidx, _, batch_idx = cute.arch.thread_idx()
        token_idx, _, batch_idx = cute.arch.block_idx()
        tidx_in_wg = tidx - self.reduce_warp_id[0] * self.threads_per_warp
        dp_idx = tidx_in_wg % 128
        wg_idx = tidx_in_wg // (4 * self.threads_per_warp)
        num_warp_groups = self.num_reduce_warps // 4

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)),
            self.acc_dtype,
        )

        tiled_t2r_dKV = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV)
        thr_t2r_dKV = tiled_t2r_dKV.get_slice(dp_idx)

        cdKV = cute.make_identity_tensor((self.dKV4_mma_tiler[0], self.dKV4_mma_tiler[1]))
        tTR_cdKV_p = thr_t2r_dKV.partition_D(cdKV)
        tTR_cdKV = self.split_wg(tTR_cdKV_p, num_warp_groups, wg_idx)
        tTR_rdKV = cute.make_rmem_tensor(tTR_cdKV.shape, self.acc_dtype)
        tTR_tdKV = thr_t2r_dKV.partition_S(tdKVtdKV)
        tTR_tdKV = self.split_wg(tTR_tdKV, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dKV, tTR_tdKV, tTR_rdKV)

        cute.arch.fence_view_async_tmem_load()

        # Same compact store as store_dKV: dp_idx//4 indexes into 2-element
        # groups. convert() handles the different unscramble for this 64-tile.
        for i in cutlass.range_constexpr(8):
            coord_base = i * 2 - i % 2

            rdKV_frg = cute.make_rmem_tensor((2,), self.acc_dtype)
            rdKV_frg[0] = tTR_rdKV[coord_base]
            rdKV_frg[1] = tTR_rdKV[coord_base + 2]

            topk_idx = rTopkIdx[i]
            if topk_idx >= 0:
                dKV_row = dKV_acc[None, topk_idx, (0, batch_idx)]
                tile_dKV_row = cute.flat_divide(dKV_row, (64,))  # (64, D/64)
                tile_dKV_row = tile_dKV_row[None, self.head_dim_main // 64]  # last 64-elem tile
                tile_dKV_row = cute.flat_divide(tile_dKV_row, (2,))  # (2, 32)
                cur_dKV_frg = tile_dKV_row[None, dp_idx // 4]
                cute.arch.atomic_add(cur_dKV_frg.iterator.llvm_ptr, rdKV_frg.load())

        self.reduce_sync_barrier.arrive_and_wait()

    @cute.jit
    def store_dQ(
        self,
        tma_atom_dQ: cute.CopyAtom,
        sdQ: cute.Tensor,
        tdQsdQ: cute.Tensor,
        tdQgdQ_mkl: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dp_idx: Int32,
        warp_idx: Int32,
    ):
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(8)),
            self.acc_dtype,
        )

        cdQ = cute.make_identity_tensor(cute.select(self.KdS_mma_tiler, mode=[0, 1]))
        num_warp_groups = self.num_compute_warps // 4

        tiled_t2r_dQ = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_t2r_dQ = tiled_t2r_dQ.get_slice(dp_idx)

        tTR_cdQ = thr_t2r_dQ.partition_D(cdQ)
        tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)

        tTR_tdQ = thr_t2r_dQ.partition_S(tdQtdQ)

        cute.copy(tiled_t2r_dQ, tTR_tdQ, tTR_rdQ)

        tRS_rdQ = self.quantize(tTR_rdQ, 4)

        cute.arch.fence_view_async_tmem_load()

        # ((64,2),(8,8),(1,1))
        thread_layout = cute.make_ordered_layout((128, 64), (0, 1))
        sdQ_slice_tmp = cute.composition(sdQ, thread_layout)
        sdQ_slice = cute.composition(sdQ_slice_tmp[dp_idx, None], cute.make_layout(tTR_cdQ.shape))
        cute.autovec_copy(tRS_rdQ, sdQ_slice)

        self.compute_sync_barrier.arrive_and_wait()

        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )

        self.compute_sync_barrier.arrive_and_wait()

        if warp_idx == self.compute_warp_id[0]:
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ_mkl)

    @cute.jit
    def store_dQ_64(
        self,
        tma_atom_dQ: cute.CopyAtom,
        sdQ: cute.Tensor,
        tdQsdQ: cute.Tensor,
        tdQgdQ_mkl: cute.Tensor,
        tdQtdQ: cute.Tensor,
        dp_idx: Int32,
        wg_idx: Int32,
        warp_idx: Int32,
    ):
        # Use same Ld16x256bOp as store_dKV for the 64-wide TMEM tile
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(2)),
            self.acc_dtype,
        )
        num_warp_groups = self.num_compute_warps // 4

        cdQ = cute.make_identity_tensor(cute.select(self.dQ4_mma_tiler, mode=[0, 1]))

        tiled_t2r_dQ = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ)
        thr_t2r_dQ = tiled_t2r_dQ.get_slice(dp_idx)

        tTR_cdQ_p = thr_t2r_dQ.partition_D(cdQ)
        tTR_cdQ = self.split_wg(tTR_cdQ_p, num_warp_groups, wg_idx)
        tTR_rdQ = cute.make_rmem_tensor(tTR_cdQ.shape, self.acc_dtype)

        tTR_tdQ = thr_t2r_dQ.partition_S(tdQtdQ)
        tTR_tdQ = self.split_wg(tTR_tdQ, num_warp_groups, wg_idx)

        cute.copy(tiled_t2r_dQ, tTR_tdQ, tTR_rdQ)

        cute.arch.fence_view_async_tmem_load()

        # Write dQ4 to smem element-by-element using coord tensor
        for i in cutlass.range_constexpr(cute.size(tTR_rdQ)):
            row = cute.get(tTR_cdQ[i], mode=[0])
            col = cute.get(tTR_cdQ[i], mode=[1])
            sdQ[row, col] = self.element_dtype(tTR_rdQ[i])

        self.compute_sync_barrier.arrive_and_wait()

        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )

        self.compute_sync_barrier.arrive_and_wait()

        if warp_idx == self.compute_warp_id[0]:
            cute.copy(tma_atom_dQ, tdQsdQ, tdQgdQ_mkl)

    @cute.jit
    def quantize(
        self,
        input: cute.Tensor,
        frg_cnt: Int32,
        softmax_scale: Optional[Float32] = None,
    ):
        output = cute.make_rmem_tensor(input.shape, self.element_dtype)
        frg_tile = cute.size(input) // frg_cnt
        t_frg = cute.logical_divide(input, cute.make_layout(frg_cnt))
        output_frg = cute.make_tensor(output.iterator, t_frg.layout)
        for i in cutlass.range(frg_tile, unroll_full=True):
            frg_vec = t_frg[None, i].load()
            if cutlass.const_expr(softmax_scale is not None):
                frg_vec = frg_vec * softmax_scale
            output_frg[None, i].store(frg_vec.to(self.element_dtype))
        return output

    def split_wg(self, t: cute.Tensor, num_warp_groups: int, wg_idx: int):
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 4):
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        t.shape[2],
                        (cute.size(t, mode=[3]) // num_warp_groups, num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        (cute.size(t, mode=[2]) // num_warp_groups, num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 2):
            p = cute.composition(
                t,
                cute.make_layout(
                    (t.shape[0], (cute.size(t, mode=[1]) // num_warp_groups, num_warp_groups))
                ),
            )
            ret = p[None, (None, wg_idx)]
        if cutlass.const_expr(cute.rank(t.layout) == 1):
            p = cute.composition(
                t, cute.make_layout((t.shape[0] // num_warp_groups, num_warp_groups))
            )
            ret = p[None, wg_idx]
        return ret

    def interleave_wg(self, t: cute.Tensor, num_warp_groups: int, wg_idx: int):
        """Interleave split on last mode across warp groups.
        For shape (16, 4) with 2 warp groups: last mode (4) → (2, 2),
        wg0 gets tiles 0,2 and wg1 gets tiles 1,3 → each wg gets (16, 2).
        """
        ret = None
        if cutlass.const_expr(cute.rank(t.layout) == 2):
            p = cute.composition(
                t,
                cute.make_layout(
                    (t.shape[0], (num_warp_groups, cute.size(t, mode=[1]) // num_warp_groups))
                ),
            )
            ret = p[None, (wg_idx, None)]
        if cutlass.const_expr(cute.rank(t.layout) == 3):
            p = cute.composition(
                t,
                cute.make_layout(
                    (
                        t.shape[0],
                        t.shape[1],
                        (num_warp_groups, cute.size(t, mode=[2]) // num_warp_groups),
                    )
                ),
            )
            ret = p[None, None, (wg_idx, None)]
        return ret

    @cute.jit
    def get_tmem_tensor(
        self,
        QK_tiled_mma: cute.TiledMma,
        dOV_tiled_mma: cute.TiledMma,
        QdS_tiled_mma: cute.TiledMma,
        KdS_tiled_mma: cute.TiledMma,
        dKV4_tiled_mma: Optional[cute.TiledMma],
        dQ4_tiled_mma: Optional[cute.TiledMma],
        tmem_ptr_base: cute.Pointer,
    ):
        tStS_shape = QK_tiled_mma.partition_shape_C(cute.select(self.QK_mma_tiler, mode=[0, 1]))
        tStS = QK_tiled_mma.make_fragment_C(tStS_shape)
        tStS = cute.make_tensor(tmem_ptr_base + self.tmem_S_offset, tStS.layout)

        tdPtdP_shape = dOV_tiled_mma.partition_shape_C(
            cute.select(self.dOV_mma_tiler, mode=[0, 1])
        )
        tdPtdP = dOV_tiled_mma.make_fragment_C(tdPtdP_shape)
        lane_id = 16 << 16
        # lane_id = 0
        tdPtdP = cute.make_tensor(tmem_ptr_base + self.tmem_dP_offset + lane_id, tdPtdP.layout)

        tdKVtdKV_shape = QdS_tiled_mma.partition_shape_C(
            cute.select(self.QdS_mma_tiler, mode=[0, 1])
        )
        tdKVtdKV_base = QdS_tiled_mma.make_fragment_C(tdKVtdKV_shape)
        tdKVtdKV0 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV0_offset, tdKVtdKV_base.layout)
        tdKVtdKV1 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV1_offset, tdKVtdKV_base.layout)
        tdKVtdKV2 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV2_offset, tdKVtdKV_base.layout)
        tdKVtdKV3 = cute.make_tensor(tmem_ptr_base + self.tmem_dKV3_offset, tdKVtdKV_base.layout)

        tdQtdQ_shape = KdS_tiled_mma.partition_shape_C(
            cute.select(self.KdS_mma_tiler, mode=[0, 1])
        )
        tdQtdQ_base = KdS_tiled_mma.make_fragment_C(tdQtdQ_shape)
        tdQtdQ0 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ0_offset, tdQtdQ_base.layout)
        tdQtdQ1 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ1_offset, tdQtdQ_base.layout)
        tdQtdQ2 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ2_offset, tdQtdQ_base.layout)
        tdQtdQ3 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ3_offset, tdQtdQ_base.layout)

        if cutlass.const_expr(not self.same_hdim_kv):
            tdKVtdKV4_shape = dKV4_tiled_mma.partition_shape_C(
                cute.select(self.dKV4_mma_tiler, mode=[0, 1])
            )
            tdKVtdKV4_base = dKV4_tiled_mma.make_fragment_C(tdKVtdKV4_shape)
            tdKVtdKV4 = cute.make_tensor(
                tmem_ptr_base + self.tmem_dKV4_offset, tdKVtdKV4_base.layout
            )

            tdQtdQ4_shape = dQ4_tiled_mma.partition_shape_C(
                cute.select(self.dQ4_mma_tiler, mode=[0, 1])
            )
            tdQtdQ4_base = dQ4_tiled_mma.make_fragment_C(tdQtdQ4_shape)
            tdQtdQ4 = cute.make_tensor(tmem_ptr_base + self.tmem_dQ4_offset, tdQtdQ4_base.layout)
        else:
            tdKVtdKV4 = None
            tdQtdQ4 = None

        return (
            tStS,
            tdPtdP,
            tdKVtdKV0,
            tdKVtdKV1,
            tdKVtdKV2,
            tdKVtdKV3,
            tdQtdQ0,
            tdQtdQ1,
            tdQtdQ2,
            tdQtdQ3,
            tdKVtdKV4,
            tdQtdQ4,
        )

    def make_and_init_load_mma_QdO_pipeline(self, load_mma_QdO_mbar_ptr):
        load_mma_QdO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_warp_id])
        )
        load_mma_QdO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_mma_QdO_mbar_ptr,
            num_stages=self.load_mma_QdO_stage,
            producer_group=load_mma_QdO_producer_group,
            consumer_group=load_mma_QdO_consumer_group,
            tx_count=self.tma_copy_QdO_bytes,
            defer_sync=True,
        )

    def make_and_init_load_mma_K_pipeline(self, load_mma_K_mbar_ptr):
        load_mma_K_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.threads_per_warp * self.num_load_KV_warps
        )
        load_mma_K_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=load_mma_K_mbar_ptr,
            num_stages=self.load_mma_K_stage,
            producer_group=load_mma_K_producer_group,
            consumer_group=load_mma_K_consumer_group,
            defer_sync=True,
        )

    def make_and_init_load_compute_LSE_pipeline(self, load_compute_lse_mbar_ptr):
        load_compute_lse_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_lse_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_lse_mbar_ptr,
            num_stages=self.load_compute_LSE_stage,
            producer_group=load_compute_lse_producer_group,
            consumer_group=load_compute_lse_consumer_group,
            defer_sync=True,
        )

    def make_and_init_load_compute_sum_OdO_pipeline(self, load_compute_sum_OdO_mbar_ptr):
        load_compute_sum_OdO_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp,
        )
        load_compute_sum_OdO_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineCpAsync.create(
            barrier_storage=load_compute_sum_OdO_mbar_ptr,
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=load_compute_sum_OdO_producer_group,
            consumer_group=load_compute_sum_OdO_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_compute_S_pipeline(self, mma_compute_S_mbar_ptr):
        mma_compute_S_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        mma_compute_S_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_S_mbar_ptr,
            num_stages=self.mma_compute_S_stage,
            producer_group=mma_compute_S_producer_group,
            consumer_group=mma_compute_S_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_compute_dQ_pipeline(self, mma_compute_dQ_mbar_ptr):
        mma_compute_dQ_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        mma_compute_dQ_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dQ_mbar_ptr,
            num_stages=self.mma_compute_dQ_stage,
            producer_group=mma_compute_dQ_producer_group,
            consumer_group=mma_compute_dQ_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_compute_dP_pipeline(self, mma_compute_dP_mbar_ptr):
        mma_compute_dP_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        mma_compute_dP_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_compute_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_compute_dP_mbar_ptr,
            num_stages=self.mma_compute_dP_stage,
            producer_group=mma_compute_dP_producer_group,
            consumer_group=mma_compute_dP_consumer_group,
            defer_sync=True,
        )

    def make_and_init_compute_mma_P_pipeline(self, compute_mma_P_mbar_ptr):
        compute_mma_P_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_P_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_P_mbar_ptr,
            num_stages=self.compute_mma_P_stage,
            producer_group=compute_mma_P_producer_group,
            consumer_group=compute_mma_P_consumer_group,
            defer_sync=True,
        )

    def make_and_init_compute_mma_dS_pipeline(self, compute_mma_dS_mbar_ptr):
        compute_mma_dS_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        compute_mma_dS_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            len([self.mma_warp_id]),
        )
        return pipeline.PipelineAsyncUmma.create(
            barrier_storage=compute_mma_dS_mbar_ptr,
            num_stages=self.compute_mma_dS_stage,
            producer_group=compute_mma_dS_producer_group,
            consumer_group=compute_mma_dS_consumer_group,
            defer_sync=True,
        )

    def make_and_init_mma_reduce_dKV_pipeline(self, mma_reduce_dKV_mbar_ptr):
        mma_reduce_dKV_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        mma_reduce_dKV_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * self.num_reduce_warps,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_reduce_dKV_mbar_ptr,
            num_stages=self.mma_reduce_dKV_stage,
            producer_group=mma_reduce_dKV_producer_group,
            consumer_group=mma_reduce_dKV_consumer_group,
            defer_sync=True,
        )

    def make_and_init_compute_tmastore_dQ_pipeline(self):
        compute_tmastore_dQ_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.threads_per_warp,
        )
        return pipeline.PipelineTmaStore.create(
            num_stages=self.compute_tmastore_dQ_stage,
            producer_group=compute_tmastore_dQ_producer_group,
        )


# The code below is Attention Gym's local PyTorch launcher. It replaces the
# cudnn.deepseek_sparse_attention APIBase/interface layers; the kernel above is vendored.
from collections import OrderedDict

import torch
from cutlass.cute.runtime import from_dlpack


_TORCH_TO_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}


def _to_cute_tensor(
    tensor: torch.Tensor,
    *,
    assumed_align: int = 16,
    divisibility: int | None = None,
) -> cute.Tensor:
    result = from_dlpack(
        tensor.detach(), assumed_align=assumed_align, enable_tvm_ffi=True
    ).mark_layout_dynamic(leading_dim=tensor.ndim - 1)
    if divisibility is not None:
        result = result.mark_compact_shape_dynamic(
            mode=tensor.ndim - 1,
            stride_order=tensor.dim_order(),
            divisibility=divisibility,
        )
    return result


def sparse_attention_backward_wrapper(
    q: torch.Tensor,
    kv: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float | None = None,
    topk_length: torch.Tensor | None = None,
    dq: torch.Tensor | None = None,
    dkv: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Launch the vendored SM100 DSA backward and return its three gradients."""
    if torch.cuda.get_device_capability(q.device) != (10, 0):
        raise RuntimeError("The vendored DSA backward targets SM100 exclusively.")
    total_q, num_heads, head_dim = q.shape
    total_kv = kv.shape[0]
    head_dim_v = 512 if head_dim == 576 else head_dim
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("The vendored DSA backward requires FP16 or BF16 inputs.")
    if not (q.dtype == kv.dtype == out.dtype == dout.dtype):
        raise TypeError("Q, KV, output, and output gradient must have the same dtype.")
    if lse.dtype != torch.float32 or attn_sink.dtype != torch.float32:
        raise TypeError("LSE and attention sink must be FP32.")
    if topk_idxs.dtype != torch.int32:
        raise TypeError("Sparse indices must be INT32.")

    q, kv, out, dout, lse = (tensor.contiguous() for tensor in (q, kv, out, dout, lse))
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if dq is None:
        dq = torch.empty_like(q)
    if dkv is None:
        dkv = torch.zeros(total_kv, head_dim, dtype=kv.dtype, device=kv.device)
    else:
        dkv.zero_()
    d_sink = torch.zeros_like(attn_sink)

    block_tile = 64
    batch_size = 1
    workspace_lse_odo = torch.zeros(
        *FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(
            total_q, head_dim, num_heads, batch_size, cutlass.Float32
        ),
        dtype=torch.uint8,
        device=q.device,
    )
    workspace_dkv = torch.zeros(
        *FlashAttentionDSABackwardSm100._get_workspace_size_dKV(
            total_kv, head_dim, batch_size, cutlass.Float32
        ),
        dtype=torch.uint8,
        device=q.device,
    )
    problem_shape = (total_q, total_kv, head_dim, (num_heads, batch_size))
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    dtype = _TORCH_TO_CUTE_DTYPE[q.dtype]
    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        num_heads,
        block_tile,
        topk_length is not None,
    )
    compile_cache = sparse_attention_backward_wrapper.compile_cache
    compiled = compile_cache.get(compile_key)
    if compiled is None:
        kernel = FlashAttentionDSABackwardSm100(head_dim, head_dim_v, block_tile)
        compiled = cute.compile(
            kernel,
            problem_shape,
            _to_cute_tensor(q, divisibility=head_dim),
            _to_cute_tensor(kv, divisibility=head_dim),
            _to_cute_tensor(out, divisibility=head_dim_v),
            _to_cute_tensor(dout, divisibility=head_dim_v),
            _to_cute_tensor(lse, assumed_align=4),
            _to_cute_tensor(attn_sink),
            _to_cute_tensor(topk_idxs),
            _to_cute_tensor(topk_length) if topk_length is not None else None,
            _to_cute_tensor(dq, divisibility=head_dim),
            _to_cute_tensor(dkv, divisibility=head_dim),
            _to_cute_tensor(d_sink),
            _to_cute_tensor(workspace_lse_odo),
            _to_cute_tensor(workspace_dkv),
            softmax_scale,
            stream,
            options="--enable-tvm-ffi",
        )
        compile_cache[compile_key] = compiled
        if len(compile_cache) > 64:
            compile_cache.popitem(last=False)
    else:
        compile_cache.move_to_end(compile_key)

    compiled(
        problem_shape,
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_idxs,
        topk_length,
        dq,
        dkv,
        d_sink,
        workspace_lse_odo,
        workspace_dkv,
        softmax_scale,
        stream,
    )
    return {"dq": dq, "dkv": dkv, "d_sink": d_sink}


sparse_attention_backward_wrapper.compile_cache = OrderedDict()
