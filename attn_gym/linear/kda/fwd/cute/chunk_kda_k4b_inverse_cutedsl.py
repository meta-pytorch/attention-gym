# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CuTe DSL K4b: 4×4 block lower-triangular matrix inverse.

Grid: (total_chunks, H, 1) — one CTA per (chunk, head).
128 threads (4 warps):
    Phase 1: Per-warp OD register load + parallel forward substitution
    Phase 2a: Warps 0-2 parallel Schur L1 (Ai10, Ai21, Ai32), Warp 3 stores diags
    Phase 2b+c: Warp 0 interleaved Schur L2+L3 with B-fragment reuse
    Phase 2b: Warp 1 Schur L2 (Ai31) with B-fragment reuse

Input:  mAkkOD (fp32, off-diag from K3b), mAkkd (fp32, diagonal Akk blocks)
Output: mAkk (bf16, full 4×4 block inverse)

Optimizations:
  - 4 warps (128 threads) with parallel Schur complement (9-GEMM critical path vs 16)
  - OD blocks in registers (zero-latency staging reads, 6KB SMEM saved)
  - B-fragment reuse: GEMMs 2→3→6' share B=sAi0_T, GEMMs 4→7 share B=Ai10
  - Parallel forward substitution (4 blocks simultaneously)
  - Overlapped diagonal stores (warp 3 stores while warps 0-2 compute)
  - Flat padded SMEM layout (no swizzle) for reduced address computation
"""

import cutlass
from cutlass import Int32, cute
from cutlass.cute.nvgpu import warp

from attn_gym.linear.kda.fwd.cute.chunk_schedule import ChunkSchedule
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import load_ragged_chunk_work


class ChunkKDAFwdK4bInverseCuteDSL:
    WARP_SIZE = 32

    def __init__(
        self,
        BC: int = 16,
        chunk_size: int = 64,
        num_subchunks: int = 4,
        fwd_sub_mode: str = "cute_recurrence",
        skip_fwd_sub: bool = False,
        schedule: ChunkSchedule = ChunkSchedule.DENSE,
    ):
        assert num_subchunks == 4, (
            f"ChunkKDAFwdK4bInverseCuteDSL only supports four subchunks, got {num_subchunks}"
        )
        assert chunk_size == num_subchunks * BC, (
            f"chunk_size must equal num_subchunks * BC, got {chunk_size} and "
            f"{num_subchunks} * {BC}"
        )
        self.BC = BC
        self.BT = chunk_size
        self.num_offdiag_blocks = num_subchunks * (num_subchunks - 1) // 2
        self.num_threads = 128
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (1, 1, 1)
        self.fwd_sub_mode = "skip" if skip_fwd_sub else fwd_sub_mode
        self.schedule = schedule

    @cute.jit
    def _fwd_sub_block(self, mAkkd, sAi, block_start, tidx, akkd_col_off, valid_rows):
        """Forward substitution: inverse of (I - L) for a single BC×BC diagonal block."""
        if cutlass.const_expr(self.fwd_sub_mode == "preinverted"):
            for k in cutlass.range_constexpr(self.BC * self.BC // 32):
                linear_idx = tidx + k * 32
                row = linear_idx // self.BC
                col = linear_idx % self.BC
                if row < valid_rows:
                    sAi[row, col] = self._sai_dtype(mAkkd[block_start + row, akkd_col_off + col])
                else:
                    sAi[row, col] = self._sai_dtype(0.0)
            cute.arch.sync_warp()
            return

        for k in cutlass.range_constexpr(self.BC * self.BC // 32):
            linear_idx = tidx + k * 32
            row = linear_idx // self.BC
            col = linear_idx % self.BC
            if row < valid_rows:
                val = mAkkd[block_start + row, akkd_col_off + col]
                if row > col:
                    sAi[row, col] = -val
                else:
                    sAi[row, col] = 0.0
            else:
                sAi[row, col] = 0.0
        cute.arch.sync_warp()

        if cutlass.const_expr(self.fwd_sub_mode == "cute_recurrence"):
            for i in cutlass.range_constexpr(2, self.BC):
                if tidx < self.BC:
                    my_col = tidx
                    acc = sAi[i, my_col]
                    for j in cutlass.range_constexpr(i):
                        acc = acc + sAi[i, j] * sAi[j, my_col]
                    sAi[i, my_col] = acc
                cute.arch.sync_warp()

        if tidx < self.BC:
            sAi[tidx, tidx] = sAi[tidx, tidx] + 1.0
        cute.arch.sync_warp()

    @cute.jit
    def __call__(
        self,
        mAkkOD: cute.Tensor,
        mAkkd: cute.Tensor,
        mAkk: cute.Tensor,
        H: int,
        total_chunks: int,
        cu_seqlens: cute.Tensor | None,
        chunk_indices: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        stream,
    ):
        self._dtype: type[cutlass.Numeric] = mAkk.element_type
        self._sai_dtype: type[cutlass.Numeric] = cutlass.Float32
        self._sai_stride = self.BC + 1
        if cutlass.const_expr(self.fwd_sub_mode == "preinverted"):
            self._sai_dtype = self._dtype
            self._sai_stride = self.BC

        if cutlass.const_expr(self.schedule is ChunkSchedule.ALIGNED):
            NT = cute.size(chunk_indices, mode=[0]) // 2
            grid = (NT, H, 1)
        else:
            grid = (total_chunks, H, 1)

        sSchurA_layout = cute.make_layout((self.BC, self.BC), stride=(self.BC + 8, 1))
        sAi_layout = cute.make_layout((self.BC, self.BC), stride=(self._sai_stride, 1))
        sSchurB_layout = cute.make_layout((self.BC, self.BC), stride=(self.BC + 8, 1))

        @cute.struct
        class SharedStorage:
            work: cute.struct.MemRange[Int32, 3]
            sAi0: cute.struct.Align[
                cute.struct.MemRange[self._sai_dtype, self.BC * self._sai_stride], 128
            ]
            sAi1: cute.struct.Align[
                cute.struct.MemRange[self._sai_dtype, self.BC * self._sai_stride], 128
            ]
            sAi2: cute.struct.Align[
                cute.struct.MemRange[self._sai_dtype, self.BC * self._sai_stride], 128
            ]
            sAi3: cute.struct.Align[
                cute.struct.MemRange[self._sai_dtype, self.BC * self._sai_stride], 128
            ]
            sSchurA0: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self.BC * (self.BC + 8)], 128
            ]
            sSchurB0: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self.BC * (self.BC + 8)], 128
            ]
            sSchurA1: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self.BC * (self.BC + 8)], 128
            ]
            sSchurB1: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self.BC * (self.BC + 8)], 128
            ]
            sSchurA2: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self.BC * (self.BC + 8)], 128
            ]
            sSchurB2: cute.struct.Align[
                cute.struct.MemRange[self._dtype, self.BC * (self.BC + 8)], 128
            ]

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, self.mma_inst_shape),
            self.atom_layout_mnk,
            permutation_mnk=(
                self.atom_layout_mnk[0] * self.mma_inst_shape[0],
                self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
                self.mma_inst_shape[2],
            ),
        )

        smem_copy_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            ),
            tiled_mma,
        )
        smem_copy_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            ),
            tiled_mma,
        )

        self.kernel(
            mAkkOD,
            mAkkd,
            mAkk,
            cu_seqlens,
            chunk_indices,
            chunk_offsets,
            sSchurA_layout,
            sAi_layout,
            sSchurB_layout,
            smem_copy_A,
            smem_copy_B,
            tiled_mma,
            SharedStorage,
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mAkkOD: cute.Tensor,
        mAkkd: cute.Tensor,
        mAkk: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        chunk_indices: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        sSchurA_layout: cute.Layout,
        sAi_layout: cute.Layout,
        sSchurB_layout: cute.Layout,
        smem_copy_A: cute.TiledCopy,
        smem_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk_idx, head_idx, _ = cute.arch.block_idx()
        warp_idx = tidx // self.WARP_SIZE
        lane_idx = tidx % self.WARP_SIZE

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        work = storage.work.get_tensor(cute.make_layout(3))
        sAi0 = storage.sAi0.get_tensor(sAi_layout)
        sAi1 = storage.sAi1.get_tensor(sAi_layout)
        sAi2 = storage.sAi2.get_tensor(sAi_layout)
        sAi3 = storage.sAi3.get_tensor(sAi_layout)
        sSchurA0 = storage.sSchurA0.get_tensor(sSchurA_layout)
        sSchurB0 = storage.sSchurB0.get_tensor(sSchurB_layout)
        sSchurA1 = storage.sSchurA1.get_tensor(sSchurA_layout)
        sSchurB1 = storage.sSchurB1.get_tensor(sSchurB_layout)
        sSchurA2 = storage.sSchurA2.get_tensor(sSchurA_layout)
        sSchurB2 = storage.sSchurB2.get_tensor(sSchurB_layout)

        if cutlass.const_expr(self.schedule is ChunkSchedule.RAGGED):
            if tidx == 0:
                num_sequences = Int32(cute.size(chunk_offsets)) - 1
                active_chunks = Int32(chunk_offsets[num_sequences])
                work[0] = Int32(0)
                work[1] = Int32(0)
                work[2] = Int32(0)
                if chunk_idx < active_chunks:
                    _, _, token_start, valid_tokens = load_ragged_chunk_work(
                        cu_seqlens,
                        chunk_offsets,
                        Int32(chunk_idx),
                        Int32(self.BT),
                    )
                    work[0] = Int32(1)
                    work[1] = token_start
                    work[2] = token_start + valid_tokens
            cute.arch.sync_threads()
            is_active = work[0]
            chunk_base = work[1]
            eos = work[2]
        elif cutlass.const_expr(self.schedule is ChunkSchedule.ALIGNED):
            i_n = chunk_indices[chunk_idx * 2]
            i_t = chunk_indices[chunk_idx * 2 + 1]
            bos = cu_seqlens[i_n]
            eos = cu_seqlens[i_n + 1]
            chunk_base = bos + i_t * self.BT
            is_active = Int32(1)
        else:
            chunk_base = chunk_idx * self.BT
            eos = cute.size(mAkk, mode=[0])
            is_active = Int32(1)

        i_tc0 = chunk_base
        i_tc1 = chunk_base + self.BC
        i_tc2 = chunk_base + 2 * self.BC
        i_tc3 = chunk_base + 3 * self.BC

        h_akkd_col = head_idx * self.BC
        h_akk_col = head_idx * self.BT

        vr0 = cutlass.min(cutlass.max(eos - i_tc0, 0), self.BC)
        vr1 = cutlass.min(cutlass.max(eos - i_tc1, 0), self.BC)
        vr2 = cutlass.min(cutlass.max(eos - i_tc2, 0), self.BC)
        vr3 = cutlass.min(cutlass.max(eos - i_tc3, 0), self.BC)

        thr_mma = tiled_mma.get_slice(lane_idx)
        thrA = smem_copy_A.get_slice(lane_idx)
        thrB = smem_copy_B.get_slice(lane_idx)

        tCsA0 = thrA.partition_S(sSchurA0)
        tCsB0 = thrB.partition_S(sSchurB0)
        tCsA1 = thrA.partition_S(sSchurA1)
        tCsB1 = thrB.partition_S(sSchurB1)
        tCsA2 = thrA.partition_S(sSchurA2)
        tCsB2 = thrB.partition_S(sSchurB2)

        _tA = thr_mma.partition_A(sSchurA0)
        _tB = thr_mma.partition_B(sSchurB0)
        tCrA = thr_mma.make_fragment_A(_tA)
        tCrB = thr_mma.make_fragment_B(_tB)
        tCrAc = thrA.retile(tCrA)
        tCrBc = thrB.retile(tCrB)

        acc_shape = thr_mma.partition_shape_C((self.BC, self.BC))
        cC = cute.make_identity_tensor((self.BC, self.BC))
        tCcC = thr_mma.partition_C(cC)

        acc_tmp = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_res1 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_res2 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_res3 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_od0 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_od1 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_od2 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_od3 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_od4 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_od5 = cute.make_rmem_tensor(acc_shape, cutlass.Float32)

        # ══════════════════════════════════════════════════════════
        # PHASE 1: Per-warp OD loading + parallel forward substitution
        # ══════════════════════════════════════════════════════════
        od_base_row = chunk_idx * self.num_offdiag_blocks

        if is_active and warp_idx == 0:
            for i in cutlass.range_constexpr(cute.size(acc_od0)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                rc = head_idx * self.BC * self.BC + row * self.BC + col
                acc_od0[i] = mAkkOD[od_base_row + 0, rc]
                acc_od1[i] = mAkkOD[od_base_row + 1, rc]
                acc_od2[i] = mAkkOD[od_base_row + 2, rc]
                acc_od3[i] = mAkkOD[od_base_row + 3, rc]
                acc_od4[i] = mAkkOD[od_base_row + 4, rc]
                acc_od5[i] = mAkkOD[od_base_row + 5, rc]
            self._fwd_sub_block(mAkkd, sAi0, i_tc0, lane_idx, h_akkd_col, vr0)

        if is_active and warp_idx == 1:
            for i in cutlass.range_constexpr(cute.size(acc_od0)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                rc = head_idx * self.BC * self.BC + row * self.BC + col
                acc_od2[i] = mAkkOD[od_base_row + 2, rc]
                acc_od4[i] = mAkkOD[od_base_row + 4, rc]
                acc_od5[i] = mAkkOD[od_base_row + 5, rc]
            self._fwd_sub_block(mAkkd, sAi1, i_tc1, lane_idx, h_akkd_col, vr1)

        if is_active and warp_idx == 2:
            for i in cutlass.range_constexpr(cute.size(acc_od0)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                rc = head_idx * self.BC * self.BC + row * self.BC + col
                acc_od5[i] = mAkkOD[od_base_row + 5, rc]
            self._fwd_sub_block(mAkkd, sAi2, i_tc2, lane_idx, h_akkd_col, vr2)

        if is_active and warp_idx == 3:
            self._fwd_sub_block(mAkkd, sAi3, i_tc3, lane_idx, h_akkd_col, vr3)

        cute.arch.barrier(barrier_id=0, number_of_threads=128)

        # ══════════════════════════════════════════════════════════
        # PHASE 2: Parallel Schur Complement + Overlapped Stores
        # ══════════════════════════════════════════════════════════

        # ── WARP 0: 9 GEMMs (reordered for B-fragment reuse) ──
        if is_active and warp_idx == 0:
            # GEMM 1: tmp = Ai11 @ Akk10
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA0[c0[0], c0[1]] = self._dtype(sAi1[c0[0], c0[1]])
                sSchurA0[c1[0], c1[1]] = self._dtype(sAi1[c1[0], c1[1]])
                sSchurB0[c0[1], c0[0]] = self._dtype(acc_od0[k * 2])
                sSchurB0[c1[1], c1[0]] = self._dtype(acc_od0[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB0[None, None, 0], tCrBc[None, None, 0])
            acc_tmp.fill(0.0)
            cute.gemm(tiled_mma, acc_tmp, tCrA[None, None, 0], tCrB[None, None, 0], acc_tmp)

            # GEMM 2: Ai10 = -(tmp @ Ai00)  [loads B=sAi0_T]
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA0[c0[0], c0[1]] = self._dtype(acc_tmp[k * 2])
                sSchurA0[c1[0], c1[1]] = self._dtype(acc_tmp[k * 2 + 1])
                sSchurB0[c0[0], c0[1]] = self._dtype(sAi0[c0[1], c0[0]])
                sSchurB0[c1[0], c1[1]] = self._dtype(sAi0[c1[1], c1[0]])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB0[None, None, 0], tCrBc[None, None, 0])
            acc_res1.fill(0.0)
            cute.gemm(tiled_mma, acc_res1, tCrA[None, None, 0], tCrB[None, None, 0], acc_res1)
            for i in cutlass.range_constexpr(cute.size(acc_res1)):
                acc_res1[i] = -acc_res1[i]

            # GEMM 3: tmp = Akk20 @ Ai00  [REUSE B]
            for i in cutlass.range_constexpr(cute.size(acc_tmp)):
                coord = tCcC[i]
                sSchurA0[coord[0], coord[1]] = self._dtype(acc_od1[i])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            acc_tmp.fill(0.0)
            cute.gemm(tiled_mma, acc_tmp, tCrA[None, None, 0], tCrB[None, None, 0], acc_tmp)

            # GEMM 6': tmp_2c = Akk30 @ Ai00  [REUSE B]
            for i in cutlass.range_constexpr(cute.size(acc_res3)):
                coord = tCcC[i]
                sSchurA0[coord[0], coord[1]] = self._dtype(acc_od3[i])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            acc_res3.fill(0.0)
            cute.gemm(tiled_mma, acc_res3, tCrA[None, None, 0], tCrB[None, None, 0], acc_res3)

            # GEMM 4: tmp += Akk21 @ Ai10  [loads B=Ai10, paired A stores]
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA0[c0[0], c0[1]] = self._dtype(acc_od2[k * 2])
                sSchurA0[c1[0], c1[1]] = self._dtype(acc_od2[k * 2 + 1])
                sSchurB0[c0[1], c0[0]] = self._dtype(acc_res1[k * 2])
                sSchurB0[c1[1], c1[0]] = self._dtype(acc_res1[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB0[None, None, 0], tCrBc[None, None, 0])
            cute.gemm(tiled_mma, acc_tmp, tCrA[None, None, 0], tCrB[None, None, 0], acc_tmp)

            # GEMM 7: tmp_2c += Akk31 @ Ai10  [REUSE B]
            for i in cutlass.range_constexpr(cute.size(acc_res3)):
                coord = tCcC[i]
                sSchurA0[coord[0], coord[1]] = self._dtype(acc_od4[i])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            cute.gemm(tiled_mma, acc_res3, tCrA[None, None, 0], tCrB[None, None, 0], acc_res3)

            # GEMM 5: Ai20 = -(Ai22 @ tmp)
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA0[c0[0], c0[1]] = self._dtype(sAi2[c0[0], c0[1]])
                sSchurA0[c1[0], c1[1]] = self._dtype(sAi2[c1[0], c1[1]])
                sSchurB0[c0[1], c0[0]] = self._dtype(acc_tmp[k * 2])
                sSchurB0[c1[1], c1[0]] = self._dtype(acc_tmp[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB0[None, None, 0], tCrBc[None, None, 0])
            acc_res2.fill(0.0)
            cute.gemm(tiled_mma, acc_res2, tCrA[None, None, 0], tCrB[None, None, 0], acc_res2)
            for i in cutlass.range_constexpr(cute.size(acc_res2)):
                acc_res2[i] = -acc_res2[i]

            # GEMM 8: tmp_2c += Akk32 @ Ai20
            for k in cutlass.range_constexpr(cute.size(acc_res3) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA0[c0[0], c0[1]] = self._dtype(acc_od5[k * 2])
                sSchurA0[c1[0], c1[1]] = self._dtype(acc_od5[k * 2 + 1])
                sSchurB0[c0[1], c0[0]] = self._dtype(acc_res2[k * 2])
                sSchurB0[c1[1], c1[0]] = self._dtype(acc_res2[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB0[None, None, 0], tCrBc[None, None, 0])
            cute.gemm(tiled_mma, acc_res3, tCrA[None, None, 0], tCrB[None, None, 0], acc_res3)

            # GEMM 9: Ai30 = -(Ai33 @ tmp_2c)
            for k in cutlass.range_constexpr(cute.size(acc_res3) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA0[c0[0], c0[1]] = self._dtype(sAi3[c0[0], c0[1]])
                sSchurA0[c1[0], c1[1]] = self._dtype(sAi3[c1[0], c1[1]])
                sSchurB0[c0[1], c0[0]] = self._dtype(acc_res3[k * 2])
                sSchurB0[c1[1], c1[0]] = self._dtype(acc_res3[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA0[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB0[None, None, 0], tCrBc[None, None, 0])
            acc_tmp.fill(0.0)
            cute.gemm(tiled_mma, acc_tmp, tCrA[None, None, 0], tCrB[None, None, 0], acc_tmp)
            for i in cutlass.range_constexpr(cute.size(acc_tmp)):
                acc_tmp[i] = -acc_tmp[i]

            # Store: Ai10, Ai20, Ai30
            for i in cutlass.range_constexpr(cute.size(acc_res1)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                if i_tc1 + row < eos:
                    mAkk[i_tc1 + row, h_akk_col + 0 * self.BC + col] = self._dtype(acc_res1[i])
                if i_tc2 + row < eos:
                    mAkk[i_tc2 + row, h_akk_col + 0 * self.BC + col] = self._dtype(acc_res2[i])
                if i_tc3 + row < eos:
                    mAkk[i_tc3 + row, h_akk_col + 0 * self.BC + col] = self._dtype(acc_tmp[i])

        # ── WARP 1: 5 GEMMs with B-reuse on sAi1_T ──
        if is_active and warp_idx == 1:
            # GEMM 1: tmp = Ai22 @ Akk21
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA1[c0[0], c0[1]] = self._dtype(sAi2[c0[0], c0[1]])
                sSchurA1[c1[0], c1[1]] = self._dtype(sAi2[c1[0], c1[1]])
                sSchurB1[c0[1], c0[0]] = self._dtype(acc_od2[k * 2])
                sSchurB1[c1[1], c1[0]] = self._dtype(acc_od2[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA1[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB1[None, None, 0], tCrBc[None, None, 0])
            acc_tmp.fill(0.0)
            cute.gemm(tiled_mma, acc_tmp, tCrA[None, None, 0], tCrB[None, None, 0], acc_tmp)

            # GEMM 2: Ai21 = -(tmp @ Ai11)  [loads B=sAi1_T]
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA1[c0[0], c0[1]] = self._dtype(acc_tmp[k * 2])
                sSchurA1[c1[0], c1[1]] = self._dtype(acc_tmp[k * 2 + 1])
                sSchurB1[c0[0], c0[1]] = self._dtype(sAi1[c0[1], c0[0]])
                sSchurB1[c1[0], c1[1]] = self._dtype(sAi1[c1[1], c1[0]])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA1[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB1[None, None, 0], tCrBc[None, None, 0])
            acc_res1.fill(0.0)
            cute.gemm(tiled_mma, acc_res1, tCrA[None, None, 0], tCrB[None, None, 0], acc_res1)
            for i in cutlass.range_constexpr(cute.size(acc_res1)):
                acc_res1[i] = -acc_res1[i]

            # GEMM 3: tmp = Akk31 @ Ai11  [REUSE B]
            for i in cutlass.range_constexpr(cute.size(acc_tmp)):
                coord = tCcC[i]
                sSchurA1[coord[0], coord[1]] = self._dtype(acc_od4[i])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA1[None, None, 0], tCrAc[None, None, 0])
            acc_tmp.fill(0.0)
            cute.gemm(tiled_mma, acc_tmp, tCrA[None, None, 0], tCrB[None, None, 0], acc_tmp)

            # GEMM 4: tmp += Akk32 @ Ai21
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA1[c0[0], c0[1]] = self._dtype(acc_od5[k * 2])
                sSchurA1[c1[0], c1[1]] = self._dtype(acc_od5[k * 2 + 1])
                sSchurB1[c0[1], c0[0]] = self._dtype(acc_res1[k * 2])
                sSchurB1[c1[1], c1[0]] = self._dtype(acc_res1[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA1[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB1[None, None, 0], tCrBc[None, None, 0])
            cute.gemm(tiled_mma, acc_tmp, tCrA[None, None, 0], tCrB[None, None, 0], acc_tmp)

            # GEMM 5: Ai31 = -(Ai33 @ tmp)
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA1[c0[0], c0[1]] = self._dtype(sAi3[c0[0], c0[1]])
                sSchurA1[c1[0], c1[1]] = self._dtype(sAi3[c1[0], c1[1]])
                sSchurB1[c0[1], c0[0]] = self._dtype(acc_tmp[k * 2])
                sSchurB1[c1[1], c1[0]] = self._dtype(acc_tmp[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA1[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB1[None, None, 0], tCrBc[None, None, 0])
            acc_res2.fill(0.0)
            cute.gemm(tiled_mma, acc_res2, tCrA[None, None, 0], tCrB[None, None, 0], acc_res2)
            for i in cutlass.range_constexpr(cute.size(acc_res2)):
                acc_res2[i] = -acc_res2[i]

            # Store: Ai21, Ai31
            for i in cutlass.range_constexpr(cute.size(acc_res1)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                if i_tc2 + row < eos:
                    mAkk[i_tc2 + row, h_akk_col + 1 * self.BC + col] = self._dtype(acc_res1[i])
                if i_tc3 + row < eos:
                    mAkk[i_tc3 + row, h_akk_col + 1 * self.BC + col] = self._dtype(acc_res2[i])

        # ── WARP 2: 2 GEMMs + store Ai32 ──
        if is_active and warp_idx == 2:
            # GEMM 1: tmp = Ai33 @ Akk32
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA2[c0[0], c0[1]] = self._dtype(sAi3[c0[0], c0[1]])
                sSchurA2[c1[0], c1[1]] = self._dtype(sAi3[c1[0], c1[1]])
                sSchurB2[c0[1], c0[0]] = self._dtype(acc_od5[k * 2])
                sSchurB2[c1[1], c1[0]] = self._dtype(acc_od5[k * 2 + 1])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA2[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB2[None, None, 0], tCrBc[None, None, 0])
            acc_tmp.fill(0.0)
            cute.gemm(tiled_mma, acc_tmp, tCrA[None, None, 0], tCrB[None, None, 0], acc_tmp)

            # GEMM 2: Ai32 = -(tmp @ Ai22)
            for k in cutlass.range_constexpr(cute.size(acc_tmp) // 2):
                c0 = tCcC[k * 2]
                c1 = tCcC[k * 2 + 1]
                sSchurA2[c0[0], c0[1]] = self._dtype(acc_tmp[k * 2])
                sSchurA2[c1[0], c1[1]] = self._dtype(acc_tmp[k * 2 + 1])
                sSchurB2[c0[0], c0[1]] = self._dtype(sAi2[c0[1], c0[0]])
                sSchurB2[c1[0], c1[1]] = self._dtype(sAi2[c1[1], c1[0]])
            cute.arch.sync_warp()
            cute.copy(smem_copy_A, tCsA2[None, None, 0], tCrAc[None, None, 0])
            cute.copy(smem_copy_B, tCsB2[None, None, 0], tCrBc[None, None, 0])
            acc_res1.fill(0.0)
            cute.gemm(tiled_mma, acc_res1, tCrA[None, None, 0], tCrB[None, None, 0], acc_res1)
            for i in cutlass.range_constexpr(cute.size(acc_res1)):
                acc_res1[i] = -acc_res1[i]

            # Store Ai32
            for i in cutlass.range_constexpr(cute.size(acc_res1)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                if i_tc3 + row < eos:
                    mAkk[i_tc3 + row, h_akk_col + 2 * self.BC + col] = self._dtype(acc_res1[i])

        # ── WARP 3: Store all 4 diagonal blocks ──
        if is_active and warp_idx == 3:
            for k in cutlass.range_constexpr(self.BC * self.BC // 32):
                linear_idx = lane_idx + k * 32
                row = linear_idx // self.BC
                col_idx = linear_idx % self.BC
                value0 = sAi0[row, col_idx]
                value1 = sAi1[row, col_idx]
                value2 = sAi2[row, col_idx]
                value3 = sAi3[row, col_idx]
                if col_idx > row:
                    value0 = self._sai_dtype(0.0)
                    value1 = self._sai_dtype(0.0)
                    value2 = self._sai_dtype(0.0)
                    value3 = self._sai_dtype(0.0)
                if i_tc0 + row < eos:
                    mAkk[i_tc0 + row, h_akk_col + 0 * self.BC + col_idx] = self._dtype(value0)
                if i_tc1 + row < eos:
                    mAkk[i_tc1 + row, h_akk_col + 1 * self.BC + col_idx] = self._dtype(value1)
                if i_tc2 + row < eos:
                    mAkk[i_tc2 + row, h_akk_col + 2 * self.BC + col_idx] = self._dtype(value2)
                if i_tc3 + row < eos:
                    mAkk[i_tc3 + row, h_akk_col + 3 * self.BC + col_idx] = self._dtype(value3)
