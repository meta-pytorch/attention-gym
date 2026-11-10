# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verbatim copies of the pre-warpdrive K3b/K4b kernels and their compile functions.

Kept only for the bitwise parity test and the A/B timing of the warpdrive port; the next commit
deletes this module with ``test/test_kda_inter_solve_legacy_parity.py``.
"""

import cutlass
from cutlass import Int32, cute
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor

from attn_gym.linear._delta_rule.chunk_schedule import ScheduleKind
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
    _check_compile_target,
    _make_fake_chunk_index,
    _validate_specialization,
)
from attn_gym.linear.kda.fwd.cute.chunk_schedule import ChunkSchedule
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import (
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache
from attn_gym.warpdrive.compat import SmemAllocator


class LegacyChunkKDAFwdK3bOffdiagCuteDSL:
    WARP_SIZE = 32

    def __init__(
        self,
        BC: int = 16,
        D: int = 128,
        chunk_size: int = 64,
        num_subchunks: int = 4,
        chunk_schedule: ChunkSchedule = ChunkSchedule.DENSE,
        schedule_kind: ScheduleKind = ScheduleKind.STATIC,
        chunk_workers: int = 0,
        use_int64_offsets: bool = False,
        fastmath: bool = True,
    ):
        assert num_subchunks == 4, (
            f"ChunkKDAFwdK3bOffdiagCuteDSL only supports four subchunks, got {num_subchunks}"
        )
        assert chunk_size == num_subchunks * BC, (
            f"chunk_size must equal num_subchunks * BC, got {chunk_size} and "
            f"{num_subchunks} * {BC}"
        )
        if schedule_kind is ScheduleKind.PERSISTENT:
            assert chunk_schedule is ChunkSchedule.RAGGED and chunk_workers > 0, (
                "persistent execution requires a positive ragged chunk-worker grid"
            )
        else:
            assert chunk_workers == 0, "static execution does not use chunk workers"
        self.schedule_kind = schedule_kind
        self.chunk_workers = chunk_workers
        self.use_int64_offsets = use_int64_offsets
        self.fastmath = fastmath
        self.BC = BC
        self.D = D
        self.chunk_schedule = chunk_schedule
        self.BT = chunk_size
        self.num_offdiag_blocks = num_subchunks * (num_subchunks - 1) // 2
        self.num_threads = 128
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (1, 1, 1)

    @cute.jit
    def upcast(self, value):
        """Promote an address operand before its first overflowing multiply."""
        return cutlass.Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mAqk: cute.Tensor,
        mAkkOD: cute.Tensor,
        scale: cutlass.Float32,
        H: int,
        total_chunks: int,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        stream,
    ):
        self._dtype: type[cutlass.Numeric] = mQ.element_type
        chunk_axis = (
            self.chunk_workers
            if cutlass.const_expr(self.schedule_kind is ScheduleKind.PERSISTENT)
            else total_chunks
        )
        grid = (chunk_axis * self.num_offdiag_blocks, H, 1)

        smem_k_block_size = 64
        swizzle_bits = 3
        sQK_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sGated_layout = cute.tile_to_shape(sQK_layout_atom, (self.BC, self.D), (0, 1))
        sGref_layout = cute.make_layout((self.D,), stride=(1,))
        sBeta_layout = cute.make_layout((self.BC,), stride=(1,))

        @cute.struct
        class SharedStorage:
            sQg: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sGated_layout)], 128
            ]
            sKp: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sGated_layout)], 128
            ]
            sKn: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sGated_layout)], 128
            ]
            sGref: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.D], 128]
            sBeta: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.BC], 128]

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

        self.kernel.set_name_prefix(f"legacy_kda_fwd_k3b_fm{int(self.fastmath)}")
        self.kernel(
            mQ,
            mK,
            mG,
            mBeta,
            mAqk,
            mAkkOD,
            scale,
            cu_seqlens,
            chunk_offsets,
            sGated_layout,
            sGref_layout,
            sBeta_layout,
            smem_copy_A,
            smem_copy_B,
            tiled_mma,
            SharedStorage,
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mAqk: cute.Tensor,
        mAkkOD: cute.Tensor,
        scale: cutlass.Float32,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        sGated_layout: cute.ComposedLayout,
        sGref_layout: cute.Layout,
        sBeta_layout: cute.Layout,
        smem_copy_A: cute.TiledCopy,
        smem_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, head_idx, _ = cute.arch.block_idx()

        chunk_idx = block_idx // self.num_offdiag_blocks
        pair_idx = block_idx % self.num_offdiag_blocks

        ri = 1
        if pair_idx >= 1:
            ri = 2
        if pair_idx >= 3:
            ri = 3
        ci = pair_idx - (ri * (ri - 1)) // 2

        # ── Shared memory, allocated once and reused across persistent iterations ──
        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQg = storage.sQg.get_tensor(sGated_layout)
        sKp = storage.sKp.get_tensor(sGated_layout)
        sKn = storage.sKn.get_tensor(sGated_layout)
        sGref = storage.sGref.get_tensor(sGref_layout)
        sBeta = storage.sBeta.get_tensor(sBeta_layout)

        if cutlass.const_expr(self.schedule_kind is ScheduleKind.PERSISTENT):
            active_chunks = load_ragged_chunk_count(chunk_offsets)
            # unroll=2 interleaves two chunks' gating loads, restoring the
            # load->use distance the scheduler achieves in the static body
            # (median 46 vs 14 instructions; long_scoreboard 25 -> ~12 cyc/inst).
            # unroll=4 crosses a code-size/register cliff and loses it again.
            for current_chunk in cutlass.range(
                chunk_idx, active_chunks, self.chunk_workers, unroll=2
            ):
                self._process_chunk(
                    mQ,
                    mK,
                    mG,
                    mBeta,
                    mAqk,
                    mAkkOD,
                    scale,
                    cu_seqlens,
                    chunk_offsets,
                    sQg,
                    sKp,
                    sKn,
                    sGref,
                    sBeta,
                    smem_copy_A,
                    smem_copy_B,
                    tiled_mma,
                    current_chunk,
                    pair_idx,
                    ri,
                    ci,
                    head_idx,
                    tidx,
                )
                # Synchronize all warps before the next chunk's gating reuses SMEM.
                cute.arch.sync_threads()
        else:
            self._process_chunk(
                mQ,
                mK,
                mG,
                mBeta,
                mAqk,
                mAkkOD,
                scale,
                cu_seqlens,
                chunk_offsets,
                sQg,
                sKp,
                sKn,
                sGref,
                sBeta,
                smem_copy_A,
                smem_copy_B,
                tiled_mma,
                chunk_idx,
                pair_idx,
                ri,
                ci,
                head_idx,
                tidx,
            )

    @cute.jit
    def _process_chunk(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mAqk: cute.Tensor,
        mAkkOD: cute.Tensor,
        scale: cutlass.Float32,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        sQg: cute.Tensor,
        sKp: cute.Tensor,
        sKn: cute.Tensor,
        sGref: cute.Tensor,
        sBeta: cute.Tensor,
        smem_copy_A: cute.TiledCopy,
        smem_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        chunk_idx: Int32,
        pair_idx: Int32,
        ri: Int32,
        ci: Int32,
        head_idx: Int32,
        tidx: Int32,
    ):
        """Gate, multiply, and store one (chunk, pair, head) off-diagonal tile."""
        warp_idx = tidx // self.WARP_SIZE
        lane_idx = tidx % self.WARP_SIZE

        if cutlass.const_expr(self.chunk_schedule is ChunkSchedule.RAGGED):
            active_chunks = load_ragged_chunk_count(chunk_offsets)
            chunk_base = Int32(0)
            eos = Int32(0)
            if chunk_idx < active_chunks:
                _, _, chunk_base, valid_tokens = load_ragged_chunk_work(
                    cu_seqlens,
                    chunk_offsets,
                    Int32(chunk_idx),
                    Int32(self.BT),
                )
                eos = chunk_base + valid_tokens
        else:
            chunk_base = chunk_idx * self.BT
            eos = cute.size(mQ, mode=[0])

        chunk_base = self.upcast(chunk_base)
        ti_row = chunk_base + ri * self.BC
        ti_col = chunk_base + ci * self.BC
        h_offset = head_idx * self.D
        aqk_col = head_idx * self.BT
        akkod_col = head_idx * self.BC * self.BC

        # ══════════════════════════════════════════════════════════
        # Phase 1: Gating into SMEM
        # ══════════════════════════════════════════════════════════
        if cutlass.const_expr(self.chunk_schedule is not ChunkSchedule.DENSE):
            col = tidx
            h_col = h_offset + col
            if chunk_base + self.BT <= eos:
                if tidx < self.D:
                    sGref[tidx] = mG[ti_row, h_col]
                if tidx < self.BC:
                    sBeta[tidx] = cutlass.Float32(mBeta[ti_row + tidx, head_idx])
                for _r in cutlass.range_constexpr(self.BC):
                    g_ref_val = sGref[col]
                    q_val = cutlass.Float32(mQ[ti_row + _r, h_col])
                    k_val_r = cutlass.Float32(mK[ti_row + _r, h_col])
                    g_val_r = mG[ti_row + _r, h_col]
                    gate_r = cute.math.exp2(g_val_r - g_ref_val, fastmath=self.fastmath)
                    sQg[_r, col] = self._dtype(q_val * gate_r)
                    sKp[_r, col] = self._dtype(k_val_r * gate_r)

                    k_val_c = cutlass.Float32(mK[ti_col + _r, h_col])
                    g_val_c = mG[ti_col + _r, h_col]
                    gate_c = cute.math.exp2(g_ref_val - g_val_c, fastmath=self.fastmath)
                    sKn[_r, col] = self._dtype(k_val_c * gate_c)
            else:
                # ── Varlen tail: predicated constexpr-offset loads (R2P-style) ──
                actual_row = cutlass.min(self.BC, cutlass.max(eos - ti_row, 0))
                actual_col = cutlass.min(self.BC, cutlass.max(eos - ti_col, 0))
                safe_ti_row = cutlass.min(ti_row, cutlass.max(eos - 1, 0))

                if tidx < self.D and eos > 0:
                    sGref[tidx] = mG[safe_ti_row, h_col]
                if tidx < self.D and eos == 0:
                    # Capacity-only CTAs (eos stays 0) must not touch token storage; a zero
                    # reference gate keeps their SMEM tiles deterministic zeros.
                    sGref[tidx] = mG.element_type(0.0)
                if tidx < self.BC:
                    beta_val = cutlass.Float32(0.0)
                    if tidx < actual_row:
                        beta_val = cutlass.Float32(mBeta[ti_row + tidx, head_idx])
                    sBeta[tidx] = beta_val

                for _r in cutlass.range_constexpr(self.BC):
                    g_ref_val = sGref[col]

                    q_val = cutlass.Float32(0.0)
                    k_val_r = cutlass.Float32(0.0)
                    g_val_r = g_ref_val
                    if _r < actual_row:
                        q_val = cutlass.Float32(mQ[ti_row + _r, h_col])
                        k_val_r = cutlass.Float32(mK[ti_row + _r, h_col])
                        g_val_r = mG[ti_row + _r, h_col]
                    gate_r = cute.math.exp2(g_val_r - g_ref_val, fastmath=self.fastmath)
                    sQg[_r, col] = self._dtype(q_val * gate_r)
                    sKp[_r, col] = self._dtype(k_val_r * gate_r)

                    k_val_c = cutlass.Float32(0.0)
                    g_val_c = g_ref_val
                    if _r < actual_col:
                        k_val_c = cutlass.Float32(mK[ti_col + _r, h_col])
                        g_val_c = mG[ti_col + _r, h_col]
                    gate_c = cute.math.exp2(g_ref_val - g_val_c, fastmath=self.fastmath)
                    sKn[_r, col] = self._dtype(k_val_c * gate_c)
        else:
            # ── Non-varlen: constexpr-unrolled, zero guards ──
            if tidx < self.D:
                sGref[tidx] = mG[ti_row, h_offset + tidx]
            if tidx < self.BC:
                sBeta[tidx] = cutlass.Float32(mBeta[ti_row + tidx, head_idx])
            for _r in cutlass.range_constexpr(self.BC):
                col = tidx
                g_ref_val = sGref[col]
                q_val = cutlass.Float32(mQ[ti_row + _r, h_offset + col])
                k_val_r = cutlass.Float32(mK[ti_row + _r, h_offset + col])
                g_val_r = mG[ti_row + _r, h_offset + col]
                gate_r = cute.math.exp2(g_val_r - g_ref_val, fastmath=self.fastmath)
                sQg[_r, col] = self._dtype(q_val * gate_r)
                sKp[_r, col] = self._dtype(k_val_r * gate_r)
                k_val_c = cutlass.Float32(mK[ti_col + _r, h_offset + col])
                g_val_c = mG[ti_col + _r, h_offset + col]
                gate_c = cute.math.exp2(g_ref_val - g_val_c, fastmath=self.fastmath)
                sKn[_r, col] = self._dtype(k_val_c * gate_c)

        cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads)

        # ══════════════════════════════════════════════════════════
        # Phase 2: MMA — Warp 0: Aqk = sQg @ sKn^T
        #                  Warp 1: Akk = sKp @ sKn^T
        # ══════════════════════════════════════════════════════════
        thr_mma = tiled_mma.get_slice(lane_idx)

        thrA = smem_copy_A.get_slice(lane_idx)
        _tCsQg = thrA.partition_S(sQg)
        _tCsKp = thrA.partition_S(sKp)
        _tMmaQg = thr_mma.partition_A(sQg)
        _tMmaKp = thr_mma.partition_A(sKp)
        tCrQg = thr_mma.make_fragment_A(_tMmaQg)
        tCrKp = thr_mma.make_fragment_A(_tMmaKp)
        tCrQgc = thrA.retile(tCrQg)
        tCrKpc = thrA.retile(tCrKp)

        thrB = smem_copy_B.get_slice(lane_idx)
        _tCsKn_B = thrB.partition_S(sKn)
        _tMmaKn = thr_mma.partition_B(sKn)
        tCrKn = thr_mma.make_fragment_B(_tMmaKn)
        tCrKnc = thrB.retile(tCrKn)

        acc_shape = thr_mma.partition_shape_C((self.BC, self.BC))
        acc_Aqk = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_Akk = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_Aqk.fill(0.0)
        acc_Akk.fill(0.0)

        num_k_blocks = cute.size(_tCsQg, mode=[2])

        if warp_idx == 0:
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.copy(smem_copy_A, _tCsQg[None, None, kb], tCrQgc[None, None, kb])
                cute.copy(smem_copy_B, _tCsKn_B[None, None, kb], tCrKnc[None, None, kb])
                cute.gemm(
                    tiled_mma,
                    acc_Aqk,
                    tCrQg[None, None, kb],
                    tCrKn[None, None, kb],
                    acc_Aqk,
                )
        elif warp_idx == 1:
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.copy(smem_copy_A, _tCsKp[None, None, kb], tCrKpc[None, None, kb])
                cute.copy(smem_copy_B, _tCsKn_B[None, None, kb], tCrKnc[None, None, kb])
                cute.gemm(
                    tiled_mma,
                    acc_Akk,
                    tCrKp[None, None, kb],
                    tCrKn[None, None, kb],
                    acc_Akk,
                )

        # ══════════════════════════════════════════════════════════
        # Phase 3: Scaled stores to GMEM
        # ══════════════════════════════════════════════════════════
        cC = cute.make_identity_tensor((self.BC, self.BC))
        tCcC = thr_mma.partition_C(cC)

        od_row = self.upcast(chunk_idx) * self.num_offdiag_blocks + pair_idx

        if warp_idx == 0:
            out_dtype = mAqk.element_type
            for i in cutlass.range_constexpr(cute.size(acc_Aqk)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                if cutlass.const_expr(self.chunk_schedule is not ChunkSchedule.DENSE):
                    if chunk_base + self.BT <= eos or ti_row + row < eos:
                        mAqk[ti_row + row, aqk_col + ci * self.BC + col] = out_dtype(
                            acc_Aqk[i] * scale
                        )
                else:
                    mAqk[ti_row + row, aqk_col + ci * self.BC + col] = out_dtype(
                        acc_Aqk[i] * scale
                    )
        elif warp_idx == 1:
            for i in cutlass.range_constexpr(cute.size(acc_Akk)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                beta_val = sBeta[row]
                mAkkOD[od_row, akkod_col + row * self.BC + col] = acc_Akk[i] * beta_val
        elif warp_idx == 2:
            # Define the noncausal block while this CTA already owns the
            # corresponding lower block. Aqk is exposed as custom-op factor, so
            # its complete storage must be deterministic for fake/AOT checks.
            out_dtype = mAqk.element_type
            for i in cutlass.range_constexpr(cute.size(acc_Aqk)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                if cutlass.const_expr(self.chunk_schedule is not ChunkSchedule.DENSE):
                    if ti_col + row < eos:
                        mAqk[ti_col + row, aqk_col + ri * self.BC + col] = out_dtype(0.0)
                else:
                    mAqk[ti_col + row, aqk_col + ri * self.BC + col] = out_dtype(0.0)


class LegacyChunkKDAFwdK4bInverseCuteDSL:
    WARP_SIZE = 32

    def __init__(
        self,
        BC: int = 16,
        chunk_size: int = 64,
        num_subchunks: int = 4,
        chunk_schedule: ChunkSchedule = ChunkSchedule.DENSE,
        schedule_kind: ScheduleKind = ScheduleKind.STATIC,
        chunk_workers: int = 0,
        use_int64_offsets: bool = False,
    ):
        assert num_subchunks == 4, (
            f"ChunkKDAFwdK4bInverseCuteDSL only supports four subchunks, got {num_subchunks}"
        )
        assert chunk_size == num_subchunks * BC, (
            f"chunk_size must equal num_subchunks * BC, got {chunk_size} and "
            f"{num_subchunks} * {BC}"
        )
        if schedule_kind is ScheduleKind.PERSISTENT:
            assert chunk_schedule is ChunkSchedule.RAGGED and chunk_workers > 0, (
                "persistent execution requires a positive ragged chunk-worker grid"
            )
        else:
            assert chunk_workers == 0, "static execution does not use chunk workers"
        self.schedule_kind = schedule_kind
        self.chunk_workers = chunk_workers
        self.use_int64_offsets = use_int64_offsets
        self.BC = BC
        self.BT = chunk_size
        self.num_offdiag_blocks = num_subchunks * (num_subchunks - 1) // 2
        self.num_threads = 128
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (1, 1, 1)
        self.chunk_schedule = chunk_schedule

    @cute.jit
    def upcast(self, value):
        """Promote an address operand before its first overflowing multiply."""
        return cutlass.Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    @cute.jit
    def _load_diagonal_inverse_block(
        self, mAkkd, sAi, block_start, tidx, akkd_col_off, valid_rows
    ):
        """Load one preinverted BC×BC diagonal block, zero-filling invalid rows."""
        for k in cutlass.range_constexpr(self.BC * self.BC // 32):
            linear_idx = tidx + k * 32
            row = linear_idx // self.BC
            col = linear_idx % self.BC
            if row < valid_rows:
                sAi[row, col] = self._dtype(mAkkd[block_start + row, akkd_col_off + col])
            else:
                sAi[row, col] = self._dtype(0.0)
        cute.arch.sync_warp()

    @cute.jit
    def _store_zero_block(self, mAkk, row_start, col_start, lane_idx, eos):
        """Store one explicit zero 16x16 upper block."""
        for k in cutlass.range_constexpr(self.BC * self.BC // self.WARP_SIZE):
            linear_idx = lane_idx + k * self.WARP_SIZE
            row = linear_idx // self.BC
            col = linear_idx % self.BC
            if row_start + row < eos:
                mAkk[row_start + row, col_start + col] = self._dtype(0.0)

    @cute.jit
    def __call__(
        self,
        mAkkOD: cute.Tensor,
        mAkkd: cute.Tensor,
        mAkk: cute.Tensor,
        H: int,
        total_chunks: int,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        stream,
    ):
        self._dtype: type[cutlass.Numeric] = mAkk.element_type
        chunk_axis = (
            self.chunk_workers
            if cutlass.const_expr(self.schedule_kind is ScheduleKind.PERSISTENT)
            else total_chunks
        )
        grid = (chunk_axis, H, 1)

        sSchurA_layout = cute.make_layout((self.BC, self.BC), stride=(self.BC + 8, 1))
        sAi_layout = cute.make_layout((self.BC, self.BC), stride=(self.BC, 1))
        sSchurB_layout = cute.make_layout((self.BC, self.BC), stride=(self.BC + 8, 1))

        @cute.struct
        class SharedStorage:
            work: cute.struct.MemRange[Int32, 3]
            sAi0: cute.struct.Align[cute.struct.MemRange[self._dtype, self.BC * self.BC], 128]
            sAi1: cute.struct.Align[cute.struct.MemRange[self._dtype, self.BC * self.BC], 128]
            sAi2: cute.struct.Align[cute.struct.MemRange[self._dtype, self.BC * self.BC], 128]
            sAi3: cute.struct.Align[cute.struct.MemRange[self._dtype, self.BC * self.BC], 128]
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

        # ── Shared memory, allocated once and reused across persistent iterations ──
        smem = SmemAllocator()
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
        smem_tensors = (
            work,
            sAi0,
            sAi1,
            sAi2,
            sAi3,
            sSchurA0,
            sSchurB0,
            sSchurA1,
            sSchurB1,
            sSchurA2,
            sSchurB2,
        )

        if cutlass.const_expr(self.schedule_kind is ScheduleKind.PERSISTENT):
            active_chunks = load_ragged_chunk_count(chunk_offsets)
            # No unroll: K4b's register-heavy Schur body regresses under
            # unroll=2 (0.97x -> 0.87x measured vs the static grid).
            for current_chunk in cutlass.range(chunk_idx, active_chunks, self.chunk_workers):
                self._process_chunk(
                    mAkkOD,
                    mAkkd,
                    mAkk,
                    cu_seqlens,
                    chunk_offsets,
                    smem_tensors,
                    smem_copy_A,
                    smem_copy_B,
                    tiled_mma,
                    current_chunk,
                    head_idx,
                    tidx,
                )
                # Synchronize all warps before the next chunk's loads reuse SMEM.
                cute.arch.sync_threads()
        else:
            self._process_chunk(
                mAkkOD,
                mAkkd,
                mAkk,
                cu_seqlens,
                chunk_offsets,
                smem_tensors,
                smem_copy_A,
                smem_copy_B,
                tiled_mma,
                chunk_idx,
                head_idx,
                tidx,
            )

    @cute.jit
    def _process_chunk(
        self,
        mAkkOD: cute.Tensor,
        mAkkd: cute.Tensor,
        mAkk: cute.Tensor,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        smem_tensors,
        smem_copy_A: cute.TiledCopy,
        smem_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        chunk_idx: Int32,
        head_idx: Int32,
        tidx: Int32,
    ):
        """Invert one chunk's 4x4 block lower-triangular system."""
        warp_idx = tidx // self.WARP_SIZE
        lane_idx = tidx % self.WARP_SIZE
        (
            work,
            sAi0,
            sAi1,
            sAi2,
            sAi3,
            sSchurA0,
            sSchurB0,
            sSchurA1,
            sSchurB1,
            sSchurA2,
            sSchurB2,
        ) = smem_tensors

        if cutlass.const_expr(self.chunk_schedule is ChunkSchedule.RAGGED):
            if tidx == 0:
                active_chunks = load_ragged_chunk_count(chunk_offsets)
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
        else:
            chunk_base = chunk_idx * self.BT
            eos = cute.size(mAkk, mode=[0])
            is_active = Int32(1)

        chunk_base = self.upcast(chunk_base)
        i_tc0 = chunk_base
        i_tc1 = chunk_base + self.BC
        i_tc2 = chunk_base + 2 * self.BC
        i_tc3 = chunk_base + 3 * self.BC

        h_akkd_col = head_idx * self.BC
        h_akk_col = head_idx * self.BT
        h_akkod_col = head_idx * self.BC * self.BC

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
        # PHASE 1: Per-warp OD loading + preinverted diagonal-block loading
        # ══════════════════════════════════════════════════════════
        od_base_row = self.upcast(chunk_idx) * self.num_offdiag_blocks

        if is_active and warp_idx == 0:
            for i in cutlass.range_constexpr(cute.size(acc_od0)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                rc = h_akkod_col + row * self.BC + col
                acc_od0[i] = mAkkOD[od_base_row + 0, rc]
                acc_od1[i] = mAkkOD[od_base_row + 1, rc]
                acc_od2[i] = mAkkOD[od_base_row + 2, rc]
                acc_od3[i] = mAkkOD[od_base_row + 3, rc]
                acc_od4[i] = mAkkOD[od_base_row + 4, rc]
                acc_od5[i] = mAkkOD[od_base_row + 5, rc]
            self._load_diagonal_inverse_block(mAkkd, sAi0, i_tc0, lane_idx, h_akkd_col, vr0)

        if is_active and warp_idx == 1:
            for i in cutlass.range_constexpr(cute.size(acc_od0)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                rc = h_akkod_col + row * self.BC + col
                acc_od2[i] = mAkkOD[od_base_row + 2, rc]
                acc_od4[i] = mAkkOD[od_base_row + 4, rc]
                acc_od5[i] = mAkkOD[od_base_row + 5, rc]
            self._load_diagonal_inverse_block(mAkkd, sAi1, i_tc1, lane_idx, h_akkd_col, vr1)

        if is_active and warp_idx == 2:
            for i in cutlass.range_constexpr(cute.size(acc_od0)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                rc = h_akkod_col + row * self.BC + col
                acc_od5[i] = mAkkOD[od_base_row + 5, rc]
            self._load_diagonal_inverse_block(mAkkd, sAi2, i_tc2, lane_idx, h_akkd_col, vr2)

        if is_active and warp_idx == 3:
            self._load_diagonal_inverse_block(mAkkd, sAi3, i_tc3, lane_idx, h_akkd_col, vr3)

        cute.arch.barrier(barrier_id=0, number_of_threads=128)

        # ══════════════════════════════════════════════════════════
        # PHASE 2: Parallel Schur Complement + Overlapped Stores
        # ══════════════════════════════════════════════════════════
        # Each compute warp owns one sSchurA/B pair and writes then reads it in
        # converged warp order. The CTA barriers protect cross-warp diagonal
        # handoff above and SMEM reuse between persistent iterations.

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

            # Store Ai32 and three upper-zero blocks while warp 0 remains busy.
            for i in cutlass.range_constexpr(cute.size(acc_res1)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                if i_tc3 + row < eos:
                    mAkk[i_tc3 + row, h_akk_col + 2 * self.BC + col] = self._dtype(acc_res1[i])
            self._store_zero_block(mAkk, i_tc0, h_akk_col + self.BC, lane_idx, eos)
            self._store_zero_block(mAkk, i_tc1, h_akk_col + 2 * self.BC, lane_idx, eos)
            self._store_zero_block(mAkk, i_tc2, h_akk_col + 3 * self.BC, lane_idx, eos)

        # ── WARP 3: Store all 4 diagonal blocks and remaining upper zeros ──
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
                    value0 = self._dtype(0.0)
                    value1 = self._dtype(0.0)
                    value2 = self._dtype(0.0)
                    value3 = self._dtype(0.0)
                if i_tc0 + row < eos:
                    mAkk[i_tc0 + row, h_akk_col + 0 * self.BC + col_idx] = self._dtype(value0)
                if i_tc1 + row < eos:
                    mAkk[i_tc1 + row, h_akk_col + 1 * self.BC + col_idx] = self._dtype(value1)
                if i_tc2 + row < eos:
                    mAkk[i_tc2 + row, h_akk_col + 2 * self.BC + col_idx] = self._dtype(value2)
                if i_tc3 + row < eos:
                    mAkk[i_tc3 + row, h_akk_col + 3 * self.BC + col_idx] = self._dtype(value3)
            self._store_zero_block(mAkk, i_tc0, h_akk_col + 2 * self.BC, lane_idx, eos)
            self._store_zero_block(mAkk, i_tc0, h_akk_col + 3 * self.BC, lane_idx, eos)
            self._store_zero_block(mAkk, i_tc1, h_akk_col + 3 * self.BC, lane_idx, eos)


@jit_cache
def _compile_legacy_k3b(
    heads: int,
    head_dim: int,
    chunk_size: int,
    subchunk_size: int,
    io_type: type[cutlass.Numeric],
    io_name: str,
    chunk_schedule: ChunkSchedule,
    schedule_kind: ScheduleKind,
    chunk_workers: int = 0,
    use_int64_offsets: bool = False,
    fastmath: bool = True,
):
    """Compile K3b for independent layout and execution schedules."""
    _check_compile_target()
    offdiag_blocks = _validate_specialization(head_dim, chunk_size, subchunk_size)
    num_subchunks = chunk_size // subchunk_size
    op = LegacyChunkKDAFwdK3bOffdiagCuteDSL(
        BC=subchunk_size,
        D=head_dim,
        chunk_size=chunk_size,
        num_subchunks=num_subchunks,
        chunk_schedule=chunk_schedule,
        schedule_kind=schedule_kind,
        chunk_workers=chunk_workers,
        use_int64_offsets=use_int64_offsets,
        fastmath=fastmath,
    )
    tokens, chunks, sequences = (cute.sym_int() for _ in range(3))
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    q = make_fake_tensor(
        io_type,
        (tokens, heads * head_dim),
        stride=(sym_int(divisibility=8), 1),
        assumed_align=16,
    )
    k = make_fake_tensor(
        io_type,
        (tokens, heads * head_dim),
        stride=(sym_int(divisibility=8), 1),
        assumed_align=16,
    )
    g = make_fake_compact_tensor(
        cutlass.Float32,
        (tokens, heads * head_dim),
        stride_order=(1, 0),
        assumed_align=16,
    )
    beta = make_fake_compact_tensor(
        cutlass.Float32,
        (tokens, heads),
        stride_order=(1, 0),
        assumed_align=16,
    )
    Aqk = make_fake_compact_tensor(
        io_type,
        (tokens, heads * chunk_size),
        stride_order=(1, 0),
        assumed_align=16,
    )
    AkkOD = make_fake_compact_tensor(
        cutlass.Float32,
        (chunks * offdiag_blocks, heads * subchunk_size * subchunk_size),
        stride_order=(1, 0),
        assumed_align=16,
    )
    cu_seqlens = _make_fake_chunk_index(chunk_schedule, sequences)
    chunk_offsets = _make_fake_chunk_index(chunk_schedule, sequences)
    return compile_tvm_ffi(
        op,
        q,
        k,
        g,
        beta,
        Aqk,
        AkkOD,
        cutlass.Float32(1.0),
        heads,
        1,
        cu_seqlens,
        chunk_offsets,
        name=(
            f"legacy_kda_fwd_k3b_h{heads}_d{head_dim}_c{chunk_size}_sc{subchunk_size}_{io_name}"
            f"_layout_{chunk_schedule.value}_execution_{schedule_kind.value}"
            f"_cw{chunk_workers}_i64{int(use_int64_offsets)}_fm{int(fastmath)}"
        ),
    )


@jit_cache
def _compile_legacy_k4b(
    heads: int,
    head_dim: int,
    chunk_size: int,
    subchunk_size: int,
    io_type: type[cutlass.Numeric],
    io_name: str,
    chunk_schedule: ChunkSchedule,
    schedule_kind: ScheduleKind,
    chunk_workers: int = 0,
    use_int64_offsets: bool = False,
):
    """Compile K4b for independent layout and execution schedules."""
    _check_compile_target()
    offdiag_blocks = _validate_specialization(head_dim, chunk_size, subchunk_size)
    num_subchunks = chunk_size // subchunk_size
    op = LegacyChunkKDAFwdK4bInverseCuteDSL(
        BC=subchunk_size,
        chunk_size=chunk_size,
        num_subchunks=num_subchunks,
        chunk_schedule=chunk_schedule,
        schedule_kind=schedule_kind,
        chunk_workers=chunk_workers,
        use_int64_offsets=use_int64_offsets,
    )
    tokens, chunks, sequences = (cute.sym_int() for _ in range(3))
    AkkOD = make_fake_compact_tensor(
        cutlass.Float32,
        (chunks * offdiag_blocks, heads * subchunk_size * subchunk_size),
        stride_order=(1, 0),
        assumed_align=16,
    )
    Akkd = make_fake_compact_tensor(
        cutlass.Float32,
        (tokens, heads * subchunk_size),
        stride_order=(1, 0),
        assumed_align=16,
    )
    Akk = make_fake_compact_tensor(
        io_type,
        (tokens, heads * chunk_size),
        stride_order=(1, 0),
        assumed_align=16,
    )
    cu_seqlens = _make_fake_chunk_index(chunk_schedule, sequences)
    chunk_offsets = _make_fake_chunk_index(chunk_schedule, sequences)
    return compile_tvm_ffi(
        op,
        AkkOD,
        Akkd,
        Akk,
        heads,
        1,
        cu_seqlens,
        chunk_offsets,
        name=(
            f"legacy_kda_fwd_k4b_h{heads}_d{head_dim}_c{chunk_size}_sc{subchunk_size}_{io_name}"
            f"_layout_{chunk_schedule.value}_execution_{schedule_kind.value}"
            f"_cw{chunk_workers}_i64{int(use_int64_offsets)}"
        ),
    )
