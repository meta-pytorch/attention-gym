# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gated DeltaNet (GDN) Cutlass-primitives recompute (state/H-only) kernel config (fixed
compile-time constants; the per-compile attributes live on ``GdnCfg`` in the kernel file).

Target arch: Blackwell SM100 (GB200) / SM103 (GB300).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 64  # chunk size / token tile (the mma N or K of every GEMM)
    D_K: int = 128  # key head dim (contraction of GEMMs 1/3, M of GEMM 7)
    D_V: int = 128  # value head dim (M of GEMMs 3/5, N of GEMM 7)

    # --- TMA descriptor pool ---

    # --- warp assignments (12 warps total) ---
    COMPUTE_GROUP_0_WARP_IDS: Tuple[int, ...] = (0, 1, 2, 3)  # T-pairwise / kk_epi / inverse
    COMPUTE_GROUP_1_WARP_IDS: Tuple[int, ...] = (4, 5, 6, 7)  # kv_decay_v / v-k*state / epi ops
    LOAD_GATE_BETA_WARP_ID: int = 8  # gate/beta chunk loads + TMEM lifecycle
    TMA_KV_WARP_ID: int = 9
    MMA_WARP_ID: int = 10  # sole tcgen05 issuer: KK pairs + KS/U/KV per chunk
    EPILOGUE_WARP_ID: int = 11

    # --- register split ---
    NUM_REGS_COMPUTE_GROUP_0: int = 224
    NUM_REGS_COMPUTE_GROUP_1: int = 256
    NUM_REGS_OTHER: int = 24

    THREADS_PER_WARP: int = 32

    CLUSTER_SHAPE_MNK: Tuple[int, int, int] = (1, 1, 1)

    # --- SMEM stage counts ---
    SMEM_SCHED_STAGES: int = 2
    SMEM_KQ_STAGES: int = 4
    SMEM_V_STAGES: int = 2
    SMEM_T_INV_STAGES: int = 3
    SMEM_GATE_STAGES: int = 3
    SMEM_BETA_STAGES: int = 3

    # --- TMEM stage counts ---
    TMEM_KV_ACC_STAGES: int = 1
    TMEM_STATE_INP_STAGES: int = 1
    TMEM_CG0_ACC_STAGES: int = 2
    TMEM_CG1_ACC_STAGES: int = 1

    BUFFER_ALIGN_BYTES: int = 1024


CFG = Cfg()
