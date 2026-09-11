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

"""Gated DeltaNet (GDN) Cutlass-primitives bprop kernel config (fixed compile-time
constants; the per-compile attributes live on ``GdnBwdCfg`` in the kernel
file).

Target arch: Blackwell SM100 (GB200) / SM103 (GB300).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 64  # chunk size / token tile (the mma N or K of every GEMM)
    D_K: int = 128  # query/key head dim
    D_V: int = 128  # value head dim

    # --- TMA descriptor pool ---

    # --- warp assignments (12 warps total) ---
    COMPUTE_GROUP_0_WARP_IDS: tuple[int, ...] = (
        0,
        1,
        2,
        3,
    )  # T-pairwise / kk_epi / qk_epi / inverse / parts
    COMPUTE_GROUP_1_WARP_IDS: tuple[int, ...] = (
        4,
        5,
        6,
        7,
    )  # dH prep / dV-dK-dQ epilogues / dq dot
    MMA_WARP_ID: int = 8
    TMA_QKV_WARP_ID: int = 9
    LOAD_GATE_BETA_WARP_ID: int = 10
    EPILOGUE_WARP_ID: int = 11

    # --- register split ---
    NUM_REGS_COMPUTE_GROUP_0: int = 224
    NUM_REGS_COMPUTE_GROUP_1: int = 248
    NUM_REGS_OTHER: int = 32

    THREADS_PER_WARP: int = 32

    CLUSTER_SHAPE_MNK: tuple[int, int, int] = (1, 1, 1)
    SMEM_SCHED_STAGES: int = 2

    # --- SMEM stage counts ---
    SMEM_Q_STAGES: int = 1
    SMEM_K_STAGES: int = 2
    SMEM_V_STAGES: int = 1
    SMEM_T_INV_STAGES: int = 1
    SMEM_A_STAGES: int = 1

    # --- TMEM stage counts ---
    TMEM_DH_ACC_STAGES: int = 1
    TMEM_DVDK_ACC_STAGES: int = 1
    TMEM_DH_INP_STAGES: int = 1
    TMEM_SHARED_INP_STAGES: int = 2
    TMEM_SHARED_ACC_STAGES: int = 2

    BUFFER_ALIGN_BYTES: int = 1024


CFG = Cfg()
