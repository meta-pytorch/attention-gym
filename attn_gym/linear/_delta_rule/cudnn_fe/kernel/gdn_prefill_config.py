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
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Gated DeltaNet (GDN) Cutlass-primitives prefill kernel config (fixed compile-time
constants; the per-compile attributes live on ``GdnCfg`` in the kernel file).

Target arch: SM100 / SM103 / SM107.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 64

    # --- warp assignments (12 warps total) ---
    COMPUTE_GROUP_0_WARP_IDS: tuple[int, ...] = (0, 1, 2, 3)
    COMPUTE_GROUP_1_WARP_IDS: tuple[int, ...] = (4, 5, 6, 7)
    LOAD_GATE_WARP_ID: int = 8
    TMA_QKV_WARP_ID: int = 9
    TCGEN05_MMA_WARP_ID: int = 10
    EPILOGUE_WARP_ID: int = 11

    # --- register split ---
    NUM_REGS_COMPUTE_GROUP_0: int = 224
    NUM_REGS_COMPUTE_GROUP_1: int = 256
    NUM_REGS_OTHER: int = 24

    THREADS_PER_WARP: int = 32

    CLUSTER_SHAPE_MNK: tuple[int, int, int] = (1, 1, 1)

    # --- SMEM stage counts ---
    SMEM_SCHEDULER_STAGES: int = 2
    SMEM_KQ_STAGES: int = 4
    SMEM_V_STAGES: int = 2
    SMEM_T_INV_STAGES: int = 3
    SMEM_A_STAGES: int = 3
    SMEM_O_STAGES: int = 1
    SMEM_GATE_STAGES: int = 3

    # --- TMEM stage counts ---
    TMEM_KV_ACC_STAGES: int = 1
    TMEM_Q_STATE_ACC_STAGES: int = 1
    TMEM_STATE_INP_STAGES: int = 1
    TMEM_CG0_ACC_STAGES: int = 2
    TMEM_CG1_ACC_STAGES: int = 1

    BUFFER_ALIGN_BYTES: int = 1024


CFG = Cfg()
