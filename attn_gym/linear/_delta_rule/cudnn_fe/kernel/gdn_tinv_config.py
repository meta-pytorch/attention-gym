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

"""Fixed compile-time constants of the GDN chunk-factor (T_inv) pass (SM100 / SM103 / SM107); the
per-compile attributes live on ``GdnTinvCfg`` in the kernel file.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 64

    # --- warp assignments (12 warps total) ---
    COMPUTE_GROUP_WARP_IDS: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3), (4, 5, 6, 7))
    TMA_K_WARP_ID: int = 8
    TCGEN05_MMA_WARP_ID: int = 9
    EPILOGUE_WARP_ID: int = 10
    LOAD_GATE_WARP_ID: int = 11

    # --- register split ---
    LAUNCH_REGS: int = 168
    NUM_REGS_OTHER: int = 96

    THREADS_PER_WARP: int = 32

    # --- SMEM stage counts ---
    SMEM_K_STAGES: int = 4
    SMEM_TILE_STAGES: int = 4
    SMEM_GATE_STAGES: int = 4

    # --- TMEM stage counts ---
    TMEM_ACC_STAGES: int = 4

    BUFFER_ALIGN_BYTES: int = 1024


CFG = Cfg()
