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

"""Kimi Delta Attention (KDA) Cutlass DSL backward kernel config (fixed
compile-time constants).  The BT=16 backward mirrors the prefill's 16-warp
(512-thread) specialization; the derived SMEM/TMEM sizes and offsets are
stamped by ``build_cfg`` in ``kda_bprop_f16.py``.

Target arch: Blackwell SM100 (GB200) / SM103 (GB300).
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 16  # chunk-inner token tile (BT=16 KDA schedule)
    D_K: int = 128  # query/key head dim
    D_V: int = 128  # value head dim

    # --- warp assignments (16 warps = 512 threads) ---
    COMPUTE_GROUP_0_WARP_IDS: Tuple[int, ...] = (0, 1, 2, 3)  # forward gate cumsum + decay-operand materialize
    COMPUTE_GROUP_1_WARP_IDS: Tuple[int, ...] = (4, 5, 6, 7)  # value-side TMEM staging / restages / dH capture
    COMPUTE_GROUP_2_WARP_IDS: Tuple[int, ...] = (8, 9, 10, 11)  # dq/dk-bank drain, dG assembly + reverse cumsum
    SUPER_MMA_WARP_ID: int = 12  # register-MMA KK/A/dA/dM + Neumann T_inv
    TCGEN05_MMA_WARP_ID: int = 13  # tcgen05 GEMM schedule
    TMA_WARP_ID: int = 14  # q/k/v/gate/do/S(H) TMA loads
    EPILOGUE_WARP_ID: int = 15  # dQ/dK/dV/dGate TMA stores

    # --- register split ---
    NUM_REGS_COMPUTE_GROUP_0: int = 144
    NUM_REGS_COMPUTE_GROUP_1: int = 168
    NUM_REGS_COMPUTE_GROUP_2: int = 144
    NUM_REGS_OTHER: int = 56

    THREADS_PER_WARP: int = 32

    BUFFER_ALIGN_BYTES: int = 1024

    # --- SMEM / TMEM ring stage counts ---
    SMEM_RAW_STAGES: int = 2
    SMEM_S_STAGES: int = 1
    SMEM_DECAY_STAGES: int = 2
    SMEM_INTERMEDIATE_STAGES: int = 2
    SMEM_DA_DIAG_STAGES: int = 4
    SMEM_STATE_SCALE_DIAG_STAGES: int = 2
    SMEM_DQ_STAGES: int = 1
    SMEM_DK_STAGES: int = 1
    SMEM_DGATE_STAGES: int = 1
    SMEM_DV_STAGES: int = 2

    CLUSTER_SHAPE_MNK: Tuple[int, int, int] = (1, 1, 1)


CFG = Cfg()
