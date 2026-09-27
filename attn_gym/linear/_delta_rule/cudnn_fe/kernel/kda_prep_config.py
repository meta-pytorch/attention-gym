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

"""Fixed compile-time constants of the KDA prep (SM100 / SM103 / SM107): the five per-(chunk, head)
prep records (k_decay, q_decay, t, a, diag) of the BT = 16 schedule ahead of the prep-fed prefill;
the per-compile attributes live on ``KdaPrepCfg`` in the kernel file.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Cfg:
    # --- tile shape ---
    B_T: int = 16

    # --- warp assignments (4 warps = 128 threads) ---
    COMPUTE_WARPS: int = 4

    # --- occupancy (CTAS_PER_SM CTAs per SM, SMEM fit asserted by build_cfg) ---
    CTAS_PER_SM: int = 4
    SMEM_PER_SM_BYTES: int = 232448
    SMEM_CTA_RESERVED_BYTES: int = 1024

    THREADS_PER_WARP: int = 32

    BUFFER_ALIGN_BYTES: int = 1024

    # --- SMEM stage counts ---
    SMEM_RAW_STAGES: int = 2
    SMEM_RECORD_STAGES: int = 1


CFG = Cfg()
