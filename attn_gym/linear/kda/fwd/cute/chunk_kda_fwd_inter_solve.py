# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# CuTe DSL KDA forward inter-solve stage.
#
# This is the isolated K3b+K4b replacement for Triton's
# chunk_kda_fwd_kernel_inter_solve_fused_forloop. The caller supplies the
# already-computed diagonal inverse blocks (Akkd); this helper only computes the
# off-diagonal Aqk blocks and final solved Akk blocks.

from __future__ import annotations

import cutlass
import torch
import triton
from cuda.bindings import driver as cuda_drv
from cutlass.cute.runtime import from_dlpack
from attn_gym.linear.kda.utils import (
    DEFAULT_CHUNK_SIZE,
    prepare_chunk_indices,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_k3b_offdiag_cutedsl import (
    ChunkKDAFwdK3bOffdiagCuteDSL,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_k4b_inverse_cutedsl import (
    ChunkKDAFwdK4bInverseCuteDSL,
)
from torch._subclasses.fake_tensor import FakeTensor


def _to_cute_tensor(tensor: torch.Tensor, assumed_align: int = 16):
    return from_dlpack(tensor.detach(), assumed_align=assumed_align)


def chunk_kda_fwd_inter_solve_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Akkd: torch.Tensor,
    scale: float,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_indices: torch.LongTensor | None = None,
    num_chunks: int | torch.Tensor | None = None,
    Aqk: torch.Tensor | None = None,
    Akk: torch.Tensor | None = None,
    AkkOD: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del num_chunks

    assert chunk_size == 64, "chunk_kda_fwd_inter_solve_cute requires chunk_size=64"
    B, T, H, K = k.shape
    BT = chunk_size
    BC = 16
    assert B == 1, f"chunk_kda_fwd_inter_solve_cute requires B=1, got B={B}"
    assert K == 128, f"chunk_kda_fwd_inter_solve_cute requires K=128, got K={K}"
    assert Akkd.shape == (B, T, H, BC), (
        f"Akkd must have shape {(B, T, H, BC)}, got {tuple(Akkd.shape)}"
    )

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if Aqk is None:
        Aqk_flat = torch.empty(B * T, H * BT, device=k.device, dtype=k.dtype)
        Aqk = Aqk_flat.reshape(B, T, H, BT)
    else:
        assert Aqk.shape == (B, T, H, BT), (
            f"Aqk must have shape {(B, T, H, BT)}, got {tuple(Aqk.shape)}"
        )
        Aqk_flat = Aqk.reshape(B * T, H * BT)

    if Akk is None:
        Akk_flat = torch.empty(B * T, H * BT, device=k.device, dtype=k.dtype)
        Akk = Akk_flat.reshape(B, T, H, BT)
    else:
        assert Akk.shape == (B, T, H, BT), (
            f"Akk must have shape {(B, T, H, BT)}, got {tuple(Akk.shape)}"
        )
        Akk_flat = Akk.reshape(B * T, H * BT)

    if isinstance(k, FakeTensor):
        return Aqk, Akk

    q_flat = q.reshape(B * T, H * K).contiguous()
    k_flat = k.reshape(B * T, H * K).contiguous()
    g_flat = gk.reshape(B * T, H * K).contiguous()
    beta_flat = beta.reshape(B * T, H).contiguous()
    akkd_flat = Akkd.reshape(B * T, H * BC).contiguous()
    if AkkOD is None:
        akk_od = torch.empty(NT * 6, H * BC * BC, device=k.device, dtype=torch.float32)
    else:
        assert AkkOD.shape == (NT * 6, H * BC * BC), (
            f"AkkOD must have shape {(NT * 6, H * BC * BC)}, got {tuple(AkkOD.shape)}"
        )
        akk_od = AkkOD

    m_q = _to_cute_tensor(q_flat)
    m_k = _to_cute_tensor(k_flat)
    m_g = _to_cute_tensor(g_flat)
    m_beta = _to_cute_tensor(beta_flat)
    m_aqk = _to_cute_tensor(Aqk_flat)
    m_akk_od = _to_cute_tensor(akk_od)
    m_akkd = _to_cute_tensor(akkd_flat)
    m_akk = _to_cute_tensor(Akk_flat)

    m_cu_seqlens = None
    m_chunk_indices = None
    if cu_seqlens is not None:
        assert chunk_indices is not None
        cu_seqlens_i32 = cu_seqlens.to(torch.int32).contiguous()
        chunk_indices_i32 = chunk_indices.to(torch.int32).flatten().contiguous()
        m_cu_seqlens = _to_cute_tensor(cu_seqlens_i32, assumed_align=4)
        m_chunk_indices = _to_cute_tensor(chunk_indices_i32, assumed_align=4)

    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    k3b = ChunkKDAFwdK3bOffdiagCuteDSL(BC=BC, D=K)
    # The repo's diagonal sub-chunk stage already stores (I - Akkd)^-1. The
    # standalone prototype passed raw diagonal blocks, so K4b's default mode
    # performs that diagonal forward substitution internally. Use the
    # pre-inverted mode here to avoid inverting/sign-flipping Akkd twice.
    k4b = ChunkKDAFwdK4bInverseCuteDSL(BC=BC, BK=64, fwd_sub_mode="preinverted")
    k3b(
        m_q,
        m_k,
        m_g,
        m_beta,
        m_aqk,
        m_akk_od,
        cutlass.Float32(scale),
        int(H),
        int(NT),
        stream,
        cu_seqlens=m_cu_seqlens,
        chunk_indices=m_chunk_indices,
    )
    k4b(
        m_akk_od,
        m_akkd,
        m_akk,
        int(H),
        int(NT),
        stream,
        cu_seqlens=m_cu_seqlens,
        chunk_indices=m_chunk_indices,
    )
    return Aqk, Akk


__all__ = ["chunk_kda_fwd_inter_solve_cute"]
