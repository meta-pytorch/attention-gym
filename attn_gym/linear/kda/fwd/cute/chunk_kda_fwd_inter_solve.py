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

from contextlib import nullcontext

import cutlass
import torch
import triton
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.target import get_compile_target
from attn_gym.linear.kda.fwd.cute.chunk_kda_k3b_offdiag_cutedsl import (
    ChunkKDAFwdK3bOffdiagCuteDSL,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_k4b_inverse_cutedsl import (
    ChunkKDAFwdK4bInverseCuteDSL,
)
from attn_gym.linear.kda.utils import (
    DEFAULT_CHUNK_SIZE,
    prepare_chunk_indices,
)

_SUPPORTED_HEAD_DIM = 128
_SUPPORTED_CHUNK_SIZE = 64
_SUPPORTED_SUBCHUNK_SIZE = 16
_SUPPORTED_NUM_SUBCHUNKS = 4


def _check_compile_target() -> None:
    target = get_compile_target()
    if target.device_type != "cuda" or target.capability is None or target.capability < (10, 0):
        raise ValueError(f"KDA inter-solve requires CUDA capability >= 10.0; got target={target}")


def _validate_specialization(head_dim: int, chunk_size: int, subchunk_size: int) -> int:
    assert head_dim == _SUPPORTED_HEAD_DIM, (
        f"KDA inter-solve requires head_dim={_SUPPORTED_HEAD_DIM}, got {head_dim}"
    )
    assert chunk_size == _SUPPORTED_CHUNK_SIZE, (
        f"KDA inter-solve requires chunk_size={_SUPPORTED_CHUNK_SIZE}, got {chunk_size}"
    )
    assert subchunk_size == _SUPPORTED_SUBCHUNK_SIZE, (
        f"KDA inter-solve requires subchunk_size={_SUPPORTED_SUBCHUNK_SIZE}, got {subchunk_size}"
    )
    assert chunk_size % subchunk_size == 0
    num_subchunks = chunk_size // subchunk_size
    assert num_subchunks == _SUPPORTED_NUM_SUBCHUNKS
    return num_subchunks * (num_subchunks - 1) // 2


@jit_cache
def _compile_k3b(
    heads: int,
    head_dim: int,
    chunk_size: int,
    subchunk_size: int,
    varlen: bool,
):
    """Compile one persistent K3b TVM-FFI specialization."""
    _check_compile_target()
    offdiag_blocks = _validate_specialization(head_dim, chunk_size, subchunk_size)
    num_subchunks = chunk_size // subchunk_size
    op = ChunkKDAFwdK3bOffdiagCuteDSL(
        BC=subchunk_size,
        D=head_dim,
        chunk_size=chunk_size,
        num_subchunks=num_subchunks,
        varlen=varlen,
    )
    tokens, chunks, sequences = (cute.sym_int() for _ in range(3))
    q = make_fake_compact_tensor(
        cutlass.BFloat16,
        (tokens, heads * head_dim),
        stride_order=(1, 0),
        assumed_align=16,
    )
    k = make_fake_compact_tensor(
        cutlass.BFloat16,
        (tokens, heads * head_dim),
        stride_order=(1, 0),
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
        cutlass.BFloat16,
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
    cu_seqlens = make_fake_compact_tensor(
        cutlass.Int32,
        (sequences,),
        stride_order=(0,),
        assumed_align=4,
    )
    chunk_indices = make_fake_compact_tensor(
        cutlass.Int32,
        (chunks * 2,),
        stride_order=(0,),
        assumed_align=4,
    )
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
        chunk_indices,
        name=(f"kda_fwd_k3b_h{heads}_d{head_dim}_c{chunk_size}_sc{subchunk_size}_vl{int(varlen)}"),
    )


@jit_cache
def _compile_k4b(
    heads: int,
    head_dim: int,
    chunk_size: int,
    subchunk_size: int,
    varlen: bool,
):
    """Compile one persistent K4b TVM-FFI specialization."""
    _check_compile_target()
    offdiag_blocks = _validate_specialization(head_dim, chunk_size, subchunk_size)
    num_subchunks = chunk_size // subchunk_size
    op = ChunkKDAFwdK4bInverseCuteDSL(
        BC=subchunk_size,
        chunk_size=chunk_size,
        num_subchunks=num_subchunks,
        fwd_sub_mode="preinverted",
        varlen=varlen,
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
        cutlass.BFloat16,
        (tokens, heads * chunk_size),
        stride_order=(1, 0),
        assumed_align=16,
    )
    cu_seqlens = make_fake_compact_tensor(
        cutlass.Int32,
        (sequences,),
        stride_order=(0,),
        assumed_align=4,
    )
    chunk_indices = make_fake_compact_tensor(
        cutlass.Int32,
        (chunks * 2,),
        stride_order=(0,),
        assumed_align=4,
    )
    return compile_tvm_ffi(
        op,
        AkkOD,
        Akkd,
        Akk,
        heads,
        1,
        cu_seqlens,
        chunk_indices,
        name=(f"kda_fwd_k4b_h{heads}_d{head_dim}_c{chunk_size}_sc{subchunk_size}_vl{int(varlen)}"),
    )


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
    Aqk: torch.Tensor | None = None,
    Akk: torch.Tensor | None = None,
    AkkOD: torch.Tensor | None = None,
    profile_ranges: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert Akkd.ndim == 4, f"Akkd must be 4D, got shape {tuple(Akkd.shape)}"
    B, T, H, K = k.shape
    BT = chunk_size
    BC = Akkd.shape[-1]
    offdiag_blocks = _validate_specialization(K, BT, BC)
    assert B == 1, f"chunk_kda_fwd_inter_solve_cute requires B=1, got B={B}"
    assert Akkd.shape == (B, T, H, BC), (
        f"Akkd must have shape {(B, T, H, BC)}, got {tuple(Akkd.shape)}"
    )
    if cu_seqlens is None:
        assert T % BT == 0, (
            "fixed-length KDA inter-solve requires complete chunks, "
            f"got tokens={T}, chunk_size={BT}"
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
        Akk_flat = torch.zeros(B * T, H * BT, device=k.device, dtype=k.dtype)
        Akk = Akk_flat.reshape(B, T, H, BT)
    else:
        assert Akk.shape == (B, T, H, BT), (
            f"Akk must have shape {(B, T, H, BT)}, got {tuple(Akk.shape)}"
        )
        Akk.zero_()
        Akk_flat = Akk.reshape(B * T, H * BT)

    if isinstance(k, FakeTensor):
        return Aqk, Akk

    q_flat = q.reshape(B * T, H * K).contiguous()
    k_flat = k.reshape(B * T, H * K).contiguous()
    g_flat = gk.reshape(B * T, H * K).contiguous()
    beta_flat = beta.reshape(B * T, H).contiguous()
    akkd_flat = Akkd.reshape(B * T, H * BC).contiguous()
    akk_od_shape = (NT * offdiag_blocks, H * BC * BC)
    if AkkOD is None:
        akk_od = torch.empty(akk_od_shape, device=k.device, dtype=torch.float32)
    else:
        assert AkkOD.shape == akk_od_shape, (
            f"AkkOD must have shape {akk_od_shape}, got {tuple(AkkOD.shape)}"
        )
        akk_od = AkkOD

    varlen = cu_seqlens is not None
    if varlen:
        assert chunk_indices is not None
        cu_seqlens_i32 = cu_seqlens.to(torch.int32).contiguous()
        chunk_indices_i32 = chunk_indices.to(torch.int32).flatten().contiguous()
    else:
        cu_seqlens_i32 = torch.empty(1, dtype=torch.int32, device=k.device)
        chunk_indices_i32 = torch.empty(2, dtype=torch.int32, device=k.device)

    with (
        torch.profiler.record_function("kda/cute/k3b_offdiag") if profile_ranges else nullcontext()
    ):
        k3b = _compile_k3b(H, K, BT, BC, varlen)
        k3b(
            q_flat,
            k_flat,
            g_flat,
            beta_flat,
            Aqk_flat,
            akk_od,
            cutlass.Float32(scale),
            H,
            NT,
            cu_seqlens_i32,
            chunk_indices_i32,
        )
    with (
        torch.profiler.record_function("kda/cute/k4b_inverse") if profile_ranges else nullcontext()
    ):
        k4b = _compile_k4b(H, K, BT, BC, varlen)
        k4b(
            akk_od,
            akkd_flat,
            Akk_flat,
            H,
            NT,
            cu_seqlens_i32,
            chunk_indices_i32,
        )
    return Aqk, Akk


__all__ = ["chunk_kda_fwd_inter_solve_cute"]
