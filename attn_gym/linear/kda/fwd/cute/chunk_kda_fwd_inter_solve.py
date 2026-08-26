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
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear.kda.chunk_schedule import (
    RaggedChunkMetadata,
    ResolvedSchedule,
    ScheduleKind,
    ScheduleRequest,
    validate_schedule_request,
)
from attn_gym.linear.kda.chunk_scheduler import GridScheduler
from attn_gym.linear.kda.constants import DEFAULT_CHUNK_SIZE
from attn_gym.linear.kda.fwd.cute.chunk_kda_k3b_offdiag_cutedsl import (
    ChunkKDAFwdK3bOffdiagCuteDSL,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_k4b_inverse_cutedsl import (
    ChunkKDAFwdK4bInverseCuteDSL,
)
from attn_gym.linear.kda.fwd.cute.chunk_schedule import ChunkSchedule

_SUPPORTED_HEAD_DIM = 128
_SUPPORTED_CHUNK_SIZE = 64
_SUPPORTED_SUBCHUNK_SIZE = 16
_SUPPORTED_NUM_SUBCHUNKS = 4
# Short average sequences benefit from persistence as soon as the capacity grid
# exceeds one machine-sized wave. Longer recurrences retain the conservative
# three-wave crossover because the persistent chunk loop can otherwise lose.
_INTER_SOLVE_SHORT_AUTO_WAVES = 1
_INTER_SOLVE_LONG_AUTO_WAVES = 3
_IO_TYPES = {
    torch.float16: (cutlass.Float16, "fp16"),
    torch.bfloat16: (cutlass.BFloat16, "bf16"),
}


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


def _resolve_ragged_execution(
    tensor: torch.Tensor,
    metadata: RaggedChunkMetadata,
    request: ScheduleRequest,
) -> ResolvedSchedule:
    """Resolve a real launch or provide the unused fake-tensor placeholder plan."""
    validate_schedule_request(request)
    if isinstance(tensor, FakeTensor):
        return ResolvedSchedule(ScheduleKind.STATIC, 0, metadata.capacity)
    resolved = GridScheduler(metadata).resolve_chunk(request, tensor.device)
    num_sequences = metadata.cu_seqlens.shape[0] - 1
    short_average = tensor.shape[1] <= num_sequences * _SUPPORTED_CHUNK_SIZE
    auto_waves = _INTER_SOLVE_SHORT_AUTO_WAVES if short_average else _INTER_SOLVE_LONG_AUTO_WAVES
    if (
        request is ScheduleRequest.AUTO
        and resolved.kind is ScheduleKind.STATIC
        and metadata.capacity > auto_waves * resolved.workers
    ):
        return ResolvedSchedule(ScheduleKind.PERSISTENT, resolved.workers, metadata.capacity)
    return resolved


def _make_fake_chunk_index(
    chunk_schedule: ChunkSchedule,
    sequences: int,
):
    """Create one optional fake ``[N+1]`` routing tensor."""
    if chunk_schedule is ChunkSchedule.DENSE:
        return None
    return make_fake_compact_tensor(
        cutlass.Int32,
        (sequences,),
        stride_order=(0,),
        assumed_align=4,
    )


@jit_cache
def _compile_k3b(
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
    """Compile K3b for independent layout and execution schedules."""
    _check_compile_target()
    offdiag_blocks = _validate_specialization(head_dim, chunk_size, subchunk_size)
    num_subchunks = chunk_size // subchunk_size
    op = ChunkKDAFwdK3bOffdiagCuteDSL(
        BC=subchunk_size,
        D=head_dim,
        chunk_size=chunk_size,
        num_subchunks=num_subchunks,
        chunk_schedule=chunk_schedule,
        schedule_kind=schedule_kind,
        chunk_workers=chunk_workers,
        use_int64_offsets=use_int64_offsets,
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
            f"kda_fwd_k3b_h{heads}_d{head_dim}_c{chunk_size}_sc{subchunk_size}_{io_name}"
            f"_layout_{chunk_schedule.value}_execution_{schedule_kind.value}"
            f"_cw{chunk_workers}_i64{int(use_int64_offsets)}"
        ),
    )


@jit_cache
def _compile_k4b(
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
    op = ChunkKDAFwdK4bInverseCuteDSL(
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
            f"kda_fwd_k4b_h{heads}_d{head_dim}_c{chunk_size}_sc{subchunk_size}_{io_name}"
            f"_layout_{chunk_schedule.value}_execution_{schedule_kind.value}"
            f"_cw{chunk_workers}_i64{int(use_int64_offsets)}"
        ),
    )


def _chunk_kda_fwd_k3b_ragged_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata,
    resolved: ResolvedSchedule,
    subchunk_size: int = _SUPPORTED_SUBCHUNK_SIZE,
    AkkOD: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute K3 with an execution plan shared by composed ragged stages.

    Persistent execution defines only the active ``AkkOD`` rows.
    """
    batch, tokens, heads, head_dim = k.shape
    chunk_size = _SUPPORTED_CHUNK_SIZE
    metadata.validate_chunk_size(chunk_size)
    offdiag_blocks = _validate_specialization(head_dim, chunk_size, subchunk_size)
    if batch != 1:
        raise ValueError(f"ragged K3 requires B=1, got {batch}")
    if Aqk.shape != (batch, tokens, heads, chunk_size):
        raise ValueError("Aqk has an incompatible shape")

    akk_od_shape = (metadata.capacity * offdiag_blocks, heads * subchunk_size**2)
    if AkkOD is None:
        AkkOD = torch.empty(akk_od_shape, device=k.device, dtype=torch.float32)
    elif AkkOD.shape != akk_od_shape:
        raise ValueError(f"AkkOD must have shape {akk_od_shape}, got {tuple(AkkOD.shape)}")
    if isinstance(k, FakeTensor):
        return Aqk, AkkOD
    if tokens == 0:
        AkkOD.zero_()
        return Aqk, AkkOD

    if q.dtype != k.dtype or q.dtype != Aqk.dtype or q.dtype not in _IO_TYPES:
        raise TypeError("q, k, and Aqk must share dtype float16 or bfloat16")
    io_type, io_name = _IO_TYPES[q.dtype]
    q_flat = q[0].reshape(tokens, heads * head_dim)
    k_flat = k[0].reshape(tokens, heads * head_dim)
    g_flat = gk.reshape(tokens, heads * head_dim).contiguous()
    beta_flat = beta.reshape(tokens, heads).contiguous()
    Aqk_flat = Aqk.reshape(tokens, heads * chunk_size)
    chunk_workers = resolved.workers if resolved.kind is ScheduleKind.PERSISTENT else 0
    k3b = _compile_k3b(
        heads,
        head_dim,
        chunk_size,
        subchunk_size,
        io_type,
        io_name,
        ChunkSchedule.RAGGED,
        resolved.kind,
        chunk_workers,
        use_int64_offsets=requires_int64_abi(q_flat, k_flat, g_flat, beta_flat, Aqk_flat, AkkOD),
    )
    k3b(
        q_flat,
        k_flat,
        g_flat,
        beta_flat,
        Aqk_flat,
        AkkOD,
        cutlass.Float32(scale),
        heads,
        metadata.capacity,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
    )
    return Aqk, AkkOD


def chunk_kda_fwd_k3b_ragged_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eager ragged K3 with functional input semantics."""
    resolved = _resolve_ragged_execution(k, metadata, schedule)
    return _chunk_kda_fwd_k3b_ragged_impl(q, k, gk, beta, Aqk.clone(), scale, metadata, resolved)


def _chunk_kda_fwd_k4b_ragged_impl(
    AkkOD: torch.Tensor,
    Akkd: torch.Tensor,
    metadata: RaggedChunkMetadata,
    resolved: ResolvedSchedule,
    subchunk_size: int = _SUPPORTED_SUBCHUNK_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute K4 with the same execution plan that produced ``AkkOD``."""
    if Akkd.ndim != 4:
        raise ValueError(f"Akkd must be 4D, got shape {tuple(Akkd.shape)}")
    batch, tokens, heads, diagonal_width = Akkd.shape
    chunk_size = _SUPPORTED_CHUNK_SIZE
    metadata.validate_chunk_size(chunk_size)
    offdiag_blocks = _validate_specialization(
        _SUPPORTED_HEAD_DIM,
        chunk_size,
        subchunk_size,
    )
    if batch != 1:
        raise ValueError(f"ragged K4 requires B=1, got {batch}")
    if diagonal_width != subchunk_size:
        raise ValueError(
            f"Akkd must have trailing dimension {subchunk_size}, got {diagonal_width}"
        )
    akk_od_shape = (metadata.capacity * offdiag_blocks, heads * subchunk_size**2)
    if AkkOD.shape != akk_od_shape:
        raise ValueError(f"AkkOD must have shape {akk_od_shape}, got {tuple(AkkOD.shape)}")

    # Only rows owned by active ragged chunks are defined; downstream ragged consumers route from
    # metadata and must not read capacity slack.
    if output_dtype not in _IO_TYPES:
        raise TypeError("Akk output dtype must be float16 or bfloat16")
    io_type, io_name = _IO_TYPES[output_dtype]
    Akk = torch.empty(
        (batch, tokens, heads, chunk_size),
        dtype=output_dtype,
        device=Akkd.device,
    )
    if isinstance(Akkd, FakeTensor) or metadata.capacity == 0:
        return Akk

    chunk_workers = resolved.workers if resolved.kind is ScheduleKind.PERSISTENT else 0
    akkd_flat = Akkd.reshape(tokens, heads * subchunk_size).contiguous()
    akk_flat = Akk.reshape(tokens, heads * chunk_size)
    k4b = _compile_k4b(
        heads,
        _SUPPORTED_HEAD_DIM,
        chunk_size,
        subchunk_size,
        io_type,
        io_name,
        ChunkSchedule.RAGGED,
        resolved.kind,
        chunk_workers,
        use_int64_offsets=requires_int64_abi(AkkOD, akkd_flat, akk_flat),
    )
    k4b(
        AkkOD,
        akkd_flat,
        akk_flat,
        heads,
        metadata.capacity,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
    )
    return Akk


def chunk_kda_fwd_k4b_ragged_cute(
    AkkOD: torch.Tensor,
    Akkd: torch.Tensor,
    metadata: RaggedChunkMetadata,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Run the eager ragged K4 inverse stage."""
    resolved = _resolve_ragged_execution(Akkd, metadata, schedule)
    return _chunk_kda_fwd_k4b_ragged_impl(
        AkkOD, Akkd, metadata, resolved, output_dtype=output_dtype
    )


def chunk_kda_fwd_k4b_dense_cute(
    AkkOD: torch.Tensor,
    Akkd: torch.Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Assemble dense BT=64 inverse blocks from precomputed K3-compatible factors."""
    if Akkd.ndim != 4:
        raise ValueError(f"Akkd must be 4D, got shape {tuple(Akkd.shape)}")
    batch, tokens, heads, subchunk_size = Akkd.shape
    offdiag_blocks = _validate_specialization(
        _SUPPORTED_HEAD_DIM,
        chunk_size,
        subchunk_size,
    )
    if batch != 1 or tokens % chunk_size:
        raise ValueError("dense K4 requires B=1 and complete chunks")
    chunks = tokens // chunk_size
    expected_od = (chunks * offdiag_blocks, heads * subchunk_size**2)
    if AkkOD.shape != expected_od:
        raise ValueError(f"AkkOD must have shape {expected_od}, got {tuple(AkkOD.shape)}")

    if output_dtype not in _IO_TYPES:
        raise TypeError("Akk output dtype must be float16 or bfloat16")
    io_type, io_name = _IO_TYPES[output_dtype]
    Akk = torch.empty(
        (batch, tokens, heads, chunk_size),
        dtype=output_dtype,
        device=Akkd.device,
    )
    if isinstance(Akkd, FakeTensor) or chunks == 0:
        return Akk
    akkd_flat = Akkd.reshape(tokens, heads * subchunk_size).contiguous()
    akk_flat = Akk.reshape(tokens, heads * chunk_size)
    k4b = _compile_k4b(
        heads,
        _SUPPORTED_HEAD_DIM,
        chunk_size,
        subchunk_size,
        io_type,
        io_name,
        ChunkSchedule.DENSE,
        ScheduleKind.STATIC,
        use_int64_offsets=requires_int64_abi(AkkOD, akkd_flat, akk_flat),
    )
    k4b(AkkOD, akkd_flat, akk_flat, heads, chunks, None, None)
    return Akk


def chunk_kda_fwd_inter_solve_ragged_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Akkd: torch.Tensor,
    Aqk: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run K3 and K4 with one shared automatic scheduling decision."""
    resolved = _resolve_ragged_execution(k, metadata, ScheduleRequest.AUTO)
    Aqk, AkkOD = _chunk_kda_fwd_k3b_ragged_impl(q, k, gk, beta, Aqk, scale, metadata, resolved)
    return Aqk, _chunk_kda_fwd_k4b_ragged_impl(
        AkkOD, Akkd, metadata, resolved, output_dtype=q.dtype
    )


def chunk_kda_fwd_inter_solve_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Akkd: torch.Tensor,
    scale: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
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
    assert T % BT == 0, (
        f"fixed-length KDA inter-solve requires complete chunks, got tokens={T}, chunk_size={BT}"
    )
    NT = T // BT

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

    if q.dtype != k.dtype or q.dtype not in _IO_TYPES:
        raise TypeError("q and k must share dtype float16 or bfloat16")
    if Aqk.dtype != q.dtype or Akk.dtype != q.dtype:
        raise TypeError("Aqk and Akk must match the Q/K dtype")
    io_type, io_name = _IO_TYPES[q.dtype]
    if isinstance(k, FakeTensor):
        return Aqk, Akk

    # Q/K heads are compact, but token rows may retain a packed-projection pitch.
    q_flat = q[0].reshape(T, H * K)
    k_flat = k[0].reshape(T, H * K)
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

    with (
        torch.profiler.record_function("kda/cute/k3b_offdiag") if profile_ranges else nullcontext()
    ):
        k3b = _compile_k3b(
            H,
            K,
            BT,
            BC,
            io_type,
            io_name,
            ChunkSchedule.DENSE,
            ScheduleKind.STATIC,
            use_int64_offsets=requires_int64_abi(
                q_flat, k_flat, g_flat, beta_flat, Aqk_flat, akk_od
            ),
        )
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
            None,
            None,
        )
    with (
        torch.profiler.record_function("kda/cute/k4b_inverse") if profile_ranges else nullcontext()
    ):
        k4b = _compile_k4b(
            H,
            K,
            BT,
            BC,
            io_type,
            io_name,
            ChunkSchedule.DENSE,
            ScheduleKind.STATIC,
            use_int64_offsets=requires_int64_abi(akk_od, akkd_flat, Akk_flat),
        )
        k4b(
            akk_od,
            akkd_flat,
            Akk_flat,
            H,
            NT,
            None,
            None,
        )
    return Aqk, Akk


__all__ = [
    "chunk_kda_fwd_inter_solve_cute",
    "chunk_kda_fwd_inter_solve_ragged_cute",
    "chunk_kda_fwd_k3b_ragged_cute",
    "chunk_kda_fwd_k4b_dense_cute",
    "chunk_kda_fwd_k4b_ragged_cute",
]
