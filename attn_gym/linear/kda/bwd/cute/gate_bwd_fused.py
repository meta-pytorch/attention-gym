# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused CuTeDSL backward for KDA's bounded gate and chunk-local prefix sum.

One CTA owns one ``(batch, chunk, head)`` tile. Each thread owns one gate
channel, scans the cumulative-gate gradient backward in FP32, applies the
bounded gate derivative, and writes the raw-gate gradient. This removes the
full FP32 intermediate written by the standalone reverse-cumsum kernel and read by the
standalone gate-backward kernel.

Inputs are staged through a TMA pipeline on CUDA capability 9.0 or newer. The
static chunk size selects one, two, or four tokens per stage so every positive
chunk size uses the same TMA kernel without crossing chunk boundaries.
"""

from __future__ import annotations

import math
from enum import IntEnum
from numbers import Real
from typing import NamedTuple

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute, pipeline
from cutlass.cute.nvgpu import cpasync

from attn_gym._backends.cute import (
    TMA_ALIGNMENT_BYTES,
    ceildiv,
    compile_tvm_ffi,
    jit_cache,
    tensor_supports_tma,
)
from attn_gym._backends.cute.device import cta_reduce_sum
from attn_gym._backends.cute.target import get_compile_target


class WarpRole(IntEnum):
    """Warp roles used by the TMA pipeline."""

    TMA_PRODUCER = 0


class FusedGateBwdOutput(NamedTuple):
    """BF16 raw-gate gradient plus the FP32 parameter gradients.

    ``d_dt_bias`` is final; the A-log gradient is ``dA_partial.sum((0, 1))``.
    """

    dg: torch.Tensor
    dA_partial: torch.Tensor
    d_dt_bias: torch.Tensor


class FusedGateBwdOp:
    """TMA-staged fused reverse-cumsum and bounded-gate backward."""

    def __init__(
        self,
        heads: int,
        head_dim: int,
        chunk_size: int,
        lower_bound: float,
        fastmath: bool,
    ):
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive int, got {chunk_size!r}")
        if not isinstance(lower_bound, Real) or isinstance(lower_bound, bool):
            raise TypeError(f"lower_bound must be a real scalar, got {type(lower_bound).__name__}")
        lower_bound = float(lower_bound)
        if not math.isfinite(lower_bound):
            raise ValueError(f"lower_bound must be finite, got {lower_bound}")
        if not isinstance(fastmath, bool):
            raise TypeError(f"fastmath must be bool, got {type(fastmath).__name__}")
        if not isinstance(heads, int) or isinstance(heads, bool) or heads < 1:
            raise ValueError(f"heads must be a positive int, got {heads!r}")
        if (
            not isinstance(head_dim, int)
            or isinstance(head_dim, bool)
            or head_dim < 32
            or head_dim > 1024
            or head_dim % 32 != 0
        ):
            raise ValueError(f"head_dim must be a multiple of 32 in [32, 1024], got {head_dim}")
        self.heads = heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.lower_bound = lower_bound
        self.fastmath = fastmath
        self.tokens_per_stage = next(tokens for tokens in (4, 2, 1) if chunk_size % tokens == 0)
        subtiles = chunk_size // self.tokens_per_stage
        # Every TMA stage must begin at a 128-byte-aligned shared address.
        bf16_stage_bytes = self.tokens_per_stage * head_dim * 2
        self.stages = min(2, subtiles) if bf16_stage_bytes % 128 == 0 else 1

    def get_name(self) -> str:
        """Return a stable profiler and artifact name for this specialization."""
        lower_bound = self.lower_bound.hex().replace("-", "m").replace("+", "p").replace(".", "_")
        return (
            f"kda_fused_gate_bwd_h{self.heads}_d{self.head_dim}_bt{self.chunk_size}"
            f"_lb{lower_bound}_tma_tps{self.tokens_per_stage}_s{self.stages}"
            f"_fm{int(self.fastmath)}"
        )

    def _staged_layout(self):
        return cute.make_layout(
            (self.tokens_per_stage, self.head_dim, self.stages),
            stride=(
                self.head_dim,
                1,
                self.tokens_per_stage * self.head_dim,
            ),
        )

    @cute.jit
    def _issue_subtile(
        self,
        tile_pipeline: pipeline.PipelineTmaAsync,
        producer_state: pipeline.PipelineState,
        tma_atom_g: cute.CopyAtom,
        tGgG: cute.Tensor,
        tGsG: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        tDgD: cute.Tensor,
        tDsD: cute.Tensor,
        tile_index: Int32,
    ):
        tile_pipeline.producer_acquire(producer_state)
        barrier = tile_pipeline.producer_get_barrier(producer_state)
        cute.copy(
            tma_atom_g,
            tGgG[(None, tile_index, 0)],
            tGsG[(None, producer_state.index)],
            tma_bar_ptr=barrier,
        )
        cute.copy(
            tma_atom_d,
            tDgD[(None, tile_index, 0)],
            tDsD[(None, producer_state.index)],
            tma_bar_ptr=barrier,
        )

    @cute.kernel
    def kernel(
        self,
        mG: cute.Tensor,
        mA_log: cute.Tensor,
        mDtBias: cute.Tensor,
        mDg: cute.Tensor,
        mDA_partial: cute.Tensor,
        mDDtBias_partial: cute.Tensor,
        tma_atom_g: cute.CopyAtom,
        tma_tensor_g: cute.Tensor,
        tma_atom_d: cute.CopyAtom,
        tma_tensor_d: cute.Tensor,
    ):
        """Scan one chunk backward over asynchronously TMA-staged subtiles."""
        tidx, _, _ = cute.arch.thread_idx()
        chunk, head, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        subtiles = self.chunk_size // self.tokens_per_stage

        tile_start = chunk.to(Int64) * Int64(self.chunk_size)
        valid = cutlass.min(
            Int64(self.chunk_size),
            Int64(mG.shape[1]) - tile_start,
        ).to(Int32)

        smem = cutlass.utils.SmemAllocator()
        tile_bar = smem.allocate_array(Int64, self.stages * 2)
        warp_partials = smem.allocate_tensor(
            Float32,
            cute.make_layout(self.head_dim // 32),
            byte_alignment=16,
        )
        sG = smem.allocate_tensor(
            cutlass.BFloat16,
            self._staged_layout(),
            byte_alignment=128,
        )
        sD = smem.allocate_tensor(
            Float32,
            self._staged_layout(),
            byte_alignment=128,
        )

        tile = cute.make_layout(
            (self.tokens_per_stage, self.head_dim),
            stride=(self.head_dim, 1),
        )
        tile_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.head_dim // 32,
            ),
            tx_count=cute.size_in_bytes(cutlass.BFloat16, tile)
            + cute.size_in_bytes(Float32, tile),
            barrier_storage=tile_bar,
            tidx=tidx,
        )

        cta_tiler = (self.tokens_per_stage, self.head_dim)
        gG_tiles = cute.local_tile(tma_tensor_g[None, None, head, batch], cta_tiler, (None, None))
        gD_tiles = cute.local_tile(tma_tensor_d[None, None, head, batch], cta_tiler, (None, None))
        tGsG, tGgG = cpasync.tma_partition(
            tma_atom_g,
            0,
            cute.make_layout(1),
            cute.group_modes(sG, 0, 2),
            cute.group_modes(gG_tiles, 0, 2),
        )
        tDsD, tDgD = cpasync.tma_partition(
            tma_atom_d,
            0,
            cute.make_layout(1),
            cute.group_modes(sD, 0, 2),
            cute.group_modes(gD_tiles, 0, 2),
        )

        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.stages,
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.stages,
        )
        tile_base = chunk * Int32(subtiles)

        if warp_idx == WarpRole.TMA_PRODUCER:
            cpasync.prefetch_descriptor(tma_atom_g)
            cpasync.prefetch_descriptor(tma_atom_d)
            for issued in cutlass.range_constexpr(self.stages):
                self._issue_subtile(
                    tile_pipeline,
                    producer_state,
                    tma_atom_g,
                    tGgG,
                    tGsG,
                    tma_atom_d,
                    tDgD,
                    tDsD,
                    tile_base + Int32(subtiles - 1 - issued),
                )
                producer_state.advance()

        reverse_sum = Float32(0.0)
        dA_log = Float32(0.0)
        d_dt_bias = Float32(0.0)
        dg = Float32(0.0)
        gate_scale = cute.math.exp(
            mA_log[head].to(Float32),
            fastmath=self.fastmath,
        )
        gradient_scale = Float32(math.log2(math.e) * self.lower_bound) * gate_scale
        dt_bias = mDtBias[head, tidx].to(Float32)

        for step in cutlass.range_constexpr(subtiles):
            subtile = subtiles - 1 - step
            tile_pipeline.consumer_wait(consumer_state)
            stage = consumer_state.index
            for offset in cutlass.range_constexpr(self.tokens_per_stage):
                slot = self.tokens_per_stage - 1 - offset
                local_token = subtile * self.tokens_per_stage + slot
                if Int32(local_token) < valid:
                    reverse_sum = reverse_sum + sD[slot, tidx, stage]
                    z = sG[slot, tidx, stage].to(Float32) + dt_bias
                    sigmoid = Float32(1.0) / (
                        Float32(1.0) + cute.math.exp(-(gate_scale * z), fastmath=self.fastmath)
                    )
                    sigmoid_derivative = sigmoid + (-sigmoid) * sigmoid
                    dg = reverse_sum * (gradient_scale * sigmoid_derivative)
                    dA_log = dA_log + dg * z
                    d_dt_bias = d_dt_bias + dg
                    mDg[batch, tile_start + Int64(local_token), head, tidx] = dg.to(
                        cutlass.BFloat16
                    )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tile_pipeline.consumer_release(consumer_state)
            consumer_state.advance()

            if cutlass.const_expr(subtile >= self.stages) and warp_idx == WarpRole.TMA_PRODUCER:
                self._issue_subtile(
                    tile_pipeline,
                    producer_state,
                    tma_atom_g,
                    tGgG,
                    tGsG,
                    tma_atom_d,
                    tDgD,
                    tDsD,
                    tile_base + Int32(subtile - self.stages),
                )
                producer_state.advance()

        cta_sum = cta_reduce_sum(dA_log, warp_partials)
        if tidx == 0:
            mDA_partial[batch, chunk, head] = cta_sum
        mDDtBias_partial[batch, chunk, head, tidx] = d_dt_bias

    @cute.jit
    def __call__(
        self,
        mG: cute.Tensor,
        mA_log: cute.Tensor,
        mDtBias: cute.Tensor,
        mD_cumulative: cute.Tensor,
        mDg: cute.Tensor,
        mDA_partial: cute.Tensor,
        mDDtBias_partial: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Build TMA descriptors and launch one CTA per batch, chunk, and head."""
        g_view = cute.make_tensor(mG.iterator, cute.select(mG.layout, mode=[1, 3, 2, 0]))
        d_view = cute.make_tensor(
            mD_cumulative.iterator,
            cute.select(mD_cumulative.layout, mode=[1, 3, 2, 0]),
        )
        load_op = cpasync.CopyBulkTensorTileG2SOp()
        staged = self._staged_layout()
        cta_tiler = (self.tokens_per_stage, self.head_dim)
        tma_atom_g, tma_tensor_g = cpasync.make_tiled_tma_atom(
            load_op,
            g_view,
            staged,
            cta_tiler,
        )
        tma_atom_d, tma_tensor_d = cpasync.make_tiled_tma_atom(
            load_op,
            d_view,
            staged,
            cta_tiler,
        )

        self.kernel(
            mG,
            mA_log,
            mDtBias,
            mDg,
            mDA_partial,
            mDDtBias_partial,
            tma_atom_g,
            tma_tensor_g,
            tma_atom_d,
            tma_tensor_d,
            _name_prefix=self.get_name(),
        ).launch(
            grid=(cute.ceil_div(mG.shape[1], self.chunk_size), self.heads, mG.shape[0]),
            block=(self.head_dim, 1, 1),
            stream=stream,
        )


@jit_cache
def _compile_fused_gate_bwd(
    heads: int,
    head_dim: int,
    chunk_size: int,
    lower_bound: float,
    fastmath: bool,
):
    """Compile one fake-tensor TVM-FFI specialization."""
    op = FusedGateBwdOp(heads, head_dim, chunk_size, lower_bound, fastmath)
    target = get_compile_target()
    if target.device_type != "cuda" or target.capability is None or target.capability < (9, 0):
        raise ValueError(
            f"fused_gate_bwd requires TMA on CUDA capability >= 9.0; got target={target}"
        )

    batch = cute.sym_int()
    tokens = cute.sym_int()
    chunks = cute.sym_int()

    def strided_rows(dtype, alignment_elements: int):
        return cute.runtime.make_fake_tensor(
            dtype,
            (batch, tokens, heads, head_dim),
            stride=(
                cute.sym_int(divisibility=alignment_elements),
                cute.sym_int(divisibility=alignment_elements),
                cute.sym_int(divisibility=alignment_elements),
                1,
            ),
            assumed_align=TMA_ALIGNMENT_BYTES,
        )

    g = strided_rows(cutlass.BFloat16, TMA_ALIGNMENT_BYTES // 2)
    d_cumulative = strided_rows(cutlass.Float32, TMA_ALIGNMENT_BYTES // 4)
    dg = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (batch, tokens, heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=TMA_ALIGNMENT_BYTES,
    )
    A_log = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (heads,),
        stride=(cute.sym_int(),),
        assumed_align=4,
    )
    dt_bias = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (heads, head_dim),
        stride=(cute.sym_int(), cute.sym_int()),
        assumed_align=4,
    )
    dA_partial = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (batch, chunks, heads),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    d_dt_bias_partial = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (batch, chunks, heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    return compile_tvm_ffi(
        op,
        g,
        A_log,
        dt_bias,
        d_cumulative,
        dg,
        dA_partial,
        d_dt_bias_partial,
    )


def _validate_inputs(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
    fastmath: bool,
) -> FusedGateBwdOp:
    """Validate the runtime ABI and return its canonical specialization."""
    if g.ndim != 4:
        raise ValueError(f"g must have shape (B, T, H, D), got {tuple(g.shape)}")
    batch, tokens, heads, head_dim = g.shape
    if batch == 0 or tokens == 0 or heads == 0:
        raise ValueError(f"g must have nonempty shape (B, T, H, D), got {tuple(g.shape)}")
    if head_dim < 32 or head_dim > 1024 or head_dim % 32 != 0:
        raise ValueError(
            f"g head dimension must be a multiple of 32 in [32, 1024], got {head_dim}"
        )
    if g.dtype != torch.bfloat16:
        raise TypeError(f"g must be bfloat16, got {g.dtype}")
    if d_cumulative.shape != g.shape or d_cumulative.dtype != torch.float32:
        raise ValueError(
            f"d_cumulative must be float32 with shape {g.shape}, "
            f"got {tuple(d_cumulative.shape)} and {d_cumulative.dtype}"
        )
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        raise ValueError(
            f"A_log must be float32 with shape {(heads,)}, "
            f"got {tuple(A_log.shape)} and {A_log.dtype}"
        )
    if dt_bias.shape != (heads, head_dim) or dt_bias.dtype != torch.float32:
        raise ValueError(
            f"dt_bias must be float32 with shape {(heads, head_dim)}, "
            f"got {tuple(dt_bias.shape)} and {dt_bias.dtype}"
        )
    tensors = (g, A_log, dt_bias, d_cumulative)
    if not all(tensor.is_cuda and tensor.device == g.device for tensor in tensors):
        raise ValueError("all inputs must be CUDA tensors on the same device")
    if not all(tensor_supports_tma(tensor) for tensor in (g, d_cumulative)):
        raise ValueError(
            "fused_gate_bwd requires contiguous trailing dimensions and aligned outer strides"
        )
    return FusedGateBwdOp(heads, head_dim, chunk_size, lower_bound, fastmath)


@torch.library.custom_op("attn_gym::kda_fused_gate_bwd", mutates_args=())
def _fused_gate_bwd_custom_op(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep compilation and the CuTeDSL launcher behind an opaque operator."""
    g, d_cumulative = (
        tensor
        if tensor_supports_tma(tensor)
        else tensor.clone(memory_format=torch.contiguous_format)
        for tensor in (g, d_cumulative)
    )
    op = _validate_inputs(
        g,
        A_log,
        dt_bias,
        d_cumulative,
        chunk_size,
        lower_bound,
        fastmath,
    )
    dg = torch.empty(g.shape, dtype=g.dtype, device=g.device)
    partial_shape = (g.shape[0], ceildiv(g.shape[1], op.chunk_size), g.shape[2])
    dA_partial = torch.empty(partial_shape, device=g.device, dtype=torch.float32)
    d_dt_bias_partial = torch.empty(
        (*partial_shape, g.shape[3]), device=g.device, dtype=torch.float32
    )
    compiled = _compile_fused_gate_bwd(
        g.shape[2],
        g.shape[3],
        op.chunk_size,
        op.lower_bound,
        op.fastmath,
    )
    compiled(g, A_log, dt_bias, d_cumulative, dg, dA_partial, d_dt_bias_partial)
    # Reducing the per-chunk partials keeps the post-pass off the full [B, T, H, D] tensors.
    return dg, dA_partial, d_dt_bias_partial.sum((0, 1))


@_fused_gate_bwd_custom_op.register_fake
def _fused_gate_bwd_fake(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    chunk_size: int,
    lower_bound: float,
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe symbolic output metadata without invoking the compiler."""
    del A_log, lower_bound, fastmath
    partial_shape = (g.shape[0], ceildiv(g.shape[1], chunk_size), g.shape[2])
    return (
        g.new_empty(g.shape),
        d_cumulative.new_empty(partial_shape),
        dt_bias.new_empty(dt_bias.shape),
    )


def fused_gate_bwd(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_cumulative: torch.Tensor,
    *,
    chunk_size: int = 64,
    lower_bound: float = -5.0,
    fastmath: bool = False,
) -> FusedGateBwdOutput:
    """Run fused reverse-cumsum and bounded gate backward with TMA staging.

    ``g`` is the BF16 raw gate with shape ``[B, T, H, D]`` and ``d_cumulative`` is
    the matching FP32 gradient. Their aligned contiguous trailing dimensions are consumed
    directly; unsupported layouts use a compact fallback. Parameter tensors may have arbitrary
    strides. The result contains
    BF16 ``dg``, the final FP32 ``d_dt_bias``, and one FP32 ``dA_log`` partial per batch,
    static chunk, and head.

    ``chunk_size``, ``lower_bound``, and ``fastmath`` are compile-time
    specializations. ``fastmath`` defaults to ``False``. The operation requires
    CUDA capability 9.0 or newer, is a first-order backward leaf, and does not
    support higher-order autograd.
    """
    if not torch.compiler.is_compiling() and g.ndim == 4:
        op = FusedGateBwdOp(g.shape[2], g.shape[3], chunk_size, lower_bound, fastmath)
        chunk_size, lower_bound, fastmath = op.chunk_size, op.lower_bound, op.fastmath
    if torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (g, A_log, dt_bias, d_cumulative)
    ):
        raise RuntimeError("fused_gate_bwd does not support higher-order autograd")

    dg, dA_partial, d_dt_bias = _fused_gate_bwd_custom_op(
        g,
        A_log,
        dt_bias,
        d_cumulative,
        chunk_size,
        lower_bound,
        fastmath,
    )
    return FusedGateBwdOutput(dg, dA_partial, d_dt_bias)


__all__ = [
    "FusedGateBwdOp",
    "FusedGateBwdOutput",
    "WarpRole",
    "fused_gate_bwd",
]
