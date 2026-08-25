# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TMA CuTeDSL backward for the Kimi bounded-gate transform.

For raw projection output ``r``, per-head parameter ``A_log``, per-channel bias ``b``,
and nonpositive lower bound ``L``, the model produces per-token natural-log decay as::

    z = r.float() + b
    a = exp(A_log)
    s = sigmoid(a * z)
    gate = L * s

The recurrent state is multiplied by ``exp(gate)``. This transform is pointwise in the
token dimension: sequence boundaries do not reset it, and it never forms cumulative or
log2 gates.

The matching forward uses direct vector loads and stores without shared-memory staging.
Given the natural-log cotangent ``d_gate``, this module computes::

    d_raw = d_gate * L * a * s * (1 - s)
    d_dt_bias = sum(d_raw, dim=(batch, token))
    dA_log = sum(d_raw * z, dim=(batch, token, channel))

One CTA owns one ``(batch, 32-token tile, head)`` tile, stages input-precision ``r`` and
FP32 ``d_gate`` with TMA, writes input-precision ``d_raw``, and emits deterministic FP32 parameter
partials. There is intentionally no ``log2(e)`` factor or reverse scan.
"""

from __future__ import annotations

from enum import IntEnum

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute, pipeline
from cutlass.cute.nvgpu import cpasync

from attn_gym._backends.cute import (
    TMA_ALIGNMENT_BYTES,
    compile_tvm_ffi,
    jit_cache,
    make_fake_strided_tensor,
    tensor_supports_tma,
)
from attn_gym._backends.cute.device import cta_reduce_sum
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear.kda.fwd.cute.gate_fwd import _BOUND_GATE_DTYPES, _BoundGateDType
from attn_gym.utils import ceildiv

_TILE_TOKENS = 32


class _WarpRole(IntEnum):
    """Warp roles used by the TMA pipeline."""

    TMA_PRODUCER = 0


class _BoundedGateBwdTmaOp:
    """TMA-staged bounded-gate backward without a sequence scan."""

    def __init__(
        self,
        dtype: _BoundGateDType,
        heads: int,
        head_dim: int,
        lower_bound: float,
        fastmath: bool,
        flatten_batch: bool,
        use_int64_offsets: bool = False,
    ):
        self.dtype = dtype
        self.heads = heads
        self.head_dim = head_dim
        self.chunk_size = _TILE_TOKENS
        self.lower_bound = lower_bound
        self.fastmath = fastmath
        self.flatten_batch = flatten_batch
        self.use_int64_offsets = use_int64_offsets
        self.tokens_per_stage = 8
        subtiles = self.chunk_size // self.tokens_per_stage
        # Every TMA stage must begin at a 128-byte-aligned shared address.
        stage_bytes = self.tokens_per_stage * head_dim * self.dtype.cute_type.width // 8
        self.stages = min(2, subtiles) if stage_bytes % 128 == 0 else 1

    def get_name(self) -> str:
        """Return a stable profiler and artifact name for this specialization."""
        lower_bound = self.lower_bound.hex().replace("-", "m").replace("+", "p").replace(".", "_")
        return (
            f"kda_bound_gate_bwd_{self.dtype.name}_h{self.heads}_d{self.head_dim}"
            f"_bt{self.chunk_size}"
            f"_lb{lower_bound}_tma_tps{self.tokens_per_stage}_s{self.stages}"
            f"_fb{int(self.flatten_batch)}_fm{int(self.fastmath)}"
            f"_i64{int(self.use_int64_offsets)}"
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
        """Differentiate one asynchronously TMA-staged token tile."""
        tidx, _, _ = cute.arch.thread_idx()
        chunk, head, batch = cute.arch.block_idx()
        if cutlass.const_expr(self.flatten_batch):
            chunks = cute.ceil_div(mG.shape[1], self.chunk_size)
            batch = chunk // chunks
            chunk = chunk % chunks
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
            self.dtype.cute_type,
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
            tx_count=cute.size_in_bytes(self.dtype.cute_type, tile)
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

        if warp_idx == _WarpRole.TMA_PRODUCER:
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
                    tile_base + Int32(issued),
                )
                producer_state.advance()

        dA_log = Float32(0.0)
        d_dt_bias = Float32(0.0)
        dg = Float32(0.0)
        gate_scale = cute.math.exp(
            mA_log[head].to(Float32),
            fastmath=self.fastmath,
        )
        gradient_scale = Float32(self.lower_bound) * gate_scale
        dt_bias = mDtBias[head, tidx].to(Float32)

        for step in cutlass.range_constexpr(subtiles):
            subtile = step
            tile_pipeline.consumer_wait(consumer_state)
            stage = consumer_state.index
            for offset in cutlass.range_constexpr(self.tokens_per_stage):
                slot = offset
                local_token = subtile * self.tokens_per_stage + slot
                if Int32(local_token) < valid:
                    z = sG[slot, tidx, stage].to(Float32) + dt_bias
                    sigmoid = Float32(1.0) / (
                        Float32(1.0) + cute.math.exp(-(gate_scale * z), fastmath=self.fastmath)
                    )
                    sigmoid_derivative = sigmoid + (-sigmoid) * sigmoid
                    dg = sD[slot, tidx, stage] * (gradient_scale * sigmoid_derivative)
                    dA_log = dA_log + dg * z
                    d_dt_bias = d_dt_bias + dg
                    mDg[batch, tile_start + Int64(local_token), head, tidx] = dg.to(
                        self.dtype.cute_type
                    )
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tile_pipeline.consumer_release(consumer_state)
            consumer_state.advance()

            if (
                cutlass.const_expr(subtile + self.stages < subtiles)
                and warp_idx == _WarpRole.TMA_PRODUCER
            ):
                self._issue_subtile(
                    tile_pipeline,
                    producer_state,
                    tma_atom_g,
                    tGgG,
                    tGsG,
                    tma_atom_d,
                    tDgD,
                    tDsD,
                    tile_base + Int32(subtile + self.stages),
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
        mD_gate: cute.Tensor,
        mDg: cute.Tensor,
        mDA_partial: cute.Tensor,
        mDDtBias_partial: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Build TMA descriptors and launch one CTA per batch, chunk, and head."""
        # TMA stages contiguous [token, channel] boxes, so reinterpret the same
        # [B, T, H, D] storage with the exact mode permutation [T, D, H, B].
        g_view = cute.make_tensor(mG.iterator, cute.select(mG.layout, mode=[1, 3, 2, 0]))
        d_view = cute.make_tensor(
            mD_gate.iterator,
            cute.select(mD_gate.layout, mode=[1, 3, 2, 0]),
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

        self.kernel.set_name_prefix(self.get_name())
        chunks = cute.ceil_div(mG.shape[1], self.chunk_size)
        batch_grid = 1 if cutlass.const_expr(self.flatten_batch) else mG.shape[0]
        batch_tiles = mG.shape[0] if cutlass.const_expr(self.flatten_batch) else 1
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
        ).launch(
            grid=(chunks * batch_tiles, self.heads, batch_grid),
            block=(self.head_dim, 1, 1),
            stream=stream,
        )


@jit_cache
def _compile_bounded_gate_bwd(
    dtype: _BoundGateDType,
    heads: int,
    head_dim: int,
    lower_bound: float,
    fastmath: bool,
    flatten_batch: bool,
    use_int64_offsets: bool = False,
):
    """Compile one fake-tensor TVM-FFI specialization."""
    op = _BoundedGateBwdTmaOp(
        dtype,
        heads,
        head_dim,
        lower_bound,
        fastmath,
        flatten_batch,
        use_int64_offsets=use_int64_offsets,
    )
    target = get_compile_target()
    if target.device_type != "cuda" or target.capability is None or target.capability < (9, 0):
        raise ValueError(
            f"bounded_gate_bwd requires TMA on CUDA capability >= 9.0; got target={target}"
        )

    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    batch = sym_int()
    tokens = sym_int()
    chunks = sym_int()

    alignment_elements = TMA_ALIGNMENT_BYTES * 8 // dtype.cute_type.width
    g = make_fake_strided_tensor(
        dtype.cute_type,
        (batch, tokens, heads, head_dim),
        stride_divisibility=alignment_elements,
        assumed_align=TMA_ALIGNMENT_BYTES,
        use_int64_strides=use_int64_offsets,
    )
    d_gate = make_fake_strided_tensor(
        cutlass.Float32,
        (batch, tokens, heads, head_dim),
        stride_divisibility=TMA_ALIGNMENT_BYTES // 4,
        assumed_align=TMA_ALIGNMENT_BYTES,
        use_int64_strides=use_int64_offsets,
    )
    dg = cute.runtime.make_fake_compact_tensor(
        dtype.cute_type,
        (batch, tokens, heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=TMA_ALIGNMENT_BYTES,
    )
    A_log = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (heads,),
        stride=(sym_int(),),
        assumed_align=4,
    )
    dt_bias = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (heads, head_dim),
        stride=(sym_int(), sym_int()),
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
        d_gate,
        dg,
        dA_partial,
        d_dt_bias_partial,
    )


def _bound_gate_bwd_cuda(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_gate: torch.Tensor,
    lower_bound: float,
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep compilation and the CuTeDSL launcher behind an opaque operator."""
    if not tensor_supports_tma(raw_gate):
        raw_gate = raw_gate.clone(memory_format=torch.contiguous_format)
    if not tensor_supports_tma(d_gate):
        d_gate = d_gate.clone(memory_format=torch.contiguous_format)

    d_raw_gate = torch.empty_like(raw_gate, memory_format=torch.contiguous_format)
    partial_shape = (
        raw_gate.shape[0],
        ceildiv(raw_gate.shape[1], _TILE_TOKENS),
        raw_gate.shape[2],
    )
    dA_log_partial = torch.empty(partial_shape, device=raw_gate.device, dtype=torch.float32)
    d_dt_bias_partial = torch.empty(
        (*partial_shape, raw_gate.shape[3]), device=raw_gate.device, dtype=torch.float32
    )
    dtype = _BOUND_GATE_DTYPES.get(raw_gate.dtype)
    if dtype is None:
        raise TypeError(f"unsupported raw_gate dtype: {raw_gate.dtype}")
    compiled = _compile_bounded_gate_bwd(
        dtype,
        raw_gate.shape[2],
        raw_gate.shape[3],
        lower_bound,
        fastmath,
        raw_gate.shape[0] > 65535,
        use_int64_offsets=requires_int64_abi(
            raw_gate,
            A_log,
            dt_bias,
            d_gate,
            d_raw_gate,
            dA_log_partial,
            d_dt_bias_partial,
        ),
    )
    compiled(
        raw_gate,
        A_log,
        dt_bias,
        d_gate,
        d_raw_gate,
        dA_log_partial,
        d_dt_bias_partial,
    )
    # Reducing the per-chunk partials keeps the post-pass off the full [B, T, H, D] tensors.
    return d_raw_gate, dA_log_partial, d_dt_bias_partial.sum((0, 1))
