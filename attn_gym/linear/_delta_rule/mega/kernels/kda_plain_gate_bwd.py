# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TMA-staged dense reverse cumsum for plain KDA gate gradients.

For each BT64 chunk, the forward scan computes
``c[t] = log2(e) * sum(g[j], j <= t)``. Its backward is therefore the channelwise suffix scan
``dg[t] = log2(e) * sum(dc[j], j >= t)`` over the same chunk. Each CTA owns one
``(batch, chunk, head)`` tile: a producer warp streams four FP32 token rows per TMA stage, while
128 channel threads consume the stages in reverse and carry one suffix value across the chunk.
This is the dense counterpart of the Triton packed reverse scan and keeps the composed Mega
backward on CuTeDSL without materializing a transpose or per-channel work table.
"""

from __future__ import annotations

import math
from enum import IntEnum

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute, pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute import (
    TMA_ALIGNMENT_BYTES,
    compile_tvm_ffi,
    jit_cache,
    make_fake_strided_tensor,
)
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import requires_int64_abi


class WarpRole(IntEnum):
    """Fixed warp assignments for the dense gate scan."""

    TMA_PRODUCER = 0


class PlainGateBwdOp:
    """Scan one dense BT64 gate-gradient tile per CTA."""

    chunk_size = 64
    tokens_per_stage = 4
    stages = 2

    def __init__(self, heads: int, head_dim: int, use_int64_offsets: bool):
        if head_dim != 128:
            raise ValueError(f"plain gate backward requires D=128, got {head_dim}")
        self.heads = heads
        self.head_dim = head_dim
        self.use_int64_offsets = use_int64_offsets

    def get_name(self) -> str:
        return (
            f"kda_plain_gate_bwd_tma_h{self.heads}_d{self.head_dim}"
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
        tma_atom: cute.CopyAtom,
        tGgD: cute.Tensor,
        tGsD: cute.Tensor,
        tile_index: Int32,
    ):
        tile_pipeline.producer_acquire(producer_state)
        cute.copy(
            tma_atom,
            tGgD[(None, tile_index, 0)],
            tGsD[(None, producer_state.index)],
            tma_bar_ptr=tile_pipeline.producer_get_barrier(producer_state),
        )

    @cute.kernel
    def kernel(
        self,
        mD_cumulative: cute.Tensor,
        mD_gate: cute.Tensor,
        tma_atom: cute.CopyAtom,
        tma_tensor: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        chunk, head, batch = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        subtiles = self.chunk_size // self.tokens_per_stage
        tile_start = chunk.to(Int64) * Int64(self.chunk_size)

        smem = cutlass.utils.SmemAllocator()
        barriers = smem.allocate_array(Int64, self.stages * 2)
        sD = smem.allocate_tensor(Float32, self._staged_layout(), byte_alignment=128)
        tile_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.head_dim // 32,
            ),
            tx_count=cute.size_in_bytes(
                Float32,
                cute.make_layout(
                    (self.tokens_per_stage, self.head_dim),
                    stride=(self.head_dim, 1),
                ),
            ),
            barrier_storage=barriers,
            tidx=tidx,
        )

        cta_tiler = (self.tokens_per_stage, self.head_dim)
        gD_tiles = cute.local_tile(
            tma_tensor[None, None, head, batch],
            cta_tiler,
            (None, None),
        )
        tGsD, tGgD = cpasync.tma_partition(
            tma_atom,
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
            cpasync.prefetch_descriptor(tma_atom)
            for issued in cutlass.range_constexpr(self.stages):
                self._issue_subtile(
                    tile_pipeline,
                    producer_state,
                    tma_atom,
                    tGgD,
                    tGsD,
                    tile_base + Int32(subtiles - 1 - issued),
                )
                producer_state.advance()

        suffix = Float32(0.0)
        scale = Float32(math.log2(math.e))
        for step in cutlass.range_constexpr(subtiles):
            subtile = subtiles - 1 - step
            tile_pipeline.consumer_wait(consumer_state)
            stage = consumer_state.index
            for offset in cutlass.range_constexpr(self.tokens_per_stage):
                slot = self.tokens_per_stage - 1 - offset
                local_token = subtile * self.tokens_per_stage + slot
                suffix = suffix + sD[slot, tidx, stage]
                mD_gate[batch, tile_start + Int64(local_token), head, tidx] = suffix * scale
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            tile_pipeline.consumer_release(consumer_state)
            consumer_state.advance()
            if cutlass.const_expr(subtile >= self.stages) and warp_idx == WarpRole.TMA_PRODUCER:
                self._issue_subtile(
                    tile_pipeline,
                    producer_state,
                    tma_atom,
                    tGgD,
                    tGsD,
                    tile_base + Int32(subtile - self.stages),
                )
                producer_state.advance()

    @cute.jit
    def __call__(
        self,
        mD_cumulative: cute.Tensor,
        mD_gate: cute.Tensor,
        stream: cuda.CUstream,
    ):
        d_view = cute.make_tensor(
            mD_cumulative.iterator,
            cute.select(mD_cumulative.layout, mode=[1, 3, 2, 0]),
        )
        load_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            load_op,
            d_view,
            self._staged_layout(),
            (self.tokens_per_stage, self.head_dim),
        )
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(mD_cumulative, mD_gate, tma_atom, tma_tensor).launch(
            grid=(
                cute.ceil_div(mD_cumulative.shape[1], self.chunk_size),
                self.heads,
                mD_cumulative.shape[0],
            ),
            block=(self.head_dim, 1, 1),
            stream=stream,
        )


@jit_cache
def _compile_plain_gate_bwd(heads: int, head_dim: int, use_int64_offsets: bool):
    target = get_compile_target()
    if target.device_type != "cuda" or target.capability is None or target.capability < (9, 0):
        raise ValueError(f"plain gate backward requires SM90+, got target={target}")
    op = PlainGateBwdOp(heads, head_dim, use_int64_offsets)
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    batch = sym_int()
    tokens = sym_int()
    source = make_fake_strided_tensor(
        Float32,
        (batch, tokens, heads, head_dim),
        stride_divisibility=TMA_ALIGNMENT_BYTES // 4,
        assumed_align=TMA_ALIGNMENT_BYTES,
        use_int64_strides=use_int64_offsets,
    )
    output = make_fake_compact_tensor(
        Float32,
        (batch, tokens, heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=TMA_ALIGNMENT_BYTES,
    )
    return compile_tvm_ffi(op, source, output)


def plain_gate_cumsum_dense_bwd_cute(d_cumulative: torch.Tensor) -> torch.Tensor:
    """Apply the dense BT64 reverse cumsum with a TMA-staged CuTeDSL kernel."""
    if d_cumulative.ndim != 4 or d_cumulative.shape[-1] != 128:
        raise ValueError("d_cumulative must have shape [B, T, H, 128]")
    if d_cumulative.dtype != torch.float32 or not d_cumulative.is_contiguous():
        raise TypeError("d_cumulative must be contiguous float32")
    if d_cumulative.shape[1] % 64:
        raise ValueError("dense plain gate backward requires T divisible by 64")
    with torch.cuda.device(d_cumulative.device):
        output = torch.empty_like(d_cumulative)
        compiled = _compile_plain_gate_bwd(
            d_cumulative.shape[2],
            d_cumulative.shape[3],
            requires_int64_abi(d_cumulative, output),
        )
        compiled(d_cumulative, output)
        return output


__all__ = ["plain_gate_cumsum_dense_bwd_cute"]
