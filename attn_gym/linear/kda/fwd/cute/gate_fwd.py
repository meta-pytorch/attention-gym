# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Direct-vector CuTeDSL forward for the Kimi bounded-gate transform.

Each CTA owns eight heads of one physical token. The 128-thread/8-value thread-value map
is::

    head = head_group * 8 + thread // 16
    channel = (thread % 16) * 8 + value

It covers ``8 * 128`` elements exactly, gives each thread one contiguous eight-value
load and one contiguous 32-byte FP32 store, and predicates only the final partial head
group. Token count and batch size are represented directly by grid dimensions, so no
sequence-length divisibility or token padding is required.

The launcher selects a compact 16-byte-aligned signature for the model path and a general
signature with a contiguous last mode, dynamic outer strides, and element-size alignment.
Only an input whose last mode is noncontiguous is materialized into a supported layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, cute

from attn_gym._backends.cute import (
    compile_tvm_ffi,
    jit_cache,
    make_fake_strided_tensor,
    tensor_supports_contiguous_dim,
)
from attn_gym._backends.cute.target import get_compile_target
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.utils import ceildiv

_HEAD_DIM = 128
_HEADS_PER_BLOCK = 8
_THREADS = 128
_VALUES_PER_THREAD = 8
_ALIGNMENT = 16


@dataclass(frozen=True)
class _BoundGateDType:
    """Map Torch storage to its CuTeDSL type and profiler tag."""

    cute_type: type[cutlass.Numeric]
    name: str


_BOUND_GATE_DTYPES = {
    torch.float16: _BoundGateDType(cutlass.Float16, "fp16"),
    torch.bfloat16: _BoundGateDType(cutlass.BFloat16, "bf16"),
    torch.float32: _BoundGateDType(cutlass.Float32, "fp32"),
}


class _BoundGateForward:
    """Apply one bounded-gate specialization with direct vector loads and stores."""

    def __init__(
        self,
        dtype: _BoundGateDType,
        heads: int,
        lower_bound: float,
        fastmath: bool,
        compact_layout: bool,
        flatten_batch: bool,
        use_int64_offsets: bool,
    ) -> None:
        self.dtype = dtype
        self.heads = heads
        self.head_groups = ceildiv(heads, _HEADS_PER_BLOCK)
        self.lower_bound = lower_bound
        self.fastmath = fastmath
        self.compact_layout = compact_layout
        self.flatten_batch = flatten_batch
        self.use_int64_offsets = use_int64_offsets

    def get_name(self) -> str:
        """Return a stable profiler name for this specialization."""
        lower_bound = self.lower_bound.hex().replace("-", "m").replace("+", "p").replace(".", "_")
        return (
            f"kda_bound_gate_fwd_{self.dtype.name}_h{self.heads}_d{_HEAD_DIM}"
            f"_v{_VALUES_PER_THREAD}"
            f"_lb{lower_bound}_c{int(self.compact_layout)}_fb{int(self.flatten_batch)}"
            f"_fm{int(self.fastmath)}_i64{int(self.use_int64_offsets)}"
        )

    @cute.kernel
    def kernel(
        self,
        raw_gate: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        gate: cute.Tensor,
    ) -> None:
        """Transform one token and up to eight heads."""
        thread, _, _ = cute.arch.thread_idx()
        block, _, batch = cute.arch.block_idx()
        head_group = block % self.head_groups
        token_or_token_batch = block // self.head_groups
        if cutlass.const_expr(self.flatten_batch):
            batch = token_or_token_batch // raw_gate.shape[1]
            token = token_or_token_batch % raw_gate.shape[1]
        else:
            token = token_or_token_batch
        head = head_group * _HEADS_PER_BLOCK + thread // 16
        channel_group = thread % 16

        if head < self.heads:
            raw_vector = cute.local_tile(
                raw_gate[batch, token, head, None],
                (_VALUES_PER_THREAD,),
                (channel_group,),
            )
            bias_vector = cute.local_tile(
                dt_bias[head, None],
                (_VALUES_PER_THREAD,),
                (channel_group,),
            )
            gate_vector = cute.local_tile(
                gate[batch, token, head, None],
                (_VALUES_PER_THREAD,),
                (channel_group,),
            )
            raw = raw_vector.load().to(Float32)
            bias = bias_vector.load()
            amplitude = cute.math.exp(A_log[head].to(Float32), fastmath=self.fastmath)
            sigmoid = Float32(1.0) / (
                Float32(1.0) + cute.math.exp(-(amplitude * (raw + bias)), fastmath=self.fastmath)
            )
            gate_vector.store(Float32(self.lower_bound) * sigmoid)

    @cute.jit
    def __call__(
        self,
        raw_gate: cute.Tensor,
        A_log: cute.Tensor,
        dt_bias: cute.Tensor,
        gate: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        """Launch one CTA per ``(head group, token, batch)`` coordinate."""
        self.kernel.set_name_prefix(self.get_name())
        batch_grid = 1 if cutlass.const_expr(self.flatten_batch) else raw_gate.shape[0]
        batch_tiles = raw_gate.shape[0] if cutlass.const_expr(self.flatten_batch) else 1
        self.kernel(raw_gate, A_log, dt_bias, gate).launch(
            grid=(self.head_groups * raw_gate.shape[1] * batch_tiles, 1, batch_grid),
            block=(_THREADS, 1, 1),
            stream=stream,
        )


def _fake_compact(dtype: type[cutlass.Numeric], shape: tuple[object, ...]):
    """Create a row-major compact tensor with the vectorized ABI alignment."""
    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=_ALIGNMENT,
    )


@jit_cache
def _compile_bound_gate_fwd(
    dtype: _BoundGateDType,
    heads: int,
    lower_bound: float,
    fastmath: bool,
    compact_layout: bool,
    flatten_batch: bool,
    use_int64_offsets: bool,
):
    """Compile one dynamic-batch/token TVM-FFI specialization."""
    target = get_compile_target()
    if target.device_type != "cuda" or target.capability is None or target.capability < (9, 0):
        raise ValueError(f"bound_gate requires CUDA capability >= 9.0; got target={target}")
    op = _BoundGateForward(
        dtype,
        heads,
        lower_bound,
        fastmath,
        compact_layout,
        flatten_batch,
        use_int64_offsets,
    )
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    batch = sym_int()
    tokens = sym_int()

    if compact_layout:
        raw_gate = _fake_compact(dtype.cute_type, (batch, tokens, heads, _HEAD_DIM))
        A_log = _fake_compact(Float32, (heads,))
        dt_bias = _fake_compact(Float32, (heads, _HEAD_DIM))
    else:
        raw_gate = make_fake_strided_tensor(
            dtype.cute_type,
            (batch, tokens, heads, _HEAD_DIM),
            use_int64_strides=use_int64_offsets,
        )
        A_log = make_fake_strided_tensor(
            Float32,
            (heads,),
            use_int64_strides=use_int64_offsets,
        )
        dt_bias = make_fake_strided_tensor(
            Float32,
            (heads, _HEAD_DIM),
            use_int64_strides=use_int64_offsets,
        )
    return compile_tvm_ffi(
        op,
        raw_gate,
        A_log,
        dt_bias,
        _fake_compact(Float32, (batch, tokens, heads, _HEAD_DIM)),
    )


def _bound_gate_fwd_cuda(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    fastmath: bool,
) -> torch.Tensor:
    """Normalize vector alignment and launch the CuTeDSL forward."""
    dtype = _BOUND_GATE_DTYPES.get(raw_gate.dtype)
    if dtype is None:
        raise TypeError(f"unsupported raw_gate dtype: {raw_gate.dtype}")
    inputs = (raw_gate, A_log, dt_bias)
    inputs = tuple(
        tensor
        if tensor_supports_contiguous_dim(
            tensor,
            alignment_bytes=tensor.element_size(),
        )
        else tensor.contiguous()
        for tensor in inputs
    )
    raw_gate, A_log, dt_bias = inputs
    compact_layout = all(
        tensor.is_contiguous()
        and tensor_supports_contiguous_dim(tensor, alignment_bytes=_ALIGNMENT)
        for tensor in inputs
    )
    gate = torch.empty(raw_gate.shape, device=raw_gate.device, dtype=torch.float32)
    compiled = _compile_bound_gate_fwd(
        dtype,
        raw_gate.shape[2],
        lower_bound,
        fastmath,
        compact_layout,
        raw_gate.shape[0] > 65535,
        requires_int64_abi(raw_gate, A_log, dt_bias, gate),
    )
    compiled(raw_gate, A_log, dt_bias, gate)
    return gate
