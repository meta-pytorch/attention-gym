# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Public KDA operations.

``chunk_kda`` supports training and prefill; ``recurrent_kda`` supports decode and
inference prefill. Use ``impl`` to select fused kernels or the eager reference.
"""

from __future__ import annotations

import math
from functools import partial
from numbers import Real

import torch

from attn_gym.linear.kda.impl.fused import chunk_forward as _fused_chunk_forward
from attn_gym.linear.kda.impl.reference import reference_kda
from attn_gym.linear.kda.naive import naive_chunk_kda_from_cumulative, naive_recurrent_kda
from attn_gym.linear.kda.ops import recurrent_decode_forward as _fused_recurrent_decode_forward
from attn_gym.linear.kda.ops import recurrent_forward as _fused_recurrent_forward
from attn_gym.linear.kda.validation import SUPPORTED_INPUT_DTYPES, validate_kda_inputs
from attn_gym.linear.types import Impl, resolve_impl

_CHUNK_SIZE = 64


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
    fastmath: bool = False,
    autotune: bool = True,
    impl: Impl | str = Impl.FUSED,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply chunk-parallel KDA for training and prefill.

    Args:
        q: Queries shaped ``[B, T, H, K]``, scaled by ``1/sqrt(K)`` internally.
        k: Keys shaped like ``q``.
        v: Values shaped ``[B, T, H, V]``.
        cumulative_gate: Inclusive cumulative log2 decay within each 64-token
            chunk, shaped like ``q`` and produced by
            ``bounded_gate_cumsum(chunk_size=64)``.
        beta: Per-token write gate shaped ``[B, T, H]``.
        initial_state: Starting recurrent state, with one ``[H, K, V]`` entry per
            logical sequence.
        cu_seqlens: Packed offsets shaped ``[N + 1]`` for batch-one inputs, as
            contiguous ``int32`` on ``q.device``; they start at zero, never
            decrease, may repeat for empty sequences whose states pass through,
            and may end before ``T``.
        output_final_state: Return the final recurrent state with the output.
        fastmath: Allow less precise fused math for speed; rejected with
            ``"reference"``.
        autotune: Benchmark candidate kernel configurations when true (winners
            are cached and reused); use fixed heuristics when false for
            repeatable selection across machines and cache states.
        impl: ``"fused"`` uses the Blackwell kernels with first-order autograd;
            ``"reference"`` uses differentiable eager PyTorch in FP32, with no
            automatic fallback.

    Returns:
        The output in ``q.dtype`` and either an FP32 final state with one entry
        per logical sequence or ``None``.
    """
    selected_impl = resolve_impl(impl)
    if selected_impl is Impl.REFERENCE and fastmath:
        raise ValueError("fastmath applies only to impl='fused'")
    validate_kda_inputs(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        op_name="chunk_kda",
        gate_name="cumulative_gate",
    )
    if selected_impl is Impl.FUSED:
        return _fused_chunk_forward(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
            fastmath=fastmath,
            autotune=autotune,
        )
    return reference_kda(
        partial(naive_chunk_kda_from_cumulative, chunk_size=_CHUNK_SIZE),
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        output_final_state,
    )


def _validate_paged_state(
    q: torch.Tensor,
    v: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor,
    output_final_state: bool,
) -> None:
    """Validate the paged state pool and its slot indices."""
    if initial_state is None:
        raise ValueError("state_indices requires initial_state as the paged state pool")
    if output_final_state:
        raise ValueError("state_indices advances the pool in place; drop output_final_state")
    if initial_state.ndim != 4 or initial_state.shape[1:] != (*q.shape[2:], v.shape[-1]):
        raise ValueError(
            "the paged state pool must have shape "
            f"[num_slots, {q.shape[2]}, {q.shape[3]}, {v.shape[-1]}], "
            f"got {tuple(initial_state.shape)}"
        )
    if initial_state.dtype != torch.float32:
        raise TypeError("the paged state pool must use float32")
    expected_inner_strides = (q.shape[3] * v.shape[-1], v.shape[-1], 1)
    if initial_state.stride()[1:] != expected_inner_strides:
        raise TypeError("the paged state pool must be contiguous within each [H, K, V] slot")
    if initial_state.stride(0) < q.shape[2] * q.shape[3] * v.shape[-1]:
        raise ValueError("paged state pool slots must not overlap")
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    if (
        tuple(state_indices.shape) != (num_sequences,)
        or state_indices.dtype != torch.int32
        or not state_indices.is_contiguous()
        or state_indices.device != q.device
    ):
        raise ValueError(
            f"state_indices must be a contiguous int32 tensor of shape ({num_sequences},) "
            f"on q.device, got {tuple(state_indices.shape)} of {state_indices.dtype}"
        )


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_indices: torch.Tensor | None = None,
    autotune: bool = True,
    impl: Impl | str = Impl.FUSED,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply recurrent KDA for decoding and inference prefill.

    Args:
        q: Queries shaped ``[B, T, H, K]``, scaled by ``1/sqrt(K)`` internally.
        k: Keys shaped like ``q``.
        v: Values shaped ``[B, T, H, V]``.
        gate: Per-token log2 decay shaped like ``q``, as produced by
            ``bounded_gate_cumsum(chunk_size=1)``; do not pass chunk-cumulative
            gates.
        beta: Per-token write gate shaped ``[B, T, H]``.
        initial_state: Starting recurrent state, with one ``[H, K, V]`` entry per
            logical sequence.
        cu_seqlens: Packed offsets shaped ``[N + 1]`` for batch-one inputs, as
            contiguous ``int32`` on ``q.device``; they start at zero, never
            decrease, may repeat for empty sequences whose states pass through,
            and may end before ``T``.
        output_final_state: Return the final recurrent state with the output. Rejected
            together with ``state_indices``, which advances the pool in place instead.
        state_indices: Contiguous ``int32`` slot indices, one per logical sequence,
            selecting rows of a paged ``initial_state`` pool shaped
            ``[num_slots, H, K, V]``. Each sequence reads and advances
            ``initial_state[state_indices[i]]`` **in place**. An index not in
            [1, num_slots) implies padding and are ignored by the kernel. The active
            indices also have to be unique to prevent two sequences from writing
            the same slot concurrently.
            Example:
                num_slots = 6
                state_indices = [3, -1, 5, 0]

                seq 0 reads and updates initial_state[3]
                seq 1 is padding (index is -1) so this is ignored
                seq 2 reads and updates initial_state[5]
                seq 3 is padding bc 0
        autotune: Reserved for implementation parity with ``chunk_kda``. The
            current recurrent kernel uses the same fixed launch policy for both
            values.
        impl: ``"fused"`` uses the inference-only optimized scan; ``"reference"``
            uses differentiable eager PyTorch in FP32, with no automatic
            fallback.

    Returns:
        The output in ``q.dtype`` and either an FP32 final state with one entry
        per logical sequence or ``None``.

    Serving limitations: without ``state_indices`` state rows map directly to logical
    sequences and final states are written out of place, decode preprocessing and scan are
    separate launches, and speculative-decoding rollback is unsupported.
    """
    del autotune
    selected_impl = resolve_impl(impl)
    # A paged pool's leading dimension is the slot count, not the sequence count, so the
    # shared per-sequence state check does not apply; the pool is checked below instead.
    validate_kda_inputs(
        q,
        k,
        v,
        gate,
        beta,
        None if state_indices is not None else initial_state,
        cu_seqlens,
        op_name="recurrent_kda",
        gate_name="gate",
    )
    if state_indices is not None:
        _validate_paged_state(q, v, initial_state, cu_seqlens, state_indices, output_final_state)
        if selected_impl is not Impl.FUSED:
            raise ValueError("state_indices requires impl='fused'")
    if selected_impl is Impl.FUSED:
        return _fused_recurrent_forward(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            output_final_state=output_final_state,
            state_indices=state_indices,
        )
    return reference_kda(
        naive_recurrent_kda, q, k, v, gate, beta, initial_state, cu_seqlens, output_final_state
    )


def recurrent_kda_decode(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    lower_bound: float | None = -5.0,
    scale: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one-token paged KDA decode with preprocessing fused into the recurrence.

    Args:
        packed_qkv: Post-convolution QKV shaped ``[B, H * (2 * K + V)]``. Each
            token stores ``[Q for all heads | K for all heads | V for all heads]``;
            within each section, head rows are contiguous.
        raw_gate: Unactivated gate shaped ``[1, B, H, K]``.
        raw_beta: Unactivated write gate shaped ``[1, B, H]``.
        A_log: Per-head log decay parameter shaped ``[H]``.
        dt_bias: Per-head/channel gate bias shaped ``[H, K]``.
        state_cache: FP32 paged state pool shaped ``[num_slots, H, V, K]``. Slots
            may have padding between them but each ``[H, V, K]`` row must be dense.
            ``K`` must be at most 256.
            The prefill kernel uses ``[H, K, V]``, while this decode kernel uses
            ``[H, V, K]`` because decode benefits significantly from a contiguous
            ``K`` dimension. A prefill-to-decode transition therefore requires a
            transpose/layout conversion.
        state_indices: Contiguous int32 slot indices shaped ``[B]``. Non-positive
            indices are padding/null entries: they produce zero output and leave the
            cache untouched. Each positive index must be in ``[1, num_slots)`` and
            unique among active rows because duplicate in-place updates race. These
            value constraints are caller responsibilities and are not host-validated.
        lower_bound: Negative bound for
            ``lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))``. When
            ``None``, use
            ``-exp(A_log) * softplus(raw_gate + dt_bias)`` instead.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        out: Optional caller-owned contiguous output buffer shaped ``[1, B, H, V]``
            in ``packed_qkv.dtype`` on the same device. When supplied, the kernel
            writes into and returns this exact tensor. It must not alias any input.

    Returns:
        Decode output shaped ``[1, B, H, V]`` in ``packed_qkv.dtype``. This is
        ``out`` itself when a buffer is supplied.

    Q/K L2 normalization, gate activation, beta sigmoid, the recurrent update, and
    the output projection from recurrent state are performed in one Triton kernel.
    The operation is inference-only and advances ``state_cache`` in place.
    """
    if packed_qkv.ndim != 2 or packed_qkv.shape[0] < 1 or packed_qkv.stride(1) != 1:
        raise ValueError("packed_qkv must have shape [B, C] and be contiguous within each token")
    if state_cache.ndim != 4:
        raise ValueError("state_cache must have shape [num_slots, H, V, K]")
    num_slots, heads, value_dim, key_dim = state_cache.shape
    batch = packed_qkv.shape[0]
    if num_slots < 1 or heads < 1 or key_dim < 1 or value_dim < 1:
        raise ValueError(
            f"state_cache must have nonempty dimensions, got {tuple(state_cache.shape)}"
        )
    expected_channels = heads * (2 * key_dim + value_dim)
    if packed_qkv.shape[1] != expected_channels:
        raise ValueError(
            f"packed_qkv must have shape ({batch}, {expected_channels}), "
            f"got {tuple(packed_qkv.shape)}"
        )
    if packed_qkv.stride(0) < expected_channels:
        raise ValueError("packed_qkv token rows must not overlap")
    if raw_gate.shape != (1, batch, heads, key_dim) or raw_gate.stride()[2:] != (key_dim, 1):
        raise ValueError(
            f"raw_gate must have shape {(1, batch, heads, key_dim)} and dense [H, K] rows"
        )
    if raw_gate.stride(1) < heads * key_dim:
        raise ValueError("raw_gate batch rows must not overlap")
    if raw_beta.shape != (1, batch, heads) or raw_beta.stride(2) != 1:
        raise ValueError(f"raw_beta must have shape {(1, batch, heads)} with contiguous heads")
    if raw_beta.stride(1) < heads:
        raise ValueError("raw_beta batch rows must not overlap")
    if A_log.shape != (heads,) or not A_log.is_contiguous():
        raise ValueError(f"A_log must be contiguous with shape ({heads},)")
    if dt_bias.shape != (heads, key_dim) or not dt_bias.is_contiguous():
        raise ValueError(f"dt_bias must be contiguous with shape ({heads}, {key_dim})")
    if state_cache.dtype != torch.float32:
        raise TypeError("state_cache must use float32")
    expected_state_strides = (value_dim * key_dim, key_dim, 1)
    if state_cache.stride()[1:] != expected_state_strides:
        raise TypeError("state_cache must be contiguous within each [H, V, K] slot")
    if state_cache.stride(0) < heads * key_dim * value_dim:
        raise ValueError("state_cache slots must not overlap")
    if (
        state_indices.shape != (batch,)
        or state_indices.dtype != torch.int32
        or not state_indices.is_contiguous()
    ):
        raise ValueError(f"state_indices must be contiguous int32 with shape ({batch},)")
    if key_dim > 256:
        raise ValueError(f"recurrent_kda_decode requires K in [1, 256], got {key_dim}")

    device = packed_qkv.device
    data_tensors = (packed_qkv, raw_gate, raw_beta, A_log, dt_bias)
    if any(tensor.device != device for tensor in (*data_tensors, state_cache, state_indices)):
        raise ValueError("all recurrent_kda_decode inputs must be on the same device")
    if any(tensor.dtype not in SUPPORTED_INPUT_DTYPES for tensor in data_tensors):
        supported = ", ".join(str(dtype) for dtype in SUPPORTED_INPUT_DTYPES)
        raise TypeError(f"decode data inputs must use one of {supported}")

    if lower_bound is not None:
        if not isinstance(lower_bound, Real) or isinstance(lower_bound, bool):
            raise TypeError("lower_bound must be a real scalar or None")
        lower_bound = float(lower_bound)
        if not math.isfinite(lower_bound) or lower_bound >= 0:
            raise ValueError(f"lower_bound must be finite and negative, got {lower_bound}")
    if scale is None:
        scale = key_dim**-0.5
    elif not isinstance(scale, Real) or isinstance(scale, bool):
        raise TypeError("scale must be a real scalar or None")
    else:
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"scale must be finite and positive, got {scale}")

    expected_output_shape = (1, batch, heads, value_dim)
    if out is None:
        out = packed_qkv.new_empty(expected_output_shape)
    else:
        if out.shape != expected_output_shape:
            raise ValueError(
                f"out must have shape {expected_output_shape}, got {tuple(out.shape)}"
            )
        if out.dtype != packed_qkv.dtype:
            raise TypeError(f"out must use packed_qkv.dtype ({packed_qkv.dtype}), got {out.dtype}")
        if out.device != device:
            raise ValueError("out must be on packed_qkv.device")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    return _fused_recurrent_decode_forward(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        out,
        0.0 if lower_bound is None else lower_bound,
        lower_bound is not None,
        scale,
    )


__all__ = ["Impl", "chunk_kda", "recurrent_kda", "recurrent_kda_decode"]
